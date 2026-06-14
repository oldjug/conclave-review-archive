//! Baseline-DCT JPEG decoder per ITU-T T.81 / JFIF.
//!
//! Supports:
//!   - Color types: YCbCr (3-component) and grayscale (1-component)
//!   - Subsampling: 4:4:4, 4:2:2, 4:2:0
//!   - Huffman entropy coding (baseline; no progressive, no arithmetic)
//!   - Quantization tables (8-bit)
//!
//! Out of scope for V1: progressive, lossless, arithmetic coding, ICC
//! profile decoding, CMYK, JPEG 2000.

use super::png::{ImageError, RgbaImage};
use std::sync::OnceLock;

const M_SOI: u8 = 0xD8;
const M_EOI: u8 = 0xD9;
const M_SOF0: u8 = 0xC0;
const M_DHT: u8 = 0xC4;
const M_DQT: u8 = 0xDB;
const M_SOS: u8 = 0xDA;
const M_APP0: u8 = 0xE0;
const M_APPF: u8 = 0xEF;
const M_COM: u8 = 0xFE;

pub fn decode_jpeg(input: &[u8]) -> Result<RgbaImage, ImageError> {
    if input.len() < 4 || input[0] != 0xFF || input[1] != M_SOI {
        return Err(ImageError::BadSignature);
    }
    let mut dec = JpegDecoder::default();
    let mut i = 2;
    loop {
        if i + 1 >= input.len() {
            return Err(ImageError::Truncated);
        }
        // Walk past 0xFF fill bytes.
        while i < input.len() && input[i] == 0xFF && input[i + 1] == 0xFF {
            i += 1;
        }
        if input[i] != 0xFF {
            return Err(ImageError::Malformed("expected marker"));
        }
        let marker = input[i + 1];
        i += 2;
        match marker {
            M_EOI => break,
            M_SOS => {
                // Segment + entropy-coded data + EOI.
                let len = (u16::from(input[i]) << 8 | u16::from(input[i + 1])) as usize;
                let segment = &input[i + 2..i + len];
                dec.read_sos(segment)?;
                i += len;
                // Find EOI by scanning entropy data — bytes with 0xFF followed by
                // non-zero non-RST non-EOI are end markers. RST (0xD0-0xD7) and
                // 0xFF00 stuffing are part of the stream.
                let entropy_start = i;
                while i + 1 < input.len() {
                    if input[i] == 0xFF {
                        let nm = input[i + 1];
                        if nm == 0x00 || (0xD0..=0xD7).contains(&nm) {
                            i += 2;
                            continue;
                        }
                        break;
                    }
                    i += 1;
                }
                let entropy = unstuff(&input[entropy_start..i]);
                dec.decode_scan(&entropy)?;
            }
            M_SOF0 => {
                let len = (u16::from(input[i]) << 8 | u16::from(input[i + 1])) as usize;
                dec.read_sof0(&input[i + 2..i + len])?;
                i += len;
            }
            M_DQT => {
                let len = (u16::from(input[i]) << 8 | u16::from(input[i + 1])) as usize;
                dec.read_dqt(&input[i + 2..i + len])?;
                i += len;
            }
            M_DHT => {
                let len = (u16::from(input[i]) << 8 | u16::from(input[i + 1])) as usize;
                dec.read_dht(&input[i + 2..i + len])?;
                i += len;
            }
            // App segments and comments — skip.
            m if (M_APP0..=M_APPF).contains(&m) || m == M_COM => {
                let len = (u16::from(input[i]) << 8 | u16::from(input[i + 1])) as usize;
                i += len;
            }
            // Other unsupported markers with payload — skip by length.
            _ => {
                if i + 1 >= input.len() {
                    return Err(ImageError::Truncated);
                }
                let len = (u16::from(input[i]) << 8 | u16::from(input[i + 1])) as usize;
                i += len;
            }
        }
    }
    dec.finalize()
}

/// Remove JPEG byte stuffing: a literal 0xFF in the entropy stream is
/// encoded as `0xFF 0x00`. RST markers (0xFF D0-D7) get dropped.
fn unstuff(input: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(input.len());
    let mut i = 0;
    while i < input.len() {
        if input[i] == 0xFF && i + 1 < input.len() {
            let nm = input[i + 1];
            if nm == 0x00 {
                out.push(0xFF);
                i += 2;
            } else if (0xD0..=0xD7).contains(&nm) {
                // restart marker — skip (no restart-interval handling)
                i += 2;
            } else {
                i += 1;
            }
        } else {
            out.push(input[i]);
            i += 1;
        }
    }
    out
}

const ZIGZAG: [usize; 64] = [
    0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20,
    13, 6, 7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59,
    52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
];

struct JpegDecoder {
    width: u16,
    height: u16,
    components: Vec<Component>,
    qt: [[u8; 64]; 4],
    huff_dc: [HuffTable; 4],
    huff_ac: [HuffTable; 4],
    /// Per scan: which component → (component_idx, dc_table, ac_table).
    scan_components: Vec<(usize, usize, usize)>,
    /// Raw 8x8 block data, decoded but not yet placed into the framebuffer.
    blocks: Vec<Vec<[i32; 64]>>,
    max_h: usize,
    max_v: usize,
}

impl Default for JpegDecoder {
    fn default() -> Self {
        Self {
            width: 0,
            height: 0,
            components: Vec::new(),
            qt: [[0u8; 64]; 4],
            huff_dc: [
                HuffTable::default(),
                HuffTable::default(),
                HuffTable::default(),
                HuffTable::default(),
            ],
            huff_ac: [
                HuffTable::default(),
                HuffTable::default(),
                HuffTable::default(),
                HuffTable::default(),
            ],
            scan_components: Vec::new(),
            blocks: Vec::new(),
            max_h: 1,
            max_v: 1,
        }
    }
}

#[derive(Default, Clone, Copy, Debug)]
struct Component {
    id: u8,
    h_sample: u8,
    v_sample: u8,
    qt_index: u8,
}

#[derive(Default, Clone)]
struct HuffTable {
    /// `(code_len_bits, code_value) -> symbol`. We store as a generated
    /// lookup for fast bit-by-bit decoding.
    bits_to_symbol: Vec<Vec<u8>>, // bits_to_symbol[len][index]
    counts: [u32; 17],
}

impl HuffTable {
    fn from_jpeg(counts: &[u8; 16], symbols: &[u8]) -> Self {
        let mut t = HuffTable {
            bits_to_symbol: vec![Vec::new(); 17],
            counts: [0; 17],
        };
        let mut k = 0usize;
        for (i, &c) in counts.iter().enumerate() {
            let len = i + 1;
            t.counts[len] = u32::from(c);
            for _ in 0..c {
                if k >= symbols.len() {
                    break;
                }
                t.bits_to_symbol[len].push(symbols[k]);
                k += 1;
            }
        }
        t
    }

    fn decode(&self, br: &mut BitReader<'_>) -> Result<u8, ImageError> {
        let mut code: u32 = 0;
        let mut first: u32 = 0;
        for len in 1..=16 {
            let bit = br.read_bit()?;
            code = (code << 1) | u32::from(bit);
            let count = self.counts[len];
            if code < first + count {
                let local = (code - first) as usize;
                return Ok(self.bits_to_symbol[len][local]);
            }
            first = (first + count) << 1;
        }
        Err(ImageError::Malformed("bad huffman code"))
    }
}

struct BitReader<'a> {
    bytes: &'a [u8],
    pos: usize,
    bit_buf: u32,
    bit_count: u32,
}

impl<'a> BitReader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self {
            bytes,
            pos: 0,
            bit_buf: 0,
            bit_count: 0,
        }
    }
    fn read_bit(&mut self) -> Result<u8, ImageError> {
        if self.bit_count == 0 {
            if self.pos >= self.bytes.len() {
                return Err(ImageError::Truncated);
            }
            self.bit_buf = u32::from(self.bytes[self.pos]);
            self.pos += 1;
            self.bit_count = 8;
        }
        self.bit_count -= 1;
        Ok(((self.bit_buf >> self.bit_count) & 1) as u8)
    }
    fn read_bits(&mut self, n: u32) -> Result<u32, ImageError> {
        let mut v = 0u32;
        for _ in 0..n {
            v = (v << 1) | u32::from(self.read_bit()?);
        }
        Ok(v)
    }
}

fn sign_extend(value: u32, n_bits: u32) -> i32 {
    if n_bits == 0 {
        return 0;
    }
    let vt = 1u32 << (n_bits - 1);
    if value < vt {
        value as i32 - ((1i32 << n_bits) - 1)
    } else {
        value as i32
    }
}

impl JpegDecoder {
    fn read_dqt(&mut self, data: &[u8]) -> Result<(), ImageError> {
        let mut i = 0;
        while i < data.len() {
            let pq_tq = data[i];
            i += 1;
            let pq = pq_tq >> 4;
            let tq = (pq_tq & 0x0F) as usize;
            if pq != 0 {
                return Err(ImageError::Malformed(
                    "16-bit quantization tables unsupported",
                ));
            }
            if tq >= 4 {
                return Err(ImageError::Malformed("bad qt index"));
            }
            for k in 0..64 {
                self.qt[tq][k] = data[i + k];
            }
            i += 64;
        }
        Ok(())
    }

    fn read_dht(&mut self, data: &[u8]) -> Result<(), ImageError> {
        let mut i = 0;
        while i < data.len() {
            let tc_th = data[i];
            i += 1;
            let tc = tc_th >> 4; // 0 = DC, 1 = AC
            let th = (tc_th & 0x0F) as usize;
            if th >= 4 {
                return Err(ImageError::Malformed("bad huffman index"));
            }
            let mut counts = [0u8; 16];
            counts.copy_from_slice(&data[i..i + 16]);
            i += 16;
            let total: usize = counts.iter().map(|&c| c as usize).sum();
            let symbols = &data[i..i + total];
            i += total;
            let table = HuffTable::from_jpeg(&counts, symbols);
            if tc == 0 {
                self.huff_dc[th] = table;
            } else {
                self.huff_ac[th] = table;
            }
        }
        Ok(())
    }

    fn read_sof0(&mut self, data: &[u8]) -> Result<(), ImageError> {
        if data[0] != 8 {
            return Err(ImageError::UnsupportedBitDepth(data[0]));
        }
        self.height = u16::from(data[1]) << 8 | u16::from(data[2]);
        self.width = u16::from(data[3]) << 8 | u16::from(data[4]);
        let n_components = data[5] as usize;
        if !matches!(n_components, 1 | 3) {
            return Err(ImageError::UnsupportedColorType(n_components as u8));
        }
        let mut comps = Vec::with_capacity(n_components);
        for k in 0..n_components {
            let off = 6 + k * 3;
            let id = data[off];
            let hv = data[off + 1];
            let tq = data[off + 2];
            comps.push(Component {
                id,
                h_sample: hv >> 4,
                v_sample: hv & 0x0F,
                qt_index: tq,
            });
        }
        let max_h = comps.iter().map(|c| c.h_sample).max().unwrap_or(1) as usize;
        let max_v = comps.iter().map(|c| c.v_sample).max().unwrap_or(1) as usize;
        self.max_h = max_h;
        self.max_v = max_v;
        self.components = comps;
        Ok(())
    }

    fn read_sos(&mut self, data: &[u8]) -> Result<(), ImageError> {
        let n = data[0] as usize;
        let mut scan = Vec::with_capacity(n);
        for k in 0..n {
            let off = 1 + k * 2;
            let id = data[off];
            let tdta = data[off + 1];
            // Find component index in our list.
            let ci = self
                .components
                .iter()
                .position(|c| c.id == id)
                .ok_or(ImageError::Malformed("unknown component in SOS"))?;
            scan.push((ci, (tdta >> 4) as usize, (tdta & 0x0F) as usize));
        }
        self.scan_components = scan;
        Ok(())
    }

    fn decode_scan(&mut self, entropy: &[u8]) -> Result<(), ImageError> {
        let mut br = BitReader::new(entropy);
        // MCU dimensions in 8-pixel blocks.
        let mcus_x = ((self.width as usize) + 8 * self.max_h - 1) / (8 * self.max_h);
        let mcus_y = ((self.height as usize) + 8 * self.max_v - 1) / (8 * self.max_v);

        // Per-component blocks buffer, sized for full image MCUs.
        let mut comp_blocks: Vec<Vec<[i32; 64]>> =
            self.components.iter().map(|_| Vec::new()).collect();
        let mut last_dc = vec![0i32; self.components.len()];

        for _ in 0..mcus_y {
            for _ in 0..mcus_x {
                for &(ci, td, ta) in &self.scan_components {
                    let comp = self.components[ci];
                    let h = comp.h_sample as usize;
                    let v = comp.v_sample as usize;
                    let qt = &self.qt[comp.qt_index as usize];
                    for _ in 0..(h * v) {
                        let block = self.decode_block(&mut br, td, ta, qt, &mut last_dc[ci])?;
                        comp_blocks[ci].push(block);
                    }
                }
            }
        }
        self.blocks = comp_blocks;
        Ok(())
    }

    fn decode_block(
        &self,
        br: &mut BitReader<'_>,
        td: usize,
        ta: usize,
        qt: &[u8; 64],
        last_dc: &mut i32,
    ) -> Result<[i32; 64], ImageError> {
        let mut zz = [0i32; 64];

        // DC
        let t = self.huff_dc[td].decode(br)?;
        let dc_diff = if t == 0 {
            0
        } else {
            let bits = br.read_bits(u32::from(t))?;
            sign_extend(bits, u32::from(t))
        };
        *last_dc += dc_diff;
        zz[0] = *last_dc * i32::from(qt[0]);

        // AC
        let mut k = 1;
        while k < 64 {
            let rs = self.huff_ac[ta].decode(br)?;
            let r = (rs >> 4) as usize; // run of zeros
            let s = (rs & 0x0F) as u32; // size in bits
            if s == 0 {
                if r != 15 {
                    break; // EOB
                }
                k += 16;
                continue;
            }
            k += r;
            if k >= 64 {
                return Err(ImageError::Malformed("AC overflow"));
            }
            let bits = br.read_bits(s)?;
            let coef = sign_extend(bits, s);
            zz[k] = coef * i32::from(qt[k]);
            k += 1;
        }

        // De-zigzag: rearrange into 8x8 row-major order.
        let mut block = [0i32; 64];
        for (i, &z) in ZIGZAG.iter().enumerate() {
            block[z] = zz[i];
        }
        Ok(block)
    }

    fn finalize(self) -> Result<RgbaImage, ImageError> {
        let w = self.width as usize;
        let h = self.height as usize;
        if self.components.is_empty() {
            return Err(ImageError::NoIhdr);
        }

        // Inverse-DCT each block and write Y/Cb/Cr into per-component planes
        // sized to the image's MCU grid.
        let mcus_x = (w + 8 * self.max_h - 1) / (8 * self.max_h);
        let mcus_y = (h + 8 * self.max_v - 1) / (8 * self.max_v);

        let comp_count = self.components.len();
        let mut planes: Vec<Vec<u8>> = self
            .components
            .iter()
            .map(|c| {
                let cw = mcus_x * c.h_sample as usize * 8;
                let ch = mcus_y * c.v_sample as usize * 8;
                vec![0u8; cw * ch]
            })
            .collect();

        let mut block_idx = vec![0usize; comp_count];
        for my in 0..mcus_y {
            for mx in 0..mcus_x {
                for (ci, comp) in self.components.iter().enumerate() {
                    let h = comp.h_sample as usize;
                    let v = comp.v_sample as usize;
                    let cw = mcus_x * h * 8;
                    for vy in 0..v {
                        for hx in 0..h {
                            let block = &self.blocks[ci][block_idx[ci]];
                            block_idx[ci] += 1;
                            let mut out_block = [0u8; 64];
                            idct_block(block, &mut out_block);
                            // Plot into the plane at MCU position (mx, my)
                            // offset by (hx, vy) sub-blocks.
                            for yy in 0..8 {
                                for xx in 0..8 {
                                    let px = mx * h * 8 + hx * 8 + xx;
                                    let py = my * v * 8 + vy * 8 + yy;
                                    planes[ci][py * cw + px] = out_block[yy * 8 + xx];
                                }
                            }
                        }
                    }
                }
            }
        }

        // Compose RGBA. Upsample chroma planes to luma size.
        let mut pixels = Vec::with_capacity(w * h);
        for y in 0..h {
            for x in 0..w {
                let (r, g, b) = if comp_count == 1 {
                    let cw = mcus_x * self.components[0].h_sample as usize * 8;
                    let v = planes[0][y * cw + x];
                    (v, v, v)
                } else {
                    let yv = sample(&self.components, &planes, mcus_x, 0, x, y);
                    let cb = sample(&self.components, &planes, mcus_x, 1, x, y);
                    let cr = sample(&self.components, &planes, mcus_x, 2, x, y);
                    ycbcr_to_rgb(yv, cb, cr)
                };
                let bgra =
                    (255u32 << 24) | (u32::from(r) << 16) | (u32::from(g) << 8) | u32::from(b);
                pixels.push(bgra);
            }
        }
        Ok(RgbaImage {
            width: self.width as u32,
            height: self.height as u32,
            pixels,
        })
    }
}

fn sample(
    components: &[Component],
    planes: &[Vec<u8>],
    mcus_x: usize,
    ci: usize,
    x: usize,
    y: usize,
) -> u8 {
    let comp = components[ci];
    // Max sampling factors of any component in this scan.
    let max_h = components.iter().map(|c| c.h_sample).max().unwrap_or(1) as usize;
    let max_v = components.iter().map(|c| c.v_sample).max().unwrap_or(1) as usize;
    let sx = (x * comp.h_sample as usize) / max_h;
    let sy = (y * comp.v_sample as usize) / max_v;
    let cw = mcus_x * comp.h_sample as usize * 8;
    let row = sy * cw;
    planes[ci][row + sx]
}

fn ycbcr_to_rgb(y: u8, cb: u8, cr: u8) -> (u8, u8, u8) {
    let yf = y as f32;
    let cbf = cb as f32 - 128.0;
    let crf = cr as f32 - 128.0;
    let r = yf + 1.402 * crf;
    let g = yf - 0.344136 * cbf - 0.714136 * crf;
    let b = yf + 1.772 * cbf;
    (
        r.max(0.0).min(255.0) as u8,
        g.max(0.0).min(255.0) as u8,
        b.max(0.0).min(255.0) as u8,
    )
}

fn cos_table() -> &'static [[f32; 8]; 8] {
    static TABLE: OnceLock<[[f32; 8]; 8]> = OnceLock::new();
    TABLE.get_or_init(|| {
        let mut t = [[0.0_f32; 8]; 8];
        for i in 0..8 {
            for j in 0..8 {
                t[i][j] = ((2.0 * i as f32 + 1.0) * j as f32 * std::f32::consts::PI / 16.0).cos();
            }
        }
        t
    })
}

fn idct_block(input: &[i32; 64], output: &mut [u8; 64]) {
    let cos = cos_table();
    let inv_sqrt2 = 1.0 / std::f32::consts::SQRT_2;
    for x in 0..8 {
        for y in 0..8 {
            let mut sum = 0.0_f32;
            for u in 0..8 {
                for v in 0..8 {
                    let cu = if u == 0 { inv_sqrt2 } else { 1.0 };
                    let cv = if v == 0 { inv_sqrt2 } else { 1.0 };
                    sum += cu * cv * input[u * 8 + v] as f32 * cos[x][u] * cos[y][v];
                }
            }
            let v = (sum * 0.25 + 128.0).round() as i32;
            output[x * 8 + y] = v.clamp(0, 255) as u8;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn signature_check() {
        let bad: &[u8] = &[0, 0, 0, 0];
        assert!(matches!(decode_jpeg(bad), Err(ImageError::BadSignature)));
    }
}
