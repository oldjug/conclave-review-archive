//! GIF87a / GIF89a decoder.
//!
//! Per the CompuServe GIF89a specification. We decode the first image
//! descriptor only — animated GIFs surface as their first frame. Output
//! matches the project's BGRA `u32` pixel format used by the rest of
//! `cv_image`.

use crate::png::{ImageError, RgbaImage};

const GIF87A: &[u8; 6] = b"GIF87a";
const GIF89A: &[u8; 6] = b"GIF89a";

pub fn decode_gif(input: &[u8]) -> Result<RgbaImage, ImageError> {
    if input.len() < 6 || (&input[..6] != GIF87A && &input[..6] != GIF89A) {
        return Err(ImageError::BadSignature);
    }
    if input.len() < 13 {
        return Err(ImageError::Truncated);
    }
    let mut i = 6;
    let lsw_w = u16::from_le_bytes([input[i], input[i + 1]]) as u32;
    let lsw_h = u16::from_le_bytes([input[i + 2], input[i + 3]]) as u32;
    let flags = input[i + 4];
    let bg_index = input[i + 5];
    let _aspect = input[i + 6];
    i += 7;

    // Global Color Table.
    let mut gct: Vec<[u8; 3]> = Vec::new();
    if flags & 0x80 != 0 {
        let size = 1usize << ((flags & 0x07) + 1);
        if i + size * 3 > input.len() {
            return Err(ImageError::Truncated);
        }
        gct.reserve(size);
        for k in 0..size {
            gct.push([input[i + k * 3], input[i + k * 3 + 1], input[i + k * 3 + 2]]);
        }
        i += size * 3;
    }

    // Walk blocks until we find the first Image Descriptor. Honour
    // a Graphics Control Extension immediately preceding it for the
    // transparent-index flag.
    let mut transparent_index: Option<u8> = None;
    while i < input.len() {
        let id = input[i];
        i += 1;
        match id {
            0x21 => {
                if i >= input.len() {
                    return Err(ImageError::Truncated);
                }
                let label = input[i];
                i += 1;
                if label == 0xF9 {
                    // Graphics Control Extension: <block_size=4> flags
                    // delay(2) transparent_index <terminator=0>.
                    if i + 6 > input.len() {
                        return Err(ImageError::Truncated);
                    }
                    let bsize = input[i] as usize;
                    if bsize >= 4 {
                        let gflags = input[i + 1];
                        if gflags & 0x01 != 0 {
                            transparent_index = Some(input[i + 4]);
                        }
                    }
                    i += 1 + bsize;
                    // Terminator block (length 0).
                    while i < input.len() && input[i] != 0 {
                        let sz = input[i] as usize;
                        i += 1 + sz;
                    }
                    if i < input.len() {
                        i += 1;
                    }
                } else {
                    // Skip extension data sub-blocks until terminator.
                    while i < input.len() && input[i] != 0 {
                        let sz = input[i] as usize;
                        i += 1 + sz;
                        if i > input.len() {
                            return Err(ImageError::Truncated);
                        }
                    }
                    if i < input.len() {
                        i += 1;
                    }
                }
            }
            0x2C => {
                // Image Descriptor.
                if i + 9 > input.len() {
                    return Err(ImageError::Truncated);
                }
                let _left = u16::from_le_bytes([input[i], input[i + 1]]) as u32;
                let _top = u16::from_le_bytes([input[i + 2], input[i + 3]]) as u32;
                let w = u16::from_le_bytes([input[i + 4], input[i + 5]]) as u32;
                let h = u16::from_le_bytes([input[i + 6], input[i + 7]]) as u32;
                let img_flags = input[i + 8];
                i += 9;
                let interlace = img_flags & 0x40 != 0;
                let mut lct: Vec<[u8; 3]> = Vec::new();
                if img_flags & 0x80 != 0 {
                    let size = 1usize << ((img_flags & 0x07) + 1);
                    if i + size * 3 > input.len() {
                        return Err(ImageError::Truncated);
                    }
                    lct.reserve(size);
                    for k in 0..size {
                        lct.push([input[i + k * 3], input[i + k * 3 + 1], input[i + k * 3 + 2]]);
                    }
                    i += size * 3;
                }
                let palette: &[[u8; 3]] = if !lct.is_empty() { &lct } else { &gct };
                if palette.is_empty() {
                    return Err(ImageError::Malformed("GIF: no palette"));
                }

                // LZW minimum code size + sub-blocks.
                if i >= input.len() {
                    return Err(ImageError::Truncated);
                }
                let min_code_size = input[i];
                i += 1;
                let mut payload: Vec<u8> = Vec::new();
                while i < input.len() {
                    let sz = input[i] as usize;
                    i += 1;
                    if sz == 0 {
                        break;
                    }
                    if i + sz > input.len() {
                        return Err(ImageError::Truncated);
                    }
                    payload.extend_from_slice(&input[i..i + sz]);
                    i += sz;
                }
                let indices = lzw_decompress(&payload, min_code_size, (w * h) as usize)?;

                let pixel_count = (w * h) as usize;
                let mut indices = indices;
                if indices.len() < pixel_count {
                    indices.resize(pixel_count, 0);
                }
                if interlace {
                    indices = deinterlace(&indices, w as usize, h as usize);
                }

                let mut pixels = vec![0u32; pixel_count];
                for (px, &idx) in pixels.iter_mut().zip(indices.iter()) {
                    if let Some(t) = transparent_index {
                        if idx == t {
                            *px = 0; // fully transparent (BGRA = 0x00000000)
                            continue;
                        }
                    }
                    let p = palette.get(idx as usize).copied().unwrap_or([0, 0, 0]);
                    *px = (255u32 << 24)
                        | (u32::from(p[0]) << 16)
                        | (u32::from(p[1]) << 8)
                        | u32::from(p[2]);
                }
                let _ = bg_index;
                let _ = lsw_w;
                let _ = lsw_h;
                return Ok(RgbaImage {
                    width: w,
                    height: h,
                    pixels,
                });
            }
            0x3B => {
                return Err(ImageError::Malformed("GIF: trailer before any image"));
            }
            _ => {
                return Err(ImageError::Malformed("GIF: unknown block"));
            }
        }
    }
    Err(ImageError::Truncated)
}

/// LZW-decompress a GIF sub-block-concatenated payload back into a
/// stream of palette indices. `min_code_size` is the byte from the GIF
/// preceding the sub-blocks. `expected` is used only as a sanity hint
/// to pre-reserve output.
fn lzw_decompress(data: &[u8], min_code_size: u8, expected: usize) -> Result<Vec<u8>, ImageError> {
    if min_code_size < 2 || min_code_size > 11 {
        return Err(ImageError::Malformed("GIF: bad min_code_size"));
    }
    let clear: u16 = 1 << min_code_size;
    let eoi: u16 = clear + 1;
    let mut code_size = (min_code_size + 1) as u32;
    let max_code_size = 12u32;

    // Each code maps to a sequence of bytes. For speed we store sequences
    // as (prefix_code, suffix_byte). We materialize the sequence on
    // demand via a small recursion / iterative reversal.
    let mut prefix: Vec<i32> = Vec::with_capacity(4096);
    let mut suffix: Vec<u8> = Vec::with_capacity(4096);
    let init_codes = clear as usize + 2;
    for k in 0..clear {
        prefix.push(-1);
        suffix.push(k as u8);
    }
    // Clear + EOI placeholders.
    prefix.push(-1);
    suffix.push(0);
    prefix.push(-1);
    suffix.push(0);

    let mut out = Vec::with_capacity(expected.max(64));
    let mut bit_buf: u32 = 0;
    let mut bit_len: u32 = 0;
    let mut data_iter = data.iter();
    let mut prev_code: Option<u16> = None;
    let mut next_code: usize = init_codes;
    let mut mask: u32 = (1u32 << code_size) - 1;
    let mut first_byte_buf: Vec<u8> = Vec::with_capacity(64);

    loop {
        while bit_len < code_size {
            let Some(b) = data_iter.next() else {
                // Stream ended without EOI — treat as success on what we
                // have so far.
                return Ok(out);
            };
            bit_buf |= (*b as u32) << bit_len;
            bit_len += 8;
        }
        let code = (bit_buf & mask) as u16;
        bit_buf >>= code_size;
        bit_len -= code_size;

        if code == clear {
            code_size = (min_code_size + 1) as u32;
            mask = (1u32 << code_size) - 1;
            next_code = init_codes;
            prefix.truncate(init_codes);
            suffix.truncate(init_codes);
            prev_code = None;
            continue;
        }
        if code == eoi {
            return Ok(out);
        }
        // Materialize `code` into bytes. Codes >= next_code are the
        // classic LZW "KwKwK" case: emit prev_code's seq + first byte of
        // prev_code's seq.
        first_byte_buf.clear();
        let mut walker: i32 = if (code as usize) < next_code {
            code as i32
        } else if let Some(p) = prev_code {
            // KwKwK: build prev's sequence first.
            let mut w: i32 = p as i32;
            while w >= 0 {
                first_byte_buf.push(suffix[w as usize]);
                w = prefix[w as usize];
            }
            // first_byte_buf is in reverse; the first byte of the
            // sequence is the last element pushed.
            let first = *first_byte_buf.last().unwrap();
            // Emit (rev of first_byte_buf) then first.
            for b in first_byte_buf.iter().rev() {
                out.push(*b);
            }
            out.push(first);
            // Add new entry (prev, first).
            if next_code < 4096 {
                prefix.push(p as i32);
                suffix.push(first);
                next_code += 1;
                if next_code == (1usize << code_size) && code_size < max_code_size {
                    code_size += 1;
                    mask = if code_size == 32 {
                        u32::MAX
                    } else {
                        (1u32 << code_size) - 1
                    };
                }
            }
            prev_code = Some(code);
            continue;
        } else {
            return Err(ImageError::Malformed("GIF: bad first code"));
        };
        while walker >= 0 {
            first_byte_buf.push(suffix[walker as usize]);
            walker = prefix[walker as usize];
        }
        let first = *first_byte_buf.last().unwrap();
        for b in first_byte_buf.iter().rev() {
            out.push(*b);
        }
        if let Some(p) = prev_code {
            if next_code < 4096 {
                prefix.push(p as i32);
                suffix.push(first);
                next_code += 1;
                if next_code == (1usize << code_size) && code_size < max_code_size {
                    code_size += 1;
                    mask = if code_size == 32 {
                        u32::MAX
                    } else {
                        (1u32 << code_size) - 1
                    };
                }
            }
        }
        prev_code = Some(code);
    }
}

/// GIF interlace passes: rows 0, 8, 16…; 4, 12, 20…; 2, 6, 10…; 1, 3, 5…
fn deinterlace(src: &[u8], w: usize, h: usize) -> Vec<u8> {
    let mut out = vec![0u8; w * h];
    let mut src_row = 0;
    let passes: [(usize, usize); 4] = [(0, 8), (4, 8), (2, 4), (1, 2)];
    for (start, step) in passes {
        let mut y = start;
        while y < h {
            let src_off = src_row * w;
            let dst_off = y * w;
            if src_off + w <= src.len() {
                out[dst_off..dst_off + w].copy_from_slice(&src[src_off..src_off + w]);
            }
            src_row += 1;
            y += step;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Hand-built 2×2 GIF89a, 4-color palette, all 4 pixels distinct.
    /// LZW-encoded by hand: clear=4, codes 0,1,2,3, EOI=5, code_size=3.
    fn build_2x2_gif() -> Vec<u8> {
        let mut g = Vec::new();
        g.extend_from_slice(b"GIF89a");
        // LSD: 2x2, flags=GCT 4-entry (size=2→ (flags&7)+1=2→size=4), bg=0
        g.extend_from_slice(&[
            0x02, 0x00, 0x02, 0x00,
            0xA1, // 0b10100001: GCT yes, color res=2, sort=0, size=1 (2^(1+1)=4)
            0x00, 0x00,
        ]);
        // GCT: black, red, green, blue
        g.extend_from_slice(&[0, 0, 0, 255, 0, 0, 0, 255, 0, 0, 0, 255]);
        // Image Descriptor: left=0,top=0,w=2,h=2,flags=0 (no LCT, no interlace).
        g.extend_from_slice(&[0x2C, 0, 0, 0, 0, 2, 0, 2, 0, 0x00]);
        // LZW min code size = 2; clear=4 (3 bits), then codes 0,1,2,3, EOI=5.
        // With min_code_size=2 the initial code size is 3 bits.
        // Bit stream (LSB first): clear(4), 0, 1, 2, 3, EOI(5). Each = 3 bits.
        // Construct: bits emitted in order, packed LSB-first into bytes.
        let codes = [4u32, 0, 1, 2, 3, 5];
        let mut buf: u32 = 0;
        let mut buf_bits: u32 = 0;
        let mut data = Vec::new();
        // After we emit clear (#0), code size is 3. After 4 dictionary
        // adds (one per non-clear non-EOI code) next_code goes 6,7,8,9.
        // 1<<3 = 8 — we should bump to 4 bits after the 4th code (when
        // next_code became 8). But in the GIF flow, code size grows when
        // we *add* the 2^codesize-th entry, before reading the next code.
        // To keep the test simple, all 6 codes (4 emit + clear + EOI) fit
        // in 3 bits each (max value 7). The grow boundary at next_code=8
        // happens after our 4th non-special code; the EOI that follows
        // would then need 4 bits. Our decoder handles that — but for the
        // *encoder* simplicity we send EOI at 4 bits.
        let mut code_size = 3u32;
        for (i, c) in codes.iter().enumerate() {
            buf |= c << buf_bits;
            buf_bits += code_size;
            while buf_bits >= 8 {
                data.push((buf & 0xFF) as u8);
                buf >>= 8;
                buf_bits -= 8;
            }
            // Mirror decoder's grow rule. After clear (i=0) we reset.
            // After codes #1..#4 (dictionary adds), next_code goes 6,7,8,9.
            // Bump at next_code==8 → i==3.
            if i == 3 {
                code_size = 4;
            }
        }
        if buf_bits > 0 {
            data.push((buf & 0xFF) as u8);
        }
        // LZW minimum code size.
        g.push(2u8);
        // Sub-block.
        g.push(data.len() as u8);
        g.extend_from_slice(&data);
        // Block terminator.
        g.push(0);
        // Trailer.
        g.push(0x3B);
        g
    }

    #[test]
    fn signature_check_rejects_other_data() {
        assert!(matches!(
            decode_gif(b"NOTAGIF12345"),
            Err(ImageError::BadSignature)
        ));
    }

    #[test]
    fn truncated_header_errors() {
        // 6-byte signature present, but the LSD that should follow is
        // missing. We expect Truncated, not BadSignature.
        assert!(matches!(decode_gif(b"GIF89a"), Err(ImageError::Truncated)));
    }

    #[test]
    fn decodes_handbuilt_2x2() {
        let gif = build_2x2_gif();
        let img = decode_gif(&gif).expect("decode 2x2");
        assert_eq!(img.width, 2);
        assert_eq!(img.height, 2);
        assert_eq!(img.pixels.len(), 4);
        // Palette order: black, red, green, blue.
        // Pixel order in input: 0,1,2,3 → black, red, green, blue.
        let bgra = |r: u8, g: u8, b: u8| {
            (255u32 << 24) | (u32::from(r) << 16) | (u32::from(g) << 8) | u32::from(b)
        };
        assert_eq!(img.pixels[0], bgra(0, 0, 0));
        assert_eq!(img.pixels[1], bgra(255, 0, 0));
        assert_eq!(img.pixels[2], bgra(0, 255, 0));
        assert_eq!(img.pixels[3], bgra(0, 0, 255));
    }
}
