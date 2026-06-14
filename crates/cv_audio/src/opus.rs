//! Opus (RFC 6716) — TOC byte parser + range decoder.
//!
//! Opus packets start with a TOC ("table of contents") byte that
//! identifies configuration, stereo flag, and frame-count code.
//! After the TOC come 1..6 frames, each compressed with either
//! SILK (low-rate speech) or CELT (high-rate music) or a hybrid.
//!
//! This slice implements:
//!   * TOC byte decode → config / stereo / frame-count
//!   * Range decoder bit reader (§4.1) — the substrate every
//!     downstream Opus parser uses

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpusMode {
    Silk,
    Hybrid,
    Celt,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpusBandwidth {
    Narrowband,    // 4 kHz
    Mediumband,    // 6 kHz
    Wideband,      // 8 kHz
    Superwideband, // 12 kHz
    Fullband,      // 20 kHz
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OpusToc {
    pub config: u8, // 0..=31
    pub stereo: bool,
    /// Frame-count code: 0=one, 1=two-equal, 2=two-arbitrary, 3=arbitrary
    pub frame_count_code: u8,
    pub mode: OpusMode,
    pub bandwidth: OpusBandwidth,
    /// Frame duration in samples at 48 kHz.
    pub frame_size_48k: u32,
}

pub fn parse_toc(byte: u8) -> Option<OpusToc> {
    let config = byte >> 3;
    let stereo = (byte & 0x04) != 0;
    let frame_count_code = byte & 0x03;
    let (mode, bandwidth, frame_size_48k) = match config {
        0..=3 => (
            OpusMode::Silk,
            OpusBandwidth::Narrowband,
            samples_silk(config),
        ),
        4..=7 => (
            OpusMode::Silk,
            OpusBandwidth::Mediumband,
            samples_silk(config - 4),
        ),
        8..=11 => (
            OpusMode::Silk,
            OpusBandwidth::Wideband,
            samples_silk(config - 8),
        ),
        12..=13 => (
            OpusMode::Hybrid,
            OpusBandwidth::Superwideband,
            if config == 12 { 480 } else { 960 },
        ),
        14..=15 => (
            OpusMode::Hybrid,
            OpusBandwidth::Fullband,
            if config == 14 { 480 } else { 960 },
        ),
        16..=19 => (
            OpusMode::Celt,
            OpusBandwidth::Narrowband,
            samples_celt(config - 16),
        ),
        20..=23 => (
            OpusMode::Celt,
            OpusBandwidth::Wideband,
            samples_celt(config - 20),
        ),
        24..=27 => (
            OpusMode::Celt,
            OpusBandwidth::Superwideband,
            samples_celt(config - 24),
        ),
        28..=31 => (
            OpusMode::Celt,
            OpusBandwidth::Fullband,
            samples_celt(config - 28),
        ),
        _ => return None,
    };
    Some(OpusToc {
        config,
        stereo,
        frame_count_code,
        mode,
        bandwidth,
        frame_size_48k,
    })
}

const fn samples_silk(idx: u8) -> u32 {
    // 10ms, 20ms, 40ms, 60ms @ 48k
    match idx {
        0 => 480,
        1 => 960,
        2 => 1920,
        _ => 2880,
    }
}

const fn samples_celt(idx: u8) -> u32 {
    // 2.5ms, 5ms, 10ms, 20ms @ 48k
    match idx {
        0 => 120,
        1 => 240,
        2 => 480,
        _ => 960,
    }
}

/// Opus range decoder (RFC 6716 §4.1). Reads a symbol from a
/// pre-built ICDF (inverse cumulative distribution function). The
/// substrate every Opus entropy stage uses.
#[derive(Debug)]
pub struct RangeDecoder<'a> {
    data: &'a [u8],
    pos: usize,
    val: u32,
    rng: u32,
}

impl<'a> RangeDecoder<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        let mut d = Self {
            data,
            pos: 0,
            val: 0,
            rng: 128,
        };
        // Spec §4.1.1 initialization: prime with a 7-bit symbol.
        let first = d.read_byte().unwrap_or(0);
        d.val = (127 - (first as u32 >> 1)) & 0x7F;
        d.rng <<= 7;
        d.normalize();
        d
    }

    fn read_byte(&mut self) -> Option<u8> {
        if self.pos >= self.data.len() {
            return None;
        }
        let b = self.data[self.pos];
        self.pos += 1;
        Some(b)
    }

    fn normalize(&mut self) {
        while self.rng <= (1 << 23) {
            self.val = ((self.val << 8) | (self.read_byte().unwrap_or(0) as u32)) & 0x7FFF_FFFF;
            self.rng <<= 8;
        }
    }

    /// Decode one symbol from an inverse CDF. `icdf` is monotonically
    /// non-increasing from `ft` down to 0. Returns the index of the
    /// matched bucket.
    pub fn dec_icdf(&mut self, icdf: &[u16], ft: u32) -> usize {
        // Promote to u64 for the intermediate products — `val * ft`
        // can hit ~2^31 * 2^15 which overflows u32 multiplication.
        let val = self.val as u64;
        let rng = self.rng as u64;
        let ft = ft as u64;
        let scaled = (val * ft) / rng + 1;
        let mut k = 0;
        while k < icdf.len() && (ft - icdf[k] as u64) < scaled {
            k += 1;
        }
        if k > 0 {
            self.val = (val - rng * (ft - icdf[k - 1] as u64) / ft) as u32;
        }
        let next_icdf = icdf.get(k).copied().unwrap_or(0) as u64;
        self.rng = (rng * next_icdf / ft) as u32;
        if self.rng == 0 {
            self.rng = ft as u32;
        }
        self.normalize();
        k
    }
}

// ------------- CELT MDCT band layout (RFC 6716 §4.3) ---------------
//
// 21 critical bands. Each band's start sample in the 480-sample
// MDCT spectrum varies with frame size. eBands_5ms is the canonical
// 5-ms layout; multiply by frame_size/120 for other durations.

pub const CELT_NUM_BANDS: usize = 21;

/// Band boundaries for a 5-ms / 120-sample CELT frame. Element i
/// is the start of band i; element 21 is the end.
pub const EBANDS_5MS: [u16; 22] = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 34, 40, 48, 60, 78, 100,
];

/// Compute band boundaries for a given MDCT frame size by scaling
/// EBANDS_5MS by frame_size/120.
pub fn band_boundaries(frame_size: usize) -> [u16; 22] {
    let mut out = [0u16; 22];
    for i in 0..22 {
        out[i] = ((EBANDS_5MS[i] as usize * frame_size) / 120) as u16;
    }
    out
}

/// Compute the per-band energy (sum of squared MDCT coefficients) for
/// a frame. Output is one f32 per band — the input to CELT's
/// coarse-energy quantization stage.
pub fn band_energies(mdct: &[f32], boundaries: &[u16; 22]) -> [f32; CELT_NUM_BANDS] {
    let mut out = [0.0f32; CELT_NUM_BANDS];
    for band in 0..CELT_NUM_BANDS {
        let lo = boundaries[band] as usize;
        let hi = boundaries[band + 1] as usize;
        let mut acc = 0.0f32;
        for &v in &mdct[lo..hi.min(mdct.len())] {
            acc += v * v;
        }
        out[band] = acc;
    }
    out
}

/// Normalize each band so its sum-of-squares becomes 1.0. Returns the
/// normalized coefficients; the per-band gain that was divided out is
/// stored separately as `band_energies` for the coarse-energy path.
pub fn normalize_bands(mdct: &mut [f32], boundaries: &[u16; 22]) -> [f32; CELT_NUM_BANDS] {
    let energies = band_energies(mdct, boundaries);
    let mut gains = [0.0f32; CELT_NUM_BANDS];
    for band in 0..CELT_NUM_BANDS {
        let lo = boundaries[band] as usize;
        let hi = boundaries[band + 1] as usize;
        let e = energies[band];
        if e <= 0.0 {
            gains[band] = 0.0;
            continue;
        }
        let g = e.sqrt();
        gains[band] = g;
        let inv = 1.0 / g;
        let len = mdct.len();
        for v in mdct[lo..hi.min(len)].iter_mut() {
            *v *= inv;
        }
    }
    gains
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn toc_silk_nb_10ms() {
        let toc = parse_toc(0b0000_0_000).unwrap();
        assert_eq!(toc.config, 0);
        assert!(!toc.stereo);
        assert_eq!(toc.mode, OpusMode::Silk);
        assert_eq!(toc.bandwidth, OpusBandwidth::Narrowband);
        assert_eq!(toc.frame_size_48k, 480);
    }

    #[test]
    fn toc_celt_fb_20ms_stereo() {
        let toc = parse_toc(0b11111_1_00).unwrap();
        assert_eq!(toc.config, 31);
        assert!(toc.stereo);
        assert_eq!(toc.mode, OpusMode::Celt);
        assert_eq!(toc.bandwidth, OpusBandwidth::Fullband);
        assert_eq!(toc.frame_size_48k, 960);
    }

    #[test]
    fn toc_hybrid_swb() {
        let toc = parse_toc(0b01100_0_00).unwrap();
        assert_eq!(toc.mode, OpusMode::Hybrid);
        assert_eq!(toc.bandwidth, OpusBandwidth::Superwideband);
    }

    #[test]
    fn frame_count_code_extracted() {
        let toc = parse_toc(0b00000_0_11).unwrap();
        assert_eq!(toc.frame_count_code, 3);
    }

    #[test]
    fn range_decoder_initializes_without_panic() {
        let data = [0xAA, 0xBB, 0xCC, 0xDD, 0xEE];
        let _rd = RangeDecoder::new(&data);
    }

    #[test]
    fn ebands_5ms_starts_at_zero_ends_past_100() {
        assert_eq!(EBANDS_5MS[0], 0);
        assert_eq!(EBANDS_5MS[21], 100);
        assert_eq!(EBANDS_5MS.len(), 22);
    }

    #[test]
    fn band_boundaries_scale_with_frame_size() {
        let b240 = band_boundaries(240);
        // Each boundary should double when frame_size doubles.
        for i in 0..22 {
            assert_eq!(b240[i] as usize, EBANDS_5MS[i] as usize * 2);
        }
    }

    #[test]
    fn band_energies_sums_squared_coeffs() {
        let bounds = band_boundaries(120);
        let mdct: Vec<f32> = (0..120).map(|i| if i < 4 { 1.0 } else { 0.0 }).collect();
        let energies = band_energies(&mdct, &bounds);
        // First 4 bands have a single 1.0 coeff each → energy=1.
        assert_eq!(energies[0], 1.0);
        assert_eq!(energies[1], 1.0);
        assert_eq!(energies[2], 1.0);
        assert_eq!(energies[3], 1.0);
        assert_eq!(energies[4], 0.0);
    }

    #[test]
    fn normalize_bands_unit_norm_per_band() {
        let bounds = band_boundaries(120);
        let mut mdct: Vec<f32> = (0..120).map(|_| 1.0).collect();
        let gains = normalize_bands(&mut mdct, &bounds);
        for band in 0..CELT_NUM_BANDS {
            let lo = bounds[band] as usize;
            let hi = bounds[band + 1] as usize;
            let sum_sq: f32 = mdct[lo..hi].iter().map(|x| x * x).sum();
            assert!((sum_sq - 1.0).abs() < 1e-5, "band {band} sum_sq={sum_sq}");
            assert!(gains[band] > 0.0);
        }
    }

    #[test]
    fn icdf_decode_picks_some_bucket() {
        let data = [0x80, 0x00, 0x00, 0x00];
        let mut rd = RangeDecoder::new(&data);
        // 4-bucket uniform ICDF.
        let icdf = [192u16, 128, 64, 0];
        let k = rd.dec_icdf(&icdf, 256);
        assert!(k < 4);
    }
}
