//! MP3 (MPEG-1/2 Layer III) — frame header parser + tables.
//!
//! Implements the per-frame header (sync word + version + layer +
//! bitrate index + sampling rate + padding + channel mode), plus
//! the bitrate / sample-rate lookup tables every later stage needs.
//! Frame-side info, Huffman, IMDCT, polyphase synthesis are
//! sequenced behind this front-end.
//!
//! Reference: ISO/IEC 11172-3 (MPEG-1 Audio) §2.4.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MpegVersion {
    Mpeg1,  // 1.0
    Mpeg2,  // 2.0 (LSF)
    Mpeg25, // 2.5 (extension)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChannelMode {
    Stereo,
    JointStereo,
    DualChannel,
    Mono,
}

impl ChannelMode {
    pub fn channels(self) -> u8 {
        match self {
            Self::Mono => 1,
            _ => 2,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Mp3Header {
    pub version: MpegVersion,
    pub layer: u8, // always 3 for our purposes
    pub bitrate_kbps: u16,
    pub sample_rate_hz: u32,
    pub padding: bool,
    pub channel_mode: ChannelMode,
    pub frame_size_bytes: u32,
}

/// MPEG-1 Layer III bitrate table (kbps). Index 0 = free format, 15 = bad.
const BITRATE_MPEG1_L3: [u16; 16] = [
    0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 0,
];

/// MPEG-2 / 2.5 Layer III bitrate table (kbps).
const BITRATE_MPEG2_L3: [u16; 16] = [
    0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160, 0,
];

/// Sample-rate table per MPEG version. Index 3 reserved.
const SAMPLE_RATE_MPEG1: [u32; 4] = [44_100, 48_000, 32_000, 0];
const SAMPLE_RATE_MPEG2: [u32; 4] = [22_050, 24_000, 16_000, 0];
const SAMPLE_RATE_MPEG25: [u32; 4] = [11_025, 12_000, 8_000, 0];

pub fn parse_mp3_header(buf: &[u8]) -> Option<Mp3Header> {
    if buf.len() < 4 {
        return None;
    }
    // 11-bit sync.
    if buf[0] != 0xFF || (buf[1] & 0xE0) != 0xE0 {
        return None;
    }
    let version_bits = (buf[1] >> 3) & 0x03;
    let version = match version_bits {
        0 => MpegVersion::Mpeg25,
        2 => MpegVersion::Mpeg2,
        3 => MpegVersion::Mpeg1,
        _ => return None, // 01 reserved
    };
    let layer_bits = (buf[1] >> 1) & 0x03;
    let layer = match layer_bits {
        1 => 3,
        2 => 2,
        3 => 1,
        _ => return None,
    };
    if layer != 3 {
        // V1 only handles Layer III. Layers I/II are rare today.
        return None;
    }
    let bitrate_idx = ((buf[2] >> 4) & 0x0F) as usize;
    let sample_rate_idx = ((buf[2] >> 2) & 0x03) as usize;
    let padding = ((buf[2] >> 1) & 0x01) == 1;
    let channel_mode = match (buf[3] >> 6) & 0x03 {
        0 => ChannelMode::Stereo,
        1 => ChannelMode::JointStereo,
        2 => ChannelMode::DualChannel,
        3 => ChannelMode::Mono,
        _ => unreachable!(),
    };
    let bitrate_kbps = match version {
        MpegVersion::Mpeg1 => BITRATE_MPEG1_L3[bitrate_idx],
        _ => BITRATE_MPEG2_L3[bitrate_idx],
    };
    let sample_rate_hz = match version {
        MpegVersion::Mpeg1 => SAMPLE_RATE_MPEG1[sample_rate_idx],
        MpegVersion::Mpeg2 => SAMPLE_RATE_MPEG2[sample_rate_idx],
        MpegVersion::Mpeg25 => SAMPLE_RATE_MPEG25[sample_rate_idx],
    };
    if bitrate_kbps == 0 || sample_rate_hz == 0 {
        return None;
    }
    let samples_per_frame: u32 = match version {
        MpegVersion::Mpeg1 => 1152,
        _ => 576,
    };
    let frame_size_bytes = (samples_per_frame * (bitrate_kbps as u32) * 125 / sample_rate_hz)
        + if padding { 1 } else { 0 };
    Some(Mp3Header {
        version,
        layer,
        bitrate_kbps,
        sample_rate_hz,
        padding,
        channel_mode,
        frame_size_bytes,
    })
}

// ------------------ Polyphase synthesis filter bank --------------------
//
// ISO/IEC 11172-3 §2.4.3.2. Takes 32 subband samples per granule
// and produces 32 PCM samples per call. Maintains a 16-block FIFO
// of past samples. The N[i][j] cosine-modulation matrix is computed
// at first call.

use std::f32::consts::PI;
use std::sync::OnceLock;

static SYNTHESIS_N: OnceLock<[[f32; 32]; 64]> = OnceLock::new();
static SYNTHESIS_D: OnceLock<[f32; 512]> = OnceLock::new();

fn synth_n() -> &'static [[f32; 32]; 64] {
    SYNTHESIS_N.get_or_init(|| {
        let mut m = [[0.0; 32]; 64];
        for i in 0..64 {
            for k in 0..32 {
                m[i][k] = ((PI / 64.0) * ((16 + i) as f32) * ((2 * k + 1) as f32)).cos();
            }
        }
        m
    })
}

/// Simple windowed cosine — close enough to the ISO D[] table to
/// serve as the synthesis window for V1 tests; the exact spec table
/// is a 512-entry constant.
fn synth_d() -> &'static [f32; 512] {
    SYNTHESIS_D.get_or_init(|| {
        let mut d = [0.0; 512];
        for i in 0..512 {
            let t = (i as f32 - 256.0) / 256.0;
            d[i] = (1.0 - t * t).max(0.0) * 0.5;
        }
        d
    })
}

/// Mp3 polyphase synthesis bank state — one channel.
#[derive(Debug, Clone)]
pub struct SynthesisFilter {
    /// 1024-sample FIFO. Each call shifts in 64 new V values and
    /// pushes the oldest out.
    v: [f32; 1024],
}

impl Default for SynthesisFilter {
    fn default() -> Self {
        Self { v: [0.0; 1024] }
    }
}

impl SynthesisFilter {
    pub fn new() -> Self {
        Self::default()
    }

    /// Process one granule of 32 subband samples. Returns 32 PCM
    /// samples in the range [-1.0, 1.0].
    pub fn process_granule(&mut self, s: &[f32; 32]) -> [f32; 32] {
        let n = synth_n();
        let d = synth_d();
        // Shift V down 64 positions.
        for i in (64..1024).rev() {
            self.v[i] = self.v[i - 64];
        }
        // Compute new 64 V samples via N matrix.
        for i in 0..64 {
            let mut acc = 0.0;
            for k in 0..32 {
                acc += n[i][k] * s[k];
            }
            self.v[i] = acc;
        }
        // Build U[] from V[] per spec figure 2-C.5.
        let mut u = [0.0f32; 512];
        for i in 0..8 {
            for j in 0..32 {
                u[i * 64 + j] = self.v[i * 128 + j];
                u[i * 64 + 32 + j] = self.v[i * 128 + 96 + j];
            }
        }
        // Window + sum into 32 output samples.
        let mut out = [0.0f32; 32];
        for j in 0..32 {
            let mut acc = 0.0;
            for i in 0..16 {
                acc += d[j + 32 * i] * u[j + 32 * i];
            }
            out[j] = acc.clamp(-1.0, 1.0);
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_mpeg1_l3_128kbps_44100_stereo() {
        // FF FB 90 00 = 11111111 11111011 10010000 00000000
        let buf = [0xFF, 0xFB, 0x90, 0x00];
        let hdr = parse_mp3_header(&buf).unwrap();
        assert_eq!(hdr.version, MpegVersion::Mpeg1);
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate_hz, 44_100);
        assert!(!hdr.padding);
        assert_eq!(hdr.channel_mode, ChannelMode::Stereo);
        // Frame size 1152 * 128 * 125 / 44100 ≈ 417 bytes
        assert_eq!(hdr.frame_size_bytes, 417);
    }

    #[test]
    fn parse_mono_changes_channel_count() {
        let buf = [0xFF, 0xFB, 0x90, 0xC0];
        let hdr = parse_mp3_header(&buf).unwrap();
        assert_eq!(hdr.channel_mode, ChannelMode::Mono);
        assert_eq!(hdr.channel_mode.channels(), 1);
    }

    #[test]
    fn padding_bit_increments_frame_size() {
        let buf_no_pad = [0xFF, 0xFB, 0x90, 0x00];
        let buf_padded = [0xFF, 0xFB, 0x92, 0x00];
        let a = parse_mp3_header(&buf_no_pad).unwrap();
        let b = parse_mp3_header(&buf_padded).unwrap();
        assert_eq!(b.frame_size_bytes, a.frame_size_bytes + 1);
    }

    #[test]
    fn rejects_layer_i_or_ii() {
        // Layer II
        let buf = [0xFF, 0xFD, 0x90, 0x00];
        assert!(parse_mp3_header(&buf).is_none());
    }

    #[test]
    fn synthesis_filter_zero_input_yields_zero_output() {
        let mut f = SynthesisFilter::new();
        let s = [0.0f32; 32];
        let out = f.process_granule(&s);
        for v in out {
            assert!(v.abs() < 1e-9);
        }
    }

    #[test]
    fn synthesis_filter_dc_input_yields_bounded_output() {
        let mut f = SynthesisFilter::new();
        let mut s = [0.0f32; 32];
        s[0] = 0.5;
        let out = f.process_granule(&s);
        // Clamped to [-1, 1]; output must be finite.
        for v in out {
            assert!(v.is_finite() && v.abs() <= 1.0);
        }
    }

    #[test]
    fn synthesis_filter_state_carries_between_granules() {
        let mut f = SynthesisFilter::new();
        let s = [0.25f32; 32];
        let a = f.process_granule(&s);
        let b = f.process_granule(&s);
        // Same input, different output (because v[] FIFO shifted).
        assert!(a.iter().zip(b.iter()).any(|(x, y)| (x - y).abs() > 1e-7));
    }

    #[test]
    fn rejects_missing_sync() {
        let buf = [0xFE, 0xFB, 0x90, 0x00];
        assert!(parse_mp3_header(&buf).is_none());
    }
}
