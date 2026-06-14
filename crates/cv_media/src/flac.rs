//! FLAC (Free Lossless Audio Codec) frame parser.
//!
//! Surfaces the stream marker + STREAMINFO metadata block so the
//! pipeline knows sample rate, channel count, and bits per sample.
//! The frame decoder (subframe Rice/LPC stages) follows once the
//! audio renderer plumbs it through; for V1 the metadata is what
//! `<audio src=foo.flac>` needs to advertise to the engine.

#[derive(Debug, Clone, Default)]
pub struct FlacStreamInfo {
    pub min_block_size: u16,
    pub max_block_size: u16,
    pub min_frame_size: u32,
    pub max_frame_size: u32,
    pub sample_rate: u32,
    pub channels: u8,
    pub bits_per_sample: u8,
    pub total_samples: u64,
    pub md5: [u8; 16],
}

pub fn parse_streaminfo(buf: &[u8]) -> Option<FlacStreamInfo> {
    if buf.len() < 4 || &buf[0..4] != b"fLaC" {
        return None;
    }
    if buf.len() < 4 + 4 + 34 {
        return None;
    }
    let block_type = buf[4] & 0x7F;
    if block_type != 0 {
        return None;
    }
    let block_len = ((buf[5] as usize) << 16) | ((buf[6] as usize) << 8) | buf[7] as usize;
    if block_len < 34 || buf.len() < 8 + block_len {
        return None;
    }
    let s = &buf[8..8 + 34];
    let min_block_size = u16::from_be_bytes([s[0], s[1]]);
    let max_block_size = u16::from_be_bytes([s[2], s[3]]);
    let min_frame_size = ((s[4] as u32) << 16) | ((s[5] as u32) << 8) | s[6] as u32;
    let max_frame_size = ((s[7] as u32) << 16) | ((s[8] as u32) << 8) | s[9] as u32;
    // 20 bits sample rate
    let sample_rate = ((s[10] as u32) << 12) | ((s[11] as u32) << 4) | ((s[12] as u32) >> 4);
    let channels = (((s[12] >> 1) & 0x07) + 1) as u8;
    let bits_per_sample = ((((s[12] & 0x01) << 4) | (s[13] >> 4)) + 1) as u8;
    // 36 bits total samples
    let total_samples = (((s[13] as u64) & 0x0F) << 32)
        | ((s[14] as u64) << 24)
        | ((s[15] as u64) << 16)
        | ((s[16] as u64) << 8)
        | s[17] as u64;
    let mut md5 = [0u8; 16];
    md5.copy_from_slice(&s[18..34]);
    Some(FlacStreamInfo {
        min_block_size,
        max_block_size,
        min_frame_size,
        max_frame_size,
        sample_rate,
        channels,
        bits_per_sample,
        total_samples,
        md5,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_synthetic_streaminfo() {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"fLaC");
        buf.extend_from_slice(&[0, 0, 0, 34]); // STREAMINFO block, len=34
        // 34-byte payload — choose 48000Hz, 2ch, 16bit.
        let mut s = [0u8; 34];
        s[0..2].copy_from_slice(&4096u16.to_be_bytes());
        s[2..4].copy_from_slice(&4096u16.to_be_bytes());
        // sample rate 48000 = 0x0BB80; channels=2; bits=16
        // bits 0..20 of bytes 10..12.5 hold sample_rate
        s[10] = ((48000u32 >> 12) & 0xFF) as u8;
        s[11] = ((48000u32 >> 4) & 0xFF) as u8;
        s[12] = (((48000u32 << 4) & 0xF0) as u8) | ((2u8 - 1) << 1) | (((16u8 - 1) >> 4) & 1);
        s[13] = (((16u8 - 1) & 0x0F) << 4) | 0;
        buf.extend_from_slice(&s);
        let info = parse_streaminfo(&buf).unwrap();
        assert_eq!(info.sample_rate, 48000);
        assert_eq!(info.channels, 2);
        assert_eq!(info.bits_per_sample, 16);
    }
}
