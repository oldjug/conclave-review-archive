//! Vorbis (RFC 5215) audio — identification header parse.
//!
//! The Vorbis bitstream starts with three header packets: id,
//! comment, setup. The id packet carries the sample rate + channel
//! count we need to bring up the WASAPI sink. Frame decode (residue
//! / floor / iMDCT) lives behind a feature gate.

#[derive(Debug, Clone, Default)]
pub struct VorbisIdHeader {
    pub version: u32,
    pub channels: u8,
    pub sample_rate: u32,
    pub bitrate_maximum: i32,
    pub bitrate_nominal: i32,
    pub bitrate_minimum: i32,
}

pub fn parse_id_header(buf: &[u8]) -> Option<VorbisIdHeader> {
    if buf.len() < 30 {
        return None;
    }
    if buf[0] != 1 || &buf[1..7] != b"vorbis" {
        return None;
    }
    Some(VorbisIdHeader {
        version: u32::from_le_bytes(buf[7..11].try_into().unwrap()),
        channels: buf[11],
        sample_rate: u32::from_le_bytes(buf[12..16].try_into().unwrap()),
        bitrate_maximum: i32::from_le_bytes(buf[16..20].try_into().unwrap()),
        bitrate_nominal: i32::from_le_bytes(buf[20..24].try_into().unwrap()),
        bitrate_minimum: i32::from_le_bytes(buf[24..28].try_into().unwrap()),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_id_header() {
        let mut buf = vec![0u8; 30];
        buf[0] = 1;
        buf[1..7].copy_from_slice(b"vorbis");
        buf[11] = 2;
        buf[12..16].copy_from_slice(&48000u32.to_le_bytes());
        let h = parse_id_header(&buf).unwrap();
        assert_eq!(h.channels, 2);
        assert_eq!(h.sample_rate, 48000);
    }
}
