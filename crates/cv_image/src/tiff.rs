//! TIFF 6.0 (Adobe) baseline reader — Image File Directory walker.
//!
//! Surfaces the canvas geometry and primary IFD entries (compression
//! type, photometric, samples-per-pixel, bits-per-sample) so the
//! pipeline can decide whether to fall back to a different decoder
//! (e.g. JPEG-in-TIFF) or attempt a baseline read.

#[derive(Debug, Clone, Default)]
pub struct TiffHeader {
    pub width: u32,
    pub height: u32,
    pub bits_per_sample: u16,
    pub samples_per_pixel: u16,
    pub compression: u16,
    pub photometric: u16,
}

pub fn parse_header(buf: &[u8]) -> Option<TiffHeader> {
    if buf.len() < 8 {
        return None;
    }
    let (read_u16, read_u32): (fn(&[u8]) -> u16, fn(&[u8]) -> u32) = match &buf[0..2] {
        b"II" => (
            |b| u16::from_le_bytes([b[0], b[1]]),
            |b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]),
        ),
        b"MM" => (
            |b| u16::from_be_bytes([b[0], b[1]]),
            |b| u32::from_be_bytes([b[0], b[1], b[2], b[3]]),
        ),
        _ => return None,
    };
    if read_u16(&buf[2..4]) != 42 {
        return None;
    }
    let ifd_off = read_u32(&buf[4..8]) as usize;
    if ifd_off + 2 > buf.len() {
        return None;
    }
    let n = read_u16(&buf[ifd_off..ifd_off + 2]) as usize;
    let mut h = TiffHeader::default();
    for i in 0..n {
        let off = ifd_off + 2 + i * 12;
        if off + 12 > buf.len() {
            break;
        }
        let tag = read_u16(&buf[off..off + 2]);
        let value_off = off + 8;
        let val_u16 = read_u16(&buf[value_off..value_off + 2]);
        let val_u32 = read_u32(&buf[value_off..value_off + 4]);
        match tag {
            256 => h.width = val_u32,
            257 => h.height = val_u32,
            258 => h.bits_per_sample = val_u16,
            259 => h.compression = val_u16,
            262 => h.photometric = val_u16,
            277 => h.samples_per_pixel = val_u16,
            _ => {}
        }
    }
    Some(h)
}
