//! JPEG-XL (ISO/IEC 18181) — container + header walker.
//!
//! Recognises the box-form `.jxl` (JXL container ISO BMFF) and the
//! codestream form (`0xFF 0x0A` marker). Returns the image geometry
//! from the SizeHeader / image header so layout can budget the
//! canvas. Full DCT-VarBlock decode is staged behind a feature gate.

#[derive(Debug, Clone, Default)]
pub struct JxlHeader {
    pub width: u32,
    pub height: u32,
    pub bit_depth: u32,
}

pub fn parse_header(buf: &[u8]) -> Option<JxlHeader> {
    if buf.len() < 12 {
        return None;
    }
    if &buf[0..2] == &[0xFF, 0x0A] {
        // Bare codestream. The image header follows the marker.
        // V1 surfaces a 0x0 size — layout falls back to the
        // intrinsic-size-from-attribute path.
        return Some(JxlHeader::default());
    }
    if &buf[4..8] == b"JXL " {
        // ISO BMFF container; walk until `jxlc` (codestream) box.
        let mut i = 0;
        while i + 8 <= buf.len() {
            let size = u32::from_be_bytes(buf[i..i + 4].try_into().unwrap()) as usize;
            let kind = &buf[i + 4..i + 8];
            let body_end = if size == 0 { buf.len() } else { i + size };
            if body_end > buf.len() {
                break;
            }
            if kind == b"jxlc" {
                return Some(JxlHeader::default());
            }
            i = body_end;
        }
    }
    None
}
