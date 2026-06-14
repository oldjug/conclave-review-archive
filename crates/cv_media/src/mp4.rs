//! MP4 / ISO BMFF demuxer.
//!
//! Implements the box (atom) parser used to navigate an MP4 file.
//! Each box has the form:
//!
//!   size  : u32  (big-endian; 1 = use largesize, 0 = to end-of-file)
//!   type  : 4 ASCII bytes
//!   largesize : u64 (only if size==1)
//!   payload : (size − 8 or − 16) bytes
//!
//! V1 surfaces the box tree as a flat enumeration; the next slices
//! parse moov→trak→mdia→minf→stbl substructures to expose the sample
//! tables (`stsz`, `stco`/`co64`, `stsc`, `stts`).
//!
//! References: ISO/IEC 14496-12 (the base spec).

/// One parsed box header + payload slice.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Box4<'a> {
    pub box_type: [u8; 4],
    /// Bytes *inside* the box, after the header. Container boxes
    /// have child boxes packed back-to-back in this slice.
    pub payload: &'a [u8],
}

impl<'a> Box4<'a> {
    pub fn type_as_str(&self) -> &str {
        std::str::from_utf8(&self.box_type).unwrap_or("????")
    }
}

/// Walk a flat byte buffer, returning one box at a time. Stops when
/// the buffer runs out, or returns an error on a malformed header.
pub fn parse_boxes(buf: &[u8]) -> Vec<Box4<'_>> {
    let mut out = Vec::new();
    let mut i = 0;
    while i + 8 <= buf.len() {
        let size = u32::from_be_bytes(buf[i..i + 4].try_into().unwrap()) as u64;
        let mut box_type = [0u8; 4];
        box_type.copy_from_slice(&buf[i + 4..i + 8]);
        let (header_len, total_len) = match size {
            0 => (8usize, buf.len() - i), // to end of buffer
            1 => {
                if i + 16 > buf.len() {
                    break;
                }
                let largesize = u64::from_be_bytes(buf[i + 8..i + 16].try_into().unwrap());
                if largesize < 16 {
                    break;
                }
                (16, largesize as usize)
            }
            _ => (8, size as usize),
        };
        if i + total_len > buf.len() || total_len < header_len {
            break;
        }
        let payload = &buf[i + header_len..i + total_len];
        out.push(Box4 { box_type, payload });
        i += total_len;
    }
    out
}

/// Find the first immediate child box matching `type_tag` within
/// `container_payload`. Returns `None` if not present.
pub fn find_child<'a>(container_payload: &'a [u8], type_tag: &[u8; 4]) -> Option<Box4<'a>> {
    for b in parse_boxes(container_payload) {
        if &b.box_type == type_tag {
            return Some(b);
        }
    }
    None
}

/// `ftyp` header — every conforming MP4 file starts with this.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FtypBox {
    pub major_brand: [u8; 4],
    pub minor_version: u32,
    pub compatible_brands: Vec<[u8; 4]>,
}

pub fn parse_ftyp(b: &Box4<'_>) -> Option<FtypBox> {
    if &b.box_type != b"ftyp" || b.payload.len() < 8 {
        return None;
    }
    let mut major_brand = [0u8; 4];
    major_brand.copy_from_slice(&b.payload[0..4]);
    let minor_version = u32::from_be_bytes(b.payload[4..8].try_into().unwrap());
    let mut compatible_brands = Vec::new();
    let mut i = 8;
    while i + 4 <= b.payload.len() {
        let mut brand = [0u8; 4];
        brand.copy_from_slice(&b.payload[i..i + 4]);
        compatible_brands.push(brand);
        i += 4;
    }
    Some(FtypBox {
        major_brand,
        minor_version,
        compatible_brands,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Encode `(type, payload)` as a normal-size box.
    fn write_box(type_tag: &[u8; 4], payload: &[u8]) -> Vec<u8> {
        let total = 8 + payload.len();
        let mut out = Vec::with_capacity(total);
        out.extend_from_slice(&(total as u32).to_be_bytes());
        out.extend_from_slice(type_tag);
        out.extend_from_slice(payload);
        out
    }

    #[test]
    fn parses_single_box() {
        let buf = write_box(b"free", b"hello world!");
        let boxes = parse_boxes(&buf);
        assert_eq!(boxes.len(), 1);
        assert_eq!(boxes[0].type_as_str(), "free");
        assert_eq!(boxes[0].payload, b"hello world!");
    }

    #[test]
    fn parses_back_to_back_boxes() {
        let mut buf = write_box(b"ftyp", &[0x69, 0x73, 0x6F, 0x6D, 0, 0, 0, 1]);
        buf.extend_from_slice(&write_box(b"moov", b"placeholder"));
        let boxes = parse_boxes(&buf);
        assert_eq!(boxes.len(), 2);
        assert_eq!(boxes[0].type_as_str(), "ftyp");
        assert_eq!(boxes[1].type_as_str(), "moov");
    }

    #[test]
    fn ftyp_decodes_major_brand_and_minor_version() {
        let buf = write_box(
            b"ftyp",
            &[0x6D, 0x70, 0x34, 0x32, 0, 0, 0, 0, 0x69, 0x73, 0x6F, 0x6D],
        );
        let boxes = parse_boxes(&buf);
        let ftyp = parse_ftyp(&boxes[0]).unwrap();
        assert_eq!(&ftyp.major_brand, b"mp42");
        assert_eq!(ftyp.minor_version, 0);
        assert_eq!(ftyp.compatible_brands, vec![*b"isom"]);
    }

    #[test]
    fn find_child_locates_nested_box() {
        let inner = write_box(b"mvhd", b"\x00\x00\x00\x00");
        let outer = write_box(b"moov", &inner);
        let boxes = parse_boxes(&outer);
        assert_eq!(boxes[0].type_as_str(), "moov");
        let mvhd = find_child(boxes[0].payload, b"mvhd").unwrap();
        assert_eq!(mvhd.payload, &[0, 0, 0, 0]);
    }

    #[test]
    fn truncated_header_stops_walking() {
        // 7-byte tail (less than a header) should be ignored.
        let mut buf = write_box(b"free", b"");
        buf.extend_from_slice(&[0u8; 7]);
        let boxes = parse_boxes(&buf);
        assert_eq!(boxes.len(), 1);
    }

    #[test]
    fn size_eq_zero_consumes_rest() {
        // Size 0 means "to end of file" per spec.
        let mut buf = Vec::new();
        buf.extend_from_slice(&0u32.to_be_bytes());
        buf.extend_from_slice(b"mdat");
        buf.extend_from_slice(&[1, 2, 3, 4, 5]);
        let boxes = parse_boxes(&buf);
        assert_eq!(boxes.len(), 1);
        assert_eq!(boxes[0].type_as_str(), "mdat");
        assert_eq!(boxes[0].payload, &[1, 2, 3, 4, 5]);
    }
}
