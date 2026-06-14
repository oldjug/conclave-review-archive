//! EBML (Extensible Binary Meta Language) — Matroska / WebM substrate.
//!
//! Implements the variable-length integer (VINT) decoder and the
//! generic element walker (ID + size + payload). Matroska element
//! IDs are themselves VINTs. This is the layer Cluster / SimpleBlock
//! / Tracks parsing builds on.
//!
//! References:
//!   * RFC 8794 — EBML specification.
//!   * Matroska element registry (https://www.matroska.org/technical/elements.html)

/// Decode an EBML variable-length integer. The first byte's leading
/// zeros indicate the total length (1..=8); the length-marker bit
/// is the highest 1-bit. Returns `(value, length)` on success.
///
/// `mask_marker` controls whether the marker bit is included in the
/// returned value. Element IDs keep it (it's part of the canonical
/// ID); the size field strips it (it's just a length marker).
pub fn read_vint(buf: &[u8], mask_marker: bool) -> Option<(u64, usize)> {
    if buf.is_empty() {
        return None;
    }
    let first = buf[0];
    if first == 0 {
        return None; // reserved
    }
    let length = first.leading_zeros() as usize + 1;
    if length > 8 || buf.len() < length {
        return None;
    }
    let mut v: u64 = if mask_marker {
        first as u64
    } else {
        (first as u64) & ((1u64 << (8 - length)) - 1)
    };
    for &b in &buf[1..length] {
        v = (v << 8) | (b as u64);
    }
    Some((v, length))
}

/// One parsed EBML element header.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Element<'a> {
    pub id: u64,
    pub payload: &'a [u8],
}

/// Walk a flat EBML buffer, producing top-level elements.
pub fn parse_elements(buf: &[u8]) -> Vec<Element<'_>> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < buf.len() {
        let (id, id_len) = match read_vint(&buf[i..], true) {
            Some(x) => x,
            None => break,
        };
        i += id_len;
        let (size, size_len) = match read_vint(&buf[i..], false) {
            Some(x) => x,
            None => break,
        };
        i += size_len;
        if i + size as usize > buf.len() {
            break;
        }
        let payload = &buf[i..i + size as usize];
        out.push(Element { id, payload });
        i += size as usize;
    }
    out
}

/// Matroska / WebM element ID constants (the most useful ones).
pub mod ids {
    pub const EBML_HEADER: u64 = 0x1A45DFA3;
    pub const SEGMENT: u64 = 0x18538067;
    pub const SEEK_HEAD: u64 = 0x114D9B74;
    pub const INFO: u64 = 0x1549A966;
    pub const TRACKS: u64 = 0x1654AE6B;
    pub const CLUSTER: u64 = 0x1F43B675;
    pub const TIMESTAMP: u64 = 0xE7; // per-cluster timestamp
    pub const SIMPLE_BLOCK: u64 = 0xA3;
    pub const BLOCK_GROUP: u64 = 0xA0;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vint_single_byte_root() {
        // 0b1000_0001 = length 1, value 1
        let (v, len) = read_vint(&[0x81], false).unwrap();
        assert_eq!(v, 1);
        assert_eq!(len, 1);
    }

    #[test]
    fn vint_two_byte_value() {
        // 0b0100_0001 0xFF = length 2, value (0x01_FF & 0x3FFF) = 0x01FF
        let (v, len) = read_vint(&[0x41, 0xFF], false).unwrap();
        assert_eq!(v, 0x01FF);
        assert_eq!(len, 2);
    }

    #[test]
    fn vint_ebml_header_id() {
        // 0x1A 0x45 0xDF 0xA3 — the EBML root ID. With mask_marker
        // = true the full ID is preserved.
        let buf = [0x1A, 0x45, 0xDF, 0xA3];
        let (v, len) = read_vint(&buf, true).unwrap();
        assert_eq!(v, 0x1A45DFA3);
        assert_eq!(len, 4);
    }

    #[test]
    fn vint_rejects_zero_first_byte() {
        assert!(read_vint(&[0x00, 0x80], false).is_none());
    }

    #[test]
    fn parse_walks_two_elements() {
        // Element 1: ID 0x82 (1 byte, marker-preserving val 0x82),
        // size 0x82 = 2 bytes payload [0xAA, 0xBB]. Then element 2.
        let buf: Vec<u8> = vec![
            0x82, 0x82, 0xAA, 0xBB, // first element
            0x83, 0x81, 0xCC, // second element id=0x83 size=1 payload=CC
        ];
        let elts = parse_elements(&buf);
        assert_eq!(elts.len(), 2);
        assert_eq!(elts[0].id, 0x82);
        assert_eq!(elts[0].payload, &[0xAA, 0xBB]);
        assert_eq!(elts[1].id, 0x83);
        assert_eq!(elts[1].payload, &[0xCC]);
    }
}
