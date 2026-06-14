//! HTTP/3 (RFC 9114) — frame framing.
//!
//! HTTP/3 runs on QUIC streams. Every frame is `type | length |
//! payload` where both type and length are QUIC varints. This slice
//! implements the framing layer and the well-known frame type
//! constants; QPACK header compression and the connection-level
//! state machine land on top in follow-ups.

use crate::quic::{decode_varint, encode_varint};

/// QPACK static-table prefix encoder. Real QPACK requires both static
/// and dynamic tables with the encoder stream; we emit the simpler
/// "literal field line with name reference" form for known headers
/// and "literal field line" for the rest. The block prefix is two
/// zero bytes (no dynamic-table reference, no required-insert-count).
pub fn qpack_encode_block(headers: &[(&str, &str)]) -> Vec<u8> {
    let mut out = Vec::with_capacity(2 + headers.len() * 16);
    out.push(0); // required insert count = 0
    out.push(0); // delta base = 0, S = 0
    for (n, v) in headers {
        let lc = n.to_ascii_lowercase();
        if let Some(i) = QPACK_STATIC
            .iter()
            .position(|(qn, qv)| *qn == lc && (qv.is_empty() || *qv == *v))
        {
            let exact = !QPACK_STATIC[i].1.is_empty();
            if exact {
                // Indexed field line, static table — 1Txxxxxx, T=1
                qpack_encode_int(&mut out, i as u32, 6, 0xC0);
            } else {
                // Literal field line with name reference — 01NTxxxx
                qpack_encode_int(&mut out, i as u32, 4, 0x50);
                qpack_encode_string(&mut out, v);
            }
        } else {
            // Literal field line with literal name — 001NHxxx
            out.push(0x20);
            qpack_encode_string(&mut out, &lc);
            qpack_encode_string(&mut out, v);
        }
    }
    out
}

fn qpack_encode_int(out: &mut Vec<u8>, value: u32, prefix_bits: u8, first_byte: u8) {
    let max = (1u32 << prefix_bits) - 1;
    if value < max {
        out.push(first_byte | value as u8);
        return;
    }
    out.push(first_byte | max as u8);
    let mut v = value - max;
    while v >= 128 {
        out.push(((v & 0x7F) | 0x80) as u8);
        v >>= 7;
    }
    out.push(v as u8);
}

fn qpack_encode_string(out: &mut Vec<u8>, s: &str) {
    qpack_encode_int(out, s.len() as u32, 7, 0x00);
    out.extend_from_slice(s.as_bytes());
}

/// QPACK static table (RFC 9204 Appendix A). Entries with an empty
/// value mean name-only; the encoder uses them for literal-with-name-
/// reference form.
const QPACK_STATIC: &[(&str, &str)] = &[
    (":authority", ""),
    (":path", "/"),
    ("age", "0"),
    ("content-disposition", ""),
    ("content-length", "0"),
    ("cookie", ""),
    ("date", ""),
    ("etag", ""),
    ("if-modified-since", ""),
    ("if-none-match", ""),
    ("last-modified", ""),
    ("link", ""),
    ("location", ""),
    ("referer", ""),
    ("set-cookie", ""),
    (":method", "CONNECT"),
    (":method", "DELETE"),
    (":method", "GET"),
    (":method", "HEAD"),
    (":method", "OPTIONS"),
    (":method", "POST"),
    (":method", "PUT"),
    (":scheme", "http"),
    (":scheme", "https"),
    (":status", "103"),
    (":status", "200"),
    (":status", "304"),
    (":status", "404"),
    (":status", "503"),
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameType {
    Data = 0x00,
    Headers = 0x01,
    CancelPush = 0x03,
    Settings = 0x04,
    PushPromise = 0x05,
    Goaway = 0x07,
    MaxPushId = 0x0D,
    /// Anything we don't recognize.
    Unknown,
}

impl FrameType {
    pub fn from_raw(v: u64) -> Self {
        match v {
            0x00 => Self::Data,
            0x01 => Self::Headers,
            0x03 => Self::CancelPush,
            0x04 => Self::Settings,
            0x05 => Self::PushPromise,
            0x07 => Self::Goaway,
            0x0D => Self::MaxPushId,
            _ => Self::Unknown,
        }
    }
    pub fn to_raw(self) -> u64 {
        match self {
            Self::Data => 0x00,
            Self::Headers => 0x01,
            Self::CancelPush => 0x03,
            Self::Settings => 0x04,
            Self::PushPromise => 0x05,
            Self::Goaway => 0x07,
            Self::MaxPushId => 0x0D,
            Self::Unknown => 0xFF, // not emittable
        }
    }
}

#[derive(Debug, Clone)]
pub struct Frame<'a> {
    pub kind: FrameType,
    pub payload: &'a [u8],
}

/// Decode the next frame from `buf`. Returns the frame + bytes consumed.
pub fn decode_frame(buf: &[u8]) -> Option<(Frame<'_>, usize)> {
    let (ty, ty_len) = decode_varint(buf)?;
    let (len, len_len) = decode_varint(&buf[ty_len..])?;
    let header_len = ty_len + len_len;
    let payload_len = len as usize;
    if buf.len() < header_len + payload_len {
        return None;
    }
    Some((
        Frame {
            kind: FrameType::from_raw(ty),
            payload: &buf[header_len..header_len + payload_len],
        },
        header_len + payload_len,
    ))
}

pub fn encode_frame(kind: FrameType, payload: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(&encode_varint(kind.to_raw()));
    out.extend_from_slice(&encode_varint(payload.len() as u64));
    out.extend_from_slice(payload);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn data_frame_roundtrip() {
        let payload = b"GET payload";
        let enc = encode_frame(FrameType::Data, payload);
        let (f, consumed) = decode_frame(&enc).unwrap();
        assert_eq!(f.kind, FrameType::Data);
        assert_eq!(f.payload, payload);
        assert_eq!(consumed, enc.len());
    }

    #[test]
    fn headers_frame_roundtrip() {
        let payload = b"\x00:method GET\x00";
        let enc = encode_frame(FrameType::Headers, payload);
        let (f, _) = decode_frame(&enc).unwrap();
        assert_eq!(f.kind, FrameType::Headers);
        assert_eq!(f.payload.len(), payload.len());
    }

    #[test]
    fn unknown_type_decodes_as_unknown() {
        // Manually construct a frame with type 0x21 (single-byte
        // varint, not in the FrameType registry) and an empty body.
        let bad = vec![0x21, 0x00];
        let (f, _) = decode_frame(&bad).unwrap();
        assert_eq!(f.kind, FrameType::Unknown);
    }

    #[test]
    fn truncated_frame_returns_none() {
        let enc = encode_frame(FrameType::Data, b"hello");
        assert!(decode_frame(&enc[..enc.len() - 2]).is_none());
    }
}
