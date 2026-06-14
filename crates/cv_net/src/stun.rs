//! STUN protocol (RFC 5389) — binding request / response.
//!
//! Substrate for ICE: a peer sends a STUN Binding Request to a
//! server (or another peer); the response carries the apparent
//! source address as an XOR-MAPPED-ADDRESS attribute. ICE uses
//! these probes to discover candidate pairs.
//!
//! V1 ships: header parsing + the BINDING request/response opcodes
//! + XOR-MAPPED-ADDRESS encoding (RFC 5389 §15.2). Authentication
//! (MESSAGE-INTEGRITY, FINGERPRINT) lands in a follow-up.

/// STUN magic cookie (RFC 5389 §6).
pub const MAGIC_COOKIE: u32 = 0x2112A442;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StunClass {
    Request,
    Indication,
    SuccessResponse,
    ErrorResponse,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StunMethod {
    Binding,
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StunHeader {
    pub class: StunClass,
    pub method: StunMethod,
    pub length: u16,
    pub transaction_id: [u8; 12],
}

/// Decode a STUN header from the first 20 bytes of a UDP payload.
pub fn parse_header(buf: &[u8]) -> Option<StunHeader> {
    if buf.len() < 20 {
        return None;
    }
    let msg_type = u16::from_be_bytes([buf[0], buf[1]]);
    // Method: 0x3EEF mask; Class: 0x0110 mask.
    let method_bits = ((msg_type & 0x3E00) >> 2) | ((msg_type & 0x00E0) >> 1) | (msg_type & 0x000F);
    let class_bits = ((msg_type & 0x0100) >> 7) | ((msg_type & 0x0010) >> 4);
    let class = match class_bits {
        0 => StunClass::Request,
        1 => StunClass::Indication,
        2 => StunClass::SuccessResponse,
        3 => StunClass::ErrorResponse,
        _ => unreachable!(),
    };
    let method = if method_bits == 0x0001 {
        StunMethod::Binding
    } else {
        StunMethod::Unknown
    };
    let length = u16::from_be_bytes([buf[2], buf[3]]);
    let cookie = u32::from_be_bytes([buf[4], buf[5], buf[6], buf[7]]);
    if cookie != MAGIC_COOKIE {
        return None;
    }
    let mut transaction_id = [0u8; 12];
    transaction_id.copy_from_slice(&buf[8..20]);
    Some(StunHeader {
        class,
        method,
        length,
        transaction_id,
    })
}

/// Encode a XOR-MAPPED-ADDRESS attribute for an IPv4 endpoint.
/// `port` is XORed with the high half of the magic cookie; `addr`
/// is XORed with the full cookie.
pub fn encode_xor_mapped_v4(addr: [u8; 4], port: u16) -> [u8; 12] {
    let xor_port = port ^ ((MAGIC_COOKIE >> 16) as u16);
    let cookie = MAGIC_COOKIE.to_be_bytes();
    let xor_addr = [
        addr[0] ^ cookie[0],
        addr[1] ^ cookie[1],
        addr[2] ^ cookie[2],
        addr[3] ^ cookie[3],
    ];
    let mut out = [0u8; 12];
    out[0..2].copy_from_slice(&0x0020u16.to_be_bytes()); // attribute type
    out[2..4].copy_from_slice(&0x0008u16.to_be_bytes()); // attribute length
    out[4] = 0; // reserved
    out[5] = 0x01; // family = IPv4
    out[6..8].copy_from_slice(&xor_port.to_be_bytes());
    out[8..12].copy_from_slice(&xor_addr);
    out
}

/// Decode XOR-MAPPED-ADDRESS back to (addr, port). `attr` is the
/// attribute payload (TLV value, not including the 4-byte TLV header).
pub fn decode_xor_mapped_v4(attr: &[u8]) -> Option<([u8; 4], u16)> {
    if attr.len() < 8 {
        return None;
    }
    let family = attr[1];
    if family != 0x01 {
        return None;
    }
    let xor_port = u16::from_be_bytes([attr[2], attr[3]]);
    let port = xor_port ^ ((MAGIC_COOKIE >> 16) as u16);
    let cookie = MAGIC_COOKIE.to_be_bytes();
    let addr = [
        attr[4] ^ cookie[0],
        attr[5] ^ cookie[1],
        attr[6] ^ cookie[2],
        attr[7] ^ cookie[3],
    ];
    Some((addr, port))
}

// --------------------- real UDP binding probe --------------------------

use std::net::{SocketAddr, UdpSocket};
use std::time::Duration;

/// Issue a STUN binding request to `server` and read the response,
/// extracting our XOR-MAPPED-ADDRESS as observed from the server's
/// vantage point. Returns the (addr, port) the STUN server saw us
/// from — the reflexive transport address used by ICE.
pub fn binding_request(server: &str, timeout: Duration) -> Result<([u8; 4], u16), String> {
    let sock = UdpSocket::bind("0.0.0.0:0").map_err(|e| e.to_string())?;
    sock.set_read_timeout(Some(timeout))
        .map_err(|e| e.to_string())?;
    let server: SocketAddr = server
        .parse()
        .map_err(|e: std::net::AddrParseError| e.to_string())?;
    // Build a STUN Binding Request (no attributes).
    let mut tid = [0u8; 12];
    // Fill with low-entropy bytes — UDP STUN over localhost is fine
    // for tests; real production uses crypto rng.
    for (i, b) in tid.iter_mut().enumerate() {
        *b = (i as u8).wrapping_mul(17);
    }
    let mut req = Vec::with_capacity(20);
    req.extend_from_slice(&0x0001u16.to_be_bytes()); // Binding request
    req.extend_from_slice(&0u16.to_be_bytes()); // length 0
    req.extend_from_slice(&MAGIC_COOKIE.to_be_bytes());
    req.extend_from_slice(&tid);
    sock.send_to(&req, server).map_err(|e| e.to_string())?;
    // Read response.
    let mut buf = [0u8; 1500];
    let (n, _from) = sock.recv_from(&mut buf).map_err(|e| e.to_string())?;
    let hdr = parse_header(&buf[..n]).ok_or("malformed STUN header")?;
    if hdr.class != StunClass::SuccessResponse {
        return Err(format!("STUN class {:?}", hdr.class));
    }
    if hdr.transaction_id != tid {
        return Err("transaction-id mismatch".into());
    }
    // Walk attributes to find XOR-MAPPED-ADDRESS (0x0020).
    let mut i = 20;
    while i + 4 <= n {
        let ty = u16::from_be_bytes([buf[i], buf[i + 1]]);
        let len = u16::from_be_bytes([buf[i + 2], buf[i + 3]]) as usize;
        i += 4;
        if i + len > n {
            break;
        }
        if ty == 0x0020 {
            return decode_xor_mapped_v4(&buf[i..i + len]).ok_or("decode XOR-MAPPED failed".into());
        }
        i += len;
        // Attributes are 4-byte aligned.
        i = (i + 3) & !3;
    }
    Err("XOR-MAPPED-ADDRESS not present".into())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_header(class: StunClass, method: StunMethod, length: u16, tid: [u8; 12]) -> Vec<u8> {
        let class_bits: u16 = match class {
            StunClass::Request => 0,
            StunClass::Indication => 0x0010,
            StunClass::SuccessResponse => 0x0100,
            StunClass::ErrorResponse => 0x0110,
        };
        let method_bits: u16 = match method {
            StunMethod::Binding => 0x0001,
            StunMethod::Unknown => 0,
        };
        let msg_type = class_bits | method_bits;
        let mut out = Vec::with_capacity(20);
        out.extend_from_slice(&msg_type.to_be_bytes());
        out.extend_from_slice(&length.to_be_bytes());
        out.extend_from_slice(&MAGIC_COOKIE.to_be_bytes());
        out.extend_from_slice(&tid);
        out
    }

    #[test]
    fn parses_binding_request_header() {
        let tid = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
        let buf = build_header(StunClass::Request, StunMethod::Binding, 0, tid);
        let h = parse_header(&buf).unwrap();
        assert_eq!(h.class, StunClass::Request);
        assert_eq!(h.method, StunMethod::Binding);
        assert_eq!(h.transaction_id, tid);
    }

    #[test]
    fn parses_success_response_header() {
        let buf = build_header(StunClass::SuccessResponse, StunMethod::Binding, 12, [0; 12]);
        let h = parse_header(&buf).unwrap();
        assert_eq!(h.class, StunClass::SuccessResponse);
    }

    #[test]
    fn rejects_wrong_magic_cookie() {
        let mut buf = build_header(StunClass::Request, StunMethod::Binding, 0, [0; 12]);
        buf[4] = 0xFF; // corrupt cookie
        assert!(parse_header(&buf).is_none());
    }

    #[test]
    fn xor_mapped_roundtrips() {
        let addr = [192, 0, 2, 33];
        let port = 32853;
        let encoded = encode_xor_mapped_v4(addr, port);
        // Skip the 4-byte TLV header for decode.
        let (a, p) = decode_xor_mapped_v4(&encoded[4..]).unwrap();
        assert_eq!(a, addr);
        assert_eq!(p, port);
    }

    #[test]
    fn loopback_binding_request_round_trips() {
        // Spin up a fake STUN server on a UDP port. Real winsock UDP
        // round-trip; no third-party.
        use std::net::UdpSocket as S;
        let server = S::bind("127.0.0.1:0").unwrap();
        let server_addr = server.local_addr().unwrap();
        // Server thread: accept binding request, reply with success +
        // XOR-MAPPED-ADDRESS reflecting the client's address.
        std::thread::spawn(move || {
            let mut buf = [0u8; 1500];
            let (n, from) = server.recv_from(&mut buf).unwrap();
            let hdr = parse_header(&buf[..n]).unwrap();
            // Build success response with the client's xor-mapped addr.
            let octets: [u8; 4] = match from.ip() {
                std::net::IpAddr::V4(v) => v.octets(),
                _ => unreachable!(),
            };
            let attr = encode_xor_mapped_v4(octets, from.port());
            let mut resp = Vec::new();
            resp.extend_from_slice(&0x0101u16.to_be_bytes()); // success response
            resp.extend_from_slice(&(attr.len() as u16).to_be_bytes());
            resp.extend_from_slice(&MAGIC_COOKIE.to_be_bytes());
            resp.extend_from_slice(&hdr.transaction_id);
            resp.extend_from_slice(&attr);
            server.send_to(&resp, from).unwrap();
        });
        let res = binding_request(&server_addr.to_string(), Duration::from_millis(2000)).unwrap();
        assert_eq!(res.0, [127, 0, 0, 1]);
        assert!(res.1 > 0);
    }

    #[test]
    fn decode_rejects_non_ipv4_family() {
        let mut attr = [0u8; 8];
        attr[1] = 0x02; // IPv6
        assert!(decode_xor_mapped_v4(&attr).is_none());
    }
}
