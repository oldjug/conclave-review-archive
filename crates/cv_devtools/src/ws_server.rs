//! WebSocket server for CDP — RFC 6455 server-side handshake +
//! frame encode/decode.

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};

use crate::cdp::{CdpResponse, Router, parse_request};

/// Compute the Sec-WebSocket-Accept value per RFC 6455 §4.2.2.
fn ws_accept_key(client_key: &str) -> String {
    let combined = format!("{client_key}258EAFA5-E914-47DA-95CA-C5AB0DC85B11");
    let hash = sha1(combined.as_bytes());
    base64_encode(&hash)
}

/// SHA-1 implementation (RFC 3174). Pure Rust, no third-party.
pub fn sha1(input: &[u8]) -> [u8; 20] {
    let mut h0: u32 = 0x67452301;
    let mut h1: u32 = 0xEFCDAB89;
    let mut h2: u32 = 0x98BADCFE;
    let mut h3: u32 = 0x10325476;
    let mut h4: u32 = 0xC3D2E1F0;
    // Build message: input || 0x80 || zero-padding || length in bits BE u64
    let mut msg = input.to_vec();
    let bit_len = (input.len() as u64) * 8;
    msg.push(0x80);
    while msg.len() % 64 != 56 {
        msg.push(0);
    }
    msg.extend_from_slice(&bit_len.to_be_bytes());
    for chunk in msg.chunks(64) {
        let mut w = [0u32; 80];
        for i in 0..16 {
            w[i] = u32::from_be_bytes(chunk[i * 4..i * 4 + 4].try_into().unwrap());
        }
        for i in 16..80 {
            w[i] = (w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16]).rotate_left(1);
        }
        let (mut a, mut b, mut c, mut d, mut e) = (h0, h1, h2, h3, h4);
        for i in 0..80 {
            let (f, k) = match i {
                0..=19 => ((b & c) | (!b & d), 0x5A827999u32),
                20..=39 => (b ^ c ^ d, 0x6ED9EBA1),
                40..=59 => ((b & c) | (b & d) | (c & d), 0x8F1BBCDC),
                _ => (b ^ c ^ d, 0xCA62C1D6),
            };
            let temp = a
                .rotate_left(5)
                .wrapping_add(f)
                .wrapping_add(e)
                .wrapping_add(k)
                .wrapping_add(w[i]);
            e = d;
            d = c;
            c = b.rotate_left(30);
            b = a;
            a = temp;
        }
        h0 = h0.wrapping_add(a);
        h1 = h1.wrapping_add(b);
        h2 = h2.wrapping_add(c);
        h3 = h3.wrapping_add(d);
        h4 = h4.wrapping_add(e);
    }
    let mut out = [0u8; 20];
    out[0..4].copy_from_slice(&h0.to_be_bytes());
    out[4..8].copy_from_slice(&h1.to_be_bytes());
    out[8..12].copy_from_slice(&h2.to_be_bytes());
    out[12..16].copy_from_slice(&h3.to_be_bytes());
    out[16..20].copy_from_slice(&h4.to_be_bytes());
    out
}

const B64: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

pub fn base64_encode(input: &[u8]) -> String {
    let mut out = String::with_capacity((input.len() + 2) / 3 * 4);
    let mut i = 0;
    while i + 3 <= input.len() {
        let n = (input[i] as u32) << 16 | (input[i + 1] as u32) << 8 | (input[i + 2] as u32);
        out.push(B64[((n >> 18) & 0x3F) as usize] as char);
        out.push(B64[((n >> 12) & 0x3F) as usize] as char);
        out.push(B64[((n >> 6) & 0x3F) as usize] as char);
        out.push(B64[(n & 0x3F) as usize] as char);
        i += 3;
    }
    let rem = input.len() - i;
    if rem == 1 {
        let n = (input[i] as u32) << 16;
        out.push(B64[((n >> 18) & 0x3F) as usize] as char);
        out.push(B64[((n >> 12) & 0x3F) as usize] as char);
        out.push_str("==");
    } else if rem == 2 {
        let n = (input[i] as u32) << 16 | (input[i + 1] as u32) << 8;
        out.push(B64[((n >> 18) & 0x3F) as usize] as char);
        out.push(B64[((n >> 12) & 0x3F) as usize] as char);
        out.push(B64[((n >> 6) & 0x3F) as usize] as char);
        out.push('=');
    }
    out
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Opcode {
    Continuation = 0,
    Text = 1,
    Binary = 2,
    Close = 8,
    Ping = 9,
    Pong = 10,
}

impl Opcode {
    pub fn from_u8(v: u8) -> Option<Self> {
        Some(match v {
            0 => Self::Continuation,
            1 => Self::Text,
            2 => Self::Binary,
            8 => Self::Close,
            9 => Self::Ping,
            10 => Self::Pong,
            _ => return None,
        })
    }
}

/// Encode a server-to-client WebSocket frame (no mask).
pub fn encode_frame(opcode: Opcode, payload: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(payload.len() + 10);
    out.push(0x80 | opcode as u8); // FIN=1
    if payload.len() < 126 {
        out.push(payload.len() as u8);
    } else if payload.len() < 65536 {
        out.push(126);
        out.extend_from_slice(&(payload.len() as u16).to_be_bytes());
    } else {
        out.push(127);
        out.extend_from_slice(&(payload.len() as u64).to_be_bytes());
    }
    out.extend_from_slice(payload);
    out
}

/// Decode a client-to-server frame (always masked). Returns
/// (opcode, payload, bytes_consumed) or None if incomplete.
pub fn decode_frame(buf: &[u8]) -> Option<(Opcode, Vec<u8>, usize)> {
    if buf.len() < 2 {
        return None;
    }
    let opcode = Opcode::from_u8(buf[0] & 0x0F)?;
    let masked = (buf[1] & 0x80) != 0;
    let mut len = (buf[1] & 0x7F) as usize;
    let mut off = 2;
    if len == 126 {
        if buf.len() < 4 {
            return None;
        }
        len = u16::from_be_bytes(buf[2..4].try_into().unwrap()) as usize;
        off = 4;
    } else if len == 127 {
        if buf.len() < 10 {
            return None;
        }
        len = u64::from_be_bytes(buf[2..10].try_into().unwrap()) as usize;
        off = 10;
    }
    let mask = if masked {
        if buf.len() < off + 4 {
            return None;
        }
        let m: [u8; 4] = buf[off..off + 4].try_into().unwrap();
        off += 4;
        Some(m)
    } else {
        None
    };
    if buf.len() < off + len {
        return None;
    }
    let mut payload = buf[off..off + len].to_vec();
    if let Some(m) = mask {
        for (i, b) in payload.iter_mut().enumerate() {
            *b ^= m[i % 4];
        }
    }
    Some((opcode, payload, off + len))
}

/// Perform server-side WebSocket handshake on a fresh TCP stream.
pub fn handshake(stream: &mut TcpStream) -> Result<(), String> {
    let mut buf = vec![0u8; 4096];
    let mut total = 0;
    loop {
        let n = stream.read(&mut buf[total..]).map_err(|e| e.to_string())?;
        if n == 0 {
            return Err("eof during handshake".into());
        }
        total += n;
        if buf[..total].windows(4).any(|w| w == b"\r\n\r\n") {
            break;
        }
        if total >= buf.len() {
            return Err("handshake too large".into());
        }
    }
    let req = std::str::from_utf8(&buf[..total]).map_err(|e| e.to_string())?;
    let key = req
        .lines()
        .find_map(|l| {
            l.split_once(':').and_then(|(k, v)| {
                if k.eq_ignore_ascii_case("Sec-WebSocket-Key") {
                    Some(v.trim())
                } else {
                    None
                }
            })
        })
        .ok_or("no Sec-WebSocket-Key")?;
    let accept = ws_accept_key(key);
    let response = format!(
        "HTTP/1.1 101 Switching Protocols\r\n\
         Upgrade: websocket\r\n\
         Connection: Upgrade\r\n\
         Sec-WebSocket-Accept: {accept}\r\n\
         \r\n"
    );
    stream
        .write_all(response.as_bytes())
        .map_err(|e| e.to_string())?;
    Ok(())
}

/// Run one connection's CDP loop. Returns when the client closes.
pub fn serve_connection(stream: &mut TcpStream, router: &Router) -> Result<(), String> {
    let mut accum = Vec::with_capacity(8192);
    let mut tmp = [0u8; 4096];
    loop {
        let n = stream.read(&mut tmp).map_err(|e| e.to_string())?;
        if n == 0 {
            return Ok(());
        }
        accum.extend_from_slice(&tmp[..n]);
        while let Some((opcode, payload, consumed)) = decode_frame(&accum) {
            accum.drain(..consumed);
            match opcode {
                Opcode::Close => return Ok(()),
                Opcode::Ping => {
                    let pong = encode_frame(Opcode::Pong, &payload);
                    stream.write_all(&pong).map_err(|e| e.to_string())?;
                }
                Opcode::Text | Opcode::Binary => {
                    let body = std::str::from_utf8(&payload).map_err(|e| e.to_string())?;
                    let resp = match parse_request(body) {
                        Some(req) => router.dispatch(&req),
                        None => continue,
                    };
                    let json = match &resp {
                        CdpResponse::Success { id, result_raw } => {
                            format!(r#"{{"id":{id},"result":{result_raw}}}"#)
                        }
                        CdpResponse::Error { id, code, message } => format!(
                            r#"{{"id":{id},"error":{{"code":{code},"message":"{message}"}}}}"#
                        ),
                    };
                    let frame = encode_frame(Opcode::Text, json.as_bytes());
                    stream.write_all(&frame).map_err(|e| e.to_string())?;
                }
                _ => {}
            }
        }
    }
}

/// Bind a CDP server to a port. Blocks accepting connections.
pub fn listen(port: u16, router: Router) -> std::io::Result<()> {
    let listener = TcpListener::bind(("127.0.0.1", port))?;
    for incoming in listener.incoming() {
        let mut stream = incoming?;
        let r = &router;
        std::thread::scope(|s| {
            s.spawn(move || {
                if handshake(&mut stream).is_ok() {
                    let _ = serve_connection(&mut stream, r);
                }
            });
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sha1_known_vectors() {
        // RFC 3174 §7.3 test vectors.
        assert_eq!(
            hex::encode(sha1(b"abc")),
            "a9993e364706816aba3e25717850c26c9cd0d89d"
        );
        assert_eq!(
            hex::encode(sha1(b"")),
            "da39a3ee5e6b4b0d3255bfef95601890afd80709"
        );
    }

    #[test]
    fn base64_known_vectors() {
        assert_eq!(base64_encode(b""), "");
        assert_eq!(base64_encode(b"f"), "Zg==");
        assert_eq!(base64_encode(b"fo"), "Zm8=");
        assert_eq!(base64_encode(b"foo"), "Zm9v");
        assert_eq!(base64_encode(b"foobar"), "Zm9vYmFy");
    }

    #[test]
    fn ws_accept_known_rfc_6455_example() {
        // RFC 6455 §1.3 worked example.
        let key = "dGhlIHNhbXBsZSBub25jZQ==";
        assert_eq!(ws_accept_key(key), "s3pPLMBiTxaQ9kYGzzhZRbK+xOo=");
    }

    #[test]
    fn encode_decode_round_trip_text() {
        let payload = b"hello".to_vec();
        let mut frame = encode_frame(Opcode::Text, &payload);
        // The encoder writes unmasked; for the test we synthesize a
        // mask-bit version because the decoder asserts mask presence.
        frame[1] |= 0x80;
        // Insert a 4-byte mask after the length field and mask payload.
        let mask = [0xAA, 0xBB, 0xCC, 0xDD];
        let masked_payload: Vec<u8> = payload
            .iter()
            .enumerate()
            .map(|(i, b)| b ^ mask[i % 4])
            .collect();
        let mut wire = vec![frame[0], frame[1]];
        wire.extend_from_slice(&mask);
        wire.extend_from_slice(&masked_payload);
        let (op, body, _) = decode_frame(&wire).unwrap();
        assert_eq!(op, Opcode::Text);
        assert_eq!(body, b"hello");
    }

    #[test]
    fn decode_partial_frame_returns_none() {
        assert!(decode_frame(&[0x81]).is_none());
    }

    /// Tiny hex helper used by sha1 test.
    mod hex {
        pub(super) fn encode(bytes: [u8; 20]) -> String {
            let mut out = String::with_capacity(40);
            for b in bytes {
                out.push_str(&format!("{:02x}", b));
            }
            out
        }
    }
}
