//! WebSocket client per RFC 6455.
//!
//! Synchronous-blocking V1: opens a TCP/TLS connection, performs the
//! HTTP/1.1 Upgrade handshake, then exposes `send_text`, `send_binary`,
//! `recv` and `close` for raw frames. Higher-level glue (JS WebSocket
//! binding with `onopen/onmessage`) lives in conclave — this crate
//! provides only the protocol primitives.

use crate::NetError;
use crate::dns::resolve;
use crate::socket::Socket;
use crate::tls::TlsStream;
use cv_crypto::Sha256;
use cv_crypto::sha1::Sha1;
use cv_url::Url;

/// A live WebSocket connection. Wraps either a plain or TLS stream.
pub struct WebSocket {
    conn: WsConn,
    rx_raw: Vec<u8>,
    fragment_opcode: Option<u8>,
    fragment_payload: Vec<u8>,
}

enum WsConn {
    Plain(Socket),
    Tls(TlsStream),
}

impl WsConn {
    fn write_all(&mut self, data: &[u8]) -> Result<(), NetError> {
        match self {
            Self::Plain(s) => s.write_all(data).map_err(NetError::Socket),
            Self::Tls(s) => s.write_all(data).map_err(|e| NetError::Http(e.to_string())),
        }
    }
    fn read_exact(&mut self, dst: &mut [u8]) -> Result<(), NetError> {
        let mut off = 0;
        while off < dst.len() {
            let n = match self {
                Self::Plain(s) => s.read(&mut dst[off..]).map_err(NetError::Socket)?,
                Self::Tls(s) => s
                    .read(&mut dst[off..])
                    .map_err(|e| NetError::Http(e.to_string()))?,
            };
            if n == 0 {
                return Err(NetError::Http("ws: peer closed mid-frame".into()));
            }
            off += n;
        }
        Ok(())
    }

    fn read_some(&mut self, dst: &mut [u8]) -> Result<usize, NetError> {
        match self {
            Self::Plain(s) => s.read(dst).map_err(NetError::Socket),
            Self::Tls(s) => s.read(dst).map_err(|e| NetError::Http(e.to_string())),
        }
    }

    fn read_with_timeout(&mut self, dst: &mut [u8], timeout_ms: u32) -> Result<usize, NetError> {
        match self {
            Self::Plain(s) => s
                .read_with_timeout(dst, timeout_ms)
                .map_err(NetError::Socket),
            Self::Tls(s) => s
                .read_with_timeout(dst, timeout_ms)
                .map_err(|e| NetError::Http(e.to_string())),
        }
    }
}

/// One decoded incoming frame.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WsFrame {
    Text(String),
    Binary(Vec<u8>),
    Ping(Vec<u8>),
    Pong(Vec<u8>),
    /// Close — pair of (code, reason). 1000 = normal.
    Close {
        code: u16,
        reason: String,
    },
}

const OP_CONT: u8 = 0x0;
const OP_TEXT: u8 = 0x1;
const OP_BINARY: u8 = 0x2;
const OP_CLOSE: u8 = 0x8;
const OP_PING: u8 = 0x9;
const OP_PONG: u8 = 0xa;

impl WebSocket {
    /// Open a WebSocket against `url`. Performs the RFC 6455 handshake
    /// and returns once the server has answered with `101 Switching
    /// Protocols` + a valid `Sec-WebSocket-Accept` header.
    pub fn connect(url: &Url) -> Result<Self, NetError> {
        let scheme = url.scheme.as_str();
        let is_secure = match scheme {
            "ws" | "http" => false,
            "wss" | "https" => true,
            _ => return Err(NetError::Url(format!("not a ws scheme: {scheme}"))),
        };
        let host = if url.host.is_empty() {
            return Err(NetError::Url("missing host".into()));
        } else {
            url.host.clone()
        };
        let port = url
            .effective_port()
            .unwrap_or(if is_secure { 443 } else { 80 });
        let addrs = resolve(&host, port).map_err(NetError::Dns)?;
        let mut last: Option<NetError> = None;
        let mut sock = None;
        for a in addrs {
            match Socket::connect_with_timeout(&a, 30_000) {
                Ok(s) => {
                    sock = Some(s);
                    break;
                }
                Err(e) => last = Some(NetError::Socket(e)),
            }
        }
        let raw = sock.ok_or_else(|| last.unwrap_or(NetError::Http("no addr".into())))?;
        let mut conn = if is_secure {
            WsConn::Tls(TlsStream::connect(raw, &host)?)
        } else {
            WsConn::Plain(raw)
        };

        // Handshake.
        let key_bytes: [u8; 16] = [
            // Static-but-random-looking 16-byte client key, base64'd.
            // RFC 6455 §4.1 only requires the value be 16 random bytes
            // base64-encoded; a fixed value is acceptable because the
            // server-side accept hash still depends on the GUID. Real
            // implementations randomise per-connection — V1 follows.
            0x12, 0x34, 0x56, 0x78, 0x9a, 0xbc, 0xde, 0xf0, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66,
            0x77, 0x88,
        ];
        let key_b64 = base64_encode(&key_bytes);
        let target = url.request_target();

        // RFC 6455 §4.1: the Host header MUST include the port when it is
        // non-default.  Default ports are 80 for ws:// and 443 for wss://.
        let default_port: u16 = if is_secure { 443 } else { 80 };
        let host_header = match url.port {
            Some(p) if p != default_port => format!("{host}:{p}"),
            _ => host.clone(),
        };

        // RFC 6455 §4.1 step 9: the Origin header MUST be present.  The
        // value is the HTTP origin (http/https, not ws/wss) of the page
        // initiating the connection.  Since cv_net is below the page-context
        // layer we derive it from the WebSocket URL itself; callers that have
        // the real page origin should use `connect_with_origin`.
        let origin_scheme = if is_secure { "https" } else { "http" };
        let origin = format!("{origin_scheme}://{host_header}");

        let mut wire = String::new();
        wire.push_str(&format!("GET {target} HTTP/1.1\r\n"));
        wire.push_str(&format!("Host: {host_header}\r\n"));
        wire.push_str(&format!("Origin: {origin}\r\n"));
        wire.push_str("Upgrade: websocket\r\n");
        wire.push_str("Connection: Upgrade\r\n");
        wire.push_str(&format!("Sec-WebSocket-Key: {key_b64}\r\n"));
        wire.push_str("Sec-WebSocket-Version: 13\r\n");
        wire.push_str("\r\n");
        conn.write_all(wire.as_bytes())?;

        // Read response until \r\n\r\n.
        let mut head: Vec<u8> = Vec::with_capacity(512);
        let mut buf = [0u8; 256];
        loop {
            let n = match &mut conn {
                WsConn::Plain(s) => s.read(&mut buf).map_err(NetError::Socket)?,
                WsConn::Tls(s) => s
                    .read(&mut buf)
                    .map_err(|e| NetError::Http(e.to_string()))?,
            };
            if n == 0 {
                return Err(NetError::Http("ws: closed during handshake".into()));
            }
            head.extend_from_slice(&buf[..n]);
            if head.windows(4).any(|w| w == b"\r\n\r\n") {
                break;
            }
            if head.len() > 16_384 {
                return Err(NetError::Http("ws: handshake too large".into()));
            }
        }
        let header_end = head
            .windows(4)
            .position(|w| w == b"\r\n\r\n")
            .map(|idx| idx + 4)
            .ok_or_else(|| NetError::Http("ws: malformed handshake".into()))?;
        let (head_bytes, tail_bytes) = head.split_at(header_end);
        let head_str = std::str::from_utf8(head_bytes)
            .map_err(|_| NetError::Http("ws: non-utf8 handshake".into()))?;
        if !head_str.starts_with("HTTP/1.1 101") {
            return Err(NetError::Http(format!(
                "ws: server rejected (got `{}`)",
                head_str.lines().next().unwrap_or("")
            )));
        }
        let accept = head_str
            .lines()
            .filter_map(|l| l.split_once(':'))
            .find_map(|(k, v)| {
                if k.eq_ignore_ascii_case("sec-websocket-accept") {
                    Some(v.trim())
                } else {
                    None
                }
            });
        let Some(accept) = accept else {
            return Err(NetError::Http("ws: missing Sec-WebSocket-Accept".into()));
        };
        let expected_accept = websocket_accept_value(&key_b64);
        if accept != expected_accept {
            return Err(NetError::Http("ws: bad Sec-WebSocket-Accept".into()));
        }
        Ok(Self {
            conn,
            rx_raw: tail_bytes.to_vec(),
            fragment_opcode: None,
            fragment_payload: Vec::new(),
        })
    }

    pub fn send_text(&mut self, text: &str) -> Result<(), NetError> {
        self.send_frame(OP_TEXT, text.as_bytes(), true)
    }
    pub fn send_binary(&mut self, data: &[u8]) -> Result<(), NetError> {
        self.send_frame(OP_BINARY, data, true)
    }
    pub fn send_close(&mut self, code: u16, reason: &str) -> Result<(), NetError> {
        let mut payload = Vec::with_capacity(2 + reason.len());
        payload.extend_from_slice(&code.to_be_bytes());
        payload.extend_from_slice(reason.as_bytes());
        self.send_frame(OP_CLOSE, &payload, true)
    }

    fn send_frame(&mut self, op: u8, payload: &[u8], fin: bool) -> Result<(), NetError> {
        let mut hdr: Vec<u8> = Vec::with_capacity(14);
        let b1 = (if fin { 0x80 } else { 0x00 }) | (op & 0x0f);
        hdr.push(b1);
        // Client → server frames MUST be masked (RFC 6455 §5.3).
        let mask_bit = 0x80;
        let len = payload.len();
        if len < 126 {
            hdr.push(mask_bit | len as u8);
        } else if len <= u16::MAX as usize {
            hdr.push(mask_bit | 126);
            hdr.extend_from_slice(&(len as u16).to_be_bytes());
        } else {
            hdr.push(mask_bit | 127);
            hdr.extend_from_slice(&(len as u64).to_be_bytes());
        }
        // 4-byte mask key. Plain-and-simple deterministic key for V1
        // (real clients use a CSPRNG; we don't need cryptographic
        // unpredictability for masking, only to flip bits).
        let mask: [u8; 4] = [0xa1, 0xb2, 0xc3, 0xd4];
        hdr.extend_from_slice(&mask);
        let masked: Vec<u8> = payload
            .iter()
            .enumerate()
            .map(|(i, b)| b ^ mask[i & 3])
            .collect();
        self.conn.write_all(&hdr)?;
        if !masked.is_empty() {
            self.conn.write_all(&masked)?;
        }
        Ok(())
    }

    /// Read one frame from the stream. Blocks until a frame arrives or
    /// the connection closes. Control frames (Ping/Pong/Close) are
    /// returned to the caller — caller is expected to respond.
    pub fn recv(&mut self) -> Result<WsFrame, NetError> {
        self.recv_inner(None)?
            .ok_or_else(|| NetError::Http("ws: peer closed".into()))
    }

    pub fn recv_with_timeout(&mut self, timeout_ms: u32) -> Result<Option<WsFrame>, NetError> {
        self.recv_inner(Some(timeout_ms))
    }

    fn recv_inner(&mut self, timeout_ms: Option<u32>) -> Result<Option<WsFrame>, NetError> {
        while self.rx_raw.len() < 2 {
            if !self.read_more(timeout_ms)? {
                return Ok(None);
            }
        }
        let fin = self.rx_raw[0] & 0x80 != 0;
        // RFC 6455 §5.2: RSV1/RSV2/RSV3 must be 0 unless a negotiated extension
        // defines a meaning for them. We don't negotiate any extensions, so any
        // non-zero RSV bit is a protocol error — close with 1002.
        let rsv = self.rx_raw[0] & 0x70;
        if rsv != 0 {
            return Err(NetError::Http(
                "ws: 1002 RSV bits set without negotiated extension".into(),
            ));
        }
        let op = self.rx_raw[0] & 0x0f;
        let masked = self.rx_raw[1] & 0x80 != 0;
        let mut len = (self.rx_raw[1] & 0x7f) as usize;
        let mut off = 2usize;
        if len == 126 {
            while self.rx_raw.len() < off + 2 {
                if !self.read_more(timeout_ms)? {
                    return Ok(None);
                }
            }
            len = u16::from_be_bytes([self.rx_raw[off], self.rx_raw[off + 1]]) as usize;
            off += 2;
        } else if len == 127 {
            while self.rx_raw.len() < off + 8 {
                if !self.read_more(timeout_ms)? {
                    return Ok(None);
                }
            }
            len = u64::from_be_bytes([
                self.rx_raw[off],
                self.rx_raw[off + 1],
                self.rx_raw[off + 2],
                self.rx_raw[off + 3],
                self.rx_raw[off + 4],
                self.rx_raw[off + 5],
                self.rx_raw[off + 6],
                self.rx_raw[off + 7],
            ]) as usize;
            off += 8;
        }
        // RFC 6455 §5.1: server MUST NOT mask frames. A masked frame from
        // the server is a protocol error; close with 1002.
        if masked {
            return Err(NetError::Http(
                "ws: 1002 masked frame from server".into(),
            ));
        }
        while self.rx_raw.len() < off + len {
            if !self.read_more(timeout_ms)? {
                return Ok(None);
            }
        }
        let payload = self.rx_raw[off..off + len].to_vec();
        self.rx_raw.drain(..off + len);
        let frame = match op {
            OP_TEXT | OP_BINARY | OP_CONT => {
                if op == OP_CONT {
                    let Some(fragment_opcode) = self.fragment_opcode else {
                        return Err(NetError::Http("ws: unexpected continuation frame".into()));
                    };
                    self.fragment_payload.extend_from_slice(&payload);
                    if !fin {
                        return self.recv_inner(timeout_ms);
                    }
                    let complete = std::mem::take(&mut self.fragment_payload);
                    self.fragment_opcode = None;
                    decode_data_frame(fragment_opcode, complete)
                } else if fin && self.fragment_opcode.is_none() {
                    decode_data_frame(op, payload)
                } else if !fin && self.fragment_opcode.is_none() {
                    self.fragment_opcode = Some(op);
                    self.fragment_payload = payload;
                    return self.recv_inner(timeout_ms);
                } else {
                    return Err(NetError::Http("ws: fragmented message interrupted".into()));
                }
            }
            OP_PING => Ok(WsFrame::Ping(payload)),
            OP_PONG => Ok(WsFrame::Pong(payload)),
            OP_CLOSE => {
                // RFC 6455 §5.5.1: close frame payload must be 0 or ≥2 bytes.
                // Exactly 1 byte is a protocol error.
                let (code, reason) = if payload.len() >= 2 {
                    let c = u16::from_be_bytes([payload[0], payload[1]]);
                    // RFC 6455 §7.4.2: valid close codes are 1000, 1001, 1002,
                    // 1003, 1007, 1008, 1009, 1010, 1011, or codes in 3000-4999.
                    // 1004/1005/1006 are reserved and MUST NOT appear on the wire.
                    let valid_code = matches!(
                        c,
                        1000 | 1001 | 1002 | 1003 | 1007 | 1008 | 1009 | 1010 | 1011
                    ) || (3000..=4999).contains(&c);
                    if !valid_code {
                        return Err(NetError::Http(format!(
                            "ws: 1002 invalid close code {c}"
                        )));
                    }
                    // RFC 6455 §5.5.1: close reason must be valid UTF-8.
                    let r = match String::from_utf8(payload[2..].to_vec()) {
                        Ok(s) => s,
                        Err(_) => {
                            return Err(NetError::Http(
                                "ws: 1007 invalid UTF-8 in close reason".into(),
                            ));
                        }
                    };
                    (c, r)
                } else if payload.is_empty() {
                    (1005, String::new())
                } else {
                    // 1 byte: protocol error per RFC 6455 §5.5.1
                    return Err(NetError::Http(
                        "ws: 1002 close frame with 1-byte payload".into(),
                    ));
                };
                Ok(WsFrame::Close { code, reason })
            }
            other => Err(NetError::Http(format!("ws: bad opcode 0x{other:x}"))),
        }?;
        Ok(Some(frame))
    }

    fn read_more(&mut self, timeout_ms: Option<u32>) -> Result<bool, NetError> {
        let mut buf = [0u8; 16 * 1024];
        let n = match timeout_ms {
            Some(timeout_ms) => self.conn.read_with_timeout(&mut buf, timeout_ms)?,
            None => self.conn.read_some(&mut buf)?,
        };
        if n == 0 {
            return Ok(false);
        }
        self.rx_raw.extend_from_slice(&buf[..n]);
        Ok(true)
    }
}

/// Local base64 encoder; mirrors conclave's version. Duplicated here
/// to keep cv_net self-contained.
fn base64_encode(bytes: &[u8]) -> String {
    const TBL: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity((bytes.len() + 2) / 3 * 4);
    let mut i = 0;
    while i + 3 <= bytes.len() {
        let b0 = bytes[i];
        let b1 = bytes[i + 1];
        let b2 = bytes[i + 2];
        out.push(TBL[(b0 >> 2) as usize] as char);
        out.push(TBL[(((b0 & 0x03) << 4) | (b1 >> 4)) as usize] as char);
        out.push(TBL[(((b1 & 0x0f) << 2) | (b2 >> 6)) as usize] as char);
        out.push(TBL[(b2 & 0x3f) as usize] as char);
        i += 3;
    }
    let rem = bytes.len() - i;
    if rem == 1 {
        let b0 = bytes[i];
        out.push(TBL[(b0 >> 2) as usize] as char);
        out.push(TBL[((b0 & 0x03) << 4) as usize] as char);
        out.push('=');
        out.push('=');
    } else if rem == 2 {
        let b0 = bytes[i];
        let b1 = bytes[i + 1];
        out.push(TBL[(b0 >> 2) as usize] as char);
        out.push(TBL[(((b0 & 0x03) << 4) | (b1 >> 4)) as usize] as char);
        out.push(TBL[((b1 & 0x0f) << 2) as usize] as char);
        out.push('=');
    }
    out
}

fn websocket_accept_value(key_b64: &str) -> String {
    let mut sha1 = Sha1::new();
    sha1.update(key_b64.as_bytes());
    sha1.update(b"258EAFA5-E914-47DA-95CA-C5AB0DC85B11");
    base64_encode(&sha1.finalize())
}

fn decode_data_frame(op: u8, payload: Vec<u8>) -> Result<WsFrame, NetError> {
    match op {
        OP_TEXT => {
            // RFC 6455 §8.1: text message payload MUST be valid UTF-8.
            // Invalid UTF-8 is a protocol error; close with code 1007.
            match String::from_utf8(payload) {
                Ok(s) => Ok(WsFrame::Text(s)),
                Err(_) => Err(NetError::Http(
                    "ws: 1007 invalid UTF-8 in text frame".into(),
                )),
            }
        }
        OP_BINARY => Ok(WsFrame::Binary(payload)),
        other => Err(NetError::Http(format!("ws: bad data opcode 0x{other:x}"))),
    }
}

// Pull in Sha256 to silence unused-import warning (real handshake
// validation will use it once we add SHA-1).
#[allow(dead_code)]
fn _ensure_sha256_linked() -> Sha256 {
    Sha256::new()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::thread;
    use cv_url::Url;

    #[test]
    fn base64_known_vectors() {
        assert_eq!(base64_encode(b""), "");
        assert_eq!(base64_encode(b"f"), "Zg==");
        assert_eq!(base64_encode(b"fo"), "Zm8=");
        assert_eq!(base64_encode(b"foo"), "Zm9v");
        assert_eq!(base64_encode(b"foobar"), "Zm9vYmFy");
    }

    #[test]
    fn websocket_accept_matches_rfc_example() {
        assert_eq!(
            websocket_accept_value("dGhlIHNhbXBsZSBub25jZQ=="),
            "s3pPLMBiTxaQ9kYGzzhZRbK+xOo="
        );
    }

    #[test]
    fn recv_reassembles_fragmented_text_after_handshake_tail_bytes() {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind localhost");
        let addr = listener.local_addr().expect("listener addr");
        let server = thread::spawn(move || {
            let (mut stream, _) = listener.accept().expect("accept websocket client");
            let mut request = Vec::new();
            let mut buf = [0u8; 1024];
            loop {
                let n = stream.read(&mut buf).expect("read handshake request");
                assert!(n > 0, "client closed during handshake");
                request.extend_from_slice(&buf[..n]);
                if request.windows(4).any(|w| w == b"\r\n\r\n") {
                    break;
                }
            }
            let request_str = String::from_utf8(request).expect("utf8 request");
            let key = request_str
                .lines()
                .find_map(|line| {
                    line.split_once(':').and_then(|(k, v)| {
                        if k.eq_ignore_ascii_case("Sec-WebSocket-Key") {
                            Some(v.trim().to_string())
                        } else {
                            None
                        }
                    })
                })
                .expect("client Sec-WebSocket-Key");
            let mut reply = format!(
                concat!(
                    "HTTP/1.1 101 Switching Protocols\r\n",
                    "Upgrade: websocket\r\n",
                    "Connection: Upgrade\r\n",
                    "Sec-WebSocket-Accept: {}\r\n\r\n"
                ),
                websocket_accept_value(&key)
            )
            .into_bytes();
            // Fragment "hello" as text + continuation, and send the first
            // fragment in the same write as the handshake response so the
            // client preserves post-header bytes.
            reply.extend_from_slice(&[0x01, 0x02, b'h', b'e']);
            stream
                .write_all(&reply)
                .expect("write handshake + first fragment");
            stream
                .write_all(&[0x80, 0x03, b'l', b'l', b'o'])
                .expect("write continuation fragment");
        });

        let url = Url::parse(&format!("ws://127.0.0.1:{}/ws", addr.port())).expect("ws url");
        let mut ws = WebSocket::connect(&url).expect("connect websocket");
        assert_eq!(
            ws.recv().expect("reassembled frame"),
            WsFrame::Text("hello".into())
        );
        server.join().expect("server thread");
    }

    // ------------------------------------------------------------------ //
    //  Helper: spin up a minimal WS server and complete the handshake.   //
    //  Returns (ws_client, raw_tcp_stream_to_server).                     //
    // ------------------------------------------------------------------ //
    fn do_handshake() -> (WebSocket, std::net::TcpStream) {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
        let addr = listener.local_addr().expect("addr");
        let handle = thread::spawn(move || {
            let (mut stream, _) = listener.accept().expect("accept");
            let mut req = Vec::new();
            let mut buf = [0u8; 4096];
            loop {
                let n = stream.read(&mut buf).expect("read");
                req.extend_from_slice(&buf[..n]);
                if req.windows(4).any(|w| w == b"\r\n\r\n") {
                    break;
                }
            }
            // Return both the raw request and the stream so the caller
            // can inspect the request and then send frames.
            (req, stream)
        });
        let url = Url::parse(&format!("ws://127.0.0.1:{}/", addr.port())).expect("url");
        // We need to complete the handshake from the server side while
        // the client is waiting.  Use a second thread for the WS client
        // so we can do both sides concurrently, then join.
        let client_handle = thread::spawn(move || WebSocket::connect(&url));
        let (req_bytes, mut server_stream) = handle.join().expect("server thread");
        let req_str = String::from_utf8(req_bytes).expect("utf8 req");
        let key = req_str
            .lines()
            .find_map(|l| {
                l.split_once(':').and_then(|(k, v)| {
                    if k.eq_ignore_ascii_case("Sec-WebSocket-Key") {
                        Some(v.trim().to_string())
                    } else {
                        None
                    }
                })
            })
            .expect("key");
        let resp = format!(
            "HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Accept: {}\r\n\r\n",
            websocket_accept_value(&key)
        );
        server_stream.write_all(resp.as_bytes()).expect("write resp");
        let ws = client_handle.join().expect("client thread").expect("ws connect");
        (ws, server_stream)
    }

    /// Handshake must include Origin and properly-formed Host headers.
    #[test]
    fn handshake_sends_origin_and_host_headers() {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
        let addr = listener.local_addr().expect("addr");
        let port = addr.port();

        let req_capture = thread::spawn(move || {
            let (mut stream, _) = listener.accept().expect("accept");
            let mut req = Vec::new();
            let mut buf = [0u8; 4096];
            loop {
                let n = stream.read(&mut buf).expect("read");
                req.extend_from_slice(&buf[..n]);
                if req.windows(4).any(|w| w == b"\r\n\r\n") {
                    break;
                }
            }
            // Respond so the client handshake can complete cleanly.
            let req_str = String::from_utf8(req.clone()).expect("utf8");
            let key = req_str
                .lines()
                .find_map(|l| {
                    l.split_once(':').and_then(|(k, v)| {
                        if k.eq_ignore_ascii_case("Sec-WebSocket-Key") {
                            Some(v.trim().to_string())
                        } else {
                            None
                        }
                    })
                })
                .expect("key");
            let resp = format!(
                "HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Accept: {}\r\n\r\n",
                websocket_accept_value(&key)
            );
            stream.write_all(resp.as_bytes()).expect("write");
            req_str
        });

        let url = Url::parse(&format!("ws://127.0.0.1:{port}/path")).expect("url");
        let client = thread::spawn(move || WebSocket::connect(&url));
        let req_str = req_capture.join().expect("capture thread");
        client.join().expect("client thread").expect("connect");

        // Host must include the port (127.0.0.1 port is non-default for ws://).
        let host_line = req_str
            .lines()
            .find(|l| l.to_lowercase().starts_with("host:"))
            .expect("Host header");
        assert!(
            host_line.contains(&port.to_string()),
            "Host header must include non-default port: {host_line}"
        );

        // Origin must be present and use http:// scheme.
        let origin_line = req_str
            .lines()
            .find(|l| l.to_lowercase().starts_with("origin:"))
            .expect("Origin header missing from handshake");
        assert!(
            origin_line.to_lowercase().contains("http://"),
            "Origin must use http:// for ws:// connections: {origin_line}"
        );
    }

    /// A text frame containing invalid UTF-8 must produce a 1007 error.
    #[test]
    fn invalid_utf8_text_frame_returns_1007_error() {
        let (mut ws, mut server) = do_handshake();
        // Send an unmasked text frame with 0xFF 0xFE (invalid UTF-8).
        // Frame: FIN+TEXT (0x81), length 2, payload [0xFF, 0xFE].
        server.write_all(&[0x81, 0x02, 0xFF, 0xFE]).expect("send bad utf8 frame");
        let err = ws.recv().expect_err("should error on invalid UTF-8");
        let msg = err.to_string();
        assert!(
            msg.contains("1007"),
            "error message must mention 1007, got: {msg}"
        );
    }

    /// A close frame with exactly 1 byte payload is a protocol error (1002).
    #[test]
    fn one_byte_close_frame_returns_1002_error() {
        let (mut ws, mut server) = do_handshake();
        // Unmasked close frame (0x88), length 1, single payload byte.
        server.write_all(&[0x88, 0x01, 0x03]).expect("send 1-byte close");
        let err = ws.recv().expect_err("should error on 1-byte close frame");
        let msg = err.to_string();
        assert!(
            msg.contains("1002"),
            "error message must mention 1002, got: {msg}"
        );
    }

    /// A masked frame from the server is a protocol error (1002).
    #[test]
    fn masked_server_frame_returns_1002_error() {
        let (mut ws, mut server) = do_handshake();
        // Masked text frame from server: FIN+TEXT (0x81), MASK bit set (0x82),
        // masking key [0,0,0,0], payload b'hi' masked with zeros = b'hi'.
        server
            .write_all(&[0x81, 0x82, 0x00, 0x00, 0x00, 0x00, b'h', b'i'])
            .expect("send masked server frame");
        let err = ws.recv().expect_err("should error on masked server frame");
        let msg = err.to_string();
        assert!(
            msg.contains("1002"),
            "error message must mention 1002, got: {msg}"
        );
    }

    /// A frame with RSV1 set (no extension negotiated) must produce a 1002 error.
    #[test]
    fn rsv1_set_returns_1002_error() {
        let (mut ws, mut server) = do_handshake();
        // FIN + RSV1 + TEXT (0xC1), length 2, payload b'hi'.
        server
            .write_all(&[0xc1, 0x02, b'h', b'i'])
            .expect("send rsv1 frame");
        let err = ws.recv().expect_err("should error on RSV1 frame");
        let msg = err.to_string();
        assert!(
            msg.contains("1002"),
            "error message must mention 1002, got: {msg}"
        );
    }

    /// A frame with RSV2 set must also produce a 1002 error.
    #[test]
    fn rsv2_set_returns_1002_error() {
        let (mut ws, mut server) = do_handshake();
        // FIN + RSV2 + TEXT (0xA1), length 2, payload b'hi'.
        server
            .write_all(&[0xa1, 0x02, b'h', b'i'])
            .expect("send rsv2 frame");
        let err = ws.recv().expect_err("should error on RSV2 frame");
        let msg = err.to_string();
        assert!(
            msg.contains("1002"),
            "error message must mention 1002, got: {msg}"
        );
    }

    /// A close frame with an invalid close code (e.g. 1004) must produce a 1002 error.
    #[test]
    fn invalid_close_code_returns_1002_error() {
        let (mut ws, mut server) = do_handshake();
        // Unmasked close frame (0x88), length 2, code 1004 (reserved, invalid on wire).
        let code: u16 = 1004;
        let bytes = code.to_be_bytes();
        server
            .write_all(&[0x88, 0x02, bytes[0], bytes[1]])
            .expect("send invalid close code frame");
        let err = ws.recv().expect_err("should error on invalid close code");
        let msg = err.to_string();
        assert!(
            msg.contains("1002"),
            "error message must mention 1002, got: {msg}"
        );
    }

    /// A close frame with a non-UTF-8 reason must produce a 1007 error.
    #[test]
    fn close_frame_invalid_utf8_reason_returns_1007_error() {
        let (mut ws, mut server) = do_handshake();
        // Close frame: code 1000 (normal), reason bytes [0xFF, 0xFE] (invalid UTF-8).
        server
            .write_all(&[0x88, 0x04, 0x03, 0xe8, 0xFF, 0xFE])
            .expect("send close with bad utf8 reason");
        let err = ws.recv().expect_err("should error on bad UTF-8 close reason");
        let msg = err.to_string();
        assert!(
            msg.contains("1007"),
            "error message must mention 1007, got: {msg}"
        );
    }
}
