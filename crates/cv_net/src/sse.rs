//! Server-Sent Events (`text/event-stream`) streaming client per
//! WHATWG HTML §9.2. Unlike [`crate::http1::Client::fetch`], which buffers
//! the entire response body before returning, this opens the connection and
//! exposes the body bytes *incrementally* via [`SseConnection::poll`] — which
//! is mandatory for an event stream that stays open indefinitely and trickles
//! events out over time.
//!
//! The transport mirrors the WebSocket client (a raw [`Socket`] or
//! [`TlsStream`]): we write a plain HTTP/1.1 GET, read+parse the response
//! status line and headers, then return remaining buffered bytes plus a handle
//! that keeps reading the body. Both `Content-Length`/EOF (identity) and
//! `Transfer-Encoding: chunked` framings are decoded on the fly.
//!
//! The JS-facing reconnection logic (retry interval, `Last-Event-ID`, the
//! CONNECTING/OPEN/CLOSED state machine and event dispatch) lives one layer up
//! in `cv_browser`; this module only owns "open the stream and hand me the next
//! decoded body bytes without blocking."

use crate::NetError;
use crate::dns::resolve;
use crate::socket::Socket;
use crate::tls::TlsStream;
use cv_url::Url;

/// One end of an open `text/event-stream` connection. Either a plain TCP
/// socket or a TLS stream, plus the decode state for the response body.
#[derive(Debug)]
pub struct SseConnection {
    conn: SseConn,
    /// HTTP status of the response (200 on success).
    pub status: u16,
    /// Lowercased response header (name, value) pairs.
    pub headers: Vec<(String, String)>,
    /// Raw (still-encoded) body bytes read but not yet decoded — needed
    /// because a chunk header may straddle a read boundary.
    rx_raw: Vec<u8>,
    /// Whether the body uses `Transfer-Encoding: chunked`.
    chunked: bool,
    /// Chunked decode cursor / remaining bytes in the current chunk.
    chunk_remaining: usize,
    /// `Some(n)` when the response carried a `Content-Length: n` (identity
    /// framing). Once we have delivered `n` body bytes the stream is complete
    /// and `eof` is set — this is how we detect that a finite SSE batch (which
    /// many servers send per-connection with `Connection: close`) has ended and
    /// the client should reconnect.
    content_length: Option<usize>,
    /// Count of decoded body bytes delivered so far (identity framing).
    delivered: usize,
    /// Set once the chunked terminator (size-0 chunk) is seen, the
    /// Content-Length has been fully delivered, or the peer closed the stream.
    eof: bool,
}

#[derive(Debug)]
enum SseConn {
    Plain(Socket),
    Tls(TlsStream),
}

impl SseConn {
    fn write_all(&mut self, data: &[u8]) -> Result<(), NetError> {
        match self {
            Self::Plain(s) => s.write_all(data).map_err(NetError::Socket),
            Self::Tls(s) => s.write_all(data).map_err(|e| NetError::Http(e.to_string())),
        }
    }

    /// Read up to `dst.len()` bytes, returning 0 on timeout (no data yet) and
    /// signalling clean EOF separately to the caller via `Ok(0)` when the peer
    /// closed. We disambiguate timeout-vs-EOF at the call site by checking the
    /// socket; here a 0 simply means "nothing right now".
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

    /// Blocking read used during the handshake (header) phase.
    fn read(&mut self, dst: &mut [u8]) -> Result<usize, NetError> {
        match self {
            Self::Plain(s) => s.read(dst).map_err(NetError::Socket),
            Self::Tls(s) => s.read(dst).map_err(|e| NetError::Http(e.to_string())),
        }
    }

    /// After a 0-byte read, distinguish a clean peer close (FIN/RST → socket
    /// becomes readable-with-0) from a quiet keep-alive (nothing readable).
    /// `is_reuse_safe()` is `true` only when the connection looks healthy and
    /// idle, so its negation is "the peer closed (or pushed an error)".
    fn is_peer_closed(&self) -> bool {
        match self {
            Self::Plain(s) => !s.is_reuse_safe(),
            Self::Tls(s) => !s.is_reuse_safe(),
        }
    }
}

impl SseConnection {
    /// Open a `text/event-stream` GET against `url`. `last_event_id` (when
    /// non-empty) is sent as the `Last-Event-ID` header per HTML §9.2 so the
    /// server can resume the stream after the last event the client saw.
    /// `cookie_header` (when non-empty) is sent verbatim as `Cookie:`.
    ///
    /// Returns once the response headers have been read. Body bytes are then
    /// pulled incrementally via [`Self::poll`].
    pub fn connect(
        url: &Url,
        last_event_id: &str,
        cookie_header: &str,
    ) -> Result<Self, NetError> {
        let is_secure = match url.scheme.as_str() {
            "http" => false,
            "https" => true,
            other => return Err(NetError::Url(format!("not an http(s) scheme: {other}"))),
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
        let mut last_err: Option<NetError> = None;
        let mut sock = None;
        for a in addrs {
            // Give the connect a bounded budget; the stream itself is polled
            // non-blocking afterwards.
            match Socket::connect_with_timeout(&a, 30_000) {
                Ok(s) => {
                    sock = Some(s);
                    break;
                }
                Err(e) => last_err = Some(NetError::Socket(e)),
            }
        }
        let raw = sock.ok_or_else(|| last_err.unwrap_or_else(|| NetError::Http("no addr".into())))?;
        let mut conn = if is_secure {
            SseConn::Tls(TlsStream::connect(raw, &host)?)
        } else {
            SseConn::Plain(raw)
        };

        // Build the request. Per HTML §9.2 the request's Accept is
        // `text/event-stream` and caching is bypassed (`Cache-Control: no-store`).
        let default_port: u16 = if is_secure { 443 } else { 80 };
        let host_header = match url.port {
            Some(p) if p != default_port => format!("{host}:{p}"),
            _ => host.clone(),
        };
        let target = {
            let t = url.request_target();
            if t.is_empty() { "/".to_string() } else { t }
        };
        let mut wire = String::new();
        wire.push_str(&format!("GET {target} HTTP/1.1\r\n"));
        wire.push_str(&format!("Host: {host_header}\r\n"));
        wire.push_str("Accept: text/event-stream\r\n");
        wire.push_str("Cache-Control: no-store\r\n");
        wire.push_str("Connection: keep-alive\r\n");
        if !last_event_id.is_empty() {
            // HTML §9.2: "Let lastEventIDValue be the EventSource object's last
            // event ID string. ... set (`Last-Event-ID`, lastEventIDValue)."
            wire.push_str(&format!("Last-Event-ID: {last_event_id}\r\n"));
        }
        if !cookie_header.is_empty() {
            wire.push_str(&format!("Cookie: {cookie_header}\r\n"));
        }
        wire.push_str("\r\n");
        conn.write_all(wire.as_bytes())?;

        // Read until the header terminator \r\n\r\n.
        let mut head: Vec<u8> = Vec::with_capacity(512);
        let mut buf = [0u8; 1024];
        loop {
            let n = conn.read(&mut buf)?;
            if n == 0 {
                return Err(NetError::Http("sse: closed during handshake".into()));
            }
            head.extend_from_slice(&buf[..n]);
            if find_double_crlf(&head).is_some() {
                break;
            }
            if head.len() > 64 * 1024 {
                return Err(NetError::Http("sse: response headers too large".into()));
            }
        }
        let header_end = find_double_crlf(&head).expect("checked above") + 4;
        let (head_bytes, body_tail) = head.split_at(header_end);
        let head_str = std::str::from_utf8(head_bytes)
            .map_err(|_| NetError::Http("sse: non-utf8 headers".into()))?;
        let mut lines = head_str.split("\r\n");
        let status_line = lines.next().unwrap_or("");
        let status = status_line
            .split_whitespace()
            .nth(1)
            .and_then(|c| c.parse::<u16>().ok())
            .unwrap_or(0);
        let mut headers: Vec<(String, String)> = Vec::new();
        let mut chunked = false;
        let mut content_length: Option<usize> = None;
        for line in lines {
            if line.is_empty() {
                continue;
            }
            if let Some((k, v)) = line.split_once(':') {
                let k_lc = k.trim().to_ascii_lowercase();
                let v = v.trim().to_string();
                if k_lc == "transfer-encoding" && v.to_ascii_lowercase().contains("chunked") {
                    chunked = true;
                } else if k_lc == "content-length" {
                    content_length = v.parse::<usize>().ok();
                }
                headers.push((k_lc, v));
            }
        }
        // Transfer-Encoding: chunked takes precedence over Content-Length
        // (RFC 9112 §6.1); ignore a stray Content-Length when chunked.
        if chunked {
            content_length = None;
        }

        Ok(Self {
            conn,
            status,
            headers,
            rx_raw: body_tail.to_vec(),
            chunked,
            chunk_remaining: 0,
            content_length,
            delivered: 0,
            eof: false,
        })
    }

    /// `text/event-stream` is the required Content-Type for a valid SSE
    /// response (HTML §9.2: "if res's Content-Type is not `text/event-stream`,
    /// then fail the connection"). The check is on the essence (ignoring any
    /// `;charset=` parameters).
    pub fn is_event_stream(&self) -> bool {
        self.headers
            .iter()
            .find(|(k, _)| k == "content-type")
            .map(|(_, v)| {
                v.split(';')
                    .next()
                    .unwrap_or("")
                    .trim()
                    .eq_ignore_ascii_case("text/event-stream")
            })
            .unwrap_or(false)
    }

    /// Has the stream ended (peer closed / chunked terminator seen)?
    pub fn is_eof(&self) -> bool {
        self.eof
    }

    /// Pull any decoded body bytes that have arrived since the last poll,
    /// waiting at most `timeout_ms` for the socket. Returns:
    ///   - `Ok(Some(bytes))` — newly decoded UTF-8-ish stream bytes (possibly a
    ///     partial event; the caller's SSE parser buffers across calls).
    ///   - `Ok(None)`        — nothing new yet (timeout) but the stream is live.
    ///   - the `eof` flag flips to true on clean peer close; callers should
    ///     check [`Self::is_eof`] to decide whether to reconnect.
    ///   - `Err(_)`          — a transport error; treat like EOF + reconnect.
    pub fn poll(&mut self, timeout_ms: u32) -> Result<Option<Vec<u8>>, NetError> {
        if self.eof {
            return Ok(None);
        }
        // Only hit the socket when nothing is already buffered. The header read
        // can over-read into the body (the whole first SSE batch can arrive with
        // the headers), so decode the leftover `rx_raw` FIRST — otherwise a small
        // single-read response would never be parsed.
        if self.rx_raw.is_empty() {
            let mut buf = [0u8; 8192];
            let n = self.conn.read_with_timeout(&mut buf, timeout_ms)?;
            if n == 0 {
                // `read_with_timeout` maps WSAETIMEDOUT → Ok(0), so a 0 is EITHER
                // a read timeout (no data yet, stream still live) OR a clean peer
                // close (FIN). Probe to disambiguate: if the peer closed, mark
                // eof so the worker reconnects; otherwise it's just an idle stream.
                if self.conn.is_peer_closed() {
                    self.eof = true;
                }
                return Ok(None);
            }
            self.rx_raw.extend_from_slice(&buf[..n]);
        }
        let decoded = if self.chunked {
            decode_chunked(&mut self.rx_raw, &mut self.chunk_remaining, &mut self.eof)
        } else {
            // Identity framing: every raw byte is body.
            std::mem::take(&mut self.rx_raw)
        };
        // Identity framing with a known Content-Length: once we have delivered
        // exactly that many bytes the response is complete (a finite per-batch
        // SSE response that the server ends with `Connection: close`).
        if !self.chunked {
            self.delivered += decoded.len();
            if let Some(len) = self.content_length {
                if self.delivered >= len {
                    self.eof = true;
                }
            }
        }
        if decoded.is_empty() {
            Ok(None)
        } else {
            Ok(Some(decoded))
        }
    }
}

/// Decode as many complete (or partial) chunks as `rx_raw` currently holds,
/// per RFC 9112 §7.1. Leaves any straddling chunk header / partial data in
/// `rx_raw` for the next poll. Sets `*eof` on the size-0 terminating chunk and
/// carries `*chunk_remaining` across calls. Pulled out as a free function so it
/// is unit-testable without a live socket.
fn decode_chunked(rx_raw: &mut Vec<u8>, chunk_remaining: &mut usize, eof: &mut bool) -> Vec<u8> {
    let mut out = Vec::new();
    loop {
        if *chunk_remaining > 0 {
            // Emit up to chunk_remaining bytes of data from the front.
            let take = (*chunk_remaining).min(rx_raw.len());
            out.extend_from_slice(&rx_raw[..take]);
            rx_raw.drain(..take);
            *chunk_remaining -= take;
            if *chunk_remaining > 0 {
                // Ran out of buffered bytes mid-chunk; wait for more.
                break;
            }
            // Consume the trailing CRLF after the chunk data.
            if rx_raw.starts_with(b"\r\n") {
                rx_raw.drain(..2);
            } else {
                break; // CRLF straddles a read boundary (or not arrived yet).
            }
            continue;
        }
        // Read the next chunk-size line.
        let Some(eol) = find_crlf(rx_raw) else {
            break; // header straddles a boundary.
        };
        let line = String::from_utf8_lossy(&rx_raw[..eol]).into_owned();
        // chunk-size [ ; chunk-ext ]
        let size_str = line.split(';').next().unwrap_or("").trim();
        let size = usize::from_str_radix(size_str, 16).unwrap_or(0);
        rx_raw.drain(..eol + 2); // drop the size line + CRLF.
        if size == 0 {
            // Last chunk. Trailers (until the final CRLF) are ignored.
            *eof = true;
            break;
        }
        *chunk_remaining = size;
    }
    out
}

/// Find the byte offset of the first `\r\n\r\n` (header terminator).
fn find_double_crlf(buf: &[u8]) -> Option<usize> {
    buf.windows(4).position(|w| w == b"\r\n\r\n")
}

/// Find the byte offset of the first `\r\n`.
fn find_crlf(buf: &[u8]) -> Option<usize> {
    buf.windows(2).position(|w| w == b"\r\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunked_decodes_across_boundaries() {
        // "data: hi\n\n" split into two chunks: "data: hi" then "\n\n".
        let mut raw = b"8\r\ndata: hi\r\n2\r\n\n\n\r\n0\r\n\r\n".to_vec();
        let mut rem = 0usize;
        let mut eof = false;
        let got = decode_chunked(&mut raw, &mut rem, &mut eof);
        assert_eq!(got, b"data: hi\n\n");
        assert!(eof, "size-0 chunk must flip eof");
    }

    #[test]
    fn chunked_partial_header_waits() {
        // Only part of a 5-byte chunk present — decoder emits the 3 arrived
        // bytes and remembers that 2 are still outstanding.
        let mut raw = b"5\r\nhel".to_vec();
        let mut rem = 0usize;
        let mut eof = false;
        let got = decode_chunked(&mut raw, &mut rem, &mut eof);
        assert_eq!(got, b"hel");
        assert_eq!(rem, 2, "2 bytes of the 5-byte chunk remain");
        assert!(!eof);
    }

    #[test]
    fn chunked_resumes_after_more_bytes() {
        // First poll has a partial chunk; the second poll completes it.
        let mut raw = b"5\r\nhel".to_vec();
        let mut rem = 0usize;
        let mut eof = false;
        let first = decode_chunked(&mut raw, &mut rem, &mut eof);
        assert_eq!(first, b"hel");
        raw.extend_from_slice(b"lo\r\n0\r\n\r\n");
        let second = decode_chunked(&mut raw, &mut rem, &mut eof);
        assert_eq!(second, b"lo");
        assert!(eof);
    }

    #[test]
    fn find_helpers() {
        assert_eq!(find_crlf(b"abc\r\ndef"), Some(3));
        assert_eq!(find_double_crlf(b"h: v\r\n\r\nbody"), Some(4));
        assert_eq!(find_crlf(b"nocrlf"), None);
    }
}
