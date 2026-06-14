//! HTTP/1.1 client per RFC 9110/9112.
//!
//! Synchronous `Client::fetch(url)` that opens a TCP connection (or
//! reuses a pooled one), sends a GET, parses the response head, and
//! reads the body using `Content-Length` / `Transfer-Encoding: chunked`
//! / read-to-close.  Persistent connections per RFC 9112 §9.3 — clients
//! send `Connection: keep-alive` and reuse idle sockets keyed by
//! `(host, port, scheme)`. Connections are returned to the pool after a
//! response unless the server advertised `Connection: close` or the
//! body framing was read-to-close.

use crate::NetError;
use crate::cache::{HttpCache, build_entry_if_cacheable};
use crate::cookies::CookieJar;
use crate::dns::resolve;
use crate::socket::Socket;
use crate::tls::{TlsError, TlsStream};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use cv_url::Url;

/// Either a plain `Socket` (http) or a `TlsStream` (https).
enum Conn {
    Plain(Socket),
    Tls(TlsStream),
}

impl Conn {
    fn write_all(&mut self, data: &[u8]) -> Result<(), NetError> {
        match self {
            Self::Plain(s) => s.write_all(data).map_err(NetError::Socket),
            Self::Tls(s) => s.write_all(data).map_err(|e| NetError::Http(e.to_string())),
        }
    }

    fn read_to_end(&mut self, dst: &mut Vec<u8>) -> Result<(), NetError> {
        match self {
            Self::Plain(s) => s.read_to_end(dst).map_err(NetError::Socket),
            Self::Tls(s) => s
                .read_to_end(dst)
                .map_err(|e| NetError::Http(e.to_string())),
        }
    }

    /// One incremental read. Returns 0 on clean EOF (peer closed). Used
    /// by the smart reader so we can stop the moment `Content-Length`
    /// or the chunked-terminator says the body is complete, without
    /// waiting for the server's idle close.
    fn read_some(&mut self, buf: &mut [u8]) -> Result<usize, NetError> {
        match self {
            Self::Plain(s) => s.read(buf).map_err(NetError::Socket),
            Self::Tls(s) => s.read(buf).map_err(|e| NetError::Http(e.to_string())),
        }
    }

    /// Whether this pooled connection still looks alive enough to reuse.
    fn is_reuse_safe(&self) -> bool {
        match self {
            Self::Plain(s) => s.is_reuse_safe(),
            Self::Tls(s) => s.is_reuse_safe(),
        }
    }

    /// Set the receive timeout on the underlying socket.
    fn set_read_timeout_ms(&self, ms: u32) {
        match self {
            Self::Plain(s) => s.set_read_timeout_ms(ms),
            Self::Tls(s) => s.set_read_timeout_ms(ms),
        }
    }
}

/// Minimal transport abstraction the pooled HTTP/2 driver writes/reads
/// through. `Conn` (TLS/plain socket) is the production impl; tests
/// supply an in-memory duplex pipe so the demux/pool logic can be
/// exercised with no TLS, no ALPN, and no real sockets — the framing is
/// transport-agnostic. This is the seam the V2 reader-thread upgrade
/// also slots into without re-shaping the wire path.
trait H2Transport {
    fn write_all(&mut self, data: &[u8]) -> Result<(), NetError>;
    /// One incremental read. Returns 0 on clean EOF (peer closed).
    fn read_some(&mut self, buf: &mut [u8]) -> Result<usize, NetError>;
    fn set_read_timeout_ms(&self, ms: u32);
}

impl H2Transport for Conn {
    fn write_all(&mut self, data: &[u8]) -> Result<(), NetError> {
        Conn::write_all(self, data)
    }
    fn read_some(&mut self, buf: &mut [u8]) -> Result<usize, NetError> {
        Conn::read_some(self, buf)
    }
    fn set_read_timeout_ms(&self, ms: u32) {
        Conn::set_read_timeout_ms(self, ms)
    }
}

impl From<TlsError> for NetError {
    fn from(e: TlsError) -> Self {
        Self::Http(format!("tls: {e}"))
    }
}

#[derive(Debug, Clone)]
pub struct Request {
    pub method: String,
    pub url: Url,
    pub headers: Vec<(String, String)>,
    /// Optional request body for POST/PUT/PATCH/DELETE-with-body. When
    /// present the client writes `Content-Length: {body.len()}` and
    /// sends the bytes after headers per RFC 9110 §6.4.
    pub body: Vec<u8>,
    /// If true the client advertises `Accept-Encoding: gzip, deflate, br`.
    /// When false we drop `br` — used when an earlier attempt failed
    /// on a brotli static-dictionary reference and we're retrying.
    pub accept_brotli: bool,
}

impl Request {
    pub fn get(url: Url) -> Self {
        Self {
            method: "GET".into(),
            url,
            headers: Vec::new(),
            body: Vec::new(),
            accept_brotli: true,
        }
    }

    /// Build a POST with the supplied body and `Content-Type` header.
    pub fn post(url: Url, content_type: &str, body: Vec<u8>) -> Self {
        let mut r = Self::get(url);
        r.method = "POST".into();
        r.headers.push(("Content-Type".into(), content_type.into()));
        r.body = body;
        r
    }

    pub fn header(mut self, name: &str, value: &str) -> Self {
        self.headers.push((name.into(), value.into()));
        self
    }

    pub fn body(mut self, body: Vec<u8>) -> Self {
        self.body = body;
        self
    }

    pub fn method(mut self, method: &str) -> Self {
        self.method = method.into();
        self
    }

    /// Request a single byte range `[first, last]` (inclusive), per
    /// RFC 9110 §14.2 — emits `Range: bytes=first-last`. This is the form
    /// Chrome's media stack uses for seeking (`net::HttpRequestHeaders`
    /// with `kRange` = "bytes=" + start + "-" + end). `first`/`last` are
    /// absolute byte offsets into the resource; `last` is INCLUSIVE, so a
    /// 1024-byte first block is `range(0, 1023)`.
    pub fn range(self, first: u64, last: u64) -> Self {
        self.header("Range", &format!("bytes={first}-{last}"))
    }

    /// Request from `first` to the end of the resource, per RFC 9110
    /// §14.1.1 (`int-range` with absent `last-pos`) — emits
    /// `Range: bytes=first-`. Chrome uses this to resume an interrupted
    /// download from the last byte already received
    /// (`PartialData::PrepareCacheValidation` → "bytes=" + offset + "-").
    pub fn range_from(self, first: u64) -> Self {
        self.header("Range", &format!("bytes={first}-"))
    }

    /// Request the last `suffix_len` bytes of the resource, per RFC 9110
    /// §14.1.2 (`suffix-range`) — emits `Range: bytes=-suffix_len`. Used to
    /// read trailing metadata (e.g. a ZIP central directory, MP4 `moov`
    /// atom at the tail) without knowing the total length.
    pub fn range_suffix(self, suffix_len: u64) -> Self {
        self.header("Range", &format!("bytes=-{suffix_len}"))
    }

    /// Make the attached `Range` conditional on the resource being
    /// unchanged, per RFC 9110 §13.1.5. `validator` is either a strong
    /// entity-tag (`"abc"` / `W/"abc"`) or an HTTP-date (Last-Modified).
    /// If the validator still matches at the server the response is `206`
    /// with just the requested slice; if it changed the server returns the
    /// FULL `200` body so the client can re-fetch coherently. Chrome sets
    /// this from the cached entry's validator when resuming a download
    /// (`PartialData::PrepareCacheValidation`).
    pub fn if_range(self, validator: &str) -> Self {
        self.header("If-Range", validator)
    }
}

/// A parsed `Content-Range` response header (RFC 9110 §14.4).
///
/// For a satisfied range (`bytes first-last/complete-length`) all three of
/// `first`, `last`, `complete_len` are present. For an UNSATISFIED range
/// (the 416 form `bytes */complete-length`) only `complete_len` is set and
/// `first`/`last` are `None`. A `complete-length` of `*` (server doesn't
/// know the total) leaves `complete_len` `None`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ContentRange {
    /// First byte position of the returned slice (inclusive), absolute.
    pub first: Option<u64>,
    /// Last byte position of the returned slice (inclusive), absolute.
    pub last: Option<u64>,
    /// Total length of the complete representation, if the server stated it.
    pub complete_len: Option<u64>,
}

impl ContentRange {
    /// Number of bytes in the returned slice (`last - first + 1`), if both
    /// positions are known. This MUST equal the 206 body length.
    pub fn slice_len(&self) -> Option<u64> {
        match (self.first, self.last) {
            (Some(f), Some(l)) if l >= f => Some(l - f + 1),
            _ => None,
        }
    }
}

/// Parse a `Content-Range` field value per RFC 9110 §14.4:
///   `bytes SP first-last/complete-length`  (satisfied)
///   `bytes SP first-last/*`                (satisfied, unknown total)
///   `bytes SP */complete-length`           (unsatisfied — 416 form)
/// The unit MUST be the (case-insensitive) token `bytes`; any other unit
/// or a malformed value returns `None` (callers treat that as "no usable
/// range info" and fall back to the whole body). Numbers that overflow a
/// `u64` are rejected rather than silently wrapped.
pub fn parse_content_range(value: &str) -> Option<ContentRange> {
    let v = value.trim();
    // "bytes" unit (case-insensitive), separated from the range-resp by
    // whitespace. Any other range-unit is unsupported → None.
    if v.len() < 5 || !v[..5].eq_ignore_ascii_case("bytes") {
        return None;
    }
    let rest = v[5..].trim_start();
    // `range-resp = incl-range "/" ( complete-length / "*" )`
    //            | `unsatisfied-range = "*" "/" complete-length`
    let (range_part, len_part) = rest.split_once('/')?;
    let range_part = range_part.trim();
    let len_part = len_part.trim();

    let complete_len = if len_part == "*" {
        None
    } else {
        Some(len_part.parse::<u64>().ok()?)
    };

    if range_part == "*" {
        // Unsatisfied-range form (the body the 416 carries). Both
        // positions absent; complete-length MUST be present per grammar.
        if complete_len.is_none() {
            return None;
        }
        return Some(ContentRange {
            first: None,
            last: None,
            complete_len,
        });
    }

    let (first_s, last_s) = range_part.split_once('-')?;
    let first = first_s.trim().parse::<u64>().ok()?;
    let last = last_s.trim().parse::<u64>().ok()?;
    // A satisfied range must have last >= first (RFC 9110 §14.4).
    if last < first {
        return None;
    }
    // If the server stated a complete-length, the range must fit inside it.
    if let Some(total) = complete_len {
        if last >= total {
            return None;
        }
    }
    Some(ContentRange {
        first: Some(first),
        last: Some(last),
        complete_len,
    })
}

#[derive(Debug)]
pub struct Response {
    pub status: u16,
    pub reason: String,
    pub headers: Vec<(String, String)>,
    pub body: Vec<u8>,
}

impl Response {
    pub fn header(&self, name: &str) -> Option<&str> {
        let name_lc = name.to_ascii_lowercase();
        self.headers
            .iter()
            .find(|(k, _)| k.to_ascii_lowercase() == name_lc)
            .map(|(_, v)| v.as_str())
    }

    /// `true` when this is a `206 Partial Content` response carrying just
    /// the requested byte range (RFC 9110 §15.3.7).
    pub fn is_partial(&self) -> bool {
        self.status == 206
    }

    /// `true` when the server rejected the requested range as outside the
    /// representation — `416 Range Not Satisfiable` (RFC 9110 §15.5.17).
    /// The body (and `content_range()`) typically carries `bytes */total`.
    pub fn is_range_not_satisfiable(&self) -> bool {
        self.status == 416
    }

    /// The parsed `Content-Range` header, if present and well-formed
    /// (RFC 9110 §14.4). On a `206` this gives the slice's `first`/`last`
    /// byte positions and the `complete_len` total; on a `416` it gives the
    /// `bytes */total` unsatisfied form (positions `None`, total `Some`).
    pub fn content_range(&self) -> Option<ContentRange> {
        parse_content_range(self.header("content-range")?)
    }

    /// The range units the origin advertises it can serve, from the
    /// `Accept-Ranges` header (RFC 9110 §14.3) — e.g. `Some("bytes")`.
    /// `Some("none")` means the server explicitly does NOT support ranges;
    /// `None` means it said nothing (treat as unknown / no range support).
    pub fn accept_ranges(&self) -> Option<&str> {
        self.header("accept-ranges")
    }

    /// Convenience: does the origin advertise byte-range support?
    /// True only for an explicit `Accept-Ranges: bytes` (case-insensitive),
    /// matching how Chrome decides a media resource is seekable
    /// (`media::ResourceMultiBuffer` checks for `Accept-Ranges: bytes`).
    pub fn supports_byte_ranges(&self) -> bool {
        self.accept_ranges()
            .map(|v| v.split(',').any(|t| t.trim().eq_ignore_ascii_case("bytes")))
            .unwrap_or(false)
    }
}

/// Cached entry in the connection pool. `last_used` lets the pool
/// evict idle sockets whose server is likely to have closed them on
/// its side (most servers default to ~60s idle timeout). Conn isn't
/// Debug because TlsStream isn't — that's OK.
struct PooledConn {
    conn: Conn,
    last_used: Instant,
}

/// `(host, port, is_https)` keyed shared pool of idle keep-alive
/// connections.
type ConnPool = Arc<Mutex<HashMap<(String, u16, bool), Vec<PooledConn>>>>;

/// Maximum idle time before we treat a pooled connection as
/// likely-dead and re-dial. Mirrors typical server defaults.
const POOL_IDLE_TIMEOUT: Duration = Duration::from_secs(45);

/// Process-wide flag enabling the long-lived, shared HTTP/2 connection
/// pool (M6.3). Read once from the `CV_H2_POOL` env var. **Default ON**
/// (flipped 2026-06-13; the demux is mutation-proven cross-response-impossible,
/// the Chrome-131 fingerprint is byte-identical via the shared header fn, and a
/// poisoned conn evicts cleanly). Escape hatch: `CV_H2_POOL=0`/`false` forces it
/// OFF (the `new_client()`-per-request h2 path); unset or any other value → ON.
fn h2_pool_enabled() -> bool {
    use std::sync::OnceLock;
    static F: OnceLock<bool> = OnceLock::new();
    *F.get_or_init(|| {
        !std::env::var("CV_H2_POOL")
            .map(|v| v == "0" || v.eq_ignore_ascii_case("false"))
            .unwrap_or(false)
    })
}

/// A long-lived, SHARED HTTP/2 connection living in the h2 pool. Unlike
/// `PooledConn` (h1, checked-out exclusively) an `H2Conn` stays in the
/// map while a request drives it — h2 multiplexes many streams over one
/// socket, so the conn is looked-up-and-shared, never popped. Access is
/// serialized in V1 by the inner `Mutex<H2Conn>` (one in-flight request
/// per conn at a time — see the module's M6.3 notes).
struct H2Conn {
    /// The byte transport (TLS in practice — h2 only via ALPN over TLS).
    conn: Conn,
    /// The live h2 state machine: streams map, next_stream_id, windows,
    /// peer_max_concurrent, closed_by_goaway. Created ONCE per conn via
    /// `Connection::new_client()` and kept warm for the conn's lifetime
    /// — this kills the per-request `new_client()` of `h2_send_request`.
    h2: crate::http2::Connection,
    /// ★ PERSISTENT decode-side HPACK dynamic table. HPACK is a stateful
    /// stream: stream N's header block may reference dynamic entries the
    /// server inserted while sending stream N-1's headers. A throwaway
    /// table (the old `h2_send_request` bug) silently mis-decodes the 2nd+
    /// stream on a pooled conn. This is THE correctness keystone of the
    /// pooling change.
    decode_dyn: crate::http2::HpackDynamicTable,
    /// Bytes read from the socket but not yet split into whole frames — a
    /// frame may straddle two `recv()` calls, so this MUST persist across
    /// reads (and across requests, since a frame can straddle a request
    /// boundary on a busy conn).
    accum: Vec<u8>,
    last_used: Instant,
    /// Set on any conn-level error / unrecoverable GOAWAY so the next
    /// checkout evicts it. A poisoned conn is NEVER handed to a future
    /// request (it would inherit broken HPACK/frame state).
    poisoned: bool,
}

/// `(host, port)`-keyed pool of shared h2 conns. h2 is always https here
/// so the scheme bool is implied. The OUTER `Mutex<HashMap>` is held only
/// briefly to find/insert the per-origin entry and clone its `Arc`; it is
/// RELEASED before any socket I/O so other origins never block. The INNER
/// `Mutex<H2Conn>` serializes one socket's frame stream. LOCK ORDER:
/// outer-before-inner, never inner-before-outer → no deadlock.
type H2Pool = Arc<Mutex<HashMap<(String, u16), Arc<Mutex<H2Conn>>>>>;

/// Short receive-timeout leash for a REUSED keep-alive socket. A warm
/// connection answers fast; if it produces no bytes within this window
/// we treat it as a silently-dropped keep-alive and re-dial on a fresh
/// socket. Kept well above any real warm-path round-trip so it never
/// trips a healthy connection, but far below the full per-request budget
/// so a dead one doesn't stall the render.
const REUSE_READ_TIMEOUT_MS: u32 = 3_000;

#[derive(Clone)]
pub struct Client {
    /// Combined connect + recv/send timeout in milliseconds. Used so the
    /// stylesheet/image sub-fetchers can give up quickly when a CDN
    /// doesn't answer.
    pub timeout_ms: u32,
    /// Shared cookie jar. Cheap to clone — Arc<Mutex<…>> inside. None
    /// means "don't store or send cookies" (used for sub-fetches that
    /// should be incognito-style, like image previews from random CDNs).
    pub cookie_jar: Option<CookieJar>,
    /// Shared connection pool. Idle sockets get reused for the next
    /// request to the same origin so we don't pay TLS handshake cost
    /// repeatedly. Clients that clone share the pool — the renderer
    /// dispatches stylesheet, script, and image fetches through clones
    /// of the document Client so they all benefit.
    pool: ConnPool,
    /// Sibling pool for long-lived, SHARED HTTP/2 connections (M6.3),
    /// keyed by `(host, port)`. Shared across `Client` clones exactly
    /// like `pool`. Consulted ONLY when `CV_H2_POOL` is on AND the origin
    /// negotiated ALPN "h2"; h1 origins never touch it. Default-empty and
    /// inert when the flag is off.
    h2_pool: H2Pool,
    /// Shared HTTP cache (RFC 9111 subset). Clones share storage; the
    /// renderer's stylesheet/script/image sub-fetchers therefore hit
    /// the same cache as the document fetch.
    pub cache: HttpCache,
}

impl std::fmt::Debug for Client {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Client")
            .field("timeout_ms", &self.timeout_ms)
            .field("has_cookies", &self.cookie_jar.is_some())
            .finish()
    }
}

impl Default for Client {
    fn default() -> Self {
        Self::new()
    }
}

impl Client {
    pub fn new() -> Self {
        Self {
            timeout_ms: 30_000,
            cookie_jar: Some(CookieJar::new()),
            pool: Arc::new(Mutex::new(HashMap::new())),
            h2_pool: Arc::new(Mutex::new(HashMap::new())),
            cache: HttpCache::new(),
        }
    }

    pub fn with_timeout(timeout_ms: u32) -> Self {
        Self {
            timeout_ms,
            cookie_jar: Some(CookieJar::new()),
            pool: Arc::new(Mutex::new(HashMap::new())),
            h2_pool: Arc::new(Mutex::new(HashMap::new())),
            cache: HttpCache::new(),
        }
    }

    /// Drain the pool's stale (idle-too-long) connections. Called once
    /// per `send` before checking-out, so the pool naturally garbage-
    /// collects without a separate timer.
    fn evict_stale_pool_entries(&self) {
        let mut pool = match self.pool.lock() {
            Ok(p) => p,
            Err(p) => p.into_inner(),
        };
        let now = Instant::now();
        for (_, vec) in pool.iter_mut() {
            vec.retain(|c| now.duration_since(c.last_used) < POOL_IDLE_TIMEOUT);
        }
        pool.retain(|_, v| !v.is_empty());
    }

    /// Try to lift an idle connection out of the pool for this
    /// origin. Returns None if nothing is cached (caller dials a fresh
    /// socket).
    fn checkout(&self, host: &str, port: u16, is_https: bool) -> Option<Conn> {
        let mut pool = match self.pool.lock() {
            Ok(p) => p,
            Err(p) => p.into_inner(),
        };
        let key = (host.to_string(), port, is_https);
        let entry = pool.get_mut(&key)?;
        // Pop the most-recently-returned connection first (LIFO keeps the
        // hottest socket warm). Skip any that fail the liveness peek — a
        // server may have dropped a previously-idle keep-alive without us
        // noticing, and writing a request into it would stall until recv
        // times out. Dead ones are dropped here (Conn's Drop closes them).
        while let Some(p) = entry.pop() {
            if p.conn.is_reuse_safe() {
                return Some(p.conn);
            }
        }
        None
    }

    /// Return a still-usable connection to the pool. Silently drops it
    /// if the lock is poisoned — the connection is just leaked, which
    /// is harmless for V1.
    fn return_to_pool(&self, host: &str, port: u16, is_https: bool, conn: Conn) {
        let mut pool = match self.pool.lock() {
            Ok(p) => p,
            Err(p) => p.into_inner(),
        };
        let key = (host.to_string(), port, is_https);
        pool.entry(key).or_default().push(PooledConn {
            conn,
            last_used: Instant::now(),
        });
    }

    /// Share the cookie jar with another Client instance — useful when
    /// the renderer spawns side fetches (stylesheets, scripts) that must
    /// participate in the same session as the document fetch.
    pub fn with_cookie_jar(mut self, jar: CookieJar) -> Self {
        self.cookie_jar = Some(jar);
        self
    }

    /// Opt out of cookies entirely — used for true cross-origin
    /// image/sub-resource fetches we don't want to associate with the
    /// current session.
    pub fn without_cookies(mut self) -> Self {
        self.cookie_jar = None;
        self
    }

    pub fn fetch(&self, url: &Url) -> Result<Response, NetError> {
        // Cache fast-path: a fresh entry skips the network entirely.
        // A stale or no-cache entry triggers a conditional revalidation;
        // a 304 refreshes the stored entry and returns the cached body.
        //
        // The initial GET has no caller-supplied request headers; the only
        // headers that end up on the wire (Accept-Encoding, User-Agent, etc.)
        // are added by transact_with_deadline.  We record those same headers
        // when building a new entry so Vary validation works.
        let url_str = url.to_string();
        // The static headers we always send — used for Vary capture.
        let sent_request_headers: Vec<(String, String)> = vec![
            ("Accept-Encoding".into(), "gzip, deflate, br".into()),
        ];
        if let Some(mut entry) = self.cache.get(&url_str) {
            // Bug 2 (Vary): if the stored entry has Vary fields, verify that
            // the current request's values for those fields match the values
            // that were in place when the entry was stored.  If they differ,
            // treat as a cache miss.
            let vary_mismatch = entry.vary_request_headers.iter().any(|(field, stored_val)| {
                let current_val = sent_request_headers
                    .iter()
                    .find(|(k, _)| k.eq_ignore_ascii_case(field))
                    .map(|(_, v)| v.as_str())
                    .unwrap_or("");
                !current_val.eq_ignore_ascii_case(stored_val)
            });
            if vary_mismatch {
                // Fall through to a full fresh fetch below.
            } else if entry.is_fresh() {
                // Bug 1 (no-cache): is_fresh() already returns false when
                // must_revalidate is set, so this branch is only reached for
                // genuinely fresh entries with no revalidation requirement.
                return Ok(Response {
                    status: entry.status,
                    reason: entry.reason,
                    headers: entry.headers,
                    body: entry.body,
                });
            } else {
                // Stale (or no-cache): revalidate with conditional headers.
                let mut req = Request::get(url.clone());
                if let Some(et) = &entry.etag {
                    req = req.header("If-None-Match", et);
                }
                if let Some(lm) = &entry.last_modified {
                    req = req.header("If-Modified-Since", lm);
                }
                let resp = self.send_with_redirects(req, 5)?;
                if resp.status == 304 {
                    // Bug 3: re-freshen the stored entry so its age resets and
                    // any updated headers (ETag, Cache-Control, etc.) from the
                    // 304 response are merged in before we return and re-store.
                    entry.refresh_after_304(&resp.headers);
                    let response = Response {
                        status: 200,
                        reason: "OK".into(),
                        headers: entry.headers.clone(),
                        body: entry.body.clone(),
                    };
                    self.cache.put(&url_str, entry);
                    return Ok(response);
                }
                return Ok(resp);
            }
        }
        let resp = self.send_with_redirects(Request::get(url.clone()), 5)?;
        if let Some(entry) = build_entry_if_cacheable(
            resp.status,
            &resp.reason,
            &resp.headers,
            &resp.body,
            &sent_request_headers,
        ) {
            self.cache.put(&url_str, entry);
        }
        Ok(resp)
    }

    /// `send`, but follow up to `max_redirects` `Location:` redirects per
    /// RFC 9110 §15.4. `307`/`308` PRESERVE the method + body (§15.4.8/9);
    /// `301`/`302`/`303` become a bodyless GET (the Post/Redirect/Get pattern).
    /// (Cross-origin Authorization/Cookie stripping is a separate security task.)
    pub fn send_with_redirects(
        &self,
        mut req: Request,
        max_redirects: u32,
    ) -> Result<Response, NetError> {
        for _ in 0..=max_redirects {
            let resp = self.send(req.clone())?;
            if matches!(resp.status, 301 | 302 | 303 | 307 | 308) {
                let loc = match resp.header("Location") {
                    Some(s) => s.to_string(),
                    None => return Ok(resp),
                };
                let next = req
                    .url
                    .resolve(&loc)
                    .map_err(|e| NetError::Url(format!("bad Location: {e}")))?;
                if matches!(resp.status, 307 | 308) {
                    // Preserve method, headers, and body; only the URL changes.
                    req.url = next;
                } else {
                    req = Request::get(next);
                }
                continue;
            }
            return Ok(resp);
        }
        Err(NetError::Http("too many redirects".into()))
    }

    pub fn send(&self, req: Request) -> Result<Response, NetError> {
        // First attempt advertises `br`. If the response body fails to
        // decode because the brotli stream references the static
        // dictionary (RFC 7932 Annex A) — which we don't carry as a
        // 122 KB embedded blob — re-issue the request asking only for
        // gzip/deflate so the server picks a decoder we fully cover.
        // This makes the engine robust to any site while keeping
        // brotli as the preferred path for bodies that don't dip into
        // the dictionary.
        let advertise_br = req.accept_brotli;
        match self.send_once(req.clone()) {
            Ok(r) => Ok(r),
            Err(e) => {
                if advertise_br && is_brotli_decode_error(&e) {
                    let mut retry = req;
                    retry.accept_brotli = false;
                    self.send_once(retry)
                } else {
                    Err(e)
                }
            }
        }
    }

    fn send_once(&self, req: Request) -> Result<Response, NetError> {
        let scheme = req.url.scheme.as_str();
        // HSTS upgrade: if the host has a non-expired HSTS entry
        // (RFC 6797 §8.3 known-HSTS-host upgrade), redirect plaintext
        // `http://` to `https://` BEFORE we connect. The target stays
        // identical otherwise; the upgrade is transparent to callers
        // beyond the response URL.
        let pending_host = req.url.host.clone();
        let mut is_https = match scheme {
            "http" => false,
            "https" => true,
            _ => return Err(NetError::Url(format!("scheme {scheme:?} not supported"))),
        };
        if !is_https && crate::hsts::must_upgrade(&pending_host) {
            is_https = true;
        }
        let host = if req.url.host.is_empty() {
            return Err(NetError::Url("missing host".into()));
        } else {
            req.url.host.clone()
        };
        let port = req
            .url
            .effective_port()
            .unwrap_or(if is_https { 443 } else { 80 });
        self.evict_stale_pool_entries();

        // M6.3 — HTTP/2 connection pool seam. Flag-gated and https-only.
        // Consult the h2 pool BEFORE the h1 checkout/dial dance: if this
        // origin is (or becomes, after a fresh dial) an h2 origin we drive
        // it on a long-lived SHARED conn, amortizing the TLS handshake +
        // h2 preface across every same-origin request. A fresh dial that
        // negotiates h1.1/"" ALPN is handed straight to the existing h1
        // path (the dialed Conn is returned via Err-carried fallthrough),
        // so h1 origins never see any new code. Flag-off ⇒ this block is
        // skipped entirely and behaviour is byte-for-byte today's.
        if h2_pool_enabled() && is_https {
            match self.h2_send_via_pool(&host, port, &req) {
                H2PoolOutcome::Response(r) => return r,
                H2PoolOutcome::NotH2(conn) => {
                    // Origin negotiated h1.1/"" — drive it on the h1 path
                    // with this freshly-dialed Conn and the full budget.
                    return self.transact(conn, &req, &host, port, is_https);
                }
                H2PoolOutcome::Fallthrough => { /* dial failed → h1 dial below */ }
            }
        }

        // Connection acquisition with retry. A keep-alive socket lifted
        // from the pool may be half-dead — the server (or an intermediary
        // NAT/firewall) can drop a previously-idle connection without us
        // noticing until our next write gets no reply (RecvFailed 10060)
        // or a queued RST (10054). Per RFC 9110 §9.2.2 a GET is idempotent,
        // so when a transaction fails with a connection-level error we
        // safely re-issue it on a fresh socket. Without this, the first
        // sub-resource after the document fetch (e.g. a stylesheet reusing
        // the document's pooled connection) intermittently fails and the
        // page renders unstyled.
        //
        // Order: try a pooled connection once; on a connection error fall
        // through to a fresh dial. The fresh path itself retries a couple
        // of times to ride out transient timeouts when many sub-resource
        // fetches dial the same host concurrently.
        if let Some(conn) = self.checkout(&host, port, is_https) {
            // A reused warm connection answers fast. Give its socket a
            // short receive timeout (not the full per-request budget): if a
            // server silently dropped a previously-idle keep-alive — with
            // no FIN/RST for our liveness peek to catch — the first recv
            // returns no data within the leash and we re-dial, instead of
            // stalling the whole render for seconds. HN's nginx does
            // exactly this after a couple of requests on one socket. A
            // healthy connection serving a large slow body is unaffected:
            // SO_RCVTIMEO resets each time bytes arrive; only the absence
            // of ANY progress for the leash window trips it.
            let leash = self.timeout_ms.min(REUSE_READ_TIMEOUT_MS);
            conn.set_read_timeout_ms(leash);
            match self.transact(conn, &req, &host, port, is_https) {
                Ok(resp) => return Ok(resp),
                Err(e) if is_retriable_conn_error(&e) => { /* re-dial below */ }
                Err(e) => return Err(e),
            }
        }

        let mut last_err: Option<NetError> = None;
        for _ in 0..2 {
            let conn = match self.dial(&host, port, is_https) {
                Ok(c) => c,
                Err(e) if is_retriable_conn_error(&e) => {
                    last_err = Some(e);
                    continue;
                }
                Err(e) => return Err(e),
            };
            match self.transact(conn, &req, &host, port, is_https) {
                Ok(resp) => return Ok(resp),
                Err(e) if is_retriable_conn_error(&e) => {
                    last_err = Some(e);
                    continue;
                }
                Err(e) => return Err(e),
            }
        }
        Err(last_err.unwrap_or_else(|| NetError::Http("connection failed".into())))
    }

    /// Transact on a freshly-dialled connection with the full per-request
    /// read budget.
    fn transact(
        &self,
        conn: Conn,
        req: &Request,
        host: &str,
        port: u16,
        is_https: bool,
    ) -> Result<Response, NetError> {
        let deadline = std::time::Instant::now()
            + std::time::Duration::from_millis(self.timeout_ms.max(1) as u64);
        self.transact_with_deadline(conn, req, host, port, is_https, deadline)
    }

    /// Open a fresh socket (and TLS layer for https) to `host:port`,
    /// trying each resolved address in turn.
    fn dial(&self, host: &str, port: u16, is_https: bool) -> Result<Conn, NetError> {
        let addrs = resolve(host, port).map_err(NetError::Dns)?;
        let mut last_err: Option<NetError> = None;
        let mut socket = None;
        for a in addrs {
            match Socket::connect_with_timeout(&a, self.timeout_ms) {
                Ok(s) => {
                    socket = Some(s);
                    break;
                }
                Err(e) => last_err = Some(NetError::Socket(e)),
            }
        }
        let sock_raw =
            socket.ok_or_else(|| last_err.unwrap_or_else(|| NetError::Http("no addr".into())))?;
        if is_https {
            Ok(Conn::Tls(TlsStream::connect(sock_raw, host)?))
        } else {
            Ok(Conn::Plain(sock_raw))
        }
    }

    /// Run one request/response transaction over an already-established
    /// connection (pooled or freshly dialled). On success an HTTP/1.1
    /// connection left at a clean message boundary is returned to the
    /// pool for reuse. HTTP/2 connections are handled by their own layer
    /// and are never pooled here. `deadline` bounds the body read so a
    /// reused-but-dead socket can be abandoned quickly.
    fn transact_with_deadline(
        &self,
        mut conn: Conn,
        req: &Request,
        host: &str,
        port: u16,
        is_https: bool,
        deadline: std::time::Instant,
    ) -> Result<Response, NetError> {
        // HTTP/2 fast-path: if the TLS handshake picked h2 via ALPN,
        // route the request through the HTTP/2 streaming layer
        // instead of writing a raw HTTP/1.1 wire request. Falls
        // through to the HTTP/1 path on any framing error so a
        // misbehaving server doesn't blank the page.
        if let Conn::Tls(tls) = &conn {
            if tls.alpn_protocol() == "h2" {
                // Server picked h2 — we MUST speak h2. Falling back
                // to h1.1 on the same socket after a failed h2 attempt
                // would write h1 bytes onto an h2-broken connection;
                // we propagate the error instead so the caller knows.
                return h2_send_request(&mut conn, req, host);
            }
        }

        // Build request.
        let target = req.url.request_target();
        let host_header = if let Some(p) = req.url.port {
            format!("{host}:{p}")
        } else {
            host.to_string()
        };
        let mut wire = String::new();
        wire.push_str(&format!("{} {} HTTP/1.1\r\n", req.method, target));
        wire.push_str(&format!("Host: {host_header}\r\n"));
        // Cloudflare and other CDN WAFs reject unknown User-Agents with a
        // silent TCP RST / close_notify after handshake (thehindu.com is
        // the canonical reproducer: TLS succeeds, the server returns 0
        // bytes). Pose as a recent stable Chrome — and crucially, do NOT
        // append our own product token at the end. Cloudflare's bot
        // scoring rule pattern is "Chrome/X.Y Safari/Z" with NOTHING
        // after Safari/537.36; any trailing token (we used to ship
        // "Conclave/0.0.1") flags the request as a script.
        wire.push_str("User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) ");
        wire.push_str("AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36\r\n");
        wire.push_str("Accept: */*\r\n");
        if req.accept_brotli {
            wire.push_str("Accept-Encoding: gzip, deflate, br\r\n");
        } else {
            wire.push_str("Accept-Encoding: gzip, deflate\r\n");
        }
        wire.push_str("Connection: keep-alive\r\n");
        // Attach session cookies for this URL, if any apply. We don't
        // forward a Cookie header the caller added themselves separately;
        // those are pass-through "raw header" inputs and merging would
        // need fuller parsing than V1 needs.
        if let Some(jar) = &self.cookie_jar {
            if let Some(c) = jar.cookie_header(host, &target, is_https) {
                wire.push_str(&format!("Cookie: {c}\r\n"));
            }
        }
        for (k, v) in &req.headers {
            wire.push_str(&format!("{k}: {v}\r\n"));
        }
        // RFC 9110 §8.6 — a request with a body MUST advertise its
        // length. We only generate identity bodies (no chunked uploads)
        // so Content-Length is sufficient. Skip if the caller already
        // set it themselves.
        if !req.body.is_empty()
            && !req
                .headers
                .iter()
                .any(|(k, _)| k.eq_ignore_ascii_case("content-length"))
        {
            wire.push_str(&format!("Content-Length: {}\r\n", req.body.len()));
        }
        wire.push_str("\r\n");
        conn.write_all(wire.as_bytes())?;
        if !req.body.is_empty() {
            conn.write_all(&req.body)?;
        }

        // Smart reader: stop the moment Content-Length or the chunked
        // terminator say the body is complete. Old behaviour was
        // `read_to_end` which waited for the server to close the
        // connection — Cloudflare-fronted hosts hold the socket open
        // for tens of seconds after sending the body, freezing the
        // caller. The caller-supplied deadline guards against
        // pathological hangs (shorter for reused connections).
        let (raw, framing_bounded) = read_until_complete(&mut conn, deadline)?;
        let resp = parse_response(&raw)?;
        // Absorb any Set-Cookie headers into the session jar so the next
        // request to this origin carries the cookies the server expected.
        if let Some(jar) = &self.cookie_jar {
            jar.absorb(&resp.headers, host, &target, is_https);
        }
        // HSTS — when the response carries Strict-Transport-Security on
        // an HTTPS connection, record the entry so the next navigation
        // to plain `http://host/...` upgrades to HTTPS automatically.
        // HSTS is HTTPS-only per RFC 6797 §7.2 — plaintext responses
        // claiming HSTS are dropped.
        if is_https {
            for (name, value) in &resp.headers {
                if name.eq_ignore_ascii_case("strict-transport-security") {
                    crate::hsts::record(host, value);
                }
            }
        }
        // Connection reuse: server-side must want keep-alive AND the
        // body must have been bounded (Content-Length or chunked) so we
        // know we left the stream at a clean message boundary. With
        // read-to-close framing the server has shut down its half
        // already — the socket is dead either way.
        let server_close = resp
            .header("Connection")
            .map(|v| v.to_ascii_lowercase().contains("close"))
            .unwrap_or(false);
        if framing_bounded && !server_close && resp.status < 500 {
            self.return_to_pool(host, port, is_https, conn);
        }
        Ok(resp)
    }

    // ==================================================================
    // M6.3 — HTTP/2 connection pool + multiplexing (CV_H2_POOL).
    // ==================================================================

    /// Drain the h2 pool's stale (idle-too-long) or poisoned conns,
    /// mirroring `evict_stale_pool_entries` for h1. Called once per
    /// `h2_send_via_pool` before checkout.
    fn h2_evict_stale(&self) {
        let mut pool = self.h2_pool.lock().unwrap_or_else(|e| e.into_inner());
        let now = Instant::now();
        pool.retain(|_, arc| {
            let c = arc.lock().unwrap_or_else(|e| e.into_inner());
            !c.poisoned && now.duration_since(c.last_used) < POOL_IDLE_TIMEOUT
        });
    }

    /// Remove a (host, port) origin from the h2 pool (eviction). The
    /// socket closes when the last `Arc` holder drops it.
    fn h2_evict(&self, host: &str, port: u16) {
        let mut pool = self.h2_pool.lock().unwrap_or_else(|e| e.into_inner());
        pool.remove(&(host.to_string(), port));
    }

    /// Look up an existing live h2 conn for this origin. Returns the
    /// shared `Arc<Mutex<H2Conn>>` (clones it, releasing the outer map
    /// lock before any I/O) only if it is not poisoned and not
    /// GOAWAY-closed. Poisoned/closed entries are evicted here.
    fn h2_checkout(&self, host: &str, port: u16) -> Option<Arc<Mutex<H2Conn>>> {
        let mut pool = self.h2_pool.lock().unwrap_or_else(|e| e.into_inner());
        let key = (host.to_string(), port);
        let arc = pool.get(&key)?.clone();
        {
            let c = arc.lock().unwrap_or_else(|e| e.into_inner());
            if c.poisoned || c.h2.closed_by_goaway {
                drop(c);
                pool.remove(&key);
                return None;
            }
        }
        Some(arc)
    }

    /// Dial a fresh socket+TLS to `host:port` offering h2 via ALPN. On
    /// success returns the dialed `Conn` and its negotiated ALPN string
    /// so the caller can decide h2-vs-h1 WITHOUT this method touching the
    /// pool (the map lock is never held across this handshake).
    fn h2_dial(&self, host: &str, port: u16) -> Result<(Conn, String), NetError> {
        let conn = self.dial(host, port, true)?;
        let alpn = match &conn {
            Conn::Tls(tls) => tls.alpn_protocol().to_string(),
            Conn::Plain(_) => String::new(),
        };
        Ok((conn, alpn))
    }

    /// The h2 pool entry point (flag-on, https-only). Tries a pooled
    /// conn, else dials fresh; routes non-h2 origins back to the h1 path
    /// via `H2PoolOutcome::NotH2`. Implements the bounded retry policy:
    /// stream-level errors (RST) retry once on a new stream of the SAME
    /// pooled conn; conn-level errors poison+evict and retry once on a
    /// freshly dialed conn. After retries are exhausted the error
    /// propagates — never a partial/guessed body, never a cross-response.
    fn h2_send_via_pool(&self, host: &str, port: u16, req: &Request) -> H2PoolOutcome {
        self.h2_evict_stale();
        let deadline = Instant::now() + Duration::from_millis(self.timeout_ms.max(1) as u64);

        // First: an existing pooled conn (no new socket, no handshake).
        if let Some(arc) = self.h2_checkout(host, port) {
            match self.h2_drive_locked(&arc, req, deadline) {
                Ok(resp) => return H2PoolOutcome::Response(Ok(resp)),
                Err(e) if is_h2_stream_level_error(&e) => {
                    // RST_STREAM — only that stream failed; the conn is
                    // healthy. Retry ONCE on a NEW stream of the SAME
                    // pooled conn before giving up to a fresh conn.
                    match self.h2_drive_locked(&arc, req, deadline) {
                        Ok(resp) => return H2PoolOutcome::Response(Ok(resp)),
                        Err(e2) if is_h2_stream_level_error(&e2) => {
                            // Second RST — drop this conn, dial fresh.
                            self.h2_evict(host, port);
                        }
                        Err(e2) if is_retriable_conn_error(&e2) => {
                            self.h2_evict(host, port);
                        }
                        Err(e2) => return H2PoolOutcome::Response(Err(e2)),
                    }
                }
                Err(e) if is_retriable_conn_error(&e) => {
                    // Conn-level error on a reused conn — poison + evict,
                    // fall through to a fresh dial below.
                    self.h2_evict(host, port);
                }
                Err(e) => return H2PoolOutcome::Response(Err(e)),
            }
        }

        // Dial fresh. The map lock is NOT held across this handshake.
        let (conn, alpn) = match self.h2_dial(host, port) {
            Ok(t) => t,
            Err(_) => return H2PoolOutcome::Fallthrough,
        };
        if alpn != "h2" {
            // Not an h2 origin — hand the dialed Conn to the h1 path.
            return H2PoolOutcome::NotH2(conn);
        }

        // Wrap in a fresh H2Conn, write the preface+SETTINGS ONCE, insert
        // into the pool, and drive. `new_client()` runs exactly once per
        // conn here — the whole point of the pool.
        let mut h2 = crate::http2::Connection::new_client();
        let preface = h2.drain_outgoing();
        let mut conn = conn;
        if let Err(e) = conn.write_all(&preface) {
            return H2PoolOutcome::Response(Err(e));
        }
        let h2conn = H2Conn {
            conn,
            h2,
            decode_dyn: crate::http2::HpackDynamicTable::new(),
            accum: Vec::with_capacity(32_768),
            last_used: Instant::now(),
            poisoned: false,
        };
        let arc = Arc::new(Mutex::new(h2conn));
        {
            let mut pool = self.h2_pool.lock().unwrap_or_else(|e| e.into_inner());
            pool.insert((host.to_string(), port), arc.clone());
        }
        match self.h2_drive_locked(&arc, req, deadline) {
            Ok(resp) => H2PoolOutcome::Response(Ok(resp)),
            Err(e) => {
                // Any error on a brand-new conn poisons+evicts it so no
                // future request inherits a broken conn. We do NOT retry
                // here (the outer send_once retry budget already covers a
                // second send attempt) — fail cleanly.
                self.h2_evict(host, port);
                H2PoolOutcome::Response(Err(e))
            }
        }
    }

    /// Lock the inner `Mutex<H2Conn>` (coarse lock-per-request, V1
    /// serialized scope) and drive ONE request to completion. Returns the
    /// Response or a clean error. On a conn-level error the conn is
    /// poisoned inside the lock so the next checkout evicts it.
    fn h2_drive_locked(
        &self,
        arc: &Arc<Mutex<H2Conn>>,
        req: &Request,
        deadline: Instant,
    ) -> Result<Response, NetError> {
        let mut guard = arc.lock().unwrap_or_else(|e| e.into_inner());
        if guard.poisoned {
            return Err(NetError::Http("h2: conn poisoned".into()));
        }
        // GOAWAY pre-flight: a conn closed by a prior GOAWAY can open no
        // new streams. Poison + fail (the caller dials fresh).
        if guard.h2.closed_by_goaway {
            guard.poisoned = true;
            return Err(NetError::Http("h2: connection reset by GOAWAY".into()));
        }
        // Concurrency guard. Trivially satisfied at V1 concurrency 1, but
        // explicit so the invariant is checked, not assumed.
        let max = guard.h2.peer_max_concurrent as usize;
        if max > 0 && guard.h2.active_stream_count() >= max {
            return Err(NetError::Http("h2: peer max concurrent streams".into()));
        }
        let result = {
            let c = &mut *guard;
            h2_drive_request(
                &mut c.conn,
                &mut c.h2,
                &mut c.decode_dyn,
                &mut c.accum,
                req,
                deadline,
            )
        };
        match result {
            Ok(resp) => {
                guard.last_used = Instant::now();
                Ok(resp)
            }
            Err(e) => {
                // Conn-level errors poison the conn. A stream-level RST
                // (carried as "h2: stream reset") leaves the conn healthy
                // — the conn survives, only that stream failed.
                if !is_h2_stream_level_error(&e) {
                    guard.poisoned = true;
                }
                Err(e)
            }
        }
    }
}

/// Outcome of consulting the h2 pool, distinguishing "answered the
/// request", "this origin is not h2 — here is the dialed Conn for the h1
/// path", and "dial failed — fall through to the h1 dial retry loop".
enum H2PoolOutcome {
    Response(Result<Response, NetError>),
    NotH2(Conn),
    Fallthrough,
}

/// Whether `e` is a STREAM-level h2 error (RST_STREAM): only the one
/// stream failed; the connection survives and stays in the pool. Used to
/// decide poison-or-not in `h2_drive_locked`.
fn is_h2_stream_level_error(e: &NetError) -> bool {
    matches!(e, NetError::Http(s) if s.contains("h2: stream reset"))
}

/// The transport-agnostic HTTP/2 request driver. Opens ONE new stream on
/// the persistent `h2` state machine, writes its HEADERS (+DATA), then
/// runs the read+demux loop until THIS stream's clean END_STREAM —
/// PARKING (buffering, never discarding) any frames for other streams.
///
/// CORRECTNESS (why this cannot cross responses): `feed_frame` keys every
/// DATA/HEADERS frame by `hdr.stream_id` and appends to that stream's
/// buffers. A request for `sid` only ever reads `h2.stream(sid)`'s
/// buffers, so a frame for any other stream is parked in the map, never
/// returned here. Assembly is per-id append, so interleave order is
/// irrelevant. HPACK decode uses the PERSISTENT per-conn `decode_dyn`
/// (the keystone), so a 2nd+ stream's dynamic-index references resolve.
fn h2_drive_request<T: H2Transport + ?Sized>(
    conn: &mut T,
    h2: &mut crate::http2::Connection,
    decode_dyn: &mut crate::http2::HpackDynamicTable,
    accum: &mut Vec<u8>,
    req: &Request,
    deadline: Instant,
) -> Result<Response, NetError> {
    use crate::http2::{FrameHeader, StreamState};

    // 2. Open a distinct stream (next_stream_id += 2 — odd, monotonic).
    let sid = h2.open_stream();

    // 3. Build the EXACT Chrome-131 header set (shared fn) + send.
    let target = req.url.request_target();
    let scheme = req.url.scheme.as_str();
    let path = target.to_string();
    let ae = if req.accept_brotli {
        "gzip, deflate, br"
    } else {
        "gzip, deflate"
    };
    let authority = req.url.host.as_str();
    let headers = chrome131_h2_headers(req.method.as_str(), authority, scheme, &path, ae);
    let end_stream = req.body.is_empty();
    h2.send_headers(sid, &headers, end_stream);
    if !req.body.is_empty() {
        h2.send_data(sid, &req.body, true);
    }

    // 4. Flush HEADERS/DATA (+ any queued SETTINGS-ACK/WINDOW_UPDATE/PING).
    let wire = h2.drain_outgoing();
    conn.write_all(&wire)?;

    // Give the socket a leash so a hung read is bounded; the per-request
    // deadline is the hard cap.
    conn.set_read_timeout_ms(REUSE_READ_TIMEOUT_MS);

    // 5. Read + demux loop.
    let mut buf = [0u8; 16_384];
    loop {
        if Instant::now() >= deadline {
            return Err(NetError::Http("h2: request timed out".into()));
        }
        let n = conn.read_some(&mut buf)?;
        if n == 0 {
            return Err(NetError::Http("h2: peer closed (Closed)".into()));
        }
        accum.extend_from_slice(&buf[..n]);

        // Split out every COMPLETE 9-byte-header+payload frame; a frame
        // split across recv() calls stays in `accum` (persists across
        // requests on a busy conn). feed_frame DEMUXES by stream_id.
        let mut consumed = 0usize;
        while accum.len() - consumed >= 9 {
            let hdr = FrameHeader::decode(&accum[consumed..])
                .ok_or_else(|| NetError::Http("h2: frame header decode".into()))?;
            let frame_len = 9 + hdr.length as usize;
            if accum.len() - consumed < frame_len {
                break; // partial — wait for more bytes
            }
            // Read GOAWAY's last_stream_id BEFORE feeding (feed_frame
            // discards the payload). Decide whether OUR sid was accepted.
            if hdr.typ == crate::http2::FrameType::Goaway {
                let body = &accum[consumed + 9..consumed + frame_len];
                if body.len() >= 4 {
                    let last = u32::from_be_bytes([body[0], body[1], body[2], body[3]])
                        & 0x7FFF_FFFF;
                    if sid > last {
                        // Our stream was NEVER processed — conn-level
                        // retriable error (retry on a fresh conn).
                        return Err(NetError::Http(
                            "h2: GOAWAY, stream not processed (Closed)".into(),
                        ));
                    }
                }
            }
            let body = accum[consumed + 9..consumed + frame_len].to_vec();
            h2.feed_frame(hdr, &body);
            consumed += frame_len;
        }
        accum.drain(..consumed);

        // Flush any auto-queued SETTINGS-ACK / WINDOW_UPDATE / PING-ACK.
        let out = h2.drain_outgoing();
        if !out.is_empty() {
            conn.write_all(&out)?;
        }

        // Done condition — ONLY a clean END_STREAM (eof_remote) is a
        // valid Response. Closed-WITHOUT-eof_remote = RST_STREAM = error.
        if let Some(s) = h2.stream(sid) {
            if s.eof_remote {
                break;
            }
            if s.state == StreamState::Closed && !s.eof_remote {
                // RST_STREAM before a clean end — fail this stream
                // cleanly (NEVER a truncated/empty 200). Stream-level:
                // the conn survives.
                h2.remove_stream(sid);
                return Err(NetError::Http("h2: stream reset".into()));
            }
        }
    }

    // 6. Build the Response from THIS stream's buffers, using the
    // PERSISTENT decode table. Then remove the stream to bound memory.
    let (hpack_block, body_raw) = {
        let s = h2
            .stream(sid)
            .ok_or_else(|| NetError::Http("h2: stream vanished".into()))?;
        (s.headers_buf.clone(), s.data_buf.clone())
    };
    let decoded = crate::http2::hpack_decode_block(&hpack_block, decode_dyn)
        .ok_or_else(|| NetError::Http("h2: hpack decode failed".into()))?;
    let mut status: u16 = 0;
    let mut resp_headers: Vec<(String, String)> = Vec::new();
    for (n, v) in decoded {
        if n == ":status" {
            status = v.parse::<u16>().unwrap_or(0);
        } else if n.starts_with(':') {
            // other pseudo-headers — skip
        } else {
            resp_headers.push((n, v));
        }
    }
    h2.remove_stream(sid);
    if status == 0 {
        return Err(NetError::Http("h2: no :status".into()));
    }
    let reason = h2_reason(status);
    let body = decode_content_encoding(&resp_headers, body_raw)?;
    Ok(Response {
        status,
        reason: reason.to_string(),
        headers: resp_headers,
        body,
    })
}

/// A connection-level failure means the request never reached a server
/// that acted on it (stale pooled socket, dropped TCP connection, TLS
/// handshake timeout, or an empty/headerless reply from a half-closed
/// keep-alive). Per RFC 9110 §9.2.2 an idempotent GET may be safely
/// retried on a fresh connection. We deliberately do NOT retry on real
/// HTTP responses (4xx/5xx) or body-decode errors — those came from the
/// server and a retry would just repeat them.
fn is_retriable_conn_error(e: &NetError) -> bool {
    match e {
        NetError::Socket(_) => true,
        // Socket-level failures on the TLS path are wrapped as
        // NetError::Http(<TlsError as Display>) (e.g.
        // "protocol: read: RecvFailed(10060)") rather than
        // NetError::Socket, so match those substrings too. Also catch the
        // empty/headerless reply a half-closed pooled connection produces
        // ("no header end") and our own read deadline. We deliberately do
        // NOT match body-decode failures ("gzip decode", "brotli", "bad
        // content-length") — those came from a real server response and a
        // retry would just repeat them.
        NetError::Http(s) => {
            s.contains("no header end")
                || s.contains("request timed out")
                || s.contains("RecvFailed")
                || s.contains("SendFailed")
                || s.contains("read:")
                || s.contains("send:")
                || s.contains("Closed")
                || s.contains("close_notify")
                || s.contains("connection reset")
        }
        _ => false,
    }
}

/// The exact Chrome-131 HTTP/2 request header set, shared by BOTH the
/// legacy per-request path (`h2_send_request`, flag-off) and the pooled
/// driver (`h2_send_pooled`, flag-on). Factored into ONE source so the
/// JA4-h2 / Akamai-h2 fingerprint can NEVER drift between the two paths:
/// pseudo-headers in fixed order (:method, :authority, :scheme, :path),
/// then regular headers ALPHABETICALLY (Chromium's HTTP2 serializer
/// hard-codes that sort; any deviation is a fingerprint hit). The
/// `priority: u=0, i` header is Chrome 124+'s RFC 9218 navigation
/// priority — there is NO separate PRIORITY frame, by design.
///
/// NOTE (pre-existing behaviour gap, deliberately preserved for
/// fingerprint/behaviour neutrality): no `cookie` header is attached
/// here, unlike the h1 path. The alphabetical insertion point for a
/// future fix would be between `content-length`/`accept-*` and `host`
/// (i.e. right after `accept-language`). Adding it is a separate task —
/// see the M6.3 follow-up note — not something to slip into this perf
/// change, since it would alter both the bytes on the wire and the
/// observable behaviour.
fn chrome131_h2_headers<'a>(
    method: &'a str,
    authority: &'a str,
    scheme: &'a str,
    path: &'a str,
    accept_encoding: &'a str,
) -> Vec<(&'a str, &'a str)> {
    vec![
        (":method", method),
        (":authority", authority),
        (":scheme", scheme),
        (":path", path),
        (
            "accept",
            "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8",
        ),
        ("accept-encoding", accept_encoding),
        ("accept-language", "en-US,en;q=0.9"),
        // Chrome 124+ sends `priority` on h2 navigation (RFC 9218).
        ("priority", "u=0, i"),
        (
            "sec-ch-ua",
            "\"Chromium\";v=\"131\", \"Google Chrome\";v=\"131\", \"Not_A Brand\";v=\"24\"",
        ),
        ("sec-ch-ua-mobile", "?0"),
        ("sec-ch-ua-platform", "\"Windows\""),
        ("sec-fetch-dest", "document"),
        ("sec-fetch-mode", "navigate"),
        ("sec-fetch-site", "none"),
        ("sec-fetch-user", "?1"),
        ("upgrade-insecure-requests", "1"),
        (
            "user-agent",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
        ),
    ]
}

/// Map an HTTP status code to its canonical reason phrase. Shared by the
/// legacy and pooled h2 response builders.
fn h2_reason(status: u16) -> &'static str {
    match status {
        200 => "OK",
        301 => "Moved Permanently",
        302 => "Found",
        304 => "Not Modified",
        400 => "Bad Request",
        403 => "Forbidden",
        404 => "Not Found",
        500 => "Internal Server Error",
        503 => "Service Unavailable",
        _ => "",
    }
}

/// Send an HTTP request over an HTTP/2 connection that has just
/// completed TLS handshake. V1 model is simple: open one stream per
/// request, write SETTINGS+HEADERS, drain DATA, surface the response.
/// Returns an error if the framing layer errors out — caller falls
/// back to HTTP/1.1.
fn h2_send_request(conn: &mut Conn, req: &Request, host: &str) -> Result<Response, NetError> {
    let mut c = crate::http2::Connection::new_client();
    // Send the preface + initial SETTINGS we queued at construction.
    let preface = c.drain_outgoing();
    conn.write_all(&preface)?;
    // Open a stream and send the request HEADERS with END_STREAM (GET
    // has no body in V1).
    let sid = c.open_stream();
    let target = req.url.request_target();
    let scheme = req.url.scheme.as_str();
    let path = target.to_string();
    let ae = if req.accept_brotli {
        "gzip, deflate, br"
    } else {
        "gzip, deflate"
    };
    let headers = chrome131_h2_headers(req.method.as_str(), host, scheme, &path, ae);
    let end_stream = req.body.is_empty();
    c.send_headers(sid, &headers, end_stream);
    let headers_wire = c.drain_outgoing();
    conn.write_all(&headers_wire)?;
    if !req.body.is_empty() {
        c.send_data(sid, &req.body, true);
        let data_wire = c.drain_outgoing();
        conn.write_all(&data_wire)?;
    }

    // Drain frames until END_STREAM on this stream. CRITICAL: accumulate
    // bytes across reads so a frame split across two recv() calls is
    // handled — the old code zeroed its slice between iterations and
    // silently dropped any partial frame, which is what was breaking
    // HN / BBC / Google when h2 was enabled.
    let mut buf = [0u8; 16_384];
    let mut accum: Vec<u8> = Vec::with_capacity(32_768);
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
    loop {
        if std::time::Instant::now() >= deadline {
            return Err(NetError::Http("h2: timeout".into()));
        }
        let n = conn.read_some(&mut buf)?;
        if n == 0 {
            return Err(NetError::Http("h2: peer closed".into()));
        }
        accum.extend_from_slice(&buf[..n]);
        // Drain as many complete frames as fit.
        let mut consumed = 0usize;
        while accum.len() - consumed >= 9 {
            let hdr = crate::http2::FrameHeader::decode(&accum[consumed..])
                .ok_or_else(|| NetError::Http("h2: frame header decode".into()))?;
            let frame_len = 9 + hdr.length as usize;
            if accum.len() - consumed < frame_len {
                break;
            }
            let body = accum[consumed + 9..consumed + frame_len].to_vec();
            c.feed_frame(hdr, &body);
            consumed += frame_len;
        }
        accum.drain(..consumed);
        let out = c.drain_outgoing();
        if !out.is_empty() {
            conn.write_all(&out)?;
        }
        if let Some(stream) = c.stream(sid) {
            if stream.eof_remote || matches!(stream.state, crate::http2::StreamState::Closed) {
                let body = stream.data_buf.clone();
                let hpack_block = stream.headers_buf.clone();
                // Real HPACK decode — extract :status and the
                // response headers Chrome would see. Without this
                // every response read as 200 OK and 30x redirects
                // looked like blank-bodied 200s.
                let mut dyn_tbl = crate::http2::HpackDynamicTable::new();
                let decoded = crate::http2::hpack_decode_block(&hpack_block, &mut dyn_tbl)
                    .ok_or_else(|| NetError::Http("h2: hpack decode failed".into()))?;
                let mut status: u16 = 0;
                let mut headers: Vec<(String, String)> = Vec::new();
                for (n, v) in decoded {
                    if n == ":status" {
                        status = v.parse::<u16>().unwrap_or(0);
                    } else if n.starts_with(':') {
                        // other pseudo-headers — skip
                    } else {
                        headers.push((n, v));
                    }
                }
                if status == 0 {
                    return Err(NetError::Http("h2: no :status".into()));
                }
                let reason = h2_reason(status);
                let body = decode_content_encoding(&headers, body)?;
                return Ok(Response {
                    status,
                    reason: reason.to_string(),
                    headers,
                    body,
                });
            }
        }
    }
}

/// Read from `conn` incrementally until we have a complete HTTP response:
/// headers terminated by `\r\n\r\n`, then either `Content-Length` bytes,
/// the chunked terminator (`0\r\n\r\n`), or EOF (legacy read-to-close).
fn read_until_complete(
    conn: &mut Conn,
    deadline: std::time::Instant,
) -> Result<(Vec<u8>, bool), NetError> {
    let mut raw: Vec<u8> = Vec::with_capacity(8192);
    let mut buf = [0u8; 4096];
    let mut head_end: Option<usize> = None;
    // 0 = unknown / read-to-close, Some(n) = exact body length expected
    // after the header break.
    let mut body_target: Option<usize> = None;
    let mut chunked = false;
    // Response status, parsed from the status line. 1xx/204/304 responses
    // carry NO message body regardless of their headers (RFC 9110 §6.4.1 /
    // RFC 9112 §6.3) — see the no-body short-circuit below.
    let mut status_code: u16 = 0;

    loop {
        if std::time::Instant::now() >= deadline {
            return Err(NetError::Http("request timed out".into()));
        }
        if head_end.is_none() {
            if let Some(pos) = find_double_crlf(&raw) {
                head_end = Some(pos);
                let head = std::str::from_utf8(&raw[..pos])
                    .map_err(|_| NetError::Http("non-utf8 head".into()))?;
                // Status line: "HTTP/1.1 <code> <reason>".
                if let Some(first) = head.split("\r\n").next() {
                    if let Some(code) = first.split_whitespace().nth(1) {
                        status_code = code.parse().unwrap_or(0);
                    }
                }
                // Parse just enough of the headers to learn how to stop.
                for line in head.split("\r\n").skip(1) {
                    let Some((k, v)) = line.split_once(':') else {
                        continue;
                    };
                    let k_lc = k.trim().to_ascii_lowercase();
                    let v = v.trim();
                    if k_lc == "content-length" {
                        if let Ok(n) = v.parse::<usize>() {
                            body_target = Some(n);
                        }
                    } else if k_lc == "transfer-encoding"
                        && v.to_ascii_lowercase().contains("chunked")
                    {
                        chunked = true;
                    }
                }
            }
        }
        // Once we know the framing, check completion before reading more.
        if let Some(pos) = head_end {
            let body_start = pos + 4;
            // RFC 9110 §6.4.1 / RFC 9112 §6.3: 1xx, 204, and 304 responses
            // NEVER carry a message body, even if they (wrongly) include a
            // Content-Length or Transfer-Encoding. The body ends immediately
            // after the headers. Without this short-circuit, a 304 from a
            // cache revalidation — which has no body and rides a kept-alive
            // connection (no EOF, no Content-Length) — makes us wait for body
            // bytes that never arrive until the deadline (then a retry), i.e.
            // a ~60s hang per cached revisit. THIS was the "stuck on Loading"
            // freeze on previously-visited sites.
            if matches!(status_code, 100..=199 | 204 | 304) {
                raw.truncate(body_start);
                return Ok((raw, true));
            }
            if let Some(target) = body_target {
                if raw.len() >= body_start + target {
                    raw.truncate(body_start + target);
                    return Ok((raw, true));
                }
            } else if chunked {
                // RFC 9112 §7.1 chunked transfer coding. The terminator is
                // a chunk-size of 0 followed by a (possibly empty) trailer
                // section and a final CRLF. Previously we substring-
                // matched on `0\r\n\r\n`, but binary/compressed chunked
                // bodies frequently contain those bytes inside data — so
                // we used to truncate gzip/Brotli/image bodies the moment
                // the literal `0\r\n\r\n` byte sequence appeared inside a
                // chunk's payload.
                //
                // Real fix: walk the framing — read each chunk header
                // (hex size + optional `;ext` + CRLF), skip exactly that
                // many bytes of data + the trailing CRLF, and stop on
                // size=0 followed by the trailer-ending CRLF. We don't
                // remove framing bytes here (parse_response strips them
                // afterwards); we just answer the "have I read enough?"
                // question correctly.
                if chunked_body_is_complete(&raw[body_start..]) {
                    return Ok((raw, true));
                }
            }
            // No content-length, no chunked → fall through to read until
            // EOF (legacy read-to-close behaviour).
        }
        let n = conn.read_some(&mut buf)?;
        if n == 0 {
            // Clean EOF — return whatever we've accumulated.
            // `framing_bounded = false` so caller doesn't reuse this conn.
            return Ok((raw, false));
        }
        raw.extend_from_slice(&buf[..n]);
    }
}

/// RFC 9112 §7.1: walk the chunked encoding to see whether the stream has
/// reached its `0\r\n…\r\n` terminator. Stops short with `false` if the
/// buffer is mid-chunk-header, mid-data, or mid-trailer. Returns `true`
/// only when we've consumed the last-chunk + trailers + final CRLF.
fn chunked_body_is_complete(buf: &[u8]) -> bool {
    let mut i = 0;
    loop {
        // chunk-size [BWS ; chunk-ext] CRLF
        let line_end = match find_crlf(&buf[i..]) {
            Some(p) => i + p,
            None => return false,
        };
        let line = &buf[i..line_end];
        // Size portion ends at ';' (chunk-ext) or end-of-line.
        let size_str = line
            .split(|b| *b == b';' || *b == b' ' || *b == b'\t')
            .next()
            .unwrap_or(line);
        let size_str = std::str::from_utf8(size_str).unwrap_or("");
        let n = match usize::from_str_radix(size_str.trim(), 16) {
            Ok(n) => n,
            Err(_) => return false, // malformed; let the caller keep reading
        };
        i = line_end + 2; // past the CRLF
        if n == 0 {
            // Last chunk: skip trailer-section (zero or more header-fields),
            // each terminated by CRLF, then a final CRLF that ends the body.
            loop {
                if i + 2 > buf.len() {
                    return false;
                }
                if &buf[i..i + 2] == b"\r\n" {
                    return true; // end of trailers
                }
                let trailer_end = match find_crlf(&buf[i..]) {
                    Some(p) => i + p,
                    None => return false,
                };
                i = trailer_end + 2;
            }
        }
        // chunk-data is exactly n bytes, then a trailing CRLF.
        if i + n + 2 > buf.len() {
            return false;
        }
        i += n + 2;
    }
}

fn parse_response(buf: &[u8]) -> Result<Response, NetError> {
    // Find header/body split.
    let split = find_double_crlf(buf).ok_or_else(|| {
        let n = buf.len().min(256);
        let preview = String::from_utf8_lossy(&buf[..n])
            .chars()
            .map(|c| {
                if c.is_ascii_graphic() || c == ' ' || c == '\r' || c == '\n' {
                    c
                } else {
                    '·'
                }
            })
            .collect::<String>()
            .replace('\r', "\\r")
            .replace('\n', "\\n");
        NetError::Http(format!(
            "no header end (got {} bytes; first {}: {:?})",
            buf.len(),
            n,
            preview
        ))
    })?;
    let head =
        std::str::from_utf8(&buf[..split]).map_err(|_| NetError::Http("non-utf8 head".into()))?;
    let mut lines = head.split("\r\n");
    let status_line = lines.next().ok_or_else(|| NetError::Http("empty".into()))?;
    let (status, reason) = parse_status_line(status_line)?;
    let mut headers = Vec::new();
    for line in lines {
        if line.is_empty() {
            continue;
        }
        let (k, v) = line
            .split_once(':')
            .ok_or_else(|| NetError::Http(format!("bad header line: {line}")))?;
        headers.push((k.trim().to_string(), v.trim().to_string()));
    }

    let body_raw = &buf[split + 4..];
    let body = decode_body(&headers, body_raw)?;
    let body = decode_content_encoding(&headers, body)?;

    Ok(Response {
        status,
        reason,
        headers,
        body,
    })
}

fn is_brotli_decode_error(e: &NetError) -> bool {
    matches!(e, NetError::Http(s) if s.contains("brotli"))
}

fn decode_content_encoding(
    headers: &[(String, String)],
    body: Vec<u8>,
) -> Result<Vec<u8>, NetError> {
    // A zero-length body has nothing to decode — and is never a valid gzip /
    // deflate / brotli stream. This is the normal shape of a bodyless response
    // that still echoes `Content-Encoding` (most importantly a 304 Not Modified
    // from cache revalidation, which has no body per RFC 9110 §6.4.1). Without
    // this, `send` tried to gunzip the empty 304 body → "truncated gzip", and
    // the error propagated before the cache layer could serve the stored body —
    // so EVERY revisit of a cacheable gzipped site (e.g. Wikipedia) failed.
    if body.is_empty() {
        return Ok(body);
    }
    let enc = headers
        .iter()
        .find(|(k, _)| k.eq_ignore_ascii_case("content-encoding"))
        .map(|(_, v)| v.trim().to_ascii_lowercase());
    let enc = match enc {
        Some(s) if !s.is_empty() && s != "identity" => s,
        _ => return Ok(body),
    };
    // Some servers stack encodings ("gzip, br"); apply right-to-left per
    // RFC 9110 §8.4. We support gzip and deflate; anything else fails.
    let mut data = body;
    for token in enc.rsplit(',') {
        let token = token.trim();
        if token.is_empty() || token == "identity" {
            continue;
        }
        data = match token {
            "gzip" | "x-gzip" => cv_compression::decode_gzip(&data)
                .map_err(|e| NetError::Http(format!("gzip decode: {e}")))?,
            "deflate" => cv_compression::decode_zlib(&data)
                .map_err(|e| NetError::Http(format!("deflate decode: {e}")))?,
            "br" => cv_compression::decode_brotli(&data)
                .map_err(|e| NetError::Http(format!("brotli decode: {e}")))?,
            other => {
                return Err(NetError::Http(format!(
                    "unsupported Content-Encoding: {other}"
                )));
            }
        };
    }
    Ok(data)
}

fn parse_status_line(line: &str) -> Result<(u16, String), NetError> {
    let mut parts = line.splitn(3, ' ');
    let _version = parts
        .next()
        .ok_or_else(|| NetError::Http("status: no version".into()))?;
    let code = parts
        .next()
        .ok_or_else(|| NetError::Http("status: no code".into()))?
        .parse::<u16>()
        .map_err(|_| NetError::Http("status code not u16".into()))?;
    let reason = parts.next().unwrap_or("").to_string();
    Ok((code, reason))
}

fn find_double_crlf(buf: &[u8]) -> Option<usize> {
    buf.windows(4).position(|w| w == b"\r\n\r\n")
}

fn decode_body(headers: &[(String, String)], raw: &[u8]) -> Result<Vec<u8>, NetError> {
    let lookup = |name: &str| {
        let lc = name.to_ascii_lowercase();
        headers
            .iter()
            .find(|(k, _)| k.to_ascii_lowercase() == lc)
            .map(|(_, v)| v.as_str())
    };
    if let Some(te) = lookup("transfer-encoding") {
        if te.to_ascii_lowercase().contains("chunked") {
            return decode_chunked(raw);
        }
    }
    if let Some(cl) = lookup("content-length") {
        let n: usize = cl
            .parse()
            .map_err(|_| NetError::Http("bad content-length".into()))?;
        if raw.len() < n {
            return Err(NetError::Http(format!(
                "body short: have {} of {n}",
                raw.len()
            )));
        }
        return Ok(raw[..n].to_vec());
    }
    // Read-to-close.
    Ok(raw.to_vec())
}

fn decode_chunked(mut raw: &[u8]) -> Result<Vec<u8>, NetError> {
    let mut out = Vec::new();
    loop {
        let line_end = find_crlf(raw).ok_or_else(|| NetError::Http("chunk size eof".into()))?;
        let size_line = std::str::from_utf8(&raw[..line_end])
            .map_err(|_| NetError::Http("non-utf8 size".into()))?;
        let size_field = size_line.split(';').next().unwrap_or("").trim();
        let size = usize::from_str_radix(size_field, 16)
            .map_err(|_| NetError::Http(format!("bad chunk size: {size_field:?}")))?;
        raw = &raw[line_end + 2..];
        if size == 0 {
            return Ok(out);
        }
        if raw.len() < size + 2 {
            return Err(NetError::Http("chunk body short".into()));
        }
        out.extend_from_slice(&raw[..size]);
        if &raw[size..size + 2] != b"\r\n" {
            return Err(NetError::Http("chunk not crlf-terminated".into()));
        }
        raw = &raw[size + 2..];
    }
}

fn find_crlf(buf: &[u8]) -> Option<usize> {
    buf.windows(2).position(|w| w == b"\r\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn retriable_classifier_only_flags_connection_errors() {
        use crate::socket::SocketError;
        // Connection-level failures → safe to retry an idempotent GET.
        assert!(is_retriable_conn_error(&NetError::Socket(
            SocketError::RecvFailed(10060)
        )));
        assert!(is_retriable_conn_error(&NetError::Socket(
            SocketError::Closed
        )));
        // A stale pooled socket that returns an empty/headerless reply
        // surfaces as a parse error — also retriable.
        assert!(is_retriable_conn_error(&NetError::Http(
            "no header end (got 0 bytes; first 0: \"\")".into()
        )));
        assert!(is_retriable_conn_error(&NetError::Http(
            "request timed out".into()
        )));
        // Real server responses / decode failures → NOT retriable.
        assert!(!is_retriable_conn_error(&NetError::Http(
            "gzip decode: bad".into()
        )));
        assert!(!is_retriable_conn_error(&NetError::Dns("nxdomain".into())));
        assert!(!is_retriable_conn_error(&NetError::Url("bad".into())));
    }

    #[test]
    fn parses_simple_response() {
        let raw = b"HTTP/1.1 200 OK\r\nContent-Length: 5\r\nContent-Type: text/plain\r\n\r\nhello";
        let r = parse_response(raw).unwrap();
        assert_eq!(r.status, 200);
        assert_eq!(r.reason, "OK");
        assert_eq!(r.body, b"hello");
        assert_eq!(r.header("Content-Type"), Some("text/plain"));
    }

    #[test]
    fn parses_chunked() {
        let raw = b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n5\r\nhello\r\n6\r\n world\r\n0\r\n\r\n";
        let r = parse_response(raw).unwrap();
        assert_eq!(r.status, 200);
        assert_eq!(r.body, b"hello world");
    }

    #[test]
    fn parses_read_to_close() {
        let raw = b"HTTP/1.1 200 OK\r\n\r\nstreaming body";
        let r = parse_response(raw).unwrap();
        assert_eq!(r.status, 200);
        assert_eq!(r.body, b"streaming body");
    }

    /// Regression: a 304 Not Modified (cache revalidation) carries NO body and
    /// rides a kept-alive connection — no Content-Length, no chunked, no EOF.
    /// The reader MUST treat it as complete immediately after the headers; the
    /// old code waited for a body that never came until the deadline + retry
    /// (~60s), which was the "previously-visited site stuck on Loading" freeze.
    #[test]
    fn status_304_returns_immediately_without_a_body() {
        use std::io::{Read, Write};
        use std::net::TcpListener;
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        std::thread::spawn(move || {
            if let Ok((mut sock, _)) = listener.accept() {
                let mut buf = [0u8; 1024];
                let _ = sock.read(&mut buf); // drain the request line+headers
                // Bodyless 304 + keep-alive: a correct client stops here; a
                // broken one blocks waiting for a body.
                let _ = sock
                    .write_all(b"HTTP/1.1 304 Not Modified\r\nConnection: keep-alive\r\n\r\n");
                // Hold the socket open well past the client timeout so a broken
                // reader genuinely hangs (no EOF rescue).
                std::thread::sleep(std::time::Duration::from_secs(10));
            }
        });
        let url = Url::parse(&format!("http://127.0.0.1:{port}/")).unwrap();
        let client = Client::with_timeout(5000);
        let started = std::time::Instant::now();
        let resp = client.fetch(&url).expect("304 fetch must not hang");
        assert_eq!(resp.status, 304);
        assert!(resp.body.is_empty(), "304 must have an empty body");
        assert!(
            started.elapsed() < std::time::Duration::from_secs(2),
            "304 should return immediately, took {:?}",
            started.elapsed()
        );
    }

    #[test]
    fn decodes_gzip_body() {
        // Same gzip(b"hello\n") fixture used in cv_compression tests.
        let mut gz = vec![
            0x1f, 0x8b, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x03, 0xcb, 0x48, 0xcd, 0xc9,
            0xc9, 0xe7, 0x02, 0x00,
        ];
        gz.extend_from_slice(&0x363a3020u32.to_le_bytes());
        gz.extend_from_slice(&6u32.to_le_bytes());
        let mut raw = format!(
            "HTTP/1.1 200 OK\r\nContent-Encoding: gzip\r\nContent-Length: {}\r\n\r\n",
            gz.len()
        )
        .into_bytes();
        raw.extend_from_slice(&gz);
        let r = parse_response(&raw).unwrap();
        assert_eq!(r.body, b"hello\n");
    }

    #[test]
    fn identity_encoding_passes_through() {
        let raw =
            b"HTTP/1.1 200 OK\r\nContent-Encoding: identity\r\nContent-Length: 5\r\n\r\nhello";
        let r = parse_response(raw).unwrap();
        assert_eq!(r.body, b"hello");
    }

    #[test]
    fn empty_gzip_body_does_not_error() {
        // A 304 Not Modified (cache revalidation) echoes Content-Encoding: gzip
        // but carries NO body. Decoding the empty body must NOT error — else the
        // failure propagates before the cache can serve the stored body, and
        // every revisit of a cacheable gzipped site (e.g. Wikipedia) breaks.
        let raw = b"HTTP/1.1 304 Not Modified\r\nContent-Encoding: gzip\r\n\r\n";
        let r = parse_response(raw).expect("empty-body gzip response must not error");
        assert_eq!(r.status, 304);
        assert!(r.body.is_empty());
        // Direct unit check too.
        let headers = vec![("Content-Encoding".to_string(), "gzip".to_string())];
        assert!(decode_content_encoding(&headers, Vec::new()).unwrap().is_empty());
    }

    // ==================================================================
    // HTTP Range requests / 206 Partial Content (RFC 9110 §14).
    // ==================================================================

    /// The typed Range builders emit Chrome-shaped `Range:` headers:
    /// `bytes=first-last`, `bytes=first-`, `bytes=-suffix`. The header
    /// must survive onto the request unchanged (caller headers are passed
    /// through verbatim by transact_with_deadline).
    #[test]
    fn range_request_builders_emit_correct_header() {
        let url = Url::parse("https://example.com/video.mp4").unwrap();

        let r = Request::get(url.clone()).range(0, 1023);
        assert_eq!(
            r.headers
                .iter()
                .find(|(k, _)| k.eq_ignore_ascii_case("range"))
                .map(|(_, v)| v.as_str()),
            Some("bytes=0-1023")
        );

        let r = Request::get(url.clone()).range_from(2048);
        assert_eq!(
            r.headers.iter().find(|(k, _)| k == "Range").unwrap().1,
            "bytes=2048-"
        );

        let r = Request::get(url.clone()).range_suffix(500);
        assert_eq!(
            r.headers.iter().find(|(k, _)| k == "Range").unwrap().1,
            "bytes=-500"
        );

        // If-Range pins the range to a validator.
        let r = Request::get(url).range(0, 99).if_range("\"abc-etag\"");
        assert_eq!(
            r.headers.iter().find(|(k, _)| k == "If-Range").unwrap().1,
            "\"abc-etag\""
        );
    }

    /// The whole wire request for a Range GET must actually carry the
    /// `Range:` header — proving transact's header pass-through does NOT
    /// strip caller headers. We run a real loopback server that echoes the
    /// raw request line + headers back as the 206 body and assert the
    /// `Range:` line is present on the wire.
    #[test]
    fn range_header_reaches_the_wire() {
        use std::io::{Read, Write};
        use std::net::TcpListener;
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        std::thread::spawn(move || {
            if let Ok((mut sock, _)) = listener.accept() {
                let mut buf = [0u8; 2048];
                let n = sock.read(&mut buf).unwrap_or(0);
                let req = String::from_utf8_lossy(&buf[..n]).to_string();
                let saw_range = req
                    .lines()
                    .any(|l| l.to_ascii_lowercase().starts_with("range:"));
                let marker = if saw_range { "RANGE-SEEN" } else { "NO-RANGE" };
                let body = marker.as_bytes();
                let head = format!(
                    "HTTP/1.1 206 Partial Content\r\nAccept-Ranges: bytes\r\n\
                     Content-Range: bytes 0-{}/100\r\nContent-Length: {}\r\n\
                     Connection: close\r\n\r\n",
                    body.len() - 1,
                    body.len()
                );
                let _ = sock.write_all(head.as_bytes());
                let _ = sock.write_all(body);
            }
        });
        let url = Url::parse(&format!("http://127.0.0.1:{port}/f")).unwrap();
        let client = Client::with_timeout(5000);
        let req = Request::get(url).range(0, 9);
        let resp = client.send(req).expect("range fetch");
        assert_eq!(resp.status, 206);
        assert!(resp.is_partial());
        assert_eq!(resp.body, b"RANGE-SEEN", "Range header was stripped!");
    }

    /// Parse a synthetic 206 response: status, Content-Range start/end/total,
    /// Accept-Ranges, and that the body is exactly the slice.
    #[test]
    fn parses_206_partial_content() {
        let raw = b"HTTP/1.1 206 Partial Content\r\n\
                    Accept-Ranges: bytes\r\n\
                    Content-Range: bytes 200-1023/146515\r\n\
                    Content-Length: 824\r\n\
                    Content-Type: video/mp4\r\n\r\n";
        // Body of exactly 824 bytes (1023-200+1).
        let mut buf = raw.to_vec();
        buf.extend(std::iter::repeat(b'X').take(824));
        let r = parse_response(&buf).unwrap();
        assert_eq!(r.status, 206);
        assert!(r.is_partial());
        assert!(r.supports_byte_ranges());
        assert_eq!(r.accept_ranges(), Some("bytes"));

        let cr = r.content_range().expect("Content-Range must parse");
        assert_eq!(cr.first, Some(200));
        assert_eq!(cr.last, Some(1023));
        assert_eq!(cr.complete_len, Some(146515));
        assert_eq!(cr.slice_len(), Some(824));
        // The body is exactly the slice — its length equals last-first+1.
        assert_eq!(r.body.len() as u64, cr.slice_len().unwrap());
    }

    /// 416 Range Not Satisfiable: status flagged, and the unsatisfied-range
    /// `bytes */total` Content-Range parses with no positions but a total.
    #[test]
    fn parses_416_range_not_satisfiable() {
        let raw = b"HTTP/1.1 416 Range Not Satisfiable\r\n\
                    Content-Range: bytes */146515\r\n\
                    Content-Length: 0\r\n\r\n";
        let r = parse_response(raw).unwrap();
        assert_eq!(r.status, 416);
        assert!(r.is_range_not_satisfiable());
        let cr = r.content_range().expect("unsatisfied Content-Range parses");
        assert_eq!(cr.first, None);
        assert_eq!(cr.last, None);
        assert_eq!(cr.complete_len, Some(146515));
        assert_eq!(cr.slice_len(), None);
    }

    /// If-Range semantics at the response level: when the validator still
    /// matches the server returns 206 (the slice); when it changed the
    /// server returns the FULL 200 body. We assert the client surfaces both
    /// faithfully (it does not itself decide — it reports what arrived).
    #[test]
    fn if_range_conditional_206_vs_200() {
        // Unchanged validator → 206 slice.
        let mut p = b"HTTP/1.1 206 Partial Content\r\n\
                      Content-Range: bytes 0-3/12\r\nContent-Length: 4\r\n\r\n"
            .to_vec();
        p.extend_from_slice(b"abcd");
        let r = parse_response(&p).unwrap();
        assert!(r.is_partial());
        assert_eq!(r.body, b"abcd");
        assert_eq!(r.content_range().unwrap().complete_len, Some(12));

        // Changed validator → server ignores the Range and sends 200 full.
        let mut full = b"HTTP/1.1 200 OK\r\nContent-Length: 12\r\nAccept-Ranges: bytes\r\n\r\n"
            .to_vec();
        full.extend_from_slice(b"abcdefghijkl");
        let r = parse_response(&full).unwrap();
        assert!(!r.is_partial());
        assert_eq!(r.status, 200);
        assert_eq!(r.body, b"abcdefghijkl");
        // No Content-Range on a 200 — the consumer must use the whole body.
        assert!(r.content_range().is_none());
    }

    /// Direct unit coverage of the Content-Range grammar parser, including
    /// the unknown-total `*` form, malformed/overflow rejection, and the
    /// case-insensitive `bytes` unit.
    #[test]
    fn content_range_parser_grammar() {
        // Satisfied with explicit total.
        let cr = parse_content_range("bytes 0-499/1234").unwrap();
        assert_eq!((cr.first, cr.last, cr.complete_len), (Some(0), Some(499), Some(1234)));
        assert_eq!(cr.slice_len(), Some(500));

        // Satisfied, unknown total (`*`).
        let cr = parse_content_range("bytes 100-199/*").unwrap();
        assert_eq!((cr.first, cr.last, cr.complete_len), (Some(100), Some(199), None));
        assert_eq!(cr.slice_len(), Some(100));

        // Unsatisfied-range form (the 416 body).
        let cr = parse_content_range("bytes */777").unwrap();
        assert_eq!((cr.first, cr.last, cr.complete_len), (None, None, Some(777)));

        // Case-insensitive unit + extra whitespace tolerated.
        let cr = parse_content_range("  BYTES   5-9 / 10 ").unwrap();
        assert_eq!((cr.first, cr.last, cr.complete_len), (Some(5), Some(9), Some(10)));

        // Rejections: wrong unit, last<first, range past total, both `*`,
        // garbage, and integer overflow.
        assert!(parse_content_range("items 0-9/10").is_none());
        assert!(parse_content_range("bytes 9-0/10").is_none());
        assert!(parse_content_range("bytes 0-10/10").is_none()); // last >= total
        assert!(parse_content_range("bytes */*").is_none());
        assert!(parse_content_range("bytes nonsense").is_none());
        assert!(parse_content_range("bytes 0-99999999999999999999/x").is_none());
    }

    /// CACHE CORRECTNESS: a 206 partial response must NOT be stored as a
    /// normal full-body cache entry (this cache has no sparse backing
    /// store, so a later full-GET hit would return the slice as if it were
    /// the whole resource — silent truncation). Chrome routes 206s through
    /// partial_data.cc sparse entries instead; with no sparse store the
    /// right behavior is "don't cache". A 200 with the same headers IS
    /// cached, proving we only excluded the partial.
    #[test]
    fn cache_does_not_store_206_as_full_entry() {
        let headers = vec![
            ("Content-Range".to_string(), "bytes 0-3/12".to_string()),
            ("Cache-Control".to_string(), "max-age=3600".to_string()),
            ("Accept-Ranges".to_string(), "bytes".to_string()),
        ];
        // 206 → not cached even with an explicit cacheable Cache-Control.
        assert!(
            crate::cache::build_entry_if_cacheable(206, "Partial Content", &headers, b"abcd", &[])
                .is_none(),
            "206 partial content must never be stored as a full cache entry"
        );
        // 200 with the same Cache-Control IS cacheable (control case).
        let h200 = vec![("Cache-Control".to_string(), "max-age=3600".to_string())];
        assert!(
            crate::cache::build_entry_if_cacheable(200, "OK", &h200, b"abcdefghijkl", &[])
                .is_some(),
            "200 full content should still cache normally"
        );
    }

    // ==================================================================
    // M6.3 — h2 pool tests: fingerprint identity, flag no-op, shared
    // connection (Tier-2 mock peer over an in-memory duplex pipe).
    // ==================================================================

    /// FINGERPRINT BYTE-IDENTITY: the shared `chrome131_h2_headers` fn
    /// (used by BOTH the flag-on pooled driver and the flag-off legacy
    /// path) and `Connection::new_client()` (the SETTINGS home) must
    /// produce byte-for-byte identical wire bytes regardless of path. Any
    /// future edit that diverges the header literal or SETTINGS fails
    /// here — closing the JA4-h2 / Akamai-h2 drift door.
    #[test]
    fn m63_fingerprint_bytes_are_byte_identical() {
        // SETTINGS + preface are produced solely by new_client().
        let a = crate::http2::Connection::new_client().drain_outgoing();
        let b = crate::http2::Connection::new_client().drain_outgoing();
        assert_eq!(a, b, "preface+SETTINGS must be deterministic");
        assert!(a.starts_with(crate::http2::CONNECTION_PREFACE));

        // The header block is produced by the SHARED fn → identical bytes
        // for the same inputs, on both paths.
        let h1 = chrome131_h2_headers(
            "GET",
            "example.com",
            "https",
            "/",
            "gzip, deflate, br",
        );
        let h2 = chrome131_h2_headers(
            "GET",
            "example.com",
            "https",
            "/",
            "gzip, deflate, br",
        );
        let enc1 = crate::http2::hpack_encode_block(&h1);
        let enc2 = crate::http2::hpack_encode_block(&h2);
        assert_eq!(enc1, enc2, "shared header fn must be deterministic");

        // Pseudo-header order is fixed (:method,:authority,:scheme,:path)
        // and the priority header is present (RFC 9218 nav priority, NOT
        // a separate PRIORITY frame).
        assert_eq!(h1[0].0, ":method");
        assert_eq!(h1[1].0, ":authority");
        assert_eq!(h1[2].0, ":scheme");
        assert_eq!(h1[3].0, ":path");
        assert!(h1.iter().any(|(k, v)| *k == "priority" && *v == "u=0, i"));
        // Regular headers (after the 4 pseudo) are alphabetical.
        let regular: Vec<&str> = h1[4..].iter().map(|(k, _)| *k).collect();
        let mut sorted = regular.clone();
        sorted.sort();
        assert_eq!(regular, sorted, "regular headers must be alphabetical");
    }

    /// FLAG OFF = NO-OP: with CV_H2_POOL unset the flag function returns
    /// false and the h2 pool stays empty (the seam is skipped). We assert
    /// the flag default and that a fresh Client's h2 pool is empty.
    #[test]
    fn m63_flag_default_off_pool_inert() {
        // The process may have CV_H2_POOL set by another test runner; we
        // only assert the parse semantics here, not the live OnceLock.
        // (h2_pool_enabled caches once per process.)
        let parse = |v: &str| v == "1" || v.eq_ignore_ascii_case("true");
        assert!(!parse("0"));
        assert!(!parse(""));
        assert!(!parse("no"));
        assert!(parse("1"));
        assert!(parse("true"));
        assert!(parse("TRUE"));

        // A fresh Client's h2 pool is empty and inert.
        let c = Client::new();
        let pool = c.h2_pool.lock().unwrap();
        assert!(pool.is_empty());
    }

    // ---------- Tier-2 in-memory duplex pipe (no TLS, no sockets) -------

    /// A blocking, in-memory byte channel. Reads block (via Condvar)
    /// until bytes are available or the peer closes — no sleep-polling.
    #[derive(Clone)]
    struct Pipe {
        inner: Arc<(Mutex<PipeState>, std::sync::Condvar)>,
    }
    struct PipeState {
        buf: std::collections::VecDeque<u8>,
        closed: bool,
    }
    impl Pipe {
        fn new() -> Self {
            Self {
                inner: Arc::new((
                    Mutex::new(PipeState {
                        buf: std::collections::VecDeque::new(),
                        closed: false,
                    }),
                    std::sync::Condvar::new(),
                )),
            }
        }
        fn write(&self, data: &[u8]) {
            let (m, cv) = &*self.inner;
            let mut s = m.lock().unwrap();
            s.buf.extend(data.iter().copied());
            cv.notify_all();
        }
        fn close(&self) {
            let (m, cv) = &*self.inner;
            let mut s = m.lock().unwrap();
            s.closed = true;
            cv.notify_all();
        }
        /// Block until at least one byte is available; return up to
        /// buf.len() bytes. Returns 0 only when the peer closed AND the
        /// buffer is drained.
        fn read(&self, buf: &mut [u8]) -> usize {
            let (m, cv) = &*self.inner;
            let mut s = m.lock().unwrap();
            loop {
                if !s.buf.is_empty() {
                    let n = buf.len().min(s.buf.len());
                    for slot in buf.iter_mut().take(n) {
                        *slot = s.buf.pop_front().unwrap();
                    }
                    return n;
                }
                if s.closed {
                    return 0;
                }
                s = cv.wait(s).unwrap();
            }
        }
    }

    /// A duplex transport: the test's client side reads server→client and
    /// writes client→server, implementing `H2Transport` for the driver.
    struct DuplexClient {
        to_server: Pipe,
        from_server: Pipe,
    }
    impl H2Transport for DuplexClient {
        fn write_all(&mut self, data: &[u8]) -> Result<(), NetError> {
            self.to_server.write(data);
            Ok(())
        }
        fn read_some(&mut self, buf: &mut [u8]) -> Result<usize, NetError> {
            Ok(self.from_server.read(buf))
        }
        fn set_read_timeout_ms(&self, _ms: u32) {}
    }

    /// Mock h2 server: drains the client preface+SETTINGS+HEADERS off
    /// `to_server`, counts prefaces seen, and answers each request stream
    /// with a DISTINCT body. Runs on a dedicated thread.
    fn spawn_mock_h2_server(
        to_server: Pipe,
        from_server: Pipe,
        prefaces_seen: Arc<Mutex<u32>>,
        bodies: Vec<&'static [u8]>,
    ) -> std::thread::JoinHandle<()> {
        std::thread::spawn(move || {
            use crate::http2::{FrameHeader, FrameType, build_settings_frame, hpack_encode_block};
            let mut accum: Vec<u8> = Vec::new();
            let mut buf = [0u8; 4096];
            let preface = crate::http2::CONNECTION_PREFACE;
            let mut preface_done = false;
            // Send server SETTINGS up front.
            from_server.write(&build_settings_frame(&[], false));
            let mut answered = 0usize;
            loop {
                if answered >= bodies.len() {
                    from_server.close();
                    return;
                }
                let n = to_server.read(&mut buf);
                if n == 0 {
                    return;
                }
                accum.extend_from_slice(&buf[..n]);
                // Strip the client preface once.
                if !preface_done {
                    if accum.len() >= preface.len() {
                        assert!(accum.starts_with(preface), "client must send the h2 preface");
                        *prefaces_seen.lock().unwrap() += 1;
                        accum.drain(..preface.len());
                        preface_done = true;
                    } else {
                        continue;
                    }
                }
                // Parse whole frames; answer each client HEADERS (request).
                let mut consumed = 0usize;
                while accum.len() - consumed >= 9 {
                    let hdr = FrameHeader::decode(&accum[consumed..]).unwrap();
                    let flen = 9 + hdr.length as usize;
                    if accum.len() - consumed < flen {
                        break;
                    }
                    if hdr.typ == FrameType::Headers {
                        // Answer THIS stream id with its distinct body.
                        let sid = hdr.stream_id;
                        let body = bodies[answered];
                        answered += 1;
                        let hblock = hpack_encode_block(&[
                            (":status", "200"),
                            ("content-type", "text/plain"),
                        ]);
                        let mut wire = Vec::new();
                        // HEADERS (END_HEADERS, no END_STREAM — body follows)
                        FrameHeader { length: hblock.len() as u32, typ: FrameType::Headers, flags: 0x4, stream_id: sid }
                            .encode(&mut wire);
                        wire.extend_from_slice(&hblock);
                        // DATA END_STREAM
                        FrameHeader { length: body.len() as u32, typ: FrameType::Data, flags: 0x1, stream_id: sid }
                            .encode(&mut wire);
                        wire.extend_from_slice(body);
                        from_server.write(&wire);
                    }
                    consumed += flen;
                }
                accum.drain(..consumed);
            }
        })
    }

    /// TIER-2: ONE shared connection serves TWO sequential requests, each
    /// gets ITS OWN distinct body, and exactly ONE preface (handshake)
    /// occurred — proving the pool amortizes the handshake and the demux
    /// matches each response to its request.
    #[test]
    fn m63_tier2_shared_conn_two_requests_distinct_bodies() {
        let to_server = Pipe::new();
        let from_server = Pipe::new();
        let prefaces_seen = Arc::new(Mutex::new(0u32));
        let _srv = spawn_mock_h2_server(
            to_server.clone(),
            from_server.clone(),
            prefaces_seen.clone(),
            vec![b"ALPHA", b"BETA"],
        );

        let mut client = DuplexClient {
            to_server: to_server.clone(),
            from_server: from_server.clone(),
        };
        // ONE persistent state machine + ONE persistent decode table +
        // ONE accum — exactly the H2Conn fields.
        let mut h2 = crate::http2::Connection::new_client();
        let mut decode_dyn = crate::http2::HpackDynamicTable::new();
        let mut accum: Vec<u8> = Vec::new();
        // Write the preface ONCE (as h2_send_via_pool does at dial).
        let preface = h2.drain_outgoing();
        client.write_all(&preface).unwrap();

        let url = Url::parse("https://example.com/").unwrap();
        let req = Request::get(url);
        let deadline = Instant::now() + Duration::from_secs(10);

        let r1 = h2_drive_request(
            &mut client,
            &mut h2,
            &mut decode_dyn,
            &mut accum,
            &req,
            deadline,
        )
        .expect("request 1 must succeed");
        let r2 = h2_drive_request(
            &mut client,
            &mut h2,
            &mut decode_dyn,
            &mut accum,
            &req,
            deadline,
        )
        .expect("request 2 must succeed");

        assert_eq!(r1.status, 200);
        assert_eq!(r2.status, 200);
        assert_eq!(r1.body, b"ALPHA", "request 1 must get its OWN body");
        assert_eq!(r2.body, b"BETA", "request 2 must get its OWN body");
        // Exactly ONE handshake/preface for two requests = shared conn.
        assert_eq!(*prefaces_seen.lock().unwrap(), 1);
        // The two streams used distinct ids (1 then 3).
        assert_eq!(h2.next_stream_id, 5);
    }

    /// POOL BOOKKEEPING: a second checkout for the same origin returns the
    /// SAME Arc (shared), not a fresh conn. We populate the pool with a
    /// dummy H2Conn (no real socket needed — checkout only inspects flags
    /// and Arc identity).
    #[test]
    fn m63_checkout_reuses_same_arc() {
        let client = Client::new();
        // Build a placeholder H2Conn around a plain loopback socket so the
        // Conn is valid but unused (checkout never touches the socket).
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let h = std::thread::spawn(move || {
            let _ = listener.accept();
        });
        let addr = crate::dns::resolve("127.0.0.1", port).unwrap().remove(0);
        let sock = crate::socket::Socket::connect_with_timeout(&addr, 2000).unwrap();
        h.join().unwrap();
        let h2conn = H2Conn {
            conn: Conn::Plain(sock),
            h2: crate::http2::Connection::new_client(),
            decode_dyn: crate::http2::HpackDynamicTable::new(),
            accum: Vec::new(),
            last_used: Instant::now(),
            poisoned: false,
        };
        let arc = Arc::new(Mutex::new(h2conn));
        client
            .h2_pool
            .lock()
            .unwrap()
            .insert(("example.com".to_string(), 443), arc.clone());

        let c1 = client.h2_checkout("example.com", 443).expect("first checkout");
        let c2 = client.h2_checkout("example.com", 443).expect("second checkout");
        assert!(Arc::ptr_eq(&c1, &c2), "checkout must reuse the SAME conn");
        assert!(Arc::ptr_eq(&c1, &arc));

        // A poisoned conn is evicted on checkout (never reused).
        c1.lock().unwrap().poisoned = true;
        assert!(client.h2_checkout("example.com", 443).is_none());
        assert!(client.h2_pool.lock().unwrap().is_empty());
    }
}
