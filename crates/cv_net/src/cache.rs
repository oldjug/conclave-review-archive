//! HTTP response cache — RFC 9111 subset.
//!
//! In-memory only for V1: a HashMap keyed by `(url, vary_signature)`
//! that stores the response head + body plus parsed freshness metadata
//! (Date, ETag, Last-Modified, Cache-Control directives). The cache
//! decides one of:
//!   * `Fresh` — return cached response without touching the network.
//!   * `Stale` — issue a conditional revalidation (If-None-Match /
//!     If-Modified-Since); a 304 keeps the cached body, a 200 replaces.
//!   * `Bypass` — uncacheable response (Cache-Control: no-store, error
//!     status, Authorization header, etc.). Just fetch fresh.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// Decoded Cache-Control directives that affect freshness.
#[derive(Debug, Clone, Default)]
pub struct CacheControl {
    pub no_store: bool,
    pub no_cache: bool,
    pub must_revalidate: bool,
    pub immutable: bool,
    pub private: bool,
    /// max-age in seconds, if present.
    pub max_age: Option<u64>,
    /// s-maxage (shared cache). We're a private cache so we ignore it.
    pub s_max_age: Option<u64>,
    /// stale-while-revalidate window in seconds.
    pub swr: Option<u64>,
}

impl CacheControl {
    pub fn parse(value: &str) -> Self {
        let mut cc = Self::default();
        for tok in value.split(',') {
            let tok = tok.trim();
            let (name, val) = tok
                .split_once('=')
                .map_or((tok, ""), |(n, v)| (n.trim(), v.trim().trim_matches('"')));
            match name.to_ascii_lowercase().as_str() {
                "no-store" => cc.no_store = true,
                "no-cache" => cc.no_cache = true,
                "must-revalidate" => cc.must_revalidate = true,
                "immutable" => cc.immutable = true,
                "private" => cc.private = true,
                "max-age" => cc.max_age = val.parse().ok(),
                "s-maxage" => cc.s_max_age = val.parse().ok(),
                "stale-while-revalidate" => cc.swr = val.parse().ok(),
                _ => {}
            }
        }
        cc
    }
}

#[derive(Debug, Clone)]
pub struct CachedEntry {
    pub status: u16,
    pub reason: String,
    pub headers: Vec<(String, String)>,
    pub body: Vec<u8>,
    /// Wall-clock store time (seconds since the Unix epoch). Wall-clock — not a
    /// monotonic `Instant` — so freshness survives serialization to disk and a
    /// process restart.
    pub stored_at: u64,
    pub max_age: Option<u64>,
    pub etag: Option<String>,
    pub last_modified: Option<String>,
    /// Header names listed in the response `Vary` header. An empty list means
    /// no Vary constraints.
    pub vary: Vec<String>,
    /// The request header values (name, value) for each header named in `vary`,
    /// captured when the entry was stored. Used to validate that a subsequent
    /// request matches the stored variant.
    pub vary_request_headers: Vec<(String, String)>,
    /// True when the response carried `Cache-Control: no-cache` or
    /// `must-revalidate`, meaning the entry MUST be revalidated before reuse
    /// even if the age is within max-age.
    pub must_revalidate: bool,
}

/// Seconds since the Unix epoch (0 if the clock is before the epoch).
fn now_unix() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

impl CachedEntry {
    pub fn is_fresh(&self) -> bool {
        // no-cache / must-revalidate: MUST revalidate before reuse regardless
        // of age (RFC 9111 §5.2.2.2 and §5.2.2.3).
        if self.must_revalidate {
            return false;
        }
        let age = Duration::from_secs(now_unix().saturating_sub(self.stored_at));
        match self.max_age {
            Some(s) => age < Duration::from_secs(s),
            None => false,
        }
    }

    /// Re-freshen this entry after receiving a 304 Not Modified response.
    /// Updates `stored_at` to now and merges any updated headers the 304
    /// carries (e.g. a new `ETag`, `Cache-Control`, `Last-Modified`).
    /// Per RFC 9111 §4.3.4 the 304 headers are authoritative and replace the
    /// corresponding stored header values.
    pub fn refresh_after_304(&mut self, headers_304: &[(String, String)]) {
        self.stored_at = now_unix();
        for (k, v) in headers_304 {
            let kl = k.to_ascii_lowercase();
            match kl.as_str() {
                "etag" => self.etag = Some(v.clone()),
                "last-modified" => self.last_modified = Some(v.clone()),
                "cache-control" => {
                    // Replace the stored Cache-Control with the 304's value
                    // and reparse to update max_age / must_revalidate.
                    self.headers.retain(|(hk, _)| !hk.eq_ignore_ascii_case("cache-control"));
                    self.headers.push((k.clone(), v.clone()));
                    let cc = CacheControl::parse(v);
                    self.max_age = cc.max_age;
                    self.must_revalidate = cc.no_cache || cc.must_revalidate;
                }
                // For all other headers, replace the stored copy if present,
                // otherwise append.
                _ => {
                    if let Some(pos) = self.headers.iter().position(|(hk, _)| hk.eq_ignore_ascii_case(k)) {
                        self.headers[pos].1 = v.clone();
                    } else {
                        self.headers.push((k.clone(), v.clone()));
                    }
                }
            }
        }
    }
}

/// Max entries kept in the in-memory cache before LRU eviction. The disk store
/// (when configured) is unbounded by this and still backs misses.
const MAX_MEM_ENTRIES: usize = 4096;
/// Max total body bytes kept in memory before LRU eviction (256 MiB) — stops a
/// few huge responses from pinning the whole budget.
const MAX_MEM_BYTES: usize = 256 * 1024 * 1024;

/// In-memory store with LRU ordering. `order` front = least-recently-used.
#[derive(Default)]
struct CacheStore {
    map: HashMap<String, CachedEntry>,
    order: std::collections::VecDeque<String>,
    bytes: usize,
}

impl CacheStore {
    fn touch(&mut self, url: &str) {
        if let Some(pos) = self.order.iter().position(|k| k == url) {
            let k = self.order.remove(pos).unwrap();
            self.order.push_back(k);
        }
    }
    fn insert(&mut self, url: String, entry: CachedEntry) {
        if let Some(old) = self.map.remove(&url) {
            self.bytes = self.bytes.saturating_sub(old.body.len());
            if let Some(pos) = self.order.iter().position(|k| *k == url) {
                self.order.remove(pos);
            }
        }
        self.bytes += entry.body.len();
        self.map.insert(url.clone(), entry);
        self.order.push_back(url);
        // Evict LRU until within both caps. Always keep at least one entry.
        while self.map.len() > MAX_MEM_ENTRIES
            || (self.bytes > MAX_MEM_BYTES && self.map.len() > 1)
        {
            let Some(victim) = self.order.pop_front() else { break };
            if let Some(e) = self.map.remove(&victim) {
                self.bytes = self.bytes.saturating_sub(e.body.len());
            }
        }
    }
    fn remove(&mut self, url: &str) {
        if let Some(e) = self.map.remove(url) {
            self.bytes = self.bytes.saturating_sub(e.body.len());
        }
        if let Some(pos) = self.order.iter().position(|k| k == url) {
            self.order.remove(pos);
        }
    }
}

/// Thread-safe shared cache. Cheap to clone.
#[derive(Clone, Default)]
pub struct HttpCache {
    inner: Arc<Mutex<CacheStore>>,
    /// When set, entries also persist to this directory so the cache survives a
    /// process restart — cold launches reuse subresources instead of
    /// re-downloading every CSS/JS/font/image.
    disk_dir: Option<Arc<std::path::PathBuf>>,
}

impl std::fmt::Debug for HttpCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HttpCache").finish_non_exhaustive()
    }
}

impl HttpCache {
    pub fn new() -> Self {
        Self::default()
    }

    /// Construct a cache that also persists entries to `dir` (created if
    /// missing). Falls back to memory-only behavior if the directory can't be
    /// created.
    pub fn with_disk_dir<P: Into<std::path::PathBuf>>(dir: P) -> Self {
        let dir = dir.into();
        let _ = std::fs::create_dir_all(&dir);
        Self {
            inner: Arc::new(Mutex::new(CacheStore::default())),
            disk_dir: Some(Arc::new(dir)),
        }
    }

    /// Look up a cached entry for the URL. Returns the entry whether
    /// fresh or stale — caller decides whether to revalidate. Misses fall back
    /// to the disk store (and promote the entry into memory, LRU-bounded).
    pub fn get(&self, url: &str) -> Option<CachedEntry> {
        if let Ok(mut s) = self.inner.lock() {
            if let Some(e) = s.map.get(url).cloned() {
                s.touch(url);
                return Some(e);
            }
        }
        let dir = self.disk_dir.as_ref()?;
        let bytes = std::fs::read(dir.join(cache_filename(url))).ok()?;
        let entry = deserialize_entry(&bytes)?;
        if let Ok(mut s) = self.inner.lock() {
            s.insert(url.to_string(), entry.clone());
        }
        Some(entry)
    }

    /// Store (or replace) a response in the cache (memory + disk).
    pub fn put(&self, url: &str, entry: CachedEntry) {
        if let Some(dir) = &self.disk_dir {
            let path = dir.join(cache_filename(url));
            let tmp = path.with_extension("tmp");
            // Write-then-rename so a kill mid-write can't leave a torn file.
            if std::fs::write(&tmp, serialize_entry(&entry)).is_ok() {
                let _ = std::fs::rename(&tmp, &path);
            }
        }
        if let Ok(mut s) = self.inner.lock() {
            s.insert(url.to_string(), entry);
        }
    }

    /// Drop a stale or invalidated entry (memory + disk).
    pub fn invalidate(&self, url: &str) {
        if let Ok(mut s) = self.inner.lock() {
            s.remove(url);
        }
        if let Some(dir) = &self.disk_dir {
            let _ = std::fs::remove_file(dir.join(cache_filename(url)));
        }
    }

    /// Total in-memory entries stored, useful for diagnostics.
    pub fn len(&self) -> usize {
        self.inner.lock().map(|s| s.map.len()).unwrap_or(0)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Build a CachedEntry from response headers + body. Returns None if
/// the response is uncacheable (Cache-Control: no-store, 5xx with no
/// other directive, etc.).
///
/// `request_headers` is the list of headers that were sent with the original
/// request; it is used to capture the values of any headers named in a `Vary`
/// response header so a subsequent request can be validated against them.
pub fn build_entry_if_cacheable(
    status: u16,
    reason: &str,
    headers: &[(String, String)],
    body: &[u8],
    request_headers: &[(String, String)],
) -> Option<CachedEntry> {
    // 2xx, 3xx (except 302/307 without explicit Cache-Control), and a
    // few specific 4xxs are cacheable. We accept 200 / 203 / 204 / 206
    // / 300 / 301 / 308 / 404 / 405 / 410 / 414 / 501 by default per
    // RFC 9110 §4.2.2.
    let default_cacheable = matches!(
        status,
        200 | 203 | 204 | 206 | 300 | 301 | 308 | 404 | 405 | 410 | 414 | 501
    );
    let mut etag: Option<String> = None;
    let mut last_modified: Option<String> = None;
    let mut vary: Vec<String> = Vec::new();
    // RFC 9110 §5.3: a recipient MAY combine multiple header fields with
    // the same name by joining their values with commas. Servers (and
    // proxies in front of them) often emit Cache-Control as several
    // separate lines — e.g. `Cache-Control: max-age=...` AND
    // `Cache-Control: public`. Parsing each line in isolation and
    // overwriting would let the last (directive-only) line clobber the
    // max-age from an earlier one, leaving the entry with no freshness
    // lifetime → permanently "stale" → revalidated on every use. Collect
    // ALL Cache-Control values and parse them together.
    let mut cc_values: Vec<String> = Vec::new();
    for (k, v) in headers {
        let k_lc = k.to_ascii_lowercase();
        match k_lc.as_str() {
            "cache-control" => cc_values.push(v.clone()),
            "etag" => etag = Some(v.clone()),
            "last-modified" => last_modified = Some(v.clone()),
            "vary" => vary.extend(v.split(',').map(|s| s.trim().to_string())),
            _ => {}
        }
    }
    let has_explicit = !cc_values.is_empty();
    let cc = CacheControl::parse(&cc_values.join(", "));
    if cc.no_store {
        return None;
    }
    if !default_cacheable && !has_explicit {
        return None;
    }
    // If Vary: * we treat the response as uncacheable — we can't
    // distinguish stored responses by every conceivable input.
    if vary.iter().any(|v| v == "*") {
        return None;
    }
    // Capture the request header values for each name listed in Vary so we
    // can validate that a later request sends the same values before serving
    // this entry (RFC 9111 §4.1).
    let vary_request_headers: Vec<(String, String)> = vary
        .iter()
        .filter_map(|field_name| {
            let val = request_headers
                .iter()
                .find(|(k, _)| k.eq_ignore_ascii_case(field_name))
                .map(|(_, v)| v.clone())
                .unwrap_or_default();
            Some((field_name.clone(), val))
        })
        .collect();
    let must_revalidate = cc.no_cache || cc.must_revalidate;
    // The `body` handed to the cache has ALREADY been content-decoded
    // (gzip/deflate/br stripped) upstream. Persisting the original headers would
    // leave `Content-Encoding: gzip` and the *encoded* `Content-Length` on an
    // entry whose stored bytes are the DECODED representation — a self-
    // inconsistent response. On a later cache HIT a consumer then re-frames the
    // body to the (smaller) encoded length and/or re-runs gzip on already-plain
    // bytes → "gzip decode: truncated gzip" (this is why a revisited gzipped
    // page — e.g. Wikipedia — failed to load while a fresh fetch worked).
    // Normalize: drop Content-Encoding + Transfer-Encoding and set an accurate
    // Content-Length so the stored entry truthfully describes its decoded body.
    let mut normalized_headers: Vec<(String, String)> = headers
        .iter()
        .filter(|(k, _)| {
            !k.eq_ignore_ascii_case("content-encoding")
                && !k.eq_ignore_ascii_case("transfer-encoding")
                && !k.eq_ignore_ascii_case("content-length")
        })
        .cloned()
        .collect();
    normalized_headers.push(("Content-Length".to_string(), body.len().to_string()));
    Some(CachedEntry {
        status,
        reason: reason.to_string(),
        headers: normalized_headers,
        body: body.to_vec(),
        stored_at: now_unix(),
        max_age: cc.max_age,
        etag,
        last_modified,
        vary,
        vary_request_headers,
        must_revalidate,
    })
}

// ---------------------------------------------------------------------------
// Disk persistence: a compact, dependency-free serialization (no serde).
// ---------------------------------------------------------------------------

/// Stable on-disk filename for a URL (hashed so it's filesystem-safe and short).
fn cache_filename(url: &str) -> String {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    url.hash(&mut h);
    format!("{:016x}.tbc", h.finish())
}

fn wr_str(out: &mut Vec<u8>, s: &str) {
    out.extend_from_slice(&(s.len() as u32).to_le_bytes());
    out.extend_from_slice(s.as_bytes());
}
fn wr_opt_u64(out: &mut Vec<u8>, v: Option<u64>) {
    match v {
        Some(n) => {
            out.push(1);
            out.extend_from_slice(&n.to_le_bytes());
        }
        None => out.push(0),
    }
}
fn wr_opt_str(out: &mut Vec<u8>, v: Option<&str>) {
    match v {
        Some(s) => {
            out.push(1);
            wr_str(out, s);
        }
        None => out.push(0),
    }
}

fn serialize_entry(e: &CachedEntry) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(b"TBC2"); // magic + version
    out.extend_from_slice(&e.status.to_le_bytes());
    wr_str(&mut out, &e.reason);
    out.extend_from_slice(&(e.headers.len() as u32).to_le_bytes());
    for (k, v) in &e.headers {
        wr_str(&mut out, k);
        wr_str(&mut out, v);
    }
    out.extend_from_slice(&(e.body.len() as u64).to_le_bytes());
    out.extend_from_slice(&e.body);
    out.extend_from_slice(&e.stored_at.to_le_bytes());
    wr_opt_u64(&mut out, e.max_age);
    wr_opt_str(&mut out, e.etag.as_deref());
    wr_opt_str(&mut out, e.last_modified.as_deref());
    out.extend_from_slice(&(e.vary.len() as u32).to_le_bytes());
    for v in &e.vary {
        wr_str(&mut out, v);
    }
    // v2: vary_request_headers and must_revalidate
    out.extend_from_slice(&(e.vary_request_headers.len() as u32).to_le_bytes());
    for (k, v) in &e.vary_request_headers {
        wr_str(&mut out, k);
        wr_str(&mut out, v);
    }
    out.push(u8::from(e.must_revalidate));
    out
}

struct Reader<'a> {
    b: &'a [u8],
    pos: usize,
}
impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Option<&'a [u8]> {
        let s = self.b.get(self.pos..self.pos.checked_add(n)?)?;
        self.pos += n;
        Some(s)
    }
    fn rd_u16(&mut self) -> Option<u16> {
        Some(u16::from_le_bytes(self.take(2)?.try_into().ok()?))
    }
    fn rd_u32(&mut self) -> Option<u32> {
        Some(u32::from_le_bytes(self.take(4)?.try_into().ok()?))
    }
    fn rd_u64(&mut self) -> Option<u64> {
        Some(u64::from_le_bytes(self.take(8)?.try_into().ok()?))
    }
    fn rd_str(&mut self) -> Option<String> {
        let len = self.rd_u32()? as usize;
        Some(String::from_utf8_lossy(self.take(len)?).into_owned())
    }
    fn rd_opt_u64(&mut self) -> Option<Option<u64>> {
        if self.take(1)?[0] == 0 {
            Some(None)
        } else {
            Some(Some(self.rd_u64()?))
        }
    }
    fn rd_opt_str(&mut self) -> Option<Option<String>> {
        if self.take(1)?[0] == 0 {
            Some(None)
        } else {
            Some(Some(self.rd_str()?))
        }
    }
}

fn deserialize_entry(b: &[u8]) -> Option<CachedEntry> {
    let mut c = Reader { b, pos: 0 };
    let magic = c.take(4)?;
    // Accept both TBC1 (legacy, missing v2 fields) and TBC2.
    let is_v2 = magic == b"TBC2";
    if !is_v2 && magic != b"TBC1" {
        return None;
    }
    let status = c.rd_u16()?;
    let reason = c.rd_str()?;
    let nh = c.rd_u32()? as usize;
    let mut headers = Vec::with_capacity(nh);
    for _ in 0..nh {
        let k = c.rd_str()?;
        let v = c.rd_str()?;
        headers.push((k, v));
    }
    let blen = c.rd_u64()? as usize;
    let body = c.take(blen)?.to_vec();
    let stored_at = c.rd_u64()?;
    let max_age = c.rd_opt_u64()?;
    let etag = c.rd_opt_str()?;
    let last_modified = c.rd_opt_str()?;
    let nv = c.rd_u32()? as usize;
    let mut vary = Vec::with_capacity(nv);
    for _ in 0..nv {
        vary.push(c.rd_str()?);
    }
    // v2 fields: vary_request_headers + must_revalidate
    let (vary_request_headers, must_revalidate) = if is_v2 {
        let nvrh = c.rd_u32()? as usize;
        let mut vrh = Vec::with_capacity(nvrh);
        for _ in 0..nvrh {
            let k = c.rd_str()?;
            let v = c.rd_str()?;
            vrh.push((k, v));
        }
        let mr = c.take(1)?[0] != 0;
        (vrh, mr)
    } else {
        // TBC1: no vary_request_headers or must_revalidate stored; derive
        // must_revalidate conservatively from the response headers.
        let mr = headers.iter().any(|(k, v)| {
            k.eq_ignore_ascii_case("cache-control")
                && (v.to_ascii_lowercase().contains("no-cache")
                    || v.to_ascii_lowercase().contains("must-revalidate"))
        });
        (Vec::new(), mr)
    };
    Some(CachedEntry {
        status,
        reason,
        headers,
        body,
        stored_at,
        max_age,
        etag,
        last_modified,
        vary,
        vary_request_headers,
        must_revalidate,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_directives() {
        let cc = CacheControl::parse("max-age=300, must-revalidate");
        assert_eq!(cc.max_age, Some(300));
        assert!(cc.must_revalidate);
    }
    #[test]
    fn no_store_blocks_caching() {
        let e = build_entry_if_cacheable(
            200,
            "OK",
            &[("Cache-Control".into(), "no-store".into())],
            b"hi",
            &[],
        );
        assert!(e.is_none());
    }
    #[test]
    fn vary_star_blocks_caching() {
        let e = build_entry_if_cacheable(
            200,
            "OK",
            &[
                ("Cache-Control".into(), "max-age=60".into()),
                ("Vary".into(), "*".into()),
            ],
            b"hi",
            &[],
        );
        assert!(e.is_none());
    }
    #[test]
    fn multiple_cache_control_lines_are_combined() {
        // A server emitting Cache-Control across two lines (max-age on one,
        // a bare directive on another) must still yield the max-age — the
        // second line must not clobber the first. This is the HN news.css
        // case: `Cache-Control: max-age=301959800` + `Cache-Control: public`
        // arrived as separate header lines and the entry ended up with
        // max_age=None, forcing a revalidation on every reuse.
        let e = build_entry_if_cacheable(
            200,
            "OK",
            &[
                ("Cache-Control".into(), "max-age=301959800".into()),
                ("Cache-Control".into(), "public".into()),
            ],
            b"body",
            &[],
        )
        .unwrap();
        assert_eq!(e.max_age, Some(301959800));
        assert!(e.is_fresh());
    }

    /// Bug 1: Cache-Control: no-cache must force revalidation (is_fresh = false).
    #[test]
    fn no_cache_forces_revalidation() {
        let e = build_entry_if_cacheable(
            200,
            "OK",
            &[("Cache-Control".into(), "max-age=3600, no-cache".into())],
            b"data",
            &[],
        )
        .unwrap();
        // Even though max-age=3600, no-cache means we must revalidate.
        assert!(e.must_revalidate);
        assert!(!e.is_fresh(), "no-cache entry must not be served as fresh");
    }

    /// Bug 1 variant: must-revalidate also forces revalidation once stale.
    #[test]
    fn must_revalidate_flag_set() {
        let e = build_entry_if_cacheable(
            200,
            "OK",
            &[("Cache-Control".into(), "max-age=60, must-revalidate".into())],
            b"data",
            &[],
        )
        .unwrap();
        assert!(e.must_revalidate);
        assert!(!e.is_fresh());
    }

    /// Bug 2: Vary request headers are captured at store time.
    #[test]
    fn vary_request_headers_captured() {
        let req_headers = vec![
            ("Accept-Language".into(), "en-US".into()),
            ("Accept-Encoding".into(), "gzip".into()),
        ];
        let e = build_entry_if_cacheable(
            200,
            "OK",
            &[
                ("Cache-Control".into(), "max-age=60".into()),
                ("Vary".into(), "Accept-Language".into()),
            ],
            b"hello",
            &req_headers,
        )
        .unwrap();
        assert_eq!(e.vary, &["Accept-Language"]);
        assert_eq!(
            e.vary_request_headers,
            &[("Accept-Language".to_string(), "en-US".to_string())]
        );
    }

    /// Bug 3: refresh_after_304 updates stored_at and merges headers.
    #[test]
    fn refresh_after_304_updates_stored_at() {
        let mut e = build_entry_if_cacheable(
            200,
            "OK",
            &[
                ("Cache-Control".into(), "max-age=1".into()),
                ("ETag".into(), r#""old""#.into()),
            ],
            b"body",
            &[],
        )
        .unwrap();
        // Force the entry to appear old.
        e.stored_at = now_unix().saturating_sub(100);
        assert!(!e.is_fresh(), "precondition: entry is stale");

        let headers_304 = vec![
            ("ETag".into(), r#""new""#.into()),
            ("Cache-Control".into(), "max-age=3600".into()),
        ];
        e.refresh_after_304(&headers_304);

        // stored_at should be ~now, so the entry is fresh again.
        assert!(e.is_fresh(), "entry should be fresh after 304 refresh");
        // ETag should be updated.
        assert_eq!(e.etag.as_deref(), Some(r#""new""#));
        // max_age updated from the 304 Cache-Control.
        assert_eq!(e.max_age, Some(3600));
    }

    #[test]
    fn round_trip() {
        let c = HttpCache::new();
        let e = build_entry_if_cacheable(
            200,
            "OK",
            &[("Cache-Control".into(), "max-age=60".into())],
            b"body",
            &[],
        )
        .unwrap();
        c.put("https://example.com/", e);
        assert_eq!(c.len(), 1);
        let got = c.get("https://example.com/").unwrap();
        assert!(got.is_fresh());
        assert_eq!(got.body, b"body");
    }

    #[test]
    fn disk_round_trip_survives_new_cache() {
        let dir = std::env::temp_dir().join("tbnet_cache_disk_test");
        let _ = std::fs::remove_dir_all(&dir);
        let url = "https://example.com/app.js";
        {
            let c = HttpCache::with_disk_dir(&dir);
            let e = build_entry_if_cacheable(
                200,
                "OK",
                &[
                    ("Cache-Control".into(), "max-age=3600".into()),
                    ("ETag".into(), "\"abc\"".into()),
                ],
                b"console.log(1)",
                &[],
            )
            .unwrap();
            c.put(url, e);
        }
        // A FRESH cache (simulating a process restart) reads it back from disk.
        let c2 = HttpCache::with_disk_dir(&dir);
        let got = c2.get(url).expect("entry loaded from disk");
        assert_eq!(got.status, 200);
        assert_eq!(got.body, b"console.log(1)");
        assert_eq!(got.etag.as_deref(), Some("\"abc\""));
        assert!(got.is_fresh());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn memory_cache_is_bounded_and_evicts_lru() {
        let c = HttpCache::new(); // memory-only
        let mk = || {
            build_entry_if_cacheable(
                200,
                "OK",
                &[("Cache-Control".into(), "max-age=60".into())],
                b"x",
                &[],
            )
            .unwrap()
        };
        for i in 0..(MAX_MEM_ENTRIES + 64) {
            c.put(&format!("https://e/{i}"), mk());
        }
        // Never exceeds the entry cap (the unbounded-growth liability is gone).
        assert!(c.len() <= MAX_MEM_ENTRIES, "cache exceeded cap: {}", c.len());
        // The most recently inserted key survives; the very first was evicted.
        assert!(c.get(&format!("https://e/{}", MAX_MEM_ENTRIES + 63)).is_some());
        assert!(c.get("https://e/0").is_none());
    }

    #[test]
    fn cached_entry_normalizes_decoded_body_headers() {
        // A response whose body was gzip-decoded upstream still arrives with
        // Content-Encoding: gzip + the ENCODED Content-Length. The cache must
        // store headers consistent with the DECODED body, else a later cache
        // hit re-frames/re-decodes it → "truncated gzip" (the Wikipedia bug).
        let decoded = b"<html><body>decoded plaintext, not gzip</body></html>".to_vec();
        let headers = vec![
            ("Content-Encoding".into(), "gzip".into()),
            ("Content-Length".into(), "20".into()), // wrong: the encoded length
            ("Transfer-Encoding".into(), "chunked".into()),
            ("Cache-Control".into(), "max-age=60".into()),
            ("Content-Type".into(), "text/html".into()),
        ];
        let entry =
            build_entry_if_cacheable(200, "OK", &headers, &decoded, &[]).expect("cacheable");
        assert!(
            entry
                .headers
                .iter()
                .all(|(k, _)| !k.eq_ignore_ascii_case("content-encoding")),
            "content-encoding must be stripped from a decoded-body cache entry"
        );
        assert!(
            entry
                .headers
                .iter()
                .all(|(k, _)| !k.eq_ignore_ascii_case("transfer-encoding")),
            "transfer-encoding must be stripped"
        );
        let cl = entry
            .headers
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case("content-length"))
            .map(|(_, v)| v.clone());
        assert_eq!(
            cl,
            Some(decoded.len().to_string()),
            "content-length must match the stored decoded body"
        );
        assert_eq!(entry.body, decoded);
        // Content-Type and Cache-Control survive normalization.
        assert!(entry.headers.iter().any(|(k, _)| k.eq_ignore_ascii_case("content-type")));
    }
}
