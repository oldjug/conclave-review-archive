//! HTTP cookies — RFC 6265bis-shaped.
//!
//! Scope: parse `Set-Cookie` response headers, store them in a
//! thread-safe jar, and emit a single `Cookie:` request header
//! containing every cookie whose Domain/Path/Secure attributes admit the
//! current request. Persistent cookies (those with `Expires`/`Max-Age`)
//! survive process exit when a persistence path is attached
//! (see [`CookieJar::with_persistence`]); the jar serializes them to a
//! flat TSV on disk, mirroring the `localStorage.tsv` idiom, and drops
//! any whose expiry is in the past at load time. Session cookies (no
//! expiry) stay in-memory only, per spec.
//!
//! We enforce the `__Host-` / `__Secure-` cookie name prefixes
//! (RFC 6265bis §5.7 steps 20-21; see [`parse_set_cookie`]).
//!
//! What we don't do yet:
//!   * Public-suffix-list check (a malicious server could set a cookie
//!     against `.co.uk`). Handled by `psl` for the common cases.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

/// Whether the outgoing request's URL site matches the cookie's origin
/// site. Used by `cookie_header_for_request` to enforce SameSite per
/// RFC 6265bis §5.4.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RequestSite {
    /// The URL is same-site with the cookie's origin (registrable
    /// domain match + same scheme family).
    SameSite,
    /// The URL is cross-site from the cookie's origin.
    CrossSite,
}

/// Decide whether a cookie's SameSite policy permits it on a given
/// request shape per RFC 6265bis §5.4.
fn same_site_permits(
    policy: SameSite,
    site: RequestSite,
    is_top_level_nav: bool,
    is_https: bool,
) -> bool {
    match policy {
        SameSite::Strict => site == RequestSite::SameSite,
        // Modern browsers default unspecified cookies to "Lax".
        SameSite::Lax | SameSite::Unset => match site {
            RequestSite::SameSite => true,
            RequestSite::CrossSite => is_top_level_nav,
        },
        // `None` requires Secure — refuse on plain http even though
        // the cookie was authored to allow cross-site use.
        SameSite::None => is_https,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SameSite {
    Strict,
    Lax,
    None,
    Unset,
}

#[derive(Debug, Clone)]
struct StoredCookie {
    name: String,
    value: String,
    /// Effective domain. For a host-only cookie (no Domain attr), this is
    /// the exact request host. Otherwise it's the Domain attribute value
    /// with any leading '.' stripped.
    domain: String,
    /// `true` iff Set-Cookie omitted a Domain attribute — only the exact
    /// host can match this cookie.
    host_only: bool,
    path: String,
    /// Absolute expiry time, as an `Instant`. None means session cookie
    /// (kept until process exit). Drives the in-memory expiry sweep.
    expires_at: Option<Instant>,
    /// Absolute expiry time as Unix seconds. Mirrors `expires_at` but in a
    /// process-independent clock so it can be serialized to disk and
    /// re-evaluated on a later launch. `None` ⇔ session cookie ⇔
    /// `expires_at` is `None` (the two are kept in lock-step).
    expires_unix: Option<u64>,
    secure: bool,
    http_only: bool,
    same_site: SameSite,
}

#[derive(Debug, Default)]
struct CookieStore {
    /// Order matters for the wire format only — RFC 6265 §5.4 says emit
    /// longer Path first. We keep insertion order here and sort at
    /// emission time. Replaces use (name, domain, path) as the identity.
    cookies: Vec<StoredCookie>,
    /// When set, persistent cookies are mirrored to this TSV file on every
    /// change. `None` ⇒ in-memory only (the historical behavior, and what
    /// tests that don't opt into persistence get).
    persist_path: Option<PathBuf>,
}

/// Thread-safe shared cookie jar. Cheap to clone — backed by `Arc<Mutex>`.
#[derive(Debug, Clone)]
pub struct CookieJar {
    inner: Arc<Mutex<CookieStore>>,
}

impl Default for CookieJar {
    fn default() -> Self {
        Self::new()
    }
}

impl CookieJar {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(CookieStore::default())),
        }
    }

    /// Build a jar that mirrors its persistent cookies to `path` (a flat
    /// TSV, written atomically via write-then-rename). On construction the
    /// jar is seeded from `path`, dropping any cookie whose expiry is in
    /// the past — expired cookies are never loaded and never sent. Session
    /// cookies (no expiry) are not stored on disk, so a fresh jar holds
    /// only the persistent cookies from the previous run.
    ///
    /// The returned jar behaves identically to [`CookieJar::new`] for all
    /// in-memory operations; persistence is purely additive.
    pub fn with_persistence(path: PathBuf) -> Self {
        let mut store = CookieStore::default();
        load_from_disk(&mut store, &path);
        store.persist_path = Some(path);
        let jar = Self {
            inner: Arc::new(Mutex::new(store)),
        };
        // Loading may have dropped expired rows; rewrite the file so it
        // reflects only live cookies (and so a corrupt line is healed).
        {
            let store = jar.inner.lock().unwrap();
            flush_to_disk(&store);
        }
        jar
    }

    /// Number of cookies in the jar (after expiry sweep). Mostly for tests.
    pub fn len(&self) -> usize {
        let mut store = self.inner.lock().unwrap();
        sweep_expired(&mut store);
        store.cookies.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Ingest every `Set-Cookie` header in a response, scoped to the
    /// request that produced it. `host` is the request host (lowercased
    /// ASCII); `path` is the request path. `is_https` rejects Secure
    /// cookies coming in over plain HTTP.
    pub fn absorb(
        &self,
        response_headers: &[(String, String)],
        host: &str,
        path: &str,
        is_https: bool,
    ) {
        let mut store = self.inner.lock().unwrap();
        for (k, v) in response_headers {
            if !k.eq_ignore_ascii_case("set-cookie") {
                continue;
            }
            if let Some(c) = parse_set_cookie(v, host, path, is_https) {
                upsert(&mut store, c);
            }
        }
        sweep_expired(&mut store);
        flush_to_disk(&store);
    }

    /// Build the value of the `Cookie:` request header for this URL. Returns
    /// `None` if no cookies apply (caller skips the header entirely).
    ///
    /// Variant assuming a same-site, top-level navigation. The new
    /// `cookie_header_for_request` is the spec-aware entry point — this
    /// one stays for callers that don't yet thread a site comparison.
    pub fn cookie_header(&self, host: &str, path: &str, is_https: bool) -> Option<String> {
        self.cookie_header_for_request(host, path, is_https, RequestSite::SameSite, true)
    }

    /// Spec-aware variant that consults SameSite per RFC 6265bis §5.4.
    ///
    /// - `Strict` cookies are sent only on same-site requests.
    /// - `Lax` cookies are sent on same-site requests and on cross-site
    ///   top-level navigations (`is_top_level_nav = true`).
    /// - `None` cookies require `Secure` (https) and may be sent
    ///   cross-site; we still refuse them on plain http.
    /// - `Unset` is treated as `Lax` per the new default in modern browsers.
    pub fn cookie_header_for_request(
        &self,
        host: &str,
        path: &str,
        is_https: bool,
        site: RequestSite,
        is_top_level_nav: bool,
    ) -> Option<String> {
        let mut store = self.inner.lock().unwrap();
        sweep_expired(&mut store);
        let host_lc = host.to_ascii_lowercase();
        let mut matches: Vec<&StoredCookie> = store
            .cookies
            .iter()
            .filter(|c| cookie_applies(c, &host_lc, path, is_https))
            .filter(|c| same_site_permits(c.same_site, site, is_top_level_nav, is_https))
            .collect();
        // RFC 6265 §5.4: longer Path first; ties broken by earlier insertion.
        matches.sort_by(|a, b| b.path.len().cmp(&a.path.len()));
        if matches.is_empty() {
            return None;
        }
        let mut out = String::new();
        for (i, c) in matches.iter().enumerate() {
            if i > 0 {
                out.push_str("; ");
            }
            out.push_str(&c.name);
            out.push('=');
            out.push_str(&c.value);
        }
        Some(out)
    }

    /// Like [`Self::cookie_header`], but EXCLUDES `HttpOnly` cookies. This is
    /// what `document.cookie` must read: per RFC 6265 §5.4 script must NEVER see
    /// HttpOnly cookies (else an XSS can exfiltrate the session cookie). The
    /// HTTP request path keeps using `cookie_header`, so HttpOnly cookies are
    /// still SENT to the server.
    pub fn cookie_header_script_visible(
        &self,
        host: &str,
        path: &str,
        is_https: bool,
    ) -> Option<String> {
        let mut store = self.inner.lock().unwrap();
        sweep_expired(&mut store);
        let host_lc = host.to_ascii_lowercase();
        let mut matches: Vec<&StoredCookie> = store
            .cookies
            .iter()
            .filter(|c| cookie_applies(c, &host_lc, path, is_https))
            .filter(|c| !c.http_only)
            .collect();
        matches.sort_by(|a, b| b.path.len().cmp(&a.path.len()));
        if matches.is_empty() {
            return None;
        }
        let mut out = String::new();
        for (i, c) in matches.iter().enumerate() {
            if i > 0 {
                out.push_str("; ");
            }
            out.push_str(&c.name);
            out.push('=');
            out.push_str(&c.value);
        }
        Some(out)
    }

    /// Drop everything. Used by tests and "Clear browsing data" UI.
    /// Also truncates the on-disk file if persistence is attached.
    pub fn clear(&self) {
        let mut store = self.inner.lock().unwrap();
        store.cookies.clear();
        flush_to_disk(&store);
    }
}

fn upsert(store: &mut CookieStore, c: StoredCookie) {
    // (name, domain, path) is the unique identity per RFC 6265 §5.3.
    if let Some(existing) = store
        .cookies
        .iter_mut()
        .find(|e| e.name == c.name && e.domain == c.domain && e.path == c.path)
    {
        *existing = c;
    } else {
        store.cookies.push(c);
    }
}

fn sweep_expired(store: &mut CookieStore) {
    let now = Instant::now();
    store
        .cookies
        .retain(|c| c.expires_at.map(|t| t > now).unwrap_or(true));
}

/// Current wall-clock time in Unix seconds (0 if the clock is before the
/// epoch, which cannot happen on a sane host).
fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn same_site_to_str(s: SameSite) -> &'static str {
    match s {
        SameSite::Strict => "Strict",
        SameSite::Lax => "Lax",
        SameSite::None => "None",
        SameSite::Unset => "Unset",
    }
}

fn same_site_from_str(s: &str) -> SameSite {
    match s {
        "Strict" => SameSite::Strict,
        "Lax" => SameSite::Lax,
        "None" => SameSite::None,
        _ => SameSite::Unset,
    }
}

/// Escape a field for the flat TSV: backslash, tab and newline become
/// `\\`, `\t`, `\n`. Mirrors the localStorage.tsv escaping so the two
/// files share semantics. A field never contains a literal tab/newline
/// after escaping, so each row is exactly one line of tab-separated
/// fields.
fn escape_field(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\t' => out.push_str("\\t"),
            other => out.push(other),
        }
    }
    out
}

fn unescape_field(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars();
    while let Some(c) = chars.next() {
        if c == '\\' {
            match chars.next() {
                Some('n') => out.push('\n'),
                Some('t') => out.push('\t'),
                Some('\\') => out.push('\\'),
                Some(other) => {
                    out.push('\\');
                    out.push(other);
                }
                None => out.push('\\'),
            }
        } else {
            out.push(c);
        }
    }
    out
}

/// Serialize one persistent cookie to a single TSV row. Fields, in order:
/// name, value, domain, host_only(0/1), path, expires_unix, secure(0/1),
/// http_only(0/1), same_site. Session cookies (no `expires_unix`) are
/// never serialized — callers must skip them.
fn serialize_cookie(c: &StoredCookie) -> Option<String> {
    let exp = c.expires_unix?; // session cookie → not persisted
    Some(format!(
        "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
        escape_field(&c.name),
        escape_field(&c.value),
        escape_field(&c.domain),
        if c.host_only { 1 } else { 0 },
        escape_field(&c.path),
        exp,
        if c.secure { 1 } else { 0 },
        if c.http_only { 1 } else { 0 },
        same_site_to_str(c.same_site),
    ))
}

/// Parse one TSV row written by [`serialize_cookie`]. Returns `None` for a
/// malformed or already-expired row (relative to `now_unix`) so a corrupt
/// or stale line is dropped rather than poisoning the jar.
fn deserialize_cookie(line: &str, now_unix: u64) -> Option<StoredCookie> {
    let mut f = line.split('\t');
    let name = unescape_field(f.next()?);
    let value = unescape_field(f.next()?);
    let domain = unescape_field(f.next()?);
    let host_only = f.next()? == "1";
    let path = unescape_field(f.next()?);
    let expires_unix: u64 = f.next()?.parse().ok()?;
    let secure = f.next()? == "1";
    let http_only = f.next()? == "1";
    let same_site = same_site_from_str(f.next()?);
    if name.is_empty() {
        return None;
    }
    // Honor expiry on load: an expired cookie is never resurrected.
    if expires_unix <= now_unix {
        return None;
    }
    // Reconstruct the relative Instant from the remaining lifetime.
    let remaining = Duration::from_secs(expires_unix - now_unix);
    Some(StoredCookie {
        name,
        value,
        domain,
        host_only,
        path,
        expires_at: Some(Instant::now() + remaining),
        expires_unix: Some(expires_unix),
        secure,
        http_only,
        same_site,
    })
}

/// Load persistent cookies from `path` into `store`, dropping any that are
/// already expired. Missing/unreadable file ⇒ empty jar (not an error).
fn load_from_disk(store: &mut CookieStore, path: &std::path::Path) {
    let data = match std::fs::read_to_string(path) {
        Ok(d) => d,
        Err(_) => return,
    };
    let now = unix_now();
    for line in data.lines() {
        if line.is_empty() {
            continue;
        }
        if let Some(c) = deserialize_cookie(line, now) {
            // Use upsert so a duplicate identity in a partially-written
            // file collapses to the last row, matching live semantics.
            upsert(store, c);
        }
    }
}

/// Mirror the store's persistent cookies to disk. No-op if no persistence
/// path is attached. Session cookies are skipped. Written atomically via a
/// sibling `.tmp` file + rename so a crash mid-write can't corrupt the jar.
fn flush_to_disk(store: &CookieStore) {
    let path = match &store.persist_path {
        Some(p) => p,
        None => return,
    };
    let mut buf = String::new();
    for c in &store.cookies {
        if let Some(row) = serialize_cookie(c) {
            buf.push_str(&row);
            buf.push('\n');
        }
    }
    let mut tmp = path.clone();
    let mut name = tmp
        .file_name()
        .map(|s| s.to_os_string())
        .unwrap_or_default();
    name.push(".tmp");
    tmp.set_file_name(name);
    if std::fs::write(&tmp, &buf).is_ok() {
        let _ = std::fs::rename(&tmp, path);
    }
}

fn cookie_applies(c: &StoredCookie, host: &str, path: &str, is_https: bool) -> bool {
    if c.secure && !is_https {
        return false;
    }
    if !domain_matches(host, &c.domain, c.host_only) {
        return false;
    }
    path_matches(path, &c.path)
}

/// RFC 6265 §5.1.3 — domain matching.
/// `host_only` cookies require an exact match. Otherwise the request host
/// must equal the cookie domain or be a subdomain of it.
fn domain_matches(host: &str, cookie_domain: &str, host_only: bool) -> bool {
    if host == cookie_domain {
        return true;
    }
    if host_only {
        return false;
    }
    // Subdomain match: host ends with "." + cookie_domain, and the host
    // is a DNS hostname (not an IP literal). We don't IP-check yet — a
    // numeric host setting Domain=. would be a server bug anyway.
    host.len() > cookie_domain.len()
        && host.ends_with(cookie_domain)
        && host.as_bytes()[host.len() - cookie_domain.len() - 1] == b'.'
}

/// RFC 6265 §5.1.4 — path matching.
fn path_matches(request_path: &str, cookie_path: &str) -> bool {
    if request_path == cookie_path {
        return true;
    }
    if !request_path.starts_with(cookie_path) {
        return false;
    }
    // After the prefix, either the cookie path already ends in '/' or
    // the request path's next character is '/'.
    cookie_path.ends_with('/')
        || request_path
            .as_bytes()
            .get(cookie_path.len())
            .map(|b| *b == b'/')
            .unwrap_or(false)
}

/// Default-Path algorithm — RFC 6265 §5.1.4.
fn default_path(request_path: &str) -> String {
    if !request_path.starts_with('/') {
        return "/".into();
    }
    // Trim from the last '/'. If only the leading '/' remains, default
    // path is "/". Otherwise it's everything up to (but not including)
    // the last slash.
    let last = request_path.rfind('/').unwrap_or(0);
    if last == 0 {
        return "/".into();
    }
    request_path[..last].to_string()
}

/// ASCII case-insensitive `starts_with`, used for cookie name-prefix
/// matching (RFC 6265bis §5.7 specifies a "case-insensitive match", and
/// Chrome's `GetCookiePrefix` uses `base::CompareCase::INSENSITIVE_ASCII`).
fn starts_with_ascii_ci(s: &str, prefix: &str) -> bool {
    s.len() >= prefix.len() && s.as_bytes()[..prefix.len()].eq_ignore_ascii_case(prefix.as_bytes())
}

fn parse_set_cookie(
    value: &str,
    request_host: &str,
    request_path: &str,
    is_https: bool,
) -> Option<StoredCookie> {
    let mut parts = value.split(';');
    let nv = parts.next()?.trim();
    let (name, val) = nv.split_once('=')?;
    let name = name.trim();
    let val = val.trim();
    if name.is_empty() {
        return None;
    }

    let mut domain_attr: Option<String> = None;
    let mut path_attr: Option<String> = None;
    let mut max_age_attr: Option<i64> = None;
    let mut expires_attr_at: Option<SystemTime> = None;
    let mut secure = false;
    let mut http_only = false;
    let mut same_site = SameSite::Unset;

    for attr in parts {
        let attr = attr.trim();
        if attr.is_empty() {
            continue;
        }
        let (k, v) = match attr.split_once('=') {
            Some((k, v)) => (k.trim(), Some(v.trim())),
            None => (attr, None),
        };
        match k.to_ascii_lowercase().as_str() {
            "domain" => {
                if let Some(v) = v {
                    let trimmed = v.trim_start_matches('.').to_ascii_lowercase();
                    if !trimmed.is_empty() {
                        domain_attr = Some(trimmed);
                    }
                }
            }
            "path" => {
                if let Some(v) = v {
                    if v.starts_with('/') {
                        path_attr = Some(v.to_string());
                    }
                }
            }
            "max-age" => {
                if let Some(v) = v {
                    if let Ok(n) = v.parse::<i64>() {
                        max_age_attr = Some(n);
                    }
                }
            }
            "expires" => {
                // RFC 6265 §5.1.1 — parse the value as an HTTP-date.
                // All three RFC 7231 formats accepted: IMF-fixdate,
                // obsolete RFC-850, and asctime. Failure leaves the
                // attribute unset (cookie degrades to session cookie).
                if let Some(v) = v {
                    if let Some(t) = parse_http_date(v) {
                        expires_attr_at = Some(t);
                    }
                }
            }
            "secure" => secure = true,
            "httponly" => http_only = true,
            "samesite" => {
                if let Some(v) = v {
                    same_site = match v.to_ascii_lowercase().as_str() {
                        "strict" => SameSite::Strict,
                        "lax" => SameSite::Lax,
                        "none" => SameSite::None,
                        _ => SameSite::Unset,
                    };
                }
            }
            _ => {}
        }
    }

    // Resolve domain. A Domain attr that doesn't domain-match the request
    // host is rejected (per §5.3 step 6). Otherwise host-only is implied.
    let host_lc = request_host.to_ascii_lowercase();
    let (domain, host_only) = match domain_attr {
        Some(d) => {
            if !domain_matches(&host_lc, &d, false) && host_lc != d {
                return None;
            }
            // RFC 6265bis §5.3 step 5: a Domain attribute that is a public suffix
            // is a super-cookie shared across every unrelated site under it
            // (e.g. Domain=co.uk, or Domain=github.io). Reject it — unless it is
            // exactly the request host, in which case it degrades to host-only.
            if crate::psl::public_suffix(&d) == d {
                if host_lc == d {
                    (host_lc, true)
                } else {
                    return None;
                }
            } else {
                (d, false)
            }
        }
        None => (host_lc, true),
    };

    // Secure flag from a non-https origin is rejected. (RFC 6265bis §5.6
    // step 11 — Set-Cookie over plain HTTP can't set Secure cookies.)
    if secure && !is_https {
        return None;
    }

    let path = path_attr.unwrap_or_else(|| default_path(request_path));

    // Cookie name-prefix enforcement (RFC 6265bis §5.7 steps 20-21,
    // Chrome net/cookies/cookie_util.cc `IsCookiePrefixValid` /
    // `HasValidSecurePrefixAttributes` / `HasValidHostPrefixAttributes`).
    //
    // The prefix match is CASE-INSENSITIVE: the spec says "a case-insensitive
    // match for the string", and current Chrome's `GetCookiePrefix` uses
    // `base::CompareCase::INSENSITIVE_ASCII`. (We deliberately follow the
    // spec + live Chrome source here.)
    //
    //   * `__Secure-` requires the Secure attribute (and, since Secure from a
    //     non-https origin is already rejected above, a cryptographic origin —
    //     mirroring `HasValidSecurePrefixAttributes`, which demands
    //     `secure && ProvisionalAccessScheme(url) != kNonCryptographic`).
    //   * `__Host-` additionally requires the cookie to be host-only (no Domain
    //     attribute that broadens scope) and the resolved Path to be exactly
    //     "/". Chrome's `HasValidHostPrefixAttributes` compares the *resolved*
    //     path to "/" (`path != "/"`), not whether a Path attribute was given,
    //     so we compare the resolved `path` here. Our `host_only` flag is the
    //     analogue of Chrome's `domain.empty()` host-only requirement (a Domain
    //     attribute that exactly equals a public-suffix host degraded to
    //     host-only upstream, matching Chrome's IP-literal carve-out closely
    //     enough for the host-only-flag check).
    //
    // On any violation the cookie is ignored entirely (not stored).
    if starts_with_ascii_ci(name, "__Secure-") {
        // Step 20: ignore unless secure-only-flag is true. (is_https is
        // implied because `secure && !is_https` was already rejected.)
        if !secure {
            return None;
        }
    }
    if starts_with_ascii_ci(name, "__Host-") {
        // Step 21: ignore unless secure-only-flag && host-only-flag && path=="/".
        if !secure || !host_only || path != "/" {
            return None;
        }
    }

    // Expiry: Max-Age wins over Expires per spec. A negative or zero
    // Max-Age means "expire immediately". We compute the relative
    // `Instant` (drives the in-memory sweep) and the absolute Unix-second
    // expiry (drives serialization + cross-launch expiry) in lock-step.
    let now_unix = unix_now();
    let (expires_at, expires_unix) = if let Some(n) = max_age_attr {
        if n <= 0 {
            // Past-dated cookies are still inserted so they can replace
            // an existing live cookie of the same identity; sweep removes
            // them immediately after. The absolute time is in the past too
            // so a reload would never resurrect it.
            (
                Some(Instant::now() - Duration::from_secs(1)),
                Some(now_unix.saturating_sub(1)),
            )
        } else {
            (
                Some(Instant::now() + Duration::from_secs(n as u64)),
                Some(now_unix.saturating_add(n as u64)),
            )
        }
    } else if let Some(target) = expires_attr_at {
        // Translate absolute SystemTime → relative Instant. If the
        // expiry is already in the past, mark the cookie expired so
        // sweep removes it (and any same-identity predecessor).
        let abs_unix = target
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        match target.duration_since(SystemTime::now()) {
            Ok(d) => (Some(Instant::now() + d), Some(abs_unix)),
            Err(_) => (
                Some(Instant::now() - Duration::from_secs(1)),
                Some(abs_unix.min(now_unix.saturating_sub(1))),
            ),
        }
    } else {
        (None, None)
    };

    Some(StoredCookie {
        name: name.to_string(),
        value: val.to_string(),
        domain,
        host_only,
        path,
        expires_at,
        expires_unix,
        secure,
        http_only,
        same_site,
    })
}

/// Parse an HTTP-date per RFC 7231 §7.1.1.1 into a `SystemTime`.
/// Accepts the three forms:
///
/// * IMF-fixdate:  `Sun, 06 Nov 1994 08:49:37 GMT`
/// * RFC-850:      `Sunday, 06-Nov-94 08:49:37 GMT`
/// * asctime:      `Sun Nov  6 08:49:37 1994`
///
/// Day-of-week is parsed but not validated against the date. Time zone
/// must be GMT/UTC (the only legal value for HTTP-date in modern specs).
fn parse_http_date(s: &str) -> Option<SystemTime> {
    let s = s.trim();
    // Try each form in turn. They're disambiguated by the punctuation in
    // their date component.
    parse_imf_fixdate(s)
        .or_else(|| parse_rfc850(s))
        .or_else(|| parse_asctime(s))
}

/// `Sun, 06 Nov 1994 08:49:37 GMT`
fn parse_imf_fixdate(s: &str) -> Option<SystemTime> {
    let (_dow, rest) = split_after_comma(s)?;
    let rest = rest.trim_start();
    let mut bits = rest.split_ascii_whitespace();
    let day: u32 = bits.next()?.parse().ok()?;
    let month = parse_month_abbrev(bits.next()?)?;
    let year: i32 = bits.next()?.parse().ok()?;
    let time = bits.next()?;
    let zone = bits.next().unwrap_or("GMT");
    if !zone.eq_ignore_ascii_case("GMT") && !zone.eq_ignore_ascii_case("UTC") {
        return None;
    }
    let (h, m, sec) = parse_hms(time)?;
    civil_to_system_time(year, month, day, h, m, sec)
}

/// `Sunday, 06-Nov-94 08:49:37 GMT` — two-digit year, dash-separated date.
fn parse_rfc850(s: &str) -> Option<SystemTime> {
    let (_dow, rest) = split_after_comma(s)?;
    let rest = rest.trim_start();
    let mut bits = rest.split_ascii_whitespace();
    let date = bits.next()?;
    let time = bits.next()?;
    let zone = bits.next().unwrap_or("GMT");
    if !zone.eq_ignore_ascii_case("GMT") && !zone.eq_ignore_ascii_case("UTC") {
        return None;
    }
    let mut date_parts = date.split('-');
    let day: u32 = date_parts.next()?.parse().ok()?;
    let month = parse_month_abbrev(date_parts.next()?)?;
    let yy: i32 = date_parts.next()?.parse().ok()?;
    // Two-digit year: per RFC 6265 §5.1.1 step 3, years 0..69 are 2000..2069,
    // 70..99 are 1970..1999.
    let year = if yy < 70 { 2000 + yy } else { 1900 + yy };
    let (h, m, sec) = parse_hms(time)?;
    civil_to_system_time(year, month, day, h, m, sec)
}

/// `Sun Nov  6 08:49:37 1994` — note possible double-space before single-digit day.
fn parse_asctime(s: &str) -> Option<SystemTime> {
    let mut bits = s.split_ascii_whitespace();
    let _dow = bits.next()?;
    let month = parse_month_abbrev(bits.next()?)?;
    let day: u32 = bits.next()?.parse().ok()?;
    let time = bits.next()?;
    let year: i32 = bits.next()?.parse().ok()?;
    let (h, m, sec) = parse_hms(time)?;
    civil_to_system_time(year, month, day, h, m, sec)
}

fn split_after_comma(s: &str) -> Option<(&str, &str)> {
    let i = s.find(',')?;
    Some((&s[..i], &s[i + 1..]))
}

fn parse_month_abbrev(s: &str) -> Option<u32> {
    Some(match s.to_ascii_lowercase().as_str() {
        "jan" => 1,
        "feb" => 2,
        "mar" => 3,
        "apr" => 4,
        "may" => 5,
        "jun" => 6,
        "jul" => 7,
        "aug" => 8,
        "sep" => 9,
        "oct" => 10,
        "nov" => 11,
        "dec" => 12,
        _ => return None,
    })
}

fn parse_hms(s: &str) -> Option<(u32, u32, u32)> {
    let mut parts = s.split(':');
    let h: u32 = parts.next()?.parse().ok()?;
    let m: u32 = parts.next()?.parse().ok()?;
    let sec: u32 = parts.next()?.parse().ok()?;
    if h > 23 || m > 59 || sec > 60 {
        return None;
    }
    Some((h, m, sec))
}

/// Convert a (year, month, day, hour, minute, second) UTC civil date
/// to a `SystemTime`. Uses Howard Hinnant's days-from-civil algorithm —
/// overflows for years outside roughly ±5 million but exact for every
/// value HTTP cares about.
fn civil_to_system_time(
    year: i32,
    month: u32,
    day: u32,
    hour: u32,
    minute: u32,
    second: u32,
) -> Option<SystemTime> {
    if !(1..=12).contains(&month) || !(1..=31).contains(&day) {
        return None;
    }
    // Shift so March is month 1 — keeps Feb at the *end* of the prior
    // year so the leap-day edge cases dissolve.
    let shifted_year = if month <= 2 { year - 1 } else { year };
    let shifted_month: u32 = if month <= 2 { month + 9 } else { month - 3 };
    let era = if shifted_year >= 0 {
        shifted_year / 400
    } else {
        (shifted_year - 399) / 400
    };
    let yoe = (shifted_year - era * 400) as u32; // [0, 399]
    let doy = (153 * shifted_month + 2) / 5 + day - 1; // [0, 365]
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy; // [0, 146096]
    let days_from_epoch = era as i64 * 146097 + doe as i64 - 719468;
    let total_secs =
        days_from_epoch * 86_400 + hour as i64 * 3_600 + minute as i64 * 60 + second as i64;
    Some(if total_secs >= 0 {
        UNIX_EPOCH + Duration::from_secs(total_secs as u64)
    } else {
        UNIX_EPOCH - Duration::from_secs((-total_secs) as u64)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_basic_set_cookie() {
        let jar = CookieJar::new();
        jar.absorb(
            &[("Set-Cookie".into(), "sid=abc123; Path=/; HttpOnly".into())],
            "example.com",
            "/login",
            true,
        );
        assert_eq!(jar.len(), 1);
        let h = jar
            .cookie_header("example.com", "/dashboard", true)
            .unwrap();
        assert_eq!(h, "sid=abc123");
    }

    #[test]
    fn host_only_does_not_leak_to_subdomain() {
        let jar = CookieJar::new();
        jar.absorb(
            &[("Set-Cookie".into(), "a=1".into())],
            "example.com",
            "/",
            true,
        );
        // Host-only cookie must NOT match a subdomain.
        assert!(jar.cookie_header("www.example.com", "/", true).is_none());
        assert!(jar.cookie_header("example.com", "/", true).is_some());
    }

    #[test]
    fn domain_attr_admits_subdomain() {
        let jar = CookieJar::new();
        jar.absorb(
            &[("Set-Cookie".into(), "a=1; Domain=example.com".into())],
            "example.com",
            "/",
            true,
        );
        assert!(jar.cookie_header("www.example.com", "/", true).is_some());
        assert!(jar.cookie_header("evil.com", "/", true).is_none());
    }

    #[test]
    fn rejects_domain_not_matching_origin() {
        let jar = CookieJar::new();
        jar.absorb(
            &[("Set-Cookie".into(), "a=1; Domain=other.com".into())],
            "example.com",
            "/",
            true,
        );
        assert_eq!(jar.len(), 0);
    }

    #[test]
    fn rejects_public_suffix_domain_supercookie() {
        // Domain=co.uk from a co.uk site is a super-cookie shared across every
        // unrelated co.uk site → must be rejected (PSL, RFC 6265bis §5.3 step 5).
        let jar = CookieJar::new();
        jar.absorb(
            &[("Set-Cookie".into(), "a=1; Domain=co.uk".into())],
            "example.co.uk",
            "/",
            true,
        );
        assert_eq!(jar.len(), 0, "Domain=co.uk (public suffix) must be rejected");

        // Two GitHub Pages users must not be able to set a shared github.io cookie.
        let jar2 = CookieJar::new();
        jar2.absorb(
            &[("Set-Cookie".into(), "a=1; Domain=github.io".into())],
            "alice.github.io",
            "/",
            true,
        );
        assert_eq!(jar2.len(), 0, "Domain=github.io (public suffix) must be rejected");

        // A genuinely registrable Domain attribute is still accepted.
        let jar3 = CookieJar::new();
        jar3.absorb(
            &[("Set-Cookie".into(), "a=1; Domain=example.co.uk".into())],
            "www.example.co.uk",
            "/",
            true,
        );
        assert_eq!(jar3.len(), 1, "Domain=example.co.uk (registrable) is allowed");
    }

    #[test]
    fn secure_cookie_does_not_send_over_http() {
        let jar = CookieJar::new();
        jar.absorb(
            &[("Set-Cookie".into(), "a=1; Secure".into())],
            "example.com",
            "/",
            true,
        );
        assert!(jar.cookie_header("example.com", "/", false).is_none());
        assert!(jar.cookie_header("example.com", "/", true).is_some());
    }

    #[test]
    fn secure_attr_rejected_from_plain_http() {
        let jar = CookieJar::new();
        jar.absorb(
            &[("Set-Cookie".into(), "a=1; Secure".into())],
            "example.com",
            "/",
            false,
        );
        assert_eq!(jar.len(), 0);
    }

    #[test]
    fn max_age_zero_expires_immediately() {
        let jar = CookieJar::new();
        jar.absorb(
            &[("Set-Cookie".into(), "a=1".into())],
            "example.com",
            "/",
            true,
        );
        assert_eq!(jar.len(), 1);
        jar.absorb(
            &[("Set-Cookie".into(), "a=expired; Max-Age=0".into())],
            "example.com",
            "/",
            true,
        );
        assert_eq!(jar.len(), 0);
    }

    #[test]
    fn longer_path_first_in_header() {
        let jar = CookieJar::new();
        jar.absorb(
            &[("Set-Cookie".into(), "a=root; Path=/".into())],
            "example.com",
            "/",
            true,
        );
        jar.absorb(
            &[("Set-Cookie".into(), "b=deep; Path=/x/y".into())],
            "example.com",
            "/x/y",
            true,
        );
        let h = jar.cookie_header("example.com", "/x/y/z", true).unwrap();
        // /x/y is more specific → comes first.
        assert!(h.starts_with("b=deep"));
        assert!(h.contains("a=root"));
    }

    #[test]
    fn path_match_respects_segment_boundary() {
        assert!(path_matches("/foo", "/foo"));
        assert!(path_matches("/foo/bar", "/foo"));
        assert!(path_matches("/foo/bar", "/foo/"));
        // /foobar does NOT match Path=/foo per §5.1.4
        assert!(!path_matches("/foobar", "/foo"));
    }

    #[test]
    fn default_path_strips_filename() {
        assert_eq!(default_path("/foo/bar"), "/foo");
        assert_eq!(default_path("/foo"), "/");
        assert_eq!(default_path("/"), "/");
    }

    #[test]
    fn http_date_imf_fixdate_parses() {
        // Canonical example from RFC 7231 §7.1.1.1.
        let t = parse_http_date("Sun, 06 Nov 1994 08:49:37 GMT").unwrap();
        let secs = t.duration_since(UNIX_EPOCH).unwrap().as_secs();
        assert_eq!(secs, 784_111_777);
    }

    #[test]
    fn http_date_rfc850_parses_two_digit_year() {
        let t = parse_http_date("Sunday, 06-Nov-94 08:49:37 GMT").unwrap();
        let secs = t.duration_since(UNIX_EPOCH).unwrap().as_secs();
        assert_eq!(secs, 784_111_777);
    }

    #[test]
    fn http_date_asctime_parses_with_padded_day() {
        // asctime form has a leading space before single-digit days; the
        // ASCII-whitespace split eats it transparently.
        let t = parse_http_date("Sun Nov  6 08:49:37 1994").unwrap();
        let secs = t.duration_since(UNIX_EPOCH).unwrap().as_secs();
        assert_eq!(secs, 784_111_777);
    }

    #[test]
    fn cookie_expires_in_past_does_not_persist() {
        let jar = CookieJar::new();
        jar.absorb(
            &[(
                "Set-Cookie".into(),
                "a=1; Expires=Sun, 06 Nov 1994 08:49:37 GMT".into(),
            )],
            "example.com",
            "/",
            true,
        );
        assert_eq!(jar.len(), 0);
    }

    #[test]
    fn cookie_expires_far_future_is_kept() {
        // Year 3000 — far past anything realistic, but unambiguously
        // future and tests the four-digit year branch end-to-end.
        let jar = CookieJar::new();
        jar.absorb(
            &[(
                "Set-Cookie".into(),
                "a=1; Expires=Sat, 06 Nov 3000 08:49:37 GMT".into(),
            )],
            "example.com",
            "/",
            true,
        );
        assert_eq!(jar.len(), 1);
    }

    #[test]
    fn rejects_set_cookie_without_equals() {
        let jar = CookieJar::new();
        jar.absorb(
            &[("Set-Cookie".into(), "garbage".into())],
            "example.com",
            "/",
            true,
        );
        assert_eq!(jar.len(), 0);
    }

    // ---- persistence ----------------------------------------------------

    /// A unique temp file path that does NOT touch the real %APPDATA%
    /// cookies.tsv. Uses the process id + an atomic counter so parallel
    /// test threads never collide.
    fn temp_cookie_path(tag: &str) -> PathBuf {
        use std::sync::atomic::{AtomicU64, Ordering};
        static N: AtomicU64 = AtomicU64::new(0);
        let n = N.fetch_add(1, Ordering::Relaxed);
        let mut p = std::env::temp_dir();
        p.push(format!(
            "tb_cookies_test_{}_{}_{}.tsv",
            tag,
            std::process::id(),
            n
        ));
        let _ = std::fs::remove_file(&p);
        p
    }

    #[test]
    fn persistent_cookie_round_trips_across_jars() {
        let path = temp_cookie_path("roundtrip");
        {
            let jar = CookieJar::with_persistence(path.clone());
            // Two long-lived persistent cookies + one already-expired one.
            jar.absorb(
                &[
                    ("Set-Cookie".into(), "alive=1; Max-Age=86400; Path=/".into()),
                    (
                        "Set-Cookie".into(),
                        "secured=2; Max-Age=86400; Secure; HttpOnly; SameSite=Strict".into(),
                    ),
                    ("Set-Cookie".into(), "dead=3; Max-Age=86400".into()),
                ],
                "example.com",
                "/",
                true,
            );
            // Now expire `dead` by re-setting it with Max-Age=0.
            jar.absorb(
                &[("Set-Cookie".into(), "dead=3; Max-Age=0".into())],
                "example.com",
                "/",
                true,
            );
            assert_eq!(jar.len(), 2);
        }
        // A brand-new jar over the same file rehydrates the survivors.
        let jar2 = CookieJar::with_persistence(path.clone());
        assert_eq!(jar2.len(), 2, "non-expired persistent cookies survive");
        // Flags are preserved: HttpOnly hides `secured` from script view.
        let script = jar2
            .cookie_header_script_visible("example.com", "/", true)
            .unwrap_or_default();
        assert!(script.contains("alive=1"));
        assert!(
            !script.contains("secured=2"),
            "HttpOnly preserved across reload"
        );
        // Secure cookie not sent over plain http; sent over https.
        let http = jar2
            .cookie_header("example.com", "/", false)
            .unwrap_or_default();
        assert!(!http.contains("secured=2"), "Secure preserved across reload");
        let https = jar2
            .cookie_header("example.com", "/", true)
            .unwrap_or_default();
        assert!(https.contains("secured=2"));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn expired_cookie_is_dropped_on_load() {
        // Hand-craft a file with one future and one past row, simulating a
        // jar persisted earlier whose clock has since advanced.
        let path = temp_cookie_path("expiry_load");
        let now = unix_now();
        let mut buf = String::new();
        // future
        buf.push_str(&format!("future\tF\texample.com\t1\t/\t{}\t0\t0\tUnset\n", now + 100_000));
        // past
        buf.push_str(&format!("past\tP\texample.com\t1\t/\t{}\t0\t0\tUnset\n", now.saturating_sub(10)));
        std::fs::write(&path, buf).unwrap();

        let jar = CookieJar::with_persistence(path.clone());
        assert_eq!(jar.len(), 1, "expired row dropped at load");
        let h = jar.cookie_header("example.com", "/", true).unwrap_or_default();
        assert!(h.contains("future=F"));
        assert!(!h.contains("past=P"), "expired cookie is never sent");
        // The file is rewritten on load to drop the stale row.
        let on_disk = std::fs::read_to_string(&path).unwrap();
        assert!(on_disk.contains("future\t"));
        assert!(!on_disk.contains("past\t"));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn max_age_zero_deletes_from_disk() {
        let path = temp_cookie_path("maxage0");
        let jar = CookieJar::with_persistence(path.clone());
        jar.absorb(
            &[("Set-Cookie".into(), "a=1; Max-Age=86400".into())],
            "example.com",
            "/",
            true,
        );
        assert!(std::fs::read_to_string(&path).unwrap().contains("a\t1\t"));
        // Max-Age=0 deletes the cookie and removes it from disk.
        jar.absorb(
            &[("Set-Cookie".into(), "a=1; Max-Age=0".into())],
            "example.com",
            "/",
            true,
        );
        assert_eq!(jar.len(), 0);
        let on_disk = std::fs::read_to_string(&path).unwrap();
        assert!(
            !on_disk.contains("a\t1\t"),
            "Max-Age=0 removes the cookie from disk"
        );
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn past_expires_deletes_from_disk() {
        let path = temp_cookie_path("pastexp");
        let jar = CookieJar::with_persistence(path.clone());
        jar.absorb(
            &[(
                "Set-Cookie".into(),
                "a=1; Expires=Sat, 06 Nov 3000 08:49:37 GMT".into(),
            )],
            "example.com",
            "/",
            true,
        );
        assert_eq!(jar.len(), 1);
        // A past Expires for the same identity deletes it (and from disk).
        jar.absorb(
            &[(
                "Set-Cookie".into(),
                "a=1; Expires=Sun, 06 Nov 1994 08:49:37 GMT".into(),
            )],
            "example.com",
            "/",
            true,
        );
        assert_eq!(jar.len(), 0);
        assert!(!std::fs::read_to_string(&path).unwrap().contains("a\t1\t"));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn session_cookie_is_not_persisted_to_disk() {
        let path = temp_cookie_path("session");
        {
            let jar = CookieJar::with_persistence(path.clone());
            // No Expires / Max-Age ⇒ session cookie ⇒ in-memory only.
            jar.absorb(
                &[("Set-Cookie".into(), "sess=xyz; Path=/".into())],
                "example.com",
                "/",
                true,
            );
            // It IS live in this process.
            assert_eq!(jar.len(), 1);
            let h = jar.cookie_header("example.com", "/", true).unwrap();
            assert_eq!(h, "sess=xyz");
            // But it was NOT written to disk.
            let on_disk = std::fs::read_to_string(&path).unwrap_or_default();
            assert!(
                !on_disk.contains("sess"),
                "session cookie must not be persisted"
            );
        }
        // A new jar over the same file has no session cookie.
        let jar2 = CookieJar::with_persistence(path.clone());
        assert_eq!(jar2.len(), 0, "session cookie does not survive process exit");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn persistence_preserves_value_with_special_chars() {
        // A value containing characters the TSV escaping must survive.
        let path = temp_cookie_path("escape");
        {
            let jar = CookieJar::with_persistence(path.clone());
            jar.absorb(
                &[(
                    "Set-Cookie".into(),
                    "weird=a\\b; Max-Age=86400".into(),
                )],
                "example.com",
                "/",
                true,
            );
        }
        let jar2 = CookieJar::with_persistence(path.clone());
        let h = jar2.cookie_header("example.com", "/", true).unwrap();
        assert_eq!(h, "weird=a\\b", "backslash survives the TSV round trip");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn no_persistence_jar_writes_nothing() {
        // The default jar (no path) must not regress to writing files and
        // behaves exactly as before.
        let jar = CookieJar::new();
        jar.absorb(
            &[("Set-Cookie".into(), "a=1; Max-Age=86400".into())],
            "example.com",
            "/",
            true,
        );
        assert_eq!(jar.len(), 1);
        let h = jar.cookie_header("example.com", "/", true).unwrap();
        assert_eq!(h, "a=1");
    }

    // --- Cookie name-prefix enforcement (RFC 6265bis §5.7 steps 20-21;
    //     Chrome net/cookies/cookie_util.cc IsCookiePrefixValid) ---

    #[test]
    fn starts_with_ascii_ci_helper() {
        assert!(starts_with_ascii_ci("__Host-x", "__Host-"));
        assert!(starts_with_ascii_ci("__HOST-x", "__Host-"));
        assert!(starts_with_ascii_ci("__host-x", "__Host-"));
        assert!(starts_with_ascii_ci("__SeCuRe-x", "__Secure-"));
        assert!(!starts_with_ascii_ci("__Hos", "__Host-")); // shorter than prefix
        assert!(!starts_with_ascii_ci("X__Host-", "__Host-")); // not a prefix
    }

    #[test]
    fn host_prefix_with_domain_attr_rejected() {
        // __Host- forbids any Domain attribute (host-only-flag must be true).
        let c = parse_set_cookie(
            "__Host-id=1; Secure; Path=/; Domain=example.com",
            "example.com",
            "/",
            true,
        );
        assert!(c.is_none(), "__Host- with Domain must be rejected");
        let jar = CookieJar::new();
        jar.absorb(
            &[(
                "Set-Cookie".into(),
                "__Host-id=1; Secure; Path=/; Domain=example.com".into(),
            )],
            "example.com",
            "/",
            true,
        );
        assert_eq!(jar.len(), 0);
    }

    #[test]
    fn host_prefix_without_secure_rejected() {
        // Even over https, no Secure attribute => reject.
        let c = parse_set_cookie("__Host-id=1; Path=/", "example.com", "/", true);
        assert!(c.is_none(), "__Host- without Secure must be rejected");
    }

    #[test]
    fn host_prefix_with_non_root_path_rejected() {
        // Resolved path must be exactly "/".
        let c = parse_set_cookie(
            "__Host-id=1; Secure; Path=/foo",
            "example.com",
            "/foo",
            true,
        );
        assert!(c.is_none(), "__Host- with Path=/foo must be rejected");
    }

    #[test]
    fn host_prefix_with_implicit_non_root_path_rejected() {
        // No explicit Path attribute: the default-path is derived from the
        // request path. Chrome compares the *resolved* path to "/", so a
        // request like /foo/bar (default-path "/foo") must also be rejected.
        let c = parse_set_cookie("__Host-id=1; Secure", "example.com", "/foo/bar", true);
        assert!(
            c.is_none(),
            "__Host- whose resolved (default) path is not / must be rejected"
        );
    }

    #[test]
    fn valid_host_prefix_accepted_host_only() {
        // Secure + no Domain + Path=/ over https => accepted, and host-only.
        let c = parse_set_cookie("__Host-id=abc; Secure; Path=/", "example.com", "/", true)
            .expect("valid __Host- cookie must be accepted");
        assert_eq!(c.name, "__Host-id");
        assert_eq!(c.value, "abc");
        assert!(c.secure, "must carry Secure");
        assert!(c.host_only, "__Host- cookie is host-only");
        assert_eq!(c.path, "/");

        // End-to-end via the jar: sent to the exact host, never to a subdomain.
        let jar = CookieJar::new();
        jar.absorb(
            &[(
                "Set-Cookie".into(),
                "__Host-id=abc; Secure; Path=/".into(),
            )],
            "example.com",
            "/",
            true,
        );
        assert_eq!(jar.len(), 1);
        assert!(
            jar.cookie_header("www.example.com", "/", true).is_none(),
            "host-only __Host- cookie must not leak to a subdomain"
        );
        let h = jar.cookie_header("example.com", "/page", true).unwrap();
        assert_eq!(h, "__Host-id=abc");
    }

    #[test]
    fn host_prefix_accepts_default_root_path_without_explicit_attr() {
        // Chrome checks the resolved path, not whether a Path attribute was
        // given. A request to "/" yields default-path "/", so a __Host-
        // cookie with no explicit Path is valid.
        let c = parse_set_cookie("__Host-id=abc; Secure", "example.com", "/", true)
            .expect("__Host- with default root path must be accepted");
        assert_eq!(c.path, "/");
        assert!(c.host_only);
    }

    #[test]
    fn secure_prefix_without_secure_rejected() {
        // __Secure- requires the Secure attribute.
        let c = parse_set_cookie("__Secure-id=1", "example.com", "/", true);
        assert!(c.is_none(), "__Secure- without Secure must be rejected");
        let jar = CookieJar::new();
        jar.absorb(
            &[("Set-Cookie".into(), "__Secure-id=1".into())],
            "example.com",
            "/",
            true,
        );
        assert_eq!(jar.len(), 0);
    }

    #[test]
    fn secure_prefix_with_secure_over_https_accepted() {
        // __Secure- with Secure over https => accepted (may carry a Domain).
        let c = parse_set_cookie(
            "__Secure-id=v; Secure; Domain=example.com",
            "example.com",
            "/",
            true,
        )
        .expect("valid __Secure- cookie must be accepted");
        assert_eq!(c.name, "__Secure-id");
        assert!(c.secure);
        // Domain is allowed for __Secure- (only __Host- forbids it), so this
        // cookie is NOT host-only.
        assert!(!c.host_only, "__Secure- may scope a Domain");

        let jar = CookieJar::new();
        jar.absorb(
            &[(
                "Set-Cookie".into(),
                "__Secure-id=v; Secure; Domain=example.com".into(),
            )],
            "example.com",
            "/",
            true,
        );
        // Domain-scoped => reaches subdomains.
        assert!(jar.cookie_header("www.example.com", "/", true).is_some());
    }

    #[test]
    fn secure_prefix_over_plain_http_rejected() {
        // A __Secure- cookie can only be set with Secure, and Secure from a
        // non-https origin is itself rejected, so it can never be set on http.
        let c = parse_set_cookie("__Secure-id=1; Secure", "example.com", "/", false);
        assert!(
            c.is_none(),
            "__Secure- (Secure) from plain http must be rejected"
        );
    }

    #[test]
    fn prefix_match_is_case_insensitive_like_chrome() {
        // RFC 6265bis §5.7 + Chrome GetCookiePrefix: case-insensitive match.
        // So "__host-" / "__SECURE-" are also enforced.
        assert!(
            parse_set_cookie("__host-id=1; Domain=example.com; Secure; Path=/", "example.com", "/", true).is_none(),
            "lowercase __host- prefix must still be enforced (Domain rejected)"
        );
        assert!(
            parse_set_cookie("__SECURE-id=1", "example.com", "/", true).is_none(),
            "uppercase __SECURE- prefix must still be enforced (no Secure rejected)"
        );
        // And a valid lowercase variant is accepted.
        assert!(
            parse_set_cookie("__host-id=1; Secure; Path=/", "example.com", "/", true).is_some(),
            "valid lowercase __host- cookie must be accepted"
        );
    }

    #[test]
    fn ordinary_cookie_unaffected_by_prefix_rules() {
        // A cookie whose name merely contains, but does not start with, a
        // prefix string is untouched.
        let c = parse_set_cookie("x__Host-=1", "example.com", "/", true)
            .expect("name not starting with __Host- is unaffected");
        assert_eq!(c.name, "x__Host-");
    }
}
