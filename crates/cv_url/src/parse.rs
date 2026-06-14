//! URL parser. WHATWG state-machine layout, scoped today to absolute URLs
//! with `http`/`https` (everything M0 needs). The state names below match
//! the spec so filling in the remaining states is a matter of dropping in
//! more match arms — no restructuring.

use crate::origin::Origin;
use crate::scheme::Scheme;
use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UrlError {
    Empty,
    InvalidScheme,
    MissingHost,
    InvalidPort,
    InvalidHost,
    Unsupported(String),
}

impl fmt::Display for UrlError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty => f.write_str("empty URL"),
            Self::InvalidScheme => f.write_str("invalid scheme"),
            Self::MissingHost => f.write_str("missing host"),
            Self::InvalidPort => f.write_str("invalid port"),
            Self::InvalidHost => f.write_str("invalid host"),
            Self::Unsupported(s) => write!(f, "unsupported URL form: {s}"),
        }
    }
}

impl std::error::Error for UrlError {}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Url {
    pub scheme: Scheme,
    pub username: String,
    pub password: String,
    pub host: String,
    pub port: Option<u16>,
    pub path: String,
    pub query: Option<String>,
    pub fragment: Option<String>,
}

impl Url {
    pub fn parse(input: &str) -> Result<Self, UrlError> {
        // 1. Trim leading/trailing C0 + space per spec (callers normally
        //    have already done this, but we double-check).
        let s = input.trim_matches(|c: char| c <= ' ');
        if s.is_empty() {
            return Err(UrlError::Empty);
        }

        // 2. Strip ASCII tab/newline anywhere (WHATWG: "remove all ASCII
        //    tab or newline").
        let cleaned: String = s
            .chars()
            .filter(|c| !matches!(*c, '\t' | '\n' | '\r'))
            .collect();

        // 3. Scheme.
        let (scheme_str, rest) = split_scheme(&cleaned).ok_or(UrlError::InvalidScheme)?;
        let scheme = Scheme::from_lowercase(&scheme_str);

        // For now we only fully parse special-scheme URLs (http/https/ws/wss/file).
        // Opaque-path schemes (data:, about:, blob:, custom) get a single bucket.
        if !scheme.is_special() {
            return Ok(Self {
                scheme,
                username: String::new(),
                password: String::new(),
                host: String::new(),
                port: None,
                path: rest.to_string(),
                query: None,
                fragment: None,
            });
        }

        // 4. WHATWG "special-authority-ignore-slashes" state: for special
        //    schemes accept `//`, `/`, `\\`, `\/`, `\\`, or no slashes at
        //    all. Consume up to two leading slash/backslash characters so
        //    `http:example.com` and `http:\\x` both work like Chrome.
        let after_slashes = {
            let mut s = rest;
            // Consume up to 2 leading slashes or backslashes.
            let mut consumed = 0usize;
            for ch in s.chars() {
                if consumed >= 2 {
                    break;
                }
                if ch == '/' || ch == '\\' {
                    consumed += ch.len_utf8();
                } else {
                    break;
                }
            }
            &s[consumed..]
        };

        // 5. Split off fragment, then query, then path-vs-authority.
        let (no_frag, fragment) = split_off(after_slashes, '#');
        let fragment = fragment.map(|f| percent_encode_fragment(&f));
        let (no_query, query) = split_off(no_frag.as_str(), '?');
        let query = query.map(|q| percent_encode_query(&q));

        // 6. Authority is everything up to first '/', '\\' (for special), or end.
        let (authority, path_rest) = match no_query.find(|c: char| c == '/' || c == '\\') {
            Some(i) => (&no_query[..i], &no_query[i..]),
            None => (no_query.as_str(), ""),
        };

        // 7. Userinfo / host / port.
        let (userinfo, host_port) = match authority.rfind('@') {
            Some(i) => (Some(&authority[..i]), &authority[i + 1..]),
            None => (None, authority),
        };
        let (username, password) = match userinfo {
            Some(ui) => match ui.find(':') {
                Some(i) => (decode_userinfo(&ui[..i]), decode_userinfo(&ui[i + 1..])),
                None => (decode_userinfo(ui), String::new()),
            },
            None => (String::new(), String::new()),
        };

        let (host, port) = parse_host_port(host_port, scheme)?;
        if scheme != Scheme::File && host.is_empty() {
            return Err(UrlError::MissingHost);
        }

        // 8. Path. For special schemes, treat '\\' as '/', then percent-encode.
        let mut path = String::new();
        if path_rest.is_empty() {
            path.push('/');
        } else {
            let normalized: String = path_rest
                .chars()
                .map(|c| if c == '\\' { '/' } else { c })
                .collect();
            path = percent_encode_path(&normalized);
        }

        Ok(Self {
            scheme,
            username,
            password,
            host,
            port,
            path,
            query,
            fragment,
        })
    }

    pub fn origin(&self) -> Origin {
        Origin::new(
            self.scheme,
            self.host.clone(),
            self.port.or_else(|| self.scheme.default_port()),
        )
    }

    pub fn effective_port(&self) -> Option<u16> {
        self.port.or_else(|| self.scheme.default_port())
    }

    /// Resolve `reference` against this URL per RFC 3986 §5. Subset: handles
    /// absolute references, scheme-relative `//host/...`, root-relative
    /// `/path`, fragment-only `#frag`, query-only `?q`, and relative path
    /// references with `.`/`..` normalization. Does not yet handle opaque
    /// schemes other than passing them through.
    pub fn resolve(&self, reference: &str) -> Result<Self, UrlError> {
        let r = reference.trim();
        if r.is_empty() {
            return Ok(self.clone());
        }
        // Try parse as absolute first.
        if Self::parse(r).is_ok() {
            return Self::parse(r);
        }
        // Scheme-relative: //host/path
        if let Some(rest) = r.strip_prefix("//") {
            let synthetic = format!("{}://{}", self.scheme.as_str(), rest);
            return Self::parse(&synthetic);
        }
        // Fragment-only.
        if let Some(frag) = r.strip_prefix('#') {
            let mut out = self.clone();
            out.fragment = Some(percent_encode_fragment(frag));
            return Ok(out);
        }
        // Query-only.
        if let Some(q) = r.strip_prefix('?') {
            let mut out = self.clone();
            let (q_main, frag) = match q.split_once('#') {
                Some((a, b)) => (a.to_string(), Some(b.to_string())),
                None => (q.to_string(), None),
            };
            out.query = Some(percent_encode_query(&q_main));
            out.fragment = frag.map(|f| percent_encode_fragment(&f));
            return Ok(out);
        }
        // Split reference into path / query / fragment.
        let (path_q, frag) = match r.split_once('#') {
            Some((a, b)) => (a, Some(b.to_string())),
            None => (r, None),
        };
        let (rpath, rquery) = match path_q.split_once('?') {
            Some((a, b)) => (a, Some(b.to_string())),
            None => (path_q, None),
        };
        let merged_path = if rpath.starts_with('/') {
            rpath.to_string()
        } else {
            // Strip the last segment of base path, then append.
            let base = &self.path;
            let last_slash = base.rfind('/').unwrap_or(0);
            let mut merged = base[..=last_slash].to_string();
            merged.push_str(rpath);
            merged
        };
        let normalized = remove_dot_segments(&merged_path);
        let mut out = self.clone();
        out.path = percent_encode_path(&normalized);
        out.query = rquery.map(|q| percent_encode_query(&q));
        out.fragment = frag.map(|f| percent_encode_fragment(&f));
        Ok(out)
    }

    /// "/path?query" — the request-target form used by HTTP/1.1.
    pub fn request_target(&self) -> String {
        let mut s = self.path.clone();
        if let Some(q) = &self.query {
            s.push('?');
            s.push_str(q);
        }
        s
    }
}

impl fmt::Display for Url {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}://", self.scheme.as_str())?;
        if !self.username.is_empty() || !self.password.is_empty() {
            f.write_str(&self.username)?;
            if !self.password.is_empty() {
                write!(f, ":{}", self.password)?;
            }
            f.write_str("@")?;
        }
        f.write_str(&self.host)?;
        if let Some(p) = self.port {
            if Some(p) != self.scheme.default_port() {
                write!(f, ":{p}")?;
            }
        }
        f.write_str(&self.path)?;
        if let Some(q) = &self.query {
            write!(f, "?{q}")?;
        }
        if let Some(fr) = &self.fragment {
            write!(f, "#{fr}")?;
        }
        Ok(())
    }
}

/// Normalize a path segment for dot-segment comparison.
/// Returns `true` if the segment is a single dot (`.` or `%2e` or `%2E`).
fn is_dot_segment(seg: &str) -> bool {
    seg == "." || seg.eq_ignore_ascii_case("%2e")
}

/// Returns `true` if the segment is a double dot (`..` or any case-insensitive
/// combination of `.` and `%2e`).
fn is_dotdot_segment(seg: &str) -> bool {
    seg == ".."
        || seg.eq_ignore_ascii_case("%2e%2e")
        || seg.eq_ignore_ascii_case(".%2e")
        || seg.eq_ignore_ascii_case("%2e.")
}

/// RFC 3986 §5.2.4 — remove `.` and `..` segments from a path.
/// Also handles percent-encoded dot segments per WHATWG URL §4.4:
///   `%2e` == `.`  and  `%2e%2e` / `.%2e` / `%2e.` == `..`
fn remove_dot_segments(input: &str) -> String {
    let segments: Vec<&str> = input.split('/').collect();
    let mut out: Vec<&str> = Vec::new();
    let n = segments.len();
    // Track whether the last action was consuming a `..` at the final position
    // so we can append a trailing empty segment (trailing slash) per WHATWG.
    let mut trailing_slash = false;
    for (idx, seg) in segments.iter().enumerate() {
        if is_dot_segment(seg) {
            // Single dot: skip this segment; if it is the last segment we need
            // a trailing slash (same semantics as ending with `/`).
            if idx == n - 1 {
                trailing_slash = true;
            }
        } else if is_dotdot_segment(seg) {
            // Double dot: go up one level.
            out.pop();
            // If this is the last segment, the result must end with `/`.
            if idx == n - 1 {
                trailing_slash = true;
            }
        } else {
            out.push(seg);
            trailing_slash = false;
        }
    }
    let mut s = out.join("/");
    // Preserve leading slash.
    if input.starts_with('/') && !s.starts_with('/') {
        s.insert(0, '/');
    }
    // Apply trailing slash from dot-segment consumption or from original input.
    let needs_trailing = trailing_slash || input.ends_with('/');
    if needs_trailing && !s.ends_with('/') {
        s.push('/');
    }
    if s.is_empty() {
        s.push('/');
    }
    s
}

fn split_scheme(s: &str) -> Option<(String, &str)> {
    let bytes = s.as_bytes();
    if bytes.is_empty() || !bytes[0].is_ascii_alphabetic() {
        return None;
    }
    for (i, b) in bytes.iter().enumerate().skip(1) {
        match *b {
            b':' => {
                let scheme = s[..i].to_ascii_lowercase();
                return Some((scheme, &s[i + 1..]));
            }
            c if c.is_ascii_alphanumeric() || matches!(c, b'+' | b'-' | b'.') => {}
            _ => return None,
        }
    }
    None
}

fn split_off(s: &str, ch: char) -> (String, Option<String>) {
    match s.find(ch) {
        Some(i) => (s[..i].to_string(), Some(s[i + 1..].to_string())),
        None => (s.to_string(), None),
    }
}

// ---------------------------------------------------------------------------
// Percent-encoding (WHATWG URL §2.1).
// ---------------------------------------------------------------------------

/// Apply percent-encoding byte-by-byte, preserving existing valid `%XX`
/// sequences so that already-encoded URLs are not double-encoded.
/// A lone `%` that is NOT followed by two hex digits is encoded as `%25`
/// (per WHATWG URL §2.1).
fn percent_encode_with(s: &str, needs_encode: impl Fn(u8) -> bool) -> String {
    use std::fmt::Write as _;
    let bytes = s.as_bytes();
    let mut out = String::with_capacity(s.len() + 16);
    let mut i = 0;
    while i < bytes.len() {
        let b = bytes[i];
        // Preserve an existing valid percent-encoded sequence.
        if b == b'%'
            && i + 2 < bytes.len()
            && bytes[i + 1].is_ascii_hexdigit()
            && bytes[i + 2].is_ascii_hexdigit()
        {
            out.push('%');
            out.push(bytes[i + 1].to_ascii_uppercase() as char);
            out.push(bytes[i + 2].to_ascii_uppercase() as char);
            i += 3;
            continue;
        }
        // A lone `%` (not a valid escape) must itself be encoded as `%25`.
        let must_encode = needs_encode(b) || b == b'%';
        if must_encode {
            // Write `%XX` for this byte.  For non-ASCII code points all
            // constituent UTF-8 bytes are encoded individually in the loop.
            let _ = write!(out, "%{b:02X}");
        } else {
            out.push(b as char);
        }
        i += 1;
    }
    out
}

/// WHATWG path percent-encode set.
#[inline]
fn path_needs_encode(b: u8) -> bool {
    // Encode: C0 controls, DEL, > 0x7E (non-ASCII), plus space " # < > ? ` { }
    b > 0x7E
        || b < 0x21
        || matches!(b, b'"' | b'#' | b'<' | b'>' | b'?' | b'`' | b'{' | b'}')
}

/// WHATWG special-query percent-encode set (used for http/https query strings).
#[inline]
fn query_needs_encode(b: u8) -> bool {
    // C0 + DEL + non-ASCII + space " # < > '
    b > 0x7E || b < 0x21 || matches!(b, b'"' | b'#' | b'<' | b'>' | b'\'')
}

/// WHATWG fragment percent-encode set.
#[inline]
fn fragment_needs_encode(b: u8) -> bool {
    // C0 + DEL + non-ASCII + space " < > `
    b > 0x7E || b < 0x21 || matches!(b, b'"' | b'<' | b'>' | b'`')
}

pub fn percent_encode_path(s: &str) -> String {
    percent_encode_with(s, path_needs_encode)
}

pub fn percent_encode_query(s: &str) -> String {
    percent_encode_with(s, query_needs_encode)
}

pub fn percent_encode_fragment(s: &str) -> String {
    percent_encode_with(s, fragment_needs_encode)
}

/// WHATWG IPv6 parser (§4.2).  `inner` is the address string *without* brackets.
/// Returns the normalized lowercase address (e.g. `"::1"`, `"2001:db8::1"`) or
/// `UrlError::InvalidHost` if malformed.
fn parse_ipv6(inner: &str) -> Result<String, UrlError> {
    // Each piece is a u16.  We work with Option-slots and fill them.
    let mut pieces: [Option<u16>; 8] = [None; 8];

    if inner.is_empty() {
        return Err(UrlError::InvalidHost);
    }

    // Detect `::` (compressed zeros).
    let compress_pos = inner.find("::");
    if let Some(pos) = compress_pos {
        // Ensure there is at most one `::`.
        if inner[pos + 2..].contains("::") {
            return Err(UrlError::InvalidHost);
        }
        let before = &inner[..pos];
        let after = &inner[pos + 2..];
        let mut idx = 0usize;
        if !before.is_empty() {
            for part in before.split(':') {
                if idx >= 8 {
                    return Err(UrlError::InvalidHost);
                }
                pieces[idx] = Some(parse_ipv6_piece(part)?);
                idx += 1;
            }
        }
        // Fill the compressed zeros from the right.
        let mut ridx = 7usize;
        if !after.is_empty() {
            let rparts: Vec<&str> = after.split(':').collect();
            for part in rparts.iter().rev() {
                if ridx < idx {
                    return Err(UrlError::InvalidHost);
                }
                pieces[ridx] = Some(parse_ipv6_piece(part)?);
                if ridx == 0 {
                    break;
                }
                ridx -= 1;
            }
        }
        // Any None slots stay as 0.
        for p in pieces.iter_mut() {
            if p.is_none() {
                *p = Some(0);
            }
        }
    } else {
        // No `::` — must be exactly 8 colon-separated pieces.
        let parts: Vec<&str> = inner.split(':').collect();
        if parts.len() != 8 {
            return Err(UrlError::InvalidHost);
        }
        for (i, part) in parts.iter().enumerate() {
            pieces[i] = Some(parse_ipv6_piece(part)?);
        }
    }

    // Serialize to lowercase hex with minimal `::` compression (WHATWG normalization).
    let groups: Vec<u16> = pieces.iter().map(|p| p.unwrap_or(0)).collect();
    Ok(serialize_ipv6(&groups))
}

fn parse_ipv6_piece(s: &str) -> Result<u16, UrlError> {
    if s.is_empty() || s.len() > 4 || !s.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err(UrlError::InvalidHost);
    }
    u16::from_str_radix(s, 16).map_err(|_| UrlError::InvalidHost)
}

/// Serialize 8 u16 groups to lowercase hex with WHATWG `::` compression
/// (find the longest run of zero groups; ties broken by leftmost run).
fn serialize_ipv6(groups: &[u16]) -> String {
    // Find longest run of zeros.
    let mut best_start = usize::MAX;
    let mut best_len = 0usize;
    let mut cur_start = 0;
    let mut cur_len = 0;
    for (i, &g) in groups.iter().enumerate() {
        if g == 0 {
            if cur_len == 0 {
                cur_start = i;
            }
            cur_len += 1;
            if cur_len > best_len {
                best_len = cur_len;
                best_start = cur_start;
            }
        } else {
            cur_len = 0;
        }
    }
    // Only compress if the run is at least 2 groups.
    let compress = if best_len >= 2 { Some((best_start, best_start + best_len)) } else { None };

    let mut out = String::new();
    let mut i = 0;
    while i < 8 {
        if let Some((start, end)) = compress {
            if i == start {
                out.push_str("::");
                i = end;
                continue;
            }
        }
        if !out.is_empty() && !out.ends_with("::") {
            out.push(':');
        }
        out.push_str(&format!("{:x}", groups[i]));
        i += 1;
    }
    out
}

fn parse_host_port(s: &str, scheme: Scheme) -> Result<(String, Option<u16>), UrlError> {
    if s.is_empty() {
        return Ok((String::new(), None));
    }
    // IPv6 literal: starts with '['.
    if let Some(stripped) = s.strip_prefix('[') {
        let close = stripped.find(']').ok_or(UrlError::InvalidHost)?;
        let inner = &stripped[..close];
        let validated = parse_ipv6(inner)?;
        // Store as "[<normalized>]" so Display can output it correctly.
        let host = format!("[{validated}]");
        let after = &stripped[close + 1..];
        let port = parse_port_suffix(after)?;
        return Ok((host, port));
    }
    let (host, port_str) = match s.rfind(':') {
        Some(i) => (&s[..i], Some(&s[i + 1..])),
        None => (s, None),
    };
    if host.is_empty() {
        return Err(UrlError::InvalidHost);
    }
    let host = normalize_host(host, scheme)?;
    let port = match port_str {
        Some(p) if p.is_empty() => None,
        Some(p) => {
            let n = parse_port_str(p)?;
            // WHATWG URL §4.1: if the port equals the scheme's default port,
            // it is normalized to None (omitted). This keeps origin equality
            // correct: new URL("http://x.com").origin === new URL("http://x.com:80").origin
            if scheme.default_port() == Some(n) {
                None
            } else {
                Some(n)
            }
        }
        None => None,
    };
    Ok((host, port))
}

fn parse_port_suffix(after: &str) -> Result<Option<u16>, UrlError> {
    if after.is_empty() {
        Ok(None)
    } else if let Some(rest) = after.strip_prefix(':') {
        Ok(Some(parse_port_str(rest)?))
    } else {
        Err(UrlError::InvalidPort)
    }
}

fn parse_port_str(s: &str) -> Result<u16, UrlError> {
    if s.is_empty() || !s.bytes().all(|b| b.is_ascii_digit()) {
        return Err(UrlError::InvalidPort);
    }
    let v: u32 = s.parse().map_err(|_| UrlError::InvalidPort)?;
    if v > 0xFFFF {
        return Err(UrlError::InvalidPort);
    }
    Ok(v as u16)
}

/// WHATWG IPv4 parser (§4.1).  Returns `Some("a.b.c.d")` if `s` looks like
/// a numeric IPv4 address (all dot-separated parts are decimal/hex/octal
/// numbers); returns `None` if it doesn't look like an IP at all; returns
/// `Err(UrlError::InvalidHost)` if it looks like an IP but is malformed.
fn try_parse_ipv4(s: &str) -> Result<Option<String>, UrlError> {
    // Quick check: every character must be ASCII alphanumeric or '.'
    // (hex digits, 0x prefix, octal digits, decimal).
    if !s.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'.') {
        return Ok(None);
    }
    let parts: Vec<&str> = s.split('.').collect();
    // Must have 1-4 parts and every part must be a non-empty string of
    // only digits / hex-with-0x-prefix.
    if parts.is_empty() || parts.len() > 4 {
        return Ok(None);
    }
    // Check that every part looks numeric (decimal, 0x…, or 0…).
    fn looks_numeric(p: &str) -> bool {
        if p.is_empty() {
            return false;
        }
        if let Some(hex) = p.strip_prefix("0x").or_else(|| p.strip_prefix("0X")) {
            return !hex.is_empty() && hex.bytes().all(|b| b.is_ascii_hexdigit());
        }
        p.bytes().all(|b| b.is_ascii_digit())
    }
    if !parts.iter().all(|p| looks_numeric(p)) {
        return Ok(None);
    }

    // Parse each part as the right numeric base.
    fn parse_part(p: &str) -> Result<u32, UrlError> {
        if let Some(hex) = p.strip_prefix("0x").or_else(|| p.strip_prefix("0X")) {
            u32::from_str_radix(hex, 16).map_err(|_| UrlError::InvalidHost)
        } else if p.len() > 1 && p.starts_with('0') {
            // Octal.
            u32::from_str_radix(p, 8).map_err(|_| UrlError::InvalidHost)
        } else {
            p.parse::<u32>().map_err(|_| UrlError::InvalidHost)
        }
    }

    let mut nums: Vec<u32> = Vec::with_capacity(parts.len());
    for p in &parts {
        nums.push(parse_part(p)?);
    }

    // Combine into a 32-bit address (WHATWG §4.1 steps 8-10).
    // The last part can represent multiple octets if there are fewer than 4 parts.
    let addr: u32 = match nums.as_slice() {
        [a] => {
            if *a > 0xFFFF_FFFF {
                return Err(UrlError::InvalidHost);
            }
            *a
        }
        [a, b] => {
            if *a > 0xFF || *b > 0x00FF_FFFF {
                return Err(UrlError::InvalidHost);
            }
            (*a << 24) | *b
        }
        [a, b, c] => {
            if *a > 0xFF || *b > 0xFF || *c > 0x0000_FFFF {
                return Err(UrlError::InvalidHost);
            }
            (*a << 24) | (*b << 16) | *c
        }
        [a, b, c, d] => {
            if *a > 0xFF || *b > 0xFF || *c > 0xFF || *d > 0xFF {
                return Err(UrlError::InvalidHost);
            }
            (*a << 24) | (*b << 16) | (*c << 8) | *d
        }
        _ => return Ok(None),
    };

    Ok(Some(format!(
        "{}.{}.{}.{}",
        (addr >> 24) & 0xFF,
        (addr >> 16) & 0xFF,
        (addr >> 8) & 0xFF,
        addr & 0xFF
    )))
}

/// Host normalization implementing WHATWG "domain-to-ASCII":
/// - ASCII host letters are lowercased.
/// - Numeric IPv4 addresses (decimal/hex/octal) are canonicalized to dotted-decimal.
/// - Labels that contain non-ASCII code points are Punycode-encoded
///   per RFC 3492 and prefixed with `xn--`.
/// - Invalid host bytes are rejected.
fn normalize_host(s: &str, _scheme: Scheme) -> Result<String, UrlError> {
    if s.is_empty() {
        return Err(UrlError::InvalidHost);
    }
    // Reject forbidden host code points.
    for c in s.chars() {
        if matches!(c, '\0' | '\t' | '\n' | '\r' | ' ' | '#' | '/' | ':' | '?' | '@' | '[' | '\\' | ']') {
            return Err(UrlError::InvalidHost);
        }
    }
    // Check if any character is non-ASCII — if not, fast path.
    if s.bytes().all(|b| b.is_ascii()) {
        let lower = s.to_ascii_lowercase();
        // Try IPv4 canonicalization on the lowercased ASCII form.
        if let Some(ipv4) = try_parse_ipv4(&lower)? {
            return Ok(ipv4);
        }
        return Ok(lower);
    }
    // IDNA: split by '.', encode each label that contains non-ASCII, rejoin.
    let labels: Vec<String> = s
        .split('.')
        .map(|label| {
            if label.bytes().all(|b| b.is_ascii()) {
                label.to_ascii_lowercase()
            } else {
                // ToASCII: lowercase + Punycode encode → "xn--<encoded>"
                let lowered: String = label.to_lowercase();
                let encoded = punycode_encode(&lowered);
                format!("xn--{encoded}")
            }
        })
        .collect();
    Ok(labels.join("."))
}

/// Punycode encoding per RFC 3492.
/// Encodes the code points above U+007E (non-ASCII) in `input`, assuming
/// `input` is already NFC-lowercased. Returns the encoded suffix
/// (without the `xn--` prefix — the caller adds it).
fn punycode_encode(input: &str) -> String {
    // RFC 3492 constants.
    const BASE: u32 = 36;
    const TMIN: u32 = 1;
    const TMAX: u32 = 26;
    const SKEW: u32 = 38;
    const DAMP: u32 = 700;
    const INITIAL_BIAS: u32 = 72;
    const INITIAL_N: u32 = 128;

    fn adapt(mut delta: u32, num_points: u32, first_time: bool) -> u32 {
        delta = if first_time { delta / DAMP } else { delta / 2 };
        delta += delta / num_points;
        let mut k = 0u32;
        while delta > ((BASE - TMIN) * TMAX) / 2 {
            delta /= BASE - TMIN;
            k += BASE;
        }
        k + (BASE - TMIN + 1) * delta / (delta + SKEW)
    }

    fn digit_to_char(d: u32) -> char {
        if d < 26 { (b'a' + d as u8) as char } else { (b'0' + (d - 26) as u8) as char }
    }

    let code_points: Vec<u32> = input.chars().map(|c| c as u32).collect();

    // Basic code points go first (as-is).
    let mut output = String::new();
    for &cp in &code_points {
        if cp < 128 {
            output.push(char::from_u32(cp).unwrap_or('?'));
        }
    }
    let b = output.len() as u32;
    if b > 0 {
        output.push('-');
    }

    let mut n = INITIAL_N;
    let mut delta: u32 = 0;
    let mut bias = INITIAL_BIAS;
    let h = b; // number of handled code points so far
    let mut h = h;

    while h < code_points.len() as u32 {
        // Find the smallest non-basic code point >= n.
        let m = code_points
            .iter()
            .filter(|&&cp| cp >= n)
            .copied()
            .min()
            .unwrap_or(n);

        // Increase delta by (m - n) * (h + 1), checking for overflow.
        delta = delta.saturating_add((m - n).saturating_mul(h + 1));
        n = m;

        for &cp in &code_points {
            if cp < n {
                delta = delta.saturating_add(1);
            }
            if cp == n {
                // Emit a generalized variable-length integer for delta.
                let mut q = delta;
                let mut k = BASE;
                loop {
                    let t = if k <= bias + TMIN {
                        TMIN
                    } else if k >= bias + TMAX {
                        TMAX
                    } else {
                        k - bias
                    };
                    if q < t { break; }
                    output.push(digit_to_char(t + (q - t) % (BASE - t)));
                    q = (q - t) / (BASE - t);
                    k += BASE;
                }
                output.push(digit_to_char(q));
                bias = adapt(delta, h + 1, h == b);
                delta = 0;
                h += 1;
            }
        }
        delta += 1;
        n += 1;
    }
    output
}

fn decode_userinfo(s: &str) -> String {
    // Userinfo is percent-decoded for display but we keep the encoded form
    // for the wire. For now keep both equal — full credential handling is
    // a network-stack concern (auth modules, M5+).
    s.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_https_example() {
        let u = Url::parse("https://example.com").unwrap();
        assert_eq!(u.scheme, Scheme::Https);
        assert_eq!(u.host, "example.com");
        assert_eq!(u.port, None);
        assert_eq!(u.effective_port(), Some(443));
        assert_eq!(u.path, "/");
        assert_eq!(u.request_target(), "/");
    }

    #[test]
    fn parses_path_and_query() {
        let u = Url::parse("https://example.com/foo/bar?x=1&y=2").unwrap();
        assert_eq!(u.path, "/foo/bar");
        assert_eq!(u.query.as_deref(), Some("x=1&y=2"));
        assert_eq!(u.request_target(), "/foo/bar?x=1&y=2");
    }

    #[test]
    fn parses_explicit_port() {
        let u = Url::parse("http://example.com:8080/").unwrap();
        assert_eq!(u.port, Some(8080));
        assert_eq!(u.effective_port(), Some(8080));
    }

    #[test]
    fn lowercases_scheme_and_host() {
        let u = Url::parse("HTTPS://Example.COM/").unwrap();
        assert_eq!(u.scheme, Scheme::Https);
        assert_eq!(u.host, "example.com");
    }

    #[test]
    fn parses_userinfo() {
        let u = Url::parse("https://alice:secret@example.com/").unwrap();
        assert_eq!(u.username, "alice");
        assert_eq!(u.password, "secret");
    }

    #[test]
    fn parses_ipv6_literal() {
        let u = Url::parse("https://[::1]:8443/").unwrap();
        assert_eq!(u.host, "[::1]");
        assert_eq!(u.port, Some(8443));
    }

    #[test]
    fn rejects_empty() {
        assert!(matches!(Url::parse(""), Err(UrlError::Empty)));
        assert!(matches!(Url::parse("   "), Err(UrlError::Empty)));
    }

    #[test]
    fn rejects_bad_scheme() {
        assert!(matches!(
            Url::parse("1http://x"),
            Err(UrlError::InvalidScheme)
        ));
    }

    #[test]
    fn fragment_split() {
        let u = Url::parse("https://example.com/a#section").unwrap();
        assert_eq!(u.fragment.as_deref(), Some("section"));
        assert_eq!(u.path, "/a");
    }

    #[test]
    fn opaque_scheme_keeps_rest() {
        let u = Url::parse("data:text/plain,hello").unwrap();
        assert_eq!(u.scheme, Scheme::Data);
        assert_eq!(u.path, "text/plain,hello");
    }

    #[test]
    fn origin_strips_default_port() {
        let u = Url::parse("https://example.com/").unwrap();
        assert_eq!(u.origin().to_string(), "https://example.com");
        let u2 = Url::parse("https://example.com:9443/").unwrap();
        assert_eq!(u2.origin().to_string(), "https://example.com:9443");
    }

    #[test]
    fn resolve_absolute() {
        let base = Url::parse("https://example.com/a/b").unwrap();
        let r = base.resolve("https://other.test/x").unwrap();
        assert_eq!(r.to_string(), "https://other.test/x");
    }

    #[test]
    fn resolve_root() {
        let base = Url::parse("https://example.com/a/b").unwrap();
        let r = base.resolve("/c").unwrap();
        assert_eq!(r.to_string(), "https://example.com/c");
    }

    #[test]
    fn resolve_relative_with_dotdot() {
        let base = Url::parse("https://example.com/a/b/c").unwrap();
        let r = base.resolve("../d").unwrap();
        assert_eq!(r.to_string(), "https://example.com/a/d");
    }

    #[test]
    fn resolve_fragment_only() {
        let base = Url::parse("https://example.com/a?x=1").unwrap();
        let r = base.resolve("#section").unwrap();
        assert_eq!(r.to_string(), "https://example.com/a?x=1#section");
    }

    #[test]
    fn resolve_scheme_relative() {
        let base = Url::parse("https://example.com/").unwrap();
        let r = base.resolve("//other.test/x").unwrap();
        assert_eq!(r.to_string(), "https://other.test/x");
    }

    #[test]
    fn roundtrip_display() {
        let s = "https://example.com/path?q=1#frag";
        let u = Url::parse(s).unwrap();
        assert_eq!(u.to_string(), s);
    }

    // Percent-encoding tests (WHATWG URL §2.1)
    #[test]
    fn path_space_encoded() {
        let u = Url::parse("https://example.com/hello world").unwrap();
        assert_eq!(u.path, "/hello%20world");
        assert_eq!(u.request_target(), "/hello%20world");
    }

    #[test]
    fn query_space_encoded() {
        let u = Url::parse("https://example.com/search?q=hello world").unwrap();
        assert_eq!(u.query.as_deref(), Some("q=hello%20world"));
    }

    #[test]
    fn fragment_space_encoded() {
        let u = Url::parse("https://example.com/page#hello world").unwrap();
        assert_eq!(u.fragment.as_deref(), Some("hello%20world"));
    }

    #[test]
    fn already_encoded_not_doubled() {
        // Existing `%20` must not become `%2520`.
        let u = Url::parse("https://example.com/hello%20world").unwrap();
        assert_eq!(u.path, "/hello%20world");
    }

    #[test]
    fn non_ascii_path_encoded() {
        // UTF-8: São (S + U+00E3 o) encodes to S%C3%A3o
        let u = Url::parse("https://example.com/São_Paulo").unwrap();
        assert!(u.path.contains("%C3%A3"), "non-ASCII should be percent-encoded: {}", u.path);
    }

    #[test]
    fn path_encode_ascii_clean() {
        // Common ASCII URL chars that should NOT be encoded.
        let u = Url::parse("https://example.com/a-b_c.d/e:f@g").unwrap();
        assert_eq!(u.path, "/a-b_c.d/e:f@g");
    }

    #[test]
    fn lone_percent_encoded_as_percent25() {
        // An invalid `%GG` sequence: `%` should become `%25`.
        let u = Url::parse("https://example.com/100%off").unwrap();
        assert!(u.path.contains("%25"), "lone % should be %25: {}", u.path);
    }

    // WHATWG "special-authority-ignore-slashes": special schemes accept
    // no slashes, one slash, or backslashes after the colon (Bug 5).
    #[test]
    fn no_slashes_special_scheme() {
        let u = Url::parse("http:example.com/path").unwrap();
        assert_eq!(u.host, "example.com");
        assert_eq!(u.path, "/path");
    }

    #[test]
    fn single_slash_special_scheme() {
        let u = Url::parse("http:/example.com/path").unwrap();
        assert_eq!(u.host, "example.com");
        assert_eq!(u.path, "/path");
    }

    #[test]
    fn backslash_special_scheme() {
        let u = Url::parse("http:\\\\example.com\\path").unwrap();
        assert_eq!(u.host, "example.com");
        assert!(u.path.starts_with('/'), "path should start with /: {}", u.path);
    }

    // Bug 3 (port normalization): explicit default port → stored as None.
    #[test]
    fn default_port_normalized_to_none() {
        let u = Url::parse("http://example.com:80/").unwrap();
        assert_eq!(u.port, None, "http:80 must normalize to None");
        let u2 = Url::parse("https://example.com:443/").unwrap();
        assert_eq!(u2.port, None, "https:443 must normalize to None");
        let u3 = Url::parse("ws://example.com:80/").unwrap();
        assert_eq!(u3.port, None, "ws:80 must normalize to None");
        let u4 = Url::parse("wss://example.com:443/").unwrap();
        assert_eq!(u4.port, None, "wss:443 must normalize to None");
        // Non-default port must still be kept.
        let u5 = Url::parse("http://example.com:8080/").unwrap();
        assert_eq!(u5.port, Some(8080));
    }

    // Origin equality must hold between explicit-default-port and no-port URLs.
    #[test]
    fn origin_equality_with_default_port() {
        let a = Url::parse("http://example.com/").unwrap();
        let b = Url::parse("http://example.com:80/").unwrap();
        assert_eq!(a.origin(), b.origin(), "http://x vs http://x:80 must have equal origins");
        let c = Url::parse("https://example.com/").unwrap();
        let d = Url::parse("https://example.com:443/").unwrap();
        assert_eq!(c.origin(), d.origin(), "https://x vs https://x:443 must have equal origins");
    }

    // Bug 1: host serialization must include non-default port.
    #[test]
    fn host_includes_nondefault_port() {
        let u = Url::parse("https://example.com:8443/path").unwrap();
        // Non-default port 8443 must appear in the Display output.
        assert_eq!(u.to_string(), "https://example.com:8443/path");
        // Default port 443 must be omitted in Display output (Chrome compat).
        let u2 = Url::parse("https://example.com:443/").unwrap();
        assert_eq!(u2.to_string(), "https://example.com/");
    }

    // Bug 5 (IDNA/Punycode): non-ASCII hostnames must be Punycode-encoded.
    #[test]
    fn idna_non_ascii_host_punycode() {
        // "münchen.de" — ü is U+00FC, label "münchen" → punycode "mnchen-3ya"
        // so full xn-- prefix → "xn--mnchen-3ya.de"
        let u = Url::parse("https://münchen.de/").unwrap();
        assert!(
            u.host.starts_with("xn--"),
            "non-ASCII host should be Punycode-encoded: {}",
            u.host
        );
        assert!(u.host.contains(".de"), "TLD must be preserved: {}", u.host);
    }

    // punycode_encode unit tests.
    #[test]
    fn punycode_ascii_label_unchanged() {
        // All-ASCII label: punycode_encode returns the label + "-" (basic suffix).
        let enc = super::punycode_encode("example");
        // Basic-only output ends with a lone hyphen before the zero extended codes.
        assert!(enc.starts_with("example-"), "ASCII labels must start with original chars + '-': {enc}");
    }

    #[test]
    fn punycode_encode_u_umlaut() {
        // "ü" is U+00FC.  RFC 3492 test vector: "ü" → "tda" (basic part empty,
        // so xn--tda).  Our encoder must produce "tda" for the lone char.
        let enc = super::punycode_encode("ü");
        assert_eq!(enc, "tda", "punycode(ü) must be 'tda', got: {enc}");
    }

    // -------------------------------------------------------------------------
    // Bug 1: %2e/%2E dot segments must be treated as `.` in path normalization.
    // -------------------------------------------------------------------------
    #[test]
    fn percent_encoded_dotdot_resolves() {
        // %2e%2e == .. → /a/b/%2e%2e/c should resolve to /a/c (via resolve).
        let base = Url::parse("https://example.com/a/b/%2e%2e/c").unwrap();
        // After parse, the path should be normalized already through resolve();
        // test via resolve("") which returns self.
        // The actual normalization happens in remove_dot_segments used by resolve().
        let normalized = super::remove_dot_segments("/a/b/%2e%2e/c");
        assert_eq!(normalized, "/a/c", "percent-encoded .. must be resolved: {normalized}");
    }

    #[test]
    fn percent_encoded_single_dot_resolves() {
        let normalized = super::remove_dot_segments("/a/b/%2E/c");
        assert_eq!(normalized, "/a/b/c", "percent-encoded . must be skipped: {normalized}");
    }

    #[test]
    fn mixed_percent_dot_resolves() {
        // .%2e == ..
        let normalized = super::remove_dot_segments("/a/b/.%2e/c");
        assert_eq!(normalized, "/a/c", ".%2e must be treated as ..: {normalized}");
        // %2e. == ..
        let normalized2 = super::remove_dot_segments("/a/b/%2E./c");
        assert_eq!(normalized2, "/a/c", "%2E. must be treated as ..: {normalized2}");
    }

    #[test]
    fn resolve_with_percent_encoded_dotdot() {
        let base = Url::parse("https://example.com/a/b/c").unwrap();
        let r = base.resolve("..%2f").unwrap_or_else(|_| base.resolve("../").unwrap());
        // %2e%2e via resolve
        let base2 = Url::parse("https://example.com/a/b/c").unwrap();
        let r2 = base2.resolve("%2e%2e/d").unwrap();
        assert_eq!(r2.path, "/a/d", "resolve with %2e%2e must go up: {}", r2.path);
        let _ = r;
    }

    // -------------------------------------------------------------------------
    // Bug 2: `..` as the last segment must leave a trailing slash.
    // -------------------------------------------------------------------------
    #[test]
    fn dotdot_last_segment_trailing_slash() {
        // /a/b/.. → /a/  (trailing slash)
        let normalized = super::remove_dot_segments("/a/b/..");
        assert_eq!(normalized, "/a/", ".. as last segment must leave trailing slash: {normalized}");
    }

    #[test]
    fn resolve_dotdot_last_segment_trailing_slash() {
        let base = Url::parse("https://example.com/a/b/c").unwrap();
        let r = base.resolve("..").unwrap();
        assert_eq!(r.to_string(), "https://example.com/a/", "resolve(..) must end with /: {}", r);
    }

    #[test]
    fn single_dot_last_segment_trailing_slash() {
        // /a/b/. → /a/b/  (trailing slash)
        let normalized = super::remove_dot_segments("/a/b/.");
        assert_eq!(normalized, "/a/b/", ". as last segment must leave trailing slash: {normalized}");
    }

    // -------------------------------------------------------------------------
    // Bug 3: IPv4 address canonicalization (hex, octal, decimal u32).
    // -------------------------------------------------------------------------
    #[test]
    fn ipv4_hex_canonicalized() {
        let u = Url::parse("http://0x7f.0.0.1/").unwrap();
        assert_eq!(u.host, "127.0.0.1", "hex IPv4 must canonicalize: {}", u.host);
    }

    #[test]
    fn ipv4_decimal_u32_canonicalized() {
        // 3232235521 = 192.168.0.1
        let u = Url::parse("http://3232235521/").unwrap();
        assert_eq!(u.host, "192.168.0.1", "decimal u32 IPv4 must canonicalize: {}", u.host);
    }

    #[test]
    fn ipv4_octal_canonicalized() {
        // 0177 = 127 in octal
        let u = Url::parse("http://0177.0.0.1/").unwrap();
        assert_eq!(u.host, "127.0.0.1", "octal IPv4 must canonicalize: {}", u.host);
    }

    #[test]
    fn ipv4_normal_dotted_decimal_unchanged() {
        let u = Url::parse("http://192.168.1.1/").unwrap();
        assert_eq!(u.host, "192.168.1.1");
    }

    // -------------------------------------------------------------------------
    // Bug 4: IPv6 validation and normalization.
    // -------------------------------------------------------------------------
    #[test]
    fn ipv6_loopback_normalized() {
        let u = Url::parse("http://[::1]:8080/").unwrap();
        assert_eq!(u.host, "[::1]", "::1 must normalize: {}", u.host);
        assert_eq!(u.port, Some(8080));
    }

    #[test]
    fn ipv6_full_address_normalized() {
        let u = Url::parse("http://[2001:0DB8:0000:0000:0000:0000:0000:0001]/").unwrap();
        // Should normalize zeros and use :: compression; at minimum lowercase.
        assert!(
            u.host.starts_with('[') && u.host.ends_with(']'),
            "IPv6 must be stored with brackets: {}",
            u.host
        );
        // Must be lowercase.
        assert_eq!(u.host, u.host.to_ascii_lowercase(), "IPv6 must be lowercase: {}", u.host);
    }

    #[test]
    fn ipv6_malformed_rejected() {
        // Three colons in a row is invalid.
        assert!(
            Url::parse("http://[:::1]/").is_err(),
            "malformed IPv6 must be rejected"
        );
    }

    // -------------------------------------------------------------------------
    // Bug 5: query-only and fragment-only relative resolution (already fixed —
    // regression test to keep working).
    // -------------------------------------------------------------------------
    #[test]
    fn resolve_query_only_preserves_path() {
        let base = Url::parse("https://x.com/path/page").unwrap();
        let r = base.resolve("?search").unwrap();
        assert_eq!(r.to_string(), "https://x.com/path/page?search");
    }

    #[test]
    fn resolve_fragment_only_preserves_path_and_query() {
        let base = Url::parse("https://x.com/path?q=1").unwrap();
        let r = base.resolve("#anchor").unwrap();
        assert_eq!(r.to_string(), "https://x.com/path?q=1#anchor");
    }
}
