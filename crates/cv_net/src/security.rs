//! Browser security primitives — CORS, CSP, SameSite enforcement.
//!
//! Three independent layers:
//!   * **Origin** — RFC 6454 site/origin computation used everywhere
//!     security-sensitive (cookies, CORS, CSP, mixed-content).
//!   * **CORS** — Fetch spec §3.2 cross-origin resource sharing.
//!     `cors_decision()` classifies a request as same-origin
//!     (passthrough), no-cors (opaque), simple cross-origin (needs
//!     ACAO check on response), or preflight (needs OPTIONS first).
//!     `validate_cors_response()` runs the ACAO/ACAH/ACAM checks.
//!   * **CSP** — Content-Security-Policy parser + enforcer.
//!     `parse_csp()` returns a Policy; `Policy::allows_source()`
//!     answers whether a candidate URL passes a given directive.

use cv_url::Url;

/// Canonical origin form: `<scheme>://<host>[:port]`. Empty scheme or
/// host yields the special opaque origin (compares unequal to everything).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Origin(pub String);

impl Origin {
    pub fn of(url: &Url) -> Self {
        if url.scheme.as_str().is_empty() || url.host.is_empty() {
            return Self("null".into());
        }
        let host = url.host.to_ascii_lowercase();
        let scheme = url.scheme.as_str();
        match url.port {
            Some(p) => Self(format!("{scheme}://{host}:{p}")),
            None => Self(format!("{scheme}://{host}")),
        }
    }

    pub fn is_opaque(&self) -> bool {
        self.0 == "null"
    }
}

// ----------------------------------------------------------------------
// CORS
// ----------------------------------------------------------------------

/// Request modes per Fetch spec §3.2.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RequestMode {
    /// `same-origin` — only same-origin allowed. Cross-origin = network error.
    SameOrigin,
    /// `cors` — cross-origin allowed if server opts in via ACAO.
    Cors,
    /// `no-cors` — request goes out with credentials but response is
    /// opaque (script can't read body or headers).
    NoCors,
    /// `navigate` — top-level navigation. No CORS check.
    Navigate,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CorsDecision {
    /// Same origin — pass through normally.
    SameOrigin,
    /// Cross-origin "simple" request — no preflight needed, but the
    /// response must carry the right ACAO header (validated post-fetch).
    SimpleCorsCheck,
    /// Cross-origin request that triggers a preflight: send OPTIONS,
    /// validate the preflight response's ACAH/ACAM, then send the real
    /// request. The included `requested_headers` is the value of
    /// Access-Control-Request-Headers on the preflight.
    Preflight {
        method: String,
        requested_headers: Vec<String>,
    },
    /// no-cors request — go out, but flag the response as opaque.
    Opaque,
    /// Forbidden — same-origin mode against a cross-origin URL.
    Forbidden,
}

/// Classify a request.
///
/// `headers` is the full request header list as `(name, value)` pairs
/// (names should be lower-cased).  The method must be one of
/// `GET`/`HEAD`/`POST` and every header must be CORS-safelisted for the
/// request to qualify as a "simple" request (no preflight).
pub fn cors_decision(
    request_origin: &Origin,
    target_origin: &Origin,
    mode: RequestMode,
    method: &str,
    headers: &[(String, String)],
) -> CorsDecision {
    if request_origin == target_origin {
        return CorsDecision::SameOrigin;
    }
    match mode {
        RequestMode::Navigate => CorsDecision::SameOrigin,
        RequestMode::SameOrigin => CorsDecision::Forbidden,
        RequestMode::NoCors => CorsDecision::Opaque,
        RequestMode::Cors => {
            let simple_method = matches!(
                method.to_ascii_uppercase().as_str(),
                "GET" | "HEAD" | "POST"
            );
            let non_simple_header = headers
                .iter()
                .any(|(n, v)| !is_cors_safelisted_header_with_value(n, Some(v)));
            if simple_method && !non_simple_header {
                CorsDecision::SimpleCorsCheck
            } else {
                CorsDecision::Preflight {
                    method: method.to_ascii_uppercase(),
                    requested_headers: headers.iter().map(|(n, _)| n.clone()).collect(),
                }
            }
        }
    }
}

/// Per Fetch spec §3.2.5 "CORS-safelisted request-header".
/// For `content-type`, only the three MIME essences listed in the spec
/// qualify as safelisted; anything else (e.g. `application/json`) requires
/// a preflight.
pub fn is_cors_safelisted_header(name: &str) -> bool {
    is_cors_safelisted_header_with_value(name, None)
}

/// Variant that accepts an optional header value so that `content-type`
/// can be checked against the safelisted MIME types.  Pass `None` when
/// the value is not available (conservatively returns false for
/// content-type).
pub fn is_cors_safelisted_header_with_value(name: &str, value: Option<&str>) -> bool {
    match name.to_ascii_lowercase().as_str() {
        "accept" | "accept-language" | "content-language" => true,
        "content-type" => {
            // Fetch spec §3.2.5 — Content-Type is safelisted ONLY when
            // its MIME type essence (type/subtype without parameters) is
            // one of these three values.
            let val = match value {
                Some(v) => v,
                // No value supplied → can't confirm it's safe → preflight.
                None => return false,
            };
            // Strip parameters (e.g. "; charset=utf-8") to get the essence.
            let essence = val
                .split(';')
                .next()
                .unwrap_or("")
                .trim()
                .to_ascii_lowercase();
            matches!(
                essence.as_str(),
                "application/x-www-form-urlencoded"
                    | "multipart/form-data"
                    | "text/plain"
            )
        }
        _ => false,
    }
}

/// Validate ACAO + (optionally) ACAH/ACAM on a CORS response.
/// Returns Ok(()) if the response is acceptable, Err(reason) if not.
pub fn validate_cors_response(
    response_headers: &[(String, String)],
    // The REQUESTING (document) origin — per Fetch spec §4.10, the response's
    // `Access-Control-Allow-Origin` must equal the origin that MADE the request,
    // NOT the server being fetched. (Was mistakenly passed the target/server
    // origin, which both blocked legitimate cross-origin reads and could accept
    // a server that echoed its own origin.)
    request_origin: &Origin,
    requested_headers: &[String],
    credentials_mode: bool,
) -> Result<(), String> {
    let acao = response_headers
        .iter()
        .find(|(k, _)| k.eq_ignore_ascii_case("access-control-allow-origin"))
        .map(|(_, v)| v.trim());
    match acao {
        None => Err("missing Access-Control-Allow-Origin".into()),
        Some("*") => {
            if credentials_mode {
                Err("ACAO: * disallowed when credentials included".into())
            } else {
                Ok(())
            }
        }
        Some(v) if v == request_origin.0 => Ok(()),
        // Fetch spec §3.2.3: ACAO "null" satisfies the check only when
        // the request origin is itself opaque (an empty / null origin).
        // For regular https:// page origins, "null" is NOT a match and
        // must be treated as a mismatch to prevent sandbox-iframe CSRF.
        Some("null") => {
            if request_origin.is_opaque() {
                Ok(())
            } else {
                Err("ACAO: null is not accepted for non-opaque origins".into())
            }
        }
        Some(other) => Err(format!("ACAO origin mismatch (got {other})")),
    }?;
    if !requested_headers.is_empty() {
        let acah = response_headers
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case("access-control-allow-headers"))
            .map(|(_, v)| v.clone())
            .unwrap_or_default();
        let allowed: Vec<String> = acah
            .split(',')
            .map(|s| s.trim().to_ascii_lowercase())
            .collect();
        for h in requested_headers {
            let hlc = h.to_ascii_lowercase();
            if !allowed.iter().any(|a| a == &hlc || a == "*") {
                return Err(format!("ACAH disallowed header: {h}"));
            }
        }
    }
    Ok(())
}

// ----------------------------------------------------------------------
// Content Security Policy
// ----------------------------------------------------------------------

#[derive(Debug, Clone, Default)]
pub struct Policy {
    /// `default-src` fallback list (lower-cased).
    pub default_src: Vec<String>,
    pub script_src: Option<Vec<String>>,
    pub style_src: Option<Vec<String>>,
    pub img_src: Option<Vec<String>>,
    pub connect_src: Option<Vec<String>>,
    pub font_src: Option<Vec<String>>,
    pub frame_src: Option<Vec<String>>,
    pub media_src: Option<Vec<String>>,
    pub object_src: Option<Vec<String>>,
    pub form_action: Option<Vec<String>>,
    pub upgrade_insecure_requests: bool,
    pub block_all_mixed_content: bool,
    pub raw: String,
}

/// Per-directive resource categories the browser actually fetches.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CspDirective {
    Script,
    Style,
    Img,
    Connect,
    Font,
    Frame,
    Media,
    Object,
    FormAction,
}

pub fn parse_csp(value: &str) -> Policy {
    let mut p = Policy::default();
    p.raw = value.to_string();
    for directive in value.split(';') {
        let directive = directive.trim();
        if directive.is_empty() {
            continue;
        }
        let mut parts = directive.split_whitespace();
        let name = match parts.next() {
            Some(n) => n.to_ascii_lowercase(),
            None => continue,
        };
        let sources: Vec<String> = parts.map(|s| s.to_string()).collect();
        match name.as_str() {
            "default-src" => p.default_src = sources,
            "script-src" => p.script_src = Some(sources),
            "style-src" => p.style_src = Some(sources),
            "img-src" => p.img_src = Some(sources),
            "connect-src" => p.connect_src = Some(sources),
            "font-src" => p.font_src = Some(sources),
            "frame-src" => p.frame_src = Some(sources),
            "media-src" => p.media_src = Some(sources),
            "object-src" => p.object_src = Some(sources),
            "form-action" => p.form_action = Some(sources),
            "upgrade-insecure-requests" => p.upgrade_insecure_requests = true,
            "block-all-mixed-content" => p.block_all_mixed_content = true,
            _ => {}
        }
    }
    p
}

impl Policy {
    /// Look up the source list governing `directive`, falling back to
    /// `default-src` when the specific directive isn't set.
    fn source_list(&self, directive: CspDirective) -> &[String] {
        let specific = match directive {
            CspDirective::Script => &self.script_src,
            CspDirective::Style => &self.style_src,
            CspDirective::Img => &self.img_src,
            CspDirective::Connect => &self.connect_src,
            CspDirective::Font => &self.font_src,
            CspDirective::Frame => &self.frame_src,
            CspDirective::Media => &self.media_src,
            CspDirective::Object => &self.object_src,
            CspDirective::FormAction => &self.form_action,
        };
        specific.as_deref().unwrap_or(self.default_src.as_slice())
    }

    /// The raw source-list governing `directive` (after the default-src
    /// fallback). Public so callers can run the SEPARATE inline/nonce/hash
    /// decision path that `allows_source` (a URL matcher) deliberately does
    /// not handle.
    pub fn directive_list(&self, directive: CspDirective) -> &[String] {
        self.source_list(directive)
    }

    /// Whether `directive`'s source list contains a `'nonce-...'` or
    /// `'sha256-...'`/`'sha384-...'`/`'sha512-...'` token. When it does, the
    /// CSP spec says `'unsafe-inline'` is IGNORED for elements (Chrome
    /// behavior) — inline execution then requires a matching nonce or hash.
    fn has_nonce_or_hash(list: &[String]) -> bool {
        list.iter().any(|s| {
            let t = s.trim().trim_matches('\'');
            t.starts_with("nonce-")
                || t.starts_with("sha256-")
                || t.starts_with("sha384-")
                || t.starts_with("sha512-")
        })
    }

    /// Whether classic `'unsafe-inline'` execution is permitted for
    /// `directive`. Per the CSP spec, `'unsafe-inline'` is honored ONLY when
    /// the list contains no nonce- and no hash- token; otherwise it is ignored
    /// (a real XSS-bypass guard — top_traps #3).
    pub fn inline_allowed_unsafe(&self, directive: CspDirective) -> bool {
        let list = self.source_list(directive);
        list.iter().any(|s| s.trim() == "'unsafe-inline'")
            && !Self::has_nonce_or_hash(list)
    }

    /// The base64 nonce values (the part after `'nonce-`) declared for
    /// `directive`. Matched case-sensitively against an element's `nonce`.
    pub fn nonces(&self, directive: CspDirective) -> Vec<String> {
        self.source_list(directive)
            .iter()
            .filter_map(|s| {
                let t = s.trim().trim_matches('\'');
                t.strip_prefix("nonce-").map(|n| n.to_string())
            })
            .collect()
    }

    /// The `(alg, base64)` hash sources declared for `directive`
    /// (`'sha256-...'`/`'sha384-...'`/`'sha512-...'`).
    pub fn hashes(&self, directive: CspDirective) -> Vec<(String, String)> {
        self.source_list(directive)
            .iter()
            .filter_map(|s| {
                let t = s.trim().trim_matches('\'');
                for alg in ["sha256", "sha384", "sha512"] {
                    if let Some(b64) = t.strip_prefix(&format!("{alg}-")) {
                        return Some((alg.to_string(), b64.to_string()));
                    }
                }
                None
            })
            .collect()
    }

    /// Whether the inline source list for `directive` is empty (no specific
    /// directive AND no default-src) → allow-all default (today's behavior).
    pub fn inline_list_empty(&self, directive: CspDirective) -> bool {
        self.source_list(directive).is_empty()
    }

    /// Whether the source list for `directive` is the explicit `'none'` deny.
    pub fn is_none(&self, directive: CspDirective) -> bool {
        self.source_list(directive).iter().any(|s| s.trim() == "'none'")
    }

    /// Whether `candidate_url` is allowed for `directive` under this
    /// policy. `page_origin` is the document the policy was attached to.
    pub fn allows_source(
        &self,
        directive: CspDirective,
        page_origin: &Origin,
        candidate_url: &Url,
    ) -> bool {
        let list = self.source_list(directive);
        if list.is_empty() {
            // No directive AND no default-src → allow everything (matches
            // browser default-permit when no policy is present).
            return true;
        }
        // 'none' is the explicit deny-all sentinel.
        if list.iter().any(|s| s == "'none'") {
            return false;
        }
        let target_origin = Origin::of(candidate_url);
        for src in list {
            if matches_source(src, page_origin, &target_origin, candidate_url) {
                return true;
            }
        }
        false
    }
}

fn matches_source(
    src: &str,
    page_origin: &Origin,
    target_origin: &Origin,
    target_url: &Url,
) -> bool {
    let s = src.trim();
    match s {
        "*" => return true,
        "'self'" => return page_origin == target_origin,
        "'unsafe-inline'" | "'unsafe-eval'" | "'strict-dynamic'" => return false,
        "data:" => return target_url.scheme.as_str() == "data",
        "blob:" => return target_url.scheme.as_str() == "blob",
        "https:" => return target_url.scheme.as_str() == "https",
        "http:" => return target_url.scheme.as_str() == "http",
        _ => {}
    }
    // Hostname / host pattern.
    // Pattern shapes:
    //   - `https://*.example.com:443` — scheme + host wildcard + port
    //   - `*.example.com` — host pattern, any scheme
    //   - `example.com` — exact host
    let (scheme, rest) = if let Some(r) = s.strip_prefix("https://") {
        (Some("https"), r)
    } else if let Some(r) = s.strip_prefix("http://") {
        (Some("http"), r)
    } else if let Some(r) = s.strip_prefix("ws://") {
        (Some("ws"), r)
    } else if let Some(r) = s.strip_prefix("wss://") {
        (Some("wss"), r)
    } else {
        (None, s)
    };
    if let Some(req_scheme) = scheme {
        if target_url.scheme.as_str() != req_scheme {
            return false;
        }
    }
    let (host_pat, port_pat) = match rest.split_once(':') {
        Some((h, p)) => (h, Some(p)),
        None => (rest, None),
    };
    if !host_matches(host_pat, &target_url.host) {
        return false;
    }
    if let Some(port_pat) = port_pat {
        if port_pat != "*" {
            let target_port = target_url.port.map(|p| p.to_string()).unwrap_or_default();
            if port_pat != target_port {
                return false;
            }
        }
    }
    true
}

fn host_matches(pattern: &str, host: &str) -> bool {
    let host = host.to_ascii_lowercase();
    let pattern = pattern.to_ascii_lowercase();
    if let Some(rest) = pattern.strip_prefix("*.") {
        // CSP spec §6.7.2.5 — `*.example.com` requires at least one
        // subdomain label, so `example.com` itself does NOT match.
        // Only `host.ends_with(".{rest}")` satisfies the rule; the
        // `host == rest` branch (bare apex match) is intentionally removed.
        return host.ends_with(&format!(".{rest}"));
    }
    pattern == host
}

#[cfg(test)]
mod tests {
    use super::*;

    fn url(s: &str) -> Url {
        Url::parse(s).unwrap()
    }

    #[test]
    fn origin_basic() {
        assert_eq!(
            Origin::of(&url("https://example.com/x")).0,
            "https://example.com"
        );
        assert_eq!(Origin::of(&url("http://a.b:8080/")).0, "http://a.b:8080");
    }

    #[test]
    fn cors_same_origin_passes() {
        let req = Origin::of(&url("https://a.com/x"));
        let tgt = req.clone();
        assert_eq!(
            cors_decision(&req, &tgt, RequestMode::Cors, "GET", &[]),
            CorsDecision::SameOrigin
        );
    }

    #[test]
    fn cors_simple_get() {
        let a = Origin::of(&url("https://a.com/"));
        let b = Origin::of(&url("https://b.com/"));
        let d = cors_decision(&a, &b, RequestMode::Cors, "GET", &[]);
        assert_eq!(d, CorsDecision::SimpleCorsCheck);
    }

    #[test]
    fn cors_preflight_on_put() {
        let a = Origin::of(&url("https://a.com/"));
        let b = Origin::of(&url("https://b.com/"));
        let d = cors_decision(&a, &b, RequestMode::Cors, "PUT", &[]);
        assert!(matches!(d, CorsDecision::Preflight { .. }));
    }

    #[test]
    fn cors_response_validates_acao() {
        let target = Origin::of(&url("https://b.com/"));
        let ok = validate_cors_response(
            &[("Access-Control-Allow-Origin".into(), "https://b.com".into())],
            &target,
            &[],
            false,
        );
        assert!(ok.is_ok());
    }

    #[test]
    fn csp_parses_self_directive() {
        let p = parse_csp("default-src 'self'; script-src 'self' https://cdn.example.com");
        assert_eq!(p.default_src, vec!["'self'"]);
        assert_eq!(
            p.script_src.as_deref().unwrap(),
            ["'self'", "https://cdn.example.com"]
        );
    }

    #[test]
    fn csp_blocks_cross_origin_script() {
        let p = parse_csp("script-src 'self'");
        let page = Origin::of(&url("https://app.example.com/"));
        let bad = url("https://evil.com/x.js");
        assert!(!p.allows_source(CspDirective::Script, &page, &bad));
    }

    #[test]
    fn csp_wildcard_subdomain_allowed() {
        let p = parse_csp("img-src *.cdn.example.com");
        let page = Origin::of(&url("https://app.example.com/"));
        let ok = url("https://images.cdn.example.com/a.png");
        assert!(p.allows_source(CspDirective::Img, &page, &ok));
    }

    // --- Bug 1: Content-Type CORS safelist ---

    #[test]
    fn cors_post_application_json_requires_preflight() {
        // application/json is NOT a safelisted Content-Type — must preflight.
        let a = Origin::of(&url("https://a.com/"));
        let b = Origin::of(&url("https://b.com/"));
        let headers = vec![
            ("content-type".to_string(), "application/json".to_string()),
        ];
        let d = cors_decision(&a, &b, RequestMode::Cors, "POST", &headers);
        assert!(
            matches!(d, CorsDecision::Preflight { .. }),
            "POST application/json must trigger preflight, got {d:?}"
        );
    }

    #[test]
    fn cors_post_form_urlencoded_is_simple() {
        // application/x-www-form-urlencoded IS safelisted.
        let a = Origin::of(&url("https://a.com/"));
        let b = Origin::of(&url("https://b.com/"));
        let headers = vec![(
            "content-type".to_string(),
            "application/x-www-form-urlencoded".to_string(),
        )];
        let d = cors_decision(&a, &b, RequestMode::Cors, "POST", &headers);
        assert_eq!(
            d,
            CorsDecision::SimpleCorsCheck,
            "POST form-urlencoded should be simple"
        );
    }

    #[test]
    fn cors_post_text_plain_is_simple() {
        let a = Origin::of(&url("https://a.com/"));
        let b = Origin::of(&url("https://b.com/"));
        let headers = vec![("content-type".to_string(), "text/plain; charset=utf-8".to_string())];
        let d = cors_decision(&a, &b, RequestMode::Cors, "POST", &headers);
        assert_eq!(d, CorsDecision::SimpleCorsCheck);
    }

    #[test]
    fn cors_post_multipart_is_simple() {
        let a = Origin::of(&url("https://a.com/"));
        let b = Origin::of(&url("https://b.com/"));
        let headers = vec![(
            "content-type".to_string(),
            "multipart/form-data; boundary=abc".to_string(),
        )];
        let d = cors_decision(&a, &b, RequestMode::Cors, "POST", &headers);
        assert_eq!(d, CorsDecision::SimpleCorsCheck);
    }

    // --- Bug 2: ACAO null ---

    #[test]
    fn cors_acao_null_rejected_for_non_opaque_origin() {
        // A regular https:// page origin must NOT match ACAO: null.
        let target = Origin::of(&url("https://api.example.com/"));
        let result = validate_cors_response(
            &[("Access-Control-Allow-Origin".into(), "null".into())],
            &target,
            &[],
            false,
        );
        assert!(result.is_err(), "ACAO null must fail for non-opaque origins");
    }

    #[test]
    fn cors_acao_null_accepted_for_opaque_origin() {
        // An opaque origin (empty scheme/host → "null") should match ACAO: null.
        let opaque = Origin("null".into());
        let result = validate_cors_response(
            &[("Access-Control-Allow-Origin".into(), "null".into())],
            &opaque,
            &[],
            false,
        );
        assert!(result.is_ok(), "ACAO null must be OK for opaque origins");
    }

    // --- Bug 3: CSP wildcard apex ---

    #[test]
    fn csp_wildcard_does_not_match_apex() {
        // *.example.com must NOT match example.com itself.
        let p = parse_csp("img-src *.example.com");
        let page = Origin::of(&url("https://app.example.com/"));
        let apex = url("https://example.com/img.png");
        assert!(
            !p.allows_source(CspDirective::Img, &page, &apex),
            "*.example.com must not match bare example.com"
        );
    }

    #[test]
    fn csp_wildcard_matches_subdomain() {
        // *.example.com MUST match sub.example.com.
        let p = parse_csp("img-src *.example.com");
        let page = Origin::of(&url("https://app.example.com/"));
        let sub = url("https://sub.example.com/img.png");
        assert!(
            p.allows_source(CspDirective::Img, &page, &sub),
            "*.example.com must match sub.example.com"
        );
    }

    // --- M9.1: inline / nonce / hash helpers ---

    #[test]
    fn csp_inline_unsafe_honored_without_nonce_or_hash() {
        let p = parse_csp("script-src 'self' 'unsafe-inline'");
        assert!(p.inline_allowed_unsafe(CspDirective::Script));
    }

    #[test]
    fn csp_inline_unsafe_ignored_when_nonce_present() {
        // unsafe-inline is IGNORED when a nonce token is also present (XSS guard).
        let p = parse_csp("script-src 'unsafe-inline' 'nonce-abc123'");
        assert!(!p.inline_allowed_unsafe(CspDirective::Script));
    }

    #[test]
    fn csp_inline_unsafe_ignored_when_hash_present() {
        let p = parse_csp("script-src 'unsafe-inline' 'sha256-AAAA'");
        assert!(!p.inline_allowed_unsafe(CspDirective::Script));
    }

    #[test]
    fn csp_nonces_extracted() {
        let p = parse_csp("script-src 'self' 'nonce-abc123' 'nonce-def456'");
        let n = p.nonces(CspDirective::Script);
        assert_eq!(n, vec!["abc123".to_string(), "def456".to_string()]);
    }

    #[test]
    fn csp_hashes_extracted() {
        let p = parse_csp("script-src 'sha256-AAAA' 'sha384-BBBB'");
        let h = p.hashes(CspDirective::Script);
        assert_eq!(
            h,
            vec![
                ("sha256".to_string(), "AAAA".to_string()),
                ("sha384".to_string(), "BBBB".to_string())
            ]
        );
    }

    #[test]
    fn csp_inline_list_empty_when_no_directive() {
        // img-src only set → script-src inline list is empty → allow-all.
        let p = parse_csp("img-src 'self'");
        assert!(p.inline_list_empty(CspDirective::Script));
    }

    #[test]
    fn csp_inline_none_blocks() {
        let p = parse_csp("script-src 'none'");
        assert!(p.is_none(CspDirective::Script));
        assert!(!p.inline_allowed_unsafe(CspDirective::Script));
    }
}
