//! Fetch Metadata request headers (`Sec-Fetch-*`) + `Referer` computation
//! per the Referrer Policy.
//!
//! Two browser-controlled (forbidden) header families ride on (almost)
//! every request Chrome issues:
//!
//! * **Fetch Metadata** — `Sec-Fetch-Site`, `Sec-Fetch-Mode`,
//!   `Sec-Fetch-Dest`, `Sec-Fetch-User`. These let the server reason about
//!   *who* and *how* a request was made. Specified by the W3C Fetch
//!   Metadata Request Headers draft
//!   (<https://w3c.github.io/webappsec-fetch-metadata/>) and implemented in
//!   Chromium at `services/network/sec_header_helpers.cc`
//!   (`SetFetchMetadataHeaders`). They are emitted **only on
//!   potentially-trustworthy request URLs** (HTTPS, localhost, …) — plain
//!   `http://` to a public host gets none.
//!
//! * **Referer** — the (mis-spelled, per RFC 9110 §10.1.3) referrer URL,
//!   trimmed according to the document's **Referrer Policy** (W3C Referrer
//!   Policy, <https://www.w3.org/TR/referrer-policy/>, §8.3 "Determine
//!   request's Referrer" + §8.4 strip-url). The default policy is
//!   `strict-origin-when-cross-origin`. The URL's fragment, username and
//!   password are ALWAYS stripped (§8.4) before it goes on the wire.
//!
//! This module is the pure-logic core: given the request URL, the
//! initiator origin, the document referrer + policy, and the resource kind,
//! it produces the exact header bytes. The wire builder in `http1.rs`
//! consumes [`FetchMetadata`] and appends the headers.

use cv_url::{Scheme, Url};

/// A request's *destination* — the kind of resource being fetched. Mirrors
/// Fetch's request-destination enum. Serialized in `Sec-Fetch-Dest`; the
/// empty destination is reported as the explicit token `empty`
/// (Fetch Metadata §"set `Sec-Fetch-Dest`").
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Destination {
    /// Top-level / nested document navigation (`<a>`, address bar, iframe).
    Document,
    /// `<img>`, `background-image`, etc.
    Image,
    /// Classic or module `<script>` / `importScripts`.
    Script,
    /// `<link rel=stylesheet>` / `@import`.
    Style,
    /// `@font-face` / web font.
    Font,
    /// `<audio>` / `<video>` media.
    Audio,
    Video,
    /// `<track>` text track.
    Track,
    /// `fetch()` / `XMLHttpRequest` with no specific destination — the
    /// "empty" destination, serialized as `empty`.
    Empty,
}

impl Destination {
    /// The `Sec-Fetch-Dest` token. The empty destination maps to `empty`
    /// (Fetch Metadata: "We map Fetch's empty string destination onto an
    /// explicit `empty` token").
    pub fn token(self) -> &'static str {
        match self {
            Self::Document => "document",
            Self::Image => "image",
            Self::Script => "script",
            Self::Style => "style",
            Self::Font => "font",
            Self::Audio => "audio",
            Self::Video => "video",
            Self::Track => "track",
            Self::Empty => "empty",
        }
    }
}

/// A request's *mode*. Serialized verbatim in `Sec-Fetch-Mode` (Fetch
/// Metadata: "The header value directly mirrors the request's mode").
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FetchMode {
    /// Top-level / nested navigation.
    Navigate,
    /// CORS-enabled cross-origin fetch.
    Cors,
    /// `no-cors` (opaque) — `<img>`, classic `<script src>`, CSS, fonts.
    NoCors,
    /// `same-origin` — must not leave the origin.
    SameOrigin,
    /// WebSocket handshake.
    Websocket,
}

impl FetchMode {
    pub fn token(self) -> &'static str {
        match self {
            Self::Navigate => "navigate",
            Self::Cors => "cors",
            Self::NoCors => "no-cors",
            Self::SameOrigin => "same-origin",
            Self::Websocket => "websocket",
        }
    }
}

/// The Referrer Policy in effect for the document making the request, per
/// the W3C Referrer Policy spec §8.2 referrer-policy tokens.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReferrerPolicy {
    NoReferrer,
    NoReferrerWhenDowngrade,
    SameOrigin,
    Origin,
    StrictOrigin,
    OriginWhenCrossOrigin,
    /// The default policy when none is specified (Referrer Policy §3 +
    /// Fetch). Chrome's default is `strict-origin-when-cross-origin`.
    StrictOriginWhenCrossOrigin,
    UnsafeUrl,
}

impl Default for ReferrerPolicy {
    fn default() -> Self {
        // Fetch / Referrer-Policy default since Chrome 85.
        Self::StrictOriginWhenCrossOrigin
    }
}

impl ReferrerPolicy {
    /// Parse a Referrer-Policy token (response header value, `<meta
    /// name=referrer>` content, or `referrerpolicy` attribute). Unknown /
    /// empty tokens return `None` so the caller can keep the inherited
    /// policy (Referrer Policy §8.1: "If policy is the empty string … do
    /// nothing"). Comma-separated lists take the last valid token
    /// (Referrer Policy §"parse a referrer policy from a Referrer-Policy
    /// header").
    pub fn parse(value: &str) -> Option<Self> {
        let mut found = None;
        for token in value.split(',') {
            let t = token.trim().to_ascii_lowercase();
            let p = match t.as_str() {
                "no-referrer" => Self::NoReferrer,
                "no-referrer-when-downgrade" => Self::NoReferrerWhenDowngrade,
                "same-origin" => Self::SameOrigin,
                "origin" => Self::Origin,
                "strict-origin" => Self::StrictOrigin,
                "origin-when-cross-origin" => Self::OriginWhenCrossOrigin,
                "strict-origin-when-cross-origin" => Self::StrictOriginWhenCrossOrigin,
                "unsafe-url" => Self::UnsafeUrl,
                _ => continue,
            };
            found = Some(p);
        }
        found
    }
}

/// Everything the wire builder needs to emit `Sec-Fetch-*` + `Referer` for
/// one request. Carried on [`crate::Request`].
#[derive(Debug, Clone)]
pub struct FetchMetadata {
    /// The resource kind → `Sec-Fetch-Dest`.
    pub destination: Destination,
    /// The request mode → `Sec-Fetch-Mode`.
    pub mode: FetchMode,
    /// The origin of the document that initiated this request, if any. A
    /// browser-initiated navigation (typed URL, bookmark, new tab) has
    /// `None` here, which yields `Sec-Fetch-Site: none`.
    pub initiator: Option<Origin>,
    /// Whether this navigation was caused by genuine user activation
    /// (click on a link, Enter in the address bar). Drives
    /// `Sec-Fetch-User: ?1` — emitted only for navigation requests that
    /// were user-activated.
    pub user_activated: bool,
    /// The document's referrer URL (the page making the request), if any,
    /// plus the Referrer Policy that governs how much of it is sent. `None`
    /// referrer ⇒ no `Referer` header.
    pub referrer: Option<Url>,
    /// The Referrer Policy in effect for the initiating document.
    pub policy: ReferrerPolicy,
}

impl FetchMetadata {
    /// A document navigation (top-level). Default mode/dest for a nav.
    pub fn navigation(initiator: Option<Origin>, user_activated: bool) -> Self {
        Self {
            destination: Destination::Document,
            mode: FetchMode::Navigate,
            initiator,
            user_activated,
            referrer: None,
            policy: ReferrerPolicy::default(),
        }
    }

    /// A subresource fetch (`<img>`, `<script>`, CSS, font, …). Subresources
    /// are never user-activated navigations, so `Sec-Fetch-User` is never
    /// emitted for them.
    pub fn subresource(destination: Destination, mode: FetchMode, initiator: Option<Origin>) -> Self {
        Self {
            destination,
            mode,
            initiator,
            user_activated: false,
            referrer: None,
            policy: ReferrerPolicy::default(),
        }
    }

    /// Attach the document referrer + policy.
    pub fn with_referrer(mut self, referrer: Option<Url>, policy: ReferrerPolicy) -> Self {
        self.referrer = referrer;
        self.policy = policy;
        self
    }
}

/// A minimal origin tuple `(scheme, host, port)`. We carry our own copy
/// (rather than `cv_url::Origin`) so `FetchMetadata` is self-contained and
/// callers can build it from any source. Port is the *effective* port
/// (default-filled) so `https://a.com` and `https://a.com:443` compare
/// equal, matching HTML's origin tuple.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Origin {
    pub scheme: Scheme,
    pub host: String,
    pub port: Option<u16>,
}

impl Origin {
    pub fn of(url: &Url) -> Self {
        Self {
            scheme: url.scheme,
            host: url.host.to_ascii_lowercase(),
            port: url.effective_port(),
        }
    }

    fn same_origin(&self, other: &Self) -> bool {
        self.scheme == other.scheme && self.host == other.host && self.port == other.port
    }

    /// "Schemelessly same site"-ish: registrable domain equality. Chrome's
    /// `Sec-Fetch-Site` same-site rung also requires scheme equality
    /// (a `https`→`http` step on the same registrable domain is
    /// cross-site). We follow that.
    fn same_site(&self, other: &Self) -> bool {
        self.scheme == other.scheme
            && crate::psl::registrable_domain(&self.host)
                == crate::psl::registrable_domain(&other.host)
    }
}

/// Is `url` a *potentially trustworthy* URL (W3C Secure Contexts)?
/// Chrome only appends `Sec-Fetch-*` to such URLs. We cover the cases the
/// engine actually issues: any `https`/`wss` URL, and `http`/`ws` to
/// loopback (`localhost`, `127.0.0.0/8`, `[::1]`).
pub fn is_potentially_trustworthy(url: &Url) -> bool {
    match url.scheme {
        Scheme::Https | Scheme::Wss => true,
        Scheme::Http | Scheme::Ws => is_loopback_host(&url.host),
        // data:/file:/about:/blob: never carry Sec-Fetch-* in our path.
        _ => false,
    }
}

fn is_loopback_host(host: &str) -> bool {
    let h = host.trim().trim_start_matches('[').trim_end_matches(']');
    if h.eq_ignore_ascii_case("localhost") {
        return true;
    }
    if h == "::1" {
        return true;
    }
    // 127.0.0.0/8
    if let Ok(v4) = h.parse::<std::net::Ipv4Addr>() {
        return v4.octets()[0] == 127;
    }
    false
}

/// Compute the `Sec-Fetch-Site` value for `target` given the initiating
/// origin, per Fetch Metadata "set `Sec-Fetch-Site`":
///
/// * no initiator (browser-initiated navigation) ⇒ `none`
/// * initiator same-origin with target ⇒ `same-origin`
/// * same registrable domain + scheme ⇒ `same-site`
/// * otherwise ⇒ `cross-site`
///
/// (We compute against the final `target` rather than walking a redirect
/// chain; cross-site is downgrade-monotone, so a same-origin initiator that
/// is redirected cross-site lands on `cross-site` once redirect targets are
/// fed back in as the new `target` — which is how `http1.rs` re-issues.)
pub fn sec_fetch_site(initiator: Option<&Origin>, target: &Url) -> &'static str {
    let Some(init) = initiator else {
        return "none";
    };
    let tgt = Origin::of(target);
    if init.same_origin(&tgt) {
        "same-origin"
    } else if init.same_site(&tgt) {
        "same-site"
    } else {
        "cross-site"
    }
}

/// Compute the `Referer` header value for `target` per the Referrer Policy
/// algorithm (Referrer Policy §8.3 + §8.4 strip-url). Returns `None` when
/// no referrer is to be sent.
///
/// `referrer` is the FULL document URL; we strip fragment/username/password
/// always, and additionally strip path+query down to the origin when the
/// policy calls for "origin only".
pub fn compute_referer(
    policy: ReferrerPolicy,
    referrer: &Url,
    target: &Url,
) -> Option<String> {
    use ReferrerPolicy as Rp;

    // §8.4: local schemes (data:, blob:, about:, file:) never produce a
    // referrer. Only http(s)/ws(s) referrers are sent.
    if !matches!(
        referrer.scheme,
        Scheme::Http | Scheme::Https | Scheme::Ws | Scheme::Wss
    ) {
        return None;
    }

    let ref_origin = Origin::of(referrer);
    let tgt_origin = Origin::of(target);
    let same_origin = ref_origin.same_origin(&tgt_origin);

    // "Downgrade": the referrer environment is TLS-protected and the
    // request URL is NOT potentially trustworthy (HTTPS → plain HTTP).
    let referrer_is_tls = matches!(referrer.scheme, Scheme::Https | Scheme::Wss);
    let downgrade = referrer_is_tls && !is_potentially_trustworthy(target);

    let want_origin_only = match policy {
        Rp::NoReferrer => return None,
        Rp::UnsafeUrl => false,
        Rp::Origin => true,
        Rp::SameOrigin => {
            if same_origin {
                false
            } else {
                return None;
            }
        }
        Rp::OriginWhenCrossOrigin => !same_origin,
        Rp::StrictOrigin => {
            if downgrade {
                return None;
            }
            true
        }
        Rp::StrictOriginWhenCrossOrigin => {
            if same_origin {
                false
            } else if downgrade {
                return None;
            } else {
                true
            }
        }
        Rp::NoReferrerWhenDowngrade => {
            if downgrade {
                return None;
            }
            false
        }
    };

    Some(if want_origin_only {
        strip_to_origin(referrer)
    } else {
        strip_url(referrer)
    })
}

/// §8.4 strip-url with the origin-only flag UNSET: drop fragment, username,
/// password; keep scheme://host[:port]/path?query.
fn strip_url(url: &Url) -> String {
    let mut s = String::new();
    s.push_str(url.scheme.as_str());
    s.push_str("://");
    s.push_str(&url.host);
    if let Some(p) = url.port {
        if Some(p) != url.scheme.default_port() {
            s.push(':');
            s.push_str(&p.to_string());
        }
    }
    s.push_str(&url.path);
    if let Some(q) = &url.query {
        s.push('?');
        s.push_str(q);
    }
    s
}

/// §8.4 strip-url with the origin-only flag SET: serialize the origin
/// (scheme://host[:port]) with a trailing `/` — Chrome sends the origin's
/// URL form, e.g. `https://example.com/`.
fn strip_to_origin(url: &Url) -> String {
    let mut s = String::new();
    s.push_str(url.scheme.as_str());
    s.push_str("://");
    s.push_str(&url.host);
    if let Some(p) = url.port {
        if Some(p) != url.scheme.default_port() {
            s.push(':');
            s.push_str(&p.to_string());
        }
    }
    s.push('/');
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    fn url(s: &str) -> Url {
        Url::parse(s).unwrap()
    }

    // ---- Sec-Fetch-Site ----

    #[test]
    fn site_none_for_browser_initiated() {
        assert_eq!(sec_fetch_site(None, &url("https://example.com/")), "none");
    }

    #[test]
    fn site_same_origin() {
        let init = Origin::of(&url("https://example.com/page"));
        assert_eq!(
            sec_fetch_site(Some(&init), &url("https://example.com/img.png")),
            "same-origin"
        );
    }

    #[test]
    fn site_same_origin_default_port_equivalence() {
        // https://a.com and https://a.com:443 must be same-origin.
        let init = Origin::of(&url("https://a.com:443/"));
        assert_eq!(
            sec_fetch_site(Some(&init), &url("https://a.com/x")),
            "same-origin"
        );
    }

    #[test]
    fn site_same_site_subdomain() {
        // cdn.example.com vs www.example.com: same registrable domain.
        let init = Origin::of(&url("https://www.example.com/"));
        assert_eq!(
            sec_fetch_site(Some(&init), &url("https://cdn.example.com/a.js")),
            "same-site"
        );
    }

    #[test]
    fn site_cross_site_different_domain() {
        let init = Origin::of(&url("https://example.com/"));
        assert_eq!(
            sec_fetch_site(Some(&init), &url("https://evil.com/track")),
            "cross-site"
        );
    }

    #[test]
    fn site_cross_site_scheme_mismatch_same_domain() {
        // https://a.com → http://a.com is cross-site (scheme differs).
        let init = Origin::of(&url("https://a.com/"));
        assert_eq!(
            sec_fetch_site(Some(&init), &url("http://a.com/")),
            "cross-site"
        );
    }

    // ---- Destination / Mode tokens ----

    #[test]
    fn dest_document_for_nav_image_for_img() {
        let nav = FetchMetadata::navigation(None, true);
        assert_eq!(nav.destination.token(), "document");
        let img = FetchMetadata::subresource(
            Destination::Image,
            FetchMode::NoCors,
            Some(Origin::of(&url("https://example.com/"))),
        );
        assert_eq!(img.destination.token(), "image");
    }

    #[test]
    fn empty_destination_token() {
        assert_eq!(Destination::Empty.token(), "empty");
    }

    #[test]
    fn mode_tokens() {
        assert_eq!(FetchMode::Navigate.token(), "navigate");
        assert_eq!(FetchMode::Cors.token(), "cors");
        assert_eq!(FetchMode::NoCors.token(), "no-cors");
        assert_eq!(FetchMode::SameOrigin.token(), "same-origin");
        assert_eq!(FetchMode::Websocket.token(), "websocket");
    }

    // ---- potentially-trustworthy ----

    #[test]
    fn trustworthy_classification() {
        assert!(is_potentially_trustworthy(&url("https://a.com/")));
        assert!(!is_potentially_trustworthy(&url("http://a.com/")));
        assert!(is_potentially_trustworthy(&url("http://localhost/")));
        assert!(is_potentially_trustworthy(&url("http://127.0.0.1/")));
        assert!(is_potentially_trustworthy(&url("http://127.5.0.9/")));
        assert!(!is_potentially_trustworthy(&url("http://128.0.0.1/")));
    }

    // ---- Referrer Policy parsing ----

    #[test]
    fn policy_parse_known_and_default() {
        assert_eq!(ReferrerPolicy::parse("no-referrer"), Some(ReferrerPolicy::NoReferrer));
        assert_eq!(
            ReferrerPolicy::parse("strict-origin-when-cross-origin"),
            Some(ReferrerPolicy::StrictOriginWhenCrossOrigin)
        );
        assert_eq!(ReferrerPolicy::parse("garbage"), None);
        assert_eq!(ReferrerPolicy::default(), ReferrerPolicy::StrictOriginWhenCrossOrigin);
    }

    #[test]
    fn policy_parse_list_takes_last_valid() {
        // Browsers take the last recognized token in a comma list.
        assert_eq!(
            ReferrerPolicy::parse("no-referrer, strict-origin-when-cross-origin"),
            Some(ReferrerPolicy::StrictOriginWhenCrossOrigin)
        );
        assert_eq!(
            ReferrerPolicy::parse("unsafe-url, bogus"),
            Some(ReferrerPolicy::UnsafeUrl)
        );
    }

    // ---- Referer computation: per-policy matrix ----

    const SAME_A: &str = "https://example.com/page?x=1#frag";
    const SAME_B: &str = "https://example.com/other";
    const CROSS: &str = "https://other.com/landing";
    const HTTP_TGT: &str = "http://example.com/insecure"; // downgrade target

    #[test]
    fn referer_strips_fragment_and_credentials() {
        // unsafe-url sends the full URL but ALWAYS strips fragment + creds.
        let r = compute_referer(
            ReferrerPolicy::UnsafeUrl,
            &url("https://user:pass@example.com/page?q=1#frag"),
            &url(CROSS),
        );
        assert_eq!(r.as_deref(), Some("https://example.com/page?q=1"));
    }

    #[test]
    fn referer_no_referrer() {
        assert_eq!(
            compute_referer(ReferrerPolicy::NoReferrer, &url(SAME_A), &url(SAME_B)),
            None
        );
        assert_eq!(
            compute_referer(ReferrerPolicy::NoReferrer, &url(SAME_A), &url(CROSS)),
            None
        );
    }

    #[test]
    fn referer_origin_policy() {
        // origin: always just the origin, even same-origin.
        assert_eq!(
            compute_referer(ReferrerPolicy::Origin, &url(SAME_A), &url(SAME_B)).as_deref(),
            Some("https://example.com/")
        );
        assert_eq!(
            compute_referer(ReferrerPolicy::Origin, &url(SAME_A), &url(CROSS)).as_deref(),
            Some("https://example.com/")
        );
    }

    #[test]
    fn referer_same_origin_policy() {
        // same-origin: full URL same-origin, nothing cross-origin.
        assert_eq!(
            compute_referer(ReferrerPolicy::SameOrigin, &url(SAME_A), &url(SAME_B)).as_deref(),
            Some("https://example.com/page?x=1")
        );
        assert_eq!(
            compute_referer(ReferrerPolicy::SameOrigin, &url(SAME_A), &url(CROSS)),
            None
        );
    }

    #[test]
    fn referer_origin_when_cross_origin() {
        // full URL same-origin, origin-only cross-origin.
        assert_eq!(
            compute_referer(ReferrerPolicy::OriginWhenCrossOrigin, &url(SAME_A), &url(SAME_B))
                .as_deref(),
            Some("https://example.com/page?x=1")
        );
        assert_eq!(
            compute_referer(ReferrerPolicy::OriginWhenCrossOrigin, &url(SAME_A), &url(CROSS))
                .as_deref(),
            Some("https://example.com/")
        );
    }

    #[test]
    fn referer_strict_origin() {
        // origin-only always, EXCEPT https->http downgrade sends nothing.
        assert_eq!(
            compute_referer(ReferrerPolicy::StrictOrigin, &url(SAME_A), &url(SAME_B)).as_deref(),
            Some("https://example.com/")
        );
        assert_eq!(
            compute_referer(ReferrerPolicy::StrictOrigin, &url(SAME_A), &url(CROSS)).as_deref(),
            Some("https://example.com/")
        );
        assert_eq!(
            compute_referer(ReferrerPolicy::StrictOrigin, &url(SAME_A), &url(HTTP_TGT)),
            None
        );
    }

    #[test]
    fn referer_no_referrer_when_downgrade() {
        // full URL always, EXCEPT https->http downgrade sends nothing.
        assert_eq!(
            compute_referer(ReferrerPolicy::NoReferrerWhenDowngrade, &url(SAME_A), &url(CROSS))
                .as_deref(),
            Some("https://example.com/page?x=1")
        );
        assert_eq!(
            compute_referer(ReferrerPolicy::NoReferrerWhenDowngrade, &url(SAME_A), &url(HTTP_TGT)),
            None
        );
    }

    #[test]
    fn referer_strict_origin_when_cross_origin_default() {
        // THE DEFAULT. Same-origin: full URL. Cross-origin: origin only.
        // Downgrade (https->http): nothing.
        assert_eq!(
            compute_referer(
                ReferrerPolicy::StrictOriginWhenCrossOrigin,
                &url(SAME_A),
                &url(SAME_B)
            )
            .as_deref(),
            Some("https://example.com/page?x=1")
        );
        assert_eq!(
            compute_referer(
                ReferrerPolicy::StrictOriginWhenCrossOrigin,
                &url(SAME_A),
                &url(CROSS)
            )
            .as_deref(),
            Some("https://example.com/")
        );
        assert_eq!(
            compute_referer(
                ReferrerPolicy::StrictOriginWhenCrossOrigin,
                &url(SAME_A),
                &url(HTTP_TGT)
            ),
            None
        );
    }

    #[test]
    fn referer_unsafe_url() {
        // Full URL everywhere, even on downgrade (the unsafe part).
        assert_eq!(
            compute_referer(ReferrerPolicy::UnsafeUrl, &url(SAME_A), &url(HTTP_TGT)).as_deref(),
            Some("https://example.com/page?x=1")
        );
        assert_eq!(
            compute_referer(ReferrerPolicy::UnsafeUrl, &url(SAME_A), &url(CROSS)).as_deref(),
            Some("https://example.com/page?x=1")
        );
    }

    #[test]
    fn referer_non_default_port_preserved() {
        let r = compute_referer(
            ReferrerPolicy::Origin,
            &url("https://example.com:8443/page"),
            &url(CROSS),
        );
        assert_eq!(r.as_deref(), Some("https://example.com:8443/"));
    }

    #[test]
    fn referer_data_url_referrer_is_none() {
        // A data: page has no usable referrer.
        let r = compute_referer(
            ReferrerPolicy::UnsafeUrl,
            &url("data:text/html,hi"),
            &url(CROSS),
        );
        assert_eq!(r, None);
    }
}
