//! Per-fetch security enforcement — CSP / CORP / SameSite.
//!
//! Consolidates the gate-keeping the network path applies before
//! issuing a request: Content-Security-Policy directives from the
//! document, Cross-Origin-Resource-Policy from the response, and
//! SameSite handling from the cookie jar.

use crate::cookies::{RequestSite, SameSite};
use crate::security::{CspDirective, Origin, parse_csp};
use cv_url::Url;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FetchDecision {
    Allow,
    BlockedByCsp,
    BlockedByCorp,
    BlockedBySameSite,
}

/// Evaluate every gate. Returns the first failure or Allow if all pass.
pub fn evaluate(
    document_csp_header: Option<&str>,
    target_origin: &Origin,
    response_corp: Option<&str>,
    document_origin: &Origin,
    samesite: SameSite,
    request_site: RequestSite,
    is_top_level_nav: bool,
    is_https: bool,
) -> FetchDecision {
    if let Some(csp) = document_csp_header {
        let policy = parse_csp(csp);
        let target_url = match Url::parse(&target_origin.0) {
            Ok(u) => u,
            Err(_) => return FetchDecision::Allow,
        };
        if !policy.allows_source(CspDirective::Connect, document_origin, &target_url) {
            return FetchDecision::BlockedByCsp;
        }
    }
    if let Some(corp) = response_corp {
        if !corp_permits(corp, target_origin, document_origin) {
            return FetchDecision::BlockedByCorp;
        }
    }
    // SameSite check: emulate the cookies::CookieJar logic inline since
    // the helper is module-private.
    let permitted = match samesite {
        SameSite::Strict => matches!(request_site, RequestSite::SameSite),
        SameSite::Lax => matches!(request_site, RequestSite::SameSite) || is_top_level_nav,
        SameSite::None => is_https,
        SameSite::Unset => matches!(request_site, RequestSite::SameSite) || is_top_level_nav,
    };
    if !permitted {
        return FetchDecision::BlockedBySameSite;
    }
    FetchDecision::Allow
}

fn corp_permits(corp: &str, target: &Origin, doc: &Origin) -> bool {
    let v = corp.trim().to_ascii_lowercase();
    match v.as_str() {
        "same-origin" => target == doc,
        "same-site" => same_site(target, doc),
        "cross-origin" => true,
        _ => true,
    }
}

fn same_site(a: &Origin, b: &Origin) -> bool {
    let a_site = registrable_domain(host_of(&a.0));
    let b_site = registrable_domain(host_of(&b.0));
    a_site == b_site
}

fn host_of(origin: &str) -> &str {
    let s = origin.splitn(2, "://").nth(1).unwrap_or(origin);
    let h = s.split('/').next().unwrap_or(s);
    h.split(':').next().unwrap_or(h)
}

fn registrable_domain(host: &str) -> String {
    // Was last-two-labels (returned `co.uk` / `github.io` as registrable — a
    // same-site security bug). Now uses the real PSL (Milestone 1.5).
    crate::psl::registrable_domain(host)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn corp_same_origin_blocks_cross() {
        let target = Origin("https://evil.com".into());
        let doc = Origin("https://example.com".into());
        let d = evaluate(
            None,
            &target,
            Some("same-origin"),
            &doc,
            SameSite::Lax,
            RequestSite::SameSite,
            false,
            true,
        );
        assert_eq!(d, FetchDecision::BlockedByCorp);
    }
}
