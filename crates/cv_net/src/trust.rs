//! Trusted Types + Subresource Integrity + Mixed Content gates.
//!
//! Each module here returns an "allow / block" decision the fetch
//! pipeline + DOM sinks consult before letting work proceed.

/// Trusted Types — HTML sinks (innerHTML, document.write, etc.)
/// route their string argument through this policy before assigning.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TrustedTypeDecision {
    Allow,
    Block { reason: String },
}

#[derive(Debug, Default)]
pub struct TrustedTypePolicy {
    pub enforce: bool,
    pub allowlist: Vec<String>,
}

impl TrustedTypePolicy {
    pub fn evaluate(&self, marked_trusted: bool, sink: &str) -> TrustedTypeDecision {
        if !self.enforce {
            return TrustedTypeDecision::Allow;
        }
        if marked_trusted {
            return TrustedTypeDecision::Allow;
        }
        if self.allowlist.iter().any(|s| s == sink) {
            return TrustedTypeDecision::Allow;
        }
        TrustedTypeDecision::Block {
            reason: format!("untrusted assignment to {sink}"),
        }
    }
}

/// Subresource Integrity. The fetch pipeline holds the
/// `integrity="sha256-..."` hash from `<script>` / `<link>`; after
/// downloading the bytes, it compares against this.
pub fn sri_matches(integrity_attr: &str, actual_hex_sha256: &str) -> bool {
    let expected = integrity_attr.trim();
    if let Some(hex) = expected.strip_prefix("sha256-") {
        return hex.eq_ignore_ascii_case(actual_hex_sha256);
    }
    false
}

/// Mixed Content. HTTPS pages must not load passive HTTP subresources
/// (upgrade them) and must NEVER load active HTTP subresources
/// (script, iframe, websocket, fetch).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResourceKind {
    /// `<img>`, `<audio>`, `<video>` — auto-upgradeable.
    PassiveImage,
    /// `<script>`, `<iframe>`, fetch, websocket.
    ActiveScript,
    Stylesheet,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MixedContentDecision {
    Allow,
    Upgrade,
    Block,
}

pub fn mixed_content_decision(
    page_https: bool,
    resource_https: bool,
    kind: ResourceKind,
) -> MixedContentDecision {
    if !page_https || resource_https {
        return MixedContentDecision::Allow;
    }
    match kind {
        ResourceKind::PassiveImage => MixedContentDecision::Upgrade,
        ResourceKind::ActiveScript | ResourceKind::Stylesheet => MixedContentDecision::Block,
    }
}

// -------- Cross-origin isolation -----------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CrossOriginOpenerPolicy {
    /// Default — no isolation.
    UnsafeNone,
    /// Same-origin top-level — isolated browsing-context group.
    SameOrigin,
    /// Allows window.open popups even at SameOrigin.
    SameOriginAllowPopups,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CrossOriginEmbedderPolicy {
    UnsafeNone,
    RequireCorp,
    CredentialLess,
}

pub fn parse_coop(value: &str) -> CrossOriginOpenerPolicy {
    match value.trim() {
        "same-origin" => CrossOriginOpenerPolicy::SameOrigin,
        "same-origin-allow-popups" => CrossOriginOpenerPolicy::SameOriginAllowPopups,
        _ => CrossOriginOpenerPolicy::UnsafeNone,
    }
}

pub fn parse_coep(value: &str) -> CrossOriginEmbedderPolicy {
    match value.trim() {
        "require-corp" => CrossOriginEmbedderPolicy::RequireCorp,
        "credentialless" => CrossOriginEmbedderPolicy::CredentialLess,
        _ => CrossOriginEmbedderPolicy::UnsafeNone,
    }
}

/// `crossOriginIsolated` per HTML spec — gating SharedArrayBuffer.
pub fn is_cross_origin_isolated(
    coop: CrossOriginOpenerPolicy,
    coep: CrossOriginEmbedderPolicy,
) -> bool {
    !matches!(coop, CrossOriginOpenerPolicy::UnsafeNone)
        && !matches!(coep, CrossOriginEmbedderPolicy::UnsafeNone)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tt_disabled_always_allows() {
        let p = TrustedTypePolicy::default();
        let d = p.evaluate(false, "Element.innerHTML");
        assert_eq!(d, TrustedTypeDecision::Allow);
    }

    #[test]
    fn tt_enforced_blocks_unsigned_assignment() {
        let mut p = TrustedTypePolicy::default();
        p.enforce = true;
        let d = p.evaluate(false, "Element.innerHTML");
        assert!(matches!(d, TrustedTypeDecision::Block { .. }));
    }

    #[test]
    fn tt_trusted_value_passes_when_enforced() {
        let mut p = TrustedTypePolicy::default();
        p.enforce = true;
        assert_eq!(p.evaluate(true, "x"), TrustedTypeDecision::Allow);
    }

    #[test]
    fn sri_sha256_matches_lowercase_hex() {
        assert!(sri_matches("sha256-abcdef", "abcdef"));
        assert!(sri_matches("sha256-ABCDEF", "abcdef"));
        assert!(!sri_matches("sha256-abc", "def"));
    }

    #[test]
    fn mc_passive_https_to_http_upgrades() {
        assert_eq!(
            mixed_content_decision(true, false, ResourceKind::PassiveImage),
            MixedContentDecision::Upgrade
        );
    }

    #[test]
    fn mc_active_https_to_http_blocks() {
        assert_eq!(
            mixed_content_decision(true, false, ResourceKind::ActiveScript),
            MixedContentDecision::Block
        );
        assert_eq!(
            mixed_content_decision(true, false, ResourceKind::Stylesheet),
            MixedContentDecision::Block
        );
    }

    #[test]
    fn mc_http_to_http_is_allowed() {
        assert_eq!(
            mixed_content_decision(false, false, ResourceKind::ActiveScript),
            MixedContentDecision::Allow
        );
    }

    #[test]
    fn coop_parses_same_origin() {
        assert_eq!(
            parse_coop("same-origin"),
            CrossOriginOpenerPolicy::SameOrigin
        );
        assert_eq!(
            parse_coop("unsafe-none"),
            CrossOriginOpenerPolicy::UnsafeNone
        );
    }

    #[test]
    fn coep_parses_require_corp() {
        assert_eq!(
            parse_coep("require-corp"),
            CrossOriginEmbedderPolicy::RequireCorp
        );
    }

    #[test]
    fn isolation_requires_both_headers() {
        assert!(is_cross_origin_isolated(
            CrossOriginOpenerPolicy::SameOrigin,
            CrossOriginEmbedderPolicy::RequireCorp
        ));
        assert!(!is_cross_origin_isolated(
            CrossOriginOpenerPolicy::SameOrigin,
            CrossOriginEmbedderPolicy::UnsafeNone
        ));
        assert!(!is_cross_origin_isolated(
            CrossOriginOpenerPolicy::UnsafeNone,
            CrossOriginEmbedderPolicy::RequireCorp
        ));
    }

    #[test]
    fn mc_https_to_https_is_allowed() {
        assert_eq!(
            mixed_content_decision(true, true, ResourceKind::ActiveScript),
            MixedContentDecision::Allow
        );
    }
}
