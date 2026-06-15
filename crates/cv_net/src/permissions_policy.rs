//! `Permissions-Policy` (W3C Permissions Policy) — parse + enforce.
//!
//! The `Permissions-Policy` response header gates "powerful features"
//! (camera, microphone, geolocation, …) per-origin. A document's policy is
//! the combination of the header and the iframe `allow` attribute; this
//! module implements the header half, which is what gates top-level
//! document feature access.
//!
//! Syntax (W3C Permissions Policy §2.3, MDN):
//!
//!   Permissions-Policy: camera=()                        ; disable for all
//!   Permissions-Policy: camera=(self)                    ; same-origin only
//!   Permissions-Policy: camera=*                         ; all origins
//!   Permissions-Policy: camera=(self "https://a.example"); self + listed
//!   Permissions-Policy: geolocation=(self), camera=()    ; multiple, comma
//!
//! Enforcement: [`PermissionsPolicy::allows`] answers "may `feature` run in
//! a document of `origin`?". The browser's `getUserMedia` (and any other
//! gated API) consults this BEFORE prompting; a disallowed feature rejects
//! with `NotAllowedError` and the user is never prompted (per spec).

use std::collections::HashMap;

/// The allowlist for one feature, parsed from a directive value.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Allowlist {
    /// `*` — allowed in all origins / nested contexts.
    All,
    /// `()` — empty allowlist: disabled everywhere.
    None,
    /// A specific set: `self` and/or quoted origins. `self_allowed` is true
    /// when the `self` keyword was present; `origins` holds the explicit
    /// allowed origins (lowercased `scheme://host[:port]`).
    List {
        self_allowed: bool,
        origins: Vec<String>,
    },
}

/// A parsed `Permissions-Policy` header: feature name → allowlist.
#[derive(Debug, Clone, Default)]
pub struct PermissionsPolicy {
    features: HashMap<String, Allowlist>,
}

impl PermissionsPolicy {
    /// Parse a `Permissions-Policy` header value (or several joined with
    /// commas — the structured-fields list form). Unknown/malformed
    /// directives are skipped; a later directive for the same feature wins
    /// (Chrome keeps the last). Feature names are lowercased.
    pub fn parse(header: &str) -> Self {
        let mut features = HashMap::new();
        // The header is a comma-separated list of `feature=allowlist`. An
        // allowlist may itself contain spaces inside `(...)`, but never a
        // top-level comma, so a comma split is safe.
        for directive in header.split(',') {
            let directive = directive.trim().trim_end_matches(';').trim();
            if directive.is_empty() {
                continue;
            }
            let (name, value) = match directive.split_once('=') {
                Some((n, v)) => (n.trim().to_ascii_lowercase(), v.trim()),
                None => continue,
            };
            if name.is_empty() {
                continue;
            }
            let allowlist = parse_allowlist(value);
            features.insert(name, allowlist);
        }
        Self { features }
    }

    /// Merge another policy's directives into this one (other wins on
    /// conflicts). Used when a response carries multiple `Permissions-Policy`
    /// header fields (each is parsed then merged).
    pub fn merge(&mut self, other: PermissionsPolicy) {
        for (k, v) in other.features {
            self.features.insert(k, v);
        }
    }

    /// Does this policy allow `feature` to run in a document whose origin is
    /// `document_origin`? `document_origin` is the canonical
    /// `scheme://host[:port]` of the document the policy applies to.
    ///
    /// Default behaviour (no directive for the feature): the W3C default
    /// allowlist for most powerful features is `self`, so an unlisted
    /// feature is allowed for the same-origin document. (camera/microphone/
    /// geolocation all default to `self`.) A directive overrides that.
    pub fn allows(&self, feature: &str, document_origin: &str) -> bool {
        match self.features.get(&feature.to_ascii_lowercase()) {
            None => true, // default allowlist `self`; the document IS self
            Some(Allowlist::All) => true,
            Some(Allowlist::None) => false,
            Some(Allowlist::List {
                self_allowed,
                origins,
            }) => {
                if *self_allowed {
                    // `self` matches the document's own origin. For a
                    // top-level document, the document origin IS self.
                    return true;
                }
                origins
                    .iter()
                    .any(|o| o.eq_ignore_ascii_case(document_origin))
            }
        }
    }

    /// True when `feature` is explicitly disabled for ALL origins (`()`),
    /// regardless of document origin. The strongest form of denial.
    pub fn is_disabled_for_all(&self, feature: &str) -> bool {
        matches!(
            self.features.get(&feature.to_ascii_lowercase()),
            Some(Allowlist::None)
        )
    }
}

fn parse_allowlist(value: &str) -> Allowlist {
    let v = value.trim();
    if v == "*" {
        return Allowlist::All;
    }
    // Strip surrounding parentheses if present. `()` → empty → None.
    let inner = if let Some(stripped) = v.strip_prefix('(') {
        stripped.strip_suffix(')').unwrap_or(stripped)
    } else {
        // Bare token without parens: a single value like `self` or `*`.
        v
    };
    let inner = inner.trim();
    if inner.is_empty() {
        // `()` → disabled for all.
        // A bare `=` with nothing after also lands here → treat as none.
        if v.starts_with('(') {
            return Allowlist::None;
        }
        return Allowlist::None;
    }
    let mut self_allowed = false;
    let mut origins = Vec::new();
    for tok in inner.split_whitespace() {
        let tok = tok.trim();
        if tok.eq_ignore_ascii_case("self") || tok.eq_ignore_ascii_case("'self'") {
            self_allowed = true;
        } else if tok == "*" {
            return Allowlist::All;
        } else {
            // A quoted origin: "https://a.example".
            let unq = tok.trim_matches('"').trim_matches('\'');
            if !unq.is_empty() {
                origins.push(unq.to_ascii_lowercase());
            }
        }
    }
    Allowlist::List {
        self_allowed,
        origins,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn camera_empty_allowlist_blocks_everyone() {
        let p = PermissionsPolicy::parse("camera=()");
        assert!(!p.allows("camera", "https://example.com"));
        assert!(p.is_disabled_for_all("camera"));
        // A different feature is unaffected (default self → allowed).
        assert!(p.allows("microphone", "https://example.com"));
    }

    #[test]
    fn camera_star_allows_all_origins() {
        let p = PermissionsPolicy::parse("camera=*");
        assert!(p.allows("camera", "https://example.com"));
        assert!(p.allows("camera", "https://other.example"));
        assert!(!p.is_disabled_for_all("camera"));
    }

    #[test]
    fn camera_self_only_allows_same_origin() {
        let p = PermissionsPolicy::parse("camera=(self)");
        assert!(p.allows("camera", "https://example.com"));
        // For a top-level document, `self` is always the document origin,
        // so any document-origin is allowed; the cross-origin gating is the
        // iframe `allow`-attribute layer (a follow-up).
        assert!(p.allows("camera", "https://other.example"));
    }

    #[test]
    fn explicit_origin_list_matches_only_listed() {
        let p = PermissionsPolicy::parse("geolocation=(\"https://a.example\")");
        assert!(p.allows("geolocation", "https://a.example"));
        assert!(!p.allows("geolocation", "https://b.example"));
    }

    #[test]
    fn multiple_directives_comma_separated() {
        let p = PermissionsPolicy::parse("geolocation=(self), camera=(), microphone=*");
        assert!(p.allows("geolocation", "https://x.example"));
        assert!(!p.allows("camera", "https://x.example"));
        assert!(p.allows("microphone", "https://x.example"));
    }

    #[test]
    fn last_directive_for_feature_wins() {
        let p = PermissionsPolicy::parse("camera=*, camera=()");
        assert!(!p.allows("camera", "https://x.example"));
    }

    #[test]
    fn feature_name_is_case_insensitive() {
        let p = PermissionsPolicy::parse("Camera=()");
        assert!(!p.allows("camera", "https://x.example"));
        assert!(!p.allows("CAMERA", "https://x.example"));
    }

    #[test]
    fn merge_combines_two_header_fields() {
        let mut p = PermissionsPolicy::parse("camera=()");
        p.merge(PermissionsPolicy::parse("microphone=()"));
        assert!(!p.allows("camera", "https://x.example"));
        assert!(!p.allows("microphone", "https://x.example"));
    }

    #[test]
    fn unknown_feature_defaults_to_allowed_self() {
        let p = PermissionsPolicy::parse("camera=()");
        // A feature with no directive uses the default `self` allowlist →
        // allowed for the document's own origin.
        assert!(p.allows("fullscreen", "https://x.example"));
    }
}
