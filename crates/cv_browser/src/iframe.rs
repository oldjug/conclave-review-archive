//! `<iframe>` — nested browsing context support.
//!
//! This module holds the *origin-/sandbox-/postMessage-policy* core for the
//! iframe element: the parts that are pure data transformations and therefore
//! unit-testable offline (no window, no network). The rendering wiring (parse
//! the child `src`/`srcdoc`, lay it out, bake it into the iframe box's
//! `embedded_image`) and the JS wiring (`contentWindow`/`contentDocument`,
//! cross-frame `postMessage` delivery) live in `main.rs` and call into the
//! decision functions here.
//!
//! Spec references:
//!   * HTML Standard §4.8.5 "The `iframe` element" (nested browsing context
//!     creation; `srcdoc` priority over `src`; the `about:srcdoc` document).
//!     <https://html.spec.whatwg.org/multipage/iframe-embed-object.html>
//!   * HTML Standard §4.8.5 + §"sandboxing flag set" — the `sandbox`
//!     attribute keyword tokens. Without `allow-scripts`, scripting is
//!     disabled; without `allow-same-origin`, the content document gets a
//!     fresh *opaque* origin (`null`) that compares unequal to everything.
//!   * HTML Standard §"Cross-origin objects" + WindowProxy/Location: a
//!     cross-origin `contentDocument` returns `null`; `contentWindow` is a
//!     restricted proxy (here: postMessage-only).
//!   * HTML Standard §"window.postMessage(message, targetOrigin, transfer)"
//!     and the MessageEvent delivered with `data`/`origin`/`source`.
//!     <https://html.spec.whatwg.org/multipage/web-messaging.html>

use cv_net::security::Origin;
use cv_url::Url;

/// The sandboxing flag set lowered to the booleans we actually act on.
///
/// A sandbox flag is "active" (restriction ON) when the corresponding
/// `allow-*` keyword is ABSENT from the attribute value. We store the
/// *capability* (allow-*) booleans here — `true` means the capability is
/// granted (restriction lifted), which is the inverse of the spec's flag.
/// This is the easier polarity to consume at call sites ("may this frame run
/// scripts?").
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SandboxFlags {
    /// Whether the element carried a `sandbox` attribute at all. When `false`,
    /// nothing is sandboxed (every capability granted, real origin).
    pub present: bool,
    /// `allow-scripts` — scripts (and auto-triggered features) may run.
    pub allow_scripts: bool,
    /// `allow-same-origin` — the content keeps its real origin instead of a
    /// fresh opaque origin.
    pub allow_same_origin: bool,
    /// `allow-forms` — form submission permitted.
    pub allow_forms: bool,
    /// `allow-popups` — `window.open` / target=_blank may spawn a context.
    pub allow_popups: bool,
    /// `allow-modals` — `alert`/`confirm`/`prompt` permitted.
    pub allow_modals: bool,
    /// `allow-top-navigation` — the frame may navigate the top-level context.
    pub allow_top_navigation: bool,
    /// `allow-pointer-lock`.
    pub allow_pointer_lock: bool,
    /// `allow-downloads`.
    pub allow_downloads: bool,
    /// `allow-orientation-lock`.
    pub allow_orientation_lock: bool,
    /// `allow-presentation`.
    pub allow_presentation: bool,
}

impl SandboxFlags {
    /// No `sandbox` attribute present: a normal nested context — every
    /// capability granted and the content uses its real origin.
    pub fn unsandboxed() -> Self {
        SandboxFlags {
            present: false,
            allow_scripts: true,
            allow_same_origin: true,
            allow_forms: true,
            allow_popups: true,
            allow_modals: true,
            allow_top_navigation: true,
            allow_pointer_lock: true,
            allow_downloads: true,
            allow_orientation_lock: true,
            allow_presentation: true,
        }
    }

    /// A bare `sandbox` attribute (no keywords) — the maximally restricted
    /// set: no scripts, opaque origin, no forms, no popups, no modals, etc.
    /// Every capability is denied.
    fn fully_sandboxed() -> Self {
        SandboxFlags {
            present: true,
            allow_scripts: false,
            allow_same_origin: false,
            allow_forms: false,
            allow_popups: false,
            allow_modals: false,
            allow_top_navigation: false,
            allow_pointer_lock: false,
            allow_downloads: false,
            allow_orientation_lock: false,
            allow_presentation: false,
        }
    }

    /// Parse the value of a `sandbox="..."` attribute into the flag set.
    ///
    /// Per the HTML Standard the value is an unordered set of space-separated,
    /// ASCII-case-insensitive keyword tokens; unknown tokens are ignored.
    /// Each recognised `allow-*` token lifts one restriction. A `None`
    /// argument means the attribute is absent ⇒ unsandboxed; `Some(value)`
    /// (including the empty string for a bare `sandbox`) starts from the
    /// fully-sandboxed set and grants only the listed capabilities.
    pub fn parse(attr_value: Option<&str>) -> Self {
        let Some(value) = attr_value else {
            return SandboxFlags::unsandboxed();
        };
        let mut f = SandboxFlags::fully_sandboxed();
        for token in value.split_whitespace() {
            // ASCII-case-insensitive keyword match.
            match token.to_ascii_lowercase().as_str() {
                "allow-scripts" => f.allow_scripts = true,
                "allow-same-origin" => f.allow_same_origin = true,
                "allow-forms" => f.allow_forms = true,
                "allow-popups" => f.allow_popups = true,
                "allow-modals" => f.allow_modals = true,
                "allow-top-navigation"
                | "allow-top-navigation-by-user-activation"
                | "allow-top-navigation-to-custom-protocols" => f.allow_top_navigation = true,
                "allow-pointer-lock" => f.allow_pointer_lock = true,
                "allow-downloads" => f.allow_downloads = true,
                "allow-orientation-lock" => f.allow_orientation_lock = true,
                "allow-presentation" => f.allow_presentation = true,
                // Unknown / unsupported tokens are ignored (forward-compat).
                _ => {}
            }
        }
        f
    }

    /// Whether the child document's scripts may execute. This is the gate the
    /// renderer consults before running any of the frame's `<script>`s.
    ///
    /// Per spec, scripting in a sandboxed frame requires BOTH `allow-scripts`
    /// and `allow-same-origin` is NOT required for scripts — but the
    /// well-known dangerous combination `allow-scripts allow-same-origin`
    /// lets a frame remove its own sandbox; we still honour scripts in that
    /// case (matching Chrome). The minimal rule: run scripts iff
    /// `allow-scripts` (or no sandbox at all).
    pub fn scripts_enabled(&self) -> bool {
        self.allow_scripts
    }
}

/// Compute the *effective origin* of a frame's content document.
///
/// When the frame is unsandboxed, or sandboxed WITH `allow-same-origin`, the
/// content keeps the real origin derived from its document URL. When
/// sandboxed WITHOUT `allow-same-origin`, the content document is assigned a
/// fresh opaque origin (`Origin("null")`) that compares unequal to every
/// other origin — including the parent's and another copy of itself. This is
/// exactly the WHATWG "sandboxed origin browsing context flag" behaviour.
///
/// `doc_url` is the resolved URL the frame loaded (for `srcdoc`, pass the
/// parent document URL — `about:srcdoc` inherits the parent's URL for origin
/// purposes per spec; sandbox then still opaques it when applicable).
pub fn effective_frame_origin(doc_url: &Url, flags: &SandboxFlags) -> Origin {
    if flags.present && !flags.allow_same_origin {
        // Sandboxed without allow-same-origin → fresh opaque origin.
        return Origin("null".to_string());
    }
    Origin::of(doc_url)
}

/// Whether a script running in `accessor_origin` may reach into the frame's
/// DOM (`contentDocument`, same-origin `contentWindow` members beyond
/// `postMessage`). True only when both origins are concrete (non-opaque) AND
/// equal — the WHATWG same-origin check. An opaque origin on either side is
/// never same-origin with anything (an opaque origin only equals *itself* by
/// identity, which two distinct documents never share).
pub fn can_access_frame_dom(accessor_origin: &Origin, frame_origin: &Origin) -> bool {
    if accessor_origin.is_opaque() || frame_origin.is_opaque() {
        return false;
    }
    accessor_origin == frame_origin
}

/// Resolve the value of an iframe's `src` against the embedding document's
/// base URL. Returns the absolute URL string, or `None` for an empty/missing
/// `src` or a value that does not resolve.
pub fn resolve_frame_src(base_url: &str, src: &str) -> Option<String> {
    let trimmed = src.trim();
    if trimmed.is_empty() {
        return None;
    }
    // `about:blank` / `about:srcdoc` resolve to themselves.
    if trimmed == "about:blank" || trimmed == "about:srcdoc" {
        return Some(trimmed.to_string());
    }
    let base = Url::parse(base_url).ok()?;
    base.resolve(trimmed).ok().map(|u| u.to_string())
}

/// Outcome of a `postMessage` targetOrigin check.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PostMessageDelivery {
    /// Deliver: targetOrigin is `*`, or matches the receiver's origin, or is
    /// `/` (meaning "same origin as the SENDER", which matched the receiver).
    Deliver,
    /// Drop silently — the receiver's origin does not match targetOrigin.
    Drop,
}

/// Decide whether a `postMessage(message, targetOrigin)` call should deliver
/// to a receiver whose current origin is `receiver_origin`, where the sender
/// is at `sender_origin`.
///
/// Per the HTML Standard:
///   * `targetOrigin == "*"`  → always deliver.
///   * `targetOrigin == "/"`  → deliver iff receiver origin == sender origin.
///   * otherwise the string is parsed as an origin; deliver iff it equals the
///     receiver's current origin (scheme+host+port exact match). On a parse
///     failure we drop (a malformed targetOrigin can never match).
///
/// An opaque receiver origin never matches a concrete targetOrigin, and a
/// `"/"` (same-origin) target never matches when either side is opaque.
pub fn post_message_delivery(
    target_origin: &str,
    sender_origin: &Origin,
    receiver_origin: &Origin,
) -> PostMessageDelivery {
    match target_origin {
        "*" => PostMessageDelivery::Deliver,
        "/" => {
            // "same origin as the script context that is invoking the method".
            if !sender_origin.is_opaque()
                && !receiver_origin.is_opaque()
                && sender_origin == receiver_origin
            {
                PostMessageDelivery::Deliver
            } else {
                PostMessageDelivery::Drop
            }
        }
        other => {
            // Parse `other` as a serialized origin (scheme://host[:port]).
            match Url::parse(other) {
                Ok(u) => {
                    let want = Origin::of(&u);
                    if !want.is_opaque()
                        && !receiver_origin.is_opaque()
                        && want == *receiver_origin
                    {
                        PostMessageDelivery::Deliver
                    } else {
                        PostMessageDelivery::Drop
                    }
                }
                Err(_) => PostMessageDelivery::Drop,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn origin(s: &str) -> Origin {
        Origin(s.to_string())
    }

    // ---- sandbox parsing -------------------------------------------------

    #[test]
    fn no_sandbox_attr_is_unsandboxed() {
        let f = SandboxFlags::parse(None);
        assert!(!f.present);
        assert!(f.allow_scripts);
        assert!(f.allow_same_origin);
        assert!(f.scripts_enabled());
    }

    #[test]
    fn bare_sandbox_denies_everything() {
        // `<iframe sandbox>` → value is "" → fully restricted.
        let f = SandboxFlags::parse(Some(""));
        assert!(f.present);
        assert!(!f.allow_scripts, "bare sandbox disables scripts");
        assert!(!f.allow_same_origin, "bare sandbox → opaque origin");
        assert!(!f.allow_forms);
        assert!(!f.scripts_enabled());
    }

    #[test]
    fn allow_scripts_only_enables_scripts_not_same_origin() {
        let f = SandboxFlags::parse(Some("allow-scripts"));
        assert!(f.allow_scripts);
        assert!(!f.allow_same_origin, "still opaque origin");
        assert!(f.scripts_enabled());
    }

    #[test]
    fn allow_same_origin_keeps_real_origin_but_no_scripts() {
        let f = SandboxFlags::parse(Some("allow-same-origin"));
        assert!(!f.allow_scripts, "scripts still blocked");
        assert!(f.allow_same_origin);
        assert!(!f.scripts_enabled());
    }

    #[test]
    fn multiple_tokens_case_insensitive_and_unknown_ignored() {
        let f = SandboxFlags::parse(Some("ALLOW-Scripts   allow-forms  bogus-token"));
        assert!(f.allow_scripts);
        assert!(f.allow_forms);
        assert!(!f.allow_same_origin);
        assert!(!f.allow_popups);
    }

    // ---- effective origin / sandbox opaque ------------------------------

    #[test]
    fn sandboxed_without_same_origin_gets_opaque_origin() {
        let url = Url::parse("https://example.com/frame.html").unwrap();
        let f = SandboxFlags::parse(Some("allow-scripts"));
        let o = effective_frame_origin(&url, &f);
        assert!(o.is_opaque(), "no allow-same-origin → opaque");
    }

    #[test]
    fn sandboxed_with_same_origin_keeps_real_origin() {
        let url = Url::parse("https://example.com/frame.html").unwrap();
        let f = SandboxFlags::parse(Some("allow-scripts allow-same-origin"));
        let o = effective_frame_origin(&url, &f);
        assert_eq!(o, origin("https://example.com"));
        assert!(!o.is_opaque());
    }

    #[test]
    fn unsandboxed_frame_uses_real_origin() {
        let url = Url::parse("https://child.example/page").unwrap();
        let f = SandboxFlags::parse(None);
        let o = effective_frame_origin(&url, &f);
        assert_eq!(o, origin("https://child.example"));
    }

    // ---- DOM access (same-origin vs cross-origin) -----------------------

    #[test]
    fn same_origin_frame_dom_accessible() {
        let parent = origin("https://a.test");
        let frame = origin("https://a.test");
        assert!(can_access_frame_dom(&parent, &frame));
    }

    #[test]
    fn cross_origin_frame_dom_blocked() {
        let parent = origin("https://a.test");
        let frame = origin("https://b.test");
        assert!(!can_access_frame_dom(&parent, &frame));
    }

    #[test]
    fn cross_port_is_cross_origin() {
        let parent = origin("https://a.test");
        let frame = origin("https://a.test:8443");
        assert!(!can_access_frame_dom(&parent, &frame));
    }

    #[test]
    fn opaque_frame_dom_never_accessible() {
        let parent = origin("https://a.test");
        let frame = origin("null");
        assert!(!can_access_frame_dom(&parent, &frame));
        // Even parent-against-parent through an opaque accessor is blocked.
        assert!(!can_access_frame_dom(&Origin("null".into()), &parent));
    }

    // ---- src resolution --------------------------------------------------

    #[test]
    fn resolve_relative_src() {
        let abs = resolve_frame_src("https://host.test/dir/page.html", "child.html");
        assert_eq!(abs.as_deref(), Some("https://host.test/dir/child.html"));
    }

    #[test]
    fn resolve_absolute_src() {
        let abs = resolve_frame_src("https://host.test/", "https://other.test/x");
        assert_eq!(abs.as_deref(), Some("https://other.test/x"));
    }

    #[test]
    fn empty_src_is_none() {
        assert_eq!(resolve_frame_src("https://host.test/", "   "), None);
    }

    #[test]
    fn about_blank_resolves_to_itself() {
        assert_eq!(
            resolve_frame_src("https://host.test/", "about:blank").as_deref(),
            Some("about:blank")
        );
    }

    // ---- postMessage targetOrigin ---------------------------------------

    #[test]
    fn post_message_wildcard_always_delivers() {
        let s = origin("https://sender.test");
        let r = origin("https://anything.test");
        assert_eq!(
            post_message_delivery("*", &s, &r),
            PostMessageDelivery::Deliver
        );
        // Even an opaque receiver gets a wildcard message.
        assert_eq!(
            post_message_delivery("*", &s, &Origin("null".into())),
            PostMessageDelivery::Deliver
        );
    }

    #[test]
    fn post_message_exact_origin_match_delivers() {
        let s = origin("https://sender.test");
        let r = origin("https://recv.test");
        assert_eq!(
            post_message_delivery("https://recv.test", &s, &r),
            PostMessageDelivery::Deliver
        );
    }

    #[test]
    fn post_message_origin_mismatch_drops() {
        let s = origin("https://sender.test");
        let r = origin("https://recv.test");
        assert_eq!(
            post_message_delivery("https://evil.test", &s, &r),
            PostMessageDelivery::Drop
        );
    }

    #[test]
    fn post_message_port_mismatch_drops() {
        let s = origin("https://sender.test");
        let r = origin("https://recv.test");
        assert_eq!(
            post_message_delivery("https://recv.test:8443", &s, &r),
            PostMessageDelivery::Drop
        );
    }

    #[test]
    fn post_message_slash_means_same_origin_as_sender() {
        let s = origin("https://same.test");
        let r_same = origin("https://same.test");
        let r_diff = origin("https://other.test");
        assert_eq!(
            post_message_delivery("/", &s, &r_same),
            PostMessageDelivery::Deliver
        );
        assert_eq!(
            post_message_delivery("/", &s, &r_diff),
            PostMessageDelivery::Drop
        );
    }

    #[test]
    fn post_message_malformed_target_drops() {
        let s = origin("https://sender.test");
        let r = origin("https://recv.test");
        assert_eq!(
            post_message_delivery("not a url", &s, &r),
            PostMessageDelivery::Drop
        );
    }

    #[test]
    fn post_message_opaque_receiver_never_matches_concrete_target() {
        let s = origin("https://sender.test");
        let r = Origin("null".into());
        assert_eq!(
            post_message_delivery("https://sender.test", &s, &r),
            PostMessageDelivery::Drop
        );
    }
}
