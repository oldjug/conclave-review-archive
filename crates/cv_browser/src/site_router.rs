//! A3 — SiteInstance routing policy (the pure, platform-agnostic core).
//!
//! Top-level **site isolation**: a cross-site top-level navigation is served
//! by a DIFFERENT renderer process than the one that served the previous
//! site. This module owns the *decision* logic — "given the command the UI
//! just sent, which renderer (existing or new) should serve it, and may the
//! previous renderer now be released?" — with NO Windows FFI, so it is unit
//! testable on every platform. The browser process pairs it with
//! [`cv_ipc::renderer_host::RendererHost`] (the site→child_id allocator) and
//! the real process/pipe handles in `main.rs`.
//!
//! ## What counts as a routing decision
//!
//! The UI drives the renderer over a single `cv_ui::ToPage` channel whose
//! `Cmd { epoch, cmd }` carries an already-encoded navigator command STRING
//! (the same `javascript:` / `tb-link-click:` / `tb-key:` / `tb-mouse:` /
//! plain-URL encoding the in-process navigator understands). Most commands —
//! input, JS, same-site link clicks, history (`back://`/`forward://`),
//! `reload://` — are served by whichever renderer currently owns the active
//! tab. Only a **cross-site top-level URL navigation** must move to a
//! different process.
//!
//! [`classify_command`] turns a raw command into a [`RouteClass`]; given the
//! site the active renderer currently serves, [`route_decision`] yields a
//! [`RouteDecision`] the process layer executes (route to the same renderer,
//! or swap to the site renderer for a new site).
//!
//! History navigations (`back://`/`forward://`) are deliberately treated as
//! "active renderer" here: the destination URL lives in the renderer's own
//! per-tab history stack and is not on the command string, so the browser
//! cannot pre-compute its site. A back-nav that crosses a site boundary is a
//! known limitation of this V1 (the destination renders in the current
//! process); a later milestone can have the renderer report its post-nav URL
//! so the browser can re-home it. Documented, not faked.

use cv_ipc::renderer_host::site_for_url;

/// The navigation/site relevance of a single UI command.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RouteClass {
    /// A top-level navigation to an absolute URL whose site we can compute.
    /// The browser may need to spawn/route to that site's renderer.
    TopLevelNavigation { url: String, site: String },
    /// A command that stays on the active renderer: input, JS, same-process
    /// SPA link clicks, history, reload, or a navigation whose destination
    /// site cannot be derived from the command (non-http scheme, relative
    /// link with no absolute href, history hops).
    ActiveRenderer,
    /// Graceful shutdown — tear everything down.
    Shutdown,
}

/// What the process layer should do with a command.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RouteDecision {
    /// Send the command to the renderer already serving the active tab.
    /// No process change.
    SendToActive,
    /// The navigation targets `site`, which the active renderer already
    /// serves (same-site). Send to the active renderer; no swap.
    SameSite { site: String },
    /// The navigation targets a DIFFERENT `site`. The process layer must
    /// look up / spawn the renderer for `site`, make it active, route the
    /// command there, and (once the new site commits) release the
    /// `previous_site` reference so its renderer can be torn down when no
    /// tab still references it.
    CrossSiteSwap {
        site: String,
        url: String,
        previous_site: Option<String>,
    },
    /// Graceful shutdown of all renderers.
    Shutdown,
}

/// Extract the absolute destination URL of a navigation command, if one is
/// derivable from the command string alone. Returns `None` for input/JS
/// commands and for navigations whose destination is not an absolute URL on
/// the command (history hops, relative SPA links).
///
/// Encodings handled (mirrors `cmd_is_navigation` + the navigator decode in
/// `main.rs`):
///   * `tb-link-click:<path>|||<href>` — SPA link click; the href (3rd field)
///     is the destination. Only routed when it is absolute (`scheme://…`); a
///     relative href is resolved against the current page INSIDE the
///     renderer, so the browser cannot site-classify it here → `None`.
///   * `back://` / `forward://` / `reload://` — history/reload; destination is
///     renderer-owned → `None` (ActiveRenderer).
///   * `tb-…:` / `javascript:` — input/JS → `None`.
///   * anything else containing `://` — a plain absolute-URL navigation
///     (URL bar / programmatic). A trailing `|||shown` display segment, if
///     present, is stripped.
#[must_use]
pub fn nav_target_url(cmd: &str) -> Option<String> {
    if cmd == "back://" || cmd == "forward://" || cmd == "reload://" {
        return None;
    }
    if let Some(rest) = cmd.strip_prefix("tb-link-click:") {
        // Format: "<path>|||<href>". The href is the destination.
        let href = rest.split_once("|||").map(|(_p, h)| h).unwrap_or("");
        if href.contains("://") {
            return Some(strip_display_segment(href).to_string());
        }
        return None;
    }
    // Other tb-* commands and javascript: are input/JS, never navigations.
    if cmd.starts_with("tb-") || cmd.starts_with("javascript:") {
        return None;
    }
    // A plain navigation. The UI may append a "|||shown" display segment to
    // some encodings; the real URL is the first field. Accept only when it
    // looks like an absolute URL we can site-classify.
    let url = strip_display_segment(cmd);
    if url.contains("://") {
        Some(url.to_string())
    } else {
        None
    }
}

fn strip_display_segment(s: &str) -> &str {
    s.split("|||").next().unwrap_or(s)
}

/// Classify a raw UI command into its routing-relevant shape.
#[must_use]
pub fn classify_command(cmd: &str) -> RouteClass {
    if cmd == "__shutdown__" {
        return RouteClass::Shutdown;
    }
    match nav_target_url(cmd) {
        Some(url) => match site_for_url(&url) {
            Some(site) => RouteClass::TopLevelNavigation { url, site },
            // An absolute non-http URL (e.g. `about:`, `data:`): no site
            // key, so it stays on the active renderer.
            None => RouteClass::ActiveRenderer,
        },
        None => RouteClass::ActiveRenderer,
    }
}

/// Decide how to route `cmd` given the site the active renderer currently
/// serves (`active_site`, `None` before the first navigation commits).
#[must_use]
pub fn route_decision(cmd: &str, active_site: Option<&str>) -> RouteDecision {
    match classify_command(cmd) {
        RouteClass::Shutdown => RouteDecision::Shutdown,
        RouteClass::ActiveRenderer => RouteDecision::SendToActive,
        RouteClass::TopLevelNavigation { url, site } => {
            if active_site == Some(site.as_str()) {
                RouteDecision::SameSite { site }
            } else {
                RouteDecision::CrossSiteSwap {
                    site,
                    url,
                    previous_site: active_site.map(str::to_string),
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plain_absolute_url_is_top_level_navigation() {
        match classify_command("https://news.example.com/story") {
            RouteClass::TopLevelNavigation { url, site } => {
                assert_eq!(url, "https://news.example.com/story");
                assert_eq!(site, "https://example.com");
            }
            other => panic!("expected top-level nav, got {other:?}"),
        }
    }

    #[test]
    fn history_and_reload_stay_on_active_renderer() {
        for c in ["back://", "forward://", "reload://"] {
            assert_eq!(classify_command(c), RouteClass::ActiveRenderer, "{c}");
            assert_eq!(nav_target_url(c), None, "{c}");
        }
    }

    #[test]
    fn input_and_js_commands_stay_on_active_renderer() {
        for c in [
            "tb-key:13",
            "tb-mouse:100,200",
            "tb-element:0/3/2",
            "javascript:7|||doThing()",
            "tb-typed:hello",
        ] {
            assert_eq!(classify_command(c), RouteClass::ActiveRenderer, "{c}");
            assert_eq!(nav_target_url(c), None, "{c}");
        }
    }

    #[test]
    fn spa_link_click_with_absolute_href_routes_by_href_site() {
        let cmd = "tb-link-click:0/2/1|||https://other.example.org/page";
        match classify_command(cmd) {
            RouteClass::TopLevelNavigation { url, site } => {
                assert_eq!(url, "https://other.example.org/page");
                assert_eq!(site, "https://example.org");
            }
            other => panic!("expected top-level nav, got {other:?}"),
        }
    }

    #[test]
    fn spa_link_click_with_relative_href_stays_active() {
        // A relative href is resolved inside the renderer; the browser cannot
        // site-classify it, so it must NOT trigger a process swap.
        let cmd = "tb-link-click:0/2/1|||/relative/path";
        assert_eq!(classify_command(cmd), RouteClass::ActiveRenderer);
        assert_eq!(nav_target_url(cmd), None);
    }

    #[test]
    fn non_http_absolute_scheme_has_no_site_and_stays_active() {
        // about:/data: have a "://"? about: does not; data: does not. Use a
        // scheme with "://" but no host so site_for_url returns None.
        assert_eq!(classify_command("about:blank"), RouteClass::ActiveRenderer);
        assert_eq!(classify_command("foo:///x"), RouteClass::ActiveRenderer);
    }

    #[test]
    fn same_site_navigation_does_not_swap() {
        let d = route_decision(
            "https://mail.example.com/inbox",
            Some("https://example.com"),
        );
        match d {
            RouteDecision::SameSite { site } => assert_eq!(site, "https://example.com"),
            other => panic!("expected same-site, got {other:?}"),
        }
    }

    #[test]
    fn cross_site_navigation_swaps_and_carries_previous_site() {
        let d = route_decision("https://other.com/x", Some("https://example.com"));
        match d {
            RouteDecision::CrossSiteSwap {
                site,
                url,
                previous_site,
            } => {
                assert_eq!(site, "https://other.com");
                assert_eq!(url, "https://other.com/x");
                assert_eq!(previous_site.as_deref(), Some("https://example.com"));
            }
            other => panic!("expected cross-site swap, got {other:?}"),
        }
    }

    #[test]
    fn first_navigation_with_no_active_site_swaps_with_no_previous() {
        let d = route_decision("https://example.com/", None);
        match d {
            RouteDecision::CrossSiteSwap {
                site,
                previous_site,
                ..
            } => {
                assert_eq!(site, "https://example.com");
                assert_eq!(previous_site, None);
            }
            other => panic!("expected cross-site swap, got {other:?}"),
        }
    }

    #[test]
    fn input_command_routes_to_active_regardless_of_site() {
        assert_eq!(
            route_decision("tb-key:13", Some("https://example.com")),
            RouteDecision::SendToActive
        );
    }

    #[test]
    fn subdomain_navigation_is_same_site() {
        // Different host, same registrable site → no swap.
        let d = route_decision("https://cdn.example.com/a", Some("https://example.com"));
        assert!(matches!(d, RouteDecision::SameSite { .. }));
    }

    #[test]
    fn display_segment_is_stripped_from_plain_url() {
        assert_eq!(
            nav_target_url("https://example.com/p|||Example Page"),
            Some("https://example.com/p".to_string())
        );
    }

    #[test]
    fn shutdown_sentinel_is_classified() {
        assert_eq!(classify_command("__shutdown__"), RouteClass::Shutdown);
        assert_eq!(
            route_decision("__shutdown__", Some("https://example.com")),
            RouteDecision::Shutdown
        );
    }
}
