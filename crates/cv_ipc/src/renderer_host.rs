//! Browser-process renderer host — site-isolation broker.
//!
//! Maintains one renderer entry per **site** (scheme + eTLD+1). The
//! browser process consults this map when navigating: same site →
//! reuse existing renderer; new site → spawn a fresh sandboxed
//! process. Renderers are killed when no tab references them.
//!
//! This slice ships the data structures + the policy decisions
//! (`site_for_url`, `lookup_or_spawn`). The spawn callback is
//! supplied by the caller so this module stays platform-agnostic
//! and unit-testable; the real wiring uses `SandboxedChild`.

use std::collections::HashMap;

const COMMON_TWO_LABEL_SUFFIXES: &[&str] = &[
    "ac.uk", "co.jp", "co.kr", "co.nz", "co.uk", "com.au", "com.br", "com.cn", "com.mx", "com.tr",
    "gov.uk", "net.au", "org.au", "org.uk",
];

/// One sandboxed renderer's broker-side record.
#[derive(Debug, Clone)]
pub struct RendererRecord {
    pub site: String,
    /// Opaque ID the caller maps back to its `SandboxedChild`.
    pub child_id: u32,
    /// Tabs currently routed to this renderer.
    pub refcount: u32,
}

#[derive(Debug, Default)]
pub struct RendererHost {
    by_site: HashMap<String, RendererRecord>,
    next_child_id: u32,
}

impl RendererHost {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn count(&self) -> usize {
        self.by_site.len()
    }

    pub fn lookup(&self, site: &str) -> Option<&RendererRecord> {
        self.by_site.get(site)
    }

    pub fn lookup_for_url(&self, url: &str) -> Option<&RendererRecord> {
        let site = site_for_url(url)?;
        self.lookup(&site)
    }

    /// Get-or-spawn a renderer for `site`. `spawn_fn` is invoked
    /// only when a new process is needed and is expected to start
    /// the sandboxed child process; it returns the broker-side ID
    /// the caller will reference later (e.g. an index into its
    /// `SandboxedChild` vector).
    pub fn lookup_or_spawn<F>(&mut self, site: &str, mut spawn_fn: F) -> u32
    where
        F: FnMut() -> u32,
    {
        if let Some(rec) = self.by_site.get_mut(site) {
            rec.refcount += 1;
            return rec.child_id;
        }
        let child_id = spawn_fn();
        self.next_child_id = self.next_child_id.max(child_id) + 1;
        self.by_site.insert(
            site.to_string(),
            RendererRecord {
                site: site.to_string(),
                child_id,
                refcount: 1,
            },
        );
        child_id
    }

    /// Canonical get-or-spawn entry point. The browser should use the
    /// computed site key derived from the URL, rather than trusting a
    /// caller-supplied site string.
    pub fn lookup_or_spawn_for_url<F>(&mut self, url: &str, spawn_fn: F) -> Option<u32>
    where
        F: FnMut() -> u32,
    {
        let site = site_for_url(url)?;
        Some(self.lookup_or_spawn(&site, spawn_fn))
    }

    /// Drop one tab's reference to `site`. Returns `Some(child_id)`
    /// if the renderer should be torn down (refcount hit zero).
    pub fn release(&mut self, site: &str) -> Option<u32> {
        let rec = self.by_site.get_mut(site)?;
        rec.refcount = rec.refcount.saturating_sub(1);
        if rec.refcount == 0 {
            let id = rec.child_id;
            self.by_site.remove(site);
            return Some(id);
        }
        None
    }

    pub fn release_for_url(&mut self, url: &str) -> Option<u32> {
        let site = site_for_url(url)?;
        self.release(&site)
    }
}

fn registrable_host(host: &str) -> String {
    let host = host.trim_end_matches('.').to_ascii_lowercase();
    let segments: Vec<&str> = host.split('.').filter(|s| !s.is_empty()).collect();
    if segments.len() <= 2 {
        return host;
    }

    let last_two = format!(
        "{}.{}",
        segments[segments.len() - 2],
        segments[segments.len() - 1]
    );
    if COMMON_TWO_LABEL_SUFFIXES.contains(&last_two.as_str()) && segments.len() >= 3 {
        return format!(
            "{}.{}.{}",
            segments[segments.len() - 3],
            segments[segments.len() - 2],
            segments[segments.len() - 1]
        );
    }

    last_two
}

/// Compute the "site" key for a URL. Per HTML's origin-keyed
/// agent-cluster spec, sites are `scheme://eTLD+1`. V1 uses a
/// simplified registrable-domain heuristic with a small allowlist of
/// common two-label public suffixes (so `foo.bar.example.co.uk` →
/// `https://example.co.uk`).
pub fn site_for_url(url: &str) -> Option<String> {
    let (scheme, after_scheme) = url.split_once("://")?;
    let host_end = after_scheme
        .find('/')
        .or_else(|| after_scheme.find('?'))
        .or_else(|| after_scheme.find('#'))
        .unwrap_or(after_scheme.len());
    let host = &after_scheme[..host_end];
    // Strip port.
    let host = host.split(':').next()?;
    if host.is_empty() {
        return None;
    }
    let site_host = registrable_host(host);
    Some(format!("{}://{}", scheme, site_host))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn site_for_url_reduces_to_etld_plus_one() {
        assert_eq!(
            site_for_url("https://news.example.com/path"),
            Some("https://example.com".to_string())
        );
        assert_eq!(
            site_for_url("http://example.com"),
            Some("http://example.com".to_string())
        );
    }

    #[test]
    fn site_for_url_strips_port() {
        assert_eq!(
            site_for_url("https://example.com:8443/foo"),
            Some("https://example.com".to_string())
        );
    }

    #[test]
    fn site_for_url_handles_common_two_label_suffixes() {
        assert_eq!(
            site_for_url("https://www.service.example.co.uk/path"),
            Some("https://example.co.uk".to_string())
        );
        assert_eq!(
            site_for_url("https://a.b.c.portal.com.au"),
            Some("https://portal.com.au".to_string())
        );
    }

    #[test]
    fn canonical_url_lookup_and_release_use_site_key() {
        let mut host = RendererHost::new();
        let id1 = host
            .lookup_or_spawn_for_url("https://news.example.co.uk/article", || 7)
            .unwrap();
        let id2 = host
            .lookup_or_spawn_for_url("https://mail.example.co.uk/inbox", || 9)
            .unwrap();
        assert_eq!(id1, id2);
        assert!(host.lookup_for_url("https://shop.example.co.uk").is_some());
        assert!(host.release_for_url("https://api.example.co.uk").is_none());
        assert_eq!(host.release_for_url("https://cdn.example.co.uk"), Some(7));
    }

    #[test]
    fn lookup_or_spawn_returns_existing_renderer() {
        let mut host = RendererHost::new();
        let mut spawned = 0;
        let id1 = host.lookup_or_spawn("https://example.com", || {
            spawned += 1;
            42
        });
        let id2 = host.lookup_or_spawn("https://example.com", || {
            spawned += 1;
            99
        });
        assert_eq!(id1, id2);
        assert_eq!(spawned, 1, "second lookup must reuse the existing process");
    }

    #[test]
    fn distinct_sites_spawn_distinct_processes() {
        let mut host = RendererHost::new();
        let a = host.lookup_or_spawn("https://example.com", || 1);
        let b = host.lookup_or_spawn("https://other.com", || 2);
        assert_ne!(a, b);
        assert_eq!(host.count(), 2);
    }

    #[test]
    fn release_tears_down_when_refcount_zero() {
        let mut host = RendererHost::new();
        host.lookup_or_spawn("https://example.com", || 1);
        host.lookup_or_spawn("https://example.com", || 99);
        // First release: still ref'd.
        assert!(host.release("https://example.com").is_none());
        // Second release: should report tear-down.
        assert_eq!(host.release("https://example.com"), Some(1));
        assert_eq!(host.count(), 0);
    }
}
