//! Isolated worlds — content scripts run in a separate JS global
//! that shares the DOM but not the page's JS globals.
//!
//! Each content script gets its own `Realm` (own `globalThis`, own
//! built-ins), but the `document` / `window.location` they see is
//! the same DOM the page sees. Mutations to the DOM are visible
//! across realms; mutations to a global on one side are invisible to
//! the other.

use std::collections::HashMap;

/// Identifier for a JS realm.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RealmId(u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RealmKind {
    /// The page's main world (real `window`).
    Main,
    /// An extension's isolated content-script world.
    Isolated { extension_id: u32 },
}

/// One JS realm. Owns globals; DOM references resolve through the
/// shared document handle that `RealmRegistry` injects on creation.
#[derive(Debug)]
pub struct Realm {
    pub id: RealmId,
    pub kind: RealmKind,
    pub origin: String,
    /// Per-realm `globalThis` properties. Stored as opaque JSON-shaped
    /// strings so the interp can hand back its own Value types when
    /// it reads them.
    globals: HashMap<String, String>,
}

impl Realm {
    pub fn set_global(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.globals.insert(key.into(), value.into());
    }
    pub fn get_global(&self, key: &str) -> Option<&str> {
        self.globals.get(key).map(String::as_str)
    }
    pub fn global_count(&self) -> usize {
        self.globals.len()
    }
}

#[derive(Debug, Default)]
pub struct RealmRegistry {
    realms: HashMap<RealmId, Realm>,
    next: u32,
    /// Per-extension active realms by tab/frame id.
    by_extension: HashMap<u32, Vec<RealmId>>,
}

impl RealmRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn create_main(&mut self, origin: impl Into<String>) -> RealmId {
        self.next += 1;
        let id = RealmId(self.next);
        self.realms.insert(
            id,
            Realm {
                id,
                kind: RealmKind::Main,
                origin: origin.into(),
                globals: HashMap::new(),
            },
        );
        id
    }

    pub fn create_isolated(&mut self, extension_id: u32, origin: impl Into<String>) -> RealmId {
        self.next += 1;
        let id = RealmId(self.next);
        self.realms.insert(
            id,
            Realm {
                id,
                kind: RealmKind::Isolated { extension_id },
                origin: origin.into(),
                globals: HashMap::new(),
            },
        );
        self.by_extension.entry(extension_id).or_default().push(id);
        id
    }

    pub fn get(&self, id: RealmId) -> Option<&Realm> {
        self.realms.get(&id)
    }
    pub fn get_mut(&mut self, id: RealmId) -> Option<&mut Realm> {
        self.realms.get_mut(&id)
    }

    pub fn realms_for_extension(&self, extension_id: u32) -> Vec<RealmId> {
        self.by_extension
            .get(&extension_id)
            .cloned()
            .unwrap_or_default()
    }

    /// Tear down a realm.
    pub fn destroy(&mut self, id: RealmId) {
        if let Some(r) = self.realms.remove(&id) {
            if let RealmKind::Isolated { extension_id } = r.kind {
                if let Some(list) = self.by_extension.get_mut(&extension_id) {
                    list.retain(|&x| x != id);
                }
            }
        }
    }

    pub fn len(&self) -> usize {
        self.realms.len()
    }
    pub fn is_empty(&self) -> bool {
        self.realms.is_empty()
    }
}

/// content_scripts run_at lifecycle. The renderer ticks the realm
/// registry at each phase, injecting content scripts whose `run_at`
/// matches.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DocumentLifecycle {
    Loading,
    Interactive, // document_end
    Complete,    // document_idle
}

#[derive(Debug, Clone)]
pub struct PendingInjection {
    pub extension_id: u32,
    pub script_url: String,
    pub run_at: DocumentLifecycle,
    pub matches: Vec<String>,
}

#[derive(Debug, Default)]
pub struct InjectionQueue {
    pending: Vec<PendingInjection>,
    injected: Vec<(u32, String, RealmId)>,
}

impl InjectionQueue {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn enqueue(&mut self, inj: PendingInjection) {
        self.pending.push(inj);
    }

    /// Inject all pending content scripts whose match list contains
    /// `frame_url` and whose run_at is `phase`. Returns one fresh
    /// isolated realm per injection.
    pub fn fire(
        &mut self,
        registry: &mut RealmRegistry,
        frame_url: &str,
        phase: DocumentLifecycle,
    ) -> Vec<RealmId> {
        let mut spawned = Vec::new();
        let mut keep = Vec::new();
        for inj in std::mem::take(&mut self.pending) {
            if inj.run_at == phase && match_url(&inj.matches, frame_url) {
                let realm = registry.create_isolated(inj.extension_id, frame_url.to_string());
                self.injected
                    .push((inj.extension_id, inj.script_url.clone(), realm));
                spawned.push(realm);
            } else {
                keep.push(inj);
            }
        }
        self.pending = keep;
        spawned
    }

    pub fn injected_count(&self) -> usize {
        self.injected.len()
    }
    pub fn pending_count(&self) -> usize {
        self.pending.len()
    }
}

fn match_url(patterns: &[String], url: &str) -> bool {
    for p in patterns {
        if p == "<all_urls>" || p == "*://*/*" {
            return true;
        }
        if matches_glob(p, url) {
            return true;
        }
    }
    false
}

/// Tiny `<all_urls>`-flavored glob matcher: supports `*` in scheme,
/// host, path. Used to match content-script `matches` patterns.
pub fn matches_glob(pattern: &str, url: &str) -> bool {
    let pat: Vec<&str> = pattern.split("*").collect();
    if pat.len() == 1 {
        return pattern == url;
    }
    if !url.starts_with(pat[0]) {
        return false;
    }
    let mut cursor = pat[0].len();
    for piece in &pat[1..pat.len() - 1] {
        match url[cursor..].find(piece) {
            Some(off) => cursor += off + piece.len(),
            None => return false,
        }
    }
    url[cursor..].ends_with(pat.last().unwrap())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn main_and_isolated_realms_have_separate_globals() {
        let mut reg = RealmRegistry::new();
        let main = reg.create_main("https://example.com");
        let iso = reg.create_isolated(7, "https://example.com");
        reg.get_mut(main).unwrap().set_global("foo", "main_val");
        reg.get_mut(iso).unwrap().set_global("foo", "iso_val");
        assert_eq!(reg.get(main).unwrap().get_global("foo"), Some("main_val"));
        assert_eq!(reg.get(iso).unwrap().get_global("foo"), Some("iso_val"));
    }

    #[test]
    fn realm_count_grows_per_extension() {
        let mut reg = RealmRegistry::new();
        reg.create_main("https://a.com");
        reg.create_isolated(1, "https://a.com");
        reg.create_isolated(1, "https://a.com");
        reg.create_isolated(2, "https://a.com");
        assert_eq!(reg.realms_for_extension(1).len(), 2);
        assert_eq!(reg.realms_for_extension(2).len(), 1);
        assert_eq!(reg.len(), 4);
    }

    #[test]
    fn destroy_removes_from_extension_list() {
        let mut reg = RealmRegistry::new();
        let r = reg.create_isolated(1, "x");
        reg.destroy(r);
        assert!(reg.realms_for_extension(1).is_empty());
        assert_eq!(reg.len(), 0);
    }

    #[test]
    fn injection_queue_fires_on_matching_phase() {
        let mut reg = RealmRegistry::new();
        let mut q = InjectionQueue::new();
        q.enqueue(PendingInjection {
            extension_id: 5,
            script_url: "content.js".into(),
            run_at: DocumentLifecycle::Interactive,
            matches: vec!["https://example.com/*".into()],
        });
        let spawned = q.fire(
            &mut reg,
            "https://example.com/page",
            DocumentLifecycle::Interactive,
        );
        assert_eq!(spawned.len(), 1);
        assert_eq!(q.injected_count(), 1);
        assert_eq!(q.pending_count(), 0);
    }

    #[test]
    fn injection_doesnt_fire_on_wrong_phase() {
        let mut reg = RealmRegistry::new();
        let mut q = InjectionQueue::new();
        q.enqueue(PendingInjection {
            extension_id: 5,
            script_url: "s.js".into(),
            run_at: DocumentLifecycle::Complete,
            matches: vec!["<all_urls>".into()],
        });
        let s = q.fire(&mut reg, "https://x.com/", DocumentLifecycle::Loading);
        assert!(s.is_empty());
        assert_eq!(q.pending_count(), 1);
    }

    #[test]
    fn matches_glob_matches_subdomain_wildcard() {
        assert!(matches_glob(
            "https://*.example.com/*",
            "https://www.example.com/x"
        ));
        assert!(matches_glob(
            "https://example.com/*",
            "https://example.com/foo"
        ));
        assert!(!matches_glob(
            "https://example.com/*",
            "http://example.com/foo"
        ));
    }

    #[test]
    fn all_urls_matches_anything() {
        assert!(match_url(&["<all_urls>".into()], "https://x.com/y"));
        assert!(match_url(&["<all_urls>".into()], "http://a.b/c"));
    }
}
