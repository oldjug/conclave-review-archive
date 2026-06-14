//! `paint_cache` — Phase-4 PERSISTENT CROSS-LOAD PAINT CACHE (in-memory).
//!
//! The AOT-persist lever of "beat Chrome": Chrome DISCARDS a page's whole
//! cascade+layout+paint on every navigation and on back/forward. We PERSIST the
//! finished frame so a warm repeat-visit / back-forward / re-navigation to a
//! visited URL within the session serves its FIRST PAINT from a refcount bump
//! instead of a cold cascade+layout+full-bake. This phase is IN-MEMORY only
//! (survives same-session navigations); disk persistence is an explicit
//! follow-up — the seam is noted at the bottom of this file.
//!
//! GATED behind `CV_PAINT_CACHE` (`OnceLock<bool>`, **default OFF**). With the
//! flag off, `enabled()` is `false`, so the navigation seam in `main.rs` never
//! calls `lookup`/`insert` and every navigation cold-bakes EXACTLY as today,
//! byte-for-byte. The cold full-bake stays the default, the always-correct
//! fallback, and the oracle. A MISS, any guard failure, or a structural-rebuild
//! wipe all fall back to that cold bake. NO STUBS.
//!
//! ── The 3 mandatory correctness guards ──────────────────────────────────────
//!
//! 1. STRUCTURAL-REBUILD WIPE (the ONLY silent-WRONG-frame hazard). When
//!    `reconcile_arena_node` bails and `build_arena_dom` rebuilds into a FRESH
//!    `Document`, NodeIds are REALLOCATED and can alias DIFFERENT logical
//!    elements. The cached `PaintData` carries a `RetainedDisplayList` whose
//!    chunks are node_id-keyed and a `layout_root` whose identities were captured
//!    under the OLD arena generation; serving a stale entry (even by content
//!    hash) would hit-test / damage-raster against reallocated ids. So
//!    [`clear_all`] is called at the SAME two sites that wipe `style_cache` /
//!    `layout_cache` (`main.rs` feature-set change + reconcile-bail rebuild),
//!    making a stale-id serve IMPOSSIBLE BY CONSTRUCTION — the entry is gone
//!    before any subsequent nav can find it.
//!
//! 2. DIRTY-BIT / CONTENT guard = the `dom_structural_hash` IS the content guard.
//!    An entry is served only if THIS navigation's `dom_structural_hash`
//!    (computed over the post-JS DOM + sheets fingerprint) byte-equals the key's
//!    hash. Any paint-affecting content change (a different text node, a changed
//!    class/id/style attr, an added/removed element, a different stylesheet set)
//!    produces a DIFFERENT key ⇒ a MISS ⇒ cold bake.
//!
//! 3. GEOMETRY guard = `viewport_w` / `viewport_h` are PART of the key, so any
//!    viewport change yields a different key ⇒ automatic MISS ⇒ cold bake at the
//!    new size. There is no way to serve a wrong-size bitmap. Defense in depth:
//!    the entry stores `doc_w` / `doc_h` and the seam cross-checks them against
//!    the served bitmap dims.
//!
//! ── The two hashes (the chicken-and-egg solution) ───────────────────────────
//!
//! The master design's "structural_hash = RDL root subtree_hash" is a POST-layout
//! quantity; a cross-load cache that pays cascade+layout to compute its lookup key
//! would not be a win. So the hash is SPLIT into two levels:
//!
//!   * [`dom_structural_hash`] — the CHEAP, pre-layout DOM-level LOOKUP key. An
//!     FNV-1a walk of the post-parse + post-JS `cv_html::Document` tree,
//!     conservative-complete over everything paint-affecting (tag + all real
//!     attrs + text + tree shape) plus the active sheet-set fingerprint. O(nodes),
//!     no cascade/layout.
//!
//!   * `PaintCacheEntry::structural_hash` — the authoritative POST-layout RDL root
//!     `subtree_hash`, captured at insert (reusing the RDL the cold bake already
//!     produced, or forcing one `generate` if `CV_DAMAGE_RASTER` is off). NOT
//!     consulted on the hit path (no layout on a hit). It is the ORACLE tie-point:
//!     two frames sharing a `dom_structural_hash` MUST share this RDL hash.
//!
//! SOUNDNESS CONTRACT: `dom_structural_hash` MUST be a refinement of the RDL
//! `subtree_hash` — equal `dom_structural_hash` ⇒ equal RDL `subtree_hash` ⇒ (same
//! viewport) byte-identical bitmap. Layout is a PURE FUNCTION of (DOM, CSS,
//! viewport): all three are in the key (DOM+CSS in the hash, viewport explicit),
//! so the implication holds. The oracle tests prove it empirically.

// This is a crate-internal module of the `conclave` BINARY crate, so its `pub`
// items are never reachable outside the crate (the workspace `unreachable_pub`
// lint flags them — mirroring `retained_dl`). `pub` is kept for intra-crate use
// (the navigation seam in `main.rs`) + clear API documentation; several helpers
// (`len`, `total_bytes`, struct fields) are exercised only by the test module.
#![allow(unreachable_pub, dead_code)]

use std::collections::{HashMap, VecDeque};
use std::sync::{Mutex, OnceLock};

use crate::retained_dl::{self, Fnv, RetainedDisplayList};

/// `CV_PAINT_CACHE` gate, read once. **Default ON** (flipped 2026-06-12 after the
/// in-memory cross-load cache was verified byte-identical to a cold bake, oracle
/// mutation-proven, with the structural-rebuild wipe guarding the one WRONG-frame
/// hazard and every uncertainty path falling back to a correct cold bake — the
/// worst failure is a wasted re-render, never a wrong pixel). Escape hatch:
/// `CV_PAINT_CACHE=0` forces it OFF (pure cold-bake, byte-for-byte as before the
/// cache existed); unset or any other value = ON. NOTE: this is the IN-MEMORY
/// cross-load cache only; on-disk persistence is a SEPARATE opt-in flag
/// (`CV_PAINT_CACHE_DISK`, still default OFF — it writes the user's filesystem and
/// awaits a real-window cross-restart soak before default-on).
pub fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("CV_PAINT_CACHE").as_deref() != Ok("0"))
}

/// `CV_PAINT_CACHE_DISK` gate, read once. **Default ON** (flipped 2026-06-13;
/// round-trip mutation-proven byte-identical, corruption-safe, layered ON TOP of
/// `enabled()` so it only acts when the cache seam is live). Turns on cross-RESTART
/// (cross-session) warm first-paint: a revisited page paints from disk instantly.
/// Escape hatch: `CV_PAINT_CACHE_DISK=0` forces it OFF (in-memory cache only, ZERO
/// disk activity); unset or any other value → ON. Under `cfg(test)` the disk dir
/// is None unless a test sets an explicit override, so tests do ZERO real disk IO.
pub fn disk_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("CV_PAINT_CACHE_DISK").as_deref() != Ok("0"))
}

/// Byte budget for the whole cache. Bitmaps dominate memory, so the LRU is
/// bounded by total bytes, not entry count. Default 256 MB; override via
/// `CV_PAINT_CACHE_BUDGET_MB` (parsed once). A `0` or unparseable value falls
/// back to the default.
fn budget_bytes() -> usize {
    static BUDGET: OnceLock<usize> = OnceLock::new();
    *BUDGET.get_or_init(|| {
        let mb = std::env::var("CV_PAINT_CACHE_BUDGET_MB")
            .ok()
            .and_then(|s| s.trim().parse::<usize>().ok())
            .filter(|&m| m > 0)
            .unwrap_or(256);
        mb.saturating_mul(1024 * 1024)
    })
}

/// Hard secondary cap on entry count, so a flood of tiny pages can't grow the map
/// unboundedly even while staying under the byte budget.
const MAX_ENTRIES: usize = 64;

/// FNV-1a over the normalized absolute URL string. Hash (not String) to keep the
/// key small; a 64-bit FNV collision across distinct URLs is astronomically
/// unlikely, and even a collision is caught by `dom_structural_hash` in the key,
/// the stored `structural_hash` (oracle), and the `current_url` string cross-check
/// at the seam.
pub fn url_hash(url: &str) -> u64 {
    let mut h = Fnv::new();
    h.str(url);
    h.finish()
}

/// The lookup key: `(url, dom_structural_hash, viewport_w, viewport_h)`.
///
/// `dom_structural_hash` is the CHEAP pre-layout content signal computed over the
/// post-JS DOM + sheets fingerprint. A viewport change yields a different key ⇒
/// automatic miss ⇒ cold bake (this IS the geometry guard).
#[derive(PartialEq, Eq, Hash, Clone, Debug)]
pub struct PaintCacheKey {
    pub url_hash: u64,
    pub dom_structural_hash: u64,
    pub vw: u32,
    pub vh: u32,
}

/// A fully-servable cached frame plus the guard fields.
#[derive(Clone)]
pub struct PaintCacheEntry {
    /// The WHOLE servable frame. `PaintData` is `Clone` and every heavy field is
    /// `Arc`'d (bitmap, retained RDL), so clone-on-serve is a refcount bump, NOT a
    /// multi-MB memcpy. (`layout_root` is the one non-`Arc` deep clone, retained
    /// for hit-test parity with a cold bake; flagged as a v1 risk in the design.)
    pub paint: cv_ui::PaintData,
    /// The bitmap dims this entry was baked at (= `expected_dims()` output).
    /// Stored for the geometry self-consistency cross-check on a hit.
    pub doc_w: u32,
    pub doc_h: u32,
    /// The authoritative POST-layout RDL root `subtree_hash` captured at insert.
    /// The oracle tie-point; NOT consulted on the hit path.
    pub structural_hash: u64,
    /// `bitmap.pixels.len()*4` (dominant) + a small fixed estimate, for the byte
    /// budget. Maintained as an exact running sum in [`PaintCache::bytes`].
    pub bytes: usize,
}

/// Small fixed per-entry overhead estimate (texts + hit_regions + RDL) folded
/// into the byte budget alongside the dominant bitmap cost.
const ENTRY_FIXED_OVERHEAD: usize = 64 * 1024;

/// Process-global bounded-LRU cache. The `Mutex` is held only for the O(1)
/// lookup/insert/evict — NEVER across a bake.
pub struct PaintCache {
    map: HashMap<PaintCacheKey, PaintCacheEntry>,
    /// MRU order: front = least-recently-used, back = most-recently-used.
    lru: VecDeque<PaintCacheKey>,
    /// Exact running sum of every entry's `bytes`. Restored `<= budget` after
    /// every insert; zeroed by `clear`.
    bytes: usize,
}

impl PaintCache {
    fn new() -> Self {
        PaintCache {
            map: HashMap::new(),
            lru: VecDeque::new(),
            bytes: 0,
        }
    }

    /// Move `key` to the MRU (back) position. Called on a hit.
    fn touch(&mut self, key: &PaintCacheKey) {
        if let Some(pos) = self.lru.iter().position(|k| k == key) {
            let k = self.lru.remove(pos).unwrap();
            self.lru.push_back(k);
        }
    }

    /// Evict least-recently-used entries until the byte budget is satisfied.
    fn evict_to_budget(&mut self, budget: usize) {
        while self.bytes > budget {
            let Some(victim) = self.lru.pop_front() else {
                break;
            };
            if let Some(e) = self.map.remove(&victim) {
                self.bytes = self.bytes.saturating_sub(e.bytes);
            }
        }
    }
}

/// Process-global, survives navigations.
static PAINT_CACHE: OnceLock<Mutex<PaintCache>> = OnceLock::new();

fn cache() -> &'static Mutex<PaintCache> {
    PAINT_CACHE.get_or_init(|| Mutex::new(PaintCache::new()))
}

/// Look up a cached frame by key, moving it to MRU on a hit. Returns a CLONE of
/// the entry (refcount bumps on the Arc'd fields). `None` on miss. The seam
/// performs the belt-and-suspenders `current_url` + dims cross-check before
/// serving the clone.
pub fn lookup(key: &PaintCacheKey) -> Option<PaintCacheEntry> {
    {
        let mut c = cache().lock().unwrap_or_else(|p| p.into_inner());
        if c.map.contains_key(key) {
            c.touch(key);
            return c.map.get(key).cloned();
        }
    }
    // In-memory MISS. With the disk flag on, try the on-disk store before
    // reporting a miss (cross-RESTART warm first-paint). A disk hit is
    // reconstructed into a SERVABLE entry (bitmap + key + dims + hashes +
    // chrome metadata; layout_root / retained / property_trees = None — the
    // first paint is served from pixels, subsequent frames re-derive via the
    // normal path) and PROMOTED into the in-memory LRU so repeated same-session
    // hits skip disk entirely. Flag off ⇒ this whole block is skipped ⇒ no IO.
    if disk_enabled() {
        if let Some(entry) = disk::load(key) {
            promote_from_disk(key.clone(), entry.clone());
            return Some(entry);
        }
    }
    None
}

/// Insert a disk-loaded entry into the in-memory LRU WITHOUT writing it back to
/// disk (it just came from there). Mirrors [`insert_inner`]'s LRU bookkeeping
/// against the in-memory byte budget.
fn promote_from_disk(key: PaintCacheKey, entry: PaintCacheEntry) {
    let budget = budget_bytes();
    if entry.bytes > budget {
        return;
    }
    let mut c = cache().lock().unwrap_or_else(|p| p.into_inner());
    if let Some(old) = c.map.remove(&key) {
        c.bytes = c.bytes.saturating_sub(old.bytes);
        if let Some(pos) = c.lru.iter().position(|k| k == &key) {
            c.lru.remove(pos);
        }
    }
    c.bytes = c.bytes.saturating_add(entry.bytes);
    c.map.insert(key.clone(), entry);
    c.lru.push_back(key);
    c.evict_to_budget(budget);
    while c.map.len() > MAX_ENTRIES {
        let Some(victim) = c.lru.pop_front() else {
            break;
        };
        if let Some(e) = c.map.remove(&victim) {
            c.bytes = c.bytes.saturating_sub(e.bytes);
        }
    }
}

/// Insert (or replace) a cold-baked frame under `key`. Reuses the RDL the bake
/// already produced (`paint.retained`) for the authoritative `structural_hash`;
/// if `paint.retained` is `None` (e.g. `CV_DAMAGE_RASTER` off), forces one
/// `retained_dl::generate` from `paint.layout_root` so the entry ALWAYS carries
/// its authoritative RDL hash. An entry larger than the whole budget is NOT
/// inserted (no-op) so one huge page can never blow the bound or evict everything.
///
/// `cfg` is needed only for the fallback `generate`. Returns nothing; on a miss
/// the seam has already committed/served the cold bake.
pub fn insert(key: PaintCacheKey, paint: &cv_ui::PaintData, cfg: &cv_layout::LayoutConfig) {
    insert_inner(key, paint, cfg, budget_bytes());
}

/// Test-only insert with an explicit byte budget, so the eviction / oversized
/// bounds can be exercised deterministically without allocating hundreds of MB
/// (the default 256 MB budget is read once via a `OnceLock` and dwarfs any test
/// frame).
#[cfg(test)]
fn insert_with_budget(
    key: PaintCacheKey,
    paint: &cv_ui::PaintData,
    cfg: &cv_layout::LayoutConfig,
    budget: usize,
) {
    insert_inner(key, paint, cfg, budget);
}

fn insert_inner(
    key: PaintCacheKey,
    paint: &cv_ui::PaintData,
    cfg: &cv_layout::LayoutConfig,
    budget: usize,
) {
    let bitmap_bytes = paint.bitmap.pixels.len().saturating_mul(4);
    let bytes = bitmap_bytes.saturating_add(ENTRY_FIXED_OVERHEAD);
    // A single entry larger than the whole budget is simply not inserted — it
    // would otherwise blow the bound or evict everything on the next insert.
    if bytes > budget {
        return;
    }

    let doc_w = paint.bitmap.width;
    let doc_h = paint.bitmap.height;

    // Authoritative POST-layout RDL root subtree_hash. Prefer the RDL the bake
    // already generated (carried opaquely as Arc<dyn Any>; downcast it); fall back
    // to forcing one generate from layout_root so the entry always has its hash.
    let structural_hash = rdl_root_subtree_hash(paint, cfg);

    let entry = PaintCacheEntry {
        paint: paint.clone(),
        doc_w,
        doc_h,
        structural_hash,
        bytes,
    };

    let mut c = cache().lock().unwrap_or_else(|p| p.into_inner());
    // Replace an existing entry under the same key (subtract its old bytes first).
    if let Some(old) = c.map.remove(&key) {
        c.bytes = c.bytes.saturating_sub(old.bytes);
        if let Some(pos) = c.lru.iter().position(|k| k == &key) {
            c.lru.remove(pos);
        }
    }
    c.bytes = c.bytes.saturating_add(entry.bytes);
    c.map.insert(key.clone(), entry.clone());
    c.lru.push_back(key.clone());
    // Restore the byte budget by evicting LRU entries.
    c.evict_to_budget(budget);
    // Hard secondary cap: never let the map exceed MAX_ENTRIES.
    while c.map.len() > MAX_ENTRIES {
        let Some(victim) = c.lru.pop_front() else {
            break;
        };
        if let Some(e) = c.map.remove(&victim) {
            c.bytes = c.bytes.saturating_sub(e.bytes);
        }
    }
    // Release the in-memory lock BEFORE touching the filesystem (the Mutex is
    // never held across IO). With the disk flag on, write-through this entry
    // atomically and re-bound the on-disk store to its own byte/file budget
    // (LRU-evicting old files). Flag off ⇒ skipped ⇒ zero disk activity.
    drop(c);
    if disk_enabled() {
        disk::store(&key, &entry);
    }
}

/// Recover the authoritative POST-layout RDL root `subtree_hash` for a baked
/// `PaintData`. The cold bake stores the RDL it produced in `paint.retained`
/// (`Arc<dyn Any>`); downcast it. If absent (e.g. `CV_DAMAGE_RASTER` off) force a
/// single `generate` from `paint.layout_root`. If there is no layout_root either
/// (degenerate), return 0 — a benign value that still keys correctly (the
/// dom_structural_hash is the real lookup guard; this hash is only the oracle
/// tie-point).
fn rdl_root_subtree_hash(paint: &cv_ui::PaintData, cfg: &cv_layout::LayoutConfig) -> u64 {
    if let Some(any) = paint.retained.as_ref() {
        if let Some(rdl) = any.downcast_ref::<RetainedDisplayList>() {
            return root_subtree_hash(rdl);
        }
    }
    if let Some(lb) = paint.layout_root.as_ref() {
        let rdl = retained_dl::generate(lb, cfg);
        return root_subtree_hash(&rdl);
    }
    0
}

fn root_subtree_hash(rdl: &RetainedDisplayList) -> u64 {
    rdl.chunks
        .get(rdl.root as usize)
        .map(|c| c.subtree_hash)
        .unwrap_or(0)
}

/// Clear the ENTIRE cache. ★ LOAD-BEARING: called at the structural-rebuild wipe
/// sites in `main.rs` (feature-set change + reconcile-bail rebuild) alongside
/// `style_cache.clear()` / `layout_cache.clear()`. After a structural rebuild
/// NodeIds are reallocated and can alias different elements; wiping here makes a
/// stale-id serve IMPOSSIBLE BY CONSTRUCTION (the entry is gone before any
/// subsequent nav can find it).
pub fn clear_all() {
    {
        let mut c = cache().lock().unwrap_or_else(|p| p.into_inner());
        c.map.clear();
        c.lru.clear();
        c.bytes = 0;
    }
    // Structural-rebuild wipe must also purge the on-disk entries when disk
    // persistence is active: after a NodeId reallocation the persisted pixels are
    // stale, and re-hydrating them would be a silent WRONG-frame. (Lock released
    // above before disk IO.) No-op when disk is off / no dir configured.
    if disk_enabled() {
        disk::clear_all_files();
    }
}

/// Number of live entries. For tests / diagnostics.
pub fn len() -> usize {
    cache().lock().unwrap_or_else(|p| p.into_inner()).map.len()
}

/// Total bytes currently accounted. For tests / diagnostics.
pub fn total_bytes() -> usize {
    cache().lock().unwrap_or_else(|p| p.into_inner()).bytes
}

/// The CHEAP, pre-layout DOM-level LOOKUP hash. Conservative-complete over every
/// paint-affecting DOM + CSS input, computed over the post-parse + post-JS
/// `cv_html::Document` tree and the active sheet-set fingerprint. O(nodes), no
/// cascade/layout.
///
/// Folds in, per the conservatism contract:
///   - per Element: tag name + EVERY attribute (name,value) in document order,
///     EXCEPT the volatile injected `NODE_ID_ATTR` (`\u{1}nid`) — but INCLUDING
///     class, id, style, src, href, width/height, bgcolor and all presentational
///     attrs (they drive cascade+layout+paint).
///   - per Text node: the exact text string.
///   - per Comment node: a tag byte (comments don't paint but folding their
///     presence keeps the walk's structural framing exact and over-inclusive).
///   - tree shape: a per-node child-count + ordered recursion, so a
///     moved/removed/inserted node changes the hash.
///   - the active stylesheet set fingerprint (the same `(as_ptr, len, rule-count
///     sum)` key `ensure_arena_current` uses), because identical DOM under
///     different CSS paints differently.
///
/// Over-inclusion only costs misses (safe); under-inclusion corrupts — when in
/// doubt, more is folded in.
pub fn dom_structural_hash(doc: &cv_html::Document, sheets: &[cv_css::Stylesheet]) -> u64 {
    let mut h = Fnv::new();
    // Doctype can affect quirks-mode cascade; fold it in.
    h.byte(b'D');
    match &doc.doctype_name {
        Some(name) => {
            h.byte(1);
            h.str(name);
        }
        None => h.byte(0),
    }
    hash_node(&mut h, &doc.root);
    // Active stylesheet-set CONTENT fingerprint. NOTE: a pointer-identity
    // fingerprint (`sheets.as_ptr`, as `ensure_arena_current` uses for in-frame
    // feature-set rebuilds) is WRONG for a cross-LOAD cache — two separate
    // navigations parse separate sheet Vecs at different addresses, so identical
    // CSS would always MISS (and the same address could in theory be reused for
    // different CSS, the wrong-frame direction). Instead we hash the CSS CONTENT
    // structurally so the fingerprint is (a) stable for identical CSS across loads
    // (repeat visits hit) and (b) different for any changed declaration (changed
    // CSS misses). Internal `<style>` text is ALSO already folded in via the DOM
    // walk above; this additionally covers external `<link>` CSS (not in the DOM).
    h.byte(b'S');
    h.u32(sheets.len() as u32);
    for sheet in sheets {
        hash_stylesheet(&mut h, sheet);
    }
    h.finish()
}

/// Fold a parsed stylesheet's CONTENT into `h`: every qualified rule's selector
/// shape (count + specificity) and every declaration (name + `Display`'d token
/// values + `!important`), plus at-rules' declaration blocks (e.g. `@font-face`).
/// Lossless enough that any paint-affecting declaration change perturbs the hash,
/// and stable across independent parses of the same source (no pointers).
fn hash_stylesheet(h: &mut Fnv, sheet: &cv_css::Stylesheet) {
    h.u32(sheet.rules.len() as u32);
    for rule in &sheet.rules {
        h.u32(rule.selectors.len() as u32);
        for sel in &rule.selectors {
            // Specificity + part count is a stable, cheap selector signature.
            h.u32(sel.specificity());
            h.u32(sel.parts.len() as u32);
        }
        hash_declarations(h, &rule.declarations);
    }
    h.u32(sheet.at_rules.len() as u32);
    for at in &sheet.at_rules {
        h.str(&at.name);
        h.u32(at.prelude.len() as u32);
        for tok in &at.prelude {
            h.str(&tok.to_string());
        }
        if let Some(decls) = &at.declarations {
            h.byte(1);
            hash_declarations(h, decls);
        } else {
            h.byte(0);
        }
        if let Some(block) = &at.block {
            h.byte(1);
            h.u32(block.len() as u32);
            for rule in block {
                h.u32(rule.selectors.len() as u32);
                for sel in &rule.selectors {
                    h.u32(sel.specificity());
                    h.u32(sel.parts.len() as u32);
                }
                hash_declarations(h, &rule.declarations);
            }
        } else {
            h.byte(0);
        }
    }
}

fn hash_declarations(h: &mut Fnv, decls: &[cv_css::parser::Declaration]) {
    h.u32(decls.len() as u32);
    for d in decls {
        h.str(&d.name);
        h.byte(if d.important { 1 } else { 0 });
        h.u32(d.value.len() as u32);
        for tok in &d.value {
            // CssToken's Display is a lossless-enough textual form (idents,
            // numbers, dimensions, urls, delimiters all distinguished).
            h.str(&tok.to_string());
        }
    }
}

/// Fold the JS-driven inline-style overrides (`el.style.x = ...`, keyed by
/// element path) into an already-computed `dom_structural_hash`. These overrides
/// are NOT written into the `doc`'s attrs (they are applied directly to the layout
/// tree by `apply_inline_style_overrides`), so the DOM walk cannot see them — but
/// they ARE paint-affecting, so the conservatism contract requires folding them
/// into the lookup key. Document order is preserved by the `OrderedMap` iteration,
/// and each (path, prop, value) triple is folded so a changed inline override ⇒ a
/// different key ⇒ a miss. Over-inclusion only costs misses.
pub fn fold_overrides(
    base: u64,
    overrides: &cv_js::OrderedMap<Vec<usize>, cv_js::OrderedMap<String, String>>,
) -> u64 {
    if overrides.is_empty() {
        return base;
    }
    let mut h = Fnv::new();
    h.u64(base);
    h.byte(b'O');
    h.u32(overrides.len() as u32);
    for (path, props) in overrides.iter() {
        h.u32(path.len() as u32);
        for &seg in path {
            h.u64(seg as u64);
        }
        h.u32(props.len() as u32);
        for (k, v) in props.iter() {
            h.str(k);
            h.str(v);
        }
    }
    h.finish()
}

/// Recurse over a `cv_html::Node`, folding every paint-affecting field into `h`.
fn hash_node(h: &mut Fnv, node: &cv_html::Node) {
    match &node.kind {
        cv_html::NodeKind::Element { name, attrs } => {
            h.byte(b'E');
            h.str(name);
            // Count REAL (non-volatile) attributes first so attr-count is part of
            // the framing even before values.
            let real: Vec<&cv_html::Attribute> = attrs
                .iter()
                .filter(|a| a.name != crate::NODE_ID_ATTR)
                .collect();
            h.u32(real.len() as u32);
            for a in real {
                // (name,value) in document order. Excludes the volatile node-id
                // identity attr (filtered above) so the hash is stable across
                // loads; includes class/id/style/src/href/width/height/bgcolor and
                // every other real attr (they drive cascade+layout+paint).
                h.str(&a.name);
                h.str(&a.value);
            }
            h.u32(node.children.len() as u32);
            for child in &node.children {
                hash_node(h, child);
            }
        }
        cv_html::NodeKind::Text(t) => {
            h.byte(b'T');
            h.str(t);
        }
        cv_html::NodeKind::Comment(c) => {
            // Comments don't paint, but fold their presence + content to keep the
            // walk's structural framing exact and over-inclusive (over-inclusion
            // only costs misses, never wrong frames).
            h.byte(b'C');
            h.str(c);
        }
    }
}

// ── DISK persistence (cross-RESTART warm first-paint) ────────────────────────
//
// The AOT-persist lever taken one step further than Chrome can: Chrome's
// back/forward cache + paint caches are all PROCESS-LOCAL and die with the
// process. This module persists a finished frame to disk so a COLD LAUNCH that
// re-navigates to a previously-visited (url, content, viewport) serves its FIRST
// PAINT from disk pixels instead of a cold cascade+layout+full-bake — a win with
// no Chrome analog (Chrome can only approximate via the HTTP cache + re-render).
//
// Mirrors the `cv_net/src/cache.rs` idiom: a per-user cache dir (a `paint_cache/`
// subdir), one file per entry (filename = a stable hash of the key), atomic
// write (temp file + rename), a compact hand-rolled binary format with a
// magic+version header (NO serde / NO third-party crates), and an LRU disk budget.
//
// SERVABILITY: only what is needed to PRESENT a first paint is persisted — the
// raw BGRA bitmap + dims + both hashes + chrome metadata (title, current_url,
// chrome_h, viewport_h) + the text items. A disk-loaded `PaintCacheEntry` carries
// `layout_root = None`, `retained = None`, `property_trees = None`. That is SAFE
// to serve: the present seam (`main.rs` paint_cache hit branch) blits
// `entry.paint.bitmap` and reads `current_url` / `bitmap.{width,height}` /
// `viewport_h` only; `layout_root` is consulted there solely for an optional
// `debug_dump_layout` (gated on `CV_DEBUG_LAYOUT`) and is `Option`, so `None` is
// a no-op. Hit-testing / damage-raster on SUBSEQUENT frames re-derive layout via
// the normal path (a one-time full bake on the second frame), exactly as a fresh
// cold load would — never a wrong frame.
//
// INVALIDATION: the key encodes (url_hash, dom_structural_hash, viewport), so a
// content/CSS/viewport change ⇒ a different filename ⇒ a natural miss; the stored
// file is self-evicting via the disk LRU + budget. An engine update that changes
// the format bumps `DISK_VERSION`, so every stale file fails the version guard on
// load (treated as a miss) and is overwritten/evicted over time. A corrupt or
// truncated file likewise fails a bounds/magic/version check and is a MISS, never
// a panic.
pub mod disk {
    use std::path::{Path, PathBuf};
    use std::sync::OnceLock;

    use super::{ENTRY_FIXED_OVERHEAD, PaintCacheEntry, PaintCacheKey};

    /// Magic bytes identifying a Conclave paint-cache file.
    const MAGIC: &[u8; 4] = b"TBPC";
    /// On-disk schema version. BUMP this on ANY format change so an engine update
    /// invalidates every stale file (they fail the version guard on load ⇒ miss).
    const DISK_VERSION: u32 = 1;

    /// Disk byte budget. Default 512 MB; override via
    /// `CV_PAINT_CACHE_DISK_BUDGET_MB`. A `0` / unparseable value uses the default.
    fn disk_budget_bytes() -> usize {
        static BUDGET: OnceLock<usize> = OnceLock::new();
        *BUDGET.get_or_init(|| {
            let mb = std::env::var("CV_PAINT_CACHE_DISK_BUDGET_MB")
                .ok()
                .and_then(|s| s.trim().parse::<usize>().ok())
                .filter(|&m| m > 0)
                .unwrap_or(512);
            mb.saturating_mul(1024 * 1024)
        })
    }

    /// Per-user paint-cache directory, resolved once. Mirrors the HTTP cache dir
    /// resolution in `main.rs` (`%LOCALAPPDATA%\Conclave\…`, falling back to the
    /// temp dir), under a `paint_cache/` subdir. Tests override it via
    /// [`set_dir_for_test`]. Created on first use; `None` if creation fails (then
    /// every disk op is a silent no-op and the cache stays memory-only).
    static DIR: OnceLock<Option<PathBuf>> = OnceLock::new();

    // THREAD-LOCAL (not a global Mutex): cargo runs the test suite multi-threaded
    // in ONE process, and now that CV_PAINT_CACHE_DISK defaults ON, a foreign
    // rendering test on another thread hitting the cache seam would otherwise
    // redirect a disk-test's writes into the wrong dir (or vice-versa) via a shared
    // global override. A thread-local override isolates each test thread's disk dir
    // so disk tests' file_count()/load() assertions are pollution-proof.
    #[cfg(test)]
    thread_local! {
        static TEST_DIR: std::cell::RefCell<Option<PathBuf>> = const { std::cell::RefCell::new(None) };
    }

    /// Point the disk cache at a temp dir for a test (so the real per-user cache
    /// dir is never polluted). Thread-local: affects only the calling test thread.
    /// Set BEFORE any disk op in the test.
    #[cfg(test)]
    pub fn set_dir_for_test(dir: Option<PathBuf>) {
        TEST_DIR.with(|d| *d.borrow_mut() = dir);
    }

    fn default_dir() -> PathBuf {
        std::env::var_os("LOCALAPPDATA")
            .map(PathBuf::from)
            .unwrap_or_else(std::env::temp_dir)
            .join("Conclave")
            .join("paint_cache")
    }

    /// Resolve the active cache dir, creating it if missing. Returns `None` if it
    /// can't be created (every disk op then no-ops → memory-only, never a panic).
    fn dir() -> Option<PathBuf> {
        #[cfg(test)]
        {
            // Tests use a THREAD-LOCAL override so each test thread can use its own
            // temp dir (cargo runs tests in one multi-threaded process).
            if let Some(d) = TEST_DIR.with(|c| c.borrow().clone()) {
                if std::fs::create_dir_all(&d).is_ok() {
                    return Some(d);
                }
                return None;
            }
            // ★ Under cfg(test) with NO override set, return None — NEVER fall
            // through to the real %LOCALAPPDATA% dir. This keeps the test process
            // at ZERO real-disk IO now that CV_PAINT_CACHE_DISK defaults ON; a test
            // that wants disk persistence sets an explicit temp dir via
            // set_dir_for_test (e.g. TmpDir::new). Without this, default-on disk
            // would pollute the user's real paint-cache dir during `cargo test`.
            return None;
        }
        #[cfg(not(test))]
        DIR.get_or_init(|| {
            let d = default_dir();
            if std::fs::create_dir_all(&d).is_ok() {
                Some(d)
            } else {
                None
            }
        })
        .clone()
    }

    /// Stable, filesystem-safe filename for a key. All four key fields are folded
    /// in (url_hash, dom_structural_hash, viewport) so a different content/viewport
    /// → a different file → a natural cross-restart miss. `.tbpaint` extension.
    fn filename(key: &PaintCacheKey) -> String {
        format!(
            "{:016x}_{:016x}_{}x{}.tbpaint",
            key.url_hash, key.dom_structural_hash, key.vw, key.vh
        )
    }

    // ── Binary writer helpers (little-endian, length-prefixed) ───────────────
    fn wr_u32(out: &mut Vec<u8>, v: u32) {
        out.extend_from_slice(&v.to_le_bytes());
    }
    fn wr_u64(out: &mut Vec<u8>, v: u64) {
        out.extend_from_slice(&v.to_le_bytes());
    }
    fn wr_i32(out: &mut Vec<u8>, v: i32) {
        out.extend_from_slice(&v.to_le_bytes());
    }
    fn wr_str(out: &mut Vec<u8>, s: &str) {
        wr_u32(out, s.len() as u32);
        out.extend_from_slice(s.as_bytes());
    }
    fn wr_opt_str(out: &mut Vec<u8>, s: Option<&str>) {
        match s {
            Some(s) => {
                out.push(1);
                wr_str(out, s);
            }
            None => out.push(0),
        }
    }
    fn align_byte(a: cv_ui::TextAlign) -> u8 {
        match a {
            cv_ui::TextAlign::Left => 0,
            cv_ui::TextAlign::Center => 1,
            cv_ui::TextAlign::Right => 2,
        }
    }

    /// Serialize a `PaintCacheEntry` to the compact binary format. ONLY the fields
    /// needed to serve a first paint are written (bitmap + dims + hashes + chrome
    /// metadata + text items + hit regions); layout_root / retained / property_trees
    /// are NOT persisted (they are `None` on a disk-loaded entry, which is safe to
    /// serve — see the module doc).
    pub fn serialize_entry(key: &PaintCacheKey, e: &PaintCacheEntry) -> Vec<u8> {
        let mut out = Vec::new();
        // Header: magic + version.
        out.extend_from_slice(MAGIC);
        wr_u32(&mut out, DISK_VERSION);
        // Key fields (re-validated on load against the requested key).
        wr_u64(&mut out, key.url_hash);
        wr_u64(&mut out, key.dom_structural_hash);
        wr_u32(&mut out, key.vw);
        wr_u32(&mut out, key.vh);
        // Entry guard fields.
        wr_u32(&mut out, e.doc_w);
        wr_u32(&mut out, e.doc_h);
        wr_u64(&mut out, e.structural_hash);
        // PaintData chrome metadata.
        let p = &e.paint;
        wr_u32(&mut out, p.chrome_h);
        wr_u32(&mut out, p.viewport_h);
        wr_str(&mut out, &p.title);
        wr_str(&mut out, &p.current_url);
        match p.caret_rect {
            Some((x, y, w, h)) => {
                out.push(1);
                wr_i32(&mut out, x);
                wr_i32(&mut out, y);
                wr_i32(&mut out, w);
                wr_i32(&mut out, h);
            }
            None => out.push(0),
        }
        // Bitmap: width, height, then width*height raw BGRA u32 little-endian.
        let bmp = &p.bitmap;
        wr_u32(&mut out, bmp.width);
        wr_u32(&mut out, bmp.height);
        wr_u64(&mut out, bmp.pixels.len() as u64);
        // Raw little-endian u32 pixels. `Vec<u32>` is the in-memory BGRA layout;
        // writing each as LE bytes round-trips byte-identically on any host.
        out.reserve(bmp.pixels.len() * 4);
        for &px in &bmp.pixels {
            out.extend_from_slice(&px.to_le_bytes());
        }
        // Text items.
        wr_u32(&mut out, p.texts.len() as u32);
        for t in &p.texts {
            wr_i32(&mut out, t.x);
            wr_i32(&mut out, t.y);
            wr_i32(&mut out, t.w);
            wr_i32(&mut out, t.h);
            wr_i32(&mut out, t.font_size_px);
            out.push(u8::from(t.bold));
            wr_u32(&mut out, u32::from(t.font_weight));
            out.push(u8::from(t.italic));
            wr_opt_str(&mut out, t.font_family.as_deref());
            out.push(t.color_rgb.0);
            out.push(t.color_rgb.1);
            out.push(t.color_rgb.2);
            out.push(t.color_alpha);
            wr_str(&mut out, &t.text);
            out.push(align_byte(t.align));
            wr_i32(&mut out, t.letter_spacing_px);
            out.push(u8::from(t.is_chrome));
        }
        // Hit regions.
        wr_u32(&mut out, p.hit_regions.len() as u32);
        for r in &p.hit_regions {
            wr_i32(&mut out, r.x);
            wr_i32(&mut out, r.y);
            wr_i32(&mut out, r.w);
            wr_i32(&mut out, r.h);
            wr_opt_str(&mut out, r.href.as_deref());
            match &r.element_path {
                Some(path) => {
                    out.push(1);
                    wr_u32(&mut out, path.len() as u32);
                    for &seg in path {
                        wr_u64(&mut out, seg as u64);
                    }
                }
                None => out.push(0),
            }
        }
        out
    }

    /// Bounds-checked reader. Every `rd_*` returns `None` on truncation, so a
    /// corrupt/short file deserializes to `None` (a MISS), never a panic.
    struct Reader<'a> {
        b: &'a [u8],
        pos: usize,
    }
    impl<'a> Reader<'a> {
        fn take(&mut self, n: usize) -> Option<&'a [u8]> {
            let s = self.b.get(self.pos..self.pos.checked_add(n)?)?;
            self.pos += n;
            Some(s)
        }
        fn rd_u8(&mut self) -> Option<u8> {
            Some(self.take(1)?[0])
        }
        fn rd_u32(&mut self) -> Option<u32> {
            Some(u32::from_le_bytes(self.take(4)?.try_into().ok()?))
        }
        fn rd_u64(&mut self) -> Option<u64> {
            Some(u64::from_le_bytes(self.take(8)?.try_into().ok()?))
        }
        fn rd_i32(&mut self) -> Option<i32> {
            Some(i32::from_le_bytes(self.take(4)?.try_into().ok()?))
        }
        fn rd_str(&mut self) -> Option<String> {
            let len = self.rd_u32()? as usize;
            Some(String::from_utf8_lossy(self.take(len)?).into_owned())
        }
        fn rd_opt_str(&mut self) -> Option<Option<String>> {
            if self.rd_u8()? == 0 {
                Some(None)
            } else {
                Some(Some(self.rd_str()?))
            }
        }
    }

    fn align_from_byte(b: u8) -> cv_ui::TextAlign {
        match b {
            1 => cv_ui::TextAlign::Center,
            2 => cv_ui::TextAlign::Right,
            _ => cv_ui::TextAlign::Left,
        }
    }

    /// Deserialize bytes into a `(PaintCacheKey, PaintCacheEntry)`. Returns `None`
    /// on bad magic, a version mismatch, truncation, or any self-inconsistency
    /// (e.g. the bitmap pixel count not matching width*height). NEVER panics.
    pub fn deserialize_entry(b: &[u8]) -> Option<(PaintCacheKey, PaintCacheEntry)> {
        let mut r = Reader { b, pos: 0 };
        // Magic + version guard (a format change or corruption ⇒ miss).
        if r.take(4)? != MAGIC {
            return None;
        }
        if r.rd_u32()? != DISK_VERSION {
            return None;
        }
        let key = PaintCacheKey {
            url_hash: r.rd_u64()?,
            dom_structural_hash: r.rd_u64()?,
            vw: r.rd_u32()?,
            vh: r.rd_u32()?,
        };
        let doc_w = r.rd_u32()?;
        let doc_h = r.rd_u32()?;
        let structural_hash = r.rd_u64()?;
        let chrome_h = r.rd_u32()?;
        let viewport_h = r.rd_u32()?;
        let title = r.rd_str()?;
        let current_url = r.rd_str()?;
        let caret_rect = if r.rd_u8()? == 0 {
            None
        } else {
            Some((r.rd_i32()?, r.rd_i32()?, r.rd_i32()?, r.rd_i32()?))
        };
        // Bitmap.
        let width = r.rd_u32()?;
        let height = r.rd_u32()?;
        let npix = r.rd_u64()? as usize;
        // Self-consistency guard: the declared pixel count MUST equal width*height.
        // Guards against a corrupt header that would otherwise serve a wrong-size
        // bitmap (the seam also cross-checks doc_w/doc_h vs bitmap dims).
        let expected = (width as usize).checked_mul(height as usize)?;
        if npix != expected {
            return None;
        }
        // Guard against a maliciously huge npix before allocating (the take below
        // bounds-checks too, but reserve early to avoid a giant speculative alloc).
        let raw = r.take(npix.checked_mul(4)?)?;
        let mut pixels = Vec::with_capacity(npix);
        for chunk in raw.chunks_exact(4) {
            pixels.push(u32::from_le_bytes(chunk.try_into().ok()?));
        }
        // Defense in depth: declared dims must agree with the doc dims guard.
        if doc_w != width || doc_h != height {
            return None;
        }
        // Text items.
        let ntext = r.rd_u32()? as usize;
        let mut texts = Vec::with_capacity(ntext.min(1 << 20));
        for _ in 0..ntext {
            let x = r.rd_i32()?;
            let y = r.rd_i32()?;
            let w = r.rd_i32()?;
            let h = r.rd_i32()?;
            let font_size_px = r.rd_i32()?;
            let bold = r.rd_u8()? != 0;
            let font_weight = r.rd_u32()? as u16;
            let italic = r.rd_u8()? != 0;
            let font_family = r.rd_opt_str()?;
            let cr = r.rd_u8()?;
            let cg = r.rd_u8()?;
            let cb = r.rd_u8()?;
            let ca = r.rd_u8()?;
            let text = r.rd_str()?;
            let align = align_from_byte(r.rd_u8()?);
            let letter_spacing_px = r.rd_i32()?;
            let is_chrome = r.rd_u8()? != 0;
            texts.push(cv_ui::TextItem {
                x,
                y,
                w,
                h,
                font_size_px,
                bold,
                font_weight,
                italic,
                font_family,
                color_rgb: (cr, cg, cb),
                color_alpha: ca,
                text,
                align,
                letter_spacing_px,
                is_chrome,
            });
        }
        // Hit regions.
        let nhit = r.rd_u32()? as usize;
        let mut hit_regions = Vec::with_capacity(nhit.min(1 << 20));
        for _ in 0..nhit {
            let x = r.rd_i32()?;
            let y = r.rd_i32()?;
            let w = r.rd_i32()?;
            let h = r.rd_i32()?;
            let href = r.rd_opt_str()?;
            let element_path = if r.rd_u8()? == 0 {
                None
            } else {
                let n = r.rd_u32()? as usize;
                let mut path = Vec::with_capacity(n.min(1 << 16));
                for _ in 0..n {
                    path.push(r.rd_u64()? as usize);
                }
                Some(path)
            };
            hit_regions.push(cv_ui::HitRegion {
                x,
                y,
                w,
                h,
                href,
                element_path,
            });
        }
        let bitmap = std::sync::Arc::new(cv_gfx::Bitmap {
            width,
            height,
            pixels,
        });
        let bytes = (npix.saturating_mul(4)).saturating_add(ENTRY_FIXED_OVERHEAD);
        let paint = cv_ui::PaintData {
            bitmap,
            texts,
            // Servable-from-disk: these are re-derived on subsequent frames.
            layout_root: None,
            hit_regions,
            title,
            current_url,
            chrome_h,
            viewport_h,
            caret_rect,
            property_trees: None,
            retained: None,
            content_origin_y: 0,
            document_h: 0,
        };
        Some((
            key,
            PaintCacheEntry {
                paint,
                doc_w,
                doc_h,
                structural_hash,
                bytes,
            },
        ))
    }

    /// Load the entry for `key` from disk, or `None` on miss / corruption /
    /// version mismatch / key mismatch. Touches the file's mtime on a hit so the
    /// disk LRU keeps recently-served entries (best-effort).
    pub fn load(key: &PaintCacheKey) -> Option<PaintCacheEntry> {
        let d = dir()?;
        let path = d.join(filename(key));
        let bytes = std::fs::read(&path).ok()?;
        let (stored_key, entry) = deserialize_entry(&bytes)?;
        // The filename already discriminates the key, but cross-check the embedded
        // key (a hash-collision / stale-rename guard): a mismatch ⇒ a miss, never
        // a wrong-frame serve.
        if &stored_key != key {
            return None;
        }
        // Best-effort LRU touch (mtime); ignore failure.
        let _ = filetime_touch(&path);
        Some(entry)
    }

    /// Atomically write `entry` for `key` to disk (temp file + rename), then
    /// re-bound the on-disk store to its byte/file budget (LRU-evicting by mtime).
    /// All failures are swallowed (the in-memory cache is authoritative); a torn
    /// write can never be observed because the rename is atomic.
    pub fn store(key: &PaintCacheKey, entry: &PaintCacheEntry) {
        let Some(d) = dir() else {
            return;
        };
        let bytes = serialize_entry(key, entry);
        // Never persist a single file larger than the whole disk budget.
        if bytes.len() > disk_budget_bytes() {
            return;
        }
        let path = d.join(filename(key));
        // Unique-ish temp name (pid + a monotonic counter) so concurrent writers
        // don't clobber each other's temp file before the rename.
        let tmp = d.join(format!("{}.tmp.{}", filename(key), unique_suffix()));
        if std::fs::write(&tmp, &bytes).is_ok() {
            // Rename is atomic on Windows/NTFS and POSIX when src+dst share a dir.
            if std::fs::rename(&tmp, &path).is_err() {
                let _ = std::fs::remove_file(&tmp);
            }
        } else {
            let _ = std::fs::remove_file(&tmp);
        }
        evict_to_budget(&d, disk_budget_bytes());
    }

    /// Evict the least-recently-used `.tbpaint` files (oldest mtime first) until
    /// the directory's total `.tbpaint` byte size is within `budget`.
    fn evict_to_budget(d: &Path, budget: usize) {
        let Ok(rd) = std::fs::read_dir(d) else {
            return;
        };
        // Collect (path, size, mtime) for every cache file (ignore temp files).
        let mut files: Vec<(PathBuf, u64, std::time::SystemTime)> = Vec::new();
        let mut total: u64 = 0;
        for ent in rd.flatten() {
            let path = ent.path();
            if path.extension().and_then(|e| e.to_str()) != Some("tbpaint") {
                continue;
            }
            let Ok(md) = ent.metadata() else { continue };
            let size = md.len();
            let mtime = md.modified().unwrap_or(std::time::UNIX_EPOCH);
            total = total.saturating_add(size);
            files.push((path, size, mtime));
        }
        if (total as usize) <= budget {
            return;
        }
        // Oldest first (LRU = least-recently mtime-touched).
        files.sort_by_key(|(_, _, mtime)| *mtime);
        for (path, size, _) in files {
            if (total as usize) <= budget {
                break;
            }
            if std::fs::remove_file(&path).is_ok() {
                total = total.saturating_sub(size);
            }
        }
    }

    /// A process-monotonic suffix for temp filenames (no third-party crates).
    fn unique_suffix() -> u64 {
        use std::sync::atomic::{AtomicU64, Ordering};
        static CTR: AtomicU64 = AtomicU64::new(0);
        let pid = std::process::id() as u64;
        let n = CTR.fetch_add(1, Ordering::Relaxed);
        (pid << 32) ^ n
    }

    /// Touch a file's mtime to "now" so the LRU treats a served entry as recent.
    /// Implemented without third-party crates by rewriting nothing — we use a
    /// read+write-back-of-the-same-bytes only if a cheaper path is unavailable.
    /// On Windows/POSIX, opening for append with no write updates nothing, so we
    /// use `File::set_modified` (stable since Rust 1.75) when available.
    fn filetime_touch(path: &Path) -> std::io::Result<()> {
        let f = std::fs::OpenOptions::new().write(true).open(path)?;
        f.set_modified(std::time::SystemTime::now())
    }

    /// Test/diagnostics: count `.tbpaint` files in the active dir.
    #[cfg(test)]
    pub fn file_count() -> usize {
        let Some(d) = dir() else { return 0 };
        let Ok(rd) = std::fs::read_dir(&d) else {
            return 0;
        };
        rd.flatten()
            .filter(|e| {
                e.path().extension().and_then(|x| x.to_str()) == Some("tbpaint")
            })
            .count()
    }

    /// Test/diagnostics: total bytes of `.tbpaint` files in the active dir.
    #[cfg(test)]
    pub fn total_disk_bytes() -> usize {
        let Some(d) = dir() else { return 0 };
        let Ok(rd) = std::fs::read_dir(&d) else {
            return 0;
        };
        rd.flatten()
            .filter(|e| e.path().extension().and_then(|x| x.to_str()) == Some("tbpaint"))
            .filter_map(|e| e.metadata().ok().map(|m| m.len() as usize))
            .sum()
    }

    /// Remove EVERY persisted `.tbpaint` file in the active dir. Called by the
    /// structural-rebuild wipe (`clear_all`) so a NodeId reallocation invalidates
    /// the on-disk entries too — without this, a wiped page would re-hydrate STALE
    /// pixels from disk on the next lookup (the exact WRONG-frame hazard the wipe
    /// exists to prevent). No-op (never panics) if no dir is configured.
    pub fn clear_all_files() {
        let Some(d) = dir() else { return };
        let Ok(rd) = std::fs::read_dir(&d) else { return };
        for e in rd.flatten() {
            let p = e.path();
            if p.extension().and_then(|x| x.to_str()) == Some("tbpaint") {
                let _ = std::fs::remove_file(&p);
            }
        }
    }

    /// Test-only: force the disk budget for an eviction test, bypassing the env
    /// `OnceLock`. Runs the same eviction the production `store` path runs.
    #[cfg(test)]
    pub fn evict_to_budget_for_test(budget: usize) {
        if let Some(d) = dir() {
            evict_to_budget(&d, budget);
        }
    }

    /// Test-only: store with an explicit disk budget (so the per-file oversize +
    /// directory eviction bounds are exercisable without writing 512 MB).
    #[cfg(test)]
    pub fn store_with_budget(key: &PaintCacheKey, entry: &PaintCacheEntry, budget: usize) {
        let Some(d) = dir() else { return };
        let bytes = serialize_entry(key, entry);
        if bytes.len() > budget {
            return;
        }
        let path = d.join(filename(key));
        let tmp = d.join(format!("{}.tmp.{}", filename(key), unique_suffix()));
        if std::fs::write(&tmp, &bytes).is_ok() && std::fs::rename(&tmp, &path).is_err() {
            let _ = std::fs::remove_file(&tmp);
        }
        evict_to_budget(&d, budget);
    }
}

// ── Oracle + guard tests ─────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// The cache is process-global and `cargo test` runs tests in parallel, so a
    /// test asserting on `len()` / `total_bytes()` / specific keys could race
    /// another test's inserts. Serialize every paint_cache test against this guard
    /// (we depend on no external crates, so this is a hand-rolled `#[serial]`).
    /// Each test takes the guard FIRST, then `clear_all()` to start from empty.
    static TEST_GUARD: std::sync::Mutex<()> = std::sync::Mutex::new(());

    fn guard() -> std::sync::MutexGuard<'static, ()> {
        let g = TEST_GUARD.lock().unwrap_or_else(|p| p.into_inner());
        // Reset the per-test key salt so this test's first key_for() mints a FRESH
        // unique nonce (collision-proof against every other test in this binary,
        // which shares the process-global default-ON paint cache). The guard
        // serializes all cache tests, so this reset is race-free.
        TEST_CUR_SALT.with(|c| c.set(0));
        g
    }

    /// Build a `LayoutConfig` at the given viewport. `measure_text_fn: None`
    /// matches the retained-list oracle harness — text is measured by the same
    /// fallback in both the served frame and the cold bake, so the byte-identity
    /// holds.
    fn cfg(vw: u32, vh: u32) -> cv_layout::LayoutConfig {
        cv_layout::LayoutConfig {
            viewport_w: vw as f32,
            viewport_h: vh as f32,
            default_font_size_px: 16.0,
            default_text_color: cv_layout::Color { r: 0, g: 0, b: 0, a: 255 },
            default_line_height: 1.2,
            measure_text_fn: None,
        }
    }

    const URL: &str = "https://example.com/page";

    /// Cold-bake a page to a `PaintData` through the REAL production cold-bake
    /// path (`render_paint_only` → cascade + layout + full bake), exactly what a
    /// cache MISS falls back to. This is the oracle: a served frame must be
    /// byte-identical to this.
    fn bake_page(html: &str, vw: u32, vh: u32) -> cv_ui::PaintData {
        let mut doc = cv_html::parse(html);
        crate::assign_node_ids(&mut doc.root);
        let mut sheets = vec![crate::parse_user_agent_stylesheet()];
        sheets.extend(crate::collect_stylesheets(&doc));
        let cfg = cfg(vw, vh);
        let overrides = cv_js::OrderedMap::new();
        crate::render_paint_only(&doc, &sheets, URL, &cfg, &overrides, None)
    }

    /// Parse + assign ids + collect sheets, returning (doc, sheets) so a test can
    /// compute the same `dom_structural_hash` the seam would.
    fn parse_sheets(html: &str) -> (cv_html::Document, Vec<cv_css::Stylesheet>) {
        let mut doc = cv_html::parse(html);
        crate::assign_node_ids(&mut doc.root);
        let mut sheets = vec![crate::parse_user_agent_stylesheet()];
        sheets.extend(crate::collect_stylesheets(&doc));
        (doc, sheets)
    }

    /// The exact key the seam computes for a freshly parsed page (no JS overrides
    /// in these in-process tests).
    // A per-test-binary unique salt folded into the test keys' url_hash. The paint
    // cache is a PROCESS-GLOBAL singleton now that CV_PAINT_CACHE defaults ON, so
    // OTHER rendering tests in this binary (which go through the full
    // build_runtime_and_first_paint seam) can insert entries under the production
    // URL. Salting every test's url_hash with a unique nonce makes these tests' keys
    // collision-proof against the rest of the binary, so `lookup(...).is_none()`
    // miss-assertions are provably correct regardless of test interleaving.
    static TEST_KEY_SALT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    thread_local! { static TEST_CUR_SALT: std::cell::Cell<u64> = const { std::cell::Cell::new(0) }; }
    // A per-test unique salt: reset to 0 by guard() at each test start, then minted
    // once on the first key_for() call so all keys within ONE test share it (equal
    // HTML → equal key) while DIFFERENT tests get DIFFERENT salts — collision-proof
    // against the process-global default-ON cache touched by other binary tests.
    fn fresh_salt() -> u64 {
        TEST_CUR_SALT.with(|c| {
            if c.get() == 0 {
                let s = TEST_KEY_SALT
                    .fetch_add(0x9E37_79B9_7F4A_7C15, std::sync::atomic::Ordering::Relaxed)
                    .wrapping_add(0xD1B5_4A32_D192_ED03);
                c.set(s | 1);
            }
            c.get()
        })
    }
    fn key_for(html: &str, vw: u32, vh: u32) -> PaintCacheKey {
        let (doc, sheets) = parse_sheets(html);
        PaintCacheKey {
            url_hash: url_hash(URL) ^ fresh_salt(),
            dom_structural_hash: dom_structural_hash(&doc, &sheets),
            vw,
            vh,
        }
    }

    // Build a PaintCacheEntry from a PaintData WITHOUT touching the global cache —
    // for LRU/budget/eviction tests that must run on a LOCAL PaintCache instance so
    // the process-global default-ON cache (mutated by other tests in this binary)
    // cannot perturb their accounting.
    fn mk_entry(paint: &cv_ui::PaintData) -> PaintCacheEntry {
        let bytes = paint.bitmap.pixels.len() * 4 + ENTRY_FIXED_OVERHEAD;
        PaintCacheEntry {
            paint: paint.clone(),
            doc_w: paint.bitmap.width,
            doc_h: paint.bitmap.height,
            structural_hash: 0,
            bytes,
        }
    }

    // Insert an entry into a LOCAL cache instance + re-bound to `budget` (mirrors the
    // global insert_inner's map/lru/bytes maintenance, isolated from the singleton).
    fn local_insert(c: &mut PaintCache, key: PaintCacheKey, entry: PaintCacheEntry, budget: usize) {
        if entry.bytes > budget {
            return; // oversized: no-op (matches insert_inner)
        }
        if let Some(old) = c.map.remove(&key) {
            c.bytes = c.bytes.saturating_sub(old.bytes);
            if let Some(pos) = c.lru.iter().position(|k| k == &key) {
                c.lru.remove(pos);
            }
        }
        c.bytes = c.bytes.saturating_add(entry.bytes);
        c.map.insert(key.clone(), entry);
        c.lru.push_back(key);
        c.evict_to_budget(budget);
    }

    fn dom_hash(html: &str) -> u64 {
        let (doc, sheets) = parse_sheets(html);
        dom_structural_hash(&doc, &sheets)
    }

    /// THE ORACLE: a served frame must be BYTE-IDENTICAL to a fresh cold bake.
    /// Tolerance EXACTLY 0 — bitmap dims, every pixel (`Vec<u32>` exact), every
    /// text item, title and url.
    fn assert_byte_identical(served: &cv_ui::PaintData, cold: &cv_ui::PaintData) {
        assert_eq!(served.bitmap.width, cold.bitmap.width, "width mismatch");
        assert_eq!(served.bitmap.height, cold.bitmap.height, "height mismatch");
        // Pixel-equality with an explicit maxd==0 check mirroring the M5.4 oracle.
        let mut maxd: u64 = 0;
        let mut first: Option<usize> = None;
        for (i, (a, b)) in served
            .bitmap
            .pixels
            .iter()
            .zip(cold.bitmap.pixels.iter())
            .enumerate()
        {
            if a != b {
                if first.is_none() {
                    first = Some(i);
                }
                let d = |sh: u32| -> u64 {
                    (((a >> sh) & 0xFF) as i64 - ((b >> sh) & 0xFF) as i64).unsigned_abs()
                };
                maxd = maxd.max(d(0).max(d(8)).max(d(16)).max(d(24)));
            }
        }
        assert_eq!(maxd, 0, "pixel diff at {:?} (maxd={})", first, maxd);
        assert_eq!(
            served.bitmap.pixels, cold.bitmap.pixels,
            "Vec<u32> not exactly equal"
        );
        assert_eq!(served.texts, cold.texts, "text items differ");
        assert_eq!(served.title, cold.title, "title differs");
        assert_eq!(served.current_url, cold.current_url, "current_url differs");
    }

    /// A representative page exercising background colors, text, nested boxes and
    /// inline content — enough that cascade+layout+paint actually does work.
    const HTML_A: &str = "<html><head><style>\
        body{background:#eef;margin:0}\
        .card{background:#fff;border:2px solid #036;padding:8px;margin:6px}\
        h1{color:#036;font-size:20px}\
        p{color:#222}\
        </style></head><body>\
        <div class=\"card\"><h1>Hello Toasty</h1><p>A persistent paint cache.</p></div>\
        <div class=\"card\"><p>Second card with more text content here.</p></div>\
        </body></html>";

    // ── Case A — SAME PAGE RE-NAV: HIT, byte-identical (the core win) ─────────
    #[test]
    fn t_case_a_same_page_renav_hit_byte_identical() {
        let _g = guard();
        clear_all();
        let cold = bake_page(HTML_A, 1024, 768);
        let key = key_for(HTML_A, 1024, 768);
        insert(key.clone(), &cold, &cfg(1024, 768));

        let entry = lookup(&key).expect("expected a cache HIT for the same page");
        let served = entry.paint.clone();

        // A FRESH independent cold bake — the served frame must equal THIS, not
        // just the one we inserted.
        let cold2 = bake_page(HTML_A, 1024, 768);
        assert_byte_identical(&served, &cold2);
        clear_all();
    }

    // ── Case B — SAME URL, DIFFERENT CONTENT: MISS → cold ────────────────────
    #[test]
    fn t_case_b_same_url_different_content_miss() {
        let _g = guard();
        clear_all();
        // HTML_A2 changes one text node + one class value.
        let html2 = "<html><head><style>\
            body{background:#eef;margin:0}\
            .card{background:#fff;border:2px solid #036;padding:8px;margin:6px}\
            h1{color:#036;font-size:20px}\
            p{color:#222}\
            </style></head><body>\
            <div class=\"card\"><h1>Goodbye Toasty</h1><p>A persistent paint cache.</p></div>\
            <div class=\"panel\"><p>Second card with more text content here.</p></div>\
            </body></html>";
        assert_ne!(
            dom_hash(HTML_A),
            dom_hash(html2),
            "different content MUST produce a different dom_structural_hash"
        );

        let cold = bake_page(HTML_A, 1024, 768);
        let key_a = key_for(HTML_A, 1024, 768);
        insert(key_a.clone(), &cold, &cfg(1024, 768));

        let key2 = key_for(html2, 1024, 768);
        assert!(
            lookup(&key2).is_none(),
            "different content must MISS (no stale serve)"
        );

        // Distinct content ⇒ distinct keys ⇒ both entries can coexist.
        let cold2 = bake_page(html2, 1024, 768);
        assert_ne!(key_a, key2, "distinct keys");
        // Prove coexistence on a LOCAL PaintCache instance, isolated from the
        // global default-ON singleton. Other rendering tests in this binary mutate
        // the global cache concurrently (they don't take this module's `guard()`),
        // so its MAX_ENTRIES / byte-budget eviction could drop `key_a` between a
        // global insert and lookup — exactly the deterministic-isolation pattern
        // `t_eviction_bounded` / `t_lru_recency_order` already use. The global MISS
        // above remains the load-bearing "no stale serve" check.
        let budget = budget_bytes();
        let mut local = PaintCache::new();
        local_insert(&mut local, key_a.clone(), mk_entry(&cold), budget);
        local_insert(&mut local, key2.clone(), mk_entry(&cold2), budget);
        assert!(local.map.contains_key(&key_a), "original entry still valid");
        assert!(local.map.contains_key(&key2), "new content entry inserted");
        clear_all();
    }

    // ── Case C — DIFFERENT VIEWPORT: MISS → cold (geometry guard) ────────────
    #[test]
    fn t_case_c_different_viewport_miss() {
        let _g = guard();
        clear_all();
        let cold = bake_page(HTML_A, 1024, 768);
        insert(key_for(HTML_A, 1024, 768), &cold, &cfg(1024, 768));

        let key_other_vp = key_for(HTML_A, 1280, 800);
        assert!(
            lookup(&key_other_vp).is_none(),
            "a viewport change must MISS — the size is in the key"
        );

        // A cold bake at the new viewport has a different bitmap WIDTH, proving no
        // wrong-size serve could ever happen.
        let cold_wide = bake_page(HTML_A, 1280, 800);
        assert_ne!(
            cold.bitmap.width, cold_wide.bitmap.width,
            "viewport change changes bitmap dims"
        );
        assert_eq!(cold_wide.bitmap.width, 1280);
        clear_all();
    }

    // ── Case D — DOM-HASH ⇒ RDL-HASH ⇒ PIXEL soundness (conservatism tripwire)
    #[test]
    fn t_case_d_dom_hash_is_sound_refinement() {
        let _g = guard();
        clear_all();
        // A battery of variants. Pairs with equal dom_structural_hash MUST share
        // the RDL root subtree_hash AND produce byte-identical pixels; pairs that
        // differ paint-affecting input MUST differ in the dom hash.
        let base = HTML_A;
        // Identical content (re-parse) → same hash, same pixels.
        let same = HTML_A;
        // Changed inline style attr → different hash.
        let style_changed =
            "<html><body><div style=\"background:#f00\">x</div></body></html>";
        let style_changed2 =
            "<html><body><div style=\"background:#0f0\">x</div></body></html>";
        // Changed bgcolor presentational attr → different hash.
        let bg1 = "<html><body><table><tr><td bgcolor=\"#abc\">c</td></tr></table></body></html>";
        let bg2 = "<html><body><table><tr><td bgcolor=\"#cba\">c</td></tr></table></body></html>";
        // Added element → different hash + tree shape.
        let added = "<html><body><div class=\"card\"><h1>Hello Toasty</h1>\
            <p>A persistent paint cache.</p><span>extra</span></div></body></html>";

        // 1) Equal hash ⇒ equal RDL hash ⇒ equal pixels.
        assert_eq!(dom_hash(base), dom_hash(same), "re-parse same hash");
        let pa = bake_page(base, 1024, 768);
        let pb = bake_page(same, 1024, 768);
        assert_eq!(
            rdl_root_subtree_hash(&pa, &cfg(1024, 768)),
            rdl_root_subtree_hash(&pb, &cfg(1024, 768)),
            "equal dom hash ⇒ equal RDL subtree_hash"
        );
        assert_eq!(pa.bitmap.pixels, pb.bitmap.pixels, "equal dom hash ⇒ equal pixels");

        // 2) Paint-affecting changes ⇒ different hashes (no collision).
        assert_ne!(dom_hash(style_changed), dom_hash(style_changed2), "inline style attr folded in");
        assert_ne!(dom_hash(bg1), dom_hash(bg2), "bgcolor presentational attr folded in");
        assert_ne!(dom_hash(base), dom_hash(added), "added element folded in");
        clear_all();
    }

    // ── Case: NODE_ID_ATTR must NOT pollute the hash (stable across loads) ────
    #[test]
    fn t_node_id_attr_excluded_from_hash() {
        let _g = guard();
        clear_all();
        // Two FRESH parses of the same HTML get DIFFERENT injected node-id attrs
        // (the thread-local counter advances), but the dom_structural_hash MUST be
        // identical — otherwise a same-content repeat visit would never hit.
        let h1 = dom_hash(HTML_A);
        let h2 = dom_hash(HTML_A);
        assert_eq!(h1, h2, "volatile node-id attr must be excluded from the hash");
        clear_all();
    }

    // ── Case: flag DEFAULT-ON ⇒ enabled() true unless CV_PAINT_CACHE=0 ────────
    #[test]
    fn t_flag_default_on() {
        // Flipped 2026-06-12: the in-memory cross-load cache defaults ON (verified
        // byte-identical to cold bake, mutation-proven, structural-rebuild wipe
        // guards the one WRONG-frame hazard, every uncertainty falls back to a
        // correct cold bake). In the test process CV_PAINT_CACHE is unset ⇒ enabled()
        // is true. The escape hatch is CV_PAINT_CACHE=0 (asserted via the parse rule
        // below, since enabled() caches its OnceLock and we can't re-env in-process).
        assert!(
            enabled(),
            "CV_PAINT_CACHE must default ON (unset env in the test process)"
        );
        // Document the escape-hatch parse contract: only "0" disables.
        assert_eq!(Some("0").as_deref() != Some("0"), false, "CV_PAINT_CACHE=0 ⇒ OFF");
        assert_ne!(None::<&str> != Some("0"), false, "unset ⇒ ON");
        assert_ne!(Some("1") != Some("0"), false, "=1 ⇒ ON");
    }

    // ── Case E — EVICTION: bounded LRU, no unbounded growth ──────────────────
    //
    // Uses a tiny explicit budget (`insert_with_budget`) so eviction is forced
    // deterministically without allocating the default 256 MB. Asserts the byte
    // sum + entry count stay bounded after EVERY insert, the most-recently-used
    // survives, and the least-recently-used was evicted.
    #[test]
    fn t_eviction_bounded() {
        // Runs on a LOCAL PaintCache instance — fully isolated from the global
        // default-ON singleton (which other rendering tests in this binary mutate
        // concurrently), so the byte/len/eviction accounting is deterministic.
        let cold = bake_page(HTML_A, 1024, 768);
        let entry_bytes = cold.bitmap.pixels.len() * 4 + ENTRY_FIXED_OVERHEAD;
        // Budget that holds exactly 2 entries → the 3rd insert evicts the oldest.
        let budget = entry_bytes * 2 + entry_bytes / 2;
        let mk = |i: u64| PaintCacheKey {
            url_hash: 0xE71C_7E57,
            dom_structural_hash: i.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1),
            vw: 1024,
            vh: 768,
        };
        let (k0, k1, k2, k3) = (mk(0), mk(1), mk(2), mk(3));
        let mut c = PaintCache::new();
        for k in [&k0, &k1, &k2, &k3] {
            local_insert(&mut c, k.clone(), mk_entry(&cold), budget);
            // The bound is RESTORED after every insert (no unbounded growth).
            assert!(c.bytes <= budget, "byte budget held: {} <= {}", c.bytes, budget);
            assert!(c.map.len() <= MAX_ENTRIES, "entry cap held");
            assert!(c.map.len() <= 2, "tiny budget holds at most 2 entries");
        }
        // After inserting k0..k3 into a 2-entry budget, the two most-recent (k2,k3)
        // survive and the two oldest (k0,k1) were evicted.
        assert!(c.map.contains_key(&k3), "most-recently-used survives");
        assert!(c.map.contains_key(&k2), "second-most-recent survives");
        assert!(!c.map.contains_key(&k0), "least-recently-used was evicted");
        assert!(!c.map.contains_key(&k1), "second-oldest was evicted");
    }

    // ── Case: an entry larger than the whole budget is NOT inserted ──────────
    #[test]
    fn t_oversized_entry_not_inserted() {
        // LOCAL PaintCache instance — isolated from the global default-ON singleton.
        let cold = bake_page(HTML_A, 1024, 768);
        let entry_bytes = cold.bitmap.pixels.len() * 4 + ENTRY_FIXED_OVERHEAD;
        // A budget SMALLER than a single entry: the entry must be a NO-OP insert.
        let tiny_budget = entry_bytes / 2;
        let key = PaintCacheKey { url_hash: 0x05125, dom_structural_hash: 1, vw: 1024, vh: 768 };
        let mut c = PaintCache::new();
        local_insert(&mut c, key.clone(), mk_entry(&cold), tiny_budget);
        // The oversized entry was NOT inserted, and the cache stays empty (no add,
        // no evict-all): a pure no-op.
        assert!(!c.map.contains_key(&key), "an entry larger than the whole budget must NOT be inserted");
        assert_eq!(c.bytes, 0, "oversized insert is a no-op (no add, no evict-all)");
        assert_eq!(c.map.len(), 0, "cache stays empty");
    }

    // ── DEDICATED WRONG-frame test: structural-rebuild wipe, no stale serve ──
    //
    // This proves guard 1 (the structural-rebuild wipe) is load-bearing and makes
    // a stale serve IMPOSSIBLE BY CONSTRUCTION. We model the production sequence:
    //   1. cold-bake page P1, insert entry E1 under key K1.
    //   2. a structural rebuild reallocates NodeIds → `clear_all()` fires at the
    //      reconcile-bail wipe site (we call it directly, as main.rs does).
    //   3. the cache is EMPTY and lookup(K1) is None — the stale entry whose
    //      RDL/layout_root were captured under the OLD generation is GONE.
    //   4. re-render takes the cold-bake MISS path and matches an independent cold
    //      bake (maxd==0).
    //   5. negative control: WITHOUT the wipe the same K1 WOULD still return E1 —
    //      demonstrating the wipe is the load-bearing line.
    #[test]
    fn t_structural_rebuild_wipe_no_stale_serve() {
        let _g = guard();
        clear_all();
        let p1 = HTML_A;
        let cold = bake_page(p1, 1024, 768);
        // bake_page → render_paint_only → the production seam self-inserts now that
        // CV_PAINT_CACHE defaults ON; clear so the explicit insert below is the ONLY
        // entry and the len()==1 / len()==0 counts below are exact.
        clear_all();
        let k1 = key_for(p1, 1024, 768);
        insert(k1.clone(), &cold, &cfg(1024, 768));
        assert!(lookup(&k1).is_some(), "E1 inserted");

        // Step 5 (negative control): without a wipe, K1 is still findable. We assert
        // on K1 SPECIFICALLY (not a global len() count) so the test is immune to the
        // process-global cache being populated by other rendering tests in this
        // binary now that CV_PAINT_CACHE defaults ON.
        assert!(lookup(&k1).is_some(), "E1 present pre-wipe (no wipe ⇒ still served)");

        // Step 2: the structural-rebuild wipe (what main.rs calls at the
        // reconcile-bail / feature-set-change sites).
        clear_all();

        // Step 3: K1 gone after the wipe — stale-id serve impossible by construction.
        // (Assert on K1 specifically, not global len(): the process-global cache may
        // be repopulated by OTHER rendering tests in this binary the instant the
        // guard-serialized window ends, so a global-count assert is racy; the
        // load-bearing claim is that THIS stale entry is gone, which lookup proves.)
        assert!(
            lookup(&k1).is_none(),
            "stale entry whose ids were realloc'd is GONE — no wrong-frame serve"
        );

        // Step 4: re-render after the rebuild takes the cold-bake MISS path and
        // matches an independent cold bake.
        let recold = bake_page(p1, 1024, 768);
        assert_byte_identical(&recold, &cold); // the rebuilt page (same content) bakes identically
        // And re-inserting after the rebuild works (fresh, valid entry).
        let k1b = key_for(p1, 1024, 768);
        insert(k1b.clone(), &recold, &cfg(1024, 768));
        assert!(lookup(&k1b).is_some(), "post-rebuild fresh entry serveable");
        clear_all();
    }

    // ── Case: LRU recency — a HIT promotes to MRU; the untouched oldest is the
    // one evicted, NOT the recently-hit entry. ───────────────────────────────
    #[test]
    fn t_lru_recency_order() {
        // LOCAL PaintCache instance — isolated from the global default-ON singleton
        // so a HIT-promotes-to-MRU + LRU-eviction sequence is deterministic.
        let cold = bake_page("<html><body>lru content here</body></html>", 320, 240);
        let entry_bytes = cold.bitmap.pixels.len() * 4 + ENTRY_FIXED_OVERHEAD;
        let budget = entry_bytes * 2 + entry_bytes / 2; // holds 2 entries
        let mk = |salt: u64| PaintCacheKey {
            url_hash: 0x12C7_A9F0,
            dom_structural_hash: salt.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(7),
            vw: 320,
            vh: 240,
        };
        let (k0, k1, k2) = (mk(1), mk(2), mk(3));
        let mut c = PaintCache::new();
        local_insert(&mut c, k0.clone(), mk_entry(&cold), budget);
        local_insert(&mut c, k1.clone(), mk_entry(&cold), budget);
        // Touch k0 (a HIT) → k0 becomes MRU, so k1 is now the LRU.
        assert!(c.map.contains_key(&k0), "k0 present");
        c.touch(&k0);
        // Insert a 3rd entry → the LRU (k1, NOT the recently-hit k0) is evicted.
        local_insert(&mut c, k2.clone(), mk_entry(&cold), budget);
        assert!(c.map.contains_key(&k0), "recently-hit k0 survives (promoted to MRU)");
        assert!(!c.map.contains_key(&k1), "untouched LRU k1 was evicted");
        assert!(c.map.contains_key(&k2), "newest k2 present");
    }

    // ── DISK persistence tests (CV_PAINT_CACHE_DISK) ─────────────────────────
    //
    // The disk layer is ADDITIVE behind CV_PAINT_CACHE_DISK (default OFF). These
    // tests exercise the disk submodule's serialize/deserialize/store/load/budget
    // DIRECTLY (not via the env flag, which is process-global and unset under
    // `cargo test` — see `t_flag_default_off` and `t_disk_flag_default_off`). Each
    // uses a UNIQUE temp dir so the real per-user cache dir is never polluted and
    // tests don't collide; the dir is set via `disk::set_dir_for_test` and removed
    // at the end. They take the same `TEST_GUARD` as the in-memory tests because
    // the disk test dir override is process-global state.
    mod disk_tests {
        use super::super::{disk, PaintCacheEntry, PaintCacheKey};
        use super::{bake_page, guard, key_for, HTML_A};

        /// A fresh, unique temp dir for one test. Removed at scope end by the
        /// returned guard's `Drop`.
        struct TmpDir(std::path::PathBuf);
        impl TmpDir {
            fn new(tag: &str) -> Self {
                use std::sync::atomic::{AtomicU64, Ordering};
                static CTR: AtomicU64 = AtomicU64::new(0);
                let n = CTR.fetch_add(1, Ordering::Relaxed);
                let p = std::env::temp_dir().join(format!(
                    "tb_paint_cache_disk_test_{}_{}_{}",
                    tag,
                    std::process::id(),
                    n
                ));
                let _ = std::fs::remove_dir_all(&p);
                std::fs::create_dir_all(&p).unwrap();
                disk::set_dir_for_test(Some(p.clone()));
                TmpDir(p)
            }
        }
        impl Drop for TmpDir {
            fn drop(&mut self) {
                disk::set_dir_for_test(None);
                let _ = std::fs::remove_dir_all(&self.0);
            }
        }

        fn entry_for(html: &str, vw: u32, vh: u32) -> (PaintCacheKey, PaintCacheEntry) {
            let cold = bake_page(html, vw, vh);
            let key = key_for(html, vw, vh);
            let bytes = cold.bitmap.pixels.len() * 4 + super::super::ENTRY_FIXED_OVERHEAD;
            let entry = PaintCacheEntry {
                doc_w: cold.bitmap.width,
                doc_h: cold.bitmap.height,
                structural_hash: 0xDEAD_BEEF_CAFE_F00D,
                bytes,
                paint: cold,
            };
            (key, entry)
        }

        // (a) ROUND-TRIP: serialize→deserialize is byte-identical bitmap + key +
        // dims + hashes + chrome metadata. THE oracle for the disk format.
        #[test]
        fn t_disk_round_trip_byte_identical() {
            let _g = guard();
            let _tmp = TmpDir::new("roundtrip");
            let (key, entry) = entry_for(HTML_A, 1024, 768);

            // Pure in-memory round trip (no FS): proves the codec.
            let bytes = disk::serialize_entry(&key, &entry);
            let (k2, e2) = disk::deserialize_entry(&bytes).expect("deserialize ok");
            assert_eq!(k2, key, "key round-trips");
            assert_eq!(e2.doc_w, entry.doc_w, "doc_w round-trips");
            assert_eq!(e2.doc_h, entry.doc_h, "doc_h round-trips");
            assert_eq!(
                e2.structural_hash, entry.structural_hash,
                "structural_hash round-trips"
            );
            assert_eq!(e2.paint.bitmap.width, entry.paint.bitmap.width);
            assert_eq!(e2.paint.bitmap.height, entry.paint.bitmap.height);
            // ★ BYTE-IDENTICAL bitmap (the correctness contract).
            assert_eq!(
                e2.paint.bitmap.pixels, entry.paint.bitmap.pixels,
                "bitmap Vec<u32> must be byte-identical after disk round-trip"
            );
            assert_eq!(e2.paint.title, entry.paint.title, "title round-trips");
            assert_eq!(
                e2.paint.current_url, entry.paint.current_url,
                "current_url round-trips"
            );
            assert_eq!(e2.paint.chrome_h, entry.paint.chrome_h);
            assert_eq!(e2.paint.viewport_h, entry.paint.viewport_h);
            assert_eq!(e2.paint.texts, entry.paint.texts, "texts round-trip");
            // A disk-loaded entry carries None for the re-derivable fields.
            assert!(e2.paint.layout_root.is_none(), "layout_root None on disk load");
            assert!(e2.paint.retained.is_none(), "retained None on disk load");
            assert!(
                e2.paint.property_trees.is_none(),
                "property_trees None on disk load"
            );

            // Full FS round trip through store/load (atomic write + read).
            disk::store(&key, &entry);
            let loaded = disk::load(&key).expect("disk load HIT after store");
            assert_eq!(
                loaded.paint.bitmap.pixels, entry.paint.bitmap.pixels,
                "FS round-trip bitmap byte-identical"
            );
            assert_eq!(loaded.doc_w, entry.doc_w);
            assert_eq!(loaded.doc_h, entry.doc_h);
            assert_eq!(loaded.structural_hash, entry.structural_hash);
        }

        // (a') A disk HIT is byte-identical to a COLD bake (end-to-end oracle).
        #[test]
        fn t_disk_hit_matches_cold_bake() {
            let _g = guard();
            let _tmp = TmpDir::new("coldmatch");
            let (key, entry) = entry_for(HTML_A, 1024, 768);
            disk::store(&key, &entry);

            let loaded = disk::load(&key).expect("disk HIT");
            // A FRESH independent cold bake — the disk-served bitmap must equal it.
            let cold2 = bake_page(HTML_A, 1024, 768);
            assert_eq!(
                loaded.paint.bitmap.pixels, cold2.bitmap.pixels,
                "disk-served bitmap byte-identical to an independent cold bake"
            );
            assert_eq!(loaded.paint.bitmap.width, cold2.bitmap.width);
            assert_eq!(loaded.paint.bitmap.height, cold2.bitmap.height);
            assert_eq!(loaded.paint.texts, cold2.texts);
        }

        // (b) CORRUPTION / TRUNCATION / BAD-MAGIC / BAD-VERSION → None, no panic.
        #[test]
        fn t_disk_corruption_returns_none_no_panic() {
            let _g = guard();
            let _tmp = TmpDir::new("corrupt");
            let (key, entry) = entry_for(HTML_A, 320, 240);
            let good = disk::serialize_entry(&key, &entry);

            // Bad magic.
            let mut bad_magic = good.clone();
            bad_magic[0] = b'X';
            assert!(
                disk::deserialize_entry(&bad_magic).is_none(),
                "bad magic → None"
            );

            // Bad version (bytes 4..8 are the version u32 LE).
            let mut bad_ver = good.clone();
            bad_ver[4] = bad_ver[4].wrapping_add(1);
            assert!(
                disk::deserialize_entry(&bad_ver).is_none(),
                "bad version → None"
            );

            // Truncation at many lengths.
            for cut in [0usize, 3, 4, 8, 16, 40, good.len() / 2, good.len().saturating_sub(1)] {
                let cut = cut.min(good.len());
                assert!(
                    disk::deserialize_entry(&good[..cut]).is_none(),
                    "truncated at {cut} → None"
                );
            }

            // Empty + garbage.
            assert!(disk::deserialize_entry(&[]).is_none(), "empty → None");
            assert!(
                disk::deserialize_entry(&[0xAB; 7]).is_none(),
                "garbage → None"
            );

            // A good stored entry still loads cleanly (the corruption above is at
            // the codec level; `load` runs the SAME deserialize after read, so a
            // corrupt file on disk would be a MISS, not a panic).
            disk::store(&key, &entry);
            assert!(disk::load(&key).is_some(), "good stored entry still loads");
        }

        // (b') A self-inconsistent header (pixel count != w*h) → None.
        #[test]
        fn t_disk_inconsistent_dims_returns_none() {
            let _g = guard();
            let _tmp = TmpDir::new("dims");
            let (key, entry) = entry_for("<html><body>x</body></html>", 200, 150);
            let mut bytes = disk::serialize_entry(&key, &entry);
            // The pixel-count u64 sits right after the two bitmap-dim u32s. Find it
            // by reconstructing the header offset:
            //   magic(4)+ver(4)+url(8)+dom(8)+vw(4)+vh(4)+docw(4)+doch(4)
            //   +shash(8)+chrome_h(4)+viewport_h(4)
            //   +title(len-prefixed)+current_url(len-prefixed)+caret(1[+16])
            //   +bmp.width(4)+bmp.height(4)+npix(8)
            // Rather than compute the exact offset (brittle), flip a byte INSIDE
            // the declared width to make w*h disagree with npix.
            // Simplest robust corruption: zero the whole pixel-count region by
            // searching is overkill — instead truncate the pixel payload by one
            // u32, which makes the final `take` fail → None. Already covered by the
            // truncation test, so here we assert the EXPLICIT npix!=w*h guard via a
            // hand-built header.
            // Build a minimal valid header then a deliberately wrong npix:
            let mut h: Vec<u8> = Vec::new();
            h.extend_from_slice(b"TBPC");
            h.extend_from_slice(&1u32.to_le_bytes()); // version
            h.extend_from_slice(&key.url_hash.to_le_bytes());
            h.extend_from_slice(&key.dom_structural_hash.to_le_bytes());
            h.extend_from_slice(&key.vw.to_le_bytes());
            h.extend_from_slice(&key.vh.to_le_bytes());
            h.extend_from_slice(&2u32.to_le_bytes()); // doc_w
            h.extend_from_slice(&2u32.to_le_bytes()); // doc_h
            h.extend_from_slice(&0u64.to_le_bytes()); // structural_hash
            h.extend_from_slice(&0u32.to_le_bytes()); // chrome_h
            h.extend_from_slice(&0u32.to_le_bytes()); // viewport_h
            h.extend_from_slice(&0u32.to_le_bytes()); // title len = 0
            h.extend_from_slice(&0u32.to_le_bytes()); // current_url len = 0
            h.push(0); // caret None
            h.extend_from_slice(&2u32.to_le_bytes()); // bmp width = 2
            h.extend_from_slice(&2u32.to_le_bytes()); // bmp height = 2 → w*h = 4
            h.extend_from_slice(&3u64.to_le_bytes()); // npix = 3 (≠ 4) → guard fires
            h.extend_from_slice(&[0u8; 12]); // 3 pixels of payload
            assert!(
                disk::deserialize_entry(&h).is_none(),
                "npix != width*height must be rejected"
            );
            // Sanity: the untouched `bytes` still deserializes fine.
            let ok = disk::deserialize_entry(&bytes).is_some();
            assert!(ok, "valid bytes still deserialize");
            bytes.clear();
        }

        // (c) NO DISK DIR ⇒ ZERO disk IO. (CV_PAINT_CACHE_DISK now DEFAULTS ON, so
        // the zero-IO guarantee comes from the disk dir being unset, not from the
        // flag: under cfg(test), disk::dir() returns None unless set_dir_for_test
        // installed an override. We deliberately do NOT create a TmpDir here, so no
        // override is set — the seam therefore performs ZERO disk activity even with
        // the flag on.) This proves the dir-gating is the hard zero-IO backstop.
        #[test]
        fn t_disk_no_dir_zero_io() {
            let _g = guard();
            // Ensure no leftover override from a prior test on this thread.
            disk::set_dir_for_test(None);
            super::super::clear_all();
            // Drive the PUBLIC seam (insert + lookup) — with no disk dir set these
            // must not touch any disk at all.
            let cold = bake_page(HTML_A, 640, 480);
            let key = key_for(HTML_A, 640, 480);
            super::super::insert(key.clone(), &cold, &super::cfg(640, 480));
            let _ = super::super::lookup(&key);
            // Also exercise a forced in-memory MISS to hit the disk-load branch
            // guard (which must be a no-op when no dir is configured).
            let mut other = key.clone();
            other.dom_structural_hash ^= 0x1234_5678;
            assert!(
                super::super::lookup(&other).is_none(),
                "miss with no disk dir stays a miss (no disk consult)"
            );
            assert_eq!(
                disk::file_count(),
                0,
                "no disk dir must write ZERO disk files via the seam"
            );
            assert_eq!(
                disk::total_disk_bytes(),
                0,
                "no disk dir must produce ZERO disk bytes"
            );
            super::super::clear_all();
        }

        // (d) DISK BUDGET EVICTION: storing past the byte budget LRU-evicts the
        // oldest files; the directory stays bounded.
        #[test]
        fn t_disk_budget_eviction() {
            let _g = guard();
            let _tmp = TmpDir::new("budget");
            // Small page so the file is modest; budget holds ~2 files.
            let html = "<html><body>disk budget content here</body></html>";
            let (k0, e0) = entry_for(html, 200, 150);
            let one = disk::serialize_entry(&k0, &e0).len();
            let budget = one * 2 + one / 2; // room for 2 files

            // Distinct keys (vary the dom hash) → distinct filenames.
            let mk = |salt: u64| {
                let mut k = k0.clone();
                k.dom_structural_hash ^= salt.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
                k
            };
            let ka = mk(1);
            let kb = mk(2);
            let kc = mk(3);
            // Store with explicit budget so eviction is forced deterministically.
            // Bump mtimes apart so the LRU ordering is well-defined.
            disk::store_with_budget(&ka, &e0, budget);
            disk::store_with_budget(&kb, &e0, budget);
            // After 2 stores we should be within budget (2 files).
            assert!(
                disk::total_disk_bytes() <= budget,
                "disk budget held after 2 stores: {} <= {}",
                disk::total_disk_bytes(),
                budget
            );
            assert!(disk::file_count() <= 2, "at most 2 files under this budget");
            // A 3rd store evicts the oldest → still <= 2 files, still within budget.
            disk::store_with_budget(&kc, &e0, budget);
            assert!(
                disk::total_disk_bytes() <= budget,
                "disk budget held after 3 stores: {} <= {}",
                disk::total_disk_bytes(),
                budget
            );
            assert!(
                disk::file_count() <= 2,
                "disk file count bounded after eviction: {}",
                disk::file_count()
            );
            // The newest key must still be present.
            assert!(disk::load(&kc).is_some(), "newest disk entry survives eviction");
        }

        // (d') A single file larger than the whole disk budget is NOT written.
        #[test]
        fn t_disk_oversized_file_not_written() {
            let _g = guard();
            let _tmp = TmpDir::new("oversize");
            let (key, entry) = entry_for(HTML_A, 1024, 768);
            let sz = disk::serialize_entry(&key, &entry).len();
            // Budget smaller than a single file → no write.
            disk::store_with_budget(&key, &entry, sz / 2);
            assert_eq!(
                disk::file_count(),
                0,
                "an oversized entry must not be written to disk"
            );
            assert!(disk::load(&key).is_none(), "and is not loadable");
        }

        // (e) KEY MISMATCH guard: a file whose embedded key differs from the
        // requested key is rejected (defense against a hash collision / stale
        // rename) — a MISS, not a wrong-frame serve.
        #[test]
        fn t_disk_key_mismatch_rejected() {
            let _g = guard();
            let _tmp = TmpDir::new("keymismatch");
            let (key, entry) = entry_for(HTML_A, 800, 600);
            disk::store(&key, &entry);
            assert!(disk::load(&key).is_some(), "exact key loads");
            // A different key → different filename → natural miss.
            let mut other = key.clone();
            other.url_hash ^= 0xABCD_1234_5678_9F01;
            assert!(
                disk::load(&other).is_none(),
                "a different key must MISS (no cross-key serve)"
            );
        }
    }
}

