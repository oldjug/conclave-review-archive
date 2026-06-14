//! `cv_dom` — WHATWG DOM, real implementation.
//!
//! Self-contained DOM tree built on stable handles into an arena.
//! Every node has an `NodeId`; parent / sibling / child links are
//! stored in the arena so JS-driven mutations don't invalidate
//! pointers held by other parts of the engine.

#![allow(missing_debug_implementations)]

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

pub mod invalidation;
pub mod mutation;
pub mod range;
pub mod traversal;

pub use invalidation::StageMask;
pub use mutation::{MutationObserver, MutationRecord, MutationType};
pub use range::Range;
pub use traversal::TreeWalker;

/// Stable, generational handle to a node in the [`Document`] arena. Carries a
/// per-slot generation (index + generation) so a freed-then-reused slot yields a
/// DIFFERENT `NodeId` — anything keyed on `NodeId` (the style/layout caches)
/// therefore auto-invalidates that one entry on reuse, finer-grained than the
/// old global generation bump. `Copy`, 8 bytes, `Eq`/`Hash`/`Debug`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeId(cv_arena::Handle<NodeRecord>);

impl NodeId {
    /// Pack the full identity (index + generation) into a `u64` for the
    /// cross-crate stash (e.g. a `u64` slot on a `cv_layout::LayoutBox`). Never
    /// zero. Round-trips through [`Self::from_bits`].
    pub fn to_bits(self) -> u64 {
        self.0.to_bits()
    }
    /// Reconstruct a `NodeId` from [`Self::to_bits`]. `None` if the packed
    /// generation is zero (an invalid / never-issued handle).
    pub fn from_bits(bits: u64) -> Option<NodeId> {
        cv_arena::Handle::from_bits(bits).map(NodeId)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NodeKind {
    Document,
    DocumentType {
        name: String,
        public_id: String,
        system_id: String,
    },
    DocumentFragment,
    Element {
        tag: String,
        namespace: Option<String>,
    },
    Text(String),
    Comment(String),
    ProcessingInstruction {
        target: String,
        data: String,
    },
    CDataSection(String),
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct AttrMap {
    pairs: Vec<(String, String)>,
}

impl AttrMap {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn get(&self, name: &str) -> Option<&str> {
        self.pairs
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case(name))
            .map(|(_, v)| v.as_str())
    }
    pub fn set(&mut self, name: &str, value: &str) -> Option<String> {
        for (k, v) in self.pairs.iter_mut() {
            if k.eq_ignore_ascii_case(name) {
                let old = std::mem::replace(v, value.to_string());
                return Some(old);
            }
        }
        self.pairs.push((name.to_string(), value.to_string()));
        None
    }
    pub fn remove(&mut self, name: &str) -> Option<String> {
        let pos = self
            .pairs
            .iter()
            .position(|(k, _)| k.eq_ignore_ascii_case(name))?;
        Some(self.pairs.remove(pos).1)
    }
    pub fn has(&self, name: &str) -> bool {
        self.pairs.iter().any(|(k, _)| k.eq_ignore_ascii_case(name))
    }
    pub fn names(&self) -> Vec<String> {
        self.pairs.iter().map(|(k, _)| k.clone()).collect()
    }
    pub fn len(&self) -> usize {
        self.pairs.len()
    }
    pub fn is_empty(&self) -> bool {
        self.pairs.is_empty()
    }
}

#[derive(Debug)]
struct NodeRecord {
    kind: NodeKind,
    attrs: AttrMap,
    parent: Option<NodeId>,
    children: Vec<NodeId>,
    element_id_cache: Option<String>,
    /// Blink-style typed invalidation lattice (M2.1): one byte holding every
    /// render-pipeline stage's SELF + DESCENDANT dirty bits. The STYLE bits
    /// (`NEEDS_STYLE` / `CHILD_NEEDS_STYLE`) carry the cascade's incremental
    /// state — exactly what the two old `style_dirty` / `child_styles_dirty`
    /// bools did, byte-for-byte; the public style methods delegate to them. The
    /// LAYOUT / PAINT bits are new capability with no consumer yet (M2.2–M2.4).
    /// New nodes start dirty in all three stages.
    stage: StageMask,
}

#[derive(Debug, Default)]
pub struct Document {
    /// Generational arena: every node lives in a stable slot addressed by a
    /// `NodeId` (= `Handle<NodeRecord>`). A freed slot's next occupant gets a
    /// bumped generation, so a stale `NodeId` never aliases the new node and
    /// `slab.get` rejects it — per-slot identity replaces the old global
    /// generation counter for cache invalidation.
    slab: cv_arena::Slab<NodeRecord>,
    root: Option<NodeId>,
    id_index: HashMap<String, NodeId>,
    observers: Vec<Rc<RefCell<MutationObserver>>>,
}

impl Document {
    pub fn new() -> Self {
        let mut d = Self::default();
        let root = d.alloc(NodeKind::Document);
        d.root = Some(root);
        d
    }

    pub fn root(&self) -> NodeId {
        self.root.expect("Document always has a root")
    }

    fn alloc(&mut self, kind: NodeKind) -> NodeId {
        // The slab reuses a freed slot if one is available (bumping that slot's
        // generation so the returned NodeId differs from the old occupant's),
        // else appends a fresh slot. New nodes start dirty in every stage: the
        // STYLE SELF+DESCENDANT bits reproduce the old `style_dirty:true` /
        // `child_styles_dirty:true` start state, and LAYOUT/PAINT are dirty-on-
        // create too (correct, currently unobserved).
        NodeId(self.slab.insert(NodeRecord {
            kind,
            attrs: AttrMap::new(),
            parent: None,
            children: Vec::new(),
            element_id_cache: None,
            stage: StageMask::from_bits(StageMask::SELF_ANY | StageMask::CHILD_ANY),
        }))
    }

    /// Cache generation, retained for the style/layout cache's `cache_gen` guard.
    /// Per-slot generations now live inside `NodeId` itself — a freed-then-reused
    /// slot yields a DIFFERENT `NodeId`, so a cache keyed on `NodeId` auto-misses
    /// the stale entry (finer-grained than a global bump). This therefore returns
    /// a constant: the per-`NodeId` generation is the sole invalidation source.
    pub fn generation(&self) -> u32 {
        0
    }

    // ----------------------------------------------------------------------
    // Typed invalidation lattice (M2.1).
    //
    // `mark` is the single monotone-propagation primitive: it sets a SELF stage
    // bit on `id` and walks the ancestor chain setting the matching DESCENDANT
    // bit, stopping at the first ancestor that already carries it (so repeated
    // marks on an already-flagged subtree are O(1)). Every stage — style now,
    // layout/paint in M2.2–M2.4 — shares this one walk.
    // ----------------------------------------------------------------------

    /// Read a node's full [`StageMask`]. Missing/stale node → an all-dirty mask
    /// (conservative: a stage walk will visit it). Mostly for the lattice
    /// consumers and tests.
    pub fn stage_mask(&self, id: NodeId) -> StageMask {
        self.rec(id)
            .map(|r| r.stage)
            .unwrap_or_else(|| StageMask::from_bits(StageMask::SELF_ANY | StageMask::CHILD_ANY))
    }

    /// Generic monotone marker: set `self_bit` on `id` and propagate the matching
    /// CHILD_ bit up the ancestor chain, early-exiting once it's already set.
    /// `self_bit` must be one of `NEEDS_STYLE` / `NEEDS_LAYOUT` / `NEEDS_PAINT`.
    pub fn mark(&mut self, id: NodeId, self_bit: u8) {
        match self.rec_mut(id) {
            Some(r) => {
                if r.stage.is_set(self_bit) {
                    return; // already dirty for this stage (ancestors flagged)
                }
                r.stage.insert(self_bit);
            }
            None => return,
        }
        let child_bit = StageMask::child_bit_for(self_bit);
        let mut cur = self.parent(id);
        while let Some(p) = cur {
            match self.rec_mut(p) {
                Some(r) if !r.stage.is_set(child_bit) => r.stage.insert(child_bit),
                _ => break, // already flagged up to here, or gone
            }
            cur = self.parent(p);
        }
    }

    /// Clear only `self_bit`'s SELF bit on `id` (the DESCENDANT bits and other
    /// stages are untouched). `self_bit` is one of the `NEEDS_*` consts.
    pub fn clear_stage(&mut self, id: NodeId, self_bit: u8) {
        if let Some(r) = self.rec_mut(id) {
            r.stage.remove(self_bit);
        }
    }

    // Per-stage convenience markers (all delegate to `mark`).
    /// Mark `id` needing style recompute, propagating CHILD_NEEDS_STYLE up.
    pub fn mark_needs_style(&mut self, id: NodeId) {
        self.mark(id, StageMask::NEEDS_STYLE);
    }
    /// Mark `id` needing layout, propagating CHILD_NEEDS_LAYOUT up.
    pub fn mark_needs_layout(&mut self, id: NodeId) {
        self.mark(id, StageMask::NEEDS_LAYOUT);
    }
    /// Mark `id` needing paint, propagating CHILD_NEEDS_PAINT up.
    pub fn mark_needs_paint(&mut self, id: NodeId) {
        self.mark(id, StageMask::NEEDS_PAINT);
    }

    // Per-stage SELF queries. Missing/stale node → dirty (conservative).
    /// True if `id` needs its style recomputed this frame.
    pub fn needs_style(&self, id: NodeId) -> bool {
        self.rec(id)
            .map(|r| r.stage.is_set(StageMask::NEEDS_STYLE))
            .unwrap_or(true)
    }
    /// True if `id` needs its layout recomputed.
    pub fn needs_layout(&self, id: NodeId) -> bool {
        self.rec(id)
            .map(|r| r.stage.is_set(StageMask::NEEDS_LAYOUT))
            .unwrap_or(true)
    }
    /// True if `id` needs to be repainted.
    pub fn needs_paint(&self, id: NodeId) -> bool {
        self.rec(id)
            .map(|r| r.stage.is_set(StageMask::NEEDS_PAINT))
            .unwrap_or(true)
    }

    // Per-stage DESCENDANT queries. Missing/stale node → dirty (conservative).
    /// True if some descendant of `id` needs its style recomputed.
    pub fn child_needs_style(&self, id: NodeId) -> bool {
        self.rec(id)
            .map(|r| r.stage.is_set(StageMask::CHILD_NEEDS_STYLE))
            .unwrap_or(true)
    }
    /// True if some descendant of `id` needs its layout recomputed.
    pub fn child_needs_layout(&self, id: NodeId) -> bool {
        self.rec(id)
            .map(|r| r.stage.is_set(StageMask::CHILD_NEEDS_LAYOUT))
            .unwrap_or(true)
    }
    /// True if some descendant of `id` needs to be repainted.
    pub fn child_needs_paint(&self, id: NodeId) -> bool {
        self.rec(id)
            .map(|r| r.stage.is_set(StageMask::CHILD_NEEDS_PAINT))
            .unwrap_or(true)
    }

    // ----------------------------------------------------------------------
    // Style-stage public API — preserved byte-for-byte, delegating to the
    // STYLE bits of the lattice. The conclave cascade calls these and MUST
    // see identical behaviour to the old two-bool design.
    // ----------------------------------------------------------------------

    /// True if `id`'s style must be recomputed this frame.
    pub fn style_dirty(&self, id: NodeId) -> bool {
        self.needs_style(id)
    }

    /// True if any descendant of `id` is style-dirty.
    pub fn child_styles_dirty(&self, id: NodeId) -> bool {
        self.child_needs_style(id)
    }

    /// Mark `id` style-dirty and propagate `child_styles_dirty` up the ancestor
    /// chain so the cascade walk can reach it from the root.
    pub fn mark_style_dirty(&mut self, id: NodeId) {
        self.mark(id, StageMask::NEEDS_STYLE);
    }

    /// Clear `id`'s own dirty bit (called after the cascade recomputes it).
    pub fn clear_style_dirty(&mut self, id: NodeId) {
        self.clear_stage(id, StageMask::NEEDS_STYLE);
    }

    /// Mark every live node style-dirty (used on a full rebuild / sheet change).
    pub fn mark_all_style_dirty(&mut self) {
        for r in self.slab.values_mut() {
            r.stage.insert(StageMask::NEEDS_STYLE);
            r.stage.insert(StageMask::CHILD_NEEDS_STYLE);
        }
    }

    /// Clear every live node's STYLE dirty bits — the per-frame reset before the
    /// invalidator marks this frame's actual changes. O(nodes) bit writes, far
    /// cheaper than re-cascading. (Layout/paint bits are untouched.)
    pub fn clear_all_style_dirty(&mut self) {
        for r in self.slab.values_mut() {
            r.stage.remove(StageMask::NEEDS_STYLE);
            r.stage.remove(StageMask::CHILD_NEEDS_STYLE);
        }
    }

    /// Mark `id` and its entire descendant subtree style-dirty (the coarse
    /// invalidation tier: `whole_subtree` sets, attribute/text changes we don't
    /// model precisely, etc.). Propagates `child_styles_dirty` up from `id`.
    pub fn mark_style_dirty_subtree(&mut self, id: NodeId) {
        let mut stack = vec![id];
        while let Some(n) = stack.pop() {
            if let Some(r) = self.rec_mut(n) {
                r.stage.insert(StageMask::NEEDS_STYLE);
                r.stage.insert(StageMask::CHILD_NEEDS_STYLE);
                stack.extend(r.children.iter().copied());
            }
        }
        // Reach the subtree from the root: flag the ancestor chain above `id`.
        let mut cur = self.parent(id);
        while let Some(p) = cur {
            match self.rec_mut(p) {
                Some(r) if !r.stage.is_set(StageMask::CHILD_NEEDS_STYLE) => {
                    r.stage.insert(StageMask::CHILD_NEEDS_STYLE)
                }
                _ => break,
            }
            cur = self.parent(p);
        }
    }

    fn rec(&self, id: NodeId) -> Option<&NodeRecord> {
        // `slab.get` returns `None` for a stale handle (slot removed/reused) or
        // an out-of-range index — subsuming the old `live` bool check.
        self.slab.get(id.0)
    }
    fn rec_mut(&mut self, id: NodeId) -> Option<&mut NodeRecord> {
        self.slab.get_mut(id.0)
    }

    pub fn create_element(&mut self, tag: &str) -> NodeId {
        self.alloc(NodeKind::Element {
            tag: tag.to_string(),
            namespace: None,
        })
    }
    pub fn create_element_ns(&mut self, ns: &str, tag: &str) -> NodeId {
        self.alloc(NodeKind::Element {
            tag: tag.to_string(),
            namespace: Some(ns.to_string()),
        })
    }
    pub fn create_text_node(&mut self, data: &str) -> NodeId {
        self.alloc(NodeKind::Text(data.to_string()))
    }
    pub fn create_comment(&mut self, data: &str) -> NodeId {
        self.alloc(NodeKind::Comment(data.to_string()))
    }
    pub fn create_document_fragment(&mut self) -> NodeId {
        self.alloc(NodeKind::DocumentFragment)
    }

    pub fn kind(&self, id: NodeId) -> Option<&NodeKind> {
        Some(&self.rec(id)?.kind)
    }
    pub fn parent(&self, id: NodeId) -> Option<NodeId> {
        self.rec(id)?.parent
    }
    pub fn children(&self, id: NodeId) -> Vec<NodeId> {
        self.rec(id).map(|r| r.children.clone()).unwrap_or_default()
    }
    pub fn first_child(&self, id: NodeId) -> Option<NodeId> {
        self.rec(id)?.children.first().copied()
    }
    pub fn last_child(&self, id: NodeId) -> Option<NodeId> {
        self.rec(id)?.children.last().copied()
    }
    pub fn next_sibling(&self, id: NodeId) -> Option<NodeId> {
        let p = self.parent(id)?;
        let kids = self.children(p);
        let i = kids.iter().position(|&c| c == id)?;
        kids.get(i + 1).copied()
    }
    pub fn previous_sibling(&self, id: NodeId) -> Option<NodeId> {
        let p = self.parent(id)?;
        let kids = self.children(p);
        let i = kids.iter().position(|&c| c == id)?;
        if i == 0 {
            None
        } else {
            kids.get(i - 1).copied()
        }
    }

    pub fn append_child(&mut self, parent: NodeId, child: NodeId) -> Result<NodeId, DomError> {
        self.validate_insert(parent, child)?;
        self.detach(child);
        let prev = self.rec(parent).and_then(|r| r.children.last().copied());
        self.rec_mut(parent)
            .ok_or(DomError::Stale)?
            .children
            .push(child);
        self.rec_mut(child).ok_or(DomError::Stale)?.parent = Some(parent);
        self.maybe_register_id(child);
        self.emit_mutation(MutationRecord {
            target: parent,
            kind: MutationType::ChildListAdded {
                added: vec![child],
                removed: Vec::new(),
                previous_sibling: prev,
                next_sibling: None,
            },
        });
        Ok(child)
    }

    pub fn insert_before(
        &mut self,
        parent: NodeId,
        new_node: NodeId,
        ref_node: Option<NodeId>,
    ) -> Result<NodeId, DomError> {
        self.validate_insert(parent, new_node)?;
        let ref_idx = match ref_node {
            Some(r) => {
                let kids = self.children(parent);
                kids.iter()
                    .position(|&c| c == r)
                    .ok_or(DomError::NotFound)?
            }
            None => return self.append_child(parent, new_node),
        };
        self.detach(new_node);
        let prev = if ref_idx == 0 {
            None
        } else {
            self.rec(parent)
                .and_then(|r| r.children.get(ref_idx - 1).copied())
        };
        self.rec_mut(parent)
            .ok_or(DomError::Stale)?
            .children
            .insert(ref_idx, new_node);
        self.rec_mut(new_node).ok_or(DomError::Stale)?.parent = Some(parent);
        self.maybe_register_id(new_node);
        self.emit_mutation(MutationRecord {
            target: parent,
            kind: MutationType::ChildListAdded {
                added: vec![new_node],
                removed: Vec::new(),
                previous_sibling: prev,
                next_sibling: ref_node,
            },
        });
        Ok(new_node)
    }

    pub fn remove_child(&mut self, parent: NodeId, child: NodeId) -> Result<NodeId, DomError> {
        let kids = self.children(parent);
        let idx = kids
            .iter()
            .position(|&c| c == child)
            .ok_or(DomError::NotFound)?;
        let prev = if idx == 0 {
            None
        } else {
            kids.get(idx - 1).copied()
        };
        let next = kids.get(idx + 1).copied();
        self.rec_mut(parent)
            .ok_or(DomError::Stale)?
            .children
            .remove(idx);
        self.rec_mut(child).ok_or(DomError::Stale)?.parent = None;
        self.unregister_id(child);
        self.emit_mutation(MutationRecord {
            target: parent,
            kind: MutationType::ChildListAdded {
                added: Vec::new(),
                removed: vec![child],
                previous_sibling: prev,
                next_sibling: next,
            },
        });
        Ok(child)
    }

    pub fn replace_child(
        &mut self,
        parent: NodeId,
        new_child: NodeId,
        old_child: NodeId,
    ) -> Result<NodeId, DomError> {
        let kids = self.children(parent);
        let idx = kids
            .iter()
            .position(|&c| c == old_child)
            .ok_or(DomError::NotFound)?;
        self.validate_insert(parent, new_child)?;
        self.detach(new_child);
        self.unregister_id(old_child);
        let prev = if idx == 0 {
            None
        } else {
            kids.get(idx - 1).copied()
        };
        let next = kids.get(idx + 1).copied();
        self.rec_mut(parent).ok_or(DomError::Stale)?.children[idx] = new_child;
        self.rec_mut(new_child).ok_or(DomError::Stale)?.parent = Some(parent);
        self.rec_mut(old_child).ok_or(DomError::Stale)?.parent = None;
        self.maybe_register_id(new_child);
        self.emit_mutation(MutationRecord {
            target: parent,
            kind: MutationType::ChildListAdded {
                added: vec![new_child],
                removed: vec![old_child],
                previous_sibling: prev,
                next_sibling: next,
            },
        });
        Ok(old_child)
    }

    fn detach(&mut self, id: NodeId) {
        let p = match self.parent(id) {
            Some(p) => p,
            None => return,
        };
        if let Some(rec) = self.rec_mut(p) {
            rec.children.retain(|&c| c != id);
        }
        if let Some(rec) = self.rec_mut(id) {
            rec.parent = None;
        }
        self.unregister_id(id);
    }

    fn validate_insert(&self, parent: NodeId, child: NodeId) -> Result<(), DomError> {
        if parent == child {
            return Err(DomError::HierarchyRequest);
        }
        let mut cur = self.parent(parent);
        while let Some(p) = cur {
            if p == child {
                return Err(DomError::HierarchyRequest);
            }
            cur = self.parent(p);
        }
        Ok(())
    }

    fn maybe_register_id(&mut self, id: NodeId) {
        let val = self
            .rec(id)
            .and_then(|r| r.attrs.get("id").map(String::from));
        if let Some(s) = val {
            if let Some(r) = self.rec_mut(id) {
                r.element_id_cache = Some(s.clone());
            }
            self.id_index.insert(s, id);
        }
    }
    fn unregister_id(&mut self, id: NodeId) {
        let cache = self.rec_mut(id).and_then(|r| r.element_id_cache.take());
        if let Some(v) = cache {
            if self.id_index.get(&v).copied() == Some(id) {
                self.id_index.remove(&v);
            }
        }
    }

    pub fn get_element_by_id(&self, id: &str) -> Option<NodeId> {
        self.id_index.get(id).copied()
    }

    /// Find the first element (document order) whose `attr` equals `value`.
    /// Linear scan. Used by the JS-wrapper↔NodeId bridge for the
    /// arena-source-of-truth migration: the engine stamps a stable identity
    /// attribute (`\u{1}nid`) on parse-time nodes, and a wrapper resolves its
    /// NodeId by looking that value up here. (Distinct from `get_element_by_id`,
    /// which indexes only the real `id` attribute.)
    pub fn get_by_attr(&self, attr: &str, value: &str) -> Option<NodeId> {
        let mut found: Option<NodeId> = None;
        self.walk_pre(self.root(), &mut |id, rec| {
            if found.is_none()
                && matches!(rec.kind, NodeKind::Element { .. })
                && rec.attrs.get(attr) == Some(value)
            {
                found = Some(id);
            }
        });
        found
    }

    pub fn get_attribute(&self, id: NodeId, name: &str) -> Option<String> {
        self.rec(id)?.attrs.get(name).map(String::from)
    }
    /// Borrowed attribute value (no allocation) — for hot selector-matching
    /// paths that need `&str` lifetimes rather than owned `String`s.
    pub fn attr_raw(&self, id: NodeId, name: &str) -> Option<&str> {
        self.rec(id)?.attrs.get(name)
    }
    /// Borrowed element tag exactly as stored (lowercase from the parser);
    /// `None` for non-element nodes. Selector matching is case-insensitive.
    pub fn tag_raw(&self, id: NodeId) -> Option<&str> {
        match &self.rec(id)?.kind {
            NodeKind::Element { tag, .. } => Some(tag.as_str()),
            _ => None,
        }
    }
    pub fn has_attribute(&self, id: NodeId, name: &str) -> bool {
        self.rec(id).map(|r| r.attrs.has(name)).unwrap_or(false)
    }
    pub fn set_attribute(&mut self, id: NodeId, name: &str, value: &str) {
        let old = self.rec_mut(id).and_then(|r| r.attrs.set(name, value));
        if name.eq_ignore_ascii_case("id") {
            if let Some(prev) = &old {
                if self.id_index.get(prev).copied() == Some(id) {
                    self.id_index.remove(prev);
                }
            }
            self.id_index.insert(value.to_string(), id);
            if let Some(r) = self.rec_mut(id) {
                r.element_id_cache = Some(value.to_string());
            }
        }
        self.emit_mutation(MutationRecord {
            target: id,
            kind: MutationType::Attributes {
                name: name.to_string(),
                old_value: old,
                new_value: Some(value.to_string()),
            },
        });
    }
    pub fn remove_attribute(&mut self, id: NodeId, name: &str) {
        let old = self.rec_mut(id).and_then(|r| r.attrs.remove(name));
        if name.eq_ignore_ascii_case("id") {
            if let Some(prev) = &old {
                if self.id_index.get(prev).copied() == Some(id) {
                    self.id_index.remove(prev);
                }
            }
            if let Some(r) = self.rec_mut(id) {
                r.element_id_cache = None;
            }
        }
        self.emit_mutation(MutationRecord {
            target: id,
            kind: MutationType::Attributes {
                name: name.to_string(),
                old_value: old,
                new_value: None,
            },
        });
    }
    pub fn attribute_names(&self, id: NodeId) -> Vec<String> {
        self.rec(id).map(|r| r.attrs.names()).unwrap_or_default()
    }

    pub fn text_content(&self, id: NodeId) -> String {
        let mut out = String::new();
        self.collect_text(id, &mut out);
        out
    }
    fn collect_text(&self, id: NodeId, out: &mut String) {
        let rec = match self.rec(id) {
            Some(r) => r,
            None => return,
        };
        match &rec.kind {
            NodeKind::Text(s) | NodeKind::CDataSection(s) => out.push_str(s),
            _ => {
                for &c in &rec.children {
                    self.collect_text(c, out);
                }
            }
        }
    }
    /// Update a Text / Comment / CDATA node's data in place. Returns
    /// the previous value for MutationRecord emission.
    pub fn set_text_data(&mut self, id: NodeId, data: &str) -> Result<String, DomError> {
        let rec = self.rec_mut(id).ok_or(DomError::Stale)?;
        let old = match &mut rec.kind {
            NodeKind::Text(s) => std::mem::replace(s, data.to_string()),
            NodeKind::Comment(s) => std::mem::replace(s, data.to_string()),
            NodeKind::CDataSection(s) => std::mem::replace(s, data.to_string()),
            _ => {
                return Err(DomError::InvalidArgument(
                    "not a character-data node".into(),
                ));
            }
        };
        self.emit_mutation(MutationRecord {
            target: id,
            kind: MutationType::CharacterData {
                old_value: old.clone(),
                new_value: data.to_string(),
            },
        });
        Ok(old)
    }

    pub fn set_text_content(&mut self, id: NodeId, text: &str) {
        for c in self.children(id) {
            let _ = self.remove_child(id, c);
        }
        if !text.is_empty() {
            let tn = self.create_text_node(text);
            let _ = self.append_child(id, tn);
        }
    }

    pub fn tag_name(&self, id: NodeId) -> Option<String> {
        match &self.rec(id)?.kind {
            NodeKind::Element { tag, .. } => Some(tag.to_ascii_uppercase()),
            _ => None,
        }
    }
    pub fn element_children(&self, id: NodeId) -> Vec<NodeId> {
        self.children(id)
            .into_iter()
            .filter(|c| matches!(self.kind(*c), Some(NodeKind::Element { .. })))
            .collect()
    }
    pub fn child_element_count(&self, id: NodeId) -> usize {
        self.element_children(id).len()
    }

    pub fn get_elements_by_tag_name(&self, tag: &str) -> Vec<NodeId> {
        let tag_lower = tag.to_ascii_lowercase();
        let all = tag == "*";
        let mut out = Vec::new();
        self.walk_pre(self.root(), &mut |id, rec| {
            if let NodeKind::Element { tag: t, .. } = &rec.kind {
                if all || t.eq_ignore_ascii_case(&tag_lower) {
                    out.push(id);
                }
            }
        });
        out
    }
    pub fn get_elements_by_class_name(&self, cls: &str) -> Vec<NodeId> {
        let needed: Vec<String> = cls.split_whitespace().map(String::from).collect();
        let mut out = Vec::new();
        self.walk_pre(self.root(), &mut |id, rec| {
            if matches!(rec.kind, NodeKind::Element { .. }) {
                if let Some(class_attr) = rec.attrs.get("class") {
                    let have: Vec<&str> = class_attr.split_whitespace().collect();
                    if needed.iter().all(|n| have.iter().any(|h| *h == n)) {
                        out.push(id);
                    }
                }
            }
        });
        out
    }

    fn walk_pre<F: FnMut(NodeId, &NodeRecord)>(&self, start: NodeId, f: &mut F) {
        let mut stack = vec![start];
        while let Some(id) = stack.pop() {
            if let Some(rec) = self.rec(id) {
                f(id, rec);
                for &c in rec.children.iter().rev() {
                    stack.push(c);
                }
            }
        }
    }

    fn emit_mutation(&mut self, rec: MutationRecord) {
        for obs in &self.observers {
            obs.borrow_mut().push(rec.clone());
        }
    }
    pub fn add_observer(&mut self, obs: Rc<RefCell<MutationObserver>>) {
        self.observers.push(obs);
    }
    pub fn live_node_count(&self) -> usize {
        self.slab.len()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DomError {
    NotFound,
    HierarchyRequest,
    Stale,
    InvalidArgument(String),
}

impl std::fmt::Display for DomError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotFound => f.write_str("NotFoundError"),
            Self::HierarchyRequest => f.write_str("HierarchyRequestError"),
            Self::Stale => f.write_str("InvalidStateError"),
            Self::InvalidArgument(s) => write!(f, "InvalidArgument: {s}"),
        }
    }
}

impl std::error::Error for DomError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn append_and_traverse() {
        let mut d = Document::new();
        let html = d.create_element("html");
        let body = d.create_element("body");
        let p = d.create_element("p");
        d.append_child(d.root(), html).unwrap();
        d.append_child(html, body).unwrap();
        d.append_child(body, p).unwrap();
        assert_eq!(d.children(body), vec![p]);
        assert_eq!(d.parent(p), Some(body));
        assert_eq!(d.tag_name(html).as_deref(), Some("HTML"));
    }

    #[test]
    fn id_indexed_on_setattribute() {
        let mut d = Document::new();
        let div = d.create_element("div");
        d.append_child(d.root(), div).unwrap();
        d.set_attribute(div, "id", "main");
        assert_eq!(d.get_element_by_id("main"), Some(div));
        d.set_attribute(div, "id", "renamed");
        assert!(d.get_element_by_id("main").is_none());
        d.remove_attribute(div, "id");
        assert!(d.get_element_by_id("renamed").is_none());
    }

    #[test]
    fn style_dirty_bits_mark_clear_and_propagate() {
        let mut d = Document::new();
        let a = d.create_element("div");
        let b = d.create_element("span");
        d.append_child(d.root(), a).unwrap();
        d.append_child(a, b).unwrap();
        // Fresh nodes start dirty.
        assert!(d.style_dirty(a) && d.style_dirty(b), "new nodes start style-dirty");
        // Clear, then re-mark a deeper node and confirm it propagates up.
        d.clear_style_dirty(a);
        d.clear_style_dirty(b);
        assert!(!d.style_dirty(b), "cleared");
        d.mark_style_dirty(b);
        assert!(d.style_dirty(b), "re-marked dirty");
        assert!(
            d.child_styles_dirty(a),
            "child_styles_dirty propagated to the parent"
        );
        // clear_style_dirty only clears the node's own bit, not child markers.
        d.clear_style_dirty(b);
        assert!(!d.style_dirty(b));
    }

    #[test]
    fn needs_layout_marks_self_and_all_ancestors() {
        // mark_needs_layout sets NEEDS_LAYOUT on the node and CHILD_NEEDS_LAYOUT
        // on every ancestor up to the root — and touches no other stage.
        let mut d = Document::new();
        let a = d.create_element("div");
        let b = d.create_element("span");
        let c = d.create_element("em");
        d.append_child(d.root(), a).unwrap();
        d.append_child(a, b).unwrap();
        d.append_child(b, c).unwrap();
        // Start from a fully clean slate (nodes are born all-dirty) so we can
        // observe that a layout mark touches ONLY the layout stage.
        for n in [d.root(), a, b, c] {
            if let Some(r) = d.rec_mut(n) {
                r.stage = StageMask::empty();
            }
        }
        assert!(!d.needs_layout(c), "cleared");
        d.mark_needs_layout(c);
        assert!(d.needs_layout(c), "self NEEDS_LAYOUT set");
        // Every ancestor carries CHILD_NEEDS_LAYOUT.
        assert!(d.child_needs_layout(b));
        assert!(d.child_needs_layout(a));
        assert!(d.child_needs_layout(d.root()));
        // The node itself does NOT get a descendant marker from its own mark.
        assert!(!d.child_needs_layout(c));
        // Layout marking left the STYLE and PAINT descendant stages alone — we
        // cleared all stages' descendant bits above, and only marked layout.
        assert!(!d.child_needs_style(a), "style stage untouched by a layout mark");
        assert!(!d.child_needs_paint(a), "paint stage untouched by a layout mark");
    }

    #[test]
    fn style_stage_delegation_matches_old_bool_semantics() {
        // The preserved style API must behave exactly as the old two bools did.
        let mut d = Document::new();
        let a = d.create_element("div");
        let b = d.create_element("span");
        d.append_child(d.root(), a).unwrap();
        d.append_child(a, b).unwrap();
        // Born style-dirty.
        assert!(d.style_dirty(a) && d.style_dirty(b));
        // The style query is exactly the NEEDS_STYLE self bit.
        assert_eq!(d.style_dirty(b), d.needs_style(b));
        assert_eq!(d.child_styles_dirty(a), d.child_needs_style(a));
        d.clear_all_style_dirty();
        assert!(!d.style_dirty(a) && !d.style_dirty(b));
        assert!(!d.child_styles_dirty(a));
        // mark_style_dirty is mark(NEEDS_STYLE): sets self + propagates up.
        d.mark_style_dirty(b);
        assert!(d.style_dirty(b));
        assert!(d.child_styles_dirty(a), "propagated to parent");
        assert!(d.child_styles_dirty(d.root()), "propagated to root");
        // A missing/stale node reports dirty (conservative default preserved).
        let detached = d.create_element("p");
        d.clear_style_dirty(detached);
        assert!(!d.style_dirty(detached));
    }

    #[test]
    fn clear_stage_clears_only_that_stage() {
        let mut d = Document::new();
        let a = d.create_element("div");
        d.append_child(d.root(), a).unwrap();
        // Born dirty in all three self stages.
        assert!(d.needs_style(a) && d.needs_layout(a) && d.needs_paint(a));
        d.clear_stage(a, StageMask::NEEDS_LAYOUT);
        assert!(!d.needs_layout(a), "layout self bit cleared");
        assert!(d.needs_style(a), "style untouched");
        assert!(d.needs_paint(a), "paint untouched");
        // Clearing the self bit does not disturb the descendant bits.
        assert!(d.child_needs_layout(a), "descendant-layout bit preserved");
    }

    #[test]
    fn mark_early_out_on_already_flagged_ancestor() {
        // Once an ancestor carries the CHILD_ bit, a second mark deeper in the
        // tree must stop at it (the chain above is already flagged).
        let mut d = Document::new();
        let a = d.create_element("div");
        let b = d.create_element("span");
        let c = d.create_element("em");
        d.append_child(d.root(), a).unwrap();
        d.append_child(a, b).unwrap();
        d.append_child(b, c).unwrap();
        for n in [d.root(), a, b, c] {
            d.clear_stage(n, StageMask::NEEDS_PAINT);
            if let Some(r) = d.rec_mut(n) {
                r.stage.remove(StageMask::CHILD_NEEDS_PAINT);
            }
        }
        // First mark propagates CHILD_NEEDS_PAINT to b, a, root.
        d.mark_needs_paint(c);
        assert!(d.child_needs_paint(a) && d.child_needs_paint(d.root()));
        // Marking c again is a no-op self-already-set early return.
        d.mark_needs_paint(c);
        assert!(d.needs_paint(c));
        // Marking a SIBLING under b: b already has CHILD_NEEDS_PAINT, so the
        // walk sets the sibling's self bit then early-outs at b (root unchanged
        // from already-set, but still set).
        let c2 = d.create_element("i");
        d.clear_stage(c2, StageMask::NEEDS_PAINT);
        if let Some(r) = d.rec_mut(c2) {
            r.stage.remove(StageMask::CHILD_NEEDS_PAINT);
        }
        d.append_child(b, c2).unwrap();
        // (append_child does not touch paint bits; b still has CHILD_NEEDS_PAINT.)
        d.mark_needs_paint(c2);
        assert!(d.needs_paint(c2), "sibling self bit set");
        assert!(d.child_needs_paint(b), "b still flagged");
    }

    #[test]
    fn get_by_attr_finds_first_match() {
        let mut d = Document::new();
        let a = d.create_element("div");
        let b = d.create_element("span");
        d.append_child(d.root(), a).unwrap();
        d.append_child(d.root(), b).unwrap();
        d.set_attribute(a, "\u{1}nid", "7");
        d.set_attribute(b, "data-x", "hello");
        assert_eq!(d.get_by_attr("\u{1}nid", "7"), Some(a));
        assert_eq!(d.get_by_attr("data-x", "hello"), Some(b));
        assert!(d.get_by_attr("\u{1}nid", "999").is_none());
    }

    #[test]
    fn insert_before_orders() {
        let mut d = Document::new();
        let p = d.create_element("p");
        d.append_child(d.root(), p).unwrap();
        let a = d.create_element("a");
        let b = d.create_element("b");
        let c = d.create_element("c");
        d.append_child(p, a).unwrap();
        d.append_child(p, c).unwrap();
        d.insert_before(p, b, Some(c)).unwrap();
        assert_eq!(d.children(p), vec![a, b, c]);
    }

    #[test]
    fn replace_child_swaps() {
        let mut d = Document::new();
        let p = d.create_element("p");
        let a = d.create_element("a");
        let b = d.create_element("b");
        d.append_child(d.root(), p).unwrap();
        d.append_child(p, a).unwrap();
        d.replace_child(p, b, a).unwrap();
        assert_eq!(d.children(p), vec![b]);
    }

    #[test]
    fn hierarchy_request_blocks_cycle() {
        let mut d = Document::new();
        let a = d.create_element("div");
        let b = d.create_element("div");
        d.append_child(d.root(), a).unwrap();
        d.append_child(a, b).unwrap();
        assert!(matches!(
            d.append_child(b, a),
            Err(DomError::HierarchyRequest)
        ));
    }

    #[test]
    fn text_content_concatenates() {
        let mut d = Document::new();
        let p = d.create_element("p");
        d.append_child(d.root(), p).unwrap();
        let t1 = d.create_text_node("hello ");
        let span = d.create_element("span");
        let t2 = d.create_text_node("world");
        d.append_child(p, t1).unwrap();
        d.append_child(p, span).unwrap();
        d.append_child(span, t2).unwrap();
        assert_eq!(d.text_content(p), "hello world");
    }

    #[test]
    fn get_elements_by_class() {
        let mut d = Document::new();
        let a = d.create_element("div");
        let b = d.create_element("div");
        d.append_child(d.root(), a).unwrap();
        d.append_child(d.root(), b).unwrap();
        d.set_attribute(a, "class", "x y");
        d.set_attribute(b, "class", "x");
        assert_eq!(d.get_elements_by_class_name("x y"), vec![a]);
        assert_eq!(d.get_elements_by_class_name("x").len(), 2);
    }

    #[test]
    fn reparent_detaches() {
        let mut d = Document::new();
        let p1 = d.create_element("p");
        let p2 = d.create_element("p");
        let kid = d.create_element("span");
        d.append_child(d.root(), p1).unwrap();
        d.append_child(d.root(), p2).unwrap();
        d.append_child(p1, kid).unwrap();
        d.append_child(p2, kid).unwrap();
        assert!(d.children(p1).is_empty());
        assert_eq!(d.children(p2), vec![kid]);
    }
}
