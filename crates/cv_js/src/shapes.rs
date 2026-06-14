//! Hidden classes (shape transitions) for object property storage.
//!
//! Replaces a per-object `HashMap<String, Value>` with a slot vector
//! `Vec<Value>` indexed by a per-shape table. Objects sharing the
//! same insertion sequence share a `Shape`, so a property lookup is
//! `shape.lookup(key) → Option<usize>` then `slots[i]` — much cheaper
//! than a hash probe and the basis for the inline cache in `ic.rs`.
//!
//! Transition model:
//!   - Each shape carries the set of (property, slot) tuples in
//!     insertion order.
//!   - Adding a property creates (or reuses) a *child* shape by
//!     looking up `parent.transitions[(key)]`.
//!   - Reading from an object: walk into `shape` once, use the
//!     resolved slot to index the value store.
//!
//! This table is LIVE: `bytecode.rs::object_shape_id` interns every JS object's
//! key-sequence into a `ShapeId` here, which is what lets the canonical property
//! inline cache (`bytecode.rs::PropIc`) hit across distinct same-shape objects.

use std::collections::HashMap;
use std::sync::Arc;

/// Stable identifier for a shape. Used by the inline cache so it
/// doesn't have to compare shape pointers.
pub type ShapeId = u32;

#[derive(Debug)]
pub struct Shape {
    pub id: ShapeId,
    /// Properties in insertion order — index = slot in the value vector.
    ///
    /// Stored behind an `Rc` so a Shaped object (M3.2) can hold a cheap
    /// shared handle to its key list and hand out `&String` keys (for
    /// `keys()`/`iter()`) that are valid for as long as needed: shapes are
    /// IMMORTAL (the `ShapeTable` holds every shape — and thus a live `Rc` to
    /// its `properties` — for the whole process), so the underlying `Vec`
    /// outlives any borrow. The keys therefore live ONCE per shape (shared by
    /// every same-shape object), never duplicated per object — the memory win.
    properties: Arc<Vec<String>>,
    /// Property name → slot index for O(1) lookup. Mirrored from
    /// `properties` so iteration order is preserved.
    index: HashMap<String, usize>,
    /// Cached transition: adding `key` to this shape produces the
    /// target shape. Keys are pointer-stable so we can clone-cheaply.
    transitions: HashMap<String, ShapeId>,
}

impl Shape {
    fn empty(id: ShapeId) -> Self {
        Self {
            id,
            properties: Arc::new(Vec::new()),
            index: HashMap::new(),
            transitions: HashMap::new(),
        }
    }

    pub fn property_count(&self) -> usize {
        self.properties.len()
    }

    pub fn lookup(&self, key: &str) -> Option<usize> {
        self.index.get(key).copied()
    }

    pub fn properties(&self) -> &[String] {
        &self.properties
    }

    /// A cheap shared handle to this shape's key list (an `Rc` clone). The
    /// underlying `Vec` is immortal (held by the `ShapeTable`), so borrows of
    /// its `String`s are valid for any lifetime.
    pub fn properties_rc(&self) -> Arc<Vec<String>> {
        Arc::clone(&self.properties)
    }
}

/// Shape pool — owns every shape and serves transitions. Objects
/// hold a `ShapeId` (a u32 index into `shapes`).
#[derive(Debug)]
pub struct ShapeTable {
    shapes: Vec<Shape>,
    empty_id: ShapeId,
}

impl Default for ShapeTable {
    fn default() -> Self {
        let mut shapes = Vec::new();
        shapes.push(Shape::empty(0));
        Self {
            shapes,
            empty_id: 0,
        }
    }
}

impl ShapeTable {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn empty(&self) -> ShapeId {
        self.empty_id
    }

    /// Total number of interned shapes (including the empty root). This is the
    /// quantity the M3.2 P5 soak watches: it MUST stay bounded under a churny
    /// long session (dynamic-key bags deopt at `SHAPED_SLOT_CAP` rather than
    /// minting one shape per distinct key), or the shared table would leak.
    pub fn len(&self) -> usize {
        self.shapes.len()
    }

    /// Whether the table holds only the empty root shape.
    pub fn is_empty(&self) -> bool {
        self.shapes.len() <= 1
    }

    pub fn shape(&self, id: ShapeId) -> &Shape {
        &self.shapes[id as usize]
    }

    /// Add `key` to `id`. Returns the target shape — either an
    /// existing transition target or a freshly allocated child
    /// shape.
    pub fn add_property(&mut self, id: ShapeId, key: &str) -> ShapeId {
        if self.shapes[id as usize].index.contains_key(key) {
            return id;
        }
        if let Some(&t) = self.shapes[id as usize].transitions.get(key) {
            return t;
        }
        let mut child = Shape::empty(self.shapes.len() as ShapeId);
        // Build the child's key list as a FRESH owned Vec (the parent's `Rc` is
        // shared/immutable), then re-wrap in an `Rc` the child owns. The parent's
        // `Rc` is untouched, so any object/iterator holding it stays valid.
        let mut props: Vec<String> = (*self.shapes[id as usize].properties).clone();
        child.index = self.shapes[id as usize].index.clone();
        let slot = props.len();
        props.push(key.to_string());
        child.properties = Arc::new(props);
        child.index.insert(key.to_string(), slot);
        let new_id = child.id;
        self.shapes.push(child);
        self.shapes[id as usize]
            .transitions
            .insert(key.to_string(), new_id);
        new_id
    }
}

// ───────────────────────────── shared global table ─────────────────────────────
//
// M3.2 P3: the Shaped object store (`ordered.rs`) and the property inline cache
// (`bytecode.rs`) BOTH intern key-sequences into ONE PROCESS-global `ShapeTable`,
// so a Shaped object's stored `ShapeId` and the IC's resolved `ShapeId` come from
// the same interner and are directly comparable — and, critically, a `ShapeId`
// means the SAME key-sequence on EVERY thread.
//
// Why process-global (a `Mutex`), not `thread_local!` (which is what the IC alone
// used before)? A Shaped object STORES its `ShapeId` and resolves slots by it. The
// engine's off-main renderer architecture moves the page object graph between
// threads, so an object built on thread A and read on thread B must resolve to the
// SAME shape — impossible if each thread numbered shapes independently (a
// thread-local table would let thread B's "shape 382" mean a different layout than
// thread A's, silently reading the WRONG slot). A single global table makes shape
// ids portable. The IC (which only ever re-records on a miss) is unaffected by the
// switch; the Shaped store REQUIRES it. The `Shape.properties` `Arc` (Send+Sync)
// lets the table satisfy `Send` for the global.
//
// Living here (not in `bytecode.rs`) keeps `ordered.rs` from depending on
// `bytecode.rs` (a module cycle — `bytecode` already depends on `ordered`).

use std::sync::Mutex;
use std::sync::OnceLock;

/// The one process-global hidden-class pool. Objects with the same key-sequence
/// intern to the same `ShapeId` across all threads.
fn shape_table() -> &'static Mutex<ShapeTable> {
    static TABLE: OnceLock<Mutex<ShapeTable>> = OnceLock::new();
    TABLE.get_or_init(|| Mutex::new(ShapeTable::new()))
}

/// Lock the global table, recovering from a poisoned lock (a panic mid-mutation
/// leaves the table structurally valid — it's append-only — so the poison is
/// benign; we take the inner guard rather than propagate the panic).
#[inline]
fn lock_table() -> std::sync::MutexGuard<'static, ShapeTable> {
    shape_table().lock().unwrap_or_else(|p| p.into_inner())
}

/// A `ShapeId` no real transition ever produces. The Shaped store NEVER carries
/// it; a DEOPTED (Dict) object reports it from `object_shape_id` so the IC's
/// `PropIc::lookup` can never match a deopted object (clean miss — the IC stays
/// off for deopted objects only, never for flag-off Dict objects, which keep
/// reporting their real key-rewalk shape exactly as before).
pub const DICT_SHAPE: ShapeId = u32::MAX;

/// The shared empty (zero-property) shape id.
#[inline]
pub fn global_empty_shape() -> ShapeId {
    lock_table().empty()
}

/// Add `key` to shape `id` in the shared table, returning the target shape
/// (existing transition or freshly created). The single interning entry point
/// used by both the Shaped store and the IC.
#[inline]
pub fn global_add_property(id: ShapeId, key: &str) -> ShapeId {
    lock_table().add_property(id, key)
}

/// Slot index of `key` in shape `id` (shared table), or `None` if absent.
#[inline]
pub fn global_shape_lookup(id: ShapeId, key: &str) -> Option<usize> {
    lock_table().shape(id).lookup(key)
}

/// The properties (in slot order) of shape `id` (shared table), cloned out.
/// Used by the Shaped store's deopt rematerialization.
#[inline]
pub fn global_shape_properties(id: ShapeId) -> Vec<String> {
    lock_table().shape(id).properties().to_vec()
}

/// A cheap shared (`Arc`) handle to shape `id`'s key list. The underlying `Vec`
/// is immortal (the `ShapeTable` holds the canonical `Arc` for the whole
/// process), so borrows of its `String`s are valid for any lifetime — this is
/// what lets a Shaped object's `keys()`/`iter()` hand out `&String` keys without
/// duplicating them per object.
#[inline]
pub fn global_shape_properties_rc(id: ShapeId) -> Arc<Vec<String>> {
    lock_table().shape(id).properties_rc()
}

/// Number of properties in shape `id` (shared table).
#[inline]
pub fn global_shape_property_count(id: ShapeId) -> usize {
    lock_table().shape(id).property_count()
}

/// Total number of shapes interned in the ONE process-global table. The M3.2 P5
/// soak asserts this stays BOUNDED across a long churny session: every distinct
/// key SEQUENCE mints at most one shape (shared by all same-shape objects), and
/// dynamic-key bags deopt at the slot cap rather than minting unbounded shapes,
/// so a finite set of program shapes produces a finite table — it never grows
/// with the number of OBJECTS allocated.
#[inline]
pub fn global_shape_count() -> usize {
    lock_table().len()
}

/// Run `f` with exclusive access to the global table (lets a caller resolve a
/// whole key-sequence under one lock — the IC's `object_shape_id` rewalk).
#[inline]
pub fn with_shape_table<R>(f: impl FnOnce(&mut ShapeTable) -> R) -> R {
    f(&mut lock_table())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_shape_has_no_properties() {
        let t = ShapeTable::new();
        assert_eq!(t.shape(t.empty()).property_count(), 0);
    }

    #[test]
    fn add_property_creates_child_shape() {
        let mut t = ShapeTable::new();
        let s1 = t.add_property(t.empty(), "x");
        assert_ne!(s1, t.empty());
        assert_eq!(t.shape(s1).lookup("x"), Some(0));
    }

    #[test]
    fn same_sequence_shares_shape() {
        let mut t = ShapeTable::new();
        let s1 = t.add_property(t.empty(), "x");
        let s2 = t.add_property(t.empty(), "x");
        assert_eq!(s1, s2);
    }

    #[test]
    fn diverging_sequences_make_separate_shapes() {
        let mut t = ShapeTable::new();
        let sx = t.add_property(t.empty(), "x");
        let sy = t.add_property(t.empty(), "y");
        assert_ne!(sx, sy);
    }

    #[test]
    fn long_sequence_indexes_in_order() {
        let mut t = ShapeTable::new();
        let mut s = t.empty();
        for k in ["a", "b", "c", "d"] {
            s = t.add_property(s, k);
        }
        assert_eq!(t.shape(s).lookup("a"), Some(0));
        assert_eq!(t.shape(s).lookup("d"), Some(3));
    }

    // ── transition-tree coverage repurposed from the deleted `ShapedObject`
    // tests (M3.2 Phase 1): the same observable behaviors, asserted directly on
    // `ShapeTable` (the LIVE type), so deleting the orphaned generic helper lost
    // no transition-tree coverage. ──

    /// Building a key sequence transitions to a shape whose `lookup` resolves
    /// each key to its insertion slot, and reports a missing key as `None` —
    /// the slot-resolution the value store indexes by. (Was
    /// `object_get_set_through_shape`.)
    #[test]
    fn transition_sequence_resolves_slots_in_order() {
        let mut t = ShapeTable::new();
        let s = t.add_property(t.empty(), "x");
        let s = t.add_property(s, "y");
        assert_eq!(t.shape(s).lookup("x"), Some(0));
        assert_eq!(t.shape(s).lookup("y"), Some(1));
        assert_eq!(t.shape(s).lookup("z"), None);
        assert_eq!(t.shape(s).property_count(), 2);
    }

    /// Re-adding an ALREADY-present key is NOT a transition: it returns the same
    /// shape (an overwrite changes a value, never the layout). (Was
    /// `overwriting_property_keeps_shape`.)
    #[test]
    fn readding_existing_key_keeps_same_shape() {
        let mut t = ShapeTable::new();
        let s_before = t.add_property(t.empty(), "x");
        let s_after = t.add_property(s_before, "x");
        assert_eq!(s_after, s_before);
        assert_eq!(t.shape(s_after).lookup("x"), Some(0));
    }

    /// Two INDEPENDENT build-ups of the same key sequence intern to the SAME
    /// shape — the transition tree is shared across objects, which is what lets
    /// the inline cache hit across distinct same-shape objects. (Was
    /// `two_objects_share_shape_when_same_keys_added`.)
    #[test]
    fn independent_same_sequence_buildups_share_shape() {
        let mut t = ShapeTable::new();
        // build-up A: empty -> x -> y
        let a = {
            let s = t.add_property(t.empty(), "x");
            t.add_property(s, "y")
        };
        // build-up B: empty -> x -> y (separately)
        let b = {
            let s = t.add_property(t.empty(), "x");
            t.add_property(s, "y")
        };
        assert_eq!(a, b);
        assert_eq!(t.shape(a).properties(), &["x".to_string(), "y".to_string()]);
    }
}
