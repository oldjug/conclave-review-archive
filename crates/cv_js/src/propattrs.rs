//! Per-property attribute side-table — the ECMA-262 property descriptor model
//! (writable / enumerable / configurable + accessor flag), implemented WITHOUT
//! growing the per-object `OrderedMap` (the 96 B layout budget is hard-gated by
//! the M3.2 harness).
//!
//! # Why a side-table (not a per-property struct)
//!
//! The overwhelmingly common case is a plain data property created by
//! assignment, whose attributes are all-default `{writable:true, enumerable:true,
//! configurable:true}`. Bloating every property to a full descriptor struct
//! would regress memory + the Shaped flat-slot hot path. Instead, attributes
//! that DIFFER from the all-default-data-property norm are kept in a process-/
//! thread-local side table keyed by the object's stable `Rc` pointer identity —
//! EXACTLY mirroring the existing `interp::ARRAY_PROPS` side-table for named
//! array properties. An object with NO entry here is a plain object whose every
//! property is a default data property: the value read/write hot paths never
//! consult this table, so they are byte-identical to today.
//!
//! Keying by `(obj_ptr, key)` (not by slot index) is intentional and robust: it
//! survives Shaped→Dict deopt and dictionary reindexing (the key string is the
//! stable identity), and attributes are only ever consulted on the SLOW paths
//! (`defineProperty` / `getOwnPropertyDescriptor` / the `[[Set]]` write-guard /
//! `delete` / the enumeration filter), never on the value get/set fast path.
//!
//! # The attribute byte
//!
//! A packed `u8` with four meaningful bits. The all-default DATA property is
//! `W | E | C` (= 0b0111); a slot with no side-table entry is treated as exactly
//! that. The `IS_ACCESSOR` bit marks a getter/setter property (whose get/set
//! callables still live in the value slot as today's `\u{1}__get__`/`__set__`
//! wrapper Object — this table only FLAGS it so descriptors report `{get,set}`).
//!
//! # Flag gate
//!
//! The whole model is gated by [`prop_desc_enabled`] (`CV_PROP_DESC`, DEFAULT
//! OFF during development). With the flag off NOTHING here is ever written
//! (every `define`/`freeze`/`seal` path checks the flag before stamping an
//! entry) and NOTHING here is ever read for an observable decision — so the
//! flag-off engine is byte-identical to the pre-descriptor baseline. The goal
//! is default-ON once the A/B oracle is green corpus-wide.

use std::cell::RefCell;

/// `writable` bit.
pub const WRITABLE: u8 = 0b0000_0001;
/// `enumerable` bit.
pub const ENUMERABLE: u8 = 0b0000_0010;
/// `configurable` bit.
pub const CONFIGURABLE: u8 = 0b0000_0100;
/// Accessor (getter/setter) property marker. When set, the value slot holds the
/// `\u{1}__get__`/`__set__` accessor wrapper and `writable` is meaningless.
pub const IS_ACCESSOR: u8 = 0b0000_1000;
/// Non-extensible marker bit, stored under the synthetic key
/// [`NONEXTENSIBLE_KEY`] (a per-OBJECT flag, not per-property). When present the
/// object rejects NEW own properties (`Object.preventExtensions`/`seal`/`freeze`).
pub const NONEXTENSIBLE_FLAG: u8 = 0b0001_0000;

/// The all-default DATA property attribute byte (assignment-created prop).
pub const DEFAULT_DATA: u8 = WRITABLE | ENUMERABLE | CONFIGURABLE;

/// Synthetic per-object key under which the object-level non-extensible flag is
/// stored in the side table. Never a real JS property name (begins with the
/// `\u{1}` internal sentinel + cannot collide with a user key because we only
/// ever look it up via this constant).
pub const NONEXTENSIBLE_KEY: &str = "\u{1}__nonext__";

thread_local! {
    /// Test-only per-thread override of the descriptor gate (`Some(true/false)`
    /// forces, `None` defers to the env cache). Lets a unit test exercise BOTH
    /// flag states in one process without relying on the once-read env cache.
    /// Production never sets this (so the env gate is authoritative).
    static FORCE_PROP_DESC: RefCell<Option<bool>> = const { RefCell::new(None) };
}

/// Process gate for the descriptor model. Default OFF; `CV_PROP_DESC=1` (any
/// value other than `0`) turns it on. Read once and cached, matching the
/// `CV_SHAPED_OBJ` / `CV_GC` discipline. A test-only thread-local override
/// (`set_force_prop_desc`) takes precedence when set.
pub fn prop_desc_enabled() -> bool {
    if let Some(forced) = FORCE_PROP_DESC.with(|c| *c.borrow()) {
        return forced;
    }
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("CV_PROP_DESC").is_ok_and(|v| v != "0"))
}

/// Test-only: force the descriptor gate on/off for the current thread.
pub fn set_force_prop_desc(v: Option<bool>) {
    FORCE_PROP_DESC.with(|c| *c.borrow_mut() = v);
}

/// RAII guard that forces the descriptor gate for a test scope and restores the
/// previous override on drop.
pub struct PropDescGuard(Option<bool>);
impl PropDescGuard {
    pub fn new(v: bool) -> Self {
        let prev = FORCE_PROP_DESC.with(|c| *c.borrow());
        set_force_prop_desc(Some(v));
        PropDescGuard(prev)
    }
}
impl Drop for PropDescGuard {
    fn drop(&mut self) {
        set_force_prop_desc(self.0);
    }
}

thread_local! {
    /// Side table of NON-DEFAULT property attributes, keyed by the object's
    /// stable `Rc` pointer identity → `key → attr byte`. Mirrors `ARRAY_PROPS`:
    /// an object with no entry (or whose entry lacks a given key) has all-default
    /// data properties. ABA (a freed `Rc` pointer reused by a new allocation) has
    /// the same risk profile as the long-accepted `ARRAY_PROPS` table; entries are
    /// pruned by [`gc_sweep`] against the set of live object pointers.
    static PROP_ATTRS: RefCell<std::collections::HashMap<usize, std::collections::HashMap<String, u8>>> =
        RefCell::new(std::collections::HashMap::new());
}

/// Read the attribute byte for `obj_ptr[key]`, or `None` if the object has no
/// non-default entry for that key (⇒ treat as a default data property).
#[inline]
pub fn get_attr(obj_ptr: usize, key: &str) -> Option<u8> {
    PROP_ATTRS.with(|m| m.borrow().get(&obj_ptr).and_then(|p| p.get(key).copied()))
}

/// The EFFECTIVE attribute byte for `obj_ptr[key]`: the stored entry if any,
/// else the all-default data byte. (Use [`get_attr`] when you must distinguish
/// "no entry" from "entry == DEFAULT_DATA".)
#[inline]
pub fn effective_attr(obj_ptr: usize, key: &str) -> u8 {
    get_attr(obj_ptr, key).unwrap_or(DEFAULT_DATA)
}

/// Store the attribute byte for `obj_ptr[key]`. If `bits == DEFAULT_DATA` the
/// entry is REMOVED (the absence-means-default invariant keeps the table sparse
/// and `obj_has_attrs` accurate). The object-level non-extensible flag is kept
/// even at its "default" because absence there means extensible.
#[inline]
pub fn set_attr(obj_ptr: usize, key: &str, bits: u8) {
    PROP_ATTRS.with(|m| {
        let mut t = m.borrow_mut();
        if bits == DEFAULT_DATA {
            if let Some(p) = t.get_mut(&obj_ptr) {
                p.remove(key);
                if p.is_empty() {
                    t.remove(&obj_ptr);
                }
            }
        } else {
            t.entry(obj_ptr).or_default().insert(key.to_string(), bits);
        }
    });
}

/// Remove the attribute entry for `obj_ptr[key]` (called when the property is
/// deleted, so the byte can't shadow a later same-key reassignment).
#[inline]
pub fn remove_attr(obj_ptr: usize, key: &str) {
    PROP_ATTRS.with(|m| {
        let mut t = m.borrow_mut();
        if let Some(p) = t.get_mut(&obj_ptr) {
            p.remove(key);
            if p.is_empty() {
                t.remove(&obj_ptr);
            }
        }
    });
}

/// Whether `obj_ptr` carries ANY non-default attribute entry (including the
/// object-level non-extensible flag). This is the per-object "descriptor-aware
/// mode" marker: an object that returns true here must route its writes/deletes
/// through the spec-correct slow path in BOTH tiers (the VM checks this to send
/// such writes to the host hook, exactly as it already does for accessors).
#[inline]
pub fn obj_has_attrs(obj_ptr: usize) -> bool {
    PROP_ATTRS.with(|m| m.borrow().get(&obj_ptr).map(|p| !p.is_empty()).unwrap_or(false))
}

/// Whether `obj_ptr` is non-extensible (preventExtensions/seal/freeze stamped it).
#[inline]
pub fn is_nonextensible(obj_ptr: usize) -> bool {
    get_attr(obj_ptr, NONEXTENSIBLE_KEY)
        .map(|b| b & NONEXTENSIBLE_FLAG != 0)
        .unwrap_or(false)
}

/// Mark `obj_ptr` non-extensible.
#[inline]
pub fn set_nonextensible(obj_ptr: usize) {
    PROP_ATTRS.with(|m| {
        m.borrow_mut()
            .entry(obj_ptr)
            .or_default()
            .insert(NONEXTENSIBLE_KEY.to_string(), NONEXTENSIBLE_FLAG);
    });
}

/// All keys with a stored attribute entry for `obj_ptr` (excludes the synthetic
/// non-extensible key). Used by `freeze`/`seal` to enumerate already-stamped
/// props and by the descriptor-copy helpers.
#[inline]
pub fn attr_keys(obj_ptr: usize) -> Vec<String> {
    PROP_ATTRS.with(|m| {
        m.borrow()
            .get(&obj_ptr)
            .map(|p| {
                p.keys()
                    .filter(|k| k.as_str() != NONEXTENSIBLE_KEY)
                    .cloned()
                    .collect()
            })
            .unwrap_or_default()
    })
}

/// Drop the entire attribute record for `obj_ptr` (called when the object is
/// cleared/rebuilt so stale bytes can't leak into a reused layout).
#[inline]
pub fn drop_obj(obj_ptr: usize) {
    PROP_ATTRS.with(|m| {
        m.borrow_mut().remove(&obj_ptr);
    });
}

/// Prune attribute records for objects no longer live. `is_live(ptr)` returns
/// whether an object pointer is still reachable. Called from the GC sweep so the
/// side-table can't grow unbounded across a long-running page (mirrors the
/// `ARRAY_PROPS`/object-registry retain discipline).
pub fn gc_sweep<F: Fn(usize) -> bool>(is_live: F) {
    PROP_ATTRS.with(|m| {
        m.borrow_mut().retain(|ptr, _| is_live(*ptr));
    });
}

/// Test/diagnostic: number of objects with a stored attribute record.
pub fn tracked_object_count() -> usize {
    PROP_ATTRS.with(|m| m.borrow().len())
}

/// Test helper: clear the entire side-table (isolate one test's allocations).
pub fn reset_for_test() {
    PROP_ATTRS.with(|m| m.borrow_mut().clear());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn absence_means_default() {
        reset_for_test();
        assert_eq!(get_attr(1, "x"), None);
        assert_eq!(effective_attr(1, "x"), DEFAULT_DATA);
        assert!(!obj_has_attrs(1));
    }

    #[test]
    fn storing_default_removes_entry() {
        reset_for_test();
        set_attr(1, "x", WRITABLE); // non-default (no E, no C)
        assert!(obj_has_attrs(1));
        assert_eq!(get_attr(1, "x"), Some(WRITABLE));
        set_attr(1, "x", DEFAULT_DATA); // back to default ⇒ entry dropped
        assert_eq!(get_attr(1, "x"), None);
        assert!(!obj_has_attrs(1));
    }

    #[test]
    fn nonextensible_roundtrip() {
        reset_for_test();
        assert!(!is_nonextensible(7));
        set_nonextensible(7);
        assert!(is_nonextensible(7));
        assert!(obj_has_attrs(7));
        // The synthetic key is excluded from attr_keys.
        assert!(attr_keys(7).is_empty());
    }

    #[test]
    fn attr_keys_lists_only_real_keys() {
        reset_for_test();
        set_attr(9, "a", ENUMERABLE); // non-default
        set_attr(9, "b", CONFIGURABLE); // non-default
        set_nonextensible(9);
        let mut ks = attr_keys(9);
        ks.sort();
        assert_eq!(ks, vec!["a".to_string(), "b".to_string()]);
    }

    #[test]
    fn gc_sweep_prunes_dead() {
        reset_for_test();
        set_attr(100, "x", ENUMERABLE);
        set_attr(200, "y", ENUMERABLE);
        gc_sweep(|p| p == 100);
        assert!(obj_has_attrs(100));
        assert!(!obj_has_attrs(200));
    }
}
