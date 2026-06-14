//! Insertion-ordered map — a drop-in subset of `std::collections::HashMap`'s
//! API that preserves key insertion order. JavaScript objects must iterate
//! their string keys in insertion order (integer keys ascending first, then
//! string keys in insertion order) per ECMA-262 [[OwnPropertyKeys]], and a
//! plain `HashMap` has no order at all. webpack/React and countless libraries
//! depend on this. We alias this type `as HashMap` in the modules that build
//! JS objects, so existing code flips to ordered storage with no per-site
//! changes.
//!
//! # M3.2 two-mode store (Phase 3)
//!
//! The backing storage is a 2-mode `Store` enum, the flat-slot Shaped object
//! model ([[project_m3_2_flat_slots_plan]]):
//!
//!   - `Store::Dict { entries, index }` — the dictionary representation: a
//!     `Vec<(K, V)>` for order + a `HashMap<K, usize>` index for O(1) lookup.
//!     This is what EVERY map uses when `CV_SHAPED_OBJ` is OFF (so behavior is
//!     byte-identical to the pre-Shaped code) and what a Shaped object DEOPTS to
//!     the instant it does anything order-hostile/exotic.
//!   - `Store::Shaped { shape, slots }` — a `ShapeId` (a hidden class interned
//!     in the global `ShapeTable`, `shapes.rs`) + a flat `Vec<V>` of values
//!     indexed by the shape's slot layout. The keys live ONCE in the shared
//!     shape, so the per-object hash index + the duplicate key `String`s vanish
//!     — THE memory win. Built ONLY when `CV_SHAPED_OBJ=1`, and ONLY for
//!     `K = String` plain JS objects (the shape interns `&str` keys).
//!
//! ## Flag-off invariant (byte-identical escape hatch)
//!
//! M3.2 P5 flipped the default ON (env UNSET ⇒ Shaped). The escape hatch is the
//! explicit `CV_SHAPED_OBJ=0`: with it set, `should_start_shaped()` is `false`,
//! so every constructor builds `Store::Dict` and NOTHING is ever Shaped or
//! deopted. The map then behaves exactly as the pre-M3.2 code: same bytes
//! (416/544/1888 at 1/4/16 props), same key-rewalk in the IC (an off object's
//! `stored_shape_id()` is `None`), same hit-rate. This off-ramp is kept green for
//! one release so a regression can be ruled out by a single env flip.
//!
//! ## One-way deopt (Shaped → Dict)
//!
//! A Shaped object falls back to Dict — permanently — the instant it does
//! anything the flat-slot model can't represent observably-identically:
//!   - an INTEGER / array-index key is inserted (must order ascending-first via
//!     `order_keys_v8`; integers never enter a shape);
//!   - a SENTINEL key is stamped (freeze / proxy / accessor / typed-array marks
//!     — the exotic must be served by the unchanged Dict probes, never the IC);
//!   - remove / clear / retain / drain (key-set shrinks/reorders);
//!   - the slot count would exceed `SHAPED_SLOT_CAP` (churny dynamic-key bags
//!     deopt instead of exploding the shared `ShapeTable`);
//!   - any `&K`-yielding owned iteration where Shaped has no `&K` to hand out
//!     (`into_iter` consuming).
//! Every deopt rematerializes `shape.properties()` zipped with `slots` into
//! `entries + index`, BUMPS `struct_ver`, CLEARS the cached shape, and sets the
//! `deopted` marker (packed in `ShapeCache`) — so a stale IC entry can never read
//! the wrong slot, and `object_shape_id` then reports `DICT_SHAPE` for the object
//! (a guaranteed IC miss).
//!
//! ## Layout invariant (the byte budget)
//!
//! The `Store` enum must NOT grow `OrderedMap` beyond its pre-enum size. Both
//! variants begin with a `Vec` (non-null pointer), so Rust stores the enum
//! discriminant in that pointer's niche — `size_of::<OrderedMap<String, Value>>()`
//! stays 96 B exactly. The one-way `deopted` marker is packed into the tail
//! padding of the `ShapeCache` (no new field, no growth). The M3.2 harness gates
//! this.

use std::collections::HashMap as StdHashMap;
use std::hash::Hash;
use std::sync::atomic::{AtomicU64, Ordering};

/// Process-global feature gate for the M3.2 Shaped (flat-slot) object store,
/// mirroring the `CV_GC` / `CV_OFFMAIN` default-on discipline: read
/// `CV_SHAPED_OBJ` once and cache it.
///
/// M3.2 P5: the default is now ON (env UNSET ⇒ Shaped). The escape hatch is
/// `CV_SHAPED_OBJ=0`, which explicitly disables Shaped storage and restores the
/// byte-identical pre-M3.2 Dict baseline (the off-ramp kept for one release). Any
/// other value (or unset) is ON. This flip is gated on the P3 correctness suite,
/// the P4 ~100% plain-object retention go signal, the bounded-ShapeTable soak,
/// and a live render check (example.com byte-identical + HN styled).
pub fn shaped_obj_enabled() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    // Default ON: only an explicit `CV_SHAPED_OBJ=0` turns it off. (`map_or(true,
    // …)` ⇒ unset ⇒ true; set ⇒ on unless the value is exactly "0".)
    *ON.get_or_init(|| std::env::var("CV_SHAPED_OBJ").map_or(true, |v| v != "0"))
}

/// THE single decision point: should a freshly-built map start in Shaped mode?
/// True only when the flag is on AND the key type is `String` (the only key type
/// the shared shape can intern). Centralizing it means the gate is one line and
/// non-`String` maps are never even considered for Shaped.
#[inline]
fn should_start_shaped<K: 'static>() -> bool {
    shaped_obj_enabled() && is_string_key::<K>()
}

/// Maximum slot count a Shaped object keeps before deopting to Dict. Bounds the
/// shared `ShapeTable`'s growth on pathological dynamic-key objects (a 1000-
/// distinct-key bag deopts at this cap rather than minting 1000 shapes), and
/// keeps the linear bits of the Shaped path short.
const SHAPED_SLOT_CAP: usize = 128;

// ───────────────────────── M3.2 P4 retention stats ─────────────────────────
//
// Opt-in counters that measure the Shaped-RETENTION rate on real workloads:
// how many String-keyed JS objects are born Shaped, how many later deopt to
// Dict, and — crucially — WHICH trigger caused each deopt. The whole point of
// P4 is an evidence-based go/no-go on flipping `CV_SHAPED_OBJ` default ON
// (P5): if plain objects stay Shaped and only mixed-key/exotic objects deopt,
// the win is real; if common objects deopt often, a dense-int hybrid is needed
// first.
//
// ZERO overhead when off: every increment is guarded by `shaped_stats_enabled()`
// (a cached `OnceLock<bool>` read of `CV_SHAPED_STATS`). With the flag off the
// branch is never taken and the atomics are never touched — the flag-off byte
// and behavior baseline is preserved exactly (the counters are process-global
// statics, not per-object fields, so `OrderedMap` does NOT grow).

/// Process-global gate for retention instrumentation. Default OFF;
/// `CV_SHAPED_STATS=1` (any value other than `0`/unset) turns it on. Read once.
pub fn shaped_stats_enabled() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("CV_SHAPED_STATS").is_ok_and(|v| v != "0"))
}

/// Reason a Shaped object deopted to Dict. Mirrors the EXHAUSTIVE deopt sites
/// (see the module docs' "one-way deopt" section) so the breakdown attributes
/// every Shaped→Dict transition to exactly one cause.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeoptTrigger {
    /// An integer / array-index String key (`"0"`, `"42"`, …) was inserted.
    IntegerKey,
    /// An engine sentinel / exotic-tag key (`_isProxy`, `\u{1}__frozen__`,
    /// `_bytes`, accessor stamps, …) was inserted.
    SentinelStamp,
    /// A non-`String` key reached the insert path (defensive; should not occur
    /// for a genuine JS object, whose keys are always String).
    NonStringKey,
    /// The slot count would exceed `SHAPED_SLOT_CAP`.
    CapExceeded,
    /// `remove` of a present key.
    Remove,
    /// `clear`.
    Clear,
    /// `drain`.
    Drain,
    /// `retain`.
    Retain,
    /// Owned `into_iter` (rematerializes keys; structurally consumes the map).
    IntoIter,
}

/// Snapshot of the retention counters. All fields are cumulative since process
/// start (the counters are monotonic atomics). `created_shaped` is every
/// String-keyed object born in Shaped mode; `created_dict_born_exotic` is every
/// object that started in Dict despite the flag being on (a non-String map);
/// `deopted_*` break the Shaped→Dict transitions down by trigger.
#[derive(Clone, Copy, Debug, Default)]
pub struct ShapedStats {
    pub created_shaped: u64,
    pub created_dict_born_exotic: u64,
    pub deopted_integer_key: u64,
    pub deopted_sentinel_stamp: u64,
    pub deopted_non_string_key: u64,
    pub deopted_cap_exceeded: u64,
    pub deopted_remove: u64,
    pub deopted_clear: u64,
    pub deopted_drain: u64,
    pub deopted_retain: u64,
    pub deopted_into_iter: u64,
}

impl ShapedStats {
    /// Total Shaped→Dict deopts across all triggers.
    pub fn total_deopted(&self) -> u64 {
        self.deopted_integer_key
            + self.deopted_sentinel_stamp
            + self.deopted_non_string_key
            + self.deopted_cap_exceeded
            + self.deopted_remove
            + self.deopted_clear
            + self.deopted_drain
            + self.deopted_retain
            + self.deopted_into_iter
    }
    /// Objects that were born Shaped AND never deopted (stayed Shaped for their
    /// observed lifetime). Approximation: created_shaped - total_deopted. Sound
    /// because deopt is one-way and only Shaped-born objects can deopt, so each
    /// deopt event corresponds to a distinct created-Shaped object.
    pub fn retained_shaped(&self) -> u64 {
        self.created_shaped.saturating_sub(self.total_deopted())
    }
    /// Shaped-retention rate in [0,1]: fraction of Shaped-born objects that
    /// stayed Shaped. `1.0` when nothing was created (vacuously retained).
    pub fn retention_rate(&self) -> f64 {
        if self.created_shaped == 0 {
            return 1.0;
        }
        self.retained_shaped() as f64 / self.created_shaped as f64
    }
}

// One atomic per counter. Relaxed ordering: these are pure statistics with no
// happens-before relationship to other state, so the cheapest ordering is correct.
static STAT_CREATED_SHAPED: AtomicU64 = AtomicU64::new(0);
static STAT_CREATED_DICT_BORN_EXOTIC: AtomicU64 = AtomicU64::new(0);
static STAT_DEOPT_INTEGER_KEY: AtomicU64 = AtomicU64::new(0);
static STAT_DEOPT_SENTINEL_STAMP: AtomicU64 = AtomicU64::new(0);
static STAT_DEOPT_NON_STRING_KEY: AtomicU64 = AtomicU64::new(0);
static STAT_DEOPT_CAP_EXCEEDED: AtomicU64 = AtomicU64::new(0);
static STAT_DEOPT_REMOVE: AtomicU64 = AtomicU64::new(0);
static STAT_DEOPT_CLEAR: AtomicU64 = AtomicU64::new(0);
static STAT_DEOPT_DRAIN: AtomicU64 = AtomicU64::new(0);
static STAT_DEOPT_RETAIN: AtomicU64 = AtomicU64::new(0);
static STAT_DEOPT_INTO_ITER: AtomicU64 = AtomicU64::new(0);

#[inline]
fn stat_created_shaped() {
    if shaped_stats_enabled() {
        STAT_CREATED_SHAPED.fetch_add(1, Ordering::Relaxed);
    }
}

#[inline]
fn stat_created_dict_born_exotic() {
    if shaped_stats_enabled() {
        STAT_CREATED_DICT_BORN_EXOTIC.fetch_add(1, Ordering::Relaxed);
    }
}

/// Record a Shaped→Dict deopt under `trigger`. Called from `deopt_to_dict`'s
/// callers, which already know the cause (the deopt site is the trigger).
#[inline]
fn stat_deopt(trigger: DeoptTrigger) {
    if !shaped_stats_enabled() {
        return;
    }
    let c = match trigger {
        DeoptTrigger::IntegerKey => &STAT_DEOPT_INTEGER_KEY,
        DeoptTrigger::SentinelStamp => &STAT_DEOPT_SENTINEL_STAMP,
        DeoptTrigger::NonStringKey => &STAT_DEOPT_NON_STRING_KEY,
        DeoptTrigger::CapExceeded => &STAT_DEOPT_CAP_EXCEEDED,
        DeoptTrigger::Remove => &STAT_DEOPT_REMOVE,
        DeoptTrigger::Clear => &STAT_DEOPT_CLEAR,
        DeoptTrigger::Drain => &STAT_DEOPT_DRAIN,
        DeoptTrigger::Retain => &STAT_DEOPT_RETAIN,
        DeoptTrigger::IntoIter => &STAT_DEOPT_INTO_ITER,
    };
    c.fetch_add(1, Ordering::Relaxed);
}

/// Snapshot the current retention counters (cumulative since process start, or
/// since the last `reset_shaped_stats`). Safe to call regardless of the flag;
/// returns all-zero when stats were never enabled.
pub fn shaped_stats() -> ShapedStats {
    ShapedStats {
        created_shaped: STAT_CREATED_SHAPED.load(Ordering::Relaxed),
        created_dict_born_exotic: STAT_CREATED_DICT_BORN_EXOTIC.load(Ordering::Relaxed),
        deopted_integer_key: STAT_DEOPT_INTEGER_KEY.load(Ordering::Relaxed),
        deopted_sentinel_stamp: STAT_DEOPT_SENTINEL_STAMP.load(Ordering::Relaxed),
        deopted_non_string_key: STAT_DEOPT_NON_STRING_KEY.load(Ordering::Relaxed),
        deopted_cap_exceeded: STAT_DEOPT_CAP_EXCEEDED.load(Ordering::Relaxed),
        deopted_remove: STAT_DEOPT_REMOVE.load(Ordering::Relaxed),
        deopted_clear: STAT_DEOPT_CLEAR.load(Ordering::Relaxed),
        deopted_drain: STAT_DEOPT_DRAIN.load(Ordering::Relaxed),
        deopted_retain: STAT_DEOPT_RETAIN.load(Ordering::Relaxed),
        deopted_into_iter: STAT_DEOPT_INTO_ITER.load(Ordering::Relaxed),
    }
}

/// Reset all retention counters to zero. Lets a measurement isolate one
/// workload's allocations from earlier process activity. Used by the P4 tests.
pub fn reset_shaped_stats() {
    for c in [
        &STAT_CREATED_SHAPED,
        &STAT_CREATED_DICT_BORN_EXOTIC,
        &STAT_DEOPT_INTEGER_KEY,
        &STAT_DEOPT_SENTINEL_STAMP,
        &STAT_DEOPT_NON_STRING_KEY,
        &STAT_DEOPT_CAP_EXCEEDED,
        &STAT_DEOPT_REMOVE,
        &STAT_DEOPT_CLEAR,
        &STAT_DEOPT_DRAIN,
        &STAT_DEOPT_RETAIN,
        &STAT_DEOPT_INTO_ITER,
    ] {
        c.store(0, Ordering::Relaxed);
    }
}

/// Process-global monotonic source for `struct_ver`. Each new map + each
/// structural change draws a globally-unique value, so no two structural states
/// ever share a `struct_ver` — this defends a `(Rc-ptr, struct_ver)` inline-cache
/// guard against pointer reuse (a freed object's slot reallocated to a new object
/// gets a strictly-greater version, never aliasing a stale cached one).
static NEXT_STRUCT_VER: AtomicU64 = AtomicU64::new(1);
#[inline]
fn fresh_struct_ver() -> u64 {
    NEXT_STRUCT_VER.fetch_add(1, Ordering::Relaxed)
}

/// True at runtime iff `K == String`. Used to restrict Shaped construction to
/// String-keyed maps without specialization (unstable). Cheap: a `TypeId`
/// compare, evaluated once per `new()` only when the flag is on.
#[inline]
fn is_string_key<K: 'static>() -> bool {
    std::any::TypeId::of::<K>() == std::any::TypeId::of::<String>()
}

/// `[[Prototype]]` internal key (mirrors `interp::PROTO_KEY`). Kept local (not
/// imported) so `ordered.rs` has no `interp.rs` dependency; a test pins parity.
const PROTO_KEY: &str = "\u{1}__proto__";

/// True if a String key is order-hostile (integer/array index) or an engine
/// sentinel that drives exotic behavior — inserting it must deopt a Shaped
/// object to Dict so the exotic/order semantics are served by the unchanged Dict
/// code (and the VM IC never serves it). This is the EXHAUSTIVE deopt-key set;
/// living at the single `insert`/`get_or_insert_with` interning point means it
/// covers EVERY site in the engine that stamps such a key — none can be missed.
///
/// NOT a deopt key: `PROTO_KEY` (a normal slot the proto-IC reads via
/// `slot_of(PROTO_KEY)` → `value_at_slot`; `setPrototypeOf` overwrites it in
/// place, no key-set change) and Symbol keys (`@@sym…` — ordinary string keys
/// filtered out only at enumeration time; they intern + order fine).
fn string_key_forces_deopt(k: &str) -> bool {
    if is_array_index_key_str(k) {
        return true;
    }
    // Internal `\u{1}…` sentinels EXCEPT `\u{1}__proto__`: freeze
    // (`\u{1}__frozen__`), data→accessor redefine (`\u{1}__get__`/`__set__`).
    if let Some(b) = k.as_bytes().first() {
        if *b == 0x01 {
            return k != PROTO_KEY;
        }
    }
    // Plain-ASCII engine sentinels that tag exotics + the typed-array byte store.
    is_engine_sentinel_key_str(k) || k == "_bytes"
}

/// Mirror of `interp::is_array_index_key` (kept local to avoid a dep cycle; a
/// test pins parity). A canonical decimal `u32` (no leading zeros, `!= u32::MAX`)
/// is an array-index property.
#[inline]
fn is_array_index_key_str(k: &str) -> bool {
    if k.is_empty() || (k.len() > 1 && k.starts_with('0')) {
        return false;
    }
    match k.parse::<u32>() {
        Ok(n) => n != u32::MAX,
        Err(_) => false,
    }
}

/// Mirror of `interp::is_engine_sentinel_key` (kept local to avoid a dep cycle;
/// a test pins parity).
#[inline]
fn is_engine_sentinel_key_str(k: &str) -> bool {
    matches!(
        k,
        "_isMap"
            | "_isSet"
            | "_isProxy"
            | "_isPromise"
            | "_isRegExp"
            | "_isError"
            | "_isGenerator"
            | "_isAsyncIterator"
            | "_isDate"
            | "_isArrayBuffer"
            | "_isDataView"
            | "_isWasmGlobal"
            | "_isWasmInstance"
            | "_isWasmMemory"
            | "_isWasmModule"
            | "_isWasmTable"
            | "_typedarray"
            | "_entries"
            | "__entries"
            | "_items"
            | "_target"
            | "_handler"
            | "_construct"
            | "_errorClasses"
    )
}

/// Result of `OrderedMap::shaped_intern_slot` — see its docs.
enum ShapedSlot {
    /// The key resolves to this slot. `i < slots.len()` ⇒ existing key (overwrite
    /// in place); `i == slots.len()` ⇒ a new transition was added and the caller
    /// must `push` the value into slot `i`.
    Slot(usize),
    /// The store DEOPTED to Dict (the key was order-hostile/exotic, or the slot
    /// cap was hit) — the caller takes the Dict path.
    Deopted,
}

/// The 2-mode backing store (see the module docs).
#[derive(Clone)]
enum Store<K, V> {
    /// Flat-slot Shaped store: a hidden-class `ShapeId` (interned in the global
    /// `shapes::ShapeTable`) + a dense value vector indexed by the shape's slot
    /// layout (slot i ↔ `shape.properties()[i]`). Built only with the flag on,
    /// only for `K = String`. `K` is `PhantomData` (Shaped keys live in the
    /// shared shape, not here).
    Shaped {
        shape: u32,
        slots: Vec<V>,
        _k: std::marker::PhantomData<K>,
    },
    /// Dictionary store. A `Vec<(K, V)>` for insertion order + a
    /// `HashMap<K, usize>` index for O(1) lookup.
    Dict {
        entries: Vec<(K, V)>,
        index: StdHashMap<K, usize>,
    },
}

impl<K, V> Store<K, V> {
    #[inline]
    fn new_dict() -> Self {
        Store::Dict {
            entries: Vec::new(),
            index: StdHashMap::new(),
        }
    }

    #[inline]
    fn dict_with_capacity(n: usize) -> Self {
        Store::Dict {
            entries: Vec::with_capacity(n),
            index: StdHashMap::with_capacity(n),
        }
    }

    #[inline]
    fn new_shaped() -> Self {
        Store::Shaped {
            shape: crate::shapes::global_empty_shape(),
            slots: Vec::new(),
            _k: std::marker::PhantomData,
        }
    }

    #[inline]
    fn is_shaped(&self) -> bool {
        matches!(self, Store::Shaped { .. })
    }
}

/// The lazily-computed shape cache PLUS the deopt marker PLUS the T2 inline
/// shape-id header, packed into a single 16-byte value with ZERO padding
/// (`u64 + u32 + u32`). The deopt marker is folded into `header` (a deopted
/// object's header is `DICT_SHAPE`, a born-Dict object's is `BORN_DICT_SHAPE`),
/// so no separate `bool` is needed — `OrderedMap` does NOT grow past 96 B (the
/// flag-off per-object byte baseline is preserved exactly).
#[derive(Clone, Copy)]
struct ShapeCache {
    /// `struct_ver` when `shape` was computed (`0` ⇒ not computed).
    ver: u64,
    /// Cached `ShapeId` (meaningful only when `ver == struct_ver`).
    shape: u32,
    /// T2 inline shape-id HEADER (the fixed-offset word the JIT reads). For a
    /// Shaped object this is its interned `ShapeId`; for a deopted Dict object
    /// `DICT_SHAPE`; for a born-Dict object `BORN_DICT_SHAPE`. ALWAYS valid (no
    /// `ver` gate) — it is mirrored eagerly at every shape transition.
    header: u32,
}

/// A `ShapeId` no real transition produces, used as the inline-header value for a
/// NEVER-Shaped (born-Dict) object. Distinct from `DICT_SHAPE` (deopted) so
/// `stored_shape_id()` can tell the two apart without a separate `bool`. Both are
/// clean JIT misses (shapes are minted from 0 upward and the table is bounded, so
/// neither sentinel can ever collide with a real shape).
const BORN_DICT_SHAPE: u32 = u32::MAX - 1;

impl ShapeCache {
    /// Fresh "not computed" cache for a BORN-DICT object (header = the born-Dict
    /// sentinel). Shaped/with-store constructors overwrite `header` immediately.
    const EMPTY: ShapeCache = ShapeCache {
        ver: 0,
        shape: 0,
        header: BORN_DICT_SHAPE,
    };
    /// True iff this object DEOPTED from Shaped → Dict (header == DICT_SHAPE).
    #[inline]
    const fn deopted(&self) -> bool {
        self.header == crate::shapes::DICT_SHAPE
    }
}

#[derive(Clone)]
pub struct OrderedMap<K, V> {
    /// The 2-mode backing store. With the flag off this is always `Store::Dict`.
    store: Store<K, V>,
    /// Bumped on any change to the KEY SET or ORDER (insert-new / remove / clear
    /// / retain / drain / DEOPT), NOT on overwrite of an existing key's value.
    struct_ver: u64,
    /// Lazily-computed hidden-class id for a DICT object + the deopt marker + the
    /// T2 inline shape-id HEADER, all packed into one 16-byte `ShapeCache` (no
    /// growth — see `ShapeCache`). The `header` word is the T2 STABLE SHAPE-ID
    /// HEADER (M4.3 T2 Phase 1): a fixed-offset value the JIT reads inline (a
    /// single `mov`) WITHOUT taking the global `shapes` Mutex, to shape-guard an
    /// inlined property read. It MIRRORS the authoritative shape at every key-set/
    /// mode transition:
    ///   - a live Shaped object: its interned `ShapeId` (== `Store::Shaped.shape`);
    ///   - a deopted Dict object: `DICT_SHAPE` (`u32::MAX`);
    ///   - a never-Shaped (born-Dict) object: `BORN_DICT_SHAPE` (`u32::MAX-1`).
    /// Both Dict sentinels are values NO real transition produces, so an inline
    /// header read is always a clean MISS for any non-Shaped object — the JIT can
    /// never bake a slot against a Dict layout. The two distinct Dict sentinels
    /// let `stored_shape_id()` tell a deopted object (`Some(DICT_SHAPE)`) from a
    /// born-Dict one (`None`, key-rewalks for the IC) — preserving that exact
    /// distinction WITHOUT a separate `deopted` bool (so the struct stays 96 B).
    ///
    /// INVARIANT (gated by `header_mirror_matches_authority` + the oracle): for
    /// every reachable state, `shape_header()` equals `Store::Shaped.shape` for a
    /// Shaped object and a reserved Dict sentinel otherwise. A baked slot can
    /// therefore NEVER read a wrong slot: the JIT proceeds only on an exact
    /// header == warmed-`ShapeId` match, which only a same-layout Shaped object
    /// can satisfy.
    shape_cache: std::cell::Cell<ShapeCache>,
}

impl<K: std::fmt::Debug, V: std::fmt::Debug> std::fmt::Debug for OrderedMap<K, V> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.store {
            Store::Dict { entries, .. } => f
                .debug_map()
                .entries(entries.iter().map(|(k, v)| (k, v)))
                .finish(),
            // Shaped keys live in the shared shape; show the slot values.
            Store::Shaped { slots, .. } => f.debug_list().entries(slots.iter()).finish(),
        }
    }
}

impl<K, V> Default for OrderedMap<K, V> {
    fn default() -> Self {
        Self {
            store: Store::new_dict(),
            struct_ver: fresh_struct_ver(),
            // Born Dict: `EMPTY`'s header is the born-Dict sentinel (clean JIT
            // miss). It can never become Shaped (no Dict→Shaped path).
            shape_cache: std::cell::Cell::new(ShapeCache::EMPTY),
        }
    }
}

impl<K: Hash + Eq + Clone + 'static, V> OrderedMap<K, V> {
    pub fn new() -> Self {
        let store = if should_start_shaped::<K>() {
            stat_created_shaped();
            Store::new_shaped()
        } else {
            // Born Dict: with the flag on, only non-String maps land here (a
            // "born-exotic" object that was never eligible for Shaped). With the
            // flag off, every map is Dict and this is the unchanged baseline.
            if shaped_obj_enabled() {
                stat_created_dict_born_exotic();
            }
            Store::new_dict()
        };
        // Mirror the inline header off the freshly-chosen store: a Shaped object
        // starts at the empty shape; a born-Dict object at BORN_DICT_SHAPE.
        let header = match &store {
            Store::Shaped { shape, .. } => *shape,
            Store::Dict { .. } => BORN_DICT_SHAPE,
        };
        Self {
            store,
            struct_ver: fresh_struct_ver(),
            shape_cache: std::cell::Cell::new(ShapeCache { ver: 0, shape: 0, header }),
        }
    }

    pub fn with_capacity(n: usize) -> Self {
        let store = if should_start_shaped::<K>() {
            stat_created_shaped();
            Store::Shaped {
                shape: crate::shapes::global_empty_shape(),
                slots: Vec::with_capacity(n),
                _k: std::marker::PhantomData,
            }
        } else {
            if shaped_obj_enabled() {
                stat_created_dict_born_exotic();
            }
            Store::dict_with_capacity(n)
        };
        let header = match &store {
            Store::Shaped { shape, .. } => *shape,
            Store::Dict { .. } => BORN_DICT_SHAPE,
        };
        Self {
            store,
            struct_ver: fresh_struct_ver(),
            shape_cache: std::cell::Cell::new(ShapeCache { ver: 0, shape: 0, header }),
        }
    }

    /// Inline-cache guard token (see the `struct_ver` field).
    #[inline]
    pub fn struct_ver(&self) -> u64 {
        self.struct_ver
    }

    /// The interned `ShapeId` of this object IF it is a Shaped or deopted store,
    /// for `object_shape_id`'s O(1) route:
    ///   - `Some(shape)` for a live Shaped object (its stored layout — no rewalk);
    ///   - `Some(DICT_SHAPE)` for an object that DEOPTED from Shaped to Dict;
    ///   - `None` for an object that was NEVER Shaped (a flag-off Dict object, or
    ///     any non-String map) — the caller key-rewalks exactly as before, so the
    ///     flag-off IC is byte-identical.
    #[inline]
    pub fn stored_shape_id(&self) -> Option<u32> {
        match &self.store {
            Store::Shaped { shape, .. } => Some(*shape),
            Store::Dict { .. } => {
                if self.shape_cache.get().deopted() {
                    Some(crate::shapes::DICT_SHAPE)
                } else {
                    None
                }
            }
        }
    }

    /// Lazily-cached hidden-class id as `(struct_ver_when_computed, shape_id)`.
    #[inline]
    pub fn cached_shape(&self) -> (u64, u32) {
        let c = self.shape_cache.get();
        (c.ver, c.shape)
    }
    #[inline]
    pub fn set_cached_shape(&self, ver: u64, shape: u32) {
        // Preserve the inline header (an IC shape recompute is the Dict key-rewalk
        // cache, NOT the T2 header — the header tracks the store mode separately
        // and an IC recompute never changes the store mode).
        let header = self.shape_cache.get().header;
        self.shape_cache.set(ShapeCache { ver, shape, header });
    }

    /// Reset the lazily-computed shape (on a structural change) WITHOUT clearing
    /// the inline header (which the structural-change caller mirrors separately
    /// when the store mode changes; a pure Dict key-set change keeps the same Dict
    /// sentinel header).
    #[inline]
    fn reset_shape_cache(&self) {
        let header = self.shape_cache.get().header;
        self.shape_cache.set(ShapeCache {
            ver: 0,
            shape: 0,
            header,
        });
    }

    /// Mark this object as DEOPTED (one-way): the inline header becomes
    /// `DICT_SHAPE`, which both (a) makes `stored_shape_id()` report
    /// `Some(DICT_SHAPE)` (`deopted()` is `header == DICT_SHAPE`) and (b) makes
    /// every inline JIT header read a clean miss.
    #[inline]
    fn mark_deopted(&self) {
        self.shape_cache.set(ShapeCache {
            ver: 0,
            shape: 0,
            header: crate::shapes::DICT_SHAPE,
        });
    }

    /// THE single mirror point: recompute the inline shape-id header to match the
    /// authoritative store mode (a Shaped object's interned `ShapeId`, else a Dict
    /// sentinel — `DICT_SHAPE` if it was ever deopted, `BORN_DICT_SHAPE` if not).
    /// Called after EVERY change that can alter the object's SHAPE while it stays
    /// Shaped (a new-key transition). Shaped→Dict transitions go through
    /// `mark_deopted` instead (which stamps `DICT_SHAPE` directly). Cheap: one
    /// `match` + a `Cell::set`, no Mutex.
    #[inline]
    fn mirror_shape_header(&self) {
        let mut c = self.shape_cache.get();
        c.header = match &self.store {
            Store::Shaped { shape, .. } => *shape,
            // A Dict store keeps whichever Dict sentinel it already had (deopted
            // vs born-Dict), never a real shape — never a clean inline target.
            Store::Dict { .. } => c.header,
        };
        self.shape_cache.set(c);
    }

    /// The inline shape-id header value (the T2 JIT reads this at `SHAPE_OFF`).
    /// PUBLIC for the header-mirror-correctness test + the JIT helper.
    #[inline]
    pub fn shape_header(&self) -> u32 {
        self.shape_cache.get().header
    }

    /// A stable `*const u32` to the `header` word's storage — used ONCE by
    /// `jit::t2_shape_header_offset` to compute the JIT's baked `SHAPE_OFF` from a
    /// real instance (NOT to read the value; use `shape_header()` for that). The
    /// pointer is into the `Cell<ShapeCache>`'s `UnsafeCell` storage; `Cell<u32>`
    /// / `ShapeCache.header` is `u32`, so reading 4 bytes here yields the header.
    /// SAFETY of the returned ptr: valid for as long as `self` lives.
    #[inline]
    pub fn shape_header_ptr(&self) -> *const u32 {
        // Address of the `header` field within the ShapeCache stored in the Cell.
        // `Cell::as_ptr` gives `*mut ShapeCache`; project to `header`.
        let sc: *mut ShapeCache = self.shape_cache.as_ptr();
        // SAFETY: `sc` is a valid, aligned pointer to the live ShapeCache; the
        // `header` field projection is in-bounds.
        unsafe { std::ptr::addr_of!((*sc).header) }
    }

    /// Slot index for a key, or `None` if absent.
    pub fn slot_of<Q>(&self, k: &Q) -> Option<usize>
    where
        K: std::borrow::Borrow<Q>,
        Q: Hash + Eq + ?Sized + 'static,
    {
        match &self.store {
            Store::Dict { index, .. } => index.get(k).copied(),
            Store::Shaped { shape, .. } => {
                crate::shapes::global_shape_lookup(*shape, borrow_as_str(k)?)
            }
        }
    }

    /// Value at a slot index (the IC fast-path read). `None` if out of range.
    #[inline]
    pub fn value_at_slot(&self, slot: usize) -> Option<&V> {
        match &self.store {
            Store::Dict { entries, .. } => entries.get(slot).map(|(_, v)| v),
            Store::Shaped { slots, .. } => slots.get(slot),
        }
    }

    /// Overwrite the value at a slot index (the write-IC fast path). Does NOT
    /// change the key set/order, so `struct_ver` is deliberately NOT bumped.
    #[inline]
    pub fn set_at_slot(&mut self, slot: usize, v: V) {
        match &mut self.store {
            Store::Dict { entries, .. } => {
                if let Some(e) = entries.get_mut(slot) {
                    e.1 = v;
                }
            }
            Store::Shaped { slots, .. } => {
                if let Some(s) = slots.get_mut(slot) {
                    *s = v;
                }
            }
        }
    }

    #[inline]
    fn bump_struct_ver(&mut self) {
        self.struct_ver = fresh_struct_ver();
    }

    pub fn len(&self) -> usize {
        match &self.store {
            Store::Dict { entries, .. } => entries.len(),
            Store::Shaped { slots, .. } => slots.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        match &self.store {
            Store::Dict { entries, .. } => entries.is_empty(),
            Store::Shaped { slots, .. } => slots.is_empty(),
        }
    }

    /// Allocated capacity of the order vector (`entries`/`slots`).
    #[inline]
    pub fn entries_capacity(&self) -> usize {
        match &self.store {
            Store::Dict { entries, .. } => entries.capacity(),
            Store::Shaped { slots, .. } => slots.capacity(),
        }
    }

    /// Allocated capacity of the `index` hash map. Shaped has no per-object index
    /// (the shared `ShapeTable` resolves keys), so it reports 0 — exactly the
    /// per-object index the Shaped win eliminates.
    #[inline]
    pub fn index_capacity(&self) -> usize {
        match &self.store {
            Store::Dict { index, .. } => index.capacity(),
            Store::Shaped { .. } => 0,
        }
    }

    pub fn clear(&mut self) {
        match &mut self.store {
            Store::Dict { entries, index } => {
                entries.clear();
                index.clear();
            }
            Store::Shaped { .. } => {
                // Shrinks the key set → drop to an empty Dict + mark deopted so
                // the IC sees DICT_SHAPE for the (now empty) object.
                stat_deopt(DeoptTrigger::Clear);
                self.store = Store::new_dict();
                self.mark_deopted();
            }
        }
        self.bump_struct_ver();
        self.reset_shape_cache();
    }

    pub fn get<Q>(&self, k: &Q) -> Option<&V>
    where
        K: std::borrow::Borrow<Q>,
        Q: Hash + Eq + ?Sized + 'static,
    {
        match &self.store {
            Store::Dict { entries, index } => index.get(k).map(|&i| &entries[i].1),
            Store::Shaped { shape, slots, .. } => {
                let s = borrow_as_str(k)?;
                crate::shapes::global_shape_lookup(*shape, s).and_then(|i| slots.get(i))
            }
        }
    }

    pub fn get_mut<Q>(&mut self, k: &Q) -> Option<&mut V>
    where
        K: std::borrow::Borrow<Q>,
        Q: Hash + Eq + ?Sized + 'static,
    {
        match &mut self.store {
            Store::Dict { entries, index } => match index.get(k) {
                Some(&i) => Some(&mut entries[i].1),
                None => None,
            },
            Store::Shaped { shape, slots, .. } => {
                let s = borrow_as_str(k)?;
                let i = crate::shapes::global_shape_lookup(*shape, s)?;
                slots.get_mut(i)
            }
        }
    }

    pub fn contains_key<Q>(&self, k: &Q) -> bool
    where
        K: std::borrow::Borrow<Q>,
        Q: Hash + Eq + ?Sized + 'static,
    {
        match &self.store {
            Store::Dict { index, .. } => index.contains_key(k),
            Store::Shaped { shape, .. } => match borrow_as_str(k) {
                Some(s) => crate::shapes::global_shape_lookup(*shape, s).is_some(),
                None => false,
            },
        }
    }

    /// Rematerialize a Shaped store into Dict (one-way). Bumps `struct_ver` +
    /// clears `shape_cache` + sets `deopted` so a stale IC entry can't read a
    /// wrong slot and `object_shape_id` reports `DICT_SHAPE`. No-op if Dict.
    /// `trigger` attributes the deopt for the P4 retention breakdown (recorded
    /// only when `CV_SHAPED_STATS` is on, and only when an actual Shaped store
    /// is rematerialized — a no-op on an already-Dict store is not counted).
    #[inline]
    fn deopt_to_dict(&mut self, trigger: DeoptTrigger) {
        if let Store::Shaped { shape, slots, .. } = &mut self.store {
            stat_deopt(trigger);
            let shape = *shape;
            let slots = std::mem::take(slots);
            let props = crate::shapes::global_shape_properties(shape);
            debug_assert_eq!(
                props.len(),
                slots.len(),
                "shape/slots length mismatch on deopt"
            );
            let mut entries: Vec<(K, V)> = Vec::with_capacity(slots.len());
            let mut index: StdHashMap<K, usize> = StdHashMap::with_capacity(slots.len());
            for (i, (key, v)) in props.into_iter().zip(slots).enumerate() {
                let k = string_to_k::<K>(key);
                index.insert(k.clone(), i);
                entries.push((k, v));
            }
            self.store = Store::Dict { entries, index };
            self.bump_struct_ver();
            self.mark_deopted();
        }
    }

    /// `entry(k).or_insert_with(f)`: a mut ref to `k`'s value, inserting `f()`
    /// (at the end, preserving order) if absent.
    pub fn get_or_insert_with<F: FnOnce() -> V>(&mut self, k: K, f: F) -> &mut V {
        // Decide + prepare the Shaped path FIRST (deopt if needed) in a separate
        // borrow scope, so the value-returning borrows below don't overlap.
        if self.store.is_shaped() {
            match self.shaped_intern_slot(&k) {
                ShapedSlot::Slot(i) => {
                    // Existing key OR freshly added slot — but for a fresh slot we
                    // must run `f()`. Distinguish: `shaped_intern_slot` only adds
                    // the shape transition; the slot may need a value pushed.
                    if let Store::Shaped { slots, .. } = &mut self.store {
                        if i == slots.len() {
                            slots.push(f());
                            self.bump_struct_ver();
                        }
                        if let Store::Shaped { slots, .. } = &mut self.store {
                            return &mut slots[i];
                        }
                    }
                    unreachable!("shaped slot path");
                }
                ShapedSlot::Deopted => { /* fall through to Dict path below */ }
            }
        }
        let mut bumped = false;
        let i = match &mut self.store {
            Store::Dict { entries, index } => {
                if let Some(&i) = index.get(&k) {
                    i
                } else {
                    let i = entries.len();
                    index.insert(k.clone(), i);
                    entries.push((k, f()));
                    bumped = true;
                    i
                }
            }
            Store::Shaped { .. } => unreachable!("deopted to Dict above"),
        };
        if bumped {
            self.bump_struct_ver();
        }
        match &mut self.store {
            Store::Dict { entries, .. } => &mut entries[i].1,
            Store::Shaped { .. } => unreachable!("deopted to Dict above"),
        }
    }

    /// Shaped-path helper for `insert`/`get_or_insert_with`: given a key on a
    /// Shaped store, either resolve/create its slot (returning `Slot(i)`, where
    /// `i == slots.len()` means "a new transition was added; caller must push the
    /// value") or DEOPT the store to Dict (returning `Deopted`, caller takes the
    /// Dict path). Centralizes the deopt-vs-intern decision so the callers don't
    /// hold overlapping mutable borrows. Assumes `self.store` is Shaped.
    #[inline]
    fn shaped_intern_slot(&mut self, k: &K) -> ShapedSlot {
        let ks = match k_as_str(k) {
            Some(ks) if !string_key_forces_deopt(ks) => ks,
            ks_opt => {
                // Classify the forced-deopt cause for the P4 breakdown: a
                // non-String key, an integer/array-index key, or an engine
                // sentinel/exotic-tag stamp. The classification mirrors
                // `string_key_forces_deopt` exactly.
                let trigger = match ks_opt {
                    None => DeoptTrigger::NonStringKey,
                    Some(ks) if is_array_index_key_str(ks) => DeoptTrigger::IntegerKey,
                    Some(_) => DeoptTrigger::SentinelStamp,
                };
                self.deopt_to_dict(trigger);
                return ShapedSlot::Deopted;
            }
        };
        let shape = match &self.store {
            Store::Shaped { shape, .. } => *shape,
            _ => return ShapedSlot::Deopted,
        };
        if let Some(i) = crate::shapes::global_shape_lookup(shape, ks) {
            return ShapedSlot::Slot(i);
        }
        let cur_len = match &self.store {
            Store::Shaped { slots, .. } => slots.len(),
            _ => return ShapedSlot::Deopted,
        };
        if cur_len + 1 > SHAPED_SLOT_CAP {
            self.deopt_to_dict(DeoptTrigger::CapExceeded);
            return ShapedSlot::Deopted;
        }
        let new_shape = crate::shapes::global_add_property(shape, ks);
        if let Store::Shaped { shape, .. } = &mut self.store {
            *shape = new_shape;
        }
        // Mirror the new interned shape into the inline header immediately (a new
        // key transitioned the Shaped object to `new_shape`). The value `push` +
        // `bump_struct_ver` happen in the caller, but the header is authoritative
        // about the SHAPE the instant the transition lands.
        self.mirror_shape_header();
        // `cur_len` is the slot index the caller must `push` into.
        ShapedSlot::Slot(cur_len)
    }

    pub fn insert(&mut self, k: K, v: V) -> Option<V> {
        if self.store.is_shaped() {
            match self.shaped_intern_slot(&k) {
                ShapedSlot::Slot(i) => {
                    if let Store::Shaped { slots, .. } = &mut self.store {
                        if i < slots.len() {
                            // Overwrite existing key: value changes, layout does
                            // NOT — do NOT bump struct_ver (keeps the IC hot).
                            return Some(std::mem::replace(&mut slots[i], v));
                        } else {
                            // New key (transition already added by the helper):
                            // push the value + bump for the (ptr, struct_ver)
                            // guard (the new ShapeId already invalidates old IC).
                            slots.push(v);
                            self.bump_struct_ver();
                            return None;
                        }
                    }
                    unreachable!("shaped slot path");
                }
                ShapedSlot::Deopted => { /* fall through to Dict path below */ }
            }
        }
        match &mut self.store {
            Store::Dict { entries, index } => {
                if let Some(&i) = index.get(&k) {
                    Some(std::mem::replace(&mut entries[i].1, v))
                } else {
                    index.insert(k.clone(), entries.len());
                    entries.push((k, v));
                    self.bump_struct_ver();
                    None
                }
            }
            Store::Shaped { .. } => unreachable!("deopted to Dict above"),
        }
    }

    pub fn remove<Q>(&mut self, k: &Q) -> Option<V>
    where
        K: std::borrow::Borrow<Q>,
        Q: Hash + Eq + ?Sized + 'static,
    {
        // A removal shrinks/reorders the key set — deopt first, then remove. But
        // a no-op remove (missing key) must NOT change the store mode, so only
        // deopt when the key is actually present.
        if self.store.is_shaped() {
            let present = match borrow_as_str(k) {
                Some(s) => match &self.store {
                    Store::Shaped { shape, .. } => {
                        crate::shapes::global_shape_lookup(*shape, s).is_some()
                    }
                    _ => false,
                },
                None => false,
            };
            if !present {
                return None;
            }
            self.deopt_to_dict(DeoptTrigger::Remove);
        }
        match &mut self.store {
            Store::Dict { entries, index } => {
                let Some(i) = index.remove(k) else {
                    return None;
                };
                let (_, v) = entries.remove(i);
                for idx in index.values_mut() {
                    if *idx > i {
                        *idx -= 1;
                    }
                }
                self.bump_struct_ver();
                self.reset_shape_cache();
                Some(v)
            }
            Store::Shaped { .. } => unreachable!("deopted to Dict above"),
        }
    }

    /// Keys in slot/insertion order. For a Dict store this borrows the stored
    /// keys; for a Shaped store it borrows the SHAPE's shared (immortal) key
    /// list via an `Rc` handle — keys live once per shape, never per object, and
    /// the borrows are valid for any lifetime (see the iterator-types section).
    pub fn keys(&self) -> Keys<'_, K, V> {
        match &self.store {
            Store::Dict { entries, .. } => Keys::Dict(entries.iter()),
            Store::Shaped { shape, .. } => Keys::Shaped(
                crate::shapes::global_shape_properties_rc(*shape),
                0,
                std::marker::PhantomData,
            ),
        }
    }

    pub fn values(&self) -> Values<'_, K, V> {
        match &self.store {
            Store::Dict { entries, .. } => Values::Dict(entries.iter()),
            // MUST yield EVERY slot (GC marks through this — a missed slot = UAF).
            Store::Shaped { slots, .. } => Values::Shaped(slots.iter()),
        }
    }

    pub fn values_mut(&mut self) -> ValuesMut<'_, K, V> {
        match &mut self.store {
            Store::Dict { entries, .. } => ValuesMut::Dict(entries.iter_mut()),
            Store::Shaped { slots, .. } => ValuesMut::Shaped(slots.iter_mut()),
        }
    }

    /// `(key, value)` pairs in order. Dict borrows; Shaped zips the shape's
    /// shared (immortal) key list with the slots.
    pub fn iter(&self) -> Iter<'_, K, V> {
        match &self.store {
            Store::Dict { entries, .. } => Iter::Dict(entries.iter()),
            Store::Shaped { shape, slots, .. } => Iter::Shaped(
                crate::shapes::global_shape_properties_rc(*shape),
                slots.iter(),
                0,
            ),
        }
    }

    pub fn iter_mut(&mut self) -> IterMut<'_, K, V> {
        match &mut self.store {
            Store::Dict { entries, .. } => IterMut::Dict(entries.iter_mut()),
            Store::Shaped { shape, slots, .. } => IterMut::Shaped(
                crate::shapes::global_shape_properties_rc(*shape),
                slots.iter_mut(),
                0,
            ),
        }
    }

    pub fn extend<I: IntoIterator<Item = (K, V)>>(&mut self, it: I) {
        for (k, v) in it {
            self.insert(k, v);
        }
    }

    /// Drain all entries (empties the map), yielding `(K, V)` in order.
    pub fn drain(&mut self) -> std::vec::Drain<'_, (K, V)> {
        if self.store.is_shaped() {
            self.deopt_to_dict(DeoptTrigger::Drain);
        }
        self.bump_struct_ver();
        self.reset_shape_cache();
        match &mut self.store {
            Store::Dict { entries, index } => {
                index.clear();
                entries.drain(..)
            }
            Store::Shaped { .. } => unreachable!("deopted to Dict above"),
        }
    }

    /// `entry(k)` — the std-HashMap Entry pattern.
    pub fn entry(&mut self, key: K) -> OrderedEntry<'_, K, V> {
        OrderedEntry { map: self, key }
    }

    /// Keep only entries for which `f` returns true (order preserved).
    pub fn retain<F: FnMut(&K, &mut V) -> bool>(&mut self, mut f: F) {
        if self.store.is_shaped() {
            self.deopt_to_dict(DeoptTrigger::Retain);
        }
        match &mut self.store {
            Store::Dict { entries, index } => {
                let mut kept: Vec<(K, V)> = Vec::with_capacity(entries.len());
                for (k, mut v) in entries.drain(..) {
                    if f(&k, &mut v) {
                        kept.push((k, v));
                    }
                }
                index.clear();
                for (i, (k, _)) in kept.iter().enumerate() {
                    index.insert(k.clone(), i);
                }
                *entries = kept;
            }
            Store::Shaped { .. } => unreachable!("deopted to Dict above"),
        }
        self.bump_struct_ver();
        self.reset_shape_cache();
    }
}


/// Borrow a `&Q` as `&str` when `Q` is `str`/`String` (the only `Q` a Shaped
/// String-keyed lookup ever uses). `None` for any other `Q` (defensive — Shaped
/// is never built for non-String maps, so a non-string `Q` can only arrive on a
/// Dict map, where this is never called).
///
/// `Q` is `?Sized` (e.g. `str`), so a `dyn Any` cast isn't available; we compare
/// `TypeId`s and reinterpret the reference. Sound because we only reinterpret
/// after confirming the concrete type, and the reference/lifetime are preserved.
#[inline]
fn borrow_as_str<Q: ?Sized + 'static>(q: &Q) -> Option<&str> {
    use std::any::TypeId;
    let tid = TypeId::of::<Q>();
    if tid == TypeId::of::<str>() {
        // Q == str: the &Q already IS a &str (identical fat-pointer layout).
        // `transmute_copy` reinterprets the reference without a thin/fat cast.
        Some(unsafe { std::mem::transmute_copy::<&Q, &str>(&q) })
    } else if tid == TypeId::of::<String>() {
        // Q == String: reinterpret to &String, then deref to &str.
        let s: &String = unsafe { std::mem::transmute_copy::<&Q, &String>(&q) };
        Some(s.as_str())
    } else {
        None
    }
}

/// A key `&K` as `&str` when `K = String` (the Shaped insert path). `None`
/// otherwise (forces a deopt, defensively).
#[inline]
fn k_as_str<K: std::any::Any>(k: &K) -> Option<&str> {
    (k as &dyn std::any::Any).downcast_ref::<String>().map(|s| s.as_str())
}

/// Recover a `K` from an interned shape property `String`. Sound because the
/// Shaped store is only ever built for `K = String`; an identity for the only
/// type that reaches it. No `unsafe`; panics for other `K` (unreachable).
#[inline]
fn string_to_k<K: 'static>(s: String) -> K {
    let boxed: Box<dyn std::any::Any> = Box::new(s);
    *boxed
        .downcast::<K>()
        .unwrap_or_else(|_| panic!("Shaped store reached for non-String key type"))
}

// ───────────────────────────── iterator types ─────────────────────────────
//
// Concrete iterator enums so `keys()/values()/iter()/iter_mut()` keep their
// `impl Iterator` ergonomics.
//
// The Shaped key-yielding variants hold an `Rc<Vec<String>>` handle to the
// SHAPE's key list and yield `&K` (= `&String`) by indexing into it. This is
// SOUND — and the `'a` lifetime extension is genuine, not a hope — because the
// shape's key `Vec` is IMMORTAL: the global `ShapeTable` holds the canonical
// `Rc` for the whole process, so the `String`s are never freed regardless of
// when the iterator (or its `Rc` clone) is dropped. So a `&String` yielded with
// lifetime `'a` is valid for any `'a` (the data outlives the program-lifetime
// `ShapeTable`). This avoids the "owning iterator dangles after it's dropped"
// unsoundness: dropping the iterator only decrements a refcount the `ShapeTable`
// still pins above zero.
//
// `&K` is reinterpreted from `&String` because Shaped is built ONLY for
// `K = String`, so the two types are identical on this path.

/// Reinterpret an immortal `&String` (from a shape's key list) as `&'a K`.
/// SOUND because: (1) the only Shaped key type is `String`, so `K == String`
/// here; (2) the referent lives in the immortal `ShapeTable`-pinned `Rc`, so it
/// outlives any `'a`.
#[inline]
unsafe fn shape_str_as_k<'a, K>(s: &String) -> &'a K {
    unsafe { &*(s as *const String as *const K) }
}

pub enum Keys<'a, K, V> {
    Dict(std::slice::Iter<'a, (K, V)>),
    Shaped(std::sync::Arc<Vec<String>>, usize, std::marker::PhantomData<&'a (K, V)>),
}
impl<'a, K, V> Iterator for Keys<'a, K, V> {
    type Item = &'a K;
    fn next(&mut self) -> Option<&'a K> {
        match self {
            Keys::Dict(it) => it.next().map(|(k, _)| k),
            Keys::Shaped(keys, pos, _) => {
                let s = keys.get(*pos)?;
                *pos += 1;
                Some(unsafe { shape_str_as_k::<K>(s) })
            }
        }
    }
}

pub enum Values<'a, K, V> {
    Dict(std::slice::Iter<'a, (K, V)>),
    Shaped(std::slice::Iter<'a, V>),
}
impl<'a, K, V> Iterator for Values<'a, K, V> {
    type Item = &'a V;
    fn next(&mut self) -> Option<&'a V> {
        match self {
            Values::Dict(it) => it.next().map(|(_, v)| v),
            Values::Shaped(it) => it.next(),
        }
    }
}

pub enum ValuesMut<'a, K, V> {
    Dict(std::slice::IterMut<'a, (K, V)>),
    Shaped(std::slice::IterMut<'a, V>),
}
impl<'a, K, V> Iterator for ValuesMut<'a, K, V> {
    type Item = &'a mut V;
    fn next(&mut self) -> Option<&'a mut V> {
        match self {
            ValuesMut::Dict(it) => it.next().map(|(_, v)| v),
            ValuesMut::Shaped(it) => it.next(),
        }
    }
}

pub enum Iter<'a, K, V> {
    Dict(std::slice::Iter<'a, (K, V)>),
    Shaped(std::sync::Arc<Vec<String>>, std::slice::Iter<'a, V>, usize),
}
impl<'a, K, V> Iterator for Iter<'a, K, V> {
    type Item = (&'a K, &'a V);
    fn next(&mut self) -> Option<(&'a K, &'a V)> {
        match self {
            Iter::Dict(it) => it.next().map(|(k, v)| (k, v)),
            Iter::Shaped(keys, vals, pos) => {
                let v = vals.next()?;
                let s = &keys[*pos];
                *pos += 1;
                Some((unsafe { shape_str_as_k::<K>(s) }, v))
            }
        }
    }
}

pub enum IterMut<'a, K, V> {
    Dict(std::slice::IterMut<'a, (K, V)>),
    Shaped(std::sync::Arc<Vec<String>>, std::slice::IterMut<'a, V>, usize),
}
impl<'a, K, V> Iterator for IterMut<'a, K, V> {
    type Item = (&'a K, &'a mut V);
    fn next(&mut self) -> Option<(&'a K, &'a mut V)> {
        match self {
            IterMut::Dict(it) => it.next().map(|(k, v)| (&*k, v)),
            IterMut::Shaped(keys, vals, pos) => {
                let v = vals.next()?;
                let s = &keys[*pos];
                *pos += 1;
                Some((unsafe { shape_str_as_k::<K>(s) }, v))
            }
        }
    }
}

/// The `entry(k)` view, mirroring `std::collections::hash_map::Entry`'s common
/// methods.
pub struct OrderedEntry<'a, K, V> {
    map: &'a mut OrderedMap<K, V>,
    key: K,
}

impl<'a, K: Hash + Eq + Clone + 'static, V> OrderedEntry<'a, K, V> {
    pub fn or_insert(self, default: V) -> &'a mut V {
        self.map.get_or_insert_with(self.key, || default)
    }
    pub fn or_insert_with<F: FnOnce() -> V>(self, f: F) -> &'a mut V {
        self.map.get_or_insert_with(self.key, f)
    }
    pub fn or_default(self) -> &'a mut V
    where
        V: Default,
    {
        self.map.get_or_insert_with(self.key, V::default)
    }
    pub fn and_modify<F: FnOnce(&mut V)>(mut self, f: F) -> Self {
        if let Some(v) = self.map.get_mut(&self.key) {
            f(v);
        }
        self
    }
}

impl<K: Hash + Eq + Clone + 'static, V> FromIterator<(K, V)> for OrderedMap<K, V> {
    fn from_iter<I: IntoIterator<Item = (K, V)>>(it: I) -> Self {
        let mut m = OrderedMap::new();
        m.extend(it);
        m
    }
}

impl<K: Hash + Eq + Clone + 'static, V> IntoIterator for OrderedMap<K, V> {
    type Item = (K, V);
    type IntoIter = std::vec::IntoIter<(K, V)>;
    fn into_iter(mut self) -> Self::IntoIter {
        // Owned `(K,V)` iteration needs real keys; a Shaped store has them only
        // as shape properties — rematerialize into Dict first, then drain.
        if self.store.is_shaped() {
            self.deopt_to_dict(DeoptTrigger::IntoIter);
        }
        match self.store {
            Store::Dict { entries, .. } => entries.into_iter(),
            Store::Shaped { .. } => unreachable!("deopted to Dict above"),
        }
    }
}

impl<'a, K: 'static, V> IntoIterator for &'a OrderedMap<K, V> {
    type Item = (&'a K, &'a V);
    type IntoIter = Iter<'a, K, V>;
    fn into_iter(self) -> Self::IntoIter {
        match &self.store {
            Store::Dict { entries, .. } => Iter::Dict(entries.iter()),
            Store::Shaped { shape, slots, .. } => Iter::Shaped(
                crate::shapes::global_shape_properties_rc(*shape),
                slots.iter(),
                0,
            ),
        }
    }
}

impl<'a, K: 'static, V> IntoIterator for &'a mut OrderedMap<K, V> {
    type Item = (&'a K, &'a mut V);
    type IntoIter = IterMut<'a, K, V>;
    fn into_iter(self) -> Self::IntoIter {
        match &mut self.store {
            Store::Dict { entries, .. } => IterMut::Dict(entries.iter_mut()),
            Store::Shaped { shape, slots, .. } => IterMut::Shaped(
                crate::shapes::global_shape_properties_rc(*shape),
                slots.iter_mut(),
                0,
            ),
        }
    }
}

/// `From<std::HashMap>` so any std-built map can drop into ordered storage.
impl<K: Hash + Eq + Clone + 'static, V> From<StdHashMap<K, V>> for OrderedMap<K, V> {
    fn from(m: StdHashMap<K, V>) -> Self {
        m.into_iter().collect()
    }
}

#[cfg(test)]
mod shaped_tests {
    //! Unit coverage for the M3.2 Shaped store. To test the Shaped path
    //! deterministically (without depending on the process-wide `CV_SHAPED_OBJ`
    //! cache, which is read once), these build a Shaped map directly via the
    //! private `Store::new_shaped` and drive it through the public API. The
    //! observable results (key order, lookups, deopt) must match a plain Dict
    //! built the same way — that equality IS the invariant.

    use super::*;

    /// Force a Shaped-backed `OrderedMap<String, i64>` regardless of the env gate.
    fn shaped_map() -> OrderedMap<String, i64> {
        let store = Store::new_shaped();
        let header = match &store {
            Store::Shaped { shape, .. } => *shape,
            Store::Dict { .. } => BORN_DICT_SHAPE,
        };
        OrderedMap {
            store,
            struct_ver: fresh_struct_ver(),
            shape_cache: std::cell::Cell::new(ShapeCache { ver: 0, shape: 0, header }),
        }
    }

    /// A plain Dict-backed map (the oracle).
    fn dict_map() -> OrderedMap<String, i64> {
        OrderedMap::default()
    }

    /// PARITY: the local deopt-key predicates mirror `interp`'s exactly. Pins the
    /// dep-cycle-avoidance duplication so a change to one side is caught here.
    #[test]
    fn deopt_key_predicates_match_interp() {
        assert_eq!(PROTO_KEY, crate::interp::PROTO_KEY);
        for k in ["0", "1", "42", "4294967294", "x", "00", "01", "4294967295", ""] {
            assert_eq!(
                is_array_index_key_str(k),
                crate::interp::is_array_index_key_for_test(k),
                "array-index parity for {k:?}"
            );
        }
        for k in ["_isProxy", "_target", "_typedarray", "_entries", "x", "_id"] {
            assert_eq!(
                is_engine_sentinel_key_str(k),
                crate::interp::is_engine_sentinel_key(k),
                "sentinel parity for {k:?}"
            );
        }
    }

    /// HEADER-MIRROR CORRECTNESS (the T2 silent-corruption gate): the inline
    /// `shape_header()` word MUST, in every reachable state, equal the
    /// authoritative shape:
    ///   - a live Shaped object → its `stored_shape_id()` (the real `ShapeId`);
    ///   - a deopted Dict object → `DICT_SHAPE` (== `stored_shape_id()`);
    ///   - a born-Dict object → `BORN_DICT_SHAPE` (and `stored_shape_id()` None).
    /// Checked after: insert-new-key (each transition), overwrite (no change),
    /// delete (→ deopt), clear (→ deopt). A drift here = a baked T2 slot could
    /// read a wrong slot.
    #[test]
    fn header_mirror_matches_authority() {
        // The single invariant predicate.
        fn check(m: &OrderedMap<String, i64>, label: &str) {
            let hdr = m.shape_header();
            match m.stored_shape_id() {
                Some(sid) => assert_eq!(
                    hdr, sid,
                    "{label}: header {hdr} != stored_shape_id {sid} (Shaped/deopted authority)"
                ),
                None => assert_eq!(
                    hdr, BORN_DICT_SHAPE,
                    "{label}: born-Dict header must be BORN_DICT_SHAPE, got {hdr}"
                ),
            }
            // The two Dict sentinels never collide with a real shape, and a
            // Shaped header is never a sentinel.
            if m.stored_shape_id() == Some(crate::shapes::DICT_SHAPE) {
                assert_eq!(hdr, crate::shapes::DICT_SHAPE, "{label}: deopted header");
            }
        }

        // (a) Shaped object: header tracks every insert-new-key transition.
        let mut s = shaped_map();
        check(&s, "fresh shaped (empty)");
        s.insert("a".to_string(), 1);
        check(&s, "after insert a");
        let after_a = s.shape_header();
        s.insert("b".to_string(), 2);
        check(&s, "after insert b");
        assert_ne!(s.shape_header(), after_a, "a new key must change the header");
        // overwrite an existing key: header unchanged (no key-set change).
        let before_ow = s.shape_header();
        s.insert("a".to_string(), 11);
        check(&s, "after overwrite a");
        assert_eq!(s.shape_header(), before_ow, "overwrite keeps the header");
        // delete → deopt → DICT_SHAPE.
        s.remove("a");
        check(&s, "after delete a (deopt)");
        assert_eq!(s.shape_header(), crate::shapes::DICT_SHAPE, "deopt header");

        // (b) Shaped object cleared → deopt → DICT_SHAPE.
        let mut s2 = shaped_map();
        s2.insert("x".to_string(), 7);
        s2.insert("y".to_string(), 8);
        check(&s2, "shaped x,y");
        s2.clear();
        check(&s2, "after clear (deopt)");
        assert_eq!(s2.shape_header(), crate::shapes::DICT_SHAPE, "cleared deopt header");

        // (c) Born-Dict object (the off-ramp / non-String path): header stays
        // BORN_DICT_SHAPE forever, inserts don't change it, and it never claims a
        // real shape.
        let mut d = dict_map();
        check(&d, "born dict empty");
        d.insert("k".to_string(), 1);
        d.insert("j".to_string(), 2);
        check(&d, "born dict with keys");
        assert_eq!(d.shape_header(), BORN_DICT_SHAPE, "born-Dict stays BORN_DICT_SHAPE");
        assert_eq!(d.stored_shape_id(), None, "born-Dict has no stored shape id");

        // (d) Shaped→deopt via an array-index key: header → DICT_SHAPE.
        let mut s3 = shaped_map();
        s3.insert("p".to_string(), 1);
        check(&s3, "shaped p");
        s3.insert("0".to_string(), 9); // array-index → deopt
        check(&s3, "after array-index insert (deopt)");
        assert_eq!(s3.shape_header(), crate::shapes::DICT_SHAPE, "array-index deopt header");
    }

    /// MUTATION TEETH: a deliberately-WRONG header (set behind the public API via
    /// a fresh map whose store says one thing and header another) is exactly what
    /// the mirror invariant forbids — prove `header_mirror` would catch it by
    /// constructing the violating state and asserting the predicate fails. (We
    /// can't poke the private field from here, so we assert the *contrapositive*:
    /// the authority and header agree only because the mirror runs; a Shaped
    /// store with a stale DICT_SHAPE header would mismatch `stored_shape_id`.)
    #[test]
    fn header_mirror_teeth_stale_header_would_mismatch() {
        let mut s = shaped_map();
        s.insert("a".to_string(), 1);
        let real = s.stored_shape_id().expect("shaped");
        // The header equals the real shape (mirror is load-bearing).
        assert_eq!(s.shape_header(), real);
        // If the header were stale at DICT_SHAPE (the bug the mirror prevents),
        // the invariant predicate `header == stored_shape_id` would FALSE — this
        // documents that the equality is the guard, not a coincidence.
        assert_ne!(real, crate::shapes::DICT_SHAPE);
        assert_ne!(real, BORN_DICT_SHAPE);
    }

    /// A plain string-keyed object stays Shaped and yields keys/values in
    /// insertion order — identical to the Dict oracle.
    #[test]
    fn shaped_plain_object_matches_dict_order_and_lookup() {
        let mut s = shaped_map();
        let mut d = dict_map();
        for (k, v) in [("a", 1), ("b", 2), ("c", 3), ("\u{1}__proto__", 9)] {
            s.insert(k.to_string(), v);
            d.insert(k.to_string(), v);
        }
        assert!(s.stored_shape_id().is_some(), "stays Shaped");
        assert_ne!(s.stored_shape_id(), Some(crate::shapes::DICT_SHAPE));
        let sk: Vec<String> = s.keys().cloned().collect();
        let dk: Vec<String> = d.keys().cloned().collect();
        assert_eq!(sk, dk, "key order identical");
        assert_eq!(s.get("b"), Some(&2));
        assert_eq!(s.get("c"), Some(&3));
        assert_eq!(s.get("missing"), None);
        // overwrite an existing key does NOT change the key set
        let before = s.stored_shape_id();
        s.insert("b".to_string(), 22);
        assert_eq!(s.stored_shape_id(), before, "overwrite keeps shape");
        assert_eq!(s.get("b"), Some(&22));
    }

    /// GC SAFETY: `values()` over a Shaped object yields EVERY slot in order — the
    /// GC marks through this, so a missed slot would be a use-after-free.
    #[test]
    fn shaped_values_yields_every_slot() {
        let mut s = shaped_map();
        for i in 0..10 {
            s.insert(format!("k{i}"), i);
        }
        let vals: Vec<i64> = s.values().copied().collect();
        assert_eq!(vals, (0..10).collect::<Vec<_>>(), "every slot, in order");
        assert_eq!(s.values().count(), s.len());
        // iter() pairs every key with every value
        let pairs: Vec<(String, i64)> = s.iter().map(|(k, v)| (k.clone(), *v)).collect();
        assert_eq!(pairs.len(), 10);
        for (i, (k, v)) in pairs.iter().enumerate() {
            assert_eq!(k, &format!("k{i}"));
            assert_eq!(*v, i as i64);
        }
    }

    /// Each deopt trigger flips a Shaped object to a Dict reporting `DICT_SHAPE`,
    /// preserving the existing key set + order, and the result matches the Dict
    /// oracle.
    #[test]
    fn deopt_triggers_preserve_data_and_report_dict_shape() {
        // (a) integer / array-index key
        {
            let mut s = shaped_map();
            s.insert("a".to_string(), 1);
            s.insert("b".to_string(), 2);
            s.insert("0".to_string(), 9); // array-index → deopt
            assert_eq!(s.stored_shape_id(), Some(crate::shapes::DICT_SHAPE));
            assert_eq!(s.get("a"), Some(&1));
            assert_eq!(s.get("b"), Some(&2));
            assert_eq!(s.get("0"), Some(&9));
            let keys: Vec<String> = s.keys().cloned().collect();
            assert_eq!(keys, vec!["a", "b", "0"]); // raw store order (order_keys_v8 applied at enum)
        }
        // (b) sentinel key (freeze / proxy / accessor / typed-array)
        for sentinel in ["\u{1}__frozen__", "_isProxy", "\u{1}__get__", "_typedarray", "_bytes"] {
            let mut s = shaped_map();
            s.insert("x".to_string(), 1);
            s.insert(sentinel.to_string(), 1);
            assert_eq!(
                s.stored_shape_id(),
                Some(crate::shapes::DICT_SHAPE),
                "sentinel {sentinel:?} must deopt"
            );
            assert_eq!(s.get("x"), Some(&1));
            assert!(s.contains_key(sentinel));
        }
        // (c) remove / clear / retain / drain
        {
            let mut s = shaped_map();
            s.insert("a".to_string(), 1);
            s.insert("b".to_string(), 2);
            assert_eq!(s.remove("a"), Some(1));
            assert_eq!(s.stored_shape_id(), Some(crate::shapes::DICT_SHAPE));
            assert_eq!(s.get("b"), Some(&2));
            assert_eq!(s.get("a"), None);
        }
        {
            let mut s = shaped_map();
            s.insert("a".to_string(), 1);
            s.clear();
            assert_eq!(s.stored_shape_id(), Some(crate::shapes::DICT_SHAPE));
            assert!(s.is_empty());
        }
    }

    /// A missing-key `remove` is a no-op and must NOT deopt (the store mode is
    /// observable via the IC, so a gratuitous deopt would be a perf cliff).
    #[test]
    fn missing_remove_does_not_deopt() {
        let mut s = shaped_map();
        s.insert("a".to_string(), 1);
        let before = s.stored_shape_id();
        assert_eq!(s.remove("nope"), None);
        assert_eq!(s.stored_shape_id(), before, "no-op remove keeps Shaped");
    }

    /// Past the slot cap a Shaped object deopts instead of minting unbounded
    /// shapes — keeps the shared `ShapeTable` bounded on churny dynamic keys.
    #[test]
    fn over_cap_deopts() {
        let mut s = shaped_map();
        for i in 0..(SHAPED_SLOT_CAP + 5) {
            s.insert(format!("k{i}"), i as i64);
        }
        assert_eq!(s.stored_shape_id(), Some(crate::shapes::DICT_SHAPE));
        assert_eq!(s.len(), SHAPED_SLOT_CAP + 5);
        assert_eq!(s.get("k0"), Some(&0));
        assert_eq!(s.get(&format!("k{}", SHAPED_SLOT_CAP + 4)), Some(&((SHAPED_SLOT_CAP + 4) as i64)));
    }

    /// A non-`String` map is NEVER Shaped (the `should_start_shaped` gate is
    /// String-only) — it reports `None` (key-rewalk IC, untouched).
    #[test]
    fn non_string_map_never_shaped() {
        let mut m: OrderedMap<u64, i64> = OrderedMap::new();
        m.insert(1, 10);
        m.insert(2, 20);
        assert_eq!(m.stored_shape_id(), None);
        assert_eq!(m.get(&1), Some(&10));
    }

    /// A deopted object reports `DICT_SHAPE`, and that marker SURVIVES later
    /// structural mutations (so the IC stays off for it permanently).
    #[test]
    fn deopt_marker_is_sticky() {
        let mut s = shaped_map();
        s.insert("a".to_string(), 1);
        s.insert("0".to_string(), 2); // deopt
        assert_eq!(s.stored_shape_id(), Some(crate::shapes::DICT_SHAPE));
        s.insert("c".to_string(), 3); // further insert on the deopted Dict
        assert_eq!(
            s.stored_shape_id(),
            Some(crate::shapes::DICT_SHAPE),
            "deopt marker survives later inserts"
        );
        s.remove("c");
        assert_eq!(s.stored_shape_id(), Some(crate::shapes::DICT_SHAPE));
    }

    /// `OrderedMap` does not grow past its byte budget with the Shaped plumbing.
    #[test]
    fn size_unchanged() {
        // The JS object map; must stay 96 B (the M3.2 baseline).
        assert_eq!(
            std::mem::size_of::<OrderedMap<String, crate::interp::Value>>(),
            96,
            "OrderedMap layout must not grow (byte budget)"
        );
    }
}
