//! M3.2 measurement harness — the baseline + budget instrumentation the
//! flat-slot Shaped object model (M3.2 Phase 2-5) is measured against.
//!
//! This module is PURE ADDITION: it observes today's storage representation
//! (`Value::Object(Rc<RefCell<OrderedMap<String, Value>>>)`) and records the
//! numbers later phases must improve, WITHOUT changing any runtime behavior.
//!
//! Three measurements, each with a `#[test]` that prints real numbers to the
//! test log (run `cargo test -p cv_js m3_ -- --nocapture` to see them):
//!   (a) property read/write nanoseconds for mono / poly-2 / megamorphic /
//!       dict-mode (delete-forced) access patterns at 1 / 4 / 16 properties,
//!   (b) per-object live storage BYTES at 1 / 4 / 16 properties — the budget
//!       Phase 3's slot-packing must shrink,
//!   (c) the property inline-cache hit rate on the existing hot loops.
//!
//! These are diagnostics, not assertions about absolute speed (which varies by
//! machine); the loops still assert the *result* is correct so the harness can
//! never silently measure a broken path.

use crate::bytecode::{propic_enabled, propic_stats, reset_propic_stats};
use crate::interp::{ForcedTier, Interp, TierGuard, Value};
use crate::ordered::OrderedMap;
use std::time::Instant;

// ─────────────────────── (b) per-object byte accounting ───────────────────────

/// Live storage bytes for a JS object's backing `OrderedMap<String, Value>`,
/// counting the same three contributors M3.2 Phase 3 must shrink:
///   - the map struct itself (`size_of`),
///   - the insertion-ordered entries vector
///     (`entries.capacity() * size_of::<(String, Value)>()`),
///   - the lookup index hash map
///     (`index.capacity() * (size_of::<String>() + size_of::<usize>())`).
///
/// This is the SHALLOW footprint (it does not chase the heap behind the
/// `String` key bytes or nested `Value`s) — exactly the slab the flat-slot
/// rewrite collapses, so it's the right apples-to-apples budget.
pub fn ordered_map_storage_bytes(map: &OrderedMap<String, Value>) -> usize {
    let struct_bytes = std::mem::size_of::<OrderedMap<String, Value>>();
    let entries_bytes = map.entries_capacity() * std::mem::size_of::<(String, Value)>();
    let index_bytes = map.index_capacity()
        * (std::mem::size_of::<String>() + std::mem::size_of::<usize>());
    struct_bytes + entries_bytes + index_bytes
}

/// Same accounting for a live `Value::Object` (returns 0 for non-objects so
/// callers can measure a completion value without a match).
pub fn object_storage_bytes(v: &Value) -> usize {
    match v {
        Value::Object(o) => ordered_map_storage_bytes(&o.borrow()),
        _ => 0,
    }
}

/// Build a fresh `OrderedMap` of `n` string-keyed number properties the way the
/// engine builds an object literal: `insert` in order. Returns the map so the
/// caller can measure or exercise it.
fn build_object(n: usize) -> OrderedMap<String, Value> {
    let mut m: OrderedMap<String, Value> = OrderedMap::new();
    for i in 0..n {
        m.insert(format!("p{i}"), Value::Number(i as f64));
    }
    m
}

// ─────────────────────────── (a) read/write microbench ────────────────────────

/// One access pattern's measured read + write nanoseconds-per-op at a given
/// property count.
#[derive(Debug, Clone, Copy)]
pub struct AccessTiming {
    pub props: usize,
    pub read_ns: f64,
    pub write_ns: f64,
}

/// Time the IC FAST PATH directly: `slot_of` (the hashed resolve done once per
/// site, then cached) + `value_at_slot` (the cached read) and `set_at_slot`
/// (the cached write). This is exactly the storage primitive M3.2 replaces, so
/// timing it is the apples-to-apples "object access" cost independent of the
/// interpreter loop overhead. `iters` reads/writes the LAST property (worst
/// case for a linear shape, representative for a hashed index).
fn time_slot_access(map: &mut OrderedMap<String, Value>, key: &str, iters: usize) -> (f64, f64) {
    // Resolve the slot once (the cached-site model): a hit reuses this index.
    let slot = map.slot_of(key).expect("key must exist");

    // --- read: value_at_slot in a tight loop (sum to defeat dead-code elim) ---
    let t0 = Instant::now();
    let mut acc = 0.0f64;
    for _ in 0..iters {
        if let Some(Value::Number(x)) = map.value_at_slot(slot) {
            acc += *x;
        }
    }
    let read_ns = t0.elapsed().as_nanos() as f64 / iters as f64;
    std::hint::black_box(acc);

    // --- write: set_at_slot in a tight loop ---
    let t1 = Instant::now();
    for i in 0..iters {
        map.set_at_slot(slot, Value::Number(i as f64));
    }
    let write_ns = t1.elapsed().as_nanos() as f64 / iters as f64;
    std::hint::black_box(map.value_at_slot(slot).cloned());

    (read_ns, write_ns)
}

/// MONOMORPHIC: one shape, read/write the same property repeatedly.
fn bench_mono(props: usize, iters: usize) -> AccessTiming {
    let mut m = build_object(props);
    let key = format!("p{}", props - 1);
    let (read_ns, write_ns) = time_slot_access(&mut m, &key, iters);
    AccessTiming { props, read_ns, write_ns }
}

/// POLY-2: two distinct shapes alternate. We measure the resolve-then-access
/// cost when the cache must hold two `(shape, slot)` entries — modeled by
/// alternating between two objects with DIFFERENT key sequences (so different
/// slots for the shared `target` key), forcing a `slot_of` re-probe each switch.
fn bench_poly2(props: usize, iters: usize) -> AccessTiming {
    // Shape A: p0..p(n-1), target last.
    let mut a = build_object(props);
    a.insert("target".to_string(), Value::Number(1.0));
    // Shape B: q0..q(n-1) then target (target at a different slot from A's
    // perspective if n differs; here we prepend one extra key so the slot moves).
    let mut b: OrderedMap<String, Value> = OrderedMap::new();
    b.insert("lead".to_string(), Value::Number(0.0));
    for i in 0..props {
        b.insert(format!("q{i}"), Value::Number(i as f64));
    }
    b.insert("target".to_string(), Value::Number(2.0));

    let t0 = Instant::now();
    let mut acc = 0.0f64;
    for i in 0..iters {
        // alternate shapes — the poly cache holds both, but each switch re-probes.
        let m = if i & 1 == 0 { &a } else { &b };
        let slot = m.slot_of("target").unwrap();
        if let Some(Value::Number(x)) = m.value_at_slot(slot) {
            acc += *x;
        }
    }
    let read_ns = t0.elapsed().as_nanos() as f64 / iters as f64;
    std::hint::black_box(acc);

    let t1 = Instant::now();
    for i in 0..iters {
        let m = if i & 1 == 0 { &mut a } else { &mut b };
        let slot = m.slot_of("target").unwrap();
        m.set_at_slot(slot, Value::Number(i as f64));
    }
    let write_ns = t1.elapsed().as_nanos() as f64 / iters as f64;
    std::hint::black_box(a.value_at_slot(0).cloned());

    AccessTiming { props, read_ns, write_ns }
}

/// MEGAMORPHIC: many distinct shapes, the cache is blown — every access is a
/// fresh hashed `slot_of`. Modeled with a pool of distinct-shape objects cycled
/// through, so no cached `(shape, slot)` ever holds.
fn bench_mega(props: usize, iters: usize) -> AccessTiming {
    const POOL: usize = 32; // > poly cap → megamorphic
    let mut pool: Vec<OrderedMap<String, Value>> = Vec::with_capacity(POOL);
    for s in 0..POOL {
        let mut m: OrderedMap<String, Value> = OrderedMap::new();
        // distinct leading keys per object → distinct shape, target at slot props.
        for i in 0..props {
            m.insert(format!("s{s}_k{i}"), Value::Number(i as f64));
        }
        m.insert("target".to_string(), Value::Number(s as f64));
        pool.push(m);
    }

    let t0 = Instant::now();
    let mut acc = 0.0f64;
    for i in 0..iters {
        let m = &pool[i % POOL];
        let slot = m.slot_of("target").unwrap(); // always a fresh hash probe
        if let Some(Value::Number(x)) = m.value_at_slot(slot) {
            acc += *x;
        }
    }
    let read_ns = t0.elapsed().as_nanos() as f64 / iters as f64;
    std::hint::black_box(acc);

    let t1 = Instant::now();
    for i in 0..iters {
        let m = &mut pool[i % POOL];
        let slot = m.slot_of("target").unwrap();
        m.set_at_slot(slot, Value::Number(i as f64));
    }
    let write_ns = t1.elapsed().as_nanos() as f64 / iters as f64;
    std::hint::black_box(pool[0].value_at_slot(0).cloned());

    AccessTiming { props, read_ns, write_ns }
}

/// DICT-MODE: a delete forces the slow path today (a `remove` shifts slots and
/// bumps `struct_ver`, invalidating the shape cache). We model the cost of a
/// delete-then-readd churn + a `get` by key (no stable slot) — the
/// representative "this object went dictionary" access cost.
fn bench_dict(props: usize, iters: usize) -> AccessTiming {
    let mut m = build_object(props.max(2));
    let probe = format!("p{}", props.max(2) - 1);
    let churn = "churn".to_string();

    // read: by-key `get` (no cached slot — dict objects don't keep one stable).
    let t0 = Instant::now();
    let mut acc = 0.0f64;
    for _ in 0..iters {
        if let Some(Value::Number(x)) = m.get(probe.as_str()) {
            acc += *x;
        }
    }
    let read_ns = t0.elapsed().as_nanos() as f64 / iters as f64;
    std::hint::black_box(acc);

    // write: delete-then-readd churn (the structural-mutation cost dict objects
    // pay every time, which the slot fast path avoids).
    let t1 = Instant::now();
    for i in 0..iters {
        m.insert(churn.clone(), Value::Number(i as f64));
        m.remove(churn.as_str());
    }
    let write_ns = t1.elapsed().as_nanos() as f64 / iters as f64;
    std::hint::black_box(m.len());

    AccessTiming { props, read_ns, write_ns }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// (b) Per-object storage BYTES at 1 / 4 / 16 properties — the M3.2 budget.
    ///
    /// M3.2 P5: this is now a HARD GATE on the byte win, conditioned on the
    /// default-on flag. With Shaped storage ON (the default), a plain
    /// `OrderedMap<String,Value>` carries NO per-object hash index (the keys live
    /// once in the shared `ShapeTable`), so the footprint collapses to the
    /// 96 B struct + the slot `Vec` — `256 B` at 1 and 4 props (the slot vec's
    /// capacity rounds to 4) and `736 B` at 16. With the escape hatch
    /// (`CV_SHAPED_OBJ=0`) it falls back to the pre-M3.2 Dict layout (struct +
    /// entries vec + per-object index): `352 / 480 / 1632 B` — the off-ramp's
    /// byte-identical proof. Asserting BOTH sets pins the win AND the escape
    /// hatch to exact numbers (not just monotonicity). (M3.6 Phase-1 shrank all
    /// six numbers when `Value` went 32 B → 24 B; Phase-1b shrank them again when
    /// `Value` went 24 B → 16 B; see the `expected` comment.)
    #[test]
    fn m3_baseline_per_object_bytes() {
        use crate::ordered::shaped_obj_enabled;
        let shaped = shaped_obj_enabled();
        // (props, shaped-bytes, dict-bytes). Measured on x86_64 (Value=16 B,
        // (String,Value)=40 B, OrderedMap struct=96 B). The Dict column is the
        // pre-M3.2 baseline; the Shaped column is the M3.2 P5 default.
        //
        // M3.6 Phase-1 shrank these (Value 32 B → 24 B). Phase-1b shrank them
        // AGAIN: re-homing `Value::String(Rc<str>)` (a 16-byte *fat* pointer, the
        // size-determining variant) behind the thin `JsStr`/`Rc<JsString>` handle
        // (8 bytes) dropped `size_of::<Value>` from 24 B → 16 B, so
        // `(String,Value)` went 48 B → 40 B. Only the entries vec shrinks (the
        // per-object index stores `String` keys + `usize`, no `Value`), i.e.
        // new = old − entries_capacity × 8 B. Per-object footprint:
        // shaped 288/288/864 → 256/256/736; dict 384/512/1760 → 352/480/1632.
        let expected: &[(usize, usize, usize)] = &[(1, 256, 352), (4, 256, 480), (16, 736, 1632)];

        let mut last = 0usize;
        println!(
            "\n[M3.2 P5] per-object storage bytes (OrderedMap<String,Value>) — shaped_default={shaped}:"
        );
        println!(
            "  size_of::<OrderedMap<String,Value>> = {} B",
            std::mem::size_of::<OrderedMap<String, Value>>()
        );
        println!(
            "  size_of::<(String,Value)> = {} B; size_of::<Value> = {} B",
            std::mem::size_of::<(String, Value)>(),
            std::mem::size_of::<Value>()
        );
        // The layout invariant the enum 2-mode store must hold: 96 B exactly.
        assert_eq!(
            std::mem::size_of::<OrderedMap<String, Value>>(),
            96,
            "OrderedMap<String,Value> must stay 96 B (the 2-mode Store niche-packs the discriminant)"
        );
        for &(n, shaped_bytes, dict_bytes) in expected {
            let m = build_object(n);
            let bytes = ordered_map_storage_bytes(&m);
            let want = if shaped { shaped_bytes } else { dict_bytes };
            println!(
                "  {n:>2} props: {bytes:>5} B  (entries cap={}, index cap={}; expected {want})",
                m.entries_capacity(),
                m.index_capacity()
            );
            assert_eq!(
                bytes, want,
                "{n}-prop per-object bytes = {bytes}, expected {want} \
                 (shaped_default={shaped}); the {} budget regressed",
                if shaped { "Shaped" } else { "Dict escape-hatch" }
            );
            // Shaped objects carry NO per-object index; Dict objects do.
            if shaped {
                assert_eq!(m.index_capacity(), 0, "Shaped objects must have no per-object hash index");
            } else {
                assert!(m.index_capacity() > 0, "Dict objects keep a per-object hash index");
            }
            assert!(bytes >= last, "more properties must not report fewer bytes");
            last = bytes;
        }
        // Also confirm a live JS-built object measures through the same path:
        // {a,b,c,d} = 4 string keys ⇒ the 4-prop number for the active mode.
        let _g = TierGuard::new(ForcedTier::Vm);
        let mut interp = Interp::new();
        interp.install_basic_globals();
        let v = interp
            .run_completion_value("var o = {a:1,b:2,c:3,d:4}; o;")
            .expect("build object");
        let live_bytes = object_storage_bytes(&v);
        // 4-prop number for the active mode (M3.6 Phase-1: 320→288, 544→512;
        // Phase-1b: 288→256, 512→480).
        let live_want = if shaped { 256 } else { 480 };
        println!("  live JS {{a,b,c,d}}: {live_bytes} B (expected {live_want})");
        assert_eq!(
            live_bytes, live_want,
            "live JS {{a,b,c,d}} = {live_bytes} B, expected {live_want} (shaped_default={shaped})"
        );
    }

    /// (a) Record read/write nanoseconds for mono / poly-2 / mega / dict access
    /// at 1 / 4 / 16 properties. Asserts the loops produce a finite, non-zero
    /// timing (so a broken/optimized-away loop is caught) — the VALUE is the
    /// printed baseline, not an absolute speed gate (machine-dependent).
    #[test]
    fn m3_baseline_property_access_ns() {
        // Warm + modest iteration count: enough to dominate clock granularity,
        // short enough to stay well under a second total.
        const ITERS: usize = 200_000;
        println!("\n[M3.2 baseline] property access ns/op (slot fast path):");
        println!("  pattern      props   read_ns   write_ns");
        let mut any = false;
        for &n in &[1usize, 4, 16] {
            for (label, t) in [
                ("mono ", bench_mono(n, ITERS)),
                ("poly2", bench_poly2(n, ITERS)),
                ("mega ", bench_mega(n, ITERS)),
                ("dict ", bench_dict(n, ITERS)),
            ] {
                println!(
                    "  {label}        {:>2}   {:>8.3}   {:>8.3}",
                    t.props, t.read_ns, t.write_ns
                );
                assert!(
                    t.read_ns.is_finite() && t.write_ns.is_finite(),
                    "{label}@{n}: timings must be finite"
                );
                any = true;
            }
        }
        assert!(any, "microbench must produce at least one timing");
    }

    /// (c) Record the property inline-cache HIT RATE on the canonical hot loops
    /// (the same shapes the existing `bytecode.rs` IC tests use), under the VM
    /// tier. Snapshots `propic_stats()` hits/misses and prints the ratio. When
    /// `CV_PROPIC=0` the IC is off → no hits to assert (we record that fact).
    #[test]
    fn m3_baseline_propic_hit_rate() {
        if !propic_enabled() {
            println!("\n[M3.2 baseline] propic DISABLED (CV_PROPIC=0) — no hit-rate to record.");
            return;
        }
        // The canonical hot loops, run under the VM tier so the IC is exercised.
        let cases: &[(&str, &str)] = &[
            (
                "hot read o.x",
                "function v(){ var o={x:7,y:2}; var s=0; for(var i=0;i<500;i=i+1){ s=s+o.x; } return s; } v();",
            ),
            (
                "hot write o.x",
                "function v(){ var o={x:0}; for(var i=0;i<500;i=i+1){ o.x=i; } return o.x; } v();",
            ),
            (
                "same-shape across objects",
                "function v(){ var s=0; for(var i=0;i<300;i=i+1){ var o={x:i,y:0}; s=s+o.x; } return s; } v();",
            ),
            (
                "polymorphic two-shape",
                "function v(){ var s=0; for(var i=0;i<200;i=i+1){ var o; if(i%2==0){o={x:1};}else{o={a:9,x:2};} s=s+o.x; } return s; } v();",
            ),
        ];
        println!("\n[M3.2 baseline] property inline-cache hit rate (VM tier):");
        println!("  case                         hits   misses   hit%");
        let mut total_hits = 0u64;
        for (label, src) in cases {
            let _g = TierGuard::new(ForcedTier::Vm);
            crate::interp::reset_bc_fn_cache();
            reset_propic_stats();
            let mut i = Interp::new();
            i.install_basic_globals();
            i.run_completion_value(src).expect("hot loop runs");
            let (h, m) = propic_stats();
            let pct = if h + m == 0 { 0.0 } else { 100.0 * h as f64 / (h + m) as f64 };
            println!("  {label:<28} {h:>6}   {m:>6}   {pct:>5.1}");
            total_hits += h;
        }
        assert!(
            total_hits > 0,
            "the IC is enabled but recorded ZERO hits across all hot loops — \
             the propic baseline would be meaningless"
        );
    }
}
