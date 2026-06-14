//! T2 Phase 2 — JIT BANK GC-ROOT REGISTRY: the UAF/GC-soak oracle.
//!
//! This is the existential safety gate for the upcoming T2 heap fast paths (P3
//! owning + heap slots, P4 call inlining). Those phases will store HEAP `JsVal`s
//! (object/array/string/bigint pointers) in the JIT register bank. The KEYSTONE
//! HAZARD: the cycle collector's `gc_sweep` CLEARS (empties) any registered
//! container that is not in the mark set even when its `Rc` strong count is > 0
//! (clear-not-free design). So an owning bank slot is NOT enough — a
//! bank-only-reachable heap value would be silently emptied out from under the
//! JIT. The fix proven here: the bank's pointer-lane slots are registered as GC
//! ROOTS via `register_jit_bank`, and `gc_collect` seeds the marker from them.
//!
//! This file is an INTEGRATION test (its own process) so the global `OnceLock`
//! GC gate (`gc_enabled`) is deterministic: it pins `CV_GC=1` at the very top
//! before any code path reads the lock.
//!
//! What it proves:
//!   * POSITIVE SOAK: a registered bank holding the sole (bank-only) reference to
//!     each heap lane (Object / Array / String / BigInt) SURVIVES `gc_collect` —
//!     `gc_sweep` does NOT clear/empty it — and unboxes to the IDENTICAL `Rc`
//!     (`ptr_eq`) with correct content afterward.
//!   * NEGATIVE CONTROL: the SAME setup WITHOUT registering the bank → the
//!     bank-only Object/Array is CLEARED (emptied) by the sweep — confirming the
//!     positive test's safety comes from registration, not luck.
//!   * RAII GAP-FREE: the registration is popped on normal return, on `?`
//!     early-return, and on panic-unwind.
//!   * NO FALSE ROOTS: `is_pointer()` roots only pointer lanes; an all-immediate
//!     bank seeds nothing.
//!   * CHURN BOUNDED: register/collect/unregister many times keeps the GC live
//!     count bounded (no leak), registry returns to empty.
//!
//! Run with output:
//!   cargo test -p cv_js --test jit_bank_gc_root_p2 -- --nocapture

use std::cell::RefCell;
use std::rc::Rc;

use cv_js::interp::{
    gc_enabled, gc_live_object_count, gc_register_array, gc_register_object, jit_bank_registry_len,
    register_jit_bank, Interp, Value,
};
use cv_js::jsval::JsVal;
use cv_js::ordered::OrderedMap;

/// Pin the GC ON before ANY code touches its `OnceLock`. Edition 2024: `set_var`
/// is `unsafe`; sound because this runs before any other thread reads the env in
/// this single-threaded test binary.
fn pin_gc_on() {
    unsafe {
        std::env::set_var("CV_GC", "1");
    }
    assert!(gc_enabled(), "GC must be enabled for the soak oracle");
}

// ----------------------------------------------------------------------------
// Heap-value builders that mirror what a future T2 heap fast path would stash in
// the bank. Each returns (the boxed JsVal bits to put in the bank, an OWNING Rc
// clone kept alive to model the future owning bank slot — so the pointee is not
// freed). The owning clone is reachable ONLY through the test's local, never
// through a GC root, so `gc_sweep` would clear it unless the bank is registered.
// ----------------------------------------------------------------------------

/// A GC-registered Object with one property. The container is what `gc_sweep`
/// would `.clear()` if it were unreachable from the roots.
fn make_object() -> (JsVal, Rc<RefCell<OrderedMap<String, Value>>>) {
    let mut map: OrderedMap<String, Value> = OrderedMap::new();
    map.insert("kept".to_string(), Value::Number(42.0));
    let rc = Rc::new(RefCell::new(map));
    // Register it so the sweep TRACKS it (and would clear it if unmarked).
    gc_register_object(&rc);
    let jv = JsVal::object(&rc);
    assert!(jv.is_object() && jv.is_pointer());
    (jv, rc)
}

/// A GC-registered Array with one element.
fn make_array() -> (JsVal, Rc<RefCell<Vec<Value>>>) {
    let rc = Rc::new(RefCell::new(vec![Value::Number(7.0)]));
    gc_register_array(&rc);
    let jv = JsVal::array(&rc);
    assert!(jv.is_array() && jv.is_pointer());
    (jv, rc)
}

/// A String heap value. Strings are not tracked by `gc_sweep` (cannot be
/// cleared), but rooting must still handle the STRBIG lane (discriminator bit)
/// and round-trip to the identical Rc.
fn make_string() -> (JsVal, Value) {
    let v = Value::str("survive me");
    let jv = JsVal::try_from_value(&v).expect("String is JsVal-representable");
    assert!(jv.is_string() && jv.is_pointer());
    (jv, v)
}

/// A BigInt heap value (STRBIG lane, discriminator clear).
fn make_bigint() -> (JsVal, Value) {
    // Build via the public string parser (the owned-JsBigInt constructors are
    // crate-private); Value::bigint wraps it in the Rc the variant stores.
    let big = cv_js::interp::JsBigInt::parse_str("1234567890123").expect("parses");
    let v = Value::bigint(big);
    let jv = JsVal::try_from_value(&v).expect("BigInt is JsVal-representable");
    assert!(jv.is_bigint() && jv.is_pointer());
    (jv, v)
}

// ----------------------------------------------------------------------------
// POSITIVE SOAK ORACLE
// ----------------------------------------------------------------------------

#[test]
fn soak_registered_bank_survives_gc_all_lanes() {
    pin_gc_on();
    let interp = Interp::new();

    // Build a heap value of each lane. We keep the owning Rc/Value clones in
    // `owners` (modeling the future owning bank slot) so the pointees are not
    // freed; CRUCIALLY they are reachable ONLY through this local, never through
    // a GC root — so `gc_sweep` WOULD clear the Object/Array unless the bank is a
    // registered root.
    let (obj_jv, obj_rc) = make_object();
    let (arr_jv, arr_rc) = make_array();
    let (str_jv, str_v) = make_string();
    let (big_jv, big_v) = make_bigint();

    // Record original pointer identities for the ptr_eq check after GC.
    let obj_ptr = Rc::as_ptr(&obj_rc) as *const () as usize;
    let arr_ptr = Rc::as_ptr(&arr_rc) as *const () as usize;
    let str_ptr = match &str_v {
        Value::String(s) => Rc::as_ptr(s.as_rc()) as *const () as usize,
        _ => unreachable!(),
    };
    let big_ptr = match &big_v {
        Value::BigInt(b) => Rc::as_ptr(b) as *const () as usize,
        _ => unreachable!(),
    };

    // Build the BANK: a Vec<JsVal> sized ONCE (the no-grow invariant) holding the
    // heap JsVals plus an immediate, exactly as a T2 heap fast path would.
    let bank: Vec<JsVal> = vec![
        obj_jv,
        arr_jv,
        str_jv,
        big_jv,
        JsVal::number(99.0), // an immediate — must NOT be mis-rooted
    ];

    // Register the bank as a GC root for the guard's lifetime.
    let guard = register_jit_bank(&bank);
    assert_eq!(jit_bank_registry_len(), 1, "bank registered");

    // Run the collector. The ONLY thing keeping the Object/Array reachable is the
    // bank registration (they are not in the global / external_roots).
    let _cleared = interp.gc_collect(&[]);

    // SURVIVAL: the Object/Array were NOT cleared — content intact.
    assert!(
        matches!(obj_rc.borrow().get("kept"), Some(Value::Number(n)) if *n == 42.0),
        "registered Object was CLEARED by gc_sweep (UAF defense FAILED)"
    );
    {
        let a = arr_rc.borrow();
        assert!(
            a.len() == 1 && matches!(&a[0], Value::Number(n) if *n == 7.0),
            "registered Array was CLEARED by gc_sweep (UAF defense FAILED)"
        );
    }

    // IDENTITY + CONTENT: each bank slot still unboxes to the IDENTICAL Rc and
    // correct content after the collect.
    {
        let back = unsafe { bank[0].to_value() };
        match back {
            Value::Object(o) => {
                assert_eq!(Rc::as_ptr(&o) as *const () as usize, obj_ptr, "Object ptr_eq");
                assert!(matches!(o.borrow().get("kept"), Some(Value::Number(n)) if *n == 42.0));
            }
            other => panic!("slot0 not Object: {other:?}"),
        }
    }
    {
        let back = unsafe { bank[1].to_value() };
        match back {
            Value::Array(a) => {
                assert_eq!(Rc::as_ptr(&a) as *const () as usize, arr_ptr, "Array ptr_eq");
                let a = a.borrow();
                assert!(a.len() == 1 && matches!(&a[0], Value::Number(n) if *n == 7.0));
            }
            other => panic!("slot1 not Array: {other:?}"),
        }
    }
    {
        let back = unsafe { bank[2].to_value() };
        match back {
            Value::String(s) => {
                assert_eq!(
                    Rc::as_ptr(s.as_rc()) as *const () as usize,
                    str_ptr,
                    "String ptr_eq"
                );
                assert_eq!(&*s, "survive me");
            }
            other => panic!("slot2 not String: {other:?}"),
        }
    }
    {
        let back = unsafe { bank[3].to_value() };
        match back {
            Value::BigInt(b) => {
                assert_eq!(Rc::as_ptr(&b) as *const () as usize, big_ptr, "BigInt ptr_eq");
            }
            other => panic!("slot3 not BigInt: {other:?}"),
        }
    }

    // RAII: dropping the guard pops the registration.
    drop(guard);
    assert_eq!(jit_bank_registry_len(), 0, "guard popped on drop");

    // Keep owners alive until here so nothing was freed under us.
    drop((obj_rc, arr_rc, str_v, big_v, bank));
}

// ----------------------------------------------------------------------------
// NEGATIVE CONTROL — proves registration is what prevents the clear.
// ----------------------------------------------------------------------------

#[test]
fn negative_control_unregistered_bank_object_is_cleared() {
    pin_gc_on();
    let interp = Interp::new();

    // Same setup as the positive test for Object + Array, but DO NOT register the
    // bank. The owning Rc keeps the pointee alive (no free), yet it is unreachable
    // from the GC roots → `gc_sweep` must CLEAR (empty) it.
    let (obj_jv, obj_rc) = make_object();
    let (arr_jv, arr_rc) = make_array();
    let bank: Vec<JsVal> = vec![obj_jv, arr_jv];

    assert_eq!(jit_bank_registry_len(), 0, "bank intentionally NOT registered");
    assert!(!obj_rc.borrow().is_empty(), "object populated before GC");
    assert!(!arr_rc.borrow().is_empty(), "array populated before GC");

    let _cleared = interp.gc_collect(&[]);

    // The container is now EMPTY — the bank-only object was swept (cleared). This
    // is the detectable failure mode the registration prevents.
    assert!(
        obj_rc.borrow().is_empty(),
        "unregistered bank-only Object should be CLEARED by gc_sweep (it was not — \
         the negative control failed to detect the hazard)"
    );
    assert!(
        arr_rc.borrow().is_empty(),
        "unregistered bank-only Array should be CLEARED by gc_sweep"
    );

    // The bits still point at the (now-empty) container — exactly the silent
    // data-loss a registered root prevents. The pointer is still valid (owning Rc
    // alive), so unboxing is safe; it just sees an emptied container.
    let back = unsafe { bank[0].to_value() };
    match back {
        Value::Object(o) => assert!(o.borrow().is_empty(), "emptied container observable"),
        other => panic!("slot0 not Object: {other:?}"),
    }

    drop((obj_rc, arr_rc, bank));
}

// ----------------------------------------------------------------------------
// NO FALSE ROOTS — an all-immediate bank seeds nothing; immediates not rooted.
// ----------------------------------------------------------------------------

#[test]
fn immediates_are_not_rooted() {
    pin_gc_on();
    let interp = Interp::new();

    // A bank of ONLY immediates (number / bool / int32 / singletons). None are
    // pointers, so the seeding pass pushes zero work from this bank. We assert
    // is_pointer() agrees (zero false roots) and that registering + collecting is
    // a clean no-op for these lanes.
    let bank: Vec<JsVal> = vec![
        JsVal::number(3.14),
        JsVal::boolean(true),
        JsVal::boolean(false),
        JsVal::int32(-5),
        JsVal::undefined(),
        JsVal::null(),
    ];
    for (i, slot) in bank.iter().enumerate() {
        assert!(!slot.is_pointer(), "immediate slot {i} mis-classified as pointer: {slot:?}");
    }

    let guard = register_jit_bank(&bank);
    let _ = interp.gc_collect(&[]); // must not panic / mis-root
    drop(guard);
    assert_eq!(jit_bank_registry_len(), 0);
}

// ----------------------------------------------------------------------------
// RAII GAP-FREE — popped on early-return (`?`) and on panic-unwind.
// ----------------------------------------------------------------------------

#[test]
fn raii_popped_on_question_mark_early_return() {
    pin_gc_on();

    // A function that registers a bank then early-returns via `?`. The guard must
    // pop regardless of which path is taken.
    fn body(fail: bool) -> Result<(), &'static str> {
        let bank: Vec<JsVal> = vec![JsVal::number(1.0)];
        let _guard = register_jit_bank(&bank);
        assert_eq!(jit_bank_registry_len(), 1, "registered inside body");
        // `?` early-return path:
        if fail {
            Err("early out")?;
        }
        Ok(())
    }

    let _ = body(true); // takes the `?` early-return path
    assert_eq!(jit_bank_registry_len(), 0, "guard popped on `?` early-return");

    let _ = body(false); // takes the normal-return path
    assert_eq!(jit_bank_registry_len(), 0, "guard popped on normal return");
}

#[test]
fn raii_popped_on_panic_unwind() {
    pin_gc_on();
    assert_eq!(jit_bank_registry_len(), 0, "clean start");

    let r = std::panic::catch_unwind(|| {
        let bank: Vec<JsVal> = vec![JsVal::number(2.0)];
        let _guard = register_jit_bank(&bank);
        assert_eq!(jit_bank_registry_len(), 1);
        panic!("boom");
    });
    assert!(r.is_err(), "the closure panicked");
    assert_eq!(
        jit_bank_registry_len(),
        0,
        "guard popped on panic-unwind (Drop ran during unwinding)"
    );
}

// ----------------------------------------------------------------------------
// CHURN BOUNDED — register/collect/unregister many times, no leak.
// ----------------------------------------------------------------------------

#[test]
fn churn_keeps_gc_live_count_bounded() {
    pin_gc_on();
    let interp = Interp::new();

    // Warm up a baseline live count from a few rounds.
    let warmup = 5usize;
    for _ in 0..warmup {
        let (obj_jv, obj_rc) = make_object();
        let bank: Vec<JsVal> = vec![obj_jv];
        let guard = register_jit_bank(&bank);
        let _ = interp.gc_collect(&[]);
        // While registered + owner alive, the object survives.
        assert!(!obj_rc.borrow().is_empty());
        drop(guard);
        // After the guard pops AND the owning Rc drops at end of scope, the next
        // collect can reclaim it — proving registration does not pin forever.
        drop((obj_rc, bank));
    }
    let _ = interp.gc_collect(&[]);
    let live_after_warmup = gc_live_object_count();

    // A 50x-larger churn: if registration leaked roots, the live count would grow
    // with the number of objects ever created. It must stay bounded instead.
    let churn = warmup * 50;
    for _ in 0..churn {
        let (obj_jv, obj_rc) = make_object();
        let bank: Vec<JsVal> = vec![obj_jv];
        let guard = register_jit_bank(&bank);
        let _ = interp.gc_collect(&[]);
        drop(guard);
        drop((obj_rc, bank));
    }
    let _ = interp.gc_collect(&[]);
    let live_after_churn = gc_live_object_count();

    assert_eq!(jit_bank_registry_len(), 0, "registry drained after churn");
    assert!(
        live_after_churn <= live_after_warmup + 4,
        "GC live count grew with churn (leak): warmup={live_after_warmup}, churn={live_after_churn}"
    );
}
