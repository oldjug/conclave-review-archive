//! M6.1 — `sessionStorage` Web Storage semantics, and its independence
//! from `localStorage`.
//!
//! These are integration tests: each runs a real `Interp` with both
//! storage globals installed over INDEPENDENT backing maps (exactly how
//! the browser wires them — a persisted localStorage map and a fresh
//! per-session sessionStorage map). They assert the full Storage surface
//! (setItem/getItem/removeItem/clear/length/key) and that writes to one
//! store are never visible in the other.

use std::sync::{Arc, Mutex};
use cv_js::interp::Interp;
use cv_js::ordered::OrderedMap;
use cv_js::Value;

/// The browser backs `install_storage*` with `OrderedMap` (re-exported as
/// `HashMap` inside the engine). Mirror that here so the test exercises the
/// exact type the production path uses.
fn fresh_store() -> Arc<Mutex<OrderedMap<String, String>>> {
    Arc::new(Mutex::new(OrderedMap::new()))
}

fn run(i: &mut Interp, src: &str) -> Value {
    i.run_completion_value(src)
        .unwrap_or_else(|e| panic!("JS error running {src:?}: {e:?}"))
}

fn run_str(i: &mut Interp, src: &str) -> String {
    match run(i, src) {
        Value::String(s) => s.to_string(),
        other => panic!("expected string from {src:?}, got {other:?}"),
    }
}

fn run_num(i: &mut Interp, src: &str) -> f64 {
    match run(i, src) {
        Value::Number(n) => n,
        other => panic!("expected number from {src:?}, got {other:?}"),
    }
}

#[test]
fn session_storage_basic_set_get_round_trip() {
    let mut i = Interp::new();
    i.install_basic_globals();
    i.install_session_storage(fresh_store());

    run(&mut i, "sessionStorage.setItem('k', 'v');");
    assert_eq!(run_str(&mut i, "sessionStorage.getItem('k');"), "v");
}

#[test]
fn session_storage_remove_and_clear() {
    let mut i = Interp::new();
    i.install_basic_globals();
    i.install_session_storage(fresh_store());

    run(&mut i, "sessionStorage.setItem('a','1'); sessionStorage.setItem('b','2');");
    assert_eq!(run_num(&mut i, "sessionStorage.length;"), 2.0);

    run(&mut i, "sessionStorage.removeItem('a');");
    assert_eq!(run_num(&mut i, "sessionStorage.length;"), 1.0);
    // Removed key reads back as null per the Storage spec.
    assert!(matches!(
        run(&mut i, "sessionStorage.getItem('a');"),
        Value::Null
    ));

    run(&mut i, "sessionStorage.clear();");
    assert_eq!(run_num(&mut i, "sessionStorage.length;"), 0.0);
}

#[test]
fn session_storage_key_and_length() {
    let mut i = Interp::new();
    i.install_basic_globals();
    i.install_session_storage(fresh_store());

    run(&mut i, "sessionStorage.setItem('first','1'); sessionStorage.setItem('second','2');");
    assert_eq!(run_num(&mut i, "sessionStorage.length;"), 2.0);
    // OrderedMap preserves insertion order, so key(0) is the first set key.
    assert_eq!(run_str(&mut i, "sessionStorage.key(0);"), "first");
    assert_eq!(run_str(&mut i, "sessionStorage.key(1);"), "second");
    // Out-of-range index yields null.
    assert!(matches!(
        run(&mut i, "sessionStorage.key(99);"),
        Value::Null
    ));
}

#[test]
fn session_storage_is_independent_from_local_storage() {
    let mut i = Interp::new();
    i.install_basic_globals();
    // Two SEPARATE backing maps — exactly the browser's wiring.
    let local = fresh_store();
    let session = fresh_store();
    i.install_storage(local.clone());
    i.install_session_storage(session.clone());

    // Write distinct values under the same key in each store.
    run(&mut i, "localStorage.setItem('token', 'LOCAL');");
    run(&mut i, "sessionStorage.setItem('token', 'SESSION');");

    // Each store reads back ITS OWN value — no aliasing.
    assert_eq!(run_str(&mut i, "localStorage.getItem('token');"), "LOCAL");
    assert_eq!(run_str(&mut i, "sessionStorage.getItem('token');"), "SESSION");

    // Clearing one must not touch the other.
    run(&mut i, "sessionStorage.clear();");
    assert_eq!(run_num(&mut i, "sessionStorage.length;"), 0.0);
    assert_eq!(run_str(&mut i, "localStorage.getItem('token');"), "LOCAL");

    // And the underlying maps reflect the same independence.
    assert_eq!(local.lock().unwrap().get("token").map(|s| s.as_str()), Some("LOCAL"));
    assert!(session.lock().unwrap().get("token").is_none());
}

#[test]
fn session_storage_overwrite_updates_value() {
    let mut i = Interp::new();
    i.install_basic_globals();
    i.install_session_storage(fresh_store());

    run(&mut i, "sessionStorage.setItem('k','one');");
    run(&mut i, "sessionStorage.setItem('k','two');");
    assert_eq!(run_str(&mut i, "sessionStorage.getItem('k');"), "two");
    // Overwriting an existing key does not grow the length.
    assert_eq!(run_num(&mut i, "sessionStorage.length;"), 1.0);
}
