//! CV_T2=0 ESCAPE-HATCH oracle (own process so the env gate is deterministic).
//!
//! The ship escape hatch is "one switch = pure VM, byte-identical". The bench's
//! attribution fix must NOT have weakened that. This test pins `CV_T2=0` at the
//! VERY TOP of the one test — before any tier env lock is read — so the gate
//! genuinely engages, then asserts the hot kernels still produce the SAME numeric
//! result they do with the JIT on. (This is the env-driven sibling of the
//! programmatic `vm_tier_result_matches_tree_walk` check in jit_engagement_oracle.)
//!
//! Note we assert RESULT identity, not that native_exec_count goes to 0: `CV_T2=0`
//! disables the T2 tier and (per `t2_heap_enabled`) the T2 heap path, but the P6
//! numeric JIT is gated separately by `CV_NOJIT`. The contract we verify is the
//! one that matters for shipping: flipping the escape hatch never changes output.
//!
//! Run with output:
//!   cargo test -p cv_js --test jit_escape_hatch_cv_t2 -- --nocapture

use cv_js::interp::{Interp, Value};

const LOOP_SRC: &str = r#"
function work(n) {
  var s = 0;
  for (var i = 0; i < n; i = i + 1) {
    s = s + i;
  }
  return s;
}
var __bench_loop_result = 0;
for (var j = 0; j < 2000; j++) {
  __bench_loop_result = __bench_loop_result + work(400);
}
__bench_loop_result;
"#;

const JIT_SRC: &str = r#"
function f(x) {
  return ((x * x * 0.5 + x * 3.0 - 1.0) * (x - 2.0) + x * x * x * 0.25) / (x + 1.0) - x * 0.5 + x * x * 0.125 - x * 7.0;
}
var __bench_jit_result = 0;
for (var i = 0; i < 50000; i++) {
  __bench_jit_result = __bench_jit_result + f(i);
}
__bench_jit_result;
"#;

/// Independently-computed reference values (host f64, same arithmetic + summation
/// order as the kernels) so the test is not circular — the engine must match a
/// value derived OUTSIDE the engine, not just match itself.
fn ref_loop() -> f64 {
    let mut total = 0.0f64;
    for _ in 0..2000 {
        let mut s = 0.0f64;
        let mut i = 0.0f64;
        while i < 400.0 {
            s += i;
            i += 1.0;
        }
        total += s;
    }
    total
}

fn ref_jit() -> f64 {
    let f = |x: f64| {
        ((x * x * 0.5 + x * 3.0 - 1.0) * (x - 2.0) + x * x * x * 0.25) / (x + 1.0) - x * 0.5
            + x * x * 0.125
            - x * 7.0
    };
    let mut total = 0.0f64;
    let mut i = 0.0f64;
    while i < 50000.0 {
        total += f(i);
        i += 1.0;
    }
    total
}

fn run(src: &str) -> f64 {
    let i = Interp::new();
    i.install_basic_globals();
    i.install_json();
    let mut i = i;
    match i.run_completion_value(src) {
        Ok(Value::Number(n)) => n,
        other => panic!("script did not return a Number: {other:?}"),
    }
}

#[test]
fn cv_t2_zero_preserves_results() {
    // Edition 2024: set_var is unsafe. Pin the escape hatch BEFORE any tier gate
    // reads its env (this is the first test code to run in this dedicated binary).
    unsafe {
        std::env::set_var("CV_T2", "0");
    }
    let got_loop = run(LOOP_SRC);
    let got_jit = run(JIT_SRC);
    let want_loop = ref_loop();
    let want_jit = ref_jit();
    assert_eq!(
        got_loop.to_bits(),
        want_loop.to_bits(),
        "CV_T2=0 loop result {got_loop} != reference {want_loop}"
    );
    assert_eq!(
        got_jit.to_bits(),
        want_jit.to_bits(),
        "CV_T2=0 jit result {got_jit} != reference {want_jit}"
    );
}
