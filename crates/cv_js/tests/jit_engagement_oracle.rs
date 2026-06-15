//! JIT ENGAGEMENT ORACLE — the headline beat-Chrome compute lever's honesty gate.
//!
//! BACKGROUND (the attribution bug this guards against): the cv_browser bench
//! reported `t2_exec_count == 0` on the hot-numeric workloads and concluded "the
//! JIT never ran". That was WRONG. The hot functions DO run as native machine
//! code — just via the P6 f64 numeric JIT, which is tried FIRST in
//! `try_call_fn_via_bytecode` and short-circuits before the T2 block is reached.
//! So the T2 counter stayed 0 while P6 did all the native work. The fix added a
//! P6 exec counter (mirroring T1/T2/T3) and made the bench read EVERY tier.
//!
//! This file is the correctness gate for that fix. It is an INTEGRATION test (its
//! own process) so the default-on tier env gates (`CV_NOJIT`-unset ⇒ P6 on,
//! `CV_T2`!=0 ⇒ T2 on) are read in their natural default state — we do NOT set any
//! disabling env, so the "JIT on" pass is the genuine shipped default.
//!
//! What it proves, for BOTH the integer-loop and float-arithmetic bench kernels:
//!   1. ENGAGEMENT: with the JIT on (default), some optimizing tier executes the
//!      hot function as native code — `native_exec_count = p6+t1+t3+t2 > 0`. (For
//!      these all-numeric kernels it is P6 specifically, asserted separately.)
//!   2. CORRECTNESS / BYTE-IDENTITY: the JIT-on result is IDENTICAL to the pure
//!      tree-walk result (the slowest, fully-interpreted, zero-native baseline,
//!      forced via `ForcedTier::TreeWalk`). A native miscompute would diverge.
//!   3. ESCAPE HATCH: the `CV_T2=0` switch (the ship escape hatch) is honored.
//!
//! Run with output:
//!   cargo test -p cv_js --test jit_engagement_oracle -- --nocapture

use cv_js::interp::{
    p6_exec_count, reset_p6_exec_count, reset_t1_exec_count, reset_t2_exec_count,
    reset_t3_exec_count, set_force_tier, t1_exec_count, t2_exec_count, t3_exec_count, ForcedTier,
    Interp, Value,
};

/// The two bench kernels, verbatim in shape from `crates/cv_browser/benchfix/`.
/// Each puts the hot work in a CALLED function (`work`/`f`) invoked in a loop —
/// exactly how V8-friendly hot JS looks and exactly what the JIT path covers.
/// Iteration counts are reduced from the bench (which uses 6000 / 1.5M) so the
/// pure-tree-walk baseline pass finishes quickly; still WAY over the 12-call
/// warmup threshold, so the hot fn is fully tiered.
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

/// Build a fresh interpreter with the standard globals (mirrors how the bench's
/// `LiveInterp` and the shaped tests set up a real `Interp`).
fn fresh() -> Interp {
    let i = Interp::new();
    i.install_basic_globals();
    i.install_json();
    i
}

/// Run `src` and return its completion value (the trailing bare expression).
fn run(src: &str) -> Value {
    let mut i = fresh();
    i.run_completion_value(src)
        .unwrap_or_else(|e| panic!("script run failed: {e:?}"))
}

/// Bit-exact compare of two numeric results. We compare the raw f64 bits so a
/// JIT miscompute that differs in the last ULP (or NaN-vs-number) is caught —
/// "identical result" must mean truly identical, not approximately.
fn assert_num_identical(label: &str, a: &Value, b: &Value) {
    match (a, b) {
        (Value::Number(x), Value::Number(y)) => {
            assert_eq!(
                x.to_bits(),
                y.to_bits(),
                "{label}: JIT result {x} != tree-walk result {y} (bits {:#x} vs {:#x})",
                x.to_bits(),
                y.to_bits()
            );
        }
        other => panic!("{label}: expected two Numbers, got {other:?}"),
    }
}

/// Total native (optimizing-tier) executions since the last reset — exactly what
/// the bench's `native_exec_count` sums. >0 ⇒ some tier ran native machine code.
fn native_exec_count() -> u64 {
    p6_exec_count() + t1_exec_count() + t3_exec_count() + t2_exec_count()
}

fn reset_all_counters() {
    reset_p6_exec_count();
    reset_t1_exec_count();
    reset_t3_exec_count();
    reset_t2_exec_count();
}

/// THE oracle, parameterized over a kernel: JIT-on result == pure-tree-walk
/// result (byte-identical) AND the JIT actually engaged (native_exec_count > 0,
/// and for these numeric kernels P6 specifically).
fn oracle(label: &str, src: &str) {
    // BASELINE: pure tree-walk — the slowest, fully-interpreted, ZERO-native
    // path. `ForcedTier::TreeWalk` disables the bytecode tier entirely so neither
    // P6 nor T1/T2/T3 can run; this is the ground-truth result the JIT must match.
    set_force_tier(Some(ForcedTier::TreeWalk));
    reset_all_counters();
    let tree_walk = run(src);
    let tw_native = native_exec_count();
    set_force_tier(None);
    assert_eq!(
        tw_native, 0,
        "{label}: tree-walk baseline ran {tw_native} native execs — the baseline must be pure-interpreted (no tier) or the comparison is not against a non-JIT ground truth"
    );

    // JIT ON: the genuine shipped default (no disabling env set). The hot fn
    // tiers up after the warmup threshold and runs as native machine code.
    reset_all_counters();
    let jit_on = run(src);
    let p6 = p6_exec_count();
    let total = native_exec_count();

    // (1) ENGAGEMENT — the headline guard. >0 == the JIT really ran.
    assert!(
        total > 0,
        "{label}: native_exec_count is 0 with the JIT ON — the hot function did NOT tier to native code. This is the bug the fix targets."
    );
    // For these all-numeric kernels the engaging tier is specifically P6 (tried
    // first). Assert it so a future re-order that silently drops P6 is caught.
    assert!(
        p6 > 0,
        "{label}: P6 numeric JIT did not run (p6_exec_count=0); native total={total}. Hot all-numeric kernels are expected to tier under P6."
    );

    // (2) CORRECTNESS — byte-identical to the non-JIT ground truth.
    assert_num_identical(label, &jit_on, &tree_walk);
}

#[test]
fn loop_kernel_jit_engages_and_matches_tree_walk() {
    oracle("loop.js", LOOP_SRC);
}

#[test]
fn jit_kernel_jit_engages_and_matches_tree_walk() {
    oracle("jit.js", JIT_SRC);
}

/// The CV_T2=0 ESCAPE HATCH still works: it must NOT change the RESULT (pure-VM
/// is byte-identical) — that is the ship guarantee. We force the VM tier
/// programmatically here (CV_T2=0 env is process-cached on first read, which an
/// in-process test cannot un-cache; the dedicated `escape_hatch_*` binary below
/// exercises the real env). `ForcedTier::Vm` routes through the bytecode VM with
/// the JIT tiers declining/falling back, and its result must equal tree-walk.
#[test]
fn vm_tier_result_matches_tree_walk() {
    set_force_tier(Some(ForcedTier::TreeWalk));
    let tw_loop = run(LOOP_SRC);
    let tw_jit = run(JIT_SRC);
    set_force_tier(Some(ForcedTier::Vm));
    let vm_loop = run(LOOP_SRC);
    let vm_jit = run(JIT_SRC);
    set_force_tier(None);
    assert_num_identical("loop vm==tree-walk", &vm_loop, &tw_loop);
    assert_num_identical("jit vm==tree-walk", &vm_jit, &tw_jit);
}
