//! T4 P5 AOT-persist tests — the cold-repeat beat-Chrome lever's gates.
//!
//! Two layers:
//!   1. BLOB-LEVEL (platform-independent): serialize → deserialize round-trips the
//!      native bytes + DeoptSite table + embedded modules byte-identically; the
//!      bytecode digest gates drift (a changed program MISSES); the stale-digest
//!      mutation hook proves the digest is load-bearing; corruption/ABI/cpu
//!      mismatch fail closed (never fabricate a blob).
//!   2. END-TO-END (Windows): persist a freshly-compiled T4 function, simulate a
//!      COLD REPEAT VISIT (clear the live T4 cache), reload the native blob from
//!      disk with ZERO codegen, and prove the reloaded run is byte-identical to the
//!      VM — AND that a guard miss on the reloaded code still deopts correctly (the
//!      deopt backstop survives persistence). Plus the honesty counters (a green
//!      round-trip oracle that never actually re-installed a blob would be a lie).

use super::*;
use crate::bytecode::{BcFunction, Module, Op};
use crate::interp::Value;
use crate::osr::{DeoptReason, DeoptSite};

/// A tiny numeric-subset `BcFunction` for blob tests: `f(x){ return x*x + 1.0; }`.
fn sample_fn(name: &str) -> BcFunction {
    BcFunction {
        name: name.to_string(),
        n_params: 1,
        rest_reg: None,
        n_regs: 4,
        consts: vec![Value::Number(1.0)],
        code: vec![
            Op::Mul { dst: 1, lhs: 0, rhs: 0 }, // r1 = x*x
            Op::LoadConst { dst: 2, k: 0 },     // r2 = 1.0
            Op::Add { dst: 1, lhs: 1, rhs: 2 }, // r1 = r1 + r2
            Op::Ret { src: 1 },
        ],
        ic: std::cell::RefCell::new(Vec::new()),
        feedback: std::cell::RefCell::new(Vec::new()),
        strict: false,
    }
}

fn sample_module(name: &str) -> Module {
    Module { fns: vec![sample_fn(name)], script_forinit_syncs: Vec::new() }
}

fn sample_code_and_sites() -> (Vec<u8>, Vec<DeoptSite>) {
    // Representative relocation-free bytes + a small DeoptSite table. The exact
    // bytes don't matter for the blob round-trip (they're opaque); we just need a
    // non-empty buffer and a couple of sites with distinct reasons + offsets.
    let code = vec![0x55, 0x48, 0x89, 0xE5, 0xC3, 0x90, 0x90]; // push rbp; mov rbp,rsp; ret; nops
    let sites = vec![
        DeoptSite { native_off: 1, bc_pc: 0, reason: DeoptReason::NonNumber },
        DeoptSite { native_off: 5, bc_pc: 2, reason: DeoptReason::FallThrough },
    ];
    (code, sites)
}

// ── BLOB-LEVEL round-trip ────────────────────────────────────────────────────

#[test]
fn blob_roundtrips_code_sites_and_modules() {
    let fused = sample_module("f+inline");
    let orig = sample_module("f");
    let (code, sites) = sample_code_and_sites();

    let blob = serialize_blob(&code, &sites, &fused, &orig, false).expect("serialize");
    let key = compute_native_key(&fused, &orig);
    let parts = deserialize_blob(&blob, key).expect("deserialize on matching key");

    // Native bytes round-trip byte-identically.
    assert_eq!(parts.code, code, "native code bytes must round-trip exactly");
    // DeoptSite table round-trips (offset + bc_pc + reason).
    assert_eq!(parts.deopt_sites.len(), sites.len());
    for (got, want) in parts.deopt_sites.iter().zip(sites.iter()) {
        assert_eq!(got.native_off, want.native_off);
        assert_eq!(got.bc_pc, want.bc_pc);
        assert_eq!(got.reason, want.reason);
    }
    // Embedded modules round-trip structurally. `Op`/`Value` don't impl PartialEq,
    // so we prove structural identity the load-bearing way: the AOT key recomputed
    // from the RELOADED modules must equal the original key (the key folds the full
    // op stream + consts + arity, so an equal key ⇒ a structurally identical
    // program — exactly the property the cache relies on). Plus spot-check lengths.
    assert_eq!(parts.fused_module.fns.len(), 1);
    assert_eq!(parts.fused_module.fns[0].code.len(), fused.fns[0].code.len());
    assert_eq!(parts.fused_module.fns[0].consts.len(), fused.fns[0].consts.len());
    assert_eq!(parts.original_caller.fns[0].code.len(), orig.fns[0].code.len());
    let reloaded_key = compute_native_key(&parts.fused_module, &parts.original_caller);
    assert_eq!(
        reloaded_key, key,
        "the reloaded modules must recompute the SAME AOT key (structural identity)"
    );
}

/// THE digest gate: a CHANGED program (different bytecode) produces a DIFFERENT
/// key, so a blob persisted for program A is REJECTED when looked up with the key
/// of program B — a clean MISS (recompile), never stale wrong code.
#[test]
fn drifted_program_misses_on_key_mismatch() {
    let fused_a = sample_module("f");
    let orig_a = sample_module("f");
    let (code, sites) = sample_code_and_sites();
    let blob = serialize_blob(&code, &sites, &fused_a, &orig_a, false).expect("serialize A");

    // Program B differs by ONE op (Sub instead of Add) — a real source edit.
    let mut fused_b = sample_module("f");
    fused_b.fns[0].code[2] = Op::Sub { dst: 1, lhs: 1, rhs: 2 };
    let orig_b = fused_b.clone();
    let key_b = compute_native_key(&fused_b, &orig_b);

    // Blob A under key B → rejected (digest drift). This is the safety property.
    assert!(
        deserialize_blob(&blob, key_b).is_none(),
        "a drifted program's key must REJECT the stale blob (recompile, never wrong)"
    );
    // Sanity: blob A under key A still loads.
    let key_a = compute_native_key(&fused_a, &orig_a);
    assert!(deserialize_blob(&blob, key_a).is_some());
}

/// A changed CONST (1.0 → 2.0) also drifts the digest → miss. Proves the digest
/// folds const bit patterns (NaN-exact), not just the op stream.
#[test]
fn changed_const_drifts_the_digest() {
    let fused_a = sample_module("f");
    let mut fused_b = sample_module("f");
    fused_b.fns[0].consts = vec![Value::Number(2.0)];
    assert_ne!(
        compute_native_key(&fused_a, &fused_a).0,
        compute_native_key(&fused_b, &fused_b).0,
        "a changed const must change the key (digest folds consts)"
    );
}

/// THE mutation hook proving the digest is LOAD-BEARING: with `set_force_stale_digest`
/// engaged, the key OMITS the digest, so program A and program B (which differ)
/// compute the SAME key → a drifted blob is WRONGLY accepted. This proves the
/// drift-rejection test above is non-vacuous (the digest is what makes it reject).
#[test]
fn stale_digest_mutation_makes_drift_undetected() {
    let fused_a = sample_module("f");
    let mut fused_b = sample_module("f");
    fused_b.fns[0].code[2] = Op::Sub { dst: 1, lhs: 1, rhs: 2 };

    // Production default: A and B keys DIFFER (the digest catches the drift).
    assert_ne!(
        compute_native_key(&fused_a, &fused_a).0,
        compute_native_key(&fused_b, &fused_b).0,
        "without the mutation hook, drift MUST change the key"
    );

    // With the hook engaged: the digest is omitted, so A and B keys are EQUAL —
    // drift goes undetected. This is exactly the failure the digest prevents; the
    // round-trip oracle would then run B's source against A's stale code and diverge.
    let _g = StaleDigestGuard::new(true);
    assert_eq!(
        compute_native_key(&fused_a, &fused_a).0,
        compute_native_key(&fused_b, &fused_b).0,
        "with the stale-digest hook, drift is (wrongly) undetected — proving the \
         digest is the load-bearing invalidator"
    );
}

/// Corruption / truncation fails closed (returns None, never a fabricated blob).
#[test]
fn corruption_and_truncation_fail_closed() {
    let fused = sample_module("f");
    let (code, sites) = sample_code_and_sites();
    let blob = serialize_blob(&code, &sites, &fused, &fused, false).expect("serialize");
    let key = compute_native_key(&fused, &fused);

    // Truncated at every prefix length → never produces a blob (fails closed).
    for n in 0..blob.len() {
        assert!(
            deserialize_blob(&blob[..n], key).is_none(),
            "a truncated blob (len {n}) must fail closed"
        );
    }
    // A flipped magic byte → reject.
    let mut bad = blob.clone();
    bad[0] ^= 0xFF;
    assert!(deserialize_blob(&bad, key).is_none(), "bad magic must reject");
}

/// An empty native-code buffer is never serialized (we don't persist a code-less
/// blob that would re-install to a zero-length page).
#[test]
fn empty_code_is_not_serialized() {
    let fused = sample_module("f");
    assert!(
        serialize_blob(&[], &[], &fused, &fused, false).is_none(),
        "an empty code buffer must not be persisted"
    );
}

/// The DeoptReason tag mapping is a bijection over all variants (so the table
/// round-trips every reason a guard could carry, and a new variant can't silently
/// collide).
#[test]
fn deopt_reason_tags_round_trip_all_variants() {
    let all = [
        DeoptReason::NonNumber,
        DeoptReason::NonObject,
        DeoptReason::ShapeMiss,
        DeoptReason::CallDecline,
        DeoptReason::NonArray,
        DeoptReason::BadIndex,
        DeoptReason::HoleOrSpecial,
        DeoptReason::FallThrough,
    ];
    let mut seen = std::collections::HashSet::new();
    for r in all {
        let t = reason_tag(r);
        assert!(seen.insert(t), "reason tag {t} must be unique per variant");
        assert_eq!(tag_reason(t), Some(r), "tag {t} must round-trip to {r:?}");
    }
    // An unknown tag fails closed.
    assert_eq!(tag_reason(200), None, "an unknown reason tag must reject");
}

/// The honesty counters bump on a real serialize (so a green test that never
/// actually serialized would be caught by a zero count).
#[test]
fn serialize_bumps_the_store_counter() {
    reset_aot_store_count();
    let fused = sample_module("f");
    let (code, sites) = sample_code_and_sites();
    let before = aot_store_count();
    let _ = serialize_blob(&code, &sites, &fused, &fused, false).expect("serialize");
    assert_eq!(
        aot_store_count(),
        before + 1,
        "serialize must bump the store honesty counter"
    );
}

/// The AOT-persist gate defaults OFF (env-driven), and the in-process force guard
/// toggles it + restores on drop (scope safety, mirrors FeedbackGuard).
#[test]
fn aot_persist_force_guard_toggles_and_restores() {
    let before = force_aot_persist();
    {
        let _g = AotPersistGuard::new(true);
        assert!(force_aot_persist());
        assert!(aot_persist_enabled());
    }
    assert_eq!(force_aot_persist(), before);
}

// ── END-TO-END (Windows) — persist → cold-repeat reload → byte-identical ──────

#[cfg(target_os = "windows")]
mod end_to_end {
    use crate::ab_oracle::assert_aot_roundtrip_matches_vm;

    /// THE cold-repeat beat-Chrome gate: a float-dense function called in a loop
    /// (the jit.js shape) is compiled + persisted on a first visit, then on a
    /// simulated COLD REPEAT VISIT (live T4 cache cleared) re-installed from the
    /// disk blob with ZERO codegen — and the reloaded run is byte-identical to the
    /// VM. The oracle asserts the reload PATH actually fired (non-vacuity).
    #[test]
    fn aot_cold_repeat_jit_shape_matches_vm() {
        let src = "function f(x){ return ((x*x*0.5 + x*3.0 - 1.0) * (x - 2.0) + x*x*x*0.25) \
                   / (x + 1.0) - x*0.5 + x*x*0.125 - x*7.0; } \
                   var s = 0; for (var i = 0; i < 300; i = i+1) { s = s + f(i); } s;";
        let fired = assert_aot_roundtrip_matches_vm(src)
            .expect("AOT cold-repeat reload must be byte-identical to the VM");
        assert!(
            fired,
            "the AOT reload path must actually fire (persist + re-install from disk) \
             on the inlinable jit.js shape — a vacuous round-trip is not a gate"
        );
    }

    /// An integer-loop callee (loop.js shape) also persists + cold-reloads
    /// byte-identically.
    #[test]
    fn aot_cold_repeat_integer_loop_matches_vm() {
        let src = "function add1(x){ return x + 1; } \
                   var s = 0; for (var i = 0; i < 200; i = i+1) { s = add1(s); } s;";
        let _ = assert_aot_roundtrip_matches_vm(src)
            .expect("AOT cold-repeat reload (integer) must be byte-identical to the VM");
        // (Engagement is shape-dependent; the float-dense test pins non-vacuity.)
    }

    /// DEOPT-AFTER-RELOAD: a function whose guard FAILS at run time (a non-number
    /// argument forces a deopt) must, after being persisted + cold-reloaded, STILL
    /// deopt correctly to the VM frame and produce the byte-identical result — the
    /// deopt backstop survives persistence (the re-attached DeoptSite table is what
    /// makes a stale/mismatched blob safe). The mixed-type call exercises the
    /// natural runtime deopt path on the reloaded native code.
    #[test]
    fn aot_reloaded_code_still_deopts_correctly() {
        // `g(a)` is float-dense and inlined; it is called with numbers (T4 fast path)
        // AND with a string (forces the NonNumber guard → deopt to the VM). After
        // reload, both paths must match the VM exactly.
        let src = "function g(a){ return a*a + a*2.0 - 1.0; } \
                   var s = 0; \
                   for (var i = 0; i < 60; i = i+1) { s = s + g(i); } \
                   s = s + g('x'); \
                   for (var j = 0; j < 60; j = j+1) { s = s + g(j*1.5); } \
                   s;";
        let _ = assert_aot_roundtrip_matches_vm(src).expect(
            "reloaded AOT code must still deopt correctly on a non-number operand \
             and match the VM byte-for-byte",
        );
    }

    /// Round-trip byte-identity across a small numeric corpus (the gate every later
    /// AOT use rides on). Each snippet: persist → cold-repeat reload → == VM.
    #[test]
    fn aot_cold_repeat_corpus_is_byte_identical() {
        let corpus = [
            "function f(x){ return x*x + x*2.0 - 3.0; } \
             var s=0; for(var i=0;i<40;i=i+1){ s=s+f(i); } s;",
            "function h(a){ return a/2.0 + 1.0; } \
             var s=0; for(var i=0;i<40;i=i+1){ s=s+h(i); } s;",
            // branchy callee
            "function pick(x){ if (x < 10) return x*2.0; return x+5.0; } \
             var s=0; for(var i=0;i<40;i=i+1){ s=s+pick(i); } s;",
            // special-number callee (NaN/Inf through the reloaded unboxed path)
            "function d(x){ return x/x; } \
             var s=0; for(var i=0;i<10;i=i+1){ s=s+d(i); } s;",
        ];
        for src in corpus {
            assert_aot_roundtrip_matches_vm(src)
                .unwrap_or_else(|e| panic!("AOT cold-repeat corpus diverged on {src:?}: {e}"));
        }
    }

    /// THE end-to-end stale-rejection gate (the design's "a changed source correctly
    /// MISSES — no stale wrong code"): persist program A's native code into a shared
    /// store, then run a DRIFTED program B (same function NAME + shape, different
    /// arithmetic) against the SAME store. B's optimized bytecode digest differs from
    /// A's, so B's AOT lookup MISSES A's blob (a clean miss → fresh compile), and B's
    /// result is its OWN correct value — NEVER A's stale code. This proves the digest
    /// gate is load-bearing at the FULL run level, not just the key level.
    #[test]
    fn drifted_source_misses_stale_blob_and_recompiles_correctly() {
        use crate::interp::{ForcedTier, Interp, TierGuard};

        // A shared store dir for BOTH programs (the realistic "same site, edited
        // source" scenario). Thread-local override → no parallel-test race.
        let dir = std::env::temp_dir().join(format!(
            "tbjs_aot_stale_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        let _ = std::fs::remove_dir_all(&dir);
        crate::t4::aot::set_thread_dir_override(Some(dir.clone()));
        let _persist = crate::t4::aot::AotPersistGuard::new(true);

        let run = |src: &str| -> f64 {
            let _g = TierGuard::new(ForcedTier::T4);
            crate::interp::reset_bc_fn_cache();
            crate::interp::reset_t4_cache();
            let mut i = Interp::new();
            i.install_basic_globals();
            match i.run_completion_value(src) {
                Ok(crate::interp::Value::Number(n)) => n,
                other => panic!("expected a Number result, got {other:?}"),
            }
        };
        let vm_run = |src: &str| -> f64 {
            let _g = TierGuard::new(ForcedTier::Vm);
            crate::interp::reset_bc_fn_cache();
            crate::interp::reset_t4_cache();
            let mut i = Interp::new();
            i.install_basic_globals();
            match i.run_completion_value(src) {
                Ok(crate::interp::Value::Number(n)) => n,
                other => panic!("expected a Number result, got {other:?}"),
            }
        };

        // Program A: `f(x) = x*x + 1.0`. Run it (persists A's native blob).
        let prog_a = "function f(x){ return x*x + 1.0; } \
                      var s=0; for(var i=0;i<80;i=i+1){ s=s+f(i); } s;";
        let a_t4 = run(prog_a);
        assert_eq!(a_t4, vm_run(prog_a), "program A T4 must match the VM");

        // Program B: the SAME function name + same loop, DRIFTED arithmetic
        // (`x*x - 1.0`). Its optimized bytecode digest differs → B must MISS A's
        // persisted blob and compute B's OWN result, not A's stale `+ 1.0` code.
        crate::t4::aot::reset_aot_miss_count();
        crate::t4::aot::reset_aot_load_count();
        let prog_b = "function f(x){ return x*x - 1.0; } \
                      var s=0; for(var i=0;i<80;i=i+1){ s=s+f(i); } s;";
        let b_t4 = run(prog_b);
        let b_vm = vm_run(prog_b);
        assert_eq!(
            b_t4, b_vm,
            "DRIFTED program B must recompile + run its OWN correct result (never A's \
             stale persisted code): T4 {b_t4} vs VM {b_vm}"
        );
        // A and B differ by a constant per iteration, so their sums differ — proves B
        // did NOT silently run A's code.
        assert_ne!(a_t4, b_t4, "A and B must differ (drift is real, not a no-op)");
        // The drift was detected as a MISS (B looked up its key, found A's file absent
        // for B's key → miss → recompile), proving the digest gate fired end-to-end.
        assert!(
            crate::t4::aot::aot_miss_count() >= 1,
            "B's drifted lookup must register an AOT MISS (the digest rejected the \
             stale blob), proving the gate is load-bearing at the run level"
        );

        crate::t4::aot::set_thread_dir_override(None);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
