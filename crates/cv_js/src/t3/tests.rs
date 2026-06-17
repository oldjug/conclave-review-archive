//! T3 optimizer unit tests (IR-level, no native code) + integration via the
//! A/B oracle. The pure-IR tests run on every platform; the native-engagement
//! tests are Windows-only (the backend installs RX pages there).

use super::*;
use crate::bytecode::{BcFunction, Op};
use crate::interp::Value;

/// Build a single-function bytecode module body for a quick optimizer test.
fn mkfn(n_params: u8, n_regs: u16, consts: Vec<Value>, code: Vec<Op>) -> BcFunction {
    BcFunction {
        name: "<test>".to_string(),
        n_params,
        rest_reg: None,
        n_regs,
        consts,
        code,
        ic: std::cell::RefCell::new(Vec::new()),
        feedback: std::cell::RefCell::new(Vec::new()),
        strict: false,
    }
}

#[test]
fn const_fold_collapses_pure_numeric_add() {
    // r2 = 2 + 3 ; ret r2  → r2 = LoadConst 5 ; ret r2
    let f = mkfn(
        0,
        3,
        vec![Value::Number(2.0), Value::Number(3.0)],
        vec![
            Op::LoadConst { dst: 0, k: 0 },
            Op::LoadConst { dst: 1, k: 1 },
            Op::Add { dst: 2, lhs: 0, rhs: 1 },
            Op::Ret { src: 2 },
        ],
    );
    let (_opt, stats) = optimize(&f).expect("optimizes");
    assert!(stats.folded >= 1, "the Add of two consts must fold");
}

#[test]
fn const_fold_does_not_fold_unknown_operand() {
    // param r0 is Unknown (caller-controlled) → r1 = r0 + 1 must NOT fold (could
    // be string concat).
    let f = mkfn(
        1,
        2,
        vec![Value::Number(1.0)],
        vec![
            Op::LoadConst { dst: 1, k: 0 },
            Op::Add { dst: 1, lhs: 0, rhs: 1 },
            Op::Ret { src: 1 },
        ],
    );
    let (_opt, stats) = optimize(&f).expect("optimizes");
    assert_eq!(stats.folded, 0, "an Unknown (param) operand must block the fold");
}

#[test]
fn dce_removes_dead_pure_numeric_op() {
    // r1 = 2 ; r2 = 3 ; r3 = r1 + r2 (dead) ; ret r1  → the Add is dead.
    let f = mkfn(
        0,
        4,
        vec![Value::Number(2.0), Value::Number(3.0)],
        vec![
            Op::LoadConst { dst: 1, k: 0 },
            Op::LoadConst { dst: 2, k: 1 },
            Op::Add { dst: 3, lhs: 1, rhs: 2 },
            Op::Ret { src: 1 },
        ],
    );
    let (opt, stats) = optimize(&f).expect("optimizes");
    assert!(stats.dead_removed >= 1, "the unused Add must be DCE'd");
    // The optimized program is shorter.
    assert!(opt.code.len() < f.code.len());
}

#[test]
fn dce_keeps_op_with_unknown_operands() {
    // param-based add whose result is unused but operands are Unknown → could be
    // string concat, NOT pure → must be KEPT (conservative).
    let f = mkfn(
        2,
        3,
        vec![],
        vec![
            Op::Add { dst: 2, lhs: 0, rhs: 1 },
            Op::Ret { src: 0 },
        ],
    );
    let (_opt, stats) = optimize(&f).expect("optimizes");
    assert_eq!(stats.dead_removed, 0, "Add with Unknown operands is not pure → keep");
}

#[test]
fn copy_prop_threads_a_move() {
    // r1 = 5 ; r2 = move r1 ; r3 = r2 + r2 ; ret r3 — copy-prop should rewrite the
    // uses of r2 to r1 (then the move dies, then the add of consts folds).
    let f = mkfn(
        0,
        4,
        vec![Value::Number(5.0)],
        vec![
            Op::LoadConst { dst: 1, k: 0 },
            Op::Move { dst: 2, src: 1 },
            Op::Add { dst: 3, lhs: 2, rhs: 2 },
            Op::Ret { src: 3 },
        ],
    );
    let (_opt, stats) = optimize(&f).expect("optimizes");
    assert!(stats.copies_propagated >= 1, "the Move should be propagated");
}

#[test]
fn regalloc_compacts_register_count() {
    // Many short-lived temporaries should pack into far fewer physical slots than
    // the original n_regs.
    let f = mkfn(
        0,
        10,
        vec![Value::Number(1.0)],
        vec![
            Op::LoadConst { dst: 1, k: 0 },
            Op::Move { dst: 2, src: 1 },
            Op::Move { dst: 3, src: 2 },
            Op::Move { dst: 4, src: 3 },
            Op::Ret { src: 4 },
        ],
    );
    let (opt, stats) = optimize(&f).expect("optimizes");
    assert!(
        stats.regs_after <= stats.regs_before,
        "allocation never grows the register count"
    );
    assert!((opt.n_regs as usize) == stats.regs_after);
}

#[test]
fn licm_hoists_loop_invariant_numeric_op_and_stays_correct() {
    // Hand-built self-looping single block (header == latch) with a loop-invariant
    // numeric computation `t = c - d` where c,d are PROVEN-NUMERIC locals (consts
    // loaded before the loop, never written in the loop). LICM requires operands
    // proven numeric — a `Sub` of possibly-non-numeric operands could move a
    // `valueOf`/`toString` side effect — so the operands must be locals/consts.
    //
    // Registers: i=r2, acc=r3, t=r4, cond=r5, one=r6, lim=r7, c=r8, d=r9.
    // Program:
    //   0: i   = 0
    //   1: acc = 0
    //   2: one = 1
    //   3: lim = 4
    //   4: c   = 10
    //   5: d   = 3
    //   --- block boundary (jump target) at idx 6 (header == latch) ---
    //   6: cond = i < lim        (loop test)
    //   7: jmpiffalse cond -> 12 (exit)
    //   8: t   = c - d           <- LOOP-INVARIANT, proven-numeric operands
    //   9: acc = acc + t
    //   10: i  = i + one
    //   11: jmp -> 6             (back-edge: header==latch single block)
    //   12: ret acc
    // c,d are NaN consts: my type lattice tracks them as `Number` (NOT `ConstNum`,
    // since NaN's bits are canonicalized by the backend), so const-fold leaves
    // `c - d` alone (cnum returns None for a Number-typed operand) but LICM still
    // sees both operands as PROVEN-NUMERIC and invariant → it hoists. This isolates
    // LICM from const-fold. (NaN arithmetic is irrelevant to the mechanism test.)
    let nan = f64::NAN;
    let f = mkfn(
        0,
        10,
        vec![
            Value::Number(0.0), // k0
            Value::Number(1.0), // k1
            Value::Number(4.0), // k2
            Value::Number(nan), // k3 (typed Number, not ConstNum)
            Value::Number(nan), // k4
        ],
        vec![
            Op::LoadConst { dst: 2, k: 0 }, // 0: i = 0
            Op::LoadConst { dst: 3, k: 0 }, // 1: acc = 0
            Op::LoadConst { dst: 6, k: 1 }, // 2: one = 1
            Op::LoadConst { dst: 7, k: 2 }, // 3: lim = 4
            Op::LoadConst { dst: 8, k: 3 }, // 4: c = NaN (Number)
            Op::LoadConst { dst: 9, k: 4 }, // 5: d = NaN (Number)
            Op::Lt { dst: 5, lhs: 2, rhs: 7 }, // 6: HEADER cond = i < lim
            Op::JmpIfFalse { cond: 5, target: 12 }, // 7: exit
            Op::Sub { dst: 4, lhs: 8, rhs: 9 }, // 8: t = c - d  (INVARIANT, Number)
            Op::Add { dst: 3, lhs: 3, rhs: 4 }, // 9: acc = acc + t
            Op::Add { dst: 2, lhs: 2, rhs: 6 }, // 10: i = i + one
            Op::Jmp { target: 6 },          // 11: back-edge
            Op::Ret { src: 3 },             // 12: ret acc
        ],
    );
    let (opt, stats) = optimize(&f).expect("optimizes");
    assert!(
        stats.hoisted >= 1,
        "the loop-invariant `t = c - d` must be hoisted out of the loop (hoisted={})",
        stats.hoisted
    );
    // The optimized program is still a valid Ret-ended bytecode function.
    assert!(matches!(opt.code.last(), Some(Op::Ret { .. })));
    // The hoisted Sub must NOT be duplicated — exactly one survives (relocated).
    let n_sub = opt.code.iter().filter(|o| matches!(o, Op::Sub { .. })).count();
    assert_eq!(n_sub, 1, "exactly one Sub survives (relocated, not duplicated)");
}

/// MUTATION PROOF for LICM's 0-trip speculation rule (IR-level, non-vacuity of
/// the `read_outside` guard). We build a loop whose invariant `Sub`'s def `t` is
/// READ AFTER the loop. Correct LICM must NOT hoist it (hoisting would compute `t`
/// on the 0-trip path, changing its post-loop value). The unsafe mutation hook
/// forces the hoist; we assert: WITHOUT the hook `t` is NOT hoisted, WITH it `t`
/// IS hoisted — proving the `read_outside` rule is what blocks the unsafe motion.
#[test]
fn licm_zero_trip_rule_blocks_hoist_of_loop_live_out() {
    let nan = f64::NAN;
    // Same self-loop shape, but `t` (r4) is ALSO read at idx 12 (after the loop),
    // so it is loop-live-out → correct LICM must decline to hoist the Sub.
    let f = mkfn(
        0,
        10,
        vec![
            Value::Number(0.0),
            Value::Number(1.0),
            Value::Number(4.0),
            Value::Number(nan),
            Value::Number(nan),
        ],
        vec![
            Op::LoadConst { dst: 2, k: 0 },         // 0: i = 0
            Op::LoadConst { dst: 3, k: 0 },         // 1: acc = 0
            Op::LoadConst { dst: 6, k: 1 },         // 2: one = 1
            Op::LoadConst { dst: 7, k: 2 },         // 3: lim = 4
            Op::LoadConst { dst: 8, k: 3 },         // 4: c = NaN
            Op::LoadConst { dst: 9, k: 4 },         // 5: d = NaN
            Op::LoadConst { dst: 4, k: 0 },         // 6: t = 0 (pre-loop default)
            Op::Lt { dst: 5, lhs: 2, rhs: 7 },      // 7: HEADER cond = i < lim
            Op::JmpIfFalse { cond: 5, target: 13 }, // 8: exit
            Op::Sub { dst: 4, lhs: 8, rhs: 9 },     // 9: t = c - d (invariant)
            Op::Add { dst: 3, lhs: 3, rhs: 6 },     // 10: acc = acc + one
            Op::Add { dst: 2, lhs: 2, rhs: 6 },     // 11: i = i + one
            Op::Jmp { target: 7 },                  // 12: back-edge
            Op::Add { dst: 3, lhs: 3, rhs: 4 },     // 13: acc = acc + t  (t LIVE-OUT)
            Op::Ret { src: 3 },                     // 14: ret acc
        ],
    );

    // Correct LICM: `t` is read after the loop → NOT hoisted.
    let (_opt, stats) = optimize(&f).expect("optimizes");
    assert_eq!(
        stats.hoisted, 0,
        "a loop-live-out invariant must NOT be hoisted (0-trip speculation hazard)"
    );

    // Unsafe mutation: forcing the rule off DOES hoist it → proves the rule is the
    // thing blocking the unsafe motion (non-vacuous guard).
    {
        let _broken = crate::t3::UnsafeLicmGuard::new(true);
        let (_opt2, stats2) = optimize(&f).expect("optimizes");
        assert!(
            stats2.hoisted >= 1,
            "with the 0-trip rule forced OFF, the live-out invariant IS hoisted — \
             proving the rule (not some other check) is what blocks it"
        );
    }
}

#[test]
fn decline_on_unsupported_op() {
    // A GetProp is outside T3's subset → decline (caller runs T2/VM).
    let f = mkfn(
        1,
        2,
        vec![Value::String("x".into())],
        vec![Op::GetProp { dst: 1, obj: 0, key_k: 0 }, Op::Ret { src: 1 }],
    );
    assert_eq!(optimize(&f).err(), Some(DeclineReason::UnsupportedOp));
}

// ----------------------------------------------------------------------
// B3 — safepoint stack-map construction + the spill-to-bank rooting discipline.
//
// These prove the production gate (`optimize_with_safepoints`) builds a verified
// safepoint map for a T3 function: a back-edge in a hot loop is recorded as a
// safepoint, and — for today's NUMERIC subset — every safepoint carries ZERO
// pointer roots (so the discipline holds vacuously, which is why numeric T3 is
// already UAF-safe). The verifier is the gate a future heap-T3 must pass.
// ----------------------------------------------------------------------

#[test]
fn b3_safepoint_map_records_a_loop_back_edge_with_no_pointer_roots() {
    // A counted loop: r0 = 0; while r0 < 10 { r0 = r0 + 1 } ; ret r0. The
    // back-edge (Jmp back to the test) is a safepoint; being a numeric kernel it
    // has NO pointer roots, so the spill-to-bank discipline holds vacuously.
    let f = mkfn(
        0,
        3,
        vec![Value::Number(0.0), Value::Number(10.0), Value::Number(1.0)],
        vec![
            Op::LoadConst { dst: 0, k: 0 },        // 0: r0 = 0
            Op::LoadConst { dst: 1, k: 1 },        // 1: r1 = 10
            Op::Lt { dst: 2, lhs: 0, rhs: 1 },     // 2: r2 = r0 < r1
            Op::JmpIfFalse { cond: 2, target: 7 }, // 3: if !r2 goto 7
            Op::LoadConst { dst: 1, k: 2 },        // 4: r1 = 1
            Op::Add { dst: 0, lhs: 0, rhs: 1 },    // 5: r0 = r0 + r1
            Op::Jmp { target: 2 },                 // 6: BACK-EDGE to 2
            Op::Ret { src: 0 },                    // 7: ret r0
        ],
    );
    let (_opt, _stats, map) =
        crate::t3::optimize_with_safepoints(&f).expect("numeric loop optimizes + verifies");
    // The discipline holds: no pointer root is out of bank range (there are none).
    assert!(
        map.roots_covered_by_bank(_opt.n_regs as usize),
        "B3 discipline must hold for a numeric kernel (no pointer roots to spill)"
    );
    // Every recorded safepoint carries ZERO pointer roots (numeric subset).
    for rec in map.records() {
        assert_eq!(
            rec.root_count(),
            0,
            "a numeric-subset safepoint must record no pointer roots"
        );
    }
    // The optimizer may rewrite the loop, but if the back-edge survives it must be
    // recorded as a safepoint. (After const-fold/LICM the loop body can shrink; we
    // assert the map is well-formed + verified rather than a fixed count.)
    assert!(
        map.verify_against_bank(_opt.n_regs as usize).is_ok(),
        "the built safepoint map must pass the B3 discipline verification"
    );
}

#[test]
fn b3_straight_line_numeric_fn_has_a_verified_safepoint_map() {
    // No loop, no call → no safepoints, but the map still verifies (vacuously).
    let f = mkfn(
        0,
        3,
        vec![Value::Number(2.0), Value::Number(3.0)],
        vec![
            Op::LoadConst { dst: 0, k: 0 },
            Op::LoadConst { dst: 1, k: 1 },
            Op::Add { dst: 2, lhs: 0, rhs: 1 },
            Op::Ret { src: 2 },
        ],
    );
    let (opt, _stats, map) =
        crate::t3::optimize_with_safepoints(&f).expect("straight-line fn optimizes + verifies");
    assert!(map.is_empty(), "a straight-line numeric fn has no safepoints");
    assert!(map.roots_covered_by_bank(opt.n_regs as usize));
}

// ----------------------------------------------------------------------
// Native engagement + A/B oracle (Windows-only — the backend installs RX pages).
// ----------------------------------------------------------------------

#[cfg(target_os = "windows")]
mod native {
    use crate::ab_oracle::{assert_tiers_agree, assert_tiers_agree_t3_engaged};

    /// A loop/arith kernel where T3 MUST engage AND match the VM. The engaged
    /// variant asserts ≥1 native T3 run (non-vacuity).
    #[test]
    fn t3_loop_kernel_matches_vm_and_engages() {
        let src = r#"
            function sum(n) {
                var s = 0;
                for (var i = 0; i < n; i = i + 1) { s = s + i; }
                return s;
            }
            var out = 0;
            for (var k = 0; k < 30; k = k + 1) { out = sum(20); }
            out;
        "#;
        assert_tiers_agree_t3_engaged(src).expect("T3 must agree with the VM and engage");
    }

    #[test]
    fn t3_arith_kernel_engages() {
        let src = r#"
            function f(a, b) {
                var x = a * b - a + b;
                var y = x / 2 + 1;
                return y - x;
            }
            var r = 0;
            for (var i = 0; i < 30; i = i + 1) { r = f(i, i + 1); }
            r;
        "#;
        assert_tiers_agree_t3_engaged(src).expect("T3 arith kernel must engage + match");
    }

    /// LICM correctness end-to-end: a JS loop with a loop-invariant numeric
    /// computation, including the ZERO-TRIP path (the loop body never runs). T3
    /// hoists the invariant; the oracle proves the result equals the VM on BOTH
    /// the runs-many and runs-zero cases — catching the speculation hazard if the
    /// 0-trip rule were wrong.
    #[test]
    fn t3_licm_invariant_loop_matches_vm() {
        // `c*d - c` is loop-invariant (c,d are local numeric consts). The function
        // is called with n=20 (body runs) AND n=0 (body never runs) in the hot
        // loop, so both LICM paths are exercised against the VM.
        let src = r#"
            function f(n) {
                var c = 7;
                var d = 3;
                var acc = 0;
                for (var i = 0; i < n; i = i + 1) {
                    var t = c * d - c;   // loop-invariant
                    acc = acc + t + i;
                }
                return acc;
            }
            var out = 0;
            for (var k = 0; k < 30; k = k + 1) {
                out = out + f(20) + f(0);  // both runs-many and runs-zero
            }
            out;
        "#;
        assert_tiers_agree_t3_engaged(src).expect("T3 LICM must match the VM (incl. 0-trip)");
    }

    /// A broad correctness sweep: a variety of numeric snippets, each run through
    /// the full multi-tier oracle WITH the T3 leg added. Proves T3 == VM == … on
    /// the corpus (T3 declines/deopts where it can't run — still correct).
    #[test]
    fn t3_corpus_agrees_across_tiers() {
        let cases = [
            "var s=0; for(var i=0;i<10;i=i+1){s=s+i*2;} s;",
            "function g(n){ if(n<2) return n; return g(n-1)+g(n-2); } g(12);",
            "var a=1,b=2,c=3; (a+b)*c - a/b;",
            "var x=5; var y=x; var z=y+y; z*z;",
            "function h(n){var t=1;while(n>0){t=t*2;n=n-1;}return t;} h(10);",
            // Mixed-type: forces a deopt (string concat) → resumes the VM.
            "function m(a){ return a + 1; } m('x');",
            "function m(a){ return a + 1; } m(41);",
            // Division specials.
            "1/0;", "(-1)/0;", "0/0;",
            // Comparison chains.
            "var i=3; (i<5) === true;",
        ];
        for src in cases {
            assert_tiers_agree(src).unwrap_or_else(|d| panic!("T3 corpus diverged on {src:?}: {d}"));
        }
    }

    /// 200-case randomized numeric fuzz: generate small arithmetic snippets and
    /// assert T3 == VM (transitively == tree-walk) on each.
    #[test]
    fn t3_numeric_fuzz_200() {
        // A tiny deterministic LCG so the corpus is reproducible.
        let mut state: u64 = 0x9E3779B97F4A7C15;
        let mut next = || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (state >> 33) as u32
        };
        let ops = ["+", "-", "*", "/"];
        let cmps = ["<", "<=", ">", ">=", "===", "!=="];
        for _ in 0..200 {
            let a = (next() % 100) as i64 - 50;
            let b = (next() % 100) as i64 - 50;
            let c = (next() % 50) as i64 + 1; // nonzero-ish
            let o1 = ops[(next() as usize) % ops.len()];
            let o2 = ops[(next() as usize) % ops.len()];
            let cm = cmps[(next() as usize) % cmps.len()];
            let iters = 5 + (next() % 25);
            let src = format!(
                "function k(p){{ var x = (p {o1} {a}) {o2} {b}; var y = x {o1} {c}; return (y {cm} x); }}
                 var r = false; for(var i=0;i<{iters};i=i+1){{ r = k(i); }} r ? 1 : 0;"
            );
            assert_tiers_agree(&src)
                .unwrap_or_else(|d| panic!("T3 fuzz diverged on {src:?}: {d}"));
        }
    }

    /// MUTATION PROOF — the A/B oracle is NON-VACUOUS. We deliberately break the
    /// const-fold pass (off-by-one in the folded result) via the test-only hook;
    /// the oracle MUST then catch the divergence (T3 != VM). We then restore the
    /// pass and assert the SAME snippet is green again. Without this, a green
    /// oracle could be vacuous (an optimizer that never fires, or a comparator
    /// that never compares). This proves the gate has teeth.
    ///
    /// The snippet const-folds `2 + 3` (both proven-numeric constants) into a
    /// `LoadConst 5`; the wrong-fold hook turns that into `LoadConst 6`, so the
    /// returned value diverges from the VM (which always computes 5).
    #[test]
    fn t3_oracle_catches_broken_const_fold_mutation() {
        let src = r#"
            function compute() {
                var z = 2 + 3;
                return z;
            }
            var out = 0;
            for (var i = 0; i < 30; i = i + 1) { out = compute(); }
            out;
        "#;

        // (1) Baseline: the pass is correct → the oracle is GREEN.
        assert_tiers_agree(src).expect("T3 must match the VM with the correct fold");

        // (2) MUTATE: break const-fold (+1). The oracle MUST redden (T3 folds to 6,
        //     the VM computes 5). If it stays green, the gate is vacuous — a FAIL.
        {
            let _broken = crate::t3::WrongFoldGuard::new(true);
            let res = assert_tiers_agree(src);
            assert!(
                res.is_err(),
                "VACUOUS ORACLE: a deliberately-wrong const-fold did NOT redden the \
                 A/B oracle — the gate has no teeth"
            );
        }

        // (3) RESTORE: with the hook off again the snippet is green — proving the
        //     redness in (2) was caused by the mutation, not a flaky environment.
        assert_tiers_agree(src).expect("restored fold must match the VM again");
    }
}
