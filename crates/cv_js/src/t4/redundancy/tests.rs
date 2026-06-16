//! T4 P4 — redundancy / load / check elimination tests.
//!
//! Two layers:
//!   * PURE-IR structural tests (run everywhere): the LVN transform folds an
//!     available expression to a copy, forwards copies, kills on operand clobber,
//!     respects the `Allow` resume-path gate, preserves the op count (so the
//!     resume-pc map stays valid), and — via the mutation hook — proves the
//!     kill-on-clobber is load-bearing (non-vacuity at the IR level).
//!   * WINDOWS oracle/engagement tests (in `t4/tests.rs`, exercising the full
//!     native path + deopt) ride the same A/B oracle the prior phases established;
//!     the dedicated P4 oracle + deopt + non-vacuity gates live there.

use super::*;
use crate::bytecode::Op;
use crate::interp::Value;

/// CSE folds a redundant pure arith op (the jit.js `x*x` shape) to a copy of the
/// dominating computation — and the op COUNT is preserved (the resume-pc map
/// invariant). `Allow::Always` is the inlined-path mode (unconditional CSE).
#[test]
fn cse_folds_redundant_mul_to_copy() {
    // r1 = x*x ; r2 = x*x  (redundant) ; r3 = r1 + r2 ; ret r3
    let mut code = vec![
        Op::Mul { dst: 1, lhs: 0, rhs: 0 },
        Op::Mul { dst: 2, lhs: 0, rhs: 0 },
        Op::Add { dst: 3, lhs: 1, rhs: 2 },
        Op::Ret { src: 3 },
    ];
    let before = code.len();
    let st = redundancy_eliminate(&mut code, 4, &[], Allow::Always);
    assert_eq!(code.len(), before, "op count must be preserved (resume-pc map)");
    assert_eq!(st.cse_folded, 1, "the second x*x must be folded to a copy");
    // r2's op is now a Move from r1 (the dominating x*x result).
    assert!(
        matches!(code[1], Op::Move { dst: 2, src: 1 }),
        "redundant Mul became Move r2, r1; got {:?}",
        code[1]
    );
    // The dominating op A is untouched.
    assert!(matches!(code[0], Op::Mul { dst: 1, lhs: 0, rhs: 0 }));
    // The consumer's read of r2 is then ALSO copy-forwarded to r1 (r2 aliases r1),
    // so the Add becomes `r3 = r1 + r1` — observably identical to `r1 + r2` (both
    // are the unique x*x value) and a further win. Accept either form.
    assert!(
        matches!(code[2], Op::Add { dst: 3, lhs: 1, rhs: 2 } | Op::Add { dst: 3, lhs: 1, rhs: 1 }),
        "consumer reads the dominating value (possibly copy-forwarded); got {:?}",
        code[2]
    );
}

/// `x*x*x` reuses `x*x`: the bytecode `t1=x*x; t2=t1*x; t3=x*x` — the SECOND `x*x`
/// folds even though `t1*x` came between, because `x` (and t1) are unchanged.
#[test]
fn cse_reuses_across_independent_ops() {
    let mut code = vec![
        Op::Mul { dst: 1, lhs: 0, rhs: 0 }, // t1 = x*x
        Op::Mul { dst: 2, lhs: 1, rhs: 0 }, // t2 = t1*x   (x*x*x)
        Op::Mul { dst: 3, lhs: 0, rhs: 0 }, // t3 = x*x  (redundant with t1)
        Op::Add { dst: 4, lhs: 2, rhs: 3 },
        Op::Ret { src: 4 },
    ];
    let st = redundancy_eliminate(&mut code, 5, &[], Allow::Always);
    assert_eq!(st.cse_folded, 1);
    assert!(
        matches!(code[2], Op::Move { dst: 3, src: 1 }),
        "x*x at op 2 folds to a copy of t1; got {:?}",
        code[2]
    );
}

/// KILL-ON-CLOBBER: when an operand is REDEFINED between two identical ops, the
/// second is NOT folded (its inputs changed → a different result). This is the
/// load-bearing correctness rule.
#[test]
fn cse_does_not_fold_across_operand_clobber() {
    // r1 = x*x ; x = x+1 (clobbers operand 0) ; r2 = x*x  → must NOT fold r2
    let mut code = vec![
        Op::Mul { dst: 1, lhs: 0, rhs: 0 },
        Op::Add { dst: 0, lhs: 0, rhs: 0 }, // redefines reg 0 (x)
        Op::Mul { dst: 2, lhs: 0, rhs: 0 },
        Op::Ret { src: 2 },
    ];
    let st = redundancy_eliminate(&mut code, 3, &[], Allow::Always);
    assert_eq!(st.cse_folded, 0, "an operand was clobbered — must not fold");
    assert!(matches!(code[2], Op::Mul { dst: 2, lhs: 0, rhs: 0 }));
}

/// Availability does NOT cross a basic-block boundary: the same expression in two
/// blocks is recomputed (the per-block scope, matching the backend's XMM cache).
#[test]
fn cse_does_not_cross_blocks() {
    // block0: r1 = x*x ; jmp 3
    // (op 2 is a block start — jump target)
    // block1: r2 = x*x ; ret r2
    let mut code = vec![
        Op::Mul { dst: 1, lhs: 0, rhs: 0 }, // 0
        Op::Jmp { target: 2 },              // 1
        Op::Mul { dst: 2, lhs: 0, rhs: 0 }, // 2  (block start)
        Op::Ret { src: 2 },                 // 3
    ];
    let st = redundancy_eliminate(&mut code, 3, &[], Allow::Always);
    assert_eq!(st.cse_folded, 0, "availability must not cross a block boundary");
    assert!(matches!(code[2], Op::Mul { dst: 2, lhs: 0, rhs: 0 }));
}

/// STORE-TO-LOAD FORWARDING: a `Move b, a` makes later reads of `b` read `a` while
/// the copy is valid (copy propagation).
#[test]
fn copy_prop_forwards_reads() {
    // r1 = Move r0 ; r2 = r1 + r1  →  r2 = r0 + r0
    let mut code = vec![
        Op::Move { dst: 1, src: 0 },
        Op::Add { dst: 2, lhs: 1, rhs: 1 },
        Op::Ret { src: 2 },
    ];
    let st = redundancy_eliminate(&mut code, 3, &[], Allow::Always);
    assert!(st.copies_forwarded >= 2, "both reads of r1 forward to r0");
    assert!(
        matches!(code[1], Op::Add { dst: 2, lhs: 0, rhs: 0 }),
        "reads of the copy forwarded to its source; got {:?}",
        code[1]
    );
}

/// Copy aliases are KILLED when the source is redefined to a DIFFERENT value:
/// `r1 = Move r0 ; r0 = r0 + 1 (r0 now differs from r1) ; use r1`. The read of r1
/// must NOT forward to r0 (they no longer hold the same value). A `Sub r1, r3`
/// consumer (no CSE possible) isolates the forwarding-kill from CSE.
#[test]
fn copy_prop_kills_on_redef() {
    let consts = vec![Value::Number(1.0)];
    let mut code = vec![
        Op::LoadConst { dst: 3, k: 0 },     // r3 = 1
        Op::Move { dst: 1, src: 0 },        // r1 = r0  (r1 joins r0's class)
        Op::Add { dst: 0, lhs: 0, rhs: 3 }, // r0 = r0 + 1  (r0 gets a FRESH class)
        Op::Sub { dst: 4, lhs: 1, rhs: 3 }, // r4 = r1 - 1
        Op::Ret { src: 4 },
    ];
    redundancy_eliminate(&mut code, 5, &consts, Allow::Always);
    assert!(
        matches!(code[3], Op::Sub { dst: 4, lhs: 1, rhs: 3 }),
        "r1's read must NOT forward to the now-different r0; got {:?}",
        code[3]
    );
}

/// The `OnlyNumericOperands` gate (single-function path): a redundant op whose
/// operands are NOT proven numeric is NOT folded (so a resume on the optimized
/// module never skips a coercion side effect). With proven-numeric operands it IS
/// folded.
#[test]
fn allow_only_numeric_operands_gate() {
    // operands = param x (reg 0, NOT proven numeric). Two x*x — under
    // OnlyNumericOperands neither folds (x unknown); under Always the 2nd folds.
    let mut code_a = vec![
        Op::Mul { dst: 1, lhs: 0, rhs: 0 },
        Op::Mul { dst: 2, lhs: 0, rhs: 0 },
        Op::Ret { src: 2 },
    ];
    let st = redundancy_eliminate(&mut code_a, 3, &[], Allow::OnlyNumericOperands);
    assert_eq!(st.cse_folded, 0, "unproven operand x: no CSE on the single-fn path");

    // Now operands are RESULTS of arith ops (proven numeric). r1=a+b, then two
    // r1*r1 — the second folds even under OnlyNumericOperands (r1 is proven num).
    let mut code_b = vec![
        Op::Add { dst: 2, lhs: 0, rhs: 1 }, // r2 = a+b  (proven numeric)
        Op::Mul { dst: 3, lhs: 2, rhs: 2 }, // r3 = r2*r2
        Op::Mul { dst: 4, lhs: 2, rhs: 2 }, // r4 = r2*r2  (redundant)
        Op::Ret { src: 4 },
    ];
    let st = redundancy_eliminate(&mut code_b, 5, &[], Allow::OnlyNumericOperands);
    assert_eq!(st.cse_folded, 1, "proven-numeric operands: CSE fires");
    assert!(matches!(code_b[2], Op::Move { dst: 4, src: 3 }));
}

/// A numeric `LoadConst` operand is proven numeric → CSE fires under the
/// single-function gate.
#[test]
fn allow_only_numeric_const_operand() {
    // r1 = const 2.0 ; r2 = x*r1 ... no — both must be numeric. Use two const muls.
    // r0 = const 2.0 ; r1 = const 3.0 ; r2 = r0*r1 ; r3 = r0*r1 (redundant)
    let consts = vec![Value::Number(2.0), Value::Number(3.0)];
    let mut code = vec![
        Op::LoadConst { dst: 0, k: 0 },
        Op::LoadConst { dst: 1, k: 1 },
        Op::Mul { dst: 2, lhs: 0, rhs: 1 },
        Op::Mul { dst: 3, lhs: 0, rhs: 1 },
        Op::Ret { src: 3 },
    ];
    let st = redundancy_eliminate(&mut code, 4, &consts, Allow::OnlyNumericOperands);
    assert_eq!(st.cse_folded, 1, "const operands are proven numeric → CSE fires");
    assert!(matches!(code[3], Op::Move { dst: 3, src: 2 }));
}

/// NON-VACUITY (the mutation hook): with `force_unsafe_cse` set, the kill-on-
/// clobber is SKIPPED, so the second `x*x` (after x was redefined) is WRONGLY
/// folded to a copy of the stale result — a divergence the IR test detects (and
/// the A/B oracle catches at runtime). Proves the kill logic is load-bearing.
#[test]
fn mutation_hook_forces_unsafe_fold() {
    let mut code = vec![
        Op::Mul { dst: 1, lhs: 0, rhs: 0 },
        Op::Add { dst: 0, lhs: 0, rhs: 0 }, // redefines operand 0
        Op::Mul { dst: 2, lhs: 0, rhs: 0 }, // redundant ONLY if the clobber is ignored
        Op::Ret { src: 2 },
    ];
    // Sanity: without the hook, the clobber prevents the (wrong) fold.
    {
        let mut clean = code.clone();
        let st = redundancy_eliminate(&mut clean, 3, &[], Allow::Always);
        assert_eq!(st.cse_folded, 0, "clean: clobber prevents the fold");
    }
    // With the hook, the unsafe fold IS produced (this is the wrong-codegen the
    // oracle must redden on — proving the kill is non-vacuous).
    let _g = UnsafeCseGuard::new(true);
    let st = redundancy_eliminate(&mut code, 3, &[], Allow::Always);
    assert_eq!(st.cse_folded, 1, "hook ON: the unsafe fold across a clobber is produced");
    assert!(
        matches!(code[2], Op::Move { dst: 2, src: 1 }),
        "the hook wrongly folds r2 to the stale r1; got {:?}",
        code[2]
    );
}

/// The pass is a NO-OP (zero rewrites, op count preserved) on code with no
/// redundancy — it never changes a correct, already-minimal program.
#[test]
fn no_redundancy_is_a_noop() {
    let mut code = vec![
        Op::Mul { dst: 1, lhs: 0, rhs: 0 },
        Op::Add { dst: 2, lhs: 1, rhs: 0 },
        Op::Sub { dst: 3, lhs: 2, rhs: 1 },
        Op::Ret { src: 3 },
    ];
    let snapshot = format!("{:?}", code);
    let st = redundancy_eliminate(&mut code, 4, &[], Allow::Always);
    assert!(!st.is_nonvacuous(), "no redundancy → no rewrite");
    assert_eq!(format!("{:?}", code), snapshot, "code unchanged");
}
