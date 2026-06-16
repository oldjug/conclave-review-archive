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
