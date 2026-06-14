//! Differential A/B oracle: tree-walk == bytecode-VM.
//!
//! M3.0 mission. This engine has TWO execution tiers — the tree-walk
//! interpreter (`interp.rs`) and a per-function bytecode VM (`bytecode.rs`) that
//! the tree-walk call path routes eligible function bodies into. The two have
//! ALREADY diverged once (a `==`/`!=` coercion "split-brain" bug that only
//! manifested inside VM-compiled hot functions). A wrong slot index in the
//! upcoming M3 flat-shaped-slot object representation would be SILENT data
//! corruption, not a crash.
//!
//! So before any storage change we build the JS analogue of the M2.4
//! `CV_LAYOUT_VERIFY` oracle: run the SAME source through BOTH tiers and prove
//! byte-identical observable behavior. The enabling primitive is the
//! programmatic per-call tier override (`set_force_tier` / `TierGuard` in
//! `interp.rs`), which lets one process drive both tiers — the env gate
//! (`CV_BYTECODE`) is process-global and can't be A/B'd by toggling.
//!
//! "Observable" here is three things, compared after the script's synchronous
//! body settles AND its microtask checkpoint drains:
//!   (a) the completion value (deep structural equality — own-enumerable keys in
//!       ECMA [[OwnPropertyKeys]] order + recursively-equal values; arrays incl.
//!       holes; primitives incl. -0 / NaN / BigInt / Symbol keys),
//!   (b) thrown-error parity (same error constructor name + message, or both
//!       no-throw),
//!   (c) side effects — the ordered `console.*` output the snippet produced.

use crate::interp::{
    enumerable_string_keys_with_own_symbols, ForcedTier, Interp, JsError, T2HeapGuard, TierGuard,
    Value,
};

/// One tier's full observable outcome from running a snippet.
#[derive(Debug, Clone)]
pub struct TierOutcome {
    /// The completion value of the top-level script (its final expression, or
    /// `undefined`). `Err` carries a thrown JS value (already reduced to
    /// `(constructor-name, message)` so it survives comparison across tiers
    /// without depending on object identity).
    pub result: Result<Value, ThrownError>,
    /// Ordered `console.*` output captured during the run — the snippet's
    /// observable side effects.
    pub output: Vec<String>,
}

/// A thrown JS value reduced to its spec-observable identity: the error
/// constructor name (`TypeError`, `RangeError`, …) and the message string. We
/// deliberately do NOT compare object identity / stack — only what spec-level
/// `catch (e) { e.name; e.message }` can observe.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ThrownError {
    pub name: String,
    pub message: String,
}

/// A precise structured description of a tier disagreement.
#[derive(Debug, Clone)]
pub struct Divergence {
    /// What category of observable diverged.
    pub kind: DivergenceKind,
    /// A human-readable path into the value where they first differ
    /// (e.g. `<result>.a[2].x`), or a label for non-value divergences.
    pub path: String,
    /// The tree-walk tier's rendering at that path.
    pub tree_walk: String,
    /// The VM tier's rendering at that path.
    pub vm: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DivergenceKind {
    /// One tier threw and the other didn't, or both threw different errors.
    Throw,
    /// The completion values are structurally unequal.
    Value,
    /// The captured `console.*` side-effect streams differ.
    SideEffect,
}

impl std::fmt::Display for Divergence {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "tier divergence ({:?}) at {}:\n  tree-walk: {}\n  vm:        {}",
            self.kind, self.path, self.tree_walk, self.vm
        )
    }
}

/// Run `src` under one forced tier and capture its full observable outcome.
/// Uses a fresh `Interp` with the standard globals, the `TierGuard` scope guard
/// to install the override (restored on drop), and clears the per-function
/// bytecode cache first so a prior tier's compile decisions can't leak in.
fn run_one_tier(src: &str, tier: ForcedTier) -> TierOutcome {
    let _guard = TierGuard::new(tier);
    crate::interp::reset_bc_fn_cache();
    // T1 caches a per-function compile decision keyed by FunctionValue pointer;
    // clear it (and the exec counter) so a prior tier's decision can't leak in
    // and the JIT tier's engagement is measured fresh.
    crate::interp::reset_t1_cache();
    crate::interp::reset_t1_exec_count();
    // Same for the T2-lite tier.
    crate::interp::reset_t2_cache();
    crate::interp::reset_t2_exec_count();
    // Same for the T3 optimizing tier.
    crate::interp::reset_t3_cache();
    crate::interp::reset_t3_exec_count();
    // T2→T2 native-to-native registry + counter (so a prior tier's installed
    // callee code can't be resolved by a later tier's caller).
    crate::interp::reset_t2_module_registry();
    crate::interp::reset_t2_t2_call_count();
    let mut interp = Interp::new();
    interp.install_basic_globals();
    let result = match interp.run_completion_value(src) {
        Ok(v) => Ok(v),
        Err(e) => Err(reduce_thrown(&e)),
    };
    TierOutcome {
        result,
        output: interp.output.clone(),
    }
}

/// Reduce a `JsError` to a tier-stable `ThrownError` (constructor name +
/// message). Mirrors what `catch (e) { e.name; e.message }` observes.
fn reduce_thrown(e: &JsError) -> ThrownError {
    match e {
        JsError::Throw(v) => thrown_error_of(v),
        JsError::Internal(s) => ThrownError {
            name: "InternalError".to_string(),
            message: s.clone(),
        },
    }
}

/// Extract `(name, message)` from a thrown JS value the way `catch` would.
fn thrown_error_of(v: &Value) -> ThrownError {
    if let Value::Object(o) = v {
        let m = o.borrow();
        let name = match m.get("name") {
            Some(Value::String(s)) => s.to_string(),
            _ => "Error".to_string(),
        };
        let message = match m.get("message") {
            Some(Value::String(s)) => s.to_string(),
            Some(other) => other.to_display_string(),
            None => String::new(),
        };
        return ThrownError { name, message };
    }
    // A non-object throw (string/number/…): there's no constructor; compare the
    // thrown primitive's canonical form as the "message" so two tiers throwing
    // the same primitive agree.
    ThrownError {
        name: "<non-error>".to_string(),
        message: canon(v),
    }
}

/// Compare two tier outcomes for byte-identical observable behavior (throw
/// parity → completion value → side effects). `a_label`/`b_label` name the tiers
/// for divergence reports (rendered into the `tree_walk`/`vm` fields). Returns
/// the first `Divergence`, or `Ok(())` if they agree.
fn compare_outcomes(
    a: &TierOutcome,
    b: &TierOutcome,
    a_label: &str,
    b_label: &str,
) -> Result<(), Divergence> {
    let _ = (a_label, b_label); // labels embedded in field strings below
    // (b) Throw parity first — a thrown error short-circuits the value.
    match (&a.result, &b.result) {
        (Err(ea), Err(eb)) => {
            if ea != eb {
                return Err(Divergence {
                    kind: DivergenceKind::Throw,
                    path: "<thrown>".to_string(),
                    tree_walk: format!("{} {}: {}", a_label, ea.name, ea.message),
                    vm: format!("{} {}: {}", b_label, eb.name, eb.message),
                });
            }
        }
        (Err(ea), Ok(vb)) => {
            return Err(Divergence {
                kind: DivergenceKind::Throw,
                path: "<thrown>".to_string(),
                tree_walk: format!("{} threw {}: {}", a_label, ea.name, ea.message),
                vm: format!("{} returned {}", b_label, canon(vb)),
            });
        }
        (Ok(va), Err(eb)) => {
            return Err(Divergence {
                kind: DivergenceKind::Throw,
                path: "<thrown>".to_string(),
                tree_walk: format!("{} returned {}", a_label, canon(va)),
                vm: format!("{} threw {}: {}", b_label, eb.name, eb.message),
            });
        }
        (Ok(va), Ok(vb)) => {
            // (a) Completion-value deep structural equality.
            if let Some(d) = deep_diff(va, vb, "<result>", 0) {
                return Err(d);
            }
        }
    }
    // (c) Side-effect (console.*) stream parity.
    if a.output != b.output {
        let (tw, vm, path) = first_output_diff(&a.output, &b.output);
        return Err(Divergence {
            kind: DivergenceKind::SideEffect,
            path,
            tree_walk: format!("{a_label}: {tw}"),
            vm: format!("{b_label}: {vm}"),
        });
    }
    Ok(())
}

/// THE oracle entry point. Evaluate the SAME `src` under `ForcedTier::TreeWalk`,
/// `ForcedTier::Vm`, AND `ForcedTier::Jit` (the T1 baseline JIT) and assert
/// byte-identical observable outcome across all three. Returns `Ok(())` if the
/// tiers agree, or the first structured `Divergence`.
///
/// The Jit run is byte-compared against the tree-walk too (transitively the VM):
/// if T1 declines a function it falls back to the VM, so the Jit outcome equals
/// the VM outcome there; where T1 actually compiles, this proves T1==VM==tree-
/// walk. Engagement (that T1 truly ran native code) is asserted separately by
/// `assert_tiers_agree_engaged` / tests reading `t1_exec_count`.
pub fn assert_tiers_agree(src: &str) -> Result<(), Divergence> {
    let a = run_one_tier(src, ForcedTier::TreeWalk);
    let b = run_one_tier(src, ForcedTier::Vm);
    compare_outcomes(&a, &b, "tree-walk", "vm")?;
    let c = run_one_tier(src, ForcedTier::Jit);
    compare_outcomes(&a, &c, "tree-walk", "jit")?;
    // T2-lite must agree too (it deopts/declines to the VM where it can't run,
    // so this transitively proves T2-lite == VM == tree-walk on what it runs).
    let d = run_one_tier(src, ForcedTier::T2Lite);
    compare_outcomes(&a, &d, "tree-walk", "t2lite")?;
    // T3 (the optimizing tier) must agree too: it declines unsupported ops to T2
    // and deopts any typed divergence to the VM (on the OPTIMIZED module, which is
    // observationally identical to the original), so this transitively proves
    // T3 == VM == tree-walk on what it runs. The whole corpus re-runs against T3
    // here, so any optimizer miscompile reddens the oracle.
    let e = run_one_tier(src, ForcedTier::T3);
    compare_outcomes(&a, &e, "tree-walk", "t3")?;
    Ok(())
}

/// Like `assert_tiers_agree`, but ALSO requires the T3 optimizing tier to have
/// genuinely executed native code (≥1 T3 invocation) — guarding against a
/// vacuously-green oracle where T3 silently declines/deopts everything. Use for
/// the numeric/loop/arith kernel corpus where T3 MUST engage.
pub fn assert_tiers_agree_t3_engaged(src: &str) -> Result<(), Divergence> {
    let a = run_one_tier(src, ForcedTier::TreeWalk);
    let b = run_one_tier(src, ForcedTier::Vm);
    compare_outcomes(&a, &b, "tree-walk", "vm")?;
    let c = {
        let _guard = TierGuard::new(ForcedTier::T3);
        crate::interp::reset_bc_fn_cache();
        crate::interp::reset_t3_cache();
        crate::interp::reset_t3_exec_count();
        let mut interp = Interp::new();
        interp.install_basic_globals();
        let result = match interp.run_completion_value(src) {
            Ok(v) => Ok(v),
            Err(e) => Err(reduce_thrown(&e)),
        };
        let outcome = TierOutcome {
            result,
            output: interp.output.clone(),
        };
        let execed = crate::interp::t3_exec_count();
        (outcome, execed)
    };
    compare_outcomes(&a, &c.0, "tree-walk", "t3")?;
    if c.1 == 0 {
        return Err(Divergence {
            kind: DivergenceKind::SideEffect,
            path: "<t3-engagement>".to_string(),
            tree_walk: "expected T3 to execute ≥1 function natively".to_string(),
            vm: "T3 executed 0 functions (vacuously green — FAIL)".to_string(),
        });
    }
    Ok(())
}

/// Like `assert_tiers_agree`, but ALSO requires the T2-lite tier to have
/// genuinely executed native code (≥1 T2-lite invocation) — guarding against a
/// vacuously-green oracle where T2-lite silently declines/deopts everything. Use
/// for the numeric-subset corpus where T2-lite MUST engage.
pub fn assert_tiers_agree_t2_engaged(src: &str) -> Result<(), Divergence> {
    let a = run_one_tier(src, ForcedTier::TreeWalk);
    let b = run_one_tier(src, ForcedTier::Vm);
    compare_outcomes(&a, &b, "tree-walk", "vm")?;
    let c = {
        let _guard = TierGuard::new(ForcedTier::T2Lite);
        crate::interp::reset_bc_fn_cache();
        crate::interp::reset_t2_cache();
        crate::interp::reset_t2_exec_count();
        let mut interp = Interp::new();
        interp.install_basic_globals();
        let result = match interp.run_completion_value(src) {
            Ok(v) => Ok(v),
            Err(e) => Err(reduce_thrown(&e)),
        };
        let outcome = TierOutcome {
            result,
            output: interp.output.clone(),
        };
        let execed = crate::interp::t2_exec_count();
        (outcome, execed)
    };
    compare_outcomes(&a, &c.0, "tree-walk", "t2lite")?;
    if c.1 == 0 {
        return Err(Divergence {
            kind: DivergenceKind::SideEffect,
            path: "<t2-engagement>".to_string(),
            tree_walk: "expected T2-lite to execute ≥1 function natively".to_string(),
            vm: "T2-lite executed 0 functions (vacuously green — FAIL)".to_string(),
        });
    }
    Ok(())
}

/// Like `assert_tiers_agree_t2_engaged`, but runs the T2-lite tier with T2 HEAP
/// MODE engaged (`T2HeapGuard`) — the owning + GC-rooted bank path that can hold a
/// HEAP GetProp result in a bank slot. Proves tree-walk == VM == T2(heap) on a
/// kernel that loads-and-holds a heap value across ops, with ≥1 native T2 run. This
/// is the P3 "RESULTS == VM" oracle for the first heap-resident use.
pub fn assert_tiers_agree_t2_heap_engaged(src: &str) -> Result<(), Divergence> {
    let a = run_one_tier(src, ForcedTier::TreeWalk);
    let b = run_one_tier(src, ForcedTier::Vm);
    compare_outcomes(&a, &b, "tree-walk", "vm")?;
    let c = {
        let _tier = TierGuard::new(ForcedTier::T2Lite);
        let _heap = T2HeapGuard::new(true); // engage the owning heap bank path
        crate::interp::reset_bc_fn_cache();
        crate::interp::reset_t2_cache();
        crate::interp::reset_t2_exec_count();
        let mut interp = Interp::new();
        interp.install_basic_globals();
        let result = match interp.run_completion_value(src) {
            Ok(v) => Ok(v),
            Err(e) => Err(reduce_thrown(&e)),
        };
        let outcome = TierOutcome {
            result,
            output: interp.output.clone(),
        };
        let execed = crate::interp::t2_exec_count();
        (outcome, execed)
    };
    compare_outcomes(&a, &c.0, "tree-walk", "t2lite-heap")?;
    if c.1 == 0 {
        return Err(Divergence {
            kind: DivergenceKind::SideEffect,
            path: "<t2-heap-engagement>".to_string(),
            tree_walk: "expected T2-lite (heap) to execute ≥1 function natively".to_string(),
            vm: "T2-lite (heap) executed 0 functions (vacuously green — FAIL)".to_string(),
        });
    }
    Ok(())
}

/// Like `assert_tiers_agree_t2_heap_engaged`, but ALSO requires the T2→T2
/// NATIVE-TO-NATIVE call path to have engaged (≥1 callee resolved to a Ready T2
/// slot and run via the JsVal-args entry, not the VM re-entry). Proves a T2 caller
/// calling a T2 callee runs BOTH natively (caller's `t2_exec_count` > 0 AND the
/// callee ran native-to-native via `t2_t2_call_count` > 0) and is == tree-walk ==
/// VM. This is THE T2→T2 correctness+engagement gate.
pub fn assert_tiers_agree_t2_t2_engaged(src: &str) -> Result<(), Divergence> {
    let a = run_one_tier(src, ForcedTier::TreeWalk);
    let b = run_one_tier(src, ForcedTier::Vm);
    compare_outcomes(&a, &b, "tree-walk", "vm")?;
    let (outcome, t2_execed, t2t2) = {
        let _tier = TierGuard::new(ForcedTier::T2Lite);
        let _heap = T2HeapGuard::new(true);
        crate::interp::reset_bc_fn_cache();
        crate::interp::reset_t2_cache();
        crate::interp::reset_t2_exec_count();
        crate::interp::reset_t2_module_registry();
        crate::interp::reset_t2_t2_call_count();
        let mut interp = Interp::new();
        interp.install_basic_globals();
        let result = match interp.run_completion_value(src) {
            Ok(v) => Ok(v),
            Err(e) => Err(reduce_thrown(&e)),
        };
        let outcome = TierOutcome {
            result,
            output: interp.output.clone(),
        };
        (
            outcome,
            crate::interp::t2_exec_count(),
            crate::interp::t2_t2_call_count(),
        )
    };
    compare_outcomes(&a, &outcome, "tree-walk", "t2lite-t2t2")?;
    if t2_execed == 0 {
        return Err(Divergence {
            kind: DivergenceKind::SideEffect,
            path: "<t2-engagement>".to_string(),
            tree_walk: "expected the T2 CALLER to execute natively".to_string(),
            vm: "T2-lite executed 0 functions (vacuously green — FAIL)".to_string(),
        });
    }
    if t2t2 == 0 {
        return Err(Divergence {
            kind: DivergenceKind::SideEffect,
            path: "<t2-t2-engagement>".to_string(),
            tree_walk: "expected ≥1 T2→T2 native-to-native callee invocation".to_string(),
            vm: "0 native-to-native T2→T2 calls (callee silently on the VM — FAIL)"
                .to_string(),
        });
    }
    Ok(())
}

/// Compare ONLY the VM tier against the T2-lite HEAP tier (skipping the tree-walk
/// leg). Used where the tree-walk leg has an UNRELATED, pre-existing divergence
/// (e.g. `hole === undefined`) that would mask the T2-vs-VM correctness we actually
/// want to gate — the VM is the canonical oracle T2 deopt-resumes into, so VM ==
/// T2(heap) is the load-bearing equality for a path that DEOPTs to the VM.
pub fn assert_t2_heap_matches_vm(src: &str) -> Result<(), Divergence> {
    let b = run_one_tier(src, ForcedTier::Vm);
    let c = {
        let _tier = TierGuard::new(ForcedTier::T2Lite);
        let _heap = T2HeapGuard::new(true);
        crate::interp::reset_bc_fn_cache();
        crate::interp::reset_t2_cache();
        crate::interp::reset_t2_exec_count();
        let mut interp = Interp::new();
        interp.install_basic_globals();
        let result = match interp.run_completion_value(src) {
            Ok(v) => Ok(v),
            Err(e) => Err(reduce_thrown(&e)),
        };
        TierOutcome { result, output: interp.output.clone() }
    };
    compare_outcomes(&b, &c, "vm", "t2lite-heap")
}

/// Like `assert_tiers_agree`, but ALSO requires the T1 tier to have genuinely
/// executed native code (≥1 T1 function invocation) — guarding against a
/// vacuously-green oracle where T1 silently declines everything. Use for the
/// supported-subset corpus where T1 MUST engage.
pub fn assert_tiers_agree_engaged(src: &str) -> Result<(), Divergence> {
    let a = run_one_tier(src, ForcedTier::TreeWalk);
    let b = run_one_tier(src, ForcedTier::Vm);
    compare_outcomes(&a, &b, "tree-walk", "vm")?;
    // Run T1 and capture how many functions actually ran as native code.
    let c = {
        let _guard = TierGuard::new(ForcedTier::Jit);
        crate::interp::reset_bc_fn_cache();
        crate::interp::reset_t1_cache();
        crate::interp::reset_t1_exec_count();
        let mut interp = Interp::new();
        interp.install_basic_globals();
        let result = match interp.run_completion_value(src) {
            Ok(v) => Ok(v),
            Err(e) => Err(reduce_thrown(&e)),
        };
        let outcome = TierOutcome {
            result,
            output: interp.output.clone(),
        };
        let execed = crate::interp::t1_exec_count();
        (outcome, execed)
    };
    compare_outcomes(&a, &c.0, "tree-walk", "jit")?;
    if c.1 == 0 {
        return Err(Divergence {
            kind: DivergenceKind::SideEffect,
            path: "<t1-engagement>".to_string(),
            tree_walk: "expected T1 to execute ≥1 function natively".to_string(),
            vm: "T1 executed 0 functions (vacuously green — FAIL)".to_string(),
        });
    }
    Ok(())
}

/// Find the first differing console line (or a length mismatch) for a precise
/// side-effect report.
fn first_output_diff(a: &[String], b: &[String]) -> (String, String, String) {
    for (i, (la, lb)) in a.iter().zip(b.iter()).enumerate() {
        if la != lb {
            return (la.clone(), lb.clone(), format!("<console>[{i}]"));
        }
    }
    // Prefix-equal but different lengths.
    (
        format!("{} line(s)", a.len()),
        format!("{} line(s)", b.len()),
        "<console>.len".to_string(),
    )
}

/// Recursion guard: a malformed cyclic structure shouldn't hang the comparator.
const MAX_DEPTH: usize = 64;

/// Deep structural comparison of two completion values. Returns the FIRST
/// divergence (with a path) or `None` if structurally identical.
///
/// Distinguishes everything the engine can observe:
///   - `-0` vs `+0` (bit-different, `Object.is`-distinct),
///   - `NaN` (equal to NaN — two NaNs are the "same" observable value here),
///   - `BigInt` magnitude + sign,
///   - arrays incl. HOLES (a hole is not `undefined`),
///   - objects: own-ENUMERABLE keys in ECMA [[OwnPropertyKeys]] order (integer
///     keys ascending, then string keys in insertion order) AND own symbol keys,
///     recursively-equal values,
///   - function/native/closure: treated as opaque-but-present (callables aren't
///     deep-comparable; we only assert both sides are callable).
fn deep_diff(a: &Value, b: &Value, path: &str, depth: usize) -> Option<Divergence> {
    if depth > MAX_DEPTH {
        return None; // give up rather than loop on a cycle; equal-enough.
    }
    match (a, b) {
        (Value::Undefined, Value::Undefined) => None,
        (Value::Null, Value::Null) => None,
        (Value::Hole, Value::Hole) => None,
        (Value::Bool(x), Value::Bool(y)) if x == y => None,
        (Value::Number(x), Value::Number(y)) => {
            if numbers_same(*x, *y) {
                None
            } else {
                Some(value_div(path, a, b))
            }
        }
        (Value::BigInt(x), Value::BigInt(y)) if x == y => None,
        (Value::String(x), Value::String(y)) if x == y => None,
        (Value::Array(x), Value::Array(y)) => {
            let (xa, ya) = (x.borrow(), y.borrow());
            if xa.len() != ya.len() {
                return Some(Divergence {
                    kind: DivergenceKind::Value,
                    path: format!("{path}.length"),
                    tree_walk: xa.len().to_string(),
                    vm: ya.len().to_string(),
                });
            }
            for i in 0..xa.len() {
                // Holes are observably distinct from `undefined` — compare the
                // raw slot (Value::Hole vs Value::Undefined) directly.
                if let Some(d) = deep_diff(&xa[i], &ya[i], &format!("{path}[{i}]"), depth + 1) {
                    return Some(d);
                }
            }
            None
        }
        (Value::Object(_), Value::Object(_)) => deep_diff_object(a, b, path, depth),
        // Callables: not deep-comparable across tiers (different Rc identity, the
        // VM may even wrap a body as a BcClosure). Assert both are callable.
        (av, bv) if is_callable(av) && is_callable(bv) => None,
        _ => Some(value_div(path, a, b)),
    }
}

/// Object deep-compare: own-enumerable string keys (ECMA order) + own symbol
/// keys must match as a SEQUENCE, and each value must recursively agree.
fn deep_diff_object(a: &Value, b: &Value, path: &str, depth: usize) -> Option<Divergence> {
    let ka = enumerable_string_keys_with_own_symbols(a);
    let kb = enumerable_string_keys_with_own_symbols(b);
    if ka != kb {
        return Some(Divergence {
            kind: DivergenceKind::Value,
            path: format!("{path}.<keys>"),
            tree_walk: format!("[{}]", ka.join(", ")),
            vm: format!("[{}]", kb.join(", ")),
        });
    }
    let (oa, ob) = match (a, b) {
        (Value::Object(x), Value::Object(y)) => (x, y),
        _ => return Some(value_div(path, a, b)),
    };
    for k in &ka {
        let va = oa.borrow().get(k).cloned().unwrap_or(Value::Undefined);
        let vb = ob.borrow().get(k).cloned().unwrap_or(Value::Undefined);
        let child_path = format!("{path}.{k}");
        if let Some(d) = deep_diff(&va, &vb, &child_path, depth + 1) {
            return Some(d);
        }
    }
    None
}

/// Two `f64`s are the "same" observable value iff `Object.is`-equal: NaN matches
/// NaN, and `-0` is DISTINCT from `+0`.
fn numbers_same(x: f64, y: f64) -> bool {
    if x.is_nan() && y.is_nan() {
        return true;
    }
    if x == 0.0 && y == 0.0 {
        // Distinguish -0 from +0 by sign bit.
        return x.is_sign_negative() == y.is_sign_negative();
    }
    x == y
}

fn is_callable(v: &Value) -> bool {
    matches!(
        v,
        Value::Function(_) | Value::NativeFunction(_) | Value::BcClosure(_)
    )
}

fn value_div(path: &str, a: &Value, b: &Value) -> Divergence {
    Divergence {
        kind: DivergenceKind::Value,
        path: path.to_string(),
        tree_walk: canon(a),
        vm: canon(b),
    }
}

/// Canonical, type-distinguishing rendering of a value for divergence reports.
/// Unlike `to_display_string`, this keeps `-0`, `NaN`, `BigInt`, holes, and the
/// string-vs-number distinction visible.
fn canon(v: &Value) -> String {
    match v {
        Value::Undefined => "undefined".to_string(),
        Value::Null => "null".to_string(),
        Value::Hole => "<hole>".to_string(),
        Value::Bool(b) => b.to_string(),
        Value::Number(n) => {
            if n.is_nan() {
                "NaN".to_string()
            } else if *n == 0.0 && n.is_sign_negative() {
                "-0".to_string()
            } else {
                format!("{n}")
            }
        }
        Value::BigInt(b) => format!("{b}n"),
        Value::String(s) => format!("{s:?}"),
        Value::Array(a) => {
            let items: Vec<String> = a.borrow().iter().map(canon).collect();
            format!("[{}]", items.join(", "))
        }
        Value::Object(_) => {
            let keys = enumerable_string_keys_with_own_symbols(v);
            format!("{{{}}}", keys.join(", "))
        }
        Value::Function(_) | Value::NativeFunction(_) | Value::BcClosure(_) => {
            "<function>".to_string()
        }
    }
}

#[cfg(test)]
mod tests;
