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
    // Same for the T4 Maglev-class tier (its own per-function compile cache +
    // engagement counter, so a prior tier's T4 decision / count can't leak in).
    crate::interp::reset_t4_cache();
    crate::interp::reset_t4_exec_count();
    // P4 redundancy-elimination rewrite counter (so a P4 engagement assertion
    // measures THIS run's CSE/copy folds, not a prior tier's).
    crate::t4::reset_redundancy_rewrite_count();
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

/// T4 P1 — run `src` under `tier` with the binary/compare TYPE-FEEDBACK RECORDER
/// force-enabled (`feedback::FeedbackGuard`). The recorder mutates only a side
/// table on the `BcFunction` (never a JS value), so the outcome MUST be
/// byte-identical to [`run_one_tier`] with recording off — that observational
/// invisibility is exactly what the P1 oracle leg proves. The guard is scoped to
/// this run (restored on drop) so it cannot leak into other legs.
fn run_one_tier_feedback(src: &str, tier: ForcedTier) -> TierOutcome {
    let _fb = crate::feedback::FeedbackGuard::new(true);
    run_one_tier(src, tier)
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
    // T4 (the Maglev-class speculative tier) must agree too. In P0 T4 has no
    // codegen and DECLINES on every function, so a T4 run falls through to T3 → T2
    // → VM and is transitively byte-identical to tree-walk; this leg is wired NOW
    // so the moment P2 emits real T4 code (representation selection / inlining)
    // every snippet in the corpus is already gated T4 == VM == tree-walk, and any
    // T4 miscompile reddens the oracle. The deopt-fuzz + the inlined-frame
    // reconstruction fuzzer gate the per-guard bailout separately.
    let g = run_one_tier(src, ForcedTier::T4);
    compare_outcomes(&a, &g, "tree-walk", "t4")?;
    // T4 P1 — TYPE-FEEDBACK RECORDING is observationally INVISIBLE. Re-run the
    // VM and T4 with the binary/compare/call feedback recorder force-ON; the
    // recorder only mutates a side table on the `BcFunction`, never a JS value, so
    // the outcome MUST still be byte-identical to tree-walk. This leg PROVES P1's
    // central claim (recording changes nothing observable) on the whole corpus and
    // is wired NOW so the moment P2 consumes the vector for specialization, every
    // snippet is already gated "recording-on == recording-off == VM == tree-walk".
    let h = run_one_tier_feedback(src, ForcedTier::Vm);
    compare_outcomes(&a, &h, "tree-walk", "vm+feedback")?;
    let k = run_one_tier_feedback(src, ForcedTier::T4);
    compare_outcomes(&a, &k, "tree-walk", "t4+feedback")?;
    Ok(())
}

/// TOP-LEVEL VM oracle — the differential gate for the `CV_TOPLEVEL_VM` lever.
///
/// `assert_tiers_agree` drives FORCED tiers, which `Interp::run`'s top-level VM
/// seam deliberately skips (it only fires when `forced_tier().is_none()`). So this
/// oracle compares the PRODUCTION top-level path two ways through `Interp::run`
/// itself: once with the top-level VM gate OFF (the script's top level is
/// tree-walked — the byte-identical baseline) and once with it ON (the eligible
/// top-level body is compiled to a bytecode Module and run on the register VM).
///
/// "Observable" for the top-level path is exactly what a script can affect:
///   (a) throw parity (same error name+message, or both no-throw) — `run` itself,
///   (b) the ordered `console.*` side effects, and
///   (c) the FINAL VALUE OF EVERY GLOBAL THE SCRIPT TOUCHED — `run` discards the
///       completion value (it returns `undefined` for a normal completion, like
///       the spec ScriptEvaluation a host ignores), so the meaningful output is
///       the global state the body produced (e.g. jit.js's `__bench_jit_result`).
///
/// Returns the first `Divergence`, or `Ok(())` if the two paths agree. Used by the
/// top-level test corpus; also wired into the runtime `CV_JS_VERIFY` gate is NOT
/// done here (the forced-tier oracle already covers per-function VM identity).
pub fn assert_toplevel_vm_agrees(src: &str) -> Result<(), Divergence> {
    // Run `src` through `Interp::run` with the top-level VM gate set to `on`,
    // capturing throw/value parity, console output, and the post-run global state.
    fn run_toplevel(src: &str, on: bool) -> (TierOutcome, Vec<(String, Value)>) {
        // No FORCED tier (so the top-level seam is allowed to fire); reset the
        // per-function caches so a prior run's compile decisions can't leak in.
        let _no_tier = crate::interp::set_force_tier(None);
        crate::interp::reset_bc_fn_cache();
        crate::interp::reset_t1_cache();
        crate::interp::reset_t2_cache();
        crate::interp::reset_t3_cache();
        crate::interp::reset_t4_cache();
        let _g = crate::interp::TopLevelVmGuard::new(on);
        let mut interp = Interp::new();
        interp.install_basic_globals();
        // Snapshot the pristine globals so we can diff only what the SCRIPT added
        // or changed (the standard library is identical on both paths). The global
        // bindings are an insertion-ordered `OrderedMap`; `get`/`&iter` work the
        // same as a HashMap here.
        let base = interp.globals_snapshot();
        let result = match interp.run(src) {
            Ok(v) => Ok(v),
            Err(e) => Err(reduce_thrown(&e)),
        };
        let after = interp.globals_snapshot();
        // Keys whose value the script CREATED or CHANGED vs the pristine library.
        // IMPORTANT: use a CHEAP shallow identity/primitive check to decide "did the
        // script touch this key" — a full `deep_diff` over every global would walk
        // the ENTIRE standard-library object graph (Object/Array/Function prototypes,
        // cyclic `.constructor` back-refs, …) on every call, which is pathologically
        // slow. Unchanged stdlib globals share the SAME Rc, so `shallow_same` returns
        // true in O(1); only a genuinely new/mutated binding is collected, and THOSE
        // (the script's own output) are then deep-compared between the two paths.
        let mut touched: Vec<(String, Value)> = Vec::new();
        for (k, v) in &after {
            let changed = match base.get(k) {
                None => true,
                Some(old) => !shallow_same(old, v),
            };
            if changed {
                touched.push((k.clone(), v.clone()));
            }
        }
        touched.sort_by(|a, b| a.0.cmp(&b.0));
        (
            TierOutcome {
                result,
                output: interp.output.clone(),
            },
            touched,
        )
    }

    let (a, ga) = run_toplevel(src, false);
    let (b, gb) = run_toplevel(src, true);
    // (a)+(b) throw parity, completion (always `undefined` for `run`), side effects.
    compare_outcomes(&a, &b, "toplevel-treewalk", "toplevel-vm")?;

    // (★ PRODUCTION-FAITHFUL OBSERVATION) — the false-green this oracle existed to
    // miss. `globals_snapshot()` reads the raw bindings table AFTER the top-level
    // VM has flushed its deferred writeback; but what a PAGE (and the cv_browser
    // `LiveInterp` host) actually observes is a read THROUGH the `globalThis` /
    // `window` global OBJECT, which happens DURING the same script execution. Inside
    // the top-level VM a `globalThis.X` property read does NOT see the VM's pending
    // `StoreGlobal` writes (they sit in the deferred `globals` cell until the module
    // finishes) — so a script whose written-back snapshot agrees can STILL diverge
    // on the value a page reads back through the global object. Reproduce exactly
    // that production observation here: re-run BOTH paths with a trailing probe that
    // serializes `globalThis[key]` for every touched key through the real property-
    // get path (a console line), and require the two paths to agree line-for-line.
    //
    // This is what turns the BUG3 / BUG2-with-try-catch cases RED: their snapshot
    // writeback matches, but `globalThis.i` read inside the VM run yields `undefined`
    // while the tree-walk path yields the spec value (2 / 3) — exactly the
    // divergence cv_browser sees with CV_TOPLEVEL_VM=1.
    {
        // Candidate keys = the UNION of what either path touched (sorted, deduped),
        // observed through the global object on each path.
        let mut keys: Vec<String> = ga.iter().map(|(k, _)| k.clone()).collect();
        for (k, _) in &gb {
            if !keys.contains(k) {
                keys.push(k.clone());
            }
        }
        keys.sort();
        keys.dedup();
        if !keys.is_empty() {
            let obs_tw = observe_globals_through_object(src, false, &keys);
            let obs_vm = observe_globals_through_object(src, true, &keys);
            if obs_tw != obs_vm {
                let (tw, vm, path) = first_output_diff(&obs_tw, &obs_vm);
                return Err(Divergence {
                    kind: DivergenceKind::Value,
                    path: format!("<globalThis-read>{path}"),
                    tree_walk: format!("toplevel-treewalk: {tw}"),
                    vm: format!("toplevel-vm: {vm}"),
                });
            }
        }
    }

    // (c) the touched-global SET must match key-for-key, value-for-value.
    if ga.len() != gb.len() {
        return Err(Divergence {
            kind: DivergenceKind::Value,
            path: "<globals>.len".to_string(),
            tree_walk: format!(
                "{} touched: [{}]",
                ga.len(),
                ga.iter().map(|(k, _)| k.clone()).collect::<Vec<_>>().join(", ")
            ),
            vm: format!(
                "{} touched: [{}]",
                gb.len(),
                gb.iter().map(|(k, _)| k.clone()).collect::<Vec<_>>().join(", ")
            ),
        });
    }
    for ((ka, va), (kb, vb)) in ga.iter().zip(gb.iter()) {
        if ka != kb {
            return Err(Divergence {
                kind: DivergenceKind::Value,
                path: "<globals>.<keys>".to_string(),
                tree_walk: ka.clone(),
                vm: kb.clone(),
            });
        }
        if let Some(d) = deep_diff(va, vb, &format!("<global>.{ka}"), 0) {
            return Err(d);
        }
    }
    Ok(())
}

/// Run `src` through `Interp::run` with the top-level VM gate set to `on`, with a
/// trailing probe APPENDED TO THE SAME SCRIPT BODY that reads each `keys` entry
/// back THROUGH the global object (`globalThis[key]`) and emits a stable,
/// tier-independent serialization as a console line — exactly the read path a page
/// (or the cv_browser `LiveInterp` host) uses to observe a script's global side
/// effects.
///
/// Returns the captured `__CV_OBS__:…` console lines (one per key). This is THE
/// production-faithful observation that the rest of the oracle was missing:
/// `globals_snapshot()` reads the raw bindings table AFTER the top-level VM has
/// flushed its deferred writeback, but a page reads through `globalThis`/`window`
/// DURING the same script execution — and inside the top-level VM a `globalThis.X`
/// property read does NOT see the VM's pending `StoreGlobal` writes (they sit in the
/// deferred `globals` cell until the module's boundary). Appending the probe to the
/// SAME body makes the read cross the real [[Get]] path at the same point a page's
/// `console.log(window.i)` would, so a snapshot-agreeing-but-page-diverging script
/// (BUG3, BUG2-with-try/catch) is caught.
///
/// A top-level throw in the body skips the appended probe on BOTH paths identically
/// (empty observation sets, which agree) — throw parity is gated separately by the
/// caller. The serialization (`typeof + ':' + String(value)`) is identical across
/// tiers, so a divergence in the emitted lines is a genuine observable-value
/// divergence, never a formatting artifact.
fn observe_globals_through_object(src: &str, on: bool, keys: &[String]) -> Vec<String> {
    let _no_tier = crate::interp::set_force_tier(None);
    crate::interp::reset_bc_fn_cache();
    crate::interp::reset_t1_cache();
    crate::interp::reset_t2_cache();
    crate::interp::reset_t3_cache();
    crate::interp::reset_t4_cache();
    let _g = crate::interp::TopLevelVmGuard::new(on);
    let mut interp = Interp::new();
    interp.install_basic_globals();

    // Build the probe APPENDED to the same body. Each key is read via a computed
    // member access on `globalThis` (the real [[Get]] path) and serialized as
    // `typeof:String(value)`. A throw in one key's read still emits the remaining
    // keys. JSON-escape the key so an unusual global name can't break the syntax.
    //
    // CALLABLE-STABLE serialization: for a FUNCTION-typed global, serialize ONLY
    // `typeof` (always `"function"`) and the sentinel `<callable>`, NOT the
    // source-text `String(fn)`. A function's `.toString()` representation is
    // legitimately tier-dependent (a tree-walk closure renders its source while a
    // VM `BcClosure` renders `function … bc#N`) and is NOT a correctness-observable
    // value — exactly the "callable = opaque-but-present" rule `deep_diff` already
    // applies to completion values. Without this, observing a top-level `var f =
    // function(){…}` global would report a false divergence purely on the function
    // body's textual rendering. Non-function values keep the full `String(value)`.
    let mut probe = String::from("\n;(function(){\n");
    for k in keys {
        let key_lit = json_string_literal(k);
        probe.push_str(&format!(
            "  try {{ var __cv_v = globalThis[{key}]; var __cv_s = (typeof __cv_v === 'function') ? '<callable>' : String(__cv_v); console.log('__CV_OBS__:' + {key} + '=' + (typeof __cv_v) + ':' + __cv_s); }} catch (__cv_e) {{ console.log('__CV_OBS__:' + {key} + '=THROW:' + String(__cv_e)); }}\n",
            key = key_lit
        ));
    }
    probe.push_str("})();\n");

    let full = format!("{src}{probe}");
    let _ = interp.run(&full);
    interp
        .output
        .iter()
        .filter(|l| l.contains("__CV_OBS__:"))
        .cloned()
        .collect()
}

/// Minimal JSON string-literal encoder for embedding a global name safely into the
/// probe source (handles quotes/backslashes/control chars). Never panics.
fn json_string_literal(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

/// Like `assert_toplevel_vm_agrees`, but ALSO requires the top-level VM path to
/// have GENUINELY executed (`toplevel_vm_took_count` > 0 on the `on` run) — the
/// anti-vacuity guard so a green oracle can't be a script that silently declined
/// to the tree-walker on BOTH passes. Use for the corpus where the VM path MUST
/// engage (the `var`+loop+call shape the bottleneck is).
pub fn assert_toplevel_vm_agrees_engaged(src: &str) -> Result<(), Divergence> {
    assert_toplevel_vm_agrees(src)?;
    let engaged = {
        let _no_tier = crate::interp::set_force_tier(None);
        crate::interp::reset_bc_fn_cache();
        let _g = crate::interp::TopLevelVmGuard::new(true);
        crate::interp::reset_toplevel_vm_took_count();
        let mut interp = Interp::new();
        interp.install_basic_globals();
        let _ = interp.run(src);
        crate::interp::toplevel_vm_took_count() > 0
    };
    if !engaged {
        return Err(Divergence {
            kind: DivergenceKind::SideEffect,
            path: "<toplevel-vm-engagement>".to_string(),
            tree_walk: "expected the top-level VM path to run the script".to_string(),
            vm: "top-level VM declined to the tree-walker (vacuously green — FAIL)".to_string(),
        });
    }
    Ok(())
}

/// T4 P1 oracle — like [`assert_tiers_agree`], but ALSO asserts the feedback
/// vector is NON-VACUOUS on a snippet that contains recordable (arith/compare/
/// call) ops: with recording forced on through the VM, at least one feedback slot
/// MUST have filled (`has_any_feedback`). This guards against a vacuously-green
/// P1 oracle where the recorder silently never runs (e.g. a mis-wired gate). It
/// returns the engaged flag so a test can assert it; the byte-identity is checked
/// exactly as `assert_tiers_agree` does.
pub fn assert_tiers_agree_feedback_engaged(src: &str) -> Result<bool, Divergence> {
    assert_tiers_agree(src)?;
    // Run the VM once with recording on and check the honesty counter — the
    // recorder bumps it once per classified observation, so >0 proves the vector
    // truly filled (a mis-wired no-op gate would leave it at 0).
    let engaged = {
        let _fb = crate::feedback::FeedbackGuard::new(true);
        let _guard = TierGuard::new(ForcedTier::Vm);
        crate::interp::reset_bc_fn_cache();
        crate::feedback::reset_feedback_record_count();
        let mut interp = Interp::new();
        interp.install_basic_globals();
        let _ = interp.run_completion_value(src);
        crate::feedback::feedback_record_count() > 0
    };
    Ok(engaged)
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

/// Like `assert_tiers_agree`, but ALSO requires the T4 Maglev-class tier to have
/// genuinely executed native code (≥1 T4 invocation) — guarding against a
/// vacuously-green oracle where T4 silently declines/deopts everything. Use for
/// the numeric/loop/arith kernel corpus where T4's representation-selection
/// backend MUST engage. Byte-identity to tree-walk (transitively VM) is checked
/// exactly as `assert_tiers_agree`; engagement via `t4_exec_count`.
pub fn assert_tiers_agree_t4_engaged(src: &str) -> Result<(), Divergence> {
    let a = run_one_tier(src, ForcedTier::TreeWalk);
    let b = run_one_tier(src, ForcedTier::Vm);
    compare_outcomes(&a, &b, "tree-walk", "vm")?;
    let c = {
        let _guard = TierGuard::new(ForcedTier::T4);
        crate::interp::reset_bc_fn_cache();
        crate::interp::reset_t4_cache();
        crate::interp::reset_t4_exec_count();
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
        let execed = crate::interp::t4_exec_count();
        (outcome, execed)
    };
    compare_outcomes(&a, &c.0, "tree-walk", "t4")?;
    if c.1 == 0 {
        return Err(Divergence {
            kind: DivergenceKind::SideEffect,
            path: "<t4-engagement>".to_string(),
            tree_walk: "expected T4 to execute ≥1 function natively".to_string(),
            vm: "T4 executed 0 functions (vacuously green — FAIL)".to_string(),
        });
    }
    Ok(())
}

/// T4 P4 oracle — like [`assert_tiers_agree_t4_engaged`], but ALSO requires the
/// P4 REDUNDANCY/LOAD/CHECK ELIMINATION pass to have genuinely rewritten ≥1 op
/// (`redundancy_rewrite_count` > 0) on this snippet, so a green P4 oracle can never
/// be vacuously green (T4 engaging but the redundancy pass folding nothing). The
/// byte-identity (T4 == VM == tree-walk) gate is exactly `assert_tiers_agree_t4_engaged`;
/// this adds the non-vacuity engagement check for P4 specifically. Use on a corpus
/// with KNOWN redundancy (a repeated `x*x`, a copied-then-reused value).
pub fn assert_tiers_agree_t4_redundancy_engaged(src: &str) -> Result<(), Divergence> {
    let a = run_one_tier(src, ForcedTier::TreeWalk);
    let b = run_one_tier(src, ForcedTier::Vm);
    compare_outcomes(&a, &b, "tree-walk", "vm")?;
    let (outcome, t4_execed, redun) = {
        let _guard = TierGuard::new(ForcedTier::T4);
        crate::interp::reset_bc_fn_cache();
        crate::interp::reset_t4_cache();
        crate::interp::reset_t4_exec_count();
        crate::t4::reset_redundancy_rewrite_count();
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
            crate::interp::t4_exec_count(),
            crate::t4::redundancy_rewrite_count(),
        )
    };
    compare_outcomes(&a, &outcome, "tree-walk", "t4")?;
    if t4_execed == 0 {
        return Err(Divergence {
            kind: DivergenceKind::SideEffect,
            path: "<t4-engagement>".to_string(),
            tree_walk: "expected T4 to execute ≥1 function natively".to_string(),
            vm: "T4 executed 0 functions (vacuously green — FAIL)".to_string(),
        });
    }
    if redun == 0 {
        return Err(Divergence {
            kind: DivergenceKind::SideEffect,
            path: "<t4-p4-redundancy-engagement>".to_string(),
            tree_walk: "expected the P4 redundancy pass to rewrite ≥1 op".to_string(),
            vm: "P4 redundancy pass folded 0 ops (vacuously green — FAIL)".to_string(),
        });
    }
    Ok(())
}

/// T4 P5 AOT-PERSIST round-trip oracle — the ★ beat-Chrome cold-repeat gate.
///
/// Proves the persisted optimized NATIVE CODE is byte-identical to the VM AFTER a
/// full persist → reload → run cycle (the cold-repeat-visit path V8 cannot take):
///   1. Run `src` under `ForcedTier::T4` with AOT persist FORCED ON in a FRESH,
///      isolated on-disk store (a unique temp dir per call). This compiles the hot
///      function fresh and PERSISTS its native blob (`aot_store_count` > 0).
///   2. SIMULATE A COLD REPEAT VISIT: clear the in-process T4 compile cache (so the
///      next run can't reuse the live `JitFunction` — it MUST reconstruct from
///      disk), reset the AOT load counter, and run `src` AGAIN under `ForcedTier::T4`
///      + AOT on. This time the function's native code is RE-INSTALLED FROM THE DISK
///      BLOB with zero codegen (`aot_load_count` > 0 — the non-vacuity guard).
///   3. The reloaded run's outcome MUST be byte-identical to the plain VM (and thus
///      tree-walk) — the persisted code + re-attached DeoptSite table runs exactly
///      as fresh T4 would. This is the round-trip correctness gate.
///
/// Returns `Ok(true)` iff the reload path actually fired (a blob was re-installed
/// from disk), so a caller can assert non-vacuity; `Ok(false)` means T4 declined to
/// produce a persistable native function for this snippet (no inlinable numeric
/// call — the round-trip is vacuous, not a failure). Any byte-identity divergence
/// is returned as `Err` (the round-trip produced a WRONG value — a hard fail).
#[cfg(target_os = "windows")]
pub fn assert_aot_roundtrip_matches_vm(src: &str) -> Result<bool, Divergence> {
    // The VM baseline (and tree-walk, transitively) — the source of truth.
    let a = run_one_tier(src, ForcedTier::TreeWalk);
    let b = run_one_tier(src, ForcedTier::Vm);
    compare_outcomes(&a, &b, "tree-walk", "vm")?;

    // Isolate the on-disk AOT store to a unique temp dir so this oracle never reads
    // a stale blob from a prior run / another test and the reload it observes is
    // exactly the one IT persisted. Cleaned up at the end (best-effort). The dir
    // must be unique even across PARALLEL test threads, so we use a PROCESS-WIDE
    // atomic sequence (not a thread-local — two threads would both start at 0 and
    // collide) plus the pid + a nanosecond clock component.
    let seq = AOT_ORACLE_SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let dir = std::env::temp_dir().join(format!(
        "tbjs_aot_oracle_{}_{}_{}",
        std::process::id(),
        seq,
        nanos
    ));
    let _ = std::fs::remove_dir_all(&dir);
    // CV_AOT_PERSIST_DIR is read fresh by `aot_dir()` on each store/load, so setting
    // it here routes THIS oracle call's disk I/O to the unique dir. NOTE: the env var
    // is process-global; parallel oracle calls would race on it. To avoid that, the
    // AOT disk seam ALSO honors a thread-local dir override (set below), which takes
    // precedence over the env — so parallel oracle threads never collide.
    crate::t4::aot::set_thread_dir_override(Some(dir.clone()));
    let _persist = crate::t4::aot::AotPersistGuard::new(true);

    // ── Run 1 — fresh compile + PERSIST the native blob.
    let run_t4 = |store_before: u64| -> (TierOutcome, u64, u64) {
        let _guard = TierGuard::new(ForcedTier::T4);
        crate::interp::reset_bc_fn_cache();
        crate::interp::reset_t4_cache();
        crate::interp::reset_t4_exec_count();
        crate::t4::aot::reset_aot_load_count();
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
            crate::t4::aot::aot_store_count().saturating_sub(store_before),
            crate::t4::aot::aot_load_count(),
        )
    };

    crate::t4::aot::reset_aot_store_count();
    let (out1, stored, _l1) = run_t4(0);
    compare_outcomes(&a, &out1, "tree-walk", "t4-aot-warm")?;

    // ── Run 2 — SIMULATE A COLD REPEAT VISIT: the in-process T4 cache is cleared
    //    (run_t4 resets it), so the function MUST reconstruct from the disk blob.
    let (out2, _stored2, loaded) = run_t4(stored);
    compare_outcomes(&a, &out2, "tree-walk", "t4-aot-cold-repeat")?;

    // Cleanup (best-effort).
    crate::t4::aot::set_thread_dir_override(None);
    let _ = std::fs::remove_dir_all(&dir);

    // Non-vacuity: the reload fired iff a blob was both persisted (run 1) AND
    // re-installed from disk (run 2). If T4 never produced a persistable native
    // function (no inlinable numeric call), `stored == 0` and the round-trip is
    // vacuous — Ok(false), not a failure.
    Ok(stored > 0 && loaded > 0)
}

#[cfg(not(target_os = "windows"))]
pub fn assert_aot_roundtrip_matches_vm(src: &str) -> Result<bool, Divergence> {
    // No RX install on non-Windows → no native AOT path; just prove VM==tree-walk.
    let a = run_one_tier(src, ForcedTier::TreeWalk);
    let b = run_one_tier(src, ForcedTier::Vm);
    compare_outcomes(&a, &b, "tree-walk", "vm")?;
    Ok(false)
}

/// Process-wide sequence so each `assert_aot_roundtrip_matches_vm` call (across ALL
/// test threads) uses a unique on-disk store dir (no cross-call contamination).
static AOT_ORACLE_SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

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
/// CHEAP, NON-RECURSIVE "is this the same binding" test used by the top-level
/// oracle to decide whether the SCRIPT touched a given global. Compares two values
/// taken from the SAME interpreter (a pristine snapshot vs the post-run snapshot):
/// heap values are compared by Rc POINTER IDENTITY (an unchanged stdlib global is
/// the same Rc → O(1) true, so we never walk the huge cyclic stdlib object graph),
/// and primitives by value. A reassigned-or-new binding fails this and is then
/// deep-compared across the two paths by `deep_diff`.
fn shallow_same(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Undefined, Value::Undefined) => true,
        (Value::Null, Value::Null) => true,
        (Value::Hole, Value::Hole) => true,
        (Value::Bool(x), Value::Bool(y)) => x == y,
        (Value::Number(x), Value::Number(y)) => numbers_same(*x, *y),
        (Value::BigInt(x), Value::BigInt(y)) => std::rc::Rc::ptr_eq(x, y) || x == y,
        (Value::String(x), Value::String(y)) => x == y,
        (Value::Object(x), Value::Object(y)) => std::rc::Rc::ptr_eq(x, y),
        (Value::Array(x), Value::Array(y)) => std::rc::Rc::ptr_eq(x, y),
        (Value::Function(x), Value::Function(y)) => std::rc::Rc::ptr_eq(x, y),
        (Value::NativeFunction(x), Value::NativeFunction(y)) => std::rc::Rc::ptr_eq(x, y),
        (Value::BcClosure(x), Value::BcClosure(y)) => std::rc::Rc::ptr_eq(x, y),
        _ => false,
    }
}

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
