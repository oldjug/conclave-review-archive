//! T4 — the MAGLEV-CLASS speculative optimizing tier (PHASE P2:
//! representation selection + unboxed Float64).
//!
//! T4 sits ABOVE T3. It REUSES T3's optimizer pipeline verbatim (bytecode → CFG →
//! SSA-ish → const-fold/copy-prop/DCE/redundant-guard/LICM → linear-scan regalloc,
//! plus the B3 safepoint-map UAF gate via `t3::optimize_with_safepoints`), then —
//! instead of handing the optimized bytecode to the T2-lite backend that re-boxes
//! and re-tag-checks every operand on every op — emits native code through the new
//! REPRESENTATION-AWARE backend `jit::compile_t4_unboxed_with_deopt`.
//!
//! ## What P2 adds over T3/T2 (the float-dense win, V8 Maglev-shaped)
//!
//! V8's Maglev assigns each `ValueNode` a concrete `ValueRepresentation`
//! (kInt32/kFloat64/kTagged) and places type-check guards (CheckNumber/CheckSmi/
//! CheckedTaggedToFloat64) only WHERE a tagged value enters the unboxed domain,
//! then runs the arithmetic UNBOXED. T4 P2 does the Float64 case of exactly this:
//! within a basic block it keeps each register's unboxed f64 resident in an XMM
//! and reads same-block operands straight from the XMM, so the per-op
//! reload + is-number tag-check + unbox round-trip the T2-lite backend pays on
//! every intermediate is ELIMINATED. The guard (a `DeoptSite`) fires once, where a
//! value first enters the unboxed domain (a fresh bank operand — a parameter or a
//! cross-block value), exactly as Maglev places its CheckNumber.
//!
//! ## Why it is byte-identical to the VM (the non-negotiable gate)
//!
//! Every op STILL boxes its result and stores it to its bank slot (the
//! identity-map invariant: the bank is the exact pre-op VM register image at every
//! op boundary), and every guard is the SAME per-guard resume `DeoptSite` the
//! proven T2-lite path emits (bc_pc == the op index). So a non-number operand
//! deopts to the VM frame BYTE-IDENTICALLY — the T4 native code resumes the VM on
//! the OPTIMIZED module (carried on the `JitFunction`, exactly as T3), which is
//! observationally equivalent to the original. The XMM cache is a pure
//! performance shadow of the bank, invalidated at every basic-block boundary, and
//! is NEVER read on the deopt path (deopt decodes the bank). The A/B oracle
//! (`ForcedTier::T4`) proves `T4 == VM == tree-walk` across the corpus, and the
//! deopt-fuzzer force-deopts every op to prove the resumed VM result is identical.
//!
//! ## Gating
//!
//! `t3::t4_enabled()` (env `CV_T4`, DEFAULT OFF; `ForcedTier::T4` override for the
//! oracle). When off, the default build is byte-identical (T4 declines, the
//! dispatcher falls to T3/T2/VM). Any function outside the numeric/control-flow
//! subset declines → T3/T2/VM run it (always correct).

use crate::bytecode::Module;

/// The outcome of a T4 compile attempt (mirrors `T3CompileStatus`).
pub enum T4CompileStatus {
    /// Optimized AND installed as representation-specialized native code.
    Ready(crate::jit::JitFunction),
    /// T4 declined (unsupported op / shape) — the caller runs T3/T2/VM.
    Decline,
}

/// Compile `module.fns[fn_idx]` through the T3 optimizer and then the T4
/// representation-aware backend (`jit::compile_t4_unboxed_with_deopt`). Returns
/// `Ready` with installed native code, or `Decline` (the caller falls through to
/// T3/T2/VM — always correct).
///
/// The optimized function is wrapped in a single-function `Module` and stashed on
/// the `JitFunction` (`with_t3_module`) so a deopt resumes the VM on the OPTIMIZED
/// module (the identity-map module the native code mirrors) — observationally
/// identical to the original, hence bit-identical to running the original on the
/// VM (the A/B oracle proves this). The same `run_t3_call` runner executes it.
#[cfg(target_os = "windows")]
pub fn try_compile_t4_status(module: &Module, fn_idx: usize) -> T4CompileStatus {
    let f = match module.fns.get(fn_idx) {
        Some(f) => f,
        None => return T4CompileStatus::Decline,
    };
    // REUSE T3's optimizer + the B3 safepoint UAF gate verbatim. A map whose
    // pointer roots aren't all bank-resident is declined (never installed). For
    // the numeric subset the map carries no pointer roots, so this always passes;
    // it becomes load-bearing when T4 widens to heap ops (P3 inlining).
    let (optimized, _stats, safepoints) = match crate::t3::optimize_with_safepoints(f) {
        Ok(x) => x,
        Err(_) => return T4CompileStatus::Decline,
    };

    // Compile the optimized bytecode with the REPRESENTATION-AWARE backend. It
    // emits the same prolog / DeoptSites / epilogue as the T2-lite numeric path
    // PLUS the per-block unboxed-f64 value cache (the win). Any op outside the
    // numeric subset → None → decline (T3/T2/VM run it).
    let consts = optimized.consts.clone();
    let compiled = crate::jit::compile_t4_unboxed_with_deopt(&optimized.code, move |k| {
        match consts.get(k as usize) {
            Some(crate::interp::Value::Number(n)) => Some(*n),
            _ => None,
        }
    });
    let (code, deopt_sites) = match compiled {
        Some(x) => x,
        None => return T4CompileStatus::Decline,
    };

    // Pin heap mode OFF for the run (numeric store mode), matching the compile —
    // the T4 backend always uses `T2StoreMode::Numeric`, so the run-time bank must
    // be the numeric bank. (`run_t3_call` already forces heap off; this guard
    // covers the in-process install path symmetrically.)
    let _heap = crate::interp::T2HeapGuard::new(false);
    let native = match crate::jit::JitFunction::install(&code) {
        Ok(jf) => jf,
        Err(_) => return T4CompileStatus::Decline,
    };
    let opt_module = std::rc::Rc::new(Module { fns: vec![optimized] });
    T4CompileStatus::Ready(
        native
            .with_deopt_sites(deopt_sites)
            .with_t3_module(opt_module)
            .with_safepoints(safepoints),
    )
}

#[cfg(not(target_os = "windows"))]
pub fn try_compile_t4_status(_module: &Module, _fn_idx: usize) -> T4CompileStatus {
    T4CompileStatus::Decline
}

/// Thin wrapper returning `Some` only on `Ready`.
#[cfg(target_os = "windows")]
pub fn try_compile_t4(module: &Module, fn_idx: usize) -> Option<crate::jit::JitFunction> {
    match try_compile_t4_status(module, fn_idx) {
        T4CompileStatus::Ready(jf) => Some(jf),
        _ => None,
    }
}

#[cfg(not(target_os = "windows"))]
pub fn try_compile_t4(_module: &Module, _fn_idx: usize) -> Option<crate::jit::JitFunction> {
    None
}

#[cfg(test)]
mod tests;
