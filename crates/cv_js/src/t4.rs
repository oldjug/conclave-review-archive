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

use crate::bytecode::{BcFunction, Module, Op};

pub mod aot;
pub mod redundancy;

// ======================================================================
// T4 (Maglev-class) PHASE P3 — CROSS-FUNCTION INLINING (the jit.js keystone).
//
// V8 SOURCE MODELED: `src/compiler/js-inlining.*` (JSInliner) + Maglev's
// `MaglevGraphBuilder::BuildInlined` — at a monomorphic call site whose target is
// a small, known callee, the optimizer SPLICES the callee's IR into the caller and
// re-runs representation selection over the fused body, eliminating the call frame
// + call overhead. The deopt at a guard inside the inlined region reconstructs the
// caller (and, in V8's full translation, the callee) frame; our INLINE-DEOPT-TO-
// CALLER design (osr.rs Extension 1, fuzz-proven in P0) resumes the CALLER at the
// `Call` op so the VM performs the ordinary non-inlined call — one extra rare
// re-execution, never a wrong value.
//
// WHAT THIS PHASE DOES: a bytecode→bytecode transform that inlines a `CallFn` to a
// small numeric-subset sibling callee into the caller, then feeds the FUSED
// function through the existing representation-aware backend
// (`jit::compile_t4_unboxed_with_deopt_mapped`) with a RESUME-PC MAP that routes
// every fused-op guard's deopt back to the corresponding ORIGINAL caller op (the
// inlined region → the caller's `Call` op). The result runs on the EXISTING T2/T3
// resume machinery; only the resume MODULE is the original caller (carried via
// `JitFunction::with_t4_deopt_module`).
//
// WHY IT IS BYTE-IDENTICAL TO THE VM (the non-negotiable gate): (1) the fused body
// computes EXACTLY the caller's observable result — the callee body is spliced with
// its registers remapped to a fresh window ABOVE the caller's regs, its params
// seeded by copying (not moving) the caller's arg slots, and its `Ret` replaced by
// a single store of the call result to the call's `dst`; (2) the bank store-after-
// every-op invariant is preserved, so at every guard `bank[0..caller_n_regs]` is a
// complete caller register image at the MAPPED resume op; (3) a guard failure deopts
// to the original caller at that op (the INLINE-DEOPT-TO-CALLER reconstruction),
// where the VM re-runs the real call/op byte-identically. The A/B oracle
// (`ForcedTier::T4` + the inline-engagement leg) proves T4-with-inlining == VM ==
// tree-walk, and the inlined-frame deopt fuzzer force-deopts every inlined-region
// op to prove the resumed VM result is identical.
//
// HEURISTICS (V8-shaped, bounded): only a `CallFn` whose callee is in the numeric/
// control-flow subset, whose op count is ≤ `INLINE_MAX_CALLEE_OPS`, whose param
// count matches `n_args`, that contains NO further calls (inline depth bounded to
// 1), and that does not use a rest param / closure capture. Anything else → no
// inlining (the function compiles without inlining, or declines to T3/T2/VM).
// ======================================================================

/// V8-style bounded-inlining size cap: the maximum callee bytecode op count T4 will
/// splice into a caller. Small enough that the fused body stays compile-cheap (the
/// Maglev tradeoff — cheap compile, not maximal inlining), large enough to absorb a
/// jit.js-shaped numeric helper. A bigger callee declines inlining (the call stays,
/// and the caller — now containing a call — declines T4 entirely → T3/T2/VM).
const INLINE_MAX_CALLEE_OPS: usize = 64;

/// Whether op `op` is in the T4 numeric/control-flow subset the inliner + backend
/// accept. MUST stay a subset of `jit::compile_t4_unboxed_with_deopt_mapped`'s
/// accepted set — an op the inliner admits but the backend rejects would just make
/// the fused compile decline (still correct, never wrong), but keeping them aligned
/// avoids a pointless decline.
pub(crate) fn op_in_numeric_subset(op: &Op) -> bool {
    matches!(
        op,
        Op::LoadConst { .. }
            | Op::LoadUndef { .. }
            | Op::LoadTrue { .. }
            | Op::LoadFalse { .. }
            | Op::LoadNull { .. }
            | Op::Move { .. }
            | Op::Add { .. }
            | Op::Sub { .. }
            | Op::Mul { .. }
            | Op::Div { .. }
            | Op::Lt { .. }
            | Op::Le { .. }
            | Op::Gt { .. }
            | Op::Ge { .. }
            | Op::Eq { .. }
            | Op::Neq { .. }
            | Op::LooseEq { .. }
            | Op::LooseNeq { .. }
            | Op::Jmp { .. }
            | Op::JmpIfFalse { .. }
            | Op::Ret { .. }
    )
}

/// Is `callee` a legal INLINE TARGET? It must be entirely numeric-subset, small,
/// have exactly `n_args` params, no rest param, and contain NO call op (so the
/// inline depth is bounded to 1 — a callee that itself calls would need recursive
/// inlining + multi-frame deopt translation, deferred). Returns true iff safe.
pub(crate) fn callee_is_inlinable(callee: &BcFunction, n_args: usize) -> bool {
    if callee.n_params as usize != n_args {
        return false; // arity mismatch — the VM would bind missing/extra args; decline.
    }
    if callee.rest_reg.is_some() {
        return false; // rest param: variadic gather the inliner doesn't model.
    }
    if callee.code.is_empty() || callee.code.len() > INLINE_MAX_CALLEE_OPS {
        return false;
    }
    // Every op must be numeric-subset (no calls/heap/try/closures). This also
    // rules out a callee with a nested call (depth bound) since CallFn/CallValue
    // are not in the subset.
    callee.code.iter().all(op_in_numeric_subset)
}

/// The result of inlining: the FUSED caller `BcFunction` (callee spliced in), the
/// per-fused-op RESUME-PC MAP (fused index → original caller op index, for the
/// backend's `DeoptSite.bc_pc`), and the count of inlined call sites (for the
/// engagement honesty guard). The fused function is what codegen runs over; the
/// ORIGINAL caller module is what a deopt resumes the VM on.
pub struct InlineResult {
    /// The fused caller function (callee body spliced in place of the call).
    pub fused: BcFunction,
    /// `bc_pc_map[i]` = the ORIGINAL caller bytecode op index a guard emitted during
    /// fused op `i` resumes the VM at. Length == `fused.code.len()`.
    pub bc_pc_map: Vec<usize>,
    /// Number of call sites inlined (≥1 on success — the engagement guard).
    pub inlined_calls: usize,
}

/// Inline the FIRST inlinable `CallFn` in `caller` (depth bound 1, single-site —
/// the smallest correct unit; widening to multi-site is additive). Returns `None`
/// if there is no inlinable call (the caller then compiles without inlining, or
/// declines because it still contains a call).
///
/// THE TRANSFORM (single `CallFn { dst, fn_idx, first_arg, n_args }` at caller op
/// `call_pc`, callee `g`):
///   * Caller regs `0..caller_n_regs` stay in place. The callee window starts at
///     `base = caller_n_regs`: callee reg `r` → fused reg `base + r`. This keeps
///     EVERY caller slot at its original index (the INLINE-DEOPT-TO-CALLER bank
///     invariant) and gives the callee disjoint scratch.
///   * Before the inlined body: copy each arg into the callee's param slot
///     (`Move { dst: base + p, src: first_arg + p }`) — a COPY, so the caller's arg
///     slots are preserved for the deopt-to-call reconstruction.
///   * The callee body is appended with every register remapped by `+base` and
///     every jump target remapped into the inlined region. The callee's `Ret { src
///     }` becomes `Move { dst: call.dst, src: base + src }` (store the result) — the
///     ONLY write to a caller slot the inlined region performs, and it is the LAST
///     op of the inlined region (after all guards), so a mid-inline deopt never
///     leaves a half-written `dst`. A callee may have MULTIPLE `Ret`s (early
///     returns); each becomes the store + a `Jmp` to the post-call continuation.
///   * Every op AFTER the call shifts later in the fused code; its `bc_pc_map` entry
///     is its ORIGINAL caller index so a deopt there resumes the original caller at
///     the right op.
///   * Every fused op in the inlined region maps to `call_pc` (the caller's `Call`
///     op) so an inlined-region guard deopts to the call — the VM re-runs it.
#[cfg(target_os = "windows")]
pub fn inline_first_call(module: &Module, caller_idx: usize) -> Option<InlineResult> {
    let caller = module.fns.get(caller_idx)?;
    // The caller itself must be numeric-subset EXCEPT for the single call we inline
    // (and any other call → decline, since the post-inline body must be fully
    // numeric for the backend). Find the first inlinable CallFn.
    let mut call_site: Option<(usize, u16, u16, u16, u16)> = None; // (pc,dst,fn_idx,first_arg,n_args)
    for (pc, op) in caller.code.iter().enumerate() {
        if let Op::CallFn { dst, fn_idx, first_arg, n_args } = *op {
            let callee = module.fns.get(fn_idx as usize)?;
            if callee_is_inlinable(callee, n_args as usize) {
                call_site = Some((pc, dst, fn_idx, first_arg, n_args as u16));
                break;
            } else {
                return None; // a CallFn we can't inline → caller can't go numeric.
            }
        }
    }
    let (call_pc, dst, fn_idx, first_arg, n_args) = call_site?;
    // Every OTHER op in the caller must be numeric-subset (else the fused body has a
    // non-subset op the backend rejects → just decline here cleanly).
    for (pc, op) in caller.code.iter().enumerate() {
        if pc == call_pc {
            continue;
        }
        if matches!(op, Op::CallFn { .. } | Op::CallValue { .. } | Op::New { .. }) {
            return None; // a second call — single-site inlining only (bounded).
        }
        if !op_in_numeric_subset(op) {
            return None;
        }
    }
    let callee = &module.fns[fn_idx as usize];
    let base = caller.n_regs; // callee window start (callee reg r → base + r)
    // Guard against register-index overflow (u16): base + callee.n_regs must fit.
    let fused_n_regs = (base as u32) + (callee.n_regs as u32);
    if fused_n_regs > u16::MAX as u32 {
        return None;
    }
    let fused_n_regs = fused_n_regs as u16;

    // ── Build the fused code in three regions: [0..call_pc) caller prefix,
    //    [inlined callee body], [call_pc+1..] caller suffix. We track, for each
    //    region, the mapping fused-index → original-caller-index for the resume map,
    //    and the fused offsets of every caller op so the caller's jumps re-target.
    let mut fused: Vec<Op> = Vec::with_capacity(caller.code.len() + callee.code.len() + n_args as usize);
    let mut bc_pc_map: Vec<usize> = Vec::with_capacity(fused.capacity());
    // caller original op index -> fused offset (for re-targeting caller jumps).
    let mut caller_fused_off: Vec<usize> = vec![usize::MAX; caller.code.len()];

    // Region 1 — caller PREFIX [0..call_pc).
    for (pc, op) in caller.code.iter().enumerate().take(call_pc) {
        caller_fused_off[pc] = fused.len();
        fused.push(*op);
        bc_pc_map.push(pc); // resume at this caller op (identity in the prefix).
    }

    // Region 2a — seed callee params by COPYING caller args (preserves arg slots).
    // These copies resume at the Call op if they deopt (they can't — a Move never
    // guards — but the map must still point somewhere valid → the call op).
    for p in 0..n_args {
        fused.push(Op::Move { dst: base + p, src: first_arg + p });
        bc_pc_map.push(call_pc);
    }

    // Region 2b — the INLINED CALLEE BODY (remap regs +base, jumps into-region,
    // Ret → store-result + jump-to-continuation). The continuation target is the
    // first op of the caller SUFFIX (i.e. just after the whole inlined region); we
    // patch the Ret-jumps once we know it.
    let callee_region_start = fused.len();
    // Reserve placeholder offsets for callee ops so we can remap jump targets.
    let mut callee_fused_off: Vec<usize> = Vec::with_capacity(callee.code.len());
    // First pass: lay out callee ops (remap regs + record offsets), recording which
    // pushed ops are Ret-stores that need a jmp-to-continuation patched.
    let mut ret_jmp_patch: Vec<usize> = Vec::new(); // fused indices of the inserted Jmp ops
    for cop in &callee.code {
        callee_fused_off.push(fused.len());
        match *cop {
            Op::Ret { src } => {
                // Store the callee's return value into the call's dst (the only
                // caller-slot write the inlined region makes), then jump to the
                // continuation. For the FINAL Ret that is the natural fall-through
                // we still emit the store + a jmp (patched to the continuation) —
                // uniform handling of multiple/early returns.
                fused.push(Op::Move { dst, src: base + src });
                bc_pc_map.push(call_pc);
                // A placeholder Jmp (target patched after we know the continuation).
                let jmp_idx = fused.len();
                fused.push(Op::Jmp { target: 0 });
                bc_pc_map.push(call_pc);
                ret_jmp_patch.push(jmp_idx);
            }
            other => {
                let remapped = remap_callee_op(other, base);
                fused.push(remapped);
                bc_pc_map.push(call_pc); // inlined-region guard → resume at the Call op.
            }
        }
    }
    let _ = callee_region_start;

    // Region 3 — caller SUFFIX (call_pc+1 .. end). Its first op is the continuation.
    let continuation = fused.len();
    for (pc, op) in caller.code.iter().enumerate().skip(call_pc + 1) {
        caller_fused_off[pc] = fused.len();
        fused.push(*op);
        bc_pc_map.push(pc);
    }

    // ── Patch jump targets.
    // (a) callee internal jumps: target was a callee op index → its fused offset.
    for (k, cop) in callee.code.iter().enumerate() {
        let fused_idx = callee_fused_off[k];
        match *cop {
            Op::Jmp { target } => {
                if let Op::Jmp { target: t } = &mut fused[fused_idx] {
                    *t = callee_fused_off
                        .get(target as usize)
                        .copied()
                        .map(|o| o as u16)?;
                }
            }
            Op::JmpIfFalse { target, .. } => {
                if let Op::JmpIfFalse { target: t, .. } = &mut fused[fused_idx] {
                    *t = callee_fused_off
                        .get(target as usize)
                        .copied()
                        .map(|o| o as u16)?;
                }
            }
            _ => {}
        }
    }
    // (b) the Ret-store Jmps → the continuation.
    for &jmp_idx in &ret_jmp_patch {
        if let Op::Jmp { target } = &mut fused[jmp_idx] {
            *target = continuation as u16;
        }
    }
    // (c) caller jumps: their original target op index → its fused offset (the
    // caller suffix/prefix moved). A target landing on the inlined-away Call op is
    // impossible (you can't jump INTO the middle of a call's result), so every
    // caller jump target is a real caller op with a recorded fused offset.
    for pc in 0..caller.code.len() {
        if pc == call_pc {
            continue;
        }
        let fused_idx = caller_fused_off[pc];
        if fused_idx == usize::MAX {
            continue;
        }
        match caller.code[pc] {
            Op::Jmp { target } => {
                let off = *caller_fused_off.get(target as usize)?;
                if off == usize::MAX {
                    return None;
                }
                if let Op::Jmp { target: t } = &mut fused[fused_idx] {
                    *t = off as u16;
                }
            }
            Op::JmpIfFalse { target, .. } => {
                let off = *caller_fused_off.get(target as usize)?;
                if off == usize::MAX {
                    return None;
                }
                if let Op::JmpIfFalse { target: t, .. } = &mut fused[fused_idx] {
                    *t = off as u16;
                }
            }
            _ => {}
        }
    }

    // The fused function carries the caller's consts (LoadConst k indices unchanged
    // for caller ops) followed by the callee's consts (remapped). To keep const
    // indices valid for BOTH, append the callee consts and remap the callee's
    // LoadConst k by the caller const-pool length.
    let mut consts = caller.consts.clone();
    let callee_const_base = consts.len();
    consts.extend(callee.consts.iter().cloned());
    // Remap callee LoadConst k indices (they were laid down with the callee's own k;
    // bump by callee_const_base). Walk the inlined region we just emitted.
    for (k, cop) in callee.code.iter().enumerate() {
        if let Op::LoadConst { .. } = *cop {
            let fused_idx = callee_fused_off[k];
            if let Op::LoadConst { k: kk, .. } = &mut fused[fused_idx] {
                *kk = (*kk as usize + callee_const_base) as u16;
            }
        }
    }
    // A const index that overflows u16 declines (correctness over coverage).
    if consts.len() > u16::MAX as usize {
        return None;
    }

    debug_assert_eq!(fused.len(), bc_pc_map.len(), "resume-pc map must cover every fused op");

    let fused_fn = BcFunction {
        name: format!("{}+inline", caller.name),
        n_params: caller.n_params,
        rest_reg: caller.rest_reg,
        n_regs: fused_n_regs,
        consts,
        code: fused,
        ic: std::cell::RefCell::new(Vec::new()),
        feedback: std::cell::RefCell::new(Vec::new()),
        strict: caller.strict,
    };
    Some(InlineResult {
        fused: fused_fn,
        bc_pc_map,
        inlined_calls: 1,
    })
}

#[cfg(not(target_os = "windows"))]
pub fn inline_first_call(_module: &Module, _caller_idx: usize) -> Option<InlineResult> {
    None
}

/// Remap a callee op's register operands by `+base` (the callee window offset).
/// Jump TARGETS are remapped separately (after offsets are known); here we copy the
/// target through unchanged and fix it in the patch pass. Only numeric-subset ops
/// reach here (Ret is handled by the caller). LoadConst's `k` is remapped to the
/// fused const pool by the caller, not here.
pub(crate) fn remap_callee_op(op: Op, base: u16) -> Op {
    match op {
        Op::LoadConst { dst, k } => Op::LoadConst { dst: dst + base, k },
        Op::LoadUndef { dst } => Op::LoadUndef { dst: dst + base },
        Op::LoadTrue { dst } => Op::LoadTrue { dst: dst + base },
        Op::LoadFalse { dst } => Op::LoadFalse { dst: dst + base },
        Op::LoadNull { dst } => Op::LoadNull { dst: dst + base },
        Op::Move { dst, src } => Op::Move { dst: dst + base, src: src + base },
        Op::Add { dst, lhs, rhs } => Op::Add { dst: dst + base, lhs: lhs + base, rhs: rhs + base },
        Op::Sub { dst, lhs, rhs } => Op::Sub { dst: dst + base, lhs: lhs + base, rhs: rhs + base },
        Op::Mul { dst, lhs, rhs } => Op::Mul { dst: dst + base, lhs: lhs + base, rhs: rhs + base },
        Op::Div { dst, lhs, rhs } => Op::Div { dst: dst + base, lhs: lhs + base, rhs: rhs + base },
        Op::Lt { dst, lhs, rhs } => Op::Lt { dst: dst + base, lhs: lhs + base, rhs: rhs + base },
        Op::Le { dst, lhs, rhs } => Op::Le { dst: dst + base, lhs: lhs + base, rhs: rhs + base },
        Op::Gt { dst, lhs, rhs } => Op::Gt { dst: dst + base, lhs: lhs + base, rhs: rhs + base },
        Op::Ge { dst, lhs, rhs } => Op::Ge { dst: dst + base, lhs: lhs + base, rhs: rhs + base },
        Op::Eq { dst, lhs, rhs } => Op::Eq { dst: dst + base, lhs: lhs + base, rhs: rhs + base },
        Op::Neq { dst, lhs, rhs } => Op::Neq { dst: dst + base, lhs: lhs + base, rhs: rhs + base },
        Op::LooseEq { dst, lhs, rhs } => Op::LooseEq { dst: dst + base, lhs: lhs + base, rhs: rhs + base },
        Op::LooseNeq { dst, lhs, rhs } => Op::LooseNeq { dst: dst + base, lhs: lhs + base, rhs: rhs + base },
        // Jump targets are remapped in the patch pass; pass through here.
        Op::Jmp { target } => Op::Jmp { target },
        Op::JmpIfFalse { cond, target } => Op::JmpIfFalse { cond: cond + base, target },
        // Ret is handled by the caller (store + jmp); any other op is rejected by
        // `callee_is_inlinable` before we get here, so this is unreachable for a
        // valid inline target — pass it through (the backend will then decline).
        other => other,
    }
}

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
    // P3 — try CROSS-FUNCTION INLINING FIRST. If the function has an inlinable
    // monomorphic `CallFn` to a small numeric callee, inline it and compile the
    // FUSED body (the call op disappears). On no-inline (no call, or an un-inlinable
    // call) fall through to the single-function P2 path. The inlined path is byte-
    // identical to the VM (the inlined-frame deopt resumes the original caller); the
    // A/B oracle + the deopt fuzzer prove it. Inlining is gated by `inline_enabled()`
    // (env `CV_T4_INLINE`, default ON when CV_T4 is on — it is the keystone of T4;
    // the design flag `CV_T4_INLINE` is honored as an explicit opt-OUT).
    if inline_enabled() {
        if let T4CompileStatus::Ready(jf) = try_compile_t4_inlined_status(module, fn_idx) {
            return T4CompileStatus::Ready(jf);
        }
    }
    let f = match module.fns.get(fn_idx) {
        Some(f) => f,
        None => return T4CompileStatus::Decline,
    };
    // REUSE T3's optimizer + the B3 safepoint UAF gate verbatim. A map whose
    // pointer roots aren't all bank-resident is declined (never installed). For
    // the numeric subset the map carries no pointer roots, so this always passes;
    // it becomes load-bearing when T4 widens to heap ops (P3 inlining).
    // P4 — REDUNDANCY / LOAD / CHECK ELIMINATION over the T4-specialized graph.
    //
    // CSE MUST run BEFORE T3's linear-scan register allocator (exactly V8 Maglev's
    // ordering: redundancy/check elimination over the SSA value graph precedes
    // register allocation). Regalloc aggressively REUSES registers and inserts
    // copies, which DESTROYS the value-availability CSE needs — a recomputed `r*r`
    // whose dominating result has already been overwritten into a reused register
    // can no longer be reused. So we run the pass on the PRE-regalloc body (the
    // ORIGINAL `f`), where value identity is intact; the redundant pure expressions
    // fold to copies (dropping the arithmetic AND its implicit operand checks), then
    // T3's own copy-prop + DCE clean up the introduced `Move`s and the now-dead
    // recomputations, and regalloc renumbers the smaller result. Register-preserving
    // is unnecessary here (T3 renumbers afterward anyway), but op-count preservation
    // keeps the body well-formed for T3.
    //
    // The resume module is the T3-OPTIMIZED body, so CSE is gated by
    // `OnlyNumericOperands`: a folded recomputation the VM would re-run as a `Move`
    // must not skip a `valueOf`/`toString` side effect, so its operands must be
    // proven numeric. Store-to-load forwarding (copy prop) is unconditionally safe.
    // The A/B oracle proves byte-identity; the unsafe-CSE mutation hook proves the
    // kill-on-clobber is load-bearing.
    let mut pre = f.clone();
    let redun =
        redundancy::redundancy_eliminate_fn(&mut pre, redundancy::Allow::OnlyNumericOperands);
    bump_redundancy(&redun);
    let (optimized, _stats, safepoints) = match crate::t3::optimize_with_safepoints(&pre) {
        Ok(x) => x,
        Err(_) => return T4CompileStatus::Decline,
    };

    // ── P5 AOT-PERSIST (single-function P2 path; gated CV_AOT_PERSIST, DEFAULT OFF).
    //    The OPTIMIZED module is the program identity here: codegen consumes it AND a
    //    deopt resumes the VM on it (the identity-map module). On a COLD REPEAT VISIT
    //    we re-install the persisted native code with ZERO codegen + ZERO warmup —
    //    the path PAST V8. For the single-fn path the "fused" and "original" key
    //    inputs are BOTH the optimized module (no inlining; resume is on it), and the
    //    reloaded blob is marked is_inlined=false so `run_t4_call` falls to the proven
    //    `run_t3_call` resume — identical to this fresh single-fn install below.
    let opt_key_module = Module { fns: vec![optimized.clone()], script_forinit_syncs: Vec::new() };
    if aot::aot_persist_enabled() {
        if let Some(jf) = aot::load_from_disk(&opt_key_module, &opt_key_module) {
            return T4CompileStatus::Ready(jf);
        }
    }

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

    // ── P5 AOT-PERSIST store (single-fn path; best-effort, gated CV_AOT_PERSIST).
    //    We reached here on an AOT miss (or persist off) and produced fresh
    //    relocation-free native code. Persist it keyed by the optimized module so the
    //    NEXT cold visit re-installs it with zero codegen. is_inlined=false → the
    //    reload uses the `run_t3_call` (non-inlined) resume, matching this install.
    //    GUARD: only persist when the safepoint map is EMPTY (the numeric subset
    //    carries NO heap-pointer roots, so it always is). The reload path does not
    //    carry a safepoint map, so persisting a function WITH live roots would drop
    //    its UAF rooting — decline AOT-store in that (currently-unreachable) case
    //    rather than ship a blob that loses a root. Never wrong: it just won't cache.
    if safepoints.is_empty() {
        aot::store_to_disk(&code, &deopt_sites, &opt_key_module, &opt_key_module, false);
    }

    // Pin heap mode OFF for the run (numeric store mode), matching the compile —
    // the T4 backend always uses `T2StoreMode::Numeric`, so the run-time bank must
    // be the numeric bank. (`run_t3_call` already forces heap off; this guard
    // covers the in-process install path symmetrically.)
    let _heap = crate::interp::T2HeapGuard::new(false);
    let native = match crate::jit::JitFunction::install(&code) {
        Ok(jf) => jf,
        Err(_) => return T4CompileStatus::Decline,
    };
    let opt_module = std::rc::Rc::new(Module { fns: vec![optimized], script_forinit_syncs: Vec::new() });
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

/// Whether P3 cross-function inlining is enabled. It is the KEYSTONE of T4, so it
/// is ON whenever T4 is on, UNLESS explicitly opted out with `CV_T4_INLINE=0` (the
/// design's named flag, kept as an escape hatch so the P2 single-function path can
/// be A/B'd against the P3 inlined path). A `ForcedTier::T4` oracle run leaves it
/// on so the inline path is exercised. There is no separate default-OFF gate: the
/// whole T4 tier is already behind `CV_T4`/`ForcedTier::T4` (default off), so
/// inlining only ever runs when T4 itself is engaged.
pub fn inline_enabled() -> bool {
    thread_local! {
        static ON: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
    }
    ON.with(|c| match c.get() {
        Some(v) => v,
        None => {
            let v = std::env::var("CV_T4_INLINE").as_deref() != Ok("0");
            c.set(Some(v));
            v
        }
    })
}

/// Honesty guard — number of T4 functions compiled WITH ≥1 inlined call site. Lets
/// the oracle/tests prove the inliner is NON-VACUOUS (a green inline-oracle that
/// never actually inlined would be a lie). Bumped only on a successful inlined
/// compile; the default build (T4 off) never touches it.
pub fn inline_compile_count() -> u64 {
    INLINE_COMPILE_COUNT.with(|c| c.get())
}
pub fn reset_inline_compile_count() {
    INLINE_COMPILE_COUNT.with(|c| c.set(0));
}
thread_local! {
    static INLINE_COMPILE_COUNT: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

/// P4 honesty guard — total number of redundancy-elimination rewrites (CSE folds +
/// copy forwards) applied across T4 compiles. Lets the oracle/tests prove the P4
/// pass is NON-VACUOUS (a green P4 oracle that never actually eliminated anything
/// would be a lie). Bumped by `try_compile_t4_status` / `try_compile_t4_inlined_status`;
/// the default build (T4 off) never touches it.
pub fn redundancy_rewrite_count() -> u64 {
    REDUNDANCY_REWRITE_COUNT.with(|c| c.get())
}
pub fn reset_redundancy_rewrite_count() {
    REDUNDANCY_REWRITE_COUNT.with(|c| c.set(0));
}
fn bump_redundancy(st: &redundancy::RedundancyStats) {
    let n = (st.cse_folded + st.copies_forwarded) as u64;
    if n > 0 {
        REDUNDANCY_REWRITE_COUNT.with(|c| c.set(c.get() + n));
    }
}
thread_local! {
    static REDUNDANCY_REWRITE_COUNT: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

/// P3 — compile `module.fns[fn_idx]` WITH cross-function inlining. Inlines the first
/// inlinable monomorphic `CallFn` (depth 1), compiles the FUSED body through the
/// representation-aware backend with a resume-pc map routing every guard's deopt to
/// the ORIGINAL caller op, and installs native code that resumes the VM on the
/// ORIGINAL (un-inlined) caller module on a deopt (the INLINE-DEOPT-TO-CALLER
/// design). Returns `Decline` when there is no inlinable call (the caller then runs
/// the single-function path) or any compile step declines — always correct.
///
/// NOTE on the optimizer: the inlined path compiles the FUSED bytecode DIRECTLY
/// (NOT through `t3::optimize` regalloc), because the T3 optimizer renumbers
/// registers — which would invalidate the resume-pc map AND the caller-bank-slot
/// identity the inline-deopt-to-caller reconstruction relies on. The big P3 win is
/// the eliminated call frame + the unboxed-f64 representation selection the T4
/// backend already does over the fused body; the T3 regalloc is a separate (smaller)
/// optimization deferred to keep the deopt reconstruction provably correct.
#[cfg(target_os = "windows")]
pub fn try_compile_t4_inlined_status(module: &Module, fn_idx: usize) -> T4CompileStatus {
    // The inlined-deopt resume runs `module.fns[0]` as the caller, so the caller
    // MUST be at index 0 (the dispatch path always passes 0); a caller elsewhere
    // can't preserve both the caller AND the callee indices in one resume module —
    // decline up front (the single-fn path runs it correctly, never a wrong resume).
    if fn_idx != 0 {
        return T4CompileStatus::Decline;
    }
    let mut result = match inline_first_call(module, fn_idx) {
        Some(r) => r,
        None => return T4CompileStatus::Decline, // nothing to inline → single-fn path.
    };
    // P4 — REDUNDANCY / LOAD / CHECK ELIMINATION over the FUSED body. Runs IN PLACE
    // and register-PRESERVING, so the `bc_pc_map` (fused op index → caller resume
    // op) and the per-op `DeoptSite.bc_pc` stay aligned by construction — the pass
    // never inserts/deletes/reorders ops, only rewrites an op (e.g. a redundant
    // `Mul dst x x` → `Move dst prev`) or an operand read in place. The inlined-
    // fused deopt resumes the VM on the PRISTINE ORIGINAL caller (`t4_deopt_module`,
    // which re-runs the un-inlined `f` — every side effect performed), so CSE here
    // is UNCONDITIONALLY safe (`Allow::Always`): folding a recomputation can never
    // drop a side effect the resuming VM would otherwise perform. This is the jit.js
    // win — `f(x)`'s repeated `x*x` (and `x*x*x` reusing `x*x`) fold to copies,
    // dropping both the arithmetic AND its implicit operand checks. The inlined-
    // frame deopt fuzzer + the A/B oracle prove byte-identity after the fold.
    let redun =
        redundancy::redundancy_eliminate_fn(&mut result.fused, redundancy::Allow::Always);
    bump_redundancy(&redun);
    debug_assert_eq!(
        result.fused.code.len(),
        result.bc_pc_map.len(),
        "P4 redundancy elim must preserve the op count (and thus the resume-pc map)"
    );
    // VERIFY every inlined-region resume target is a real op in the ORIGINAL caller
    // (the inlined-frame analogue of the SafepointMap UAF gate). A guard whose
    // mapped bc_pc is out of the caller's code range would resume at a garbage op;
    // reject the whole inlined compile (the caller falls to the single-fn path or a
    // lower tier — never a wrong resume).
    let caller = match module.fns.get(fn_idx) {
        Some(c) => c,
        None => return T4CompileStatus::Decline,
    };
    let caller_code_len = caller.code.len();
    if result.bc_pc_map.iter().any(|&pc| pc >= caller_code_len) {
        return T4CompileStatus::Decline;
    }

    // ── P5 AOT-PERSIST (★ the cold-repeat beat-Chrome lever; gated CV_AOT_PERSIST,
    //    DEFAULT OFF). At this point the FUSED body (after inlining + P4 redundancy)
    //    and the ORIGINAL caller module are FULLY DETERMINED — they are exactly what
    //    codegen would consume and what a deopt resumes on. So they are the program
    //    identity the persisted native code is keyed by. On a COLD REPEAT VISIT
    //    (fresh process, warm AOT store) we re-install the already-optimized native
    //    code with ZERO codegen + ZERO warmup — the path PAST V8, which re-JITs every
    //    cold load. A digest miss / corruption falls through to a fresh compile
    //    (below) — never wrong, just a recompile. The re-installed DeoptSite table
    //    re-checks every guard on the new load, so even an (astronomically unlikely)
    //    digest collision deopts to the VM, never produces a wrong value.
    //
    //    The fused module is wrapped as a single-fn `Module` exactly as the runtime
    //    carries it (`with_t3_module`); the original caller is the whole module
    //    (so the resume's `fns[0]` caller + every callee sibling is intact).
    let fused_for_key = Module { fns: vec![result.fused.clone()], script_forinit_syncs: Vec::new() };
    if aot::aot_persist_enabled() {
        if let Some(jf) = aot::load_from_disk(&fused_for_key, module) {
            // The persisted blob carries its own fused + original modules + the
            // DeoptSite table; `load_from_disk` already attached them. We still
            // bump the inline-compile honesty counter (an inlined function WAS
            // produced — it just came from the AOT store, not a fresh codegen).
            INLINE_COMPILE_COUNT.with(|c| c.set(c.get() + result.inlined_calls as u64));
            return T4CompileStatus::Ready(jf);
        }
    }

    // Compile the FUSED body with the resume-pc map (inlined-region guards → caller
    // Call op; caller-region ops → their own original index).
    let consts = result.fused.consts.clone();
    let bc_pc_map = result.bc_pc_map.clone();
    let compiled = crate::jit::compile_t4_unboxed_with_deopt_mapped(
        &result.fused.code,
        move |k| match consts.get(k as usize) {
            Some(crate::interp::Value::Number(n)) => Some(*n),
            _ => None,
        },
        Some(&bc_pc_map),
    );
    let (code, deopt_sites) = match compiled {
        Some(x) => x,
        None => return T4CompileStatus::Decline,
    };
    // Final structural gate: every emitted DeoptSite's resume bc_pc must be in the
    // ORIGINAL caller code range (it came from the map, but assert mechanically —
    // the UAF/garbage-resume catcher, the inlined-frame analogue of
    // verify_against_bank). A violation declines the install (never installs code
    // that could resume at a bogus op).
    if deopt_sites.iter().any(|s| s.bc_pc >= caller_code_len) {
        return T4CompileStatus::Decline;
    }

    // ── P5 AOT-PERSIST store (best-effort, gated CV_AOT_PERSIST). We reached here
    //    on an AOT MISS (or with persist off) and just produced fresh relocation-
    //    free native code + its DeoptSite table. Persist it keyed by the SAME
    //    (fused, original) program identity so the NEXT cold visit re-installs it
    //    with zero codegen. The numeric subset is relocation-free by construction,
    //    so the bytes are safe to re-run verbatim on the next load. No-op when
    //    persist is off / a module is non-serializable / the disk write fails.
    //    is_inlined=true → the reload attaches `t4_deopt_module` (INLINE-DEOPT-TO-
    //    CALLER resume), exactly as this fresh install does below.
    aot::store_to_disk(&code, &deopt_sites, &fused_for_key, module, true);

    let _heap = crate::interp::T2HeapGuard::new(false);
    let native = match crate::jit::JitFunction::install(&code) {
        Ok(jf) => jf,
        Err(_) => return T4CompileStatus::Decline,
    };
    // The fused module (codegen reference) — carried on `t3_module` for the bank
    // SIZE (run_t4_call sizes the bank from it). The ORIGINAL CALLER module is the
    // resume target on a deopt; it MUST contain the inlined CALLEE too, because an
    // inlined-region deopt resumes the caller at the `Call` op and the VM re-runs
    // the ORDINARY (non-inlined) call — which needs the callee `fns[fn_idx]` present.
    // For the dispatch path `fn_idx == 0` (the caller is `fns[0]`) and the callee
    // sibling lives in the same module, so cloning the WHOLE module preserves both
    // the caller (still `fns[0]`) and every callee index. (The fused module's bank
    // is what codegen targets; the resume's `fns[0]` is the caller.)
    let fused_module = std::rc::Rc::new(Module { fns: vec![result.fused], script_forinit_syncs: Vec::new() });
    // fn_idx == 0 is guaranteed above, so the whole-module clone keeps the caller at
    // fns[0] AND every callee sibling index intact for the re-run-the-call resume.
    let orig_caller_module = std::rc::Rc::new(module.clone());
    INLINE_COMPILE_COUNT.with(|c| c.set(c.get() + result.inlined_calls as u64));
    T4CompileStatus::Ready(
        native
            .with_deopt_sites(deopt_sites)
            .with_t3_module(fused_module)
            .with_t4_deopt_module(orig_caller_module),
    )
}

#[cfg(not(target_os = "windows"))]
pub fn try_compile_t4_inlined_status(_module: &Module, _fn_idx: usize) -> T4CompileStatus {
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
