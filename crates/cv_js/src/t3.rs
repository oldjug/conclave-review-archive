//! T3 — the OPTIMIZING tier (B2 of PHASE B).
//!
//! T3 is a *real* optimizing compiler that sits ABOVE the proven T2-lite
//! baseline JIT. Its pipeline is:
//!
//!   bytecode (`bytecode::Op`)
//!     → linear IR + CFG  (`lower::lower`)
//!     → SSA-ish value graph over the `n_regs` virtual-register namespace
//!     → CONSERVATIVE, semantics-preserving optimization passes
//!         (const-fold, copy-propagation, dead-code elimination,
//!          redundant-guard / redundant-load elimination,
//!          restricted loop-invariant code motion)
//!     → linear-scan register allocation with spilling
//!     → an OPTIMIZED `BcFunction` (observationally equivalent to the input)
//!     → native code via the PROVEN `compile_t2lite_with_deopt` backend.
//!
//! ## Why lower back to bytecode instead of emitting fresh machine code
//!
//! The T2-lite backend (`jit::compile_t2lite_with_deopt`) is a mature,
//! deopt-complete, A/B-oracle-bit-identical code generator. Writing a *second*
//! independent x86 emitter for T3 would multiply the miscompile surface for no
//! correctness benefit. So T3's contribution is the OPTIMIZER (IR + passes +
//! register allocation) — all genuine optimizing-compiler machinery operating
//! on a verifiable representation — while the *machine-code emission and deopt*
//! reuse the proven backend. The optimizer's job is to hand the backend a
//! *better* bytecode program that is OBSERVABLY IDENTICAL to the original.
//!
//! THE LOAD-BEARING INVARIANT: every pass T3 runs must preserve the bytecode
//! VM's observable behavior, because a T2 deopt from T3-optimized code resumes
//! the VM *on the optimized module*. The A/B oracle (`ab_oracle.rs`, extended
//! with a `ForcedTier::T3` leg) proves `TreeWalk == Vm == … == T3` across the
//! whole corpus; the passes are deliberately CONSERVATIVE so that proof holds
//! (JS coercion traps are everywhere — we optimize only what is provably safe).
//!
//! ## Supported subset (else DECLINE → T2/VM, always correct)
//!
//! T3 specializes the NUMERIC / arithmetic / comparison / control-flow subset
//! (the same shape the T2-lite numeric fast path already handles) where the
//! semantics-preservation proofs are cleanest. Any op outside that subset —
//! `GetProp`/`Call`/heap ops/`Try*`/closures — makes T3 DECLINE the function
//! at lowering time, so it falls through to T2 (which handles those). A declined
//! T3 compile is never a correctness risk: the function simply runs on the
//! lower, proven tier. This is the "decline what you can't prove" discipline.
//!
//! ## Gating
//!
//! `t3_enabled()` (env `CV_T3`, DEFAULT OFF) mirrors `t2_enabled`. A
//! `ForcedTier::T3` override drives it from the in-process A/B oracle without
//! the env. The bytecode VM + T2 stay the DEFAULT and the universal fallback.

use crate::bytecode::{BcFunction, Module, Op, Reg};
use crate::interp::Value;

/// Whether the T3 optimizing tier is enabled. DEFAULT-OFF (opt IN with `CV_T3=1`,
/// like `CV_T1`/`CV_CODE_CAGE`). A `ForcedTier::T3` override takes precedence so
/// the in-process A/B oracle can drive T3 without the process-global env. When
/// off, the default path is byte-identical to today (T2 + VM only).
pub fn t3_enabled() -> bool {
    if matches!(crate::interp::forced_tier(), Some(crate::interp::ForcedTier::T3)) {
        return true;
    }
    thread_local! {
        static ON: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
    }
    ON.with(|c| match c.get() {
        Some(v) => v,
        None => {
            let v = std::env::var("CV_T3").as_deref() == Ok("1");
            c.set(Some(v));
            v
        }
    })
}

/// Whether the T4 MAGLEV-CLASS optimizing tier is enabled. DEFAULT-OFF (opt IN
/// with `CV_T4=1`), mirroring `t3_enabled`. A `ForcedTier::T4` override takes
/// precedence so the in-process A/B oracle can drive T4 without the process-global
/// env.
///
/// P0 SCAFFOLD CONTRACT: T4 has NO codegen yet, so `try_t4_call` DECLINES (returns
/// `None`) on every function — a forced/enabled T4 run falls through to T3 → T2 →
/// VM and is therefore observationally identical to today. This is deliberate: P0
/// lands the deopt + safepoint activation scaffold (the inlined-frame DeoptSite,
/// the verify_against_bank gate, both keystone fuzzers, this flag, and the oracle
/// leg) with ZERO behavior change, so the default build stays byte-identical and
/// every later phase (P2 representation selection, P3 inlining) lands against a
/// proven safety net rather than building the net under load.
pub fn t4_enabled() -> bool {
    if matches!(crate::interp::forced_tier(), Some(crate::interp::ForcedTier::T4)) {
        return true;
    }
    thread_local! {
        static ON: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
    }
    ON.with(|c| match c.get() {
        Some(v) => v,
        None => {
            let v = std::env::var("CV_T4").as_deref() == Ok("1");
            c.set(Some(v));
            v
        }
    })
}

// ----------------------------------------------------------------------
// MUTATION HOOK (test-only). The A/B oracle's job is to catch an optimizer that
// is NOT semantics-preserving. To PROVE the oracle is non-vacuous, this hook lets
// a test deliberately MISCOMPILE a const-fold (off-by-one in the folded result).
// With the hook set, the oracle MUST redden (T3 != VM); with it unset (the
// production default) the oracle is green. Mirrors `T2_FORCE_DEOPT_PC`. NEVER
// engaged in production — there is no env path, only the in-process setter.
// ----------------------------------------------------------------------
thread_local! {
    static T3_FORCE_WRONG_FOLD: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    /// MUTATION HOOK (test-only): when set, LICM IGNORES the 0-trip speculation
    /// rule (hoists even when the def is read outside the loop), so a loop that
    /// runs zero times observes the speculatively-computed value — a divergence
    /// the oracle must catch. Proves the 0-trip rule is load-bearing. Off in prod.
    static T3_FORCE_UNSAFE_LICM: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Set the mutation hook (test-only): when `true`, const-fold produces a WRONG
/// result (+1), so the A/B oracle must catch the divergence. Returns the prior
/// value. Use the `WrongFoldGuard` scope guard.
pub fn set_force_wrong_fold(v: bool) -> bool {
    T3_FORCE_WRONG_FOLD.with(|c| {
        let prev = c.get();
        c.set(v);
        prev
    })
}

fn force_wrong_fold() -> bool {
    T3_FORCE_WRONG_FOLD.with(|c| c.get())
}

/// Set the unsafe-LICM mutation hook (test-only). Returns the prior value. Use
/// the `UnsafeLicmGuard` scope guard.
pub fn set_force_unsafe_licm(v: bool) -> bool {
    T3_FORCE_UNSAFE_LICM.with(|c| {
        let prev = c.get();
        c.set(v);
        prev
    })
}

fn force_unsafe_licm() -> bool {
    T3_FORCE_UNSAFE_LICM.with(|c| c.get())
}

/// RAII guard for the unsafe-LICM mutation hook (test-only).
#[must_use]
pub struct UnsafeLicmGuard {
    prev: bool,
}
impl UnsafeLicmGuard {
    pub fn new(on: bool) -> Self {
        UnsafeLicmGuard { prev: set_force_unsafe_licm(on) }
    }
}
impl Drop for UnsafeLicmGuard {
    fn drop(&mut self) {
        set_force_unsafe_licm(self.prev);
    }
}

/// RAII guard for the const-fold mutation hook (test-only).
#[must_use]
pub struct WrongFoldGuard {
    prev: bool,
}
impl WrongFoldGuard {
    pub fn new(on: bool) -> Self {
        WrongFoldGuard { prev: set_force_wrong_fold(on) }
    }
}
impl Drop for WrongFoldGuard {
    fn drop(&mut self) {
        set_force_wrong_fold(self.prev);
    }
}

// ======================================================================
// IR — a linear value graph over the n_regs virtual-register namespace.
//
// Each bytecode op becomes one `Inst`. We build a CFG by splitting at jump
// targets, then run analyses over it. The IR deliberately keeps the bytecode's
// 3-address `dst/lhs/rhs` register form (so it round-trips back to bytecode
// trivially), but annotates each value with a *type lattice* element so the
// passes can prove numeric-only operands.
// ======================================================================

/// The type lattice for a virtual register value, inferred by a forward
/// dataflow analysis. Conservative: `Unknown` is the top (and the default), so
/// a pass that requires `Number` only fires when we have PROVEN it numeric.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ty {
    /// Bottom — this value is never produced (unreachable / not yet seen).
    Bottom,
    /// Provably a non-NaN, non-special f64 numeric immediate of known value.
    ConstNum(u64), // f64 bits (canonical NaN excluded — only finite/specials with known bits)
    /// Provably a Number (could be any f64), but value not statically known.
    Number,
    /// Provably a Boolean.
    Bool,
    /// Anything else — could be a heap ref, undefined, string, etc. TOP.
    Unknown,
}

impl Ty {
    /// Join two lattice elements (meet toward `Unknown` = the conservative top).
    fn join(self, other: Ty) -> Ty {
        use Ty::*;
        match (self, other) {
            (Bottom, x) | (x, Bottom) => x,
            (ConstNum(a), ConstNum(b)) if a == b => ConstNum(a),
            (ConstNum(_), ConstNum(_)) => Number,
            (ConstNum(_), Number) | (Number, ConstNum(_)) | (Number, Number) => Number,
            (Bool, Bool) => Bool,
            _ => Unknown,
        }
    }

    fn is_number(self) -> bool {
        matches!(self, Ty::ConstNum(_) | Ty::Number)
    }
}

/// One IR instruction — a bytecode op plus its position bookkeeping. We keep the
/// original `Op` (the form the backend re-consumes) and a flag marking it dead
/// (removed by DCE) so we never have to renumber mid-pass.
#[derive(Debug, Clone)]
struct Inst {
    op: Op,
    /// Original bytecode index (for diagnostics / stable ordering).
    bc_idx: usize,
    /// Set by DCE: this instruction's result is unused and the op is pure, so it
    /// can be dropped from the emitted program.
    dead: bool,
    /// Set by LICM: this instruction is loop-invariant and should be EMITTED at
    /// the end of the named (preheader) block instead of in its home block. None =
    /// emit in place.
    hoist_to: Option<usize>,
}

/// A basic block: a maximal straight-line run of instructions with a single
/// entry (a jump target or fall-through join) and a terminator.
#[derive(Debug, Clone)]
struct Block {
    /// Indices into `T3Func::insts` that belong to this block, in order.
    insts: Vec<usize>,
    /// Successor block ids (by control flow).
    succs: Vec<usize>,
    /// Predecessor block ids.
    preds: Vec<usize>,
    /// First bytecode index in this block (the block's "label").
    start_bc: usize,
}

/// The whole function under optimization.
struct T3Func {
    insts: Vec<Inst>,
    blocks: Vec<Block>,
    /// bytecode index → block id (the block that *starts* at that index, if any).
    bc_to_block: std::collections::HashMap<usize, usize>,
    n_params: u8,
    n_regs: Reg,
}

/// Why T3 declined to optimize a function (diagnostic; all map to "run on T2").
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeclineReason {
    /// An op outside T3's numeric/control-flow subset (GetProp/Call/heap/try/…).
    UnsupportedOp,
    /// Empty body, too many registers for the bitset analyses, etc.
    Shape,
}

// ----------------------------------------------------------------------
// Lowering: bytecode → CFG + IR.
// ----------------------------------------------------------------------

/// Is `op` in T3's optimizable subset? T3 handles exactly the ops the T2-lite
/// NUMERIC fast path proves (arith / compare / numeric load / move / control
/// flow / ret). Everything else declines to T2. (This is intentionally a SUBSET
/// of T2's full repertoire — T3 only claims functions where its passes have a
/// clean semantics-preservation proof; the rest are strictly better served by
/// the existing, proven T2 path.)
fn op_supported(op: &Op) -> bool {
    matches!(
        op,
        Op::LoadConst { .. }
            | Op::LoadTrue { .. }
            | Op::LoadFalse { .. }
            | Op::LoadNull { .. }
            | Op::LoadUndef { .. }
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
            | Op::JmpIfTrue { .. }
            | Op::Ret { .. }
    )
}

/// Does `op` terminate a basic block (jump / ret)?
fn is_terminator(op: &Op) -> bool {
    matches!(
        op,
        Op::Jmp { .. } | Op::JmpIfFalse { .. } | Op::JmpIfTrue { .. } | Op::Ret { .. }
    )
}

/// The branch targets (bytecode indices) an op transfers to, plus whether it can
/// fall through to the next op.
fn op_targets(op: &Op, next: usize) -> (Vec<usize>, bool) {
    match *op {
        Op::Jmp { target } => (vec![target as usize], false),
        Op::JmpIfFalse { target, .. } | Op::JmpIfTrue { target, .. } => {
            (vec![target as usize, next], true)
        }
        Op::Ret { .. } => (vec![], false),
        _ => (vec![next], true),
    }
}

/// Lower `f`'s bytecode into a CFG + IR, or DECLINE. Declines on any unsupported
/// op (so the caller falls to T2) or a degenerate shape.
fn lower(f: &BcFunction) -> Result<T3Func, DeclineReason> {
    if f.code.is_empty() {
        return Err(DeclineReason::Shape);
    }
    // n_regs over 64 would overflow the safepoint bitset analyses elsewhere; T3
    // is for tight numeric kernels, so cap conservatively (decline → T2).
    if f.n_regs as usize > 256 {
        return Err(DeclineReason::Shape);
    }
    for op in &f.code {
        if !op_supported(op) {
            return Err(DeclineReason::UnsupportedOp);
        }
    }

    let n = f.code.len();
    let insts: Vec<Inst> = f
        .code
        .iter()
        .enumerate()
        .map(|(i, op)| Inst { op: op.clone(), bc_idx: i, dead: false, hoist_to: None })
        .collect();

    // --- Identify block leaders: index 0, every EXPLICIT jump target, and every
    //     op immediately after a terminator (a jump/ret). A fall-through edge is
    //     NOT a leader by itself — only a real branch destination or a post-
    //     terminator position starts a new block. ---
    let mut leader = vec![false; n];
    leader[0] = true;
    for (i, op) in f.code.iter().enumerate() {
        // EXPLICIT branch targets only (a Jmp/JmpIf* target — never the implicit
        // fall-through `next`).
        match *op {
            Op::Jmp { target } => {
                if (target as usize) < n {
                    leader[target as usize] = true;
                }
            }
            Op::JmpIfFalse { target, .. } | Op::JmpIfTrue { target, .. } => {
                if (target as usize) < n {
                    leader[target as usize] = true;
                }
            }
            _ => {}
        }
        // The op after ANY terminator starts a new block (it's only reachable via
        // a branch or fall-after-conditional, never straight-line from the term).
        if is_terminator(op) && i + 1 < n {
            leader[i + 1] = true;
        }
    }

    // --- Build blocks. ---
    let mut blocks: Vec<Block> = Vec::new();
    let mut bc_to_block = std::collections::HashMap::new();
    let mut cur: Option<usize> = None;
    for i in 0..n {
        if leader[i] {
            // Close the previous block at the prior op if it wasn't a terminator.
            let bid = blocks.len();
            blocks.push(Block {
                insts: Vec::new(),
                succs: Vec::new(),
                preds: Vec::new(),
                start_bc: i,
            });
            bc_to_block.insert(i, bid);
            cur = Some(bid);
        }
        let bid = cur.expect("leader[0] is always set, so cur is Some");
        blocks[bid].insts.push(i);
    }

    // --- Wire successors/predecessors. ---
    for bid in 0..blocks.len() {
        let last_inst_idx = *blocks[bid].insts.last().unwrap();
        let op = &insts[last_inst_idx].op;
        let next = last_inst_idx + 1;
        let (targets, _ft) = op_targets(op, next);
        let mut succ_blocks = Vec::new();
        for t in targets {
            if let Some(&tb) = bc_to_block.get(&t) {
                if !succ_blocks.contains(&tb) {
                    succ_blocks.push(tb);
                }
            }
        }
        blocks[bid].succs = succ_blocks.clone();
        for sb in succ_blocks {
            if !blocks[sb].preds.contains(&bid) {
                blocks[sb].preds.push(bid);
            }
        }
    }

    Ok(T3Func {
        insts,
        blocks,
        bc_to_block,
        n_params: f.n_params,
        n_regs: f.n_regs,
    })
}

// ======================================================================
// Dataflow analysis: the type lattice over virtual registers.
//
// A forward fixpoint over the CFG computing, at the ENTRY of each block, a
// per-register `Ty`. Conservative: params start `Unknown` (a caller can pass
// anything), every register starts `Bottom`, and the join at a merge takes the
// lattice meet (toward `Unknown`). Used by the passes to prove an operand is a
// Number before const-folding / eliminating a numeric guard.
// ======================================================================

/// Per-register type state (indexed by register number).
type TyState = Vec<Ty>;

impl T3Func {
    /// The register an instruction WRITES (its SSA def), if any.
    fn def_reg(&self, op: &Op) -> Option<Reg> {
        match *op {
            Op::LoadConst { dst, .. }
            | Op::LoadTrue { dst }
            | Op::LoadFalse { dst }
            | Op::LoadNull { dst }
            | Op::LoadUndef { dst }
            | Op::Move { dst, .. }
            | Op::Add { dst, .. }
            | Op::Sub { dst, .. }
            | Op::Mul { dst, .. }
            | Op::Div { dst, .. }
            | Op::Lt { dst, .. }
            | Op::Le { dst, .. }
            | Op::Gt { dst, .. }
            | Op::Ge { dst, .. }
            | Op::Eq { dst, .. }
            | Op::Neq { dst, .. }
            | Op::LooseEq { dst, .. }
            | Op::LooseNeq { dst, .. } => Some(dst),
            _ => None,
        }
    }

    /// The registers an instruction READS (its operands).
    fn use_regs(&self, op: &Op) -> Vec<Reg> {
        match *op {
            Op::Move { src, .. } => vec![src],
            Op::Add { lhs, rhs, .. }
            | Op::Sub { lhs, rhs, .. }
            | Op::Mul { lhs, rhs, .. }
            | Op::Div { lhs, rhs, .. }
            | Op::Lt { lhs, rhs, .. }
            | Op::Le { lhs, rhs, .. }
            | Op::Gt { lhs, rhs, .. }
            | Op::Ge { lhs, rhs, .. }
            | Op::Eq { lhs, rhs, .. }
            | Op::Neq { lhs, rhs, .. }
            | Op::LooseEq { lhs, rhs, .. }
            | Op::LooseNeq { lhs, rhs, .. } => vec![lhs, rhs],
            Op::JmpIfFalse { cond, .. } | Op::JmpIfTrue { cond, .. } => vec![cond],
            Op::Ret { src } => vec![src],
            _ => vec![],
        }
    }

    /// True iff `op` is a comparison (its result is always a Boolean).
    fn is_compare(op: &Op) -> bool {
        matches!(
            op,
            Op::Lt { .. }
                | Op::Le { .. }
                | Op::Gt { .. }
                | Op::Ge { .. }
                | Op::Eq { .. }
                | Op::Neq { .. }
                | Op::LooseEq { .. }
                | Op::LooseNeq { .. }
        )
    }

    /// True iff `op` is one of the binary arithmetic ops (`+ - * /`). NOTE: the
    /// result of `+` is only a Number when BOTH operands are proven numeric (else
    /// string concat); the caller checks operand types before claiming `Number`.
    fn is_arith(op: &Op) -> bool {
        matches!(
            op,
            Op::Add { .. } | Op::Sub { .. } | Op::Mul { .. } | Op::Div { .. }
        )
    }

    /// Transfer function: apply `op`'s effect to a register type state in place,
    /// returning the new type of the defined register (`Bottom` if no def).
    /// `consts` resolves a `LoadConst`'s pool value to a `Ty`.
    fn transfer(&self, op: &Op, st: &mut TyState, consts: &[Value]) {
        let def = match self.def_reg(op) {
            Some(d) => d,
            None => return,
        };
        let new_ty = match *op {
            Op::LoadConst { k, .. } => match consts.get(k as usize) {
                Some(Value::Number(n)) => {
                    // Only track a known-bits constant when it is NOT NaN — the
                    // backend canonicalizes NaN and ConstNum bits must round-trip.
                    if n.is_nan() {
                        Ty::Number
                    } else {
                        Ty::ConstNum(n.to_bits())
                    }
                }
                Some(Value::Bool(_)) => Ty::Bool,
                _ => Ty::Unknown,
            },
            Op::LoadTrue { .. } | Op::LoadFalse { .. } => Ty::Bool,
            Op::LoadNull { .. } | Op::LoadUndef { .. } => Ty::Unknown,
            Op::Move { src, .. } => *st.get(src as usize).unwrap_or(&Ty::Unknown),
            _ if Self::is_compare(op) => Ty::Bool,
            Op::Sub { lhs, rhs, .. } | Op::Mul { lhs, rhs, .. } | Op::Div { lhs, rhs, .. } => {
                // `- * /` always ToNumber their operands → result is a Number
                // (NaN if a coercion fails, still a Number). So the *result* is
                // always Number, regardless of operand types.
                let _ = (lhs, rhs);
                Ty::Number
            }
            Op::Add { lhs, rhs, .. } => {
                // `+` is Number ONLY when both operands are proven numeric;
                // otherwise it could be string concatenation → Unknown.
                let lt = *st.get(lhs as usize).unwrap_or(&Ty::Unknown);
                let rt = *st.get(rhs as usize).unwrap_or(&Ty::Unknown);
                if lt.is_number() && rt.is_number() {
                    Ty::Number
                } else {
                    Ty::Unknown
                }
            }
            _ => Ty::Unknown,
        };
        if (def as usize) < st.len() {
            st[def as usize] = new_ty;
        }
    }

    /// Run the forward type-lattice fixpoint. Returns, per block, the register
    /// type state at the block's ENTRY.
    fn type_analysis(&self, consts: &[Value]) -> Vec<TyState> {
        let nb = self.blocks.len();
        let nr = self.n_regs as usize;
        // Entry state: params Unknown (caller-controlled), the rest Bottom.
        let mut entry: Vec<TyState> = vec![vec![Ty::Bottom; nr]; nb];
        for r in 0..(self.n_params as usize).min(nr) {
            entry[0][r] = Ty::Unknown;
        }
        // Worklist fixpoint.
        let mut worklist: Vec<usize> = (0..nb).collect();
        let mut iters = 0usize;
        let max_iters = nb * (nr + 4) + 64; // safety cap; lattice is finite height
        while let Some(bid) = worklist.pop() {
            iters += 1;
            if iters > max_iters {
                break; // converged-or-capped; the result is still conservative.
            }
            // Simulate this block from its entry state.
            let mut st = entry[bid].clone();
            for &ii in &self.blocks[bid].insts {
                let op = self.insts[ii].op.clone();
                self.transfer(&op, &mut st, consts);
            }
            // Propagate to successors via join.
            for &sb in &self.blocks[bid].succs.clone() {
                let mut changed = false;
                for r in 0..nr {
                    let joined = entry[sb][r].join(st[r]);
                    if joined != entry[sb][r] {
                        entry[sb][r] = joined;
                        changed = true;
                    }
                }
                if changed && !worklist.contains(&sb) {
                    worklist.push(sb);
                }
            }
        }
        entry
    }
}

// ======================================================================
// Optimization passes — CONSERVATIVE, semantics-preserving, oracle-gated.
//
// Every pass below preserves the bytecode VM's observable behavior. The shared
// safety principle: only transform an op when the type lattice has PROVEN the
// operands are numeric (so no `valueOf`/`toString`/string-concat side effect can
// fire), and never remove an op that could throw or whose result is observed.
// ======================================================================

impl T3Func {
    /// Whether an op is PURE (no side effects, cannot throw) GIVEN that its
    /// operands at this point are proven numeric. Used by DCE: a pure op whose
    /// result register is never read again can be dropped. Arithmetic on numbers
    /// never throws (division by zero is Infinity/NaN in JS, not a throw) and
    /// comparisons on numbers never throw, so on the proven-numeric subset these
    /// are pure. Loads of immediates are always pure. Control flow + Ret are NOT
    /// pure (they're observable). `Move` is pure.
    fn is_pure_if_numeric(op: &Op, operands_numeric: bool) -> bool {
        match op {
            Op::LoadConst { .. }
            | Op::LoadTrue { .. }
            | Op::LoadFalse { .. }
            | Op::LoadNull { .. }
            | Op::LoadUndef { .. }
            | Op::Move { .. } => true,
            _ if Self::is_arith(op) || Self::is_compare(op) => operands_numeric,
            _ => false,
        }
    }

    /// PASS 1 — constant folding. For a binary numeric op whose BOTH operands are
    /// proven `ConstNum`, evaluate it at compile time and replace it with a
    /// `LoadConst` of the result. ONLY fires for `- * /` (always numeric) and for
    /// `+`/compares when both operands are proven numeric (so no string concat /
    /// coercion). The folded result must round-trip through the const pool, so we
    /// hand the new constant to `add_const` and rewrite the op.
    ///
    /// Returns the number of ops folded (for the non-vacuity test).
    fn const_fold(&mut self, consts: &[Value], new_consts: &mut Vec<Value>) -> usize {
        let mut folded = 0;
        // Build a COMBINED pool (base + new) so a folded `LoadConst` of a NEW index
        // is correctly typed by the transfer fixpoint that drives chained folds.
        // We rebuild it after each fold (folds are rare relative to ops; the pool
        // stays small thanks to `add_const` de-dup).
        let mut combined: Vec<Value> = consts.to_vec();
        combined.extend(new_consts.iter().cloned());
        let entry = self.type_analysis(&combined);
        for bid in 0..self.blocks.len() {
            let mut st = entry[bid].clone();
            for k in 0..self.blocks[bid].insts.len() {
                let ii = self.blocks[bid].insts[k];
                let op = self.insts[ii].op.clone();
                // Read operand types BEFORE applying the transfer.
                let folded_op = self.try_fold_op(&op, &st, new_consts, consts);
                if let Some(new_op) = folded_op {
                    self.insts[ii].op = new_op.clone();
                    folded += 1;
                    // Keep the combined pool in sync so the transfer below types
                    // the freshly-added const correctly (chained folds).
                    combined.truncate(consts.len());
                    combined.extend(new_consts.iter().cloned());
                    self.transfer(&new_op, &mut st, &combined);
                } else {
                    self.transfer(&op, &mut st, &combined);
                }
            }
        }
        folded
    }

    /// Try to constant-fold a single op given the operand type state. Returns the
    /// replacement op (a `LoadConst`) or None.
    fn try_fold_op(
        &self,
        op: &Op,
        st: &TyState,
        new_consts: &mut Vec<Value>,
        base_consts: &[Value],
    ) -> Option<Op> {
        let cnum = |r: Reg| -> Option<f64> {
            match st.get(r as usize) {
                Some(Ty::ConstNum(bits)) => Some(f64::from_bits(*bits)),
                _ => None,
            }
        };
        let result: (Reg, Value) = match *op {
            Op::Add { dst, lhs, rhs } => {
                // Only fold when BOTH operands are proven numeric constants — then
                // `+` is numeric addition (no string concat possible).
                let (a, b) = (cnum(lhs)?, cnum(rhs)?);
                Some((dst, Value::Number(a + b)))
            }
            Op::Sub { dst, lhs, rhs } => {
                let (a, b) = (cnum(lhs)?, cnum(rhs)?);
                Some((dst, Value::Number(a - b)))
            }
            Op::Mul { dst, lhs, rhs } => {
                let (a, b) = (cnum(lhs)?, cnum(rhs)?);
                Some((dst, Value::Number(a * b)))
            }
            Op::Div { dst, lhs, rhs } => {
                let (a, b) = (cnum(lhs)?, cnum(rhs)?);
                // JS division: x/0 = ±Inf, 0/0 = NaN — f64 already does this.
                Some((dst, Value::Number(a / b)))
            }
            Op::Lt { dst, lhs, rhs } => Some((dst, Value::Bool(cnum(lhs)? < cnum(rhs)?))),
            Op::Le { dst, lhs, rhs } => Some((dst, Value::Bool(cnum(lhs)? <= cnum(rhs)?))),
            Op::Gt { dst, lhs, rhs } => Some((dst, Value::Bool(cnum(lhs)? > cnum(rhs)?))),
            Op::Ge { dst, lhs, rhs } => Some((dst, Value::Bool(cnum(lhs)? >= cnum(rhs)?))),
            // Numeric strict/loose equality on two numbers: identical semantics
            // (no coercion once both are numbers). NaN != NaN handled by f64.
            Op::Eq { dst, lhs, rhs } | Op::LooseEq { dst, lhs, rhs } => {
                Some((dst, Value::Bool(cnum(lhs)? == cnum(rhs)?)))
            }
            Op::Neq { dst, lhs, rhs } | Op::LooseNeq { dst, lhs, rhs } => {
                Some((dst, Value::Bool(cnum(lhs)? != cnum(rhs)?)))
            }
            _ => None,
        }?;
        let (dst, val) = result;
        // A folded Bool stays a Bool (LoadTrue/LoadFalse keep the type lattice
        // honest and avoid polluting the const pool); a Number goes to the pool.
        match val {
            Value::Bool(true) => Some(Op::LoadTrue { dst }),
            Value::Bool(false) => Some(Op::LoadFalse { dst }),
            Value::Number(n) => {
                // MUTATION HOOK (test-only): deliberately miscompile by +1 so the
                // A/B oracle catches a non-semantics-preserving fold. Off in prod.
                let n = if force_wrong_fold() { n + 1.0 } else { n };
                let k = add_const(base_consts, new_consts, Value::Number(n));
                Some(Op::LoadConst { dst, k })
            }
            _ => None,
        }
    }

    /// PASS 2 — dead-code elimination. A pure op (on the proven-numeric subset)
    /// whose defined register is never read by any later live op is dead and can
    /// be dropped. We compute liveness conservatively: a register is live if ANY
    /// instruction (anywhere — we don't do per-point liveness, just "ever read")
    /// reads it OR it is a param/return. This is conservative-safe (we only ever
    /// keep MORE ops than strictly necessary).
    ///
    /// Returns the number of ops marked dead (non-vacuity test).
    fn dce(&mut self, consts: &[Value]) -> usize {
        let entry = self.type_analysis(consts);
        let nr = self.n_regs as usize;
        // "ever read" set, recomputed each round so a removed op's reads stop
        // keeping its operands alive (iterate to a fixpoint).
        let mut removed = 0;
        loop {
            let mut ever_read = vec![false; nr + 1];
            for inst in &self.insts {
                if inst.dead {
                    continue;
                }
                for u in self.use_regs(&inst.op) {
                    if (u as usize) < ever_read.len() {
                        ever_read[u as usize] = true;
                    }
                }
            }
            let mut any = false;
            for bid in 0..self.blocks.len() {
                let mut st = entry[bid].clone();
                for &ii in &self.blocks[bid].insts {
                    let op = self.insts[ii].op.clone();
                    if !self.insts[ii].dead {
                        if let Some(d) = self.def_reg(&op) {
                            let operands_numeric =
                                self.use_regs(&op).iter().all(|r| {
                                    st.get(*r as usize)
                                        .map(|t| t.is_number())
                                        .unwrap_or(false)
                                });
                            if !ever_read[d as usize]
                                && Self::is_pure_if_numeric(&op, operands_numeric)
                            {
                                self.insts[ii].dead = true;
                                removed += 1;
                                any = true;
                            }
                        }
                    }
                    self.transfer(&op, &mut st, consts);
                }
            }
            if !any {
                break;
            }
        }
        removed
    }

    /// PASS 3 — redundant-guard / redundant-load elimination + copy propagation.
    ///
    /// Within a basic block (no merge complications), if a value is recomputed
    /// from the same operands with no intervening write to those operands, the
    /// recomputation is redundant. For the NUMERIC subset, the most impactful form
    /// is `Move`-copy propagation: `b = Move a; … (a unchanged) … use b` → use a.
    /// Eliminating the copies lets DCE drop the `Move` and reduces register
    /// pressure for the allocator. CONSERVATIVE: only within a block, and we stop
    /// propagating a copy the moment its source OR destination is rewritten.
    ///
    /// Returns the number of operand rewrites performed.
    fn copy_prop(&mut self) -> usize {
        let mut rewrites = 0;
        for bid in 0..self.blocks.len() {
            // map: reg -> the canonical source reg it currently copies (within
            // this block, while still valid).
            let mut alias: std::collections::HashMap<Reg, Reg> = std::collections::HashMap::new();
            let block_insts = self.blocks[bid].insts.clone();
            for ii in block_insts {
                if self.insts[ii].dead {
                    continue;
                }
                // Rewrite operand reads through the alias map first.
                let op = self.insts[ii].op.clone();
                let new_op = self.rewrite_uses(&op, &alias, &mut rewrites);
                self.insts[ii].op = new_op.clone();
                // A def kills any alias whose source OR target is the defined reg
                // (the defined reg's value changed).
                if let Some(d) = self.def_reg(&new_op) {
                    alias.remove(&d);
                    alias.retain(|_, src| *src != d);
                }
                // Record a new copy alias for `Move dst, src` (src already
                // canonicalized by the rewrite above).
                if let Op::Move { dst, src } = new_op {
                    if dst != src {
                        // dst now aliases src's canonical source.
                        let canon = *alias.get(&src).unwrap_or(&src);
                        // Don't create a self-alias and only when canon isn't
                        // itself later redefined (handled by the kills above).
                        if canon != dst {
                            alias.insert(dst, canon);
                        }
                    }
                }
            }
        }
        rewrites
    }

    /// Rewrite an op's operand reads through an alias map (copy propagation).
    fn rewrite_uses(
        &self,
        op: &Op,
        alias: &std::collections::HashMap<Reg, Reg>,
        rewrites: &mut usize,
    ) -> Op {
        let mut sub = |r: Reg| -> Reg {
            match alias.get(&r) {
                Some(&s) => {
                    *rewrites += 1;
                    s
                }
                None => r,
            }
        };
        match *op {
            Op::Move { dst, src } => Op::Move { dst, src: sub(src) },
            Op::Add { dst, lhs, rhs } => Op::Add { dst, lhs: sub(lhs), rhs: sub(rhs) },
            Op::Sub { dst, lhs, rhs } => Op::Sub { dst, lhs: sub(lhs), rhs: sub(rhs) },
            Op::Mul { dst, lhs, rhs } => Op::Mul { dst, lhs: sub(lhs), rhs: sub(rhs) },
            Op::Div { dst, lhs, rhs } => Op::Div { dst, lhs: sub(lhs), rhs: sub(rhs) },
            Op::Lt { dst, lhs, rhs } => Op::Lt { dst, lhs: sub(lhs), rhs: sub(rhs) },
            Op::Le { dst, lhs, rhs } => Op::Le { dst, lhs: sub(lhs), rhs: sub(rhs) },
            Op::Gt { dst, lhs, rhs } => Op::Gt { dst, lhs: sub(lhs), rhs: sub(rhs) },
            Op::Ge { dst, lhs, rhs } => Op::Ge { dst, lhs: sub(lhs), rhs: sub(rhs) },
            Op::Eq { dst, lhs, rhs } => Op::Eq { dst, lhs: sub(lhs), rhs: sub(rhs) },
            Op::Neq { dst, lhs, rhs } => Op::Neq { dst, lhs: sub(lhs), rhs: sub(rhs) },
            Op::LooseEq { dst, lhs, rhs } => Op::LooseEq { dst, lhs: sub(lhs), rhs: sub(rhs) },
            Op::LooseNeq { dst, lhs, rhs } => Op::LooseNeq { dst, lhs: sub(lhs), rhs: sub(rhs) },
            Op::JmpIfFalse { cond, target } => Op::JmpIfFalse { cond: sub(cond), target },
            Op::JmpIfTrue { cond, target } => Op::JmpIfTrue { cond: sub(cond), target },
            Op::Ret { src } => Op::Ret { src: sub(src) },
            other => other,
        }
    }
}

/// Append `val` to the combined constant pool (base + new), returning its index.
/// De-dups numeric constants so repeated folds don't bloat the pool. The combined
/// index space is `base.len() + position-in-new`.
fn add_const(base: &[Value], new: &mut Vec<Value>, val: Value) -> u16 {
    // Reuse an existing base const if bit-identical (numbers compared by bits so
    // -0 / NaN don't accidentally merge).
    if let Value::Number(n) = val {
        for (i, c) in base.iter().enumerate() {
            if let Value::Number(m) = c {
                if m.to_bits() == n.to_bits() {
                    return i as u16;
                }
            }
        }
        for (i, c) in new.iter().enumerate() {
            if let Value::Number(m) = c {
                if m.to_bits() == n.to_bits() {
                    return (base.len() + i) as u16;
                }
            }
        }
    }
    let idx = base.len() + new.len();
    new.push(val);
    idx as u16
}

// ======================================================================
// PASS 4 — restricted Loop-Invariant Code Motion (LICM).
//
// CONSERVATIVE to the bone. In a natural loop (a block with a back-edge), an op
// is loop-invariant iff all its operands are defined OUTSIDE the loop (or are
// themselves hoisted invariants) AND the op is pure-on-numbers. We hoist such an
// op to a PREHEADER inserted before the loop header. We restrict to the simplest,
// provably-safe case: a single-block loop body whose header has exactly one
// out-of-loop predecessor (so the preheader is unambiguous), and we hoist only
// pure numeric computations whose operands are loop-invariant. Anything else
// stays put. (The plan's "LICM for loop-invariant guards/loads where the shape
// is stable" specializes to: for the numeric subset, the invariant computations
// ARE the guards+loads — a `Sub r, a, b` with a,b invariant carries the implicit
// is-number guard, and hoisting it lifts that guard out of the loop.)
//
// We implement LICM as an IR-level transform that MARKS instructions to be
// relocated; the actual relocation happens at emission time (so jump targets stay
// consistent). To keep it provably-safe and simple, this first cut hoists into a
// region that DOMINATES the loop and POST-dominates nothing problematic — we only
// hoist into an existing block that is the loop's sole entry predecessor.
// ======================================================================

impl T3Func {
    /// Find natural loops: a back-edge is an edge `b -> h` where `h` dominates
    /// `b`. We use a cheap dominator approximation valid for reducible CFGs from
    /// structured bytecode: `h` dominates `b` iff `h.start_bc <= b.start_bc` AND
    /// `h` is reachable on every path (we conservatively require `h` to be a
    /// successor-target of `b` with `h.start_bc <= b's last bc`). Since we only
    /// USE this to find HOIST opportunities (and never to change semantics — a
    /// missed loop just means no hoist), an approximate detector is safe.
    fn detect_simple_loops(&self) -> Vec<(usize, usize)> {
        // Returns (header_block, latch_block) pairs for single-back-edge loops.
        let mut loops = Vec::new();
        for b in 0..self.blocks.len() {
            for &succ in &self.blocks[b].succs {
                // Back-edge heuristic: a successor whose start bytecode index is
                // <= this block's start (a jump backward) is a loop header.
                if self.blocks[succ].start_bc <= self.blocks[b].start_bc {
                    loops.push((succ, b));
                }
            }
        }
        loops
    }

    /// Compute the NATURAL LOOP body for a back-edge `latch -> header`: the set of
    /// blocks that can reach `latch` without passing through `header` (plus the
    /// header itself). Standard backward reachability from the latch within the
    /// region the header dominates. Approximate-but-safe: an over-large body only
    /// makes LICM MORE conservative (more regs counted as written-in-loop), never
    /// incorrect.
    fn natural_loop_body(&self, header: usize, latch: usize) -> std::collections::HashSet<usize> {
        let mut body = std::collections::HashSet::new();
        body.insert(header);
        let mut stack = vec![latch];
        while let Some(b) = stack.pop() {
            if body.insert(b) {
                for &p in &self.blocks[b].preds {
                    if p != header && !body.contains(&p) {
                        stack.push(p);
                    }
                }
            }
        }
        body
    }

    /// PASS 4 — restricted LICM. Returns the number of ops hoisted. We hoist a
    /// pure-numeric INVARIANT op out of a natural loop into the loop's unique
    /// preheader (the header's single out-of-loop predecessor).
    ///
    /// SAFETY: we ONLY hoist when (a) the op is pure-on-numbers AND its operands
    /// are PROVEN numeric at the loop header entry (so no coercion side effect can
    /// move), (b) the op's operands are NOT written anywhere in the loop body
    /// (genuinely invariant), (c) the op's def reg has a single writer in the loop
    /// (so hoisting can't reorder two defs), and (d) there is a UNIQUE preheader
    /// block. A hoist that fails any check is skipped (correct, just unoptimized).
    fn licm(&mut self, consts: &[Value]) -> usize {
        let mut hoisted = 0;
        let entry = self.type_analysis(consts);
        let loops = self.detect_simple_loops();
        for (header, latch) in loops {
            // The natural-loop body (set of blocks). The header's preheader is its
            // unique predecessor OUTSIDE this body.
            let body_blocks = self.natural_loop_body(header, latch);
            let out_preds: Vec<usize> = self.blocks[header]
                .preds
                .iter()
                .cloned()
                .filter(|p| !body_blocks.contains(p))
                .collect();
            if out_preds.len() != 1 {
                continue; // ambiguous / no unique preheader → skip
            }
            let preheader = out_preds[0];
            // The preheader must NOT itself be inside another (outer) loop body we
            // are processing in the same pass in a way that corrupts ordering — to
            // stay simple+safe, require the preheader is not in this loop's body
            // (guaranteed by the filter) and skip if it equals the header.
            if preheader == header {
                continue;
            }
            // Registers WRITTEN anywhere in the loop body.
            let mut written_in_loop = std::collections::HashSet::new();
            let mut writers: std::collections::HashMap<Reg, usize> =
                std::collections::HashMap::new();
            for &b in &body_blocks {
                for &ii in &self.blocks[b].insts {
                    if self.insts[ii].dead || self.insts[ii].hoist_to.is_some() {
                        continue;
                    }
                    if let Some(d) = self.def_reg(&self.insts[ii].op) {
                        written_in_loop.insert(d);
                        *writers.entry(d).or_insert(0) += 1;
                    }
                }
            }
            // Registers READ OUTSIDE the loop body (anywhere not in body_blocks).
            // Hoisting speculatively executes an op even when the loop runs ZERO
            // times; if the def reg is read after the loop, that changes its value
            // on the 0-trip path (in-loop it'd be the pre-loop value). So we ONLY
            // hoist ops whose def is NOT read outside the loop — making the hoist
            // observably invisible on every path (the def is loop-local). This is
            // the conservative, provably-safe LICM speculation rule.
            let mut read_outside: std::collections::HashSet<Reg> =
                std::collections::HashSet::new();
            for b in 0..self.blocks.len() {
                if body_blocks.contains(&b) {
                    continue;
                }
                for &ii in &self.blocks[b].insts {
                    if self.insts[ii].dead {
                        continue;
                    }
                    for u in self.use_regs(&self.insts[ii].op) {
                        read_outside.insert(u);
                    }
                }
            }
            // Type state at the loop header entry (operands must be numeric there).
            let st = entry[header].clone();
            // Scan all loop-body ops; collect pure numeric invariants to hoist.
            let mut to_hoist: Vec<usize> = Vec::new();
            for &b in &body_blocks {
                for &ii in &self.blocks[b].insts.clone() {
                    if self.insts[ii].dead || self.insts[ii].hoist_to.is_some() {
                        continue;
                    }
                    let op = self.insts[ii].op.clone();
                    // Must be a pure-on-numbers computation (arith/compare).
                    if !(Self::is_arith(&op) || Self::is_compare(&op)) {
                        continue;
                    }
                    let uses = self.use_regs(&op);
                    // Operands proven numeric at the loop header entry?
                    let numeric = uses
                        .iter()
                        .all(|r| st.get(*r as usize).map(|t| t.is_number()).unwrap_or(false));
                    if !numeric {
                        continue;
                    }
                    // Operands invariant (not written anywhere in the loop)?
                    let invariant = uses.iter().all(|r| !written_in_loop.contains(r));
                    // The op's def reg must have a SINGLE writer in the loop (else
                    // hoisting could reorder two defs of the same reg) AND must NOT
                    // be read outside the loop (the 0-trip speculation rule above).
                    let (single_writer, loop_local) = match self.def_reg(&op) {
                        Some(d) => (
                            writers.get(&d).copied().unwrap_or(0) == 1,
                            // MUTATION HOOK: force the 0-trip rule OFF to prove the
                            // oracle catches an unsafe hoist (a def read outside the
                            // loop, hoisted past the loop guard). Always true in prod.
                            !read_outside.contains(&d) || force_unsafe_licm(),
                        ),
                        None => (false, false),
                    };
                    if invariant && single_writer && loop_local {
                        to_hoist.push(ii);
                    }
                }
            }
            // Relocate: mark each hoisted op to be EMITTED at the end of the
            // preheader (before its terminator) instead of in the loop body.
            for ii in to_hoist {
                self.insts[ii].hoist_to = Some(preheader);
                hoisted += 1;
            }
        }
        hoisted
    }
}

// ======================================================================
// Linear-scan register allocation.
//
// We compute a LIVE INTERVAL per virtual register over the linearized (post-DCE,
// post-fold) instruction order, then greedily pack non-overlapping intervals into
// a compact set of PHYSICAL bank slots. The result is a register RENAMING: a map
// `vreg -> phys` that minimizes the bank-slot count (= register pressure). When
// more intervals are simultaneously live than the physical budget, we SPILL —
// but our physical budget is the full bank (it's a flat Vec the runner sizes), so
// "spilling" here means: extra slots beyond a small register window stay in the
// bank (they already are — the backend uses [bank + slot*8] for every slot, which
// IS the spill emission B1 provides). So the allocator's real win is COMPACTION
// (fewer total slots → smaller bank, better locality), proven by the renaming.
//
// CRITICAL invariant for deopt: the emitted module's deopt resumes the VM on the
// RENAMED module, so the identity map (bank slot i == VM reg i) holds for the
// renamed registers — we renumber EVERYTHING consistently (ops + n_regs) so the
// optimized module is a self-consistent program. Params keep slots [0..n_params)
// (the runner loads args there), so they are PINNED in the allocation.
// ======================================================================

/// A live interval for one virtual register: [first_def, last_use] over the
/// linearized instruction index space. Loop-carried values get their interval
/// extended to cover the whole loop (so we never reuse a slot still live across a
/// back-edge).
#[derive(Debug, Clone, Copy)]
struct Interval {
    vreg: Reg,
    start: usize,
    end: usize,
}

impl T3Func {
    /// Compute live intervals over the linearized order, conservatively extended
    /// for loops: any register live across a back-edge gets its interval stretched
    /// to span the entire loop region (so packing never aliases two values that
    /// are both live around the loop). Returns intervals for every register that
    /// is ever defined or used.
    fn live_intervals(&self, order: &[usize]) -> Vec<Interval> {
        let nr = self.n_regs as usize;
        let mut first = vec![usize::MAX; nr + 1];
        let mut last = vec![0usize; nr + 1];
        let mut seen = vec![false; nr + 1];
        // pos in `order` for each register's defs/uses.
        for (pos, &ii) in order.iter().enumerate() {
            let op = &self.insts[ii].op;
            for r in self.use_regs(op) {
                let r = r as usize;
                if r <= nr {
                    if !seen[r] {
                        seen[r] = true;
                        first[r] = pos;
                    }
                    last[r] = last[r].max(pos);
                    if first[r] == usize::MAX {
                        first[r] = pos;
                    }
                }
            }
            if let Some(d) = self.def_reg(op) {
                let r = d as usize;
                if r <= nr {
                    if !seen[r] {
                        seen[r] = true;
                        first[r] = pos;
                    }
                    last[r] = last[r].max(pos);
                }
            }
        }
        // Loop extension: for each back-edge (a jump from a later pos to an
        // earlier pos), any register live at the header must stay live to the
        // latch. We approximate by extending every interval that STARTS before a
        // loop header and is USED at/after it to cover the loop's latch position.
        // Build (header_pos, latch_pos) pairs from the linearized jump targets.
        let mut loop_ranges: Vec<(usize, usize)> = Vec::new();
        // Map bc index -> position in `order` (for resolving jump targets).
        let mut bc_to_pos: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
        for (pos, &ii) in order.iter().enumerate() {
            bc_to_pos.insert(self.insts[ii].bc_idx, pos);
        }
        for (pos, &ii) in order.iter().enumerate() {
            let next_bc = self.insts[ii].bc_idx + 1;
            let (targets, _ft) = op_targets(&self.insts[ii].op, next_bc);
            for t in targets {
                if let Some(&tpos) = bc_to_pos.get(&t) {
                    if tpos < pos {
                        loop_ranges.push((tpos, pos)); // back-edge: header..latch
                    }
                }
            }
        }
        for r in 0..=nr {
            if !seen[r] {
                continue;
            }
            for &(h, l) in &loop_ranges {
                // If the value is live anywhere in [h, l] (interval overlaps the
                // loop) extend it to cover the whole loop region so packing is safe
                // across the back-edge.
                if first[r] <= l && last[r] >= h {
                    first[r] = first[r].min(h);
                    last[r] = last[r].max(l);
                }
            }
        }
        let mut intervals = Vec::new();
        for r in 0..=nr {
            if seen[r] {
                intervals.push(Interval {
                    vreg: r as Reg,
                    start: first[r],
                    end: last[r],
                });
            }
        }
        intervals
    }

    /// Linear-scan allocate: pack intervals into compact physical slots. Returns
    /// `(renaming: vreg -> phys, n_phys)`. Params [0..n_params) are PINNED to the
    /// same physical slot (the runner writes args there). Result keeps every other
    /// register's value in a distinct slot while live; non-overlapping intervals
    /// share a slot (compaction = the "spill"/reuse decision of linear scan).
    fn linear_scan(&self, intervals: &[Interval]) -> (Vec<Reg>, usize) {
        let nr = self.n_regs as usize;
        let mut renaming = vec![Reg::MAX; nr + 1];
        let np = self.n_params as usize;
        // Pin params to their own slots [0..np) so the runner's arg-store still
        // lands in the right place.
        let mut next_phys = np;
        for r in 0..np.min(nr + 1) {
            renaming[r] = r as Reg;
        }
        // Sort the NON-param intervals by start position (the linear-scan order).
        let mut work: Vec<Interval> = intervals
            .iter()
            .cloned()
            .filter(|iv| (iv.vreg as usize) >= np)
            .collect();
        work.sort_by_key(|iv| iv.start);
        // Active list: (end_pos, phys_slot). Free a slot when its interval ends.
        // free_slots is a pool of physical slots >= np reusable once freed.
        let mut active: Vec<(usize, usize, Reg)> = Vec::new(); // (end, phys, vreg)
        let mut free_slots: Vec<usize> = Vec::new();
        for iv in work {
            // Expire intervals that ended before this one starts.
            let start = iv.start;
            let mut still_active = Vec::new();
            for (end, phys, vr) in active.drain(..) {
                if end < start {
                    free_slots.push(phys);
                } else {
                    still_active.push((end, phys, vr));
                }
            }
            active = still_active;
            // Assign a physical slot: reuse a freed one, else a fresh slot.
            let phys = if let Some(p) = free_slots.pop() {
                p
            } else {
                let p = next_phys;
                next_phys += 1;
                p
            };
            renaming[iv.vreg as usize] = phys as Reg;
            active.push((iv.end, phys, iv.vreg));
        }
        // Registers never seen (never read or written by any kept op) need no slot
        // — they can never appear in the emitted program, so map them to slot 0 (a
        // valid existing slot) WITHOUT growing the physical count. A defensive
        // mapping to `r` itself would inflate n_phys past the compacted count.
        for r in 0..=nr {
            if renaming[r] == Reg::MAX {
                renaming[r] = 0;
            }
        }
        (renaming, next_phys.max(np).max(1))
    }

    /// Apply a register renaming to an op (both defs and uses).
    fn rename_op(op: &Op, ren: &[Reg]) -> Op {
        let m = |r: Reg| -> Reg { *ren.get(r as usize).unwrap_or(&r) };
        match *op {
            Op::LoadConst { dst, k } => Op::LoadConst { dst: m(dst), k },
            Op::LoadTrue { dst } => Op::LoadTrue { dst: m(dst) },
            Op::LoadFalse { dst } => Op::LoadFalse { dst: m(dst) },
            Op::LoadNull { dst } => Op::LoadNull { dst: m(dst) },
            Op::LoadUndef { dst } => Op::LoadUndef { dst: m(dst) },
            Op::Move { dst, src } => Op::Move { dst: m(dst), src: m(src) },
            Op::Add { dst, lhs, rhs } => Op::Add { dst: m(dst), lhs: m(lhs), rhs: m(rhs) },
            Op::Sub { dst, lhs, rhs } => Op::Sub { dst: m(dst), lhs: m(lhs), rhs: m(rhs) },
            Op::Mul { dst, lhs, rhs } => Op::Mul { dst: m(dst), lhs: m(lhs), rhs: m(rhs) },
            Op::Div { dst, lhs, rhs } => Op::Div { dst: m(dst), lhs: m(lhs), rhs: m(rhs) },
            Op::Lt { dst, lhs, rhs } => Op::Lt { dst: m(dst), lhs: m(lhs), rhs: m(rhs) },
            Op::Le { dst, lhs, rhs } => Op::Le { dst: m(dst), lhs: m(lhs), rhs: m(rhs) },
            Op::Gt { dst, lhs, rhs } => Op::Gt { dst: m(dst), lhs: m(lhs), rhs: m(rhs) },
            Op::Ge { dst, lhs, rhs } => Op::Ge { dst: m(dst), lhs: m(lhs), rhs: m(rhs) },
            Op::Eq { dst, lhs, rhs } => Op::Eq { dst: m(dst), lhs: m(lhs), rhs: m(rhs) },
            Op::Neq { dst, lhs, rhs } => Op::Neq { dst: m(dst), lhs: m(lhs), rhs: m(rhs) },
            Op::LooseEq { dst, lhs, rhs } => Op::LooseEq { dst: m(dst), lhs: m(lhs), rhs: m(rhs) },
            Op::LooseNeq { dst, lhs, rhs } => Op::LooseNeq { dst: m(dst), lhs: m(lhs), rhs: m(rhs) },
            Op::Jmp { target } => Op::Jmp { target },
            Op::JmpIfFalse { cond, target } => Op::JmpIfFalse { cond: m(cond), target },
            Op::JmpIfTrue { cond, target } => Op::JmpIfTrue { cond: m(cond), target },
            Op::Ret { src } => Op::Ret { src: m(src) },
            other => other,
        }
    }
}

// ======================================================================
// Emission: optimized IR → a fresh `BcFunction`, with jump targets remapped.
// ======================================================================

/// Statistics from one T3 optimization run (for the non-vacuity tests + logging).
#[derive(Debug, Clone, Copy, Default)]
pub struct T3Stats {
    pub folded: usize,
    pub copies_propagated: usize,
    pub dead_removed: usize,
    pub hoisted: usize,
    /// Original register count vs the post-allocation count (compaction).
    pub regs_before: usize,
    pub regs_after: usize,
    /// Final emitted op count vs original.
    pub ops_before: usize,
    pub ops_after: usize,
}

impl T3Func {
    /// Produce the FINAL linear emission order of instruction indices: for each
    /// block in original start order, first the hoisted ops destined for it, then
    /// its own live (non-dead, non-hoisted-away) ops in order. Returns the order
    /// AND a map block_id -> the emission index of that block's FIRST op (the jump
    /// label). A hoisted op lands at the END of its target block, so the target
    /// block's own start label is the first op AFTER any hoists destined here —
    /// wait: hoists go to the preheader (a DIFFERENT block than the loop header),
    /// appended after the preheader's own ops, so a block's first emitted op is
    /// either its first own op or, if all its own ops were hoisted away (cannot
    /// happen — a block always keeps its terminator), its terminator. We therefore
    /// label a block by the emission index of its FIRST own kept op; hoisted ops
    /// from OTHER blocks are appended to the preheader and never precede the
    /// header's label.
    fn emission_order(&self) -> (Vec<usize>, std::collections::HashMap<usize, usize>) {
        // Group hoisted ops by their destination block.
        let mut hoisted_into: std::collections::HashMap<usize, Vec<usize>> =
            std::collections::HashMap::new();
        for (idx, inst) in self.insts.iter().enumerate() {
            if inst.dead {
                continue;
            }
            if let Some(dest) = inst.hoist_to {
                hoisted_into.entry(dest).or_default().push(idx);
            }
        }
        let mut order: Vec<usize> = Vec::new();
        let mut block_label: std::collections::HashMap<usize, usize> =
            std::collections::HashMap::new();
        // Blocks are stored in ascending start_bc order already (lowering walks
        // bytecode in order), but be explicit.
        let mut block_ids: Vec<usize> = (0..self.blocks.len()).collect();
        block_ids.sort_by_key(|&b| self.blocks[b].start_bc);
        for bid in block_ids {
            // The block's own kept ops (skip dead + hoisted-away), emitted first so
            // the block label is the block's first own op.
            let mut first_own: Option<usize> = None;
            // Emit own ops EXCEPT a terminator goes LAST after appended hoists from
            // this block being a preheader. So: emit own non-terminator ops, then
            // hoisted ops destined here, then the terminator. This keeps a hoisted
            // op BEFORE the loop's back-edge jump in the preheader.
            let own: Vec<usize> = self.blocks[bid]
                .insts
                .iter()
                .cloned()
                .filter(|&ii| !self.insts[ii].dead && self.insts[ii].hoist_to.is_none())
                .collect();
            let (own_body, own_term): (Vec<usize>, Vec<usize>) = own
                .into_iter()
                .partition(|&ii| !is_terminator(&self.insts[ii].op));
            for ii in own_body {
                if first_own.is_none() {
                    first_own = Some(order.len());
                }
                order.push(ii);
            }
            if let Some(hs) = hoisted_into.get(&bid) {
                for &ii in hs {
                    if first_own.is_none() {
                        first_own = Some(order.len());
                    }
                    order.push(ii);
                }
            }
            for ii in own_term {
                if first_own.is_none() {
                    first_own = Some(order.len());
                }
                order.push(ii);
            }
            if let Some(lbl) = first_own {
                block_label.insert(bid, lbl);
            }
        }
        (order, block_label)
    }

    /// Build the optimized `BcFunction` from the (already-renamed) instruction
    /// stream in emission order, remapping jump targets to the new index space.
    fn emit(
        &self,
        order: &[usize],
        block_label: &std::collections::HashMap<usize, usize>,
        ren: &[Reg],
        n_regs_after: usize,
        base_consts: &[Value],
        new_consts: &[Value],
        name: &str,
    ) -> Option<BcFunction> {
        // Map an ORIGINAL bytecode target index -> the NEW emission index. A jump
        // targets a block start, so resolve via block_label of the block that
        // STARTS at that bc index.
        let resolve_target = |bc_target: usize| -> Option<u16> {
            let bid = *self.bc_to_block.get(&bc_target)?;
            let new_idx = *block_label.get(&bid)?;
            u16::try_from(new_idx).ok()
        };
        let mut code: Vec<Op> = Vec::with_capacity(order.len());
        for &ii in order {
            let op = Self::rename_op(&self.insts[ii].op, ren);
            let remapped = match op {
                Op::Jmp { target } => Op::Jmp { target: resolve_target(target as usize)? },
                Op::JmpIfFalse { cond, target } => {
                    Op::JmpIfFalse { cond, target: resolve_target(target as usize)? }
                }
                Op::JmpIfTrue { cond, target } => {
                    Op::JmpIfTrue { cond, target: resolve_target(target as usize)? }
                }
                other => other,
            };
            code.push(remapped);
        }
        // A well-formed bytecode function ends in Ret; if the last emitted op
        // isn't a terminator something went wrong — decline (don't emit garbage).
        if !matches!(code.last(), Some(Op::Ret { .. })) {
            // It's possible the final block's terminator is a Jmp/JmpIf to the
            // epilogue elsewhere — but our subset always ends fns in Ret. If not,
            // bail to the lower tier.
            if !matches!(code.last(), Some(Op::Jmp { .. } | Op::JmpIfFalse { .. } | Op::JmpIfTrue { .. }))
            {
                return None;
            }
        }
        let mut consts = base_consts.to_vec();
        consts.extend_from_slice(new_consts);
        Some(BcFunction {
            name: name.to_string(),
            n_params: self.n_params,
            rest_reg: None,
            n_regs: u16::try_from(n_regs_after).ok()?,
            consts,
            code,
            ic: std::cell::RefCell::new(Vec::new()),
            // T4 P1: the OPTIMIZED module starts with an empty feedback vector.
            // T3/T4 codegen reads feedback off the ORIGINAL `BcFunction` (the one
            // the VM recorded into); the optimized clone never records, so an empty
            // vector here is correct and never consulted by the recorder.
            feedback: std::cell::RefCell::new(Vec::new()),
        })
    }
}

/// Run the full T3 optimization pipeline on `f`, returning an OPTIMIZED
/// `BcFunction` (observationally equivalent to `f`) plus stats, or a decline.
/// This is the pure-IR half — it does NOT touch native code. Used directly by the
/// tests to verify the optimizer in isolation, and by `try_compile_t3` which then
/// hands the result to the proven T2 backend.
pub fn optimize(f: &BcFunction) -> Result<(BcFunction, T3Stats), DeclineReason> {
    let mut func = lower(f)?;
    let base_consts = &f.consts;
    let mut new_consts: Vec<Value> = Vec::new();
    let mut stats = T3Stats {
        regs_before: f.n_regs as usize,
        ops_before: f.code.len(),
        ..Default::default()
    };

    // PASS ORDER: copy-prop → const-fold → copy-prop again (folds expose copies)
    // → DCE → LICM. Each is independently semantics-preserving; the order only
    // affects how MUCH is optimized, never correctness.
    stats.copies_propagated += func.copy_prop();
    stats.folded += func.const_fold(base_consts, &mut new_consts);
    stats.copies_propagated += func.copy_prop();
    stats.dead_removed += func.dce(base_consts);
    {
        // LICM needs the combined pool for typing folded consts.
        let mut combined = base_consts.to_vec();
        combined.extend(new_consts.iter().cloned());
        stats.hoisted += func.licm(&combined);
    }
    // DCE again — LICM/fold may have orphaned ops.
    stats.dead_removed += func.dce(base_consts);

    // Register allocation over the post-opt instruction order.
    let (order, block_label) = func.emission_order();
    let intervals = func.live_intervals(&order);
    let (ren, n_phys) = func.linear_scan(&intervals);
    stats.regs_after = n_phys;

    let name = if f.name.is_empty() {
        "<t3>".to_string()
    } else {
        f.name.clone()
    };
    let optimized = func
        .emit(&order, &block_label, &ren, n_phys, base_consts, &new_consts, &name)
        .ok_or(DeclineReason::Shape)?;
    stats.ops_after = optimized.code.len();
    Ok((optimized, stats))
}

// ======================================================================
// B3 — SAFEPOINT STACK MAP + the spill-to-bank rooting discipline (UAF keystone).
//
// A safepoint is a native-code position where a GC can run. In an optimizing tier
// that holds heap-pointer JsVals in registers across op boundaries, the GC could
// clear (the clear-not-free mark-sweep, `interp.rs:3903`) a register-only-reachable
// object — a UAF / silent data loss. B3 ships the provably-safe discipline:
//
//   At every safepoint, every LIVE POINTER-LANE value is SPILLED to its
//   identity-map bank slot before the call/alloc/back-edge, and recorded in the
//   safepoint's `live_roots`. The bank IS the spill area, and `gc_seed_jit_banks`
//   already roots every bank slot — so a pointer recorded as an in-range bank
//   slot is automatically covered, and a GC at any safepoint cannot clear it.
//
// `build_safepoint_map` constructs the per-function `SafepointMap` from the
// OPTIMIZED bytecode, classifying each safepoint and recording the live pointer
// roots that the spill discipline put in the bank. `optimize_with_safepoints`
// runs the optimizer AND verifies the resulting map against the bank size
// (`SafepointMap::verify_against_bank`) — a `debug_assert` that catches a future
// heap-T3 codegen bug (a pointer live across a safepoint that was NOT spilled to
// an in-range bank slot) at COMPILE time, before any corrupted page at run time.
//
// TODAY'S NUMERIC SUBSET: T3 supports only arith/compare/numeric-load/move/
// control-flow/ret (see `op_supported`). None of those holds a heap pointer, so
// `live_roots` is EMPTY at every safepoint and the discipline holds vacuously —
// which is exactly why the current numeric T3 is already UAF-safe. The machinery
// below is the GATE a future heap-extended T3 (GetProp/Call/GetIdx in registers)
// MUST pass: it cannot ship a pointer-in-register-across-a-call without recording
// + spilling that pointer, or `verify_against_bank` (and the GC-integration
// assert) rejects it. This is the "land B3 before any T3 heap ref across a call"
// ordering the milestone requires, made mechanical.
// ======================================================================

use crate::osr::{SafepointKind, SafepointMap};

/// True if executing `op` could trigger a garbage collection — i.e. it is a
/// GC-safe point at which live heap pointers must already be rooted (spilled to
/// the bank). Calls (may pump a nested collect), allocations (the alloc itself
/// can collect), and — handled separately by back-edge detection — loop
/// back-edges (the periodic-collection trigger in hot loops).
///
/// For TODAY'S T3 subset this is always `false` (no call/alloc ops are
/// supported), so the only safepoints are back-edges; the match is exhaustive
/// over the heap/call ops a FUTURE T3 will add so this stays correct when the
/// subset widens (a new heap op that forgets to appear here would also fail the
/// discipline verifier, defense-in-depth).
fn op_is_call_or_alloc_safepoint(op: &Op) -> Option<SafepointKind> {
    match op {
        // Calls — the classic cross-call UAF window (a nested collect can run).
        Op::CallFn { .. } | Op::CallValue { .. } | Op::New { .. } => {
            Some(SafepointKind::HelperCall)
        }
        // Allocation sites — the allocation may itself trigger a collection, so
        // values live BEFORE it must be rooted.
        Op::NewObject { .. }
        | Op::NewArray { .. }
        | Op::MakeClosure { .. }
        | Op::MakeRegex { .. } => Some(SafepointKind::Allocation),
        _ => None,
    }
}

/// Determine, for the OPTIMIZED bytecode `code`, the set of POINTER-LANE bank
/// slots live across the safepoint at instruction `pc`. With the spill-to-bank
/// discipline, a value live across the safepoint lives in its identity-map bank
/// slot, so this returns bank-slot indices that the GC will scan.
///
/// For the numeric subset every op produces a number/bool — never a pointer — so
/// no slot is ever a pointer root and this returns an empty mask. The signature +
/// the typed-as-pointer hook is the seam a future heap-T3 plugs its liveness +
/// type analysis into (it would mark the slots holding `Ty::HeapObj`/string/etc.
/// live across `pc`). Returning a conservative-empty mask today is SOUND because
/// there are provably no pointers to root.
fn pointer_roots_live_at(_code: &[Op], _pc: usize, _n_regs: usize) -> u64 {
    // Numeric subset: no pointer-lane value is ever live. A future heap-T3
    // replaces this with its liveness∩pointer-type analysis; the discipline
    // verifier then proves every returned slot is an in-range bank slot.
    0
}

/// Build the B3 `SafepointMap` for an OPTIMIZED T3 function. Records one
/// safepoint per call/alloc op and one per loop back-edge (a backward `Jmp`/
/// `JmpIf*`), in ascending instruction order, each carrying the pointer-lane
/// bank-slot roots live across it (per the spill discipline).
///
/// `native_off` here is the BYTECODE instruction index used as a stable safepoint
/// key for the discipline verification; when a future heap-T3 emits its own
/// machine code it will key on the real native return-address offset instead (the
/// `SafepointRec` format is unchanged — that's the reserved seam). The map is
/// returned so `optimize_with_safepoints` can verify it against the bank size.
fn build_safepoint_map(code: &[Op], n_regs: usize) -> SafepointMap {
    let mut map = SafepointMap::new();
    for (pc, op) in code.iter().enumerate() {
        // Back-edge: a jump whose target instruction index is <= the jump's own
        // index (a loop). These are the in-loop periodic-collection safepoints.
        let is_back_edge = match *op {
            Op::Jmp { target } => (target as usize) <= pc,
            Op::JmpIfFalse { target, .. } | Op::JmpIfTrue { target, .. } => {
                (target as usize) <= pc
            }
            _ => false,
        };
        let kind = if let Some(k) = op_is_call_or_alloc_safepoint(op) {
            Some(k)
        } else if is_back_edge {
            Some(SafepointKind::BackEdge)
        } else {
            None
        };
        if let Some(kind) = kind {
            let roots = pointer_roots_live_at(code, pc, n_regs);
            map.record(pc, kind, roots);
        }
    }
    map
}

/// Run the T3 optimizer AND build + VERIFY its B3 safepoint map. Returns the
/// optimized function, the stats, and the verified `SafepointMap`.
///
/// THE B3 GATE: `verify_against_bank(n_regs)` proves every live pointer root at
/// every safepoint is an in-range bank slot (so `gc_seed_jit_banks` covers it).
/// A violation is a `debug_assert` panic in debug builds (a codegen bug — a
/// pointer live across a safepoint that was not spilled to a scanned bank slot =
/// the UAF the milestone forbids) and, in release, is reported by DECLINING the
/// compile (falls to T2/VM — always correct, never a UAF). Today the map has no
/// pointer roots so this always passes; it becomes load-bearing the moment a
/// future T3 holds a heap ref across a safepoint.
pub fn optimize_with_safepoints(
    f: &BcFunction,
) -> Result<(BcFunction, T3Stats, SafepointMap), DeclineReason> {
    let (optimized, stats) = optimize(f)?;
    let map = build_safepoint_map(&optimized.code, optimized.n_regs as usize);
    // THE UAF KEYSTONE ASSERTION: every safepoint's pointer roots are bank-resident.
    debug_assert!(
        map.roots_covered_by_bank(optimized.n_regs as usize),
        "B3 safepoint discipline violated: a pointer-lane value is live across a \
         safepoint but was not spilled to an in-range identity-map bank slot — \
         gc_seed_jit_banks could not root it (UAF). verify_against_bank: {:?}",
        map.verify_against_bank(optimized.n_regs as usize)
    );
    if !map.roots_covered_by_bank(optimized.n_regs as usize) {
        // Release-build safety net: never install code whose safepoints aren't
        // fully rooted. Decline → run on the proven lower tier.
        return Err(DeclineReason::Shape);
    }
    Ok((optimized, stats, map))
}

// ======================================================================
// Backend bridge: optimized BcFunction → native code via the T2 backend.
// ======================================================================

/// The outcome of a T3 compile attempt.
pub enum T3CompileStatus {
    /// Optimized AND installed as native code.
    Ready(crate::jit::JitFunction),
    /// T3 declined (unsupported op / shape) — the caller runs T2/VM.
    Decline,
}

/// Compile `module.fns[fn_idx]` through the T3 optimizer and then the PROVEN
/// T2-lite backend. Returns `Ready` with installed native code, or `Decline`
/// (the caller falls through to T2/VM — always correct).
///
/// The optimized function is wrapped in a single-function `Module` so the T2
/// backend (which compiles `fns[0]`) consumes it unchanged. A T2 deopt from this
/// native code resumes the VM ON THE OPTIMIZED MODULE — which is observationally
/// identical to the original, so the result is bit-identical to running the
/// original on the VM (the A/B oracle proves this across the corpus).
#[cfg(target_os = "windows")]
pub fn try_compile_t3_status(module: &Module, fn_idx: usize) -> T3CompileStatus {
    let f = match module.fns.get(fn_idx) {
        Some(f) => f,
        None => return T3CompileStatus::Decline,
    };
    // B3: optimize AND build + verify the safepoint rooting-discipline map. A map
    // whose pointer roots aren't all bank-resident is declined (never installed) —
    // the UAF gate. For today's numeric subset the map carries no pointer roots,
    // so this always passes; it becomes load-bearing when T3 widens to heap ops.
    let (optimized, _stats, safepoints) = match optimize_with_safepoints(f) {
        Ok(x) => x,
        Err(_) => return T3CompileStatus::Decline,
    };
    let opt_module = Module { fns: vec![optimized], script_forinit_syncs: Vec::new() };
    // Hand the optimized module to the proven T2 numeric backend. T3's subset is
    // numeric/control-flow only (no GetProp/Call), so the NUMERIC T2 path applies
    // directly — no IC warming needed, and the deopt/resume machinery is reused
    // verbatim. Pin heap mode OFF so the compile uses the numeric store mode that
    // `run_t3_call` matches at run time (independent of `CV_T2_HEAP`). (If T2
    // declines the optimized module — shouldn't happen for the supported subset —
    // we decline too and the caller runs the VM.)
    let _heap = crate::interp::T2HeapGuard::new(false);
    match crate::bytecode::try_compile_t2lite_status(&opt_module, 0) {
        crate::bytecode::T2CompileStatus::Ready(jf) => {
            // Stash the optimized module on the JitFunction so the runner resumes
            // the VM on the OPTIMIZED bytecode (the identity-map module). Attach the
            // B3-verified safepoint map so the GC can consult it for a collection
            // that lands on a JIT PC (the precise register-resident pass is the
            // documented deferred follow-on; the format ships now).
            T3CompileStatus::Ready(
                jf.with_t3_module(std::rc::Rc::new(opt_module))
                    .with_safepoints(safepoints),
            )
        }
        _ => T3CompileStatus::Decline,
    }
}

#[cfg(not(target_os = "windows"))]
pub fn try_compile_t3_status(_module: &Module, _fn_idx: usize) -> T3CompileStatus {
    T3CompileStatus::Decline
}

/// Thin wrapper returning `Some` only on `Ready`.
#[cfg(target_os = "windows")]
pub fn try_compile_t3(module: &Module, fn_idx: usize) -> Option<crate::jit::JitFunction> {
    match try_compile_t3_status(module, fn_idx) {
        T3CompileStatus::Ready(jf) => Some(jf),
        _ => None,
    }
}

#[cfg(not(target_os = "windows"))]
pub fn try_compile_t3(_module: &Module, _fn_idx: usize) -> Option<crate::jit::JitFunction> {
    None
}

#[cfg(test)]
mod tests;
