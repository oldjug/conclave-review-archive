//! T4 (Maglev-class) PHASE P4 — REDUNDANCY ELIMINATION + LOAD ELIMINATION +
//! CHECK ELIMINATION over the T4-specialized (numeric-subset) graph.
//!
//! ## What this phase models (V8 / Maglev source)
//!
//! V8's optimizing tiers run, over their SSA value graph, three closely-related
//! cleanups that this phase brings to the T4 numeric graph:
//!
//!   * **Redundancy elimination / GVN-CSE on the pure value sub-graph** —
//!     `src/compiler/common-operator-reducer` + the Maglev
//!     `MaglevGraphBuilder` "known node" deduplication: an identical PURE
//!     computation (`x*x`, `x+1.0`, …) recomputed with the SAME inputs and no
//!     intervening redefinition of those inputs reuses the first node's result
//!     instead of re-emitting the operation.
//!   * **Check elimination** — `src/compiler/redundancy-elimination` (the
//!     `RedundancyElimination` reducer) + Maglev's `known_node_aspects`
//!     CheckMaps/CheckNumber dedup: a type check that is DOMINATED by an
//!     equivalent check with no clobbering store in between is redundant. In the
//!     T4 Float64 backend every arithmetic op carries an implicit `CheckNumber`
//!     on its operands (the is-number guard that emits a `DeoptSite`); folding a
//!     redundant `x*x` into a copy of the dominating `x*x` ALSO removes that
//!     op's implicit operand checks — exactly Maglev's check elimination.
//!   * **Store-to-load forwarding / copy propagation** —
//!     `src/compiler/load-elimination` (the `LoadElimination` reducer): a value
//!     written into a slot and later read back, with no intervening write, reads
//!     the known value directly. For the register-file numeric subset this is
//!     `Move dst, src` alias forwarding: a later read of `dst` becomes a read of
//!     `src` while the copy is still valid.
//!
//! ## Why a BYTECODE→BYTECODE, register-PRESERVING transform
//!
//! The T4 backend (`jit::compile_t4_unboxed_with_deopt_mapped`) consumes
//! bytecode and emits per-op, keying every guard's `DeoptSite.bc_pc` by the op
//! INDEX. The P3 inliner additionally carries a `bc_pc_map` (fused op index →
//! original-caller resume op). To stay compatible with BOTH the single-function
//! path AND the inlined-fused path WITHOUT invalidating the resume-pc map, this
//! pass:
//!   * NEVER renumbers registers (so the identity-map bank invariant — bank slot
//!     i == VM reg i — and the inlined-frame deopt reconstruction are untouched);
//!   * NEVER reorders, inserts, or deletes ops — it only REWRITES an op IN PLACE
//!     (a redundant arith op `Op::Mul dst a b` → `Op::Move dst prev`, or an
//!     operand read rewritten through a copy alias). The op COUNT and every op's
//!     INDEX are preserved, so `bc_pc_map[i]` and the per-op `DeoptSite.bc_pc`
//!     stay valid by construction.
//!
//! ## Why it is byte-identical to the VM (the non-negotiable gate)
//!
//! Every rewritten op STILL writes its `dst` bank slot with the SAME value the
//! original op would produce:
//!   * **Copy propagation** rewrites only operand READS through a `Move` alias
//!     (`b = Move a; … use b → use a`) while `a` and `b` are both unchanged. The
//!     value is bit-identical, so the result is identical regardless of operand
//!     types — and a `Move` has no side effect to skip. This is sound on BOTH
//!     resume paths (it never changes which ops run, only which register an
//!     equal-valued read names).
//!   * **CSE of a pure arithmetic op** replaces `Op::Mul dst a b` (op B) with
//!     `Op::Move dst prev` where `prev` holds the result of a DOMINATING
//!     `Op::Mul prev a b` (op A) earlier in the SAME basic block with NO write to
//!     `a`, `b`, or `prev` in between. At runtime, when op B's slot is reached:
//!       - native (no deopt): op A already computed `a OP b` into `prev`; the
//!         `Move` copies the identical value into `dst`. Byte-identical.
//!       - deopt: the ONLY observable difference a CSE could introduce is SKIPPING
//!         a coercion side effect (e.g. a second `valueOf` if an operand were an
//!         object). CSE is therefore restricted to the case where SKIPPING op B's
//!         recomputation cannot drop a side effect the resuming VM would perform:
//!           (a) the INLINED-FUSED path resumes the VM on the PRISTINE original
//!               caller (`run_t4_call` → `t4_deopt_module`), re-running the
//!               ORIGINAL non-CSE'd `f` — every side effect is performed — so CSE
//!               is UNCONDITIONALLY safe there (`Allow::Always`);
//!           (b) the SINGLE-FUNCTION path resumes on the OPTIMIZED (CSE'd) module,
//!               so CSE there is allowed ONLY when op B's operands are PROVEN
//!               primitive/number (no `valueOf`/`toString` to skip) by the local
//!               numeric analysis (`Allow::OnlyNumericOperands`).
//!
//! The A/B oracle (`ForcedTier::T4`) proves `T4 == VM == tree-walk` across the
//! corpus after this pass runs, and the mutation hook
//! (`set_force_unsafe_cse`) — which deliberately forwards an expression ACROSS a
//! redefinition of one of its operands — MUST redden the oracle, proving the
//! kill-on-clobber logic is load-bearing (non-vacuity).

use crate::bytecode::{BcFunction, Op, Reg};
use crate::interp::Value;

// ----------------------------------------------------------------------
// MUTATION HOOK (test-only). Proves the oracle / tests are non-vacuous: with the
// hook set, CSE IGNORES the operand-clobber kill (it forwards an available
// expression even after one of its operands was redefined), producing a WRONG
// result the oracle must catch. Mirrors `t3::set_force_wrong_fold`. NEVER engaged
// in production — there is no env path, only the in-process setter.
// ----------------------------------------------------------------------
thread_local! {
    static FORCE_UNSAFE_CSE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Set the unsafe-CSE mutation hook; returns the previous value (test-only).
pub fn set_force_unsafe_cse(on: bool) -> bool {
    FORCE_UNSAFE_CSE.with(|c| {
        let p = c.get();
        c.set(on);
        p
    })
}
fn force_unsafe_cse() -> bool {
    FORCE_UNSAFE_CSE.with(|c| c.get())
}

/// RAII guard for the unsafe-CSE mutation hook (test-only).
#[must_use]
pub struct UnsafeCseGuard {
    prev: bool,
}
impl UnsafeCseGuard {
    pub fn new(on: bool) -> Self {
        UnsafeCseGuard { prev: set_force_unsafe_cse(on) }
    }
}
impl Drop for UnsafeCseGuard {
    fn drop(&mut self) {
        set_force_unsafe_cse(self.prev);
    }
}

/// How aggressively CSE may eliminate a recomputation, set by the RESUME path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Allow {
    /// The deopt resumes the VM on the PRISTINE original (the inlined-fused path):
    /// a deopt re-runs every original op, so eliminating a recomputation can never
    /// drop a side effect. CSE any pure arith with unchanged operands.
    Always,
    /// The deopt resumes the VM on the OPTIMIZED (rewritten) module (the
    /// single-function path): a CSE'd op B that the VM would re-run as a `Move`
    /// must not skip a coercion side effect, so CSE only when op B's operands are
    /// PROVEN numeric (no `valueOf`/`toString`).
    OnlyNumericOperands,
}

/// The stats a redundancy run produced (for the non-vacuity / engagement guards).
#[derive(Debug, Default, Clone, Copy)]
pub struct RedundancyStats {
    /// Number of pure arithmetic ops folded to a copy of a dominating equal op.
    pub cse_folded: usize,
    /// Number of operand reads rewritten through a copy alias (store-to-load fwd).
    pub copies_forwarded: usize,
}

impl RedundancyStats {
    /// Whether the pass changed anything (drives the honesty / engagement guards).
    pub fn is_nonvacuous(&self) -> bool {
        self.cse_folded + self.copies_forwarded > 0
    }
}

/// The def register an op writes (its SSA def), if any. Mirrors the T4 numeric
/// subset; an op not in the subset returns `None` and is treated as opaque (kills
/// nothing it doesn't have to). Comparisons DO define a register (a Bool).
fn def_reg(op: &Op) -> Option<Reg> {
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

/// A value-numbering KEY for a PURE op — `(tag, lhs, rhs)`. Only the ops whose
/// result is a deterministic function of their (named) operand REGISTERS with no
/// side effect on the numeric path are keyed; everything else returns `None` and
/// is never CSE'd. The tag distinguishes the operation; `lhs`/`rhs` are the
/// operand registers. Note `Add`/`Mul` are NOT treated as commutative (a
/// conservative choice — `a+b` and `b+a` get distinct keys; correct, just a
/// missed fold), keeping the key a literal mirror of the op.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct VnKey {
    tag: u8,
    lhs: Reg,
    rhs: Reg,
}

/// The opcode tag for the value-numbering key, or `None` if `op` is not a CSE-able
/// pure binary op. Strict and loose equality have IDENTICAL numeric semantics on
/// numbers, but they DIFFER on non-numbers; since a CSE under `OnlyNumericOperands`
/// proves operands numeric and a CSE under `Always` re-runs the pristine op on
/// deopt, we still give each opcode its OWN tag (never merge `Eq` with `LooseEq`)
/// — the most conservative, obviously-correct keying.
fn vn_key(op: &Op) -> Option<VnKey> {
    let (tag, lhs, rhs) = match *op {
        Op::Add { lhs, rhs, .. } => (0u8, lhs, rhs),
        Op::Sub { lhs, rhs, .. } => (1, lhs, rhs),
        Op::Mul { lhs, rhs, .. } => (2, lhs, rhs),
        Op::Div { lhs, rhs, .. } => (3, lhs, rhs),
        Op::Lt { lhs, rhs, .. } => (4, lhs, rhs),
        Op::Le { lhs, rhs, .. } => (5, lhs, rhs),
        Op::Gt { lhs, rhs, .. } => (6, lhs, rhs),
        Op::Ge { lhs, rhs, .. } => (7, lhs, rhs),
        Op::Eq { lhs, rhs, .. } => (8, lhs, rhs),
        Op::Neq { lhs, rhs, .. } => (9, lhs, rhs),
        Op::LooseEq { lhs, rhs, .. } => (10, lhs, rhs),
        Op::LooseNeq { lhs, rhs, .. } => (11, lhs, rhs),
        _ => return None,
    };
    Some(VnKey { tag, lhs, rhs })
}

/// Does `op` terminate a basic block? (Block boundaries reset the LVN tables — a
/// value computed in one block is not available across a branch, matching the T4
/// backend's per-block XMM-cache invalidation.)
fn is_terminator(op: &Op) -> bool {
    matches!(
        op,
        Op::Jmp { .. } | Op::JmpIfFalse { .. } | Op::JmpIfTrue { .. } | Op::Ret { .. }
    )
}

/// Compute the set of bytecode indices that START a basic block: index 0, every
/// explicit jump target, and the op immediately after any terminator. The LVN
/// tables are cleared at each block start so an available expression / copy alias
/// never crosses a branch (the conservative, obviously-correct scope — identical
/// to the T4 backend's per-block reasoning).
fn block_starts(code: &[Op]) -> Vec<bool> {
    let n = code.len();
    let mut start = vec![false; n];
    if n == 0 {
        return start;
    }
    start[0] = true;
    for (i, op) in code.iter().enumerate() {
        match *op {
            Op::Jmp { target }
            | Op::JmpIfFalse { target, .. }
            | Op::JmpIfTrue { target, .. } => {
                if (target as usize) < n {
                    start[target as usize] = true;
                }
            }
            _ => {}
        }
        if is_terminator(op) && i + 1 < n {
            start[i + 1] = true;
        }
    }
    start
}

/// A register is PROVEN NUMERIC at a program point (for the single-function
/// `OnlyNumericOperands` rule) iff it currently holds a value that cannot trigger a
/// `valueOf`/`toString` side effect when used as an arithmetic operand. We track a
/// conservative forward fact WITHIN a block: a register becomes proven-numeric when
/// it is defined by a numeric `LoadConst`, by an arithmetic op (`Add/Sub/Mul/Div` —
/// each produces a Number or DEOPTS, and a deopt resumes BEFORE op B so op B never
/// runs with a non-number there), and is cleared by any other definition or at a
/// block boundary. Booleans (compare results, LoadTrue/False) are primitives too
/// (no coercion side effect), but we conservatively do NOT mark them numeric since
/// they are not arithmetic operands in practice; the rule only needs SOUNDNESS.
struct NumericFacts {
    proven: Vec<bool>,
}
impl NumericFacts {
    fn new(n_regs: usize) -> Self {
        NumericFacts { proven: vec![false; n_regs + 1] }
    }
    fn clear(&mut self) {
        for p in self.proven.iter_mut() {
            *p = false;
        }
    }
    fn is(&self, r: Reg) -> bool {
        self.proven.get(r as usize).copied().unwrap_or(false)
    }
    fn set(&mut self, r: Reg, v: bool) {
        if let Some(slot) = self.proven.get_mut(r as usize) {
            *slot = v;
        }
    }
    /// Apply `op`'s effect on the proven-numeric facts. Called AFTER any rewrite.
    fn transfer(&mut self, op: &Op, consts: &[Value]) {
        match *op {
            Op::LoadConst { dst, k } => {
                let num = matches!(consts.get(k as usize), Some(Value::Number(_)));
                self.set(dst, num);
            }
            // Arithmetic: result is Number-or-deopt. The deopt (an operand isn't a
            // number) resumes BEFORE this op stores, so when control passes this op
            // natively the result is a Number — sound to mark proven.
            Op::Add { dst, .. }
            | Op::Sub { dst, .. }
            | Op::Mul { dst, .. }
            | Op::Div { dst, .. } => self.set(dst, true),
            // A copy inherits the source's proven-numeric fact.
            Op::Move { dst, src } => {
                let s = self.is(src);
                self.set(dst, s);
            }
            // Any other def (compares → Bool, loads of non-number) clears the fact.
            other => {
                if let Some(d) = def_reg(&other) {
                    self.set(d, false);
                }
            }
        }
    }
}


/// Run REDUNDANCY ELIMINATION + LOAD ELIMINATION + CHECK ELIMINATION over the
/// numeric-subset `code`, IN PLACE and register-PRESERVING. Returns the stats; the
/// rewritten code is left in `code`. `allow` sets how aggressively CSE may fold a
/// recomputation (see [`Allow`]). `n_regs`/`consts` come from the function under
/// optimization.
///
/// SCOPE: local value numbering per basic block (the LVN tables reset at each block
/// boundary so nothing crosses a branch — the conservative, obviously-correct scope
/// that matches the T4 backend's per-block XMM cache). Two transforms, in one pass:
///   1. STORE-TO-LOAD FORWARDING (copy propagation) — operand reads through a live
///      `Move` alias are rewritten to the alias source.
///   2. CSE / REDUNDANCY ELIM — an arith/compare op whose `(tag, lhs, rhs)` matches
///      a dominating op in this block (with no clobber of lhs/rhs/the prior dst
///      since) is rewritten to `Move dst prev`, dropping the recomputation AND its
///      implicit operand checks (check elimination).
pub fn redundancy_eliminate(
    code: &mut [Op],
    n_regs: usize,
    consts: &[Value],
    allow: Allow,
) -> RedundancyStats {
    let n = code.len();
    let mut stats = RedundancyStats::default();
    if n == 0 {
        return stats;
    }
    let starts = block_starts(code);
    let nr = n_regs + 1;

    // VALUE-CLASS NUMBERING (the LVN core). `class[r]` is the value-class id of the
    // value currently in register `r` — two registers with the SAME class hold the
    // SAME value. This unifies copy propagation and CSE: a `Move d, s` puts `d` in
    // `s`'s class (so they value-number identically), and a pure op is CSE'd by
    // keying on its operands' CLASSES (not their textual registers), so `r2*r2` and
    // an earlier `r3*r3` fold together IFF r2 and r3 currently share a class. A
    // redefinition gives the register a FRESH class (its value changed), which
    // automatically invalidates any availability that depended on it.
    let mut class: Vec<u32> = vec![0; nr];
    let mut next_class: u32 = 1;
    // Each class points at the register that currently HOLDS that class's value (the
    // canonical register a folded copy / forwarded read should name). Updated as
    // registers are (re)assigned. A class with no live holder cannot be reused.
    let mut class_holder: std::collections::HashMap<u32, Reg> = std::collections::HashMap::new();
    // Available pure expressions: (tag, class_lhs, class_rhs) → the class id of its
    // result. Cleared at block boundaries.
    let mut avail: std::collections::HashMap<(u8, u32, u32), u32> =
        std::collections::HashMap::new();
    let mut facts = NumericFacts::new(n_regs);

    // Give every register a distinct fresh class at the start (params/entry values
    // are all mutually-unknown-but-self-equal).
    let reset_classes = |class: &mut Vec<u32>, next: &mut u32, holder: &mut std::collections::HashMap<u32, Reg>| {
        holder.clear();
        for r in 0..nr {
            class[r] = *next;
            holder.insert(*next, r as Reg);
            *next += 1;
        }
    };
    reset_classes(&mut class, &mut next_class, &mut class_holder);

    let cls = |class: &[u32], r: Reg| -> u32 { class.get(r as usize).copied().unwrap_or(0) };

    for i in 0..n {
        if starts[i] {
            // BLOCK BOUNDARY: every value-class fact is per-block (the T4 backend
            // reloads-with-guard from the bank across a branch), so reset to all-
            // distinct classes and clear availability.
            reset_classes(&mut class, &mut next_class, &mut class_holder);
            avail.clear();
            facts.clear();
        }

        // (1) STORE-TO-LOAD FORWARDING — rewrite each operand read to the canonical
        //     register that currently holds that operand's value-class (an earlier
        //     equal register), so the backend reads a stable, dominating definition.
        {
            let mut forwarded = 0usize;
            let class_snapshot = class.clone();
            let holder_snapshot = class_holder.clone();
            let rewritten = rewrite_uses_by_class(
                &code[i],
                &class_snapshot,
                &holder_snapshot,
                &mut forwarded,
            );
            code[i] = rewritten;
            stats.copies_forwarded += forwarded;
        }
        let op = code[i];

        // (2) CSE — fold a recomputation of an available pure expression (keyed on
        //     operand CLASSES).
        let mut folded = false;
        if let (Some(vk), Some(dst)) = (vn_key(&op), def_reg(&op)) {
            let key = (vk.tag, cls(&class, vk.lhs), cls(&class, vk.rhs));
            if let Some(&res_class) = avail.get(&key) {
                if let Some(&prev) = class_holder.get(&res_class) {
                    // The expression's result is live in `prev`. Fold IFF dropping
                    // op B's recomputation cannot skip a coercion side effect the
                    // resuming VM would perform:
                    //   * Allow::Always — pristine resume re-runs op B; always safe.
                    //   * Allow::OnlyNumericOperands — optimized-module resume re-runs
                    //     op B as the `Move`, so op B's OPERANDS must be proven numeric
                    //     (no valueOf/toString).
                    let ok = match allow {
                        Allow::Always => true,
                        Allow::OnlyNumericOperands => {
                            facts.is(vk.lhs) && facts.is(vk.rhs)
                        }
                    };
                    if ok && prev != dst {
                        // Fold: op B becomes a copy of the dominating result, and
                        // `dst` joins the result's value-class.
                        code[i] = Op::Move { dst, src: prev };
                        stats.cse_folded += 1;
                        folded = true;
                    }
                }
            }
        }
        let op = code[i]; // re-read after a possible fold

        // (3) UPDATE the value-class numbering for THIS op's definition.
        if let Some(d) = def_reg(&op) {
            // MUTATION HOOK (test-only): the LOAD-BEARING kill in this value-class
            // design is "a register whose value CHANGES gets a FRESH value-class, so
            // any availability keyed on its old class no longer matches". When
            // `force_unsafe_cse` is set we SKIP that kill: a redefined register KEEPS
            // its OLD value-class, so a later op reading it (whose value actually
            // changed) WRONGLY value-numbers equal to a stale computation and folds
            // to a copy of the wrong result — exactly the divergence the A/B oracle
            // must catch (proving the kill is non-vacuous). NEVER on in production.
            if force_unsafe_cse() && !matches!(op, Op::Move { .. }) {
                // Keep d's class AND re-publish this op's expression availability so a
                // later identical-textual op still folds (the "no invalidation" bug).
                if let Some(vk) = vn_key(&op) {
                    let key = (vk.tag, cls(&class, vk.lhs), cls(&class, vk.rhs));
                    let c = cls(&class, d);
                    avail.insert(key, c);
                    class_holder.entry(c).or_insert(d);
                }
                facts.transfer(&op, consts);
                continue;
            }

            // The OLD class `d` held loses `d` as a holder (its value changed). If
            // `d` was that class's canonical holder, the class becomes unavailable
            // unless another register still holds it (we don't track multiple
            // holders per class — conservatively drop the holder mapping, so a later
            // fold to that class declines, which is safe, never wrong).
            let old = cls(&class, d);
            if class_holder.get(&old) == Some(&d) {
                class_holder.remove(&old);
            }

            // Assign `d`'s NEW value-class.
            let new_class = if let Op::Move { src, .. } = op {
                // A copy: `d` JOINS src's value-class (store-to-load forwarding —
                // this is what makes a folded CSE result reusable too).
                cls(&class, src)
            } else if let Some(vk) = vn_key(&op) {
                // A pure computed value. With the hook OFF (production) a computed op
                // ALWAYS gets a FRESH class (the load-bearing kill: its result is a
                // new value, distinct from any prior expression's), then we publish
                // the availability keyed on operand classes for future CSE.
                let key = (vk.tag, cls(&class, vk.lhs), cls(&class, vk.rhs));
                if folded {
                    // Defensive: a folded op is a Move (handled above); keep d's class.
                    cls(&class, d)
                } else {
                    let c = next_class;
                    next_class += 1;
                    avail.insert(key, c);
                    c
                }
            } else {
                // An opaque/unknown def (LoadConst, LoadUndef/True/False/Null, a
                // compare, …): a fresh, distinct value-class (its value is unrelated
                // to any tracked expression). LoadConst could be value-numbered by
                // its constant, but a fresh class is conservative-correct (just a
                // missed const-dedup, which T3's const-fold already handles).
                let c = next_class;
                next_class += 1;
                c
            };
            class[d as usize] = new_class;
            // Keep the EARLIEST (dominating) register as the class's canonical holder
            // — only claim the slot if the class has no live holder. This makes
            // store-to-load forwarding name the original dominating definition (better
            // codegen + shorter live ranges) rather than a later copy.
            class_holder.entry(new_class).or_insert(d);
        }

        // Advance the proven-numeric facts past this op.
        facts.transfer(&op, consts);
    }

    stats
}

/// Rewrite an op's operand READS to the canonical register currently holding each
/// operand's value-class (an earlier, dominating register that holds the same
/// value). Bumps `forwarded` for each rewrite to a DIFFERENT register. This is the
/// store-to-load-forwarding / copy-propagation step, expressed over value classes.
fn rewrite_uses_by_class(
    op: &Op,
    class: &[u32],
    holder: &std::collections::HashMap<u32, Reg>,
    forwarded: &mut usize,
) -> Op {
    let mut sub = |r: Reg| -> Reg {
        let c = class.get(r as usize).copied().unwrap_or(0);
        match holder.get(&c).copied() {
            Some(h) if h != r => {
                *forwarded += 1;
                h
            }
            _ => r,
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

/// Convenience wrapper: run [`redundancy_eliminate`] over a `BcFunction`'s code in
/// place, returning the stats. Used by the single-function and inlined T4 paths.
pub fn redundancy_eliminate_fn(f: &mut BcFunction, allow: Allow) -> RedundancyStats {
    let n_regs = f.n_regs as usize;
    let consts = f.consts.clone();
    redundancy_eliminate(&mut f.code, n_regs, &consts, allow)
}

#[cfg(test)]
mod tests;
