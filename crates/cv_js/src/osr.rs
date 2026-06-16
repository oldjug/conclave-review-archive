//! Deopt / on-stack-replacement support for the T2 JIT ↔ bytecode VM.
//!
//! Two halves:
//!
//!  * `LoopProfiler` — a per-back-edge hotness counter (kept from the original
//!    OSR slice) PLUS a per-function deopt counter that drives the T2 Phase-5
//!    **re-tiering** policy: once a function has deopted `decline_after` times we
//!    mark it `Declined` so a polymorphic function can't thrash compile→deopt.
//!
//!  * `DeoptSite` / `DeoptFrame` — T2 Phase 5 real per-guard deopt. When a JIT
//!    guard fails it returns the `T2_DEOPT_RESUME` status and writes its
//!    `DeoptSite` index into `*out`. The runner looks up the `DeoptSite`, decodes
//!    the JIT register bank (a complete VM register image at the guard's op
//!    boundary — the IDENTITY map, bank slot `i` == VM reg `i`) into `Value`
//!    registers via `DeoptFrame`, and RESUMES the bytecode VM mid-function at
//!    `bc_pc` with bit-identical results. This replaces the old i64-only
//!    `OsrFrame` (which had no external callers).

use std::collections::HashMap;

/// One back-edge profile entry.
#[derive(Debug, Clone, Copy)]
struct EdgeCounter {
    hits: u32,
    compiled: bool,
}

impl Default for EdgeCounter {
    fn default() -> Self {
        Self {
            hits: 0,
            compiled: false,
        }
    }
}

/// Per-function loop profiler. Keyed by bytecode offset of the
/// back-edge target (the loop header).
#[derive(Debug, Default)]
pub struct LoopProfiler {
    edges: HashMap<u32, EdgeCounter>,
    /// Trigger threshold — back-edges seen before we OSR.
    pub osr_threshold: u32,
}

impl LoopProfiler {
    pub fn new(osr_threshold: u32) -> Self {
        Self {
            edges: HashMap::new(),
            osr_threshold,
        }
    }

    /// Record a back-edge to `target`. Returns true the *first* time
    /// the hot-loop threshold trips for this edge — that's the
    /// signal to compile + transfer.
    pub fn record_back_edge(&mut self, target_pc: u32) -> bool {
        let entry = self.edges.entry(target_pc).or_default();
        if entry.compiled {
            return false;
        }
        entry.hits += 1;
        if entry.hits >= self.osr_threshold {
            entry.compiled = true;
            return true;
        }
        false
    }

    pub fn is_compiled(&self, target_pc: u32) -> bool {
        self.edges
            .get(&target_pc)
            .map(|e| e.compiled)
            .unwrap_or(false)
    }

    /// Mark `target_pc` as deopted — the JIT bailed back to interp.
    /// Resets the hit counter so we don't immediately retry compile.
    pub fn deopt(&mut self, target_pc: u32) {
        if let Some(e) = self.edges.get_mut(&target_pc) {
            e.hits = 0;
            e.compiled = false;
        }
    }
}

/// T2 Phase-5 re-tiering policy: a per-function deopt counter. The runner bumps
/// it every time a compiled T2 function takes a real (resume) deopt; once the
/// count reaches `decline_after`, the function should be marked `Declined` in the
/// T2 cache so it stops being recompiled (preventing a compile→deopt thrash on
/// genuinely polymorphic code). Keyed by `Rc::as_ptr(FunctionValue) as usize`.
#[derive(Debug, Default)]
pub struct DeoptPolicy {
    counts: HashMap<usize, u32>,
}

impl DeoptPolicy {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record one deopt for `fn_key`. Returns the new total.
    pub fn record(&mut self, fn_key: usize) -> u32 {
        let v = self.counts.entry(fn_key).or_insert(0);
        *v += 1;
        *v
    }

    /// True once `fn_key` has deopted at least `decline_after` times — the signal
    /// to stop recompiling it.
    pub fn should_decline(&self, fn_key: usize, decline_after: u32) -> bool {
        self.counts.get(&fn_key).copied().unwrap_or(0) >= decline_after
    }

    /// Current deopt count for `fn_key` (0 if never deopted).
    pub fn count(&self, fn_key: usize) -> u32 {
        self.counts.get(&fn_key).copied().unwrap_or(0)
    }

    pub fn clear(&mut self) {
        self.counts.clear();
    }
}

/// Why a JIT guard can fail. Recorded per `DeoptSite` so the deopt-fuzzer can
/// sweep every guard of every reason and the resume oracle can attribute a
/// divergence to the exact guard kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeoptReason {
    /// An arithmetic / comparison / JmpIfFalse operand was not a number (the
    /// `t2_emit_load_num` boxed-non-int32 guard, or the JmpIfFalse non-boolean
    /// non-number guard).
    NonNumber,
    /// A GetProp receiver was not an object.
    NonObject,
    /// A GetProp shape guard missed (no warmed shape matched), or the immediate
    /// helper returned a non-immediate slot value.
    ShapeMiss,
    /// A re-entry CALL / LoadGlobal declined before any side effect (a
    /// non-callable callee) — a pre-effect deopt.
    CallDecline,
    /// A GetIdx / SetIdx receiver was not a `Value::Array` (a named-property
    /// indexing or a string/object index — outside the array fast path).
    NonArray,
    /// A GetIdx / SetIdx index was not a non-negative integer (negative /
    /// fractional / NaN / non-number → a NAMED-property lookup on the VM).
    BadIndex,
    /// A GetIdx element read a HOLE, an accessor-wrapper, or another non-admitted
    /// lane (the VM produces the exact image — e.g. the `Value::Hole` sentinel or
    /// a getter result). A SetIdx that would extend the array (OOB write) also
    /// resumes here (a structural change the VM performs).
    HoleOrSpecial,
    /// The compiled function ran off its end without a `Ret` (defensive
    /// fall-through; a bytecode function always ends in Ret, so this never fires
    /// in practice — but it resumes the VM rather than corrupting).
    FallThrough,
}

/// One recorded deopt point in a compiled T2 function. The guard's deopt stub
/// returns `T2_DEOPT_RESUME` and writes THIS site's index into `*out`; the runner
/// looks the site up and resumes the VM at `bc_pc`.
///
/// INVARIANT (the load-bearing correctness contract): `bc_pc` is the bytecode
/// index of the op whose INPUT guard failed, BEFORE that op stored its output to
/// its bank slot. So when the VM takes over at `bc_pc` it re-executes exactly that
/// op (and everything after) over a register image that has NOT yet seen the op's
/// effect — the JIT bank is the exact pre-op VM register file. Every op stores its
/// result back to its bank slot before the next op, so at ANY guard the bank is a
/// complete register image (the identity reconstruction map).
#[derive(Debug, Clone, Copy)]
pub struct DeoptSite {
    /// Machine-code byte offset of the guard's deopt stub (diagnostic + the
    /// bc_pc-mutation test arm reaches in by index, not by offset).
    pub native_off: usize,
    /// Bytecode index to RESUME the VM at (the op boundary; see the invariant).
    pub bc_pc: usize,
    /// Why this guard can fail (for fuzzing + attribution).
    pub reason: DeoptReason,
}

/// The reconstructed VM frame for a resume deopt: the JIT bank decoded to `Value`
/// registers plus the bytecode index to resume at. Built by the runner from a
/// `DeoptSite` + the live JIT bank; `into_value_regs` produces the register `Vec`
/// `run_function_inner` resumes over.
///
/// Replaces the old i64-only `OsrFrame`. The bank holds `JsVal`s (one `u64`
/// each); decoding via `JsVal::to_value` reconstructs the IDENTICAL `Value`
/// (numbers round-trip exactly — NaN is already canonical on box, so no
/// tagged-value-as-NaN hazard — and pointer lanes round-trip to the same `Rc`).
#[derive(Debug, Clone)]
pub struct DeoptFrame {
    /// Module function index the resumed VM runs.
    pub fn_idx: usize,
    /// Bytecode index to resume at (`DeoptSite::bc_pc`).
    pub bc_pc: usize,
    /// The decoded register image — bank slot `i` == VM reg `i` (identity map).
    pub regs: Vec<crate::interp::Value>,
}

impl DeoptFrame {
    /// Build a frame by decoding a JIT bank (raw `JsVal` bits) into `Value`
    /// registers. `bank` is the live, complete pre-op register image.
    ///
    /// # Safety
    /// Each pointer-lane slot's originating `Rc` must be alive (the owning bank
    /// keeps a +1 of every pointer slot for the whole run, so this holds while the
    /// bank is alive — the runner decodes BEFORE dropping the bank). `to_value`
    /// takes its own +1 per pointer slot (a borrowed-handle clone), so the
    /// returned `Value`s outlive the bank's teardown.
    pub unsafe fn from_bank(fn_idx: usize, bc_pc: usize, bank: &[crate::jsval::JsVal]) -> Self {
        let regs: Vec<crate::interp::Value> =
            bank.iter().map(|jv| unsafe { jv.to_value() }).collect();
        DeoptFrame {
            fn_idx,
            bc_pc,
            regs,
        }
    }

    /// Consume the frame, yielding the register image the VM resumes over.
    pub fn into_value_regs(self) -> Vec<crate::interp::Value> {
        self.regs
    }
}

// ======================================================================
// T4 EXTENSION 1 — INLINED-FRAME DEOPT (INLINE-DEOPT-TO-CALLER).
//
// This is the ONLY genuinely new deopt DATA the T4 (Maglev-class) tier requires,
// and it is scaffolded HERE (P0) — as a pure data + reconstruction-math addition
// — BEFORE any inliner exists, so the inlined-frame-deopt fuzzer can prove the
// reconstruction is byte-identical to the un-inlined VM on the existing corpus.
//
// THE PROBLEM (when P3 inlining lands): T4 inlines a small pure callee `g` into
// caller `f`'s hot loop, so the body of `g` runs with NO separate VM frame. A
// type/shape guard INSIDE the inlined `g` body fails. The naive thing — synthesize
// a mid-`g` VM frame AND resume `f` at the Call — needs two reconstructed frames
// and breaks the per-frame identity-map invariant (the bank is `f`'s register file,
// not `g`'s).
//
// THE CHOSEN DESIGN (mirrors V8/Maglev's deopt-to-the-call-site for an
// eager-deopt without a full multi-frame translation): INLINE-DEOPT-TO-CALLER.
// A guard inside the inlined region deopts to the CALLER's `Call` bytecode op,
// reconstructing the call's ARGUMENT registers from the bank, and lets the VM
// perform the ORDINARY (non-inlined) call — which re-enters `g` on the VM/T2.
// This keeps the proven per-frame identity-map invariant (the reconstructed frame
// is `f`'s, exactly as the single-frame deopt does) and costs, on the rare deopt,
// ONE extra (re-)execution of the call — never a wrong value. Because `bc_pc`
// points at the `Call` op (BEFORE its `dst` store, like every other DeoptSite),
// the VM re-executes the whole call and stores its result, identical to a plain
// VM run.
//
// V8 SOURCE MODELED: V8's deoptimizer reconstructs interpreter frames from a
// `TranslationArray` describing each (possibly inlined) frame's registers +
// bytecode offset; Maglev emits a `LazyDeoptInfo`/`EagerDeoptInfo` whose
// `DeoptFrame` carries the bytecode position + a value list. Our INLINE-DEOPT-TO-
// CALLER is the eager-deopt-to-the-call special case: instead of translating a
// mid-callee frame we resume the caller AT the call, which the deoptimizer is
// always free to do for an eager type/shape guard (the call has no committed
// effect yet — its `dst` is unwritten). This is strictly simpler and provably
// correct (the fuzzer below proves it == the non-inlined VM result).
//
// EAGER, not LAZY: like the existing single-frame DeoptSite, this fires
// synchronously at the guard, over a complete pre-call bank image. (Lazy
// invalidate-on-event deopt is deferred to T5.)
// ======================================================================

/// The extra data an INLINED-FRAME `DeoptSite` carries: enough to reconstruct the
/// CALLER frame at the inlined `Call` op (the INLINE-DEOPT-TO-CALLER design).
///
/// When a guard inside an inlined callee fails, the runner does NOT resume in the
/// (non-existent) callee VM frame. Instead it:
///   1. reconstructs the caller's register file from the bank (the IDENTITY map,
///      exactly as `DeoptFrame::from_bank`), then
///   2. resumes the caller VM at `caller_bc_pc_of_call` — the `Call` op — so the
///      VM performs the ordinary call with the live argument registers and stores
///      the result to the call's `dst`, exactly as an un-inlined run would.
///
/// `arg_slot_map` records WHICH caller bank slots hold the call's arguments at the
/// moment of the (inlined) call, in argument order. It is the seam the inliner
/// fills; the reconstruction MUST find those slots already populated in the bank
/// (the inliner spills/keeps the args in their caller slots across the inlined
/// region, just as the bytecode `CallFn`/`CallValue` op expects `first_arg..` to
/// be live). `callee_entry_bc_pc` is recorded for diagnostics + the deferred
/// precise multi-frame translation (T5); the chosen INLINE-DEOPT-TO-CALLER form
/// does not consult it at run time (it resumes the caller, not the callee).
#[derive(Debug, Clone)]
pub struct InlinedFrame {
    /// Bytecode index of the CALLER's `Call`/`CallFn`/`CallValue`/`New` op that
    /// was inlined away. The runner resumes the CALLER VM here (BEFORE the op's
    /// `dst` store — the bank is the exact pre-call register image), so the VM
    /// re-runs the real call. This is the load-bearing resume target.
    pub caller_bc_pc_of_call: usize,
    /// Bytecode index of the inlined callee's entry (offset 0 of the callee body).
    /// DIAGNOSTIC + reserved for the T5 precise multi-frame translation; the
    /// INLINE-DEOPT-TO-CALLER form does not use it at run time.
    pub callee_entry_bc_pc: usize,
    /// Caller bank-slot indices holding the call's arguments, in argument order
    /// (`args[0]` = bank slot `arg_slot_map[0]`, …). These slots are decoded from
    /// the bank into the resumed caller register file by the ordinary identity-map
    /// reconstruction; the resumed `Call` op then reads them as its `first_arg..`
    /// operands. Recorded for the precise translation + the verifier; the
    /// identity-map resume reconstructs the WHOLE bank, so these are guaranteed
    /// present iff they are in-range bank slots (the verifier below checks that).
    pub arg_slot_map: Vec<usize>,
}

/// An INLINED-FRAME deopt site: a base `DeoptSite` (whose `bc_pc` is the CALLER's
/// `Call` op per the INLINE-DEOPT-TO-CALLER design) PLUS the `InlinedFrame`
/// reconstruction data. Kept as a SEPARATE type so the existing single-frame
/// `DeoptSite` path (jit.rs emission, `run_t2lite_call` resume) is byte-for-byte
/// UNTOUCHED — the inliner (P3) records these in a parallel table consulted only
/// when a guard's site is an inlined one. P0 ships the type + the reconstruction
/// math + the fuzzer; nothing emits one yet (no inliner), so the production build
/// is unchanged.
#[derive(Debug, Clone)]
pub struct InlinedDeoptSite {
    /// The base site. `bc_pc` is the CALLER's `Call` op index (the resume target);
    /// `reason` is the inner guard's reason; `native_off` is the inner guard's stub.
    pub base: DeoptSite,
    /// The caller-frame reconstruction data (INLINE-DEOPT-TO-CALLER).
    pub frame: InlinedFrame,
}

impl InlinedDeoptSite {
    /// VERIFY this inlined-frame site against the CALLER function's bank size
    /// (`n_regs`). The reconstruction resumes the caller at `caller_bc_pc_of_call`
    /// over the full identity-map register image, so it is correct iff:
    ///   * the resume `bc_pc` (the Call op index) is in range of the caller code,
    ///     and
    ///   * every recorded argument slot is an in-range bank slot (so the
    ///     identity-map decode populates it — exactly the same in-range discipline
    ///     the SafepointMap verifier applies to roots).
    /// This is the inlined-frame analogue of `SafepointMap::verify_against_bank`:
    /// it catches an inliner that records an out-of-range arg slot (which would
    /// resume the call with a garbage / missing argument) at COMPILE time. P0
    /// runs it in the fuzzer; P3 runs it as a debug-assert on every inlined compile.
    pub fn verify_against_caller(&self, caller_code_len: usize, caller_n_regs: usize) -> bool {
        if self.frame.caller_bc_pc_of_call >= caller_code_len {
            return false;
        }
        self.frame
            .arg_slot_map
            .iter()
            .all(|&slot| slot < caller_n_regs)
    }
}

/// Reconstruct the CALLER's VM register file for an INLINE-DEOPT-TO-CALLER bailout.
///
/// This is the Extension-1 reconstruction MATH, proven by the inlined-frame-deopt
/// fuzzer before any inliner exists. Given the live JIT bank (the caller's
/// identity-map register image at the inlined call boundary) and the
/// `InlinedFrame`, it produces the `(regs, resume_bc_pc)` the VM resumes the
/// CALLER over. The decode is IDENTICAL to `DeoptFrame::from_bank` (every bank
/// slot → its `Value` via `JsVal::to_value`, the identity map) — the inlined-frame
/// design's whole point is that NO new reconstruction primitive is needed: the
/// caller frame is reconstructed exactly as a single-frame deopt would, and the
/// resume `bc_pc` is the recorded Call op. The arg slots are already part of the
/// full register image, so the re-run `Call` op reads them as its operands.
///
/// # Safety
/// Same contract as [`DeoptFrame::from_bank`]: the bank's pointer-lane slots must
/// be alive for the decode (the owning bank keeps a +1 per slot; `to_value` takes
/// its own +1 so the returned `Value`s outlive the bank teardown).
pub unsafe fn reconstruct_caller_frame(
    fn_idx: usize,
    site: &InlinedDeoptSite,
    bank: &[crate::jsval::JsVal],
) -> DeoptFrame {
    // The identity-map decode is the proven single-frame reconstruction; the only
    // inlined-frame-specific choice is the resume bc_pc (the Call op, not the inner
    // guard's op). The full register image already contains the argument slots.
    unsafe { DeoptFrame::from_bank(fn_idx, site.frame.caller_bc_pc_of_call, bank) }
}

// ----------------------------------------------------------------------
// B1 / B3 — safepoint stack maps (the GC-rooting groundwork for T3).
//
// A SAFEPOINT is a native-code position where a GC can run: every helper /
// runtime call, every allocation site, and every loop back-edge. At such a
// point an OPTIMIZING tier (T3) may hold heap-pointer JsVals in HOST REGISTERS
// or SPILL SLOTS that are NOT yet written back to their identity-map bank slot —
// so the GC would not see them via `gc_seed_jit_banks` and could clear (the
// clear-not-free mark-sweep) the object out from under live code: a UAF / silent
// data-loss.
//
// THE B1 DELIVERABLE here is the EMISSION + RECORDING format only (kept on the
// codegen side, NOT in cv_asm — cv_asm stays dependency-free; it only exposes
// `Emitter::here()`). B3 consumes a `SafepointMap` to drive rooting, and the B1
// design ships the SIMPLER, provably-safe discipline: T3 SPILLS every live
// pointer-lane value to its identity-map bank slot BEFORE each safepoint, so the
// already-proven `gc_seed_jit_banks` (which roots bank slots) covers it. The
// `SafepointRec` `live_roots` bitset is recorded NOW so the precise
// register-resident-roots seeding pass (`gc_seed_safepoint_regs`) can land later
// WITHOUT a format change — it is the explicit-but-deferred follow-on.
//
// CRITICAL SEMANTICS: `live_roots` records ONLY bank/spill slots that hold a
// POINTER-LANE JsVal (Object/Array/Function/Native/BcClosure/StrBig — anything
// the GC must mark). It MUST EXCLUDE slots/registers holding UNBOXED numbers
// (xmm f64 or the int32 GPR lane): those are not heap refs and rooting them is a
// type-confusion bug (a raw f64 bit pattern is not a valid Rc).

/// What kind of GC-safe point this is — for attribution + so a future precise
/// scanner can special-case (e.g. an allocation site's result reg is not yet
/// live, a back-edge's are).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SafepointKind {
    /// A runtime/helper call (`rt_getprop_slot_owning_store`, `rt_call_value`, …).
    /// The classic UAF window: pointer values live across the call.
    HelperCall,
    /// An allocation site (object/array literal, boxing). The allocation itself
    /// may trigger a collection; values live BEFORE the alloc must be rooted.
    Allocation,
    /// A loop back-edge — the periodic-collection trigger point inside hot loops.
    BackEdge,
}

/// One recorded safepoint: a native-code offset plus the set of bank/spill slots
/// holding live HEAP POINTERS at that offset (the roots the GC must mark).
///
/// `live_roots` is a bitset over identity-map bank slots (bit `i` set ⇒ bank slot
/// `i` holds a pointer-lane JsVal that is live across this safepoint). It NEVER
/// includes a slot known to hold an unboxed number — see the module note.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SafepointRec {
    /// Machine-code byte offset of the safepoint (from `Emitter::here()`).
    pub native_off: usize,
    /// Why this is a safepoint (call / alloc / back-edge).
    pub kind: SafepointKind,
    /// Bitset of identity-map bank slots holding live heap pointers. Bit `i` ⇒
    /// slot `i` is a root. Up to 64 slots; T3 functions over 64 regs fall back to
    /// the spill-everything discipline (the bitset is advisory for the precise
    /// pass, not load-bearing for the B1 spill-to-bank correctness).
    pub live_roots: u64,
}

impl SafepointRec {
    /// True if bank slot `i` is recorded as a live heap root here.
    pub fn is_root(&self, slot: usize) -> bool {
        slot < 64 && (self.live_roots & (1u64 << slot)) != 0
    }

    /// Number of live heap roots at this safepoint.
    pub fn root_count(&self) -> u32 {
        self.live_roots.count_ones()
    }
}

/// The per-compiled-function safepoint map: an ordered list of `SafepointRec`s
/// (by ascending `native_off`). Attached beside `DeoptSite`s on the installed
/// function. B3's GC integration walks this to root register/spill-resident
/// pointers; B1 builds + unit-tests the recording.
#[derive(Debug, Clone, Default)]
pub struct SafepointMap {
    recs: Vec<SafepointRec>,
}

impl SafepointMap {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a safepoint at `native_off` (typically `em.here()` captured right
    /// AFTER the call/alloc instruction, i.e. the return address — the PC the GC
    /// would observe). `live_roots` is the pointer-lane bank-slot bitset.
    ///
    /// Records are kept in insertion order, which T3 codegen produces in ascending
    /// native offset; `find` relies on that ordering.
    pub fn record(&mut self, native_off: usize, kind: SafepointKind, live_roots: u64) {
        debug_assert!(
            self.recs.last().is_none_or(|r| r.native_off <= native_off),
            "safepoints must be recorded in ascending native-offset order"
        );
        self.recs.push(SafepointRec {
            native_off,
            kind,
            live_roots,
        });
    }

    /// Build a `live_roots` bitset from an iterator of pointer-lane slot indices.
    /// Helper for codegen: pass the slots the allocator currently has holding a
    /// pointer-lane JsVal. Slots ≥ 64 are dropped from the bitset (the function
    /// then relies on the spill-everything discipline; see the module note).
    pub fn roots_from_slots(slots: impl IntoIterator<Item = usize>) -> u64 {
        let mut mask = 0u64;
        for s in slots {
            if s < 64 {
                mask |= 1u64 << s;
            }
        }
        mask
    }

    /// Exact-match lookup of the safepoint at `native_off` (the GC has the PC and
    /// wants its root set). None if `native_off` is not a recorded safepoint.
    pub fn find(&self, native_off: usize) -> Option<&SafepointRec> {
        // Records are sorted ascending by native_off; binary search.
        self.recs
            .binary_search_by_key(&native_off, |r| r.native_off)
            .ok()
            .map(|i| &self.recs[i])
    }

    /// All recorded safepoints (ascending native offset).
    pub fn records(&self) -> &[SafepointRec] {
        &self.recs
    }

    pub fn len(&self) -> usize {
        self.recs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.recs.is_empty()
    }
}

// ============================================================================
// B3 — SAFEPOINT ROOTING DISCIPLINE (the UAF keystone).
//
// THE HAZARD (ground-truth `interp.rs:3903`): the live GC is a clear-not-free
// mark-sweep that EMPTIES any unmarked container even if its `Rc` strong count
// is > 0. So a heap value reachable ONLY through an optimizing tier's HOST
// REGISTER / un-spilled spill-slot across a GC-safe point is NOT seen by
// `gc_seed_jit_banks` (which scans only the registered bank's slots) → the GC
// clears it out from under live code = a use-after-free-shaped silent data loss.
//
// T2 dodges this by writing the bank immediately after every op (the identity-
// map invariant) so the only un-rooted window is the brief native-register
// shuffle between two stores — and across a CALL, T2 stores back BEFORE the call
// by construction. An OPTIMIZING T3 register allocator DELIBERATELY keeps values
// in registers across op boundaries (that is the whole point), so the T2 dodge
// no longer holds for free: T3 MUST establish + PROVE a rooting discipline
// before it may hold any heap ref in a register across a safepoint.
//
// THE DISCIPLINE B3 SHIPS (the simpler, provably-safe one; the precise register
// scanner is the documented deferred follow-on below):
//
//   At EVERY safepoint, every live pointer-lane value is SPILLED to its
//   identity-map bank slot before the call/alloc/back-edge. The bank IS the
//   spill area: a value "spilled to slot i" lives in `bank[i]`, which
//   `gc_seed_jit_banks` already scans + roots. So a pointer root recorded at a
//   safepoint as a BANK-SLOT INDEX is automatically covered by the proven
//   bank-rooting — B3 reduces to (a) PROVING codegen records each live pointer
//   root as an in-range bank slot (the verifier below), and (b) a force-GC-at-
//   safepoint mutation test that the rooting is load-bearing.
//
// This ALSO retro-fixes T2's latent callee-saved-register hazard: T2 already
// stores-before-call by construction, and `verify_against_bank` lets a T2/T3
// safepoint map ASSERT that discipline mechanically rather than by inspection.
// ============================================================================

/// Why a safepoint map failed the rooting-discipline verification. A `Ok`
/// result means every live pointer-lane root at every safepoint is a bank-slot
/// index in range — i.e. it is covered by `gc_seed_jit_banks` and a GC at any of
/// these safepoints cannot clear a live heap value out from under the code.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SafepointDisciplineError {
    /// A safepoint records a live pointer root at bank slot `slot`, but the
    /// owning function's bank has only `bank_len` slots — so `gc_seed_jit_banks`
    /// (which scans `bank[0..bank_len]`) would NOT root it. This is the exact
    /// UAF-enabling codegen bug: a pointer live across the safepoint that is not
    /// mirrored into a scanned bank slot.
    RootOutOfBankRange { native_off: usize, slot: usize, bank_len: usize },
}

impl SafepointMap {
    /// B3 VERIFIER — prove the spill-to-bank rooting discipline for this map
    /// against a bank of `bank_len` slots (the owning function's `n_regs`).
    ///
    /// The contract: a pointer-lane value live across a safepoint must have been
    /// spilled to an identity-map bank slot, recorded in that safepoint's
    /// `live_roots`. Since `gc_seed_jit_banks` scans `bank[0..bank_len]`, every
    /// recorded root MUST be an index `< bank_len` — otherwise the GC cannot see
    /// it and could clear the value (the UAF). This is a pure structural check
    /// over the recorded bitsets; it is cheap enough to run on every compile (a
    /// debug-assert in production codegen) so a future heap-T3 allocator that
    /// "forgets" to spill a pointer before a safepoint is caught at compile time,
    /// not by a corrupted page at run time.
    ///
    /// Returns the FIRST violation (or `Ok(())` if the discipline holds).
    pub fn verify_against_bank(&self, bank_len: usize) -> Result<(), SafepointDisciplineError> {
        for rec in &self.recs {
            // Walk the set bits of the root bitset; each set bit is a bank slot
            // index that MUST be in range.
            let mut bits = rec.live_roots;
            while bits != 0 {
                let slot = bits.trailing_zeros() as usize;
                bits &= bits - 1; // clear lowest set bit
                if slot >= bank_len {
                    return Err(SafepointDisciplineError::RootOutOfBankRange {
                        native_off: rec.native_off,
                        slot,
                        bank_len,
                    });
                }
            }
        }
        Ok(())
    }

    /// True iff every live pointer root at every safepoint is covered by a bank
    /// of `bank_len` slots (the boolean form of [`verify_against_bank`], for a
    /// `debug_assert!` in codegen / the GC integration).
    pub fn roots_covered_by_bank(&self, bank_len: usize) -> bool {
        self.verify_against_bank(bank_len).is_ok()
    }
}

impl SafepointRec {
    /// B3 GC-INTEGRATION precondition check. Given the collection PC landed on
    /// THIS safepoint and the owning function's `bank_len`, return whether every
    /// live pointer root here is bank-resident (slot `< bank_len`). When true,
    /// `gc_seed_jit_banks` covers this safepoint's roots completely — the GC will
    /// mark every heap value the optimized code holds live across it, so the
    /// clear-not-free sweep cannot empty one. The GC asserts this on a collection
    /// that observes a JIT return-address PC (see the integration note in
    /// `interp::gc_collect`); a false here means a missing spill (the UAF), which
    /// must be a hard error in debug (a codegen bug, not a recoverable state).
    pub fn roots_bank_resident(&self, bank_len: usize) -> bool {
        let mut bits = self.live_roots;
        while bits != 0 {
            let slot = bits.trailing_zeros() as usize;
            bits &= bits - 1;
            if slot >= bank_len {
                return false;
            }
        }
        true
    }
}

// ----------------------------------------------------------------------
// B3 DEFERRED FOLLOW-ON (format ready now, pass lands later WITHOUT a recompile-
// format change) — PRECISE REGISTER-RESIDENT ROOTS ACROSS A SAFEPOINT.
//
// The shipped B3 discipline is spill-pointers-to-bank-before-safepoint, which is
// provably safe via the proven `gc_seed_jit_banks` bank scan. The FASTER variant
// keeps a pointer in a callee-saved host register (rbx/rdi/rsi/r12-r15) ACROSS a
// call and roots it directly — avoiding the spill store. To do that soundly the
// helper-call trampoline's prolog must capture the live callee-saved registers
// into a thread-local `SafepointFrame`, and the GC must, on a collection that
// lands on a JIT return-address PC, consult the `SafepointRec` at that PC to know
// WHICH captured registers hold pointers (vs unboxed numbers) and root only
// those.
//
// This is EXPLICITLY DEFERRED (it needs a trampoline change + a precise PC→frame
// walk that the current message-loop-boundary GC does not require). The
// `SafepointRec` already carries the per-register/slot live-root bitset, so this
// pass lands later with NO format change — exactly the seam B1 reserved. The
// scaffold below documents the shape; `gc_seed_safepoint_regs` is a no-op until
// the trampoline captures a real `SafepointFrame` (it never fabricates roots, so
// shipping the dormant scaffold cannot create a false root or a UAF).
// ----------------------------------------------------------------------

/// A snapshot of the callee-saved host registers captured by a helper-call
/// trampoline at a safepoint, paired with the safepoint's native PC so the GC can
/// look up which of them hold pointer-lane `JsVal`s (the precise root set).
///
/// DEFERRED: nothing captures one yet (the shipped discipline spills to the bank
/// instead). The struct is the reserved format so the precise pass lands without
/// a recompile-format change. `regs` is indexed by a fixed callee-saved register
/// numbering the trampoline + codegen agree on.
#[derive(Debug, Clone, Copy)]
pub struct SafepointFrame {
    /// Native return-address PC of the safepoint the collection landed on — keys
    /// into the owning function's `SafepointMap` to recover the live-root bitset.
    pub native_off: usize,
    /// Captured callee-saved register values (raw `JsVal` bit patterns). Only the
    /// indices set in the matching `SafepointRec.live_roots` are pointers; the
    /// rest are unboxed numbers / scratch and MUST NOT be rooted.
    pub regs: [u64; 8],
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn back_edge_below_threshold_does_not_compile() {
        let mut p = LoopProfiler::new(10);
        for _ in 0..9 {
            assert!(!p.record_back_edge(42));
        }
        assert!(!p.is_compiled(42));
    }

    #[test]
    fn back_edge_at_threshold_triggers_compile() {
        let mut p = LoopProfiler::new(5);
        let mut triggered = false;
        for _ in 0..5 {
            triggered |= p.record_back_edge(7);
        }
        assert!(triggered);
        assert!(p.is_compiled(7));
    }

    #[test]
    fn already_compiled_edges_do_not_retrigger() {
        let mut p = LoopProfiler::new(1);
        assert!(p.record_back_edge(0));
        assert!(!p.record_back_edge(0));
        assert!(!p.record_back_edge(0));
    }

    #[test]
    fn deopt_resets_counter_and_compile_flag() {
        let mut p = LoopProfiler::new(2);
        p.record_back_edge(1);
        p.record_back_edge(1);
        assert!(p.is_compiled(1));
        p.deopt(1);
        assert!(!p.is_compiled(1));
        assert!(!p.record_back_edge(1));
        assert!(p.record_back_edge(1));
    }

    // ------------------------------------------------------------------
    // B1 — safepoint stack-map recording + bitset semantics.
    // ------------------------------------------------------------------

    #[test]
    fn safepoint_record_and_find_by_native_offset() {
        let mut m = SafepointMap::new();
        // Three safepoints at ascending native offsets, each with its own root set.
        m.record(16, SafepointKind::HelperCall, 0b0010); // slot 1 is a root
        m.record(48, SafepointKind::Allocation, 0b0000); // no roots (alloc result)
        m.record(80, SafepointKind::BackEdge, 0b1001); // slots 0 and 3 are roots
        assert_eq!(m.len(), 3);

        let sp = m.find(16).expect("safepoint at 16");
        assert_eq!(sp.kind, SafepointKind::HelperCall);
        assert!(sp.is_root(1));
        assert!(!sp.is_root(0));
        assert_eq!(sp.root_count(), 1);

        let bp = m.find(80).expect("safepoint at 80");
        assert_eq!(bp.kind, SafepointKind::BackEdge);
        assert!(bp.is_root(0) && bp.is_root(3) && !bp.is_root(1));
        assert_eq!(bp.root_count(), 2);

        // A non-safepoint offset (between recorded sites) returns None.
        assert!(m.find(40).is_none());
        // The allocation site has zero roots — its result reg is not yet live.
        assert_eq!(m.find(48).unwrap().root_count(), 0);
    }

    #[test]
    fn safepoint_roots_from_slots_excludes_number_lane_slots() {
        // THE load-bearing semantic: live_roots records ONLY pointer-lane slots.
        // Codegen passes the slots holding HEAP pointers (here slots 2 and 5); the
        // slots holding unboxed numbers (xmm/int32 lane — say slots 0,1,3,4) are
        // NEVER passed in, so they are NOT roots. Rooting a raw f64 bit pattern as
        // an Rc would be a type-confusion UAF — this is what the exclusion prevents.
        let pointer_slots = [2usize, 5];
        let mask = SafepointMap::roots_from_slots(pointer_slots);
        let mut m = SafepointMap::new();
        m.record(0, SafepointKind::HelperCall, mask);
        let sp = m.find(0).unwrap();
        assert!(sp.is_root(2) && sp.is_root(5));
        // Number-lane slots must be excluded.
        for num_slot in [0usize, 1, 3, 4, 6, 7] {
            assert!(
                !sp.is_root(num_slot),
                "slot {num_slot} holds a number and must NOT be a GC root"
            );
        }
        assert_eq!(sp.root_count(), 2);
    }

    #[test]
    fn safepoint_slots_over_63_are_dropped_from_bitset() {
        // Slots ≥ 64 don't fit the u64 bitset; they're dropped (the function then
        // relies on the spill-everything discipline). No panic, no OOB.
        let mask = SafepointMap::roots_from_slots([3usize, 64, 100]);
        assert_eq!(mask, 1u64 << 3, "only slot 3 fits the bitset");
        let mut m = SafepointMap::new();
        m.record(0, SafepointKind::HelperCall, mask);
        let sp = m.find(0).unwrap();
        assert!(sp.is_root(3));
        assert!(!sp.is_root(64)); // is_root bounds-checks, never panics
    }

    #[test]
    #[should_panic(expected = "ascending")]
    fn safepoint_record_out_of_order_is_caught_in_debug() {
        // Records MUST be in ascending native-offset order (find binary-searches).
        // A debug_assert catches a codegen ordering bug.
        let mut m = SafepointMap::new();
        m.record(100, SafepointKind::HelperCall, 0);
        m.record(50, SafepointKind::HelperCall, 0); // out of order → panic in debug
    }

    #[test]
    fn deopt_policy_declines_after_threshold() {
        let mut pol = DeoptPolicy::new();
        let key = 0xABCDusize;
        assert!(!pol.should_decline(key, 3));
        assert_eq!(pol.record(key), 1);
        assert_eq!(pol.record(key), 2);
        assert!(!pol.should_decline(key, 3));
        assert_eq!(pol.record(key), 3);
        assert!(pol.should_decline(key, 3));
        // A different function is independent.
        assert!(!pol.should_decline(0x1234, 3));
    }

    // ------------------------------------------------------------------
    // B3 — safepoint rooting-discipline verifier (the spill-to-bank check).
    // ------------------------------------------------------------------

    #[test]
    fn safepoint_discipline_holds_when_every_root_is_in_bank_range() {
        // A function with a 6-slot bank. Two safepoints, each rooting pointer
        // slots that are all < 6 (spilled to bank). The discipline holds.
        let mut m = SafepointMap::new();
        m.record(16, SafepointKind::HelperCall, SafepointMap::roots_from_slots([1usize, 4]));
        m.record(48, SafepointKind::BackEdge, SafepointMap::roots_from_slots([0usize, 5]));
        assert_eq!(m.verify_against_bank(6), Ok(()));
        assert!(m.roots_covered_by_bank(6));
        // Per-safepoint: every recorded root is bank-resident.
        assert!(m.find(16).unwrap().roots_bank_resident(6));
        assert!(m.find(48).unwrap().roots_bank_resident(6));
    }

    #[test]
    fn safepoint_discipline_no_roots_is_vacuously_covered() {
        // A numeric-only kernel (today's T3 subset) records safepoints with NO
        // pointer roots — nothing to spill, so the discipline holds for ANY bank
        // size, including an empty bank. This is why today's numeric T3 is already
        // safe: it never holds a heap ref across a safepoint.
        let mut m = SafepointMap::new();
        m.record(8, SafepointKind::BackEdge, 0);
        m.record(24, SafepointKind::HelperCall, 0);
        assert_eq!(m.verify_against_bank(0), Ok(()));
        assert!(m.roots_covered_by_bank(0));
    }

    #[test]
    fn safepoint_discipline_catches_a_root_outside_the_bank() {
        // THE codegen-bug detector: a safepoint records a pointer root at bank
        // slot 7, but the bank has only 6 slots. `gc_seed_jit_banks` scans
        // bank[0..6], so slot 7 would NOT be rooted — a live heap value the GC
        // can't see = the UAF. The verifier MUST reject this.
        let mut m = SafepointMap::new();
        m.record(16, SafepointKind::HelperCall, SafepointMap::roots_from_slots([1usize, 7]));
        let err = m.verify_against_bank(6).unwrap_err();
        assert_eq!(
            err,
            SafepointDisciplineError::RootOutOfBankRange { native_off: 16, slot: 7, bank_len: 6 }
        );
        assert!(!m.roots_covered_by_bank(6));
        assert!(!m.find(16).unwrap().roots_bank_resident(6));
        // With a bank large enough to cover slot 7, the SAME map is fine — proving
        // the check is the range relation, not a constant rejection.
        assert_eq!(m.verify_against_bank(8), Ok(()));
        assert!(m.find(16).unwrap().roots_bank_resident(8));
    }

    #[test]
    fn safepoint_discipline_reports_the_first_violation() {
        // The verifier returns the FIRST out-of-range root (lowest native_off,
        // lowest slot) so a codegen diagnostic points at the earliest bug.
        let mut m = SafepointMap::new();
        m.record(10, SafepointKind::HelperCall, SafepointMap::roots_from_slots([2usize])); // ok
        m.record(20, SafepointKind::BackEdge, SafepointMap::roots_from_slots([3usize, 9])); // 9 oob
        m.record(30, SafepointKind::HelperCall, SafepointMap::roots_from_slots([12usize])); // also oob
        let err = m.verify_against_bank(4).unwrap_err();
        // First violation is at native_off 20, slot 9 (the earliest safepoint with
        // an out-of-range root; within it, the lowest such slot).
        assert_eq!(
            err,
            SafepointDisciplineError::RootOutOfBankRange { native_off: 20, slot: 9, bank_len: 4 }
        );
    }

    // ------------------------------------------------------------------
    // T4 EXTENSION 1 — INLINED-FRAME DEOPT site data + verifier (the
    // reconstruction-math structural gate, before any inliner exists).
    // ------------------------------------------------------------------

    /// An inlined-frame site whose resume target + every arg slot are in range of
    /// the caller verifies — it can be reconstructed by the identity-map decode.
    #[test]
    fn inlined_frame_site_verifies_when_in_caller_range() {
        let site = InlinedDeoptSite {
            base: DeoptSite { native_off: 0, bc_pc: 5, reason: DeoptReason::NonNumber },
            frame: InlinedFrame {
                caller_bc_pc_of_call: 5, // the caller's Call op (resume target)
                callee_entry_bc_pc: 0,
                arg_slot_map: vec![2, 3], // caller bank slots holding the args
            },
        };
        // Caller has 12 bytecode ops and an 8-slot bank: bc_pc 5 < 12 and slots
        // 2,3 < 8, so the reconstruction can run.
        assert!(site.verify_against_caller(12, 8));
    }

    /// An inlined-frame site whose Call resume bc_pc is OUT of the caller code is
    /// rejected — resuming there would read past the bytecode (a wrong/garbage op).
    #[test]
    fn inlined_frame_site_rejects_out_of_range_resume_pc() {
        let site = InlinedDeoptSite {
            base: DeoptSite { native_off: 0, bc_pc: 99, reason: DeoptReason::ShapeMiss },
            frame: InlinedFrame {
                caller_bc_pc_of_call: 99, // past the end of a 12-op caller
                callee_entry_bc_pc: 0,
                arg_slot_map: vec![0],
            },
        };
        assert!(!site.verify_against_caller(12, 8));
    }

    /// An inlined-frame site recording an arg slot OUTSIDE the caller bank is
    /// rejected — the identity-map decode would not populate it, so the re-run Call
    /// would read a missing argument (the inlined-frame analogue of the
    /// out-of-bank-root UAF the SafepointMap verifier catches). This is the teeth:
    /// a SAME site that is in range against a bigger bank passes, proving the check
    /// is the range relation, not a constant rejection.
    #[test]
    fn inlined_frame_site_rejects_arg_slot_outside_caller_bank() {
        let site = InlinedDeoptSite {
            base: DeoptSite { native_off: 0, bc_pc: 5, reason: DeoptReason::NonNumber },
            frame: InlinedFrame {
                caller_bc_pc_of_call: 5,
                callee_entry_bc_pc: 0,
                arg_slot_map: vec![3, 9], // slot 9 is out of an 8-slot bank
            },
        };
        assert!(!site.verify_against_caller(12, 8));
        // The SAME site verifies against a bank large enough to hold slot 9.
        assert!(site.verify_against_caller(12, 10));
    }
}
