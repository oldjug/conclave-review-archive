//! T4 (Maglev-class) PHASE P1 — binary/compare TYPE-FEEDBACK VECTOR.
//!
//! This is the PROFILING substrate the speculative T4 tier needs and Conclave
//! lacked: a per-bytecode-slot, monotone (widen-only) type-hint lattice recorded
//! by the bytecode VM's arithmetic / comparison / call handlers, exposed to the
//! T4 lowering so representation selection (P2) and inlining (P3) can speculate on
//! the OBSERVED operand types instead of "decline unless provably numeric".
//!
//! ## What this phase does (and pointedly does NOT do)
//!
//! P1 RECORDS ONLY. It performs NO specialization, emits NO code, and changes NO
//! observable behavior — a feedback write is observationally invisible (it touches
//! a side table on the `BcFunction`, never a JS value). The A/B oracle therefore
//! stays byte-identical with `CV_FEEDBACK` on or off. P2 will *consume* this
//! vector; P5 will *persist* it. Keeping recording strictly separate from
//! specialization is the same discipline V8 uses: Ignition collects feedback into
//! the `FeedbackVector` while interpreting; Maglev/TurboFan later *read* it.
//!
//! ## V8 source modeled
//!
//! The lattice shape mirrors V8's **`BinaryOperationHint`** and
//! **`CompareOperationHint`** (`src/compiler/feedback-source.*`,
//! `src/objects/feedback-vector.h`). V8's binary hint lattice is, widening:
//!   `kNone ⊂ kSignedSmall ⊂ kSignedSmallInputs ⊂ kNumber ⊂ kNumberOrOddball
//!    ⊂ {kString | kBigInt | …} ⊂ kAny`
//! and its compare hint adds `kNumberOrBoolean`/`kInternalizedString`/`kReceiver`
//! flavors. The LOAD-BEARING property V8 relies on — and the one this module
//! enforces structurally and a unit test proves — is that the hint is a
//! **monotone JOIN (least-upper-bound)**: every observation can only *widen* the
//! recorded hint, NEVER narrow it. `BinaryOperationFeedback::Combine` in V8 is a
//! bitwise-OR over a flag lattice; we use an explicit total order whose `join` is
//! `max`, which is the same monotone semantics in a form that is trivially
//! checkable (`join(a,b) >= a && join(a,b) >= b`).
//!
//! V8 also STOPS refining a site once it is "generic"/megamorphic (the feedback
//! slot saturates at `kAny` and Ignition no longer pays to refine it). We mirror
//! that "monotone settle": once a slot reaches `Any` it is frozen and the VM skips
//! the per-op classification work, bounding the recording overhead (the phase
//! gate: <1% on loop.js). This is the analogue of V8 not re-profiling a
//! megamorphic IC.
//!
//! ## Why a separate side table (not on `Value`)
//!
//! Like `PropIc`, feedback is keyed by **bytecode instruction index** and stored
//! in a `RefCell<Vec<TypeFeedback>>` on the `BcFunction`, lazily sized to
//! `code.len()`. Only arith/compare/call op indices are ever written. This keeps
//! the JS value representation untouched (so it stays NaN-box-totally-encodable)
//! and lets the feedback warm across calls of the same function, exactly as the
//! property IC does.

use crate::interp::Value;

/// Whether the P1 binary/compare feedback vector is RECORDED by the VM.
///
/// DEFAULT-OFF (opt IN with `CV_FEEDBACK=1`), mirroring `CV_T3`/`CV_T4`. With the
/// flag off the VM's `record_*` calls are gated out entirely, so the default
/// build pays ZERO recording cost and is byte-identical to before this phase.
/// When on, the recording is still observationally invisible (it only mutates the
/// side table), so the A/B oracle is green either way — the oracle additionally
/// runs the corpus with recording FORCED on (via [`set_force_feedback`]) to prove
/// non-vacuity (the vector actually fills) without changing results.
pub fn feedback_enabled() -> bool {
    if force_feedback() {
        return true;
    }
    thread_local! {
        static ON: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
    }
    ON.with(|c| match c.get() {
        Some(v) => v,
        None => {
            let v = std::env::var("CV_FEEDBACK").as_deref() == Ok("1");
            c.set(Some(v));
            v
        }
    })
}

thread_local! {
    /// In-process force-on switch for the oracle / tests (mirrors `FORCED_TIER`):
    /// when set, [`feedback_enabled`] returns true regardless of the env. Lets the
    /// oracle exercise recording on the whole corpus to prove the vector fills,
    /// without a process-global env and without changing observable results.
    static FORCE_FEEDBACK: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Set (returns the prior value) the in-process force-feedback switch. Prefer the
/// [`FeedbackGuard`] scope guard so it is always restored.
pub fn set_force_feedback(v: bool) -> bool {
    FORCE_FEEDBACK.with(|c| {
        let prev = c.get();
        c.set(v);
        prev
    })
}

/// Whether feedback recording is force-enabled in-process (oracle/tests).
pub fn force_feedback() -> bool {
    FORCE_FEEDBACK.with(|c| c.get())
}

thread_local! {
    /// HONESTY GUARD — counts the number of feedback OBSERVATIONS the VM recorded
    /// (one per arith/compare/call op that actually classified an operand). Like
    /// `t2_exec_count`, this proves the recorder is NON-VACUOUS: a test/oracle
    /// resets it, runs a recordable snippet, and asserts it is > 0 (the vector
    /// truly filled, not a mis-wired no-op). Bumped only when recording is on, so
    /// the default build never touches it.
    static FEEDBACK_RECORD_COUNT: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

/// Current feedback-observation count (the honesty guard; see
/// [`reset_feedback_record_count`]).
pub fn feedback_record_count() -> u64 {
    FEEDBACK_RECORD_COUNT.with(|c| c.get())
}

/// Reset the feedback-observation honesty counter (oracle/tests call before a
/// recorded run, then assert `feedback_record_count() > 0`).
pub fn reset_feedback_record_count() {
    FEEDBACK_RECORD_COUNT.with(|c| c.set(0));
}

#[inline]
pub(crate) fn bump_feedback_record_count() {
    FEEDBACK_RECORD_COUNT.with(|c| c.set(c.get().wrapping_add(1)));
}

// ----------------------------------------------------------------------
// MUTATION HOOK (test-only) — proves the P1 feedback-on oracle leg is NON-VACUOUS.
//
// P1's safety claim is "recording is observationally INVISIBLE": the recorder
// must never touch a JS value. To prove the oracle would CATCH a recorder that
// violated that, this hook makes the VM's recording macro deliberately CLOBBER an
// operand register while recording (a recorder side effect on a JS value). With
// the hook set, the feedback-ON oracle leg MUST redden (results diverge from the
// recording-off tree-walk); with it unset (the production default) the oracle is
// green. There is NO env path — only the in-process setter (mirrors
// `set_force_wrong_fold` / `set_force_deopt_pc`). NEVER engaged in production.
// ----------------------------------------------------------------------
thread_local! {
    static FORCE_RECORD_CLOBBER: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Set (returns prior) the test-only recording-clobber mutation hook. When true,
/// the VM's feedback recorder overwrites the binary op's `rhs` register with a
/// bogus value AFTER classifying it — a forbidden side effect on a JS value that
/// the feedback-on oracle leg must catch. Prefer the [`RecordClobberGuard`].
pub fn set_force_record_clobber(v: bool) -> bool {
    FORCE_RECORD_CLOBBER.with(|c| {
        let prev = c.get();
        c.set(v);
        prev
    })
}

/// Whether the recording-clobber mutation hook is engaged (test-only).
#[inline]
pub fn force_record_clobber() -> bool {
    FORCE_RECORD_CLOBBER.with(|c| c.get())
}

/// RAII guard for [`set_force_record_clobber`].
pub struct RecordClobberGuard {
    prev: bool,
}
impl RecordClobberGuard {
    pub fn new(v: bool) -> Self {
        RecordClobberGuard {
            prev: set_force_record_clobber(v),
        }
    }
}
impl Drop for RecordClobberGuard {
    fn drop(&mut self) {
        set_force_record_clobber(self.prev);
    }
}

/// RAII scope guard for [`set_force_feedback`] — restores the prior value on drop.
pub struct FeedbackGuard {
    prev: bool,
}
impl FeedbackGuard {
    pub fn new(v: bool) -> Self {
        let prev = set_force_feedback(v);
        FeedbackGuard { prev }
    }
}
impl Drop for FeedbackGuard {
    fn drop(&mut self) {
        set_force_feedback(self.prev);
    }
}

// ======================================================================
// THE LATTICE — a monotone (widen-only) type hint per operand class.
//
// Encoded as an explicit TOTAL ORDER (a `u8` discriminant) whose `join` is `max`.
// The order is the V8 BinaryOperationHint widening chain, collapsed to the
// distinctions T4 representation selection actually uses:
//
//   None              — never observed (the bottom; a slot that never ran).
//   SignedSmall       — every operand seen was a small (Smi-range) integer: a
//                       safe-integer f64 in i32 range. V8 `kSignedSmall`. T4 may
//                       pick an unboxed Int32 representation with an overflow
//                       guard.
//   Number            — every operand seen was a JS Number (any f64, including
//                       non-integer / out-of-Smi-range / ±Inf / NaN). V8
//                       `kNumber`. T4 may pick unboxed Float64.
//   NumberOrOddball   — operands were Numbers OR "oddballs" (boolean / null /
//                       undefined), which ToNumber-coerce to a number in arith
//                       and compare. V8 `kNumberOrOddball`. T4 must insert the
//                       oddball→number conversion before unboxing.
//   String            — a String operand was seen (e.g. `"a" + b`, or a relational
//                       string compare). V8 `kString`. Not a numeric speculation
//                       target; recorded so T4 declines unboxing rather than
//                       guessing.
//   BigInt            — a BigInt operand was seen. V8 `kBigInt`. Distinct numeric
//                       tower; T4 declines f64 unboxing.
//   Any               — mixed / heap / object / symbol / anything else, OR a join
//                       of two incompatible non-bottom hints (e.g. String⊔BigInt).
//                       V8 `kAny` (megamorphic). The TOP: once a slot is Any it is
//                       FROZEN and the VM stops classifying it (the settle).
//
// MONOTONE INVARIANT (the load-bearing contract, V8 BinaryOperationFeedback):
//   join(a, b) >= a  AND  join(a, b) >= b   for all a, b   (i.e. join == max).
// A hint can only ever move UP this chain. `record_*` only ever assigns the join
// of the current hint and the observation, so the recorded hint is monotone by
// construction; `lattice_join_only_widens` (unit test) proves it for all pairs.
// ======================================================================

/// A monotone type hint for one operand position (or the joined hint for a binary
/// site). Ordered bottom→top; `join` is `max` over the discriminant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u8)]
pub enum TypeHint {
    /// Bottom — never observed.
    None = 0,
    /// Small (Smi-range) integer Number. V8 `kSignedSmall`.
    SignedSmall = 1,
    /// Any JS Number (f64). V8 `kNumber`.
    Number = 2,
    /// Number or boolean/null/undefined (ToNumber-coercible oddball). V8
    /// `kNumberOrOddball`.
    NumberOrOddball = 3,
    /// A String operand was seen. V8 `kString`.
    String = 4,
    /// A BigInt operand was seen. V8 `kBigInt`.
    BigInt = 5,
    /// Top — mixed / heap / incompatible. Megamorphic; recording settles here. V8
    /// `kAny`.
    Any = 6,
}

impl TypeHint {
    /// Classify a single runtime operand `Value` into its bottom-most hint — the
    /// narrowest hint that admits this value. Joining the per-operand hints of a
    /// binary op gives the site hint.
    ///
    /// Smi range follows V8: a Number is `SignedSmall` iff it is an integer in the
    /// signed-32-bit range with no negative-zero ambiguity (−0 is NOT a Smi in V8;
    /// it widens to `Number` so an Int32 speculation never has to special-case the
    /// −0 sign bit). Any other finite/non-finite Number is `Number`.
    #[inline]
    pub fn classify(v: &Value) -> TypeHint {
        match v {
            Value::Number(n) => {
                let n = *n;
                // Integer in i32 range, and not negative zero (V8 excludes −0 from
                // SignedSmall so an unboxed Int32 never loses the sign of zero).
                if n.fract() == 0.0
                    && n >= i32::MIN as f64
                    && n <= i32::MAX as f64
                    && !(n == 0.0 && n.is_sign_negative())
                {
                    TypeHint::SignedSmall
                } else {
                    TypeHint::Number
                }
            }
            // Oddballs ToNumber-coerce in arithmetic/compare (true→1, false→0,
            // null→0, undefined→NaN). V8 groups them under kNumberOrOddball.
            Value::Bool(_) | Value::Null | Value::Undefined => TypeHint::NumberOrOddball,
            Value::String(_) => TypeHint::String,
            Value::BigInt(_) => TypeHint::BigInt,
            // Objects, arrays, functions, symbols-as-strings, holes: not a numeric
            // speculation target. Top.
            _ => TypeHint::Any,
        }
    }

    /// The monotone JOIN (least upper bound) of two hints — `max` over the order.
    ///
    /// This is the whole correctness contract: a recorded hint is updated to
    /// `join(current, observed)`, so it only ever widens (`join >= both`). The
    /// order places the non-speculatable hints `String < BigInt < Any` ABOVE the
    /// numeric speculation band `SignedSmall < Number < NumberOrOddball`, so any
    /// observation of a String/BigInt/heap operand pushes the site out of the
    /// numeric band permanently — exactly the "decline f64 speculation" answer T4
    /// needs. We do NOT need V8's richer FLAG lattice for P1: T4 only ever ACTS on
    /// `is_numeric_speculatable()` (the numeric band); for the consumer,
    /// `String`/`BigInt`/`Any` are all equally "do not speculate", so collapsing
    /// their joins to `max` (e.g. `String ⊔ BigInt = BigInt`) is observationally
    /// equivalent to V8's generic state and keeps the lattice a trivially-checkable
    /// total order. (The distinction is retained only for diagnostics/persistence.)
    #[inline]
    pub fn join(self, other: TypeHint) -> TypeHint {
        if self >= other { self } else { other }
    }

    /// True once this hint has SETTLED at the top (`Any`) — V8's megamorphic
    /// state. The VM stops classifying a settled slot (the recording-overhead
    /// bound). A `None` slot has not settled; only `Any` is settled.
    #[inline]
    pub fn is_settled(self) -> bool {
        self == TypeHint::Any
    }

    /// True if this hint is in the NUMERIC speculation band — the range over which
    /// T4 (P2) may pick an unboxed representation. `SignedSmall`/`Number`/
    /// `NumberOrOddball` are speculatable; `None` (never ran) / `String` / `BigInt`
    /// / `Any` are not. Exposed for the T4 lowering.
    #[inline]
    pub fn is_numeric_speculatable(self) -> bool {
        matches!(
            self,
            TypeHint::SignedSmall | TypeHint::Number | TypeHint::NumberOrOddball
        )
    }
}

/// One feedback slot for one bytecode op. A binary arith/compare op records the
/// JOIN of its two operands' hints; a call op records its target monomorphism.
///
/// `Copy` + `repr` so the side-table `Vec<TypeFeedback>` is cheap and resets to
/// `INVALID` (all-`None`) by `vec![TypeFeedback::INVALID; len]` — exactly like
/// `PropIc::INVALID`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TypeFeedback {
    /// The monotone joined operand hint for a binary arith/compare site. For a
    /// call site this stays `None` (calls use the fields below).
    pub binop: TypeHint,
    /// CALL feedback — the module function index of the SINGLE observed call
    /// target, valid iff `call_mono` is true and `call_target != u32::MAX`. A
    /// second, different target sets `call_mono = false` (polymorphic → T4
    /// declines inlining), mirroring V8's `CALL_IC` monomorphic→polymorphic→
    /// megamorphic progression keyed by target.
    pub call_target: u32,
    /// True iff every call observed at this site went to the SAME `call_target`
    /// (monomorphic). Starts true on the first record; flips to false (and stays
    /// false — monotone) the first time a different target is seen. A site that
    /// never recorded a call has `call_target == u32::MAX` (then `call_mono` is
    /// meaningless; check the target sentinel first).
    pub call_mono: bool,
}

impl TypeFeedback {
    /// The empty (never-observed) feedback slot — analogous to `PropIc::INVALID`.
    pub const INVALID: TypeFeedback = TypeFeedback {
        binop: TypeHint::None,
        call_target: u32::MAX,
        call_mono: true,
    };

    /// Record a BINARY arith/compare observation: widen the site hint to the join
    /// of the current hint and the two operand classifications. MONOTONE — the
    /// hint only ever moves up. A settled (`Any`) site is left untouched (the VM
    /// should not even call this for a settled slot; the guard here makes it
    /// idempotent so a stray call cannot narrow).
    #[inline]
    pub fn record_binop(&mut self, lhs: &Value, rhs: &Value) {
        if self.binop.is_settled() {
            return; // settled at Any — frozen (the V8 megamorphic-settle).
        }
        let observed = TypeHint::classify(lhs).join(TypeHint::classify(rhs));
        self.binop = self.binop.join(observed);
        bump_feedback_record_count();
    }

    /// Record a UNARY arith observation (Neg / unary-plus / BitNot etc.): widen the
    /// site hint to the join of the current hint and the single operand.
    #[inline]
    pub fn record_unop(&mut self, operand: &Value) {
        if self.binop.is_settled() {
            return;
        }
        let observed = TypeHint::classify(operand);
        self.binop = self.binop.join(observed);
        bump_feedback_record_count();
    }

    /// Record a CALL observation against module function index `target`.
    /// MONOTONE: monomorphic (one target) → polymorphic (`call_mono=false`, never
    /// flips back). Mirrors V8's call-feedback target tracking. Once polymorphic
    /// the VM may stop recording (the target is already pinned false).
    #[inline]
    pub fn record_call(&mut self, target: u32) {
        bump_feedback_record_count();
        if self.call_target == u32::MAX {
            // First call observed at this site → monomorphic on `target`.
            self.call_target = target;
            self.call_mono = true;
        } else if self.call_target != target {
            // A second, different target → polymorphic. Monotone: stays false.
            self.call_mono = false;
        }
        // Same target again → no change (still monomorphic).
    }

    /// True if this site carries ANY observation worth exposing/persisting (a
    /// non-bottom binop hint, or a recorded call target). A slot that never ran is
    /// vacuous. Mirrors `PropIc::has_feedback`.
    #[inline]
    pub fn has_feedback(&self) -> bool {
        self.binop != TypeHint::None || self.call_target != u32::MAX
    }

    /// The binary/compare hint for the T4 lowering (P2 representation selection).
    #[inline]
    pub fn binop_hint(&self) -> TypeHint {
        self.binop
    }

    /// The monomorphic call target for the T4 inliner (P3), if this site is a
    /// recorded MONOMORPHIC call. `None` if the site never recorded a call or has
    /// gone polymorphic — in which case T4 declines inlining (correct + safe).
    #[inline]
    pub fn mono_call_target(&self) -> Option<u32> {
        if self.call_target != u32::MAX && self.call_mono {
            Some(self.call_target)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::interp::Value;
    use std::rc::Rc;

    fn num(n: f64) -> Value {
        Value::Number(n)
    }

    #[test]
    fn classify_maps_values_to_the_narrowest_hint() {
        assert_eq!(TypeHint::classify(&num(3.0)), TypeHint::SignedSmall);
        assert_eq!(TypeHint::classify(&num(-7.0)), TypeHint::SignedSmall);
        assert_eq!(TypeHint::classify(&num(0.0)), TypeHint::SignedSmall);
        // Out-of-i32-range integer, non-integer, and non-finite are Number.
        assert_eq!(TypeHint::classify(&num(3.5)), TypeHint::Number);
        assert_eq!(TypeHint::classify(&num(5e9)), TypeHint::Number);
        assert_eq!(TypeHint::classify(&num(f64::INFINITY)), TypeHint::Number);
        assert_eq!(TypeHint::classify(&num(f64::NAN)), TypeHint::Number);
        // −0 widens to Number (V8 excludes −0 from SignedSmall).
        assert_eq!(TypeHint::classify(&num(-0.0)), TypeHint::Number);
        // Oddballs.
        assert_eq!(TypeHint::classify(&Value::Bool(true)), TypeHint::NumberOrOddball);
        assert_eq!(TypeHint::classify(&Value::Null), TypeHint::NumberOrOddball);
        assert_eq!(TypeHint::classify(&Value::Undefined), TypeHint::NumberOrOddball);
        // String / BigInt.
        assert_eq!(TypeHint::classify(&Value::str("hi")), TypeHint::String);
        assert_eq!(
            TypeHint::classify(&Value::bigint(
                crate::interp::JsBigInt::parse_str("5").unwrap()
            )),
            TypeHint::BigInt
        );
        // Heap / object → Any.
        let obj = Value::Object(Rc::new(std::cell::RefCell::new(Default::default())));
        assert_eq!(TypeHint::classify(&obj), TypeHint::Any);
    }

    /// THE load-bearing invariant (V8 BinaryOperationFeedback monotone-OR): the
    /// join only ever WIDENS. Exhaustively over every ordered pair of hints,
    /// `join(a,b) >= a && join(a,b) >= b` and `join` is commutative and
    /// idempotent. This proves a recorded hint can never narrow.
    #[test]
    fn lattice_join_only_widens() {
        let all = [
            TypeHint::None,
            TypeHint::SignedSmall,
            TypeHint::Number,
            TypeHint::NumberOrOddball,
            TypeHint::String,
            TypeHint::BigInt,
            TypeHint::Any,
        ];
        for &a in &all {
            // Idempotent.
            assert_eq!(a.join(a), a);
            for &b in &all {
                let j = a.join(b);
                // Upper bound of both (only widens).
                assert!(j >= a, "join({a:?},{b:?})={j:?} must be >= {a:?}");
                assert!(j >= b, "join({a:?},{b:?})={j:?} must be >= {b:?}");
                // Commutative.
                assert_eq!(a.join(b), b.join(a));
                // Least: no hint strictly between max(a,b) and j (j IS the max).
                assert_eq!(j, a.max(b));
            }
        }
    }

    /// `record_binop` is monotone: a sequence of observations only widens the
    /// site hint, and once it reaches `Any` it FREEZES (the settle). This is the
    /// VM-handler-level guarantee.
    #[test]
    fn record_binop_is_monotone_and_settles() {
        let mut fb = TypeFeedback::INVALID;
        assert_eq!(fb.binop, TypeHint::None);
        // Small ints → SignedSmall.
        fb.record_binop(&num(1.0), &num(2.0));
        assert_eq!(fb.binop, TypeHint::SignedSmall);
        // A float widens to Number.
        fb.record_binop(&num(1.5), &num(2.0));
        assert_eq!(fb.binop, TypeHint::Number);
        // Going back to small ints does NOT narrow.
        fb.record_binop(&num(1.0), &num(2.0));
        assert_eq!(fb.binop, TypeHint::Number);
        // An oddball widens to NumberOrOddball.
        fb.record_binop(&num(1.0), &Value::Bool(true));
        assert_eq!(fb.binop, TypeHint::NumberOrOddball);
        // A heap operand saturates to Any.
        let obj = Value::Object(Rc::new(std::cell::RefCell::new(Default::default())));
        fb.record_binop(&num(1.0), &obj);
        assert_eq!(fb.binop, TypeHint::Any);
        assert!(fb.binop.is_settled());
        // Once settled, NOTHING can move it — not even a "narrower" all-int obs.
        fb.record_binop(&num(1.0), &num(2.0));
        assert_eq!(fb.binop, TypeHint::Any);
        assert!(!fb.binop.is_numeric_speculatable());
    }

    /// String and BigInt both land ABOVE the numeric band (never speculatable);
    /// neither admits an unboxed-f64 speculation. The exact widen target of a
    /// String→BigInt sequence is `max` (BigInt) — what matters is that it stays
    /// out of the numeric band (the consumer treats String/BigInt/Any identically
    /// as "do not speculate").
    #[test]
    fn string_and_bigint_are_not_speculatable() {
        let mut fb = TypeFeedback::INVALID;
        fb.record_binop(&Value::str("a"), &Value::str("b"));
        assert_eq!(fb.binop, TypeHint::String);
        assert!(!fb.binop.is_numeric_speculatable());
        // Now a BigInt observation: monotone widen to max(String, BigInt)=BigInt,
        // still firmly out of the numeric speculation band.
        fb.record_binop(
            &Value::bigint(crate::interp::JsBigInt::parse_str("1").unwrap()),
            &Value::bigint(crate::interp::JsBigInt::parse_str("2").unwrap()),
        );
        assert_eq!(fb.binop, TypeHint::BigInt);
        assert!(!fb.binop.is_numeric_speculatable());
        // A heap operand DOES saturate to Any (the true top).
        let obj = Value::Object(Rc::new(std::cell::RefCell::new(Default::default())));
        fb.record_binop(&Value::str("a"), &obj);
        assert_eq!(fb.binop, TypeHint::Any);
    }

    #[test]
    fn numeric_band_is_speculatable() {
        for h in [
            TypeHint::SignedSmall,
            TypeHint::Number,
            TypeHint::NumberOrOddball,
        ] {
            assert!(h.is_numeric_speculatable(), "{h:?} should be speculatable");
        }
        for h in [TypeHint::None, TypeHint::String, TypeHint::BigInt, TypeHint::Any] {
            assert!(!h.is_numeric_speculatable(), "{h:?} must NOT be speculatable");
        }
    }

    /// Call feedback: monomorphic → polymorphic is monotone (never flips back),
    /// and the mono target is only exposed while monomorphic.
    #[test]
    fn call_feedback_tracks_monomorphism_monotonically() {
        let mut fb = TypeFeedback::INVALID;
        assert_eq!(fb.mono_call_target(), None); // never called
        fb.record_call(7);
        assert_eq!(fb.mono_call_target(), Some(7)); // monomorphic on 7
        fb.record_call(7); // same target — still mono
        assert_eq!(fb.mono_call_target(), Some(7));
        fb.record_call(9); // different target — polymorphic
        assert_eq!(fb.mono_call_target(), None);
        fb.record_call(7); // back to 7 — does NOT re-monomorphize (monotone)
        assert_eq!(fb.mono_call_target(), None);
        assert!(!fb.call_mono);
    }

    #[test]
    fn has_feedback_distinguishes_a_never_run_slot() {
        let fb = TypeFeedback::INVALID;
        assert!(!fb.has_feedback());
        let mut fb2 = TypeFeedback::INVALID;
        fb2.record_binop(&num(1.0), &num(2.0));
        assert!(fb2.has_feedback());
        let mut fb3 = TypeFeedback::INVALID;
        fb3.record_call(3);
        assert!(fb3.has_feedback());
    }

    #[test]
    fn record_unop_widens_like_binop() {
        let mut fb = TypeFeedback::INVALID;
        fb.record_unop(&num(5.0));
        assert_eq!(fb.binop, TypeHint::SignedSmall);
        fb.record_unop(&num(5.5));
        assert_eq!(fb.binop, TypeHint::Number);
        fb.record_unop(&Value::str("x"));
        assert_eq!(fb.binop, TypeHint::String);
    }

    /// The feedback guard restores the prior force state on drop (scope safety).
    #[test]
    fn feedback_guard_restores_prior_force_state() {
        let before = force_feedback();
        {
            let _g = FeedbackGuard::new(true);
            assert!(force_feedback());
            assert!(feedback_enabled());
        }
        assert_eq!(force_feedback(), before);
    }
}
