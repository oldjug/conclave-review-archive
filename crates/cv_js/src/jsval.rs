//! `JsVal(u64)` — a NaN-boxed single-word representation of a JS value.
//!
//! # Status: M3.6 Phase-0 (PURE ADDITIVE / BEHAVIOR-NEUTRAL)
//!
//! This module defines the *type* and its exhaustive correctness tests. It does
//! **not** replace [`crate::interp::Value`] anywhere — `Value` remains the live
//! engine representation. Nothing outside this file and its tests constructs or
//! consumes a `JsVal` yet. The point of Phase-0 is to land a *provably correct*
//! encoding (round-trip bijection + NaN canonicalization + no tag/double
//! collision) so the later migration phases can flip onto it with confidence.
//!
//! # Why NaN-boxing
//!
//! The data (M4.2b) says the only path to a real JS speedup is a JIT that emits
//! inlined, *unboxed* numeric fast paths (numbers kept in xmm registers). That
//! requires a value the JIT can test/branch/unbox in registers cheaply — a flat
//! `u64`, not a 32-byte tagged enum. NaN-boxing is the standard technique
//! (V8 Smi/HeapObject tagging on 32-bit + pointer-compression elsewhere; JSC's
//! "JSValue" 64-bit encoding; SpiderMonkey's `jsval`). We use the **JSC-style
//! "doubles are themselves"** layout because it makes the number lane (the hot
//! lane for arithmetic) a zero-cost reinterpret and leaves the entire quiet-NaN
//! region free for tagged immediates and pointers.
//!
//! # The bit scheme
//!
//! A `JsVal` is a `u64` interpreted as one of two things:
//!
//! 1. **A double.** Any IEEE-754 `f64` that is *not* a NaN is stored as its own
//!    bits and decodes straight back to [`Value::Number`]. Exactly one NaN — the
//!    *canonical* quiet NaN `0x7FF8_0000_0000_0000` — also decodes to a
//!    `Number` (whose value is NaN). Every other NaN bit pattern is forbidden in
//!    a `JsVal`: [`JsVal::number`] **canonicalizes** all NaNs to that one
//!    pattern on the way in (see "Canonicalization" below).
//!
//! 2. **A boxed value** (everything that is not a double). These live in the
//!    *signalling-looking* quiet-NaN space whose top 13 bits are all 1:
//!
//!    ```text
//!     63   62        52 51  50 49 48 47                                   0
//!    ┌───┬─────────────┬───┬──────┬──────────────────────────────────────┐
//!    │ S │  exponent   │ Q │ TAG  │            48-bit payload             │
//!    │ 1 │ 1 1 … 1 (11)│ 1 │ 3 bit│   pointer / int32 / singleton id      │
//!    └───┴─────────────┴───┴──────┴──────────────────────────────────────┘
//!    ```
//!
//!    The discriminator is `bits & QNAN_MASK == QNAN_BITS`, where
//!    `QNAN_BITS = 0xFFF8_0000_0000_0000` (sign=1, exp=all-ones, quiet=1). The
//!    *canonical* Number-NaN has sign=0, so it is **disjoint** from the boxed
//!    space — that is the property that makes the encoding collision-free.
//!
//!    Within the boxed space a 3-bit `TAG` (bits 48..=50) selects the lane and a
//!    48-bit payload carries the data. On x86-64 / AArch64 user-space pointers
//!    are 48-bit canonical (high 16 bits zero — verified on this platform), so a
//!    heap pointer fits in the payload losslessly.
//!
//! # Tags
//!
//! | tag | name        | payload                                             |
//! |-----|-------------|-----------------------------------------------------|
//! | 0   | `Object`    | `Rc<RefCell<HashMap<String,Value>>>` thin pointer   |
//! | 1   | `Array`     | `Rc<RefCell<Vec<Value>>>` thin pointer              |
//! | 2   | `Function`  | `Rc<FunctionValue>` thin pointer                    |
//! | 3   | `Native`    | `Rc<NativeFn>` thin pointer                         |
//! | 4   | `BcClosure` | `Rc<BcClosure>` thin pointer                        |
//! | 5   | `Int32`     | i32 in low 32 bits (a fast small-int lane for JIT)  |
//! | 6   | `Singleton` | small id: Undefined / Null / false / true / Hole   |
//! | 7   | `StrBig`    | String (`Rc<JsString>`) / BigInt (`Rc<JsBigInt>`)   |
//!
//! Tag 7 carries the two formerly-inline primitives, now re-homed behind thin
//! `Rc` pointers: `Value::BigInt(Rc<JsBigInt>)` (Phase 1) and
//! `Value::String(JsStr)` where `JsStr` wraps a thin `Rc<JsString>` (Phase 1b).
//! A 1-bit discriminator inside the payload ([`STRBIG_IS_STRING`]) selects the
//! kind. **As of Phase 1b `JsVal` is TOTAL**: [`JsVal::try_from_value`] succeeds
//! for *every* `Value` variant and the round-trip tests cover all of them.
//!
//! # Canonicalization (the #1 silent-corruption hazard)
//!
//! A *computed* NaN (e.g. `0.0/0.0`, `Inf - Inf`, or an arbitrary payload-NaN)
//! can have *any* mantissa and the sign bit set — which would alias the boxed
//! tag space and be mis-decoded as a pointer/int/singleton. [`JsVal::number`]
//! therefore replaces *any* NaN with the single [`CANONICAL_NAN`] before
//! storing. After that, the only NaN a `JsVal` can ever hold is the canonical
//! one, and it lives at sign=0 (outside the boxed space). This is tested
//! exhaustively.
//!
//! # Pointer / GC ownership contract (for the migration)
//!
//! The boxed pointer lanes store `Rc::as_ptr(&rc) as usize` — the *same* idiom
//! the engine already uses (`interp.rs`: GC identity set keys on
//! `Rc::as_ptr(...) as usize`; equality uses `Rc::ptr_eq`). A `JsVal` is a
//! **borrowed/weak** handle in Phase-0 semantics: boxing does **not** bump the
//! refcount and unboxing reconstructs the `Rc` via [`Rc::from_raw`] +
//! [`std::mem::forget`] of a transient clone so the original `Rc`'s refcount is
//! preserved exactly. The decoded `Rc` round-trips to the **identical** pointer,
//! so GC marking and inline caches that key on the pointer keep working.
//!
//! SAFETY INVARIANT for the future migration: a `JsVal` carrying a pointer tag is
//! only valid while the originating `Rc` (or another clone of it) is alive. The
//! register bank / value stack that holds `JsVal`s must therefore either (a) hold
//! a parallel owning `Rc` for the lifetime, or (b) the migration must make
//! boxing take ownership (refcount +1) and unboxing/`drop` release it. Phase-0
//! uses the borrowed model purely so the round-trip tests can observe pointer
//! identity without perturbing refcounts; the migration plan calls out the
//! owning model as a required step before `JsVal` becomes a *stored* type.

use std::cell::RefCell;
use std::rc::Rc;

// The engine stores object/property maps in an insertion-ordered map, aliased
// as `HashMap` throughout `interp.rs` (`use crate::ordered::OrderedMap as
// HashMap`). We mirror that alias so the pointer-lane types line up exactly with
// `Value`'s heap payloads.
use crate::interp::{BcClosure, FunctionValue, JsBigInt, JsStr, JsString, NativeFn, Value};
use crate::ordered::OrderedMap as HashMap;

/// Mask isolating sign + exponent(11) + quiet bit = the top 13 bits.
const QNAN_MASK: u64 = 0xFFF8_0000_0000_0000;
/// A boxed (non-double) `JsVal` has exactly these top 13 bits set.
const QNAN_BITS: u64 = 0xFFF8_0000_0000_0000;

/// The single canonical quiet NaN that decodes to `Value::Number(NaN)`.
/// Note sign bit = 0, so it is **outside** the boxed space (`QNAN_BITS` has
/// sign = 1) — this disjointness is what makes the encoding collision-free.
pub const CANONICAL_NAN: u64 = 0x7FF8_0000_0000_0000;

/// Tag occupies bits 48..=50 (3 bits), payload occupies bits 0..=47 (48 bits).
const TAG_SHIFT: u64 = 48;
const TAG_MASK: u64 = 0x7; // 3 bits
const PAYLOAD_MASK: u64 = 0x0000_FFFF_FFFF_FFFF; // low 48 bits

// ---- Tags (3-bit) ----
const TAG_OBJECT: u64 = 0;
const TAG_ARRAY: u64 = 1;
const TAG_FUNCTION: u64 = 2;
const TAG_NATIVE: u64 = 3;
const TAG_BCCLOSURE: u64 = 4;
const TAG_INT32: u64 = 5;
const TAG_SINGLETON: u64 = 6;
/// String / BigInt heap-pointer lane. Both `Value::String(JsStr)` and
/// `Value::BigInt(Rc<JsBigInt>)` are re-homed behind **thin** `Rc` pointers, so
/// both box losslessly here exactly like the other heap lanes.
///
/// `Rc<JsBigInt>` is thin because `JsBigInt` is `Sized` (Phase 1).
///
/// `Value::String` was `Rc<str>` — a *fat* pointer (`*const str` carries a
/// 64-bit length alongside the 64-bit data pointer = 16 bytes) that could not
/// fit the 48-bit payload, because stable Rust cannot recover a `str` slice's
/// length from its data pointer alone (the length lives only in the fat pointer,
/// not in the `RcInner<str>` allocation). Phase 1b fixes this with a thin string
/// handle: `JsStr` wraps `Rc<JsString>`, and `JsString` wraps a `Box<str>` (a
/// `Sized` field), so `Rc<JsString>` is a **thin** 8-byte pointer — the length
/// is recovered on deref by reading the `Box<str>` out of the heap allocation.
/// `Value::String` is therefore fully `JsVal`-representable now.
///
/// Within this tag a payload discriminator bit selects the kind:
/// [`STRBIG_IS_STRING`] bit set → String, clear → BigInt.
const TAG_STRBIG: u64 = 7;

/// Discriminator bit inside the [`TAG_STRBIG`] payload: set → `String`,
/// clear → `BigInt`. Placed at bit 47, the highest payload bit. User-space heap
/// pointers on x86-64 / AArch64 are < 2^47 (canonical, bit 47 == bit 63 == 0 for
/// user addresses), so this bit is always 0 in a real `Rc` data pointer and is
/// free to use as a tag. (Verified by `ptr_box`'s payload-fits assertion, which
/// already requires the whole pointer in the low 48 bits.)
const STRBIG_IS_STRING: u64 = 1 << 47;

// ---- Singleton payload ids (under TAG_SINGLETON) ----
const SINGLE_UNDEFINED: u64 = 0;
const SINGLE_NULL: u64 = 1;
const SINGLE_FALSE: u64 = 2;
const SINGLE_TRUE: u64 = 3;
const SINGLE_HOLE: u64 = 4;

/// A NaN-boxed JS value: a single 64-bit word.
///
/// `#[repr(transparent)]` so it is layout-identical to a `u64` — the JIT can
/// load/store/test it as a raw register value.
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct JsVal(pub u64);

impl JsVal {
    // ------------------------------------------------------------------
    // Raw construction helpers
    // ------------------------------------------------------------------

    /// Construct a boxed (non-double) value from a tag + 48-bit payload.
    #[inline]
    const fn boxed(tag: u64, payload: u64) -> JsVal {
        debug_assert!(tag <= TAG_MASK);
        debug_assert!(payload <= PAYLOAD_MASK);
        JsVal(QNAN_BITS | (tag << TAG_SHIFT) | (payload & PAYLOAD_MASK))
    }

    /// The 3-bit tag of a boxed value. Only meaningful when [`Self::is_boxed`].
    #[inline]
    const fn tag(self) -> u64 {
        (self.0 >> TAG_SHIFT) & TAG_MASK
    }

    /// The 48-bit payload of a boxed value.
    #[inline]
    const fn payload(self) -> u64 {
        self.0 & PAYLOAD_MASK
    }

    /// True if this word lies in the boxed (quiet-NaN) space — i.e. it is *not*
    /// a plain double.
    #[inline]
    pub const fn is_boxed(self) -> bool {
        (self.0 & QNAN_MASK) == QNAN_BITS
    }

    // ------------------------------------------------------------------
    // Number lane
    // ------------------------------------------------------------------

    /// Box an `f64`. **Canonicalizes every NaN** to [`CANONICAL_NAN`] so a
    /// computed NaN can never alias the boxed tag space.
    #[inline]
    pub fn number(n: f64) -> JsVal {
        if n.is_nan() {
            JsVal(CANONICAL_NAN)
        } else {
            JsVal(n.to_bits())
        }
    }

    /// True if this is a `Number` (a plain double, including the canonical NaN).
    #[inline]
    pub const fn is_number(self) -> bool {
        !self.is_boxed()
    }

    /// Decode the `f64` if this is a `Number`. (Int32 lane is *not* a double;
    /// use [`Self::as_int32`] for that — see also [`Self::to_f64`].)
    #[inline]
    pub fn as_f64(self) -> Option<f64> {
        if self.is_number() {
            Some(f64::from_bits(self.0))
        } else {
            None
        }
    }

    // ------------------------------------------------------------------
    // Int32 lane (a fast small-int channel the JIT can branch on)
    // ------------------------------------------------------------------

    /// Box an `i32` in the dedicated small-int lane.
    #[inline]
    pub fn int32(n: i32) -> JsVal {
        // Zero-extend the i32 bit pattern into the low 32 bits of the payload.
        JsVal::boxed(TAG_INT32, (n as u32) as u64)
    }

    /// True if this is an `Int32`.
    #[inline]
    pub fn is_int32(self) -> bool {
        self.is_boxed() && self.tag() == TAG_INT32
    }

    /// Decode the `i32` if this is an `Int32`.
    #[inline]
    pub fn as_int32(self) -> Option<i32> {
        if self.is_int32() {
            Some(self.payload() as u32 as i32)
        } else {
            None
        }
    }

    /// The numeric value as an `f64` whether stored as a double *or* an int32.
    /// (Convenience for arithmetic that accepts either numeric lane.)
    #[inline]
    pub fn to_f64(self) -> Option<f64> {
        if let Some(n) = self.as_f64() {
            Some(n)
        } else {
            self.as_int32().map(|i| i as f64)
        }
    }

    // ------------------------------------------------------------------
    // Singletons
    // ------------------------------------------------------------------

    #[inline]
    pub const fn undefined() -> JsVal {
        JsVal::boxed(TAG_SINGLETON, SINGLE_UNDEFINED)
    }
    #[inline]
    pub const fn null() -> JsVal {
        JsVal::boxed(TAG_SINGLETON, SINGLE_NULL)
    }
    #[inline]
    pub const fn hole() -> JsVal {
        JsVal::boxed(TAG_SINGLETON, SINGLE_HOLE)
    }
    #[inline]
    pub const fn boolean(b: bool) -> JsVal {
        JsVal::boxed(TAG_SINGLETON, if b { SINGLE_TRUE } else { SINGLE_FALSE })
    }

    #[inline]
    pub fn is_undefined(self) -> bool {
        self.0 == JsVal::undefined().0
    }
    #[inline]
    pub fn is_null(self) -> bool {
        self.0 == JsVal::null().0
    }
    #[inline]
    pub fn is_hole(self) -> bool {
        self.0 == JsVal::hole().0
    }
    #[inline]
    pub fn is_bool(self) -> bool {
        self.0 == JsVal::boolean(true).0 || self.0 == JsVal::boolean(false).0
    }
    #[inline]
    pub fn as_bool(self) -> Option<bool> {
        if self.0 == JsVal::boolean(true).0 {
            Some(true)
        } else if self.0 == JsVal::boolean(false).0 {
            Some(false)
        } else {
            None
        }
    }

    // ------------------------------------------------------------------
    // Pointer lanes
    //
    // Boxing stores `Rc::as_ptr(...) as usize` WITHOUT bumping the refcount
    // (borrowed handle). Unboxing reconstructs the `Rc` via `Rc::from_raw`
    // and immediately `forget`s a clone, so the original refcount is preserved
    // exactly and the returned `Rc` round-trips to the same pointer.
    // ------------------------------------------------------------------

    #[inline]
    fn ptr_box<T>(tag: u64, rc: &Rc<T>) -> JsVal {
        let p = Rc::as_ptr(rc) as usize as u64;
        debug_assert_eq!(
            p & !PAYLOAD_MASK,
            0,
            "heap pointer exceeds 48 bits — NaN-box payload cannot hold it"
        );
        JsVal::boxed(tag, p)
    }

    /// Reconstruct an owned `Rc<T>` from a pointer-lane payload while preserving
    /// the original strong count.
    ///
    /// # Safety
    /// The caller must guarantee `self` was produced by [`Self::ptr_box`] with a
    /// pointer obtained from a live `Rc<T>` of the *same* `T`, and that some
    /// `Rc<T>` clone is still alive (so the pointee is valid).
    #[inline]
    unsafe fn ptr_unbox<T>(self) -> Rc<T> {
        let p = self.payload() as usize as *const T;
        // Reconstitute ownership of one strong ref…
        let rc = unsafe { Rc::from_raw(p) };
        // …then hand back a clone (count +1) and leak the reconstructed one
        // (count -1 cancelled) so the net strong count is unchanged.
        let out = rc.clone();
        std::mem::forget(rc);
        out
    }

    #[inline]
    pub fn object(rc: &Rc<RefCell<HashMap<String, Value>>>) -> JsVal {
        JsVal::ptr_box(TAG_OBJECT, rc)
    }
    #[inline]
    pub fn array(rc: &Rc<RefCell<Vec<Value>>>) -> JsVal {
        JsVal::ptr_box(TAG_ARRAY, rc)
    }
    #[inline]
    pub fn function(rc: &Rc<FunctionValue>) -> JsVal {
        JsVal::ptr_box(TAG_FUNCTION, rc)
    }
    #[inline]
    pub fn native(rc: &Rc<NativeFn>) -> JsVal {
        JsVal::ptr_box(TAG_NATIVE, rc)
    }
    #[inline]
    pub fn bcclosure(rc: &Rc<BcClosure>) -> JsVal {
        JsVal::ptr_box(TAG_BCCLOSURE, rc)
    }

    /// Box a `Rc<JsBigInt>` in the [`TAG_STRBIG`] lane (discriminator bit clear).
    /// `Rc<JsBigInt>` is a thin pointer, so this is a lossless pointer-lane box
    /// identical in mechanics to the other heap lanes (borrowed handle: refcount
    /// untouched; round-trips to the same pointer).
    #[inline]
    pub fn bigint(rc: &Rc<JsBigInt>) -> JsVal {
        let p = Rc::as_ptr(rc) as usize as u64;
        debug_assert_eq!(
            p & !PAYLOAD_MASK,
            0,
            "heap pointer exceeds 48 bits — NaN-box payload cannot hold it"
        );
        debug_assert_eq!(
            p & STRBIG_IS_STRING,
            0,
            "BigInt pointer collides with the String discriminator bit"
        );
        // Discriminator bit stays 0 → BigInt.
        JsVal::boxed(TAG_STRBIG, p)
    }

    /// True if this is a `BigInt` (TAG_STRBIG with the String bit clear).
    #[inline]
    pub fn is_bigint(self) -> bool {
        self.is_boxed() && self.tag() == TAG_STRBIG && (self.payload() & STRBIG_IS_STRING) == 0
    }

    /// Box a `Rc<JsString>` in the [`TAG_STRBIG`] lane with the discriminator bit
    /// **set** (→ String). `Rc<JsString>` is a thin pointer (the bytes live in a
    /// `Box<str>` inside the allocation), so this is a lossless pointer-lane box
    /// identical in mechanics to the other heap lanes (borrowed handle: refcount
    /// untouched; round-trips to the same pointer).
    #[inline]
    pub fn string(rc: &Rc<JsString>) -> JsVal {
        let p = Rc::as_ptr(rc) as usize as u64;
        debug_assert_eq!(
            p & !PAYLOAD_MASK,
            0,
            "heap pointer exceeds 48 bits — NaN-box payload cannot hold it"
        );
        debug_assert_eq!(
            p & STRBIG_IS_STRING,
            0,
            "String pointer already has the discriminator bit set (>2^47 address)"
        );
        // Set the discriminator bit → String.
        JsVal::boxed(TAG_STRBIG, p | STRBIG_IS_STRING)
    }

    /// True if this is a `String` (TAG_STRBIG with the String bit set).
    #[inline]
    pub fn is_string(self) -> bool {
        self.is_boxed() && self.tag() == TAG_STRBIG && (self.payload() & STRBIG_IS_STRING) != 0
    }

    /// Reconstruct the `Rc<JsString>` if this is a `String`.
    ///
    /// # Safety
    /// `self` must be a `String` `JsVal` whose pointee `Rc` is still live.
    #[inline]
    pub unsafe fn as_string(self) -> Option<Rc<JsString>> {
        if self.is_string() {
            // Mask off the discriminator bit to recover the canonical pointer.
            let p = (self.payload() & !STRBIG_IS_STRING) as usize as *const JsString;
            let rc = unsafe { Rc::from_raw(p) };
            let out = rc.clone();
            std::mem::forget(rc);
            Some(out)
        } else {
            None
        }
    }

    /// Reconstruct the `Rc<JsBigInt>` if this is a `BigInt`.
    ///
    /// # Safety
    /// `self` must be a `BigInt` `JsVal` whose pointee `Rc` is still live.
    #[inline]
    pub unsafe fn as_bigint(self) -> Option<Rc<JsBigInt>> {
        if self.is_bigint() {
            // Mask off the (always-zero for BigInt) discriminator bit before
            // reconstructing the pointer — defensive; it's already clear.
            let p = (self.payload() & !STRBIG_IS_STRING) as usize as *const JsBigInt;
            let rc = unsafe { Rc::from_raw(p) };
            let out = rc.clone();
            std::mem::forget(rc);
            Some(out)
        } else {
            None
        }
    }

    #[inline]
    pub fn is_object(self) -> bool {
        self.is_boxed() && self.tag() == TAG_OBJECT
    }
    #[inline]
    pub fn is_array(self) -> bool {
        self.is_boxed() && self.tag() == TAG_ARRAY
    }
    #[inline]
    pub fn is_function(self) -> bool {
        self.is_boxed() && self.tag() == TAG_FUNCTION
    }
    #[inline]
    pub fn is_native(self) -> bool {
        self.is_boxed() && self.tag() == TAG_NATIVE
    }
    #[inline]
    pub fn is_bcclosure(self) -> bool {
        self.is_boxed() && self.tag() == TAG_BCCLOSURE
    }
    /// True for any of the heap (pointer) lanes: the five object lanes plus the
    /// `TAG_STRBIG` BigInt lane.
    #[inline]
    pub fn is_pointer(self) -> bool {
        self.is_boxed() && (self.tag() <= TAG_BCCLOSURE || self.tag() == TAG_STRBIG)
    }

    /// The raw heap pointer (as `usize`) for any pointer lane — the value the
    /// GC identity set / inline caches key on. `None` for non-pointer lanes.
    /// For the `TAG_STRBIG` lane the discriminator bit is masked off so the
    /// result is the canonical `Rc` data pointer.
    #[inline]
    pub fn as_ptr_usize(self) -> Option<usize> {
        if self.is_boxed() && self.tag() <= TAG_BCCLOSURE {
            Some(self.payload() as usize)
        } else if self.is_boxed() && self.tag() == TAG_STRBIG {
            Some((self.payload() & !STRBIG_IS_STRING) as usize)
        } else {
            None
        }
    }

    /// # Safety
    /// `self` must be an `Object` `JsVal` whose pointee `Rc` is still live.
    #[inline]
    pub unsafe fn as_object(self) -> Option<Rc<RefCell<HashMap<String, Value>>>> {
        if self.is_object() {
            Some(unsafe { self.ptr_unbox() })
        } else {
            None
        }
    }
    /// # Safety
    /// `self` must be an `Array` `JsVal` whose pointee `Rc` is still live.
    #[inline]
    pub unsafe fn as_array(self) -> Option<Rc<RefCell<Vec<Value>>>> {
        if self.is_array() {
            Some(unsafe { self.ptr_unbox() })
        } else {
            None
        }
    }
    /// # Safety
    /// `self` must be a `Function` `JsVal` whose pointee `Rc` is still live.
    #[inline]
    pub unsafe fn as_function(self) -> Option<Rc<FunctionValue>> {
        if self.is_function() {
            Some(unsafe { self.ptr_unbox() })
        } else {
            None
        }
    }
    /// # Safety
    /// `self` must be a `NativeFunction` `JsVal` whose pointee `Rc` is still live.
    #[inline]
    pub unsafe fn as_native(self) -> Option<Rc<NativeFn>> {
        if self.is_native() {
            Some(unsafe { self.ptr_unbox() })
        } else {
            None
        }
    }
    /// # Safety
    /// `self` must be a `BcClosure` `JsVal` whose pointee `Rc` is still live.
    #[inline]
    pub unsafe fn as_bcclosure(self) -> Option<Rc<BcClosure>> {
        if self.is_bcclosure() {
            Some(unsafe { self.ptr_unbox() })
        } else {
            None
        }
    }

    // ------------------------------------------------------------------
    // Value <-> JsVal bridge
    // ------------------------------------------------------------------

    /// Encode a [`Value`] as a `JsVal`.
    ///
    /// **TOTAL as of Phase 1b**: every `Value` variant boxes losslessly and this
    /// never returns `None` for any input. Both formerly-inline primitives now
    /// ride the [`TAG_STRBIG`] lane behind thin `Rc` pointers — [`Value::BigInt`]
    /// is `Rc<JsBigInt>` (discriminator clear) and [`Value::String`] is `JsStr`
    /// wrapping `Rc<JsString>` (discriminator set). The `Option` return is kept
    /// for API stability (callers can still treat it as fallible), but no current
    /// variant produces `None`.
    pub fn try_from_value(v: &Value) -> Option<JsVal> {
        Some(match v {
            Value::Undefined => JsVal::undefined(),
            Value::Null => JsVal::null(),
            Value::Hole => JsVal::hole(),
            Value::Bool(b) => JsVal::boolean(*b),
            Value::Number(n) => JsVal::number(*n),
            Value::Object(rc) => JsVal::object(rc),
            Value::Array(rc) => JsVal::array(rc),
            Value::Function(rc) => JsVal::function(rc),
            Value::NativeFunction(rc) => JsVal::native(rc),
            Value::BcClosure(rc) => JsVal::bcclosure(rc),
            // BigInt is a thin `Rc<JsBigInt>` → fully boxable (discriminator clear).
            Value::BigInt(rc) => JsVal::bigint(rc),
            // String is now a thin `JsStr`/`Rc<JsString>` → fully boxable
            // (discriminator set). JsVal is TOTAL.
            Value::String(s) => JsVal::string(s.as_rc()),
        })
    }

    /// Decode a `JsVal` back to a [`Value`].
    ///
    /// # Safety
    /// If `self` carries a pointer lane, the originating `Rc` (or a clone) must
    /// still be alive. For non-pointer lanes this is always safe; the `unsafe`
    /// is required only because the pointer lanes deref a stored address. The
    /// reconstructed `Rc` round-trips to the identical pointer (preserving GC /
    /// IC identity) and leaves the strong count unchanged net.
    pub unsafe fn to_value(self) -> Value {
        if self.is_number() {
            return Value::Number(f64::from_bits(self.0));
        }
        match self.tag() {
            TAG_OBJECT => Value::Object(unsafe { self.ptr_unbox() }),
            TAG_ARRAY => Value::Array(unsafe { self.ptr_unbox() }),
            TAG_FUNCTION => Value::Function(unsafe { self.ptr_unbox() }),
            TAG_NATIVE => Value::NativeFunction(unsafe { self.ptr_unbox() }),
            TAG_BCCLOSURE => Value::BcClosure(unsafe { self.ptr_unbox() }),
            TAG_INT32 => Value::Number(self.payload() as u32 as i32 as f64),
            TAG_SINGLETON => match self.payload() {
                SINGLE_UNDEFINED => Value::Undefined,
                SINGLE_NULL => Value::Null,
                SINGLE_FALSE => Value::Bool(false),
                SINGLE_TRUE => Value::Bool(true),
                SINGLE_HOLE => Value::Hole,
                other => unreachable!("invalid singleton id {other} in JsVal"),
            },
            TAG_STRBIG => {
                // The discriminator bit selects String (set) vs BigInt (clear).
                if self.is_string() {
                    Value::String(JsStr::from_rc(unsafe {
                        self.as_string().unwrap_unchecked()
                    }))
                } else {
                    Value::BigInt(unsafe { self.as_bigint().unwrap_unchecked() })
                }
            }
            other => unreachable!("invalid tag {other} in JsVal"),
        }
    }

    // ------------------------------------------------------------------
    // OWNING-bank refcount hooks (T2 Phase 3).
    //
    // `JsVal` is `Copy`/no-`Drop` by design (the JIT loads/stores it as a raw
    // register word), so OWNERSHIP of a heap value held in a `JsVal` slot must be
    // imposed by an OUTSIDE wrapper (the owning `RegBank`), not by `JsVal` itself.
    // These two methods are that wrapper's primitives: `rc_inc` bumps and `rc_dec`
    // drops exactly one strong ref of the pointee `Rc<T>` for the slot's lane,
    // WITHOUT changing the box bits (so the slot keeps round-tripping to the same
    // pointer). They are the inverse of each other and net-zero in pairs.
    //
    // The per-lane `T` table below MUST stay in lock-step with `to_value` above
    // (the canonical tag→Rc<T> table) — a wrong `T` is a wrong `RcInner` layout
    // and silent heap corruption. They are co-located here for exactly that
    // reason, and `jsval_rc_inc_dec_is_net_zero` (the leak oracle's unit-level
    // sibling) proves each lane round-trips the strong count.
    //
    // STRBIG masking: the String/BigInt lane shares `TAG_STRBIG`; both are thin
    // `Rc` pointers but to DIFFERENT `T` (`Rc<JsString>` vs `Rc<JsBigInt>`), and
    // the String lane has the `STRBIG_IS_STRING` discriminator bit SET in the
    // payload. We therefore mask that bit off FIRST (and pick the right `T` by the
    // bit) before reconstructing the pointer — otherwise inc/dec would run on a
    // wild `+2^47` address.
    // ------------------------------------------------------------------

    /// Increment the strong count of the pointee `Rc<T>` for this slot's lane by
    /// one. No-op for a non-pointer (immediate) lane. Leaves the box bits
    /// unchanged.
    ///
    /// # Safety
    /// If `self.is_pointer()`, the pointee `Rc` (or a clone) must currently be
    /// alive so the `RcInner` it points at is valid. After this call the strong
    /// count is +1 (the wrapper must pair it with exactly one `rc_dec`).
    #[inline]
    pub unsafe fn rc_inc(self) {
        if !self.is_pointer() {
            return;
        }
        // SAFETY (per lane): the payload is a live `Rc::as_ptr` of the matching
        // `T` (the same `T` `to_value` decodes for this tag). `increment_strong_count`
        // reads the `RcInner` header at that address. STRBIG masks the discriminator
        // bit + picks `T` by it first.
        match self.tag() {
            TAG_OBJECT => unsafe {
                Rc::increment_strong_count(self.payload() as usize as *const RefCell<HashMap<String, Value>>)
            },
            TAG_ARRAY => unsafe {
                Rc::increment_strong_count(self.payload() as usize as *const RefCell<Vec<Value>>)
            },
            TAG_FUNCTION => unsafe {
                Rc::increment_strong_count(self.payload() as usize as *const FunctionValue)
            },
            TAG_NATIVE => unsafe {
                Rc::increment_strong_count(self.payload() as usize as *const NativeFn)
            },
            TAG_BCCLOSURE => unsafe {
                Rc::increment_strong_count(self.payload() as usize as *const BcClosure)
            },
            TAG_STRBIG => {
                // Mask the String discriminator bit FIRST, then pick T by it.
                let p = (self.payload() & !STRBIG_IS_STRING) as usize;
                if (self.payload() & STRBIG_IS_STRING) != 0 {
                    unsafe { Rc::increment_strong_count(p as *const JsString) }
                } else {
                    unsafe { Rc::increment_strong_count(p as *const JsBigInt) }
                }
            }
            // Unreachable: is_pointer() is true only for the lanes above.
            _ => {}
        }
    }

    /// Decrement the strong count of the pointee `Rc<T>` for this slot's lane by
    /// one (may run the pointee's `Drop` if this was the last ref). No-op for a
    /// non-pointer (immediate) lane. Leaves the box bits unchanged (the caller
    /// must overwrite/forget the slot — reading it after the last dec is a UAF).
    ///
    /// # Safety
    /// If `self.is_pointer()`, the pointee must currently have strong count ≥ 1
    /// (this call owns exactly one of those refs, e.g. one paired with a prior
    /// `rc_inc` or the box's original owner if `mem::forget`-transferred).
    #[inline]
    pub unsafe fn rc_dec(self) {
        if !self.is_pointer() {
            return;
        }
        // SAFETY: mirror of `rc_inc` — same lane→T table; `decrement_strong_count`
        // drops one ref of the same `RcInner`. STRBIG masked + dispatched first.
        match self.tag() {
            TAG_OBJECT => unsafe {
                Rc::decrement_strong_count(self.payload() as usize as *const RefCell<HashMap<String, Value>>)
            },
            TAG_ARRAY => unsafe {
                Rc::decrement_strong_count(self.payload() as usize as *const RefCell<Vec<Value>>)
            },
            TAG_FUNCTION => unsafe {
                Rc::decrement_strong_count(self.payload() as usize as *const FunctionValue)
            },
            TAG_NATIVE => unsafe {
                Rc::decrement_strong_count(self.payload() as usize as *const NativeFn)
            },
            TAG_BCCLOSURE => unsafe {
                Rc::decrement_strong_count(self.payload() as usize as *const BcClosure)
            },
            TAG_STRBIG => {
                let p = (self.payload() & !STRBIG_IS_STRING) as usize;
                if (self.payload() & STRBIG_IS_STRING) != 0 {
                    unsafe { Rc::decrement_strong_count(p as *const JsString) }
                } else {
                    unsafe { Rc::decrement_strong_count(p as *const JsBigInt) }
                }
            }
            _ => {}
        }
    }

    /// The raw 64-bit word (for the JIT / debugging).
    #[inline]
    pub const fn bits(self) -> u64 {
        self.0
    }
}

impl std::fmt::Debug for JsVal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.is_number() {
            return write!(f, "JsVal::Number({})", f64::from_bits(self.0));
        }
        match self.tag() {
            TAG_OBJECT => write!(f, "JsVal::Object({:#x})", self.payload()),
            TAG_ARRAY => write!(f, "JsVal::Array({:#x})", self.payload()),
            TAG_FUNCTION => write!(f, "JsVal::Function({:#x})", self.payload()),
            TAG_NATIVE => write!(f, "JsVal::Native({:#x})", self.payload()),
            TAG_BCCLOSURE => write!(f, "JsVal::BcClosure({:#x})", self.payload()),
            TAG_INT32 => write!(f, "JsVal::Int32({})", self.payload() as u32 as i32),
            TAG_SINGLETON => match self.payload() {
                SINGLE_UNDEFINED => f.write_str("JsVal::Undefined"),
                SINGLE_NULL => f.write_str("JsVal::Null"),
                SINGLE_FALSE => f.write_str("JsVal::Bool(false)"),
                SINGLE_TRUE => f.write_str("JsVal::Bool(true)"),
                SINGLE_HOLE => f.write_str("JsVal::Hole"),
                other => write!(f, "JsVal::BadSingleton({other:#x})"),
            },
            TAG_STRBIG => {
                if self.is_string() {
                    write!(f, "JsVal::String({:#x})", self.payload() & !STRBIG_IS_STRING)
                } else {
                    write!(f, "JsVal::BigInt({:#x})", self.payload())
                }
            }
            other => write!(f, "JsVal::BadTag({other})"),
        }
    }
}

// ============================================================================
// EXHAUSTIVE CORRECTNESS TESTS — the Phase-0 gate.
//
// Correctness of the encoding is the entire deliverable: a wrong bit scheme is
// silent data corruption when the migration flips onto it. These tests prove
// the encoding is a BIJECTION on representable values, that all NaNs canonicalize,
// that -0.0 keeps its sign, that no finite double collides with any tag, and that
// the pointer lanes round-trip to the EXACT same Rc pointer (GC/IC identity).
// ============================================================================
#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: a fresh Object Value.
    fn obj() -> Value {
        Value::Object(Rc::new(RefCell::new(HashMap::new())))
    }
    fn arr() -> Value {
        Value::Array(Rc::new(RefCell::new(vec![Value::Number(1.0), Value::Null])))
    }
    /// A real `Function` Value, obtained via the public interpreter (so we never
    /// touch the private `Env`/`Scope` types). It is a genuine `Rc<FunctionValue>`.
    fn func() -> Value {
        let mut interp = crate::interp::Interp::new();
        let v = interp
            .eval_source_to_value("(function f(x){ return x + 1; })")
            .expect("eval function expr");
        assert!(matches!(v, Value::Function(_)), "expected a Function");
        v
    }
    fn native() -> Value {
        crate::interp::native_fn("nf", |_args| Ok(Value::Undefined))
    }

    /// Round-trip a Value through JsVal and assert structural+identity equality.
    fn roundtrip(v: Value) -> Value {
        let jv = JsVal::try_from_value(&v).expect("representable in Phase-0");
        unsafe { jv.to_value() }
    }

    // ---- (a) ROUND-TRIP: primitives & singletons ----
    #[test]
    fn roundtrip_singletons() {
        assert!(matches!(roundtrip(Value::Undefined), Value::Undefined));
        assert!(matches!(roundtrip(Value::Null), Value::Null));
        assert!(matches!(roundtrip(Value::Hole), Value::Hole));
        assert!(matches!(roundtrip(Value::Bool(true)), Value::Bool(true)));
        assert!(matches!(roundtrip(Value::Bool(false)), Value::Bool(false)));
    }

    // ---- (a) ROUND-TRIP: a spread of f64s ----
    #[test]
    fn roundtrip_doubles() {
        let cases = [
            0.0f64,
            1.0,
            -1.0,
            3.141592653589793,
            f64::MIN,
            f64::MAX,
            f64::MIN_POSITIVE,         // smallest normal
            f64::from_bits(1),         // smallest subnormal
            f64::from_bits(0x000F_FFFF_FFFF_FFFF), // largest subnormal
            f64::INFINITY,
            f64::NEG_INFINITY,
            123456789.0,
            -987654321.0,
            1e308,
            -1e-308,
        ];
        for &n in &cases {
            match roundtrip(Value::Number(n)) {
                Value::Number(m) => assert_eq!(
                    m.to_bits(),
                    n.to_bits(),
                    "double {n} round-trip changed bits"
                ),
                other => panic!("expected Number, got {other:?}"),
            }
        }
    }

    // ---- (d) -0.0 keeps its sign ----
    #[test]
    fn roundtrip_negative_zero_keeps_sign() {
        let jv = JsVal::number(-0.0);
        let back = jv.as_f64().unwrap();
        assert!(back.is_sign_negative(), "-0.0 lost its sign");
        assert_eq!(back.to_bits(), (-0.0f64).to_bits());
        // and through the Value bridge:
        match roundtrip(Value::Number(-0.0)) {
            Value::Number(m) => {
                assert!(m.is_sign_negative());
                assert_eq!(m.to_bits(), (-0.0f64).to_bits());
            }
            other => panic!("expected Number, got {other:?}"),
        }
        // +0.0 stays positive and distinct in bits from -0.0
        assert!(JsVal::number(0.0).as_f64().unwrap().is_sign_positive());
        assert_ne!(JsVal::number(0.0).bits(), JsVal::number(-0.0).bits());
    }

    // ---- (c) CANONICALIZATION: every NaN boxes to the canonical NaN ----
    #[test]
    fn all_nans_canonicalize() {
        let nans = [
            f64::NAN,
            0.0f64 / 0.0,
            f64::INFINITY - f64::INFINITY,
            f64::from_bits(0x7FF8_0000_0000_0001), // quiet NaN, payload
            f64::from_bits(0x7FF0_0000_0000_0001), // signalling NaN
            f64::from_bits(0xFFF8_0000_0000_0000), // sign-bit NaN (would alias box space!)
            f64::from_bits(0xFFFF_FFFF_FFFF_FFFF), // all-ones NaN
            -f64::NAN,
        ];
        for &n in &nans {
            assert!(n.is_nan(), "test input must be NaN");
            let jv = JsVal::number(n);
            assert_eq!(
                jv.bits(),
                CANONICAL_NAN,
                "NaN {:#x} did not canonicalize",
                n.to_bits()
            );
            // canonical NaN must decode to a Number that is NaN…
            assert!(jv.is_number(), "canonical NaN must be a Number, not boxed");
            assert!(jv.as_f64().unwrap().is_nan());
            match unsafe { jv.to_value() } {
                Value::Number(m) => assert!(m.is_nan()),
                other => panic!("canonical NaN decoded to {other:?}"),
            }
        }
    }

    /// The sign-bit NaN `0xFFF8...` is *exactly* the boxed discriminator bits.
    /// This is the collision that canonicalization exists to prevent: prove that
    /// after boxing, that input is NOT mistaken for a boxed value.
    #[test]
    fn signbit_nan_does_not_alias_box_space() {
        let dangerous = f64::from_bits(0xFFF8_0000_0000_0000);
        assert!(dangerous.is_nan());
        // If we had NOT canonicalized, these bits == QNAN_BITS with tag 0,
        // payload 0 == "Object at null". Canonicalization moves it to sign=0.
        let jv = JsVal::number(dangerous);
        assert!(!jv.is_object(), "sign-bit NaN must not look like an Object");
        assert!(jv.is_number());
    }

    // ---- Int32 lane ----
    #[test]
    fn int32_lane_roundtrips() {
        for &n in &[0i32, 1, -1, i32::MIN, i32::MAX, 42, -42, 1 << 30, -(1 << 30)] {
            let jv = JsVal::int32(n);
            assert!(jv.is_int32(), "int32 {n} not recognized");
            assert!(!jv.is_number(), "int32 must be boxed, not a double");
            assert_eq!(jv.as_int32(), Some(n));
            assert_eq!(jv.to_f64(), Some(n as f64));
            // Int32 decodes to a Number Value (JS has one number type).
            match unsafe { jv.to_value() } {
                Value::Number(m) => assert_eq!(m, n as f64),
                other => panic!("int32 decoded to {other:?}"),
            }
        }
    }

    // ---- (a) ROUND-TRIP: pointer lanes round-trip to the SAME Rc pointer ----
    #[test]
    fn pointer_lanes_roundtrip_identity() {
        for v in [obj(), arr(), func(), native(), bcclosure_value()] {
            let original_ptr = value_ptr(&v);
            let original_strong = value_strong_count(&v);

            let jv = JsVal::try_from_value(&v).unwrap();
            let back = unsafe { jv.to_value() };

            // Same heap pointer (GC identity / inline-cache key preserved).
            assert_eq!(
                value_ptr(&back),
                original_ptr,
                "pointer lane lost identity for {v:?}"
            );
            // The engine's own strict-equality uses Rc::ptr_eq for heap kinds,
            // so equality here means the SAME underlying allocation.
            assert!(
                crate::interp::Value::strict_eq(&v, &back),
                "decoded Value != original for {v:?}"
            );
            // Strong count unchanged by the box→unbox round-trip net
            // (the borrowed-handle contract): `back` is one extra live clone,
            // so count is original+1 now; drop it and confirm we're back.
            assert_eq!(
                value_strong_count(&v),
                original_strong + 1,
                "round-trip changed refcount unexpectedly for {v:?}"
            );
            drop(back);
            assert_eq!(
                value_strong_count(&v),
                original_strong,
                "dropping decoded clone did not restore refcount for {v:?}"
            );
            // The JsVal's raw pointer matches the canonical pointer too.
            assert_eq!(jv.as_ptr_usize(), Some(original_ptr));
        }
    }

    /// Box the SAME Rc twice → identical bits (deterministic pointer encoding).
    #[test]
    fn same_rc_boxes_to_same_bits() {
        let v = obj();
        let a = JsVal::try_from_value(&v).unwrap();
        let b = JsVal::try_from_value(&v).unwrap();
        assert_eq!(a.bits(), b.bits());
        assert_eq!(a, b);
        // Different objects → different bits.
        let w = obj();
        let c = JsVal::try_from_value(&w).unwrap();
        assert_ne!(a.bits(), c.bits(), "distinct objects collided");
    }

    // ---- (b) NO COLLISION: doubles never land in the boxed space ----
    #[test]
    fn no_finite_double_collides_with_box_space() {
        // Sweep a wide spread of doubles; none (post-canonicalization) may have
        // the boxed discriminator bits set.
        let mut bits = 0x0000_0000_0000_0001u64;
        let mut tested = 0u64;
        // Walk an exponential spread of bit patterns across the whole u64 space.
        loop {
            let f = f64::from_bits(bits);
            let jv = JsVal::number(f);
            if f.is_nan() {
                assert_eq!(jv.bits(), CANONICAL_NAN);
            } else {
                assert!(
                    jv.is_number() && !jv.is_boxed(),
                    "double {f} ({bits:#x}) landed in boxed space"
                );
            }
            tested += 1;
            // Multiplicative walk hits ~64 distinct magnitudes; also flip sign.
            let next = bits.wrapping_mul(2).wrapping_add(1);
            if next <= bits {
                break;
            }
            bits = next;
        }
        assert!(tested > 50);

        // And explicitly: the boxed discriminator value as a double is a NaN,
        // so it canonicalizes — it can never be produced as a "double" JsVal.
        assert!(f64::from_bits(QNAN_BITS).is_nan());
    }

    /// Every tag/singleton combination is disjoint from every other and from the
    /// number space. (Full enumeration of the immediate value space.)
    #[test]
    fn all_immediates_are_distinct() {
        use std::collections::HashSet;
        let mut seen: HashSet<u64> = HashSet::new();
        let mut push = |jv: JsVal| {
            assert!(seen.insert(jv.bits()), "collision at {jv:?}");
        };
        push(JsVal::undefined());
        push(JsVal::null());
        push(JsVal::hole());
        push(JsVal::boolean(true));
        push(JsVal::boolean(false));
        // A spread of int32s.
        for n in [i32::MIN, -1, 0, 1, i32::MAX] {
            push(JsVal::int32(n));
        }
        // A spread of doubles.
        for n in [0.0f64, -0.0, 1.0, -1.0, f64::INFINITY, f64::NEG_INFINITY] {
            push(JsVal::number(n));
        }
        push(JsVal::number(f64::NAN)); // canonical NaN, one entry
                                       // boxing NaN again must NOT add a new entry (same bits).
        assert!(!seen.insert(JsVal::number(0.0 / 0.0).bits()));
    }

    /// Predicates are mutually exclusive for every representable kind.
    #[test]
    fn predicates_are_exclusive() {
        let samples: Vec<(JsVal, &str)> = vec![
            (JsVal::undefined(), "undef"),
            (JsVal::null(), "null"),
            (JsVal::hole(), "hole"),
            (JsVal::boolean(true), "bool"),
            (JsVal::int32(7), "int32"),
            (JsVal::number(2.5), "num"),
            (JsVal::try_from_value(&obj()).unwrap(), "obj"),
            (JsVal::try_from_value(&arr()).unwrap(), "arr"),
        ];
        for (jv, _name) in &samples {
            let flags = [
                jv.is_undefined(),
                jv.is_null(),
                jv.is_hole(),
                jv.is_bool(),
                jv.is_int32(),
                jv.is_number(),
                jv.is_object(),
                jv.is_array(),
                jv.is_function(),
                jv.is_native(),
                jv.is_bcclosure(),
            ];
            let n_true = flags.iter().filter(|b| **b).count();
            assert_eq!(n_true, 1, "exactly one predicate must hold for {jv:?}");
        }
    }

    // ---- Phase-1b: String is now boxable via the thin `JsStr`/`Rc<JsString>` ----

    /// The structural reason String is now boxable: `JsStr` and `Rc<JsString>`
    /// are thin 8-byte pointers (the old `Rc<str>` was a 16-byte fat pointer).
    #[test]
    fn jsstring_handle_is_thin() {
        assert_eq!(std::mem::size_of::<Rc<JsString>>(), 8, "Rc<JsString> is thin");
        assert_eq!(std::mem::size_of::<JsStr>(), 8, "JsStr is a thin newtype");
        assert_eq!(std::mem::size_of::<Rc<JsBigInt>>(), 8, "Rc<JsBigInt> is thin");
        // For contrast, the type we moved AWAY from:
        assert_eq!(std::mem::size_of::<Rc<str>>(), 16, "Rc<str> was a fat pointer");
    }

    /// A `Value::String` round-trips through `JsVal` to the IDENTICAL `Rc<JsString>`
    /// allocation (pointer identity preserved, refcount net-unchanged, content
    /// preserved) — exactly like the other heap pointer lanes.
    #[test]
    fn string_lane_roundtrips_identity() {
        let v = Value::str("hello world");
        let rc = match &v {
            Value::String(js) => js.as_rc().clone(),
            _ => unreachable!(),
        };
        let original_ptr = Rc::as_ptr(&rc) as *const () as usize;
        let original_strong = Rc::strong_count(&rc);

        let jv = JsVal::try_from_value(&v).expect("String is representable in Phase-1b");
        assert!(jv.is_string(), "expected the String lane, got {jv:?}");
        assert!(!jv.is_bigint());
        assert!(jv.is_pointer(), "String is a heap/pointer lane");
        assert_eq!(jv.as_ptr_usize(), Some(original_ptr), "GC/IC key preserved");

        let back = unsafe { jv.to_value() };
        match &back {
            Value::String(js2) => {
                assert_eq!(
                    Rc::as_ptr(js2.as_rc()) as *const () as usize,
                    original_ptr,
                    "String lane lost pointer identity"
                );
                assert_eq!(&**js2, "hello world", "String content changed");
            }
            other => panic!("String decoded to {other:?}"),
        }
        assert!(Value::strict_eq(&v, &back), "decoded String != original");
        // Borrowed-handle contract: `back` is one extra live clone now.
        assert_eq!(Rc::strong_count(&rc), original_strong + 1);
        drop(back);
        assert_eq!(Rc::strong_count(&rc), original_strong);
    }

    /// A spread of string contents (empty, ascii, unicode, long) all round-trip
    /// byte-identically and stay distinct from BigInt and every other lane.
    #[test]
    fn string_values_roundtrip_and_are_distinct() {
        use std::collections::HashSet;
        // Keep every Value alive for the whole test so a freed allocation can't
        // be recycled into the next (which would alias pointers — a test
        // artifact, not an encoding bug).
        let vals: Vec<Value> = ["", "a", "hello", "😀 unicode ✓", &"x".repeat(1000)]
            .iter()
            .map(|s| Value::str(*s))
            .collect();
        let mut seen = HashSet::new();
        for v in &vals {
            let jv = JsVal::try_from_value(v).unwrap();
            assert!(jv.is_string(), "expected String lane for {v:?}");
            assert!(!jv.is_bigint() && !jv.is_number() && !jv.is_object() && !jv.is_array());
            assert!(seen.insert(jv.bits()), "string {v:?} collided in JsVal bits");
            let back = unsafe { jv.to_value() };
            assert!(Value::strict_eq(v, &back), "string {v:?} round-trip changed value");
        }
    }

    /// String and BigInt share `TAG_STRBIG` but the discriminator bit keeps them
    /// disjoint even when (hypothetically) at equal payloads, and boxing the same
    /// `Rc<JsString>` twice is deterministic.
    #[test]
    fn string_bigint_lanes_are_disjoint() {
        let s = Value::str("42");
        let s_rc = match &s {
            Value::String(js) => js.as_rc().clone(),
            _ => unreachable!(),
        };
        let a = JsVal::string(&s_rc);
        let b = JsVal::string(&s_rc);
        assert_eq!(a.bits(), b.bits(), "same Rc<JsString> must box deterministically");
        assert!(a.is_string() && !a.is_bigint());
        assert_eq!(a.tag(), TAG_STRBIG);
        // The discriminator bit is what distinguishes the kinds.
        assert_ne!(a.payload() & STRBIG_IS_STRING, 0, "String bit must be set");

        let bi: Rc<JsBigInt> = Rc::new(crate::interp::parse_bigint_from_string("42").unwrap());
        let big = JsVal::bigint(&bi);
        assert!(big.is_bigint() && !big.is_string());
        assert_eq!(big.payload() & STRBIG_IS_STRING, 0, "BigInt bit must be clear");
        assert_eq!(big.tag(), TAG_STRBIG);
        // Same tag, different discriminator → never confused.
        assert_ne!(a.is_string(), big.is_string());
    }

    /// TOTALITY: `try_from_value` succeeds for EVERY `Value` variant (one sample
    /// per variant, exhaustively enumerated) and each round-trips back to an
    /// equal Value. This is the property Phase 1b delivers.
    #[test]
    fn jsval_is_total_over_all_variants() {
        // One representative of every `Value` variant. The match below is
        // exhaustive (no `_` arm) so adding a future variant forces this test to
        // be updated — totality cannot silently regress.
        let samples: Vec<Value> = vec![
            Value::Undefined,
            Value::Null,
            Value::Hole,
            Value::Bool(true),
            Value::Bool(false),
            Value::Number(3.5),
            Value::Number(f64::NAN),
            Value::str("string variant"),
            Value::BigInt(Rc::new(crate::interp::parse_bigint_from_string("7").unwrap())),
            obj(),
            arr(),
            func(),
            native(),
            bcclosure_value(),
        ];
        // Compile-time exhaustiveness guard: every variant must be listed above.
        for v in &samples {
            // try_from_value must NEVER return None now (JsVal is total).
            let jv = JsVal::try_from_value(v)
                .unwrap_or_else(|| panic!("JsVal not total: {v:?} returned None"));
            let back = unsafe { jv.to_value() };
            // NaN compares unequal to itself, so handle it specially.
            if let (Value::Number(a), Value::Number(b)) = (v, &back) {
                if a.is_nan() {
                    assert!(b.is_nan(), "NaN round-trip lost NaN-ness");
                    continue;
                }
            }
            // `strict_eq` (JS `===`) is intentionally non-reflexive for `Hole`
            // (it reads as `undefined`; there is no `(Hole, Hole)` arm), so for
            // it we assert the variant is preserved via the discriminant instead.
            if matches!(v, Value::Hole) {
                assert!(
                    matches!(back, Value::Hole),
                    "Hole round-trip changed variant -> {back:?}"
                );
                continue;
            }
            assert!(
                Value::strict_eq(v, &back),
                "totality round-trip changed {v:?} -> {back:?}"
            );
        }

        // Exhaustiveness assertion: this match has no wildcard, so the test fails
        // to compile if a new `Value` variant is added without covering it here —
        // a structural guarantee that the totality enumeration above stays
        // complete.
        fn _exhaustive(v: &Value) {
            match v {
                Value::Undefined
                | Value::Null
                | Value::Hole
                | Value::Bool(_)
                | Value::Number(_)
                | Value::String(_)
                | Value::BigInt(_)
                | Value::Object(_)
                | Value::Array(_)
                | Value::Function(_)
                | Value::NativeFunction(_)
                | Value::BcClosure(_) => {}
            }
        }
    }

    /// A `Value::BigInt` round-trips through `JsVal` to the IDENTICAL `Rc`
    /// allocation (pointer identity preserved, refcount net-unchanged) — exactly
    /// like the other heap pointer lanes.
    #[test]
    fn bigint_lane_roundtrips_identity() {
        let rc: Rc<JsBigInt> = Rc::new(crate::interp::parse_bigint_from_string("123").unwrap());
        let v = Value::BigInt(rc.clone());
        let original_ptr = Rc::as_ptr(&rc) as *const () as usize;
        let original_strong = Rc::strong_count(&rc);

        let jv = JsVal::try_from_value(&v).expect("BigInt is representable in Phase-1");
        assert!(jv.is_bigint(), "expected the BigInt lane, got {jv:?}");
        assert!(!jv.is_string());
        assert!(jv.is_pointer(), "BigInt is a heap/pointer lane");
        assert_eq!(jv.as_ptr_usize(), Some(original_ptr), "GC/IC key preserved");

        let back = unsafe { jv.to_value() };
        match &back {
            Value::BigInt(rc2) => {
                assert_eq!(
                    Rc::as_ptr(rc2) as *const () as usize,
                    original_ptr,
                    "BigInt lane lost pointer identity"
                );
            }
            other => panic!("BigInt decoded to {other:?}"),
        }
        assert!(Value::strict_eq(&v, &back), "decoded BigInt != original");
        // Borrowed-handle contract: `back` is one extra live clone now.
        assert_eq!(Rc::strong_count(&rc), original_strong + 1);
        drop(back);
        assert_eq!(Rc::strong_count(&rc), original_strong);
    }

    /// A spread of BigInt magnitudes (incl. negative, zero, multi-limb) all
    /// round-trip and stay distinct from every other immediate/pointer lane.
    #[test]
    fn bigint_values_roundtrip_and_are_distinct() {
        use std::collections::HashSet;
        // Keep every Value alive for the whole test so the allocator can't
        // recycle a freed BigInt's address into the next one (which would make
        // two distinct values share a pointer → identical JsVal bits — a test
        // artifact, not an encoding bug).
        let vals: Vec<Value> = ["0", "1", "-1", "123", "-9876543210", "18446744073709551616"]
            .iter()
            .map(|s| Value::BigInt(Rc::new(crate::interp::parse_bigint_from_string(s).unwrap())))
            .collect();
        let mut seen = HashSet::new();
        for v in &vals {
            let jv = JsVal::try_from_value(v).unwrap();
            assert!(jv.is_bigint());
            assert!(!jv.is_number() && !jv.is_object() && !jv.is_array());
            // distinct (simultaneously-live) allocation → distinct bits
            assert!(seen.insert(jv.bits()), "bigint {v:?} collided in JsVal bits");
            let back = unsafe { jv.to_value() };
            assert!(Value::strict_eq(v, &back), "bigint {v:?} round-trip changed value");
        }
    }

    /// Boxing the SAME BigInt `Rc` twice yields identical bits (deterministic),
    /// and the BigInt lane does not alias the object pointer lanes.
    #[test]
    fn bigint_same_rc_same_bits_and_no_alias() {
        let rc: Rc<JsBigInt> = Rc::new(crate::interp::parse_bigint_from_string("42").unwrap());
        let a = JsVal::bigint(&rc);
        let b = JsVal::bigint(&rc);
        assert_eq!(a.bits(), b.bits());
        // An Object JsVal with the same raw pointer would have tag 0, not 7 —
        // prove the tag keeps the lanes disjoint even at equal payloads.
        assert_ne!(a.tag(), TAG_OBJECT);
        assert_eq!(a.tag(), TAG_STRBIG);
    }

    #[test]
    fn jsval_is_one_word() {
        assert_eq!(std::mem::size_of::<JsVal>(), 8);
        assert_eq!(std::mem::align_of::<JsVal>(), std::mem::align_of::<u64>());
    }

    /// LEAK ORACLE (unit level, per heap lane): `rc_inc` then `rc_dec` on a slot
    /// is NET-ZERO on the pointee's strong count, and a lone `rc_inc` raises it by
    /// exactly one. This is the contract the owning `RegBank` is built on (the
    /// inc/dec must pair perfectly or the bank leaks / double-frees). Every
    /// pointer lane is covered (incl. both STRBIG sub-lanes, which share a tag but
    /// need different `T`); `rc_inc`/`rc_dec` on an immediate is a no-op.
    #[test]
    fn jsval_rc_inc_dec_is_net_zero_per_lane() {
        // (a JsVal slot, an owning Value/Rc kept alive, the slot's strong count)
        let obj_v = obj();
        let arr_v = arr();
        let fun_v = func();
        let nat_v = native();
        let bcc_v = bcclosure_value();
        let str_v = Value::str("rc lane");
        let big_v =
            Value::BigInt(Rc::new(crate::interp::parse_bigint_from_string("99").unwrap()));

        let lanes: Vec<(JsVal, &Value)> = vec![
            (JsVal::try_from_value(&obj_v).unwrap(), &obj_v),
            (JsVal::try_from_value(&arr_v).unwrap(), &arr_v),
            (JsVal::try_from_value(&fun_v).unwrap(), &fun_v),
            (JsVal::try_from_value(&nat_v).unwrap(), &nat_v),
            (JsVal::try_from_value(&bcc_v).unwrap(), &bcc_v),
            (JsVal::try_from_value(&str_v).unwrap(), &str_v),
            (JsVal::try_from_value(&big_v).unwrap(), &big_v),
        ];

        for (jv, owner) in &lanes {
            assert!(jv.is_pointer(), "lane must be a pointer: {jv:?}");
            let before = strong_of(owner);
            // A lone inc raises the count by exactly one…
            unsafe { jv.rc_inc() };
            assert_eq!(strong_of(owner), before + 1, "rc_inc must +1 for {jv:?}");
            // …and the paired dec restores it (net-zero).
            unsafe { jv.rc_dec() };
            assert_eq!(strong_of(owner), before, "inc+dec must net-zero for {jv:?}");
        }

        // Immediates: inc/dec are no-ops (and never touch a wild pointer).
        for imm in [
            JsVal::number(1.5),
            JsVal::boolean(true),
            JsVal::int32(7),
            JsVal::undefined(),
            JsVal::null(),
        ] {
            assert!(!imm.is_pointer());
            unsafe { imm.rc_inc() };
            unsafe { imm.rc_dec() };
        }
    }

    /// The strong count of a heap-kind Value (matches the owning-bank accounting).
    fn strong_of(v: &Value) -> usize {
        match v {
            Value::Object(rc) => Rc::strong_count(rc),
            Value::Array(rc) => Rc::strong_count(rc),
            Value::Function(rc) => Rc::strong_count(rc),
            Value::NativeFunction(rc) => Rc::strong_count(rc),
            Value::BcClosure(rc) => Rc::strong_count(rc),
            Value::String(s) => Rc::strong_count(s.as_rc()),
            Value::BigInt(rc) => Rc::strong_count(rc),
            _ => 0,
        }
    }

    // ---- helpers that need engine internals ----

    fn bcclosure_value() -> Value {
        // `Module` is a fully-public struct (`{ fns: Vec<BcFunction> }`) and
        // `BcClosure`'s fields are all public, so we can build one directly with
        // no private-type access.
        let module = Rc::new(crate::bytecode::Module { fns: vec![], script_forinit_syncs: Vec::new() });
        Value::BcClosure(Rc::new(BcClosure {
            fn_idx: 0,
            upvalues: RefCell::new(vec![]),
            props: RefCell::new(HashMap::new()),
            module,
        }))
    }

    /// Canonical heap pointer of a heap-kind Value (matches the GC identity key).
    fn value_ptr(v: &Value) -> usize {
        match v {
            Value::Object(rc) => Rc::as_ptr(rc) as *const () as usize,
            Value::Array(rc) => Rc::as_ptr(rc) as *const () as usize,
            Value::Function(rc) => Rc::as_ptr(rc) as *const () as usize,
            Value::NativeFunction(rc) => Rc::as_ptr(rc) as *const () as usize,
            Value::BcClosure(rc) => Rc::as_ptr(rc) as *const () as usize,
            _ => 0,
        }
    }

    fn value_strong_count(v: &Value) -> usize {
        match v {
            Value::Object(rc) => Rc::strong_count(rc),
            Value::Array(rc) => Rc::strong_count(rc),
            Value::Function(rc) => Rc::strong_count(rc),
            Value::NativeFunction(rc) => Rc::strong_count(rc),
            Value::BcClosure(rc) => Rc::strong_count(rc),
            _ => 0,
        }
    }
}
