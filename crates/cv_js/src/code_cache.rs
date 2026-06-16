//! B5 — PERSISTED BYTECODE + TYPE-FEEDBACK CODE CACHE.
//!
//! Persists a compiled top-level script (`bytecode::Module`) PLUS its warmed
//! property inline-cache feedback (`PropIc`) to an on-disk blob, keyed by a hash
//! of `(source, engine version, relevant flags, shape-assumptions digest)`. On a
//! later run the cache is read back, the key VALIDATED, and on a match the Module
//! is installed DIRECTLY — skipping the parse + bytecode-compile pass — and the
//! warmed feedback is pre-seeded so a hot function tiers up to T2/T3 INSTANTLY
//! without re-profiling. On ANY key mismatch (source changed, engine-version
//! bump, flag change, shape-assumption drift) OR a corrupt/truncated blob, the
//! entry is DISCARDED and the normal parse+compile path runs unchanged — so a bad
//! cache can only ever cost a recompile, never correctness.
//!
//! ## Honest scope (and what is DEFERRED)
//!
//! This stage persists BYTECODE + TYPE-FEEDBACK, **not** native machine code.
//! Native T3 code is position-dependent (rel32 inter-function calls into the code
//! cage, baked helper addresses, baked shape ids), so serializing+relocating it
//! is large and fragile — it is the documented FOLLOW-ON (the B1 code cage is the
//! prerequisite groundwork). Persisting bytecode + feedback is still a real,
//! non-fake win: the next run pays no cold re-parse and no re-warm, and the hot
//! functions are immediately tier-up candidates. This module never fabricates a
//! value: a miss/mismatch/corruption returns `None` and the caller recompiles.
//!
//! ## Soundness: shape ids are NOT portable — descriptors are
//!
//! A `ShapeId` is a run-LOCAL interned integer; the same integer can name a
//! DIFFERENT key-sequence in a different run. Persisting a raw `ShapeId` in the
//! IC would be SILENT CORRUPTION (a baked guard could match a same-numbered but
//! differently-laid-out shape and read the WRONG slot). So the IC is persisted as
//! the SHAPE DESCRIPTOR — the actual key-sequence — and re-interned on load to
//! recover THIS run's `ShapeId`. The shape-assumptions DIGEST folds every
//! persisted descriptor into the cache key, so a layout change that the digest is
//! supposed to capture invalidates the whole entry. (The IC's own `record`/
//! `lookup` guard is a second backstop: a re-interned shape that no longer occurs
//! this run simply produces clean misses — self-correcting, never wrong.)
//!
//! ## Gating
//!
//! `CV_CODE_CACHE` (DEFAULT OFF). When off, nothing reads or writes the cache and
//! the engine is byte-identical to today. The format carries a magic + version so
//! a stale on-disk format is rejected rather than misread.

use crate::bytecode::{BcFunction, Module, Op, PropIc};
use crate::interp::Value;
use std::path::PathBuf;

/// Bump on ANY change to: the `Op` enum (variants/encoding), `BcFunction`/`Module`
/// layout, the compiler's lowering, the IC serialization, or the shape-id
/// semantics. A version mismatch rejects every on-disk entry (clean recompile).
/// This is the coarse "engine version" leg of the cache key — finer-grained flag
/// + source + shape digests live alongside it in the key.
pub const ENGINE_VERSION: u32 = 2;

/// Magic header bytes (`"TBCC"` = Toasty-Blum Code Cache) + a format version.
const MAGIC: u32 = 0x5442_4343; // 'T''B''C''C'
const FORMAT_VERSION: u32 = 1;

/// Whether the persisted code cache is enabled. DEFAULT-OFF (opt IN with
/// `CV_CODE_CACHE=1`), mirroring `CV_T3`/`CV_CODE_CAGE`/`CV_GEN_GC`. Read once.
pub fn code_cache_enabled() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("CV_CODE_CACHE").as_deref() == Ok("1"))
}

// ----------------------------------------------------------------------
// MUTATION HOOK (test-only). To PROVE the shape-assumptions digest is
// load-bearing, this hook makes `compute_key` OMIT the shape digest. With it set,
// a layout change does NOT change the key, so a stale entry is wrongly accepted —
// the round-trip oracle must then DIVERGE. With it unset (production default) the
// digest is included and the oracle is green. There is no env path — only the
// in-process setter behind the `StaleKeyGuard`.
// ----------------------------------------------------------------------
thread_local! {
    static FORCE_STALE_KEY: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Set the stale-key mutation hook (test-only). Returns the prior value. Use the
/// `StaleKeyGuard` scope guard.
pub fn set_force_stale_key(v: bool) -> bool {
    FORCE_STALE_KEY.with(|c| {
        let prev = c.get();
        c.set(v);
        prev
    })
}

fn force_stale_key() -> bool {
    FORCE_STALE_KEY.with(|c| c.get())
}

/// RAII guard for the stale-key mutation hook (test-only).
#[must_use]
pub struct StaleKeyGuard {
    prev: bool,
}
impl StaleKeyGuard {
    pub fn new(on: bool) -> Self {
        StaleKeyGuard { prev: set_force_stale_key(on) }
    }
}
impl Drop for StaleKeyGuard {
    fn drop(&mut self) {
        set_force_stale_key(self.prev);
    }
}

// ----------------------------------------------------------------------
// Cache key.
// ----------------------------------------------------------------------

/// The validation key for one cached module: a 64-bit hash folding the source
/// bytes, the engine version, the relevant flags, and the shape-assumptions
/// digest. Two runs that share a key are guaranteed to share an identical
/// bytecode program AND a compatible shape-feedback assumption set.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CacheKey(pub u64);

/// FNV-1a 64-bit — the same hash family the retained-display-list `content_hash`
/// uses elsewhere in the codebase. No third-party crate; deterministic.
struct Fnv1a(u64);
impl Fnv1a {
    #[inline]
    fn new() -> Self {
        Fnv1a(0xcbf2_9ce4_8422_2325)
    }
    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 ^= b as u64;
            self.0 = self.0.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    #[inline]
    fn write_u32(&mut self, v: u32) {
        self.write(&v.to_le_bytes());
    }
    #[inline]
    fn write_u64(&mut self, v: u64) {
        self.write(&v.to_le_bytes());
    }
    #[inline]
    fn finish(&self) -> u64 {
        self.0
    }
}

/// The relevant flag bits that, if changed, must invalidate the cache (because
/// they change the *shape* of the compiled program or the feedback's meaning).
/// Read from the same env vars the engine reads. We deliberately fold the TIER
/// flags too: although the persisted bytecode is tier-independent, the warmed
/// feedback bakes shape assumptions whose validity is tied to the object model,
/// and a conservative key never costs more than a recompile.
fn flag_bits() -> u32 {
    let mut bits = 0u32;
    // Object model / IC meaning.
    if crate::ordered::shaped_obj_enabled() {
        bits |= 1 << 0;
    }
    if crate::interp::gc_enabled() {
        bits |= 1 << 1;
    }
    if crate::bytecode::propic_enabled() {
        bits |= 1 << 2;
    }
    // Tier flags (read directly to avoid coupling to private accessors).
    if std::env::var("CV_BYTECODE").as_deref() != Ok("0") {
        bits |= 1 << 3;
    }
    if std::env::var("CV_T2").as_deref() != Ok("0") {
        bits |= 1 << 4;
    }
    if std::env::var("CV_T3").as_deref() == Ok("1") {
        bits |= 1 << 5;
    }
    bits
}

/// A digest of the SHAPE ASSUMPTIONS baked into a module's warmed IC feedback:
/// for every persisted IC entry, the property KEY-SEQUENCE (descriptor) of every
/// guarded shape, plus the recorded slot. If a shape's layout (its key-sequence,
/// or the slot a key maps to) ever differs from what was warmed, this digest
/// changes and the cache key no longer matches → the entry is rejected and
/// recompiled. This is the leg the stale-key mutation arm omits to prove it is
/// load-bearing.
/// The PORTABLE warmed IC feedback for one function: a sparse list of
/// `(ip, [(shape_descriptor, slot)], mega)`. This is THE single source of truth
/// shared by the shape-assumptions digest AND the on-disk writer — so the key
/// computed at WRITE time and the key recomputed from a RELOADED module are
/// identical by construction (both fold the same descriptors with the same
/// filtering). Each shape is rendered as its key-SEQUENCE (portable across runs);
/// DICT_SHAPE / uninitialized-slot entries are dropped (they never bake a valid
/// guard and re-warm at runtime). A site contributes only if it has ≥1 portable
/// descriptor OR has gone megamorphic.
fn portable_feedback(f: &BcFunction) -> Vec<(u32, Vec<(Vec<String>, u32)>, bool)> {
    let ics = f.ic.borrow();
    let mut out: Vec<(u32, Vec<(Vec<String>, u32)>, bool)> = Vec::new();
    for (ip, ic) in ics.iter().enumerate() {
        if !ic.has_feedback() {
            continue;
        }
        let (entries, mega) = ic.serialize_own();
        let mut descr: Vec<(Vec<String>, u32)> = Vec::new();
        for (shape_id, slot) in entries {
            if shape_id == crate::shapes::DICT_SHAPE || slot == u32::MAX {
                continue;
            }
            let keys = crate::shapes::global_shape_properties(shape_id);
            descr.push((keys, slot));
        }
        if descr.is_empty() && !mega {
            continue;
        }
        out.push((ip as u32, descr, mega));
    }
    out
}

fn shape_assumptions_digest(module: &Module) -> u64 {
    let mut h = Fnv1a::new();
    for (fi, f) in module.fns.iter().enumerate() {
        for (ip, descr, mega) in portable_feedback(f) {
            h.write_u32(fi as u32);
            h.write_u32(ip);
            h.write(&[mega as u8]);
            for (keys, slot) in &descr {
                h.write_u32(keys.len() as u32);
                for k in keys {
                    h.write_u32(k.len() as u32);
                    h.write(k.as_bytes());
                }
                h.write_u32(*slot);
            }
        }
    }
    h.finish()
}

/// Compute the validation key for `source` given the freshly-compiled `module`
/// (whose IC may already be warmed). The shape digest is included UNLESS the
/// stale-key mutation hook is engaged (test-only), which proves the digest is the
/// load-bearing invalidator.
pub fn compute_key(source: &str, module: &Module) -> CacheKey {
    let mut h = Fnv1a::new();
    h.write(b"tbcc-key-v1");
    h.write_u32(ENGINE_VERSION);
    h.write_u32(flag_bits());
    h.write_u64(source.len() as u64);
    h.write(source.as_bytes());
    if !force_stale_key() {
        h.write_u64(shape_assumptions_digest(module));
    }
    CacheKey(h.finish())
}

// ----------------------------------------------------------------------
// Binary (de)serialization — hand-rolled, length-prefixed, no third-party crates.
// ----------------------------------------------------------------------

/// A minimal append-only little-endian byte writer.
struct Writer {
    buf: Vec<u8>,
}
impl Writer {
    fn new() -> Self {
        Writer { buf: Vec::new() }
    }
    fn u8(&mut self, v: u8) {
        self.buf.push(v);
    }
    fn u16(&mut self, v: u16) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }
    fn u32(&mut self, v: u32) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }
    fn u64(&mut self, v: u64) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }
    fn f64(&mut self, v: f64) {
        // Persist the exact bit pattern so NaN payloads / -0 round-trip.
        self.buf.extend_from_slice(&v.to_bits().to_le_bytes());
    }
    fn bytes(&mut self, b: &[u8]) {
        self.u32(b.len() as u32);
        self.buf.extend_from_slice(b);
    }
    fn str(&mut self, s: &str) {
        self.bytes(s.as_bytes());
    }
}

/// A bounds-checked little-endian byte reader. Every read returns `None` on
/// truncation, so a corrupt blob fails closed (→ caller recompiles).
struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}
impl<'a> Reader<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Reader { buf, pos: 0 }
    }
    fn take(&mut self, n: usize) -> Option<&'a [u8]> {
        let end = self.pos.checked_add(n)?;
        let s = self.buf.get(self.pos..end)?;
        self.pos = end;
        Some(s)
    }
    fn u8(&mut self) -> Option<u8> {
        Some(self.take(1)?[0])
    }
    fn u16(&mut self) -> Option<u16> {
        let b = self.take(2)?;
        Some(u16::from_le_bytes([b[0], b[1]]))
    }
    fn u32(&mut self) -> Option<u32> {
        let b = self.take(4)?;
        Some(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }
    fn u64(&mut self) -> Option<u64> {
        let b = self.take(8)?;
        Some(u64::from_le_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }
    fn f64(&mut self) -> Option<f64> {
        Some(f64::from_bits(self.u64()?))
    }
    fn bytes(&mut self) -> Option<&'a [u8]> {
        let n = self.u32()? as usize;
        self.take(n)
    }
    fn str(&mut self) -> Option<String> {
        let b = self.bytes()?;
        std::str::from_utf8(b).ok().map(|s| s.to_string())
    }
}

// --- consts: only Number/String ever reach a bytecode const pool, but we encode
// the full serializable primitive set defensively. A non-serializable const
// (Object/Array/Function/…) makes `serialize_module` DECLINE (returns None) — we
// never fabricate; the caller just doesn't cache that module. ---

const C_UNDEF: u8 = 0;
const C_NULL: u8 = 1;
const C_BOOL: u8 = 2;
const C_NUM: u8 = 3;
const C_STR: u8 = 4;

fn write_const(w: &mut Writer, v: &Value) -> bool {
    match v {
        Value::Undefined => w.u8(C_UNDEF),
        Value::Null => w.u8(C_NULL),
        Value::Bool(b) => {
            w.u8(C_BOOL);
            w.u8(*b as u8);
        }
        Value::Number(n) => {
            w.u8(C_NUM);
            w.f64(*n);
        }
        Value::String(s) => {
            w.u8(C_STR);
            w.str(s.as_str());
        }
        // BigInt / Object / Array / callables never appear in a bytecode const
        // pool (verified: the compiler only adds Number/String consts). If one
        // somehow did, we DECLINE to cache rather than fake a representation.
        _ => return false,
    }
    true
}

fn read_const(r: &mut Reader) -> Option<Value> {
    match r.u8()? {
        C_UNDEF => Some(Value::Undefined),
        C_NULL => Some(Value::Null),
        C_BOOL => Some(Value::Bool(r.u8()? != 0)),
        C_NUM => Some(Value::Number(r.f64()?)),
        C_STR => Some(Value::str(r.str()?)),
        _ => None,
    }
}

// --- Op encoding: a 1-byte tag + the operand fields. The tag order MUST stay
// stable (bump ENGINE_VERSION to change it). A decode of an unknown tag fails
// closed. ---

macro_rules! op_tags {
    ($($name:ident = $val:expr),+ $(,)?) => {
        $(const $name: u8 = $val;)+
    };
}
op_tags! {
    T_LOAD_CONST = 0, T_LOAD_TRUE = 1, T_LOAD_FALSE = 2, T_LOAD_NULL = 3,
    T_LOAD_UNDEF = 4, T_MOVE = 5, T_ADD = 6, T_SUB = 7, T_MUL = 8, T_DIV = 9,
    T_MOD = 10, T_POW = 11, T_EQ = 12, T_NEQ = 13, T_LOOSE_EQ = 14,
    T_LOOSE_NEQ = 15, T_LT = 16, T_LE = 17, T_GT = 18, T_GE = 19, T_BIT_AND = 20,
    T_BIT_OR = 21, T_BIT_XOR = 22, T_SHL = 23, T_SHR = 24, T_USHR = 25,
    T_NEG = 26, T_NOT = 27, T_BIT_NOT = 28, T_TO_NUMBER = 29, T_TYPEOF = 30,
    T_IN = 31, T_DELETE_PROP = 32, T_DELETE_IDX = 33, T_MAKE_REGEX = 34,
    T_JMP = 35, T_JMP_IF_FALSE = 36, T_JMP_IF_TRUE = 37, T_CALL_FN = 38,
    T_LOAD_GLOBAL = 39, T_LOAD_GLOBAL_CHECKED = 40, T_STORE_GLOBAL = 41,
    T_CALL_VALUE = 42, T_NEW = 43, T_LOAD_THIS = 44, T_LOAD_SELF = 45,
    T_GET_PROP = 46, T_GET_IDX = 47, T_SET_PROP = 48, T_SET_IDX = 49,
    T_NEW_ARRAY = 50, T_ARRAY_PUSH = 51, T_ARRAY_PUSH_SPREAD = 52,
    T_NEW_OBJECT = 53, T_THROW = 54, T_TRY_ENTER = 55, T_TRY_EXIT = 56,
    T_ENUM_KEYS = 57, T_MAKE_CLOSURE = 58, T_LOAD_UP = 59, T_STORE_UP = 60,
    T_RET = 61, T_INSTANCEOF = 62, T_TO_STR = 63,
}

fn write_op(w: &mut Writer, op: &Op) {
    macro_rules! bin {
        ($tag:expr, $dst:expr, $lhs:expr, $rhs:expr) => {{
            w.u8($tag);
            w.u16($dst);
            w.u16($lhs);
            w.u16($rhs);
        }};
    }
    macro_rules! un {
        ($tag:expr, $dst:expr, $src:expr) => {{
            w.u8($tag);
            w.u16($dst);
            w.u16($src);
        }};
    }
    match *op {
        Op::LoadConst { dst, k } => {
            w.u8(T_LOAD_CONST);
            w.u16(dst);
            w.u16(k);
        }
        Op::LoadTrue { dst } => {
            w.u8(T_LOAD_TRUE);
            w.u16(dst);
        }
        Op::LoadFalse { dst } => {
            w.u8(T_LOAD_FALSE);
            w.u16(dst);
        }
        Op::LoadNull { dst } => {
            w.u8(T_LOAD_NULL);
            w.u16(dst);
        }
        Op::LoadUndef { dst } => {
            w.u8(T_LOAD_UNDEF);
            w.u16(dst);
        }
        Op::Move { dst, src } => un!(T_MOVE, dst, src),
        Op::Add { dst, lhs, rhs } => bin!(T_ADD, dst, lhs, rhs),
        Op::Sub { dst, lhs, rhs } => bin!(T_SUB, dst, lhs, rhs),
        Op::Mul { dst, lhs, rhs } => bin!(T_MUL, dst, lhs, rhs),
        Op::Div { dst, lhs, rhs } => bin!(T_DIV, dst, lhs, rhs),
        Op::Mod { dst, lhs, rhs } => bin!(T_MOD, dst, lhs, rhs),
        Op::Pow { dst, lhs, rhs } => bin!(T_POW, dst, lhs, rhs),
        Op::Eq { dst, lhs, rhs } => bin!(T_EQ, dst, lhs, rhs),
        Op::Neq { dst, lhs, rhs } => bin!(T_NEQ, dst, lhs, rhs),
        Op::LooseEq { dst, lhs, rhs } => bin!(T_LOOSE_EQ, dst, lhs, rhs),
        Op::LooseNeq { dst, lhs, rhs } => bin!(T_LOOSE_NEQ, dst, lhs, rhs),
        Op::Lt { dst, lhs, rhs } => bin!(T_LT, dst, lhs, rhs),
        Op::Le { dst, lhs, rhs } => bin!(T_LE, dst, lhs, rhs),
        Op::Gt { dst, lhs, rhs } => bin!(T_GT, dst, lhs, rhs),
        Op::Ge { dst, lhs, rhs } => bin!(T_GE, dst, lhs, rhs),
        Op::BitAnd { dst, lhs, rhs } => bin!(T_BIT_AND, dst, lhs, rhs),
        Op::BitOr { dst, lhs, rhs } => bin!(T_BIT_OR, dst, lhs, rhs),
        Op::BitXor { dst, lhs, rhs } => bin!(T_BIT_XOR, dst, lhs, rhs),
        Op::Shl { dst, lhs, rhs } => bin!(T_SHL, dst, lhs, rhs),
        Op::Shr { dst, lhs, rhs } => bin!(T_SHR, dst, lhs, rhs),
        Op::Ushr { dst, lhs, rhs } => bin!(T_USHR, dst, lhs, rhs),
        Op::Neg { dst, src } => un!(T_NEG, dst, src),
        Op::Not { dst, src } => un!(T_NOT, dst, src),
        Op::BitNot { dst, src } => un!(T_BIT_NOT, dst, src),
        Op::ToNumber { dst, src } => un!(T_TO_NUMBER, dst, src),
        Op::ToStr { dst, src } => un!(T_TO_STR, dst, src),
        Op::Typeof { dst, src } => un!(T_TYPEOF, dst, src),
        Op::In { dst, lhs, rhs } => bin!(T_IN, dst, lhs, rhs),
        Op::Instanceof { dst, lhs, rhs } => bin!(T_INSTANCEOF, dst, lhs, rhs),
        Op::DeleteProp { dst, obj, key_k } => {
            w.u8(T_DELETE_PROP);
            w.u16(dst);
            w.u16(obj);
            w.u16(key_k);
        }
        Op::DeleteIdx { dst, obj, key } => {
            w.u8(T_DELETE_IDX);
            w.u16(dst);
            w.u16(obj);
            w.u16(key);
        }
        Op::MakeRegex { dst, source_k, flags_k } => {
            w.u8(T_MAKE_REGEX);
            w.u16(dst);
            w.u16(source_k);
            w.u16(flags_k);
        }
        Op::Jmp { target } => {
            w.u8(T_JMP);
            w.u16(target);
        }
        Op::JmpIfFalse { cond, target } => {
            w.u8(T_JMP_IF_FALSE);
            w.u16(cond);
            w.u16(target);
        }
        Op::JmpIfTrue { cond, target } => {
            w.u8(T_JMP_IF_TRUE);
            w.u16(cond);
            w.u16(target);
        }
        Op::CallFn { dst, fn_idx, first_arg, n_args } => {
            w.u8(T_CALL_FN);
            w.u16(dst);
            w.u16(fn_idx);
            w.u16(first_arg);
            w.u8(n_args);
        }
        Op::LoadGlobal { dst, name_k } => {
            w.u8(T_LOAD_GLOBAL);
            w.u16(dst);
            w.u16(name_k);
        }
        Op::LoadGlobalChecked { dst, name_k } => {
            w.u8(T_LOAD_GLOBAL_CHECKED);
            w.u16(dst);
            w.u16(name_k);
        }
        Op::StoreGlobal { name_k, src } => {
            w.u8(T_STORE_GLOBAL);
            w.u16(name_k);
            w.u16(src);
        }
        Op::CallValue { dst, callee, this_reg, first_arg, n_args } => {
            w.u8(T_CALL_VALUE);
            w.u16(dst);
            w.u16(callee);
            w.u16(this_reg);
            w.u16(first_arg);
            w.u8(n_args);
        }
        Op::New { dst, ctor, first_arg, n_args } => {
            w.u8(T_NEW);
            w.u16(dst);
            w.u16(ctor);
            w.u16(first_arg);
            w.u8(n_args);
        }
        Op::LoadThis { dst } => {
            w.u8(T_LOAD_THIS);
            w.u16(dst);
        }
        Op::LoadSelf { dst } => {
            w.u8(T_LOAD_SELF);
            w.u16(dst);
        }
        Op::GetProp { dst, obj, key_k } => {
            w.u8(T_GET_PROP);
            w.u16(dst);
            w.u16(obj);
            w.u16(key_k);
        }
        Op::GetIdx { dst, obj, key } => {
            w.u8(T_GET_IDX);
            w.u16(dst);
            w.u16(obj);
            w.u16(key);
        }
        Op::SetProp { obj, key_k, src } => {
            w.u8(T_SET_PROP);
            w.u16(obj);
            w.u16(key_k);
            w.u16(src);
        }
        Op::SetIdx { obj, key, src } => {
            w.u8(T_SET_IDX);
            w.u16(obj);
            w.u16(key);
            w.u16(src);
        }
        Op::NewArray { dst, first_elem, n_elems } => {
            w.u8(T_NEW_ARRAY);
            w.u16(dst);
            w.u16(first_elem);
            w.u8(n_elems);
        }
        Op::ArrayPush { arr, val } => {
            w.u8(T_ARRAY_PUSH);
            w.u16(arr);
            w.u16(val);
        }
        Op::ArrayPushSpread { arr, spread } => {
            w.u8(T_ARRAY_PUSH_SPREAD);
            w.u16(arr);
            w.u16(spread);
        }
        Op::NewObject { dst } => {
            w.u8(T_NEW_OBJECT);
            w.u16(dst);
        }
        Op::Throw { src } => {
            w.u8(T_THROW);
            w.u16(src);
        }
        Op::TryEnter { catch_target, catch_reg } => {
            w.u8(T_TRY_ENTER);
            w.u16(catch_target);
            w.u16(catch_reg);
        }
        Op::TryExit => {
            w.u8(T_TRY_EXIT);
        }
        Op::EnumKeys { dst, obj } => un!(T_ENUM_KEYS, dst, obj),
        Op::MakeClosure { dst, fn_idx, first_upvalue, n_upvalues } => {
            w.u8(T_MAKE_CLOSURE);
            w.u16(dst);
            w.u16(fn_idx);
            w.u16(first_upvalue);
            w.u8(n_upvalues);
        }
        Op::LoadUp { dst, slot } => {
            w.u8(T_LOAD_UP);
            w.u16(dst);
            w.u8(slot);
        }
        Op::StoreUp { src, slot } => {
            w.u8(T_STORE_UP);
            w.u16(src);
            w.u8(slot);
        }
        Op::Ret { src } => {
            w.u8(T_RET);
            w.u16(src);
        }
    }
}

fn read_op(r: &mut Reader) -> Option<Op> {
    let tag = r.u8()?;
    Some(match tag {
        T_LOAD_CONST => Op::LoadConst { dst: r.u16()?, k: r.u16()? },
        T_LOAD_TRUE => Op::LoadTrue { dst: r.u16()? },
        T_LOAD_FALSE => Op::LoadFalse { dst: r.u16()? },
        T_LOAD_NULL => Op::LoadNull { dst: r.u16()? },
        T_LOAD_UNDEF => Op::LoadUndef { dst: r.u16()? },
        T_MOVE => Op::Move { dst: r.u16()?, src: r.u16()? },
        T_ADD => Op::Add { dst: r.u16()?, lhs: r.u16()?, rhs: r.u16()? },
        T_SUB => Op::Sub { dst: r.u16()?, lhs: r.u16()?, rhs: r.u16()? },
        T_MUL => Op::Mul { dst: r.u16()?, lhs: r.u16()?, rhs: r.u16()? },
        T_DIV => Op::Div { dst: r.u16()?, lhs: r.u16()?, rhs: r.u16()? },
        T_MOD => Op::Mod { dst: r.u16()?, lhs: r.u16()?, rhs: r.u16()? },
        T_POW => Op::Pow { dst: r.u16()?, lhs: r.u16()?, rhs: r.u16()? },
        T_EQ => Op::Eq { dst: r.u16()?, lhs: r.u16()?, rhs: r.u16()? },
        T_NEQ => Op::Neq { dst: r.u16()?, lhs: r.u16()?, rhs: r.u16()? },
        T_LOOSE_EQ => Op::LooseEq { dst: r.u16()?, lhs: r.u16()?, rhs: r.u16()? },
        T_LOOSE_NEQ => Op::LooseNeq { dst: r.u16()?, lhs: r.u16()?, rhs: r.u16()? },
        T_LT => Op::Lt { dst: r.u16()?, lhs: r.u16()?, rhs: r.u16()? },
        T_LE => Op::Le { dst: r.u16()?, lhs: r.u16()?, rhs: r.u16()? },
        T_GT => Op::Gt { dst: r.u16()?, lhs: r.u16()?, rhs: r.u16()? },
        T_GE => Op::Ge { dst: r.u16()?, lhs: r.u16()?, rhs: r.u16()? },
        T_BIT_AND => Op::BitAnd { dst: r.u16()?, lhs: r.u16()?, rhs: r.u16()? },
        T_BIT_OR => Op::BitOr { dst: r.u16()?, lhs: r.u16()?, rhs: r.u16()? },
        T_BIT_XOR => Op::BitXor { dst: r.u16()?, lhs: r.u16()?, rhs: r.u16()? },
        T_SHL => Op::Shl { dst: r.u16()?, lhs: r.u16()?, rhs: r.u16()? },
        T_SHR => Op::Shr { dst: r.u16()?, lhs: r.u16()?, rhs: r.u16()? },
        T_USHR => Op::Ushr { dst: r.u16()?, lhs: r.u16()?, rhs: r.u16()? },
        T_NEG => Op::Neg { dst: r.u16()?, src: r.u16()? },
        T_NOT => Op::Not { dst: r.u16()?, src: r.u16()? },
        T_BIT_NOT => Op::BitNot { dst: r.u16()?, src: r.u16()? },
        T_TO_NUMBER => Op::ToNumber { dst: r.u16()?, src: r.u16()? },
        T_TO_STR => Op::ToStr { dst: r.u16()?, src: r.u16()? },
        T_TYPEOF => Op::Typeof { dst: r.u16()?, src: r.u16()? },
        T_IN => Op::In { dst: r.u16()?, lhs: r.u16()?, rhs: r.u16()? },
        T_INSTANCEOF => Op::Instanceof { dst: r.u16()?, lhs: r.u16()?, rhs: r.u16()? },
        T_DELETE_PROP => Op::DeleteProp { dst: r.u16()?, obj: r.u16()?, key_k: r.u16()? },
        T_DELETE_IDX => Op::DeleteIdx { dst: r.u16()?, obj: r.u16()?, key: r.u16()? },
        T_MAKE_REGEX => Op::MakeRegex { dst: r.u16()?, source_k: r.u16()?, flags_k: r.u16()? },
        T_JMP => Op::Jmp { target: r.u16()? },
        T_JMP_IF_FALSE => Op::JmpIfFalse { cond: r.u16()?, target: r.u16()? },
        T_JMP_IF_TRUE => Op::JmpIfTrue { cond: r.u16()?, target: r.u16()? },
        T_CALL_FN => Op::CallFn {
            dst: r.u16()?,
            fn_idx: r.u16()?,
            first_arg: r.u16()?,
            n_args: r.u8()?,
        },
        T_LOAD_GLOBAL => Op::LoadGlobal { dst: r.u16()?, name_k: r.u16()? },
        T_LOAD_GLOBAL_CHECKED => Op::LoadGlobalChecked { dst: r.u16()?, name_k: r.u16()? },
        T_STORE_GLOBAL => Op::StoreGlobal { name_k: r.u16()?, src: r.u16()? },
        T_CALL_VALUE => Op::CallValue {
            dst: r.u16()?,
            callee: r.u16()?,
            this_reg: r.u16()?,
            first_arg: r.u16()?,
            n_args: r.u8()?,
        },
        T_NEW => Op::New {
            dst: r.u16()?,
            ctor: r.u16()?,
            first_arg: r.u16()?,
            n_args: r.u8()?,
        },
        T_LOAD_THIS => Op::LoadThis { dst: r.u16()? },
        T_LOAD_SELF => Op::LoadSelf { dst: r.u16()? },
        T_GET_PROP => Op::GetProp { dst: r.u16()?, obj: r.u16()?, key_k: r.u16()? },
        T_GET_IDX => Op::GetIdx { dst: r.u16()?, obj: r.u16()?, key: r.u16()? },
        T_SET_PROP => Op::SetProp { obj: r.u16()?, key_k: r.u16()?, src: r.u16()? },
        T_SET_IDX => Op::SetIdx { obj: r.u16()?, key: r.u16()?, src: r.u16()? },
        T_NEW_ARRAY => Op::NewArray { dst: r.u16()?, first_elem: r.u16()?, n_elems: r.u8()? },
        T_ARRAY_PUSH => Op::ArrayPush { arr: r.u16()?, val: r.u16()? },
        T_ARRAY_PUSH_SPREAD => Op::ArrayPushSpread { arr: r.u16()?, spread: r.u16()? },
        T_NEW_OBJECT => Op::NewObject { dst: r.u16()? },
        T_THROW => Op::Throw { src: r.u16()? },
        T_TRY_ENTER => Op::TryEnter { catch_target: r.u16()?, catch_reg: r.u16()? },
        T_TRY_EXIT => Op::TryExit,
        T_ENUM_KEYS => Op::EnumKeys { dst: r.u16()?, obj: r.u16()? },
        T_MAKE_CLOSURE => Op::MakeClosure {
            dst: r.u16()?,
            fn_idx: r.u16()?,
            first_upvalue: r.u16()?,
            n_upvalues: r.u8()?,
        },
        T_LOAD_UP => Op::LoadUp { dst: r.u16()?, slot: r.u8()? },
        T_STORE_UP => Op::StoreUp { src: r.u16()?, slot: r.u8()? },
        T_RET => Op::Ret { src: r.u16()? },
        _ => return None,
    })
}

/// Serialize one `BcFunction`. Returns `false` (and leaves a partial write the
/// caller discards) if a const is non-serializable.
fn write_fn(w: &mut Writer, f: &BcFunction) -> bool {
    w.str(&f.name);
    w.u8(f.n_params);
    // rest_reg: a tag + the value (Reg = u16).
    match f.rest_reg {
        Some(r) => {
            w.u8(1);
            w.u16(r);
        }
        None => w.u8(0),
    }
    w.u16(f.n_regs);
    w.u32(f.consts.len() as u32);
    for c in &f.consts {
        if !write_const(w, c) {
            return false;
        }
    }
    w.u32(f.code.len() as u32);
    for op in &f.code {
        write_op(w, op);
    }
    // Warmed IC feedback: the SAME portable (ip, descriptors, mega) list the
    // shape-assumptions digest folds — so the write-time key and a reloaded
    // module's recomputed key are identical by construction. Each guarded shape is
    // written as its DESCRIPTOR (key-sequence), portable across runs (re-interned
    // on load).
    let warm = portable_feedback(f);
    w.u32(warm.len() as u32);
    for (ip, descr, mega) in &warm {
        w.u32(*ip);
        w.u8(*mega as u8);
        w.u32(descr.len() as u32);
        for (keys, slot) in descr {
            w.u32(keys.len() as u32);
            for k in keys {
                w.str(k);
            }
            w.u32(*slot);
        }
    }
    true
}

fn read_fn(r: &mut Reader, code_len_for_ic: &mut usize) -> Option<BcFunction> {
    let name = r.str()?;
    let n_params = r.u8()?;
    let rest_reg = match r.u8()? {
        0 => None,
        1 => Some(r.u16()?),
        _ => return None,
    };
    let n_regs = r.u16()?;
    let n_consts = r.u32()? as usize;
    let mut consts = Vec::with_capacity(n_consts.min(1 << 20));
    for _ in 0..n_consts {
        consts.push(read_const(r)?);
    }
    let n_code = r.u32()? as usize;
    let mut code = Vec::with_capacity(n_code.min(1 << 22));
    for _ in 0..n_code {
        code.push(read_op(r)?);
    }
    *code_len_for_ic = code.len();
    // Warmed IC feedback: re-intern each shape descriptor into THIS run's shape
    // table to recover its run-local ShapeId, then rebuild the PropIc.
    let n_warm = r.u32()? as usize;
    let mut ic_vec: Vec<PropIc> = Vec::new();
    if n_warm > 0 {
        ic_vec = vec![PropIc::INVALID; code.len()];
    }
    for _ in 0..n_warm {
        let ip = r.u32()? as usize;
        let mega = r.u8()? != 0;
        let n_entries = r.u32()? as usize;
        let mut entries: Vec<(u32, u32)> = Vec::with_capacity(n_entries.min(8));
        for _ in 0..n_entries {
            let n_keys = r.u32()? as usize;
            let mut keys: Vec<String> = Vec::with_capacity(n_keys.min(256));
            for _ in 0..n_keys {
                keys.push(r.str()?);
            }
            let slot = r.u32()?;
            // Re-intern the key-sequence to recover this run's ShapeId.
            let shape_id = crate::shapes::with_shape_table(|t| {
                let mut s = t.empty();
                for k in &keys {
                    s = t.add_property(s, k);
                }
                s
            });
            entries.push((shape_id, slot));
        }
        // A corrupt ip would be a silent wrong-slot hazard; bounds-check it.
        if ip < ic_vec.len() {
            ic_vec[ip] = PropIc::from_serialized_own(&entries, mega);
        }
    }
    Some(BcFunction {
        name,
        n_params,
        rest_reg,
        n_regs,
        consts,
        code,
        ic: std::cell::RefCell::new(ic_vec),
        // T4 P1: a freshly deserialized module starts with an EMPTY feedback
        // vector — it lazily re-fills at runtime (clean, monotone-correct). P5
        // will serialize/restore the binary/compare/call hints alongside `ic`;
        // P1 deliberately does not persist them (recording only), so a reload
        // simply re-warms the lattice from scratch.
        feedback: std::cell::RefCell::new(Vec::new()),
    })
}

/// Serialize a `Module` + its warmed feedback + the validation key into a
/// self-describing blob. Returns `None` if any const is non-serializable (we
/// never fabricate — the caller just doesn't cache that module).
pub fn serialize_module(source: &str, module: &Module) -> Option<Vec<u8>> {
    let key = compute_key(source, module);
    let mut w = Writer::new();
    w.u32(MAGIC);
    w.u32(FORMAT_VERSION);
    w.u32(ENGINE_VERSION);
    w.u64(key.0);
    w.u32(module.fns.len() as u32);
    for f in &module.fns {
        if !write_fn(&mut w, f) {
            return None;
        }
    }
    // Script-frame `for (var i = …)` init→global throw-flush map. Persisted so a
    // disk-cache RELOAD keeps the mid-loop-throw global write-back (without it a
    // throwing cached script would diverge from the tree-walker — a silent
    // wrong-global hazard on a cache hit). `(global name, register)` pairs.
    w.u32(module.script_forinit_syncs.len() as u32);
    for (name, reg) in &module.script_forinit_syncs {
        w.str(name);
        w.u16(*reg);
    }
    Some(w.buf)
}

/// Deserialize a blob produced by `serialize_module`, validating it against
/// `expected_key`. Returns `None` on magic/version mismatch, key mismatch
/// (source/flag/shape-assumption drift), or any truncation/corruption — the
/// caller then recompiles from source. The returned `Module`'s IC carries the
/// re-interned warmed feedback so hot functions tier up without re-profiling.
pub fn deserialize_module(blob: &[u8], expected_key: CacheKey) -> Option<Module> {
    let mut r = Reader::new(blob);
    if r.u32()? != MAGIC {
        return None;
    }
    if r.u32()? != FORMAT_VERSION {
        return None;
    }
    if r.u32()? != ENGINE_VERSION {
        return None;
    }
    let stored_key = r.u64()?;
    if stored_key != expected_key.0 {
        return None; // source / flags / shape-assumption drift → reject.
    }
    let n_fns = r.u32()? as usize;
    // Guard against an absurd count from a corrupt header.
    if n_fns > (1 << 20) {
        return None;
    }
    let mut fns = Vec::with_capacity(n_fns.min(1 << 16));
    for _ in 0..n_fns {
        let mut clen = 0usize;
        let f = read_fn(&mut r, &mut clen)?;
        fns.push(f);
    }
    // Script-frame for-init `var` throw-flush map (see `serialize_module`).
    let n_syncs = r.u32()? as usize;
    if n_syncs > (1 << 16) {
        return None; // absurd count from a corrupt blob
    }
    let mut script_forinit_syncs = Vec::with_capacity(n_syncs.min(1 << 12));
    for _ in 0..n_syncs {
        let name = r.str()?;
        let reg = r.u16()?;
        script_forinit_syncs.push((name, reg));
    }
    Some(Module {
        fns,
        script_forinit_syncs,
    })
}

// ----------------------------------------------------------------------
// Disk store.
// ----------------------------------------------------------------------

/// The on-disk cache directory (`<temp>/tbjs_code_cache`). Created on demand.
/// A dedicated subdir keeps the entries together and out of the way.
fn cache_dir() -> PathBuf {
    if let Ok(custom) = std::env::var("CV_CODE_CACHE_DIR") {
        return PathBuf::from(custom);
    }
    std::env::temp_dir().join("tbjs_code_cache")
}

/// The on-disk filename for a source, keyed by a SOURCE-ONLY hash (so a changed
/// source lands in a different file; the in-blob key then validates flags + shape
/// assumptions). Hex-encoded to be filesystem-safe.
fn cache_filename(source: &str) -> String {
    let mut h = Fnv1a::new();
    h.write(b"tbcc-file");
    h.write_u32(ENGINE_VERSION);
    h.write_u64(source.len() as u64);
    h.write(source.as_bytes());
    format!("{:016x}.tbcc", h.finish())
}

/// Try to LOAD a cached module for `source`. Returns `None` (→ caller compiles)
/// when the cache is disabled, the file is missing, or the blob fails validation
/// (`validate_and_deserialize`). The filename is a source-only hash (so a changed
/// source lands in a different file); the full source+flag+shape-digest key inside
/// the blob is what actually validates the entry, so a hash collision or a flag/
/// shape drift is still rejected.
pub fn load(source: &str) -> Option<Module> {
    if !code_cache_enabled() {
        return None;
    }
    let path = cache_dir().join(cache_filename(source));
    let blob = std::fs::read(&path).ok()?;
    validate_and_deserialize(source, &blob)
}

/// Validate a blob against `source` + current flags and deserialize it. Split out
/// for testability (no disk). The validation: the blob's stored full key must
/// equal the key recomputed from `source`, the current flags, AND the shape
/// digest of the DESERIALIZED module (which carries the same descriptors that
/// produced the stored key). If anything drifted (source edit, flag change,
/// engine bump, format change, truncation), it fails closed → `None`.
pub fn validate_and_deserialize(source: &str, blob: &[u8]) -> Option<Module> {
    // Peek the header to get the stored key without trusting the body yet.
    let mut peek = Reader::new(blob);
    if peek.u32()? != MAGIC {
        return None;
    }
    if peek.u32()? != FORMAT_VERSION {
        return None;
    }
    if peek.u32()? != ENGINE_VERSION {
        return None;
    }
    let stored_key = CacheKey(peek.u64()?);
    // Deserialize against the stored key (this enforces the stored key matches
    // itself + the structural integrity of the blob, and re-interns the IC).
    let module = deserialize_module(blob, stored_key)?;
    // Now RE-DERIVE the expected key from the live source + current flags + the
    // shape assumptions of the just-loaded module. If the source changed, the
    // flags changed, the engine bumped, or the (re-interned) shape assumptions
    // don't reproduce the stored key, REJECT — recompile instead.
    let expected = compute_key(source, &module);
    if expected != stored_key {
        return None;
    }
    Some(module)
}

/// STORE a freshly-compiled module for `source` to disk. No-op (silently) when
/// the cache is disabled, the module has a non-serializable const, or the disk
/// write fails — storing is best-effort and never affects correctness.
pub fn store(source: &str, module: &Module) {
    if !code_cache_enabled() {
        return;
    }
    let Some(blob) = serialize_module(source, module) else {
        return;
    };
    let dir = cache_dir();
    if std::fs::create_dir_all(&dir).is_err() {
        return;
    }
    let path = dir.join(cache_filename(source));
    // Write to a temp file then rename, so a crash mid-write can't leave a
    // half-written (corrupt-but-validates-magic) blob that a later run trusts.
    let tmp = dir.join(format!("{}.tmp", cache_filename(source)));
    if std::fs::write(&tmp, &blob).is_ok() {
        let _ = std::fs::rename(&tmp, &path);
    }
}

/// THE production seam (gated by `CV_CODE_CACHE`, DEFAULT OFF). Returns a runnable
/// `Module` for `source`, preferring a validated on-disk entry over a fresh
/// parse+compile:
///   1. cache ENABLED + a valid on-disk entry exists → return it (NO parse, NO
///      compile, and the warmed feedback is pre-seeded) — the win.
///   2. otherwise → `bytecode::compile_program(source)`. On success, and with the
///      cache enabled, WRITE the fresh module to disk for the next run (best-effort).
///
/// When the cache is DISABLED this is exactly `compile_program` — byte-identical
/// to today, no disk touch. A corrupt/stale/mismatched on-disk entry is rejected
/// inside `load`, so the worst case is a recompile (never a wrong module). The
/// returned module is observationally identical to a freshly-compiled one in all
/// cases (a cache hit returns the same bytecode that produced the blob).
pub fn compile_program_cached(source: &str) -> Result<Module, crate::bytecode::CompileError> {
    if code_cache_enabled() {
        if let Some(m) = load(source) {
            return Ok(m);
        }
    }
    let module = crate::bytecode::compile_program(source)?;
    // Best-effort persist for the next run (no-op when disabled / non-serializable).
    store(source, &module);
    Ok(module)
}

#[cfg(test)]
mod tests;
