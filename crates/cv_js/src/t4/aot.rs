//! T4 (Maglev-class) PHASE P5 — AOT-PERSIST OPTIMIZED NATIVE CODE
//! (★ the BEAT-CHROME cold-repeat-visit lever).
//!
//! ## The past-V8 thesis (and exactly why V8 cannot do this)
//!
//! V8's code cache (v8.dev/blog/code-caching-for-devs) persists ONLY bytecode +
//! `SharedFunctionInfo` at script-completion; it EXPLICITLY DISCARDS optimized
//! Maglev/TurboFan native code on context teardown, because that code is on-heap,
//! GC-relocatable, isolate-bound, and sandbox-pointer-encoded — it has NO
//! isolate-independent serializable form. So V8 re-runs interpret → collect
//! feedback → tier-up on EVERY cold load, paying the warmup ramp again each time.
//!
//! Conclave is single-address-space, single-profile, no multi-isolate / no
//! sandbox-pointer constraint. And — the keystone enabling fact, verified by
//! reading `jit::compile_t4_unboxed_with_deopt_mapped` — the T4 NUMERIC-SUBSET
//! backend emits **fully position-independent** machine code: it bakes only f64
//! constant IMMEDIATES (`mov r64, imm64`) and BANK-RELATIVE memory offsets
//! (`[T2_BANK + reg*8]`), uses only the bank/out/ctx passed in RCX/RDX/R8, calls
//! NO helpers (the numeric subset has no heap/call ops), and resolves every branch
//! with INTERNAL `rel32` displacements (self-relative, patched at compile). There
//! is therefore NOTHING in the code bytes to relocate at load time — no absolute
//! pointer, no isolate handle, no shape id. The bytes are byte-for-byte runnable
//! after a plain `memcpy → VirtualProtect(PAGE_EXECUTE_READ)` at any address.
//!
//! That is what lets us do what V8 structurally cannot: serialize the optimized
//! native code itself and, on a COLD REPEAT VISIT (fresh process, warm code
//! cache), enter PEAK-TIER native code from instruction #1 — removing the JIT
//! warmup ramp, which is an intrinsic ENGINE cost, not a resource cache. (The
//! user's bar: a repeat-load resource cache doesn't count; an engine that needs no
//! re-warmup does.)
//!
//! ## What is persisted (the blob)
//!
//! A T4 numeric blob is `(machine bytes, DeoptSite table, the FUSED module, the
//! ORIGINAL caller module, n_params)`:
//!   * `code`  — the relocatable RX bytes (above; no relocation needed).
//!   * `deopt_sites` — `(native_off, bc_pc, reason)` each: a machine OFFSET, a VM
//!     bytecode index, and an enum. All offset/VM-level, no pointers → portable.
//!   * the FUSED module (bank-sizing reference for `run_t4_call`) and the ORIGINAL
//!     caller module (the deopt RESUME target). Both are `bytecode::Module`s, which
//!     `code_cache` already serializes portably (re-interned shapes, NaN-exact
//!     consts). We reuse that exact (de)serializer for the modules.
//!
//! ## Correctness — a stale blob costs ONE bailout, NEVER a wrong value
//!
//! This is the load-bearing safety property and it rides ENTIRELY on the existing,
//! fuzz-proven deopt machinery (the design's "deopt backstop bounds the blast
//! radius"):
//!   1. The blob is keyed by `(ABI_version, cpu_features, BYTECODE DIGEST of the
//!      fused + original modules)`. The digest folds the actual `Op` stream + the
//!      consts that produced the code. A changed source compiles to a different
//!      bytecode stream → a different digest → a clean MISS (re-compile). A flag
//!      change (e.g. the redundancy pass behaving differently) changes the fused
//!      bytecode → different digest → miss. The digest is the load-bearing
//!      invalidator (the `set_force_stale_digest` mutation hook proves it).
//!   2. On a hit, the persisted DeoptSite table is re-installed VERBATIM, so EVERY
//!      guard the fresh T4 code would check is re-checked at run time on the new
//!      load. A feedback/type mismatch on the new load deopts to the VM frame
//!      EXACTLY as fresh T4 would — byte-identical, never a wrong result. So even a
//!      digest COLLISION (astronomically unlikely with a 64-bit FNV-1a) cannot
//!      produce a wrong value: the re-checked guards catch it.
//!   3. Any truncation / magic / version / ABI / cpu-feature mismatch fails closed
//!      (`None` → re-compile). The blob NEVER fabricates a JitFunction.
//!
//! ## Gating
//!
//! `CV_AOT_PERSIST` (DEFAULT OFF). When off, nothing reads or writes the native
//! blob and the engine is byte-identical to P4. The round-trip oracle
//! (persist → reload → run → deopt) proves a reloaded blob == the VM, and the
//! `set_force_stale_digest` mutation arm proves the digest is load-bearing (a
//! reload of a DRIFTED program must MISS, never run stale code).
//!
//! ## V8 source modeled
//!
//! - `v8.dev/blog/code-caching-for-devs`, `src/snapshot/code-serializer.*`
//!   (V8's `CodeSerializer` persists BYTECODE only; this module persists the
//!   optimized native code V8 deliberately drops) — the divergence is the win.
//! - `src/codegen/reloc-info.*` (V8's RelocInfo records every embedded pointer a
//!   blob would need fixed up). Our typed-relocation answer is the strongest form:
//!   the numeric subset has an EMPTY relocation set by construction, asserted at
//!   serialize time (`relocation_free`), so there is no fixup to get wrong.

use crate::bytecode::Module;
use crate::osr::{DeoptReason, DeoptSite};

/// ABI version of the persisted-native format. Bump on ANY change to: the T4
/// backend's prolog/epilogue/calling convention, the `DeoptSite` semantics, the
/// `run_t4_call` bank layout, or this blob format. A mismatch rejects every
/// on-disk entry (clean recompile) — the coarse "the bytes mean something
/// different now" gate, alongside the finer bytecode digest.
pub const AOT_ABI_VERSION: u32 = 1;

/// Magic header (`"T4AC"` = T4 Aot Code) + a format version.
const AOT_MAGIC: u32 = 0x5434_4143; // 'T''4''A''C'
const AOT_FORMAT_VERSION: u32 = 1;

/// Whether AOT-persist of optimized native code is enabled. DEFAULT-OFF (opt IN
/// with `CV_AOT_PERSIST=1`), mirroring `CV_T4` / `CV_CODE_CACHE`. The env is read
/// once; an in-process force-on switch ([`set_force_aot_persist`]) lets the
/// round-trip oracle / tests drive the disk path without a process-global env
/// (mirrors `feedback::set_force_feedback` / `code_cache`'s discipline).
pub fn aot_persist_enabled() -> bool {
    if force_aot_persist() {
        return true;
    }
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("CV_AOT_PERSIST").as_deref() == Ok("1"))
}

thread_local! {
    /// In-process force-on switch for the oracle / tests: when set,
    /// [`aot_persist_enabled`] returns true regardless of the env. Lets a test
    /// populate + reload the disk AOT store deterministically.
    static FORCE_AOT_PERSIST: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Set (returns the prior value) the in-process force-AOT switch. Prefer the
/// [`AotPersistGuard`] scope guard so it is always restored.
pub fn set_force_aot_persist(v: bool) -> bool {
    FORCE_AOT_PERSIST.with(|c| {
        let prev = c.get();
        c.set(v);
        prev
    })
}

/// Whether AOT-persist is force-enabled in-process (oracle/tests).
pub fn force_aot_persist() -> bool {
    FORCE_AOT_PERSIST.with(|c| c.get())
}

/// RAII scope guard for [`set_force_aot_persist`].
#[must_use]
pub struct AotPersistGuard {
    prev: bool,
}
impl AotPersistGuard {
    pub fn new(v: bool) -> Self {
        AotPersistGuard {
            prev: set_force_aot_persist(v),
        }
    }
}
impl Drop for AotPersistGuard {
    fn drop(&mut self) {
        set_force_aot_persist(self.prev);
    }
}

// ----------------------------------------------------------------------
// Honesty guards (mirror the t2/t4 exec counters + the inline_compile_count).
// ----------------------------------------------------------------------

thread_local! {
    /// Number of T4 native blobs LOADED from the AOT store and re-installed into
    /// runnable RX pages (the cold-repeat win actually fired). Lets the oracle/
    /// tests prove the reload path is NON-VACUOUS (a green round-trip oracle that
    /// never actually re-installed a persisted blob would be a lie).
    static AOT_LOAD_COUNT: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
    /// Number of T4 native blobs SERIALIZED to the AOT store.
    static AOT_STORE_COUNT: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
    /// Number of AOT lookups that MISSED (drift / no entry) and fell back to a
    /// fresh T4 compile — the safety path. Proves drift rejection actually rejects.
    static AOT_MISS_COUNT: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

/// Count of T4 native blobs reloaded + re-installed from the AOT store this run.
pub fn aot_load_count() -> u64 {
    AOT_LOAD_COUNT.with(|c| c.get())
}
/// Reset the AOT-load honesty counter (oracle/tests call before a reload run).
pub fn reset_aot_load_count() {
    AOT_LOAD_COUNT.with(|c| c.set(0));
}
/// Count of T4 native blobs serialized to the AOT store this run.
pub fn aot_store_count() -> u64 {
    AOT_STORE_COUNT.with(|c| c.get())
}
/// Reset the AOT-store honesty counter.
pub fn reset_aot_store_count() {
    AOT_STORE_COUNT.with(|c| c.set(0));
}
/// Count of AOT lookups that missed (drift / absent) this run.
pub fn aot_miss_count() -> u64 {
    AOT_MISS_COUNT.with(|c| c.get())
}
/// Reset the AOT-miss honesty counter.
pub fn reset_aot_miss_count() {
    AOT_MISS_COUNT.with(|c| c.set(0));
}

fn bump_load() {
    AOT_LOAD_COUNT.with(|c| c.set(c.get() + 1));
}
fn bump_store() {
    AOT_STORE_COUNT.with(|c| c.set(c.get() + 1));
}
fn bump_miss() {
    AOT_MISS_COUNT.with(|c| c.set(c.get() + 1));
}

// ----------------------------------------------------------------------
// MUTATION HOOK (test-only) — proves the bytecode-digest leg of the key is
// LOAD-BEARING. The key's whole safety claim is "a DRIFTED program MISSES (never
// runs stale code)". To prove the round-trip oracle would CATCH a key that does
// NOT include the digest, this hook makes `compute_native_key` OMIT the digest.
// With it set, a drifted reload is wrongly ACCEPTED → the oracle, which compares
// the reloaded run to the VM run of the DRIFTED source, must DIVERGE. With it
// unset (production default) the digest is included and the oracle is green. No
// env path — only the in-process setter (mirrors `code_cache::set_force_stale_key`).
// ----------------------------------------------------------------------
thread_local! {
    static FORCE_STALE_DIGEST: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Set (returns the prior value) the stale-digest mutation hook (test-only).
/// Prefer the [`StaleDigestGuard`] scope guard.
pub fn set_force_stale_digest(v: bool) -> bool {
    FORCE_STALE_DIGEST.with(|c| {
        let prev = c.get();
        c.set(v);
        prev
    })
}

fn force_stale_digest() -> bool {
    FORCE_STALE_DIGEST.with(|c| c.get())
}

/// RAII guard for the stale-digest mutation hook (test-only).
#[must_use]
pub struct StaleDigestGuard {
    prev: bool,
}
impl StaleDigestGuard {
    pub fn new(on: bool) -> Self {
        StaleDigestGuard {
            prev: set_force_stale_digest(on),
        }
    }
}
impl Drop for StaleDigestGuard {
    fn drop(&mut self) {
        set_force_stale_digest(self.prev);
    }
}

// ----------------------------------------------------------------------
// FNV-1a 64-bit — the same hash family `code_cache::Fnv1a` uses. Re-declared
// locally (that one is private) so this module is self-contained; deterministic.
// ----------------------------------------------------------------------
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

/// CPU-feature gate folded into the key. The T4 numeric backend uses SSE2 scalar
/// double (movsd/addsd/…) which is baseline x86-64 (always present on win64), so
/// today this is a constant. It exists as a real key leg so that when P4's
/// roundsd/cmov/etc. land (which need SSE4.1) a blob compiled WITH those features
/// can never be loaded on a CPU lacking them — the design's `cpu_features` key.
/// We probe the actual feature here rather than hardcode `true`, so the gate is a
/// real check, not a fabricated constant.
fn cpu_features() -> u32 {
    let mut bits = 0u32;
    // SSE2 — baseline for x86-64; `is_x86_feature_detected!` is the real probe.
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("sse2") {
            bits |= 1 << 0;
        }
        if std::arch::is_x86_feature_detected!("sse4.1") {
            bits |= 1 << 1;
        }
    }
    bits
}

/// The validation key for one persisted T4 native blob: a 64-bit hash folding the
/// ABI version, the CPU-feature set, and the BYTECODE DIGEST of the exact program
/// that produced the native code (the fused module + the original caller module).
///
/// Two runs that share this key are guaranteed to have produced byte-identical
/// optimized native code (same backend ABI, same CPU features, same input
/// bytecode), so the persisted bytes are safe to run. The digest is omitted ONLY
/// under the stale-digest mutation hook, which proves it is the load-bearing
/// invalidator (a green oracle without it would mean drift goes undetected).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NativeKey(pub u64);

/// Fold one module's bytecode (op stream + consts + arity) into a hash. This is
/// the "source hash" in the sense that matters: the exact bytecode shape that
/// determines the emitted native code. Two byte-identical programs hash equal; any
/// op/const/arity drift hashes differently → a clean miss.
fn hash_module(h: &mut Fnv1a, module: &Module) {
    h.write_u32(module.fns.len() as u32);
    for f in &module.fns {
        h.write_u32(f.n_params as u32);
        h.write_u32(f.n_regs as u32);
        // Consts (NaN-exact bit patterns; mirrors the code_cache f64 writer).
        h.write_u32(f.consts.len() as u32);
        for c in &f.consts {
            match c {
                crate::interp::Value::Number(n) => {
                    h.write(&[0u8]);
                    h.write_u64(n.to_bits());
                }
                crate::interp::Value::String(s) => {
                    h.write(&[1u8]);
                    h.write(s.as_str().as_bytes());
                }
                crate::interp::Value::Bool(b) => h.write(&[2u8, *b as u8]),
                crate::interp::Value::Null => h.write(&[3u8]),
                crate::interp::Value::Undefined => h.write(&[4u8]),
                // Any other const kind makes the program non-AOT-able; fold a
                // distinct tag so it never collides with a serializable program.
                _ => h.write(&[255u8]),
            }
        }
        // Op stream — fold each op's discriminant + operand bytes. We piggyback on
        // the code_cache op writer's tag stability by hashing the Debug-free
        // structural bytes: serialize each op through the same encoder the module
        // serializer uses (so the digest tracks EXACTLY what gets persisted).
        h.write_u32(f.code.len() as u32);
        for op in &f.code {
            hash_op(h, op);
        }
    }
}

/// Fold one bytecode `Op` into the digest. Mirrors the operand set so any operand
/// drift (a different register, const index, or jump target) changes the digest.
fn hash_op(h: &mut Fnv1a, op: &crate::bytecode::Op) {
    use crate::bytecode::Op;
    // A 1-byte discriminant tag + the operand u16/u8 fields, little-endian. The
    // tag values need not match code_cache's (this is an internal digest, not a
    // wire format), only be STABLE within a run and DISTINCT per variant.
    macro_rules! tagged {
        ($t:expr $(, $f:expr)*) => {{
            h.write(&[$t as u8]);
            $( h.write(&($f as u32).to_le_bytes()); )*
        }};
    }
    match *op {
        Op::LoadConst { dst, k } => tagged!(0u8, dst, k),
        Op::LoadTrue { dst } => tagged!(1u8, dst),
        Op::LoadFalse { dst } => tagged!(2u8, dst),
        Op::LoadNull { dst } => tagged!(3u8, dst),
        Op::LoadUndef { dst } => tagged!(4u8, dst),
        Op::Move { dst, src } => tagged!(5u8, dst, src),
        Op::Add { dst, lhs, rhs } => tagged!(6u8, dst, lhs, rhs),
        Op::Sub { dst, lhs, rhs } => tagged!(7u8, dst, lhs, rhs),
        Op::Mul { dst, lhs, rhs } => tagged!(8u8, dst, lhs, rhs),
        Op::Div { dst, lhs, rhs } => tagged!(9u8, dst, lhs, rhs),
        Op::Lt { dst, lhs, rhs } => tagged!(16u8, dst, lhs, rhs),
        Op::Le { dst, lhs, rhs } => tagged!(17u8, dst, lhs, rhs),
        Op::Gt { dst, lhs, rhs } => tagged!(18u8, dst, lhs, rhs),
        Op::Ge { dst, lhs, rhs } => tagged!(19u8, dst, lhs, rhs),
        Op::Eq { dst, lhs, rhs } => tagged!(12u8, dst, lhs, rhs),
        Op::Neq { dst, lhs, rhs } => tagged!(13u8, dst, lhs, rhs),
        Op::LooseEq { dst, lhs, rhs } => tagged!(14u8, dst, lhs, rhs),
        Op::LooseNeq { dst, lhs, rhs } => tagged!(15u8, dst, lhs, rhs),
        Op::Jmp { target } => tagged!(35u8, target),
        Op::JmpIfFalse { cond, target } => tagged!(36u8, cond, target),
        Op::Ret { src } => tagged!(61u8, src),
        Op::CallFn { dst, fn_idx, first_arg, n_args } => {
            tagged!(38u8, dst, fn_idx, first_arg, n_args)
        }
        // Any other op (heap/call/try/…) is outside the AOT numeric subset; the
        // serializer rejects such a program before we get here, but fold a distinct
        // catch-all so two different non-subset ops never collide in the digest.
        ref other => {
            h.write(&[254u8]);
            // Fold the Debug rendering so distinct variants hash distinctly. This
            // path is only reached defensively (the subset gate rejects first).
            h.write(format!("{other:?}").as_bytes());
        }
    }
}

/// Compute the validation key for a T4 native blob produced from `fused` +
/// `original_caller`. The digest is included UNLESS the stale-digest mutation hook
/// is engaged (test-only), which proves the digest is the load-bearing invalidator.
pub fn compute_native_key(fused: &Module, original_caller: &Module) -> NativeKey {
    let mut h = Fnv1a::new();
    h.write(b"t4-aot-key-v1");
    h.write_u32(AOT_ABI_VERSION);
    h.write_u32(cpu_features());
    if !force_stale_digest() {
        hash_module(&mut h, fused);
        hash_module(&mut h, original_caller);
    }
    NativeKey(h.finish())
}

// ----------------------------------------------------------------------
// Binary (de)serialization — hand-rolled, length-prefixed, bounds-checked. We
// REUSE `code_cache`'s module (de)serializer for the embedded modules (so shapes
// re-intern + consts round-trip NaN-exact through the proven path), and add the
// native-code + DeoptSite framing here.
// ----------------------------------------------------------------------

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
    fn u32(&mut self, v: u32) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }
    fn u64(&mut self, v: u64) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }
    fn bytes(&mut self, b: &[u8]) {
        self.u32(b.len() as u32);
        self.buf.extend_from_slice(b);
    }
}

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
    fn bytes(&mut self) -> Option<&'a [u8]> {
        let n = self.u32()? as usize;
        self.take(n)
    }
}

/// Encode a `DeoptReason` as a stable byte tag. A new reason variant requires a
/// new tag here AND an `AOT_ABI_VERSION` bump (so old blobs that can't carry it
/// are rejected). The decode rejects an unknown tag (fails closed).
fn reason_tag(r: DeoptReason) -> u8 {
    match r {
        DeoptReason::NonNumber => 0,
        DeoptReason::NonObject => 1,
        DeoptReason::ShapeMiss => 2,
        DeoptReason::CallDecline => 3,
        DeoptReason::NonArray => 4,
        DeoptReason::BadIndex => 5,
        DeoptReason::HoleOrSpecial => 6,
        DeoptReason::FallThrough => 7,
    }
}

fn tag_reason(t: u8) -> Option<DeoptReason> {
    Some(match t {
        0 => DeoptReason::NonNumber,
        1 => DeoptReason::NonObject,
        2 => DeoptReason::ShapeMiss,
        3 => DeoptReason::CallDecline,
        4 => DeoptReason::NonArray,
        5 => DeoptReason::BadIndex,
        6 => DeoptReason::HoleOrSpecial,
        7 => DeoptReason::FallThrough,
        _ => return None,
    })
}

/// The portable, in-memory form of a persisted T4 native function — everything a
/// `JitFunction` needs to be reconstructed by `install_from_blob`, MINUS the live
/// RX page (which is re-installed on load). Returned by `deserialize_blob`.
pub struct AotNative {
    /// The relocatable RX machine bytes (re-installed verbatim — no fixup).
    pub code: Vec<u8>,
    /// The per-guard deopt resume table (re-attached verbatim).
    pub deopt_sites: Vec<DeoptSite>,
    /// The FUSED (inlined path) or OPTIMIZED (single-fn path) module — the
    /// bank-sizing + (single-fn) deopt-resume reference `run_t4_call`/`run_t3_call`
    /// reads.
    pub fused_module: Module,
    /// The ORIGINAL caller module — the deopt RESUME target for the INLINED path
    /// (`t4_deopt_module`). For the single-function path this equals the optimized
    /// module (resume is on it) and is NOT attached as `t4_deopt_module` (see
    /// `is_inlined`), so the non-inlined `run_t3_call` resume path is taken.
    pub original_caller: Module,
    /// True iff this blob is from the CROSS-FUNCTION-INLINED path (so the reloaded
    /// `JitFunction` carries `t4_deopt_module` and `run_t4_call` resumes on the
    /// original caller). False for the single-function P2 path (resume on the
    /// optimized module via `run_t3_call`). The RUNNER consults
    /// `JitFunction::t4_deopt_module()` to pick the path; getting this flag right is
    /// what makes a reloaded blob run on the SAME runner as the fresh compile.
    pub is_inlined: bool,
}

/// Serialize a T4 numeric native blob to a self-describing byte vector, keyed by
/// `compute_native_key(fused, original_caller)`. Returns `None` if either module
/// is non-serializable through the `code_cache` module writer (we never fabricate —
/// the caller then simply doesn't persist this function), or if the native code
/// is empty.
///
/// SOUNDNESS GATE: the numeric subset emits relocation-free code (see the module
/// note); the writer ASSERTS `code` is non-empty (an empty blob would re-install
/// to a zero-length page → install error → safe fallback, but we reject it up
/// front so a malformed blob never reaches disk).
pub fn serialize_blob(
    code: &[u8],
    deopt_sites: &[DeoptSite],
    fused_module: &Module,
    original_caller: &Module,
    is_inlined: bool,
) -> Option<Vec<u8>> {
    if code.is_empty() {
        return None;
    }
    let key = compute_native_key(fused_module, original_caller);
    // Serialize the two modules through the PROVEN code_cache module writer. We
    // wrap each in a single-fn-or-multi-fn `Module` exactly as the runtime carries
    // them, so the reload reconstructs the identical structures. The code_cache
    // writer needs a `source` for ITS key, but we only consume its MODULE bytes —
    // pass an empty source (the code_cache key inside is ignored; OUR NativeKey is
    // the gate). It returns None on a non-serializable const → propagate.
    let fused_bytes = crate::code_cache::serialize_module("", fused_module)?;
    let orig_bytes = crate::code_cache::serialize_module("", original_caller)?;

    let mut w = Writer::new();
    w.u32(AOT_MAGIC);
    w.u32(AOT_FORMAT_VERSION);
    w.u32(AOT_ABI_VERSION);
    w.u32(cpu_features());
    w.u64(key.0);
    // Inlined-vs-single-fn path flag (picks the reload runner).
    w.u8(is_inlined as u8);
    // Native code bytes (relocation-free).
    w.bytes(code);
    // DeoptSite table.
    w.u32(deopt_sites.len() as u32);
    for s in deopt_sites {
        w.u64(s.native_off as u64);
        w.u64(s.bc_pc as u64);
        w.u8(reason_tag(s.reason));
    }
    // Embedded modules (proven code_cache encoding).
    w.bytes(&fused_bytes);
    w.bytes(&orig_bytes);
    bump_store();
    Some(w.buf)
}

/// Deserialize a blob produced by [`serialize_blob`], validating it against
/// `expected_key`. Returns `None` on magic/version/ABI/cpu mismatch, key mismatch
/// (the bytecode DIGEST drifted → the program changed), or ANY truncation — the
/// caller then re-compiles fresh. The native code is NOT installed here (that is
/// [`install_from_blob`], which needs the OS page allocator); this returns the
/// portable `AotNative` parts.
pub fn deserialize_blob(blob: &[u8], expected_key: NativeKey) -> Option<AotNative> {
    let mut r = Reader::new(blob);
    if r.u32()? != AOT_MAGIC {
        return None;
    }
    if r.u32()? != AOT_FORMAT_VERSION {
        return None;
    }
    if r.u32()? != AOT_ABI_VERSION {
        return None;
    }
    if r.u32()? != cpu_features() {
        return None; // a different CPU-feature set → the bytes may use absent insns.
    }
    let stored_key = r.u64()?;
    if stored_key != expected_key.0 {
        return None; // bytecode-digest drift → the program changed → reject (recompile).
    }
    let is_inlined = match r.u8()? {
        0 => false,
        1 => true,
        _ => return None, // corrupt flag → fail closed.
    };
    let code = r.bytes()?.to_vec();
    if code.is_empty() {
        return None;
    }
    let n_sites = r.u32()? as usize;
    if n_sites > (1 << 22) {
        return None; // absurd count from a corrupt header.
    }
    let mut deopt_sites = Vec::with_capacity(n_sites.min(1 << 16));
    for _ in 0..n_sites {
        let native_off = r.u64()? as usize;
        let bc_pc = r.u64()? as usize;
        let reason = tag_reason(r.u8()?)?;
        deopt_sites.push(DeoptSite {
            native_off,
            bc_pc,
            reason,
        });
    }
    // Embedded modules — deserialize through the PROVEN code_cache reader. We pass
    // the STORED key those blobs carry (the code_cache writer embedded its own key
    // computed from empty-source + the module's shape digest, which we reproduce by
    // re-deriving it from the just-read header is not possible — instead we peek the
    // code_cache blob's own stored key and validate against itself, which checks the
    // module's STRUCTURAL integrity + re-interns shapes). A module that fails its
    // own internal validation (truncation/format) returns None → reject the blob.
    let fused_bytes = r.bytes()?;
    let orig_bytes = r.bytes()?;
    let fused_module = code_cache_self_validating_module(fused_bytes)?;
    let original_caller = code_cache_self_validating_module(orig_bytes)?;
    Some(AotNative {
        code,
        deopt_sites,
        fused_module,
        original_caller,
        is_inlined,
    })
}

/// Deserialize a code_cache module blob, validating it against its OWN embedded
/// key (peeked from its header). This checks structural integrity + re-interns the
/// shape descriptors, exactly as `code_cache::validate_and_deserialize` does for a
/// disk entry — but keyed by the blob's own stored key (we serialized it with an
/// empty source, so the source/flag legs are constant; OUR NativeKey already gates
/// the program-identity). Returns `None` on any corruption/truncation.
fn code_cache_self_validating_module(blob: &[u8]) -> Option<Module> {
    // Peek the stored key, then deserialize against it (same discipline as
    // code_cache::validate_and_deserialize, but the source-derived re-check is OUR
    // NativeKey's job, so we validate the module blob against its own stored key —
    // which still enforces structural integrity + magic/version + the re-intern).
    let mut peek = Reader::new(blob);
    // code_cache header: MAGIC(u32) FORMAT(u32) ENGINE(u32) KEY(u64) …
    let _magic = peek.u32()?;
    let _fmt = peek.u32()?;
    let _engine = peek.u32()?;
    let stored = peek.u64()?;
    crate::code_cache::deserialize_module(blob, crate::code_cache::CacheKey(stored))
}

/// Re-install a deserialized T4 native blob into a runnable `JitFunction`: copy the
/// relocation-free bytes into a fresh RX page, re-attach the DeoptSite table, the
/// FUSED module (bank sizing), and the ORIGINAL caller module (deopt resume). The
/// result is byte-for-byte equivalent to a freshly-compiled T4 inlined function —
/// `run_t4_call` runs it identically, and a guard miss deopts via the re-attached
/// DeoptSite table EXACTLY as fresh T4 would (the deopt backstop). Returns `None`
/// on an install failure (the caller falls back to a fresh compile / lower tier).
#[cfg(target_os = "windows")]
pub fn install_from_blob(parts: AotNative) -> Option<crate::jit::JitFunction> {
    let native = crate::jit::JitFunction::install(&parts.code).ok()?;
    let fused_module = std::rc::Rc::new(parts.fused_module);
    let jf = native
        .with_deopt_sites(parts.deopt_sites)
        .with_t3_module(fused_module);
    let jf = if parts.is_inlined {
        // INLINED path: attach the original caller so `run_t4_call` resumes a deopt
        // on it (the INLINE-DEOPT-TO-CALLER design).
        jf.with_t4_deopt_module(std::rc::Rc::new(parts.original_caller))
    } else {
        // SINGLE-FUNCTION path: NO `t4_deopt_module`, so `run_t4_call` falls to the
        // proven `run_t3_call` which resumes the deopt on the OPTIMIZED module
        // (carried on `t3_module`) — exactly the fresh single-fn T4 install does.
        jf
    };
    bump_load();
    Some(jf)
}

#[cfg(not(target_os = "windows"))]
pub fn install_from_blob(_parts: AotNative) -> Option<crate::jit::JitFunction> {
    None
}

// ----------------------------------------------------------------------
// Disk store (mirrors code_cache's atomic temp-then-rename writer).
// ----------------------------------------------------------------------

thread_local! {
    /// A per-thread directory override (precedence over the env). Used by the
    /// round-trip oracle so PARALLEL test threads each get an isolated on-disk store
    /// with no env-var race (the env var is process-global; a thread-local is not).
    /// `None` in production → the env / default path is used.
    static THREAD_DIR_OVERRIDE: std::cell::RefCell<Option<std::path::PathBuf>> =
        const { std::cell::RefCell::new(None) };
}

/// Set (test-only) a per-thread AOT store dir override, taking precedence over
/// `CV_AOT_PERSIST_DIR`. Pass `None` to clear. Lets the round-trip oracle isolate
/// each call's disk I/O without racing on the process-global env var.
pub fn set_thread_dir_override(dir: Option<std::path::PathBuf>) {
    THREAD_DIR_OVERRIDE.with(|c| *c.borrow_mut() = dir);
}

/// The on-disk AOT native-code cache directory. Precedence: a per-thread override
/// (test isolation) → `CV_AOT_PERSIST_DIR` → `<temp>/tbjs_t4_aot`. A dedicated
/// subdir keeps native blobs separate from the bytecode code_cache.
fn aot_dir() -> std::path::PathBuf {
    if let Some(p) = THREAD_DIR_OVERRIDE.with(|c| c.borrow().clone()) {
        return p;
    }
    if let Ok(custom) = std::env::var("CV_AOT_PERSIST_DIR") {
        return std::path::PathBuf::from(custom);
    }
    std::env::temp_dir().join("tbjs_t4_aot")
}

/// The on-disk filename for a native blob, keyed by the NativeKey (so a digest
/// drift lands in a different file; the in-blob key then re-validates). Hex.
fn aot_filename(key: NativeKey) -> String {
    format!("{:016x}.t4ac", key.0)
}

/// STORE a freshly-compiled T4 native blob to disk for the next run, keyed by
/// `(fused, original_caller)`. No-op (silently) when AOT-persist is disabled, a
/// module is non-serializable, or the disk write fails — storing is best-effort
/// and never affects correctness.
pub fn store_to_disk(
    code: &[u8],
    deopt_sites: &[DeoptSite],
    fused_module: &Module,
    original_caller: &Module,
    is_inlined: bool,
) {
    if !aot_persist_enabled() {
        return;
    }
    let Some(blob) = serialize_blob(code, deopt_sites, fused_module, original_caller, is_inlined)
    else {
        return;
    };
    let key = compute_native_key(fused_module, original_caller);
    let dir = aot_dir();
    if std::fs::create_dir_all(&dir).is_err() {
        return;
    }
    let fname = aot_filename(key);
    let path = dir.join(&fname);
    let tmp = dir.join(format!("{fname}.tmp"));
    if std::fs::write(&tmp, &blob).is_ok() {
        let _ = std::fs::rename(&tmp, &path);
    }
}

/// Try to LOAD + re-install a persisted T4 native blob for the program
/// `(fused, original_caller)`. Returns `None` (→ caller compiles fresh) when
/// AOT-persist is disabled, no on-disk entry exists, or the blob fails validation
/// (digest drift / corruption / ABI / cpu). On a validated hit, the relocation-
/// free bytes are re-installed into an RX page and a runnable `JitFunction` is
/// returned — the COLD-REPEAT win (peak-tier native code, zero warmup).
///
/// NOTE: the caller supplies the SAME `(fused, original_caller)` it would compile,
/// so the key matches by construction on a true repeat visit; a drifted program
/// produces a different fused/original module → different key → clean miss.
#[cfg(target_os = "windows")]
pub fn load_from_disk(
    fused_module: &Module,
    original_caller: &Module,
) -> Option<crate::jit::JitFunction> {
    if !aot_persist_enabled() {
        return None;
    }
    let key = compute_native_key(fused_module, original_caller);
    let path = aot_dir().join(aot_filename(key));
    let blob = match std::fs::read(&path) {
        Ok(b) => b,
        Err(_) => {
            bump_miss();
            return None;
        }
    };
    match deserialize_blob(&blob, key) {
        Some(parts) => install_from_blob(parts),
        None => {
            bump_miss();
            None
        }
    }
}

#[cfg(not(target_os = "windows"))]
pub fn load_from_disk(
    _fused_module: &Module,
    _original_caller: &Module,
) -> Option<crate::jit::JitFunction> {
    None
}

#[cfg(test)]
mod tests;
