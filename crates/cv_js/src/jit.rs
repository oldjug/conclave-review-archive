//! cv_js JIT — compile hot integer-arithmetic bytecode to native x86_64.
//!
//! V1 scope: a baseline trace-style JIT that compiles short, branch-
//! free integer-arithmetic sequences (the inner loops of Number-heavy
//! code like benchmarks and physics ticks) and falls back to the
//! interpreter on anything it doesn't recognise.
//!
//! Flow:
//!   1. `Profiler` counts execution per bytecode function.
//!   2. When count crosses a threshold, `JitCompiler::compile()` walks
//!      the function's ops, allocates registers via a tiny linear-scan
//!      allocator, and emits native code via `cv_asm::Emitter`.
//!   3. The emitted bytes are written into a page-aligned RWX buffer
//!      and exposed as a `JitFunction` whose `call(args)` dispatches
//!      to the native function pointer.
//!
//! The compiler bails out (returning None) on unsupported opcodes —
//! the host then runs the bytecode interpreter as usual. This is a
//! deliberately conservative design: bailouts are correctness-
//! preserving and let us ship a small JIT today and grow it organically.

use std::collections::HashMap;
use cv_asm::{Cc, Emitter, R64, Xmm};

/// A simple bytecode op the JIT understands. The real bytecode VM
/// (`bytecode::Op`) is larger; the JIT only digests a subset and
/// bails on the rest.
#[derive(Debug, Clone, Copy)]
pub enum JitOp {
    /// Load a constant integer into virtual register `dst`.
    ConstInt { dst: u16, value: i32 },
    /// Move `src` → `dst`.
    Mov { dst: u16, src: u16 },
    /// Integer add: `dst = a + b`.
    Add { dst: u16, a: u16, b: u16 },
    /// Integer subtract: `dst = a - b`.
    Sub { dst: u16, a: u16, b: u16 },
    /// Integer multiply: `dst = a * b`.
    Mul { dst: u16, a: u16, b: u16 },
    /// Return register `reg`.
    Return { reg: u16 },
}

/// f64 (double) JIT op — JS numbers are IEEE-754 doubles, so a JIT that
/// produces correct JS results must compute in `xmm` registers, not the i32
/// `JitOp` path. Pass 2 of the JIT build-out: straight-line double arithmetic.
/// `dst`/`a`/`b` are virtual registers mapped 1:1 to xmm0..xmm5 (bail if >5,
/// caller falls back to the interpreter).
#[derive(Debug, Clone, Copy)]
pub enum FJitOp {
    /// Load a double constant (its raw bits) into `dst`.
    FConst { dst: u16, bits: u64 },
    /// `dst = src`.
    FMove { dst: u16, src: u16 },
    /// `dst = a + b`.
    FAdd { dst: u16, a: u16, b: u16 },
    /// `dst = a - b`.
    FSub { dst: u16, a: u16, b: u16 },
    /// `dst = a * b`.
    FMul { dst: u16, a: u16, b: u16 },
    /// `dst = a / b`.
    FDiv { dst: u16, a: u16, b: u16 },
    /// Return `reg` (as the function's f64 result, in xmm0 per Win64 ABI).
    FRet { reg: u16 },
}

/// Map a virtual register to an xmm host register. xmm0..xmm5 are volatile
/// (no save needed); xmm6..xmm15 are callee-saved in the Win64 ABI and must be
/// preserved by the prolog/epilog when used.
fn vreg_xmm(v: u16) -> Result<Xmm, JitError> {
    Ok(match v {
        0 => Xmm::Xmm0,
        1 => Xmm::Xmm1,
        2 => Xmm::Xmm2,
        3 => Xmm::Xmm3,
        4 => Xmm::Xmm4,
        5 => Xmm::Xmm5,
        6 => Xmm::Xmm6,
        7 => Xmm::Xmm7,
        8 => Xmm::Xmm8,
        9 => Xmm::Xmm9,
        10 => Xmm::Xmm10,
        11 => Xmm::Xmm11,
        12 => Xmm::Xmm12,
        13 => Xmm::Xmm13,
        14 => Xmm::Xmm14,
        15 => Xmm::Xmm15,
        _ => return Err(JitError::OutOfRegisters),
    })
}

/// The Win64 ABI requires callee-saved xmm registers (xmm6..xmm15) to be
/// preserved across a call in their FULL 128 bits, not just the low 64 (a
/// caller may hold packed data or the upper lane of a vector in xmm6+). We save
/// and restore with 128-bit `movaps`, which requires a 16-byte-aligned memory
/// operand. So the xmm save area is laid out as one 16-byte slot per saved
/// register and the frame is sized + padded so the area base is 16-aligned.
///
/// On function entry (after the `call` pushed the return address) `RSP ≡ 8
/// (mod 16)`. We reserve `frame = saved*16 + 8`: the trailing 8 makes RSP
/// 16-aligned after `sub rsp, frame`, and the saved registers occupy
/// `[rsp + (r-6)*16]` (all 16-aligned), with the 8-byte pad at the top of the
/// frame. Returns the total frame size (0 when no callee-saved reg is used).
fn xmm_save_frame(max_reg: u16) -> i32 {
    if max_reg >= 6 {
        let saved = max_reg as i32 - 5;
        saved * 16 + 8
    } else {
        0
    }
}

/// Byte offset (from RSP) of the 16-aligned save slot for callee-saved `reg`.
fn xmm_save_slot(reg: u16) -> i32 {
    ((reg - 6) as i32) * 16
}

/// Save callee-saved xmm6..=max_reg with 128-bit `movaps` (full register, per
/// the Win64 ABI). The frame must already be reserved (RSP 16-aligned).
fn emit_xmm_save(em: &mut Emitter, max_reg: u16) {
    let mut r = 6u16;
    while r <= max_reg {
        em.movaps_mem_xmm(R64::Rsp, xmm_save_slot(r), vreg_xmm(r).unwrap());
        r += 1;
    }
}

/// Restore saved callee-saved xmm regs (xmm6..=max_reg) with 128-bit `movaps`
/// (FULL 128-bit register preserved — Win64 ABI, the M4.2b epilog bug fix), pop
/// the frame, and `ret`. Emitted at every return site.
fn emit_jit_epilog(em: &mut Emitter, max_reg: u16, frame: i32) {
    let mut r = 6u16;
    while r <= max_reg {
        em.movaps_xmm_mem(vreg_xmm(r).unwrap(), R64::Rsp, xmm_save_slot(r));
        r += 1;
    }
    if frame > 0 {
        em.add_r64_imm32(R64::Rsp, frame);
    }
    em.ret();
}

/// Compile a straight-line f64 op sequence to native x86_64. Doubles live in
/// xmm0..xmm5; the result is returned in xmm0 (Win64 f64 return register). No
/// callee-saved registers are touched, so the prolog/epilog is just `ret`.
pub fn compile_f64(ops: &[FJitOp]) -> Result<Vec<u8>, JitError> {
    if ops.is_empty() {
        return Err(JitError::EmptyFunction);
    }
    let mut em = Emitter::new();
    let mut returned = false;
    for op in ops {
        match *op {
            FJitOp::FConst { dst, bits } => {
                // Materialise the double's bits in a scratch GPR, then movq → xmm.
                em.mov_r64_imm64(R64::Rax, bits as i64);
                em.movq_xmm_r64(vreg_xmm(dst)?, R64::Rax);
            }
            FJitOp::FMove { dst, src } => {
                let (d, s) = (vreg_xmm(dst)?, vreg_xmm(src)?);
                if d != s {
                    em.movsd_xmm_xmm(d, s);
                }
            }
            FJitOp::FAdd { dst, a, b } => {
                let (d, a, b) = (vreg_xmm(dst)?, vreg_xmm(a)?, vreg_xmm(b)?);
                if d != a {
                    em.movsd_xmm_xmm(d, a);
                }
                em.addsd_xmm_xmm(d, b);
            }
            FJitOp::FSub { dst, a, b } => {
                let (d, a, b) = (vreg_xmm(dst)?, vreg_xmm(a)?, vreg_xmm(b)?);
                if d != a {
                    em.movsd_xmm_xmm(d, a);
                }
                em.subsd_xmm_xmm(d, b);
            }
            FJitOp::FMul { dst, a, b } => {
                let (d, a, b) = (vreg_xmm(dst)?, vreg_xmm(a)?, vreg_xmm(b)?);
                if d != a {
                    em.movsd_xmm_xmm(d, a);
                }
                em.mulsd_xmm_xmm(d, b);
            }
            FJitOp::FDiv { dst, a, b } => {
                let (d, a, b) = (vreg_xmm(dst)?, vreg_xmm(a)?, vreg_xmm(b)?);
                if d != a {
                    em.movsd_xmm_xmm(d, a);
                }
                em.divsd_xmm_xmm(d, b);
            }
            FJitOp::FRet { reg } => {
                let r = vreg_xmm(reg)?;
                if r != Xmm::Xmm0 {
                    em.movsd_xmm_xmm(Xmm::Xmm0, r);
                }
                em.ret();
                returned = true;
                break;
            }
        }
    }
    if !returned {
        // Implicit `return 0.0` (xmm0 = 0).
        em.xorpd_xmm_xmm(Xmm::Xmm0, Xmm::Xmm0);
        em.ret();
    }
    Ok(em.code)
}

/// Compile a whole bytecode function (Pass 4 — WITH control flow) to native
/// f64 code. Handles straight-line arithmetic plus for/while/if by FUSING each
/// comparison with the `JmpIfFalse` that consumes it, emitting a NaN-correct
/// conditional branch (JS `<`/`<=` are false for NaN, so the JmpIfFalse must
/// also jump on unordered). Loop back-edges (`Jmp` to an earlier op) are patched
/// from a recorded bytecode-index→offset table. Bails (None → interpreter) on
/// anything outside the supported subset, a register > 5, or a jump into a fused
/// pair. Params arrive in xmm0..xmm3 (= bytecode regs 0..n_params).
pub fn compile_bytecode_f64(
    code: &[crate::bytecode::Op],
    n_params: u8,
    const_f64: impl Fn(u16) -> Option<f64>,
) -> Option<Vec<u8>> {
    use crate::bytecode::Op;
    use cv_asm::{Cc, Xmm};
    if n_params > 4 || code.is_empty() {
        return None;
    }
    // Pre-scan: bail on any unsupported op, and find the highest register so we
    // know which callee-saved xmm regs (6..=max) to preserve.
    let mut max_reg = 0u16;
    for op in code {
        match *op {
            Op::LoadConst { dst, .. } | Op::LoadUndef { dst } => max_reg = max_reg.max(dst),
            Op::Move { dst, src } => max_reg = max_reg.max(dst).max(src),
            Op::Add { dst, lhs, rhs }
            | Op::Sub { dst, lhs, rhs }
            | Op::Mul { dst, lhs, rhs }
            | Op::Div { dst, lhs, rhs }
            | Op::Lt { dst, lhs, rhs }
            | Op::Le { dst, lhs, rhs }
            | Op::Gt { dst, lhs, rhs }
            | Op::Ge { dst, lhs, rhs } => max_reg = max_reg.max(dst).max(lhs).max(rhs),
            Op::JmpIfFalse { cond, .. } => max_reg = max_reg.max(cond),
            Op::Ret { src } => max_reg = max_reg.max(src),
            Op::Jmp { .. } => {}
            _ => return None,
        }
    }
    if max_reg > 15 {
        return None;
    }
    // Stack frame to preserve the callee-saved xmm6..=max_reg in FULL 128 bits
    // (16-byte slots, 16-aligned — see `xmm_save_frame`).
    let frame: i32 = xmm_save_frame(max_reg);

    let n = code.len();
    let mut em = Emitter::new();
    let mut offsets = vec![0usize; n]; // bytecode index → machine-code offset
    let mut consumed = vec![false; n]; // the JmpIfFalse fused into a preceding cmp
    let mut patches: Vec<(usize, usize)> = Vec::new(); // (disp byte offset, target bc index)

    // Prolog: reserve the frame and save the callee-saved xmm regs we'll use
    // (128-bit movaps, Win64 ABI).
    if frame > 0 {
        em.sub_r64_imm32(R64::Rsp, frame);
        emit_xmm_save(&mut em, max_reg);
    }

    let mut i = 0usize;
    while i < n {
        offsets[i] = em.code.len();
        match code[i] {
            Op::LoadUndef { dst } => {
                // ECMA-262: undefined coerces to NaN in numeric context
                // (e.g. missing function parameters). Emit the canonical
                // quiet-NaN bit pattern (0x7FF8000000000000) so that
                // arithmetic on an unset register produces NaN, not 0.0.
                let d = vreg_xmm(dst).ok()?;
                const NAN_BITS: u64 = 0x7FF8_0000_0000_0000;
                em.mov_r64_imm64(R64::Rax, NAN_BITS as i64);
                em.movq_xmm_r64(d, R64::Rax);
            }
            Op::LoadConst { dst, k } => {
                let d = vreg_xmm(dst).ok()?;
                let f = const_f64(k)?;
                em.mov_r64_imm64(R64::Rax, f.to_bits() as i64);
                em.movq_xmm_r64(d, R64::Rax);
            }
            Op::Move { dst, src } => {
                let (d, s) = (vreg_xmm(dst).ok()?, vreg_xmm(src).ok()?);
                if d != s {
                    em.movsd_xmm_xmm(d, s);
                }
            }
            Op::Add { dst, lhs, rhs } => {
                let (d, a, b) = (
                    vreg_xmm(dst).ok()?,
                    vreg_xmm(lhs).ok()?,
                    vreg_xmm(rhs).ok()?,
                );
                if d != a {
                    em.movsd_xmm_xmm(d, a);
                }
                em.addsd_xmm_xmm(d, b);
            }
            Op::Sub { dst, lhs, rhs } => {
                let (d, a, b) = (
                    vreg_xmm(dst).ok()?,
                    vreg_xmm(lhs).ok()?,
                    vreg_xmm(rhs).ok()?,
                );
                if d != a {
                    em.movsd_xmm_xmm(d, a);
                }
                em.subsd_xmm_xmm(d, b);
            }
            Op::Mul { dst, lhs, rhs } => {
                let (d, a, b) = (
                    vreg_xmm(dst).ok()?,
                    vreg_xmm(lhs).ok()?,
                    vreg_xmm(rhs).ok()?,
                );
                if d != a {
                    em.movsd_xmm_xmm(d, a);
                }
                em.mulsd_xmm_xmm(d, b);
            }
            Op::Div { dst, lhs, rhs } => {
                let (d, a, b) = (
                    vreg_xmm(dst).ok()?,
                    vreg_xmm(lhs).ok()?,
                    vreg_xmm(rhs).ok()?,
                );
                if d != a {
                    em.movsd_xmm_xmm(d, a);
                }
                em.divsd_xmm_xmm(d, b);
            }
            // A comparison MUST be immediately consumed by a `JmpIfFalse` on its
            // result — that's the for/while/if shape. Fuse them into a branch.
            Op::Lt { dst, lhs, rhs }
            | Op::Le { dst, lhs, rhs }
            | Op::Gt { dst, lhs, rhs }
            | Op::Ge { dst, lhs, rhs } => {
                if i + 1 >= n {
                    return None;
                }
                let target = match code[i + 1] {
                    Op::JmpIfFalse { cond, target } if cond == dst => target as usize,
                    _ => return None, // bool not consumed by an immediate JmpIfFalse → bail
                };
                let (a, b) = (vreg_xmm(lhs).ok()?, vreg_xmm(rhs).ok()?);
                em.ucomisd_xmm_xmm(a, b);
                // JmpIfFalse jumps when the comparison is FALSE — including the
                // NaN/unordered case (PF=1), since `<`,`<=` are false for NaN.
                match code[i] {
                    Op::Lt { .. } => {
                        let o = em.jcc_rel32_placeholder(Cc::Parity);
                        patches.push((o, target));
                        let o2 = em.jcc_rel32_placeholder(Cc::AboveEq); // !(a<b): a>=b
                        patches.push((o2, target));
                    }
                    Op::Le { .. } => {
                        let o = em.jcc_rel32_placeholder(Cc::Parity);
                        patches.push((o, target));
                        let o2 = em.jcc_rel32_placeholder(Cc::Above); // !(a<=b): a>b
                        patches.push((o2, target));
                    }
                    // a>b false = a<=b OR NaN; both set CF=1 or ZF=1 → JBE catches all.
                    Op::Gt { .. } => {
                        let o = em.jcc_rel32_placeholder(Cc::BelowEq);
                        patches.push((o, target));
                    }
                    // a>=b false = a<b OR NaN; both set CF=1 → JB catches all.
                    Op::Ge { .. } => {
                        let o = em.jcc_rel32_placeholder(Cc::Below);
                        patches.push((o, target));
                    }
                    _ => unreachable!(),
                }
                consumed[i + 1] = true;
                offsets[i + 1] = em.code.len();
                i += 2;
                continue;
            }
            Op::Jmp { target } => {
                let o = em.jmp_rel32_placeholder();
                patches.push((o, target as usize));
            }
            Op::Ret { src } => {
                // Capture the result into xmm0 BEFORE restoring callee-saved regs.
                let r = vreg_xmm(src).ok()?;
                if r != Xmm::Xmm0 {
                    em.movsd_xmm_xmm(Xmm::Xmm0, r);
                }
                emit_jit_epilog(&mut em, max_reg, frame);
            }
            // Branches not fused with a comparison, calls, property access, etc.
            _ => return None,
        }
        i += 1;
    }
    // Defensive fall-through guard (bytecode always ends in Ret, but never run
    // off the end of the page if it somehow doesn't).
    em.xorpd_xmm_xmm(Xmm::Xmm0, Xmm::Xmm0);
    emit_jit_epilog(&mut em, max_reg, frame);
    // A jump may not land inside a fused pair (would skip the ucomisd).
    for (_, t) in &patches {
        if *t >= n || consumed[*t] {
            return None;
        }
    }
    for (disp_off, t) in patches {
        em.patch_rel32_to(disp_off, offsets[t]);
    }
    Some(em.code)
}

// ======================================================================
// M4.3 — T2-LITE: INLINED `JsVal` JIT (the NaN-box validation gate).
//
// THE question this answers: does an inlined JIT that keeps values as `JsVal`
// (one u64) and does an INLINE tag-check + UNBOXED arithmetic actually BEAT the
// VM — unlike the helper-based T1 (call-per-op), which M4.2b measured at
// 0.88–0.99x the VM (never faster)? If yes, the whole NaN-box bet pays off and
// the full T2 (owning handle + shape-guarded property loads + calls + deopt +
// GC) is justified. If no, that is itself the critical finding.
//
// SCOPE (numeric-only, so the bank holds only numbers during a JIT run ⇒ no Rc
// to manage ⇒ borrowed-safe): a hot function operating on a `JsVal` REGISTER
// BANK (`*mut JsVal`, 8 bytes/slot, base in callee-saved RBX). Every arithmetic
// op INLINE-tag-checks both operands are numbers; if so it unboxes to xmm, does
// the f64 op, canonicalizes a NaN result, re-boxes to a `JsVal`, and stores it.
// If EITHER operand is not a number (the double lane OR the int32 lane), the
// emitted code DEOPTs: it returns the `T2_DEOPT` status and the caller re-runs
// the function on the VM (correct, just slower). Any op outside the subset is
// DECLINED at compile time (returns None → VM), exactly like the f64 JIT.
//
// ABI: `extern "system" fn(bank: *mut JsVal, out: *mut u64) -> u64`.
//   * RCX = bank base, RDX = out-slot pointer (kept in callee-saved RDI).
//   * Returns `T2_RETURNED(0)` (and stores the result `JsVal` bits to `*out`)
//     or `T2_DEOPT(1)` (out untouched — caller falls back to the VM).
// Only volatile xmm0/xmm1 are used, so NO callee-saved xmm save area is needed;
// the prolog just preserves RBX/RDI.
// ======================================================================

/// T2-lite native status tags (returned in RAX).
pub const T2_RETURNED: u64 = 0;
pub const T2_DEOPT: u64 = 1;
/// T2 Phase 4 (CALL inlining): a re-entrant CALL op's callee THREW a catchable
/// error. The error payload is stashed in the `T2CallCtx` out-slot; the runner
/// maps it to the outer machinery's `Err`. DISTINCT from `T2_DEOPT`: a THREW
/// happens AFTER the call has run (a committed side effect), so the runner must
/// NOT re-run the function on the VM (that would re-do the call) — it propagates
/// the error directly. (Functions that could deopt AFTER a call are declined at
/// compile time, so a THREW is the only post-call non-return exit.)
pub const T2_THREW: u64 = 2;
/// T2 Phase 4: a re-entrant CALL op hit the wall-clock watchdog (`Deadline`).
/// UNCATCHABLE — the runner returns `RuntimeError::Deadline` and never re-runs.
pub const T2_DEADLINE: u64 = 3;
/// T2 Phase 5: a per-guard REAL deopt. The guard's stub wrote its `DeoptSite`
/// index (the `deopt_id`) to `*out` and returned this status; the runner decodes
/// the JIT bank into VM `Value` registers and RESUMES the bytecode VM at the
/// site's `bc_pc` (mid-function), producing bit-identical results. DISTINCT from
/// `T2_DEOPT` (the Tier-A whole-function re-run, used only where no side effect
/// precedes the guard): a RESUME continues AFTER any committed effect, so it must
/// never re-run from ip=0.
pub const T2_DEOPT_RESUME: u64 = 4;

// `JsVal` bit constants, replicated here for the emitted asm. These MUST stay in
// lock-step with `jsval.rs`; the unit tests below assert the exact values so a
// drift in either place is caught immediately (a wrong tag-check is silent
// corruption — the #1 hazard).
const JV_QNAN_MASK: u64 = 0xFFF8_0000_0000_0000;
const JV_QNAN_BITS: u64 = 0xFFF8_0000_0000_0000;
const JV_CANONICAL_NAN: u64 = 0x7FF8_0000_0000_0000;
/// Top-16-bit signature of the int32 lane: `QNAN_BITS | (TAG_INT32(5) << 48)`.
const JV_INT32_TOP16: u64 = 0xFFFD_0000_0000_0000;
/// Mask isolating the top 16 bits (used to test the int32-lane signature).
const JV_TOP16_MASK: u64 = 0xFFFF_0000_0000_0000;
/// Boolean singleton bits: `false = QNAN_BITS | (TAG_SINGLETON(6)<<48) | 2`,
/// `true = …| 3`. So `true == false + 1`, letting us add a 0/1 flag to the
/// false base to box a comparison result.
const JV_FALSE: u64 = 0xFFFE_0000_0000_0002;
const JV_TRUE: u64 = 0xFFFE_0000_0000_0003;
/// Top-16-bit signature of the OBJECT lane: `QNAN_BITS | (TAG_OBJECT(0) << 48)`
/// = `QNAN_BITS`. A `JsVal` is an `Object` iff `bits & JV_TOP16_MASK ==
/// JV_OBJECT_TOP16`. (Array/Function/Native/BcClosure/Int32/Singleton/StrBig all
/// have a non-zero tag in bits 48..=50, so they DON'T match — only `Object`.)
const JV_OBJECT_TOP16: u64 = 0xFFF8_0000_0000_0000;
/// Top-16-bit signature of the ARRAY lane: `QNAN_BITS | (TAG_ARRAY(1) << 48)` =
/// `0xFFF9_0000_0000_0000`. A `JsVal` is an `Array` iff `bits & JV_TOP16_MASK ==
/// JV_ARRAY_TOP16`. (TAG_ARRAY = 1, so bit 48 is set — distinct from Object's
/// all-zero tag and every other lane.) Asserted against `jsval.rs` in the tests.
const JV_ARRAY_TOP16: u64 = 0xFFF9_0000_0000_0000;
/// Mask isolating the 48-bit pointer payload (the `Rc::as_ptr` of the object).
const JV_PAYLOAD_MASK: u64 = 0x0000_FFFF_FFFF_FFFF;
/// DEOPT sentinel returned by `rt_getprop_slot_immediate`: the `hole` `JsVal`
/// bits. The helper NEVER returns a hole as a SUCCESS value (a hole slot value
/// deopts), so this is an unambiguous "deopt" signal in the single u64 return.
const JV_HOLE: u64 = 0xFFFE_0000_0000_0004;

/// AUDITED extern-system helper for the T2 inline GetProp fast path. Given a
/// boxed-OBJECT pointer payload (`obj_ptr` = `Rc::as_ptr(&Rc<RefCell<OrderedMap>>)
/// as u64`, the low-48 bits already extracted by the emitted code) and a
/// pre-resolved `slot`, read `slots[slot]` (a 32-byte `Value`) and:
///   * if it is a `Number` or `Bool` IMMEDIATE → return its boxed `JsVal` bits
///     (the bank may then hold it — it is an immediate, never a heap pointer);
///   * otherwise (object/array/string/function/undefined/null/hole/bigint, an
///     out-of-range slot, …) → return `JV_HOLE` (DEOPT).
///
/// BORROWED-SAFE: this reads `&Value` and EXTRACTS only the immediate; it does
/// NOT clone the `Value` and NEVER hands a heap `JsVal` back. No refcount touched.
///
/// REFCELL ALIASING (documented, asserted by the leaf-no-reentry contract): the
/// read goes through `RefCell::as_ptr()` → `&*` , bypassing the borrow flag. This
/// is sound because (a) the receiver is a borrowed function ARG whose `Rc` is
/// kept alive by the caller's `args: &[Value]` for the whole call, and (b) the
/// emitted code calls this helper as a LEAF between the shape guard and the bank
/// store with NO intervening op that could re-enter the VM and take a conflicting
/// `&mut` borrow of the SAME object — so no live `Ref`/`RefMut` overlaps this
/// read window. (The T2 body makes no call op; the only call it emits IS this
/// helper, which neither recurses nor mutates.)
///
/// # Safety
/// `obj_ptr` must be the live `Rc::as_ptr` of a `Value::Object`'s
/// `Rc<RefCell<OrderedMap<String, Value>>>` (guaranteed by the emitted is-object
/// tag check + the bank slot holding an unmodified arg whose `Rc` the caller
/// keeps alive). `slot` is the per-shape pre-resolved index.
pub extern "system" fn rt_getprop_slot_immediate(obj_ptr: u64, slot: u64) -> u64 {
    use crate::interp::Value;
    use crate::jsval::JsVal;
    if obj_ptr == 0 {
        return JV_HOLE;
    }
    // SAFETY: see the fn-level contract. The pointer is a live RefCell<OrderedMap>.
    let rc_ptr =
        obj_ptr as usize as *const std::cell::RefCell<crate::ordered::OrderedMap<String, Value>>;
    let cell = unsafe { &*rc_ptr };
    // Bypass the borrow flag (leaf, no reentry, no overlapping &mut — see docs).
    let map: &crate::ordered::OrderedMap<String, Value> = unsafe { &*cell.as_ptr() };
    match map.value_at_slot(slot as usize) {
        Some(Value::Number(n)) => JsVal::number(*n).bits(),
        Some(Value::Bool(b)) => JsVal::boolean(*b).bits(),
        // Any non-immediate slot value (object/array/string/fn/undef/null/hole/
        // bigint) or an out-of-range slot → DEOPT (the bank must only ever
        // receive an immediate in this phase).
        _ => JV_HOLE,
    }
}

/// T2 Phase 3 — the FIRST helper that lets a GetProp result be a HEAP value held
/// in an OWNING bank slot. Given the receiver object pointer (`obj_ptr`,
/// low-48 already extracted), a pre-resolved `slot`, the bank base (`bank`,
/// `*mut JsVal`) and the destination slot index `dst`, read `slots[slot]` and:
///   * Object / Array / String — store the result into `bank[dst]` with OWNING
///     semantics (inc the new heap value's `Rc`, then dec the old slot's `Rc` if
///     it was a pointer — INC-NEW-BEFORE-DEC-OLD, so a self-store is safe and the
///     last ref is never transiently dropped), and return `T2_RETURNED`. The
///     bank is now the owner of one strong ref of the heap value; `RegBank::Drop`
///     (or a later overwrite) releases it.
///   * Number / Bool IMMEDIATE — store the immediate (owning store dec's any old
///     pointer), return `T2_RETURNED`.
///   * anything else (function/native/bcclosure/bigint/undefined/null/hole/oob)
///     — DEOPT (return `T2_DEOPT`); the bank is left UNTOUCHED so the VM re-run
///     produces the identical result (region stays re-run-identical).
///
/// SCOPE: this phase admits Object/Array/String heap results (the task's first
/// heap-resident use) — Function/Native/BcClosure/BigInt deopt for now (no use
/// site needs them yet; keeping the admit-set minimal keeps the proof tight).
///
/// # Safety
/// * `obj_ptr` is a live `Rc::as_ptr` of the receiver's `RefCell<OrderedMap>`
///   (guaranteed by the emitted is-object + shape guards + a pure-arg receiver
///   the caller's `args` keeps alive) — same contract as
///   [`rt_getprop_slot_immediate`].
/// * `bank` points at the live, non-reallocated, GC-registered `JsVal` bank and
///   `dst` is `< bank_len` (the emitted code bakes `dst` from the validated
///   bytecode register, always in range). The store reads + overwrites
///   `bank[dst]` only.
/// * The new heap value's `Rc` is alive (it lives inside the receiver object the
///   caller keeps alive) at the moment we `rc_inc` it, so the +1 is valid; the
///   bank then owns it independently of the receiver.
pub extern "system" fn rt_getprop_slot_owning_store(
    obj_ptr: u64,
    slot: u64,
    bank: *mut u64,
    dst: u64,
) -> u64 {
    use crate::interp::Value;
    use crate::jsval::JsVal;
    if obj_ptr == 0 || bank.is_null() {
        return T2_DEOPT;
    }
    let rc_ptr =
        obj_ptr as usize as *const std::cell::RefCell<crate::ordered::OrderedMap<String, Value>>;
    // SAFETY: live RefCell<OrderedMap> (fn contract). Leaf read, no reentry.
    let cell = unsafe { &*rc_ptr };
    let map: &crate::ordered::OrderedMap<String, Value> = unsafe { &*cell.as_ptr() };
    // Compute the NEW `JsVal` to store (admit set: immediates + plain Object /
    // Array / String heap lanes). Anything else → DEOPT, bank untouched.
    //
    // ACCESSOR GUARD (correctness, matches the VM): an own slot can hold an
    // ACCESSOR-WRAPPER Object (carrying `__get__`/`__set__`) — reading `o.k` must
    // INVOKE the getter, not return the wrapper. The immediate helper deopts on
    // all objects so it never saw this; the heap helper admits Objects, so it MUST
    // explicitly DEOPT on an accessor wrapper and let the VM invoke the getter.
    let slot_val = match map.value_at_slot(slot as usize) {
        Some(v) => v,
        None => return T2_DEOPT,
    };
    let new_val: JsVal = match slot_val {
        Value::Number(n) => JsVal::number(*n),
        Value::Bool(b) => JsVal::boolean(*b),
        Value::Object(rc) => {
            // Deopt on an accessor wrapper (getter/setter) — the VM must run it.
            if crate::interp::accessor_parts(slot_val).is_some() {
                return T2_DEOPT;
            }
            JsVal::object(rc)
        }
        Value::Array(rc) => JsVal::array(rc),
        Value::String(s) => JsVal::string(s.as_rc()),
        _ => return T2_DEOPT,
    };
    // OWNING STORE into bank[dst] (the RegBank::store contract, performed in place
    // on the raw bank the emitted code passed). INC-NEW-BEFORE-DEC-OLD.
    // SAFETY: bank[dst] is in range (caller contract); `new_val`'s pointee Rc is
    // alive (it lives in the receiver object).
    let dst_ptr = unsafe { bank.add(dst as usize) } as *mut JsVal;
    unsafe {
        new_val.rc_inc(); // +1: the bank takes ownership of the heap value (no-op for immediate)
        let old = std::ptr::read(dst_ptr); // current slot value (Copy JsVal)
        std::ptr::write(dst_ptr, new_val); // overwrite
        old.rc_dec(); // -1: release the previous owner (no-op for immediate)
    }
    T2_RETURNED
}

/// T2 GetIdx — the audited owning-store helper for a COMPUTED ARRAY READ
/// `bank[dst] = arr[idx]`. The emitted code has already proven (with inline
/// guards that DEOPT on a miss) that:
///   * the receiver is a `Value::Array` (so `arr_ptr` is a live `Rc::as_ptr` of
///     its `RefCell<Vec<Value>>`), and
///   * `idx` is a NON-NEGATIVE integer (an int32 lane, or an integer-valued
///     double in `[0, i32::MAX]`) — negative / fractional / NaN indices DEOPT to
///     the VM (they become a NAMED-property lookup there, which yields undefined
///     or a method binding — outside this fast path).
/// Given those, this LEAF helper does the bounds-checked element read + owning
/// bank store, MIRRORING `rt_getprop_slot_owning_store` exactly:
///   * IN-BOUNDS (`idx < len`):
///       - Number / Bool IMMEDIATE  → owning-store the immediate, RETURNED;
///       - plain Object / Array / String HEAP value → owning-store (inc-new-
///         before-dec-old), RETURNED (the bank now owns +1 of the element);
///       - a HOLE (`delete arr[i]` sparse slot) → DEOPT. The VM's `GetIdx` stores
///         the internal `Value::Hole` sentinel into the register (NOT `undefined`
///         — verified against the VM), and the oracle DISTINGUISHES hole from
///         undefined, so we must let the VM produce the exact `Hole` image rather
///         than synthesise `undefined`. (Holes are rare in dense hot loops.)
///       - an accessor-wrapper Object (`get`/`set`), Function / Native /
///         BcClosure / BigInt / Undefined / Null → DEOPT (the VM may need to run
///         a getter or apply named-property/host resolution; admitting these
///         would diverge — decline rather than risk).
///   * OUT-OF-BOUNDS (`idx >= len`): JS `arr[past-end] === undefined`. Store the
///     `undefined` IMMEDIATE (owning store dec's any old heap value in `dst`),
///     RETURNED. This is NOT a deopt — it's the correct, total JS result, and the
///     VM agrees (verified: `oob -> undefined`).
/// On any DEOPT the bank is left UNTOUCHED so the VM resume produces the identical
/// register image.
///
/// REFCELL ALIASING / leaf-no-reentry: identical contract to
/// `rt_getprop_slot_owning_store` — the read goes through `RefCell::as_ptr()`
/// (bypassing the borrow flag), sound because the array `Rc` is kept alive by the
/// owning bank slot (or the caller's args) for the whole call and the emitted
/// code calls this as a LEAF with no intervening op that could take a conflicting
/// `&mut` borrow of the SAME array.
///
/// # Safety
/// * `arr_ptr` is a live `Rc::as_ptr` of a `Value::Array`'s `RefCell<Vec<Value>>`
///   (guaranteed by the emitted is-array guard + the owning bank keeping the
///   receiver alive).
/// * `idx` is a non-negative integer index the emitted code validated.
/// * `bank` is the live, non-reallocated, GC-registered owning bank and `dst <
///   bank_len` (the emitted code bakes `dst` from a validated bytecode register).
///   The store reads + overwrites `bank[dst]` only.
/// * The new heap value's `Rc` (if any) is alive (it lives inside the array the
///   caller keeps alive) at the moment we `rc_inc` it.
pub extern "system" fn rt_getidx_owning_store(
    arr_ptr: u64,
    idx: u64,
    bank: *mut u64,
    dst: u64,
) -> u64 {
    use crate::interp::Value;
    use crate::jsval::JsVal;
    if arr_ptr == 0 || bank.is_null() {
        return T2_DEOPT;
    }
    let rc_ptr = arr_ptr as usize as *const std::cell::RefCell<Vec<Value>>;
    // SAFETY: live RefCell<Vec<Value>> (fn contract). Leaf read, no reentry, no
    // overlapping &mut of the SAME array (the T2 body makes no array mutation op
    // between the guard and this call).
    let cell = unsafe { &*rc_ptr };
    let vec: &Vec<Value> = unsafe { &*cell.as_ptr() };
    let i = idx as usize;
    // Compute the NEW `JsVal` to store. OOB → undefined; in-bounds → the element
    // (immediate or admitted heap lane); hole / non-admitted → DEOPT (bank
    // untouched, VM resumes identically).
    let new_val: JsVal = match vec.get(i) {
        // OUT-OF-BOUNDS → undefined (total JS result, == VM). NOT a deopt.
        None => JsVal::undefined(),
        Some(elem) => match elem {
            Value::Number(n) => JsVal::number(*n),
            Value::Bool(b) => JsVal::boolean(*b),
            Value::Object(rc) => {
                // Accessor-wrapper element → the VM may invoke a getter. DEOPT.
                if crate::interp::accessor_parts(elem).is_some() {
                    return T2_DEOPT;
                }
                JsVal::object(rc)
            }
            Value::Array(rc) => JsVal::array(rc),
            Value::String(s) => JsVal::string(s.as_rc()),
            // A HOLE reads as the internal `Value::Hole` on the VM (distinct from
            // undefined per the oracle) — let the VM produce that image. Everything
            // else (Function/Native/BcClosure/BigInt/Undefined/Null) also DEOPTs.
            _ => return T2_DEOPT,
        },
    };
    // OWNING STORE into bank[dst] — the RegBank::store contract, INC-NEW-BEFORE-
    // DEC-OLD. SAFETY: bank[dst] is in range (caller contract); `new_val`'s pointee
    // Rc (if any) is alive (it lives in the array).
    let dst_ptr = unsafe { bank.add(dst as usize) } as *mut JsVal;
    unsafe {
        new_val.rc_inc(); // +1: the bank takes ownership (no-op for an immediate)
        let old = std::ptr::read(dst_ptr);
        std::ptr::write(dst_ptr, new_val);
        old.rc_dec(); // -1: release the previous owner (no-op for an immediate)
    }
    T2_RETURNED
}

/// T2 SetIdx — the audited owning element-replace helper for a COMPUTED ARRAY
/// WRITE `arr[idx] = val`. The emitted code has already proven the receiver is a
/// `Value::Array` and `idx` is a NON-NEGATIVE integer.
///   * IN-BOUNDS (`idx < len`): replace `arr[idx]` with the new value. The element
///     replace is REFCOUNT-CORRECT for free: we decode `val_bits` into an OWNED
///     `Value` via `JsVal::to_value()` (which takes its OWN +1 — a borrowed-handle
///     clone independent of the bank's ref), then `arr[idx] = value` MOVES it in
///     and DROPS the OLD element (releasing its `Rc`, possibly freeing). This is
///     the same inc-new-before-dec-old discipline the owning bank uses, expressed
///     through Rust's move + `Drop`, so a self-store (`arr[j] = arr[j]`) and a
///     last-ref overwrite are both correct (the new value's +1 is taken before the
///     old is dropped). Returns RETURNED. The bank slot `src` is UNCHANGED (the
///     bank still owns its own ref of `val`).
///   * OUT-OF-BOUNDS (`idx >= len`): JS extends the array (`arr[len]=x` grows it;
///     `arr[len+k]=x` creates holes) — a STRUCTURAL change. DEOPT so the VM
///     performs the resize exactly (rare in a hot fixed-size loop). Returns
///     T2_DEOPT, the array UNTOUCHED.
///
/// SIDE-EFFECT / P5: the in-bounds write COMMITS here, before any later op's
/// guard. A later guard that deopts resumes the VM AFTER this op's bc_pc (P5 real
/// per-guard resume), so the committed write is NEVER re-run (no duplicate effect).
///
/// REFCELL ALIASING / leaf-no-reentry: `to_value()` reconstructs the value's `Rc`
/// from the bank-slot pointer WITHOUT borrowing the array's `RefCell` (it only
/// touches the value's own allocation refcount), so the subsequent `borrow_mut()`
/// of the array cannot conflict. The helper is a LEAF (no VM reentry) so no other
/// borrow of the SAME array overlaps this write window.
///
/// # Safety
/// * `arr_ptr` is a live `Rc::as_ptr` of a `Value::Array`'s `RefCell<Vec<Value>>`
///   (emitted is-array guard + owning bank keeps the receiver alive).
/// * `idx` is a non-negative integer (emitted validation).
/// * `val_bits` is a `JsVal` whose pointee `Rc` (if any) is alive (it is the
///   owning bank's slot value, alive for the call).
pub extern "system" fn rt_setidx_owning_store(arr_ptr: u64, idx: u64, val_bits: u64) -> u64 {
    use crate::interp::Value;
    use crate::jsval::JsVal;
    if arr_ptr == 0 {
        return T2_DEOPT;
    }
    let rc_ptr = arr_ptr as usize as *const std::cell::RefCell<Vec<Value>>;
    // SAFETY: live RefCell<Vec<Value>> (fn contract). Leaf, no reentry.
    let cell = unsafe { &*rc_ptr };
    let i = idx as usize;
    // Bounds-check against the CURRENT length (a borrow scope kept minimal).
    {
        // SAFETY: leaf read of the length; no overlapping borrow.
        let len = unsafe { (*cell.as_ptr()).len() };
        if i >= len {
            // OOB write extends/holes the array — structural change → VM.
            return T2_DEOPT;
        }
    }
    // Decode the new value, taking its OWN +1 (independent of the bank's ref).
    // SAFETY: `val_bits` is a live bank-slot JsVal (its `Rc`, if any, is alive).
    let value = unsafe { JsVal(val_bits).to_value() };
    // In-bounds replace: MOVE the new value in, DROP the old (Rc released). Takes a
    // fresh &mut borrow of the array (no overlapping borrow — see the aliasing note).
    {
        let mut arr = cell.borrow_mut();
        arr[i] = value;
    }
    T2_RETURNED
}

/// T2 Phase 3 — the GENERIC owning bank store the heap-mode codegen routes EVERY
/// slot-write through, so the bank maintains the UNIFORM PER-SLOT invariant: every
/// slot owns exactly one strong ref of its (pointer) value. `val` is the `JsVal`
/// bits to store into `bank[dst]`. INC-NEW-BEFORE-DEC-OLD:
///   1. `rc_inc(val)` — the bank takes one ref of the new value (no-op immediate);
///   2. read the old slot, overwrite with `val`;
///   3. `rc_dec(old)` — release the previous owner (no-op immediate; may free).
/// This makes a Move (which would otherwise alias a heap value without an inc),
/// an overwrite of a heap slot by an immediate (which would otherwise leak the old
/// ref), and a self-store ALL correct, and makes `OwningRegBank::Drop`'s blanket
/// dec-all exactly balanced (one dec per slot's one owned ref).
///
/// This is the SAME contract as [`OwningRegBank::store`]; both exist because the
/// store happens from native code (this helper) AND from Rust (the method, for the
/// arg-seed/teardown accounting + the leak-oracle/mutation-arm tests).
///
/// # Safety
/// `bank` is the live, non-reallocated, GC-registered bank; `dst < bank_len` (the
/// emitted code bakes `dst` from a validated bytecode register). `val`'s pointee
/// `Rc` (if a pointer) is alive at call time (it is a value the native code just
/// produced from a live source — an arg slot, a constant, or a GetProp result
/// whose object the caller keeps alive).
pub extern "system" fn rt_bank_store(bank: *mut u64, dst: u64, val: u64) {
    use crate::jsval::JsVal;
    if bank.is_null() {
        return;
    }
    let new_val = JsVal(val);
    let dst_ptr = unsafe { bank.add(dst as usize) } as *mut JsVal;
    unsafe {
        new_val.rc_inc();
        let old = std::ptr::read(dst_ptr);
        std::ptr::write(dst_ptr, new_val);
        old.rc_dec();
    }
}

/// How `compile_t2lite` writes a slot. NUMERIC mode emits a raw `mov [bank+dst],
/// rax` (byte-identical to the P1/P2-dormant codegen). HEAP mode routes EVERY
/// store through [`rt_bank_store`] (whose address is carried here) so the bank
/// keeps the uniform per-slot ownership invariant — the prerequisite for holding a
/// heap value in a slot.
#[derive(Clone, Copy)]
pub enum T2StoreMode {
    /// Raw `mov [bank+dst*8], rax`. The bank holds only borrowed immediates.
    Numeric,
    /// Owning store via `rt_bank_store` at this address (inc-new/dec-old).
    Heap { store_helper: usize },
}

/// Emit the store of the `JsVal` currently in RAX into `bank[dst]`, per `mode`.
/// MUST be the last action of an op (the heap path's helper call clobbers volatile
/// registers). At a T2 op boundary RSP ≡ 8 (mod 16); the heap path's `sub 40`
/// realigns to 0 for the `call` (32 shadow + 8 pad), then restores.
fn t2_emit_bank_store(em: &mut Emitter, dst: u16, mode: T2StoreMode) {
    match mode {
        T2StoreMode::Numeric => {
            em.mov_mem_r64(T2_BANK, (dst as i32) * 8, R64::Rax);
        }
        T2StoreMode::Heap { store_helper } => {
            // rt_bank_store(bank=RCX, dst=RDX, val=R8). Value is in RAX → R8 first
            // (so a later RCX/RDX set can't clobber it), then RDX = dst, RCX = bank.
            em.mov_r64_r64(R64::R8, R64::Rax); // arg3 = val
            em.mov_r64_imm32(R64::Rdx, dst as i32); // arg2 = dst slot index
            em.mov_r64_r64(R64::Rcx, T2_BANK); // arg1 = bank base
            em.sub_r64_imm32(R64::Rsp, 40);
            em.mov_r64_imm64(R64::R11, store_helper as i64);
            em.call_r64(R64::R11);
            em.add_r64_imm32(R64::Rsp, 40);
        }
    }
}

/// Byte offset from a `Value::Object`'s `Rc::as_ptr` (= the `RefCell` pointer the
/// T2 emitted code holds in the bank's low-48 payload) to the `u32` shape-id
/// HEADER word inside the `OrderedMap`. Computed at RUNTIME from a real instance
/// (NOT assumed) so a layout change in `RefCell`/`OrderedMap`/`ShapeCache` is
/// absorbed automatically — and pinned by `t2_shape_off_is_stable` (a drift in
/// the computed value vs a baked one would be silent corruption). The header is a
/// `Cell<u32>` (transparent over `u32`), so a 4-byte `mov_r32_mem` at this offset
/// reads exactly the header value.
pub fn t2_shape_header_offset() -> i32 {
    use crate::interp::Value;
    use std::cell::RefCell;
    use std::rc::Rc;
    let rc: Rc<RefCell<crate::ordered::OrderedMap<String, Value>>> =
        Rc::new(RefCell::new(crate::ordered::OrderedMap::new()));
    let base = Rc::as_ptr(&rc) as usize;
    // The header address: borrow the map and take the address the public
    // accessor reads from. `shape_header_ptr` returns a stable `*const u32` to the
    // Cell's storage (Cell<u32> is repr(transparent) over u32).
    let header_addr = {
        let b = rc.borrow();
        b.shape_header_ptr() as usize
    };
    (header_addr - base) as i32
}

/// The comparison op subset the T2-lite JIT inlines.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum T2Cmp {
    Lt,
    Le,
    Gt,
    Ge,
    Eq,
    Neq,
}

/// The arithmetic op subset the T2-lite JIT inlines.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum T2Arith {
    Add,
    Sub,
    Mul,
    Div,
}

/// Register convention inside a T2-lite function. `BANK` holds the `*mut JsVal`
/// bank base; `OUT` holds the `*mut u64` result slot; `CTX` holds the `*mut
/// T2CallCtx` (P4 — the re-entry context for CALL ops; null when the function
/// has no call op). All three are callee-saved, so they survive a re-entrant
/// CALL (which can clobber every volatile reg AND reallocate the bank) — exactly
/// the P4 aliasing discipline: hold NOTHING across a call except these saved
/// pointers (BANK/CTX) + RSP, and re-read every bank slot AFTER the call. Scratch
/// is all volatile.
const T2_BANK: R64 = R64::Rbx;
const T2_OUT: R64 = R64::Rdi;
const T2_CTX: R64 = R64::Rsi;
const T2_XA: Xmm = Xmm::Xmm0;
const T2_XB: Xmm = Xmm::Xmm1;

thread_local! {
    /// DEOPT-FUZZ force mode: when `Some(P)`, op at bytecode index `P` is forced to
    /// deopt at its op boundary — an unconditional `jmp` to a dedicated FORCE stub
    /// (bc_pc == P) is emitted as the FIRST instruction of op P's codegen, BEFORE
    /// the op runs (so the bank is the exact pre-op register image, identical to a
    /// real input-guard miss). Used by the deopt-fuzz oracle to make EVERY op's
    /// resume point fire one-at-a-time over type-correct inputs, proving the
    /// resumed-on-VM result == the tree-walk oracle for each.
    static T2_FORCE_DEOPT_PC: std::cell::Cell<Option<usize>> = const { std::cell::Cell::new(None) };
}

/// Set (or clear with `None`) the deopt-fuzz force-deopt op index for the next
/// `compile_t2lite_with_deopt` call. Returns the previous value. Test-only — the
/// production path never sets it (so ops are emitted normally).
pub fn set_force_deopt_pc(pc: Option<usize>) -> Option<usize> {
    T2_FORCE_DEOPT_PC.with(|c| {
        let prev = c.get();
        c.set(pc);
        prev
    })
}

thread_local! {
    /// T4 EXTENSION-1 FUZZER HOOK (test-only). When set to `Some(call_pc)`, the
    /// T2 RESUME runner treats a forced deopt landing at bytecode op `call_pc`
    /// (which must be a `CallFn`/`CallValue`) as an INLINED-FRAME deopt: instead
    /// of the ordinary single-frame `bank.decode_to_values()` + resume, it builds
    /// an `osr::InlinedDeoptSite` for that call op and reconstructs the CALLER
    /// frame via `osr::reconstruct_caller_frame` (the INLINE-DEOPT-TO-CALLER
    /// math), then resumes the caller at the Call op. This drives the Extension-1
    /// reconstruction over the REAL JIT bank so the inlined-frame-deopt fuzzer can
    /// prove it is byte-identical to the un-inlined VM — BEFORE any inliner exists.
    /// Never set in production (no env path; only the in-process setter below).
    static T2_FORCE_INLINED_RECONSTRUCT_PC: std::cell::Cell<Option<usize>> =
        const { std::cell::Cell::new(None) };
}

/// Set (or clear with `None`) the T4 Extension-1 inlined-frame-reconstruction
/// fuzzer hook for the resume runner. Returns the previous value. Test-only.
pub fn set_force_inlined_reconstruct_pc(pc: Option<usize>) -> Option<usize> {
    T2_FORCE_INLINED_RECONSTRUCT_PC.with(|c| {
        let prev = c.get();
        c.set(pc);
        prev
    })
}

/// Read the T4 Extension-1 inlined-frame-reconstruction fuzzer hook. The resume
/// runner consults this on a deopt to decide whether to route through the
/// inlined-frame caller-reconstruction path (test-only; `None` in production).
pub fn force_inlined_reconstruct_pc() -> Option<usize> {
    T2_FORCE_INLINED_RECONSTRUCT_PC.with(|c| c.get())
}

/// Emit: load `bank[slot]` (a `JsVal`) into `xmm`, jumping to a (later-patched)
/// DEOPT site — its rel32 byte offset is pushed to `deopt_patches` — if the slot
/// is NOT a number. Uses RAX/RCX/R10/R11 (all volatile) as scratch.
///
/// CORRECTNESS (the silent-corruption-critical half): the number test mirrors
/// `JsVal::is_number` EXACTLY — `(bits & QNAN_MASK) != QNAN_BITS` ⇒ a plain
/// double (the hot lane), which we `movq` straight to xmm. Otherwise it is
/// boxed; we admit ONLY the int32 lane (top-16 bits == `QNAN_BITS |
/// TAG_INT32<<48`), sign-extend its low 32 bits and `cvtsi2sd`. Every other
/// boxed value (undefined/null/bool/object/string/…) DEOPTs to the VM.
fn t2_emit_load_num(
    em: &mut Emitter,
    slot: u16,
    xmm: Xmm,
    deopt_patches: &mut Vec<(usize, crate::osr::DeoptReason)>,
) {
    // rax = bank[slot]
    em.mov_r64_mem(R64::Rax, T2_BANK, (slot as i32) * 8);
    // rcx = rax & QNAN_MASK ; cmp rcx, QNAN_BITS
    em.mov_r64_r64(R64::Rcx, R64::Rax);
    em.mov_r64_imm64(R64::R10, JV_QNAN_MASK as i64);
    em.and_r64_r64(R64::Rcx, R64::R10);
    em.mov_r64_imm64(R64::R11, JV_QNAN_BITS as i64);
    em.cmp_r64_r64(R64::Rcx, R64::R11);
    // jne is_double  (not in the boxed space ⇒ a plain double / number)
    let to_double = em.jcc_rel32_placeholder(Cc::NotEqual);
    // boxed: admit only the int32 lane. rcx = rax & TOP16_MASK ; cmp INT32_TOP16
    em.mov_r64_r64(R64::Rcx, R64::Rax);
    em.mov_r64_imm64(R64::R10, JV_TOP16_MASK as i64);
    em.and_r64_r64(R64::Rcx, R64::R10);
    em.mov_r64_imm64(R64::R11, JV_INT32_TOP16 as i64);
    em.cmp_r64_r64(R64::Rcx, R64::R11);
    // jne DEOPT  (boxed and not int32 ⇒ not a number)
    let dp = em.jcc_rel32_placeholder(Cc::NotEqual);
    deopt_patches.push((dp, crate::osr::DeoptReason::NonNumber));
    // int32 lane: sign-extend the low 32 bits, cvtsi2sd → xmm.
    em.movsxd_r64_r32(R64::Rcx, R64::Rax);
    em.cvtsi2sd_xmm_r64(xmm, R64::Rcx);
    let to_done = em.jmp_rel32_placeholder();
    // is_double: movq xmm, rax  (raw double bits).
    em.patch_rel32(to_double);
    em.movq_xmm_r64(xmm, R64::Rax);
    // done:
    em.patch_rel32(to_done);
}

/// Emit: box the f64 result in `xmm` to a `JsVal` and store it to `bank[slot]`.
/// CRITICAL: a computed NaN result is canonicalized to `CANONICAL_NAN` before
/// boxing (an arbitrary-payload NaN would alias the boxed tag space — the silent
/// corruption hazard, now handled inline). Any non-NaN double's bits are already
/// a valid number-lane `JsVal`. Uses RAX/RCX as scratch. The result is always an
/// IMMEDIATE (a number), so the owning store's inc/dec are no-ops for the new
/// value — but it STILL goes through `t2_emit_bank_store` in Heap mode so the OLD
/// slot (which may have held a heap value) is dec'd (no leak).
fn t2_emit_box_store(em: &mut Emitter, xmm: Xmm, slot: u16, store_mode: T2StoreMode) {
    // NaN test: ucomisd xmm, xmm sets PF=1 iff unordered (NaN).
    em.ucomisd_xmm_xmm(xmm, xmm);
    let to_nan = em.jcc_rel32_placeholder(Cc::Parity);
    // not NaN: rax = raw bits (a valid double JsVal).
    em.movq_r64_xmm(R64::Rax, xmm);
    let to_store = em.jmp_rel32_placeholder();
    // is_nan: rax = CANONICAL_NAN.
    em.patch_rel32(to_nan);
    em.mov_r64_imm64(R64::Rax, JV_CANONICAL_NAN as i64);
    // store: result JsVal is in RAX.
    em.patch_rel32(to_store);
    t2_emit_bank_store(em, slot, store_mode);
}

/// Emit a comparison: both operands already in `T2_XA`/`T2_XB` as f64; set
/// `bank[dst]` to the boolean `JsVal` (`true`/`false`) per JS/IEEE semantics
/// (relational ops are false when either operand is NaN; `===` is false for NaN,
/// `!==` is true). Uses RAX/RCX as scratch.
///
/// Operand order is chosen so the "true" condition for the relational ops is
/// `seta`/`setae` (CF=0 ∧ ZF=0) — both are 0 when unordered (NaN sets CF=1), so
/// NaN ⇒ false automatically. Eq/Neq combine the equal flag with the
/// parity flag to honor NaN (ZF=1 is also set by unordered, so `==` must also
/// require PF=0).
fn t2_emit_cmp_store(em: &mut Emitter, cmp: T2Cmp, dst: u16, store_mode: T2StoreMode) {
    match cmp {
        // a<b ⟺ b>a ; a<=b ⟺ b>=a — swap operands so we can use seta/setae.
        T2Cmp::Lt => {
            em.ucomisd_xmm_xmm(T2_XB, T2_XA);
            em.setcc(Cc::Above, R64::Rax);
            em.movzx_r64_r8(R64::Rax, R64::Rax);
        }
        T2Cmp::Le => {
            em.ucomisd_xmm_xmm(T2_XB, T2_XA);
            em.setcc(Cc::AboveEq, R64::Rax);
            em.movzx_r64_r8(R64::Rax, R64::Rax);
        }
        T2Cmp::Gt => {
            em.ucomisd_xmm_xmm(T2_XA, T2_XB);
            em.setcc(Cc::Above, R64::Rax);
            em.movzx_r64_r8(R64::Rax, R64::Rax);
        }
        T2Cmp::Ge => {
            em.ucomisd_xmm_xmm(T2_XA, T2_XB);
            em.setcc(Cc::AboveEq, R64::Rax);
            em.movzx_r64_r8(R64::Rax, R64::Rax);
        }
        // a==b: equal AND ordered  → sete & setnp.
        T2Cmp::Eq => {
            em.ucomisd_xmm_xmm(T2_XA, T2_XB);
            em.setcc(Cc::Equal, R64::Rax);
            em.movzx_r64_r8(R64::Rax, R64::Rax);
            em.setcc(Cc::NoParity, R64::Rcx);
            em.movzx_r64_r8(R64::Rcx, R64::Rcx);
            em.and_r64_r64(R64::Rax, R64::Rcx);
        }
        // a!=b: not-equal OR unordered → setne | setp.
        T2Cmp::Neq => {
            em.ucomisd_xmm_xmm(T2_XA, T2_XB);
            em.setcc(Cc::NotEqual, R64::Rax);
            em.movzx_r64_r8(R64::Rax, R64::Rax);
            em.setcc(Cc::Parity, R64::Rcx);
            em.movzx_r64_r8(R64::Rcx, R64::Rcx);
            em.or_r64_r64(R64::Rax, R64::Rcx);
        }
    }
    // rax ∈ {0,1}. Boolean JsVal = JV_FALSE + rax  (JV_TRUE == JV_FALSE + 1).
    em.mov_r64_imm64(R64::Rcx, JV_FALSE as i64);
    em.add_r64_r64(R64::Rax, R64::Rcx);
    // The result is an IMMEDIATE (bool), so its inc is a no-op — but route through
    // the owning store so an OLD heap value in `dst` is dec'd (no leak) in Heap mode.
    t2_emit_bank_store(em, dst, store_mode);
}

/// One T2 inlinable GetProp site (M4.3 T2 Phase 1). Built by the caller from the
/// site's WARMED `PropIc` (the poly-≤4 `(shape, slot)` entries, slot pre-resolved
/// per shape) PLUS the receiver-is-pure-arg proof. The JIT bakes the shapes as
/// `cmp; je` guards against the inline header at `shape_off`, and on a hit calls
/// `helper_addr` (`rt_getprop_slot_immediate`) with `(obj_ptr, slot)`.
#[derive(Debug, Clone)]
pub struct T2GetPropSite {
    /// Warmed `(shape_id, slot)` entries (≤4). Never empty, never `DICT_SHAPE`.
    pub shapes_slots: Vec<(u32, u32)>,
    /// T2 Phase 3: when `true`, this site may produce a HEAP result (Object /
    /// Array / String) that is stored into the bank with OWNING semantics via the
    /// `rt_getprop_slot_owning_store` helper (which inc/dec's the slot's `Rc`).
    /// When `false` (P1 behaviour), the site uses the immediate-only helper and
    /// declines/deopts on any non-immediate slot. The caller sets this only when
    /// the bank is GC-registered + owning for the run (the safety prerequisite).
    pub heap_result: bool,
}

/// Compile-time inputs for the T2 inline GetProp fast path. Passing `None` (or a
/// per-op plan that returns `None`) makes EVERY GetProp decline the whole compile
/// (the numeric-only path, byte-identical to before).
pub struct T2GetPropConfig<'a> {
    /// Per-bytecode-index inlinable GetProp site (only the indices that are
    /// `Op::GetProp` with a pure-arg receiver + warm IC return `Some`).
    pub site_at: &'a dyn Fn(usize) -> Option<T2GetPropSite>,
    /// Byte offset from the receiver `Rc::as_ptr` to the `u32` shape-id header
    /// (`t2_shape_header_offset()`).
    pub shape_off: i32,
    /// Address of `rt_getprop_slot_immediate` (cast to `usize`).
    pub helper_addr: usize,
    /// Address of `rt_getprop_slot_owning_store` (cast to `usize`) — used by a
    /// site whose `heap_result` is `true` (T2 Phase 3 owning heap store). `0`
    /// when no heap site is configured (P1-only callers leave it unset).
    pub heap_helper_addr: usize,
    /// Address of `rt_getidx_owning_store` (cast to `usize`) — the COMPUTED-ARRAY-
    /// READ (`arr[j]`) owning-store helper. `0` when GetIdx is not wired (numeric /
    /// non-heap callers), in which case any `Op::GetIdx` DECLINES the whole compile
    /// (the array fast path needs the owning + GC-rooted bank, like heap GetProp).
    pub getidx_helper_addr: usize,
    /// Address of `rt_setidx_owning_store` (cast to `usize`) — the COMPUTED-ARRAY-
    /// WRITE (`arr[j] = v`) owning element-replace helper. `0` when SetIdx is not
    /// wired, in which case any `Op::SetIdx` DECLINES the whole compile.
    pub setidx_helper_addr: usize,
}

/// Compile-time inputs for the T2 Phase 4 CALL inlining path. When `None` (the
/// numeric-only callers), every `CallValue`/`CallFn`/`LoadGlobal*` op DECLINES the
/// whole compile exactly as before. When `Some`, those ops emit a `call` to the
/// re-entry helpers, which re-dispatch through the VM. The bank MUST be the OWNING
/// + GC-rooted bank (heap store mode) for a call site — a re-entrant call can
/// produce a heap result, can GC, and can hold heap args, all of which require the
/// owning per-slot invariant. The caller wires this only in heap mode.
pub struct T2CallConfig {
    /// Address of the `CallValue` re-entry helper (`extern "system"`).
    pub call_helper_addr: usize,
    /// Address of the `CallFn` re-entry helper (`extern "system"`).
    pub call_fn_helper_addr: usize,
    /// Address of the `LoadGlobal[Checked]` re-entry helper (`extern "system"`):
    /// loads `globals[consts[name_k]]` into a bank slot with owning semantics.
    pub load_global_helper_addr: usize,
}

impl T2CallConfig {
    #[inline]
    fn call_fn_helper_addr(&self) -> usize {
        self.call_fn_helper_addr
    }
}

/// Compile `code` (a bytecode function) to a T2-lite INLINED-`JsVal` native
/// function. Returns `None` (decline → run on the VM) on any op outside the
/// supported subset, a register out of range, or a non-number constant in an
/// arithmetic path. The supported subset is:
///   LoadConst(number) / LoadUndef / LoadTrue / LoadFalse / LoadNull / Move,
///   Add / Sub / Mul / Div, Lt/Le/Gt/Ge/Eq/Neq,
///   Jmp / JmpIfFalse / Ret,
///   GetProp (ONLY when `getprop` supplies an inlinable site for that index —
///   shape-guarded, immediate-result; else the GetProp declines the compile).
/// `n_params`/args are bound into the bank by the caller; this compiles the body.
///
/// `store_mode` selects how slots are written: `Numeric` (raw store, the P1 path,
/// borrowed immediate-only bank) or `Heap` (every store routed through
/// [`rt_bank_store`] for uniform owning per-slot ownership — required for a heap
/// value to live in a slot). The two are gated by `CV_T2_HEAP` at the caller.
/// Back-compat wrapper returning only the code bytes (drops the deopt-site table).
/// Used by the unit tests that assert compile success/decline; the production path
/// uses [`compile_t2lite_with_deopt`] to install the resume map alongside.
pub fn compile_t2lite(
    code: &[crate::bytecode::Op],
    const_f64: impl Fn(u16) -> Option<f64>,
    getprop: Option<&T2GetPropConfig<'_>>,
    store_mode: T2StoreMode,
    call_cfg: Option<&T2CallConfig>,
) -> Option<Vec<u8>> {
    compile_t2lite_with_deopt(code, const_f64, getprop, store_mode, call_cfg).map(|(c, _)| c)
}

/// Compile a bytecode function to T2-lite native code AND its per-guard deopt-site
/// table (the T2 Phase-5 resume map). See [`compile_t2lite`] for the op subset.
/// The returned `Vec<DeoptSite>` is indexed by the `deopt_id` a guard's stub writes
/// to `*out`; the runner uses it to resume the VM mid-function on a guard miss.
pub fn compile_t2lite_with_deopt(
    code: &[crate::bytecode::Op],
    const_f64: impl Fn(u16) -> Option<f64>,
    getprop: Option<&T2GetPropConfig<'_>>,
    store_mode: T2StoreMode,
    call_cfg: Option<&T2CallConfig>,
) -> Option<(Vec<u8>, Vec<crate::osr::DeoptSite>)> {
    use crate::bytecode::Op;
    if code.is_empty() {
        return None;
    }
    let n = code.len();

    // T2 Phase 5 — DEOPT-SOUNDNESS pre-scan (compile-time decline).
    //
    // With REAL per-guard resume deopt (every guard returns `T2_DEOPT_RESUME` and
    // the runner resumes the VM mid-function at the guard's `bc_pc`), the old P4
    // positional rules — decline loops-with-calls / guard-after-call / second-call
    // / SetProp-after-committed-effect — are GONE: a guard that fires AFTER a
    // committed side effect now resumes the VM from that guard's op boundary (NOT a
    // whole-function re-run from ip=0), so the committed effect is never re-done.
    // The JIT bank IS the exact pre-op VM register image at every guard (each op
    // stores its result to its bank slot before the next op, and no value lives in
    // a host register across an op boundary — the identity reconstruction map), so
    // the resumed VM continues bit-identically.
    //
    // What STILL declines:
    //   * A try-handler op (TryEnter/TryExit): a resumed frame assumes `try_stack`
    //     is EMPTY at `bc_pc` (the resume builds a fresh empty try_stack), so a
    //     function that could be mid-try at a guard would resume with the wrong
    //     handler state. try_stack reconstruction is a separate proof — out of
    //     scope; decline these outright.
    //   * A call op WITHOUT a `call_cfg`: the numeric-only callers can't re-enter
    //     the VM for a call, so the op declines the whole compile (unchanged).
    {
        // Decline try-handler-containing functions (resume assumes empty try_stack).
        for op in code {
            if matches!(op, Op::TryEnter { .. } | Op::TryExit { .. }) {
                return None;
            }
        }
        // A call op requires `call_cfg`; without it (numeric-only callers) the op
        // declines at its codegen site anyway, but bail early so the decline reason
        // is unambiguous.
        let is_call = |op: &Op| matches!(op, Op::CallValue { .. } | Op::CallFn { .. });
        if call_cfg.is_none() && code.iter().any(is_call) {
            return None;
        }
    }

    let mut em = Emitter::new();
    let mut offsets = vec![0usize; n]; // bytecode index → machine-code offset
    let mut jump_patches: Vec<(usize, usize)> = Vec::new(); // (disp off, target bc idx)
    // T2 Phase 5 — RESUME deopt patches: each input guard pushes `(rel32 off,
    // reason)`; the main loop tags each with the current op's `bc_pc` (the op
    // boundary, BEFORE the op's output store) and builds a per-guard `DeoptSite` +
    // a resume stub that returns `T2_DEOPT_RESUME` with the site's `deopt_id`.
    let mut deopt_patches: Vec<(usize, crate::osr::DeoptReason)> = Vec::new();
    // TIER-A deopt patches → the shared `T2_DEOPT` pad (whole-function re-run).
    // Used ONLY for pre-side-effect declines (call/loadglobal non-callable callee)
    // and the defensive fall-through, where re-running from ip=0 is identical.
    let mut tier_a_deopt_patches: Vec<usize> = Vec::new();
    let mut epilogue_patches: Vec<usize> = Vec::new(); // rel32 offsets → RET epilogue
    let mut threw_patches: Vec<usize> = Vec::new(); // rel32 offsets → THREW pad (P4 calls)
    let mut deadline_patches: Vec<usize> = Vec::new(); // rel32 offsets → DEADLINE pad (P4)
    // The per-guard resume site table + the patches that target each site's stub.
    // `deopt_sites[k]` is the k-th `DeoptSite`; `resume_patches` holds `(guard rel32
    // off, site index)` resolved after the stubs are emitted.
    let mut deopt_sites: Vec<crate::osr::DeoptSite> = Vec::new();
    let mut resume_patches: Vec<(usize, usize)> = Vec::new();

    // Prolog: preserve RBX/RDI/RSI (callee-saved), load the bank/out/ctx pointers.
    // Win64: RCX = bank, RDX = out, R8 = ctx (P4 re-entry context; may be null for
    // a call-free function). We push a FOURTH callee-saved reg (RBP, harmless — we
    // never use it) PURELY to keep the stack-alignment invariant identical to the
    // pre-P4 code: 4 pushes (32 bytes) leave RSP ≡ 8 (mod 16) at every op boundary
    // (entry ≡8; 8 - 32 ≡ 8 mod 16), so every existing helper-call site's `sub 40`
    // (≡8 → ≡0 at the `call`) accounting stays byte-correct. (With only 3 pushes
    // the boundary would be ≡0 and every `sub 40` would misalign.)
    em.push_r64(T2_BANK);
    em.push_r64(T2_OUT);
    em.push_r64(T2_CTX);
    em.push_r64(R64::Rbp);
    em.mov_r64_r64(T2_BANK, R64::Rcx);
    em.mov_r64_r64(T2_OUT, R64::Rdx);
    em.mov_r64_r64(T2_CTX, R64::R8);

    // Helper closures can't borrow `em` mutably twice, so we inline arith/cmp
    // dispatch in the loop below.
    // DEOPT-FUZZ: read the forced-deopt op index once for this compile.
    let force_deopt_pc = T2_FORCE_DEOPT_PC.with(|c| c.get());
    let mut i = 0usize;
    while i < n {
        offsets[i] = em.code.len();
        // Snapshot the resume-patch count BEFORE this op so we can attribute every
        // guard emitted during op `i` to bc_pc == `i` (the op boundary, before the
        // op stores its output). This is the IDENTITY invariant: at any such guard
        // the bank is the exact pre-op VM register image, so resuming the VM at
        // `i` re-executes this op + the rest with bit-identical results.
        let deopt_before = deopt_patches.len();
        // DEOPT-FUZZ force: if op `i` is the forced one, emit an UNCONDITIONAL jump
        // to a deopt stub (bc_pc == i) as the op's FIRST instruction — BEFORE the op
        // runs, so the bank is the exact pre-op image (identical to a real input
        // guard miss). The op's normal codegen follows but is unreachable. This
        // pushes a resume patch attributed (by the per-op drain) to bc_pc == i.
        if force_deopt_pc == Some(i) {
            let fj = em.jmp_rel32_placeholder();
            deopt_patches.push((fj, crate::osr::DeoptReason::NonNumber));
        }
        match code[i] {
            Op::LoadConst { dst, k } => {
                let f = const_f64(k)?;
                // Box the constant at COMPILE time (canonicalize NaN) and store.
                let bits = if f.is_nan() { JV_CANONICAL_NAN } else { f.to_bits() };
                em.mov_r64_imm64(R64::Rax, bits as i64);
                t2_emit_bank_store(&mut em, dst, store_mode);
            }
            Op::LoadUndef { dst } => {
                // ECMA: undefined coerces to NaN numerically. But we store the
                // ACTUAL undefined JsVal so a later op that reads it deopts
                // (undefined is not a number) — matching the VM, which would do
                // ToNumber(undefined)=NaN only at the arithmetic site. Storing
                // the canonical NaN here would silently diverge if the value is
                // observed via a non-arith op. Undefined singleton bits:
                // QNAN_BITS | (TAG_SINGLETON(6)<<48) | 0 = 0xFFFE_0000_0000_0000.
                em.mov_r64_imm64(R64::Rax, 0xFFFE_0000_0000_0000u64 as i64);
                t2_emit_bank_store(&mut em, dst, store_mode);
            }
            Op::LoadTrue { dst } => {
                em.mov_r64_imm64(R64::Rax, JV_TRUE as i64);
                t2_emit_bank_store(&mut em, dst, store_mode);
            }
            Op::LoadFalse { dst } => {
                em.mov_r64_imm64(R64::Rax, JV_FALSE as i64);
                t2_emit_bank_store(&mut em, dst, store_mode);
            }
            Op::LoadNull { dst } => {
                // null singleton: QNAN_BITS | (6<<48) | 1 = 0xFFFE_0000_0000_0001.
                em.mov_r64_imm64(R64::Rax, 0xFFFE_0000_0000_0001u64 as i64);
                t2_emit_bank_store(&mut em, dst, store_mode);
            }
            Op::Move { dst, src } => {
                // Load the 8-byte JsVal from src into RAX, store to dst. In Heap
                // mode the store goes through the owning helper so an aliased heap
                // value is inc'd (and the old dst dec'd) — no double-dec on Drop.
                em.mov_r64_mem(R64::Rax, T2_BANK, (src as i32) * 8);
                t2_emit_bank_store(&mut em, dst, store_mode);
            }
            Op::Add { dst, lhs, rhs } => {
                t2_emit_arith(&mut em, T2Arith::Add, dst, lhs, rhs, &mut deopt_patches, store_mode);
            }
            Op::Sub { dst, lhs, rhs } => {
                t2_emit_arith(&mut em, T2Arith::Sub, dst, lhs, rhs, &mut deopt_patches, store_mode);
            }
            Op::Mul { dst, lhs, rhs } => {
                t2_emit_arith(&mut em, T2Arith::Mul, dst, lhs, rhs, &mut deopt_patches, store_mode);
            }
            Op::Div { dst, lhs, rhs } => {
                t2_emit_arith(&mut em, T2Arith::Div, dst, lhs, rhs, &mut deopt_patches, store_mode);
            }
            Op::Lt { dst, lhs, rhs } => {
                t2_emit_load_num(&mut em, lhs, T2_XA, &mut deopt_patches);
                t2_emit_load_num(&mut em, rhs, T2_XB, &mut deopt_patches);
                t2_emit_cmp_store(&mut em, T2Cmp::Lt, dst, store_mode);
            }
            Op::Le { dst, lhs, rhs } => {
                t2_emit_load_num(&mut em, lhs, T2_XA, &mut deopt_patches);
                t2_emit_load_num(&mut em, rhs, T2_XB, &mut deopt_patches);
                t2_emit_cmp_store(&mut em, T2Cmp::Le, dst, store_mode);
            }
            Op::Gt { dst, lhs, rhs } => {
                t2_emit_load_num(&mut em, lhs, T2_XA, &mut deopt_patches);
                t2_emit_load_num(&mut em, rhs, T2_XB, &mut deopt_patches);
                t2_emit_cmp_store(&mut em, T2Cmp::Gt, dst, store_mode);
            }
            Op::Ge { dst, lhs, rhs } => {
                t2_emit_load_num(&mut em, lhs, T2_XA, &mut deopt_patches);
                t2_emit_load_num(&mut em, rhs, T2_XB, &mut deopt_patches);
                t2_emit_cmp_store(&mut em, T2Cmp::Ge, dst, store_mode);
            }
            // `===`/`!==` AND `==`/`!=` map to the same f64 comparison HERE: once
            // both operands pass the inline is-number tag-check, abstract
            // equality (`==`) performs NO coercion (number==number) and is
            // identical to strict equality (`===`). If either operand is not a
            // number, the load deopts and the VM does the correct LooseEq
            // coercion — so both lower safely to the numeric compare.
            Op::Eq { dst, lhs, rhs } | Op::LooseEq { dst, lhs, rhs } => {
                t2_emit_load_num(&mut em, lhs, T2_XA, &mut deopt_patches);
                t2_emit_load_num(&mut em, rhs, T2_XB, &mut deopt_patches);
                t2_emit_cmp_store(&mut em, T2Cmp::Eq, dst, store_mode);
            }
            Op::Neq { dst, lhs, rhs } | Op::LooseNeq { dst, lhs, rhs } => {
                t2_emit_load_num(&mut em, lhs, T2_XA, &mut deopt_patches);
                t2_emit_load_num(&mut em, rhs, T2_XB, &mut deopt_patches);
                t2_emit_cmp_store(&mut em, T2Cmp::Neq, dst, store_mode);
            }
            Op::Jmp { target } => {
                let o = em.jmp_rel32_placeholder();
                jump_patches.push((o, target as usize));
            }
            Op::JmpIfFalse { cond, target } => {
                t2_emit_jmp_if_false(&mut em, cond, &mut jump_patches, target as usize, &mut deopt_patches);
            }
            Op::Ret { src } => {
                // Store the result JsVal to *out, then jump to the epilogue
                // (which returns T2_RETURNED).
                em.mov_r64_mem(R64::Rax, T2_BANK, (src as i32) * 8);
                em.mov_mem_r64(T2_OUT, 0, R64::Rax);
                let e = em.jmp_rel32_placeholder();
                epilogue_patches.push(e);
            }
            // T2 Phase 1 INLINE GetProp: only when the caller supplies an
            // inlinable site for THIS bytecode index (receiver is a pure arg +
            // the IC is warm). The site bakes the warmed poly-≤4 shapes as inline
            // header guards; a hit calls the audited helper which returns an
            // IMMEDIATE (or DEOPT). Any miss / non-immediate slot / cold-or-mega
            // IC / non-arg receiver declines here → the whole compile declines →
            // VM runs it (correct).
            Op::GetProp { dst, obj, .. } => {
                let cfg = getprop?;
                let site = (cfg.site_at)(i)?;
                if site.heap_result {
                    // T2 Phase 3: this site may produce a HEAP result stored into
                    // the OWNING bank via `rt_getprop_slot_owning_store`. Decline
                    // the compile if no heap helper was wired (a misconfiguration
                    // — never emit a call to address 0).
                    if cfg.heap_helper_addr == 0 {
                        return None;
                    }
                    t2_emit_getprop_owning(
                        &mut em,
                        dst,
                        obj,
                        &site,
                        cfg.shape_off,
                        cfg.heap_helper_addr,
                        &mut deopt_patches,
                    );
                } else {
                    t2_emit_getprop_immediate(
                        &mut em,
                        dst,
                        obj,
                        &site,
                        cfg.shape_off,
                        cfg.helper_addr,
                        &mut deopt_patches,
                        store_mode,
                    );
                }
            }
            // T2 GetIdx — COMPUTED ARRAY READ `bank[dst] = bank[obj][bank[key]]`.
            // Emits the is-array + non-negative-integer-index guards inline, then
            // calls `rt_getidx_owning_store` to do the bounds-checked, owning
            // element read (OOB → undefined; hole/accessor/special → DEOPT). Needs
            // the owning + GC-rooted bank, so it requires a wired helper address
            // (heap mode); declines otherwise (the numeric path never wires it).
            Op::GetIdx { dst, obj, key } => {
                let cfg = getprop?;
                if cfg.getidx_helper_addr == 0 {
                    return None;
                }
                t2_emit_getidx(
                    &mut em,
                    dst,
                    obj,
                    key,
                    cfg.getidx_helper_addr,
                    &mut deopt_patches,
                );
            }
            // T2 SetIdx — COMPUTED ARRAY WRITE `bank[obj][bank[key]] = bank[src]`.
            // Emits the is-array + non-negative-integer-index guards inline, then
            // calls `rt_setidx_owning_store` for the in-bounds owning element
            // replace; an OOB write (extend / hole-create — a structural change) or
            // a non-admitted element returns DEOPT and resumes the VM. SetIdx is a
            // SIDE EFFECT: the helper performs the COMMITTED write before any later
            // op's guard, and a later guard's resume continues AFTER it (P5), so the
            // write is never duplicated. Requires the owning bank; declines without
            // a wired helper.
            Op::SetIdx { obj, key, src } => {
                let cfg = getprop?;
                if cfg.setidx_helper_addr == 0 {
                    return None;
                }
                t2_emit_setidx(
                    &mut em,
                    obj,
                    key,
                    src,
                    cfg.setidx_helper_addr,
                    &mut deopt_patches,
                );
            }
            // P4 — LOAD a global into a bank slot (the callee load before a call,
            // and any other value-context global read). Routes through the owning
            // re-entry helper `rt_load_global(ctx, name_k, bank, dst)`: it reads
            // `globals[consts[name_k]]`, owning-stores it into `bank[dst]`, and
            // returns RETURNED, or THREW (a `LoadGlobalChecked` of an undeclared
            // name → catchable ReferenceError) / DEADLINE. Declines without a
            // call_cfg (numeric-only callers never see this op anyway).
            Op::LoadGlobal { dst, name_k } | Op::LoadGlobalChecked { dst, name_k } => {
                let ccfg = call_cfg?;
                // `checked` = 1 means an undeclared name throws (value context);
                // 0 (plain LoadGlobal) yields undefined. Pack into arg.
                let checked: u64 = matches!(code[i], Op::LoadGlobalChecked { .. }) as u64;
                t2_emit_load_global(
                    &mut em,
                    dst,
                    name_k,
                    checked,
                    ccfg,
                    &mut threw_patches,
                    &mut deadline_patches,
                );
            }
            // P4 — CALL a value (the dominant call shape in a per-fn module: a
            // global/local callee + `CallValue`). All inputs already live in bank
            // slots (callee, this, args). The re-entry helper stores the result
            // into `bank[dst]` (owning) and returns RETURNED / THREW / DEADLINE /
            // DEOPT. ALIASING: we hold nothing across the call except BANK/CTX/RSP
            // (callee-saved); the helper re-reads the bank, and we re-read after.
            Op::CallValue { dst, callee, this_reg, first_arg, n_args } => {
                let ccfg = call_cfg?;
                t2_emit_call_value(
                    &mut em,
                    dst,
                    callee,
                    this_reg,
                    first_arg,
                    n_args,
                    ccfg,
                    &mut deopt_patches,
                    &mut threw_patches,
                    &mut deadline_patches,
                );
            }
            // P4 — CALL a module-local function by index (`CallFn`). The callee is
            // NOT in a bank slot (it's a compile-time `fn_idx`), so we pass the
            // fn_idx in the callee field with a CALLFN marker; the helper runs
            // `run_function(module, fn_idx, …)`. `this` is always undefined for a
            // direct CallFn (matches the VM's `Op::CallFn`).
            Op::CallFn { dst, fn_idx, first_arg, n_args } => {
                let ccfg = call_cfg?;
                t2_emit_call_fn(
                    &mut em,
                    dst,
                    fn_idx,
                    first_arg,
                    n_args,
                    ccfg,
                    &mut deopt_patches,
                    &mut threw_patches,
                    &mut deadline_patches,
                );
            }
            // `%` (Mod) has no single SSE instruction (it is `a - trunc(a/b)*b`
            // with truncation toward zero, not IEEE remainder) — out of scope
            // for this minimal proof; decline so the VM runs it. Any other op
            // (property access, bitwise, …) likewise declines.
            _ => return None,
        }
        // Attribute every guard emitted during op `i` to a per-guard `DeoptSite`
        // with bc_pc == `i` (the op boundary — the guard fires on this op's INPUTS
        // before its OUTPUT store, so the bank is the exact pre-op register image).
        // `native_off` is filled in when the site's resume stub is emitted below.
        for (off, reason) in deopt_patches.drain(deopt_before..) {
            let site_idx = deopt_sites.len();
            deopt_sites.push(crate::osr::DeoptSite {
                native_off: 0, // patched after stub emission
                bc_pc: i,
                reason,
            });
            resume_patches.push((off, site_idx));
        }
        i += 1;
    }
    debug_assert!(
        deopt_patches.is_empty(),
        "all resume guards must be drained per-op into DeoptSites"
    );

    // Fall-through guard: a bytecode function always ends in Ret, but never run
    // off the page if it somehow doesn't — treat it as a TIER-A deopt (whole-fn
    // re-run). This never fires (bytecode always ends in Ret), and there is no
    // sound resume bc_pc past the end, so a Tier-A re-run is the safe choice.
    let deopt_fallthrough = em.jmp_rel32_placeholder();
    tier_a_deopt_patches.push(deopt_fallthrough);

    // Shared epilogue tail: restore the 4 pushed callee-saved regs (reverse push
    // order: RBP, RSI/CTX, RDI/OUT, RBX/BANK) and `ret`. RAX (the status tag) is
    // already set by the caller of this closure.
    let emit_restore_ret = |em: &mut Emitter| {
        em.pop_r64(R64::Rbp);
        em.pop_r64(T2_CTX);
        em.pop_r64(T2_OUT);
        em.pop_r64(T2_BANK);
        em.ret();
    };

    // RET epilogue: status = T2_RETURNED.
    let epilogue = em.code.len();
    em.mov_r64_imm64(R64::Rax, T2_RETURNED as i64);
    emit_restore_ret(&mut em);

    // DEOPT pad: status = T2_DEOPT. (*out untouched → VM re-run is identical.)
    let deopt = em.code.len();
    em.mov_r64_imm64(R64::Rax, T2_DEOPT as i64);
    emit_restore_ret(&mut em);

    // P4 — THREW pad: status = T2_THREW. A re-entrant CALL's callee threw a
    // catchable error (already stashed in the ctx out-slot by the helper). This is
    // a POST-CALL exit (the call's side effect is committed), so the runner
    // PROPAGATES the error — it must NOT re-run on the VM (no duplicate call).
    let threw = em.code.len();
    em.mov_r64_imm64(R64::Rax, T2_THREW as i64);
    emit_restore_ret(&mut em);

    // P4 — DEADLINE pad: status = T2_DEADLINE (watchdog fired inside a call).
    // Uncatchable; the runner returns `RuntimeError::Deadline`, never re-runs.
    let deadline = em.code.len();
    em.mov_r64_imm64(R64::Rax, T2_DEADLINE as i64);
    emit_restore_ret(&mut em);

    // T2 Phase 5 — RESUME tail + per-guard stubs. A guard's stub puts its
    // `deopt_id` (the DeoptSite index) into RCX, writes it to `*out`, sets RAX =
    // T2_DEOPT_RESUME, and falls into the shared restore+ret. The runner reads the
    // id from `*out`, looks up the site's `bc_pc`, decodes the bank into VM regs,
    // and resumes the VM mid-function. RCX is scratch at a guard (the guard tests
    // are done), so clobbering it is safe; T2_OUT (RDI, callee-saved) is live.
    //
    // Each stub is `mov rcx, id ; mov [out], rcx ; jmp resume_tail`. We emit the
    // tail first, then one stub per site, recording each stub's `native_off` into
    // its DeoptSite so the runner / tests can introspect it.
    let resume_tail = em.code.len();
    em.mov_r64_imm64(R64::Rax, T2_DEOPT_RESUME as i64);
    emit_restore_ret(&mut em);
    // Per-site stubs. Patch each site's `native_off`, then patch the guard jumps.
    let mut stub_off: Vec<usize> = Vec::with_capacity(deopt_sites.len());
    for (idx, site) in deopt_sites.iter_mut().enumerate() {
        let off = em.code.len();
        site.native_off = off;
        stub_off.push(off);
        em.mov_r64_imm64(R64::Rcx, idx as i64); // deopt_id
        em.mov_mem_r64(T2_OUT, 0, R64::Rcx); // *out = deopt_id
        let j = em.jmp_rel32_placeholder();
        em.patch_rel32_to(j, resume_tail);
    }

    // Resolve patches. A jump target must be a valid bytecode index.
    for (_, t) in &jump_patches {
        if *t >= n {
            return None;
        }
    }
    for (disp_off, t) in jump_patches {
        em.patch_rel32_to(disp_off, offsets[t]);
    }
    for disp_off in &threw_patches {
        em.patch_rel32_to(*disp_off, threw);
    }
    for disp_off in &deadline_patches {
        em.patch_rel32_to(*disp_off, deadline);
    }
    for disp_off in epilogue_patches {
        em.patch_rel32_to(disp_off, epilogue);
    }
    // TIER-A deopts (call decline + fall-through) → the shared whole-fn-re-run pad.
    for disp_off in tier_a_deopt_patches {
        em.patch_rel32_to(disp_off, deopt);
    }
    // RESUME guards → their per-site stub.
    for (disp_off, site_idx) in resume_patches {
        em.patch_rel32_to(disp_off, stub_off[site_idx]);
    }
    Some((em.code, deopt_sites))
}

/// T4 (Maglev-class) P2 — compile an OPTIMIZED numeric bytecode function to native
/// code with REPRESENTATION SELECTION (unboxed Float64) + the proven per-guard
/// resume deopt. This is `compile_t2lite_with_deopt`'s numeric subset PLUS the
/// per-block unboxed-f64 value cache (see the big comment below): same prolog,
/// same DeoptSites, same epilogue/stubs, same bank store-after-every-op — the
/// ONLY difference is that an arithmetic/compare operand whose unboxed f64 is
/// already resident in an XMM from earlier in the same basic block is read
/// straight from that XMM instead of being reloaded + tag-checked + unboxed.
///
/// SUBSET: the numeric/control-flow ops T3 lowers (LoadConst/LoadUndef/bool/Move/
/// Add/Sub/Mul/Div/compares/Jmp/JmpIfFalse/Ret). Any other op (GetProp/Call/heap/
/// try) declines (`None`) — the caller falls to T3/T2/VM. NUMERIC store mode only
/// (the bank holds immediates), so no helper call ever clobbers the XMM cache.
///
/// Returns the code bytes + the per-guard DeoptSite table, exactly like the
/// T2-lite backend, so the EXISTING T3 runner (`run_t3_call`) executes it and the
/// EXISTING resume machinery handles a deopt — T4 needs no new runner.
#[cfg(target_os = "windows")]
pub fn compile_t4_unboxed_with_deopt(
    code: &[crate::bytecode::Op],
    const_f64: impl Fn(u16) -> Option<f64>,
) -> Option<(Vec<u8>, Vec<crate::osr::DeoptSite>)> {
    // The single-function entry: the resume `bc_pc` of every guard is the op's own
    // index (the identity map, exactly as before). P3 inlining uses the `_mapped`
    // entry below with a custom resume-pc table.
    compile_t4_unboxed_with_deopt_mapped(code, const_f64, None)
}

/// T4 (Maglev-class) P3 backend entry — same representation-aware codegen as
/// [`compile_t4_unboxed_with_deopt`], but with a custom per-op RESUME-PC MAP so an
/// INLINED region's guards can deopt to the CALLER's `Call` op (the INLINE-DEOPT-
/// TO-CALLER design) instead of resuming on the fused module (which no longer has
/// the call). `bc_pc_map[i]` is the bytecode index the VM resumes at for a guard
/// emitted during fused op `i`; `None` means the identity map (`bc_pc == i`), used
/// by the single-function path. The resume MODULE is the caller's responsibility
/// (the fused-vs-original choice rides on the `JitFunction`, see `t4::inline`).
///
/// CORRECTNESS: the bank store-after-every-op invariant is unchanged, so at every
/// guard the fused bank is a complete pre-op register image. For an inlined-region
/// op the resume target is the caller's `Call` op (before its `dst` store), so the
/// VM re-runs the ordinary call over the caller register image carried in the bank
/// slots `0..caller_n_regs` — byte-identical to a non-inlined run (the inlined-
/// frame-deopt fuzzer proves this).
#[cfg(target_os = "windows")]
pub fn compile_t4_unboxed_with_deopt_mapped(
    code: &[crate::bytecode::Op],
    const_f64: impl Fn(u16) -> Option<f64>,
    bc_pc_map: Option<&[usize]>,
) -> Option<(Vec<u8>, Vec<crate::osr::DeoptSite>)> {
    use crate::bytecode::Op;
    if code.is_empty() {
        return None;
    }
    if let Some(m) = bc_pc_map {
        if m.len() != code.len() {
            return None; // a malformed map is a compile-time decline (never wrong)
        }
    }
    let n = code.len();
    let store_mode = T2StoreMode::Numeric;

    // Decline ops outside the numeric subset up front (the cache reasoning + the
    // no-XMM-clobber guarantee both rely on the numeric-only subset).
    for op in code {
        let ok = matches!(
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
        );
        if !ok {
            return None;
        }
    }

    // Block-boundary set: every op that is a JUMP TARGET begins a basic block, so
    // the XMM cache (which is per-block) must be invalidated at its entry — a value
    // computed before a back-edge / branch is NOT live in an XMM after the jump.
    // We mark each target index; the main loop invalidates the cache when it
    // reaches a marked op (and after every branch/jump/ret, which END a block).
    let mut is_block_start = vec![false; n];
    is_block_start[0] = true;
    for op in code {
        match *op {
            Op::Jmp { target }
            | Op::JmpIfFalse { target, .. }
            | Op::JmpIfTrue { target, .. } => {
                if (target as usize) < n {
                    is_block_start[target as usize] = true;
                }
            }
            _ => {}
        }
        // The op AFTER a branch/jump/ret also starts a block (a fall-through join).
        // Handled in the loop by invalidating after those ops.
    }

    let mut em = Emitter::new();
    let mut offsets = vec![0usize; n];
    let mut jump_patches: Vec<(usize, usize)> = Vec::new();
    let mut deopt_patches: Vec<(usize, crate::osr::DeoptReason)> = Vec::new();
    let mut tier_a_deopt_patches: Vec<usize> = Vec::new();
    let mut epilogue_patches: Vec<usize> = Vec::new();
    let mut deopt_sites: Vec<crate::osr::DeoptSite> = Vec::new();
    let mut resume_patches: Vec<(usize, usize)> = Vec::new();

    // Prolog — IDENTICAL to the T2-lite backend (4 callee-saved pushes keep RSP ≡ 8
    // (mod 16) at every op boundary; T4 uses only volatile XMM0..=XMM5 so it needs
    // no xmm save area). BANK/OUT/CTX loaded from RCX/RDX/R8 (CTX unused — no calls).
    em.push_r64(T2_BANK);
    em.push_r64(T2_OUT);
    em.push_r64(T2_CTX);
    em.push_r64(R64::Rbp);
    em.mov_r64_r64(T2_BANK, R64::Rcx);
    em.mov_r64_r64(T2_OUT, R64::Rdx);
    em.mov_r64_r64(T2_CTX, R64::R8);

    let force_deopt_pc = T2_FORCE_DEOPT_PC.with(|c| c.get());
    let mut cache = T4ValueCache::new();
    let mut i = 0usize;
    while i < n {
        offsets[i] = em.code.len();
        // BLOCK BOUNDARY: a jump target begins a fresh block — invalidate the XMM
        // cache so cross-block reads reload-with-guard from the bank (the value is
        // not live in an XMM across the branch). This is what makes the per-block
        // cache sound with NO dataflow analysis: out-of-block always reloads.
        if is_block_start[i] {
            cache.invalidate();
        }
        let deopt_before = deopt_patches.len();
        // DEOPT-FUZZ force (same as T2-lite): force a deopt at op `i`'s boundary.
        if force_deopt_pc == Some(i) {
            let fj = em.jmp_rel32_placeholder();
            deopt_patches.push((fj, crate::osr::DeoptReason::NonNumber));
        }
        match code[i] {
            Op::LoadConst { dst, k } => {
                let f = const_f64(k)?;
                let bits = if f.is_nan() { JV_CANONICAL_NAN } else { f.to_bits() };
                em.mov_r64_imm64(R64::Rax, bits as i64);
                t2_emit_bank_store(&mut em, dst, store_mode);
                // The constant's value isn't put in an XMM here (it's stored boxed);
                // forget any stale cache entry so a later read reloads it (cheap —
                // a constant operand is a one-time movq). Keeping it out of the
                // cache avoids a redundant materialize for a const that's never
                // re-read.
                cache.forget(dst);
            }
            Op::LoadUndef { dst } => {
                em.mov_r64_imm64(R64::Rax, 0xFFFE_0000_0000_0000u64 as i64);
                t2_emit_bank_store(&mut em, dst, store_mode);
                cache.forget(dst);
            }
            Op::LoadTrue { dst } => {
                em.mov_r64_imm64(R64::Rax, JV_TRUE as i64);
                t2_emit_bank_store(&mut em, dst, store_mode);
                cache.forget(dst);
            }
            Op::LoadFalse { dst } => {
                em.mov_r64_imm64(R64::Rax, JV_FALSE as i64);
                t2_emit_bank_store(&mut em, dst, store_mode);
                cache.forget(dst);
            }
            Op::LoadNull { dst } => {
                em.mov_r64_imm64(R64::Rax, 0xFFFE_0000_0000_0001u64 as i64);
                t2_emit_bank_store(&mut em, dst, store_mode);
                cache.forget(dst);
            }
            Op::Move { dst, src } => {
                // A Move just copies the JsVal (whatever lane). Do the raw copy as
                // T2-lite does (so a non-number value moves correctly), and update
                // the cache: if `src` is cached, `dst` now aliases the same f64 —
                // copy it into a dst cache XMM; else forget `dst`.
                em.mov_r64_mem(R64::Rax, T2_BANK, (src as i32) * 8);
                t2_emit_bank_store(&mut em, dst, store_mode);
                if let Some(sx) = cache.get(src) {
                    let dx = cache.assign(dst);
                    if dx != sx {
                        em.movsd_xmm_xmm(dx, sx);
                    }
                } else {
                    cache.forget(dst);
                }
            }
            Op::Add { dst, lhs, rhs } => {
                t4_emit_arith(&mut em, T2Arith::Add, dst, lhs, rhs, &mut cache, &mut deopt_patches, store_mode);
            }
            Op::Sub { dst, lhs, rhs } => {
                t4_emit_arith(&mut em, T2Arith::Sub, dst, lhs, rhs, &mut cache, &mut deopt_patches, store_mode);
            }
            Op::Mul { dst, lhs, rhs } => {
                t4_emit_arith(&mut em, T2Arith::Mul, dst, lhs, rhs, &mut cache, &mut deopt_patches, store_mode);
            }
            Op::Div { dst, lhs, rhs } => {
                t4_emit_arith(&mut em, T2Arith::Div, dst, lhs, rhs, &mut cache, &mut deopt_patches, store_mode);
            }
            Op::Lt { dst, lhs, rhs } => {
                t4_emit_cmp(&mut em, T2Cmp::Lt, dst, lhs, rhs, &cache, &mut deopt_patches, store_mode);
                cache.forget(dst); // result is a Bool, not an unboxed f64
            }
            Op::Le { dst, lhs, rhs } => {
                t4_emit_cmp(&mut em, T2Cmp::Le, dst, lhs, rhs, &cache, &mut deopt_patches, store_mode);
                cache.forget(dst);
            }
            Op::Gt { dst, lhs, rhs } => {
                t4_emit_cmp(&mut em, T2Cmp::Gt, dst, lhs, rhs, &cache, &mut deopt_patches, store_mode);
                cache.forget(dst);
            }
            Op::Ge { dst, lhs, rhs } => {
                t4_emit_cmp(&mut em, T2Cmp::Ge, dst, lhs, rhs, &cache, &mut deopt_patches, store_mode);
                cache.forget(dst);
            }
            Op::Eq { dst, lhs, rhs } | Op::LooseEq { dst, lhs, rhs } => {
                t4_emit_cmp(&mut em, T2Cmp::Eq, dst, lhs, rhs, &cache, &mut deopt_patches, store_mode);
                cache.forget(dst);
            }
            Op::Neq { dst, lhs, rhs } | Op::LooseNeq { dst, lhs, rhs } => {
                t4_emit_cmp(&mut em, T2Cmp::Neq, dst, lhs, rhs, &cache, &mut deopt_patches, store_mode);
                cache.forget(dst);
            }
            Op::Jmp { target } => {
                let o = em.jmp_rel32_placeholder();
                jump_patches.push((o, target as usize));
                cache.invalidate(); // end of block
            }
            Op::JmpIfFalse { cond, target } => {
                // Read `cond` from the bank (it is a Bool/number — the cache holds
                // only f64 values; a Bool result was `forget`-ed above, so a fresh
                // bank read is correct). The branch ENDS the block.
                t2_emit_jmp_if_false(&mut em, cond, &mut jump_patches, target as usize, &mut deopt_patches);
                cache.invalidate();
            }
            Op::Ret { src } => {
                // Store the bank's boxed value (the canonical image — NOT the cache,
                // so the returned value is byte-identical to the VM) to *out.
                em.mov_r64_mem(R64::Rax, T2_BANK, (src as i32) * 8);
                em.mov_mem_r64(T2_OUT, 0, R64::Rax);
                let e = em.jmp_rel32_placeholder();
                epilogue_patches.push(e);
                cache.invalidate();
            }
            _ => return None,
        }
        // Attribute every guard emitted during op `i` to a DeoptSite at bc_pc == i
        // (the op boundary) — IDENTICAL to the T2-lite backend.
        for (off, reason) in deopt_patches.drain(deopt_before..) {
            let site_idx = deopt_sites.len();
            // The resume bc_pc is the op's own index (identity map) UNLESS a custom
            // map routes it elsewhere — the P3 inlined-region → caller-Call-op case.
            let resume_pc = bc_pc_map.map(|m| m[i]).unwrap_or(i);
            deopt_sites.push(crate::osr::DeoptSite {
                native_off: 0,
                bc_pc: resume_pc,
                reason,
            });
            resume_patches.push((off, site_idx));
        }
        i += 1;
    }
    debug_assert!(deopt_patches.is_empty());

    // Fall-through guard → Tier-A deopt (same as T2-lite).
    let deopt_fallthrough = em.jmp_rel32_placeholder();
    tier_a_deopt_patches.push(deopt_fallthrough);

    // Epilogue + deopt pads + per-guard resume stubs — IDENTICAL to T2-lite.
    let emit_restore_ret = |em: &mut Emitter| {
        em.pop_r64(R64::Rbp);
        em.pop_r64(T2_CTX);
        em.pop_r64(T2_OUT);
        em.pop_r64(T2_BANK);
        em.ret();
    };
    let epilogue = em.code.len();
    em.mov_r64_imm64(R64::Rax, T2_RETURNED as i64);
    emit_restore_ret(&mut em);
    let deopt = em.code.len();
    em.mov_r64_imm64(R64::Rax, T2_DEOPT as i64);
    emit_restore_ret(&mut em);
    let resume_tail = em.code.len();
    em.mov_r64_imm64(R64::Rax, T2_DEOPT_RESUME as i64);
    emit_restore_ret(&mut em);
    let mut stub_off: Vec<usize> = Vec::with_capacity(deopt_sites.len());
    for (idx, site) in deopt_sites.iter_mut().enumerate() {
        let off = em.code.len();
        site.native_off = off;
        stub_off.push(off);
        em.mov_r64_imm64(R64::Rcx, idx as i64);
        em.mov_mem_r64(T2_OUT, 0, R64::Rcx);
        let j = em.jmp_rel32_placeholder();
        em.patch_rel32_to(j, resume_tail);
    }
    for (_, t) in &jump_patches {
        if *t >= n {
            return None;
        }
    }
    for (disp_off, t) in jump_patches {
        em.patch_rel32_to(disp_off, offsets[t]);
    }
    for disp_off in epilogue_patches {
        em.patch_rel32_to(disp_off, epilogue);
    }
    for disp_off in tier_a_deopt_patches {
        em.patch_rel32_to(disp_off, deopt);
    }
    for (disp_off, site_idx) in resume_patches {
        em.patch_rel32_to(disp_off, stub_off[site_idx]);
    }
    Some((em.code, deopt_sites))
}

#[cfg(not(target_os = "windows"))]
pub fn compile_t4_unboxed_with_deopt(
    _code: &[crate::bytecode::Op],
    _const_f64: impl Fn(u16) -> Option<f64>,
) -> Option<(Vec<u8>, Vec<crate::osr::DeoptSite>)> {
    None
}

#[cfg(not(target_os = "windows"))]
pub fn compile_t4_unboxed_with_deopt_mapped(
    _code: &[crate::bytecode::Op],
    _const_f64: impl Fn(u16) -> Option<f64>,
    _bc_pc_map: Option<&[usize]>,
) -> Option<(Vec<u8>, Vec<crate::osr::DeoptSite>)> {
    None
}

/// T4 arith: get both operands (cache-aware — same-block operands skip the
/// reload+tag-check), compute into T2_XA, then box+store to the bank AND cache the
/// unboxed result for same-block consumers.
fn t4_emit_arith(
    em: &mut Emitter,
    op: T2Arith,
    dst: u16,
    lhs: u16,
    rhs: u16,
    cache: &mut T4ValueCache,
    deopt_patches: &mut Vec<(usize, crate::osr::DeoptReason)>,
    store_mode: T2StoreMode,
) {
    t4_emit_get_operand(em, cache, lhs, T2_XA, deopt_patches);
    t4_emit_get_operand(em, cache, rhs, T2_XB, deopt_patches);
    match op {
        T2Arith::Add => em.addsd_xmm_xmm(T2_XA, T2_XB),
        T2Arith::Sub => em.subsd_xmm_xmm(T2_XA, T2_XB),
        T2Arith::Mul => em.mulsd_xmm_xmm(T2_XA, T2_XB),
        T2Arith::Div => em.divsd_xmm_xmm(T2_XA, T2_XB),
    }
    t4_emit_box_store_cached(em, T2_XA, dst, cache, store_mode);
}

/// T4 compare: get both operands cache-aware (T2_XA/T2_XB), then the SAME
/// `t2_emit_cmp_store` boolean-result emission as T2-lite. The Bool result is not
/// an unboxed f64, so the caller `forget`s `dst` after.
fn t4_emit_cmp(
    em: &mut Emitter,
    cmp: T2Cmp,
    dst: u16,
    lhs: u16,
    rhs: u16,
    cache: &T4ValueCache,
    deopt_patches: &mut Vec<(usize, crate::osr::DeoptReason)>,
    store_mode: T2StoreMode,
) {
    t4_emit_get_operand(em, cache, lhs, T2_XA, deopt_patches);
    t4_emit_get_operand(em, cache, rhs, T2_XB, deopt_patches);
    t2_emit_cmp_store(em, cmp, dst, store_mode);
}

// ======================================================================
// T4 (Maglev-class) P2 — REPRESENTATION SELECTION + UNBOXED FLOAT64.
//
// The T2-lite numeric backend above re-does the FULL box/unbox round-trip on
// EVERY arithmetic op: it reloads both operands from the bank, runs the
// `is-number` tag-check guard, unboxes each to an XMM, computes, re-boxes
// (NaN-canonicalize), and stores back to the bank. For a value produced and
// immediately consumed within the SAME basic block (the dominant shape of an
// arithmetic-heavy function like jit.js's `f(x)` — a long chain of `t = a OP b`
// temporaries), the reload + tag-check + unbox of that operand is PURE WASTE: we
// just boxed it from an XMM one op ago and it is provably a number.
//
// T4 P2 adds REPRESENTATION SELECTION over the FLOAT64 representation: within a
// basic block it keeps each register's UNBOXED f64 value resident in an XMM and
// reads same-block operands straight from that XMM — eliminating the redundant
// reload + tag-check guard + unbox on every intermediate. This is exactly V8
// Maglev's `ValueRepresentation::kFloat64` selection + its CheckNumber/CheckSmi
// guard placement: the guard (a DeoptSite) fires ONCE where a TAGGED value first
// enters the unboxed domain (a fresh bank operand, e.g. the parameter `x` or a
// cross-block value), NOT on every use. Modeled on V8
// `src/maglev/maglev-graph-builder` representation selection + the
// `CheckedTaggedToFloat64`/`CheckSmi` conversion nodes.
//
// CORRECTNESS — the deopt identity-map invariant is PRESERVED VERBATIM. Every op
// STILL boxes its result and stores it to its bank slot (so the bank remains the
// exact pre-op VM register image at every op boundary), and every guard is the
// SAME DeoptSite the T2-lite path emits, with bc_pc == the op index — so a
// non-number operand deopts to the VM frame byte-identically. The ONLY thing T4
// changes is WHERE an operand's f64 value comes from when it is already proven
// unboxed in-block (an XMM read instead of a reload+guard). The XMM cache is a
// pure performance shadow of the bank; it is INVALIDATED at every basic-block
// boundary (any jump target, and after every branch/jump/ret), so a cross-block
// value always reloads-with-guard from the bank — cross-block correctness is
// trivially the same as T2-lite. The A/B oracle (ForcedTier::T4) proves
// byte-identity to the VM across the corpus, and the deopt-fuzzer force-deopts
// every op to prove the resumed VM result is identical.
//
// REPRESENTATION (P2 scope): FLOAT64 only — always correct for JS Numbers (every
// JS number is an f64; the `Value::Number(f64)` model has no separate Int32
// value, and `JsVal::try_from_value` always boxes a number in the DOUBLE lane).
// So T4 stores every numeric result in the double lane exactly as T2-lite does
// and the bank decode on deopt is byte-identical. (kInt32 with an overflow guard
// is the documented next step; it would only help if int values stayed in GPRs
// across ops, which requires the register-resident-roots pass — deferred. The
// f64 representation already removes the per-op tag-check, which is the in-block
// win, with zero new correctness surface.)
// ======================================================================

/// The XMM registers T4 uses as the per-block UNBOXED-f64 value cache. XMM0/XMM1
/// stay the arithmetic scratch (T2_XA/T2_XB) exactly as in the T2-lite path;
/// XMM2..=XMM5 are the cache pool. All four are caller-saved (volatile) under the
/// Win64 ABI, so T4 needs NO callee-saved xmm save area — the prolog/epilogue are
/// byte-identical to the T2-lite backend. A function that needs more than 4
/// simultaneously-live in-block values simply evicts the oldest cache entry (the
/// value is still in the bank, so the next read reloads-with-guard — always
/// correct, just one missed fast path).
const T4_CACHE_XMMS: [Xmm; 4] = [Xmm::Xmm2, Xmm::Xmm3, Xmm::Xmm4, Xmm::Xmm5];

/// The per-basic-block UNBOXED-f64 value cache: maps a bank slot to the XMM that
/// currently holds its unboxed f64 value (valid only within the current block).
/// `slot_of[k]` is the bank slot whose value lives in `T4_CACHE_XMMS[k]`, or
/// `None` if that cache register is free. `next_evict` is a round-robin victim
/// pointer for when all cache registers are occupied. The whole cache is
/// `invalidate`d at every block boundary so a cross-block read reloads-with-guard.
struct T4ValueCache {
    /// bank slot resident in each cache XMM (parallel to `T4_CACHE_XMMS`).
    slot_of: [Option<u16>; 4],
    /// Round-robin eviction pointer.
    next_evict: usize,
}

impl T4ValueCache {
    fn new() -> Self {
        T4ValueCache {
            slot_of: [None; 4],
            next_evict: 0,
        }
    }

    /// Drop ALL cached values — called at every basic-block boundary so a value
    /// produced in one block is never read as "unboxed in XMM" from another (the
    /// XMM is dead across the branch). After this, every operand read reloads from
    /// the bank with its tag-check guard, exactly as T2-lite does.
    fn invalidate(&mut self) {
        self.slot_of = [None; 4];
        self.next_evict = 0;
    }

    /// If `slot`'s unboxed f64 value is currently cached, return its XMM.
    fn get(&self, slot: u16) -> Option<Xmm> {
        for (k, s) in self.slot_of.iter().enumerate() {
            if *s == Some(slot) {
                return Some(T4_CACHE_XMMS[k]);
            }
        }
        None
    }

    /// Forget any cached XMM for `slot` (the slot is about to be overwritten by a
    /// non-cached store, e.g. a constant load or a Move from an unknown value —
    /// the old XMM no longer reflects the bank).
    fn forget(&mut self, slot: u16) {
        for s in self.slot_of.iter_mut() {
            if *s == Some(slot) {
                *s = None;
            }
        }
    }

    /// Reserve a cache XMM for `slot` (evicting round-robin if full). Returns the
    /// XMM the caller should write `slot`'s freshly-computed unboxed f64 into. The
    /// value MUST also be boxed+stored to the bank by the caller (identity-map
    /// invariant) — the cache is only a fast-read shadow.
    fn assign(&mut self, slot: u16) -> Xmm {
        // First clear any prior cache entry for this slot (re-defining it).
        self.forget(slot);
        // Prefer a free register.
        if let Some(k) = self.slot_of.iter().position(|s| s.is_none()) {
            self.slot_of[k] = Some(slot);
            return T4_CACHE_XMMS[k];
        }
        // Full → evict round-robin (the evicted value is still in the bank).
        let k = self.next_evict % T4_CACHE_XMMS.len();
        self.next_evict = self.next_evict.wrapping_add(1);
        self.slot_of[k] = Some(slot);
        T4_CACHE_XMMS[k]
    }
}

/// T4: get operand `slot`'s unboxed f64 value into `want` (a scratch XMM,
/// T2_XA/T2_XB). If the value is already cached in an XMM from earlier in this
/// block, copy it (no reload, NO tag-check guard) — the representation-selection
/// win. Otherwise fall back to the proven `t2_emit_load_num` (reload + tag-check
/// DeoptSite + unbox), exactly as T2-lite. Either way `want` ends holding the f64.
fn t4_emit_get_operand(
    em: &mut Emitter,
    cache: &T4ValueCache,
    slot: u16,
    want: Xmm,
    deopt_patches: &mut Vec<(usize, crate::osr::DeoptReason)>,
) {
    if let Some(src) = cache.get(slot) {
        // FAST PATH: the value is proven-unboxed in `src` (it was produced and
        // boxed/stored earlier in this block, and the bank still mirrors it). Read
        // it directly — no reload, no tag-check, no unbox. This is the per-op
        // tag-check elimination (the float-dense win).
        if src != want {
            em.movsd_xmm_xmm(want, src);
        }
    } else {
        // SLOW PATH (cross-block value, function arg, or evicted): reload from the
        // bank with the SAME is-number guard the T2-lite path emits — a DeoptSite
        // at this op's boundary, so a non-number operand deopts to the VM frame
        // byte-identically.
        t2_emit_load_num(em, slot, want, deopt_patches);
    }
}

/// T4: box the f64 result in `from` (canonicalize NaN), store it to `bank[dst]`
/// (the identity-map invariant — the bank stays a complete pre-op image), AND
/// record `dst` in the value cache with its OWN freshly-allocated cache XMM
/// (copying the f64 in so a same-block consumer reads it without a reload). The
/// boxing is identical to `t2_emit_box_store`; the only addition is the cache
/// bookkeeping + the copy into the cache register.
fn t4_emit_box_store_cached(
    em: &mut Emitter,
    from: Xmm,
    dst: u16,
    cache: &mut T4ValueCache,
    store_mode: T2StoreMode,
) {
    // Box + store to the bank exactly as T2-lite (deopt identity-map preserved).
    t2_emit_box_store(em, from, dst, store_mode);
    // Cache the unboxed f64 for same-block consumers: allocate a cache XMM for
    // `dst` and copy the result in. (The result is the NON-canonicalized `from`
    // value; for a NON-NaN result that equals the canonical stored value, and a
    // NaN result compares-unequal to everything so any later arithmetic on it is
    // NaN either way — observationally identical to reloading the canonical NaN.
    // To be exactly bit-faithful we copy `from`, whose only possible divergence
    // from the stored value is a NON-canonical NaN payload, which is unobservable
    // through f64 arithmetic/comparison. The conservative choice — and the one
    // that keeps the cache a faithful shadow — is to NOT cache a NaN result; but
    // detecting NaN here costs a branch, so instead we cache `from` and rely on
    // the invariant that a cached value is only ever CONSUMED by f64 arith/compare
    // ops, where a non-canonical-NaN payload is indistinguishable from the
    // canonical NaN. The deopt path NEVER reads the cache — it decodes the BANK,
    // which holds the canonical value — so deopt stays byte-identical.)
    let cx = cache.assign(dst);
    if cx != from {
        em.movsd_xmm_xmm(cx, from);
    }
}

/// Emit one inlined arithmetic op: load both operands (tag-checked), do the f64
/// op into `T2_XA`, box (canonicalize NaN) + store to `bank[dst]`.
fn t2_emit_arith(
    em: &mut Emitter,
    op: T2Arith,
    dst: u16,
    lhs: u16,
    rhs: u16,
    deopt_patches: &mut Vec<(usize, crate::osr::DeoptReason)>,
    store_mode: T2StoreMode,
) {
    t2_emit_load_num(em, lhs, T2_XA, deopt_patches);
    t2_emit_load_num(em, rhs, T2_XB, deopt_patches);
    match op {
        T2Arith::Add => em.addsd_xmm_xmm(T2_XA, T2_XB),
        T2Arith::Sub => em.subsd_xmm_xmm(T2_XA, T2_XB),
        T2Arith::Mul => em.mulsd_xmm_xmm(T2_XA, T2_XB),
        T2Arith::Div => em.divsd_xmm_xmm(T2_XA, T2_XB),
    }
    t2_emit_box_store(em, T2_XA, dst, store_mode);
}

/// Emit `JmpIfFalse cond, target`: jump to `target` (a bytecode index, patched
/// via `jump_patches`) iff `bank[cond]` is FALSY. Handles the value kinds
/// T2-lite produces: boolean singletons (false → jump, true → fall through) and
/// numbers (0, -0, NaN → jump). DEOPTs on any other boxed value (its truthiness
/// can't be evaluated without the VM's ToBoolean).
fn t2_emit_jmp_if_false(
    em: &mut Emitter,
    cond: u16,
    jump_patches: &mut Vec<(usize, usize)>,
    target: usize,
    deopt_patches: &mut Vec<(usize, crate::osr::DeoptReason)>,
) {
    // rax = bank[cond]
    em.mov_r64_mem(R64::Rax, T2_BANK, (cond as i32) * 8);
    // if rax == JV_FALSE → jump to target.
    em.mov_r64_imm64(R64::Rcx, JV_FALSE as i64);
    em.cmp_r64_r64(R64::Rax, R64::Rcx);
    let to_target_false = em.jcc_rel32_placeholder(Cc::Equal);
    jump_patches.push((to_target_false, target));
    // if rax == JV_TRUE → fall through (truthy).
    em.mov_r64_imm64(R64::Rcx, JV_TRUE as i64);
    em.cmp_r64_r64(R64::Rax, R64::Rcx);
    let to_fallthrough = em.jcc_rel32_placeholder(Cc::Equal);
    // not a boolean: must be a number to evaluate. rcx = rax & QNAN_MASK.
    em.mov_r64_r64(R64::Rcx, R64::Rax);
    em.mov_r64_imm64(R64::R10, JV_QNAN_MASK as i64);
    em.and_r64_r64(R64::Rcx, R64::R10);
    em.mov_r64_imm64(R64::R11, JV_QNAN_BITS as i64);
    em.cmp_r64_r64(R64::Rcx, R64::R11);
    // je DEOPT (boxed non-boolean ⇒ can't evaluate truthiness inline).
    let dp = em.jcc_rel32_placeholder(Cc::Equal);
    deopt_patches.push((dp, crate::osr::DeoptReason::NonNumber));
    // It's a double: jump to target if it is 0.0 / -0.0 / NaN (falsy numbers).
    em.movq_xmm_r64(T2_XA, R64::Rax);
    em.xorpd_xmm_xmm(T2_XB, T2_XB);
    em.ucomisd_xmm_xmm(T2_XA, T2_XB);
    // NaN → unordered → PF=1 → falsy → jump.
    let to_target_nan = em.jcc_rel32_placeholder(Cc::Parity);
    jump_patches.push((to_target_nan, target));
    // == 0.0 (covers -0.0) → ZF=1 → falsy → jump.
    let to_target_zero = em.jcc_rel32_placeholder(Cc::Equal);
    jump_patches.push((to_target_zero, target));
    // else truthy → fall through (the `to_fallthrough` label lands here).
    em.patch_rel32(to_fallthrough);
}

/// Emit the T2 Phase-1 INLINE GetProp fast path for `bank[dst] = bank[obj].<key>`
/// where `obj` is a pure function ARG (kept alive by the caller's `args` slice for
/// the whole call) and the result is consumed as an immediate. Sequence:
///   1. Load the receiver `JsVal` from `bank[obj]`.
///   2. Inline tag-check it is an OBJECT (`(bits & TOP16) == OBJECT_TOP16`); else
///      DEOPT (a non-object receiver — number/string/array/etc. — the VM handles).
///   3. Extract the 48-bit object pointer.
///   4. Read the `u32` shape-id HEADER inline at `[obj_ptr + shape_off]` (one
///      `mov r32` — NO `shapes` Mutex).
///   5. For each warmed `(shape, slot)`: `cmp header, shape; je hit_k`. No match
///      (incl. any Dict sentinel / different shape) → DEOPT.
///   6. On a hit, call the audited helper `rt_getprop_slot_immediate(obj_ptr,
///      slot)`; if it returns the DEOPT sentinel (`JV_HOLE` — non-immediate slot)
///      → DEOPT; else store the returned IMMEDIATE `JsVal` to `bank[dst]`.
///
/// REGISTER USE: RAX holds the object pointer across the shape guards + into the
/// call (arg1=RCX); R8 holds the header; RDX = slot (the call's arg2); R9/R10/R11
/// scratch. All volatile — nothing is held across an op boundary (the only call,
/// to the helper, is bracketed by its own shadow-space frame here). The helper
/// preserves RBX/RDI (the bank/out base), so those survive.
///
/// BANK SAFETY: only an IMMEDIATE (number/bool) `JsVal` is ever stored to the
/// bank (the helper returns DEOPT for everything else) — no heap `JsVal` enters
/// the bank in this phase.
fn t2_emit_getprop_immediate(
    em: &mut Emitter,
    dst: u16,
    obj: u16,
    site: &T2GetPropSite,
    shape_off: i32,
    helper_addr: usize,
    deopt_patches: &mut Vec<(usize, crate::osr::DeoptReason)>,
    store_mode: T2StoreMode,
) {
    // (1) rax = receiver JsVal from bank[obj].
    em.mov_r64_mem(R64::Rax, T2_BANK, (obj as i32) * 8);
    // (2) is-object: (rax & TOP16_MASK) == OBJECT_TOP16, else DEOPT.
    em.mov_r64_r64(R64::Rcx, R64::Rax);
    em.mov_r64_imm64(R64::R10, JV_TOP16_MASK as i64);
    em.and_r64_r64(R64::Rcx, R64::R10);
    em.mov_r64_imm64(R64::R11, JV_OBJECT_TOP16 as i64);
    em.cmp_r64_r64(R64::Rcx, R64::R11);
    let dp_not_obj = em.jcc_rel32_placeholder(Cc::NotEqual);
    deopt_patches.push((dp_not_obj, crate::osr::DeoptReason::NonObject));
    // (3) rax = obj_ptr (rax & PAYLOAD_MASK). Kept in RAX through the guards + call.
    em.mov_r64_imm64(R64::R10, JV_PAYLOAD_MASK as i64);
    em.and_r64_r64(R64::Rax, R64::R10);
    // (4) r8d = shape header (4-byte load, zero-extended). NO Mutex.
    em.mov_r32_mem(R64::R8, R64::Rax, shape_off);
    // (5) shape guards: cmp header, shape_k; je hit_k. Collect the hit branch
    // sites; each hit block sets RDX=slot then jumps to the shared call block.
    let mut hit_branches: Vec<(usize, u32)> = Vec::with_capacity(site.shapes_slots.len());
    for &(shape, slot) in &site.shapes_slots {
        // Exact u32 compare (mov imm64 + cmp r64,r64 — no sign-extension hazard).
        em.mov_r64_imm64(R64::R9, shape as u64 as i64);
        em.cmp_r64_r64(R64::R8, R64::R9);
        let je = em.jcc_rel32_placeholder(Cc::Equal);
        hit_branches.push((je, slot));
    }
    // No shape matched → DEOPT.
    let dp_miss = em.jmp_rel32_placeholder();
    deopt_patches.push((dp_miss, crate::osr::DeoptReason::ShapeMiss));
    // Hit blocks: set RDX = slot, then jump to the shared call block.
    let mut to_call: Vec<usize> = Vec::with_capacity(hit_branches.len());
    for (je_off, slot) in hit_branches {
        em.patch_rel32(je_off); // this hit lands here
        em.mov_r64_imm32(R64::Rdx, slot as i32); // arg2 = slot (small, < cap)
        let j = em.jmp_rel32_placeholder();
        to_call.push(j);
    }
    // (6) Shared call block. All hit branches land here.
    for j in to_call {
        em.patch_rel32(j);
    }
    // arg1 = obj_ptr (still in RAX). RDX already = slot.
    em.mov_r64_r64(R64::Rcx, R64::Rax);
    // Reserve shadow space + realign for the Win64 call. At a T2 op boundary
    // RSP ≡ 8 (mod 16) (entry ≡8, push rbx → ≡0, push rdi → ≡8). `sub 40`
    // (40 ≡ 8 mod 16) → RSP ≡ 0 at the `call`; the callee sees ≡8 after its
    // pushed return address. 32 of the 40 are the mandatory shadow space.
    em.sub_r64_imm32(R64::Rsp, 40);
    em.mov_r64_imm64(R64::R11, helper_addr as i64);
    em.call_r64(R64::R11);
    em.add_r64_imm32(R64::Rsp, 40);
    // rax = result JsVal bits (immediate) OR JV_HOLE (deopt sentinel).
    em.mov_r64_imm64(R64::Rcx, JV_HOLE as i64);
    em.cmp_r64_r64(R64::Rax, R64::Rcx);
    let dp_non_imm = em.jcc_rel32_placeholder(Cc::Equal);
    deopt_patches.push((dp_non_imm, crate::osr::DeoptReason::ShapeMiss));
    // Store the immediate (in RAX) to bank[dst]. In Heap mode this still goes
    // through the owning store so the OLD dst slot (possibly a heap value) is dec'd.
    t2_emit_bank_store(em, dst, store_mode);
}

/// T2 Phase 3 — emit the OWNING-store inline GetProp fast path. Identical receiver
/// type + shape guards as `t2_emit_getprop_immediate`, but instead of returning a
/// value the helper does the OWNING bank store itself (inc-new-before-dec-old) for
/// a heap result (Object/Array/String) OR an immediate, returning T2_RETURNED /
/// T2_DEOPT. So the emitted code does NOT store to `bank[dst]` afterward (the
/// helper already did, with the Rc accounting) — it only checks the status.
///
/// Helper ABI (4 args): `rt_getprop_slot_owning_store(obj_ptr=RCX, slot=RDX,
/// bank=R8, dst=R9) -> RAX status`. The bank base is `T2_BANK` (RBX), which the
/// helper preserves (callee-saved), so it survives the call.
///
/// BANK SAFETY: a heap `JsVal` may now enter the bank — but ONLY through the
/// helper's owning store (which `rc_inc`s it), and ONLY when the caller wired this
/// site as `heap_result` because the bank is GC-registered + owning for the run.
fn t2_emit_getprop_owning(
    em: &mut Emitter,
    dst: u16,
    obj: u16,
    site: &T2GetPropSite,
    shape_off: i32,
    heap_helper_addr: usize,
    deopt_patches: &mut Vec<(usize, crate::osr::DeoptReason)>,
) {
    // (1) rax = receiver JsVal from bank[obj].
    em.mov_r64_mem(R64::Rax, T2_BANK, (obj as i32) * 8);
    // (2) is-object guard, else DEOPT.
    em.mov_r64_r64(R64::Rcx, R64::Rax);
    em.mov_r64_imm64(R64::R10, JV_TOP16_MASK as i64);
    em.and_r64_r64(R64::Rcx, R64::R10);
    em.mov_r64_imm64(R64::R11, JV_OBJECT_TOP16 as i64);
    em.cmp_r64_r64(R64::Rcx, R64::R11);
    let dp_not_obj = em.jcc_rel32_placeholder(Cc::NotEqual);
    deopt_patches.push((dp_not_obj, crate::osr::DeoptReason::NonObject));
    // (3) rax = obj_ptr. Kept through the guards + into arg1.
    em.mov_r64_imm64(R64::R10, JV_PAYLOAD_MASK as i64);
    em.and_r64_r64(R64::Rax, R64::R10);
    // (4) r8d = shape header.
    em.mov_r32_mem(R64::R8, R64::Rax, shape_off);
    // (5) shape guards.
    let mut hit_branches: Vec<(usize, u32)> = Vec::with_capacity(site.shapes_slots.len());
    for &(shape, slot) in &site.shapes_slots {
        em.mov_r64_imm64(R64::R9, shape as u64 as i64);
        em.cmp_r64_r64(R64::R8, R64::R9);
        let je = em.jcc_rel32_placeholder(Cc::Equal);
        hit_branches.push((je, slot));
    }
    // No shape matched → DEOPT.
    let dp_miss = em.jmp_rel32_placeholder();
    deopt_patches.push((dp_miss, crate::osr::DeoptReason::ShapeMiss));
    // Hit blocks: set RDX = slot, jump to the shared call block.
    let mut to_call: Vec<usize> = Vec::with_capacity(hit_branches.len());
    for (je_off, slot) in hit_branches {
        em.patch_rel32(je_off);
        em.mov_r64_imm32(R64::Rdx, slot as i32); // arg2 = slot
        let j = em.jmp_rel32_placeholder();
        to_call.push(j);
    }
    // (6) Shared call block.
    for j in to_call {
        em.patch_rel32(j);
    }
    // Marshal the 4 args: RCX = obj_ptr (in RAX), RDX = slot (set), R8 = bank base
    // (T2_BANK/RBX), R9 = dst slot index. Set RCX LAST (RAX holds obj_ptr until
    // then; R8/R9 don't clobber RAX).
    em.mov_r64_r64(R64::R8, T2_BANK); // arg3 = bank base
    em.mov_r64_imm32(R64::R9, dst as i32); // arg4 = dst slot index
    em.mov_r64_r64(R64::Rcx, R64::Rax); // arg1 = obj_ptr
    // Shadow space + 16-align (same accounting as the immediate path).
    em.sub_r64_imm32(R64::Rsp, 40);
    em.mov_r64_imm64(R64::R11, heap_helper_addr as i64);
    em.call_r64(R64::R11);
    em.add_r64_imm32(R64::Rsp, 40);
    // rax = T2_RETURNED (helper already did the owning store) or T2_DEOPT. On
    // DEOPT, jump to the shared deopt pad (bank[dst] untouched by the helper, so
    // the VM re-run is identical).
    em.test_r64_r64(R64::Rax, R64::Rax); // T2_RETURNED == 0 → ZF=1
    let dp_deopt = em.jcc_rel32_placeholder(Cc::NotEqual);
    deopt_patches.push((dp_deopt, crate::osr::DeoptReason::ShapeMiss));
    // RETURNED: nothing else to do — the helper stored bank[dst] with ownership.
}

/// Emit the index-validation prologue for a GetIdx/SetIdx: load `bank[key]` and
/// leave a VALIDATED NON-NEGATIVE INTEGER index in `dst_reg`, or jump to a DEOPT
/// site if the key is not a non-negative integer (negative / fractional / NaN /
/// out-of-i32-range / non-number). Mirrors `t2_emit_load_num`'s number test but
/// then demands the value be a non-negative WHOLE number:
///   * DOUBLE lane: `cvttsd2si` to an i64, round-trip `cvtsi2sd` + `ucomisd` —
///     any inequality (fractional, NaN, ±inf, > i64) DEOPTs; then a `< 0` test.
///   * INT32 lane: sign-extend low 32 bits; a `< 0` test.
///   * any other boxed lane (undefined/null/bool/object/array/string) → DEOPT
///     (a non-integer key is a NAMED-property lookup on the VM, not this path).
/// Scratch: RAX, RCX, R10, R11, XMM0, XMM1 (all volatile; nothing held across an
/// op boundary). The validated index lands in `dst_reg` (must be a volatile reg
/// the caller then marshals into the helper's index arg).
fn t2_emit_load_index(
    em: &mut Emitter,
    slot: u16,
    dst_reg: R64,
    deopt_patches: &mut Vec<(usize, crate::osr::DeoptReason)>,
) {
    // rax = bank[slot]
    em.mov_r64_mem(R64::Rax, T2_BANK, (slot as i32) * 8);
    // Is it a plain double? (rax & QNAN_MASK) != QNAN_BITS ⇒ number lane.
    em.mov_r64_r64(R64::Rcx, R64::Rax);
    em.mov_r64_imm64(R64::R10, JV_QNAN_MASK as i64);
    em.and_r64_r64(R64::Rcx, R64::R10);
    em.mov_r64_imm64(R64::R11, JV_QNAN_BITS as i64);
    em.cmp_r64_r64(R64::Rcx, R64::R11);
    let to_double = em.jcc_rel32_placeholder(Cc::NotEqual);
    // BOXED: admit only the int32 lane. (rax & TOP16) == INT32_TOP16 ?
    em.mov_r64_r64(R64::Rcx, R64::Rax);
    em.mov_r64_imm64(R64::R10, JV_TOP16_MASK as i64);
    em.and_r64_r64(R64::Rcx, R64::R10);
    em.mov_r64_imm64(R64::R11, JV_INT32_TOP16 as i64);
    em.cmp_r64_r64(R64::Rcx, R64::R11);
    let dp_not_int = em.jcc_rel32_placeholder(Cc::NotEqual);
    deopt_patches.push((dp_not_int, crate::osr::DeoptReason::BadIndex));
    // int32 lane: sign-extend the low 32 bits into dst_reg.
    em.movsxd_r64_r32(dst_reg, R64::Rax);
    let to_neg_check = em.jmp_rel32_placeholder();
    // DOUBLE lane: truncate to an integer + prove it round-trips (no frac/NaN/inf).
    em.patch_rel32(to_double);
    em.movq_xmm_r64(T2_XA, R64::Rax); // xmm0 = the double
    em.cvttsd2si_r64_xmm(dst_reg, T2_XA); // dst_reg = trunc(double) as i64
    em.cvtsi2sd_xmm_r64(T2_XB, dst_reg); // xmm1 = (double)dst_reg
    em.ucomisd_xmm_xmm(T2_XA, T2_XB); // compare original vs round-tripped
    // PF=1 (unordered/NaN) → DEOPT.
    let dp_nan = em.jcc_rel32_placeholder(Cc::Parity);
    deopt_patches.push((dp_nan, crate::osr::DeoptReason::BadIndex));
    // ZF=0 (not equal → had a fractional part, or out-of-i64 indefinite) → DEOPT.
    let dp_frac = em.jcc_rel32_placeholder(Cc::NotEqual);
    deopt_patches.push((dp_frac, crate::osr::DeoptReason::BadIndex));
    // NON-NEGATIVE check (both lanes converge here): dst_reg >= 0, else DEOPT.
    // A negative index is a named-property lookup on the VM (yields undefined or a
    // method) — outside this fast path.
    em.patch_rel32(to_neg_check);
    em.cmp_r64_imm32(dst_reg, 0);
    let dp_neg = em.jcc_rel32_placeholder(Cc::Less);
    deopt_patches.push((dp_neg, crate::osr::DeoptReason::BadIndex));
    // dst_reg now holds a validated non-negative integer index.
}

/// T2 GetIdx — emit the inline COMPUTED-ARRAY-READ fast path for
/// `bank[dst] = bank[obj][bank[key]]`. Sequence:
///   1. Load receiver `bank[obj]`; is-ARRAY guard (`(bits & TOP16) == ARRAY_TOP16`),
///      else DEOPT (a non-array receiver — object/string-index/named — the VM
///      handles, possibly via a different shape of indexing).
///   2. Extract the 48-bit array pointer.
///   3. Validate + extract a non-negative integer index from `bank[key]` (negative
///      / fractional / non-number → DEOPT).
///   4. Call `rt_getidx_owning_store(arr_ptr=RCX, idx=RDX, bank=R8, dst=R9)`: it
///      bounds-checks, reads the element, OWNING-stores it (immediate or admitted
///      heap lane) into `bank[dst]`, and returns RETURNED; OR (hole / accessor /
///      non-admitted element) returns DEOPT. OOB → undefined (RETURNED, not deopt).
///   5. RETURNED → fall through; DEOPT → the per-guard resume stub.
///
/// REGISTER USE: RAX holds the array pointer across the index extraction (which
/// uses RCX/R10/R11/XMM0/XMM1) and into the call (arg1=RCX, set last). The
/// validated index is built in R9 first (a volatile reg untouched by the index
/// extraction's scratch set), then moved to RDX (arg2) — keeping RAX (the array
/// ptr) live until the final RCX marshal. The helper does the owning store, so the
/// emitted code does NOT store to `bank[dst]` afterward — it only checks status.
///
/// BANK SAFETY: a heap `JsVal` may enter the bank, but ONLY through the helper's
/// owning store (which `rc_inc`s it); GetIdx is wired only in heap mode (the
/// owning + GC-rooted bank), the same prerequisite as the heap GetProp path.
fn t2_emit_getidx(
    em: &mut Emitter,
    dst: u16,
    obj: u16,
    key: u16,
    helper_addr: usize,
    deopt_patches: &mut Vec<(usize, crate::osr::DeoptReason)>,
) {
    // (1) rax = receiver JsVal from bank[obj].
    em.mov_r64_mem(R64::Rax, T2_BANK, (obj as i32) * 8);
    // (2) is-ARRAY guard: (rax & TOP16_MASK) == ARRAY_TOP16, else DEOPT.
    em.mov_r64_r64(R64::Rcx, R64::Rax);
    em.mov_r64_imm64(R64::R10, JV_TOP16_MASK as i64);
    em.and_r64_r64(R64::Rcx, R64::R10);
    em.mov_r64_imm64(R64::R11, JV_ARRAY_TOP16 as i64);
    em.cmp_r64_r64(R64::Rcx, R64::R11);
    let dp_not_arr = em.jcc_rel32_placeholder(Cc::NotEqual);
    deopt_patches.push((dp_not_arr, crate::osr::DeoptReason::NonArray));
    // (3) rax = arr_ptr (rax & PAYLOAD_MASK). The index extraction below CLOBBERS
    // RAX (it loads bank[key] into RAX), so park arr_ptr in a register the
    // extraction avoids. The extraction's scratch set is {RAX,RCX,R10,R11,XMM0,
    // XMM1}; the free volatiles are R8/R9. Park arr_ptr in R9, and have the
    // extraction write the validated index into RDX (also outside its scratch set).
    em.mov_r64_imm64(R64::R10, JV_PAYLOAD_MASK as i64);
    em.and_r64_r64(R64::Rax, R64::R10);
    em.mov_r64_r64(R64::R9, R64::Rax); // R9 = arr_ptr (survives extraction)
    // (4) validate + extract the index into RDX (the helper's arg2 slot directly).
    t2_emit_load_index(em, key, R64::Rdx, deopt_patches);
    // Marshal the 4 args. RDX (arg2 = idx) already set. R8 = bank base, R9 currently
    // holds arr_ptr → move to RCX (arg1) BEFORE overwriting R8. Order matters:
    //   RCX (arg1) = arr_ptr (from R9)   — do first (frees R9 read)
    //   R8  (arg3) = bank base
    //   R9  (arg4) = dst slot index
    em.mov_r64_r64(R64::Rcx, R64::R9); // arg1 = arr_ptr
    em.mov_r64_r64(R64::R8, T2_BANK); // arg3 = bank base
    em.mov_r64_imm32(R64::R9, dst as i32); // arg4 = dst slot index
    // Shadow space + 16-align (same accounting as the GetProp helper path).
    em.sub_r64_imm32(R64::Rsp, 40);
    em.mov_r64_imm64(R64::R11, helper_addr as i64);
    em.call_r64(R64::R11);
    em.add_r64_imm32(R64::Rsp, 40);
    // rax = T2_RETURNED (helper did the owning store, incl. OOB→undefined) or
    // T2_DEOPT (hole / accessor / non-admitted element). On DEOPT, resume the VM at
    // this op's bc_pc (bank[dst] untouched by the helper → identical VM image).
    em.test_r64_r64(R64::Rax, R64::Rax); // T2_RETURNED == 0 → ZF=1
    let dp_deopt = em.jcc_rel32_placeholder(Cc::NotEqual);
    deopt_patches.push((dp_deopt, crate::osr::DeoptReason::HoleOrSpecial));
    // RETURNED: nothing else to do — the helper stored bank[dst] with ownership.
}

/// T2 SetIdx — emit the inline COMPUTED-ARRAY-WRITE fast path for
/// `bank[obj][bank[key]] = bank[src]`. Sequence:
///   1. is-ARRAY guard on `bank[obj]`, else DEOPT.
///   2. extract the array pointer.
///   3. validate + extract a non-negative integer index from `bank[key]`.
///   4. load the value `bank[src]` (a raw `JsVal`, the helper's arg3).
///   5. call `rt_setidx_owning_store(arr_ptr=RCX, idx=RDX, val_bits=R8)`: in-bounds
///      → owning element replace, RETURNED; OOB (extend) → DEOPT (the VM resizes).
///   6. RETURNED → fall through; DEOPT → the per-guard resume stub.
///
/// SIDE EFFECT: the in-bounds write COMMITS in the helper. Under P5 a later guard
/// resumes the VM AFTER this op's bc_pc, so the write is never duplicated.
///
/// REGISTER USE: same parking discipline as GetIdx — arr_ptr parked in R9 across
/// the index extraction (scratch {RAX,RCX,R10,R11,XMM0,XMM1}); the validated index
/// in RDX. The value (`bank[src]`) is loaded into R8 (arg3) AFTER the extraction
/// (extraction never writes R8), then arr_ptr moved R9→RCX (arg1) last.
fn t2_emit_setidx(
    em: &mut Emitter,
    obj: u16,
    key: u16,
    src: u16,
    helper_addr: usize,
    deopt_patches: &mut Vec<(usize, crate::osr::DeoptReason)>,
) {
    // (1) rax = receiver JsVal; is-ARRAY guard, else DEOPT.
    em.mov_r64_mem(R64::Rax, T2_BANK, (obj as i32) * 8);
    em.mov_r64_r64(R64::Rcx, R64::Rax);
    em.mov_r64_imm64(R64::R10, JV_TOP16_MASK as i64);
    em.and_r64_r64(R64::Rcx, R64::R10);
    em.mov_r64_imm64(R64::R11, JV_ARRAY_TOP16 as i64);
    em.cmp_r64_r64(R64::Rcx, R64::R11);
    let dp_not_arr = em.jcc_rel32_placeholder(Cc::NotEqual);
    deopt_patches.push((dp_not_arr, crate::osr::DeoptReason::NonArray));
    // (2) rax = arr_ptr; park in R9 (survives the index extraction).
    em.mov_r64_imm64(R64::R10, JV_PAYLOAD_MASK as i64);
    em.and_r64_r64(R64::Rax, R64::R10);
    em.mov_r64_r64(R64::R9, R64::Rax);
    // (3) validate + extract the index into RDX (arg2).
    t2_emit_load_index(em, key, R64::Rdx, deopt_patches);
    // (4) arg3 = the value JsVal from bank[src] (raw bits; the helper takes its own
    // +1 via to_value). Loaded AFTER extraction (which never writes R8).
    em.mov_r64_mem(R64::R8, T2_BANK, (src as i32) * 8);
    // (5) arg1 = arr_ptr (R9 → RCX). RDX (idx) + R8 (val) already set.
    em.mov_r64_r64(R64::Rcx, R64::R9);
    em.sub_r64_imm32(R64::Rsp, 40);
    em.mov_r64_imm64(R64::R11, helper_addr as i64);
    em.call_r64(R64::R11);
    em.add_r64_imm32(R64::Rsp, 40);
    // rax = T2_RETURNED (in-bounds owning replace done) or T2_DEOPT (OOB extend →
    // structural change). On DEOPT, resume the VM at this op's bc_pc. CRITICAL: a
    // RETURNED here means the write COMMITTED; the deopt path only triggers when the
    // write did NOT happen (OOB), so the VM resume re-performs the (un-done) write.
    em.test_r64_r64(R64::Rax, R64::Rax); // T2_RETURNED == 0 → ZF=1
    let dp_deopt = em.jcc_rel32_placeholder(Cc::NotEqual);
    deopt_patches.push((dp_deopt, crate::osr::DeoptReason::HoleOrSpecial));
    // RETURNED: the helper committed the element replace; nothing else to do.
}

// ======================================================================
// T2 Phase 4 — CALL INLINING codegen.
//
// All three emit helpers (load-global / call-value / call-fn) share the same
// shape: marshal a small, fixed set of args (the CTX pointer + the bank base +
// a packed descriptor of slot indices), `call` a re-entry helper that does ALL
// the heavy lifting in Rust (Value marshaling, VM dispatch, the owning store of
// the result into `bank[dst]`), then branch on the returned status:
//   * RETURNED → fall through (the helper already stored bank[dst]);
//   * DEOPT    → the shared deopt pad (bank untouched; VM re-runs identically) —
//                only LoadGlobal/Call can DEOPT *before* any side effect (e.g. a
//                non-callable callee the helper declines), never after;
//   * THREW    → the THREW pad (error stashed in ctx; runner propagates, no re-run);
//   * DEADLINE → the DEADLINE pad (uncatchable).
//
// ALIASING DISCIPLINE (the load-bearing P4 contract): across the `call` we hold
// NOTHING except the callee-saved BANK (RBX), CTX (RSI), OUT (RDI) pointers and
// RSP. A re-entrant call can `gc_collect` (marking the registered bank) and can
// reallocate Value Vecs, but the bank buffer itself is fixed-size + GC-rooted, so
// its base stays valid; every bank SLOT is re-read fresh after the call (the next
// op loads from `[BANK + slot*8]`). The helper performs the owning store of the
// result, so no heap JsVal is ever held in a host register across the boundary.
//
// PACKED-DESCRIPTOR layout (one u64, little-endian 4×u16 lanes):
//   bits  0..16 : lane A (callee slot  | fn_idx)
//   bits 16..32 : lane B (this slot    | unused=0xFFFF)
//   bits 32..48 : lane C (first_arg slot)
//   bits 48..64 : lane D (argc)
// `0xFFFF` in the `this` lane = NO_THIS (this = undefined), matching `Op::CallValue`.
// ======================================================================

/// Win64 shadow-space + alignment for a re-entry `call`. At a T2 op boundary RSP ≡
/// 8 (mod 16) (4 prolog pushes; see the prolog note). `sub 40` (≡8) → RSP ≡ 0 at
/// the `call`; the callee sees ≡8 after its pushed return address. 32 of the 40
/// are the mandatory shadow space.
const T2_CALL_FRAME: i32 = 40;

/// Emit the status branch after a re-entry helper returns its tag in RAX:
///   0 RETURNED → fall through;  1 DEOPT → deopt pad;  2 THREW → threw pad;
///   3 DEADLINE → deadline pad.
fn t2_emit_status_branch(
    em: &mut Emitter,
    deopt_patches: &mut Vec<(usize, crate::osr::DeoptReason)>,
    threw_patches: &mut Vec<usize>,
    deadline_patches: &mut Vec<usize>,
) {
    // RAX == T2_RETURNED(0) → fall through (ZF set by test).
    em.test_r64_r64(R64::Rax, R64::Rax);
    let to_cont = em.jcc_rel32_placeholder(Cc::Equal);
    // RAX == T2_DEOPT(1) → a per-guard RESUME at the call op's bc_pc. A
    // call/loadglobal DEOPT is a PRE-side-effect decline (the helper returns
    // T2_DEOPT only for a non-callable callee, BEFORE running it — bank untouched),
    // so RESUMING the VM at the call op re-executes the call and produces the
    // identical error/behaviour. Using RESUME (not a whole-fn Tier-A re-run) keeps
    // it sound even when the call sits in a LOOP after an earlier committed call.
    em.cmp_r64_imm32(R64::Rax, T2_DEOPT as i32);
    let dp = em.jcc_rel32_placeholder(Cc::Equal);
    deopt_patches.push((dp, crate::osr::DeoptReason::CallDecline));
    // RAX == T2_THREW(2) → threw pad.
    em.cmp_r64_imm32(R64::Rax, T2_THREW as i32);
    let tp = em.jcc_rel32_placeholder(Cc::Equal);
    threw_patches.push(tp);
    // else (T2_DEADLINE(3) or any other) → deadline pad (uncatchable).
    let dlp = em.jmp_rel32_placeholder();
    deadline_patches.push(dlp);
    // continue label:
    em.patch_rel32(to_cont);
}

/// P4 — emit `bank[dst] = globals[consts[name_k]]` via the re-entry helper
/// `rt_load_global(ctx=RCX, packed=RDX, bank=R8, dst=R9) -> RAX status`, where
/// `packed = name_k | (checked << 16)`. The helper owning-stores the loaded value
/// into `bank[dst]`. A plain LoadGlobal of a missing name yields undefined
/// (RETURNED); a LoadGlobalChecked of an undeclared name THREWs a catchable
/// ReferenceError. (LoadGlobal never DEOPTs — it has no value-shape guard.)
fn t2_emit_load_global(
    em: &mut Emitter,
    dst: u16,
    name_k: u16,
    checked: u64,
    ccfg: &T2CallConfig,
    threw_patches: &mut Vec<usize>,
    deadline_patches: &mut Vec<usize>,
) {
    let packed: u64 = (name_k as u64) | (checked << 16);
    em.mov_r64_r64(R64::Rcx, T2_CTX); // arg1 = ctx
    em.mov_r64_imm64(R64::Rdx, packed as i64); // arg2 = name_k | checked<<16
    em.mov_r64_r64(R64::R8, T2_BANK); // arg3 = bank base
    em.mov_r64_imm32(R64::R9, dst as i32); // arg4 = dst slot
    em.sub_r64_imm32(R64::Rsp, T2_CALL_FRAME);
    em.mov_r64_imm64(R64::R11, ccfg.load_global_helper_addr as i64);
    em.call_r64(R64::R11);
    em.add_r64_imm32(R64::Rsp, T2_CALL_FRAME);
    // LoadGlobal can only RETURNED / THREW / DEADLINE (never DEOPT). Route a
    // would-be DEOPT to the deadline pad anyway for safety; it never fires.
    let mut dummy_deopt: Vec<(usize, crate::osr::DeoptReason)> = Vec::new();
    t2_emit_status_branch(em, &mut dummy_deopt, threw_patches, deadline_patches);
    // A LoadGlobal DEOPT would be a bug, but if the helper ever returned 1 we'd
    // jump to a never-patched label — so re-route the dummy to the deadline pad
    // (an uncatchable, loud failure) rather than leave it dangling.
    for (d, _reason) in dummy_deopt {
        deadline_patches.push(d);
    }
}

/// P4 — emit `bank[dst] = bank[callee](bank[this], bank[first_arg..+argc])` via
/// the re-entry helper `rt_call_value(ctx=RCX, bank=RDX, packed=R8, dst=R9) ->
/// RAX status`. `packed` carries the 4 slot lanes. The helper reconstructs the
/// `Value`s from the bank (each `to_value` = +1 owned in a temp Vec), dispatches
/// EXACTLY like `Op::CallValue`, and owning-stores the result into `bank[dst]`.
#[allow(clippy::too_many_arguments)]
fn t2_emit_call_value(
    em: &mut Emitter,
    dst: u16,
    callee: u16,
    this_reg: u16,
    first_arg: u16,
    n_args: u8,
    ccfg: &T2CallConfig,
    deopt_patches: &mut Vec<(usize, crate::osr::DeoptReason)>,
    threw_patches: &mut Vec<usize>,
    deadline_patches: &mut Vec<usize>,
) {
    let packed: u64 = (callee as u64)
        | ((this_reg as u64) << 16)
        | ((first_arg as u64) << 32)
        | ((n_args as u64) << 48);
    em.mov_r64_r64(R64::Rcx, T2_CTX); // arg1 = ctx
    em.mov_r64_r64(R64::Rdx, T2_BANK); // arg2 = bank base
    em.mov_r64_imm64(R64::R8, packed as i64); // arg3 = packed lanes
    em.mov_r64_imm32(R64::R9, dst as i32); // arg4 = dst slot
    em.sub_r64_imm32(R64::Rsp, T2_CALL_FRAME);
    em.mov_r64_imm64(R64::R11, ccfg.call_helper_addr as i64);
    em.call_r64(R64::R11);
    em.add_r64_imm32(R64::Rsp, T2_CALL_FRAME);
    t2_emit_status_branch(em, deopt_patches, threw_patches, deadline_patches);
}

/// P4 — emit `bank[dst] = module.fns[fn_idx](bank[first_arg..+argc])` (this =
/// undefined) via the re-entry helper `rt_call_fn(ctx=RCX, bank=RDX, packed=R8,
/// dst=R9) -> RAX status`, where `packed = fn_idx | (first_arg<<32) | (argc<<48)`.
#[allow(clippy::too_many_arguments)]
fn t2_emit_call_fn(
    em: &mut Emitter,
    dst: u16,
    fn_idx: u16,
    first_arg: u16,
    n_args: u8,
    ccfg: &T2CallConfig,
    deopt_patches: &mut Vec<(usize, crate::osr::DeoptReason)>,
    threw_patches: &mut Vec<usize>,
    deadline_patches: &mut Vec<usize>,
) {
    let packed: u64 =
        (fn_idx as u64) | ((first_arg as u64) << 32) | ((n_args as u64) << 48);
    em.mov_r64_r64(R64::Rcx, T2_CTX); // arg1 = ctx
    em.mov_r64_r64(R64::Rdx, T2_BANK); // arg2 = bank base
    em.mov_r64_imm64(R64::R8, packed as i64); // arg3 = packed lanes
    em.mov_r64_imm32(R64::R9, dst as i32); // arg4 = dst slot
    em.sub_r64_imm32(R64::Rsp, T2_CALL_FRAME);
    em.mov_r64_imm64(R64::R11, ccfg.call_fn_helper_addr() as i64);
    em.call_r64(R64::R11);
    em.add_r64_imm32(R64::Rsp, T2_CALL_FRAME);
    t2_emit_status_branch(em, deopt_patches, threw_patches, deadline_patches);
}

// ======================================================================
// M4.2a — T1 BASELINE JIT (dispatch-elimination tier).
//
// Unlike `compile_bytecode_f64` (which re-implements arithmetic in xmm regs and
// is value-type-limited to f64), the T1 baseline JIT emits a CONTROL-FLOW
// SKELETON only: one `call <thunk>` per bytecode op, where the thunk runs the
// op via the VM's OWN shared `op_xxx` bodies. So T1 is semantically identical to
// the VM by construction (single source of truth) and handles any Value type the
// op helpers do — the win is eliminating the fetch/decode/match dispatch
// overhead, not re-deriving the math. Honest scope: a modest, provably-correct
// foundation, OFF by default, declining to the VM on anything unsupported.
// ======================================================================

/// Compile a bytecode function to a T1 baseline-JIT native function. Returns
/// `None` (decline → caller runs the VM) if ANY op is outside the supported
/// subset (caller passes `supported`, the VM's `t1_supported_op` predicate).
///
/// The emitted function has ABI `extern "system" fn(state_ptr: *mut VmState) ->
/// u64` (Win64: RCX = state, RAX = a `T1_*` tag). It keeps the state pointer in
/// RBX (callee-saved) and, per op, calls `thunk_addr` with (RCX=state, RDX=ip),
/// then branches on the returned tag:
///   * `CONTINUE(0)`  → fall through to the next op.
///   * `JUMPED(1)`    → native jump to the emitted offset of the op's
///                      compile-time-constant bytecode target.
///   * else (RETURNED/THREW/DEADLINE) → jump to the epilogue, which returns the
///     tag in RAX (the thunk already stashed the payload in the state out-slot).
///
/// CRITICAL aliasing: the thunk may re-enter the VM (a future `call` op), which
/// can reallocate the register Vec. We hold NOTHING in registers across a
/// `call` except the state POINTER (RBX) and RSP — both stable. Every register/
/// state access happens inside the thunk through a freshly-reconstructed
/// `&mut VmState`, so a reallocated Vec is always re-read via `state.regs`.
pub fn compile_baseline_t1(
    code: &[crate::bytecode::Op],
    supported: impl Fn(&crate::bytecode::Op) -> bool,
    thunk_addr: usize,
    t1_continue: u64,
    t1_jumped: u64,
) -> Option<Vec<u8>> {
    use crate::bytecode::Op;
    if code.is_empty() {
        return None;
    }
    // Decline if any op is unsupported.
    for op in code {
        if !supported(op) {
            return None;
        }
    }
    let n = code.len();
    let mut em = Emitter::new();
    let mut offsets = vec![0usize; n]; // bytecode index → machine-code offset
    // (displacement byte offset, target bytecode index) — patched to offsets[t].
    let mut jump_patches: Vec<(usize, usize)> = Vec::new();
    // displacement byte offsets that must target the epilogue.
    let mut epilogue_patches: Vec<usize> = Vec::new();

    // Prologue: save RBX (callee-saved; will hold the state ptr), reserve 32B
    // shadow space (Win64 requires it for every `call`), keeping 16B alignment.
    em.push_r64(R64::Rbx);
    em.sub_r64_imm32(R64::Rsp, 32);
    // RCX = arg0 (state ptr) → RBX.
    em.mov_r64_r64(R64::Rbx, R64::Rcx);

    for (i, op) in code.iter().enumerate() {
        offsets[i] = em.code.len();
        // Set up the thunk call: RCX = state ptr, RDX = bytecode ip (= i).
        em.mov_r64_r64(R64::Rcx, R64::Rbx);
        em.mov_r64_imm32(R64::Rdx, i as i32);
        em.mov_r64_imm64(R64::Rax, thunk_addr as i64);
        em.call_r64(R64::Rax); // tag → RAX
        match op {
            // Unconditional jump: tag is JUMPED (or DEADLINE if the watchdog
            // fired). JUMPED → native jump to target; else → epilogue.
            Op::Jmp { target } => {
                em.cmp_r64_imm32(R64::Rax, t1_jumped as i32);
                let o = em.jcc_rel32_placeholder(Cc::Equal);
                jump_patches.push((o, *target as usize));
                let e = em.jmp_rel32_placeholder();
                epilogue_patches.push(e);
            }
            // Conditional jump: JUMPED → target; CONTINUE → fall through; else
            // (DEADLINE) → epilogue.
            Op::JmpIfFalse { target, .. } => {
                em.cmp_r64_imm32(R64::Rax, t1_jumped as i32);
                let o = em.jcc_rel32_placeholder(Cc::Equal);
                jump_patches.push((o, *target as usize));
                // not JUMPED: CONTINUE(0) falls through, anything else → epilogue.
                em.cmp_r64_imm32(R64::Rax, t1_continue as i32);
                let e = em.jcc_rel32_placeholder(Cc::NotEqual);
                epilogue_patches.push(e);
            }
            // Return: always exits to the epilogue (tag = RETURNED or DEADLINE,
            // already in RAX). No need to test.
            Op::Ret { .. } => {
                let e = em.jmp_rel32_placeholder();
                epilogue_patches.push(e);
            }
            // Every other supported op: CONTINUE(0) → fall through; any non-zero
            // tag (THREW/DEADLINE) → epilogue.
            _ => {
                em.test_r64_r64(R64::Rax, R64::Rax);
                let e = em.jcc_rel32_placeholder(Cc::NotEqual);
                epilogue_patches.push(e);
            }
        }
    }

    // Epilogue: tear down the frame and return RAX (the tag) to the caller.
    let epilogue = em.code.len();
    em.add_r64_imm32(R64::Rsp, 32);
    em.pop_r64(R64::Rbx);
    em.ret();

    // Resolve patches. A jump target must be a valid bytecode index.
    for (_, t) in &jump_patches {
        if *t >= n {
            return None;
        }
    }
    for (disp_off, t) in jump_patches {
        em.patch_rel32_to(disp_off, offsets[t]);
    }
    for disp_off in epilogue_patches {
        em.patch_rel32_to(disp_off, epilogue);
    }
    Some(em.code)
}

/// Execution counts per function. Used to pick JIT candidates.
#[derive(Debug, Default)]
pub struct Profiler {
    counts: HashMap<u64, u32>,
}

impl Profiler {
    pub fn new() -> Self {
        Self::default()
    }

    /// Bump the execution count for `fn_id`. Returns the new count.
    pub fn record(&mut self, fn_id: u64) -> u32 {
        let v = self.counts.entry(fn_id).or_insert(0);
        *v += 1;
        *v
    }

    /// True if the function has been called enough times that
    /// compiling it would pay back.
    pub fn is_hot(&self, fn_id: u64, threshold: u32) -> bool {
        self.counts.get(&fn_id).copied().unwrap_or(0) >= threshold
    }
}

/// Linear-scan register allocator over a fixed pool of x86_64 GPRs.
///
/// V1 uses 4 host registers (Rbx, R12, R13, R14 — callee-saved so we
/// don't have to save/restore them in the prolog). When the working
/// set exceeds 4, we fall back to spilling onto the stack frame; the
/// emitter generates the corresponding `mov [rbp-N], reg` sequences.
pub struct Allocator {
    free: Vec<R64>,
    /// virtual reg → host reg.
    binding: HashMap<u16, R64>,
}

impl Allocator {
    pub fn new() -> Self {
        Self {
            free: vec![R64::Rbx, R64::R12, R64::R13, R64::R14],
            binding: HashMap::new(),
        }
    }

    /// Get the host register currently bound to `vreg`, or assign one
    /// from the free pool. Returns None if all registers are in use.
    pub fn host_of(&mut self, vreg: u16) -> Option<R64> {
        if let Some(r) = self.binding.get(&vreg) {
            return Some(*r);
        }
        let r = self.free.pop()?;
        self.binding.insert(vreg, r);
        Some(r)
    }

    pub fn release(&mut self, vreg: u16) {
        if let Some(r) = self.binding.remove(&vreg) {
            self.free.push(r);
        }
    }
}

impl Default for Allocator {
    fn default() -> Self {
        Self::new()
    }
}

/// Errors the JIT can hit. All are recoverable — caller falls back
/// to the interpreter.
#[derive(Debug, Clone)]
pub enum JitError {
    OutOfRegisters,
    UnsupportedOp,
    EmptyFunction,
}

/// Compile a JIT op sequence into native x86_64. Returns the byte
/// buffer (caller maps it executable via VirtualAlloc + jumps to it).
pub fn compile(ops: &[JitOp]) -> Result<Vec<u8>, JitError> {
    if ops.is_empty() {
        return Err(JitError::EmptyFunction);
    }
    let mut em = Emitter::new();
    // Prolog: save callee-saved regs we'll touch. Reuse them for our
    // virtual registers via the allocator below.
    em.push_r64(R64::Rbp);
    em.mov_r64_r64(R64::Rbp, R64::Rsp);
    em.push_r64(R64::Rbx);
    em.push_r64(R64::R12);
    em.push_r64(R64::R13);
    em.push_r64(R64::R14);

    let mut alloc = Allocator::new();
    let mut return_reg: Option<R64> = None;

    for op in ops {
        match *op {
            JitOp::ConstInt { dst, value } => {
                let r = alloc.host_of(dst).ok_or(JitError::OutOfRegisters)?;
                em.mov_r64_imm32(r, value);
            }
            JitOp::Mov { dst, src } => {
                let src_r = alloc.host_of(src).ok_or(JitError::OutOfRegisters)?;
                let dst_r = alloc.host_of(dst).ok_or(JitError::OutOfRegisters)?;
                em.mov_r64_r64(dst_r, src_r);
            }
            JitOp::Add { dst, a, b } => {
                let a_r = alloc.host_of(a).ok_or(JitError::OutOfRegisters)?;
                let b_r = alloc.host_of(b).ok_or(JitError::OutOfRegisters)?;
                let dst_r = alloc.host_of(dst).ok_or(JitError::OutOfRegisters)?;
                if dst_r != a_r {
                    em.mov_r64_r64(dst_r, a_r);
                }
                em.add_r64_r64(dst_r, b_r);
            }
            JitOp::Sub { dst, a, b } => {
                let a_r = alloc.host_of(a).ok_or(JitError::OutOfRegisters)?;
                let b_r = alloc.host_of(b).ok_or(JitError::OutOfRegisters)?;
                let dst_r = alloc.host_of(dst).ok_or(JitError::OutOfRegisters)?;
                if dst_r != a_r {
                    em.mov_r64_r64(dst_r, a_r);
                }
                em.sub_r64_r64(dst_r, b_r);
            }
            JitOp::Mul { dst, a, b } => {
                let a_r = alloc.host_of(a).ok_or(JitError::OutOfRegisters)?;
                let b_r = alloc.host_of(b).ok_or(JitError::OutOfRegisters)?;
                let dst_r = alloc.host_of(dst).ok_or(JitError::OutOfRegisters)?;
                if dst_r != a_r {
                    em.mov_r64_r64(dst_r, a_r);
                }
                em.imul_r64_r64(dst_r, b_r);
            }
            JitOp::Return { reg } => {
                let r = alloc.host_of(reg).ok_or(JitError::OutOfRegisters)?;
                if r != R64::Rax {
                    em.mov_r64_r64(R64::Rax, r);
                }
                return_reg = Some(R64::Rax);
                break;
            }
        }
    }
    // If the trace didn't end in an explicit return, fall through to
    // an implicit `return 0` so the function is callable.
    if return_reg.is_none() {
        em.xor_r64_r64(R64::Rax, R64::Rax);
    }
    // Epilog: restore callee-saved regs and ret.
    em.pop_r64(R64::R14);
    em.pop_r64(R64::R13);
    em.pop_r64(R64::R12);
    em.pop_r64(R64::Rbx);
    em.pop_r64(R64::Rbp);
    em.ret();
    Ok(em.code)
}

// ----------------------------------------------------------------------
// Executable buffer + fn-pointer dispatch
// ----------------------------------------------------------------------

/// One compiled JIT function. Owns the executable code page; the
/// function's signature is `extern "system" fn() -> u64`. Drop frees
/// the page via VirtualFree.
pub struct JitFunction {
    base: *mut core::ffi::c_void,
    size: usize,
    /// True when `base` points into the process-wide code cage (jit_cage). Cage
    /// pages are bump-allocated and live for the process, so Drop must NOT
    /// `VirtualFree` them — only private per-page installs are freed on drop.
    cage_owned: bool,
    /// T2 Phase 5: per-guard deopt sites, indexed by the `deopt_id` a guard's
    /// stub writes to `*out` on a `T2_DEOPT_RESUME`. Empty for non-T2 functions
    /// (T1 / f64) and for T2 functions compiled without resume support (Tier-A
    /// whole-function re-run). The runner uses `deopt_site(id)` to find the
    /// `bc_pc` + reason for a resume.
    deopt_sites: Vec<crate::osr::DeoptSite>,
    /// B3 GC-rooting groundwork: the safepoint stack map (native-offset → live
    /// heap-root bank-slot set) emitted by the T3 backend. Empty for T1/T2/f64
    /// functions (they store-after-every-op, so the bank is always a complete
    /// root image — no register-resident roots to map). The GC consults this to
    /// root register/spill-resident pointers across a collection (B3).
    safepoints: crate::osr::SafepointMap,
    /// B2 T3: the OPTIMIZED bytecode module this native code was compiled from.
    /// `Some` only for T3-compiled functions; the T3 runner resumes the VM on
    /// THIS module on a deopt (the identity-map module the native code mirrors),
    /// NOT the original — they are observationally equivalent, so the result is
    /// bit-identical to the original on the VM. `None` for T1/T2/f64 (they resume
    /// on the caller-supplied module directly).
    t3_module: Option<std::rc::Rc<crate::bytecode::Module>>,
    /// T4 P3 INLINING: the ORIGINAL (un-inlined) CALLER module that EVERY deopt
    /// from this inlined T4 function resumes the VM on. When T4 inlines a callee
    /// into the caller's body, codegen runs over a FUSED module (callee body spliced
    /// in) but each guard's `DeoptSite.bc_pc` is mapped (via the backend's
    /// `bc_pc_map`) back to the corresponding ORIGINAL caller op index — an inlined-
    /// region guard maps to the caller's `Call` op (so the VM re-runs the ordinary
    /// non-inlined call), and a caller-region op maps to its own original index.
    /// Because the inliner keeps every caller register in its original bank slot
    /// (the callee window lives ABOVE the caller's regs) and only writes the call's
    /// `dst` once, after all inlined guards, the bank slots `0..caller_n_regs` are
    /// always a valid caller register image at the mapped resume op — the INLINE-
    /// DEOPT-TO-CALLER invariant (osr.rs Extension 1). `Some` only for an inlined T4
    /// function; `None` for every other tier (resume uses `t3_module` / the
    /// caller-supplied module as before).
    t4_deopt_module: Option<std::rc::Rc<crate::bytecode::Module>>,
}

impl std::fmt::Debug for JitFunction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("JitFunction")
            .field("base", &self.base)
            .field("size", &self.size)
            .field("cage_owned", &self.cage_owned)
            .field("deopt_sites", &self.deopt_sites.len())
            .finish()
    }
}

// SAFETY: The code page is read-only-executable at run time and the
// JitFunction owns the underlying allocation. Send is safe because
// dropping releases the page via VirtualFree on any thread.
unsafe impl Send for JitFunction {}

// `pub(crate)` so the code-cage module (`jit_cage`) reuses these bindings —
// there must be exactly ONE declaration of each kernel32 symbol in the crate.
#[cfg(target_os = "windows")]
pub(crate) mod win {
    use core::ffi::c_void;
    pub(crate) type LPVOID = *mut c_void;
    pub(crate) type SIZE_T = usize;
    pub(crate) type DWORD = u32;
    pub(crate) const MEM_COMMIT: DWORD = 0x1000;
    pub(crate) const MEM_RESERVE: DWORD = 0x2000;
    pub(crate) const MEM_RELEASE: DWORD = 0x8000;
    pub(crate) const MEM_DECOMMIT: DWORD = 0x4000;
    pub(crate) const PAGE_READWRITE: DWORD = 0x04;
    pub(crate) const PAGE_EXECUTE_READ: DWORD = 0x20;
    #[link(name = "kernel32")]
    unsafe extern "system" {
        pub(crate) fn VirtualAlloc(
            lpAddress: LPVOID,
            dwSize: SIZE_T,
            flAllocationType: DWORD,
            flProtect: DWORD,
        ) -> LPVOID;
        pub(crate) fn VirtualFree(lpAddress: LPVOID, dwSize: SIZE_T, dwFreeType: DWORD) -> i32;
        pub(crate) fn VirtualProtect(
            lpAddress: LPVOID,
            dwSize: SIZE_T,
            flNewProtect: DWORD,
            lpflOldProtect: *mut DWORD,
        ) -> i32;
        pub(crate) fn FlushInstructionCache(
            hProcess: *mut c_void,
            lpBaseAddress: *const c_void,
            dwSize: SIZE_T,
        ) -> i32;
        pub(crate) fn GetCurrentProcess() -> *mut c_void;
    }
}

impl JitFunction {
    /// Install a code buffer into a fresh RWX → RX page. The bytes are
    /// copied; caller may discard the input vector after this returns.
    #[cfg(target_os = "windows")]
    pub fn install(code: &[u8]) -> Result<Self, JitError> {
        if code.is_empty() {
            return Err(JitError::EmptyFunction);
        }
        // Opt-in code-cage path (CV_CODE_CAGE=1): install into the process-wide
        // RX arena so functions are within rel32 range of each other. Falls back
        // to the private per-page path below on any cage decline (disabled, full,
        // or syscall failure) — the cage can only ever cost us the cage path,
        // never correctness.
        if let Some(cf) = crate::jit_cage::install_into_cage(code) {
            return Ok(Self {
                base: cf.rx_ptr as *mut core::ffi::c_void,
                size: cf.code_len,
                cage_owned: true,
                deopt_sites: Vec::new(),
                safepoints: crate::osr::SafepointMap::new(),
                t3_module: None,
                t4_deopt_module: None,
            });
        }
        unsafe {
            let base = win::VirtualAlloc(
                core::ptr::null_mut(),
                code.len(),
                win::MEM_COMMIT | win::MEM_RESERVE,
                win::PAGE_READWRITE,
            );
            if base.is_null() {
                return Err(JitError::OutOfRegisters);
            }
            core::ptr::copy_nonoverlapping(code.as_ptr(), base as *mut u8, code.len());
            let mut old: win::DWORD = 0;
            let ok = win::VirtualProtect(base, code.len(), win::PAGE_EXECUTE_READ, &raw mut old);
            if ok == 0 {
                win::VirtualFree(base, 0, win::MEM_RELEASE);
                return Err(JitError::OutOfRegisters);
            }
            win::FlushInstructionCache(win::GetCurrentProcess(), base, code.len());
            Ok(Self {
                base,
                size: code.len(),
                cage_owned: false,
                deopt_sites: Vec::new(),
                safepoints: crate::osr::SafepointMap::new(),
                t3_module: None,
                t4_deopt_module: None,
            })
        }
    }

    /// Stub for non-Windows targets — JIT codegen still runs in
    /// tests, but installation requires the OS page allocator. This
    /// returns an error so callers can fall back to the interpreter.
    #[cfg(not(target_os = "windows"))]
    pub fn install(_code: &[u8]) -> Result<Self, JitError> {
        Err(JitError::UnsupportedOp)
    }

    /// Attach the T2 Phase-5 deopt site table (the resume map) to this installed
    /// function, returning self for chaining. Used right after `install` for a
    /// T2 function compiled with resume support.
    pub fn with_deopt_sites(mut self, sites: Vec<crate::osr::DeoptSite>) -> Self {
        self.deopt_sites = sites;
        self
    }

    /// The deopt site for a `deopt_id` a guard stub wrote to `*out` (None on an
    /// out-of-range id — a corruption guard; the runner falls back to a Tier-A
    /// whole-function re-run rather than resume at a bogus pc).
    pub fn deopt_site(&self, id: usize) -> Option<crate::osr::DeoptSite> {
        self.deopt_sites.get(id).copied()
    }

    /// Number of recorded deopt sites (test / oracle introspection).
    pub fn deopt_site_count(&self) -> usize {
        self.deopt_sites.len()
    }

    /// Attach the B3 safepoint stack map (native-offset → live heap-root set) to
    /// this installed function. Used right after `install` for a T3 function that
    /// holds heap refs across calls. T1/T2/f64 functions leave it empty.
    pub fn with_safepoints(mut self, map: crate::osr::SafepointMap) -> Self {
        self.safepoints = map;
        self
    }

    /// The safepoint stack map (B3 GC rooting consults this for a collection PC).
    pub fn safepoints(&self) -> &crate::osr::SafepointMap {
        &self.safepoints
    }

    /// The safepoint record at a native code offset, if `native_off` is a recorded
    /// safepoint (the GC has the return-address PC and wants its root set).
    pub fn safepoint_at(&self, native_off: usize) -> Option<&crate::osr::SafepointRec> {
        self.safepoints.find(native_off)
    }

    /// Attach the T3 OPTIMIZED module (the identity-map module a T3 deopt resumes
    /// the VM on). Used right after `install` for a T3-compiled function.
    pub fn with_t3_module(mut self, module: std::rc::Rc<crate::bytecode::Module>) -> Self {
        self.t3_module = Some(module);
        self
    }

    /// The T3 optimized module this native code was compiled from, if any. The T3
    /// runner resumes the VM on THIS module on a deopt.
    pub fn t3_module(&self) -> Option<&std::rc::Rc<crate::bytecode::Module>> {
        self.t3_module.as_ref()
    }

    /// T4 P3 — attach the ORIGINAL (un-inlined) caller module that an INLINED T4
    /// function's deopts resume on (see the field doc). When set, the T4 runner
    /// resumes the VM on THIS module (whose `Call` op is intact) instead of the
    /// fused `t3_module`, using the mapped `DeoptSite.bc_pc`.
    pub fn with_t4_deopt_module(mut self, module: std::rc::Rc<crate::bytecode::Module>) -> Self {
        self.t4_deopt_module = Some(module);
        self
    }

    /// The original caller module an inlined T4 function resumes the VM on, if this
    /// is an inlined T4 function (`None` otherwise — the runner falls back to
    /// `t3_module`).
    pub fn t4_deopt_module(&self) -> Option<&std::rc::Rc<crate::bytecode::Module>> {
        self.t4_deopt_module.as_ref()
    }

    /// Call the installed function. V1 signature is `() -> u64` — the
    /// register allocator routes returns through Rax, so the value
    /// the JIT-emitted `ret` leaves in Rax is what the native ABI
    /// returns as a 64-bit integer.
    ///
    /// SAFETY: caller asserts that the installed bytes are a valid
    /// `extern "system" fn() -> u64` per the System V/Win64 ABI.
    #[cfg(target_os = "windows")]
    pub unsafe fn call(&self) -> u64 {
        let f: extern "system" fn() -> u64 = unsafe { core::mem::transmute(self.base) };
        f()
    }
    #[cfg(not(target_os = "windows"))]
    pub unsafe fn call(&self) -> u64 {
        0
    }

    /// Call a `compile_f64`-produced function. Win64 returns f64 in xmm0, so the
    /// native signature is `extern "system" fn() -> f64`.
    ///
    /// SAFETY: caller asserts the installed bytes are such a function.
    #[cfg(target_os = "windows")]
    pub unsafe fn call_f64(&self) -> f64 {
        let f: extern "system" fn() -> f64 = unsafe { core::mem::transmute(self.base) };
        f()
    }
    #[cfg(not(target_os = "windows"))]
    pub unsafe fn call_f64(&self) -> f64 {
        0.0
    }

    /// Call a `compile_f64` function with up to 4 f64 args. Win64 passes f64
    /// args in xmm0..xmm3 and returns in xmm0 — matching the JIT's vreg→xmm
    /// mapping (params = registers 0..n).
    ///
    /// SAFETY: caller asserts the installed bytes match the chosen arity.
    #[cfg(target_os = "windows")]
    pub unsafe fn call_f64_args(&self, args: &[f64]) -> f64 {
        unsafe {
            match args.len() {
                0 => (core::mem::transmute::<_, extern "system" fn() -> f64>(self.base))(),
                1 => {
                    (core::mem::transmute::<_, extern "system" fn(f64) -> f64>(self.base))(args[0])
                }
                2 => (core::mem::transmute::<_, extern "system" fn(f64, f64) -> f64>(self.base))(
                    args[0], args[1],
                ),
                3 => (core::mem::transmute::<_, extern "system" fn(f64, f64, f64) -> f64>(
                    self.base,
                ))(args[0], args[1], args[2]),
                _ => (core::mem::transmute::<_, extern "system" fn(f64, f64, f64, f64) -> f64>(
                    self.base,
                ))(args[0], args[1], args[2], args[3]),
            }
        }
    }
    #[cfg(not(target_os = "windows"))]
    pub unsafe fn call_f64_args(&self, _args: &[f64]) -> f64 {
        0.0
    }

    /// Call a `compile_baseline_t1`-produced function. The native ABI is
    /// `extern "system" fn(state_ptr) -> u64`: Win64 passes the single pointer
    /// arg in RCX and returns the `T1_*` status TAG in RAX. `state` is an opaque
    /// `*mut VmState` (kept opaque here so `cv_asm`/`jit` don't depend on the VM
    /// layout); the cv_js caller supplies and interprets it.
    ///
    /// SAFETY: caller asserts the installed bytes are exactly such a function and
    /// that `state` points to a live `VmState` (with a live `out` slot) for the
    /// duration of the call.
    #[cfg(target_os = "windows")]
    pub unsafe fn call_t1(&self, state: *mut core::ffi::c_void) -> u64 {
        let f: extern "system" fn(*mut core::ffi::c_void) -> u64 =
            unsafe { core::mem::transmute(self.base) };
        f(state)
    }
    #[cfg(not(target_os = "windows"))]
    pub unsafe fn call_t1(&self, _state: *mut core::ffi::c_void) -> u64 {
        0
    }

    /// Call a `compile_t2lite`-produced function. Win64 ABI (P4):
    /// `extern "system" fn(bank: *mut u64, out: *mut u64, ctx: *mut c_void) -> u64`
    /// (RCX = bank, RDX = out, R8 = ctx, RAX = a `T2_*` status). `bank` points to
    /// the `JsVal` register bank (each `JsVal` is a `u64`); on `T2_RETURNED` the
    /// result `JsVal` bits are written to `*out`. `ctx` is the P4 re-entry context
    /// (`*mut T2CallCtx`), or NULL when the function has no call op.
    ///
    /// This 2-arg form passes a NULL ctx — for the numeric-only / no-call callers
    /// and the existing tests. A function with a call op MUST be invoked via
    /// [`Self::call_t2lite_ctx`] with a live ctx, or the helper deref of a null ctx
    /// would fault. The compiler only emits a call op when a `T2CallConfig` is
    /// supplied, and the production runner always uses the ctx form for those.
    ///
    /// SAFETY: caller asserts the installed bytes are exactly such a function and
    /// that `bank` (≥ the function's max register, 8 bytes/slot) and `out` are
    /// live for the call. With a NULL ctx the bank must hold only immediates and
    /// the function must contain no call op (else a null deref). The compiled
    /// function ALWAYS has the 3-arg ABI; passing the 3rd arg is ABI-required.
    #[cfg(target_os = "windows")]
    pub unsafe fn call_t2lite(&self, bank: *mut u64, out: *mut u64) -> u64 {
        unsafe { self.call_t2lite_ctx(bank, out, core::ptr::null_mut()) }
    }
    #[cfg(not(target_os = "windows"))]
    pub unsafe fn call_t2lite(&self, _bank: *mut u64, _out: *mut u64) -> u64 {
        T2_DEOPT
    }

    /// Call a `compile_t2lite` function with the P4 re-entry context (the 3-arg
    /// ABI). `ctx` is a `*mut T2CallCtx` (kept opaque as `*mut c_void` so
    /// `jit`/`cv_asm` stay independent of the VM layout); the cv_js runner supplies
    /// and interprets it. Used for any function containing a CALL op.
    ///
    /// SAFETY: as [`Self::call_t2lite`], plus `ctx` (when non-null) must point to a
    /// live `T2CallCtx` for the whole call (its module/globals/dispatch borrows
    /// outlive the call), and the bank must be the OWNING + GC-rooted bank (a
    /// re-entrant call can GC + produce heap results).
    #[cfg(target_os = "windows")]
    pub unsafe fn call_t2lite_ctx(
        &self,
        bank: *mut u64,
        out: *mut u64,
        ctx: *mut core::ffi::c_void,
    ) -> u64 {
        let f: extern "system" fn(*mut u64, *mut u64, *mut core::ffi::c_void) -> u64 =
            unsafe { core::mem::transmute(self.base) };
        f(bank, out, ctx)
    }
    #[cfg(not(target_os = "windows"))]
    pub unsafe fn call_t2lite_ctx(
        &self,
        _bank: *mut u64,
        _out: *mut u64,
        _ctx: *mut core::ffi::c_void,
    ) -> u64 {
        T2_DEOPT
    }

    pub fn size(&self) -> usize {
        self.size
    }

    /// The raw base address of the installed code page (for building native
    /// trampolines / cross-calling one installed function from another, e.g. the
    /// movaps ABI regression test). The bytes are RX; treat as a code pointer.
    pub fn base_ptr(&self) -> *const core::ffi::c_void {
        self.base
    }
}

impl Drop for JitFunction {
    fn drop(&mut self) {
        #[cfg(target_os = "windows")]
        unsafe {
            // Cage-owned pages are bump-allocated and process-lifetime; freeing
            // one would VirtualFree a region we don't own the whole reservation
            // of. Only private per-page installs are released here.
            if !self.cage_owned && !self.base.is_null() {
                win::VirtualFree(self.base, 0, win::MEM_RELEASE);
            }
        }
    }
}

/// Top-level entry point used by conclave / cv_js to JIT-compile a
/// trace and install it as a callable function. Returns None on any
/// failure — caller falls back to the interpreter.
pub fn try_compile_and_install(ops: &[JitOp]) -> Option<JitFunction> {
    let code = compile(ops).ok()?;
    JitFunction::install(&code).ok()
}

/// Lower a bytecode function's ops into JitOps. Bails (returns None)
/// the moment it hits anything unsupported — caller falls back to the
/// interpreter for that function. V1 supports:
///   * LoadConst with integer constants
///   * Move
///   * Add, Sub, Mul (integer; non-integer constants bail)
///   * Return (final register is whatever the last write was)
///
/// The constants pool must be supplied (the bytecode VM stores them
/// alongside each function); only integer pool entries pass through.
pub fn lower_bytecode(
    ops: &[crate::bytecode::Op],
    const_int: impl Fn(u16) -> Option<i32>,
) -> Option<Vec<JitOp>> {
    use crate::bytecode::Op;
    let mut out: Vec<JitOp> = Vec::with_capacity(ops.len() + 1);
    let mut last_written: Option<u16> = None;
    for op in ops {
        match *op {
            Op::LoadConst { dst, k } => {
                let v = const_int(k)?;
                out.push(JitOp::ConstInt { dst, value: v });
                last_written = Some(dst);
            }
            Op::Move { dst, src } => {
                out.push(JitOp::Mov { dst, src });
                last_written = Some(dst);
            }
            Op::Add { dst, lhs, rhs } => {
                out.push(JitOp::Add {
                    dst,
                    a: lhs,
                    b: rhs,
                });
                last_written = Some(dst);
            }
            Op::Sub { dst, lhs, rhs } => {
                out.push(JitOp::Sub {
                    dst,
                    a: lhs,
                    b: rhs,
                });
                last_written = Some(dst);
            }
            Op::Mul { dst, lhs, rhs } => {
                out.push(JitOp::Mul {
                    dst,
                    a: lhs,
                    b: rhs,
                });
                last_written = Some(dst);
            }
            // Anything else makes the trace untranslatable.
            _ => return None,
        }
    }
    let return_reg = last_written?;
    out.push(JitOp::Return { reg: return_reg });
    Some(out)
}

/// Lower a bytecode function to **f64** Jit ops (the correct path for JS numbers).
/// Params arrive in xmm0..xmm3 (Win64 f64 ABI) = bytecode registers 0..n_params,
/// so no explicit arg-load is emitted. Bails (None → interpreter) on anything
/// outside straight-line double arithmetic: any non-arithmetic op, a register
/// index > 5 (no spilling yet), or a non-number constant. Supports:
///   LoadConst(number), Move, Add, Sub, Mul, Div, Ret.
pub fn lower_bytecode_f64(
    ops: &[crate::bytecode::Op],
    const_f64: impl Fn(u16) -> Option<f64>,
) -> Option<Vec<FJitOp>> {
    use crate::bytecode::Op;
    const MAX_REG: u16 = 5;
    let ok = |r: u16| -> Option<()> { if r <= MAX_REG { Some(()) } else { None } };
    let mut out: Vec<FJitOp> = Vec::with_capacity(ops.len());
    for op in ops {
        match *op {
            Op::LoadConst { dst, k } => {
                ok(dst)?;
                let f = const_f64(k)?;
                out.push(FJitOp::FConst {
                    dst,
                    bits: f.to_bits(),
                });
            }
            Op::Move { dst, src } => {
                ok(dst)?;
                ok(src)?;
                out.push(FJitOp::FMove { dst, src });
            }
            Op::Add { dst, lhs, rhs } => {
                ok(dst)?;
                ok(lhs)?;
                ok(rhs)?;
                out.push(FJitOp::FAdd {
                    dst,
                    a: lhs,
                    b: rhs,
                });
            }
            Op::Sub { dst, lhs, rhs } => {
                ok(dst)?;
                ok(lhs)?;
                ok(rhs)?;
                out.push(FJitOp::FSub {
                    dst,
                    a: lhs,
                    b: rhs,
                });
            }
            Op::Mul { dst, lhs, rhs } => {
                ok(dst)?;
                ok(lhs)?;
                ok(rhs)?;
                out.push(FJitOp::FMul {
                    dst,
                    a: lhs,
                    b: rhs,
                });
            }
            Op::Div { dst, lhs, rhs } => {
                ok(dst)?;
                ok(lhs)?;
                ok(rhs)?;
                out.push(FJitOp::FDiv {
                    dst,
                    a: lhs,
                    b: rhs,
                });
            }
            Op::Ret { src } => {
                ok(src)?;
                out.push(FJitOp::FRet { reg: src });
                return Some(out);
            }
            // Anything else (branches, calls, property access, …) makes the
            // function untranslatable for this V1 f64 JIT.
            _ => return None,
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    // ───────────────────── B1 — safepoint-map emission seam ─────────────────────

    #[test]
    fn safepoint_map_emission_seam_records_at_call_and_backedge_offsets() {
        use crate::osr::{SafepointKind, SafepointMap};
        // Simulate a tiny T3 codegen that records a safepoint at the return
        // address of an emitted helper call and at a loop back-edge. We assert the
        // map keys on the EXACT native offsets `Emitter::here()` reports, and that
        // the live-root bitset excludes the number-lane register.
        let mut em = Emitter::new();
        em.push_r64(R64::Rbp);
        em.mov_r64_r64(R64::Rbp, R64::Rsp);

        let mut sp = SafepointMap::new();

        // Helper call: emit `call rax`, then record the safepoint at the RETURN
        // address (the PC the GC observes) — i.e. `here()` AFTER the call. Pointer
        // values live across the call are in bank slots 1 and 4 (the receiver and a
        // captured object); bank slot 2 holds an unboxed number and is EXCLUDED.
        em.call_r64(R64::Rax);
        let call_ret_off = em.here();
        sp.record(
            call_ret_off,
            SafepointKind::HelperCall,
            SafepointMap::roots_from_slots([1usize, 4]),
        );

        // Some straight-line code, then a back-edge (a backward jmp). Record the
        // back-edge safepoint at its native offset; the loop-carried receiver in
        // slot 1 is still a live root.
        em.add_r64_r64(R64::Rbx, R64::Rdi);
        let back_edge_off = em.here();
        sp.record(
            back_edge_off,
            SafepointKind::BackEdge,
            SafepointMap::roots_from_slots([1usize]),
        );
        em.pop_r64(R64::Rbp);
        em.ret();

        // Install into a real page and attach the safepoint map.
        let f = JitFunction::install(&em.code)
            .expect("install")
            .with_safepoints(sp);

        assert_eq!(f.safepoints().len(), 2);

        // The GC, given the helper-call return-address offset, recovers exactly the
        // pointer-lane roots (slots 1 and 4) and NOT the number-lane slot 2.
        let hc = f.safepoint_at(call_ret_off).expect("helper-call safepoint");
        assert_eq!(hc.kind, SafepointKind::HelperCall);
        assert!(hc.is_root(1) && hc.is_root(4));
        assert!(!hc.is_root(2), "number-lane slot 2 must NOT be a GC root");
        assert_eq!(hc.root_count(), 2);

        // The back-edge safepoint has only the loop-carried receiver rooted.
        let be = f.safepoint_at(back_edge_off).expect("back-edge safepoint");
        assert_eq!(be.kind, SafepointKind::BackEdge);
        assert!(be.is_root(1));
        assert_eq!(be.root_count(), 1);

        // A non-safepoint native offset has no record (GC at a non-safe PC is not a
        // thing T3 emits; the map is precise).
        assert!(f.safepoint_at(call_ret_off + 1).is_none());

        // B3 GC-INTEGRATION precondition: at the helper-call safepoint the live
        // pointer roots are slots 1 and 4, so a bank of >= 5 slots COVERS them
        // (gc_seed_jit_banks scans bank[0..5]). A bank of only 4 slots does NOT
        // cover slot 4 → the integration check rejects it (the would-be UAF a
        // missing spill / undersized bank causes).
        assert!(
            crate::interp::gc_safepoint_roots_covered(&f, call_ret_off, 5),
            "a 5-slot bank covers pointer roots in slots 1 and 4"
        );
        assert!(
            !crate::interp::gc_safepoint_roots_covered(&f, call_ret_off, 4),
            "a 4-slot bank does NOT cover the root in slot 4 — the integration \
             check must reject it (missing-spill UAF)"
        );
        // A PC that is not a safepoint is vacuously covered (nothing claimed live).
        assert!(crate::interp::gc_safepoint_roots_covered(&f, call_ret_off + 1, 0));
    }

    // ───────────────────── M4.2a — T1 baseline codegen ─────────────────────

    #[test]
    fn t1_emits_skeleton_and_declines_unsupported() {
        use crate::bytecode::Op;
        // Supported subset: LoadConst + Ret. Per op we emit one `call`.
        let supported = |o: &Op| {
            matches!(
                o,
                Op::LoadConst { .. } | Op::Ret { .. } | Op::Add { .. } | Op::Jmp { .. }
            )
        };
        let code = vec![Op::LoadConst { dst: 0, k: 0 }, Op::Ret { src: 0 }];
        let bytes = compile_baseline_t1(&code, &supported, 0xDEAD_BEEF, 0, 1)
            .expect("subset-only function must compile");
        assert!(!bytes.is_empty());
        // Prologue `push rbx` = 0x53 is the first byte.
        assert_eq!(bytes[0], 0x53, "prologue must start with push rbx");
        // Ends in `ret` (0xC3) at the epilogue.
        assert_eq!(*bytes.last().unwrap(), 0xC3, "must end in ret");
        // Two ops → at least two indirect `call rax` (FF D0) sequences.
        let calls = bytes.windows(2).filter(|w| *w == [0xFF, 0xD0]).count();
        assert!(calls >= 2, "expected >=2 call rax, got {calls}");

        // An unsupported op anywhere → decline (None), never a partial compile.
        let bad = vec![
            Op::LoadConst { dst: 0, k: 0 },
            Op::Mul { dst: 1, lhs: 0, rhs: 0 }, // not in `supported` above
            Op::Ret { src: 1 },
        ];
        assert!(
            compile_baseline_t1(&bad, &supported, 0xDEAD_BEEF, 0, 1).is_none(),
            "a function with an unsupported op must decline"
        );

        // Empty code → decline.
        assert!(compile_baseline_t1(&[], &supported, 0, 0, 1).is_none());
    }

    #[test]
    fn t1_jump_target_out_of_range_declines() {
        use crate::bytecode::Op;
        let supported = |o: &Op| matches!(o, Op::Jmp { .. } | Op::Ret { .. });
        // Jmp to bytecode index 99 which doesn't exist (only 2 ops) → decline.
        let code = vec![Op::Jmp { target: 99 }, Op::Ret { src: 0 }];
        assert!(
            compile_baseline_t1(&code, &supported, 0x1000, 0, 1).is_none(),
            "an out-of-range jump target must decline (never emit a wild jump)"
        );
    }

    #[test]
    fn profiler_marks_hot_after_threshold() {
        let mut p = Profiler::new();
        for _ in 0..10 {
            p.record(0xabcd);
        }
        assert!(p.is_hot(0xabcd, 10));
        assert!(!p.is_hot(0xabcd, 100));
    }

    #[test]
    fn allocator_assigns_then_releases() {
        let mut a = Allocator::new();
        let r0 = a.host_of(0).unwrap();
        let r1 = a.host_of(1).unwrap();
        assert_ne!(r0, r1);
        a.release(0);
        let r2 = a.host_of(2).unwrap();
        assert_eq!(r2, r0);
    }

    #[test]
    fn allocator_runs_out_with_too_many_live_vregs() {
        let mut a = Allocator::new();
        for v in 0..4 {
            a.host_of(v).unwrap();
        }
        assert!(a.host_of(5).is_none());
    }

    #[test]
    fn compile_const_then_return_emits_native_code() {
        let ops = vec![
            JitOp::ConstInt { dst: 0, value: 42 },
            JitOp::Return { reg: 0 },
        ];
        let code = compile(&ops).unwrap();
        // Must be non-empty and end with `ret` (0xC3).
        assert!(!code.is_empty());
        assert_eq!(*code.last().unwrap(), 0xC3);
    }

    #[test]
    fn compile_arithmetic_sequence() {
        // (10 + 20) * 3 - 5
        let ops = vec![
            JitOp::ConstInt { dst: 0, value: 10 },
            JitOp::ConstInt { dst: 1, value: 20 },
            JitOp::Add { dst: 2, a: 0, b: 1 },
            JitOp::ConstInt { dst: 3, value: 3 },
            JitOp::Mul { dst: 0, a: 2, b: 3 },
            JitOp::ConstInt { dst: 1, value: 5 },
            JitOp::Sub { dst: 2, a: 0, b: 1 },
            JitOp::Return { reg: 2 },
        ];
        let code = compile(&ops).unwrap();
        assert!(code.len() > 30);
        // Prolog mov rbp, rsp: 48 89 E5.
        assert!(code.windows(3).any(|w| w == [0x48, 0x89, 0xE5]));
        // Epilog ret.
        assert_eq!(*code.last().unwrap(), 0xC3);
    }

    #[test]
    fn empty_program_is_rejected() {
        assert!(matches!(compile(&[]), Err(JitError::EmptyFunction)));
    }

    #[test]
    fn lower_bytecode_translates_arithmetic() {
        use crate::bytecode::Op;
        let ops = vec![
            Op::LoadConst { dst: 0, k: 0 }, // 10
            Op::LoadConst { dst: 1, k: 1 }, // 20
            Op::Add {
                dst: 2,
                lhs: 0,
                rhs: 1,
            },
        ];
        let consts = |k: u16| match k {
            0 => Some(10),
            1 => Some(20),
            _ => None,
        };
        let lowered = lower_bytecode(&ops, consts).expect("should lower");
        // 3 source ops → 3 JitOps + Return.
        assert_eq!(lowered.len(), 4);
        assert!(matches!(lowered[3], JitOp::Return { reg: 2 }));
    }

    #[test]
    fn lower_bytecode_bails_on_unsupported_op() {
        use crate::bytecode::Op;
        // Op::JmpIfFalse — not in the supported subset.
        let ops = vec![
            Op::LoadConst { dst: 0, k: 0 },
            Op::JmpIfFalse { cond: 0, target: 5 },
        ];
        let consts = |_| Some(0);
        assert!(lower_bytecode(&ops, consts).is_none());
    }

    #[cfg(target_os = "windows")]
    #[test]
    fn install_and_call_returns_constant() {
        // const 42; return 0  →  JIT returns 42.
        let ops = vec![
            JitOp::ConstInt { dst: 0, value: 42 },
            JitOp::Return { reg: 0 },
        ];
        let f = try_compile_and_install(&ops).expect("install should succeed");
        let v = unsafe { f.call() };
        // Rax is set by mov_r64_imm32 (sign-extended). 42 fits in 32 bits.
        assert_eq!(v as i64, 42);
    }

    #[test]
    fn compile_f64_emits_code() {
        // (1.5 + 2.5) * 2.0
        let ops = vec![
            FJitOp::FConst {
                dst: 0,
                bits: 1.5f64.to_bits(),
            },
            FJitOp::FConst {
                dst: 1,
                bits: 2.5f64.to_bits(),
            },
            FJitOp::FAdd { dst: 2, a: 0, b: 1 },
            FJitOp::FConst {
                dst: 3,
                bits: 2.0f64.to_bits(),
            },
            FJitOp::FMul { dst: 0, a: 2, b: 3 },
            FJitOp::FRet { reg: 0 },
        ];
        let code = compile_f64(&ops).unwrap();
        assert!(!code.is_empty());
        assert_eq!(*code.last().unwrap(), 0xC3); // ends in ret
    }

    #[cfg(target_os = "windows")]
    #[test]
    fn jit_f64_executes_arithmetic() {
        // (1.5 + 2.5) * 2.0 = 8.0 — full f64 codegen → install → native call.
        let ops = vec![
            FJitOp::FConst {
                dst: 0,
                bits: 1.5f64.to_bits(),
            },
            FJitOp::FConst {
                dst: 1,
                bits: 2.5f64.to_bits(),
            },
            FJitOp::FAdd { dst: 2, a: 0, b: 1 },
            FJitOp::FConst {
                dst: 3,
                bits: 2.0f64.to_bits(),
            },
            FJitOp::FMul { dst: 0, a: 2, b: 3 },
            FJitOp::FRet { reg: 0 },
        ];
        let code = compile_f64(&ops).unwrap();
        let f = JitFunction::install(&code).expect("install");
        let v = unsafe { f.call_f64() };
        assert!((v - 8.0).abs() < 1e-12, "got {v}");
    }

    #[cfg(target_os = "windows")]
    #[test]
    fn jit_lowers_and_runs_real_function_with_args() {
        // Compile a REAL JS function to bytecode, lower it to the f64 JIT, run
        // it natively with args. `poly(a,b) = a*a + b` → poly(5,3) = 28.
        use crate::bytecode::compile_single_function;
        let prog =
            crate::parser::parse_program("function poly(a, b) { return a * a + b; }").unwrap();
        let (params, body) = match &prog[0] {
            crate::ast::Stmt::FunctionDecl { params, body, .. } => (params.clone(), body.clone()),
            other => panic!("expected fn decl, got {other:?}"),
        };
        let (module, _ups) = compile_single_function(&params, &body, &[]).unwrap();
        let f = &module.fns[0];
        let lowered = lower_bytecode_f64(&f.code, |k| match f.consts.get(k as usize) {
            Some(crate::interp::Value::Number(n)) => Some(*n),
            _ => None,
        })
        .expect("function should lower to the f64 JIT");
        let code = compile_f64(&lowered).unwrap();
        let jf = JitFunction::install(&code).expect("install");
        let v = unsafe { jf.call_f64_args(&[5.0, 3.0]) };
        assert!((v - 28.0).abs() < 1e-9, "poly(5,3) got {v}");
    }

    #[cfg(target_os = "windows")]
    #[test]
    fn jit_compiles_and_runs_loop_function() {
        // The Pass-4 payoff: a function with an internal loop runs as native
        // code. sumTo(n) = 0+1+…+(n-1); sumTo(100) = 4950, sumTo(1000) = 499500.
        use crate::bytecode::compile_single_function;
        let prog = crate::parser::parse_program(
            "function sumTo(n){ var s = 0; for (var i = 0; i < n; i = i + 1) { s = s + i; } return s; }",
        )
        .unwrap();
        let (params, body) = match &prog[0] {
            crate::ast::Stmt::FunctionDecl { params, body, .. } => (params.clone(), body.clone()),
            other => panic!("expected fn decl, got {other:?}"),
        };
        let (module, _ups) = compile_single_function(&params, &body, &[]).unwrap();
        let f = &module.fns[0];
        let code = compile_bytecode_f64(&f.code, f.n_params, |k| match f.consts.get(k as usize) {
            Some(crate::interp::Value::Number(n)) => Some(*n),
            _ => None,
        })
        .expect("loop function should compile to native code");
        let jf = JitFunction::install(&code).expect("install");
        let v100 = unsafe { jf.call_f64_args(&[100.0]) };
        assert!((v100 - 4950.0).abs() < 1e-9, "sumTo(100) got {v100}");
        let v1000 = unsafe { jf.call_f64_args(&[1000.0]) };
        assert!((v1000 - 499500.0).abs() < 1e-9, "sumTo(1000) got {v1000}");
    }

    #[cfg(target_os = "windows")]
    #[test]
    fn jit_f64_division_executes() {
        // 7.0 / 2.0 = 3.5
        let ops = vec![
            FJitOp::FConst {
                dst: 0,
                bits: 7.0f64.to_bits(),
            },
            FJitOp::FConst {
                dst: 1,
                bits: 2.0f64.to_bits(),
            },
            FJitOp::FDiv { dst: 2, a: 0, b: 1 },
            FJitOp::FRet { reg: 2 },
        ];
        let f = JitFunction::install(&compile_f64(&ops).unwrap()).expect("install");
        let v = unsafe { f.call_f64() };
        assert!((v - 3.5).abs() < 1e-12, "got {v}");
    }

    #[cfg(target_os = "windows")]
    #[test]
    fn install_and_call_returns_arithmetic_result() {
        // (10 + 20) * 3 - 5 = 85
        let ops = vec![
            JitOp::ConstInt { dst: 0, value: 10 },
            JitOp::ConstInt { dst: 1, value: 20 },
            JitOp::Add { dst: 2, a: 0, b: 1 },
            JitOp::ConstInt { dst: 3, value: 3 },
            JitOp::Mul { dst: 0, a: 2, b: 3 },
            JitOp::ConstInt { dst: 1, value: 5 },
            JitOp::Sub { dst: 2, a: 0, b: 1 },
            JitOp::Return { reg: 2 },
        ];
        let f = try_compile_and_install(&ops).expect("install should succeed");
        let v = unsafe { f.call() };
        assert_eq!(v as i64, 85);
    }

    // ─────────────── M4.2b movaps epilog ABI fix — the regression test ────────
    //
    // The Win64 ABI requires xmm6..xmm15 preserved across a call in their FULL
    // 128 bits. The old f64-JIT epilog restored them with 64-bit `movsd`, which
    // ZEROES the upper 64 bits — corrupting a caller that holds data there. This
    // test builds a tiny native WRAPPER that: writes a known 128-bit pattern into
    // xmm6 (low + UPPER lane), calls a callee that uses xmm6 (so it saves +
    // restores it), then reads xmm6's UPPER 64 bits back. With the movaps fix the
    // upper lane survives; with the old movsd it would read back zero.

    /// Build a callee that forces the f64 JIT to use xmm6 (≥7 registers) so its
    /// prolog/epilog save/restore xmm6. A function `g(a)` with a long
    /// straight-line chain into high registers does it: returns `a` but touches
    /// xmm6. We just need any compiled fn whose body uses reg 6.
    #[cfg(target_os = "windows")]
    fn callee_using_xmm6() -> JitFunction {
        use crate::bytecode::Op;
        // r6 = const ; r6 = r6 + a(r0) ; ret r0  → uses xmm6, returns the arg.
        let code = vec![
            Op::LoadConst { dst: 6, k: 0 },
            Op::Add { dst: 6, lhs: 6, rhs: 0 },
            Op::Ret { src: 0 },
        ];
        let bytes = compile_bytecode_f64(&code, 1, |_k| Some(1.0))
            .expect("callee must compile (uses xmm6)");
        JitFunction::install(&bytes).expect("install callee")
    }

    #[cfg(target_os = "windows")]
    #[test]
    fn f64_epilog_preserves_full_xmm6_via_movaps() {
        // The wrapper ABI: extern "system" fn(callee_ptr: u64) -> u64. It returns
        // the UPPER 64 bits of xmm6 observed AFTER calling the callee.
        let mut em = Emitter::new();
        // Prolog: align + shadow space for the call (Win64 needs 32B shadow,
        // 16B-aligned RSP at the call site). Entry RSP≡8; sub 40 → ≡0 (8 pad +
        // 32 shadow). We stash the callee ptr in a callee-saved reg (rbx).
        em.push_r64(R64::Rbx); // RSP ≡ 0
        em.sub_r64_imm32(R64::Rsp, 32); // shadow space, RSP ≡ 0 (32 is 16-aligned)
        em.mov_r64_r64(R64::Rbx, R64::Rcx); // rbx = callee ptr
        // Build a 128-bit pattern on the stack and load it into xmm6.
        //   low  = 1.0 bits, upper = 0xDEADBEEFCAFEBABE (the canary).
        // Use the 32B shadow region as scratch (we own it before the call sets
        // up its own; but the callee will clobber shadow — so write+load BEFORE
        // setting the arg, and keep the canary only in xmm6 across the call).
        em.mov_r64_imm64(R64::Rax, 1.0f64.to_bits() as i64);
        em.mov_mem_r64(R64::Rsp, 0, R64::Rax); // [rsp+0] = low (1.0)
        em.mov_r64_imm64(R64::Rax, 0xDEAD_BEEF_CAFE_BABEu64 as i64);
        em.mov_mem_r64(R64::Rsp, 8, R64::Rax); // [rsp+8] = upper canary
        // [rsp] is 16-aligned (RSP≡0) → movaps is legal.
        em.movaps_xmm_mem(Xmm::Xmm6, R64::Rsp, 0); // xmm6 = {1.0, canary}
        // Set the callee arg (xmm0 = 1.0) and call it. If the callee's epilog
        // uses movsd it will zero xmm6's upper half; movaps preserves it.
        em.mov_r64_imm64(R64::Rax, 1.0f64.to_bits() as i64);
        em.movq_xmm_r64(Xmm::Xmm0, R64::Rax);
        em.call_r64(R64::Rbx);
        // Read xmm6's UPPER 64 bits back: store xmm6 (movaps) to the stack, load
        // [rsp+8] into rax = the return value.
        em.movaps_mem_xmm(R64::Rsp, 0, Xmm::Xmm6);
        em.mov_r64_mem(R64::Rax, R64::Rsp, 8);
        // Epilog.
        em.add_r64_imm32(R64::Rsp, 32);
        em.pop_r64(R64::Rbx);
        em.ret();

        let wrapper = JitFunction::install(&em.code).expect("install wrapper");
        let callee = callee_using_xmm6();
        let callee_ptr = {
            // The callee's code base address. JitFunction doesn't expose it
            // directly, so call through a transmute of its own call mechanism:
            // we need the raw pointer. Expose via a tiny unsafe read.
            callee.base_ptr() as u64
        };
        let f: extern "system" fn(u64) -> u64 =
            unsafe { core::mem::transmute(wrapper.base_ptr()) };
        let upper = f(callee_ptr);
        assert_eq!(
            upper, 0xDEAD_BEEF_CAFE_BABE,
            "xmm6 UPPER 64 bits were clobbered across the call — the Win64 ABI \
             requires the FULL 128-bit callee-saved xmm to survive (movaps fix). \
             A zero here means the epilog used 64-bit movsd (the M4.2b bug)."
        );
    }

    // ─────────────────────── M4.3 T2-lite codegen shape ──────────────────────

    /// The T2 OBJECT-lane constants must equal what `jsval.rs` produces (a drift
    /// is silent wrong-tag corruption — the #1 hazard). Box a real object and a
    /// real array; assert only the OBJECT matches `JV_OBJECT_TOP16` (so a wrong
    /// receiver kind can never pass the inline is-object guard), and that the
    /// payload mask recovers the `Rc::as_ptr`.
    #[test]
    fn t2_object_lane_constants_match_jsval() {
        use crate::interp::Value;
        use crate::jsval::JsVal;
        use std::cell::RefCell;
        use std::rc::Rc;
        let obj: Rc<RefCell<crate::ordered::OrderedMap<String, Value>>> =
            Rc::new(RefCell::new(crate::ordered::OrderedMap::new()));
        let jv = JsVal::object(&obj);
        assert_eq!(
            jv.bits() & super::JV_TOP16_MASK,
            super::JV_OBJECT_TOP16,
            "an object JsVal's top-16 must equal JV_OBJECT_TOP16"
        );
        assert_eq!(
            (jv.bits() & super::JV_PAYLOAD_MASK) as usize,
            Rc::as_ptr(&obj) as usize,
            "the payload mask must recover the Rc::as_ptr"
        );
        // An ARRAY must NOT match the object signature (tag=1, not 0)…
        let arr: Rc<RefCell<Vec<Value>>> = Rc::new(RefCell::new(vec![]));
        let av = JsVal::array(&arr);
        assert_ne!(
            av.bits() & super::JV_TOP16_MASK,
            super::JV_OBJECT_TOP16,
            "an array JsVal must NOT pass the inline is-object guard"
        );
        // …but it MUST match the ARRAY signature the GetIdx/SetIdx guard bakes (a
        // drift here is a silent wrong-receiver-kind hazard for the array path).
        assert_eq!(
            av.bits() & super::JV_TOP16_MASK,
            super::JV_ARRAY_TOP16,
            "an array JsVal's top-16 must equal JV_ARRAY_TOP16"
        );
        // An OBJECT must NOT pass the array guard (so a non-array receiver deopts).
        assert_ne!(
            jv.bits() & super::JV_TOP16_MASK,
            super::JV_ARRAY_TOP16,
            "an object JsVal must NOT pass the inline is-array guard"
        );
        // The array payload mask recovers the Vec's Rc::as_ptr.
        assert_eq!(
            (av.bits() & super::JV_PAYLOAD_MASK) as usize,
            Rc::as_ptr(&arr) as usize,
            "the payload mask must recover the array Rc::as_ptr"
        );
        // The DEOPT sentinel equals the hole bits (never a success return).
        assert_eq!(super::JV_HOLE, JsVal::hole().bits());
    }

    /// OFFSET-DRIFT GATE: the JIT bakes `t2_shape_header_offset()` as a constant.
    /// It MUST equal the offset the public `shape_header_ptr()` actually reads, on
    /// a fresh instance — a drift between the baked offset and the real header
    /// location is silent wrong-slot corruption. Also pin the `Value` size + the
    /// 96-byte `OrderedMap` budget (the layout the offset depends on).
    #[test]
    fn t2_shape_off_is_stable() {
        use crate::interp::Value;
        use std::cell::RefCell;
        use std::rc::Rc;
        let baked = super::t2_shape_header_offset();
        // Recompute independently from a different instance: must be identical
        // (the offset is a layout constant, not instance-specific).
        let rc: Rc<RefCell<crate::ordered::OrderedMap<String, Value>>> =
            Rc::new(RefCell::new(crate::ordered::OrderedMap::new()));
        let base = Rc::as_ptr(&rc) as usize;
        let hdr = rc.borrow().shape_header_ptr() as usize;
        assert_eq!(
            baked as usize,
            hdr - base,
            "baked SHAPE_OFF drifted from the real header offset — silent corruption risk"
        );
        // The offset must be in-bounds of the object (sane), and 4-byte aligned
        // (a u32 load).
        assert!(baked >= 0 && (baked as usize) + 4 <= std::mem::size_of::<RefCellInnerProbe>());
        assert_eq!(baked % 4, 0, "header must be 4-byte aligned for mov r32");
        // The layout the offset depends on (drift gate):
        assert_eq!(std::mem::size_of::<Value>(), 16, "Value size drifted");
        assert_eq!(
            std::mem::size_of::<crate::ordered::OrderedMap<String, Value>>(),
            96,
            "OrderedMap must stay 96 B (the T2 header packs into ShapeCache padding)"
        );
    }

    /// Sizing helper for the bounds assertion above: a `RefCell<OrderedMap>` is at
    /// least this big, so `SHAPE_OFF + 4` fitting inside it proves the inline read
    /// never over-runs the allocation.
    type RefCellInnerProbe =
        std::cell::RefCell<crate::ordered::OrderedMap<String, crate::interp::Value>>;

    /// END-TO-END T2 inline GetProp: build a real shaped object `{x: 7.5, y: 2}`,
    /// compile a one-op `GetProp dst=1, obj=0` + `Ret 1` with a baked site for the
    /// object's real shape/slot, install, and call with the object's JsVal in
    /// bank[0]. The native code must (a) pass the is-object guard, (b) match the
    /// baked shape inline, (c) call the helper, (d) return the IMMEDIATE 7.5.
    #[cfg(target_os = "windows")]
    #[test]
    fn t2_inline_getprop_reads_immediate_field() {
        use crate::bytecode::Op;
        use crate::interp::Value;
        use crate::jsval::JsVal;
        use std::cell::RefCell;
        use std::rc::Rc;
        // Build {x:7.5, y:2}. Force Shaped via OrderedMap::new (default-on path).
        let mut m: crate::ordered::OrderedMap<String, Value> = crate::ordered::OrderedMap::new();
        m.insert("x".to_string(), Value::Number(7.5));
        m.insert("y".to_string(), Value::Number(2.0));
        let shape = m.shape_header();
        let xslot = m.slot_of("x").expect("x has a slot") as u32;
        // Only meaningful when Shaped is on (header != a Dict sentinel). If the
        // env disabled Shaped, the header is a sentinel → skip (the inline path
        // correctly never engages then; covered by the deopt test).
        if shape == u32::MAX || shape == u32::MAX - 1 {
            return;
        }
        let obj: Rc<RefCell<crate::ordered::OrderedMap<String, Value>>> =
            Rc::new(RefCell::new(m));

        let code = [
            Op::GetProp { dst: 1, obj: 0, key_k: 0 },
            Op::Ret { src: 1 },
        ];
        let site = super::T2GetPropSite { shapes_slots: vec![(shape, xslot)], heap_result: false };
        let cfg = super::T2GetPropConfig {
            site_at: &|i| if i == 0 { Some(site.clone()) } else { None },
            shape_off: super::t2_shape_header_offset(),
            helper_addr: super::rt_getprop_slot_immediate as *const () as usize,
            heap_helper_addr: super::rt_getprop_slot_owning_store as *const () as usize,
            getidx_helper_addr: super::rt_getidx_owning_store as *const () as usize,
            setidx_helper_addr: super::rt_setidx_owning_store as *const () as usize,
        };
        let bytes = super::compile_t2lite(&code, |_k| None, Some(&cfg), super::T2StoreMode::Numeric, None).expect("compiles");
        let jf = JitFunction::install(&bytes).expect("install");
        let mut bank: [u64; 2] = [JsVal::object(&obj).bits(), JsVal::undefined().bits()];
        let mut out: u64 = 0;
        let tag = unsafe { jf.call_t2lite(bank.as_mut_ptr(), &mut out as *mut u64) };
        assert_eq!(tag, super::T2_RETURNED, "must return, not deopt");
        assert_eq!(JsVal(out).as_f64(), Some(7.5), "inline GetProp read obj.x = 7.5");

        // NEGATIVE arm 1: a NON-object receiver (a number) in bank[0] must DEOPT.
        let mut bank2: [u64; 2] = [JsVal::number(42.0).bits(), JsVal::undefined().bits()];
        let mut out2: u64 = 0;
        let tag2 = unsafe { jf.call_t2lite(bank2.as_mut_ptr(), &mut out2 as *mut u64) };
        // T2 Phase 5: an input guard now takes a RESUME deopt (the runner resumes
        // the VM at the guard's bc_pc), writing the DeoptSite id to *out. (This jf
        // was installed without `with_deopt_sites`, so the id table is empty here —
        // the resume-map round-trip is proven by the deopt-fuzz oracle below.)
        assert_eq!(tag2, super::T2_DEOPT_RESUME, "non-object receiver must deopt");

        // NEGATIVE arm 2: a DIFFERENT-shape object (header mismatch) must DEOPT.
        let mut m2: crate::ordered::OrderedMap<String, Value> = crate::ordered::OrderedMap::new();
        m2.insert("z".to_string(), Value::Number(9.0)); // different key sequence
        let obj2: Rc<RefCell<crate::ordered::OrderedMap<String, Value>>> =
            Rc::new(RefCell::new(m2));
        let mut bank3: [u64; 2] = [JsVal::object(&obj2).bits(), JsVal::undefined().bits()];
        let mut out3: u64 = 0;
        let tag3 = unsafe { jf.call_t2lite(bank3.as_mut_ptr(), &mut out3 as *mut u64) };
        assert_eq!(tag3, super::T2_DEOPT_RESUME, "shape-mismatch receiver must deopt");
    }

    /// NON-IMMEDIATE slot value → the helper returns the DEOPT sentinel, so the
    /// native code deopts (the bank must never receive a heap value this phase).
    #[cfg(target_os = "windows")]
    #[test]
    fn t2_inline_getprop_non_immediate_deopts() {
        use crate::bytecode::Op;
        use crate::interp::Value;
        use crate::jsval::JsVal;
        use std::cell::RefCell;
        use std::rc::Rc;
        // {x: <a string>} — x is NOT an immediate, so reading it must deopt.
        let mut m: crate::ordered::OrderedMap<String, Value> = crate::ordered::OrderedMap::new();
        m.insert("x".to_string(), Value::str("hello"));
        let shape = m.shape_header();
        let xslot = m.slot_of("x").expect("x slot") as u32;
        if shape == u32::MAX || shape == u32::MAX - 1 {
            return;
        }
        let obj: Rc<RefCell<crate::ordered::OrderedMap<String, Value>>> =
            Rc::new(RefCell::new(m));
        let code = [Op::GetProp { dst: 1, obj: 0, key_k: 0 }, Op::Ret { src: 1 }];
        let site = super::T2GetPropSite { shapes_slots: vec![(shape, xslot)], heap_result: false };
        let cfg = super::T2GetPropConfig {
            site_at: &|i| if i == 0 { Some(site.clone()) } else { None },
            shape_off: super::t2_shape_header_offset(),
            helper_addr: super::rt_getprop_slot_immediate as *const () as usize,
            heap_helper_addr: super::rt_getprop_slot_owning_store as *const () as usize,
            getidx_helper_addr: super::rt_getidx_owning_store as *const () as usize,
            setidx_helper_addr: super::rt_setidx_owning_store as *const () as usize,
        };
        let bytes = super::compile_t2lite(&code, |_k| None, Some(&cfg), super::T2StoreMode::Numeric, None).expect("compiles");
        let jf = JitFunction::install(&bytes).expect("install");
        let mut bank: [u64; 2] = [JsVal::object(&obj).bits(), JsVal::undefined().bits()];
        let mut out: u64 = 0;
        let tag = unsafe { jf.call_t2lite(bank.as_mut_ptr(), &mut out as *mut u64) };
        assert_eq!(tag, super::T2_DEOPT_RESUME, "non-immediate (string) slot must deopt");
    }

    /// MUTATION TEETH (codegen level): prove the baked slot + shape guard are
    /// LOAD-BEARING. (1) A baked WRONG slot reads a DIFFERENT field's value (so a
    /// slot corruption would produce an observably-wrong number — the oracle would
    /// redden). (2) A baked WRONG shape makes the inline guard MISS → deopt (so a
    /// stale/forged header can never read a same-object wrong slot; it falls to the
    /// correct VM). This is the proof the guard isn't decorative.
    #[cfg(target_os = "windows")]
    #[test]
    fn t2_inline_getprop_mutation_teeth() {
        use crate::bytecode::Op;
        use crate::interp::Value;
        use crate::jsval::JsVal;
        use std::cell::RefCell;
        use std::rc::Rc;
        let mut m: crate::ordered::OrderedMap<String, Value> = crate::ordered::OrderedMap::new();
        m.insert("x".to_string(), Value::Number(7.5));
        m.insert("y".to_string(), Value::Number(2.0));
        let shape = m.shape_header();
        let xslot = m.slot_of("x").unwrap() as u32;
        let yslot = m.slot_of("y").unwrap() as u32;
        if shape == u32::MAX || shape == u32::MAX - 1 {
            return; // Shaped disabled in this env — inline path can't engage.
        }
        assert_ne!(xslot, yslot, "x and y are distinct slots");
        let obj: Rc<RefCell<crate::ordered::OrderedMap<String, Value>>> =
            Rc::new(RefCell::new(m));
        let code = [Op::GetProp { dst: 1, obj: 0, key_k: 0 }, Op::Ret { src: 1 }];
        let off = super::t2_shape_header_offset();
        let helper = super::rt_getprop_slot_immediate as *const () as usize;
        let run = |shapes_slots: Vec<(u32, u32)>| -> (u64, u64) {
            let site = super::T2GetPropSite { shapes_slots, heap_result: false };
            let cfg = super::T2GetPropConfig {
                site_at: &|i| if i == 0 { Some(site.clone()) } else { None },
                shape_off: off,
                helper_addr: helper,
                heap_helper_addr: 0,
                getidx_helper_addr: 0,
                setidx_helper_addr: 0,
            };
            let bytes = super::compile_t2lite(&code, |_k| None, Some(&cfg), super::T2StoreMode::Numeric, None).unwrap();
            let jf = JitFunction::install(&bytes).unwrap();
            let mut bank: [u64; 2] = [JsVal::object(&obj).bits(), JsVal::undefined().bits()];
            let mut out: u64 = 0;
            let tag = unsafe { jf.call_t2lite(bank.as_mut_ptr(), &mut out as *mut u64) };
            (tag, out)
        };
        // Correct: (shape, xslot) reads x=7.5.
        let (tag_ok, out_ok) = run(vec![(shape, xslot)]);
        assert_eq!(tag_ok, super::T2_RETURNED);
        assert_eq!(JsVal(out_ok).as_f64(), Some(7.5), "correct slot reads x");
        // (1) WRONG SLOT baked (yslot under x's shape) → reads y=2.0, NOT 7.5.
        // A different observable number is exactly what the oracle catches.
        let (tag_bad, out_bad) = run(vec![(shape, yslot)]);
        assert_eq!(tag_bad, super::T2_RETURNED);
        assert_eq!(
            JsVal(out_bad).as_f64(),
            Some(2.0),
            "TEETH: a corrupted baked slot reads a DIFFERENT field (oracle would redden)"
        );
        assert_ne!(out_ok, out_bad, "TEETH: wrong slot ⇒ wrong value ⇒ catchable");
        // (2) WRONG SHAPE baked → inline guard misses → DEOPT (never a wrong slot
        // on the real object). The header guard is the load-bearing safety.
        let forged = if shape == 0 { 1 } else { shape - 1 };
        let (tag_miss, _out) = run(vec![(forged, xslot)]);
        assert_eq!(
            tag_miss, super::T2_DEOPT_RESUME,
            "TEETH: a forged shape must MISS the header guard → deopt, never wrong-slot"
        );
    }

    /// The audited helper in isolation: returns the boxed immediate for number/
    /// bool slots and the DEOPT sentinel for everything else + out-of-range.
    #[test]
    fn t2_rt_getprop_helper_extracts_immediates_only() {
        use crate::interp::Value;
        use crate::jsval::JsVal;
        use std::cell::RefCell;
        use std::rc::Rc;
        let mut m: crate::ordered::OrderedMap<String, Value> = crate::ordered::OrderedMap::new();
        m.insert("n".to_string(), Value::Number(3.25));
        m.insert("b".to_string(), Value::Bool(true));
        m.insert("s".to_string(), Value::str("x"));
        let (ns, bs, ss) = (
            m.slot_of("n").unwrap() as u64,
            m.slot_of("b").unwrap() as u64,
            m.slot_of("s").unwrap() as u64,
        );
        let obj: Rc<RefCell<crate::ordered::OrderedMap<String, Value>>> =
            Rc::new(RefCell::new(m));
        let p = Rc::as_ptr(&obj) as usize as u64;
        assert_eq!(super::rt_getprop_slot_immediate(p, ns), JsVal::number(3.25).bits());
        assert_eq!(super::rt_getprop_slot_immediate(p, bs), JsVal::boolean(true).bits());
        assert_eq!(super::rt_getprop_slot_immediate(p, ss), super::JV_HOLE, "string slot → deopt");
        assert_eq!(super::rt_getprop_slot_immediate(p, 999), super::JV_HOLE, "oob slot → deopt");
        assert_eq!(super::rt_getprop_slot_immediate(0, 0), super::JV_HOLE, "null ptr → deopt");
    }

    /// The GetIdx owning-store helper in isolation: every element-class edge.
    /// Builds an owning bank of 4 slots, points the helper at a 3-element array
    /// `[42, <hole>, <child-obj>]` and asserts the bank slot + status per index.
    #[test]
    fn t2_rt_getidx_helper_edges() {
        use crate::interp::Value;
        use crate::jsval::JsVal;
        use std::cell::RefCell;
        use std::rc::Rc;
        // child heap object element.
        let mut childmap: crate::ordered::OrderedMap<String, Value> =
            crate::ordered::OrderedMap::new();
        childmap.insert("tag".to_string(), Value::Number(9.0));
        let child_rc = Rc::new(RefCell::new(childmap));
        let child_ptr = Rc::as_ptr(&child_rc) as usize;
        let arr: Rc<RefCell<Vec<Value>>> = Rc::new(RefCell::new(vec![
            Value::Number(42.0),           // [0] immediate
            Value::Hole,                   // [1] hole → DEOPT
            Value::Object(child_rc.clone()), // [2] heap object
        ]));
        let arr_ptr = Rc::as_ptr(&arr) as usize as u64;
        // An owning bank (4 slots, all undefined) to receive the read.
        let mut bank: Vec<u64> = vec![JsVal::undefined().bits(); 4];
        let bp = bank.as_mut_ptr();
        // [0] immediate → RETURNED, slot holds 42.
        let st0 = super::rt_getidx_owning_store(arr_ptr, 0, bp, 0);
        assert_eq!(st0, super::T2_RETURNED);
        assert_eq!(JsVal(bank[0]).as_f64(), Some(42.0), "in-bounds immediate");
        // [2] heap object → RETURNED, slot holds the child ptr (owning-stored).
        let st2 = super::rt_getidx_owning_store(arr_ptr, 2, bp, 2);
        assert_eq!(st2, super::T2_RETURNED);
        assert_eq!((bank[2] & super::JV_PAYLOAD_MASK) as usize, child_ptr, "heap element owning-stored");
        // The bank now owns +1 of the child (over the arr's ref + child_rc).
        assert_eq!(Rc::strong_count(&child_rc), 3, "bank owns +1 of the heap element");
        // [1] hole → DEOPT, bank slot 1 UNTOUCHED (still undefined).
        let st1 = super::rt_getidx_owning_store(arr_ptr, 1, bp, 1);
        assert_eq!(st1, super::T2_DEOPT, "hole element → DEOPT");
        assert_eq!(bank[1], JsVal::undefined().bits(), "deopt leaves the dst slot untouched");
        // OOB (idx == len) → undefined, RETURNED (NOT a deopt).
        let st_oob = super::rt_getidx_owning_store(arr_ptr, 3, bp, 3);
        assert_eq!(st_oob, super::T2_RETURNED, "OOB read → undefined, RETURNED");
        assert_eq!(bank[3], JsVal::undefined().bits(), "arr[len] === undefined");
        // OOB far past end → undefined too.
        let st_oob2 = super::rt_getidx_owning_store(arr_ptr, 999, bp, 3);
        assert_eq!(st_oob2, super::T2_RETURNED);
        assert_eq!(bank[3], JsVal::undefined().bits());
        // null ptr → DEOPT.
        assert_eq!(super::rt_getidx_owning_store(0, 0, bp, 0), super::T2_DEOPT);
        // Release the bank's owned heap ref (slot 2) before the bank drops, so we
        // don't over-dec a Copy JsVal bank that has no Drop.
        unsafe { JsVal(bank[2]).rc_dec() };
        assert_eq!(Rc::strong_count(&child_rc), 2, "child back to arr+child_rc after manual dec");
        drop((arr, child_rc));
    }

    /// The SetIdx owning element-replace helper: in-bounds replace (incl. a heap
    /// element overwrite — old Rc released, no leak), and OOB → DEOPT.
    #[test]
    fn t2_rt_setidx_helper_edges() {
        use crate::interp::Value;
        use crate::jsval::JsVal;
        use std::cell::RefCell;
        use std::rc::Rc;
        // The OLD heap element we will overwrite (track its count for the leak edge).
        let mut oldmap: crate::ordered::OrderedMap<String, Value> =
            crate::ordered::OrderedMap::new();
        oldmap.insert("old".to_string(), Value::Number(1.0));
        let old_rc = Rc::new(RefCell::new(oldmap));
        let arr: Rc<RefCell<Vec<Value>>> = Rc::new(RefCell::new(vec![
            Value::Object(old_rc.clone()), // [0] heap element to be overwritten
            Value::Number(5.0),            // [1] immediate
        ]));
        let arr_ptr = Rc::as_ptr(&arr) as usize as u64;
        assert_eq!(Rc::strong_count(&old_rc), 2, "old element: arr + old_rc");
        // Overwrite [0] (heap) with the immediate 77 → old element's Rc released.
        let st0 = super::rt_setidx_owning_store(arr_ptr, 0, JsVal::number(77.0).bits());
        assert_eq!(st0, super::T2_RETURNED);
        assert!(matches!(arr.borrow()[0], Value::Number(n) if n == 77.0), "in-bounds replace");
        assert_eq!(Rc::strong_count(&old_rc), 1, "overwritten heap element's Rc released (no leak)");
        // Overwrite [1] (immediate) with a heap value (a fresh object) → array owns it.
        let mut newmap: crate::ordered::OrderedMap<String, Value> =
            crate::ordered::OrderedMap::new();
        newmap.insert("new".to_string(), Value::Number(2.0));
        let new_rc = Rc::new(RefCell::new(newmap));
        let new_ptr = Rc::as_ptr(&new_rc) as usize;
        let st1 = super::rt_setidx_owning_store(arr_ptr, 1, JsVal::object(&new_rc).bits());
        assert_eq!(st1, super::T2_RETURNED);
        assert_eq!(Rc::strong_count(&new_rc), 2, "array took its own +1 of the new heap element");
        match &arr.borrow()[1] {
            Value::Object(o) => assert_eq!(Rc::as_ptr(o) as usize, new_ptr, "new element stored by ptr"),
            other => panic!("expected the new object, got {other:?}"),
        }
        // OOB write (idx == len, and far past) → DEOPT, array UNCHANGED (len stays 2).
        assert_eq!(super::rt_setidx_owning_store(arr_ptr, 2, JsVal::number(9.0).bits()), super::T2_DEOPT);
        assert_eq!(super::rt_setidx_owning_store(arr_ptr, 50, JsVal::number(9.0).bits()), super::T2_DEOPT);
        assert_eq!(arr.borrow().len(), 2, "OOB write did NOT extend the array (deopted to VM)");
        // null ptr → DEOPT.
        assert_eq!(super::rt_setidx_owning_store(0, 0, JsVal::number(1.0).bits()), super::T2_DEOPT);
        drop((arr, old_rc, new_rc));
    }

    #[cfg(target_os = "windows")]
    #[test]
    fn t2lite_constants_match_jsval() {
        // The asm-replicated JsVal constants MUST equal jsval.rs (a drift is
        // silent corruption). Assert against the public CANONICAL_NAN + freshly
        // computed boolean/undefined/null/int32 bits.
        use crate::jsval::{JsVal, CANONICAL_NAN};
        assert_eq!(super::JV_CANONICAL_NAN, CANONICAL_NAN);
        assert_eq!(super::JV_FALSE, JsVal::boolean(false).bits());
        assert_eq!(super::JV_TRUE, JsVal::boolean(true).bits());
        assert_eq!(0xFFFE_0000_0000_0000u64, JsVal::undefined().bits());
        assert_eq!(0xFFFE_0000_0000_0001u64, JsVal::null().bits());
        // int32 lane top-16 signature: box an int32 and mask the top 16 bits.
        let i = JsVal::int32(12345);
        assert_eq!(i.bits() & super::JV_TOP16_MASK, super::JV_INT32_TOP16);
        // A plain double must NOT match the int32 signature nor the QNAN box bits.
        let d = JsVal::number(3.14);
        assert_ne!(d.bits() & super::JV_QNAN_MASK, super::JV_QNAN_BITS);
    }

    #[test]
    fn t2lite_compiles_subset_and_declines_rest() {
        use crate::bytecode::Op;
        // Subset-only: const + add + ret → compiles.
        let ok = [
            Op::LoadConst { dst: 1, k: 0 },
            Op::Add { dst: 2, lhs: 0, rhs: 1 },
            Op::Ret { src: 2 },
        ];
        assert!(
            compile_t2lite(&ok, |_k| Some(2.0), None, T2StoreMode::Numeric, None).is_some(),
            "subset-only function must compile"
        );
        // Unsupported op (Mod) → decline.
        let bad = [
            Op::Mod { dst: 2, lhs: 0, rhs: 1 },
            Op::Ret { src: 2 },
        ];
        assert!(
            compile_t2lite(&bad, |_k| Some(2.0), None, T2StoreMode::Numeric, None).is_none(),
            "a Mod op must decline (no inline SSE remainder)"
        );
        // Non-number const → decline.
        let bad2 = [
            Op::LoadConst { dst: 1, k: 0 },
            Op::Ret { src: 1 },
        ];
        assert!(
            compile_t2lite(&bad2, |_k| None, None, T2StoreMode::Numeric, None).is_none(),
            "a non-number constant must decline"
        );
        // Empty → decline.
        assert!(compile_t2lite(&[], |_k| Some(0.0), None, T2StoreMode::Numeric, None).is_none());
        // A GetProp with NO getprop config → declines (numeric-only path).
        let getprop = [
            Op::GetProp { dst: 1, obj: 0, key_k: 0 },
            Op::Ret { src: 1 },
        ];
        assert!(
            compile_t2lite(&getprop, |_k| Some(0.0), None, T2StoreMode::Numeric, None).is_none(),
            "a GetProp with no inline-site config must decline"
        );
    }

    /// P4 — the CALL codegen + deopt-soundness pre-scan at the compile level.
    #[test]
    fn t2lite_call_codegen_and_deopt_soundness() {
        use crate::bytecode::Op;
        // A dummy non-zero call config (addresses never executed in this test —
        // we only inspect compile success/decline + emitted prolog).
        let ccfg = T2CallConfig {
            call_helper_addr: 0xAAAA,
            call_fn_helper_addr: 0xBBBB,
            load_global_helper_addr: 0xCCCC,
        };
        // (1) A caller whose CALL is the LAST deopting point compiles WITH a config:
        //     LoadGlobalChecked callee ; CallValue(dst, callee, NO_THIS, args) ; Ret.
        let caller = [
            Op::LoadGlobalChecked { dst: 1, name_k: 0 },
            Op::Move { dst: 2, src: 0 }, // arg
            Op::CallValue { dst: 3, callee: 1, this_reg: u16::MAX, first_arg: 2, n_args: 1 },
            Op::Ret { src: 3 },
        ];
        let bytes = compile_t2lite(&caller, |_k| None, None, T2StoreMode::Heap { store_helper: 0xDDDD }, Some(&ccfg))
            .expect("a caller with the call as the last deopting op must compile");
        // 4-push prolog: push rbx(0x53), push rdi(0x57), push rsi(0x56), push rbp(0x55).
        assert_eq!(&bytes[0..4], &[0x53, 0x57, 0x56, 0x55], "P4 prolog must push rbx/rdi/rsi/rbp");

        // (2) Without a call config, the CallValue declines (numeric-only path).
        assert!(
            compile_t2lite(&caller, |_k| None, None, T2StoreMode::Numeric, None).is_none(),
            "a CallValue without a call config must decline"
        );

        // (3) T2 Phase 5: DEOPT-AFTER-CALL now COMPILES (was declined under P4). A
        //     deopt-capable op (Add) after a committed call is sound because the Add
        //     guard, if it fires, RESUMES the VM at the Add's bc_pc (continuing after
        //     the call) — it never re-runs the call.
        let deopt_after_call = [
            Op::LoadGlobalChecked { dst: 1, name_k: 0 },
            Op::CallValue { dst: 2, callee: 1, this_reg: u16::MAX, first_arg: 0, n_args: 0 },
            Op::Add { dst: 3, lhs: 2, rhs: 0 }, // deopt-capable AFTER a committed call
            Op::Ret { src: 3 },
        ];
        assert!(
            compile_t2lite(&deopt_after_call, |_k| None, None, T2StoreMode::Heap { store_helper: 0xDDDD }, Some(&ccfg)).is_some(),
            "P5: a deopt-capable op after a call now COMPILES (resume, no re-run)"
        );

        // (4) T2 Phase 5: a loop containing a call now COMPILES (was declined under
        //     P4). A pre-call guard firing on iteration 2 resumes mid-function — it
        //     does not re-run iteration 1's committed call.
        let loop_with_call = [
            Op::LoadConst { dst: 0, k: 0 },                                   // i = 0
            Op::Lt { dst: 1, lhs: 0, rhs: 0 },                                // i < n (deopt-capable)
            Op::JmpIfFalse { cond: 1, target: 6 },
            Op::LoadGlobalChecked { dst: 2, name_k: 0 },
            Op::CallValue { dst: 3, callee: 2, this_reg: u16::MAX, first_arg: 0, n_args: 1 },
            Op::Jmp { target: 1 },                                            // back-edge
            Op::Ret { src: 3 },
        ];
        assert!(
            compile_t2lite(&loop_with_call, |_k| Some(0.0), None, T2StoreMode::Heap { store_helper: 0xDDDD }, Some(&ccfg)).is_some(),
            "P5: a loop containing a call now COMPILES (per-guard resume)"
        );

        // (5) A try-handler-containing function STILL declines (resume assumes an
        //     empty try_stack at the resume bc_pc; try_stack reconstruction is out of
        //     scope).
        let with_try = [
            Op::TryEnter { catch_target: 3, catch_reg: 0 },
            Op::LoadGlobalChecked { dst: 1, name_k: 0 },
            Op::CallValue { dst: 2, callee: 1, this_reg: u16::MAX, first_arg: 0, n_args: 0 },
            Op::TryExit,
            Op::Ret { src: 2 },
        ];
        assert!(
            compile_t2lite(&with_try, |_k| None, None, T2StoreMode::Heap { store_helper: 0xDDDD }, Some(&ccfg)).is_none(),
            "a function with a try-handler op must STILL DECLINE (try_stack reconstruction is out of scope)"
        );

        // (6) T2 Phase 5: TWO calls now COMPILE (was declined under P4). A second
        //     call's pre-effect DEOPT resumes at the second call's bc_pc — it never
        //     re-runs the first (committed) call.
        let two_calls = [
            Op::LoadGlobalChecked { dst: 1, name_k: 0 },
            Op::CallValue { dst: 2, callee: 1, this_reg: u16::MAX, first_arg: 0, n_args: 0 },
            Op::LoadGlobalChecked { dst: 3, name_k: 0 },
            Op::CallValue { dst: 4, callee: 3, this_reg: u16::MAX, first_arg: 2, n_args: 1 },
            Op::Ret { src: 4 },
        ];
        assert!(
            compile_t2lite(&two_calls, |_k| None, None, T2StoreMode::Heap { store_helper: 0xDDDD }, Some(&ccfg)).is_some(),
            "P5: two calls now COMPILE (second-call resume, no re-run of the first)"
        );
    }
}
