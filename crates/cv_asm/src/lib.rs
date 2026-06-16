//! `cv_asm` — x86_64 instruction emitter.
//!
//! V1 ships the instruction set needed to JIT integer-arithmetic-heavy
//! hot paths of cv_js's bytecode VM: register-register and immediate
//! moves, arithmetic, comparisons + conditional jumps, and the
//! function-frame primitives (push/pop, ret). Output is a `Vec<u8>` of
//! machine code that the JIT trampoline then maps executable and calls
//! via a function pointer.
//!
//! Encoding follows the Intel SDM / AMD APM canonical form: REX
//! prefix → opcode → ModR/M → SIB → displacement → immediate.

#![allow(unused, missing_debug_implementations, unreachable_pub)]

/// 16 general-purpose 64-bit registers per x86_64 ABI.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum R64 {
    Rax = 0,
    Rcx = 1,
    Rdx = 2,
    Rbx = 3,
    Rsp = 4,
    Rbp = 5,
    Rsi = 6,
    Rdi = 7,
    R8 = 8,
    R9 = 9,
    R10 = 10,
    R11 = 11,
    R12 = 12,
    R13 = 13,
    R14 = 14,
    R15 = 15,
}

impl R64 {
    pub fn idx(self) -> u8 {
        self as u8
    }
    pub fn is_extended(self) -> bool {
        (self as u8) >= 8
    }
}

/// One-letter condition codes for cc-suffixed instructions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Cc {
    /// Overflow.
    O = 0,
    NoOverflow = 1,
    Below = 2,
    AboveEq = 3,
    Equal = 4,
    NotEqual = 5,
    BelowEq = 6,
    Above = 7,
    Sign = 8,
    NoSign = 9,
    Parity = 10,
    NoParity = 11,
    Less = 12,
    GreaterEq = 13,
    LessEq = 14,
    Greater = 15,
}

/// 16 SSE/AVX 128-bit registers. The JIT uses these for f64 (double) JS
/// arithmetic — `movsd`/`addsd`/… operate on the low 64 bits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Xmm {
    Xmm0 = 0,
    Xmm1 = 1,
    Xmm2 = 2,
    Xmm3 = 3,
    Xmm4 = 4,
    Xmm5 = 5,
    Xmm6 = 6,
    Xmm7 = 7,
    Xmm8 = 8,
    Xmm9 = 9,
    Xmm10 = 10,
    Xmm11 = 11,
    Xmm12 = 12,
    Xmm13 = 13,
    Xmm14 = 14,
    Xmm15 = 15,
}

impl Xmm {
    pub fn idx(self) -> u8 {
        self as u8
    }
    pub fn is_extended(self) -> bool {
        (self as u8) >= 8
    }
}

/// x86_64 instruction emitter. Append-only bytecode buffer.
#[derive(Debug, Default, Clone)]
pub struct Emitter {
    pub code: Vec<u8>,
}

impl Emitter {
    pub fn new() -> Self {
        Self::default()
    }

    fn rex(&mut self, w: bool, r: bool, x: bool, b: bool) {
        let mut byte = 0x40u8;
        if w {
            byte |= 0x08;
        }
        if r {
            byte |= 0x04;
        }
        if x {
            byte |= 0x02;
        }
        if b {
            byte |= 0x01;
        }
        if byte != 0x40 {
            self.code.push(byte);
        }
    }

    fn modrm(&mut self, mode: u8, reg: u8, rm: u8) {
        self.code
            .push(((mode & 0x3) << 6) | ((reg & 0x7) << 3) | (rm & 0x7));
    }

    /// `mov dst, src` — 64-bit reg→reg.
    pub fn mov_r64_r64(&mut self, dst: R64, src: R64) {
        self.rex(true, src.is_extended(), false, dst.is_extended());
        self.code.push(0x89);
        self.modrm(0b11, src.idx() & 0x7, dst.idx() & 0x7);
    }

    /// `mov dst, imm64` — wide immediate. Uses the 10-byte REX + B8+r form.
    pub fn mov_r64_imm64(&mut self, dst: R64, imm: i64) {
        self.rex(true, false, false, dst.is_extended());
        self.code.push(0xB8 | (dst.idx() & 0x7));
        self.code.extend_from_slice(&imm.to_le_bytes());
    }

    /// `mov dst, imm32` — sign-extended 32-bit immediate. 7-byte form.
    pub fn mov_r64_imm32(&mut self, dst: R64, imm: i32) {
        self.rex(true, false, false, dst.is_extended());
        self.code.push(0xC7);
        self.modrm(0b11, 0, dst.idx() & 0x7);
        self.code.extend_from_slice(&imm.to_le_bytes());
    }

    /// `add dst, src` — 64-bit reg→reg.
    pub fn add_r64_r64(&mut self, dst: R64, src: R64) {
        self.rex(true, src.is_extended(), false, dst.is_extended());
        self.code.push(0x01);
        self.modrm(0b11, src.idx() & 0x7, dst.idx() & 0x7);
    }

    /// `sub dst, src` — 64-bit reg→reg.
    pub fn sub_r64_r64(&mut self, dst: R64, src: R64) {
        self.rex(true, src.is_extended(), false, dst.is_extended());
        self.code.push(0x29);
        self.modrm(0b11, src.idx() & 0x7, dst.idx() & 0x7);
    }

    /// `imul dst, src` — signed multiply, two-operand form.
    pub fn imul_r64_r64(&mut self, dst: R64, src: R64) {
        self.rex(true, dst.is_extended(), false, src.is_extended());
        self.code.push(0x0F);
        self.code.push(0xAF);
        self.modrm(0b11, dst.idx() & 0x7, src.idx() & 0x7);
    }

    /// `cmp lhs, rhs` — 64-bit reg vs reg.
    pub fn cmp_r64_r64(&mut self, lhs: R64, rhs: R64) {
        self.rex(true, rhs.is_extended(), false, lhs.is_extended());
        self.code.push(0x39);
        self.modrm(0b11, rhs.idx() & 0x7, lhs.idx() & 0x7);
    }

    /// `cmp reg, imm32` — sign-extended 32-bit immediate.
    pub fn cmp_r64_imm32(&mut self, reg: R64, imm: i32) {
        self.rex(true, false, false, reg.is_extended());
        self.code.push(0x81);
        self.modrm(0b11, 7, reg.idx() & 0x7);
        self.code.extend_from_slice(&imm.to_le_bytes());
    }

    /// `sub reg, imm32` — sign-extended 32-bit immediate (e.g. `sub rsp, N`).
    pub fn sub_r64_imm32(&mut self, reg: R64, imm: i32) {
        self.rex(true, false, false, reg.is_extended());
        self.code.push(0x81);
        self.modrm(0b11, 5, reg.idx() & 0x7);
        self.code.extend_from_slice(&imm.to_le_bytes());
    }

    /// `add reg, imm32` — sign-extended 32-bit immediate (e.g. `add rsp, N`).
    pub fn add_r64_imm32(&mut self, reg: R64, imm: i32) {
        self.rex(true, false, false, reg.is_extended());
        self.code.push(0x81);
        self.modrm(0b11, 0, reg.idx() & 0x7);
        self.code.extend_from_slice(&imm.to_le_bytes());
    }

    /// `xor reg, reg` — zeroes the destination.
    pub fn xor_r64_r64(&mut self, dst: R64, src: R64) {
        self.rex(true, src.is_extended(), false, dst.is_extended());
        self.code.push(0x31);
        self.modrm(0b11, src.idx() & 0x7, dst.idx() & 0x7);
    }

    /// `push reg`.
    pub fn push_r64(&mut self, reg: R64) {
        if reg.is_extended() {
            self.code.push(0x41);
        }
        self.code.push(0x50 | (reg.idx() & 0x7));
    }

    /// `pop reg`.
    pub fn pop_r64(&mut self, reg: R64) {
        if reg.is_extended() {
            self.code.push(0x41);
        }
        self.code.push(0x58 | (reg.idx() & 0x7));
    }

    /// `ret`.
    pub fn ret(&mut self) {
        self.code.push(0xC3);
    }

    /// `nop`.
    pub fn nop(&mut self) {
        self.code.push(0x90);
    }

    /// `int3` — breakpoint, useful for guarding unreachable branches.
    pub fn int3(&mut self) {
        self.code.push(0xCC);
    }

    /// `jmp rel32` — emits a 32-bit relative jump, returns the byte
    /// offset of the displacement so the caller can patch it after
    /// emitting the target.
    pub fn jmp_rel32_placeholder(&mut self) -> usize {
        self.code.push(0xE9);
        let off = self.code.len();
        self.code.extend_from_slice(&[0u8; 4]);
        off
    }

    /// `jcc rel32` — conditional jump with 32-bit displacement.
    /// Same patch-later contract as `jmp_rel32_placeholder`.
    pub fn jcc_rel32_placeholder(&mut self, cc: Cc) -> usize {
        self.code.push(0x0F);
        self.code.push(0x80 | (cc as u8));
        let off = self.code.len();
        self.code.extend_from_slice(&[0u8; 4]);
        off
    }

    /// Patch the 32-bit displacement at `disp_off` so the jump targets
    /// the current code position.
    pub fn patch_rel32(&mut self, disp_off: usize) {
        let target = self.code.len() as i32;
        let from = (disp_off + 4) as i32;
        let rel = target - from;
        let bytes = rel.to_le_bytes();
        self.code[disp_off..disp_off + 4].copy_from_slice(&bytes);
    }

    // ------------------------------------------------------------------
    // SSE2 scalar-double (f64) layer — the JIT's number arithmetic.
    // Encoding form: [mandatory prefix] [REX] 0F [opcode] [ModR/M].
    // The mandatory prefix (F2/F3/66) precedes any REX byte.
    // ------------------------------------------------------------------

    /// Emit an SSE reg,reg instruction: `prefix 0F opcode /r` with ModR/M
    /// mode=11, reg=`reg`, rm=`rm`. `w` sets REX.W (needed by the int↔xmm
    /// converters and `movq`).
    fn sse_rr(
        &mut self,
        prefix: u8,
        w: bool,
        opcode: u8,
        reg: u8,
        reg_ext: bool,
        rm: u8,
        rm_ext: bool,
    ) {
        self.code.push(prefix);
        self.rex(w, reg_ext, false, rm_ext);
        self.code.push(0x0F);
        self.code.push(opcode);
        self.modrm(0b11, reg & 0x7, rm & 0x7);
    }

    /// Emit an SSE reg,[base+disp32] instruction (load/store double).
    fn sse_rm(
        &mut self,
        prefix: u8,
        w: bool,
        opcode: u8,
        reg: u8,
        reg_ext: bool,
        base: R64,
        disp: i32,
    ) {
        self.code.push(prefix);
        self.rex(w, reg_ext, false, base.is_extended());
        self.code.push(0x0F);
        self.code.push(opcode);
        let base_low = base.idx() & 0x7;
        // mode=10 → disp32. RSP/R12 (rm=100) require a SIB byte.
        self.modrm(0b10, reg & 0x7, base_low);
        if base_low == 0b100 {
            self.code.push((0b00 << 6) | (0b100 << 3) | base_low);
        }
        self.code.extend_from_slice(&disp.to_le_bytes());
    }

    /// `movsd dst, src` — copy low double, reg→reg. (F2 0F 10 /r)
    pub fn movsd_xmm_xmm(&mut self, dst: Xmm, src: Xmm) {
        self.sse_rr(
            0xF2,
            false,
            0x10,
            dst.idx(),
            dst.is_extended(),
            src.idx(),
            src.is_extended(),
        );
    }
    /// `movsd dst, [base+disp]` — load a double. (F2 0F 10 /r)
    pub fn movsd_xmm_mem(&mut self, dst: Xmm, base: R64, disp: i32) {
        self.sse_rm(0xF2, false, 0x10, dst.idx(), dst.is_extended(), base, disp);
    }
    /// `movsd [base+disp], src` — store a double. (F2 0F 11 /r)
    pub fn movsd_mem_xmm(&mut self, base: R64, disp: i32, src: Xmm) {
        self.sse_rm(0xF2, false, 0x11, src.idx(), src.is_extended(), base, disp);
    }
    /// `addsd dst, src`. (F2 0F 58 /r)
    pub fn addsd_xmm_xmm(&mut self, dst: Xmm, src: Xmm) {
        self.sse_rr(
            0xF2,
            false,
            0x58,
            dst.idx(),
            dst.is_extended(),
            src.idx(),
            src.is_extended(),
        );
    }
    /// `subsd dst, src`. (F2 0F 5C /r)
    pub fn subsd_xmm_xmm(&mut self, dst: Xmm, src: Xmm) {
        self.sse_rr(
            0xF2,
            false,
            0x5C,
            dst.idx(),
            dst.is_extended(),
            src.idx(),
            src.is_extended(),
        );
    }
    /// `mulsd dst, src`. (F2 0F 59 /r)
    pub fn mulsd_xmm_xmm(&mut self, dst: Xmm, src: Xmm) {
        self.sse_rr(
            0xF2,
            false,
            0x59,
            dst.idx(),
            dst.is_extended(),
            src.idx(),
            src.is_extended(),
        );
    }
    /// `divsd dst, src`. (F2 0F 5E /r)
    pub fn divsd_xmm_xmm(&mut self, dst: Xmm, src: Xmm) {
        self.sse_rr(
            0xF2,
            false,
            0x5E,
            dst.idx(),
            dst.is_extended(),
            src.idx(),
            src.is_extended(),
        );
    }
    /// `sqrtsd dst, src`. (F2 0F 51 /r)
    pub fn sqrtsd_xmm_xmm(&mut self, dst: Xmm, src: Xmm) {
        self.sse_rr(
            0xF2,
            false,
            0x51,
            dst.idx(),
            dst.is_extended(),
            src.idx(),
            src.is_extended(),
        );
    }
    /// `xorpd dst, src` — bitwise; `xorpd x, x` zeroes a register. (66 0F 57 /r)
    pub fn xorpd_xmm_xmm(&mut self, dst: Xmm, src: Xmm) {
        self.sse_rr(
            0x66,
            false,
            0x57,
            dst.idx(),
            dst.is_extended(),
            src.idx(),
            src.is_extended(),
        );
    }
    /// `movaps dst, [base+disp]` — load a 128-bit aligned packed-single value.
    /// (NP 0F 28 /r). Used to restore a callee-saved xmm6+ register whose FULL
    /// 128 bits the Win64 ABI requires to survive a call — `movsd` only moves the
    /// low 64 bits and zeroes the upper half, which violates the ABI.
    /// The memory operand MUST be 16-byte aligned.
    pub fn movaps_xmm_mem(&mut self, dst: Xmm, base: R64, disp: i32) {
        // No mandatory prefix (NP). Reuse the sse_rm encoder with prefix-less
        // path by emitting REX/0F/opcode/ModR/M directly.
        self.rex(false, dst.is_extended(), false, base.is_extended());
        self.code.push(0x0F);
        self.code.push(0x28);
        let base_low = base.idx() & 0x7;
        self.modrm(0b10, dst.idx() & 0x7, base_low);
        if base_low == 0b100 {
            self.code.push((0b00 << 6) | (0b100 << 3) | base_low);
        }
        self.code.extend_from_slice(&disp.to_le_bytes());
    }
    /// `movaps [base+disp], src` — store a 128-bit aligned packed-single value.
    /// (NP 0F 29 /r). Pairs with `movaps_xmm_mem` to preserve the FULL 128 bits
    /// of a callee-saved xmm register (Win64 ABI). 16-byte-aligned operand.
    pub fn movaps_mem_xmm(&mut self, base: R64, disp: i32, src: Xmm) {
        self.rex(false, src.is_extended(), false, base.is_extended());
        self.code.push(0x0F);
        self.code.push(0x29);
        let base_low = base.idx() & 0x7;
        self.modrm(0b10, src.idx() & 0x7, base_low);
        if base_low == 0b100 {
            self.code.push((0b00 << 6) | (0b100 << 3) | base_low);
        }
        self.code.extend_from_slice(&disp.to_le_bytes());
    }
    /// `and dst, src` — 64-bit reg&reg. (REX.W 21 /r)
    pub fn and_r64_r64(&mut self, dst: R64, src: R64) {
        self.rex(true, src.is_extended(), false, dst.is_extended());
        self.code.push(0x21);
        self.modrm(0b11, src.idx() & 0x7, dst.idx() & 0x7);
    }
    /// `or dst, src` — 64-bit reg|reg. (REX.W 09 /r)
    pub fn or_r64_r64(&mut self, dst: R64, src: R64) {
        self.rex(true, src.is_extended(), false, dst.is_extended());
        self.code.push(0x09);
        self.modrm(0b11, src.idx() & 0x7, dst.idx() & 0x7);
    }
    /// `movsxd dst, src32` — sign-extend the low 32 bits of `src` into the full
    /// 64-bit `dst`. (REX.W 63 /r). The int32 lane stores an i32 zero-extended in
    /// the low 32 bits of the JsVal payload; this recovers the signed value before
    /// `cvtsi2sd`.
    pub fn movsxd_r64_r32(&mut self, dst: R64, src: R64) {
        self.rex(true, dst.is_extended(), false, src.is_extended());
        self.code.push(0x63);
        self.modrm(0b11, dst.idx() & 0x7, src.idx() & 0x7);
    }
    /// `shl reg, imm8` — logical left shift by a small immediate. (REX.W C1 /4 ib)
    ///
    /// The int32 fast lane uses this for `x * 2^k` strength-reduction and for
    /// the NaN-box payload shifts. `imm` is masked to its low 6 bits by the CPU
    /// (a 64-bit shift count is mod 64); we forward the raw byte.
    pub fn shl_r64_imm8(&mut self, reg: R64, imm: u8) {
        self.rex(true, false, false, reg.is_extended());
        self.code.push(0xC1);
        self.modrm(0b11, 4, reg.idx() & 0x7);
        self.code.push(imm);
    }

    /// `shr reg, imm8` — logical (unsigned) right shift. (REX.W C1 /5 ib)
    pub fn shr_r64_imm8(&mut self, reg: R64, imm: u8) {
        self.rex(true, false, false, reg.is_extended());
        self.code.push(0xC1);
        self.modrm(0b11, 5, reg.idx() & 0x7);
        self.code.push(imm);
    }

    /// `sar reg, imm8` — arithmetic (signed) right shift. (REX.W C1 /7 ib)
    ///
    /// Used to recover the signed int32 value from a tagged payload and for
    /// signed `>>` strength reduction.
    pub fn sar_r64_imm8(&mut self, reg: R64, imm: u8) {
        self.rex(true, false, false, reg.is_extended());
        self.code.push(0xC1);
        self.modrm(0b11, 7, reg.idx() & 0x7);
        self.code.push(imm);
    }

    /// `neg reg` — two's-complement negate. (REX.W F7 /3)
    pub fn neg_r64(&mut self, reg: R64) {
        self.rex(true, false, false, reg.is_extended());
        self.code.push(0xF7);
        self.modrm(0b11, 3, reg.idx() & 0x7);
    }

    /// `cqo` — sign-extend RAX into RDX:RAX (the dividend setup that MUST precede
    /// a signed `idiv`). (REX.W 99). The int32 lane's `/` and `%` need RDX seeded
    /// with the sign of RAX before the divide, or `idiv` reads garbage in RDX.
    pub fn cqo(&mut self) {
        self.rex(true, false, false, false);
        self.code.push(0x99);
    }

    /// `idiv reg` — signed divide RDX:RAX by `reg`; quotient → RAX, remainder →
    /// RDX. (REX.W F7 /7). MUST be preceded by `cqo` (see above) and the caller
    /// MUST guard divisor != 0 and the INT_MIN / -1 overflow case (both #DE) — the
    /// int32 lane deopts to the VM for those rather than faulting.
    pub fn idiv_r64(&mut self, reg: R64) {
        self.rex(true, false, false, reg.is_extended());
        self.code.push(0xF7);
        self.modrm(0b11, 7, reg.idx() & 0x7);
    }

    /// `add dst, [base+disp]` — fused 64-bit reg += memory. (REX.W 03 /r). A perf
    /// form letting the optimizer keep one operand in memory (e.g. a spilled value)
    /// without a separate reload. Semantically identical to reload-then-add.
    pub fn add_r64_mem(&mut self, dst: R64, base: R64, disp: i32) {
        self.gpr_mem(0x03, dst.idx(), dst.is_extended(), base, disp);
    }

    /// `sub dst, [base+disp]` — fused 64-bit reg -= memory. (REX.W 2B /r).
    pub fn sub_r64_mem(&mut self, dst: R64, base: R64, disp: i32) {
        self.gpr_mem(0x2B, dst.idx(), dst.is_extended(), base, disp);
    }

    /// `cmp dst, [base+disp]` — fused 64-bit reg vs memory. (REX.W 3B /r). Lets the
    /// optimizer compare against a spilled operand in place.
    pub fn cmp_r64_mem(&mut self, dst: R64, base: R64, disp: i32) {
        self.gpr_mem(0x3B, dst.idx(), dst.is_extended(), base, disp);
    }

    /// `ucomisd a, b` — unordered compare, sets ZF/PF/CF. (66 0F 2E /r)
    pub fn ucomisd_xmm_xmm(&mut self, a: Xmm, b: Xmm) {
        self.sse_rr(
            0x66,
            false,
            0x2E,
            a.idx(),
            a.is_extended(),
            b.idx(),
            b.is_extended(),
        );
    }
    /// `cvtsi2sd dst_xmm, src_r64` — int64 → double. (F2 REX.W 0F 2A /r)
    pub fn cvtsi2sd_xmm_r64(&mut self, dst: Xmm, src: R64) {
        self.sse_rr(
            0xF2,
            true,
            0x2A,
            dst.idx(),
            dst.is_extended(),
            src.idx(),
            src.is_extended(),
        );
    }
    /// `cvttsd2si dst_r64, src_xmm` — double → int64 (truncate). (F2 REX.W 0F 2C /r)
    pub fn cvttsd2si_r64_xmm(&mut self, dst: R64, src: Xmm) {
        self.sse_rr(
            0xF2,
            true,
            0x2C,
            dst.idx(),
            dst.is_extended(),
            src.idx(),
            src.is_extended(),
        );
    }
    /// `movq dst_xmm, src_r64` — move 64 bits gpr→xmm (raw bits, for boxing).
    /// (66 REX.W 0F 6E /r)
    pub fn movq_xmm_r64(&mut self, dst: Xmm, src: R64) {
        self.sse_rr(
            0x66,
            true,
            0x6E,
            dst.idx(),
            dst.is_extended(),
            src.idx(),
            src.is_extended(),
        );
    }
    /// `movq dst_r64, src_xmm` — move 64 bits xmm→gpr (raw bits, for unboxing).
    /// (66 REX.W 0F 7E /r) — note the XMM is the ModR/M.reg operand.
    pub fn movq_r64_xmm(&mut self, dst: R64, src: Xmm) {
        self.sse_rr(
            0x66,
            true,
            0x7E,
            src.idx(),
            src.is_extended(),
            dst.idx(),
            dst.is_extended(),
        );
    }

    // ------------------------------------------------------------------
    // AVX VEX-encoded scalar-double (f64) layer — 3-OPERAND non-destructive
    // forms. `vop dst, a, b` computes `dst = a OP b` WITHOUT first copying a
    // into dst (the legacy SSE `op dst, b` requires `dst == a`, forcing a
    // `movsd dst, a` whenever the destination differs from the first source).
    // Eliminating that copy halves the instruction count of a register-resident
    // arithmetic loop, shortening the issue stream so the CPU's out-of-order
    // window holds more loop iterations in flight (the Stage-3 ILP lever).
    //
    // Scalar f64 AVX results are BIT-IDENTICAL to the SSE2 forms (same IEEE-754
    // operation); only the encoding differs. Caller MUST runtime-detect AVX
    // (`is_x86_feature_detected!("avx")`) and fall back to the SSE2 path, and
    // SHOULD emit `vzeroupper` before returning to legacy-SSE code to avoid the
    // AVX↔SSE transition penalty.
    //
    // Encoding: 3-byte VEX `C4 RXB.mmmmm W.vvvv.L.pp opcode modrm`.
    //   * mmmmm = 0b00001 (0F map);  pp = 0b11 (F2 mandatory prefix);  L = 0 (scalar);
    //   * W = 0;  vvvv = the SECOND source `a` (1's-complement, non-destructive);
    //   * reg = dst (VEX.R inverts the extended bit), rm = `b` (VEX.B / .X invert).
    // We always emit the 3-byte form so any of xmm0..15 works in every operand.
    // ------------------------------------------------------------------

    /// Emit a VEX 3-operand scalar reg,reg,reg op: `dst = a OP b` (b in ModR/M.rm,
    /// a in VEX.vvvv, dst in ModR/M.reg). `opcode` is the 0F-map opcode (e.g. 0x58
    /// addsd). All operands are xmm; L=0 (scalar), pp=F2, W=0.
    fn vex_rrr(&mut self, opcode: u8, dst: Xmm, a: Xmm, b: Xmm) {
        let r = !dst.is_extended(); // VEX.R is inverted (1 = not extended)
        let x = true; // VEX.X unused here → 1
        let bbit = !b.is_extended(); // VEX.B inverted
        // Byte 1: C4. Byte 2: R X B mmmmm  (R,X,B are the high 3 bits, inverted).
        let byte2 = ((r as u8) << 7) | ((x as u8) << 6) | ((bbit as u8) << 5) | 0b00001;
        // Byte 3: W vvvv L pp.  vvvv = ~a (4 bits), inverted.
        let vvvv = (!a.idx()) & 0x0F;
        let byte3 = (0u8 << 7) | (vvvv << 3) | (0u8 << 2) | 0b11;
        self.code.push(0xC4);
        self.code.push(byte2);
        self.code.push(byte3);
        self.code.push(opcode);
        self.modrm(0b11, dst.idx() & 0x7, b.idx() & 0x7);
    }

    /// `vaddsd dst, a, b` → `dst = a + b`. (VEX.NDS.LIG.F2.0F.WIG 58 /r)
    pub fn vaddsd(&mut self, dst: Xmm, a: Xmm, b: Xmm) {
        self.vex_rrr(0x58, dst, a, b);
    }
    /// `vsubsd dst, a, b` → `dst = a - b`. (5C /r)
    pub fn vsubsd(&mut self, dst: Xmm, a: Xmm, b: Xmm) {
        self.vex_rrr(0x5C, dst, a, b);
    }
    /// `vmulsd dst, a, b` → `dst = a * b`. (59 /r)
    pub fn vmulsd(&mut self, dst: Xmm, a: Xmm, b: Xmm) {
        self.vex_rrr(0x59, dst, a, b);
    }
    /// `vdivsd dst, a, b` → `dst = a / b`. (5E /r)
    pub fn vdivsd(&mut self, dst: Xmm, a: Xmm, b: Xmm) {
        self.vex_rrr(0x5E, dst, a, b);
    }
    /// `vmovsd dst, src` — scalar-double reg→reg copy via the merge form
    /// `vmovsd dst, src, src`. (VEX.NDS.LIG.F2.0F.WIG 10 /r). Bit-identical to
    /// `movsd dst, src` for the low lane.
    pub fn vmovsd_xmm_xmm(&mut self, dst: Xmm, src: Xmm) {
        self.vex_rrr(0x10, dst, src, src);
    }
    /// `vmovq dst_xmm, src_r64` — VEX move 64 bits gpr→xmm (raw bits, for boxing a
    /// constant). (VEX.128.66.0F.W1 6E /r). The VEX form of `movq_xmm_r64`, so a
    /// constant load inside an otherwise-VEX body introduces no legacy-SSE op.
    /// reg = dst_xmm, rm = src_r64, vvvv unused (1111), pp=66, W=1.
    pub fn vmovq_xmm_r64(&mut self, dst: Xmm, src: R64) {
        let r = !dst.is_extended();
        let x = true;
        let bbit = !src.is_extended();
        let byte2 = ((r as u8) << 7) | ((x as u8) << 6) | ((bbit as u8) << 5) | 0b00001;
        // W=1, vvvv=1111 (unused), L=0, pp=01 (66).
        let byte3 = (1u8 << 7) | (0b1111 << 3) | (0u8 << 2) | 0b01;
        self.code.push(0xC4);
        self.code.push(byte2);
        self.code.push(byte3);
        self.code.push(0x6E);
        self.modrm(0b11, dst.idx() & 0x7, src.idx() & 0x7);
    }

    /// `vucomisd a, b` — VEX unordered compare (sets ZF/PF/CF). Two-operand:
    /// reg=a, rm=b, vvvv unused (1111). pp=66 (not F2). (VEX.LIG.66.0F.WIG 2E /r)
    /// VEX form so an all-VEX loop body has NO legacy-SSE op (no per-iteration
    /// AVX↔SSE transition).
    pub fn vucomisd(&mut self, a: Xmm, b: Xmm) {
        let r = !a.is_extended();
        let x = true;
        let bbit = !b.is_extended();
        let byte2 = ((r as u8) << 7) | ((x as u8) << 6) | ((bbit as u8) << 5) | 0b00001;
        // vvvv unused = 1111; pp = 0b01 (66 prefix); L = 0; W = 0.
        let byte3 = (0u8 << 7) | (0b1111 << 3) | (0u8 << 2) | 0b01;
        self.code.push(0xC4);
        self.code.push(byte2);
        self.code.push(byte3);
        self.code.push(0x2E);
        self.modrm(0b11, a.idx() & 0x7, b.idx() & 0x7);
    }

    /// `vzeroupper` — zero the upper 128 bits of all ymm registers, clearing the
    /// AVX↔SSE transition penalty before returning to legacy-SSE code.
    /// (VEX.128.0F.WIG 77) — 2-byte VEX `C5 F8 77`.
    pub fn vzeroupper(&mut self) {
        self.code.push(0xC5);
        self.code.push(0xF8);
        self.code.push(0x77);
    }

    // ------------------------------------------------------------------
    // GPR memory operands — base+disp32 and base+index*scale+disp32.
    //
    // These mirror `sse_rm`'s SIB logic but for plain integer mov/lea: no
    // SSE mandatory prefix, no 0F escape (the opcodes are 1-byte). The
    // encoding form is: REX → opcode → ModR/M → [SIB] → disp32.
    //
    // We always use ModR/M mode=10 (disp32) so the RBP/R13 base quirk
    // (rm=101 with mode=00 means RIP-relative, not [rbp]) never bites —
    // exactly as `sse_rm` does. A SIB byte is required whenever the low 3
    // bits of the base are 0b100 (RSP/R12), because rm=100 in ModR/M is
    // the "SIB follows" escape.
    // ------------------------------------------------------------------

    /// Shared base+disp32 GPR memory encoder. `opcode` is the 1-byte
    /// primary opcode (e.g. 0x8B for load, 0x89 for store, 0x8D for lea).
    /// `reg`/`reg_ext` is the ModR/M.reg operand (the GPR side); `base` is
    /// the memory base register; `disp` the 32-bit displacement.
    fn gpr_mem(&mut self, opcode: u8, reg: u8, reg_ext: bool, base: R64, disp: i32) {
        self.gpr_mem_w(true, opcode, reg, reg_ext, base, disp);
    }

    /// Like `gpr_mem` but with an explicit REX.W. `w=false` emits a 32-bit
    /// operand-size memory access (used by `mov_r32_mem`, which reads exactly 4
    /// bytes — required for the T2 shape-id header `u32` so the load never
    /// over-reads past the object's tail).
    fn gpr_mem_w(&mut self, w: bool, opcode: u8, reg: u8, reg_ext: bool, base: R64, disp: i32) {
        self.rex(w, reg_ext, false, base.is_extended());
        self.code.push(opcode);
        let base_low = base.idx() & 0x7;
        // mode=10 → disp32. RSP/R12 (rm=100) require a SIB byte.
        self.modrm(0b10, reg & 0x7, base_low);
        if base_low == 0b100 {
            self.code.push((0b00 << 6) | (0b100 << 3) | base_low);
        }
        self.code.extend_from_slice(&disp.to_le_bytes());
    }

    /// `mov dst, [base+disp]` — 64-bit load. (REX.W 8B /r)
    pub fn mov_r64_mem(&mut self, dst: R64, base: R64, disp: i32) {
        self.gpr_mem(0x8B, dst.idx(), dst.is_extended(), base, disp);
    }

    /// `mov [base+disp], src` — 64-bit store. (REX.W 89 /r)
    pub fn mov_mem_r64(&mut self, base: R64, disp: i32, src: R64) {
        self.gpr_mem(0x89, src.idx(), src.is_extended(), base, disp);
    }

    /// `mov dst32, [base+disp]` — 32-bit load (no REX.W; 8B /r). Reads EXACTLY 4
    /// bytes and zero-extends into the 64-bit `dst` (x86-64 zeroes the upper 32
    /// bits on any 32-bit GPR write). Used by the T2 JIT to read the `u32` shape-
    /// id header inline without over-reading the object's tail.
    pub fn mov_r32_mem(&mut self, dst: R64, base: R64, disp: i32) {
        self.gpr_mem_w(false, 0x8B, dst.idx(), dst.is_extended(), base, disp);
    }

    /// `lea dst, [base+disp]` — load effective address. (REX.W 8D /r)
    /// Same operand form as `mov_r64_mem`.
    pub fn lea_r64_mem(&mut self, dst: R64, base: R64, disp: i32) {
        self.gpr_mem(0x8D, dst.idx(), dst.is_extended(), base, disp);
    }

    /// Shared base+index*scale+disp32 GPR memory encoder (full SIB form).
    /// `scale` must be 1/2/4/8. `index` must not be RSP (rm=100 in the SIB
    /// index field means "no index"), which is asserted.
    fn gpr_mem_indexed(
        &mut self,
        opcode: u8,
        reg: u8,
        reg_ext: bool,
        base: R64,
        index: R64,
        scale: u8,
        disp: i32,
    ) {
        assert!(
            index != R64::Rsp,
            "RSP (index=0b100) cannot be used as a SIB index register"
        );
        let scale_bits = match scale {
            1 => 0b00,
            2 => 0b01,
            4 => 0b10,
            8 => 0b11,
            _ => panic!("scale must be 1, 2, 4 or 8, got {scale}"),
        };
        // REX.R = reg_ext, REX.X = index_ext, REX.B = base_ext.
        self.rex(true, reg_ext, index.is_extended(), base.is_extended());
        self.code.push(opcode);
        // ModR/M: mode=10 (disp32), reg, rm=100 (SIB follows).
        self.modrm(0b10, reg & 0x7, 0b100);
        // SIB: scale | index | base.
        self.code
            .push((scale_bits << 6) | ((index.idx() & 0x7) << 3) | (base.idx() & 0x7));
        self.code.extend_from_slice(&disp.to_le_bytes());
    }

    /// `mov dst, [base+index*scale+disp]` — indexed 64-bit load. (REX.W 8B /r + SIB)
    pub fn mov_r64_mem_indexed(
        &mut self,
        dst: R64,
        base: R64,
        index: R64,
        scale: u8,
        disp: i32,
    ) {
        self.gpr_mem_indexed(0x8B, dst.idx(), dst.is_extended(), base, index, scale, disp);
    }

    /// `mov [base+index*scale+disp], src` — indexed 64-bit store. (REX.W 89 /r + SIB)
    pub fn mov_mem_indexed_r64(
        &mut self,
        base: R64,
        index: R64,
        scale: u8,
        disp: i32,
        src: R64,
    ) {
        self.gpr_mem_indexed(0x89, src.idx(), src.is_extended(), base, index, scale, disp);
    }

    // ------------------------------------------------------------------
    // Calls — indirect (register) and direct (rel32, patch-later).
    // ------------------------------------------------------------------

    /// `call reg` — indirect near call. (FF /2). No REX.W (call is 64-bit
    /// in long mode by default); REX.B only when the register is extended.
    pub fn call_r64(&mut self, reg: R64) {
        if reg.is_extended() {
            self.code.push(0x41); // REX.B
        }
        self.code.push(0xFF);
        self.modrm(0b11, 2, reg.idx() & 0x7);
    }

    /// `call rel32` — direct near call. Emits 0xE8 + a 4-byte patchable
    /// displacement; returns the byte offset of the displacement. Same
    /// patch contract as `jmp_rel32_placeholder` (use `patch_rel32` /
    /// `patch_rel32_to`).
    pub fn call_rel32_placeholder(&mut self) -> usize {
        self.code.push(0xE8);
        let off = self.code.len();
        self.code.extend_from_slice(&[0u8; 4]);
        off
    }

    // ------------------------------------------------------------------
    // setcc + movzx — materialize a comparison RESULT into a register.
    // ------------------------------------------------------------------

    /// `setcc reg8` — set the low byte of `reg` to 0/1 by condition. (0F 90+cc /0)
    ///
    /// We ALWAYS emit a REX byte (even the bare 0x40) so that the low-byte
    /// access is unambiguous: without REX, encoding ModR/M.rm = 4/5/6/7
    /// selects AH/CH/DH/BH; with any REX prefix present it selects
    /// SPL/BPL/SIL/DIL — which is what `R64` indices 4..7 mean. This makes
    /// `setcc` correct for every R64 including RSP/RBP/RSI/RDI and r8–r15.
    pub fn setcc(&mut self, cc: Cc, reg: R64) {
        // Force a REX prefix unconditionally (REX.B for extended regs).
        let mut rex = 0x40u8;
        if reg.is_extended() {
            rex |= 0x01; // REX.B
        }
        self.code.push(rex);
        self.code.push(0x0F);
        self.code.push(0x90 | (cc as u8));
        self.modrm(0b11, 0, reg.idx() & 0x7);
    }

    /// `movzx dst, src8` — zero-extend the low byte of `src` into the full
    /// 64-bit `dst`. (REX.W 0F B6 /r). Pairs with `setcc` to produce a
    /// clean 0/1 in a 64-bit register.
    pub fn movzx_r64_r8(&mut self, dst: R64, src: R64) {
        self.rex(true, dst.is_extended(), false, src.is_extended());
        self.code.push(0x0F);
        self.code.push(0xB6);
        self.modrm(0b11, dst.idx() & 0x7, src.idx() & 0x7);
    }

    /// `test a, b` — bitwise AND, sets flags, discards result. (REX.W 85 /r)
    /// `test reg, reg` is the canonical "is this register zero?" check.
    pub fn test_r64_r64(&mut self, a: R64, b: R64) {
        self.rex(true, b.is_extended(), false, a.is_extended());
        self.code.push(0x85);
        self.modrm(0b11, b.idx() & 0x7, a.idx() & 0x7);
    }

    // ------------------------------------------------------------------
    // Spill / reload helpers — thin wrappers over the base+disp GPR
    // memory ops, indexed by a stack-frame slot. The basis for a real
    // register allocator: slot N lives at [frame_base + N*8].
    // ------------------------------------------------------------------

    /// Spill `reg` to stack slot `slot` (at `[frame_base + slot*8]`).
    pub fn spill_r64(&mut self, frame_base: R64, slot: usize, reg: R64) {
        self.mov_mem_r64(frame_base, (slot * 8) as i32, reg);
    }

    /// Reload `reg` from stack slot `slot` (at `[frame_base + slot*8]`).
    pub fn reload_r64(&mut self, reg: R64, frame_base: R64, slot: usize) {
        self.mov_r64_mem(reg, frame_base, (slot * 8) as i32);
    }

    /// Patch the 32-bit displacement at `disp_off` to target an ARBITRARY code
    /// offset (e.g. a loop back-edge target recorded earlier), not just the
    /// current position.
    pub fn patch_rel32_to(&mut self, disp_off: usize, target: usize) {
        let from = (disp_off + 4) as i32;
        let rel = target as i32 - from;
        self.code[disp_off..disp_off + 4].copy_from_slice(&rel.to_le_bytes());
    }

    /// Current code-buffer length — used by labels.
    pub fn here(&self) -> usize {
        self.code.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_bytes(emitter: &Emitter, expected: &[u8]) {
        assert_eq!(emitter.code, expected, "got: {:02x?}", emitter.code);
    }

    #[test]
    fn mov_rax_rcx_encodes_correctly() {
        let mut e = Emitter::new();
        e.mov_r64_r64(R64::Rax, R64::Rcx);
        // REX.W=1 + 89 /r where reg=rcx, rm=rax → 48 89 C8.
        assert_bytes(&e, &[0x48, 0x89, 0xC8]);
    }

    #[test]
    fn mov_imm32_loads_value() {
        let mut e = Emitter::new();
        e.mov_r64_imm32(R64::Rax, 42);
        // 48 C7 C0 2A 00 00 00.
        assert_bytes(&e, &[0x48, 0xC7, 0xC0, 0x2A, 0x00, 0x00, 0x00]);
    }

    #[test]
    fn add_sub_encodes_correctly() {
        let mut e = Emitter::new();
        e.add_r64_r64(R64::Rax, R64::Rcx);
        e.sub_r64_r64(R64::Rax, R64::Rdx);
        // ADD: 48 01 C8.  SUB: 48 29 D0.
        assert_bytes(&e, &[0x48, 0x01, 0xC8, 0x48, 0x29, 0xD0]);
    }

    #[test]
    fn extended_regs_use_rex_b() {
        let mut e = Emitter::new();
        e.mov_r64_r64(R64::R8, R64::Rax);
        // REX.W=1 + REX.B=1 → 49. Opcode 89. ModR/M reg=Rax(0), rm=R8(0+ext).
        // 49 89 C0.
        assert_bytes(&e, &[0x49, 0x89, 0xC0]);
    }

    #[test]
    fn ret_is_c3() {
        let mut e = Emitter::new();
        e.ret();
        assert_bytes(&e, &[0xC3]);
    }

    #[test]
    fn sse_arith_reg_reg_encodes() {
        let mut e = Emitter::new();
        e.movsd_xmm_xmm(Xmm::Xmm0, Xmm::Xmm1); // F2 0F 10 C1
        e.addsd_xmm_xmm(Xmm::Xmm0, Xmm::Xmm1); // F2 0F 58 C1
        e.subsd_xmm_xmm(Xmm::Xmm0, Xmm::Xmm1); // F2 0F 5C C1
        e.mulsd_xmm_xmm(Xmm::Xmm2, Xmm::Xmm3); // F2 0F 59 D3
        e.divsd_xmm_xmm(Xmm::Xmm0, Xmm::Xmm1); // F2 0F 5E C1
        assert_bytes(
            &e,
            &[
                0xF2, 0x0F, 0x10, 0xC1, 0xF2, 0x0F, 0x58, 0xC1, 0xF2, 0x0F, 0x5C, 0xC1, 0xF2, 0x0F,
                0x59, 0xD3, 0xF2, 0x0F, 0x5E, 0xC1,
            ],
        );
    }

    #[test]
    fn sse_int_double_converts_encode() {
        let mut e = Emitter::new();
        e.cvtsi2sd_xmm_r64(Xmm::Xmm0, R64::Rax); // F2 48 0F 2A C0
        e.cvttsd2si_r64_xmm(R64::Rax, Xmm::Xmm0); // F2 48 0F 2C C0
        assert_bytes(
            &e,
            &[0xF2, 0x48, 0x0F, 0x2A, 0xC0, 0xF2, 0x48, 0x0F, 0x2C, 0xC0],
        );
    }

    #[test]
    fn movq_both_directions_encode() {
        let mut e = Emitter::new();
        e.movq_xmm_r64(Xmm::Xmm0, R64::Rax); // 66 48 0F 6E C0
        e.movq_r64_xmm(R64::Rax, Xmm::Xmm0); // 66 48 0F 7E C0
        assert_bytes(
            &e,
            &[0x66, 0x48, 0x0F, 0x6E, 0xC0, 0x66, 0x48, 0x0F, 0x7E, 0xC0],
        );
    }

    #[test]
    fn ucomisd_and_xorpd_encode() {
        let mut e = Emitter::new();
        e.ucomisd_xmm_xmm(Xmm::Xmm0, Xmm::Xmm1); // 66 0F 2E C1
        e.xorpd_xmm_xmm(Xmm::Xmm0, Xmm::Xmm0); // 66 0F 57 C0
        assert_bytes(&e, &[0x66, 0x0F, 0x2E, 0xC1, 0x66, 0x0F, 0x57, 0xC0]);
    }

    #[test]
    fn sse_extended_regs_use_rex() {
        let mut e = Emitter::new();
        e.addsd_xmm_xmm(Xmm::Xmm8, Xmm::Xmm9); // F2 45 0F 58 C1
        assert_bytes(&e, &[0xF2, 0x45, 0x0F, 0x58, 0xC1]);
    }

    #[test]
    fn movsd_mem_load_store_encode() {
        let mut e = Emitter::new();
        e.movsd_xmm_mem(Xmm::Xmm1, R64::Rax, 16); // F2 0F 10 88 10 00 00 00
        e.movsd_mem_xmm(R64::Rsp, 8, Xmm::Xmm0); // F2 0F 11 84 24 08 00 00 00 (SIB for RSP)
        assert_bytes(
            &e,
            &[
                0xF2, 0x0F, 0x10, 0x88, 0x10, 0x00, 0x00, 0x00, 0xF2, 0x0F, 0x11, 0x84, 0x24, 0x08,
                0x00, 0x00, 0x00,
            ],
        );
    }

    #[test]
    fn jmp_patch_lands_on_target() {
        let mut e = Emitter::new();
        let off = e.jmp_rel32_placeholder();
        e.nop();
        e.nop();
        e.patch_rel32(off);
        e.ret();
        // After the patch, the 4 displacement bytes should equal 2
        // (two NOPs between the jump and the target).
        let disp = i32::from_le_bytes(e.code[off..off + 4].try_into().unwrap());
        assert_eq!(disp, 2);
    }

    // ------------------------------------------------------------------
    // M4.1 additions — GPR memory, calls, setcc/movzx, lea, test, jo,
    // spill/reload. Each expected-byte sequence is the Intel SDM
    // canonical encoding for the named instruction.
    // ------------------------------------------------------------------

    #[test]
    fn mov_r64_mem_load_rax_rbp_minus_8() {
        // mov rax, [rbp-8]
        // REX.W=48, 8B /r, ModR/M mode=10 reg=000 rm=101 (rbp) = 85,
        // no SIB (rbp low!=100), disp32 = -8.
        let mut e = Emitter::new();
        e.mov_r64_mem(R64::Rax, R64::Rbp, -8);
        assert_bytes(&e, &[0x48, 0x8B, 0x85, 0xF8, 0xFF, 0xFF, 0xFF]);
    }

    #[test]
    fn mov_r32_mem_load_ecx_rax_plus_12() {
        // mov ecx, [rax+12] — 32-bit load, NO REX.W (operand-size 32).
        // No REX byte (ecx, rax both low), 8B /r, ModR/M mode=10 reg=001 (ecx)
        // rm=000 (rax) = 0x88, no SIB (rax low!=100), disp32 = 12.
        let mut e = Emitter::new();
        e.mov_r32_mem(R64::Rcx, R64::Rax, 12);
        assert_bytes(&e, &[0x8B, 0x88, 0x0C, 0x00, 0x00, 0x00]);
    }

    #[test]
    fn mov_r32_mem_extended_base_still_no_w() {
        // mov ecx, [r12+12] — base r12 low==100 → SIB; REX.B (no W) = 41.
        // 41, 8B, ModR/M mode=10 reg=001 rm=100 = 0x8C, SIB 24, disp32=12.
        let mut e = Emitter::new();
        e.mov_r32_mem(R64::Rcx, R64::R12, 12);
        assert_bytes(&e, &[0x41, 0x8B, 0x8C, 0x24, 0x0C, 0x00, 0x00, 0x00]);
    }

    #[test]
    fn mov_mem_r64_store_rsp_plus_16_rbx() {
        // mov [rsp+16], rbx — exercises the SIB path (base rsp low==100).
        // REX.W=48, 89 /r, ModR/M mode=10 reg=011 (rbx) rm=100 = 9C,
        // SIB = (00<<6)|(100<<3)|100 = 24, disp32 = 16.
        let mut e = Emitter::new();
        e.mov_mem_r64(R64::Rsp, 16, R64::Rbx);
        assert_bytes(&e, &[0x48, 0x89, 0x9C, 0x24, 0x10, 0x00, 0x00, 0x00]);
    }

    #[test]
    fn mov_r64_mem_extended_r12_from_r13() {
        // mov r12, [r13+0] — extended REX.R (dst r12) + REX.B (base r13).
        // REX.WRB = 4D, 8B /r, ModR/M mode=10 reg=100 (r12&7) rm=101 (r13&7) = A5,
        // no SIB (r13 low==101!=100), disp32 = 0.
        let mut e = Emitter::new();
        e.mov_r64_mem(R64::R12, R64::R13, 0);
        assert_bytes(&e, &[0x4D, 0x8B, 0xA5, 0x00, 0x00, 0x00, 0x00]);
    }

    #[test]
    fn mov_mem_r64_store_to_r12_base_uses_sib() {
        // mov [r12+8], rax — r12 low bits == 100 → SIB required even when
        // extended. REX.W + REX.B = 49, 89 /r, ModR/M mode=10 reg=000 rm=100 = 84,
        // SIB = (00<<6)|(100<<3)|100 = 24, disp32 = 8.
        let mut e = Emitter::new();
        e.mov_mem_r64(R64::R12, 8, R64::Rax);
        assert_bytes(&e, &[0x49, 0x89, 0x84, 0x24, 0x08, 0x00, 0x00, 0x00]);
    }

    #[test]
    fn mov_r64_mem_indexed_rax_rbx_rcx_8() {
        // mov rax, [rbx+rcx*8+0]
        // REX.W = 48, 8B /r, ModR/M mode=10 reg=000 rm=100 (SIB) = 84,
        // SIB = scale(11)|index(rcx=001)|base(rbx=011) = CB, disp32 = 0.
        let mut e = Emitter::new();
        e.mov_r64_mem_indexed(R64::Rax, R64::Rbx, R64::Rcx, 8, 0);
        assert_bytes(&e, &[0x48, 0x8B, 0x84, 0xCB, 0x00, 0x00, 0x00, 0x00]);
    }

    #[test]
    fn mov_mem_indexed_store_with_extended_index() {
        // mov [rax+r9*4+32], rdx — extended index r9 sets REX.X.
        // REX.W + REX.X = 4A, 89 /r, ModR/M mode=10 reg=010 (rdx) rm=100 = 94,
        // SIB = scale(10)|index(r9&7=001)|base(rax=000) = (10<<6)|(001<<3)|000 = 88,
        // disp32 = 32.
        let mut e = Emitter::new();
        e.mov_mem_indexed_r64(R64::Rax, R64::R9, 4, 32, R64::Rdx);
        assert_bytes(&e, &[0x4A, 0x89, 0x94, 0x88, 0x20, 0x00, 0x00, 0x00]);
    }

    #[test]
    #[should_panic(expected = "RSP")]
    fn mov_indexed_rejects_rsp_index() {
        let mut e = Emitter::new();
        e.mov_r64_mem_indexed(R64::Rax, R64::Rbx, R64::Rsp, 8, 0);
    }

    #[test]
    fn call_indirect_register() {
        // call rax → FF D0 (FF /2, ModR/M 11 010 000).
        let mut e = Emitter::new();
        e.call_r64(R64::Rax);
        assert_bytes(&e, &[0xFF, 0xD0]);
    }

    #[test]
    fn call_indirect_extended_register() {
        // call r8 → 41 FF D0 (REX.B for r8, no REX.W).
        let mut e = Emitter::new();
        e.call_r64(R64::R8);
        assert_bytes(&e, &[0x41, 0xFF, 0xD0]);
    }

    #[test]
    fn call_rel32_placeholder_patches() {
        // call rel32 then a target — disp should land on the target.
        let mut e = Emitter::new();
        let off = e.call_rel32_placeholder();
        assert_eq!(e.code[0], 0xE8);
        e.nop();
        e.nop();
        e.nop();
        e.patch_rel32(off);
        let disp = i32::from_le_bytes(e.code[off..off + 4].try_into().unwrap());
        assert_eq!(disp, 3);
    }

    #[test]
    fn setcc_and_movzx_yield_zero_or_one() {
        // sete al → forced REX (40), 0F 94, ModR/M 11 000 000 (C0).
        // movzx rax, al → REX.W (48), 0F B6, ModR/M 11 000 000 (C0).
        let mut e = Emitter::new();
        e.setcc(Cc::Equal, R64::Rax);
        e.movzx_r64_r8(R64::Rax, R64::Rax);
        assert_bytes(
            &e,
            &[0x40, 0x0F, 0x94, 0xC0, 0x48, 0x0F, 0xB6, 0xC0],
        );
    }

    #[test]
    fn setcc_low_byte_reg_forces_rex() {
        // setne sil → SIL needs the REX prefix (40) to select the low byte
        // (without REX, rm=110 would be DH). 40 0F 95 ModR/M 11 000 110 (C6).
        let mut e = Emitter::new();
        e.setcc(Cc::NotEqual, R64::Rsi);
        assert_bytes(&e, &[0x40, 0x0F, 0x95, 0xC6]);
    }

    #[test]
    fn setcc_extended_reg_uses_rex_b() {
        // setl r10b → REX.B (41), 0F 9C, ModR/M 11 000 010 (C2).
        let mut e = Emitter::new();
        e.setcc(Cc::Less, R64::R10);
        assert_bytes(&e, &[0x41, 0x0F, 0x9C, 0xC2]);
    }

    #[test]
    fn lea_r64_mem_encodes() {
        // lea rax, [rbp-8] → REX.W (48), 8D /r, ModR/M mode=10 reg=000 rm=101 (85),
        // disp32 = -8.
        let mut e = Emitter::new();
        e.lea_r64_mem(R64::Rax, R64::Rbp, -8);
        assert_bytes(&e, &[0x48, 0x8D, 0x85, 0xF8, 0xFF, 0xFF, 0xFF]);
    }

    #[test]
    fn test_r64_r64_encodes() {
        // test rax, rax → REX.W (48), 85 /r, ModR/M 11 000 000 (C0).
        let mut e = Emitter::new();
        e.test_r64_r64(R64::Rax, R64::Rax);
        assert_bytes(&e, &[0x48, 0x85, 0xC0]);
    }

    #[test]
    fn jo_overflow_jump_encodes() {
        // jcc(Cc::O) → 0F 80 + 4-byte placeholder. Cc::O == 0 so 0x80|0 = 0x80.
        let mut e = Emitter::new();
        let off = e.jcc_rel32_placeholder(Cc::O);
        assert_eq!(off, 2);
        assert_bytes(&e, &[0x0F, 0x80, 0x00, 0x00, 0x00, 0x00]);
    }

    #[test]
    fn add_then_jo_overflow_deopt_usage() {
        // SMI-overflow deopt pattern: add rax, rcx ; jo <deopt>.
        // add rax, rcx = 48 01 C8 ; then 0F 80 + patchable disp.
        let mut e = Emitter::new();
        e.add_r64_r64(R64::Rax, R64::Rcx);
        let off = e.jcc_rel32_placeholder(Cc::O);
        e.nop(); // fall-through (no overflow)
        e.patch_rel32(off); // deopt landing pad here
        e.int3();
        // add: 48 01 C8 ; jo: 0F 80 <disp=1> ; nop: 90 ; int3: CC.
        assert_bytes(
            &e,
            &[0x48, 0x01, 0xC8, 0x0F, 0x80, 0x01, 0x00, 0x00, 0x00, 0x90, 0xCC],
        );
    }

    #[test]
    fn movaps_load_store_preserve_full_128() {
        // movaps xmm6, [rsp+16] → NP 0F 28 /r, ModR/M mode=10 reg=110 rm=100 (SIB),
        // SIB = 24, disp32 = 16. No REX (xmm6 not extended, NP).
        let mut e = Emitter::new();
        e.movaps_xmm_mem(Xmm::Xmm6, R64::Rsp, 16);
        assert_bytes(&e, &[0x0F, 0x28, 0xB4, 0x24, 0x10, 0x00, 0x00, 0x00]);
        // movaps [rsp+0], xmm7 → 0F 29 /r, ModR/M mode=10 reg=111 rm=100, SIB 24.
        let mut e2 = Emitter::new();
        e2.movaps_mem_xmm(R64::Rsp, 0, Xmm::Xmm7);
        assert_bytes(&e2, &[0x0F, 0x29, 0xBC, 0x24, 0x00, 0x00, 0x00, 0x00]);
        // Extended xmm uses REX.R: movaps xmm8, [rsp+0] → 44 0F 28 ...
        let mut e3 = Emitter::new();
        e3.movaps_xmm_mem(Xmm::Xmm8, R64::Rsp, 0);
        assert_bytes(&e3, &[0x44, 0x0F, 0x28, 0x84, 0x24, 0x00, 0x00, 0x00, 0x00]);
    }

    #[test]
    fn and_and_movsxd_encode() {
        // and rax, rcx → REX.W (48), 21 /r, ModR/M 11 001 000 (C8).
        let mut e = Emitter::new();
        e.and_r64_r64(R64::Rax, R64::Rcx);
        assert_bytes(&e, &[0x48, 0x21, 0xC8]);
        // movsxd rax, ecx → REX.W (48), 63 /r, ModR/M 11 000 001 (C1).
        let mut e2 = Emitter::new();
        e2.movsxd_r64_r32(R64::Rax, R64::Rcx);
        assert_bytes(&e2, &[0x48, 0x63, 0xC1]);
        // or rax, rcx → REX.W (48), 09 /r, ModR/M 11 001 000 (C8).
        let mut e3 = Emitter::new();
        e3.or_r64_r64(R64::Rax, R64::Rcx);
        assert_bytes(&e3, &[0x48, 0x09, 0xC8]);
    }

    // ------------------------------------------------------------------
    // B1 (T3) additions — int32-lane shifts/neg/idiv + fused reg,mem
    // arithmetic. Expected bytes are the Intel SDM canonical encodings.
    // ------------------------------------------------------------------

    #[test]
    fn shifts_by_imm8_encode() {
        // shl rax, 3 → REX.W (48), C1 /4, ModR/M 11 100 000 (E0), ib=03.
        let mut e = Emitter::new();
        e.shl_r64_imm8(R64::Rax, 3);
        assert_bytes(&e, &[0x48, 0xC1, 0xE0, 0x03]);
        // shr rcx, 1 → 48 C1 /5 (ModR/M 11 101 001 = E9), ib=01.
        let mut e2 = Emitter::new();
        e2.shr_r64_imm8(R64::Rcx, 1);
        assert_bytes(&e2, &[0x48, 0xC1, 0xE9, 0x01]);
        // sar rdx, 31 → 48 C1 /7 (ModR/M 11 111 010 = FA), ib=1F.
        let mut e3 = Emitter::new();
        e3.sar_r64_imm8(R64::Rdx, 31);
        assert_bytes(&e3, &[0x48, 0xC1, 0xFA, 0x1F]);
        // shl r9, 4 → REX.WB (49), C1 /4 (ModR/M 11 100 001 = E1), ib=04.
        let mut e4 = Emitter::new();
        e4.shl_r64_imm8(R64::R9, 4);
        assert_bytes(&e4, &[0x49, 0xC1, 0xE1, 0x04]);
    }

    #[test]
    fn neg_and_idiv_encode() {
        // neg rax → REX.W (48), F7 /3 (ModR/M 11 011 000 = D8).
        let mut e = Emitter::new();
        e.neg_r64(R64::Rax);
        assert_bytes(&e, &[0x48, 0xF7, 0xD8]);
        // cqo → REX.W (48), 99.
        let mut e2 = Emitter::new();
        e2.cqo();
        assert_bytes(&e2, &[0x48, 0x99]);
        // idiv rcx → REX.W (48), F7 /7 (ModR/M 11 111 001 = F9).
        let mut e3 = Emitter::new();
        e3.idiv_r64(R64::Rcx);
        assert_bytes(&e3, &[0x48, 0xF7, 0xF9]);
        // idiv r10 → REX.WB (49), F7 /7 (ModR/M 11 111 010 = FA).
        let mut e4 = Emitter::new();
        e4.idiv_r64(R64::R10);
        assert_bytes(&e4, &[0x49, 0xF7, 0xFA]);
    }

    #[test]
    fn fused_reg_mem_arith_encode() {
        // add rax, [rbp-8] → REX.W (48), 03 /r, ModR/M mode=10 reg=000 rm=101 (85),
        // disp32 = -8.
        let mut e = Emitter::new();
        e.add_r64_mem(R64::Rax, R64::Rbp, -8);
        assert_bytes(&e, &[0x48, 0x03, 0x85, 0xF8, 0xFF, 0xFF, 0xFF]);
        // sub rcx, [rsp+16] → REX.W (48), 2B /r, ModR/M mode=10 reg=001 rm=100 (8C),
        // SIB 24 (rsp base), disp32 = 16.
        let mut e2 = Emitter::new();
        e2.sub_r64_mem(R64::Rcx, R64::Rsp, 16);
        assert_bytes(&e2, &[0x48, 0x2B, 0x8C, 0x24, 0x10, 0x00, 0x00, 0x00]);
        // cmp rdx, [rax+0] → REX.W (48), 3B /r, ModR/M mode=10 reg=010 rm=000 (90),
        // disp32 = 0.
        let mut e3 = Emitter::new();
        e3.cmp_r64_mem(R64::Rdx, R64::Rax, 0);
        assert_bytes(&e3, &[0x48, 0x3B, 0x90, 0x00, 0x00, 0x00, 0x00]);
    }

    #[test]
    fn vex_scalar_double_3operand_encode() {
        // vaddsd xmm0, xmm1, xmm2 → 3-byte VEX C4 E1 73 58 C2.
        //   byte2 = R̄X̄B̄ mmmmm = 1 1 1 00001 = 0xE1 (all low regs → inverted bits set).
        //   byte3 = W v̄v̄v̄v̄ L pp = 0 1110 0 11 = 0x73 (vvvv = ~1 = 1110).
        //   opcode 58, modrm 11 000 010 = C2.
        let mut e = Emitter::new();
        e.vaddsd(Xmm::Xmm0, Xmm::Xmm1, Xmm::Xmm2);
        assert_bytes(&e, &[0xC4, 0xE1, 0x73, 0x58, 0xC2]);
        // vmulsd xmm12, xmm11, xmm3 → extended dst+a, low b.
        //   R̄=0 (xmm12 ext), X̄=1, B̄=1 (xmm3 low) → byte2 = 0 1 1 00001 = 0x61.
        //   vvvv = ~11 = ~1011 = 0100; byte3 = 0 0100 0 11 = 0x23. opcode 59.
        //   modrm = 11 (12&7=100) (3) = 11 100 011 = 0xE3.
        let mut e2 = Emitter::new();
        e2.vmulsd(Xmm::Xmm12, Xmm::Xmm11, Xmm::Xmm3);
        assert_bytes(&e2, &[0xC4, 0x61, 0x23, 0x59, 0xE3]);
        // vdivsd xmm12, xmm12, xmm13 → dst=a=xmm12 (ext), b=xmm13 (ext).
        //   R̄=0, X̄=1, B̄=0 (xmm13 ext) → byte2 = 0 1 0 00001 = 0x41.
        //   vvvv = ~12 = ~1100 = 0011; byte3 = 0 0011 0 11 = 0x1B. opcode 5E.
        //   modrm = 11 (12&7=100) (13&7=101) = 11 100 101 = 0xE5.
        let mut e3 = Emitter::new();
        e3.vdivsd(Xmm::Xmm12, Xmm::Xmm12, Xmm::Xmm13);
        assert_bytes(&e3, &[0xC4, 0x41, 0x1B, 0x5E, 0xE5]);
        // vzeroupper → C5 F8 77.
        let mut e4 = Emitter::new();
        e4.vzeroupper();
        assert_bytes(&e4, &[0xC5, 0xF8, 0x77]);
        // vucomisd xmm0, xmm2 → C4 E1 79 2E C2 (pp=66, vvvv unused=1111).
        let mut e5 = Emitter::new();
        e5.vucomisd(Xmm::Xmm0, Xmm::Xmm2);
        assert_bytes(&e5, &[0xC4, 0xE1, 0x79, 0x2E, 0xC2]);
        // vmovq xmm3, rax → W=1 pp=66. R̄=1(xmm3 low) X̄=1 B̄=1(rax low) → byte2=0xE1.
        //   byte3 = W1 vvvv=1111 L0 pp=01 = 1_1111_0_01 = 0xF9. opcode 6E.
        //   modrm = 11 (xmm3=011) (rax=000) = 11 011 000 = 0xD8.
        let mut e7 = Emitter::new();
        e7.vmovq_xmm_r64(Xmm::Xmm3, R64::Rax);
        assert_bytes(&e7, &[0xC4, 0xE1, 0xF9, 0x6E, 0xD8]);
        // vmovsd xmm0, xmm12 (copy via merge form vmovsd dst,src,src; opcode 10).
        //   dst=xmm0 (low), a=b=xmm12 (ext). R̄=1, X̄=1, B̄=0 (xmm12 ext) → 0xC1.
        //   vvvv=~12=0011; byte3 = 0 0011 0 11 = 0x1B. modrm 11 000 (12&7=100)=0xC4.
        let mut e6 = Emitter::new();
        e6.vmovsd_xmm_xmm(Xmm::Xmm0, Xmm::Xmm12);
        assert_bytes(&e6, &[0xC4, 0xC1, 0x1B, 0x10, 0xC4]);
    }

    #[test]
    fn spill_reload_use_frame_slots() {
        // spill rbx to slot 2 → mov [rbp+16], rbx.
        //   REX.W (48), 89 /r, ModR/M mode=10 reg=011 rm=101 (9D), disp32 = 16.
        // reload rax from slot 1 → mov rax, [rbp+8].
        //   REX.W (48), 8B /r, ModR/M mode=10 reg=000 rm=101 (85), disp32 = 8.
        let mut e = Emitter::new();
        e.spill_r64(R64::Rbp, 2, R64::Rbx);
        e.reload_r64(R64::Rax, R64::Rbp, 1);
        assert_bytes(
            &e,
            &[
                0x48, 0x89, 0x9D, 0x10, 0x00, 0x00, 0x00, 0x48, 0x8B, 0x85, 0x08, 0x00, 0x00, 0x00,
            ],
        );
    }
}
