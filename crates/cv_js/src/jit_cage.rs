//! Code cage — a process-wide executable region for the optimizing JIT (T3).
//!
//! WHY: the per-function `JitFunction::install` (jit.rs) does one
//! `VirtualAlloc(PAGE_READWRITE)` → copy → `VirtualProtect(PAGE_EXECUTE_READ)`
//! per function, so every compiled function lives at an arbitrary, far-apart
//! address. An *optimizing* JIT wants two things that per-page install can't
//! give:
//!
//!   1. **Relative inter-function calls** (`call rel32`): two functions emitted
//!      into the SAME reserved region are within ±2GB of each other, so a direct
//!      `E8 rel32` call from A into B is encodable — no `mov rax, imm64; call rax`
//!      detour. This is the groundwork B5 (persisted native code relocation)
//!      builds on.
//!   2. A single contiguous RX arena that a future adaptive-IC patcher could
//!      address. (Adaptive IN-PLACE patching is explicitly DEFERRED this phase —
//!      see the module doc-comment in MEMORY; we never rewrite installed code, so
//!      W^X stays strict.)
//!
//! DESIGN (commit-on-demand, strict W^X):
//!   * Reserve one large region up front (`MEM_RESERVE`, no commit, no backing
//!     pages — cheap address space only).
//!   * A bump allocator hands out page-aligned sub-regions. Each function is
//!     placed at a fresh page boundary so a page is owned by exactly one
//!     function — this is what makes W^X provably correct: we NEVER write a page
//!     that another (already-executable) function lives in.
//!   * `install_into_cage(code)`: commit the function's pages as
//!     `PAGE_READWRITE`, copy the bytes, then `VirtualProtect` those exact pages
//!     to `PAGE_EXECUTE_READ` + `FlushInstructionCache`. A page is therefore
//!     RW *xor* RX, never both — the W^X invariant.
//!
//! GATING: only used when `code_cage_enabled()` (CV_CODE_CAGE, default OFF). The
//! per-page `JitFunction::install` stays the DEFAULT and the FALLBACK: if the
//! cage is full or disabled, install falls back to a private page. A cage bug can
//! therefore only ever cost us the cage path, never correctness of the default.

#![cfg(target_os = "windows")]

use std::sync::{Mutex, OnceLock};

// Reuse the kernel32 FFI bindings + constants from the jit module so there is a
// single declaration of each symbol in the crate.
use crate::jit::win;

/// Total reserved address space for the cage. 64 MiB of *address space* (not
/// committed memory) — pages are backed only as functions are installed.
const CAGE_RESERVE_BYTES: usize = 64 * 1024 * 1024;

/// Windows allocation granularity / page size we align commits to. The page size
/// is 4 KiB on x64; aligning each function to a page keeps the W^X guarantee
/// per-function with no cross-function page sharing.
const PAGE_SIZE: usize = 4096;

#[inline]
fn round_up(n: usize, to: usize) -> usize {
    (n + to - 1) & !(to - 1)
}

/// Whether the unified code cage is enabled. DEFAULT-OFF (opt IN with
/// `CV_CODE_CAGE=1`), mirroring the `t2_enabled` / `gc_enabled` OnceLock-cached
/// escape-hatch discipline. When OFF, JIT functions install into private pages
/// exactly as before (byte-identical behavior). This is a *where the code lives*
/// switch, never a *what the code does* switch.
pub fn code_cage_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("CV_CODE_CAGE").as_deref() == Ok("1"))
}

/// A reserved executable arena with a bump allocator. Thread-safe via an internal
/// mutex (JIT compilation can happen on the renderer thread; the cage is
/// process-wide so installs from any thread are serialized).
struct CodeCage {
    /// Base of the reserved region (page-aligned by VirtualAlloc).
    base: *mut u8,
    /// Total reserved bytes.
    reserved: usize,
    /// Bump cursor — offset of the next free byte (always page-aligned because we
    /// round every allocation up to a page).
    cursor: usize,
}

// SAFETY: all access goes through the `CAGE` Mutex, and the underlying pages are
// either RW-while-filling (single installer holds the lock) or RX-forever.
unsafe impl Send for CodeCage {}

impl CodeCage {
    /// Reserve the arena. Returns None if the OS won't give us the address space
    /// (the caller then falls back to per-page install — never fatal).
    fn reserve() -> Option<Self> {
        unsafe {
            let base = win::VirtualAlloc(
                core::ptr::null_mut(),
                CAGE_RESERVE_BYTES,
                win::MEM_RESERVE,
                win::PAGE_READWRITE,
            );
            if base.is_null() {
                return None;
            }
            Some(CodeCage {
                base: base as *mut u8,
                reserved: CAGE_RESERVE_BYTES,
                cursor: 0,
            })
        }
    }

    /// Install `code` into the cage. Two-phase W^X: commit RW → copy → protect RX.
    /// Returns `(rx_ptr, cage_offset, committed_len)` on success. `None` if the
    /// cage is exhausted or a syscall fails (caller falls back to a private page).
    ///
    /// INVARIANT (W^X): the committed pages are RW only while THIS call (holding
    /// the cage lock) is copying into them; before returning we flip them to RX and
    /// never write them again. No page is ever simultaneously writable and
    /// executable.
    fn install(&mut self, code: &[u8]) -> Option<(*mut u8, usize, usize)> {
        if code.is_empty() {
            return None;
        }
        // Page-align the allocation so each function owns whole pages.
        let need = round_up(code.len(), PAGE_SIZE);
        if self.cursor + need > self.reserved {
            return None; // cage full → fall back to per-page install.
        }
        let off = self.cursor;
        let region = unsafe { self.base.add(off) } as *mut core::ffi::c_void;
        unsafe {
            // Phase 1: commit the pages as READWRITE and copy the code in.
            let committed = win::VirtualAlloc(
                region,
                need,
                win::MEM_COMMIT,
                win::PAGE_READWRITE,
            );
            if committed.is_null() {
                return None;
            }
            core::ptr::copy_nonoverlapping(code.as_ptr(), committed as *mut u8, code.len());
            // Phase 2: flip the committed pages to EXECUTE_READ. After this the
            // pages are RX and we never write them again (W^X holds).
            let mut old: win::DWORD = 0;
            let ok = win::VirtualProtect(
                committed,
                need,
                win::PAGE_EXECUTE_READ,
                &raw mut old,
            );
            if ok == 0 {
                // Leave the pages committed-but-RW; we simply don't advance the
                // cursor over a usable region. Decommit to reclaim the address
                // space cleanly.
                win::VirtualFree(region, need, win::MEM_DECOMMIT);
                return None;
            }
            win::FlushInstructionCache(win::GetCurrentProcess(), committed, need);
            self.cursor += need;
            Some((committed as *mut u8, off, need))
        }
    }
}

/// Lazily-reserved process-wide cage. `None` inside if reservation failed (the
/// `OnceLock<Option<Mutex<…>>>` distinguishes "tried and failed" from "not yet
/// tried", so we never thrash VirtualAlloc on a denied reservation).
static CAGE: OnceLock<Option<Mutex<CodeCage>>> = OnceLock::new();

fn cage() -> Option<&'static Mutex<CodeCage>> {
    CAGE.get_or_init(|| CodeCage::reserve().map(Mutex::new))
        .as_ref()
}

/// A function installed into the code cage. Unlike `JitFunction`, dropping this
/// does NOT free the underlying pages — cage pages live for the process (the cage
/// is a bump allocator; reclamation would require a moving compactor, deferred).
/// The `rx_ptr` is the callable entry point.
#[derive(Debug)]
pub struct CageFunction {
    /// Executable entry point (RX).
    pub rx_ptr: *mut u8,
    /// Byte offset of this function within the cage (for relative addressing and
    /// for B5 relocation bookkeeping).
    pub cage_off: usize,
    /// Number of bytes of `code` actually installed (not the page-rounded extent).
    pub code_len: usize,
}

// SAFETY: the pages are RX-forever after install; the pointer is stable for the
// life of the process. Moving the handle between threads is sound.
unsafe impl Send for CageFunction {}
unsafe impl Sync for CageFunction {}

/// Install `code` into the process-wide cage. Returns None if the cage is
/// disabled, unavailable, or full — the caller (JIT install) then uses the
/// per-page path. This is the public entry the JIT install seam calls.
pub fn install_into_cage(code: &[u8]) -> Option<CageFunction> {
    if !code_cage_enabled() {
        return None;
    }
    let cage = cage()?;
    let mut guard = cage.lock().ok()?;
    let (rx_ptr, cage_off, _committed) = guard.install(code)?;
    Some(CageFunction {
        rx_ptr,
        cage_off,
        code_len: code.len(),
    })
}

/// The base address of the reserved cage region, if reserved. Used by the cage
/// round-trip test to compute `call rel32` displacements between two installed
/// functions, and by B5 relocation bookkeeping. None if the cage isn't up.
pub fn cage_base() -> Option<*mut u8> {
    cage().and_then(|m| m.lock().ok().map(|g| g.base))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Force the cage on for a test scope regardless of env. We reserve the cage
    /// directly (bypassing the env gate) so the tests are deterministic and don't
    /// depend on process env. We use a dedicated local cage to avoid coupling test
    /// ordering through the process-wide `CAGE` OnceLock.
    fn local_cage() -> CodeCage {
        CodeCage::reserve().expect("cage reserve")
    }

    /// VirtualQuery to assert protection of a region. Returns the `Protect` field.
    fn query_protect(addr: *const u8) -> u32 {
        #[repr(C)]
        struct MemoryBasicInformation {
            base_address: *mut core::ffi::c_void,
            allocation_base: *mut core::ffi::c_void,
            allocation_protect: u32,
            partition_id: u16,
            _pad: u16,
            region_size: usize,
            state: u32,
            protect: u32,
            type_: u32,
        }
        #[link(name = "kernel32")]
        unsafe extern "system" {
            fn VirtualQuery(
                lpAddress: *const core::ffi::c_void,
                lpBuffer: *mut MemoryBasicInformation,
                dwLength: usize,
            ) -> usize;
        }
        unsafe {
            let mut mbi: MemoryBasicInformation = core::mem::zeroed();
            let n = VirtualQuery(
                addr as *const core::ffi::c_void,
                &mut mbi,
                core::mem::size_of::<MemoryBasicInformation>(),
            );
            assert!(n != 0, "VirtualQuery failed");
            mbi.protect
        }
    }

    /// A leaf function `() -> u64` returning `value`: mov rax, imm32 ; ret.
    fn make_const_fn(value: i32) -> Vec<u8> {
        let mut e = cv_asm::Emitter::new();
        e.mov_r64_imm32(cv_asm::R64::Rax, value);
        e.ret();
        e.code
    }

    #[test]
    fn cage_install_and_call_roundtrip() {
        let mut cage = local_cage();
        let code = make_const_fn(0x1234);
        let (rx, _off, _len) = cage.install(&code).expect("install");
        let f: extern "system" fn() -> u64 = unsafe { core::mem::transmute(rx) };
        assert_eq!(f(), 0x1234);
    }

    #[test]
    fn cage_pages_are_rx_never_rwx() {
        // W^X assertion: after install the function's page is PAGE_EXECUTE_READ
        // (0x20), NEVER PAGE_EXECUTE_READWRITE (0x40).
        const PAGE_EXECUTE_READ: u32 = 0x20;
        const PAGE_EXECUTE_READWRITE: u32 = 0x40;
        let mut cage = local_cage();
        let code = make_const_fn(7);
        let (rx, _off, len) = cage.install(&code).expect("install");
        // Check every committed page of the function.
        let mut p = rx as usize;
        let end = p + len;
        while p < end {
            let prot = query_protect(p as *const u8);
            assert_eq!(
                prot, PAGE_EXECUTE_READ,
                "page at {p:#x} prot={prot:#x} expected RX"
            );
            assert_ne!(
                prot, PAGE_EXECUTE_READWRITE,
                "W^X VIOLATION: page at {p:#x} is RWX"
            );
            p += PAGE_SIZE;
        }
    }

    #[test]
    fn cage_relative_call_between_two_functions() {
        // THE relative-addressing proof: install B (returns 41), then install A
        // which `call rel32`s into B and adds 1 → 42. The rel32 is only encodable
        // because A and B share the cage's reserved region (within ±2GB). This is
        // the inter-function relative call the per-page path cannot do.
        let mut cage = local_cage();

        // B: mov rax, 41 ; ret.
        let b_code = make_const_fn(41);
        let (b_rx, b_off, _b_len) = cage.install(&b_code).expect("install B");

        // A: push rbp ; sub rsp,32 (shadow space, Win64 ABI) ; call rel32 B ;
        //    add rax, 1 ; add rsp,32 ; pop rbp ; ret.
        // The call rel32 displacement is (b_off) - (a_off + site + 4): both are
        // cage offsets, so the difference is the true PC-relative distance.
        let mut e = cv_asm::Emitter::new();
        e.push_r64(cv_asm::R64::Rbp);
        e.mov_r64_r64(cv_asm::R64::Rbp, cv_asm::R64::Rsp);
        e.sub_r64_imm32(cv_asm::R64::Rsp, 32); // shadow space, keeps 16B alignment
        let site = e.call_rel32_placeholder();
        e.add_r64_imm32(cv_asm::R64::Rax, 1);
        e.add_r64_imm32(cv_asm::R64::Rsp, 32);
        e.pop_r64(cv_asm::R64::Rbp);
        e.ret();

        // A will be installed at the current cursor; compute its cage offset NOW.
        let a_off = cage.cursor;
        // Patch the call site: target_cage_off - (a_off + site + 4).
        let from = (a_off + site + 4) as isize;
        let to = b_off as isize;
        let rel = (to - from) as i32;
        e.code[site..site + 4].copy_from_slice(&rel.to_le_bytes());

        let (a_rx, a_off_actual, _a_len) = cage.install(&e.code).expect("install A");
        assert_eq!(a_off_actual, a_off, "A landed where we predicted");
        // Sanity: A really does sit after B in the same region.
        assert!(a_rx as usize > b_rx as usize);

        let f: extern "system" fn() -> u64 = unsafe { core::mem::transmute(a_rx) };
        assert_eq!(f(), 42, "A() called B() across the cage and added 1");
    }

    #[test]
    fn cage_full_returns_none_not_panic() {
        // A tiny reservation exhausts after one page-rounded function; the next
        // install must return None (caller falls back to per-page), never panic /
        // OOB. We test the bump-allocator boundary logic directly.
        let mut cage = local_cage();
        // Drive the cursor to near the end, then ask for more than remains.
        cage.cursor = cage.reserved - PAGE_SIZE + 1; // < 1 page free
        let code = make_const_fn(1); // rounds up to 1 page
        assert!(
            cage.install(&code).is_none(),
            "install past the reserve must decline, not overflow"
        );
    }
}
