//! Process spawning via raw Win32 `CreateProcessW`.
//!
//! Owns the spawned process via RAII: dropping `ChildProcess` closes
//! both the process and primary-thread handles. Wait on the child to
//! get its exit code; this is the building block the browser process
//! uses to launch a renderer with a connected named-pipe endpoint.

#![cfg(target_os = "windows")]
#![allow(non_camel_case_types, non_snake_case, clippy::upper_case_acronyms)]
#![allow(unreachable_pub, missing_debug_implementations, dead_code)]

use core::ffi::c_void;

type HANDLE = *mut c_void;
type BOOL = i32;
type DWORD = u32;
type LPCWSTR = *const u16;
type LPWSTR = *mut u16;
type LPVOID = *mut c_void;
type PSID = *mut c_void;

/// SID_IDENTIFIER_AUTHORITY — 6-byte big-endian authority value.
#[repr(C)]
struct SidIdentifierAuthority {
    value: [u8; 6],
}

/// SID_AND_ATTRIBUTES — a SID plus its attribute flags. Used inside
/// TOKEN_MANDATORY_LABEL to set the integrity level.
#[repr(C)]
struct SidAndAttributes {
    sid: PSID,
    attributes: DWORD,
}

/// TOKEN_MANDATORY_LABEL — the structure SetTokenInformation expects
/// for TokenIntegrityLevel.
#[repr(C)]
struct TokenMandatoryLabel {
    label: SidAndAttributes,
}

const INVALID_HANDLE_VALUE: HANDLE = -1isize as HANDLE;
const STARTF_USESTDHANDLES: DWORD = 0x0000_0100;
const CREATE_UNICODE_ENVIRONMENT: DWORD = 0x0000_0400;
const CREATE_NEW_PROCESS_GROUP: DWORD = 0x0000_0200;
const EXTENDED_STARTUPINFO_PRESENT: DWORD = 0x0008_0000;
const WAIT_OBJECT_0: DWORD = 0;
const WAIT_TIMEOUT: DWORD = 258;
const INFINITE: DWORD = 0xFFFF_FFFF;

const PROC_THREAD_ATTRIBUTE_MITIGATION_POLICY: usize = 0x0002_0007;
// PROC_THREAD_ATTRIBUTE_SECURITY_CAPABILITIES — attaches an
// AppContainer SECURITY_CAPABILITIES to the spawned child so it boots
// inside the AppContainer profile. Number=9, ThreadAttr=0, Input=1 →
// (9 << 16) | (0 << 18) | (1 << 17) ... encoded constant per WinBase.h.
const PROC_THREAD_ATTRIBUTE_SECURITY_CAPABILITIES: usize = 0x0002_0009;

// PROCESS_MITIGATION_POLICY enum values (read-back via
// GetProcessMitigationPolicy). Only the two we verify are declared.
const PROCESS_DEP_POLICY: i32 = 0;
const PROCESS_ASLR_POLICY: i32 = 1;

// TOKEN_INFORMATION_CLASS values used for query-back / integrity set.
const TOKEN_INTEGRITY_LEVEL: i32 = 25;
const TOKEN_IS_APP_CONTAINER: i32 = 29;

// Token access rights.
const TOKEN_QUERY: DWORD = 0x0008;
const TOKEN_DUPLICATE: DWORD = 0x0002;
const TOKEN_ASSIGN_PRIMARY: DWORD = 0x0001;
const TOKEN_ADJUST_DEFAULT: DWORD = 0x0080;

// SECURITY_IMPERSONATION_LEVEL / TOKEN_TYPE for DuplicateTokenEx.
const SECURITY_IMPERSONATION: i32 = 2;
const TOKEN_PRIMARY: i32 = 1;

// The well-known Low mandatory-integrity RID (S-1-16-4096).
const SECURITY_MANDATORY_LOW_RID: DWORD = 0x1000;

// SID_AND_ATTRIBUTES.Attributes flag carried in TOKEN_MANDATORY_LABEL.
const SE_GROUP_INTEGRITY: DWORD = 0x0000_0020;

/// Mitigation policy as the kernel actually reports it for a running
/// child. Returned by `ChildProcess::query_mitigation`; used only by
/// the verification tests. `None` for a field means the read-back FFI
/// failed (e.g. policy unsupported on this Windows build).
#[derive(Debug, Default, Clone, Copy)]
pub struct AppliedMitigation {
    pub dep_enabled: Option<bool>,
    pub aslr_bottom_up: Option<bool>,
    pub aslr_high_entropy: Option<bool>,
    pub aslr_force_relocate: Option<bool>,
}

// Public mitigation flag bits. Match the values Windows exposes via
// PROCESS_CREATION_MITIGATION_POLICY_* — see WinBase.h.
//
// All are "ALWAYS_ON" variants: setting the bit forces the protection
// on for the child process, ignoring registry / image-config overrides.
pub mod mitigation {
    pub const DEP_ENABLE: u64 = 1 << 0;
    pub const DEP_ATL_THUNK_ENABLE: u64 = 1 << 1;
    pub const SEHOP_ENABLE: u64 = 1 << 2;
    pub const FORCE_RELOCATE_IMAGES_ALWAYS_ON: u64 = 1 << 8;
    pub const HEAP_TERMINATE_ALWAYS_ON: u64 = 1 << 12;
    pub const BOTTOM_UP_ASLR_ALWAYS_ON: u64 = 1 << 16;
    pub const HIGH_ENTROPY_ASLR_ALWAYS_ON: u64 = 1 << 20;
    pub const STRICT_HANDLE_CHECKS_ALWAYS_ON: u64 = 1 << 24;
    pub const WIN32K_SYSTEM_CALL_DISABLE_ALWAYS_ON: u64 = 1 << 28;
    pub const EXTENSION_POINT_DISABLE_ALWAYS_ON: u64 = 1 << 32;
    pub const PROHIBIT_DYNAMIC_CODE_ALWAYS_ON: u64 = 1 << 36;
    pub const CONTROL_FLOW_GUARD_ALWAYS_ON: u64 = 1 << 40;
    pub const BLOCK_NON_MICROSOFT_BINARIES_ALWAYS_ON: u64 = 1 << 44;
    pub const IMAGE_LOAD_NO_REMOTE_ALWAYS_ON: u64 = 1 << 52;

    /// Chromium-style renderer hardening: every protection that
    /// makes sense for a rendering process. Matches the policy bits
    /// the production browser applies to its child renderers.
    pub const RENDERER_RECOMMENDED: u64 = DEP_ENABLE
        | SEHOP_ENABLE
        | FORCE_RELOCATE_IMAGES_ALWAYS_ON
        | HEAP_TERMINATE_ALWAYS_ON
        | BOTTOM_UP_ASLR_ALWAYS_ON
        | HIGH_ENTROPY_ASLR_ALWAYS_ON
        | STRICT_HANDLE_CHECKS_ALWAYS_ON
        | EXTENSION_POINT_DISABLE_ALWAYS_ON
        | PROHIBIT_DYNAMIC_CODE_ALWAYS_ON
        | CONTROL_FLOW_GUARD_ALWAYS_ON;
}

#[repr(C)]
struct STARTUPINFOW {
    cb: DWORD,
    lpReserved: LPWSTR,
    lpDesktop: LPWSTR,
    lpTitle: LPWSTR,
    dwX: DWORD,
    dwY: DWORD,
    dwXSize: DWORD,
    dwYSize: DWORD,
    dwXCountChars: DWORD,
    dwYCountChars: DWORD,
    dwFillAttribute: DWORD,
    dwFlags: DWORD,
    wShowWindow: u16,
    cbReserved2: u16,
    lpReserved2: *mut u8,
    hStdInput: HANDLE,
    hStdOutput: HANDLE,
    hStdError: HANDLE,
}

#[repr(C)]
struct PROCESS_INFORMATION {
    hProcess: HANDLE,
    hThread: HANDLE,
    dwProcessId: DWORD,
    dwThreadId: DWORD,
}

#[repr(C)]
struct STARTUPINFOEXW {
    StartupInfo: STARTUPINFOW,
    lpAttributeList: *mut c_void,
}

#[link(name = "kernel32")]
unsafe extern "system" {
    fn CreateProcessW(
        lpApplicationName: LPCWSTR,
        lpCommandLine: LPWSTR,
        lpProcessAttributes: *mut c_void,
        lpThreadAttributes: *mut c_void,
        bInheritHandles: BOOL,
        dwCreationFlags: DWORD,
        lpEnvironment: LPVOID,
        lpCurrentDirectory: LPCWSTR,
        lpStartupInfo: *mut STARTUPINFOW,
        lpProcessInformation: *mut PROCESS_INFORMATION,
    ) -> BOOL;

    fn CloseHandle(hObject: HANDLE) -> BOOL;
    fn GetLastError() -> DWORD;
    fn GetExitCodeProcess(hProcess: HANDLE, lpExitCode: *mut DWORD) -> BOOL;
    fn WaitForSingleObject(hHandle: HANDLE, dwMilliseconds: DWORD) -> DWORD;
    fn TerminateProcess(hProcess: HANDLE, uExitCode: DWORD) -> BOOL;

    fn InitializeProcThreadAttributeList(
        lpAttributeList: *mut c_void,
        dwAttributeCount: DWORD,
        dwFlags: DWORD,
        lpSize: *mut usize,
    ) -> BOOL;

    /// Read-only verification primitive: query the mitigation policy
    /// the kernel ACTUALLY applied to a running child. Used by the
    /// hardening verification tests to upgrade "the kernel accepted
    /// our packed word" (no ERROR_INVALID_PARAMETER) to "the bits are
    /// verifiably in effect on the child". Never mutates state.
    fn GetProcessMitigationPolicy(
        hProcess: HANDLE,
        MitigationPolicy: i32,
        lpBuffer: *mut c_void,
        dwLength: usize,
    ) -> BOOL;

    fn UpdateProcThreadAttribute(
        lpAttributeList: *mut c_void,
        dwFlags: DWORD,
        Attribute: usize,
        lpValue: *mut c_void,
        cbSize: usize,
        lpPreviousValue: *mut c_void,
        lpReturnSize: *mut usize,
    ) -> BOOL;

    fn DeleteProcThreadAttributeList(lpAttributeList: *mut c_void);

    fn GetCurrentProcess() -> HANDLE;
}

#[link(name = "advapi32")]
unsafe extern "system" {
    /// Token-bearing process creation. This is the keystone that lets
    /// the hardening ladder actually reach AppContainer / low-integrity
    /// tiers: the child inherits `hToken` as its primary token instead
    /// of a copy of the caller's. Needs SeAssignPrimaryTokenPrivilege +
    /// SeIncreaseQuotaPrivilege; absent those it fails and the ladder
    /// steps down (the no-fail-to-launch invariant is preserved by the
    /// caller).
    fn CreateProcessAsUserW(
        hToken: HANDLE,
        lpApplicationName: LPCWSTR,
        lpCommandLine: LPWSTR,
        lpProcessAttributes: *mut c_void,
        lpThreadAttributes: *mut c_void,
        bInheritHandles: BOOL,
        dwCreationFlags: DWORD,
        lpEnvironment: LPVOID,
        lpCurrentDirectory: LPCWSTR,
        lpStartupInfo: *mut STARTUPINFOW,
        lpProcessInformation: *mut PROCESS_INFORMATION,
    ) -> BOOL;

    fn OpenProcessToken(
        ProcessHandle: HANDLE,
        DesiredAccess: DWORD,
        TokenHandle: *mut HANDLE,
    ) -> BOOL;

    fn DuplicateTokenEx(
        hExistingToken: HANDLE,
        dwDesiredAccess: DWORD,
        lpTokenAttributes: *mut c_void,
        ImpersonationLevel: i32,
        TokenType: i32,
        phNewToken: *mut HANDLE,
    ) -> BOOL;

    /// Read-only token query — used to VERIFY a tier actually applied
    /// (TokenIsAppContainer / TokenIntegrityLevel read-back). Never
    /// claim a tier the kernel didn't confirm.
    fn GetTokenInformation(
        TokenHandle: HANDLE,
        TokenInformationClass: i32,
        TokenInformation: *mut c_void,
        TokenInformationLength: DWORD,
        ReturnLength: *mut DWORD,
    ) -> BOOL;

    fn SetTokenInformation(
        TokenHandle: HANDLE,
        TokenInformationClass: i32,
        TokenInformation: *mut c_void,
        TokenInformationLength: DWORD,
    ) -> BOOL;

    /// Build the Low-integrity SID (S-1-16-4096) from the mandatory-
    /// label authority + RID, without string parsing.
    fn AllocateAndInitializeSid(
        pIdentifierAuthority: *mut SidIdentifierAuthority,
        nSubAuthorityCount: u8,
        nSubAuthority0: DWORD,
        nSubAuthority1: DWORD,
        nSubAuthority2: DWORD,
        nSubAuthority3: DWORD,
        nSubAuthority4: DWORD,
        nSubAuthority5: DWORD,
        nSubAuthority6: DWORD,
        nSubAuthority7: DWORD,
        pSid: *mut PSID,
    ) -> BOOL;

    fn FreeSid(pSid: PSID) -> PSID;

    fn GetSidSubAuthorityCount(pSid: PSID) -> *mut u8;

    fn GetSidSubAuthority(pSid: PSID, nSubAuthority: DWORD) -> *mut DWORD;
}

#[derive(Debug, PartialEq, Eq)]
pub enum SpawnError {
    /// CreateProcessW returned 0; payload is GetLastError().
    Create(u32),
    /// Command line had an interior NUL.
    BadCommandLine,
    /// WaitForSingleObject failed or timed out.
    Wait(u32),
    /// Couldn't query exit code (process gone before we asked, etc.).
    ExitCode(u32),
    /// A token operation (Open/Duplicate/SetTokenInformation) failed;
    /// payload is GetLastError(). The ladder treats this as a
    /// step-down trigger, not a fatal error.
    Token(u32),
}

impl std::fmt::Display for SpawnError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Create(e) => write!(f, "CreateProcessW failed ({e})"),
            Self::BadCommandLine => f.write_str("command line contains NUL"),
            Self::Wait(e) => write!(f, "WaitForSingleObject failed ({e})"),
            Self::ExitCode(e) => write!(f, "GetExitCodeProcess failed ({e})"),
            Self::Token(e) => write!(f, "token operation failed ({e})"),
        }
    }
}

impl std::error::Error for SpawnError {}

fn to_utf16_with_nul(s: &str) -> Result<Vec<u16>, SpawnError> {
    if s.bytes().any(|b| b == 0) {
        return Err(SpawnError::BadCommandLine);
    }
    Ok(s.encode_utf16().chain(std::iter::once(0)).collect())
}

/// A spawned Windows process. Owns both handles returned by
/// `CreateProcessW`; both are closed when the struct drops.
pub struct ChildProcess {
    h_process: HANDLE,
    h_thread: HANDLE,
    process_id: u32,
}

impl std::fmt::Debug for ChildProcess {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ChildProcess")
            .field("process_id", &self.process_id)
            .finish_non_exhaustive()
    }
}

unsafe impl Send for ChildProcess {}

impl Drop for ChildProcess {
    fn drop(&mut self) {
        unsafe {
            if !self.h_thread.is_null() && self.h_thread != INVALID_HANDLE_VALUE {
                CloseHandle(self.h_thread);
            }
            if !self.h_process.is_null() && self.h_process != INVALID_HANDLE_VALUE {
                CloseHandle(self.h_process);
            }
        }
    }
}

/// Crate-internal handle accessor. The public API hides the raw
/// HANDLE since callers shouldn't manipulate process handles
/// directly; the sandbox module needs it to call
/// `AssignProcessToJobObject`.
pub(crate) fn process_handle(child: &ChildProcess) -> HANDLE {
    child.h_process
}

impl ChildProcess {
    /// Spawn a process. `command_line` is the full Win32 command line
    /// (program + args, single string — `CreateProcessW` parses it).
    /// `inherit_handles` controls whether the child inherits the
    /// parent's inheritable handles (needed for the named-pipe
    /// hand-off pattern; the named-pipe transport sets the handle
    /// inheritable so the child can `CreateFileW` it directly).
    pub fn spawn(command_line: &str, inherit_handles: bool) -> Result<Self, SpawnError> {
        let mut cmd_w = to_utf16_with_nul(command_line)?;

        let mut si: STARTUPINFOW = unsafe { core::mem::zeroed() };
        si.cb = core::mem::size_of::<STARTUPINFOW>() as DWORD;
        let mut pi: PROCESS_INFORMATION = unsafe { core::mem::zeroed() };

        let ok = unsafe {
            CreateProcessW(
                core::ptr::null(),
                cmd_w.as_mut_ptr(),
                core::ptr::null_mut(),
                core::ptr::null_mut(),
                if inherit_handles { 1 } else { 0 },
                CREATE_UNICODE_ENVIRONMENT,
                core::ptr::null_mut(),
                core::ptr::null(),
                &mut si,
                &mut pi,
            )
        };
        if ok == 0 {
            return Err(SpawnError::Create(unsafe { GetLastError() }));
        }
        Ok(Self {
            h_process: pi.hProcess,
            h_thread: pi.hThread,
            process_id: pi.dwProcessId,
        })
    }

    pub fn process_id(&self) -> u32 {
        self.process_id
    }

    /// Block until the child exits, then return its exit code.
    pub fn wait(&self) -> Result<u32, SpawnError> {
        let r = unsafe { WaitForSingleObject(self.h_process, INFINITE) };
        if r != WAIT_OBJECT_0 {
            return Err(SpawnError::Wait(r));
        }
        let mut code: DWORD = 0;
        let ok = unsafe { GetExitCodeProcess(self.h_process, &mut code) };
        if ok == 0 {
            return Err(SpawnError::ExitCode(unsafe { GetLastError() }));
        }
        Ok(code)
    }

    /// Wait up to `timeout_ms` for the child. Returns `Ok(Some(code))`
    /// if the process exited within the window, `Ok(None)` if it's
    /// still running, or `Err` on wait failure.
    pub fn try_wait_for(&self, timeout_ms: u32) -> Result<Option<u32>, SpawnError> {
        let r = unsafe { WaitForSingleObject(self.h_process, timeout_ms) };
        match r {
            WAIT_OBJECT_0 => {
                let mut code: DWORD = 0;
                let ok = unsafe { GetExitCodeProcess(self.h_process, &mut code) };
                if ok == 0 {
                    return Err(SpawnError::ExitCode(unsafe { GetLastError() }));
                }
                Ok(Some(code))
            }
            WAIT_TIMEOUT => Ok(None),
            other => Err(SpawnError::Wait(other)),
        }
    }

    /// Force-terminate the child with the given exit code. Used as a
    /// last-resort kill when a renderer becomes unresponsive.
    pub fn kill(&self, exit_code: u32) -> Result<(), SpawnError> {
        let ok = unsafe { TerminateProcess(self.h_process, exit_code) };
        if ok == 0 {
            return Err(SpawnError::Wait(unsafe { GetLastError() }));
        }
        Ok(())
    }

    /// Read back the mitigation policy the kernel ACTUALLY applied to
    /// this (still-running) child via `GetProcessMitigationPolicy`.
    /// Read-only — purely for verifying that the packed policy word we
    /// passed at spawn took effect. Each field is `None` if the
    /// corresponding read-back FFI failed.
    ///
    /// The DEP / ASLR layouts the kernel returns are the documented
    /// `PROCESS_MITIGATION_DEP_POLICY` (DWORD bitfield: bit0 = Enable)
    /// and `PROCESS_MITIGATION_ASLR_POLICY` (DWORD bitfield: bit0 =
    /// EnableBottomUpRandomization, bit1 = EnableForceRelocateImages,
    /// bit2 = EnableHighEntropy).
    pub fn query_mitigation(&self) -> AppliedMitigation {
        let mut out = AppliedMitigation::default();

        // PROCESS_MITIGATION_DEP_POLICY is a u32 bitfield + a u8.
        let mut dep: u32 = 0;
        let ok = unsafe {
            GetProcessMitigationPolicy(
                self.h_process,
                PROCESS_DEP_POLICY,
                &mut dep as *mut _ as *mut c_void,
                core::mem::size_of::<u32>(),
            )
        };
        if ok != 0 {
            out.dep_enabled = Some((dep & 0x1) != 0);
        }

        // PROCESS_MITIGATION_ASLR_POLICY is a single u32 bitfield.
        let mut aslr: u32 = 0;
        let ok = unsafe {
            GetProcessMitigationPolicy(
                self.h_process,
                PROCESS_ASLR_POLICY,
                &mut aslr as *mut _ as *mut c_void,
                core::mem::size_of::<u32>(),
            )
        };
        if ok != 0 {
            out.aslr_bottom_up = Some((aslr & 0x1) != 0);
            out.aslr_force_relocate = Some((aslr & 0x2) != 0);
            out.aslr_high_entropy = Some((aslr & 0x4) != 0);
        }

        out
    }

    /// QUERY-BACK: open the child's token and ask the kernel whether it
    /// is running inside an AppContainer (TokenIsAppContainer). Returns
    /// `Some(true)` only if the kernel confirms it, `Some(false)` if it
    /// confirms it is NOT, and `None` if the read-back FFI failed (so a
    /// failed query never masquerades as "applied"). This is the
    /// honesty keystone for Tier1: we never record AppContainer unless
    /// this returns `Some(true)`.
    pub fn query_is_app_container(&self) -> Option<bool> {
        let token = self.open_token()?;
        let mut is_ac: DWORD = 0;
        let mut ret_len: DWORD = 0;
        let ok = unsafe {
            GetTokenInformation(
                token,
                TOKEN_IS_APP_CONTAINER,
                &mut is_ac as *mut _ as *mut c_void,
                core::mem::size_of::<DWORD>() as DWORD,
                &mut ret_len,
            )
        };
        unsafe { CloseHandle(token) };
        if ok == 0 {
            return None;
        }
        Some(is_ac != 0)
    }

    /// QUERY-BACK: open the child's token, read TokenIntegrityLevel, and
    /// report whether its mandatory-integrity RID is at or below the Low
    /// level (S-1-16-4096). Returns `Some(true)` if the kernel confirms
    /// the child is Low (or lower) integrity, `Some(false)` if it is
    /// higher, `None` if the read-back failed. Tier2 records
    /// MitigationLowIntegrity only when this returns `Some(true)`.
    pub fn query_integrity_is_low(&self) -> Option<bool> {
        let token = self.open_token()?;
        // First call sizes the buffer.
        let mut needed: DWORD = 0;
        unsafe {
            GetTokenInformation(
                token,
                TOKEN_INTEGRITY_LEVEL,
                core::ptr::null_mut(),
                0,
                &mut needed,
            );
        }
        if needed == 0 {
            unsafe { CloseHandle(token) };
            return None;
        }
        let mut buf: Vec<u8> = vec![0u8; needed as usize];
        let ok = unsafe {
            GetTokenInformation(
                token,
                TOKEN_INTEGRITY_LEVEL,
                buf.as_mut_ptr() as *mut c_void,
                needed,
                &mut needed,
            )
        };
        if ok == 0 {
            unsafe { CloseHandle(token) };
            return None;
        }
        // The buffer is a TOKEN_MANDATORY_LABEL: a SID_AND_ATTRIBUTES
        // whose SID's last subauthority is the integrity RID.
        let label = buf.as_ptr() as *const TokenMandatoryLabel;
        let sid = unsafe { (*label).label.sid };
        let rid = unsafe { sid_last_subauthority(sid) };
        unsafe { CloseHandle(token) };
        rid.map(|r| r <= SECURITY_MANDATORY_LOW_RID)
    }

    /// Open this child's process token for read-back. Internal helper.
    fn open_token(&self) -> Option<HANDLE> {
        let mut token: HANDLE = core::ptr::null_mut();
        let ok = unsafe { OpenProcessToken(self.h_process, TOKEN_QUERY, &mut token) };
        if ok == 0 || token.is_null() {
            None
        } else {
            Some(token)
        }
    }

    /// Spawn a process bearing an explicit primary `token`, optionally
    /// with an AppContainer SECURITY_CAPABILITIES attribute. This is the
    /// primitive that makes the higher hardening rungs actually
    /// reachable:
    ///
    ///  * Tier1 (AppContainer): pass the AppContainer's restricted token
    ///    plus a `*mut SecurityCapabilities` carrying the OS-derived
    ///    AppContainer SID.
    ///  * Tier2 (low integrity): pass a duplicated, integrity-lowered
    ///    primary token and a null `security_capabilities`.
    ///
    /// `mitigation_flags` is applied via the same mitigation attribute
    /// `spawn_with_mitigation` uses (0 = skip). When
    /// `security_capabilities` is non-null it is added as a second
    /// PROC_THREAD attribute so the attribute list is sized for two.
    ///
    /// SAFETY: `token` must be a valid primary token handle owned by the
    /// caller for the duration of the call. `security_capabilities`, if
    /// non-null, must point to a valid `SecurityCapabilities` whose SID
    /// storage outlives the call.
    pub unsafe fn spawn_with_token(
        command_line: &str,
        mitigation_flags: u64,
        security_capabilities: *mut c_void,
        token: HANDLE,
    ) -> Result<Self, SpawnError> {
        let mut cmd_w = to_utf16_with_nul(command_line)?;

        // Count the attributes we will install: mitigation (if any) +
        // security-capabilities (if any). At least one must be present;
        // otherwise plain CreateProcessAsUserW without an attribute list
        // is fine, but the ladder always passes mitigation so we size
        // accordingly.
        let want_mitigation = mitigation_flags != 0;
        let want_caps = !security_capabilities.is_null();
        let attr_count: DWORD = u32::from(want_mitigation) + u32::from(want_caps);

        let mut attr_buf: Vec<u8> = Vec::new();
        let mut attr_list: *mut c_void = core::ptr::null_mut();
        if attr_count > 0 {
            let mut size: usize = 0;
            unsafe {
                InitializeProcThreadAttributeList(
                    core::ptr::null_mut(),
                    attr_count,
                    0,
                    &mut size,
                );
            }
            if size == 0 {
                return Err(SpawnError::Create(unsafe { GetLastError() }));
            }
            attr_buf = vec![0u8; size];
            attr_list = attr_buf.as_mut_ptr() as *mut c_void;
            let ok =
                unsafe { InitializeProcThreadAttributeList(attr_list, attr_count, 0, &mut size) };
            if ok == 0 {
                return Err(SpawnError::Create(unsafe { GetLastError() }));
            }
        }

        // Mitigation policy attribute (must outlive the call).
        let mut mitigation_value: u64 = mitigation_flags;
        if want_mitigation {
            let ok = unsafe {
                UpdateProcThreadAttribute(
                    attr_list,
                    0,
                    PROC_THREAD_ATTRIBUTE_MITIGATION_POLICY,
                    &mut mitigation_value as *mut _ as *mut c_void,
                    core::mem::size_of::<u64>(),
                    core::ptr::null_mut(),
                    core::ptr::null_mut(),
                )
            };
            if ok == 0 {
                let e = unsafe { GetLastError() };
                unsafe { DeleteProcThreadAttributeList(attr_list) };
                return Err(SpawnError::Create(e));
            }
        }

        // Security-capabilities (AppContainer) attribute.
        if want_caps {
            let ok = unsafe {
                UpdateProcThreadAttribute(
                    attr_list,
                    0,
                    PROC_THREAD_ATTRIBUTE_SECURITY_CAPABILITIES,
                    security_capabilities,
                    // SecurityCapabilities struct size from cv_sandbox:
                    // PSID + ptr + 2*DWORD on a 64-bit target.
                    core::mem::size_of::<cv_sandbox::SecurityCapabilities>(),
                    core::ptr::null_mut(),
                    core::ptr::null_mut(),
                )
            };
            if ok == 0 {
                let e = unsafe { GetLastError() };
                if !attr_list.is_null() {
                    unsafe { DeleteProcThreadAttributeList(attr_list) };
                }
                return Err(SpawnError::Create(e));
            }
        }

        let mut six: STARTUPINFOEXW = unsafe { core::mem::zeroed() };
        six.StartupInfo.cb = core::mem::size_of::<STARTUPINFOEXW>() as DWORD;
        six.lpAttributeList = attr_list;
        let mut pi: PROCESS_INFORMATION = unsafe { core::mem::zeroed() };

        let mut flags = CREATE_UNICODE_ENVIRONMENT;
        if attr_count > 0 {
            flags |= EXTENDED_STARTUPINFO_PRESENT;
        }

        let ok = unsafe {
            CreateProcessAsUserW(
                token,
                core::ptr::null(),
                cmd_w.as_mut_ptr(),
                core::ptr::null_mut(),
                core::ptr::null_mut(),
                0,
                flags,
                core::ptr::null_mut(),
                core::ptr::null(),
                &mut six as *mut _ as *mut STARTUPINFOW,
                &mut pi,
            )
        };
        if !attr_list.is_null() {
            unsafe { DeleteProcThreadAttributeList(attr_list) };
        }
        // Keep attr_buf alive until here.
        drop(attr_buf);
        if ok == 0 {
            return Err(SpawnError::Create(unsafe { GetLastError() }));
        }
        Ok(Self {
            h_process: pi.hProcess,
            h_thread: pi.hThread,
            process_id: pi.dwProcessId,
        })
    }

    /// Spawn a process with kernel-enforced mitigation policies. The
    /// `mitigation_flags` argument is a bit-mask combining
    /// `mitigation::*` constants. Renderers should pass
    /// `mitigation::RENDERER_RECOMMENDED` as a starting point.
    ///
    /// V1: only the mitigation-policy attribute is set. Future slices
    /// can extend with explicit handle-list (PROC_THREAD_ATTRIBUTE_
    /// HANDLE_LIST), parent process override, and child-process
    /// blocking via PROCESS_CREATION_CHILD_PROCESS_POLICY.
    pub fn spawn_with_mitigation(
        command_line: &str,
        inherit_handles: bool,
        mitigation_flags: u64,
    ) -> Result<Self, SpawnError> {
        let mut cmd_w = to_utf16_with_nul(command_line)?;

        // Discover the attribute list size, then allocate that many
        // bytes. Per MSDN the first call returns ERROR_INSUFFICIENT_
        // BUFFER and fills in `size`.
        let mut size: usize = 0;
        unsafe {
            InitializeProcThreadAttributeList(core::ptr::null_mut(), 1, 0, &mut size);
        }
        if size == 0 {
            return Err(SpawnError::Create(unsafe { GetLastError() }));
        }
        let mut attr_buf: Vec<u8> = vec![0u8; size];
        let attr_list = attr_buf.as_mut_ptr() as *mut c_void;

        let ok = unsafe { InitializeProcThreadAttributeList(attr_list, 1, 0, &mut size) };
        if ok == 0 {
            return Err(SpawnError::Create(unsafe { GetLastError() }));
        }

        // Mitigation policy must outlive the call.
        let mut mitigation_value: u64 = mitigation_flags;
        let ok = unsafe {
            UpdateProcThreadAttribute(
                attr_list,
                0,
                PROC_THREAD_ATTRIBUTE_MITIGATION_POLICY,
                &mut mitigation_value as *mut _ as *mut c_void,
                core::mem::size_of::<u64>(),
                core::ptr::null_mut(),
                core::ptr::null_mut(),
            )
        };
        if ok == 0 {
            unsafe { DeleteProcThreadAttributeList(attr_list) };
            return Err(SpawnError::Create(unsafe { GetLastError() }));
        }

        let mut six: STARTUPINFOEXW = unsafe { core::mem::zeroed() };
        six.StartupInfo.cb = core::mem::size_of::<STARTUPINFOEXW>() as DWORD;
        six.lpAttributeList = attr_list;
        let mut pi: PROCESS_INFORMATION = unsafe { core::mem::zeroed() };

        let ok = unsafe {
            CreateProcessW(
                core::ptr::null(),
                cmd_w.as_mut_ptr(),
                core::ptr::null_mut(),
                core::ptr::null_mut(),
                if inherit_handles { 1 } else { 0 },
                CREATE_UNICODE_ENVIRONMENT | EXTENDED_STARTUPINFO_PRESENT,
                core::ptr::null_mut(),
                core::ptr::null(),
                &mut six as *mut _ as *mut STARTUPINFOW,
                &mut pi,
            )
        };
        unsafe { DeleteProcThreadAttributeList(attr_list) };
        if ok == 0 {
            return Err(SpawnError::Create(unsafe { GetLastError() }));
        }
        Ok(Self {
            h_process: pi.hProcess,
            h_thread: pi.hThread,
            process_id: pi.dwProcessId,
        })
    }
}

/// Read the last subauthority (the RID) of a SID. Used to extract the
/// integrity RID from a TokenIntegrityLevel read-back.
///
/// SAFETY: `sid` must be a valid SID pointer (e.g. from a populated
/// TOKEN_MANDATORY_LABEL).
unsafe fn sid_last_subauthority(sid: PSID) -> Option<DWORD> {
    if sid.is_null() {
        return None;
    }
    let count_ptr = unsafe { GetSidSubAuthorityCount(sid) };
    if count_ptr.is_null() {
        return None;
    }
    let count = unsafe { *count_ptr };
    if count == 0 {
        return None;
    }
    let last = unsafe { GetSidSubAuthority(sid, DWORD::from(count - 1)) };
    if last.is_null() {
        return None;
    }
    Some(unsafe { *last })
}

/// Read the mandatory-integrity RID directly off a token handle (the
/// last subauthority of its TokenIntegrityLevel SID). Returns `None` if
/// the read-back FFI failed. Exposed crate-wide so a test can verify a
/// freshly-built token reads back as Low without needing a spawned
/// child.
pub(crate) fn token_integrity_rid(token: HANDLE) -> Option<DWORD> {
    let mut needed: DWORD = 0;
    unsafe {
        GetTokenInformation(
            token,
            TOKEN_INTEGRITY_LEVEL,
            core::ptr::null_mut(),
            0,
            &mut needed,
        );
    }
    if needed == 0 {
        return None;
    }
    let mut buf: Vec<u8> = vec![0u8; needed as usize];
    let ok = unsafe {
        GetTokenInformation(
            token,
            TOKEN_INTEGRITY_LEVEL,
            buf.as_mut_ptr() as *mut c_void,
            needed,
            &mut needed,
        )
    };
    if ok == 0 {
        return None;
    }
    let label = buf.as_ptr() as *const TokenMandatoryLabel;
    let sid = unsafe { (*label).label.sid };
    unsafe { sid_last_subauthority(sid) }
}

/// A primary token duplicated from the current process and lowered to
/// Low mandatory integrity. RAII: closes the token + frees the Low SID
/// on drop. Feed `.handle()` to `ChildProcess::spawn_with_token` for
/// the Tier2 (mitigation + low-integrity) rung.
pub struct LowIntegrityToken {
    token: HANDLE,
    low_sid: PSID,
}

impl std::fmt::Debug for LowIntegrityToken {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LowIntegrityToken").finish_non_exhaustive()
    }
}

impl Drop for LowIntegrityToken {
    fn drop(&mut self) {
        unsafe {
            if !self.token.is_null() {
                CloseHandle(self.token);
            }
            if !self.low_sid.is_null() {
                FreeSid(self.low_sid);
            }
        }
    }
}

impl LowIntegrityToken {
    /// Duplicate the current process token as a primary token, then set
    /// its mandatory integrity to Low (S-1-16-4096). Returns the
    /// owning wrapper. Requires no special privilege to *lower* one's
    /// own token integrity (you can always drop, never raise).
    pub fn from_current_process() -> Result<Self, SpawnError> {
        // Open our own token with the rights DuplicateTokenEx + the
        // subsequent SetTokenInformation need.
        let mut existing: HANDLE = core::ptr::null_mut();
        let ok = unsafe {
            OpenProcessToken(
                GetCurrentProcess(),
                TOKEN_DUPLICATE | TOKEN_QUERY | TOKEN_ASSIGN_PRIMARY | TOKEN_ADJUST_DEFAULT,
                &mut existing,
            )
        };
        if ok == 0 || existing.is_null() {
            return Err(SpawnError::Token(unsafe { GetLastError() }));
        }

        // Duplicate as a PRIMARY token (CreateProcessAsUserW needs a
        // primary token, not an impersonation token).
        let mut dup: HANDLE = core::ptr::null_mut();
        let ok = unsafe {
            DuplicateTokenEx(
                existing,
                TOKEN_DUPLICATE | TOKEN_QUERY | TOKEN_ASSIGN_PRIMARY | TOKEN_ADJUST_DEFAULT,
                core::ptr::null_mut(),
                SECURITY_IMPERSONATION,
                TOKEN_PRIMARY,
                &mut dup,
            )
        };
        unsafe { CloseHandle(existing) };
        if ok == 0 || dup.is_null() {
            return Err(SpawnError::Token(unsafe { GetLastError() }));
        }

        // Build the Low integrity SID: S-1-16-4096. The mandatory-label
        // authority is SECURITY_MANDATORY_LABEL_AUTHORITY = {0,0,0,0,0,16}.
        let mut auth = SidIdentifierAuthority {
            value: [0, 0, 0, 0, 0, 16],
        };
        let mut low_sid: PSID = core::ptr::null_mut();
        let ok = unsafe {
            AllocateAndInitializeSid(
                &mut auth,
                1,
                SECURITY_MANDATORY_LOW_RID,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                &mut low_sid,
            )
        };
        if ok == 0 || low_sid.is_null() {
            let e = unsafe { GetLastError() };
            unsafe { CloseHandle(dup) };
            return Err(SpawnError::Token(e));
        }

        // Apply it via SetTokenInformation(TokenIntegrityLevel).
        let mut label = TokenMandatoryLabel {
            label: SidAndAttributes {
                sid: low_sid,
                attributes: SE_GROUP_INTEGRITY,
            },
        };
        let ok = unsafe {
            SetTokenInformation(
                dup,
                TOKEN_INTEGRITY_LEVEL,
                &mut label as *mut _ as *mut c_void,
                core::mem::size_of::<TokenMandatoryLabel>() as DWORD,
            )
        };
        if ok == 0 {
            let e = unsafe { GetLastError() };
            unsafe {
                CloseHandle(dup);
                FreeSid(low_sid);
            }
            return Err(SpawnError::Token(e));
        }

        Ok(Self {
            token: dup,
            low_sid,
        })
    }

    /// The lowered primary token handle. Borrowed — the wrapper retains
    /// ownership; do not close it.
    pub fn handle(&self) -> HANDLE {
        self.token
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spawn_cmd_exit_zero() {
        // `cmd /c exit 0` exits cleanly. The test exercises the FFI +
        // wait + exit-code retrieval path end-to-end.
        let child = ChildProcess::spawn("cmd.exe /c exit 0", false).unwrap();
        assert!(child.process_id() > 0);
        let code = child.wait().unwrap();
        assert_eq!(code, 0);
    }

    #[test]
    fn spawn_cmd_exit_42() {
        let child = ChildProcess::spawn("cmd.exe /c exit 42", false).unwrap();
        let code = child.wait().unwrap();
        assert_eq!(code, 42);
    }

    #[test]
    fn rejects_command_line_with_nul() {
        let r = ChildProcess::spawn("cmd.exe /c \0exit", false);
        assert_eq!(r.unwrap_err(), SpawnError::BadCommandLine);
    }

    #[test]
    fn spawn_with_mitigation_renderer_preset() {
        // The renderer preset is real Windows-kernel-enforced
        // hardening — we just verify the spawn API accepts it and
        // the child runs to completion. The kernel rejects malformed
        // policy bits with ERROR_INVALID_PARAMETER, so a successful
        // exit code 0 also proves we didn't garble the bit-field.
        let child = ChildProcess::spawn_with_mitigation(
            "cmd.exe /c exit 0",
            false,
            mitigation::RENDERER_RECOMMENDED,
        )
        .unwrap();
        let code = child.wait().unwrap();
        assert_eq!(code, 0);
    }

    #[test]
    fn spawn_with_mitigation_subset_works() {
        // Subset of flags (just DEP + ASLR + heap-terminate) — should
        // also be accepted by the kernel.
        let policy = mitigation::DEP_ENABLE
            | mitigation::BOTTOM_UP_ASLR_ALWAYS_ON
            | mitigation::HEAP_TERMINATE_ALWAYS_ON;
        let child =
            ChildProcess::spawn_with_mitigation("cmd.exe /c exit 7", false, policy).unwrap();
        let code = child.wait().unwrap();
        assert_eq!(code, 7);
    }

    #[test]
    fn try_wait_returns_none_for_long_running() {
        // `ping localhost -n 3` waits ~2s without needing input
        // redirection (which `timeout` requires and which the test
        // harness disallows). 50ms try_wait should report "still
        // running".
        let child = ChildProcess::spawn("cmd.exe /c ping localhost -n 3", false).unwrap();
        let r = child.try_wait_for(50).unwrap();
        assert!(r.is_none());
        let _ = child.kill(0);
        let _ = child.wait();
    }
}
