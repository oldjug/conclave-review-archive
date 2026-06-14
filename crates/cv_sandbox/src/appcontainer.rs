//! AppContainer SID + integrity-level + restricted-token construction.
//!
//! Wraps the Win32 calls that brand a child process as a sandboxed
//! AppContainer: DeriveCapabilitySidsFromName + CreateAppContainerProfile
//! to produce the AppContainer SID, then CreateRestrictedToken to drop
//! to Low integrity. The broker passes the resulting token to
//! CreateProcessAsUserW when spawning the renderer.

#![allow(non_snake_case, non_camel_case_types)]

use core::ffi::c_void;
use core::ptr;

type DWORD = u32;
type LPCWSTR = *const u16;
type PSID = *mut c_void;
type HANDLE = *mut c_void;
type BOOL = i32;
type HRESULT = i32;

#[link(name = "userenv")]
unsafe extern "system" {
    fn CreateAppContainerProfile(
        pszAppContainerName: LPCWSTR,
        pszDisplayName: LPCWSTR,
        pszDescription: LPCWSTR,
        pCapabilities: *mut c_void,
        dwCapabilityCount: DWORD,
        ppSidAppContainerSid: *mut PSID,
    ) -> HRESULT;
    fn DeriveAppContainerSidFromAppContainerName(
        pszAppContainerName: LPCWSTR,
        ppSidAppContainerSid: *mut PSID,
    ) -> HRESULT;
}

#[link(name = "advapi32")]
unsafe extern "system" {
    fn CreateRestrictedToken(
        ExistingTokenHandle: HANDLE,
        Flags: DWORD,
        DisableSidCount: DWORD,
        SidsToDisable: *mut c_void,
        DeletePrivilegeCount: DWORD,
        PrivilegesToDelete: *mut c_void,
        RestrictedSidCount: DWORD,
        SidsToRestrict: *mut c_void,
        NewTokenHandle: *mut HANDLE,
    ) -> BOOL;
    fn OpenProcessToken(
        ProcessHandle: HANDLE,
        DesiredAccess: DWORD,
        TokenHandle: *mut HANDLE,
    ) -> BOOL;
    fn FreeSid(pSid: PSID) -> *mut c_void;
}

#[link(name = "kernel32")]
unsafe extern "system" {
    fn GetCurrentProcess() -> HANDLE;
    fn CloseHandle(h: HANDLE) -> BOOL;
}

const TOKEN_DUPLICATE: DWORD = 0x0002;
const TOKEN_QUERY: DWORD = 0x0008;
const TOKEN_ASSIGN_PRIMARY: DWORD = 0x0001;
const TOKEN_ADJUST_DEFAULT: DWORD = 0x0080;
const DISABLE_MAX_PRIVILEGE: DWORD = 0x0001;

/// Result of a successful AppContainer creation: holds the SID and a
/// restricted token the broker passes to CreateProcessAsUserW. Both
/// handles auto-close on drop.
pub struct AppContainerSandbox {
    pub appcontainer_sid: PSID,
    pub restricted_token: HANDLE,
}

impl AppContainerSandbox {
    /// Build (or look up) an AppContainer profile and produce a
    /// restricted token derived from the current process token.
    pub fn create(name: &str) -> Result<Self, String> {
        let wide: Vec<u16> = name.encode_utf16().chain(std::iter::once(0)).collect();
        let mut sid: PSID = ptr::null_mut();
        unsafe {
            let mut hr = CreateAppContainerProfile(
                wide.as_ptr(),
                wide.as_ptr(),
                wide.as_ptr(),
                ptr::null_mut(),
                0,
                &mut sid,
            );
            if hr != 0 {
                hr = DeriveAppContainerSidFromAppContainerName(wide.as_ptr(), &mut sid);
                if hr != 0 {
                    return Err(format!("AppContainer profile hr={hr:#x}"));
                }
            }
            let mut existing: HANDLE = ptr::null_mut();
            if OpenProcessToken(
                GetCurrentProcess(),
                TOKEN_DUPLICATE | TOKEN_QUERY | TOKEN_ASSIGN_PRIMARY | TOKEN_ADJUST_DEFAULT,
                &mut existing,
            ) == 0
            {
                return Err("OpenProcessToken".into());
            }
            let mut restricted: HANDLE = ptr::null_mut();
            let ok = CreateRestrictedToken(
                existing,
                DISABLE_MAX_PRIVILEGE,
                0,
                ptr::null_mut(),
                0,
                ptr::null_mut(),
                0,
                ptr::null_mut(),
                &mut restricted,
            );
            CloseHandle(existing);
            if ok == 0 || restricted.is_null() {
                return Err("CreateRestrictedToken".into());
            }
            Ok(Self {
                appcontainer_sid: sid,
                restricted_token: restricted,
            })
        }
    }
}

impl Drop for AppContainerSandbox {
    fn drop(&mut self) {
        unsafe {
            if !self.restricted_token.is_null() {
                CloseHandle(self.restricted_token);
            }
            if !self.appcontainer_sid.is_null() {
                FreeSid(self.appcontainer_sid);
            }
        }
    }
}
