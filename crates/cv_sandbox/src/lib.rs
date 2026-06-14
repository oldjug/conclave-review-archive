//! `cv_sandbox` — Windows-native renderer sandbox primitives.
//!
//! V1 ships the broker-side mechanics for locking down a child
//! process per Chromium's `//sandbox/win` model:
//!
//!   * `AppContainerSid` — derive a per-installation AppContainer SID
//!     from a stable string (one SID per browser install so caches
//!     persist across runs).
//!   * `restrict_token()` — build a restricted token by stripping
//!     privileges and removing all SIDs except the Everyone + the
//!     AppContainer SID.
//!   * `mitigation_policies()` — pack the PROCESS_MITIGATION_POLICY
//!     bits used by `STARTUPINFOEX` for ASLR, DEP, ACG, blocked-
//!     dynamic-code, child-process restriction, font/image-load
//!     restriction.
//!   * `apply_attributes_to_startup_info()` — populate an
//!     `STARTUPINFOEX` with the AppContainer + mitigation attributes
//!     so the spawned child inherits the lockdown.
//!
//! The actual `CreateProcessW` call sits in `cv_ipc` because that's
//! where the pipe + job-object plumbing lives. This crate is the
//! security-policy authority — the bytes / handles / SIDs the broker
//! hands to `cv_ipc::SandboxedChild`.
//!
//! NOTE: only the Win32 shapes that the broker needs are declared
//! here; the kernel32 / advapi32 functions used downstream live in
//! `cv_ipc::sandbox`.

#![allow(
    non_camel_case_types,
    non_snake_case,
    dead_code,
    missing_debug_implementations,
    unreachable_pub
)]

pub mod appcontainer;

use core::ffi::c_void;

type BOOL = i32;
type DWORD = u32;
type DWORD_PTR = usize;
type HANDLE = *mut c_void;
type PCWSTR = *const u16;
type PSID = *mut c_void;
type SECURITY_CAPABILITIES_PTR = *mut SecurityCapabilities;

/// SECURITY_CAPABILITIES per WinAPI. Used to attach an AppContainer
/// SID to a process at creation time.
#[repr(C)]
#[derive(Debug)]
pub struct SecurityCapabilities {
    pub app_container_sid: PSID,
    pub capabilities: *mut c_void,
    pub capability_count: DWORD,
    pub reserved: DWORD,
}

/// Subset of the PROCESS_MITIGATION_POLICY bits we want to set by
/// default on every spawned renderer. Packed into the 64-bit value
/// passed to `UpdateProcThreadAttribute(... ProcThreadAttributeMitigationPolicy ...)`.
#[derive(Debug, Default, Copy, Clone)]
pub struct MitigationPolicies {
    pub disable_extension_points: bool,
    pub aslr_force_high_entropy: bool,
    pub aslr_force_relocate_images: bool,
    pub aslr_require_relocations: bool,
    pub aslr_disallow_stripped_images: bool,
    pub strict_handle_checks: bool,
    pub dynamic_code_disable: bool,
    pub dynamic_code_block_remote_only: bool,
    pub child_process_restricted: bool,
    pub block_non_microsoft_binaries: bool,
    pub font_disable: bool,
    pub image_load_no_remote: bool,
    pub image_load_no_low_label: bool,
    pub image_load_prefer_system32: bool,
    pub cet_user_shadow_stacks: bool,
    pub control_flow_guard: bool,
}

impl MitigationPolicies {
    /// Default policy bundle for a renderer-class child: DEP via
    /// ASLR, ACG, no remote child processes, no win32k font load,
    /// no remote DLL loads, and CFG enforcement.
    pub fn renderer_defaults() -> Self {
        Self {
            disable_extension_points: true,
            aslr_force_high_entropy: true,
            aslr_force_relocate_images: true,
            aslr_require_relocations: true,
            aslr_disallow_stripped_images: true,
            strict_handle_checks: true,
            dynamic_code_disable: true,
            dynamic_code_block_remote_only: false,
            child_process_restricted: true,
            block_non_microsoft_binaries: false,
            font_disable: true,
            image_load_no_remote: true,
            image_load_no_low_label: true,
            image_load_prefer_system32: true,
            cet_user_shadow_stacks: true,
            control_flow_guard: true,
        }
    }

    /// Pack the bits into the 64-bit value that
    /// `UpdateProcThreadAttribute` consumes for the
    /// `PROC_THREAD_ATTRIBUTE_MITIGATION_POLICY` attribute. The bit
    /// positions match `Winnt.h`.
    pub fn pack_word(self) -> u64 {
        // Bit definitions from WinNT.h (Windows 10 SDK).
        const PROCESS_CREATION_MITIGATION_POLICY_DEP_ENABLE: u64 = 0x01;
        const PROCESS_CREATION_MITIGATION_POLICY_SEHOP_ENABLE: u64 = 0x04;
        const PROCESS_CREATION_MITIGATION_POLICY_FORCE_RELOCATE_IMAGES_ALWAYS_ON: u64 = 0x100;
        const PROCESS_CREATION_MITIGATION_POLICY_HEAP_TERMINATE_ALWAYS_ON: u64 = 0x1000;
        const PROCESS_CREATION_MITIGATION_POLICY_BOTTOM_UP_ASLR_ALWAYS_ON: u64 = 0x10000;
        const PROCESS_CREATION_MITIGATION_POLICY_HIGH_ENTROPY_ASLR_ALWAYS_ON: u64 = 0x100000;
        const PROCESS_CREATION_MITIGATION_POLICY_STRICT_HANDLE_CHECKS_ALWAYS_ON: u64 = 0x100_0000;
        const PROCESS_CREATION_MITIGATION_POLICY_WIN32K_SYSTEM_CALL_DISABLE_ALWAYS_ON: u64 =
            0x1000_0000;
        const PROCESS_CREATION_MITIGATION_POLICY_EXTENSION_POINT_DISABLE_ALWAYS_ON: u64 =
            0x1_0000_0000;
        const PROCESS_CREATION_MITIGATION_POLICY_PROHIBIT_DYNAMIC_CODE_ALWAYS_ON: u64 =
            0x1_0000_0000_0;
        const PROCESS_CREATION_MITIGATION_POLICY_CONTROL_FLOW_GUARD_ALWAYS_ON: u64 =
            0x1_0000_0000_00;
        const PROCESS_CREATION_MITIGATION_POLICY_BLOCK_NON_MICROSOFT_BINARIES_ALWAYS_ON: u64 =
            0x1_0000_0000_000;
        const PROCESS_CREATION_MITIGATION_POLICY_FONT_DISABLE_ALWAYS_ON: u64 = 0x1_0000_0000_0000;
        const PROCESS_CREATION_MITIGATION_POLICY_IMAGE_LOAD_NO_REMOTE_ALWAYS_ON: u64 =
            0x1_0000_0000_00000;
        const PROCESS_CREATION_MITIGATION_POLICY_IMAGE_LOAD_NO_LOW_LABEL_ALWAYS_ON: u64 =
            0x1_0000_0000_000000;
        const PROCESS_CREATION_MITIGATION_POLICY_IMAGE_LOAD_PREFER_SYSTEM32_ALWAYS_ON: u64 =
            0x1_0000_0000_0000000;
        let mut bits: u64 = 0;
        bits |= PROCESS_CREATION_MITIGATION_POLICY_DEP_ENABLE;
        bits |= PROCESS_CREATION_MITIGATION_POLICY_SEHOP_ENABLE;
        if self.aslr_force_relocate_images {
            bits |= PROCESS_CREATION_MITIGATION_POLICY_FORCE_RELOCATE_IMAGES_ALWAYS_ON;
        }
        if self.aslr_force_high_entropy {
            bits |= PROCESS_CREATION_MITIGATION_POLICY_HIGH_ENTROPY_ASLR_ALWAYS_ON;
            bits |= PROCESS_CREATION_MITIGATION_POLICY_BOTTOM_UP_ASLR_ALWAYS_ON;
        }
        if self.strict_handle_checks {
            bits |= PROCESS_CREATION_MITIGATION_POLICY_STRICT_HANDLE_CHECKS_ALWAYS_ON;
            bits |= PROCESS_CREATION_MITIGATION_POLICY_HEAP_TERMINATE_ALWAYS_ON;
        }
        if self.disable_extension_points {
            bits |= PROCESS_CREATION_MITIGATION_POLICY_EXTENSION_POINT_DISABLE_ALWAYS_ON;
        }
        if self.dynamic_code_disable {
            bits |= PROCESS_CREATION_MITIGATION_POLICY_PROHIBIT_DYNAMIC_CODE_ALWAYS_ON;
        }
        if self.control_flow_guard {
            bits |= PROCESS_CREATION_MITIGATION_POLICY_CONTROL_FLOW_GUARD_ALWAYS_ON;
        }
        if self.font_disable {
            bits |= PROCESS_CREATION_MITIGATION_POLICY_FONT_DISABLE_ALWAYS_ON;
            bits |= PROCESS_CREATION_MITIGATION_POLICY_WIN32K_SYSTEM_CALL_DISABLE_ALWAYS_ON;
        }
        if self.image_load_no_remote {
            bits |= PROCESS_CREATION_MITIGATION_POLICY_IMAGE_LOAD_NO_REMOTE_ALWAYS_ON;
        }
        if self.image_load_no_low_label {
            bits |= PROCESS_CREATION_MITIGATION_POLICY_IMAGE_LOAD_NO_LOW_LABEL_ALWAYS_ON;
        }
        if self.image_load_prefer_system32 {
            bits |= PROCESS_CREATION_MITIGATION_POLICY_IMAGE_LOAD_PREFER_SYSTEM32_ALWAYS_ON;
        }
        if self.block_non_microsoft_binaries {
            bits |= PROCESS_CREATION_MITIGATION_POLICY_BLOCK_NON_MICROSOFT_BINARIES_ALWAYS_ON;
        }
        bits
    }
}

/// AppContainer-style SID derived from a stable installation key.
/// Real ConvertStringSidToSidW expects a SID string in the
/// "S-1-15-2-..." subauthority form; we synthesize a plausible one
/// from a SHA-256-style mix of the install key so different installs
/// of the browser don't share AppContainer storage.
#[derive(Debug, Clone)]
pub struct AppContainerSid {
    /// 8 subauthorities (per the AppContainer convention) packed as a
    /// 32-byte vector. Real WinAPI usage converts this to a PSID via
    /// `ConvertStringSidToSidW` after formatting as a string.
    pub subauthorities: [u32; 8],
    pub display_name: String,
}

impl AppContainerSid {
    /// Derive a deterministic-per-install SID. `key` is typically a
    /// concatenation of {channel, install path, profile id} so two
    /// browsers can coexist with isolated AppContainer caches.
    pub fn from_install_key(key: &str) -> Self {
        // FNV-1a fold for V1 — cryptographic strength doesn't matter
        // here; we only need enough spread that two installs don't
        // collide. Real Chromium uses SHA-256 + a fixed namespace UUID.
        let mut hashes = [0u32; 8];
        let mut h: u32 = 0x811C_9DC5;
        for (i, b) in key.bytes().enumerate() {
            h ^= u32::from(b);
            h = h.wrapping_mul(0x0100_0193);
            hashes[i % 8] ^= h;
        }
        Self {
            subauthorities: hashes,
            display_name: format!("ConclaveAppContainer-{}", h),
        }
    }

    /// Format as a SID string ("S-1-15-2-x-x-x-...") suitable for
    /// `ConvertStringSidToSidW`. 15 = SECURITY_APP_PACKAGE_AUTHORITY.
    pub fn to_string_sid(&self) -> String {
        let parts: Vec<String> = self.subauthorities.iter().map(|x| x.to_string()).collect();
        format!("S-1-15-2-{}", parts.join("-"))
    }
}

/// Build a SecurityCapabilities struct from an AppContainer SID and an
/// (initially empty) capability set. The caller is responsible for
/// converting the SID string and keeping the storage alive for the
/// duration of the CreateProcessW call.
pub fn build_security_capabilities(sid_ptr: PSID) -> SecurityCapabilities {
    SecurityCapabilities {
        app_container_sid: sid_ptr,
        capabilities: core::ptr::null_mut(),
        capability_count: 0,
        reserved: 0,
    }
}

/// Per-channel default policy: how locked-down a renderer should be.
#[derive(Debug, Clone, Copy)]
pub struct ChannelPolicy {
    pub mitigation: MitigationPolicies,
    pub use_app_container: bool,
    pub low_integrity: bool,
    pub allow_child_processes: bool,
}

impl ChannelPolicy {
    /// Most-restrictive default for "stable" channel renderers.
    pub fn stable_renderer() -> Self {
        Self {
            mitigation: MitigationPolicies::renderer_defaults(),
            use_app_container: true,
            low_integrity: true,
            allow_child_processes: false,
        }
    }

    /// Slightly looser policy for developer builds (debugger-friendly,
    /// no AppContainer so PII paths are inspectable).
    pub fn dev_renderer() -> Self {
        Self {
            mitigation: MitigationPolicies {
                dynamic_code_disable: false,
                font_disable: false,
                ..MitigationPolicies::renderer_defaults()
            },
            use_app_container: false,
            low_integrity: false,
            allow_child_processes: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn app_container_sid_string_starts_with_s_1_15_2() {
        let sid = AppContainerSid::from_install_key("toasty-blum-stable-userprofile");
        let s = sid.to_string_sid();
        assert!(s.starts_with("S-1-15-2-"));
        assert_eq!(s.matches('-').count(), 11);
    }

    #[test]
    fn different_install_keys_produce_different_sids() {
        let a = AppContainerSid::from_install_key("install-A");
        let b = AppContainerSid::from_install_key("install-B");
        assert_ne!(a.subauthorities, b.subauthorities);
    }

    #[test]
    fn renderer_defaults_enable_aslr_and_cfg() {
        let m = MitigationPolicies::renderer_defaults();
        assert!(m.aslr_force_high_entropy);
        assert!(m.control_flow_guard);
        assert!(m.dynamic_code_disable);
        assert!(m.font_disable);
    }

    #[test]
    fn packed_word_has_dep_bit_set() {
        let m = MitigationPolicies::renderer_defaults();
        let w = m.pack_word();
        // Bit 0 = DEP_ENABLE.
        assert_ne!(w & 0x01, 0);
    }

    #[test]
    fn stable_renderer_uses_app_container() {
        let p = ChannelPolicy::stable_renderer();
        assert!(p.use_app_container);
        assert!(p.low_integrity);
        assert!(!p.allow_child_processes);
    }
}
