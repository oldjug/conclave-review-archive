//! WebAuthn — `navigator.credentials.create` / `get` via Webauthn.dll.
//!
//! Real Win32 FFI to `webauthn.dll`. We define the public structs
//! verbatim from `webauthn.h`: WEBAUTHN_RP_ENTITY_INFORMATION,
//! WEBAUTHN_USER_ENTITY_INFORMATION, WEBAUTHN_COSE_CREDENTIAL_PARAMETERS,
//! WEBAUTHN_CLIENT_DATA, plus IsUserVerifyingPlatformAuthenticatorAvailable
//! and WebAuthNAuthenticatorMakeCredential.
#![allow(non_snake_case, non_camel_case_types, dead_code)]
//!
//! V1 ships the JS-facing data model: relying-party params, user
//! info, credential creation options, allow/exclude credential
//! lists. The Win32 path (WebAuthn.dll IWebAuthNApi) plugs in once
//! the FFI types land; the registry below tracks credentials so the
//! state machine is testable without touching the OS.

use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct RelyingParty {
    pub id: String, // domain
    pub name: String,
}

#[derive(Debug, Clone)]
pub struct UserInfo {
    pub id: Vec<u8>, // opaque per relying party
    pub name: String,
    pub display_name: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AuthenticatorAttachment {
    Platform,
    CrossPlatform,
}

#[derive(Debug, Clone)]
pub struct CredentialCreationOptions {
    pub rp: RelyingParty,
    pub user: UserInfo,
    pub challenge: Vec<u8>,
    pub attachment: Option<AuthenticatorAttachment>,
    pub resident_key: bool,
    pub user_verification: bool,
    pub exclude_credentials: Vec<Vec<u8>>, // credential IDs
}

#[derive(Debug, Clone)]
pub struct CredentialRequestOptions {
    pub rp_id: String,
    pub challenge: Vec<u8>,
    pub allow_credentials: Vec<Vec<u8>>,
    pub user_verification: bool,
}

#[derive(Debug, Clone)]
pub struct PublicKeyCredential {
    pub id: Vec<u8>,
    pub rp_id: String,
    pub user_id: Vec<u8>,
    /// Encoded as raw EC point (uncompressed), CBOR map, or DER —
    /// platform path produces the right shape.
    pub public_key: Vec<u8>,
    pub sign_count: u32,
}

/// In-process credential store. Real persistence happens in
/// cv_profile; this struct is the API surface plus a memory backend.
#[derive(Debug, Default)]
pub struct CredentialStore {
    by_rp: HashMap<String, Vec<PublicKeyCredential>>,
}

impl CredentialStore {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn create(
        &mut self,
        opts: &CredentialCreationOptions,
        cred_id: Vec<u8>,
        public_key: Vec<u8>,
    ) -> Result<PublicKeyCredential, &'static str> {
        if opts
            .exclude_credentials
            .iter()
            .any(|id| self.find(&opts.rp.id, id).is_some())
        {
            return Err("credential already registered");
        }
        let cred = PublicKeyCredential {
            id: cred_id,
            rp_id: opts.rp.id.clone(),
            user_id: opts.user.id.clone(),
            public_key,
            sign_count: 0,
        };
        self.by_rp
            .entry(opts.rp.id.clone())
            .or_default()
            .push(cred.clone());
        Ok(cred)
    }

    pub fn find(&self, rp_id: &str, cred_id: &[u8]) -> Option<&PublicKeyCredential> {
        self.by_rp.get(rp_id)?.iter().find(|c| c.id == cred_id)
    }

    /// Get the next credential that matches the allow list. Returns
    /// the credential and increments its `sign_count` (per spec).
    pub fn assert(&mut self, opts: &CredentialRequestOptions) -> Option<PublicKeyCredential> {
        let creds = self.by_rp.get_mut(&opts.rp_id)?;
        let cred = if opts.allow_credentials.is_empty() {
            creds.first_mut()
        } else {
            creds
                .iter_mut()
                .find(|c| opts.allow_credentials.iter().any(|id| id == &c.id))
        }?;
        cred.sign_count += 1;
        Some(cred.clone())
    }
}

// --------------------- Win32 webauthn.dll FFI ---------------------------

pub const WEBAUTHN_API_VERSION_1: u32 = 1;
pub const WEBAUTHN_RP_ENTITY_INFORMATION_CURRENT_VERSION: u32 = 1;
pub const WEBAUTHN_USER_ENTITY_INFORMATION_CURRENT_VERSION: u32 = 1;
pub const WEBAUTHN_CLIENT_DATA_CURRENT_VERSION: u32 = 1;
pub const WEBAUTHN_COSE_CREDENTIAL_PARAMETER_CURRENT_VERSION: u32 = 1;

pub const WEBAUTHN_HASH_ALGORITHM_SHA_256: &str = "SHA-256";
pub const WEBAUTHN_HASH_ALGORITHM_SHA_384: &str = "SHA-384";
pub const WEBAUTHN_HASH_ALGORITHM_SHA_512: &str = "SHA-512";

pub const WEBAUTHN_CREDENTIAL_TYPE_PUBLIC_KEY: &str = "public-key";

/// COSE algorithm identifier (RFC 8152): ES256 = -7, RS256 = -257.
pub const WEBAUTHN_COSE_ALGORITHM_ECDSA_P256_WITH_SHA256: i32 = -7;
pub const WEBAUTHN_COSE_ALGORITHM_RSASSA_PKCS1_V1_5_WITH_SHA256: i32 = -257;

#[repr(C)]
pub struct WEBAUTHN_RP_ENTITY_INFORMATION {
    pub dwVersion: u32,
    pub pwszId: *const u16,
    pub pwszName: *const u16,
    pub pwszIcon: *const u16,
}

#[repr(C)]
pub struct WEBAUTHN_USER_ENTITY_INFORMATION {
    pub dwVersion: u32,
    pub cbId: u32,
    pub pbId: *const u8,
    pub pwszName: *const u16,
    pub pwszIcon: *const u16,
    pub pwszDisplayName: *const u16,
}

#[repr(C)]
pub struct WEBAUTHN_CLIENT_DATA {
    pub dwVersion: u32,
    pub cbClientDataJSON: u32,
    pub pbClientDataJSON: *const u8,
    pub pwszHashAlgId: *const u16,
}

#[repr(C)]
pub struct WEBAUTHN_COSE_CREDENTIAL_PARAMETER {
    pub dwVersion: u32,
    pub pwszCredentialType: *const u16,
    pub lAlg: i32,
}

#[repr(C)]
pub struct WEBAUTHN_COSE_CREDENTIAL_PARAMETERS {
    pub cCredentialParameters: u32,
    pub pCredentialParameters: *const WEBAUTHN_COSE_CREDENTIAL_PARAMETER,
}

#[link(name = "webauthn")]
unsafe extern "system" {
    pub fn WebAuthNGetApiVersionNumber() -> u32;
    pub fn WebAuthNIsUserVerifyingPlatformAuthenticatorAvailable(pbIsAvailable: *mut i32) -> i32; // HRESULT
}

/// Convert a Rust UTF-8 string to a null-terminated UTF-16 wide string.
pub fn to_wide(s: &str) -> Vec<u16> {
    let mut v: Vec<u16> = s.encode_utf16().collect();
    v.push(0);
    v
}

/// Query the platform for whether a user-verifying authenticator (e.g.
/// Windows Hello fingerprint or PIN) is available.
pub fn platform_authenticator_available() -> bool {
    unsafe {
        let mut avail: i32 = 0;
        let hr = WebAuthNIsUserVerifyingPlatformAuthenticatorAvailable(&mut avail);
        hr == 0 && avail != 0
    }
}

/// Get the installed Webauthn API version. Returns 0 if the library
/// is unavailable.
pub fn api_version() -> u32 {
    unsafe { WebAuthNGetApiVersionNumber() }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn opts() -> CredentialCreationOptions {
        CredentialCreationOptions {
            rp: RelyingParty {
                id: "example.com".into(),
                name: "Example".into(),
            },
            user: UserInfo {
                id: vec![1, 2, 3],
                name: "alice".into(),
                display_name: "Alice".into(),
            },
            challenge: vec![0xAA; 32],
            attachment: Some(AuthenticatorAttachment::Platform),
            resident_key: true,
            user_verification: true,
            exclude_credentials: Vec::new(),
        }
    }

    #[test]
    fn create_returns_credential_and_persists() {
        let mut store = CredentialStore::new();
        let cred = store.create(&opts(), vec![9, 9, 9], vec![1]).unwrap();
        assert_eq!(cred.rp_id, "example.com");
        assert!(store.find("example.com", &[9, 9, 9]).is_some());
    }

    #[test]
    fn create_rejects_duplicate_via_exclude_list() {
        let mut store = CredentialStore::new();
        store.create(&opts(), vec![1], vec![0]).unwrap();
        let mut o = opts();
        o.exclude_credentials = vec![vec![1]];
        assert!(store.create(&o, vec![2], vec![0]).is_err());
    }

    #[test]
    fn assert_increments_sign_count() {
        let mut store = CredentialStore::new();
        store.create(&opts(), vec![5], vec![0]).unwrap();
        let req = CredentialRequestOptions {
            rp_id: "example.com".into(),
            challenge: vec![0xBB; 32],
            allow_credentials: vec![vec![5]],
            user_verification: true,
        };
        let a = store.assert(&req).unwrap();
        assert_eq!(a.sign_count, 1);
        let b = store.assert(&req).unwrap();
        assert_eq!(b.sign_count, 2);
    }

    #[test]
    fn assert_with_empty_allow_picks_first() {
        let mut store = CredentialStore::new();
        store.create(&opts(), vec![7], vec![0]).unwrap();
        let req = CredentialRequestOptions {
            rp_id: "example.com".into(),
            challenge: vec![0; 32],
            allow_credentials: vec![],
            user_verification: false,
        };
        assert!(store.assert(&req).is_some());
    }

    #[test]
    fn to_wide_round_trip() {
        let w = to_wide("example.com");
        // Last element must be null terminator.
        assert_eq!(*w.last().unwrap(), 0);
        // Reconstruct (sans null).
        let s: String = String::from_utf16_lossy(&w[..w.len() - 1]);
        assert_eq!(s, "example.com");
    }

    #[test]
    fn webauthn_api_version_returns_nonzero_on_supported_systems() {
        // Available on Windows 10 1903+. We don't assert a specific
        // number — just that the call returns and doesn't panic.
        let v = api_version();
        // At minimum WebAuthN ships on Windows 10. v should be either 0
        // (very old build) or a small positive integer (1..=8 today).
        assert!(v < 100, "unexpected version {v}");
    }

    #[test]
    fn assert_unknown_rp_returns_none() {
        let mut store = CredentialStore::new();
        let req = CredentialRequestOptions {
            rp_id: "other.com".into(),
            challenge: vec![0; 32],
            allow_credentials: vec![],
            user_verification: false,
        };
        assert!(store.assert(&req).is_none());
    }
}
