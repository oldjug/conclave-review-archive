//! X.509 chain path-building + policy validation via Windows Crypt32.
//!
//! Our own parser (`cv_crypto::x509`) handles DER decoding and signature
//! algorithm dispatch. Anchoring the chain at a trusted root requires the
//! Windows root certificate store, which we do not reimplement — we hand
//! the DER blobs to `CertGetCertificateChain` and ask
//! `CertVerifyCertificateChainPolicy(CERT_CHAIN_POLICY_SSL)` to do the
//! anchor + EKU + hostname + revocation logic. Same approach Edge/Brave
//! take on Windows.

#![allow(non_camel_case_types, non_snake_case)]

use core::ffi::c_void;
use core::ptr;

type BOOL = i32;
type DWORD = u32;
type LONG = i32;
type LPCSTR = *const u8;
type LPSTR = *mut u8;
type LPWSTR = *mut u16;
type LPVOID = *mut c_void;
type HCERTSTORE = *mut c_void;
type HCERTCHAINENGINE = *mut c_void;
type PCCERT_CONTEXT = *const CERT_CONTEXT;
type PCCERT_CHAIN_CONTEXT = *const c_void;

const X509_ASN_ENCODING: DWORD = 0x0000_0001;
const PKCS_7_ASN_ENCODING: DWORD = 0x0001_0000;
const ENCODING: DWORD = X509_ASN_ENCODING | PKCS_7_ASN_ENCODING;
const CERT_STORE_PROV_MEMORY: LPCSTR = 2 as LPCSTR;
const CERT_STORE_ADD_ALWAYS: DWORD = 4;
const CERT_CHAIN_POLICY_SSL: LPCSTR = 4 as LPCSTR;
const AUTHTYPE_SERVER: DWORD = 2;
const USAGE_MATCH_TYPE_AND: DWORD = 0;

#[repr(C)]
struct CERT_CONTEXT {
    dwCertEncodingType: DWORD,
    pbCertEncoded: *const u8,
    cbCertEncoded: DWORD,
    pCertInfo: *mut c_void,
    hCertStore: HCERTSTORE,
}

#[repr(C)]
struct CERT_ENHKEY_USAGE {
    cUsageIdentifier: DWORD,
    rgpszUsageIdentifier: *mut LPSTR,
}

#[repr(C)]
struct CERT_USAGE_MATCH {
    dwType: DWORD,
    Usage: CERT_ENHKEY_USAGE,
}

#[repr(C)]
struct CERT_CHAIN_PARA {
    cbSize: DWORD,
    RequestedUsage: CERT_USAGE_MATCH,
}

#[repr(C)]
struct CERT_CHAIN_POLICY_PARA {
    cbSize: DWORD,
    dwFlags: DWORD,
    pvExtraPolicyPara: LPVOID,
}

#[repr(C)]
struct CERT_CHAIN_POLICY_STATUS {
    cbSize: DWORD,
    dwError: DWORD,
    lChainIndex: LONG,
    lElementIndex: LONG,
    pvExtraPolicyStatus: LPVOID,
}

#[repr(C)]
struct SSL_EXTRA_CERT_CHAIN_POLICY_PARA {
    cbSize: DWORD,
    dwAuthType: DWORD,
    fdwChecks: DWORD,
    pwszServerName: LPWSTR,
}

#[link(name = "crypt32")]
unsafe extern "system" {
    fn CertCreateCertificateContext(
        dwCertEncodingType: DWORD,
        pbCertEncoded: *const u8,
        cbCertEncoded: DWORD,
    ) -> PCCERT_CONTEXT;

    fn CertFreeCertificateContext(pCertContext: PCCERT_CONTEXT) -> BOOL;

    fn CertOpenStore(
        lpszStoreProvider: LPCSTR,
        dwEncodingType: DWORD,
        hCryptProv: usize,
        dwFlags: DWORD,
        pvPara: *const c_void,
    ) -> HCERTSTORE;

    fn CertCloseStore(hCertStore: HCERTSTORE, dwFlags: DWORD) -> BOOL;

    fn CertAddEncodedCertificateToStore(
        hCertStore: HCERTSTORE,
        dwCertEncodingType: DWORD,
        pbCertEncoded: *const u8,
        cbCertEncoded: DWORD,
        dwAddDisposition: DWORD,
        ppCertContext: *mut PCCERT_CONTEXT,
    ) -> BOOL;

    fn CertGetCertificateChain(
        hChainEngine: HCERTCHAINENGINE,
        pCertContext: PCCERT_CONTEXT,
        pTime: *const c_void,
        hAdditionalStore: HCERTSTORE,
        pChainPara: *const CERT_CHAIN_PARA,
        dwFlags: DWORD,
        pvReserved: LPVOID,
        ppChainContext: *mut PCCERT_CHAIN_CONTEXT,
    ) -> BOOL;

    fn CertFreeCertificateChain(pChainContext: PCCERT_CHAIN_CONTEXT);

    fn CertVerifyCertificateChainPolicy(
        pszPolicyOID: LPCSTR,
        pChainContext: PCCERT_CHAIN_CONTEXT,
        pPolicyPara: *const CERT_CHAIN_POLICY_PARA,
        pPolicyStatus: *mut CERT_CHAIN_POLICY_STATUS,
    ) -> BOOL;
}

#[link(name = "kernel32")]
unsafe extern "system" {
    fn GetLastError() -> DWORD;
}

fn last_error() -> u32 {
    unsafe { GetLastError() }
}

#[derive(Debug)]
pub enum ChainError {
    EmptyChain,
    LeafContextFailed(u32),
    StoreOpenFailed(u32),
    AddIntermediateFailed(u32),
    GetChainFailed(u32),
    VerifyPolicyFailed(u32),
    /// `dwError` from `CERT_CHAIN_POLICY_STATUS` — Win32 error code or
    /// `CERT_E_*`. Notable: 0x800B0109 untrusted root, 0x800B010A no chain,
    /// 0x800B0101 expired, 0x800B010F CN_NO_MATCH, 0x80092012 revoked.
    PolicyError(u32),
}

impl core::fmt::Display for ChainError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::EmptyChain => f.write_str("empty certificate chain"),
            Self::LeafContextFailed(e) => write!(f, "CertCreateCertificateContext: 0x{e:08x}"),
            Self::StoreOpenFailed(e) => write!(f, "CertOpenStore: 0x{e:08x}"),
            Self::AddIntermediateFailed(e) => {
                write!(f, "CertAddEncodedCertificateToStore: 0x{e:08x}")
            }
            Self::GetChainFailed(e) => write!(f, "CertGetCertificateChain: 0x{e:08x}"),
            Self::VerifyPolicyFailed(e) => write!(f, "CertVerifyCertificateChainPolicy: 0x{e:08x}"),
            Self::PolicyError(e) => write!(f, "chain policy rejected: 0x{e:08x}"),
        }
    }
}

impl std::error::Error for ChainError {}

/// Verify `cert_chain` (DER, leaf at index 0, intermediates after) anchors
/// at a trusted Windows root and satisfies the SSL/TLS server policy for
/// `host`. Returns `Ok` only when Windows fully trusts the chain.
pub fn verify_chain(cert_chain: &[Vec<u8>], host: &str) -> Result<(), ChainError> {
    if cert_chain.is_empty() {
        return Err(ChainError::EmptyChain);
    }
    unsafe {
        let leaf = CertCreateCertificateContext(
            ENCODING,
            cert_chain[0].as_ptr(),
            cert_chain[0].len() as DWORD,
        );
        if leaf.is_null() {
            return Err(ChainError::LeafContextFailed(last_error()));
        }

        let store = CertOpenStore(CERT_STORE_PROV_MEMORY, ENCODING, 0, 0, ptr::null());
        if store.is_null() {
            let err = last_error();
            CertFreeCertificateContext(leaf);
            return Err(ChainError::StoreOpenFailed(err));
        }

        for inter in &cert_chain[1..] {
            let ok = CertAddEncodedCertificateToStore(
                store,
                ENCODING,
                inter.as_ptr(),
                inter.len() as DWORD,
                CERT_STORE_ADD_ALWAYS,
                ptr::null_mut(),
            );
            if ok == 0 {
                let err = last_error();
                CertCloseStore(store, 0);
                CertFreeCertificateContext(leaf);
                return Err(ChainError::AddIntermediateFailed(err));
            }
        }

        // szOID_PKIX_KP_SERVER_AUTH = 1.3.6.1.5.5.7.3.1
        let server_auth_oid: &[u8] = b"1.3.6.1.5.5.7.3.1\0";
        let mut usage_ptrs: [LPSTR; 1] = [server_auth_oid.as_ptr() as LPSTR];
        let para = CERT_CHAIN_PARA {
            cbSize: core::mem::size_of::<CERT_CHAIN_PARA>() as DWORD,
            RequestedUsage: CERT_USAGE_MATCH {
                dwType: USAGE_MATCH_TYPE_AND,
                Usage: CERT_ENHKEY_USAGE {
                    cUsageIdentifier: 1,
                    rgpszUsageIdentifier: usage_ptrs.as_mut_ptr(),
                },
            },
        };

        let mut chain_ctx: PCCERT_CHAIN_CONTEXT = ptr::null();
        let ok = CertGetCertificateChain(
            ptr::null_mut(),
            leaf,
            ptr::null(),
            store,
            &para,
            0,
            ptr::null_mut(),
            &mut chain_ctx,
        );
        if ok == 0 {
            let err = last_error();
            CertCloseStore(store, 0);
            CertFreeCertificateContext(leaf);
            return Err(ChainError::GetChainFailed(err));
        }

        let mut host_w: Vec<u16> = host.encode_utf16().collect();
        host_w.push(0);
        let mut ssl_para = SSL_EXTRA_CERT_CHAIN_POLICY_PARA {
            cbSize: core::mem::size_of::<SSL_EXTRA_CERT_CHAIN_POLICY_PARA>() as DWORD,
            dwAuthType: AUTHTYPE_SERVER,
            fdwChecks: 0,
            pwszServerName: host_w.as_mut_ptr(),
        };
        let policy_para = CERT_CHAIN_POLICY_PARA {
            cbSize: core::mem::size_of::<CERT_CHAIN_POLICY_PARA>() as DWORD,
            dwFlags: 0,
            pvExtraPolicyPara: &mut ssl_para as *mut _ as LPVOID,
        };
        let mut status = CERT_CHAIN_POLICY_STATUS {
            cbSize: core::mem::size_of::<CERT_CHAIN_POLICY_STATUS>() as DWORD,
            dwError: 0,
            lChainIndex: 0,
            lElementIndex: 0,
            pvExtraPolicyStatus: ptr::null_mut(),
        };
        let ok = CertVerifyCertificateChainPolicy(
            CERT_CHAIN_POLICY_SSL,
            chain_ctx,
            &policy_para,
            &mut status,
        );
        let policy_err = status.dwError;
        CertFreeCertificateChain(chain_ctx);
        CertCloseStore(store, 0);
        CertFreeCertificateContext(leaf);

        if ok == 0 {
            return Err(ChainError::VerifyPolicyFailed(last_error()));
        }
        if policy_err != 0 {
            return Err(ChainError::PolicyError(policy_err));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_chain_rejected() {
        assert!(matches!(
            verify_chain(&[], "example.com"),
            Err(ChainError::EmptyChain)
        ));
    }

    #[test]
    fn garbage_leaf_rejected() {
        let junk = vec![0x00u8; 32];
        let res = verify_chain(&[junk], "example.com");
        assert!(matches!(res, Err(ChainError::LeafContextFailed(_))));
    }
}
