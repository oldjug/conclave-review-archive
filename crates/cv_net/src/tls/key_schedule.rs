//! TLS 1.3 key schedule per RFC 8446 §7.1.
//!
//! Provides `hkdf_expand_label`, `derive_secret`, and a `KeySchedule`
//! state object that walks the early/handshake/master secret chain.

#![allow(clippy::needless_range_loop)]

use cv_crypto::hkdf;
use cv_crypto::hmac::{HmacSha256, HmacSha384};
use cv_crypto::sha256::Sha256;
use cv_crypto::sha384::Sha384;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum HashAlg {
    Sha256,
    Sha384,
}

impl HashAlg {
    pub fn output_len(self) -> usize {
        match self {
            Self::Sha256 => 32,
            Self::Sha384 => 48,
        }
    }

    pub fn hash(self, data: &[u8]) -> Vec<u8> {
        match self {
            Self::Sha256 => Sha256::oneshot(data).to_vec(),
            Self::Sha384 => Sha384::oneshot(data).to_vec(),
        }
    }

    /// HMAC with this hash. Used inside HKDF and Finished MACs.
    pub fn hmac(self, key: &[u8], data: &[u8]) -> Vec<u8> {
        match self {
            Self::Sha256 => HmacSha256::oneshot(key, data).to_vec(),
            Self::Sha384 => HmacSha384::oneshot(key, data).to_vec(),
        }
    }
}

/// RFC 8446 §7.1: HKDF-Expand-Label(Secret, Label, Context, Length).
pub fn hkdf_expand_label(
    alg: HashAlg,
    secret: &[u8],
    label: &[u8],
    context: &[u8],
    length: u16,
) -> Vec<u8> {
    let mut info = Vec::with_capacity(2 + 1 + 6 + label.len() + 1 + context.len());
    info.extend_from_slice(&length.to_be_bytes());
    let full_label = {
        let mut v = Vec::with_capacity(6 + label.len());
        v.extend_from_slice(b"tls13 ");
        v.extend_from_slice(label);
        v
    };
    assert!(full_label.len() <= 255);
    info.push(full_label.len() as u8);
    info.extend_from_slice(&full_label);
    assert!(context.len() <= 255);
    info.push(context.len() as u8);
    info.extend_from_slice(context);

    let mut out = vec![0u8; length as usize];
    match alg {
        HashAlg::Sha256 => hkdf::expand(secret, &info, &mut out),
        HashAlg::Sha384 => hkdf::expand_sha384(secret, &info, &mut out),
    }
    out
}

/// RFC 8446 §7.1: Derive-Secret(Secret, Label, Messages).
/// Hashes the transcript bytes and uses the hash as the context.
pub fn derive_secret(alg: HashAlg, secret: &[u8], label: &[u8], transcript: &[u8]) -> Vec<u8> {
    let h = alg.hash(transcript);
    hkdf_expand_label(alg, secret, label, &h, alg.output_len() as u16)
}

/// HKDF-Extract for the chosen hash. RFC 5869.
pub fn extract(alg: HashAlg, salt: &[u8], ikm: &[u8]) -> Vec<u8> {
    match alg {
        HashAlg::Sha256 => hkdf::extract(salt, ikm).to_vec(),
        HashAlg::Sha384 => hkdf::extract_sha384(salt, ikm).to_vec(),
    }
}

/// Convenience: the constant "all zeros" buffer used as default PSK / IKM.
pub fn zero_ikm(alg: HashAlg) -> Vec<u8> {
    vec![0u8; alg.output_len()]
}

/// Phase tracker for the TLS 1.3 key schedule.
pub struct KeySchedule {
    pub alg: HashAlg,
    pub early_secret: Vec<u8>,
    pub handshake_secret: Vec<u8>,
    pub master_secret: Vec<u8>,
}

impl std::fmt::Debug for KeySchedule {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("KeySchedule")
            .field("alg", &self.alg)
            .finish_non_exhaustive()
    }
}

impl KeySchedule {
    /// Step 1 (PSK-free): early_secret = HKDF-Extract(0, 0).
    pub fn new_no_psk(alg: HashAlg) -> Self {
        let zero = zero_ikm(alg);
        let early_secret = extract(alg, &zero, &zero);
        Self {
            alg,
            early_secret,
            handshake_secret: Vec::new(),
            master_secret: Vec::new(),
        }
    }

    /// Step 2: handshake_secret = HKDF-Extract(Derive-Secret(early, "derived", ""), ECDHE).
    pub fn advance_to_handshake(&mut self, ecdhe_shared: &[u8]) {
        let salt = derive_secret(self.alg, &self.early_secret, b"derived", b"");
        self.handshake_secret = extract(self.alg, &salt, ecdhe_shared);
    }

    /// Step 3: master_secret = HKDF-Extract(Derive-Secret(handshake, "derived", ""), 0).
    pub fn advance_to_master(&mut self) {
        let salt = derive_secret(self.alg, &self.handshake_secret, b"derived", b"");
        let zero = zero_ikm(self.alg);
        self.master_secret = extract(self.alg, &salt, &zero);
    }

    pub fn client_handshake_traffic_secret(&self, transcript: &[u8]) -> Vec<u8> {
        derive_secret(
            self.alg,
            &self.handshake_secret,
            b"c hs traffic",
            transcript,
        )
    }

    pub fn server_handshake_traffic_secret(&self, transcript: &[u8]) -> Vec<u8> {
        derive_secret(
            self.alg,
            &self.handshake_secret,
            b"s hs traffic",
            transcript,
        )
    }

    pub fn client_application_traffic_secret(&self, transcript: &[u8]) -> Vec<u8> {
        derive_secret(self.alg, &self.master_secret, b"c ap traffic", transcript)
    }

    pub fn server_application_traffic_secret(&self, transcript: &[u8]) -> Vec<u8> {
        derive_secret(self.alg, &self.master_secret, b"s ap traffic", transcript)
    }
}

/// Derive the AEAD key + IV for a given traffic secret. RFC 8446 §7.3.
pub struct TrafficKeys {
    pub key: Vec<u8>,
    pub iv: Vec<u8>,
}

impl std::fmt::Debug for TrafficKeys {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TrafficKeys")
            .field("key_len", &self.key.len())
            .field("iv_len", &self.iv.len())
            .finish()
    }
}

pub fn traffic_keys(
    alg: HashAlg,
    traffic_secret: &[u8],
    key_len: usize,
    iv_len: usize,
) -> TrafficKeys {
    TrafficKeys {
        key: hkdf_expand_label(alg, traffic_secret, b"key", b"", key_len as u16),
        iv: hkdf_expand_label(alg, traffic_secret, b"iv", b"", iv_len as u16),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hex(b: &[u8]) -> String {
        let mut s = String::new();
        for x in b {
            s.push_str(&format!("{x:02x}"));
        }
        s
    }

    /// RFC 8446 §7.1 — for SHA-256 with empty PSK,
    ///   early_secret = HKDF-Extract(0, 0) = 33ad0a1c607ec03b09e6cd9893680ce210adf300aa1f2660e1b22e10f170f92a
    #[test]
    fn early_secret_no_psk_sha256() {
        let ks = KeySchedule::new_no_psk(HashAlg::Sha256);
        assert_eq!(
            hex(&ks.early_secret),
            "33ad0a1c607ec03b09e6cd9893680ce210adf300aa1f2660e1b22e10f170f92a"
        );
    }

    /// RFC 8446 §7.1 — "derived" secret from early_secret with empty transcript:
    ///   Derive-Secret(early_secret, "derived", "") =
    ///   6f2615a108c702c5678f54fc9dbab69716c076189c48250cebeac3576c3611ba
    #[test]
    fn derived_from_early_sha256() {
        let ks = KeySchedule::new_no_psk(HashAlg::Sha256);
        let d = derive_secret(HashAlg::Sha256, &ks.early_secret, b"derived", b"");
        assert_eq!(
            hex(&d),
            "6f2615a108c702c5678f54fc9dbab69716c076189c48250cebeac3576c3611ba"
        );
    }
}
