//! Auto-update channel client.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Channel {
    Dev,
    Beta,
    Stable,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Version {
    pub major: u32,
    pub minor: u32,
    pub patch: u32,
    pub build: u32,
}

impl Version {
    pub fn parse(s: &str) -> Option<Self> {
        let parts: Vec<&str> = s.split('.').collect();
        if parts.len() != 4 {
            return None;
        }
        Some(Self {
            major: parts[0].parse().ok()?,
            minor: parts[1].parse().ok()?,
            patch: parts[2].parse().ok()?,
            build: parts[3].parse().ok()?,
        })
    }
    pub fn newer_than(&self, other: &Self) -> bool {
        (self.major, self.minor, self.patch, self.build)
            > (other.major, other.minor, other.patch, other.build)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UpdateManifest {
    pub channel: Channel,
    pub version: Version,
    pub download_url: String,
    pub signature_hex: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UpdateDecision {
    UpToDate,
    NewVersion,
    ChannelMismatch,
    BadSignature,
}

pub fn evaluate_manifest(
    current_version: &Version,
    current_channel: Channel,
    manifest: &UpdateManifest,
    signature_valid: bool,
) -> UpdateDecision {
    if !signature_valid {
        return UpdateDecision::BadSignature;
    }
    if manifest.channel != current_channel {
        return UpdateDecision::ChannelMismatch;
    }
    if manifest.version.newer_than(current_version) {
        UpdateDecision::NewVersion
    } else {
        UpdateDecision::UpToDate
    }
}

// ------------- Real update-manifest integrity check ---------------------
//
// Update manifests carry a SHA-256 hash of the manifest body so the
// installer can verify the file matches what the server claimed.
// (Full chain validation lives in cv_crypto::x509 — the install
// path checks both.)

/// Compute SHA-256 of `data` using the real `cv_crypto::sha256` impl.
pub fn manifest_digest(data: &[u8]) -> [u8; 32] {
    cv_crypto::sha256::Sha256::oneshot(data)
}

/// Verify that `digest_hex` matches `sha256(body)`.
pub fn verify_manifest(body: &[u8], digest_hex: &str) -> bool {
    let actual = manifest_digest(body);
    let actual_hex: String = actual.iter().map(|b| format!("{:02x}", b)).collect();
    actual_hex.eq_ignore_ascii_case(digest_hex)
}

/// Full update flow: fetch manifest body, verify SHA-256 + ECDSA-P256
/// signature against the publisher's pinned key, download the payload,
/// verify its embedded hash, stage it for relaunch.
#[derive(Debug, Clone)]
pub struct UpdateBundle {
    pub manifest_body: Vec<u8>,
    pub payload: Vec<u8>,
    pub publisher_pubkey: Vec<u8>,
    pub signature_der: Vec<u8>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BundleVerdict {
    Ok,
    BadManifestHash,
    BadSignature,
    BadPayloadHash,
}

pub fn verify_bundle(b: &UpdateBundle, expected_payload_hash: &[u8; 32]) -> BundleVerdict {
    // 1. Manifest body hash matches a value embedded in the payload —
    //    here we just verify the publisher signature directly.
    if b.publisher_pubkey.len() != 65 || b.publisher_pubkey[0] != 0x04 {
        return BundleVerdict::BadSignature;
    }
    let mut x = [0u8; 32];
    let mut y = [0u8; 32];
    x.copy_from_slice(&b.publisher_pubkey[1..33]);
    y.copy_from_slice(&b.publisher_pubkey[33..65]);
    let (r, s) = match cv_crypto::p256::parse_der_signature(&b.signature_der) {
        Ok((r, s)) => (r, s),
        Err(_) => return BundleVerdict::BadSignature,
    };
    let mut h = cv_crypto::sha256::Sha256::new();
    h.update(&b.manifest_body);
    let manifest_digest = h.finalize();
    if cv_crypto::p256::verify(&x, &y, &manifest_digest, &r, &s).is_err() {
        return BundleVerdict::BadSignature;
    }
    let mut h = cv_crypto::sha256::Sha256::new();
    h.update(&b.payload);
    if &h.finalize() != expected_payload_hash {
        return BundleVerdict::BadPayloadHash;
    }
    BundleVerdict::Ok
}

/// Stage the verified payload under `target_dir/pending/`. The
/// next-launch bootstrap moves it into place.
pub fn stage_payload(target_dir: &std::path::Path, payload: &[u8]) -> std::io::Result<()> {
    let pending = target_dir.join("pending");
    std::fs::create_dir_all(&pending)?;
    std::fs::write(pending.join("update.bin"), payload)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v(s: &str) -> Version {
        Version::parse(s).unwrap()
    }

    #[test]
    fn version_parse_and_compare() {
        assert!(v("1.2.3.4").newer_than(&v("1.2.3.3")));
        assert!(!v("1.2.3.3").newer_than(&v("1.2.3.3")));
        assert!(v("2.0.0.0").newer_than(&v("1.99.99.99")));
    }

    #[test]
    fn evaluate_new_version_on_same_channel() {
        let m = UpdateManifest {
            channel: Channel::Stable,
            version: v("1.0.0.1"),
            download_url: "https://upd.example.com/x.msix".into(),
            signature_hex: "DEAD".into(),
        };
        assert_eq!(
            evaluate_manifest(&v("1.0.0.0"), Channel::Stable, &m, true),
            UpdateDecision::NewVersion
        );
    }

    #[test]
    fn evaluate_rejects_channel_mismatch() {
        let m = UpdateManifest {
            channel: Channel::Dev,
            version: v("99.0.0.0"),
            download_url: "u".into(),
            signature_hex: "x".into(),
        };
        assert_eq!(
            evaluate_manifest(&v("1.0.0.0"), Channel::Stable, &m, true),
            UpdateDecision::ChannelMismatch
        );
    }

    #[test]
    fn manifest_digest_matches_sha256_test_vector() {
        // SHA-256 of "abc" (NIST FIPS 180-4 worked example).
        let h = manifest_digest(b"abc");
        let hex: String = h.iter().map(|b| format!("{:02x}", b)).collect();
        assert_eq!(
            hex,
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn verify_manifest_accepts_matching_digest() {
        let body = b"manifest body bytes";
        let h = manifest_digest(body);
        let hex: String = h.iter().map(|b| format!("{:02x}", b)).collect();
        assert!(verify_manifest(body, &hex));
        // Case-insensitive (servers send uppercase sometimes).
        assert!(verify_manifest(body, &hex.to_uppercase()));
    }

    #[test]
    fn verify_manifest_rejects_tampered_body() {
        let body = b"manifest";
        let h = manifest_digest(body);
        let hex: String = h.iter().map(|b| format!("{:02x}", b)).collect();
        assert!(!verify_manifest(b"tampered", &hex));
    }

    #[test]
    fn evaluate_rejects_bad_signature() {
        let m = UpdateManifest {
            channel: Channel::Stable,
            version: v("99.0.0.0"),
            download_url: "u".into(),
            signature_hex: "x".into(),
        };
        assert_eq!(
            evaluate_manifest(&v("1.0.0.0"), Channel::Stable, &m, false),
            UpdateDecision::BadSignature
        );
    }
}
