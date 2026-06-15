//! Certificate Transparency (RFC 6962 + RFC 9162).
//!
//! V1 surfaces the SignedCertificateTimestamp data model + the
//! CT-log-list verification policy. Real signature verification
//! routes through `crate::p256`.

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SignedCertificateTimestamp {
    pub log_id: [u8; 32],
    pub timestamp_ms: u64,
    pub extensions: Vec<u8>,
    pub signature: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CtLog {
    pub log_id: [u8; 32],
    pub key_pem: String,
    pub trusted: bool,
}

#[derive(Debug, Default, Clone)]
pub struct CtPolicy {
    pub logs: Vec<CtLog>,
    /// How many SCTs from distinct trusted logs are required.
    pub min_scts: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CtDecision {
    Compliant,
    NotEnoughScts,
    UntrustedLog,
}

pub fn evaluate(policy: &CtPolicy, scts: &[SignedCertificateTimestamp]) -> CtDecision {
    let mut from_trusted = 0u32;
    let mut seen_ids = std::collections::HashSet::new();
    for sct in scts {
        if let Some(log) = policy.logs.iter().find(|l| l.log_id == sct.log_id) {
            if log.trusted && seen_ids.insert(sct.log_id) {
                from_trusted += 1;
            }
        } else {
            return CtDecision::UntrustedLog;
        }
    }
    if from_trusted >= policy.min_scts {
        CtDecision::Compliant
    } else {
        CtDecision::NotEnoughScts
    }
}

/// Verify an SCT's ECDSA-P256-SHA256 signature against the log's public
/// key per RFC 6962 §3.2. The caller supplies the canonical TBS bytes
/// that were signed (TLS-encoded SignedCertificateTimestamp.signed_data).
/// `log_pubkey_uncompressed` is the 65-byte SEC1 uncompressed encoding
/// (0x04 || X || Y) of the log's P-256 public key.
///
/// Returns true on a valid signature, false on invalid signature OR
/// wrong key format. This is the building block CT enforcement calls
/// after locating the matching log entry in the bundled CtLog list.
pub fn verify_sct_signature(
    sct: &SignedCertificateTimestamp,
    signed_data: &[u8],
    log_pubkey_uncompressed: &[u8],
) -> bool {
    // Parse the TLS digitally-signed structure inside `sct.signature`:
    //   struct { SignatureAndHashAlgorithm algorithm; opaque sig<0..2^16-1>; }
    // First two bytes are the algorithm pair (hash || sig). RFC 6962 §3.2
    // mandates SHA-256 + ECDSA (algorithm = 0x04, 0x03).
    if sct.signature.len() < 4 {
        return false;
    }
    let hash_alg = sct.signature[0];
    let sig_alg = sct.signature[1];
    let sig_len = u16::from_be_bytes([sct.signature[2], sct.signature[3]]) as usize;
    if hash_alg != 4 || sig_alg != 3 {
        return false; // require SHA-256 + ECDSA
    }
    if sct.signature.len() < 4 + sig_len {
        return false;
    }
    let der_sig = &sct.signature[4..4 + sig_len];
    // Parse the ECDSA-Sig-Value DER: SEQUENCE { r INTEGER, s INTEGER }.
    let (r, s) = match decode_ecdsa_signature(der_sig) {
        Some(rs) => rs,
        None => return false,
    };
    // Parse the uncompressed public key (0x04 || X || Y).
    if log_pubkey_uncompressed.len() != 65 || log_pubkey_uncompressed[0] != 0x04 {
        return false;
    }
    let mut x = [0u8; 32];
    let mut y = [0u8; 32];
    x.copy_from_slice(&log_pubkey_uncompressed[1..33]);
    y.copy_from_slice(&log_pubkey_uncompressed[33..65]);
    // SHA-256 over the canonical signed data.
    let mut h = crate::sha256::Sha256::new();
    h.update(signed_data);
    let digest = h.finalize();
    // Verify via the P-256 ECDSA routine — the verifier reduces the
    // digest mod n internally and returns Ok on a valid signature.
    crate::p256::verify(&x, &y, &digest, &r, &s).is_ok()
}

/// Parse a `SignedCertificateTimestampList` (RFC 6962 §3.3 / RFC 5246
/// digitally-signed framing). The TLS `signed_certificate_timestamp`
/// extension body — and the SCT-list OCTET STRING embedded in a cert /
/// OCSP response — is:
///
///   struct {
///     SerializedSCT sct_list<1..2^16-1>;   // outer u16 length
///   };
///   opaque SerializedSCT<1..2^16-1>;        // each: u16 length + body
///
/// Each `SerializedSCT` body is a `SignedCertificateTimestamp`:
///
///   struct {
///     Version sct_version;          // 1 byte (v1 == 0)
///     LogID id;                     // 32 bytes (SHA-256 of log key)
///     uint64 timestamp;            // 8 bytes, ms since epoch
///     CtExtensions extensions;     // u16 length + body
///     digitally-signed { ... } signature; // 2-byte alg + u16 len + sig
///   };
///
/// Returns every SCT we can fully parse; a malformed trailing entry stops
/// the walk without discarding the SCTs already parsed.
pub fn parse_sct_list(body: &[u8]) -> Vec<SignedCertificateTimestamp> {
    let mut out = Vec::new();
    if body.len() < 2 {
        return out;
    }
    let list_len = u16::from_be_bytes([body[0], body[1]]) as usize;
    let end = (2 + list_len).min(body.len());
    let mut i = 2usize;
    while i + 2 <= end {
        let sct_len = u16::from_be_bytes([body[i], body[i + 1]]) as usize;
        i += 2;
        if i + sct_len > end {
            break;
        }
        if let Some(sct) = parse_one_sct(&body[i..i + sct_len]) {
            out.push(sct);
        }
        i += sct_len;
    }
    out
}

fn parse_one_sct(b: &[u8]) -> Option<SignedCertificateTimestamp> {
    // version(1) + log_id(32) + timestamp(8) + ext_len(2) = 43 minimum.
    if b.len() < 43 {
        return None;
    }
    let _version = b[0];
    let mut log_id = [0u8; 32];
    log_id.copy_from_slice(&b[1..33]);
    let timestamp_ms = u64::from_be_bytes(b[33..41].try_into().ok()?);
    let ext_len = u16::from_be_bytes([b[41], b[42]]) as usize;
    let mut p = 43;
    if p + ext_len > b.len() {
        return None;
    }
    let extensions = b[p..p + ext_len].to_vec();
    p += ext_len;
    // The remainder is the digitally-signed `signature` struct: the
    // verify path re-parses the 2-byte alg pair + u16 length internally,
    // so keep the bytes verbatim.
    let signature = b[p..].to_vec();
    Some(SignedCertificateTimestamp {
        log_id,
        timestamp_ms,
        extensions,
        signature,
    })
}

/// Parse a DER `ECDSA-Sig-Value ::= SEQUENCE { r INTEGER, s INTEGER }`
/// into 32-byte big-endian (r, s) pairs. Handles the standard ASN.1
/// leading-zero stripping for unsigned integers.
fn decode_ecdsa_signature(der: &[u8]) -> Option<([u8; 32], [u8; 32])> {
    let mut i = 0;
    if der.get(i)? != &0x30 {
        return None;
    }
    i += 1;
    // Length: short form (high bit clear) or long form. CT SCTs use the
    // short form because two 32-byte integers fit in well under 128 bytes.
    let total_len = der.get(i).copied()? as usize;
    i += 1;
    if i + total_len > der.len() {
        return None;
    }
    fn take_int(buf: &[u8], i: &mut usize) -> Option<[u8; 32]> {
        if buf.get(*i)? != &0x02 {
            return None;
        }
        *i += 1;
        let len = buf.get(*i).copied()? as usize;
        *i += 1;
        if *i + len > buf.len() {
            return None;
        }
        let mut raw = &buf[*i..*i + len];
        *i += len;
        // Strip leading 0x00 used to mark the integer as positive.
        if raw.len() > 32 && raw[0] == 0x00 {
            raw = &raw[1..];
        }
        if raw.len() > 32 {
            return None;
        }
        let mut out = [0u8; 32];
        out[32 - raw.len()..].copy_from_slice(raw);
        Some(out)
    }
    let r = take_int(der, &mut i)?;
    let s = take_int(der, &mut i)?;
    Some((r, s))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn log(id: u8, trusted: bool) -> CtLog {
        let mut log_id = [0u8; 32];
        log_id[0] = id;
        CtLog {
            log_id,
            key_pem: "PEM".into(),
            trusted,
        }
    }

    fn sct(log_id_first_byte: u8, ts: u64) -> SignedCertificateTimestamp {
        let mut id = [0u8; 32];
        id[0] = log_id_first_byte;
        SignedCertificateTimestamp {
            log_id: id,
            timestamp_ms: ts,
            extensions: Vec::new(),
            signature: vec![1, 2, 3],
        }
    }

    #[test]
    fn evaluate_compliant_when_min_scts_met() {
        let mut p = CtPolicy::default();
        p.logs.push(log(1, true));
        p.logs.push(log(2, true));
        p.min_scts = 2;
        let scts = vec![sct(1, 0), sct(2, 0)];
        assert_eq!(evaluate(&p, &scts), CtDecision::Compliant);
    }

    #[test]
    fn evaluate_insufficient_when_below_threshold() {
        let mut p = CtPolicy::default();
        p.logs.push(log(1, true));
        p.min_scts = 3;
        let scts = vec![sct(1, 0)];
        assert_eq!(evaluate(&p, &scts), CtDecision::NotEnoughScts);
    }

    #[test]
    fn evaluate_blocks_untrusted_log() {
        let mut p = CtPolicy::default();
        p.logs.push(log(1, false));
        p.min_scts = 1;
        let scts = vec![sct(99, 0)];
        assert_eq!(evaluate(&p, &scts), CtDecision::UntrustedLog);
    }

    #[test]
    fn duplicate_log_counts_once() {
        let mut p = CtPolicy::default();
        p.logs.push(log(1, true));
        p.min_scts = 2;
        let scts = vec![sct(1, 0), sct(1, 100)];
        assert_eq!(evaluate(&p, &scts), CtDecision::NotEnoughScts);
    }

    // ---- SignedCertificateTimestampList parser (RFC 6962 §3.3) ----------

    /// Serialize one SCT into a SerializedSCT body (without the outer u16
    /// SerializedSCT length prefix).
    fn serialize_sct_body(log_id_byte: u8, ts: u64, sig: &[u8]) -> Vec<u8> {
        let mut b = Vec::new();
        b.push(0); // version v1
        let mut id = [0u8; 32];
        id[0] = log_id_byte;
        b.extend_from_slice(&id);
        b.extend_from_slice(&ts.to_be_bytes());
        b.extend_from_slice(&[0x00, 0x00]); // empty extensions
        b.extend_from_slice(sig); // digitally-signed signature struct
        b
    }

    fn build_sct_list(scts: &[(u8, u64, Vec<u8>)]) -> Vec<u8> {
        let mut inner = Vec::new();
        for (id, ts, sig) in scts {
            let body = serialize_sct_body(*id, *ts, sig);
            inner.extend_from_slice(&(body.len() as u16).to_be_bytes());
            inner.extend_from_slice(&body);
        }
        let mut out = Vec::new();
        out.extend_from_slice(&(inner.len() as u16).to_be_bytes());
        out.extend_from_slice(&inner);
        out
    }

    #[test]
    fn parse_two_scts_from_tls_extension_body() {
        // A digitally-signed signature struct: alg(0x04,0x03) + u16 len + sig.
        let sig = vec![0x04, 0x03, 0x00, 0x02, 0xAB, 0xCD];
        let list = build_sct_list(&[(1, 100, sig.clone()), (2, 200, sig.clone())]);
        let parsed = parse_sct_list(&list);
        assert_eq!(parsed.len(), 2);
        assert_eq!(parsed[0].log_id[0], 1);
        assert_eq!(parsed[0].timestamp_ms, 100);
        assert_eq!(parsed[1].log_id[0], 2);
        assert_eq!(parsed[1].signature, sig);
    }

    #[test]
    fn parsed_scts_drive_ct_policy_decision() {
        // End-to-end: parse the wire bytes, then feed them to `evaluate`.
        let sig = vec![0x04, 0x03, 0x00, 0x01, 0x00];
        let list = build_sct_list(&[(1, 0, sig.clone()), (2, 0, sig.clone())]);
        let scts = parse_sct_list(&list);
        let mut p = CtPolicy::default();
        p.logs.push(log(1, true));
        p.logs.push(log(2, true));
        p.min_scts = 2;
        assert_eq!(evaluate(&p, &scts), CtDecision::Compliant);

        // A cert that presents zero SCTs is NotEnoughScts under the same policy.
        assert_eq!(evaluate(&p, &[]), CtDecision::NotEnoughScts);
    }

    #[test]
    fn parse_empty_and_truncated_lists_are_safe() {
        assert!(parse_sct_list(&[]).is_empty());
        assert!(parse_sct_list(&[0x00, 0x00]).is_empty());
        // Declares a 100-byte list but supplies none → no panic, no SCTs.
        assert!(parse_sct_list(&[0x00, 0x64]).is_empty());
    }
}
