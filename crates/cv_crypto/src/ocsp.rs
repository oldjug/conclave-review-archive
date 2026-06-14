//! OCSP — Online Certificate Status Protocol (RFC 6960).

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CertStatus {
    Good,
    Revoked { reason: u8, revocation_time_ms: u64 },
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OcspResponse {
    pub responder_id: Vec<u8>,
    pub cert_status: CertStatus,
    pub produced_at_ms: u64,
    pub this_update_ms: u64,
    pub next_update_ms: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OcspDecision {
    /// Certificate is valid per the responder.
    Allow,
    /// Responder said revoked. Block.
    Block,
    /// Responder said unknown or stale. Soft-fail.
    SoftFail,
}

pub fn evaluate(resp: &OcspResponse, now_ms: u64) -> OcspDecision {
    if now_ms > resp.next_update_ms {
        return OcspDecision::SoftFail;
    }
    match resp.cert_status {
        CertStatus::Good => OcspDecision::Allow,
        CertStatus::Revoked { .. } => OcspDecision::Block,
        CertStatus::Unknown => OcspDecision::SoftFail,
    }
}

/// CRL distribution-points - list of URLs the cert names for full
/// CRL retrieval. The fetch helper for those URLs lives in cv_net.
#[derive(Debug, Default)]
pub struct Crl {
    pub revoked_serials: Vec<Vec<u8>>,
}

impl Crl {
    pub fn contains(&self, serial: &[u8]) -> bool {
        self.revoked_serials.iter().any(|s| s == serial)
    }
}

// --------------- DER OCSP request builder (RFC 6960 §4.1) -------------
//
// Real DER encoder.  Produces the bytes the browser POSTs to the
// responder URL extracted from the cert's Authority Information
// Access extension.
//
//   OCSPRequest ::= SEQUENCE { tbsRequest TBSRequest }
//   TBSRequest  ::= SEQUENCE { requestList SEQUENCE OF Request }
//   Request     ::= SEQUENCE { reqCert CertID }
//   CertID      ::= SEQUENCE {
//       hashAlgorithm   AlgorithmIdentifier,  -- SHA-1
//       issuerNameHash  OCTET STRING,
//       issuerKeyHash   OCTET STRING,
//       serialNumber    INTEGER }

/// Encode an ASN.1 DER length (short or long form).
fn der_length(n: usize, out: &mut Vec<u8>) {
    if n < 0x80 {
        out.push(n as u8);
    } else {
        let mut buf = [0u8; 8];
        let mut i = 0;
        let mut x = n;
        while x > 0 {
            buf[i] = (x & 0xFF) as u8;
            i += 1;
            x >>= 8;
        }
        out.push(0x80 | i as u8);
        for j in (0..i).rev() {
            out.push(buf[j]);
        }
    }
}

fn der_tlv(tag: u8, value: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(value.len() + 6);
    out.push(tag);
    der_length(value.len(), &mut out);
    out.extend_from_slice(value);
    out
}

/// SHA-1 algorithm identifier (1.3.14.3.2.26) with NULL parameters,
/// DER-encoded.  This is the hash typically required by responders.
fn alg_id_sha1() -> Vec<u8> {
    // OID 1.3.14.3.2.26 in DER = 06 05 2B 0E 03 02 1A
    let oid = vec![0x06, 0x05, 0x2B, 0x0E, 0x03, 0x02, 0x1A];
    let null = vec![0x05, 0x00];
    let mut inner = oid;
    inner.extend(null);
    der_tlv(0x30, &inner) // SEQUENCE
}

/// DER-encode a positive INTEGER from raw bytes (big-endian).  Adds
/// a leading 0x00 if the high bit is set, to preserve sign.
fn der_integer(bytes: &[u8]) -> Vec<u8> {
    // Strip any leading zero bytes that aren't needed to disambiguate
    // from negative.
    let mut start = 0;
    while start < bytes.len() - 1 && bytes[start] == 0 && bytes[start + 1] & 0x80 == 0 {
        start += 1;
    }
    let stripped = &bytes[start..];
    if stripped[0] & 0x80 != 0 {
        // Prepend 0x00 so it's read as positive.
        let mut v = Vec::with_capacity(stripped.len() + 1);
        v.push(0x00);
        v.extend_from_slice(stripped);
        der_tlv(0x02, &v)
    } else {
        der_tlv(0x02, stripped)
    }
}

/// Build a DER-encoded OCSPRequest for a single certificate.
/// `issuer_name_hash` and `issuer_key_hash` must be SHA-1 of the
/// issuer Subject DN bytes and the issuer SubjectPublicKey bit-string
/// respectively, per RFC 6960 §4.1.1.
pub fn build_request(
    issuer_name_hash: &[u8; 20],
    issuer_key_hash: &[u8; 20],
    serial_be: &[u8],
) -> Vec<u8> {
    // CertID
    let mut cert_id_inner = Vec::new();
    cert_id_inner.extend(alg_id_sha1());
    cert_id_inner.extend(der_tlv(0x04, issuer_name_hash)); // OCTET STRING
    cert_id_inner.extend(der_tlv(0x04, issuer_key_hash)); // OCTET STRING
    cert_id_inner.extend(der_integer(serial_be)); // INTEGER
    let cert_id = der_tlv(0x30, &cert_id_inner);

    // Request ::= SEQUENCE { reqCert CertID }
    let request = der_tlv(0x30, &cert_id);

    // requestList SEQUENCE OF Request
    let request_list = der_tlv(0x30, &request);

    // TBSRequest ::= SEQUENCE { requestList }
    let tbs = der_tlv(0x30, &request_list);

    // OCSPRequest ::= SEQUENCE { tbsRequest }
    der_tlv(0x30, &tbs)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn good_resp() -> OcspResponse {
        OcspResponse {
            responder_id: vec![1, 2, 3],
            cert_status: CertStatus::Good,
            produced_at_ms: 0,
            this_update_ms: 0,
            next_update_ms: 10_000,
        }
    }

    #[test]
    fn good_status_inside_window_allows() {
        assert_eq!(evaluate(&good_resp(), 5_000), OcspDecision::Allow);
    }

    #[test]
    fn stale_response_softfails() {
        assert_eq!(evaluate(&good_resp(), 50_000), OcspDecision::SoftFail);
    }

    #[test]
    fn revoked_status_blocks() {
        let mut r = good_resp();
        r.cert_status = CertStatus::Revoked {
            reason: 1,
            revocation_time_ms: 0,
        };
        assert_eq!(evaluate(&r, 5_000), OcspDecision::Block);
    }

    #[test]
    fn unknown_status_softfails() {
        let mut r = good_resp();
        r.cert_status = CertStatus::Unknown;
        assert_eq!(evaluate(&r, 5_000), OcspDecision::SoftFail);
    }

    #[test]
    fn der_length_short_form_under_128() {
        let mut v = Vec::new();
        der_length(0x42, &mut v);
        assert_eq!(v, vec![0x42]);
    }

    #[test]
    fn der_length_long_form_encodes_byte_count() {
        let mut v = Vec::new();
        der_length(300, &mut v);
        // 300 = 0x012C → 0x82 0x01 0x2C
        assert_eq!(v, vec![0x82, 0x01, 0x2C]);
    }

    #[test]
    fn der_integer_handles_high_bit_serial() {
        // 0xFF should be encoded as 02 02 00 FF (extra 00 byte to
        // keep positive sign).
        let v = der_integer(&[0xFF]);
        assert_eq!(v, vec![0x02, 0x02, 0x00, 0xFF]);
    }

    #[test]
    fn der_integer_strips_unnecessary_leading_zero() {
        // 0x00 0x42 → 02 01 42 (leading 00 removed since high bit
        // is clear).
        let v = der_integer(&[0x00, 0x42]);
        assert_eq!(v, vec![0x02, 0x01, 0x42]);
    }

    #[test]
    fn ocsp_request_starts_with_outer_sequence() {
        let name_hash = [0xAA; 20];
        let key_hash = [0xBB; 20];
        let serial = [0x01, 0x02, 0x03, 0x04];
        let req = build_request(&name_hash, &key_hash, &serial);
        // First byte is SEQUENCE tag.
        assert_eq!(req[0], 0x30);
        // Length byte (or first byte of long-form length).
        assert!(req.len() > 60); // 20 + 20 + algid + serial + wrappers
    }

    #[test]
    fn ocsp_request_contains_hashes_and_serial() {
        let name_hash = [0xAA; 20];
        let key_hash = [0xBB; 20];
        let serial = [0x12, 0x34, 0x56];
        let req = build_request(&name_hash, &key_hash, &serial);
        // Searching for the two octet strings + integer payload.
        assert!(req.windows(20).any(|w| w == name_hash));
        assert!(req.windows(20).any(|w| w == key_hash));
        assert!(req.windows(3).any(|w| w == serial));
    }

    #[test]
    fn ocsp_request_sha1_algorithm_oid_present() {
        let req = build_request(&[0; 20], &[0; 20], &[1]);
        // OID 1.3.14.3.2.26 (SHA-1) DER bytes: 06 05 2B 0E 03 02 1A
        let sha1_oid = [0x06, 0x05, 0x2B, 0x0E, 0x03, 0x02, 0x1A];
        assert!(req.windows(7).any(|w| w == sha1_oid));
    }

    #[test]
    fn crl_contains_serial() {
        let mut crl = Crl::default();
        crl.revoked_serials.push(vec![0xAA, 0xBB]);
        assert!(crl.contains(&[0xAA, 0xBB]));
        assert!(!crl.contains(&[0xCC]));
    }
}
