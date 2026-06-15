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

// --------------- DER OCSP response parser (RFC 6960 §4.2) -------------
//
// The stapled response a server sends in the TLS CertificateStatus
// message is a DER `OCSPResponse`:
//
//   OCSPResponse ::= SEQUENCE {
//       responseStatus   OCSPResponseStatus,  -- ENUMERATED
//       responseBytes    [0] EXPLICIT ResponseBytes OPTIONAL }
//   ResponseBytes ::= SEQUENCE {
//       responseType   OBJECT IDENTIFIER,     -- id-pkix-ocsp-basic
//       response       OCTET STRING }         -- DER BasicOCSPResponse
//   BasicOCSPResponse ::= SEQUENCE {
//       tbsResponseData  ResponseData,
//       signatureAlgorithm AlgorithmIdentifier,
//       signature        BIT STRING,
//       certs            [0] EXPLICIT SEQUENCE OF Certificate OPTIONAL }
//   ResponseData ::= SEQUENCE {
//       version          [0] EXPLICIT Version DEFAULT v1,
//       responderID      ResponderID,
//       producedAt       GeneralizedTime,
//       responses        SEQUENCE OF SingleResponse,
//       responseExtensions [1] EXPLICIT Extensions OPTIONAL }
//   SingleResponse ::= SEQUENCE {
//       certID           CertID,
//       certStatus       CertStatus,
//       thisUpdate       GeneralizedTime,
//       nextUpdate       [0] EXPLICIT GeneralizedTime OPTIONAL,
//       singleExtensions [1] EXPLICIT Extensions OPTIONAL }
//   CertStatus ::= CHOICE {
//       good     [0] IMPLICIT NULL,
//       revoked  [1] IMPLICIT RevokedInfo,
//       unknown  [2] IMPLICIT UnknownInfo }
//   RevokedInfo ::= SEQUENCE {
//       revocationTime   GeneralizedTime,
//       revocationReason [0] EXPLICIT CRLReason OPTIONAL }

use crate::asn1::Reader as AsnReader;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OcspParseError {
    Asn1(String),
    /// responseStatus != 0 (successful). The byte is the status value.
    ResponseStatus(u8),
    /// responseType OID was not id-pkix-ocsp-basic (1.3.6.1.5.5.7.48.1.1).
    NotBasicResponse,
    /// The response carried no SingleResponse entries.
    NoSingleResponse,
}

impl core::fmt::Display for OcspParseError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Asn1(s) => write!(f, "ocsp asn.1: {s}"),
            Self::ResponseStatus(s) => write!(f, "ocsp responseStatus={s} (not successful)"),
            Self::NotBasicResponse => f.write_str("ocsp response is not id-pkix-ocsp-basic"),
            Self::NoSingleResponse => f.write_str("ocsp response has no SingleResponse"),
        }
    }
}

impl std::error::Error for OcspParseError {}

/// id-pkix-ocsp-basic = 1.3.6.1.5.5.7.48.1.1 (DER OID bytes).
const ID_PKIX_OCSP_BASIC: &[u8] = &[0x2B, 0x06, 0x01, 0x05, 0x05, 0x07, 0x30, 0x01, 0x01];

fn asn<E>(e: E) -> OcspParseError
where
    E: core::fmt::Display,
{
    OcspParseError::Asn1(e.to_string())
}

/// Parse a stapled DER `OCSPResponse` and return the first SingleResponse's
/// status + validity window. Returns an error when the response is not a
/// successful BasicOCSPResponse — the caller decides whether that hard-fails
/// or soft-fails. This is the real RFC 6960 §4.2.1 decoder; it does NOT
/// verify the responder's signature (the stapled response is delivered
/// inside the authenticated TLS handshake, and Windows' chain engine
/// performs full revocation checking in `chain_validate`; this decode is
/// the staple-revocation hard-fail Chrome applies on top).
pub fn parse_basic_response(der: &[u8]) -> Result<OcspResponse, OcspParseError> {
    let mut top = AsnReader::new(der);
    let mut ocsp_resp = top.read_sequence().map_err(asn)?;

    // responseStatus ::= ENUMERATED (tag 0x0A). 0 == successful.
    let status_bytes = ocsp_resp.read_tagged(0x0A).map_err(asn)?;
    let status = status_bytes.last().copied().unwrap_or(0xFF);
    if status != 0 {
        return Err(OcspParseError::ResponseStatus(status));
    }

    // responseBytes ::= [0] EXPLICIT ResponseBytes.
    let response_bytes = ocsp_resp.read_tagged(0xA0).map_err(asn)?;
    let mut rb = AsnReader::new(response_bytes);
    let mut rb_seq = rb.read_sequence().map_err(asn)?;
    let response_type = rb_seq.read_oid().map_err(asn)?;
    if response_type.0 != ID_PKIX_OCSP_BASIC {
        return Err(OcspParseError::NotBasicResponse);
    }
    let basic_der = rb_seq.read_octet_string().map_err(asn)?;

    // BasicOCSPResponse ::= SEQUENCE { tbsResponseData, ... }
    let mut basic = AsnReader::new(basic_der);
    let mut basic_seq = basic.read_sequence().map_err(asn)?;
    let mut tbs = basic_seq.read_sequence().map_err(asn)?;

    // version [0] EXPLICIT (optional, default v1) — skip if present.
    if tbs.peek_tag() == Some(0xA0) {
        tbs.read_tagged(0xA0).map_err(asn)?;
    }
    // responderID ::= CHOICE { [1] Name, [2] KeyHash } — capture raw bytes.
    let (_tag, responder_id, _full) = tbs.read_any().map_err(asn)?;
    // producedAt ::= GeneralizedTime (tag 0x18).
    let produced_at = tbs.read_tagged(0x18).map_err(asn)?;
    let produced_at_ms = parse_generalized_time_ms(produced_at).unwrap_or(0);

    // responses ::= SEQUENCE OF SingleResponse.
    let mut responses = tbs.read_sequence().map_err(asn)?;
    let mut single = responses.read_sequence().map_err(asn)?;
    if single.is_empty() {
        return Err(OcspParseError::NoSingleResponse);
    }

    // certID ::= SEQUENCE — skip (we trust the staple's single response
    // applies to the leaf; a multi-cert staple would need certID match).
    single.read_sequence().map_err(asn)?;

    // certStatus ::= CHOICE { [0] good NULL, [1] revoked, [2] unknown }.
    let (cs_tag, cs_value, _cs_full) = single.read_any().map_err(asn)?;
    let cert_status = match cs_tag {
        // [0] good IMPLICIT NULL.
        0x80 => CertStatus::Good,
        // [1] revoked IMPLICIT RevokedInfo SEQUENCE.
        0xA1 => {
            let mut ri = AsnReader::new(cs_value);
            let rt = ri.read_tagged(0x18).map_err(asn)?;
            let revocation_time_ms = parse_generalized_time_ms(rt).unwrap_or(0);
            // revocationReason [0] EXPLICIT CRLReason OPTIONAL.
            let reason = if ri.peek_tag() == Some(0xA0) {
                let r = ri.read_tagged(0xA0).map_err(asn)?;
                // [0] EXPLICIT { ENUMERATED } — last byte is the reason.
                r.last().copied().unwrap_or(0)
            } else {
                0
            };
            CertStatus::Revoked {
                reason,
                revocation_time_ms,
            }
        }
        // [2] unknown IMPLICIT NULL.
        _ => CertStatus::Unknown,
    };

    // thisUpdate ::= GeneralizedTime.
    let this_update = single.read_tagged(0x18).map_err(asn)?;
    let this_update_ms = parse_generalized_time_ms(this_update).unwrap_or(0);

    // nextUpdate ::= [0] EXPLICIT GeneralizedTime OPTIONAL.
    let next_update_ms = if single.peek_tag() == Some(0xA0) {
        let nu_explicit = single.read_tagged(0xA0).map_err(asn)?;
        let mut nu = AsnReader::new(nu_explicit);
        let nu_time = nu.read_tagged(0x18).map_err(asn)?;
        parse_generalized_time_ms(nu_time).unwrap_or(u64::MAX)
    } else {
        // No nextUpdate ⇒ response is good "now"; treat as far-future so
        // `evaluate` doesn't soft-fail a response with no stated expiry.
        u64::MAX
    };

    Ok(OcspResponse {
        responder_id: responder_id.to_vec(),
        cert_status,
        produced_at_ms,
        this_update_ms,
        next_update_ms,
    })
}

/// Parse an ASN.1 `GeneralizedTime` (`YYYYMMDDHHMMSSZ`, RFC 5280 §4.1.2.5.2)
/// into Unix milliseconds. Only the canonical UTC `Z`-terminated form DER
/// mandates is accepted; fractional seconds and offsets (forbidden in DER)
/// return None.
pub fn parse_generalized_time_ms(bytes: &[u8]) -> Option<u64> {
    // Minimum: YYYYMMDDHHMMSSZ = 15 chars.
    if bytes.len() < 15 || *bytes.last()? != b'Z' {
        return None;
    }
    let s = core::str::from_utf8(&bytes[..14]).ok()?;
    let year: i32 = s.get(0..4)?.parse().ok()?;
    let month: u32 = s.get(4..6)?.parse().ok()?;
    let day: u32 = s.get(6..8)?.parse().ok()?;
    let hour: u32 = s.get(8..10)?.parse().ok()?;
    let minute: u32 = s.get(10..12)?.parse().ok()?;
    let second: u32 = s.get(12..14)?.parse().ok()?;
    let secs = civil_to_unix_secs(year, month, day, hour, minute, second)?;
    Some(secs.checked_mul(1000)?)
}

/// Howard Hinnant days-from-civil → Unix seconds (UTC). Exact for every
/// date X.509 / OCSP carries. Returns None on out-of-range fields or a
/// pre-epoch result (OCSP times are always post-epoch).
fn civil_to_unix_secs(
    year: i32,
    month: u32,
    day: u32,
    hour: u32,
    minute: u32,
    second: u32,
) -> Option<u64> {
    if !(1..=12).contains(&month) || !(1..=31).contains(&day) || hour > 23 || minute > 59 || second > 60 {
        return None;
    }
    let shifted_year = if month <= 2 { year - 1 } else { year };
    let shifted_month: u32 = if month <= 2 { month + 9 } else { month - 3 };
    let era = if shifted_year >= 0 {
        shifted_year / 400
    } else {
        (shifted_year - 399) / 400
    };
    let yoe = (shifted_year - era * 400) as u32;
    let doy = (153 * shifted_month + 2) / 5 + day - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    let days_from_epoch = era as i64 * 146097 + doe as i64 - 719468;
    let total = days_from_epoch * 86_400
        + hour as i64 * 3_600
        + minute as i64 * 60
        + second as i64;
    if total < 0 {
        return None;
    }
    Some(total as u64)
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

    // ---- DER OCSPResponse parser (RFC 6960 §4.2) ------------------------

    /// Build a minimal-but-real DER OCSPResponse with the given certStatus
    /// CHOICE bytes (`good` = `[0x80,0x00]`, etc.). Wraps it through the
    /// full OCSPResponse → ResponseBytes → BasicOCSPResponse → ResponseData
    /// → SingleResponse nesting so the parser exercises every layer.
    fn build_ocsp_der(cert_status: &[u8], with_next_update: bool) -> Vec<u8> {
        // GeneralizedTime YYYYMMDDHHMMSSZ.
        let this_update = der_tlv(0x18, b"20240101000000Z");
        let next_update = der_tlv(0x18, b"20990101000000Z");
        // certID: minimal SEQUENCE { algid, name-hash, key-hash, serial }.
        let cert_id = der_tlv(
            0x30,
            &[
                alg_id_sha1(),
                der_tlv(0x04, &[0u8; 20]),
                der_tlv(0x04, &[0u8; 20]),
                der_integer(&[0x01]),
            ]
            .concat(),
        );
        let mut single_inner = cert_id;
        single_inner.extend_from_slice(cert_status);
        single_inner.extend_from_slice(&this_update);
        if with_next_update {
            // nextUpdate [0] EXPLICIT GeneralizedTime.
            single_inner.extend_from_slice(&der_tlv(0xA0, &next_update));
        }
        let single = der_tlv(0x30, &single_inner);
        let responses = der_tlv(0x30, &single); // SEQUENCE OF SingleResponse

        // ResponseData: responderID [1] KeyHash + producedAt + responses.
        let responder_id = der_tlv(0xA1, &der_tlv(0x04, &[0xAB; 20]));
        let produced_at = der_tlv(0x18, b"20240101000000Z");
        let mut rd_inner = responder_id;
        rd_inner.extend_from_slice(&produced_at);
        rd_inner.extend_from_slice(&responses);
        let response_data = der_tlv(0x30, &rd_inner);

        // BasicOCSPResponse: tbs + sigalg + signature BIT STRING.
        let sig_alg = der_tlv(0x30, &[0x06, 0x05, 0x2B, 0x0E, 0x03, 0x02, 0x1A]);
        let signature = der_tlv(0x03, &[0x00, 0xDE, 0xAD]); // dummy
        let mut basic_inner = response_data;
        basic_inner.extend_from_slice(&sig_alg);
        basic_inner.extend_from_slice(&signature);
        let basic = der_tlv(0x30, &basic_inner);

        // ResponseBytes: responseType OID (id-pkix-ocsp-basic) + response OCTET STRING.
        let response_type = der_tlv(0x06, ID_PKIX_OCSP_BASIC);
        let response_octet = der_tlv(0x04, &basic);
        let mut rb_inner = response_type;
        rb_inner.extend_from_slice(&response_octet);
        let response_bytes = der_tlv(0x30, &rb_inner);
        let response_bytes_explicit = der_tlv(0xA0, &response_bytes);

        // OCSPResponse: responseStatus ENUMERATED(0) + [0] ResponseBytes.
        let status = der_tlv(0x0A, &[0x00]); // successful
        let mut top_inner = status;
        top_inner.extend_from_slice(&response_bytes_explicit);
        der_tlv(0x30, &top_inner)
    }

    #[test]
    fn parse_good_staple_decodes_status_good() {
        let der = build_ocsp_der(&[0x80, 0x00], true); // [0] good NULL
        let resp = parse_basic_response(&der).expect("good staple parses");
        assert_eq!(resp.cert_status, CertStatus::Good);
        assert!(resp.next_update_ms > resp.this_update_ms);
    }

    #[test]
    fn parse_revoked_staple_decodes_status_revoked() {
        // [1] revoked RevokedInfo { revocationTime, reason [0] EXPLICIT }.
        let rev_time = der_tlv(0x18, b"20240601000000Z");
        let reason = der_tlv(0xA0, &der_tlv(0x0A, &[0x01])); // keyCompromise
        let mut ri = rev_time;
        ri.extend_from_slice(&reason);
        let revoked = der_tlv(0xA1, &ri);
        let der = build_ocsp_der(&revoked, true);
        let resp = parse_basic_response(&der).expect("revoked staple parses");
        match resp.cert_status {
            CertStatus::Revoked { reason, .. } => assert_eq!(reason, 1),
            other => panic!("expected Revoked, got {other:?}"),
        }
    }

    #[test]
    fn revoked_staple_evaluates_to_block() {
        let rev_time = der_tlv(0x18, b"20240601000000Z");
        let revoked = der_tlv(0xA1, &rev_time);
        let der = build_ocsp_der(&revoked, true);
        let resp = parse_basic_response(&der).unwrap();
        // 2024-06-15 in ms, inside the validity window → Block.
        let now_ms = 1_718_409_600_000u64;
        assert_eq!(evaluate(&resp, now_ms), OcspDecision::Block);
    }

    #[test]
    fn parse_rejects_nonsuccessful_status() {
        // responseStatus = 6 (unauthorized) → ResponseStatus error.
        let status = der_tlv(0x0A, &[0x06]);
        let der = der_tlv(0x30, &status);
        assert!(matches!(
            parse_basic_response(&der),
            Err(OcspParseError::ResponseStatus(6))
        ));
    }

    #[test]
    fn parse_rejects_non_basic_response_type() {
        // Swap the responseType OID for a non-basic one.
        let response_type = der_tlv(0x06, &[0x2B, 0x06, 0x01, 0x05, 0x05, 0x07, 0x30, 0x02]);
        let inner_basic = der_tlv(0x30, &[0x30, 0x00]);
        let response_octet = der_tlv(0x04, &inner_basic);
        let mut rb_inner = response_type;
        rb_inner.extend_from_slice(&response_octet);
        let response_bytes = der_tlv(0x30, &rb_inner);
        let rb_explicit = der_tlv(0xA0, &response_bytes);
        let status = der_tlv(0x0A, &[0x00]);
        let mut top = status;
        top.extend_from_slice(&rb_explicit);
        let der = der_tlv(0x30, &top);
        assert!(matches!(
            parse_basic_response(&der),
            Err(OcspParseError::NotBasicResponse)
        ));
    }

    #[test]
    fn generalized_time_parses_known_value() {
        // 2024-01-01T00:00:00Z = 1704067200 s.
        let ms = parse_generalized_time_ms(b"20240101000000Z").unwrap();
        assert_eq!(ms, 1_704_067_200_000);
    }

    #[test]
    fn generalized_time_rejects_non_utc() {
        assert!(parse_generalized_time_ms(b"20240101000000").is_none());
        assert!(parse_generalized_time_ms(b"short").is_none());
    }
}
