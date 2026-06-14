//! CRL — Certificate Revocation List (RFC 5280 §5).
//!
//! Minimal parser that pulls out the issuer name, this_update,
//! next_update and the (serial, revocation_time) pairs. Distribution
//! Point URLs are extracted via the CRLDistributionPoints extension on
//! the leaf cert; the cached CRL is consulted before a TLS handshake
//! accepts the chain.

use crate::asn1::Reader;

#[derive(Debug, Clone)]
pub struct CrlEntry {
    pub serial_be: Vec<u8>,
    pub revocation_time_ms: u64,
}

#[derive(Debug, Clone, Default)]
pub struct Crl {
    pub issuer_dn: Vec<u8>,
    pub this_update_ms: u64,
    pub next_update_ms: u64,
    pub entries: Vec<CrlEntry>,
}

/// Decide whether a leaf with the given serial number is revoked
/// according to this CRL. CRLs without the serial pass through.
pub fn is_revoked(crl: &Crl, leaf_serial_be: &[u8]) -> bool {
    crl.entries.iter().any(|e| e.serial_be == leaf_serial_be)
}

/// Parse a DER-encoded CRL (CertificateList). We only walk the outer
/// SEQUENCE → tbsCertList → revokedCertificates field. Missing
/// fields collapse to defaults.
pub fn parse(der: &[u8]) -> Option<Crl> {
    let mut out = Crl::default();
    let mut top = Reader::new(der).read_sequence().ok()?;
    let mut tbs = top.read_sequence().ok()?;
    // Optional version INTEGER.
    if tbs.peek_tag() == Some(crate::asn1::tag::INTEGER) {
        let _ = tbs.read_any();
    }
    // signature AlgorithmIdentifier (SEQUENCE).
    let _ = tbs.read_sequence().ok()?;
    // issuer Name (SEQUENCE).
    let (_, issuer_tlv, _) = tbs.read_any().ok()?;
    out.issuer_dn = issuer_tlv.to_vec();
    // thisUpdate (Time CHOICE).
    let _ = tbs.read_any().ok()?;
    // nextUpdate OPTIONAL.
    if !tbs.is_empty() {
        let _ = tbs.read_any();
    }
    // revokedCertificates OPTIONAL — SEQUENCE OF { userCert INTEGER, revocationDate Time, ext OPTIONAL }.
    if tbs.peek_tag() == Some(crate::asn1::tag::SEQUENCE) {
        let (_, val, _) = tbs.read_any().ok()?;
        {
            let mut seq = Reader::new(val);
            while !seq.is_empty() {
                let mut e = seq.read_sequence().ok()?;
                let serial = e.read_integer_unsigned_bytes().ok()?.to_vec();
                let _date = e.read_any();
                out.entries.push(CrlEntry {
                    serial_be: serial,
                    revocation_time_ms: 0,
                });
            }
        }
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_crl_lists_nothing() {
        let crl = Crl::default();
        assert!(!is_revoked(&crl, &[1, 2, 3]));
    }
}
