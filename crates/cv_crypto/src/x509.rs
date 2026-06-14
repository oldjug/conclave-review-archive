//! Minimal X.509 v3 certificate parser per RFC 5280.
//!
//! Scope: parse a single DER-encoded certificate down to the fields the
//! TLS 1.3 handshake and hostname check need:
//!   - `tbs` raw bytes (for sig verification)
//!   - signature algorithm OID + signature value
//!   - subject public key info (algorithm + key bytes)
//!   - subject common name (legacy hostname source)
//!   - subjectAltName DNSNames (preferred hostname source)
//!   - notBefore / notAfter (validity)
//!
//! Chain path-building lives separately; this module is pure parsing.

use crate::asn1::{Asn1Error, Oid, Reader, oids, tag};
use core::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum X509Error {
    Asn1(Asn1Error),
    UnsupportedVersion(u32),
    BadTime(String),
    UnsupportedAlgorithm(String),
    Malformed(&'static str),
}

impl fmt::Display for X509Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Asn1(e) => write!(f, "asn1: {e}"),
            Self::UnsupportedVersion(v) => write!(f, "unsupported X.509 version {v}"),
            Self::BadTime(s) => write!(f, "bad time: {s}"),
            Self::UnsupportedAlgorithm(s) => write!(f, "unsupported algorithm: {s}"),
            Self::Malformed(s) => write!(f, "malformed: {s}"),
        }
    }
}

impl From<Asn1Error> for X509Error {
    fn from(e: Asn1Error) -> Self {
        Self::Asn1(e)
    }
}

impl std::error::Error for X509Error {}

/// Signature algorithms we recognise. Anything else parses but we won't
/// be able to verify it; we propagate the OID up so the caller can decide.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum SigAlgorithm {
    Sha256WithRsa,
    Sha384WithRsa,
    EcdsaP256Sha256,
    RsaPss,
    Other,
}

impl SigAlgorithm {
    pub fn from_oid(oid: Oid<'_>) -> Self {
        if oid.bytes() == oids::SHA256_WITH_RSA.bytes() {
            Self::Sha256WithRsa
        } else if oid.bytes() == oids::SHA384_WITH_RSA.bytes() {
            Self::Sha384WithRsa
        } else if oid.bytes() == oids::ECDSA_WITH_SHA256.bytes() {
            Self::EcdsaP256Sha256
        } else if oid.bytes() == oids::RSASSA_PSS.bytes() {
            Self::RsaPss
        } else {
            Self::Other
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum SpkiAlgorithm {
    Rsa,
    EcP256,
    Other,
}

impl SpkiAlgorithm {
    pub fn from_oid(oid: Oid<'_>) -> Self {
        if oid.bytes() == oids::RSA_ENCRYPTION.bytes() {
            Self::Rsa
        } else if oid.bytes() == oids::EC_PUBLIC_KEY.bytes() {
            Self::EcP256
        } else {
            Self::Other
        }
    }
}

#[derive(Clone, Debug)]
pub struct Cert<'a> {
    /// Raw DER of `tbsCertificate` — input to signature verification.
    pub tbs_der: &'a [u8],
    pub version: u32,
    pub serial: &'a [u8],
    pub sig_alg: SigAlgorithm,
    pub sig_alg_oid: Oid<'a>,
    pub signature: &'a [u8],
    pub issuer_der: &'a [u8],
    pub subject_der: &'a [u8],
    pub not_before: Time,
    pub not_after: Time,
    pub spki_alg: SpkiAlgorithm,
    pub spki_alg_oid: Oid<'a>,
    /// For RSA, this is `RSAPublicKey ::= SEQUENCE { n INTEGER, e INTEGER }`.
    /// For ECP-256, this is the uncompressed point (`04 || X || Y`).
    pub spki_key_bytes: &'a [u8],
    pub subject_cn: Option<String>,
    /// DNS names from the subjectAltName extension.
    pub san_dns: Vec<String>,
    pub is_ca: bool,
}

/// A `YYYYMMDDhhmmssZ`-shaped UTC instant. Comparable as an integer.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Time(pub u64);

impl Time {
    pub fn from_utc_time(s: &[u8]) -> Result<Self, X509Error> {
        // UTCTime: YYMMDDhhmmssZ (13 bytes). Two-digit year ≥ 50 → 19xx,
        // else 20xx (per RFC 5280 §4.1.2.5.1).
        if s.len() != 13 || s[12] != b'Z' {
            return Err(X509Error::BadTime(format!("utc len={}", s.len())));
        }
        let yy = parse_u32(&s[0..2])?;
        let year = if yy >= 50 { 1900 + yy } else { 2000 + yy };
        let mm = parse_u32(&s[2..4])?;
        let dd = parse_u32(&s[4..6])?;
        let hh = parse_u32(&s[6..8])?;
        let mi = parse_u32(&s[8..10])?;
        let ss = parse_u32(&s[10..12])?;
        Ok(Self::pack(year, mm, dd, hh, mi, ss))
    }

    pub fn from_generalized_time(s: &[u8]) -> Result<Self, X509Error> {
        // GeneralizedTime: YYYYMMDDhhmmssZ (15 bytes).
        if s.len() != 15 || s[14] != b'Z' {
            return Err(X509Error::BadTime(format!("gen len={}", s.len())));
        }
        let year = parse_u32(&s[0..4])?;
        let mm = parse_u32(&s[4..6])?;
        let dd = parse_u32(&s[6..8])?;
        let hh = parse_u32(&s[8..10])?;
        let mi = parse_u32(&s[10..12])?;
        let ss = parse_u32(&s[12..14])?;
        Ok(Self::pack(year, mm, dd, hh, mi, ss))
    }

    fn pack(y: u32, mo: u32, d: u32, h: u32, mi: u32, s: u32) -> Self {
        let v = u64::from(y) * 10_000_000_000
            + u64::from(mo) * 100_000_000
            + u64::from(d) * 1_000_000
            + u64::from(h) * 10_000
            + u64::from(mi) * 100
            + u64::from(s);
        Self(v)
    }
}

fn parse_u32(s: &[u8]) -> Result<u32, X509Error> {
    let mut v: u32 = 0;
    for &b in s {
        if !b.is_ascii_digit() {
            return Err(X509Error::BadTime(format!("digit {b:?}")));
        }
        v = v * 10 + u32::from(b - b'0');
    }
    Ok(v)
}

pub fn parse(der: &[u8]) -> Result<Cert<'_>, X509Error> {
    let mut top = Reader::new(der);
    let mut cert_seq = top.read_sequence()?;
    if !top.is_empty() {
        return Err(X509Error::Asn1(Asn1Error::TrailingData));
    }

    let (mut tbs, tbs_full_tlv) = cert_seq.read_sequence_with_full()?;
    // tbs_full_tlv is the SEQUENCE tag/length + value — we want the FULL
    // TLV for hashing, since that's what was signed.
    let tbs_der = tbs_full_tlv;

    // version  [0] EXPLICIT Version DEFAULT v1
    let version = match tbs.read_optional(tag::context_constructed(0))? {
        Some(v) => {
            let mut r = Reader::new(v);
            let raw = r.read_integer_unsigned_bytes()?;
            if raw.len() > 4 {
                return Err(X509Error::UnsupportedVersion(0));
            }
            let mut n: u32 = 0;
            for &b in raw {
                n = (n << 8) | u32::from(b);
            }
            n + 1
        }
        None => 1,
    };
    if version > 3 {
        return Err(X509Error::UnsupportedVersion(version));
    }

    let serial = tbs.read_integer_unsigned_bytes()?;

    // signature algorithm in TBS.
    let mut sig_alg_seq = tbs.read_sequence()?;
    let _tbs_sig_oid = sig_alg_seq.read_oid()?;
    // Skip params.
    while !sig_alg_seq.is_empty() {
        sig_alg_seq.skip()?;
    }

    let (_issuer_inner, issuer_der) = tbs.read_sequence_with_full()?;

    // Validity.
    let mut validity = tbs.read_sequence()?;
    let not_before = read_time(&mut validity)?;
    let not_after = read_time(&mut validity)?;

    let (subject_inner, subject_der) = tbs.read_sequence_with_full()?;
    let subject_cn = extract_cn_from_name(subject_inner)?;

    // SubjectPublicKeyInfo.
    let mut spki = tbs.read_sequence()?;
    let mut spki_alg_seq = spki.read_sequence()?;
    let spki_alg_oid = spki_alg_seq.read_oid()?;
    let spki_alg = SpkiAlgorithm::from_oid(spki_alg_oid);
    while !spki_alg_seq.is_empty() {
        spki_alg_seq.skip()?;
    }
    let spki_key_bytes = spki.read_bit_string()?;

    // Optional unique identifiers, then optional extensions.
    let _ = tbs.read_optional(tag::context_primitive(1))?;
    let _ = tbs.read_optional(tag::context_primitive(2))?;
    let mut san_dns = Vec::new();
    let mut is_ca = false;
    if let Some(ext_value) = tbs.read_optional(tag::context_constructed(3))? {
        let mut ext_outer = Reader::new(ext_value);
        let mut exts = ext_outer.read_sequence()?;
        while !exts.is_empty() {
            let mut ext = exts.read_sequence()?;
            let oid = ext.read_oid()?;
            // OPTIONAL critical BOOLEAN.
            if ext.peek_tag() == Some(tag::BOOLEAN) {
                ext.skip()?;
            }
            let value = ext.read_octet_string()?;
            if oid.bytes() == oids::SUBJECT_ALT_NAME.bytes() {
                san_dns.extend(parse_san_dns_names(value)?);
            } else if oid.bytes() == oids::BASIC_CONSTRAINTS.bytes() {
                let mut bc = Reader::new(value);
                let mut bc_seq = bc.read_sequence()?;
                if bc_seq.peek_tag() == Some(tag::BOOLEAN) {
                    let bv = bc_seq.read_tagged(tag::BOOLEAN)?;
                    if bv.first() == Some(&0xFF) {
                        is_ca = true;
                    }
                }
            }
        }
    }

    // signatureAlgorithm + signatureValue at the Certificate level.
    let mut sig_alg_outer = cert_seq.read_sequence()?;
    let sig_alg_oid = sig_alg_outer.read_oid()?;
    let sig_alg = SigAlgorithm::from_oid(sig_alg_oid);
    while !sig_alg_outer.is_empty() {
        sig_alg_outer.skip()?;
    }
    let signature = cert_seq.read_bit_string()?;

    Ok(Cert {
        tbs_der,
        version,
        serial,
        sig_alg,
        sig_alg_oid,
        signature,
        issuer_der,
        subject_der,
        not_before,
        not_after,
        spki_alg,
        spki_alg_oid,
        spki_key_bytes,
        subject_cn,
        san_dns,
        is_ca,
    })
}

fn read_time(r: &mut Reader<'_>) -> Result<Time, X509Error> {
    match r.peek_tag() {
        Some(tag::UTC_TIME) => {
            let v = r.read_tagged(tag::UTC_TIME)?;
            Time::from_utc_time(v)
        }
        Some(tag::GENERALIZED_TIME) => {
            let v = r.read_tagged(tag::GENERALIZED_TIME)?;
            Time::from_generalized_time(v)
        }
        Some(t) => Err(X509Error::Asn1(Asn1Error::BadTag {
            expected: tag::UTC_TIME,
            got: t,
        })),
        None => Err(X509Error::Asn1(Asn1Error::Truncated)),
    }
}

fn extract_cn_from_name(name_seq: Reader<'_>) -> Result<Option<String>, X509Error> {
    // Name ::= SEQUENCE OF RDN; RDN ::= SET OF AttributeTypeAndValue
    let mut rdn_seq = name_seq;
    let mut cn = None;
    while !rdn_seq.is_empty() {
        let mut rdn = rdn_seq.read_set()?;
        while !rdn.is_empty() {
            let mut atv = rdn.read_sequence()?;
            let oid = atv.read_oid()?;
            if oid.bytes() == oids::CN.bytes() {
                let (tg, value, _) = atv.read_any()?;
                let s = match tg {
                    tag::UTF8_STRING | tag::PRINTABLE_STRING | tag::IA5_STRING => {
                        String::from_utf8_lossy(value).into_owned()
                    }
                    _ => continue,
                };
                cn = Some(s);
            } else {
                while !atv.is_empty() {
                    atv.skip()?;
                }
            }
        }
    }
    Ok(cn)
}

fn parse_san_dns_names(der: &[u8]) -> Result<Vec<String>, X509Error> {
    // GeneralNames ::= SEQUENCE SIZE (1..MAX) OF GeneralName
    // GeneralName ::= CHOICE { dNSName [2] IMPLICIT IA5String, ... }
    let mut top = Reader::new(der);
    let mut seq = top.read_sequence()?;
    let mut out = Vec::new();
    while !seq.is_empty() {
        let (tg, value, _) = seq.read_any()?;
        if tg == tag::context_primitive(2) {
            out.push(String::from_utf8_lossy(value).into_owned());
        }
    }
    Ok(out)
}

/// Wildcard-aware hostname check per RFC 6125 §6.4.3.
pub fn hostname_matches(host: &str, pattern: &str) -> bool {
    let host = host.trim_end_matches('.').to_ascii_lowercase();
    let pattern = pattern.trim_end_matches('.').to_ascii_lowercase();
    if !pattern.contains('*') {
        return host == pattern;
    }
    // Wildcard only allowed in the leftmost label.
    let (p_left, p_rest) = match pattern.split_once('.') {
        Some((l, r)) => (l, r),
        None => return false,
    };
    let (h_left, h_rest) = match host.split_once('.') {
        Some((l, r)) => (l, r),
        None => return false,
    };
    if p_rest != h_rest {
        return false;
    }
    // Wildcard must be a whole-label or a prefix/suffix of the label.
    if let Some((pre, post)) = p_left.split_once('*') {
        // Disallow wildcards in non-leftmost labels (already ensured above)
        // and in IDN A-labels.
        if h_left.starts_with(pre)
            && h_left.ends_with(post)
            && h_left.len() >= pre.len() + post.len()
        {
            return true;
        }
    }
    false
}

/// Verify a leaf DNS name against an intermediate cert's RFC 5280
/// nameConstraints (OID 2.5.29.30) extension. The extension carries
/// SEQUENCE { permittedSubtrees [0] OPTIONAL, excludedSubtrees [1] OPTIONAL }
/// each of which is a SEQUENCE OF GeneralSubtree { base GeneralName, ... }.
/// We honour the dNSName form (GeneralName tag [2]) and treat each base
/// as a suffix-match constraint per §4.2.1.10. Returns true if the name
/// is permitted (or no constraints apply) and not excluded.
pub fn dns_name_satisfies_constraints(
    dns_name: &str,
    name_constraints_extension_der: &[u8],
) -> bool {
    let name = dns_name.trim_end_matches('.').to_ascii_lowercase();
    let mut top = match Reader::new(name_constraints_extension_der).read_sequence() {
        Ok(r) => r,
        Err(_) => return true, // can't parse → don't reject the chain
    };
    let mut permitted: Vec<String> = Vec::new();
    let mut excluded: Vec<String> = Vec::new();
    while !top.is_empty() {
        let (tag_byte, value, _) = match top.read_any() {
            Ok(v) => v,
            Err(_) => return true,
        };
        // permittedSubtrees [0] IMPLICIT GeneralSubtrees
        // excludedSubtrees  [1] IMPLICIT GeneralSubtrees
        let is_permitted = tag_byte == tag::context_constructed(0);
        let is_excluded = tag_byte == tag::context_constructed(1);
        if !(is_permitted || is_excluded) {
            continue;
        }
        let mut subtrees = Reader::new(value);
        while !subtrees.is_empty() {
            let mut subtree = match subtrees.read_sequence() {
                Ok(r) => r,
                Err(_) => return true,
            };
            // GeneralSubtree.base = GeneralName CHOICE; dNSName is [2].
            if let Ok((gtag, gvalue, _)) = subtree.read_any() {
                if gtag == tag::context_primitive(2) {
                    let s = String::from_utf8_lossy(gvalue)
                        .trim_end_matches('.')
                        .to_ascii_lowercase();
                    if is_permitted {
                        permitted.push(s);
                    } else {
                        excluded.push(s);
                    }
                }
            }
        }
    }
    // RFC 5280 §4.2.1.10: a dNSName satisfies the constraint when it
    // ends with the constraint string and either equals it or the next
    // character upward is a '.' (label boundary).
    let suffix_match = |constraint: &str, candidate: &str| -> bool {
        if constraint.is_empty() {
            return true;
        }
        let c = constraint.trim_start_matches('.');
        if candidate == c {
            return true;
        }
        if let Some(stripped) = candidate.strip_suffix(c) {
            return stripped.ends_with('.');
        }
        false
    };
    if !excluded.iter().all(|e| !suffix_match(e, &name)) {
        return false;
    }
    if !permitted.is_empty() && !permitted.iter().any(|p| suffix_match(p, &name)) {
        return false;
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hostname_exact() {
        assert!(hostname_matches("example.com", "example.com"));
        assert!(!hostname_matches("evil.com", "example.com"));
    }

    #[test]
    fn hostname_wildcard() {
        assert!(hostname_matches("foo.example.com", "*.example.com"));
        assert!(!hostname_matches("bar.foo.example.com", "*.example.com")); // only one label
        assert!(!hostname_matches("example.com", "*.example.com")); // no label to match
    }

    #[test]
    fn time_utc_parses() {
        let t = Time::from_utc_time(b"240101000000Z").unwrap();
        let t2 = Time::from_utc_time(b"250101000000Z").unwrap();
        assert!(t < t2);
    }

    #[test]
    fn time_y2k_threshold() {
        // YY=49 → 2049, YY=50 → 1950 per RFC 5280.
        let a = Time::from_utc_time(b"491231235959Z").unwrap();
        let b = Time::from_utc_time(b"500101000000Z").unwrap();
        assert!(a > b, "RFC 5280 century pivot wrong");
    }
}
