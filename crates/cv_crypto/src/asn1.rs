//! Minimal ASN.1 DER reader, scoped to what X.509 needs.
//!
//! Read-only: enough to walk a `Certificate` tree, pull out fields, and
//! hand raw substrate (e.g. `tbsCertificate`) off to a signature verifier.
//! Encoding routines come if/when we need to write CSRs or OCSP requests.
//!
//! Reference: ITU-T X.690 §8.

use core::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Asn1Error {
    Truncated,
    BadTag { expected: u8, got: u8 },
    BadLength,
    TrailingData,
    BadOid,
    BadInteger,
    BadValue(String),
}

impl fmt::Display for Asn1Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Truncated => f.write_str("truncated"),
            Self::BadTag { expected, got } => {
                write!(f, "bad tag: expected 0x{expected:02x}, got 0x{got:02x}")
            }
            Self::BadLength => f.write_str("bad length"),
            Self::TrailingData => f.write_str("trailing data"),
            Self::BadOid => f.write_str("bad OID"),
            Self::BadInteger => f.write_str("bad INTEGER"),
            Self::BadValue(s) => write!(f, "bad value: {s}"),
        }
    }
}

impl std::error::Error for Asn1Error {}

/// Universal-class tags we care about.
pub mod tag {
    pub const BOOLEAN: u8 = 0x01;
    pub const INTEGER: u8 = 0x02;
    pub const BIT_STRING: u8 = 0x03;
    pub const OCTET_STRING: u8 = 0x04;
    pub const NULL: u8 = 0x05;
    pub const OID: u8 = 0x06;
    pub const UTF8_STRING: u8 = 0x0c;
    pub const SEQUENCE: u8 = 0x30;
    pub const SET: u8 = 0x31;
    pub const PRINTABLE_STRING: u8 = 0x13;
    pub const IA5_STRING: u8 = 0x16;
    pub const UTC_TIME: u8 = 0x17;
    pub const GENERALIZED_TIME: u8 = 0x18;

    /// Context-specific, constructed, tag number `n`.
    pub const fn context_constructed(n: u8) -> u8 {
        0xA0 | n
    }
    /// Context-specific, primitive, tag number `n`.
    pub const fn context_primitive(n: u8) -> u8 {
        0x80 | n
    }
}

#[derive(Copy, Clone)]
pub struct Reader<'a> {
    buf: &'a [u8],
}

impl<'a> fmt::Debug for Reader<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Reader({} bytes remaining)", self.buf.len())
    }
}

impl<'a> Reader<'a> {
    pub fn new(buf: &'a [u8]) -> Self {
        Self { buf }
    }

    pub fn is_empty(&self) -> bool {
        self.buf.is_empty()
    }

    pub fn remaining(&self) -> usize {
        self.buf.len()
    }

    pub fn peek_tag(&self) -> Option<u8> {
        self.buf.first().copied()
    }

    /// Read one TLV. Returns (tag, value-bytes, full-tlv-bytes).
    pub fn read_any(&mut self) -> Result<(u8, &'a [u8], &'a [u8]), Asn1Error> {
        if self.buf.is_empty() {
            return Err(Asn1Error::Truncated);
        }
        let tag = self.buf[0];
        let (len, header_len) = read_length(&self.buf[1..])?;
        let total = 1 + header_len + len;
        if self.buf.len() < total {
            return Err(Asn1Error::Truncated);
        }
        let value = &self.buf[1 + header_len..total];
        let full = &self.buf[..total];
        self.buf = &self.buf[total..];
        Ok((tag, value, full))
    }

    /// Read a TLV with a specific expected tag. Returns the value bytes.
    pub fn read_tagged(&mut self, expected: u8) -> Result<&'a [u8], Asn1Error> {
        let (tag, value, _) = self.read_any()?;
        if tag != expected {
            return Err(Asn1Error::BadTag { expected, got: tag });
        }
        Ok(value)
    }

    pub fn read_tagged_with_full(
        &mut self,
        expected: u8,
    ) -> Result<(&'a [u8], &'a [u8]), Asn1Error> {
        let (tag, value, full) = self.read_any()?;
        if tag != expected {
            return Err(Asn1Error::BadTag { expected, got: tag });
        }
        Ok((value, full))
    }

    pub fn read_optional(&mut self, expected: u8) -> Result<Option<&'a [u8]>, Asn1Error> {
        if self.peek_tag() == Some(expected) {
            self.read_tagged(expected).map(Some)
        } else {
            Ok(None)
        }
    }

    pub fn read_sequence(&mut self) -> Result<Reader<'a>, Asn1Error> {
        Ok(Reader::new(self.read_tagged(tag::SEQUENCE)?))
    }

    pub fn read_sequence_with_full(&mut self) -> Result<(Reader<'a>, &'a [u8]), Asn1Error> {
        let (value, full) = self.read_tagged_with_full(tag::SEQUENCE)?;
        Ok((Reader::new(value), full))
    }

    pub fn read_set(&mut self) -> Result<Reader<'a>, Asn1Error> {
        Ok(Reader::new(self.read_tagged(tag::SET)?))
    }

    pub fn read_oid(&mut self) -> Result<Oid<'a>, Asn1Error> {
        let v = self.read_tagged(tag::OID)?;
        Ok(Oid(v))
    }

    /// Read an unsigned INTEGER as raw bytes (BE), without the
    /// possible leading 0x00 byte that DER inserts to keep the value
    /// non-negative.
    pub fn read_integer_unsigned_bytes(&mut self) -> Result<&'a [u8], Asn1Error> {
        let v = self.read_tagged(tag::INTEGER)?;
        if v.is_empty() {
            return Err(Asn1Error::BadInteger);
        }
        // Strip a single leading zero used for sign disambiguation.
        let stripped = if v.len() > 1 && v[0] == 0 { &v[1..] } else { v };
        Ok(stripped)
    }

    pub fn read_bit_string(&mut self) -> Result<&'a [u8], Asn1Error> {
        let v = self.read_tagged(tag::BIT_STRING)?;
        if v.is_empty() {
            return Err(Asn1Error::BadValue("empty BIT STRING".into()));
        }
        let unused = v[0];
        if unused > 7 {
            return Err(Asn1Error::BadValue("BIT STRING unused-bits > 7".into()));
        }
        // We only return the byte payload; bit alignment is the caller's job.
        Ok(&v[1..])
    }

    pub fn read_octet_string(&mut self) -> Result<&'a [u8], Asn1Error> {
        self.read_tagged(tag::OCTET_STRING)
    }

    pub fn read_null(&mut self) -> Result<(), Asn1Error> {
        let v = self.read_tagged(tag::NULL)?;
        if !v.is_empty() {
            return Err(Asn1Error::BadValue("non-empty NULL".into()));
        }
        Ok(())
    }

    pub fn skip(&mut self) -> Result<(), Asn1Error> {
        self.read_any()?;
        Ok(())
    }
}

fn read_length(after_tag: &[u8]) -> Result<(usize, usize), Asn1Error> {
    if after_tag.is_empty() {
        return Err(Asn1Error::Truncated);
    }
    let first = after_tag[0];
    if first < 0x80 {
        Ok((first as usize, 1))
    } else {
        let n = (first & 0x7F) as usize;
        if n == 0 || n > 8 || after_tag.len() < 1 + n {
            return Err(Asn1Error::BadLength);
        }
        let mut len: usize = 0;
        for i in 0..n {
            len = (len << 8) | (after_tag[1 + i] as usize);
        }
        // DER prohibits using long-form when short would do.
        if len < 0x80 {
            return Err(Asn1Error::BadLength);
        }
        Ok((len, 1 + n))
    }
}

/// An OID stored as its raw DER value bytes (without tag/length).
#[derive(Copy, Clone, PartialEq, Eq, Hash)]
pub struct Oid<'a>(pub &'a [u8]);

impl<'a> Oid<'a> {
    pub fn bytes(self) -> &'a [u8] {
        self.0
    }

    /// Decode to a dotted-decimal string.
    pub fn to_string(self) -> String {
        let b = self.0;
        if b.is_empty() {
            return String::new();
        }
        let first = b[0] as u64;
        let mut s = format!("{}.{}", first / 40, first % 40);
        let mut i = 1;
        while i < b.len() {
            let mut v: u64 = 0;
            loop {
                if i >= b.len() {
                    return String::from("<malformed>");
                }
                let byte = b[i];
                v = (v << 7) | u64::from(byte & 0x7F);
                i += 1;
                if byte & 0x80 == 0 {
                    break;
                }
            }
            s.push('.');
            s.push_str(&v.to_string());
        }
        s
    }
}

impl<'a> fmt::Debug for Oid<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "OID({})", self.to_string())
    }
}

/// Common OIDs.
pub mod oids {
    use super::Oid;

    pub const RSA_ENCRYPTION: Oid<'static> = Oid(&[
        0x2a, 0x86, 0x48, 0x86, 0xf7, 0x0d, 0x01, 0x01, 0x01, // 1.2.840.113549.1.1.1
    ]);
    pub const SHA256_WITH_RSA: Oid<'static> = Oid(&[
        0x2a, 0x86, 0x48, 0x86, 0xf7, 0x0d, 0x01, 0x01, 0x0b, // 1.2.840.113549.1.1.11
    ]);
    pub const SHA384_WITH_RSA: Oid<'static> = Oid(&[
        0x2a, 0x86, 0x48, 0x86, 0xf7, 0x0d, 0x01, 0x01, 0x0c, // 1.2.840.113549.1.1.12
    ]);
    pub const RSASSA_PSS: Oid<'static> = Oid(&[
        0x2a, 0x86, 0x48, 0x86, 0xf7, 0x0d, 0x01, 0x01, 0x0a, // 1.2.840.113549.1.1.10
    ]);
    pub const ECDSA_WITH_SHA256: Oid<'static> = Oid(&[
        0x2a, 0x86, 0x48, 0xce, 0x3d, 0x04, 0x03, 0x02, // 1.2.840.10045.4.3.2
    ]);
    pub const EC_PUBLIC_KEY: Oid<'static> = Oid(&[
        0x2a, 0x86, 0x48, 0xce, 0x3d, 0x02, 0x01, // 1.2.840.10045.2.1
    ]);
    pub const P256_CURVE: Oid<'static> = Oid(&[
        0x2a, 0x86, 0x48, 0xce, 0x3d, 0x03, 0x01, 0x07, // 1.2.840.10045.3.1.7
    ]);
    pub const CN: Oid<'static> = Oid(&[0x55, 0x04, 0x03]); // 2.5.4.3
    pub const SUBJECT_ALT_NAME: Oid<'static> = Oid(&[0x55, 0x1d, 0x11]); // 2.5.29.17
    pub const BASIC_CONSTRAINTS: Oid<'static> = Oid(&[0x55, 0x1d, 0x13]); // 2.5.29.19
    pub const KEY_USAGE: Oid<'static> = Oid(&[0x55, 0x1d, 0x0f]); // 2.5.29.15
    pub const EXT_KEY_USAGE: Oid<'static> = Oid(&[0x55, 0x1d, 0x25]); // 2.5.29.37
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_short_sequence() {
        // SEQUENCE { INTEGER 0x42, NULL }
        let der = [0x30, 0x05, 0x02, 0x01, 0x42, 0x05, 0x00];
        let mut r = Reader::new(&der);
        let mut inner = r.read_sequence().unwrap();
        let i = inner.read_tagged(tag::INTEGER).unwrap();
        assert_eq!(i, &[0x42]);
        inner.read_null().unwrap();
        assert!(inner.is_empty());
        assert!(r.is_empty());
    }

    #[test]
    fn parse_long_length() {
        // SEQUENCE { OCTET STRING (128 bytes of 0xAA) }
        //   - inner TLV: 0x04 0x81 0x80 + 128 bytes = 131 bytes
        //   - outer SEQUENCE length = 0x83 = 131
        let mut der = vec![0x30, 0x81, 0x83, 0x04, 0x81, 0x80];
        der.extend(std::iter::repeat_n(0xAA, 128));
        let mut r = Reader::new(&der);
        let mut inner = r.read_sequence().unwrap();
        let v = inner.read_octet_string().unwrap();
        assert_eq!(v.len(), 128);
        assert!(v.iter().all(|&b| b == 0xAA));
    }

    #[test]
    fn rejects_truncated() {
        let der = [0x30, 0x05, 0x02, 0x01];
        let mut r = Reader::new(&der);
        assert!(matches!(r.read_sequence(), Err(Asn1Error::Truncated)));
    }

    #[test]
    fn rejects_non_minimal_long_form() {
        // Long-form length encoding a value < 0x80 is illegal in DER.
        let der = [0x04, 0x81, 0x05, 1, 2, 3, 4, 5];
        let mut r = Reader::new(&der);
        assert!(matches!(r.read_octet_string(), Err(Asn1Error::BadLength)));
    }

    #[test]
    fn oid_decodes() {
        // 1.2.840.113549.1.1.11 — sha256WithRSAEncryption.
        assert_eq!(
            super::oids::SHA256_WITH_RSA.to_string(),
            "1.2.840.113549.1.1.11"
        );
        assert_eq!(super::oids::CN.to_string(), "2.5.4.3");
    }

    #[test]
    fn integer_strips_leading_zero() {
        // INTEGER 0x00 0x80 → 0x80 (would otherwise look negative).
        let der = [0x02, 0x02, 0x00, 0x80];
        let mut r = Reader::new(&der);
        let v = r.read_integer_unsigned_bytes().unwrap();
        assert_eq!(v, &[0x80]);
    }
}
