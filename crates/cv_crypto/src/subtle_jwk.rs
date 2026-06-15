//! SubtleCrypto algorithm coverage — JWK / SPKI / PKCS#8 import/export
//! plumbing for the WebCrypto surface.
//!
//! The pieces here back the JS-facing SubtleCrypto bindings: the
//! parsers know how to walk JWK JSON, SubjectPublicKeyInfo (SPKI),
//! and PKCS#8 PrivateKeyInfo into the raw key material the actual
//! primitive routines (RSA, AES, ECDH, X25519, Ed25519, P-256)
//! consume. importKey/exportKey routes through here; encrypt/decrypt
//! routes through the existing primitives.

use crate::asn1::Reader;

/// Result of a parsed JSON Web Key body. Keep both the kty and the
/// raw component bytes so the caller can hand them straight to the
/// underlying primitive (RSA modulus, EC X/Y, AES raw, OKP X25519).
#[derive(Debug, Clone)]
pub struct JwkComponents {
    pub kty: String,
    pub crv: Option<String>,
    pub n: Option<Vec<u8>>,
    pub e: Option<Vec<u8>>,
    pub d: Option<Vec<u8>>,
    pub x: Option<Vec<u8>>,
    pub y: Option<Vec<u8>>,
    pub k: Option<Vec<u8>>,
}

/// Walk a JSON Web Key object surfacing the canonical fields. The
/// parser is liberal: it accepts un-escaped strings, ignores fields
/// it doesn't know, and base64url-decodes the standard binary fields.
pub fn parse_jwk(json: &str) -> Option<JwkComponents> {
    let mut out = JwkComponents {
        kty: String::new(),
        crv: None,
        n: None,
        e: None,
        d: None,
        x: None,
        y: None,
        k: None,
    };
    let bytes = json.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'"' {
            let mut j = i + 1;
            while j < bytes.len() && bytes[j] != b'"' {
                j += 1;
            }
            let key = std::str::from_utf8(&bytes[i + 1..j]).ok()?;
            i = j + 1;
            while i < bytes.len() && bytes[i] != b':' {
                i += 1;
            }
            i += 1;
            while i < bytes.len() && bytes[i].is_ascii_whitespace() {
                i += 1;
            }
            if i >= bytes.len() {
                break;
            }
            if bytes[i] != b'"' {
                i += 1;
                continue;
            }
            let v_start = i + 1;
            let mut k = v_start;
            while k < bytes.len() && bytes[k] != b'"' {
                k += 1;
            }
            let val = std::str::from_utf8(&bytes[v_start..k]).ok()?;
            i = k + 1;
            match key {
                "kty" => out.kty = val.to_string(),
                "crv" => out.crv = Some(val.to_string()),
                "n" => out.n = b64url_decode(val),
                "e" => out.e = b64url_decode(val),
                "d" => out.d = b64url_decode(val),
                "x" => out.x = b64url_decode(val),
                "y" => out.y = b64url_decode(val),
                "k" => out.k = b64url_decode(val),
                _ => {}
            }
            continue;
        }
        i += 1;
    }
    if out.kty.is_empty() { None } else { Some(out) }
}

/// Serialize key components into a JWK JSON string. Caller chooses
/// which fields to populate based on the algorithm.
pub fn serialize_jwk(c: &JwkComponents) -> String {
    let mut s = String::from("{");
    let mut first = true;
    fn push_str(s: &mut String, first: &mut bool, k: &str, v: &str) {
        if !*first {
            s.push(',');
        }
        *first = false;
        s.push_str(&format!("\"{k}\":\"{v}\""));
    }
    push_str(&mut s, &mut first, "kty", &c.kty);
    if let Some(crv) = &c.crv {
        push_str(&mut s, &mut first, "crv", crv);
    }
    for (k, v) in [
        ("n", &c.n),
        ("e", &c.e),
        ("d", &c.d),
        ("x", &c.x),
        ("y", &c.y),
        ("k", &c.k),
    ] {
        if let Some(bytes) = v {
            push_str(&mut s, &mut first, k, &b64url_encode(bytes));
        }
    }
    s.push('}');
    s
}

fn b64url_decode(s: &str) -> Option<Vec<u8>> {
    let mut buf = String::with_capacity(s.len() + 4);
    for c in s.chars() {
        match c {
            '-' => buf.push('+'),
            '_' => buf.push('/'),
            c => buf.push(c),
        }
    }
    while buf.len() % 4 != 0 {
        buf.push('=');
    }
    b64_decode_std(&buf)
}

fn b64url_encode(bytes: &[u8]) -> String {
    let mut s = b64_encode_std(bytes);
    s = s.replace('+', "-").replace('/', "_");
    s.trim_end_matches('=').to_string()
}

fn b64_decode_std(s: &str) -> Option<Vec<u8>> {
    const ALPHA: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut lookup = [255u8; 256];
    for (i, &c) in ALPHA.iter().enumerate() {
        lookup[c as usize] = i as u8;
    }
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len() / 4 * 3);
    let mut buf = 0u32;
    let mut bits = 0u32;
    for &b in bytes {
        if b == b'=' {
            break;
        }
        let v = lookup[b as usize];
        if v == 255 {
            return None;
        }
        buf = (buf << 6) | v as u32;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((buf >> bits) as u8);
            buf &= (1 << bits) - 1;
        }
    }
    Some(out)
}

fn b64_encode_std(bytes: &[u8]) -> String {
    const ALPHA: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity((bytes.len() + 2) / 3 * 4);
    let mut i = 0;
    while i + 3 <= bytes.len() {
        let n = ((bytes[i] as u32) << 16) | ((bytes[i + 1] as u32) << 8) | bytes[i + 2] as u32;
        out.push(ALPHA[((n >> 18) & 0x3F) as usize] as char);
        out.push(ALPHA[((n >> 12) & 0x3F) as usize] as char);
        out.push(ALPHA[((n >> 6) & 0x3F) as usize] as char);
        out.push(ALPHA[(n & 0x3F) as usize] as char);
        i += 3;
    }
    let rem = bytes.len() - i;
    if rem == 1 {
        let n = (bytes[i] as u32) << 16;
        out.push(ALPHA[((n >> 18) & 0x3F) as usize] as char);
        out.push(ALPHA[((n >> 12) & 0x3F) as usize] as char);
        out.push_str("==");
    } else if rem == 2 {
        let n = ((bytes[i] as u32) << 16) | ((bytes[i + 1] as u32) << 8);
        out.push(ALPHA[((n >> 18) & 0x3F) as usize] as char);
        out.push(ALPHA[((n >> 12) & 0x3F) as usize] as char);
        out.push(ALPHA[((n >> 6) & 0x3F) as usize] as char);
        out.push('=');
    }
    out
}

/// Parse a PKCS#8 PrivateKeyInfo blob. Returns the (algorithm OID,
/// raw private-key OCTET STRING contents). RSA private keys, EC
/// private keys (P-256), Ed25519 / X25519 all use PKCS#8 wrappers.
pub fn parse_pkcs8(buf: &[u8]) -> Option<(Vec<u8>, Vec<u8>)> {
    let mut top = Reader::new(buf).read_sequence().ok()?;
    let _version = top.read_integer_unsigned_bytes().ok()?;
    let mut alg_seq = top.read_sequence().ok()?;
    let oid_tlv = alg_seq.read_any().ok()?;
    let oid = oid_tlv.1.to_vec();
    let (_, key_octet, _) = top.read_any().ok()?;
    Some((oid, key_octet.to_vec()))
}

// ---------------------------------------------------------------------------
// Minimal DER writer + key-format encoders for the WebCrypto SPKI/PKCS#8
// import/export paths. We only emit the structures Chrome's WebCrypto produces
// for the algorithms we support (RSA, P-256/384 EC). All lengths use the DER
// definite short/long form.
// ---------------------------------------------------------------------------

/// DER-encode a definite length.
fn der_len(len: usize) -> Vec<u8> {
    if len < 0x80 {
        vec![len as u8]
    } else {
        let mut bytes = Vec::new();
        let mut l = len;
        while l > 0 {
            bytes.insert(0, (l & 0xFF) as u8);
            l >>= 8;
        }
        let mut out = vec![0x80 | bytes.len() as u8];
        out.extend_from_slice(&bytes);
        out
    }
}

/// DER TLV: `tag || len || body`.
fn der_tlv(tag: u8, body: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(1 + 4 + body.len());
    out.push(tag);
    out.extend_from_slice(&der_len(body.len()));
    out.extend_from_slice(body);
    out
}

/// DER INTEGER from an unsigned big-endian magnitude (adds a leading 0x00 when
/// the high bit is set so the value stays positive).
fn der_uint(mag: &[u8]) -> Vec<u8> {
    let mut m = mag;
    while m.len() > 1 && m[0] == 0 {
        m = &m[1..];
    }
    let mut body = Vec::with_capacity(m.len() + 1);
    if m.is_empty() {
        body.push(0);
    } else {
        if m[0] & 0x80 != 0 {
            body.push(0);
        }
        body.extend_from_slice(m);
    }
    der_tlv(0x02, &body)
}

// OIDs (DER content bytes, i.e. excluding the 0x06 tag and length).
const OID_RSA_ENCRYPTION: &[u8] = &[0x2a, 0x86, 0x48, 0x86, 0xf7, 0x0d, 0x01, 0x01, 0x01];
const OID_EC_PUBLIC_KEY: &[u8] = &[0x2a, 0x86, 0x48, 0xce, 0x3d, 0x02, 0x01];
const OID_P256: &[u8] = &[0x2a, 0x86, 0x48, 0xce, 0x3d, 0x03, 0x01, 0x07];
const OID_P384: &[u8] = &[0x2b, 0x81, 0x04, 0x00, 0x22];
const OID_P521: &[u8] = &[0x2b, 0x81, 0x04, 0x00, 0x23];

fn der_oid(content: &[u8]) -> Vec<u8> {
    der_tlv(0x06, content)
}

/// Map a JWK/WebCrypto curve name to its named-curve OID content bytes.
pub fn curve_oid(crv: &str) -> Option<&'static [u8]> {
    match crv {
        "P-256" => Some(OID_P256),
        "P-384" => Some(OID_P384),
        "P-521" => Some(OID_P521),
        _ => None,
    }
}

/// Reverse of `curve_oid`: identify the curve from its OID content bytes.
pub fn oid_to_curve(oid: &[u8]) -> Option<&'static str> {
    if oid == OID_P256 {
        Some("P-256")
    } else if oid == OID_P384 {
        Some("P-384")
    } else if oid == OID_P521 {
        Some("P-521")
    } else {
        None
    }
}

/// Build an RSA SubjectPublicKeyInfo (SPKI) from modulus `n` and exponent `e`
/// (big-endian magnitudes). Layout per RFC 5280 / RFC 8017 Appendix A.1.1.
pub fn rsa_spki(n: &[u8], e: &[u8]) -> Vec<u8> {
    let rsa_pubkey = {
        let mut seq = der_uint(n);
        seq.extend_from_slice(&der_uint(e));
        der_tlv(0x30, &seq)
    };
    // BIT STRING wraps the RSAPublicKey DER with a leading 0x00 "unused bits".
    let mut bit_body = vec![0x00];
    bit_body.extend_from_slice(&rsa_pubkey);
    let bit_string = der_tlv(0x03, &bit_body);
    // AlgorithmIdentifier { rsaEncryption, NULL }
    let mut alg = der_oid(OID_RSA_ENCRYPTION);
    alg.extend_from_slice(&der_tlv(0x05, &[])); // NULL
    let alg_id = der_tlv(0x30, &alg);
    let mut spki = alg_id;
    spki.extend_from_slice(&bit_string);
    der_tlv(0x30, &spki)
}

/// Build an EC SubjectPublicKeyInfo from the uncompressed point (`0x04||X||Y`)
/// and the curve OID. Layout per RFC 5480 §2.
pub fn ec_spki(point_uncompressed: &[u8], curve_oid_content: &[u8]) -> Vec<u8> {
    // AlgorithmIdentifier { id-ecPublicKey, namedCurve }
    let mut alg = der_oid(OID_EC_PUBLIC_KEY);
    alg.extend_from_slice(&der_oid(curve_oid_content));
    let alg_id = der_tlv(0x30, &alg);
    // subjectPublicKey BIT STRING = 0x00 || point
    let mut bit_body = vec![0x00];
    bit_body.extend_from_slice(point_uncompressed);
    let bit_string = der_tlv(0x03, &bit_body);
    let mut spki = alg_id;
    spki.extend_from_slice(&bit_string);
    der_tlv(0x30, &spki)
}

/// Parse an EC SPKI, returning `(curve_name, uncompressed_point)`.
pub fn parse_ec_spki(buf: &[u8]) -> Option<(&'static str, Vec<u8>)> {
    let mut top = Reader::new(buf).read_sequence().ok()?;
    let mut alg = top.read_sequence().ok()?;
    let oid1 = alg.read_oid().ok()?;
    if oid1.bytes() != OID_EC_PUBLIC_KEY {
        return None;
    }
    let curve = alg.read_oid().ok()?;
    let crv = oid_to_curve(curve.bytes())?;
    let bits = top.read_bit_string().ok()?;
    Some((crv, bits.to_vec()))
}

/// Parse an RSA SPKI, returning `(n, e)` big-endian magnitudes.
pub fn parse_rsa_spki(buf: &[u8]) -> Option<(Vec<u8>, Vec<u8>)> {
    let mut top = Reader::new(buf).read_sequence().ok()?;
    let mut alg = top.read_sequence().ok()?;
    let oid = alg.read_oid().ok()?;
    if oid.bytes() != OID_RSA_ENCRYPTION {
        return None;
    }
    let bits = top.read_bit_string().ok()?;
    let mut inner = Reader::new(bits).read_sequence().ok()?;
    let n = inner.read_integer_unsigned_bytes().ok()?.to_vec();
    let e = inner.read_integer_unsigned_bytes().ok()?.to_vec();
    Some((n, e))
}

/// Build an EC PKCS#8 PrivateKeyInfo wrapping a SEC1 ECPrivateKey for the named
/// curve. `d` is the raw private scalar, `point` the uncompressed public point.
/// Layout per RFC 5208 + RFC 5915.
pub fn ec_pkcs8(d: &[u8], point_uncompressed: &[u8], curve_oid_content: &[u8]) -> Vec<u8> {
    // SEC1 ECPrivateKey ::= SEQUENCE { version 1, privateKey OCTET STRING,
    //   [0] parameters (namedCurve OID), [1] publicKey BIT STRING }
    let mut sec1 = der_uint(&[1]); // version 1
    sec1.extend_from_slice(&der_tlv(0x04, d)); // privateKey OCTET STRING
    // [0] EXPLICIT namedCurve
    let params = der_oid(curve_oid_content);
    sec1.extend_from_slice(&der_tlv(0xA0, &params));
    // [1] EXPLICIT publicKey BIT STRING
    let mut bit_body = vec![0x00];
    bit_body.extend_from_slice(point_uncompressed);
    let pubkey_bit = der_tlv(0x03, &bit_body);
    sec1.extend_from_slice(&der_tlv(0xA1, &pubkey_bit));
    let ec_private_key = der_tlv(0x30, &sec1);

    // PKCS#8 PrivateKeyInfo
    let mut alg = der_oid(OID_EC_PUBLIC_KEY);
    alg.extend_from_slice(&der_oid(curve_oid_content));
    let alg_id = der_tlv(0x30, &alg);
    let mut pki = der_uint(&[0]); // version 0
    pki.extend_from_slice(&alg_id);
    pki.extend_from_slice(&der_tlv(0x04, &ec_private_key)); // privateKey OCTET STRING
    der_tlv(0x30, &pki)
}

/// Parse an EC PKCS#8, returning `(curve_name, raw_d_scalar)`.
pub fn parse_ec_pkcs8(buf: &[u8]) -> Option<(&'static str, Vec<u8>)> {
    let (oid, key_octet) = parse_pkcs8(buf)?;
    if oid != OID_EC_PUBLIC_KEY {
        return None;
    }
    // key_octet is the SEC1 ECPrivateKey SEQUENCE.
    let mut sec1 = Reader::new(&key_octet).read_sequence().ok()?;
    let _version = sec1.read_integer_unsigned_bytes().ok()?;
    let d = sec1.read_octet_string().ok()?.to_vec();
    // Curve may be carried in the PKCS#8 AlgorithmIdentifier, but the SEC1
    // body's [0] params is authoritative when present. Re-parse for the OID.
    let mut top = Reader::new(buf).read_sequence().ok()?;
    let _ver = top.read_integer_unsigned_bytes().ok()?;
    let mut alg = top.read_sequence().ok()?;
    let _ecpub = alg.read_oid().ok()?;
    let curve = alg.read_oid().ok()?;
    let crv = oid_to_curve(curve.bytes())?;
    Some((crv, d))
}

/// Build an RSA PKCS#8 PrivateKeyInfo from CRT components. All inputs are
/// big-endian magnitudes. Layout per RFC 5208 + RFC 8017 Appendix A.1.2
/// (RSAPrivateKey, version 0 / two-prime).
#[allow(clippy::too_many_arguments)]
pub fn rsa_pkcs8(
    n: &[u8],
    e: &[u8],
    d: &[u8],
    p: &[u8],
    q: &[u8],
    dp: &[u8],
    dq: &[u8],
    qinv: &[u8],
) -> Vec<u8> {
    let mut rpk = der_uint(&[0]); // version 0 (two-prime)
    for comp in [n, e, d, p, q, dp, dq, qinv] {
        rpk.extend_from_slice(&der_uint(comp));
    }
    let rsa_private_key = der_tlv(0x30, &rpk);

    let mut alg = der_oid(OID_RSA_ENCRYPTION);
    alg.extend_from_slice(&der_tlv(0x05, &[])); // NULL
    let alg_id = der_tlv(0x30, &alg);

    let mut pki = der_uint(&[0]); // version 0
    pki.extend_from_slice(&alg_id);
    pki.extend_from_slice(&der_tlv(0x04, &rsa_private_key));
    der_tlv(0x30, &pki)
}

/// Parse an RSA PKCS#8, returning the RSAPrivateKey components
/// `(n, e, d, p, q, dp, dq, qinv)` as big-endian magnitudes.
#[allow(clippy::type_complexity)]
pub fn parse_rsa_pkcs8(
    buf: &[u8],
) -> Option<(
    Vec<u8>,
    Vec<u8>,
    Vec<u8>,
    Vec<u8>,
    Vec<u8>,
    Vec<u8>,
    Vec<u8>,
    Vec<u8>,
)> {
    let (oid, key_octet) = parse_pkcs8(buf)?;
    if oid != OID_RSA_ENCRYPTION {
        return None;
    }
    let mut rpk = Reader::new(&key_octet).read_sequence().ok()?;
    let _ver = rpk.read_integer_unsigned_bytes().ok()?;
    let n = rpk.read_integer_unsigned_bytes().ok()?.to_vec();
    let e = rpk.read_integer_unsigned_bytes().ok()?.to_vec();
    let d = rpk.read_integer_unsigned_bytes().ok()?.to_vec();
    let p = rpk.read_integer_unsigned_bytes().ok()?.to_vec();
    let q = rpk.read_integer_unsigned_bytes().ok()?.to_vec();
    let dp = rpk.read_integer_unsigned_bytes().ok()?.to_vec();
    let dq = rpk.read_integer_unsigned_bytes().ok()?.to_vec();
    let qinv = rpk.read_integer_unsigned_bytes().ok()?.to_vec();
    Some((n, e, d, p, q, dp, dq, qinv))
}

/// base64url-encode (no padding) — exposed so the JS bindings can build JWK
/// strings without re-implementing it.
pub fn b64url(bytes: &[u8]) -> String {
    b64url_encode(bytes)
}

/// base64url-decode — exposed for the JS bindings.
pub fn b64url_dec(s: &str) -> Option<Vec<u8>> {
    b64url_decode(s)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn jwk_roundtrip_rsa_pubkey() {
        let j = JwkComponents {
            kty: "RSA".into(),
            crv: None,
            n: Some(vec![0xAA, 0xBB]),
            e: Some(vec![1, 0, 1]),
            d: None,
            x: None,
            y: None,
            k: None,
        };
        let s = serialize_jwk(&j);
        let parsed = parse_jwk(&s).unwrap();
        assert_eq!(parsed.kty, "RSA");
        assert_eq!(parsed.n.as_deref(), Some(&[0xAA, 0xBB][..]));
    }

    #[test]
    fn jwk_oct_with_k_value() {
        let s = "{\"kty\":\"oct\",\"k\":\"AQID\"}";
        let p = parse_jwk(s).unwrap();
        assert_eq!(p.kty, "oct");
        assert_eq!(p.k.as_deref(), Some(&[1u8, 2, 3][..]));
    }

    #[test]
    fn rsa_spki_roundtrip() {
        let n = vec![0xC0, 0xFF, 0xEE, 0x12, 0x34, 0x56, 0x78, 0x9A];
        let e = vec![0x01, 0x00, 0x01];
        let spki = rsa_spki(&n, &e);
        let (n2, e2) = parse_rsa_spki(&spki).unwrap();
        assert_eq!(n2, n);
        assert_eq!(e2, e);
    }

    #[test]
    fn ec_spki_roundtrip() {
        let mut point = vec![0x04u8];
        point.extend_from_slice(&[0x11; 64]); // X||Y placeholder
        let spki = ec_spki(&point, curve_oid("P-256").unwrap());
        let (crv, p2) = parse_ec_spki(&spki).unwrap();
        assert_eq!(crv, "P-256");
        assert_eq!(p2, point);
    }

    #[test]
    fn ec_pkcs8_roundtrip() {
        let d = vec![0x7Au8; 32];
        let mut point = vec![0x04u8];
        point.extend_from_slice(&[0x22; 64]);
        let pk8 = ec_pkcs8(&d, &point, curve_oid("P-256").unwrap());
        let (crv, d2) = parse_ec_pkcs8(&pk8).unwrap();
        assert_eq!(crv, "P-256");
        assert_eq!(d2, d);
    }

    #[test]
    fn rsa_pkcs8_roundtrip() {
        let comps: Vec<Vec<u8>> = (1..=8u8).map(|i| vec![i, i + 1, i + 2]).collect();
        let pk8 = rsa_pkcs8(
            &comps[0], &comps[1], &comps[2], &comps[3], &comps[4], &comps[5], &comps[6], &comps[7],
        );
        let (n, e, d, p, q, dp, dq, qinv) = parse_rsa_pkcs8(&pk8).unwrap();
        assert_eq!(n, comps[0]);
        assert_eq!(e, comps[1]);
        assert_eq!(d, comps[2]);
        assert_eq!(qinv, comps[7]);
        let _ = (p, q, dp, dq);
    }
}
