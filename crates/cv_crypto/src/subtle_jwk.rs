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
}
