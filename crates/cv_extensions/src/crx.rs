//! CRX3 (`.crx`) loader.
//!
//! CRX3 format = `Cr24` magic + version + header_len + header_pb +
//! ZIP payload. We don't parse the protobuf header (publisher key
//! verification lands later); we just locate the ZIP payload so the
//! manifest.json is reachable.

pub const CRX3_MAGIC: &[u8; 4] = b"Cr24";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Crx3 {
    pub version: u32,
    pub header_bytes: Vec<u8>,
    pub zip_offset: usize,
    pub zip_size: usize,
}

pub fn parse_crx3(buf: &[u8]) -> Option<Crx3> {
    if buf.len() < 12 || &buf[0..4] != CRX3_MAGIC {
        return None;
    }
    let version = u32::from_le_bytes(buf[4..8].try_into().unwrap());
    if version != 3 {
        return None;
    }
    let header_len = u32::from_le_bytes(buf[8..12].try_into().unwrap()) as usize;
    let zip_offset = 12 + header_len;
    if zip_offset > buf.len() {
        return None;
    }
    Some(Crx3 {
        version,
        header_bytes: buf[12..zip_offset].to_vec(),
        zip_offset,
        zip_size: buf.len() - zip_offset,
    })
}

/// CRX3 protobuf header — RSA SHA-256 PKCS#1 v1.5 signed
/// `sha256_with_rsa` proofs over `signed_header_data || zip_payload`.
/// We extract the public-key bytes and signature bytes per proof.
#[derive(Debug, Clone)]
pub struct CrxProof {
    pub public_key: Vec<u8>,
    pub signature: Vec<u8>,
}

/// Parse the CRX3 protobuf header. Protobuf wire format: each field
/// is a varint tag-and-wire-type pair followed by the payload.
/// `sha256_with_rsa` proofs are tag #2 length-delimited messages
/// inside the top-level `CrxFileHeader`. Each proof is a length-
/// delimited submessage with `public_key=1` (bytes) and
/// `signature=2` (bytes).
pub fn parse_crx3_proofs(header_pb: &[u8]) -> Vec<CrxProof> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < header_pb.len() {
        let (tag, n) = match read_varint(&header_pb[i..]) {
            Some(x) => x,
            None => break,
        };
        i += n;
        let field = tag >> 3;
        let wire = (tag & 7) as u32;
        if wire != 2 {
            // skip non-length-delimited
            if wire == 0 {
                if let Some((_, n2)) = read_varint(&header_pb[i..]) {
                    i += n2;
                    continue;
                }
            }
            break;
        }
        let (len, n2) = match read_varint(&header_pb[i..]) {
            Some(x) => x,
            None => break,
        };
        i += n2;
        let len = len as usize;
        if i + len > header_pb.len() {
            break;
        }
        let sub = &header_pb[i..i + len];
        i += len;
        if field == 2 || field == 3 {
            // sha256_with_rsa or sha256_with_ecdsa AsymmetricKeyProof.
            if let Some(p) = parse_proof(sub) {
                out.push(p);
            }
        }
    }
    out
}

fn parse_proof(buf: &[u8]) -> Option<CrxProof> {
    let mut i = 0;
    let mut key: Option<Vec<u8>> = None;
    let mut sig: Option<Vec<u8>> = None;
    while i < buf.len() {
        let (tag, n) = read_varint(&buf[i..])?;
        i += n;
        let field = tag >> 3;
        let wire = (tag & 7) as u32;
        if wire != 2 {
            return None;
        }
        let (len, n2) = read_varint(&buf[i..])?;
        i += n2;
        let len = len as usize;
        if i + len > buf.len() {
            return None;
        }
        let val = &buf[i..i + len];
        i += len;
        match field {
            1 => key = Some(val.to_vec()),
            2 => sig = Some(val.to_vec()),
            _ => {}
        }
    }
    Some(CrxProof {
        public_key: key?,
        signature: sig?,
    })
}

fn read_varint(buf: &[u8]) -> Option<(u64, usize)> {
    let mut value: u64 = 0;
    let mut shift = 0;
    let mut i = 0;
    while i < buf.len() {
        let b = buf[i];
        i += 1;
        value |= ((b & 0x7F) as u64) << shift;
        if b & 0x80 == 0 {
            return Some((value, i));
        }
        shift += 7;
        if shift >= 64 {
            return None;
        }
    }
    None
}

/// Verify the CRX3 RSA-SHA256 signature over the canonical signed
/// data: utf8("CRX3 SignedData") || u32-LE(signed_header_size) ||
/// signed_header_data || zip_payload. `header_pb` is the proto header
/// bytes (between offset 12 and `zip_offset`).
///
/// Returns true iff at least one rsa proof verifies.
pub fn verify_crx3_signature(
    header_pb: &[u8],
    zip_payload: &[u8],
    signed_header_data: &[u8],
) -> bool {
    let proofs = parse_crx3_proofs(header_pb);
    if proofs.is_empty() {
        return false;
    }
    let mut to_sign = Vec::with_capacity(15 + 4 + signed_header_data.len() + zip_payload.len());
    to_sign.extend_from_slice(b"CRX3 SignedData");
    to_sign.extend_from_slice(&(signed_header_data.len() as u32).to_le_bytes());
    to_sign.extend_from_slice(signed_header_data);
    to_sign.extend_from_slice(zip_payload);
    for p in &proofs {
        // Pull n, e out of the SPKI bytes.
        let (n_be, e_be) = match parse_rsa_spki(&p.public_key) {
            Some(x) => x,
            None => continue,
        };
        let key = cv_crypto::rsa::RsaPublicKey::from_components(&n_be, &e_be);
        if cv_crypto::rsa::verify_pkcs1_v15(
            &key,
            cv_crypto::rsa::Hash::Sha256,
            &to_sign,
            &p.signature,
        )
        .is_ok()
        {
            return true;
        }
    }
    false
}

/// Walk a DER SubjectPublicKeyInfo to extract the RSA modulus and
/// exponent. The structure is:
///   SEQUENCE { algorithm SEQUENCE, subjectPublicKey BIT STRING containing SEQUENCE { n INTEGER, e INTEGER } }.
fn parse_rsa_spki(spki: &[u8]) -> Option<(Vec<u8>, Vec<u8>)> {
    // outer SEQUENCE
    let (top, _) = read_tlv(spki)?;
    if top.tag != 0x30 {
        return None;
    }
    // skip algorithm SEQUENCE
    let (_alg, rest) = read_tlv(top.value)?;
    // BIT STRING
    let (bs, _) = read_tlv(rest)?;
    if bs.tag != 0x03 {
        return None;
    }
    // first byte of bit string is the count of unused bits — must be 0.
    if bs.value.is_empty() || bs.value[0] != 0 {
        return None;
    }
    let inner = &bs.value[1..];
    let (rsa_seq, _) = read_tlv(inner)?;
    if rsa_seq.tag != 0x30 {
        return None;
    }
    let (n_int, after_n) = read_tlv(rsa_seq.value)?;
    if n_int.tag != 0x02 {
        return None;
    }
    let (e_int, _) = read_tlv(after_n)?;
    if e_int.tag != 0x02 {
        return None;
    }
    Some((
        strip_leading_zero(n_int.value),
        strip_leading_zero(e_int.value),
    ))
}

fn strip_leading_zero(v: &[u8]) -> Vec<u8> {
    if v.first().copied() == Some(0) && v.len() > 1 {
        v[1..].to_vec()
    } else {
        v.to_vec()
    }
}

struct Tlv<'a> {
    tag: u8,
    value: &'a [u8],
}

fn read_tlv(buf: &[u8]) -> Option<(Tlv<'_>, &[u8])> {
    if buf.len() < 2 {
        return None;
    }
    let tag = buf[0];
    let (len, hdr) = if buf[1] & 0x80 == 0 {
        (buf[1] as usize, 2)
    } else {
        let n = (buf[1] & 0x7F) as usize;
        if buf.len() < 2 + n {
            return None;
        }
        let mut len = 0usize;
        for &b in &buf[2..2 + n] {
            len = (len << 8) | b as usize;
        }
        (len, 2 + n)
    };
    if buf.len() < hdr + len {
        return None;
    }
    Some((
        Tlv {
            tag,
            value: &buf[hdr..hdr + len],
        },
        &buf[hdr + len..],
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_valid_minimal_crx() {
        let mut buf = Vec::new();
        buf.extend_from_slice(CRX3_MAGIC);
        buf.extend_from_slice(&3u32.to_le_bytes());
        buf.extend_from_slice(&8u32.to_le_bytes()); // header_len
        buf.extend_from_slice(&[0xAA; 8]); // header payload
        buf.extend_from_slice(&[0xBB; 16]); // zip payload
        let c = parse_crx3(&buf).unwrap();
        assert_eq!(c.version, 3);
        assert_eq!(c.zip_offset, 20);
        assert_eq!(c.zip_size, 16);
    }

    #[test]
    fn rejects_wrong_magic() {
        let mut buf = vec![0u8; 32];
        buf[0..4].copy_from_slice(b"NotC");
        assert!(parse_crx3(&buf).is_none());
    }

    #[test]
    fn rejects_v2() {
        let mut buf = Vec::new();
        buf.extend_from_slice(CRX3_MAGIC);
        buf.extend_from_slice(&2u32.to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes());
        assert!(parse_crx3(&buf).is_none());
    }

    #[test]
    fn rejects_truncated_header() {
        let mut buf = Vec::new();
        buf.extend_from_slice(CRX3_MAGIC);
        buf.extend_from_slice(&3u32.to_le_bytes());
        buf.extend_from_slice(&100u32.to_le_bytes()); // says 100 but buf has 0
        assert!(parse_crx3(&buf).is_none());
    }
}
