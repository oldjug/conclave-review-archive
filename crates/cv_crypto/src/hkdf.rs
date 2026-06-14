//! HKDF-SHA256 + HKDF-SHA384 per RFC 5869.  The KDFs TLS 1.3 builds
//! its key schedule on (SHA-256 for the default suite, SHA-384 for
//! `TLS_AES_256_GCM_SHA384`).

use crate::hmac::{HMAC_SHA384_OUTPUT_SIZE, HmacSha256, HmacSha384, OUTPUT_SIZE};

pub fn extract(salt: &[u8], ikm: &[u8]) -> [u8; OUTPUT_SIZE] {
    let salt = if salt.is_empty() {
        &[0u8; OUTPUT_SIZE][..]
    } else {
        salt
    };
    HmacSha256::oneshot(salt, ikm)
}

pub fn expand(prk: &[u8], info: &[u8], out: &mut [u8]) {
    assert!(
        out.len() <= 255 * OUTPUT_SIZE,
        "HKDF expand length too large"
    );
    let mut prev: Vec<u8> = Vec::new();
    let mut written = 0usize;
    let mut counter: u8 = 1;
    while written < out.len() {
        let mut h = HmacSha256::new(prk);
        h.update(&prev);
        h.update(info);
        h.update(&[counter]);
        let block = h.finalize();
        let take = (out.len() - written).min(OUTPUT_SIZE);
        out[written..written + take].copy_from_slice(&block[..take]);
        written += take;
        prev = block.to_vec();
        counter = counter.wrapping_add(1);
    }
}

pub fn extract_and_expand(salt: &[u8], ikm: &[u8], info: &[u8], out: &mut [u8]) {
    let prk = extract(salt, ikm);
    expand(&prk, info, out);
}

// ----------------------- HKDF-SHA384 (RFC 5869) ----------------------

pub fn extract_sha384(salt: &[u8], ikm: &[u8]) -> [u8; HMAC_SHA384_OUTPUT_SIZE] {
    let salt = if salt.is_empty() {
        &[0u8; HMAC_SHA384_OUTPUT_SIZE][..]
    } else {
        salt
    };
    HmacSha384::oneshot(salt, ikm)
}

pub fn expand_sha384(prk: &[u8], info: &[u8], out: &mut [u8]) {
    assert!(
        out.len() <= 255 * HMAC_SHA384_OUTPUT_SIZE,
        "HKDF expand length too large"
    );
    let mut prev: Vec<u8> = Vec::new();
    let mut written = 0usize;
    let mut counter: u8 = 1;
    while written < out.len() {
        let mut h = HmacSha384::new(prk);
        h.update(&prev);
        h.update(info);
        h.update(&[counter]);
        let block = h.finalize();
        let take = (out.len() - written).min(HMAC_SHA384_OUTPUT_SIZE);
        out[written..written + take].copy_from_slice(&block[..take]);
        written += take;
        prev = block.to_vec();
        counter = counter.wrapping_add(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hex(b: &[u8]) -> String {
        let mut s = String::with_capacity(b.len() * 2);
        for x in b {
            s.push_str(&format!("{x:02x}"));
        }
        s
    }

    fn unhex(s: &str) -> Vec<u8> {
        let s: String = s.chars().filter(|c| !c.is_whitespace()).collect();
        (0..s.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap())
            .collect()
    }

    /// RFC 5869 Appendix A.1.
    #[test]
    fn rfc5869_a1() {
        let ikm = unhex("0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b");
        let salt = unhex("000102030405060708090a0b0c");
        let info = unhex("f0f1f2f3f4f5f6f7f8f9");

        let prk = extract(&salt, &ikm);
        assert_eq!(
            hex(&prk),
            "077709362c2e32df0ddc3f0dc47bba6390b6c73bb50f9c3122ec844ad7c2b3e5"
        );

        let mut okm = [0u8; 42];
        expand(&prk, &info, &mut okm);
        assert_eq!(
            hex(&okm),
            "3cb25f25faacd57a90434f64d0362f2a2d2d0a90cf1a5a4c5db02d56ecc4c5bf34007208d5b887185865"
        );
    }

    /// RFC 5869 Appendix A.3: empty salt and info.
    #[test]
    fn rfc5869_a3() {
        let ikm = unhex("0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b");
        let salt: Vec<u8> = vec![];
        let info: Vec<u8> = vec![];
        let prk = extract(&salt, &ikm);
        assert_eq!(
            hex(&prk),
            "19ef24a32c717b167f33a91d6f648bdf96596776afdb6377ac434c1c293ccb04"
        );
        let mut okm = [0u8; 42];
        expand(&prk, &info, &mut okm);
        assert_eq!(
            hex(&okm),
            "8da4e775a563c18f715f802a063c5a31b8a11f5c5ee1879ec3454e5f3c738d2d9d201395faa4b61a96c8"
        );
    }
}
