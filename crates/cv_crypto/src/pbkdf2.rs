//! PBKDF2 per RFC 8018. Iterates HMAC over a (password || salt || i)
//! input to derive an output of arbitrary length. Hash variants
//! exposed: SHA-1, SHA-256, SHA-384, SHA-512.
//!
//! Used by `crypto.subtle.deriveBits({name:"PBKDF2", hash, salt,
//! iterations}, key, length)` — the standard password-stretching
//! primitive every modern auth flow ships.

/// Derive `output_len` bytes using PBKDF2-HMAC-SHA1.
pub fn pbkdf2_sha1(password: &[u8], salt: &[u8], iterations: u32, output_len: usize) -> Vec<u8> {
    pbkdf2(password, salt, iterations, output_len, 20, |k, d| {
        crate::hmac::HmacSha1::oneshot(k, d).to_vec()
    })
}

pub fn pbkdf2_sha256(password: &[u8], salt: &[u8], iterations: u32, output_len: usize) -> Vec<u8> {
    pbkdf2(password, salt, iterations, output_len, 32, |k, d| {
        crate::hmac::HmacSha256::oneshot(k, d).to_vec()
    })
}

pub fn pbkdf2_sha384(password: &[u8], salt: &[u8], iterations: u32, output_len: usize) -> Vec<u8> {
    pbkdf2(password, salt, iterations, output_len, 48, |k, d| {
        crate::hmac::HmacSha384::oneshot(k, d).to_vec()
    })
}

pub fn pbkdf2_sha512(password: &[u8], salt: &[u8], iterations: u32, output_len: usize) -> Vec<u8> {
    pbkdf2(password, salt, iterations, output_len, 64, |k, d| {
        crate::hmac::HmacSha512::oneshot(k, d).to_vec()
    })
}

fn pbkdf2<F>(
    password: &[u8],
    salt: &[u8],
    iterations: u32,
    output_len: usize,
    hash_len: usize,
    prf: F,
) -> Vec<u8>
where
    F: Fn(&[u8], &[u8]) -> Vec<u8>,
{
    let mut out = Vec::with_capacity(output_len);
    let mut block_index: u32 = 1;
    while out.len() < output_len {
        // U_1 = PRF(password, salt || INT(i))
        let mut concat = salt.to_vec();
        concat.extend_from_slice(&block_index.to_be_bytes());
        let mut u = prf(password, &concat);
        let mut t = u.clone();
        // U_n = PRF(password, U_{n-1}); T_i = U_1 XOR U_2 XOR ... XOR U_c
        for _ in 1..iterations {
            u = prf(password, &u);
            for (tb, ub) in t.iter_mut().zip(u.iter()) {
                *tb ^= *ub;
            }
        }
        let take = (output_len - out.len()).min(hash_len);
        out.extend_from_slice(&t[..take]);
        block_index = block_index.wrapping_add(1);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hex(b: &[u8]) -> String {
        let mut s = String::with_capacity(b.len() * 2);
        for byte in b {
            s.push_str(&format!("{:02x}", byte));
        }
        s
    }

    // RFC 6070 test vector 1.
    #[test]
    fn pbkdf2_sha1_rfc6070_1() {
        let dk = pbkdf2_sha1(b"password", b"salt", 1, 20);
        assert_eq!(hex(&dk), "0c60c80f961f0e71f3a9b524af6012062fe037a6");
    }

    // RFC 6070 test vector 2.
    #[test]
    fn pbkdf2_sha1_rfc6070_2() {
        let dk = pbkdf2_sha1(b"password", b"salt", 2, 20);
        assert_eq!(hex(&dk), "ea6c014dc72d6f8ccd1ed92ace1d41f0d8de8957");
    }

    // RFC 7914 test vector for PBKDF2-HMAC-SHA256 (used by scrypt).
    #[test]
    fn pbkdf2_sha256_rfc7914() {
        let dk = pbkdf2_sha256(b"passwd", b"salt", 1, 64);
        assert_eq!(
            hex(&dk),
            "55ac046e56e3089fec1691c22544b605f94185216dde0465e68b9d57c20dacbc49ca9cccf179b645991664b39d77ef317c71b845b1e30bd509112041d3a19783"
        );
    }
}
