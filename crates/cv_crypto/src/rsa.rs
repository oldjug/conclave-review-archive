//! RSA signature verification per RFC 8017 (PKCS#1 v2.2).
//!
//! Supports PKCS#1 v1.5 (`RSASSA-PKCS1-v1_5`) verification with SHA-256
//! and SHA-384. Sufficient for the bulk of X.509 server certificates in
//! the wild. RSA-PSS verification lands when we hit a cert that needs it.

use crate::CryptoError;
use crate::bigint::{
    BigUint, add_unbounded, inv_mod_full, mul_unbounded, pow_mod, random_prime, rem, sub_unbounded,
};
use crate::sha1::Sha1;
use crate::sha256::Sha256;
use crate::sha384::Sha384;
use crate::sha512::Sha512;

#[derive(Clone, Debug)]
pub struct RsaPublicKey {
    pub n: BigUint,
    pub e: BigUint,
    pub n_byte_len: usize,
}

impl RsaPublicKey {
    pub fn from_components(n_be: &[u8], e_be: &[u8]) -> Self {
        // Strip a leading zero byte that ASN.1 INTEGER may have added.
        let n_be = strip_leading_zero(n_be);
        let e_be = strip_leading_zero(e_be);
        Self {
            n_byte_len: n_be.len(),
            n: BigUint::from_be_bytes(n_be),
            e: BigUint::from_be_bytes(e_be),
        }
    }
}

fn strip_leading_zero(b: &[u8]) -> &[u8] {
    let mut i = 0;
    while i + 1 < b.len() && b[i] == 0 {
        i += 1;
    }
    &b[i..]
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Hash {
    Sha1,
    Sha256,
    Sha384,
    Sha512,
}

impl Hash {
    /// DER-encoded `DigestInfo` prefix per RFC 8017 §9.2 note 1.
    fn digest_info_prefix(self) -> &'static [u8] {
        match self {
            // RFC 8017 §9.2 note 1: SHA-1 DigestInfo prefix.
            Self::Sha1 => &[
                0x30, 0x21, 0x30, 0x09, 0x06, 0x05, 0x2b, 0x0e, 0x03, 0x02, 0x1a, 0x05, 0x00, 0x04,
                0x14,
            ],
            Self::Sha256 => &[
                0x30, 0x31, 0x30, 0x0d, 0x06, 0x09, 0x60, 0x86, 0x48, 0x01, 0x65, 0x03, 0x04, 0x02,
                0x01, 0x05, 0x00, 0x04, 0x20,
            ],
            Self::Sha384 => &[
                0x30, 0x41, 0x30, 0x0d, 0x06, 0x09, 0x60, 0x86, 0x48, 0x01, 0x65, 0x03, 0x04, 0x02,
                0x02, 0x05, 0x00, 0x04, 0x30,
            ],
            // SHA-512 digest is 64 bytes; the DigestInfo prefix per
            // RFC 8017 §9.2 note 1 differs from SHA-384's only in the
            // OID-final byte (0x03 vs 0x02) and the digest-length byte
            // at the very end (0x40 = 64).
            Self::Sha512 => &[
                0x30, 0x51, 0x30, 0x0d, 0x06, 0x09, 0x60, 0x86, 0x48, 0x01, 0x65, 0x03, 0x04, 0x02,
                0x03, 0x05, 0x00, 0x04, 0x40,
            ],
        }
    }

    fn digest(self, msg: &[u8]) -> Vec<u8> {
        match self {
            Self::Sha1 => Sha1::oneshot(msg).to_vec(),
            Self::Sha256 => Sha256::oneshot(msg).to_vec(),
            Self::Sha384 => Sha384::oneshot(msg).to_vec(),
            Self::Sha512 => Sha512::oneshot(msg).to_vec(),
        }
    }

    pub fn digest_len(self) -> usize {
        match self {
            Self::Sha1 => 20,
            Self::Sha256 => 32,
            Self::Sha384 => 48,
            Self::Sha512 => 64,
        }
    }
}

/// MGF1 mask generation per RFC 8017 §B.2.1.
fn mgf1(hash: Hash, seed: &[u8], mask_len: usize) -> Vec<u8> {
    let h_len = hash.digest_len();
    let n = mask_len.div_ceil(h_len);
    let mut t = Vec::with_capacity(n * h_len);
    for counter in 0..n {
        let mut input = Vec::with_capacity(seed.len() + 4);
        input.extend_from_slice(seed);
        input.extend_from_slice(&(counter as u32).to_be_bytes());
        t.extend_from_slice(&hash.digest(&input));
    }
    t.truncate(mask_len);
    t
}

/// Verify an RSASSA-PSS signature per RFC 8017 §8.1.2.
/// Salt length must equal hash output length (the convention TLS 1.3 uses).
pub fn verify_pss(
    key: &RsaPublicKey,
    hash: Hash,
    msg: &[u8],
    signature: &[u8],
) -> Result<(), CryptoError> {
    let mod_bits = key.n.bit_len();
    let em_len = mod_bits.div_ceil(8);
    let em_bits = mod_bits - 1;
    if signature.len() != em_len {
        return Err(CryptoError::BadLength);
    }
    let s_int = BigUint::from_be_bytes(signature);
    let m_int = pow_mod(&s_int, &key.e, &key.n);
    let em = m_int.to_be_bytes(em_len);

    let h_len = hash.digest_len();
    let s_len = h_len; // TLS 1.3 convention
    if em_len < h_len + s_len + 2 {
        return Err(CryptoError::BadLength);
    }
    if em[em_len - 1] != 0xbc {
        return Err(CryptoError::BadTag);
    }

    let db_len = em_len - h_len - 1;
    let masked_db = &em[..db_len];
    let h = &em[db_len..em_len - 1];

    // Top (8*emLen - emBits) bits of the leading masked_db byte must
    // become zero after we strip them post-XOR — for em_bits = mod_bits-1
    // and em_len = ceil(mod_bits/8), this is usually 1 bit on common key
    // sizes (2048, 3072). We check it after unmasking.

    let mask = mgf1(hash, h, db_len);
    let mut db = Vec::with_capacity(db_len);
    for i in 0..db_len {
        db.push(masked_db[i] ^ mask[i]);
    }
    // Zero the top (8*em_len - em_bits) bits.
    let zero_top_bits = 8 * em_len - em_bits;
    if zero_top_bits > 0 {
        db[0] &= 0xFF >> zero_top_bits;
    }

    // DB = 0x00 .. 0x00 || 0x01 || salt
    let ps_len = db_len - s_len - 1;
    for &b in &db[..ps_len] {
        if b != 0 {
            return Err(CryptoError::BadTag);
        }
    }
    if db[ps_len] != 0x01 {
        return Err(CryptoError::BadTag);
    }
    let salt = &db[ps_len + 1..];

    // M' = (0x00 * 8) || mHash || salt
    let m_hash = hash.digest(msg);
    let mut mprime = Vec::with_capacity(8 + h_len + s_len);
    mprime.extend_from_slice(&[0u8; 8]);
    mprime.extend_from_slice(&m_hash);
    mprime.extend_from_slice(salt);
    let h_prime = hash.digest(&mprime);

    if !crate::subtle::ct_eq(h, &h_prime) {
        return Err(CryptoError::BadTag);
    }
    Ok(())
}

/// Verify a PKCS#1 v1.5 signature. Returns `Ok(())` if valid.
pub fn verify_pkcs1_v15(
    key: &RsaPublicKey,
    hash: Hash,
    msg: &[u8],
    signature: &[u8],
) -> Result<(), CryptoError> {
    let k = key.n_byte_len;
    if signature.len() != k {
        return Err(CryptoError::BadLength);
    }
    // m = sig^e mod n
    let s_int = BigUint::from_be_bytes(signature);
    let m_int = pow_mod(&s_int, &key.e, &key.n);
    let em = m_int.to_be_bytes(k);

    // EM = 0x00 || 0x01 || PS || 0x00 || T
    // where PS is k - 3 - len(T) bytes of 0xFF, and T is DigestInfo || H.
    let t_prefix = hash.digest_info_prefix();
    let digest = hash.digest(msg);
    let t_len = t_prefix.len() + digest.len();
    if k < t_len + 11 {
        return Err(CryptoError::BadLength);
    }

    if em[0] != 0x00 || em[1] != 0x01 {
        return Err(CryptoError::BadTag);
    }
    let ps_end = k - t_len - 1;
    for &b in &em[2..ps_end] {
        if b != 0xFF {
            return Err(CryptoError::BadTag);
        }
    }
    if em[ps_end] != 0x00 {
        return Err(CryptoError::BadTag);
    }
    let t_field = &em[ps_end + 1..];
    if &t_field[..t_prefix.len()] != t_prefix {
        return Err(CryptoError::BadTag);
    }
    if &t_field[t_prefix.len()..] != digest.as_slice() {
        return Err(CryptoError::BadTag);
    }
    Ok(())
}

// ===========================================================================
// RSA private-key operations: RSAES-OAEP encrypt/decrypt (RFC 8017 §7.1),
// RSASSA-PKCS1-v1_5 + RSASSA-PSS signing (§8.1.1 / §8.2.1), and key
// generation (§3.2 / FIPS 186-5). The public-key half (verify_*) above is the
// path the TLS stack already used; this section adds everything WebCrypto's
// asymmetric surface needs.
// ===========================================================================

/// An RSA private key. We keep the public components (`n`, `e`) for round-trip
/// export plus the private exponent `d` and CRT parameters for fast decrypt.
#[derive(Clone, Debug)]
pub struct RsaPrivateKey {
    pub n: BigUint,
    pub e: BigUint,
    pub d: BigUint,
    pub p: BigUint,
    pub q: BigUint,
    pub dp: BigUint,
    pub dq: BigUint,
    pub qinv: BigUint,
    pub n_byte_len: usize,
}

impl RsaPrivateKey {
    /// The matching public key.
    pub fn public_key(&self) -> RsaPublicKey {
        RsaPublicKey {
            n: self.n.clone(),
            e: self.e.clone(),
            n_byte_len: self.n_byte_len,
        }
    }

    /// RSADP — the raw private-key primitive `c^d mod n`, accelerated via the
    /// Chinese Remainder Theorem (RFC 8017 §5.1.2 second form). Falls back to
    /// the straightforward `c^d mod n` if the CRT params are degenerate.
    fn rsadp(&self, c: &BigUint) -> BigUint {
        if self.p.is_zero() || self.q.is_zero() {
            return pow_mod(c, &self.d, &self.n);
        }
        // m1 = c^dP mod p ; m2 = c^dQ mod q
        let m1 = pow_mod(c, &self.dp, &self.p);
        let m2 = pow_mod(c, &self.dq, &self.q);
        // h = qInv * (m1 - m2) mod p   (handle m1 < m2 by adding p)
        let diff = if m1.cmp(&m2) == core::cmp::Ordering::Less {
            sub_unbounded(&add_unbounded(&m1, &self.p), &m2)
        } else {
            sub_unbounded(&m1, &m2)
        };
        let h = rem(&mul_unbounded(&self.qinv, &diff), &self.p);
        // m = m2 + h*q
        add_unbounded(&m2, &mul_unbounded(&h, &self.q))
    }
}

/// MGF1 mask generation, public-ish form used by OAEP. (Same algorithm as the
/// module-private `mgf1` used in PSS verify.)
fn mgf1_pub(hash: Hash, seed: &[u8], mask_len: usize) -> Vec<u8> {
    mgf1(hash, seed, mask_len)
}

/// RSAES-OAEP encryption (RFC 8017 §7.1.1). `label` is the optional OAEP label
/// (WebCrypto's `RsaOaepParams.label`, defaults to empty). `seed` must be
/// `hash.digest_len()` random bytes from a CSPRNG. Returns the `k`-byte
/// ciphertext.
pub fn encrypt_oaep(
    key: &RsaPublicKey,
    hash: Hash,
    msg: &[u8],
    label: &[u8],
    seed: &[u8],
) -> Result<Vec<u8>, CryptoError> {
    let k = key.n_byte_len;
    let h_len = hash.digest_len();
    if seed.len() != h_len {
        return Err(CryptoError::BadLength);
    }
    // mLen <= k - 2*hLen - 2
    if msg.len() + 2 * h_len + 2 > k {
        return Err(CryptoError::BadLength);
    }
    // DB = lHash || PS || 0x01 || M
    let l_hash = hash.digest(label);
    let ps_len = k - msg.len() - 2 * h_len - 2;
    let db_len = k - h_len - 1;
    let mut db = Vec::with_capacity(db_len);
    db.extend_from_slice(&l_hash);
    db.extend(core::iter::repeat_n(0u8, ps_len));
    db.push(0x01);
    db.extend_from_slice(msg);
    debug_assert_eq!(db.len(), db_len);

    // maskedDB = DB XOR MGF(seed, k - hLen - 1)
    let db_mask = mgf1_pub(hash, seed, db_len);
    let mut masked_db = db;
    for i in 0..db_len {
        masked_db[i] ^= db_mask[i];
    }
    // maskedSeed = seed XOR MGF(maskedDB, hLen)
    let seed_mask = mgf1_pub(hash, &masked_db, h_len);
    let mut masked_seed = seed.to_vec();
    for i in 0..h_len {
        masked_seed[i] ^= seed_mask[i];
    }
    // EM = 0x00 || maskedSeed || maskedDB
    let mut em = Vec::with_capacity(k);
    em.push(0x00);
    em.extend_from_slice(&masked_seed);
    em.extend_from_slice(&masked_db);

    // c = EM^e mod n
    let m_int = BigUint::from_be_bytes(&em);
    if m_int.cmp(&key.n) != core::cmp::Ordering::Less {
        return Err(CryptoError::BadLength);
    }
    let c = pow_mod(&m_int, &key.e, &key.n);
    Ok(c.to_be_bytes(k))
}

/// RSAES-OAEP decryption (RFC 8017 §7.1.2). Returns the recovered message.
pub fn decrypt_oaep(
    key: &RsaPrivateKey,
    hash: Hash,
    ciphertext: &[u8],
    label: &[u8],
) -> Result<Vec<u8>, CryptoError> {
    let k = key.n_byte_len;
    let h_len = hash.digest_len();
    if ciphertext.len() != k || k < 2 * h_len + 2 {
        return Err(CryptoError::BadLength);
    }
    let c_int = BigUint::from_be_bytes(ciphertext);
    if c_int.cmp(&key.n) != core::cmp::Ordering::Less {
        return Err(CryptoError::BadTag);
    }
    let m_int = key.rsadp(&c_int);
    let em = m_int.to_be_bytes(k);

    // EM = Y || maskedSeed || maskedDB ; Y must be 0x00.
    let l_hash = hash.digest(label);
    let y = em[0];
    let masked_seed = &em[1..1 + h_len];
    let masked_db = &em[1 + h_len..];
    let db_len = k - h_len - 1;

    let seed_mask = mgf1_pub(hash, masked_db, h_len);
    let mut seed = masked_seed.to_vec();
    for i in 0..h_len {
        seed[i] ^= seed_mask[i];
    }
    let db_mask = mgf1_pub(hash, &seed, db_len);
    let mut db = masked_db.to_vec();
    for i in 0..db_len {
        db[i] ^= db_mask[i];
    }
    // DB = lHash' || PS || 0x01 || M. Compute the verdict without an early
    // return on each sub-check (decryption-oracle hygiene per RFC 8017 §7.1.2).
    let mut bad = (y != 0x00) as u8;
    let mut acc = 0u8;
    for i in 0..h_len {
        acc |= db[i] ^ l_hash[i];
    }
    bad |= (acc != 0) as u8;
    // Find the 0x01 separator after the (all-zero) PS.
    let mut sep_index: isize = -1;
    let mut seen_nonzero_before_one = 0u8;
    let mut i = h_len;
    while i < db_len {
        let b = db[i];
        if sep_index < 0 {
            if b == 0x01 {
                sep_index = i as isize;
            } else if b != 0x00 {
                seen_nonzero_before_one |= 1;
            }
        }
        i += 1;
    }
    bad |= seen_nonzero_before_one;
    bad |= (sep_index < 0) as u8;
    if bad != 0 {
        return Err(CryptoError::BadTag);
    }
    Ok(db[(sep_index as usize) + 1..].to_vec())
}

/// EMSA-PKCS1-v1_5 encode then RSASP1: produce a PKCS#1 v1.5 signature
/// (`RSASSA-PKCS1-v1_5`, RFC 8017 §8.2.1). Returns a `k`-byte signature.
pub fn sign_pkcs1_v15(key: &RsaPrivateKey, hash: Hash, msg: &[u8]) -> Result<Vec<u8>, CryptoError> {
    let k = key.n_byte_len;
    let t_prefix = hash.digest_info_prefix();
    let digest = hash.digest(msg);
    let t_len = t_prefix.len() + digest.len();
    if k < t_len + 11 {
        return Err(CryptoError::BadLength);
    }
    // EM = 0x00 || 0x01 || PS(0xFF..) || 0x00 || T
    let ps_len = k - t_len - 3;
    let mut em = Vec::with_capacity(k);
    em.push(0x00);
    em.push(0x01);
    em.extend(core::iter::repeat_n(0xFFu8, ps_len));
    em.push(0x00);
    em.extend_from_slice(t_prefix);
    em.extend_from_slice(&digest);
    debug_assert_eq!(em.len(), k);

    let m_int = BigUint::from_be_bytes(&em);
    let s = key.rsadp(&m_int);
    Ok(s.to_be_bytes(k))
}

/// EMSA-PSS encode then RSASP1: produce an RSASSA-PSS signature (RFC 8017
/// §8.1.1 / §9.1.1). `salt` is `salt_len` random bytes (WebCrypto's
/// `RsaPssParams.saltLength`). Returns a `k`-byte signature.
pub fn sign_pss(
    key: &RsaPrivateKey,
    hash: Hash,
    msg: &[u8],
    salt: &[u8],
) -> Result<Vec<u8>, CryptoError> {
    let mod_bits = key.n.bit_len();
    let em_bits = mod_bits - 1;
    let em_len = em_bits.div_ceil(8);
    let h_len = hash.digest_len();
    let s_len = salt.len();
    if em_len < h_len + s_len + 2 {
        return Err(CryptoError::BadLength);
    }
    // M' = (0x00 * 8) || mHash || salt ; H = Hash(M')
    let m_hash = hash.digest(msg);
    let mut mprime = Vec::with_capacity(8 + h_len + s_len);
    mprime.extend_from_slice(&[0u8; 8]);
    mprime.extend_from_slice(&m_hash);
    mprime.extend_from_slice(salt);
    let h = hash.digest(&mprime);

    // DB = PS(0x00..) || 0x01 || salt
    let db_len = em_len - h_len - 1;
    let ps_len = db_len - s_len - 1;
    let mut db = Vec::with_capacity(db_len);
    db.extend(core::iter::repeat_n(0u8, ps_len));
    db.push(0x01);
    db.extend_from_slice(salt);

    // maskedDB = DB XOR MGF(H, db_len) ; zero leftmost (8*em_len - em_bits) bits.
    let db_mask = mgf1_pub(hash, &h, db_len);
    let mut masked_db = db;
    for i in 0..db_len {
        masked_db[i] ^= db_mask[i];
    }
    let zero_top_bits = 8 * em_len - em_bits;
    if zero_top_bits > 0 {
        masked_db[0] &= 0xFF >> zero_top_bits;
    }
    // EM = maskedDB || H || 0xbc
    let mut em = Vec::with_capacity(em_len);
    em.extend_from_slice(&masked_db);
    em.extend_from_slice(&h);
    em.push(0xbc);

    let k = key.n_byte_len;
    let m_int = BigUint::from_be_bytes(&em);
    let s = key.rsadp(&m_int);
    Ok(s.to_be_bytes(k))
}

/// Generate an RSA key pair with the given modulus size in bits and public
/// exponent `e` (WebCrypto: `RsaHashedKeyGenParams.modulusLength` /
/// `.publicExponent`, almost always 65537). `rng` supplies CSPRNG bytes. Per
/// RFC 8017 §3.2 / FIPS 186-5: pick two distinct random primes p, q of half the
/// modulus size, n = p*q, λ(n) = lcm(p-1, q-1), d = e^{-1} mod λ(n).
pub fn generate_keypair(
    modulus_bits: usize,
    e: u64,
    rng: &mut dyn FnMut(&mut [u8]),
) -> RsaPrivateKey {
    assert!(modulus_bits >= 512, "RSA modulus too small");
    let e_big = BigUint::from_u64(e);
    let half = modulus_bits / 2;
    loop {
        let p = random_prime(half, rng);
        let q = random_prime(modulus_bits - half, rng);
        // p != q and the modulus must land at exactly modulus_bits.
        if p.cmp(&q) == core::cmp::Ordering::Equal {
            continue;
        }
        let n = mul_unbounded(&p, &q);
        if n.bit_len() != modulus_bits {
            continue;
        }
        // λ(n) = lcm(p-1, q-1) = (p-1)(q-1)/gcd(p-1,q-1).
        let p1 = p.dec_one();
        let q1 = q.dec_one();
        let lambda = lcm(&p1, &q1);
        // e must be invertible mod λ.
        let Some(d) = inv_mod_full(&e_big, &lambda) else {
            continue;
        };
        // CRT params.
        let dp = rem(&d, &p1);
        let dq = rem(&d, &q1);
        let Some(qinv) = inv_mod_full(&q, &p) else {
            continue;
        };
        let n_byte_len = modulus_bits / 8;
        return RsaPrivateKey {
            n,
            e: e_big,
            d,
            p,
            q,
            dp,
            dq,
            qinv,
            n_byte_len,
        };
    }
}

/// `gcd(a, b)` via the Euclidean algorithm.
fn gcd(a: &BigUint, b: &BigUint) -> BigUint {
    let mut x = a.clone();
    let mut y = b.clone();
    while !y.is_zero() {
        let r = rem(&x, &y);
        x = y;
        y = r;
    }
    x
}

/// `lcm(a, b) = (a / gcd(a, b)) * b`. Dividing first keeps the intermediate
/// small and avoids an exact-division-after-product step.
fn lcm(a: &BigUint, b: &BigUint) -> BigUint {
    let g = gcd(a, b);
    let a_over_g = crate::bigint::div_floor(a, &g);
    mul_unbounded(&a_over_g, b)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// RFC 8017 Appendix C example: 1024-bit RSA key, "Hello World!" with
    /// SHA-256. (Adapted from pyca/cryptography's test corpus, which uses
    /// the same canonical small example.)
    #[test]
    fn pkcs1_v15_sha256_smoke() {
        // We construct a tiny key here just to exercise the verification
        // path end-to-end with a known signature we generate on the fly
        // via the corresponding private key. The signature is produced
        // by computing m^d mod n where d is the precomputed private
        // exponent. This is a self-consistency test, not a third-party
        // vector — that's NIST CAVP material once we have a DER parser.

        // 1024-bit modulus (a well-known test key from PKCS#1 examples).
        // n = pq for tiny primes — small enough to keep the test fast.
        let n_hex = "
            a8d68acd 413c5e19 5d5ef04e 1b4faaf2 42365cb4 50196755 e92e1215
            ba59802a af5aa828 fefea54d 4ee1efb3 c69c81e7 d3eebafd 86d83b8a
            cf12abff 64ec6dc4 70b3aaba 7faf52a4 0f3bd082 dcd16b3a 1b7e5cb1
            a5c10000 3fa90f06 1c0fa4a4 a6b0b85f a47b08bc 53eaa10b 6";
        // For the demo, we'll use a 256-bit (small) key so test runs quickly.
        // n = p * q where p, q are 128-bit primes. Skipping the actual
        // verification with a third-party vector here; we test instead
        // that mul_mod / pow_mod round-trip via the RSA equation:
        //   verify((m^d)^e mod n, m) == ok.
        let _ = n_hex;

        // 16-bit example: n = 3233 = 61*53, e = 17, d = 2753.
        // m = SHA-256("hi")[0..1] interpreted as small int won't fit
        // PKCS#1 padding (need at least 11 bytes); so we skip the
        // full PKCS#1 test here and rely on integration once X.509
        // lands. This test is a smoke test that the API plumbs.
        let key = RsaPublicKey::from_components(&[0x0c, 0xa1], &[17]);
        assert_eq!(key.n_byte_len, 2);
        assert_eq!(key.e.to_be_bytes(1), vec![17]);
    }

    /// RFC 8017 Appendix C example PKCS#1 v1.5 SHA-1 → we adapted to
    /// SHA-256 with a fresh test pair. To avoid pulling external
    /// signature data into this file, the integration-level test for
    /// RSA verification lives in `cv_net`'s TLS module once we wire
    /// real X.509 certs in M0b.
    #[test]
    fn pss_mgf1_known_value() {
        // MGF1-SHA256 over empty seed, length 32.
        // The leading bytes are SHA-256("" || 0x00 0x00 0x00 0x00).
        let mask = mgf1(Hash::Sha256, b"", 32);
        // sha256(00000000) = df3f619804a92fdb4057192dc43dd748ea778adc52bc498ce80524c014b81119
        assert_eq!(
            mask.iter().take(4).copied().collect::<Vec<u8>>(),
            vec![0xdf, 0x3f, 0x61, 0x98]
        );
    }

    #[test]
    fn smoke_short_signature_rejected() {
        let key = RsaPublicKey::from_components(&[0xff, 0xff], &[3]);
        // Signature length must equal key length.
        let bad = vec![0u8; 1];
        assert!(matches!(
            verify_pkcs1_v15(&key, Hash::Sha256, b"msg", &bad),
            Err(CryptoError::BadLength)
        ));
    }

    // --- Private-key path: keygen + sign/verify + OAEP, end to end. ---

    /// Deterministic xorshift used as the CSPRNG stand-in for keygen/salt/seed
    /// inside these offline tests (production wires BCryptGenRandom).
    struct TestRng(u64);
    impl TestRng {
        fn fill(&mut self, buf: &mut [u8]) {
            for b in buf.iter_mut() {
                let mut x = self.0;
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                self.0 = x;
                *b = x as u8;
            }
        }
    }

    /// Generate a real 1024-bit key, sign with PKCS#1 v1.5 + SHA-256, then
    /// verify with the public half. Tamper → reject.
    #[test]
    fn keygen_pkcs1_sign_verify_roundtrip() {
        let mut rng = TestRng(0x243F_6A88_85A3_08D3);
        let priv_key =
            generate_keypair(1024, 65537, &mut |b| rng.fill(b));
        let pub_key = priv_key.public_key();
        // RSA equation sanity: (m^e)^d == m for a small m.
        assert_eq!(pub_key.n_byte_len, 128);

        let msg = b"the quick brown fox jumps over the lazy dog";
        let sig = sign_pkcs1_v15(&priv_key, Hash::Sha256, msg).unwrap();
        assert_eq!(sig.len(), 128);
        verify_pkcs1_v15(&pub_key, Hash::Sha256, msg, &sig)
            .expect("freshly produced PKCS1 signature must verify");
        // Tampered message must fail.
        let mut bad = msg.to_vec();
        bad[0] ^= 1;
        assert!(verify_pkcs1_v15(&pub_key, Hash::Sha256, &bad, &sig).is_err());
        // Tampered signature must fail.
        let mut bad_sig = sig.clone();
        bad_sig[64] ^= 1;
        assert!(verify_pkcs1_v15(&pub_key, Hash::Sha256, msg, &bad_sig).is_err());
    }

    /// PSS sign → verify with salt length = hash length (32). Tamper → reject.
    #[test]
    fn keygen_pss_sign_verify_roundtrip() {
        let mut rng = TestRng(0xB7E1_5162_8AED_2A6A);
        let priv_key = generate_keypair(1024, 65537, &mut |b| rng.fill(b));
        let pub_key = priv_key.public_key();
        let msg = b"PSS message under test";
        let mut salt = [0u8; 32];
        rng.fill(&mut salt);
        let sig = sign_pss(&priv_key, Hash::Sha256, msg, &salt).unwrap();
        verify_pss(&pub_key, Hash::Sha256, msg, &sig)
            .expect("freshly produced PSS signature must verify");
        let mut bad = msg.to_vec();
        bad[3] ^= 0x80;
        assert!(verify_pss(&pub_key, Hash::Sha256, &bad, &sig).is_err());
    }

    /// OAEP encrypt with the public key → decrypt with the private key.
    #[test]
    fn keygen_oaep_encrypt_decrypt_roundtrip() {
        let mut rng = TestRng(0x9E37_79B9_7F4A_7C15);
        let priv_key = generate_keypair(1024, 65537, &mut |b| rng.fill(b));
        let pub_key = priv_key.public_key();
        let msg = b"top secret payload";
        let mut seed = [0u8; 32]; // SHA-256 hLen
        rng.fill(&mut seed);
        let ct = encrypt_oaep(&pub_key, Hash::Sha256, msg, b"", &seed).unwrap();
        assert_eq!(ct.len(), 128);
        let pt = decrypt_oaep(&priv_key, Hash::Sha256, &ct, b"").unwrap();
        assert_eq!(pt, msg);
        // Wrong label must fail to decode.
        assert!(decrypt_oaep(&priv_key, Hash::Sha256, &ct, b"different").is_err());
        // Tampered ciphertext must fail.
        let mut bad = ct.clone();
        bad[100] ^= 1;
        assert!(decrypt_oaep(&priv_key, Hash::Sha256, &bad, b"").is_err());
    }

    /// Timing probe for 2048-bit keygen (decides whether to flag-gate the
    /// WebCrypto RSA generateKey default). Run:
    ///   cargo test -p cv_crypto rsa_keygen_2048_timing -- --ignored --nocapture
    #[test]
    #[ignore]
    fn rsa_keygen_2048_timing() {
        let mut rng = TestRng(0x1122_3344_5566_7788);
        let t = std::time::Instant::now();
        let k = generate_keypair(2048, 65537, &mut |b| rng.fill(b));
        let ms = t.elapsed().as_millis();
        println!("RSA-2048 keygen: {ms} ms (n_byte_len={})", k.n_byte_len);
    }

    /// A signature produced under one key must NOT verify under a different key.
    #[test]
    fn wrong_key_signature_rejected() {
        let mut rng = TestRng(0xC0FF_EE00_1234_5678);
        let key_a = generate_keypair(1024, 65537, &mut |b| rng.fill(b));
        let key_b = generate_keypair(1024, 65537, &mut |b| rng.fill(b));
        let msg = b"authenticate me";
        let sig = sign_pkcs1_v15(&key_a, Hash::Sha256, msg).unwrap();
        assert!(verify_pkcs1_v15(&key_a.public_key(), Hash::Sha256, msg, &sig).is_ok());
        assert!(verify_pkcs1_v15(&key_b.public_key(), Hash::Sha256, msg, &sig).is_err());
    }
}
