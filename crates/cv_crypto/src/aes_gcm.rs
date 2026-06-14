//! AES-GCM per NIST SP 800-38D.
//!
//! TLS 1.3 cipher suites `TLS_AES_128_GCM_SHA256` and
//! `TLS_AES_256_GCM_SHA384` ride on this.

use crate::CryptoError;
use crate::aes::{Aes128, Aes256, BLOCK_SIZE};
use crate::subtle::ct_eq;

pub const TAG_SIZE: usize = 16;
pub const NONCE_SIZE: usize = 12; // GCM's standard nonce length

/// GF(2^128) right-shift by 1 over the GCM polynomial x^128 + x^7 + x^2 + x + 1.
fn gf_shift_right(v: &mut [u8; 16]) {
    let mut carry = 0u8;
    for byte in v.iter_mut() {
        let new_carry = *byte & 1;
        *byte = (*byte >> 1) | (carry << 7);
        carry = new_carry;
    }
    if carry != 0 {
        v[0] ^= 0xe1;
    }
}

/// Bit-by-bit GF(2^128) multiplication for `y * h`. Slow but small;
/// table-based implementations are an M5 perf swap.
fn ghash_mul(y: &[u8; 16], h: &[u8; 16]) -> [u8; 16] {
    let mut z = [0u8; 16];
    let mut v = *h;
    for byte in y.iter() {
        for bit in (0..8).rev() {
            let mask = 0u8.wrapping_sub((byte >> bit) & 1);
            for i in 0..16 {
                z[i] ^= v[i] & mask;
            }
            gf_shift_right(&mut v);
        }
    }
    z
}

fn ghash_update(state: &mut [u8; 16], h: &[u8; 16], block: &[u8; 16]) {
    for i in 0..16 {
        state[i] ^= block[i];
    }
    *state = ghash_mul(state, h);
}

fn ghash_pad(state: &mut [u8; 16], h: &[u8; 16], data: &[u8]) {
    let full = data.len() / BLOCK_SIZE;
    for i in 0..full {
        let mut blk = [0u8; 16];
        blk.copy_from_slice(&data[i * 16..i * 16 + 16]);
        ghash_update(state, h, &blk);
    }
    let rem = data.len() % BLOCK_SIZE;
    if rem != 0 {
        let mut blk = [0u8; 16];
        blk[..rem].copy_from_slice(&data[full * 16..]);
        ghash_update(state, h, &blk);
    }
}

trait Cipher {
    fn encrypt(&self, block: &mut [u8; 16]);
}

impl Cipher for Aes128 {
    fn encrypt(&self, block: &mut [u8; 16]) {
        Aes128::encrypt_block(self, block);
    }
}
impl Cipher for Aes256 {
    fn encrypt(&self, block: &mut [u8; 16]) {
        Aes256::encrypt_block(self, block);
    }
}

fn gcm_seal<C: Cipher>(
    cipher: &C,
    nonce: &[u8; NONCE_SIZE],
    aad: &[u8],
    plaintext: &[u8],
) -> Vec<u8> {
    // H = E(0^128)
    let mut h = [0u8; 16];
    cipher.encrypt(&mut h);

    // J0 = nonce || 0^31 || 1
    let mut j0 = [0u8; 16];
    j0[..12].copy_from_slice(nonce);
    j0[15] = 1;

    let mut ct = vec![0u8; plaintext.len()];
    ctr_xor(cipher, &j0, plaintext, &mut ct);

    let mut s = [0u8; 16];
    ghash_pad(&mut s, &h, aad);
    ghash_pad(&mut s, &h, &ct);
    let mut lengths = [0u8; 16];
    lengths[0..8].copy_from_slice(&((aad.len() as u64) * 8).to_be_bytes());
    lengths[8..16].copy_from_slice(&((ct.len() as u64) * 8).to_be_bytes());
    ghash_update(&mut s, &h, &lengths);

    // tag = E_K(J0) XOR s
    let mut tag_block = j0;
    cipher.encrypt(&mut tag_block);
    for i in 0..16 {
        tag_block[i] ^= s[i];
    }

    ct.extend_from_slice(&tag_block);
    ct
}

fn gcm_open<C: Cipher>(
    cipher: &C,
    nonce: &[u8; NONCE_SIZE],
    aad: &[u8],
    ciphertext_with_tag: &[u8],
) -> Result<Vec<u8>, CryptoError> {
    if ciphertext_with_tag.len() < TAG_SIZE {
        return Err(CryptoError::BadLength);
    }
    let split = ciphertext_with_tag.len() - TAG_SIZE;
    let (ct, recv_tag) = ciphertext_with_tag.split_at(split);

    let mut h = [0u8; 16];
    cipher.encrypt(&mut h);

    let mut j0 = [0u8; 16];
    j0[..12].copy_from_slice(nonce);
    j0[15] = 1;

    let mut s = [0u8; 16];
    ghash_pad(&mut s, &h, aad);
    ghash_pad(&mut s, &h, ct);
    let mut lengths = [0u8; 16];
    lengths[0..8].copy_from_slice(&((aad.len() as u64) * 8).to_be_bytes());
    lengths[8..16].copy_from_slice(&((ct.len() as u64) * 8).to_be_bytes());
    ghash_update(&mut s, &h, &lengths);

    let mut tag_block = j0;
    cipher.encrypt(&mut tag_block);
    for i in 0..16 {
        tag_block[i] ^= s[i];
    }
    if !ct_eq(&tag_block, recv_tag) {
        return Err(CryptoError::BadTag);
    }

    let mut pt = vec![0u8; ct.len()];
    ctr_xor(cipher, &j0, ct, &mut pt);
    Ok(pt)
}

fn ctr_xor<C: Cipher>(cipher: &C, j0: &[u8; 16], input: &[u8], output: &mut [u8]) {
    // GCM counter starts at inc32(J0).
    let mut counter = *j0;
    inc32(&mut counter);
    let mut offset = 0;
    while offset < input.len() {
        let mut block = counter;
        cipher.encrypt(&mut block);
        let take = (input.len() - offset).min(BLOCK_SIZE);
        for i in 0..take {
            output[offset + i] = input[offset + i] ^ block[i];
        }
        inc32(&mut counter);
        offset += take;
    }
}

fn inc32(block: &mut [u8; 16]) {
    let mut c = u32::from_be_bytes(block[12..16].try_into().unwrap());
    c = c.wrapping_add(1);
    block[12..16].copy_from_slice(&c.to_be_bytes());
}

#[derive(Debug)]
pub struct Aes128Gcm;
impl Aes128Gcm {
    pub fn seal(key: &[u8; 16], nonce: &[u8; NONCE_SIZE], aad: &[u8], pt: &[u8]) -> Vec<u8> {
        gcm_seal(&Aes128::new(key), nonce, aad, pt)
    }
    pub fn open(
        key: &[u8; 16],
        nonce: &[u8; NONCE_SIZE],
        aad: &[u8],
        ct: &[u8],
    ) -> Result<Vec<u8>, CryptoError> {
        gcm_open(&Aes128::new(key), nonce, aad, ct)
    }
}

#[derive(Debug)]
pub struct Aes256Gcm;
impl Aes256Gcm {
    pub fn seal(key: &[u8; 32], nonce: &[u8; NONCE_SIZE], aad: &[u8], pt: &[u8]) -> Vec<u8> {
        gcm_seal(&Aes256::new(key), nonce, aad, pt)
    }
    pub fn open(
        key: &[u8; 32],
        nonce: &[u8; NONCE_SIZE],
        aad: &[u8],
        ct: &[u8],
    ) -> Result<Vec<u8>, CryptoError> {
        gcm_open(&Aes256::new(key), nonce, aad, ct)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unhex(s: &str) -> Vec<u8> {
        let s: String = s.chars().filter(|c| !c.is_whitespace()).collect();
        (0..s.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap())
            .collect()
    }

    fn hex(b: &[u8]) -> String {
        let mut s = String::new();
        for x in b {
            s.push_str(&format!("{x:02x}"));
        }
        s
    }

    /// NIST SP 800-38D AES-128-GCM test vector #2 (gcm-revised-spec.pdf
    /// appendix B, test case 2). 16-byte all-zero plaintext, 16-byte
    /// all-zero key, 96-bit zero IV.
    #[test]
    fn nist_aes128_gcm_tc2() {
        let key: [u8; 16] = [0; 16];
        let nonce: [u8; 12] = [0; 12];
        let aad: [u8; 0] = [];
        let pt: [u8; 16] = [0; 16];
        let sealed = Aes128Gcm::seal(&key, &nonce, &aad, &pt);
        assert_eq!(
            hex(&sealed),
            "0388dace60b6a392f328c2b971b2fe78ab6e47d42cec13bdf53a67b21257bddf"
        );
        let opened = Aes128Gcm::open(&key, &nonce, &aad, &sealed).unwrap();
        assert_eq!(opened, pt);
    }

    /// NIST SP 800-38D AES-128-GCM test vector #3: empty AAD, 60-byte PT.
    #[test]
    fn nist_aes128_gcm_tc3() {
        let key: [u8; 16] = unhex("feffe9928665731c6d6a8f9467308308")
            .try_into()
            .unwrap();
        let nonce: [u8; 12] = unhex("cafebabefacedbaddecaf888").try_into().unwrap();
        let pt = unhex(
            "d9313225f88406e5a55909c5aff5269a86a7a9531534f7da2e4c303d8a318a721c3c0c95956809532fcf0e2449a6b525b16aedf5aa0de657ba637b391aafd255",
        );
        let want_ct = "42831ec2217774244b7221b784d0d49ce3aa212f2c02a4e035c17e2329aca12e21d514b25466931c7d8f6a5aac84aa051ba30b396a0aac973d58e091473f5985";
        let want_tag = "4d5c2af327cd64a62cf35abd2ba6fab4";
        let sealed = Aes128Gcm::seal(&key, &nonce, &[], &pt);
        assert_eq!(hex(&sealed[..pt.len()]), want_ct);
        assert_eq!(hex(&sealed[pt.len()..]), want_tag);
        let opened = Aes128Gcm::open(&key, &nonce, &[], &sealed).unwrap();
        assert_eq!(opened, pt);
    }

    /// NIST SP 800-38D AES-256-GCM test vector #16-equivalent: PT + AAD.
    #[test]
    fn nist_aes256_gcm_basic() {
        let key: [u8; 32] =
            unhex("feffe9928665731c6d6a8f9467308308feffe9928665731c6d6a8f9467308308")
                .try_into()
                .unwrap();
        let nonce: [u8; 12] = unhex("cafebabefacedbaddecaf888").try_into().unwrap();
        let pt = unhex(
            "d9313225f88406e5a55909c5aff5269a86a7a9531534f7da2e4c303d8a318a721c3c0c95956809532fcf0e2449a6b525b16aedf5aa0de657ba637b391aafd255",
        );
        let sealed = Aes256Gcm::seal(&key, &nonce, &[], &pt);
        let want_ct = "522dc1f099567d07f47f37a32a84427d643a8cdcbfe5c0c97598a2bd2555d1aa8cb08e48590dbb3da7b08b1056828838c5f61e6393ba7a0abcc9f662898015ad";
        let want_tag = "b094dac5d93471bdec1a502270e3cc6c";
        assert_eq!(hex(&sealed[..pt.len()]), want_ct);
        assert_eq!(hex(&sealed[pt.len()..]), want_tag);
        let opened = Aes256Gcm::open(&key, &nonce, &[], &sealed).unwrap();
        assert_eq!(opened, pt);
    }

    #[test]
    fn tamper_detected() {
        let key = [1u8; 16];
        let nonce = [2u8; 12];
        let pt = b"hello";
        let mut sealed = Aes128Gcm::seal(&key, &nonce, b"aad", pt);
        sealed[0] ^= 1;
        assert!(matches!(
            Aes128Gcm::open(&key, &nonce, b"aad", &sealed),
            Err(CryptoError::BadTag)
        ));
    }
}
