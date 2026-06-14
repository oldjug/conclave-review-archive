//! ChaCha20-Poly1305 AEAD per RFC 8439 §2.8.
//!
//! TLS 1.3 cipher suite `TLS_CHACHA20_POLY1305_SHA256` rides on this.

use crate::CryptoError;
use crate::chacha20::{ChaCha20, KEY_SIZE, NONCE_SIZE};
use crate::poly1305::{Poly1305, TAG_SIZE};
use crate::subtle::ct_eq;

#[derive(Debug)]
pub struct ChaCha20Poly1305;

impl ChaCha20Poly1305 {
    /// Encrypt-and-tag. Returns ciphertext (same length as plaintext) || tag.
    /// `aad` is associated data covered by the MAC but not encrypted.
    pub fn seal(
        key: &[u8; KEY_SIZE],
        nonce: &[u8; NONCE_SIZE],
        aad: &[u8],
        plaintext: &[u8],
    ) -> Vec<u8> {
        // Per RFC 8439 §2.6: derive Poly1305 one-time key from ChaCha20
        // block 0 (first 32 bytes).
        let block0 = ChaCha20::new(key, nonce, 0).block();
        let mut poly_key = [0u8; 32];
        poly_key.copy_from_slice(&block0[..32]);

        // Encrypt with counter starting at 1.
        let mut ct = plaintext.to_vec();
        let mut stream = ChaCha20::new(key, nonce, 1);
        stream.xor(&mut ct);

        // Authenticate AAD || pad16 || CT || pad16 || aad_len_u64_le || ct_len_u64_le.
        let mut mac = Poly1305::new(&poly_key);
        mac.update(aad);
        mac.update(&zero_pad16(aad.len()));
        mac.update(&ct);
        mac.update(&zero_pad16(ct.len()));
        let mut lengths = [0u8; 16];
        lengths[..8].copy_from_slice(&(aad.len() as u64).to_le_bytes());
        lengths[8..].copy_from_slice(&(ct.len() as u64).to_le_bytes());
        mac.update(&lengths);
        let tag = mac.finalize();

        ct.extend_from_slice(&tag);
        ct
    }

    /// Verify-and-decrypt. Input is ciphertext || tag. Returns plaintext on
    /// success, `CryptoError::BadTag` on auth failure.
    pub fn open(
        key: &[u8; KEY_SIZE],
        nonce: &[u8; NONCE_SIZE],
        aad: &[u8],
        ciphertext_with_tag: &[u8],
    ) -> Result<Vec<u8>, CryptoError> {
        if ciphertext_with_tag.len() < TAG_SIZE {
            return Err(CryptoError::BadLength);
        }
        let split = ciphertext_with_tag.len() - TAG_SIZE;
        let (ct, recv_tag) = ciphertext_with_tag.split_at(split);

        let block0 = ChaCha20::new(key, nonce, 0).block();
        let mut poly_key = [0u8; 32];
        poly_key.copy_from_slice(&block0[..32]);

        let mut mac = Poly1305::new(&poly_key);
        mac.update(aad);
        mac.update(&zero_pad16(aad.len()));
        mac.update(ct);
        mac.update(&zero_pad16(ct.len()));
        let mut lengths = [0u8; 16];
        lengths[..8].copy_from_slice(&(aad.len() as u64).to_le_bytes());
        lengths[8..].copy_from_slice(&(ct.len() as u64).to_le_bytes());
        mac.update(&lengths);
        let tag = mac.finalize();

        if !ct_eq(&tag, recv_tag) {
            return Err(CryptoError::BadTag);
        }

        let mut pt = ct.to_vec();
        let mut stream = ChaCha20::new(key, nonce, 1);
        stream.xor(&mut pt);
        Ok(pt)
    }
}

fn zero_pad16(len: usize) -> Vec<u8> {
    let rem = len % 16;
    if rem == 0 {
        Vec::new()
    } else {
        vec![0u8; 16 - rem]
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

    /// RFC 8439 §2.8.2 test vector.
    #[test]
    fn rfc8439_seal() {
        let key: [u8; 32] = [
            0x80, 0x81, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89, 0x8a, 0x8b, 0x8c, 0x8d,
            0x8e, 0x8f, 0x90, 0x91, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9a, 0x9b,
            0x9c, 0x9d, 0x9e, 0x9f,
        ];
        let nonce: [u8; 12] = [
            0x07, 0x00, 0x00, 0x00, 0x40, 0x41, 0x42, 0x43, 0x44, 0x45, 0x46, 0x47,
        ];
        let aad: [u8; 12] = [
            0x50, 0x51, 0x52, 0x53, 0xc0, 0xc1, 0xc2, 0xc3, 0xc4, 0xc5, 0xc6, 0xc7,
        ];
        let plaintext =
            b"Ladies and Gentlemen of the class of '99: If I could offer you only one tip for the future, sunscreen would be it.";

        let sealed = ChaCha20Poly1305::seal(&key, &nonce, &aad, plaintext);
        // Tag from the RFC.
        let want_tag = "1ae10b594f09e26a7e902ecbd0600691";
        assert_eq!(hex(&sealed[sealed.len() - 16..]), want_tag);

        let recovered = ChaCha20Poly1305::open(&key, &nonce, &aad, &sealed).expect("auth");
        assert_eq!(&recovered, plaintext);
    }

    #[test]
    fn tampered_ciphertext_rejected() {
        let key = [0u8; 32];
        let nonce = [0u8; 12];
        let pt = b"hello, world";
        let mut sealed = ChaCha20Poly1305::seal(&key, &nonce, b"", pt);
        sealed[0] ^= 1;
        assert!(matches!(
            ChaCha20Poly1305::open(&key, &nonce, b"", &sealed),
            Err(CryptoError::BadTag)
        ));
    }
}
