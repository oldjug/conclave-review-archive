//! TLS 1.3 record layer encryption per RFC 8446 §5.
//!
//! After the handshake reaches the point where keys are available, every
//! record on the wire is a `TLSCiphertext`:
//!
//! ```text
//! struct {
//!     ContentType opaque_type = application_data;   // 0x17 always
//!     ProtocolVersion legacy_record_version = 0x0303;
//!     uint16 length;                                 // of encrypted_record
//!     opaque encrypted_record[TLSCiphertext.length]; // AEAD(plaintext, AAD)
//! }
//! ```
//!
//! `plaintext` = `inner_plaintext || ContentType_byte || zero-padding`.
//! `AAD` = `opaque_type || legacy_record_version || length`.
//! `nonce` = sequence number padded to 12 bytes, XOR'd with traffic IV.

use core::fmt;
use cv_crypto::CryptoError;
use cv_crypto::aes_gcm::{Aes128Gcm, Aes256Gcm};
use cv_crypto::chacha20poly1305::ChaCha20Poly1305;

use crate::tls::messages::ContentType;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Aead {
    Aes128Gcm,
    Aes256Gcm,
    ChaCha20Poly1305,
}

impl Aead {
    pub fn key_len(self) -> usize {
        match self {
            Self::Aes128Gcm => 16,
            Self::Aes256Gcm => 32,
            Self::ChaCha20Poly1305 => 32,
        }
    }

    pub const fn iv_len(self) -> usize {
        12
    }

    pub const fn tag_len(self) -> usize {
        16
    }
}

#[derive(Debug)]
pub enum RecordError {
    Crypto(CryptoError),
    Short,
    SeqOverflow,
    UnknownInnerType(u8),
}

impl fmt::Display for RecordError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Crypto(e) => write!(f, "aead: {e:?}"),
            Self::Short => f.write_str("short record"),
            Self::SeqOverflow => f.write_str("sequence number overflow"),
            Self::UnknownInnerType(t) => write!(f, "unknown inner type {t:#x}"),
        }
    }
}

impl std::error::Error for RecordError {}

impl From<CryptoError> for RecordError {
    fn from(e: CryptoError) -> Self {
        Self::Crypto(e)
    }
}

/// One direction of the record layer.
pub struct AeadKey {
    pub aead: Aead,
    pub key: Vec<u8>,
    pub iv: [u8; 12],
    pub seq: u64,
}

impl fmt::Debug for AeadKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AeadKey")
            .field("aead", &self.aead)
            .field("seq", &self.seq)
            .finish_non_exhaustive()
    }
}

impl AeadKey {
    pub fn new(aead: Aead, key: Vec<u8>, iv_bytes: &[u8]) -> Self {
        assert_eq!(key.len(), aead.key_len(), "key length mismatch");
        assert_eq!(iv_bytes.len(), 12, "iv must be 12 bytes");
        let mut iv = [0u8; 12];
        iv.copy_from_slice(iv_bytes);
        Self {
            aead,
            key,
            iv,
            seq: 0,
        }
    }

    fn nonce(&self) -> [u8; 12] {
        // 64-bit seq number left-padded with zeros to 96 bits, XOR with iv.
        let mut n = self.iv;
        let seq_be = self.seq.to_be_bytes();
        for i in 0..8 {
            n[4 + i] ^= seq_be[i];
        }
        n
    }

    /// Encrypt a single record. `inner_content` is the plaintext payload
    /// (handshake bytes, app data bytes, etc.); `inner_type` is the
    /// `ContentType` that goes at the end of the inner plaintext before
    /// padding. Output is the full on-the-wire `TLSCiphertext` record
    /// including the 5-byte header.
    pub fn seal_record(
        &mut self,
        inner_type: ContentType,
        inner_content: &[u8],
    ) -> Result<Vec<u8>, RecordError> {
        // Build inner plaintext: content || type_byte (no padding).
        let mut inner = Vec::with_capacity(inner_content.len() + 1);
        inner.extend_from_slice(inner_content);
        inner.push(inner_type as u8);

        // AAD: opaque_type=0x17 || 0x0303 || length(u16 BE) of ciphertext.
        let ct_len = inner.len() + self.aead.tag_len();
        if ct_len > 0x4000 + 256 {
            // RFC 8446 §5.2 max TLSCiphertext.length.
            return Err(RecordError::Short);
        }
        let mut aad = [0u8; 5];
        aad[0] = ContentType::ApplicationData as u8;
        aad[1] = 0x03;
        aad[2] = 0x03;
        aad[3] = ((ct_len >> 8) & 0xFF) as u8;
        aad[4] = (ct_len & 0xFF) as u8;

        let nonce = self.nonce();
        let sealed = match self.aead {
            Aead::Aes128Gcm => {
                let key: [u8; 16] = self.key.as_slice().try_into().unwrap();
                Aes128Gcm::seal(&key, &nonce, &aad, &inner)
            }
            Aead::Aes256Gcm => {
                let key: [u8; 32] = self.key.as_slice().try_into().unwrap();
                Aes256Gcm::seal(&key, &nonce, &aad, &inner)
            }
            Aead::ChaCha20Poly1305 => {
                let key: [u8; 32] = self.key.as_slice().try_into().unwrap();
                ChaCha20Poly1305::seal(&key, &nonce, &aad, &inner)
            }
        };

        self.seq = self.seq.checked_add(1).ok_or(RecordError::SeqOverflow)?;

        let mut record = Vec::with_capacity(5 + sealed.len());
        record.extend_from_slice(&aad);
        record.extend_from_slice(&sealed);
        Ok(record)
    }

    /// Decrypt one TLSCiphertext record fragment. `header` is the 5-byte
    /// record header (used as AAD). Returns `(inner_type, inner_content)`.
    pub fn open_record(
        &mut self,
        header: &[u8; 5],
        encrypted_fragment: &[u8],
    ) -> Result<(ContentType, Vec<u8>), RecordError> {
        let nonce = self.nonce();
        let mut inner = match self.aead {
            Aead::Aes128Gcm => {
                let key: [u8; 16] = self.key.as_slice().try_into().unwrap();
                Aes128Gcm::open(&key, &nonce, header, encrypted_fragment)?
            }
            Aead::Aes256Gcm => {
                let key: [u8; 32] = self.key.as_slice().try_into().unwrap();
                Aes256Gcm::open(&key, &nonce, header, encrypted_fragment)?
            }
            Aead::ChaCha20Poly1305 => {
                let key: [u8; 32] = self.key.as_slice().try_into().unwrap();
                ChaCha20Poly1305::open(&key, &nonce, header, encrypted_fragment)?
            }
        };

        self.seq = self.seq.checked_add(1).ok_or(RecordError::SeqOverflow)?;

        // Strip trailing zero padding, then read the ContentType byte.
        while inner.last() == Some(&0) {
            inner.pop();
        }
        let inner_type = inner.pop().ok_or(RecordError::Short)?;
        let typed = match inner_type {
            20 => ContentType::ChangeCipherSpec,
            21 => ContentType::Alert,
            22 => ContentType::Handshake,
            23 => ContentType::ApplicationData,
            other => return Err(RecordError::UnknownInnerType(other)),
        };
        Ok((typed, inner))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nonce_xor() {
        let mut k = AeadKey::new(Aead::Aes128Gcm, vec![0u8; 16], &[0u8; 12]);
        // First nonce equals the IV (XOR with seq=0 is no-op).
        assert_eq!(k.nonce(), [0u8; 12]);
        k.seq = 1;
        let mut want = [0u8; 12];
        want[11] = 1;
        assert_eq!(k.nonce(), want);
        k.seq = 0x1234_5678_9abc_def0;
        let want = [0, 0, 0, 0, 0x12, 0x34, 0x56, 0x78, 0x9a, 0xbc, 0xde, 0xf0];
        assert_eq!(k.nonce(), want);
    }

    #[test]
    fn seal_open_roundtrip_chacha() {
        let mut send = AeadKey::new(Aead::ChaCha20Poly1305, vec![7u8; 32], &[3u8; 12]);
        let mut recv = AeadKey::new(Aead::ChaCha20Poly1305, vec![7u8; 32], &[3u8; 12]);
        let record = send
            .seal_record(ContentType::Handshake, b"hello tls")
            .unwrap();
        let header: [u8; 5] = record[..5].try_into().unwrap();
        let fragment = &record[5..];
        let (typed, msg) = recv.open_record(&header, fragment).unwrap();
        assert_eq!(typed, ContentType::Handshake);
        assert_eq!(&msg, b"hello tls");
    }

    #[test]
    fn seal_open_roundtrip_aes128() {
        let mut send = AeadKey::new(Aead::Aes128Gcm, vec![1u8; 16], &[2u8; 12]);
        let mut recv = AeadKey::new(Aead::Aes128Gcm, vec![1u8; 16], &[2u8; 12]);
        let r1 = send
            .seal_record(ContentType::ApplicationData, b"first")
            .unwrap();
        let r2 = send
            .seal_record(ContentType::ApplicationData, b"second")
            .unwrap();
        let h1: [u8; 5] = r1[..5].try_into().unwrap();
        let (_, m1) = recv.open_record(&h1, &r1[5..]).unwrap();
        let h2: [u8; 5] = r2[..5].try_into().unwrap();
        let (_, m2) = recv.open_record(&h2, &r2[5..]).unwrap();
        assert_eq!(&m1, b"first");
        assert_eq!(&m2, b"second");
    }

    #[test]
    fn tamper_rejected() {
        let mut send = AeadKey::new(Aead::Aes128Gcm, vec![1u8; 16], &[2u8; 12]);
        let mut recv = AeadKey::new(Aead::Aes128Gcm, vec![1u8; 16], &[2u8; 12]);
        let mut rec = send
            .seal_record(ContentType::ApplicationData, b"data")
            .unwrap();
        let last = rec.len() - 1;
        rec[last] ^= 1;
        let h: [u8; 5] = rec[..5].try_into().unwrap();
        let result = recv.open_record(&h, &rec[5..]);
        assert!(result.is_err());
    }
}
