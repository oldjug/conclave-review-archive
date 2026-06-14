//! `cv_crypto` — our from-scratch crypto stack.
//!
//! Scope today: symmetric primitives needed for TLS 1.3 AEAD record
//! protection — SHA-256, HMAC-SHA256, HKDF-SHA256, ChaCha20, Poly1305,
//! ChaCha20-Poly1305 AEAD. Each module has the NIST/RFC test vector(s)
//! wired into `#[test]`.
//!
//! Next pass adds: SHA-384, AES-128/256-GCM (GHASH), X25519, P-256
//! ECDSA, RSA verification, X.509 chain validation.
//!
//! References:
//! - SHA-256: FIPS 180-4
//! - HMAC: FIPS 198-1
//! - HKDF: RFC 5869
//! - ChaCha20: RFC 8439
//! - Poly1305: RFC 8439
//! - ChaCha20-Poly1305 AEAD: RFC 8439

#![allow(clippy::many_single_char_names)]
#![allow(clippy::unreadable_literal)]
#![allow(
    dead_code,
    missing_debug_implementations,
    unreachable_pub,
    unused_imports
)]

pub mod aes;
pub mod aes_gcm;
pub mod asn1;
pub mod bigint;
pub mod chacha20;
pub mod chacha20poly1305;
pub mod checksums;
pub mod crl;
pub mod ct;
pub mod ed25519;
pub mod hkdf;
pub mod hmac;
pub mod md5;
pub mod ocsp;
pub mod p256;
pub mod p384;
pub mod p521;
pub mod pbkdf2;
pub mod poly1305;
pub mod ripemd160;
pub mod rsa;
pub mod scrypt;
pub mod sha1;
pub mod sha256;
pub mod sha3;
pub mod sha384;
pub mod sha512;
pub mod subtle;
pub mod subtle_jwk;
pub mod x25519;
pub mod x509;

pub use chacha20poly1305::ChaCha20Poly1305;
pub use sha256::Sha256;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CryptoError {
    BadLength,
    BadTag,
}
