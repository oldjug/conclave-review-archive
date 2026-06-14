//! TLS 1.3 client per RFC 8446.
//!
//! Layered:
//! - `key_schedule` — HKDF-Expand-Label, derive_secret, traffic keys.
//! - `messages` — handshake and record wire formats.
//! - `record` — record layer encryption / decryption.
//! - `client` — handshake state machine + `TlsStream` wrapper around
//!   `cv_net::Socket`.
//!
//! Cipher suites: `TLS_AES_128_GCM_SHA256`, `TLS_CHACHA20_POLY1305_SHA256`,
//! `TLS_AES_256_GCM_SHA384`. Group: `x25519` only for V1.
//! Sig algs: `rsa_pkcs1_sha256` (cert chain), `rsa_pss_rsae_sha256`
//! (CertificateVerify) — PSS lands once we have a verifier.

pub mod chain_validate;
pub mod client;
pub mod key_schedule;
pub mod messages;
pub mod record;
pub mod resumption;
pub(crate) mod sys_rand;
pub mod tls12;

pub use client::{TlsError, TlsStream};
