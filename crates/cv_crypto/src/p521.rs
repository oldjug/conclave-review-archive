//! NIST P-521 (secp521r1) ECDSA verification — RFC 6979 + FIPS 186-4.
//!
//! Reuses the curve-agnostic Jacobian arithmetic from `p256` (both
//! curves have `a = -3` so the doubling formula in `jac_double`
//! applies unchanged). Only the curve parameters — prime modulus,
//! group order, generator point — and the hash algorithm differ.
//!
//! P-521 elements are 521 bits, so big-endian byte buffers are 66
//! bytes (with the top byte holding only one bit of value). Real
//! certificates and TLS wire formats may zero-pad shorter integers,
//! so the verify entry point accepts arbitrary-length `r`/`s`/`q*`
//! slices and normalises into `BigUint` via the existing big-endian
//! parser, which strips leading zeros.

use crate::CryptoError;
use crate::bigint::{BigUint, add_mod, mul_mod, rem};
use crate::p256::{inv_mod, scalar_mul_affine};
use crate::sha512::Sha512;

/// P-521 prime: `p = 2^521 − 1`. As 66 big-endian bytes the top byte
/// is `0x01` (bit 521 set) and the rest are `0xFF`.
fn p_prime() -> BigUint {
    let mut bytes = [0xFFu8; 66];
    bytes[0] = 0x01;
    BigUint::from_be_bytes(&bytes)
}

/// P-521 group order `n` per FIPS 186-4 §D.1.2.5.
fn order_n() -> BigUint {
    BigUint::from_be_bytes(&[
        0x01, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
        0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
        0xFF, 0xFF, 0xFA, 0x51, 0x86, 0x87, 0x83, 0xBF, 0x2F, 0x96, 0x6B, 0x7F, 0xCC, 0x01, 0x48,
        0xF7, 0x09, 0xA5, 0xD0, 0x3B, 0xB5, 0xC9, 0xB8, 0x89, 0x9C, 0x47, 0xAE, 0xBB, 0x6F, 0xB7,
        0x1E, 0x91, 0x38, 0x64, 0x09,
    ])
}

/// Base point `G = (Gx, Gy)` per FIPS 186-4 §D.1.2.5.
fn base_point_xy() -> (BigUint, BigUint) {
    let gx = BigUint::from_be_bytes(&[
        0x00, 0xC6, 0x85, 0x8E, 0x06, 0xB7, 0x04, 0x04, 0xE9, 0xCD, 0x9E, 0x3E, 0xCB, 0x66, 0x23,
        0x95, 0xB4, 0x42, 0x9C, 0x64, 0x81, 0x39, 0x05, 0x3F, 0xB5, 0x21, 0xF8, 0x28, 0xAF, 0x60,
        0x6B, 0x4D, 0x3D, 0xBA, 0xA1, 0x4B, 0x5E, 0x77, 0xEF, 0xE7, 0x59, 0x28, 0xFE, 0x1D, 0xC1,
        0x27, 0xA2, 0xFF, 0xA8, 0xDE, 0x33, 0x48, 0xB3, 0xC1, 0x85, 0x6A, 0x42, 0x9B, 0xF9, 0x7E,
        0x7E, 0x31, 0xC2, 0xE5, 0xBD, 0x66,
    ]);
    let gy = BigUint::from_be_bytes(&[
        0x01, 0x18, 0x39, 0x29, 0x6A, 0x78, 0x9A, 0x3B, 0xC0, 0x04, 0x5C, 0x8A, 0x5F, 0xB4, 0x2C,
        0x7D, 0x1B, 0xD9, 0x98, 0xF5, 0x44, 0x49, 0x57, 0x9B, 0x44, 0x68, 0x17, 0xAF, 0xBD, 0x17,
        0x27, 0x3E, 0x66, 0x2C, 0x97, 0xEE, 0x72, 0x99, 0x5E, 0xF4, 0x26, 0x40, 0xC5, 0x50, 0xB9,
        0x01, 0x3F, 0xAD, 0x07, 0x61, 0x35, 0x3C, 0x70, 0x86, 0xA2, 0x72, 0xC2, 0x40, 0x88, 0xBE,
        0x94, 0x76, 0x9F, 0xD1, 0x66, 0x50,
    ]);
    (gx, gy)
}

/// Truncate / reduce a SHA-512 digest into the scalar field per
/// FIPS 186-4 §6.4.2 step 5: take the leftmost `qlen` bits of the
/// hash. For P-521 `qlen = 521` and SHA-512 produces 512 bits, so
/// the hash is *shorter* than `qlen` and is just converted to an
/// integer in [0, 2^512 − 1]. No bit-trimming needed.
fn hash_to_int(h: &[u8], n: &BigUint) -> BigUint {
    let i = BigUint::from_be_bytes(h);
    rem(&i, n)
}

/// Verify an ECDSA-P521-SHA512 signature.
///
/// `qx` / `qy` are the public-key affine coordinates (big-endian,
/// up to 66 bytes each). `r` / `s` are the signature components.
/// `msg` is the message that was hashed (caller passes the raw
/// signed bytes — we compute SHA-512 ourselves).
pub fn verify(qx: &[u8], qy: &[u8], msg: &[u8], r: &[u8], s: &[u8]) -> Result<(), CryptoError> {
    let n = order_n();
    let prime = p_prime();
    let r_int = BigUint::from_be_bytes(r);
    let s_int = BigUint::from_be_bytes(s);
    // §6.4.2 step 1: r, s in [1, n-1].
    if r_int.is_zero()
        || s_int.is_zero()
        || r_int.cmp(&n) != core::cmp::Ordering::Less
        || s_int.cmp(&n) != core::cmp::Ordering::Less
    {
        return Err(CryptoError::BadTag);
    }
    // §6.4.2 step 4: e = leftmost qlen bits of H(m). SHA-512 → 512
    // bits < qlen=521, so direct convert.
    let h = Sha512::oneshot(msg);
    let e = hash_to_int(&h, &n);

    // §6.4.2 step 6: w = s⁻¹ mod n; u1 = ew mod n; u2 = rw mod n.
    let w = inv_mod(&s_int, &n);
    let u1 = mul_mod(&e, &w, &n);
    let u2 = mul_mod(&r_int, &w, &n);

    // §6.4.2 step 7: (x1, y1) = u1·G + u2·Q.
    let (gx, gy) = base_point_xy();
    let qx_int = BigUint::from_be_bytes(qx);
    let qy_int = BigUint::from_be_bytes(qy);
    let p1 = scalar_mul_affine(&u1, &gx, &gy, &prime);
    let p2 = scalar_mul_affine(&u2, &qx_int, &qy_int, &prime);
    let sum = crate::p256::jac_add_pub(&p1, &p2, &prime);
    if sum.is_identity() {
        return Err(CryptoError::BadTag);
    }

    // Convert sum to affine: x_aff = X / Z² mod p.
    let z_inv = inv_mod(&sum.z, &prime);
    let z_inv_sq = mul_mod(&z_inv, &z_inv, &prime);
    let x_aff = mul_mod(&sum.x, &z_inv_sq, &prime);
    // §6.4.2 step 8: r ≡ x_aff (mod n).
    let r_check = rem(&x_aff, &n);
    if r_check == r_int {
        Ok(())
    } else {
        Err(CryptoError::BadTag)
    }
}

/// Parse an ASN.1 DER ECDSA-Sig-Value into raw big-endian `r`/`s`.
/// P-521 signatures DER-encode r/s as integers — same wire format
/// as P-256, just longer (≤66 bytes each). Reuses the P-256 parser.
pub fn parse_der_signature(der: &[u8]) -> Result<(Vec<u8>, Vec<u8>), CryptoError> {
    crate::p256::parse_der_signature(der)
}
