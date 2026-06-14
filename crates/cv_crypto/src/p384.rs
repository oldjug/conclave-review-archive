//! NIST P-384 (secp384r1) ECDSA verification.
//!
//! Same approach as `p521`: reuse the curve-agnostic Jacobian
//! arithmetic from `p256` (P-384 also has `a = -3`). 48-byte field
//! elements; SHA-384 hash.

use crate::CryptoError;
use crate::bigint::{BigUint, mul_mod, rem};
use crate::p256::{inv_mod, scalar_mul_affine};
use crate::sha384::Sha384;

fn p_prime() -> BigUint {
    BigUint::from_be_bytes(&[
        0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
        0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
        0xFF, 0xFE, 0xFF, 0xFF, 0xFF, 0xFF, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xFF,
        0xFF, 0xFF, 0xFF,
    ])
}

fn order_n() -> BigUint {
    BigUint::from_be_bytes(&[
        0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
        0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xC7, 0x63, 0x4D, 0x81, 0xF4, 0x37,
        0x2D, 0xDF, 0x58, 0x1A, 0x0D, 0xB2, 0x48, 0xB0, 0xA7, 0x7A, 0xEC, 0xEC, 0x19, 0x6A, 0xCC,
        0xC5, 0x29, 0x73,
    ])
}

fn base_point_xy() -> (BigUint, BigUint) {
    let gx = BigUint::from_be_bytes(&[
        0xAA, 0x87, 0xCA, 0x22, 0xBE, 0x8B, 0x05, 0x37, 0x8E, 0xB1, 0xC7, 0x1E, 0xF3, 0x20, 0xAD,
        0x74, 0x6E, 0x1D, 0x3B, 0x62, 0x8B, 0xA7, 0x9B, 0x98, 0x59, 0xF7, 0x41, 0xE0, 0x82, 0x54,
        0x2A, 0x38, 0x55, 0x02, 0xF2, 0x5D, 0xBF, 0x55, 0x29, 0x6C, 0x3A, 0x54, 0x5E, 0x38, 0x72,
        0x76, 0x0A, 0xB7,
    ]);
    let gy = BigUint::from_be_bytes(&[
        0x36, 0x17, 0xDE, 0x4A, 0x96, 0x26, 0x2C, 0x6F, 0x5D, 0x9E, 0x98, 0xBF, 0x92, 0x92, 0xDC,
        0x29, 0xF8, 0xF4, 0x1D, 0xBD, 0x28, 0x9A, 0x14, 0x7C, 0xE9, 0xDA, 0x31, 0x13, 0xB5, 0xF0,
        0xB8, 0xC0, 0x0A, 0x60, 0xB1, 0xCE, 0x1D, 0x7E, 0x81, 0x9D, 0x7A, 0x43, 0x1D, 0x7C, 0x90,
        0xEA, 0x0E, 0x5F,
    ]);
    (gx, gy)
}

fn hash_to_int(h: &[u8], n: &BigUint) -> BigUint {
    rem(&BigUint::from_be_bytes(h), n)
}

pub fn verify(qx: &[u8], qy: &[u8], msg: &[u8], r: &[u8], s: &[u8]) -> Result<(), CryptoError> {
    let n = order_n();
    let prime = p_prime();
    let r_int = BigUint::from_be_bytes(r);
    let s_int = BigUint::from_be_bytes(s);
    if r_int.is_zero()
        || s_int.is_zero()
        || r_int.cmp(&n) != core::cmp::Ordering::Less
        || s_int.cmp(&n) != core::cmp::Ordering::Less
    {
        return Err(CryptoError::BadTag);
    }
    let h = Sha384::oneshot(msg);
    let e = hash_to_int(&h, &n);

    let w = inv_mod(&s_int, &n);
    let u1 = mul_mod(&e, &w, &n);
    let u2 = mul_mod(&r_int, &w, &n);

    let (gx, gy) = base_point_xy();
    let qx_int = BigUint::from_be_bytes(qx);
    let qy_int = BigUint::from_be_bytes(qy);
    let p1 = scalar_mul_affine(&u1, &gx, &gy, &prime);
    let p2 = scalar_mul_affine(&u2, &qx_int, &qy_int, &prime);
    let sum = crate::p256::jac_add_pub(&p1, &p2, &prime);
    if sum.is_identity() {
        return Err(CryptoError::BadTag);
    }

    let z_inv = inv_mod(&sum.z, &prime);
    let z_inv_sq = mul_mod(&z_inv, &z_inv, &prime);
    let x_aff = mul_mod(&sum.x, &z_inv_sq, &prime);
    let r_check = rem(&x_aff, &n);
    if r_check == r_int {
        Ok(())
    } else {
        Err(CryptoError::BadTag)
    }
}

pub fn parse_der_signature(der: &[u8]) -> Result<(Vec<u8>, Vec<u8>), CryptoError> {
    crate::p256::parse_der_signature(der)
}

/// Derive the public key for a P-384 ECDH private scalar `d`. Output is
/// `0x04 || X || Y` — the uncompressed SEC1 point encoding used by TLS
/// 1.3 §4.2.8.2 / RFC 8422 §5.4.1 for secp384r1 (group 0x0018). 97 bytes.
pub fn public_key_uncompressed(d: &[u8; 48]) -> Result<[u8; 97], CryptoError> {
    let n = order_n();
    let d_int = BigUint::from_be_bytes(d);
    if d_int.is_zero() || d_int.cmp(&n) != core::cmp::Ordering::Less {
        return Err(CryptoError::BadLength);
    }
    let (gx, gy) = base_point_xy();
    let prime = p_prime();
    let pt = scalar_mul_affine(&d_int, &gx, &gy, &prime);
    if pt.is_identity() {
        return Err(CryptoError::BadLength);
    }
    let z_inv = inv_mod(&pt.z, &prime);
    let z_inv_sq = mul_mod(&z_inv, &z_inv, &prime);
    let z_inv_cu = mul_mod(&z_inv_sq, &z_inv, &prime);
    let x_aff = mul_mod(&pt.x, &z_inv_sq, &prime);
    let y_aff = mul_mod(&pt.y, &z_inv_cu, &prime);
    let mut out = [0u8; 97];
    out[0] = 0x04;
    let xb = x_aff.to_be_bytes(48);
    let yb = y_aff.to_be_bytes(48);
    out[1..49].copy_from_slice(&xb);
    out[49..97].copy_from_slice(&yb);
    Ok(out)
}

/// P-384 ECDH per SEC1 §3.3.1 / TLS 1.3 §7.4.2: multiply the peer's
/// uncompressed public point (`0x04 || X || Y`, 97 bytes) by our
/// private scalar `d` and return the X coordinate of the result.
/// Returns 48 big-endian bytes — the raw shared secret used as IKM.
pub fn ecdh_shared(d: &[u8; 48], peer_uncompressed: &[u8]) -> Result<[u8; 48], CryptoError> {
    if peer_uncompressed.len() != 97 || peer_uncompressed[0] != 0x04 {
        return Err(CryptoError::BadLength);
    }
    let n = order_n();
    let d_int = BigUint::from_be_bytes(d);
    if d_int.is_zero() || d_int.cmp(&n) != core::cmp::Ordering::Less {
        return Err(CryptoError::BadLength);
    }
    let qx = BigUint::from_be_bytes(&peer_uncompressed[1..49]);
    let qy = BigUint::from_be_bytes(&peer_uncompressed[49..97]);
    let prime = p_prime();
    let pt = scalar_mul_affine(&d_int, &qx, &qy, &prime);
    if pt.is_identity() {
        return Err(CryptoError::BadLength);
    }
    let z_inv = inv_mod(&pt.z, &prime);
    let z_inv_sq = mul_mod(&z_inv, &z_inv, &prime);
    let x_aff = mul_mod(&pt.x, &z_inv_sq, &prime);
    let xb = x_aff.to_be_bytes(48);
    let mut out = [0u8; 48];
    out.copy_from_slice(&xb);
    Ok(out)
}

#[cfg(test)]
mod ecdh_tests {
    use super::*;

    /// RFC 5903 §3.1 — secp384r1 ECDH known-answer vector. i and r are
    /// the two private scalars; the test runs i*G to get its public
    /// point, then has r combine its private scalar with i's public
    /// point and asserts the X coordinate matches the RFC's `g^ir`.
    #[test]
    fn rfc5903_secp384r1() {
        // i's private key
        let i_priv_hex = "099F3C7034D4A2C699884D73A375A67F7624EF7C6B3C0F160647B67414DCE655E35B538041E649EE3FAEF896783AB194";
        // r's private key
        let r_priv_hex = "41CB0779B4BDB85D47846725FBEC3C9430FAB46CC8DC5060855CC9BDA0AA2942E0308312916B8ED2960E4BD55A7448FC";
        // Expected ECDH X coordinate (g^ir)
        let xs_hex = "11187331C279962D93D604243FD592CB9D0A926F422E47187521287E7156C5C4D603135569B9E9D09CF5D4A270F59746";
        let i_priv = hex_to_arr48(i_priv_hex);
        let r_priv = hex_to_arr48(r_priv_hex);
        let i_pub = public_key_uncompressed(&i_priv).unwrap();
        let shared = ecdh_shared(&r_priv, &i_pub).unwrap();
        assert_eq!(hex_arr(&shared), xs_hex.to_ascii_lowercase());
    }

    fn hex_to_arr48(h: &str) -> [u8; 48] {
        let v = (0..h.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&h[i..i + 2], 16).unwrap())
            .collect::<Vec<u8>>();
        let mut a = [0u8; 48];
        a.copy_from_slice(&v);
        a
    }
    fn hex_arr(b: &[u8]) -> String {
        let mut s = String::with_capacity(b.len() * 2);
        for &x in b {
            s.push_str(&format!("{:02x}", x));
        }
        s
    }
}
