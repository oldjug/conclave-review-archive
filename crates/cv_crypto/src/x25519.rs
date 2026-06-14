//! X25519 Diffie-Hellman per RFC 7748.
//!
//! Curve25519 is `y² = x³ + 486662 x² + x` over GF(2²⁵⁵ - 19). X25519
//! operates on x-coordinates only, via the Montgomery ladder. Field
//! elements are kept as 5×51-bit limbs in `u64`.
//!
//! Constant-time throughout: ladder steps perform a conditional swap
//! by mask, never by branch.

pub const SCALAR_SIZE: usize = 32;
pub const POINT_SIZE: usize = 32;

const P_LIMBS: [u64; 5] = [
    // 2^255 - 19, split into 5 × 51-bit limbs.
    0x7_ffff_ffff_ffed,
    0x7_ffff_ffff_ffff,
    0x7_ffff_ffff_ffff,
    0x7_ffff_ffff_ffff,
    0x7_ffff_ffff_ffff,
];

type Fe = [u64; 5];

const ZERO: Fe = [0; 5];
const ONE: Fe = [1, 0, 0, 0, 0];

fn fe_from_bytes(b: &[u8; 32]) -> Fe {
    let mut t = [0u64; 4];
    for i in 0..4 {
        t[i] = u64::from_le_bytes(b[i * 8..i * 8 + 8].try_into().unwrap());
    }
    // Mask off the top bit per RFC 7748 §5: u[31] &= 0x7F.
    t[3] &= !(1u64 << 63);

    let mask = (1u64 << 51) - 1;
    [
        t[0] & mask,
        ((t[0] >> 51) | (t[1] << 13)) & mask,
        ((t[1] >> 38) | (t[2] << 26)) & mask,
        ((t[2] >> 25) | (t[3] << 39)) & mask,
        (t[3] >> 12) & mask,
    ]
}

fn fe_to_bytes(h: &Fe) -> [u8; 32] {
    // Reduce mod p one more time.
    let mut h = *h;
    fe_reduce(&mut h);
    let t0 = h[0] | (h[1] << 51);
    let t1 = (h[1] >> 13) | (h[2] << 38);
    let t2 = (h[2] >> 26) | (h[3] << 25);
    let t3 = (h[3] >> 39) | (h[4] << 12);

    let mut out = [0u8; 32];
    out[0..8].copy_from_slice(&t0.to_le_bytes());
    out[8..16].copy_from_slice(&t1.to_le_bytes());
    out[16..24].copy_from_slice(&t2.to_le_bytes());
    out[24..32].copy_from_slice(&t3.to_le_bytes());
    out
}

fn fe_carry(h: &mut Fe) {
    let mask = (1u64 << 51) - 1;
    let c0 = h[0] >> 51;
    h[0] &= mask;
    h[1] += c0;
    let c1 = h[1] >> 51;
    h[1] &= mask;
    h[2] += c1;
    let c2 = h[2] >> 51;
    h[2] &= mask;
    h[3] += c2;
    let c3 = h[3] >> 51;
    h[3] &= mask;
    h[4] += c3;
    let c4 = h[4] >> 51;
    h[4] &= mask;
    h[0] += c4 * 19;
}

fn fe_reduce(h: &mut Fe) {
    fe_carry(h);
    // Conditionally subtract p.
    let mask = (1u64 << 51) - 1;
    // Compute h - p; if no borrow we use the difference.
    let mut t = *h;
    // Add 19 (since p = 2^255 - 19, h + 19 mod 2^255 is h - p when h >= p).
    t[0] += 19;
    fe_carry(&mut t);
    // After carry, t[4] high bit (bit 51) indicates whether the +19 caused
    // a reduction (i.e. h >= p - 19 → h >= p if bit 255 is set).
    let take = (t[4] >> 51) & 1;
    t[4] &= mask;
    if take == 1 {
        *h = t;
    }
    // One more carry to make sure all limbs are in [0, 2^51).
    fe_carry(h);
}

fn fe_add(a: &Fe, b: &Fe) -> Fe {
    [
        a[0] + b[0],
        a[1] + b[1],
        a[2] + b[2],
        a[3] + b[3],
        a[4] + b[4],
    ]
}

fn fe_sub(a: &Fe, b: &Fe) -> Fe {
    // Add 2p so we never underflow.
    // 2p_limbs = 2 * p, with low limb 2 * (2^51 - 19) + 2^51 carry etc.
    // Easier: compute a + 2*p - b.
    const TWO_P: [u64; 5] = [
        2 * (P_LIMBS[0]),
        2 * P_LIMBS[1],
        2 * P_LIMBS[2],
        2 * P_LIMBS[3],
        2 * P_LIMBS[4],
    ];
    let mut r = [
        a[0] + TWO_P[0] - b[0],
        a[1] + TWO_P[1] - b[1],
        a[2] + TWO_P[2] - b[2],
        a[3] + TWO_P[3] - b[3],
        a[4] + TWO_P[4] - b[4],
    ];
    fe_carry(&mut r);
    r
}

fn fe_mul(a: &Fe, b: &Fe) -> Fe {
    let a0 = a[0] as u128;
    let a1 = a[1] as u128;
    let a2 = a[2] as u128;
    let a3 = a[3] as u128;
    let a4 = a[4] as u128;
    let b0 = b[0] as u128;
    let b1 = b[1] as u128;
    let b2 = b[2] as u128;
    let b3 = b[3] as u128;
    let b4 = b[4] as u128;

    // 19 * b_i (for the wrap-around).
    let b1_19 = 19 * b1;
    let b2_19 = 19 * b2;
    let b3_19 = 19 * b3;
    let b4_19 = 19 * b4;

    let r0 = a0 * b0 + a1 * b4_19 + a2 * b3_19 + a3 * b2_19 + a4 * b1_19;
    let r1 = a0 * b1 + a1 * b0 + a2 * b4_19 + a3 * b3_19 + a4 * b2_19;
    let r2 = a0 * b2 + a1 * b1 + a2 * b0 + a3 * b4_19 + a4 * b3_19;
    let r3 = a0 * b3 + a1 * b2 + a2 * b1 + a3 * b0 + a4 * b4_19;
    let r4 = a0 * b4 + a1 * b3 + a2 * b2 + a3 * b1 + a4 * b0;

    let mask = (1u128 << 51) - 1;
    let c = r0 >> 51;
    let h0 = (r0 & mask) as u64;
    let r1 = r1 + c;
    let c = r1 >> 51;
    let h1 = (r1 & mask) as u64;
    let r2 = r2 + c;
    let c = r2 >> 51;
    let h2 = (r2 & mask) as u64;
    let r3 = r3 + c;
    let c = r3 >> 51;
    let h3 = (r3 & mask) as u64;
    let r4 = r4 + c;
    let c = r4 >> 51;
    let h4 = (r4 & mask) as u64;
    let h0 = h0 + (c as u64) * 19;
    let mut h = [h0, h1, h2, h3, h4];
    fe_carry(&mut h);
    h
}

fn fe_sq(a: &Fe) -> Fe {
    fe_mul(a, a)
}

fn fe_mul_small(a: &Fe, n: u32) -> Fe {
    let n = n as u128;
    let r0 = (a[0] as u128) * n;
    let r1 = (a[1] as u128) * n;
    let r2 = (a[2] as u128) * n;
    let r3 = (a[3] as u128) * n;
    let r4 = (a[4] as u128) * n;
    let mask = (1u128 << 51) - 1;
    let c = r0 >> 51;
    let h0 = (r0 & mask) as u64;
    let r1 = r1 + c;
    let c = r1 >> 51;
    let h1 = (r1 & mask) as u64;
    let r2 = r2 + c;
    let c = r2 >> 51;
    let h2 = (r2 & mask) as u64;
    let r3 = r3 + c;
    let c = r3 >> 51;
    let h3 = (r3 & mask) as u64;
    let r4 = r4 + c;
    let c = r4 >> 51;
    let h4 = (r4 & mask) as u64;
    let h0 = h0 + (c as u64) * 19;
    let mut h = [h0, h1, h2, h3, h4];
    fe_carry(&mut h);
    h
}

/// Compute `a^(p-2) mod p` via the addition chain in RFC 7748 §6.1 /
/// Bernstein's classic. Constant-time.
fn fe_invert(z: &Fe) -> Fe {
    let z2 = fe_sq(z);
    let z9 = fe_sq(&fe_sq(&z2));
    let z9 = fe_mul(&z9, z);
    let z11 = fe_mul(&z9, &z2);
    let mut z2_5_0 = fe_sq(&z11);
    z2_5_0 = fe_mul(&z2_5_0, &z9);

    let mut z2_10_0 = fe_sq(&z2_5_0);
    for _ in 0..4 {
        z2_10_0 = fe_sq(&z2_10_0);
    }
    z2_10_0 = fe_mul(&z2_10_0, &z2_5_0);

    let mut z2_20_0 = fe_sq(&z2_10_0);
    for _ in 0..9 {
        z2_20_0 = fe_sq(&z2_20_0);
    }
    z2_20_0 = fe_mul(&z2_20_0, &z2_10_0);

    let mut z2_40_0 = fe_sq(&z2_20_0);
    for _ in 0..19 {
        z2_40_0 = fe_sq(&z2_40_0);
    }
    z2_40_0 = fe_mul(&z2_40_0, &z2_20_0);

    let mut z2_50_0 = fe_sq(&z2_40_0);
    for _ in 0..9 {
        z2_50_0 = fe_sq(&z2_50_0);
    }
    z2_50_0 = fe_mul(&z2_50_0, &z2_10_0);

    let mut z2_100_0 = fe_sq(&z2_50_0);
    for _ in 0..49 {
        z2_100_0 = fe_sq(&z2_100_0);
    }
    z2_100_0 = fe_mul(&z2_100_0, &z2_50_0);

    let mut z2_200_0 = fe_sq(&z2_100_0);
    for _ in 0..99 {
        z2_200_0 = fe_sq(&z2_200_0);
    }
    z2_200_0 = fe_mul(&z2_200_0, &z2_100_0);

    let mut z2_250_0 = fe_sq(&z2_200_0);
    for _ in 0..49 {
        z2_250_0 = fe_sq(&z2_250_0);
    }
    z2_250_0 = fe_mul(&z2_250_0, &z2_50_0);

    let mut result = fe_sq(&z2_250_0);
    for _ in 0..4 {
        result = fe_sq(&result);
    }
    fe_mul(&result, &z11)
}

fn fe_cswap(swap: u64, a: &mut Fe, b: &mut Fe) {
    let mask = 0u64.wrapping_sub(swap);
    for i in 0..5 {
        let t = mask & (a[i] ^ b[i]);
        a[i] ^= t;
        b[i] ^= t;
    }
}

/// Scalar-by-point on Curve25519's Montgomery form (X25519 per RFC 7748).
pub fn x25519(scalar: &[u8; SCALAR_SIZE], u: &[u8; POINT_SIZE]) -> [u8; POINT_SIZE] {
    // Clamp scalar per RFC 7748 §5.
    let mut k = *scalar;
    k[0] &= 248;
    k[31] &= 127;
    k[31] |= 64;

    let x1 = fe_from_bytes(u);
    let mut x2 = ONE;
    let mut z2 = ZERO;
    let mut x3 = x1;
    let mut z3 = ONE;
    let mut swap: u64 = 0;

    for t in (0..=254).rev() {
        let k_t = ((k[t / 8] >> (t & 7)) & 1) as u64;
        swap ^= k_t;
        fe_cswap(swap, &mut x2, &mut x3);
        fe_cswap(swap, &mut z2, &mut z3);
        swap = k_t;

        let a = fe_add(&x2, &z2);
        let aa = fe_sq(&a);
        let b = fe_sub(&x2, &z2);
        let bb = fe_sq(&b);
        let e = fe_sub(&aa, &bb);
        let c = fe_add(&x3, &z3);
        let d = fe_sub(&x3, &z3);
        let da = fe_mul(&d, &a);
        let cb = fe_mul(&c, &b);
        x3 = fe_sq(&fe_add(&da, &cb));
        z3 = fe_mul(&x1, &fe_sq(&fe_sub(&da, &cb)));
        x2 = fe_mul(&aa, &bb);
        z2 = fe_mul(&e, &fe_add(&aa, &fe_mul_small(&e, 121_665)));
    }

    fe_cswap(swap, &mut x2, &mut x3);
    fe_cswap(swap, &mut z2, &mut z3);

    let z2_inv = fe_invert(&z2);
    let out = fe_mul(&x2, &z2_inv);
    fe_to_bytes(&out)
}

/// Public key from a 32-byte private scalar: multiply the base point u=9.
pub fn x25519_public(scalar: &[u8; SCALAR_SIZE]) -> [u8; POINT_SIZE] {
    let mut base = [0u8; 32];
    base[0] = 9;
    x25519(scalar, &base)
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
        let mut s = String::with_capacity(b.len() * 2);
        for x in b {
            s.push_str(&format!("{x:02x}"));
        }
        s
    }

    /// RFC 7748 §5.2 first test vector.
    #[test]
    fn rfc7748_vec1() {
        let scalar: [u8; 32] =
            unhex("a546e36bf0527c9d3b16154b82465edd62144c0ac1fc5a18506a2244ba449ac4")
                .try_into()
                .unwrap();
        let u: [u8; 32] = unhex("e6db6867583030db3594c1a424b15f7c726624ec26b3353b10a903a6d0ab1c4c")
            .try_into()
            .unwrap();
        let want = "c3da55379de9c6908e94ea4df28d084f32eccf03491c71f754b4075577a28552";
        assert_eq!(hex(&x25519(&scalar, &u)), want);
    }

    /// RFC 7748 §5.2 second test vector.
    #[test]
    fn rfc7748_vec2() {
        let scalar: [u8; 32] =
            unhex("4b66e9d4d1b4673c5ad22691957d6af5c11b6421e0ea01d42ca4169e7918ba0d")
                .try_into()
                .unwrap();
        let u: [u8; 32] = unhex("e5210f12786811d3f4b7959d0538ae2c31dbe7106fc03c3efc4cd549c715a493")
            .try_into()
            .unwrap();
        let want = "95cbde9476e8907d7aade45cb4b873f88b595a68799fa152e6f8f7647aac7957";
        assert_eq!(hex(&x25519(&scalar, &u)), want);
    }

    /// RFC 7748 §6.1 ECDH key agreement vector.
    #[test]
    fn rfc7748_ecdh() {
        let alice_priv: [u8; 32] =
            unhex("77076d0a7318a57d3c16c17251b26645df4c2f87ebc0992ab177fba51db92c2a")
                .try_into()
                .unwrap();
        let bob_priv: [u8; 32] =
            unhex("5dab087e624a8a4b79e17f8b83800ee66f3bb1292618b6fd1c2f8b27ff88e0eb")
                .try_into()
                .unwrap();

        let alice_pub = x25519_public(&alice_priv);
        assert_eq!(
            hex(&alice_pub),
            "8520f0098930a754748b7ddcb43ef75a0dbf3a0d26381af4eba4a98eaa9b4e6a"
        );
        let bob_pub = x25519_public(&bob_priv);
        assert_eq!(
            hex(&bob_pub),
            "de9edb7d7b7dc1b4d35b61c2ece435373f8343c85b78674dadfc7e146f882b4f"
        );

        let shared_a = x25519(&alice_priv, &bob_pub);
        let shared_b = x25519(&bob_priv, &alice_pub);
        assert_eq!(shared_a, shared_b);
        assert_eq!(
            hex(&shared_a),
            "4a5d9d5ba4ce2de1728e3bf480350f25e07e21c947d19e3376f09b3c1e161742"
        );
    }
}
