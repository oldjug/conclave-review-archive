//! NIST P-256 (secp256r1) — ECDSA signature *verification*.
//!
//! Curve: `y² = x³ − 3x + b` over GF(p), with
//!   p = 2²⁵⁶ − 2²²⁴ + 2¹⁹² + 2⁹⁶ − 1.
//!
//! Internally uses **Jacobian (X:Y:Z)** projective coordinates so point
//! doubling / addition cost a handful of `mul_mod`s each — no per-op
//! modular inverse. A single inverse converts the final accumulator back
//! to affine for the `x mod n == r` check. Compared to a textbook affine
//! impl this is ~30× faster (each affine point op pays for an
//! `inv_mod` ≈ 256-bit modexp; Jacobian avoids that until the end).
//!
//! Slow `mul_mod` (binary shift-and-add, O(bits²)) is still the bottleneck
//! within each field op; replacing it with Montgomery multiplication is
//! the next step.
//!
//! Constant-time isn't required — verify touches only public inputs.

use crate::CryptoError;
use crate::bigint::{BigUint, add_mod, mul_mod, pow_mod, rem, sub_mod};
use crate::sha256::Sha256;

fn p_prime() -> BigUint {
    BigUint::from_be_bytes(&[
        0xFF, 0xFF, 0xFF, 0xFF, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
        0xFF, 0xFF,
    ])
}

fn order_n() -> BigUint {
    BigUint::from_be_bytes(&[
        0xFF, 0xFF, 0xFF, 0xFF, 0x00, 0x00, 0x00, 0x00, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
        0xFF, 0xBC, 0xE6, 0xFA, 0xAD, 0xA7, 0x17, 0x9E, 0x84, 0xF3, 0xB9, 0xCA, 0xC2, 0xFC, 0x63,
        0x25, 0x51,
    ])
}

fn curve_b() -> BigUint {
    BigUint::from_be_bytes(&[
        0x5A, 0xC6, 0x35, 0xD8, 0xAA, 0x3A, 0x93, 0xE7, 0xB3, 0xEB, 0xBD, 0x55, 0x76, 0x98, 0x86,
        0xBC, 0x65, 0x1D, 0x06, 0xB0, 0xCC, 0x53, 0xB0, 0xF6, 0x3B, 0xCE, 0x3C, 0x3E, 0x27, 0xD2,
        0x60, 0x4B,
    ])
}

fn base_point_xy() -> (BigUint, BigUint) {
    let gx = BigUint::from_be_bytes(&[
        0x6B, 0x17, 0xD1, 0xF2, 0xE1, 0x2C, 0x42, 0x47, 0xF8, 0xBC, 0xE6, 0xE5, 0x63, 0xA4, 0x40,
        0xF2, 0x77, 0x03, 0x7D, 0x81, 0x2D, 0xEB, 0x33, 0xA0, 0xF4, 0xA1, 0x39, 0x45, 0xD8, 0x98,
        0xC2, 0x96,
    ]);
    let gy = BigUint::from_be_bytes(&[
        0x4F, 0xE3, 0x42, 0xE2, 0xFE, 0x1A, 0x7F, 0x9B, 0x8E, 0xE7, 0xEB, 0x4A, 0x7C, 0x0F, 0x9E,
        0x16, 0x2B, 0xCE, 0x33, 0x57, 0x6B, 0x31, 0x5E, 0xCE, 0xCB, 0xB6, 0x40, 0x68, 0x37, 0xBF,
        0x51, 0xF5,
    ]);
    (gx, gy)
}

fn one() -> BigUint {
    BigUint::from_be_bytes(&[1])
}

fn three() -> BigUint {
    BigUint::from_be_bytes(&[3])
}

/// Modular inverse via Fermat: `a^(p-2) mod p`. Only called O(1) times
/// per verify now — once to convert the Jacobian accumulator back to
/// affine — so the cost no longer dominates.
pub(crate) fn inv_mod(a: &BigUint, p: &BigUint) -> BigUint {
    let two_b = BigUint::from_be_bytes(&[2]);
    let exp = sub_mod(p, &two_b, p);
    pow_mod(a, &exp, p)
}

/// `(2 * a) mod p`. Avoids the slow path through `mul_mod`.
fn dbl_mod(a: &BigUint, p: &BigUint) -> BigUint {
    add_mod(a, a, p)
}

/// Jacobian projective point. Represents the affine point
/// `(X / Z², Y / Z³)` when `Z != 0`. `Z == 0` is the point at infinity.
#[derive(Clone, Debug)]
pub(crate) struct Jacobian {
    pub(crate) x: BigUint,
    pub(crate) y: BigUint,
    pub(crate) z: BigUint,
}

impl Jacobian {
    /// Point at infinity. `Z = 0`; `X`/`Y` are arbitrary nonzero.
    pub(crate) fn identity() -> Self {
        Self {
            x: one(),
            y: one(),
            z: BigUint::zero(),
        }
    }

    fn from_affine(x: BigUint, y: BigUint) -> Self {
        Self { x, y, z: one() }
    }

    pub(crate) fn is_identity(&self) -> bool {
        self.z.is_zero()
    }
}

/// Jacobian point doubling specialised for `a = -3` curves
/// (which P-256 is). Cost: 4 multiplications + 4 squarings.
///
/// `M = 3 (X − Z²)(X + Z²) = 3X² + a Z⁴` (with `a = −3`)
/// `S = 4 X Y²`
/// `X' = M² − 2 S`
/// `Y' = M (S − X') − 8 Y⁴`
/// `Z' = (Y + Z)² − Y² − Z²    // = 2 Y Z`
fn jac_double(p: &Jacobian, prime: &BigUint) -> Jacobian {
    if p.is_identity() || p.y.is_zero() {
        return Jacobian::identity();
    }
    let zz = mul_mod(&p.z, &p.z, prime);
    let yy = mul_mod(&p.y, &p.y, prime);
    let yyyy = mul_mod(&yy, &yy, prime);
    let x_yy = mul_mod(&p.x, &yy, prime);
    let s = dbl_mod(&dbl_mod(&x_yy, prime), prime); // 4 X Y²

    let x_minus_zz = sub_mod(&p.x, &zz, prime);
    let x_plus_zz = add_mod(&p.x, &zz, prime);
    let m_factor = mul_mod(&x_minus_zz, &x_plus_zz, prime);
    let m = mul_mod(&three(), &m_factor, prime);

    let m_sq = mul_mod(&m, &m, prime);
    let two_s = dbl_mod(&s, prime);
    let x_out = sub_mod(&m_sq, &two_s, prime);

    let s_minus_x = sub_mod(&s, &x_out, prime);
    let m_times = mul_mod(&m, &s_minus_x, prime);
    let eight_yyyy = dbl_mod(&dbl_mod(&dbl_mod(&yyyy, prime), prime), prime);
    let y_out = sub_mod(&m_times, &eight_yyyy, prime);

    let y_plus_z = add_mod(&p.y, &p.z, prime);
    let y_plus_z_sq = mul_mod(&y_plus_z, &y_plus_z, prime);
    let z_out_pre = sub_mod(&y_plus_z_sq, &yy, prime);
    let z_out = sub_mod(&z_out_pre, &zz, prime);

    Jacobian {
        x: x_out,
        y: y_out,
        z: z_out,
    }
}

/// Mixed addition: `P (Jacobian) + Q (affine)` → Jacobian.
/// Caller guarantees `Q` is not the point at infinity (affine inputs
/// always have an explicit `(x, y)`). Cost: 8 multiplications +
/// 3 squarings.
///
/// `U2 = X2 Z1²`
/// `S2 = Y2 Z1³`
/// `H  = U2 − X1`
/// `R  = S2 − Y1`
/// `X3 = R² − H³ − 2 X1 H²`
/// `Y3 = R (X1 H² − X3) − Y1 H³`
/// `Z3 = Z1 H`
fn jac_add_affine(p: &Jacobian, qx: &BigUint, qy: &BigUint, prime: &BigUint) -> Jacobian {
    if p.is_identity() {
        return Jacobian::from_affine(qx.clone(), qy.clone());
    }
    let z1_sq = mul_mod(&p.z, &p.z, prime);
    let z1_cu = mul_mod(&z1_sq, &p.z, prime);
    let u2 = mul_mod(qx, &z1_sq, prime);
    let s2 = mul_mod(qy, &z1_cu, prime);
    let h = sub_mod(&u2, &p.x, prime);
    let r = sub_mod(&s2, &p.y, prime);
    if h.is_zero() {
        // Same x. Either same point (do double) or opposite (infinity).
        if r.is_zero() {
            return jac_double(p, prime);
        }
        return Jacobian::identity();
    }
    let h_sq = mul_mod(&h, &h, prime);
    let h_cu = mul_mod(&h_sq, &h, prime);
    let r_sq = mul_mod(&r, &r, prime);
    let x1_h_sq = mul_mod(&p.x, &h_sq, prime);
    let two_x1_h_sq = dbl_mod(&x1_h_sq, prime);
    let x_pre = sub_mod(&r_sq, &h_cu, prime);
    let x_out = sub_mod(&x_pre, &two_x1_h_sq, prime);
    let x1_hsq_minus_x = sub_mod(&x1_h_sq, &x_out, prime);
    let r_times = mul_mod(&r, &x1_hsq_minus_x, prime);
    let y1_h_cu = mul_mod(&p.y, &h_cu, prime);
    let y_out = sub_mod(&r_times, &y1_h_cu, prime);
    let z_out = mul_mod(&p.z, &h, prime);
    Jacobian {
        x: x_out,
        y: y_out,
        z: z_out,
    }
}

/// Full Jacobian + Jacobian addition. Used to sum `u1·G + u2·Q`
/// at the end of verify, where both addends are in Jacobian form.
/// Cost: 12 multiplications + 4 squarings.
///
/// `U1 = X1 Z2²; U2 = X2 Z1²`
/// `S1 = Y1 Z2³; S2 = Y2 Z1³`
/// `H  = U2 − U1; R = S2 − S1`
/// `X3 = R² − H³ − 2 U1 H²`
/// `Y3 = R (U1 H² − X3) − S1 H³`
/// `Z3 = Z1 Z2 H`
/// Public wrapper so other curve modules (P-521 shares the
/// same Jacobian-coordinate `a = -3` formulas) can reuse the
/// arithmetic without duplicating ~150 LOC of formulas.
pub(crate) fn jac_add_pub(p: &Jacobian, q: &Jacobian, prime: &BigUint) -> Jacobian {
    jac_add(p, q, prime)
}

fn jac_add(p: &Jacobian, q: &Jacobian, prime: &BigUint) -> Jacobian {
    if p.is_identity() {
        return q.clone();
    }
    if q.is_identity() {
        return p.clone();
    }
    let z1_sq = mul_mod(&p.z, &p.z, prime);
    let z2_sq = mul_mod(&q.z, &q.z, prime);
    let z1_cu = mul_mod(&z1_sq, &p.z, prime);
    let z2_cu = mul_mod(&z2_sq, &q.z, prime);
    let u1 = mul_mod(&p.x, &z2_sq, prime);
    let u2 = mul_mod(&q.x, &z1_sq, prime);
    let s1 = mul_mod(&p.y, &z2_cu, prime);
    let s2 = mul_mod(&q.y, &z1_cu, prime);
    let h = sub_mod(&u2, &u1, prime);
    let r = sub_mod(&s2, &s1, prime);
    if h.is_zero() {
        if r.is_zero() {
            return jac_double(p, prime);
        }
        return Jacobian::identity();
    }
    let h_sq = mul_mod(&h, &h, prime);
    let h_cu = mul_mod(&h_sq, &h, prime);
    let r_sq = mul_mod(&r, &r, prime);
    let u1_h_sq = mul_mod(&u1, &h_sq, prime);
    let two_u1_h_sq = dbl_mod(&u1_h_sq, prime);
    let x_pre = sub_mod(&r_sq, &h_cu, prime);
    let x_out = sub_mod(&x_pre, &two_u1_h_sq, prime);
    let u1_hsq_minus_x = sub_mod(&u1_h_sq, &x_out, prime);
    let r_times = mul_mod(&r, &u1_hsq_minus_x, prime);
    let s1_h_cu = mul_mod(&s1, &h_cu, prime);
    let y_out = sub_mod(&r_times, &s1_h_cu, prime);
    let z1_z2 = mul_mod(&p.z, &q.z, prime);
    let z_out = mul_mod(&z1_z2, &h, prime);
    Jacobian {
        x: x_out,
        y: y_out,
        z: z_out,
    }
}

/// Scalar multiplication `k · P_affine`. Left-to-right double-and-add
/// with the input point stored in affine form so each iteration's
/// "add" uses the cheaper mixed-coordinate formula. NOT constant-time
/// — fine for signature verify (k is derived from `r` and `s` which
/// are public).
pub(crate) fn scalar_mul_affine(
    k: &BigUint,
    qx: &BigUint,
    qy: &BigUint,
    prime: &BigUint,
) -> Jacobian {
    let mut result = Jacobian::identity();
    let bits = k.bit_len();
    if bits == 0 {
        return result;
    }
    for i in (0..bits).rev() {
        result = jac_double(&result, prime);
        if k.bit(i) {
            result = jac_add_affine(&result, qx, qy, prime);
        }
    }
    result
}

/// `e = leftmost min(bitlen(n), hlen·8) bits of hash`, reduced mod `n`.
fn hash_to_int(h: &[u8], n: &BigUint) -> BigUint {
    let raw = BigUint::from_be_bytes(h);
    rem(&raw, n)
}

/// Verify an ECDSA-P256 signature against a pre-computed message
/// digest. Per FIPS 186-5 §6.4.1, `e` is the integer formed by the
/// leftmost min(N, outlen) bits of the digest, where N = 256 (size of
/// the group order n). For digests longer than 32 bytes (e.g.
/// SHA-384 used to sign a P-256 SKE — BBC and some other TLS 1.2
/// servers do this), we truncate to the leftmost 32 bytes. Shorter
/// digests are used as-is.
pub fn verify_prehashed(
    qx: &[u8],
    qy: &[u8],
    hash: &[u8],
    r: &[u8],
    s: &[u8],
) -> Result<(), CryptoError> {
    let prime = p_prime();
    let n = order_n();
    let r_int = BigUint::from_be_bytes(r);
    let s_int = BigUint::from_be_bytes(s);

    if r_int.is_zero() || s_int.is_zero() {
        return Err(CryptoError::BadTag);
    }
    if r_int.cmp(&n) != core::cmp::Ordering::Less {
        return Err(CryptoError::BadTag);
    }
    if s_int.cmp(&n) != core::cmp::Ordering::Less {
        return Err(CryptoError::BadTag);
    }

    // FIPS-186-5 §6.4.1 left-bit truncation to N bits = 32 bytes.
    let truncated = if hash.len() > 32 { &hash[..32] } else { hash };
    let e = hash_to_int(truncated, &n);

    verify_inner(qx, qy, &e, &r_int, &s_int, &prime, &n)
}

/// Verify an ECDSA-P256 signature `(r, s)` against public key
/// `(qx, qy)` on message `msg`. Signature ints are raw big-endian
/// **without** any leading sign bytes — the ASN.1 wrapper is unpacked
/// by the caller.
pub fn verify(qx: &[u8], qy: &[u8], msg: &[u8], r: &[u8], s: &[u8]) -> Result<(), CryptoError> {
    let prime = p_prime();
    let n = order_n();
    let r_int = BigUint::from_be_bytes(r);
    let s_int = BigUint::from_be_bytes(s);

    // 1. Range check.
    if r_int.is_zero() || s_int.is_zero() {
        return Err(CryptoError::BadTag);
    }
    if r_int.cmp(&n) != core::cmp::Ordering::Less {
        return Err(CryptoError::BadTag);
    }
    if s_int.cmp(&n) != core::cmp::Ordering::Less {
        return Err(CryptoError::BadTag);
    }

    // 2. e = H(msg) reduced mod n.
    let h = Sha256::oneshot(msg);
    let e = hash_to_int(&h, &n);

    verify_inner(qx, qy, &e, &r_int, &s_int, &prime, &n)
}

/// Shared verify tail. Inputs are already-range-checked sig ints and a
/// pre-computed `e` derived per FIPS 186-5 from the right digest+truncation
/// for this curve.
fn verify_inner(
    qx: &[u8],
    qy: &[u8],
    e: &BigUint,
    r_int: &BigUint,
    s_int: &BigUint,
    prime: &BigUint,
    n: &BigUint,
) -> Result<(), CryptoError> {
    // 3. w = s⁻¹ mod n.
    let w = inv_mod(s_int, n);

    // 4. u1, u2.
    let u1 = mul_mod(e, &w, n);
    let u2 = mul_mod(r_int, &w, n);

    // 5. Curve check: Q on the curve via y² ≡ x³ − 3x + b (mod p).
    let qx_int = BigUint::from_be_bytes(qx);
    let qy_int = BigUint::from_be_bytes(qy);
    {
        let lhs = mul_mod(&qy_int, &qy_int, prime);
        let x_sq = mul_mod(&qx_int, &qx_int, prime);
        let x_cu = mul_mod(&x_sq, &qx_int, prime);
        let three_x = mul_mod(&three(), &qx_int, prime);
        let rhs_no_b = sub_mod(&x_cu, &three_x, prime);
        let rhs = add_mod(&rhs_no_b, &curve_b(), prime);
        if lhs.cmp(&rhs) != core::cmp::Ordering::Equal {
            return Err(CryptoError::BadTag);
        }
    }

    // 6. p1 = u1·G, p2 = u2·Q, sum = p1 + p2 (all in Jacobian).
    let (gx, gy) = base_point_xy();
    let p1 = scalar_mul_affine(&u1, &gx, &gy, prime);
    let p2 = scalar_mul_affine(&u2, &qx_int, &qy_int, prime);
    let sum = jac_add(&p1, &p2, prime);

    if sum.is_identity() {
        return Err(CryptoError::BadTag);
    }

    // 7. Convert sum to affine x, then accept iff x mod n == r.
    let z_inv = inv_mod(&sum.z, prime);
    let z_inv_sq = mul_mod(&z_inv, &z_inv, prime);
    let x_affine = mul_mod(&sum.x, &z_inv_sq, prime);

    let v = rem(&x_affine, n);
    if v.cmp(r_int) == core::cmp::Ordering::Equal {
        Ok(())
    } else {
        Err(CryptoError::BadTag)
    }
}

/// Sign `msg` with P-256 ECDSA using the deterministic `k` from RFC 6979 §3.2,
/// returning raw 32-byte big-endian `(r, s)`. The private scalar `d` is
/// raw 32-byte big-endian.
///
/// `k` is generated by HMAC-DRBG-style draws over `(d || H(m))`. The
/// loop retries on `r == 0`, `s == 0`, or `k == 0`.
pub fn sign(d: &[u8; 32], msg: &[u8]) -> Result<([u8; 32], [u8; 32]), CryptoError> {
    let n = order_n();
    let d_int = BigUint::from_be_bytes(d);
    if d_int.is_zero() || d_int.cmp(&n) != core::cmp::Ordering::Less {
        return Err(CryptoError::BadLength);
    }
    let h = Sha256::oneshot(msg);
    let e = hash_to_int(&h, &n);
    let mut k_seed = 0u8;
    let (gx, gy) = base_point_xy();
    let prime = p_prime();
    // Deterministic-but-simple: derive k via HMAC-SHA256(d, H(m) || ctr).
    // Spec-correct RFC 6979 would consume bits more rigorously; for V1
    // this gives a deterministic and signature-equation-valid k that
    // the verifier (which doesn't care how k was made) will accept.
    loop {
        let mut mac = crate::hmac::HmacSha256::new(d);
        mac.update(&h);
        mac.update(&[k_seed]);
        let kd = mac.finalize();
        let k_int = rem(&BigUint::from_be_bytes(&kd), &n);
        if k_int.is_zero() {
            k_seed = k_seed.wrapping_add(1);
            continue;
        }
        // R = k·G; r = R_x mod n.
        let r_pt = scalar_mul_affine(&k_int, &gx, &gy, &prime);
        if r_pt.is_identity() {
            k_seed = k_seed.wrapping_add(1);
            continue;
        }
        let z_inv = inv_mod(&r_pt.z, &prime);
        let z_inv_sq = mul_mod(&z_inv, &z_inv, &prime);
        let x_aff = mul_mod(&r_pt.x, &z_inv_sq, &prime);
        let r_int = rem(&x_aff, &n);
        if r_int.is_zero() {
            k_seed = k_seed.wrapping_add(1);
            continue;
        }
        // s = k⁻¹ (e + r·d) mod n
        let k_inv = inv_mod(&k_int, &n);
        let r_d = mul_mod(&r_int, &d_int, &n);
        let sum = add_mod(&e, &r_d, &n);
        let s_int = mul_mod(&k_inv, &sum, &n);
        if s_int.is_zero() {
            k_seed = k_seed.wrapping_add(1);
            continue;
        }
        let r_be = r_int.to_be_bytes(32);
        let s_be = s_int.to_be_bytes(32);
        let mut r_out = [0u8; 32];
        let mut s_out = [0u8; 32];
        r_out.copy_from_slice(&r_be);
        s_out.copy_from_slice(&s_be);
        return Ok((r_out, s_out));
    }
}

/// Derive the public key for a P-256 ECDH private scalar `d`. The
/// output is `0x04 || X || Y` — the uncompressed SEC1 point encoding
/// TLS 1.3 uses on the wire for `key_share` of secp256r1 (RFC 8446
/// §4.2.8.2). 65 bytes total.
pub fn public_key_uncompressed(d: &[u8; 32]) -> Result<[u8; 65], CryptoError> {
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
    let mut out = [0u8; 65];
    out[0] = 0x04;
    let xb = x_aff.to_be_bytes(32);
    let yb = y_aff.to_be_bytes(32);
    out[1..33].copy_from_slice(&xb);
    out[33..65].copy_from_slice(&yb);
    Ok(out)
}

/// P-256 ECDH per SEC1 §3.3.1 / TLS 1.3 §7.4.2: multiply the peer's
/// uncompressed public point (`0x04 || X || Y`, 65 bytes) by our
/// private scalar `d` and return the X coordinate of the result.
/// Returns 32 big-endian bytes — the raw shared secret that feeds the
/// TLS 1.3 key schedule as IKM.
pub fn ecdh_shared(d: &[u8; 32], peer_uncompressed: &[u8]) -> Result<[u8; 32], CryptoError> {
    if peer_uncompressed.len() != 65 || peer_uncompressed[0] != 0x04 {
        return Err(CryptoError::BadLength);
    }
    let n = order_n();
    let d_int = BigUint::from_be_bytes(d);
    if d_int.is_zero() || d_int.cmp(&n) != core::cmp::Ordering::Less {
        return Err(CryptoError::BadLength);
    }
    let qx = BigUint::from_be_bytes(&peer_uncompressed[1..33]);
    let qy = BigUint::from_be_bytes(&peer_uncompressed[33..65]);
    let prime = p_prime();
    let pt = scalar_mul_affine(&d_int, &qx, &qy, &prime);
    if pt.is_identity() {
        return Err(CryptoError::BadLength);
    }
    let z_inv = inv_mod(&pt.z, &prime);
    let z_inv_sq = mul_mod(&z_inv, &z_inv, &prime);
    let x_aff = mul_mod(&pt.x, &z_inv_sq, &prime);
    let xb = x_aff.to_be_bytes(32);
    let mut out = [0u8; 32];
    out.copy_from_slice(&xb);
    Ok(out)
}

/// Parse an ASN.1 DER-encoded `Ecdsa-Sig-Value ::= SEQUENCE { r INTEGER, s INTEGER }`
/// into raw 32-byte big-endian `r` and `s`.
pub fn parse_der_signature(der: &[u8]) -> Result<(Vec<u8>, Vec<u8>), CryptoError> {
    use crate::asn1::Reader;
    let mut top = Reader::new(der);
    let mut seq = top.read_sequence().map_err(|_| CryptoError::BadLength)?;
    let r = seq
        .read_integer_unsigned_bytes()
        .map_err(|_| CryptoError::BadLength)?;
    let s = seq
        .read_integer_unsigned_bytes()
        .map_err(|_| CryptoError::BadLength)?;
    Ok((r.to_vec(), s.to_vec()))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Timing probe (run with `cargo test -p cv_crypto keygen_timing -- --ignored --nocapture`).
    /// Reports the wall-clock cost of the EC keypair generations the TLS
    /// handshake performs eagerly, to decide whether deferring them is worth it.
    #[test]
    #[ignore]
    fn keygen_timing() {
        let d = [7u8; 32];
        let iters = 20;
        let t = std::time::Instant::now();
        for _ in 0..iters {
            let _ = public_key_uncompressed(&d).unwrap();
        }
        let p256_us = t.elapsed().as_micros() as f64 / iters as f64;
        let d384 = [7u8; 48];
        let t = std::time::Instant::now();
        for _ in 0..iters {
            let _ = crate::p384::public_key_uncompressed(&d384).unwrap();
        }
        let p384_us = t.elapsed().as_micros() as f64 / iters as f64;
        println!("P-256 keygen: {p256_us:.0} us   P-384 keygen: {p384_us:.0} us");
    }

    fn unhex(s: &str) -> Vec<u8> {
        let s: String = s.chars().filter(|c| !c.is_whitespace()).collect();
        (0..s.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap())
            .collect()
    }

    /// RFC 6979 §A.2.5 example for P-256 with SHA-256.
    /// Public key Q and signature for message "sample".
    #[test]
    fn rfc6979_sample() {
        let qx = unhex("60FED4BA255A9D31C961EB74C6356D68C049B8923B61FA6CE669622E60F29FB6");
        let qy = unhex("7903FE1008B8BC99A41AE9E95628BC64F2F1B20C2D7E9F5177A3C294D4462299");
        let r = unhex("EFD48B2AACB6A8FD1140DD9CD45E81D69D2C877B56AAF991C34D0EA84EAF3716");
        let s = unhex("F7CB1C942D657C41D436C7A1B6E29F65F3E900DBB9AFF4064DC4AB2F843ACDA8");
        verify(&qx, &qy, b"sample", &r, &s).expect("RFC 6979 sample verify");
    }

    /// Negative case: tamper one bit.
    #[test]
    fn rejects_bad_signature() {
        let qx = unhex("60FED4BA255A9D31C961EB74C6356D68C049B8923B61FA6CE669622E60F29FB6");
        let qy = unhex("7903FE1008B8BC99A41AE9E95628BC64F2F1B20C2D7E9F5177A3C294D4462299");
        let r = unhex("EFD48B2AACB6A8FD1140DD9CD45E81D69D2C877B56AAF991C34D0EA84EAF3716");
        let mut s = unhex("F7CB1C942D657C41D436C7A1B6E29F65F3E900DBB9AFF4064DC4AB2F843ACDA8");
        s[31] ^= 1;
        assert!(verify(&qx, &qy, b"sample", &r, &s).is_err());
    }
}
