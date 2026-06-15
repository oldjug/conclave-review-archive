//! Small big-integer module — only what RSA verification and ECDSA
//! validation need.
//!
//! Representation: little-endian `Vec<u64>` limbs, unsigned. Operations:
//! `mul_mod`, `pow_mod`, `add`, `sub_mod`, `cmp`. Multiplication is
//! schoolbook with 128-bit intermediates; reduction is Knuth Algorithm D
//! long division (one 64-bit limb per step). This replaced the old
//! O(bits²) bit-serial "shift-and-add" modmul that allocated a fresh `Vec`
//! on every bit — that churn was the ~1s-per-TLS-handshake bottleneck.

use core::cmp::Ordering;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct BigUint {
    limbs: Vec<u64>,
}

impl BigUint {
    pub fn zero() -> Self {
        Self { limbs: vec![] }
    }

    pub fn from_be_bytes(bytes: &[u8]) -> Self {
        let mut padded = bytes.to_vec();
        // Pad to multiple of 8 from the left.
        while padded.len() % 8 != 0 {
            padded.insert(0, 0);
        }
        let n_limbs = padded.len() / 8;
        let mut limbs = vec![0u64; n_limbs];
        for i in 0..n_limbs {
            let off = padded.len() - (i + 1) * 8;
            limbs[i] = u64::from_be_bytes(padded[off..off + 8].try_into().unwrap());
        }
        let mut v = Self { limbs };
        v.normalize();
        v
    }

    pub fn to_be_bytes(&self, len: usize) -> Vec<u8> {
        let mut out = vec![0u8; len];
        for (i, &lm) in self.limbs.iter().enumerate() {
            let be = lm.to_be_bytes();
            // Place this limb at position len - (i+1)*8 .. len - i*8.
            let end = len.saturating_sub(i * 8);
            let start = len.saturating_sub((i + 1) * 8);
            let take = end - start;
            // be is 8 bytes BE; we want the low `take` bytes.
            out[start..end].copy_from_slice(&be[8 - take..]);
            if end <= 8 && i * 8 + 8 > len {
                break;
            }
        }
        out
    }

    fn normalize(&mut self) {
        while self.limbs.last() == Some(&0) {
            self.limbs.pop();
        }
    }

    pub fn is_zero(&self) -> bool {
        self.limbs.iter().all(|&l| l == 0)
    }

    pub fn bit_len(&self) -> usize {
        match self.limbs.iter().rposition(|&l| l != 0) {
            Some(i) => i * 64 + (64 - self.limbs[i].leading_zeros() as usize),
            None => 0,
        }
    }

    pub fn bit(&self, idx: usize) -> bool {
        let li = idx / 64;
        let off = idx % 64;
        self.limbs.get(li).is_some_and(|&l| (l >> off) & 1 == 1)
    }

    pub fn cmp(&self, other: &Self) -> Ordering {
        let la = self.limbs.iter().rposition(|&l| l != 0);
        let lb = other.limbs.iter().rposition(|&l| l != 0);
        match (la, lb) {
            (None, None) => Ordering::Equal,
            (None, Some(_)) => Ordering::Less,
            (Some(_), None) => Ordering::Greater,
            (Some(a), Some(b)) => {
                if a != b {
                    return a.cmp(&b);
                }
                for i in (0..=a).rev() {
                    match self.limbs[i].cmp(&other.limbs[i]) {
                        Ordering::Equal => continue,
                        o => return o,
                    }
                }
                Ordering::Equal
            }
        }
    }
}

/// `a + b`.
fn add(a: &BigUint, b: &BigUint) -> BigUint {
    let n = a.limbs.len().max(b.limbs.len());
    let mut out = Vec::with_capacity(n + 1);
    let mut carry: u128 = 0;
    for i in 0..n {
        let x = a.limbs.get(i).copied().unwrap_or(0) as u128;
        let y = b.limbs.get(i).copied().unwrap_or(0) as u128;
        let s = x + y + carry;
        out.push(s as u64);
        carry = s >> 64;
    }
    if carry != 0 {
        out.push(carry as u64);
    }
    let mut r = BigUint { limbs: out };
    r.normalize();
    r
}

/// `a - b`. Caller must ensure `a >= b`.
fn sub(a: &BigUint, b: &BigUint) -> BigUint {
    let n = a.limbs.len();
    let mut out = vec![0u64; n];
    let mut borrow: i128 = 0;
    for i in 0..n {
        let x = a.limbs[i] as i128;
        let y = b.limbs.get(i).copied().unwrap_or(0) as i128;
        let d = x - y - borrow;
        if d < 0 {
            out[i] = (d + (1i128 << 64)) as u64;
            borrow = 1;
        } else {
            out[i] = d as u64;
            borrow = 0;
        }
    }
    let mut r = BigUint { limbs: out };
    r.normalize();
    r
}

/// `(a + b) mod n`. Assumes a < n and b < n.
pub fn add_mod(a: &BigUint, b: &BigUint, n: &BigUint) -> BigUint {
    let s = add(a, b);
    if s.cmp(n) != Ordering::Less {
        sub(&s, n)
    } else {
        s
    }
}

/// `(a - b) mod n`. Assumes a < n and b < n.
pub fn sub_mod(a: &BigUint, b: &BigUint, n: &BigUint) -> BigUint {
    if a.cmp(b) != Ordering::Less {
        sub(a, b)
    } else {
        sub(&add(a, n), b)
    }
}

/// Full product `a * b` (no modulus) via schoolbook multiplication with
/// 128-bit intermediates. O(limbs²), allocation-light (one output Vec).
fn mul_full(a: &BigUint, b: &BigUint) -> BigUint {
    if a.is_zero() || b.is_zero() {
        return BigUint::zero();
    }
    let la = a.limbs.len();
    let lb = b.limbs.len();
    let mut out = vec![0u64; la + lb];
    for i in 0..la {
        let ai = a.limbs[i] as u128;
        let mut carry: u128 = 0;
        for j in 0..lb {
            let prod = ai * (b.limbs[j] as u128) + (out[i + j] as u128) + carry;
            out[i + j] = prod as u64;
            carry = prod >> 64;
        }
        out[i + lb] = carry as u64;
    }
    let mut r = BigUint { limbs: out };
    r.normalize();
    r
}

/// Shift a little-endian limb slice left by `shift` bits (`0..64`),
/// returning a fresh limb vector (one longer if it carries out).
fn shl_limbs(limbs: &[u64], shift: u32) -> Vec<u64> {
    if shift == 0 {
        return limbs.to_vec();
    }
    let mut out = Vec::with_capacity(limbs.len() + 1);
    let mut carry = 0u64;
    for &l in limbs {
        out.push((l << shift) | carry);
        carry = l >> (64 - shift);
    }
    if carry != 0 {
        out.push(carry);
    }
    out
}

/// Shift a little-endian limb slice right by `shift` bits (`0..64`).
fn shr_limbs(limbs: &[u64], shift: u32) -> Vec<u64> {
    if shift == 0 {
        return limbs.to_vec();
    }
    let mut out = vec![0u64; limbs.len()];
    let mut carry = 0u64;
    for i in (0..limbs.len()).rev() {
        let l = limbs[i];
        out[i] = (l >> shift) | carry;
        carry = l << (64 - shift);
    }
    out
}

/// `(u / v, u % v)` via Knuth's Algorithm D (TAOCP vol. 2 §4.3.1) with base
/// B = 2⁶⁴. O(limbs²) and processes a whole 64-bit limb per step — this is
/// what makes P-256 / RSA fast enough for a real TLS handshake. `v` must be
/// non-zero. Robust to non-normalized inputs (uses significant-limb counts).
fn divmod(u: &BigUint, v: &BigUint) -> (BigUint, BigUint) {
    let n = v
        .limbs
        .iter()
        .rposition(|&l| l != 0)
        .map_or(0, |i| i + 1);
    assert!(n > 0, "divmod by zero");
    if u.cmp(v) == Ordering::Less {
        return (BigUint::zero(), u.clone());
    }
    let un_full = u
        .limbs
        .iter()
        .rposition(|&l| l != 0)
        .map_or(0, |i| i + 1);

    // Single-limb divisor: straight long division, no normalization needed.
    if n == 1 {
        let d = v.limbs[0] as u128;
        let mut q = vec![0u64; un_full];
        let mut r: u128 = 0;
        for i in (0..un_full).rev() {
            let cur = (r << 64) | (u.limbs[i] as u128);
            q[i] = (cur / d) as u64;
            r = cur % d;
        }
        let mut qq = BigUint { limbs: q };
        qq.normalize();
        let mut rr = BigUint {
            limbs: vec![r as u64],
        };
        rr.normalize();
        return (qq, rr);
    }

    let m = un_full - n;
    // D1. Normalize so the divisor's top limb has its high bit set.
    let shift = v.limbs[n - 1].leading_zeros();
    let vn = shl_limbs(&v.limbs[..n], shift);
    let mut un = shl_limbs(&u.limbs[..un_full], shift);
    if un.len() == un_full {
        un.push(0); // guarantee exactly m+n+1 limbs
    }
    debug_assert_eq!(un.len(), m + n + 1);

    let base: u128 = 1u128 << 64;
    let mask: u128 = u64::MAX as u128;
    let mut q = vec![0u64; m + 1];

    for j in (0..=m).rev() {
        // D3. Estimate the quotient limb qhat (corrected to be exact or 1 too high).
        let num = ((un[j + n] as u128) << 64) | (un[j + n - 1] as u128);
        let mut qhat = num / (vn[n - 1] as u128);
        let mut rhat = num % (vn[n - 1] as u128);
        loop {
            if qhat >= base
                || qhat * (vn[n - 2] as u128) > (rhat << 64) + (un[j + n - 2] as u128)
            {
                qhat -= 1;
                rhat += vn[n - 1] as u128;
                if rhat < base {
                    continue;
                }
            }
            break;
        }

        // D4. Multiply and subtract qhat*v from un[j..=j+n] (signed running borrow).
        let mut k: i128 = 0;
        for i in 0..n {
            let p = qhat * (vn[i] as u128);
            let t = (un[j + i] as i128) + k - ((p & mask) as i128);
            un[j + i] = t as u64;
            k = (t >> 64) - ((p >> 64) as i128);
        }
        let t = (un[j + n] as i128) + k;
        un[j + n] = t as u64;

        // D5/D6. If qhat was one too large, add the divisor back and fix qhat.
        let mut qj = qhat as u64;
        if t < 0 {
            qj = qj.wrapping_sub(1);
            let mut carry: u128 = 0;
            for i in 0..n {
                let s = (un[j + i] as u128) + (vn[i] as u128) + carry;
                un[j + i] = s as u64;
                carry = s >> 64;
            }
            un[j + n] = (un[j + n] as u128 + carry) as u64;
        }
        q[j] = qj;
    }

    // D8. Denormalize the remainder: un[0..n] >> shift.
    let r_limbs = shr_limbs(&un[..n], shift);
    let mut qq = BigUint { limbs: q };
    qq.normalize();
    let mut rr = BigUint { limbs: r_limbs };
    rr.normalize();
    (qq, rr)
}

/// `(a * b) mod n` — full product, then a single fast reduction.
pub fn mul_mod(a: &BigUint, b: &BigUint, n: &BigUint) -> BigUint {
    divmod(&mul_full(a, b), n).1
}

/// `a mod n` via Knuth Algorithm D long division.
pub fn rem(a: &BigUint, n: &BigUint) -> BigUint {
    divmod(a, n).1
}

/// Floor division `a / n` (the quotient from Knuth Algorithm D).
pub fn div_floor(a: &BigUint, n: &BigUint) -> BigUint {
    divmod(a, n).0
}

/// `base^exp mod modulus` via left-to-right square-and-multiply.
pub fn pow_mod(base: &BigUint, exp: &BigUint, modulus: &BigUint) -> BigUint {
    if modulus.is_zero() {
        panic!("pow_mod: zero modulus");
    }
    let bits = exp.bit_len();
    if bits == 0 {
        // exp == 0 → 1 (assuming modulus > 1).
        return BigUint { limbs: vec![1] };
    }
    let base_mod = rem(base, modulus);
    let mut result = base_mod.clone();
    for i in (0..bits - 1).rev() {
        result = mul_mod(&result, &result, modulus);
        if exp.bit(i) {
            result = mul_mod(&result, &base_mod, modulus);
        }
    }
    result
}

// ---------------------------------------------------------------------------
// Number-theory primitives required for RSA private-key operations and key
// generation (RFC 8017): unbounded add/sub, modular inverse via the extended
// Euclidean algorithm, Miller–Rabin probabilistic primality, and random prime
// generation. These are NOT constant-time; RSA key generation and signing in a
// browser run on attacker-uncorrelated material, but we still use a CSPRNG for
// every random draw (the caller supplies one). Verification/encryption use only
// the public modulus and were already covered by `pow_mod`/`mul_mod` above.
// ---------------------------------------------------------------------------

/// Public unbounded addition `a + b` (no modulus).
pub fn add_unbounded(a: &BigUint, b: &BigUint) -> BigUint {
    add(a, b)
}

/// Public unbounded subtraction `a - b`. Caller must ensure `a >= b`.
pub fn sub_unbounded(a: &BigUint, b: &BigUint) -> BigUint {
    sub(a, b)
}

/// Public full product `a * b` (no modulus).
pub fn mul_unbounded(a: &BigUint, b: &BigUint) -> BigUint {
    mul_full(a, b)
}

impl BigUint {
    /// Construct from a single small value (used for constants 0,1,2,e).
    pub fn from_u64(v: u64) -> BigUint {
        let mut b = BigUint { limbs: vec![v] };
        b.normalize();
        b
    }

    /// `true` iff the integer is exactly one.
    pub fn is_one(&self) -> bool {
        let sig = self.limbs.iter().rposition(|&l| l != 0);
        matches!(sig, Some(0)) && self.limbs[0] == 1
    }

    /// Whether this is even (low bit clear). Zero counts as even.
    pub fn is_even(&self) -> bool {
        self.limbs.first().is_none_or(|&l| l & 1 == 0)
    }

    /// `self - 1`. Caller must ensure `self >= 1`.
    pub fn dec_one(&self) -> BigUint {
        sub(self, &BigUint::from_u64(1))
    }
}

/// Signed big integer used internally by the extended Euclidean algorithm.
/// Only needed transiently inside `inv_mod_full`.
#[derive(Clone)]
struct SignedBig {
    neg: bool,
    mag: BigUint,
}

impl SignedBig {
    fn zero() -> Self {
        Self { neg: false, mag: BigUint::zero() }
    }
    fn from_uint(m: BigUint) -> Self {
        Self { neg: false, mag: m }
    }
    fn is_zero(&self) -> bool {
        self.mag.is_zero()
    }
    /// `self - other` over the integers.
    fn sub(&self, other: &SignedBig) -> SignedBig {
        // a - b  with signs handled explicitly.
        match (self.neg, other.neg) {
            (false, false) => match self.mag.cmp(&other.mag) {
                Ordering::Less => SignedBig { neg: true, mag: sub(&other.mag, &self.mag) },
                _ => SignedBig { neg: false, mag: sub(&self.mag, &other.mag) },
            },
            (true, true) => match self.mag.cmp(&other.mag) {
                Ordering::Less => SignedBig { neg: false, mag: sub(&other.mag, &self.mag) },
                _ => SignedBig { neg: true, mag: sub(&self.mag, &other.mag) },
            },
            (false, true) => SignedBig { neg: false, mag: add(&self.mag, &other.mag) },
            (true, false) => SignedBig { neg: true, mag: add(&self.mag, &other.mag) },
        }
    }
    /// `self * (unsigned q)`.
    fn mul_uint(&self, q: &BigUint) -> SignedBig {
        let mag = mul_full(&self.mag, q);
        SignedBig { neg: self.neg && !mag.is_zero(), mag }
    }
}

/// Modular inverse `a^{-1} mod m` for an ARBITRARY (not necessarily prime)
/// modulus, via the extended Euclidean algorithm. Returns `None` when the
/// inverse does not exist (`gcd(a, m) != 1`). Needed for RSA: the private
/// exponent `d = e^{-1} mod λ(n)`, and CRT coefficient `qInv = q^{-1} mod p`.
pub fn inv_mod_full(a: &BigUint, m: &BigUint) -> Option<BigUint> {
    if m.is_zero() || m.is_one() {
        return None;
    }
    // Reduce a mod m first.
    let a0 = rem(a, m);
    if a0.is_zero() {
        return None;
    }
    // Extended Euclid on (old_r, r) = (m, a) carrying Bézout t coefficients.
    let mut old_r = m.clone();
    let mut r = a0;
    let mut old_t = SignedBig::zero();
    let mut t = SignedBig::from_uint(BigUint::from_u64(1));
    while !r.is_zero() {
        let (q, rem_v) = divmod(&old_r, &r);
        // (old_r, r) = (r, old_r - q*r)
        old_r = r;
        r = rem_v;
        // (old_t, t) = (t, old_t - q*t)
        let qt = t.mul_uint(&q);
        let new_t = old_t.sub(&qt);
        old_t = t;
        t = new_t;
    }
    // gcd is old_r; invertible only if gcd == 1.
    if !old_r.is_one() {
        return None;
    }
    // old_t is the Bézout coefficient; bring it into [0, m).
    let result = if old_t.neg {
        // old_t mod m  =  m - ((-old_t) mod m)
        let pos = rem(&old_t.mag, m);
        if pos.is_zero() { pos } else { sub(m, &pos) }
    } else {
        rem(&old_t.mag, m)
    };
    Some(result)
}

/// Miller–Rabin probabilistic primality test with `rounds` random witnesses.
/// `rng` must return uniformly random bytes (a CSPRNG in production). The
/// standard error bound is 4^{-rounds}; RSA key generation uses ~40 rounds
/// (FIPS 186-5 Appendix B / handbook of applied cryptography). Small primes
/// are trial-divided first.
pub fn is_probable_prime(n: &BigUint, rounds: usize, rng: &mut dyn FnMut(&mut [u8])) -> bool {
    // 0,1 not prime; 2,3 prime.
    if n.is_zero() || n.is_one() {
        return false;
    }
    let two = BigUint::from_u64(2);
    let three = BigUint::from_u64(3);
    if n.cmp(&three) != Ordering::Greater {
        // n == 2 or n == 3
        return n.cmp(&two) == Ordering::Equal || n.cmp(&three) == Ordering::Equal;
    }
    if n.is_even() {
        return false;
    }
    // Trial-divide by small primes to reject the obvious composites cheaply.
    const SMALL: [u64; 25] = [
        3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89,
        97, 101,
    ];
    for &p in &SMALL {
        let pb = BigUint::from_u64(p);
        if n.cmp(&pb) == Ordering::Equal {
            return true;
        }
        if rem(n, &pb).is_zero() {
            return false;
        }
    }
    // Write n-1 = 2^s * d with d odd.
    let n_minus_1 = n.dec_one();
    let mut d = n_minus_1.clone();
    let mut s = 0usize;
    while d.is_even() {
        d = shr_one(&d);
        s += 1;
    }
    let nbytes = n.bit_len().div_ceil(8);
    'witness: for _ in 0..rounds {
        // Random base a in [2, n-2].
        let a = loop {
            let mut buf = vec![0u8; nbytes];
            rng(&mut buf);
            let cand = rem(&BigUint::from_be_bytes(&buf), &n_minus_1.dec_one()); // in [0, n-3]
            let a = add(&cand, &two); // in [2, n-1]
            if a.cmp(&two) != Ordering::Less && a.cmp(&n_minus_1) == Ordering::Less {
                break a;
            }
        };
        let mut x = pow_mod(&a, &d, n);
        if x.is_one() || x.cmp(&n_minus_1) == Ordering::Equal {
            continue 'witness;
        }
        for _ in 0..s.saturating_sub(1) {
            x = mul_mod(&x, &x, n);
            if x.cmp(&n_minus_1) == Ordering::Equal {
                continue 'witness;
            }
        }
        return false; // composite
    }
    true
}

/// `n >> 1` (single-bit right shift).
fn shr_one(n: &BigUint) -> BigUint {
    let limbs = shr_limbs(&n.limbs, 1);
    let mut b = BigUint { limbs };
    b.normalize();
    b
}

/// Generate a random probable prime of exactly `bits` bits, with the top two
/// bits set (so a product of two such primes has the intended modulus size,
/// per FIPS 186-5 / OpenSSL convention) and the low bit set (odd). `rng`
/// supplies random bytes (CSPRNG). The Miller–Rabin witness count is 40.
pub fn random_prime(bits: usize, rng: &mut dyn FnMut(&mut [u8])) -> BigUint {
    assert!(bits >= 16, "RSA prime too small");
    let nbytes = bits.div_ceil(8);
    loop {
        let mut buf = vec![0u8; nbytes];
        rng(&mut buf);
        // Force exact bit length: set the top bit of the most-significant byte,
        // and the next bit, so p*q lands in the right modulus-size window.
        let top_bit_pos = bits - 1; // 0-indexed
        // Clear any bits above `bits`.
        let excess = nbytes * 8 - bits;
        if excess > 0 {
            buf[0] &= 0xFF >> excess;
        }
        let mut cand = BigUint::from_be_bytes(&buf);
        // Set the top two bits.
        cand = set_bit(&cand, top_bit_pos);
        if top_bit_pos >= 1 {
            cand = set_bit(&cand, top_bit_pos - 1);
        }
        // Set the low bit (odd).
        cand = set_bit(&cand, 0);
        if cand.bit_len() != bits {
            continue;
        }
        if is_probable_prime(&cand, 40, rng) {
            return cand;
        }
    }
}

/// Return a copy of `n` with bit `idx` set.
fn set_bit(n: &BigUint, idx: usize) -> BigUint {
    let li = idx / 64;
    let off = idx % 64;
    let mut limbs = n.limbs.clone();
    if limbs.len() <= li {
        limbs.resize(li + 1, 0);
    }
    limbs[li] |= 1u64 << off;
    let mut b = BigUint { limbs };
    b.normalize();
    b
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_bytes() {
        let b = BigUint::from_be_bytes(&[0xde, 0xad, 0xbe, 0xef]);
        assert_eq!(b.to_be_bytes(4), vec![0xde, 0xad, 0xbe, 0xef]);
        assert_eq!(b.bit_len(), 32);
    }

    #[test]
    fn add_and_sub_mod() {
        let n = BigUint::from_be_bytes(&[7]);
        let a = BigUint::from_be_bytes(&[3]);
        let b = BigUint::from_be_bytes(&[5]);
        let s = add_mod(&a, &b, &n); // 8 mod 7 = 1
        assert_eq!(s.to_be_bytes(1), vec![1]);
        let d = sub_mod(&a, &b, &n); // -2 mod 7 = 5
        assert_eq!(d.to_be_bytes(1), vec![5]);
    }

    #[test]
    fn small_pow_mod() {
        // 2^10 mod 1000 = 1024 mod 1000 = 24
        let base = BigUint::from_be_bytes(&[2]);
        let exp = BigUint::from_be_bytes(&[10]);
        let n = BigUint::from_be_bytes(&[3, 0xe8]);
        let r = pow_mod(&base, &exp, &n);
        assert_eq!(r.to_be_bytes(1), vec![24]);
    }

    #[test]
    fn fermat_little() {
        // For prime p=13: 5^12 mod 13 == 1.
        let base = BigUint::from_be_bytes(&[5]);
        let exp = BigUint::from_be_bytes(&[12]);
        let p = BigUint::from_be_bytes(&[13]);
        assert_eq!(pow_mod(&base, &exp, &p).to_be_bytes(1), vec![1]);
    }

    #[test]
    fn rsa_like_small() {
        // Tiny "RSA": n=3233 (=61*53), e=17, d=413.
        // Encrypt m=65: c = 65^17 mod 3233 = 2790.
        // Decrypt: 2790^413 mod 3233 = 65.
        let n = BigUint::from_be_bytes(&[0x0c, 0xa1]); // 3233
        let m = BigUint::from_be_bytes(&[65]);
        let e = BigUint::from_be_bytes(&[17]);
        let c = pow_mod(&m, &e, &n);
        assert_eq!(c.to_be_bytes(2), vec![0x0a, 0xe6]); // 2790
        let d = BigUint::from_be_bytes(&[0x01, 0x9d]); // 413
        let m2 = pow_mod(&c, &d, &n);
        assert_eq!(m2.to_be_bytes(1), vec![65]);
    }

    // --- Algorithm-D division: cross-check against an independent reference ---

    /// Naive O(bits²) bit-serial remainder — the old implementation, kept
    /// here purely as an independent oracle for the fast `divmod`/`rem`.
    fn rem_ref(a: &BigUint, n: &BigUint) -> BigUint {
        assert!(!n.is_zero());
        if a.cmp(n) == Ordering::Less {
            return a.clone();
        }
        let mut r = BigUint::zero();
        let bits = a.bit_len();
        for i in (0..bits).rev() {
            // r = (r << 1) | bit(a, i)
            let mut limbs = vec![0u64; r.limbs.len() + 1];
            let mut carry = 0u64;
            for (j, &lm) in r.limbs.iter().enumerate() {
                limbs[j] = (lm << 1) | carry;
                carry = lm >> 63;
            }
            if carry != 0 {
                limbs[r.limbs.len()] = carry;
            }
            let mut s = BigUint { limbs };
            s.normalize();
            if a.bit(i) {
                s = add(&s, &BigUint { limbs: vec![1] });
            }
            if s.cmp(n) != Ordering::Less {
                s = sub(&s, n);
            }
            r = s;
        }
        r
    }

    /// Deterministic xorshift64 so the randomized tests are reproducible.
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            x
        }
        fn big(&mut self, max_limbs: usize) -> BigUint {
            let nl = (self.next() as usize % max_limbs) + 1;
            let mut bytes = vec![0u8; nl * 8];
            for b in &mut bytes {
                *b = self.next() as u8;
            }
            BigUint::from_be_bytes(&bytes)
        }
    }

    #[test]
    fn divmod_identity_random() {
        // For random u, v (v != 0): q*v + r == u and r < v, and r matches
        // the independent bit-serial oracle.
        let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
        for _ in 0..4000 {
            let u = rng.big(8);
            let mut v = rng.big(5);
            if v.is_zero() {
                v = BigUint::from_be_bytes(&[1]);
            }
            let (q, r) = divmod(&u, &v);
            // r < v
            assert_eq!(r.cmp(&v), Ordering::Less, "remainder not reduced");
            // q*v + r == u
            let recon = add(&mul_full(&q, &v), &r);
            assert_eq!(recon.cmp(&u), Ordering::Equal, "q*v+r != u");
            // matches the oracle
            assert_eq!(rem(&u, &v).cmp(&rem_ref(&u, &v)), Ordering::Equal);
        }
    }

    #[test]
    fn mul_mod_matches_reference_random() {
        let mut rng = Rng(0xD1B5_4A32_D192_ED03);
        for _ in 0..4000 {
            let a = rng.big(5);
            let b = rng.big(5);
            let mut n = rng.big(5);
            if n.is_zero() {
                n = BigUint::from_be_bytes(&[1]);
            }
            // reference: (a mod n)*(b mod n) reduced via the bit-serial oracle
            let expect = {
                let am = rem_ref(&a, &n);
                let bm = rem_ref(&b, &n);
                rem_ref(&mul_full(&am, &bm), &n)
            };
            assert_eq!(mul_mod(&a, &b, &n).cmp(&expect), Ordering::Equal);
        }
    }

    #[test]
    fn divmod_edge_cases() {
        let one = BigUint::from_be_bytes(&[1]);
        let zero = BigUint::zero();
        // a < n  →  q=0, r=a
        let a = BigUint::from_be_bytes(&[0x12, 0x34]);
        let n = BigUint::from_be_bytes(&[0xFF, 0xFF]);
        let (q, r) = divmod(&a, &n);
        assert!(q.is_zero());
        assert_eq!(r.cmp(&a), Ordering::Equal);
        // a == n  →  q=1, r=0
        let (q, r) = divmod(&n, &n);
        assert_eq!(q.cmp(&one), Ordering::Equal);
        assert!(r.is_zero());
        // 0 / n  →  0, 0
        let (q, r) = divmod(&zero, &n);
        assert!(q.is_zero() && r.is_zero());
        // exact multiple: (n * k) % n == 0  for a multi-limb k
        let k = BigUint::from_be_bytes(&[0xAB, 0xCD, 0xEF, 0x01, 0x23]);
        let prod = mul_full(&n, &k);
        assert!(rem(&prod, &n).is_zero());
        // borrow / add-back stress: divisor whose top limb forces qhat correction
        let big = BigUint::from_be_bytes(&[
            0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x01,
        ]);
        let div = BigUint::from_be_bytes(&[0x80, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01]);
        let (q, r) = divmod(&big, &div);
        assert_eq!(add(&mul_full(&q, &div), &r).cmp(&big), Ordering::Equal);
        assert_eq!(r.cmp(&div), Ordering::Less);
    }

    #[test]
    fn inv_mod_full_small() {
        // 3 * 4 = 12 ≡ 1 mod 11  → 3^{-1} mod 11 == 4.
        let inv = inv_mod_full(&BigUint::from_u64(3), &BigUint::from_u64(11)).unwrap();
        assert_eq!(inv.cmp(&BigUint::from_u64(4)), Ordering::Equal);
        // 17^{-1} mod 3120 = 2753 (the classic RSA example d for e=17).
        let inv = inv_mod_full(&BigUint::from_u64(17), &BigUint::from_u64(3120)).unwrap();
        assert_eq!(inv.cmp(&BigUint::from_u64(2753)), Ordering::Equal);
        // Non-invertible: gcd(6, 9) = 3 ≠ 1.
        assert!(inv_mod_full(&BigUint::from_u64(6), &BigUint::from_u64(9)).is_none());
    }

    #[test]
    fn inv_mod_full_roundtrip_random() {
        // For random a and odd modulus m with gcd(a,m)=1: (a * a^{-1}) mod m == 1.
        let mut rng = Rng(0xABCD_1234_5678_9012);
        let mut checked = 0;
        for _ in 0..400 {
            let a = rng.big(3);
            let mut m = rng.big(3);
            if m.is_even() {
                m = add(&m, &BigUint::from_u64(1));
            }
            if m.cmp(&BigUint::from_u64(3)) == Ordering::Less {
                continue;
            }
            if let Some(inv) = inv_mod_full(&a, &m) {
                let prod = mul_mod(&a, &inv, &m);
                assert!(prod.is_one(), "a*a^-1 != 1 mod m");
                checked += 1;
            }
        }
        assert!(checked > 50, "too few invertible cases exercised");
    }

    #[test]
    fn miller_rabin_known_primes_and_composites() {
        let mut rng = Rng(0x1357_9BDF_2468_ACE0);
        let mut fill = |buf: &mut [u8]| {
            for b in buf.iter_mut() {
                *b = rng.next() as u8;
            }
        };
        // Known primes.
        for &p in &[97u64, 101, 7919, 104729, 1_000_003] {
            assert!(
                is_probable_prime(&BigUint::from_u64(p), 20, &mut fill),
                "{p} should be prime"
            );
        }
        // Known composites.
        for &c in &[1u64, 4, 9, 15, 91, 7917, 104730, 1_000_000] {
            assert!(
                !is_probable_prime(&BigUint::from_u64(c), 20, &mut fill),
                "{c} should be composite"
            );
        }
        // Carmichael number 561 = 3·11·17 must be caught (the reason we use MR
        // not Fermat).
        assert!(!is_probable_prime(&BigUint::from_u64(561), 20, &mut fill));
    }
}
