//! scrypt per RFC 7914 — Colin Percival's memory-hard password KDF.
//! Used by Bitcoin/Litecoin/Dogecoin proof-of-work variants, Ethereum
//! wallet keystore files, Cisco PIX/ASA password storage, Tarsnap and
//! a handful of modern auth flows.
//!
//! Construction:
//!     T1 = PBKDF2-HMAC-SHA256(password, salt, 1, p * 128 * r)
//!     for i in 0..p:  B[i] = scryptROMix(r, T1[i*128*r..(i+1)*128*r], N)
//!     DK = PBKDF2-HMAC-SHA256(password, B, 1, dkLen)
//!
//! ROMix uses Salsa20/8 core to derive cheaply but require N * 128 * r
//! bytes of memory to evaluate, making it costly to parallelise on
//! ASICs / GPUs.

/// Derive `dk_len` bytes via scrypt with cost params (N, r, p).
///
/// `n` must be a power of two ≥ 2. `r` and `p` are typically small
/// (8 and 1 in the Litecoin profile). Memory usage is roughly
/// `128 * n * r * p` bytes.
pub fn scrypt(password: &[u8], salt: &[u8], n: u32, r: u32, p: u32, dk_len: usize) -> Vec<u8> {
    assert!(n >= 2 && (n & (n - 1)) == 0, "N must be a power of two ≥ 2");
    let r = r as usize;
    let p = p as usize;
    let n = n as usize;
    let block_size = 128 * r;

    // Step 1: B = PBKDF2-HMAC-SHA256(password, salt, 1, p * 128 * r)
    let mut b = crate::pbkdf2::pbkdf2_sha256(password, salt, 1, p * block_size);

    // Step 2: for i in 0..p: B[i] = scryptROMix(r, B[i], N)
    for i in 0..p {
        let start = i * block_size;
        let end = start + block_size;
        scrypt_romix(&mut b[start..end], n, r);
    }

    // Step 3: DK = PBKDF2-HMAC-SHA256(password, B, 1, dkLen)
    crate::pbkdf2::pbkdf2_sha256(password, &b, 1, dk_len)
}

/// scryptROMix per RFC 7914 §4 — fills a 2r-block scratchpad then
/// pseudo-randomly mixes back through it.
fn scrypt_romix(block: &mut [u8], n: usize, r: usize) {
    let block_size = 128 * r;
    debug_assert_eq!(block.len(), block_size);
    let mut x = block.to_vec();
    let mut v: Vec<Vec<u8>> = Vec::with_capacity(n);
    for _ in 0..n {
        v.push(x.clone());
        scrypt_block_mix(&mut x, r);
    }
    for _ in 0..n {
        // integerify — the rightmost 64 bits of x interpreted as
        // little-endian, modulo N, picks the V-block to XOR in.
        let j = integerify(&x, r) & (n as u64 - 1);
        let vj = &v[j as usize];
        for i in 0..block_size {
            x[i] ^= vj[i];
        }
        scrypt_block_mix(&mut x, r);
    }
    block.copy_from_slice(&x);
}

/// scryptBlockMix per RFC 7914 §3 — apply Salsa20/8 over chained
/// 64-byte sub-blocks.
fn scrypt_block_mix(b: &mut [u8], r: usize) {
    // X = B[2*r - 1] (last 64-byte block)
    let mut x = [0u8; 64];
    x.copy_from_slice(&b[(2 * r - 1) * 64..2 * r * 64]);
    let mut y = vec![0u8; 128 * r];
    for i in 0..2 * r {
        let mut t = [0u8; 64];
        for j in 0..64 {
            t[j] = x[j] ^ b[i * 64 + j];
        }
        salsa20_8(&mut t);
        x = t;
        // Place X into Y[i] in the interleaved order: even-i first half,
        // odd-i second half (the BlockMix permutation).
        let dst_index = if i % 2 == 0 { i / 2 } else { r + i / 2 };
        y[dst_index * 64..(dst_index + 1) * 64].copy_from_slice(&x);
    }
    b.copy_from_slice(&y);
}

/// Salsa20/8 core: 8 double-round mixings on a 64-byte state. Smaller
/// rounds than full Salsa20/20, chosen for scrypt's perf profile.
fn salsa20_8(block: &mut [u8; 64]) {
    let mut x = [0u32; 16];
    for i in 0..16 {
        x[i] = u32::from_le_bytes([
            block[i * 4],
            block[i * 4 + 1],
            block[i * 4 + 2],
            block[i * 4 + 3],
        ]);
    }
    let initial = x;
    for _ in 0..4 {
        x[4] ^= x[0].wrapping_add(x[12]).rotate_left(7);
        x[8] ^= x[4].wrapping_add(x[0]).rotate_left(9);
        x[12] ^= x[8].wrapping_add(x[4]).rotate_left(13);
        x[0] ^= x[12].wrapping_add(x[8]).rotate_left(18);
        x[9] ^= x[5].wrapping_add(x[1]).rotate_left(7);
        x[13] ^= x[9].wrapping_add(x[5]).rotate_left(9);
        x[1] ^= x[13].wrapping_add(x[9]).rotate_left(13);
        x[5] ^= x[1].wrapping_add(x[13]).rotate_left(18);
        x[14] ^= x[10].wrapping_add(x[6]).rotate_left(7);
        x[2] ^= x[14].wrapping_add(x[10]).rotate_left(9);
        x[6] ^= x[2].wrapping_add(x[14]).rotate_left(13);
        x[10] ^= x[6].wrapping_add(x[2]).rotate_left(18);
        x[3] ^= x[15].wrapping_add(x[11]).rotate_left(7);
        x[7] ^= x[3].wrapping_add(x[15]).rotate_left(9);
        x[11] ^= x[7].wrapping_add(x[3]).rotate_left(13);
        x[15] ^= x[11].wrapping_add(x[7]).rotate_left(18);
        x[1] ^= x[0].wrapping_add(x[3]).rotate_left(7);
        x[2] ^= x[1].wrapping_add(x[0]).rotate_left(9);
        x[3] ^= x[2].wrapping_add(x[1]).rotate_left(13);
        x[0] ^= x[3].wrapping_add(x[2]).rotate_left(18);
        x[6] ^= x[5].wrapping_add(x[4]).rotate_left(7);
        x[7] ^= x[6].wrapping_add(x[5]).rotate_left(9);
        x[4] ^= x[7].wrapping_add(x[6]).rotate_left(13);
        x[5] ^= x[4].wrapping_add(x[7]).rotate_left(18);
        x[11] ^= x[10].wrapping_add(x[9]).rotate_left(7);
        x[8] ^= x[11].wrapping_add(x[10]).rotate_left(9);
        x[9] ^= x[8].wrapping_add(x[11]).rotate_left(13);
        x[10] ^= x[9].wrapping_add(x[8]).rotate_left(18);
        x[12] ^= x[15].wrapping_add(x[14]).rotate_left(7);
        x[13] ^= x[12].wrapping_add(x[15]).rotate_left(9);
        x[14] ^= x[13].wrapping_add(x[12]).rotate_left(13);
        x[15] ^= x[14].wrapping_add(x[13]).rotate_left(18);
    }
    for i in 0..16 {
        x[i] = x[i].wrapping_add(initial[i]);
        block[i * 4..i * 4 + 4].copy_from_slice(&x[i].to_le_bytes());
    }
}

fn integerify(block: &[u8], r: usize) -> u64 {
    // Take the LAST 64-byte sub-block; the leading 8 bytes interpreted
    // little-endian are the integer.
    let last = (2 * r - 1) * 64;
    u64::from_le_bytes(block[last..last + 8].try_into().unwrap())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hex(b: &[u8]) -> String {
        let mut s = String::with_capacity(b.len() * 2);
        for byte in b {
            s.push_str(&format!("{:02x}", byte));
        }
        s
    }

    // RFC 7914 §11 test vector 1: N=16, r=1, p=1, dkLen=64
    #[test]
    fn rfc7914_v1() {
        let dk = scrypt(b"", b"", 16, 1, 1, 64);
        assert_eq!(
            hex(&dk),
            "77d6576238657b203b19ca42c18a0497f16b4844e3074ae8dfdffa3fede21442fcd0069ded0948f8326a753a0fc81f17e8d3e0fb2e0d3628cf35e20c38d18906"
        );
    }
}
