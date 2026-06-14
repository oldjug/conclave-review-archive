//! Keccak / SHA-3 per NIST FIPS 202. We expose:
//!
//! * `Keccak256` — Ethereum's address hash (legacy "keccak", NOT
//!   FIPS-202 SHA3-256; it predates the spec change and uses the
//!   original Keccak padding `0x01`).
//! * `Sha3_256` — FIPS 202 SHA3-256 (padding byte `0x06`).
//! * `Sha3_512` — FIPS 202 SHA3-512 (padding byte `0x06`).
//!
//! Web Crypto doesn't ship SHA-3 directly but many JS wallet
//! libraries probe for it via `crypto.subtle.digest("SHA3-256", ...)`.

const RHO: [u32; 24] = [
    1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 2, 14, 27, 41, 56, 8, 25, 43, 62, 18, 39, 61, 20, 44,
];
const PI: [usize; 24] = [
    10, 7, 11, 17, 18, 3, 5, 16, 8, 21, 24, 4, 15, 23, 19, 13, 12, 2, 20, 14, 22, 9, 6, 1,
];
const RC: [u64; 24] = [
    0x0000000000000001,
    0x0000000000008082,
    0x800000000000808a,
    0x8000000080008000,
    0x000000000000808b,
    0x0000000080000001,
    0x8000000080008081,
    0x8000000000008009,
    0x000000000000008a,
    0x0000000000000088,
    0x0000000080008009,
    0x000000008000000a,
    0x000000008000808b,
    0x800000000000008b,
    0x8000000000008089,
    0x8000000000008003,
    0x8000000000008002,
    0x8000000000000080,
    0x000000000000800a,
    0x800000008000000a,
    0x8000000080008081,
    0x8000000000008080,
    0x0000000080000001,
    0x8000000080008008,
];

fn keccak_f(state: &mut [u64; 25]) {
    for round in 0..24 {
        // Theta
        let mut c = [0u64; 5];
        for x in 0..5 {
            c[x] = state[x] ^ state[x + 5] ^ state[x + 10] ^ state[x + 15] ^ state[x + 20];
        }
        let mut d = [0u64; 5];
        for x in 0..5 {
            d[x] = c[(x + 4) % 5] ^ c[(x + 1) % 5].rotate_left(1);
        }
        for x in 0..5 {
            for y in 0..5 {
                state[x + 5 * y] ^= d[x];
            }
        }
        // Rho and Pi
        let mut t = state[1];
        for i in 0..24 {
            let j = PI[i];
            let tmp = state[j];
            state[j] = t.rotate_left(RHO[i]);
            t = tmp;
        }
        // Chi
        for y in 0..5 {
            let mut row = [0u64; 5];
            for x in 0..5 {
                row[x] = state[x + 5 * y];
            }
            for x in 0..5 {
                state[x + 5 * y] = row[x] ^ ((!row[(x + 1) % 5]) & row[(x + 2) % 5]);
            }
        }
        // Iota
        state[0] ^= RC[round];
    }
}

fn keccak<const OUT: usize>(data: &[u8], rate: usize, pad_byte: u8) -> [u8; OUT] {
    let mut state = [0u64; 25];
    let mut buf = vec![0u8; rate];
    let mut filled = 0usize;
    for &byte in data {
        buf[filled] = byte;
        filled += 1;
        if filled == rate {
            for i in 0..(rate / 8) {
                let mut w = 0u64;
                for j in 0..8 {
                    w |= (buf[i * 8 + j] as u64) << (8 * j);
                }
                state[i] ^= w;
            }
            keccak_f(&mut state);
            filled = 0;
        }
    }
    // Padding: append pad_byte (`0x01` for legacy keccak, `0x06` for
    // FIPS SHA-3), then zero-fill, then set the high bit.
    for i in filled..rate {
        buf[i] = 0;
    }
    buf[filled] = pad_byte;
    buf[rate - 1] |= 0x80;
    for i in 0..(rate / 8) {
        let mut w = 0u64;
        for j in 0..8 {
            w |= (buf[i * 8 + j] as u64) << (8 * j);
        }
        state[i] ^= w;
    }
    keccak_f(&mut state);
    let mut out = [0u8; OUT];
    for i in 0..OUT {
        out[i] = ((state[i / 8] >> (8 * (i % 8))) & 0xff) as u8;
    }
    out
}

/// Ethereum's address hash — original Keccak with `0x01` padding.
pub fn keccak256(data: &[u8]) -> [u8; 32] {
    keccak::<32>(data, 136, 0x01)
}

/// FIPS 202 SHA3-256.
pub fn sha3_256(data: &[u8]) -> [u8; 32] {
    keccak::<32>(data, 136, 0x06)
}

/// FIPS 202 SHA3-512.
pub fn sha3_512(data: &[u8]) -> [u8; 64] {
    keccak::<64>(data, 72, 0x06)
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

    // FIPS 202 test vectors (empty input).
    #[test]
    fn sha3_256_empty() {
        assert_eq!(
            hex(&sha3_256(b"")),
            "a7ffc6f8bf1ed76651c14756a061d662f580ff4de43b49fa82d80a4b80f8434a"
        );
    }

    #[test]
    fn sha3_512_empty() {
        assert_eq!(
            hex(&sha3_512(b"")),
            "a69f73cca23a9ac5c8b567dc185a756e97c982164fe25859e0d1dcc1475c80a615b2123af1f5f94c11e3e9402c3ac558f500199d95b6d3e301758586281dcd26"
        );
    }

    // Ethereum keccak256 — official test vector ("" → empty input).
    // Source: Ethereum yellow paper appendix and many wallet test suites.
    #[test]
    fn keccak256_empty() {
        assert_eq!(
            hex(&keccak256(b"")),
            "c5d2460186f7233c927e7db2dcc703c0e500b653ca82273b7bfad8045d85a470"
        );
    }
}
