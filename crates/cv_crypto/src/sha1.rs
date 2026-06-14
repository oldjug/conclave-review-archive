//! SHA-1 per FIPS 180-4 §6.1. 160-bit digest, 64-byte block.
//!
//! Deprecated for new protocols, but Web Crypto still must expose it
//! (`crypto.subtle.digest("SHA-1", ...)`), WebSocket handshake uses
//! it (`Sec-WebSocket-Accept`), git uses it, and a large surface of
//! legacy auth code (CRAM-MD5 / HMAC-SHA1 / TOTP) depends on it.

#[derive(Clone)]
pub struct Sha1 {
    h: [u32; 5],
    buf: [u8; 64],
    buf_len: usize,
    msg_len_bits: u64,
}

impl Default for Sha1 {
    fn default() -> Self {
        Self::new()
    }
}

impl Sha1 {
    pub fn new() -> Self {
        Self {
            h: [0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476, 0xC3D2E1F0],
            buf: [0u8; 64],
            buf_len: 0,
            msg_len_bits: 0,
        }
    }

    pub fn update(&mut self, data: &[u8]) {
        self.msg_len_bits = self.msg_len_bits.wrapping_add((data.len() as u64) * 8);
        let mut i = 0;
        if self.buf_len > 0 {
            let need = 64 - self.buf_len;
            let take = need.min(data.len());
            self.buf[self.buf_len..self.buf_len + take].copy_from_slice(&data[..take]);
            self.buf_len += take;
            i += take;
            if self.buf_len == 64 {
                let block = self.buf;
                self.compress(&block);
                self.buf_len = 0;
            }
        }
        while i + 64 <= data.len() {
            let mut block = [0u8; 64];
            block.copy_from_slice(&data[i..i + 64]);
            self.compress(&block);
            i += 64;
        }
        let rem = data.len() - i;
        if rem > 0 {
            self.buf[..rem].copy_from_slice(&data[i..]);
            self.buf_len = rem;
        }
    }

    pub fn finalize(mut self) -> [u8; 20] {
        // Padding: 0x80, then zeros, then 8-byte big-endian length.
        let mut pad = [0u8; 128];
        pad[0] = 0x80;
        let len_be = self.msg_len_bits.to_be_bytes();
        // Need total padded len % 64 == 56.
        let cur = self.buf_len;
        let pad_len = if cur < 56 { 56 - cur } else { 120 - cur };
        self.update(&pad[..pad_len]);
        self.update(&len_be);
        debug_assert_eq!(self.buf_len, 0);
        let mut out = [0u8; 20];
        for (i, w) in self.h.iter().enumerate() {
            out[i * 4..i * 4 + 4].copy_from_slice(&w.to_be_bytes());
        }
        out
    }

    pub fn oneshot(data: &[u8]) -> [u8; 20] {
        let mut h = Self::new();
        h.update(data);
        h.finalize()
    }

    fn compress(&mut self, block: &[u8; 64]) {
        let mut w = [0u32; 80];
        for i in 0..16 {
            w[i] = u32::from_be_bytes([
                block[i * 4],
                block[i * 4 + 1],
                block[i * 4 + 2],
                block[i * 4 + 3],
            ]);
        }
        for i in 16..80 {
            w[i] = (w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16]).rotate_left(1);
        }
        let mut a = self.h[0];
        let mut b = self.h[1];
        let mut c = self.h[2];
        let mut d = self.h[3];
        let mut e = self.h[4];
        for i in 0..80 {
            let (f, k) = match i {
                0..=19 => ((b & c) | ((!b) & d), 0x5A827999u32),
                20..=39 => (b ^ c ^ d, 0x6ED9EBA1u32),
                40..=59 => ((b & c) | (b & d) | (c & d), 0x8F1BBCDCu32),
                _ => (b ^ c ^ d, 0xCA62C1D6u32),
            };
            let temp = a
                .rotate_left(5)
                .wrapping_add(f)
                .wrapping_add(e)
                .wrapping_add(k)
                .wrapping_add(w[i]);
            e = d;
            d = c;
            c = b.rotate_left(30);
            b = a;
            a = temp;
        }
        self.h[0] = self.h[0].wrapping_add(a);
        self.h[1] = self.h[1].wrapping_add(b);
        self.h[2] = self.h[2].wrapping_add(c);
        self.h[3] = self.h[3].wrapping_add(d);
        self.h[4] = self.h[4].wrapping_add(e);
    }
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

    // FIPS 180-2 / NIST test vectors.
    #[test]
    fn empty() {
        assert_eq!(
            hex(&Sha1::oneshot(b"")),
            "da39a3ee5e6b4b0d3255bfef95601890afd80709"
        );
    }

    #[test]
    fn abc() {
        assert_eq!(
            hex(&Sha1::oneshot(b"abc")),
            "a9993e364706816aba3e25717850c26c9cd0d89d"
        );
    }

    #[test]
    fn quickbrown() {
        assert_eq!(
            hex(&Sha1::oneshot(
                b"The quick brown fox jumps over the lazy dog"
            )),
            "2fd4e1c67a2d28fced849ee1bb76e7391b93eb12"
        );
    }

    #[test]
    fn long_alphabet() {
        let msg = b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq";
        assert_eq!(
            hex(&Sha1::oneshot(msg)),
            "84983e441c3bd26ebaae4aa1f95129e5e54670f1"
        );
    }

    #[test]
    fn million_a() {
        let mut h = Sha1::new();
        for _ in 0..1_000_000 {
            h.update(b"a");
        }
        assert_eq!(
            hex(&h.finalize()),
            "34aa973cd4c4daa4f61eeb2bdbad27316534016f"
        );
    }
}
