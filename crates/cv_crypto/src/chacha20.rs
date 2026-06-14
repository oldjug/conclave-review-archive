//! ChaCha20 stream cipher per RFC 8439 §2.
//!
//! 256-bit key, 96-bit nonce, 32-bit counter. Block function produces 64
//! bytes of keystream per call.

pub const KEY_SIZE: usize = 32;
pub const NONCE_SIZE: usize = 12;
pub const BLOCK_SIZE: usize = 64;

#[derive(Debug)]
pub struct ChaCha20 {
    state: [u32; 16],
}

impl ChaCha20 {
    /// `counter` is the *initial* 32-bit block counter. RFC 8439 §2.4 uses
    /// 1 for the payload of an AEAD; Poly1305 key derivation uses 0.
    pub fn new(key: &[u8; KEY_SIZE], nonce: &[u8; NONCE_SIZE], counter: u32) -> Self {
        // Constants "expand 32-byte k" per RFC 8439 §2.3.
        let mut state = [0u32; 16];
        state[0] = 0x6170_7865;
        state[1] = 0x3320_646e;
        state[2] = 0x7962_2d32;
        state[3] = 0x6b20_6574;
        for i in 0..8 {
            state[4 + i] = u32::from_le_bytes(key[i * 4..i * 4 + 4].try_into().unwrap());
        }
        state[12] = counter;
        for i in 0..3 {
            state[13 + i] = u32::from_le_bytes(nonce[i * 4..i * 4 + 4].try_into().unwrap());
        }
        Self { state }
    }

    pub fn block(&self) -> [u8; BLOCK_SIZE] {
        let mut s = self.state;
        for _ in 0..10 {
            // Column rounds
            quarter(&mut s, 0, 4, 8, 12);
            quarter(&mut s, 1, 5, 9, 13);
            quarter(&mut s, 2, 6, 10, 14);
            quarter(&mut s, 3, 7, 11, 15);
            // Diagonal rounds
            quarter(&mut s, 0, 5, 10, 15);
            quarter(&mut s, 1, 6, 11, 12);
            quarter(&mut s, 2, 7, 8, 13);
            quarter(&mut s, 3, 4, 9, 14);
        }
        for i in 0..16 {
            s[i] = s[i].wrapping_add(self.state[i]);
        }
        let mut out = [0u8; BLOCK_SIZE];
        for i in 0..16 {
            out[i * 4..i * 4 + 4].copy_from_slice(&s[i].to_le_bytes());
        }
        out
    }

    pub fn next_block(&mut self) -> [u8; BLOCK_SIZE] {
        let b = self.block();
        self.state[12] = self.state[12].wrapping_add(1);
        b
    }

    /// In-place XOR encryption/decryption (symmetric for a stream cipher).
    pub fn xor(&mut self, data: &mut [u8]) {
        let mut offset = 0;
        while offset < data.len() {
            let ks = self.next_block();
            let take = (data.len() - offset).min(BLOCK_SIZE);
            for j in 0..take {
                data[offset + j] ^= ks[j];
            }
            offset += take;
        }
    }
}

fn quarter(s: &mut [u32; 16], a: usize, b: usize, c: usize, d: usize) {
    s[a] = s[a].wrapping_add(s[b]);
    s[d] = (s[d] ^ s[a]).rotate_left(16);
    s[c] = s[c].wrapping_add(s[d]);
    s[b] = (s[b] ^ s[c]).rotate_left(12);
    s[a] = s[a].wrapping_add(s[b]);
    s[d] = (s[d] ^ s[a]).rotate_left(8);
    s[c] = s[c].wrapping_add(s[d]);
    s[b] = (s[b] ^ s[c]).rotate_left(7);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hex(b: &[u8]) -> String {
        let mut s = String::with_capacity(b.len() * 2);
        for x in b {
            s.push_str(&format!("{x:02x}"));
        }
        s
    }

    /// RFC 8439 §2.3.2 block-function test vector.
    #[test]
    fn rfc8439_block() {
        let key: [u8; 32] = [
            0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d,
            0x0e, 0x0f, 0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x1b,
            0x1c, 0x1d, 0x1e, 0x1f,
        ];
        let nonce: [u8; 12] = [
            0x00, 0x00, 0x00, 0x09, 0x00, 0x00, 0x00, 0x4a, 0x00, 0x00, 0x00, 0x00,
        ];
        let c = ChaCha20::new(&key, &nonce, 1);
        let blk = c.block();
        assert_eq!(
            hex(&blk),
            "10f1e7e4d13b5915500fdd1fa32071c4c7d1f4c733c068030422aa9ac3d46c4e\
             d2826446079faa0914c2d705d98b02a2b5129cd1de164eb9cbd083e8a2503c4e"
        );
    }

    /// RFC 8439 §2.4.2 encryption test vector.
    #[test]
    fn rfc8439_encrypt() {
        let key: [u8; 32] = [
            0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d,
            0x0e, 0x0f, 0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x1b,
            0x1c, 0x1d, 0x1e, 0x1f,
        ];
        let nonce: [u8; 12] = [
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x4a, 0x00, 0x00, 0x00, 0x00,
        ];
        let plaintext =
            b"Ladies and Gentlemen of the class of '99: If I could offer you only one tip for the future, sunscreen would be it.";
        let mut buf = plaintext.to_vec();
        let mut c = ChaCha20::new(&key, &nonce, 1);
        c.xor(&mut buf);
        // First 16 bytes from the RFC.
        assert_eq!(hex(&buf[..16]), "6e2e359a2568f98041ba0728dd0d6981");
        // Decrypt with a fresh instance produces the original.
        let mut c2 = ChaCha20::new(&key, &nonce, 1);
        c2.xor(&mut buf);
        assert_eq!(&buf, plaintext);
    }
}
