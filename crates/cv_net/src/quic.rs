//! QUIC v1 (RFC 9000) — varint + packet header parsing.
//!
//! Lays the foundation for HTTP/3 (RFC 9114). V1 here implements:
//!   * Variable-length integers (§16) — 1/2/4/8-byte encoding.
//!   * Long-header parsing (§17.2) — initial / handshake / 0-RTT.
//!   * Short-header parsing (§17.3) — 1-RTT data frames.
//!
//! Crypto, packet protection, congestion control, and the full
//! frame catalog are sequenced behind this in subsequent slices.

/// Encode a u64 as a QUIC variable-length integer. Returns the
/// minimum-length encoding (RFC 9000 §16).
pub fn encode_varint(v: u64) -> Vec<u8> {
    if v < (1 << 6) {
        vec![v as u8]
    } else if v < (1 << 14) {
        let mut out = (v as u16).to_be_bytes().to_vec();
        out[0] |= 0x40;
        out
    } else if v < (1 << 30) {
        let mut out = (v as u32).to_be_bytes().to_vec();
        out[0] |= 0x80;
        out
    } else if v < (1 << 62) {
        let mut out = v.to_be_bytes().to_vec();
        out[0] |= 0xC0;
        out
    } else {
        panic!("varint overflow: {v}");
    }
}

/// Decode one varint from `buf`. Returns `(value, length)` on success.
pub fn decode_varint(buf: &[u8]) -> Option<(u64, usize)> {
    if buf.is_empty() {
        return None;
    }
    let two_bits = buf[0] >> 6;
    let len = 1usize << two_bits;
    if buf.len() < len {
        return None;
    }
    let mut bytes = buf[..len].to_vec();
    bytes[0] &= 0x3F;
    let mut padded = [0u8; 8];
    padded[8 - len..].copy_from_slice(&bytes);
    Some((u64::from_be_bytes(padded), len))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LongPacketType {
    Initial = 0,
    ZeroRtt = 1,
    Handshake = 2,
    Retry = 3,
}

impl LongPacketType {
    pub fn from_bits(b: u8) -> Self {
        match b & 0x03 {
            0 => Self::Initial,
            1 => Self::ZeroRtt,
            2 => Self::Handshake,
            3 => Self::Retry,
            _ => unreachable!(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LongHeader {
    pub packet_type: LongPacketType,
    pub version: u32,
    pub dcid: Vec<u8>,
    pub scid: Vec<u8>,
    /// Byte index in the input buffer where the payload starts.
    pub payload_offset: usize,
}

/// Parse a QUIC long header. Returns the parsed metadata plus the
/// offset into `buf` where the type-specific payload (token, packet
/// number, frames) starts.
pub fn parse_long_header(buf: &[u8]) -> Option<LongHeader> {
    if buf.len() < 7 {
        return None;
    }
    let first = buf[0];
    if (first & 0x80) == 0 {
        return None; // short header
    }
    let packet_type = LongPacketType::from_bits(first >> 4);
    let version = u32::from_be_bytes(buf[1..5].try_into().unwrap());
    let dcid_len = buf[5] as usize;
    if 6 + dcid_len + 1 > buf.len() {
        return None;
    }
    let dcid = buf[6..6 + dcid_len].to_vec();
    let scid_off = 6 + dcid_len;
    let scid_len = buf[scid_off] as usize;
    if scid_off + 1 + scid_len > buf.len() {
        return None;
    }
    let scid = buf[scid_off + 1..scid_off + 1 + scid_len].to_vec();
    Some(LongHeader {
        packet_type,
        version,
        dcid,
        scid,
        payload_offset: scid_off + 1 + scid_len,
    })
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShortHeader {
    /// Spin bit (latency measurement, §17.4).
    pub spin: bool,
    /// Key phase bit (header protection / key updates, §6).
    pub key_phase: bool,
    pub dcid: Vec<u8>,
}

pub fn parse_short_header(buf: &[u8], dcid_len: usize) -> Option<ShortHeader> {
    if buf.len() < 1 + dcid_len {
        return None;
    }
    let first = buf[0];
    if (first & 0x80) != 0 {
        return None; // long header
    }
    Some(ShortHeader {
        spin: (first & 0x20) != 0,
        key_phase: (first & 0x04) != 0,
        dcid: buf[1..1 + dcid_len].to_vec(),
    })
}

// --------------- QUIC initial-packet protection (RFC 9001) ---------------
//
// AES-128-GCM authenticated encryption for Initial packets.  The
// nonce is the packet number XORed into the lower bytes of the IV.
// Real HKDF-Expand-Label derivation uses cv_crypto::hkdf; here we
// expose helpers that produce (key, iv, hp) tuples per RFC 9001 §5.2.

/// Initial salt (RFC 9001 §5.2): static; both endpoints derive
/// their initial keys from `HKDF-Extract(initial_salt, client_dcid)`.
pub const INITIAL_SALT_V1: [u8; 20] = [
    0x38, 0x76, 0x2c, 0xf7, 0xf5, 0x59, 0x34, 0xb3, 0x4d, 0x17, 0x9a, 0xe6, 0xa4, 0xc8, 0x0c, 0xad,
    0xcc, 0xbb, 0x7f, 0x0a,
];

/// HKDF-Expand-Label (RFC 8446 §7.1) over SHA-256 — the KDF QUIC v1
/// Initial keys use (RFC 9001 §5.2). `label` is prefixed with `tls13 `.
pub fn hkdf_expand_label(secret: &[u8], label: &[u8], context: &[u8], out_len: usize) -> Vec<u8> {
    let mut info = Vec::with_capacity(4 + 6 + label.len() + context.len());
    info.extend_from_slice(&(out_len as u16).to_be_bytes());
    let full_label_len = 6 + label.len();
    info.push(full_label_len as u8);
    info.extend_from_slice(b"tls13 ");
    info.extend_from_slice(label);
    info.push(context.len() as u8);
    info.extend_from_slice(context);
    let mut out = vec![0u8; out_len];
    cv_crypto::hkdf::expand(secret, &info, &mut out);
    out
}

/// QUIC v1 Initial keys for one direction (RFC 9001 §5.2).
#[derive(Debug, Clone)]
pub struct InitialKeys {
    /// AES-128-GCM key ("quic key").
    pub key: [u8; 16],
    /// AEAD IV ("quic iv").
    pub iv: [u8; 12],
    /// Header-protection key ("quic hp").
    pub hp: [u8; 16],
}

/// Derive the client's Initial secret + keys from the client-chosen
/// Destination Connection ID, per RFC 9001 §5.2:
///
///   initial_secret = HKDF-Extract(initial_salt, client_dst_connection_id)
///   client_initial_secret = HKDF-Expand-Label(initial_secret, "client in", "", 32)
///   key = HKDF-Expand-Label(client_initial_secret, "quic key", "", 16)
///   iv  = HKDF-Expand-Label(client_initial_secret, "quic iv",  "", 12)
///   hp  = HKDF-Expand-Label(client_initial_secret, "quic hp",  "", 16)
pub fn derive_client_initial_keys(dcid: &[u8]) -> InitialKeys {
    let initial_secret = cv_crypto::hkdf::extract(&INITIAL_SALT_V1, dcid);
    let client_secret = hkdf_expand_label(&initial_secret, b"client in", b"", 32);
    let key_v = hkdf_expand_label(&client_secret, b"quic key", b"", 16);
    let iv_v = hkdf_expand_label(&client_secret, b"quic iv", b"", 12);
    let hp_v = hkdf_expand_label(&client_secret, b"quic hp", b"", 16);
    let mut key = [0u8; 16];
    let mut iv = [0u8; 12];
    let mut hp = [0u8; 16];
    key.copy_from_slice(&key_v);
    iv.copy_from_slice(&iv_v);
    hp.copy_from_slice(&hp_v);
    InitialKeys { key, iv, hp }
}

/// QUIC v1 version number (RFC 9000 §15).
pub const QUIC_VERSION_1: u32 = 0x0000_0001;

/// Build the AES-128-GCM nonce for a QUIC packet.  Per RFC 9001
/// §5.3, the nonce is the IV XORed with the packet number expanded
/// to a 12-byte big-endian field (right-aligned).
pub fn build_nonce(iv: &[u8; 12], packet_number: u64) -> [u8; 12] {
    let mut nonce = *iv;
    for i in 0..8 {
        nonce[11 - i] ^= ((packet_number >> (8 * i)) & 0xFF) as u8;
    }
    nonce
}

/// Encrypt a QUIC Initial packet payload.  `key` is the
/// 16-byte AES-128 key from HKDF-Expand-Label("quic key"),
/// `iv` is the 12-byte IV from HKDF-Expand-Label("quic iv"),
/// `aad` is the unprotected packet header, `payload` is the cleartext
/// frame data.  Returns (ciphertext || 16-byte auth tag).
pub fn encrypt_payload(
    key: &[u8; 16],
    iv: &[u8; 12],
    packet_number: u64,
    aad: &[u8],
    payload: &[u8],
) -> Vec<u8> {
    let nonce = build_nonce(iv, packet_number);
    cv_crypto::aes_gcm::Aes128Gcm::seal(key, &nonce, aad, payload)
}

/// Decrypt a QUIC Initial packet.  Verifies the AEAD tag; returns
/// None on failure.
pub fn decrypt_payload(
    key: &[u8; 16],
    iv: &[u8; 12],
    packet_number: u64,
    aad: &[u8],
    ciphertext: &[u8],
) -> Option<Vec<u8>> {
    let nonce = build_nonce(iv, packet_number);
    cv_crypto::aes_gcm::Aes128Gcm::open(key, &nonce, aad, ciphertext).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn varint_1byte_encoding() {
        assert_eq!(encode_varint(37), vec![37]);
        let (v, len) = decode_varint(&[37]).unwrap();
        assert_eq!(v, 37);
        assert_eq!(len, 1);
    }

    #[test]
    fn varint_2byte_encoding() {
        // 15293 = RFC 9000 §A.1 worked example.
        let enc = encode_varint(15293);
        assert_eq!(enc, vec![0x7B, 0xBD]);
        let (v, len) = decode_varint(&enc).unwrap();
        assert_eq!(v, 15293);
        assert_eq!(len, 2);
    }

    #[test]
    fn varint_4byte_encoding() {
        let enc = encode_varint(494878333);
        assert_eq!(enc[0] >> 6, 0b10);
        let (v, _) = decode_varint(&enc).unwrap();
        assert_eq!(v, 494878333);
    }

    #[test]
    fn varint_8byte_encoding() {
        let enc = encode_varint(151288809941952652);
        assert_eq!(enc[0] >> 6, 0b11);
        let (v, _) = decode_varint(&enc).unwrap();
        assert_eq!(v, 151288809941952652);
    }

    #[test]
    fn long_header_initial_packet_parsed() {
        // first byte 0xC0 = long, type Initial.
        // version 0x00000001, dcid_len 4, scid_len 0.
        let mut buf = vec![
            0xC0, 0x00, 0x00, 0x00, 0x01, 4, 0xDE, 0xAD, 0xBE, 0xEF, 0, 0xAA, 0xBB,
        ];
        let h = parse_long_header(&mut buf).unwrap();
        assert_eq!(h.packet_type, LongPacketType::Initial);
        assert_eq!(h.version, 1);
        assert_eq!(h.dcid, vec![0xDE, 0xAD, 0xBE, 0xEF]);
        assert!(h.scid.is_empty());
        assert_eq!(h.payload_offset, 11);
    }

    #[test]
    fn short_header_parses_spin_and_key_phase() {
        // 0b01100100 = fixed bit + spin + reserved(0) + key_phase + packet num len 00
        let buf = [0b0110_0100, 1, 2, 3, 4];
        let h = parse_short_header(&buf, 4).unwrap();
        assert!(h.spin);
        assert!(h.key_phase);
        assert_eq!(h.dcid, vec![1, 2, 3, 4]);
    }

    #[test]
    fn initial_salt_v1_matches_rfc_9001() {
        assert_eq!(INITIAL_SALT_V1[0], 0x38);
        assert_eq!(INITIAL_SALT_V1[19], 0x0a);
        assert_eq!(INITIAL_SALT_V1.len(), 20);
    }

    #[test]
    fn build_nonce_xors_packet_number_low_bytes() {
        let iv = [0u8; 12];
        let n = build_nonce(&iv, 0x0102_0304_0506_0708);
        assert_eq!(n[11], 0x08);
        assert_eq!(n[10], 0x07);
        assert_eq!(n[4], 0x01);
        assert_eq!(n[0], 0x00);
    }

    #[test]
    fn nonce_zero_packet_number_equals_iv() {
        let iv: [u8; 12] = [
            0xfa, 0xcd, 0xc5, 0xed, 0x8c, 0xe0, 0x07, 0x4f, 0x4d, 0x3b, 0x12, 0x7f,
        ];
        let n = build_nonce(&iv, 0);
        assert_eq!(n, iv);
    }

    #[test]
    fn encrypt_then_decrypt_round_trips() {
        let key = [0x42u8; 16];
        let iv = [0x33u8; 12];
        let aad = b"unprotected header";
        let pt = b"some quic frames here";
        let ct = encrypt_payload(&key, &iv, 1, aad, pt);
        assert!(ct.len() == pt.len() + 16); // AEAD tag appended
        let recovered = decrypt_payload(&key, &iv, 1, aad, &ct).expect("decrypt");
        assert_eq!(recovered, pt);
    }

    #[test]
    fn decrypt_with_wrong_packet_number_fails() {
        let key = [0x42u8; 16];
        let iv = [0x33u8; 12];
        let ct = encrypt_payload(&key, &iv, 1, b"", b"hello");
        // Bumping the packet number changes the nonce → AEAD tag
        // verification must fail.
        assert!(decrypt_payload(&key, &iv, 2, b"", &ct).is_none());
    }

    #[test]
    fn decrypt_with_tampered_ciphertext_fails() {
        let key = [0x42u8; 16];
        let iv = [0x33u8; 12];
        let mut ct = encrypt_payload(&key, &iv, 1, b"", b"hello");
        ct[0] ^= 0x01; // tamper
        assert!(decrypt_payload(&key, &iv, 1, b"", &ct).is_none());
    }

    #[test]
    fn short_header_rejects_long_header_byte() {
        let buf = [0xC0, 1, 2, 3];
        assert!(parse_short_header(&buf, 0).is_none());
    }

    fn hex(s: &str) -> Vec<u8> {
        (0..s.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap())
            .collect()
    }

    #[test]
    fn initial_keys_match_rfc9001_appendix_a1() {
        // RFC 9001 §A.1 — the canonical worked example. Client DCID is
        // 0x8394c8f03e515708; the derived client_initial key/iv/hp are
        // fixed test vectors that any conforming implementation must
        // reproduce. This is the real proof our HKDF-Expand-Label + salt
        // are correct, not just self-consistent.
        let dcid = hex("8394c8f03e515708");
        let keys = derive_client_initial_keys(&dcid);
        assert_eq!(keys.key.to_vec(), hex("1f369613dd76d5467730efcbe3b1a22d"));
        assert_eq!(keys.iv.to_vec(), hex("fa044b2f42a3fd3b46fb255c"));
        assert_eq!(keys.hp.to_vec(), hex("9f50449e04a0e810283a1e9933adedd2"));
    }

    #[test]
    fn hkdf_expand_label_client_in_matches_rfc9001() {
        // Intermediate vector: client_initial_secret from §A.1 is
        // c00cf151ca5be075ed0ebfb5c80323c42d6b7db67881289af4008f1f6c357aea.
        let dcid = hex("8394c8f03e515708");
        let initial_secret = cv_crypto::hkdf::extract(&INITIAL_SALT_V1, &dcid);
        let client_secret = hkdf_expand_label(&initial_secret, b"client in", b"", 32);
        assert_eq!(
            client_secret,
            hex("c00cf151ca5be075ed0ebfb5c80323c42d6b7db67881289af4008f1f6c357aea")
        );
    }
}
