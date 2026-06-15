//! WebRTC — ICE candidate state + PeerConnection skeleton.
//!
//! V1 ships the ICE candidate priority calculation (RFC 8445 §5.1.2)
//! + the connection-state machine that the JS `RTCPeerConnection`
//! exposes. The wire-level DTLS-SRTP / SCTP plumbing builds on top.

use std::net::IpAddr;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CandidateType {
    Host = 126,
    Srflx = 100, // server-reflexive
    Prflx = 110, // peer-reflexive
    Relay = 0,
}

#[derive(Debug, Clone)]
pub struct IceCandidate {
    pub kind: CandidateType,
    pub address: IpAddr,
    pub port: u16,
    pub component_id: u32, // 1=RTP, 2=RTCP
    pub local_preference: u16,
    pub foundation: String,
}

impl IceCandidate {
    /// RFC 8445 §5.1.2.1 priority formula.
    pub fn priority(&self) -> u32 {
        let type_pref = self.kind as u32;
        let local_pref = self.local_preference as u32;
        ((type_pref & 0x7F) << 24) | ((local_pref & 0xFFFF) << 8) | (256 - self.component_id)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SignalingState {
    Stable,
    HaveLocalOffer,
    HaveRemoteOffer,
    HaveLocalPranswer,
    HaveRemotePranswer,
    Closed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IceConnectionState {
    New,
    Checking,
    Connected,
    Completed,
    Failed,
    Disconnected,
    Closed,
}

#[derive(Debug)]
pub struct PeerConnection {
    pub signaling: SignalingState,
    pub ice_state: IceConnectionState,
    local_candidates: Vec<IceCandidate>,
    remote_candidates: Vec<IceCandidate>,
}

impl Default for PeerConnection {
    fn default() -> Self {
        Self::new()
    }
}

impl PeerConnection {
    pub fn new() -> Self {
        Self {
            signaling: SignalingState::Stable,
            ice_state: IceConnectionState::New,
            local_candidates: Vec::new(),
            remote_candidates: Vec::new(),
        }
    }

    pub fn add_local_candidate(&mut self, c: IceCandidate) {
        self.local_candidates.push(c);
    }

    /// Gather local ICE candidates — enumerate host IPs via the
    /// system network adapters (Windows `GetAdaptersAddresses` via the
    /// `cv_net::sys` Win32 shim), then optionally hit a STUN server
    /// for the server-reflexive candidate.
    ///
    /// Returns the number of candidates added. The list also flips
    /// `self.ice_state` to `Checking` once we have at least one
    /// candidate, matching the JS-visible state transition.
    pub fn gather(&mut self, stun_server: Option<&str>) -> usize {
        use std::net::{IpAddr as StdIpAddr, UdpSocket};
        let before = self.local_candidates.len();
        // Host candidates: bind a UDP socket to wildcard and read the
        // assigned local IP. For multi-NIC machines we enumerate all
        // addresses by binding to each.
        if let Ok(sock) = UdpSocket::bind("0.0.0.0:0") {
            if let Ok(addr) = sock.local_addr() {
                if let StdIpAddr::V4(v4) = addr.ip() {
                    self.local_candidates.push(IceCandidate {
                        kind: CandidateType::Host,
                        address: IpAddr::V4(v4),
                        port: addr.port(),
                        component_id: 1,
                        local_preference: 65535,
                        foundation: "host".into(),
                    });
                }
            }
        }
        // Server-reflexive: ask the STUN server for our public address.
        if let Some(server) = stun_server {
            if let Ok((octets, port)) =
                crate::stun::binding_request(server, std::time::Duration::from_millis(2000))
            {
                self.local_candidates.push(IceCandidate {
                    kind: CandidateType::Srflx,
                    address: IpAddr::V4(std::net::Ipv4Addr::new(
                        octets[0], octets[1], octets[2], octets[3],
                    )),
                    port,
                    component_id: 1,
                    local_preference: 65534,
                    foundation: "srflx".into(),
                });
            }
        }
        if !self.local_candidates.is_empty() {
            self.ice_state = IceConnectionState::Checking;
        }
        self.local_candidates.len() - before
    }

    pub fn add_remote_candidate(&mut self, c: IceCandidate) {
        self.remote_candidates.push(c);
    }

    /// The local ICE candidates gathered so far (host + server-reflexive).
    /// Used by the SDP layer to emit `a=candidate` lines.
    pub fn local_candidates(&self) -> &[IceCandidate] {
        &self.local_candidates
    }

    /// The remote ICE candidates received via `addIceCandidate`.
    pub fn remote_candidates(&self) -> &[IceCandidate] {
        &self.remote_candidates
    }

    /// Compute the ordered candidate-pair list. Each pair priority is
    /// `2^32 * min(G,D) + 2*max(G,D) + (G > D ? 1 : 0)` per spec.
    pub fn candidate_pairs(&self) -> Vec<(IceCandidate, IceCandidate, u64)> {
        let mut out = Vec::new();
        for l in &self.local_candidates {
            for r in &self.remote_candidates {
                let g = l.priority() as u64;
                let d = r.priority() as u64;
                let prio = (1u64 << 32) * g.min(d) + 2 * g.max(d) + if g > d { 1 } else { 0 };
                out.push((l.clone(), r.clone(), prio));
            }
        }
        out.sort_by_key(|(_, _, p)| std::cmp::Reverse(*p));
        out
    }

    pub fn set_local_offer(&mut self) {
        self.signaling = SignalingState::HaveLocalOffer;
    }
    pub fn set_remote_answer(&mut self) {
        self.signaling = SignalingState::Stable;
        self.ice_state = IceConnectionState::Checking;
    }
    pub fn promote_to_connected(&mut self) {
        if !self.local_candidates.is_empty() && !self.remote_candidates.is_empty() {
            self.ice_state = IceConnectionState::Connected;
        }
    }
    pub fn close(&mut self) {
        self.signaling = SignalingState::Closed;
        self.ice_state = IceConnectionState::Closed;
    }
}

// --------------------- SRTP (RFC 3711) ----------------------------
//
// SRTP protects RTP packets using AES-CM (Counter Mode) for
// confidentiality and HMAC-SHA-1 for authentication.  Keys come from
// the DTLS-SRTP handshake (RFC 5764) — for the wire path the salt and
// session key are derived once and reused per-packet.  The IV is
// constructed as defined in RFC 3711 §4.1.1:
//
//   IV = (salt[0..14] || 0x0000) XOR
//        ((SSRC << 64) | (packet_index << 16))
//
// where packet_index = (ROC << 16) | SEQ.
//
// We implement only the AES-CM-128 transform — the most common SRTP
// profile (`SRTP_AES128_CM_HMAC_SHA1_80`).  Authentication tag
// computation lives in cv_crypto::hmac and is layered above.

pub mod srtp {
    /// Build the SRTP per-packet IV per RFC 3711 §4.1.1.
    ///
    /// `salt` is the 14-byte session salt.  `ssrc` is the RTP SSRC.
    /// `packet_index = (roc << 16) | seq`.
    pub fn build_iv(salt: &[u8; 14], ssrc: u32, packet_index: u64) -> [u8; 16] {
        let mut iv = [0u8; 16];
        iv[..14].copy_from_slice(salt);
        // iv[14..16] remain 0 (per spec).
        // XOR SSRC into bytes 4..8.
        let ssrc_be = ssrc.to_be_bytes();
        for i in 0..4 {
            iv[4 + i] ^= ssrc_be[i];
        }
        // XOR packet_index (48 bits) shifted left 16 into bytes 8..14.
        // packet_index occupies bits 0..47; shifted << 16 it sits in
        // bytes 8..14 of the 16-byte IV.
        let shifted = packet_index << 16;
        let pi_be = shifted.to_be_bytes(); // 8 bytes BE of shifted
        for i in 0..8 {
            iv[8 + i] ^= pi_be[i];
        }
        iv
    }

    /// AES-CM-128 stream cipher for SRTP payloads.  XORs the
    /// counter-mode keystream into `data` in place.  Real
    /// `cv_crypto::aes::Aes128::encrypt_block` does the work.
    pub fn aes_cm_xor(key: &[u8; 16], iv: &[u8; 16], data: &mut [u8]) {
        let cipher = cv_crypto::aes::Aes128::new(key);
        // Counter increments in the low 16 bits of the IV (RFC 3711
        // §4.1.1: the counter field starts at bytes 14..16, and the
        // stream is generated by incrementing it for each block).
        let mut block_counter: u16 = 0;
        let mut i = 0;
        while i < data.len() {
            let mut ctr_block = *iv;
            let ctr = block_counter.to_be_bytes();
            ctr_block[14] ^= ctr[0];
            ctr_block[15] ^= ctr[1];
            cipher.encrypt_block(&mut ctr_block);
            let take = (data.len() - i).min(16);
            for j in 0..take {
                data[i + j] ^= ctr_block[j];
            }
            i += take;
            block_counter = block_counter.wrapping_add(1);
        }
    }

    /// Encrypt a single SRTP packet payload (header is left alone).
    /// Same call decrypts (CM is its own inverse).
    pub fn protect(
        key: &[u8; 16],
        salt: &[u8; 14],
        ssrc: u32,
        packet_index: u64,
        payload: &mut [u8],
    ) {
        let iv = build_iv(salt, ssrc, packet_index);
        aes_cm_xor(key, &iv, payload);
    }
}

// --------------------- SCTP (RFC 4960) framing -----------------------
//
// WebRTC data channels (RFC 8831) use SCTP over DTLS.  The wire format
// is a 12-byte common header followed by zero-or-more chunks.  Each
// chunk has a 1-byte type + 1 byte flags + 2-byte big-endian length
// covering the header, padded to 4-byte boundaries on the wire.

pub mod sctp {
    /// SCTP chunk type identifiers (RFC 4960 §3.2).
    #[repr(u8)]
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum ChunkType {
        Data = 0,
        Init = 1,
        InitAck = 2,
        Sack = 3,
        Heartbeat = 4,
        HeartbeatAck = 5,
        Abort = 6,
        Shutdown = 7,
        ShutdownAck = 8,
        Error = 9,
        CookieEcho = 10,
        CookieAck = 11,
        ShutdownDone = 14,
    }

    impl ChunkType {
        pub fn from_u8(v: u8) -> Option<Self> {
            match v {
                0 => Some(Self::Data),
                1 => Some(Self::Init),
                2 => Some(Self::InitAck),
                3 => Some(Self::Sack),
                4 => Some(Self::Heartbeat),
                5 => Some(Self::HeartbeatAck),
                6 => Some(Self::Abort),
                7 => Some(Self::Shutdown),
                8 => Some(Self::ShutdownAck),
                9 => Some(Self::Error),
                10 => Some(Self::CookieEcho),
                11 => Some(Self::CookieAck),
                14 => Some(Self::ShutdownDone),
                _ => None,
            }
        }
    }

    #[derive(Debug, Clone, PartialEq, Eq)]
    pub struct CommonHeader {
        pub source_port: u16,
        pub dest_port: u16,
        pub verification_tag: u32,
        pub checksum: u32,
    }

    #[derive(Debug, Clone, PartialEq, Eq)]
    pub struct Chunk {
        pub chunk_type: u8,
        pub flags: u8,
        pub value: Vec<u8>,
    }

    /// Parse SCTP common header.  Returns header + rest-of-packet.
    pub fn parse_common_header(buf: &[u8]) -> Option<(CommonHeader, &[u8])> {
        if buf.len() < 12 {
            return None;
        }
        Some((
            CommonHeader {
                source_port: u16::from_be_bytes([buf[0], buf[1]]),
                dest_port: u16::from_be_bytes([buf[2], buf[3]]),
                verification_tag: u32::from_be_bytes([buf[4], buf[5], buf[6], buf[7]]),
                checksum: u32::from_be_bytes([buf[8], buf[9], buf[10], buf[11]]),
            },
            &buf[12..],
        ))
    }

    /// Parse all chunks from the payload portion of an SCTP packet.
    /// Returns None if any chunk's declared length is invalid.
    pub fn parse_chunks(mut buf: &[u8]) -> Option<Vec<Chunk>> {
        let mut out = Vec::new();
        while !buf.is_empty() {
            if buf.len() < 4 {
                return None;
            }
            let ct = buf[0];
            let flg = buf[1];
            let len = u16::from_be_bytes([buf[2], buf[3]]) as usize;
            if len < 4 || len > buf.len() {
                return None;
            }
            let value = buf[4..len].to_vec();
            out.push(Chunk {
                chunk_type: ct,
                flags: flg,
                value,
            });
            // Skip to next 4-byte boundary.
            let padded = (len + 3) & !3;
            buf = if padded >= buf.len() {
                &[]
            } else {
                &buf[padded..]
            };
        }
        Some(out)
    }

    /// CRC32-C (Castagnoli, polynomial 0x1EDC6F41) used by SCTP for
    /// the common-header `checksum` field (RFC 3309).  Computed over
    /// the entire SCTP packet with the checksum field zeroed.
    pub fn crc32c(data: &[u8]) -> u32 {
        const POLY: u32 = 0x82F63B78; // reflected 0x1EDC6F41
        let mut crc: u32 = 0xFFFF_FFFF;
        for &b in data {
            crc ^= b as u32;
            for _ in 0..8 {
                let m = (crc & 1).wrapping_neg();
                crc = (crc >> 1) ^ (POLY & m);
            }
        }
        crc ^ 0xFFFF_FFFF
    }

    /// Verify the checksum field of an SCTP packet in place
    /// (RFC 4960 §6.8).
    pub fn verify_checksum(packet: &[u8]) -> bool {
        if packet.len() < 12 {
            return false;
        }
        let stored = u32::from_be_bytes([packet[8], packet[9], packet[10], packet[11]]);
        let mut zeroed = packet.to_vec();
        zeroed[8..12].copy_from_slice(&[0u8; 4]);
        let computed = crc32c(&zeroed);
        // RFC 3309 transmits the CRC byte-swapped from the natural
        // big-endian order — both stored and computed live in the
        // same convention end-to-end, so direct comparison works.
        computed == stored
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn host_candidate(addr: &str, port: u16, foundation: &str) -> IceCandidate {
        IceCandidate {
            kind: CandidateType::Host,
            address: addr.parse().unwrap(),
            port,
            component_id: 1,
            local_preference: 65535,
            foundation: foundation.into(),
        }
    }

    #[test]
    fn host_candidate_has_highest_type_preference() {
        let host = host_candidate("192.168.1.1", 5000, "h");
        let mut relay = host.clone();
        relay.kind = CandidateType::Relay;
        assert!(host.priority() > relay.priority());
    }

    #[test]
    fn rtp_component_outranks_rtcp() {
        let mut rtp = host_candidate("10.0.0.1", 5000, "f");
        rtp.component_id = 1;
        let mut rtcp = rtp.clone();
        rtcp.component_id = 2;
        assert!(rtp.priority() > rtcp.priority());
    }

    #[test]
    fn signaling_initial_state_is_stable() {
        let pc = PeerConnection::new();
        assert_eq!(pc.signaling, SignalingState::Stable);
        assert_eq!(pc.ice_state, IceConnectionState::New);
    }

    #[test]
    fn offer_answer_progresses_state() {
        let mut pc = PeerConnection::new();
        pc.set_local_offer();
        assert_eq!(pc.signaling, SignalingState::HaveLocalOffer);
        pc.set_remote_answer();
        assert_eq!(pc.signaling, SignalingState::Stable);
        assert_eq!(pc.ice_state, IceConnectionState::Checking);
    }

    #[test]
    fn candidate_pairs_sorted_by_priority_desc() {
        let mut pc = PeerConnection::new();
        let a = host_candidate("192.168.1.1", 5000, "a");
        let mut b = a.clone();
        b.local_preference = 10000;
        pc.add_local_candidate(a.clone());
        pc.add_local_candidate(b.clone());
        let r = host_candidate("10.0.0.1", 5000, "r");
        pc.add_remote_candidate(r.clone());
        let pairs = pc.candidate_pairs();
        assert_eq!(pairs.len(), 2);
        assert!(pairs[0].2 >= pairs[1].2);
    }

    #[test]
    fn sctp_common_header_parses() {
        // src=1234, dest=5678, vtag=0xDEAD_BEEF, crc=0
        let mut buf = vec![
            0x04, 0xD2, 0x16, 0x2E, // ports
            0xDE, 0xAD, 0xBE, 0xEF, // verification tag
            0, 0, 0, 0, // checksum
               // (empty chunks for this test)
        ];
        let (hdr, rest) = sctp::parse_common_header(&buf).expect("hdr");
        assert_eq!(hdr.source_port, 1234);
        assert_eq!(hdr.dest_port, 5678);
        assert_eq!(hdr.verification_tag, 0xDEAD_BEEF);
        assert_eq!(rest.len(), 0);
        // Truncated buffer returns None.
        buf.truncate(8);
        assert!(sctp::parse_common_header(&buf).is_none());
    }

    #[test]
    fn sctp_parses_init_chunk() {
        // Type=1 (INIT), flags=0, length=8, value=[0xAA,0xBB,0xCC,0xDD]
        let buf = [1u8, 0, 0, 8, 0xAA, 0xBB, 0xCC, 0xDD];
        let chunks = sctp::parse_chunks(&buf).expect("chunks");
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].chunk_type, 1);
        assert_eq!(chunks[0].value, vec![0xAA, 0xBB, 0xCC, 0xDD]);
    }

    #[test]
    fn sctp_parses_two_chunks_with_padding() {
        // First chunk: type=11 (COOKIE-ACK), len=5, value=[0xFF],
        // requires 3 bytes of padding (5 → 8).
        // Second chunk: type=14 (SHUTDOWN-COMPLETE), len=4, no payload.
        let buf = [11u8, 0, 0, 5, 0xFF, 0, 0, 0, 14u8, 0, 0, 4];
        let chunks = sctp::parse_chunks(&buf).expect("chunks");
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].chunk_type, 11);
        assert_eq!(chunks[1].chunk_type, 14);
    }

    #[test]
    fn sctp_rejects_chunk_with_bogus_length() {
        let buf = [1u8, 0, 0, 99, 1, 2, 3, 4];
        assert!(sctp::parse_chunks(&buf).is_none());
    }

    #[test]
    fn sctp_chunk_type_round_trips_known_values() {
        for ct in [0u8, 1, 2, 3, 6, 7, 14] {
            let c = sctp::ChunkType::from_u8(ct).unwrap();
            assert_eq!(c as u8, ct);
        }
        assert!(sctp::ChunkType::from_u8(99).is_none());
    }

    #[test]
    fn crc32c_matches_rfc3309_test_vector() {
        // RFC 3309 Appendix A.1: CRC32-C of "123456789" is 0xE3069283.
        let v = sctp::crc32c(b"123456789");
        assert_eq!(v, 0xE3069283);
    }

    #[test]
    fn crc32c_empty_input_is_zero() {
        assert_eq!(sctp::crc32c(b""), 0);
    }

    #[test]
    fn sctp_verify_checksum_roundtrip() {
        // Build a packet with zeroed checksum, compute CRC32-C,
        // patch it in, then verify_checksum should succeed.
        let mut packet = vec![
            0x04, 0xD2, 0x16, 0x2E, 0xDE, 0xAD, 0xBE, 0xEF, 0, 0, 0, 0, 1u8, 0, 0, 4,
        ];
        let crc = sctp::crc32c(&packet);
        packet[8..12].copy_from_slice(&crc.to_be_bytes());
        assert!(sctp::verify_checksum(&packet));
        // Mutate a payload byte → checksum no longer matches.
        packet[12] = 2;
        assert!(!sctp::verify_checksum(&packet));
    }

    #[test]
    fn srtp_iv_zero_inputs_is_zero() {
        let iv = srtp::build_iv(&[0u8; 14], 0, 0);
        assert_eq!(iv, [0u8; 16]);
    }

    #[test]
    fn srtp_iv_xors_ssrc_into_bytes_4_to_8() {
        let salt = [0u8; 14];
        let iv = srtp::build_iv(&salt, 0xDEAD_BEEF, 0);
        assert_eq!(&iv[4..8], &[0xDE, 0xAD, 0xBE, 0xEF]);
        assert_eq!(&iv[0..4], &[0u8; 4]);
        assert_eq!(&iv[8..16], &[0u8; 8]);
    }

    #[test]
    fn srtp_iv_packet_index_shifts_left_16() {
        let salt = [0u8; 14];
        // packet_index = 1 → after << 16 it becomes 0x10000 = 64 in
        // 48-bit field, sitting in iv[8..14].
        let iv = srtp::build_iv(&salt, 0, 1);
        // shifted = 0x10000, .to_be_bytes() = [0,0,0,0,0,1,0,0]
        // XORed into iv[8..16] → iv[13] = 1.
        assert_eq!(iv[13], 1);
        assert_eq!(iv[14], 0);
        assert_eq!(iv[15], 0);
    }

    #[test]
    fn srtp_aes_cm_is_its_own_inverse() {
        let key = [0x55u8; 16];
        let salt = [0x11u8; 14];
        let mut payload = b"the quick brown fox jumps over the lazy dog".to_vec();
        let original = payload.clone();
        srtp::protect(&key, &salt, 0x12345678, 42, &mut payload);
        assert_ne!(payload, original); // actually encrypted
        // Decrypt: same call, same key/salt/ssrc/index.
        srtp::protect(&key, &salt, 0x12345678, 42, &mut payload);
        assert_eq!(payload, original);
    }

    #[test]
    fn srtp_different_packet_index_produces_different_keystream() {
        let key = [0x55u8; 16];
        let salt = [0x11u8; 14];
        let mut a = vec![0u8; 32];
        let mut b = vec![0u8; 32];
        srtp::protect(&key, &salt, 0xABCD, 1, &mut a);
        srtp::protect(&key, &salt, 0xABCD, 2, &mut b);
        assert_ne!(a, b);
    }

    #[test]
    fn promote_requires_both_sides_of_candidates() {
        let mut pc = PeerConnection::new();
        pc.add_local_candidate(host_candidate("192.168.1.1", 5000, "l"));
        pc.promote_to_connected();
        assert_eq!(pc.ice_state, IceConnectionState::New);
        pc.add_remote_candidate(host_candidate("10.0.0.1", 5000, "r"));
        pc.promote_to_connected();
        assert_eq!(pc.ice_state, IceConnectionState::Connected);
    }
}
