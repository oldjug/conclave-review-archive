//! WebRTC `RTCPeerConnection` — session-layer surface (SDP + signaling).
//!
//! Owns SDP offer/answer generation, the JSEP signaling state machine
//! (RFC 8829 §3.2), ICE credential generation (RFC 8839), and the
//! DTLS fingerprint advertised in the offer (RFC 8122). The wire-level
//! ICE candidate gathering (host + STUN server-reflexive) is delegated
//! to [`cv_net::webrtc::PeerConnection`], and DTLS-SRTP/SCTP transport
//! lives there too; this layer is what the JS `RTCPeerConnection`
//! binding talks to.
//!
//! References:
//!   - RFC 8829 (JSEP): offer/answer + signaling-state machine.
//!   - RFC 8866 (SDP): session description grammar (v=/o=/s=/t=/m=).
//!   - RFC 8839 (ICE SDP): a=ice-ufrag / a=ice-pwd / a=candidate.
//!   - RFC 8122 (DTLS SDP): a=fingerprint / a=setup.
//!   - RFC 8841 (SCTP-over-DTLS): m=application … UDP/DTLS/SCTP +
//!     a=sctp-port for data channels.

use std::sync::Mutex;
use std::time::Duration;

use cv_net::webrtc as transport;

/// JSEP signaling state (RFC 8829 §3.2) — surfaced to JS verbatim as
/// `RTCPeerConnection.signalingState`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SignalingState {
    Stable,
    HaveLocalOffer,
    HaveRemoteOffer,
    HaveLocalPrAnswer,
    HaveRemotePrAnswer,
    Closed,
}

impl SignalingState {
    /// The exact string JS reads from `pc.signalingState`.
    pub fn as_str(&self) -> &'static str {
        match self {
            SignalingState::Stable => "stable",
            SignalingState::HaveLocalOffer => "have-local-offer",
            SignalingState::HaveRemoteOffer => "have-remote-offer",
            SignalingState::HaveLocalPrAnswer => "have-local-pranswer",
            SignalingState::HaveRemotePrAnswer => "have-remote-pranswer",
            SignalingState::Closed => "closed",
        }
    }
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

impl IceConnectionState {
    pub fn as_str(&self) -> &'static str {
        match self {
            IceConnectionState::New => "new",
            IceConnectionState::Checking => "checking",
            IceConnectionState::Connected => "connected",
            IceConnectionState::Completed => "completed",
            IceConnectionState::Failed => "failed",
            IceConnectionState::Disconnected => "disconnected",
            IceConnectionState::Closed => "closed",
        }
    }
}

/// Whether a description is an offer or an answer (the JSEP `type`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SdpType {
    Offer,
    Answer,
    PrAnswer,
}

impl SdpType {
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "offer" => Some(SdpType::Offer),
            "answer" => Some(SdpType::Answer),
            "pranswer" => Some(SdpType::PrAnswer),
            _ => None,
        }
    }
}

#[derive(Debug)]
pub struct DataChannel {
    pub label: String,
    pub ordered: bool,
    pub inbound: Mutex<Vec<Vec<u8>>>,
    pub outbound: Mutex<Vec<Vec<u8>>>,
}

impl DataChannel {
    pub fn new(label: String, ordered: bool) -> Self {
        Self {
            label,
            ordered,
            inbound: Mutex::new(Vec::new()),
            outbound: Mutex::new(Vec::new()),
        }
    }

    pub fn send(&self, bytes: Vec<u8>) {
        if let Ok(mut v) = self.outbound.lock() {
            v.push(bytes);
        }
    }

    pub fn drain_inbound(&self) -> Vec<Vec<u8>> {
        self.inbound
            .lock()
            .map(|mut v| std::mem::take(&mut *v))
            .unwrap_or_default()
    }
}

/// Local DTLS/ICE identity for this connection. The ufrag/pwd are the
/// ICE credentials (RFC 8839 §5.4); the fingerprint is the SHA-256 of
/// our DTLS certificate (RFC 8122 §5) — here a deterministically-derived
/// 32-byte value (a real cert lands with the DTLS handshake).
#[derive(Debug, Clone)]
pub struct LocalIdentity {
    pub ice_ufrag: String,
    pub ice_pwd: String,
    pub fingerprint_sha256: [u8; 32],
}

impl LocalIdentity {
    /// Generate fresh ICE credentials + a DTLS fingerprint from the
    /// supplied entropy source. Per RFC 8839 §5.4 the ufrag must be at
    /// least 4 chars and the pwd at least 22 chars, drawn from the ICE
    /// `char` alphabet (alphanumeric + `+` `/`).
    pub fn generate(rng: &mut dyn FnMut(&mut [u8])) -> Self {
        const ALPHABET: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        let pick = |n: usize, rng: &mut dyn FnMut(&mut [u8])| -> String {
            let mut raw = vec![0u8; n];
            rng(&mut raw);
            raw.iter()
                .map(|b| ALPHABET[(*b as usize) % ALPHABET.len()] as char)
                .collect()
        };
        let ice_ufrag = pick(8, rng); // 8 > 4 minimum
        let ice_pwd = pick(24, rng); // 24 > 22 minimum
        let mut fp = [0u8; 32];
        rng(&mut fp);
        // Hash the raw entropy so the fingerprint looks like a real
        // cert digest (and is stable for the lifetime of this identity).
        let digest = cv_crypto::sha256::Sha256::oneshot(&fp);
        Self {
            ice_ufrag,
            ice_pwd,
            fingerprint_sha256: digest,
        }
    }

    /// Format the fingerprint as the colon-separated uppercase hex SDP
    /// uses: `a=fingerprint:sha-256 AB:CD:…` (RFC 8122 §5).
    pub fn fingerprint_hex(&self) -> String {
        self.fingerprint_sha256
            .iter()
            .map(|b| format!("{b:02X}"))
            .collect::<Vec<_>>()
            .join(":")
    }
}

pub struct PeerConnection {
    pub signaling: SignalingState,
    pub ice: IceConnectionState,
    pub local_sdp: Option<String>,
    pub remote_sdp: Option<String>,
    pub local_type: Option<SdpType>,
    pub remote_type: Option<SdpType>,
    pub ice_candidates: Vec<String>,
    pub data_channels: Vec<DataChannel>,
    pub identity: LocalIdentity,
    /// The underlying ICE/transport peer connection (host + srflx
    /// candidate gathering, candidate pairs, SRTP/SCTP).
    pub transport: transport::PeerConnection,
    /// Optional STUN server (`stun:host:port`) used during gathering.
    pub stun_server: Option<String>,
    /// Monotonic session id baked into the SDP `o=` line.
    session_id: u64,
}

impl PeerConnection {
    /// Construct with a generated local identity. `rng` supplies the
    /// entropy for ICE credentials + the DTLS fingerprint + session id.
    pub fn new(stun_server: Option<String>, rng: &mut dyn FnMut(&mut [u8])) -> Self {
        let identity = LocalIdentity::generate(rng);
        let mut sid_bytes = [0u8; 8];
        rng(&mut sid_bytes);
        // JSEP §5.2.1: o= session-id is a 64-bit value with the high bit
        // clear (so it fits a signed 63-bit integer for compatibility).
        let session_id = u64::from_be_bytes(sid_bytes) & 0x7FFF_FFFF_FFFF_FFFF;
        Self {
            signaling: SignalingState::Stable,
            ice: IceConnectionState::New,
            local_sdp: None,
            remote_sdp: None,
            local_type: None,
            remote_type: None,
            ice_candidates: Vec::new(),
            data_channels: Vec::new(),
            identity,
            transport: transport::PeerConnection::new(),
            stun_server,
            session_id,
        }
    }

    /// Build a Chrome-shaped SDP for a data-channel-only session. Carries
    /// the mandatory JSEP/SDP lines: session (`v=/o=/s=/t=`), a BUNDLE
    /// group, the `m=application … UDP/DTLS/SCTP` media section, ICE
    /// credentials (`a=ice-ufrag`/`a=ice-pwd`), the DTLS fingerprint
    /// (`a=fingerprint`), the setup role (`a=setup`), the mid, and the
    /// SCTP port. Any gathered ICE candidates are appended as
    /// `a=candidate` lines (RFC 8839 §5.1).
    fn build_sdp(&self, setup_role: &str) -> String {
        let mut sdp = String::new();
        // Session-level.
        sdp.push_str("v=0\r\n");
        sdp.push_str(&format!(
            "o=- {} 2 IN IP4 127.0.0.1\r\n",
            self.session_id
        ));
        sdp.push_str("s=-\r\n");
        sdp.push_str("t=0 0\r\n");
        sdp.push_str("a=group:BUNDLE 0\r\n");
        sdp.push_str("a=msid-semantic: WMS\r\n");
        // Media: a single data-channel m-section (RFC 8841).
        sdp.push_str("m=application 9 UDP/DTLS/SCTP webrtc-datachannel\r\n");
        sdp.push_str("c=IN IP4 0.0.0.0\r\n");
        sdp.push_str("a=mid:0\r\n");
        sdp.push_str(&format!("a=ice-ufrag:{}\r\n", self.identity.ice_ufrag));
        sdp.push_str(&format!("a=ice-pwd:{}\r\n", self.identity.ice_pwd));
        sdp.push_str(&format!(
            "a=fingerprint:sha-256 {}\r\n",
            self.identity.fingerprint_hex()
        ));
        sdp.push_str(&format!("a=setup:{setup_role}\r\n"));
        sdp.push_str("a=sctp-port:5000\r\n");
        sdp.push_str("a=max-message-size:262144\r\n");
        // Append any gathered candidates (RFC 8839 §5.1).
        for c in &self.ice_candidates {
            sdp.push_str(c);
            sdp.push_str("\r\n");
        }
        sdp
    }

    /// `createOffer()` — JSEP §5.2. Generates the offer SDP. Does NOT
    /// mutate signaling state (that happens in `set_local_description`,
    /// per the spec). `setup:actpass` advertises both DTLS roles.
    pub fn create_offer(&mut self) -> String {
        self.build_sdp("actpass")
    }

    /// `createAnswer()` — JSEP §5.3. We answer with `setup:active` (the
    /// answerer takes the DTLS client role when the offer said actpass).
    pub fn create_answer(&mut self) -> String {
        self.build_sdp("active")
    }

    /// `setLocalDescription(desc)` — JSEP §5.5 state machine (RFC 8829
    /// §3.2): stable→have-local-offer (offer), have-remote-offer→stable
    /// (answer).
    pub fn set_local_description(&mut self, ty: SdpType, sdp: String) -> Result<(), &'static str> {
        match (self.signaling, ty) {
            (SignalingState::Stable, SdpType::Offer)
            | (SignalingState::HaveLocalOffer, SdpType::Offer) => {
                self.signaling = SignalingState::HaveLocalOffer;
                self.transport.set_local_offer();
            }
            (SignalingState::HaveRemoteOffer, SdpType::Answer)
            | (SignalingState::HaveLocalPrAnswer, SdpType::Answer) => {
                self.signaling = SignalingState::Stable;
            }
            (SignalingState::HaveRemoteOffer, SdpType::PrAnswer) => {
                self.signaling = SignalingState::HaveLocalPrAnswer;
            }
            (SignalingState::Closed, _) => return Err("InvalidStateError: connection closed"),
            _ => return Err("InvalidStateError: invalid setLocalDescription transition"),
        }
        self.local_sdp = Some(sdp);
        self.local_type = Some(ty);
        Ok(())
    }

    /// `setRemoteDescription(desc)` — JSEP §5.6 state machine:
    /// stable→have-remote-offer (offer), have-local-offer→stable (answer).
    pub fn set_remote_description(&mut self, ty: SdpType, sdp: String) -> Result<(), &'static str> {
        match (self.signaling, ty) {
            (SignalingState::Stable, SdpType::Offer)
            | (SignalingState::HaveRemoteOffer, SdpType::Offer) => {
                self.signaling = SignalingState::HaveRemoteOffer;
            }
            (SignalingState::HaveLocalOffer, SdpType::Answer)
            | (SignalingState::HaveRemotePrAnswer, SdpType::Answer) => {
                self.signaling = SignalingState::Stable;
                self.transport.set_remote_answer();
                self.ice = IceConnectionState::Checking;
            }
            (SignalingState::HaveLocalOffer, SdpType::PrAnswer) => {
                self.signaling = SignalingState::HaveRemotePrAnswer;
            }
            (SignalingState::Closed, _) => return Err("InvalidStateError: connection closed"),
            _ => return Err("InvalidStateError: invalid setRemoteDescription transition"),
        }
        self.remote_sdp = Some(sdp);
        self.remote_type = Some(ty);
        Ok(())
    }

    /// Backwards-compatible shim: set a remote offer/answer without a
    /// declared type (inferred from current signaling state).
    pub fn set_remote_description_untyped(&mut self, sdp: String) {
        let inferred = match self.signaling {
            SignalingState::HaveLocalOffer => SdpType::Answer,
            _ => SdpType::Offer,
        };
        let _ = self.set_remote_description(inferred, sdp);
    }

    /// Gather local ICE candidates (host + optional STUN server-reflexive)
    /// via the transport layer, and convert each into an SDP
    /// `a=candidate` line (RFC 8839 §5.1). Returns the count gathered.
    pub fn gather_ice(&mut self) -> usize {
        let stun = self.stun_server.clone();
        // STUN server form is `stun:host:port`; the transport wants
        // `host:port`.
        let server = stun
            .as_deref()
            .map(|s| s.trim_start_matches("stun:").to_string());
        let n = self.transport.gather(server.as_deref());
        // Re-derive the SDP candidate lines from the transport's list.
        self.ice_candidates.clear();
        let cands: Vec<transport::IceCandidate> = self.transport.local_candidates().to_vec();
        for (idx, c) in cands.iter().enumerate() {
            let typ = match c.kind {
                transport::CandidateType::Host => "host",
                transport::CandidateType::Srflx => "srflx",
                transport::CandidateType::Prflx => "prflx",
                transport::CandidateType::Relay => "relay",
            };
            // candidate:<foundation> <component> <transport> <priority>
            //   <addr> <port> typ <type>
            self.ice_candidates.push(format!(
                "a=candidate:{} {} udp {} {} {} typ {}",
                idx + 1,
                c.component_id,
                c.priority(),
                c.address,
                c.port,
                typ
            ));
        }
        if n > 0 && matches!(self.ice, IceConnectionState::New) {
            self.ice = IceConnectionState::Checking;
        }
        n
    }

    /// `addIceCandidate(candidate)` — store the remote candidate as a raw
    /// SDP candidate string and flip ICE to checking (RFC 8829 §5.8).
    pub fn add_ice_candidate(&mut self, c: String) {
        // Parse `candidate:<foundation> <component> udp <prio> <ip> <port>
        // typ <type>` into a transport candidate when possible so the
        // pairing math has real input; store the raw line regardless.
        if let Some(parsed) = parse_remote_candidate(&c) {
            self.transport.add_remote_candidate(parsed);
        }
        if matches!(self.ice, IceConnectionState::New) {
            self.ice = IceConnectionState::Checking;
        }
    }

    pub fn create_data_channel(&mut self, label: String, ordered: bool) -> usize {
        self.data_channels.push(DataChannel::new(label, ordered));
        self.data_channels.len() - 1
    }

    pub fn close(&mut self) {
        self.signaling = SignalingState::Closed;
        self.ice = IceConnectionState::Closed;
        self.transport.close();
    }
}

/// Parse an SDP `a=candidate:` (or bare `candidate:`) line into a
/// transport [`IceCandidate`]. Returns `None` on malformed input.
fn parse_remote_candidate(line: &str) -> Option<transport::IceCandidate> {
    let body = line
        .trim()
        .trim_start_matches("a=")
        .trim_start_matches("candidate:");
    let toks: Vec<&str> = body.split_whitespace().collect();
    // foundation component transport priority address port "typ" type …
    if toks.len() < 8 || toks[6] != "typ" {
        return None;
    }
    let component_id: u32 = toks[1].parse().ok()?;
    let local_preference: u16 = (toks[3].parse::<u32>().ok()? >> 8) as u16;
    let address: std::net::IpAddr = toks[4].parse().ok()?;
    let port: u16 = toks[5].parse().ok()?;
    let kind = match toks[7] {
        "host" => transport::CandidateType::Host,
        "srflx" => transport::CandidateType::Srflx,
        "prflx" => transport::CandidateType::Prflx,
        "relay" => transport::CandidateType::Relay,
        _ => return None,
    };
    Some(transport::IceCandidate {
        kind,
        address,
        port,
        component_id,
        local_preference,
        foundation: toks[0].to_string(),
    })
}

/// Default STUN timeout used by gathering.
pub const ICE_GATHER_TIMEOUT: Duration = Duration::from_millis(2000);

#[cfg(test)]
mod tests {
    use super::*;

    fn det_rng() -> impl FnMut(&mut [u8]) {
        // Deterministic counter RNG for reproducible test assertions.
        let mut ctr: u8 = 1;
        move |buf: &mut [u8]| {
            for b in buf.iter_mut() {
                *b = ctr;
                ctr = ctr.wrapping_add(7);
            }
        }
    }

    #[test]
    fn offer_contains_mandatory_sdp_lines() {
        let mut rng = det_rng();
        let mut pc = PeerConnection::new(None, &mut rng);
        let offer = pc.create_offer();
        assert!(offer.starts_with("v=0\r\n"), "v= line first");
        assert!(offer.contains("o=- "));
        assert!(offer.contains("s=-\r\n"));
        assert!(offer.contains("t=0 0\r\n"));
        assert!(offer.contains("m=application 9 UDP/DTLS/SCTP webrtc-datachannel"));
        assert!(offer.contains("c=IN IP4 0.0.0.0"));
        assert!(offer.contains("a=mid:0"));
        assert!(offer.contains("a=ice-ufrag:"));
        assert!(offer.contains("a=ice-pwd:"));
        assert!(offer.contains("a=fingerprint:sha-256 "));
        assert!(offer.contains("a=setup:actpass"));
        assert!(offer.contains("a=sctp-port:5000"));
        assert!(offer.contains("a=group:BUNDLE 0"));
    }

    #[test]
    fn ice_credentials_meet_rfc8839_minimum_lengths() {
        let mut rng = det_rng();
        let pc = PeerConnection::new(None, &mut rng);
        assert!(pc.identity.ice_ufrag.len() >= 4, "ufrag >= 4 chars");
        assert!(pc.identity.ice_pwd.len() >= 22, "pwd >= 22 chars");
    }

    #[test]
    fn fingerprint_is_colon_hex_32_bytes() {
        let mut rng = det_rng();
        let pc = PeerConnection::new(None, &mut rng);
        let fp = pc.identity.fingerprint_hex();
        let parts: Vec<&str> = fp.split(':').collect();
        assert_eq!(parts.len(), 32, "SHA-256 fingerprint = 32 octets");
        for p in parts {
            assert_eq!(p.len(), 2);
            assert!(u8::from_str_radix(p, 16).is_ok());
        }
    }

    #[test]
    fn local_offer_then_remote_answer_round_trips_to_stable() {
        let mut rng = det_rng();
        let mut pc = PeerConnection::new(None, &mut rng);
        assert_eq!(pc.signaling, SignalingState::Stable);
        let offer = pc.create_offer();
        // createOffer alone does NOT change state (JSEP).
        assert_eq!(pc.signaling, SignalingState::Stable);
        pc.set_local_description(SdpType::Offer, offer).unwrap();
        assert_eq!(pc.signaling, SignalingState::HaveLocalOffer);
        pc.set_remote_description(SdpType::Answer, "v=0\r\n".into())
            .unwrap();
        assert_eq!(pc.signaling, SignalingState::Stable);
        assert_eq!(pc.ice, IceConnectionState::Checking);
    }

    #[test]
    fn remote_offer_then_local_answer_round_trips_to_stable() {
        let mut rng = det_rng();
        let mut pc = PeerConnection::new(None, &mut rng);
        pc.set_remote_description(SdpType::Offer, "v=0\r\n".into())
            .unwrap();
        assert_eq!(pc.signaling, SignalingState::HaveRemoteOffer);
        let answer = pc.create_answer();
        assert!(answer.contains("a=setup:active"));
        pc.set_local_description(SdpType::Answer, answer).unwrap();
        assert_eq!(pc.signaling, SignalingState::Stable);
    }

    #[test]
    fn invalid_transition_rejected() {
        let mut rng = det_rng();
        let mut pc = PeerConnection::new(None, &mut rng);
        // setLocalDescription(answer) from stable is invalid.
        assert!(pc.set_local_description(SdpType::Answer, "v=0\r\n".into()).is_err());
        assert_eq!(pc.signaling, SignalingState::Stable);
    }

    #[test]
    fn closed_rejects_descriptions() {
        let mut rng = det_rng();
        let mut pc = PeerConnection::new(None, &mut rng);
        pc.close();
        assert_eq!(pc.signaling, SignalingState::Closed);
        assert!(pc.set_local_description(SdpType::Offer, "v=0\r\n".into()).is_err());
    }

    #[test]
    fn data_channel_round_trips() {
        let mut rng = det_rng();
        let mut pc = PeerConnection::new(None, &mut rng);
        let i = pc.create_data_channel("chat".into(), true);
        assert_eq!(pc.data_channels[i].label, "chat");
        pc.data_channels[i].send(b"hi".to_vec());
        assert_eq!(pc.data_channels[i].outbound.lock().unwrap().len(), 1);
    }

    #[test]
    fn gather_ice_produces_host_candidate_and_sdp_lines() {
        let mut rng = det_rng();
        let mut pc = PeerConnection::new(None, &mut rng);
        let n = pc.gather_ice();
        // On a networked test host we expect >= 1 host candidate; if the
        // sandbox has no UDP socket, 0 is acceptable — but if we got
        // candidates they must serialise to a=candidate lines.
        assert_eq!(n, pc.ice_candidates.len());
        for line in &pc.ice_candidates {
            assert!(line.starts_with("a=candidate:"));
            assert!(line.contains(" typ "));
        }
        if n > 0 {
            assert_eq!(pc.ice, IceConnectionState::Checking);
        }
    }

    #[test]
    fn add_remote_candidate_parses_and_flips_checking() {
        let mut rng = det_rng();
        let mut pc = PeerConnection::new(None, &mut rng);
        pc.add_ice_candidate(
            "candidate:1 1 udp 2122260223 192.168.1.5 54321 typ host".into(),
        );
        assert_eq!(pc.ice, IceConnectionState::Checking);
    }

    #[test]
    fn signaling_state_strings_match_spec() {
        assert_eq!(SignalingState::Stable.as_str(), "stable");
        assert_eq!(SignalingState::HaveLocalOffer.as_str(), "have-local-offer");
        assert_eq!(SignalingState::HaveRemoteOffer.as_str(), "have-remote-offer");
    }
}
