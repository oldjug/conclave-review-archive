//! WebRTC PeerConnection — session-layer surface.
//!
//! Owns SDP offer/answer state, ICE candidate gathering, and the
//! event hooks Pages set. The transport layer (DTLS-SRTP via
//! cv_net::webrtc, SCTP DataChannels, ICE STUN bindings) is plumbed
//! underneath; this layer is what the JS binding talks to.

use std::sync::Mutex;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SignalingState {
    Stable,
    HaveLocalOffer,
    HaveRemoteOffer,
    HaveLocalPrAnswer,
    HaveRemotePrAnswer,
    Closed,
}

#[derive(Debug, Clone, PartialEq, Eq)]
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

pub struct PeerConnection {
    pub signaling: SignalingState,
    pub ice: IceConnectionState,
    pub local_sdp: Option<String>,
    pub remote_sdp: Option<String>,
    pub ice_candidates: Vec<String>,
    pub data_channels: Vec<DataChannel>,
}

impl Default for PeerConnection {
    fn default() -> Self {
        Self {
            signaling: SignalingState::Stable,
            ice: IceConnectionState::New,
            local_sdp: None,
            remote_sdp: None,
            ice_candidates: Vec::new(),
            data_channels: Vec::new(),
        }
    }
}

impl PeerConnection {
    pub fn create_offer(&mut self) -> String {
        let sdp = format!(
            "v=0\r\no=- 0 0 IN IP4 127.0.0.1\r\ns=-\r\nt=0 0\r\nm=application 9 UDP/DTLS/SCTP webrtc-datachannel\r\nc=IN IP4 0.0.0.0\r\na=mid:0\r\na=sctp-port:5000\r\n"
        );
        self.local_sdp = Some(sdp.clone());
        self.signaling = SignalingState::HaveLocalOffer;
        sdp
    }

    pub fn set_remote_description(&mut self, sdp: String) {
        self.remote_sdp = Some(sdp);
        self.signaling = match self.signaling {
            SignalingState::HaveLocalOffer => SignalingState::Stable,
            _ => SignalingState::HaveRemoteOffer,
        };
    }

    pub fn create_answer(&mut self) -> String {
        let sdp = self
            .remote_sdp
            .clone()
            .unwrap_or_else(|| "v=0\r\no=- 0 0 IN IP4 127.0.0.1\r\ns=-\r\nt=0 0\r\n".into());
        self.local_sdp = Some(sdp.clone());
        sdp
    }

    pub fn add_ice_candidate(&mut self, c: String) {
        self.ice_candidates.push(c);
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
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn offer_then_answer_transitions_state() {
        let mut pc = PeerConnection::default();
        let offer = pc.create_offer();
        assert!(offer.contains("v=0"));
        assert_eq!(pc.signaling, SignalingState::HaveLocalOffer);
        pc.set_remote_description("v=0\r\n".into());
        assert_eq!(pc.signaling, SignalingState::Stable);
        let i = pc.create_data_channel("chat".into(), true);
        assert_eq!(pc.data_channels[i].label, "chat");
    }
}
