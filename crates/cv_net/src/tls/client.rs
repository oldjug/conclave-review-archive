//! TLS 1.3 client handshake state machine + `TlsStream`.
//!
//! Drives the handshake from `ClientHello` through `Finished`, deriving
//! traffic keys at each phase per RFC 8446 §7.1. After `Connected`, app
//! data flows through the established AEAD record layer.
//!
//! Chain validation: the presented certificate list is anchored at a
//! trusted Windows root via `CertGetCertificateChain` +
//! `CertVerifyCertificateChainPolicy(CERT_CHAIN_POLICY_SSL)` (see
//! `chain_validate.rs`). The leaf's signature on the handshake transcript
//! is checked against the SPKI we parsed ourselves with `cv_crypto::x509`.

use core::fmt;
use std::io;

use cv_crypto::asn1::Reader as AsnReader;
use cv_crypto::hmac::HmacSha256;
use cv_crypto::p256;
use cv_crypto::rsa::{Hash as RsaHash, RsaPublicKey, verify_pkcs1_v15, verify_pss};
use cv_crypto::sha256::Sha256;
use cv_crypto::x509::{self, Cert};
use cv_crypto::x25519::{x25519, x25519_public};

use crate::socket::Socket;
use crate::tls::chain_validate;
use crate::tls::key_schedule::{HashAlg, KeySchedule, hkdf_expand_label, traffic_keys};
use crate::tls::messages::{
    CipherSuite, ContentType, Decoder, HandshakeIter, HandshakeType, NamedGroup, SignatureScheme,
    TLS13_REAL_VERSION, build_client_hello_body, parse_server_hello, wrap_handshake, wrap_record,
};
use crate::tls::record::{Aead, AeadKey};

#[derive(Debug)]
pub enum TlsError {
    Io(io::Error),
    Decode(String),
    Protocol(String),
    Crypto(String),
    /// Server picked a TLS version older than 1.3.
    UnsupportedServerVersion(u16),
    UnsupportedCipherSuite(u16),
    UnsupportedKeyShareGroup(u16),
    UnsupportedSignatureScheme(u16),
    CertVerifyFailed,
    ChainInvalid(String),
    ServerFinishedMismatch,
    NoCertificate,
}

impl fmt::Display for TlsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(e) => write!(f, "io: {e}"),
            Self::Decode(s) => write!(f, "decode: {s}"),
            Self::Protocol(s) => write!(f, "protocol: {s}"),
            Self::Crypto(s) => write!(f, "crypto: {s}"),
            Self::UnsupportedServerVersion(v) => write!(f, "unsupported version 0x{v:04x}"),
            Self::UnsupportedCipherSuite(c) => write!(f, "unsupported cipher 0x{c:04x}"),
            Self::UnsupportedKeyShareGroup(g) => write!(f, "unsupported group 0x{g:04x}"),
            Self::UnsupportedSignatureScheme(s) => write!(f, "unsupported sig scheme 0x{s:04x}"),
            Self::CertVerifyFailed => f.write_str("CertificateVerify signature invalid"),
            Self::ChainInvalid(s) => write!(f, "certificate chain invalid: {s}"),
            Self::ServerFinishedMismatch => f.write_str("server Finished MAC mismatch"),
            Self::NoCertificate => f.write_str("server sent no certificate"),
        }
    }
}

impl std::error::Error for TlsError {}

impl From<io::Error> for TlsError {
    fn from(e: io::Error) -> Self {
        Self::Io(e)
    }
}

/// Convenience: a 32-byte random buffer drawn from the OS RNG.
fn rand_bytes(len: usize) -> Vec<u8> {
    // Pulled from `BCryptGenRandom` via `RtlGenRandom` (BCRYPT_USE_SYSTEM_PREFERRED_RNG).
    // For now use `getrandom`-like behaviour via Win32 directly.
    use super::sys_rand::fill;
    let mut out = vec![0u8; len];
    fill(&mut out);
    out
}

/// Established TLS 1.3 connection.
pub struct TlsStream {
    sock: Socket,
    /// Receive-side AEAD (server → client, application phase).
    rx: AeadKey,
    /// Send-side AEAD (client → server, application phase).
    tx: AeadKey,
    /// Decrypted-but-not-yet-consumed plaintext from inbound records.
    rx_plain: Vec<u8>,
    /// Raw socket bytes we've read but haven't yet framed into records.
    rx_raw: Vec<u8>,
    /// Set if the peer sent a close_notify.
    closed: bool,
    /// ALPN protocol the server selected from our offer list. Empty
    /// means the server didn't send an ALPN extension, in which case
    /// callers should default to HTTP/1.1.
    alpn: String,
    /// `Some` when the connection negotiated TLS 1.2 instead of 1.3.
    /// On that path `rx`/`tx` carry placeholder zero keys; all record
    /// reads/writes use the TLS 1.2 AEAD paths in `tls12.rs` instead.
    tls12: Option<Tls12RecordState>,
}

/// TLS 1.2 AEAD state (RFC 5288 §3 + RFC 7905 §2): symmetric keys +
/// per-direction sequence numbers + IV (4 bytes for AES-GCM, 12 for
/// ChaCha20-Poly1305) + raw cipher code so the record path knows how
/// to AEAD-seal/open.
pub(crate) struct Tls12RecordState {
    pub(crate) client_write_key: Vec<u8>,
    pub(crate) server_write_key: Vec<u8>,
    pub(crate) client_write_salt: Vec<u8>,
    pub(crate) server_write_salt: Vec<u8>,
    pub(crate) seq_client: u64,
    pub(crate) seq_server: u64,
    pub(crate) cipher_code: u16,
}

impl TlsStream {
    /// The protocol the server picked via ALPN (RFC 7301). Common
    /// values: "h2", "http/1.1", "" (none negotiated).
    pub fn alpn_protocol(&self) -> &str {
        &self.alpn
    }

    /// Set the receive timeout on the underlying socket. See
    /// `Socket::set_read_timeout_ms`.
    pub fn set_read_timeout_ms(&self, ms: u32) {
        self.sock.set_read_timeout_ms(ms);
    }

    /// Safe to reuse from the keep-alive pool? Only when the peer hasn't
    /// closed, we have no buffered inbound plaintext or raw record bytes
    /// left over (an idle connection at a clean message boundary has
    /// none), and the underlying socket reports no pending FIN/RST/data.
    pub fn is_reuse_safe(&self) -> bool {
        !self.closed
            && self.rx_plain.is_empty()
            && self.rx_raw.is_empty()
            && self.sock.is_reuse_safe()
    }
}

impl fmt::Debug for TlsStream {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TlsStream")
            .field("rx_seq", &self.rx.seq)
            .field("tx_seq", &self.tx.seq)
            .field("closed", &self.closed)
            .finish_non_exhaustive()
    }
}

impl TlsStream {
    /// Open a TLS 1.3 connection. `host` is the SNI / hostname-validation target.
    pub fn connect(mut sock: Socket, host: &str) -> Result<Self, TlsError> {
        let mut hs = HandshakeDriver::new(host)?;

        // 1. Send ClientHello.
        let client_hello_record = hs.build_client_hello()?;
        sock.write_all(&client_hello_record)
            .map_err(|e| TlsError::Protocol(format!("send CH: {e:?}")))?;

        // 2. Read ServerHello (plaintext).
        let sh_record = read_one_record(&mut sock)?;
        if sh_record.content_type != ContentType::Handshake as u8 {
            // Most useful failure mode here is "server sent an Alert
            // instead of our ServerHello" — the server rejected the
            // ClientHello before completing the handshake. Parse the
            // alert so the error names *which* of cipher/curve/version
            // negotiation failed; the wire-format "got 21" was useless
            // for debugging real-site failures.
            if sh_record.content_type == ContentType::Alert as u8 && sh_record.fragment.len() >= 2 {
                let level = sh_record.fragment[0];
                let desc = sh_record.fragment[1];
                return Err(TlsError::Protocol(format!(
                    "server rejected ClientHello: alert level={} {} ({})",
                    level,
                    desc,
                    alert_description_name(desc),
                )));
            }
            return Err(TlsError::Protocol(format!(
                "expected handshake, got {}",
                sh_record.content_type
            )));
        }
        hs.consume_server_hello(&sh_record.fragment)?;

        // TLS 1.2 fork — hand off to the dedicated 1.2 driver. From
        // here the 1.2 path is plaintext Certificate / SKE /
        // ServerHelloDone, then ClientKeyExchange / ChangeCipherSpec /
        // encrypted Finished. We then keep the socket and build a
        // TlsStream whose read/write paths use AES-GCM with TLS-1.2
        // nonce/AAD framing.
        if hs.tls12_selected {
            let state = super::tls12::Tls12HandoffState {
                host: hs.host.clone(),
                client_random: hs.client_random,
                server_random: {
                    // We didn't capture the server_random separately —
                    // re-parse it from the ServerHello body we just
                    // appended to the transcript. The ServerHello body
                    // is the second-to-last handshake message in the
                    // transcript right now (we haven't added anything
                    // after it). It begins after the 4-byte handshake
                    // header inside the only-handshake-message in
                    // `sh_record.fragment`.
                    let hs_iter = HandshakeIter::new(&sh_record.fragment);
                    let mut sr = [0u8; 32];
                    for msg in hs_iter {
                        let (kind, body, _full) =
                            msg.map_err(|e| TlsError::Decode(e.to_string()))?;
                        if kind == HandshakeType::ServerHello as u8 {
                            // Skip 2-byte legacy_version, then 32 bytes of random.
                            if body.len() >= 34 {
                                sr.copy_from_slice(&body[2..34]);
                            }
                            break;
                        }
                    }
                    sr
                },
                session_id: Vec::new(),
                cipher_suite: hs.cipher_suite_raw,
                transcript: hs.transcript.clone(),
                client_priv_p256: hs.client_priv_p256,
                client_priv_x25519: hs.client_priv,
                client_priv_p384: hs.client_priv_p384,
                ems_used: server_hello_has_ems(&sh_record.fragment),
            };
            let outcome = super::tls12::drive_tls12(&mut sock, state)?;
            // Build a TlsStream whose 1.3 AEAD slots hold dummies and
            // whose 1.2 record state holds the real keys.
            return Ok(Self {
                sock,
                rx: AeadKey::new(Aead::Aes128Gcm, vec![0u8; 16], &[0u8; 12]),
                tx: AeadKey::new(Aead::Aes128Gcm, vec![0u8; 16], &[0u8; 12]),
                rx_plain: Vec::new(),
                rx_raw: Vec::new(),
                closed: false,
                // Prefer the ALPN we parsed from ServerHello in
                // consume_server_hello (the TLS 1.2 driver doesn't
                // populate outcome.alpn — that field stays empty).
                alpn: if !hs.negotiated_alpn().is_empty() {
                    hs.negotiated_alpn().to_string()
                } else {
                    outcome.alpn
                },
                tls12: Some(Tls12RecordState {
                    client_write_key: outcome.client_write_key,
                    server_write_key: outcome.server_write_key,
                    client_write_salt: outcome.client_write_salt,
                    server_write_salt: outcome.server_write_salt,
                    seq_client: 1, // we already sent Finished as seq 0
                    seq_server: 1, // we already received Finished as seq 0
                    cipher_code: outcome.kind_code,
                }),
            });
        }

        // 3. Read encrypted records until we see server Finished.
        loop {
            let rec = read_one_record(&mut sock)?;
            match rec.content_type {
                // Server may send a dummy ChangeCipherSpec (TLS 1.2 compat). Ignore.
                t if t == ContentType::ChangeCipherSpec as u8 => continue,
                t if t == ContentType::ApplicationData as u8 => {
                    let mut header = [0u8; 5];
                    header[0] = ContentType::ApplicationData as u8;
                    header[1] = 0x03;
                    header[2] = 0x03;
                    let len = rec.fragment.len() as u16;
                    header[3..5].copy_from_slice(&len.to_be_bytes());
                    let (inner_type, inner) = hs
                        .server_handshake_aead
                        .as_mut()
                        .ok_or_else(|| TlsError::Protocol("no server hs keys".into()))?
                        .open_record(&header, &rec.fragment)
                        .map_err(|e| TlsError::Crypto(format!("open: {e}")))?;
                    if inner_type != ContentType::Handshake {
                        return Err(TlsError::Protocol(format!(
                            "expected encrypted handshake, got {inner_type:?}"
                        )));
                    }
                    if hs.consume_encrypted_handshake(&inner)? {
                        // server Finished done — break to send client Finished.
                        break;
                    }
                }
                t => {
                    return Err(TlsError::Protocol(format!(
                        "unexpected content type {t} in handshake"
                    )));
                }
            }
        }

        // 4. Send dummy ChangeCipherSpec for middlebox compatibility
        // (RFC 8446 §D.4).
        let ccs_record: [u8; 6] = [0x14, 0x03, 0x03, 0x00, 0x01, 0x01];
        sock.write_all(&ccs_record)
            .map_err(|e| TlsError::Protocol(format!("send CCS: {e:?}")))?;

        // 5. Send client Finished (encrypted with client handshake keys).
        let client_finished_record = hs.build_client_finished()?;
        sock.write_all(&client_finished_record)
            .map_err(|e| TlsError::Protocol(format!("send Fin: {e:?}")))?;

        // 6. Switch to application AEAD on both sides.
        let alpn = hs.negotiated_alpn().to_string();
        let (tx, rx) = hs.finalize_application_keys()?;
        let mut stream = Self {
            sock,
            rx,
            tx,
            rx_plain: Vec::new(),
            rx_raw: Vec::new(),
            closed: false,
            alpn,
            tls12: None,
        };

        // 7. Drain server post-handshake messages (NewSessionTicket) with
        // a short timeout. CRITICAL for Cloudflare-fronted hosts: their
        // BoringSSL record layer checks `tls_has_unprocessed_handshake_data`
        // BEFORE accepting any non-HANDSHAKE record. If we send our HTTP
        // GET so quickly that it lands in the same recv() as our Finished
        // (TCP can coalesce even with TCP_NODELAY when the round-trip is
        // short), the server hits that guard with our Finished still in
        // its hs_buf and replies with `unexpected_message`. By eagerly
        // reading what the server queued post-handshake (NSTs come first,
        // before any app data we'd care about) we force their state
        // machine to fully drain our Finished, so a subsequent app-data
        // record finds the buffer empty. Decrypted handshake records
        // (NSTs) are silently dropped here; their internal seq counter
        // does still advance via `open_record`.
        let mut scratch = [0u8; 8192];
        loop {
            match stream.sock.read_with_timeout(&mut scratch, 50) {
                Ok(0) => break, // timed out — no more queued data
                Ok(n) => {
                    stream.rx_raw.extend_from_slice(&scratch[..n]);
                    // Process all complete records in the buffer.
                    loop {
                        if stream.rx_raw.len() < 5 {
                            break;
                        }
                        let body_len =
                            u16::from_be_bytes([stream.rx_raw[3], stream.rx_raw[4]]) as usize;
                        let total = 5 + body_len;
                        if stream.rx_raw.len() < total {
                            break;
                        }
                        let mut header = [0u8; 5];
                        header.copy_from_slice(&stream.rx_raw[..5]);
                        let fragment = stream.rx_raw[5..total].to_vec();
                        if header[0] == ContentType::ChangeCipherSpec as u8 {
                            // Server middlebox-compat CCS — drop.
                            stream.rx_raw.drain(..total);
                            continue;
                        }
                        match stream.rx.open_record(&header, &fragment) {
                            Ok((inner_type, inner)) => {
                                stream.rx_raw.drain(..total);
                                if inner_type == ContentType::Alert && inner.len() >= 2 {
                                    let level = inner[0];
                                    let desc = inner[1];
                                    if level == 2 || desc != 0 {
                                        return Err(TlsError::Protocol(format!(
                                            "tls1.3 alert during drain lvl={} {} ({})",
                                            level,
                                            alert_description_name(desc),
                                            desc,
                                        )));
                                    }
                                    // close_notify — leave loop, surface
                                    // closure to caller via empty reads.
                                    stream.closed = true;
                                    break;
                                }
                                if inner_type == ContentType::ApplicationData {
                                    // Server coalesced an application-data record
                                    // (e.g. HTTP response) with its Finished in the
                                    // same TCP segment. Buffer the plaintext so the
                                    // first tls read() returns it immediately instead
                                    // of discarding it and hanging until timeout.
                                    stream.rx_plain.extend_from_slice(&inner);
                                    // Don't break — there may be more records in the
                                    // buffer (more coalesced app-data or NSTs).
                                } else if inner_type != ContentType::Handshake {
                                    break;
                                }
                            }
                            Err(_) => break,
                        }
                    }
                }
                Err(_) => break,
            }
        }

        Ok(stream)
    }

    /// Send application data. May split across multiple records if large.
    pub fn write_all(&mut self, data: &[u8]) -> Result<(), TlsError> {
        if self.tls12.is_some() {
            return self.write_all_tls12(data);
        }
        // RFC 8446 §5.2 — TLSPlaintext.length max 2^14. After AEAD add
        // a tag + content-type byte: keep payload under 2^14 - 17 to be safe.
        const MAX_PAYLOAD: usize = (1 << 14) - 17;
        for chunk in data.chunks(MAX_PAYLOAD) {
            let rec = self
                .tx
                .seal_record(ContentType::ApplicationData, chunk)
                .map_err(|e| TlsError::Crypto(format!("seal: {e}")))?;
            self.sock
                .write_all(&rec)
                .map_err(|e| TlsError::Protocol(format!("send app: {e:?}")))?;
        }
        Ok(())
    }

    /// TLS 1.2 AEAD application data — dispatches to whichever cipher
    /// was negotiated via `tls12.rs`. Record-layer overhead is 24 for
    /// AES-GCM (8-byte explicit nonce + 16-byte tag) or 16 for
    /// ChaCha20-Poly1305 (no explicit nonce per RFC 7905, just tag).
    fn write_all_tls12(&mut self, data: &[u8]) -> Result<(), TlsError> {
        const MAX_PAYLOAD: usize = (1 << 14) - 24;
        for chunk in data.chunks(MAX_PAYLOAD) {
            let st = self.tls12.as_mut().unwrap();
            let kind = super::tls12::tls12_kind_from_code(st.cipher_code).ok_or_else(|| {
                TlsError::Crypto(format!("tls1.2 bad cipher code {:04x}", st.cipher_code))
            })?;
            let frag = super::tls12::encrypt_app_record(
                kind,
                ContentType::ApplicationData as u8,
                chunk,
                &st.client_write_key,
                &st.client_write_salt,
                st.seq_client,
            )?;
            let rec = crate::tls::messages::wrap_record(ContentType::ApplicationData, &frag);
            self.sock
                .write_all(&rec)
                .map_err(|e| TlsError::Protocol(format!("tls1.2 send: {e:?}")))?;
            st.seq_client += 1;
        }
        Ok(())
    }

    pub fn read(&mut self, dst: &mut [u8]) -> Result<usize, TlsError> {
        while self.rx_plain.is_empty() && !self.closed {
            self.pull_one_record()?;
        }
        let n = self.rx_plain.len().min(dst.len());
        dst[..n].copy_from_slice(&self.rx_plain[..n]);
        self.rx_plain.drain(..n);
        Ok(n)
    }

    pub fn read_with_timeout(
        &mut self,
        dst: &mut [u8],
        timeout_ms: u32,
    ) -> Result<usize, TlsError> {
        while self.rx_plain.is_empty() && !self.closed {
            if !self.pull_one_record_with_timeout(timeout_ms)? {
                return Ok(0);
            }
        }
        let n = self.rx_plain.len().min(dst.len());
        dst[..n].copy_from_slice(&self.rx_plain[..n]);
        self.rx_plain.drain(..n);
        Ok(n)
    }

    pub fn read_to_end(&mut self, dst: &mut Vec<u8>) -> Result<(), TlsError> {
        while !self.closed {
            self.pull_one_record()?;
        }
        dst.append(&mut self.rx_plain);
        Ok(())
    }

    fn pull_one_record(&mut self) -> Result<(), TlsError> {
        // Read until we have a full record header + body.
        while self.rx_raw.len() < 5 {
            if !self.read_more()? {
                self.closed = true;
                return Ok(());
            }
        }
        let body_len = u16::from_be_bytes([self.rx_raw[3], self.rx_raw[4]]) as usize;
        let total = 5 + body_len;
        while self.rx_raw.len() < total {
            if !self.read_more()? {
                return Err(TlsError::Protocol("EOF mid-record".into()));
            }
        }
        let header: [u8; 5] = self.rx_raw[..5].try_into().unwrap();
        let fragment = self.rx_raw[5..total].to_vec();
        self.rx_raw.drain(..total);

        if header[0] == ContentType::ChangeCipherSpec as u8 {
            return Ok(()); // ignore
        }
        if header[0] != ContentType::ApplicationData as u8 {
            if header[0] == ContentType::Alert as u8 && self.tls12.is_some() {
                // TLS 1.2 alerts arrive encrypted under app data type
                // too, but a server can also send plaintext close_notify.
                self.closed = true;
                return Ok(());
            }
            return Err(TlsError::Protocol(format!(
                "unexpected post-handshake type {}",
                header[0]
            )));
        }
        if self.tls12.is_some() {
            return self.decrypt_app_record_tls12(&header, &fragment);
        }
        let (inner_type, mut inner) = self
            .rx
            .open_record(&header, &fragment)
            .map_err(|e| TlsError::Crypto(format!("open: {e}")))?;
        match inner_type {
            ContentType::ApplicationData => self.rx_plain.append(&mut inner),
            ContentType::Alert => {
                // Alert is `level(1) || description(1)`. close_notify
                // is level=1 (warning), desc=0; everything else fatal
                // we must SURFACE so the caller knows why the server
                // dropped us. We were silently setting `closed = true`
                // — that's what made thehindu's bot-management
                // rejection look like a clean EOF.
                self.closed = true;
                if inner.len() >= 2 {
                    let level = inner[0];
                    let desc = inner[1];
                    if level == 2 || desc != 0 {
                        return Err(TlsError::Protocol(format!(
                            "tls1.3 server alert lvl={} {} ({})",
                            level,
                            alert_description_name(desc),
                            desc,
                        )));
                    }
                }
            }
            ContentType::Handshake => {
                // Post-handshake messages (NewSessionTicket, KeyUpdate). Ignore.
            }
            _ => return Err(TlsError::Protocol(format!("inner type {inner_type:?}"))),
        }
        Ok(())
    }

    fn pull_one_record_with_timeout(&mut self, timeout_ms: u32) -> Result<bool, TlsError> {
        while self.rx_raw.len() < 5 {
            if !self.read_more_with_timeout(timeout_ms)? {
                self.closed = true;
                return Ok(false);
            }
        }
        let body_len = u16::from_be_bytes([self.rx_raw[3], self.rx_raw[4]]) as usize;
        let total = 5 + body_len;
        while self.rx_raw.len() < total {
            if !self.read_more_with_timeout(timeout_ms)? {
                return Ok(false);
            }
        }
        let header: [u8; 5] = self.rx_raw[..5].try_into().unwrap();
        let fragment = self.rx_raw[5..total].to_vec();
        self.rx_raw.drain(..total);

        if header[0] == ContentType::ChangeCipherSpec as u8 {
            return Ok(true);
        }
        if header[0] != ContentType::ApplicationData as u8 {
            if header[0] == ContentType::Alert as u8 && self.tls12.is_some() {
                self.closed = true;
                return Ok(false);
            }
            return Err(TlsError::Protocol(format!(
                "unexpected post-handshake type {}",
                header[0]
            )));
        }
        if self.tls12.is_some() {
            self.decrypt_app_record_tls12(&header, &fragment)?;
            return Ok(true);
        }
        let (inner_type, mut inner) = self
            .rx
            .open_record(&header, &fragment)
            .map_err(|e| TlsError::Crypto(format!("open: {e}")))?;
        match inner_type {
            ContentType::ApplicationData => self.rx_plain.append(&mut inner),
            ContentType::Alert => {
                self.closed = true;
                if inner.len() >= 2 {
                    let level = inner[0];
                    let desc = inner[1];
                    if level == 2 || desc != 0 {
                        return Err(TlsError::Protocol(format!(
                            "tls1.3 server alert lvl={} {} ({})",
                            level,
                            alert_description_name(desc),
                            desc,
                        )));
                    }
                }
                return Ok(false);
            }
            ContentType::Handshake => {}
            _ => return Err(TlsError::Protocol(format!("inner type {inner_type:?}"))),
        }
        Ok(true)
    }

    /// TLS 1.2 application-data record decrypt — dispatches via
    /// `tls12::decrypt_app_record` so AES-GCM-128/256 and
    /// ChaCha20-Poly1305 are all handled by the same code path that
    /// drove the Finished round-trip.
    fn decrypt_app_record_tls12(
        &mut self,
        _header: &[u8; 5],
        fragment: &[u8],
    ) -> Result<(), TlsError> {
        let st = self.tls12.as_mut().unwrap();
        let kind = super::tls12::tls12_kind_from_code(st.cipher_code).ok_or_else(|| {
            TlsError::Crypto(format!("tls1.2 bad cipher code {:04x}", st.cipher_code))
        })?;
        let mut plain = super::tls12::decrypt_app_record(
            kind,
            ContentType::ApplicationData as u8,
            fragment,
            &st.server_write_key,
            &st.server_write_salt,
            st.seq_server,
        )?;
        st.seq_server += 1;
        self.rx_plain.append(&mut plain);
        Ok(())
    }

    fn read_more(&mut self) -> Result<bool, TlsError> {
        let mut buf = [0u8; 16 * 1024];
        let n = self
            .sock
            .read(&mut buf)
            .map_err(|e| TlsError::Protocol(format!("read: {e:?}")))?;
        if n == 0 {
            return Ok(false);
        }
        self.rx_raw.extend_from_slice(&buf[..n]);
        Ok(true)
    }

    fn read_more_with_timeout(&mut self, timeout_ms: u32) -> Result<bool, TlsError> {
        let mut buf = [0u8; 16 * 1024];
        let n = self
            .sock
            .read_with_timeout(&mut buf, timeout_ms)
            .map_err(|e| TlsError::Protocol(format!("read timeout: {e:?}")))?;
        if n == 0 {
            return Ok(false);
        }
        self.rx_raw.extend_from_slice(&buf[..n]);
        Ok(true)
    }
}

/// Walk the ServerHello extensions to see whether the server echoed
/// `extended_master_secret` (RFC 7627). Strict scan: we re-parse the
/// HS message wrapper so we don't need to re-export ServerHello state.
fn server_hello_has_ems(fragment: &[u8]) -> bool {
    let iter = HandshakeIter::new(fragment);
    for msg in iter {
        let Ok((kind, body, _full)) = msg else {
            return false;
        };
        if kind != HandshakeType::ServerHello as u8 {
            continue;
        }
        // ServerHello layout: legacy_version(2) + random(32) +
        //   session_id<1..32> + cipher_suite(2) + compression_method(1)
        //   + extensions<2..>.
        if body.len() < 35 {
            return false;
        }
        let mut p = 34; // 2 (legacy_version) + 32 (random)
        let sid_len = body[p] as usize;
        p += 1 + sid_len;
        if p + 3 > body.len() {
            return false;
        }
        p += 2; // cipher_suite
        p += 1; // compression_method
        if p + 2 > body.len() {
            return false;
        }
        let ext_len = u16::from_be_bytes([body[p], body[p + 1]]) as usize;
        p += 2;
        if p + ext_len > body.len() {
            return false;
        }
        let mut e = p;
        while e + 4 <= p + ext_len {
            let etype = u16::from_be_bytes([body[e], body[e + 1]]);
            let edata_len = u16::from_be_bytes([body[e + 2], body[e + 3]]) as usize;
            if etype == 23
            /* extended_master_secret */
            {
                return true;
            }
            e += 4 + edata_len;
        }
        return false;
    }
    false
}

fn read_one_record(sock: &mut Socket) -> Result<OwnedRecord, TlsError> {
    let mut header = [0u8; 5];
    read_exact(sock, &mut header)?;
    let body_len = u16::from_be_bytes([header[3], header[4]]) as usize;
    let mut body = vec![0u8; body_len];
    read_exact(sock, &mut body)?;
    Ok(OwnedRecord {
        content_type: header[0],
        legacy_version: u16::from_be_bytes([header[1], header[2]]),
        fragment: body,
    })
}

fn read_exact(sock: &mut Socket, dst: &mut [u8]) -> Result<(), TlsError> {
    let mut off = 0;
    while off < dst.len() {
        let n = sock
            .read(&mut dst[off..])
            .map_err(|e| TlsError::Protocol(format!("read: {e:?}")))?;
        if n == 0 {
            return Err(TlsError::Protocol("unexpected EOF".into()));
        }
        off += n;
    }
    Ok(())
}

#[derive(Debug)]
struct OwnedRecord {
    content_type: u8,
    #[allow(dead_code)]
    legacy_version: u16,
    fragment: Vec<u8>,
}

/// The mutable handshake-time state. Discarded once `TlsStream` is built.
struct HandshakeDriver {
    host: String,
    client_priv: [u8; 32],
    client_pub: [u8; 32],
    /// P-256 ECDH ephemeral keypair. Generated alongside X25519 so we
    /// can offer both groups in a single ClientHello.
    client_priv_p256: [u8; 32],
    client_pub_p256: [u8; 65],
    /// P-384 ECDH ephemeral keypair. Some servers (PostgreSQL, certain
    /// banking sites) only allow secp384r1 ECDHE; we offer it up-front
    /// alongside X25519 and P-256 to skip HelloRetryRequest.
    client_priv_p384: [u8; 48],
    client_pub_p384: [u8; 97],
    client_random: [u8; 32],
    session_id: [u8; 32],
    transcript: Vec<u8>,

    cipher_suite: Option<CipherSuite>,
    aead: Option<Aead>,
    hash_alg: HashAlg,
    key_schedule: Option<KeySchedule>,

    // Set after ServerHello processing.
    client_handshake_aead: Option<AeadKey>,
    server_handshake_aead: Option<AeadKey>,
    client_handshake_secret: Vec<u8>,
    server_handshake_secret: Vec<u8>,

    // Captured during handshake.
    server_certs: Vec<Vec<u8>>,
    cert_verify_done: bool,
    server_finished_done: bool,
    /// Set when the server sent CertificateRequest during the handshake.
    /// Per RFC 8446 §4.4.2 the client MUST then respond with a Certificate
    /// message (even an empty one) AND a CertificateVerify (only if the
    /// Certificate was non-empty) BEFORE its Finished. We don't carry a
    /// client cert, so an empty Certificate goes out — but we must still
    /// send it or the server's `ssl_check_message_type(SSL3_MT_CERTIFICATE)`
    /// rejects our Finished with a fatal `unexpected_message` alert.
    /// thehindu.com / Cloudflare strict-bot-mode hosts do this.
    cert_request_seen: bool,
    /// CertificateRequest's `certificate_request_context` bytes — must be
    /// echoed verbatim in the client Certificate's same-named field.
    cert_request_context: Vec<u8>,
    /// Transcript hash at the point we send client Finished.
    /// Master secret + app keys derive from transcript through server Finished.
    transcript_through_server_finished: Vec<u8>,
    /// ALPN protocol the server picked from our ServerHello+EncryptedExtensions
    /// extension list. Empty when none was negotiated.
    alpn: String,
    /// Set when the server's ServerHello selects TLS 1.2 rather than 1.3.
    /// On that path we hand control off to `tls/tls12.rs` and skip the
    /// rest of the 1.3 state machine.
    tls12_selected: bool,
    /// Cipher suite as a raw u16 — populated on both 1.3 and 1.2 paths.
    /// The 1.3 path also fills `cipher_suite: Option<CipherSuite>`; on
    /// the 1.2 path that stays None because the suite (0xc02f) isn't one
    /// of the three 1.3 codepoints.
    cipher_suite_raw: u16,
    /// Server's ECDHE point as received in ServerHello key_share — only
    /// populated on the 1.3 path. Captured here so we can compute the
    /// shared secret in one place. (For 1.2 we never see a key_share in
    /// ServerHello — the ECDHE point is in ServerKeyExchange instead.)
    _phantom: (),
    /// Accumulator for encrypted-handshake plaintext bytes that haven't
    /// yet formed a complete handshake message. Meta's Certificate and
    /// some long Akamai cross-signed chains exceed the 16 KiB record
    /// size and arrive in multiple TLS records; we coalesce here so
    /// `consume_encrypted_handshake` only walks complete messages.
    pending_hs_buf: Vec<u8>,
}

impl HandshakeDriver {
    pub fn negotiated_alpn(&self) -> &str {
        &self.alpn
    }
}

impl HandshakeDriver {
    fn new(host: &str) -> Result<Self, TlsError> {
        let priv_bytes = rand_bytes(32);
        let mut client_priv = [0u8; 32];
        client_priv.copy_from_slice(&priv_bytes);
        let client_pub = x25519_public(&client_priv);
        // P-256 ephemeral keypair. The private scalar must be in
        // [1, n-1]; loop until rand_bytes lands in range. Tight loop
        // because the chance of failure on a single 256-bit draw is
        // negligibly small (~1/2^128).
        let mut client_priv_p256 = [0u8; 32];
        let client_pub_p256 = loop {
            let seed = rand_bytes(32);
            client_priv_p256.copy_from_slice(&seed);
            match cv_crypto::p256::public_key_uncompressed(&client_priv_p256) {
                Ok(pk) => break pk,
                Err(_) => continue,
            }
        };
        // P-384 ephemeral keypair, same loop-until-in-range pattern as
        // P-256. 48-byte scalar; chance of failure per draw ~ 1/2^192.
        let mut client_priv_p384 = [0u8; 48];
        let client_pub_p384 = loop {
            let seed = rand_bytes(48);
            client_priv_p384.copy_from_slice(&seed);
            match cv_crypto::p384::public_key_uncompressed(&client_priv_p384) {
                Ok(pk) => break pk,
                Err(_) => continue,
            }
        };
        let mut client_random = [0u8; 32];
        client_random.copy_from_slice(&rand_bytes(32));
        let mut session_id = [0u8; 32];
        session_id.copy_from_slice(&rand_bytes(32));

        Ok(Self {
            host: host.to_string(),
            client_priv,
            client_pub,
            client_priv_p256,
            client_pub_p256,
            client_priv_p384,
            client_pub_p384,
            client_random,
            session_id,
            transcript: Vec::new(),
            cipher_suite: None,
            aead: None,
            hash_alg: HashAlg::Sha256,
            key_schedule: None,
            client_handshake_aead: None,
            server_handshake_aead: None,
            client_handshake_secret: Vec::new(),
            server_handshake_secret: Vec::new(),
            server_certs: Vec::new(),
            cert_verify_done: false,
            server_finished_done: false,
            cert_request_seen: false,
            cert_request_context: Vec::new(),
            transcript_through_server_finished: Vec::new(),
            alpn: String::new(),
            tls12_selected: false,
            cipher_suite_raw: 0,
            _phantom: (),
            pending_hs_buf: Vec::new(),
        })
    }

    fn build_client_hello(&mut self) -> Result<Vec<u8>, TlsError> {
        let body = build_client_hello_body(
            &self.client_random,
            &self.session_id,
            &self.client_pub,
            &self.client_pub_p256,
            &self.client_pub_p384,
            &self.host,
        );
        let hs = wrap_handshake(HandshakeType::ClientHello, body);
        self.transcript.extend_from_slice(&hs);
        Ok(wrap_record(ContentType::Handshake, &hs))
    }

    fn consume_server_hello(&mut self, fragment: &[u8]) -> Result<(), TlsError> {
        // Fragment is one (or more) handshake messages.
        let mut iter = HandshakeIter::new(fragment);
        let first = iter
            .next()
            .ok_or_else(|| TlsError::Protocol("empty SH record".into()))?
            .map_err(|e| TlsError::Decode(e.to_string()))?;
        let (kind, body, full) = first;
        if kind != HandshakeType::ServerHello as u8 {
            return Err(TlsError::Protocol(format!("expected SH, got {kind}")));
        }
        self.transcript.extend_from_slice(full);

        let sh = parse_server_hello(body).map_err(|e| TlsError::Decode(e.to_string()))?;
        self.cipher_suite_raw = sh.cipher_suite;
        // TLS 1.2 detection: server didn't echo `supported_versions`
        // (so `selected_version` is None) and `legacy_version` is 0x0303.
        // The TLS 1.3 path requires `supported_versions: 0x0304`.
        let server_picked_tls12 = match sh.selected_version {
            None => sh.legacy_version == 0x0303,
            Some(v) => v == 0x0303,
        };
        // Capture ALPN regardless of selected version — TLS 1.2 echoes
        // it here in ServerHello; TLS 1.3 will overwrite via
        // EncryptedExtensions later. Without this 1.2 connections lost
        // ALPN entirely and the http1 fast-path tried to send HTTP/1.1
        // on h2-negotiated sockets, getting binary frames back.
        if let Some(name) = &sh.alpn {
            self.alpn = name.clone();
        }
        if server_picked_tls12 {
            // Hand the rest of the handshake off to the 1.2 driver. We've
            // already added ServerHello to the transcript; the 1.2 driver
            // will keep extending it for SKE / CKE / Finished.
            self.tls12_selected = true;
            return Ok(());
        }
        if sh.selected_version != Some(TLS13_REAL_VERSION) {
            return Err(TlsError::UnsupportedServerVersion(
                sh.selected_version.unwrap_or(sh.legacy_version),
            ));
        }
        self.cipher_suite = Some(match sh.cipher_suite {
            x if x == CipherSuite::Aes128GcmSha256 as u16 => CipherSuite::Aes128GcmSha256,
            x if x == CipherSuite::Aes256GcmSha384 as u16 => CipherSuite::Aes256GcmSha384,
            x if x == CipherSuite::ChaCha20Poly1305Sha256 as u16 => {
                CipherSuite::ChaCha20Poly1305Sha256
            }
            other => return Err(TlsError::UnsupportedCipherSuite(other)),
        });
        let (aead, hash_alg) = match self.cipher_suite.unwrap() {
            CipherSuite::Aes128GcmSha256 => (Aead::Aes128Gcm, HashAlg::Sha256),
            CipherSuite::ChaCha20Poly1305Sha256 => (Aead::ChaCha20Poly1305, HashAlg::Sha256),
            CipherSuite::Aes256GcmSha384 => (Aead::Aes256Gcm, HashAlg::Sha384),
        };
        self.aead = Some(aead);
        self.hash_alg = hash_alg;

        let group = sh
            .key_share_group
            .ok_or_else(|| TlsError::Protocol("no key_share in SH".into()))?;
        let ke = sh
            .key_share_key_exchange
            .ok_or_else(|| TlsError::Protocol("no key_share KE".into()))?;
        let shared = if group == NamedGroup::X25519 as u16 {
            if ke.len() != 32 {
                return Err(TlsError::Protocol(format!(
                    "x25519 KE wrong len {}",
                    ke.len()
                )));
            }
            let mut server_pub = [0u8; 32];
            server_pub.copy_from_slice(ke);
            x25519(&self.client_priv, &server_pub).to_vec()
        } else if group == NamedGroup::Secp256r1 as u16 {
            // SEC1 uncompressed: 0x04 || X(32) || Y(32) = 65 bytes.
            if ke.len() != 65 || ke[0] != 0x04 {
                return Err(TlsError::Protocol(format!(
                    "secp256r1 KE wrong format len={} byte0={:#x}",
                    ke.len(),
                    ke.first().copied().unwrap_or(0)
                )));
            }
            let s = cv_crypto::p256::ecdh_shared(&self.client_priv_p256, ke)
                .map_err(|_| TlsError::Protocol("p256 ecdh failed".into()))?;
            s.to_vec()
        } else if group == NamedGroup::Secp384r1 as u16 {
            // SEC1 uncompressed: 0x04 || X(48) || Y(48) = 97 bytes.
            if ke.len() != 97 || ke[0] != 0x04 {
                return Err(TlsError::Protocol(format!(
                    "secp384r1 KE wrong format len={} byte0={:#x}",
                    ke.len(),
                    ke.first().copied().unwrap_or(0)
                )));
            }
            let s = cv_crypto::p384::ecdh_shared(&self.client_priv_p384, ke)
                .map_err(|_| TlsError::Protocol("p384 ecdh failed".into()))?;
            s.to_vec()
        } else {
            return Err(TlsError::UnsupportedKeyShareGroup(group));
        };

        // Key schedule advance to handshake secret.
        let mut ks = KeySchedule::new_no_psk(self.hash_alg);
        ks.advance_to_handshake(&shared);
        self.client_handshake_secret = ks.client_handshake_traffic_secret(&self.transcript);
        self.server_handshake_secret = ks.server_handshake_traffic_secret(&self.transcript);

        let tk_c = traffic_keys(
            self.hash_alg,
            &self.client_handshake_secret,
            aead.key_len(),
            aead.iv_len(),
        );
        let tk_s = traffic_keys(
            self.hash_alg,
            &self.server_handshake_secret,
            aead.key_len(),
            aead.iv_len(),
        );
        self.client_handshake_aead = Some(AeadKey::new(aead, tk_c.key, &tk_c.iv));
        self.server_handshake_aead = Some(AeadKey::new(aead, tk_s.key, &tk_s.iv));

        self.key_schedule = Some(ks);
        Ok(())
    }

    /// Process a decrypted inner fragment of handshake messages. Returns
    /// `Ok(true)` after server Finished is consumed.
    ///
    /// Large handshake messages (Certificate from servers with cross-signed
    /// chains — Meta, Akamai, some Cloudflare edges) routinely exceed the
    /// 16 KiB TLS record size and arrive split across two or three
    /// encrypted records. Each record decrypts to plaintext that, on its
    /// own, contains only a partial handshake message and won't parse.
    /// To handle that we accumulate plaintext in a per-driver buffer and
    /// only drain complete (`type` + `u24 length` + `body`) messages,
    /// leaving any trailing partial header/body for the next record.
    fn consume_encrypted_handshake(&mut self, fragment: &[u8]) -> Result<bool, TlsError> {
        self.pending_hs_buf.extend_from_slice(fragment);
        // Drain as many complete handshake messages as the buffer holds.
        // Hold ownership of the buffer across the loop so we can splice
        // out consumed bytes at the end.
        let mut buf = std::mem::take(&mut self.pending_hs_buf);
        let mut messages: Vec<(u8, Vec<u8>, Vec<u8>)> = Vec::new();
        let mut consumed = 0usize;
        while consumed + 4 <= buf.len() {
            let len_u24 = ((buf[consumed + 1] as usize) << 16)
                | ((buf[consumed + 2] as usize) << 8)
                | (buf[consumed + 3] as usize);
            let total = 4 + len_u24;
            if consumed + total > buf.len() {
                break;
            }
            let kind = buf[consumed];
            let body = buf[consumed + 4..consumed + total].to_vec();
            let full = buf[consumed..consumed + total].to_vec();
            messages.push((kind, body, full));
            consumed += total;
        }
        buf.drain(..consumed);
        self.pending_hs_buf = buf;

        for msg in messages
            .into_iter()
            .map(Ok::<_, crate::tls::messages::DecodeError>)
        {
            let (kind, body, full) = msg.map_err(|e| TlsError::Decode(e.to_string()))?;
            let body = body.as_slice();
            let full = full.as_slice();
            match kind {
                k if k == HandshakeType::EncryptedExtensions as u8 => {
                    self.transcript.extend_from_slice(full);
                    // Walk the extensions list and capture ALPN.
                    // EncryptedExtensions body: <u16 ext_list_len><ext*>.
                    // Each ext: <u16 type><u16 len><body>.
                    if body.len() >= 2 {
                        let ext_total = u16::from_be_bytes([body[0], body[1]]) as usize;
                        let mut i = 2usize;
                        let end = (2 + ext_total).min(body.len());
                        while i + 4 <= end {
                            let ty = u16::from_be_bytes([body[i], body[i + 1]]);
                            let len = u16::from_be_bytes([body[i + 2], body[i + 3]]) as usize;
                            if i + 4 + len > end {
                                break;
                            }
                            let ext_body = &body[i + 4..i + 4 + len];
                            // 0x0010 = ALPN. Body is
                            //   <u16 list_len><proto*>; proto = <u8 len><name bytes>.
                            // We take the first listed proto.
                            if ty == 0x0010 && ext_body.len() >= 3 {
                                let proto_len = ext_body[2] as usize;
                                if 3 + proto_len <= ext_body.len() {
                                    if let Ok(name) =
                                        core::str::from_utf8(&ext_body[3..3 + proto_len])
                                    {
                                        self.alpn = name.to_string();
                                    }
                                }
                            }
                            i += 4 + len;
                        }
                    }
                }
                k if k == HandshakeType::Certificate as u8 => {
                    self.parse_certificate(body)?;
                    self.transcript.extend_from_slice(full);
                }
                k if k == HandshakeType::CertificateVerify as u8 => {
                    self.verify_certificate_verify(body)?;
                    self.transcript.extend_from_slice(full);
                    self.cert_verify_done = true;
                }
                k if k == HandshakeType::Finished as u8 => {
                    self.verify_server_finished(body)?;
                    self.transcript.extend_from_slice(full);
                    // Snapshot for app secret derivation.
                    self.transcript_through_server_finished = self.transcript.clone();
                    self.server_finished_done = true;
                    return Ok(true);
                }
                k if k == HandshakeType::CertificateRequest as u8 => {
                    // Per RFC 8446 §4.3.2: CertificateRequest body is
                    //   opaque certificate_request_context<0..2^8-1>;
                    //   Extension extensions<2..2^16-1>;
                    // We MUST echo certificate_request_context verbatim
                    // in our client Certificate response.
                    if !body.is_empty() {
                        let ctx_len = body[0] as usize;
                        if body.len() >= 1 + ctx_len {
                            self.cert_request_context = body[1..1 + ctx_len].to_vec();
                        }
                    }
                    self.cert_request_seen = true;
                    self.transcript.extend_from_slice(full);
                }
                k if k == HandshakeType::NewSessionTicket as u8 => {
                    // Post-handshake; shouldn't arrive here but tolerate.
                    self.transcript.extend_from_slice(full);
                }
                25 => {
                    // CompressedCertificate (RFC 8879 §4) — server is
                    // honouring the `compress_certificate: brotli` we
                    // advertised. Decompress and treat as Certificate.
                    //   algorithm(2) | uncompressed_length(3) | compressed_body<...>
                    if body.len() < 5 {
                        return Err(TlsError::Decode("compr cert: short".into()));
                    }
                    let algo = u16::from_be_bytes([body[0], body[1]]);
                    let uncompressed_len =
                        ((body[2] as usize) << 16) | ((body[3] as usize) << 8) | body[4] as usize;
                    if 5 + 3 > body.len() {
                        return Err(TlsError::Decode("compr cert: no compr len".into()));
                    }
                    // The next 3 bytes are an inner length of the compressed payload.
                    let comp_len =
                        ((body[5] as usize) << 16) | ((body[6] as usize) << 8) | body[7] as usize;
                    if 8 + comp_len > body.len() {
                        return Err(TlsError::Decode("compr cert: body trunc".into()));
                    }
                    let comp = &body[8..8 + comp_len];
                    let decompressed = match algo {
                        2 /* brotli */ => cv_compression::brotli::decode_brotli(comp)
                            .map_err(|e| TlsError::Decode(format!("brotli: {e:?}")))?,
                        other => return Err(TlsError::Protocol(format!(
                            "compr cert: unsupported algo {other}"
                        ))),
                    };
                    if decompressed.len() != uncompressed_len {
                        return Err(TlsError::Decode(format!(
                            "compr cert: len mismatch {} vs {}",
                            decompressed.len(),
                            uncompressed_len
                        )));
                    }
                    // The decompressed payload is the *body* of a plain
                    // Certificate message — feed it through the regular
                    // parser. Transcript hash MUST use the original
                    // (compressed) message bytes per RFC 8879 §5.
                    self.parse_certificate(&decompressed)?;
                    self.transcript.extend_from_slice(full);
                }
                other => {
                    return Err(TlsError::Protocol(format!(
                        "unexpected handshake type {other}"
                    )));
                }
            }
        }
        Ok(false)
    }

    fn parse_certificate(&mut self, body: &[u8]) -> Result<(), TlsError> {
        // Certificate message body:
        //   opaque certificate_request_context<0..2^8-1>;
        //   CertificateEntry certificate_list<0..2^24-1>;
        //   CertificateEntry { cert_data<1..2^24-1>; extensions<0..2^16-1>; }
        let mut d = Decoder::new(body);
        let _ctx = d
            .vec_u8()
            .map_err(|e| TlsError::Decode(format!("cert ctx: {e}")))?;
        let cert_list = d
            .vec_u24()
            .map_err(|e| TlsError::Decode(format!("cert list: {e}")))?;
        let mut entries = Decoder::new(cert_list);
        while !entries.is_empty() {
            let cert = entries
                .vec_u24()
                .map_err(|e| TlsError::Decode(format!("cert entry: {e}")))?;
            let _ext = entries
                .vec_u16()
                .map_err(|e| TlsError::Decode(format!("cert ext: {e}")))?;
            self.server_certs.push(cert.to_vec());
        }
        if self.server_certs.is_empty() {
            return Err(TlsError::NoCertificate);
        }
        Ok(())
    }

    fn verify_certificate_verify(&mut self, body: &[u8]) -> Result<(), TlsError> {
        let mut d = Decoder::new(body);
        let scheme = d
            .u16()
            .map_err(|e| TlsError::Decode(format!("cv scheme: {e}")))?;
        let sig = d
            .vec_u16()
            .map_err(|e| TlsError::Decode(format!("cv sig: {e}")))?;

        // Anchor the presented chain at a trusted Windows root and verify
        // EKU + hostname + revocation BEFORE trusting any cert in it. If
        // this fails we must not even consider the leaf's signature on the
        // transcript, because the leaf itself is untrusted.
        chain_validate::verify_chain(&self.server_certs, &self.host)
            .map_err(|e| TlsError::ChainInvalid(e.to_string()))?;

        // Transcript hash UP TO BUT NOT INCLUDING this CertVerify message.
        // self.transcript currently does not include this message yet.
        // The hash MUST match the cipher suite — TLS 1.3 binds the
        // CertificateVerify signature to a digest using the suite's
        // hash. Hard-coding SHA-256 broke verify whenever the server
        // selected an SHA-384 cipher suite.
        let th = self.hash_alg.hash(&self.transcript);

        // Build signed-content per RFC 8446 §4.4.3.
        let mut signed = Vec::with_capacity(64 + 33 + 1 + th.len());
        signed.extend(std::iter::repeat_n(0x20, 64));
        signed.extend_from_slice(b"TLS 1.3, server CertificateVerify");
        signed.push(0x00);
        signed.extend_from_slice(&th);

        // Pull RSA public key from leaf cert.
        let leaf = x509::parse(&self.server_certs[0])
            .map_err(|e| TlsError::Decode(format!("leaf cert: {e}")))?;

        // Hostname check.
        if !cert_hostname_matches(&leaf, &self.host) {
            return Err(TlsError::Protocol(format!(
                "leaf cert does not match host {}",
                self.host
            )));
        }

        let scheme_enum = match scheme {
            x if x == SignatureScheme::RsaPssRsaeSha256 as u16 => SignatureScheme::RsaPssRsaeSha256,
            x if x == SignatureScheme::RsaPssRsaeSha384 as u16 => SignatureScheme::RsaPssRsaeSha384,
            x if x == SignatureScheme::RsaPssRsaeSha512 as u16 => SignatureScheme::RsaPssRsaeSha512,
            x if x == SignatureScheme::RsaPkcs1Sha256 as u16 => SignatureScheme::RsaPkcs1Sha256,
            x if x == SignatureScheme::EcdsaSecp256r1Sha256 as u16 => {
                SignatureScheme::EcdsaSecp256r1Sha256
            }
            x if x == SignatureScheme::EcdsaSecp384r1Sha384 as u16 => {
                SignatureScheme::EcdsaSecp384r1Sha384
            }
            x if x == SignatureScheme::EcdsaSecp521r1Sha512 as u16 => {
                SignatureScheme::EcdsaSecp521r1Sha512
            }
            other => return Err(TlsError::UnsupportedSignatureScheme(other)),
        };

        match (scheme_enum, leaf.spki_alg) {
            (
                SignatureScheme::RsaPssRsaeSha256
                | SignatureScheme::RsaPssRsaeSha384
                | SignatureScheme::RsaPssRsaeSha512
                | SignatureScheme::RsaPkcs1Sha256,
                cv_crypto::x509::SpkiAlgorithm::Rsa,
            ) => {
                // SPKI carries RSAPublicKey ::= SEQUENCE { n INTEGER, e INTEGER }.
                let mut spki = AsnReader::new(leaf.spki_key_bytes);
                let mut rsa_seq = spki
                    .read_sequence()
                    .map_err(|e| TlsError::Decode(format!("rsa pk: {e}")))?;
                let n = rsa_seq
                    .read_integer_unsigned_bytes()
                    .map_err(|e| TlsError::Decode(format!("rsa n: {e}")))?;
                let e = rsa_seq
                    .read_integer_unsigned_bytes()
                    .map_err(|e| TlsError::Decode(format!("rsa e: {e}")))?;
                let key = RsaPublicKey::from_components(n, e);
                let res = match scheme_enum {
                    SignatureScheme::RsaPssRsaeSha256 => {
                        verify_pss(&key, RsaHash::Sha256, &signed, sig)
                    }
                    SignatureScheme::RsaPssRsaeSha384 => {
                        verify_pss(&key, RsaHash::Sha384, &signed, sig)
                    }
                    SignatureScheme::RsaPssRsaeSha512 => {
                        verify_pss(&key, RsaHash::Sha512, &signed, sig)
                    }
                    _ => verify_pkcs1_v15(&key, RsaHash::Sha256, &signed, sig),
                };
                res.map_err(|_| TlsError::CertVerifyFailed)?;
            }
            (SignatureScheme::EcdsaSecp256r1Sha256, cv_crypto::x509::SpkiAlgorithm::EcP256) => {
                // SPKI bit string is uncompressed point: 0x04 || X(32) || Y(32).
                let pk = leaf.spki_key_bytes;
                if pk.len() != 65 || pk[0] != 0x04 {
                    return Err(TlsError::Protocol(format!(
                        "unexpected EC pubkey format len={} byte0={:#x}",
                        pk.len(),
                        pk.first().copied().unwrap_or(0)
                    )));
                }
                let qx = &pk[1..33];
                let qy = &pk[33..65];
                let (r_bytes, s_bytes) =
                    p256::parse_der_signature(sig).map_err(|_| TlsError::CertVerifyFailed)?;
                p256::verify(qx, qy, &signed, &r_bytes, &s_bytes)
                    .map_err(|_| TlsError::CertVerifyFailed)?;
            }
            (SignatureScheme::EcdsaSecp384r1Sha384, cv_crypto::x509::SpkiAlgorithm::EcP256) => {
                // P-384 uncompressed point: 0x04 || X(48) || Y(48) = 97 bytes.
                let pk = leaf.spki_key_bytes;
                if pk.len() != 97 || pk[0] != 0x04 {
                    return Err(TlsError::Protocol(format!(
                        "unexpected P-384 pubkey format len={} byte0={:#x}",
                        pk.len(),
                        pk.first().copied().unwrap_or(0)
                    )));
                }
                let qx = &pk[1..49];
                let qy = &pk[49..97];
                let (r_bytes, s_bytes) = cv_crypto::p384::parse_der_signature(sig)
                    .map_err(|_| TlsError::CertVerifyFailed)?;
                cv_crypto::p384::verify(qx, qy, &signed, &r_bytes, &s_bytes)
                    .map_err(|_| TlsError::CertVerifyFailed)?;
            }
            (SignatureScheme::EcdsaSecp521r1Sha512, cv_crypto::x509::SpkiAlgorithm::EcP256) => {
                // The SpkiAlgorithm enum lumps every NIST EC curve under
                // `EcP256` (the OID `1.2.840.10045.2.1` is generic
                // ecPublicKey; the curve is in the parameter we don't
                // currently parse). When the signature scheme is
                // secp521r1+SHA-512, the SPKI bit string holds a
                // 0x04 || X(66) || Y(66) = 133-byte uncompressed point
                // instead of a 65-byte P-256 point. Detect by length and
                // dispatch to the P-521 verify path.
                let pk = leaf.spki_key_bytes;
                if pk.len() != 133 || pk[0] != 0x04 {
                    return Err(TlsError::Protocol(format!(
                        "unexpected P-521 pubkey format len={} byte0={:#x}",
                        pk.len(),
                        pk.first().copied().unwrap_or(0)
                    )));
                }
                let qx = &pk[1..67];
                let qy = &pk[67..133];
                let (r_bytes, s_bytes) = cv_crypto::p521::parse_der_signature(sig)
                    .map_err(|_| TlsError::CertVerifyFailed)?;
                cv_crypto::p521::verify(qx, qy, &signed, &r_bytes, &s_bytes)
                    .map_err(|_| TlsError::CertVerifyFailed)?;
            }
            (sch, alg) => {
                return Err(TlsError::Protocol(format!(
                    "sig scheme {sch:?} incompatible with leaf SPKI {alg:?}"
                )));
            }
        }

        Ok(())
    }

    fn verify_server_finished(&self, body: &[u8]) -> Result<(), TlsError> {
        let h_len = self.hash_alg.output_len();
        if body.len() != h_len {
            return Err(TlsError::Protocol(format!(
                "Finished wrong len {}",
                body.len()
            )));
        }
        let finished_key = hkdf_expand_label(
            self.hash_alg,
            &self.server_handshake_secret,
            b"finished",
            b"",
            h_len as u16,
        );
        // Transcript hash up to (but not including) the Finished itself.
        let th = self.hash_alg.hash(&self.transcript);
        let expected = self.hash_alg.hmac(&finished_key, &th);
        if !cv_crypto::subtle::ct_eq(&expected, body) {
            return Err(TlsError::ServerFinishedMismatch);
        }
        Ok(())
    }

    fn build_client_finished(&mut self) -> Result<Vec<u8>, TlsError> {
        // If the server sent CertificateRequest, RFC 8446 §4.4.2 requires
        // us to first send a Certificate message (even an empty one) so
        // its `ssl_check_message_type(SSL3_MT_CERTIFICATE)` passes before
        // it expects Finished. The Certificate goes into the transcript
        // and the Finished's verify_data covers the updated transcript.
        let mut wire = Vec::new();
        if self.cert_request_seen {
            // TLS 1.3 Certificate body:
            //   opaque certificate_request_context<0..2^8-1>;
            //   CertificateEntry certificate_list<0..2^24-1>;
            // We send an empty list (no client cert) and echo back the
            // certificate_request_context the server gave us.
            let mut body = Vec::new();
            body.push(self.cert_request_context.len() as u8);
            body.extend_from_slice(&self.cert_request_context);
            // u24 length of empty certificate_list = 0
            body.push(0);
            body.push(0);
            body.push(0);
            let cert_msg = wrap_handshake(HandshakeType::Certificate, body);
            self.transcript.extend_from_slice(&cert_msg);
            let aead = self
                .client_handshake_aead
                .as_mut()
                .ok_or_else(|| TlsError::Protocol("no client hs keys".into()))?;
            let cert_record = aead
                .seal_record(ContentType::Handshake, &cert_msg)
                .map_err(|e| TlsError::Crypto(format!("seal cert: {e}")))?;
            wire.extend_from_slice(&cert_record);
            // No CertificateVerify — that's only required when the
            // Certificate message contains an actual cert. Per spec a
            // client that sends an empty Certificate skips CertVerify.
        }

        let h_len = self.hash_alg.output_len();
        let finished_key = hkdf_expand_label(
            self.hash_alg,
            &self.client_handshake_secret,
            b"finished",
            b"",
            h_len as u16,
        );
        // transcript NOW includes server Finished + (optional) our
        // empty client Certificate.
        let th = self.hash_alg.hash(&self.transcript);
        let verify_data = self.hash_alg.hmac(&finished_key, &th);

        let body = verify_data.to_vec();
        let hs = wrap_handshake(HandshakeType::Finished, body);

        let aead = self
            .client_handshake_aead
            .as_mut()
            .ok_or_else(|| TlsError::Protocol("no client hs keys".into()))?;
        let record = aead
            .seal_record(ContentType::Handshake, &hs)
            .map_err(|e| TlsError::Crypto(format!("seal Fin: {e}")))?;
        wire.extend_from_slice(&record);
        Ok(wire)
    }

    /// After client Finished is sent: derive application traffic secrets,
    /// build the AEAD keys for the post-handshake phase, return
    /// `(client_send_key, server_recv_key)`.
    fn finalize_application_keys(mut self) -> Result<(AeadKey, AeadKey), TlsError> {
        let mut ks = self
            .key_schedule
            .take()
            .ok_or_else(|| TlsError::Protocol("no key schedule".into()))?;
        ks.advance_to_master();
        let aead = self.aead.unwrap();
        let alg = self.hash_alg;
        let through_sf = &self.transcript_through_server_finished;
        let c_app = ks.client_application_traffic_secret(through_sf);
        let s_app = ks.server_application_traffic_secret(through_sf);
        let tk_c = traffic_keys(alg, &c_app, aead.key_len(), aead.iv_len());
        let tk_s = traffic_keys(alg, &s_app, aead.key_len(), aead.iv_len());
        let tx = AeadKey::new(aead, tk_c.key, &tk_c.iv);
        let rx = AeadKey::new(aead, tk_s.key, &tk_s.iv);
        Ok((tx, rx))
    }
}

/// Map a TLS Alert AlertDescription byte to its IANA name, per
/// RFC 8446 §6.2 + RFC 8446bis. The text is what tells us *why* a
/// server rejected ClientHello when we get an alert in place of
/// ServerHello — `handshake_failure` means cipher/curve/version
/// negotiation, `protocol_version` means we offered a version they
/// don't support, `unrecognized_name` means our SNI was wrong, etc.
fn alert_description_name(code: u8) -> &'static str {
    match code {
        0 => "close_notify",
        10 => "unexpected_message",
        20 => "bad_record_mac",
        21 => "decryption_failed_RESERVED",
        22 => "record_overflow",
        30 => "decompression_failure_RESERVED",
        40 => "handshake_failure",
        41 => "no_certificate_RESERVED",
        42 => "bad_certificate",
        43 => "unsupported_certificate",
        44 => "certificate_revoked",
        45 => "certificate_expired",
        46 => "certificate_unknown",
        47 => "illegal_parameter",
        48 => "unknown_ca",
        49 => "access_denied",
        50 => "decode_error",
        51 => "decrypt_error",
        60 => "export_restriction_RESERVED",
        70 => "protocol_version",
        71 => "insufficient_security",
        80 => "internal_error",
        86 => "inappropriate_fallback",
        90 => "user_canceled",
        100 => "no_renegotiation_RESERVED",
        109 => "missing_extension",
        110 => "unsupported_extension",
        111 => "certificate_unobtainable_RESERVED",
        112 => "unrecognized_name",
        113 => "bad_certificate_status_response",
        114 => "bad_certificate_hash_value_RESERVED",
        115 => "unknown_psk_identity",
        116 => "certificate_required",
        120 => "no_application_protocol",
        _ => "unknown",
    }
}

fn cert_hostname_matches(cert: &Cert<'_>, host: &str) -> bool {
    for name in &cert.san_dns {
        if x509::hostname_matches(host, name) {
            return true;
        }
    }
    if cert.san_dns.is_empty() {
        if let Some(cn) = &cert.subject_cn {
            return x509::hostname_matches(host, cn);
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn handshake_driver_init() {
        let hs = HandshakeDriver::new("example.com").unwrap();
        assert_eq!(hs.client_pub.len(), 32);
        // Random bytes should not be zero (statistically safe).
        assert!(hs.client_random.iter().any(|&b| b != 0));
    }
}
