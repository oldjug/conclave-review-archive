//! TLS 1.3 handshake message types and wire encoding per RFC 8446 §4.
//!
//! Today we only serialize ClientHello (we send it) and parse a handful
//! of server messages. Full message coverage lands as the state machine
//! grows.

use core::fmt;

pub const TLS13_LEGACY_VERSION: u16 = 0x0303; // TLS 1.2 — TLS 1.3 hides under this
pub const TLS13_REAL_VERSION: u16 = 0x0304;
/// Alias for clarity — when a server's `supported_versions` selection is
/// 0x0303 it really did pick TLS 1.2, not a legacy framing.
pub const TLS12_VERSION: u16 = 0x0303;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum ContentType {
    ChangeCipherSpec = 20,
    Alert = 21,
    Handshake = 22,
    ApplicationData = 23,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum HandshakeType {
    ClientHello = 1,
    ServerHello = 2,
    NewSessionTicket = 4,
    EndOfEarlyData = 5,
    EncryptedExtensions = 8,
    Certificate = 11,
    /// TLS 1.2 only.
    ServerKeyExchange = 12,
    CertificateRequest = 13,
    /// TLS 1.2 only.
    ServerHelloDone = 14,
    CertificateVerify = 15,
    /// TLS 1.2 only.
    ClientKeyExchange = 16,
    Finished = 20,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(u16)]
pub enum CipherSuite {
    Aes128GcmSha256 = 0x1301,
    Aes256GcmSha384 = 0x1302,
    ChaCha20Poly1305Sha256 = 0x1303,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(u16)]
pub enum NamedGroup {
    X25519 = 0x001d,
    Secp256r1 = 0x0017,
    Secp384r1 = 0x0018,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(u16)]
pub enum SignatureScheme {
    RsaPkcs1Sha256 = 0x0401,
    RsaPkcs1Sha384 = 0x0501,
    RsaPkcs1Sha512 = 0x0601,
    EcdsaSecp256r1Sha256 = 0x0403,
    EcdsaSecp384r1Sha384 = 0x0503,
    EcdsaSecp521r1Sha512 = 0x0603,
    RsaPssRsaeSha256 = 0x0804,
    RsaPssRsaeSha384 = 0x0805,
    RsaPssRsaeSha512 = 0x0806,
    Ed25519 = 0x0807,
    Ed448 = 0x0808,
    RsaPssPssSha256 = 0x0809,
    RsaPssPssSha384 = 0x080a,
    RsaPssPssSha512 = 0x080b,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(u16)]
pub enum ExtensionType {
    ServerName = 0,
    /// status_request (RFC 6066 §8) — OCSP stapling request.
    StatusRequest = 5,
    /// extended_master_secret (RFC 7627). TLS 1.2 only by spec but
    /// some TLS 1.3 servers (notably whatwg.org / Fastly edges) reject
    /// ClientHellos that don't include it — JA3 fingerprint signal.
    ExtendedMasterSecret = 23,
    SupportedGroups = 10,
    /// ec_point_formats (RFC 8422 §5.1.2) — TLS 1.2 point compression
    /// negotiation. Chrome sends `uncompressed` even on TLS 1.3
    /// ClientHellos; absence is a JA3 mismatch.
    EcPointFormats = 11,
    SignatureAlgorithms = 13,
    /// signed_certificate_timestamp (RFC 6962 §3.3.1). Empty body.
    /// Chrome sends this on every TLS 1.3 ClientHello to opt into CT.
    SignedCertificateTimestamp = 18,
    /// padding (RFC 7685). Chrome pads ClientHello so its total wire
    /// length lands on the next 512-byte boundary; lots of strict
    /// servers (Cloudflare's bot management in particular) fingerprint
    /// the missing padding extension.
    Padding = 21,
    /// ALPN — RFC 7301. Cloudflare and other CDNs require this; servers
    /// reject (often via silent TCP RST) connections without it.
    ApplicationLayerProtocolNegotiation = 16,
    /// compress_certificate (RFC 8879). Chrome advertises this in every
    /// TLS 1.3 ClientHello; absence is a JA3 mismatch signal.
    CompressCertificate = 27,
    /// record_size_limit (RFC 8449). Reasonable hint; included by
    /// Firefox and modern Chrome — JA3 signal.
    RecordSizeLimit = 28,
    /// session_ticket — TLS 1.2 era but Chrome still ships it for
    /// JA3 compatibility.
    SessionTicket = 35,
    /// psk_key_exchange_modes (RFC 8446 §4.2.9). MUST accompany any
    /// pre_shared_key, but Chrome ships it unconditionally — some
    /// strict servers reject ClientHellos that lack it.
    PskKeyExchangeModes = 45,
    SupportedVersions = 43,
    KeyShare = 51,
    /// renegotiation_info (RFC 5746). Required by strict TLS 1.2 servers;
    /// modern ones treat absence as a downgrade signal and reject with
    /// handshake_failure (this is precisely what news.ycombinator.com
    /// does when we don't include it).
    RenegotiationInfo = 0xff01,
}

/// Wire encoder. Streams bytes into a `Vec<u8>`, with helpers for the
/// `<length><bytes>` "vector" idiom TLS uses everywhere.
#[derive(Default)]
pub struct Encoder {
    pub buf: Vec<u8>,
}

impl fmt::Debug for Encoder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Encoder")
            .field("len", &self.buf.len())
            .finish()
    }
}

impl Encoder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn u8(&mut self, v: u8) {
        self.buf.push(v);
    }

    pub fn u16(&mut self, v: u16) {
        self.buf.extend_from_slice(&v.to_be_bytes());
    }

    pub fn u24(&mut self, v: u32) {
        assert!(v < (1 << 24));
        self.buf.push(((v >> 16) & 0xFF) as u8);
        self.buf.push(((v >> 8) & 0xFF) as u8);
        self.buf.push((v & 0xFF) as u8);
    }

    pub fn bytes(&mut self, v: &[u8]) {
        self.buf.extend_from_slice(v);
    }

    /// Write a length-prefixed vector: `<u8 length><bytes>`.
    pub fn vec_u8<F: FnOnce(&mut Self)>(&mut self, body: F) {
        let len_pos = self.buf.len();
        self.buf.push(0);
        body(self);
        let len = self.buf.len() - len_pos - 1;
        assert!(len <= u8::MAX as usize, "vec_u8 overflow");
        self.buf[len_pos] = len as u8;
    }

    /// `<u16 length><bytes>`.
    pub fn vec_u16<F: FnOnce(&mut Self)>(&mut self, body: F) {
        let len_pos = self.buf.len();
        self.buf.push(0);
        self.buf.push(0);
        body(self);
        let len = self.buf.len() - len_pos - 2;
        assert!(len <= u16::MAX as usize, "vec_u16 overflow");
        let bytes = (len as u16).to_be_bytes();
        self.buf[len_pos] = bytes[0];
        self.buf[len_pos + 1] = bytes[1];
    }

    /// `<u24 length><bytes>`. Handshake messages use this.
    pub fn vec_u24<F: FnOnce(&mut Self)>(&mut self, body: F) {
        let len_pos = self.buf.len();
        self.buf.extend_from_slice(&[0, 0, 0]);
        body(self);
        let len = self.buf.len() - len_pos - 3;
        assert!(len < (1 << 24), "vec_u24 overflow");
        self.buf[len_pos] = ((len >> 16) & 0xFF) as u8;
        self.buf[len_pos + 1] = ((len >> 8) & 0xFF) as u8;
        self.buf[len_pos + 2] = (len & 0xFF) as u8;
    }
}

/// Build the ClientHello handshake message *body* (no record framing,
/// RFC 8701 GREASE values are `0x0A0A, 0x1A1A, 0x2A2A, ..., 0xFAFA`
/// (top nibble 0..F, low byte 0xA, high byte = low byte). They have
/// no meaning — the server is required to ignore them — but their
/// presence signals "this is a real Chrome-like client" to picky
/// servers (Fastly edges fingerprint via JA3). We pick deterministic
/// values so the same connection's CH and CCS-echo paths align.
fn grease_value_a() -> u16 {
    0x6A6A // arbitrary fixed GREASE; Chrome's are random per-connection
}

fn grease_value_b() -> u16 {
    0xBABA
}

/// no `HandshakeType` byte — caller wraps).
///
/// `client_random`: 32 random bytes.
/// `session_id`: legacy 32-byte session ID (random per RFC 8446 §4.1.2).
/// `key_share_x25519_pub`: 32-byte X25519 public key.
/// `key_share_p256_pub`: 65-byte uncompressed P-256 public point
///   (`0x04 || X || Y`). We send both key shares up front so the
///   server can pick whichever curve it prefers without a
///   HelloRetryRequest round trip — that doubles handshake bytes
///   but matches Chrome's behaviour and unblocks servers that only
///   accept secp256r1 (e.g. some Cloudflare strict configs).
/// `server_name`: SNI hostname.
pub fn build_client_hello_body(
    client_random: &[u8; 32],
    session_id: &[u8; 32],
    key_share_x25519_pub: &[u8; 32],
    key_share_p256_pub: &[u8; 65],
    key_share_p384_pub: &[u8; 97],
    server_name: &str,
) -> Vec<u8> {
    let mut e = Encoder::new();

    // legacy_version
    e.u16(TLS13_LEGACY_VERSION);
    // random
    e.bytes(client_random);
    // legacy_session_id
    e.vec_u8(|w| w.bytes(session_id));
    // cipher_suites — NO GREASE. Cloudflare's bot manager rejects
    // any ClientHello whose JA3 looks "Chrome" (GREASE + ALPS +
    // padding to 512) but whose ALPN advertises only http/1.1 —
    // because real Chrome would have picked h2. Our h2 client has
    // open bugs, so we ship h1.1 only; the only way through CF
    // bot mode is to ALSO drop the Chrome-shaped JA3 markers, so
    // the handshake reads as "generic TLS library" (which CF
    // accepts on h1.1).
    e.vec_u16(|w| {
        w.u16(CipherSuite::Aes128GcmSha256 as u16); // 0x1301
        w.u16(CipherSuite::Aes256GcmSha384 as u16); // 0x1302
        w.u16(CipherSuite::ChaCha20Poly1305Sha256 as u16); // 0x1303
        w.u16(0xc02b); // ECDHE-ECDSA-AES128-GCM-SHA256
        w.u16(0xc02f); // ECDHE-RSA-AES128-GCM-SHA256
        w.u16(0xc02c); // ECDHE-ECDSA-AES256-GCM-SHA384
        w.u16(0xc030); // ECDHE-RSA-AES256-GCM-SHA384
        w.u16(0xcca9); // ECDHE-ECDSA-CHACHA20-POLY1305
        w.u16(0xcca8); // ECDHE-RSA-CHACHA20-POLY1305
    });
    // legacy_compression_methods: { null }
    e.vec_u8(|w| w.u8(0));

    // Extensions — Chrome 131 shape. Order is: leading GREASE,
    // server_name, extended_master_secret, renegotiation_info,
    // supported_groups, ec_point_formats, session_ticket, ALPN,
    // status_request, signature_algorithms,
    // signed_certificate_timestamp, key_share,
    // psk_key_exchange_modes, supported_versions, trailing GREASE,
    // padding. Cloudflare's bot management compares the (sorted)
    // extension multiset and the cipher list multiset against a
    // known-Chrome fingerprint (JA4); any missing extension or
    // wrong cipher is a downgrade.
    //
    // We build the extension block into a side buffer first so we
    // can compute its length and pad the whole ClientHello to a
    // 512-byte boundary (RFC 7685).
    let mut ext = Encoder::new();
    {
        let w = &mut ext;

        // (no leading GREASE — see cipher list comment)

        // 2. server_name (SNI).
        w.u16(ExtensionType::ServerName as u16);
        w.vec_u16(|w| {
            w.vec_u16(|w| {
                w.u8(0); // host_name name_type
                w.vec_u16(|w| w.bytes(server_name.as_bytes()));
            });
        });

        // 3. extended_master_secret — empty body.
        w.u16(ExtensionType::ExtendedMasterSecret as u16);
        w.vec_u16(|_| {});

        // 4. renegotiation_info — empty body (initial handshake).
        w.u16(ExtensionType::RenegotiationInfo as u16);
        w.vec_u16(|w| w.vec_u8(|_| {}));

        // 5. supported_groups — x25519 + P-256 + P-384 (no GREASE).
        w.u16(ExtensionType::SupportedGroups as u16);
        w.vec_u16(|w| {
            w.vec_u16(|w| {
                w.u16(NamedGroup::X25519 as u16);
                w.u16(NamedGroup::Secp256r1 as u16);
                w.u16(NamedGroup::Secp384r1 as u16);
            });
        });

        // 6. ec_point_formats — uncompressed only (Chrome's value).
        w.u16(ExtensionType::EcPointFormats as u16);
        w.vec_u16(|w| w.vec_u8(|w| w.u8(0)));

        // 7. session_ticket — empty body.
        w.u16(ExtensionType::SessionTicket as u16);
        w.vec_u16(|_| {});

        // 8. ALPN — http/1.1 only. Tried "h2,http/1.1" with a full
        // Chrome-shaped h2 client (SETTINGS values, WINDOW_UPDATE,
        // alphabetical headers, Huffman HPACK, priority header).
        // Cloudflare's bot manager (thehindu) and Google's frontend
        // BOTH still reject the h2 stream — they fingerprint deeper
        // than the SETTINGS+headers level: PRIORITY_UPDATE frames,
        // exact frame ordering, possibly TLS-fingerprint cross-check.
        // Advertising h2 forces those servers onto an h2 channel
        // they then close, breaking sites that worked over h1.1.
        // h1.1-only is the strictly-better net until the h2 path
        // can clear those checks (separate slice — needs Wireshark).
        w.u16(ExtensionType::ApplicationLayerProtocolNegotiation as u16);
        w.vec_u16(|w| {
            w.vec_u16(|w| {
                w.vec_u8(|w| w.bytes(b"http/1.1"));
            });
        });

        // 9. status_request — OCSP.
        w.u16(ExtensionType::StatusRequest as u16);
        w.vec_u16(|w| {
            w.u8(1); // OCSP
            w.u16(0); // empty responder_id_list
            w.u16(0); // empty request_extensions
        });

        // 10. signature_algorithms — Chrome's exact set / order.
        w.u16(ExtensionType::SignatureAlgorithms as u16);
        w.vec_u16(|w| {
            w.vec_u16(|w| {
                w.u16(SignatureScheme::EcdsaSecp256r1Sha256 as u16);
                w.u16(SignatureScheme::RsaPssRsaeSha256 as u16);
                w.u16(SignatureScheme::RsaPkcs1Sha256 as u16);
                w.u16(SignatureScheme::EcdsaSecp384r1Sha384 as u16);
                w.u16(SignatureScheme::RsaPssRsaeSha384 as u16);
                w.u16(SignatureScheme::RsaPkcs1Sha384 as u16);
                w.u16(SignatureScheme::RsaPssRsaeSha512 as u16);
                w.u16(SignatureScheme::RsaPkcs1Sha512 as u16);
            });
        });

        // 11. signed_certificate_timestamp — empty body.
        w.u16(ExtensionType::SignedCertificateTimestamp as u16);
        w.vec_u16(|_| {});

        // 12. key_share — x25519 + P-256 + P-384. Chrome 131 doesn't
        // send the P-384 entry, so this is a JA4-size deviation, but
        // we lack HelloRetryRequest support and postgresql.org's TLS
        // 1.3 endpoint is P-384-only — without the share up-front
        // it issues HRR and we die. Keep it; thehindu/Google JA4
        // mismatch wasn't this anyway.
        w.u16(ExtensionType::KeyShare as u16);
        w.vec_u16(|w| {
            w.vec_u16(|w| {
                w.u16(NamedGroup::X25519 as u16);
                w.vec_u16(|w| w.bytes(key_share_x25519_pub));
                w.u16(NamedGroup::Secp256r1 as u16);
                w.vec_u16(|w| w.bytes(key_share_p256_pub));
                w.u16(NamedGroup::Secp384r1 as u16);
                w.vec_u16(|w| w.bytes(key_share_p384_pub));
            });
        });

        // 13. psk_key_exchange_modes — DHE only.
        w.u16(ExtensionType::PskKeyExchangeModes as u16);
        w.vec_u16(|w| w.vec_u8(|w| w.u8(1)));

        // 14. supported_versions — TLS 1.3 + TLS 1.2.
        w.u16(ExtensionType::SupportedVersions as u16);
        w.vec_u16(|w| {
            w.vec_u8(|w| {
                w.u16(TLS13_REAL_VERSION);
                w.u16(TLS12_VERSION);
            })
        });

        // (no ALPS — only meaningful with h2)
        // (no trailing GREASE)
    }

    // (no padding extension — would also be a Chrome marker)

    // Stitch the ext block into e as a length-prefixed vector.
    e.vec_u16(|w| w.bytes(&ext.buf));

    e.buf
}

/// Wrap a handshake-message body in its `<HandshakeType><u24 length><body>`.
pub fn wrap_handshake(kind: HandshakeType, body: Vec<u8>) -> Vec<u8> {
    let mut e = Encoder::new();
    e.u8(kind as u8);
    e.vec_u24(|w| w.bytes(&body));
    e.buf
}

/// Wrap a handshake message in a TLSPlaintext record:
///   `ContentType(1) || legacy_version(2) || u16 length || fragment`.
pub fn wrap_record(content_type: ContentType, body: &[u8]) -> Vec<u8> {
    let mut e = Encoder::new();
    e.u8(content_type as u8);
    e.u16(TLS13_LEGACY_VERSION);
    e.vec_u16(|w| w.bytes(body));
    e.buf
}

// ---------- Decoder ----------

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DecodeError {
    Short,
    Bad(&'static str),
    Unexpected { expected: u8, got: u8 },
}

impl fmt::Display for DecodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Short => f.write_str("truncated"),
            Self::Bad(s) => write!(f, "bad: {s}"),
            Self::Unexpected { expected, got } => {
                write!(f, "unexpected: expected {expected}, got {got}")
            }
        }
    }
}

impl std::error::Error for DecodeError {}

#[derive(Copy, Clone)]
pub struct Decoder<'a> {
    pub buf: &'a [u8],
}

impl<'a> fmt::Debug for Decoder<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Decoder({} bytes)", self.buf.len())
    }
}

impl<'a> Decoder<'a> {
    pub fn new(buf: &'a [u8]) -> Self {
        Self { buf }
    }

    pub fn is_empty(&self) -> bool {
        self.buf.is_empty()
    }

    pub fn remaining(&self) -> usize {
        self.buf.len()
    }

    pub fn u8(&mut self) -> Result<u8, DecodeError> {
        if self.buf.is_empty() {
            return Err(DecodeError::Short);
        }
        let v = self.buf[0];
        self.buf = &self.buf[1..];
        Ok(v)
    }

    pub fn u16(&mut self) -> Result<u16, DecodeError> {
        if self.buf.len() < 2 {
            return Err(DecodeError::Short);
        }
        let v = u16::from_be_bytes([self.buf[0], self.buf[1]]);
        self.buf = &self.buf[2..];
        Ok(v)
    }

    pub fn u24(&mut self) -> Result<u32, DecodeError> {
        if self.buf.len() < 3 {
            return Err(DecodeError::Short);
        }
        let v =
            (u32::from(self.buf[0]) << 16) | (u32::from(self.buf[1]) << 8) | u32::from(self.buf[2]);
        self.buf = &self.buf[3..];
        Ok(v)
    }

    pub fn take(&mut self, n: usize) -> Result<&'a [u8], DecodeError> {
        if self.buf.len() < n {
            return Err(DecodeError::Short);
        }
        let (head, tail) = self.buf.split_at(n);
        self.buf = tail;
        Ok(head)
    }

    /// Read a `<u8 len><bytes>` vector.
    pub fn vec_u8(&mut self) -> Result<&'a [u8], DecodeError> {
        let n = usize::from(self.u8()?);
        self.take(n)
    }

    pub fn vec_u16(&mut self) -> Result<&'a [u8], DecodeError> {
        let n = usize::from(self.u16()?);
        self.take(n)
    }

    pub fn vec_u24(&mut self) -> Result<&'a [u8], DecodeError> {
        let n = self.u24()? as usize;
        self.take(n)
    }
}

// ---------- ServerHello ----------

#[derive(Debug, Clone)]
pub struct ServerHello<'a> {
    pub legacy_version: u16,
    pub server_random: [u8; 32],
    pub legacy_session_id_echo: &'a [u8],
    pub cipher_suite: u16,
    pub legacy_compression_method: u8,
    pub extensions: &'a [u8],
    /// Selected key_share group + key exchange bytes, if present.
    pub key_share_group: Option<u16>,
    pub key_share_key_exchange: Option<&'a [u8]>,
    pub selected_version: Option<u16>,
    /// ALPN selected by server, if echoed in ServerHello. TLS 1.2 sends
    /// the chosen protocol here (plaintext); TLS 1.3 moves it to
    /// EncryptedExtensions and this stays None.
    pub alpn: Option<String>,
}

pub fn parse_server_hello(body: &[u8]) -> Result<ServerHello<'_>, DecodeError> {
    let mut d = Decoder::new(body);
    let legacy_version = d.u16()?;
    let random_bytes = d.take(32)?;
    let mut server_random = [0u8; 32];
    server_random.copy_from_slice(random_bytes);
    let legacy_session_id_echo = d.vec_u8()?;
    let cipher_suite = d.u16()?;
    let legacy_compression_method = d.u8()?;
    let extensions = d.vec_u16()?;
    if !d.is_empty() {
        return Err(DecodeError::Bad("trailing ServerHello data"));
    }

    // Walk extensions for key_share + supported_versions.
    let mut ext = Decoder::new(extensions);
    let mut key_share_group = None;
    let mut key_share_key_exchange = None;
    let mut selected_version = None;
    let mut alpn: Option<String> = None;
    while !ext.is_empty() {
        let ext_type = ext.u16()?;
        let ext_data = ext.vec_u16()?;
        match ext_type {
            x if x == ExtensionType::KeyShare as u16 => {
                let mut k = Decoder::new(ext_data);
                key_share_group = Some(k.u16()?);
                key_share_key_exchange = Some(k.vec_u16()?);
            }
            x if x == ExtensionType::SupportedVersions as u16 => {
                let mut s = Decoder::new(ext_data);
                selected_version = Some(s.u16()?);
            }
            x if x == ExtensionType::ApplicationLayerProtocolNegotiation as u16 => {
                // Body: <u16 list_len><proto*>; proto = <u8 len><name>.
                // Server returns exactly one selected protocol.
                let mut a = Decoder::new(ext_data);
                let _list_len = a.u16()?;
                let proto = a.vec_u8()?;
                if let Ok(s) = core::str::from_utf8(proto) {
                    alpn = Some(s.to_string());
                }
            }
            _ => {} // ignore
        }
    }

    Ok(ServerHello {
        legacy_version,
        server_random,
        legacy_session_id_echo,
        cipher_suite,
        legacy_compression_method,
        extensions,
        key_share_group,
        key_share_key_exchange,
        selected_version,
        alpn,
    })
}

/// A parsed `TLSPlaintext` record header + payload slice.
#[derive(Debug, Clone)]
pub struct Record<'a> {
    pub content_type: u8,
    pub legacy_version: u16,
    pub fragment: &'a [u8],
}

pub fn parse_record_header(buf: &[u8]) -> Result<(Record<'_>, usize), DecodeError> {
    let mut d = Decoder::new(buf);
    let content_type = d.u8()?;
    let legacy_version = d.u16()?;
    let len = usize::from(d.u16()?);
    let fragment = d.take(len)?;
    Ok((
        Record {
            content_type,
            legacy_version,
            fragment,
        },
        5 + len,
    ))
}

/// Inside a single record's fragment for content_type=Handshake, walk
/// each `<HandshakeType><u24 length><body>`.
pub struct HandshakeIter<'a> {
    buf: &'a [u8],
}

impl<'a> fmt::Debug for HandshakeIter<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "HandshakeIter({} bytes)", self.buf.len())
    }
}

impl<'a> HandshakeIter<'a> {
    pub fn new(buf: &'a [u8]) -> Self {
        Self { buf }
    }
}

impl<'a> Iterator for HandshakeIter<'a> {
    type Item = Result<(u8, &'a [u8], &'a [u8]), DecodeError>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.buf.is_empty() {
            return None;
        }
        let mut d = Decoder::new(self.buf);
        let kind = match d.u8() {
            Ok(k) => k,
            Err(e) => return Some(Err(e)),
        };
        let body = match d.vec_u24() {
            Ok(b) => b,
            Err(e) => return Some(Err(e)),
        };
        let consumed = self.buf.len() - d.buf.len();
        let full_tlv = &self.buf[..consumed];
        self.buf = d.buf;
        Some(Ok((kind, body, full_tlv)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decoder_basic() {
        let buf = [
            0x12, 0x34, 0x56, 0x02, b'h', b'i', 0x00, 0x00, 0x03, b'b', b'y', b'e',
        ];
        let mut d = Decoder::new(&buf);
        assert_eq!(d.u8().unwrap(), 0x12);
        assert_eq!(d.u16().unwrap(), 0x3456);
        assert_eq!(d.vec_u8().unwrap(), b"hi");
        assert_eq!(d.vec_u24().unwrap(), b"bye");
        assert!(d.is_empty());
    }

    #[test]
    fn handshake_iter_walks() {
        // Two handshake messages back-to-back.
        let mut e = Encoder::new();
        e.u8(HandshakeType::ServerHello as u8);
        e.vec_u24(|w| w.bytes(&[0xab, 0xcd]));
        e.u8(HandshakeType::Finished as u8);
        e.vec_u24(|w| w.bytes(&[0xee, 0xff, 0x00]));
        let mut it = HandshakeIter::new(&e.buf);
        let (kind, body, _) = it.next().unwrap().unwrap();
        assert_eq!(kind, HandshakeType::ServerHello as u8);
        assert_eq!(body, &[0xab, 0xcd]);
        let (kind, body, _) = it.next().unwrap().unwrap();
        assert_eq!(kind, HandshakeType::Finished as u8);
        assert_eq!(body, &[0xee, 0xff, 0x00]);
        assert!(it.next().is_none());
    }

    #[test]
    fn record_header_roundtrip() {
        let body = b"some fragment payload";
        let rec = wrap_record(ContentType::Handshake, body);
        let (parsed, consumed) = parse_record_header(&rec).unwrap();
        assert_eq!(consumed, rec.len());
        assert_eq!(parsed.content_type, ContentType::Handshake as u8);
        assert_eq!(parsed.legacy_version, TLS13_LEGACY_VERSION);
        assert_eq!(parsed.fragment, body);
    }

    #[test]
    fn server_hello_roundtrip_wraps_parse() {
        roundtrip_server_hello();
    }

    #[test]
    fn encoder_vec_lengths() {
        let mut e = Encoder::new();
        e.vec_u8(|w| {
            w.u8(0xaa);
            w.u8(0xbb);
        });
        e.vec_u16(|w| {
            w.u16(0x1234);
        });
        e.vec_u24(|w| {
            w.bytes(b"hi");
        });
        assert_eq!(
            e.buf,
            vec![
                0x02, 0xaa, 0xbb, 0x00, 0x02, 0x12, 0x34, 0x00, 0x00, 0x02, b'h', b'i'
            ]
        );
    }

    fn roundtrip_server_hello() {
        // Build a fake ServerHello we can parse back.
        let cr = [0x55u8; 32];
        let mut e = Encoder::new();
        e.u16(TLS13_LEGACY_VERSION);
        e.bytes(&cr);
        e.vec_u8(|w| w.bytes(&[0xAA; 32])); // session id echo
        e.u16(CipherSuite::Aes128GcmSha256 as u16);
        e.u8(0); // compression
        e.vec_u16(|w| {
            // supported_versions extension picking 0x0304
            w.u16(ExtensionType::SupportedVersions as u16);
            w.vec_u16(|w| w.u16(TLS13_REAL_VERSION));
            // key_share extension with x25519 + 32 bytes
            w.u16(ExtensionType::KeyShare as u16);
            w.vec_u16(|w| {
                w.u16(NamedGroup::X25519 as u16);
                w.vec_u16(|w| w.bytes(&[0x99; 32]));
            });
        });

        let sh = parse_server_hello(&e.buf).unwrap();
        assert_eq!(sh.server_random, cr);
        assert_eq!(sh.cipher_suite, CipherSuite::Aes128GcmSha256 as u16);
        assert_eq!(sh.selected_version, Some(TLS13_REAL_VERSION));
        assert_eq!(sh.key_share_group, Some(NamedGroup::X25519 as u16));
        assert_eq!(sh.key_share_key_exchange.unwrap().len(), 32);
    }

    #[test]
    fn client_hello_smoke() {
        let cr = [0x42u8; 32];
        let sid = [0x88u8; 32];
        let ks_pub = [0x11u8; 32];
        let mut ks_p256 = [0u8; 65];
        ks_p256[0] = 0x04;
        let mut ks_p384 = [0u8; 97];
        ks_p384[0] = 0x04;
        let body = build_client_hello_body(&cr, &sid, &ks_pub, &ks_p256, &ks_p384, "example.com");
        // Sanity: starts with legacy version + client random.
        assert_eq!(&body[..2], &[0x03, 0x03]);
        assert_eq!(&body[2..34], &cr);
        // Session ID block: length 32, then 32 bytes 0x88.
        assert_eq!(body[34], 32);
        assert_eq!(&body[35..67], &sid);

        // Wrap and check the handshake header length sums match.
        let hs = wrap_handshake(HandshakeType::ClientHello, body.clone());
        assert_eq!(hs[0], 1); // ClientHello
        let len = u32::from_be_bytes([0, hs[1], hs[2], hs[3]]) as usize;
        assert_eq!(len, body.len());

        // Wrap in record.
        let rec = wrap_record(ContentType::Handshake, &hs);
        assert_eq!(rec[0], 22);
        assert_eq!(&rec[1..3], &[0x03, 0x03]);
        let rec_len = u16::from_be_bytes([rec[3], rec[4]]) as usize;
        assert_eq!(rec_len, hs.len());
    }
}
