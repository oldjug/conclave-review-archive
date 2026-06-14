//! Minimal TLS 1.2 client — RFC 5246 + RFC 5288 (AES-GCM in 1.2).
//!
//! Built specifically to handle servers that haven't upgraded to TLS
//! 1.3 (Hacker News, some older static-content hosts). The 1.3 driver
//! in `client.rs` is the preferred path; we only fall through to here
//! when the server's `supported_versions` selection picks 0x0303.
//!
//! Scope:
//! - One cipher suite: `TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256` (0xc02f)
//! - Named curve: `secp256r1` (P-256). X25519 isn't valid in TLS 1.2.
//! - Server cert: RSA (for the ECDHE-RSA suite signature).
//! - No client cert, no session resumption, no renegotiation.
//!
//! Nonce / AAD differences from 1.3 are subtle but real:
//! - 1.2 AEAD nonce = salt(4) || explicit_nonce_on_wire(8).
//! - 1.2 AAD = seq(8) || type(1) || version(2) || plaintext_len(2).
//! - 1.2 Finished verify_data is only 12 bytes.
//! - 1.2 master_secret derivation is via the TLS PRF, not HKDF.

use cv_crypto::aes_gcm::{Aes128Gcm, Aes256Gcm};
use cv_crypto::chacha20poly1305::ChaCha20Poly1305;
use cv_crypto::hmac::{HmacSha256, HmacSha384};
use cv_crypto::p256;
use cv_crypto::rsa::{Hash as RsaHash, RsaPublicKey, verify_pkcs1_v15, verify_pss};
use cv_crypto::sha256::Sha256;
use cv_crypto::sha384::Sha384;
use cv_crypto::x509;

use crate::socket::Socket;
use crate::tls::chain_validate;
use crate::tls::client::TlsError;
use crate::tls::messages::{ContentType, HandshakeIter, HandshakeType, TLS12_VERSION, wrap_record};

/// State the 1.3 driver hands off when it sees a 1.2 ServerHello.
/// Contains everything we need to drive the rest of the handshake.
pub struct Tls12HandoffState {
    pub host: String,
    pub client_random: [u8; 32],
    pub server_random: [u8; 32],
    pub session_id: Vec<u8>,
    pub cipher_suite: u16,
    /// Transcript accumulated so far — currently includes ClientHello +
    /// ServerHello. We keep appending every subsequent handshake
    /// message so the Finished verify_data hash is correct.
    pub transcript: Vec<u8>,
    /// P-256 ephemeral private scalar generated when ClientHello was
    /// built. Used to compute the shared secret once we see SKE.
    pub client_priv_p256: [u8; 32],
    /// X25519 ephemeral private scalar. Used when the server picks
    /// curve=29 (x25519) in its TLS 1.2 ServerKeyExchange — that's
    /// the default for Cloudflare-fronted hosts negotiating 1.2.
    pub client_priv_x25519: [u8; 32],
    /// P-384 ephemeral private scalar. Used when the server picks
    /// curve=24 (secp384r1) in its TLS 1.2 ServerKeyExchange — some
    /// PostgreSQL / older banking hosts only allow this curve.
    pub client_priv_p384: [u8; 48],
    /// Set when the server echoed the `extended_master_secret`
    /// extension in its ServerHello. If true, master_secret derives
    /// via PRF(pms, "extended master secret", H(handshake)). If
    /// false, the legacy PRF(pms, "master secret", c_rand||s_rand)
    /// form is used — RFC 7627 §5.
    pub ems_used: bool,
}

/// Drive the remainder of the TLS 1.2 handshake after ServerHello has
/// already been read by the caller. On success, returns the established
/// AEAD keys for the new TlsStream.
pub struct Tls12Outcome {
    pub server_certs: Vec<Vec<u8>>,
    pub client_write_key: Vec<u8>,
    pub server_write_key: Vec<u8>,
    pub client_write_salt: Vec<u8>,
    pub server_write_salt: Vec<u8>,
    pub master_secret: Vec<u8>,
    pub alpn: String,
    pub kind_code: u16,
}

/// Symmetric primitives selected by the negotiated 1.2 suite.
#[derive(Copy, Clone, Debug)]
pub(crate) enum Tls12CipherKind {
    Aes128GcmSha256,
    Aes256GcmSha384,
    ChaCha20Poly1305Sha256,
}

impl Tls12CipherKind {
    fn from_code(code: u16) -> Option<(Self, /* rsa_cert */ bool)> {
        match code {
            0xc02f => Some((Self::Aes128GcmSha256, true)),
            0xc030 => Some((Self::Aes256GcmSha384, true)),
            0xcca8 => Some((Self::ChaCha20Poly1305Sha256, true)),
            0xc02b => Some((Self::Aes128GcmSha256, false)),
            0xc02c => Some((Self::Aes256GcmSha384, false)),
            0xcca9 => Some((Self::ChaCha20Poly1305Sha256, false)),
            _ => None,
        }
    }
    fn key_len(self) -> usize {
        match self {
            Self::Aes128GcmSha256 => 16,
            Self::Aes256GcmSha384 | Self::ChaCha20Poly1305Sha256 => 32,
        }
    }
    /// ChaCha20-Poly1305 uses an implicit (zero-byte explicit) nonce
    /// per RFC 7905 §2; AES-GCM has a 4-byte salt + 8-byte explicit.
    fn explicit_nonce_len(self) -> usize {
        match self {
            Self::Aes128GcmSha256 | Self::Aes256GcmSha384 => 8,
            Self::ChaCha20Poly1305Sha256 => 0,
        }
    }
    fn fixed_iv_len(self) -> usize {
        match self {
            Self::Aes128GcmSha256 | Self::Aes256GcmSha384 => 4,
            Self::ChaCha20Poly1305Sha256 => 12,
        }
    }
}

pub fn drive_tls12(
    sock: &mut Socket,
    mut state: Tls12HandoffState,
) -> Result<Tls12Outcome, TlsError> {
    let (kind, is_rsa_cert) = Tls12CipherKind::from_code(state.cipher_suite)
        .ok_or(TlsError::UnsupportedCipherSuite(state.cipher_suite))?;
    let _ = is_rsa_cert; // ECDSA paths get used via verify_ske_signature.

    // Read records until we have all of: Certificate, ServerKeyExchange,
    // ServerHelloDone. They arrive in plaintext under content_type=22.
    let mut server_certs: Vec<Vec<u8>> = Vec::new();
    let mut server_ecdhe_pub: Option<Vec<u8>> = None;
    let mut server_signed_params: Option<Vec<u8>> = None;
    let mut server_sig_scheme: u16 = 0;
    let mut server_sig: Vec<u8> = Vec::new();
    let mut server_curve: u16 = 0;
    let mut server_hello_done = false;
    while !server_hello_done {
        let rec = read_one_record_plaintext(sock)?;
        if rec.content_type != ContentType::Handshake as u8 {
            if rec.content_type == ContentType::Alert as u8 && rec.fragment.len() >= 2 {
                return Err(TlsError::Protocol(format!(
                    "tls1.2 server alert level={} desc={}",
                    rec.fragment[0], rec.fragment[1]
                )));
            }
            return Err(TlsError::Protocol(format!(
                "tls1.2 expected handshake, got record type {}",
                rec.content_type
            )));
        }
        let iter = HandshakeIter::new(&rec.fragment);
        for msg in iter {
            let (kind, body, full) = msg.map_err(|e| TlsError::Decode(e.to_string()))?;
            state.transcript.extend_from_slice(full);
            match kind {
                t if t == HandshakeType::Certificate as u8 => {
                    server_certs = parse_certificate_chain(body)?;
                }
                t if t == HandshakeType::ServerKeyExchange as u8 => {
                    let (curve, pubkey, signed, sig_scheme, sig) =
                        parse_server_key_exchange(body)?;
                    // 23 = secp256r1, 24 = secp384r1, 29 = x25519
                    // (RFC 8422 §5.1.1).
                    if curve != 23 && curve != 24 && curve != 29 {
                        return Err(TlsError::Protocol(format!(
                            "tls1.2 unsupported curve {curve}"
                        )));
                    }
                    server_ecdhe_pub = Some(pubkey);
                    server_signed_params = Some(signed);
                    server_sig_scheme = sig_scheme;
                    server_sig = sig;
                    server_curve = curve;
                }
                14 /* server_hello_done */ => {
                    server_hello_done = true;
                    break;
                }
                12 /* server_key_exchange */ => {
                    // covered above
                }
                other => {
                    // CertificateRequest etc. — skip silently. We don't
                    // offer client certs.
                    let _ = other;
                }
            }
        }
    }

    // Verify chain, then verify SKE signature over (client_random ||
    // server_random || curve_params).
    if server_certs.is_empty() {
        return Err(TlsError::NoCertificate);
    }
    chain_validate::verify_chain(&server_certs, &state.host)
        .map_err(|e| TlsError::ChainInvalid(e.to_string()))?;

    let signed_payload =
        server_signed_params.ok_or_else(|| TlsError::Protocol("tls1.2 missing SKE".into()))?;
    let mut sig_input = Vec::with_capacity(64 + signed_payload.len());
    sig_input.extend_from_slice(&state.client_random);
    sig_input.extend_from_slice(&state.server_random);
    sig_input.extend_from_slice(&signed_payload);

    let leaf = x509::parse(&server_certs[0]).map_err(|e| TlsError::Decode(format!("leaf: {e}")))?;
    verify_ske_signature(&leaf, server_sig_scheme, &sig_input, &server_sig)?;

    // ECDH: shared_secret depends on the curve the server picked.
    //   23 (secp256r1) — peer point is uncompressed `04 || X || Y` (65B).
    //   24 (secp384r1) — peer point is uncompressed `04 || X || Y` (97B).
    //   29 (x25519)    — peer point is the 32-byte u-coordinate.
    let peer_pub = server_ecdhe_pub.unwrap();
    let pre_master_secret = match server_curve {
        23 => p256::ecdh_shared(&state.client_priv_p256, &peer_pub)
            .map_err(|_| TlsError::Crypto("tls1.2 P-256 ECDH failed".into()))?
            .to_vec(),
        24 => cv_crypto::p384::ecdh_shared(&state.client_priv_p384, &peer_pub)
            .map_err(|_| TlsError::Crypto("tls1.2 P-384 ECDH failed".into()))?
            .to_vec(),
        29 => {
            if peer_pub.len() != 32 {
                return Err(TlsError::Protocol(format!(
                    "tls1.2 x25519 peer point len {}",
                    peer_pub.len()
                )));
            }
            let mut pp = [0u8; 32];
            pp.copy_from_slice(&peer_pub);
            cv_crypto::x25519::x25519(&state.client_priv_x25519, &pp).to_vec()
        }
        other => {
            return Err(TlsError::Protocol(format!(
                "tls1.2 ECDH curve {other} unsupported"
            )));
        }
    };

    // Build CKE so we can include it in the EMS session_hash. We
    // append-and-then-send so the transcript-hash math works either
    // way the master_secret derivation goes.
    let our_pub: Vec<u8> = match server_curve {
        23 => p256::public_key_uncompressed(&state.client_priv_p256)
            .map_err(|_| TlsError::Crypto("p256 pubkey".into()))?
            .to_vec(),
        24 => cv_crypto::p384::public_key_uncompressed(&state.client_priv_p384)
            .map_err(|_| TlsError::Crypto("p384 pubkey".into()))?
            .to_vec(),
        29 => cv_crypto::x25519::x25519_public(&state.client_priv_x25519).to_vec(),
        _ => unreachable!(),
    };
    let mut cke_body = Vec::with_capacity(1 + our_pub.len());
    cke_body.push(our_pub.len() as u8);
    cke_body.extend_from_slice(&our_pub);
    let cke = wrap_handshake_12(HandshakeType::ClientKeyExchange as u8, &cke_body);
    state.transcript.extend_from_slice(&cke);

    // master_secret derivation per RFC 7627 §4 (EMS) or RFC 5246 §8.1
    // (legacy). Picking the wrong one gives bad_record_mac when the
    // server tries to verify our Finished — that's the Hacker News
    // failure that drove this branching.
    let master_secret = if state.ems_used {
        let session_hash = transcript_hash(kind, &state.transcript);
        prf(
            kind,
            &pre_master_secret,
            b"extended master secret",
            &session_hash,
            48,
        )
    } else {
        let mut seed = Vec::with_capacity(64);
        seed.extend_from_slice(&state.client_random);
        seed.extend_from_slice(&state.server_random);
        prf(kind, &pre_master_secret, b"master secret", &seed, 48)
    };

    // key_block layout per cipher (no MAC keys for AEAD suites):
    //   client_write_key | server_write_key | client_IV | server_IV
    // AEAD AES-GCM IV is 4 bytes (salt); ChaCha20-Poly1305 IV is the
    // full 12-byte fixed implicit IV per RFC 7905.
    let kl = kind.key_len();
    let il = kind.fixed_iv_len();
    let need = 2 * kl + 2 * il;
    let mut kx_seed = Vec::with_capacity(64);
    kx_seed.extend_from_slice(&state.server_random);
    kx_seed.extend_from_slice(&state.client_random);
    let key_block = prf(kind, &master_secret, b"key expansion", &kx_seed, need);

    let client_write_key = key_block[0..kl].to_vec();
    let server_write_key = key_block[kl..2 * kl].to_vec();
    let client_write_salt = key_block[2 * kl..2 * kl + il].to_vec();
    let server_write_salt = key_block[2 * kl + il..2 * kl + 2 * il].to_vec();

    // CKE already in the transcript (we appended it before computing
    // session_hash). Just put it on the wire.
    sock.write_all(&wrap_record(ContentType::Handshake, &cke))
        .map_err(|e| TlsError::Protocol(format!("send CKE: {e:?}")))?;

    // ChangeCipherSpec — single byte 0x01, type 20.
    sock.write_all(&wrap_record(ContentType::ChangeCipherSpec, &[0x01]))
        .map_err(|e| TlsError::Protocol(format!("send CCS: {e:?}")))?;

    // Client Finished — encrypted with the new keys. verify_data is
    // PRF(master, "client finished", H(transcript))[:12] where H is the
    // suite's hash (SHA-256 or SHA-384). Verify_data length is fixed
    // at 12 bytes for both per RFC 5246 §7.4.9.
    let th = transcript_hash(kind, &state.transcript);
    let verify_data = prf(kind, &master_secret, b"client finished", &th, 12);
    let mut finished_body = Vec::with_capacity(verify_data.len());
    finished_body.extend_from_slice(&verify_data);
    let finished_msg = wrap_handshake_12(HandshakeType::Finished as u8, &finished_body);
    let mut seq_client: u64 = 0;
    let enc_finished = encrypt_record_12(
        kind,
        ContentType::Handshake as u8,
        &finished_msg,
        &client_write_key,
        &client_write_salt,
        seq_client,
    )?;
    seq_client += 1;
    sock.write_all(&wrap_record(ContentType::Handshake, &enc_finished))
        .map_err(|e| TlsError::Protocol(format!("send Finished: {e:?}")))?;
    state.transcript.extend_from_slice(&finished_msg);

    // Receive server's post-Finished records. The sequence is:
    //   [optional NewSessionTicket (plaintext handshake, RFC 5077 §3.3)]
    //   ChangeCipherSpec (plaintext)
    //   Finished (encrypted)
    // We must track whether we've crossed the server's CCS so we know
    // whether a Handshake-typed record is plaintext (NST) or encrypted
    // (Finished). Plaintext NST still goes into the transcript so the
    // server's Finished verify_data matches (server computes its
    // verify_data over a transcript that includes NST).
    let mut seq_server: u64 = 0;
    let mut saw_server_ccs = false;
    loop {
        let rec = read_one_record_plaintext(sock)?;
        match rec.content_type {
            t if t == ContentType::ChangeCipherSpec as u8 => {
                saw_server_ccs = true;
                continue;
            }
            t if t == ContentType::Alert as u8 => {
                if rec.fragment.len() >= 2 {
                    return Err(TlsError::Protocol(format!(
                        "tls1.2 server alert lvl={} desc={}",
                        rec.fragment[0], rec.fragment[1]
                    )));
                }
                return Err(TlsError::Protocol("tls1.2 truncated alert".into()));
            }
            t if t == ContentType::Handshake as u8 && !saw_server_ccs => {
                // Plaintext handshake message before server CCS — this
                // is the NewSessionTicket (RFC 5077 §3.3). Add to the
                // transcript so server-Finished verify_data matches,
                // then keep reading.
                let iter = HandshakeIter::new(&rec.fragment);
                for msg in iter {
                    let (_kind, _body, full) = msg.map_err(|e| TlsError::Decode(e.to_string()))?;
                    state.transcript.extend_from_slice(full);
                }
                continue;
            }
            t if t == ContentType::Handshake as u8 => {
                let plain = decrypt_record_12(
                    kind,
                    ContentType::Handshake as u8,
                    &rec.fragment,
                    &server_write_key,
                    &server_write_salt,
                    seq_server,
                )
                .map_err(|e| {
                    TlsError::Crypto(format!(
                        "tls1.2 server-finished decrypt failed (cipher=0x{:04x} \
                     ems={} curve={}): {e:?}",
                        state.cipher_suite, state.ems_used, server_curve
                    ))
                })?;
                seq_server += 1;
                if plain.first().copied() != Some(HandshakeType::Finished as u8) {
                    return Err(TlsError::Protocol(format!(
                        "tls1.2 expected Finished, got hs type {:?}",
                        plain.first()
                    )));
                }
                if plain.len() < 16 {
                    return Err(TlsError::Protocol("tls1.2 short Finished".into()));
                }
                let len_field =
                    (plain[1] as usize) << 16 | (plain[2] as usize) << 8 | plain[3] as usize;
                if len_field != 12 || plain.len() != 4 + 12 {
                    return Err(TlsError::Protocol("tls1.2 bad Finished len".into()));
                }
                let server_verify = &plain[4..16];
                let th_server = transcript_hash(kind, &state.transcript);
                let expected = prf(kind, &master_secret, b"server finished", &th_server, 12);
                if !cv_crypto::subtle::ct_eq(&expected, server_verify) {
                    return Err(TlsError::ServerFinishedMismatch);
                }
                state.transcript.extend_from_slice(&plain);
                break;
            }
            other => {
                return Err(TlsError::Protocol(format!(
                    "tls1.2 unexpected record type {other}"
                )));
            }
        }
    }
    let _ = seq_client;

    Ok(Tls12Outcome {
        server_certs,
        client_write_key,
        server_write_key,
        client_write_salt,
        server_write_salt,
        master_secret,
        alpn: String::new(),
        kind_code: state.cipher_suite,
    })
}

pub const CIPHER_ECDHE_RSA_AES128_GCM_SHA256: u16 = 0xc02f;

/// Map a wire cipher-suite codepoint to the symmetric kind for
/// post-handshake record processing. `None` means "we don't speak that
/// suite" — TlsStream should refuse to use it for app data.
pub(crate) fn tls12_kind_from_code(code: u16) -> Option<Tls12CipherKind> {
    Tls12CipherKind::from_code(code).map(|(k, _)| k)
}

/// Same as the internal `encrypt_record_12` but exposed for the
/// `TlsStream` app-data write path. The record fragment returned is
/// what goes after the 5-byte record header.
pub(crate) fn encrypt_app_record(
    kind: Tls12CipherKind,
    content_type: u8,
    plaintext: &[u8],
    key: &[u8],
    salt: &[u8],
    seq: u64,
) -> Result<Vec<u8>, TlsError> {
    encrypt_record_12(kind, content_type, plaintext, key, salt, seq)
}

/// Same as `decrypt_record_12` but exposed for `TlsStream`. Caller has
/// already stripped the 5-byte record header off `fragment`.
pub(crate) fn decrypt_app_record(
    kind: Tls12CipherKind,
    content_type: u8,
    fragment: &[u8],
    key: &[u8],
    salt: &[u8],
    seq: u64,
) -> Result<Vec<u8>, TlsError> {
    decrypt_record_12(kind, content_type, fragment, key, salt, seq)
}

/// TLS 1.2 handshake-message wire form: type(1) + length(3) + body.
fn wrap_handshake_12(hs_type: u8, body: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(4 + body.len());
    out.push(hs_type);
    let n = body.len() as u32;
    out.push(((n >> 16) & 0xFF) as u8);
    out.push(((n >> 8) & 0xFF) as u8);
    out.push((n & 0xFF) as u8);
    out.extend_from_slice(body);
    out
}

fn parse_certificate_chain(body: &[u8]) -> Result<Vec<Vec<u8>>, TlsError> {
    if body.len() < 3 {
        return Err(TlsError::Decode("certs: too short".into()));
    }
    let list_len = ((body[0] as usize) << 16) | ((body[1] as usize) << 8) | body[2] as usize;
    if list_len + 3 > body.len() {
        return Err(TlsError::Decode("certs: list len".into()));
    }
    let mut p = 3;
    let mut out: Vec<Vec<u8>> = Vec::new();
    while p + 3 <= 3 + list_len {
        let cl = ((body[p] as usize) << 16) | ((body[p + 1] as usize) << 8) | body[p + 2] as usize;
        p += 3;
        if p + cl > 3 + list_len {
            return Err(TlsError::Decode("certs: entry len".into()));
        }
        out.push(body[p..p + cl].to_vec());
        p += cl;
    }
    Ok(out)
}

/// Parse `ServerKeyExchange` body for an ECDHE-RSA / ECDHE-ECDSA suite.
/// Returns (named_curve, uncompressed_point, signed_params, sig_scheme, sig).
fn parse_server_key_exchange(
    body: &[u8],
) -> Result<(u16, Vec<u8>, Vec<u8>, u16, Vec<u8>), TlsError> {
    if body.len() < 4 {
        return Err(TlsError::Decode("ske: short".into()));
    }
    // ECParameters: curve_type(1) named_curve(2)
    if body[0] != 3 {
        return Err(TlsError::Protocol(format!(
            "tls1.2 SKE curve_type {}",
            body[0]
        )));
    }
    let named_curve = u16::from_be_bytes([body[1], body[2]]);
    // ECPoint: opaque<1..255>
    let plen = body[3] as usize;
    if 4 + plen > body.len() {
        return Err(TlsError::Decode("ske: point len".into()));
    }
    let pubkey = body[4..4 + plen].to_vec();
    let signed_end = 4 + plen;
    let signed_params = body[..signed_end].to_vec();

    // SignatureAndHashAlgorithm(2) + signature opaque<2..>
    if body.len() < signed_end + 4 {
        return Err(TlsError::Decode("ske: sig hdr".into()));
    }
    let sig_scheme = u16::from_be_bytes([body[signed_end], body[signed_end + 1]]);
    let sig_len = u16::from_be_bytes([body[signed_end + 2], body[signed_end + 3]]) as usize;
    if signed_end + 4 + sig_len > body.len() {
        return Err(TlsError::Decode("ske: sig body".into()));
    }
    let sig = body[signed_end + 4..signed_end + 4 + sig_len].to_vec();
    Ok((named_curve, pubkey, signed_params, sig_scheme, sig))
}

fn verify_ske_signature(
    leaf: &x509::Cert<'_>,
    sig_scheme: u16,
    sig_input: &[u8],
    sig: &[u8],
) -> Result<(), TlsError> {
    // ECDSA branch — used by ECDHE-ECDSA suites (0xc02b/c02c/cca9).
    //
    // In TLS 1.2 the sig_scheme byte pair encodes *hash* + *signature
    // algorithm family*, NOT curve. The curve is whatever the leaf
    // cert's key actually uses (RFC 5246 §7.4.1.4.1 + RFC 8422 §5.4).
    // BBC's cert is P-256 with SKE signed by ECDSA-SHA-384 (sig_scheme
    // 0x0503) — a perfectly legal combination — so we dispatch the
    // verifier off the SPKI length, not the sig_scheme byte.
    let (sig_byte, hash_byte) = (sig_scheme & 0xff, (sig_scheme >> 8) & 0xff);
    if sig_byte == 0x03 {
        // ECDSA family. Pick the digest by the hash byte, then verify
        // with whichever curve matches the cert's SPKI.
        let pk = leaf.spki_key_bytes;
        let hash: Vec<u8> = match hash_byte {
            4 => cv_crypto::sha256::Sha256::oneshot(sig_input).to_vec(),
            5 => cv_crypto::sha384::Sha384::oneshot(sig_input).to_vec(),
            6 => cv_crypto::sha512::Sha512::oneshot(sig_input).to_vec(),
            _ => {
                return Err(TlsError::UnsupportedSignatureScheme(sig_scheme));
            }
        };
        return match pk.len() {
            65 if pk[0] == 0x04 => {
                let (qx, qy) = (&pk[1..33], &pk[33..65]);
                let (r, s) = cv_crypto::p256::parse_der_signature(sig)
                    .map_err(|_| TlsError::Decode("ecdsa P-256 sig der".into()))?;
                cv_crypto::p256::verify_prehashed(qx, qy, &hash, &r, &s)
                    .map_err(|_| TlsError::CertVerifyFailed)
            }
            97 if pk[0] == 0x04 => {
                let (qx, qy) = (&pk[1..49], &pk[49..97]);
                let (r, s) = cv_crypto::p384::parse_der_signature(sig)
                    .map_err(|_| TlsError::Decode("ecdsa P-384 sig der".into()))?;
                cv_crypto::p384::verify(qx, qy, sig_input, &r, &s)
                    .map_err(|_| TlsError::CertVerifyFailed)
            }
            other => Err(TlsError::Decode(format!(
                "ecdsa SPKI bad: len={other} first={:#04x}",
                pk.first().copied().unwrap_or(0)
            ))),
        };
    }
    // RSA branch — parse the modulus + exponent out of the SPKI key
    // bitstring (already strip-headered by x509::parse).
    let mut spki = cv_crypto::asn1::Reader::new(leaf.spki_key_bytes);
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
    let res = match sig_scheme {
        0x0401 => verify_pkcs1_v15(&key, RsaHash::Sha256, sig_input, sig),
        0x0501 => verify_pkcs1_v15(&key, RsaHash::Sha384, sig_input, sig),
        0x0601 => verify_pkcs1_v15(&key, RsaHash::Sha512, sig_input, sig),
        0x0804 => verify_pss(&key, RsaHash::Sha256, sig_input, sig),
        0x0805 => verify_pss(&key, RsaHash::Sha384, sig_input, sig),
        0x0806 => verify_pss(&key, RsaHash::Sha512, sig_input, sig),
        other => {
            return Err(TlsError::UnsupportedSignatureScheme(other));
        }
    };
    res.map_err(|_| TlsError::CertVerifyFailed)
}

/// TLS 1.2 PRF (RFC 5246 §5) dispatched by the cipher suite's MAC hash.
fn prf(kind: Tls12CipherKind, secret: &[u8], label: &[u8], seed: &[u8], n: usize) -> Vec<u8> {
    match kind {
        Tls12CipherKind::Aes256GcmSha384 => prf_h::<Sha384Mac>(secret, label, seed, n),
        _ => prf_h::<Sha256Mac>(secret, label, seed, n),
    }
}

/// HMAC type-class so we can write `prf_h::<H>` once. We can't be
/// generic over an Hmac function directly without a trait — define one
/// rather than copy-pasting the PRF body twice.
trait HmacAlg {
    fn mac(key: &[u8], msg: &[u8]) -> Vec<u8>;
}
struct Sha256Mac;
impl HmacAlg for Sha256Mac {
    fn mac(key: &[u8], msg: &[u8]) -> Vec<u8> {
        HmacSha256::oneshot(key, msg).to_vec()
    }
}
struct Sha384Mac;
impl HmacAlg for Sha384Mac {
    fn mac(key: &[u8], msg: &[u8]) -> Vec<u8> {
        HmacSha384::oneshot(key, msg).to_vec()
    }
}

fn prf_h<H: HmacAlg>(secret: &[u8], label: &[u8], seed: &[u8], n: usize) -> Vec<u8> {
    let mut label_seed = Vec::with_capacity(label.len() + seed.len());
    label_seed.extend_from_slice(label);
    label_seed.extend_from_slice(seed);
    let mut out: Vec<u8> = Vec::with_capacity(n);
    let mut a = H::mac(secret, &label_seed);
    while out.len() < n {
        let mut step_in = Vec::with_capacity(a.len() + label_seed.len());
        step_in.extend_from_slice(&a);
        step_in.extend_from_slice(&label_seed);
        let block = H::mac(secret, &step_in);
        let take = (n - out.len()).min(block.len());
        out.extend_from_slice(&block[..take]);
        a = H::mac(secret, &a);
    }
    out
}

fn transcript_hash(kind: Tls12CipherKind, t: &[u8]) -> Vec<u8> {
    match kind {
        Tls12CipherKind::Aes256GcmSha384 => Sha384::oneshot(t).to_vec(),
        _ => Sha256::oneshot(t).to_vec(),
    }
}

fn encrypt_record_12(
    kind: Tls12CipherKind,
    content_type: u8,
    plaintext: &[u8],
    key: &[u8],
    salt: &[u8],
    seq: u64,
) -> Result<Vec<u8>, TlsError> {
    let seq_bytes = seq.to_be_bytes();
    // AAD = seq(8) || type(1) || version(2) || plaintext_len(2)
    let mut aad = Vec::with_capacity(13);
    aad.extend_from_slice(&seq_bytes);
    aad.push(content_type);
    aad.extend_from_slice(&TLS12_VERSION.to_be_bytes());
    aad.extend_from_slice(&(plaintext.len() as u16).to_be_bytes());

    let ct_tag = match kind {
        Tls12CipherKind::Aes128GcmSha256 => {
            let mut nonce = [0u8; 12];
            nonce[..4].copy_from_slice(&salt[..4]);
            nonce[4..].copy_from_slice(&seq_bytes);
            let key_arr: [u8; 16] = key
                .try_into()
                .map_err(|_| TlsError::Crypto("aes128 key len".into()))?;
            Aes128Gcm::seal(&key_arr, &nonce, &aad, plaintext)
        }
        Tls12CipherKind::Aes256GcmSha384 => {
            let mut nonce = [0u8; 12];
            nonce[..4].copy_from_slice(&salt[..4]);
            nonce[4..].copy_from_slice(&seq_bytes);
            let key_arr: [u8; 32] = key
                .try_into()
                .map_err(|_| TlsError::Crypto("aes256 key len".into()))?;
            Aes256Gcm::seal(&key_arr, &nonce, &aad, plaintext)
        }
        Tls12CipherKind::ChaCha20Poly1305Sha256 => {
            // RFC 7905 §2: nonce = padded_seq XOR fixed_iv (no explicit_nonce on the wire).
            let mut nonce = [0u8; 12];
            nonce[..4].copy_from_slice(&[0; 4]);
            nonce[4..].copy_from_slice(&seq_bytes);
            for (i, b) in nonce.iter_mut().enumerate() {
                *b ^= salt[i];
            }
            let key_arr: [u8; 32] = key
                .try_into()
                .map_err(|_| TlsError::Crypto("chacha key len".into()))?;
            ChaCha20Poly1305::seal(&key_arr, &nonce, &aad, plaintext)
        }
    };

    let mut out = Vec::with_capacity(8 + ct_tag.len());
    if kind.explicit_nonce_len() == 8 {
        out.extend_from_slice(&seq_bytes); // explicit_nonce on wire
    }
    out.extend_from_slice(&ct_tag);
    Ok(out)
}

fn decrypt_record_12(
    kind: Tls12CipherKind,
    content_type: u8,
    fragment: &[u8],
    key: &[u8],
    salt: &[u8],
    seq: u64,
) -> Result<Vec<u8>, TlsError> {
    let exp = kind.explicit_nonce_len();
    if fragment.len() < exp + 16 {
        return Err(TlsError::Protocol("tls1.2 short record".into()));
    }
    let seq_bytes = seq.to_be_bytes();
    let mut nonce = [0u8; 12];
    let ct_and_tag = match kind {
        Tls12CipherKind::Aes128GcmSha256 | Tls12CipherKind::Aes256GcmSha384 => {
            nonce[..4].copy_from_slice(&salt[..4]);
            nonce[4..].copy_from_slice(&fragment[..8]);
            &fragment[8..]
        }
        Tls12CipherKind::ChaCha20Poly1305Sha256 => {
            let mut padded = [0u8; 12];
            padded[4..].copy_from_slice(&seq_bytes);
            for (i, b) in nonce.iter_mut().enumerate() {
                *b = salt[i] ^ padded[i];
            }
            &fragment[..]
        }
    };
    let pt_len = ct_and_tag.len() - 16;
    let mut aad = Vec::with_capacity(13);
    aad.extend_from_slice(&seq_bytes);
    aad.push(content_type);
    aad.extend_from_slice(&TLS12_VERSION.to_be_bytes());
    aad.extend_from_slice(&(pt_len as u16).to_be_bytes());
    match kind {
        Tls12CipherKind::Aes128GcmSha256 => {
            let key_arr: [u8; 16] = key
                .try_into()
                .map_err(|_| TlsError::Crypto("aes128 key len".into()))?;
            Aes128Gcm::open(&key_arr, &nonce, &aad, ct_and_tag)
                .map_err(|_| TlsError::Crypto("aead open".into()))
        }
        Tls12CipherKind::Aes256GcmSha384 => {
            let key_arr: [u8; 32] = key
                .try_into()
                .map_err(|_| TlsError::Crypto("aes256 key len".into()))?;
            Aes256Gcm::open(&key_arr, &nonce, &aad, ct_and_tag)
                .map_err(|_| TlsError::Crypto("aead open".into()))
        }
        Tls12CipherKind::ChaCha20Poly1305Sha256 => {
            let key_arr: [u8; 32] = key
                .try_into()
                .map_err(|_| TlsError::Crypto("chacha key len".into()))?;
            ChaCha20Poly1305::open(&key_arr, &nonce, &aad, ct_and_tag)
                .map_err(|_| TlsError::Crypto("aead open".into()))
        }
    }
}

struct RawRecord {
    content_type: u8,
    fragment: Vec<u8>,
}

fn read_one_record_plaintext(sock: &mut Socket) -> Result<RawRecord, TlsError> {
    let mut hdr = [0u8; 5];
    read_n(sock, &mut hdr)?;
    let len = u16::from_be_bytes([hdr[3], hdr[4]]) as usize;
    let mut frag = vec![0u8; len];
    read_n(sock, &mut frag)?;
    Ok(RawRecord {
        content_type: hdr[0],
        fragment: frag,
    })
}

fn read_n(sock: &mut Socket, buf: &mut [u8]) -> Result<(), TlsError> {
    let mut got = 0;
    while got < buf.len() {
        let n = sock
            .read(&mut buf[got..])
            .map_err(|e| TlsError::Protocol(format!("tls1.2 read: {e:?}")))?;
        if n == 0 {
            return Err(TlsError::Protocol("tls1.2 unexpected EOF".into()));
        }
        got += n;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// BoringSSL FIPS self-test vector for TLS 1.2 PRF-SHA256.
    /// Source: crypto/fipsmodule/self_check/self_check.cc.inc — the
    /// constants kTLS12Secret, kTLSLabel, kTLSSeed1, kTLSSeed2,
    /// kTLS12Output. The label is the 15-byte byte array
    /// "FIPS self test\0" (sizeof includes the null terminator —
    /// BoringSSL passes `sizeof(kTLSLabel)` so we do too).
    #[test]
    fn boringssl_fips_tls12_prf_sha256() {
        let secret: [u8; 32] = [
            0xc5, 0x43, 0x8e, 0xe2, 0x6f, 0xd4, 0xac, 0xbd, 0x25, 0x9f, 0xc9, 0x18, 0x55, 0xdc,
            0x69, 0xbf, 0x88, 0x4e, 0xe2, 0x93, 0x22, 0xfc, 0xbf, 0xd2, 0x96, 0x6a, 0x46, 0x23,
            0xd4, 0x2e, 0xc7, 0x81,
        ];
        let label = b"FIPS self test\0"; // 15 bytes incl. NUL
        let seed1: [u8; 16] = [
            0x8f, 0x0d, 0xe8, 0xb6, 0x90, 0x8f, 0xb1, 0xd2, 0x6d, 0x51, 0xf4, 0x79, 0x18, 0x63,
            0x51, 0x65,
        ];
        let seed2: [u8; 16] = [
            0x7d, 0x24, 0x1a, 0x9d, 0x3c, 0x59, 0xbf, 0x3c, 0x31, 0x1e, 0x2b, 0x21, 0x41, 0x8d,
            0x32, 0x81,
        ];
        let expected: [u8; 32] = [
            0xee, 0x4a, 0xcd, 0x3f, 0xa3, 0xd3, 0x55, 0x89, 0x9e, 0x6f, 0xf1, 0x38, 0x46, 0x9d,
            0x2b, 0x33, 0xaa, 0x7f, 0xc4, 0x7f, 0x51, 0x85, 0x8a, 0xf3, 0x13, 0x84, 0xbf, 0x53,
            0x6a, 0x65, 0x37, 0x51,
        ];
        // BoringSSL calls tls1_prf with seed1, seed2 as separate spans.
        // Our prf() takes a single combined seed.
        let mut combined_seed = Vec::new();
        combined_seed.extend_from_slice(&seed1);
        combined_seed.extend_from_slice(&seed2);
        let out = prf(
            Tls12CipherKind::Aes128GcmSha256,
            &secret,
            label,
            &combined_seed,
            32,
        );
        assert_eq!(
            &out[..],
            &expected[..],
            "PRF-SHA256 disagrees with BoringSSL FIPS vector"
        );
    }
}
