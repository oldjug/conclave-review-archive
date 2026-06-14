//! Renderer ↔ browser IPC wire protocol.
//!
//! Message catalog used over the named pipe `cv_ipc::SandboxedChild`
//! establishes between the browser-process broker and a sandboxed
//! renderer process spawned with `--type renderer`.
//!
//! Wire format (per RFC-style framing):
//!   magic (u32, MAGIC) | kind (u32) | payload_len (u32) | payload bytes
//!
//! All multi-byte integers are little-endian. Strings are UTF-8 with
//! a leading u32 length.
//!
//! Browser → Renderer messages:
//!   * `Handshake { protocol_version }` — sent immediately after pipe
//!     connection. Renderer replies with `HandshakeAck`.
//!   * `LayoutRequest { url, viewport_w, viewport_h, html }` — ask the
//!     renderer to lay out `html` against the given viewport.
//!   * `Shutdown` — graceful termination request.
//!
//! Renderer → Browser messages:
//!   * `HandshakeAck { renderer_version }`
//!   * `PaintReady { width, height, bitmap_bgra }` — raw BGRA pixel
//!     buffer ready for the parent to blit into the OS window.
//!   * `LogLine { level, text }` — diagnostic output (replaces direct
//!     stderr writes that would otherwise be eaten by the sandbox).
//!   * `Error { message }` — fatal render error.
//!
//! Persistent-renderer protocol (A2 — the cross-process analog of the
//! in-process `cv_ui::ToPage` / `cv_ui::FromPage` channels). After the
//! first-paint `LayoutRequest`/`PaintReady` smoke path, a persistent
//! renderer process owns the full !Send page graph and is driven by:
//!   Browser → Renderer:
//!     * `NavCmd { epoch, cmd }` — an encoded navigator command STRING
//!       (the `javascript:` / `tb-link-click:` / `tb-key:` / `tb-mouse:`
//!       / plain-URL encoding the navigator already understands), so input
//!       handling needs no reimplementation across the boundary.
//!     * `ResizePage { epoch, w, h }` — viewport resize.
//!     * `HostCmd { epoch, payload }` — tab/window chrome command. The
//!       payload is a `cv_ui::HostCommand` serialized by the browser-side
//!       codec; `cv_ipc` stays free of any `cv_ui` dependency.
//!   Renderer → Browser:
//!     * `CommitFrame { epoch, payload }` — a finished, self-contained
//!       frame (bitmap + chrome overlays + hit regions + url/title/tabs
//!       metadata) serialized by the browser-side codec. `epoch` rides
//!       OUTSIDE the opaque payload so the stale-frame drop survives the
//!       boundary exactly like the in-process `FromPage::Commit { epoch }`.
//!
//! `epoch` is bumped by the browser on every navigation; the UI drops any
//! `CommitFrame` whose epoch is older than the current navigation, which
//! is the race-free replacement for the single-thread `nav_in_flight`
//! guard. The opaque `payload` keeps the heavy, `cv_ui`-typed marshaling
//! in `conclave` (which sees both crates) while `cv_ipc` only frames
//! and length-guards the bytes.

use crate::MAGIC;
use crate::codec::{Decode, DecodeError, Encode, Reader, Writer};

/// Current renderer ↔ browser wire-protocol version. Bumped whenever a
/// breaking change is made to the `Msg` catalog. The browser sends this
/// in its opening `Handshake`; the renderer verifies compatibility
/// (`protocol_compatible`) before replying with `HandshakeAck` carrying
/// the version IT speaks. A mismatch is a hard, honest failure rather
/// than a silent best-effort — a half-understood wire is worse than a
/// clean refusal.
pub const PROTOCOL_VERSION: u32 = 1;

/// Whether a peer speaking `peer_version` can interoperate with this
/// build. V1 requires an exact match — the protocol is young and we do
/// not yet keep back-compat shims. Centralised here so both ends apply
/// the identical rule and a future "accept N and N-1" policy lives in
/// one place.
#[must_use]
pub fn protocol_compatible(peer_version: u32) -> bool {
    peer_version == PROTOCOL_VERSION
}

const MAX_PAINT_DIMENSION: u32 = 16_384;
const MAX_PAINT_BYTES: usize = 256 * 1024 * 1024;

/// Upper bound on an opaque persistent-protocol payload (a marshaled
/// frame or a host command). A full-document BGRA frame can be tens of
/// MB; the cap is generous (300MB ≥ a 16384²×4 frame plus its chrome/hit
/// metadata) but bounded so a corrupt length cannot drive an unbounded
/// allocation. Frames over this are rejected as malformed, the same hard
/// refusal `PaintReady` applies to oversized pixel buffers.
const MAX_OPAQUE_PAYLOAD_BYTES: usize = 300 * 1024 * 1024;

fn expected_paint_bytes(width: u32, height: u32) -> Option<usize> {
    if width == 0 || height == 0 {
        return None;
    }
    if width > MAX_PAINT_DIMENSION || height > MAX_PAINT_DIMENSION {
        return None;
    }
    let pixels = (width as usize).checked_mul(height as usize)?;
    let bytes = pixels.checked_mul(4)?;
    if bytes > MAX_PAINT_BYTES {
        return None;
    }
    Some(bytes)
}

/// Wire-level message kind tag. Stable across versions — we only ever
/// add to the end. Reserved bits (top 16) for the message direction
/// future protocols.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum MsgKind {
    Handshake = 0x0001,
    HandshakeAck = 0x0002,
    LayoutRequest = 0x0010,
    PaintReady = 0x0011,
    LogLine = 0x0020,
    ErrorMsg = 0x0030,
    // Persistent-renderer protocol (A2). Appended after the first-paint
    // catalog to keep the existing tags stable.
    NavCmd = 0x0040,
    ResizePage = 0x0041,
    HostCmd = 0x0042,
    CommitFrame = 0x0043,
    Shutdown = 0x00FF,
}

impl MsgKind {
    pub fn from_u32(v: u32) -> Option<Self> {
        Some(match v {
            0x0001 => Self::Handshake,
            0x0002 => Self::HandshakeAck,
            0x0010 => Self::LayoutRequest,
            0x0011 => Self::PaintReady,
            0x0020 => Self::LogLine,
            0x0030 => Self::ErrorMsg,
            0x0040 => Self::NavCmd,
            0x0041 => Self::ResizePage,
            0x0042 => Self::HostCmd,
            0x0043 => Self::CommitFrame,
            0x00FF => Self::Shutdown,
            _ => return None,
        })
    }
}

/// Decoded message bodies. Each variant corresponds 1:1 to a
/// `MsgKind`. The encode/decode pair is in `encode_*` / `decode_*`
/// below.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Msg {
    Handshake {
        protocol_version: u32,
    },
    HandshakeAck {
        renderer_version: u32,
    },
    LayoutRequest {
        url: String,
        viewport_w: u32,
        viewport_h: u32,
        html: Vec<u8>,
    },
    PaintReady {
        width: u32,
        height: u32,
        bitmap_bgra: Vec<u8>,
    },
    LogLine {
        level: u32,
        text: String,
    },
    ErrorMsg {
        message: String,
    },
    /// Persistent protocol: an encoded navigator command string for the
    /// running page (input, link click, JS, plain-URL navigation).
    NavCmd {
        epoch: u64,
        cmd: String,
    },
    /// Persistent protocol: viewport resize.
    ResizePage {
        epoch: u64,
        w: u32,
        h: u32,
    },
    /// Persistent protocol: a tab/window chrome command, opaque payload
    /// serialized by the browser-side codec (a `cv_ui::HostCommand`).
    HostCmd {
        epoch: u64,
        payload: Vec<u8>,
    },
    /// Persistent protocol: a finished frame, opaque payload serialized by
    /// the browser-side codec (bitmap + chrome overlays + hit regions +
    /// url/title/tabs metadata). `epoch` rides outside the payload so the
    /// browser can drop a stale frame without decoding the (large) bytes.
    CommitFrame {
        epoch: u64,
        payload: Vec<u8>,
    },
    Shutdown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MsgDirection {
    BrowserToRenderer,
    RendererToBrowser,
}

impl Msg {
    #[must_use]
    pub fn direction(&self) -> MsgDirection {
        match self {
            Self::Handshake { .. } => MsgDirection::BrowserToRenderer,
            Self::LayoutRequest { .. } => MsgDirection::BrowserToRenderer,
            Self::NavCmd { .. } => MsgDirection::BrowserToRenderer,
            Self::ResizePage { .. } => MsgDirection::BrowserToRenderer,
            Self::HostCmd { .. } => MsgDirection::BrowserToRenderer,
            Self::Shutdown => MsgDirection::BrowserToRenderer,
            Self::HandshakeAck { .. } => MsgDirection::RendererToBrowser,
            Self::PaintReady { .. } => MsgDirection::RendererToBrowser,
            Self::CommitFrame { .. } => MsgDirection::RendererToBrowser,
            Self::LogLine { .. } => MsgDirection::RendererToBrowser,
            Self::ErrorMsg { .. } => MsgDirection::RendererToBrowser,
        }
    }

    #[must_use]
    pub fn is_valid_for_direction(&self, direction: MsgDirection) -> bool {
        self.direction() == direction
    }
}

/// Encode a message into a self-framed byte vector ready to write to
/// the pipe. The reader uses `decode_message` to peel it back.
pub fn encode_message(msg: &Msg) -> Vec<u8> {
    let (kind, payload) = match msg {
        Msg::Handshake { protocol_version } => {
            (MsgKind::Handshake, protocol_version.to_le_bytes().to_vec())
        }
        Msg::HandshakeAck { renderer_version } => (
            MsgKind::HandshakeAck,
            renderer_version.to_le_bytes().to_vec(),
        ),
        Msg::LayoutRequest {
            url,
            viewport_w,
            viewport_h,
            html,
        } => {
            let mut p = Vec::with_capacity(url.len() + html.len() + 16);
            encode_str(&mut p, url);
            p.extend_from_slice(&viewport_w.to_le_bytes());
            p.extend_from_slice(&viewport_h.to_le_bytes());
            encode_bytes(&mut p, html);
            (MsgKind::LayoutRequest, p)
        }
        Msg::PaintReady {
            width,
            height,
            bitmap_bgra,
        } => {
            let mut p = Vec::with_capacity(bitmap_bgra.len() + 8);
            p.extend_from_slice(&width.to_le_bytes());
            p.extend_from_slice(&height.to_le_bytes());
            encode_bytes(&mut p, bitmap_bgra);
            (MsgKind::PaintReady, p)
        }
        Msg::LogLine { level, text } => {
            let mut p = Vec::with_capacity(text.len() + 8);
            p.extend_from_slice(&level.to_le_bytes());
            encode_str(&mut p, text);
            (MsgKind::LogLine, p)
        }
        Msg::ErrorMsg { message } => {
            let mut p = Vec::with_capacity(message.len() + 4);
            encode_str(&mut p, message);
            (MsgKind::ErrorMsg, p)
        }
        Msg::NavCmd { epoch, cmd } => {
            let mut p = Vec::with_capacity(cmd.len() + 12);
            p.extend_from_slice(&epoch.to_le_bytes());
            encode_str(&mut p, cmd);
            (MsgKind::NavCmd, p)
        }
        Msg::ResizePage { epoch, w, h } => {
            let mut p = Vec::with_capacity(16);
            p.extend_from_slice(&epoch.to_le_bytes());
            p.extend_from_slice(&w.to_le_bytes());
            p.extend_from_slice(&h.to_le_bytes());
            (MsgKind::ResizePage, p)
        }
        Msg::HostCmd { epoch, payload } => {
            let mut p = Vec::with_capacity(payload.len() + 12);
            p.extend_from_slice(&epoch.to_le_bytes());
            encode_bytes(&mut p, payload);
            (MsgKind::HostCmd, p)
        }
        Msg::CommitFrame { epoch, payload } => {
            let mut p = Vec::with_capacity(payload.len() + 12);
            p.extend_from_slice(&epoch.to_le_bytes());
            encode_bytes(&mut p, payload);
            (MsgKind::CommitFrame, p)
        }
        Msg::Shutdown => (MsgKind::Shutdown, Vec::new()),
    };
    let mut out = Vec::with_capacity(12 + payload.len());
    out.extend_from_slice(&MAGIC.to_le_bytes());
    out.extend_from_slice(&(kind as u32).to_le_bytes());
    out.extend_from_slice(&(payload.len() as u32).to_le_bytes());
    out.extend_from_slice(&payload);
    out
}

/// Decode a single message from `buf`. Returns the decoded `Msg` and
/// the number of bytes consumed (so the caller can slide its read
/// buffer forward). Returns `None` if `buf` doesn't contain a
/// complete message yet — callers should append more bytes and retry.
pub fn decode_message(buf: &[u8]) -> Option<(Msg, usize)> {
    if buf.len() < 12 {
        return None;
    }
    let magic = u32::from_le_bytes(buf[0..4].try_into().unwrap());
    if magic != MAGIC {
        return None;
    }
    let kind_raw = u32::from_le_bytes(buf[4..8].try_into().unwrap());
    let payload_len = u32::from_le_bytes(buf[8..12].try_into().unwrap()) as usize;
    if buf.len() < 12 + payload_len {
        return None;
    }
    let p = &buf[12..12 + payload_len];
    let kind = MsgKind::from_u32(kind_raw)?;
    let msg = match kind {
        MsgKind::Handshake => {
            if p.len() < 4 {
                return None;
            }
            Msg::Handshake {
                protocol_version: u32::from_le_bytes(p[0..4].try_into().unwrap()),
            }
        }
        MsgKind::HandshakeAck => {
            if p.len() < 4 {
                return None;
            }
            Msg::HandshakeAck {
                renderer_version: u32::from_le_bytes(p[0..4].try_into().unwrap()),
            }
        }
        MsgKind::LayoutRequest => {
            let (url, mut off) = decode_str(p)?;
            if p.len() < off + 8 {
                return None;
            }
            let viewport_w = u32::from_le_bytes(p[off..off + 4].try_into().unwrap());
            off += 4;
            let viewport_h = u32::from_le_bytes(p[off..off + 4].try_into().unwrap());
            off += 4;
            let (html, _) = decode_bytes(&p[off..])?;
            Msg::LayoutRequest {
                url,
                viewport_w,
                viewport_h,
                html,
            }
        }
        MsgKind::PaintReady => {
            if p.len() < 8 {
                return None;
            }
            let width = u32::from_le_bytes(p[0..4].try_into().unwrap());
            let height = u32::from_le_bytes(p[4..8].try_into().unwrap());
            let (bitmap_bgra, _) = decode_bytes(&p[8..])?;
            if expected_paint_bytes(width, height)? != bitmap_bgra.len() {
                return None;
            }
            Msg::PaintReady {
                width,
                height,
                bitmap_bgra,
            }
        }
        MsgKind::LogLine => {
            if p.len() < 4 {
                return None;
            }
            let level = u32::from_le_bytes(p[0..4].try_into().unwrap());
            let (text, _) = decode_str(&p[4..])?;
            Msg::LogLine { level, text }
        }
        MsgKind::ErrorMsg => {
            let (message, _) = decode_str(p)?;
            Msg::ErrorMsg { message }
        }
        MsgKind::NavCmd => {
            if p.len() < 8 {
                return None;
            }
            let epoch = u64::from_le_bytes(p[0..8].try_into().unwrap());
            let (cmd, _) = decode_str(&p[8..])?;
            Msg::NavCmd { epoch, cmd }
        }
        MsgKind::ResizePage => {
            if p.len() < 16 {
                return None;
            }
            let epoch = u64::from_le_bytes(p[0..8].try_into().unwrap());
            let w = u32::from_le_bytes(p[8..12].try_into().unwrap());
            let h = u32::from_le_bytes(p[12..16].try_into().unwrap());
            Msg::ResizePage { epoch, w, h }
        }
        MsgKind::HostCmd => {
            if p.len() < 8 {
                return None;
            }
            let epoch = u64::from_le_bytes(p[0..8].try_into().unwrap());
            let (payload, _) = decode_bytes(&p[8..])?;
            if payload.len() > MAX_OPAQUE_PAYLOAD_BYTES {
                return None;
            }
            Msg::HostCmd { epoch, payload }
        }
        MsgKind::CommitFrame => {
            if p.len() < 8 {
                return None;
            }
            let epoch = u64::from_le_bytes(p[0..8].try_into().unwrap());
            let (payload, _) = decode_bytes(&p[8..])?;
            if payload.len() > MAX_OPAQUE_PAYLOAD_BYTES {
                return None;
            }
            Msg::CommitFrame { epoch, payload }
        }
        MsgKind::Shutdown => Msg::Shutdown,
    };
    Some((msg, 12 + payload_len))
}

fn encode_str(out: &mut Vec<u8>, s: &str) {
    encode_bytes(out, s.as_bytes());
}

fn encode_bytes(out: &mut Vec<u8>, b: &[u8]) {
    out.extend_from_slice(&(b.len() as u32).to_le_bytes());
    out.extend_from_slice(b);
}

fn decode_str(p: &[u8]) -> Option<(String, usize)> {
    let (b, consumed) = decode_bytes(p)?;
    String::from_utf8(b).ok().map(|s| (s, consumed))
}

fn decode_bytes(p: &[u8]) -> Option<(Vec<u8>, usize)> {
    if p.len() < 4 {
        return None;
    }
    let len = u32::from_le_bytes(p[0..4].try_into().unwrap()) as usize;
    if p.len() < 4 + len {
        return None;
    }
    Some((p[4..4 + len].to_vec(), 4 + len))
}

// -------- cv_ipc Encode/Decode bridge -------------------------------------
//
// In addition to the raw `encode_message` / `decode_message` byte API above
// (handy for write-to-buffer and unit testing), we implement Encode/Decode
// so a `Msg` can ride directly on a `NamedPipeEndpoint`. The outer
// `magic|id|payload_len` frame is supplied by the pipe; our payload is a
// 1-byte tag plus tagged body.

impl Encode for Msg {
    fn encode(&self, w: &mut Writer) {
        match self {
            Msg::Handshake { protocol_version } => {
                w.write_u8(MsgKind::Handshake as u32 as u8);
                w.write_u32(*protocol_version);
            }
            Msg::HandshakeAck { renderer_version } => {
                w.write_u8(MsgKind::HandshakeAck as u32 as u8);
                w.write_u32(*renderer_version);
            }
            Msg::LayoutRequest {
                url,
                viewport_w,
                viewport_h,
                html,
            } => {
                w.write_u8(MsgKind::LayoutRequest as u32 as u8);
                w.write_str(url);
                w.write_u32(*viewport_w);
                w.write_u32(*viewport_h);
                w.write_bytes(html);
            }
            Msg::PaintReady {
                width,
                height,
                bitmap_bgra,
            } => {
                w.write_u8(MsgKind::PaintReady as u32 as u8);
                w.write_u32(*width);
                w.write_u32(*height);
                w.write_bytes(bitmap_bgra);
            }
            Msg::LogLine { level, text } => {
                w.write_u8(MsgKind::LogLine as u32 as u8);
                w.write_u32(*level);
                w.write_str(text);
            }
            Msg::ErrorMsg { message } => {
                w.write_u8(MsgKind::ErrorMsg as u32 as u8);
                w.write_str(message);
            }
            Msg::NavCmd { epoch, cmd } => {
                w.write_u8(MsgKind::NavCmd as u32 as u8);
                w.write_u64(*epoch);
                w.write_str(cmd);
            }
            Msg::ResizePage { epoch, w: vw, h: vh } => {
                w.write_u8(MsgKind::ResizePage as u32 as u8);
                w.write_u64(*epoch);
                w.write_u32(*vw);
                w.write_u32(*vh);
            }
            Msg::HostCmd { epoch, payload } => {
                w.write_u8(MsgKind::HostCmd as u32 as u8);
                w.write_u64(*epoch);
                w.write_bytes(payload);
            }
            Msg::CommitFrame { epoch, payload } => {
                w.write_u8(MsgKind::CommitFrame as u32 as u8);
                w.write_u64(*epoch);
                w.write_bytes(payload);
            }
            Msg::Shutdown => {
                w.write_u8(MsgKind::Shutdown as u32 as u8);
            }
        }
    }
}

impl Decode for Msg {
    fn decode(r: &mut Reader<'_>) -> Result<Self, DecodeError> {
        let tag = r.read_u8()?;
        let kind = MsgKind::from_u32(tag as u32).ok_or(DecodeError::OutOfRange)?;
        Ok(match kind {
            MsgKind::Handshake => Msg::Handshake {
                protocol_version: r.read_u32()?,
            },
            MsgKind::HandshakeAck => Msg::HandshakeAck {
                renderer_version: r.read_u32()?,
            },
            MsgKind::LayoutRequest => Msg::LayoutRequest {
                url: r.read_str()?,
                viewport_w: r.read_u32()?,
                viewport_h: r.read_u32()?,
                html: r.read_bytes()?,
            },
            MsgKind::PaintReady => {
                let width = r.read_u32()?;
                let height = r.read_u32()?;
                let bitmap_bgra = r.read_bytes()?;
                if expected_paint_bytes(width, height)
                    .filter(|expected| *expected == bitmap_bgra.len())
                    .is_none()
                {
                    return Err(DecodeError::OutOfRange);
                }
                Msg::PaintReady {
                    width,
                    height,
                    bitmap_bgra,
                }
            }
            MsgKind::LogLine => Msg::LogLine {
                level: r.read_u32()?,
                text: r.read_str()?,
            },
            MsgKind::ErrorMsg => Msg::ErrorMsg {
                message: r.read_str()?,
            },
            MsgKind::NavCmd => Msg::NavCmd {
                epoch: r.read_u64()?,
                cmd: r.read_str()?,
            },
            MsgKind::ResizePage => Msg::ResizePage {
                epoch: r.read_u64()?,
                w: r.read_u32()?,
                h: r.read_u32()?,
            },
            MsgKind::HostCmd => {
                let epoch = r.read_u64()?;
                let payload = r.read_bytes()?;
                if payload.len() > MAX_OPAQUE_PAYLOAD_BYTES {
                    return Err(DecodeError::OutOfRange);
                }
                Msg::HostCmd { epoch, payload }
            }
            MsgKind::CommitFrame => {
                let epoch = r.read_u64()?;
                let payload = r.read_bytes()?;
                if payload.len() > MAX_OPAQUE_PAYLOAD_BYTES {
                    return Err(DecodeError::OutOfRange);
                }
                Msg::CommitFrame { epoch, payload }
            }
            MsgKind::Shutdown => Msg::Shutdown,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn handshake_roundtrip() {
        let m = Msg::Handshake {
            protocol_version: 7,
        };
        let wire = encode_message(&m);
        let (decoded, consumed) = decode_message(&wire).unwrap();
        assert_eq!(decoded, m);
        assert_eq!(consumed, wire.len());
    }

    #[test]
    fn layout_request_roundtrip() {
        let m = Msg::LayoutRequest {
            url: "https://example.com/x".into(),
            viewport_w: 1024,
            viewport_h: 768,
            html: b"<html><body>hi</body></html>".to_vec(),
        };
        let wire = encode_message(&m);
        let (decoded, _) = decode_message(&wire).unwrap();
        assert_eq!(decoded, m);
    }

    #[test]
    fn paint_ready_roundtrip() {
        let m = Msg::PaintReady {
            width: 2,
            height: 2,
            // 2x2 BGRA: red, green, blue, white.
            bitmap_bgra: vec![
                0, 0, 255, 255, 0, 255, 0, 255, 255, 0, 0, 255, 255, 255, 255, 255,
            ],
        };
        let wire = encode_message(&m);
        let (decoded, _) = decode_message(&wire).unwrap();
        assert_eq!(decoded, m);
    }

    #[test]
    fn malformed_paint_ready_length_is_rejected() {
        let m = Msg::PaintReady {
            width: 2,
            height: 2,
            bitmap_bgra: vec![0; 12],
        };
        let wire = encode_message(&m);
        assert!(decode_message(&wire).is_none());
    }

    #[test]
    fn oversized_paint_ready_dimensions_are_rejected() {
        let mut w = Writer::new();
        w.write_u8(MsgKind::PaintReady as u32 as u8);
        w.write_u32(MAX_PAINT_DIMENSION + 1);
        w.write_u32(1);
        w.write_bytes(&[0, 0, 0, 0]);

        let bytes = w.into_bytes();
        let mut r = Reader::new(&bytes);
        assert_eq!(Msg::decode(&mut r), Err(DecodeError::OutOfRange));
    }

    #[test]
    fn log_and_error_roundtrip() {
        let m1 = Msg::LogLine {
            level: 3,
            text: "renderer paint complete".into(),
        };
        let m2 = Msg::ErrorMsg {
            message: "out of GPU memory".into(),
        };
        for m in [m1, m2] {
            let w = encode_message(&m);
            let (d, _) = decode_message(&w).unwrap();
            assert_eq!(d, m);
        }
    }

    #[test]
    fn message_direction_validation_matches_protocol_roles() {
        let browser_msg = Msg::LayoutRequest {
            url: "https://example.com".into(),
            viewport_w: 800,
            viewport_h: 600,
            html: b"<p>x</p>".to_vec(),
        };
        assert!(browser_msg.is_valid_for_direction(MsgDirection::BrowserToRenderer));
        assert!(!browser_msg.is_valid_for_direction(MsgDirection::RendererToBrowser));

        let renderer_msg = Msg::PaintReady {
            width: 1,
            height: 1,
            bitmap_bgra: vec![0, 0, 0, 255],
        };
        assert!(renderer_msg.is_valid_for_direction(MsgDirection::RendererToBrowser));
        assert!(!renderer_msg.is_valid_for_direction(MsgDirection::BrowserToRenderer));
    }

    #[test]
    fn shutdown_has_no_payload() {
        let m = Msg::Shutdown;
        let wire = encode_message(&m);
        assert_eq!(wire.len(), 12);
        let (decoded, _) = decode_message(&wire).unwrap();
        assert_eq!(decoded, m);
    }

    #[test]
    fn truncated_buffer_returns_none() {
        let m = Msg::Handshake {
            protocol_version: 7,
        };
        let wire = encode_message(&m);
        assert!(decode_message(&wire[..11]).is_none());
        assert!(decode_message(&wire[..15]).is_none());
    }

    #[test]
    fn bad_magic_rejected() {
        let mut wire = encode_message(&Msg::Shutdown);
        wire[0] = 0xDE; // corrupt magic
        assert!(decode_message(&wire).is_none());
    }

    #[test]
    fn protocol_compatible_only_accepts_exact_version() {
        assert!(protocol_compatible(PROTOCOL_VERSION));
        assert!(!protocol_compatible(PROTOCOL_VERSION + 1));
        assert!(!protocol_compatible(PROTOCOL_VERSION.wrapping_sub(1)));
    }

    // ---- A2: persistent-renderer protocol round-trips -------------------

    /// Round-trip a message through BOTH the raw `encode_message` /
    /// `decode_message` byte API AND the `Encode`/`Decode`-over-`Reader`
    /// pipe API, asserting byte-identity on both paths. The two encoders
    /// share the catalog but frame differently (self-framed vs pipe-framed),
    /// so exercising both guards against a variant being added to one and
    /// not the other.
    fn assert_msg_roundtrips_both_apis(m: &Msg) {
        let wire = encode_message(m);
        let (decoded, consumed) = decode_message(&wire).expect("decode_message");
        assert_eq!(&decoded, m, "encode_message/decode_message round-trip");
        assert_eq!(consumed, wire.len());

        let mut w = Writer::new();
        m.encode(&mut w);
        let bytes = w.into_bytes();
        let mut r = Reader::new(&bytes);
        let decoded2 = Msg::decode(&mut r).expect("Msg::decode");
        assert_eq!(&decoded2, m, "Encode/Decode pipe round-trip");
        assert_eq!(r.remaining(), 0, "decoder must consume the whole payload");
    }

    #[test]
    fn nav_cmd_roundtrip_preserves_epoch_and_cmd() {
        let m = Msg::NavCmd {
            epoch: 0xDEAD_BEEF_0000_0042,
            cmd: "javascript:7|||onclick()|||nav:https://example.com/next".into(),
        };
        assert_msg_roundtrips_both_apis(&m);
    }

    #[test]
    fn resize_page_roundtrip_preserves_dimensions() {
        let m = Msg::ResizePage {
            epoch: 5,
            w: 1920,
            h: 1080,
        };
        assert_msg_roundtrips_both_apis(&m);
    }

    #[test]
    fn host_cmd_roundtrip_preserves_opaque_payload() {
        let m = Msg::HostCmd {
            epoch: 9,
            payload: vec![1, 2, 3, 0, 255, 128],
        };
        assert_msg_roundtrips_both_apis(&m);
    }

    #[test]
    fn commit_frame_roundtrip_preserves_epoch_and_payload() {
        // A nontrivial opaque payload, larger than a frame header, to prove
        // the length-prefixed body re-assembles byte-identically.
        let payload: Vec<u8> = (0..4096u32).map(|i| (i % 251) as u8).collect();
        let m = Msg::CommitFrame { epoch: 123, payload };
        assert_msg_roundtrips_both_apis(&m);
    }

    #[test]
    fn persistent_message_directions_match_protocol_roles() {
        // Browser → Renderer.
        for m in [
            Msg::NavCmd {
                epoch: 1,
                cmd: "https://example.com".into(),
            },
            Msg::ResizePage {
                epoch: 1,
                w: 800,
                h: 600,
            },
            Msg::HostCmd {
                epoch: 1,
                payload: vec![0],
            },
        ] {
            assert!(m.is_valid_for_direction(MsgDirection::BrowserToRenderer));
            assert!(!m.is_valid_for_direction(MsgDirection::RendererToBrowser));
        }
        // Renderer → Browser.
        let commit = Msg::CommitFrame {
            epoch: 1,
            payload: vec![0, 1, 2],
        };
        assert!(commit.is_valid_for_direction(MsgDirection::RendererToBrowser));
        assert!(!commit.is_valid_for_direction(MsgDirection::BrowserToRenderer));
    }

    #[test]
    fn oversized_opaque_payload_is_rejected_by_pipe_decoder() {
        // Forge a CommitFrame frame whose declared body length exceeds the
        // cap. We can't allocate 300MB in a test, so hand-build a frame that
        // *claims* an over-cap length but supplies fewer bytes — the decoder
        // must refuse it (either as truncated or out-of-range, never accept).
        let mut w = Writer::new();
        w.write_u8(MsgKind::CommitFrame as u32 as u8);
        w.write_u64(1); // epoch
        // write_bytes writes a u32 length prefix then the bytes; forge a
        // length larger than the cap without the matching payload.
        w.write_u32((MAX_OPAQUE_PAYLOAD_BYTES as u32).wrapping_add(1));
        let bytes = w.into_bytes();
        let mut r = Reader::new(&bytes);
        assert!(Msg::decode(&mut r).is_err());
    }

    #[test]
    fn handshake_carries_protocol_version_over_the_wire() {
        // The browser opens with PROTOCOL_VERSION; the renderer must be
        // able to read it back byte-identically to verify compatibility.
        let m = Msg::Handshake {
            protocol_version: PROTOCOL_VERSION,
        };
        let wire = encode_message(&m);
        let (decoded, _) = decode_message(&wire).unwrap();
        match decoded {
            Msg::Handshake { protocol_version } => {
                assert_eq!(protocol_version, PROTOCOL_VERSION);
                assert!(protocol_compatible(protocol_version));
            }
            other => panic!("expected Handshake, got {other:?}"),
        }
    }
}
