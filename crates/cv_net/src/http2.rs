//! HTTP/2 framing per RFC 9113 — V1 framing primitives.
//!
//! Ships the wire-level structures needed to negotiate HTTP/2 over
//! TLS via ALPN h2:
//!   - Connection preface bytes.
//!   - Frame header + typed enum (DATA, HEADERS, SETTINGS, PING,
//!     GOAWAY, WINDOW_UPDATE, RST_STREAM, PRIORITY, PUSH_PROMISE,
//!     CONTINUATION).
//!   - Frame encode / decode for the fixed-size headers + body slice.
//!   - SETTINGS payload encode/decode.
//!   - HPACK header field representation with the most common static-
//!     table indexed forms + literal-with-incremental-indexing.
//!
//! Higher layers (streams, flow control, multiplexing) are bounded by
//! the design in RFC 9113 §5.1/6 and will land on top of this without
//! re-shaping the byte-level layer.

/// The fixed connection preface every HTTP/2 client sends as the
/// first bytes after a successful upgrade / direct TLS connection.
pub const CONNECTION_PREFACE: &[u8] = b"PRI * HTTP/2.0\r\n\r\nSM\r\n\r\n";

/// Frame type byte values per RFC 9113 §6.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameType {
    Data,
    Headers,
    Priority,
    RstStream,
    Settings,
    PushPromise,
    Ping,
    Goaway,
    WindowUpdate,
    Continuation,
    Unknown(u8),
}

impl FrameType {
    pub fn from_u8(b: u8) -> Self {
        match b {
            0x0 => Self::Data,
            0x1 => Self::Headers,
            0x2 => Self::Priority,
            0x3 => Self::RstStream,
            0x4 => Self::Settings,
            0x5 => Self::PushPromise,
            0x6 => Self::Ping,
            0x7 => Self::Goaway,
            0x8 => Self::WindowUpdate,
            0x9 => Self::Continuation,
            other => Self::Unknown(other),
        }
    }
    pub fn to_u8(self) -> u8 {
        match self {
            Self::Data => 0x0,
            Self::Headers => 0x1,
            Self::Priority => 0x2,
            Self::RstStream => 0x3,
            Self::Settings => 0x4,
            Self::PushPromise => 0x5,
            Self::Ping => 0x6,
            Self::Goaway => 0x7,
            Self::WindowUpdate => 0x8,
            Self::Continuation => 0x9,
            Self::Unknown(b) => b,
        }
    }
}

/// 9-byte HTTP/2 frame header.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FrameHeader {
    pub length: u32,    // 24-bit
    pub typ: FrameType, // 8-bit
    pub flags: u8,
    pub stream_id: u32, // top bit reserved
}

impl FrameHeader {
    pub fn encode(self, out: &mut Vec<u8>) {
        let l = self.length & 0x00FF_FFFF;
        out.push(((l >> 16) & 0xFF) as u8);
        out.push(((l >> 8) & 0xFF) as u8);
        out.push((l & 0xFF) as u8);
        out.push(self.typ.to_u8());
        out.push(self.flags);
        let sid = self.stream_id & 0x7FFF_FFFF;
        out.extend_from_slice(&sid.to_be_bytes());
    }
    pub fn decode(buf: &[u8]) -> Option<Self> {
        if buf.len() < 9 {
            return None;
        }
        let length = ((buf[0] as u32) << 16) | ((buf[1] as u32) << 8) | buf[2] as u32;
        let typ = FrameType::from_u8(buf[3]);
        let flags = buf[4];
        let raw_sid = u32::from_be_bytes([buf[5], buf[6], buf[7], buf[8]]);
        Some(Self {
            length,
            typ,
            flags,
            stream_id: raw_sid & 0x7FFF_FFFF,
        })
    }
}

// ----------------------------------------------------------------------
// SETTINGS — RFC 9113 §6.5
// ----------------------------------------------------------------------

/// SETTINGS parameter ids the client cares about.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SettingId {
    HeaderTableSize = 0x1,
    EnablePush = 0x2,
    MaxConcurrentStreams = 0x3,
    InitialWindowSize = 0x4,
    MaxFrameSize = 0x5,
    MaxHeaderListSize = 0x6,
}

pub fn encode_settings(out: &mut Vec<u8>, settings: &[(u16, u32)]) {
    for (id, val) in settings {
        out.extend_from_slice(&id.to_be_bytes());
        out.extend_from_slice(&val.to_be_bytes());
    }
}

pub fn decode_settings(body: &[u8]) -> Vec<(u16, u32)> {
    let mut out = Vec::new();
    let mut i = 0;
    while i + 6 <= body.len() {
        let id = u16::from_be_bytes([body[i], body[i + 1]]);
        let val = u32::from_be_bytes([body[i + 2], body[i + 3], body[i + 4], body[i + 5]]);
        out.push((id, val));
        i += 6;
    }
    out
}

/// Build the bytes of a complete SETTINGS frame (header + body).
pub fn build_settings_frame(settings: &[(u16, u32)], ack: bool) -> Vec<u8> {
    let mut body = Vec::with_capacity(settings.len() * 6);
    if !ack {
        encode_settings(&mut body, settings);
    }
    let flags = if ack { 0x1 } else { 0x0 };
    let mut wire = Vec::with_capacity(9 + body.len());
    FrameHeader {
        length: body.len() as u32,
        typ: FrameType::Settings,
        flags,
        stream_id: 0,
    }
    .encode(&mut wire);
    wire.extend_from_slice(&body);
    wire
}

/// Build a PING frame body (8 bytes opaque data + ACK flag).
pub fn build_ping_frame(data: [u8; 8], ack: bool) -> Vec<u8> {
    let mut wire = Vec::with_capacity(17);
    FrameHeader {
        length: 8,
        typ: FrameType::Ping,
        flags: if ack { 0x1 } else { 0x0 },
        stream_id: 0,
    }
    .encode(&mut wire);
    wire.extend_from_slice(&data);
    wire
}

/// Build a WINDOW_UPDATE frame.
pub fn build_window_update(stream_id: u32, increment: u32) -> Vec<u8> {
    let mut wire = Vec::with_capacity(13);
    FrameHeader {
        length: 4,
        typ: FrameType::WindowUpdate,
        flags: 0,
        stream_id,
    }
    .encode(&mut wire);
    wire.extend_from_slice(&(increment & 0x7FFF_FFFF).to_be_bytes());
    wire
}

/// Build a GOAWAY frame body.
pub fn build_goaway(last_stream_id: u32, error_code: u32, debug: &[u8]) -> Vec<u8> {
    let mut wire = Vec::with_capacity(17 + debug.len());
    FrameHeader {
        length: (8 + debug.len()) as u32,
        typ: FrameType::Goaway,
        flags: 0,
        stream_id: 0,
    }
    .encode(&mut wire);
    wire.extend_from_slice(&(last_stream_id & 0x7FFF_FFFF).to_be_bytes());
    wire.extend_from_slice(&error_code.to_be_bytes());
    wire.extend_from_slice(debug);
    wire
}

// ----------------------------------------------------------------------
// HPACK — RFC 7541 (minimal: static-table indexed + literal-with-
// incremental-indexing using Huffman OFF, no dynamic table mutations
// on the encode side).
// ----------------------------------------------------------------------

/// Static-table entries the client actually uses.
const HPACK_STATIC_TABLE: &[(&str, &str)] = &[
    (":authority", ""),
    (":method", "GET"),
    (":method", "POST"),
    (":path", "/"),
    (":path", "/index.html"),
    (":scheme", "http"),
    (":scheme", "https"),
    (":status", "200"),
    (":status", "204"),
    (":status", "206"),
    (":status", "304"),
    (":status", "400"),
    (":status", "404"),
    (":status", "500"),
    ("accept-charset", ""),
    ("accept-encoding", "gzip, deflate"),
    ("accept-language", ""),
    ("accept-ranges", ""),
    ("accept", ""),
    ("access-control-allow-origin", ""),
    ("age", ""),
    ("allow", ""),
    ("authorization", ""),
    ("cache-control", ""),
    ("content-disposition", ""),
    ("content-encoding", ""),
    ("content-language", ""),
    ("content-length", ""),
    ("content-location", ""),
    ("content-range", ""),
    ("content-type", ""),
    ("cookie", ""),
    ("date", ""),
    ("etag", ""),
    ("expect", ""),
    ("expires", ""),
    ("from", ""),
    ("host", ""),
    ("if-match", ""),
    ("if-modified-since", ""),
    ("if-none-match", ""),
    ("if-range", ""),
    ("if-unmodified-since", ""),
    ("last-modified", ""),
    ("link", ""),
    ("location", ""),
    ("max-forwards", ""),
    ("proxy-authenticate", ""),
    ("proxy-authorization", ""),
    ("range", ""),
    ("referer", ""),
    ("refresh", ""),
    ("retry-after", ""),
    ("server", ""),
    ("set-cookie", ""),
    ("strict-transport-security", ""),
    ("transfer-encoding", ""),
    ("user-agent", ""),
    ("vary", ""),
    ("via", ""),
    ("www-authenticate", ""),
];

/// Encode an HPACK integer per RFC 7541 §5.1, prefix `n_bits`.
fn encode_int(out: &mut Vec<u8>, value: u32, prefix_bits: u8, first_byte: u8) {
    let max_prefix = (1u32 << prefix_bits) - 1;
    if value < max_prefix {
        out.push(first_byte | (value as u8));
        return;
    }
    out.push(first_byte | (max_prefix as u8));
    let mut rem = value - max_prefix;
    while rem >= 128 {
        out.push(((rem & 0x7F) | 0x80) as u8);
        rem >>= 7;
    }
    out.push(rem as u8);
}

/// Encode a literal header field with incremental indexing (RFC 7541
/// §6.2.1) using Huffman encoding when it produces shorter output —
/// which is true for every realistic ASCII header value. Matches
/// Chrome's HPACK byte output shape (Cloudflare bot mode hashes the
/// HEADERS-frame payload bytes; non-Huffman strings stand out).
fn encode_literal_string(out: &mut Vec<u8>, s: &str) {
    let huff = hpack_huffman_encode(s.as_bytes());
    if huff.len() < s.len() {
        encode_int(out, huff.len() as u32, 7, 0x80); // H=1 bit
        out.extend_from_slice(&huff);
    } else {
        encode_int(out, s.len() as u32, 7, 0x00); // H=0 bit
        out.extend_from_slice(s.as_bytes());
    }
}

/// Encode a single header into an HPACK block. Uses a static-table
/// lookup for the name where possible, otherwise emits the name
/// inline. Always uses literal-with-incremental-indexing form (`0x40`).
pub fn hpack_encode_header(out: &mut Vec<u8>, name: &str, value: &str) {
    let lc = name.to_ascii_lowercase();
    // Find static-table index for name (and exact value if present).
    let mut name_index: Option<u8> = None;
    for (i, (n, v)) in HPACK_STATIC_TABLE.iter().enumerate() {
        if *n == lc {
            if *v == value {
                // Fully-indexed header field (§6.1).
                encode_int(out, (i as u32) + 1, 7, 0x80);
                return;
            }
            if name_index.is_none() {
                name_index = Some((i as u8) + 1);
            }
        }
    }
    match name_index {
        Some(idx) => {
            // Literal with incremental indexing, name indexed (§6.2.1).
            encode_int(out, idx as u32, 6, 0x40);
            encode_literal_string(out, value);
        }
        None => {
            // Literal with incremental indexing, name as literal.
            out.push(0x40);
            encode_literal_string(out, &lc);
            encode_literal_string(out, value);
        }
    }
}

/// Encode a complete header block — sequence of (name, value) pairs.
pub fn hpack_encode_block(headers: &[(&str, &str)]) -> Vec<u8> {
    let mut out = Vec::new();
    for (n, v) in headers {
        hpack_encode_header(&mut out, n, v);
    }
    out
}

/// HPACK dynamic table (RFC 7541 §2.3.2). Insertions push at index 0;
/// evictions pop from the high index when `bytes()` exceeds capacity.
#[derive(Debug, Clone)]
pub struct HpackDynamicTable {
    pub max_bytes: usize,
    pub entries: Vec<(String, String)>,
}

impl HpackDynamicTable {
    pub fn new() -> Self {
        Self {
            max_bytes: 4096,
            entries: Vec::new(),
        }
    }
    pub fn bytes(&self) -> usize {
        self.entries
            .iter()
            .map(|(n, v)| n.len() + v.len() + 32)
            .sum()
    }
    pub fn insert(&mut self, name: String, value: String) {
        self.entries.insert(0, (name, value));
        while self.bytes() > self.max_bytes && !self.entries.is_empty() {
            self.entries.pop();
        }
    }
}

impl Default for HpackDynamicTable {
    fn default() -> Self {
        Self::new()
    }
}

/// Decode a single HPACK integer per §5.1 with `prefix_bits` LSBs of
/// the leading byte. Returns `(value, advance)`.
pub fn hpack_decode_int(buf: &[u8], prefix_bits: u8) -> Option<(u32, usize)> {
    if buf.is_empty() {
        return None;
    }
    let mask = (1u32 << prefix_bits) - 1;
    let mut value = (buf[0] as u32) & mask;
    let mut i = 1;
    if value < mask {
        return Some((value, i));
    }
    let mut m = 0u32;
    loop {
        if i >= buf.len() {
            return None;
        }
        let b = buf[i];
        i += 1;
        value += ((b & 0x7F) as u32) << m;
        if b & 0x80 == 0 {
            return Some((value, i));
        }
        m += 7;
        if m > 28 {
            return None;
        }
    }
}

/// RFC 7541 Appendix B Huffman code, encoded as (code, code_len) pairs
/// keyed by symbol value 0..256 (256 = EOS).
const HUFFMAN_CODES: [(u32, u8); 257] = include!("hpack_huffman.in");

/// HPACK Huffman-encode `input` per RFC 7541 §5.2 / Appendix B. Pad
/// the final byte with EOS-prefix bits (all 1s). Used to compress
/// literal-header strings — Chrome always Huffman-encodes; servers
/// fingerprint the resulting byte length against an expected range.
pub fn hpack_huffman_encode(input: &[u8]) -> Vec<u8> {
    // Count total bits.
    let mut total_bits: u64 = 0;
    for &b in input {
        total_bits += HUFFMAN_CODES[b as usize].1 as u64;
    }
    let total_bytes = ((total_bits + 7) / 8) as usize;
    let mut out = vec![0u8; total_bytes];
    let mut bit_pos: u64 = 0;
    for &b in input {
        let (code, len) = HUFFMAN_CODES[b as usize];
        // Write `len` bits of `code` MSB-first at bit_pos.
        for i in 0..len {
            let bit = ((code >> (len - 1 - i)) & 1) as u8;
            let byte_idx = (bit_pos / 8) as usize;
            let shift = 7 - (bit_pos % 8) as u8;
            out[byte_idx] |= bit << shift;
            bit_pos += 1;
        }
    }
    // Pad remainder of last byte with 1s (EOS prefix).
    if bit_pos % 8 != 0 {
        let byte_idx = (bit_pos / 8) as usize;
        let rem = 8 - (bit_pos % 8) as u8;
        let pad: u8 = (1u8 << rem) - 1;
        out[byte_idx] |= pad;
    }
    out
}

/// Decode HPACK Huffman-encoded bytes. Returns None on malformed input.
pub fn hpack_huffman_decode(input: &[u8]) -> Option<Vec<u8>> {
    let mut out = Vec::with_capacity(input.len());
    // Build a bitstream and walk the codes table by length. For each
    // length L from 5..=30, slide a window of L bits and check whether
    // any code matches. This is O(N · L_max / 8) — fast enough for
    // header blocks.
    let mut bit_pos: usize = 0;
    let total_bits = input.len() * 8;
    while bit_pos < total_bits {
        let mut matched = false;
        for code_len in 5..=30u8 {
            if bit_pos + code_len as usize > total_bits {
                break;
            }
            let window = bits_at(input, bit_pos, code_len);
            if let Some(sym) = lookup_huffman(window, code_len) {
                if sym == 256 {
                    return Some(out);
                }
                out.push(sym as u8);
                bit_pos += code_len as usize;
                matched = true;
                break;
            }
        }
        if !matched {
            // Allow up to 7 trailing 1-bits as padding per §5.2.
            let pad = total_bits - bit_pos;
            if pad <= 7 {
                let trailing = bits_at(input, bit_pos, pad as u8);
                let ones = (1u32 << pad) - 1;
                if trailing == ones {
                    return Some(out);
                }
            }
            return None;
        }
    }
    Some(out)
}

fn bits_at(buf: &[u8], pos: usize, len: u8) -> u32 {
    let mut out: u32 = 0;
    for i in 0..len as usize {
        let p = pos + i;
        let byte = buf[p / 8];
        let bit = (byte >> (7 - (p % 8))) & 1;
        out = (out << 1) | bit as u32;
    }
    out
}

fn lookup_huffman(window: u32, len: u8) -> Option<u16> {
    for (sym, &(code, code_len)) in HUFFMAN_CODES.iter().enumerate() {
        if code_len == len && code == window {
            return Some(sym as u16);
        }
    }
    None
}

/// Read an HPACK string literal: H-bit (1) || length (7-bit prefix) ||
/// raw or Huffman-encoded bytes.
fn hpack_decode_string(buf: &[u8]) -> Option<(String, usize)> {
    if buf.is_empty() {
        return None;
    }
    let huff = (buf[0] & 0x80) != 0;
    let (len, adv) = hpack_decode_int(buf, 7)?;
    let len = len as usize;
    let start = adv;
    let end = start.checked_add(len)?;
    if end > buf.len() {
        return None;
    }
    let raw = &buf[start..end];
    let s = if huff {
        let bytes = hpack_huffman_decode(raw)?;
        String::from_utf8(bytes).ok()?
    } else {
        std::str::from_utf8(raw).ok()?.to_string()
    };
    Some((s, end))
}

/// Look up an HPACK index in the combined (static + dynamic) table.
/// Indices 1..=61 are the static table; 62.. are dynamic-table entries
/// in order of insertion-recency (entry 62 = most recent).
fn hpack_lookup(idx: u32, dyn_tbl: &HpackDynamicTable) -> Option<(String, String)> {
    if idx == 0 {
        return None;
    }
    let i = idx as usize;
    if i <= HPACK_STATIC_TABLE.len() {
        let (n, v) = HPACK_STATIC_TABLE[i - 1];
        return Some((n.to_string(), v.to_string()));
    }
    let off = i - HPACK_STATIC_TABLE.len() - 1;
    dyn_tbl.entries.get(off).cloned()
}

/// Decode a single HPACK block (a HEADERS+CONTINUATION concatenation)
/// into a `Vec<(name, value)>`. Honours dynamic-table-size updates and
/// inserts literal-with-incremental-indexing entries into `dyn_tbl`.
/// Returns None on malformed input.
pub fn hpack_decode_block(
    buf: &[u8],
    dyn_tbl: &mut HpackDynamicTable,
) -> Option<Vec<(String, String)>> {
    let mut out: Vec<(String, String)> = Vec::new();
    let mut i = 0;
    while i < buf.len() {
        let b = buf[i];
        if b & 0x80 != 0 {
            // 6.1 — indexed header field. 7-bit prefix.
            let (idx, adv) = hpack_decode_int(&buf[i..], 7)?;
            i += adv;
            let (n, v) = hpack_lookup(idx, dyn_tbl)?;
            out.push((n, v));
        } else if b & 0xc0 == 0x40 {
            // 6.2.1 — literal with incremental indexing. 6-bit prefix.
            let (idx, adv) = hpack_decode_int(&buf[i..], 6)?;
            i += adv;
            let name = if idx == 0 {
                let (s, a) = hpack_decode_string(&buf[i..])?;
                i += a;
                s
            } else {
                hpack_lookup(idx, dyn_tbl)?.0
            };
            let (value, a) = hpack_decode_string(&buf[i..])?;
            i += a;
            dyn_tbl.insert(name.clone(), value.clone());
            out.push((name, value));
        } else if b & 0xe0 == 0x20 {
            // 6.3 — dynamic table size update. 5-bit prefix. No emit.
            let (new_size, adv) = hpack_decode_int(&buf[i..], 5)?;
            i += adv;
            dyn_tbl.max_bytes = new_size as usize;
            while dyn_tbl.bytes() > dyn_tbl.max_bytes && !dyn_tbl.entries.is_empty() {
                dyn_tbl.entries.pop();
            }
        } else {
            // 6.2.2 / 6.2.3 — literal without indexing / never indexed.
            // 4-bit prefix. Same decode shape.
            let (idx, adv) = hpack_decode_int(&buf[i..], 4)?;
            i += adv;
            let name = if idx == 0 {
                let (s, a) = hpack_decode_string(&buf[i..])?;
                i += a;
                s
            } else {
                hpack_lookup(idx, dyn_tbl)?.0
            };
            let (value, a) = hpack_decode_string(&buf[i..])?;
            i += a;
            out.push((name, value));
        }
    }
    Some(out)
}

// ----------------------------------------------------------------------
// Streaming layer
// ----------------------------------------------------------------------

/// State of a single HTTP/2 stream per RFC 9113 §5.1.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamState {
    Idle,
    ReservedLocal,
    ReservedRemote,
    Open,
    HalfClosedLocal,
    HalfClosedRemote,
    Closed,
}

/// Per-stream tracking: state machine, flow-control window, and
/// accumulators for fragmented HEADERS + CONTINUATION + DATA.
#[derive(Debug, Clone)]
pub struct Stream {
    pub id: u32,
    pub state: StreamState,
    /// Server's flow-control window for sending data to us.
    pub recv_window: i32,
    /// Our flow-control window for sending data to the server.
    pub send_window: i32,
    pub headers_buf: Vec<u8>,
    pub data_buf: Vec<u8>,
    /// Set once END_STREAM is observed in either direction.
    pub eof_local: bool,
    pub eof_remote: bool,
}

impl Stream {
    pub fn new(id: u32, initial_window: i32) -> Self {
        Self {
            id,
            state: StreamState::Idle,
            recv_window: initial_window,
            send_window: initial_window,
            headers_buf: Vec::new(),
            data_buf: Vec::new(),
            eof_local: false,
            eof_remote: false,
        }
    }
}

/// HTTP/2 multiplexing connection. Owns the streams map + the global
/// flow-control window. Does NOT own the byte transport — caller
/// drives it by feeding bytes to `feed_frame` and pulling outgoing
/// bytes via `drain_outgoing`.
#[derive(Debug)]
pub struct Connection {
    streams: std::collections::HashMap<u32, Stream>,
    pub next_stream_id: u32,
    /// Settings the *peer* sent us; we honour their MAX_FRAME_SIZE etc.
    pub peer_initial_window: i32,
    pub peer_max_frame_size: u32,
    pub peer_max_concurrent: u32,
    /// Our connection-level flow-control window for incoming bytes.
    pub conn_recv_window: i32,
    /// Server's connection-level window for our outgoing bytes.
    pub conn_send_window: i32,
    /// Bytes queued to send to the peer.
    pub outgoing: Vec<u8>,
    pub last_seen_stream_id: u32,
    pub got_settings_ack: bool,
    pub closed_by_goaway: bool,
}

impl Connection {
    /// Build a fresh client-side connection, queueing the preface +
    /// initial SETTINGS frame in `outgoing`.
    pub fn new_client() -> Self {
        let mut me = Self {
            streams: std::collections::HashMap::new(),
            next_stream_id: 1, // client-initiated odd-numbered
            peer_initial_window: 65_535,
            peer_max_frame_size: 16_384,
            peer_max_concurrent: 100,
            conn_recv_window: 65_535,
            conn_send_window: 65_535,
            outgoing: Vec::new(),
            last_seen_stream_id: 0,
            got_settings_ack: false,
            closed_by_goaway: false,
        };
        me.outgoing.extend_from_slice(CONNECTION_PREFACE);
        // Initial SETTINGS — Chrome 131's exact values and order. The
        // "Akamai h2 fingerprint" (a separate hash from JA4) hashes
        // the (id, value) pairs in order; matching Chrome here is what
        // gets us past Cloudflare bot-manager's h2 check.
        me.outgoing.extend_from_slice(&build_settings_frame(
            &[
                (SettingId::HeaderTableSize as u16, 65_536),
                (SettingId::EnablePush as u16, 0),
                (SettingId::InitialWindowSize as u16, 6_291_456),
                (SettingId::MaxHeaderListSize as u16, 262_144),
            ],
            false,
        ));
        // Chrome bumps the connection-level recv window to ~15 MB
        // right after SETTINGS — also part of the h2 fingerprint.
        me.outgoing
            .extend_from_slice(&build_window_update(0, 15_663_105));
        me
    }

    /// Reserve a fresh stream id for an outgoing request.
    pub fn open_stream(&mut self) -> u32 {
        let id = self.next_stream_id;
        self.next_stream_id += 2; // odd-numbered for client
        let stream = Stream::new(id, self.peer_initial_window);
        self.streams.insert(id, stream);
        id
    }

    /// Look up a stream — caller may want to inspect data / headers.
    pub fn stream(&self, id: u32) -> Option<&Stream> {
        self.streams.get(&id)
    }

    /// Remove a finished stream from the map, returning it. The pooled
    /// h2 driver calls this after building a Response so a long-lived
    /// connection's `streams` map doesn't grow without bound across the
    /// thousands of requests it will serve over its lifetime.
    pub fn remove_stream(&mut self, id: u32) -> Option<Stream> {
        self.streams.remove(&id)
    }

    /// Count streams that are not yet Closed — i.e. occupying one of the
    /// peer's MAX_CONCURRENT_STREAMS slots. The pooled driver gates
    /// `open_stream` on this against `peer_max_concurrent`. In serialized
    /// V1 (one in-flight request per conn) this is trivially satisfied,
    /// but the guard makes the invariant explicit and is what V2's
    /// parallel-in-flight path will lean on.
    pub fn active_stream_count(&self) -> usize {
        self.streams
            .values()
            .filter(|s| s.state != StreamState::Closed)
            .count()
    }

    /// Build + queue a HEADERS frame (END_STREAM=true if no body).
    pub fn send_headers(&mut self, stream_id: u32, headers: &[(&str, &str)], end_stream: bool) {
        let block = hpack_encode_block(headers);
        let flags = 0x4 | if end_stream { 0x1 } else { 0x0 }; // END_HEADERS | END_STREAM
        let mut wire = Vec::with_capacity(9 + block.len());
        FrameHeader {
            length: block.len() as u32,
            typ: FrameType::Headers,
            flags,
            stream_id,
        }
        .encode(&mut wire);
        wire.extend_from_slice(&block);
        self.outgoing.extend_from_slice(&wire);
        if let Some(s) = self.streams.get_mut(&stream_id) {
            s.state = if end_stream {
                StreamState::HalfClosedLocal
            } else {
                StreamState::Open
            };
            s.eof_local = end_stream;
        }
    }

    /// Build + queue a DATA frame. Subtracts from flow-control windows.
    pub fn send_data(&mut self, stream_id: u32, data: &[u8], end_stream: bool) {
        let need = data.len() as i32;
        if need > self.conn_send_window {
            return;
        }
        let flags = if end_stream { 0x1 } else { 0x0 }; // END_STREAM
        let mut wire = Vec::with_capacity(9 + data.len());
        FrameHeader {
            length: data.len() as u32,
            typ: FrameType::Data,
            flags,
            stream_id,
        }
        .encode(&mut wire);
        wire.extend_from_slice(data);
        self.outgoing.extend_from_slice(&wire);
        self.conn_send_window -= need;
        if let Some(s) = self.streams.get_mut(&stream_id) {
            s.send_window -= need;
            if end_stream {
                s.eof_local = true;
                s.state = match s.state {
                    StreamState::Open => StreamState::HalfClosedLocal,
                    StreamState::HalfClosedRemote => StreamState::Closed,
                    other => other,
                };
            }
        }
    }

    /// Feed a fully-decoded frame to the connection. Updates stream
    /// state, accumulates headers/data, queues replies.
    pub fn feed_frame(&mut self, hdr: FrameHeader, body: &[u8]) {
        if hdr.stream_id > self.last_seen_stream_id {
            self.last_seen_stream_id = hdr.stream_id;
        }
        match hdr.typ {
            FrameType::Settings => {
                if hdr.flags & 0x1 != 0 {
                    self.got_settings_ack = true;
                    return;
                }
                for (id, val) in decode_settings(body) {
                    match id {
                        x if x == SettingId::InitialWindowSize as u16 => {
                            self.peer_initial_window = val as i32;
                        }
                        x if x == SettingId::MaxFrameSize as u16 => {
                            self.peer_max_frame_size = val;
                        }
                        x if x == SettingId::MaxConcurrentStreams as u16 => {
                            self.peer_max_concurrent = val;
                        }
                        _ => {}
                    }
                }
                // ACK the SETTINGS.
                self.outgoing
                    .extend_from_slice(&build_settings_frame(&[], true));
            }
            FrameType::Ping => {
                if hdr.flags & 0x1 == 0 && body.len() == 8 {
                    let mut data = [0u8; 8];
                    data.copy_from_slice(&body[..8]);
                    self.outgoing
                        .extend_from_slice(&build_ping_frame(data, true));
                }
            }
            FrameType::WindowUpdate => {
                if body.len() < 4 {
                    return;
                }
                let inc = u32::from_be_bytes([body[0], body[1], body[2], body[3]]) & 0x7FFF_FFFF;
                if hdr.stream_id == 0 {
                    self.conn_send_window = self.conn_send_window.saturating_add(inc as i32);
                } else if let Some(s) = self.streams.get_mut(&hdr.stream_id) {
                    s.send_window = s.send_window.saturating_add(inc as i32);
                }
            }
            FrameType::Headers => {
                let s = self
                    .streams
                    .entry(hdr.stream_id)
                    .or_insert_with(|| Stream::new(hdr.stream_id, self.peer_initial_window));
                // RFC 9113 §6.2: strip PADDED (0x8) and PRIORITY (0x20) prefix
                // fields before feeding the payload to HPACK. These prefixes are
                // NOT part of the header block fragment and corrupt HPACK if left
                // in. Common with CDN servers (Cloudflare, Akamai).
                //
                // Wire layout when both flags set:
                //   [pad_length: 1 byte]
                //   [exclusive(1b) + stream_dep(31b): 4 bytes][weight: 1 byte]
                //   [header block fragment: N bytes]
                //   [padding: pad_length bytes]
                let padded = hdr.flags & 0x8 != 0;
                let priority = hdr.flags & 0x20 != 0;
                let mut payload = body;
                let mut pad_length: usize = 0;
                if padded {
                    if payload.is_empty() {
                        return; // malformed frame — ignore
                    }
                    pad_length = payload[0] as usize;
                    payload = &payload[1..];
                }
                if priority {
                    if payload.len() < 5 {
                        return; // malformed frame — ignore
                    }
                    payload = &payload[5..]; // skip stream_dep(4) + weight(1)
                }
                if padded {
                    if payload.len() < pad_length {
                        return; // malformed frame — ignore
                    }
                    let end = payload.len() - pad_length;
                    payload = &payload[..end];
                }
                s.headers_buf.extend_from_slice(payload);
                if hdr.flags & 0x4 != 0 {
                    // END_HEADERS — full block is in `headers_buf`.
                }
                if hdr.flags & 0x1 != 0 {
                    s.eof_remote = true;
                    s.state = match s.state {
                        StreamState::Open => StreamState::HalfClosedRemote,
                        StreamState::HalfClosedLocal => StreamState::Closed,
                        other => other,
                    };
                }
            }
            FrameType::Continuation => {
                if let Some(s) = self.streams.get_mut(&hdr.stream_id) {
                    s.headers_buf.extend_from_slice(body);
                }
            }
            FrameType::Data => {
                // RFC 9113 §6.1: strip PADDED (0x8) prefix before buffering
                // data bytes. pad_length byte is not part of the data payload.
                let data_payload: &[u8] = if hdr.flags & 0x8 != 0 {
                    if body.is_empty() {
                        &[]
                    } else {
                        let pad_length = body[0] as usize;
                        let rest = &body[1..];
                        if rest.len() < pad_length {
                            &[] // malformed — treat as empty
                        } else {
                            &rest[..rest.len() - pad_length]
                        }
                    }
                } else {
                    body
                };
                self.conn_recv_window -= body.len() as i32;
                if let Some(s) = self.streams.get_mut(&hdr.stream_id) {
                    s.data_buf.extend_from_slice(data_payload);
                    s.recv_window -= body.len() as i32;
                    if hdr.flags & 0x1 != 0 {
                        s.eof_remote = true;
                        s.state = match s.state {
                            StreamState::Open => StreamState::HalfClosedRemote,
                            StreamState::HalfClosedLocal => StreamState::Closed,
                            other => other,
                        };
                    }
                }
                // Replenish flow-control windows aggressively.
                if self.conn_recv_window < 32_768 {
                    let inc = 65_535 - self.conn_recv_window as u32;
                    self.outgoing
                        .extend_from_slice(&build_window_update(0, inc));
                    self.conn_recv_window += inc as i32;
                }
                let stream_inc = if let Some(s) = self.streams.get(&hdr.stream_id) {
                    if s.recv_window < 32_768 {
                        Some(65_535 - s.recv_window as u32)
                    } else {
                        None
                    }
                } else {
                    None
                };
                if let Some(inc) = stream_inc {
                    self.outgoing
                        .extend_from_slice(&build_window_update(hdr.stream_id, inc));
                    if let Some(s) = self.streams.get_mut(&hdr.stream_id) {
                        s.recv_window += inc as i32;
                    }
                }
            }
            FrameType::RstStream => {
                if let Some(s) = self.streams.get_mut(&hdr.stream_id) {
                    s.state = StreamState::Closed;
                }
            }
            FrameType::Goaway => {
                self.closed_by_goaway = true;
            }
            _ => {}
        }
    }

    /// Pull queued bytes for the transport to flush.
    pub fn drain_outgoing(&mut self) -> Vec<u8> {
        std::mem::take(&mut self.outgoing)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn settings_roundtrip() {
        let wire = build_settings_frame(
            &[
                (SettingId::EnablePush as u16, 0),
                (SettingId::MaxConcurrentStreams as u16, 100),
            ],
            false,
        );
        let hdr = FrameHeader::decode(&wire).unwrap();
        assert_eq!(hdr.typ, FrameType::Settings);
        assert_eq!(hdr.length, 12);
        let body = &wire[9..];
        let parsed = decode_settings(body);
        assert_eq!(parsed.len(), 2);
        assert_eq!(parsed[0], (0x2, 0));
        assert_eq!(parsed[1], (0x3, 100));
    }

    #[test]
    fn window_update_frames() {
        let wire = build_window_update(0, 65535);
        let hdr = FrameHeader::decode(&wire).unwrap();
        assert_eq!(hdr.typ, FrameType::WindowUpdate);
        assert_eq!(hdr.length, 4);
        let inc = u32::from_be_bytes([wire[9], wire[10], wire[11], wire[12]]);
        assert_eq!(inc, 65535);
    }

    #[test]
    fn hpack_encodes_indexed_pair() {
        // ":method", "GET" → fully indexed (entry 2 → value 0x82).
        let mut out = Vec::new();
        hpack_encode_header(&mut out, ":method", "GET");
        assert_eq!(out, vec![0x82]);
    }

    #[test]
    fn hpack_encodes_literal_with_indexed_name() {
        // ":authority", "example.com" → name is indexed (1), value
        // literal. Encoder now Huffman-encodes when shorter; the
        // result should still roundtrip through the decoder.
        let mut out = Vec::new();
        hpack_encode_header(&mut out, ":authority", "example.com");
        // 0x40 | 1 = 0x41 (name idx 1, lit w/ incremental indexing).
        assert_eq!(out[0], 0x41);
        // Decode back and confirm "example.com" round-trips.
        let mut dyn_tbl = HpackDynamicTable::new();
        let decoded = hpack_decode_block(&out, &mut dyn_tbl).unwrap();
        assert_eq!(
            decoded,
            vec![(":authority".to_string(), "example.com".to_string())]
        );
    }

    #[test]
    fn client_connection_queues_preface_and_settings() {
        let c = Connection::new_client();
        // Preface present.
        assert!(c.outgoing.starts_with(CONNECTION_PREFACE));
        // First frame after preface is SETTINGS (type 0x4).
        let after_preface = &c.outgoing[CONNECTION_PREFACE.len()..];
        let hdr = FrameHeader::decode(after_preface).unwrap();
        assert_eq!(hdr.typ, FrameType::Settings);
        assert_eq!(hdr.stream_id, 0);
    }

    #[test]
    fn ping_ack_is_emitted_on_ping() {
        let mut c = Connection::new_client();
        c.drain_outgoing();
        let ping_body = [1u8, 2, 3, 4, 5, 6, 7, 8];
        c.feed_frame(
            FrameHeader {
                length: 8,
                typ: FrameType::Ping,
                flags: 0,
                stream_id: 0,
            },
            &ping_body,
        );
        let out = c.drain_outgoing();
        let hdr = FrameHeader::decode(&out).unwrap();
        assert_eq!(hdr.typ, FrameType::Ping);
        assert_eq!(hdr.flags & 0x1, 0x1); // ACK
        assert_eq!(&out[9..17], &ping_body);
    }

    #[test]
    fn settings_ack_round_trip() {
        let mut c = Connection::new_client();
        c.drain_outgoing();
        // Peer sends SETTINGS with InitialWindowSize=128k.
        let mut body = Vec::new();
        encode_settings(&mut body, &[(SettingId::InitialWindowSize as u16, 131_072)]);
        c.feed_frame(
            FrameHeader {
                length: body.len() as u32,
                typ: FrameType::Settings,
                flags: 0,
                stream_id: 0,
            },
            &body,
        );
        assert_eq!(c.peer_initial_window, 131_072);
        // We should have queued an ACK.
        let out = c.drain_outgoing();
        let hdr = FrameHeader::decode(&out).unwrap();
        assert_eq!(hdr.typ, FrameType::Settings);
        assert_eq!(hdr.flags & 0x1, 0x1);
        assert_eq!(hdr.length, 0);
    }

    #[test]
    fn stream_open_and_headers_flow() {
        let mut c = Connection::new_client();
        c.drain_outgoing();
        let sid = c.open_stream();
        c.send_headers(
            sid,
            &[(":method", "GET"), (":path", "/"), (":scheme", "https")],
            true,
        );
        let s = c.stream(sid).unwrap();
        assert_eq!(s.state, StreamState::HalfClosedLocal);
        assert!(s.eof_local);
        // Check the wire — HEADERS frame.
        let out = c.drain_outgoing();
        let hdr = FrameHeader::decode(&out).unwrap();
        assert_eq!(hdr.typ, FrameType::Headers);
        assert_eq!(hdr.flags & 0x4, 0x4); // END_HEADERS
        assert_eq!(hdr.flags & 0x1, 0x1); // END_STREAM
    }

    #[test]
    fn inbound_data_replenishes_windows() {
        let mut c = Connection::new_client();
        c.drain_outgoing();
        let _sid = c.open_stream();
        let big = vec![0xCDu8; 40_000];
        c.feed_frame(
            FrameHeader {
                length: big.len() as u32,
                typ: FrameType::Data,
                flags: 0,
                stream_id: 1,
            },
            &big,
        );
        // 40k > 32k threshold → WINDOW_UPDATE on conn (id=0) should
        // have been queued.
        let out = c.drain_outgoing();
        let hdr = FrameHeader::decode(&out).unwrap();
        assert_eq!(hdr.typ, FrameType::WindowUpdate);
    }

    #[test]
    fn frame_header_roundtrip() {
        let h = FrameHeader {
            length: 12345,
            typ: FrameType::Headers,
            flags: 0x4,
            stream_id: 3,
        };
        let mut wire = Vec::new();
        h.encode(&mut wire);
        let decoded = FrameHeader::decode(&wire).unwrap();
        assert_eq!(decoded, h);
    }

    // ==================================================================
    // M6.3 TIER-1 ORACLE — frame demux correctness (no socket, no TLS).
    //
    // Drives the `Connection` state machine directly with a HAND-BUILT,
    // deliberately INTERLEAVED server byte stream and asserts each stream
    // assembles ITS OWN correct body — proving incoming frames are keyed
    // by stream_id and a request for `sid` can NEVER receive another
    // stream's bytes.
    // ==================================================================

    /// Craft a server-side frame (header + body) on the wire.
    fn srv_frame(typ: FrameType, flags: u8, stream_id: u32, body: &[u8]) -> Vec<u8> {
        let mut wire = Vec::with_capacity(9 + body.len());
        FrameHeader {
            length: body.len() as u32,
            typ,
            flags,
            stream_id,
        }
        .encode(&mut wire);
        wire.extend_from_slice(body);
        wire
    }

    /// Feed a whole server byte stream to `c` using the SAME
    /// accum-reassembly logic the production driver uses, but fed in
    /// arbitrary `chunk` sizes so a frame straddling chunk boundaries is
    /// exercised (proves accum reassembly).
    fn feed_stream_chunked(c: &mut Connection, wire: &[u8], chunk: usize) {
        let mut accum: Vec<u8> = Vec::new();
        let mut pos = 0;
        while pos < wire.len() {
            let end = (pos + chunk).min(wire.len());
            accum.extend_from_slice(&wire[pos..end]);
            pos = end;
            let mut consumed = 0usize;
            while accum.len() - consumed >= 9 {
                let hdr = FrameHeader::decode(&accum[consumed..]).unwrap();
                let frame_len = 9 + hdr.length as usize;
                if accum.len() - consumed < frame_len {
                    break;
                }
                let fbody = accum[consumed + 9..consumed + frame_len].to_vec();
                c.feed_frame(hdr, &fbody);
                consumed += frame_len;
            }
            accum.drain(..consumed);
        }
    }

    #[test]
    fn m63_oracle_interleaved_streams_get_own_bodies() {
        // Two requests share one conn: sid1=1, sid3=3.
        let mut c = Connection::new_client();
        c.drain_outgoing();
        let sid1 = c.open_stream();
        let sid3 = c.open_stream();
        assert_eq!((sid1, sid3), (1, 3));

        // Server byte stream, INTERLEAVED on purpose — stream-3 completes
        // FIRST, before stream-1's END_STREAM.
        let h1 = hpack_encode_block(&[
            (":status", "200"),
            ("content-type", "text/plain"),
        ]);
        let h3 = hpack_encode_block(&[(":status", "404")]);
        let mut wire = Vec::new();
        // server SETTINGS
        wire.extend_from_slice(&build_settings_frame(&[], false));
        // HEADERS(1) END_HEADERS (no END_STREAM — body follows)
        wire.extend_from_slice(&srv_frame(FrameType::Headers, 0x4, 1, &h1));
        // HEADERS(3) END_HEADERS (no END_STREAM — body follows)
        wire.extend_from_slice(&srv_frame(FrameType::Headers, 0x4, 3, &h3));
        // DATA(3) END_STREAM — stream-3 done FIRST
        wire.extend_from_slice(&srv_frame(FrameType::Data, 0x1, 3, b"BODY-THREE"));
        // DATA(1) part A (no END)
        wire.extend_from_slice(&srv_frame(FrameType::Data, 0x0, 1, b"BODY-ONE-PART-A"));
        // DATA(1) part B END_STREAM
        wire.extend_from_slice(&srv_frame(FrameType::Data, 0x1, 1, b"BODY-ONE-PART-B"));

        // Feed in TINY 7-byte chunks so frames straddle boundaries.
        feed_stream_chunked(&mut c, &wire, 7);

        // ASSERT: exact bodies to exact streams despite stream-3 first.
        let s1 = c.stream(1).unwrap();
        let s3 = c.stream(3).unwrap();
        assert_eq!(s1.data_buf, b"BODY-ONE-PART-ABODY-ONE-PART-B");
        assert!(s1.eof_remote);
        assert_eq!(s3.data_buf, b"BODY-THREE");
        assert!(s3.eof_remote);

        // Headers decode to correct status per stream (persistent table).
        let mut dyn_tbl = HpackDynamicTable::new();
        let d1 = hpack_decode_block(&s1.headers_buf, &mut dyn_tbl).unwrap();
        assert!(d1.iter().any(|(n, v)| n == ":status" && v == "200"));
        let d3 = hpack_decode_block(&s3.headers_buf, &mut dyn_tbl).unwrap();
        assert!(d3.iter().any(|(n, v)| n == ":status" && v == "404"));

        // NON-VACUOUS GUARD (mutation proof): the two bodies differ and
        // are NOT swapped — a demux that routed DATA(3) into stream 1
        // would fail these.
        assert_ne!(s1.data_buf, s3.data_buf);
        assert_ne!(s1.data_buf, b"BODY-THREE");
        assert_ne!(s3.data_buf, b"BODY-ONE-PART-ABODY-ONE-PART-B");
    }

    #[test]
    fn m63_oracle_persistent_hpack_table_required() {
        // The server inserts a dynamic entry while sending stream-1's
        // headers (literal-with-incremental-indexing for "x-trace") then
        // references it BY DYNAMIC INDEX in stream-3's headers. Decoding
        // with the PERSISTENT table resolves it; a FRESH table per stream
        // cannot — proving the per-conn decode table is a real
        // requirement, not a nicety.
        let mut persistent = HpackDynamicTable::new();

        // stream-1 header block: :status 200 (indexed) + x-trace: abc123
        // as literal-with-incremental-indexing (0x40, name literal). This
        // INSERTS ("x-trace","abc123") at dynamic index 62.
        let mut h1 = Vec::new();
        hpack_encode_header(&mut h1, ":status", "200"); // indexed static
        // literal w/ incremental indexing, literal name + value:
        h1.push(0x40);
        encode_literal_string_test(&mut h1, "x-trace");
        encode_literal_string_test(&mut h1, "abc123");

        // Decode stream-1 with persistent table → inserts dyn entry 62.
        let d1 = hpack_decode_block(&h1, &mut persistent).unwrap();
        assert!(d1.iter().any(|(n, v)| n == "x-trace" && v == "abc123"));
        assert_eq!(persistent.entries.len(), 1);

        // stream-3 header block references dynamic index 62 (the static
        // table is 61 entries, so 62 = most-recent dynamic entry).
        let mut h3 = Vec::new();
        hpack_encode_header(&mut h3, ":status", "200");
        encode_int_test(&mut h3, 62, 7, 0x80); // indexed header field, idx 62

        // With the PERSISTENT table, index 62 resolves to x-trace:abc123.
        let mut persistent_for_3 = persistent.clone();
        let d3_persistent = hpack_decode_block(&h3, &mut persistent_for_3).unwrap();
        assert!(
            d3_persistent.iter().any(|(n, v)| n == "x-trace" && v == "abc123"),
            "persistent table must resolve the dynamic index"
        );

        // With a FRESH table, index 62 has nothing → decode FAILS (None).
        let mut fresh = HpackDynamicTable::new();
        let d3_fresh = hpack_decode_block(&h3, &mut fresh);
        assert!(
            d3_fresh.is_none(),
            "a throwaway per-stream table MUST fail to resolve the dynamic index"
        );
    }

    // Local re-exports of the private encode helpers so the oracle can
    // craft server-side blocks the same way the encoder does.
    fn encode_int_test(out: &mut Vec<u8>, value: u32, prefix_bits: u8, first_byte: u8) {
        encode_int(out, value, prefix_bits, first_byte);
    }
    fn encode_literal_string_test(out: &mut Vec<u8>, s: &str) {
        encode_literal_string(out, s);
    }

    #[test]
    fn m63_oracle_rst_stream_marks_closed_without_eof() {
        // An RST_STREAM closes the stream WITHOUT a clean END_STREAM. The
        // driver must distinguish this (Closed && !eof_remote) from a
        // clean completion so it never returns a truncated/empty 200.
        let mut c = Connection::new_client();
        c.drain_outgoing();
        let sid = c.open_stream();
        let hdr_block = hpack_encode_block(&[(":status", "200")]);
        let mut wire = Vec::new();
        wire.extend_from_slice(&srv_frame(FrameType::Headers, 0x4, sid, &hdr_block));
        // RST_STREAM (error code in body, ignored by feed_frame).
        wire.extend_from_slice(&srv_frame(FrameType::RstStream, 0x0, sid, &[0, 0, 0, 8]));
        feed_stream_chunked(&mut c, &wire, 5);
        let s = c.stream(sid).unwrap();
        assert_eq!(s.state, StreamState::Closed);
        assert!(!s.eof_remote, "RST must NOT set eof_remote");
    }

    #[test]
    fn m63_active_stream_count_and_remove() {
        let mut c = Connection::new_client();
        c.drain_outgoing();
        let a = c.open_stream();
        let b = c.open_stream();
        assert_eq!(c.active_stream_count(), 2);
        // Complete `a` cleanly via END_STREAM DATA.
        c.feed_frame(
            FrameHeader { length: 3, typ: FrameType::Data, flags: 0x1, stream_id: a },
            b"hey",
        );
        // `a` is now Closed (was HalfClosedLocal after send? we never
        // sent, so Idle→… ) — regardless, removing bounds memory.
        assert!(c.remove_stream(a).is_some());
        assert!(c.stream(a).is_none());
        assert!(c.stream(b).is_some());
    }
}
