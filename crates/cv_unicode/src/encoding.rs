//! Character-encoding detection + legacy decoders, per the WHATWG Encoding
//! Standard (<https://encoding.spec.whatwg.org/>) and the HTML "encoding
//! sniffing algorithm" (HTML §13.2.3.x,
//! <https://html.spec.whatwg.org/multipage/parsing.html#encoding-sniffing-algorithm>).
//!
//! This is what Chrome/Blink does in `third_party/blink/.../text/text_codec*`:
//! a page's bytes are NOT blindly `from_utf8`'d. The encoding is determined by
//! (in priority order):
//!   1. a leading BOM (UTF-8 `EF BB BF`, UTF-16BE `FE FF`, UTF-16LE `FF FE`) —
//!      authoritative, overrides everything else;
//!   2. an explicit HTTP `Content-Type; charset=…`;
//!   3. a prescan of the first 1024 bytes for `<meta charset>` /
//!      `<meta http-equiv=content-type content="…charset=…">`;
//!   4. otherwise the locale default, which for the Latin-script web is
//!      **windows-1252**, NOT UTF-8 — unlabeled legacy pages rely on this.
//!
//! Labels are mapped through the Encoding Standard's "get an encoding" alias
//! table (§4.2). The decoders are the spec's per-encoding decoder handlers
//! (§9–14) driven off the published index tables (see `encoding_tables.rs`).
//!
//! UTF-8 content decodes byte-for-byte identically to `from_utf8_lossy`, so
//! switching the call sites over is a no-op for UTF-8 pages; only labeled or
//! high-byte legacy content changes (and changes for the better — real glyphs
//! instead of `U+FFFD` mojibake).

use crate::encoding_tables as tbl;

/// The Unicode REPLACEMENT CHARACTER, emitted by every decoder for an
/// undecodable byte sequence (Encoding Standard §4.3 "error mode": replacement).
const REPLACEMENT: char = '\u{FFFD}';

/// The set of encodings this engine can determine + decode. Mirrors the
/// Encoding Standard's enumerated encodings; the names are the spec's
/// canonical (output) encoding names.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Encoding {
    Utf8,
    Utf16Le,
    Utf16Be,
    Windows1252,
    Iso8859_2,
    Iso8859_15,
    ShiftJis,
    EucJp,
    Gbk,
    Gb18030,
    EucKr,
    Big5,
}

impl Encoding {
    /// The spec's canonical name (the "name" column of the encodings table).
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Encoding::Utf8 => "UTF-8",
            Encoding::Utf16Le => "UTF-16LE",
            Encoding::Utf16Be => "UTF-16BE",
            Encoding::Windows1252 => "windows-1252",
            Encoding::Iso8859_2 => "ISO-8859-2",
            Encoding::Iso8859_15 => "ISO-8859-15",
            Encoding::ShiftJis => "Shift_JIS",
            Encoding::EucJp => "EUC-JP",
            Encoding::Gbk => "GBK",
            Encoding::Gb18030 => "gb18030",
            Encoding::EucKr => "EUC-KR",
            Encoding::Big5 => "Big5",
        }
    }
}

/// "Get an encoding" (Encoding Standard §4.2): map a label (a charset string
/// from an HTTP header or a `<meta>` tag) to an encoding. The label is first
/// trimmed of leading/trailing ASCII whitespace and lowercased, then matched
/// against the alias table. Returns `None` for an unknown label (caller then
/// falls back to the next step of the sniffing algorithm).
#[must_use]
pub fn encoding_for_label(label: &str) -> Option<Encoding> {
    // §4.2 step 1: remove leading/trailing ASCII whitespace
    // (0x09 TAB, 0x0A LF, 0x0C FF, 0x0D CR, 0x20 SPACE).
    let trimmed = label.trim_matches(|c| matches!(c, '\t' | '\n' | '\x0C' | '\r' | ' '));
    // §4.2 step 2: ASCII-lowercase, then compare.
    let lc = trimmed.to_ascii_lowercase();
    Some(match lc.as_str() {
        // --- UTF-8 ---
        "unicode-1-1-utf-8" | "unicode11utf8" | "unicode20utf8" | "utf-8" | "utf8"
        | "x-unicode20utf8" => Encoding::Utf8,

        // --- windows-1252 (the web's "latin1"/"ascii") ---
        "ansi_x3.4-1968" | "ascii" | "cp1252" | "cp819" | "csisolatin1" | "ibm819"
        | "iso-8859-1" | "iso-ir-100" | "iso8859-1" | "iso88591" | "iso_8859-1"
        | "iso_8859-1:1987" | "l1" | "latin1" | "us-ascii" | "windows-1252" | "x-cp1252" => {
            Encoding::Windows1252
        }

        // --- ISO-8859-2 ---
        "csisolatin2" | "iso-8859-2" | "iso-ir-101" | "iso8859-2" | "iso88592" | "iso_8859-2"
        | "iso_8859-2:1987" | "l2" | "latin2" => Encoding::Iso8859_2,

        // --- ISO-8859-15 ---
        "csisolatin9" | "iso-8859-15" | "iso8859-15" | "iso885915" | "iso_8859-15" | "l9" => {
            Encoding::Iso8859_15
        }

        // --- UTF-16 ---
        "csunicode" | "iso-10646-ucs-2" | "ucs-2" | "unicode" | "unicodefeff" | "utf-16"
        | "utf-16le" => Encoding::Utf16Le,
        "unicodefffe" | "utf-16be" => Encoding::Utf16Be,

        // --- Shift_JIS ---
        "csshiftjis" | "ms932" | "ms_kanji" | "shift-jis" | "shift_jis" | "sjis"
        | "windows-31j" | "x-sjis" => Encoding::ShiftJis,

        // --- EUC-JP ---
        "cseucpkdfmtjapanese" | "euc-jp" | "x-euc-jp" => Encoding::EucJp,

        // --- GBK (decodes via the gb18030 index) ---
        "chinese" | "csgb2312" | "csiso58gb231280" | "gb2312" | "gb_2312" | "gb_2312-80"
        | "gbk" | "iso-ir-58" | "x-gbk" => Encoding::Gbk,

        // --- gb18030 ---
        "gb18030" => Encoding::Gb18030,

        // --- EUC-KR (a.k.a. windows-949) ---
        "cseuckr" | "csksc56011987" | "euc-kr" | "iso-ir-149" | "korean" | "ks_c_5601-1987"
        | "ks_c_5601-1989" | "ksc5601" | "ksc_5601" | "windows-949" => Encoding::EucKr,

        // --- Big5 ---
        "big5" | "big5-hkscs" | "cn-big5" | "csbig5" | "x-x-big5" => Encoding::Big5,

        _ => return None,
    })
}

/// Extract a `charset` value from a `Content-Type` header field value, e.g.
/// `text/html; charset=Shift_JIS`. Per the Encoding Standard's "get an
/// encoding from a Content-Type" / MIME-type parsing rules and HTML
/// §13.2.3.4, we look for the `charset` parameter (case-insensitive),
/// honoring an optionally-quoted value. Returns the raw label (caller maps it
/// through `encoding_for_label`).
#[must_use]
pub fn charset_from_content_type(content_type: &str) -> Option<String> {
    let lower = content_type.to_ascii_lowercase();
    let mut search_from = 0usize;
    while let Some(rel) = lower[search_from..].find("charset") {
        let idx = search_from + rel;
        let after = idx + "charset".len();
        // skip ASCII whitespace
        let rest = &content_type[after..];
        let rest_trimmed = rest.trim_start_matches([' ', '\t']);
        if let Some(eq_rest) = rest_trimmed.strip_prefix('=') {
            let val = eq_rest.trim_start_matches([' ', '\t']);
            let val = val.trim_start();
            if let Some(q) = val.strip_prefix('"') {
                // quoted-string: up to the next unescaped quote
                if let Some(end) = q.find('"') {
                    return Some(q[..end].to_string());
                }
                return Some(q.to_string());
            }
            // token: up to the next ';' or whitespace
            let end = val
                .find(|c: char| c == ';' || c.is_ascii_whitespace())
                .unwrap_or(val.len());
            let token = &val[..end];
            if !token.is_empty() {
                return Some(token.to_string());
            }
        }
        search_from = after;
    }
    None
}

/// The full HTML "encoding sniffing algorithm" decision (HTML §13.2.3.x +
/// the Encoding Standard "BOM sniff"). Given the response bytes and the
/// optional HTTP `Content-Type` header value, decide which encoding to use.
///
/// Priority (Chrome/Blink order):
///   1. BOM (authoritative);
///   2. HTTP `charset`;
///   3. `<meta>` prescan of the first 1024 bytes;
///   4. default windows-1252.
#[must_use]
pub fn detect_encoding(bytes: &[u8], http_content_type: Option<&str>) -> Encoding {
    // (1) BOM sniff — authoritative. Encoding Standard "BOM sniff".
    if let Some(enc) = sniff_bom(bytes) {
        return enc;
    }
    // (2) explicit HTTP charset.
    if let Some(ct) = http_content_type {
        if let Some(label) = charset_from_content_type(ct) {
            if let Some(enc) = encoding_for_label(&label) {
                // Per HTML §13.2.3.3, a UTF-16 label from the transport layer
                // is treated as UTF-8 ... actually the spec says if the
                // returned encoding is UTF-16BE/LE, use UTF-8 instead when it
                // came from a meta; for HTTP charset we honor it. Blink honors
                // an explicit HTTP UTF-16 charset, so we do too.
                return enc;
            }
        }
    }
    // (3) prescan first 1024 bytes for a <meta charset>.
    if let Some(enc) = prescan_meta(bytes) {
        return enc;
    }
    // (4) default: windows-1252 (Chrome's effective fallback for the Latin web;
    // unlabeled legacy pages with high bytes rely on this, NOT UTF-8).
    Encoding::Windows1252
}

/// Encoding determination for a *non-HTML* resource whose fallback is UTF-8
/// rather than the HTML default of windows-1252 — i.e. CSS (CSS Syntax §3.2:
/// BOM → HTTP charset → @charset → environment → UTF-8), classic scripts
/// (HTML "fetch a classic script"), and XHR/fetch text (default UTF-8). This
/// does BOM → HTTP `charset` → UTF-8; it deliberately does NOT run the HTML
/// `<meta>` prescan (that is HTML-document-only) and never falls back to
/// windows-1252.
#[must_use]
pub fn detect_encoding_default_utf8(bytes: &[u8], http_content_type: Option<&str>) -> Encoding {
    if let Some(enc) = sniff_bom(bytes) {
        return enc;
    }
    if let Some(ct) = http_content_type {
        if let Some(label) = charset_from_content_type(ct) {
            if let Some(enc) = encoding_for_label(&label) {
                return enc;
            }
        }
    }
    Encoding::Utf8
}

/// Decode a non-HTML text resource (CSS/script/XHR) with a UTF-8 fallback.
/// Honors a BOM and an explicit HTTP `charset`; otherwise UTF-8. For UTF-8
/// content this is byte-identical to `from_utf8_lossy`.
#[must_use]
pub fn decode_text_default_utf8(bytes: &[u8], http_content_type: Option<&str>) -> String {
    let enc = detect_encoding_default_utf8(bytes, http_content_type);
    decode(bytes, enc)
}

/// Encoding Standard "BOM sniff": the first 2–3 bytes. A BOM overrides any
/// other declaration. Returns the encoding and (implicitly) the BOM is
/// stripped by `decode` before decoding the rest.
#[must_use]
pub fn sniff_bom(bytes: &[u8]) -> Option<Encoding> {
    if bytes.len() >= 3 && bytes[0] == 0xEF && bytes[1] == 0xBB && bytes[2] == 0xBF {
        return Some(Encoding::Utf8);
    }
    if bytes.len() >= 2 && bytes[0] == 0xFE && bytes[1] == 0xFF {
        return Some(Encoding::Utf16Be);
    }
    if bytes.len() >= 2 && bytes[0] == 0xFF && bytes[1] == 0xFE {
        return Some(Encoding::Utf16Le);
    }
    None
}

/// HTML §13.2.3.4 "prescan a byte stream to determine its encoding": scan the
/// first 1024 bytes for `<meta charset=…>` or
/// `<meta http-equiv=content-type content="…charset=…">`. This is a tag-soup
/// scan (it deliberately does NOT fully parse), matching the spec's state
/// machine closely enough to find a declared charset before the real parse.
#[must_use]
pub fn prescan_meta(bytes: &[u8]) -> Option<Encoding> {
    let limit = bytes.len().min(1024);
    let data = &bytes[..limit];
    let mut i = 0usize;
    while i < data.len() {
        // Look for "<meta" followed by a space/slash (case-insensitive).
        if data[i] == b'<' {
            // Comment: <!-- ... -->
            if data[i..].starts_with(b"<!--") {
                if let Some(rel) = find_subslice(&data[i + 4..], b"-->") {
                    i += 4 + rel + 3;
                    continue;
                }
                return None; // unterminated comment in first 1024 bytes
            }
            // <meta
            if matches_ascii_ci(&data[i + 1..], b"meta")
                && data
                    .get(i + 5)
                    .is_some_and(|&b| b == b' ' || b == b'\t' || b == b'\n' || b == b'\r' || b == 0x0C || b == b'/')
            {
                // Parse the tag's attributes until '>'.
                let tag_end = find_byte(&data[i..], b'>').map_or(data.len(), |e| i + e);
                let attrs = &data[i + 5..tag_end];
                if let Some(enc) = parse_meta_attrs(attrs) {
                    return Some(enc);
                }
                i = tag_end;
                continue;
            }
            // Any other tag: skip to '>'.
            if let Some(e) = find_byte(&data[i..], b'>') {
                i += e + 1;
                continue;
            }
            return None;
        }
        i += 1;
    }
    None
}

/// Given the bytes of a `<meta …>` tag's attribute region, extract a charset
/// from either `charset=X` or `http-equiv=content-type` + `content=…charset=X`.
fn parse_meta_attrs(attrs: &[u8]) -> Option<Encoding> {
    let s = String::from_utf8_lossy(attrs);
    let lower = s.to_ascii_lowercase();

    // Form 1: <meta charset="X">
    if let Some(pos) = lower.find("charset") {
        // Make sure this is the `charset` attribute, not the substring inside
        // `content="...charset=..."` — handle both: try the attribute form
        // first (charset immediately followed by '=').
        let after = pos + "charset".len();
        let rest = &s[after..];
        let rest_t = rest.trim_start_matches([' ', '\t', '\n', '\r', '\x0C']);
        if let Some(v) = rest_t.strip_prefix('=') {
            // But only if this `charset` is a standalone attribute, i.e. what
            // precedes it is whitespace, tag start, or quote — otherwise it's
            // the `charset=` inside a content="..." value, handled below.
            let preceding_is_attr_boundary = pos == 0
                || matches!(
                    s.as_bytes()[pos - 1],
                    b' ' | b'\t' | b'\n' | b'\r' | 0x0C | b'\'' | b'"' | b';' | b'/'
                );
            if preceding_is_attr_boundary {
                if let Some(label) = read_attr_value(v) {
                    if let Some(enc) = encoding_for_label(&label) {
                        return Some(meta_normalize(enc));
                    }
                }
            }
        }
    }

    // Form 2: <meta http-equiv="content-type" content="text/html; charset=X">
    if lower.contains("http-equiv") && lower.contains("content-type") {
        if let Some(cpos) = lower.find("content") {
            // find content=value after that
            if let Some(eq) = lower[cpos..].find('=') {
                let vstart = cpos + eq + 1;
                if let Some(value) = read_attr_value(&s[vstart..]) {
                    if let Some(label) = charset_from_content_type(&value) {
                        if let Some(enc) = encoding_for_label(&label) {
                            return Some(meta_normalize(enc));
                        }
                    }
                }
            }
        }
    }

    None
}

/// HTML §13.2.3.3: a `<meta>`-declared UTF-16 is treated as UTF-8, and a
/// declared x-user-defined as windows-1252. We don't implement x-user-defined;
/// the UTF-16→UTF-8 normalization is the load-bearing one (a misdeclared
/// UTF-16 meta in a UTF-8 doc must not flip us to UTF-16).
fn meta_normalize(enc: Encoding) -> Encoding {
    match enc {
        Encoding::Utf16Le | Encoding::Utf16Be => Encoding::Utf8,
        other => other,
    }
}

/// Read an HTML attribute value starting at `s` (which begins just after the
/// `=`). Honors single/double quotes; otherwise reads an unquoted token.
fn read_attr_value(s: &str) -> Option<String> {
    let s = s.trim_start_matches([' ', '\t', '\n', '\r', '\x0C']);
    let bytes = s.as_bytes();
    if bytes.is_empty() {
        return None;
    }
    match bytes[0] {
        b'"' => {
            let rest = &s[1..];
            let end = rest.find('"').unwrap_or(rest.len());
            Some(rest[..end].to_string())
        }
        b'\'' => {
            let rest = &s[1..];
            let end = rest.find('\'').unwrap_or(rest.len());
            Some(rest[..end].to_string())
        }
        _ => {
            let end = s
                .find(|c: char| c.is_ascii_whitespace() || c == ';' || c == '>')
                .unwrap_or(s.len());
            if end == 0 {
                None
            } else {
                Some(s[..end].to_string())
            }
        }
    }
}

fn matches_ascii_ci(haystack: &[u8], needle: &[u8]) -> bool {
    haystack.len() >= needle.len()
        && haystack[..needle.len()]
            .iter()
            .zip(needle)
            .all(|(a, b)| a.eq_ignore_ascii_case(b))
}

fn find_byte(haystack: &[u8], b: u8) -> Option<usize> {
    haystack.iter().position(|&x| x == b)
}

fn find_subslice(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() || haystack.len() < needle.len() {
        return None;
    }
    haystack
        .windows(needle.len())
        .position(|w| w == needle)
}

// ===========================================================================
// Top-level decode
// ===========================================================================

/// Decode `bytes` using `enc`, stripping a leading BOM that matches the
/// encoding (Encoding Standard "decode" algorithm step 1–2). Always succeeds:
/// undecodable sequences become `U+FFFD` (the spec's replacement error mode,
/// which is what the HTML parser uses).
#[must_use]
pub fn decode(bytes: &[u8], enc: Encoding) -> String {
    // "decode" §4.3: if the byte stream starts with a BOM for the chosen
    // encoding (or a BOM at all that overrides), strip it. To keep behavior
    // aligned with the determination step, we strip any BOM that matches the
    // active family.
    let body = strip_bom(bytes, enc);
    match enc {
        Encoding::Utf8 => decode_utf8(body),
        Encoding::Utf16Le => decode_utf16(body, false),
        Encoding::Utf16Be => decode_utf16(body, true),
        Encoding::Windows1252 => decode_single_byte(body, &tbl::WINDOWS_1252),
        Encoding::Iso8859_2 => decode_single_byte(body, &tbl::ISO_8859_2),
        Encoding::Iso8859_15 => decode_single_byte(body, &tbl::ISO_8859_15),
        Encoding::ShiftJis => decode_shift_jis(body),
        Encoding::EucJp => decode_euc_jp(body),
        Encoding::Gbk | Encoding::Gb18030 => decode_gb18030(body),
        Encoding::EucKr => decode_euc_kr(body),
        Encoding::Big5 => decode_big5(body),
    }
}

/// Convenience: detect the encoding then decode. This is the call-site
/// replacement for `String::from_utf8_lossy(&body).into_owned()`.
#[must_use]
pub fn decode_with_detection(bytes: &[u8], http_content_type: Option<&str>) -> String {
    let enc = detect_encoding(bytes, http_content_type);
    decode(bytes, enc)
}

fn strip_bom(bytes: &[u8], enc: Encoding) -> &[u8] {
    match enc {
        Encoding::Utf8 if bytes.starts_with(&[0xEF, 0xBB, 0xBF]) => &bytes[3..],
        Encoding::Utf16Be if bytes.starts_with(&[0xFE, 0xFF]) => &bytes[2..],
        Encoding::Utf16Le if bytes.starts_with(&[0xFF, 0xFE]) => &bytes[2..],
        _ => bytes,
    }
}

// ===========================================================================
// UTF-8 (Encoding Standard §10.1) — byte-identical to from_utf8_lossy
// ===========================================================================

fn decode_utf8(bytes: &[u8]) -> String {
    // std's lossy decoder already implements the spec's UTF-8 decoder with
    // the replacement error mode, including the "maximal subpart" recovery
    // (one U+FFFD per maximal subpart). Reuse it for byte-for-byte parity.
    String::from_utf8_lossy(bytes).into_owned()
}

// ===========================================================================
// UTF-16 (Encoding Standard §14.4 shared UTF-16 decoder)
// ===========================================================================

fn decode_utf16(bytes: &[u8], big_endian: bool) -> String {
    let mut out = String::with_capacity(bytes.len() / 2 + 1);
    let mut i = 0usize;
    let mut pending_lead: Option<u16> = None;
    while i + 1 < bytes.len() {
        let unit = if big_endian {
            (u16::from(bytes[i]) << 8) | u16::from(bytes[i + 1])
        } else {
            u16::from(bytes[i]) | (u16::from(bytes[i + 1]) << 8)
        };
        i += 2;
        if let Some(lead) = pending_lead.take() {
            // Expect a trail surrogate.
            if (0xDC00..=0xDFFF).contains(&unit) {
                let cp = 0x10000
                    + ((u32::from(lead) - 0xD800) << 10)
                    + (u32::from(unit) - 0xDC00);
                out.push(char::from_u32(cp).unwrap_or(REPLACEMENT));
            } else {
                // Unpaired lead: emit replacement, reprocess this unit.
                out.push(REPLACEMENT);
                i -= 2;
            }
            continue;
        }
        if (0xD800..=0xDBFF).contains(&unit) {
            pending_lead = Some(unit);
        } else if (0xDC00..=0xDFFF).contains(&unit) {
            // Unexpected trail surrogate.
            out.push(REPLACEMENT);
        } else {
            out.push(char::from_u32(u32::from(unit)).unwrap_or(REPLACEMENT));
        }
    }
    if pending_lead.is_some() {
        // Trailing unpaired lead surrogate.
        out.push(REPLACEMENT);
    }
    if i < bytes.len() {
        // A trailing odd byte is an error per the spec.
        out.push(REPLACEMENT);
    }
    out
}

// ===========================================================================
// Single-byte decoders (Encoding Standard §9.1) — windows-1252, ISO-8859-x
// ===========================================================================

/// Single-byte decoder: bytes 0x00–0x7F are ASCII; 0x80–0xFF index the
/// 128-entry table (pointer = byte − 0x80). A table entry of 0 means the byte
/// has no mapping → `U+FFFD`.
fn decode_single_byte(bytes: &[u8], table: &[u16; 128]) -> String {
    let mut out = String::with_capacity(bytes.len());
    for &b in bytes {
        if b < 0x80 {
            out.push(b as char);
        } else {
            let cp = table[(b - 0x80) as usize];
            if cp == 0 {
                out.push(REPLACEMENT);
            } else {
                out.push(char::from_u32(u32::from(cp)).unwrap_or(REPLACEMENT));
            }
        }
    }
    out
}

// ===========================================================================
// Multi-byte index helpers
// ===========================================================================

#[inline]
fn index_code_point(table: &[u32], pointer: usize) -> Option<char> {
    match table.get(pointer) {
        Some(&0) | None => None,
        Some(&cp) => char::from_u32(cp),
    }
}

// ===========================================================================
// Shift_JIS (Encoding Standard §12.3.1)
// ===========================================================================

fn decode_shift_jis(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len());
    let mut lead: u8 = 0x00;
    let mut i = 0usize;
    while i < bytes.len() {
        let byte = bytes[i];
        i += 1;
        if lead != 0x00 {
            let leading = lead;
            lead = 0x00;
            let offset = if byte < 0x7F { 0x40u16 } else { 0x41 };
            let leading_offset = if leading < 0xA0 { 0x81u16 } else { 0xC1 };
            let mut pointer: Option<usize> = None;
            if (0x40..=0x7E).contains(&byte) || (0x80..=0xFC).contains(&byte) {
                let p = (u16::from(leading) - leading_offset) * 188 + u16::from(byte) - offset;
                pointer = Some(p as usize);
            }
            if let Some(p) = pointer {
                // EUDC range 8836..=10715 maps linearly to the PUA at U+E000.
                if (8836..=10715).contains(&p) {
                    out.push(char::from_u32(0xE000 - 8836 + p as u32).unwrap_or(REPLACEMENT));
                    continue;
                }
                if let Some(c) = index_code_point(&tbl::JIS0208, p) {
                    out.push(c);
                    continue;
                }
            }
            // No mapping. If the current byte is ASCII, reprocess it.
            out.push(REPLACEMENT);
            if byte < 0x80 {
                i -= 1;
            }
            continue;
        }
        if byte < 0x80 || byte == 0x80 {
            out.push(byte as char);
        } else if (0xA1..=0xDF).contains(&byte) {
            // half-width katakana
            out.push(
                char::from_u32(0xFF61 - 0xA1 + u32::from(byte)).unwrap_or(REPLACEMENT),
            );
        } else if (0x81..=0x9F).contains(&byte) || (0xE0..=0xFC).contains(&byte) {
            lead = byte;
        } else {
            out.push(REPLACEMENT);
        }
    }
    if lead != 0x00 {
        out.push(REPLACEMENT);
    }
    out
}

// ===========================================================================
// EUC-JP (Encoding Standard §12.1.1) — uses jis0208 for the main plane.
// (jis0212 / the 0x8F three-byte plane is rare; those bytes decode to U+FFFD,
//  matching the spec's null-pointer error mode for an unmapped plane.)
// ===========================================================================

fn decode_euc_jp(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len());
    let mut lead: u8 = 0x00;
    let mut jis0212 = false;
    let mut i = 0usize;
    while i < bytes.len() {
        let byte = bytes[i];
        i += 1;
        if lead == 0x8E && (0xA1..=0xDF).contains(&byte) {
            // half-width katakana via the 0x8E single-shift
            lead = 0x00;
            out.push(char::from_u32(0xFF61 - 0xA1 + u32::from(byte)).unwrap_or(REPLACEMENT));
            continue;
        }
        if lead == 0x8F && (0xA1..=0xFE).contains(&byte) {
            // 0x8F introduces a JIS X 0212 char (3-byte form). Mark and read
            // the next byte as the second.
            jis0212 = true;
            lead = byte;
            continue;
        }
        if lead != 0x00 {
            let leading = lead;
            lead = 0x00;
            if (0xA1..=0xFE).contains(&leading) && (0xA1..=0xFE).contains(&byte) {
                let p = (usize::from(leading) - 0xA1) * 94 + usize::from(byte) - 0xA1;
                // We carry only jis0208; jis0212 is not tabled here, so a
                // 0x8F sequence falls through to replacement (spec null mode).
                if !jis0212 {
                    if let Some(c) = index_code_point(&tbl::JIS0208, p) {
                        jis0212 = false;
                        out.push(c);
                        continue;
                    }
                }
                jis0212 = false;
                out.push(REPLACEMENT);
                continue;
            }
            jis0212 = false;
            out.push(REPLACEMENT);
            if byte < 0x80 {
                i -= 1;
            }
            continue;
        }
        if byte < 0x80 {
            out.push(byte as char);
        } else if byte == 0x8E || byte == 0x8F || (0xA1..=0xFE).contains(&byte) {
            lead = byte;
        } else {
            out.push(REPLACEMENT);
        }
    }
    if lead != 0x00 {
        out.push(REPLACEMENT);
    }
    out
}

// ===========================================================================
// EUC-KR (Encoding Standard §11.1.1) — index EUC-KR
// ===========================================================================

fn decode_euc_kr(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len());
    let mut lead: u8 = 0x00;
    let mut i = 0usize;
    while i < bytes.len() {
        let byte = bytes[i];
        i += 1;
        if lead != 0x00 {
            let leading = lead;
            lead = 0x00;
            if (0x41..=0xFE).contains(&byte) {
                let p = (usize::from(leading) - 0x81) * 190 + usize::from(byte) - 0x41;
                if let Some(c) = index_code_point(&tbl::EUC_KR, p) {
                    out.push(c);
                    continue;
                }
            }
            out.push(REPLACEMENT);
            if byte < 0x80 {
                i -= 1;
            }
            continue;
        }
        if byte < 0x80 {
            out.push(byte as char);
        } else if (0x81..=0xFE).contains(&byte) {
            lead = byte;
        } else {
            out.push(REPLACEMENT);
        }
    }
    if lead != 0x00 {
        out.push(REPLACEMENT);
    }
    out
}

// ===========================================================================
// Big5 (Encoding Standard §13.1.1) — index Big5
// ===========================================================================

fn decode_big5(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len());
    let mut lead: u8 = 0x00;
    let mut i = 0usize;
    while i < bytes.len() {
        let byte = bytes[i];
        i += 1;
        if lead != 0x00 {
            let leading = lead;
            lead = 0x00;
            let offset = if byte < 0x7F { 0x40usize } else { 0x62 };
            if (0x40..=0x7E).contains(&byte) || (0xA1..=0xFE).contains(&byte) {
                let p = (usize::from(leading) - 0x81) * 157 + usize::from(byte) - offset;
                // Big5 has a few pointers that decode to two code points
                // (1133, 1135, 1164, 1166 -> combining-mark pairs).
                match p {
                    1133 => {
                        out.push('\u{00CA}');
                        out.push('\u{0304}');
                        continue;
                    }
                    1135 => {
                        out.push('\u{00CA}');
                        out.push('\u{030C}');
                        continue;
                    }
                    1164 => {
                        out.push('\u{00EA}');
                        out.push('\u{0304}');
                        continue;
                    }
                    1166 => {
                        out.push('\u{00EA}');
                        out.push('\u{030C}');
                        continue;
                    }
                    _ => {}
                }
                if let Some(c) = index_code_point(&tbl::BIG5, p) {
                    out.push(c);
                    continue;
                }
            }
            out.push(REPLACEMENT);
            if byte < 0x80 {
                i -= 1;
            }
            continue;
        }
        if byte < 0x80 {
            out.push(byte as char);
        } else if (0x81..=0xFE).contains(&byte) {
            lead = byte;
        } else {
            out.push(REPLACEMENT);
        }
    }
    if lead != 0x00 {
        out.push(REPLACEMENT);
    }
    out
}

// ===========================================================================
// gb18030 / GBK (Encoding Standard §10.2.1) — index gb18030 + ranges
// ===========================================================================

fn decode_gb18030(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len());
    let mut first: u8 = 0x00;
    let mut second: u8 = 0x00;
    let mut third: u8 = 0x00;
    let mut i = 0usize;
    while i < bytes.len() {
        let byte = bytes[i];
        i += 1;

        // Four-byte: third byte already set.
        if third != 0x00 {
            if (0x30..=0x39).contains(&byte) {
                let pointer = (u32::from(first) - 0x81) * 12600
                    + (u32::from(second) - 0x30) * 1260
                    + (u32::from(third) - 0x81) * 10
                    + (u32::from(byte) - 0x30);
                let cp = gb18030_ranges_code_point(pointer);
                first = 0;
                second = 0;
                third = 0;
                match cp {
                    Some(c) => out.push(c),
                    None => out.push(REPLACEMENT),
                }
                continue;
            }
            // bad fourth byte: emit replacement, reprocess second/third/byte
            out.push(REPLACEMENT);
            // restore second, third, current byte to the stream
            i -= 1;
            // Reprocess from `second` onward: simplest correct approach is to
            // push back by reconstructing. We reset state and re-feed the two
            // saved bytes followed by current byte.
            let s = second;
            let t = third;
            first = 0;
            second = 0;
            third = 0;
            // Re-run those bytes through the single-byte fallthrough by
            // prepending: since this is rare, decode them as standalone.
            for rb in [s, t] {
                if rb < 0x80 {
                    out.push(rb as char);
                } else {
                    out.push(REPLACEMENT);
                }
            }
            continue;
        }

        // Three-byte: second already set.
        if second != 0x00 {
            if (0x81..=0xFE).contains(&byte) {
                third = byte;
                continue;
            }
            // bad third byte
            out.push(REPLACEMENT);
            let s = second;
            first = 0;
            second = 0;
            if s < 0x80 {
                out.push(s as char);
            } else {
                out.push(REPLACEMENT);
            }
            i -= 1;
            continue;
        }

        // Two-byte: first already set.
        if first != 0x00 {
            if (0x30..=0x39).contains(&byte) {
                second = byte;
                continue;
            }
            let leading = first;
            first = 0;
            let offset = if byte < 0x7F { 0x40usize } else { 0x41 };
            if (0x40..=0x7E).contains(&byte) || (0x80..=0xFE).contains(&byte) {
                let p = (usize::from(leading) - 0x81) * 190 + usize::from(byte) - offset;
                if let Some(c) = index_code_point(&tbl::GB18030, p) {
                    out.push(c);
                    continue;
                }
            }
            out.push(REPLACEMENT);
            if byte < 0x80 {
                i -= 1;
            }
            continue;
        }

        // Start state.
        if byte < 0x80 {
            out.push(byte as char);
        } else if byte == 0x80 {
            // 0x80 is the legacy GBK euro sign in some decoders, but the
            // Encoding Standard treats a leading 0x80 as an error.
            out.push(REPLACEMENT);
        } else if (0x81..=0xFE).contains(&byte) {
            first = byte;
        } else {
            out.push(REPLACEMENT);
        }
    }
    if first != 0x00 || second != 0x00 || third != 0x00 {
        out.push(REPLACEMENT);
    }
    out
}

/// "Index gb18030 ranges code point" (Encoding Standard §5.3): convert a
/// four-byte pointer to a code point via the ranges table's linear segments.
fn gb18030_ranges_code_point(pointer: u32) -> Option<char> {
    // Spec special cases.
    if (pointer > 39419 && pointer < 189000) || pointer > 1_237_575 {
        return None;
    }
    if pointer == 7457 {
        return char::from_u32(0xE7C7);
    }
    // Find the last range pointer <= pointer.
    let ptrs = &tbl::GB18030_RANGES_PTR;
    let cps = &tbl::GB18030_RANGES_CP;
    // Binary search for the greatest ptr <= pointer.
    let mut lo = 0usize;
    let mut hi = ptrs.len();
    while lo < hi {
        let mid = (lo + hi) / 2;
        if ptrs[mid] <= pointer {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    if lo == 0 {
        return None;
    }
    let idx = lo - 1;
    let offset_pointer = ptrs[idx];
    let code_point_offset = cps[idx];
    char::from_u32(code_point_offset + (pointer - offset_pointer))
}

#[cfg(test)]
mod tests;
