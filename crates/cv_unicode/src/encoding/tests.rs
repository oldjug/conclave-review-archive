//! Tests for the encoding-detection + legacy-decoder module. Every assertion
//! checks a REAL decoded glyph (the right Unicode scalar), not "doesn't
//! panic" — verified against the WHATWG Encoding Standard index tables.

use super::*;

// --------------------------------------------------------------------------
// Label resolution (Encoding Standard §4.2 "get an encoding")
// --------------------------------------------------------------------------

#[test]
fn labels_map_to_encodings() {
    assert_eq!(encoding_for_label("utf-8"), Some(Encoding::Utf8));
    assert_eq!(encoding_for_label("UTF8"), Some(Encoding::Utf8));
    // The web's "latin1"/"iso-8859-1"/"ascii" are all windows-1252.
    assert_eq!(encoding_for_label("latin1"), Some(Encoding::Windows1252));
    assert_eq!(encoding_for_label("ISO-8859-1"), Some(Encoding::Windows1252));
    assert_eq!(encoding_for_label("ascii"), Some(Encoding::Windows1252));
    assert_eq!(encoding_for_label("cp1252"), Some(Encoding::Windows1252));
    assert_eq!(encoding_for_label("iso-8859-2"), Some(Encoding::Iso8859_2));
    assert_eq!(encoding_for_label("iso-8859-15"), Some(Encoding::Iso8859_15));
    assert_eq!(encoding_for_label("shift_jis"), Some(Encoding::ShiftJis));
    assert_eq!(encoding_for_label("sjis"), Some(Encoding::ShiftJis));
    assert_eq!(encoding_for_label("euc-jp"), Some(Encoding::EucJp));
    assert_eq!(encoding_for_label("gb2312"), Some(Encoding::Gbk));
    assert_eq!(encoding_for_label("gbk"), Some(Encoding::Gbk));
    assert_eq!(encoding_for_label("gb18030"), Some(Encoding::Gb18030));
    assert_eq!(encoding_for_label("euc-kr"), Some(Encoding::EucKr));
    assert_eq!(encoding_for_label("big5"), Some(Encoding::Big5));
    assert_eq!(encoding_for_label("  Shift_JIS  "), Some(Encoding::ShiftJis));
    assert_eq!(encoding_for_label("totally-bogus"), None);
}

// --------------------------------------------------------------------------
// windows-1252 — the C1 punctuation mapping (the whole point vs ISO-8859-1)
// --------------------------------------------------------------------------

#[test]
fn windows1252_curly_quotes_not_replacement() {
    // 0x93 / 0x94 are LEFT/RIGHT DOUBLE QUOTATION MARK in windows-1252,
    // NOT U+FFFD (which is what from_utf8_lossy would produce).
    let s = decode(&[0x93, 0x48, 0x69, 0x94], Encoding::Windows1252);
    assert_eq!(s, "\u{201C}Hi\u{201D}");
    assert!(!s.contains('\u{FFFD}'), "must not be replacement chars");
}

#[test]
fn windows1252_euro_and_others() {
    assert_eq!(decode(&[0x80], Encoding::Windows1252), "\u{20AC}"); // €
    assert_eq!(decode(&[0x85], Encoding::Windows1252), "\u{2026}"); // …
    assert_eq!(decode(&[0x99], Encoding::Windows1252), "\u{2122}"); // ™
    // 0xE9 in windows-1252 is é (same as ISO-8859-1 for this byte).
    assert_eq!(decode(&[0xE9], Encoding::Windows1252), "é");
}

// --------------------------------------------------------------------------
// ISO-8859-1 (== windows-1252 per spec) and ISO-8859-2/-15
// --------------------------------------------------------------------------

#[test]
fn iso8859_1_e_acute() {
    // 0xE9 in ISO-8859-1 -> é (U+00E9). "iso-8859-1" resolves to windows-1252.
    let enc = encoding_for_label("iso-8859-1").unwrap();
    assert_eq!(decode(&[0xE9], enc), "é");
}

#[test]
fn iso8859_2_distinct_from_1252() {
    // 0xB1 in ISO-8859-2 is U+0105 (ą, LATIN SMALL LETTER A WITH OGONEK),
    // which differs from windows-1252 (0xB1 = ±). Proves the right table.
    assert_eq!(decode(&[0xB1], Encoding::Iso8859_2), "\u{0105}");
}

#[test]
fn iso8859_15_euro() {
    // ISO-8859-15 0xA4 is the EURO SIGN (U+20AC) — that's the -15 change.
    assert_eq!(decode(&[0xA4], Encoding::Iso8859_15), "\u{20AC}");
}

// --------------------------------------------------------------------------
// UTF-16
// --------------------------------------------------------------------------

#[test]
fn utf16le_bom_plus_hi() {
    // FF FE BOM + "hi" little-endian.
    let bytes = [0xFF, 0xFE, 0x68, 0x00, 0x69, 0x00];
    assert_eq!(decode_with_detection(&bytes, None), "hi");
}

#[test]
fn utf16be_bom_plus_hi() {
    let bytes = [0xFE, 0xFF, 0x00, 0x68, 0x00, 0x69];
    assert_eq!(decode_with_detection(&bytes, None), "hi");
}

#[test]
fn utf16le_surrogate_pair() {
    // U+1F600 (😀) = surrogate pair D83D DE00, little-endian.
    let bytes = [0xFF, 0xFE, 0x3D, 0xD8, 0x00, 0xDE];
    assert_eq!(decode_with_detection(&bytes, None), "\u{1F600}");
}

// --------------------------------------------------------------------------
// Shift_JIS
// --------------------------------------------------------------------------

#[test]
fn shift_jis_hiragana() {
    // 0x82 0xA0 -> U+3042 (HIRAGANA LETTER A, あ).
    assert_eq!(decode(&[0x82, 0xA0], Encoding::ShiftJis), "\u{3042}");
}

#[test]
fn shift_jis_kanji() {
    // 0x93 0x5C -> U+8CBC.
    assert_eq!(decode(&[0x93, 0x5C], Encoding::ShiftJis), "\u{8CBC}");
}

#[test]
fn shift_jis_halfwidth_katakana() {
    // single byte 0xB1 -> U+FF71 (HALFWIDTH KATAKANA LETTER A, ｱ).
    assert_eq!(decode(&[0xB1], Encoding::ShiftJis), "\u{FF71}");
}

#[test]
fn shift_jis_ascii_passthrough() {
    assert_eq!(decode(b"Hello", Encoding::ShiftJis), "Hello");
}

// --------------------------------------------------------------------------
// EUC-JP / EUC-KR / GBK / Big5
// --------------------------------------------------------------------------

#[test]
fn euc_jp_hiragana() {
    // 0xA4 0xA2 -> U+3042 (あ).
    assert_eq!(decode(&[0xA4, 0xA2], Encoding::EucJp), "\u{3042}");
}

#[test]
fn euc_jp_halfwidth_katakana_via_8e() {
    // 0x8E 0xB1 -> U+FF71 (ｱ) via the 0x8E single-shift.
    assert_eq!(decode(&[0x8E, 0xB1], Encoding::EucJp), "\u{FF71}");
}

#[test]
fn euc_kr_hangul() {
    // 0xB0 0xA1 -> U+AC00 (HANGUL SYLLABLE GA, 가).
    assert_eq!(decode(&[0xB0, 0xA1], Encoding::EucKr), "\u{AC00}");
}

#[test]
fn gbk_zhong() {
    // 0xD6 0xD0 -> U+4E2D (中).
    assert_eq!(decode(&[0xD6, 0xD0], Encoding::Gbk), "\u{4E2D}");
}

#[test]
fn big5_yi() {
    // 0xA4 0x40 -> U+4E00 (一).
    assert_eq!(decode(&[0xA4, 0x40], Encoding::Big5), "\u{4E00}");
}

// --------------------------------------------------------------------------
// gb18030 four-byte sequences (the ranges table)
// --------------------------------------------------------------------------

#[test]
fn gb18030_four_byte() {
    // 0x81 0x30 0x81 0x30 is the first four-byte pointer (0) -> U+0080.
    assert_eq!(decode(&[0x81, 0x30, 0x81, 0x30], Encoding::Gb18030), "\u{0080}");
    // gb18030 must still decode two-byte GBK content.
    assert_eq!(decode(&[0xD6, 0xD0], Encoding::Gb18030), "\u{4E2D}");
}

#[test]
fn gb18030_four_byte_supplementary() {
    // 0x90 0x30 0x81 0x30 -> pointer 189000 -> U+10000 (a supplementary char),
    // exercising the gb18030 ranges' last linear segment.
    assert_eq!(decode(&[0x90, 0x30, 0x81, 0x30], Encoding::Gb18030), "\u{10000}");
}

#[test]
fn gb18030_state_resets_after_four_byte() {
    // A four-byte sequence followed by a two-byte sequence must decode both,
    // proving the decoder's first/second/third state is cleared.
    let bytes = [0x81, 0x30, 0x81, 0x30, 0xD6, 0xD0];
    assert_eq!(decode(&bytes, Encoding::Gb18030), "\u{0080}\u{4E2D}");
}

// --------------------------------------------------------------------------
// The encoding sniffing algorithm (HTML §13.2.3.x)
// --------------------------------------------------------------------------

#[test]
fn bom_beats_conflicting_meta() {
    // A UTF-8 BOM must win even when a <meta> declares a different charset.
    let mut bytes = vec![0xEF, 0xBB, 0xBF];
    bytes.extend_from_slice(b"<meta charset=shift_jis>");
    assert_eq!(detect_encoding(&bytes, None), Encoding::Utf8);
}

#[test]
fn http_charset_beats_meta() {
    let bytes = b"<meta charset=utf-8>";
    assert_eq!(
        detect_encoding(bytes, Some("text/html; charset=Shift_JIS")),
        Encoding::ShiftJis
    );
}

#[test]
fn meta_charset_prescan() {
    let html = b"<!doctype html><html><head><meta charset=shift_jis></head>";
    assert_eq!(prescan_meta(html), Some(Encoding::ShiftJis));
    assert_eq!(detect_encoding(html, None), Encoding::ShiftJis);
}

#[test]
fn meta_http_equiv_content_type_prescan() {
    let html =
        br#"<meta http-equiv="content-type" content="text/html; charset=euc-jp">"#;
    assert_eq!(prescan_meta(html), Some(Encoding::EucJp));
}

#[test]
fn meta_utf16_normalized_to_utf8() {
    // A <meta> declaring UTF-16 is treated as UTF-8 (HTML §13.2.3.3).
    let html = br#"<meta charset="utf-16">"#;
    assert_eq!(prescan_meta(html), Some(Encoding::Utf8));
}

#[test]
fn unlabeled_high_bytes_default_windows1252() {
    // No BOM, no header, no meta, high bytes present -> default windows-1252,
    // NOT UTF-8 (which would mojibake them to U+FFFD).
    let bytes = [0x93, 0x48, 0x69, 0x94]; // "Hi" in curly quotes (1252)
    assert_eq!(detect_encoding(&bytes, None), Encoding::Windows1252);
    let s = decode_with_detection(&bytes, None);
    assert_eq!(s, "\u{201C}Hi\u{201D}");
    assert!(!s.contains('\u{FFFD}'));
}

#[test]
fn meta_after_comment_still_found() {
    let html = b"<!-- a comment with charset=bogus --><meta charset=big5>";
    assert_eq!(prescan_meta(html), Some(Encoding::Big5));
}

#[test]
fn prescan_only_first_1024_bytes() {
    // A meta beyond byte 1024 is NOT honored (matches the spec window).
    let mut html = vec![b' '; 1100];
    html.extend_from_slice(b"<meta charset=shift_jis>");
    assert_eq!(prescan_meta(&html), None);
}

// --------------------------------------------------------------------------
// UTF-8 parity: switching call sites must NOT change UTF-8 output.
// --------------------------------------------------------------------------

#[test]
fn utf8_identical_to_lossy() {
    let samples: &[&[u8]] = &[
        b"plain ascii",
        "héllo wörld — \u{1F600} \u{4E2D}\u{6587}".as_bytes(),
        &[0xE2, 0x82, 0xAC], // €
        &[0xFF, 0xFE, 0x41], // invalid utf-8 (no BOM match for utf8 path)
        &[],
    ];
    for s in samples {
        // Force the UTF-8 decoder and compare to std lossy.
        let ours = decode(s, Encoding::Utf8);
        let std = String::from_utf8_lossy(s).into_owned();
        assert_eq!(ours, std, "UTF-8 decode must match from_utf8_lossy");
    }
}

#[test]
fn utf8_bom_stripped() {
    let bytes = [0xEF, 0xBB, 0xBF, b'h', b'i'];
    assert_eq!(detect_encoding(&bytes, None), Encoding::Utf8);
    assert_eq!(decode_with_detection(&bytes, None), "hi");
}

#[test]
fn content_type_charset_parsing() {
    assert_eq!(
        charset_from_content_type("text/html; charset=UTF-8").as_deref(),
        Some("UTF-8")
    );
    assert_eq!(
        charset_from_content_type("text/html;charset=\"shift_jis\"").as_deref(),
        Some("shift_jis")
    );
    assert_eq!(
        charset_from_content_type("text/html; charset = euc-kr ; foo=bar").as_deref(),
        Some("euc-kr")
    );
    assert_eq!(charset_from_content_type("text/html").as_deref(), None);
}
