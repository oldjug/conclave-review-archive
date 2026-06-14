//! Ground-truth test: decode a real Google Fonts variable-font WOFF2
//! (Orbitron) and validate the resulting SFNT is structurally sound.

use std::convert::TryInto;

fn be_u16(b: &[u8], o: usize) -> u16 {
    u16::from_be_bytes(b[o..o + 2].try_into().unwrap())
}
fn be_u32(b: &[u8], o: usize) -> u32 {
    u32::from_be_bytes(b[o..o + 4].try_into().unwrap())
}

// Blocked on the Brotli decoder fully decompressing this font's stream.
// The decoder is byte-exact for the first ~15.4 KB but a residual literal
// desync remains; see cv_compression's `brotli_font` tests. Ignored until
// that lands; run with `--ignored` to check progress.
#[test]
fn orbitron_woff2_decodes_to_valid_sfnt() {
    let input = include_bytes!("orbitron.woff2");
    let sfnt = cv_text::woff2::decode_woff2(input).expect("decode_woff2 failed");

    // Offset table sanity.
    assert!(sfnt.len() > 12, "sfnt too small");
    let num_tables = be_u16(&sfnt, 4) as usize;
    assert!(num_tables > 0 && num_tables < 100, "numTables={num_tables}");

    // Walk the directory; collect tags + verify offsets/lengths in range
    // and the per-table checksum matches what we wrote.
    let mut tags = Vec::new();
    let mut head_off = None;
    for i in 0..num_tables {
        let de = 12 + i * 16;
        let tag = &sfnt[de..de + 4];
        let stored_csum = be_u32(&sfnt, de + 4);
        let off = be_u32(&sfnt, de + 8) as usize;
        let len = be_u32(&sfnt, de + 12) as usize;
        let tagstr = String::from_utf8_lossy(tag).to_string();
        assert!(
            off + len <= sfnt.len(),
            "table {tagstr} out of range off={off} len={len} total={}",
            sfnt.len()
        );
        // Recompute checksum.
        let mut sum: u32 = 0;
        let mut j = 0;
        let body = &sfnt[off..off + len];
        while j + 4 <= body.len() {
            sum = sum.wrapping_add(be_u32(body, j));
            j += 4;
        }
        if j < body.len() {
            let mut last = [0u8; 4];
            last[..body.len() - j].copy_from_slice(&body[j..]);
            sum = sum.wrapping_add(u32::from_be_bytes(last));
        }
        if tagstr != "head" {
            assert_eq!(sum, stored_csum, "checksum mismatch for {tagstr}");
        }
        if tagstr == "head" {
            head_off = Some(off);
        }
        tags.push(tagstr);
    }

    eprintln!("tables: {tags:?}");

    // A variable font must carry these for GDI to render correctly.
    for required in ["glyf", "loca", "head", "hhea", "hmtx", "maxp", "cmap"] {
        assert!(tags.iter().any(|t| t == required), "missing {required}");
    }

    // head.magicNumber must be 0x5F0F3CF5.
    let ho = head_off.expect("no head");
    let magic = be_u32(&sfnt, ho + 12);
    assert_eq!(magic, 0x5F0F_3CF5, "head.magicNumber wrong");

    // head.checkSumAdjustment must equal 0xB1B0AFBA - sum(whole file with
    // this field zeroed). Validate the file-level checksum invariant.
    let mut whole = sfnt.clone();
    // zero the checkSumAdjustment field (head+8).
    whole[ho + 8..ho + 12].copy_from_slice(&[0, 0, 0, 0]);
    let mut total: u32 = 0;
    let mut j = 0;
    while j + 4 <= whole.len() {
        total = total.wrapping_add(be_u32(&whole, j));
        j += 4;
    }
    let expected_adj = 0xB1B0_AFBAu32.wrapping_sub(total);
    let stored_adj = be_u32(&sfnt, ho + 8);
    assert_eq!(
        stored_adj, expected_adj,
        "head.checkSumAdjustment wrong (stored={stored_adj:08X} expected={expected_adj:08X})"
    );
}

// Regression for the WOFF2 glyf reconstruction, verified byte-for-byte
// against fonttools (the same decode Chrome's google/woff2 produces).
// The reference decode of this exact Orbitron fixture has 125 simple
// glyphs with 1872 on-curve and 995 off-curve points. Two bugs both
// corrupted this: (1) the triplet Group-C/D delta cross-index (wrong
// coordinates), (2) the on-curve flag was inverted (`bit7 set` was
// treated as on-curve, but google/woff2 `woff2_dec.cc` does
// `on_curve = !(flag >> 7)`), which swapped every curve anchor with its
// control point. If either regresses, these totals shift (the inversion
// swaps on/off → 995/1872) and this fails.
#[test]
fn orbitron_glyf_on_off_curve_point_totals_match_reference() {
    let input = include_bytes!("orbitron.woff2");
    let sfnt = cv_text::woff2::decode_woff2(input).expect("decode");

    let num_tables = be_u16(&sfnt, 4) as usize;
    let mut find = |want: &[u8; 4]| -> (usize, usize) {
        for i in 0..num_tables {
            let de = 12 + i * 16;
            if &sfnt[de..de + 4] == want {
                return (
                    be_u32(&sfnt, de + 8) as usize,
                    be_u32(&sfnt, de + 12) as usize,
                );
            }
        }
        panic!("missing table {:?}", std::str::from_utf8(want));
    };
    let (head_off, _) = find(b"head");
    let index_to_loc = be_u16(&sfnt, head_off + 50); // 0 = short, 1 = long
    let (maxp_off, _) = find(b"maxp");
    let num_glyphs = be_u16(&sfnt, maxp_off + 4) as usize;
    let (loca_off, _) = find(b"loca");
    let (glyf_off, _) = find(b"glyf");

    let loca = |i: usize| -> usize {
        if index_to_loc == 0 {
            (be_u16(&sfnt, loca_off + i * 2) as usize) * 2
        } else {
            be_u32(&sfnt, loca_off + i * 4) as usize
        }
    };

    let (mut on_curve, mut off_curve, mut simple) = (0usize, 0usize, 0usize);
    for g in 0..num_glyphs {
        let start = glyf_off + loca(g);
        let end = glyf_off + loca(g + 1);
        if end <= start {
            continue; // empty glyph
        }
        let n_contours = i16::from_be_bytes([sfnt[start], sfnt[start + 1]]);
        if n_contours <= 0 {
            continue; // composite or empty
        }
        simple += 1;
        let nc = n_contours as usize;
        // numberOfContours(2) + bbox(8); endPtsOfContours[nc].
        let mut p = start + 10;
        let last_pt = be_u16(&sfnt, p + (nc - 1) * 2) as usize;
        let num_points = last_pt + 1;
        p += nc * 2;
        // instructionLength(2) + instructions.
        let inst_len = be_u16(&sfnt, p) as usize;
        p += 2 + inst_len;
        // Flags array (handles the TrueType REPEAT_FLAG, bit 3).
        let mut read = 0;
        while read < num_points {
            let flag = sfnt[p];
            p += 1;
            let mut repeat = 1;
            if flag & 0x08 != 0 {
                repeat += sfnt[p] as usize;
                p += 1;
            }
            for _ in 0..repeat {
                if read >= num_points {
                    break;
                }
                if flag & 0x01 != 0 {
                    on_curve += 1;
                } else {
                    off_curve += 1;
                }
                read += 1;
            }
        }
    }
    assert_eq!(
        (simple, on_curve, off_curve),
        (125, 1872, 995),
        "glyf on/off-curve totals diverged from the Chrome/fonttools reference \
         (got simple={simple} on={on_curve} off={off_curve})"
    );
}

// Debug hook: when CV_DUMP_SFNT is set, write our decoder's SFNT output
// to that path so it can be diffed byte-for-byte against a reference
// (fonttools) decode. Ground-truth comparison per the Chrome-alignment
// directive — never trust eyeballed screenshots for decoder correctness.
#[test]
fn dump_sfnt_for_reference_diff() {
    if let Ok(path) = std::env::var("CV_DUMP_SFNT") {
        let input = include_bytes!("orbitron.woff2");
        let sfnt = cv_text::woff2::decode_woff2(input).expect("decode");
        std::fs::write(path, sfnt).unwrap();
    }
}

// End-to-end #492: the explorer's Google Fonts (@font-face → fetch →
// register) chain. `register_font_bytes` detects WOFF2, decodes to SFNT
// via the now-byte-exact Brotli + triplet decoders, then hands the SFNT
// to GDI's AddFontMemResourceEx. A `true` return means GDI accepted the
// repacked font — the same path conclave's `fetch_and_register_web_fonts`
// drives for `font-family: 'Orbitron'`. This was the real blocker behind
// #492: before the decoder was correct, this returned false.
#[test]
fn orbitron_woff2_registers_through_gdi() {
    let input = include_bytes!("orbitron.woff2");
    let ok = cv_text::register_font_bytes(input, "Orbitron|test-fixture", "Orbitron");
    assert!(ok, "GDI rejected the decoded Orbitron SFNT");
    // Idempotent: a second call with the same dedupe key short-circuits
    // to true without re-registering.
    let ok2 = cv_text::register_font_bytes(input, "Orbitron|test-fixture", "Orbitron");
    assert!(ok2, "dedupe path should report success");
}
