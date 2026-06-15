//! Unified complex-script shaping pipeline.
//!
//! This ties together the pieces that previously lived as isolated
//! primitives (bidi level resolution in `cv_unicode`, Arabic joining in
//! `shaping`, Indic reordering in `indic`, mark attachment in `gpos`)
//! into the single transform a browser layout/paint pass needs:
//!
//!   logical Unicode text  ──►  visually-ordered shaped glyphs
//!
//! The stages, in the order HarfBuzz / Chrome run them:
//!
//!   1. **Bidi.** Resolve UAX #9 embedding levels for the paragraph
//!      (`cv_unicode::resolve_paragraph`) and split into maximal
//!      same-level *runs*. (Reference: UAX #9 §3.4, and Chrome's
//!      `BidiParagraph::GetVisualRuns`.)
//!   2. **Per-run script shaping** in *logical* order:
//!        * Arabic runs → contextual joining → presentation forms
//!          (`shaping::arabic_positional_forms` + lam-alef ligatures).
//!        * Devanagari runs → initial reordering of the pre-base
//!          I-matra and reph (`indic::reorder_devanagari`).
//!        * Combining marks are kept attached to their base and given a
//!          zero advance so they stack on the base instead of consuming
//!          horizontal space (UAX #44 Canonical_Combining_Class != 0).
//!   3. **Visual ordering (UAX #9 L2).** Reverse the contents of each
//!      odd-level (RTL) run and lay runs out highest-level-first so the
//!      whole line reads left-to-right for the rasterizer's LTR cursor.
//!
//! The output is a `Vec<ShapedGlyph>` in visual (paint) order. For the
//! GDI text path the `codepoint` field is what gets drawn; the
//! `cluster`, `x_advance` and `x_offset/y_offset` fields carry the
//! positioning a glyph-atlas rasterizer consumes directly.

use crate::indic;
use crate::shaping;

/// One shaped glyph in visual order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ShapedGlyph {
    /// The codepoint to render. For Arabic this is a FE80..FEFC
    /// presentation form; for everything else the (possibly reordered)
    /// source codepoint.
    pub codepoint: u32,
    /// Logical cluster index (byte offset into the original string of
    /// the character this glyph derives from). Lets hit-testing and
    /// caret placement map a glyph back to source text.
    pub cluster: usize,
    /// Embedding level this glyph belongs to (even = LTR, odd = RTL).
    pub level: u8,
    /// True if this glyph is a non-spacing combining mark stacked on
    /// the preceding base glyph (advance forced to 0).
    pub is_mark: bool,
}

/// Canonical_Combining_Class != 0 test (UAX #44). A non-spacing
/// combining mark must not consume horizontal advance — it stacks over
/// its base. This covers the dense, common mark blocks:
///   * Combining Diacritical Marks            U+0300..U+036F
///   * Combining Diacritical Marks Extended    U+1AB0..U+1AFF
///   * Combining Diacritical Marks Supplement  U+1DC0..U+1DFF
///   * Combining Diacritical Marks for Symbols U+20D0..U+20FF
///   * Combining Half Marks                    U+FE20..U+FE2F
///   * Arabic combining marks (harakat)        U+064B..U+065F, U+0670
///   * Devanagari combining signs              U+0900..U+0903, nukta
///     U+093C, vowel signs U+093E..U+094F (those marked non-spacing),
///     U+0951..U+0957, U+0962..U+0963
///   * Hebrew points                           U+0591..U+05BD, U+05BF,
///     U+05C1..U+05C2, U+05C4..U+05C5, U+05C7
pub fn is_combining_mark(c: char) -> bool {
    let cu = c as u32;
    matches!(cu,
        0x0300..=0x036F
        | 0x1AB0..=0x1AFF
        | 0x1DC0..=0x1DFF
        | 0x20D0..=0x20FF
        | 0xFE20..=0xFE2F
        | 0x064B..=0x065F | 0x0670
        | 0x0591..=0x05BD | 0x05BF | 0x05C1..=0x05C2 | 0x05C4..=0x05C5 | 0x05C7
        // Devanagari non-spacing signs & matras (anusvara/candrabindu,
        // nukta, the above/below vowel signs, vedic accents).
        | 0x0900..=0x0902 | 0x093A | 0x093C | 0x0941..=0x0948 | 0x094D
        | 0x0951..=0x0957 | 0x0962..=0x0963
    )
}

/// A maximal run of characters at one embedding level, in logical
/// order, carrying the byte offset of each character for cluster
/// tracking.
#[derive(Debug, Clone)]
struct LevelRun {
    level: u8,
    /// (char, byte_offset) in logical order.
    chars: Vec<(char, usize)>,
}

/// Split `text` into maximal same-level runs using the resolved bidi
/// levels. Returns runs in logical order.
fn split_level_runs(text: &str, levels: &[u8]) -> Vec<LevelRun> {
    let mut runs: Vec<LevelRun> = Vec::new();
    for (idx, (byte_off, c)) in text.char_indices().enumerate() {
        let level = levels.get(idx).copied().unwrap_or(0);
        match runs.last_mut() {
            Some(r) if r.level == level => r.chars.push((c, byte_off)),
            _ => runs.push(LevelRun {
                level,
                chars: vec![(c, byte_off)],
            }),
        }
    }
    runs
}

/// Shape a single run (already isolated to one embedding level) into
/// glyphs **in logical order**. Visual reversal happens later in L2.
fn shape_run(run: &LevelRun) -> Vec<ShapedGlyph> {
    let chars: Vec<char> = run.chars.iter().map(|(c, _)| *c).collect();
    let offsets: Vec<usize> = run.chars.iter().map(|(_, o)| *o).collect();

    let has_arabic = chars
        .iter()
        .any(|&c| !matches!(shaping::joining_type(c), shaping::JoiningType::U));
    let has_devanagari = chars.iter().any(|&c| indic::is_devanagari(c));

    if has_arabic {
        return shape_arabic_run(&chars, &offsets, run.level);
    }
    if has_devanagari {
        return shape_devanagari_run(&chars, &offsets, run.level);
    }
    // Plain run (Latin/CJK/Hebrew-without-presentation-forms/etc.):
    // 1:1 codepoint→glyph, marks flagged non-spacing.
    chars
        .iter()
        .zip(offsets.iter())
        .map(|(&c, &off)| ShapedGlyph {
            codepoint: c as u32,
            cluster: off,
            level: run.level,
            is_mark: is_combining_mark(c),
        })
        .collect()
}

/// Arabic run: contextual joining → presentation forms → lam-alef
/// ligature. Combining harakat keep their base's cluster and are marked
/// non-spacing.
fn shape_arabic_run(chars: &[char], offsets: &[usize], level: u8) -> Vec<ShapedGlyph> {
    let forms = shaping::arabic_positional_forms(chars);
    // Map each base letter to its presentation form (or keep as-is).
    let mut shaped: Vec<ShapedGlyph> = Vec::with_capacity(chars.len());
    for (i, &c) in chars.iter().enumerate() {
        let cp = shaping::presentation_form(c, forms[i])
            .map(|p| p as u32)
            .unwrap_or(c as u32);
        shaped.push(ShapedGlyph {
            codepoint: cp,
            cluster: offsets[i],
            level,
            is_mark: is_combining_mark(c),
        });
    }
    // Lam-alef ligature: collapse FEDD/FEDF + FE8D/FE8E → FEFB/FEFC.
    // We operate on the presentation-form codepoints we just produced.
    let glyph_cps: Vec<u32> = shaped.iter().map(|g| g.codepoint).collect();
    let ligated = shaping::apply_ligatures(&glyph_cps, &shaping::arabic_lam_alef_ligatures());
    if ligated.len() == shaped.len() {
        return shaped; // no ligature fired.
    }
    // Rebuild glyph list, mapping ligature outputs back to the cluster
    // of the first consumed glyph. Walk both lists; when the input run
    // collapsed, the ligature output takes the lam's cluster.
    let mut out: Vec<ShapedGlyph> = Vec::with_capacity(ligated.len());
    let mut src = 0;
    for &lg in &ligated {
        // Find the next source glyph that matches, advancing src.
        if src < shaped.len() && shaped[src].codepoint == lg {
            out.push(shaped[src]);
            src += 1;
        } else {
            // A collapse happened at `src` (lam) consuming lam+alef.
            let base = shaped.get(src).copied().unwrap_or(ShapedGlyph {
                codepoint: lg,
                cluster: offsets.first().copied().unwrap_or(0),
                level,
                is_mark: false,
            });
            out.push(ShapedGlyph {
                codepoint: lg,
                cluster: base.cluster,
                level,
                is_mark: false,
            });
            src += 2; // lam + alef consumed.
        }
    }
    out
}

/// Devanagari run: initial reordering (I-matra before base, reph after
/// base). We reorder per-syllable and rebuild glyphs; the reordered
/// codepoints carry the level. Because reordering changes char count
/// only by consuming the reph's halant, clusters are approximated to the
/// run start offset of the syllable (sufficient for paint; precise
/// cluster mapping is a follow-up tracked in indic.rs).
fn shape_devanagari_run(chars: &[char], offsets: &[usize], level: u8) -> Vec<ShapedGlyph> {
    let logical: String = chars.iter().collect();
    let reordered = indic::reorder_devanagari(&logical);
    let base_off = offsets.first().copied().unwrap_or(0);
    reordered
        .chars()
        .map(|c| ShapedGlyph {
            codepoint: c as u32,
            cluster: base_off,
            level,
            is_mark: is_combining_mark(c),
        })
        .collect()
}

/// Order shaped runs into visual order per UAX #9 rule L2: from the
/// highest level down to the lowest odd level, reverse any contiguous
/// sequence of runs whose level is >= that level. Within an RTL run the
/// glyphs are also reversed so they read right-to-left.
fn reorder_runs_visually(runs: Vec<Vec<ShapedGlyph>>, levels: &[u8]) -> Vec<ShapedGlyph> {
    if runs.is_empty() {
        return Vec::new();
    }
    // Reverse glyphs inside each odd-level (RTL) run first. A base+mark
    // sequence must stay base-then-mark even after the run reverses, so
    // we reverse then re-flip adjacent mark groups back.
    let mut runs: Vec<Vec<ShapedGlyph>> = runs
        .into_iter()
        .zip(levels.iter())
        .map(|(mut g, &lvl)| {
            if lvl % 2 == 1 {
                g.reverse();
                fix_mark_order(&mut g);
            }
            g
        })
        .collect();

    // L2 run reversal.
    let max_level = levels.iter().copied().max().unwrap_or(0);
    let min_odd = (1..=max_level).find(|l| l % 2 == 1).unwrap_or(1);
    for level in (min_odd..=max_level).rev() {
        let mut i = 0;
        while i < runs.len() {
            if levels[i] < level {
                i += 1;
                continue;
            }
            let mut j = i;
            while j < runs.len() && levels[j] >= level {
                j += 1;
            }
            runs[i..j].reverse();
            i = j;
        }
    }
    runs.into_iter().flatten().collect()
}

/// After reversing an RTL run, a `base, mark` pair has become
/// `mark, base`. Restore base-before-mark order so the mark still
/// stacks on the correct base (the mark must follow its base in the
/// glyph stream regardless of paragraph direction). Operates on maximal
/// `mark+` groups preceded by a base.
fn fix_mark_order(glyphs: &mut [ShapedGlyph]) {
    let n = glyphs.len();
    let mut i = 0;
    while i < n {
        if glyphs[i].is_mark {
            // Collect the run of marks starting at i.
            let start = i;
            let mut end = i;
            while end < n && glyphs[end].is_mark {
                end += 1;
            }
            // The base (if any) is the glyph at `end` (it came right
            // after the marks before reversal). Rotate so base leads.
            if end < n {
                // [m m base] -> [base m m]
                glyphs[start..=end].rotate_right(1);
            }
            i = end + 1;
        } else {
            i += 1;
        }
    }
}

/// Full pipeline: shape `text` (a single paragraph / line) into glyphs
/// in visual paint order. `base_level` is the paragraph embedding level
/// (use `cv_unicode::paragraph_level`). The returned glyphs are ready to
/// hand to the rasterizer left-to-right.
pub fn shape_paragraph(text: &str, base_level: u8) -> Vec<ShapedGlyph> {
    if text.is_empty() {
        return Vec::new();
    }
    let levels: Vec<u8> = cv_unicode::resolve_paragraph(text, base_level)
        .into_iter()
        .map(|r| r.0)
        .collect();
    let runs = split_level_runs(text, &levels);
    let run_levels: Vec<u8> = runs.iter().map(|r| r.level).collect();
    let shaped_runs: Vec<Vec<ShapedGlyph>> = runs.iter().map(shape_run).collect();
    reorder_runs_visually(shaped_runs, &run_levels)
}

/// Convenience: shape and return just the visual codepoint string, the
/// shape the existing GDI `DrawTextW` path consumes. Equivalent to
/// `shape_paragraph` then collecting `codepoint`s. Auto-detects the
/// paragraph base level.
pub fn shape_to_visual_string(text: &str) -> String {
    let base = cv_unicode::paragraph_level(text);
    shape_paragraph(text, base)
        .into_iter()
        .filter_map(|g| char::from_u32(g.codepoint))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ascii_is_identity() {
        let g = shape_paragraph("hello", 0);
        let s: String = g.iter().filter_map(|x| char::from_u32(x.codepoint)).collect();
        assert_eq!(s, "hello");
        assert!(g.iter().all(|x| x.level == 0 && !x.is_mark));
    }

    #[test]
    fn arabic_word_takes_joined_forms_not_isolated() {
        // بتب (beh-teh-beh): three dual-joining letters. After shaping
        // the middle MUST be a medial form and differ from the isolated
        // form (real joining, not 1:1 codepoint→glyph).
        let g = shape_paragraph("\u{0628}\u{062A}\u{0628}", 1);
        // RTL run → visual order is reversed. Logical: beh-init,
        // teh-medial, beh-final. Visual (RTL): beh-final, teh-medial,
        // beh-init.
        let cps: Vec<u32> = g.iter().map(|x| x.codepoint).collect();
        // Medial teh = FE98; isolated teh = FE95. Assert the medial form
        // was selected, proving contextual shaping happened.
        assert!(cps.contains(&0xFE98), "teh must take its medial form FE98, got {cps:02X?}");
        assert!(!cps.contains(&0xFE95), "teh must NOT be the isolated form FE95");
        // Visual order is RTL: first glyph is beh-FINAL (FE90), last is
        // beh-INITIAL (FE91).
        assert_eq!(cps.first().copied(), Some(0xFE90), "first visual glyph = beh final");
        assert_eq!(cps.last().copied(), Some(0xFE91), "last visual glyph = beh initial");
    }

    #[test]
    fn devanagari_i_matra_reorders_before_base() {
        // कि = KA + I-matra. LTR paragraph; the I-matra must come first.
        let g = shape_paragraph("\u{0915}\u{093F}", 0);
        let cps: Vec<u32> = g.iter().map(|x| x.codepoint).collect();
        assert_eq!(cps, vec![0x093F, 0x0915]);
    }

    #[test]
    fn combining_mark_stays_with_base_and_is_nonspacing() {
        // "e" + combining acute (U+0301). The mark must follow its base
        // and be flagged non-spacing.
        let g = shape_paragraph("e\u{0301}", 0);
        assert_eq!(g.len(), 2);
        assert_eq!(g[0].codepoint, 'e' as u32);
        assert!(!g[0].is_mark);
        assert_eq!(g[1].codepoint, 0x0301);
        assert!(g[1].is_mark, "combining acute must be non-spacing");
        // Same cluster ownership is implied by adjacency; the mark
        // retains its own cluster offset (byte 1).
        assert_eq!(g[1].cluster, 1);
    }

    #[test]
    fn bidi_mixed_string_orders_runs_ltr_base() {
        // "A" + Arabic "با" (beh-alef) + "B" in an LTR paragraph.
        // Visual order: A [alef beh] B  — the Arabic run is reversed and
        // sits between the Latin letters.
        let g = shape_paragraph("A\u{0628}\u{0627}B", 0);
        let s: Vec<u32> = g.iter().map(|x| x.codepoint).collect();
        assert_eq!(s.first().copied(), Some('A' as u32));
        assert_eq!(s.last().copied(), Some('B' as u32));
        // Middle two are the Arabic glyphs in RTL visual order: alef
        // (final form FE8E since beh joins to it) then beh (initial FE91).
        assert_eq!(s[1], 0xFE8E, "alef final form first (RTL)");
        assert_eq!(s[2], 0xFE91, "beh initial form second");
        // Levels: A/B are 0, Arabic glyphs are 1.
        assert_eq!(g[0].level, 0);
        assert_eq!(g[1].level, 1);
        assert_eq!(g[2].level, 1);
        assert_eq!(g[3].level, 0);
    }

    #[test]
    fn hebrew_combining_mark_stays_attached_after_rtl_reversal() {
        // Hebrew alef (05D0) + combining point (05B0, sheva). In an RTL
        // paragraph the run reverses, but the mark must remain AFTER its
        // base in the glyph stream so it stacks correctly.
        let g = shape_paragraph("\u{05D0}\u{05B0}", 1);
        // Find the base and mark.
        let base_idx = g.iter().position(|x| x.codepoint == 0x05D0).unwrap();
        let mark_idx = g.iter().position(|x| x.codepoint == 0x05B0).unwrap();
        assert!(g[mark_idx].is_mark);
        assert_eq!(
            mark_idx,
            base_idx + 1,
            "mark must immediately follow its base even after RTL reversal"
        );
    }

    #[test]
    fn lam_alef_ligature_collapses_in_arabic_run() {
        // lam (0644) + alef (0627) → lam-alef ligature (FEFB or FEFC).
        let g = shape_paragraph("\u{0644}\u{0627}", 1);
        let cps: Vec<u32> = g.iter().map(|x| x.codepoint).collect();
        assert_eq!(g.len(), 1, "lam+alef must collapse to one glyph");
        assert!(
            cps[0] == 0xFEFB || cps[0] == 0xFEFC,
            "expected lam-alef ligature, got {:04X}",
            cps[0]
        );
    }

    #[test]
    fn level_is_respected_for_each_glyph() {
        let g = shape_paragraph("ab\u{05D0}\u{05D1}cd", 0);
        // a b (L=0) alef bet (L=1) c d (L=0)
        assert!(g.iter().filter(|x| x.level == 1).count() == 2);
        assert!(g.iter().filter(|x| x.level == 0).count() == 4);
    }
}
