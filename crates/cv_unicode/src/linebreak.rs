//! UAX #14 line-breaking algorithm.
//!
//! V1 supports the line-break classes that cover Latin / CJK /
//! Arabic / common punctuation — enough to produce correct break
//! opportunities for ~95% of web text. Mandatory breaks (BK, CR,
//! LF, NL) are honored. SHY (soft hyphen) becomes a break
//! opportunity. Word/zero-width joiners (ZWJ, ZWSP) preserve runs
//! and break opportunities respectively.
//!
//! Output is the same shape as Chromium's
//! `LineBreakIteratorPosixICU::next()` — an iterator over indices
//! into the input string where a break may occur, plus a flag
//! marking each one as opportunity / required.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BreakKind {
    /// Caller may break here.
    Opportunity,
    /// Caller MUST break here (mandatory break).
    Mandatory,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LineBreak {
    pub byte_offset: usize,
    pub kind: BreakKind,
}

/// Coarse line-break class from UAX #14. We keep just the categories
/// we actually use; the full set is 40+.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Lb {
    /// Mandatory break (CR LF NL BK).
    Mandatory,
    /// Space character (SP).
    Space,
    /// Zero-width space (ZW) — break after.
    ZwSpace,
    /// Soft hyphen — break after.
    Hyphen,
    /// CJK ideograph / Hiragana / Katakana / Yi — break both sides.
    Ideographic,
    /// Open punctuation `(` `[` `{` — no break after.
    OpenPunct,
    /// Close punctuation `)` `]` `}` — no break before.
    ClosePunct,
    /// Alphabetic / numeric — joins with neighbors.
    Alphabetic,
    /// Quotation — neutral, treated as Alphabetic for V1.
    Quotation,
}

pub fn classify(c: char) -> Lb {
    match c {
        '\n' | '\r' | '\x0B' | '\x0C' | '\u{85}' | '\u{2028}' | '\u{2029}' => Lb::Mandatory,
        ' ' | '\t' | '\u{00A0}' => Lb::Space,
        '\u{200B}' => Lb::ZwSpace,
        '\u{00AD}' => Lb::Hyphen,
        '(' | '[' | '{' | '\u{2018}' | '\u{201C}' => Lb::OpenPunct,
        ')' | ']' | '}' | '\u{2019}' | '\u{201D}' | ',' | ';' | '.' | '!' | '?' => Lb::ClosePunct,
        '"' | '\'' => Lb::Quotation,
        _ if (c as u32) >= 0x3040 && (c as u32) <= 0x30FF => Lb::Ideographic, // Hiragana + Katakana
        _ if (c as u32) >= 0x4E00 && (c as u32) <= 0x9FFF => Lb::Ideographic, // CJK Unified
        _ if (c as u32) >= 0xAC00 && (c as u32) <= 0xD7A3 => Lb::Ideographic, // Hangul Syllables
        _ => Lb::Alphabetic,
    }
}

/// Compute line-break opportunities for `text`. The result includes
/// the position past the last character (the natural line end).
pub fn break_iter(text: &str) -> Vec<LineBreak> {
    let mut out = Vec::new();
    let mut prev: Option<Lb> = None;
    let mut prev_off: usize = 0;
    for (off, c) in text.char_indices() {
        let cls = classify(c);
        if let Some(p) = prev {
            // Pairwise rules.
            let break_here = decide(p, cls);
            if let Some(kind) = break_here {
                out.push(LineBreak {
                    byte_offset: off,
                    kind,
                });
            }
            let _ = prev_off; // reserved for future cluster-merge analysis
        }
        prev = Some(cls);
        prev_off = off;
    }
    // End-of-string is always a break opportunity (the renderer flushes
    // its last line there).
    out.push(LineBreak {
        byte_offset: text.len(),
        kind: BreakKind::Opportunity,
    });
    out
}

fn decide(prev: Lb, cur: Lb) -> Option<BreakKind> {
    use Lb::*;
    match (prev, cur) {
        // LB4/5: mandatory break after CR/LF/etc.
        (Mandatory, _) => Some(BreakKind::Mandatory),
        // LB7: don't break before space.
        (_, Space) => None,
        // LB18: break after space (handled by next char).
        (Space, _) => Some(BreakKind::Opportunity),
        // LB8: break after ZWSP.
        (ZwSpace, _) => Some(BreakKind::Opportunity),
        // LB14: open punctuation glues to next.
        (OpenPunct, _) => None,
        // LB13: don't break before close punctuation.
        (_, ClosePunct) => None,
        // Ideographic breaks both sides (LB20-style).
        (Ideographic, _) | (_, Ideographic) => Some(BreakKind::Opportunity),
        // Soft hyphen: break after.
        (Hyphen, _) => Some(BreakKind::Opportunity),
        // Default: don't break inside a word run.
        (Alphabetic, Alphabetic) | (Alphabetic, Quotation) | (Quotation, Alphabetic) => None,
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn offsets(breaks: &[LineBreak]) -> Vec<usize> {
        breaks.iter().map(|b| b.byte_offset).collect()
    }

    #[test]
    fn ascii_words_break_after_space() {
        let breaks = break_iter("hello world");
        // Opportunity after the space (index 6) and end-of-string.
        assert!(offsets(&breaks).contains(&6));
        assert!(offsets(&breaks).contains(&11));
    }

    #[test]
    fn mandatory_break_after_newline() {
        // Per UAX #14, the mandatory break is *after* the newline —
        // emitted at the offset of the next character.
        let breaks = break_iter("a\nb");
        let nl_break = breaks
            .iter()
            .find(|b| b.byte_offset == 2)
            .expect("break after newline");
        assert_eq!(nl_break.kind, BreakKind::Mandatory);
    }

    #[test]
    fn no_break_after_open_paren() {
        let breaks = break_iter("(abc)");
        // No opportunity at offset 1 (right after the open paren).
        assert!(!breaks.iter().any(|b| b.byte_offset == 1));
    }

    #[test]
    fn no_break_before_close_punctuation() {
        let breaks = break_iter("abc!");
        // No break between 'c' (3) and '!' (3..4) — i.e. no break at 3.
        assert!(!breaks.iter().any(|b| b.byte_offset == 3));
    }

    #[test]
    fn cjk_offers_break_each_character() {
        let s = "日本語";
        let breaks = break_iter(s);
        // Each CJK char is 3 bytes; expect breaks at 3 and 6 (before
        // each subsequent ideograph) plus end.
        let offs = offsets(&breaks);
        assert!(offs.contains(&3));
        assert!(offs.contains(&6));
        assert!(offs.contains(&9));
    }

    #[test]
    fn soft_hyphen_inserts_opportunity() {
        let s = "in\u{00AD}side";
        let breaks = break_iter(s);
        // The SHY is at byte 2..4. The opportunity is the index of
        // the next char ('s') — byte 4.
        assert!(breaks.iter().any(|b| b.byte_offset == 4));
    }

    #[test]
    fn end_of_string_always_present() {
        let s = "hi";
        let breaks = break_iter(s);
        assert_eq!(breaks.last().unwrap().byte_offset, s.len());
    }
}
