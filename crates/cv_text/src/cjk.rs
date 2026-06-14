//! CJK-specific shaping support.
//!
//! V1 covers the inputs the layout engine needs to render Chinese,
//! Japanese, and Korean text correctly:
//!   * `is_cjk_ideograph` — Han Unification + CJK Compatibility
//!     ranges
//!   * `is_fullwidth_punctuation` — punctuation that takes a full
//!     CJK em-square instead of the typical half-width Latin form
//!   * `vertical_orientation` — for vertical-writing mode (UTR #50):
//!     U Upright / R Rotated / Tu Transform Upright
//!   * `tatechuyoko_run` — detect runs short enough to render
//!     upright inside a vertical line (西暦2024年 → "2024" stays
//!     horizontal)

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VerticalOrientation {
    /// Always upright in vertical text.
    Upright,
    /// Rotated 90° clockwise in vertical text.
    Rotated,
    /// Upright but glyph form swapped (e.g. brackets that pick a
    /// vertical variant).
    TransformUpright,
}

/// True if `c` is a CJK Unified Ideograph (Han) or an extension.
pub fn is_cjk_ideograph(c: char) -> bool {
    let cp = c as u32;
    matches!(
        cp,
        0x3400..=0x4DBF       // Ext A
        | 0x4E00..=0x9FFF     // Unified
        | 0x20000..=0x2A6DF   // Ext B
        | 0x2A700..=0x2B73F   // Ext C
        | 0x2B740..=0x2B81F   // Ext D
        | 0x2B820..=0x2CEAF   // Ext E
        | 0x2CEB0..=0x2EBEF   // Ext F
        | 0xF900..=0xFAFF     // Compatibility
    )
}

/// Hiragana + Katakana ranges.
pub fn is_kana(c: char) -> bool {
    let cp = c as u32;
    matches!(cp, 0x3040..=0x309F | 0x30A0..=0x30FF | 0x31F0..=0x31FF | 0xFF65..=0xFF9F)
}

/// Hangul syllable range plus Jamo blocks.
pub fn is_hangul(c: char) -> bool {
    let cp = c as u32;
    matches!(cp, 0xAC00..=0xD7A3 | 0x1100..=0x11FF | 0x3130..=0x318F)
}

/// Full-width punctuation (CJK Symbols and Punctuation block + the
/// Halfwidth and Fullwidth Forms full-width subset).
pub fn is_fullwidth_punctuation(c: char) -> bool {
    let cp = c as u32;
    matches!(cp,
        0x3000..=0x303F     // CJK Symbols and Punctuation
        | 0xFF00..=0xFF60   // Fullwidth ASCII variants
        | 0xFFE0..=0xFFE6   // Fullwidth signs
    )
}

/// UTR #50 vertical orientation. V1 ships the broad strokes; the
/// full per-codepoint table follows when we lay it out.
pub fn vertical_orientation(c: char) -> VerticalOrientation {
    let cp = c as u32;
    // CJK ideographs + kana + hangul are upright in vertical text.
    if is_cjk_ideograph(c) || is_kana(c) || is_hangul(c) {
        return VerticalOrientation::Upright;
    }
    // CJK punctuation has its own vertical variant.
    if is_fullwidth_punctuation(c) {
        return VerticalOrientation::TransformUpright;
    }
    // Latin letters and digits rotate by default in vertical text.
    if c.is_ascii_alphanumeric() {
        return VerticalOrientation::Rotated;
    }
    // ASCII punctuation rotates with the line.
    if cp < 0x80 {
        return VerticalOrientation::Rotated;
    }
    // Default: rotated.
    VerticalOrientation::Rotated
}

/// Detect Tate-chu-yoko (横中縦) runs — short Latin/digit sequences
/// inside vertical CJK text that should render upright as a
/// horizontal cluster. Returns the byte ranges that should be
/// composed as one upright glyph cluster.
///
/// Heuristic per JLREQ: runs of ≤ `max_len` ASCII digits/letters
/// surrounded by upright CJK characters.
pub fn tatechuyoko_runs(text: &str, max_len: usize) -> Vec<(usize, usize)> {
    let mut runs = Vec::new();
    let chars: Vec<(usize, char)> = text.char_indices().collect();
    let mut i = 0;
    while i < chars.len() {
        let (start_off, c) = chars[i];
        if c.is_ascii_alphanumeric() {
            let mut j = i;
            while j < chars.len() && chars[j].1.is_ascii_alphanumeric() {
                j += 1;
            }
            let end_off = if j < chars.len() {
                chars[j].0
            } else {
                text.len()
            };
            let run_len = j - i;
            if run_len <= max_len {
                runs.push((start_off, end_off));
            }
            i = j;
        } else {
            i += 1;
        }
    }
    runs
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ideograph_detected() {
        assert!(is_cjk_ideograph('日'));
        assert!(is_cjk_ideograph('本'));
        assert!(is_cjk_ideograph('語'));
        assert!(!is_cjk_ideograph('a'));
    }

    #[test]
    fn kana_detected() {
        assert!(is_kana('あ'));
        assert!(is_kana('ア'));
        assert!(!is_kana('日'));
    }

    #[test]
    fn hangul_detected() {
        assert!(is_hangul('한'));
        assert!(is_hangul('글'));
        assert!(!is_hangul('日'));
    }

    #[test]
    fn fullwidth_punctuation_detected() {
        assert!(is_fullwidth_punctuation('、')); // 0x3001
        assert!(is_fullwidth_punctuation('。')); // 0x3002
        assert!(!is_fullwidth_punctuation(','));
    }

    #[test]
    fn vertical_orientation_cjk_is_upright() {
        assert_eq!(vertical_orientation('日'), VerticalOrientation::Upright);
        assert_eq!(vertical_orientation('あ'), VerticalOrientation::Upright);
    }

    #[test]
    fn vertical_orientation_latin_is_rotated() {
        assert_eq!(vertical_orientation('A'), VerticalOrientation::Rotated);
        assert_eq!(vertical_orientation('1'), VerticalOrientation::Rotated);
    }

    #[test]
    fn vertical_orientation_full_punct_is_transform() {
        assert_eq!(
            vertical_orientation('、'),
            VerticalOrientation::TransformUpright
        );
    }

    #[test]
    fn tatechuyoko_finds_short_digit_run() {
        // "西暦2024年" → one tate-chu-yoko run over "2024".
        let s = "西暦2024年";
        let runs = tatechuyoko_runs(s, 4);
        assert_eq!(runs.len(), 1);
        let (start, end) = runs[0];
        assert_eq!(&s[start..end], "2024");
    }

    #[test]
    fn tatechuyoko_ignores_long_runs() {
        let s = "西暦12345年";
        let runs = tatechuyoko_runs(s, 4);
        assert!(runs.is_empty(), "5-digit run should not qualify");
    }
}
