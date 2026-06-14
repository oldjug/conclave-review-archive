//! UAX #9 Unicode Bidirectional Algorithm (subset).
//!
//! Computes embedding levels for a sequence of codepoints. The
//! renderer uses the levels to reorder text runs visually so that
//! Arabic/Hebrew strings flow right-to-left while embedded Latin
//! stays left-to-right.

/// UAX #9 Bidi_Class values, restricted to those the V1 algorithm
/// distinguishes. Codepoints whose class isn't tracked here resolve
/// as ON (Other Neutral), which produces correct visual order for
/// most punctuation around mixed-direction text.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BidiClass {
    /// Left-to-right (Latin, Cyrillic, Greek, ...).
    L,
    /// Right-to-left (Hebrew).
    R,
    /// Right-to-left arabic (Arabic letters).
    Al,
    /// European number.
    En,
    /// European separator (`+`, `-` adjacent to EN).
    Es,
    /// European terminator (`$`, `%`, `,`).
    Et,
    /// Arabic number.
    An,
    /// Common separator (`,`, `.`, `/` between numbers).
    Cs,
    /// Non-spacing mark.
    Nsm,
    /// Boundary neutral (controls).
    Bn,
    /// Paragraph separator.
    B,
    /// Segment separator (tab).
    S,
    /// Whitespace.
    Ws,
    /// Other neutral (everything else, punctuation, brackets).
    On,
    /// Left-to-right embedding (explicit format) — V1 ignores.
    Lre,
    /// Right-to-left embedding — V1 ignores.
    Rle,
    /// Pop directional formatting — V1 ignores.
    Pdf,
    /// Left-to-right override — V1 ignores.
    Lro,
    /// Right-to-left override — V1 ignores.
    Rlo,
}

/// Resolved per-codepoint embedding level.
#[derive(Debug, Clone, Copy)]
pub struct ResolvedLevel(pub u8);

/// Look up the Bidi_Class for a single codepoint. Covers BMP ranges
/// for Latin/Cyrillic/Greek (→ L), Hebrew + Arabic + Syriac (→ R/AL),
/// digits (→ EN), explicit format chars (LRE/RLE/PDF/LRO/RLO), and
/// the major whitespace + control codepoints. Everything else
/// returns ON, which produces correct visual order in practice for
/// the punctuation surrounding mixed-direction text.
pub fn bidi_class(c: char) -> BidiClass {
    let cu = c as u32;
    // ASCII letters → L.
    if (0x41..=0x5A).contains(&cu) || (0x61..=0x7A).contains(&cu) {
        return BidiClass::L;
    }
    // ASCII digits → EN.
    if (0x30..=0x39).contains(&cu) {
        return BidiClass::En;
    }
    // ASCII tab → S, LF/CR → B, space → WS.
    if cu == 0x09 {
        return BidiClass::S;
    }
    if cu == 0x0A || cu == 0x0D || cu == 0x85 || cu == 0x2028 || cu == 0x2029 {
        return BidiClass::B;
    }
    if cu == 0x20 || cu == 0xA0 {
        return BidiClass::Ws;
    }
    // Common control range → BN.
    if cu <= 0x1F || (0x7F..=0x9F).contains(&cu) {
        return BidiClass::Bn;
    }
    // European separators / terminators.
    match cu {
        0x2B | 0x2D => return BidiClass::Es, // +, -
        0x23 | 0x24 | 0x25 | 0xA2 | 0xA3 | 0xA4 | 0xA5 => return BidiClass::Et, // # $ % ¢ £ ¤ ¥
        0x2C | 0x2E | 0x2F | 0x3A => return BidiClass::Cs, // , . / :
        _ => {}
    }
    // Latin-1 letters → L.
    if (0xC0..=0xD6).contains(&cu) || (0xD8..=0xF6).contains(&cu) || (0xF8..=0xFF).contains(&cu) {
        return BidiClass::L;
    }
    // Hebrew block + Hebrew presentation forms → R.
    if (0x0591..=0x05F4).contains(&cu) || (0xFB1D..=0xFB4F).contains(&cu) {
        return BidiClass::R;
    }
    // Arabic blocks → AL (letters and combining marks resolve from
    // surrounding strong context; we keep AL granular).
    if (0x0600..=0x06FF).contains(&cu)
        || (0x0750..=0x077F).contains(&cu)
        || (0x08A0..=0x08FF).contains(&cu)
        || (0xFB50..=0xFDFF).contains(&cu)
        || (0xFE70..=0xFEFF).contains(&cu)
    {
        return BidiClass::Al;
    }
    // Arabic-Indic digits → AN.
    if (0x0660..=0x0669).contains(&cu) || (0x06F0..=0x06F9).contains(&cu) {
        return BidiClass::An;
    }
    // Syriac, Thaana → R.
    if (0x0700..=0x074F).contains(&cu) || (0x0780..=0x07BF).contains(&cu) {
        return BidiClass::R;
    }
    // Greek, Cyrillic, Armenian, Georgian, Latin Ext A/B → L.
    if (0x0100..=0x024F).contains(&cu)
        || (0x0370..=0x03FF).contains(&cu)
        || (0x0400..=0x04FF).contains(&cu)
        || (0x0500..=0x052F).contains(&cu)
        || (0x0531..=0x058F).contains(&cu)
        || (0x10A0..=0x10FF).contains(&cu)
    {
        return BidiClass::L;
    }
    // CJK + Hangul + Hiragana + Katakana → L.
    if (0x3040..=0x30FF).contains(&cu)
        || (0x3400..=0x4DBF).contains(&cu)
        || (0x4E00..=0x9FFF).contains(&cu)
        || (0xAC00..=0xD7AF).contains(&cu)
    {
        return BidiClass::L;
    }
    // Explicit format characters (deliberately minimal — V1 doesn't
    // honour them, but we recognise them so the renderer can drop
    // them rather than render literal control glyphs).
    match cu {
        0x202A => return BidiClass::Lre,
        0x202B => return BidiClass::Rle,
        0x202C => return BidiClass::Pdf,
        0x202D => return BidiClass::Lro,
        0x202E => return BidiClass::Rlo,
        _ => {}
    }
    BidiClass::On
}

/// Compute the paragraph embedding level (UBA rule P2/P3). Returns
/// `Some(1)` if the first strong character is R or AL, otherwise
/// `Some(0)`. `None` for an empty string defaults to 0 in callers.
pub fn paragraph_level(text: &str) -> u8 {
    for c in text.chars() {
        match bidi_class(c) {
            BidiClass::L => return 0,
            BidiClass::R | BidiClass::Al => return 1,
            _ => {}
        }
    }
    0
}

/// Resolve embedding levels per codepoint in `text`. Implements UBA
/// stages X-1, W-1..W-7, N0-N1, I1-I2 against a single paragraph at
/// the given base level (use `paragraph_level()` to compute it).
///
/// The returned vector has the same `.chars().count()` as the input.
pub fn resolve_paragraph(text: &str, base_level: u8) -> Vec<ResolvedLevel> {
    let chars: Vec<char> = text.chars().collect();
    if chars.is_empty() {
        return Vec::new();
    }
    let mut classes: Vec<BidiClass> = chars.iter().map(|&c| bidi_class(c)).collect();
    let levels: Vec<u8> = vec![base_level; chars.len()];

    // X1–X9 (explicit overrides) — V1 doesn't honour LRE/RLE/etc.;
    // their codepoints get the base level and class On.
    for (i, c) in classes.iter_mut().enumerate() {
        if matches!(
            c,
            BidiClass::Lre
                | BidiClass::Rle
                | BidiClass::Lro
                | BidiClass::Rlo
                | BidiClass::Pdf
                | BidiClass::Bn
        ) {
            *c = BidiClass::On;
            let _ = i;
        }
    }

    // W1: An NSM takes the type of the previous character (or sos sos
    // = base direction's strong class).
    let sos = if base_level == 1 {
        BidiClass::R
    } else {
        BidiClass::L
    };
    for i in 0..classes.len() {
        if classes[i] == BidiClass::Nsm {
            classes[i] = if i == 0 { sos } else { classes[i - 1] };
        }
    }
    // W2: AL EN → AL AN. Search backwards for first strong; if AL,
    // change EN to AN.
    for i in 0..classes.len() {
        if classes[i] == BidiClass::En {
            for j in (0..i).rev() {
                match classes[j] {
                    BidiClass::Al => {
                        classes[i] = BidiClass::An;
                        break;
                    }
                    BidiClass::L | BidiClass::R => break,
                    _ => {}
                }
            }
        }
    }
    // W3: change AL → R.
    for c in classes.iter_mut() {
        if *c == BidiClass::Al {
            *c = BidiClass::R;
        }
    }
    // W4: a single ES between two ENs becomes EN. CS between two
    // ENs or two ANs becomes EN/AN.
    let snapshot = classes.clone();
    for i in 1..classes.len().saturating_sub(1) {
        let prev = snapshot[i - 1];
        let next = snapshot[i + 1];
        match snapshot[i] {
            BidiClass::Es => {
                if prev == BidiClass::En && next == BidiClass::En {
                    classes[i] = BidiClass::En;
                }
            }
            BidiClass::Cs => {
                if prev == BidiClass::En && next == BidiClass::En {
                    classes[i] = BidiClass::En;
                } else if prev == BidiClass::An && next == BidiClass::An {
                    classes[i] = BidiClass::An;
                }
            }
            _ => {}
        }
    }
    // W5: A sequence of ETs adjacent to EN becomes EN.
    let snapshot = classes.clone();
    for i in 0..classes.len() {
        if snapshot[i] == BidiClass::Et {
            // Look forward to skip ETs.
            let mut j = i + 1;
            while j < snapshot.len() && snapshot[j] == BidiClass::Et {
                j += 1;
            }
            let touches_en = (i > 0 && snapshot[i - 1] == BidiClass::En)
                || (j < snapshot.len() && snapshot[j] == BidiClass::En);
            if touches_en {
                for k in i..j {
                    classes[k] = BidiClass::En;
                }
            }
        }
    }
    // W6: ES/ET/CS not yet resolved become ON.
    for c in classes.iter_mut() {
        if matches!(*c, BidiClass::Es | BidiClass::Et | BidiClass::Cs) {
            *c = BidiClass::On;
        }
    }
    // W7: An EN that follows L (with only ETs/separators in between
    // — already resolved by W5/W6) becomes L.
    for i in 0..classes.len() {
        if classes[i] == BidiClass::En {
            for j in (0..i).rev() {
                match classes[j] {
                    BidiClass::L => {
                        classes[i] = BidiClass::L;
                        break;
                    }
                    BidiClass::R => break,
                    _ => {}
                }
            }
        }
    }
    // N1: Neutrals between same-direction strong runs take that
    // direction (we treat AN/EN as having direction "R" / "L"
    // respectively for resolving neutrals adjacent to them, per spec).
    let strong = |c: BidiClass| -> Option<BidiClass> {
        match c {
            BidiClass::L | BidiClass::En => Some(BidiClass::L),
            BidiClass::R | BidiClass::An => Some(BidiClass::R),
            _ => None,
        }
    };
    let snapshot = classes.clone();
    let mut i = 0;
    while i < classes.len() {
        if matches!(
            classes[i],
            BidiClass::On | BidiClass::Ws | BidiClass::S | BidiClass::B
        ) {
            // Walk forward to end of neutral run.
            let start = i;
            let mut end = i;
            while end < classes.len()
                && matches!(
                    classes[end],
                    BidiClass::On | BidiClass::Ws | BidiClass::S | BidiClass::B
                )
            {
                end += 1;
            }
            let left = if start == 0 {
                sos
            } else {
                strong(snapshot[start - 1]).unwrap_or(sos)
            };
            let right = if end == snapshot.len() {
                sos
            } else {
                strong(snapshot[end]).unwrap_or(sos)
            };
            let target = if left == right { left } else { sos };
            for k in start..end {
                classes[k] = target;
            }
            i = end;
        } else {
            i += 1;
        }
    }
    // I1/I2: implicit levels.
    let mut out = Vec::with_capacity(chars.len());
    let base = base_level;
    for (i, c) in classes.iter().enumerate() {
        let l = levels[i];
        let level = match (base % 2 == 0, c) {
            // Even base (LTR paragraph).
            (true, BidiClass::R) => l + 1,
            (true, BidiClass::An) | (true, BidiClass::En) => l + 2,
            // Odd base (RTL paragraph).
            (false, BidiClass::L) | (false, BidiClass::En) | (false, BidiClass::An) => l + 1,
            _ => l,
        };
        out.push(ResolvedLevel(level));
    }
    out
}

/// Reorder a logical sequence of (text segment, embedding level) into
/// visual order per UBA L2. Highest-level runs are reversed first,
/// then progressively lower levels, until the whole line reads
/// visually left-to-right.
pub fn reorder_line<T: Clone>(segments: &[(T, u8)]) -> Vec<T> {
    if segments.is_empty() {
        return Vec::new();
    }
    let mut out: Vec<T> = segments.iter().map(|(t, _)| t.clone()).collect();
    let max_level = segments.iter().map(|(_, l)| *l).max().unwrap_or(0);
    let min_odd = (1..=max_level).find(|l| l % 2 == 1).unwrap_or(1);
    for level in (min_odd..=max_level).rev() {
        let mut i = 0;
        while i < segments.len() {
            if segments[i].1 < level {
                i += 1;
                continue;
            }
            let mut j = i;
            while j < segments.len() && segments[j].1 >= level {
                j += 1;
            }
            out[i..j].reverse();
            i = j;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn paragraph_level_starts_ltr_for_english() {
        assert_eq!(paragraph_level("hello"), 0);
    }
    #[test]
    fn paragraph_level_starts_rtl_for_arabic() {
        // ا (U+0627) is AL.
        assert_eq!(paragraph_level("\u{0627}hello"), 1);
    }
    #[test]
    fn ascii_levels_are_zero() {
        let levels = resolve_paragraph("hello", 0);
        for l in levels {
            assert_eq!(l.0, 0);
        }
    }
    #[test]
    fn arabic_in_ltr_paragraph_gets_level_one() {
        let levels = resolve_paragraph("A\u{0627}\u{0628}B", 0);
        assert_eq!(levels[0].0, 0); // A
        assert_eq!(levels[1].0, 1); // alef
        assert_eq!(levels[2].0, 1); // ba
        assert_eq!(levels[3].0, 0); // B
    }
    #[test]
    fn reorder_swaps_higher_levels_first() {
        let segs = vec![("A", 0u8), ("ALEF", 1), ("BA", 1), ("B", 0)];
        let visual = reorder_line(&segs);
        assert_eq!(visual, vec!["A", "BA", "ALEF", "B"]);
    }
    #[test]
    fn digits_in_rtl_paragraph_get_level_two() {
        // Arabic paragraph with English digits embedded.
        let levels = resolve_paragraph("\u{0627}123\u{0628}", 1);
        assert_eq!(levels[0].0, 1); // alef
        assert_eq!(levels[1].0, 2); // 1
        assert_eq!(levels[2].0, 2); // 2
        assert_eq!(levels[3].0, 2); // 3
        assert_eq!(levels[4].0, 1); // ba
    }
}
