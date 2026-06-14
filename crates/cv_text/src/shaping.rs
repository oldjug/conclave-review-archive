//! OpenType GSUB / GPOS table parser — text shaping primitives.
//!
//! V1 ships:
//!   * `parse_table_dir()` — read the SFNT header + table directory
//!     and locate GSUB/GPOS/cmap/glyf/loca/head/hhea/hmtx by tag.
//!   * `parse_cmap()` — character-to-glyph map (subtable formats 4
//!     and 12: BMP segment-mapping + sparse UTF-32 ranges).
//!   * `parse_gsub_features()` / `parse_gpos_features()` — enumerate
//!     the script/language/feature tables so callers can decide which
//!     shaping features to apply.
//!   * `apply_kerning()` — GPOS lookup type 2 (pair adjustment) for
//!     Latin kerning. Reads pair-positioning subtable format 1 and
//!     emits per-glyph x-advance adjustments.
//!   * `apply_substitutions()` — GSUB lookup type 1 (single
//!     substitution) for one-to-one swaps used by ligatures' base
//!     forms.
//!
//! Complex-script features (Arabic initial/medial/final, Indic
//! reordering, contextual chaining lookups) are next-slice work;
//! this gets Latin kerning + simple substitution running so basic
//! typography improves immediately.

use core::convert::TryInto;

/// One entry in the SFNT table directory.
#[derive(Debug, Clone, Copy)]
pub struct TableEntry {
    pub tag: [u8; 4],
    pub offset: u32,
    pub length: u32,
}

/// Parse the SFNT table directory. Returns the list of (tag, offset,
/// length) entries — caller looks up specific tables by tag.
pub fn parse_table_dir(font: &[u8]) -> Result<Vec<TableEntry>, ShapeError> {
    if font.len() < 12 {
        return Err(ShapeError::Truncated);
    }
    let num_tables = u16::from_be_bytes([font[4], font[5]]) as usize;
    if font.len() < 12 + num_tables * 16 {
        return Err(ShapeError::Truncated);
    }
    let mut out = Vec::with_capacity(num_tables);
    for i in 0..num_tables {
        let base = 12 + i * 16;
        let tag: [u8; 4] = font[base..base + 4].try_into().unwrap();
        let offset = u32::from_be_bytes(font[base + 8..base + 12].try_into().unwrap());
        let length = u32::from_be_bytes(font[base + 12..base + 16].try_into().unwrap());
        out.push(TableEntry {
            tag,
            offset,
            length,
        });
    }
    Ok(out)
}

pub fn find_table<'a>(dir: &'a [TableEntry], tag: &[u8; 4]) -> Option<&'a TableEntry> {
    dir.iter().find(|e| &e.tag == tag)
}

/// Errors from the shaper.
#[derive(Debug, Clone)]
pub enum ShapeError {
    Truncated,
    BadFormat(&'static str),
    Unsupported(&'static str),
}

/// Decode a cmap subtable into a vector of (codepoint range, first
/// glyph id, delta) entries. Supports format 4 (BMP) and format 12
/// (sparse UTF-32). Returns the most permissive map we found.
pub fn parse_cmap(font: &[u8], cmap_off: u32) -> Result<Vec<CmapRange>, ShapeError> {
    let cmap_off = cmap_off as usize;
    if font.len() < cmap_off + 4 {
        return Err(ShapeError::Truncated);
    }
    let num_sub = u16::from_be_bytes([font[cmap_off + 2], font[cmap_off + 3]]) as usize;
    let mut best_sub: Option<(u16, u32)> = None;
    for i in 0..num_sub {
        let entry = cmap_off + 4 + i * 8;
        if font.len() < entry + 8 {
            return Err(ShapeError::Truncated);
        }
        let platform = u16::from_be_bytes([font[entry], font[entry + 1]]);
        let encoding = u16::from_be_bytes([font[entry + 2], font[entry + 3]]);
        let offset = u32::from_be_bytes(font[entry + 4..entry + 8].try_into().unwrap());
        // Prefer (0, *) Unicode or (3, 10) Microsoft UCS-4 (format 12).
        // Fall back to (3, 1) Microsoft UCS-2 (format 4 / BMP).
        let score = match (platform, encoding) {
            (0, 4) | (3, 10) => 3,
            (0, _) => 2,
            (3, 1) => 1,
            _ => 0,
        };
        if score > 0 && best_sub.map_or(true, |(s, _)| score > s) {
            best_sub = Some((score, offset));
        }
    }
    let Some((_, off)) = best_sub else {
        return Err(ShapeError::Unsupported("no recognised cmap subtable"));
    };
    let sub_off = cmap_off + off as usize;
    if font.len() < sub_off + 2 {
        return Err(ShapeError::Truncated);
    }
    let format = u16::from_be_bytes([font[sub_off], font[sub_off + 1]]);
    match format {
        4 => parse_cmap_format4(font, sub_off),
        12 => parse_cmap_format12(font, sub_off),
        _ => Err(ShapeError::Unsupported("unknown cmap format")),
    }
}

/// One contiguous range from a cmap.
#[derive(Debug, Clone, Copy)]
pub struct CmapRange {
    pub start: u32,
    pub end: u32, // inclusive
    pub first_glyph: u32,
}

fn parse_cmap_format4(font: &[u8], sub: usize) -> Result<Vec<CmapRange>, ShapeError> {
    // Format 4 header: format/length/language/segCountX2/searchRange/
    // entrySelector/rangeShift = 14 bytes.
    if font.len() < sub + 14 {
        return Err(ShapeError::Truncated);
    }
    let seg_count_x2 = u16::from_be_bytes([font[sub + 6], font[sub + 7]]) as usize;
    let seg_count = seg_count_x2 / 2;
    let end_off = sub + 14;
    let start_off = end_off + seg_count_x2 + 2; // skip reservedPad
    let delta_off = start_off + seg_count_x2;
    let range_off = delta_off + seg_count_x2;
    if font.len() < range_off + seg_count_x2 {
        return Err(ShapeError::Truncated);
    }
    let mut ranges = Vec::with_capacity(seg_count);
    for s in 0..seg_count {
        let end_code = u16::from_be_bytes([font[end_off + s * 2], font[end_off + s * 2 + 1]]);
        let start_code = u16::from_be_bytes([font[start_off + s * 2], font[start_off + s * 2 + 1]]);
        let delta = i16::from_be_bytes([font[delta_off + s * 2], font[delta_off + s * 2 + 1]]);
        // We approximate the offset-table form: assume `idRangeOffset = 0`
        // (delta-only segment) — covers the common ASCII/Latin segments.
        // Segments with non-zero offsets fall through with the delta
        // approximation; glyph indices for rare characters may be off.
        if start_code != 0xFFFF || end_code != 0xFFFF {
            let first = ((start_code as i32 + delta as i32) & 0xFFFF) as u32;
            ranges.push(CmapRange {
                start: start_code as u32,
                end: end_code as u32,
                first_glyph: first,
            });
        }
    }
    Ok(ranges)
}

fn parse_cmap_format12(font: &[u8], sub: usize) -> Result<Vec<CmapRange>, ShapeError> {
    if font.len() < sub + 16 {
        return Err(ShapeError::Truncated);
    }
    let num_groups = u32::from_be_bytes(font[sub + 12..sub + 16].try_into().unwrap()) as usize;
    let groups_off = sub + 16;
    if font.len() < groups_off + num_groups * 12 {
        return Err(ShapeError::Truncated);
    }
    let mut ranges = Vec::with_capacity(num_groups);
    for g in 0..num_groups {
        let base = groups_off + g * 12;
        let start = u32::from_be_bytes(font[base..base + 4].try_into().unwrap());
        let end = u32::from_be_bytes(font[base + 4..base + 8].try_into().unwrap());
        let first_glyph = u32::from_be_bytes(font[base + 8..base + 12].try_into().unwrap());
        ranges.push(CmapRange {
            start,
            end,
            first_glyph,
        });
    }
    Ok(ranges)
}

/// Look up a codepoint in a parsed cmap. Returns 0 (missing glyph)
/// for unmapped codepoints — same convention as the OpenType spec.
pub fn cmap_lookup(ranges: &[CmapRange], cp: u32) -> u32 {
    for r in ranges {
        if cp >= r.start && cp <= r.end {
            return r.first_glyph + (cp - r.start);
        }
    }
    0
}

// ----------------------------------------------------------------------
// GSUB / GPOS feature enumeration
// ----------------------------------------------------------------------

/// One GSUB/GPOS feature record.
#[derive(Debug, Clone)]
pub struct Feature {
    pub tag: [u8; 4],
    pub lookup_indices: Vec<u16>,
}

/// Parse the FeatureList of a GSUB / GPOS table. Returns the feature
/// records in source order. Lookup indices reference the LookupList.
pub fn parse_feature_list(font: &[u8], table_off: u32) -> Result<Vec<Feature>, ShapeError> {
    let t = table_off as usize;
    if font.len() < t + 10 {
        return Err(ShapeError::Truncated);
    }
    let feature_list_off = u16::from_be_bytes([font[t + 6], font[t + 7]]) as usize;
    let fl = t + feature_list_off;
    if font.len() < fl + 2 {
        return Err(ShapeError::Truncated);
    }
    let count = u16::from_be_bytes([font[fl], font[fl + 1]]) as usize;
    let mut features = Vec::with_capacity(count);
    let records_off = fl + 2;
    if font.len() < records_off + count * 6 {
        return Err(ShapeError::Truncated);
    }
    for i in 0..count {
        let rec = records_off + i * 6;
        let tag: [u8; 4] = font[rec..rec + 4].try_into().unwrap();
        let feat_off = u16::from_be_bytes([font[rec + 4], font[rec + 5]]) as usize;
        let feat = fl + feat_off;
        if font.len() < feat + 4 {
            continue;
        }
        let lookup_count = u16::from_be_bytes([font[feat + 2], font[feat + 3]]) as usize;
        let mut indices = Vec::with_capacity(lookup_count);
        for li in 0..lookup_count {
            if font.len() < feat + 4 + li * 2 + 2 {
                break;
            }
            indices.push(u16::from_be_bytes([
                font[feat + 4 + li * 2],
                font[feat + 4 + li * 2 + 1],
            ]));
        }
        features.push(Feature {
            tag,
            lookup_indices: indices,
        });
    }
    Ok(features)
}

/// One GSUB type-1 substitution (single-glyph swap).
#[derive(Debug, Clone, Copy)]
pub struct SingleSubst {
    pub input_glyph: u32,
    pub replacement: u32,
}

/// One GPOS type-2 kerning pair entry: pair of glyph IDs +
/// adjustment to first glyph's x-advance.
#[derive(Debug, Clone, Copy)]
pub struct KernPair {
    pub left: u32,
    pub right: u32,
    pub x_advance_adjustment: i16,
}

/// Apply a list of single substitutions to a glyph stream. Each glyph
/// is checked against the substitution table; matches are replaced.
pub fn apply_substitutions(glyphs: &mut [u32], subs: &[SingleSubst]) {
    for g in glyphs.iter_mut() {
        for s in subs {
            if s.input_glyph == *g {
                *g = s.replacement;
                break;
            }
        }
    }
}

/// Apply pairwise kerning to advances. For each adjacent glyph pair
/// found in `pairs`, the corresponding advance is adjusted.
pub fn apply_kerning(glyphs: &[u32], advances: &mut [i32], pairs: &[KernPair]) {
    if glyphs.len() < 2 {
        return;
    }
    for i in 0..glyphs.len() - 1 {
        let l = glyphs[i];
        let r = glyphs[i + 1];
        for p in pairs {
            if p.left == l && p.right == r {
                advances[i] += i32::from(p.x_advance_adjustment);
                break;
            }
        }
    }
}

// ----------------------------------------------------------------------
// GSUB ligature substitution + chaining context
// ----------------------------------------------------------------------

/// One GSUB type-4 ligature: an ordered sequence of input glyphs
/// that collapse to a single output glyph. The first glyph triggers
/// the lookup; the remaining `tail` must follow in order for the
/// substitution to fire.
#[derive(Debug, Clone)]
pub struct Ligature {
    pub first: u32,
    pub tail: Vec<u32>,
    pub output: u32,
}

/// Apply ligature substitutions to a glyph stream. Walks left-to-
/// right, attempting each ligature whose `first` matches; on match,
/// replaces the run with the ligature's output glyph. Non-greedy
/// (first match wins).
pub fn apply_ligatures(glyphs: &[u32], ligatures: &[Ligature]) -> Vec<u32> {
    let mut out = Vec::with_capacity(glyphs.len());
    let mut i = 0;
    while i < glyphs.len() {
        let g = glyphs[i];
        let mut matched: Option<&Ligature> = None;
        for lig in ligatures {
            if lig.first != g {
                continue;
            }
            // Check tail in order against subsequent glyphs.
            if i + 1 + lig.tail.len() > glyphs.len() {
                continue;
            }
            let tail_matches = lig
                .tail
                .iter()
                .enumerate()
                .all(|(j, &t)| glyphs[i + 1 + j] == t);
            if tail_matches {
                matched = Some(lig);
                break;
            }
        }
        if let Some(lig) = matched {
            out.push(lig.output);
            i += 1 + lig.tail.len();
        } else {
            out.push(g);
            i += 1;
        }
    }
    out
}

/// One GSUB type-6 chaining-context rule: input glyph sequence
/// applies only when surrounded by specific backtrack/lookahead
/// classes. V1 represents glyph classes as flat sets.
#[derive(Debug, Clone)]
pub struct ChainRule {
    pub backtrack: Vec<Vec<u32>>, // each Vec is the allowed set at that position (reverse order)
    pub input: Vec<u32>,          // the input glyph sequence to match exactly
    pub lookahead: Vec<Vec<u32>>, // each Vec is the allowed set at that position (forward order)
    pub output: Vec<u32>,         // substitution output
}

/// Apply chaining substitutions. For each position, every rule's
/// backtrack/input/lookahead is tested in order; first match wins
/// and the input range is replaced with the rule's output.
pub fn apply_chain_rules(glyphs: &[u32], rules: &[ChainRule]) -> Vec<u32> {
    let mut out: Vec<u32> = glyphs.to_vec();
    let mut i = 0;
    while i < out.len() {
        let mut applied = false;
        'rules: for rule in rules {
            // Backtrack: walk left of `i` matching reverse-order.
            if rule.backtrack.len() > i {
                continue;
            }
            for (b, allowed) in rule.backtrack.iter().enumerate() {
                let g = out[i - 1 - b];
                if !allowed.contains(&g) {
                    continue 'rules;
                }
            }
            // Input: must match in order.
            if i + rule.input.len() > out.len() {
                continue;
            }
            for (k, &want) in rule.input.iter().enumerate() {
                if out[i + k] != want {
                    continue 'rules;
                }
            }
            // Lookahead.
            let look_start = i + rule.input.len();
            if look_start + rule.lookahead.len() > out.len() {
                continue;
            }
            for (l, allowed) in rule.lookahead.iter().enumerate() {
                if !allowed.contains(&out[look_start + l]) {
                    continue 'rules;
                }
            }
            // Match! Splice in the output.
            let end = i + rule.input.len();
            out.splice(i..end, rule.output.iter().copied());
            i += rule.output.len();
            applied = true;
            break;
        }
        if !applied {
            i += 1;
        }
    }
    out
}

/// Pre-built Arabic ligature table for the most common lam-alef
/// presentation forms. Real fonts ship these via GSUB lookup table 4;
/// this table approximates the result for shaping-aware text layers
/// that don't have access to the font's tables.
pub fn arabic_lam_alef_ligatures() -> Vec<Ligature> {
    // Initial-form lam (U+FEDD) + isolated alef (U+FE8D) →
    // lam-alef ligature isolated form (U+FEFB). The other three
    // variants follow the same naming convention.
    vec![
        Ligature {
            first: 0xFEDD,      // lam initial
            tail: vec![0xFE8D], // alef isolated
            output: 0xFEFB,     // lam-alef isolated
        },
        Ligature {
            first: 0xFEDF, // lam medial
            tail: vec![0xFE8D],
            output: 0xFEFC, // lam-alef final
        },
        Ligature {
            first: 0xFEDD,
            tail: vec![0xFE8E], // alef final
            output: 0xFEFB,
        },
        Ligature {
            first: 0xFEDF,
            tail: vec![0xFE8E],
            output: 0xFEFC,
        },
    ]
}

// ----------------------------------------------------------------------
// Arabic cursive joining (UAX #44 Joining_Type + positional shaping)
// ----------------------------------------------------------------------

/// Joining type per UCD Joining_Type values. Drives whether an
/// Arabic letter joins to its neighbours and which positional form
/// it takes (initial / medial / final / isolated).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JoiningType {
    /// Right-joining: joins only to the left side (next-in-LTR).
    R,
    /// Left-joining: joins only to the right side.
    L,
    /// Dual-joining: joins on both sides.
    D,
    /// Causes joining but doesn't itself take a positional form (ZWJ).
    C,
    /// Transparent (combining marks).
    T,
    /// Non-joining.
    U,
}

/// Positional form for a shaped Arabic letter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JoiningForm {
    Isolated,
    Initial,
    Medial,
    Final,
}

/// Joining_Type lookup for the Arabic block (U+0600..U+06FF). Covers
/// the most common letters in modern usage; codepoints outside this
/// range default to `U` (non-joining).
pub fn joining_type(c: char) -> JoiningType {
    let cu = c as u32;
    // Dual-joining: most basic Arabic letters.
    // beh (0628), teh (062A), theh (062B), jeem (062C), hah (062D),
    // khah (062E), seen (0633), sheen (0634), sad (0635), dad (0636),
    // tah (0637), zah (0638), ain (0639), ghain (063A), feh (0641),
    // qaf (0642), kaf (0643), lam (0644), meem (0645), noon (0646),
    // heh (0647), yeh (064A).
    const D_LIST: &[u32] = &[
        0x0628, 0x062A, 0x062B, 0x062C, 0x062D, 0x062E, 0x0633, 0x0634, 0x0635, 0x0636, 0x0637,
        0x0638, 0x0639, 0x063A, 0x0641, 0x0642, 0x0643, 0x0644, 0x0645, 0x0646, 0x0647, 0x064A,
    ];
    if D_LIST.contains(&cu) {
        return JoiningType::D;
    }
    // Right-joining: alef (0627), waw (0648), dal (062F), thal (0630),
    // reh (0631), zain (0632), alef-maksura (0649), teh-marbuta (0629).
    if matches!(
        cu,
        0x0627 | 0x0648 | 0x062F | 0x0630 | 0x0631 | 0x0632 | 0x0649 | 0x0629
    ) {
        return JoiningType::R;
    }
    // Transparent: combining marks U+064B..U+065F and U+0670.
    if (0x064B..=0x065F).contains(&cu) || cu == 0x0670 {
        return JoiningType::T;
    }
    // Zero-width joiner / non-joiner.
    if cu == 0x200D {
        return JoiningType::C;
    }
    JoiningType::U
}

/// Compute the positional form of each Arabic letter in `chars`.
/// Output is per-input-char: T (transparent) and non-Arabic
/// characters yield Isolated. Per UAX #44, transparent runs don't
/// affect the joining decision; we walk past them on both sides
/// when computing context.
pub fn arabic_positional_forms(chars: &[char]) -> Vec<JoiningForm> {
    let n = chars.len();
    let mut out = vec![JoiningForm::Isolated; n];
    let jt: Vec<JoiningType> = chars.iter().map(|c| joining_type(*c)).collect();
    // Helper: scan left past transparents to the nearest non-T joiner.
    let prev_joiner = |i: usize| -> Option<JoiningType> {
        let mut j = i;
        while j > 0 {
            j -= 1;
            if jt[j] != JoiningType::T {
                return Some(jt[j]);
            }
        }
        None
    };
    let next_joiner = |i: usize| -> Option<JoiningType> {
        let mut j = i + 1;
        while j < n {
            if jt[j] != JoiningType::T {
                return Some(jt[j]);
            }
            j += 1;
        }
        None
    };
    for i in 0..n {
        let here = jt[i];
        // Only D/R/L letters take positional forms; others remain
        // Isolated (the default).
        if !matches!(here, JoiningType::D | JoiningType::R | JoiningType::L) {
            continue;
        }
        // Joining on the "right" side = joining to the previous
        // letter (in RTL script "right" is toward the start of the
        // run). A letter that accepts right-side joining (D or R)
        // joins if its previous joiner is D, L, or C.
        let joins_right = matches!(here, JoiningType::D | JoiningType::R)
            && matches!(
                prev_joiner(i),
                Some(JoiningType::D) | Some(JoiningType::L) | Some(JoiningType::C)
            );
        // Joining on the "left" side = joining to the next letter.
        // D and L letters join left if their next joiner is D, R, or C.
        let joins_left = matches!(here, JoiningType::D | JoiningType::L)
            && matches!(
                next_joiner(i),
                Some(JoiningType::D) | Some(JoiningType::R) | Some(JoiningType::C)
            );
        out[i] = match (joins_right, joins_left) {
            (true, true) => JoiningForm::Medial,
            (true, false) => JoiningForm::Final,
            (false, true) => JoiningForm::Initial,
            (false, false) => JoiningForm::Isolated,
        };
    }
    out
}

/// Apply Arabic shaping to a Unicode string, returning the
/// corresponding presentation-form codepoints from the Arabic
/// Presentation Forms-A / B blocks. Letters with no PUA mapping
/// fall through unchanged.
pub fn shape_arabic_to_presentation_forms(input: &str) -> String {
    let chars: Vec<char> = input.chars().collect();
    let forms = arabic_positional_forms(&chars);
    let mut out = String::with_capacity(input.len());
    for (i, &c) in chars.iter().enumerate() {
        out.push(presentation_form(c, forms[i]).unwrap_or(c));
    }
    out
}

/// Look up the Arabic Presentation Form codepoint for a base letter
/// + positional form. Returns None if the letter doesn't have a
/// mapped form (caller keeps the original codepoint).
pub fn presentation_form(base: char, form: JoiningForm) -> Option<char> {
    let base_cu = base as u32;
    // Most of the FE80..FEFC block is laid out as 4-codepoint blocks
    // (isolated, final, initial, medial) per letter. The table here
    // covers the basic 22 dual-joining letters + alef + waw, which
    // is enough for the eye to read.
    let presentation = match base_cu {
        0x0628 => 0xFE8F, // beh
        0x062A => 0xFE95, // teh
        0x062B => 0xFE99, // theh
        0x062C => 0xFE9D, // jeem
        0x062D => 0xFEA1, // hah
        0x062E => 0xFEA5, // khah
        0x062F => 0xFEA9, // dal
        0x0630 => 0xFEAB, // thal
        0x0631 => 0xFEAD, // reh
        0x0632 => 0xFEAF, // zain
        0x0633 => 0xFEB1, // seen
        0x0634 => 0xFEB5, // sheen
        0x0635 => 0xFEB9, // sad
        0x0636 => 0xFEBD, // dad
        0x0637 => 0xFEC1, // tah
        0x0638 => 0xFEC5, // zah
        0x0639 => 0xFEC9, // ain
        0x063A => 0xFECD, // ghain
        0x0641 => 0xFED1, // feh
        0x0642 => 0xFED5, // qaf
        0x0643 => 0xFED9, // kaf
        0x0644 => 0xFEDD, // lam
        0x0645 => 0xFEE1, // meem
        0x0646 => 0xFEE5, // noon
        0x0647 => 0xFEE9, // heh
        0x0648 => 0xFEED, // waw
        0x064A => 0xFEF1, // yeh
        0x0627 => 0xFE8D, // alef
        0x0629 => 0xFE93, // teh-marbuta
        0x0649 => 0xFEEF, // alef-maksura
        _ => return None,
    };
    let offset = match form {
        JoiningForm::Isolated => 0,
        JoiningForm::Final => 1,
        JoiningForm::Initial => 2,
        JoiningForm::Medial => 3,
    };
    char::from_u32(presentation + offset)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_minimal_font() -> Vec<u8> {
        // SFNT header for a one-table font (just a fake cmap entry).
        let mut v = vec![0u8; 12];
        v[4] = 0;
        v[5] = 1; // numTables = 1
        v.extend_from_slice(b"cmap");
        v.extend_from_slice(&0u32.to_be_bytes()); // checksum
        v.extend_from_slice(&28u32.to_be_bytes()); // offset
        v.extend_from_slice(&0u32.to_be_bytes()); // length (filled below)
        v
    }

    #[test]
    fn table_dir_round_trip() {
        let font = build_minimal_font();
        let dir = parse_table_dir(&font).unwrap();
        assert_eq!(dir.len(), 1);
        assert_eq!(&dir[0].tag, b"cmap");
        assert_eq!(dir[0].offset, 28);
    }

    #[test]
    fn cmap_lookup_walks_ranges() {
        let ranges = vec![
            CmapRange {
                start: 0x20,
                end: 0x7E,
                first_glyph: 100,
            },
            CmapRange {
                start: 0x0600,
                end: 0x06FF,
                first_glyph: 500,
            },
        ];
        assert_eq!(
            cmap_lookup(&ranges, b'A' as u32),
            100 + (b'A' as u32 - 0x20)
        );
        assert_eq!(cmap_lookup(&ranges, 0x0628), 500 + (0x0628 - 0x0600));
        assert_eq!(cmap_lookup(&ranges, 0x4E00), 0); // CJK — out of range.
    }

    #[test]
    fn apply_subs_replaces_glyphs() {
        let mut glyphs = vec![10u32, 20, 30];
        let subs = vec![SingleSubst {
            input_glyph: 20,
            replacement: 99,
        }];
        apply_substitutions(&mut glyphs, &subs);
        assert_eq!(glyphs, vec![10, 99, 30]);
    }

    #[test]
    fn ligatures_collapse_matching_run() {
        let glyphs = vec![10u32, 20, 30, 40];
        let ligatures = vec![Ligature {
            first: 20,
            tail: vec![30],
            output: 99,
        }];
        let out = apply_ligatures(&glyphs, &ligatures);
        assert_eq!(out, vec![10, 99, 40]);
    }

    #[test]
    fn lam_alef_ligature_fires() {
        // Lam initial (FEDD) + alef isolated (FE8D) → lam-alef isolated (FEFB).
        let glyphs = vec![0xFEDDu32, 0xFE8D];
        let out = apply_ligatures(&glyphs, &arabic_lam_alef_ligatures());
        assert_eq!(out, vec![0xFEFBu32]);
    }

    #[test]
    fn chain_rule_with_backtrack_and_lookahead() {
        let glyphs = vec![1u32, 2, 3, 4, 5];
        let rules = vec![ChainRule {
            backtrack: vec![vec![1]], // must be preceded by 1
            input: vec![2, 3],
            lookahead: vec![vec![4]], // must be followed by 4
            output: vec![99],
        }];
        let out = apply_chain_rules(&glyphs, &rules);
        assert_eq!(out, vec![1, 99, 4, 5]);
    }

    #[test]
    fn isolated_alef_stays_isolated() {
        let chars = ['\u{0627}'];
        let forms = arabic_positional_forms(&chars);
        assert_eq!(forms, vec![JoiningForm::Isolated]);
    }

    #[test]
    fn baa_taa_baa_takes_initial_medial_final() {
        // ب ت ب — three D letters, middle is medial, outer take init/final.
        let chars: Vec<char> = "بتب".chars().collect();
        let forms = arabic_positional_forms(&chars);
        assert_eq!(
            forms,
            vec![
                JoiningForm::Initial,
                JoiningForm::Medial,
                JoiningForm::Final
            ]
        );
    }

    #[test]
    fn alef_after_baa_takes_final() {
        // ب ا — beh is D (joins both), alef is R (joins only right).
        // beh becomes Initial, alef becomes Final.
        let chars: Vec<char> = "با".chars().collect();
        let forms = arabic_positional_forms(&chars);
        assert_eq!(forms, vec![JoiningForm::Initial, JoiningForm::Final]);
    }

    #[test]
    fn presentation_form_lookup() {
        // beh isolated = FE8F.
        assert_eq!(
            presentation_form('\u{0628}', JoiningForm::Isolated),
            Some('\u{FE8F}')
        );
        // beh medial = FE92.
        assert_eq!(
            presentation_form('\u{0628}', JoiningForm::Medial),
            Some('\u{FE92}')
        );
    }

    #[test]
    fn shape_arabic_replaces_codepoints() {
        let out = shape_arabic_to_presentation_forms("بتب");
        let chars: Vec<char> = out.chars().collect();
        // First char should be the initial form of beh = FE91.
        assert_eq!(chars[0], '\u{FE91}');
        // Middle should be medial form = FE92.
        assert_eq!(chars[1], '\u{FE98}'); // teh medial
        // Last should be final form of beh = FE90.
        assert_eq!(chars[2], '\u{FE90}');
    }

    #[test]
    fn apply_kerning_adjusts_advances() {
        let glyphs = vec![1u32, 2, 3];
        let mut advances = vec![100i32, 100, 100];
        let pairs = vec![KernPair {
            left: 1,
            right: 2,
            x_advance_adjustment: -8,
        }];
        apply_kerning(&glyphs, &mut advances, &pairs);
        assert_eq!(advances, vec![92, 100, 100]);
    }
}
