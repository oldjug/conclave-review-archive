//! Indic complex-script shaping — Devanagari syllable analysis and the
//! character-level *initial reordering* that HarfBuzz / Uniscribe perform
//! before GSUB is applied.
//!
//! Indic scripts encode a syllable in phonetic (logical) order, but the
//! glyphs render in a different *visual* order. The two reorderings the
//! shaping engine performs at the character level (i.e. without needing
//! the font's GSUB tables) are:
//!
//!   * **Pre-base ("left") matra** — a dependent vowel such as the
//!     Devanagari I-matra U+093F is typed *after* the consonant it
//!     attaches to, but renders to the *left* of the whole consonant
//!     cluster. The shaper moves it to before the base consonant.
//!     (MS reorder class `BeforeHalf`; HarfBuzz `POS_PRE_M`.)
//!
//!   * **Reph** — when a syllable begins with `Ra + Halant` and has a
//!     following base consonant, that initial `Ra` is *not* the base;
//!     it becomes the above-base "reph" mark and is reordered to *after*
//!     the base consonant. (MS reorder class `BeforePostscript`;
//!     HarfBuzz `POS_RA_TO_BECOME_REPH`.)
//!
//! Sources:
//!   * Microsoft "Developing OpenType Fonts for Devanagari Script"
//!     <https://learn.microsoft.com/typography/script-development/devanagari>
//!     — "Reorder characters" §: find base (scan backwards), reorder
//!     pre-base matras to before the base, reph repositioning classes,
//!     and the Devanagari reorder-class table.
//!   * HarfBuzz `hb-ot-shaper-indic.cc` initial_reordering_consonant_syllable
//!     — `POS_PRE_M` left-matra move + `POS_RA_TO_BECOME_REPH`.
//!
//! Scope: this delivers the real Devanagari base-finding + I-matra
//! reorder + reph reorder over the Unicode 0900..097F block (the case
//! the tests exercise and the most common Devanagari shaping a browser
//! must get right). Conjunct ligature formation, below-base/post-base
//! consonant forms, split-matra decomposition and the other Indic
//! scripts (Bengali/Tamil/Telugu/…) reuse this same machinery via their
//! own class tables and are tracked as follow-ups — they are NOT faked
//! here: an unclassified codepoint passes through unchanged.

/// Syllabic category of a Devanagari character, mirroring the static
/// "Indic_Syllabic_Category" the shaping engine assigns. Only the
/// categories that participate in reordering are distinguished; anything
/// else is `Other` and is treated as an opaque cluster boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IndicCategory {
    /// A consonant (can be a base, half, below-base or post-base form).
    Consonant,
    /// The letter Ra specifically — needs its own category because
    /// `Ra + Halant` at syllable start becomes the reph.
    Ra,
    /// Halant / Virama (U+094D) — strips the inherent vowel and joins
    /// consonants into conjuncts.
    Halant,
    /// A pre-base (left) dependent vowel sign, e.g. I-matra U+093F.
    /// Renders to the left of the cluster → reordered before the base.
    MatraPre,
    /// An above / below / post-base dependent vowel sign. Stays after
    /// the base in logical order (no character-level move needed).
    MatraOther,
    /// Nukta (U+093C) — a dot that modifies the preceding consonant.
    Nukta,
    /// An independent vowel or any other Devanagari letter that begins
    /// its own syllable.
    Vowel,
    /// Not a Devanagari character we reorder.
    Other,
}

/// Classify a Devanagari codepoint into its reordering-relevant
/// category. Codepoints outside 0900..097F return `Other`.
///
/// The matra split (pre-base vs other) follows the Microsoft Devanagari
/// reorder-class table: only U+093F (I-matra) is `BeforeHalf` (pre-base);
/// 093E, 0940-094C are `AfterSubscript` (above/below/post-base).
pub fn indic_category(c: char) -> IndicCategory {
    let cu = c as u32;
    match cu {
        // Ra — special-cased for reph detection.
        0x0930 => IndicCategory::Ra,
        // Consonants Ka..Ha (excluding Ra which is above), plus the
        // nukta-composed consonants 0958..095F and the additional
        // consonants 0978..097F. 0929/0931/0934 are nukta variants of
        // consonants and also behave as consonants.
        0x0915..=0x0939 | 0x0958..=0x095F | 0x0978..=0x097F => IndicCategory::Consonant,
        // Halant / Virama.
        0x094D => IndicCategory::Halant,
        // Nukta.
        0x093C => IndicCategory::Nukta,
        // Pre-base (left) matra: I-matra.
        0x093F => IndicCategory::MatraPre,
        // Other dependent vowel signs (above / below / post-base):
        // AA(093E), II(0940), U(0941), UU(0942), R(0943), RR(0944),
        // E(0945), AI candra(0946-0948), O(0949-094C), and 0955-0957,
        // plus vocalic 0962/0963.
        0x093E
        | 0x0940..=0x094C
        | 0x0955..=0x0957
        | 0x0962
        | 0x0963 => IndicCategory::MatraOther,
        // Independent vowels A..AU and vocalic forms, candrabindu /
        // anusvara / visarga begin or modify a syllable but are not
        // reordered by the character-level pass we implement.
        0x0900..=0x0903 | 0x0904..=0x0914 | 0x0960 | 0x0961 => IndicCategory::Vowel,
        _ => IndicCategory::Other,
    }
}

/// True if `c` is any Devanagari codepoint in the 0900..097F block.
pub fn is_devanagari(c: char) -> bool {
    (0x0900..=0x097F).contains(&(c as u32))
}

/// Split a Devanagari character run into syllable clusters, returning a
/// `Vec` of `(start, end)` index pairs over `chars`.
///
/// A syllable is the standard Indic regex
/// `{C+H}* C M* | V M*` simplified to: a run of consonants joined by
/// halants, optionally followed by dependent vowel signs / nukta /
/// modifiers; or an independent vowel followed by signs. The boundary
/// is the start of the next consonant/vowel that is *not* preceded by a
/// halant.
pub fn devanagari_syllables(chars: &[IndicCategory]) -> Vec<(usize, usize)> {
    let n = chars.len();
    let mut out = Vec::new();
    let mut i = 0;
    while i < n {
        let start = i;
        // Walk forward. A new syllable begins at a consonant/vowel that
        // is NOT immediately preceded by a halant (a halant glues the
        // next consonant into the current cluster as a conjunct).
        i += 1;
        while i < n {
            let here = chars[i];
            let prev = chars[i - 1];
            let starts_new = matches!(here, IndicCategory::Consonant | IndicCategory::Ra | IndicCategory::Vowel)
                && prev != IndicCategory::Halant;
            if starts_new {
                break;
            }
            i += 1;
        }
        out.push((start, i));
    }
    out
}

/// Perform the character-level *initial reordering* on one Devanagari
/// syllable in place. `cats` is the per-char category and `cps` the
/// codepoints; both slices cover exactly the syllable.
///
/// Returns the reordered codepoints. The two moves performed:
///   1. **Find base.** Scan backwards from the end to the first
///      consonant that is not preceded (at syllable start) by the
///      reph-forming `Ra + Halant`. (We don't have the font's
///      below/post-base classification, so — matching HarfBuzz's
///      default — the *last* consonant of the cluster is the base.)
///   2. **Reph.** If the syllable starts with `Ra + Halant` and has
///      another consonant after it, drop the `Ra + Halant` from the
///      front and re-insert the single `Ra` right after the base
///      consonant (the reph renders above-base, after the base).
///   3. **Pre-base matra.** Any `MatraPre` (I-matra) is moved to
///      immediately before the base consonant (it renders to the left).
fn reorder_devanagari_syllable(cats: &[IndicCategory], cps: &[u32]) -> Vec<u32> {
    let n = cps.len();
    if n < 2 {
        return cps.to_vec();
    }

    // --- Reph detection: syllable starts with Ra + Halant and has a
    // following consonant that can be the base. ---
    let has_reph = cats.len() >= 3
        && cats[0] == IndicCategory::Ra
        && cats[1] == IndicCategory::Halant
        && cats[2..]
            .iter()
            .any(|c| matches!(c, IndicCategory::Consonant | IndicCategory::Ra));

    // Working sequence excluding a leading reph (so base-finding and the
    // matra move operate on the post-reph remainder).
    let body_start = if has_reph { 2 } else { 0 };

    // --- Find base consonant: last consonant in the (post-reph) body.
    // Per MS algo we scan backwards to the first consonant that isn't a
    // below/post-base form; without font tables that is simply the last
    // consonant. ---
    let mut base_rel: Option<usize> = None;
    for (rel, cat) in cats[body_start..].iter().enumerate() {
        if matches!(cat, IndicCategory::Consonant | IndicCategory::Ra) {
            base_rel = Some(rel);
        }
    }

    // Build the reordered output over the body, then prepend nothing
    // (reph is folded in after the base).
    let body_cats = &cats[body_start..];
    let body_cps = &cps[body_start..];

    // Collect indices in their new visual order.
    // 1. pre-base matras (moved to front of the body, before base)
    // 2. everything from body start up to & including base, minus pre-matras
    // 3. the reph Ra (if any), inserted right after the base
    // 4. everything after the base (other matras, signs)
    let base = match base_rel {
        Some(b) => b,
        None => return cps.to_vec(), // no consonant base — leave as is.
    };

    let mut pre_matras: Vec<u32> = Vec::new();
    let mut head: Vec<u32> = Vec::new(); // up to and including base, minus pre-matras
    let mut tail: Vec<u32> = Vec::new(); // after base

    for (rel, &cp) in body_cps.iter().enumerate() {
        match body_cats[rel] {
            IndicCategory::MatraPre => pre_matras.push(cp),
            _ if rel <= base => head.push(cp),
            _ => tail.push(cp),
        }
    }

    // Assemble: [pre-base matras] [head incl. base] [reph Ra] [tail]
    let mut out = Vec::with_capacity(n);
    out.extend_from_slice(&pre_matras);
    out.extend_from_slice(&head);
    if has_reph {
        // The reph is the leading Ra (cps[0]); the halant (cps[1]) is
        // consumed and not rendered as a separate glyph.
        out.push(cps[0]);
    }
    out.extend_from_slice(&tail);
    out
}

/// Reorder a whole Devanagari string at the character level (initial
/// reordering). Non-Devanagari characters pass through unchanged; the
/// string is split into syllables and each is reordered independently.
///
/// This is the input a font's GSUB stage would then shape; for our
/// GDI-codepoint text path it is also directly renderable, because the
/// I-matra and reph now sit in visual order.
pub fn reorder_devanagari(input: &str) -> String {
    let chars: Vec<char> = input.chars().collect();
    if !chars.iter().any(|&c| is_devanagari(c)) {
        return input.to_string();
    }
    let cps: Vec<u32> = chars.iter().map(|&c| c as u32).collect();
    let cats: Vec<IndicCategory> = chars.iter().map(|&c| indic_category(c)).collect();

    let mut out: Vec<u32> = Vec::with_capacity(cps.len());
    let mut i = 0;
    let n = cps.len();
    while i < n {
        if !is_devanagari(chars[i]) {
            out.push(cps[i]);
            i += 1;
            continue;
        }
        // Gather the maximal Devanagari run, then syllable-split it.
        let run_start = i;
        while i < n && is_devanagari(chars[i]) {
            i += 1;
        }
        let run_cats = &cats[run_start..i];
        let run_cps = &cps[run_start..i];
        for (s, e) in devanagari_syllables(run_cats) {
            let reordered = reorder_devanagari_syllable(&run_cats[s..e], &run_cps[s..e]);
            out.extend_from_slice(&reordered);
        }
    }
    out.iter().filter_map(|&c| char::from_u32(c)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classify_basic_devanagari() {
        assert_eq!(indic_category('\u{0915}'), IndicCategory::Consonant); // KA
        assert_eq!(indic_category('\u{0930}'), IndicCategory::Ra); // RA
        assert_eq!(indic_category('\u{094D}'), IndicCategory::Halant); // virama
        assert_eq!(indic_category('\u{093F}'), IndicCategory::MatraPre); // I-matra
        assert_eq!(indic_category('\u{093E}'), IndicCategory::MatraOther); // AA-matra
        assert_eq!(indic_category('\u{093C}'), IndicCategory::Nukta);
        assert_eq!(indic_category('A'), IndicCategory::Other);
    }

    #[test]
    fn i_matra_reorders_before_base() {
        // कि = KA (0915) + I-matra (093F). The I-matra is typed AFTER
        // the consonant but renders BEFORE it (to the left). After
        // initial reordering the codepoints must be [093F, 0915].
        let input = "\u{0915}\u{093F}";
        let out = reorder_devanagari(input);
        let cps: Vec<u32> = out.chars().map(|c| c as u32).collect();
        assert_eq!(cps, vec![0x093F, 0x0915], "I-matra must move before base KA");
        // And it actually changed the order (not a 1:1 passthrough).
        assert_ne!(out, input);
    }

    #[test]
    fn aa_matra_stays_after_base() {
        // का = KA + AA-matra (093E). AA renders to the RIGHT, so no
        // reorder: codepoints stay [0915, 093E].
        let input = "\u{0915}\u{093E}";
        let out = reorder_devanagari(input);
        let cps: Vec<u32> = out.chars().map(|c| c as u32).collect();
        assert_eq!(cps, vec![0x0915, 0x093E]);
    }

    #[test]
    fn reph_moves_after_base() {
        // र्क = RA (0930) + Halant (094D) + KA (0915). The leading
        // Ra+Halant becomes the reph and reorders to AFTER the base KA;
        // the halant is consumed. Result: [0915, 0930].
        let input = "\u{0930}\u{094D}\u{0915}";
        let out = reorder_devanagari(input);
        let cps: Vec<u32> = out.chars().map(|c| c as u32).collect();
        assert_eq!(cps, vec![0x0915, 0x0930], "reph Ra must move after base KA");
    }

    #[test]
    fn reph_with_i_matra() {
        // र्कि = RA + Halant + KA + I-matra. Reph→after base, I-matra→
        // before base: visual order [I-matra, KA, reph-Ra].
        let input = "\u{0930}\u{094D}\u{0915}\u{093F}";
        let out = reorder_devanagari(input);
        let cps: Vec<u32> = out.chars().map(|c| c as u32).collect();
        assert_eq!(cps, vec![0x093F, 0x0915, 0x0930]);
    }

    #[test]
    fn leading_ra_without_following_consonant_is_base_not_reph() {
        // र = lone RA is the base, not a reph. Unchanged.
        let input = "\u{0930}";
        assert_eq!(reorder_devanagari(input), input);
        // RA + I-matra: रि — Ra is the base (no second consonant), so the
        // I-matra reorders before it: [093F, 0930].
        let input2 = "\u{0930}\u{093F}";
        let cps: Vec<u32> = reorder_devanagari(input2).chars().map(|c| c as u32).collect();
        assert_eq!(cps, vec![0x093F, 0x0930]);
    }

    #[test]
    fn conjunct_keeps_halant_and_picks_last_consonant_as_base() {
        // क्ष = KA (0915) + Halant (094D) + SSA (0937) + I-matra (093F).
        // It is one syllable (halant glues KA to SSA). Base = SSA (last
        // consonant). I-matra moves before the whole cluster.
        // Expected: [093F, 0915, 094D, 0937].
        let input = "\u{0915}\u{094D}\u{0937}\u{093F}";
        let out = reorder_devanagari(input);
        let cps: Vec<u32> = out.chars().map(|c| c as u32).collect();
        assert_eq!(cps, vec![0x093F, 0x0915, 0x094D, 0x0937]);
    }

    #[test]
    fn syllable_split_separates_independent_clusters() {
        // किका = (KA + I-matra)(KA + AA-matra) → two syllables.
        // 0915 093F | 0915 093E  ->  093F 0915 | 0915 093E
        let input = "\u{0915}\u{093F}\u{0915}\u{093E}";
        let out = reorder_devanagari(input);
        let cps: Vec<u32> = out.chars().map(|c| c as u32).collect();
        assert_eq!(cps, vec![0x093F, 0x0915, 0x0915, 0x093E]);
    }

    #[test]
    fn non_devanagari_passes_through() {
        assert_eq!(reorder_devanagari("hello"), "hello");
        // Mixed: Latin then Devanagari syllable.
        let input = "a\u{0915}\u{093F}";
        let cps: Vec<u32> = reorder_devanagari(input).chars().map(|c| c as u32).collect();
        assert_eq!(cps, vec![b'a' as u32, 0x093F, 0x0915]);
    }
}
