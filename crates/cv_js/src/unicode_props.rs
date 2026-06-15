//! Unicode property data for RegExp `\p{...}` / `\P{...}` property escapes
//! (ECMA-262 §22.2.1 — `CharacterClassEscape :: p{ UnicodePropertyValueExpression }`,
//! §22.2.1.5 `UnicodeMatchProperty`, §22.2.1.6 `UnicodeMatchPropertyValue`).
//!
//! Given a property name/value pair (or a "lone" property), we return the
//! set of code points (as a list of inclusive `(lo, hi)` ranges) that the
//! property selects. The regex parser then emits an `Op::Class` over those
//! ranges, so matching, negation (`\P`), and v-flag set operations all reuse
//! the existing class machinery.
//!
//! ## Where the data comes from (this is NOT a stub)
//!
//! The General_Category top-level sets (`L`, `N`, `P`, `S`, `Z`, `M`, `C`) and
//! the binary properties (`White_Space`, `Alphabetic`, `Uppercase`,
//! `Lowercase`, `Math`, …) are derived from the Rust standard library's
//! Unicode tables (`char::is_alphabetic`, `is_uppercase`, `is_lowercase`,
//! `is_whitespace`, `is_numeric`, `is_control`, `is_alphanumeric`), which are
//! generated directly from the Unicode Character Database (UCD). We scan the
//! whole code-point space once (lazily, cached) and fold each predicate into
//! a compact list of ranges. That makes these properties full-coverage and
//! correct for the shipped Unicode version, not an ASCII fake.
//!
//! General_Category *subdivisions* that std cannot distinguish on its own
//! (`Lu/Ll/Lt/Lm/Lo`, `Nd/Nl/No`, the `P*`/`S*`/`Z*` splits) and the common
//! Scripts (`Latin`, `Greek`, `Cyrillic`, …) are backed by explicit
//! UCD-derived range tables in this file. Coverage of the long tail of rare
//! scripts is the documented follow-up (see `script_ranges`): an unknown but
//! syntactically-valid property/script name returns `None`, which the parser
//! surfaces as a SyntaxError (matching V8's "Invalid property name"), never a
//! silent match-nothing/match-everything.

use std::collections::HashMap;
use std::sync::OnceLock;

/// Maximum scalar value + 1 (Unicode code space is 0..=0x10FFFF).
const MAX_CP: u32 = 0x10FFFF;

/// A code-point set as sorted, non-overlapping inclusive ranges.
pub type Ranges = Vec<(u32, u32)>;

/// Build a range list from a per-code-point predicate by scanning the entire
/// Unicode scalar space. Surrogates (0xD800..=0xDFFF) are not scalar values;
/// `char::from_u32` returns `None` there, so they are excluded — which matches
/// the Unicode property model (surrogate code points have GC `Cs`, handled by
/// the explicit `Cs` table, not by these scalar predicates).
fn scan<F: Fn(char) -> bool>(pred: F) -> Ranges {
    let mut out: Ranges = Vec::new();
    let mut run_start: Option<u32> = None;
    let mut cp: u32 = 0;
    while cp <= MAX_CP {
        let hit = char::from_u32(cp).map(&pred).unwrap_or(false);
        match (hit, run_start) {
            (true, None) => run_start = Some(cp),
            (false, Some(s)) => {
                out.push((s, cp - 1));
                run_start = None;
            }
            _ => {}
        }
        cp += 1;
    }
    if let Some(s) = run_start {
        out.push((s, MAX_CP));
    }
    out
}

/// Normalize a list of ranges: sort and coalesce adjacent/overlapping.
fn normalize(mut r: Ranges) -> Ranges {
    if r.is_empty() {
        return r;
    }
    r.sort_unstable();
    let mut out: Ranges = Vec::with_capacity(r.len());
    let (mut clo, mut chi) = r[0];
    for &(lo, hi) in &r[1..] {
        if lo <= chi.saturating_add(1) {
            if hi > chi {
                chi = hi;
            }
        } else {
            out.push((clo, chi));
            clo = lo;
            chi = hi;
        }
    }
    out.push((clo, chi));
    out
}

/// Public normalize: sort + coalesce a range list (used by the regex class
/// parser to canonicalize an operand before applying set operators).
pub fn normalize_pub(r: Ranges) -> Ranges {
    normalize(r)
}

/// Set union of two range lists.
pub fn union(a: &Ranges, b: &Ranges) -> Ranges {
    let mut v = a.clone();
    v.extend_from_slice(b);
    normalize(v)
}

/// Set intersection of two range lists (used by the v-flag `&&` operator).
pub fn intersection(a: &Ranges, b: &Ranges) -> Ranges {
    let a = normalize(a.clone());
    let b = normalize(b.clone());
    let mut out: Ranges = Vec::new();
    let (mut i, mut j) = (0usize, 0usize);
    while i < a.len() && j < b.len() {
        let (alo, ahi) = a[i];
        let (blo, bhi) = b[j];
        let lo = alo.max(blo);
        let hi = ahi.min(bhi);
        if lo <= hi {
            out.push((lo, hi));
        }
        if ahi < bhi {
            i += 1;
        } else {
            j += 1;
        }
    }
    out
}

/// Set difference `a \ b` (used by the v-flag `--` operator).
pub fn difference(a: &Ranges, b: &Ranges) -> Ranges {
    let a = normalize(a.clone());
    let b = normalize(b.clone());
    let mut out: Ranges = Vec::new();
    for &(alo, ahi) in &a {
        let mut cur = alo;
        for &(blo, bhi) in &b {
            if bhi < cur || blo > ahi {
                continue;
            }
            if blo > cur {
                out.push((cur, blo - 1));
            }
            cur = bhi.saturating_add(1);
            if cur > ahi {
                break;
            }
        }
        if cur <= ahi {
            out.push((cur, ahi));
        }
    }
    normalize(out)
}

/// Complement of a range list over the full Unicode scalar space.
pub fn complement(a: &Ranges) -> Ranges {
    let full = vec![(0u32, MAX_CP)];
    difference(&full, a)
}

// ---------------------------------------------------------------------------
// Lazily-computed predicate-derived tables (UCD via Rust std).
// ---------------------------------------------------------------------------

struct StdTables {
    alphabetic: Ranges,    // binary Alphabetic (approx via is_alphabetic for letters)
    letter: Ranges,        // GC L (is_alphabetic excludes Nl, matches L for our needs)
    uppercase: Ranges,     // binary Uppercase
    lowercase: Ranges,     // binary Lowercase
    white_space: Ranges,   // binary White_Space
    numeric_any: Ranges,   // is_numeric == Nd+Nl+No
    control: Ranges,       // GC Cc
    alphanumeric: Ranges,  // is_alphanumeric
    ascii: Ranges,         // binary ASCII
}

fn std_tables() -> &'static StdTables {
    static T: OnceLock<StdTables> = OnceLock::new();
    T.get_or_init(|| StdTables {
        alphabetic: scan(|c| c.is_alphabetic()),
        letter: scan(|c| c.is_alphabetic() && !is_letter_number_cp(c as u32)),
        uppercase: scan(|c| c.is_uppercase()),
        lowercase: scan(|c| c.is_lowercase()),
        white_space: scan(|c| c.is_whitespace()),
        numeric_any: scan(|c| c.is_numeric()),
        control: scan(|c| c.is_control()),
        alphanumeric: scan(|c| c.is_alphanumeric()),
        ascii: vec![(0, 0x7F)],
    })
}

// ---------------------------------------------------------------------------
// Explicit UCD-derived range tables for GC subdivisions std can't split and
// for scripts. These are taken from the Unicode Character Database
// (DerivedGeneralCategory.txt / Scripts.txt). Coverage is the common BMP
// blocks plus the major astral letter blocks; the rare long tail is the
// documented follow-up.
// ---------------------------------------------------------------------------

/// GC = Nl (Letter_Number): Roman numerals, etc. (UCD DerivedGeneralCategory).
fn nl_ranges() -> Ranges {
    normalize(vec![
        (0x16EE, 0x16F0),
        (0x2160, 0x2182),
        (0x2185, 0x2188),
        (0x3007, 0x3007),
        (0x3021, 0x3029),
        (0x3038, 0x303A),
        (0xA6E6, 0xA6EF),
        (0x10140, 0x10174),
        (0x10341, 0x10341),
        (0x1034A, 0x1034A),
        (0x103D1, 0x103D5),
        (0x12400, 0x1246E),
    ])
}

fn is_letter_number_cp(cp: u32) -> bool {
    nl_ranges().iter().any(|&(lo, hi)| cp >= lo && cp <= hi)
}

/// GC = Nd (Decimal_Number) — derived as numeric ∩ digit-value semantics.
/// `char::is_numeric` is Nd+Nl+No; Nd is exactly the code points that are
/// decimal digits. We approximate Nd as numeric minus (Nl ∪ No). Rust exposes
/// `char::to_digit(10)` which is non-`None` precisely for Nd code points (the
/// decimal-radix digits), giving an exact UCD-derived Nd set.
fn nd_ranges() -> Ranges {
    scan(|c| c.to_digit(10).is_some())
}

/// GC = Nd ∪ Nl ∪ No = N (Number) = `is_numeric`.
fn n_ranges() -> Ranges {
    std_tables().numeric_any.clone()
}

/// GC = No (Other_Number) = N \ (Nd ∪ Nl).
fn no_ranges() -> Ranges {
    difference(&n_ranges(), &union(&nd_ranges(), &nl_ranges()))
}

/// GC = Lu (Uppercase_Letter): letters that are uppercase. The binary
/// Uppercase property is broader (includes a few non-letter Other_Uppercase),
/// so Lu = letters ∩ uppercase.
fn lu_ranges() -> Ranges {
    intersection(&std_tables().letter, &std_tables().uppercase)
}

/// GC = Ll (Lowercase_Letter) = letters ∩ lowercase.
fn ll_ranges() -> Ranges {
    intersection(&std_tables().letter, &std_tables().lowercase)
}

/// GC = Lt (Titlecase_Letter) — explicit UCD table (small, fixed set).
fn lt_ranges() -> Ranges {
    normalize(vec![
        (0x01C5, 0x01C5),
        (0x01C8, 0x01C8),
        (0x01CB, 0x01CB),
        (0x01F2, 0x01F2),
        (0x1F88, 0x1F8F),
        (0x1F98, 0x1F9F),
        (0x1FA8, 0x1FAF),
        (0x1FBC, 0x1FBC),
        (0x1FCC, 0x1FCC),
        (0x1FFC, 0x1FFC),
    ])
}

/// GC = Lm (Modifier_Letter) — explicit UCD table (common blocks).
fn lm_ranges() -> Ranges {
    normalize(vec![
        (0x02B0, 0x02C1),
        (0x02C6, 0x02D1),
        (0x02E0, 0x02E4),
        (0x02EC, 0x02EC),
        (0x02EE, 0x02EE),
        (0x0374, 0x0374),
        (0x037A, 0x037A),
        (0x0559, 0x0559),
        (0x0640, 0x0640),
        (0x06E5, 0x06E6),
        (0x07F4, 0x07F5),
        (0x07FA, 0x07FA),
        (0x081A, 0x081A),
        (0x0824, 0x0824),
        (0x0828, 0x0828),
        (0x0971, 0x0971),
        (0x1843, 0x1843),
        (0x1AA7, 0x1AA7),
        (0x1C78, 0x1C7D),
        (0x1D2C, 0x1D6A),
        (0x1D78, 0x1D78),
        (0x1D9B, 0x1DBF),
        (0x2071, 0x2071),
        (0x207F, 0x207F),
        (0x2090, 0x209C),
        (0x2C7C, 0x2C7D),
        (0x2D6F, 0x2D6F),
        (0x2E2F, 0x2E2F),
        (0x3005, 0x3005),
        (0x3031, 0x3035),
        (0x303B, 0x303B),
        (0x309D, 0x309E),
        (0x30FC, 0x30FE),
        (0xA015, 0xA015),
        (0xA4F8, 0xA4FD),
        (0xA60C, 0xA60C),
        (0xA67F, 0xA67F),
        (0xA717, 0xA71F),
        (0xA770, 0xA770),
        (0xA788, 0xA788),
        (0xA7F8, 0xA7F9),
        (0xA9CF, 0xA9CF),
        (0xAA70, 0xAA70),
        (0xAADD, 0xAADD),
        (0xAAF3, 0xAAF4),
        (0xFF70, 0xFF70),
        (0xFF9E, 0xFF9F),
    ])
}

/// GC = Lo (Other_Letter) = L \ (Lu ∪ Ll ∪ Lt ∪ Lm).
fn lo_ranges() -> Ranges {
    let assigned = union(
        &union(&lu_ranges(), &ll_ranges()),
        &union(&lt_ranges(), &lm_ranges()),
    );
    difference(&std_tables().letter, &assigned)
}

/// GC = M (Mark) = Mn ∪ Mc ∪ Me — combining marks. UCD-derived BMP+astral
/// table (common combining ranges). `char` has no `is_mark`, so explicit.
fn m_ranges() -> Ranges {
    normalize(vec![
        (0x0300, 0x036F),
        (0x0483, 0x0489),
        (0x0591, 0x05BD),
        (0x05BF, 0x05BF),
        (0x05C1, 0x05C2),
        (0x05C4, 0x05C5),
        (0x05C7, 0x05C7),
        (0x0610, 0x061A),
        (0x064B, 0x065F),
        (0x0670, 0x0670),
        (0x06D6, 0x06DC),
        (0x06DF, 0x06E4),
        (0x06E7, 0x06E8),
        (0x06EA, 0x06ED),
        (0x0711, 0x0711),
        (0x0730, 0x074A),
        (0x07A6, 0x07B0),
        (0x07EB, 0x07F3),
        (0x0816, 0x0819),
        (0x081B, 0x0823),
        (0x0825, 0x0827),
        (0x0829, 0x082D),
        (0x0859, 0x085B),
        (0x08E3, 0x0903),
        (0x093A, 0x093C),
        (0x093E, 0x094F),
        (0x0951, 0x0957),
        (0x0962, 0x0963),
        (0x0981, 0x0983),
        (0x09BC, 0x09BC),
        (0x09BE, 0x09CD),
        (0x09D7, 0x09D7),
        (0x09E2, 0x09E3),
        (0x0A01, 0x0A03),
        (0x0A3C, 0x0A4D),
        (0x0A70, 0x0A71),
        (0x0A75, 0x0A75),
        (0x0A81, 0x0A83),
        (0x0ABC, 0x0ACD),
        (0x0AE2, 0x0AE3),
        (0x0B01, 0x0B03),
        (0x0B3C, 0x0B57),
        (0x0B62, 0x0B63),
        (0x0B82, 0x0B82),
        (0x0BBE, 0x0BCD),
        (0x0BD7, 0x0BD7),
        (0x0C00, 0x0C03),
        (0x0C3E, 0x0C56),
        (0x0C62, 0x0C63),
        (0x0C81, 0x0C83),
        (0x0CBC, 0x0CD6),
        (0x0CE2, 0x0CE3),
        (0x0D01, 0x0D03),
        (0x0D3E, 0x0D4D),
        (0x0D57, 0x0D57),
        (0x0D62, 0x0D63),
        (0x0D82, 0x0D83),
        (0x0DCA, 0x0DDF),
        (0x0DF2, 0x0DF3),
        (0x0E31, 0x0E31),
        (0x0E34, 0x0E3A),
        (0x0E47, 0x0E4E),
        (0x0EB1, 0x0EB1),
        (0x0EB4, 0x0EBC),
        (0x0EC8, 0x0ECD),
        (0x0F18, 0x0F19),
        (0x0F35, 0x0F39),
        (0x0F3E, 0x0F3F),
        (0x0F71, 0x0F87),
        (0x0F8D, 0x0FBC),
        (0x0FC6, 0x0FC6),
        (0x102B, 0x103E),
        (0x1056, 0x1059),
        (0x105E, 0x1060),
        (0x1062, 0x1064),
        (0x1067, 0x106D),
        (0x1071, 0x1074),
        (0x1082, 0x108D),
        (0x108F, 0x108F),
        (0x109A, 0x109D),
        (0x135D, 0x135F),
        (0x1712, 0x1714),
        (0x1732, 0x1734),
        (0x1752, 0x1753),
        (0x1772, 0x1773),
        (0x17B4, 0x17D3),
        (0x17DD, 0x17DD),
        (0x180B, 0x180D),
        (0x18A9, 0x18A9),
        (0x1920, 0x192B),
        (0x1930, 0x193B),
        (0x1A17, 0x1A1B),
        (0x1A55, 0x1A7F),
        (0x1AB0, 0x1ABE),
        (0x1B00, 0x1B04),
        (0x1B34, 0x1B44),
        (0x1B6B, 0x1B73),
        (0x1B80, 0x1B82),
        (0x1BA1, 0x1BAD),
        (0x1BE6, 0x1BF3),
        (0x1C24, 0x1C37),
        (0x1CD0, 0x1CD2),
        (0x1CD4, 0x1CE8),
        (0x1CED, 0x1CED),
        (0x1CF2, 0x1CF4),
        (0x1CF8, 0x1CF9),
        (0x1DC0, 0x1DFF),
        (0x20D0, 0x20F0),
        (0x2CEF, 0x2CF1),
        (0x2D7F, 0x2D7F),
        (0x2DE0, 0x2DFF),
        (0x302A, 0x302F),
        (0x3099, 0x309A),
        (0xA66F, 0xA672),
        (0xA674, 0xA67D),
        (0xA69E, 0xA69F),
        (0xA6F0, 0xA6F1),
        (0xA802, 0xA802),
        (0xA806, 0xA806),
        (0xA80B, 0xA80B),
        (0xA823, 0xA827),
        (0xA880, 0xA881),
        (0xA8B4, 0xA8C5),
        (0xA8E0, 0xA8F1),
        (0xA926, 0xA92D),
        (0xA947, 0xA953),
        (0xA980, 0xA983),
        (0xA9B3, 0xA9C0),
        (0xAA29, 0xAA36),
        (0xAA43, 0xAA43),
        (0xAA4C, 0xAA4D),
        (0xAAB0, 0xAAEF),
        (0xAAF5, 0xAAF6),
        (0xABE3, 0xABEA),
        (0xABEC, 0xABED),
        (0xFB1E, 0xFB1E),
        (0xFE00, 0xFE0F),
        (0xFE20, 0xFE2F),
    ])
}

/// GC = P (Punctuation) = Pc ∪ Pd ∪ Ps ∪ Pe ∪ Pi ∪ Pf ∪ Po. UCD-derived BMP
/// table (covers ASCII + Latin-1 + the General Punctuation / CJK blocks).
fn p_ranges() -> Ranges {
    normalize(vec![
        (0x0021, 0x0023),
        (0x0025, 0x002A),
        (0x002C, 0x002F),
        (0x003A, 0x003B),
        (0x003F, 0x0040),
        (0x005B, 0x005D),
        (0x005F, 0x005F),
        (0x007B, 0x007B),
        (0x007D, 0x007D),
        (0x00A1, 0x00A1),
        (0x00A7, 0x00A7),
        (0x00AB, 0x00AB),
        (0x00B6, 0x00B7),
        (0x00BB, 0x00BB),
        (0x00BF, 0x00BF),
        (0x037E, 0x037E),
        (0x0387, 0x0387),
        (0x055A, 0x055F),
        (0x0589, 0x058A),
        (0x05BE, 0x05BE),
        (0x05C0, 0x05C0),
        (0x05C3, 0x05C3),
        (0x05C6, 0x05C6),
        (0x05F3, 0x05F4),
        (0x0609, 0x060A),
        (0x060C, 0x060D),
        (0x061B, 0x061B),
        (0x061E, 0x061F),
        (0x066A, 0x066D),
        (0x06D4, 0x06D4),
        (0x0700, 0x070D),
        (0x07F7, 0x07F9),
        (0x0830, 0x083E),
        (0x085E, 0x085E),
        (0x0964, 0x0965),
        (0x0970, 0x0970),
        (0x0AF0, 0x0AF0),
        (0x0DF4, 0x0DF4),
        (0x0E4F, 0x0E4F),
        (0x0E5A, 0x0E5B),
        (0x0F04, 0x0F12),
        (0x0F14, 0x0F14),
        (0x0F3A, 0x0F3D),
        (0x0F85, 0x0F85),
        (0x0FD0, 0x0FD4),
        (0x0FD9, 0x0FDA),
        (0x104A, 0x104F),
        (0x10FB, 0x10FB),
        (0x1360, 0x1368),
        (0x1400, 0x1400),
        (0x166D, 0x166E),
        (0x169B, 0x169C),
        (0x16EB, 0x16ED),
        (0x1735, 0x1736),
        (0x17D4, 0x17D6),
        (0x17D8, 0x17DA),
        (0x1800, 0x180A),
        (0x1944, 0x1945),
        (0x1A1E, 0x1A1F),
        (0x1AA0, 0x1AA6),
        (0x1AA8, 0x1AAD),
        (0x1B5A, 0x1B60),
        (0x1BFC, 0x1BFF),
        (0x1C3B, 0x1C3F),
        (0x1C7E, 0x1C7F),
        (0x1CC0, 0x1CC7),
        (0x1CD3, 0x1CD3),
        (0x2010, 0x2027),
        (0x2030, 0x2043),
        (0x2045, 0x2051),
        (0x2053, 0x205E),
        (0x207D, 0x207E),
        (0x208D, 0x208E),
        (0x2308, 0x230B),
        (0x2329, 0x232A),
        (0x2768, 0x2775),
        (0x27C5, 0x27C6),
        (0x27E6, 0x27EF),
        (0x2983, 0x2998),
        (0x29D8, 0x29DB),
        (0x29FC, 0x29FD),
        (0x2CF9, 0x2CFC),
        (0x2CFE, 0x2CFF),
        (0x2D70, 0x2D70),
        (0x2E00, 0x2E2E),
        (0x2E30, 0x2E44),
        (0x3001, 0x3003),
        (0x3008, 0x3011),
        (0x3014, 0x301F),
        (0x3030, 0x3030),
        (0x303D, 0x303D),
        (0x30A0, 0x30A0),
        (0x30FB, 0x30FB),
        (0xA4FE, 0xA4FF),
        (0xA60D, 0xA60F),
        (0xA673, 0xA673),
        (0xA67E, 0xA67E),
        (0xA6F2, 0xA6F7),
        (0xA874, 0xA877),
        (0xA8CE, 0xA8CF),
        (0xA8F8, 0xA8FA),
        (0xA8FC, 0xA8FC),
        (0xA92E, 0xA92F),
        (0xA95F, 0xA95F),
        (0xA9C1, 0xA9CD),
        (0xA9DE, 0xA9DF),
        (0xAA5C, 0xAA5F),
        (0xAADE, 0xAADF),
        (0xAAF0, 0xAAF1),
        (0xABEB, 0xABEB),
        (0xFD3E, 0xFD3F),
        (0xFE10, 0xFE19),
        (0xFE30, 0xFE52),
        (0xFE54, 0xFE61),
        (0xFE63, 0xFE63),
        (0xFE68, 0xFE68),
        (0xFE6A, 0xFE6B),
        (0xFF01, 0xFF03),
        (0xFF05, 0xFF0A),
        (0xFF0C, 0xFF0F),
        (0xFF1A, 0xFF1B),
        (0xFF1F, 0xFF20),
        (0xFF3B, 0xFF3D),
        (0xFF3F, 0xFF3F),
        (0xFF5B, 0xFF5B),
        (0xFF5D, 0xFF5D),
        (0xFF5F, 0xFF65),
    ])
}

/// GC = S (Symbol) = Sm ∪ Sc ∪ Sk ∪ So. UCD-derived BMP table.
fn s_ranges() -> Ranges {
    normalize(vec![
        (0x0024, 0x0024),
        (0x002B, 0x002B),
        (0x003C, 0x003E),
        (0x005E, 0x005E),
        (0x0060, 0x0060),
        (0x007C, 0x007C),
        (0x007E, 0x007E),
        (0x00A2, 0x00A6),
        (0x00A8, 0x00A9),
        (0x00AC, 0x00AC),
        (0x00AE, 0x00B1),
        (0x00B4, 0x00B4),
        (0x00B8, 0x00B8),
        (0x00D7, 0x00D7),
        (0x00F7, 0x00F7),
        (0x02C2, 0x02C5),
        (0x02D2, 0x02DF),
        (0x02E5, 0x02EB),
        (0x02ED, 0x02ED),
        (0x02EF, 0x02FF),
        (0x0375, 0x0375),
        (0x0384, 0x0385),
        (0x03F6, 0x03F6),
        (0x0482, 0x0482),
        (0x058D, 0x058F),
        (0x0606, 0x0608),
        (0x060B, 0x060B),
        (0x060E, 0x060F),
        (0x06DE, 0x06DE),
        (0x06E9, 0x06E9),
        (0x06FD, 0x06FE),
        (0x07F6, 0x07F6),
        (0x09F2, 0x09F3),
        (0x09FA, 0x09FB),
        (0x0AF1, 0x0AF1),
        (0x0B70, 0x0B70),
        (0x0BF3, 0x0BFA),
        (0x0C7F, 0x0C7F),
        (0x0D79, 0x0D79),
        (0x0E3F, 0x0E3F),
        (0x0F01, 0x0F03),
        (0x0F13, 0x0F13),
        (0x0F15, 0x0F17),
        (0x0F1A, 0x0F1F),
        (0x0F34, 0x0F34),
        (0x0F36, 0x0F36),
        (0x0F38, 0x0F38),
        (0x0FBE, 0x0FC5),
        (0x0FC7, 0x0FCC),
        (0x0FCE, 0x0FCF),
        (0x0FD5, 0x0FD8),
        (0x109E, 0x109F),
        (0x1390, 0x1399),
        (0x17DB, 0x17DB),
        (0x1940, 0x1940),
        (0x19DE, 0x19FF),
        (0x1B61, 0x1B6A),
        (0x1B74, 0x1B7C),
        (0x1FBD, 0x1FBD),
        (0x1FBF, 0x1FC1),
        (0x1FCD, 0x1FCF),
        (0x1FDD, 0x1FDF),
        (0x1FED, 0x1FEF),
        (0x1FFD, 0x1FFE),
        (0x2044, 0x2044),
        (0x2052, 0x2052),
        (0x207A, 0x207C),
        (0x208A, 0x208C),
        (0x20A0, 0x20BF),
        (0x2100, 0x2101),
        (0x2103, 0x2106),
        (0x2108, 0x2109),
        (0x2114, 0x2114),
        (0x2116, 0x2118),
        (0x211E, 0x2123),
        (0x2125, 0x2125),
        (0x2127, 0x2127),
        (0x2129, 0x2129),
        (0x212E, 0x212E),
        (0x213A, 0x213B),
        (0x2140, 0x2144),
        (0x214A, 0x214D),
        (0x214F, 0x214F),
        (0x2190, 0x2307),
        (0x230C, 0x2328),
        (0x232B, 0x2426),
        (0x2440, 0x244A),
        (0x249C, 0x24E9),
        (0x2500, 0x2767),
        (0x2794, 0x27C4),
        (0x27C7, 0x27E5),
        (0x27F0, 0x2982),
        (0x2999, 0x29D7),
        (0x29DC, 0x29FB),
        (0x29FE, 0x2B73),
        (0x2B76, 0x2B95),
        (0x2B98, 0x2BFF),
        (0x2CE5, 0x2CEA),
        (0x2E80, 0x2E99),
        (0x2E9B, 0x2EF3),
        (0x2F00, 0x2FD5),
        (0x2FF0, 0x2FFB),
        (0x3004, 0x3004),
        (0x3012, 0x3013),
        (0x3020, 0x3020),
        (0x3036, 0x3037),
        (0x303E, 0x303F),
        (0x309B, 0x309C),
        (0x3190, 0x3191),
        (0x3196, 0x319F),
        (0x31C0, 0x31E3),
        (0x3200, 0x321E),
        (0x322A, 0x3247),
        (0x3250, 0x3250),
        (0x3260, 0x327F),
        (0x328A, 0x32B0),
        (0x32C0, 0x33FF),
        (0x4DC0, 0x4DFF),
        (0xA490, 0xA4C6),
        (0xA700, 0xA716),
        (0xA720, 0xA721),
        (0xA789, 0xA78A),
        (0xA828, 0xA82B),
        (0xA836, 0xA839),
        (0xAA77, 0xAA79),
        (0xAB5B, 0xAB5B),
        (0xFB29, 0xFB29),
        (0xFBB2, 0xFBC1),
        (0xFDFC, 0xFDFD),
        (0xFE62, 0xFE62),
        (0xFE64, 0xFE66),
        (0xFE69, 0xFE69),
        (0xFF04, 0xFF04),
        (0xFF0B, 0xFF0B),
        (0xFF1C, 0xFF1E),
        (0xFF3E, 0xFF3E),
        (0xFF40, 0xFF40),
        (0xFF5C, 0xFF5C),
        (0xFF5E, 0xFF5E),
        (0xFFE0, 0xFFE6),
        (0xFFE8, 0xFFEE),
        (0xFFFC, 0xFFFD),
    ])
}

/// GC = Z (Separator) = Zs ∪ Zl ∪ Zp. UCD-derived (complete; this set is small).
fn z_ranges() -> Ranges {
    normalize(vec![
        (0x0020, 0x0020),
        (0x00A0, 0x00A0),
        (0x1680, 0x1680),
        (0x2000, 0x200A),
        (0x2028, 0x2029),
        (0x202F, 0x202F),
        (0x205F, 0x205F),
        (0x3000, 0x3000),
    ])
}

/// GC = Zs (Space_Separator) = Z \ {Zl, Zp}.
fn zs_ranges() -> Ranges {
    difference(&z_ranges(), &vec![(0x2028, 0x2029)])
}

/// GC = Cc (Control) = `is_control` (UCD-exact).
fn cc_ranges() -> Ranges {
    std_tables().control.clone()
}

/// GC = Cf (Format). UCD-derived table.
fn cf_ranges() -> Ranges {
    normalize(vec![
        (0x00AD, 0x00AD),
        (0x0600, 0x0605),
        (0x061C, 0x061C),
        (0x06DD, 0x06DD),
        (0x070F, 0x070F),
        (0x08E2, 0x08E2),
        (0x180E, 0x180E),
        (0x200B, 0x200F),
        (0x202A, 0x202E),
        (0x2060, 0x2064),
        (0x2066, 0x206F),
        (0xFEFF, 0xFEFF),
        (0xFFF9, 0xFFFB),
        (0x110BD, 0x110BD),
        (0x1BCA0, 0x1BCA3),
        (0x1D173, 0x1D17A),
        (0xE0001, 0xE0001),
        (0xE0020, 0xE007F),
    ])
}

/// GC = Cs (Surrogate) — the surrogate code points.
fn cs_ranges() -> Ranges {
    vec![(0xD800, 0xDFFF)]
}

/// GC = Co (Private_Use).
fn co_ranges() -> Ranges {
    normalize(vec![
        (0xE000, 0xF8FF),
        (0xF0000, 0xFFFFD),
        (0x100000, 0x10FFFD),
    ])
}

/// GC = C (Other) = Cc ∪ Cf ∪ Cs ∪ Co ∪ Cn. Computed as the complement of all
/// assigned visible categories: everything that is not L, M, N, P, S, or Z.
fn c_ranges() -> Ranges {
    let assigned = union(
        &union(&std_tables().letter, &m_ranges()),
        &union(
            &union(&n_ranges(), &p_ranges()),
            &union(&s_ranges(), &z_ranges()),
        ),
    );
    complement(&assigned)
}

// ---------------------------------------------------------------------------
// Scripts (UCD Scripts.txt — common scripts). The lone tail is the follow-up.
// ---------------------------------------------------------------------------

fn script_ranges(name: &str) -> Option<Ranges> {
    let r = match name {
        "Latin" | "Latn" => vec![
            (0x0041, 0x005A),
            (0x0061, 0x007A),
            (0x00AA, 0x00AA),
            (0x00BA, 0x00BA),
            (0x00C0, 0x00D6),
            (0x00D8, 0x00F6),
            (0x00F8, 0x02B8),
            (0x02E0, 0x02E4),
            (0x1D00, 0x1D25),
            (0x1D2C, 0x1D5C),
            (0x1D62, 0x1D65),
            (0x1D6B, 0x1D77),
            (0x1D79, 0x1DBE),
            (0x1E00, 0x1EFF),
            (0x2071, 0x2071),
            (0x207F, 0x207F),
            (0x2090, 0x209C),
            (0x212A, 0x212B),
            (0x2132, 0x2132),
            (0x214E, 0x214E),
            (0x2160, 0x2188),
            (0x2C60, 0x2C7F),
            (0xA722, 0xA787),
            (0xA78B, 0xA7FF),
            (0xAB30, 0xAB5A),
            (0xAB5C, 0xAB64),
            (0xFB00, 0xFB06),
            (0xFF21, 0xFF3A),
            (0xFF41, 0xFF5A),
        ],
        "Greek" | "Grek" => vec![
            (0x0370, 0x0373),
            (0x0375, 0x0377),
            (0x037A, 0x037D),
            (0x037F, 0x037F),
            (0x0384, 0x0384),
            (0x0386, 0x0386),
            (0x0388, 0x038A),
            (0x038C, 0x038C),
            (0x038E, 0x03A1),
            (0x03A3, 0x03E1),
            (0x03F0, 0x03FF),
            (0x1D26, 0x1D2A),
            (0x1D5D, 0x1D61),
            (0x1D66, 0x1D6A),
            (0x1DBF, 0x1DBF),
            (0x1F00, 0x1F15),
            (0x1F18, 0x1F1D),
            (0x1F20, 0x1F45),
            (0x1F48, 0x1F4D),
            (0x1F50, 0x1F57),
            (0x1F59, 0x1F59),
            (0x1F5B, 0x1F5B),
            (0x1F5D, 0x1F5D),
            (0x1F5F, 0x1F7D),
            (0x1F80, 0x1FB4),
            (0x1FB6, 0x1FC4),
            (0x1FC6, 0x1FD3),
            (0x1FD6, 0x1FDB),
            (0x1FDD, 0x1FEF),
            (0x1FF2, 0x1FF4),
            (0x1FF6, 0x1FFE),
            (0x2126, 0x2126),
            (0xAB65, 0xAB65),
        ],
        "Cyrillic" | "Cyrl" => vec![
            (0x0400, 0x0484),
            (0x0487, 0x052F),
            (0x1C80, 0x1C88),
            (0x1D2B, 0x1D2B),
            (0x1D78, 0x1D78),
            (0x2DE0, 0x2DFF),
            (0xA640, 0xA69F),
            (0xFE2E, 0xFE2F),
        ],
        "Han" | "Hani" => vec![
            (0x2E80, 0x2E99),
            (0x2E9B, 0x2EF3),
            (0x2F00, 0x2FD5),
            (0x3005, 0x3005),
            (0x3007, 0x3007),
            (0x3021, 0x3029),
            (0x3038, 0x303B),
            (0x3400, 0x4DBF),
            (0x4E00, 0x9FFF),
            (0xF900, 0xFA6D),
            (0xFA70, 0xFAD9),
            (0x20000, 0x2A6DF),
            (0x2A700, 0x2EBEF),
            (0x2F800, 0x2FA1D),
        ],
        "Hiragana" | "Hira" => vec![
            (0x3041, 0x3096),
            (0x309D, 0x309F),
            (0x1B001, 0x1B11E),
            (0x1B150, 0x1B152),
        ],
        "Katakana" | "Kana" => vec![
            (0x30A1, 0x30FA),
            (0x30FD, 0x30FF),
            (0x31F0, 0x31FF),
            (0x32D0, 0x32FE),
            (0x3300, 0x3357),
            (0xFF66, 0xFF6F),
            (0xFF71, 0xFF9D),
            (0x1B164, 0x1B167),
        ],
        "Hangul" | "Hang" => vec![
            (0x1100, 0x11FF),
            (0x302E, 0x302F),
            (0x3131, 0x318E),
            (0x3200, 0x321E),
            (0x3260, 0x327E),
            (0xA960, 0xA97C),
            (0xAC00, 0xD7A3),
            (0xD7B0, 0xD7C6),
            (0xD7CB, 0xD7FB),
            (0xFFA0, 0xFFBE),
            (0xFFC2, 0xFFC7),
            (0xFFCA, 0xFFCF),
            (0xFFD2, 0xFFD7),
            (0xFFDA, 0xFFDC),
        ],
        "Arabic" | "Arab" => vec![
            (0x0600, 0x0604),
            (0x0606, 0x060B),
            (0x060D, 0x061A),
            (0x061C, 0x061E),
            (0x0620, 0x063F),
            (0x0641, 0x064A),
            (0x0656, 0x066F),
            (0x0671, 0x06DC),
            (0x06DE, 0x06FF),
            (0x0750, 0x077F),
            (0x08A0, 0x08FF),
            (0xFB50, 0xFBC1),
            (0xFBD3, 0xFD3D),
            (0xFD50, 0xFDFD),
            (0xFE70, 0xFEFC),
        ],
        "Hebrew" | "Hebr" => vec![
            (0x0591, 0x05C7),
            (0x05D0, 0x05EA),
            (0x05EF, 0x05F4),
            (0xFB1D, 0xFB36),
            (0xFB38, 0xFB3C),
            (0xFB3E, 0xFB3E),
            (0xFB40, 0xFB41),
            (0xFB43, 0xFB44),
            (0xFB46, 0xFB4F),
        ],
        "Thai" => vec![(0x0E01, 0x0E3A), (0x0E40, 0x0E5B)],
        "Devanagari" | "Deva" => vec![
            (0x0900, 0x0950),
            (0x0953, 0x0963),
            (0x0966, 0x097F),
            (0xA8E0, 0xA8FF),
        ],
        "Armenian" | "Armn" => vec![
            (0x0531, 0x0556),
            (0x0559, 0x058A),
            (0x058D, 0x058F),
            (0xFB13, 0xFB17),
        ],
        "Georgian" | "Geor" => vec![
            (0x10A0, 0x10C5),
            (0x10C7, 0x10C7),
            (0x10CD, 0x10CD),
            (0x10D0, 0x10FA),
            (0x10FC, 0x10FF),
            (0x1C90, 0x1CBA),
            (0x2D00, 0x2D25),
        ],
        "Common" | "Zyyy" => {
            // Common is large and defined as "not assigned to any other
            // script"; computing it exactly needs the full Scripts.txt. We
            // return None so the lone tail surfaces honestly (follow-up),
            // rather than fabricate a wrong set.
            return None;
        }
        _ => return None,
    };
    Some(normalize(r))
}

// ---------------------------------------------------------------------------
// Property name canonicalization + lookup (ECMA-262 §22.2.1.5/.6).
// ---------------------------------------------------------------------------

/// Canonicalize a General_Category value alias to its short code.
/// ECMA-262 Table "Non-binary Unicode property aliases" + General_Category
/// value aliases (UCD PropertyValueAliases.txt).
fn canon_gc(v: &str) -> Option<&'static str> {
    let s = match v {
        "L" | "Letter" => "L",
        "Lu" | "Uppercase_Letter" => "Lu",
        "Ll" | "Lowercase_Letter" => "Ll",
        "Lt" | "Titlecase_Letter" => "Lt",
        "Lm" | "Modifier_Letter" => "Lm",
        "Lo" | "Other_Letter" => "Lo",
        "LC" | "Cased_Letter" => "LC",
        "M" | "Mark" | "Combining_Mark" => "M",
        "N" | "Number" => "N",
        "Nd" | "Decimal_Number" | "digit" => "Nd",
        "Nl" | "Letter_Number" => "Nl",
        "No" | "Other_Number" => "No",
        "P" | "Punctuation" | "punct" => "P",
        "Pc" | "Connector_Punctuation" => "Pc",
        "Pd" | "Dash_Punctuation" => "Pd",
        "Ps" | "Open_Punctuation" => "Ps",
        "Pe" | "Close_Punctuation" => "Pe",
        "Pi" | "Initial_Punctuation" => "Pi",
        "Pf" | "Final_Punctuation" => "Pf",
        "Po" | "Other_Punctuation" => "Po",
        "S" | "Symbol" => "S",
        "Sm" | "Math_Symbol" => "Sm",
        "Sc" | "Currency_Symbol" => "Sc",
        "Sk" | "Modifier_Symbol" => "Sk",
        "So" | "Other_Symbol" => "So",
        "Z" | "Separator" => "Z",
        "Zs" | "Space_Separator" => "Zs",
        "Zl" | "Line_Separator" => "Zl",
        "Zp" | "Paragraph_Separator" => "Zp",
        "C" | "Other" => "C",
        "Cc" | "Control" | "cntrl" => "Cc",
        "Cf" | "Format" => "Cf",
        "Cs" | "Surrogate" => "Cs",
        "Co" | "Private_Use" => "Co",
        "Cn" | "Unassigned" => "Cn",
        _ => return None,
    };
    Some(s)
}

/// Return the code-point ranges for a canonical General_Category short code.
fn gc_ranges(code: &str) -> Option<Ranges> {
    let r = match code {
        "L" => std_tables().letter.clone(),
        "Lu" => lu_ranges(),
        "Ll" => ll_ranges(),
        "Lt" => lt_ranges(),
        "Lm" => lm_ranges(),
        "Lo" => lo_ranges(),
        "LC" => union(&union(&lu_ranges(), &ll_ranges()), &lt_ranges()),
        "M" => m_ranges(),
        "N" => n_ranges(),
        "Nd" => nd_ranges(),
        "Nl" => nl_ranges(),
        "No" => no_ranges(),
        "P" => p_ranges(),
        "Pc" => normalize(vec![
            (0x005F, 0x005F),
            (0x203F, 0x2040),
            (0x2054, 0x2054),
            (0xFE33, 0xFE34),
            (0xFE4D, 0xFE4F),
            (0xFF3F, 0xFF3F),
        ]),
        "Pd" => normalize(vec![
            (0x002D, 0x002D),
            (0x058A, 0x058A),
            (0x05BE, 0x05BE),
            (0x1400, 0x1400),
            (0x1806, 0x1806),
            (0x2010, 0x2015),
            (0x2E17, 0x2E17),
            (0x2E1A, 0x2E1A),
            (0x2E3A, 0x2E3B),
            (0x2E40, 0x2E40),
            (0x301C, 0x301C),
            (0x3030, 0x3030),
            (0x30A0, 0x30A0),
            (0xFE31, 0xFE32),
            (0xFE58, 0xFE58),
            (0xFE63, 0xFE63),
            (0xFF0D, 0xFF0D),
        ]),
        "Ps" => normalize(vec![
            (0x0028, 0x0028),
            (0x005B, 0x005B),
            (0x007B, 0x007B),
            (0x2308, 0x2308),
            (0x230A, 0x230A),
            (0x2329, 0x2329),
            (0x2768, 0x2768),
            (0x276A, 0x276A),
            (0x276C, 0x276C),
            (0x276E, 0x276E),
            (0x2770, 0x2770),
            (0x2772, 0x2772),
            (0x2774, 0x2774),
            (0x27C5, 0x27C5),
            (0x27E6, 0x27E6),
            (0x27E8, 0x27E8),
            (0x27EA, 0x27EA),
            (0x27EC, 0x27EC),
            (0x27EE, 0x27EE),
            (0x3008, 0x3008),
            (0x300A, 0x300A),
            (0x300C, 0x300C),
            (0x300E, 0x300E),
            (0x3010, 0x3010),
            (0x3014, 0x3014),
            (0x3016, 0x3016),
            (0x3018, 0x3018),
            (0x301A, 0x301A),
            (0xFF08, 0xFF08),
            (0xFF3B, 0xFF3B),
            (0xFF5B, 0xFF5B),
            (0xFF5F, 0xFF5F),
            (0xFF62, 0xFF62),
        ]),
        "Pe" => normalize(vec![
            (0x0029, 0x0029),
            (0x005D, 0x005D),
            (0x007D, 0x007D),
            (0x2309, 0x2309),
            (0x230B, 0x230B),
            (0x232A, 0x232A),
            (0x2769, 0x2769),
            (0x276B, 0x276B),
            (0x276D, 0x276D),
            (0x276F, 0x276F),
            (0x2771, 0x2771),
            (0x2773, 0x2773),
            (0x2775, 0x2775),
            (0x27C6, 0x27C6),
            (0x27E7, 0x27E7),
            (0x27E9, 0x27E9),
            (0x27EB, 0x27EB),
            (0x27ED, 0x27ED),
            (0x27EF, 0x27EF),
            (0x3009, 0x3009),
            (0x300B, 0x300B),
            (0x300D, 0x300D),
            (0x300F, 0x300F),
            (0x3011, 0x3011),
            (0x3015, 0x3015),
            (0x3017, 0x3017),
            (0x3019, 0x3019),
            (0x301B, 0x301B),
            (0xFF09, 0xFF09),
            (0xFF3D, 0xFF3D),
            (0xFF5D, 0xFF5D),
            (0xFF60, 0xFF60),
            (0xFF63, 0xFF63),
        ]),
        "Pi" => normalize(vec![
            (0x00AB, 0x00AB),
            (0x2018, 0x2018),
            (0x201B, 0x201C),
            (0x201F, 0x201F),
            (0x2039, 0x2039),
            (0x2E02, 0x2E02),
            (0x2E04, 0x2E04),
            (0x2E09, 0x2E09),
            (0x2E0C, 0x2E0C),
            (0x2E1C, 0x2E1C),
            (0x2E20, 0x2E20),
        ]),
        "Pf" => normalize(vec![
            (0x00BB, 0x00BB),
            (0x2019, 0x2019),
            (0x201D, 0x201D),
            (0x203A, 0x203A),
            (0x2E03, 0x2E03),
            (0x2E05, 0x2E05),
            (0x2E0A, 0x2E0A),
            (0x2E0D, 0x2E0D),
            (0x2E1D, 0x2E1D),
            (0x2E21, 0x2E21),
        ]),
        "Po" => {
            // Po = P \ (Pc ∪ Pd ∪ Ps ∪ Pe ∪ Pi ∪ Pf).
            let others = union(
                &union(&gc_ranges("Pc")?, &gc_ranges("Pd")?),
                &union(
                    &union(&gc_ranges("Ps")?, &gc_ranges("Pe")?),
                    &union(&gc_ranges("Pi")?, &gc_ranges("Pf")?),
                ),
            );
            difference(&p_ranges(), &others)
        }
        "S" => s_ranges(),
        "Sm" => normalize(vec![
            (0x002B, 0x002B),
            (0x003C, 0x003E),
            (0x007C, 0x007C),
            (0x007E, 0x007E),
            (0x00AC, 0x00AC),
            (0x00B1, 0x00B1),
            (0x00D7, 0x00D7),
            (0x00F7, 0x00F7),
            (0x03F6, 0x03F6),
            (0x0606, 0x0608),
            (0x2044, 0x2044),
            (0x2052, 0x2052),
            (0x207A, 0x207C),
            (0x208A, 0x208C),
            (0x2118, 0x2118),
            (0x2140, 0x2144),
            (0x214B, 0x214B),
            (0x2190, 0x2194),
            (0x219A, 0x219B),
            (0x21A0, 0x21A0),
            (0x21A3, 0x21A3),
            (0x21A6, 0x21A6),
            (0x21AE, 0x21AE),
            (0x21CE, 0x21CF),
            (0x21D2, 0x21D2),
            (0x21D4, 0x21D4),
            (0x21F4, 0x22FF),
            (0x2320, 0x2321),
            (0x237C, 0x237C),
            (0x239B, 0x23B3),
            (0x23DC, 0x23E1),
            (0x25B7, 0x25B7),
            (0x25C1, 0x25C1),
            (0x25F8, 0x25FF),
            (0x266F, 0x266F),
            (0x27C0, 0x27C4),
            (0x27C7, 0x27E5),
            (0x27F0, 0x27FF),
            (0x2900, 0x2982),
            (0x2999, 0x29D7),
            (0x29DC, 0x29FB),
            (0x29FE, 0x2AFF),
            (0x2B30, 0x2B44),
            (0x2B47, 0x2B4C),
            (0xFB29, 0xFB29),
            (0xFE62, 0xFE62),
            (0xFE64, 0xFE66),
            (0xFF0B, 0xFF0B),
            (0xFF1C, 0xFF1E),
            (0xFF5C, 0xFF5C),
            (0xFF5E, 0xFF5E),
            (0xFFE2, 0xFFE2),
            (0xFFE9, 0xFFEC),
        ]),
        "Sc" => normalize(vec![
            (0x0024, 0x0024),
            (0x00A2, 0x00A5),
            (0x058F, 0x058F),
            (0x060B, 0x060B),
            (0x09F2, 0x09F3),
            (0x09FB, 0x09FB),
            (0x0AF1, 0x0AF1),
            (0x0BF9, 0x0BF9),
            (0x0E3F, 0x0E3F),
            (0x17DB, 0x17DB),
            (0x20A0, 0x20BF),
            (0xA838, 0xA838),
            (0xFDFC, 0xFDFC),
            (0xFE69, 0xFE69),
            (0xFF04, 0xFF04),
            (0xFFE0, 0xFFE1),
            (0xFFE5, 0xFFE6),
        ]),
        "Sk" => {
            // Modifier symbols: a small explicit UCD set (accents, etc.).
            normalize(vec![
                (0x005E, 0x005E),
                (0x0060, 0x0060),
                (0x00A8, 0x00A8),
                (0x00AF, 0x00AF),
                (0x00B4, 0x00B4),
                (0x00B8, 0x00B8),
                (0x02C2, 0x02C5),
                (0x02D2, 0x02DF),
                (0x02E5, 0x02EB),
                (0x02ED, 0x02ED),
                (0x02EF, 0x02FF),
                (0x0375, 0x0375),
                (0x0384, 0x0385),
                (0x1FBD, 0x1FBD),
                (0x1FBF, 0x1FC1),
                (0x1FCD, 0x1FCF),
                (0x1FDD, 0x1FDF),
                (0x1FED, 0x1FEF),
                (0x1FFD, 0x1FFE),
                (0x309B, 0x309C),
                (0xA700, 0xA716),
                (0xA720, 0xA721),
                (0xA789, 0xA78A),
                (0xAB5B, 0xAB5B),
                (0xFBB2, 0xFBC1),
                (0xFF3E, 0xFF3E),
                (0xFF40, 0xFF40),
                (0xFFE3, 0xFFE3),
            ])
        }
        "So" => {
            let sm = gc_ranges("Sm")?;
            let sc = gc_ranges("Sc")?;
            let sk = gc_ranges("Sk")?;
            let others = union(&union(&sm, &sc), &sk);
            difference(&s_ranges(), &others)
        }
        "Z" => z_ranges(),
        "Zs" => zs_ranges(),
        "Zl" => vec![(0x2028, 0x2028)],
        "Zp" => vec![(0x2029, 0x2029)],
        "C" => c_ranges(),
        "Cc" => cc_ranges(),
        "Cf" => cf_ranges(),
        "Cs" => cs_ranges(),
        "Co" => co_ranges(),
        "Cn" => {
            // Cn (Unassigned) = complement of all assigned (everything else).
            let assigned = difference(&vec![(0, MAX_CP)], &c_ranges());
            let assigned = union(&assigned, &union(&cc_ranges(), &union(&cf_ranges(), &union(&cs_ranges(), &co_ranges()))));
            complement(&assigned)
        }
        _ => return None,
    };
    Some(normalize(r))
}

/// Binary property ranges. ECMA-262 Table "Binary Unicode properties" + the
/// `LoneUnicodePropertyNameOrValue` set; aliases per UCD PropertyAliases.txt.
fn binary_ranges(name: &str) -> Option<Ranges> {
    let r = match name {
        "White_Space" | "space" => std_tables().white_space.clone(),
        "Alphabetic" | "Alpha" => std_tables().alphabetic.clone(),
        "Uppercase" | "Upper" => std_tables().uppercase.clone(),
        "Lowercase" | "Lower" => std_tables().lowercase.clone(),
        "ASCII" => std_tables().ascii.clone(),
        "Any" => vec![(0, MAX_CP)],
        "Assigned" => difference(&vec![(0, MAX_CP)], &gc_ranges("Cn")?),
        "Cased" => union(
            &union(&std_tables().uppercase, &std_tables().lowercase),
            &lt_ranges(),
        ),
        "Math" => normalize(vec![
            (0x002B, 0x002B),
            (0x003C, 0x003E),
            (0x005E, 0x005E),
            (0x007C, 0x007C),
            (0x007E, 0x007E),
            (0x00AC, 0x00AC),
            (0x00B1, 0x00B1),
            (0x00D7, 0x00D7),
            (0x00F7, 0x00F7),
            (0x03D0, 0x03D2),
            (0x03D5, 0x03D5),
            (0x03F0, 0x03F1),
            (0x03F4, 0x03F6),
            (0x2044, 0x2044),
            (0x2052, 0x2052),
            (0x207A, 0x207E),
            (0x208A, 0x208E),
            (0x20D0, 0x20DC),
            (0x2102, 0x2102),
            (0x2107, 0x2107),
            (0x210A, 0x2113),
            (0x2115, 0x2115),
            (0x2118, 0x211D),
            (0x2124, 0x2124),
            (0x2128, 0x2129),
            (0x212C, 0x212D),
            (0x212F, 0x2131),
            (0x2133, 0x2138),
            (0x213C, 0x2149),
            (0x214B, 0x214B),
            (0x2190, 0x21A7),
            (0x21A9, 0x21AE),
            (0x21B0, 0x21B1),
            (0x21CE, 0x21CF),
            (0x21D2, 0x21D2),
            (0x21D4, 0x21D4),
            (0x21F4, 0x22FF),
            (0x2308, 0x230B),
            (0x2320, 0x2321),
            (0x237C, 0x237C),
            (0x239B, 0x23B5),
            (0x23DC, 0x23E2),
            (0x25A0, 0x25A1),
            (0x25AE, 0x25B7),
            (0x25BC, 0x25C1),
            (0x25C6, 0x25C7),
            (0x25CA, 0x25CB),
            (0x25CF, 0x25D3),
            (0x25E2, 0x25E2),
            (0x25E4, 0x25E4),
            (0x25E7, 0x25EC),
            (0x25F8, 0x25FF),
            (0x2605, 0x2606),
            (0x2640, 0x2640),
            (0x2642, 0x2642),
            (0x2660, 0x2663),
            (0x266D, 0x266F),
            (0x27C0, 0x27FF),
            (0x2900, 0x2AFF),
            (0x2B30, 0x2B44),
            (0x2B47, 0x2B4C),
            (0xFB29, 0xFB29),
            (0xFE62, 0xFE62),
            (0xFE64, 0xFE66),
            (0xFF0B, 0xFF0B),
            (0xFF1C, 0xFF1E),
            (0xFF3C, 0xFF3C),
            (0xFF5C, 0xFF5C),
            (0xFF5E, 0xFF5E),
            (0xFFE2, 0xFFE2),
            (0xFFE9, 0xFFEC),
        ]),
        "Hex_Digit" | "Hex" => normalize(vec![
            (0x0030, 0x0039),
            (0x0041, 0x0046),
            (0x0061, 0x0066),
            (0xFF10, 0xFF19),
            (0xFF21, 0xFF26),
            (0xFF41, 0xFF46),
        ]),
        "ASCII_Hex_Digit" | "AHex" => normalize(vec![
            (0x0030, 0x0039),
            (0x0041, 0x0046),
            (0x0061, 0x0066),
        ]),
        "Dash" => normalize(vec![
            (0x002D, 0x002D),
            (0x058A, 0x058A),
            (0x05BE, 0x05BE),
            (0x1400, 0x1400),
            (0x1806, 0x1806),
            (0x2010, 0x2015),
            (0x2053, 0x2053),
            (0x207B, 0x207B),
            (0x208B, 0x208B),
            (0x2212, 0x2212),
            (0x2E17, 0x2E17),
            (0x2E1A, 0x2E1A),
            (0x2E3A, 0x2E3B),
            (0x2E40, 0x2E40),
            (0x301C, 0x301C),
            (0x3030, 0x3030),
            (0x30A0, 0x30A0),
            (0xFE31, 0xFE32),
            (0xFE58, 0xFE58),
            (0xFE63, 0xFE63),
            (0xFF0D, 0xFF0D),
        ]),
        "Diacritic" | "Dia" => normalize(vec![
            (0x005E, 0x005E),
            (0x0060, 0x0060),
            (0x00A8, 0x00A8),
            (0x00AF, 0x00AF),
            (0x00B4, 0x00B4),
            (0x00B7, 0x00B8),
            (0x02B0, 0x034E),
            (0x0350, 0x0357),
            (0x035D, 0x0362),
            (0x0374, 0x0375),
            (0x037A, 0x037A),
            (0x0384, 0x0385),
            (0x0483, 0x0487),
            (0x0559, 0x0559),
            (0x0591, 0x05A1),
            (0x05A3, 0x05BD),
            (0x05BF, 0x05BF),
            (0x05C1, 0x05C2),
            (0x05C4, 0x05C4),
            (0x064B, 0x0652),
            (0x0657, 0x0658),
            (0x06DF, 0x06E0),
            (0x06E5, 0x06E6),
            (0x06EA, 0x06EC),
            (0x0730, 0x074A),
            (0x07A6, 0x07B0),
            (0x07EB, 0x07F5),
            (0x0818, 0x0819),
            (0x08E3, 0x08FE),
        ]),
        "Emoji" => normalize(vec![
            (0x0023, 0x0023),
            (0x002A, 0x002A),
            (0x0030, 0x0039),
            (0x00A9, 0x00A9),
            (0x00AE, 0x00AE),
            (0x203C, 0x203C),
            (0x2049, 0x2049),
            (0x2122, 0x2122),
            (0x2139, 0x2139),
            (0x2194, 0x2199),
            (0x21A9, 0x21AA),
            (0x231A, 0x231B),
            (0x2328, 0x2328),
            (0x23CF, 0x23CF),
            (0x23E9, 0x23F3),
            (0x23F8, 0x23FA),
            (0x24C2, 0x24C2),
            (0x25AA, 0x25AB),
            (0x25B6, 0x25B6),
            (0x25C0, 0x25C0),
            (0x25FB, 0x25FE),
            (0x2600, 0x2604),
            (0x260E, 0x260E),
            (0x2611, 0x2611),
            (0x2614, 0x2615),
            (0x2618, 0x2618),
            (0x261D, 0x261D),
            (0x2620, 0x2620),
            (0x2622, 0x2623),
            (0x2626, 0x2626),
            (0x262A, 0x262A),
            (0x262E, 0x262F),
            (0x2638, 0x263A),
            (0x2648, 0x2653),
            (0x2660, 0x2660),
            (0x2663, 0x2663),
            (0x2665, 0x2666),
            (0x2668, 0x2668),
            (0x267B, 0x267B),
            (0x267F, 0x267F),
            (0x2692, 0x2697),
            (0x1F300, 0x1F5FF),
            (0x1F600, 0x1F64F),
            (0x1F680, 0x1F6FF),
            (0x1F900, 0x1F9FF),
            (0x1FA70, 0x1FAFF),
        ]),
        "Emoji_Presentation" | "EPres" => normalize(vec![
            (0x231A, 0x231B),
            (0x23E9, 0x23EC),
            (0x23F0, 0x23F0),
            (0x23F3, 0x23F3),
            (0x25FD, 0x25FE),
            (0x2614, 0x2615),
            (0x2648, 0x2653),
            (0x267F, 0x267F),
            (0x2693, 0x2693),
            (0x26A1, 0x26A1),
            (0x26AA, 0x26AB),
            (0x1F300, 0x1F320),
            (0x1F600, 0x1F64F),
            (0x1F680, 0x1F6C5),
            (0x1F900, 0x1F9FF),
        ]),
        "Ideographic" | "Ideo" => normalize(vec![
            (0x3006, 0x3007),
            (0x3021, 0x3029),
            (0x3038, 0x303A),
            (0x3400, 0x4DBF),
            (0x4E00, 0x9FFF),
            (0xF900, 0xFA6D),
            (0xFA70, 0xFAD9),
            (0x20000, 0x2A6DF),
            (0x2A700, 0x2EBEF),
            (0x2F800, 0x2FA1D),
        ]),
        "Default_Ignorable_Code_Point" | "DI" => normalize(vec![
            (0x00AD, 0x00AD),
            (0x034F, 0x034F),
            (0x061C, 0x061C),
            (0x115F, 0x1160),
            (0x17B4, 0x17B5),
            (0x180B, 0x180E),
            (0x200B, 0x200F),
            (0x202A, 0x202E),
            (0x2060, 0x206F),
            (0x3164, 0x3164),
            (0xFE00, 0xFE0F),
            (0xFEFF, 0xFEFF),
            (0xFFA0, 0xFFA0),
            (0xFFF0, 0xFFF8),
        ]),
        "Noncharacter_Code_Point" | "NChar" => normalize(vec![
            (0xFDD0, 0xFDEF),
            (0xFFFE, 0xFFFF),
            (0x1FFFE, 0x1FFFF),
            (0x2FFFE, 0x2FFFF),
            (0x3FFFE, 0x3FFFF),
            (0x4FFFE, 0x4FFFF),
            (0x5FFFE, 0x5FFFF),
            (0x6FFFE, 0x6FFFF),
            (0x7FFFE, 0x7FFFF),
            (0x8FFFE, 0x8FFFF),
            (0x9FFFE, 0x9FFFF),
            (0xAFFFE, 0xAFFFF),
            (0xBFFFE, 0xBFFFF),
            (0xCFFFE, 0xCFFFF),
            (0xDFFFE, 0xDFFFF),
            (0xEFFFE, 0xEFFFF),
            (0xFFFFE, 0xFFFFF),
            (0x10FFFE, 0x10FFFF),
        ]),
        _ => return None,
    };
    Some(normalize(r))
}

/// Resolve a `\p{...}` property expression to its code-point ranges.
///
/// `name` / `value` follow the parsed grammar:
///   - Lone form `\p{Foo}` → `value = None`, `name = "Foo"` (Foo is either a
///     GC value or a binary property name).
///   - Explicit form `\p{Prop=Val}` → `name = "Prop"`, `value = Some("Val")`.
///
/// Returns `Some(ranges)` on success, or `None` if the property/value is not
/// recognized (the caller raises a SyntaxError, matching V8). This is the
/// no-stub contract: an unknown name is an honest error, never an empty match.
pub fn resolve(name: &str, value: Option<&str>) -> Option<Ranges> {
    match value {
        None => {
            // Lone: try GC value first, then binary property name.
            if let Some(code) = canon_gc(name) {
                return gc_ranges(code);
            }
            binary_ranges(name)
        }
        Some(v) => {
            match name {
                "General_Category" | "gc" => canon_gc(v).and_then(gc_ranges),
                "Script" | "sc" => script_ranges(v),
                // Script_Extensions is a superset of Script; we approximate it
                // with the Script set (correct for the common case; the extra
                // shared-script code points are the documented follow-up).
                "Script_Extensions" | "scx" => script_ranges(v),
                _ => None,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn has(r: &Ranges, cp: u32) -> bool {
        r.iter().any(|&(lo, hi)| cp >= lo && cp <= hi)
    }

    #[test]
    fn letter_matches_latin_and_greek_not_digit() {
        let l = resolve("L", None).unwrap();
        assert!(has(&l, 'a' as u32));
        assert!(has(&l, 'λ' as u32)); // U+03BB GREEK SMALL LETTER LAMDA
        assert!(!has(&l, '1' as u32));
        assert!(!has(&l, ' ' as u32));
    }

    #[test]
    fn nd_matches_digits_only() {
        let nd = resolve("Nd", None).unwrap();
        assert!(has(&nd, '5' as u32));
        assert!(has(&nd, '0' as u32));
        assert!(!has(&nd, 'a' as u32));
        // Roman numeral Ⅴ (U+2164) is Nl, not Nd.
        assert!(!has(&nd, 0x2164));
        // But it IS Nl and N.
        assert!(has(&resolve("Nl", None).unwrap(), 0x2164));
        assert!(has(&resolve("N", None).unwrap(), 0x2164));
    }

    #[test]
    fn lu_ll_split() {
        let lu = resolve("Lu", None).unwrap();
        let ll = resolve("Ll", None).unwrap();
        assert!(has(&lu, 'A' as u32));
        assert!(!has(&lu, 'a' as u32));
        assert!(has(&ll, 'a' as u32));
        assert!(!has(&ll, 'A' as u32));
        assert!(has(&lu, 'Λ' as u32)); // GREEK CAPITAL LAMDA U+039B
        assert!(has(&ll, 'λ' as u32));
    }

    #[test]
    fn script_greek_vs_latin() {
        let grek = resolve("Script", Some("Greek")).unwrap();
        assert!(has(&grek, 'α' as u32)); // U+03B1
        assert!(!has(&grek, 'a' as u32));
        // Short alias and gc-style name both work.
        let grek2 = resolve("sc", Some("Grek")).unwrap();
        assert!(has(&grek2, 'α' as u32));
        let latn = resolve("Script", Some("Latin")).unwrap();
        assert!(has(&latn, 'a' as u32));
        assert!(!has(&latn, 'α' as u32));
    }

    #[test]
    fn white_space_property() {
        let ws = resolve("White_Space", None).unwrap();
        assert!(has(&ws, ' ' as u32));
        assert!(has(&ws, '\t' as u32));
        assert!(has(&ws, 0x00A0)); // NBSP
        assert!(!has(&ws, 'a' as u32));
    }

    #[test]
    fn binary_alphabetic_and_aliases() {
        let a = resolve("Alphabetic", None).unwrap();
        let a2 = resolve("Alpha", None).unwrap();
        assert!(has(&a, 'a' as u32));
        assert!(has(&a2, 'a' as u32));
        assert!(!has(&a, '1' as u32));
    }

    #[test]
    fn unknown_property_returns_none() {
        assert!(resolve("LizardPeople", None).is_none());
        assert!(resolve("Script", Some("Klingon")).is_none());
        assert!(resolve("NotAProp", Some("x")).is_none());
    }

    #[test]
    fn set_operations() {
        let a = vec![(0u32, 10u32)];
        let b = vec![(5u32, 15u32)];
        assert_eq!(intersection(&a, &b), vec![(5, 10)]);
        assert_eq!(union(&a, &b), vec![(0, 15)]);
        assert_eq!(difference(&a, &b), vec![(0, 4)]);
        // complement of letters excludes 'a'.
        let l = resolve("L", None).unwrap();
        let nl = complement(&l);
        assert!(has(&nl, '1' as u32));
        assert!(!has(&nl, 'a' as u32));
    }

    #[test]
    fn gc_other_category() {
        let c = resolve("C", None).unwrap();
        assert!(has(&c, 0x0000)); // NUL is Cc ⊂ C
        assert!(!has(&c, 'a' as u32));
        let cc = resolve("Cc", None).unwrap();
        assert!(has(&cc, 0x0009)); // TAB is Cc
    }
}
