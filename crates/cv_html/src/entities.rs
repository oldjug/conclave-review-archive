//! WHATWG named character references — full canonical table + lookup.
//!
//! The table itself lives in [`crate::entities_table`], generated verbatim
//! from `html.spec.whatwg.org/entities.json` (the canonical machine-readable
//! list the HTML Standard §13.5 points at). It contains all ~2231 entries,
//! including the multi-code-point references (e.g. `&NotEqualTilde;` =
//! U+2242 U+0338) and the historical semicolon-optional forms (`&amp` and
//! `&amp;` are distinct entries).
//!
//! This module implements the lookups the tokenizer needs:
//!   * [`lookup_named`] — exact lookup of a single name (no `&`, optional
//!     trailing `;`), kept for tests and callers that already isolated a name.
//!   * [`longest_match`] — the **longest-prefix** match required by the
//!     Named character reference state (HTML Standard §13.2.5.73). Given the
//!     bytes *after* the `&`, it returns the decoded string and how many
//!     bytes of name were consumed. This is what makes `&notit;` resolve to
//!     `¬it;` (match `&not`, leave `it;` as text) rather than failing.

use crate::entities_table::{ENTITIES, MAX_NAME_LEN};

/// Exact lookup of a named reference by its *name* (the text between `&` and,
/// where present, the trailing `;` — the `;` itself may be included). Returns
/// the decoded UTF-8 string, or `None` if `name` is not in the table.
///
/// Callers pass the name with or without the trailing `;`; both forms are real
/// table entries per the spec, so the lookup is exact (no normalization).
pub fn lookup_named(name: &str) -> Option<&'static str> {
    ENTITIES
        .binary_search_by(|(k, _)| (*k).cmp(name))
        .ok()
        .map(|i| ENTITIES[i].1)
}

/// Longest-prefix match against the named-character-reference table, per the
/// HTML Standard's Named character reference state (§13.2.5.73): "Consume the
/// maximum number of characters possible, where the consumed characters are
/// one of the identifiers in the [named character references] table."
///
/// `input` is the byte slice *immediately after* the consumed `&`. Returns
/// `Some((decoded, name_len))` where `name_len` is the number of bytes of
/// `input` that form the matched reference name (including a trailing `;` when
/// the matched entry has one), or `None` if no prefix of `input` is a known
/// reference name.
///
/// The match is greedy: among all table names that are a prefix of `input`,
/// the longest is returned. Because the table is sorted, we scan candidate
/// prefix lengths from longest to shortest and binary-search each.
pub fn longest_match(input: &[u8]) -> Option<(&'static str, usize)> {
    // A reference name is ASCII (alphanumeric plus the trailing ';'), so the
    // longest possible candidate is bounded by both MAX_NAME_LEN and the input
    // length. Scanning from longest to shortest yields the longest match first.
    let limit = input.len().min(MAX_NAME_LEN);
    let mut len = limit;
    while len >= 1 {
        // Names are ASCII; if these bytes aren't valid UTF-8 they can't match,
        // but every prefix of ASCII input is valid UTF-8 anyway.
        if let Ok(candidate) = std::str::from_utf8(&input[..len]) {
            if let Some(decoded) = lookup_named(candidate) {
                return Some((decoded, len));
            }
        }
        len -= 1;
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::entities_table::ENTITY_COUNT;

    #[test]
    fn full_table_loaded() {
        // The canonical WHATWG table has 2231 entries. If this number changes,
        // the table was regenerated from a different spec snapshot.
        assert_eq!(ENTITIES.len(), ENTITY_COUNT);
        assert_eq!(ENTITIES.len(), 2231);
    }

    #[test]
    fn table_is_sorted_for_binary_search() {
        // longest_match / lookup_named rely on the table being sorted.
        assert!(ENTITIES.windows(2).all(|w| w[0].0 < w[1].0));
    }

    #[test]
    fn common_entities_resolve() {
        assert_eq!(lookup_named("amp;"), Some("&"));
        assert_eq!(lookup_named("amp"), Some("&")); // legacy no-semicolon form
        assert_eq!(lookup_named("nbsp;"), Some("\u{00A0}"));
        assert_eq!(lookup_named("mdash;"), Some("\u{2014}"));
        assert_eq!(lookup_named("rarr;"), Some("\u{2192}"));
        assert_eq!(lookup_named("copy;"), Some("\u{00A9}"));
    }

    #[test]
    fn multi_codepoint_entities() {
        // &NotEqualTilde; = U+2242 U+0338 (two code points).
        assert_eq!(lookup_named("NotEqualTilde;"), Some("\u{2242}\u{0338}"));
        // &fjlig; = U+0066 U+006A ("fj").
        assert_eq!(lookup_named("fjlig;"), Some("fj"));
    }

    #[test]
    fn long_name_resolves() {
        // The longest names in the table — verifies MAX_NAME_LEN is sized.
        assert_eq!(
            lookup_named("CounterClockwiseContourIntegral;"),
            Some("\u{2233}")
        );
    }

    #[test]
    fn unknown_returns_none() {
        assert_eq!(lookup_named("definitely_not_an_entity"), None);
        assert_eq!(lookup_named("notit;"), None);
    }

    #[test]
    fn case_sensitive_pairs_separate() {
        assert_eq!(lookup_named("lArr;"), Some("\u{21D0}"));
        assert_eq!(lookup_named("larr;"), Some("\u{2190}"));
        assert_eq!(lookup_named("Gt;"), Some("\u{226B}"));
        assert_eq!(lookup_named("gt;"), Some("\u{003E}"));
    }

    #[test]
    fn longest_match_finds_full_name() {
        // &copy; — longest match consumes "copy;" (5 bytes).
        let (decoded, len) = longest_match(b"copy;rest").unwrap();
        assert_eq!(decoded, "\u{00A9}");
        assert_eq!(len, 5);
    }

    #[test]
    fn longest_match_boundary_notit() {
        // The canonical longest-match test: "&notit;" — "not" is a known
        // reference (no-semicolon legacy form), "notit" is not, so the
        // longest match is "not" (3 bytes), leaving "it;" as literal text.
        let (decoded, len) = longest_match(b"notit;").unwrap();
        assert_eq!(decoded, "\u{00AC}"); // U+00AC NOT SIGN
        assert_eq!(len, 3); // consumed "not", left "it;"
    }

    #[test]
    fn longest_match_prefers_semicolon_form() {
        // "&not;" — both "not" and "not;" are in the table; longest match
        // takes "not;" (4 bytes).
        let (decoded, len) = longest_match(b"not;more").unwrap();
        assert_eq!(decoded, "\u{00AC}");
        assert_eq!(len, 4);
    }

    #[test]
    fn longest_match_legacy_no_semicolon() {
        // "&amp" with no semicolon and a non-name char after — legacy form.
        let (decoded, len) = longest_match(b"amp x").unwrap();
        assert_eq!(decoded, "&");
        assert_eq!(len, 3);
    }

    #[test]
    fn longest_match_none() {
        assert_eq!(longest_match(b"zzznotareference"), None);
        assert_eq!(longest_match(b""), None);
    }
}
