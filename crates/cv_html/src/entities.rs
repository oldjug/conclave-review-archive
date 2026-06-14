//! A *very* short HTML named-entity table.
//!
//! The full WHATWG named-character-references list is ~2200 entries; we
//! ship the common ones for now and parse numeric references generally.
//! Codegen of the full table from the WHATWG JSON is a build-step we'll
//! add along with `cv_unicode::ucd_gen`.

pub fn lookup_named(name: &str) -> Option<&'static str> {
    Some(match name {
        // Syntactic basics — required.
        "amp" => "&",
        "lt" => "<",
        "gt" => ">",
        "quot" => "\"",
        "apos" => "'",
        // Whitespace + non-break.
        "nbsp" => "\u{00A0}",
        "ensp" => "\u{2002}",
        "emsp" => "\u{2003}",
        "thinsp" => "\u{2009}",
        "zwj" => "\u{200D}",
        "zwnj" => "\u{200C}",
        // Trademark / legal.
        "copy" => "©",
        "reg" => "®",
        "trade" => "™",
        // Quotes.
        "lsquo" => "\u{2018}",
        "rsquo" => "\u{2019}",
        "ldquo" => "\u{201C}",
        "rdquo" => "\u{201D}",
        "laquo" => "«",
        "raquo" => "»",
        "sbquo" => "\u{201A}",
        "bdquo" => "\u{201E}",
        // Dashes + ellipsis.
        "mdash" => "—",
        "ndash" => "–",
        "hellip" => "…",
        // Bullets + middle dot.
        "bull" => "•",
        "middot" => "·",
        // Math / currency.
        "deg" => "°",
        "plusmn" => "±",
        "times" => "×",
        "divide" => "÷",
        "frac12" => "½",
        "frac14" => "¼",
        "frac34" => "¾",
        "sup2" => "²",
        "sup3" => "³",
        "micro" => "µ",
        "minus" => "−",
        "sect" => "§",
        "para" => "¶",
        "dagger" => "†",
        "Dagger" => "‡",
        "permil" => "‰",
        "pound" => "£",
        "euro" => "€",
        "yen" => "¥",
        "cent" => "¢",
        "curren" => "¤",
        // Arrows.
        "larr" => "←",
        "uarr" => "↑",
        "rarr" => "→",
        "darr" => "↓",
        "harr" => "↔",
        "lArr" => "⇐",
        "rArr" => "⇒",
        "hArr" => "⇔",
        // Misc symbols. Per the WHATWG named-reference table (HTML
        // §13.5), `&star;` is U+2606 WHITE STAR — the OUTLINED glyph.
        // Our previous mapping to U+2605 BLACK STAR was a "looks
        // close" approximation that diverged from Chrome's table
        // (audit-flagged); `&starf;` is the official name for the
        // filled form. `&check;` is NOT in the standard table at all
        // — only `&checkmark;` (U+2713) is — but we keep `check` as
        // a forgiving alias since both major engines have absorbed
        // it via real-world HTML.
        "check" => "✓",
        "checkmark" => "✓",
        "cross" => "✗",
        "star" => "☆",
        "starf" => "★",
        "hearts" => "♥",
        "diams" => "♦",
        "clubs" => "♣",
        "spades" => "♠",
        // Latin-1 letters that appear in names.
        "aacute" => "á",
        "eacute" => "é",
        "iacute" => "í",
        "oacute" => "ó",
        "uacute" => "ú",
        "ntilde" => "ñ",
        "Aacute" => "Á",
        "Eacute" => "É",
        "Iacute" => "Í",
        "Oacute" => "Ó",
        "Uacute" => "Ú",
        "Ntilde" => "Ñ",
        "auml" => "ä",
        "ouml" => "ö",
        "uuml" => "ü",
        "szlig" => "ß",
        "Auml" => "Ä",
        "Ouml" => "Ö",
        "Uuml" => "Ü",
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn common_entities_resolve() {
        assert_eq!(lookup_named("amp"), Some("&"));
        assert_eq!(lookup_named("nbsp"), Some("\u{00A0}"));
        assert_eq!(lookup_named("mdash"), Some("—"));
        assert_eq!(lookup_named("rarr"), Some("→"));
    }

    #[test]
    fn unknown_returns_none() {
        assert_eq!(lookup_named("definitely_not_an_entity"), None);
    }

    #[test]
    fn case_sensitive_pairs_separate() {
        assert_eq!(lookup_named("lArr"), Some("⇐"));
        assert_eq!(lookup_named("larr"), Some("←"));
    }

    #[test]
    fn star_entity_is_white_star_u2606() {
        // WHATWG HTML §13.5: &star; = U+2606 WHITE STAR (☆), outlined.
        // U+2605 BLACK STAR (★) is &starf;.
        assert_eq!(lookup_named("star"), Some("☆"), "&star; must be U+2606 WHITE STAR, not black");
        assert_eq!(lookup_named("starf"), Some("★"), "&starf; must be U+2605 BLACK STAR");
    }

    #[test]
    fn check_entity_alias_resolves() {
        // &check; is not in the normative WHATWG table (only &checkmark; is),
        // but we keep it as a forgiving alias.  Both must map to U+2713 ✓.
        assert_eq!(lookup_named("checkmark"), Some("✓"), "&checkmark; must be U+2713");
        assert_eq!(lookup_named("check"), Some("✓"), "&check; alias must also resolve to U+2713");
    }
}
