//! `cv_spell` — Hunspell-compatible affix dictionary spell checker.
//!
//! Reads `.aff` (affix rules) + `.dic` (root word list) pairs the way
//! Chrome does and produces:
//!   * `check(word)` — is the word in the dictionary or buildable from
//!     a root + prefix/suffix rule?
//!   * `suggest(word)` — up to N edit-distance-1 candidates that the
//!     dictionary accepts.
//!
//! From-scratch — no third-party crates, no FFI to Hunspell. The
//! `.aff` grammar implemented covers the subset every shipped Chrome
//! language pack uses: `SET`, `TRY`, `WORDCHARS`, `PFX`, `SFX`, `REP`,
//! `MAP`, comments via `#`. Multi-byte UTF-8 SET (`UTF-8`) handled.
//!
//! Wire-up: conclave instantiates a `Dictionary` per active locale
//! via `Dictionary::load_files`, then runs `check` on every editable
//! text element's tokens. Misspelled runs get a red underline at paint
//! and right-click → `suggest` opens the replacement menu.

use std::collections::HashMap;

#[derive(Debug, Default)]
pub struct Affix {
    /// Flag character / string identifying this rule from the .dic side.
    pub flag: String,
    /// Whether this is a prefix (false → suffix).
    pub is_prefix: bool,
    /// Can this stack with others? (`Y`/`N` in the .aff header.)
    pub cross_product: bool,
    pub entries: Vec<AffixEntry>,
}

#[derive(Debug, Clone)]
pub struct AffixEntry {
    /// Characters to strip from the root before adding.
    pub strip: String,
    /// Characters to add.
    pub add: String,
    /// Condition: simple regex-like class over the root edge.
    pub condition: String,
}

#[derive(Debug, Default)]
pub struct Dictionary {
    /// Root words → flags string (e.g. "dog/SM" → "SM").
    pub words: HashMap<String, String>,
    /// Affix rules, keyed by flag.
    pub affixes: HashMap<String, Affix>,
    /// REP rules: (from, to) substring replacements that suggestion
    /// gen should try (e.g. `f → ph`).
    pub rep: Vec<(String, String)>,
    /// MAP rules: groups of "similar" chars treated as one for
    /// edit distance (e.g. `aàáâ`).
    pub map: Vec<Vec<String>>,
    /// `TRY` line: characters to try inserting/swapping in suggestions,
    /// ordered by frequency.
    pub try_chars: String,
    pub wordchars: String,
    pub encoding: String,
}

impl Dictionary {
    /// Build from already-loaded text bodies (no filesystem touch — the
    /// embedder reads the bytes).
    pub fn from_strings(aff_text: &str, dic_text: &str) -> Self {
        let mut d = Self::default();
        d.parse_affix(aff_text);
        d.parse_dic(dic_text);
        d
    }

    /// Convenience for the embedder: read both files from disk.
    pub fn load_files(aff_path: &str, dic_path: &str) -> std::io::Result<Self> {
        let aff = std::fs::read_to_string(aff_path)?;
        let dic = std::fs::read_to_string(dic_path)?;
        Ok(Self::from_strings(&aff, &dic))
    }

    fn parse_affix(&mut self, text: &str) {
        let mut lines = text.lines().peekable();
        while let Some(line) = lines.next() {
            let line = strip_comment(line).trim();
            if line.is_empty() {
                continue;
            }
            let mut parts = line.split_whitespace();
            let cmd = parts.next().unwrap_or("");
            match cmd {
                "SET" => {
                    self.encoding = parts.next().unwrap_or("UTF-8").to_string();
                }
                "TRY" => {
                    self.try_chars = parts.collect::<Vec<_>>().join("");
                }
                "WORDCHARS" => {
                    self.wordchars = parts.collect::<Vec<_>>().join("");
                }
                "REP" => {
                    if let Some(first) = parts.next() {
                        // The first REP gives the count of subsequent ones.
                        if first.chars().all(|c| c.is_ascii_digit()) {
                            // Header line — entries follow.
                            continue;
                        }
                        if let Some(to) = parts.next() {
                            self.rep.push((first.to_string(), to.to_string()));
                        }
                    }
                }
                "MAP" => {
                    if let Some(first) = parts.next() {
                        if first.chars().all(|c| c.is_ascii_digit()) {
                            continue;
                        }
                        let group: Vec<String> = first.chars().map(|c| c.to_string()).collect();
                        self.map.push(group);
                    }
                }
                "PFX" | "SFX" => {
                    let flag = parts.next().unwrap_or("").to_string();
                    let cross = parts.next().unwrap_or("N") == "Y";
                    // Third arg is entry count, then entries on
                    // subsequent lines.
                    let _count: usize = parts.next().unwrap_or("0").parse().unwrap_or(0);
                    let is_prefix = cmd == "PFX";
                    let af = self.affixes.entry(flag.clone()).or_insert(Affix {
                        flag: flag.clone(),
                        is_prefix,
                        cross_product: cross,
                        entries: Vec::new(),
                    });
                    af.is_prefix = is_prefix;
                    af.cross_product = cross;

                    // Read entries until we see a non-PFX/SFX line for
                    // this flag. (Hunspell allows interleaving but in
                    // practice they're contiguous.)
                    while let Some(peek) = lines.peek().cloned() {
                        let l = strip_comment(peek).trim();
                        if l.is_empty() {
                            lines.next();
                            continue;
                        }
                        let mut p = l.split_whitespace();
                        let cmd2 = p.next().unwrap_or("");
                        if cmd2 != cmd {
                            break;
                        }
                        let f2 = p.next().unwrap_or("");
                        if f2 != flag {
                            break;
                        }
                        let strip = p.next().unwrap_or("0").to_string();
                        let add = p.next().unwrap_or("0").to_string();
                        let condition = p.next().unwrap_or(".").to_string();
                        af.entries.push(AffixEntry {
                            strip: if strip == "0" { String::new() } else { strip },
                            add: if add == "0" { String::new() } else { add },
                            condition,
                        });
                        lines.next();
                    }
                }
                _ => {}
            }
        }
    }

    fn parse_dic(&mut self, text: &str) {
        let mut lines = text.lines();
        // Hunspell .dic starts with a count line; skip it if numeric.
        if let Some(first) = lines.next() {
            let first = first.trim();
            if !first.chars().all(|c| c.is_ascii_digit()) {
                self.ingest_dic_line(first);
            }
        }
        for line in lines {
            self.ingest_dic_line(line);
        }
    }

    fn ingest_dic_line(&mut self, raw: &str) {
        let line = strip_comment(raw).trim();
        if line.is_empty() {
            return;
        }
        // Optional `/FLAGS` after a word.
        let (word, flags) = match line.split_once('/') {
            Some((w, rest)) => {
                // Some dicts add morphological data after a space;
                // keep only the flag chars.
                let flag_str = rest.split_whitespace().next().unwrap_or("");
                (w.to_string(), flag_str.to_string())
            }
            None => (
                line.split_whitespace().next().unwrap_or("").to_string(),
                String::new(),
            ),
        };
        if !word.is_empty() {
            self.words.insert(word, flags);
        }
    }

    /// Is `word` valid? Case-folded match first, then affix expansion.
    pub fn check(&self, word: &str) -> bool {
        if word.is_empty() {
            return true;
        }
        let lower: String = word.chars().flat_map(char::to_lowercase).collect();

        // Direct lookup against root list (case-insensitive).
        if self.words.contains_key(word) || self.words.contains_key(&lower) {
            return true;
        }

        // Try stripping each known prefix.
        for af in self.affixes.values().filter(|a| a.is_prefix) {
            for entry in &af.entries {
                if let Some(root) = strip_prefix_with(word, entry) {
                    if self.root_has_flag(&root, &af.flag) || self.words.contains_key(&root) {
                        return true;
                    }
                }
            }
        }
        // Try stripping each known suffix.
        for af in self.affixes.values().filter(|a| !a.is_prefix) {
            for entry in &af.entries {
                if let Some(root) = strip_suffix_with(word, entry) {
                    if self.root_has_flag(&root, &af.flag) || self.words.contains_key(&root) {
                        return true;
                    }
                }
            }
        }
        false
    }

    fn root_has_flag(&self, root: &str, flag: &str) -> bool {
        match self.words.get(root) {
            Some(flags) => flags.contains(flag),
            None => false,
        }
    }

    /// Return up to `n` candidate corrections, ranked by edit distance.
    pub fn suggest(&self, word: &str, n: usize) -> Vec<String> {
        let mut candidates: Vec<(usize, String)> = Vec::new();

        // 1) REP substitutions — these are the highest-confidence fixes.
        for (from, to) in &self.rep {
            if word.contains(from.as_str()) {
                let cand = word.replacen(from, to, 1);
                if self.check(&cand) {
                    candidates.push((0, cand));
                }
            }
        }

        // 2) Single edit-distance neighbors.
        for cand in edit1(word, &self.try_chars) {
            if self.check(&cand) {
                let d = levenshtein(word, &cand);
                candidates.push((d, cand));
            }
        }

        // De-duplicate by string, keep lowest distance, sort, then crop.
        candidates.sort_by(|a, b| a.1.cmp(&b.1));
        candidates.dedup_by(|a, b| a.1 == b.1);
        candidates.sort_by_key(|x| x.0);
        candidates.into_iter().take(n).map(|(_, s)| s).collect()
    }

    /// Tokenize a string of text into (start, end, word) triples
    /// suitable for spell-check passes. Treats anything in
    /// `wordchars` as part of a word in addition to alphanumerics.
    pub fn tokenize<'a>(&self, text: &'a str) -> Vec<(usize, usize, &'a str)> {
        let mut out = Vec::new();
        let mut start: Option<usize> = None;
        let bytes = text.as_bytes();
        let mut i = 0;
        while i < bytes.len() {
            // Walk one UTF-8 code point.
            let ch = text[i..].chars().next().unwrap();
            let n = ch.len_utf8();
            let is_word = ch.is_alphanumeric() || self.wordchars.contains(ch) || ch == '\'';
            if is_word {
                if start.is_none() {
                    start = Some(i);
                }
            } else if let Some(s) = start.take() {
                out.push((s, i, &text[s..i]));
            }
            i += n;
        }
        if let Some(s) = start {
            out.push((s, bytes.len(), &text[s..]));
        }
        out
    }
}

fn strip_comment(line: &str) -> &str {
    match line.find('#') {
        Some(i) => &line[..i],
        None => line,
    }
}

/// Check the condition expression at the prefix edge.
fn condition_matches(condition: &str, target: &str, is_prefix: bool) -> bool {
    if condition == "." || condition.is_empty() {
        return true;
    }
    // Hunspell condition class subset: `[abc]`, `[^abc]`, single
    // chars. For our subset we match one class against the start
    // (prefix) or end (suffix) edge.
    let edge_ch = if is_prefix {
        target.chars().next()
    } else {
        target.chars().rev().next()
    };
    let Some(ch) = edge_ch else { return false };
    // Simplified: if condition starts with `[`, parse a class. Else
    // it's a literal sequence; match the relevant edge char.
    if condition.starts_with('[') {
        let end = condition.find(']').unwrap_or(condition.len());
        let inner = &condition[1..end];
        let (neg, body) = if let Some(stripped) = inner.strip_prefix('^') {
            (true, stripped)
        } else {
            (false, inner)
        };
        let contains = body.chars().any(|c| c == ch);
        return neg ^ contains;
    }
    // Treat as literal — first char must match.
    condition.chars().next() == Some(ch)
}

fn strip_prefix_with(word: &str, entry: &AffixEntry) -> Option<String> {
    if !word.starts_with(entry.add.as_str()) {
        return None;
    }
    let root = &word[entry.add.len()..];
    let with_strip = format!("{}{}", entry.strip, root);
    if !condition_matches(&entry.condition, &with_strip, true) {
        return None;
    }
    Some(with_strip)
}

fn strip_suffix_with(word: &str, entry: &AffixEntry) -> Option<String> {
    if !word.ends_with(entry.add.as_str()) {
        return None;
    }
    let root_len = word.len() - entry.add.len();
    let root = &word[..root_len];
    let with_strip = format!("{}{}", root, entry.strip);
    if !condition_matches(&entry.condition, &with_strip, false) {
        return None;
    }
    Some(with_strip)
}

/// Generate single-edit-distance candidates. Covers:
///   * deletion of any one char
///   * insertion of any char from `try_chars` at any position
///   * substitution at any position with any char from `try_chars`
///   * transposition of any two adjacent chars
pub fn edit1(word: &str, try_chars: &str) -> Vec<String> {
    let chars: Vec<char> = word.chars().collect();
    let n = chars.len();
    let alphabet: Vec<char> = if try_chars.is_empty() {
        "abcdefghijklmnopqrstuvwxyz".chars().collect()
    } else {
        try_chars.chars().collect()
    };
    let mut out: Vec<String> = Vec::new();

    // deletions
    for i in 0..n {
        let mut s = String::with_capacity(word.len());
        for (j, c) in chars.iter().enumerate() {
            if j != i {
                s.push(*c);
            }
        }
        out.push(s);
    }
    // transpositions
    for i in 0..n.saturating_sub(1) {
        let mut s: Vec<char> = chars.clone();
        s.swap(i, i + 1);
        out.push(s.into_iter().collect());
    }
    // substitutions
    for i in 0..n {
        for &c in &alphabet {
            if c == chars[i] {
                continue;
            }
            let mut s: Vec<char> = chars.clone();
            s[i] = c;
            out.push(s.into_iter().collect());
        }
    }
    // insertions
    for i in 0..=n {
        for &c in &alphabet {
            let mut s = String::with_capacity(word.len() + 1);
            for (j, ch) in chars.iter().enumerate() {
                if j == i {
                    s.push(c);
                }
                s.push(*ch);
            }
            if i == n {
                s.push(c);
            }
            out.push(s);
        }
    }
    out
}

/// Plain Damerau-Levenshtein distance (insertion / deletion /
/// substitution / transposition of adjacent), for ranking suggestions.
pub fn levenshtein(a: &str, b: &str) -> usize {
    let av: Vec<char> = a.chars().collect();
    let bv: Vec<char> = b.chars().collect();
    let m = av.len();
    let n = bv.len();
    if m == 0 {
        return n;
    }
    if n == 0 {
        return m;
    }
    let mut prev: Vec<usize> = (0..=n).collect();
    let mut curr: Vec<usize> = vec![0; n + 1];
    let mut prev_prev: Vec<usize> = vec![0; n + 1];
    for i in 1..=m {
        curr[0] = i;
        for j in 1..=n {
            let cost = usize::from(av[i - 1] != bv[j - 1]);
            curr[j] = (prev[j] + 1) // deletion
                .min(curr[j - 1] + 1) // insertion
                .min(prev[j - 1] + cost); // substitution
            if i > 1 && j > 1 && av[i - 1] == bv[j - 2] && av[i - 2] == bv[j - 1] {
                curr[j] = curr[j].min(prev_prev[j - 2] + 1);
            }
        }
        prev_prev = prev.clone();
        prev = curr.clone();
    }
    prev[n]
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    const AFF: &str = "\
SET UTF-8
TRY esiarntolcdugmphbyfvkwzqxj
WORDCHARS '
PFX A Y 1
PFX A 0 re .
SFX B Y 2
SFX B 0 s [^sxyz]
SFX B 0 es [sxyz]
REP 1
REP f ph
";
    const DIC: &str = "\
4
walk/AB
ride/B
run
mother
";

    #[test]
    fn known_root_is_valid() {
        let d = Dictionary::from_strings(AFF, DIC);
        assert!(d.check("walk"));
        assert!(d.check("run"));
        assert!(d.check("mother"));
    }

    #[test]
    fn suffix_expansion_validates_walks() {
        let d = Dictionary::from_strings(AFF, DIC);
        assert!(d.check("walks"));
        assert!(d.check("rides"));
    }

    #[test]
    fn prefix_expansion_validates_rewalk() {
        let d = Dictionary::from_strings(AFF, DIC);
        assert!(d.check("rewalk"));
    }

    #[test]
    fn unknown_word_rejected() {
        let d = Dictionary::from_strings(AFF, DIC);
        assert!(!d.check("xyzzy"));
    }

    #[test]
    fn suggest_finds_typo_fix() {
        let d = Dictionary::from_strings(AFF, DIC);
        let sugg = d.suggest("walke", 5);
        assert!(sugg.iter().any(|s| s == "walk"));
    }

    #[test]
    fn rep_rule_kicks_in() {
        let d = Dictionary::from_strings("REP 1\nREP f ph\n", "1\nphat\n");
        // "fat" doesn't validate, but REP suggests "phat".
        let sugg = d.suggest("fat", 5);
        assert!(sugg.iter().any(|s| s == "phat"));
    }

    #[test]
    fn tokenize_splits_text() {
        let d = Dictionary::from_strings("WORDCHARS '\n", "1\nrun\n");
        let toks = d.tokenize("the quick brown");
        assert_eq!(toks.len(), 3);
        assert_eq!(toks[0].2, "the");
        assert_eq!(toks[2].2, "brown");
    }

    #[test]
    fn levenshtein_is_correct() {
        assert_eq!(levenshtein("kitten", "sitting"), 3);
        assert_eq!(levenshtein("abc", "abc"), 0);
        assert_eq!(levenshtein("", "abc"), 3);
        // Damerau-aware transposition: ab <-> ba is distance 1.
        assert_eq!(levenshtein("ab", "ba"), 1);
    }
}
