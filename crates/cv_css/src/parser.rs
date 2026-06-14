//! CSS rule/declaration parser — practical subset.

use crate::selectors::{Selector, parse_selector_list};
use crate::tokenizer::{CssToken, tokenize};

#[derive(Debug, Clone)]
pub struct Declaration {
    pub name: String,
    pub value: Vec<CssToken>,
    pub important: bool,
}

#[derive(Debug, Clone)]
pub struct Rule {
    pub selectors: Vec<Selector>,
    pub declarations: Vec<Declaration>,
}

#[derive(Debug, Clone)]
pub struct AtRule {
    pub name: String,
    pub prelude: Vec<CssToken>,
    /// Some at-rules (e.g. @media) have nested rules; others (@import)
    /// are statement-only.
    pub block: Option<Vec<Rule>>,
    /// Declaration-style at-rules (`@font-face`, `@page`,
    /// `@property`) carry property declarations directly in their
    /// block instead of nested qualified rules. The parser fills
    /// this whenever the at-rule's name matches one of those.
    pub declarations: Option<Vec<Declaration>>,
    /// At-rules NESTED inside this one (`@media screen { @media (min-width:640px)
    /// { ... } @supports (...) { ... } }`). Real stylesheets (Wikipedia's
    /// vector.css) nest media/supports freely. Before this existed, the nested
    /// `@` was mis-parsed as a qualified-rule prelude, which unbalanced brace
    /// tracking and LEAKED every subsequent rule into the top-level (media-
    /// unguarded) bucket — so e.g. a `@media print` heading-bold rule wrongly
    /// applied on screen. The cascade index recurses into these and folds the
    /// inner rules only when BOTH this prelude and the nested prelude match.
    pub nested: Vec<AtRule>,
}

#[derive(Debug, Clone, Default)]
pub struct Stylesheet {
    pub rules: Vec<Rule>,
    pub at_rules: Vec<AtRule>,
}

/// Parse a property-list (the contents of an inline `style="..."` HTML
/// attribute, e.g. `color: red; padding: 4px`). Returns the declarations
/// in source order.
pub fn parse_inline_style(src: &str) -> Vec<Declaration> {
    let tokens = tokenize(src);
    let mut p = Parser::new(tokens);
    p.parse_declarations()
}

pub fn parse_stylesheet(src: &str) -> Stylesheet {
    let tokens = tokenize(src);
    let mut p = Parser::new(tokens);
    let mut ss = Stylesheet::default();
    p.skip_ws();
    while p.peek_kind() != Kind::Eof {
        if matches!(p.peek(), CssToken::AtKeyword(_)) {
            if let Some(at) = p.parse_at_rule() {
                ss.at_rules.push(at);
            }
        } else if let Some(rule) = p.parse_qualified_rule() {
            ss.rules.push(rule);
        }
        p.skip_ws();
    }
    ss
}

#[derive(PartialEq, Eq)]
enum Kind {
    Eof,
    Other,
}

struct Parser {
    toks: Vec<CssToken>,
    i: usize,
}

impl Parser {
    fn new(toks: Vec<CssToken>) -> Self {
        Self { toks, i: 0 }
    }

    fn peek(&self) -> &CssToken {
        self.toks.get(self.i).unwrap_or(&CssToken::Eof)
    }

    fn peek_kind(&self) -> Kind {
        if matches!(self.peek(), CssToken::Eof) {
            Kind::Eof
        } else {
            Kind::Other
        }
    }

    fn bump(&mut self) -> CssToken {
        let t = self.toks.get(self.i).cloned().unwrap_or(CssToken::Eof);
        if !matches!(t, CssToken::Eof) {
            self.i += 1;
        }
        t
    }

    fn skip_ws(&mut self) {
        while matches!(self.peek(), CssToken::Whitespace) {
            self.bump();
        }
    }

    fn parse_qualified_rule(&mut self) -> Option<Rule> {
        let mut prelude = Vec::new();
        loop {
            match self.peek() {
                CssToken::Eof => return None,
                CssToken::LeftBrace => break,
                _ => prelude.push(self.bump()),
            }
        }
        // Consume '{'.
        self.bump();
        let declarations = self.parse_declarations();
        let selectors = parse_selector_list(&prelude);
        Some(Rule {
            selectors,
            declarations,
        })
    }

    fn parse_at_rule(&mut self) -> Option<AtRule> {
        let name = match self.bump() {
            CssToken::AtKeyword(s) => s,
            _ => return None,
        };
        let mut prelude = Vec::new();
        loop {
            match self.peek() {
                CssToken::Eof => {
                    return Some(AtRule {
                        name,
                        prelude,
                        block: None,
                        declarations: None,
                        nested: Vec::new(),
                    });
                }
                CssToken::Semicolon => {
                    self.bump();
                    return Some(AtRule {
                        name,
                        prelude,
                        block: None,
                        declarations: None,
                        nested: Vec::new(),
                    });
                }
                CssToken::LeftBrace => break,
                _ => prelude.push(self.bump()),
            }
        }
        self.bump(); // '{'
        // Declaration-style at-rules carry CSS property declarations,
        // not nested qualified rules. Parsing them as rules would
        // silently drop the contents — `@font-face { font-family:
        // "X"; src: url(...) }` would become an empty AtRule and the
        // browser would never see the font URL.
        let is_declaration_at_rule = matches!(
            name.as_str(),
            "font-face" | "page" | "property" | "counter-style" | "viewport"
        );
        if is_declaration_at_rule {
            let decls = self.parse_declarations();
            // `parse_declarations` stops at the closing brace but
            // doesn't consume it.
            if matches!(self.peek(), CssToken::RightBrace) {
                self.bump();
            }
            return Some(AtRule {
                name,
                prelude,
                block: None,
                declarations: Some(decls),
                nested: Vec::new(),
            });
        }
        // Parse the body: a mix of nested at-rules (`@media`, `@supports`,
        // `@font-face`, …) and qualified rules, until the matching '}'. Nested
        // at-rules MUST be parsed via `parse_at_rule` (not as a qualified-rule
        // prelude) — otherwise the nested `@` and its braces unbalance the loop
        // and leak every following rule into the top-level bucket.
        let mut block = Vec::new();
        let mut nested = Vec::new();
        loop {
            self.skip_ws();
            match self.peek() {
                CssToken::RightBrace => {
                    self.bump();
                    break;
                }
                CssToken::Eof => break,
                CssToken::AtKeyword(_) => {
                    if let Some(at) = self.parse_at_rule() {
                        nested.push(at);
                    } else {
                        break;
                    }
                }
                _ => {
                    if let Some(rule) = self.parse_qualified_rule() {
                        block.push(rule);
                    } else {
                        break;
                    }
                }
            }
        }
        Some(AtRule {
            name,
            prelude,
            block: Some(block),
            declarations: None,
            nested,
        })
    }

    fn parse_declarations(&mut self) -> Vec<Declaration> {
        let mut out = Vec::new();
        loop {
            self.skip_ws();
            match self.peek() {
                CssToken::Eof => return out,
                CssToken::RightBrace => {
                    self.bump();
                    return out;
                }
                CssToken::Semicolon => {
                    self.bump();
                    continue;
                }
                _ => {}
            }
            // ident: value;
            let name = match self.bump() {
                // CSS custom properties (--Foo) are case-sensitive per spec:
                // https://drafts.csswg.org/css-variables/#custom-property
                // Only lowercase standard property names, never custom ones.
                CssToken::Ident(s) if s.starts_with("--") => s,
                CssToken::Ident(s) => s.to_ascii_lowercase(),
                _ => {
                    // skip to next ; or }
                    self.skip_until_decl_break();
                    continue;
                }
            };
            self.skip_ws();
            if !matches!(self.peek(), CssToken::Colon) {
                self.skip_until_decl_break();
                continue;
            }
            self.bump();
            let mut value = Vec::new();
            let mut important = false;
            loop {
                match self.peek() {
                    CssToken::Eof | CssToken::Semicolon | CssToken::RightBrace => break,
                    CssToken::Bang => {
                        // !important
                        self.bump();
                        self.skip_ws();
                        if matches!(self.peek(), CssToken::Ident(s) if s.eq_ignore_ascii_case("important"))
                        {
                            self.bump();
                            important = true;
                        }
                    }
                    _ => value.push(self.bump()),
                }
            }
            // Normalize whitespace: collapse consecutive runs to a single
            // Whitespace token and strip leading/trailing whitespace.
            // Preserving ONE separator is necessary so that multi-value
            // shorthands like `margin-block: 10px 20px` can be split
            // correctly by `split_top_level_whitespace` in the cascade.
            // Previously we stripped ALL whitespace, which made
            // `split_top_level_whitespace` always return a single part
            // and broke two-value shorthands.
            {
                let mut normalized: Vec<CssToken> = Vec::with_capacity(value.len());
                let mut prev_was_ws = true; // start true → strips leading ws
                for t in value.drain(..) {
                    if matches!(t, CssToken::Whitespace) {
                        if !prev_was_ws {
                            normalized.push(t);
                            prev_was_ws = true;
                        }
                        // else: collapse consecutive whitespace — skip
                    } else {
                        prev_was_ws = false;
                        normalized.push(t);
                    }
                }
                // Strip trailing whitespace
                while matches!(normalized.last(), Some(CssToken::Whitespace)) {
                    normalized.pop();
                }
                value = normalized;
            }
            out.push(Declaration {
                name,
                value,
                important,
            });
            if matches!(self.peek(), CssToken::Semicolon) {
                self.bump();
            }
        }
    }

    fn skip_until_decl_break(&mut self) {
        loop {
            match self.peek() {
                CssToken::Eof | CssToken::Semicolon | CssToken::RightBrace => return,
                _ => {
                    self.bump();
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_simple_rule() {
        let ss = parse_stylesheet("h1 { color: red; font-size: 2em; }");
        assert_eq!(ss.rules.len(), 1);
        let r = &ss.rules[0];
        assert_eq!(r.declarations.len(), 2);
        assert_eq!(r.declarations[0].name, "color");
        assert_eq!(r.declarations[1].name, "font-size");
    }

    #[test]
    fn parses_important() {
        let ss = parse_stylesheet("p { color: blue !important; }");
        assert!(ss.rules[0].declarations[0].important);
    }

    #[test]
    fn parses_at_media_with_nested() {
        let ss = parse_stylesheet("@media (max-width: 600px) { body { color: green; } }");
        assert_eq!(ss.at_rules.len(), 1);
        let block = ss.at_rules[0].block.as_ref().unwrap();
        assert_eq!(block.len(), 1);
    }

    #[test]
    fn skips_bad_declarations() {
        let ss = parse_stylesheet("p { color red; font-size: 1em; }");
        // First decl is malformed (no colon); we skip it but pick up the second.
        let r = &ss.rules[0];
        assert!(r.declarations.iter().any(|d| d.name == "font-size"));
    }

    /// An `@media` block containing a NESTED `@media` must not unbalance the
    /// brace tracking — every rule that follows the nested block (and the rule
    /// AFTER the whole outer block) must stay where it belongs. Before nested
    /// at-rule support, the nested `@` was mis-parsed as a qualified-rule
    /// prelude, which leaked all following rules into the top-level bucket
    /// (bypassing the media guard — e.g. `@media print` heading-bold applied on
    /// screen). This is the exact shape from Wikipedia's vector.css.
    #[test]
    fn nested_at_media_does_not_leak_following_rules() {
        let css = "@media screen{\
                       h1{font-weight:bold}\
                       @media (min-width:640px){h1{font-size:2em}}\
                       h1{font-weight:normal}\
                   }\
                   @media print{h1{font-weight:bold}}\
                   p{color:red}";
        let ss = parse_stylesheet(css);
        // Exactly ONE top-level rule: `p{color:red}`. Nothing from inside the
        // @media blocks may leak out here.
        assert_eq!(
            ss.rules.len(),
            1,
            "only `p` is top-level; got {:?}",
            ss.rules.iter().map(|r| r.declarations.len()).collect::<Vec<_>>()
        );
        assert!(ss.rules[0].declarations.iter().any(|d| d.name == "color"));
        // Two top-level at-rules: the outer @media screen and @media print.
        assert_eq!(ss.at_rules.len(), 2);
        // The screen block holds two qualified rules directly + one nested
        // @media (the min-width one).
        let screen = &ss.at_rules[0];
        assert_eq!(screen.name, "media");
        assert_eq!(screen.block.as_ref().unwrap().len(), 2, "two direct h1 rules");
        assert_eq!(screen.nested.len(), 1, "one nested @media (min-width)");
        assert_eq!(screen.nested[0].block.as_ref().unwrap().len(), 1);
    }
}
