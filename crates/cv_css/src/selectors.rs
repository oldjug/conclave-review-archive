//! Selectors 4 — practical subset.
//!
//! Supported:
//!   - type selectors (`p`)
//!   - universal (`*`)
//!   - class (`.x`)
//!   - id (`#x`)
//!   - compound (`a.foo#bar`)
//!   - descendant (` `)
//!   - child (`>`)
//!   - selector list (`a, b, c`)
//!   - simple pseudo-class names (`:hover`) — parsed, not matched yet
//!
//! Not yet: attribute selectors, sibling combinators, `:not()`, functional
//! pseudos, `::pseudo-elements`.

use crate::tokenizer::CssToken;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SimpleSelector {
    pub element: Option<String>, // None = '*'
    pub id: Option<String>,
    pub classes: Vec<String>,
    pub pseudo_classes: Vec<String>,
    pub attrs: Vec<AttrSelector>,
    /// `:not(...)` arguments. Compound matches iff NONE of these inner
    /// selectors match the element. Multiple `:not(...)` on the same
    /// compound stack via AND.
    pub not_selectors: Vec<Selector>,
    /// `:is(...)` arguments — compound matches iff AT LEAST ONE inner
    /// selector matches. `:is()` contributes the max specificity of its
    /// argument list to the host. Empty list = no `:is` constraint.
    pub is_selectors: Vec<Selector>,
    /// `:where(...)` arguments — same matching semantics as `:is()`
    /// but contributes ZERO specificity to the host (CSS Selectors L4).
    /// Stored separately so the specificity computation can skip them.
    pub where_selectors: Vec<Selector>,
    /// `:has(...)` arguments — compound matches iff at least one inner
    /// selector matches some descendant of this element. Empty = no
    /// `:has` constraint.
    pub has_selectors: Vec<Selector>,
    /// `::before` / `::after` / etc. pseudo-element this compound
    /// targets. None = host element. The cascade routes declarations
    /// for a non-None pseudo into a separate per-element bucket so
    /// the layout tree builder can synthesize a generated child box.
    pub pseudo_element: Option<String>,
    /// `:nth-child(An+B)` / `:nth-last-child(An+B)` /
    /// `:nth-of-type(An+B)` / `:nth-last-of-type(An+B)`. Each tuple
    /// pairs the kind of position counted (forward vs backward, all
    /// siblings vs same-tag) with the An+B specification. Multiple
    /// nth-* on the same compound stack via AND.
    pub nth_selectors: Vec<(NthKind, NthArg)>,
    /// `:lang(tag, ...)` — BCP47 language-range arguments (lowercased,
    /// quotes stripped). Compound matches iff the element's resolved
    /// content language (its own or an ancestor's `lang` attribute, or the
    /// document language) matches ANY of these ranges by the standard
    /// prefix rule (`en` matches `en` and `en-US`). Empty = no `:lang`
    /// constraint. This MUST be matched against real language state — a
    /// permissive fallback wrongly applied e.g. Wikipedia's
    /// `h1:lang(ckb){font-family:Scheherazade}` to every heading.
    pub lang_args: Vec<String>,
}

/// Which sibling index `:nth-*` should compute on a matched element.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NthKind {
    /// `:nth-child(...)` — count among all sibling elements, forward.
    Child,
    /// `:nth-last-child(...)` — count from the end.
    LastChild,
    /// `:nth-of-type(...)` — count only siblings whose tag matches.
    OfType,
    /// `:nth-last-of-type(...)` — same as OfType but counted from the end.
    LastOfType,
}

/// Resolved `An+B` argument. `a` and `b` are signed because forms like
/// `-n+3` (CSS Selectors L4 §6.7) need negative slopes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NthArg {
    pub a: i32,
    pub b: i32,
}

impl NthArg {
    /// True if integer `n ≥ 1` satisfies `index = a * step + b` for some
    /// non-negative integer `step` (per the spec, `n` ranges over the
    /// non-negative integers; we require the resulting index to be ≥1
    /// since CSS uses 1-based sibling indices).
    pub fn matches(&self, index: i32) -> bool {
        if index < 1 {
            return false;
        }
        if self.a == 0 {
            return index == self.b;
        }
        let diff = index - self.b;
        if diff % self.a != 0 {
            return false;
        }
        let step = diff / self.a;
        step >= 0
    }
}

/// Parse the `An+B` form per CSS Selectors L4 §6.7. Accepts: `odd`,
/// `even`, integer literal, `n`, `An`, `An+B`, `An-B`, `-n+B`, `2n`,
/// `-n`. Whitespace tolerated around the `+`/`-`. Returns None on
/// syntactically-invalid input.
pub fn parse_an_plus_b(input: &str) -> Option<NthArg> {
    let s: String = input.chars().filter(|c| !c.is_whitespace()).collect();
    let lc = s.to_ascii_lowercase();
    if lc == "odd" {
        return Some(NthArg { a: 2, b: 1 });
    }
    if lc == "even" {
        return Some(NthArg { a: 2, b: 0 });
    }
    if let Ok(b) = lc.parse::<i32>() {
        return Some(NthArg { a: 0, b });
    }
    // From here on the value contains an `n`.
    let n_pos = lc.find('n')?;
    let a_part = &lc[..n_pos];
    let b_part = &lc[n_pos + 1..];
    let a: i32 = match a_part {
        "" => 1,
        "+" => 1,
        "-" => -1,
        x => x.parse().ok()?,
    };
    let b: i32 = if b_part.is_empty() {
        0
    } else {
        // b_part may start with `+` or `-`.
        let first = b_part.chars().next()?;
        if first == '+' || first == '-' {
            b_part.parse().ok()?
        } else {
            return None;
        }
    };
    Some(NthArg { a, b })
}

/// `[name]` / `[name="value"]` / `[name~="value"]` / `[name|="value"]` /
/// `[name^="value"]` / `[name$="value"]` / `[name*="value"]` per CSS
/// Selectors 4. Case-insensitive in attribute name; value comparisons
/// follow the operator semantics.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AttrSelector {
    pub name: String,
    pub op: AttrOp,
    pub value: String,
}

impl SimpleSelector {
    pub fn empty() -> Self {
        Self {
            element: None,
            id: None,
            classes: Vec::new(),
            pseudo_classes: Vec::new(),
            attrs: Vec::new(),
            not_selectors: Vec::new(),
            is_selectors: Vec::new(),
            where_selectors: Vec::new(),
            has_selectors: Vec::new(),
            pseudo_element: None,
            nth_selectors: Vec::new(),
            lang_args: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttrOp {
    /// `[name]` — element merely has the attribute.
    Exists,
    /// `[name="x"]` — exact match.
    Equals,
    /// `[name~="x"]` — whitespace-separated list contains x.
    Includes,
    /// `[name|="x"]` — exact match OR starts with x followed by `-`
    /// (the BCP47 language-tag matching idiom).
    DashMatch,
    /// `[name^="x"]` — starts with x.
    Prefix,
    /// `[name$="x"]` — ends with x.
    Suffix,
    /// `[name*="x"]` — contains x as substring.
    Substring,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Combinator {
    /// `A B` — B is any descendant of A.
    Descendant,
    /// `A > B` — B is a direct child of A.
    Child,
    /// `A + B` — B is the immediate next sibling of A.
    NextSibling,
    /// `A ~ B` — B is any later sibling of A in the same parent.
    SubsequentSibling,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ComplexPart {
    pub combinator: Option<Combinator>, // None for the leftmost part
    pub compound: SimpleSelector,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Selector {
    pub parts: Vec<ComplexPart>,
}

impl Selector {
    /// True if this selector's rightmost compound targets a pseudo-element
    /// (`::before` / `::after` / …). The renderer uses the aggregate of this
    /// across all loaded rules to skip the per-element pseudo-element probe
    /// entirely on the (common) pages whose sheets define no pseudo rules.
    pub fn targets_pseudo_element(&self) -> bool {
        self.parts
            .last()
            .is_some_and(|p| p.compound.pseudo_element.is_some())
    }

    /// CSS specificity per Selectors L4 §16: tuple of
    /// (id_count, class+attribute+pseudo-class_count, element+pseudo-element_count),
    /// packed into a u32 with 8 bits per slot (saturating at 255).
    ///
    /// Special pseudo-class rules:
    /// - `:where(...)` contributes ZERO specificity (its args are
    ///   matched but their specificity is discarded).
    /// - `:is(...)`, `:not(...)`, `:has(...)` contribute the
    ///   MAX specificity of their argument list to the host compound.
    /// - Plain pseudo-classes (`:hover`, `:checked`, etc.) count as a class.
    /// - Attribute selectors count as a class.
    /// - Pseudo-elements count as an element.
    pub fn specificity(&self) -> u32 {
        let (a, b, c) = self.specificity_triple();
        let a = a.min(255);
        let b = b.min(255);
        let c = c.min(255);
        (a << 16) | (b << 8) | c
    }

    /// Compute the (id, class, type) triple WITHOUT packing — useful
    /// for taking componentwise max() when nesting `:is()`/`:not()`.
    fn specificity_triple(&self) -> (u32, u32, u32) {
        let mut ids = 0u32;
        let mut classes = 0u32;
        let mut types = 0u32;
        for p in &self.parts {
            let c = &p.compound;
            if c.id.is_some() {
                ids += 1;
            }
            classes += c.classes.len() as u32;
            classes += c.pseudo_classes.len() as u32;
            classes += c.attrs.len() as u32;
            classes += c.nth_selectors.len() as u32;
            // `:lang(...)` is a plain pseudo-class → 0-1-0 each (it lives in
            // lang_args rather than pseudo_classes so matching stays
            // language-aware, but it still contributes class specificity).
            classes += c.lang_args.len() as u32;
            if c.element.is_some() {
                types += 1;
            }
            if c.pseudo_element.is_some() {
                types += 1;
            }
            // :is(...) / :not(...) / :has(...) contribute max(inner).
            let inner_max = |selectors: &[Selector]| -> (u32, u32, u32) {
                let mut best = (0u32, 0u32, 0u32);
                for s in selectors {
                    let t = s.specificity_triple();
                    // Compare lexicographically — same rule as CSS
                    // cascade uses to pick winning rules.
                    if (t.0, t.1, t.2) > (best.0, best.1, best.2) {
                        best = t;
                    }
                }
                best
            };
            let (ii, ic, it) = inner_max(&c.is_selectors);
            ids += ii;
            classes += ic;
            types += it;
            let (ni, nc, nt) = inner_max(&c.not_selectors);
            ids += ni;
            classes += nc;
            types += nt;
            let (hi, hc, ht) = inner_max(&c.has_selectors);
            ids += hi;
            classes += hc;
            types += ht;
            // :where(...) contributes ZERO — intentionally not summed.
            let _ = &c.where_selectors;
        }
        (ids, classes, types)
    }
}

/// True for the names that, when prefixed by `:` or `::`, denote a
/// pseudo-ELEMENT (per CSS Selectors Level 4 §3.4) rather than a
/// pseudo-class.
pub fn is_pseudo_element_name(name: &str) -> bool {
    matches!(
        name,
        "before"
            | "after"
            | "first-line"
            | "first-letter"
            | "placeholder"
            | "marker"
            | "selection"
            | "backdrop"
            | "file-selector-button"
            | "details-content"
            | "grammar-error"
            | "spelling-error"
            | "target-text"
            | "view-transition"
            | "view-transition-group"
            | "view-transition-image-pair"
            | "view-transition-new"
            | "view-transition-old"
            | "highlight"
            | "cue"
            | "cue-region"
            | "slotted"
            | "part"
    )
}

pub fn parse_selector_list(prelude: &[CssToken]) -> Vec<Selector> {
    let groups: Vec<Vec<CssToken>> = split_on_commas(prelude);
    let mut out = Vec::new();
    for g in groups {
        if let Some(sel) = parse_complex(&g) {
            out.push(sel);
        }
    }
    out
}

fn split_on_commas(toks: &[CssToken]) -> Vec<Vec<CssToken>> {
    let mut groups = Vec::new();
    let mut current = Vec::new();
    // Track parenthesis depth so commas inside `:not(...)`, `:is(...)`,
    // etc. don't split the outer selector list.
    let mut depth: u32 = 0;
    for t in toks {
        match t {
            CssToken::Function(_) | CssToken::LeftParen => {
                depth += 1;
                current.push(t.clone());
            }
            CssToken::RightParen => {
                if depth > 0 {
                    depth -= 1;
                }
                current.push(t.clone());
            }
            CssToken::Comma if depth == 0 => {
                groups.push(std::mem::take(&mut current));
            }
            _ => current.push(t.clone()),
        }
    }
    groups.push(current);
    groups
}

fn parse_complex(toks: &[CssToken]) -> Option<Selector> {
    // Walk tokens, building compound selectors separated by combinators.
    let mut parts: Vec<ComplexPart> = Vec::new();
    let mut current = SimpleSelector::empty();
    let mut have_part = false;
    let mut pending_combinator: Option<Option<Combinator>> = Some(None); // leftmost has no combinator
    let mut i = 0;
    while i < toks.len() {
        let t = &toks[i];
        match t {
            CssToken::Whitespace => {
                if have_part {
                    parts.push(ComplexPart {
                        combinator: pending_combinator
                            .take()
                            .unwrap_or(Some(Combinator::Descendant)),
                        compound: std::mem::replace(&mut current, SimpleSelector::empty()),
                    });
                    have_part = false;
                    pending_combinator = Some(Some(Combinator::Descendant));
                }
                i += 1;
            }
            CssToken::Delim('>') => {
                if have_part {
                    parts.push(ComplexPart {
                        combinator: pending_combinator
                            .take()
                            .unwrap_or(Some(Combinator::Descendant)),
                        compound: std::mem::replace(&mut current, SimpleSelector::empty()),
                    });
                    have_part = false;
                }
                pending_combinator = Some(Some(Combinator::Child));
                i += 1;
            }
            CssToken::Delim('+') => {
                if have_part {
                    parts.push(ComplexPart {
                        combinator: pending_combinator
                            .take()
                            .unwrap_or(Some(Combinator::Descendant)),
                        compound: std::mem::replace(&mut current, SimpleSelector::empty()),
                    });
                    have_part = false;
                }
                pending_combinator = Some(Some(Combinator::NextSibling));
                i += 1;
            }
            CssToken::Delim('~') => {
                if have_part {
                    parts.push(ComplexPart {
                        combinator: pending_combinator
                            .take()
                            .unwrap_or(Some(Combinator::Descendant)),
                        compound: std::mem::replace(&mut current, SimpleSelector::empty()),
                    });
                    have_part = false;
                }
                pending_combinator = Some(Some(Combinator::SubsequentSibling));
                i += 1;
            }
            CssToken::Delim('*') => {
                current.element = None;
                have_part = true;
                i += 1;
            }
            CssToken::Ident(name) => {
                current.element = Some(name.to_ascii_lowercase());
                have_part = true;
                i += 1;
            }
            // @keyframes step selectors (`0%`, `50%`, `100%`, and bare `0`) are
            // tokenized as Percent/Number. Real selectors never contain a
            // top-level percentage/number, so stash it as the element name for
            // keyframe_offset_from to recover (it strips the trailing `%`).
            // Without this, percentage keyframes parse to ZERO steps and CSS
            // animations using them never run (only from/to worked).
            CssToken::Percent(p) => {
                current.element = Some(format!("{p}%"));
                have_part = true;
                i += 1;
            }
            CssToken::Number(n) => {
                current.element = Some(format!("{n}%"));
                have_part = true;
                i += 1;
            }
            CssToken::Hash(name) => {
                current.id = Some(name.clone());
                have_part = true;
                i += 1;
            }
            CssToken::Delim('.') => {
                // followed by ident -> class
                if let Some(CssToken::Ident(name)) = toks.get(i + 1) {
                    current.classes.push(name.clone());
                    have_part = true;
                    i += 2;
                } else {
                    i += 1;
                }
            }
            CssToken::Colon => {
                // `:` or `::` introduces a pseudo. The modern syntax
                // uses `::name` for pseudo-elements; legacy CSS2 had
                // single-colon `:before` / `:after` etc. We recognise
                // both. The known pseudo-elements live on
                // `pseudo_element` (compound matches the GENERATED box,
                // not the host); everything else is a pseudo-class.
                let is_double_colon = matches!(toks.get(i + 1), Some(CssToken::Colon));
                let lookahead = if is_double_colon { i + 2 } else { i + 1 };
                match toks.get(lookahead) {
                    Some(CssToken::Ident(name)) => {
                        let lc = name.to_ascii_lowercase();
                        if is_double_colon || is_pseudo_element_name(&lc) {
                            current.pseudo_element = Some(lc);
                        } else {
                            current.pseudo_classes.push(lc);
                        }
                        have_part = true;
                        i = lookahead + 1;
                    }
                    Some(CssToken::Function(name)) => {
                        let name_lc = name.to_ascii_lowercase();
                        // CSS Selectors 4 §15: :not()/:is()/:has() contribute the
                        // max specificity of their argument list; :where() contributes
                        // zero; :nth-*() is already counted +1 via nth_selectors.
                        // None of those pseudo-class NAMES themselves add to
                        // specificity — only their inner arguments (or An+B count)
                        // do.  Other functional pseudo-classes (:lang(), :dir(),
                        // :matches(), etc.) are plain pseudo-classes → 0-1-0 each,
                        // so their names go into pseudo_classes as normal.
                        let specificity_delegated = matches!(
                            name_lc.as_str(),
                            "not" | "is" | "where" | "has"
                                | "nth-child"
                                | "nth-last-child"
                                | "nth-of-type"
                                | "nth-last-of-type"
                        );
                        // `:lang()` is matched from its captured args below, NOT
                        // as a bare permissive pseudo-class name — pushing "lang"
                        // into pseudo_classes would route it to the `_ => {}`
                        // catch-all and match every element regardless of language.
                        // It still counts as a normal 0-1-0 pseudo for
                        // specificity (handled in specificity_triple).
                        let arg_captured = matches!(name_lc.as_str(), "lang");
                        if !specificity_delegated && !arg_captured {
                            current.pseudo_classes.push(name_lc.clone());
                        }
                        have_part = true;
                        // Capture the function body so we can recurse
                        // for `:not(...)` / `:is(...)` / `:where(...)`.
                        let body_start = i + 2; // past `:` and the function paren
                        let mut depth = 1;
                        i += 2;
                        let mut body_end = i;
                        while i < toks.len() && depth > 0 {
                            match &toks[i] {
                                CssToken::Function(_) | CssToken::LeftParen => depth += 1,
                                CssToken::RightParen => {
                                    depth -= 1;
                                    if depth == 0 {
                                        body_end = i;
                                    }
                                }
                                _ => {}
                            }
                            i += 1;
                        }
                        if matches!(name_lc.as_str(), "not" | "is" | "where" | "has")
                            && body_end > body_start
                        {
                            let inner = parse_selector_list(&toks[body_start..body_end]);
                            match name_lc.as_str() {
                                "not" => current.not_selectors.extend(inner),
                                // `:is(...)` and `:where(...)` match if
                                // any inner selector matches. Stash them
                                // so the matcher can iterate at match
                                // time; specificity for `:where()` would
                                // be zero, but we don't track that yet.
                                "is" => current.is_selectors.extend(inner),
                                "where" => current.where_selectors.extend(inner),
                                "has" => current.has_selectors.extend(inner),
                                _ => {}
                            }
                        } else if matches!(
                            name_lc.as_str(),
                            "nth-child" | "nth-last-child" | "nth-of-type" | "nth-last-of-type"
                        ) && body_end > body_start
                        {
                            // Reassemble the function body as a string so
                            // `parse_an_plus_b` can lex it. We accept
                            // bare idents (odd/even) plus numbers, signs,
                            // and the `n` letter inside an Ident/Dim.
                            // Numbers inside `:nth-child(...)` arrive
                            // with their sign baked in (the tokenizer
                            // consumes `+2` / `-2` as a single Number
                            // with positive/negative value). When we
                            // serialize them back into the An+B parser
                            // we need an EXPLICIT sign character so a
                            // form like `-n+2` doesn't collapse to
                            // `-n2`. Emit `+` for any non-negative
                            // number unless it's the very first token
                            // in the buffer (where a bare integer like
                            // `3` should stay `3`, not `+3`).
                            let mut buf = String::new();
                            for t in &toks[body_start..body_end] {
                                match t {
                                    CssToken::Ident(s) => buf.push_str(s),
                                    CssToken::Number(n) => {
                                        let leading = buf.trim().is_empty();
                                        if !leading && *n >= 0.0 {
                                            buf.push('+');
                                        }
                                        // Emit as integer when the number
                                        // has no fractional part — the
                                        // An+B parser only accepts ints.
                                        if n.fract() == 0.0 {
                                            buf.push_str(&format!("{}", *n as i64));
                                        } else {
                                            buf.push_str(&format!("{n}"));
                                        }
                                    }
                                    CssToken::Dimension { value, unit } => {
                                        let leading = buf.trim().is_empty();
                                        if !leading && *value >= 0.0 {
                                            buf.push('+');
                                        }
                                        if value.fract() == 0.0 {
                                            buf.push_str(&format!("{}{unit}", *value as i64));
                                        } else {
                                            buf.push_str(&format!("{value}{unit}"));
                                        }
                                    }
                                    CssToken::Delim(c) => buf.push(*c),
                                    CssToken::Whitespace => buf.push(' '),
                                    _ => {}
                                }
                            }
                            if let Some(nth) = parse_an_plus_b(&buf) {
                                let kind = match name_lc.as_str() {
                                    "nth-child" => NthKind::Child,
                                    "nth-last-child" => NthKind::LastChild,
                                    "nth-of-type" => NthKind::OfType,
                                    "nth-last-of-type" => NthKind::LastOfType,
                                    _ => unreachable!(),
                                };
                                current.nth_selectors.push((kind, nth));
                            }
                        } else if name_lc == "lang" && body_end > body_start {
                            // `:lang(en)` / `:lang("en")` / `:lang(en, fr)` —
                            // collect each comma-separated language range, lower-
                            // cased with quotes stripped. Bare idents and strings
                            // are both accepted (CSS Selectors L4 §9.1).
                            for t in &toks[body_start..body_end] {
                                match t {
                                    CssToken::Ident(s) | CssToken::String(s) => {
                                        let v = s.trim().to_ascii_lowercase();
                                        if !v.is_empty() {
                                            current.lang_args.push(v);
                                        }
                                    }
                                    _ => {}
                                }
                            }
                        }
                    }
                    _ => i += 1,
                }
            }
            CssToken::LeftBracket => {
                // Walk to the matching RightBracket and parse the body.
                let mut end = i + 1;
                while end < toks.len() && !matches!(toks[end], CssToken::RightBracket) {
                    end += 1;
                }
                if let Some(attr) = parse_attr_selector(&toks[i + 1..end]) {
                    current.attrs.push(attr);
                    have_part = true;
                }
                i = end + 1;
            }
            _ => i += 1, // ignore unsupported
        }
    }
    if have_part {
        parts.push(ComplexPart {
            combinator: pending_combinator.unwrap_or(Some(Combinator::Descendant)),
            compound: current,
        });
    }
    if parts.is_empty() {
        return None;
    }
    Some(Selector { parts })
}

/// Element view that the matcher works against. The DOM crate will impl this
/// for `cv_html::Node`, but the matcher itself is DOM-agnostic.
pub trait ElementView<'a>: Copy {
    fn tag_name(&self) -> Option<&'a str>;
    fn id(&self) -> Option<&'a str>;
    fn has_class(&self, name: &str) -> bool;
    fn parent(&self) -> Option<Self>;
    /// Immediate previous *element* sibling. Used by `+` combinator.
    /// Default `None` keeps old impls compiling (sibling selectors
    /// will just not match).
    fn previous_element_sibling(&self) -> Option<Self> {
        None
    }
    /// Iterator of all previous element siblings in document order
    /// (closest first). Used by `~` (general sibling). Default empty.
    fn all_previous_element_siblings(&self) -> Vec<Self> {
        Vec::new()
    }
    /// Walk descendant elements (depth-first). Used by `:has(...)`.
    /// Default empty.
    fn all_descendants(&self) -> Vec<Self> {
        Vec::new()
    }
    /// Returns true if at least one descendant element matches `sel`, the
    /// relative-selector argument of `:has(...)`. The default delegates to
    /// `all_descendants()`, which is correct for hosts that can return
    /// descendant views cheaply (e.g. flat-index snapshots). Reference-tree
    /// hosts whose `all_descendants` can't carry owned per-descendant parent
    /// chains (and so can't satisfy combinators inside `:has`) override this
    /// to walk their subtree internally instead.
    fn has_descendant_matching(&self, sel: &Selector) -> bool {
        self.all_descendants().iter().any(|d| matches(sel, *d))
    }
    /// Look up an attribute by case-insensitive name. Default returns
    /// `None` so existing impls keep compiling — attribute selectors
    /// just won't match unless overridden.
    fn attr(&self, _name: &str) -> Option<&'a str> {
        None
    }
    /// Resolve this element's content language for `:lang()` matching: the
    /// nearest `lang` (or `xml:lang`) attribute on the element or an ancestor,
    /// lowercased. `None` when no language is declared anywhere on the chain.
    /// The default walks `attr("lang")` up the `parent()` chain, so any host
    /// that implements `attr` + `parent` gets correct `:lang()` behaviour for
    /// free.
    fn lang(&self) -> Option<String> {
        let mut cur = Some(*self);
        while let Some(node) = cur {
            if let Some(v) = node.attr("lang").or_else(|| node.attr("xml:lang")) {
                let v = v.trim();
                if !v.is_empty() {
                    return Some(v.to_ascii_lowercase());
                }
            }
            cur = node.parent();
        }
        None
    }
    /// Whether this element is currently under the pointer. Used to
    /// resolve `:hover`. Default `false` keeps old impls compiling.
    fn is_hovered(&self) -> bool {
        false
    }
    /// Whether this element has keyboard focus. Used to resolve
    /// `:focus`. Default `false`.
    fn is_focused(&self) -> bool {
        false
    }
    /// 1-based index of this element among its element-siblings. Used
    /// to resolve `:nth-child(an+b)` etc. Default 1 keeps structural
    /// pseudos matching optimistically.
    fn nth_child_index(&self) -> u32 {
        1
    }
    /// Total element-sibling count including this element. Used by
    /// `:nth-last-child(...)`. Default 1.
    fn sibling_count(&self) -> u32 {
        1
    }
    /// 1-based index among element-siblings of the same tag name.
    /// Used by `:nth-of-type(...)`. Default 1.
    fn nth_of_type_index(&self) -> u32 {
        1
    }
    /// Count of element-siblings of the same tag name, including this
    /// element. Used by `:nth-last-of-type(...)`. Default 1.
    fn sibling_of_type_count(&self) -> u32 {
        1
    }

    /// True if the host knows whether this element has form-control
    /// state (checked / disabled / required / etc). Defaults to false
    /// so the matcher stays permissive on hosts that don't plumb form
    /// state — `:checked` / `:disabled` rules then apply to anything
    /// that *might* match, which is closer to "best effort" than
    /// silently refusing every rule.
    fn knows_form_state(&self) -> bool {
        false
    }
    /// `:checked` — true for checkboxes/radios/options that are
    /// currently selected. Default false; only consulted when
    /// `knows_form_state()` returns true.
    fn is_form_checked(&self) -> bool {
        false
    }
    /// `:disabled` — true for form controls with the disabled attr or
    /// inside a disabled fieldset.
    fn is_form_disabled(&self) -> bool {
        false
    }
    /// `:required` — true for inputs with the required attribute.
    fn is_form_required(&self) -> bool {
        false
    }
    /// `:read-only` — true for non-editable form controls (also true
    /// for non-input, non-contenteditable elements per spec).
    fn is_form_read_only(&self) -> bool {
        false
    }
    /// `:placeholder-shown` — true when the input is empty and has a
    /// non-empty placeholder attribute.
    fn is_placeholder_shown(&self) -> bool {
        false
    }
    /// `:valid` / `:invalid` constraint validation. True when all
    /// constraints (pattern / type / required) are satisfied.
    fn is_form_valid(&self) -> bool {
        true
    }
}

pub fn matches<'a, E: ElementView<'a>>(sel: &Selector, el: E) -> bool {
    matches_for(sel, el, None)
}

/// Match `sel` against `el` for a specific pseudo-element target.
/// `pseudo = None` matches the host element only; `pseudo = Some("before")`
/// matches rules that target `::before` on `el`.
pub fn matches_for<'a, E: ElementView<'a>>(sel: &Selector, el: E, pseudo: Option<&str>) -> bool {
    let parts = &sel.parts;
    if parts.is_empty() {
        return false;
    }
    // The rightmost compound's pseudo_element decides which target this
    // rule applies to. Mismatch with the caller's intent → no match.
    let rightmost_pseudo = parts[parts.len() - 1].compound.pseudo_element.as_deref();
    if rightmost_pseudo != pseudo {
        return false;
    }
    // Subject (rightmost) must match `el` directly — no ancestor walk.
    if !compound_matches(&parts[parts.len() - 1].compound, el) {
        return false;
    }
    // Walk leftward through the compound parts. The combinator on
    // parts[i+1] dictates how we move from the previously-matched
    // element to a candidate for parts[i].
    //
    // We pass the *current* element (not the parent) into each step
    // because sibling combinators need siblings of the current, not
    // ancestors. The first step is rooted at `el` itself.
    // Match the remaining compounds leftward from `el`, WITH BACKTRACKING.
    if parts.len() < 2 {
        return true;
    }
    solve_parts(parts, parts.len() - 2, el)
}

/// Match `parts[0..=i]` ending at `current` (which already matches
/// `parts[i+1]`), moving leftward via the combinator on `parts[i+1]`. Unlike a
/// committing walk, descendant (` `) and general-sibling (`~`) combinators TRY
/// EACH candidate and recurse — so `.a .b .c` still matches when an intermediate
/// compound has multiple matching ancestors (the first one chosen mustn't dead-
/// end the whole match).
fn solve_parts<'a, E: ElementView<'a>>(parts: &[ComplexPart], i: usize, current: E) -> bool {
    let combinator = parts[i + 1].combinator.clone();
    let next = |cand: E| i == 0 || solve_parts(parts, i - 1, cand);
    match combinator {
        Some(Combinator::Descendant) => {
            let mut cur = current.parent();
            while let Some(n) = cur {
                if compound_matches(&parts[i].compound, n) && next(n) {
                    return true;
                }
                cur = n.parent();
            }
            false
        }
        Some(Combinator::Child) | None => match current.parent() {
            Some(p) => compound_matches(&parts[i].compound, p) && next(p),
            None => false,
        },
        Some(Combinator::NextSibling) => match current.previous_element_sibling() {
            Some(s) => compound_matches(&parts[i].compound, s) && next(s),
            None => false,
        },
        Some(Combinator::SubsequentSibling) => {
            for s in current.all_previous_element_siblings() {
                if compound_matches(&parts[i].compound, s) && next(s) {
                    return true;
                }
            }
            false
        }
    }
}

fn compound_matches<'a, E: ElementView<'a>>(c: &SimpleSelector, e: E) -> bool {
    if let Some(want) = &c.element {
        match e.tag_name() {
            Some(t) if t.eq_ignore_ascii_case(want) => {}
            _ => return false,
        }
    }
    if let Some(want) = &c.id {
        match e.id() {
            Some(actual) if actual == want => {}
            _ => return false,
        }
    }
    for class in &c.classes {
        if !e.has_class(class) {
            return false;
        }
    }
    for attr in &c.attrs {
        let actual = e.attr(&attr.name);
        if !match_attr(actual, attr) {
            return false;
        }
    }
    // `:lang(range, ...)` — match the element's resolved content language
    // against each declared BCP47 range by the prefix rule: a range matches
    // when the element's language equals it OR begins with `range-` (so
    // `:lang(en)` matches `en` and `en-US`). When the element has no declared
    // language at all, `:lang()` does NOT match (CSS Selectors L4 §9.1) — this
    // is what stops e.g. `h1:lang(ckb){font-family:Scheherazade}` from being
    // applied to every heading on a non-Kurdish page.
    if !c.lang_args.is_empty() {
        let el_lang = e.lang();
        let matched = match &el_lang {
            Some(lang) => c.lang_args.iter().any(|range| {
                lang == range || lang.starts_with(&format!("{range}-"))
            }),
            None => false,
        };
        if !matched {
            return false;
        }
    }
    // `:is(sel-list)` / `:where(sel-list)` — at least one inner must
    // match. We flatten the list during parse (so `:is(a, b):is(c)`
    // becomes a single `[a, b, c]` bucket), losing the "AND across
    // multiple :is()" distinction; for V1 this just over-applies in
    // the rare double-:is case, never under-applies.
    if !c.is_selectors.is_empty() {
        let any = c.is_selectors.iter().any(|s| matches(s, e));
        if !any {
            return false;
        }
    }
    // `:where(sel, ...)` — same matching semantics as `:is()`. Specificity
    // differs (zero for `:where()`), handled in `specificity_triple`.
    if !c.where_selectors.is_empty() {
        let any = c.where_selectors.iter().any(|s| matches(s, e));
        if !any {
            return false;
        }
    }
    // `:not(sel)` — if any inner selector matches, the compound fails.
    for n in &c.not_selectors {
        if matches(n, e) {
            return false;
        }
    }
    // `:nth-child(An+B)` and variants. Compute the right position
    // (forward vs backward, all siblings vs same-tag) and reject the
    // element if any nth_selectors does not match.
    for (kind, arg) in &c.nth_selectors {
        let index: i32 = match kind {
            NthKind::Child => e.nth_child_index() as i32,
            NthKind::LastChild => (e.sibling_count() as i32 + 1) - (e.nth_child_index() as i32),
            NthKind::OfType => e.nth_of_type_index() as i32,
            NthKind::LastOfType => {
                (e.sibling_of_type_count() as i32 + 1) - (e.nth_of_type_index() as i32)
            }
        };
        if !arg.matches(index) {
            return false;
        }
    }
    // `:has(sel)` — at least one descendant must match `sel`. The
    // descendant iterator is provided by ElementView; default impl
    // returns nothing, so unmodified hosts simply never match `:has`.
    if !c.has_selectors.is_empty() {
        let any_has_matches = c.has_selectors.iter().any(|s| e.has_descendant_matching(s));
        if !any_has_matches {
            return false;
        }
    }
    // Pseudo-classes. State-dependent ones (`:hover`, `:focus`, etc.)
    // can't match in a static render — we refuse them so rules guarded
    // by `:hover` don't bleed into the default style. Structural ones
    // we don't have DOM-position tracking for yet, so we let them match
    // optimistically — better to over-apply than to lose all rules
    // qualified by `:first-child` etc.
    for pc in &c.pseudo_classes {
        let lc = pc.to_ascii_lowercase();
        match lc.as_str() {
            "hover" => {
                if !e.is_hovered() {
                    return false;
                }
            }
            "focus" | "focus-within" | "focus-visible" => {
                if !e.is_focused() {
                    return false;
                }
            }
            // Structural pseudo-classes — drive from ElementView's
            // sibling-position trait methods. Defaults (1/1) keep
            // these matching for hosts with no position plumbing,
            // matching previous behaviour, but plumbed builders get
            // proper Selectors-L4 semantics.
            "first-child" => {
                if e.nth_child_index() != 1 {
                    return false;
                }
            }
            "last-child" => {
                if e.nth_child_index() != e.sibling_count() {
                    return false;
                }
            }
            "only-child" => {
                if e.sibling_count() != 1 {
                    return false;
                }
            }
            "first-of-type" => {
                // Match the first element of its tag among its siblings,
                // NOT the first child overall — that's :first-child. The
                // previous code used nth_child_index() which broke
                // `li:first-of-type` after preceding non-li siblings.
                if e.nth_of_type_index() != 1 {
                    return false;
                }
            }
            "last-of-type" => {
                if e.nth_of_type_index() != e.sibling_of_type_count() {
                    return false;
                }
            }
            "only-of-type" => {
                if e.sibling_of_type_count() != 1 {
                    return false;
                }
            }
            "only-of-type" => {
                if e.sibling_count() != 1 {
                    return false;
                }
            }
            "empty" => {
                // `:empty` matches when the element has no children at
                // all — including text. ElementView doesn't currently
                // expose a child count; assume non-empty (refuse) so
                // rules guarded by `:empty` don't over-apply.
                return false;
            }
            "root" => {
                if e.parent().is_some() {
                    return false;
                }
            }
            // Anchor pseudo-classes: in this engine all <a href=…>
            // elements are "unvisited links" (no history-state).
            // `:any-link` covers both.  Refuse `:visited` (no history),
            // accept `:link` and `:any-link` since matching them just
            // means "is a link" — best-effort without href tracking.
            "any-link" | "link" => {}
            "visited" => return false,
            // Form-control state pseudos. ElementView doesn't yet
            // expose form-control state, but rules guarded by these
            // are author-side opt-ins — letting them match is closer
            // to "best effort" than refusing every rule with `:checked`.
            "checked" | "enabled" | "default" => {
                // Drive from ElementView when the host exposes form
                // state; otherwise fall back to "permissive" (the rule
                // still applies, matching pre-plumbed behaviour).
                if !e.is_form_checked() && e.knows_form_state() {
                    return false;
                }
            }
            "disabled" => {
                // Refuse only when the host explicitly says the element
                // is enabled. Default unknown = "could be disabled" so
                // the rule still applies for legacy hosts.
                if e.knows_form_state() && !e.is_form_disabled() {
                    return false;
                }
            }
            "required" => {
                if e.knows_form_state() && !e.is_form_required() {
                    return false;
                }
            }
            "optional" => {
                if e.knows_form_state() && e.is_form_required() {
                    return false;
                }
            }
            "read-only" => {
                if e.knows_form_state() && !e.is_form_read_only() {
                    return false;
                }
            }
            "read-write" => {
                if e.knows_form_state() && e.is_form_read_only() {
                    return false;
                }
            }
            "placeholder-shown" => {
                if e.knows_form_state() && !e.is_placeholder_shown() {
                    return false;
                }
            }
            "valid" => {
                if e.knows_form_state() && !e.is_form_valid() {
                    return false;
                }
            }
            "invalid" | "user-invalid" => {
                if e.knows_form_state() && e.is_form_valid() {
                    return false;
                }
            }
            "user-valid" => {
                if e.knows_form_state() && !e.is_form_valid() {
                    return false;
                }
            }
            // Range / indeterminate / target / active stay refused — they
            // need richer plumbing (range inputs, ::active mouse state,
            // URL fragment match) we don't yet thread through.
            "indeterminate" | "in-range" | "out-of-range" | "target" | "active" => return false,
            // Pseudo-elements were previously folded in here as a
            // "always refuse" guard. With the new `pseudo_element`
            // field on SimpleSelector the matcher now decides at the
            // top of `matches()` whether the compound's pseudo-element
            // (if any) is what the caller asked about. So nothing
            // pseudo-element-shaped reaches this point — the legacy
            // `:before` syntax routes through the pseudo_element bucket
            // before the pseudo_classes vec is touched.
            _ => {}
        }
    }
    true
}

/// Parse the body of `[ ... ]`. Accepted shapes:
///   - `name`
///   - `name = value`        (Equals)
///   - `name ~= value`       (Includes)
///   - `name |= value`       (DashMatch)
///   - `name ^= value`       (Prefix)
///   - `name $= value`       (Suffix)
///   - `name *= value`       (Substring)
/// Both bare-ident and quoted string values are accepted.
fn parse_attr_selector(body: &[CssToken]) -> Option<AttrSelector> {
    // Skip leading whitespace.
    let mut i = 0;
    while i < body.len() && matches!(body[i], CssToken::Whitespace) {
        i += 1;
    }
    let name = match body.get(i)? {
        CssToken::Ident(n) => n.to_ascii_lowercase(),
        _ => return None,
    };
    i += 1;
    while i < body.len() && matches!(body[i], CssToken::Whitespace) {
        i += 1;
    }
    if i >= body.len() {
        return Some(AttrSelector {
            name,
            op: AttrOp::Exists,
            value: String::new(),
        });
    }
    // Look for the operator. `=` alone is Equals; preceded by ~ | ^ $ * is the modifier.
    let (op, advance) = match (&body[i], body.get(i + 1)) {
        (CssToken::Delim('~'), Some(CssToken::Delim('='))) => (AttrOp::Includes, 2),
        (CssToken::Delim('|'), Some(CssToken::Delim('='))) => (AttrOp::DashMatch, 2),
        (CssToken::Delim('^'), Some(CssToken::Delim('='))) => (AttrOp::Prefix, 2),
        (CssToken::Delim('$'), Some(CssToken::Delim('='))) => (AttrOp::Suffix, 2),
        (CssToken::Delim('*'), Some(CssToken::Delim('='))) => (AttrOp::Substring, 2),
        (CssToken::Delim('='), _) => (AttrOp::Equals, 1),
        _ => return None,
    };
    i += advance;
    while i < body.len() && matches!(body[i], CssToken::Whitespace) {
        i += 1;
    }
    let value = match body.get(i)? {
        CssToken::String(s) => s.clone(),
        CssToken::Ident(s) => s.clone(),
        CssToken::Number(n) => n.to_string(),
        _ => return None,
    };
    Some(AttrSelector { name, op, value })
}

fn match_attr(actual: Option<&str>, sel: &AttrSelector) -> bool {
    match (sel.op, actual) {
        (AttrOp::Exists, Some(_)) => true,
        (AttrOp::Exists, None) => false,
        (_, None) => false,
        (AttrOp::Equals, Some(v)) => v == sel.value,
        (AttrOp::Includes, Some(v)) => v.split_ascii_whitespace().any(|t| t == sel.value),
        (AttrOp::DashMatch, Some(v)) => {
            v == sel.value
                || (v.len() > sel.value.len()
                    && v.starts_with(&sel.value)
                    && v.as_bytes().get(sel.value.len()) == Some(&b'-'))
        }
        (AttrOp::Prefix, Some(v)) => !sel.value.is_empty() && v.starts_with(&sel.value),
        (AttrOp::Suffix, Some(v)) => !sel.value.is_empty() && v.ends_with(&sel.value),
        (AttrOp::Substring, Some(v)) => !sel.value.is_empty() && v.contains(&sel.value),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokenizer::tokenize;

    fn one(src: &str) -> Selector {
        let toks = tokenize(src);
        // strip eof
        let toks: Vec<_> = toks
            .into_iter()
            .filter(|t| !matches!(t, CssToken::Eof))
            .collect();
        parse_selector_list(&toks).into_iter().next().unwrap()
    }

    #[test]
    fn type_class_id() {
        let s = one("div.note#main");
        let c = &s.parts[0].compound;
        assert_eq!(c.element.as_deref(), Some("div"));
        assert_eq!(c.id.as_deref(), Some("main"));
        assert_eq!(c.classes, vec!["note".to_string()]);
    }

    #[test]
    fn descendant_and_child() {
        let s = one("article p > a");
        assert_eq!(s.parts.len(), 3);
        assert_eq!(s.parts[0].combinator, None);
        assert_eq!(s.parts[1].combinator, Some(Combinator::Descendant));
        assert_eq!(s.parts[2].combinator, Some(Combinator::Child));
    }

    #[test]
    fn specificity_ordering() {
        let a = one("p");
        let b = one(".foo");
        let c = one("#bar");
        let d = one(".x.y");
        assert!(a.specificity() < b.specificity());
        assert!(b.specificity() < c.specificity());
        assert!(d.specificity() > b.specificity());
        assert!(d.specificity() < c.specificity());
    }

    /// CSS Selectors 4 §15: functional pseudo-class NAMES must NOT
    /// inflate specificity — only their inner arguments count.
    ///
    ///   :not(.foo)    → 0-1-0  (same as .foo alone)
    ///   :is(.foo)     → 0-1-0  (same as .foo alone)
    ///   :where(.foo)  → 0-0-0  (:where always contributes zero)
    ///   :has(.foo)    → 0-1-0  (same as .foo alone)
    ///   :nth-child(2) → 0-1-0  (one pseudo-class, not two)
    ///   :not(#id)     → 1-0-0  (same as #id alone)
    ///
    /// Previously the pseudo-class NAME was pushed into `pseudo_classes`
    /// *and* the inner max was added separately, double-counting every
    /// functional pseudo → broke normalize/Bootstrap/Tailwind cascades.
    #[test]
    fn functional_pseudo_specificity_not_double_counted() {
        let class_spec = one(".foo").specificity();   // 0-1-0

        // :not(.foo) must equal .foo  (0-1-0), not 0-2-0
        assert_eq!(one(":not(.foo)").specificity(), class_spec,
            ":not(.foo) specificity should equal .foo (0-1-0)");

        // :is(.foo) must equal .foo  (0-1-0)
        assert_eq!(one(":is(.foo)").specificity(), class_spec,
            ":is(.foo) specificity should equal .foo (0-1-0)");

        // :where(.foo) must be zero regardless of inner selector
        assert_eq!(one(":where(.foo)").specificity(), 0,
            ":where() always contributes zero specificity");

        // :has(.foo) must equal .foo  (0-1-0)
        assert_eq!(one(":has(.foo)").specificity(), class_spec,
            ":has(.foo) specificity should equal .foo (0-1-0)");

        // :nth-child(2) must equal one plain pseudo-class  (same as .foo = 0-1-0)
        assert_eq!(one(":nth-child(2)").specificity(), class_spec,
            ":nth-child(2) specificity should be 0-1-0, not 0-2-0");

        // :not(#id) must equal #id  (1-0-0)
        let id_spec = one("#myid").specificity();
        assert_eq!(one(":not(#myid)").specificity(), id_spec,
            ":not(#id) specificity should equal #id (1-0-0)");

        // Compound: div:not(.foo) must equal div.foo  (0-1-1)
        assert_eq!(one("div:not(.foo)").specificity(), one("div.foo").specificity(),
            "div:not(.foo) specificity should equal div.foo (0-1-1)");

        // :lang() is a plain functional pseudo → still counts as 0-1-0
        assert_eq!(one(":lang(en)").specificity(), class_spec,
            ":lang(en) should still be 0-1-0 (plain functional pseudo)");
    }

    /// Regression: `:not(.a)` specificity must be exactly (0,1,0).
    ///
    /// Before the fix the pseudo-class NAME was pushed onto `pseudo_classes`
    /// AND the inner-arg max was added, so `:not(.a)` = (0,2,0).  The fix
    /// gates the name-push on `!specificity_delegated`.
    #[test]
    fn not_dot_a_specificity_is_0_1_0() {
        // (0,1,0) packed = (0 << 16) | (1 << 8) | 0 = 256
        let expected: u32 = 0x00_01_00;
        assert_eq!(
            one(":not(.a)").specificity(),
            expected,
            ":not(.a) must be (0,1,0) = 0x000100, not (0,2,0) = 0x000200"
        );
    }

    /// Named regression test: `:not(.a)` = (0,1,0), `:where(.a)` = (0,0,0).
    ///
    /// Functional pseudo-class names MUST NOT add to specificity.
    /// `:not`/`:is`/`:has` contribute the max specificity of their inner
    /// argument list; `:where` always contributes zero.
    #[test]
    fn pseudo_class_specificity_no_double_count() {
        // :not(.a) — functional pseudo name does NOT add 1; only the
        // inner argument (.a = 0-1-0) counts.  Must equal (0,1,0).
        let not_a = one(":not(.a)").specificity();
        let class_a = one(".a").specificity();
        assert_eq!(
            not_a, class_a,
            ":not(.a) must equal .a specificity (0,1,0); functional name must not double-count"
        );

        // :where(.a) — contributes ZERO even though argument is .a (0-1-0).
        assert_eq!(
            one(":where(.a)").specificity(),
            0,
            ":where(.a) must be (0,0,0) — :where() always contributes zero specificity"
        );
    }

    /// Lightweight element view for attribute-selector tests.
    #[derive(Copy, Clone)]
    struct AttrEl<'a> {
        tag: &'a str,
        attrs: &'a [(&'a str, &'a str)],
    }
    impl<'a> ElementView<'a> for AttrEl<'a> {
        fn tag_name(&self) -> Option<&'a str> {
            Some(self.tag)
        }
        fn id(&self) -> Option<&'a str> {
            None
        }
        fn has_class(&self, _: &str) -> bool {
            false
        }
        fn parent(&self) -> Option<Self> {
            None
        }
        fn attr(&self, n: &str) -> Option<&'a str> {
            let lc = n.to_ascii_lowercase();
            self.attrs
                .iter()
                .find(|(k, _)| k.eq_ignore_ascii_case(&lc))
                .map(|(_, v)| *v)
        }
    }

    #[test]
    fn attr_selector_exists() {
        let s = one("a[href]");
        let c = &s.parts[0].compound;
        assert_eq!(c.attrs.len(), 1);
        assert_eq!(c.attrs[0].op, AttrOp::Exists);
        let yes = AttrEl {
            tag: "a",
            attrs: &[("href", "/x")],
        };
        let no = AttrEl {
            tag: "a",
            attrs: &[],
        };
        assert!(matches(&s, yes));
        assert!(!matches(&s, no));
    }

    #[test]
    fn attr_selector_equals_quoted() {
        let s = one("input[type=\"text\"]");
        let yes = AttrEl {
            tag: "input",
            attrs: &[("type", "text")],
        };
        let no = AttrEl {
            tag: "input",
            attrs: &[("type", "checkbox")],
        };
        assert!(matches(&s, yes));
        assert!(!matches(&s, no));
    }

    #[test]
    fn attr_selector_includes_and_prefix() {
        let inc = one("a[class~=\"hi\"]");
        let pfx = one("a[href^=\"https\"]");
        let el = AttrEl {
            tag: "a",
            attrs: &[("class", "hello hi there"), ("href", "https://example.com")],
        };
        assert!(matches(&inc, el));
        assert!(matches(&pfx, el));
    }

    #[test]
    fn attr_selector_substring_and_suffix() {
        let sub = one("a[href*=\"example\"]");
        let sfx = one("a[href$=\".com\"]");
        let el = AttrEl {
            tag: "a",
            attrs: &[("href", "https://www.example.com")],
        };
        assert!(matches(&sub, el));
        assert!(matches(&sfx, el));
    }

    #[test]
    fn hover_pseudo_matches_when_view_says_hovered() {
        #[derive(Copy, Clone)]
        struct HoveredAnchor;
        impl<'a> ElementView<'a> for HoveredAnchor {
            fn tag_name(&self) -> Option<&'a str> {
                Some("a")
            }
            fn id(&self) -> Option<&'a str> {
                None
            }
            fn has_class(&self, _: &str) -> bool {
                false
            }
            fn parent(&self) -> Option<Self> {
                None
            }
            fn is_hovered(&self) -> bool {
                true
            }
        }
        let s = one("a:hover");
        assert!(matches(&s, HoveredAnchor));
    }

    #[test]
    fn not_excludes_matching_compound() {
        #[derive(Copy, Clone)]
        struct E<'a> {
            tag: &'a str,
            classes: &'a [&'a str],
        }
        impl<'a> ElementView<'a> for E<'a> {
            fn tag_name(&self) -> Option<&'a str> {
                Some(self.tag)
            }
            fn id(&self) -> Option<&'a str> {
                None
            }
            fn has_class(&self, n: &str) -> bool {
                self.classes.iter().any(|c| *c == n)
            }
            fn parent(&self) -> Option<Self> {
                None
            }
        }
        let sel = one("div:not(.skip)");
        let plain = E {
            tag: "div",
            classes: &[],
        };
        let skip = E {
            tag: "div",
            classes: &["skip"],
        };
        assert!(matches(&sel, plain));
        assert!(!matches(&sel, skip));

        // `:not(.a, .b)` — neither.
        let sel2 = one("p:not(.a, .b)");
        let none = E {
            tag: "p",
            classes: &[],
        };
        let with_a = E {
            tag: "p",
            classes: &["a"],
        };
        let with_b = E {
            tag: "p",
            classes: &["b"],
        };
        assert!(matches(&sel2, none));
        assert!(!matches(&sel2, with_a));
        assert!(!matches(&sel2, with_b));
    }

    #[test]
    fn pseudo_element_does_not_match_host_element() {
        // `p:before` and `p::before` target a generated box, not the
        // <p> itself. Real browsers never apply their declarations to
        // the host element. Wikipedia's `p:before { width: 120pt }`
        // had been silently bleeding onto every paragraph through our
        // "structural — assume match" fallback, collapsing article
        // bodies to 160px-wide ribbons.
        #[derive(Copy, Clone)]
        struct E<'a> {
            tag: &'a str,
        }
        impl<'a> ElementView<'a> for E<'a> {
            fn tag_name(&self) -> Option<&'a str> {
                Some(self.tag)
            }
            fn id(&self) -> Option<&'a str> {
                None
            }
            fn has_class(&self, _n: &str) -> bool {
                false
            }
            fn parent(&self) -> Option<Self> {
                None
            }
        }
        let el = E { tag: "p" };
        assert!(!matches(&one("p:before"), el));
        assert!(!matches(&one("p:after"), el));
        assert!(!matches(&one("p::before"), el));
        assert!(!matches(&one("p::after"), el));
        assert!(!matches(&one("p::first-line"), el));
        assert!(!matches(&one("p::first-letter"), el));
        // Bare `p` still matches.
        assert!(matches(&one("p"), el));
    }

    #[test]
    fn parse_an_plus_b_covers_spec_forms() {
        assert_eq!(parse_an_plus_b("odd"), Some(NthArg { a: 2, b: 1 }));
        assert_eq!(parse_an_plus_b("even"), Some(NthArg { a: 2, b: 0 }));
        assert_eq!(parse_an_plus_b("0"), Some(NthArg { a: 0, b: 0 }));
        assert_eq!(parse_an_plus_b("5"), Some(NthArg { a: 0, b: 5 }));
        assert_eq!(parse_an_plus_b("n"), Some(NthArg { a: 1, b: 0 }));
        assert_eq!(parse_an_plus_b("2n"), Some(NthArg { a: 2, b: 0 }));
        assert_eq!(parse_an_plus_b("2n+1"), Some(NthArg { a: 2, b: 1 }));
        assert_eq!(parse_an_plus_b("2n-1"), Some(NthArg { a: 2, b: -1 }));
        assert_eq!(parse_an_plus_b("-n+3"), Some(NthArg { a: -1, b: 3 }));
        // Spaces around the +/- are tolerated.
        assert_eq!(parse_an_plus_b("2n + 1"), Some(NthArg { a: 2, b: 1 }));
        assert_eq!(parse_an_plus_b("-1n+3"), Some(NthArg { a: -1, b: 3 }));
    }

    #[test]
    fn nth_arg_matches_correctly() {
        // 2n+1 (odd): matches 1, 3, 5, 7, ...
        let odd = parse_an_plus_b("odd").unwrap();
        assert!(odd.matches(1));
        assert!(!odd.matches(2));
        assert!(odd.matches(3));
        assert!(!odd.matches(4));
        assert!(odd.matches(5));

        // 2n (even): matches 2, 4, 6, ...
        let even = parse_an_plus_b("even").unwrap();
        assert!(!even.matches(1));
        assert!(even.matches(2));
        assert!(!even.matches(3));
        assert!(even.matches(4));

        // 3n+1: 1, 4, 7, ...
        let n31 = parse_an_plus_b("3n+1").unwrap();
        assert!(n31.matches(1));
        assert!(!n31.matches(2));
        assert!(!n31.matches(3));
        assert!(n31.matches(4));
        assert!(n31.matches(7));

        // -n+3: 1, 2, 3 only (CSS Selectors L4 §6.7).
        let m_n3 = parse_an_plus_b("-n+3").unwrap();
        assert!(m_n3.matches(1));
        assert!(m_n3.matches(2));
        assert!(m_n3.matches(3));
        assert!(!m_n3.matches(4));
        assert!(!m_n3.matches(5));

        // Plain integer N: matches index == N exactly.
        let three = parse_an_plus_b("3").unwrap();
        assert!(!three.matches(2));
        assert!(three.matches(3));
        assert!(!three.matches(4));
    }

    #[test]
    fn nth_child_pseudo_matches_via_view() {
        #[derive(Copy, Clone)]
        struct E<'a> {
            tag: &'a str,
            idx: u32,
            total: u32,
        }
        impl<'a> ElementView<'a> for E<'a> {
            fn tag_name(&self) -> Option<&'a str> {
                Some(self.tag)
            }
            fn id(&self) -> Option<&'a str> {
                None
            }
            fn has_class(&self, _n: &str) -> bool {
                false
            }
            fn parent(&self) -> Option<Self> {
                None
            }
            fn nth_child_index(&self) -> u32 {
                self.idx
            }
            fn sibling_count(&self) -> u32 {
                self.total
            }
            fn nth_of_type_index(&self) -> u32 {
                self.idx
            }
            fn sibling_of_type_count(&self) -> u32 {
                self.total
            }
        }

        let odd = one("li:nth-child(odd)");
        assert!(matches(
            &odd,
            E {
                tag: "li",
                idx: 1,
                total: 5
            }
        ));
        assert!(!matches(
            &odd,
            E {
                tag: "li",
                idx: 2,
                total: 5
            }
        ));
        assert!(matches(
            &odd,
            E {
                tag: "li",
                idx: 3,
                total: 5
            }
        ));
        assert!(!matches(
            &odd,
            E {
                tag: "li",
                idx: 4,
                total: 5
            }
        ));

        let third = one("li:nth-child(3)");
        assert!(!matches(
            &third,
            E {
                tag: "li",
                idx: 1,
                total: 5
            }
        ));
        assert!(!matches(
            &third,
            E {
                tag: "li",
                idx: 2,
                total: 5
            }
        ));
        assert!(matches(
            &third,
            E {
                tag: "li",
                idx: 3,
                total: 5
            }
        ));

        // :nth-last-child(1) is the last in the list.
        let last = one("li:nth-last-child(1)");
        assert!(!matches(
            &last,
            E {
                tag: "li",
                idx: 1,
                total: 5
            }
        ));
        assert!(matches(
            &last,
            E {
                tag: "li",
                idx: 5,
                total: 5
            }
        ));

        // -n+2 selects only the first two.
        let first_two = one("li:nth-child(-n+2)");
        assert!(matches(
            &first_two,
            E {
                tag: "li",
                idx: 1,
                total: 5
            }
        ));
        assert!(matches(
            &first_two,
            E {
                tag: "li",
                idx: 2,
                total: 5
            }
        ));
        assert!(!matches(
            &first_two,
            E {
                tag: "li",
                idx: 3,
                total: 5
            }
        ));
    }

    #[test]
    fn is_pseudo_matches_any_inner() {
        #[derive(Copy, Clone)]
        struct E<'a> {
            tag: &'a str,
        }
        impl<'a> ElementView<'a> for E<'a> {
            fn tag_name(&self) -> Option<&'a str> {
                Some(self.tag)
            }
            fn id(&self) -> Option<&'a str> {
                None
            }
            fn has_class(&self, _n: &str) -> bool {
                false
            }
            fn parent(&self) -> Option<Self> {
                None
            }
        }
        // `:is(h1, h2, h3)` matches an h2 element but NOT a p.
        let sel = one(":is(h1, h2, h3)");
        assert!(matches(&sel, E { tag: "h1" }));
        assert!(matches(&sel, E { tag: "h2" }));
        assert!(matches(&sel, E { tag: "h3" }));
        assert!(!matches(&sel, E { tag: "p" }));
    }

    #[test]
    fn where_pseudo_matches_any_inner() {
        #[derive(Copy, Clone)]
        struct E<'a> {
            tag: &'a str,
        }
        impl<'a> ElementView<'a> for E<'a> {
            fn tag_name(&self) -> Option<&'a str> {
                Some(self.tag)
            }
            fn id(&self) -> Option<&'a str> {
                None
            }
            fn has_class(&self, _n: &str) -> bool {
                false
            }
            fn parent(&self) -> Option<Self> {
                None
            }
        }
        // `:where()` shares matching semantics with `:is()`; only
        // specificity differs (which we don't track yet).
        let sel = one(":where(a, button)");
        assert!(matches(&sel, E { tag: "a" }));
        assert!(matches(&sel, E { tag: "button" }));
        assert!(!matches(&sel, E { tag: "div" }));
    }

    #[test]
    fn stateful_pseudo_never_matches() {
        let s = one("a:hover");
        let el = AttrEl {
            tag: "a",
            attrs: &[],
        };
        assert!(!matches(&s, el));
    }

    #[test]
    fn descendant_combinator_backtracks_when_first_ancestor_match_dead_ends() {
        // The audit flagged "descendant combinator backtracking". Verify
        // it actually backtracks: for `.a > .b .c` matched on
        //   <div class="a">
        //     <div class="b">    <!-- outer .b -->
        //       <div class="b">  <!-- inner .b — direct parent of target -->
        //         <div class="c"></div>  <-- target
        //       </div>
        //     </div>
        //   </div>
        // walking from the target's parent chain looking for `.b` (descendant
        // combinator) hits the INNER .b first. The next step requires the
        // PARENT of that .b to be `.a` (child combinator). The inner .b's
        // parent is the outer .b — not .a. A non-backtracking matcher would
        // commit to the inner .b and FAIL. The correct matcher continues
        // walking the parent chain to the outer .b, whose parent IS .a,
        // and the whole chain succeeds.
        //
        // This same shape appears in real Tailwind/Bootstrap utility CSS
        // where `.modal > .modal-body .modal-form` etc. nests classes
        // whose names are reused at multiple levels.
        #[derive(Copy, Clone)]
        struct Node<'a> {
            tag: &'a str,
            classes: &'a [&'a str],
            parent: Option<&'a Node<'a>>,
        }
        impl<'a> ElementView<'a> for Node<'a> {
            fn tag_name(&self) -> Option<&'a str> {
                Some(self.tag)
            }
            fn id(&self) -> Option<&'a str> {
                None
            }
            fn has_class(&self, n: &str) -> bool {
                self.classes.iter().any(|c| *c == n)
            }
            fn parent(&self) -> Option<Self> {
                self.parent.copied()
            }
        }
        let a = Node {
            tag: "div",
            classes: &["a"],
            parent: None,
        };
        let outer_b = Node {
            tag: "div",
            classes: &["b"],
            parent: Some(&a),
        };
        let inner_b = Node {
            tag: "div",
            classes: &["b"],
            parent: Some(&outer_b),
        };
        let target_c = Node {
            tag: "div",
            classes: &["c"],
            parent: Some(&inner_b),
        };
        let sel = one(".a > .b .c");
        assert!(
            matches(&sel, target_c),
            ".a > .b .c must match: inner .b dead-ends (its parent is outer .b, not .a), but outer .b satisfies the > .a step — requires walking the full ancestor chain"
        );
        // Negative control: `.a > .x .c` shouldn't match (no .x ancestor).
        let sel_neg = one(".a > .x .c");
        assert!(!matches(&sel_neg, target_c));
    }

    #[test]
    fn structural_pseudo_matches_optimistically() {
        // Without per-element index tracking we let structural pseudos
        // through — better to over-apply than to drop the rule entirely.
        let s = one("li:first-child");
        let el = AttrEl {
            tag: "li",
            attrs: &[],
        };
        assert!(matches(&s, el));
        let s = one("li:nth-child(odd)");
        assert!(matches(&s, el));
    }
}
