//! Modern CSS at-rules — @container, @layer, @scope, scroll-driven animation.
//!
//! V1 covers parsing + cascade-time recognition. Each at-rule has a
//! simple data model the cascade reads at compute time. Subgrid for
//! `display: grid` is exposed as a bool on the existing grid pipeline
//! (see `cv_layout`); the parsing surface here flags it so the
//! cascade can carry it forward.

/// `@container <name>? <condition>` block.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContainerRule {
    pub name: Option<String>,
    /// Raw condition string — `(min-width: 600px)` or `(inline-size > 30em)`.
    pub condition_raw: String,
    pub body_raw: String,
}

/// `@layer A, B, C;` or `@layer A { ... }`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LayerRule {
    pub names: Vec<String>,
    /// Some(body) for the block form, None for the statement form.
    pub body_raw: Option<String>,
}

/// `@scope (start) to (end)` — both arguments optional.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScopeRule {
    pub start_selector: Option<String>,
    pub end_selector: Option<String>,
    pub body_raw: String,
}

/// `@starting-style { selector { ... } }` — the body is parsed exactly
/// like a normal stylesheet but the rules only apply when an element is
/// transitioning *into* the page (first style resolution). V1 captures
/// the body so the cascade can sample initial values for animations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StartingStyleRule {
    pub body_raw: String,
}

/// `@property --name { syntax: "<type>"; initial-value: ...; inherits: ...; }`.
/// Lets stylesheets register a custom property with a typed initial
/// value, so cascade can resolve `var(--name)` against that fallback
/// when the property hasn't been set on an ancestor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PropertyRule {
    pub name: String,
    pub syntax: Option<String>,
    pub inherits: bool,
    pub initial_value: Option<String>,
}

/// Parse `@container (...) { ... }` or `@container <name> (...) { ... }`.
/// `input` is the source text *after* the `@container` keyword.
pub fn parse_container(input: &str) -> Option<ContainerRule> {
    let s = input.trim_start();
    // Optional name before the condition.
    let (name, rest) = take_optional_name(s);
    let s = rest.trim_start();
    let (condition, body) = take_paren_and_body(s)?;
    Some(ContainerRule {
        name,
        condition_raw: condition,
        body_raw: body,
    })
}

/// Parse `@layer A, B, C;` (statement) or `@layer A { ... }` (block).
pub fn parse_layer(input: &str) -> Option<LayerRule> {
    let s = input.trim_start();
    if let Some(stmt_end) = s.find(';') {
        let names_part = &s[..stmt_end];
        // Bail out if the statement contains a `{` — that means we
        // actually have a block form and a stray `;`.
        if names_part.contains('{') {
            // fall through to block form
        } else {
            let names: Vec<String> = names_part
                .split(',')
                .map(|p| p.trim().to_string())
                .filter(|p| !p.is_empty())
                .collect();
            return Some(LayerRule {
                names,
                body_raw: None,
            });
        }
    }
    // Block form: optional single name then body.
    let (name, rest) = take_optional_name(s);
    let names: Vec<String> = name.into_iter().collect();
    let body = take_body(rest.trim_start())?;
    Some(LayerRule {
        names,
        body_raw: Some(body),
    })
}

/// Parse `@scope (start) to (end) { ... }`.
pub fn parse_scope(input: &str) -> Option<ScopeRule> {
    let s = input.trim_start();
    let (start_selector, after_start) = if let Some(stripped) = s.strip_prefix('(') {
        let end = stripped.find(')')?;
        (
            Some(stripped[..end].trim().to_string()),
            &stripped[end + 1..],
        )
    } else {
        (None, s)
    };
    let after_to = after_start
        .trim_start()
        .strip_prefix("to")
        .unwrap_or(after_start);
    let (end_selector, body_input) = if let Some(stripped) = after_to.trim_start().strip_prefix('(')
    {
        let end = stripped.find(')')?;
        (
            Some(stripped[..end].trim().to_string()),
            &stripped[end + 1..],
        )
    } else {
        (None, after_to)
    };
    let body = take_body(body_input.trim_start())?;
    Some(ScopeRule {
        start_selector,
        end_selector,
        body_raw: body,
    })
}

/// Parse `@starting-style { ... }`. Returns None if no `{ ... }` block
/// follows the keyword.
pub fn parse_starting_style(input: &str) -> Option<StartingStyleRule> {
    let body = take_body(input.trim_start())?;
    Some(StartingStyleRule { body_raw: body })
}

/// Parse `@property --name { syntax: "<type>"; initial-value: ...; inherits: ...; }`.
pub fn parse_property(input: &str) -> Option<PropertyRule> {
    let s = input.trim_start();
    // Custom property name: must start with `--`.
    let mut end = 0;
    for (i, c) in s.char_indices() {
        if c.is_whitespace() || c == '{' {
            end = i;
            break;
        }
        end = i + c.len_utf8();
    }
    if end == 0 {
        return None;
    }
    let name = s[..end].trim().to_string();
    if !name.starts_with("--") {
        return None;
    }
    let body = take_body(s[end..].trim_start())?;
    let mut syntax = None;
    let mut inherits = false;
    let mut initial_value = None;
    for decl in body.split(';') {
        let decl = decl.trim();
        if let Some(idx) = decl.find(':') {
            let (k, v) = decl.split_at(idx);
            let v = v[1..].trim().trim_matches('"').to_string();
            match k.trim().to_ascii_lowercase().as_str() {
                "syntax" => syntax = Some(v),
                "inherits" => inherits = v.eq_ignore_ascii_case("true"),
                "initial-value" => initial_value = Some(v),
                _ => {}
            }
        }
    }
    Some(PropertyRule {
        name,
        syntax,
        inherits,
        initial_value,
    })
}

fn take_optional_name(s: &str) -> (Option<String>, &str) {
    if s.starts_with(['(', '{']) {
        return (None, s);
    }
    let mut end = 0;
    for (i, c) in s.char_indices() {
        if c.is_whitespace() || c == '(' || c == '{' {
            end = i;
            break;
        }
        end = i + c.len_utf8();
    }
    if end == 0 {
        (None, s)
    } else {
        (Some(s[..end].trim().to_string()), &s[end..])
    }
}

fn take_paren_and_body(s: &str) -> Option<(String, String)> {
    if !s.starts_with('(') {
        return None;
    }
    let close = match_balanced(s, '(', ')')?;
    let condition = s[1..close].to_string();
    let after = s[close + 1..].trim_start();
    let body = take_body(after)?;
    Some((condition, body))
}

fn take_body(s: &str) -> Option<String> {
    if !s.starts_with('{') {
        return None;
    }
    let close = match_balanced(s, '{', '}')?;
    Some(s[1..close].trim().to_string())
}

fn match_balanced(s: &str, open: char, close: char) -> Option<usize> {
    let bytes = s.as_bytes();
    let open_b = open as u8;
    let close_b = close as u8;
    let mut depth = 0i32;
    for (i, &b) in bytes.iter().enumerate() {
        if b == open_b {
            depth += 1;
        } else if b == close_b {
            depth -= 1;
            if depth == 0 {
                return Some(i);
            }
        }
    }
    None
}

/// Which axis sizes a query container exposes — drives whether a
/// block-axis (`height` / `block-size`) feature is queryable at all.
/// CSS Containment 3 §2.1: `container-type: inline-size` exposes ONLY the
/// inline axis; `size` exposes both. Querying an axis the container doesn't
/// establish never matches (the feature is "unknown" for that container).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ContainerAxes {
    /// Only inline-axis features (`width` / `inline-size`) are queryable.
    InlineOnly,
    /// Both inline- and block-axis features are queryable.
    Both,
}

/// Back-compat shim: evaluate an inline-axis `@container` condition against a
/// single width. Block-axis features are treated as not-queryable (matching a
/// `container-type: inline-size` container). Prefer
/// [`eval_container_condition_axes`] when both extents are known.
pub fn eval_container_condition(condition: &str, container_width_px: f32) -> bool {
    eval_container_condition_axes(
        condition,
        container_width_px,
        0.0,
        ContainerAxes::InlineOnly,
    )
}

/// Evaluate a `@container` condition string against a container's resolved
/// content-box inline (`inline_px`) and block (`block_px`) extents.
///
/// CSS Containment 3 §3 / Conditional 5 size-feature grammar. Supports:
///   * range/colon form — `min-width: 600px`, `max-inline-size: 30em`,
///     `width: 400px`, `min-block-size: 200px`, `aspect-ratio` (not yet);
///   * comparison form — `inline-size > 30em`, `width <= 1024px`,
///     `height >= 200px`;
///   * boolean combinators `and` / `or` / `not` (case-insensitive).
///
/// Block-axis features (`height`, `block-size`, `min/max-height`,
/// `min/max-block-size`) only ever match when `axes == Both`; on an
/// inline-size-only container they are NOT queryable and evaluate false
/// (Chrome/Blink `ContainerQueryEvaluator` treats them as `unknown`).
///
/// Returns `true` when the container matches and the block's contained
/// rules should apply.
pub fn eval_container_condition_axes(
    condition: &str,
    inline_px: f32,
    block_px: f32,
    axes: ContainerAxes,
) -> bool {
    fn parse_length(raw: &str) -> Option<f32> {
        let raw = raw.trim();
        if let Some(n) = raw.strip_suffix("px") {
            return n.trim().parse::<f32>().ok();
        }
        if let Some(n) = raw.strip_suffix("rem") {
            return n.trim().parse::<f32>().ok().map(|v| v * 16.0);
        }
        if let Some(n) = raw.strip_suffix("em") {
            return n.trim().parse::<f32>().ok().map(|v| v * 16.0);
        }
        // Bare number (rare; Container Queries L1 requires a unit, but
        // some examples in the wild leave it off).
        raw.parse::<f32>().ok()
    }

    // Resolve a feature name to (axis-size, queryable?). Inline-axis features
    // are always queryable; block-axis features only when `axes == Both`.
    fn feature_size(name: &str, inline_px: f32, block_px: f32, axes: ContainerAxes) -> Option<f32> {
        match name {
            "width" | "inline-size" => Some(inline_px),
            "height" | "block-size" => {
                if axes == ContainerAxes::Both {
                    Some(block_px)
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    fn eval_atom(atom: &str, inline_px: f32, block_px: f32, axes: ContainerAxes) -> bool {
        let s = atom
            .trim()
            .trim_start_matches('(')
            .trim_end_matches(')')
            .trim();
        // Comparison form: `inline-size > 30em` / `width >= 600px`.
        for op in [">=", "<=", ">", "<", "="] {
            if let Some(idx) = s.find(op) {
                let (lhs, rhs) = s.split_at(idx);
                let rhs = &rhs[op.len()..];
                let lhs = lhs.trim().to_ascii_lowercase();
                let Some(size) = feature_size(&lhs, inline_px, block_px, axes) else {
                    return false;
                };
                let Some(target) = parse_length(rhs) else {
                    return false;
                };
                return match op {
                    ">" => size > target,
                    "<" => size < target,
                    ">=" => size >= target,
                    "<=" => size <= target,
                    "=" => (size - target).abs() < 0.5,
                    _ => false,
                };
            }
        }
        // Range/colon form: `min-width: 600px` / `max-inline-size: 30em` /
        // `min-height: 200px` / `max-block-size: 30em`.
        let (key, value) = match s.split_once(':') {
            Some(p) => p,
            None => return false,
        };
        let key = key.trim().to_ascii_lowercase();
        let Some(target) = parse_length(value) else {
            return false;
        };
        let (axis_feature, is_min, is_max) = if let Some(f) = key.strip_prefix("min-") {
            (f, true, false)
        } else if let Some(f) = key.strip_prefix("max-") {
            (f, false, true)
        } else {
            (key.as_str(), false, false)
        };
        let Some(size) = feature_size(axis_feature, inline_px, block_px, axes) else {
            return false;
        };
        if is_min {
            size >= target
        } else if is_max {
            size <= target
        } else {
            (size - target).abs() < 0.5
        }
    }

    let trimmed = condition.trim();
    let lc = trimmed.to_ascii_lowercase();
    // Leading `not` negates the whole (sub)condition. CSS Conditional 5 §3.
    if let Some(rest) = lc.strip_prefix("not ") {
        let off = trimmed.len() - rest.len();
        return !eval_container_condition_axes(&trimmed[off..], inline_px, block_px, axes);
    }
    // Split top-level on `or` first (lower precedence), then `and`. Parens
    // round-trip through eval_atom so simple `(a) and (b)` works.
    if lc.contains(" or ") {
        return trimmed
            .split(" or ")
            .any(|chunk| eval_container_condition_axes(chunk, inline_px, block_px, axes));
    }
    if lc.contains(" and ") {
        return trimmed
            .split(" and ")
            .all(|chunk| eval_container_condition_axes(chunk, inline_px, block_px, axes));
    }
    eval_atom(trimmed, inline_px, block_px, axes)
}

// -------- Anchor positioning ---------------------------------------------

/// Resolved anchor reference — `anchor(--target top)` /
/// `anchor(--target right, 10px)` etc.
#[derive(Debug, Clone, PartialEq)]
pub struct AnchorRef {
    pub anchor_name: Option<String>,
    pub side: AnchorSide,
    pub fallback_px: Option<f32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnchorSide {
    Top,
    Right,
    Bottom,
    Left,
    Center,
}

/// Parse the contents of an `anchor(...)` function call.
pub fn parse_anchor(args: &str) -> Option<AnchorRef> {
    let parts: Vec<&str> = args.split(',').map(|p| p.trim()).collect();
    let first = parts.first()?;
    let mut tokens = first.split_whitespace();
    let mut anchor_name = None;
    let mut side_tok = None;
    for tok in &mut tokens {
        if tok.starts_with("--") {
            anchor_name = Some(tok.to_string());
        } else {
            side_tok = Some(tok);
            break;
        }
    }
    let side = match side_tok? {
        "top" => AnchorSide::Top,
        "right" => AnchorSide::Right,
        "bottom" => AnchorSide::Bottom,
        "left" => AnchorSide::Left,
        "center" => AnchorSide::Center,
        _ => return None,
    };
    let fallback_px = parts
        .get(1)
        .and_then(|p| p.strip_suffix("px"))
        .and_then(|n| n.trim().parse::<f32>().ok());
    Some(AnchorRef {
        anchor_name,
        side,
        fallback_px,
    })
}

// -------- Multicol --------------------------------------------------------

/// Resolved multicol parameters from `column-count` / `column-width`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Multicol {
    pub column_count: Option<u32>,
    pub column_width_px: Option<f32>,
    pub column_gap_px: f32,
}

impl Default for Multicol {
    fn default() -> Self {
        Self {
            column_count: None,
            column_width_px: None,
            column_gap_px: 16.0, // 1em default
        }
    }
}

/// Parse a CSS `<integer>` for `column-count` or `auto`.
pub fn parse_column_count(s: &str) -> Option<Option<u32>> {
    let s = s.trim();
    if s.eq_ignore_ascii_case("auto") {
        return Some(None);
    }
    s.parse::<u32>().ok().map(Some)
}

/// Parse `<length>` or `auto` for `column-width`.
pub fn parse_column_width(s: &str) -> Option<Option<f32>> {
    let s = s.trim();
    if s.eq_ignore_ascii_case("auto") {
        return Some(None);
    }
    s.strip_suffix("px")
        .and_then(|n| n.trim().parse::<f32>().ok())
        .map(Some)
}

// -------- Scroll-driven animations ----------------------------------------

/// Parse an `animation-timeline` value: `auto` | `none` |
/// `<scroll-timeline-name>` | `scroll(<axis> <scroller>)`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AnimationTimeline {
    Auto,
    None,
    Named(String),
    Scroll {
        axis: ScrollAxis,
        scroller: ScrollScroller,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScrollAxis {
    Block,
    Inline,
    X,
    Y,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScrollScroller {
    Nearest,
    Root,
    Self_,
}

pub fn parse_animation_timeline(s: &str) -> Option<AnimationTimeline> {
    let s = s.trim();
    if s.eq_ignore_ascii_case("auto") {
        return Some(AnimationTimeline::Auto);
    }
    if s.eq_ignore_ascii_case("none") {
        return Some(AnimationTimeline::None);
    }
    if let Some(inner) = s.strip_prefix("scroll(").and_then(|t| t.strip_suffix(')')) {
        let mut axis = ScrollAxis::Block;
        let mut scroller = ScrollScroller::Nearest;
        for tok in inner.split_whitespace() {
            match tok {
                "block" => axis = ScrollAxis::Block,
                "inline" => axis = ScrollAxis::Inline,
                "x" => axis = ScrollAxis::X,
                "y" => axis = ScrollAxis::Y,
                "nearest" => scroller = ScrollScroller::Nearest,
                "root" => scroller = ScrollScroller::Root,
                "self" => scroller = ScrollScroller::Self_,
                _ => {}
            }
        }
        return Some(AnimationTimeline::Scroll { axis, scroller });
    }
    Some(AnimationTimeline::Named(s.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn container_anonymous_with_min_width() {
        let r = parse_container("(min-width: 600px) { .card { padding: 1rem; } }").unwrap();
        assert!(r.name.is_none());
        assert_eq!(r.condition_raw, "min-width: 600px");
        assert!(r.body_raw.contains(".card"));
    }

    #[test]
    fn container_named_carries_name() {
        let r = parse_container("sidebar (min-width: 30em) { p { color: red; } }").unwrap();
        assert_eq!(r.name.as_deref(), Some("sidebar"));
    }

    #[test]
    fn layer_statement_form_lists_names() {
        let r = parse_layer("reset, base, components;").unwrap();
        assert_eq!(r.names, vec!["reset", "base", "components"]);
        assert!(r.body_raw.is_none());
    }

    #[test]
    fn layer_block_form_carries_body() {
        let r = parse_layer("theme { body { background: white; } }").unwrap();
        assert_eq!(r.names, vec!["theme"]);
        assert!(r.body_raw.unwrap().contains("body"));
    }

    #[test]
    fn scope_with_start_and_end() {
        let r = parse_scope("(.card) to (.footer) { a { color: blue; } }").unwrap();
        assert_eq!(r.start_selector.as_deref(), Some(".card"));
        assert_eq!(r.end_selector.as_deref(), Some(".footer"));
        assert!(r.body_raw.contains("color: blue"));
    }

    #[test]
    fn scope_implicit_root() {
        let r = parse_scope("{ p { font-size: 14px; } }").unwrap();
        assert!(r.start_selector.is_none());
        assert!(r.end_selector.is_none());
    }

    #[test]
    fn anchor_parses_side() {
        let a = parse_anchor("--popover bottom").unwrap();
        assert_eq!(a.anchor_name.as_deref(), Some("--popover"));
        assert_eq!(a.side, AnchorSide::Bottom);
        assert!(a.fallback_px.is_none());
    }

    #[test]
    fn anchor_parses_fallback() {
        let a = parse_anchor("--target top, 10px").unwrap();
        assert_eq!(a.fallback_px, Some(10.0));
    }

    #[test]
    fn column_count_auto() {
        assert_eq!(parse_column_count("auto"), Some(None));
        assert_eq!(parse_column_count("3"), Some(Some(3)));
    }

    #[test]
    fn column_width_px() {
        assert_eq!(parse_column_width("240px"), Some(Some(240.0)));
        assert_eq!(parse_column_width("auto"), Some(None));
    }

    #[test]
    fn animation_timeline_keywords() {
        assert_eq!(
            parse_animation_timeline("auto"),
            Some(AnimationTimeline::Auto)
        );
        assert_eq!(
            parse_animation_timeline("none"),
            Some(AnimationTimeline::None)
        );
    }

    #[test]
    fn animation_timeline_scroll_block_root() {
        let t = parse_animation_timeline("scroll(block root)").unwrap();
        match t {
            AnimationTimeline::Scroll { axis, scroller } => {
                assert_eq!(axis, ScrollAxis::Block);
                assert_eq!(scroller, ScrollScroller::Root);
            }
            _ => panic!("expected Scroll"),
        }
    }

    #[test]
    fn container_condition_min_width_matches_at_or_above() {
        assert!(eval_container_condition("(min-width: 600px)", 800.0));
        assert!(eval_container_condition("(min-width: 600px)", 600.0));
        assert!(!eval_container_condition("(min-width: 600px)", 480.0));
    }

    #[test]
    fn container_condition_max_inline_size_em() {
        // 30em at 16px/em = 480px.
        assert!(eval_container_condition("(max-inline-size: 30em)", 400.0));
        assert!(!eval_container_condition("(max-inline-size: 30em)", 600.0));
    }

    #[test]
    fn container_condition_comparator_form() {
        assert!(eval_container_condition("(inline-size > 30em)", 600.0));
        assert!(!eval_container_condition("(inline-size > 30em)", 400.0));
        assert!(eval_container_condition("(width <= 1024px)", 800.0));
    }

    #[test]
    fn container_condition_and_or() {
        // `and` requires every chunk to pass.
        assert!(eval_container_condition(
            "(min-width: 400px) and (max-width: 800px)",
            600.0
        ));
        assert!(!eval_container_condition(
            "(min-width: 400px) and (max-width: 800px)",
            1000.0
        ));
        // `or` requires at least one.
        assert!(eval_container_condition(
            "(max-width: 200px) or (min-width: 1000px)",
            1200.0
        ));
    }

    #[test]
    fn nested_braces_balance_correctly() {
        let r = parse_layer("page { .a { color: red; .b { color: blue; } } }").unwrap();
        assert_eq!(r.names, vec!["page"]);
        let body = r.body_raw.unwrap();
        assert!(body.contains(".a"));
        assert!(body.contains(".b"));
    }
}
