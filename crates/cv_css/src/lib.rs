//! `cv_css` — CSS Syntax 3 tokenizer + parser, Selectors 4 (subset),
//! plus minimal style computation against a `cv_html` DOM.
//!
//! Scope today:
//!   - Tokenizer: idents, hashes, numbers/dimensions/percent, strings,
//!     functions, at-keywords, delimiters, whitespace, comments.
//!   - Parser: stylesheet → rules (qualified + @media/@import/@font-face).
//!     Declarations are key/value pairs; value is kept as a raw token list.
//!   - Selectors: type, class (`.x`), id (`#x`), universal (`*`), with
//!     descendant and child combinators. Compound selectors. Selector
//!     lists (`a, b, c`). Pseudo-class `:hover`/`:focus` parsed but not
//!     matched.
//!   - Cascade: per-element compute styles using selector specificity and
//!     declaration order. Inheritance for a small set of properties.
//!
//! Not yet: `@layer`, `@scope`, container queries, calc(), env(),
//! attribute selectors, full pseudo support, complex specificity edge
//! cases. Tracked as M2 follow-ups.

#![allow(missing_debug_implementations, unreachable_patterns)]

pub mod cascade;
pub mod modern;
pub mod parser;
pub mod properties;
pub mod selectors;
pub mod tokenizer;

pub use cascade::{
    AncestorFilter, ContainerType, InvalidationSet, KeyframeRule, PseudoState,
    PseudoStateInvalidation, QueryContainer,
    QueryContainerStack, RuleFeatureSet, SelectorIndex, bloom_reset, bloom_stats,
    build_rule_feature_set, collect_keyframes, compute_pseudo, compute_pseudo_with_index,
    compute_with_index, compute_with_index_cq, compute_with_index_inheriting,
    compute_with_index_inheriting_filtered, compute_with_index_inheriting_filtered_cq,
    current_device_pixel_ratio, media_query_matches_str, media_query_matches_str_dpr,
    sample_animation, set_device_pixel_ratio, take_unknown_property_counts,
};
pub use modern::{ContainerAxes, eval_container_condition, eval_container_condition_axes};
pub use parser::{Declaration, Rule, Stylesheet, parse_inline_style, parse_stylesheet};
pub use selectors::{Selector, SimpleSelector};
pub use tokenizer::{CssToken, tokenize};
