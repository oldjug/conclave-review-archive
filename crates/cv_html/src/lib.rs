//! `cv_html` — HTML parser per the WHATWG HTML Standard.
//!
//! Today: tokenizer covering the most common states (data, tag open,
//! tag name, attributes, comments, doctype, RAWTEXT for `<script>` /
//! `<style>`), plus a basic tree constructor that produces a simple
//! DOM-like tree. Pass over WHATWG's full insertion-mode table comes
//! incrementally as we test against more pages.

pub mod entities;
pub mod fragment;
pub mod token;
pub mod tokenizer;
pub mod tree;

pub use token::{Attribute, Token};
pub use tokenizer::Tokenizer;
pub use tree::{Document, Node, NodeKind, parse};
