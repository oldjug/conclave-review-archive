//! `cv_unicode` — Unicode support for the browser.
//!
//! V1 ships:
//! - `bidi_class()` — UAX #9 Bidi_Class lookup over the most common
//!   scripts (Latin / Cyrillic / Greek, Arabic, Hebrew, Common, plus
//!   a handful of explicit-format characters). Full UCD coverage is a
//!   follow-up — what's here gets Arabic/Hebrew/English mixed-direction
//!   text rendering in the right order, which is the user-visible win.
//! - `resolve_paragraph()` — UAX #9 paragraph-level + W/N/I/L rules.
//!   Returns a per-codepoint `embedding_level` from which a renderer
//!   builds visually-ordered runs.
//! - `reorder_line()` — given a slice of `(text, level)` pairs in
//!   logical order, return them in visual order per UBA L2.
//!
//! Deliberately scoped: explicit isolates (LRI/RLI/PDI), bracket
//! pairs (BD16), and weak overrides beyond W1–W7 are simplified. Real
//! cv_unicode lands a UCD-table generator in `tools/ucd_gen` later.

#![allow(unused)]

pub mod bidi;
pub mod encoding;
mod encoding_tables;
pub mod linebreak;

pub use bidi::{
    BidiClass, ResolvedLevel, bidi_class, paragraph_level, reorder_line, resolve_paragraph,
};
pub use encoding::{
    Encoding, charset_from_content_type, decode, decode_text_default_utf8, decode_with_detection,
    detect_encoding, detect_encoding_default_utf8, encoding_for_label, prescan_meta, sniff_bom,
};
