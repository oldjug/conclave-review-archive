//! `cv_text` — web-font registration with Windows GDI.
//!
//! Browser-side flow:
//! 1. cv_css parses `@font-face { font-family: "X"; src: url("foo.ttf") }`.
//! 2. conclave walks the parsed stylesheet, pulls each `@font-face`'s
//!    family + first src URL, fetches the URL, and hands the bytes here.
//! 3. We detect TTF/OTF magic and register the bytes with GDI via
//!    `AddFontMemResourceEx` — that makes the face globally visible to
//!    all subsequent `CreateFontW` calls in the process (including the
//!    ones cv_ui issues during text paint), keyed by the family name.
//!
//! WOFF2 decoding is implemented in `woff2.rs`: header parse + Brotli
//! decompression + glyf/loca untransformation + SFNT repack, per the
//! W3C WOFF2 spec. WOFF (uncompressed-ish container) is not yet
//! supported — rare in modern web use.

#![allow(non_camel_case_types, non_snake_case, clippy::upper_case_acronyms)]
#![allow(unreachable_pub, missing_debug_implementations)]

pub mod cjk;
pub mod color_font;
pub mod gpos;
pub mod indic;
pub mod shape_text;
pub mod shaping;
pub mod variable_font;
pub mod woff2;

pub use shaping::{
    ChainRule, CmapRange, Feature, JoiningForm, JoiningType, KernPair, Ligature, ShapeError,
    SingleSubst, TableEntry, apply_chain_rules, apply_kerning, apply_ligatures,
    apply_substitutions, arabic_lam_alef_ligatures, arabic_positional_forms, cmap_lookup,
    find_table, joining_type, parse_cmap, parse_feature_list, parse_table_dir, presentation_form,
    shape_arabic_to_presentation_forms,
};

pub use indic::{IndicCategory, indic_category, is_devanagari, reorder_devanagari};
pub use shape_text::{
    ShapedGlyph, is_combining_mark, shape_paragraph, shape_to_visual_string,
};

use core::ffi::c_void;
use std::sync::Mutex;

type HANDLE = *mut c_void;
type DWORD = u32;

#[link(name = "gdi32")]
unsafe extern "system" {
    /// Register a memory-resident font with the process. The handle
    /// must be kept alive (we leak it intentionally — the font is
    /// valid for the lifetime of the program) and the font is
    /// available to all `CreateFontW` calls until process exit.
    fn AddFontMemResourceEx(
        pFileView: *const c_void,
        cjSize: DWORD,
        pvResrved: *mut c_void,
        pNumFonts: *mut DWORD,
    ) -> HANDLE;
}

/// Detected web-font file format.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum FontFormat {
    /// TrueType — `0x00010000` magic at the front.
    TrueType,
    /// OpenType / CFF — `OTTO` magic.
    OpenType,
    /// True font collection (multiple faces in one file). GDI accepts
    /// these via `AddFontMemResourceEx` and exposes all faces.
    TrueTypeCollection,
    /// WOFF or WOFF2 — needs decoding before GDI can use it. We
    /// recognise the magic so the caller can log the format, but
    /// decoding is out of scope for this slice.
    Woff,
    Woff2,
    /// Anything else — caller skips and falls back to system fonts.
    Unknown,
}

/// Sniff the first few bytes of a font file and classify the format.
pub fn detect_format(bytes: &[u8]) -> FontFormat {
    if bytes.len() < 4 {
        return FontFormat::Unknown;
    }
    match &bytes[..4] {
        // TrueType: 0x00010000 (version 1.0 fixed-point) is the
        // canonical TTF magic. Some tools emit "true" instead.
        [0x00, 0x01, 0x00, 0x00] => FontFormat::TrueType,
        b"true" => FontFormat::TrueType,
        b"OTTO" => FontFormat::OpenType,
        b"ttcf" => FontFormat::TrueTypeCollection,
        b"wOFF" => FontFormat::Woff,
        b"wOF2" => FontFormat::Woff2,
        _ => FontFormat::Unknown,
    }
}

/// Registered-fonts table. We keep the byte buffers alive forever so
/// the GDI handle stays valid; the `Mutex<Vec<...>>` is just a leak
/// container. The `HashSet` deduplicates registrations by the
/// `(family, url-hash)` pair so reloading the same page doesn't
/// re-register the same face over and over.
static FONT_BUFFERS: Mutex<Vec<Vec<u8>>> = Mutex::new(Vec::new());
static REGISTERED_KEYS: Mutex<Vec<String>> = Mutex::new(Vec::new());

/// Register a font byte buffer with the process-wide GDI font set.
/// Returns `true` if the font was successfully added (or had been
/// previously added under the same dedupe key). Returns `false` if
/// the format is unsupported (e.g. WOFF/WOFF2 today) or if GDI
/// rejected the bytes.
///
/// `dedupe_key` is a caller-supplied string (typically family +
/// absolute URL) used to avoid double-registration. `family_hint`
/// is for logging only — the actual face name GDI exposes comes
/// from the font's `name` table, which we trust the file to set.
pub fn register_font_bytes(bytes: &[u8], dedupe_key: &str, _family_hint: &str) -> bool {
    if bytes.is_empty() {
        return false;
    }
    // WOFF2 fonts get decoded to SFNT here, then fall through to
    // the GDI registration below. WOFF (the original) is rare on
    // the modern web and still unsupported.
    let decoded_owner: Vec<u8>;
    let usable_bytes: &[u8] = match detect_format(bytes) {
        FontFormat::TrueType | FontFormat::OpenType | FontFormat::TrueTypeCollection => bytes,
        FontFormat::Woff2 => match woff2::decode_woff2(bytes) {
            Ok(sfnt) => {
                decoded_owner = sfnt;
                &decoded_owner
            }
            Err(_) => return false,
        },
        FontFormat::Woff | FontFormat::Unknown => return false,
    };
    // Dedupe.
    {
        let registered = REGISTERED_KEYS.lock().unwrap();
        if registered.iter().any(|k| k == dedupe_key) {
            return true;
        }
    }
    // Copy into a long-lived buffer (GDI may scan the bytes lazily
    // and we don't trust the caller's borrow to outlive the
    // process).
    let owned = usable_bytes.to_vec();
    let mut num_fonts: DWORD = 0;
    let handle = unsafe {
        AddFontMemResourceEx(
            owned.as_ptr() as *const c_void,
            owned.len() as DWORD,
            core::ptr::null_mut(),
            &raw mut num_fonts,
        )
    };
    if handle.is_null() || num_fonts == 0 {
        return false;
    }
    // GDI now references the bytes by pointer — keep them alive
    // forever. Also record the dedupe key so a second call with
    // the same key short-circuits.
    FONT_BUFFERS.lock().unwrap().push(owned);
    REGISTERED_KEYS.lock().unwrap().push(dedupe_key.to_string());
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_truetype_magic() {
        let bytes = [0x00, 0x01, 0x00, 0x00, 0, 0, 0, 0];
        assert_eq!(detect_format(&bytes), FontFormat::TrueType);
    }

    #[test]
    fn detects_opentype_magic() {
        let bytes = b"OTTO\0\0\0\0";
        assert_eq!(detect_format(bytes), FontFormat::OpenType);
    }

    #[test]
    fn detects_woff2_magic() {
        let bytes = b"wOF2\0\0\0\0";
        assert_eq!(detect_format(bytes), FontFormat::Woff2);
    }

    #[test]
    fn detects_unknown() {
        let bytes = b"\xff\xff\xff\xff";
        assert_eq!(detect_format(bytes), FontFormat::Unknown);
    }

    #[test]
    fn empty_bytes_rejected() {
        assert!(!register_font_bytes(&[], "k", "fam"));
    }

    #[test]
    fn woff_bytes_rejected_until_decoder_lands() {
        let bytes = b"wOFFstuffstuff";
        assert!(!register_font_bytes(bytes, "k", "fam"));
    }

    #[test]
    fn truncated_woff2_rejected() {
        // Not enough bytes for a real header — the WOFF2 decoder
        // must reject this cleanly without panicking.
        let bytes = b"wOF2truncated";
        assert!(!register_font_bytes(bytes, "k", "fam"));
    }
}
