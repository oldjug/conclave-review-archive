//! OpenType GPOS — glyph positioning.
//!
//! Implements the positioning adjustments applied after `GSUB`
//! substitution. V1 covers:
//!   * Lookup type 1 — single adjustment (uniform per-glyph offset)
//!   * Lookup type 2 — pair adjustment (classic kerning)
//!   * Lookup type 4 — mark-to-base attachment (essential for
//!     Arabic and Indic diacritics)
//!
//! Lookup types 3 (cursive), 5 (mark-to-ligature), 6 (mark-to-mark),
//! 7 (context), 8 (chaining context) follow once we wire real font
//! tables; the data model here is what they extend.

/// One glyph's positioning record: per-axis offsets in font units.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct GlyphPos {
    pub x_offset: i16,
    pub y_offset: i16,
    pub x_advance: i16,
    pub y_advance: i16,
}

/// Pair-adjustment kerning record. `class` indexing is supported by
/// the spec but for V1 we use direct glyph IDs — simpler and
/// matches what most kerning tables fall back to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KernPair {
    pub left: u16,
    pub right: u16,
    /// Advance adjustment applied to `left` (negative = move closer).
    pub x_adjust: i16,
}

/// Mark-to-base anchor pair. Identifies the (base_glyph, mark_glyph)
/// pair and where the mark anchors relative to the base's origin.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MarkAttachment {
    pub base_glyph: u16,
    pub mark_glyph: u16,
    /// Base anchor in font units (relative to base glyph origin).
    pub base_anchor_x: i16,
    pub base_anchor_y: i16,
    /// Mark anchor (the mark is offset so its anchor lands on the base's anchor).
    pub mark_anchor_x: i16,
    pub mark_anchor_y: i16,
}

/// GPOS engine state — owns the lookup tables and applies them to
/// glyph runs.
#[derive(Debug, Default)]
pub struct Gpos {
    /// Single-adjustment lookups indexed by glyph id.
    pub single_adj: std::collections::HashMap<u16, GlyphPos>,
    pub kern_pairs: Vec<KernPair>,
    pub mark_attachments: Vec<MarkAttachment>,
}

impl Gpos {
    pub fn new() -> Self {
        Self::default()
    }

    /// Apply GPOS to a glyph run. Returns the per-glyph positioning
    /// vector with kerning + single adjustments + mark attachment
    /// folded in. Input `glyphs` is the post-`GSUB` glyph id list.
    pub fn apply(&self, glyphs: &[u16]) -> Vec<GlyphPos> {
        let mut out: Vec<GlyphPos> = glyphs
            .iter()
            .map(|g| self.single_adj.get(g).copied().unwrap_or_default())
            .collect();
        // Pair kerning.
        for i in 0..glyphs.len().saturating_sub(1) {
            for k in &self.kern_pairs {
                if k.left == glyphs[i] && k.right == glyphs[i + 1] {
                    out[i].x_advance += k.x_adjust;
                }
            }
        }
        // Mark-to-base attachment.
        for i in 1..glyphs.len() {
            let base = glyphs[i - 1];
            let mark = glyphs[i];
            for m in &self.mark_attachments {
                if m.base_glyph == base && m.mark_glyph == mark {
                    out[i].x_offset += m.base_anchor_x - m.mark_anchor_x;
                    out[i].y_offset += m.base_anchor_y - m.mark_anchor_y;
                }
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_gpos_returns_default_positions() {
        let g = Gpos::new();
        let out = g.apply(&[10, 20, 30]);
        assert_eq!(out.len(), 3);
        for p in out {
            assert_eq!(p, GlyphPos::default());
        }
    }

    #[test]
    fn single_adjustment_applies_per_glyph() {
        let mut g = Gpos::new();
        g.single_adj.insert(
            42,
            GlyphPos {
                x_offset: 5,
                y_offset: -2,
                x_advance: 0,
                y_advance: 0,
            },
        );
        let out = g.apply(&[42]);
        assert_eq!(out[0].x_offset, 5);
        assert_eq!(out[0].y_offset, -2);
    }

    #[test]
    fn kerning_pulls_left_advance_only() {
        let mut g = Gpos::new();
        g.kern_pairs.push(KernPair {
            left: 1,
            right: 2,
            x_adjust: -40,
        });
        let out = g.apply(&[1, 2]);
        assert_eq!(out[0].x_advance, -40);
        assert_eq!(out[1].x_advance, 0);
    }

    #[test]
    fn mark_attachment_offsets_mark_to_base_anchor() {
        let mut g = Gpos::new();
        g.mark_attachments.push(MarkAttachment {
            base_glyph: 5,
            mark_glyph: 6,
            base_anchor_x: 100,
            base_anchor_y: 200,
            mark_anchor_x: 0,
            mark_anchor_y: 0,
        });
        let out = g.apply(&[5, 6]);
        assert_eq!(out[1].x_offset, 100);
        assert_eq!(out[1].y_offset, 200);
    }

    #[test]
    fn non_matching_kerning_pair_is_no_op() {
        let mut g = Gpos::new();
        g.kern_pairs.push(KernPair {
            left: 1,
            right: 2,
            x_adjust: -100,
        });
        let out = g.apply(&[3, 4]);
        assert_eq!(out[0].x_advance, 0);
    }

    #[test]
    fn kerning_and_mark_compose() {
        let mut g = Gpos::new();
        g.kern_pairs.push(KernPair {
            left: 10,
            right: 11,
            x_adjust: -20,
        });
        g.mark_attachments.push(MarkAttachment {
            base_glyph: 10,
            mark_glyph: 11,
            base_anchor_x: 50,
            base_anchor_y: 75,
            mark_anchor_x: 0,
            mark_anchor_y: 0,
        });
        let out = g.apply(&[10, 11]);
        assert_eq!(out[0].x_advance, -20);
        assert_eq!(out[1].x_offset, 50);
        assert_eq!(out[1].y_offset, 75);
    }
}
