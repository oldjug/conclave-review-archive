//! Typed multi-stage invalidation lattice (Milestone 2.1).
//!
//! Blink/Chrome model the rendering pipeline as a sequence of stages — style,
//! layout, paint — and tracks, per node, both whether *that node* is dirty for a
//! stage (a SELF bit) and whether some *descendant* is dirty (a DESCENDANT bit,
//! so a clean ancestor can walk straight down to the dirty subtree without
//! re-running the stage on itself).
//!
//! [`StageMask`] generalises `cv_dom`'s two ad-hoc style bools
//! (`style_dirty` / `child_styles_dirty`) into one byte holding all three
//! stages. Only the STYLE bits have a live consumer today (the cascade); the
//! LAYOUT and PAINT bits are new capability wired up by M2.2–M2.4 and are
//! currently unobserved, so they cannot change rendering.

/// A per-node bitset of pending render-pipeline work. One byte: three SELF bits
/// (this node is dirty for that stage) and three matching DESCENDANT bits (a
/// child is dirty for that stage). `Copy`, cheap, stored inline on a node.
///
/// Invariant maintained by [`super::Document`]'s `mark`: whenever a SELF bit is
/// set on a node, the matching DESCENDANT bit is set on every ancestor up to the
/// first ancestor that already has it — so a stage walk from the root always
/// reaches every dirty node.
#[derive(Clone, Copy, PartialEq, Eq, Default)]
pub struct StageMask(u8);

impl StageMask {
    // --- SELF bits: this node is dirty for the named stage. ---
    /// This node needs its style recomputed.
    pub const NEEDS_STYLE: u8 = 1 << 0;
    /// This node needs its layout (box geometry) recomputed.
    pub const NEEDS_LAYOUT: u8 = 1 << 1;
    /// This node needs to be repainted.
    pub const NEEDS_PAINT: u8 = 1 << 2;

    // --- DESCENDANT bits: some descendant of this node is dirty. ---
    /// A descendant needs its style recomputed.
    pub const CHILD_NEEDS_STYLE: u8 = 1 << 3;
    /// A descendant needs its layout recomputed.
    pub const CHILD_NEEDS_LAYOUT: u8 = 1 << 4;
    /// A descendant needs to be repainted.
    pub const CHILD_NEEDS_PAINT: u8 = 1 << 5;

    /// Mask of all three SELF bits.
    pub const SELF_ANY: u8 = Self::NEEDS_STYLE | Self::NEEDS_LAYOUT | Self::NEEDS_PAINT;
    /// Mask of all three DESCENDANT bits.
    pub const CHILD_ANY: u8 =
        Self::CHILD_NEEDS_STYLE | Self::CHILD_NEEDS_LAYOUT | Self::CHILD_NEEDS_PAINT;

    /// An empty mask (nothing pending). Same as `StageMask::default()`.
    pub const fn empty() -> Self {
        Self(0)
    }

    /// Build a mask from a raw bit pattern (combination of the `*` consts above).
    pub const fn from_bits(bits: u8) -> Self {
        Self(bits)
    }

    /// The raw byte. Useful for tests / debugging.
    pub const fn bits(self) -> u8 {
        self.0
    }

    /// True if `bit` (one of the `*` consts) is set.
    pub const fn is_set(self, bit: u8) -> bool {
        self.0 & bit != 0
    }

    /// Set `bit`.
    pub fn insert(&mut self, bit: u8) {
        self.0 |= bit;
    }

    /// Clear `bit`.
    pub fn remove(&mut self, bit: u8) {
        self.0 &= !bit;
    }

    /// True if any SELF bit is set (this node has pending work in some stage).
    pub const fn any_self_dirty(self) -> bool {
        self.0 & Self::SELF_ANY != 0
    }

    /// True if any DESCENDANT bit is set (some descendant has pending work).
    pub const fn any_descendant_dirty(self) -> bool {
        self.0 & Self::CHILD_ANY != 0
    }

    /// Given a SELF bit, return the matching DESCENDANT bit (used by `mark` to
    /// propagate up the ancestor chain). Panics on a non-SELF input — internal
    /// helper, only fed the three SELF consts.
    pub(crate) const fn child_bit_for(self_bit: u8) -> u8 {
        match self_bit {
            Self::NEEDS_STYLE => Self::CHILD_NEEDS_STYLE,
            Self::NEEDS_LAYOUT => Self::CHILD_NEEDS_LAYOUT,
            Self::NEEDS_PAINT => Self::CHILD_NEEDS_PAINT,
            _ => panic!("child_bit_for expects a SELF stage bit"),
        }
    }
}

impl std::fmt::Debug for StageMask {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut first = true;
        let mut emit = |f: &mut std::fmt::Formatter<'_>, name: &str| -> std::fmt::Result {
            if !first {
                f.write_str(" | ")?;
            }
            first = false;
            f.write_str(name)
        };
        f.write_str("StageMask(")?;
        if self.0 == 0 {
            f.write_str("empty")?;
        } else {
            if self.is_set(Self::NEEDS_STYLE) {
                emit(f, "NEEDS_STYLE")?;
            }
            if self.is_set(Self::NEEDS_LAYOUT) {
                emit(f, "NEEDS_LAYOUT")?;
            }
            if self.is_set(Self::NEEDS_PAINT) {
                emit(f, "NEEDS_PAINT")?;
            }
            if self.is_set(Self::CHILD_NEEDS_STYLE) {
                emit(f, "CHILD_NEEDS_STYLE")?;
            }
            if self.is_set(Self::CHILD_NEEDS_LAYOUT) {
                emit(f, "CHILD_NEEDS_LAYOUT")?;
            }
            if self.is_set(Self::CHILD_NEEDS_PAINT) {
                emit(f, "CHILD_NEEDS_PAINT")?;
            }
        }
        f.write_str(")")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn self_and_descendant_queries() {
        let mut m = StageMask::empty();
        assert!(!m.any_self_dirty() && !m.any_descendant_dirty());
        m.insert(StageMask::NEEDS_LAYOUT);
        assert!(m.is_set(StageMask::NEEDS_LAYOUT));
        assert!(m.any_self_dirty());
        assert!(!m.any_descendant_dirty());
        m.insert(StageMask::CHILD_NEEDS_PAINT);
        assert!(m.any_descendant_dirty());
        m.remove(StageMask::NEEDS_LAYOUT);
        assert!(!m.is_set(StageMask::NEEDS_LAYOUT));
        assert!(!m.any_self_dirty());
        // Descendant bit untouched by clearing a self bit.
        assert!(m.any_descendant_dirty());
    }

    #[test]
    fn child_bit_mapping() {
        assert_eq!(
            StageMask::child_bit_for(StageMask::NEEDS_STYLE),
            StageMask::CHILD_NEEDS_STYLE
        );
        assert_eq!(
            StageMask::child_bit_for(StageMask::NEEDS_LAYOUT),
            StageMask::CHILD_NEEDS_LAYOUT
        );
        assert_eq!(
            StageMask::child_bit_for(StageMask::NEEDS_PAINT),
            StageMask::CHILD_NEEDS_PAINT
        );
    }

    #[test]
    fn debug_is_readable() {
        let mut m = StageMask::empty();
        assert_eq!(format!("{m:?}"), "StageMask(empty)");
        m.insert(StageMask::NEEDS_STYLE);
        m.insert(StageMask::CHILD_NEEDS_STYLE);
        assert_eq!(format!("{m:?}"), "StageMask(NEEDS_STYLE | CHILD_NEEDS_STYLE)");
    }
}
