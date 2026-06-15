//! `retained_dl` — M5.2 retained, node_id-keyed display list.
//!
//! A PARALLEL, flag-gated (`CV_RETAINED_DL=1`), oracle-checked path that sits
//! BESIDE the live immediate-mode painter (`paint_box_offset_t` in `main.rs`).
//! When the flag is OFF (default) this module is never reached and production
//! frames are byte-for-byte the live path.
//!
//! Structure:
//!   * [`RetainedDisplayList`] — `node_id` → [`PaintChunk`] index, plus the
//!     ordered chunk arena.
//!   * [`PaintChunk`] — one box's own draw ops (split before/after the child
//!     walk), z-order metadata captured verbatim from the live painter, a
//!     position/transform-EXCLUDING [`PaintChunk::content_hash`] and a bottom-up
//!     [`PaintChunk::subtree_hash`] rollup.
//!   * [`PaintOp`] — a FAITHFUL record of the exact `cv_gfx` call the live
//!     painter makes, with FINAL screen-space args, so replaying it re-issues
//!     the identical primitive and the rasterised bitmap is BYTE-IDENTICAL.
//!
//! [`generate`] is a sibling recursion of `paint_box_offset_t`: it threads the
//! same accumulating offset and clip, runs the same branches, but RECORDS each
//! primitive (via [`Recorder`], which also draws into a scratch bitmap so clip
//! masks / affine layer rasters behave identically) instead of only issuing it.
//! [`replay`] is the mirror recursion: it re-issues the recorded ops + appends
//! the SAME `TextItem`s, then the caller runs the SAME `bake_content_text_into_bitmap`
//! so glyph pixels match too. The oracle test ([`tests::assert_oracle`]) asserts
//! `bmp_live.pixels == bmp_replay.pixels` (0 diff) across a battery of trees.
//!
//! 5.2 BUILDS + DIFFS + ORACLE-CHECKS only. Nothing here drives a production
//! frame; the live full-bake still produces every frame. Relocate-without-raster,
//! tile invalidation, and cached per-chunk pixels are deferred to 5.4.

use std::collections::HashMap;
use std::sync::Arc;

use cv_gfx::{Bitmap, Color, FilterOp};
use cv_layout::LayoutBox;
use cv_ui::TextItem;

// ── Hashing ────────────────────────────────────────────────────────────────

/// Hand-rolled FNV-1a 64-bit, the exact pattern proven at
/// `cv_css/src/cascade.rs::ident_hash` (seed `0xcbf29ce484222325`, prime
/// `0x100000001b3`, final mix `h ^ (h >> 29)`). No third-party crate.
#[derive(Debug, Clone)]
pub struct Fnv(u64);

impl Fnv {
    pub fn new() -> Self {
        Fnv(0xcbf2_9ce4_8422_2325)
    }
    #[inline]
    pub fn byte(&mut self, b: u8) {
        self.0 ^= b as u64;
        self.0 = self.0.wrapping_mul(0x0000_0100_0000_01b3);
    }
    #[inline]
    pub fn bytes(&mut self, s: &[u8]) {
        for &b in s {
            self.byte(b);
        }
    }
    #[inline]
    pub fn u8v(&mut self, v: u8) {
        self.byte(v);
    }
    #[inline]
    pub fn u32(&mut self, v: u32) {
        self.bytes(&v.to_le_bytes());
    }
    #[inline]
    pub fn u64(&mut self, v: u64) {
        self.bytes(&v.to_le_bytes());
    }
    /// Hash an `f32` by its canonicalised bits: NaN → one pattern, `-0.0` → `0.0`,
    /// so re-bakes / moved nodes never produce phantom content changes.
    #[inline]
    pub fn f32(&mut self, v: f32) {
        let c = if v.is_nan() {
            f32::from_bits(0x7fc0_0000) // canonical quiet NaN
        } else if v == 0.0 {
            0.0 // maps -0.0 → +0.0
        } else {
            v
        };
        self.u32(c.to_bits());
    }
    #[inline]
    pub fn opt_tag(&mut self, present: bool) {
        self.byte(if present { 1 } else { 0 });
    }
    #[inline]
    pub fn color(&mut self, c: Color) {
        self.byte(c.r);
        self.byte(c.g);
        self.byte(c.b);
        self.byte(c.a);
    }
    #[inline]
    pub fn lcolor(&mut self, c: cv_layout::Color) {
        self.byte(c.r);
        self.byte(c.g);
        self.byte(c.b);
        self.byte(c.a);
    }
    #[inline]
    pub fn str(&mut self, s: &str) {
        self.u32(s.len() as u32);
        self.bytes(s.as_bytes());
    }
    #[inline]
    pub fn finish(self) -> u64 {
        self.0 ^ (self.0 >> 29)
    }
}

impl Default for Fnv {
    fn default() -> Self {
        Self::new()
    }
}

// ── PaintOp — faithful record of the live cv_gfx call ───────────────────────

/// A `cv_gfx::FilterOp` mirror that is `Clone`-able and lives in the retained
/// list (the live `FilterOp` is `Copy`, so this is a 1:1 record).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum FilterOpRec {
    Blur(f32),
    Brightness(f32),
    Contrast(f32),
    Grayscale(f32),
    Invert(f32),
    Sepia(f32),
    Saturate(f32),
    HueRotate(f32),
    Opacity(f32),
}

impl FilterOpRec {
    fn from_gfx(op: FilterOp) -> Self {
        match op {
            FilterOp::Blur(v) => FilterOpRec::Blur(v),
            FilterOp::Brightness(v) => FilterOpRec::Brightness(v),
            FilterOp::Contrast(v) => FilterOpRec::Contrast(v),
            FilterOp::Grayscale(v) => FilterOpRec::Grayscale(v),
            FilterOp::Invert(v) => FilterOpRec::Invert(v),
            FilterOp::Sepia(v) => FilterOpRec::Sepia(v),
            FilterOp::Saturate(v) => FilterOpRec::Saturate(v),
            FilterOp::HueRotate(v) => FilterOpRec::HueRotate(v),
            FilterOp::Opacity(v) => FilterOpRec::Opacity(v),
        }
    }
    fn to_gfx(self) -> FilterOp {
        match self {
            FilterOpRec::Blur(v) => FilterOp::Blur(v),
            FilterOpRec::Brightness(v) => FilterOp::Brightness(v),
            FilterOpRec::Contrast(v) => FilterOp::Contrast(v),
            FilterOpRec::Grayscale(v) => FilterOp::Grayscale(v),
            FilterOpRec::Invert(v) => FilterOp::Invert(v),
            FilterOpRec::Sepia(v) => FilterOp::Sepia(v),
            FilterOpRec::Saturate(v) => FilterOp::Saturate(v),
            FilterOpRec::HueRotate(v) => FilterOp::HueRotate(v),
            FilterOpRec::Opacity(v) => FilterOp::Opacity(v),
        }
    }
}

/// One recorded draw op. FINAL screen-space args (offset already folded in,
/// opacity already folded into `Color.a`), captured in the exact order /
/// granularity the live painter issues them, so replay is byte-identical.
#[derive(Clone, Debug)]
pub enum PaintOp {
    FillRect { x: i32, y: i32, w: i32, h: i32, c: Color },
    FillRectRounded { x: i32, y: i32, w: i32, h: i32, radius: i32, c: Color },
    FillRectRoundedRing { x: i32, y: i32, w: i32, h: i32, outer_r: i32, ring_w: i32, c: Color },
    FillEllipseRingTop { x: i32, y: i32, w: i32, h: i32, ring_w: i32, c: Color },
    FillRectGradient { x: i32, y: i32, w: i32, h: i32, angle_deg: f32, from: Color, to: Color },
    FillRectRadialGradient { x: i32, y: i32, w: i32, h: i32, radius: i32, inner: Color, outer: Color },
    BlitBgra { x: i32, y: i32, iw: u32, ih: u32, pixels: Arc<Vec<u32>> },
    BlitBgraScaled { dx: i32, dy: i32, dw: u32, dh: u32, iw: u32, ih: u32, pixels: Arc<Vec<u32>> },
    BlitBgraSprite {
        box_x: i32,
        box_y: i32,
        box_w: i32,
        box_h: i32,
        off_x: i32,
        off_y: i32,
        iw: u32,
        ih: u32,
        pixels: Arc<Vec<u32>>,
    },
    BlitMaskTintedScaled {
        dx: i32,
        dy: i32,
        dw: u32,
        dh: u32,
        mw: u32,
        mh: u32,
        mask: Arc<Vec<u32>>,
        c: Color,
    },
    ClipRoundedRect { x: i32, y: i32, w: i32, h: i32, radius: i32 },
    ClipInset { x: i32, y: i32, w: i32, h: i32, t: i32, r: i32, b: i32, l: i32 },
    ClipCircle { x: i32, y: i32, w: i32, h: i32, cx: f32, cy: f32, radius: f32 },
    ClipPolygon { x: i32, y: i32, w: i32, h: i32, pts: Vec<(f32, f32)> },
    ApplyFilterRect { x: i32, y: i32, w: i32, h: i32, op: FilterOpRec },
    /// Affine layer blit: replay re-rasters the captured layer ops/texts into a
    /// fresh layer bitmap then `blit_affine`s it. The layer pixels are NOT stored
    /// (5.2 regenerates them at replay to keep the structure cheap + Clone-able).
    BlitAffine { matrix: [f32; 6], layer: Box<AffineLayer> },
}

impl PaintOp {
    /// Re-issue this op into `bmp`, byte-identically to the live primitive.
    fn replay_into(&self, bmp: &mut Bitmap) {
        match self {
            PaintOp::FillRect { x, y, w, h, c } => bmp.fill_rect(*x, *y, *w, *h, *c),
            PaintOp::FillRectRounded { x, y, w, h, radius, c } => {
                bmp.fill_rect_rounded(*x, *y, *w, *h, *radius, *c)
            }
            PaintOp::FillRectRoundedRing { x, y, w, h, outer_r, ring_w, c } => {
                bmp.fill_rect_rounded_ring(*x, *y, *w, *h, *outer_r, *ring_w, *c)
            }
            PaintOp::FillEllipseRingTop { x, y, w, h, ring_w, c } => {
                bmp.fill_ellipse_ring_top(*x, *y, *w, *h, *ring_w, *c)
            }
            PaintOp::FillRectGradient { x, y, w, h, angle_deg, from, to } => {
                bmp.fill_rect_gradient(*x, *y, *w, *h, *angle_deg, *from, *to)
            }
            PaintOp::FillRectRadialGradient { x, y, w, h, radius, inner, outer } => {
                bmp.fill_rect_radial_gradient(*x, *y, *w, *h, *radius, *inner, *outer)
            }
            PaintOp::BlitBgra { x, y, iw, ih, pixels } => {
                bmp.blit_bgra(*x, *y, *iw, *ih, pixels)
            }
            PaintOp::BlitBgraScaled { dx, dy, dw, dh, iw, ih, pixels } => {
                bmp.blit_bgra_scaled(*dx, *dy, *dw, *dh, *iw, *ih, pixels)
            }
            PaintOp::BlitBgraSprite {
                box_x,
                box_y,
                box_w,
                box_h,
                off_x,
                off_y,
                iw,
                ih,
                pixels,
            } => bmp.blit_bgra_sprite(*box_x, *box_y, *box_w, *box_h, *off_x, *off_y, *iw, *ih, pixels),
            PaintOp::BlitMaskTintedScaled { dx, dy, dw, dh, mw, mh, mask, c } => {
                bmp.blit_mask_tinted_scaled(*dx, *dy, *dw, *dh, *mw, *mh, mask, *c)
            }
            PaintOp::ClipRoundedRect { x, y, w, h, radius } => {
                bmp.clip_rounded_rect(*x, *y, *w, *h, *radius)
            }
            PaintOp::ClipInset { x, y, w, h, t, r, b, l } => {
                bmp.clip_inset(*x, *y, *w, *h, *t, *r, *b, *l)
            }
            PaintOp::ClipCircle { x, y, w, h, cx, cy, radius } => {
                bmp.clip_circle(*x, *y, *w, *h, *cx, *cy, *radius)
            }
            PaintOp::ClipPolygon { x, y, w, h, pts } => bmp.clip_polygon(*x, *y, *w, *h, pts),
            PaintOp::ApplyFilterRect { x, y, w, h, op } => {
                bmp.apply_filter_rect(*x, *y, *w, *h, op.to_gfx())
            }
            PaintOp::BlitAffine { matrix, layer } => {
                let mut lbmp = Bitmap::new(layer.w, layer.h);
                lbmp.clear(Color::TRANSPARENT);
                let mut ltexts: Vec<TextItem> = Vec::new();
                for op in &layer.ops {
                    op.replay_into(&mut lbmp);
                }
                ltexts.extend(layer.texts.iter().cloned());
                cv_ui::bake_content_text_into_bitmap(&mut lbmp, &mut ltexts);
                bmp.blit_affine(&lbmp, *matrix, 1.0);
            }
        }
    }
}

impl PaintOp {
    /// The FINAL screen-space bounding box of every pixel this op can touch, as a
    /// `(x0, y0, x1, y1)` half-open rect (x1/y1 exclusive). Used to build the M5.4
    /// damage region and per-chunk `paint_extent`. ALL variants carry finite
    /// rects, so this never returns an unbounded extent. For `BlitAffine` the
    /// extent is the bbox of the 4 transformed layer corners (rotation makes it
    /// larger than the border rect). Clip ops touch only pixels inside their own
    /// rect, so their rect is their extent.
    fn extent_screen(&self) -> Option<(f32, f32, f32, f32)> {
        let r = |x: i32, y: i32, w: i32, h: i32| -> Option<(f32, f32, f32, f32)> {
            if w == 0 || h == 0 {
                // A zero-size op paints nothing; contribute no extent.
                return None;
            }
            let (x0, x1) = if w >= 0 { (x, x + w) } else { (x + w, x) };
            let (y0, y1) = if h >= 0 { (y, y + h) } else { (y + h, y) };
            Some((x0 as f32, y0 as f32, x1 as f32, y1 as f32))
        };
        match self {
            PaintOp::FillRect { x, y, w, h, .. } => r(*x, *y, *w, *h),
            PaintOp::FillRectRounded { x, y, w, h, .. } => r(*x, *y, *w, *h),
            PaintOp::FillRectRoundedRing { x, y, w, h, .. } => r(*x, *y, *w, *h),
            PaintOp::FillEllipseRingTop { x, y, w, h, .. } => r(*x, *y, *w, *h),
            PaintOp::FillRectGradient { x, y, w, h, .. } => r(*x, *y, *w, *h),
            PaintOp::FillRectRadialGradient { x, y, w, h, .. } => r(*x, *y, *w, *h),
            PaintOp::BlitBgra { x, y, iw, ih, .. } => r(*x, *y, *iw as i32, *ih as i32),
            PaintOp::BlitBgraScaled { dx, dy, dw, dh, .. } => r(*dx, *dy, *dw as i32, *dh as i32),
            PaintOp::BlitBgraSprite { box_x, box_y, box_w, box_h, .. } => {
                r(*box_x, *box_y, *box_w, *box_h)
            }
            PaintOp::BlitMaskTintedScaled { dx, dy, dw, dh, .. } => {
                r(*dx, *dy, *dw as i32, *dh as i32)
            }
            PaintOp::ClipRoundedRect { x, y, w, h, .. } => r(*x, *y, *w, *h),
            PaintOp::ClipInset { x, y, w, h, .. } => r(*x, *y, *w, *h),
            PaintOp::ClipCircle { x, y, w, h, .. } => r(*x, *y, *w, *h),
            PaintOp::ClipPolygon { x, y, w, h, .. } => r(*x, *y, *w, *h),
            PaintOp::ApplyFilterRect { x, y, w, h, .. } => r(*x, *y, *w, *h),
            PaintOp::BlitAffine { matrix, layer } => {
                // bbox of the 4 transformed layer corners. matrix maps layer-local
                // (lx, ly) -> screen via [m0 m2 e; m1 m3 f].
                let (m0, m1, m2, m3, e, f) =
                    (matrix[0], matrix[1], matrix[2], matrix[3], matrix[4], matrix[5]);
                let lw = layer.w as f32;
                let lh = layer.h as f32;
                let corners = [(0.0, 0.0), (lw, 0.0), (0.0, lh), (lw, lh)];
                let mut x0 = f32::INFINITY;
                let mut y0 = f32::INFINITY;
                let mut x1 = f32::NEG_INFINITY;
                let mut y1 = f32::NEG_INFINITY;
                for (lx, ly) in corners {
                    let sx = m0 * lx + m2 * ly + e;
                    let sy = m1 * lx + m3 * ly + f;
                    x0 = x0.min(sx);
                    y0 = y0.min(sy);
                    x1 = x1.max(sx);
                    y1 = y1.max(sy);
                }
                if x1 > x0 && y1 > y0 {
                    Some((x0, y0, x1, y1))
                } else {
                    None
                }
            }
        }
    }
}

/// Union of two optional `(x0,y0,x1,y1)` half-open extents.
fn union_extent(
    a: Option<(f32, f32, f32, f32)>,
    b: Option<(f32, f32, f32, f32)>,
) -> Option<(f32, f32, f32, f32)> {
    match (a, b) {
        (None, b) => b,
        (a, None) => a,
        (Some((ax0, ay0, ax1, ay1)), Some((bx0, by0, bx1, by1))) => {
            Some((ax0.min(bx0), ay0.min(by0), ax1.max(bx1), ay1.max(by1)))
        }
    }
}

/// Bounding box of a content `TextItem`'s glyph run in FINAL screen coords. The
/// run is rasterised into a DIB exactly `t.w` wide (see `render_text_run_alpha`),
/// so width is bounded by `t.w`; height can grow when text wraps, so we take a
/// generous vertical bound (the content height OR several line-heights) to keep
/// the extent a SUPERSET — over-expansion is always byte-safe, under-expansion
/// corrupts. The incremental text bake is ALSO R-clipped per glyph as a hard
/// guarantee, so even a height under-estimate here cannot write outside R.
fn text_item_extent(t: &TextItem) -> Option<(f32, f32, f32, f32)> {
    if t.w <= 0 {
        return None;
    }
    let x0 = t.x as f32;
    let x1 = (t.x + t.w.max(1)) as f32;
    let y0 = t.y as f32;
    // Generous vertical bound: at least the content height, and at least a few
    // line-heights so wrapped text is covered. font_size_px*4 comfortably covers
    // the common 1-3 line case; the R-clipped bake makes this a non-correctness
    // bound (pure over-paint budget).
    let h = (t.h as f32).max(t.font_size_px as f32 * 4.0).max(1.0);
    let y1 = y0 + h;
    Some((x0, y0, x1, y1))
}

/// The captured contents of an affine (rotate/matrix) layer: the flat op stream
/// + text items the subtree painted into the layer-local bitmap, plus its size.
/// Replay re-rasters these into a fresh layer then `blit_affine`s — mirroring the
/// live `paint_box_offset_t` layer path exactly.
#[derive(Clone, Debug)]
pub struct AffineLayer {
    pub w: u32,
    pub h: u32,
    pub ops: Vec<PaintOp>,
    pub texts: Vec<TextItem>,
}

// ── Chunk + list ────────────────────────────────────────────────────────────

/// Per-child z-order classification, captured VERBATIM from the live closures so
/// the parent can replay the 4-bucket order identically.
#[derive(Clone, Copy, Debug)]
pub struct ZMeta {
    pub effective_z: Option<i32>,
    pub is_positioned: bool,
}

/// The padding-rect clip a node with `overflow_hidden` contributes to its
/// children. `None` when the node does not clip (children inherit parent clip).
#[derive(Clone, Copy, Debug)]
pub struct ClipFrame {
    pub overflow_hidden: bool,
    /// padding rect in FINAL screen coords (offset already applied at generate).
    pub pad_rect: cv_layout::Rect,
}

/// Metadata for an affine subtree chunk (the whole subtree is an opaque layer).
#[derive(Clone, Debug)]
pub struct AffineChunk {
    pub matrix: [f32; 6],
}

#[derive(Clone, Debug)]
pub struct PaintChunk {
    pub node_id: u64,
    /// Ops emitted BEFORE the child walk (backdrop, shadow, bg, border, image,
    /// text-decoration lines) in exact emit order.
    pub ops_before_children: Vec<PaintOp>,
    /// Ops emitted AFTER the child walk (clip-path mask, filter chain).
    pub ops_after_children: Vec<PaintOp>,
    /// Text items this chunk produced (shadow copy then main), in emit order.
    pub text_items: Vec<TextItem>,
    /// Child chunk indices in DOCUMENT order (NOT z-sorted).
    pub children: Vec<u32>,
    pub z_meta: ZMeta,
    pub parent_is_flex_or_grid: bool,
    pub needs_sort: bool,
    pub clip_emit: ClipFrame,
    /// `Some` when the box has an affine transform: the whole subtree was
    /// rasterised into a layer and this single op blits it. When set, the chunk
    /// has NO before/after ops or text of its own; the [`AffineChunk::matrix`] +
    /// the affine op carry the subtree.
    pub affine: Option<AffineChunk>,
    pub affine_op: Option<PaintOp>,
    pub visibility_hidden: bool,
    pub opacity: f32,
    /// Per-node content fingerprint — EXCLUDES position/transform.
    pub content_hash: u64,
    /// Bottom-up rollup of content_hash + ordered child node_ids + subtree_hashes.
    /// POSITION-INDEPENDENT: a subtree that only MOVED keeps the same value, which
    /// is what 5.4 wants (relocate the whole cached raster). Geometry changes are
    /// tracked separately by [`PaintChunk::geom_hash`].
    pub subtree_hash: u64,
    /// Bottom-up rollup of SCREEN bounds (position + size) over the subtree. Lets
    /// the diff distinguish "truly identical" (subtree_hash == AND geom_hash ==)
    /// from "moved" (subtree_hash == but geom_hash !=) without losing the
    /// position-independence of `subtree_hash`.
    pub geom_hash: u64,
    /// border rect in FINAL screen coords (for the diff move-test + 5.4 tiles).
    pub bounds: cv_layout::Rect,
    /// M5.4 — this chunk's OWN painted extent: the bounding box (FINAL screen
    /// coords) of every pixel its `ops_before_children ∪ ops_after_children ∪
    /// affine_op ∪ text_items` can touch. UNLIKE [`PaintChunk::bounds`] (the
    /// border rect), this is shadow/blur/filter/affine/text-overflow inclusive —
    /// box-shadow rings, drop-shadow blur pads, rotated affine layers and text
    /// that overflows the content box all paint OUTSIDE the border rect, so the
    /// damage region MUST be derived from this, not `bounds`. Children carry
    /// their own extents (this is NOT a subtree rollup). Empty (w==h==0) when the
    /// chunk emits nothing.
    pub paint_extent: cv_layout::Rect,
    /// M5.4 — union of `paint_extent` over THIS chunk + every descendant. The
    /// prune key for [`replay_chunk_clipped`]: a chunk whose `subtree_paint_extent`
    /// misses the damage region R contributes no pixel inside R and is skipped;
    /// a chunk whose subtree extent intersects R is replayed in full (its own ops
    /// + recursion) so overlapping unchanged chunks recomposite in correct
    /// z-order. Filled bottom-up in `generate_rec`.
    pub subtree_paint_extent: cv_layout::Rect,
}

#[derive(Clone, Debug)]
pub struct RetainedDisplayList {
    pub chunks: Vec<PaintChunk>,
    pub index: HashMap<u64, u32>,
    pub root: u32,
    pub viewport_w: u32,
    pub doc_h: u32,
}

/// Reserved top bit marking a synthetic key for an anonymous / text / generated
/// box (`node_id == None`). Real packed node_ids are `(index << 32) | generation`
/// with `index: u32`, so the top bit is never set by a real id (would need
/// 2^31 nodes). `mix(parent_id, child_doc_index)` is stable within a box-tree
/// generation, which is all the diff compares.
const SYNTH_BIT: u64 = 0x8000_0000_0000_0000;

fn synth_key(parent_id: u64, child_doc_index: usize) -> u64 {
    let mut h = Fnv::new();
    h.u64(parent_id);
    h.u64(child_doc_index as u64);
    (h.finish() & !SYNTH_BIT) | SYNTH_BIT
}

fn chunk_key(b: &LayoutBox, parent_id: u64, child_doc_index: usize) -> u64 {
    match b.node_id {
        Some(id) => id,
        None => synth_key(parent_id, child_doc_index),
    }
}

// ── Recorder sink ────────────────────────────────────────────────────────────

/// A draw sink used by [`generate`] that BOTH applies a primitive to a scratch
/// bitmap (so clip masks / affine layer rasters behave identically to the live
/// path) AND records the op into `ops`. Method signatures mirror `Bitmap`'s, so
/// the generate recursion is a near-verbatim copy of `paint_box_offset_t` with
/// `bmp` replaced by `rec`.
struct Recorder<'a> {
    bmp: &'a mut Bitmap,
    ops: Vec<PaintOp>,
}

impl<'a> Recorder<'a> {
    fn new(bmp: &'a mut Bitmap) -> Self {
        Recorder { bmp, ops: Vec::new() }
    }
    fn take(&mut self) -> Vec<PaintOp> {
        std::mem::take(&mut self.ops)
    }

    fn fill_rect(&mut self, x: i32, y: i32, w: i32, h: i32, c: Color) {
        self.bmp.fill_rect(x, y, w, h, c);
        self.ops.push(PaintOp::FillRect { x, y, w, h, c });
    }
    fn fill_rect_rounded(&mut self, x: i32, y: i32, w: i32, h: i32, radius: i32, c: Color) {
        self.bmp.fill_rect_rounded(x, y, w, h, radius, c);
        self.ops.push(PaintOp::FillRectRounded { x, y, w, h, radius, c });
    }
    fn fill_rect_rounded_ring(
        &mut self,
        x: i32,
        y: i32,
        w: i32,
        h: i32,
        outer_r: i32,
        ring_w: i32,
        c: Color,
    ) {
        self.bmp.fill_rect_rounded_ring(x, y, w, h, outer_r, ring_w, c);
        self.ops
            .push(PaintOp::FillRectRoundedRing { x, y, w, h, outer_r, ring_w, c });
    }
    fn fill_ellipse_ring_top(&mut self, x: i32, y: i32, w: i32, h: i32, ring_w: i32, c: Color) {
        self.bmp.fill_ellipse_ring_top(x, y, w, h, ring_w, c);
        self.ops.push(PaintOp::FillEllipseRingTop { x, y, w, h, ring_w, c });
    }
    fn fill_rect_gradient(
        &mut self,
        x: i32,
        y: i32,
        w: i32,
        h: i32,
        angle_deg: f32,
        from: Color,
        to: Color,
    ) {
        self.bmp.fill_rect_gradient(x, y, w, h, angle_deg, from, to);
        self.ops
            .push(PaintOp::FillRectGradient { x, y, w, h, angle_deg, from, to });
    }
    fn fill_rect_radial_gradient(
        &mut self,
        x: i32,
        y: i32,
        w: i32,
        h: i32,
        radius: i32,
        inner: Color,
        outer: Color,
    ) {
        self.bmp.fill_rect_radial_gradient(x, y, w, h, radius, inner, outer);
        self.ops
            .push(PaintOp::FillRectRadialGradient { x, y, w, h, radius, inner, outer });
    }
    fn blit_bgra(&mut self, x: i32, y: i32, iw: u32, ih: u32, pixels: &[u32]) {
        self.bmp.blit_bgra(x, y, iw, ih, pixels);
        self.ops
            .push(PaintOp::BlitBgra { x, y, iw, ih, pixels: Arc::new(pixels.to_vec()) });
    }
    fn blit_bgra_arc(&mut self, x: i32, y: i32, iw: u32, ih: u32, pixels: Arc<Vec<u32>>) {
        self.bmp.blit_bgra(x, y, iw, ih, &pixels);
        self.ops.push(PaintOp::BlitBgra { x, y, iw, ih, pixels });
    }
    fn blit_bgra_scaled(
        &mut self,
        dx: i32,
        dy: i32,
        dw: u32,
        dh: u32,
        iw: u32,
        ih: u32,
        pixels: &[u32],
    ) {
        self.bmp.blit_bgra_scaled(dx, dy, dw, dh, iw, ih, pixels);
        self.ops.push(PaintOp::BlitBgraScaled {
            dx,
            dy,
            dw,
            dh,
            iw,
            ih,
            pixels: Arc::new(pixels.to_vec()),
        });
    }
    #[allow(clippy::too_many_arguments)]
    fn blit_bgra_sprite(
        &mut self,
        box_x: i32,
        box_y: i32,
        box_w: i32,
        box_h: i32,
        off_x: i32,
        off_y: i32,
        iw: u32,
        ih: u32,
        pixels: &[u32],
    ) {
        self.bmp
            .blit_bgra_sprite(box_x, box_y, box_w, box_h, off_x, off_y, iw, ih, pixels);
        self.ops.push(PaintOp::BlitBgraSprite {
            box_x,
            box_y,
            box_w,
            box_h,
            off_x,
            off_y,
            iw,
            ih,
            pixels: Arc::new(pixels.to_vec()),
        });
    }
    #[allow(clippy::too_many_arguments)]
    fn blit_mask_tinted_scaled(
        &mut self,
        dx: i32,
        dy: i32,
        dw: u32,
        dh: u32,
        mw: u32,
        mh: u32,
        mask: &[u32],
        c: Color,
    ) {
        self.bmp.blit_mask_tinted_scaled(dx, dy, dw, dh, mw, mh, mask, c);
        self.ops.push(PaintOp::BlitMaskTintedScaled {
            dx,
            dy,
            dw,
            dh,
            mw,
            mh,
            mask: Arc::new(mask.to_vec()),
            c,
        });
    }
    fn clip_rounded_rect(&mut self, x: i32, y: i32, w: i32, h: i32, radius: i32) {
        self.bmp.clip_rounded_rect(x, y, w, h, radius);
        self.ops.push(PaintOp::ClipRoundedRect { x, y, w, h, radius });
    }
    #[allow(clippy::too_many_arguments)]
    fn clip_inset(&mut self, x: i32, y: i32, w: i32, h: i32, t: i32, r: i32, b: i32, l: i32) {
        self.bmp.clip_inset(x, y, w, h, t, r, b, l);
        self.ops.push(PaintOp::ClipInset { x, y, w, h, t, r, b, l });
    }
    fn clip_circle(&mut self, x: i32, y: i32, w: i32, h: i32, cx: f32, cy: f32, radius: f32) {
        self.bmp.clip_circle(x, y, w, h, cx, cy, radius);
        self.ops.push(PaintOp::ClipCircle { x, y, w, h, cx, cy, radius });
    }
    fn clip_polygon(&mut self, x: i32, y: i32, w: i32, h: i32, pts: &[(f32, f32)]) {
        self.bmp.clip_polygon(x, y, w, h, pts);
        self.ops.push(PaintOp::ClipPolygon { x, y, w, h, pts: pts.to_vec() });
    }
    fn apply_filter_rect(&mut self, x: i32, y: i32, w: i32, h: i32, op: FilterOp) {
        self.bmp.apply_filter_rect(x, y, w, h, op);
        self.ops.push(PaintOp::ApplyFilterRect { x, y, w, h, op: FilterOpRec::from_gfx(op) });
    }
}

// ── generate() — sibling recursion of paint_box_offset_t ────────────────────

/// Build the retained display list from a layout tree. Parallels
/// `paint_box_offset_t` branch-for-branch, recording each primitive (via
/// [`Recorder`]) with FINAL screen-space args + capturing chunk metadata,
/// content_hash (position/transform-excluding) and a subtree_hash rollup.
/// The (viewport_w, doc_h) the produced bitmap / RDL will have for `lb` under
/// `cfg` — the SAME formula `generate` / `oracle_live_paint` /
/// `bake_layout_into_paint_inner` use. Exposed so the damage path can do a
/// CHEAP pre-check (cached-bitmap dims vs this) BEFORE paying `generate()` on a
/// frame whose document resized/reflowed (which would full-bake anyway). Keeping
/// it here guarantees the pre-check can never drift from `generate`'s real dims.
pub fn expected_dims(lb: &LayoutBox, cfg: &cv_layout::LayoutConfig) -> (u32, u32) {
    let bmp_w = cfg.viewport_w as u32;
    let layout_bottom = lb.content.y + lb.content.h;
    let document_h = layout_bottom.max(cfg.viewport_h);
    let max_bitmap_h: u32 = 100_000;
    let bmp_h = (document_h as u32).min(max_bitmap_h).max(1);
    (bmp_w, bmp_h)
}

pub fn generate(lb: &LayoutBox, cfg: &cv_layout::LayoutConfig) -> RetainedDisplayList {
    let (bmp_w, bmp_h) = expected_dims(lb, cfg);
    let mut scratch = Bitmap::new(bmp_w, bmp_h);
    scratch.clear(Color::WHITE);

    let mut list = RetainedDisplayList {
        chunks: Vec::new(),
        index: HashMap::new(),
        root: 0,
        viewport_w: bmp_w,
        doc_h: bmp_h,
    };
    let mut rec = Recorder::new(&mut scratch);
    let mut texts: Vec<TextItem> = Vec::new();
    let root = generate_rec(lb, &mut list, &mut rec, &mut texts, 0.0, 0.0, None, false, 0, 0);
    list.root = root;
    list
}

/// Returns the chunk index for `b`. Mirrors `paint_box_offset_t`.
#[allow(clippy::too_many_arguments)]
fn generate_rec(
    b: &LayoutBox,
    list: &mut RetainedDisplayList,
    rec: &mut Recorder<'_>,
    texts: &mut Vec<TextItem>,
    parent_off_x: f32,
    parent_off_y: f32,
    clip_rect: Option<cv_layout::Rect>,
    suppress_self_transform: bool,
    parent_id: u64,
    doc_index: usize,
) -> u32 {
    let key = chunk_key(b, parent_id, doc_index);

    let my_idx = list.chunks.len() as u32;
    let content_hash = compute_content_hash(b);
    let self_translate_x = if suppress_self_transform { 0.0 } else { b.translate_x_px };
    let self_translate_y = if suppress_self_transform { 0.0 } else { b.translate_y_px };
    let bounds_screen = cv_layout::Rect {
        x: b.border_rect().x + parent_off_x + self_translate_x,
        y: b.border_rect().y + parent_off_y + self_translate_y,
        w: b.border_rect().w,
        h: b.border_rect().h,
    };
    list.chunks.push(PaintChunk {
        node_id: key,
        ops_before_children: Vec::new(),
        ops_after_children: Vec::new(),
        text_items: Vec::new(),
        children: Vec::new(),
        z_meta: ZMeta { effective_z: None, is_positioned: false },
        parent_is_flex_or_grid: false,
        needs_sort: false,
        clip_emit: ClipFrame { overflow_hidden: false, pad_rect: cv_layout::Rect::default() },
        affine: None,
        affine_op: None,
        visibility_hidden: b.visibility_hidden,
        opacity: b.opacity.clamp(0.0, 1.0),
        content_hash,
        subtree_hash: content_hash,
        geom_hash: 0,
        bounds: bounds_screen,
        paint_extent: cv_layout::Rect::default(),
        subtree_paint_extent: cv_layout::Rect::default(),
    });
    list.index.insert(key, my_idx);

    let off_x = parent_off_x + self_translate_x;
    let off_y = parent_off_y + self_translate_y;

    if b.visibility_hidden {
        return my_idx;
    }
    let opacity = b.opacity.clamp(0.0, 1.0);
    if opacity < 0.01 {
        return my_idx;
    }

    // ── Affine layer path (rotate/matrix) ──
    if !suppress_self_transform && b.has_affine_transform() {
        let bb = b.subtree_bounds();
        let pad = 2.0f32;
        let lx0 = bb.x - pad;
        let ly0 = bb.y - pad;
        let lw = (bb.w + pad * 2.0).ceil().max(1.0);
        let lh = (bb.h + pad * 2.0).ceil().max(1.0);
        const MAX_LAYER_DIM: f32 = 4096.0;
        if lw <= MAX_LAYER_DIM && lh <= MAX_LAYER_DIM {
            let layer_w = lw as u32;
            let layer_h = lh as u32;
            let mut layer_bmp = Bitmap::new(layer_w, layer_h);
            layer_bmp.clear(Color::TRANSPARENT);
            let mut layer_texts: Vec<TextItem> = Vec::new();
            let mut layer_list = RetainedDisplayList {
                chunks: Vec::new(),
                index: HashMap::new(),
                root: 0,
                viewport_w: layer_w,
                doc_h: layer_h,
            };
            {
                let mut layer_rec = Recorder::new(&mut layer_bmp);
                let lroot = generate_rec(
                    b,
                    &mut layer_list,
                    &mut layer_rec,
                    &mut layer_texts,
                    -lx0,
                    -ly0,
                    None,
                    true,
                    parent_id,
                    doc_index,
                );
                layer_list.root = lroot;
            }
            let mut flat_ops: Vec<PaintOp> = Vec::new();
            flatten_layer_ops(&layer_list, layer_list.root, &mut flat_ops);

            let a = b.transform_affine();
            let (m0, m1, m2, m3, e, f) = (a[0], a[1], a[2], a[3], a[4], a[5]);
            let br = b.border_rect();
            let (origin_xp, origin_yp) = match b.transform_origin {
                Some((xp, yp)) => (xp, yp),
                None => (cv_layout::BgPos::Pct(50.0), cv_layout::BgPos::Pct(50.0)),
            };
            let ox = br.x + origin_xp.resolve(br.w, 0.0) + parent_off_x;
            let oy = br.y + origin_yp.resolve(br.h, 0.0) + parent_off_y;
            let kx = lx0 + parent_off_x;
            let ky = ly0 + parent_off_y;
            let dx = kx - ox;
            let dy = ky - oy;
            let me = ox + (m0 * dx + m2 * dy) + e;
            let mf = oy + (m1 * dx + m3 * dy) + f;
            let affine_op = PaintOp::BlitAffine {
                matrix: [m0, m1, m2, m3, me, mf],
                layer: Box::new(AffineLayer {
                    w: layer_w,
                    h: layer_h,
                    ops: flat_ops,
                    texts: layer_texts,
                }),
            };
            affine_op.replay_into(rec.bmp);
            // M5.4 paint extent for the affine chunk = the transformed layer bbox
            // (rotation makes it larger than the border rect). The whole subtree
            // lives inside this single op, so subtree_paint_extent == paint_extent.
            let ext = affine_op.extent_screen();
            let ext_rect = extent_to_rect(ext);
            // M5.4 — the affine chunk returns EARLY, bypassing the normal-path
            // hash rollup, so its content_hash/subtree_hash/geom_hash would never
            // reflect the transform (content_hash excludes transform by design) nor
            // the layer's content. For the diff to detect a rotate/matrix change
            // (which re-rasters the BlitAffine layer + changes the transformed
            // bbox), fold the FINAL matrix + the layer's own subtree_hash into this
            // chunk's hashes here. Without this a `rotate_deg 0→20` produced an
            // EMPTY diff and the incremental path wrongly reused the cache.
            let mut ah = Fnv::new();
            ah.u64(content_hash);
            for v in [m0, m1, m2, m3, me, mf] {
                ah.f32(v);
            }
            ah.u32(layer_w);
            ah.u32(layer_h);
            if let Some(lr) = layer_list.chunks.get(layer_list.root as usize) {
                ah.u64(lr.subtree_hash);
            }
            let affine_hash = ah.finish();
            let mut gh = Fnv::new();
            gh.f32(ext_rect.x);
            gh.f32(ext_rect.y);
            gh.f32(ext_rect.w);
            gh.f32(ext_rect.h);
            for v in [m0, m1, m2, m3, me, mf] {
                gh.f32(v);
            }
            let affine_geom = gh.finish();
            // M5.4 — the affine early-return BYPASSES the normal-path function
            // tail (~line 1151) that captures this chunk's stacking metadata. With
            // z_meta left at its construction-time default ({None, false}), replay's
            // painted_child_order (which re-derives paint order PURELY from each
            // child chunk's z_meta — see painted_child_order + flatten_layer_ops)
            // mis-buckets an affine child that is actually positioned / z-indexed /
            // a flex|grid item: it falls into bucket B (doc order) instead of
            // bucket A/C/D. When two overlapping affine siblings live in different
            // real buckets, replay paints them in the WRONG order vs the live
            // painter (and vs generate's own scratch, which orders directly off the
            // live LayoutBox). Capture the SAME z_meta + parent_is_flex_or_grid the
            // non-affine path stores at the tail, computed inline to mirror it
            // EXACTLY (lines ~1042-1052 + ~1151), so affine and non-affine children
            // bucket identically and replay byte-matches live.
            let self_positioned = !matches!(b.position, cv_layout::Position::Static);
            let self_parent_is_flex_or_grid = b.is_flex || b.is_grid;
            let self_effective_z = if self_positioned || self_parent_is_flex_or_grid {
                b.z_index
            } else {
                None
            };
            let c = &mut list.chunks[my_idx as usize];
            c.affine = Some(AffineChunk { matrix: [m0, m1, m2, m3, me, mf] });
            c.affine_op = Some(affine_op);
            c.paint_extent = ext_rect;
            c.subtree_paint_extent = ext_rect;
            c.content_hash = affine_hash;
            c.subtree_hash = affine_hash;
            c.geom_hash = affine_geom;
            c.z_meta = ZMeta { effective_z: self_effective_z, is_positioned: self_positioned };
            c.parent_is_flex_or_grid = self_parent_is_flex_or_grid;
            return my_idx;
        }
    }

    let scale_alpha = |a: u8| -> u8 { ((a as f32) * opacity).round() as u8 };
    let ox = off_x as i32;
    let oy = off_y as i32;
    let border_rect = cv_layout::Rect {
        x: b.border_rect().x + off_x,
        y: b.border_rect().y + off_y,
        w: b.border_rect().w,
        h: b.border_rect().h,
    };
    if let Some(clip) = clip_rect {
        if !super::rects_intersect(&clip, &border_rect) {
            return my_idx;
        }
    }

    if !b.backdrop_filters.is_empty() {
        let r = b.border_rect();
        let mut fx = r.x as i32 + ox;
        let mut fy = r.y as i32 + oy;
        let mut fw = r.w.max(0.0) as i32;
        let mut fh = r.h.max(0.0) as i32;
        if let Some(clip) = clip_rect {
            if let Some(ir) = super::intersect_rect(
                &cv_layout::Rect { x: fx as f32, y: fy as f32, w: fw as f32, h: fh as f32 },
                &clip,
            ) {
                fx = ir.x as i32;
                fy = ir.y as i32;
                fw = ir.w as i32;
                fh = ir.h as i32;
            } else {
                fw = 0;
                fh = 0;
            }
        }
        if fw > 0 && fh > 0 {
            for fe in &b.backdrop_filters {
                if let Some(op) = filter_effect_to_op(fe) {
                    rec.apply_filter_rect(fx, fy, fw, fh, op);
                }
            }
        }
    }

    if b.background.is_some()
        || b.background_gradient.is_some()
        || b.background_radial_gradient.is_some()
        || b.box_shadow.is_some()
        || b.background_image.is_some()
    {
        if let Some(sh) = b.box_shadow {
            gen_box_shadow(b, rec, sh, ox, oy, &scale_alpha);
        }
        let r = b.border_rect();
        let radius = super::used_border_radius_px(b, r.w as i32, r.h as i32);
        if let Some(g) = b.background_gradient {
            let from_a = scale_alpha(g.from.a);
            let to_a = scale_alpha(g.to.a);
            if from_a > 0 || to_a > 0 {
                let from = Color { r: g.from.r, g: g.from.g, b: g.from.b, a: from_a };
                let to = Color { r: g.to.r, g: g.to.g, b: g.to.b, a: to_a };
                rec.fill_rect_gradient(r.x as i32 + ox, r.y as i32 + oy, r.w as i32, r.h as i32, g.angle_deg, from, to);
            }
        }
        if let Some(g) = b.background_radial_gradient {
            let from_a = scale_alpha(g.from.a);
            let to_a = scale_alpha(g.to.a);
            if from_a > 0 || to_a > 0 {
                let from = Color { r: g.from.r, g: g.from.g, b: g.from.b, a: from_a };
                let to = Color { r: g.to.r, g: g.to.g, b: g.to.b, a: to_a };
                rec.fill_rect_radial_gradient(r.x as i32 + ox, r.y as i32 + oy, r.w as i32, r.h as i32, radius, from, to);
            }
        }
        if b.background_gradient.is_none() && b.background_radial_gradient.is_none() {
            if let Some(bg) = b.background {
                let alpha = scale_alpha(bg.a);
                if alpha != 0 {
                    let c = Color { r: bg.r, g: bg.g, b: bg.b, a: alpha };
                    if b.has_mask_url {
                        if let Some(mask) = &b.mask_image {
                            rec.blit_mask_tinted_scaled(
                                r.x as i32 + ox,
                                r.y as i32 + oy,
                                r.w.max(1.0) as u32,
                                r.h.max(1.0) as u32,
                                mask.width,
                                mask.height,
                                &mask.pixels,
                                c,
                            );
                        }
                    } else if radius > 0 {
                        rec.fill_rect_rounded(r.x as i32 + ox, r.y as i32 + oy, r.w as i32, r.h as i32, radius, c);
                    } else {
                        rec.fill_rect(r.x as i32 + ox, r.y as i32 + oy, r.w as i32, r.h as i32, c);
                    }
                }
            }
        }
        if let Some(img) = &b.background_image {
            if img.width > 0 && img.height > 0 {
                gen_background_image(b, rec, img, ox, oy);
            }
        }
        if b.background_gradient.is_some()
            || b.background_radial_gradient.is_some()
            || b.background_image.is_some()
        {
            let br = b.border_rect();
            let radius = super::used_border_radius_px(b, br.w as i32, br.h as i32);
            if radius > 0 {
                rec.clip_rounded_rect(br.x as i32 + ox, br.y as i32 + oy, br.w as i32, br.h as i32, radius);
            }
        }
    }

    gen_borders(b, rec, ox, oy, &scale_alpha);

    if let Some(img) = &b.embedded_image {
        gen_embedded_image(b, rec, img, ox, oy);
    }

    if let cv_layout::BoxKind::Text(t) = &b.kind {
        gen_text(b, rec, texts, t, ox, oy, &scale_alpha, list, my_idx);
    }

    let before = rec.take();

    let next_clip = if b.overflow_hidden {
        let clip_source = cv_layout::Rect {
            x: b.padding_rect().x + off_x,
            y: b.padding_rect().y + off_y,
            w: b.padding_rect().w,
            h: b.padding_rect().h,
        };
        match clip_rect {
            Some(clip) => super::intersect_rect(&clip, &clip_source),
            None => Some(clip_source),
        }
    } else {
        clip_rect
    };

    let is_positioned = |c: &LayoutBox| !matches!(c.position, cv_layout::Position::Static);
    let parent_is_flex_or_grid = b.is_flex || b.is_grid;
    let effective_z = |c: &LayoutBox| -> Option<i32> {
        if is_positioned(c) || parent_is_flex_or_grid {
            c.z_index
        } else {
            None
        }
    };
    let needs_sort =
        b.children.iter().any(|c| effective_z(c).is_some()) || b.children.iter().any(is_positioned);

    let painted_order: Vec<usize> = if needs_sort {
        let mut bucket_a: Vec<usize> = Vec::new();
        let mut bucket_b: Vec<usize> = Vec::new();
        let mut bucket_c: Vec<usize> = Vec::new();
        let mut bucket_d: Vec<usize> = Vec::new();
        for (i, c) in b.children.iter().enumerate() {
            match effective_z(c) {
                Some(z) if z < 0 => bucket_a.push(i),
                Some(z) if z > 0 => bucket_d.push(i),
                Some(_) => {
                    if is_positioned(c) {
                        bucket_c.push(i);
                    } else {
                        bucket_b.push(i);
                    }
                }
                None => {
                    if is_positioned(c) {
                        bucket_c.push(i);
                    } else {
                        bucket_b.push(i);
                    }
                }
            }
        }
        bucket_a.sort_by_key(|&i| effective_z(&b.children[i]).unwrap_or(0));
        bucket_d.sort_by_key(|&i| effective_z(&b.children[i]).unwrap_or(0));
        bucket_a
            .into_iter()
            .chain(bucket_b)
            .chain(bucket_c)
            .chain(bucket_d)
            .collect()
    } else {
        (0..b.children.len()).collect()
    };

    let mut child_idx_by_doc: Vec<Option<u32>> = vec![None; b.children.len()];
    for &i in &painted_order {
        let child = &b.children[i];
        let cidx = generate_rec(child, list, rec, texts, off_x, off_y, next_clip, false, key, i);
        child_idx_by_doc[i] = Some(cidx);
    }

    if let Some(shape) = &b.clip_shape {
        let fx = border_rect.x as i32;
        let fy = border_rect.y as i32;
        let fw = border_rect.w as i32;
        let fh = border_rect.h as i32;
        if fw > 0 && fh > 0 {
            match shape {
                cv_layout::ClipShape::Inset { top_px, right_px, bottom_px, left_px } => {
                    rec.clip_inset(fx, fy, fw, fh, *top_px as i32, *right_px as i32, *bottom_px as i32, *left_px as i32);
                }
                cv_layout::ClipShape::Circle { radius_px, cx_px, cy_px } => {
                    let sx = fw as f32 / 100.0;
                    let sy = fh as f32 / 100.0;
                    rec.clip_circle(fx, fy, fw, fh, cx_px * sx, cy_px * sy, radius_px * sx.min(sy));
                }
                cv_layout::ClipShape::Polygon(pts) => {
                    let sx = fw as f32 / 100.0;
                    let sy = fh as f32 / 100.0;
                    let scaled: Vec<(f32, f32)> = pts.iter().map(|(x, y)| (x * sx, y * sy)).collect();
                    rec.clip_polygon(fx, fy, fw, fh, &scaled);
                }
            }
        }
    }
    if !b.filters.is_empty() {
        let fx = border_rect.x as i32;
        let fy = border_rect.y as i32;
        let fw = border_rect.w as i32;
        let fh = border_rect.h as i32;
        if fw > 0 && fh > 0 {
            for fe in &b.filters {
                match *fe {
                    cv_layout::FilterEffect::DropShadow(sh) => {
                        gen_drop_shadow(b, rec, &border_rect, sh, fw, fh);
                    }
                    other => {
                        if let Some(op) = filter_effect_to_op(&other) {
                            rec.apply_filter_rect(fx, fy, fw, fh, op);
                        }
                    }
                }
            }
        }
    }

    let after = rec.take();

    let children_doc: Vec<u32> = child_idx_by_doc.iter().filter_map(|c| *c).collect();
    {
        let c = &mut list.chunks[my_idx as usize];
        c.ops_before_children = before;
        c.ops_after_children = after;
        c.children = children_doc;
        c.z_meta = ZMeta { effective_z: effective_z(b), is_positioned: is_positioned(b) };
        c.parent_is_flex_or_grid = parent_is_flex_or_grid;
        c.needs_sort = needs_sort;
        c.clip_emit = ClipFrame {
            overflow_hidden: b.overflow_hidden,
            pad_rect: cv_layout::Rect {
                x: b.padding_rect().x + off_x,
                y: b.padding_rect().y + off_y,
                w: b.padding_rect().w,
                h: b.padding_rect().h,
            },
        };
    }

    let mut sh = Fnv::new();
    sh.u64(content_hash);
    // geom rollup: this node's screen bounds (position + size) + ordered child
    // geom_hashes, so a MOVE anywhere in the subtree changes geom_hash while
    // subtree_hash (content only) stays put.
    let mut gh = Fnv::new();
    let my_bounds = list.chunks[my_idx as usize].bounds;
    gh.f32(my_bounds.x);
    gh.f32(my_bounds.y);
    gh.f32(my_bounds.w);
    gh.f32(my_bounds.h);
    let child_indices: Vec<u32> = list.chunks[my_idx as usize].children.clone();
    sh.u32(child_indices.len() as u32);
    gh.u32(child_indices.len() as u32);
    for cidx in child_indices {
        let child = &list.chunks[cidx as usize];
        sh.u64(child.node_id);
        sh.u64(child.subtree_hash);
        gh.u64(child.node_id);
        gh.u64(child.geom_hash);
    }
    list.chunks[my_idx as usize].subtree_hash = sh.finish();
    list.chunks[my_idx as usize].geom_hash = gh.finish();

    // M5.4 — own paint_extent = bbox of this chunk's own ops + text items (NOT
    // children; they carry their own). subtree_paint_extent = own ∪ every child's
    // subtree_paint_extent (bottom-up; children are already finalised above).
    {
        let mut own: Option<(f32, f32, f32, f32)> = None;
        let child_idxs: Vec<u32> = list.chunks[my_idx as usize].children.clone();
        {
            let c = &list.chunks[my_idx as usize];
            for op in c.ops_before_children.iter().chain(c.ops_after_children.iter()) {
                own = union_extent(own, op.extent_screen());
            }
            for ti in &c.text_items {
                own = union_extent(own, text_item_extent(ti));
            }
        }
        let mut subtree = own;
        for cidx in &child_idxs {
            let ce = rect_to_extent(list.chunks[*cidx as usize].subtree_paint_extent);
            subtree = union_extent(subtree, ce);
        }
        list.chunks[my_idx as usize].paint_extent = extent_to_rect(own);
        list.chunks[my_idx as usize].subtree_paint_extent = extent_to_rect(subtree);
    }

    my_idx
}

/// Convert an optional half-open `(x0,y0,x1,y1)` extent to a `Rect`. `None` ⇒ the
/// empty rect (w==h==0), which `rect_intersects_r` treats as "touches nothing".
fn extent_to_rect(e: Option<(f32, f32, f32, f32)>) -> cv_layout::Rect {
    match e {
        Some((x0, y0, x1, y1)) => cv_layout::Rect { x: x0, y: y0, w: x1 - x0, h: y1 - y0 },
        None => cv_layout::Rect::default(),
    }
}

/// Inverse of [`extent_to_rect`]: an empty rect (w<=0 or h<=0) ⇒ `None`.
fn rect_to_extent(r: cv_layout::Rect) -> Option<(f32, f32, f32, f32)> {
    if r.w > 0.0 && r.h > 0.0 {
        Some((r.x, r.y, r.x + r.w, r.y + r.h))
    } else {
        None
    }
}

/// Children in PAINTED (z-bucket) order, reproduced from captured ZMeta.
fn painted_child_order(list: &RetainedDisplayList, ci: u32) -> Vec<u32> {
    let c = &list.chunks[ci as usize];
    if !c.needs_sort {
        return c.children.clone();
    }
    let mut bucket_a: Vec<u32> = Vec::new();
    let mut bucket_b: Vec<u32> = Vec::new();
    let mut bucket_c: Vec<u32> = Vec::new();
    let mut bucket_d: Vec<u32> = Vec::new();
    for &cidx in &c.children {
        let cc = &list.chunks[cidx as usize];
        match cc.z_meta.effective_z {
            Some(z) if z < 0 => bucket_a.push(cidx),
            Some(z) if z > 0 => bucket_d.push(cidx),
            Some(_) => {
                if cc.z_meta.is_positioned {
                    bucket_c.push(cidx);
                } else {
                    bucket_b.push(cidx);
                }
            }
            None => {
                if cc.z_meta.is_positioned {
                    bucket_c.push(cidx);
                } else {
                    bucket_b.push(cidx);
                }
            }
        }
    }
    bucket_a.sort_by_key(|&cidx| list.chunks[cidx as usize].z_meta.effective_z.unwrap_or(0));
    bucket_d.sort_by_key(|&cidx| list.chunks[cidx as usize].z_meta.effective_z.unwrap_or(0));
    bucket_a
        .into_iter()
        .chain(bucket_b)
        .chain(bucket_c)
        .chain(bucket_d)
        .collect()
}

fn flatten_layer_ops(list: &RetainedDisplayList, ci: u32, out: &mut Vec<PaintOp>) {
    let c = &list.chunks[ci as usize];
    if c.visibility_hidden || c.opacity < 0.01 {
        return;
    }
    if let Some(op) = &c.affine_op {
        out.push(op.clone());
        return;
    }
    out.extend(c.ops_before_children.iter().cloned());
    for cidx in painted_child_order(list, ci) {
        flatten_layer_ops(list, cidx, out);
    }
    out.extend(c.ops_after_children.iter().cloned());
}

fn filter_effect_to_op(fe: &cv_layout::FilterEffect) -> Option<FilterOp> {
    match *fe {
        cv_layout::FilterEffect::Blur(r) => Some(FilterOp::Blur(r)),
        cv_layout::FilterEffect::Brightness(a) => Some(FilterOp::Brightness(a)),
        cv_layout::FilterEffect::Contrast(a) => Some(FilterOp::Contrast(a)),
        cv_layout::FilterEffect::Grayscale(a) => Some(FilterOp::Grayscale(a)),
        cv_layout::FilterEffect::Invert(a) => Some(FilterOp::Invert(a)),
        cv_layout::FilterEffect::Sepia(a) => Some(FilterOp::Sepia(a)),
        cv_layout::FilterEffect::Saturate(a) => Some(FilterOp::Saturate(a)),
        cv_layout::FilterEffect::HueRotate(d) => Some(FilterOp::HueRotate(d)),
        cv_layout::FilterEffect::Opacity(a) => Some(FilterOp::Opacity(a)),
        cv_layout::FilterEffect::DropShadow(_) => None,
    }
}

// ── gen_* sub-functions: faithful copies of the live painter branches ───────

fn gen_box_shadow(
    b: &LayoutBox,
    rec: &mut Recorder<'_>,
    sh: cv_layout::BoxShadow,
    ox: i32,
    oy: i32,
    scale_alpha: &dyn Fn(u8) -> u8,
) {
    let r = b.border_rect();
    let spread = sh.spread_px as i32;
    if sh.inset {
        let clip_x0 = r.x as i32 + ox;
        let clip_y0 = r.y as i32 + oy;
        let clip_x1 = clip_x0 + r.w as i32;
        let clip_y1 = clip_y0 + r.h as i32;
        let hole_x0 = clip_x0 + spread + sh.offset_x_px as i32;
        let hole_y0 = clip_y0 + spread + sh.offset_y_px as i32;
        let hole_x1 = clip_x1 - spread + sh.offset_x_px as i32;
        let hole_y1 = clip_y1 - spread + sh.offset_y_px as i32;
        let sigma = (sh.blur_px * 0.5).max(0.0);
        let steps = sigma.round().max(0.0) as i32;
        let paint_inset_rect = |rec: &mut Recorder<'_>, x: i32, y: i32, w: i32, h: i32, c: Color, cx0: i32, cy0: i32, cx1: i32, cy1: i32| {
            let rx0 = x.max(cx0);
            let ry0 = y.max(cy0);
            let rx1 = (x + w).min(cx1);
            let ry1 = (y + h).min(cy1);
            if rx1 > rx0 && ry1 > ry0 {
                rec.fill_rect(rx0, ry0, rx1 - rx0, ry1 - ry0, c);
            }
        };
        if steps == 0 {
            let sc = Color { r: sh.color.r, g: sh.color.g, b: sh.color.b, a: scale_alpha(sh.color.a) };
            paint_inset_rect(rec, clip_x0, clip_y0, clip_x1 - clip_x0, hole_y0 - clip_y0, sc, clip_x0, clip_y0, clip_x1, clip_y1);
            paint_inset_rect(rec, clip_x0, hole_y1, clip_x1 - clip_x0, clip_y1 - hole_y1, sc, clip_x0, clip_y0, clip_x1, clip_y1);
            paint_inset_rect(rec, clip_x0, hole_y0, hole_x0 - clip_x0, hole_y1 - hole_y0, sc, clip_x0, clip_y0, clip_x1, clip_y1);
            paint_inset_rect(rec, hole_x1, hole_y0, clip_x1 - hole_x1, hole_y1 - hole_y0, sc, clip_x0, clip_y0, clip_x1, clip_y1);
        } else {
            let base_a = sh.color.a as f32;
            for k in (1..=steps).rev() {
                let t = k as f32 / steps as f32;
                let falloff = (-(t * 2.0) * (t * 2.0) * 0.5).exp();
                let a = (base_a * falloff).round().clamp(0.0, 255.0) as u8;
                if a == 0 {
                    continue;
                }
                let sc = Color { r: sh.color.r, g: sh.color.g, b: sh.color.b, a: scale_alpha(a) };
                let inner_x0 = hole_x0 - k;
                let inner_y0 = hole_y0 - k;
                let inner_x1 = hole_x1 + k;
                let inner_y1 = hole_y1 + k;
                let rw = clip_x1 - clip_x0;
                let rh_top = inner_y0 - clip_y0;
                paint_inset_rect(rec, clip_x0, clip_y0, rw, rh_top, sc, clip_x0, clip_y0, clip_x1, clip_y1);
                let rh_bot = clip_y1 - inner_y1;
                paint_inset_rect(rec, clip_x0, inner_y1, rw, rh_bot, sc, clip_x0, clip_y0, clip_x1, clip_y1);
                let rl_w = inner_x0 - clip_x0;
                paint_inset_rect(rec, clip_x0, inner_y0, rl_w, inner_y1 - inner_y0, sc, clip_x0, clip_y0, clip_x1, clip_y1);
                let rr_w = clip_x1 - inner_x1;
                paint_inset_rect(rec, inner_x1, inner_y0, rr_w, inner_y1 - inner_y0, sc, clip_x0, clip_y0, clip_x1, clip_y1);
            }
        }
    } else {
        let sx = r.x as i32 + ox + sh.offset_x_px as i32 - spread;
        let sy = r.y as i32 + oy + sh.offset_y_px as i32 - spread;
        let sw = r.w as i32 + 2 * spread;
        let sh_h = r.h as i32 + 2 * spread;
        let radius = super::used_border_radius_px(b, sw, sh_h);
        let sigma = (sh.blur_px * 0.5).max(0.0);
        let steps = sigma.round().max(0.0) as i32;
        if steps == 0 {
            let sc = Color { r: sh.color.r, g: sh.color.g, b: sh.color.b, a: scale_alpha(sh.color.a) };
            if radius > 0 {
                rec.fill_rect_rounded(sx, sy, sw, sh_h, radius, sc);
            } else {
                rec.fill_rect(sx, sy, sw, sh_h, sc);
            }
        } else {
            let base_a = sh.color.a as f32;
            for k in (1..=steps).rev() {
                let t = k as f32 / steps as f32;
                let falloff = (-(t * 2.0) * (t * 2.0) * 0.5).exp();
                let a = (base_a * falloff).round().clamp(0.0, 255.0) as u8;
                if a == 0 {
                    continue;
                }
                let sc = Color { r: sh.color.r, g: sh.color.g, b: sh.color.b, a: scale_alpha(a) };
                let rx = sx - k;
                let ry = sy - k;
                let rw = sw + 2 * k;
                let rh = sh_h + 2 * k;
                if radius > 0 {
                    rec.fill_rect_rounded_ring(rx, ry, rw, rh, radius + k, 1, sc);
                } else if rw > 0 && rh > 0 {
                    rec.fill_rect(rx, ry, rw, 1, sc);
                    rec.fill_rect(rx, ry + rh - 1, rw, 1, sc);
                    if rh > 2 {
                        rec.fill_rect(rx, ry + 1, 1, rh - 2, sc);
                        rec.fill_rect(rx + rw - 1, ry + 1, 1, rh - 2, sc);
                    }
                }
            }
        }
    }
}

fn gen_background_image(
    b: &LayoutBox,
    rec: &mut Recorder<'_>,
    img: &cv_layout::EmbeddedImage,
    ox: i32,
    oy: i32,
) {
    let r = b.border_rect();
    let dx = r.x as i32 + ox;
    let dy = r.y as i32 + oy;
    let dw = r.w as i32;
    let dh = r.h as i32;
    if dw <= 0 || dh <= 0 {
        return;
    }
    let iw = img.width as i32;
    let ih = img.height as i32;
    let iwf = img.width as f32;
    let ihf = img.height as f32;
    let dwf = dw as f32;
    let dhf = dh as f32;
    let is_cover_or_contain = matches!(
        b.background_size,
        Some(cv_layout::BgSize::Cover) | Some(cv_layout::BgSize::Contain)
    );
    if is_cover_or_contain {
        let (tw, th) = match b.background_size.as_ref().unwrap() {
            cv_layout::BgSize::Cover => {
                let scale = if iwf > 0.0 && ihf > 0.0 { (dwf / iwf).max(dhf / ihf) } else { 1.0 };
                (iwf * scale, ihf * scale)
            }
            cv_layout::BgSize::Contain => {
                let scale = if iwf > 0.0 && ihf > 0.0 { (dwf / iwf).min(dhf / ihf) } else { 1.0 };
                (iwf * scale, ihf * scale)
            }
            _ => unreachable!(),
        };
        let tw_px = tw.max(0.0).round() as u32;
        let th_px = th.max(0.0).round() as u32;
        let (pos_x, pos_y) = if let Some((px, py)) = b.background_position {
            let off_x = px.resolve(dwf, tw).round() as i32;
            let off_y = py.resolve(dhf, th).round() as i32;
            (off_x, off_y)
        } else {
            (((dwf - tw) * 0.5).round() as i32, ((dhf - th) * 0.5).round() as i32)
        };
        if tw_px > 0 && th_px > 0 {
            let mut tmp_bmp = Bitmap { width: tw_px, height: th_px, pixels: vec![0u32; (tw_px as usize) * (th_px as usize)] };
            tmp_bmp.blit_bgra_scaled(0, 0, tw_px, th_px, img.width, img.height, &img.pixels);
            rec.blit_bgra_sprite(dx, dy, dw, dh, pos_x, pos_y, tmp_bmp.width, tmp_bmp.height, &tmp_bmp.pixels);
        }
    } else {
        match b.background_repeat {
            cv_layout::BackgroundRepeat::NoRepeat => {
                if let Some((px, py)) = b.background_position {
                    let off_x = px.resolve(dw as f32, img.width as f32).round() as i32;
                    let off_y = py.resolve(dh as f32, img.height as f32).round() as i32;
                    rec.blit_bgra_sprite(dx, dy, dw, dh, off_x, off_y, img.width, img.height, &img.pixels);
                } else if let Some(ref bg_size) = b.background_size {
                    let (tw, th) = match bg_size {
                        cv_layout::BgSize::Explicit(bw, bh) => {
                            let rw = bw.as_ref().map(|l| l.resolve(dwf));
                            let rh = bh.as_ref().map(|l| l.resolve(dhf));
                            match (rw, rh) {
                                (Some(w), Some(h)) => (w, h),
                                (Some(w), None) => (w, if iwf > 0.0 { w * ihf / iwf } else { ihf }),
                                (None, Some(h)) => (if ihf > 0.0 { h * iwf / ihf } else { iwf }, h),
                                (None, None) => (iwf, ihf),
                            }
                        }
                        _ => (iwf, ihf),
                    };
                    rec.blit_bgra_scaled(dx, dy, tw.max(0.0).round() as u32, th.max(0.0).round() as u32, img.width, img.height, &img.pixels);
                } else {
                    rec.blit_bgra(dx, dy, img.width, img.height, &img.pixels);
                }
            }
            cv_layout::BackgroundRepeat::RepeatX => {
                let mut x = 0;
                while x < dw {
                    rec.blit_bgra(dx + x, dy, img.width, img.height, &img.pixels);
                    x += iw.max(1);
                }
            }
            cv_layout::BackgroundRepeat::RepeatY => {
                let mut y = 0;
                while y < dh {
                    rec.blit_bgra(dx, dy + y, img.width, img.height, &img.pixels);
                    y += ih.max(1);
                }
            }
            cv_layout::BackgroundRepeat::Repeat => {
                if iw >= dw && ih >= dh {
                    rec.blit_bgra(dx, dy, img.width, img.height, &img.pixels);
                } else {
                    let mut y = 0;
                    while y < dh {
                        let mut x = 0;
                        while x < dw {
                            rec.blit_bgra(dx + x, dy + y, img.width, img.height, &img.pixels);
                            x += iw.max(1);
                        }
                        y += ih.max(1);
                    }
                }
            }
        }
    }
}

fn gen_borders(b: &LayoutBox, rec: &mut Recorder<'_>, ox: i32, oy: i32, scale_alpha: &dyn Fn(u8) -> u8) {
    let r = b.border_rect();
    let to_gfx = |bc: &cv_layout::Color| Color { r: bc.r, g: bc.g, b: bc.b, a: scale_alpha(bc.a) };
    let bt = b.border_width_top() as i32;
    let br = b.border_width_right() as i32;
    let bb = b.border_width_bottom() as i32;
    let bl = b.border_width_left() as i32;
    let radius = super::used_border_radius_px(b, r.w as i32, r.h as i32);
    let uniform_width = bt > 0 && bt == br && bt == bb && bt == bl;
    let uniform_color = match (b.border_color_for(0), b.border_color_for(1), b.border_color_for(2), b.border_color_for(3)) {
        (Some(t), Some(rr), Some(btm), Some(l)) if t == rr && t == btm && t == l => Some(t),
        _ => None,
    };
    if radius > 0 && uniform_width {
        if let Some(c) = uniform_color.as_ref().map(to_gfx) {
            if c.a > 0 {
                rec.fill_rect_rounded_ring(r.x as i32 + ox, r.y as i32 + oy, r.w as i32, r.h as i32, radius, bt, c);
            }
        }
    } else if radius > 0 && (bt > 0 || br > 0 || bb > 0 || bl > 0) {
        let base_width = [br, bb, bl, bt].into_iter().find(|w| *w > 0).unwrap_or(0);
        let base_color = b
            .border_color_for(1)
            .or_else(|| b.border_color_for(2))
            .or_else(|| b.border_color_for(3))
            .or_else(|| b.border_color_for(0));
        if base_width > 0 {
            if let Some(c) = base_color.as_ref().map(to_gfx) {
                if c.a > 0 {
                    rec.fill_rect_rounded_ring(r.x as i32 + ox, r.y as i32 + oy, r.w as i32, r.h as i32, radius, base_width, c);
                }
            }
        }
        if bt > 0 {
            if let Some(c) = b.border_color_for(0).as_ref().map(to_gfx) {
                let top_differs = bt != base_width
                    || base_color.as_ref().map(|base| b.border_color_for(0).as_ref() != Some(base)).unwrap_or(true);
                if c.a > 0 && top_differs {
                    rec.fill_ellipse_ring_top(r.x as i32 + ox, r.y as i32 + oy, r.w as i32, r.h as i32, bt, c);
                }
            }
        }
    } else {
        let draw_h_dashes = |rec: &mut Recorder<'_>, sx: i32, sy: i32, total_w: i32, bw: i32, dash: i32, gap: i32, c: Color| {
            if dash <= 0 || gap <= 0 {
                return;
            }
            let step = dash + gap;
            let mut x = sx;
            while x < sx + total_w {
                let w = (sx + total_w - x).min(dash);
                if w > 0 {
                    rec.fill_rect(x, sy, w, bw, c);
                }
                x += step;
            }
        };
        let draw_v_dashes = |rec: &mut Recorder<'_>, sx: i32, sy: i32, total_h: i32, bw: i32, dash: i32, gap: i32, c: Color| {
            if dash <= 0 || gap <= 0 {
                return;
            }
            let step = dash + gap;
            let mut y = sy;
            while y < sy + total_h {
                let h = (sy + total_h - y).min(dash);
                if h > 0 {
                    rec.fill_rect(sx, y, bw, h, c);
                }
                y += step;
            }
        };
        if bt > 0 {
            if let Some(c) = b.border_color_for(0).as_ref().map(to_gfx) {
                if c.a > 0 {
                    match b.border_style_for(0) {
                        cv_layout::BorderStyle::None | cv_layout::BorderStyle::Hidden => {}
                        cv_layout::BorderStyle::Dashed => draw_h_dashes(rec, r.x as i32 + ox, r.y as i32 + oy, r.w as i32, bt, bt * 3, bt, c),
                        cv_layout::BorderStyle::Dotted => draw_h_dashes(rec, r.x as i32 + ox, r.y as i32 + oy, r.w as i32, bt, bt, bt, c),
                        cv_layout::BorderStyle::Double => {
                            let third = (bt / 3).max(1);
                            rec.fill_rect(r.x as i32 + ox, r.y as i32 + oy, r.w as i32, third, c);
                            rec.fill_rect(r.x as i32 + ox, r.y as i32 + oy + bt - third, r.w as i32, third, c);
                        }
                        _ => rec.fill_rect(r.x as i32 + ox, r.y as i32 + oy, r.w as i32, bt, c),
                    }
                }
            }
        }
        if bb > 0 {
            if let Some(c) = b.border_color_for(2).as_ref().map(to_gfx) {
                if c.a > 0 {
                    let sy = (r.y + r.h) as i32 + oy - bb;
                    match b.border_style_for(2) {
                        cv_layout::BorderStyle::None | cv_layout::BorderStyle::Hidden => {}
                        cv_layout::BorderStyle::Dashed => draw_h_dashes(rec, r.x as i32 + ox, sy, r.w as i32, bb, bb * 3, bb, c),
                        cv_layout::BorderStyle::Dotted => draw_h_dashes(rec, r.x as i32 + ox, sy, r.w as i32, bb, bb, bb, c),
                        cv_layout::BorderStyle::Double => {
                            let third = (bb / 3).max(1);
                            rec.fill_rect(r.x as i32 + ox, sy, r.w as i32, third, c);
                            rec.fill_rect(r.x as i32 + ox, sy + bb - third, r.w as i32, third, c);
                        }
                        _ => rec.fill_rect(r.x as i32 + ox, sy, r.w as i32, bb, c),
                    }
                }
            }
        }
        if bl > 0 {
            if let Some(c) = b.border_color_for(3).as_ref().map(to_gfx) {
                if c.a > 0 {
                    match b.border_style_for(3) {
                        cv_layout::BorderStyle::None | cv_layout::BorderStyle::Hidden => {}
                        cv_layout::BorderStyle::Dashed => draw_v_dashes(rec, r.x as i32 + ox, r.y as i32 + oy, r.h as i32, bl, bl * 3, bl, c),
                        cv_layout::BorderStyle::Dotted => draw_v_dashes(rec, r.x as i32 + ox, r.y as i32 + oy, r.h as i32, bl, bl, bl, c),
                        cv_layout::BorderStyle::Double => {
                            let third = (bl / 3).max(1);
                            rec.fill_rect(r.x as i32 + ox, r.y as i32 + oy, third, r.h as i32, c);
                            rec.fill_rect(r.x as i32 + ox + bl - third, r.y as i32 + oy, third, r.h as i32, c);
                        }
                        _ => rec.fill_rect(r.x as i32 + ox, r.y as i32 + oy, bl, r.h as i32, c),
                    }
                }
            }
        }
        if br > 0 {
            if let Some(c) = b.border_color_for(1).as_ref().map(to_gfx) {
                if c.a > 0 {
                    let sx = (r.x + r.w) as i32 + ox - br;
                    match b.border_style_for(1) {
                        cv_layout::BorderStyle::None | cv_layout::BorderStyle::Hidden => {}
                        cv_layout::BorderStyle::Dashed => draw_v_dashes(rec, sx, r.y as i32 + oy, r.h as i32, br, br * 3, br, c),
                        cv_layout::BorderStyle::Dotted => draw_v_dashes(rec, sx, r.y as i32 + oy, r.h as i32, br, br, br, c),
                        cv_layout::BorderStyle::Double => {
                            let third = (br / 3).max(1);
                            rec.fill_rect(sx, r.y as i32 + oy, third, r.h as i32, c);
                            rec.fill_rect(sx + br - third, r.y as i32 + oy, third, r.h as i32, c);
                        }
                        _ => rec.fill_rect(sx, r.y as i32 + oy, br, r.h as i32, c),
                    }
                }
            }
        }
    }
}

fn gen_embedded_image(b: &LayoutBox, rec: &mut Recorder<'_>, img: &cv_layout::EmbeddedImage, ox: i32, oy: i32) {
    let box_w = b.content.w.max(0.0);
    let box_h = b.content.h.max(0.0);
    let src_w = img.width as f32;
    let src_h = img.height as f32;
    let fit = b.object_fit.unwrap_or(cv_layout::ObjectFit::Fill);
    let (dx, dy, dw, dh) =
        super::compute_object_fit_dest(fit, b.content.x, b.content.y, box_w, box_h, src_w, src_h, b.object_position);
    let box_sx = b.content.x as i32 + ox;
    let box_sy = b.content.y as i32 + oy;
    let box_sw = box_w as i32;
    let box_sh = box_h as i32;
    let img_sx = dx as i32 + ox;
    let img_sy = dy as i32 + oy;
    let needs_clip = dw > box_w + 0.5 || dh > box_h + 0.5 || img_sx < box_sx || img_sy < box_sy;
    if (dw - img.width as f32).abs() < 0.5 && (dh - img.height as f32).abs() < 0.5 {
        if needs_clip {
            let off_x = img_sx - box_sx;
            let off_y = img_sy - box_sy;
            rec.blit_bgra_sprite(box_sx, box_sy, box_sw, box_sh, off_x, off_y, img.width, img.height, &img.pixels);
        } else {
            rec.blit_bgra(img_sx, img_sy, img.width, img.height, &img.pixels);
        }
    } else if dw > 0.5 && dh > 0.5 {
        let tw = dw.round().max(1.0) as u32;
        let th = dh.round().max(1.0) as u32;
        if needs_clip {
            let mut tmp = Bitmap { width: tw, height: th, pixels: vec![0u32; (tw as usize) * (th as usize)] };
            tmp.blit_bgra_scaled(0, 0, tw, th, img.width, img.height, &img.pixels);
            let off_x = img_sx - box_sx;
            let off_y = img_sy - box_sy;
            rec.blit_bgra_sprite(box_sx, box_sy, box_sw, box_sh, off_x, off_y, tmp.width, tmp.height, &tmp.pixels);
        } else {
            rec.blit_bgra_scaled(img_sx, img_sy, tw, th, img.width, img.height, &img.pixels);
        }
    }
    {
        let br = b.border_rect();
        let radius = super::used_border_radius_px(b, br.w as i32, br.h as i32);
        if radius > 0 {
            rec.clip_rounded_rect(br.x as i32 + ox, br.y as i32 + oy, br.w as i32, br.h as i32, radius);
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn gen_text(
    b: &LayoutBox,
    rec: &mut Recorder<'_>,
    texts: &mut Vec<TextItem>,
    t: &str,
    ox: i32,
    oy: i32,
    scale_alpha: &dyn Fn(u8) -> u8,
    list: &mut RetainedDisplayList,
    my_idx: u32,
) {
    let trimmed = if b.preserve_whitespace { t.to_string() } else { super::normalize_inline_text(t) };
    let trimmed = match b.text_transform {
        Some(cv_layout::TextTransform::Uppercase) => trimmed.to_uppercase(),
        Some(cv_layout::TextTransform::Lowercase) => trimmed.to_lowercase(),
        Some(cv_layout::TextTransform::Capitalize) => {
            let mut out = String::with_capacity(trimmed.len());
            let mut at_word_start = true;
            for ch in trimmed.chars() {
                if ch.is_alphabetic() {
                    if at_word_start {
                        out.extend(ch.to_uppercase());
                    } else {
                        out.push(ch);
                    }
                    at_word_start = false;
                } else {
                    out.push(ch);
                    at_word_start = true;
                }
            }
            out
        }
        _ => trimmed,
    };
    if trimmed.is_empty() {
        return;
    }
    let fs_px = b.font_size_px.max(8.0) as i32;
    let align = match b.text_align {
        Some(cv_layout::TextAlign::Center) => cv_ui::TextAlign::Center,
        Some(cv_layout::TextAlign::Right) => cv_ui::TextAlign::Right,
        _ => cv_ui::TextAlign::Left,
    };
    let trimmed = super::shape_for_render(&trimmed);
    let mut my_texts: Vec<TextItem> = Vec::new();
    if let Some(sh) = b.text_shadow {
        let ti = TextItem {
            x: (b.content.x + sh.offset_x_px) as i32 + ox,
            y: (b.content.y + sh.offset_y_px) as i32 + oy,
            w: b.content.w as i32,
            h: b.content.h as i32,
            font_size_px: fs_px,
            bold: b.font_weight_bold,
            font_weight: b.font_weight_num,
            italic: b.font_style_italic,
            font_family: b.font_family.clone(),
            color_rgb: (sh.color.r, sh.color.g, sh.color.b),
            color_alpha: scale_alpha(sh.color.a),
            text: trimmed.clone(),
            align,
            letter_spacing_px: b.letter_spacing_px.round() as i32,
            is_chrome: false,
        };
        texts.push(ti.clone());
        my_texts.push(ti);
    }
    let main_ti = TextItem {
        x: b.content.x as i32 + ox,
        y: b.content.y as i32 + oy,
        w: b.content.w as i32,
        h: b.content.h as i32,
        font_size_px: fs_px,
        bold: b.font_weight_bold,
        font_weight: b.font_weight_num,
        italic: b.font_style_italic,
        font_family: b.font_family.clone(),
        color_rgb: (b.text_color.r, b.text_color.g, b.text_color.b),
        color_alpha: scale_alpha(b.text_color.a),
        text: trimmed.clone(),
        align,
        letter_spacing_px: b.letter_spacing_px.round() as i32,
        is_chrome: false,
    };
    texts.push(main_ti.clone());
    my_texts.push(main_ti);
    list.chunks[my_idx as usize].text_items = my_texts;

    let underline = b.text_decoration_underline;
    if underline || b.text_decoration_line_through {
        let approx_w = (trimmed.chars().count() as f32 * b.font_size_px * 0.55).min(b.content.w) as i32;
        let deco_x = match b.text_align {
            Some(cv_layout::TextAlign::Right) => b.content.x as i32 + ox + (b.content.w as i32 - approx_w).max(0),
            Some(cv_layout::TextAlign::Center) => b.content.x as i32 + ox + ((b.content.w as i32 - approx_w) / 2).max(0),
            _ => b.content.x as i32 + ox,
        };
        let deco_color = Color { r: b.text_color.r, g: b.text_color.g, b: b.text_color.b, a: scale_alpha(b.text_color.a) };
        if underline {
            let underline_y = (b.content.y + b.font_size_px * 1.05) as i32 + oy;
            rec.fill_rect(deco_x, underline_y, approx_w, 1, deco_color);
        }
        if b.text_decoration_line_through {
            let strike_y = (b.content.y + b.font_size_px * 0.55) as i32 + oy;
            rec.fill_rect(deco_x, strike_y, approx_w, 1, deco_color);
        }
    }
}

fn gen_drop_shadow(b: &LayoutBox, rec: &mut Recorder<'_>, border_rect: &cv_layout::Rect, sh: cv_layout::BoxShadow, fw: i32, fh: i32) {
    let c = Color { r: sh.color.r, g: sh.color.g, b: sh.color.b, a: sh.color.a };
    let sx = (border_rect.x + sh.offset_x_px) as i32;
    let sy = (border_rect.y + sh.offset_y_px) as i32;
    let expand = sh.blur_px.round() as i32;
    let ex = sx - expand;
    let ey = sy - expand;
    let ew = fw + expand * 2;
    let eh = fh + expand * 2;
    let radius = super::used_border_radius_px(b, fw, fh);
    if radius > 0 {
        rec.fill_rect_rounded(ex, ey, ew, eh, radius + expand, c);
    } else {
        rec.fill_rect(ex, ey, ew, eh, c);
    }
    if sh.blur_px > 0.5 {
        let sigma = sh.blur_px.max(1.0);
        let pad = (sigma * 1.5).round() as i32;
        rec.apply_filter_rect(ex - pad, ey - pad, ew + pad * 2, eh + pad * 2, FilterOp::Blur(sigma));
        rec.apply_filter_rect(ex - pad, ey - pad, ew + pad * 2, eh + pad * 2, FilterOp::Blur(sigma));
    }
}

// ── content_hash — visual content + intrinsic size, EXCLUDING position ──────

/// Per-node content fingerprint. INCLUDES every visual property + intrinsic
/// size; EXCLUDES absolute position (content.x/y, border_rect x/y), translate,
/// scale/rotate/matrix transforms, scroll, z_index, node_id, and children
/// (covered by subtree_hash). A node that only MOVED hashes identically.
pub fn compute_content_hash(b: &LayoutBox) -> u64 {
    let mut h = Fnv::new();
    // BoxKind discriminant + text string + tag.
    match &b.kind {
        cv_layout::BoxKind::Block { tag } => {
            h.byte(0);
            h.str(tag);
        }
        cv_layout::BoxKind::Anonymous => h.byte(1),
        cv_layout::BoxKind::Text(t) => {
            h.byte(2);
            h.str(t);
        }
    }
    // background color
    h.opt_tag(b.background.is_some());
    if let Some(c) = b.background {
        h.lcolor(c);
    }
    // gradients
    h.opt_tag(b.background_gradient.is_some());
    if let Some(g) = b.background_gradient {
        h.lcolor(g.from);
        h.lcolor(g.to);
        h.f32(g.angle_deg);
    }
    h.opt_tag(b.background_radial_gradient.is_some());
    if let Some(g) = b.background_radial_gradient {
        h.lcolor(g.from);
        h.lcolor(g.to);
        h.f32(g.angle_deg);
    }
    // background image identity (Arc ptr + w/h) + repeat/size/position
    h.opt_tag(b.background_image.is_some());
    if let Some(img) = &b.background_image {
        h.u64(Arc::as_ptr(img) as *const () as usize as u64);
        h.u32(img.width);
        h.u32(img.height);
    }
    h.byte(b.background_repeat as u8);
    hash_bg_size(&mut h, &b.background_size);
    hash_bg_pos_opt(&mut h, &b.background_position);
    // box-shadow
    hash_box_shadow_opt(&mut h, &b.box_shadow);
    // borders (per-side widths/colors/styles + fallbacks + radius)
    for i in 0..4 {
        h.opt_tag(b.border_widths_per_side[i].is_some());
        if let Some(w) = b.border_widths_per_side[i] {
            h.f32(w);
        }
        h.opt_tag(b.border_colors_per_side[i].is_some());
        if let Some(c) = b.border_colors_per_side[i] {
            h.lcolor(c);
        }
        h.opt_tag(b.border_styles_per_side[i].is_some());
        if let Some(s) = b.border_styles_per_side[i] {
            h.byte(s as u8);
        }
    }
    h.f32(b.border_width_px);
    h.opt_tag(b.border_color.is_some());
    if let Some(c) = b.border_color {
        h.lcolor(c);
    }
    h.f32(b.border_radius_px);
    h.opt_tag(b.border_radius_percent.is_some());
    if let Some(p) = b.border_radius_percent {
        h.f32(p);
    }
    // embedded image identity + object-fit/position
    h.opt_tag(b.embedded_image.is_some());
    if let Some(img) = &b.embedded_image {
        h.u64(Arc::as_ptr(img) as *const () as usize as u64);
        h.u32(img.width);
        h.u32(img.height);
    }
    h.opt_tag(b.object_fit.is_some());
    if let Some(f) = b.object_fit {
        h.byte(f as u8);
    }
    hash_bg_pos_opt(&mut h, &b.object_position);
    // mask
    h.opt_tag(b.has_mask_url);
    h.opt_tag(b.mask_image.is_some());
    if let Some(img) = &b.mask_image {
        h.u64(Arc::as_ptr(img) as *const () as usize as u64);
        h.u32(img.width);
        h.u32(img.height);
    }
    // text styling
    h.lcolor(b.text_color);
    h.f32(b.font_size_px);
    h.opt_tag(b.font_weight_bold);
    h.opt_tag(b.font_style_italic);
    h.opt_tag(b.font_family.is_some());
    if let Some(ff) = &b.font_family {
        h.str(ff);
    }
    h.opt_tag(b.text_transform.is_some());
    if let Some(tt) = b.text_transform {
        h.byte(tt as u8);
    }
    h.f32(b.letter_spacing_px);
    h.opt_tag(b.text_decoration_underline);
    h.opt_tag(b.text_decoration_line_through);
    h.opt_tag(b.text_align.is_some());
    if let Some(ta) = b.text_align {
        h.byte(ta as u8);
    }
    h.opt_tag(b.preserve_whitespace);
    h.opt_tag(b.line_height_px.is_some());
    if let Some(lh) = b.line_height_px {
        h.f32(lh);
    }
    hash_text_shadow_opt(&mut h, &b.text_shadow);
    // visibility / opacity
    h.opt_tag(b.visibility_hidden);
    h.f32(b.opacity);
    // filters + backdrop-filters
    h.u32(b.filters.len() as u32);
    for f in &b.filters {
        hash_filter(&mut h, f);
    }
    h.u32(b.backdrop_filters.len() as u32);
    for f in &b.backdrop_filters {
        hash_filter(&mut h, f);
    }
    // clip-path
    h.opt_tag(b.clip_shape.is_some());
    if let Some(cs) = &b.clip_shape {
        match cs {
            cv_layout::ClipShape::Inset { top_px, right_px, bottom_px, left_px } => {
                h.byte(0);
                h.f32(*top_px);
                h.f32(*right_px);
                h.f32(*bottom_px);
                h.f32(*left_px);
            }
            cv_layout::ClipShape::Circle { radius_px, cx_px, cy_px } => {
                h.byte(1);
                h.f32(*radius_px);
                h.f32(*cx_px);
                h.f32(*cy_px);
            }
            cv_layout::ClipShape::Polygon(pts) => {
                h.byte(2);
                h.u32(pts.len() as u32);
                for (x, y) in pts {
                    h.f32(*x);
                    h.f32(*y);
                }
            }
        }
    }
    // INTRINSIC SIZE (w/h only — NOT x/y).
    h.f32(b.border_rect().w);
    h.f32(b.border_rect().h);
    h.f32(b.content.w);
    h.f32(b.content.h);
    h.f32(b.padding.top);
    h.f32(b.padding.right);
    h.f32(b.padding.bottom);
    h.f32(b.padding.left);
    // overflow_hidden (affects the clip emitted to children — visual).
    h.opt_tag(b.overflow_hidden);
    h.finish()
}

fn hash_bg_size(h: &mut Fnv, s: &Option<cv_layout::BgSize>) {
    h.opt_tag(s.is_some());
    if let Some(s) = s {
        match s {
            cv_layout::BgSize::Cover => h.byte(0),
            cv_layout::BgSize::Contain => h.byte(1),
            cv_layout::BgSize::Explicit(a, b2) => {
                h.byte(2);
                hash_bg_len_opt(h, a);
                hash_bg_len_opt(h, b2);
            }
        }
    }
}

fn hash_bg_len_opt(h: &mut Fnv, l: &Option<cv_layout::BgLength>) {
    h.opt_tag(l.is_some());
    if let Some(l) = l {
        match l {
            cv_layout::BgLength::Px(v) => {
                h.byte(0);
                h.f32(*v);
            }
            cv_layout::BgLength::Percent(v) => {
                h.byte(1);
                h.f32(*v);
            }
        }
    }
}

fn hash_bg_pos_opt(h: &mut Fnv, p: &Option<(cv_layout::BgPos, cv_layout::BgPos)>) {
    h.opt_tag(p.is_some());
    if let Some((x, y)) = p {
        hash_bg_pos(h, x);
        hash_bg_pos(h, y);
    }
}

fn hash_bg_pos(h: &mut Fnv, p: &cv_layout::BgPos) {
    match p {
        cv_layout::BgPos::Px(v) => {
            h.byte(0);
            h.f32(*v);
        }
        cv_layout::BgPos::Pct(v) => {
            h.byte(1);
            h.f32(*v);
        }
    }
}

fn hash_box_shadow_opt(h: &mut Fnv, s: &Option<cv_layout::BoxShadow>) {
    h.opt_tag(s.is_some());
    if let Some(sh) = s {
        h.f32(sh.offset_x_px);
        h.f32(sh.offset_y_px);
        h.f32(sh.blur_px);
        h.f32(sh.spread_px);
        h.lcolor(sh.color);
        h.opt_tag(sh.inset);
    }
}

fn hash_text_shadow_opt(h: &mut Fnv, s: &Option<cv_layout::BoxShadow>) {
    hash_box_shadow_opt(h, s);
}

fn hash_filter(h: &mut Fnv, f: &cv_layout::FilterEffect) {
    match f {
        cv_layout::FilterEffect::Blur(v) => {
            h.byte(0);
            h.f32(*v);
        }
        cv_layout::FilterEffect::Brightness(v) => {
            h.byte(1);
            h.f32(*v);
        }
        cv_layout::FilterEffect::Contrast(v) => {
            h.byte(2);
            h.f32(*v);
        }
        cv_layout::FilterEffect::Grayscale(v) => {
            h.byte(3);
            h.f32(*v);
        }
        cv_layout::FilterEffect::Invert(v) => {
            h.byte(4);
            h.f32(*v);
        }
        cv_layout::FilterEffect::Sepia(v) => {
            h.byte(5);
            h.f32(*v);
        }
        cv_layout::FilterEffect::Saturate(v) => {
            h.byte(6);
            h.f32(*v);
        }
        cv_layout::FilterEffect::HueRotate(v) => {
            h.byte(7);
            h.f32(*v);
        }
        cv_layout::FilterEffect::Opacity(v) => {
            h.byte(8);
            h.f32(*v);
        }
        cv_layout::FilterEffect::DropShadow(sh) => {
            h.byte(9);
            h.f32(sh.offset_x_px);
            h.f32(sh.offset_y_px);
            h.f32(sh.blur_px);
            h.f32(sh.spread_px);
            h.lcolor(sh.color);
            h.opt_tag(sh.inset);
        }
    }
}

// ── replay() — mirror recursion ─────────────────────────────────────────────

/// Replay the retained list into `bmp` (+ append text items to `texts`) in the
/// exact flat op order `paint_box_offset_t` produces, so after the shared
/// `bake_content_text_into_bitmap` the bitmap is BYTE-IDENTICAL to the live path.
pub fn replay(rdl: &RetainedDisplayList, bmp: &mut Bitmap, texts: &mut Vec<TextItem>) {
    if rdl.chunks.is_empty() {
        return;
    }
    replay_chunk(rdl, rdl.root, bmp, texts);
}

fn replay_chunk(rdl: &RetainedDisplayList, ci: u32, bmp: &mut Bitmap, texts: &mut Vec<TextItem>) {
    let c = &rdl.chunks[ci as usize];
    if c.visibility_hidden || c.opacity < 0.01 {
        return;
    }
    // Affine subtree: single blit_affine op carries the whole layer.
    if let Some(op) = &c.affine_op {
        op.replay_into(bmp);
        return;
    }
    // before-children ops + text items.
    for op in &c.ops_before_children {
        op.replay_into(bmp);
    }
    texts.extend(c.text_items.iter().cloned());
    // children in painted (z-bucket) order.
    for cidx in painted_child_order(rdl, ci) {
        replay_chunk(rdl, cidx, bmp, texts);
    }
    // after-children ops.
    for op in &c.ops_after_children {
        op.replay_into(bmp);
    }
}

// ── M5.4 damage-driven incremental raster ───────────────────────────────────
//
// The full-frame `bitmap` + the `RetainedDisplayList` that produced it are the
// per-frame pixel cache (both round-trip through `PaintData`). On a small change
// we re-raster ONLY the damaged region R of the cached bitmap, reusing every
// pixel outside R verbatim. The hard safety property (gated by the oracle) is
// BYTE-IDENTITY: an incremental frame equals a fresh full bake, max diff 0.
//
// The #1 trap is z-order overlap: the bitmap is a FLAT source-over composite, so
// a changed chunk shares pixels with whatever overlaps it. We do NOT re-raster
// "just the changed chunk's rect"; we CLEAR R to background then replay EVERY
// chunk whose subtree intersects R, in full document z-order, so under/over
// overlaps recomposite correctly. Region confinement is clear-then-replay-then-
// prune-by-subtree-extent — there is NO clip-rect draw-gating primitive on
// Bitmap (clip ops are destructive absolute-coord post-masks).

/// `CV_DAMAGE_RASTER` gate, read once (mirrors `retained_dl_enabled`). Default
/// **ON** as of M5.4 (flipped after replay-vs-live proved byte-identical to the
/// live `paint_box` painter across the whole adversarial combined-overlap corpus
/// + a clean 800-frame soak with `prod_max_pixdiff == 0` and zero would-serve-wrong
/// frames with verify OFF; see `soak_general_path_runtime_loop`).
///
/// ESCAPE HATCH: `CV_DAMAGE_RASTER=0` forces it OFF again, restoring the pure
/// live full-bake path (production frames byte-for-byte the live painter). Any
/// other value (or unset) keeps the new default ON. The `CV_DAMAGE_VERIFY`
/// self-heal remains available as belt-and-suspenders.
pub fn damage_raster_enabled() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("CV_DAMAGE_RASTER").as_deref() != Ok("0"))
}

/// `CV_DAMAGE_VERIFY=1` gate, read once. Default OFF. This is the RELEASE-safe
/// runtime self-check for the damage-raster path: when on (AND CV_DAMAGE_RASTER
/// is also on), every frame that WOULD serve an incremental bitmap also produces
/// a full bake of the SAME layout and compares them by pixel hash. On MATCH the
/// incremental frame is served (the win); on MISMATCH the FULL bake is served
/// (self-heal — the user never sees a wrong frame) and a loud `[DAMAGE_VERIFY]
/// MISMATCH …` line is logged with the diff stats so the soak/verify agent can
/// triage which mutation class broke.
///
/// DISTINCT from `CV_DAMAGE_ORACLE` (the debug dev-gate that PANICS on mismatch):
/// `CV_DAMAGE_VERIFY` must NEVER panic — it is a soak/diagnostic flag that runs
/// in RELEASE soak builds and self-heals. It DOUBLES per-frame work (incremental
/// + full + hash), so it is never the production default; production default-on
/// runs WITHOUT verify and relies on the unit oracle + a clean soak.
pub fn damage_verify_enabled() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("CV_DAMAGE_VERIFY").as_deref() == Ok("1"))
}

// Thresholds: above these, a clean full bake (one white-clear + one paint walk,
// no full-document clone) is cheaper + simpler than the incremental clone+clear+
// pruned-replay, so we full-bake. Correctness does NOT depend on these (full bake
// is always correct, incremental is always byte-identical when it runs) — they
// only trade work. Debug builds let CV_DAMAGE_* override for measurement.
const FULL_BAKE_AREA_FRAC: f32 = 0.5;
const FULL_BAKE_CHUNK_FRAC: f32 = 0.4;
const SMALL_TREE_MIN: usize = 8;

fn full_bake_area_frac() -> f32 {
    if cfg!(debug_assertions) {
        if let Ok(v) = std::env::var("CV_DAMAGE_AREA_FRAC") {
            if let Ok(f) = v.parse::<f32>() {
                return f;
            }
        }
    }
    FULL_BAKE_AREA_FRAC
}
fn full_bake_chunk_frac() -> f32 {
    if cfg!(debug_assertions) {
        if let Ok(v) = std::env::var("CV_DAMAGE_CHUNK_FRAC") {
            if let Ok(f) = v.parse::<f32>() {
                return f;
            }
        }
    }
    FULL_BAKE_CHUNK_FRAC
}

/// Integer screen rect (half-open) used for the damage region. Snapped OUTWARD
/// (floor min / ceil max) so it is a SUPERSET of every sub-pixel touched —
/// over-expansion is byte-safe (re-rastering an unchanged pixel reproduces it),
/// under-expansion is a silent wrong frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct IRect {
    x0: i32,
    y0: i32,
    x1: i32,
    y1: i32,
}

impl IRect {
    fn empty() -> Self {
        IRect { x0: 0, y0: 0, x1: 0, y1: 0 }
    }
    fn is_empty(&self) -> bool {
        self.x1 <= self.x0 || self.y1 <= self.y0
    }
    fn area(&self) -> i64 {
        if self.is_empty() {
            0
        } else {
            (self.x1 - self.x0) as i64 * (self.y1 - self.y0) as i64
        }
    }
    fn union(&self, o: &IRect) -> IRect {
        if self.is_empty() {
            return *o;
        }
        if o.is_empty() {
            return *self;
        }
        IRect {
            x0: self.x0.min(o.x0),
            y0: self.y0.min(o.y0),
            x1: self.x1.max(o.x1),
            y1: self.y1.max(o.y1),
        }
    }
    /// Clamp to the bitmap bounds (and below `chrome_min_y` so chrome rows are
    /// never disturbed — chrome pixels are inherited from the cached bitmap).
    fn clamp(&self, w: i32, h: i32, chrome_min_y: i32) -> IRect {
        IRect {
            x0: self.x0.clamp(0, w),
            y0: self.y0.clamp(chrome_min_y, h),
            x1: self.x1.clamp(0, w),
            y1: self.y1.clamp(chrome_min_y, h),
        }
    }
}

/// Snap an optional float extent OUTWARD to an integer rect (floor min / ceil max).
fn extent_to_irect(e: Option<(f32, f32, f32, f32)>) -> IRect {
    match e {
        Some((x0, y0, x1, y1)) if x1 > x0 && y1 > y0 => IRect {
            x0: x0.floor() as i32,
            y0: y0.floor() as i32,
            x1: x1.ceil() as i32,
            y1: y1.ceil() as i32,
        },
        _ => IRect::empty(),
    }
}

fn rect_to_irect(r: cv_layout::Rect) -> IRect {
    extent_to_irect(rect_to_extent(r))
}

fn irects_intersect(a: &IRect, b: &IRect) -> bool {
    !a.is_empty() && !b.is_empty() && a.x0 < b.x1 && a.x1 > b.x0 && a.y0 < b.y1 && a.y1 > b.y0
}

/// The painted extent of a whole subtree in the OLD or NEW list, by unioning the
/// `paint_extent` of `ci` + every descendant. Used for moved/removed/added ids
/// where one diff entry stands for the whole subtree (the diff emits one moved id
/// for a rigid-moved subtree, retained_dl `subtree_rigid_delta`). Equivalent to
/// `subtree_paint_extent` but recomputed defensively from the live walk so it is
/// correct even if the field were stale.
fn subtree_extent_irect(list: &RetainedDisplayList, ci: u32) -> IRect {
    let c = &list.chunks[ci as usize];
    // subtree_paint_extent is the bottom-up rollup; trust it (it is recomputed
    // every generate). Fall back to a manual walk only if it is empty but the
    // subtree has children that might carry extent (defensive; normally equal).
    let r = rect_to_irect(c.subtree_paint_extent);
    if !r.is_empty() || c.children.is_empty() {
        return r;
    }
    let mut acc = rect_to_irect(c.paint_extent);
    for &cidx in &c.children {
        acc = acc.union(&subtree_extent_irect(list, cidx));
    }
    acc
}

/// Compute the damage envelope R (single bounding rect) for a diff, by unioning
/// the OLD and NEW painted extents of every changed/moved/added/removed id. Uses
/// `paint_extent`/`subtree_paint_extent` (shadow/affine/text inclusive), NOT the
/// border `bounds`. Returns `None` if the diff references an id missing from the
/// list it should be in (structural surprise ⇒ caller full-bakes).
fn compute_damage_envelope(old: &RetainedDisplayList, new: &RetainedDisplayList, d: &PaintDiff) -> Option<IRect> {
    let mut r = IRect::empty();
    // changed: present in BOTH; union old + new own/subtree extent. A changed id's
    // content (e.g. resize, opacity) may grow its subtree, so union the subtree.
    for id in &d.changed {
        let oi = *old.index.get(id)?;
        let ni = *new.index.get(id)?;
        r = r.union(&subtree_extent_irect(old, oi));
        r = r.union(&subtree_extent_irect(new, ni));
    }
    // moved: union the subtree extent in BOTH old (erase moved-FROM) and new
    // (paint moved-TO). One moved id can stand for a whole rigid-moved subtree.
    for id in &d.moved {
        let oi = *old.index.get(id)?;
        let ni = *new.index.get(id)?;
        r = r.union(&subtree_extent_irect(old, oi));
        r = r.union(&subtree_extent_irect(new, ni));
    }
    // added: only in new — paint its subtree extent.
    for id in &d.added {
        let ni = *new.index.get(id)?;
        r = r.union(&subtree_extent_irect(new, ni));
    }
    // removed: only in old — erase its old subtree extent (repaint behind).
    for id in &d.removed {
        let oi = *old.index.get(id)?;
        r = r.union(&subtree_extent_irect(old, oi));
    }
    Some(r)
}

/// Replay the NEW list into `bmp`, restricted to the damage region R: skip any
/// chunk whose `subtree_paint_extent` misses R (it contributes nothing inside R);
/// otherwise issue this chunk's ops + recurse children in painted z-order +
/// after-ops — IDENTICALLY to `replay_chunk`, appending text items to `texts`.
/// Because every chunk whose subtree intersects R is replayed in full document
/// z-order, overlapping unchanged chunks (under AND over the changed one)
/// recomposite correctly inside R. This is driven into a FRESH white full-size
/// scratch by `incremental_composite`, which then copies ONLY R back into the
/// cache — so writes an op makes outside R are simply discarded (R stays tight,
/// no growing), and inside R the result equals a full bake (same chunks, same
/// z-order, same white start). A chunk whose subtree misses R contributes no
/// pixel inside R, so pruning it cannot change an R pixel.
fn replay_chunk_clipped(rdl: &RetainedDisplayList, ci: u32, bmp: &mut Bitmap, texts: &mut Vec<TextItem>, r: &IRect) {
    let c = &rdl.chunks[ci as usize];
    if c.visibility_hidden || c.opacity < 0.01 {
        return;
    }
    // Prune: subtree extent misses R ⇒ nothing this subtree paints lands in R.
    if !irects_intersect(&rect_to_irect(c.subtree_paint_extent), r) {
        return;
    }
    if let Some(op) = &c.affine_op {
        op.replay_into(bmp);
        return;
    }
    for op in &c.ops_before_children {
        op.replay_into(bmp);
    }
    texts.extend(c.text_items.iter().cloned());
    for cidx in painted_child_order(rdl, ci) {
        replay_chunk_clipped(rdl, cidx, bmp, texts, r);
    }
    for op in &c.ops_after_children {
        op.replay_into(bmp);
    }
}

/// The incremental composite. Returns `Some(bmp)` byte-identical to a fresh full
/// bake of `new`, or `None` when the caller must full-bake (threshold/structural
/// surprise). PRECONDITIONS the caller also checks; re-checked here for safety.
pub fn incremental_composite(
    cached_bitmap: &Bitmap,
    old: &RetainedDisplayList,
    new: &RetainedDisplayList,
    diff: &PaintDiff,
    chrome_h: u32,
) -> Option<Bitmap> {
    // 0. Preconditions: same dims, cache matches new dims. Else full-bake.
    if old.viewport_w != new.viewport_w || old.doc_h != new.doc_h {
        return None;
    }
    if cached_bitmap.width != new.viewport_w || cached_bitmap.height != new.doc_h {
        return None;
    }
    let w = new.viewport_w as i32;
    let h = new.doc_h as i32;
    if w <= 0 || h <= 0 {
        return None;
    }
    // 1. Empty diff ⇒ the new frame's pixels equal the cache verbatim. (The
    //    caller's Arc reuse is even cheaper, but returning a clone keeps the API
    //    total + lets the oracle exercise the no-op path.)
    if diff.is_empty() {
        return Some(cached_bitmap.clone());
    }
    // Tiny tree ⇒ not worth the machinery; full-bake.
    if new.chunks.len() < SMALL_TREE_MIN {
        return None;
    }
    // Root re-keyed ⇒ fresh document/tree; cache not comparable. Full-bake.
    if old.chunks.is_empty()
        || new.chunks.is_empty()
        || old.chunks[old.root as usize].node_id != new.chunks[new.root as usize].node_id
    {
        return None;
    }
    // Chunk-count threshold: damage spread across many chunks ⇒ full-bake.
    let touched = diff.changed.len() + diff.moved.len() + diff.added.len() + diff.removed.len();
    if touched as f32 >= full_bake_chunk_frac() * new.chunks.len() as f32 {
        return None;
    }
    // 2. Damage envelope R (shadow/affine/text-inclusive), clamped to the bitmap.
    //    `chrome_h` is a defensive lower-bound on R.y (the caller passes 0 because
    //    this bitmap is content-only — the URL bar is a separate pinned strip; if a
    //    future build baked chrome into the bitmap it would pass URL_BAR_HEIGHT to
    //    protect those rows). Any missing id ⇒ structural surprise ⇒ full-bake.
    let chrome_min_y = (chrome_h as i32).min(h).max(0);
    let env = compute_damage_envelope(old, new, diff)?;
    let r = env.clamp(w, h, chrome_min_y);
    if r.is_empty() {
        // Nothing intersects the paintable region (off-screen / below the floor).
        // The cache already holds the correct pixels, so reuse verbatim.
        return Some(cached_bitmap.clone());
    }
    // 3. Area threshold: near-total damage ⇒ a clean full bake is cheaper.
    let total = (w as i64) * (h as i64);
    if total > 0 && (r.area() as f64) >= full_bake_area_frac() as f64 * total as f64 {
        return None;
    }
    // 4. Re-render R FROM SCRATCH the same way a full bake does, but confined to R:
    //    replay the ENTIRE new list (pruned to chunks whose subtree intersects R)
    //    into a FRESH white full-size scratch, in document z-order. Inside R the
    //    scratch is byte-identical to a full bake — every chunk that contributes a
    //    pixel inside R is replayed in the same order over the same white start;
    //    chunks whose subtree misses R contribute NOTHING inside R, so pruning them
    //    cannot change an R pixel. Outside R the scratch is partial garbage, which
    //    we never read. This keeps R TIGHT (no grow): we only COPY R back, so a
    //    replayed op's writes outside R are discarded — overlapping/opaque chunks
    //    compose correctly because the whole R is rebuilt from white in z-order.
    let mut scratch = Bitmap::new(new.viewport_w, new.doc_h);
    scratch.clear(Color::WHITE);
    let mut texts: Vec<TextItem> = Vec::new();
    replay_chunk_clipped(new, new.root, &mut scratch, &mut texts, &r);
    // Bake text clipped to R (glyphs land only inside R, on the freshly repainted
    // R background — same sequence as a full bake: all ops, then text).
    cv_ui::bake_content_text_into_bitmap_clipped(
        &mut scratch,
        &mut texts,
        Some((r.x0, r.y0, r.x1 - r.x0, r.y1 - r.y0)),
    );
    // 5. Start from the cached pixels (everything OUTSIDE R reused verbatim) and
    //    copy ONLY R's pixels from the scratch — the from-white-rebuilt region.
    let mut bmp = cached_bitmap.clone();
    copy_region(&scratch, &mut bmp, &r);
    Some(bmp)
}

/// The clamped damage-region area (px²) that `incremental_composite` WOULD
/// re-raster for this diff, alongside the document area — for the
/// `CV_DAMAGE_VERIFY` log only (it never affects correctness). Returns
/// `(damage_area, doc_area)`; damage_area is 0 when the envelope is empty or a
/// referenced id is missing (structural surprise). chrome_h=0 matches the
/// content-only bitmap convention used at the call site.
pub fn damage_area(old: &RetainedDisplayList, new: &RetainedDisplayList, d: &PaintDiff) -> (i64, i64) {
    let w = new.viewport_w as i32;
    let h = new.doc_h as i32;
    let doc_area = (new.viewport_w as i64) * (new.doc_h as i64);
    if w <= 0 || h <= 0 {
        return (0, doc_area);
    }
    match compute_damage_envelope(old, new, d) {
        Some(env) => (env.clamp(w, h, 0).area(), doc_area),
        None => (0, doc_area),
    }
}

/// Copy the rectangle `r` from `src` into `dst` (both same dims), row by row.
fn copy_region(src: &Bitmap, dst: &mut Bitmap, r: &IRect) {
    if src.width != dst.width || src.height != dst.height {
        return;
    }
    let w = dst.width as i32;
    let h = dst.height as i32;
    let x0 = r.x0.clamp(0, w);
    let y0 = r.y0.clamp(0, h);
    let x1 = r.x1.clamp(0, w);
    let y1 = r.y1.clamp(0, h);
    if x1 <= x0 || y1 <= y0 {
        return;
    }
    let stride = dst.width as usize;
    for yy in y0..y1 {
        let row = (yy as usize) * stride;
        let a = row + x0 as usize;
        let b = row + x1 as usize;
        dst.pixels[a..b].copy_from_slice(&src.pixels[a..b]);
    }
}

// ── PaintDiff ───────────────────────────────────────────────────────────────

#[derive(Clone, Debug, Default, PartialEq)]
pub struct PaintDiff {
    pub changed: Vec<u64>,
    pub moved: Vec<u64>,
    pub added: Vec<u64>,
    pub removed: Vec<u64>,
}

impl PaintDiff {
    pub fn is_empty(&self) -> bool {
        self.changed.is_empty() && self.moved.is_empty() && self.added.is_empty() && self.removed.is_empty()
    }
}

fn rects_equal(a: &cv_layout::Rect, b: &cv_layout::Rect) -> bool {
    a.x == b.x && a.y == b.y && a.w == b.w && a.h == b.h
}
fn origin_equal(a: &cv_layout::Rect, b: &cv_layout::Rect) -> bool {
    a.x == b.x && a.y == b.y
}

/// Classify every node_id as changed / moved / added / removed by comparing two
/// retained lists. Uses subtree_hash to prune unchanged subtrees.
pub fn diff(old: &RetainedDisplayList, new: &RetainedDisplayList) -> PaintDiff {
    let mut d = PaintDiff::default();
    // FAST PRUNE: whole tree content-identical AND geometrically identical.
    // Both hashes must match — subtree_hash alone is position-independent, so a
    // moved descendant keeps it equal; geom_hash catches the move.
    if !old.chunks.is_empty() && !new.chunks.is_empty() {
        let o_root = &old.chunks[old.root as usize];
        let n_root = &new.chunks[new.root as usize];
        if o_root.subtree_hash == n_root.subtree_hash && o_root.geom_hash == n_root.geom_hash {
            return d;
        }
    }
    // Walk new tree from root, pruning unchanged subtrees.
    if !new.chunks.is_empty() {
        diff_walk(old, new, new.root, &mut d);
    }
    // Removed: ids in old but not new.
    for (id, _) in &old.index {
        if !new.index.contains_key(id) {
            d.removed.push(*id);
        }
    }
    d
}

fn diff_walk(old: &RetainedDisplayList, new: &RetainedDisplayList, nci: u32, d: &mut PaintDiff) {
    let n = &new.chunks[nci as usize];
    match old.index.get(&n.node_id) {
        None => {
            // Added — entire subtree is new. Mark this node; descend to mark
            // each descendant as added too (all need raster).
            mark_subtree_added(new, nci, d);
        }
        Some(&oidx) => {
            let o = &old.chunks[oidx as usize];
            // Content-identical subtree (subtree_hash == covers content + structure).
            if o.subtree_hash == n.subtree_hash {
                if o.geom_hash == n.geom_hash {
                    // fully unchanged subtree — prune, emit nothing.
                    return;
                }
                // Content identical but geometry shifted somewhere inside.
                // If the WHOLE subtree translated rigidly (every node by the same
                // delta), relocate it as a unit (the design's cached-raster move).
                if let Some((dx, dy)) = subtree_rigid_delta(old, oidx, new, nci) {
                    if dx != 0.0 || dy != 0.0 {
                        d.moved.push(n.node_id);
                    }
                    return;
                }
                // Non-rigid: descend and mark each node whose own origin moved.
                // (content_hash is equal everywhere in this subtree.)
                if !origin_equal(&o.bounds, &n.bounds) {
                    d.moved.push(n.node_id);
                }
                for &cidx in &n.children {
                    diff_walk(old, new, cidx, d);
                }
                return;
            }
            // Subtree content differs somewhere. Classify THIS node.
            if o.content_hash != n.content_hash {
                d.changed.push(n.node_id);
            } else if !origin_equal(&o.bounds, &n.bounds) {
                d.moved.push(n.node_id);
            }
            // Descend into children (document order).
            for &cidx in &n.children {
                diff_walk(old, new, cidx, d);
            }
        }
    }
}

fn mark_subtree_added(new: &RetainedDisplayList, nci: u32, d: &mut PaintDiff) {
    let n = &new.chunks[nci as usize];
    d.added.push(n.node_id);
    for &cidx in &n.children {
        mark_subtree_added(new, cidx, d);
    }
}

/// When two content-identical subtrees (equal subtree_hash) differ geometrically,
/// return `Some((dx,dy))` iff EVERY corresponding node shifted by the same delta
/// — a rigid translation that 5.4 can relocate as a cached unit. `None` if the
/// shift is non-uniform (different nodes moved by different amounts) so the
/// caller descends to classify each moved node individually.
fn subtree_rigid_delta(
    old: &RetainedDisplayList,
    oci: u32,
    new: &RetainedDisplayList,
    nci: u32,
) -> Option<(f32, f32)> {
    let mut delta: Option<(f32, f32)> = None;
    if !rigid_walk(old, oci, new, nci, &mut delta) {
        return None;
    }
    Some(delta.unwrap_or((0.0, 0.0)))
}

fn rigid_walk(
    old: &RetainedDisplayList,
    oci: u32,
    new: &RetainedDisplayList,
    nci: u32,
    delta: &mut Option<(f32, f32)>,
) -> bool {
    let o = &old.chunks[oci as usize];
    let n = &new.chunks[nci as usize];
    // subtree_hash matched at entry => structure + content identical; child
    // counts and order agree, so a positional zip is sound.
    if o.children.len() != n.children.len() {
        return false;
    }
    let dx = n.bounds.x - o.bounds.x;
    let dy = n.bounds.y - o.bounds.y;
    match *delta {
        None => *delta = Some((dx, dy)),
        Some((edx, edy)) => {
            if dx != edx || dy != edy {
                return false;
            }
        }
    }
    for (oc, nc) in o.children.iter().zip(n.children.iter()) {
        if !rigid_walk(old, *oc, new, *nc, delta) {
            return false;
        }
    }
    true
}

// ── flag + seam helpers ─────────────────────────────────────────────────────

/// `CV_RETAINED_DL=1` gate, read once (mirrors `compositor_enabled`). Default
/// OFF — when off this module is never reached and frames are the live path.
pub fn retained_dl_enabled() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("CV_RETAINED_DL").as_deref() == Ok("1"))
}

thread_local! {
    /// Previous frame's retained list for the renderer thread, so the seam can
    /// compute a PaintDiff vs the prior bake. Single-threaded per off-main design.
    static RETAINED_PREV: std::cell::RefCell<Option<RetainedDisplayList>> =
        const { std::cell::RefCell::new(None) };
}

/// The seam body (called from `bake_layout_into_paint` when the flag is on):
/// generate the retained list, optionally oracle-check it, diff vs the previous
/// frame, and stash it. 5.2 does NOT use the result to drive rasterization.
/// Returns the diff vs the previous frame (for inspection / 5.4), if any.
pub fn record_frame(lb: &LayoutBox, cfg: &cv_layout::LayoutConfig) -> Option<PaintDiff> {
    let rdl = generate(lb, cfg);
    if cfg!(debug_assertions) || std::env::var("CV_RETAINED_ORACLE").is_ok() {
        debug_oracle_check(lb, cfg);
    }
    RETAINED_PREV.with(|p| {
        let diff = p.borrow().as_ref().map(|old| diff(old, &rdl));
        *p.borrow_mut() = Some(rdl);
        diff
    })
}

/// Re-raster the retained replay into a scratch bitmap and assert it is
/// byte-identical to a fresh live `paint_box` bake. Panics on mismatch (debug),
/// which is the loud signal the design demands. Returns the per-pixel max diff
/// (always 0 on success) for callers that want to log it.
pub fn debug_oracle_check(lb: &LayoutBox, cfg: &cv_layout::LayoutConfig) -> u64 {
    let (live, live_texts) = super::oracle_live_paint(lb, cfg);
    let (replayed, replay_texts) = oracle_replay_paint(lb, cfg);
    assert_eq!(
        live.pixels.len(),
        replayed.pixels.len(),
        "retained_dl oracle: bitmap size mismatch"
    );
    let mut maxd: u64 = 0;
    for (a, b) in live.pixels.iter().zip(replayed.pixels.iter()) {
        if a != b {
            maxd = maxd.max(pixel_abs_diff(*a, *b));
        }
    }
    assert_eq!(maxd, 0, "retained_dl oracle: replay pixels differ from live paint");
    assert!(
        live_texts == replay_texts,
        "retained_dl oracle: text items differ pre-bake"
    );
    maxd
}

/// Replay path used by the oracle: generate → fresh white bitmap → replay ops →
/// bake the SAME text vec. Mirrors the live `bake_layout_into_paint` content seq.
pub fn oracle_replay_paint(lb: &LayoutBox, cfg: &cv_layout::LayoutConfig) -> (Bitmap, Vec<TextItem>) {
    let rdl = generate(lb, cfg);
    let mut bmp = Bitmap::new(rdl.viewport_w, rdl.doc_h);
    bmp.clear(Color::WHITE);
    let mut texts: Vec<TextItem> = Vec::new();
    replay(&rdl, &mut bmp, &mut texts);
    // Snapshot the content text items BEFORE baking, because
    // `bake_content_text_into_bitmap` drains non-chrome items from the vec. The
    // oracle compares these against the live path's pre-bake content texts.
    let texts_pre = texts.clone();
    cv_ui::bake_content_text_into_bitmap(&mut bmp, &mut texts);
    (bmp, texts_pre)
}

fn pixel_abs_diff(a: u32, b: u32) -> u64 {
    let d = |sh: u32| -> u64 {
        let av = (a >> sh) & 0xFF;
        let bv = (b >> sh) & 0xFF;
        av.abs_diff(bv) as u64
    };
    d(0).max(d(8)).max(d(16)).max(d(24))
}

// ── Oracle + diff tests ─────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use cv_layout::{
        AlignItems, BackgroundRepeat, BgPos, BgSize, BorderStyle, BoxKind, BoxShadow, ClipShape,
        Color as LColor, EdgeSizes, FilterEffect, FlexDirection, FlexWrap, JustifyContent,
        LayoutBox, LayoutConfig, Position, Rect, TextAlign, TextTransform, VerticalAlign,
    };

    fn cfg() -> LayoutConfig {
        LayoutConfig {
            viewport_w: 200.0,
            viewport_h: 150.0,
            default_font_size_px: 16.0,
            default_text_color: LColor { r: 0, g: 0, b: 0, a: 255 },
            default_line_height: 1.2,
            measure_text_fn: None,
        }
    }

    /// Fully-initialised baseline box; tests mutate the fields they care about.
    fn base_box() -> LayoutBox {
        LayoutBox {
            content: Rect { x: 10.0, y: 10.0, w: 80.0, h: 40.0 },
            padding: EdgeSizes::default(),
            margin: EdgeSizes::default(),
            margin_auto: Default::default(),
            border_width_px: 0.0,
            border_color: None,
            border_widths_per_side: [None, None, None, None],
            border_colors_per_side: [None, None, None, None],
            border_styles_per_side: [None; 4],
            text_align: None,
            font_weight_bold: false,
            font_weight_num: 0,
            font_style_italic: false,
            font_family: None,
            text_transform: None,
            letter_spacing_px: 0.0,
            text_decoration_underline: false,
            text_decoration_line_through: false,
            line_height_px: None,
            preserve_whitespace: false,
            box_sizing_border_box: false,
            is_flex: false,
            is_grid: false,
            is_inline: false,
            is_inline_block: false,
            is_table: false,
            is_table_row: false,
            is_table_cell: false,
            is_table_row_group: false,
            is_list_item: false,
            is_flow_root: false,
            flex_direction: FlexDirection::Row,
            flex_wrap: FlexWrap::NoWrap,
            flex_grow: 0.0,
            flex_shrink: 1.0,
            flex_basis: None,
            justify_content: JustifyContent::Start,
            align_items: AlignItems::Stretch,
            justify_items: None,
            align_self: None,
            justify_self: None,
            gap_px: 0.0,
            column_gap_px: 0.0,
            row_gap_px: 0.0,
            grid_template_columns: None,
            grid_template_rows: None,
            grid_auto_rows: None,
            grid_auto_columns: None,
            grid_template_areas: None,
            grid_column_start: None,
            grid_column_span: None,
            grid_row_start: None,
            grid_row_span: None,
            table_col_span: None,
            table_row_span: None,
            grid_area_name: None,
            is_multicol: false,
            multicol_count: None,
            multicol_width: None,
            multicol_gap: 0.0,
            column_rule_width: 0.0,
            column_rule_style: cv_layout::BorderStyle::None,
            column_rule_color: None,
            column_span_all: false,
            multicol_used_count: 0,
            multicol_used_width: 0.0,
            overflow_hidden: false,
            overflow_x: cv_layout::Overflow::Visible,
            overflow_y: cv_layout::Overflow::Visible,
            scroll_offset_x: 0.0,
            scroll_offset_y: 0.0,
            visibility_hidden: false,
            opacity: 1.0,
            border_radius_px: 0.0,
            border_radius_percent: None,
            translate_x_px: 0.0,
            translate_y_px: 0.0,
            translate_x_percent: None,
            translate_y_percent: None,
            scale_x: 1.0,
            scale_y: 1.0,
            rotate_deg: 0.0,
            matrix_2d: None,
            transform_origin: None,
            transform_mat4: None,
            perspective_px: None,
            perspective_origin: None,
            transform_style_preserve_3d: false,
            backface_visibility_hidden: false,
            position: Position::Static,
            z_index: None,
            float_side: cv_layout::FloatSide::None,
            clear: cv_layout::ClearMode::None,
            vertical_align: VerticalAlign::Baseline,
            box_shadow: None,
            text_shadow: None,
            filters: Vec::new(),
            backdrop_filters: Vec::new(),
            mix_blend_mode: cv_layout::BlendMode::Normal,
            background_blend_mode: cv_layout::BlendMode::Normal,
            animation_name: None,
            animation_duration_ms: 0.0,
            animation_delay_ms: 0.0,
            animation_iteration_count: 0.0,
            animation_timing: 0,
            clip_shape: None,
            has_mask_url: false,
            mask_image_url: None,
            background_gradient: None,
            background_radial_gradient: None,
            background_gradient_full: None,
            background_image_url: None,
            background_repeat: BackgroundRepeat::Repeat,
            top_px: None,
            right_px: None,
            bottom_px: None,
            left_px: None,
            explicit_width: None,
            flex_override_width: None,
            explicit_height: None,
            flex_override_height: None,
            aspect_ratio: None,
            max_width: None,
            max_height: None,
            min_width: None,
            min_height: None,
            background: None,
            text_color: LColor { r: 0, g: 0, b: 0, a: 255 },
            font_size_px: 16.0,
            kind: BoxKind::Block { tag: "div".into() },
            link_href: None,
            embedded_image: None,
            mask_image: None,
            background_image: None,
            background_size: None,
            background_position: None,
            object_fit: None,
            object_position: None,
            element_path: None,
            node_id: None,
            cache_ineligible: false,
            writing_mode: cv_layout::WritingMode::default(),
            children: Vec::new(),
        }
    }

    fn nid(n: u64) -> Option<u64> {
        // Real packed node_id shape: (index << 32) | generation, generation >= 1.
        Some((n << 32) | 1)
    }

    fn img(w: u32, h: u32, fill: u32) -> std::sync::Arc<cv_layout::EmbeddedImage> {
        std::sync::Arc::new(cv_layout::EmbeddedImage {
            width: w,
            height: h,
            pixels: vec![fill; (w as usize) * (h as usize)],
        })
    }

    /// THE GATE — byte-identity oracle: live paint == retained generate+replay.
    fn assert_oracle(lb: &LayoutBox) {
        let c = cfg();
        let (live, live_texts) = super::super::oracle_live_paint(lb, &c);
        let (replayed, replay_texts) = oracle_replay_paint(lb, &c);
        assert_eq!(live.width, replayed.width, "width mismatch");
        assert_eq!(live.height, replayed.height, "height mismatch");
        let mut maxd: u64 = 0;
        let mut first: Option<usize> = None;
        for (i, (a, b)) in live.pixels.iter().zip(replayed.pixels.iter()).enumerate() {
            if a != b {
                if first.is_none() {
                    first = Some(i);
                }
                maxd = maxd.max(pixel_abs_diff(*a, *b));
            }
        }
        assert_eq!(maxd, 0, "pixel diff at index {:?} (maxd={})", first, maxd);
        assert_eq!(live.pixels, replayed.pixels, "Vec<u32> not exactly equal");
        assert_eq!(live_texts, replay_texts, "text items differ");
    }

    #[test]
    fn t1_single_block_bg_color() {
        let mut b = base_box();
        b.node_id = nid(1);
        b.background = Some(LColor { r: 20, g: 120, b: 200, a: 255 });
        assert_oracle(&b);
    }

    #[test]
    fn t2_four_disjoint_borders() {
        let mut b = base_box();
        b.node_id = nid(1);
        b.background = Some(LColor { r: 255, g: 255, b: 255, a: 255 });
        b.border_widths_per_side = [Some(2.0), Some(4.0), Some(6.0), Some(3.0)];
        b.border_colors_per_side = [
            Some(LColor { r: 255, g: 0, b: 0, a: 255 }),
            Some(LColor { r: 0, g: 255, b: 0, a: 255 }),
            Some(LColor { r: 0, g: 0, b: 255, a: 255 }),
            Some(LColor { r: 200, g: 200, b: 0, a: 255 }),
        ];
        assert_oracle(&b);
    }

    #[test]
    fn t3_border_radius_uniform_ring() {
        let mut b = base_box();
        b.node_id = nid(1);
        b.background = Some(LColor { r: 50, g: 50, b: 50, a: 255 });
        b.border_radius_px = 8.0;
        b.border_width_px = 3.0;
        b.border_widths_per_side = [Some(3.0); 4];
        b.border_color = Some(LColor { r: 240, g: 120, b: 0, a: 255 });
        b.border_colors_per_side = [Some(LColor { r: 240, g: 120, b: 0, a: 255 }); 4];
        assert_oracle(&b);
    }

    #[test]
    fn t4_box_shadow_outset_and_inset() {
        let mut b = base_box();
        b.node_id = nid(1);
        b.background = Some(LColor { r: 200, g: 200, b: 200, a: 255 });
        b.box_shadow = Some(BoxShadow {
            offset_x_px: 4.0,
            offset_y_px: 4.0,
            blur_px: 0.0,
            spread_px: 0.0,
            color: LColor { r: 0, g: 0, b: 0, a: 128 },
            inset: false,
        });
        assert_oracle(&b);
        // blurred outset ring path
        b.box_shadow = Some(BoxShadow {
            offset_x_px: 0.0,
            offset_y_px: 0.0,
            blur_px: 10.0,
            spread_px: 2.0,
            color: LColor { r: 255, g: 200, b: 0, a: 80 },
            inset: false,
        });
        assert_oracle(&b);
        // inset path
        b.box_shadow = Some(BoxShadow {
            offset_x_px: 2.0,
            offset_y_px: 2.0,
            blur_px: 0.0,
            spread_px: 1.0,
            color: LColor { r: 0, g: 0, b: 0, a: 100 },
            inset: true,
        });
        assert_oracle(&b);
    }

    #[test]
    fn t5_nested_opacity() {
        // parent 0.5, child 0.3, non-overlapping rects — verifies per-op alpha.
        let mut parent = base_box();
        parent.node_id = nid(1);
        parent.content = Rect { x: 0.0, y: 0.0, w: 180.0, h: 120.0 };
        parent.opacity = 0.5;
        parent.background = Some(LColor { r: 100, g: 0, b: 0, a: 255 });
        let mut child = base_box();
        child.node_id = nid(2);
        child.content = Rect { x: 20.0, y: 60.0, w: 60.0, h: 30.0 };
        child.opacity = 0.3;
        child.background = Some(LColor { r: 0, g: 0, b: 200, a: 255 });
        parent.children = vec![child];
        assert_oracle(&parent);
    }

    #[test]
    fn t6_nested_translate_accumulation() {
        let mut parent = base_box();
        parent.node_id = nid(1);
        parent.content = Rect { x: 0.0, y: 0.0, w: 180.0, h: 120.0 };
        parent.translate_x_px = 10.0;
        parent.translate_y_px = 5.0;
        parent.background = Some(LColor { r: 0, g: 80, b: 0, a: 255 });
        let mut child = base_box();
        child.node_id = nid(2);
        child.content = Rect { x: 10.0, y: 10.0, w: 40.0, h: 40.0 };
        child.translate_x_px = 7.0;
        child.translate_y_px = 3.0;
        child.background = Some(LColor { r: 200, g: 200, b: 0, a: 255 });
        parent.children = vec![child];
        assert_oracle(&parent);
    }

    #[test]
    fn t7_overflow_hidden_clip_straddle() {
        let mut parent = base_box();
        parent.node_id = nid(1);
        parent.content = Rect { x: 20.0, y: 20.0, w: 60.0, h: 40.0 };
        parent.overflow_hidden = true;
        parent.background = Some(LColor { r: 180, g: 180, b: 180, a: 255 });
        let mut child = base_box();
        child.node_id = nid(2);
        // straddles the parent's right/bottom clip edge
        child.content = Rect { x: 50.0, y: 40.0, w: 80.0, h: 60.0 };
        child.background = Some(LColor { r: 200, g: 0, b: 0, a: 255 });
        parent.children = vec![child];
        assert_oracle(&parent);
    }

    #[test]
    fn t8_clip_path_inset_circle_polygon() {
        for shape in [
            ClipShape::Inset { top_px: 5.0, right_px: 8.0, bottom_px: 5.0, left_px: 8.0 },
            ClipShape::Circle { radius_px: 40.0, cx_px: 50.0, cy_px: 50.0 },
            ClipShape::Polygon(vec![(0.0, 0.0), (100.0, 0.0), (50.0, 100.0)]),
        ] {
            let mut b = base_box();
            b.node_id = nid(1);
            b.content = Rect { x: 10.0, y: 10.0, w: 100.0, h: 100.0 };
            b.background = Some(LColor { r: 60, g: 120, b: 240, a: 255 });
            b.clip_shape = Some(shape);
            assert_oracle(&b);
        }
    }

    #[test]
    fn t9_z_index_buckets_with_flex_parent() {
        // children z = -1, auto, 0(positioned), 2 + non-positioned, flex parent.
        let mut parent = base_box();
        parent.node_id = nid(1);
        parent.content = Rect { x: 0.0, y: 0.0, w: 180.0, h: 120.0 };
        parent.is_flex = true;
        parent.background = Some(LColor { r: 240, g: 240, b: 240, a: 255 });

        let mk = |id: u64, x: f32, z: Option<i32>, pos: Position, col: (u8, u8, u8)| {
            let mut c = base_box();
            c.node_id = nid(id);
            c.content = Rect { x, y: 20.0, w: 50.0, h: 50.0 };
            c.z_index = z;
            c.position = pos;
            c.background = Some(LColor { r: col.0, g: col.1, b: col.2, a: 200 });
            c
        };
        parent.children = vec![
            mk(10, 10.0, Some(-1), Position::Absolute, (255, 0, 0)),
            mk(11, 30.0, None, Position::Static, (0, 255, 0)),
            mk(12, 50.0, Some(0), Position::Absolute, (0, 0, 255)),
            mk(13, 70.0, Some(2), Position::Absolute, (255, 255, 0)),
            mk(14, 20.0, None, Position::Static, (255, 0, 255)),
        ];
        assert_oracle(&parent);
    }

    #[test]
    fn t10_embedded_image_object_fit() {
        for fit in [cv_layout::ObjectFit::Cover, cv_layout::ObjectFit::Contain, cv_layout::ObjectFit::Fill] {
            let mut b = base_box();
            b.node_id = nid(1);
            b.content = Rect { x: 10.0, y: 10.0, w: 60.0, h: 40.0 };
            b.embedded_image = Some(img(20, 30, 0xFF1188CC));
            b.object_fit = Some(fit);
            assert_oracle(&b);
        }
    }

    #[test]
    fn t11_background_image_repeat_and_sprite() {
        // repeat
        let mut b = base_box();
        b.node_id = nid(1);
        b.content = Rect { x: 5.0, y: 5.0, w: 60.0, h: 50.0 };
        b.background_image = Some(img(16, 16, 0xFF33AA55));
        b.background_repeat = BackgroundRepeat::Repeat;
        assert_oracle(&b);
        // no-repeat sprite via background-position
        b.background_repeat = BackgroundRepeat::NoRepeat;
        b.background_position = Some((BgPos::Px(-4.0), BgPos::Px(-8.0)));
        assert_oracle(&b);
        // cover
        b.background_position = None;
        b.background_size = Some(BgSize::Cover);
        assert_oracle(&b);
    }

    #[test]
    fn t12_gradient_and_radial() {
        let mut b = base_box();
        b.node_id = nid(1);
        b.content = Rect { x: 5.0, y: 5.0, w: 100.0, h: 60.0 };
        b.background_gradient = Some(cv_layout::LinearGradientSpec {
            from: LColor { r: 255, g: 0, b: 0, a: 255 },
            to: LColor { r: 0, g: 0, b: 255, a: 255 },
            angle_deg: 45.0,
        });
        assert_oracle(&b);
        b.background_gradient = None;
        b.background_radial_gradient = Some(cv_layout::LinearGradientSpec {
            from: LColor { r: 255, g: 255, b: 255, a: 255 },
            to: LColor { r: 0, g: 0, b: 0, a: 255 },
            angle_deg: 0.0,
        });
        assert_oracle(&b);
    }

    #[test]
    fn t13_text_decoration_and_items() {
        let mut b = base_box();
        b.node_id = nid(1);
        b.content = Rect { x: 10.0, y: 10.0, w: 120.0, h: 20.0 };
        b.kind = BoxKind::Text("Total Blocks".into());
        b.text_color = LColor { r: 30, g: 30, b: 30, a: 255 };
        b.text_decoration_underline = true;
        b.text_decoration_line_through = true;
        b.text_transform = Some(TextTransform::Uppercase);
        b.letter_spacing_px = 1.0;
        b.text_align = Some(TextAlign::Center);
        b.text_shadow = Some(BoxShadow {
            offset_x_px: 1.0,
            offset_y_px: 1.0,
            blur_px: 0.0,
            spread_px: 0.0,
            color: LColor { r: 150, g: 150, b: 150, a: 255 },
            inset: false,
        });
        assert_oracle(&b);
    }

    #[test]
    fn t14_affine_rotate_and_matrix() {
        let mut b = base_box();
        b.node_id = nid(1);
        b.content = Rect { x: 30.0, y: 30.0, w: 60.0, h: 40.0 };
        b.background = Some(LColor { r: 220, g: 60, b: 60, a: 255 });
        b.rotate_deg = 30.0;
        assert_oracle(&b);
        // matrix() form
        b.rotate_deg = 0.0;
        b.matrix_2d = Some([0.9, 0.3, -0.3, 0.9, 5.0, 2.0]);
        assert_oracle(&b);
        // affine with a child (subtree layer)
        b.matrix_2d = None;
        b.rotate_deg = 20.0;
        let mut child = base_box();
        child.node_id = nid(2);
        child.content = Rect { x: 40.0, y: 40.0, w: 20.0, h: 20.0 };
        child.background = Some(LColor { r: 0, g: 0, b: 0, a: 255 });
        b.children = vec![child];
        assert_oracle(&b);
    }

    #[test]
    fn t15_visibility_hidden_skip() {
        let mut parent = base_box();
        parent.node_id = nid(1);
        parent.content = Rect { x: 0.0, y: 0.0, w: 120.0, h: 80.0 };
        parent.background = Some(LColor { r: 200, g: 200, b: 200, a: 255 });
        let mut hidden = base_box();
        hidden.node_id = nid(2);
        hidden.visibility_hidden = true;
        hidden.content = Rect { x: 10.0, y: 10.0, w: 40.0, h: 40.0 };
        hidden.background = Some(LColor { r: 255, g: 0, b: 0, a: 255 });
        let mut visible = base_box();
        visible.node_id = nid(3);
        visible.content = Rect { x: 60.0, y: 10.0, w: 40.0, h: 40.0 };
        visible.background = Some(LColor { r: 0, g: 0, b: 255, a: 255 });
        parent.children = vec![hidden, visible];
        assert_oracle(&parent);
    }

    #[test]
    fn t16_backdrop_filter_and_filter_chain() {
        let mut parent = base_box();
        parent.node_id = nid(1);
        parent.content = Rect { x: 0.0, y: 0.0, w: 150.0, h: 100.0 };
        parent.background = Some(LColor { r: 120, g: 200, b: 80, a: 255 });
        let mut overlay = base_box();
        overlay.node_id = nid(2);
        overlay.content = Rect { x: 20.0, y: 20.0, w: 80.0, h: 50.0 };
        overlay.backdrop_filters = vec![FilterEffect::Blur(2.0), FilterEffect::Brightness(1.2)];
        overlay.filters = vec![FilterEffect::Grayscale(1.0), FilterEffect::Contrast(1.3)];
        parent.children = vec![overlay];
        assert_oracle(&parent);
    }

    #[test]
    fn t17_deep_representative_tree() {
        let mut root = base_box();
        root.node_id = nid(1);
        root.content = Rect { x: 0.0, y: 0.0, w: 190.0, h: 140.0 };
        root.background = Some(LColor { r: 250, g: 250, b: 252, a: 255 });
        root.overflow_hidden = true;

        let mut card = base_box();
        card.node_id = nid(2);
        card.content = Rect { x: 10.0, y: 10.0, w: 120.0, h: 90.0 };
        card.background = Some(LColor { r: 255, g: 255, b: 255, a: 255 });
        card.border_radius_px = 6.0;
        card.border_width_px = 1.0;
        card.border_widths_per_side = [Some(1.0); 4];
        card.border_color = Some(LColor { r: 200, g: 200, b: 200, a: 255 });
        card.border_colors_per_side = [Some(LColor { r: 200, g: 200, b: 200, a: 255 }); 4];
        card.box_shadow = Some(BoxShadow {
            offset_x_px: 0.0,
            offset_y_px: 2.0,
            blur_px: 6.0,
            spread_px: 0.0,
            color: LColor { r: 0, g: 0, b: 0, a: 40 },
            inset: false,
        });

        let mut title = base_box();
        title.node_id = nid(3);
        title.content = Rect { x: 18.0, y: 18.0, w: 100.0, h: 18.0 };
        title.kind = BoxKind::Text("Dashboard".into());
        title.font_weight_bold = true;
        title.text_color = LColor { r: 20, g: 20, b: 20, a: 255 };

        let mut pill = base_box();
        pill.node_id = nid(4);
        pill.content = Rect { x: 18.0, y: 50.0, w: 60.0, h: 24.0 };
        pill.position = Position::Absolute;
        pill.z_index = Some(2);
        pill.background_gradient = Some(cv_layout::LinearGradientSpec {
            from: LColor { r: 80, g: 120, b: 255, a: 255 },
            to: LColor { r: 120, g: 80, b: 255, a: 255 },
            angle_deg: 90.0,
        });
        pill.border_radius_px = 12.0;

        let mut icon = base_box();
        icon.node_id = nid(5);
        icon.content = Rect { x: 90.0, y: 50.0, w: 24.0, h: 24.0 };
        icon.embedded_image = Some(img(24, 24, 0xFFAA3377));
        icon.opacity = 0.8;

        card.children = vec![title, pill, icon];
        root.children = vec![card];
        assert_oracle(&root);
    }

    #[test]
    fn fuzz_random_trees_byte_identical() {
        // Deterministic PRNG (xorshift) — no external crate.
        let mut state: u64 = 0x1234_5678_9abc_def0;
        let mut rng = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for _ in 0..40 {
            let mut node_counter: u64 = 1;
            let root = rand_tree(&mut rng, &mut node_counter, 0);
            assert_oracle(&root);
        }
    }

    fn rand_tree(rng: &mut impl FnMut() -> u64, counter: &mut u64, depth: u32) -> LayoutBox {
        let mut b = base_box();
        b.node_id = nid(*counter);
        *counter += 1;
        let r = rng();
        b.content = Rect {
            x: (r % 40) as f32,
            y: ((r >> 8) % 40) as f32,
            w: 20.0 + ((r >> 16) % 60) as f32,
            h: 15.0 + ((r >> 24) % 40) as f32,
        };
        if r & 1 != 0 {
            b.background = Some(LColor {
                r: (r >> 1) as u8,
                g: (r >> 9) as u8,
                b: (r >> 17) as u8,
                a: 255,
            });
        }
        if r & 2 != 0 {
            b.border_width_px = 1.0 + (r % 3) as f32;
            b.border_widths_per_side = [Some(b.border_width_px); 4];
            b.border_color = Some(LColor { r: 0, g: 0, b: 0, a: 255 });
            b.border_colors_per_side = [Some(LColor { r: 0, g: 0, b: 0, a: 255 }); 4];
        }
        if r & 4 != 0 {
            b.opacity = 0.5;
        }
        if r & 8 != 0 {
            b.translate_x_px = ((r >> 4) % 10) as f32;
            b.translate_y_px = ((r >> 12) % 10) as f32;
        }
        if r & 16 != 0 {
            b.overflow_hidden = true;
        }
        if depth < 2 {
            let n = (rng() % 3) as usize;
            for _ in 0..n {
                b.children.push(rand_tree(rng, counter, depth + 1));
            }
        }
        b
    }

    // ── PaintDiff minimality tests ──────────────────────────────────────────

    fn small_tree() -> LayoutBox {
        let mut root = base_box();
        root.node_id = nid(1);
        root.content = Rect { x: 0.0, y: 0.0, w: 180.0, h: 120.0 };
        root.background = Some(LColor { r: 240, g: 240, b: 240, a: 255 });
        let mut a = base_box();
        a.node_id = nid(2);
        a.content = Rect { x: 10.0, y: 10.0, w: 60.0, h: 30.0 };
        a.kind = BoxKind::Text("Alpha".into());
        let mut c = base_box();
        c.node_id = nid(3);
        c.content = Rect { x: 10.0, y: 60.0, w: 60.0, h: 30.0 };
        c.background = Some(LColor { r: 0, g: 120, b: 0, a: 255 });
        root.children = vec![a, c];
        root
    }

    #[test]
    fn diff_no_change_is_empty_and_fast_prunes() {
        let c = cfg();
        let t = small_tree();
        let r1 = generate(&t, &c);
        let r2 = generate(&t, &c);
        let d = diff(&r1, &r2);
        assert!(d.is_empty(), "identical bakes -> empty diff, got {:?}", d);
    }

    #[test]
    fn diff_single_text_change_touches_only_that_node() {
        let c = cfg();
        let t1 = small_tree();
        let mut t2 = small_tree();
        // change ONLY the text node string.
        t2.children[0].kind = BoxKind::Text("Beta".into());
        let r1 = generate(&t1, &c);
        let r2 = generate(&t2, &c);
        let d = diff(&r1, &r2);
        assert_eq!(d.changed, vec![nid(2).unwrap()], "only text node changed: {:?}", d);
        assert!(d.moved.is_empty() && d.added.is_empty() && d.removed.is_empty(), "{:?}", d);
    }

    #[test]
    fn diff_pure_move_is_moved_only() {
        let c = cfg();
        let t1 = small_tree();
        let mut t2 = small_tree();
        // move the green box (same content, new position).
        t2.children[1].content.x += 25.0;
        t2.children[1].content.y += 5.0;
        let r1 = generate(&t1, &c);
        let r2 = generate(&t2, &c);
        let d = diff(&r1, &r2);
        assert_eq!(d.moved, vec![nid(3).unwrap()], "green box moved-only: {:?}", d);
        assert!(d.changed.is_empty(), "no content change: {:?}", d);
        assert!(d.added.is_empty() && d.removed.is_empty(), "{:?}", d);
    }

    #[test]
    fn diff_add_and_remove_detected() {
        let c = cfg();
        let t1 = small_tree();
        let mut t2 = small_tree();
        // remove the green box, add a new node.
        t2.children.pop();
        let mut extra = base_box();
        extra.node_id = nid(9);
        extra.content = Rect { x: 100.0, y: 10.0, w: 30.0, h: 30.0 };
        extra.background = Some(LColor { r: 10, g: 10, b: 200, a: 255 });
        t2.children.push(extra);
        let r1 = generate(&t1, &c);
        let r2 = generate(&t2, &c);
        let d = diff(&r1, &r2);
        assert!(d.added.contains(&nid(9).unwrap()), "added node 9: {:?}", d);
        assert!(d.removed.contains(&nid(3).unwrap()), "removed node 3: {:?}", d);
    }

    #[test]
    fn content_hash_excludes_position() {
        // Two boxes identical except x/y/translate -> same content_hash.
        let mut a = base_box();
        a.node_id = nid(1);
        a.background = Some(LColor { r: 12, g: 34, b: 56, a: 255 });
        let mut b = a.clone();
        b.content.x += 100.0;
        b.content.y += 50.0;
        b.translate_x_px = 9.0;
        b.translate_y_px = 4.0;
        b.scale_x = 2.0;
        b.rotate_deg = 45.0;
        b.z_index = Some(5);
        assert_eq!(compute_content_hash(&a), compute_content_hash(&b), "position/transform must not affect content_hash");
        // Changing the size DOES change it.
        let mut c = a.clone();
        c.content.w += 5.0;
        assert_ne!(compute_content_hash(&a), compute_content_hash(&c), "size IS content");
    }

    #[test]
    fn moved_node_subtree_hash_unchanged() {
        let c = cfg();
        let t1 = small_tree();
        let mut t2 = small_tree();
        t2.children[1].content.x += 25.0;
        let r1 = generate(&t1, &c);
        let r2 = generate(&t2, &c);
        let i1 = r1.index[&nid(3).unwrap()];
        let i2 = r2.index[&nid(3).unwrap()];
        assert_eq!(
            r1.chunks[i1 as usize].subtree_hash,
            r2.chunks[i2 as usize].subtree_hash,
            "moved-only node keeps subtree_hash"
        );
        assert_ne!(
            r1.chunks[i1 as usize].bounds.x,
            r2.chunks[i2 as usize].bounds.x,
            "but its bounds moved"
        );
    }

    // ── M5.4 incremental==full byte-identity oracle ─────────────────────────
    //
    // THE 5.4 GATE: given an old cache (a full bake of old_lb) and a new_lb, the
    // damage-tracked incremental_composite must be BYTE-IDENTICAL to a fresh full
    // bake of new_lb, max diff 0 — or it must FALL BACK (return None) for a
    // documented class. Never a silent wrong frame. Covers every diff class +
    // z-order overlap + shadow overflow + a 200-pair PRNG fuzz.

    /// A wide tree with enough chunks (>= SMALL_TREE_MIN) to exercise the real
    /// incremental path (small trees full-bake by the SMALL_TREE_MIN floor).
    /// Root + a row of `n` colored cells + a couple of overlapping positioned
    /// boxes, so z-order, shadows and text all appear.
    fn grid_tree(n: u64) -> LayoutBox {
        let mut root = base_box();
        root.node_id = nid(1);
        root.content = Rect { x: 0.0, y: 0.0, w: 190.0, h: 140.0 };
        root.background = Some(LColor { r: 245, g: 245, b: 248, a: 255 });
        for i in 0..n {
            let mut cell = base_box();
            cell.node_id = nid(100 + i);
            let col = (i % 5) as f32;
            let rowy = (i / 5) as f32;
            cell.content = Rect { x: 6.0 + col * 36.0, y: 6.0 + rowy * 30.0, w: 30.0, h: 24.0 };
            cell.background = Some(LColor {
                r: (40 + i * 17) as u8,
                g: (90 + i * 7) as u8,
                b: (160 + i * 11) as u8,
                a: 255,
            });
            if i % 3 == 0 {
                cell.kind = BoxKind::Text(format!("c{i}"));
                cell.background = None;
                cell.text_color = LColor { r: 20, g: 20, b: 30, a: 255 };
            }
            root.children.push(cell);
        }
        root
    }

    /// Build the cache exactly as production does for old_lb: a full bake (==
    /// oracle_replay_paint, itself proven == live paint by M5.2).
    fn cache_for(lb: &LayoutBox, c: &LayoutConfig) -> Bitmap {
        let (bmp, _) = oracle_replay_paint(lb, c);
        bmp
    }

    /// THE 5.4 GATE. Returns true if the incremental path RAN (byte-identical),
    /// false if it FELL BACK to full bake (None). Asserts byte-identity when it
    /// ran. chrome_h=0: oracle bitmaps have no URL-bar rows.
    fn assert_incremental_oracle(old_lb: &LayoutBox, new_lb: &LayoutBox) -> bool {
        let c = cfg();
        let cached = cache_for(old_lb, &c);
        let old_rdl = generate(old_lb, &c);
        let new_rdl = generate(new_lb, &c);
        let d = diff(&old_rdl, &new_rdl);
        let (full, _) = oracle_replay_paint(new_lb, &c);
        match incremental_composite(&cached, &old_rdl, &new_rdl, &d, 0) {
            None => false, // documented full-bake fallback fired.
            Some(incr) => {
                assert_eq!(incr.width, full.width, "incr width != full");
                assert_eq!(incr.height, full.height, "incr height != full");
                let mut first: Option<usize> = None;
                let mut maxd: u64 = 0;
                for (i, (a, b)) in incr.pixels.iter().zip(full.pixels.iter()).enumerate() {
                    if a != b {
                        if first.is_none() {
                            first = Some(i);
                        }
                        maxd = maxd.max(pixel_abs_diff(*a, *b));
                    }
                }
                assert_eq!(
                    maxd, 0,
                    "INCREMENTAL != FULL: first diff at pixel {:?} maxd={} diff={:?}",
                    first, maxd, d
                );
                assert_eq!(incr.pixels, full.pixels, "incremental Vec<u32> != full bake");
                true
            }
        }
    }

    #[test]
    fn incr_no_change_reuses_verbatim() {
        let t = grid_tree(12);
        // identical old/new ⇒ empty diff ⇒ verbatim reuse, byte-identical.
        assert!(assert_incremental_oracle(&t, &t));
    }

    #[test]
    fn incr_single_text_change() {
        let t1 = grid_tree(12);
        let mut t2 = grid_tree(12);
        // node 100 (i=0) is a text cell ("c0"); change its string.
        if let Some(cell) = t2.children.iter_mut().find(|c| c.node_id == nid(100)) {
            cell.kind = BoxKind::Text("XX".into());
        }
        assert!(assert_incremental_oracle(&t1, &t2));
    }

    #[test]
    fn incr_single_color_change() {
        let t1 = grid_tree(12);
        let mut t2 = grid_tree(12);
        if let Some(cell) = t2.children.iter_mut().find(|c| c.node_id == nid(101)) {
            cell.background = Some(LColor { r: 250, g: 10, b: 10, a: 255 });
        }
        assert!(assert_incremental_oracle(&t1, &t2));
    }

    #[test]
    fn incr_moved_subtree_repaints_from_region() {
        let t1 = grid_tree(12);
        let mut t2 = grid_tree(12);
        // rigid-move one cell to a new spot; the moved-FROM region must be
        // repainted to background + whatever was behind.
        if let Some(cell) = t2.children.iter_mut().find(|c| c.node_id == nid(104)) {
            cell.content.x += 20.0;
            cell.content.y += 18.0;
        }
        assert!(assert_incremental_oracle(&t1, &t2));
    }

    #[test]
    fn incr_added_node() {
        let t1 = grid_tree(12);
        let mut t2 = grid_tree(12);
        let mut extra = base_box();
        extra.node_id = nid(900);
        extra.content = Rect { x: 70.0, y: 90.0, w: 40.0, h: 30.0 };
        extra.background = Some(LColor { r: 10, g: 200, b: 90, a: 255 });
        t2.children.push(extra);
        assert!(assert_incremental_oracle(&t1, &t2));
    }

    #[test]
    fn incr_removed_node_erases_old_pixels() {
        let t1 = grid_tree(12);
        let mut t2 = grid_tree(12);
        // remove a cell — its OLD pixels must be erased + what was behind repainted.
        t2.children.retain(|c| c.node_id != nid(106));
        assert!(assert_incremental_oracle(&t1, &t2));
    }

    #[test]
    fn incr_resized_node() {
        let t1 = grid_tree(12);
        let mut t2 = grid_tree(12);
        if let Some(cell) = t2.children.iter_mut().find(|c| c.node_id == nid(102)) {
            cell.content.w += 18.0;
            cell.content.h += 12.0;
        }
        assert!(assert_incremental_oracle(&t1, &t2));
    }

    #[test]
    fn incr_opacity_change() {
        let t1 = grid_tree(12);
        let mut t2 = grid_tree(12);
        if let Some(cell) = t2.children.iter_mut().find(|c| c.node_id == nid(103)) {
            cell.opacity = 0.4;
        }
        assert!(assert_incremental_oracle(&t1, &t2));
    }

    #[test]
    fn incr_overflow_clip_toggle() {
        let t1 = grid_tree(12);
        let mut t2 = grid_tree(12);
        // toggle overflow_hidden on the root (content_hash includes it).
        t2.overflow_hidden = true;
        // also give root a child that straddles so the clip is visible.
        assert_incremental_oracle(&t1, &t2); // may fall back (large damage) — either way no panic.
    }

    #[test]
    fn incr_affine_rotate_change() {
        let mut t1 = grid_tree(12);
        let mut t2 = grid_tree(12);
        // give one cell a rotate that changes 0 -> 20 deg (affine layer chunk).
        for t in [&mut t1, &mut t2] {
            if let Some(cell) = t.children.iter_mut().find(|c| c.node_id == nid(105)) {
                cell.content = Rect { x: 80.0, y: 80.0, w: 36.0, h: 28.0 };
                cell.background = Some(LColor { r: 220, g: 80, b: 60, a: 255 });
            }
        }
        if let Some(cell) = t2.children.iter_mut().find(|c| c.node_id == nid(105)) {
            cell.rotate_deg = 20.0;
        }
        assert!(assert_incremental_oracle(&t1, &t2));
    }

    #[test]
    fn incr_zorder_overlap_under() {
        // A low-z (document-order-early) cell changes color while an unchanged,
        // higher-z positioned sibling OVERLAPS it. Repainting only the changed
        // rect would erase the overlapping sibling; the oracle proves the sibling
        // is replayed (subtree intersects R) so the overlap composites correctly.
        let mk = |changed_color: LColor| {
            let mut root = grid_tree(10);
            // low-z cell with a known position.
            let mut low = base_box();
            low.node_id = nid(500);
            low.content = Rect { x: 40.0, y: 40.0, w: 60.0, h: 50.0 };
            low.background = Some(changed_color);
            // high-z positioned sibling that overlaps `low`.
            let mut high = base_box();
            high.node_id = nid(501);
            high.position = Position::Absolute;
            high.z_index = Some(5);
            high.content = Rect { x: 70.0, y: 60.0, w: 50.0, h: 40.0 };
            high.background = Some(LColor { r: 0, g: 0, b: 0, a: 160 });
            root.children.push(low);
            root.children.push(high);
            root
        };
        let t1 = mk(LColor { r: 200, g: 60, b: 60, a: 255 });
        let t2 = mk(LColor { r: 60, g: 200, b: 60, a: 255 });
        assert!(assert_incremental_oracle(&t1, &t2));
    }

    #[test]
    fn incr_zorder_overlap_over() {
        // A high-z (positioned) cell changes while it sits OVER unchanged lower
        // content. R is cleared then BOTH the lower content (intersecting) and the
        // changed top cell replay in z-order; proves background-behind is restored.
        let mk = |changed_color: LColor| {
            let mut root = grid_tree(10);
            let mut low = base_box();
            low.node_id = nid(500);
            low.content = Rect { x: 40.0, y: 40.0, w: 70.0, h: 55.0 };
            low.background = Some(LColor { r: 180, g: 180, b: 30, a: 255 });
            let mut high = base_box();
            high.node_id = nid(501);
            high.position = Position::Absolute;
            high.z_index = Some(5);
            high.content = Rect { x: 60.0, y: 55.0, w: 50.0, h: 40.0 };
            high.background = Some(changed_color);
            root.children.push(low);
            root.children.push(high);
            root
        };
        let t1 = mk(LColor { r: 30, g: 30, b: 200, a: 200 });
        let t2 = mk(LColor { r: 200, g: 30, b: 30, a: 200 });
        assert!(assert_incremental_oracle(&t1, &t2));
    }

    #[test]
    fn incr_shadow_overflow_uses_op_extent_not_bounds() {
        // A node whose box-shadow paints OUTSIDE its border rect changes; proves
        // damage uses op paint_extent (shadow-inclusive), else the shadow tail is
        // left stale. shadow offset+spread pushes pixels well past the border.
        let mk = |sx: f32| {
            let mut root = grid_tree(10);
            let mut node = base_box();
            node.node_id = nid(600);
            node.content = Rect { x: 60.0, y: 50.0, w: 50.0, h: 40.0 };
            node.background = Some(LColor { r: 230, g: 230, b: 230, a: 255 });
            node.box_shadow = Some(BoxShadow {
                offset_x_px: sx,
                offset_y_px: 8.0,
                blur_px: 0.0,
                spread_px: 4.0,
                color: LColor { r: 0, g: 0, b: 0, a: 120 },
                inset: false,
            });
            root.children.push(node);
            root
        };
        let t1 = mk(10.0);
        let t2 = mk(18.0); // shadow moves; its old tail must be erased.
        assert!(assert_incremental_oracle(&t1, &t2));
    }

    #[test]
    fn incr_two_disjoint_changes() {
        let t1 = grid_tree(15);
        let mut t2 = grid_tree(15);
        // two cells in opposite corners change.
        if let Some(cell) = t2.children.iter_mut().find(|c| c.node_id == nid(101)) {
            cell.background = Some(LColor { r: 255, g: 0, b: 0, a: 255 });
        }
        if let Some(cell) = t2.children.iter_mut().find(|c| c.node_id == nid(114)) {
            cell.background = Some(LColor { r: 0, g: 0, b: 255, a: 255 });
        }
        // single-envelope covers both; byte-identical (or full-bakes if envelope
        // crosses the area threshold — either way no panic).
        assert_incremental_oracle(&t1, &t2);
    }

    #[test]
    fn incr_doc_resize_falls_back() {
        // Changing total document height resizes the bitmap ⇒ must full-bake.
        let t1 = grid_tree(12);
        let mut t2 = grid_tree(12);
        t2.content.h += 200.0; // grows doc_h
        let ran = assert_incremental_oracle(&t1, &t2);
        assert!(!ran, "doc resize must fall back to full bake (return None)");
    }

    #[test]
    fn incr_tiny_tree_falls_back() {
        // < SMALL_TREE_MIN chunks ⇒ full-bake floor.
        let t1 = small_tree();
        let mut t2 = small_tree();
        t2.children[0].kind = BoxKind::Text("Beta".into());
        let ran = assert_incremental_oracle(&t1, &t2);
        assert!(!ran, "tiny tree must fall back to full bake");
    }

    #[test]
    fn incr_top_row_content_change_not_missed() {
        // REGRESSION: the content bitmap starts at row 0 (the URL bar is a separate
        // pinned strip, NOT baked into PaintData.bitmap), so chrome_h must be 0 and
        // a change in the TOP rows must be in the damage region — else it is
        // silently missed. A cell at y=0..24 changing color must be byte-identical.
        let mut t1 = grid_tree(12);
        let mut t2 = grid_tree(12);
        // node 100 (i=0) lives at y=6..30 (top band). Pin it to y=0 in both.
        for t in [&mut t1, &mut t2] {
            if let Some(cell) = t.children.iter_mut().find(|c| c.node_id == nid(100)) {
                cell.content = Rect { x: 0.0, y: 0.0, w: 40.0, h: 20.0 };
                cell.kind = BoxKind::Block { tag: "div".into() };
                cell.background = Some(LColor { r: 10, g: 10, b: 10, a: 255 });
            }
        }
        if let Some(cell) = t2.children.iter_mut().find(|c| c.node_id == nid(100)) {
            cell.background = Some(LColor { r: 240, g: 0, b: 0, a: 255 });
        }
        // assert_incremental_oracle uses chrome_h=0; this would FAIL (or fall back)
        // if the top rows were clamped out of R.
        assert!(assert_incremental_oracle(&t1, &t2));
    }

    #[test]
    fn incr_damage_region_is_minimal_single_node() {
        // MEASURED WIN: a single-node color change re-rasters only its region, not
        // the whole document. Assert the damage envelope (the only re-rastered
        // pixels) is a tiny fraction of the full document, and that the result is
        // still byte-identical to a full bake.
        let c = cfg();
        let t1 = grid_tree(12);
        let mut t2 = grid_tree(12);
        if let Some(cell) = t2.children.iter_mut().find(|c| c.node_id == nid(101)) {
            cell.background = Some(LColor { r: 250, g: 10, b: 10, a: 255 });
        }
        let old_rdl = generate(&t1, &c);
        let new_rdl = generate(&t2, &c);
        let d = diff(&old_rdl, &new_rdl);
        let env = compute_damage_envelope(&old_rdl, &new_rdl, &d).unwrap();
        let r = env.clamp(new_rdl.viewport_w as i32, new_rdl.doc_h as i32, 0);
        let doc_area = (new_rdl.viewport_w as i64) * (new_rdl.doc_h as i64);
        let frac = r.area() as f64 / doc_area as f64;
        // The changed cell is 30x24=720 px in a 200x150=30000 px document => ~2.4%.
        assert!(
            frac < 0.10,
            "single-node damage should be << document: r.area()={} doc_area={} frac={:.4}",
            r.area(),
            doc_area,
            frac
        );
        // And still byte-identical.
        assert!(assert_incremental_oracle(&t1, &t2));
        eprintln!(
            "[M5.4 MEASURED] single-node change re-rasters {} px of {} px ({:.2}% of document)",
            r.area(),
            doc_area,
            frac * 100.0
        );
    }

    #[test]
    fn incr_fuzz_byte_identical_or_documented_fallback() {
        // 200 (rand tree, random mutation) pairs. Each asserts incremental == full
        // byte-identical (tolerance 0) OR the documented full-bake fallback fired —
        // never a silent diff. Deterministic xorshift (no external crate).
        let mut state: u64 = 0x0bad_f00d_dead_beef;
        let mut rng = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let mut ran_count = 0usize;
        let mut fellback_count = 0usize;
        for _ in 0..200 {
            let mut node_counter: u64 = 1;
            let old = rand_tree_wide(&mut rng, &mut node_counter);
            let new = mutate_tree(&old, &mut rng, &mut node_counter);
            if assert_incremental_oracle(&old, &new) {
                ran_count += 1;
            } else {
                fellback_count += 1;
            }
        }
        // Sanity: the incremental path must actually RUN on a healthy fraction of
        // cases (else the fuzz proves nothing). grid-ish wide trees with single
        // mutations overwhelmingly stay incremental.
        assert!(
            ran_count > 20,
            "expected many incremental runs, got ran={} fellback={}",
            ran_count,
            fellback_count
        );
    }

    /// A wide tree (root + 8..16 leaf cells), so it clears SMALL_TREE_MIN and the
    /// incremental path runs. Stable node_ids for survivors.
    fn rand_tree_wide(rng: &mut impl FnMut() -> u64, counter: &mut u64) -> LayoutBox {
        let mut root = base_box();
        root.node_id = nid(*counter);
        *counter += 1;
        root.content = Rect { x: 0.0, y: 0.0, w: 190.0, h: 150.0 };
        root.background = Some(LColor { r: 250, g: 250, b: 252, a: 255 });
        let r0 = rng();
        let n = 8 + (r0 % 9) as usize; // 8..16 cells
        for i in 0..n {
            let r = rng();
            let mut cell = base_box();
            cell.node_id = nid(*counter);
            *counter += 1;
            let col = (i % 5) as f32;
            let rowy = (i / 5) as f32;
            cell.content = Rect {
                x: 4.0 + col * 36.0 + (r % 4) as f32,
                y: 4.0 + rowy * 30.0 + ((r >> 4) % 4) as f32,
                w: 24.0 + ((r >> 8) % 12) as f32,
                h: 18.0 + ((r >> 12) % 10) as f32,
            };
            if r & 1 != 0 {
                cell.kind = BoxKind::Text(format!("t{i}"));
                cell.text_color = LColor { r: 20, g: 20, b: 30, a: 255 };
            } else {
                cell.background = Some(LColor {
                    r: (r >> 1) as u8,
                    g: (r >> 9) as u8,
                    b: (r >> 17) as u8,
                    a: 255,
                });
            }
            if r & 2 != 0 {
                cell.box_shadow = Some(BoxShadow {
                    offset_x_px: 3.0,
                    offset_y_px: 3.0,
                    blur_px: 0.0,
                    spread_px: 2.0,
                    color: LColor { r: 0, g: 0, b: 0, a: 90 },
                    inset: false,
                });
            }
            if r & 4 != 0 {
                cell.opacity = 0.6;
            }
            if r & 8 != 0 && i % 4 == 0 {
                cell.position = Position::Absolute;
                cell.z_index = Some((r % 5) as i32 - 2);
            }
            root.children.push(cell);
        }
        root
    }

    /// Apply ONE random mutation to a clone of `old`, keeping survivor node_ids
    /// stable and minting fresh ids for inserts. Mirrors the design's mutation set.
    fn mutate_tree(old: &LayoutBox, rng: &mut impl FnMut() -> u64, counter: &mut u64) -> LayoutBox {
        let mut t = old.clone();
        if t.children.is_empty() {
            return t;
        }
        let r = rng();
        let kind = r % 10;
        let idx = (r as usize >> 4) % t.children.len();
        match kind {
            0 => {
                // recolor
                t.children[idx].background =
                    Some(LColor { r: (r >> 8) as u8, g: (r >> 16) as u8, b: (r >> 24) as u8, a: 255 });
                t.children[idx].kind = BoxKind::Block { tag: "div".into() };
            }
            1 => {
                // change text
                t.children[idx].kind = BoxKind::Text(format!("m{}", r % 1000));
            }
            2 => {
                // translate subtree
                t.children[idx].content.x += ((r >> 8) % 20) as f32;
                t.children[idx].content.y += ((r >> 16) % 15) as f32;
            }
            3 => {
                // insert a child (fresh id)
                let mut extra = base_box();
                extra.node_id = nid(*counter);
                *counter += 1;
                extra.content = Rect {
                    x: 10.0 + (r % 100) as f32,
                    y: 10.0 + ((r >> 8) % 80) as f32,
                    w: 20.0,
                    h: 16.0,
                };
                extra.background = Some(LColor { r: 30, g: 180, b: 120, a: 255 });
                t.children.push(extra);
            }
            4 => {
                // remove a child
                t.children.remove(idx);
            }
            5 => {
                // toggle opacity
                t.children[idx].opacity = if t.children[idx].opacity < 0.99 { 1.0 } else { 0.5 };
            }
            6 => {
                // toggle overflow_hidden
                t.children[idx].overflow_hidden = !t.children[idx].overflow_hidden;
            }
            7 => {
                // add/remove a box_shadow
                t.children[idx].box_shadow = if t.children[idx].box_shadow.is_some() {
                    None
                } else {
                    Some(BoxShadow {
                        offset_x_px: 5.0,
                        offset_y_px: 5.0,
                        blur_px: 0.0,
                        spread_px: 3.0,
                        color: LColor { r: 0, g: 0, b: 0, a: 110 },
                        inset: false,
                    })
                };
            }
            8 => {
                // set rotate_deg (affine)
                t.children[idx].rotate_deg = 15.0 + (r % 30) as f32;
            }
            _ => {
                // resize w/h
                t.children[idx].content.w += ((r >> 8) % 16) as f32;
                t.children[idx].content.h += ((r >> 16) % 12) as f32;
            }
        }
        t
    }

    // ── M5.4 GENERAL-PATH oracle (render_paint_only_focused widening) ───────────
    // The 2 fast-bake sites are covered by `assert_incremental_oracle` (vs
    // oracle_replay_paint). The general path (path 3 @ main.rs) builds its cache
    // via a FULL bake = `oracle_live_paint` (the live immediate-mode bake the
    // runtime actually serves), then on a small DOM delta serves the incremental
    // composite. These cases prove that the incremental result the GENERAL path
    // would serve is BYTE-IDENTICAL to the full bake the runtime would otherwise
    // produce — comparing against `oracle_live_paint` specifically (not just the
    // replay), so the runtime invariant is end-to-end, on the exact mutation
    // shapes path 3 produces: class toggle (bg/color), single text-node content
    // change, display:none↔block (added/removed subtree), attribute geometry
    // shift (moved). max diff 0 or documented full-bake fallback.

    /// Build the cache the GENERAL path uses (a full bake == `oracle_live_paint`,
    /// the live bake the runtime serves and seeds `prev.bitmap` from), run the
    /// incremental composite for old→new, and assert the served pixels equal a
    /// FULL bake of `new` (`oracle_live_paint`) byte-for-byte. Returns true if the
    /// incremental path RAN, false if it fell back (None). chrome_h=0 (content-only
    /// bitmap, matching the runtime call site).
    fn assert_general_path_oracle(old_lb: &LayoutBox, new_lb: &LayoutBox) -> bool {
        let c = cfg();
        // Cache exactly as the general path seeds it: a live full bake of old.
        let (cached, _) = super::super::oracle_live_paint(old_lb, &c);
        let old_rdl = generate(old_lb, &c);
        let new_rdl = generate(new_lb, &c);
        let d = diff(&old_rdl, &new_rdl);
        // The full-bake reference the runtime would otherwise serve for `new`.
        let (full, _) = super::super::oracle_live_paint(new_lb, &c);
        match incremental_composite(&cached, &old_rdl, &new_rdl, &d, 0) {
            None => false, // documented full-bake fallback fired (still correct).
            Some(incr) => {
                assert_eq!(incr.width, full.width, "general-path incr width != full");
                assert_eq!(incr.height, full.height, "general-path incr height != full");
                let mut first: Option<usize> = None;
                let mut maxd: u64 = 0;
                for (i, (a, b)) in incr.pixels.iter().zip(full.pixels.iter()).enumerate() {
                    if a != b {
                        if first.is_none() {
                            first = Some(i);
                        }
                        maxd = maxd.max(pixel_abs_diff(*a, *b));
                    }
                }
                assert_eq!(
                    maxd, 0,
                    "GENERAL-PATH INCREMENTAL != LIVE FULL BAKE: first diff at pixel {:?} maxd={} diff={:?}",
                    first, maxd, d
                );
                assert_eq!(
                    incr.pixels, full.pixels,
                    "general-path incremental Vec<u32> != oracle_live_paint full bake"
                );
                true
            }
        }
    }

    #[test]
    fn genpath_class_toggle_bg_color() {
        // A class toggle that changes ONE node's background color (the most common
        // path-3 micro-delta). Incremental served == live full bake, byte-identical.
        let t1 = grid_tree(12);
        let mut t2 = grid_tree(12);
        if let Some(cell) = t2.children.iter_mut().find(|c| c.node_id == nid(101)) {
            cell.background = Some(LColor { r: 12, g: 200, b: 64, a: 255 });
        }
        assert!(
            assert_general_path_oracle(&t1, &t2),
            "single bg-color toggle should stay incremental"
        );
    }

    #[test]
    fn genpath_class_toggle_text_color() {
        // Class toggle that changes a text node's COLOR (not its string) — content
        // hash changes, damage = the glyph region only.
        let t1 = grid_tree(12);
        let mut t2 = grid_tree(12);
        // node 100 (i=0) is a text cell ("c0").
        if let Some(cell) = t2.children.iter_mut().find(|c| c.node_id == nid(100)) {
            cell.text_color = LColor { r: 200, g: 0, b: 120, a: 255 };
        }
        assert!(assert_general_path_oracle(&t1, &t2));
    }

    #[test]
    fn genpath_single_text_node_content_change() {
        // A single text-node content change (e.g. el.textContent = "…") — the
        // canonical live-update mutation. Incremental == live full bake.
        let t1 = grid_tree(12);
        let mut t2 = grid_tree(12);
        if let Some(cell) = t2.children.iter_mut().find(|c| c.node_id == nid(103)) {
            cell.kind = BoxKind::Text("UPDATED".into());
            cell.background = None;
            cell.text_color = LColor { r: 10, g: 10, b: 20, a: 255 };
        }
        // ensure old also had this as a text node so it's a content change, not a
        // type switch (i=3 → i%3==0 → already a text cell "c3").
        assert!(assert_general_path_oracle(&t1, &t2));
    }

    #[test]
    fn genpath_display_none_to_block_adds_subtree() {
        // display:none → block reveals a node: the OLD tree omits it, the NEW tree
        // adds it (added-subtree class). Incremental must paint the new subtree.
        let mut t1 = grid_tree(11); // 11 cells (ids 100..110)
        let t2 = {
            let mut t = grid_tree(11);
            // The revealed node (fresh id, was display:none in old ⇒ absent).
            let mut shown = base_box();
            shown.node_id = nid(800);
            shown.content = Rect { x: 60.0, y: 100.0, w: 50.0, h: 30.0 };
            shown.background = Some(LColor { r: 180, g: 40, b: 200, a: 255 });
            t.children.push(shown);
            t
        };
        // Keep id sets disjoint only for node 800; both share 100..110.
        let _ = &mut t1;
        assert!(assert_general_path_oracle(&t1, &t2));
    }

    #[test]
    fn genpath_block_to_display_none_removes_subtree() {
        // block → display:none hides a node: the NEW tree omits it (removed-subtree
        // class). Incremental must erase the old pixels + repaint what was behind.
        let t1 = {
            let mut t = grid_tree(11);
            let mut vis = base_box();
            vis.node_id = nid(801);
            vis.content = Rect { x: 60.0, y: 100.0, w: 50.0, h: 30.0 };
            vis.background = Some(LColor { r: 40, g: 160, b: 220, a: 255 });
            t.children.push(vis);
            t
        };
        let t2 = grid_tree(11); // node 801 absent ⇒ removed.
        assert!(assert_general_path_oracle(&t1, &t2));
    }

    #[test]
    fn genpath_attribute_geometry_shift_moves_node() {
        // An attribute-driven geometry shift (e.g. style.left changes) rigid-moves
        // one node — moved class: erase moved-FROM, paint moved-TO. Byte-identical.
        let t1 = grid_tree(12);
        let mut t2 = grid_tree(12);
        if let Some(cell) = t2.children.iter_mut().find(|c| c.node_id == nid(105)) {
            cell.content.x += 22.0;
            cell.content.y += 16.0;
        }
        assert!(assert_general_path_oracle(&t1, &t2));
    }

    #[test]
    fn genpath_verify_hash_matches_on_good_frame() {
        // CV_DAMAGE_VERIFY semantics: on a correct incremental frame the
        // hash_bitmap_pixels(incremental) == hash_bitmap_pixels(live full bake), so
        // verify serves the incremental frame (the win). This exercises the exact
        // primitives the runtime self-check uses (hash_bitmap_pixels + oracle_live_paint)
        // without depending on the OnceLock env flags.
        let c = cfg();
        let t1 = grid_tree(12);
        let mut t2 = grid_tree(12);
        if let Some(cell) = t2.children.iter_mut().find(|c| c.node_id == nid(101)) {
            cell.background = Some(LColor { r: 12, g: 200, b: 64, a: 255 });
        }
        let cached = cache_for(&t1, &c);
        let old_rdl = generate(&t1, &c);
        let new_rdl = generate(&t2, &c);
        let d = diff(&old_rdl, &new_rdl);
        let incr = incremental_composite(&cached, &old_rdl, &new_rdl, &d, 0)
            .expect("small color delta should stay incremental");
        let (full, _) = super::super::oracle_live_paint(&t2, &c);
        // The runtime compares these two hashes; equal ⇒ serve incremental.
        assert_eq!(
            super::super::hash_bitmap_pixels(&incr),
            super::super::hash_bitmap_pixels(&full),
            "verify hash of incremental must equal hash of live full bake on a good frame"
        );
    }

    #[test]
    fn genpath_verify_hash_detects_corruption_and_would_self_heal() {
        // CV_DAMAGE_VERIFY self-heal: if the incremental bitmap were ever wrong, its
        // hash would differ from the live full bake's hash, the runtime would log
        // MISMATCH and serve the FULL bake (correct). Simulate corruption by flipping
        // one pixel of the incremental result and assert the hashes diverge — proving
        // the detector catches a wrong frame (the self-heal path then routes to the
        // full bake, which by construction equals `full`).
        let c = cfg();
        let t1 = grid_tree(12);
        let mut t2 = grid_tree(12);
        if let Some(cell) = t2.children.iter_mut().find(|c| c.node_id == nid(101)) {
            cell.background = Some(LColor { r: 12, g: 200, b: 64, a: 255 });
        }
        let cached = cache_for(&t1, &c);
        let old_rdl = generate(&t1, &c);
        let new_rdl = generate(&t2, &c);
        let d = diff(&old_rdl, &new_rdl);
        let mut incr = incremental_composite(&cached, &old_rdl, &new_rdl, &d, 0).unwrap();
        let (full, _) = super::super::oracle_live_paint(&t2, &c);
        // Good frame: hashes equal (sanity).
        assert_eq!(
            super::super::hash_bitmap_pixels(&incr),
            super::super::hash_bitmap_pixels(&full)
        );
        // Inject a 1-pixel corruption => the detector must see a mismatch.
        let mid = incr.pixels.len() / 2;
        incr.pixels[mid] ^= 0x00_FF_00_00;
        assert_ne!(
            super::super::hash_bitmap_pixels(&incr),
            super::super::hash_bitmap_pixels(&full),
            "verify must detect a corrupted incremental frame (then self-heal to full bake)"
        );
        // Self-heal reference: the full bake the runtime would serve on mismatch is
        // byte-identical to the live full bake (it IS bake_layout_into_paint's content).
        let (heal, _) = super::super::oracle_live_paint(&t2, &c);
        assert_eq!(heal.pixels, full.pixels, "self-heal full bake == live full bake");
    }

    #[test]
    fn genpath_no_change_serves_cache_verbatim() {
        // A path-3 frame with NO visual delta (e.g. a non-visual attribute changed)
        // ⇒ empty diff ⇒ the incremental path reuses the cache verbatim ==
        // live full bake.
        let t = grid_tree(12);
        assert!(assert_general_path_oracle(&t, &t));
    }

    #[test]
    fn genpath_large_delta_falls_back_to_full() {
        // A structural change touching MANY nodes (>= FULL_BAKE_CHUNK_FRAC) must
        // FALL BACK to full bake (return None) — path 3's common large-DOM-delta
        // case. The fallback is the always-correct full bake; assert it fires.
        let t1 = grid_tree(20);
        let mut t2 = grid_tree(20);
        // recolor 12 of 20 cells (> 0.4 * 21 chunks) ⇒ over the chunk threshold.
        for (i, cell) in t2.children.iter_mut().enumerate() {
            if i < 12 {
                cell.background = Some(LColor { r: (i * 9) as u8, g: 30, b: 200, a: 255 });
                cell.kind = BoxKind::Block { tag: "div".into() };
            }
        }
        assert!(
            !assert_general_path_oracle(&t1, &t2),
            "large multi-node delta must fall back to full bake"
        );
    }

    /// General-path incremental == `oracle_replay_paint` (the M5.4 byte-identity
    /// invariant: the incremental machinery reproduces a from-scratch replay of
    /// the new list, max diff 0) AND, since the affine z_meta fix, replay ==
    /// `oracle_live_paint` (the live immediate-mode bake the runtime serves) — also
    /// max diff 0. Returns whether the incremental path RAN (false ⇒ full-bake
    /// fallback). Used by the general-path fuzz.
    ///
    /// HISTORY: this used to assert ONLY incr-vs-replay and DOCUMENT (not assert) a
    /// replay-vs-live divergence (the affine-child z-bucket reversal — see
    /// `affine_zbucket_replay_matches_live`). That bug is now fixed (generate_rec's
    /// affine early-return captures z_meta), so the documented-divergence allowance
    /// is REMOVED and replay-vs-live is asserted at max diff 0 on every fuzz case.
    fn assert_general_path_replay_oracle(old_lb: &LayoutBox, new_lb: &LayoutBox) -> bool {
        let c = cfg();
        let cached = cache_for(old_lb, &c);
        let old_rdl = generate(old_lb, &c);
        let new_rdl = generate(new_lb, &c);
        let d = diff(&old_rdl, &new_rdl);
        let (full, _) = oracle_replay_paint(new_lb, &c);
        // replay MUST byte-match the live painter (the contract). With the affine
        // z_meta fix this holds on the entire wide-overlap corpus.
        let (live, _) = super::super::oracle_live_paint(new_lb, &c);
        assert_eq!(
            replay_vs_live_maxd(&full, &live),
            0,
            "general-path: replay diverged from live (the affine z-bucket fix should make this 0)"
        );
        match incremental_composite(&cached, &old_rdl, &new_rdl, &d, 0) {
            None => false,
            Some(incr) => {
                assert_eq!(incr.pixels, full.pixels, "general-path incr != replay full bake");
                true
            }
        }
    }

    /// Worst per-channel absolute difference between two bitmaps (0 ⇒ byte-identical).
    /// Mismatched dimensions ⇒ 255 (a maximal divergence, never silently 0).
    fn replay_vs_live_maxd(a: &Bitmap, b: &Bitmap) -> u64 {
        if a.width != b.width || a.height != b.height || a.pixels.len() != b.pixels.len() {
            return 255;
        }
        let mut maxd = 0u64;
        for (x, y) in a.pixels.iter().zip(b.pixels.iter()) {
            if x != y {
                let d = pixel_abs_diff(*x, *y);
                if d > maxd {
                    maxd = d;
                }
            }
        }
        maxd
    }

    #[test]
    fn genpath_fuzz_byte_identical() {
        // The general-path analog of the M5.4 fuzz over 200 random small deltas.
        //
        // ASSERTS BOTH (via assert_general_path_replay_oracle):
        //   * incremental == `oracle_replay_paint` (the M5.4 byte-identity invariant
        //     — the incremental composite reproduces a from-scratch replay, max diff 0).
        //   * replay == `oracle_live_paint` (the live bake the runtime serves), max
        //     diff 0 — the contract.
        //
        // HISTORY: this fuzz formerly asserted ONLY incr-vs-replay and DOCUMENTED a
        // replay-vs-live divergence on the adversarial `rand_tree_wide` corpus —
        // rotated (affine) absolute/z-index-bucketed text over shadowed 0.6-opacity
        // siblings. That was NOT a "maxd=1 rounding shimmer" but a hard z-order
        // reversal (up to maxd=180): generate_rec's affine early-return never captured
        // the affine child's z_meta, so replay's painted_child_order mis-bucketed it
        // and reversed overlapping siblings' paint order. FIXED (z_meta captured in the
        // affine early-return); the documented-divergence allowance is REMOVED and the
        // fuzz now asserts replay-vs-live == 0 on every case (the corpus GENERATES the
        // exact trigger: rand_tree_wide gives absolute+z-index cells, mutate_tree kind 8
        // rotates them).
        let mut state: u64 = 0x5eed_1234_9abc_def0;
        let mut rng = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let mut ran = 0usize;
        let mut fell = 0usize;
        for _ in 0..200 {
            let mut counter: u64 = 1;
            let old = rand_tree_wide(&mut rng, &mut counter);
            let new = mutate_tree(&old, &mut rng, &mut counter);
            if assert_general_path_replay_oracle(&old, &new) {
                ran += 1;
            } else {
                fell += 1;
            }
        }
        assert!(
            ran > 20,
            "general-path fuzz: incremental must run on many cases, got ran={} fell={}",
            ran,
            fell
        );
    }

    // ── M5.4 SUSTAINED SOAK (in-process runtime-loop simulation) ───────────────
    // The default-on soak gate, run deterministically in-process per MEMORY's hard
    // rule (no live/looped/corpus probes — that combo froze the machine). Models
    // the EXACT runtime general-path frame loop:
    //   * `prev` cache = a LIVE full bake (oracle_live_paint) — exactly how
    //     bake_layout_into_paint seeds prev.bitmap.
    //   * each frame: generate old/new RDL, diff, incremental_composite.
    //   * with CV_DAMAGE_VERIFY ON (soak/diagnostic): hash incr vs LIVE full bake;
    //     MATCH ⇒ serve incr (the win); MISMATCH ⇒ self-heal to the full bake.
    //   * the SERVED bitmap becomes the next frame's cache (modeling
    //     `tab.paint = paint.clone()`, which drops the prior frame — bounded cache).
    // over 800 frames across a stable grid SPA + the adversarial wide-overlap corpus.
    //
    // WHAT IT PROVES (all GREEN):
    //   (1) PRODUCTION CACHE MODEL: the cache is seeded from the LIVE bake (prev.bitmap),
    //       so incremental = LIVE pixels outside the damage region R + a from-scratch
    //       replay inside R. The correct correctness metric is therefore incr-vs-LIVE
    //       (NOT incr-vs-pure-replay: a pure replay differs from live wherever the
    //       pre-existing M5.2 replay-vs-live edge lands OUTSIDE R, but incr reuses the
    //       LIVE cache there and matches live — so incr-vs-replay aggregate diverges
    //       harmlessly; the existing fuzz covers the replay-seeded invariant).
    //   (2) SELF-HEAL with VERIFY ON: every served frame (incremental-served OR
    //       self-healed full bake) is byte-identical to the live full bake — with
    //       CV_DAMAGE_VERIFY on the user never sees a wrong frame. Self-heal is
    //       load-bearing (it fires on the edge below and corrects it).
    //   (3) BOUNDED CACHE: exactly one prev frame retained (the served bitmap replaces
    //       the cache each frame; the old one drops).
    //   (4) REAL WIN: incremental served on the large majority of frames.
    //
    // WHAT IT NOW ASSERTS (FLIP READINESS — was the FLIP BLOCKER):
    //   The combined-overlap frames (rotated/absolute/z-bucketed text over shadowed
    //   0.6-opacity siblings) previously made `generate`/`replay` diverge from the live
    //   `paint_box` painter — measured here at replay_vs_live up to ~75 per channel
    //   (originally mislabeled a "maxd=1 rounding shimmer"; it was a hard z-order
    //   reversal). ROOT CAUSE: generate_rec's affine early-return never captured the
    //   affine child's z_meta, so replay's painted_child_order mis-bucketed rotated
    //   positioned/z-indexed children and REVERSED overlapping siblings' paint order.
    //   FIXED (z_meta captured in the affine early-return). The soak now ASSERTS
    //   replay_vs_live == 0 (and the verify-OFF would-serve-wrong magnitude == 0)
    //   across this entire adversarial corpus, so the default is safe to flip ON for
    //   this class. The verify-on self-heal + CV_DAMAGE_VERIFY escape hatch stay as
    //   belt-and-suspenders for any future, unrelated raster gap.
    #[test]
    fn soak_general_path_runtime_loop() {
        let c = cfg();
        let mut state: u64 = 0x50AC_u64.wrapping_mul(0x9E37_79B9_7F4A_7C15);
        let mut rng = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };

        let mut served = 0usize; // frames served as incremental (the win)
        let mut healed = 0usize; // verify MISMATCH ⇒ self-healed to full bake
        let mut fellback = 0usize; // incremental_composite returned None ⇒ full bake
        let mut frames = 0usize;
        let mut wrong_frames_verify_on = 0usize; // served != live, VERIFY ON (MUST be 0)
        let mut seam_inside_r_violations = 0usize; // incr != replay INSIDE R (MUST be 0)
        // What PRODUCTION (verify OFF) would show on the pre-existing edge.
        let mut prod_max_pixdiff: u64 = 0;
        let mut prod_mismatch_frames = 0usize;
        let mut worst_replay_vs_live: u64 = 0;

        for corpus in 0..2 {
            let mut counter: u64 = 1;
            let mut cur = if corpus == 0 {
                grid_tree(14)
            } else {
                rand_tree_wide(&mut rng, &mut counter)
            };
            // Seed the prev cache from a LIVE full bake — exactly as the runtime does.
            let (mut cache, _) = super::super::oracle_live_paint(&cur, &c);
            // Parallel REPLAY-seeded cache: lets us assert the pure seam invariant
            // (incr == replay when the cache is replay-seeded) decoupled from the
            // pre-existing replay-vs-live edge, exactly as the existing M5.4 fuzz does.
            let mut cache_replay = cache_for(&cur, &c);

            for _ in 0..400 {
                frames += 1;
                let next = mutate_tree(&cur, &mut rng, &mut counter);

                let old_rdl = generate(&cur, &c);
                let new_rdl = generate(&next, &c);
                let d = diff(&old_rdl, &new_rdl);
                let (live_full, _) = super::super::oracle_live_paint(&next, &c);
                let (replay_full, _) = oracle_replay_paint(&next, &c);

                // SEAM INVARIANT (replay-seeded): incr == pure replay, max diff 0. This
                // is the M5.4 byte-identity the widening must preserve, independent of
                // the replay-vs-live edge. Computed on the parallel replay cache.
                if let Some(incr_r) =
                    incremental_composite(&cache_replay, &old_rdl, &new_rdl, &d, 0)
                {
                    if incr_r.pixels != replay_full.pixels {
                        seam_inside_r_violations += 1;
                    }
                    cache_replay = incr_r;
                } else {
                    cache_replay = replay_full.clone();
                }

                // PRODUCTION FRAME (live-seeded cache): exactly what the runtime serves.
                let served_bmp: Bitmap = match incremental_composite(
                    &cache, &old_rdl, &new_rdl, &d, 0,
                ) {
                    None => {
                        fellback += 1;
                        live_full.clone()
                    }
                    Some(incr) => {
                        // VERIFY ON: hash incr vs live full bake (incr-vs-LIVE, the real
                        // production correctness metric).
                        if super::super::hash_bitmap_pixels(&incr)
                            == super::super::hash_bitmap_pixels(&live_full)
                        {
                            served += 1;
                            incr // serve incremental (the win)
                        } else {
                            // MISMATCH ⇒ self-heal to the full bake (verify-on path).
                            healed += 1;
                            prod_mismatch_frames += 1;
                            // Production (verify-off) would have served `incr`: measure
                            // the worst per-channel diff the user would have seen.
                            let n = incr.pixels.len().min(live_full.pixels.len());
                            if incr.pixels.len() == live_full.pixels.len() {
                                for i in 0..n {
                                    if incr.pixels[i] != live_full.pixels[i] {
                                        prod_max_pixdiff = prod_max_pixdiff
                                            .max(pixel_abs_diff(incr.pixels[i], live_full.pixels[i]));
                                    }
                                }
                            } else {
                                prod_max_pixdiff = 255;
                            }
                            // Confirm the gap is the pre-existing replay-vs-live edge.
                            if replay_full.pixels.len() == live_full.pixels.len() {
                                for i in 0..replay_full.pixels.len() {
                                    if replay_full.pixels[i] != live_full.pixels[i] {
                                        worst_replay_vs_live = worst_replay_vs_live.max(
                                            pixel_abs_diff(replay_full.pixels[i], live_full.pixels[i]),
                                        );
                                    }
                                }
                            }
                            live_full.clone()
                        }
                    }
                };

                // VERIFY-ON correctness: served frame == live full bake, always.
                if served_bmp.pixels != live_full.pixels {
                    wrong_frames_verify_on += 1;
                }
                // Bounded cache: served bitmap replaces the cache, old one drops.
                cache = served_bmp;
                cur = next;
            }
        }

        // (1) Seam-widening invariant: replay-seeded incremental ALWAYS == pure replay
        //     (max diff 0). This is the M5.4 byte-identity the widening preserves; it is
        //     INDEPENDENT of the pre-existing replay-vs-live edge.
        assert_eq!(
            seam_inside_r_violations, 0,
            "SOAK: the widened incremental composite (replay-seeded) diverged from a \
             from-scratch replay on {} frame(s) — this widening introduced a raster bug.",
            seam_inside_r_violations
        );
        // (2) With verify ON the user never sees a wrong frame (self-heal works).
        assert_eq!(
            wrong_frames_verify_on, 0,
            "SOAK: {} served frame(s) diverged from the live full bake WITH VERIFY ON \
             — the self-heal failed.",
            wrong_frames_verify_on
        );
        // (4) The win is real.
        assert!(
            served > frames / 2,
            "SOAK: incremental served on too few frames (served={} of {}); win vacuous \
             (healed={} fellback={}).",
            served, frames, healed, fellback
        );

        eprintln!(
            "[SOAK] frames={} served={} healed(self-heal)={} fellback(full)={} \
             seam_violations={} wrong_frames(verify-on)={}",
            frames, served, healed, fellback, seam_inside_r_violations, wrong_frames_verify_on
        );
        eprintln!(
            "[SOAK] FLIP READINESS: PROD(verify-OFF) would-serve-wrong frames={} \
             worst per-channel pixdiff={} (== replay-vs-live={}; seam incr-vs-replay=0). \
             0 across the corpus ⇒ replay byte-matches live ⇒ READY to flip.",
            prod_mismatch_frames, prod_max_pixdiff, worst_replay_vs_live
        );

        // FLIP READINESS (was the FLIP BLOCKER): the affine z_meta bug that made
        // generate/replay diverge from the live painter on the combined-overlap class
        // (rotated absolute z-bucketed shadowed/semi-opaque siblings) is FIXED — the
        // affine early-return now captures z_meta, so replay buckets affine children
        // exactly as the live painter. Therefore replay == live across this whole
        // adversarial corpus, and a verify-OFF production frame can no longer be served
        // wrong by THIS class. Assert it: the worst replay-vs-live divergence on any
        // frame is 0, and the verify-off path would never have served a wrong frame.
        assert_eq!(
            worst_replay_vs_live, 0,
            "SOAK: replay diverged from the live painter by maxd={} on the corpus \
             (the affine z-bucket fix should make this 0).",
            worst_replay_vs_live
        );
        assert_eq!(
            prod_max_pixdiff, 0,
            "SOAK: a verify-OFF production frame would have been served with maxd={} \
             vs live (must be 0 after the affine z-bucket fix).",
            prod_max_pixdiff
        );
        assert_eq!(
            prod_mismatch_frames, 0,
            "SOAK: {} frame(s) had incr != live with verify ON (self-heal still fired); \
             after the fix the combined-overlap class no longer diverges, so this is 0.",
            prod_mismatch_frames
        );
    }

    // ── M5.4 BOUNDED-CACHE proof (no Arc/bitmap accumulation across frames) ─────
    // The runtime keeps the prev frame in `tab.paint` and commits via
    // `tab.paint = paint.clone()`, which drops the prior PaintData ⇒ its
    // Arc<Bitmap> + Arc<RetainedDisplayList> refcounts fall to 0. This models that
    // loop with Arc strong_count assertions to PROVE exactly one prev frame is
    // retained at any time — never a growing chain (the soak's RSS-flat requirement
    // as a deterministic in-process check, no live run needed).
    #[test]
    fn cache_bounded_one_prev_frame_no_accumulation() {
        use std::sync::Arc;
        let c = cfg();
        // The committed cache slot (models tab.paint's Arc fields).
        let mut tree = grid_tree(12);
        let (b0, _) = super::super::oracle_live_paint(&tree, &c);
        let mut cache_bitmap: Arc<Bitmap> = Arc::new(b0);
        let mut cache_rdl: Arc<RetainedDisplayList> = Arc::new(generate(&tree, &c));

        for frame in 0..50 {
            // Borrow prev WITHOUT cloning the Arc (downcast_ref borrows in the runtime).
            assert_eq!(
                Arc::strong_count(&cache_bitmap),
                1,
                "frame {}: prev bitmap Arc must be uniquely held (no accumulation)",
                frame
            );
            assert_eq!(
                Arc::strong_count(&cache_rdl),
                1,
                "frame {}: prev RDL Arc must be uniquely held (no accumulation)",
                frame
            );
            // Mutate one cell (a sub-threshold delta the incremental path serves).
            let id = nid(100 + (frame % 12) as u64);
            if let Some(cell) = tree.children.iter_mut().find(|x| x.node_id == id) {
                cell.background = Some(LColor {
                    r: (frame * 5) as u8,
                    g: 40,
                    b: 200,
                    a: 255,
                });
            }
            let new_rdl = generate(&tree, &c);
            let old_rdl: &RetainedDisplayList = &cache_rdl;
            let d = diff(old_rdl, &new_rdl);
            let new_bmp = incremental_composite(&cache_bitmap, old_rdl, &new_rdl, &d, 0)
                .unwrap_or_else(|| super::super::oracle_live_paint(&tree, &c).0);
            // COMMIT: wrap fresh Arcs and replace the slot — the old Arcs drop here.
            let new_bitmap_arc = Arc::new(new_bmp);
            let new_rdl_arc = Arc::new(new_rdl);
            cache_bitmap = new_bitmap_arc; // old cache_bitmap dropped (refcount→0)
            cache_rdl = new_rdl_arc; // old cache_rdl dropped (refcount→0)
        }
        // After 50 frames the slot still holds exactly ONE frame.
        assert_eq!(Arc::strong_count(&cache_bitmap), 1);
        assert_eq!(Arc::strong_count(&cache_rdl), 1);
    }

    // ── DIAGNOSIS: combined-overlap replay-vs-live divergence reproducer ────────
    //
    // Build the exact interaction the M5.4 soak surfaced: >=2 absolute-positioned,
    // z-index-bucketed, MUTUALLY-OVERLAPPING siblings, each with box-shadow,
    // element opacity 0.6, and text. Compare oracle_live_paint vs generate+replay
    // and report the worst pixel. Prints the live op walk vs the replay op walk so
    // the exact divergence is visible. Marked #[ignore] so it does not gate the
    // suite while diagnosing; run with `--ignored`.
    fn overlap_sibling(id: u64, x: f32, y: f32, z: i32, txt: &str, txtcol: LColor) -> LayoutBox {
        let mut c = base_box();
        c.node_id = nid(id);
        c.position = Position::Absolute;
        c.z_index = Some(z);
        c.opacity = 0.6;
        c.content = Rect { x, y, w: 60.0, h: 40.0 };
        c.background = Some(LColor { r: 200, g: 80, b: 40, a: 255 });
        c.box_shadow = Some(BoxShadow {
            offset_x_px: 4.0,
            offset_y_px: 4.0,
            blur_px: 0.0,
            spread_px: 2.0,
            color: LColor { r: 0, g: 0, b: 0, a: 160 },
            inset: false,
        });
        // text child carried as the box's own kind via a text sub-box so glyphs land
        let mut tnode = base_box();
        tnode.node_id = nid(id + 1000);
        tnode.kind = BoxKind::Text(txt.to_string());
        tnode.text_color = txtcol;
        tnode.content = Rect { x: x + 4.0, y: y + 4.0, w: 52.0, h: 20.0 };
        tnode.font_size_px = 14.0;
        c.children.push(tnode);
        c
    }

    fn combined_overlap_tree() -> LayoutBox {
        let mut root = base_box();
        root.node_id = nid(1);
        root.content = Rect { x: 0.0, y: 0.0, w: 200.0, h: 150.0 };
        root.background = Some(LColor { r: 250, g: 250, b: 252, a: 255 });
        // Three overlapping absolute siblings with DIFFERENT z so the bucket order
        // is non-trivial; they all overlap in the 40..70 region.
        root.children.push(overlap_sibling(10, 20.0, 20.0, 2, "AAA", LColor { r: 0, g: 0, b: 0, a: 255 }));
        root.children.push(overlap_sibling(20, 40.0, 30.0, 1, "BBB", LColor { r: 255, g: 255, b: 255, a: 255 }));
        root.children.push(overlap_sibling(30, 35.0, 25.0, 3, "CCC", LColor { r: 10, g: 10, b: 200, a: 255 }));
        root
    }

    fn flatten_live_then_replay_report(lb: &LayoutBox) -> u64 {
        let c = cfg();
        let (live, _lt) = super::super::oracle_live_paint(lb, &c);
        let (replayed, _rt) = oracle_replay_paint(lb, &c);
        assert_eq!(live.width, replayed.width);
        assert_eq!(live.height, replayed.height);
        let w = live.width as usize;
        let mut maxd: u64 = 0;
        let mut worst_i: usize = 0;
        let mut ndiff = 0usize;
        for (i, (a, b)) in live.pixels.iter().zip(replayed.pixels.iter()).enumerate() {
            if a != b {
                ndiff += 1;
                let d = pixel_abs_diff(*a, *b);
                if d > maxd {
                    maxd = d;
                    worst_i = i;
                }
            }
        }
        let unpack = |v: u32| -> (u8, u8, u8, u8) {
            ((v >> 16) as u8, (v >> 8) as u8, v as u8, (v >> 24) as u8) // bgra_u32: B<<0? check below
        };
        // to_bgra_u32 packs as (a<<24)|(r<<16)|(g<<8)|b — decode r,g,b,a:
        let dec = |v: u32| -> (u8, u8, u8, u8) {
            (((v >> 16) & 0xFF) as u8, ((v >> 8) & 0xFF) as u8, (v & 0xFF) as u8, ((v >> 24) & 0xFF) as u8)
        };
        let _ = unpack;
        let (lx, ly) = (worst_i % w, worst_i / w);
        let lc = dec(live.pixels[worst_i]);
        let rc = dec(replayed.pixels[worst_i]);
        eprintln!(
            "[OVERLAP DIAG] ndiff={} maxd={} worst=({},{}) live(rgba)={:?} replay(rgba)={:?}",
            ndiff, maxd, lx, ly, lc, rc
        );
        maxd
    }

    /// Hunt the exact soak corpus for the first `next` tree where replay diverges
    /// from live by >= 50 per channel, then dump its structure so we can minimise.
    #[test]
    #[ignore]
    fn diag_hunt_soak_divergence() {
        let c = cfg();
        let mut state: u64 = 0x50AC_u64.wrapping_mul(0x9E37_79B9_7F4A_7C15);
        let mut rng = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let mut worst: u64 = 0;
        let mut worst_tree: Option<LayoutBox> = None;
        for corpus in 0..2 {
            let mut counter: u64 = 1;
            let mut cur = if corpus == 0 { grid_tree(14) } else { rand_tree_wide(&mut rng, &mut counter) };
            for frame in 0..400 {
                let next = mutate_tree(&cur, &mut rng, &mut counter);
                let (live, _) = super::super::oracle_live_paint(&next, &c);
                let (rep, _) = oracle_replay_paint(&next, &c);
                if live.pixels.len() == rep.pixels.len() {
                    let mut md = 0u64;
                    for (a, b) in live.pixels.iter().zip(rep.pixels.iter()) {
                        if a != b { md = md.max(pixel_abs_diff(*a, *b)); }
                    }
                    if md > worst {
                        worst = md;
                        worst_tree = Some(next.clone());
                        eprintln!("[HUNT] corpus={} frame={} new worst maxd={}", corpus, frame, md);
                    }
                }
                cur = next;
            }
        }
        eprintln!("[HUNT] overall worst maxd={}", worst);
        if let Some(t) = &worst_tree {
            dump_tree_brief(t, 0);
            // Report the worst pixel + colors on the worst tree.
            let _ = flatten_live_then_replay_report(t);
        }
        // This is a hunt, not a gate — always "passes" but prints the worst case.
        assert!(worst >= 0);
    }

    fn dump_tree_brief(b: &LayoutBox, depth: usize) {
        let pad = "  ".repeat(depth);
        let kind = match &b.kind {
            BoxKind::Text(t) => format!("Text({:?})", t),
            BoxKind::Block { tag } => format!("Block({})", tag),
            _ => "?".into(),
        };
        eprintln!(
            "[HUNT] {}id={:?} {} pos={:?} z={:?} op={} content=({},{},{},{}) bg={:?} shadow={} nchild={}",
            pad, b.node_id, kind, b.position, b.z_index, b.opacity,
            b.content.x, b.content.y, b.content.w, b.content.h,
            b.background.is_some(), b.box_shadow.is_some(), b.children.len()
        );
        for ch in &b.children {
            dump_tree_brief(ch, depth + 1);
        }
    }

    /// Rebuild the EXACT frame-20 corpus tree (deterministic), then progressively
    /// strip cells to find the minimal subset that still diverges, dumping the op
    /// walk that touches the worst pixel.
    fn corpus_frame20_tree() -> LayoutBox {
        let c = cfg();
        let _ = c;
        let mut state: u64 = 0x50AC_u64.wrapping_mul(0x9E37_79B9_7F4A_7C15);
        let mut rng = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        // corpus 0 consumes a fixed number of rng draws (grid_tree(14) uses none;
        // its 400 mutate frames each draw exactly 1 rng). Then corpus 1 builds
        // rand_tree_wide and runs mutate frames. We replay deterministically.
        let mut counter: u64 = 1;
        // corpus 0
        let mut cur = grid_tree(14);
        for _ in 0..400 {
            cur = mutate_tree(&cur, &mut rng, &mut counter);
        }
        // corpus 1
        let mut counter1: u64 = 1;
        let mut cur1 = rand_tree_wide(&mut rng, &mut counter1);
        for frame in 0..=20 {
            let next = mutate_tree(&cur1, &mut rng, &mut counter1);
            if frame == 20 {
                return next;
            }
            cur1 = next;
        }
        unreachable!()
    }

    fn report_maxd(lb: &LayoutBox) -> (u64, usize, (u8, u8, u8, u8), (u8, u8, u8, u8), usize) {
        let c = cfg();
        let (live, _) = super::super::oracle_live_paint(lb, &c);
        let (rep, _) = oracle_replay_paint(lb, &c);
        let w = live.width as usize;
        let mut md = 0u64;
        let mut wi = 0usize;
        for (i, (a, b)) in live.pixels.iter().zip(rep.pixels.iter()).enumerate() {
            if a != b {
                let d = pixel_abs_diff(*a, *b);
                if d > md { md = d; wi = i; }
            }
        }
        let dec = |v: u32| (((v >> 16) & 0xFF) as u8, ((v >> 8) & 0xFF) as u8, (v & 0xFF) as u8, ((v >> 24) & 0xFF) as u8);
        (md, wi, dec(live.pixels[wi]), dec(rep.pixels[wi]), w)
    }

    #[test]
    #[ignore]
    fn diag_minimise_corpus_frame20() {
        let full = corpus_frame20_tree();
        let (md, wi, lc, rc, w) = report_maxd(&full);
        eprintln!("[MIN] full tree maxd={} worst=({},{}) live={:?} replay={:?}", md, wi % w, wi / w, lc, rc);
        dump_tree_brief(&full, 0);

        // Greedy minimisation: drop children one at a time while maxd stays >= 50.
        let mut tree = full.clone();
        let mut changed = true;
        while changed {
            changed = false;
            let n = tree.children.len();
            for i in 0..n {
                if tree.children.len() <= 1 { break; }
                let mut cand = tree.clone();
                cand.children.remove(i);
                let (cmd, _, _, _, _) = report_maxd(&cand);
                if cmd >= 50 {
                    tree = cand;
                    changed = true;
                    break;
                }
            }
        }
        let (md2, wi2, lc2, rc2, w2) = report_maxd(&tree);
        eprintln!("[MIN] MINIMAL tree ({} children) maxd={} worst=({},{}) live={:?} replay={:?}",
            tree.children.len(), md2, wi2 % w2, wi2 / w2, lc2, rc2);
        dump_tree_brief(&tree, 0);

        // Dump the op walks (live recursion vs replay flatten) touching the worst pixel.
        let c = cfg();
        let rdl = generate(&tree, &c);
        let root = rdl.root;
        eprintln!("[MIN] root doc children={:?} painted={:?}",
            rdl.chunks[root as usize].children, painted_child_order(&rdl, root));
        let px = (wi2 % w2) as i32;
        let py = (wi2 / w2) as i32;
        eprintln!("[MIN] === ops touching worst pixel ({},{}) in REPLAY z-order ===", px, py);
        dump_ops_touching(&rdl, root, px, py);
        eprintln!("[MIN] === ALL ops + texts per chunk ===");
        for ci in 0..rdl.chunks.len() as u32 {
            let ch = &rdl.chunks[ci as usize];
            eprintln!("[MIN] chunk {} id={:?} op={} z={:?} positioned={} needs_sort={} bounds=({},{},{},{})",
                ci, ch.node_id, ch.opacity, ch.z_meta.effective_z, ch.z_meta.is_positioned, ch.needs_sort,
                ch.bounds.x, ch.bounds.y, ch.bounds.w, ch.bounds.h);
            for op in &ch.ops_before_children { eprintln!("[MIN]    B {:?}", op); }
            for ti in &ch.text_items { eprintln!("[MIN]    T x={} y={} w={} h={} fs={} col={:?} a={} {:?}", ti.x, ti.y, ti.w, ti.h, ti.font_size_px, ti.color_rgb, ti.color_alpha, ti.text); }
            for op in &ch.ops_after_children { eprintln!("[MIN]    A {:?}", op); }
        }
        // Now FLATTEN the replay op stream (in z order) like flatten_layer_ops does,
        // to see exactly what replay issues — this is the actual replay sequence.
        eprintln!("[MIN] === FLATTENED REPLAY op stream (z-order) ===");
        let mut flat: Vec<PaintOp> = Vec::new();
        flatten_layer_ops(&rdl, root, &mut flat);
        for op in &flat { eprintln!("[MIN]   R {:?}", op); }
        // Dump each child's geometry as seen by generate.
        eprintln!("[MIN] === child geometry ===");
        for ch in &tree.children {
            let br = ch.border_rect();
            eprintln!("[MIN]   id={:?} pos={:?} z={:?} content=({},{},{},{}) border_rect=({},{},{},{}) bg={:?} shadow={:?}",
                ch.node_id, ch.position, ch.z_index,
                ch.content.x, ch.content.y, ch.content.w, ch.content.h,
                br.x, br.y, br.w, br.h, ch.background, ch.box_shadow.map(|s|(s.offset_x_px,s.offset_y_px,s.spread_px,s.color.a)));
        }
        assert!(md2 >= 0);
    }

    fn op_touches(op: &PaintOp, px: i32, py: i32) -> bool {
        if let Some((x0, y0, x1, y1)) = op.extent_screen() {
            (px as f32) >= x0 && (px as f32) < x1 && (py as f32) >= y0 && (py as f32) < y1
        } else {
            false
        }
    }

    fn dump_ops_touching(rdl: &RetainedDisplayList, ci: u32, px: i32, py: i32) {
        let c = &rdl.chunks[ci as usize];
        if c.visibility_hidden || c.opacity < 0.01 { return; }
        for op in &c.ops_before_children {
            if op_touches(op, px, py) {
                eprintln!("[MIN]   chunk {} (id={:?} op={}) BEFORE op {:?}", ci, c.node_id, c.opacity, op);
            }
        }
        for cidx in painted_child_order(rdl, ci) {
            dump_ops_touching(rdl, cidx, px, py);
        }
        for op in &c.ops_after_children {
            if op_touches(op, px, py) {
                eprintln!("[MIN]   chunk {} (id={:?} op={}) AFTER op {:?}", ci, c.node_id, c.opacity, op);
            }
        }
    }

    /// CLEAN hand-built reproducer of the M5.4 soak divergence, no corpus seed.
    /// Two ROTATED (affine) overlapping siblings under a parent: a STATIC one with
    /// a dark shadow (paints in z-bucket B) and an ABSOLUTE z=-1 one with an opaque
    /// bg (z-bucket A → must paint FIRST/behind). The live painter + generate's own
    /// scratch order them [absolute(A), static(B)]; replay re-derives the order from
    /// the affine chunk's z_meta, which generate_rec's affine early-return NEVER
    /// sets (stays {None,false}) — so replay buckets BOTH to B in doc order
    /// [static, absolute], REVERSING them. The opaque absolute bg then lands on top
    /// in replay but behind in live → a visible color error at the overlap.
    fn affine_zbucket_overlap_tree() -> LayoutBox {
        let mut root = base_box();
        root.node_id = nid(1);
        root.content = Rect { x: 0.0, y: 0.0, w: 200.0, h: 150.0 };
        root.background = Some(LColor { r: 250, g: 250, b: 252, a: 255 });

        // doc child 0: STATIC, dark shadow, rotated. z-bucket B (paints LAST in live).
        let mut a = base_box();
        a.node_id = nid(10);
        a.position = Position::Static;
        a.content = Rect { x: 70.0, y: 40.0, w: 50.0, h: 40.0 };
        a.box_shadow = Some(BoxShadow {
            offset_x_px: 6.0, offset_y_px: 6.0, blur_px: 0.0, spread_px: 3.0,
            color: LColor { r: 0, g: 0, b: 0, a: 200 }, inset: false,
        });
        a.rotate_deg = 20.0;
        root.children.push(a);

        // doc child 1: ABSOLUTE z=-1, opaque bg, rotated. z-bucket A (paints FIRST).
        let mut b = base_box();
        b.node_id = nid(20);
        b.position = Position::Absolute;
        b.z_index = Some(-1);
        b.content = Rect { x: 78.0, y: 48.0, w: 50.0, h: 40.0 };
        b.background = Some(LColor { r: 200, g: 230, b: 230, a: 255 });
        b.rotate_deg = 20.0;
        root.children.push(b);

        root
    }

    #[test]
    fn affine_zbucket_replay_matches_live() {
        let lb = affine_zbucket_overlap_tree();
        let c = cfg();
        let (live, _) = super::super::oracle_live_paint(&lb, &c);
        let (rep, _) = oracle_replay_paint(&lb, &c);
        let w = live.width as usize;
        let mut maxd = 0u64;
        let mut wi = 0usize;
        for (i, (x, y)) in live.pixels.iter().zip(rep.pixels.iter()).enumerate() {
            if x != y {
                let d = pixel_abs_diff(*x, *y);
                if d > maxd { maxd = d; wi = i; }
            }
        }
        let dec = |v: u32| (((v >> 16) & 0xFF) as u8, ((v >> 8) & 0xFF) as u8, (v & 0xFF) as u8, ((v >> 24) & 0xFF) as u8);
        eprintln!("[AFFINE-Z] maxd={} worst=({},{}) live={:?} replay={:?}", maxd, wi % w, wi / w, dec(live.pixels[wi]), dec(rep.pixels[wi]));
        assert_eq!(maxd, 0, "affine z-bucket order: replay must reproduce live (maxd={})", maxd);
    }

    /// Build a root whose children are a small set of MUTUALLY-OVERLAPPING siblings,
    /// each randomly given a combination of: absolute-position + z-index bucket,
    /// semi-transparent opacity, a box-shadow, text-vs-background content, and an
    /// affine rotation. This is the exact "combined-overlap class" the soak surfaced
    /// (overlapping absolute z-bucketed semi-opaque shadowed text siblings) and the
    /// class the affine z_meta bug broke. The siblings are placed at deliberately
    /// overlapping coordinates so their z-bucket paint ORDER is observable in the
    /// overlap pixels — i.e. an order reversal produces a non-zero replay-vs-live diff.
    fn combined_overlap_fuzz_tree(rng: &mut impl FnMut() -> u64, counter: &mut u64) -> LayoutBox {
        let mut root = base_box();
        root.node_id = nid(*counter);
        *counter += 1;
        root.content = Rect { x: 0.0, y: 0.0, w: 200.0, h: 160.0 };
        root.background = Some(LColor { r: 250, g: 250, b: 252, a: 255 });

        let r0 = rng();
        let n = 3 + (r0 % 4) as usize; // 3..6 overlapping siblings
        for i in 0..n {
            let r = rng();
            let mut cell = base_box();
            cell.node_id = nid(*counter);
            *counter += 1;
            // Cluster the cells in a tight region so they OVERLAP each other.
            let jitterx = (r % 24) as f32;
            let jittery = ((r >> 5) % 24) as f32;
            cell.content = Rect {
                x: 60.0 + jitterx,
                y: 40.0 + jittery,
                w: 40.0 + ((r >> 10) % 20) as f32,
                h: 32.0 + ((r >> 14) % 16) as f32,
            };
            // Content: text or background (the live painter folds opacity onto either).
            if r & 1 != 0 {
                cell.kind = BoxKind::Text(format!("o{i}"));
                cell.text_color = LColor { r: 20, g: 20, b: 30, a: 255 };
                // Text cells still get a backing bg sometimes so the overlap shows color.
                if r & 0x10000 != 0 {
                    cell.background = Some(LColor {
                        r: (r >> 1) as u8,
                        g: (r >> 9) as u8,
                        b: (r >> 17) as u8,
                        a: 255,
                    });
                }
            } else {
                cell.background = Some(LColor {
                    r: (r >> 1) as u8,
                    g: (r >> 9) as u8,
                    b: (r >> 17) as u8,
                    a: 255,
                });
            }
            // box-shadow (a near-opaque dark ring that lands on a sibling's bg).
            if r & 2 != 0 {
                cell.box_shadow = Some(BoxShadow {
                    offset_x_px: 4.0,
                    offset_y_px: 4.0,
                    blur_px: 0.0,
                    spread_px: 3.0,
                    color: LColor { r: 0, g: 0, b: 0, a: 160 + (r % 96) as u8 },
                    inset: false,
                });
            }
            // semi-transparent element opacity (the V1.5 per-op fold).
            if r & 4 != 0 {
                cell.opacity = 0.6;
            }
            // absolute position + z-index bucket (A=neg / C=pos-stacked / D=pos>0).
            if r & 8 != 0 {
                cell.position = Position::Absolute;
                cell.z_index = Some((r % 5) as i32 - 2); // -2..2
            }
            // affine rotation (the path that broke z_meta capture).
            if r & 0x20 != 0 {
                cell.rotate_deg = 15.0 + (r % 30) as f32;
            }
            root.children.push(cell);
        }
        root
    }

    /// FUZZ: the combined-overlap class — overlapping absolute z-bucketed semi-opaque
    /// shadowed text siblings, some rotated — must replay BYTE-IDENTICAL to the live
    /// painter (max diff 0). This is the deterministic generator for the bug the
    /// affine z_meta fix closed; it asserts replay-vs-LIVE (not replay-vs-replay).
    #[test]
    fn combined_overlap_fuzz_replay_eq_live() {
        let c = cfg();
        let mut state: u64 = 0xC0FF_EE13_579B_DF24;
        let mut rng = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let mut worst: u64 = 0;
        let mut worst_seed = 0u64;
        let mut affine_cases = 0usize;
        let mut abs_z_cases = 0usize;
        for iter in 0..400 {
            let mut counter: u64 = 1;
            let lb = combined_overlap_fuzz_tree(&mut rng, &mut counter);
            // Track that the corpus actually GENERATES the trigger interaction.
            if lb.children.iter().any(|x| x.rotate_deg.abs() > 1e-3) {
                affine_cases += 1;
            }
            if lb.children.iter().any(|x| {
                !matches!(x.position, Position::Static) && x.z_index.is_some()
            }) {
                abs_z_cases += 1;
            }
            let (live, _) = super::super::oracle_live_paint(&lb, &c);
            let (rep, _) = oracle_replay_paint(&lb, &c);
            let d = replay_vs_live_maxd(&rep, &live);
            if d > worst {
                worst = d;
                worst_seed = iter as u64;
            }
        }
        eprintln!(
            "[OVERLAP-FUZZ] iters=400 worst_maxd={} worst_seed={} affine_cases={} abs_z_cases={}",
            worst, worst_seed, affine_cases, abs_z_cases
        );
        // The corpus must actually exercise BOTH affine-rotated siblings AND
        // absolute/z-indexed siblings (the bug's preconditions), else the gate is vacuous.
        assert!(affine_cases > 50, "overlap fuzz never generated rotated siblings ({affine_cases})");
        assert!(abs_z_cases > 50, "overlap fuzz never generated absolute z-indexed siblings ({abs_z_cases})");
        assert_eq!(worst, 0, "combined-overlap fuzz: replay must byte-match live (worst maxd={worst})");
    }

    #[test]
    #[ignore]
    fn diag_combined_overlap_repro() {
        let lb = combined_overlap_tree();
        let maxd = flatten_live_then_replay_report(&lb);
        // Also print the painted child order vs document order at the root and the
        // op-count per chunk so the structural difference (if any) is visible.
        let c = cfg();
        let rdl = generate(&lb, &c);
        let root = rdl.root;
        eprintln!("[OVERLAP DIAG] root doc children = {:?}", rdl.chunks[root as usize].children);
        eprintln!("[OVERLAP DIAG] root painted order = {:?}", painted_child_order(&rdl, root));
        for &ci in &rdl.chunks[root as usize].children {
            let ch = &rdl.chunks[ci as usize];
            eprintln!(
                "[OVERLAP DIAG]  chunk {} z_meta={:?} before_ops={} after_ops={} texts={} children={:?}",
                ci, ch.z_meta, ch.ops_before_children.len(), ch.ops_after_children.len(),
                ch.text_items.len(), ch.children
            );
        }
        assert_eq!(maxd, 0, "combined-overlap replay diverged from live by maxd={}", maxd);
    }
}
