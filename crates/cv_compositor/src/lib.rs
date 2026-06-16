//! `cv_compositor` — layer tree, tile cache, present.
//!
//! V1 ships the compositor data structures and the per-frame produce
//! pipeline. A `LayerTree` is a flat list of `Layer`s with per-layer
//! transforms, opacity, and clip rects. The `TileCache` divides each
//! layer's painted bitmap into fixed-size tiles so partial scrolls
//! only re-blit dirty tiles. `composite_frame()` walks the layer
//! tree, samples each layer's tiles into the destination buffer,
//! applies opacity, and returns the final BGRA bitmap ready for
//! present.
//!
//! Software composite for V1; future revisions will route through
//! `cv_gpu` for DirectComposition-backed tear-free present.

use std::collections::HashMap;

pub mod display_list_gpu;
pub mod display_list_rasterize;
pub mod promote;
pub mod tree_builder;

/// One compositor layer. A layer corresponds to a stacking context or
/// to a `will-change`-promoted element.
#[derive(Debug, Clone)]
pub struct Layer {
    pub id: u32,
    /// Per-layer transform (pixel translation only in V1).
    pub translate_x: f32,
    pub translate_y: f32,
    /// Per-layer scale (identity = 1.0).
    pub scale_x: f32,
    pub scale_y: f32,
    /// 0..1 opacity blend over the underlying buffer.
    pub opacity: f32,
    /// Clip rectangle in compositor-space pixels. None = unclipped.
    pub clip_rect: Option<Rect>,
    /// Painted contents: a row-major BGRA u32 buffer plus its size.
    pub bitmap: Vec<u32>,
    pub bitmap_w: u32,
    pub bitmap_h: u32,
    /// Z-index for ordering. Higher = on top.
    pub z_index: i32,
    /// Which property tree nodes drive this layer's transform/effect/clip.
    /// The compositor uses these to resolve world-space values without
    /// re-running layout or paint.
    pub tree_state: cv_paint::PropertyTreeState,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rect {
    pub x: i32,
    pub y: i32,
    pub w: i32,
    pub h: i32,
}

impl Rect {
    pub fn intersect(self, other: Rect) -> Option<Rect> {
        let x = self.x.max(other.x);
        let y = self.y.max(other.y);
        let r = (self.x + self.w).min(other.x + other.w);
        let b = (self.y + self.h).min(other.y + other.h);
        if r > x && b > y {
            Some(Rect {
                x,
                y,
                w: r - x,
                h: b - y,
            })
        } else {
            None
        }
    }
}

/// Layer tree — flat sorted-by-z-index list.
#[derive(Debug, Default)]
pub struct LayerTree {
    pub layers: Vec<Layer>,
}

impl LayerTree {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn push(&mut self, layer: Layer) {
        self.layers.push(layer);
    }
    /// Sort layers by z_index ascending so painter's-algorithm composite
    /// renders the right thing on top.
    pub fn sort_by_z(&mut self) {
        self.layers.sort_by_key(|l| l.z_index);
    }

    /// Compositor-only update: walk the property trees and apply new
    /// transform/opacity values to each layer's screen-space fields
    /// **without** re-rasterizing any layer's bitmap. This is the
    /// fast path for CSS animations that only touch
    /// `transform`/`opacity` — they skip layout and paint entirely.
    ///
    /// Returns `true` if any layer's composite-time fields actually
    /// changed (so the caller knows whether to re-composite the tiles).
    pub fn apply_property_tree_update(
        &mut self,
        trees: &cv_paint::PropertyTrees,
    ) -> bool {
        let mut changed = false;
        for layer in &mut self.layers {
            let ts = layer.tree_state;

            // Resolve world transform from the tree chain.
            let (tx, ty, sx, sy) = trees.world_transform(ts.transform_id);
            if (layer.translate_x - tx).abs() > 0.001
                || (layer.translate_y - ty).abs() > 0.001
                || (layer.scale_x - sx).abs() > 0.001
                || (layer.scale_y - sy).abs() > 0.001
            {
                layer.translate_x = tx;
                layer.translate_y = ty;
                layer.scale_x = sx;
                layer.scale_y = sy;
                changed = true;
            }

            // Resolve world opacity from the effect tree.
            let opacity = trees.world_opacity(ts.effect_id);
            if (layer.opacity - opacity).abs() > 0.001 {
                layer.opacity = opacity;
                changed = true;
            }

            // Resolve world clip from the clip tree (if any nodes exist).
            if !trees.clips.is_empty() {
                let clip = trees.world_clip(ts.clip_id);
                let clip_rect = clip.map(|c| Rect {
                    x: c.x,
                    y: c.y,
                    w: c.w as i32,
                    h: c.h as i32,
                });
                if layer.clip_rect != clip_rect {
                    layer.clip_rect = clip_rect;
                    changed = true;
                }
            }
        }
        changed
    }
}

/// Tile size — 256x256 BGRA pixels. Smaller = more granular dirty
/// tracking; larger = less per-tile overhead. 256 is Chromium's
/// default and a reasonable starting point.
pub const TILE_SIZE: u32 = 256;

/// One tile in the cache. Stores its raw BGRA pixels plus a dirty
/// flag the compositor flips when the underlying layer repaints.
#[derive(Debug, Clone)]
pub struct Tile {
    pub pixels: Vec<u32>, // TILE_SIZE * TILE_SIZE BGRA
    pub dirty: bool,
}

/// Per-layer tile cache. Indexed by (tile_x, tile_y).
#[derive(Debug, Default)]
pub struct TileCache {
    tiles: HashMap<(u32, u32), Tile>,
    /// Last-known bitmap dimensions, updated on each refresh.
    /// `composite_viewport` uses this to clip reads to the real content
    /// area so out-of-bounds tile padding (transparent black) doesn't
    /// overwrite the white canvas background.
    bitmap_w: u32,
    bitmap_h: u32,
}

impl TileCache {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn tile(&self, tx: u32, ty: u32) -> Option<&Tile> {
        self.tiles.get(&(tx, ty))
    }

    /// Rebuild dirty tiles from `layer`'s backing bitmap. Tiles that
    /// are still clean are left alone — that's the cache's point.
    pub fn refresh_from_layer(&mut self, layer: &Layer) {
        self.bitmap_w = layer.bitmap_w;
        self.bitmap_h = layer.bitmap_h;
        let cols = (layer.bitmap_w + TILE_SIZE - 1) / TILE_SIZE;
        let rows = (layer.bitmap_h + TILE_SIZE - 1) / TILE_SIZE;
        for ty in 0..rows {
            for tx in 0..cols {
                let entry = self.tiles.entry((tx, ty)).or_insert_with(|| Tile {
                    pixels: vec![0u32; (TILE_SIZE * TILE_SIZE) as usize],
                    dirty: true,
                });
                if entry.dirty {
                    fill_tile_from_bitmap(entry, layer, tx, ty);
                    entry.dirty = false;
                }
            }
        }
    }

    /// Mark every tile dirty so the next refresh repaints everything.
    pub fn invalidate_all(&mut self) {
        for t in self.tiles.values_mut() {
            t.dirty = true;
        }
    }

    /// Mark a region dirty in cache-tile space. Called when a scroll
    /// or layout delta is bounded.
    pub fn invalidate_rect(&mut self, rect: Rect) {
        let x0 = (rect.x.max(0) as u32) / TILE_SIZE;
        let y0 = (rect.y.max(0) as u32) / TILE_SIZE;
        let x1 = ((rect.x + rect.w).max(0) as u32 + TILE_SIZE - 1) / TILE_SIZE;
        let y1 = ((rect.y + rect.h).max(0) as u32 + TILE_SIZE - 1) / TILE_SIZE;
        for ty in y0..y1 {
            for tx in x0..x1 {
                if let Some(t) = self.tiles.get_mut(&(tx, ty)) {
                    t.dirty = true;
                }
            }
        }
    }

    pub fn tile_count(&self) -> usize {
        self.tiles.len()
    }

    /// Rebuild dirty tiles directly from a raw BGRA pixel slice —
    /// avoids cloning into a `Layer` when the caller already holds the
    /// bitmap data in the same u32-per-pixel layout.
    pub fn refresh_from_raw(&mut self, pixels: &[u32], bitmap_w: u32, bitmap_h: u32) {
        self.bitmap_w = bitmap_w;
        self.bitmap_h = bitmap_h;
        let cols = (bitmap_w + TILE_SIZE - 1) / TILE_SIZE;
        let rows = (bitmap_h + TILE_SIZE - 1) / TILE_SIZE;
        for ty in 0..rows {
            for tx in 0..cols {
                let entry = self.tiles.entry((tx, ty)).or_insert_with(|| Tile {
                    pixels: vec![0u32; (TILE_SIZE * TILE_SIZE) as usize],
                    dirty: true,
                });
                if entry.dirty {
                    fill_tile_from_raw(entry, pixels, bitmap_w, bitmap_h, tx, ty);
                    entry.dirty = false;
                }
            }
        }
    }

    /// Composite the visible viewport from cached tiles into a fresh
    /// BGRA pixel buffer. `src_x`/`src_y` define the top-left corner
    /// of the viewport in bitmap space (src_y = scroll offset).
    /// The returned buffer has dimensions `width × height` and can be
    /// blitted to the screen in a single StretchDIBits call.
    pub fn composite_viewport(
        &self,
        src_x: i32,
        src_y: i32,
        width: u32,
        height: u32,
    ) -> Vec<u32> {
        let n = (width as usize) * (height as usize);
        let mut out = vec![0xFFFF_FFFFu32; n]; // white background
        if width == 0 || height == 0 {
            return out;
        }
        let ts = TILE_SIZE;
        // Clip the readable source area to the actual bitmap so we
        // never copy out-of-bounds tile padding (transparent black)
        // over the white canvas background.
        let bmp_w = self.bitmap_w as i32;
        let bmp_h = self.bitmap_h as i32;
        let clip_right = (src_x + width as i32).min(bmp_w);
        let clip_bottom = (src_y + height as i32).min(bmp_h);
        if clip_right <= src_x || clip_bottom <= src_y {
            return out; // viewport is entirely beyond the bitmap
        }
        let tx0 = (src_x.max(0) as u32) / ts;
        let ty0 = (src_y.max(0) as u32) / ts;
        let tx1 = ((clip_right.max(0) as u32) + ts - 1) / ts;
        let ty1 = ((clip_bottom.max(0) as u32) + ts - 1) / ts;

        for ty in ty0..ty1 {
            for tx in tx0..tx1 {
                if let Some(tile) = self.tiles.get(&(tx, ty)) {
                    let tile_left = (tx * ts) as i32;
                    let tile_top = (ty * ts) as i32;

                    let vis_left = tile_left.max(src_x);
                    let vis_top = tile_top.max(src_y);
                    let vis_right = (tile_left + ts as i32)
                        .min(src_x + width as i32)
                        .min(bmp_w);
                    let vis_bottom = (tile_top + ts as i32)
                        .min(src_y + height as i32)
                        .min(bmp_h);
                    if vis_left >= vis_right || vis_top >= vis_bottom {
                        continue;
                    }

                    let row_len = (vis_right - vis_left) as usize;
                    for y in vis_top..vis_bottom {
                        let tly = (y - tile_top) as u32;
                        let tlx = (vis_left - tile_left) as u32;
                        let src_off = (tly * ts + tlx) as usize;
                        let dst_y = (y - src_y) as u32;
                        let dst_x = (vis_left - src_x) as u32;
                        let dst_off = (dst_y * width + dst_x) as usize;
                        out[dst_off..dst_off + row_len]
                            .copy_from_slice(&tile.pixels[src_off..src_off + row_len]);
                    }
                }
            }
        }
        out
    }
}

/// Slice a `width × height` viewport DIRECTLY out of a raw row-major BGRA bitmap
/// at `(src_x, src_y)`, with no tile cache. This is the single-copy fast path the
/// UI present uses every frame: the full-document/band bitmap is re-rastered each
/// frame anyway (no cross-frame tile reuse), so routing it through the TileCache
/// meant copying the WHOLE bitmap into tiles (`refresh_from_raw`) AND copying the
/// visible slice back out — two ~40MB copies/frame for a 3440px page. This does
/// ONE copy of just the visible rows. Out-of-bitmap area stays white. Identical
/// output to `TileCache::composite_viewport` for a freshly-refreshed full cache
/// (verified by `direct_viewport_matches_tilecache`).
pub fn composite_viewport_direct(
    pixels: &[u32],
    bitmap_w: u32,
    bitmap_h: u32,
    src_x: i32,
    src_y: i32,
    width: u32,
    height: u32,
) -> Vec<u32> {
    let n = (width as usize) * (height as usize);
    let mut out = vec![0xFFFF_FFFFu32; n]; // white background
    if width == 0 || height == 0 {
        return out;
    }
    let bmp_w = bitmap_w as i32;
    let bmp_h = bitmap_h as i32;
    // Visible intersection of the requested viewport with the actual bitmap.
    let vis_left = src_x.max(0);
    let vis_top = src_y.max(0);
    let vis_right = (src_x + width as i32).min(bmp_w);
    let vis_bottom = (src_y + height as i32).min(bmp_h);
    if vis_left >= vis_right || vis_top >= vis_bottom {
        return out; // viewport entirely beyond the bitmap
    }
    let row_len = (vis_right - vis_left) as usize;
    for y in vis_top..vis_bottom {
        let src_off = (y as usize) * (bitmap_w as usize) + vis_left as usize;
        let dst_y = (y - src_y) as usize;
        let dst_x = (vis_left - src_x) as usize;
        let dst_off = dst_y * (width as usize) + dst_x;
        out[dst_off..dst_off + row_len].copy_from_slice(&pixels[src_off..src_off + row_len]);
    }
    out
}

fn fill_tile_from_raw(
    tile: &mut Tile,
    pixels: &[u32],
    bitmap_w: u32,
    bitmap_h: u32,
    tx: u32,
    ty: u32,
) {
    let bx0 = tx * TILE_SIZE;
    let by0 = ty * TILE_SIZE;
    for ly in 0..TILE_SIZE {
        for lx in 0..TILE_SIZE {
            let src_x = bx0 + lx;
            let src_y = by0 + ly;
            let pixel = if src_x < bitmap_w && src_y < bitmap_h {
                pixels[(src_y as usize) * (bitmap_w as usize) + src_x as usize]
            } else {
                0
            };
            tile.pixels[(ly * TILE_SIZE + lx) as usize] = pixel;
        }
    }
}

fn fill_tile_from_bitmap(tile: &mut Tile, layer: &Layer, tx: u32, ty: u32) {
    let bx0 = tx * TILE_SIZE;
    let by0 = ty * TILE_SIZE;
    for ly in 0..TILE_SIZE {
        for lx in 0..TILE_SIZE {
            let src_x = bx0 + lx;
            let src_y = by0 + ly;
            let pixel = if src_x < layer.bitmap_w && src_y < layer.bitmap_h {
                layer.bitmap[(src_y as usize) * (layer.bitmap_w as usize) + src_x as usize]
            } else {
                0
            };
            tile.pixels[(ly * TILE_SIZE + lx) as usize] = pixel;
        }
    }
}

/// Composite a layer tree into a fresh BGRA output buffer of the
/// given size. Walks each layer in z-order, applies its translate +
/// opacity, blends into the output.
pub fn composite_frame(tree: &LayerTree, out_w: u32, out_h: u32, background: u32) -> Vec<u32> {
    let n = (out_w as usize) * (out_h as usize);
    let mut out = vec![background; n];
    for layer in &tree.layers {
        composite_layer(&mut out, out_w, out_h, layer);
    }
    out
}

fn composite_layer(out: &mut [u32], out_w: u32, out_h: u32, layer: &Layer) {
    let off_x = layer.translate_x as i32;
    let off_y = layer.translate_y as i32;
    let opacity = layer.opacity.clamp(0.0, 1.0);
    if opacity < 0.01 {
        return;
    }
    let layer_rect = Rect {
        x: off_x,
        y: off_y,
        w: layer.bitmap_w as i32,
        h: layer.bitmap_h as i32,
    };
    let view = Rect {
        x: 0,
        y: 0,
        w: out_w as i32,
        h: out_h as i32,
    };
    let visible = match layer_rect.intersect(view) {
        Some(r) => r,
        None => return,
    };
    let visible = if let Some(layer_clip) = layer.clip_rect {
        match visible.intersect(layer_clip) {
            Some(r) => r,
            None => return,
        }
    } else {
        visible
    };
    for dy in visible.y..(visible.y + visible.h) {
        for dx in visible.x..(visible.x + visible.w) {
            let lx = dx - off_x;
            let ly = dy - off_y;
            if lx < 0 || ly < 0 {
                continue;
            }
            let lx = lx as u32;
            let ly = ly as u32;
            if lx >= layer.bitmap_w || ly >= layer.bitmap_h {
                continue;
            }
            let src = layer.bitmap[(ly as usize) * (layer.bitmap_w as usize) + lx as usize];
            let dst_idx = (dy as usize) * (out_w as usize) + dx as usize;
            out[dst_idx] = blend_with_opacity(out[dst_idx], src, opacity);
        }
    }
}

/// Source-over blend with a per-layer opacity multiplier. Operates on
/// BGRA u32 pixels (alpha in the high byte).
///
/// MUST stay byte-identical to `cv_gfx::blend_bgra` — the straight-alpha
/// (non-premultiplied) Porter-Duff source-over that is the faint-particle fix.
/// The previous form here produced PREMULTIPLIED output RGB
/// (`out_r = sr*sa + dr*da*(1-sa)` with no `/out_a` normalize) and TRUNCATED
/// (`* 255.0 as u32`), so colors composited onto a transparent backing store
/// were dragged toward the backdrop's zero RGB (gold 255→76) and then lost a
/// further fractional bit to truncation — exactly the bug `cv_gfx::blend_bgra`
/// was fixed for. We now run the SAME normalize-by-`out_a` + `.round()` math.
///
/// `cv_compositor` does not depend on `cv_gfx` (and must not gain that edge —
/// `cv_paint`, its only dependency, also has no `cv_gfx` edge), so the oracle
/// body is replicated here verbatim. The shared unit test
/// `converged_blend_matches_straight_alpha_oracle` pins the byte-for-byte
/// equality against hand-computed oracle values.
///
/// Bridge from the oracle's `(dst, Color)` contract to this `(dst, u32, opacity)`
/// one: fold `opacity` into the source alpha (`sa = src_a/255 * opacity`) BEFORE
/// applying the formula. Early-out contract is preserved: return `dst` unchanged
/// when the effective source alpha is non-positive (the compositor never wants to
/// disturb the backing store for a fully-transparent source); the oracle returns
/// 0 on `out_a <= 0`, which is the correct result for an isolated Color but the
/// wrong contract for an in-place layer composite.
fn blend_with_opacity(dst: u32, src: u32, opacity: f32) -> u32 {
    // Effective source alpha in 0..1, with the per-layer opacity folded in.
    let sa = ((src >> 24) & 0xFF) as f32 / 255.0 * opacity;
    if sa <= 0.0 {
        return dst;
    }
    // Source channels in 0..255 (sRGB byte space, no gamma).
    let sr = ((src >> 16) & 0xFF) as f32;
    let sg = ((src >> 8) & 0xFF) as f32;
    let sb = (src & 0xFF) as f32;
    let da = ((dst >> 24) & 0xFF) as f32 / 255.0;
    let dr = ((dst >> 16) & 0xFF) as f32;
    let dg = ((dst >> 8) & 0xFF) as f32;
    let db = (dst & 0xFF) as f32;
    let inv = 1.0 - sa;
    let out_a = sa + da * inv;
    if out_a <= 0.0 {
        return dst;
    }
    // Straight-alpha source-over: alpha-weight the destination and normalize
    // the output RGB by the output alpha (NOT premultiplied), then round.
    let r = ((sr * sa + dr * da * inv) / out_a).round() as u32;
    let g = ((sg * sa + dg * da * inv) / out_a).round() as u32;
    let b = ((sb * sa + db * da * inv) / out_a).round() as u32;
    let a = (out_a * 255.0).round() as u32;
    (a << 24) | (r << 16) | (g << 8) | b
}

#[cfg(test)]
mod tests {
    use super::*;

    fn solid_layer(id: u32, color: u32, w: u32, h: u32, z: i32, x: f32, y: f32) -> Layer {
        Layer {
            id,
            translate_x: x,
            translate_y: y,
            scale_x: 1.0,
            scale_y: 1.0,
            opacity: 1.0,
            clip_rect: None,
            bitmap: vec![color; (w * h) as usize],
            bitmap_w: w,
            bitmap_h: h,
            z_index: z,
            tree_state: cv_paint::PropertyTreeState::default(),
        }
    }

    #[test]
    fn rect_intersect_basic() {
        let a = Rect {
            x: 0,
            y: 0,
            w: 10,
            h: 10,
        };
        let b = Rect {
            x: 5,
            y: 5,
            w: 10,
            h: 10,
        };
        assert_eq!(
            a.intersect(b),
            Some(Rect {
                x: 5,
                y: 5,
                w: 5,
                h: 5
            })
        );
        let c = Rect {
            x: 100,
            y: 100,
            w: 5,
            h: 5,
        };
        assert!(a.intersect(c).is_none());
    }

    #[test]
    fn composite_single_layer_paints_pixels() {
        let mut tree = LayerTree::new();
        tree.push(solid_layer(0, 0xFFFF0000, 4, 4, 0, 0.0, 0.0));
        let out = composite_frame(&tree, 4, 4, 0xFF000000);
        for &p in &out {
            assert_eq!(p, 0xFFFF0000);
        }
    }

    #[test]
    fn z_ordering_paints_top_on_top() {
        let mut tree = LayerTree::new();
        tree.push(solid_layer(0, 0xFFFF0000, 4, 4, 0, 0.0, 0.0));
        tree.push(solid_layer(1, 0xFF00FF00, 2, 2, 5, 1.0, 1.0));
        tree.sort_by_z();
        let out = composite_frame(&tree, 4, 4, 0xFF000000);
        assert_eq!(out[0], 0xFFFF0000);
        assert_eq!(out[1 * 4 + 1], 0xFF00FF00);
    }

    #[test]
    fn translate_offsets_layer() {
        let mut tree = LayerTree::new();
        tree.push(solid_layer(0, 0xFF0000FF, 2, 2, 0, 2.0, 2.0));
        let out = composite_frame(&tree, 4, 4, 0);
        assert_eq!(out[0], 0);
        assert_eq!(out[2 * 4 + 2], 0xFF0000FF);
    }

    #[test]
    fn opacity_blends_through_to_background() {
        let mut tree = LayerTree::new();
        tree.push(Layer {
            id: 0,
            translate_x: 0.0,
            translate_y: 0.0,
            scale_x: 1.0,
            scale_y: 1.0,
            opacity: 0.5,
            clip_rect: None,
            bitmap: vec![0xFFFFFFFFu32; 4],
            bitmap_w: 2,
            bitmap_h: 2,
            z_index: 0,
            tree_state: cv_paint::PropertyTreeState::default(),
        });
        let out = composite_frame(&tree, 2, 2, 0xFF000000);
        let p = out[0];
        let r = (p >> 16) & 0xFF;
        assert!((100..=160).contains(&r));
    }

    #[test]
    fn tile_cache_refresh_writes_pixels() {
        let layer = solid_layer(0, 0xFF112233, TILE_SIZE * 2, TILE_SIZE, 0, 0.0, 0.0);
        let mut cache = TileCache::new();
        cache.refresh_from_layer(&layer);
        assert_eq!(cache.tile_count(), 2);
        let t = cache.tile(0, 0).unwrap();
        assert_eq!(t.pixels[0], 0xFF112233);
        assert!(!t.dirty);
    }

    #[test]
    fn tile_cache_invalidate_rect_marks_overlap() {
        let layer = solid_layer(0, 0xFF223344, TILE_SIZE * 4, TILE_SIZE * 4, 0, 0.0, 0.0);
        let mut cache = TileCache::new();
        cache.refresh_from_layer(&layer);
        cache.invalidate_rect(Rect {
            x: 0,
            y: 0,
            w: TILE_SIZE as i32 * 2,
            h: TILE_SIZE as i32 * 2,
        });
        for ty in 0..2 {
            for tx in 0..2 {
                assert!(cache.tile(tx, ty).unwrap().dirty);
            }
        }
        assert!(!cache.tile(3, 3).unwrap().dirty);
    }

    /// `refresh_from_raw` must produce the exact same tile contents as
    /// `refresh_from_layer` — they're two APIs to the same operation.
    #[test]
    fn refresh_from_raw_matches_layer() {
        let w = TILE_SIZE * 2 + 50;
        let h = TILE_SIZE + 100;
        // Build a gradient-ish pattern so tiles aren't all one color.
        let pixels: Vec<u32> = (0..(w * h))
            .map(|i| 0xFF000000 | (i & 0xFFFFFF))
            .collect();
        let layer = Layer {
            id: 0,
            translate_x: 0.0,
            translate_y: 0.0,
            scale_x: 1.0,
            scale_y: 1.0,
            opacity: 1.0,
            clip_rect: None,
            bitmap: pixels.clone(),
            bitmap_w: w,
            bitmap_h: h,
            z_index: 0,
            tree_state: cv_paint::PropertyTreeState::default(),
        };
        let mut from_layer = TileCache::new();
        from_layer.refresh_from_layer(&layer);

        let mut from_raw = TileCache::new();
        from_raw.refresh_from_raw(&pixels, w, h);

        assert_eq!(from_layer.tile_count(), from_raw.tile_count());
        let cols = (w + TILE_SIZE - 1) / TILE_SIZE;
        let rows = (h + TILE_SIZE - 1) / TILE_SIZE;
        for ty in 0..rows {
            for tx in 0..cols {
                let a = from_layer.tile(tx, ty).unwrap();
                let b = from_raw.tile(tx, ty).unwrap();
                assert_eq!(a.pixels, b.pixels, "tile ({tx},{ty}) mismatch");
            }
        }
    }

    /// `composite_viewport` must reconstruct any rectangular sub-region
    /// of the bitmap pixel-perfectly. This proves the WM_PAINT path
    /// through tiles is lossless compared to raw StretchDIBits.
    #[test]
    fn composite_viewport_matches_raw_bitmap() {
        let w = TILE_SIZE * 3; // 768 pixels wide
        let h = TILE_SIZE * 4; // 1024 pixels tall (scrollable page)
        let pixels: Vec<u32> = (0..(w * h))
            .map(|i| 0xFF000000 | (i & 0xFFFFFF))
            .collect();

        let mut cache = TileCache::new();
        cache.refresh_from_raw(&pixels, w, h);

        // Viewport: 768×400, scroll_y=300 (spans tile row boundary)
        let vw = w;
        let vh = 400u32;
        let scroll_y = 300i32;

        let viewport = cache.composite_viewport(0, scroll_y, vw, vh);

        // Every pixel should match the corresponding pixel in the raw bitmap
        for vy in 0..vh {
            for vx in 0..vw {
                let bmp_y = scroll_y as u32 + vy;
                let bmp_x = vx;
                let expected = pixels[(bmp_y * w + bmp_x) as usize];
                let actual = viewport[(vy * vw + vx) as usize];
                assert_eq!(
                    actual, expected,
                    "pixel ({vx},{vy}) viewport vs ({bmp_x},{bmp_y}) bitmap"
                );
            }
        }
    }

    /// The direct (no-tile-cache) viewport slice must be byte-identical to the
    /// TileCache path for a freshly-refreshed full cache, across scroll offsets,
    /// partial-width, and beyond-bitmap cases.
    #[test]
    fn direct_viewport_matches_tilecache() {
        let w = TILE_SIZE * 3 + 17; // non-tile-aligned width
        let h = TILE_SIZE * 4 + 9;
        let pixels: Vec<u32> = (0..(w * h)).map(|i| 0xFF000000 | (i & 0xFFFFFF)).collect();
        let mut cache = TileCache::new();
        cache.refresh_from_raw(&pixels, w, h);
        // Cases: in-bounds mid-scroll, top, near-bottom (viewport extends past
        // the bitmap → white fill), and viewport entirely beyond the bitmap.
        for &(sy, vw, vh) in &[
            (0i32, w, 400u32),
            (300, w, 400),
            (h as i32 - 100, w, 400), // extends past bottom
            (h as i32 + 50, w, 200),  // entirely beyond
            (0, w - 30, 250),         // partial width
        ] {
            let tiled = cache.composite_viewport(0, sy, vw, vh);
            let direct = composite_viewport_direct(&pixels, w, h, 0, sy, vw, vh);
            assert_eq!(direct, tiled, "direct vs tilecache mismatch at sy={sy} vw={vw} vh={vh}");
        }
    }

    /// composite_viewport with scroll_y=0 and full page dimensions must
    /// reproduce the entire bitmap exactly.
    #[test]
    fn composite_viewport_full_page_no_scroll() {
        let w = 500u32;
        let h = 300u32;
        let pixels: Vec<u32> = (0..(w * h))
            .map(|i| 0xAA000000 | ((i * 7) & 0xFFFFFF))
            .collect();

        let mut cache = TileCache::new();
        cache.refresh_from_raw(&pixels, w, h);

        let viewport = cache.composite_viewport(0, 0, w, h);
        assert_eq!(viewport, pixels);
    }

    // ── apply_property_tree_update tests ───────────────

    #[test]
    fn apply_property_tree_update_moves_layer() {
        let mut trees = cv_paint::PropertyTrees::new();
        let child_tf = trees.push_transform(cv_paint::TransformNode {
            parent: Some(0),
            translate_x: 10.0,
            translate_y: 20.0,
            scale_x: 1.0,
            scale_y: 1.0,
        });
        let child_ef = trees.push_effect(cv_paint::EffectNode {
            parent: Some(0),
            opacity: 0.7,
        });

        let mut tree = LayerTree::new();
        tree.push(Layer {
            id: 0,
            translate_x: 0.0,
            translate_y: 0.0,
            scale_x: 1.0,
            scale_y: 1.0,
            opacity: 1.0,
            clip_rect: None,
            bitmap: vec![0xFF000000; 16],
            bitmap_w: 4,
            bitmap_h: 4,
            z_index: 0,
            tree_state: cv_paint::PropertyTreeState {
                transform_id: child_tf,
                effect_id: child_ef,
                clip_id: 0,
            },
        });

        let changed = tree.apply_property_tree_update(&trees);
        assert!(changed);
        assert!((tree.layers[0].translate_x - 10.0).abs() < 0.01);
        assert!((tree.layers[0].translate_y - 20.0).abs() < 0.01);
        assert!((tree.layers[0].opacity - 0.7).abs() < 0.01);
    }

    #[test]
    fn apply_property_tree_update_no_change_returns_false() {
        let trees = cv_paint::PropertyTrees::new();
        let mut tree = LayerTree::new();
        tree.push(solid_layer(0, 0xFF000000, 4, 4, 0, 0.0, 0.0));
        // tree_state defaults to (0,0,0) → root identity; layer is already at (0,0) opacity 1.0
        let changed = tree.apply_property_tree_update(&trees);
        assert!(!changed);
    }

    #[test]
    fn apply_property_tree_update_chain_compose() {
        let mut trees = cv_paint::PropertyTrees::new();
        let parent = trees.push_transform(cv_paint::TransformNode {
            parent: Some(0),
            translate_x: 100.0,
            translate_y: 0.0,
            scale_x: 2.0,
            scale_y: 1.0,
        });
        let child = trees.push_transform(cv_paint::TransformNode {
            parent: Some(parent),
            translate_x: 5.0,
            translate_y: 0.0,
            scale_x: 1.0,
            scale_y: 1.0,
        });

        let mut tree = LayerTree::new();
        tree.push(Layer {
            id: 0,
            translate_x: 0.0,
            translate_y: 0.0,
            scale_x: 1.0,
            scale_y: 1.0,
            opacity: 1.0,
            clip_rect: None,
            bitmap: vec![0xFF000000; 4],
            bitmap_w: 2,
            bitmap_h: 2,
            z_index: 0,
            tree_state: cv_paint::PropertyTreeState {
                transform_id: child,
                effect_id: 0,
                clip_id: 0,
            },
        });

        tree.apply_property_tree_update(&trees);
        // World = (100 + 2*5, 0) = (110, 0), scale (2,1)
        assert!((tree.layers[0].translate_x - 110.0).abs() < 0.01);
        assert!((tree.layers[0].scale_x - 2.0).abs() < 0.01);
    }

    /// Viewport that extends beyond the bitmap edge should get white
    /// (0xFFFFFFFF) padding — the same as the WM_PAINT white-rect pad.
    #[test]
    fn composite_viewport_beyond_bitmap_is_white() {
        let w = 100u32;
        let h = 100u32;
        let pixels = vec![0xFF112233u32; (w * h) as usize];

        let mut cache = TileCache::new();
        cache.refresh_from_raw(&pixels, w, h);

        // Scroll past the end of the bitmap
        let viewport = cache.composite_viewport(0, 80, w, 50);
        // First 20 rows should be bitmap color, last 30 should be white
        for vy in 0..20u32 {
            assert_eq!(viewport[(vy * w) as usize], 0xFF112233);
        }
        for vy in 20..50u32 {
            // Beyond bitmap — tiles don't exist, so we get the white fill
            assert_eq!(viewport[(vy * w) as usize], 0xFFFFFFFF);
        }
    }

    /// `blend_with_opacity` MUST match the `cv_gfx::blend_bgra` straight-alpha
    /// oracle byte-for-byte (the faint-particle fix). We cannot depend on
    /// `cv_gfx` here, so the expected values below were produced by the oracle
    /// (straight-alpha source-over: normalize RGB by `out_a`, `.round()`).
    /// A regression toward the old premultiplied/truncating form would darken
    /// the gold-over-transparent case toward 0 and fail this test.
    #[test]
    fn converged_blend_matches_straight_alpha_oracle() {
        // (1) Opaque over opaque: red (FFFF0000) over blue → pure red.
        assert_eq!(
            blend_with_opacity(0xFF0000FF, 0xFFFF0000, 1.0),
            0xFFFF0000,
            "opaque-over-opaque"
        );

        // (2) Faint gold rgba(255,215,0, a=26) over transparent. The
        // straight-alpha oracle KEEPS the gold RGB (255,215,0) — the buggy
        // premultiplied form dragged it toward 0 (the faint-particle bug).
        assert_eq!(
            blend_with_opacity(0x00000000, 0x1AFFD700, 1.0),
            0x1AFFD700,
            "faint-gold-over-transparent must not darken toward 0"
        );

        // (3) Semi over semi: white (a=128) over blue (a=128).
        assert_eq!(
            blend_with_opacity(0x800000FF, 0x80FFFFFF, 1.0),
            0xC0AAAAFF,
            "semi-over-semi"
        );

        // (4) Per-layer opacity < 1: opaque red over opaque blue at 0.5 opacity
        // → 50/50 magenta over fully-opaque output.
        assert_eq!(
            blend_with_opacity(0xFF0000FF, 0xFFFF0000, 0.5),
            0xFF800080,
            "opacity=0.5 folds into source alpha"
        );

        // (5) Early-out contract: a fully-transparent source leaves dst untouched.
        assert_eq!(
            blend_with_opacity(0x280A141E, 0x00FF0000, 1.0),
            0x280A141E,
            "zero effective source alpha returns dst unchanged"
        );
    }

    // ── M5.5 oracle layer (1): composite_viewport thread-independence ──
    //
    // The off-main compositor calls `composite_viewport` on the compositor
    // thread instead of the UI thread. `composite_viewport` borrows `&self`
    // only and touches NO thread_local — so it must be a pure, deterministic,
    // thread-placement-independent function. This GATE builds an identical
    // TileCache, computes `gold` on the test (UI-proxy) thread, computes the
    // same call on a spawned thread, and asserts BYTE-IDENTICAL output
    // (max per-pixel diff == 0). A regression that introduced any thread-local
    // or non-deterministic state would fail here.

    /// Deterministic pattern generator so the test thread and the spawned
    /// thread build byte-identical caches without sharing state.
    fn det_pattern(w: u32, h: u32, seed: u32) -> Vec<u32> {
        (0..(w * h))
            .map(|i| {
                let x = i % w;
                let y = i / w;
                // A non-trivial, fully deterministic mix (no RNG state).
                0xFF000000
                    | (((x.wrapping_mul(2654435761).wrapping_add(seed)) & 0xFF) << 16)
                    | (((y.wrapping_mul(40503).wrapping_add(seed)) & 0xFF) << 8)
                    | ((i.wrapping_mul(2246822519).wrapping_add(seed)) & 0xFF)
            })
            .collect()
    }

    fn build_cache(w: u32, h: u32, seed: u32) -> TileCache {
        let pixels = det_pattern(w, h, seed);
        let mut cache = TileCache::new();
        cache.refresh_from_raw(&pixels, w, h);
        cache
    }

    /// Run composite_viewport on the current thread vs a spawned thread for a
    /// single (scroll, dims) case and assert byte-identity.
    fn assert_thread_independent(w: u32, h: u32, seed: u32, scroll: i32, vw: u32, vh: u32) {
        let gold = build_cache(w, h, seed).composite_viewport(0, scroll, vw, vh);
        let off = std::thread::spawn(move || {
            build_cache(w, h, seed).composite_viewport(0, scroll, vw, vh)
        })
        .join()
        .expect("composite thread panicked");
        assert_eq!(
            gold.len(),
            off.len(),
            "viewport length differs across threads (w={w} h={h} scroll={scroll} vw={vw} vh={vh})"
        );
        // memcpy-based pure function — any nonzero diff is a bug.
        assert_eq!(
            gold, off,
            "composite_viewport diverged across threads (w={w} h={h} scroll={scroll} vw={vw} vh={vh})"
        );
    }

    #[test]
    fn composite_viewport_thread_independent_matrix() {
        let w = TILE_SIZE * 3; // 768
        let h = TILE_SIZE * 4; // 1024
        let viewport_h = 400u32;
        let max_scroll = (h - viewport_h) as i32;
        // scroll ∈ {0, mid, near-max, past-end}
        let scrolls = [0i32, max_scroll / 2, max_scroll - 1, (h as i32) + 50];
        // dims ∈ {exact, smaller-than-bitmap, wider-than-bitmap}
        let dims = [
            (w, viewport_h),            // exact width
            (w / 2, viewport_h),        // smaller than bitmap
            (w + TILE_SIZE, viewport_h), // wider than bitmap
        ];
        for &scroll in &scrolls {
            for &(vw, vh) in &dims {
                assert_thread_independent(w, h, 7, scroll, vw, vh);
            }
        }
    }

    #[test]
    fn composite_viewport_thread_independent_fuzz() {
        // A 50-case deterministic fuzz of patterns/scrolls/dims. The "RNG" is a
        // simple LCG so the case set is reproducible and shareable across the
        // two threads via the seed only (no shared mutable state).
        let mut state: u32 = 0x1234_5678;
        let mut next = || {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            state
        };
        for _ in 0..50 {
            let w = 64 + (next() % (TILE_SIZE * 3));
            let h = 64 + (next() % (TILE_SIZE * 4));
            let seed = next();
            let scroll = (next() % (h + 64)) as i32 - 16;
            let vw = 16 + (next() % (w + TILE_SIZE));
            let vh = 16 + (next() % (h + 64));
            assert_thread_independent(w, h, seed, scroll, vw, vh);
        }
    }
}
