//! Layer-tree builder from paint primitives.
//!
//! Sits between the paint pass (which emits flat lists of solid
//! rects, text spans, and images) and the compositor's `LayerTree`.
//! For each paint item we decide whether it inherits the current
//! "root" layer or promotes itself to its own compositor layer.
//!
//! Promotion criteria (matches the rough Chromium heuristics):
//!   - `position: fixed` or `position: sticky`
//!   - `will-change: transform`
//!   - non-trivial transform (translate ≠ 0)
//!   - `opacity < 1` mid-stack (so siblings underneath stay solid)
//!
//! V1 doesn't yet observe sticky scroll offsets — fixed and sticky
//! both pin to compositor-space (0,0). The browser-side painter
//! flags promoted items via `PaintItem::promoted = true`; everything
//! else flattens into the root layer.

use super::{Layer, LayerTree, Rect};

/// One paint primitive: a solid-color rect drawn at `rect` with
/// `color` (premultiplied BGRA). Additional primitive kinds (text,
/// image, gradient) are handled by the painter rasterizing into the
/// promoted layer's bitmap before reaching this builder.
#[derive(Debug, Clone, Copy)]
pub struct PaintRect {
    pub rect: Rect,
    pub color: u32,
    /// 0..1 opacity; 1.0 = fully opaque.
    pub opacity: f32,
    pub z_index: i32,
    /// Mark true when this rect should live on its own compositor layer.
    pub promoted: bool,
    /// Per-layer pixel translate when promoted.
    pub translate_x: f32,
    pub translate_y: f32,
}

/// Build a `LayerTree` from a slice of `PaintRect`. The root layer
/// (id=0) gets all non-promoted rects rasterized into a single BGRA
/// bitmap of size `canvas_w x canvas_h`. Each promoted rect becomes
/// its own `Layer` of exactly its rect size, with the per-layer
/// transform applied at composite time.
pub fn build_layer_tree(
    paint: &[PaintRect],
    canvas_w: u32,
    canvas_h: u32,
    background: u32,
) -> LayerTree {
    let mut tree = LayerTree::new();
    let mut root_bitmap = vec![background; (canvas_w * canvas_h) as usize];
    for p in paint.iter().filter(|p| !p.promoted) {
        rasterize_rect_into(&mut root_bitmap, canvas_w, canvas_h, p);
    }
    tree.push(Layer {
        id: 0,
        translate_x: 0.0,
        translate_y: 0.0,
        scale_x: 1.0,
        scale_y: 1.0,
        opacity: 1.0,
        clip_rect: None,
        bitmap: root_bitmap,
        bitmap_w: canvas_w,
        bitmap_h: canvas_h,
        z_index: i32::MIN, // ensure root paints first
        tree_state: cv_paint::PropertyTreeState::default(),
    });
    let mut next_id: u32 = 1;
    for p in paint.iter().filter(|p| p.promoted) {
        let w = p.rect.w.max(0) as u32;
        let h = p.rect.h.max(0) as u32;
        if w == 0 || h == 0 {
            continue;
        }
        let bitmap = vec![p.color; (w * h) as usize];
        tree.push(Layer {
            id: next_id,
            translate_x: p.translate_x + p.rect.x as f32,
            translate_y: p.translate_y + p.rect.y as f32,
            scale_x: 1.0,
            scale_y: 1.0,
            opacity: p.opacity,
            clip_rect: None,
            bitmap,
            bitmap_w: w,
            bitmap_h: h,
            z_index: p.z_index,
            tree_state: cv_paint::PropertyTreeState::default(),
        });
        next_id += 1;
    }
    tree.sort_by_z();
    tree
}

fn rasterize_rect_into(buf: &mut [u32], w: u32, h: u32, p: &PaintRect) {
    let x0 = p.rect.x.max(0) as u32;
    let y0 = p.rect.y.max(0) as u32;
    let x1 = ((p.rect.x + p.rect.w).max(0) as u32).min(w);
    let y1 = ((p.rect.y + p.rect.h).max(0) as u32).min(h);
    let opacity = p.opacity.clamp(0.0, 1.0);
    for y in y0..y1 {
        for x in x0..x1 {
            let idx = (y as usize) * (w as usize) + x as usize;
            buf[idx] = blend_with_opacity(buf[idx], p.color, opacity);
        }
    }
}

fn blend_with_opacity(dst: u32, src: u32, opacity: f32) -> u32 {
    let sa = ((src >> 24) & 0xFF) as f32 / 255.0 * opacity;
    if sa <= 0.0 {
        return dst;
    }
    let sr = ((src >> 16) & 0xFF) as f32 / 255.0;
    let sg = ((src >> 8) & 0xFF) as f32 / 255.0;
    let sb = (src & 0xFF) as f32 / 255.0;
    let da = ((dst >> 24) & 0xFF) as f32 / 255.0;
    let dr = ((dst >> 16) & 0xFF) as f32 / 255.0;
    let dg = ((dst >> 8) & 0xFF) as f32 / 255.0;
    let db = (dst & 0xFF) as f32 / 255.0;
    let out_a = sa + da * (1.0 - sa);
    let out_r = sr * sa + dr * da * (1.0 - sa);
    let out_g = sg * sa + dg * da * (1.0 - sa);
    let out_b = sb * sa + db * da * (1.0 - sa);
    let to_u8 = |c: f32| -> u32 { (c.clamp(0.0, 1.0) * 255.0) as u32 };
    (to_u8(out_a) << 24) | (to_u8(out_r) << 16) | (to_u8(out_g) << 8) | to_u8(out_b)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::composite_frame;

    fn pr(x: i32, y: i32, w: i32, h: i32, color: u32) -> PaintRect {
        PaintRect {
            rect: Rect { x, y, w, h },
            color,
            opacity: 1.0,
            z_index: 0,
            promoted: false,
            translate_x: 0.0,
            translate_y: 0.0,
        }
    }

    #[test]
    fn root_layer_takes_all_non_promoted_rects() {
        let paint = vec![pr(0, 0, 2, 2, 0xFFFF0000), pr(2, 0, 2, 2, 0xFF00FF00)];
        let tree = build_layer_tree(&paint, 4, 2, 0xFF000000);
        assert_eq!(tree.layers.len(), 1);
        let out = composite_frame(&tree, 4, 2, 0xFF000000);
        assert_eq!(out[0], 0xFFFF0000); // left half red
        assert_eq!(out[2], 0xFF00FF00); // right half green
    }

    #[test]
    fn promoted_rect_becomes_its_own_layer() {
        let mut p = pr(1, 1, 2, 2, 0xFF0000FF);
        p.promoted = true;
        p.z_index = 5;
        let tree = build_layer_tree(&[p], 4, 4, 0xFF000000);
        assert_eq!(tree.layers.len(), 2);
        // Promoted layer should translate to (1,1) and be 2x2.
        let promoted = &tree.layers[1];
        assert_eq!(promoted.translate_x, 1.0);
        assert_eq!(promoted.translate_y, 1.0);
        assert_eq!(promoted.bitmap_w, 2);
        assert_eq!(promoted.bitmap_h, 2);
        assert_eq!(promoted.z_index, 5);
        // Composited output: (1,1)..(2,2) blue, rest black.
        let out = composite_frame(&tree, 4, 4, 0xFF000000);
        assert_eq!(out[1 * 4 + 1], 0xFF0000FF);
        assert_eq!(out[0], 0xFF000000);
    }

    #[test]
    fn z_order_after_build_is_root_first() {
        let mut p = pr(0, 0, 1, 1, 0xFFFFFFFF);
        p.promoted = true;
        p.z_index = -5;
        let tree = build_layer_tree(&[p], 2, 2, 0);
        // Even with z_index=-5 on the promoted layer, root's z_index
        // is i32::MIN so root still paints first.
        assert_eq!(tree.layers[0].id, 0);
        assert_eq!(tree.layers[1].id, 1);
    }

    #[test]
    fn opacity_in_rasterizer_blends_with_background() {
        let mut p = pr(0, 0, 1, 1, 0xFFFFFFFF);
        p.opacity = 0.5;
        let tree = build_layer_tree(&[p], 1, 1, 0xFF000000);
        let out = composite_frame(&tree, 1, 1, 0xFF000000);
        let r = (out[0] >> 16) & 0xFF;
        // 50% white over black ≈ 127.
        assert!((100..=160).contains(&r));
    }
}
