//! Compositor layer promotion + property-tree-driven recomposite.
//!
//! This module wires the (previously orphaned) [`LayerTree::apply_property_tree_update`]
//! into a real frame path: it promotes the stacking contexts that Chrome's
//! `cc` would composite (transform/opacity animations, `will-change`,
//! `position:fixed`) into their own [`Layer`]s with a CACHED raster, then —
//! when a frame only changes `transform`/`opacity` — re-composites those
//! cached layers by mutating the property-tree nodes instead of re-running
//! raster.
//!
//! ## Why this is the Chrome-shaped path
//!
//! In `cc` (Chromium compositor), an element with a transform/opacity
//! animation, `will-change:transform|opacity`, or `position:fixed` gets its
//! own composited layer with a backing texture (`cc::PictureLayerImpl`'s
//! tiles). A transform-only or opacity-only animation runs on the compositor
//! thread by mutating the layer's property-tree node
//! (`TransformTree::OnTransformAnimated` / `EffectTree::OnOpacityAnimated`) —
//! the rasterized tiles are NOT regenerated; only the per-frame composite
//! (the textured-quad draw) re-runs.
//! Refs:
//!   - `cc/trees/property_tree.h`  (TransformTree::OnTransformAnimated,
//!     EffectTree::OnOpacityAnimated, EffectiveOpacity)
//!   - `cc/trees/layer_tree_host_common.cc` (CalculateDrawProperties)
//!   - `blink/.../compositing/CompositingReasonFinder` (compositing triggers)
//!
//! ## The no-re-raster proof
//!
//! Each promoted layer carries a `raster_gen` counter that is incremented
//! **only** when its backing bitmap is (re)written. [`CompositorFrame::recomposite`]
//! drives `apply_property_tree_update` and composites from the cached bitmaps;
//! it NEVER touches `raster_gen`. Tests assert the counter is unchanged across
//! transform-only / opacity-only frames (real "skip work Chrome can't"
//! verification) and that a content change DOES bump it.

use crate::{Layer, LayerTree, composite_frame};

/// Why an element was promoted to its own compositor layer. Mirrors the
/// subset of Blink's `CompositingReasons` we currently detect. The presence
/// of ANY reason is what promotes; the specific reason is kept for diagnostics
/// and for deciding which property-tree node animates.
///
/// Reference: `third_party/blink/renderer/platform/graphics/compositing_reasons.h`
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CompositingReasons {
    /// `transform`/`translate`/`scale`/`rotate` is animating (CSS animation or
    /// transition). Blink: `kActiveTransformAnimation`.
    pub transform_animation: bool,
    /// `opacity` is animating. Blink: `kActiveOpacityAnimation`.
    pub opacity_animation: bool,
    /// `will-change: transform` (or `opacity`/`top`/`left`). Blink:
    /// `kWillChangeTransform` / `kWillChangeOpacity`.
    pub will_change: bool,
    /// `position: fixed` (pinned to the viewport). Blink: `kFixedPosition`.
    pub fixed_position: bool,
}

impl CompositingReasons {
    pub const NONE: Self = Self {
        transform_animation: false,
        opacity_animation: false,
        will_change: false,
        fixed_position: false,
    };

    /// Whether ANY compositing trigger fired — i.e. whether Chrome would give
    /// this element its own composited layer.
    #[inline]
    pub fn should_promote(self) -> bool {
        self.transform_animation
            || self.opacity_animation
            || self.will_change
            || self.fixed_position
    }

    /// Whether the promotion is for a property that the compositor can animate
    /// WITHOUT re-raster (transform / opacity). `position:fixed` alone promotes
    /// but does not by itself imply a compositor-only animation; a transform or
    /// opacity animation does.
    #[inline]
    pub fn is_compositor_animatable(self) -> bool {
        self.transform_animation || self.opacity_animation
    }
}

/// A single promoted element handed to the compositor: its already-rasterized
/// contents (`bitmap`), where it lives, and which property-tree nodes drive it.
/// This is the bridge type the browser fills from its layout/paint pass — the
/// browser rasterizes the element's subtree ONCE into `bitmap`, then the
/// compositor reuses those pixels across every transform/opacity-only frame.
#[derive(Debug, Clone)]
pub struct PromotedElement {
    /// Stable per-element id (the layout/DOM `node_id`), so the same element
    /// maps to the same layer across frames.
    pub id: u32,
    /// Rasterized contents in row-major BGRA u32.
    pub bitmap: Vec<u32>,
    pub bitmap_w: u32,
    pub bitmap_h: u32,
    /// Document-space top-left of the element's backing store, BEFORE any
    /// animated transform. The animated transform is resolved from the
    /// property trees at composite time and added to this base offset.
    pub base_x: f32,
    pub base_y: f32,
    pub z_index: i32,
    /// Property-tree assignment (transform/effect/clip node ids).
    pub tree_state: cv_paint::PropertyTreeState,
    /// Why it was promoted.
    pub reasons: CompositingReasons,
}

/// A composited frame: the root (everything NOT promoted, rasterized into one
/// backing bitmap) plus the promoted layers. The frame OWNS the cached raster
/// for every layer, and a per-layer `raster_gen` that proves whether a frame
/// re-rastered. The compositor recomposites this frame on transform/opacity-only
/// changes by mutating the property trees.
#[derive(Debug)]
pub struct CompositorFrame {
    /// Output dimensions (viewport in compositor space).
    pub width: u32,
    pub height: u32,
    /// Solid background drawn behind every layer.
    pub background: u32,
    /// The flat layer tree (root + promoted), sorted by z.
    pub tree: LayerTree,
    /// Per-layer raster generation, indexed parallel to `tree.layers`.
    /// Incremented ONLY when the layer's bitmap is (re)written. The compositor
    /// recomposite path must never touch this — that invariant is what makes
    /// "no re-raster" a checkable property.
    raster_gen: Vec<u64>,
}

impl CompositorFrame {
    /// Build a composited frame from a fully-rasterized root bitmap + the set
    /// of promoted elements. The root layer holds everything that did NOT
    /// promote (so siblings under a translucent/animated layer stay solid);
    /// each [`PromotedElement`] becomes its own [`Layer`] whose bitmap is the
    /// element's cached raster.
    pub fn build(
        width: u32,
        height: u32,
        background: u32,
        root_bitmap: Vec<u32>,
        promoted: Vec<PromotedElement>,
    ) -> Self {
        let mut tree = LayerTree::new();
        tree.push(Layer {
            id: 0,
            translate_x: 0.0,
            translate_y: 0.0,
            scale_x: 1.0,
            scale_y: 1.0,
            opacity: 1.0,
            clip_rect: None,
            bitmap: root_bitmap,
            bitmap_w: width,
            bitmap_h: height,
            z_index: i32::MIN, // root paints first
            tree_state: cv_paint::PropertyTreeState::default(),
        });
        for p in promoted {
            tree.push(Layer {
                // The promoted layer is positioned at its base doc offset; the
                // animated transform is applied by the property-tree update.
                id: p.id,
                translate_x: p.base_x,
                translate_y: p.base_y,
                scale_x: 1.0,
                scale_y: 1.0,
                opacity: 1.0,
                clip_rect: None,
                bitmap: p.bitmap,
                bitmap_w: p.bitmap_w,
                bitmap_h: p.bitmap_h,
                z_index: p.z_index,
                tree_state: p.tree_state,
            });
        }
        tree.sort_by_z();
        let raster_gen = vec![1u64; tree.layers.len()]; // each layer rastered once at build
        Self {
            width,
            height,
            background,
            tree,
            raster_gen,
        }
    }

    /// Total times any layer has been rastered (sum of all `raster_gen`). Tests
    /// assert this is unchanged across a transform/opacity-only recomposite.
    pub fn total_raster_gen(&self) -> u64 {
        self.raster_gen.iter().sum()
    }

    /// Per-layer raster generation by layer id (for fine-grained assertions).
    pub fn raster_gen_of(&self, layer_id: u32) -> Option<u64> {
        self.tree
            .layers
            .iter()
            .position(|l| l.id == layer_id)
            .map(|i| self.raster_gen[i])
    }

    /// Drive a property-tree update across all promoted layers WITHOUT
    /// re-rastering, then composite from the cached bitmaps. This is the
    /// compositor-thread fast path: it resolves each layer's animated
    /// transform/opacity from `trees` (mutating the layer's composite-time
    /// fields via [`LayerTree::apply_property_tree_update`]) and produces the
    /// frame purely from cached pixels.
    ///
    /// `base_offsets` gives the doc-space anchor of each promoted layer (by id);
    /// the property-tree world transform is added to the anchor so a `translate`
    /// animation moves the layer relative to where it was laid out (Chrome adds
    /// the layer's paint offset to its transform-tree screen-space transform).
    ///
    /// Returns the composited BGRA buffer. Crucially, `raster_gen` is NOT
    /// touched — proving the contents were reused, not regenerated.
    pub fn recomposite(
        &mut self,
        trees: &cv_paint::PropertyTrees,
        base_offsets: &[(u32, f32, f32)],
    ) -> Vec<u32> {
        // 1. Update each layer's composite-time transform/opacity/clip from the
        //    property trees. apply_property_tree_update writes WORLD-space values
        //    (relative to root identity) into translate/scale/opacity/clip.
        self.tree.apply_property_tree_update(trees);
        // 2. Re-anchor each promoted layer: apply_property_tree_update set
        //    translate to the pure property-tree world transform; add back the
        //    element's doc-space base offset so the layer composites where it
        //    was laid out, displaced by the animated transform.
        for layer in self.tree.layers.iter_mut() {
            if layer.id == 0 {
                continue; // root never moves
            }
            if let Some(&(_, bx, by)) = base_offsets.iter().find(|(id, _, _)| *id == layer.id) {
                layer.translate_x += bx;
                layer.translate_y += by;
            }
        }
        // 3. Composite from cached bitmaps. composite_frame reads each layer's
        //    bitmap as-is — no raster.
        composite_frame(&self.tree, self.width, self.height, self.background)
    }

    /// Re-rasterize a single promoted layer's contents (a genuine content
    /// change — text edit, paint property other than transform/opacity). This
    /// is the SLOW path; it bumps `raster_gen` for that layer. Provided so the
    /// "content change DOES re-raster" test exercises a real code path rather
    /// than a synthetic counter bump.
    pub fn raster_layer(&mut self, layer_id: u32, new_bitmap: Vec<u32>, w: u32, h: u32) -> bool {
        if let Some(idx) = self.tree.layers.iter().position(|l| l.id == layer_id) {
            let l = &mut self.tree.layers[idx];
            l.bitmap = new_bitmap;
            l.bitmap_w = w;
            l.bitmap_h = h;
            self.raster_gen[idx] += 1;
            true
        } else {
            false
        }
    }

    /// Composite the current frame from cached bitmaps with NO property update
    /// and NO raster (used as the baseline / first-frame present).
    pub fn composite(&self) -> Vec<u32> {
        composite_frame(&self.tree, self.width, self.height, self.background)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A solid w×h element bitmap.
    fn solid(color: u32, w: u32, h: u32) -> Vec<u32> {
        vec![color; (w * h) as usize]
    }

    fn promoted_translate_layer(id: u32, color: u32, w: u32, h: u32) -> PromotedElement {
        PromotedElement {
            id,
            bitmap: solid(color, w, h),
            bitmap_w: w,
            bitmap_h: h,
            base_x: 0.0,
            base_y: 0.0,
            z_index: 1,
            tree_state: cv_paint::PropertyTreeState::default(),
            reasons: CompositingReasons {
                transform_animation: true,
                ..CompositingReasons::NONE
            },
        }
    }

    #[test]
    fn compositing_reasons_promote_only_on_trigger() {
        assert!(!CompositingReasons::NONE.should_promote());
        let mut r = CompositingReasons::NONE;
        r.transform_animation = true;
        assert!(r.should_promote() && r.is_compositor_animatable());
        let mut r = CompositingReasons::NONE;
        r.fixed_position = true;
        assert!(r.should_promote() && !r.is_compositor_animatable());
        let mut r = CompositingReasons::NONE;
        r.opacity_animation = true;
        assert!(r.is_compositor_animatable());
    }

    /// THE headline test: an element animating ONLY transform updates its layer
    /// transform via apply_property_tree_update and its content raster is NOT
    /// re-run across frames.
    #[test]
    fn transform_only_animation_does_not_reraster() {
        // 4x4 white viewport, one 2x2 red promoted element at (0,0).
        let root = solid(0xFF000000, 4, 4);
        let elem = promoted_translate_layer(7, 0xFFFF0000, 2, 2);
        let mut frame = CompositorFrame::build(4, 4, 0xFF000000, root, vec![elem]);

        let baseline_raster = frame.total_raster_gen();
        let elem_baseline = frame.raster_gen_of(7).unwrap();

        // Frame A: transform = translate(0,0). Layer sits at (0,0).
        let mut trees_a = cv_paint::PropertyTrees::new();
        let tf_a = trees_a.push_transform(cv_paint::TransformNode {
            parent: Some(0),
            translate_x: 0.0,
            translate_y: 0.0,
            scale_x: 1.0,
            scale_y: 1.0,
        });
        // Point the layer's tree_state at this animated transform node.
        if let Some(l) = frame.tree.layers.iter_mut().find(|l| l.id == 7) {
            l.tree_state = cv_paint::PropertyTreeState {
                transform_id: tf_a,
                effect_id: 0,
                clip_id: 0,
            };
        }
        let base = [(7u32, 0.0f32, 0.0f32)];
        let out_a = frame.recomposite(&trees_a, &base);
        // Red at (0,0).
        assert_eq!(out_a[0], 0xFFFF0000, "frame A: red at origin");
        assert_eq!(out_a[2 * 4 + 2], 0xFF000000, "frame A: (2,2) is background");

        // Frame B: transform = translate(2,2). Same tree node id, new VALUE
        // (Chrome's OnTransformAnimated mutates the node in place).
        let mut trees_b = cv_paint::PropertyTrees::new();
        let _tf_b = trees_b.push_transform(cv_paint::TransformNode {
            parent: Some(0),
            translate_x: 2.0,
            translate_y: 2.0,
            scale_x: 1.0,
            scale_y: 1.0,
        });
        let out_b = frame.recomposite(&trees_b, &base);
        // Red has MOVED to (2,2); origin is now background.
        assert_eq!(out_b[2 * 4 + 2], 0xFFFF0000, "frame B: red moved to (2,2)");
        assert_eq!(out_b[0], 0xFF000000, "frame B: origin now background");

        // ★ The proof: raster generation is UNCHANGED across both frames.
        assert_eq!(
            frame.total_raster_gen(),
            baseline_raster,
            "transform-only recomposite must not re-raster ANY layer"
        );
        assert_eq!(
            frame.raster_gen_of(7).unwrap(),
            elem_baseline,
            "the animated element's content was reused, not re-rastered"
        );
    }

    /// Opacity-only animation: same no-re-raster guarantee; the layer's opacity
    /// changes and the composite blends accordingly, but contents are reused.
    #[test]
    fn opacity_only_animation_does_not_reraster() {
        let root = solid(0xFF000000, 2, 2); // black background
        let mut elem = PromotedElement {
            id: 9,
            bitmap: solid(0xFFFFFFFF, 2, 2), // white element
            bitmap_w: 2,
            bitmap_h: 2,
            base_x: 0.0,
            base_y: 0.0,
            z_index: 1,
            tree_state: cv_paint::PropertyTreeState::default(),
            reasons: CompositingReasons {
                opacity_animation: true,
                ..CompositingReasons::NONE
            },
        };
        elem.reasons.opacity_animation = true;
        let mut frame = CompositorFrame::build(2, 2, 0xFF000000, root, vec![elem]);
        let baseline_raster = frame.total_raster_gen();

        let base = [(9u32, 0.0f32, 0.0f32)];

        // Frame A: opacity 1.0 → pure white.
        let mut trees_a = cv_paint::PropertyTrees::new();
        let ef_a = trees_a.push_effect(cv_paint::EffectNode {
            parent: Some(0),
            opacity: 1.0,
        });
        if let Some(l) = frame.tree.layers.iter_mut().find(|l| l.id == 9) {
            l.tree_state.effect_id = ef_a;
        }
        let out_a = frame.recomposite(&trees_a, &base);
        let r_a = (out_a[0] >> 16) & 0xFF;
        assert_eq!(r_a, 255, "frame A: opaque white");

        // Frame B: opacity 0.5 → grey (white over black at 50%).
        let mut trees_b = cv_paint::PropertyTrees::new();
        let _ef_b = trees_b.push_effect(cv_paint::EffectNode {
            parent: Some(0),
            opacity: 0.5,
        });
        let out_b = frame.recomposite(&trees_b, &base);
        let r_b = (out_b[0] >> 16) & 0xFF;
        assert!(
            (100..=160).contains(&r_b),
            "frame B: 50% white over black ≈ 127, got {r_b}"
        );

        // ★ No re-raster across the opacity animation.
        assert_eq!(
            frame.total_raster_gen(),
            baseline_raster,
            "opacity-only recomposite must not re-raster"
        );
    }

    /// A genuine CONTENT change DOES re-raster the affected layer (the slow
    /// path), bumping its raster generation — the negative control proving the
    /// counter is real, not pinned.
    #[test]
    fn content_change_does_reraster() {
        let root = solid(0xFF000000, 4, 4);
        let elem = promoted_translate_layer(3, 0xFFFF0000, 2, 2);
        let mut frame = CompositorFrame::build(4, 4, 0xFF000000, root, vec![elem]);
        let before = frame.raster_gen_of(3).unwrap();

        // Element's text/content changed → re-raster its layer (new green bitmap).
        let changed = frame.raster_layer(3, solid(0xFF00FF00, 2, 2), 2, 2);
        assert!(changed);
        let after = frame.raster_gen_of(3).unwrap();
        assert_eq!(after, before + 1, "content change must bump raster gen");

        // And the composited output reflects the new pixels.
        let trees = cv_paint::PropertyTrees::new();
        let out = frame.composite();
        assert_eq!(out[0], 0xFF00FF00, "re-rastered green now composites");
        let _ = trees;
    }

    /// Recomposite is deterministic + pure: composing the SAME trees twice
    /// yields byte-identical output and the SAME raster generation.
    #[test]
    fn recomposite_is_pure_and_stable() {
        let root = solid(0xFF101010, 8, 8);
        let elem = promoted_translate_layer(5, 0xFF00FFFF, 3, 3);
        let mut frame = CompositorFrame::build(8, 8, 0xFF101010, root, vec![elem]);
        let mut trees = cv_paint::PropertyTrees::new();
        let _tf = trees.push_transform(cv_paint::TransformNode {
            parent: Some(0),
            translate_x: 1.0,
            translate_y: 1.0,
            scale_x: 1.0,
            scale_y: 1.0,
        });
        let base = [(5u32, 0.0f32, 0.0f32)];
        let g0 = frame.total_raster_gen();
        let a = frame.recomposite(&trees, &base);
        let b = frame.recomposite(&trees, &base);
        assert_eq!(a, b, "recomposite must be byte-identical for identical input");
        assert_eq!(frame.total_raster_gen(), g0, "stable recomposite never rasters");
    }
}
