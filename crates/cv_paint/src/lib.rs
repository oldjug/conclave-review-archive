//! `cv_paint` — display list, paint property tree.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PaintRect {
    pub x: i32,
    pub y: i32,
    pub w: u32,
    pub h: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Bgra(pub u32);

impl Bgra {
    pub fn new(r: u8, g: u8, b: u8, a: u8) -> Self {
        Self(((a as u32) << 24) | ((r as u32) << 16) | ((g as u32) << 8) | b as u32)
    }
    pub fn a(self) -> u8 {
        ((self.0 >> 24) & 0xFF) as u8
    }
    pub fn r(self) -> u8 {
        ((self.0 >> 16) & 0xFF) as u8
    }
    pub fn g(self) -> u8 {
        ((self.0 >> 8) & 0xFF) as u8
    }
    pub fn b(self) -> u8 {
        (self.0 & 0xFF) as u8
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum PaintItem {
    Fill {
        rect: PaintRect,
        color: Bgra,
    },
    LinearGradient {
        rect: PaintRect,
        from: Bgra,
        to: Bgra,
        angle_deg: f32,
    },
    Image {
        rect: PaintRect,
        pixels: Vec<u8>,
        width: u32,
        height: u32,
    },
    Text {
        x: i32,
        y: i32,
        text: String,
        size_px: f32,
        color: Bgra,
        bold: bool,
        italic: bool,
        underline: bool,
    },
    BoxShadow {
        rect: PaintRect,
        color: Bgra,
        offset_x: i32,
        offset_y: i32,
        /// Spread radius in pixels (positive = expand, negative = contract).
        spread: i32,
        /// `inset` keyword: shadow is drawn inside the rect, not outside.
        inset: bool,
    },
    Border {
        rect: PaintRect,
        top: u32,
        right: u32,
        bottom: u32,
        left: u32,
        color: Bgra,
    },
    PushClip(PaintRect),
    PopClip,
    PushTransform {
        translate_x: f32,
        translate_y: f32,
        scale_x: f32,
        scale_y: f32,
    },
    PopTransform,
    PushOpacity(f32),
    PopOpacity,
}

#[derive(Debug, Default, Clone)]
pub struct DisplayList {
    pub items: Vec<PaintItem>,
}

impl DisplayList {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn push(&mut self, item: PaintItem) {
        self.items.push(item);
    }
    pub fn len(&self) -> usize {
        self.items.len()
    }
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }
}

#[derive(Debug, Clone)]
pub struct PaintProperties {
    pub opacity: f32,
    pub translate_x: f32,
    pub translate_y: f32,
    pub scale_x: f32,
    pub scale_y: f32,
    pub clip: Option<PaintRect>,
}

impl Default for PaintProperties {
    fn default() -> Self {
        Self {
            opacity: 1.0,
            translate_x: 0.0,
            translate_y: 0.0,
            scale_x: 1.0,
            scale_y: 1.0,
            clip: None,
        }
    }
}

impl PaintProperties {
    pub fn write_push(&self, list: &mut DisplayList) -> u32 {
        let mut n = 0;
        if let Some(c) = self.clip {
            list.push(PaintItem::PushClip(c));
            n += 1;
        }
        if self.translate_x != 0.0
            || self.translate_y != 0.0
            || self.scale_x != 1.0
            || self.scale_y != 1.0
        {
            list.push(PaintItem::PushTransform {
                translate_x: self.translate_x,
                translate_y: self.translate_y,
                scale_x: self.scale_x,
                scale_y: self.scale_y,
            });
            n += 1;
        }
        if self.opacity < 1.0 {
            list.push(PaintItem::PushOpacity(self.opacity));
            n += 1;
        }
        n
    }
    pub fn write_pop(&self, list: &mut DisplayList) {
        if self.opacity < 1.0 {
            list.push(PaintItem::PopOpacity);
        }
        if self.translate_x != 0.0
            || self.translate_y != 0.0
            || self.scale_x != 1.0
            || self.scale_y != 1.0
        {
            list.push(PaintItem::PopTransform);
        }
        if self.clip.is_some() {
            list.push(PaintItem::PopClip);
        }
    }
}

pub fn apply_props_to_rect(rect: PaintRect, props: &PaintProperties) -> PaintRect {
    let new_x = (rect.x as f32 * props.scale_x + props.translate_x) as i32;
    let new_y = (rect.y as f32 * props.scale_y + props.translate_y) as i32;
    let new_w = (rect.w as f32 * props.scale_x) as u32;
    let new_h = (rect.h as f32 * props.scale_y) as u32;
    PaintRect {
        x: new_x,
        y: new_y,
        w: new_w,
        h: new_h,
    }
}

/// Walk a display list, dispatching each item via the callback.
/// Used by the rasterizer + the test harness.
pub fn walk_display_list<F: FnMut(&PaintItem)>(list: &DisplayList, mut f: F) {
    for item in &list.items {
        f(item);
    }
}

/// Hit-test: return the index of the topmost `Fill` whose rect
/// contains `(x, y)`, or `None`. Honors push/pop transforms.
pub fn hit_test(list: &DisplayList, x: i32, y: i32) -> Option<usize> {
    let mut tx = 0.0f32;
    let mut ty = 0.0f32;
    let mut sx = 1.0f32;
    let mut sy = 1.0f32;
    let mut tx_stack: Vec<(f32, f32, f32, f32)> = Vec::new();
    let mut topmost: Option<usize> = None;
    for (i, item) in list.items.iter().enumerate() {
        match item {
            PaintItem::PushTransform {
                translate_x,
                translate_y,
                scale_x,
                scale_y,
            } => {
                tx_stack.push((tx, ty, sx, sy));
                tx += *translate_x;
                ty += *translate_y;
                sx *= *scale_x;
                sy *= *scale_y;
            }
            PaintItem::PopTransform => {
                if let Some((a, b, c, d)) = tx_stack.pop() {
                    tx = a;
                    ty = b;
                    sx = c;
                    sy = d;
                }
            }
            PaintItem::Fill { rect, .. } => {
                let rx = (rect.x as f32 * sx + tx) as i32;
                let ry = (rect.y as f32 * sy + ty) as i32;
                let rw = (rect.w as f32 * sx) as i32;
                let rh = (rect.h as f32 * sy) as i32;
                if x >= rx && x < rx + rw && y >= ry && y < ry + rh {
                    topmost = Some(i);
                }
            }
            _ => {}
        }
    }
    topmost
}

// ── Property trees ─────────────────────────────────────────────
//
// Chromium's cc/ layer uses four "property trees" (transform, effect,
// clip, scroll) so the compositor can update opacity/transform
// without re-layout or re-paint. We implement three (transform,
// effect, clip) — scroll is handled separately by the tile cache's
// viewport offset.
//
// Each tree is a flat Vec of nodes; a node stores its local value and
// a parent index. World (accumulated) values walk the parent chain.
// The compositor maps each promoted layer to a (transform_id,
// effect_id, clip_id) triple — its *PropertyTreeState* — and
// resolves world values once per frame via the accumulator functions.

/// Node in the transform property tree.
#[derive(Debug, Clone, PartialEq)]
pub struct TransformNode {
    /// Index of the parent node, or `None` for the root.
    pub parent: Option<usize>,
    pub translate_x: f32,
    pub translate_y: f32,
    pub scale_x: f32,
    pub scale_y: f32,
}

impl Default for TransformNode {
    fn default() -> Self {
        Self {
            parent: None,
            translate_x: 0.0,
            translate_y: 0.0,
            scale_x: 1.0,
            scale_y: 1.0,
        }
    }
}

/// Node in the effect (opacity/filter) property tree.
#[derive(Debug, Clone, PartialEq)]
pub struct EffectNode {
    pub parent: Option<usize>,
    /// Local opacity contribution (0..1). The world opacity is the
    /// product of all ancestors' opacities.
    pub opacity: f32,
}

impl Default for EffectNode {
    fn default() -> Self {
        Self {
            parent: None,
            opacity: 1.0,
        }
    }
}

/// Node in the clip property tree.
#[derive(Debug, Clone, PartialEq)]
pub struct ClipNode {
    pub parent: Option<usize>,
    /// Local clip rect (in the node's coordinate space).
    pub clip_rect: PaintRect,
}

/// Per-element assignment into the three trees.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct PropertyTreeState {
    pub transform_id: usize,
    pub effect_id: usize,
    pub clip_id: usize,
}

/// The three property trees, built once per layout and reused across
/// compositor-only frames.
#[derive(Debug, Clone, Default)]
pub struct PropertyTrees {
    pub transforms: Vec<TransformNode>,
    pub effects: Vec<EffectNode>,
    pub clips: Vec<ClipNode>,
}

impl PropertyTrees {
    pub fn new() -> Self {
        // Start with a root node in each tree (identity transform,
        // fully opaque, no clip).
        Self {
            transforms: vec![TransformNode::default()],
            effects: vec![EffectNode::default()],
            clips: Vec::new(),
        }
    }

    /// Add a transform node and return its index.
    pub fn push_transform(&mut self, node: TransformNode) -> usize {
        let id = self.transforms.len();
        self.transforms.push(node);
        id
    }

    /// Add an effect node and return its index.
    pub fn push_effect(&mut self, node: EffectNode) -> usize {
        let id = self.effects.len();
        self.effects.push(node);
        id
    }

    /// Add a clip node and return its index.
    pub fn push_clip(&mut self, node: ClipNode) -> usize {
        let id = self.clips.len();
        self.clips.push(node);
        id
    }

    /// Accumulated world-space translate + scale for a transform node.
    /// Walks the parent chain, composing scale-then-translate at each
    /// level (same composition order as the existing `PushTransform`
    /// stack in the display-list rasterizer).
    pub fn world_transform(&self, node_id: usize) -> (f32, f32, f32, f32) {
        let mut tx = 0.0f32;
        let mut ty = 0.0f32;
        let mut sx = 1.0f32;
        let mut sy = 1.0f32;
        let mut cur = Some(node_id);
        while let Some(id) = cur {
            let n = &self.transforms[id];
            // Composition: child's local transform applies first, then
            // the parent's. In matrix terms: World = Parent · Local.
            // For scale-translate: tx' = n.tx + n.sx * old_tx, etc.
            tx = n.translate_x + n.scale_x * tx;
            ty = n.translate_y + n.scale_y * ty;
            sx = n.scale_x * sx;
            sy = n.scale_y * sy;
            cur = n.parent;
        }
        (tx, ty, sx, sy)
    }

    /// Accumulated world opacity (product of chain).
    pub fn world_opacity(&self, node_id: usize) -> f32 {
        let mut acc = 1.0f32;
        let mut cur = Some(node_id);
        while let Some(id) = cur {
            let n = &self.effects[id];
            acc *= n.opacity;
            cur = n.parent;
        }
        acc
    }

    /// Accumulated world clip rect (intersection of chain).
    pub fn world_clip(&self, node_id: usize) -> Option<PaintRect> {
        let mut acc: Option<PaintRect> = None;
        let mut cur = Some(node_id);
        while let Some(id) = cur {
            let n = &self.clips[id];
            acc = Some(match acc {
                None => n.clip_rect,
                Some(a) => intersect_rects(a, n.clip_rect),
            });
            cur = n.parent;
        }
        acc
    }

    /// Check whether two PropertyTrees differ only in compositor
    /// properties (transform translate/scale, effect opacity). Returns
    /// true when clip tree and rasterized content are unchanged — the
    /// compositor can just update layer transforms/opacities without
    /// re-rasterizing.
    pub fn is_compositor_only_change(&self, other: &PropertyTrees) -> bool {
        // Clip tree changes require re-paint (clip affects rasterization).
        if self.clips != other.clips {
            return false;
        }
        // Tree structure must match (same number of nodes, same parent
        // linkage). Only the VALUE of transform/opacity may differ.
        if self.transforms.len() != other.transforms.len() {
            return false;
        }
        for (a, b) in self.transforms.iter().zip(other.transforms.iter()) {
            if a.parent != b.parent {
                return false;
            }
        }
        if self.effects.len() != other.effects.len() {
            return false;
        }
        for (a, b) in self.effects.iter().zip(other.effects.iter()) {
            if a.parent != b.parent {
                return false;
            }
        }
        true
    }
}

/// Build property trees from a flat list of compositor-relevant properties.
///
/// The caller walks its layout box tree and feeds one entry per box that
/// has a non-identity transform, non-1.0 opacity, or an overflow:hidden
/// clip. Each entry carries a parent-index into the returned
/// `PropertyTrees` (0 = root identity). The returned trees are suitable
/// for caching and diffing via `is_compositor_only_change`.
///
/// This is the main entry point for the "build" half of the compositor-
/// only fast path. The "update" half is
/// `LayerTree::apply_property_tree_update` in `cv_compositor`.
pub struct PropertyTreeBuilder {
    trees: PropertyTrees,
}

impl PropertyTreeBuilder {
    pub fn new() -> Self {
        Self {
            trees: PropertyTrees::new(),
        }
    }

    /// Register a transform node. Returns the node ID to use as a
    /// parent for child boxes.
    pub fn push_transform(
        &mut self,
        parent_tf: usize,
        translate_x: f32,
        translate_y: f32,
        scale_x: f32,
        scale_y: f32,
    ) -> usize {
        self.trees.push_transform(TransformNode {
            parent: Some(parent_tf),
            translate_x,
            translate_y,
            scale_x,
            scale_y,
        })
    }

    /// Register an effect (opacity) node.
    pub fn push_effect(&mut self, parent_ef: usize, opacity: f32) -> usize {
        self.trees.push_effect(EffectNode {
            parent: Some(parent_ef),
            opacity,
        })
    }

    /// Register a clip node.
    pub fn push_clip(&mut self, parent_clip: Option<usize>, clip_rect: PaintRect) -> usize {
        self.trees.push_clip(ClipNode {
            parent: parent_clip,
            clip_rect,
        })
    }

    /// Consume the builder and return the finished trees.
    pub fn finish(self) -> PropertyTrees {
        self.trees
    }
}

/// Rectangle intersection (returns zero-sized rect if no overlap).
fn intersect_rects(a: PaintRect, b: PaintRect) -> PaintRect {
    let x = a.x.max(b.x);
    let y = a.y.max(b.y);
    let r = (a.x + a.w as i32).min(b.x + b.w as i32);
    let bot = (a.y + a.h as i32).min(b.y + b.h as i32);
    PaintRect {
        x,
        y,
        w: (r - x).max(0) as u32,
        h: (bot - y).max(0) as u32,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn paint_props_pushes_all_layers() {
        let mut list = DisplayList::new();
        let mut p = PaintProperties::default();
        p.opacity = 0.5;
        p.translate_x = 10.0;
        p.clip = Some(PaintRect {
            x: 0,
            y: 0,
            w: 100,
            h: 50,
        });
        assert_eq!(p.write_push(&mut list), 3);
    }

    #[test]
    fn pop_unwinds_in_reverse_order() {
        let mut list = DisplayList::new();
        let mut p = PaintProperties::default();
        p.opacity = 0.5;
        p.clip = Some(PaintRect {
            x: 0,
            y: 0,
            w: 10,
            h: 10,
        });
        p.write_push(&mut list);
        p.write_pop(&mut list);
        assert!(matches!(list.items.last().unwrap(), PaintItem::PopClip));
    }

    #[test]
    fn bgra_round_trips() {
        let c = Bgra::new(10, 20, 30, 255);
        assert_eq!((c.r(), c.g(), c.b(), c.a()), (10, 20, 30, 255));
    }

    #[test]
    fn apply_props_translates_and_scales() {
        let p = PaintProperties {
            translate_x: 5.0,
            translate_y: 10.0,
            scale_x: 2.0,
            scale_y: 0.5,
            ..Default::default()
        };
        let r = apply_props_to_rect(
            PaintRect {
                x: 1,
                y: 1,
                w: 10,
                h: 20,
            },
            &p,
        );
        assert_eq!(r.x, 7);
        assert_eq!(r.y, 10);
        assert_eq!(r.w, 20);
        assert_eq!(r.h, 10);
    }

    #[test]
    fn hit_test_picks_topmost_fill() {
        let mut list = DisplayList::new();
        list.push(PaintItem::Fill {
            rect: PaintRect {
                x: 0,
                y: 0,
                w: 100,
                h: 100,
            },
            color: Bgra::new(255, 0, 0, 255),
        });
        list.push(PaintItem::Fill {
            rect: PaintRect {
                x: 10,
                y: 10,
                w: 20,
                h: 20,
            },
            color: Bgra::new(0, 255, 0, 255),
        });
        // Click at (15, 15) hits both; topmost is index 1.
        assert_eq!(hit_test(&list, 15, 15), Some(1));
        // Click at (50, 50) hits only the first.
        assert_eq!(hit_test(&list, 50, 50), Some(0));
        // Click outside all hits nothing.
        assert!(hit_test(&list, 500, 500).is_none());
    }

    // ── Property tree tests ──────────────────────────────────

    #[test]
    fn property_trees_root_identity() {
        let trees = PropertyTrees::new();
        let (tx, ty, sx, sy) = trees.world_transform(0);
        assert_eq!((tx, ty, sx, sy), (0.0, 0.0, 1.0, 1.0));
        assert_eq!(trees.world_opacity(0), 1.0);
    }

    #[test]
    fn world_transform_single_translate() {
        let mut trees = PropertyTrees::new();
        let child = trees.push_transform(TransformNode {
            parent: Some(0),
            translate_x: 50.0,
            translate_y: 100.0,
            scale_x: 1.0,
            scale_y: 1.0,
        });
        let (tx, ty, sx, sy) = trees.world_transform(child);
        assert_eq!((tx, ty, sx, sy), (50.0, 100.0, 1.0, 1.0));
    }

    #[test]
    fn world_transform_chain_compose() {
        // Parent: translate(10,20) scale(2,3)
        // Child:  translate(5,7) scale(1,1)
        // World = Parent · Child:
        //   tx = 10 + 2*5 = 20,  ty = 20 + 3*7 = 41
        //   sx = 2*1 = 2,         sy = 3*1 = 3
        let mut trees = PropertyTrees::new();
        let parent = trees.push_transform(TransformNode {
            parent: Some(0),
            translate_x: 10.0,
            translate_y: 20.0,
            scale_x: 2.0,
            scale_y: 3.0,
        });
        let child = trees.push_transform(TransformNode {
            parent: Some(parent),
            translate_x: 5.0,
            translate_y: 7.0,
            scale_x: 1.0,
            scale_y: 1.0,
        });
        let (tx, ty, sx, sy) = trees.world_transform(child);
        assert_eq!((tx, ty), (20.0, 41.0));
        assert_eq!((sx, sy), (2.0, 3.0));
    }

    #[test]
    fn world_opacity_chain() {
        let mut trees = PropertyTrees::new();
        let a = trees.push_effect(EffectNode {
            parent: Some(0),
            opacity: 0.5,
        });
        let b = trees.push_effect(EffectNode {
            parent: Some(a),
            opacity: 0.4,
        });
        assert!((trees.world_opacity(b) - 0.2).abs() < 1e-6);
    }

    #[test]
    fn world_clip_intersection() {
        let mut trees = PropertyTrees::new();
        let a = trees.push_clip(ClipNode {
            parent: None,
            clip_rect: PaintRect {
                x: 0,
                y: 0,
                w: 100,
                h: 100,
            },
        });
        let b = trees.push_clip(ClipNode {
            parent: Some(a),
            clip_rect: PaintRect {
                x: 50,
                y: 50,
                w: 100,
                h: 100,
            },
        });
        let clip = trees.world_clip(b).unwrap();
        assert_eq!((clip.x, clip.y, clip.w, clip.h), (50, 50, 50, 50));
    }

    #[test]
    fn world_clip_no_overlap() {
        let mut trees = PropertyTrees::new();
        let a = trees.push_clip(ClipNode {
            parent: None,
            clip_rect: PaintRect {
                x: 0,
                y: 0,
                w: 10,
                h: 10,
            },
        });
        let b = trees.push_clip(ClipNode {
            parent: Some(a),
            clip_rect: PaintRect {
                x: 20,
                y: 20,
                w: 10,
                h: 10,
            },
        });
        let clip = trees.world_clip(b).unwrap();
        assert_eq!(clip.w, 0);
        assert_eq!(clip.h, 0);
    }

    #[test]
    fn compositor_only_change_detects_transform_value_diff() {
        let mut a = PropertyTrees::new();
        a.push_transform(TransformNode {
            parent: Some(0),
            translate_x: 0.0,
            translate_y: 0.0,
            scale_x: 1.0,
            scale_y: 1.0,
        });
        let mut b = PropertyTrees::new();
        b.push_transform(TransformNode {
            parent: Some(0),
            translate_x: 100.0, // value changed but structure same
            translate_y: 0.0,
            scale_x: 1.0,
            scale_y: 1.0,
        });
        assert!(a.is_compositor_only_change(&b));
    }

    #[test]
    fn compositor_only_change_rejects_clip_diff() {
        let mut a = PropertyTrees::new();
        a.push_clip(ClipNode {
            parent: None,
            clip_rect: PaintRect {
                x: 0,
                y: 0,
                w: 100,
                h: 100,
            },
        });
        let mut b = PropertyTrees::new();
        b.push_clip(ClipNode {
            parent: None,
            clip_rect: PaintRect {
                x: 0,
                y: 0,
                w: 200,
                h: 200,
            },
        });
        assert!(!a.is_compositor_only_change(&b));
    }

    #[test]
    fn compositor_only_change_rejects_structural_diff() {
        let a = PropertyTrees::new();
        let mut b = PropertyTrees::new();
        b.push_transform(TransformNode {
            parent: Some(0),
            translate_x: 0.0,
            translate_y: 0.0,
            scale_x: 1.0,
            scale_y: 1.0,
        });
        // b has an extra node → structure differs
        assert!(!a.is_compositor_only_change(&b));
    }

    #[test]
    fn hit_test_honors_transform_stack() {
        let mut list = DisplayList::new();
        list.push(PaintItem::PushTransform {
            translate_x: 100.0,
            translate_y: 0.0,
            scale_x: 1.0,
            scale_y: 1.0,
        });
        list.push(PaintItem::Fill {
            rect: PaintRect {
                x: 0,
                y: 0,
                w: 10,
                h: 10,
            },
            color: Bgra::new(255, 0, 0, 255),
        });
        list.push(PaintItem::PopTransform);
        // The fill is at logical (0,0..10,10) but visually at (100..110, 0..10).
        assert!(hit_test(&list, 5, 5).is_none());
        assert_eq!(hit_test(&list, 105, 5), Some(1));
    }

    // ── PropertyTreeBuilder tests ──────────────────────

    #[test]
    fn builder_produces_correct_world_transform() {
        let mut b = PropertyTreeBuilder::new();
        // Root = identity at index 0.
        let child = b.push_transform(0, 10.0, 20.0, 2.0, 1.0);
        let grandchild = b.push_transform(child, 5.0, 0.0, 1.0, 1.0);
        let trees = b.finish();
        // grandchild world = (10 + 2*5, 20 + 1*0) = (20, 20), scale (2, 1)
        let (tx, ty, sx, sy) = trees.world_transform(grandchild);
        assert_eq!((tx, ty), (20.0, 20.0));
        assert_eq!((sx, sy), (2.0, 1.0));
    }

    #[test]
    fn builder_produces_correct_world_opacity() {
        let mut b = PropertyTreeBuilder::new();
        let a = b.push_effect(0, 0.5);
        let c = b.push_effect(a, 0.8);
        let trees = b.finish();
        assert!((trees.world_opacity(c) - 0.4).abs() < 1e-6);
    }

    #[test]
    fn builder_clips_intersect() {
        let mut b = PropertyTreeBuilder::new();
        let a = b.push_clip(
            None,
            PaintRect {
                x: 0,
                y: 0,
                w: 200,
                h: 200,
            },
        );
        let c = b.push_clip(
            Some(a),
            PaintRect {
                x: 100,
                y: 50,
                w: 200,
                h: 200,
            },
        );
        let trees = b.finish();
        let clip = trees.world_clip(c).unwrap();
        assert_eq!((clip.x, clip.y, clip.w, clip.h), (100, 50, 100, 150));
    }

    #[test]
    fn builder_compositor_only_change() {
        // Build two trees with same structure but different transform values.
        let mut ba = PropertyTreeBuilder::new();
        ba.push_transform(0, 0.0, 0.0, 1.0, 1.0);
        ba.push_effect(0, 0.5);
        let a = ba.finish();

        let mut bb = PropertyTreeBuilder::new();
        bb.push_transform(0, 100.0, 0.0, 2.0, 2.0);
        bb.push_effect(0, 0.9);
        let b = bb.finish();

        assert!(a.is_compositor_only_change(&b));
    }
}
