//! `cv_layout` — block layout (V1).
//!
//! Walks a styled DOM tree and produces a tree of `LayoutBox`es with
//! resolved `(x, y, width, height)` coordinates. Today we implement a
//! pure block layout: every element is a block stacked vertically inside
//! its parent. Inline formatting context lands in M2.
//!
//! Box model: each box has content, padding, and margin (we skip border
//! for V1 since we don't render borders yet).

use core::fmt;

#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct Rect {
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
}

/// Resolved 2-color linear gradient that fills a box's background.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct LinearGradientSpec {
    pub from: Color,
    pub to: Color,
    /// CSS-convention angle (0 = up, 90 = right, 180 = down, 270 = left).
    pub angle_deg: f32,
}

/// One color stop of a full N-stop gradient carried to the painter.
/// Mirrors `cv_css::cascade::GradientColorStop` (this crate stays free
/// of the CSS-crate dependency).
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct GradientStopSpec {
    pub color: Color,
    /// Position as a fraction of the gradient extent (0..1), or `None`
    /// if unspecified (distributed at paint per CSS Images 3 §3.4.3).
    pub pos_frac: Option<f32>,
    /// Absolute px position (resolved against the gradient extent at
    /// paint), or `None`.
    pub pos_px: Option<f32>,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum GradRadialShape {
    Circle,
    Ellipse,
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum GradRadialSize {
    ClosestSide,
    FarthestSide,
    ClosestCorner,
    FarthestCorner,
    /// Explicit radii — px and/or percentage-of-box per axis.
    Explicit {
        rx_px: Option<f32>,
        ry_px: Option<f32>,
        rx_pct: Option<f32>,
        ry_pct: Option<f32>,
    },
}

/// A gradient center axis: px or percentage-of-box.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum GradPosAxis {
    Px(f32),
    Pct(f32),
}

/// Full N-stop gradient carried on a layout box (linear / radial /
/// conic + repeating). The painter rasterizes this. CSS Images 3 §3.
#[derive(Clone, Debug, PartialEq)]
pub enum GradientSpec {
    Linear {
        angle_deg: f32,
        stops: Vec<GradientStopSpec>,
        repeating: bool,
    },
    Radial {
        shape: GradRadialShape,
        size: GradRadialSize,
        center: Option<(GradPosAxis, GradPosAxis)>,
        stops: Vec<GradientStopSpec>,
        repeating: bool,
    },
    Conic {
        from_deg: f32,
        center: Option<(GradPosAxis, GradPosAxis)>,
        stops: Vec<GradientStopSpec>,
        repeating: bool,
    },
}

/// Resolved CSS `box-shadow`. V1 supports the single drop-shadow form
/// (no blur, no spread, no inset). Sites whose shorthand has only
/// offsets + color get a proper visual; the blur radius is parsed but
/// dropped at paint time.
/// Resolved CSS `clip-path` shape. The renderer translates the
/// LengthSpec values to pixels using the box's content rect.
#[derive(Clone, Debug, PartialEq)]
pub enum ClipShape {
    Inset {
        top_px: f32,
        right_px: f32,
        bottom_px: f32,
        left_px: f32,
    },
    Circle {
        radius_px: f32,
        cx_px: f32,
        cy_px: f32,
    },
    Polygon(Vec<(f32, f32)>),
}

/// One filter operation resolved for a layout box. conclave
/// converts the parsed `cv_css::FilterFn` into one of these at the
/// `lower_style` stage so this crate doesn't depend on cv_css.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum FilterEffect {
    Blur(f32),
    Brightness(f32),
    Contrast(f32),
    Grayscale(f32),
    Invert(f32),
    Sepia(f32),
    Saturate(f32),
    HueRotate(f32),
    Opacity(f32),
    DropShadow(BoxShadow),
}

/// A CSS `mix-blend-mode` / `background-blend-mode` value resolved at the
/// `lower_style` stage (so this crate stays independent of cv_css / cv_gfx).
/// cv_browser maps this to `cv_gfx::BlendMode` at paint time. CSS Compositing
/// & Blending Level 1 §5/§6.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
pub enum BlendMode {
    #[default]
    Normal,
    Multiply,
    Screen,
    Overlay,
    Darken,
    Lighten,
    ColorDodge,
    ColorBurn,
    HardLight,
    SoftLight,
    Difference,
    Exclusion,
    Hue,
    Saturation,
    Color,
    Luminosity,
}

impl BlendMode {
    /// Parse a CSS blend-mode keyword. Unknown / `normal` → `Normal`.
    pub fn from_str(s: &str) -> Self {
        match s.trim().to_ascii_lowercase().as_str() {
            "multiply" => Self::Multiply,
            "screen" => Self::Screen,
            "overlay" => Self::Overlay,
            "darken" => Self::Darken,
            "lighten" => Self::Lighten,
            "color-dodge" => Self::ColorDodge,
            "color-burn" => Self::ColorBurn,
            "hard-light" => Self::HardLight,
            "soft-light" => Self::SoftLight,
            "difference" => Self::Difference,
            "exclusion" => Self::Exclusion,
            "hue" => Self::Hue,
            "saturation" => Self::Saturation,
            "color" => Self::Color,
            "luminosity" => Self::Luminosity,
            _ => Self::Normal,
        }
    }

    pub fn is_normal(self) -> bool {
        matches!(self, Self::Normal)
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct BoxShadow {
    pub offset_x_px: f32,
    pub offset_y_px: f32,
    pub blur_px: f32,
    /// Spread radius in px. Positive expands the shadow box, negative
    /// contracts it before blur is applied.
    pub spread_px: f32,
    pub color: Color,
    /// `inset` keyword: shadow is drawn inside the element's box.
    pub inset: bool,
}

#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct EdgeSizes {
    pub top: f32,
    pub right: f32,
    pub bottom: f32,
    pub left: f32,
}

/// One axis of an explicit `background-size` value.
///
/// Percentages must be resolved against the actual element box dimensions at
/// paint time — they are NOT pre-resolved to px during style computation
/// because the containing block size is unknown then.
#[derive(Clone, Debug, PartialEq)]
pub enum BgLength {
    /// An absolute pixel length.
    Px(f32),
    /// A percentage of the background positioning area (0–100).
    Percent(f32),
}

impl BgLength {
    /// Resolve to a concrete px value given the background positioning area
    /// dimension along this axis (`area` is the box's content/padding width
    /// or height in px).
    pub fn resolve(&self, area: f32) -> f32 {
        match self {
            BgLength::Px(v) => *v,
            BgLength::Percent(p) => area * p / 100.0,
        }
    }
}

/// Resolved CSS `background-size` value carried on layout boxes.
///
/// `None` on the field means "property not set" → use the image's natural
/// size. When set, this enum is the semantic value; the paint code resolves
/// `Cover`/`Contain` against the actual box dimensions at blit time.
#[derive(Clone, Debug, PartialEq)]
pub enum BgSize {
    /// `background-size: cover` — scale to cover the background area.
    Cover,
    /// `background-size: contain` — scale to fit inside the background area.
    Contain,
    /// Explicit axis values (`None` = `auto`, preserve aspect ratio).
    /// Each axis is either a px length or a percentage of the positioning
    /// area.  Percentages are resolved at paint time against the element's
    /// actual box dimensions — NOT pre-resolved during style computation.
    Explicit(Option<BgLength>, Option<BgLength>),
}

/// Per-side `margin: auto` flag. Block layout uses the horizontal pair
/// to centre an explicitly-sized box inside its containing block —
/// "centre with `margin: 0 auto`" is the most common centring idiom on
/// the modern web.
#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct EdgeAuto {
    pub top: bool,
    pub right: bool,
    pub bottom: bool,
    pub left: bool,
}

#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct Color {
    pub r: u8,
    pub g: u8,
    pub b: u8,
    pub a: u8,
}

impl Color {
    /// `currentColor` sentinel — mirrors `cv_css::properties::Color::CURRENT`
    /// (0,1,0,0). Carried through lowering until `build_styled_tree_full`
    /// resolves it against the element's final (inheritance-applied)
    /// text color. CSS Color 4 §4.4.
    pub const CURRENT: Self = Self {
        r: 0,
        g: 1,
        b: 0,
        a: 0,
    };
    pub fn is_current_color(self) -> bool {
        self == Self::CURRENT
    }
}

/// CSS `border-style` values carried by layout boxes and used by the painter.
/// Mirrors `cv_css::cascade::BorderStyle`; re-defined here so `cv_layout`
/// stays free of the CSS-crate dependency.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum BorderStyle {
    None,
    Hidden,
    #[default]
    Solid,
    Dashed,
    Dotted,
    Double,
}

impl BorderStyle {
    /// Returns `true` when this style means "no border" (width collapses to 0).
    pub fn is_none(&self) -> bool {
        matches!(self, BorderStyle::None | BorderStyle::Hidden)
    }
}

#[derive(Clone, Debug)]
pub struct LayoutBox {
    pub content: Rect,
    pub padding: EdgeSizes,
    pub margin: EdgeSizes,
    /// `margin: auto` flags per side. Block layout reads
    /// `margin_auto.left` / `.right` to distribute leftover horizontal
    /// space; vertical-axis auto is reserved for the flex-item
    /// alignment fix-up.
    pub margin_auto: EdgeAuto,
    /// Uniform border width — back-compat sum. Layout chrome math
    /// (margin_rect, border_rect) reads this AND `borders_per_side`
    /// when set; both must agree, so on lowering we copy the max of
    /// the four into here.
    pub border_width_px: f32,
    pub border_color: Option<Color>,
    /// Per-side widths in px (top, right, bottom, left). `None`
    /// means "fall back to the uniform border width"; `Some(0.0)`
    /// is an explicit zero that must NOT fall back.
    pub border_widths_per_side: [Option<f32>; 4],
    /// Per-side border colours. None falls back to `border_color`.
    pub border_colors_per_side: [Option<Color>; 4],
    /// Per-side border styles (top, right, bottom, left).
    /// `None` means solid (initial value). `None`/`Hidden` collapses width.
    pub border_styles_per_side: [Option<BorderStyle>; 4],
    pub text_align: Option<TextAlign>,
    pub font_weight_bold: bool,
    /// Numeric CSS font-weight (1–1000), so heavy weights (800/900) render at
    /// their real weight instead of collapsing to bold(700). 0 means "unset →
    /// derive from `font_weight_bold`" (700 if bold else 400).
    pub font_weight_num: u16,
    pub font_style_italic: bool,
    pub font_family: Option<String>,
    /// CSS `text-transform`. `None` = inherit / initial = no transform.
    /// Applied at glyph-bake time in the paint pass — the visible
    /// difference between Tailwind's `uppercase` class (which Chrome
    /// renders as "TOTAL BLOCKS") and the raw lowercase JSX content.
    pub text_transform: Option<TextTransform>,
    /// CSS `letter-spacing` extra px between glyphs. Forwarded to
    /// `TextItem.letter_spacing_px` at paint time.
    pub letter_spacing_px: f32,
    pub text_decoration_underline: bool,
    pub text_decoration_line_through: bool,
    pub line_height_px: Option<f32>,
    pub preserve_whitespace: bool,
    pub box_sizing_border_box: bool,
    pub is_flex: bool,
    pub is_grid: bool,
    /// `display: inline` or `display: inline-block`. Flows
    /// horizontally with siblings in the parent's inline-formatting
    /// pass; participates in line breaking when the running line
    /// exceeds the container width.
    pub is_inline: bool,
    /// `display: inline-block` specifically. Keeps the box's own
    /// width/height/padding/border intact (vs. plain `inline` which
    /// only contributes its rendered text width). Used by buttons,
    /// nav pills, tag badges, and form-input rows.
    pub is_inline_block: bool,
    /// `display: table`. Triggers the CSS 2.1 §17 auto-column-width
    /// table layout in `place_table` instead of the regular block
    /// flow.
    pub is_table: bool,
    /// `display: table-row` — direct child of a table that holds cells.
    pub is_table_row: bool,
    /// `display: table-cell` — a cell inside a table row.
    pub is_table_cell: bool,
    /// `display: table-row-group` (`<tbody>`/`<thead>`/`<tfoot>`).
    /// Transparent — its rows are pulled into the parent table's flow.
    pub is_table_row_group: bool,
    /// `display: list-item` (UA `li` rule or authored). Box flows as block; this
    /// only drives the CSSOM `display` value (`getComputedStyle().display`).
    pub is_list_item: bool,
    /// `display: flow-root`. Box flows as block; drives the CSSOM display value.
    pub is_flow_root: bool,
    pub flex_direction: FlexDirection,
    pub flex_wrap: FlexWrap,
    pub flex_grow: f32,
    pub flex_shrink: f32,
    pub flex_basis: Option<LengthSpec>,
    pub justify_content: JustifyContent,
    pub align_items: AlignItems,
    /// `justify-items` — grid-only sibling of align_items along the
    /// inline axis. Falls back to align_items when unset.
    pub justify_items: Option<AlignItems>,
    /// Per-item override of the container's align-items.
    pub align_self: Option<AlignItems>,
    /// Per-item override of the container's justify-items.
    pub justify_self: Option<AlignItems>,
    pub gap_px: f32,
    pub column_gap_px: f32,
    pub row_gap_px: f32,
    pub grid_template_columns: Option<Vec<GridTrack>>,
    pub grid_template_rows: Option<Vec<GridTrack>>,
    /// `grid-auto-rows` — single track sizing applied to implicit
    /// row tracks created beyond the explicit template count.
    pub grid_auto_rows: Option<GridTrack>,
    /// `grid-auto-columns` — same as grid-auto-rows for the column axis.
    pub grid_auto_columns: Option<GridTrack>,
    /// `grid-template-areas` — rows × columns of cell names. `.` is
    /// the empty marker. A name spanning adjacent cells defines a
    /// rectangular named area whose bounds are computed at layout time
    /// when this box becomes a grid container.
    pub grid_template_areas: Option<Vec<Vec<String>>>,
    pub grid_column_start: Option<usize>,
    pub grid_column_span: Option<usize>,
    pub grid_row_start: Option<usize>,
    pub grid_row_span: Option<usize>,
    /// HTML `colspan` on a table cell — number of columns the cell spans.
    pub table_col_span: Option<usize>,
    /// HTML `rowspan` on a table cell — number of rows the cell spans.
    /// Defaults to 1 (absent = 1).
    pub table_row_span: Option<usize>,
    /// `grid-area: name` — resolved against the parent grid container's
    /// `grid_template_areas` at layout time. When present, overrides
    /// any explicit numeric `grid_*_start`/`_span` on this box.
    pub grid_area_name: Option<String>,
    pub overflow_hidden: bool,
    /// Resolved per-axis overflow. `overflow_hidden` stays as the legacy
    /// "this box clips" flag (true whenever either axis clips); these two
    /// carry the full value so paint can tell `hidden` (clip only) from
    /// `scroll`/`auto` (clip + independent scroll offset).
    pub overflow_x: Overflow,
    pub overflow_y: Overflow,
    /// The box's current scroll position (CSSOM `scrollLeft`/`scrollTop`),
    /// in CSS px. Content paints translated by `-scroll_offset`. Only
    /// meaningful when the corresponding axis `is_scrollable()` and there
    /// is scrollable overflow; clamped to `[0, scroll_size - client_size]`.
    /// Applied at paint time, NOT during box placement (children keep
    /// their layout coordinates; only the paint offset shifts) — matching
    /// Blink's `PaintLayerScrollableArea`, where the scroll offset is a
    /// paint-property-tree translation, not a re-layout.
    pub scroll_offset_x: f32,
    pub scroll_offset_y: f32,
    /// CSS `visibility: hidden` — box keeps its layout slot but the
    /// paint pass skips it (and inherits to descendants unless they
    /// explicitly set `visibility: visible`).  Critical for honoring
    /// modal-dialog / tooltip pre-DOM-ready hiding.
    pub visibility_hidden: bool,
    pub opacity: f32,
    pub border_radius_px: f32,
    pub border_radius_percent: Option<f32>,
    pub translate_x_px: f32,
    pub translate_y_px: f32,
    pub translate_x_percent: Option<f32>,
    pub translate_y_percent: Option<f32>,
    /// `transform: scale(x, y)` factors. 1.0 = identity. Negative values
    /// flip — used by some CSS-only logo animations.
    pub scale_x: f32,
    pub scale_y: f32,
    /// `transform: rotate(angle)` rotation in degrees. Painter rotates
    /// the box's content around its centre by this amount.
    pub rotate_deg: f32,
    /// `transform: matrix(a,b,c,d,e,f)` — the raw 2D affine when the
    /// author used the matrix() function. `Some` overrides the scalar
    /// scale/rotate fields in the paint transform. `[a,b,c,d,e,f]`.
    pub matrix_2d: Option<[f32; 6]>,
    /// `transform-origin` pivot point. `None` = default 50% 50% (centre).
    /// Expressed as (x, y) [`BgPos`] values that are resolved to pixels
    /// against the border-box at paint time.
    pub transform_origin: Option<(BgPos, BgPos)>,
    pub position: Position,
    /// CSS `z-index`. `None` = `auto` (participates in parent stacking
    /// context). `Some(n)` = explicit stacking order: lower n paints
    /// behind higher n among siblings within the same stacking context.
    /// Only meaningful on positioned boxes and flex/grid items per spec,
    /// but we store it on every box and apply it at paint-sort time.
    pub z_index: Option<i32>,
    /// CSS `float` — Left/Right takes the box out of normal block flow
    /// and pins it to the named edge of its containing block. Following
    /// in-flow boxes shrink to wrap until they clear the float's bottom.
    pub float_side: FloatSide,
    /// CSS `clear`. When non-`None`, the box's top is pushed down past
    /// the bottom of every active float on the named side(s).
    pub clear: ClearMode,
    /// CSS `vertical-align`. Within an inline run, `super` raises the
    /// box by ~30% of its font size and `sub` lowers it by ~20%.
    pub vertical_align: VerticalAlign,
    /// Resolved `box-shadow` (offset + color). V1 paints a solid
    /// offset rect — no blur — so the visual reads as a card shadow.
    pub box_shadow: Option<BoxShadow>,
    /// Resolved `text-shadow`. Same shape as `box-shadow` but applies
    /// to text glyphs. Paint stamps a darker copy of the text at the
    /// offset before the real text, giving the visual without alpha
    /// compositing (we don't have blur yet).
    pub text_shadow: Option<BoxShadow>,
    /// Resolved `filter:` function chain. Each entry is a tagged op
    /// (Blur, Brightness, Grayscale, ...). Paint runs them in order
    /// against the box's painted bounding rect after children finish.
    pub filters: Vec<FilterEffect>,
    /// Resolved `backdrop-filter:` chain. Paint applies these to the already
    /// painted backdrop under this box before the box's own background/border
    /// are drawn.
    pub backdrop_filters: Vec<FilterEffect>,
    /// `mix-blend-mode` — blends this element (its whole subtree) with the page
    /// backdrop beneath it. `Normal` = plain source-over.
    pub mix_blend_mode: BlendMode,
    /// `background-blend-mode` — blends this element's background image layer
    /// with the background color/gradient painted beneath it. `Normal` = none.
    pub background_blend_mode: BlendMode,
    pub animation_name: Option<String>,
    pub animation_duration_ms: f32,
    pub animation_delay_ms: f32,
    pub animation_iteration_count: f32,
    pub animation_timing: u8,
    pub clip_shape: Option<ClipShape>,
    /// `mask: url(...)` is set on this box. Painter skips the
    /// background fill so a tintable icon doesn't render as a
    /// solid coloured rect.
    pub has_mask_url: bool,
    /// Raw `mask-image: url(...)` for the painter to decode.
    pub mask_image_url: Option<String>,
    /// Resolved linear gradient — overrides solid `background` when set.
    pub background_gradient: Option<LinearGradientSpec>,
    /// Resolved radial gradient — center-out two-stop fade.
    pub background_radial_gradient: Option<LinearGradientSpec>,
    /// Full N-stop gradient (linear/radial/conic + repeating). When set,
    /// the painter rasterizes this instead of the 2-stop approximations.
    pub background_gradient_full: Option<GradientSpec>,
    /// `background-image: url(...)` URL string (unresolved). Painter
    /// looks this up in the image cache to draw the background bitmap.
    pub background_image_url: Option<String>,
    /// CSS `background-repeat`. Painter consults this when blitting the
    /// resolved `background_image` — default tiles the bitmap rather
    /// than stretches it (which is what the spec says and what every
    /// site assumes).
    pub background_repeat: BackgroundRepeat,
    /// `position` offsets. Kept as `LengthSpec` rather than a
    /// pre-resolved `f32` so percentage offsets (`left: 50%`) can
    /// resolve against the actual containing block at layout time,
    /// not against a default-zero container at cascade-lowering time.
    /// That latter mistake collapses every `position: absolute`
    /// element with a percent offset to the same (0, 0) corner — it
    /// was why Wikipedia's central language pills overlapped.
    pub top_px: Option<LengthSpec>,
    pub right_px: Option<LengthSpec>,
    pub bottom_px: Option<LengthSpec>,
    pub left_px: Option<LengthSpec>,
    pub explicit_width: Option<LengthSpec>,
    pub flex_override_width: Option<f32>,
    pub explicit_height: Option<LengthSpec>,
    pub flex_override_height: Option<f32>,
    pub aspect_ratio: Option<f32>,
    /// Resolved `max-width` clamp. After we compute a candidate
    /// content width we cap it at `min(candidate, max_width)`. With
    /// `margin-auto` on the sides this is how `max-width: 1200px;
    /// margin: 0 auto` centres a page within a wider viewport.
    pub max_width: Option<LengthSpec>,
    pub max_height: Option<LengthSpec>,
    pub min_width: Option<LengthSpec>,
    pub min_height: Option<LengthSpec>,
    pub background: Option<Color>,
    pub text_color: Color,
    pub font_size_px: f32,
    pub kind: BoxKind,
    /// If this box (or any inline that produced it) belongs to a clickable
    /// `<a href>`, the resolved href URL ends up here. Used for hit testing.
    pub link_href: Option<String>,
    /// `<img>`-emitted box: pre-decoded BGRA image.
    pub embedded_image: Option<std::sync::Arc<EmbeddedImage>>,
    /// `mask-image: url(...)` decoded to BGRA; alpha is used as the
    /// stencil for a tinted icon/background paint.
    pub mask_image: Option<std::sync::Arc<EmbeddedImage>>,
    /// `background-image: url(...)` — pre-decoded BGRA image, painted
    /// behind text and in front of the solid `background` colour. The
    /// raw URL (for use after layout has already resolved sizing) lives
    /// in `Style::background_image_url`.
    pub background_image: Option<std::sync::Arc<EmbeddedImage>>,
    /// Resolved `background-size`. `None` = not set (natural size).
    /// See [`BgSize`] for the semantic variants including `cover`/`contain`.
    pub background_size: Option<BgSize>,
    /// `background-position` (x, y) — see the field of the same name on
    /// `LayoutBox`. Carried from cascade so the painter can offset a
    /// sprite sheet to the icon the element selects.
    pub background_position: Option<(BgPos, BgPos)>,
    /// CSS `object-fit` — controls how `embedded_image` fills the
    /// box. `None` means use the painter's default (stretch / `fill`).
    pub object_fit: Option<ObjectFit>,
    /// CSS `object-position` (x, y) — where to place the image within
    /// the content box. Each axis is a [`BgPos`] (px offset or percentage
    /// of `box_extent - image_extent`). `None` = default 50% 50%.
    pub object_position: Option<(BgPos, BgPos)>,
    /// See `Style::element_path`.
    pub element_path: Option<Vec<usize>>,
    /// See `Style::node_id` — stable arena identity for incremental-layout caching.
    pub node_id: Option<u64>,
    /// Set by `build_box` on the DIRECT children of a flex/grid/table container:
    /// such boxes are positioned/sized by their parent's formatting context (a
    /// hidden layout input the fragment-cache key doesn't capture), so they are
    /// never individually cached — they're captured inside their container's
    /// cached fragment instead.
    pub cache_ineligible: bool,
    pub children: Vec<LayoutBox>,
}

#[derive(Clone, Debug)]
pub enum BoxKind {
    Block { tag: String },
    Anonymous,
    Text(String),
}

impl LayoutBox {
    /// Padding box: content + padding.
    pub fn padding_rect(&self) -> Rect {
        Rect {
            x: self.content.x - self.padding.left,
            y: self.content.y - self.padding.top,
            w: self.content.w + self.padding.left + self.padding.right,
            h: self.content.h + self.padding.top + self.padding.bottom,
        }
    }

    /// Border box: padding box + border on each side. Uses per-side
    /// widths when set; falls back to the uniform `border_width_px`.
    pub fn border_rect(&self) -> Rect {
        let p = self.padding_rect();
        let bt = self.border_width_top();
        let br = self.border_width_right();
        let bb = self.border_width_bottom();
        let bl = self.border_width_left();
        Rect {
            x: p.x - bl,
            y: p.y - bt,
            w: p.w + bl + br,
            h: p.h + bt + bb,
        }
    }

    /// True when this box needs the affine (layer-composite) paint path
    /// rather than the cheap translate-offset / geometry-scale paths —
    /// i.e. it rotates or carries a non-identity `matrix()`. Pure
    /// scale()/translate() are handled without a layer.
    pub fn has_affine_transform(&self) -> bool {
        if self.rotate_deg.abs() > 1e-3 {
            return true;
        }
        if let Some(m) = self.matrix_2d {
            return (m[0] - 1.0).abs() > 1e-4
                || m[1].abs() > 1e-4
                || m[2].abs() > 1e-4
                || (m[3] - 1.0).abs() > 1e-4
                || m[4].abs() > 1e-4
                || m[5].abs() > 1e-4;
        }
        false
    }

    /// The box's 2D affine `[a,b,c,d,e,f]` in CSS matrix order: a point
    /// (x,y) maps to (a·x + c·y + e, b·x + d·y + f). Taken from
    /// `matrix()` if present, else composed as R(rotate)·S(scale) with
    /// the translate() offset as (e,f). Used by the painter to build the
    /// layer-composite transform around the transform-origin.
    pub fn transform_affine(&self) -> [f32; 6] {
        if let Some(m) = self.matrix_2d {
            return m;
        }
        let theta = self.rotate_deg.to_radians();
        let (s, c) = theta.sin_cos();
        let sx = if self.scale_x == 0.0 {
            1.0
        } else {
            self.scale_x
        };
        let sy = if self.scale_y == 0.0 {
            1.0
        } else {
            self.scale_y
        };
        // M = R(theta) · S(sx, sy):
        //   a = cosθ·sx, b = sinθ·sx, c = -sinθ·sy, d = cosθ·sy
        [
            c * sx,
            s * sx,
            -s * sy,
            c * sy,
            self.translate_x_px,
            self.translate_y_px,
        ]
    }

    /// Union of this box's visually painted bounds with every descendant's
    /// visually painted bounds, in untransformed document coordinates. The
    /// painter rasters this region into a layer bitmap before applying an
    /// affine transform, so this must include paint-time translate overflow
    /// and shadows. Otherwise children that intentionally straddle their
    /// parent (for example orbit nodes using `translate(-50%, -50%)`) get
    /// clipped before the rotated layer is composited.
    pub fn subtree_bounds(&self) -> Rect {
        let mut r = self.visual_bounds();
        for child in &self.children {
            let cr = child.subtree_bounds();
            let x0 = r.x.min(cr.x);
            let y0 = r.y.min(cr.y);
            let x1 = (r.x + r.w).max(cr.x + cr.w);
            let y1 = (r.y + r.h).max(cr.y + cr.h);
            r = Rect {
                x: x0,
                y: y0,
                w: x1 - x0,
                h: y1 - y0,
            };
        }
        r
    }

    fn visual_bounds(&self) -> Rect {
        let mut r = self.border_rect();
        r.x += self.translate_x_px;
        r.y += self.translate_y_px;
        if let Some(sh) = self.box_shadow {
            let spread = (sh.blur_px * 0.5).max(0.0).ceil();
            let sx = r.x + sh.offset_x_px - spread;
            let sy = r.y + sh.offset_y_px - spread;
            let sw = r.w + spread * 2.0;
            let sh_h = r.h + spread * 2.0;
            let x0 = r.x.min(sx);
            let y0 = r.y.min(sy);
            let x1 = (r.x + r.w).max(sx + sw);
            let y1 = (r.y + r.h).max(sy + sh_h);
            r = Rect {
                x: x0,
                y: y0,
                w: x1 - x0,
                h: y1 - y0,
            };
        }
        r
    }

    pub fn border_width_top(&self) -> f32 {
        // CSS §8.5.1: width collapses to 0 when style is none/hidden.
        if self.border_styles_per_side[0].as_ref().map_or(false, |s| s.is_none()) {
            return 0.0;
        }
        self.border_widths_per_side[0].unwrap_or(self.border_width_px)
    }
    pub fn border_width_right(&self) -> f32 {
        if self.border_styles_per_side[1].as_ref().map_or(false, |s| s.is_none()) {
            return 0.0;
        }
        self.border_widths_per_side[1].unwrap_or(self.border_width_px)
    }
    pub fn border_width_bottom(&self) -> f32 {
        if self.border_styles_per_side[2].as_ref().map_or(false, |s| s.is_none()) {
            return 0.0;
        }
        self.border_widths_per_side[2].unwrap_or(self.border_width_px)
    }
    pub fn border_width_left(&self) -> f32 {
        if self.border_styles_per_side[3].as_ref().map_or(false, |s| s.is_none()) {
            return 0.0;
        }
        self.border_widths_per_side[3].unwrap_or(self.border_width_px)
    }

    /// Per-side colour with fallback to the uniform `border_color`.
    pub fn border_color_for(&self, side: usize) -> Option<Color> {
        self.border_colors_per_side[side].or(self.border_color)
    }

    /// Per-side border style. Falls back to `Solid` (CSS initial value).
    pub fn border_style_for(&self, side: usize) -> BorderStyle {
        self.border_styles_per_side[side].unwrap_or(BorderStyle::Solid)
    }

    pub fn margin_rect(&self) -> Rect {
        let b = self.border_rect();
        Rect {
            x: b.x - self.margin.left,
            y: b.y - self.margin.top,
            w: b.w + self.margin.left + self.margin.right,
            h: b.h + self.margin.top + self.margin.bottom,
        }
    }

    // ── Scrolling (CSSOM View §6 + CSS Overflow 3) ──────────────────────────
    // Chrome reference: a box whose computed overflow is `scroll`/`auto`
    // (and that has overflowing content) gets a `PaintLayerScrollableArea`
    // (blink/renderer/core/paint/paint_layer_scrollable_area.cc). Its scroll
    // offset translates the painted content and clips it to the padding box.

    /// True when EITHER axis establishes an independently scrollable region.
    pub fn is_scroll_container(&self) -> bool {
        self.overflow_x.is_scrollable() || self.overflow_y.is_scrollable()
    }

    /// `Element.clientWidth` (CSSOM View §6.1): width of the *padding box*,
    /// i.e. content width + left/right padding. (We don't render a UA
    /// scrollbar gutter, so nothing is subtracted for it.)
    pub fn client_width(&self) -> f32 {
        (self.content.w + self.padding.left + self.padding.right).max(0.0)
    }

    /// `Element.clientHeight`: height of the padding box.
    pub fn client_height(&self) -> f32 {
        (self.content.h + self.padding.top + self.padding.bottom).max(0.0)
    }

    /// The scrollable-overflow extent (right, bottom) of this box's
    /// content in the box's own padding-box coordinate space, where the
    /// padding-box top-left is the origin. This unions the border-box of
    /// every in-flow/positioned descendant (recursively, since a child's
    /// own overflow can escape it unless the child clips) and adds the
    /// box's end-side padding — matching how Chrome's scrolling area
    /// reaches one padding past the last child.
    fn scrollable_overflow_extent(&self) -> (f32, f32) {
        // The CONTENT box's top-left in document coordinates is the origin
        // for measuring children (the scrolling area starts at padding-top,
        // and content begins one padding-top below the padding-box origin —
        // so measuring children from the content origin and then adding back
        // BOTH paddings reproduces "padding-top .. last child .. padding-
        // bottom", matching Chrome/MDN: scrollHeight runs from padding-top to
        // padding-bottom and includes the end padding even past overflow).
        let origin_x = self.content.x;
        let origin_y = self.content.y;
        // Furthest child border-box edge relative to the content origin.
        // Floor at 0 (no children → 0) so the padding-only fallback wins.
        let mut child_right = 0.0f32;
        let mut child_bottom = 0.0f32;
        fn accumulate(
            b: &LayoutBox,
            origin_x: f32,
            origin_y: f32,
            right: &mut f32,
            bottom: &mut f32,
        ) {
            let br = b.border_rect();
            let r = (br.x + br.w) - origin_x;
            let bo = (br.y + br.h) - origin_y;
            if r > *right {
                *right = r;
            }
            if bo > *bottom {
                *bottom = bo;
            }
            // A child that itself clips/scrolls confines its own
            // descendants — they don't contribute to THIS box's
            // scrollable overflow beyond the child's border box.
            if b.overflow_x.clips() && b.overflow_y.clips() {
                return;
            }
            for c in &b.children {
                accumulate(c, origin_x, origin_y, right, bottom);
            }
        }
        for c in &self.children {
            accumulate(c, origin_x, origin_y, &mut child_right, &mut child_bottom);
        }
        // Add the start+end padding around the content-relative extent so the
        // scrolling area spans the full padding box (and one padding past the
        // last child). An empty scroller falls back to its own padding box.
        let right = (child_right + self.padding.left + self.padding.right)
            .max(self.content.w + self.padding.left + self.padding.right);
        let bottom = (child_bottom + self.padding.top + self.padding.bottom)
            .max(self.content.h + self.padding.top + self.padding.bottom);
        (right.max(0.0), bottom.max(0.0))
    }

    /// `Element.scrollWidth`: `max(clientWidth, scrollable overflow width)`.
    pub fn scroll_width(&self) -> f32 {
        let (right, _) = self.scrollable_overflow_extent();
        right.max(self.client_width())
    }

    /// `Element.scrollHeight`: `max(clientHeight, scrollable overflow height)`.
    pub fn scroll_height(&self) -> f32 {
        let (_, bottom) = self.scrollable_overflow_extent();
        bottom.max(self.client_height())
    }

    /// Largest legal `scrollLeft` for this box: `scrollWidth - clientWidth`,
    /// floored at 0. 0 when there's nothing to scroll horizontally.
    pub fn max_scroll_left(&self) -> f32 {
        (self.scroll_width() - self.client_width()).max(0.0)
    }

    /// Largest legal `scrollTop`: `scrollHeight - clientHeight`, floored at 0.
    pub fn max_scroll_top(&self) -> f32 {
        (self.scroll_height() - self.client_height()).max(0.0)
    }

    /// Clamp the current scroll offset into the legal range. Returns the
    /// clamped offset (also written back into the box). Honors per-axis
    /// scrollability: a non-scrollable axis is pinned to 0.
    pub fn clamp_scroll(&mut self) -> (f32, f32) {
        let max_x = if self.overflow_x.is_scrollable() {
            self.max_scroll_left()
        } else {
            0.0
        };
        let max_y = if self.overflow_y.is_scrollable() {
            self.max_scroll_top()
        } else {
            0.0
        };
        self.scroll_offset_x = self.scroll_offset_x.clamp(0.0, max_x);
        self.scroll_offset_y = self.scroll_offset_y.clamp(0.0, max_y);
        (self.scroll_offset_x, self.scroll_offset_y)
    }
}

impl fmt::Display for LayoutBox {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let tag = match &self.kind {
            BoxKind::Block { tag } => tag.clone(),
            BoxKind::Anonymous => "(anon)".to_string(),
            BoxKind::Text(t) => format!("text({:?})", t.chars().take(40).collect::<String>()),
        };
        write!(
            f,
            "[{} @ {:.0},{:.0} {:.0}x{:.0}]",
            tag, self.content.x, self.content.y, self.content.w, self.content.h
        )
    }
}

#[derive(Clone, Debug)]
pub struct StyledNode {
    pub kind: StyledKind,
    pub style: Style,
    pub children: Vec<StyledNode>,
}

#[derive(Clone, Debug)]
pub enum StyledKind {
    Element { tag: String },
    Text(String),
}

#[derive(Clone, Debug, Default)]
pub struct Style {
    pub display: Option<Display>,
    /// `display: list-item` was authored (or the UA list-item rule applied).
    /// The box still lays out as Block (`display` maps to Block); this only
    /// records the list-item intent so the CSSOM `display` reads back as
    /// `list-item` and marker generation can find it.
    pub display_is_list_item: bool,
    /// `display: flow-root` was authored. Lays out as Block (establishes a new
    /// BFC, not yet modelled separately); recorded so CSSOM `display` reads back
    /// as `flow-root`.
    pub display_is_flow_root: bool,
    pub background: Option<Color>,
    pub text_color: Option<Color>,
    pub font_size_px: Option<f32>,
    pub width: Option<LengthSpec>,
    pub height: Option<LengthSpec>,
    pub aspect_ratio: Option<f32>,
    pub max_width: Option<LengthSpec>,
    pub max_height: Option<LengthSpec>,
    pub min_width: Option<LengthSpec>,
    pub min_height: Option<LengthSpec>,
    pub margin: EdgeSizes,
    pub margin_auto: EdgeAuto,
    pub padding: EdgeSizes,
    pub border_width_px: Option<f32>,
    pub border_color: Option<Color>,
    /// Per-side widths. None entries fall back to `border_width_px`.
    pub border_widths_per_side: [Option<f32>; 4],
    /// Per-side colours. None entries fall back to `border_color`.
    pub border_colors_per_side: [Option<Color>; 4],
    /// Per-side styles. None means Solid (initial value).
    pub border_styles_per_side: [Option<BorderStyle>; 4],
    pub text_align: Option<TextAlign>,
    pub font_weight_bold: Option<bool>,
    /// Numeric CSS font-weight (1–1000) when specified; `None` = unset (derive
    /// from `font_weight_bold`). Threaded to the computed `Style.font_weight_num`
    /// so 800/900 render heavier than bold.
    pub font_weight_num: Option<u16>,
    pub font_style_italic: Option<bool>,
    pub font_family: Option<String>,
    /// CSS `text-transform`. Inherits; `None` here means "use the
    /// inherited value or `TextTransform::None` if root". Applied at
    /// glyph-bake time in `paint_box_offset`.
    pub text_transform: Option<TextTransform>,
    /// CSS `letter-spacing` extra px. Inherits.
    pub letter_spacing_px: Option<f32>,
    pub text_decoration_underline: Option<bool>,
    pub text_decoration_line_through: Option<bool>,
    pub line_height_px: Option<f32>,
    pub preserve_whitespace: bool,
    pub box_sizing_border_box: Option<bool>,
    pub flex_direction: Option<FlexDirection>,
    pub flex_wrap: Option<FlexWrap>,
    pub flex_grow: Option<f32>,
    pub flex_shrink: Option<f32>,
    pub flex_basis: Option<LengthSpec>,
    pub justify_content: Option<JustifyContent>,
    pub align_items: Option<AlignItems>,
    /// `justify-items`: cross-of-align in the grid inline axis.
    pub justify_items: Option<AlignItems>,
    /// Per-item override of the parent's `align-items`.
    pub align_self: Option<AlignItems>,
    /// Per-item override of the parent's `justify-items`.
    pub justify_self: Option<AlignItems>,
    pub gap_px: Option<f32>,
    /// `grid-template-columns` track list (pre-resolved).
    pub grid_template_columns: Option<Vec<GridTrack>>,
    /// `grid-template-rows` track list (pre-resolved).
    pub grid_template_rows: Option<Vec<GridTrack>>,
    /// `grid-auto-rows` — single track for implicit rows.
    pub grid_auto_rows: Option<GridTrack>,
    /// `grid-auto-columns` — single track for implicit columns.
    pub grid_auto_columns: Option<GridTrack>,
    /// `grid-template-areas` — see LayoutBox docs.
    pub grid_template_areas: Option<Vec<Vec<String>>>,
    pub grid_column_start: Option<usize>,
    pub grid_column_span: Option<usize>,
    pub grid_row_start: Option<usize>,
    pub grid_row_span: Option<usize>,
    /// HTML `colspan` on a table cell — number of columns the cell spans.
    pub table_col_span: Option<usize>,
    /// HTML `rowspan` on a table cell — number of rows the cell spans.
    pub table_row_span: Option<usize>,
    pub grid_area_name: Option<String>,
    /// `column-gap` distinct from the row axis. None → fall back to `gap`.
    pub column_gap_px: Option<f32>,
    /// `row-gap` distinct from the column axis. None → fall back to `gap`.
    pub row_gap_px: Option<f32>,
    pub overflow_hidden: bool,
    /// Per-axis overflow lowered from `cv_css`. Default `Visible`.
    pub overflow_x: Overflow,
    pub overflow_y: Overflow,
    /// CSS `visibility: hidden` flag — propagates through `build_box`
    /// onto every layout box that carries it.  Painters skip such
    /// boxes (and inherit "hidden" to descendants unless they
    /// explicitly set `visibility: visible`).
    pub visibility_hidden: Option<bool>,
    /// 0.0 = fully transparent, 1.0 = opaque. None = 1.0.
    pub opacity: Option<f32>,
    pub border_radius_px: Option<f32>,
    pub border_radius_percent: Option<f32>,
    pub translate_x_px: Option<f32>,
    pub translate_y_px: Option<f32>,
    pub translate_x_percent: Option<f32>,
    pub translate_y_percent: Option<f32>,
    pub scale_x: Option<f32>,
    pub scale_y: Option<f32>,
    pub rotate_deg: Option<f32>,
    pub matrix_2d: Option<[f32; 6]>,
    /// `transform-origin` pivot for 2D transforms. None = default 50% 50%.
    pub transform_origin: Option<(BgPos, BgPos)>,
    /// `box-shadow: Xpx Ypx [blur] [spread] color`. V1 ignores blur and
    /// spread; the painter just blits a solid colored rect offset by
    /// (X, Y) underneath the box's normal background. Reads as a
    /// "card shadow" without the cost of a real Gaussian blur.
    pub box_shadow: Option<BoxShadow>,
    /// `text-shadow: Xpx Ypx [blur] color`. V1 ignores blur. Painter
    /// stamps a copy of the text at the offset before drawing the
    /// real text, giving the same "card shadow" look on glyphs.
    pub text_shadow: Option<BoxShadow>,
    /// Resolved `filter:` chain. See ComputedStyle::filters.
    pub filters: Vec<FilterEffect>,
    /// Resolved `backdrop-filter:` chain. See LayoutBox::backdrop_filters.
    pub backdrop_filters: Vec<FilterEffect>,
    /// Resolved `mix-blend-mode`. See LayoutBox::mix_blend_mode.
    pub mix_blend_mode: BlendMode,
    /// Resolved `background-blend-mode`. See LayoutBox::background_blend_mode.
    pub background_blend_mode: BlendMode,
    /// Resolved animation-name (lookup into the keyframes table).
    pub animation_name: Option<String>,
    /// Animation duration in milliseconds.
    pub animation_duration_ms: f32,
    /// Animation delay in milliseconds.
    pub animation_delay_ms: f32,
    /// Iteration count. f32::INFINITY = `infinite`.
    pub animation_iteration_count: f32,
    /// Timing function (0=linear, 1=ease-in, 2=ease-out, 3=ease-in-out).
    pub animation_timing: u8,
    /// Resolved `clip-path` shape. Painter applies a pixel mask
    /// against the box's border-rect after all other paint passes.
    pub clip_shape: Option<ClipShape>,
    /// `mask: url(...)` flag — pass-through from cascade.
    pub has_mask_url: bool,
    pub mask_image_url: Option<String>,
    /// Resolved linear gradient — when set, the painter uses
    /// `fill_rect_gradient` for the background instead of solid fill.
    pub background_gradient: Option<LinearGradientSpec>,
    pub background_radial_gradient: Option<LinearGradientSpec>,
    /// Full N-stop gradient — rasterized verbatim by the painter.
    pub background_gradient_full: Option<GradientSpec>,
    /// `background-image: url(...)` URL, passed through to the painter
    /// so it can resolve against the document base URL and look up the
    /// bitmap in the image cache.
    pub background_image_url: Option<String>,
    /// CSS `background-repeat`. Tiles the resolved bitmap in the
    /// painter. Default per spec is `Repeat` — we used to always
    /// stretch the bitmap to the padding box, which made any small
    /// pattern explode into a giant smudge.
    pub background_repeat: BackgroundRepeat,
    /// CSS `position`. For V1 we honour `relative` (offset after normal
    /// flow) and ignore static/absolute/fixed/sticky distinctions.
    pub position: Option<Position>,
    /// CSS `z-index`. None = auto. Stored alongside other CSS state so
    /// `build_box` can carry it onto `LayoutBox.z_index`.
    pub z_index: Option<i32>,
    /// CSS `float`. When set to Left/Right the box is positioned at the
    /// corresponding edge of its containing block, subsequent in-flow
    /// blocks shrink to make room until their `y` clears the float's
    /// bottom, and the parent's flow cursor is *not* advanced by the
    /// float's height.
    pub float_side: FloatSide,
    /// CSS `clear`. Non-`None` forces this element's top below the
    /// bottom edge of every active float on the named side(s).
    pub clear: ClearMode,
    /// CSS `vertical-align`. Within an inline run, super raises the
    /// box, sub lowers it; other keywords are V1 no-ops.
    pub vertical_align: VerticalAlign,
    pub top_px: Option<LengthSpec>,
    pub right_px: Option<LengthSpec>,
    pub bottom_px: Option<LengthSpec>,
    pub left_px: Option<LengthSpec>,
    /// For `<a>` elements: the (already-absolute) target URL.
    pub link_href: Option<String>,
    /// DOM element-path corresponding to this styled node. Used by
    /// hit-testing to identify which element the user clicked so the
    /// host can dispatch `addEventListener("click")` callbacks back
    /// through the persistent JS interpreter.
    pub element_path: Option<Vec<usize>>,
    /// Stable arena NodeId (as `u64`) for incremental-layout caching. `None` for
    /// synthetic/anonymous/flattened nodes (always re-laid-out). Set on the
    /// CV_DOM arena path only. `u64` (not `cv_dom::NodeId`) to avoid a
    /// `cv_layout → cv_dom` crate edge.
    pub node_id: Option<u64>,
    /// For `<img>`: decoded image (width, height, BGRA pixels).
    pub embedded_image: Option<std::sync::Arc<EmbeddedImage>>,
    pub mask_image: Option<std::sync::Arc<EmbeddedImage>>,
    /// `background-image: url(...)` — decoded BGRA bitmap (already
    /// fetched). Distinct from `embedded_image` so a single element can
    /// carry both an `<img>` payload and a CSS background-image.
    pub background_image: Option<std::sync::Arc<EmbeddedImage>>,
    /// Resolved `background-size`. `None` = not set (natural size).
    /// See [`BgSize`] for the semantic variants including `cover`/`contain`.
    pub background_size: Option<BgSize>,
    /// `background-position` (x, y). Offsets a no-repeat background image
    /// within the box, clipped to it — the CSS sprite mechanism.
    pub background_position: Option<(BgPos, BgPos)>,
    /// `object-fit` — controls how a replaced element's content (an
    /// `<img>` mostly) fills the layout box when the two have different
    /// aspect ratios. `None` means the default (`fill` — stretch to
    /// box, which is what the painter does when this is unset).
    pub object_fit: Option<ObjectFit>,
    /// `object-position` (x, y). Like `background-position`, each axis
    /// is either an absolute px offset from the content-box origin or a
    /// percentage of `(box_extent - image_extent)`. `None` means the CSS
    /// default: 50% 50% (centre). Used by the painter together with
    /// `object_fit` to position the scaled image inside the content box.
    pub object_position: Option<(BgPos, BgPos)>,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ObjectFit {
    /// Stretch to fully fill the box, ignoring aspect ratio. The
    /// default and what we always did before this property landed.
    Fill,
    /// Scale uniformly so the image fits entirely inside the box.
    /// May leave letterbox bands; the painter centres the image.
    Contain,
    /// Scale uniformly so the image covers the box entirely. May
    /// crop on one axis; the painter centres the image.
    Cover,
    /// Use the image's intrinsic size unchanged. May overflow.
    None,
    /// Smaller of `none` or `contain`.
    ScaleDown,
}

#[derive(Debug)]
pub struct EmbeddedImage {
    pub width: u32,
    pub height: u32,
    pub pixels: Vec<u32>,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Display {
    Block,
    Inline,
    InlineBlock,
    Flex,
    /// `display: inline-flex` — inline-level flex container (shrink-wraps;
    /// lays out children as flex items).
    InlineFlex,
    Grid,
    /// `display: inline-grid` — inline-level grid container (shrink-wraps;
    /// lays out children as grid items).
    InlineGrid,
    /// `display: table` — creates a table-level box. Children with
    /// display: table-row participate in row-major layout via the
    /// automatic-width column algorithm in `place_table`.
    Table,
    /// `display: inline-table` — inline-level table container (shrinks to
    /// content width; lays out children as table items; participates in
    /// inline flow with surrounding text and inline boxes).
    InlineTable,
    /// `display: table-row` — direct children should be cells.
    TableRow,
    /// `display: table-cell` — equivalent to a flow-root block when
    /// its parent isn't a table; inside a table it's just a cell.
    TableCell,
    /// `display: table-row-group` (`<tbody>`/`<thead>`/`<tfoot>`) —
    /// transparent; we look through it to find rows.
    TableRowGroup,
    None,
}

/// CSS `position`. We honour `relative` (offset after normal flow);
/// `absolute`/`fixed`/`sticky` are treated like `relative` for V1 —
/// they still take their original flow slot, but `top`/`left` shift
/// them. Out-of-flow positioning lands later.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
pub enum Position {
    #[default]
    Static,
    Relative,
    Absolute,
    Fixed,
    Sticky,
}

/// CSS `overflow` per axis — see `cv_css::properties::Overflow`. Mirrored
/// into the layout/paint layer so the painter (which has no cv_css
/// dependency) can decide between "clip to padding box" (`Hidden`/`Clip`)
/// and "clip + translate by an independent scroll offset"
/// (`Scroll`/`Auto`). `Visible` does neither.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
pub enum Overflow {
    #[default]
    Visible,
    Hidden,
    Clip,
    Scroll,
    Auto,
}

impl Overflow {
    /// True when this value clips overflow to the padding box.
    pub fn clips(self) -> bool {
        !matches!(self, Self::Visible)
    }
    /// True when this value establishes an independently scrollable region.
    pub fn is_scrollable(self) -> bool {
        matches!(self, Self::Scroll | Self::Auto)
    }
}

/// CSS `background-repeat`. `Repeat` is the spec default and tiles the
/// background bitmap to fill the box. `NoRepeat` paints once at the
/// bitmap's natural size at the top-left of the padding box. `RepeatX`
/// / `RepeatY` tile along a single axis only.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
pub enum BackgroundRepeat {
    #[default]
    Repeat,
    NoRepeat,
    RepeatX,
    RepeatY,
}

/// One axis of `background-position`: an absolute px offset or a
/// percentage of `(box_size - image_size)`. Resolved to a pixel offset
/// by the painter, which knows both the box and the decoded image size.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum BgPos {
    Px(f32),
    Pct(f32),
}

impl BgPos {
    /// Resolve to a pixel offset given the box extent and image extent
    /// along this axis. Percentages align the X% point of the image with
    /// the X% point of the box (CSS Backgrounds §3.6).
    pub fn resolve(self, box_extent: f32, img_extent: f32) -> f32 {
        match self {
            BgPos::Px(v) => v,
            BgPos::Pct(p) => (box_extent - img_extent) * p / 100.0,
        }
    }
}

/// CSS `float` direction. `None` means "not floated"; `Left`/`Right` take
/// the element out of normal flow and pin it to the named side of its
/// containing block, with subsequent in-flow blocks shrinking to wrap.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
pub enum FloatSide {
    #[default]
    None,
    Left,
    Right,
}

/// CSS `clear`. Forces an element below the bottom of every active
/// float on the named side(s).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
pub enum ClearMode {
    #[default]
    None,
    Left,
    Right,
    Both,
}

/// CSS `vertical-align` (keyword form). Only `Sub` / `Super` are
/// applied as offsets in V1; the rest are placeholders so the cascade
/// can store the value losslessly even if we don't honour it visually.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
pub enum VerticalAlign {
    #[default]
    Baseline,
    Sub,
    Super,
    Top,
    Middle,
    Bottom,
    TextTop,
    TextBottom,
}

/// A column or row track in a CSS Grid container.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum AutoRepeatMode {
    Fit,
    Fill,
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum AutoRepeatTrack {
    Px(f32),
    Pct(f32),
    Fr(f32),
    Auto,
}

#[derive(Clone, Debug, PartialEq)]
pub struct AutoRepeat {
    pub mode: AutoRepeatMode,
    pub min_px: f32,
    pub tracks: Vec<AutoRepeatTrack>,
}

/// Track-size bound used by `MinMax { min, max }`. Chrome models the
/// full grammar (`length | percentage | fr | auto | min-content |
/// max-content`) here; we keep the V1 subset that real sites actually
/// use through Tailwind / Bootstrap utility classes.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum MinMaxBound {
    Px(f32),
    Pct(f32),
    Fr(f32),
    Auto,
}

#[derive(Clone, Debug, PartialEq)]
pub enum GridTrack {
    Px(f32),
    Pct(f32),
    /// Flexible fraction. `1fr` → `Fr(1.0)`.
    Fr(f32),
    /// `auto` — for V1 sized like `Fr(1.0)` so the row still fills.
    Auto,
    AutoRepeat(AutoRepeat),
    /// `subgrid` — inherits tracks from parent grid (V1: sized like Auto).
    Subgrid,
    /// `minmax(min, max)` — the Tailwind-canonical track. Reference:
    /// `third_party/blink/renderer/core/layout/grid/grid_track_sizing_algorithm.cc`
    /// (`Sizing` enum). Critical case is `minmax(0, 1fr)` (what
    /// `grid-cols-N` expands to): without it, layout sees `Fr(1.0)`
    /// with an implicit `min=auto` (= min-content), so one column with
    /// long text grows past its 1fr share and shoves the other columns
    /// narrow — visible on dashboards where one card's number is wider
    /// than the others' under our current layout. `minmax(0, 1fr)`
    /// puts a hard floor at 0, letting the column shrink to share.
    MinMax {
        min: MinMaxBound,
        max: MinMaxBound,
    },
}

/// CSS `flex-direction`.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
pub enum FlexDirection {
    #[default]
    Row,
    Column,
    RowReverse,
    ColumnReverse,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
pub enum FlexWrap {
    #[default]
    NoWrap,
    Wrap,
    /// Wrap lines placed in reverse cross-axis order (last line first).
    WrapReverse,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
pub enum JustifyContent {
    #[default]
    Start,
    End,
    Center,
    SpaceBetween,
    SpaceAround,
    SpaceEvenly,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
pub enum AlignItems {
    #[default]
    Stretch,
    Start,
    End,
    Center,
    /// Align on the first text baseline. Uses `LayoutBox::baseline_y`
    /// when present; falls back to `Start` otherwise.
    Baseline,
}

/// A width/height spec that can be resolved against an actual container
/// width (or viewport height) at place-time. Lets `width: 50%` come out
/// correct for nested elements instead of always anchoring to viewport.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct CalcLengthSpec {
    pub px: f32,
    pub pct: f32,
}

impl CalcLengthSpec {
    pub fn resolve(self, container: f32) -> f32 {
        self.px + container * self.pct / 100.0
    }
    pub fn has_percent(self) -> bool {
        self.pct != 0.0
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct ClampLengthSpec {
    pub min: CalcLengthSpec,
    pub preferred: CalcLengthSpec,
    pub max: CalcLengthSpec,
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum LengthSpec {
    Px(f32),
    Pct(f32),
    Calc(CalcLengthSpec),
    Clamp(ClampLengthSpec),
}

impl LengthSpec {
    pub fn resolve(self, container: f32) -> f32 {
        match self {
            Self::Px(v) => v,
            Self::Pct(p) => container * p / 100.0,
            Self::Calc(calc) => calc.resolve(container),
            Self::Clamp(clamp) => {
                // CSS clamp(MIN, VAL, MAX) = max(MIN, min(VAL, MAX)).
                // Computed directly (not via f32::clamp, which panics
                // when MIN > MAX) so inverted bounds — including the
                // synthesized clamps that back percent-bearing
                // `min()`/`max()` — resolve per spec instead of
                // crashing: when MIN > MAX the spec yields MIN.
                let lo = clamp.min.resolve(container);
                let hi = clamp.max.resolve(container);
                let v = clamp.preferred.resolve(container);
                v.min(hi).max(lo)
            }
        }
    }

    /// True when this length depends on the containing block size — i.e.
    /// percentages or calc/clamp expressions that reference percent.
    /// Used to skip max-height/min-height % when the parent's height is
    /// indefinite, matching CSS spec (treat as `none` in that case).
    pub fn is_percent_based(self) -> bool {
        match self {
            Self::Px(_) => false,
            Self::Pct(_) => true,
            Self::Calc(c) => c.has_percent(),
            Self::Clamp(c) => {
                c.preferred.has_percent() || c.min.has_percent() || c.max.has_percent()
            }
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum TextAlign {
    Left,
    Center,
    Right,
    Justify,
}

/// CSS Text 3 §2.1 — `text-transform`. Applied at text-bake time
/// to the glyph string the painter sees. Inherits by default.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum TextTransform {
    None,
    Uppercase,
    Lowercase,
    Capitalize,
}

/// Callback signature for real text measurement. Args: text, font_size_px,
/// font_family (None = default), bold, italic. Returns intrinsic pixel width
/// of the run. When `None` on `LayoutConfig`, layout falls back to the legacy
/// `chars × font_size × 0.55` heuristic (which causes wrong line breaks and
/// wrong table column widths on real pages). The browser sets this to a
/// closure that calls `cv_ui::measure_text_px` (GDI GetTextExtentPoint32W).
pub type MeasureTextFn = std::rc::Rc<dyn Fn(&str, f32, Option<&str>, bool, bool) -> f32 + 'static>;

thread_local! {
    /// Active text measurer for the current layout pass. Set by
    /// `layout_root` at entry from `cfg.measure_text_fn`, cleared on exit.
    /// The intrinsic-width helpers consult this without having to thread
    /// `cfg` through every recursive signature.
    static MEASURE_TEXT_FN: std::cell::RefCell<Option<MeasureTextFn>> =
        const { std::cell::RefCell::new(None) };
}

/// Returns true for the five CSS collapsible-whitespace characters:
/// space (U+0020), tab (U+0009), LF (U+000A), CR (U+000D), FF (U+000C).
///
/// U+00A0 (NON-BREAKING SPACE / `&nbsp;`) is intentionally NOT included —
/// it is a visible, non-collapsible character per CSS Text §4.
#[inline]
fn is_css_collapsible_ws(ch: char) -> bool {
    matches!(ch, ' ' | '\t' | '\n' | '\r' | '\x0C')
}

/// Normalise CSS `white-space: normal` text:
/// - collapse runs of CSS-collapsible whitespace to a single ASCII space
/// - strip leading/trailing CSS-collapsible whitespace
/// - **preserve** U+00A0 (NBSP) verbatim — it is non-collapsible
///
/// This replaces `str::split_whitespace().join(" ")`, which treats NBSP
/// as Unicode whitespace and collapses it away (Chrome-divergence §nbsp).
fn css_normalize_whitespace(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut pending_space = false;
    let mut started = false;
    for ch in s.chars() {
        if is_css_collapsible_ws(ch) {
            if started {
                pending_space = true;
            }
        } else {
            if pending_space {
                out.push(' ');
                pending_space = false;
            }
            out.push(ch);
            started = true;
        }
    }
    out
}

/// Split text into tokens on CSS collapsible whitespace only.
/// U+00A0 (NBSP) is **not** a split point — it binds two tokens together
/// (prevents word-breaking), matching Chrome's behaviour.
fn css_split_whitespace(s: &str) -> impl Iterator<Item = &str> {
    s.split(is_css_collapsible_ws).filter(|w| !w.is_empty())
}

/// Apply CSS `text-transform` to a string the same way the paint path
/// does, so layout MEASURES the glyphs that will actually be drawn. Chrome
/// applies the transform before measuring; we did not, so an `uppercase`
/// run (e.g. Orbitron table headers "Block"→"BLOCK") was measured narrow
/// from the lower-case source and then clipped at paint because the
/// painted CAPS are wider.
fn apply_text_transform(text: &str, tt: Option<TextTransform>) -> std::borrow::Cow<'_, str> {
    use std::borrow::Cow;
    match tt {
        Some(TextTransform::Uppercase) => Cow::Owned(text.to_uppercase()),
        Some(TextTransform::Lowercase) => Cow::Owned(text.to_lowercase()),
        Some(TextTransform::Capitalize) => {
            // Per CSS Text 3 §5: capitalize uppercases the FIRST TYPOGRAPHIC
            // LETTER of each word, where punctuation between words counts as
            // a word boundary. The previous impl only treated whitespace as
            // a boundary — so leading `"`/`(`/`-`/`'` made the next letter
            // stay lowercase (audit-flagged: a sentence opening with `"foo
            // bar"` would render as `"foo Bar"` instead of `"Foo Bar"`).
            let mut out = String::with_capacity(text.len());
            let mut at_word_start = true;
            for ch in text.chars() {
                if at_word_start && ch.is_alphabetic() {
                    at_word_start = false;
                    out.extend(ch.to_uppercase());
                } else {
                    // Any non-alphabetic character — whitespace,
                    // punctuation, digits, symbols — places us BEFORE the
                    // next typographic letter. Spec lists Unicode general
                    // categories L (letter); everything else is a separator
                    // for capitalisation purposes. Use is_alphabetic as a
                    // close approximation (covers L*) without dragging in
                    // a UCD lookup.
                    if !ch.is_alphabetic() {
                        at_word_start = true;
                    }
                    out.push(ch);
                }
            }
            Cow::Owned(out)
        }
        _ => Cow::Borrowed(text),
    }
}

fn measure_text_global(text: &str, fs: f32, family: Option<&str>, bold: bool, italic: bool) -> f32 {
    MEASURE_TEXT_FN.with(|c| {
        if let Some(f) = c.borrow().as_ref() {
            return f(text, fs, family, bold, italic);
        }
        text.chars().count() as f32 * fs * 0.55
    })
}

pub struct LayoutConfig {
    pub viewport_w: f32,
    pub viewport_h: f32,
    pub default_font_size_px: f32,
    pub default_text_color: Color,
    pub default_line_height: f32,
    /// Real text-measurement callback. When `None`, layout uses the
    /// legacy `chars × 0.55em` heuristic.
    pub measure_text_fn: Option<MeasureTextFn>,
}

impl LayoutConfig {
    /// Measure a text run in CSS pixels. Uses the configured callback
    /// (real GDI metrics) when present, otherwise falls back to the
    /// 0.55em-per-char heuristic that the engine used before Chrome
    /// parity work.
    pub fn measure_text(
        &self,
        text: &str,
        font_size_px: f32,
        font_family: Option<&str>,
        bold: bool,
        italic: bool,
    ) -> f32 {
        if let Some(f) = &self.measure_text_fn {
            return f(text, font_size_px, font_family, bold, italic);
        }
        text.chars().count() as f32 * font_size_px * 0.55
    }
    /// Cheap manual clone — LayoutConfig is plain Copy data plus a Color
    /// that's also Copy, so this is just field-wise duplication. Used by
    /// the browser to hand the same config to multiple closures (nav,
    /// tick) without having to derive Clone for an enum-bearing struct.
    pub fn clone_for_runtime(&self) -> Self {
        Self {
            viewport_w: self.viewport_w,
            viewport_h: self.viewport_h,
            default_font_size_px: self.default_font_size_px,
            default_text_color: self.default_text_color,
            default_line_height: self.default_line_height,
            measure_text_fn: self.measure_text_fn.clone(),
        }
    }
}

impl fmt::Debug for LayoutConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LayoutConfig")
            .field("viewport_w", &self.viewport_w)
            .field("viewport_h", &self.viewport_h)
            .field("default_font_size_px", &self.default_font_size_px)
            .field("default_line_height", &self.default_line_height)
            .finish()
    }
}

impl Default for LayoutConfig {
    fn default() -> Self {
        Self {
            viewport_w: 1024.0,
            viewport_h: 768.0,
            default_font_size_px: 16.0,
            default_text_color: Color {
                r: 0,
                g: 0,
                b: 0,
                a: 255,
            },
            default_line_height: 1.4,
            measure_text_fn: None,
        }
    }
}

// ===== Incremental layout (LayoutNG-style fragment cache) =====
// A clean box whose ConstraintSpace (placement position + container sizes +
// parent font-size) is unchanged reuses its cached placed fragment AS-IS — no
// translation, so there's no `shift_box` position-completeness hazard. Active
// only when a host installs a cache handle via `set_layout_cache` (the CV_DOM
// render path); otherwise every box lays out normally (legacy/tests). Bails
// entirely when the tree contains floats (the incoming float band is a hidden
// ConstraintSpace input not captured by the key). Gated correct by the
// `incremental_layout_matches_full` oracle and the runtime `CV_LAYOUT_VERIFY`
// differential check (incremental == full).
//
// NOTE on position-independence (M2.4 improvement (a), evaluated + REVERTED):
// dropping `(x, y)` from the key and `shift_box`-ing the stored origin-relative
// fragment on reuse was implemented and oracle-tested. It made a shifted-but-
// unchanged subtree a cache HIT, but the `CV_LAYOUT_VERIFY` oracle caught a
// sub-ULP (≈0.0000076px) height divergence on shifted text reuse: a freshly
// recomputed `font_size_px * line_height` does not reproduce the stored height
// bit-for-bit when an unrelated sibling's box attributes change between frames.
// Per the M2.4 safety contract (keep the oracle byte-identical-green), the
// position-IN-key form is retained: reusing a fragment only at the exact
// position+constraints it was computed under is bit-stable by construction.
// `set_layout_verify` is the safe, additive verification tooling that proved it.

// ConstraintSpace key. Includes the placement origin `(x, y)` so a fragment is
// only ever reused at the EXACT position it was computed under — making reuse
// bit-identical to a fresh layout by construction (the `CV_LAYOUT_VERIFY`
// oracle's byte-identical gate). Dropping `(x, y)` to catch shifted-but-
// unchanged subtrees was tried and reverted (see the section comment above):
// `shift_box`-reuse exposed a sub-ULP height float divergence the gate forbids.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
struct LayoutKey {
    x: u32,
    y: u32,
    cw: u32,
    ch: u32,
    fs: u32,
}
impl LayoutKey {
    fn new(x: f32, y: f32, cw: f32, ch: f32, fs: f32) -> Self {
        Self {
            x: x.to_bits(),
            y: y.to_bits(),
            cw: cw.to_bits(),
            ch: ch.to_bits(),
            fs: fs.to_bits(),
        }
    }
}

pub struct CachedFragment {
    fragment: LayoutBox,
    key: LayoutKey,
    cache_gen: u32,
}

type LayoutCacheHandle =
    std::rc::Rc<std::cell::RefCell<std::collections::HashMap<u64, CachedFragment>>>;

thread_local! {
    static LAYOUT_CACHE: std::cell::RefCell<Option<LayoutCacheHandle>> = const { std::cell::RefCell::new(None) };
    static LAYOUT_CLEAN: std::cell::RefCell<Option<Box<dyn Fn(u64) -> bool>>> = const { std::cell::RefCell::new(None) };
    static LAYOUT_GEN: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
    static LAYOUT_HAS_FLOATS: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    static RELAYOUT_STATS: std::cell::Cell<(u64, u64)> = const { std::cell::Cell::new((0, 0)) };
    /// When set, `place()` neither reads nor writes the fragment cache — every
    /// box lays out from scratch. The differential oracle (`CV_LAYOUT_VERIFY`)
    /// flips this on for its forced-full reference pass while leaving the
    /// installed cache handle untouched.
    static FORCE_NO_CACHE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Install (or clear, with `None`) the incremental-layout session: the
/// persistent cache handle, a "is this NodeId clean (reusable)?" oracle over the
/// host DOM, and the cache generation. conclave calls this before `layout()`
/// on the CV_DOM path; `None` disables the cache (legacy behaviour).
pub fn set_layout_cache(
    handle: Option<LayoutCacheHandle>,
    clean: Option<Box<dyn Fn(u64) -> bool>>,
    generation: u32,
) {
    LAYOUT_CACHE.with(|c| *c.borrow_mut() = handle);
    LAYOUT_CLEAN.with(|c| *c.borrow_mut() = clean);
    LAYOUT_GEN.with(|c| c.set(generation));
}

/// (boxes laid out, boxes reused) since the last `set_layout_cache`.
pub fn relayout_stats() -> (u64, u64) {
    RELAYOUT_STATS.with(|c| c.get())
}

fn subtree_has_floats(b: &LayoutBox) -> bool {
    b.float_side != FloatSide::None || b.children.iter().any(subtree_has_floats)
}

fn layout_cache_active() -> bool {
    !FORCE_NO_CACHE.with(|c| c.get())
        && LAYOUT_CACHE.with(|c| c.borrow().is_some())
        && !LAYOUT_HAS_FLOATS.with(|c| c.get())
}

fn layout_node_clean(id: u64) -> bool {
    LAYOUT_CLEAN.with(|c| c.borrow().as_ref().is_some_and(|f| f(id)))
}

/// Is this box safe to cache/reuse on its own? Only plain in-flow block boxes
/// with a stable identity. Inline/inline-block (line context), table parts,
/// floats, out-of-flow (absolute/fixed), flex items (forced sizes), and
/// flex/grid/table children (`cache_ineligible`) all have layout inputs the
/// position-keyed cache doesn't capture, so they're excluded (and captured
/// inside their container's cached fragment instead).
fn box_cacheable(b: &LayoutBox) -> bool {
    b.node_id.is_some()
        && !b.cache_ineligible
        && !b.is_inline
        && !b.is_inline_block
        && !b.is_table
        && !b.is_table_row
        && !b.is_table_cell
        && !b.is_table_row_group
        && b.float_side == FloatSide::None
        && b.flex_override_width.is_none()
        && !matches!(b.position, Position::Absolute | Position::Fixed)
}

/// Caching wrapper around `place_inner`: reuse a clean box's cached fragment
/// when its ConstraintSpace is unchanged, else lay out and cache the result.
#[allow(clippy::too_many_arguments)]
fn place(
    b: &mut LayoutBox,
    x: f32,
    y: f32,
    container_w: f32,
    container_h: f32,
    ctx: &mut LayoutCtx<'_>,
    parent_font_size: f32,
) {
    if layout_cache_active() && box_cacheable(b) {
        let id = b.node_id.unwrap();
        if layout_node_clean(id) {
            let key = LayoutKey::new(x, y, container_w, container_h, parent_font_size);
            let cur_gen = LAYOUT_GEN.with(|c| c.get());
            let hit = LAYOUT_CACHE.with(|c| {
                c.borrow().as_ref().and_then(|h| {
                    h.borrow()
                        .get(&id)
                        .filter(|e| e.cache_gen == cur_gen && e.key == key)
                        .map(|e| e.fragment.clone())
                })
            });
            if let Some(frag) = hit {
                // Same id + same ConstraintSpace (which here INCLUDES the (x, y)
                // origin) ⇒ reuse the fragment exactly as it was laid out, with no
                // translation. Bit-identical to a fresh place() by construction.
                *b = frag;
                RELAYOUT_STATS.with(|c| {
                    let (l, r) = c.get();
                    c.set((l, r + 1));
                });
                return;
            }
        }
    }
    RELAYOUT_STATS.with(|c| {
        let (l, r) = c.get();
        c.set((l + 1, r));
    });
    place_inner(b, x, y, container_w, container_h, ctx, parent_font_size);
    if layout_cache_active() && box_cacheable(b) {
        let id = b.node_id.unwrap();
        let key = LayoutKey::new(x, y, container_w, container_h, parent_font_size);
        let cur_gen = LAYOUT_GEN.with(|c| c.get());
        LAYOUT_CACHE.with(|c| {
            if let Some(h) = c.borrow().as_ref() {
                h.borrow_mut().insert(
                    id,
                    CachedFragment {
                        fragment: b.clone(),
                        key,
                        cache_gen: cur_gen,
                    },
                );
            }
        });
    }
}

thread_local! {
    /// Programmatic override for the differential oracle (tests / hosts that
    /// want to arm it without an env var). `None` ⇒ fall back to the
    /// `CV_LAYOUT_VERIFY` env var (read once, cached).
    static LAYOUT_VERIFY_OVERRIDE: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
    static LAYOUT_VERIFY_ENV: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
}

/// Arm (`Some(true)`) / disarm (`Some(false)`) / defer-to-env (`None`) the
/// differential layout oracle programmatically, per thread. Used by the oracle
/// integration test so it doesn't depend on env-var timing.
pub fn set_layout_verify(on: Option<bool>) {
    LAYOUT_VERIFY_OVERRIDE.with(|c| c.set(on));
}

/// Is the differential layout oracle armed? `CV_LAYOUT_VERIFY=1` (or
/// `set_layout_verify(Some(true))`) makes every `layout()` that runs with an
/// active fragment cache ALSO perform a forced from-scratch re-layout of the
/// same tree and assert the two are geometry-identical (a hard panic naming the
/// divergent node otherwise). This is the safety net that proves no incremental
/// skip ever serves a stale fragment. The env var is read once and cached
/// because env reads are not free.
fn layout_verify_enabled() -> bool {
    if let Some(v) = LAYOUT_VERIFY_OVERRIDE.with(|c| c.get()) {
        return v;
    }
    LAYOUT_VERIFY_ENV.with(|c| {
        if let Some(v) = c.get() {
            return v;
        }
        let v = std::env::var("CV_LAYOUT_VERIFY")
            .map(|s| s == "1" || s.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        c.set(Some(v));
        v
    })
}

/// Walk two laid-out box trees in lockstep, asserting every box's geometry
/// (content / padding / border-widths / margin rects) is bit-identical. Returns
/// `Err(path)` naming the first divergent node (its tree path + the two rects)
/// on a mismatch, `Ok(())` when the whole trees agree. This is the oracle's
/// comparator — `incremental == full` over every Rect, in tree order.
fn geometry_equal(inc: &LayoutBox, full: &LayoutBox, path: &str) -> Result<(), String> {
    // Compare on raw bits so a NaN-vs-NaN or -0.0-vs-0.0 divergence is caught;
    // a stale fragment with a different rect is the failure we exist to detect.
    fn rect_bits(r: &Rect) -> (u32, u32, u32, u32) {
        (r.x.to_bits(), r.y.to_bits(), r.w.to_bits(), r.h.to_bits())
    }
    fn edges_bits(e: &EdgeSizes) -> (u32, u32, u32, u32) {
        (
            e.top.to_bits(),
            e.right.to_bits(),
            e.bottom.to_bits(),
            e.left.to_bits(),
        )
    }
    if rect_bits(&inc.content) != rect_bits(&full.content) {
        return Err(format!(
            "{path}: content rect diverged: incremental={:?} full={:?} (node_id={:?})",
            inc.content, full.content, inc.node_id
        ));
    }
    if edges_bits(&inc.padding) != edges_bits(&full.padding) {
        return Err(format!(
            "{path}: padding diverged: incremental={:?} full={:?} (node_id={:?})",
            inc.padding, full.padding, inc.node_id
        ));
    }
    if edges_bits(&inc.margin) != edges_bits(&full.margin) {
        return Err(format!(
            "{path}: margin diverged: incremental={:?} full={:?} (node_id={:?})",
            inc.margin, full.margin, inc.node_id
        ));
    }
    if inc.border_width_px.to_bits() != full.border_width_px.to_bits() {
        return Err(format!(
            "{path}: border width diverged: incremental={} full={} (node_id={:?})",
            inc.border_width_px, full.border_width_px, inc.node_id
        ));
    }
    if inc.children.len() != full.children.len() {
        return Err(format!(
            "{path}: child count diverged: incremental={} full={} (node_id={:?})",
            inc.children.len(),
            full.children.len(),
            inc.node_id
        ));
    }
    for (i, (ci, cf)) in inc.children.iter().zip(full.children.iter()).enumerate() {
        geometry_equal(ci, cf, &format!("{path}/{i}"))?;
    }
    Ok(())
}

pub fn layout(root: &StyledNode, cfg: &LayoutConfig) -> LayoutBox {
    // Incremental (or, with no cache installed, plain full) layout.
    let bx = layout_once(root, cfg);

    // Differential oracle: when armed AND a fragment cache is actually in play
    // (legacy/test paths with no cache produce nothing to verify), re-lay-out the
    // SAME tree from scratch with caching forced off and assert bit-identical
    // geometry. A divergence is under-invalidation — a stale fragment — and is a
    // hard, named panic. We snapshot/restore the real incremental relayout
    // counters around the reference pass so `relayout_stats()` still reports the
    // genuine incremental work, not the forced-full pass.
    if layout_verify_enabled() && LAYOUT_CACHE.with(|c| c.borrow().is_some()) {
        let inc_stats = RELAYOUT_STATS.with(|c| c.get());
        let full = FORCE_NO_CACHE.with(|c| {
            c.set(true);
            let r = layout_once(root, cfg);
            c.set(false);
            r
        });
        RELAYOUT_STATS.with(|c| c.set(inc_stats));
        if let Err(msg) = geometry_equal(&bx, &full, "root") {
            panic!("CV_LAYOUT_VERIFY: incremental layout diverged from full re-layout:\n  {msg}");
        }
    }
    bx
}

/// One full layout pass over `root`: build the box tree, place it, then run the
/// post-place re-anchoring passes. When the fragment cache is installed and not
/// force-disabled, clean unchanged subtrees reuse their cached fragment.
fn layout_once(root: &StyledNode, cfg: &LayoutConfig) -> LayoutBox {
    // Install the per-pass text measurer so the intrinsic-width and
    // word-wrap helpers can call real font metrics without having to
    // thread `cfg` through every recursive signature. We restore the
    // prior value on exit so re-entrant layout calls (e.g. shadow tree
    // measurement) compose correctly.
    let prev = MEASURE_TEXT_FN.with(|c| c.borrow().clone());
    MEASURE_TEXT_FN.with(|c| *c.borrow_mut() = cfg.measure_text_fn.clone());
    // Fresh intrinsic-width memo tables for this layout pass. Cleared per entry
    // so a box address reused from a previous (freed) layout tree can't hit a
    // stale cached width.
    clear_intrinsic_caches();
    // Fresh relayout counters for this layout pass (reused/laid-out).
    RELAYOUT_STATS.with(|c| c.set((0, 0)));
    let mut bx = build_box(root, cfg);
    // Incremental layout: floats make the float band a hidden ConstraintSpace
    // input, so the fragment cache bails for any tree that contains one. Detect
    // it once on the built tree (before any place()), only when a session is active.
    if LAYOUT_CACHE.with(|c| c.borrow().is_some()) {
        LAYOUT_HAS_FLOATS.with(|c| c.set(subtree_has_floats(&bx)));
    }
    let mut ctx = LayoutCtx { cfg };
    place(
        &mut bx,
        0.0,
        0.0,
        cfg.viewport_w,
        cfg.viewport_h,
        &mut ctx,
        cfg.default_font_size_px,
    );
    // Re-anchor absolutely-positioned boxes to their nearest positioned
    // ancestor now that every content rect is final (the initial
    // containing block is the viewport). Runs before relative offsets so
    // a relative ancestor's shift carries its absolute descendants along.
    let initial_cb = Rect {
        x: 0.0,
        y: 0.0,
        w: cfg.viewport_w,
        h: cfg.viewport_h,
    };
    reposition_absolutes(&mut bx, initial_cb);
    apply_positioned_offsets(&mut bx);
    // CSS `transform: scale(...)` is a visual effect applied after layout
    // — bake it into the final geometry so paint renders scaled boxes,
    // text, and borders. (Translate stays a paint offset; rotate/skew
    // need the affine compositor path.)
    apply_visual_transforms(&mut bx);
    MEASURE_TEXT_FN.with(|c| *c.borrow_mut() = prev);
    bx
}

/// Walk the box tree and apply `position: relative` offsets — these
/// shift the rendered position without removing the box from flow,
/// so siblings keep their original slots.
///
/// `position: absolute` and `position: fixed` are NOT handled here —
/// those got placed against their containing block during the
/// out-of-flow pass inside `place()`. Re-shifting them here would
/// double-apply the offset.
fn apply_positioned_offsets(b: &mut LayoutBox) {
    // `position: sticky` falls back to `relative` for V1. Real sticky
    // depends on scroll state, which we don't thread through layout
    // yet — but relative-shift gives a sane default (the box ends up
    // at its specified offset within its parent), which is closer to
    // Chrome than the static-default-no-shift fallback we had before.
    let is_relative = matches!(b.position, Position::Relative | Position::Sticky);
    let has_offset =
        b.top_px.is_some() || b.left_px.is_some() || b.right_px.is_some() || b.bottom_px.is_some();
    if is_relative && has_offset {
        // `position: relative` offsets resolve against the box's
        // containing block. We don't carry that here, so fall back
        // to the box's own content width/height as the % anchor —
        // close enough for the common centred-card use-case, and a
        // strict improvement over treating % as 0px.
        let cb_w = b.content.w.max(0.0);
        let cb_h = b.content.h.max(0.0);
        let dx = b
            .left_px
            .map(|s| s.resolve(cb_w))
            .unwrap_or_else(|| -b.right_px.map(|s| s.resolve(cb_w)).unwrap_or(0.0));
        let dy = b
            .top_px
            .map(|s| s.resolve(cb_h))
            .unwrap_or_else(|| -b.bottom_px.map(|s| s.resolve(cb_h)).unwrap_or(0.0));
        shift_box(b, dx, dy);
    }
    for c in &mut b.children {
        apply_positioned_offsets(c);
    }
}

/// True if this box participates in normal flow. `position: absolute`
/// and `position: fixed` boxes are out-of-flow — they're skipped by
/// block / flex / grid sizing passes and placed separately against
/// their containing block.
fn is_in_flow(b: &LayoutBox) -> bool {
    !matches!(b.position, Position::Absolute | Position::Fixed)
}

/// A box that establishes a containing block for `position: absolute`
/// descendants — any non-static positioning. (CSS Position §4: the
/// containing block of an absolutely-positioned box is the padding
/// box of the nearest ancestor with `position != static`.)
fn establishes_abs_containing_block(p: Position) -> bool {
    matches!(
        p,
        Position::Relative | Position::Absolute | Position::Fixed | Position::Sticky
    )
}

/// The border-box origin `place_absolute` would assign to `child`
/// against containing block `cb` — mirrors its top/left (and
/// right/bottom fallback) offset resolution exactly.
fn abs_origin(child: &LayoutBox, cb: &Rect) -> (f32, f32) {
    let dx = child
        .left_px
        .map(|s| s.resolve(cb.w))
        .unwrap_or_else(|| match child.right_px {
            Some(r) => cb.w - r.resolve(cb.w),
            None => 0.0,
        });
    let dy = child
        .top_px
        .map(|s| s.resolve(cb.h))
        .unwrap_or_else(|| match child.bottom_px {
            Some(bo) => cb.h - bo.resolve(cb.h),
            None => 0.0,
        });
    (cb.x + dx, cb.y + dy)
}

/// Re-anchor `position: absolute` boxes to their *nearest positioned
/// ancestor* once every content rect is final. During layout,
/// `place_absolute` provisionally positions each out-of-flow child
/// against its immediate parent's content rect (which is all the
/// recursive pass can see). That is correct only when the parent is
/// itself positioned; when an absolute box sits inside one or more
/// static ancestors, its real containing block is a positioned box
/// further up. This post-pass translates each absolute subtree by the
/// delta between "anchored to parent" and "anchored to the nearest
/// positioned ancestor", so the common `.relative > div > .absolute`
/// nesting resolves correctly. `position: fixed` already anchors to
/// the viewport and is left alone.
fn reposition_absolutes(b: &mut LayoutBox, abs_cb: Rect) {
    // Containing block seen by *this* box's absolute children/descendants.
    // Per CSS Position §4, the containing block of an absolutely-positioned
    // element is the *padding edge* (padding box) of the nearest positioned
    // ancestor, not the content box. Use padding_rect() here.
    let child_cb = if establishes_abs_containing_block(b.position) {
        b.padding_rect()
    } else {
        abs_cb
    };
    // `place_absolute` now positions children against `b.padding_rect()`,
    // so that is the "original anchor" we must compare against.
    let parent_pad = b.padding_rect();
    for child in &mut b.children {
        if matches!(child.position, Position::Absolute) && child_cb != parent_pad {
            let (ox, oy) = abs_origin(child, &parent_pad);
            let (nx, ny) = abs_origin(child, &child_cb);
            shift_box(child, nx - ox, ny - oy);
        }
    }
    for child in &mut b.children {
        reposition_absolutes(child, child_cb);
    }
}

/// How a child contributes to its parent's normal flow.
#[derive(Copy, Clone, Debug)]
enum ChildKind {
    /// `position: absolute` or `position: fixed`. Doesn't take any
    /// space in the parent's flow; placed against the containing
    /// block in a second pass.
    OutOfFlow,
    /// `display: block` (the default). Each block child stacks on
    /// its own line, full container width.
    Block,
    /// `display: inline` or `display: inline-block`. Joins the
    /// running horizontal line; the parent's `place_inline_run`
    /// wraps to a new line when the running width overflows.
    Inline,
    /// `float: left` or `float: right`. Pinned to the named edge of
    /// the parent's content rect; subsequent in-flow blocks shrink
    /// their effective width to wrap around it until they clear.
    Float,
}

fn inline_container_children_all_inline(b: &LayoutBox) -> bool {
    b.children
        .iter()
        .filter(|child| is_in_flow(child))
        .all(|child| child.is_inline || matches!(child.kind, BoxKind::Text(_)))
}

/// Trim the parent's horizontal content rect by the widths of every
/// active float on each side. Used by the block-flow loop to compute
/// the effective `(x, w)` for each next in-flow child.
fn effective_band(content_x: f32, content_w: f32, floats: &[(FloatSide, f32, f32)]) -> (f32, f32) {
    let mut left = 0.0_f32;
    let mut right = 0.0_f32;
    for (side, _bottom, w) in floats {
        match side {
            FloatSide::Left => left = left.max(*w),
            FloatSide::Right => right = right.max(*w),
            FloatSide::None => {}
        }
    }
    let w = (content_w - left - right).max(0.0);
    (content_x + left, w)
}

/// Drop floats whose bottom edge is at or above `child_y` — those no
/// longer affect subsequent in-flow content.
fn sweep_floats(child_y: f32, floats: &mut Vec<(FloatSide, f32, f32)>) {
    floats.retain(|(_, bottom, _)| *bottom > child_y);
}

/// CSS `clear:` — push `child_y` past the bottom of every active
/// float on the named side(s). Returns the (possibly bumped) y.
fn apply_clear(child_y: f32, clear: ClearMode, floats: &[(FloatSide, f32, f32)]) -> f32 {
    if matches!(clear, ClearMode::None) {
        return child_y;
    }
    let mut y = child_y;
    for (side, bottom, _) in floats {
        let matches_side = match clear {
            ClearMode::Left => matches!(side, FloatSide::Left),
            ClearMode::Right => matches!(side, FloatSide::Right),
            ClearMode::Both => !matches!(side, FloatSide::None),
            ClearMode::None => false,
        };
        if matches_side && *bottom > y {
            y = *bottom;
        }
    }
    y
}

/// Lay out a run of consecutive inline / inline-block siblings on
/// horizontal lines, wrapping when the running width would exceed
/// `content_w`. Returns the total vertical space consumed so the
/// caller's block-level cursor can advance past the run.
///
/// V1 simplification: each inline child gets its natural width
/// from a `place(child, ...)` measurement, with no shrink behaviour
/// (a too-wide inline child gets its own line). Inline text nodes
/// inside the run are joined into the parent's flatten-inline
/// pass already, so this primarily orchestrates inline-block
/// siblings.
fn place_inline_run(
    children: &mut [LayoutBox],
    content_x: f32,
    content_y: f32,
    content_w: f32,
    container_h: f32,
    ctx: &mut LayoutCtx<'_>,
    parent_fs: f32,
    // CSS `text-align` of the parent block.  Centers/right-aligns each
    // completed inline line within `content_w`.  This is the rule that
    // puts Google's Doodle (an inline `<img>`) in the middle of its
    // 736-wide centering wrapper instead of glued to the left edge,
    // and works for any other "centered inline content" idiom.
    parent_text_align: Option<TextAlign>,
) -> f32 {
    if children.is_empty() {
        return 0.0;
    }
    // First pass: natural-size each child AND compute its baseline
    // offset (distance from the top of the child's margin box down to
    // its first-line baseline). Text uses ascent ≈ 0.8em from the
    // content top, padded for the line-height leading; everything else
    // (image, inline-block, replaced) treats its margin-box bottom as
    // the baseline per CSS §10.8 default `vertical-align: baseline`.
    let mut widths: Vec<f32> = Vec::with_capacity(children.len());
    let mut heights: Vec<f32> = Vec::with_capacity(children.len());
    let mut baselines: Vec<f32> = Vec::with_capacity(children.len());
    for child in children.iter_mut() {
        // CSS inline boxes should natural-size to their *content's*
        // intrinsic width, not to the full container. Sizing each one
        // against the container makes every text node wrap to as many
        // lines as it has chars, then the packer treats each child as
        // wider-than-container and stacks them all vertically — turning
        // a real paragraph into a tower of one-word ribbons. Cap the
        // measurement at content_w so a single very long text still
        // wraps inside its own box rather than overflowing the parent.
        // An inline-block with a DECLARED width uses that width (resolved
        // against the real containing block), not its shrink-to-fit
        // intrinsic. `intrinsic_margin_width` treats a percentage width as
        // auto (correct for min/max-content sizing) — but that's the wrong
        // *used* width for a sized inline-block. Wikipedia's search box is
        // the canonical case: `.search-input { display:inline-block;
        // width:73% }` must become 73% of the form, and the `width:100%`
        // input inside it then fills that — otherwise both collapse to
        // ~0. Pass the real container width so `place()` resolves the
        // percentage; fall back to the shrink-to-fit measurement otherwise.
        let measure_w = if child.is_inline_block
            && (child.explicit_width.is_some() || child.flex_override_width.is_some())
        {
            content_w
        } else {
            intrinsic_margin_width(child, content_w, parent_fs)
                .min(content_w)
                .max(0.0)
        };
        place(
            child,
            content_x,
            content_y,
            measure_w,
            container_h,
            ctx,
            parent_fs,
        );
        let mr = child.margin_rect();
        widths.push(mr.w);
        heights.push(mr.h);
        baselines.push(child_baseline_from_top(child, mr.h));
    }
    // Second pass: assign each child to a line and shift into place.
    // Each "line" tracks the maximum baseline-from-top (the ascent
    // height) and the maximum below-baseline depth (descent). The
    // line's height = ascent + descent, and every child gets a dy
    // that puts its own baseline on the line's baseline.
    let mut line_y = content_y;
    let mut cursor_x = content_x;
    let mut line_ascent: f32 = 0.0;
    let mut line_descent: f32 = 0.0;
    let mut line_start = 0usize;
    fn flush_line(
        children: &mut [LayoutBox],
        widths: &[f32],
        heights: &[f32],
        baselines: &[f32],
        line_start: usize,
        line_end: usize,
        line_y: f32,
        line_ascent: f32,
        content_x: f32,
        content_w: f32,
        parent_text_align: Option<TextAlign>,
        is_last_line: bool,
    ) {
        // Sum the line's actual used width, then compute a horizontal
        // shift for non-left text-align.  Apply that shift uniformly
        // to every child on the line.
        let line_w: f32 = widths[line_start..line_end].iter().sum();
        let extra = (content_w - line_w).max(0.0);
        // Per CSS 2.1 §9.5: `text-align: justify` distributes extra
        // horizontal space among gaps between in-line items on every
        // line EXCEPT the last line of a paragraph. The last line
        // aligns by `text-align-last` (default: start → left). We
        // approximate `start` as `Left`.
        let (align_dx, justify_per_gap) = match parent_text_align {
            Some(TextAlign::Center) => (extra * 0.5, 0.0),
            Some(TextAlign::Right) => (extra, 0.0),
            Some(TextAlign::Justify) if !is_last_line => {
                let n_items = line_end.saturating_sub(line_start);
                let gaps = n_items.saturating_sub(1) as f32;
                let per_gap = if gaps > 0.0 { extra / gaps } else { 0.0 };
                (0.0, per_gap)
            }
            _ => (0.0, 0.0),
        };
        let _ = content_x;
        // The line box height = ascent + descent; we computed ascent
        // already, descent isn't passed in here so derive from per-child
        // (heights[i] - baselines[i]) max.
        let line_descent_local: f32 = (line_start..line_end)
            .map(|i| (heights[i] - baselines[i]).max(0.0))
            .fold(0.0_f32, f32::max);
        let line_h = line_ascent + line_descent_local;
        for i in line_start..line_end {
            let mr = children[i].margin_rect();
            // For Top/Middle/Bottom (CSS line-box alignment values), we
            // ignore the baseline and align against the line box itself.
            // Text-top / Text-bottom anchor to the parent's font ascent /
            // descent — V1 approximates these with the line box too,
            // which is correct when there's a single font run on the line.
            let dy = match children[i].vertical_align {
                VerticalAlign::Top | VerticalAlign::TextTop => line_y - mr.y,
                VerticalAlign::Bottom | VerticalAlign::TextBottom => {
                    (line_y + line_h - heights[i]) - mr.y
                }
                VerticalAlign::Middle => (line_y + (line_h - heights[i]) * 0.5) - mr.y,
                _ => {
                    // Baseline / Sub / Super flow through the
                    // baseline-aligned placement: child_top = line_y +
                    // line_ascent - child_baseline.
                    let target_top = line_y + line_ascent - baselines[i];
                    target_top - mr.y
                }
            };
            // Justify: stagger each child's x by `per_gap * index_from_start`
            // so gap widths grow uniformly across the line.
            let justify_dx = if justify_per_gap > 0.0 {
                justify_per_gap * ((i - line_start) as f32)
            } else {
                0.0
            };
            shift_box(&mut children[i], align_dx + justify_dx, dy);
        }
    }
    for i in 0..children.len() {
        let w = widths[i];
        let h = heights[i];
        let bl = baselines[i];
        // Wrap when this child doesn't fit on the running line AND
        // we're not at the line start. (A single child wider than
        // the container still gets its own line, no shrink.)
        if cursor_x + w > content_x + content_w && cursor_x > content_x {
            flush_line(
                children,
                &widths,
                &heights,
                &baselines,
                line_start,
                i,
                line_y,
                line_ascent,
                content_x,
                content_w,
                parent_text_align,
                false, // this is NOT the last line — we're wrapping mid-paragraph
            );
            line_y += line_ascent + line_descent;
            line_ascent = 0.0;
            line_descent = 0.0;
            cursor_x = content_x;
            line_start = i;
        }
        // Position child at (cursor_x, line_y) — its top-left. The
        // second-pass `flush_line` corrects dy to baseline-align.
        let mr = children[i].margin_rect();
        let dx = cursor_x - mr.x;
        let dy = line_y - mr.y;
        shift_box(&mut children[i], dx, dy);
        cursor_x += w;
        // The line ascends by max(child ascents) and descends by
        // max(child descents); a tall image without text still keeps
        // its full height because its baseline is at its bottom.
        line_ascent = line_ascent.max(bl);
        line_descent = line_descent.max((h - bl).max(0.0));
    }
    // Flush the last line.
    flush_line(
        children,
        &widths,
        &heights,
        &baselines,
        line_start,
        children.len(),
        line_y,
        line_ascent,
        content_x,
        content_w,
        parent_text_align,
        true, // last line of the paragraph — no justify
    );
    line_y += line_ascent + line_descent;
    (line_y - content_y).max(0.0)
}

/// Baseline offset (top of margin box → baseline) for a single inline-level
/// child. Text uses 80% of its computed `font-size` as the ascent, centred
/// inside the `line-height` slot; everything else takes its margin-box bottom
/// as the baseline. `vertical-align: super` raises the box by 30% of its
/// font-size (the baseline moves down within the box → the box's top moves
/// up relative to the line baseline), and `sub` lowers it by 20%.
fn child_baseline_from_top(b: &LayoutBox, margin_h: f32) -> f32 {
    let base = match &b.kind {
        BoxKind::Text(_) => {
            let fs = b.font_size_px.max(1.0);
            let line_h = b.line_height_px.unwrap_or(fs * 1.2);
            // Leading is split evenly above and below the glyph cap.
            let half_leading = (line_h - fs * 0.95).max(0.0) / 2.0;
            half_leading + fs * 0.8
        }
        _ => margin_h,
    };
    let fs = b.font_size_px.max(1.0);
    match b.vertical_align {
        // Pull the box's baseline DOWN inside the margin box so the
        // flush algorithm shifts the whole box UP relative to the line.
        VerticalAlign::Super => base + fs * 0.3,
        // Push the baseline UP so the box drops below the line baseline.
        VerticalAlign::Sub => (base - fs * 0.2).max(0.0),
        _ => base,
    }
}

thread_local! {
    /// Per-layout memoization for `intrinsic_margin_width` / `min_content_width`,
    /// keyed by (box pointer, parent-font-size bits). After the percentage-width
    /// fix below, these results are container-independent and invariant per box,
    /// so caching collapses the old O(n×depth) re-walk — where every ancestor
    /// recomputed its descendants' intrinsic widths by walking their whole
    /// subtree — into O(n). Cleared at every `layout()` entry so a freed box's
    /// address can't collide with a stale entry from a prior layout.
    static INTRINSIC_MARGIN_CACHE: std::cell::RefCell<std::collections::HashMap<(usize, u32), f32>> =
        std::cell::RefCell::new(std::collections::HashMap::new());
    static MIN_CONTENT_CACHE: std::cell::RefCell<std::collections::HashMap<(usize, u32), f32>> =
        std::cell::RefCell::new(std::collections::HashMap::new());
}

fn clear_intrinsic_caches() {
    INTRINSIC_MARGIN_CACHE.with(|c| c.borrow_mut().clear());
    MIN_CONTENT_CACHE.with(|c| c.borrow_mut().clear());
}

fn intrinsic_margin_width(b: &LayoutBox, container_w: f32, parent_font_size: f32) -> f32 {
    let key = (b as *const LayoutBox as usize, parent_font_size.to_bits());
    if let Some(v) = INTRINSIC_MARGIN_CACHE.with(|c| c.borrow().get(&key).copied()) {
        return v;
    }
    let horiz_chrome =
        b.border_width_left() + b.border_width_right() + b.padding.left + b.padding.right;
    let margin_left = if b.margin_auto.left {
        0.0
    } else {
        b.margin.left
    };
    let margin_right = if b.margin_auto.right {
        0.0
    } else {
        b.margin.right
    };

    // A percentage width/min-width/max-width does NOT contribute to intrinsic
    // (min/max-content) sizing per CSS Sizing 3 §5 — it resolves to auto/none.
    // This both matches Chrome (the old code resolved % against the container,
    // inflating/shrinking intrinsic widths) and makes the result independent of
    // `container_w`, which is what lets us memoize it.
    let mut content_w = match b.explicit_width {
        Some(spec) if !spec.is_percent_based() => {
            let w = spec.resolve(container_w);
            if b.box_sizing_border_box {
                (w - horiz_chrome).max(0.0)
            } else {
                w.max(0.0)
            }
        }
        _ => {
            if let Some(img) = &b.embedded_image {
                img.width as f32
            } else {
                intrinsic_content_width(b, container_w, parent_font_size)
            }
        }
    };

    if let Some(maxw) = b.max_width {
        if !maxw.is_percent_based() {
            let mut cap = maxw.resolve(container_w);
            if b.box_sizing_border_box {
                cap = (cap - horiz_chrome).max(0.0);
            }
            content_w = content_w.min(cap);
        }
    }
    if let Some(minw) = b.min_width {
        if !minw.is_percent_based() {
            let mut floor = minw.resolve(container_w);
            if b.box_sizing_border_box {
                floor = (floor - horiz_chrome).max(0.0);
            }
            content_w = content_w.max(floor);
        }
    }

    let result = content_w + horiz_chrome + margin_left + margin_right;
    INTRINSIC_MARGIN_CACHE.with(|c| c.borrow_mut().insert(key, result));
    result
}

fn intrinsic_content_width(b: &LayoutBox, container_w: f32, parent_font_size: f32) -> f32 {
    match &b.kind {
        BoxKind::Text(t) => {
            let fs = if b.font_size_px > 0.0 {
                b.font_size_px
            } else {
                parent_font_size
            };
            let normalized_lines: Vec<String> = if b.preserve_whitespace {
                t.split('\n').map(|line| line.to_string()).collect()
            } else {
                // white-space:normal — newlines and CSS-collapsible whitespace
                // collapse to a single space; U+00A0 (NBSP) is preserved.
                // See the matching note in `place()`.
                vec![css_normalize_whitespace(t)]
            };
            let family = b.font_family.as_deref();
            let ls = b.letter_spacing_px;
            let tt = b.text_transform;
            normalized_lines
                .iter()
                .map(|line| {
                    let line = apply_text_transform(line, tt);
                    let line = line.as_ref();
                    measure_text_global(line, fs, family, b.font_weight_bold, b.font_style_italic)
                        // CSS `letter-spacing` adds advance after each
                        // glyph; Chrome includes it in the run's measured
                        // width. Without this the box is ~one glyph short
                        // and paint (which DOES apply letter-spacing) spills
                        // the last glyph out, clipping it (Orbitron headers
                        // "BLOCK"→"BLOC", "VALIDATOR"→"VALIDATO").
                        + ls * line.chars().count() as f32
                })
                .fold(0.0, f32::max)
        }
        _ => {
            if let Some(img) = &b.embedded_image {
                return img.width as f32;
            }

            let child_parent_fs = if b.font_size_px > 0.0 {
                b.font_size_px
            } else {
                parent_font_size
            };
            let inflow = b.children.iter().filter(|child| is_in_flow(child));

            // Max-content of any box whose children are all inline (text /
            // inline elements) is the SUM of those children — they flow on one
            // line that doesn't wrap at max-content. This must hold for block
            // containers too (e.g. a table cell), not just inline ones:
            // otherwise a cell split into several inline text segments measures
            // as only its widest segment, so its column is under-sized and the
            // table's spare width spills onto narrow neighbours (header/title
            // got pushed right).
            if inline_container_children_all_inline(b) {
                return inflow
                    .map(|child| intrinsic_margin_width(child, container_w, child_parent_fs))
                    .sum();
            }

            // A multi-column GRID's max-content is the SUM of its column tracks
            // (+ gaps), NOT the max of its items. Falling through to the `fold
            // (max)` below under-sized any grid to a single column's width — e.g.
            // `grid-template-columns: 1.1fr 1fr` reported only max(hero, card),
            // which then shrank an ancestor flex item and cascaded into a crushed
            // card + clipped tab labels (mail.hyvechain.com auth layout).
            if b.is_grid {
                if let Some(w) = intrinsic_grid_max_width(b, container_w, child_parent_fs) {
                    return w;
                }
            }

            if b.is_flex
                && matches!(
                    b.flex_direction,
                    FlexDirection::Row | FlexDirection::RowReverse
                )
            {
                let mut total = 0.0;
                let mut count = 0usize;
                for child in inflow {
                    total += intrinsic_margin_width(child, container_w, child_parent_fs);
                    count += 1;
                }
                if count > 1 {
                    total += match b.flex_direction {
                        FlexDirection::Row | FlexDirection::RowReverse => b.column_gap_px,
                        FlexDirection::Column | FlexDirection::ColumnReverse => b.row_gap_px,
                    } * (count.saturating_sub(1) as f32);
                }
                total
            } else {
                inflow
                    .map(|child| intrinsic_margin_width(child, container_w, child_parent_fs))
                    .fold(0.0, f32::max)
            }
        }
    }
}

/// Max-content width of a grid container: the SUM of its column tracks plus
/// column gaps (each content-sized track = the max max-content of the items
/// that land in it under row-major auto-placement; a fixed `px` track = its
/// length). Returns `None` for an `auto-fill`/`auto-fit` repeat (dynamic column
/// count) or no explicit `grid-template-columns`, so the caller keeps its
/// default behaviour. Percentage tracks don't contribute an intrinsic size
/// (CSS Sizing §5) so they're treated as content-sized.
fn intrinsic_grid_max_width(b: &LayoutBox, container_w: f32, parent_fs: f32) -> Option<f32> {
    let tracks = b.grid_template_columns.as_ref()?;
    if tracks.is_empty() || tracks.iter().any(|t| matches!(t, GridTrack::AutoRepeat(_))) {
        return None;
    }
    let n = tracks.len();
    let mut col_max = vec![0.0f32; n];
    // Fixed `px` tracks contribute their length regardless of content.
    for (j, t) in tracks.iter().enumerate() {
        if let GridTrack::Px(p) = t {
            col_max[j] = p.max(0.0);
        }
    }
    // Content-sized tracks take the widest item placed in that column (row-major).
    let inflow = b.children.iter().filter(|c| is_in_flow(c));
    for (i, child) in inflow.enumerate() {
        let j = i % n;
        if !matches!(tracks[j], GridTrack::Px(_)) {
            let w = intrinsic_margin_width(child, container_w, parent_fs);
            col_max[j] = col_max[j].max(w);
        }
    }
    let total: f32 = col_max.iter().sum();
    let gaps = b.column_gap_px * (n.saturating_sub(1) as f32);
    Some(total + gaps)
}

/// CSS min-content width — the widest *unbreakable* unit (longest word
/// for text; widest descendant for boxes). This is the basis for a
/// flex item's automatic minimum size (CSS Flexbox §4.5: with the
/// default `min-width: auto`, an item cannot shrink below its
/// content's min-content size). Distinct from `intrinsic_content_width`
/// (max-content / no-wrap). Without this floor, a flex row whose
/// contents overflow shrinks every item toward zero instead of
/// overflowing — squashing nav bars and toolbars. Returns the
/// CONTENT-box min width (no margin/border/padding).
fn min_content_width(b: &LayoutBox, parent_font_size: f32) -> f32 {
    let key = (b as *const LayoutBox as usize, parent_font_size.to_bits());
    if let Some(v) = MIN_CONTENT_CACHE.with(|c| c.borrow().get(&key).copied()) {
        return v;
    }
    let result = match &b.kind {
        BoxKind::Text(t) => {
            let fs = if b.font_size_px > 0.0 {
                b.font_size_px
            } else {
                parent_font_size
            };
            let family = b.font_family.as_deref();
            // Widest single whitespace-separated token. `pre`/`nowrap`
            // text is unbreakable, so the whole run is one unit.
            let ls = b.letter_spacing_px;
            let tt = b.text_transform;
            if b.preserve_whitespace {
                t.split('\n')
                    .map(|line| {
                        let line = apply_text_transform(line, tt);
                        let line = line.as_ref();
                        measure_text_global(
                            line,
                            fs,
                            family,
                            b.font_weight_bold,
                            b.font_style_italic,
                        ) + ls * line.chars().count() as f32
                    })
                    .fold(0.0, f32::max)
            } else {
                // css_split_whitespace: splits on CSS-collapsible chars only;
                // U+00A0 (NBSP) keeps adjacent tokens in one word (non-breaking).
                css_split_whitespace(t)
                    .map(|word| {
                        let word = apply_text_transform(word, tt);
                        let word = word.as_ref();
                        measure_text_global(
                            word,
                            fs,
                            family,
                            b.font_weight_bold,
                            b.font_style_italic,
                        ) + ls * word.chars().count() as f32
                    })
                    .fold(0.0, f32::max)
            }
        }
        _ => {
            if let Some(img) = &b.embedded_image {
                img.width as f32
            } else {
                let child_parent_fs = if b.font_size_px > 0.0 {
                    b.font_size_px
                } else {
                    parent_font_size
                };
                // For a block/flex container the min-content width is the
                // max of its in-flow children's min-content widths (a child
                // narrower than its own longest word can't exist). We add
                // each child's horizontal chrome so the floor reflects the
                // child's border-box.
                b.children
                    .iter()
                    .filter(|c| is_in_flow(c))
                    .map(|c| {
                        let chrome = c.padding.left
                            + c.padding.right
                            + c.border_width_left()
                            + c.border_width_right();
                        min_content_width(c, child_parent_fs) + chrome
                    })
                    .fold(0.0, f32::max)
            }
        }
    };
    MIN_CONTENT_CACHE.with(|c| c.borrow_mut().insert(key, result));
    result
}

/// Position an out-of-flow (`position: absolute` / `fixed`) child
/// against its containing block. The containing block is the
/// content rect of the nearest positioned ancestor — we approximate
/// "nearest positioned ancestor" as "the parent that owns this
/// child", which covers the common Wikipedia-style `.container
/// { position: relative; } .item { position: absolute; top:N;
/// left:N; }` idiom exactly.
fn place_absolute(
    child: &mut LayoutBox,
    cb: &Rect,
    ctx: &mut LayoutCtx<'_>,
    parent_font_size: f32,
) {
    let anchor = if matches!(child.position, Position::Fixed) {
        Rect {
            x: 0.0,
            y: 0.0,
            w: ctx.cfg.viewport_w,
            h: ctx.cfg.viewport_h,
        }
    } else {
        *cb
    };
    // Resolve top/left first; right/bottom are honoured only when
    // their partner offset is missing (proper "stretch by setting
    // both" handling needs a width-from-constraints solver we
    // don't have yet). Percentages resolve against the containing
    // block's content rect — that's the entire point of doing this
    // at layout time rather than during cascade lowering.
    let dx = child
        .left_px
        .map(|s| s.resolve(anchor.w))
        .unwrap_or_else(|| match child.right_px {
            Some(r) => anchor.w - r.resolve(anchor.w),
            None => 0.0,
        });
    let dy = child
        .top_px
        .map(|s| s.resolve(anchor.h))
        .unwrap_or_else(|| match child.bottom_px {
            Some(b) => anchor.h - b.resolve(anchor.h),
            None => 0.0,
        });
    // When the author set BOTH top and bottom (e.g. `inset: 0`) AND
    // didn't supply an explicit height, CSS Position 3 §6.2 says the
    // height resolves to `cb.h - top - bottom`. Without this rule,
    // overlay/modal patterns (`position: absolute; inset: 0;`)
    // collapse to height=0 and the layout-check fixture catches it
    // immediately. Width gets the same treatment for `left+right`.
    // Chrome-divergence #4 fix: `flex_override_width`/`_height` are
    // PER-PASS hints into `place()` — they say "for THIS layout call,
    // resolve the box at this width". Chrome's equivalent lives in a
    // fresh `NGConstraintSpace::Builder` per pass (see
    // `third_party/blink/renderer/core/layout/ng/ng_block_layout_algorithm.cc`)
    // so it can't survive into later passes. Ours stores it as a
    // mutable field on `LayoutBox`. Every set MUST pair with a restore
    // — otherwise an absolute-positioned element re-laid-out under a
    // different containing block (modal moved, sticky scrolled,
    // re-cascade after JS style write) carries the stale override and
    // the `.is_none()` guard above blocks the new computation.
    //
    // Snapshot whatever was there before, set our pass-local value,
    // run place(), then restore. This matches the pattern at the
    // flex-grow/shrink sites (lines ~2316/2326 etc.) which already
    // do explicit None-restore.
    let saved_w = child.flex_override_width;
    let saved_h = child.flex_override_height;
    if child.explicit_width.is_none() && child.flex_override_width.is_none() {
        if let (Some(l), Some(r)) = (child.left_px, child.right_px) {
            let computed = anchor.w - l.resolve(anchor.w) - r.resolve(anchor.w);
            if computed > 0.0 {
                child.flex_override_width = Some(computed);
            }
        } else {
            // Auto-width absolute/fixed boxes are shrink-to-fit rather
            // than "stretch to the full containing block width". That
            // distinction matters a lot for desktop sidebars, floating
            // menus, and panels whose width is determined by their own
            // contents. Without it, a fixed sidebar becomes a full-width
            // overlay and blows up the whole dashboard layout.
            let intrinsic_w = intrinsic_margin_width(child, 0.0, parent_font_size)
                .max(0.0)
                .min(anchor.w);
            if intrinsic_w > 0.0 {
                child.flex_override_width = Some(intrinsic_w);
            }
        }
    }
    if child.explicit_height.is_none() && child.flex_override_height.is_none() {
        if let (Some(t), Some(bo)) = (child.top_px, child.bottom_px) {
            let computed = anchor.h - t.resolve(anchor.h) - bo.resolve(anchor.h);
            if computed > 0.0 {
                child.flex_override_height = Some(computed);
            }
        }
    }
    place(
        child,
        anchor.x + dx,
        anchor.y + dy,
        anchor.w,
        anchor.h,
        ctx,
        parent_font_size,
    );
    // Per CSS Position 3 §6.3.3: when only `right` is set (left auto), the
    // RIGHT EDGE of the box is at `anchor.w - right` — so its left edge is
    // `anchor.w - right - W`. We've been computing dx as just
    // `anchor.w - right`, which placed the LEFT edge there and pushed
    // the box's right edge off by its own width. Same story for
    // `bottom` only. Now we know the box's actual width/height after
    // placement, so retroactively shift it leftward/upward by that.
    let right_only = child.left_px.is_none() && child.right_px.is_some();
    let bottom_only = child.top_px.is_none() && child.bottom_px.is_some();
    if right_only || bottom_only {
        let mr = child.margin_rect();
        let mut shift_x = 0.0;
        let mut shift_y = 0.0;
        if right_only {
            shift_x = -mr.w;
        }
        if bottom_only {
            shift_y = -mr.h;
        }
        if shift_x != 0.0 || shift_y != 0.0 {
            shift_box(child, shift_x, shift_y);
        }
    }
    // Restore the pre-call values so a later re-layout under a
    // different containing block re-derives its own override.
    child.flex_override_width = saved_w;
    child.flex_override_height = saved_h;
}

struct LayoutCtx<'a> {
    cfg: &'a LayoutConfig,
}

fn build_box(node: &StyledNode, cfg: &LayoutConfig) -> LayoutBox {
    let mut b = build_box_inner(node, cfg);
    // A flex/grid/table container positions+sizes its direct children via its own
    // formatting context (a hidden layout input the fragment cache can't key on),
    // so those children are never individually cached — mark them ineligible. The
    // container itself (a block-level box elsewhere) is still cacheable, and its
    // cached fragment captures the laid-out children.
    if b.is_flex || b.is_grid || b.is_table || b.is_table_row || b.is_table_row_group {
        for k in &mut b.children {
            k.cache_ineligible = true;
        }
    }
    b
}

fn build_box_inner(node: &StyledNode, cfg: &LayoutConfig) -> LayoutBox {
    match &node.kind {
        StyledKind::Element { tag } => {
            let mut kids: Vec<LayoutBox> = if matches!(node.style.display, Some(Display::None)) {
                Vec::new()
            } else {
                // Drop children that have `display: none` themselves so
                // they take up no space and emit no paint at all.
                // Without this they'd produce empty placeholders and
                // shift sibling layout — every hidden menu/dropdown on
                // a real site would shove visible content around.
                node.children
                    .iter()
                    .filter(|c| match c {
                        StyledNode {
                            kind: StyledKind::Element { .. },
                            style,
                            ..
                        } => !matches!(style.display, Some(Display::None)),
                        _ => true,
                    })
                    .map(|c| build_box(c, cfg))
                    .collect()
            };
            if let Some(href) = &node.style.link_href {
                for c in &mut kids {
                    propagate_href(c, href);
                }
            }
            LayoutBox {
                content: Rect::default(),
                padding: node.style.padding,
                margin: node.style.margin,
                margin_auto: node.style.margin_auto,
                border_width_px: node.style.border_width_px.unwrap_or(0.0),
                border_color: node.style.border_color.clone(),
                border_widths_per_side: node.style.border_widths_per_side,
                border_colors_per_side: node.style.border_colors_per_side.clone(),
                border_styles_per_side: node.style.border_styles_per_side,
                text_align: node.style.text_align,
                font_weight_bold: node.style.font_weight_bold.unwrap_or(false),
                font_weight_num: node.style.font_weight_num.unwrap_or(
                    if node.style.font_weight_bold.unwrap_or(false) { 700 } else { 400 },
                ),
                font_style_italic: node.style.font_style_italic.unwrap_or(false),
                font_family: node.style.font_family.clone(),
                text_transform: node.style.text_transform,
                letter_spacing_px: node.style.letter_spacing_px.unwrap_or(0.0),
                text_decoration_underline: node.style.text_decoration_underline.unwrap_or(false),
                text_decoration_line_through: node
                    .style
                    .text_decoration_line_through
                    .unwrap_or(false),
                line_height_px: node.style.line_height_px,
                preserve_whitespace: node.style.preserve_whitespace,
                box_sizing_border_box: node.style.box_sizing_border_box.unwrap_or(false),
                is_flex: matches!(
                    node.style.display,
                    Some(Display::Flex) | Some(Display::InlineFlex)
                ),
                is_grid: matches!(
                    node.style.display,
                    Some(Display::Grid) | Some(Display::InlineGrid)
                ),
                is_inline: matches!(
                    node.style.display,
                    Some(Display::Inline)
                        | Some(Display::InlineBlock)
                        | Some(Display::InlineFlex)
                        | Some(Display::InlineGrid)
                        | Some(Display::InlineTable)
                ),
                is_inline_block: matches!(
                    node.style.display,
                    Some(Display::InlineBlock)
                        | Some(Display::InlineFlex)
                        | Some(Display::InlineGrid)
                        | Some(Display::InlineTable)
                ),
                is_table: matches!(
                    node.style.display,
                    Some(Display::Table) | Some(Display::InlineTable)
                ),
                is_table_row: matches!(node.style.display, Some(Display::TableRow)),
                is_table_cell: matches!(node.style.display, Some(Display::TableCell)),
                is_table_row_group: matches!(node.style.display, Some(Display::TableRowGroup)),
                is_list_item: node.style.display_is_list_item,
                is_flow_root: node.style.display_is_flow_root,
                flex_direction: node.style.flex_direction.unwrap_or_default(),
                flex_wrap: node.style.flex_wrap.unwrap_or_default(),
                flex_grow: node.style.flex_grow.unwrap_or(0.0),
                flex_shrink: node.style.flex_shrink.unwrap_or(1.0),
                flex_basis: node.style.flex_basis,
                justify_content: node.style.justify_content.unwrap_or_default(),
                align_items: node.style.align_items.unwrap_or_default(),
                justify_items: node.style.justify_items,
                align_self: node.style.align_self,
                justify_self: node.style.justify_self,
                gap_px: node.style.gap_px.unwrap_or(0.0),
                column_gap_px: node
                    .style
                    .column_gap_px
                    .or(node.style.gap_px)
                    .unwrap_or(0.0),
                row_gap_px: node.style.row_gap_px.or(node.style.gap_px).unwrap_or(0.0),
                grid_template_columns: node.style.grid_template_columns.clone(),
                grid_template_rows: node.style.grid_template_rows.clone(),
                grid_auto_rows: node.style.grid_auto_rows.clone(),
                grid_auto_columns: node.style.grid_auto_columns.clone(),
                grid_template_areas: node.style.grid_template_areas.clone(),
                grid_area_name: node.style.grid_area_name.clone(),
                grid_column_start: node.style.grid_column_start,
                grid_column_span: node.style.grid_column_span,
                grid_row_start: node.style.grid_row_start,
                grid_row_span: node.style.grid_row_span,
                table_col_span: node.style.table_col_span,
                table_row_span: node.style.table_row_span,
                overflow_hidden: node.style.overflow_hidden,
                overflow_x: node.style.overflow_x,
                overflow_y: node.style.overflow_y,
                scroll_offset_x: 0.0,
                scroll_offset_y: 0.0,
                visibility_hidden: node.style.visibility_hidden.unwrap_or(false),
                opacity: node.style.opacity.unwrap_or(1.0),
                border_radius_px: node.style.border_radius_px.unwrap_or(0.0),
                border_radius_percent: node.style.border_radius_percent,
                translate_x_px: node.style.translate_x_px.unwrap_or(0.0),
                translate_x_percent: node.style.translate_x_percent,
                matrix_2d: node.style.matrix_2d,
                transform_origin: node.style.transform_origin,
                scale_x: node.style.scale_x.unwrap_or(1.0),
                scale_y: node.style.scale_y.unwrap_or(1.0),
                rotate_deg: node.style.rotate_deg.unwrap_or(0.0),
                translate_y_px: node.style.translate_y_px.unwrap_or(0.0),
                translate_y_percent: node.style.translate_y_percent,
                box_shadow: node.style.box_shadow,
                text_shadow: node.style.text_shadow,
                filters: node.style.filters.clone(),
                backdrop_filters: node.style.backdrop_filters.clone(),
                mix_blend_mode: node.style.mix_blend_mode,
                background_blend_mode: node.style.background_blend_mode,
                animation_name: node.style.animation_name.clone(),
                animation_duration_ms: node.style.animation_duration_ms,
                animation_delay_ms: node.style.animation_delay_ms,
                animation_iteration_count: node.style.animation_iteration_count,
                animation_timing: node.style.animation_timing,
                clip_shape: node.style.clip_shape.clone(),
                has_mask_url: node.style.has_mask_url,
                mask_image_url: node.style.mask_image_url.clone(),
                background_gradient: node.style.background_gradient,
                background_radial_gradient: node.style.background_radial_gradient,
                background_gradient_full: node.style.background_gradient_full.clone(),
                background_image_url: node.style.background_image_url.clone(),
                background_repeat: node.style.background_repeat,
                position: node.style.position.unwrap_or(Position::Static),
                z_index: node.style.z_index,
                float_side: node.style.float_side,
                clear: node.style.clear,
                vertical_align: node.style.vertical_align,
                top_px: node.style.top_px,
                right_px: node.style.right_px,
                bottom_px: node.style.bottom_px,
                left_px: node.style.left_px,
                explicit_width: node.style.width,
                flex_override_width: None,
                explicit_height: node.style.height,
                flex_override_height: None,
                aspect_ratio: node.style.aspect_ratio,
                max_width: node.style.max_width,
                max_height: node.style.max_height,
                min_width: node.style.min_width,
                min_height: node.style.min_height,
                background: node.style.background.clone(),
                text_color: node.style.text_color.unwrap_or(cfg.default_text_color),
                font_size_px: node.style.font_size_px.unwrap_or(cfg.default_font_size_px),
                kind: BoxKind::Block { tag: tag.clone() },
                link_href: node.style.link_href.clone(),
                embedded_image: node.style.embedded_image.clone(),
                mask_image: node.style.mask_image.clone(),
                background_image: node.style.background_image.clone(),
                background_size: node.style.background_size.clone(),
                background_position: node.style.background_position,
                object_fit: node.style.object_fit,
                object_position: node.style.object_position,
                element_path: node.style.element_path.clone(),
                node_id: node.style.node_id,
                cache_ineligible: false,
                children: kids,
            }
        }
        StyledKind::Text(t) => LayoutBox {
            content: Rect::default(),
            padding: EdgeSizes::default(),
            // Carry the (flattened-segment) text node's horizontal margins so
            // inline element margins survive — e.g. HN's `.hnname{margin-right}`
            // gap between "Hacker News" and "new".
            margin: node.style.margin,
            margin_auto: EdgeAuto::default(),
            border_width_px: 0.0,
            border_color: None,
            border_widths_per_side: [None; 4],
            border_colors_per_side: [None; 4],
            border_styles_per_side: [None; 4],
            text_align: node.style.text_align,
            font_weight_bold: node.style.font_weight_bold.unwrap_or(false),
            font_weight_num: node.style.font_weight_num.unwrap_or(
                if node.style.font_weight_bold.unwrap_or(false) { 700 } else { 400 },
            ),
            font_style_italic: node.style.font_style_italic.unwrap_or(false),
            font_family: node.style.font_family.clone(),
            text_transform: node.style.text_transform,
            letter_spacing_px: node.style.letter_spacing_px.unwrap_or(0.0),
            text_decoration_underline: node.style.text_decoration_underline.unwrap_or(false),
            text_decoration_line_through: node.style.text_decoration_line_through.unwrap_or(false),
            line_height_px: node.style.line_height_px,
            preserve_whitespace: node.style.preserve_whitespace,
            box_sizing_border_box: node.style.box_sizing_border_box.unwrap_or(false),
            is_flex: false,
            is_grid: false,
            is_inline: true,
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
            grid_area_name: None,
            grid_column_start: None,
            grid_column_span: None,
            grid_row_start: None,
            grid_row_span: None,
            table_col_span: None,
            table_row_span: None,
            overflow_hidden: false,
            overflow_x: Overflow::Visible,
            overflow_y: Overflow::Visible,
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
            box_shadow: None,
            text_shadow: None,
            filters: Vec::new(),
            backdrop_filters: Vec::new(),
            mix_blend_mode: BlendMode::Normal,
            background_blend_mode: BlendMode::Normal,
            animation_name: None,
            animation_duration_ms: 0.0,
            animation_delay_ms: 0.0,
            animation_iteration_count: 1.0,
            animation_timing: 3,
            clip_shape: None,
            has_mask_url: false,
            mask_image_url: None,
            background_gradient: None,
            background_radial_gradient: None,
            background_gradient_full: None,
            background_image_url: None,
            background_repeat: BackgroundRepeat::default(),
            position: Position::Static,
            z_index: None,
            float_side: FloatSide::None,
            clear: ClearMode::None,
            vertical_align: VerticalAlign::Baseline,
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
            text_color: node.style.text_color.unwrap_or(cfg.default_text_color),
            font_size_px: node.style.font_size_px.unwrap_or(0.0),
            kind: BoxKind::Text(t.clone()),
            // Carry the link/element identity onto the text box. Inline
            // flattening (conclave) emits each link's text as a Text
            // StyledNode with `link_href`/`element_path` set; hardcoding
            // None here dropped them, so hit-testing only found
            // non-flattened element links (HN's vote-arrow <a>) — the
            // story-title text links got no hand cursor and no click
            // dispatch.
            link_href: node.style.link_href.clone(),
            embedded_image: None,
            mask_image: None,
            background_image: None,
            background_size: None,
            background_position: None,
            object_fit: None,
            object_position: None,
            element_path: node.style.element_path.clone(),
            node_id: node.style.node_id,
            cache_ineligible: false,
            children: Vec::new(),
        },
    }
}

fn propagate_href(b: &mut LayoutBox, href: &str) {
    if b.link_href.is_none() {
        b.link_href = Some(href.to_string());
    }
    for c in &mut b.children {
        propagate_href(c, href);
    }
}

/// Find the deepest layout box containing `(x, y)` that carries a link
/// href. Returns its href string.
pub fn hit_test_link(root: &LayoutBox, x: f32, y: f32) -> Option<String> {
    let mut best: Option<&LayoutBox> = None;
    fn walk<'a>(b: &'a LayoutBox, x: f32, y: f32, best: &mut Option<&'a LayoutBox>) {
        let r = b.content;
        if x >= r.x && x < r.x + r.w && y >= r.y && y < r.y + r.h {
            if b.link_href.is_some() {
                *best = Some(b);
            }
            for c in &b.children {
                walk(c, x, y, best);
            }
        }
    }
    walk(root, x, y, &mut best);
    best.and_then(|b| b.link_href.clone())
}

/// Find the deepest layout box containing `(x, y)` that carries an
/// `element_path`. Used by the JS host to dispatch
/// `addEventListener("click")` callbacks for the clicked element.
pub fn hit_test_element_path(root: &LayoutBox, x: f32, y: f32) -> Option<Vec<usize>> {
    let mut best: Option<&LayoutBox> = None;
    fn walk<'a>(b: &'a LayoutBox, x: f32, y: f32, best: &mut Option<&'a LayoutBox>) {
        let r = b.content;
        if x >= r.x && x < r.x + r.w && y >= r.y && y < r.y + r.h {
            if b.element_path.is_some() {
                *best = Some(b);
            }
            for c in &b.children {
                walk(c, x, y, best);
            }
        }
    }
    walk(root, x, y, &mut best);
    best.and_then(|b| b.element_path.clone())
}

/// Hit-test for the chain of scroll containers under document point
/// `(x, y)`, innermost first. Each entry is `(node_id, max_scroll_left,
/// max_scroll_top)`. Wheel routing scrolls the innermost container that
/// can still move in the wheel's direction, then chains outward to the
/// next ancestor at the edge — matching Blink's scroll-chaining
/// (`scroll_manager.cc` / `RecursiveScrollMethod`). Boxes without a stable
/// `node_id` are skipped (we can't address their offset across frames).
///
/// IMPORTANT: the hit-test uses each box's PADDING box, and the point
/// must already have any ancestor scroll offsets applied by the caller —
/// here we walk in document/layout coordinates, which is what the
/// snapshot tree holds, and adjust for scroll as we descend.
pub fn scroll_chain_at(root: &LayoutBox, x: f32, y: f32) -> Vec<ScrollTarget> {
    let mut chain: Vec<ScrollTarget> = Vec::new();
    // `(sx, sy)` is the accumulated scroll offset of ancestors, which
    // shifts where descendants actually appear on screen.
    fn walk(b: &LayoutBox, x: f32, y: f32, sx: f32, sy: f32, chain: &mut Vec<ScrollTarget>) {
        // The padding box, shifted by ancestor scroll, is where this box's
        // children currently appear. The box itself is positioned in flow
        // (already shifted by ancestor scroll via sx/sy).
        let pad = b.padding_rect();
        let bx = pad.x - sx;
        let by = pad.y - sy;
        let inside = x >= bx && x < bx + pad.w && y >= by && y < by + pad.h;
        if !inside {
            return;
        }
        // This box's own children are further shifted by THIS box's scroll.
        let (csx, csy) = if b.is_scroll_container() {
            (sx + b.scroll_offset_x, sy + b.scroll_offset_y)
        } else {
            (sx, sy)
        };
        for c in &b.children {
            walk(c, x, y, csx, csy, chain);
        }
        // Push AFTER children so the innermost container ends up first.
        if b.is_scroll_container() {
            if let Some(id) = b.node_id {
                chain.push(ScrollTarget {
                    node_id: id,
                    cur_left: b.scroll_offset_x,
                    cur_top: b.scroll_offset_y,
                    max_left: if b.overflow_x.is_scrollable() {
                        b.max_scroll_left()
                    } else {
                        0.0
                    },
                    max_top: if b.overflow_y.is_scrollable() {
                        b.max_scroll_top()
                    } else {
                        0.0
                    },
                });
            }
        }
    }
    walk(root, x, y, 0.0, 0.0, &mut chain);
    chain
}

/// One link in a scroll chain (see [`scroll_chain_at`]): the scroll
/// container's stable id, its current offset, and the legal max offset on
/// each axis. Wheel routing scrolls the innermost target that can still move
/// in the wheel direction, then chains outward.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct ScrollTarget {
    pub node_id: u64,
    pub cur_left: f32,
    pub cur_top: f32,
    pub max_left: f32,
    pub max_top: f32,
}

/// Find the box with the given stable `node_id` (mutable). Used to apply a
/// scroll-offset delta produced by wheel/keyboard input back into the
/// layout snapshot so the next paint reflects it.
pub fn find_box_by_node_id_mut(root: &mut LayoutBox, id: u64) -> Option<&mut LayoutBox> {
    if root.node_id == Some(id) {
        return Some(root);
    }
    for c in &mut root.children {
        if let Some(found) = find_box_by_node_id_mut(c, id) {
            return Some(found);
        }
    }
    None
}

/// Find the box with the given stable `node_id` (shared).
pub fn find_box_by_node_id(root: &LayoutBox, id: u64) -> Option<&LayoutBox> {
    if root.node_id == Some(id) {
        return Some(root);
    }
    for c in &root.children {
        if let Some(found) = find_box_by_node_id(c, id) {
            return Some(found);
        }
    }
    None
}

#[allow(clippy::too_many_arguments)]
fn place_inner(
    b: &mut LayoutBox,
    x: f32,
    y: f32,
    container_w: f32,
    container_h: f32,
    ctx: &mut LayoutCtx<'_>,
    parent_font_size: f32,
) {
    // Per-side border widths — Wikipedia's `<h2>` only has a bottom
    // underline, infobox cells have 1px on all four. Uniform-border
    // code paths fall back to `border_width_px` via the accessors.
    let bw_left = b.border_width_left();
    let bw_right = b.border_width_right();

    // ---- Step 1: figure out the content width before margin auto. ----
    // "Numeric" (non-auto) margin values; auto sides start at 0 and get
    // their share once we know how much space is left over.
    let mut margin_left = if b.margin_auto.left {
        0.0
    } else {
        b.margin.left
    };
    let mut margin_right = if b.margin_auto.right {
        0.0
    } else {
        b.margin.right
    };
    let margin_top = b.margin.top;
    let horiz_chrome = bw_left + bw_right + b.padding.left + b.padding.right;
    let candidate_w = if let Some(force_w) = b.flex_override_width {
        force_w.max(0.0)
    } else if let Some(spec) = b.explicit_width {
        // Percent resolves against the containing block content width.
        let w = spec.resolve(container_w);
        if b.box_sizing_border_box {
            (w - horiz_chrome).max(0.0)
        } else {
            w
        }
    } else {
        // No explicit width → fill the containing block minus our
        // non-auto margins + chrome. Auto margins on an auto-width
        // block resolve to zero per the spec (rule 10.3.3).
        (container_w - margin_left - margin_right - horiz_chrome).max(0.0)
    };
    // Clamp to max-width (cap) and min-width (floor). Both resolve
    // against the containing block content width.
    let mut content_w = candidate_w;
    if let Some(maxw) = b.max_width {
        let mut cap = maxw.resolve(container_w);
        if b.box_sizing_border_box {
            cap = (cap - horiz_chrome).max(0.0);
        }
        if content_w > cap {
            content_w = cap;
        }
    }
    if let Some(minw) = b.min_width {
        let mut floor = minw.resolve(container_w);
        if b.box_sizing_border_box {
            floor = (floor - horiz_chrome).max(0.0);
        }
        if content_w < floor {
            content_w = floor;
        }
    }

    // ---- Step 2: distribute leftover horizontal space to auto sides. ----
    // CSS 2.1 §10.3.3: if the block has a definite width (or got one
    // via max-width clamp) and one or both horizontal margins are
    // `auto`, split the leftover space among the auto sides. When both
    // are auto the box centres in its containing block.
    let used_w = content_w + horiz_chrome + margin_left + margin_right;
    let leftover = (container_w - used_w).max(0.0);
    match (b.margin_auto.left, b.margin_auto.right) {
        (true, true) => {
            margin_left += leftover / 2.0;
            margin_right += leftover / 2.0;
        }
        (true, false) => margin_left += leftover,
        (false, true) => margin_right += leftover,
        (false, false) => {}
    }
    let content_x = x + margin_left + bw_left + b.padding.left;
    let content_y = y + margin_top + b.border_width_top() + b.padding.top;
    // Only a real specified height (or a replaced/aspect-ratio-derived
    // one) gives descendants a *definite* block-size to resolve against.
    // `min-height` is just a lower bound; treating it as definite makes
    // flex columns with `min-height: 100vh` behave like "exactly one
    // viewport tall", which pulls later siblings upward and crushes the
    // scrollable document into a single screen.
    let known_content_h =
        if b.explicit_height.is_some() || b.aspect_ratio.filter(|ratio| *ratio > 0.0).is_some() {
            Some(resolve_content_height(b, 0.0, content_w, container_h))
        } else {
            None
        };
    let child_container_h = known_content_h.unwrap_or(container_h);
    // Persist the now-resolved margins so paint and the parent's
    // child-stack math (which reads `margin_rect()`) see the final
    // numbers, not the pre-distribution placeholders.
    b.margin.left = margin_left;
    b.margin.right = margin_right;

    match &b.kind {
        BoxKind::Text(t) => {
            let fs = if b.font_size_px > 0.0 {
                b.font_size_px
            } else {
                parent_font_size
            };
            b.font_size_px = fs;
            // Floor the wrap width at ~8 glyph-widths even when the
            // parent computed content_w=0 — keeps a parent layout bug
            // from blowing a 30-char string up to 30 lines tall.
            let approx_glyph_w = fs * 0.55;
            let wrap_w = content_w.max(approx_glyph_w * 8.0);
            let family = b.font_family.as_deref();
            let bold = b.font_weight_bold;
            let italic = b.font_style_italic;
            // Normalise per CSS white-space rules. In `white-space:normal`
            // a source newline is NOT a forced break — CSS-collapsible
            // whitespace (space, tab, LF, CR, FF) collapses to single spaces
            // and the run becomes ONE logical line that the word-wrap pass
            // below breaks purely by width. Splitting on `\n` here treated
            // pretty-print source newlines as hard breaks, so a `"\n"`-only
            // node became two empty lines (a tall empty line box) — that
            // inflated Wikipedia's header card to 217px. `pre`/`pre-wrap`
            // keep their newlines. U+00A0 (NBSP) is preserved verbatim.
            let normalized_lines: Vec<String> = if b.preserve_whitespace {
                t.split('\n').map(|line| line.to_string()).collect()
            } else {
                vec![css_normalize_whitespace(t)]
            };
            // Real word-wrap pass: measure each prospective line with
            // `cfg.measure_text` and break at the last fitting word.
            // This replaces the old `chars / line_capacity` heuristic
            // that produced wildly wrong heights with proportional
            // fonts and any text-bearing CJK/ligature content.
            let mut lines = 0.0f32;
            for line in &normalized_lines {
                if b.preserve_whitespace {
                    // pre-formatted: measure verbatim, no wrapping.
                    let w = ctx.cfg.measure_text(line, fs, family, bold, italic);
                    lines += ((w / wrap_w).ceil()).max(1.0);
                    continue;
                }
                if line.is_empty() {
                    lines += 1.0;
                    continue;
                }
                // Greedy word-wrap: accumulate words while the line
                // still fits, break when it would overflow. One pass
                // through the line text.
                let mut cur = String::new();
                let mut line_count: u32 = 0;
                for word in line.split(' ') {
                    let trial = if cur.is_empty() {
                        word.to_string()
                    } else {
                        format!("{cur} {word}")
                    };
                    let w = ctx.cfg.measure_text(&trial, fs, family, bold, italic);
                    if w > wrap_w && !cur.is_empty() {
                        // current line is full — emit it, restart with `word`
                        line_count += 1;
                        cur = word.to_string();
                    } else {
                        cur = trial;
                    }
                }
                if !cur.is_empty() {
                    line_count += 1;
                }
                lines += (line_count as f32).max(1.0);
            }
            lines = lines.max(1.0);
            let line_h = b.line_height_px.unwrap_or(fs * ctx.cfg.default_line_height);
            let want_h = line_h * lines;
            b.content = Rect {
                x: content_x,
                y: content_y,
                w: content_w,
                h: want_h,
            };
        }
        _ => {
            // Image-replaced box: start from intrinsic dimensions, but
            // still honor CSS width / max-width / min-width constraints.
            if let Some(img) = &b.embedded_image {
                let natural_w = img.width as f32;
                let natural_h = img.height as f32;
                let intrinsic_ratio = if natural_w > 0.0 && natural_h > 0.0 {
                    Some(natural_w / natural_h)
                } else {
                    None
                };

                let mut image_w = if let Some(spec) = b.explicit_width {
                    let w = spec.resolve(container_w);
                    if b.box_sizing_border_box {
                        (w - horiz_chrome).max(0.0)
                    } else {
                        w
                    }
                } else {
                    natural_w
                };
                if let Some(maxw) = b.max_width {
                    let mut cap = maxw.resolve(container_w);
                    if b.box_sizing_border_box {
                        cap = (cap - horiz_chrome).max(0.0);
                    }
                    image_w = image_w.min(cap);
                }
                if let Some(minw) = b.min_width {
                    let mut floor = minw.resolve(container_w);
                    if b.box_sizing_border_box {
                        floor = (floor - horiz_chrome).max(0.0);
                    }
                    image_w = image_w.max(floor);
                }

                let mut image_h = if let Some(spec) = b.explicit_height {
                    spec.resolve(container_h)
                } else if let Some(ratio) = intrinsic_ratio {
                    image_w / ratio
                } else {
                    natural_h
                };
                // Per CSS Sizing: max-height/min-height percentages
                // resolve against the parent's content height. When the
                // parent has indefinite height (container_h == 0 because
                // the parent is auto-sized), percent values resolve to
                // none. Px / calc-without-percent values still apply.
                if let Some(maxh) = b.max_height {
                    if container_h > 0.0 || !maxh.is_percent_based() {
                        image_h = image_h.min(maxh.resolve(container_h));
                    }
                }
                if let Some(minh) = b.min_height {
                    if container_h > 0.0 || !minh.is_percent_based() {
                        image_h = image_h.max(minh.resolve(container_h));
                    }
                }

                b.content = Rect {
                    x: content_x,
                    y: content_y,
                    w: image_w.max(0.0),
                    h: image_h.max(0.0),
                };
                return;
            }
            let parent_fs = b.font_size_px;
            if b.is_flex {
                place_flex(b, content_x, content_y, content_w, child_container_h, ctx);
                return;
            }
            if b.is_grid {
                place_grid(b, content_x, content_y, content_w, child_container_h, ctx);
                return;
            }
            if b.is_table {
                place_table(b, content_x, content_y, content_w, child_container_h, ctx);
                return;
            }
            if b.is_inline && inline_container_children_all_inline(b) {
                let used_h = place_inline_run(
                    &mut b.children,
                    content_x,
                    content_y,
                    content_w,
                    child_container_h,
                    ctx,
                    parent_fs,
                    b.text_align,
                );
                // An explicit `height` (or min/max-height) must win over the
                // natural inline-run height. This matters most for an empty
                // sized inline-block (e.g. a fixed-size `<span>` or replaced
                // box with no children): its child run measures 0, but the
                // box should still occupy its declared height. Without this,
                // such boxes collapse to h=0 and break baseline alignment.
                let content_h = resolve_content_height(b, used_h.max(0.0), content_w, container_h);
                b.content = Rect {
                    x: content_x,
                    y: content_y,
                    w: content_w,
                    h: content_h,
                };
                return;
            }
            // First pass: classify each in-flow child as block-level
            // (own line) or inline-level (joins the running line).
            // Out-of-flow children are deferred to the second pass
            // below. Indices walked here let us batch consecutive
            // inline siblings into one "inline group" that
            // `place_inline_line` lays out side-by-side with wrap.
            let kinds: Vec<ChildKind> = b
                .children
                .iter()
                .map(|c| {
                    if !is_in_flow(c) {
                        ChildKind::OutOfFlow
                    } else if c.float_side != FloatSide::None {
                        ChildKind::Float
                    } else if c.is_inline {
                        ChildKind::Inline
                    } else {
                        ChildKind::Block
                    }
                })
                .collect();
            let mut child_y = content_y;
            // Active floats — siblings already pinned to left/right
            // whose vertical extent still affects subsequent in-flow
            // boxes' horizontal band. Each entry: (side, bottom_y, w).
            let mut floats: Vec<(FloatSide, f32, f32)> = Vec::new();
            let mut i = 0;
            // Margin collapsing for adjacent block siblings (CSS 2.1
            // §8.3.1): the gap between two consecutive in-flow blocks is
            // `max(prev.margin_bottom, next.margin_top)`, not their sum.
            // We track `prev_block_margin_bottom` and, on each new block
            // sibling, deduct the smaller of the two margins after
            // placement so the actual advance equals the max.
            let mut prev_block_margin_bottom: f32 = 0.0;
            let mut had_prev_block = false;
            while i < b.children.len() {
                match kinds[i] {
                    ChildKind::OutOfFlow => {
                        i += 1;
                    }
                    ChildKind::Block => {
                        // Honour `clear:` by jumping past matching floats.
                        child_y = apply_clear(child_y, b.children[i].clear, &floats);
                        sweep_floats(child_y, &mut floats);
                        let (ex, ew) = effective_band(content_x, content_w, &floats);
                        place(
                            &mut b.children[i],
                            ex,
                            child_y,
                            ew,
                            child_container_h,
                            ctx,
                            parent_fs,
                        );
                        let this_margin_top = b.children[i].margin.top;
                        let this_margin_bottom = b.children[i].margin.bottom;
                        // Sibling collapse: between two adjacent blocks
                        // the spacing is max(prev_bottom, this_top), not
                        // sum. We've already added `this_top` as part of
                        // `margin_rect().h`, so deduct
                        // `min(prev_bottom, this_top)` from the advance.
                        let collapse_saving = if had_prev_block {
                            prev_block_margin_bottom.min(this_margin_top)
                        } else {
                            0.0
                        };
                        let advance = b.children[i].margin_rect().h - collapse_saving;
                        // Shift this child up by collapse_saving so its
                        // visible position reflects the collapsed gap.
                        if collapse_saving > 0.0 {
                            shift_box(&mut b.children[i], 0.0, -collapse_saving);
                        }
                        child_y += advance;
                        prev_block_margin_bottom = this_margin_bottom;
                        had_prev_block = true;
                        i += 1;
                    }
                    ChildKind::Float => {
                        // Honour `clear:` on the float itself, then
                        // place it inside the current effective band.
                        child_y = apply_clear(child_y, b.children[i].clear, &floats);
                        sweep_floats(child_y, &mut floats);
                        let (ex, ew) = effective_band(content_x, content_w, &floats);
                        let side = b.children[i].float_side;
                        place(
                            &mut b.children[i],
                            ex,
                            child_y,
                            ew,
                            child_container_h,
                            ctx,
                            parent_fs,
                        );
                        let mr = b.children[i].margin_rect();
                        let fw = mr.w;
                        let fh = mr.h;
                        // Right floats: pin to the right side of the
                        // band. Left floats are already at the left.
                        if matches!(side, FloatSide::Right) {
                            let dx = (ex + ew) - (mr.x + mr.w);
                            shift_box(&mut b.children[i], dx, 0.0);
                        }
                        // Track for subsequent siblings; do NOT advance
                        // child_y — that's the defining feature of
                        // floats vs. blocks.
                        floats.push((side, child_y + fh, fw));
                        had_prev_block = false;
                        prev_block_margin_bottom = 0.0;
                        i += 1;
                    }
                    ChildKind::Inline => {
                        // Consume the maximal run of consecutive
                        // Inline siblings starting at `i` and place
                        // them in horizontal lines, inside the band
                        // left over by active floats.
                        let start = i;
                        while i < b.children.len() && matches!(kinds[i], ChildKind::Inline) {
                            i += 1;
                        }
                        sweep_floats(child_y, &mut floats);
                        let (ex, ew) = effective_band(content_x, content_w, &floats);
                        let used_h = place_inline_run(
                            &mut b.children[start..i],
                            ex,
                            child_y,
                            ew,
                            child_container_h,
                            ctx,
                            parent_fs,
                            b.text_align,
                        );
                        child_y += used_h;
                        had_prev_block = false;
                        prev_block_margin_bottom = 0.0;
                    }
                }
            }
            // The block's own height must extend past any unfinished
            // float so the float sits *inside* its container rather
            // than escaping below it. (Without this, a tall infobox in
            // a short article would clip out the bottom.)
            for (_, fbot, _) in &floats {
                if *fbot > child_y {
                    child_y = *fbot;
                }
            }
            let content_h =
                resolve_content_height(b, (child_y - content_y).max(0.0), content_w, container_h);
            b.content = Rect {
                x: content_x,
                y: content_y,
                w: content_w,
                h: content_h,
            };
            // Out-of-flow second pass: now that we know the parent's
            // content rect, position each absolute/fixed child against
            // it. Per CSS Position §4 the containing block is the
            // *padding box* of the nearest positioned ancestor, so we
            // pass padding_rect() rather than the content rect.
            let cb = b.padding_rect();
            for child in &mut b.children {
                if is_in_flow(child) {
                    continue;
                }
                place_absolute(child, &cb, ctx, parent_fs);
            }
        }
    }
}

fn resolve_content_height(b: &LayoutBox, natural_h: f32, content_w: f32, viewport_h: f32) -> f32 {
    let mut content_h = if let Some(force_h) = b.flex_override_height {
        force_h.max(0.0)
    } else if let Some(spec) = b.explicit_height {
        spec.resolve(viewport_h)
    } else {
        let preferred_h = b
            .aspect_ratio
            .filter(|ratio| *ratio > 0.0)
            .map(|ratio| content_w / ratio)
            .unwrap_or(0.0);
        natural_h.max(preferred_h)
    };
    if let Some(maxh) = b.max_height {
        if viewport_h > 0.0 || !maxh.is_percent_based() {
            content_h = content_h.min(maxh.resolve(viewport_h));
        }
    }
    if let Some(minh) = b.min_height {
        if viewport_h > 0.0 || !minh.is_percent_based() {
            content_h = content_h.max(minh.resolve(viewport_h));
        }
    }
    content_h.max(0.0)
}

fn flex_row_total_main(indices: &[usize], widths: &[f32], main_gap: f32) -> f32 {
    if indices.is_empty() {
        return 0.0;
    }
    indices.iter().map(|&i| widths[i]).sum::<f32>()
        + main_gap * ((indices.len().saturating_sub(1)) as f32)
}

/// Count the ROW main-axis (left/right) `auto` margin slots across a line's
/// flex items. CSS Flexbox §8.1: positive free space is distributed EQUALLY to
/// these auto margins BEFORE `justify-content` runs (and when any exist,
/// justify-content gets no free space). This is what centers a single
/// `margin: 0 auto` flex item (the `.container` / `.hero-container` idiom).
fn flex_row_auto_margin_slots(b: &LayoutBox, indices: &[usize]) -> u32 {
    indices
        .iter()
        .map(|&i| {
            let c = &b.children[i];
            u32::from(c.margin_auto.left) + u32::from(c.margin_auto.right)
        })
        .sum()
}

fn apply_flex_row_sizing(
    b: &mut LayoutBox,
    indices: &[usize],
    child_widths: &mut [f32],
    child_heights: &mut [f32],
    available_main: f32,
    main_gap: f32,
    content_x: f32,
    content_y: f32,
    container_h: f32,
    ctx: &mut LayoutCtx<'_>,
    parent_fs: f32,
) -> f32 {
    let mut total_main = flex_row_total_main(indices, child_widths, main_gap);
    if indices.is_empty() {
        return total_main;
    }
    if total_main < available_main {
        let free = available_main - total_main;
        let grow_sum: f32 = indices
            .iter()
            .map(|&i| b.children[i].flex_grow.max(0.0))
            .sum();
        if grow_sum > 0.0 {
            // CSS Flexbox §9.7: freeze over-flexed items loop.
            // Grow items proportionally, but clamp to max-width and
            // redistribute remaining free space to un-frozen items.
            // Per spec: items whose grown size >= max-main-size are
            // frozen at that size; repeat until no new items freeze.
            let mut frozen = vec![false; b.children.len()];
            // Initialise each item's working width at its flex base size.
            let mut working_widths: Vec<f32> = (0..b.children.len())
                .map(|i| child_widths[i])
                .collect();
            let mut remaining_free = free;
            loop {
                let unfrozen_grow_sum: f32 = indices
                    .iter()
                    .filter(|&&i| !frozen[i])
                    .map(|&i| b.children[i].flex_grow.max(0.0))
                    .sum();
                if unfrozen_grow_sum <= 0.0 {
                    break;
                }
                // Distribute remaining_free among un-frozen items.
                for &i in indices.iter().filter(|&&i| !frozen[i]) {
                    let grow = b.children[i].flex_grow.max(0.0);
                    if grow <= 0.0 {
                        continue;
                    }
                    working_widths[i] = child_widths[i] + remaining_free * (grow / unfrozen_grow_sum);
                }
                // Clamp to max-width and collect items to freeze.
                // Two-pass to avoid holding an immutable borrow on `frozen`
                // while also mutating it.
                let newly_frozen: Vec<(usize, f32)> = indices
                    .iter()
                    .copied()
                    .filter(|&i| !frozen[i])
                    .filter_map(|i| {
                        b.children[i].max_width.map(|max_w_spec| {
                            let chrome_w = flex_item_chrome(&b.children[i]);
                            let max_content_w = max_w_spec.resolve(available_main);
                            let max_outer_w = if b.children[i].box_sizing_border_box {
                                max_content_w
                            } else {
                                max_content_w + chrome_w
                            };
                            (i, max_outer_w)
                        })
                    })
                    .filter(|&(i, max_outer_w)| working_widths[i] > max_outer_w)
                    .map(|(i, max_outer_w)| (i, max_outer_w.max(child_widths[i])))
                    .collect();
                let any_frozen = !newly_frozen.is_empty();
                for (i, clamped_w) in newly_frozen {
                    working_widths[i] = clamped_w;
                    frozen[i] = true;
                }
                if !any_frozen {
                    break;
                }
                // Recalculate remaining free space: subtract frozen items'
                // sizes from the available space and non-frozen items' base
                // sizes.
                let frozen_total: f32 = indices
                    .iter()
                    .filter(|&&i| frozen[i])
                    .map(|&i| working_widths[i])
                    .sum::<f32>()
                    + main_gap * ((indices.len().saturating_sub(1)) as f32);
                let unfrozen_base: f32 = indices
                    .iter()
                    .filter(|&&i| !frozen[i])
                    .map(|&i| child_widths[i])
                    .sum();
                remaining_free = (available_main - frozen_total - unfrozen_base).max(0.0);
                // Reset working widths for non-frozen items to their base
                // sizes so the next iteration starts fresh.
                for &i in indices.iter().filter(|&&i| !frozen[i]) {
                    working_widths[i] = child_widths[i];
                }
            }
            // Write final working widths back to child_widths.
            for &i in indices {
                child_widths[i] = working_widths[i];
            }
            total_main = flex_row_total_main(indices, child_widths, main_gap);
        }
    } else if total_main > available_main {
        let overflow = total_main - available_main;
        let mut shrink_sum = 0.0;
        let mut shrink_weights = vec![0.0; b.children.len()];
        for &i in indices {
            let weight = b.children[i].flex_shrink.max(0.0) * child_widths[i].max(1.0);
            shrink_weights[i] = weight;
            shrink_sum += weight;
        }
        if shrink_sum > 0.0 {
            for &i in indices {
                if shrink_weights[i] <= 0.0 {
                    continue;
                }
                let target_outer_w =
                    (child_widths[i] - overflow * (shrink_weights[i] / shrink_sum)).max(0.0);
                let chrome_w = flex_item_chrome(&b.children[i]);
                let mut target_content_w = (target_outer_w - chrome_w).max(0.0);
                // CSS Flexbox §4.5 — clamp to the automatic minimum.
                // With the default `min-width: auto`, an item cannot
                // shrink below its content's min-content size. An
                // explicit `min-width` overrides. Without this floor a
                // text item collapses to ~0 instead of overflowing,
                // squashing nav/toolbar labels.
                let auto_min = if b.children[i].min_width.is_some() {
                    // explicit min-width already applied inside `place`
                    0.0
                } else {
                    min_content_width(&b.children[i], parent_fs)
                };
                target_content_w = target_content_w.max(auto_min);
                // Compute-only: store the clamped outer width; place later.
                child_widths[i] = target_content_w + chrome_w;
            }
            total_main = flex_row_total_main(indices, child_widths, main_gap);
        }
    }
    total_main
}

fn flex_column_total_main(indices: &[usize], heights: &[f32], main_gap: f32) -> f32 {
    if indices.is_empty() {
        return 0.0;
    }
    indices.iter().map(|&i| heights[i]).sum::<f32>()
        + main_gap * ((indices.len().saturating_sub(1)) as f32)
}

fn apply_flex_column_sizing(
    b: &mut LayoutBox,
    indices: &[usize],
    child_widths: &mut [f32],
    child_heights: &mut [f32],
    available_main: f32,
    content_x: f32,
    content_y: f32,
    content_w: f32,
    container_h: f32,
    main_gap: f32,
    ctx: &mut LayoutCtx<'_>,
    parent_fs: f32,
) -> f32 {
    let mut total_main = flex_column_total_main(indices, child_heights, main_gap);
    if indices.is_empty() {
        return total_main;
    }
    if total_main < available_main {
        let free = available_main - total_main;
        let grow_sum: f32 = indices
            .iter()
            .map(|&i| b.children[i].flex_grow.max(0.0))
            .sum();
        if grow_sum > 0.0 {
            for &i in indices {
                let grow = b.children[i].flex_grow.max(0.0);
                if grow <= 0.0 {
                    continue;
                }
                let target_outer_h = child_heights[i] + free * (grow / grow_sum);
                let chrome_h = (child_heights[i] - b.children[i].content.h).max(0.0);
                let target_content_h = (target_outer_h - chrome_h).max(0.0);
                b.children[i].flex_override_height = Some(target_content_h);
                place(
                    &mut b.children[i],
                    content_x,
                    content_y,
                    content_w,
                    container_h,
                    ctx,
                    parent_fs,
                );
                b.children[i].flex_override_height = None;
                let mr = b.children[i].margin_rect();
                child_widths[i] = mr.w;
                child_heights[i] = mr.h;
            }
            total_main = flex_column_total_main(indices, child_heights, main_gap);
        }
    } else if total_main > available_main {
        let overflow = total_main - available_main;
        let mut shrink_sum = 0.0;
        let mut shrink_weights = vec![0.0; b.children.len()];
        for &i in indices {
            let weight = b.children[i].flex_shrink.max(0.0) * child_heights[i].max(1.0);
            shrink_weights[i] = weight;
            shrink_sum += weight;
        }
        if shrink_sum > 0.0 {
            for &i in indices {
                if shrink_weights[i] <= 0.0 {
                    continue;
                }
                let target_outer_h =
                    (child_heights[i] - overflow * (shrink_weights[i] / shrink_sum)).max(0.0);
                let chrome_h = (child_heights[i] - b.children[i].content.h).max(0.0);
                let target_content_h = (target_outer_h - chrome_h).max(0.0);
                b.children[i].flex_override_height = Some(target_content_h);
                place(
                    &mut b.children[i],
                    content_x,
                    content_y,
                    content_w,
                    container_h,
                    ctx,
                    parent_fs,
                );
                b.children[i].flex_override_height = None;
                let mr = b.children[i].margin_rect();
                child_widths[i] = mr.w;
                child_heights[i] = mr.h;
            }
            total_main = flex_column_total_main(indices, child_heights, main_gap);
        }
    }
    total_main
}

/// Lay a flex container's children along the main axis. V1 supports
/// `flex-direction: row | row-reverse | column | column-reverse`, `gap`,
/// `justify-content`, `align-items: start | center | end | stretch`, and
/// row-axis grow/shrink. `flex-wrap` only affects row containers.
/// Horizontal chrome of a flex item: border + padding + non-auto margins.
/// Equals `margin_box.w - content_box.w` for both box-sizing modes, so it
/// converts between the outer (margin-box) main size the flex algorithm
/// distributes and the content-box width `place()` wants.
fn flex_item_chrome(c: &LayoutBox) -> f32 {
    c.border_width_left()
        + c.border_width_right()
        + c.padding.left
        + c.padding.right
        + (if c.margin_auto.left {
            0.0
        } else {
            c.margin.left
        })
        + (if c.margin_auto.right {
            0.0
        } else {
            c.margin.right
        })
}

/// Flex base main-size (margin-box width) of a ROW flex item, computed WITHOUT
/// laying the child out — the single-pass replacement for the old "place to
/// measure" step. Provably matches the old measured `margin_rect().w`:
/// flex-basis (treated as content width, as the old override did) when set,
/// else explicit width / cached max-content via `intrinsic_margin_width`.
fn flex_base_margin_width(child: &LayoutBox, content_w: f32, parent_fs: f32) -> f32 {
    // A PERCENTAGE main-size (e.g. `width: 100%`) is the flex base size and
    // resolves against the flex container's content width. `intrinsic_margin_
    // width` treats % as auto — correct for min/max-content, but it collapses a
    // `width:100%` flex item to its CONTENT width (the bug that crushed a
    // grid/flex card to ~content size instead of filling its share). Resolve it
    // here, mirroring intrinsic_margin_width's box-sizing + min/max-width logic.
    if let Some(spec) = child.explicit_width {
        if spec.is_percent_based() {
            let horiz_chrome = child.border_width_left()
                + child.border_width_right()
                + child.padding.left
                + child.padding.right;
            let margin_left = if child.margin_auto.left {
                0.0
            } else {
                child.margin.left
            };
            let margin_right = if child.margin_auto.right {
                0.0
            } else {
                child.margin.right
            };
            let resolved = spec.resolve(content_w).max(0.0);
            let mut cw = if child.box_sizing_border_box {
                (resolved - horiz_chrome).max(0.0)
            } else {
                resolved
            };
            if let Some(maxw) = child.max_width {
                let mut cap = maxw.resolve(content_w);
                if child.box_sizing_border_box {
                    cap = (cap - horiz_chrome).max(0.0);
                }
                cw = cw.min(cap);
            }
            if let Some(minw) = child.min_width {
                let mut floor = minw.resolve(content_w);
                if child.box_sizing_border_box {
                    floor = (floor - horiz_chrome).max(0.0);
                }
                cw = cw.max(floor);
            }
            return cw + horiz_chrome + margin_left + margin_right;
        }
    }
    if child.explicit_width.is_none() {
        if let Some(basis) = child.flex_basis {
            let basis_w = basis.resolve(content_w).max(0.0);
            return basis_w + flex_item_chrome(child);
        }
    }
    intrinsic_margin_width(child, content_w, parent_fs)
}

/// Place each ROW flex item exactly once at its final main size (the single
/// place that replaces the measure pass + re-place). Fills in `child_heights`
/// and the exact `child_widths` from the resulting margin box.
#[allow(clippy::too_many_arguments)]
fn place_flex_row_items(
    b: &mut LayoutBox,
    indices: &[usize],
    child_widths: &mut [f32],
    child_heights: &mut [f32],
    content_x: f32,
    content_y: f32,
    container_h: f32,
    ctx: &mut LayoutCtx<'_>,
    parent_fs: f32,
) {
    for &i in indices {
        let chrome_w = flex_item_chrome(&b.children[i]);
        let content_w_i = (child_widths[i] - chrome_w).max(0.0);
        b.children[i].flex_override_width = Some(content_w_i);
        place(
            &mut b.children[i],
            content_x,
            content_y,
            content_w_i,
            container_h,
            ctx,
            parent_fs,
        );
        b.children[i].flex_override_width = None;
        let mr = b.children[i].margin_rect();
        child_widths[i] = mr.w;
        child_heights[i] = mr.h;
    }
}

fn place_flex(
    b: &mut LayoutBox,
    content_x: f32,
    content_y: f32,
    content_w: f32,
    container_h: f32,
    ctx: &mut LayoutCtx<'_>,
) {
    let direction = b.flex_direction;
    let flex_wrap = b.flex_wrap;
    let justify = b.justify_content;
    let align = b.align_items;
    let main_gap = match direction {
        FlexDirection::Row | FlexDirection::RowReverse => b.column_gap_px,
        FlexDirection::Column | FlexDirection::ColumnReverse => b.row_gap_px,
    };
    let cross_gap = match direction {
        FlexDirection::Row | FlexDirection::RowReverse => b.row_gap_px,
        FlexDirection::Column | FlexDirection::ColumnReverse => b.column_gap_px,
    };
    let parent_fs = b.font_size_px;

    // Per-child in-flow flag (Position::Absolute / Fixed are out of
    // flow — they get placed in the second pass below). We use the
    // flag rather than `Vec::filter` so the parallel `widths` /
    // `heights` arrays still align with `b.children` by index.
    let in_flow: Vec<bool> = b.children.iter().map(is_in_flow).collect();
    let inflow_count = in_flow.iter().filter(|&&v| v).count();
    match direction {
        FlexDirection::Row | FlexDirection::RowReverse => {
            let reverse = matches!(direction, FlexDirection::RowReverse);
            // First pass: place each in-flow child at (content_x,
            // content_y) with its natural sizing to learn its width
            // and height. Out-of-flow children get zero size so the
            // distribution math below doesn't see them.
            // Single-pass: compute each item's flex base main-size from the
            // cached intrinsic/basis width WITHOUT placing it. Heights are
            // filled by ONE placement at the final width (place_flex_row_items)
            // after sizing. The old code place()-d every child here just to
            // read its size — and because measuring a flex child fully laid out
            // its subtree (recursing through nested flex), cost compounded
            // ~O(passes^depth). This was ~half of CSS-heavy layout time.
            let mut child_widths: Vec<f32> = b
                .children
                .iter()
                .enumerate()
                .map(|(i, child)| {
                    if in_flow[i] {
                        flex_base_margin_width(child, content_w, parent_fs)
                    } else {
                        0.0
                    }
                })
                .collect();
            let mut child_heights: Vec<f32> = vec![0.0; b.children.len()];
            if matches!(flex_wrap, FlexWrap::Wrap | FlexWrap::WrapReverse) {
                let mut lines: Vec<Vec<usize>> = Vec::new();
                let mut current: Vec<usize> = Vec::new();
                let mut used_w = 0.0;
                for i in 0..b.children.len() {
                    if !in_flow[i] {
                        continue;
                    }
                    let next_w = if current.is_empty() {
                        child_widths[i]
                    } else {
                        used_w + main_gap + child_widths[i]
                    };
                    if !current.is_empty() && next_w > content_w {
                        lines.push(current);
                        current = Vec::new();
                        used_w = 0.0;
                    }
                    used_w = if current.is_empty() {
                        child_widths[i]
                    } else {
                        used_w + main_gap + child_widths[i]
                    };
                    current.push(i);
                }
                if !current.is_empty() {
                    lines.push(current);
                }
                // `wrap-reverse` reverses the cross-axis stacking order:
                // lines are placed starting from the bottom edge, so the
                // first source line ends up at the bottom. Compute all line
                // heights first, then determine starting y.
                let line_heights: Vec<f32> = lines
                    .iter()
                    .map(|line| line.iter().map(|&i| child_heights[i]).fold(0.0_f32, f32::max))
                    .collect();
                let total_lines_h: f32 = line_heights.iter().sum::<f32>()
                    + cross_gap * (lines.len().saturating_sub(1)) as f32;
                let cross_reverse = matches!(flex_wrap, FlexWrap::WrapReverse);
                let mut line_y = if cross_reverse {
                    content_y + total_lines_h
                } else {
                    content_y
                };

                let mut total_h = 0.0;
                for (line_index, line) in lines.iter().enumerate() {
                    // Recompute accurate line_h after sizing this line.
                    apply_flex_row_sizing(
                        b,
                        line,
                        &mut child_widths,
                        &mut child_heights,
                        content_w,
                        main_gap,
                        content_x,
                        content_y,
                        container_h,
                        ctx,
                        parent_fs,
                    );
                    // Single placement pass: lay each item in this line out
                    // once at its final width (fills child_heights). Replaces
                    // the old per-child measure place.
                    place_flex_row_items(
                        b,
                        line,
                        &mut child_widths,
                        &mut child_heights,
                        content_x,
                        content_y,
                        container_h,
                        ctx,
                        parent_fs,
                    );
                    let line_h: f32 = line.iter().map(|&i| child_heights[i]).fold(0.0, f32::max);
                    // For wrap-reverse, the line's top edge is (line_y - line_h).
                    let effective_line_y = if cross_reverse {
                        line_y - line_h
                    } else {
                        line_y
                    };
                    let total_main = flex_row_total_main(line, &child_widths, main_gap);
                    let free_raw = (content_w - total_main).max(0.0);
                    // Auto main-axis margins absorb free space before justify
                    // (Flexbox §8.1) — same as the single-line path below.
                    let auto_slots = flex_row_auto_margin_slots(b, line);
                    let auto_share = if auto_slots > 0 && free_raw > 0.0 {
                        free_raw / auto_slots as f32
                    } else {
                        0.0
                    };
                    let free = if auto_slots > 0 { 0.0 } else { free_raw };
                    let justify = if auto_slots > 0 {
                        JustifyContent::Start
                    } else {
                        justify
                    };
                    let n = line.len().max(1) as f32;
                    let (mut cursor, between) = if reverse {
                        match justify {
                            JustifyContent::Start => (content_x + content_w, main_gap),
                            JustifyContent::End => (content_x + total_main, main_gap),
                            JustifyContent::Center => {
                                (content_x + (content_w + total_main) / 2.0, main_gap)
                            }
                            JustifyContent::SpaceBetween if n > 1.0 => {
                                (content_x + content_w, main_gap + free / (n - 1.0))
                            }
                            JustifyContent::SpaceAround => (
                                content_x + content_w - free / (2.0 * n),
                                main_gap + free / n,
                            ),
                            JustifyContent::SpaceEvenly => (
                                content_x + content_w - free / (n + 1.0),
                                main_gap + free / (n + 1.0),
                            ),
                            _ => (content_x + content_w, main_gap),
                        }
                    } else {
                        match justify {
                            JustifyContent::Start => (content_x, main_gap),
                            JustifyContent::End => (content_x + free, main_gap),
                            JustifyContent::Center => (content_x + free / 2.0, main_gap),
                            JustifyContent::SpaceBetween if n > 1.0 => {
                                (content_x, main_gap + free / (n - 1.0))
                            }
                            JustifyContent::SpaceAround => {
                                (content_x + free / (2.0 * n), main_gap + free / n)
                            }
                            JustifyContent::SpaceEvenly => {
                                (content_x + free / (n + 1.0), main_gap + free / (n + 1.0))
                            }
                            _ => (content_x, main_gap),
                        }
                    };
                    for &i in line {
                        let child = &mut b.children[i];
                        // Per CSS Flexbox L1 §8.3 align-self OVERRIDES the
                        // container's align-items for that single item. Was
                        // dropped here — `align-self: center` on one item
                        // had no effect when container was `align-items:
                        // stretch`. Resolve per-item, then fall through to
                        // the same stretch / start / center / end branches.
                        let item_align = child.align_self.unwrap_or(align);
                        // Cross-axis stretch: when align-items: stretch and the
                        // child has no explicit height, re-place at line_h
                        // (clamped to the child's own min-height).
                        if matches!(item_align, AlignItems::Stretch)
                            && child.explicit_height.is_none()
                            && line_h > child_heights[i]
                        {
                            let min_h = child
                                .min_height
                                .map(|m| m.resolve(container_h))
                                .unwrap_or(0.0);
                            let stretch_h = line_h.max(min_h);
                            child.flex_override_height = Some(stretch_h);
                            place(
                                child,
                                content_x,
                                content_y,
                                child_widths[i],
                                container_h,
                                ctx,
                                parent_fs,
                            );
                            child.flex_override_height = None;
                            child_heights[i] = child.margin_rect().h;
                        }
                        let cy = match item_align {
                            AlignItems::Start | AlignItems::Stretch | AlignItems::Baseline => {
                                effective_line_y
                            }
                            AlignItems::Center => {
                                effective_line_y + (line_h - child_heights[i]) / 2.0
                            }
                            AlignItems::End => effective_line_y + line_h - child_heights[i],
                        };
                        let lead_auto = if reverse {
                            child.margin_auto.right
                        } else {
                            child.margin_auto.left
                        };
                        if lead_auto && auto_share > 0.0 {
                            if reverse {
                                cursor -= auto_share;
                            } else {
                                cursor += auto_share;
                            }
                        }
                        let target_x = if reverse {
                            cursor - child_widths[i]
                        } else {
                            cursor
                        };
                        let dx = target_x - child.margin_rect().x;
                        let dy = cy - child.margin_rect().y;
                        shift_box(child, dx, dy);
                        if reverse {
                            cursor -= child_widths[i] + between;
                        } else {
                            cursor += child_widths[i] + between;
                        }
                        let trail_auto = if reverse {
                            child.margin_auto.left
                        } else {
                            child.margin_auto.right
                        };
                        if trail_auto && auto_share > 0.0 {
                            if reverse {
                                cursor -= auto_share;
                            } else {
                                cursor += auto_share;
                            }
                        }
                    }
                    total_h += line_h;
                    if line_index + 1 < lines.len() {
                        total_h += cross_gap;
                        if cross_reverse {
                            line_y -= line_h + cross_gap;
                        } else {
                            line_y += line_h + cross_gap;
                        }
                    }
                }
                b.content = Rect {
                    x: content_x,
                    y: content_y,
                    w: content_w,
                    h: resolve_content_height(b, total_h, content_w, container_h),
                };
            } else {
                let indices: Vec<usize> = in_flow
                    .iter()
                    .enumerate()
                    .filter_map(|(i, &flow)| flow.then_some(i))
                    .collect();
                apply_flex_row_sizing(
                    b,
                    &indices,
                    &mut child_widths,
                    &mut child_heights,
                    content_w,
                    main_gap,
                    content_x,
                    content_y,
                    container_h,
                    ctx,
                    parent_fs,
                );
                // Single placement pass: lay each in-flow item out once at its
                // final width (fills child_heights). Replaces the measure place.
                place_flex_row_items(
                    b,
                    &indices,
                    &mut child_widths,
                    &mut child_heights,
                    content_x,
                    content_y,
                    container_h,
                    ctx,
                    parent_fs,
                );
                let total_main = flex_row_total_main(&indices, &child_widths, main_gap);
                let free_raw = (content_w - total_main).max(0.0);
                // CSS Flexbox §8.1: auto main-axis margins absorb positive free
                // space BEFORE justify-content. When present, justify-content
                // sees no free space (we zero it and pack from Start, then add
                // each item's leading auto-margin share in the placement loop).
                let auto_slots = flex_row_auto_margin_slots(b, &indices);
                let auto_share = if auto_slots > 0 && free_raw > 0.0 {
                    free_raw / auto_slots as f32
                } else {
                    0.0
                };
                let free = if auto_slots > 0 { 0.0 } else { free_raw };
                let justify = if auto_slots > 0 {
                    JustifyContent::Start
                } else {
                    justify
                };
                let n = inflow_count.max(1) as f32;
                let (mut cursor, between) = if reverse {
                    match justify {
                        JustifyContent::Start => (content_x + content_w, main_gap),
                        JustifyContent::End => (content_x + total_main, main_gap),
                        JustifyContent::Center => {
                            (content_x + (content_w + total_main) / 2.0, main_gap)
                        }
                        JustifyContent::SpaceBetween if n > 1.0 => {
                            (content_x + content_w, main_gap + free / (n - 1.0))
                        }
                        JustifyContent::SpaceAround => (
                            content_x + content_w - free / (2.0 * n),
                            main_gap + free / n,
                        ),
                        JustifyContent::SpaceEvenly => (
                            content_x + content_w - free / (n + 1.0),
                            main_gap + free / (n + 1.0),
                        ),
                        _ => (content_x + content_w, main_gap),
                    }
                } else {
                    match justify {
                        JustifyContent::Start => (content_x, main_gap),
                        JustifyContent::End => (content_x + free, main_gap),
                        JustifyContent::Center => (content_x + free / 2.0, main_gap),
                        JustifyContent::SpaceBetween if n > 1.0 => {
                            (content_x, main_gap + free / (n - 1.0))
                        }
                        JustifyContent::SpaceAround => {
                            (content_x + free / (2.0 * n), main_gap + free / n)
                        }
                        JustifyContent::SpaceEvenly => {
                            (content_x + free / (n + 1.0), main_gap + free / (n + 1.0))
                        }
                        _ => (content_x, main_gap),
                    }
                };
                let row_h: f32 = child_heights.iter().copied().fold(0.0, f32::max);
                // CSS Flexbox §8.1: the cross-axis extent of a single-line
                // flex container is the container's own content height (when
                // it has a definite height) rather than just the tallest row
                // item. This matters for align-self: end / center when the
                // container is taller than its children — e.g. a 100px
                // container with 30px children: End should reach y=70, not 0.
                let cross_h = resolve_content_height(b, row_h, content_w, container_h);
                for (i, child) in b.children.iter_mut().enumerate() {
                    if !in_flow[i] {
                        continue;
                    }
                    // Per CSS Flexbox L1 §8.3: align-self OVERRIDES the
                    // container's align-items per item (also see align-self
                    // doc in flex-line branch above).
                    let item_align = child.align_self.unwrap_or(align);
                    // Cross-axis stretch: when align-items: stretch and the
                    // child has no explicit height, re-place at cross_h
                    // (clamped to the child's own min-height).
                    if matches!(item_align, AlignItems::Stretch)
                        && child.explicit_height.is_none()
                        && cross_h > child_heights[i]
                    {
                        let min_h = child
                            .min_height
                            .map(|m| m.resolve(container_h))
                            .unwrap_or(0.0);
                        let stretch_h = cross_h.max(min_h);
                        child.flex_override_height = Some(stretch_h);
                        place(
                            child,
                            content_x,
                            content_y,
                            child_widths[i],
                            container_h,
                            ctx,
                            parent_fs,
                        );
                        child.flex_override_height = None;
                        child_heights[i] = child.margin_rect().h;
                    }
                    let cy = match item_align {
                        AlignItems::Start | AlignItems::Stretch | AlignItems::Baseline => {
                            content_y
                        }
                        AlignItems::Center => content_y + (cross_h - child_heights[i]) / 2.0,
                        AlignItems::End => content_y + cross_h - child_heights[i],
                    };
                    // Leading auto-margin share (Flexbox §8.1). Forward: the
                    // item's left auto margin pushes it right; reverse: the right.
                    let lead_auto = if reverse {
                        child.margin_auto.right
                    } else {
                        child.margin_auto.left
                    };
                    if lead_auto && auto_share > 0.0 {
                        if reverse {
                            cursor -= auto_share;
                        } else {
                            cursor += auto_share;
                        }
                    }
                    let target_x = if reverse {
                        cursor - child_widths[i]
                    } else {
                        cursor
                    };
                    let dx = target_x - child.margin_rect().x;
                    let dy = cy - child.margin_rect().y;
                    shift_box(child, dx, dy);
                    if reverse {
                        cursor -= child_widths[i] + between;
                    } else {
                        cursor += child_widths[i] + between;
                    }
                    // Trailing auto-margin share consumes space AFTER the item.
                    let trail_auto = if reverse {
                        child.margin_auto.left
                    } else {
                        child.margin_auto.right
                    };
                    if trail_auto && auto_share > 0.0 {
                        if reverse {
                            cursor -= auto_share;
                        } else {
                            cursor += auto_share;
                        }
                    }
                }
                b.content = Rect {
                    x: content_x,
                    y: content_y,
                    w: content_w,
                    h: cross_h,
                };
            }
        }
        FlexDirection::Column | FlexDirection::ColumnReverse => {
            let reverse = matches!(direction, FlexDirection::ColumnReverse);
            let has_definite_main = b.flex_override_height.is_some() || b.explicit_height.is_some();
            // Pre-compute the container's definite main size so we can resolve
            // percentage flex-basis values against it. Per CSS Flexbox §9.2,
            // a percentage flex-basis resolves against the flex container's
            // inner main size. If that size is indefinite, the percentage
            // must be treated as `auto` (not resolved against the viewport).
            let definite_container_main = if has_definite_main {
                Some(resolve_content_height(b, 0.0, content_w, container_h))
            } else {
                None
            };
            // Column behaves like normal block stacking but lets us still
            // honour justify-content / align-items in the cross axis (x).
            let mut child_heights: Vec<f32> = Vec::with_capacity(b.children.len());
            let mut child_widths: Vec<f32> = Vec::with_capacity(b.children.len());
            for (i, child) in b.children.iter_mut().enumerate() {
                if !in_flow[i] {
                    child_widths.push(0.0);
                    child_heights.push(0.0);
                    continue;
                }
                // Per CSS Flexbox §9.2: flex-basis on a column item resolves
                // against the flex container's definite main size.
                // - Percentage flex-basis: only valid when the container has a
                //   definite height; otherwise treat as `auto` (None here).
                // - Non-percentage flex-basis (px / calc): only applied when the
                //   container is definite so the grow/shrink distribution step
                //   (apply_flex_column_sizing) can use it as the base size.
                //   When the container is indefinite, use the natural height so
                //   intrinsic content (e.g. flex-basis:0 + flex-grow:1 + inner
                //   content of 180px) isn't collapsed to the basis size.
                let flex_basis_h = if child.explicit_height.is_none() {
                    child.flex_basis.and_then(|basis| {
                        if basis.is_percent_based() {
                            // Resolve % against the container's own definite
                            // height, NOT the viewport (container_h). Treat as
                            // `auto` when the container height is indefinite.
                            definite_container_main.map(|dh| basis.resolve(dh).max(0.0))
                        } else {
                            // Non-percent: only apply when the container is
                            // definite (same guard as the original code).
                            definite_container_main
                                .map(|_| basis.resolve(container_h).max(0.0))
                        }
                    })
                } else {
                    None
                };
                if let Some(basis_h) = flex_basis_h {
                    child.flex_override_height = Some(basis_h);
                    place(
                        child,
                        content_x,
                        content_y,
                        content_w,
                        container_h,
                        ctx,
                        parent_fs,
                    );
                    child.flex_override_height = None;
                } else {
                    place(
                        child,
                        content_x,
                        content_y,
                        content_w,
                        container_h,
                        ctx,
                        parent_fs,
                    );
                }
                let mr = child.margin_rect();
                child_widths.push(mr.w);
                child_heights.push(mr.h);
            }
            let indices: Vec<usize> = in_flow
                .iter()
                .enumerate()
                .filter_map(|(i, &flow)| flow.then_some(i))
                .collect();
            let natural_total_main = flex_column_total_main(&indices, &child_heights, main_gap);
            // Only a definite height participates in flex grow/shrink
            // distribution along the main axis. A column with
            // `min-height: 100vh` but no explicit `height` should be free
            // to grow taller than the viewport when its children need it.
            // Reuse the pre-computed definite_container_main from above.
            let definite_main = definite_container_main;
            let total_main = if let Some(available_main) = definite_main {
                apply_flex_column_sizing(
                    b,
                    &indices,
                    &mut child_widths,
                    &mut child_heights,
                    available_main,
                    content_x,
                    content_y,
                    content_w,
                    container_h,
                    main_gap,
                    ctx,
                    parent_fs,
                )
            } else {
                natural_total_main
            };
            let flex_content_h = resolve_content_height(b, total_main, content_w, container_h);
            let free = (flex_content_h - total_main).max(0.0);
            let n = inflow_count.max(1) as f32;
            let (mut cursor, between) = if reverse {
                // Per CSS Flexbox L1 §9.5, column-reverse maps main-start
                // to the BOTTOM and main-end to the TOP. With our column
                // walker `target_y = cursor - h_i` and `cursor -= h+gap`,
                // items grow upward from `cursor`. So:
                //   - Start  → cursor at the bottom edge of the available
                //              content area (= content_y + flex_content_h)
                //   - End    → cursor at content_y + total_main (= items at
                //              the top, free space falls below them)
                //   - center → midpoint between those two
                // Previously Start/End/SpaceBetween/SpaceAround/SpaceEvenly
                // mirrored the non-reverse formulas, which packed items
                // against the WRONG edge (Start at top, End at bottom —
                // exactly inverted) and pushed SpaceBetween items off the
                // bottom of the container. Now the analogue of the
                // row-reverse formulas (content_x ↔ content_w replaced by
                // content_y ↔ flex_content_h) drives placement.
                match justify {
                    JustifyContent::Start => (content_y + flex_content_h, main_gap),
                    JustifyContent::End => (content_y + total_main, main_gap),
                    JustifyContent::Center => {
                        (content_y + (flex_content_h + total_main) / 2.0, main_gap)
                    }
                    JustifyContent::SpaceBetween if n > 1.0 => {
                        (content_y + flex_content_h, main_gap + free / (n - 1.0))
                    }
                    JustifyContent::SpaceAround => (
                        content_y + flex_content_h - free / (2.0 * n),
                        main_gap + free / n,
                    ),
                    JustifyContent::SpaceEvenly => (
                        content_y + flex_content_h - free / (n + 1.0),
                        main_gap + free / (n + 1.0),
                    ),
                    _ => (content_y + flex_content_h, main_gap),
                }
            } else {
                match justify {
                    JustifyContent::Start => (content_y, main_gap),
                    JustifyContent::End => (content_y + free, main_gap),
                    JustifyContent::Center => (content_y + free / 2.0, main_gap),
                    JustifyContent::SpaceBetween if n > 1.0 => {
                        (content_y, main_gap + free / (n - 1.0))
                    }
                    JustifyContent::SpaceAround => {
                        (content_y + free / (2.0 * n), main_gap + free / n)
                    }
                    JustifyContent::SpaceEvenly => {
                        (content_y + free / (n + 1.0), main_gap + free / (n + 1.0))
                    }
                    _ => (content_y, main_gap),
                }
            };
            for (i, child) in b.children.iter_mut().enumerate() {
                if !in_flow[i] {
                    continue;
                }
                // Per CSS Flexbox L1 §8.3: align-self overrides align-items
                // per item, also in the column direction.
                let item_align = child.align_self.unwrap_or(align);
                // Cross-axis stretch (column direction): when align-items:
                // stretch and the child has no explicit width, re-place at
                // content_w (clamped to the child's own min-width).
                if matches!(item_align, AlignItems::Stretch)
                    && child.explicit_width.is_none()
                    && content_w > child_widths[i]
                {
                    let min_w = child.min_width.map(|m| m.resolve(content_w)).unwrap_or(0.0);
                    let stretch_w = content_w.max(min_w);
                    child.flex_override_width = Some(stretch_w);
                    place(
                        child,
                        content_x,
                        content_y,
                        stretch_w,
                        container_h,
                        ctx,
                        parent_fs,
                    );
                    child.flex_override_width = None;
                    let mr = child.margin_rect();
                    child_widths[i] = mr.w;
                    child_heights[i] = mr.h;
                }
                let cx = match item_align {
                    AlignItems::Start | AlignItems::Stretch | AlignItems::Baseline => content_x,
                    AlignItems::Center => content_x + (content_w - child_widths[i]) / 2.0,
                    AlignItems::End => content_x + content_w - child_widths[i],
                };
                let target_y = if reverse {
                    cursor - child_heights[i]
                } else {
                    cursor
                };
                let dx = cx - child.margin_rect().x;
                let dy = target_y - child.margin_rect().y;
                shift_box(child, dx, dy);
                if reverse {
                    cursor -= child_heights[i] + between;
                } else {
                    cursor += child_heights[i] + between;
                }
            }
            b.content = Rect {
                x: content_x,
                y: content_y,
                w: content_w,
                h: flex_content_h,
            };
        }
    }
    // Out-of-flow second pass — same shape as the block-layout path.
    // Containing block is the padding box per CSS Position §4.
    let cb = b.padding_rect();
    for (i, child) in b.children.iter_mut().enumerate() {
        if in_flow[i] {
            continue;
        }
        place_absolute(child, &cb, ctx, parent_fs);
    }
}

/// Resolve a column track list into pixel widths against the container
/// width. `fr` units split the leftover space after fixed/pct tracks +
/// gaps. `Auto` is treated like `Fr(1.0)`.
fn resolve_tracks(tracks: &[GridTrack], container: f32, total_gap: f32) -> Vec<f32> {
    if tracks.is_empty() {
        return Vec::new();
    }
    let mut widths = vec![0.0f32; tracks.len()];
    let mut fr_sum = 0.0f32;
    let mut min_floors = vec![0.0f32; tracks.len()];
    let mut fixed_total = 0.0f32;
    for (i, t) in tracks.iter().enumerate() {
        match t {
            GridTrack::Px(v) => {
                widths[i] = v.max(0.0);
                fixed_total += widths[i];
            }
            GridTrack::Pct(p) => {
                widths[i] = (container * p / 100.0).max(0.0);
                fixed_total += widths[i];
            }
            GridTrack::Fr(f) => {
                fr_sum += f.max(0.0);
            }
            GridTrack::Auto => {
                fr_sum += 1.0;
            }
            GridTrack::AutoRepeat(_) => {}
            GridTrack::Subgrid => {
                // Subgrid inherits sizing from parent — treat as auto here.
                fr_sum += 1.0;
            }
            GridTrack::MinMax { min, max } => {
                // The MAX bound drives sizing (delegating to the
                // matching Px/Pct/Fr/Auto branch above), the MIN bound
                // acts as a hard floor applied after the fr
                // distribution pass. For `minmax(0, 1fr)` (Tailwind's
                // grid-cols-N), max=Fr(1.0) → fr distribution, min=Px(0)
                // → no floor; the track shrinks to its 1fr share and
                // long content wraps inside instead of growing the
                // column past its share.
                let floor = match min {
                    MinMaxBound::Px(v) => v.max(0.0),
                    MinMaxBound::Pct(p) => (container * p / 100.0).max(0.0),
                    MinMaxBound::Fr(_) | MinMaxBound::Auto => 0.0,
                };
                min_floors[i] = floor;
                match max {
                    MinMaxBound::Px(v) => {
                        widths[i] = v.max(floor);
                        fixed_total += widths[i];
                    }
                    MinMaxBound::Pct(p) => {
                        widths[i] = (container * p / 100.0).max(floor);
                        fixed_total += widths[i];
                    }
                    MinMaxBound::Fr(f) => {
                        fr_sum += f.max(0.0);
                    }
                    MinMaxBound::Auto => {
                        fr_sum += 1.0;
                    }
                }
            }
        }
    }
    let leftover = (container - fixed_total - total_gap).max(0.0);
    if fr_sum > 0.0 {
        for (i, t) in tracks.iter().enumerate() {
            match t {
                GridTrack::Fr(f) => widths[i] = leftover * f.max(0.0) / fr_sum,
                GridTrack::Auto => widths[i] = leftover / fr_sum,
                GridTrack::AutoRepeat(_) => {}
                GridTrack::MinMax { max, .. } => match max {
                    MinMaxBound::Fr(f) => {
                        widths[i] = (leftover * f.max(0.0) / fr_sum).max(min_floors[i]);
                    }
                    MinMaxBound::Auto => {
                        widths[i] = (leftover / fr_sum).max(min_floors[i]);
                    }
                    _ => {}
                },
                _ => {}
            }
        }
    }
    widths
}

fn expand_auto_repeat_tracks(
    tracks: &[GridTrack],
    container: f32,
    gap: f32,
    inflow_count: usize,
) -> Vec<GridTrack> {
    let mut out = Vec::new();
    for track in tracks {
        match track {
            GridTrack::AutoRepeat(auto) => {
                let pattern_len = auto.tracks.len().max(1);
                let capacity = if auto.min_px > 0.0 {
                    (((container + gap) / (auto.min_px + gap)).floor() as usize).max(1)
                } else {
                    1
                };
                let count = match auto.mode {
                    AutoRepeatMode::Fill => capacity,
                    AutoRepeatMode::Fit => {
                        let used_patterns = ((inflow_count.max(1)) + pattern_len - 1) / pattern_len;
                        capacity.min(used_patterns)
                    }
                };
                for _ in 0..count {
                    for repeated in &auto.tracks {
                        out.push(match *repeated {
                            AutoRepeatTrack::Px(v) => GridTrack::Px(v),
                            AutoRepeatTrack::Pct(v) => GridTrack::Pct(v),
                            AutoRepeatTrack::Fr(v) => GridTrack::Fr(v),
                            AutoRepeatTrack::Auto => GridTrack::Auto,
                        });
                    }
                }
            }
            other => out.push(other.clone()),
        }
    }
    if out.is_empty() {
        vec![GridTrack::Auto]
    } else {
        out
    }
}

/// Lay a grid container's children. V1 supports `grid-template-columns`
/// with px / % / fr / auto tracks (and `repeat()`), `column-gap` /
/// `row-gap`, and auto-flow row. No `grid-column`/`grid-row` item
/// placement, no spans, no implicit columns beyond the template, no
/// `grid-template-rows` driven row sizing (rows auto-grow to fit their
/// tallest child).
/// Walk the `grid-template-areas` matrix and compute the bounding
/// rectangle of each named area. A name spanning a non-rectangular
/// region is invalid per the spec; we accept the bounding rect anyway
/// (over-spans rather than dropping the area).
fn compute_named_areas(
    grid: &[Vec<String>],
) -> std::collections::HashMap<String, (usize, usize, usize, usize)> {
    let mut out: std::collections::HashMap<String, (usize, usize, usize, usize)> =
        std::collections::HashMap::new();
    for (r, row) in grid.iter().enumerate() {
        for (c, cell) in row.iter().enumerate() {
            if cell == "." || cell.is_empty() {
                continue;
            }
            let entry = out.entry(cell.clone()).or_insert((r, c, 1, 1));
            // Extend the existing bounding rect to cover (r, c).
            let r0 = entry.0.min(r);
            let c0 = entry.1.min(c);
            let r1 = (entry.0 + entry.2).max(r + 1);
            let c1 = (entry.1 + entry.3).max(c + 1);
            *entry = (r0, c0, r1 - r0, c1 - c0);
        }
    }
    out
}

fn place_grid(
    b: &mut LayoutBox,
    content_x: f32,
    content_y: f32,
    content_w: f32,
    container_h: f32,
    ctx: &mut LayoutCtx<'_>,
) {
    fn spans_fit(
        occupied: &std::collections::HashSet<(usize, usize)>,
        row: usize,
        col: usize,
        row_span: usize,
        col_span: usize,
        ncols: usize,
    ) -> bool {
        if col >= ncols || col + col_span > ncols {
            return false;
        }
        for rr in row..row + row_span {
            for cc in col..col + col_span {
                if occupied.contains(&(rr, cc)) {
                    return false;
                }
            }
        }
        true
    }

    fn occupy_span(
        occupied: &mut std::collections::HashSet<(usize, usize)>,
        row: usize,
        col: usize,
        row_span: usize,
        col_span: usize,
    ) {
        for rr in row..row + row_span {
            for cc in col..col + col_span {
                occupied.insert((rr, cc));
            }
        }
    }

    let parent_fs = b.font_size_px;
    let cgap = b.column_gap_px;
    let rgap = b.row_gap_px;
    let n = b.children.len();
    let in_flow: Vec<bool> = b.children.iter().map(is_in_flow).collect();
    let inflow_count = in_flow.iter().filter(|&&flow| flow).count();
    // Resolve `grid-template-areas` named areas (if any) into a map from
    // name → (row_start, col_start, row_span, col_span). Each child with
    // a matching `grid_area_name` gets its placement filled in before the
    // auto-flow walk runs, so it lands in the right rectangle.
    let named_areas = b
        .grid_template_areas
        .as_ref()
        .map(|grid| compute_named_areas(grid))
        .unwrap_or_default();
    if !named_areas.is_empty() {
        for child in b.children.iter_mut() {
            if let Some(name) = &child.grid_area_name {
                if let Some(&(r, c, rs, cs)) = named_areas.get(name) {
                    // Grid lines are 1-based externally; the auto-flow
                    // path subtracts 1 again to get 0-based, so we add 1
                    // here to keep the convention consistent.
                    child.grid_row_start = Some(r + 1);
                    child.grid_column_start = Some(c + 1);
                    child.grid_row_span = Some(rs);
                    child.grid_column_span = Some(cs);
                }
            }
        }
    }
    // Default to a single auto column when no template specified.
    let raw_tracks: Vec<GridTrack> = b.grid_template_columns.clone().unwrap_or_else(|| {
        // If the container declared template-areas without explicit
        // columns, infer one auto column per cell of the first row.
        if let Some(areas) = &b.grid_template_areas {
            if let Some(first) = areas.first() {
                return vec![GridTrack::Auto; first.len()];
            }
        }
        vec![GridTrack::Auto]
    });
    let tracks = expand_auto_repeat_tracks(&raw_tracks, content_w, cgap, inflow_count);
    let ncols = tracks.len().max(1);
    let total_cgap = cgap * (ncols.saturating_sub(1) as f32);
    let col_widths = resolve_tracks(&tracks, content_w, total_cgap);
    let col_xs: Vec<f32> = {
        let mut xs = Vec::with_capacity(ncols);
        let mut cx = content_x;
        for (i, w) in col_widths.iter().enumerate() {
            xs.push(cx);
            cx += w;
            if i + 1 < ncols {
                cx += cgap;
            }
        }
        xs
    };

    // Stage 1: natural-size every in-flow child into its column,
    // then group by row to compute row heights as max(child_height)
    // per row. Out-of-flow (absolute / fixed) children are skipped
    // entirely from grid auto-placement — they get positioned in the
    // dedicated second pass below.
    let mut flow_pos: Vec<Option<(usize, usize)>> = vec![None; n];
    let mut occupied: std::collections::HashSet<(usize, usize)> = std::collections::HashSet::new();
    for i in 0..n {
        if !in_flow[i] {
            continue;
        }
        let row_span = b.children[i].grid_row_span.unwrap_or(1).max(1);
        let col_span = b.children[i]
            .grid_column_span
            .unwrap_or(1)
            .max(1)
            .min(ncols);
        let row = b.children[i]
            .grid_row_start
            .and_then(|start| start.checked_sub(1));
        let col = b.children[i]
            .grid_column_start
            .and_then(|start| start.checked_sub(1));
        if let (Some(r), Some(c)) = (row, col) {
            let start_col = c.min(ncols.saturating_sub(1));
            let pos = (r, start_col.min(ncols.saturating_sub(col_span)));
            flow_pos[i] = Some(pos);
            occupy_span(&mut occupied, pos.0, pos.1, row_span, col_span);
        } else if let Some(r) = row {
            let mut c = 0usize;
            while c < ncols && !spans_fit(&occupied, r, c, row_span, col_span, ncols) {
                c += 1;
            }
            let pos = (r, c.min(ncols.saturating_sub(col_span)));
            flow_pos[i] = Some(pos);
            occupy_span(&mut occupied, pos.0, pos.1, row_span, col_span);
        } else if let Some(c) = col {
            let col = c.min(ncols.saturating_sub(col_span));
            let mut r = 0usize;
            while !spans_fit(&occupied, r, col, row_span, col_span, ncols) {
                r += 1;
            }
            let pos = (r, col);
            flow_pos[i] = Some(pos);
            occupy_span(&mut occupied, pos.0, pos.1, row_span, col_span);
        }
    }
    // Auto-flow index: only advances on in-flow children without an
    // explicit grid position, so normal packing skips reserved cells.
    let mut next = 0usize;
    for i in 0..n {
        if !in_flow[i] || flow_pos[i].is_some() {
            continue;
        }
        let row_span = b.children[i].grid_row_span.unwrap_or(1).max(1);
        let col_span = b.children[i]
            .grid_column_span
            .unwrap_or(1)
            .max(1)
            .min(ncols);
        while !spans_fit(
            &occupied,
            next / ncols,
            next % ncols,
            row_span,
            col_span,
            ncols,
        ) {
            next += 1;
        }
        let pos = (next / ncols, next % ncols);
        flow_pos[i] = Some(pos);
        occupy_span(&mut occupied, pos.0, pos.1, row_span, col_span);
        next += 1;
    }
    let inflow_count = flow_pos.iter().filter(|pos| pos.is_some()).count();
    let mut heights: Vec<f32> = vec![0.0; n];
    for i in 0..n {
        if let Some((_, c)) = flow_pos[i] {
            let col_span = b.children[i]
                .grid_column_span
                .unwrap_or(1)
                .max(1)
                .min(ncols - c);
            let cw = col_widths[c..c + col_span].iter().sum::<f32>()
                + cgap * (col_span.saturating_sub(1) as f32);
            // Inline-axis (justify) alignment of a grid item. Per CSS
            // Grid §11.2 the initial value of `justify-items`/`justify-self`
            // is `normal`, which behaves as `stretch` for grid items — and
            // it is independent of the block-axis `align-items`. So an item
            // with no explicit width fills its column track by default.
            let item_justify = b.children[i]
                .justify_self
                .unwrap_or(b.justify_items.unwrap_or(AlignItems::Stretch));
            let measure_w = if !matches!(item_justify, AlignItems::Stretch)
                && b.children[i].explicit_width.is_none()
            {
                intrinsic_margin_width(&b.children[i], cw, parent_fs)
                    .min(cw)
                    .max(0.0)
            } else {
                cw
            };
            let child = &mut b.children[i];
            place(
                child,
                col_xs[c],
                content_y,
                measure_w,
                container_h,
                ctx,
                parent_fs,
            );
            heights[i] = child.margin_rect().h;
        }
    }
    let flow_rows = flow_pos
        .iter()
        .filter_map(|pos| pos.map(|(r, _)| r + 1))
        .max()
        .unwrap_or(0);
    let raw_row_tracks: Vec<GridTrack> = b.grid_template_rows.clone().unwrap_or_default();
    let row_tracks = expand_auto_repeat_tracks(&raw_row_tracks, container_h, rgap, inflow_count);
    let nrows = flow_rows.max(row_tracks.len());
    let total_rgap = rgap * (nrows.saturating_sub(1) as f32);

    // Row-track sizing per CSS Grid §11:
    //   1) Px / Pct tracks resolve to their fixed size.
    //   2) Auto tracks (and implicit rows from auto-flow beyond the
    //      template) size to the max natural content height of items
    //      placed in them.
    //   3) Fr tracks share the *leftover* space, after fixed + auto +
    //      gap consume their share.  Crucially the leftover can be
    //      ZERO or NEGATIVE if auto rows already exceeded the container —
    //      in that case Fr tracks collapse to 0 (their `minmax(0, 1fr)`
    //      minimum) and the grid grows past its container.
    //
    // Old code seeded Fr rows with the *full* container leftover and
    // then took max() with content, so a `minmax(0, 1fr)` row plus an
    // auto-row containing content always summed to (container + content) —
    // pushing the doodle wrapper to take a full viewport-tall track on
    // Google's home page and shoving the search box ~200 px lower than
    // it should be.
    let mut row_heights: Vec<f32> = vec![0.0; nrows];

    // 1 + 2: classify tracks and pre-size fixed / auto.
    enum RowKind {
        Fixed,
        Fr(f32),
        Auto,
    }
    // Implicit tracks beyond the explicit template count use
    // `grid-auto-rows` (per CSS Grid §7.3.1) when set. Fall back to
    // Auto so children expand to content height.
    let auto_row: Option<&GridTrack> = b.grid_auto_rows.as_ref();
    let mut row_kind: Vec<RowKind> = Vec::with_capacity(nrows);
    for i in 0..nrows {
        let track = row_tracks.get(i).or(if i >= row_tracks.len() {
            auto_row
        } else {
            None
        });
        match track {
            Some(GridTrack::Px(v)) => {
                row_heights[i] = v.max(0.0);
                row_kind.push(RowKind::Fixed);
            }
            Some(GridTrack::Pct(p)) => {
                row_heights[i] = (container_h * *p / 100.0).max(0.0);
                row_kind.push(RowKind::Fixed);
            }
            Some(GridTrack::Fr(f)) => {
                row_kind.push(RowKind::Fr(*f));
            }
            Some(GridTrack::Auto) | None => {
                row_kind.push(RowKind::Auto);
            }
            Some(GridTrack::AutoRepeat(_)) => {
                // expand_auto_repeat_tracks should have removed these
                row_kind.push(RowKind::Auto);
            }
            Some(GridTrack::Subgrid) => {
                row_kind.push(RowKind::Auto);
            }
            Some(GridTrack::MinMax { max, .. }) => {
                // For row sizing, delegate to the max bound. The min
                // floor is enforced by the column-track pass; on rows
                // we expand to content for auto/fr (no overflow) so
                // the explicit min floor matters less. Px max gives
                // a fixed row; Fr max participates in fr distribution.
                match max {
                    MinMaxBound::Px(v) => {
                        row_heights[i] = v.max(0.0);
                        row_kind.push(RowKind::Fixed);
                    }
                    MinMaxBound::Pct(p) => {
                        row_heights[i] = (container_h * *p / 100.0).max(0.0);
                        row_kind.push(RowKind::Fixed);
                    }
                    MinMaxBound::Fr(f) => {
                        row_kind.push(RowKind::Fr(*f));
                    }
                    MinMaxBound::Auto => {
                        row_kind.push(RowKind::Auto);
                    }
                }
            }
        }
    }

    // Sum natural heights per row from item content (max across items
    // that share the row; spanning items distribute their height).
    for i in 0..n {
        if let Some((r, _)) = flow_pos[i] {
            let row_span = b.children[i]
                .grid_row_span
                .unwrap_or(1)
                .max(1)
                .min(nrows - r);
            let occupied_gaps = rgap * (row_span.saturating_sub(1) as f32);
            let required_total = (heights[i] - occupied_gaps).max(0.0);
            let share = required_total / row_span as f32;
            for rr in r..r + row_span {
                // Auto and Fr rows receive content height; Fixed rows
                // are not enlarged by content (they keep their declared
                // size).  This matches CSS Grid spec: fixed tracks are
                // fixed.  Fr will be replaced below with leftover share
                // but we still seed it with content as a floor.
                if !matches!(row_kind[rr], RowKind::Fixed) {
                    row_heights[rr] = row_heights[rr].max(share);
                }
            }
        }
    }

    // 3: distribute leftover among Fr rows.  Compute consumed by
    // non-Fr (fixed + auto-with-content) then split the rest by fr
    // weight.  Fr rows that already have content height keep at least
    // that — Fr's `minmax(0, 1fr)` minimum lets them grow but never
    // shrink below content.
    let mut fr_indices: Vec<(usize, f32)> = Vec::new();
    let mut consumed = total_rgap;
    for (idx, kind) in row_kind.iter().enumerate() {
        match kind {
            RowKind::Fr(w) => fr_indices.push((idx, *w)),
            _ => consumed += row_heights[idx],
        }
    }
    let fr_total: f32 = fr_indices.iter().map(|(_, w)| *w).sum();
    if fr_total > 0.0 {
        let leftover = (container_h - consumed).max(0.0);
        for (idx, w) in &fr_indices {
            let share = leftover * w / fr_total;
            row_heights[*idx] = row_heights[*idx].max(share);
        }
    }
    // Stage 2: shift each in-flow child to (col_x, row_y).
    let mut row_ys: Vec<f32> = Vec::with_capacity(nrows);
    {
        let mut ry = content_y;
        for (r, h) in row_heights.iter().enumerate() {
            row_ys.push(ry);
            ry += h;
            if r + 1 < nrows {
                ry += rgap;
            }
        }
    }
    // Per CSS Grid §11, each grid item is aligned within its track
    // cell using `align-self` / `justify-self`, falling back to the
    // container's `align-items` / `justify-items`.
    let container_align = b.align_items;
    // justify_items falls back to align_items when unset — this is
    // also how `place-items: center` (the shorthand) works
    // symmetrically (Google's home page uses it to center the
    // Doodle + search-prompt cluster in the page-height grid track).
    let container_justify = b.justify_items.unwrap_or(container_align);
    for i in 0..n {
        if let Some((r, c)) = flow_pos[i] {
            // Read span + per-item alignment overrides up front so we
            // don't hold a long immutable borrow of `b.children`
            // across the mutable shift_box call below.
            let (row_span, col_span, item_align, item_justify) = {
                let ch = &b.children[i];
                let rs = ch.grid_row_span.unwrap_or(1).max(1).min(nrows - r);
                let cs = ch.grid_column_span.unwrap_or(1).max(1).min(ncols - c);
                (
                    rs,
                    cs,
                    ch.align_self.unwrap_or(container_align),
                    ch.justify_self.unwrap_or(container_justify),
                )
            };
            let cell_h: f32 = (r..r + row_span)
                .map(|rr| row_heights.get(rr).copied().unwrap_or(0.0))
                .sum::<f32>()
                + rgap * (row_span.saturating_sub(1) as f32);
            let cell_w: f32 = (c..c + col_span)
                .map(|cc| col_widths.get(cc).copied().unwrap_or(0.0))
                .sum::<f32>()
                + cgap * (col_span.saturating_sub(1) as f32);
            let child_mr = b.children[i].margin_rect();
            let mut want_x = col_xs[c];
            let mut want_y = row_ys[r];
            match item_align {
                AlignItems::Center if cell_h > child_mr.h => {
                    want_y += (cell_h - child_mr.h) * 0.5;
                }
                AlignItems::End if cell_h > child_mr.h => {
                    want_y += cell_h - child_mr.h;
                }
                _ => {}
            }
            match item_justify {
                AlignItems::Center if cell_w > child_mr.w => {
                    want_x += (cell_w - child_mr.w) * 0.5;
                }
                AlignItems::End if cell_w > child_mr.w => {
                    want_x += cell_w - child_mr.w;
                }
                _ => {}
            }
            let child = &mut b.children[i];
            let dx = want_x - child_mr.x;
            let dy = want_y - child_mr.y;
            shift_box(child, dx, dy);
        }
    }
    let total_h: f32 = row_heights.iter().sum::<f32>() + rgap * (nrows.saturating_sub(1) as f32);
    b.content = Rect {
        x: content_x,
        y: content_y,
        w: content_w,
        h: total_h,
    };
    // Out-of-flow second pass.
    // Containing block is the padding box per CSS Position §4.
    let cb = b.padding_rect();
    for i in 0..n {
        if in_flow[i] {
            continue;
        }
        place_absolute(&mut b.children[i], &cb, ctx, parent_fs);
    }
}

/// Place a `display: table` container. Implements the CSS 2.1 §17
/// auto-width column algorithm:
///
/// 1. Walk children, expanding any `<tbody>`/`<thead>`/`<tfoot>` row
///    groups so all rows sit at the same level (`flatten_rows`).
/// 2. For each row, walk cells in document order to learn the column
///    count.
/// 3. First sizing pass: layout each cell against `container_w` to
///    learn its "max content width" (what we'd render with no
///    constraint). Track the column max-width as the max across
///    its cells. The table's preferred width is the sum.
/// 4. If the preferred width fits in `content_w` we use that.
///    Otherwise we shrink columns proportionally (this approximates
///    the spec's "constraint propagation"; full CSS 2.1 also tracks a
///    min-content width per column which we approximate as the cell
///    content's word-max width).
/// 5. Second pass: re-layout each cell at its final column width.
///    Each row's height is the max of its cells. Position cells
///    side-by-side, rows stacked.
fn place_table(
    b: &mut LayoutBox,
    content_x: f32,
    content_y: f32,
    content_w: f32,
    container_h: f32,
    ctx: &mut LayoutCtx<'_>,
) {
    let parent_fs = b.font_size_px;

    // Step 1 + 2: collect rows (expanding row groups). Each entry is
    // (row_index_in_children, cell_indices_within_row).
    //
    // To keep things ownership-friendly, we walk `b.children` twice:
    // once for row group expansion (recording the indices of all
    // rows in document order), then again to enumerate the cells of
    // each row.

    // Collect (parent_index, row_index_inside) for each row, where
    // parent_index = usize::MAX means "direct table child". We
    // restrict ourselves to one level of row-group nesting (the
    // common case); deeper nesting falls back to treating the inner
    // group as opaque.
    let mut rows: Vec<(usize, Option<usize>)> = Vec::new();
    for (i, c) in b.children.iter().enumerate() {
        if c.is_table_row {
            rows.push((i, None));
        } else if c.is_table_row_group {
            for (j, cc) in c.children.iter().enumerate() {
                if cc.is_table_row {
                    rows.push((i, Some(j)));
                }
            }
        }
        // Non-row, non-row-group children are ignored — strict CSS
        // would wrap them in an anonymous row + cell. We don't.
    }
    if rows.is_empty() {
        // No rows → fall back to block layout so any stray text
        // content still appears.
        let mut child_y = content_y;
        for child in &mut b.children {
            if !is_in_flow(child) {
                continue;
            }
            place(
                child,
                content_x,
                child_y,
                content_w,
                container_h,
                ctx,
                parent_fs,
            );
            child_y += child.margin_rect().h;
        }
        b.content = Rect {
            x: content_x,
            y: content_y,
            w: content_w,
            h: (child_y - content_y).max(0.0),
        };
        return;
    }

    // Determine column count = max number of cells across rows.
    let cols_in_row = |b: &LayoutBox, row: (usize, Option<usize>)| -> usize {
        let row_box = match row.1 {
            Some(j) => &b.children[row.0].children[j],
            None => &b.children[row.0],
        };
        row_box
            .children
            .iter()
            .filter(|c| c.is_table_cell)
            .map(|c| c.table_col_span.unwrap_or(1).max(1))
            .sum()
    };
    let ncols = rows.iter().map(|&r| cols_in_row(b, r)).max().unwrap_or(0);
    if ncols == 0 {
        b.content = Rect {
            x: content_x,
            y: content_y,
            w: content_w,
            h: 0.0,
        };
        return;
    }

    // Step 3: content-based column widths (CSS 2.1 auto table layout,
    // approximated). Earlier this used EVEN widths (`content_w / ncols`),
    // which made a narrow cell (e.g. a `1.` rank) hog 1/N of the row and
    // shoved wide cells (titles) far to the right — Hacker News, MediaWiki,
    // and most layout tables rendered wrong. We now measure each cell's
    // min/max-content width with the READ-ONLY intrinsic helpers (these
    // don't `place`, so they avoid the state-divergence that a second
    // layout pass caused), take each column's max, then fit to `content_w`.
    // The final placement is still a single `place` pass at the chosen
    // widths, so the determinism the even-width hack bought is preserved.
    let mut col_max = vec![0f32; ncols];
    let mut col_min = vec![0f32; ncols];
    for &row in &rows {
        let row_box = match row.1 {
            Some(j) => &b.children[row.0].children[j],
            None => &b.children[row.0],
        };
        let mut ci = 0usize;
        for cell in row_box.children.iter().filter(|c| c.is_table_cell) {
            if ci >= ncols {
                break;
            }
            let span = cell.table_col_span.unwrap_or(1).max(1).min(ncols - ci);
            let mx = intrinsic_margin_width(cell, content_w, parent_fs);
            let mn = min_content_width(cell, parent_fs).min(mx);
            if span == 1 {
                col_max[ci] = col_max[ci].max(mx);
                col_min[ci] = col_min[ci].max(mn);
            } else {
                // A spanning cell only needs its columns to TOTAL its width;
                // spread its demand evenly across the spanned columns as a
                // floor, so single-span cells (which set the real per-column
                // widths, e.g. the title column) still dominate.
                let share_mx = mx / span as f32;
                let share_mn = mn / span as f32;
                for cc in ci..ci + span {
                    col_max[cc] = col_max[cc].max(share_mx);
                    col_min[cc] = col_min[cc].max(share_mn);
                }
            }
            ci += span;
        }
    }
    // Empty table body (all columns measured zero) → even widths so the
    // layout stays sane.
    if col_max.iter().all(|&w| w == 0.0) {
        col_max = vec![content_w / ncols as f32; ncols];
    }
    let total_max: f32 = col_max.iter().sum();
    let total_min: f32 = col_min.iter().sum();
    let col_w: Vec<f32> = if total_max <= content_w {
        // Room to spare: each column gets its preferred width and the slack
        // is handed out in proportion to preferred width (wide columns —
        // the title — absorb most of it, narrow rank/arrow cells stay tight).
        let extra = content_w - total_max;
        if total_max > 0.0 {
            col_max
                .iter()
                .map(|&m| m + extra * (m / total_max))
                .collect()
        } else {
            vec![content_w / ncols as f32; ncols]
        }
    } else if total_min < content_w {
        // Between min and max: distribute the room above min-content by each
        // column's flexibility (max − min).
        let room = content_w - total_min;
        let span: f32 = col_max
            .iter()
            .zip(&col_min)
            .map(|(mx, mn)| (mx - mn).max(0.0))
            .sum();
        if span > 0.0 {
            col_min
                .iter()
                .zip(&col_max)
                .map(|(&mn, &mx)| mn + room * ((mx - mn).max(0.0) / span))
                .collect()
        } else {
            col_min.clone()
        }
    } else {
        // Doesn't fit even at min-content: use min widths (table overflows,
        // matching browser behaviour).
        col_min.clone()
    };
    let col_xs: Vec<f32> = {
        let mut xs = Vec::with_capacity(ncols);
        let mut x = content_x;
        for w in &col_w {
            xs.push(x);
            x += w;
        }
        xs
    };

    // Step 4a: first placement pass — place ALL cells (to determine their
    // intrinsic heights), but compute row_heights using only rowspan=1 cells.
    // Rowspan>1 cells are placed here so their heights are known for step 4b.
    // Positions assigned here are provisional; step 4c re-places everything.
    let nrows = rows.len();
    let mut row_heights: Vec<f32> = vec![0.0; nrows];
    {
        // rowspan_occ[c] = number of rows (including the current one) that
        // column c is still blocked by a spanning cell placed in a previous
        // row. Protocol:
        //   1. Decrement at the TOP of each row (so the count ticks down as
        //      we enter each new row after the one where the cell was placed).
        //   2. When placing a cell with row_span=N, store N (NOT N-1) so that
        //      the decrement at the start of the NEXT row correctly leaves N-1
        //      remaining. E.g. row_span=2 in row 0 → occ=2; row 1 start
        //      decrements to 1 (>0 → blocked); row 2 start decrements to 0
        //      (not blocked).
        let mut rowspan_occ: Vec<usize> = vec![0; ncols];
        let mut tmp_y = content_y;
        for (ri, &row) in rows.iter().enumerate() {
            // Tick down occupancy at the start of each row.
            for occ in rowspan_occ.iter_mut() {
                *occ = occ.saturating_sub(1);
            }
            let row_box: &mut LayoutBox = match row.1 {
                Some(j) => &mut b.children[row.0].children[j],
                None => &mut b.children[row.0],
            };
            let mut ci = 0usize;
            for cell in row_box.children.iter_mut() {
                if !cell.is_table_cell {
                    continue;
                }
                if ci >= ncols {
                    break;
                }
                // Skip columns occupied by a rowspan cell from a previous row.
                while ci < ncols && rowspan_occ[ci] > 0 {
                    ci += 1;
                }
                if ci >= ncols {
                    break;
                }
                let col_span = cell.table_col_span.unwrap_or(1).max(1).min(ncols - ci);
                let row_span = cell.table_row_span.unwrap_or(1).max(1).min(nrows - ri);
                let cell_w: f32 = col_w[ci..ci + col_span].iter().sum();
                place(cell, col_xs[ci], tmp_y, cell_w, container_h, ctx, parent_fs);
                // Only rowspan=1 cells drive the base row height.
                if row_span == 1 {
                    row_heights[ri] = row_heights[ri].max(cell.margin_rect().h);
                }
                // Mark the columns this cell spans for the following rows.
                // Store row_span so that the decrement at the top of the next
                // row leaves row_span-1, which is still > 0 when row_span > 1.
                if row_span > 1 {
                    for cc in ci..ci + col_span {
                        if cc < ncols {
                            rowspan_occ[cc] = rowspan_occ[cc].max(row_span);
                        }
                    }
                }
                ci += col_span;
            }
            // Honor an explicit height on the row itself.
            {
                let row_box2: &LayoutBox = match row.1 {
                    Some(j) => &b.children[row.0].children[j],
                    None => &b.children[row.0],
                };
                if let Some(LengthSpec::Px(h)) = row_box2.explicit_height {
                    row_heights[ri] = row_heights[ri].max(h);
                }
            }
            tmp_y += row_heights[ri];

        }
    }

    // Step 4b: rowspan height distribution.
    // For each rowspan>1 cell, if its content height exceeds the combined
    // height of the rows it spans, distribute the excess equally among those
    // rows so they grow to accommodate the spanning cell.
    {
        let mut rowspan_occ: Vec<usize> = vec![0; ncols];
        for (ri, &row) in rows.iter().enumerate() {
            for occ in rowspan_occ.iter_mut() {
                *occ = occ.saturating_sub(1);
            }
            let row_box: &LayoutBox = match row.1 {
                Some(j) => &b.children[row.0].children[j],
                None => &b.children[row.0],
            };
            let mut ci = 0usize;
            for cell in row_box.children.iter().filter(|c| c.is_table_cell) {
                if ci >= ncols {
                    break;
                }
                while ci < ncols && rowspan_occ[ci] > 0 {
                    ci += 1;
                }
                if ci >= ncols {
                    break;
                }
                let col_span = cell.table_col_span.unwrap_or(1).max(1).min(ncols - ci);
                let row_span = cell.table_row_span.unwrap_or(1).max(1).min(nrows - ri);
                if row_span > 1 {
                    let spanned_h: f32 = row_heights[ri..ri + row_span].iter().sum();
                    let cell_h = cell.margin_rect().h;
                    if cell_h > spanned_h {
                        let excess = cell_h - spanned_h;
                        let share = excess / row_span as f32;
                        for rr in ri..ri + row_span {
                            row_heights[rr] += share;
                        }
                    }
                    for cc in ci..ci + col_span {
                        if cc < ncols {
                            rowspan_occ[cc] = rowspan_occ[cc].max(row_span);
                        }
                    }
                }
                ci += col_span;
            }

        }
    }

    // Step 4c: final placement pass — re-place every cell at its correct
    // y position using the now-definitive row_heights. Position row boxes.
    let mut row_y = content_y;
    {
        let mut rowspan_occ: Vec<usize> = vec![0; ncols];
        for (ri, &row) in rows.iter().enumerate() {
            // Tick down occupancy at the start of each row.
            for occ in rowspan_occ.iter_mut() {
                *occ = occ.saturating_sub(1);
            }
            let row_h = row_heights[ri];
            let row_box: &mut LayoutBox = match row.1 {
                Some(j) => &mut b.children[row.0].children[j],
                None => &mut b.children[row.0],
            };
            let mut ci = 0usize;
            for cell in row_box.children.iter_mut() {
                if !cell.is_table_cell {
                    continue;
                }
                if ci >= ncols {
                    break;
                }
                // Skip columns occupied by a rowspan cell from a previous row.
                while ci < ncols && rowspan_occ[ci] > 0 {
                    ci += 1;
                }
                if ci >= ncols {
                    break;
                }
                let col_span = cell.table_col_span.unwrap_or(1).max(1).min(ncols - ci);
                let row_span = cell.table_row_span.unwrap_or(1).max(1).min(nrows - ri);
                let cell_w: f32 = col_w[ci..ci + col_span].iter().sum();
                // For rowspan>1 cells, provide the combined height of all spanned
                // rows as the available height so the cell fills its full slot.
                let cell_avail_h: f32 = row_heights[ri..ri + row_span].iter().sum();
                place(cell, col_xs[ci], row_y, cell_w, cell_avail_h, ctx, parent_fs);
                if row_span > 1 {
                    for cc in ci..ci + col_span {
                        if cc < ncols {
                            rowspan_occ[cc] = rowspan_occ[cc].max(row_span);
                        }
                    }
                }
                ci += col_span;
            }
            // Position the row box itself for hit-testing.
            row_box.content = Rect {
                x: content_x,
                y: row_y,
                w: content_w,
                h: row_h,
            };
            row_y += row_h;

        }
    }

    // If we expanded row-groups, set their content rect to the
    // span of their rows (for hit-testing / paint).
    for (i, c) in b.children.iter_mut().enumerate() {
        if !c.is_table_row_group {
            continue;
        }
        let rows_in_group: Vec<&Rect> = rows
            .iter()
            .filter_map(|&(pi, j)| {
                if pi == i && j.is_some() {
                    Some(&c.children[j.unwrap()].content)
                } else {
                    None
                }
            })
            .collect();
        if rows_in_group.is_empty() {
            continue;
        }
        let min_y = rows_in_group
            .iter()
            .map(|r| r.y)
            .fold(f32::INFINITY, f32::min);
        let max_y = rows_in_group
            .iter()
            .map(|r| r.y + r.h)
            .fold(f32::NEG_INFINITY, f32::max);
        c.content = Rect {
            x: content_x,
            y: min_y,
            w: content_w,
            h: (max_y - min_y).max(0.0),
        };
    }

    b.content = Rect {
        x: content_x,
        y: content_y,
        w: content_w,
        h: (row_y - content_y).max(0.0),
    };
}

/// Translate a layout subtree by (dx, dy). Used by flex placement to
/// move children into their final main/cross-axis slots after a natural
/// sizing pass.
fn shift_box(b: &mut LayoutBox, dx: f32, dy: f32) {
    b.content.x += dx;
    b.content.y += dy;
    for c in &mut b.children {
        shift_box(c, dx, dy);
    }
}

/// Visually scale a box and its whole subtree by `(sx, sy)` around the
/// point `(cx, cy)` — the box's transform origin. Mirrors CSS
/// `transform: scale(...)`, which is a paint-time visual effect (no
/// reflow), so we bake it straight into the laid-out geometry: every
/// rect position is scaled about the origin, every box dimension and
/// font size is multiplied. Border widths / radii use the mean scale
/// (exact for the common uniform `scale(n)` case).
fn scale_box(b: &mut LayoutBox, cx: f32, cy: f32, sx: f32, sy: f32) {
    let mean = (sx + sy) * 0.5;
    b.content.x = cx + (b.content.x - cx) * sx;
    b.content.y = cy + (b.content.y - cy) * sy;
    b.content.w *= sx;
    b.content.h *= sy;
    b.padding.left *= sx;
    b.padding.right *= sx;
    b.padding.top *= sy;
    b.padding.bottom *= sy;
    b.margin.left *= sx;
    b.margin.right *= sx;
    b.margin.top *= sy;
    b.margin.bottom *= sy;
    b.border_width_px *= mean;
    for w in b.border_widths_per_side.iter_mut().flatten() {
        *w *= mean;
    }
    b.border_radius_px *= mean;
    b.font_size_px *= sy;
    for c in &mut b.children {
        scale_box(c, cx, cy, sx, sy);
    }
}

/// Apply CSS `transform` scale (from `scale()`/`scaleX`/`scaleY` and the
/// scale component of `matrix(...)`) as a visual transform once layout
/// is final. Processed bottom-up so a child's own scale is baked in
/// before its parent's scale composes over the subtree. `translate(...)`
/// stays in the paint offset; rotation / skew aren't representable as a
/// rect scale and are handled (when present) by the compositor path, so
/// they are skipped here rather than approximated.
fn apply_visual_transforms(b: &mut LayoutBox) {
    for c in &mut b.children {
        apply_visual_transforms(c);
    }
    if let Some(p) = b.translate_x_percent {
        b.translate_x_px += b.content.w * p / 100.0;
    }
    if let Some(p) = b.translate_y_percent {
        b.translate_y_px += b.content.h * p / 100.0;
    }
    // `scale_x`/`scale_y` default to 1.0; treat 0 (unset) as identity.
    let sx = if b.scale_x == 0.0 { 1.0 } else { b.scale_x };
    let sy = if b.scale_y == 0.0 { 1.0 } else { b.scale_y };
    // Only scale when there's no rotation to honour — a rotated box can't
    // be represented by scaling axis-aligned rects, and scaling it would
    // distort rather than rotate.
    if b.rotate_deg.abs() < 1e-3
        && b.matrix_2d.is_none()
        && ((sx - 1.0).abs() > 1e-4 || (sy - 1.0).abs() > 1e-4)
    {
        let br = b.border_rect();
        let cx = br.x + br.w * 0.5;
        let cy = br.y + br.h * 0.5;
        scale_box(b, cx, cy, sx, sy);
    }
}

pub fn dump(b: &LayoutBox, depth: usize) -> String {
    let mut s = String::new();
    dump_into(b, depth, &mut s);
    s
}

fn dump_into(b: &LayoutBox, depth: usize, s: &mut String) {
    for _ in 0..depth {
        s.push(' ');
    }
    s.push_str(&format!("{b}"));
    if let Some(bg) = b.background {
        s.push_str(&format!(" bg=#{:02x}{:02x}{:02x}", bg.r, bg.g, bg.b));
    }
    s.push('\n');
    for c in &b.children {
        dump_into(c, depth + 2, s);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn block(tag: &str, style: Style, kids: Vec<StyledNode>) -> StyledNode {
        StyledNode {
            kind: StyledKind::Element {
                tag: tag.to_string(),
            },
            style,
            children: kids,
        }
    }

    fn text(t: &str) -> StyledNode {
        StyledNode {
            kind: StyledKind::Text(t.to_string()),
            style: Style::default(),
            children: Vec::new(),
        }
    }

    // ---- Flex geometry regression net ----
    // These lock in flex placement invariants BEFORE the single-pass
    // rewrite, so a regression in the new code path is caught here.
    // Invariant-based (relative position / fill / fit) rather than exact
    // pixels, so they're robust to sub-pixel rounding.
    fn flex_box(w: f32, h: f32, grow: f32, shrink: f32) -> StyledNode {
        block(
            "div",
            Style {
                display: Some(Display::Block),
                width: Some(LengthSpec::Px(w)),
                height: Some(LengthSpec::Px(h)),
                flex_grow: Some(grow),
                flex_shrink: Some(shrink),
                ..Style::default()
            },
            vec![],
        )
    }

    fn flex_container(
        dir: FlexDirection,
        wrap: FlexWrap,
        w: f32,
        kids: Vec<StyledNode>,
    ) -> StyledNode {
        block(
            "div",
            Style {
                display: Some(Display::Flex),
                flex_direction: Some(dir),
                flex_wrap: Some(wrap),
                width: Some(LengthSpec::Px(w)),
                ..Style::default()
            },
            kids,
        )
    }

    #[test]
    fn flex_row_no_grow_left_to_right() {
        let doc = flex_container(
            FlexDirection::Row,
            FlexWrap::NoWrap,
            500.0,
            vec![
                flex_box(100.0, 50.0, 0.0, 0.0),
                flex_box(100.0, 50.0, 0.0, 0.0),
                flex_box(100.0, 50.0, 0.0, 0.0),
            ],
        );
        let root = layout(&doc, &LayoutConfig::default());
        let c = &root.children;
        assert_eq!(c.len(), 3);
        assert!(c[1].content.x > c[0].content.x, "left-to-right");
        assert!(c[2].content.x > c[1].content.x, "left-to-right");
        assert!((c[0].content.y - c[1].content.y).abs() < 1.0, "same row");
        for ch in c {
            assert!(
                (ch.content.w - 100.0).abs() < 1.5,
                "natural width preserved (no grow)"
            );
        }
    }

    #[test]
    fn flex_row_grow_fills_container() {
        let doc = flex_container(
            FlexDirection::Row,
            FlexWrap::NoWrap,
            600.0,
            vec![
                flex_box(100.0, 50.0, 1.0, 1.0),
                flex_box(100.0, 50.0, 1.0, 1.0),
            ],
        );
        let root = layout(&doc, &LayoutConfig::default());
        let c = &root.children;
        assert!(c[0].content.w > 100.0, "grew past natural");
        assert!(
            (c[0].content.w + c[1].content.w - 600.0).abs() < 3.0,
            "grow fills the container (got {} + {})",
            c[0].content.w,
            c[1].content.w
        );
    }

    #[test]
    fn flex_row_shrink_when_overflow() {
        let doc = flex_container(
            FlexDirection::Row,
            FlexWrap::NoWrap,
            300.0,
            vec![
                flex_box(200.0, 50.0, 0.0, 1.0),
                flex_box(200.0, 50.0, 0.0, 1.0),
            ],
        );
        let root = layout(&doc, &LayoutConfig::default());
        let c = &root.children;
        assert!(c[0].content.w < 200.0, "shrank below natural");
        assert!(
            c[0].content.w + c[1].content.w <= 302.0,
            "shrinks to fit container"
        );
    }

    #[test]
    fn flex_column_stacks_vertically() {
        let doc = flex_container(
            FlexDirection::Column,
            FlexWrap::NoWrap,
            200.0,
            vec![
                flex_box(50.0, 30.0, 0.0, 0.0),
                flex_box(50.0, 30.0, 0.0, 0.0),
            ],
        );
        let root = layout(&doc, &LayoutConfig::default());
        let c = &root.children;
        assert!(c[1].content.y > c[0].content.y, "column stacks vertically");
    }

    #[test]
    fn flex_wrap_to_multiple_lines() {
        let doc = flex_container(
            FlexDirection::Row,
            FlexWrap::Wrap,
            250.0,
            vec![
                flex_box(100.0, 40.0, 0.0, 0.0),
                flex_box(100.0, 40.0, 0.0, 0.0),
                flex_box(100.0, 40.0, 0.0, 0.0),
            ],
        );
        let root = layout(&doc, &LayoutConfig::default());
        let c = &root.children;
        assert!(
            c[2].content.y > c[0].content.y,
            "third item wraps to the next line"
        );
    }

    #[test]
    fn flex_nested_row_lays_out() {
        let inner = block(
            "div",
            Style {
                display: Some(Display::Flex),
                flex_direction: Some(FlexDirection::Row),
                flex_grow: Some(1.0),
                ..Style::default()
            },
            vec![
                flex_box(50.0, 30.0, 0.0, 0.0),
                flex_box(50.0, 30.0, 0.0, 0.0),
            ],
        );
        let doc = flex_container(
            FlexDirection::Row,
            FlexWrap::NoWrap,
            400.0,
            vec![inner, flex_box(50.0, 30.0, 0.0, 0.0)],
        );
        let root = layout(&doc, &LayoutConfig::default());
        let outer = &root.children;
        assert_eq!(outer.len(), 2);
        let inner_box = &outer[0];
        assert_eq!(inner_box.children.len(), 2);
        assert!(
            inner_box.children[1].content.x > inner_box.children[0].content.x,
            "nested flex children lay out left-to-right"
        );
    }

    // Regression: layout must measure the post-`text-transform` glyphs
    // (what paint draws), not the raw source. Measuring lower-case "Block"
    // for an `uppercase` run under-sized the box and clipped "BLOCK" at
    // paint (explorer.hyvechain.com Orbitron table headers).
    #[test]
    fn text_transform_applied_before_measurement() {
        assert_eq!(
            apply_text_transform("Block", Some(TextTransform::Uppercase)).as_ref(),
            "BLOCK"
        );
        assert_eq!(
            apply_text_transform("HASH", Some(TextTransform::Lowercase)).as_ref(),
            "hash"
        );
        assert_eq!(
            apply_text_transform("hello world", Some(TextTransform::Capitalize)).as_ref(),
            "Hello World"
        );
        assert_eq!(apply_text_transform("Mixed", None).as_ref(), "Mixed");
    }

    #[test]
    fn block_stacks_vertically() {
        let doc = block(
            "body",
            Style {
                display: Some(Display::Block),
                ..Style::default()
            },
            vec![
                block(
                    "h1",
                    Style {
                        display: Some(Display::Block),
                        font_size_px: Some(32.0),
                        ..Style::default()
                    },
                    vec![text("Hello")],
                ),
                block(
                    "p",
                    Style {
                        display: Some(Display::Block),
                        ..Style::default()
                    },
                    vec![text("body text")],
                ),
            ],
        );
        let cfg = LayoutConfig::default();
        let root = layout(&doc, &cfg);
        assert!(root.content.x.abs() < 1e-3);
        assert!(root.content.y.abs() < 1e-3);
        assert_eq!(root.children.len(), 2);
        let h1 = &root.children[0];
        let p = &root.children[1];
        assert!(p.content.y > h1.content.y);
    }

    #[test]
    fn nested_inline_text_spans_stay_on_one_line() {
        let doc = block(
            "body",
            Style {
                display: Some(Display::Block),
                width: Some(LengthSpec::Px(400.0)),
                ..Style::default()
            },
            vec![StyledNode {
                kind: StyledKind::Element {
                    tag: "span".to_string(),
                },
                style: Style {
                    display: Some(Display::Inline),
                    font_size_px: Some(28.0),
                    ..Style::default()
                },
                children: vec![
                    text("HYVE"),
                    StyledNode {
                        kind: StyledKind::Element {
                            tag: "span".to_string(),
                        },
                        style: Style {
                            display: Some(Display::Inline),
                            font_size_px: Some(28.0),
                            ..Style::default()
                        },
                        children: vec![text("CHAIN")],
                    },
                ],
            }],
        );

        let root = layout(&doc, &LayoutConfig::default());
        let outer = &root.children[0];
        assert_eq!(outer.children.len(), 2);
        let hyve = &outer.children[0];
        let chain_wrapper = &outer.children[1];
        assert!(
            chain_wrapper.content.x > hyve.content.x + hyve.content.w - 1.0,
            "nested inline content should flow to the right, got x={} after {}",
            chain_wrapper.content.x,
            hyve.content.x + hyve.content.w
        );
    }

    #[test]
    fn aspect_ratio_sets_minimum_height_from_width() {
        let doc = block(
            "body",
            Style {
                display: Some(Display::Block),
                ..Style::default()
            },
            vec![block(
                "div",
                Style {
                    display: Some(Display::Block),
                    width: Some(LengthSpec::Px(200.0)),
                    aspect_ratio: Some(1.0),
                    ..Style::default()
                },
                Vec::new(),
            )],
        );
        let root = layout(&doc, &LayoutConfig::default());
        let child = &root.children[0];
        assert!((child.content.w - 200.0).abs() < 0.01);
        assert!((child.content.h - 200.0).abs() < 0.01);
    }

    #[test]
    fn embedded_image_honors_max_width_and_scales_height() {
        let doc = block(
            "body",
            Style {
                display: Some(Display::Block),
                ..Style::default()
            },
            vec![StyledNode {
                kind: StyledKind::Element {
                    tag: "img".to_string(),
                },
                style: Style {
                    display: Some(Display::Block),
                    max_width: Some(LengthSpec::Pct(100.0)),
                    embedded_image: Some(std::sync::Arc::new(EmbeddedImage {
                        width: 2048,
                        height: 1024,
                        pixels: Vec::new(),
                    })),
                    ..Style::default()
                },
                children: Vec::new(),
            }],
        );
        let root = layout(&doc, &LayoutConfig::default());
        let child = &root.children[0];
        assert!((child.content.w - 1024.0).abs() < 0.01);
        assert!((child.content.h - 512.0).abs() < 0.01);
    }

    #[test]
    fn percent_height_resolves_against_parent_height() {
        let doc = block(
            "body",
            Style {
                display: Some(Display::Block),
                ..Style::default()
            },
            vec![block(
                "div",
                Style {
                    display: Some(Display::Block),
                    width: Some(LengthSpec::Px(200.0)),
                    height: Some(LengthSpec::Px(300.0)),
                    ..Style::default()
                },
                vec![block(
                    "div",
                    Style {
                        display: Some(Display::Block),
                        width: Some(LengthSpec::Pct(100.0)),
                        height: Some(LengthSpec::Pct(100.0)),
                        ..Style::default()
                    },
                    Vec::new(),
                )],
            )],
        );
        let root = layout(&doc, &LayoutConfig::default());
        let parent = &root.children[0];
        let child = &parent.children[0];
        assert!((parent.content.h - 300.0).abs() < 0.01);
        assert!(
            (child.content.h - 300.0).abs() < 0.01,
            "expected child height 300, got {}",
            child.content.h
        );
    }

    #[test]
    fn table_lays_out_two_rows_three_cells() {
        // 3-col, 2-row table inside a 900px viewport. After table
        // layout each cell's `content.y` reflects its row.
        let cell = |label: &str| {
            block(
                "td",
                Style {
                    display: Some(Display::TableCell),
                    ..Style::default()
                },
                vec![text(label)],
            )
        };
        let row = |labels: &[&str]| {
            block(
                "tr",
                Style {
                    display: Some(Display::TableRow),
                    ..Style::default()
                },
                labels.iter().map(|s| cell(s)).collect(),
            )
        };
        let table = block(
            "table",
            Style {
                display: Some(Display::Table),
                ..Style::default()
            },
            vec![row(&["A", "B", "C"]), row(&["D", "E", "F"])],
        );
        let mut cfg = LayoutConfig::default();
        cfg.viewport_w = 900.0;
        let root = layout(&table, &cfg);
        assert_eq!(root.children.len(), 2);
        let r1 = &root.children[0];
        let r2 = &root.children[1];
        assert_eq!(r1.children.len(), 3);
        assert_eq!(r2.children.len(), 3);
        // Each cell sits at its row's y.
        for c in &r1.children {
            assert!(
                (c.content.y - r1.content.y).abs() < 1.0,
                "row 1 cell y should equal row y, got cell y={}, row y={}",
                c.content.y,
                r1.content.y,
            );
        }
        // Row 2 is below row 1.
        assert!(
            r2.content.y > r1.content.y + 5.0,
            "row 2 (y={}) should be below row 1 (y={})",
            r2.content.y,
            r1.content.y
        );
        // Cells progress left → right in row 1.
        assert!(r1.children[0].content.x < r1.children[1].content.x);
        assert!(r1.children[1].content.x < r1.children[2].content.x);
    }

    #[test]
    fn table_colspan_places_following_cell_in_correct_column() {
        // Mirrors Hacker News: a 3-column table whose second row starts with a
        // `colspan=2` cell, so the next cell must land in column 3 (under the
        // title), not column 2. Also checks content-based widths (narrow rank
        // column, wide title column) instead of the old even-width split.
        let cell = |label: &str, span: Option<usize>| {
            block(
                "td",
                Style {
                    display: Some(Display::TableCell),
                    table_col_span: span,
                    ..Style::default()
                },
                vec![text(label)],
            )
        };
        let row = |cells: Vec<StyledNode>| {
            block(
                "tr",
                Style {
                    display: Some(Display::TableRow),
                    ..Style::default()
                },
                cells,
            )
        };
        let table = block(
            "table",
            Style {
                display: Some(Display::Table),
                ..Style::default()
            },
            vec![
                row(vec![
                    cell("1.", None),
                    cell("^", None),
                    cell("A very long story title goes here for sizing", None),
                ]),
                row(vec![cell("", Some(2)), cell("99 points by user", None)]),
            ],
        );
        let mut cfg = LayoutConfig::default();
        cfg.viewport_w = 900.0;
        let root = layout(&table, &cfg);
        let title_row = &root.children[0];
        let subtext_row = &root.children[1];
        // The subtext cell (after a colspan=2) must start at the same x as the
        // title cell (column 3) — i.e. directly under the title.
        let title_x = title_row.children[2].content.x;
        let subtext_x = subtext_row.children[1].content.x;
        assert!(
            (title_x - subtext_x).abs() < 1.0,
            "subtext (x={subtext_x}) should align under title (x={title_x}) via colspan",
        );
        // Content-based widths: the title column dwarfs the narrow rank column.
        let rank_w = title_row.children[0].content.w;
        let title_w = title_row.children[2].content.w;
        assert!(
            title_w > rank_w * 3.0,
            "title column ({title_w}) should be far wider than rank column ({rank_w})",
        );
    }

    #[test]
    fn table_rowspan_expands_row_heights() {
        // A 2-column, 2-row table where the first cell spans both rows.
        // The "Tall cell" has a fixed 80px height; row 1 and row 2 each get
        // a short cell. The two rows combined must be at least 80px tall so
        // the spanning cell fits.
        //
        //   +----------+-------+
        //   | Tall     | Row 1 |  <- row 0
        //   | (80px)   +-------+
        //   |          | Row 2 |  <- row 1
        //   +----------+-------+
        let tall_cell = block(
            "td",
            Style {
                display: Some(Display::TableCell),
                table_row_span: Some(2),
                height: Some(LengthSpec::Px(80.0)),
                ..Style::default()
            },
            vec![text("Tall")],
        );
        let short_cell = |label: &str| {
            block(
                "td",
                Style {
                    display: Some(Display::TableCell),
                    ..Style::default()
                },
                vec![text(label)],
            )
        };
        let row0 = block(
            "tr",
            Style {
                display: Some(Display::TableRow),
                ..Style::default()
            },
            vec![tall_cell, short_cell("Row 1")],
        );
        let row1 = block(
            "tr",
            Style {
                display: Some(Display::TableRow),
                ..Style::default()
            },
            vec![short_cell("Row 2")],
        );
        let table = block(
            "table",
            Style {
                display: Some(Display::Table),
                ..Style::default()
            },
            vec![row0, row1],
        );
        let mut cfg = LayoutConfig::default();
        cfg.viewport_w = 400.0;
        let root = layout(&table, &cfg);
        assert_eq!(root.children.len(), 2, "table must have 2 rows");
        let r0 = &root.children[0];
        let r1 = &root.children[1];
        // The two row heights combined must accommodate the 80px tall cell.
        let combined_h = r0.content.h + r1.content.h;
        assert!(
            combined_h >= 79.0,
            "combined row heights ({combined_h}) must be >= 80px to fit the rowspan cell",
        );
        // Row 1 must start below row 0.
        assert!(
            r1.content.y >= r0.content.y + r0.content.h - 0.5,
            "row 1 (y={}) must start at or below row 0 bottom ({})",
            r1.content.y,
            r0.content.y + r0.content.h,
        );
        // The tall cell must sit at row 0's y.
        let tall = &r0.children[0];
        assert!(
            (tall.content.y - r0.content.y).abs() < 1.0,
            "tall cell y ({}) must equal row 0 y ({})",
            tall.content.y,
            r0.content.y,
        );
    }

    #[test]
    fn table_rowspan_cell_in_second_row_skips_occupied_column() {
        // A 2-column, 2-row table where the FIRST column of row 0 has
        // rowspan=2. Row 1 has a single cell; it must be placed in column 1,
        // NOT column 0 (which is still occupied by the spanning cell).
        //
        //   col 0         | col 1
        //   +-------------+----------+
        //   | Span (rs=2) | Row0-C1  |   row 0
        //   |             +----------+
        //   |             | Row1-C1  |   row 1  <-- must land at col 1, not 0
        //   +-------------+----------+
        let spanning_cell = block(
            "td",
            Style {
                display: Some(Display::TableCell),
                table_row_span: Some(2),
                ..Style::default()
            },
            vec![text("Span")],
        );
        let cell = |label: &str| {
            block(
                "td",
                Style {
                    display: Some(Display::TableCell),
                    ..Style::default()
                },
                vec![text(label)],
            )
        };
        let row0 = block(
            "tr",
            Style { display: Some(Display::TableRow), ..Style::default() },
            vec![spanning_cell, cell("Row0-C1")],
        );
        let row1 = block(
            "tr",
            Style { display: Some(Display::TableRow), ..Style::default() },
            // Only one cell here — must skip the occupied col 0 and land at col 1.
            vec![cell("Row1-C1")],
        );
        let table = block(
            "table",
            Style { display: Some(Display::Table), ..Style::default() },
            vec![row0, row1],
        );
        let mut cfg = LayoutConfig::default();
        cfg.viewport_w = 400.0;
        let root = layout(&table, &cfg);
        let r0 = &root.children[0];
        let r1 = &root.children[1];
        // col 1 cell in row 0 and col 1 cell in row 1 must share the same x.
        let r0_c1_x = r0.children[1].content.x;
        let r1_c0_x = r1.children[0].content.x;
        assert!(
            (r0_c1_x - r1_c0_x).abs() < 1.0,
            "row1 cell x ({r1_c0_x}) must align with row0 col-1 x ({r0_c1_x}): rowspan should skip col 0"
        );
        // The row 1 cell must not overlap with col 0 (x > 0).
        assert!(
            r1_c0_x > 1.0,
            "row1 cell ({r1_c0_x}) must be in col 1, not col 0 (x≈0)"
        );
    }

    #[test]
    fn inline_block_siblings_flow_horizontally() {
        // Three 200px inline-block boxes inside a 1024px container.
        // They should sit on the same line (200 + 200 + 200 = 600 < 1024).
        let kid = |w| {
            block(
                "span",
                Style {
                    display: Some(Display::InlineBlock),
                    width: Some(LengthSpec::Px(w)),
                    height: Some(LengthSpec::Px(20.0)),
                    ..Style::default()
                },
                vec![text("k")],
            )
        };
        let container = block(
            "div",
            Style {
                display: Some(Display::Block),
                ..Style::default()
            },
            vec![kid(200.0), kid(200.0), kid(200.0)],
        );
        let cfg = LayoutConfig::default();
        let root = layout(&container, &cfg);
        assert_eq!(root.children.len(), 3);
        let xs: Vec<f32> = root.children.iter().map(|c| c.content.x).collect();
        let ys: Vec<f32> = root.children.iter().map(|c| c.content.y).collect();
        // All three on the same y.
        assert!(
            (ys[0] - ys[1]).abs() < 1.0 && (ys[1] - ys[2]).abs() < 1.0,
            "all three inline-block siblings should share a y, got {ys:?}"
        );
        // x progresses left → right.
        assert!(
            xs[0] < xs[1] && xs[1] < xs[2],
            "inline-block siblings should flow left → right, got {xs:?}"
        );
    }

    #[test]
    fn inline_block_wraps_when_line_overflows() {
        // Four 400px inline-block boxes inside a 1024px container —
        // two fit per line (800 ≤ 1024), so we expect two lines of two.
        let kid = || {
            block(
                "span",
                Style {
                    display: Some(Display::InlineBlock),
                    width: Some(LengthSpec::Px(400.0)),
                    height: Some(LengthSpec::Px(20.0)),
                    ..Style::default()
                },
                vec![text("k")],
            )
        };
        let container = block(
            "div",
            Style {
                display: Some(Display::Block),
                ..Style::default()
            },
            vec![kid(), kid(), kid(), kid()],
        );
        let cfg = LayoutConfig::default();
        let root = layout(&container, &cfg);
        let ys: Vec<f32> = root.children.iter().map(|c| c.content.y).collect();
        // Items 0 and 1 share a y, items 2 and 3 share a y, and the
        // second line is below the first.
        assert!(
            (ys[0] - ys[1]).abs() < 1.0,
            "first line of two should share y, got {ys:?}"
        );
        assert!(
            (ys[2] - ys[3]).abs() < 1.0,
            "second line of two should share y, got {ys:?}"
        );
        assert!(
            ys[2] > ys[1] + 10.0,
            "second line must be below first, got y[1]={} y[2]={}",
            ys[1],
            ys[2]
        );
    }

    #[test]
    fn nested_inline_children_flow_horizontally_inside_inline_parent() {
        let nested = block(
            "span",
            Style {
                display: Some(Display::Inline),
                font_size_px: Some(20.0),
                font_weight_bold: Some(true),
                ..Style::default()
            },
            vec![
                block(
                    "span",
                    Style {
                        display: Some(Display::Inline),
                        font_size_px: Some(20.0),
                        font_weight_bold: Some(true),
                        ..Style::default()
                    },
                    vec![text("HYVE")],
                ),
                block(
                    "span",
                    Style {
                        display: Some(Display::Inline),
                        font_size_px: Some(20.0),
                        font_weight_bold: Some(true),
                        ..Style::default()
                    },
                    vec![text("CHAIN")],
                ),
            ],
        );
        let container = block(
            "div",
            Style {
                display: Some(Display::Block),
                ..Style::default()
            },
            vec![nested],
        );
        let cfg = LayoutConfig::default();
        let root = layout(&container, &cfg);
        let outer = &root.children[0];
        let left = &outer.children[0];
        let right = &outer.children[1];
        assert!(
            (left.content.y - right.content.y).abs() < 1.0,
            "nested inline children should share a line, got left y={} right y={}",
            left.content.y,
            right.content.y
        );
        assert!(
            right.content.x > left.content.x,
            "nested inline children should advance horizontally, got left x={} right x={}",
            left.content.x,
            right.content.x
        );
    }

    #[test]
    fn grid_template_areas_places_named_children() {
        // Vector-2022 minimum shape: a "header sidebar main" grid where
        // each child carries `grid-area: <name>` and the parent declares
        // template-areas + 3 columns.
        let header = block(
            "header",
            Style {
                display: Some(Display::Block),
                grid_area_name: Some("hdr".into()),
                height: Some(LengthSpec::Px(60.0)),
                ..Style::default()
            },
            vec![],
        );
        let sidebar = block(
            "nav",
            Style {
                display: Some(Display::Block),
                grid_area_name: Some("side".into()),
                height: Some(LengthSpec::Px(400.0)),
                ..Style::default()
            },
            vec![],
        );
        let main = block(
            "main",
            Style {
                display: Some(Display::Block),
                grid_area_name: Some("main".into()),
                height: Some(LengthSpec::Px(400.0)),
                ..Style::default()
            },
            vec![],
        );
        let container = block(
            "div",
            Style {
                display: Some(Display::Grid),
                grid_template_columns: Some(vec![GridTrack::Px(200.0), GridTrack::Auto]),
                grid_template_areas: Some(vec![
                    vec!["hdr".into(), "hdr".into()],
                    vec!["side".into(), "main".into()],
                ]),
                ..Style::default()
            },
            vec![header, sidebar, main],
        );
        let cfg = LayoutConfig {
            viewport_w: 1024.0,
            ..LayoutConfig::default()
        };
        let root = layout(&container, &cfg);
        let hdr = &root.children[0];
        let side = &root.children[1];
        let main = &root.children[2];
        // Header spans both columns → covers full container width.
        assert!(
            (hdr.content.w - 1024.0).abs() < 1.0,
            "header should span both columns (1024 wide), got {}",
            hdr.content.w
        );
        // Sidebar in column 0 (200px), main in column 1 (rest).
        assert!(side.content.x.abs() < 1.0, "side at x=0");
        assert!(
            (side.content.w - 200.0).abs() < 1.0,
            "side w=200, got {}",
            side.content.w
        );
        assert!((main.content.x - 200.0).abs() < 1.0, "main starts at x=200");
        // Sidebar and main sit below the header.
        assert!(side.content.y > hdr.content.y);
        assert!((side.content.y - main.content.y).abs() < 1.0);
    }

    #[test]
    fn float_right_pins_to_edge_and_text_wraps_around_it() {
        // 1024-px container with an infobox-style float: right, 200×400.
        // Two paragraphs follow; both should sit at content_x and have
        // their effective width reduced to (1024 - 200) = 824 while
        // their y is below the float's top and above its bottom.
        let float_box = block(
            "aside",
            Style {
                display: Some(Display::Block),
                float_side: FloatSide::Right,
                width: Some(LengthSpec::Px(200.0)),
                height: Some(LengthSpec::Px(400.0)),
                ..Style::default()
            },
            vec![],
        );
        // Paragraphs need an explicit height so the test doesn't
        // depend on the text-measurement heuristic.
        let p = || {
            block(
                "p",
                Style {
                    display: Some(Display::Block),
                    height: Some(LengthSpec::Px(50.0)),
                    ..Style::default()
                },
                vec![],
            )
        };
        let container = block(
            "div",
            Style {
                display: Some(Display::Block),
                ..Style::default()
            },
            vec![float_box, p(), p()],
        );
        let cfg = LayoutConfig::default();
        let root = layout(&container, &cfg);
        let viewport = cfg.viewport_w;
        let float_box = &root.children[0];
        let p1 = &root.children[1];
        let p2 = &root.children[2];
        // Float pinned to right edge.
        assert!(
            (float_box.content.x + float_box.content.w - viewport).abs() < 1.0,
            "float right-edge should equal viewport width ({viewport}), got x+w={}",
            float_box.content.x + float_box.content.w
        );
        // Paragraphs start at content_x = 0 (no left float).
        assert!(
            p1.content.x.abs() < 1.0,
            "p1 should start at x=0, got {}",
            p1.content.x
        );
        // Paragraphs are narrowed by the float's 200px width.
        assert!(
            (p1.content.w - (viewport - 200.0)).abs() < 1.0,
            "p1 width should be viewport-200={}, got {}",
            viewport - 200.0,
            p1.content.w
        );
        // First paragraph sits at the top.
        assert!(
            p1.content.y.abs() < 1.0,
            "p1 y should be 0, got {}",
            p1.content.y
        );
        // Second paragraph stacks below first, both still inside the
        // float's vertical range, so both are narrowed.
        assert!(p2.content.y > 40.0 && p2.content.y < 60.0);
        assert!(
            (p2.content.w - (viewport - 200.0)).abs() < 1.0,
            "p2 width should still be narrowed while inside float range"
        );
    }

    #[test]
    fn vertical_align_super_raises_inline_box() {
        // Two equal inline-block siblings; the second has vertical-align:
        // super. After layout the second sibling should sit at a smaller
        // (higher) y than the first because super raises it.
        let baseline_kid = block(
            "span",
            Style {
                display: Some(Display::InlineBlock),
                width: Some(LengthSpec::Px(50.0)),
                height: Some(LengthSpec::Px(30.0)),
                font_size_px: Some(20.0),
                ..Style::default()
            },
            vec![],
        );
        let super_kid = block(
            "sup",
            Style {
                display: Some(Display::InlineBlock),
                width: Some(LengthSpec::Px(50.0)),
                height: Some(LengthSpec::Px(30.0)),
                font_size_px: Some(20.0),
                vertical_align: VerticalAlign::Super,
                ..Style::default()
            },
            vec![],
        );
        let container = block(
            "div",
            Style {
                display: Some(Display::Block),
                ..Style::default()
            },
            vec![baseline_kid, super_kid],
        );
        let cfg = LayoutConfig::default();
        let root = layout(&container, &cfg);
        let y_baseline = root.children[0].content.y;
        let y_super = root.children[1].content.y;
        assert!(
            y_super < y_baseline - 1.0,
            "super-aligned sibling should sit higher (smaller y) than baseline sibling: \
             baseline.y={y_baseline}, super.y={y_super}"
        );
    }

    #[test]
    fn clear_both_jumps_block_past_float_bottom() {
        // Float right 100×100, then a `clear: both` block. The block's
        // top must be at y=100 (the float's bottom), not y=0.
        let f = block(
            "aside",
            Style {
                display: Some(Display::Block),
                float_side: FloatSide::Right,
                width: Some(LengthSpec::Px(100.0)),
                height: Some(LengthSpec::Px(100.0)),
                ..Style::default()
            },
            vec![],
        );
        let cleared = block(
            "div",
            Style {
                display: Some(Display::Block),
                clear: ClearMode::Both,
                height: Some(LengthSpec::Px(10.0)),
                ..Style::default()
            },
            vec![],
        );
        let container = block(
            "div",
            Style {
                display: Some(Display::Block),
                ..Style::default()
            },
            vec![f, cleared],
        );
        let cfg = LayoutConfig::default();
        let root = layout(&container, &cfg);
        let cleared_y = root.children[1].content.y;
        assert!(
            (cleared_y - 100.0).abs() < 1.0,
            "cleared block should sit at y=100, got {cleared_y}"
        );
    }

    #[test]
    fn baseline_aligns_short_and_tall_inline_blocks() {
        // A 50px-tall image sitting next to a 20px-tall inline-block
        // should have its bottom aligned to the tall image's bottom
        // (both treated as `vertical-align: baseline` → baseline at
        // margin-box bottom). We expect the shorter box to sit at a
        // higher y than the taller box BUT their bottoms to coincide.
        let tall = block(
            "span",
            Style {
                display: Some(Display::InlineBlock),
                width: Some(LengthSpec::Px(50.0)),
                height: Some(LengthSpec::Px(50.0)),
                ..Style::default()
            },
            vec![],
        );
        let short = block(
            "span",
            Style {
                display: Some(Display::InlineBlock),
                width: Some(LengthSpec::Px(50.0)),
                height: Some(LengthSpec::Px(20.0)),
                ..Style::default()
            },
            vec![],
        );
        let container = block(
            "div",
            Style {
                display: Some(Display::Block),
                ..Style::default()
            },
            vec![tall, short],
        );
        let cfg = LayoutConfig::default();
        let root = layout(&container, &cfg);
        let tall_box = &root.children[0];
        let short_box = &root.children[1];
        let tall_bottom = tall_box.content.y + tall_box.content.h;
        let short_bottom = short_box.content.y + short_box.content.h;
        assert!(
            (tall_bottom - short_bottom).abs() < 1.0,
            "inline-block siblings should baseline-align (share bottom): \
             tall={tall_bottom}, short={short_bottom}"
        );
        // And the short one must sit lower (larger y) than the tall one.
        assert!(
            short_box.content.y > tall_box.content.y,
            "shorter sibling should sit below taller one's top: \
             tall.y={} short.y={}",
            tall_box.content.y,
            short_box.content.y
        );
    }

    #[test]
    fn position_absolute_takes_child_out_of_flow() {
        // Two block children — first is absolute, second is in-flow.
        // The in-flow child should sit at content_y = 0 because the
        // absolute sibling above must NOT push it down (taking 0
        // vertical space in flow is the defining property of
        // out-of-flow positioning). The absolute child should appear
        // at (left, top) inside its containing block.
        let abs_kid = block(
            "div",
            Style {
                display: Some(Display::Block),
                position: Some(Position::Absolute),
                left_px: Some(LengthSpec::Px(50.0)),
                top_px: Some(LengthSpec::Px(30.0)),
                width: Some(LengthSpec::Px(100.0)),
                ..Style::default()
            },
            vec![text("abs")],
        );
        let inflow_kid = block(
            "div",
            Style {
                display: Some(Display::Block),
                ..Style::default()
            },
            vec![text("in-flow")],
        );
        let container = block(
            "div",
            Style {
                display: Some(Display::Block),
                position: Some(Position::Relative),
                ..Style::default()
            },
            vec![abs_kid, inflow_kid],
        );
        let cfg = LayoutConfig::default();
        let root = layout(&container, &cfg);
        assert_eq!(root.children.len(), 2);
        let abs = &root.children[0];
        let inflow = &root.children[1];
        // In-flow child sits at the container's content_y (no abs
        // sibling pushed it down). For the root container at (0,0),
        // that's y = 0.
        assert!(
            inflow.content.y.abs() < 1.0,
            "in-flow sibling must NOT be pushed down by abs sibling, got y={}",
            inflow.content.y
        );
        // Absolute child is at (50, 30) relative to container's
        // content rect (which starts at 0, 0 here).
        assert!(
            (abs.content.x - 50.0).abs() < 1.0,
            "abs child should be at x=50, got {}",
            abs.content.x
        );
        assert!(
            (abs.content.y - 30.0).abs() < 1.0,
            "abs child should be at y=30, got {}",
            abs.content.y
        );
    }

    #[test]
    fn position_absolute_resolves_against_nearest_positioned_ancestor() {
        // grandparent(relative) > parent(static, offset) > child(absolute).
        // The absolute child's containing block is the *grandparent*
        // (nearest positioned ancestor), NOT the static parent, so its
        // top/left resolve against the grandparent's content origin (0,0)
        // and the parent's 40/50 margin must NOT shift it.
        let abs_kid = block(
            "div",
            Style {
                display: Some(Display::Block),
                position: Some(Position::Absolute),
                left_px: Some(LengthSpec::Px(20.0)),
                top_px: Some(LengthSpec::Px(10.0)),
                width: Some(LengthSpec::Px(100.0)),
                ..Style::default()
            },
            vec![text("abs")],
        );
        let static_parent = block(
            "div",
            Style {
                display: Some(Display::Block),
                margin: EdgeSizes {
                    top: 50.0,
                    right: 0.0,
                    bottom: 0.0,
                    left: 40.0,
                },
                ..Style::default()
            },
            vec![abs_kid],
        );
        let grandparent = block(
            "div",
            Style {
                display: Some(Display::Block),
                position: Some(Position::Relative),
                width: Some(LengthSpec::Px(400.0)),
                height: Some(LengthSpec::Px(300.0)),
                ..Style::default()
            },
            vec![static_parent],
        );
        let root = layout(&grandparent, &LayoutConfig::default());
        let abs = &root.children[0].children[0];
        assert!(
            (abs.content.x - 20.0).abs() < 1.0,
            "abs should anchor to grandparent x=20 (not static parent +40), got {}",
            abs.content.x
        );
        assert!(
            (abs.content.y - 10.0).abs() < 1.0,
            "abs should anchor to grandparent y=10 (not static parent +50), got {}",
            abs.content.y
        );
    }

    #[test]
    fn transform_scale_bakes_into_geometry() {
        // A 100×100 box with `transform: scale(2)` doubles its painted
        // size and grows about its centre (50,50), so its top-left moves
        // to (-50,-50). Border widths/text would scale the same way.
        let kid = block(
            "div",
            Style {
                display: Some(Display::Block),
                width: Some(LengthSpec::Px(100.0)),
                height: Some(LengthSpec::Px(100.0)),
                scale_x: Some(2.0),
                scale_y: Some(2.0),
                ..Style::default()
            },
            vec![],
        );
        let root = block(
            "div",
            Style {
                display: Some(Display::Block),
                width: Some(LengthSpec::Px(400.0)),
                height: Some(LengthSpec::Px(400.0)),
                ..Style::default()
            },
            vec![kid],
        );
        let bx = layout(&root, &LayoutConfig::default());
        let k = &bx.children[0];
        assert!(
            (k.content.w - 200.0).abs() < 1.0,
            "scaled width should be 200, got {}",
            k.content.w
        );
        assert!(
            (k.content.h - 200.0).abs() < 1.0,
            "scaled height should be 200, got {}",
            k.content.h
        );
        assert!(
            (k.content.x + 50.0).abs() < 1.0,
            "scaled x should be -50 (grow about centre), got {}",
            k.content.x
        );
        assert!(
            (k.content.y + 50.0).abs() < 1.0,
            "scaled y should be -50 (grow about centre), got {}",
            k.content.y
        );
    }

    #[test]
    fn transform_rotate_uses_affine_path_not_geometry_bake() {
        // A 100×100 box with `transform: rotate(90deg)`. Rotation can't
        // be baked into axis-aligned geometry, so the painter takes the
        // affine layer path: the box keeps its laid-out size/position and
        // exposes `has_affine_transform()` + an R(90°)·S(1) matrix.
        let kid = block(
            "div",
            Style {
                display: Some(Display::Block),
                width: Some(LengthSpec::Px(100.0)),
                height: Some(LengthSpec::Px(100.0)),
                rotate_deg: Some(90.0),
                ..Style::default()
            },
            vec![],
        );
        let root = block(
            "div",
            Style {
                display: Some(Display::Block),
                width: Some(LengthSpec::Px(400.0)),
                height: Some(LengthSpec::Px(400.0)),
                ..Style::default()
            },
            vec![kid],
        );
        let bx = layout(&root, &LayoutConfig::default());
        let k = &bx.children[0];
        // Geometry untouched — rotation is a paint-time affine, not a
        // layout-time bake.
        assert!(
            (k.content.w - 100.0).abs() < 1.0 && (k.content.h - 100.0).abs() < 1.0,
            "rotated box keeps laid-out size, got {}x{}",
            k.content.w,
            k.content.h
        );
        assert!(k.has_affine_transform(), "rotate(90) needs the affine path");
        // R(90°): a=cos90·1=0, b=sin90·1=1, c=-sin90·1=-1, d=cos90·1=0.
        let m = k.transform_affine();
        assert!(m[0].abs() < 1e-3, "a≈0, got {}", m[0]);
        assert!((m[1] - 1.0).abs() < 1e-3, "b≈1, got {}", m[1]);
        assert!((m[2] + 1.0).abs() < 1e-3, "c≈-1, got {}", m[2]);
        assert!(m[3].abs() < 1e-3, "d≈0, got {}", m[3]);
    }

    #[test]
    fn transform_matrix_carried_raw_and_skips_geometry_scale() {
        // `transform: matrix(1,0,0,1,30,40)` — a pure translate expressed
        // as a matrix. It must be carried raw (no geometry bake) so the
        // painter applies the full affine.
        let kid = block(
            "div",
            Style {
                display: Some(Display::Block),
                width: Some(LengthSpec::Px(100.0)),
                height: Some(LengthSpec::Px(100.0)),
                matrix_2d: Some([1.0, 0.0, 0.0, 1.0, 30.0, 40.0]),
                ..Style::default()
            },
            vec![],
        );
        let root = block(
            "div",
            Style {
                display: Some(Display::Block),
                width: Some(LengthSpec::Px(400.0)),
                height: Some(LengthSpec::Px(400.0)),
                ..Style::default()
            },
            vec![kid],
        );
        let bx = layout(&root, &LayoutConfig::default());
        let k = &bx.children[0];
        assert!(
            (k.content.w - 100.0).abs() < 1.0 && k.content.x.abs() < 1.0,
            "matrix() must not bake into geometry: {}x at {}",
            k.content.w,
            k.content.x
        );
        assert!(k.has_affine_transform(), "matrix() needs the affine path");
        assert_eq!(k.transform_affine(), [1.0, 0.0, 0.0, 1.0, 30.0, 40.0]);
    }

    #[test]
    fn subtree_bounds_unions_overflowing_child() {
        let child = block(
            "div",
            Style {
                display: Some(Display::Block),
                position: Some(Position::Absolute),
                left_px: Some(LengthSpec::Px(80.0)),
                top_px: Some(LengthSpec::Px(80.0)),
                width: Some(LengthSpec::Px(100.0)),
                height: Some(LengthSpec::Px(100.0)),
                ..Style::default()
            },
            vec![],
        );
        let parent = block(
            "div",
            Style {
                display: Some(Display::Block),
                position: Some(Position::Relative),
                width: Some(LengthSpec::Px(100.0)),
                height: Some(LengthSpec::Px(100.0)),
                ..Style::default()
            },
            vec![child],
        );
        let bx = layout(&parent, &LayoutConfig::default());
        let b = bx.subtree_bounds();
        // Child sits at (80,80)+100 → right/bottom edge 180; parent is
        // only 100 wide, so the union must extend to ~180.
        assert!(
            b.x + b.w >= 179.0,
            "subtree bounds must include overflowing child (right={}, got w={})",
            b.x + b.w,
            b.w
        );
    }

    #[test]
    fn position_fixed_anchors_to_viewport_not_parent() {
        let fixed_kid = block(
            "div",
            Style {
                display: Some(Display::Block),
                position: Some(Position::Fixed),
                left_px: Some(LengthSpec::Px(20.0)),
                top_px: Some(LengthSpec::Px(10.0)),
                width: Some(LengthSpec::Px(100.0)),
                height: Some(LengthSpec::Px(40.0)),
                ..Style::default()
            },
            vec![text("fixed")],
        );
        let container = block(
            "div",
            Style {
                display: Some(Display::Block),
                position: Some(Position::Relative),
                margin: EdgeSizes {
                    top: 120.0,
                    right: 0.0,
                    bottom: 0.0,
                    left: 80.0,
                },
                width: Some(LengthSpec::Px(300.0)),
                height: Some(LengthSpec::Px(200.0)),
                ..Style::default()
            },
            vec![fixed_kid],
        );
        let root = layout(&container, &LayoutConfig::default());
        let fixed = &root.children[0];
        assert!(
            (fixed.content.x - 20.0).abs() < 1.0,
            "fixed child should anchor to viewport x=20, got {}",
            fixed.content.x
        );
        assert!(
            (fixed.content.y - 10.0).abs() < 1.0,
            "fixed child should anchor to viewport y=10, got {}",
            fixed.content.y
        );
    }

    #[test]
    fn position_fixed_auto_width_shrinks_to_fit_contents() {
        let fixed_kid = block(
            "aside",
            Style {
                display: Some(Display::Flex),
                flex_direction: Some(FlexDirection::Column),
                position: Some(Position::Fixed),
                top_px: Some(LengthSpec::Px(0.0)),
                left_px: Some(LengthSpec::Px(0.0)),
                ..Style::default()
            },
            vec![
                block(
                    "button",
                    Style {
                        display: Some(Display::Block),
                        ..Style::default()
                    },
                    vec![text("Home")],
                ),
                block(
                    "button",
                    Style {
                        display: Some(Display::Block),
                        ..Style::default()
                    },
                    vec![text("Transactions")],
                ),
            ],
        );
        let container = block(
            "div",
            Style {
                display: Some(Display::Block),
                position: Some(Position::Relative),
                ..Style::default()
            },
            vec![fixed_kid],
        );
        let root = layout(&container, &LayoutConfig::default());
        let root = &root.children[0];
        assert!(
            root.content.w < 300.0,
            "auto-width fixed box should shrink to fit instead of spanning viewport, got {}",
            root.content.w
        );
    }

    #[test]
    fn margin_auto_centres_explicit_width_block() {
        // 200px wide box inside a 1024px viewport with `margin: 0 auto`
        // should sit centred — content_x = (1024 - 200) / 2 = 412.
        let centred = block(
            "div",
            Style {
                display: Some(Display::Block),
                width: Some(LengthSpec::Px(200.0)),
                margin_auto: EdgeAuto {
                    top: false,
                    right: true,
                    bottom: false,
                    left: true,
                },
                ..Style::default()
            },
            vec![text("hi")],
        );
        let cfg = LayoutConfig::default();
        let root = layout(&centred, &cfg);
        assert!(
            (root.content.x - 412.0).abs() < 1.0,
            "centred box content_x should be 412, got {}",
            root.content.x
        );
        assert!(
            (root.content.w - 200.0).abs() < 1.0,
            "centred box width should be 200, got {}",
            root.content.w
        );
    }

    #[test]
    fn max_width_clamps_then_margin_auto_centres() {
        // Block with no explicit width but max-width: 300px and auto
        // horizontal margins. Should clamp to 300 and centre at
        // (1024 - 300) / 2 = 362.
        let box1 = block(
            "div",
            Style {
                display: Some(Display::Block),
                max_width: Some(LengthSpec::Px(300.0)),
                margin_auto: EdgeAuto {
                    top: false,
                    right: true,
                    bottom: false,
                    left: true,
                },
                ..Style::default()
            },
            vec![text("hi")],
        );
        let cfg = LayoutConfig::default();
        let root = layout(&box1, &cfg);
        assert!(
            (root.content.w - 300.0).abs() < 1.0,
            "max-width clamp should make width 300, got {}",
            root.content.w
        );
        assert!(
            (root.content.x - 362.0).abs() < 1.0,
            "after clamp the box should centre at 362, got {}",
            root.content.x
        );
    }

    #[test]
    fn percent_width_resolves_against_container() {
        // Outer 50% of viewport; inner 50% of outer. Outer should be 512,
        // inner should be 256 — not 512 (which is what a viewport-anchored
        // resolve would give).
        let inner = block(
            "div",
            Style {
                display: Some(Display::Block),
                width: Some(LengthSpec::Pct(50.0)),
                ..Style::default()
            },
            vec![text("x")],
        );
        let outer = block(
            "div",
            Style {
                display: Some(Display::Block),
                width: Some(LengthSpec::Pct(50.0)),
                ..Style::default()
            },
            vec![inner],
        );
        let cfg = LayoutConfig::default(); // viewport_w = 1024
        let root = layout(&outer, &cfg);
        assert!(
            (root.content.w - 512.0).abs() < 1.0,
            "outer should be 512 wide, got {}",
            root.content.w
        );
        let inner_box = &root.children[0];
        assert!(
            (inner_box.content.w - 256.0).abs() < 1.0,
            "inner should be 256 wide, got {}",
            inner_box.content.w
        );
    }

    #[test]
    fn display_none_drops_child_from_layout() {
        // A flex row with three kids — middle one is display:none.
        // Layout should only emit two boxes and the remaining kids
        // should occupy the positions that #1 and #3 would have
        // taken without the hidden middle.
        let kid = |w: f32, hidden: bool| {
            let mut s = Style {
                display: if hidden {
                    Some(Display::None)
                } else {
                    Some(Display::Block)
                },
                width: Some(LengthSpec::Px(w)),
                ..Style::default()
            };
            // Even with display:none we leave width so we can prove
            // the child takes NO space at all.
            if hidden {
                s.width = Some(LengthSpec::Px(w));
            }
            block("div", s, vec![text("x")])
        };
        let row = block(
            "div",
            Style {
                display: Some(Display::Flex),
                flex_direction: Some(FlexDirection::Row),
                gap_px: Some(10.0),
                ..Style::default()
            },
            vec![kid(100.0, false), kid(100.0, true), kid(100.0, false)],
        );
        let cfg = LayoutConfig::default();
        let root = layout(&row, &cfg);
        assert_eq!(root.children.len(), 2, "hidden child should be dropped");
        // Visible kids should start at 0 and 110 (no gap to absent #2).
        assert!((root.children[0].content.x - 0.0).abs() < 1.0);
        assert!(
            (root.children[1].content.x - 110.0).abs() < 1.0,
            "second visible should sit where the hidden would have, got {}",
            root.children[1].content.x
        );
    }

    #[test]
    fn flex_row_lays_children_horizontally() {
        let kid = |w: f32| {
            block(
                "div",
                Style {
                    display: Some(Display::Block),
                    width: Some(LengthSpec::Px(w)),
                    ..Style::default()
                },
                vec![text("x")],
            )
        };
        let row = block(
            "div",
            Style {
                display: Some(Display::Flex),
                flex_direction: Some(FlexDirection::Row),
                gap_px: Some(10.0),
                ..Style::default()
            },
            vec![kid(100.0), kid(100.0), kid(100.0)],
        );
        let cfg = LayoutConfig::default();
        let root = layout(&row, &cfg);
        assert_eq!(root.children.len(), 3);
        let xs: Vec<f32> = root.children.iter().map(|c| c.content.x).collect();
        // Three 100px-wide children with 10px gap should start at 0, 110, 220.
        assert!((xs[0] - 0.0).abs() < 1.0, "first x got {}", xs[0]);
        assert!((xs[1] - 110.0).abs() < 1.0, "second x got {}", xs[1]);
        assert!((xs[2] - 220.0).abs() < 1.0, "third x got {}", xs[2]);
        // All three should be on the same y row.
        let ys: Vec<f32> = root.children.iter().map(|c| c.content.y).collect();
        assert!((ys[0] - ys[1]).abs() < 1.0);
        assert!((ys[1] - ys[2]).abs() < 1.0);
    }

    #[test]
    fn flex_row_auto_width_children_shrink_to_contents() {
        let kid = |label: &str| {
            block(
                "div",
                Style {
                    display: Some(Display::Block),
                    ..Style::default()
                },
                vec![text(label)],
            )
        };
        let row = block(
            "div",
            Style {
                display: Some(Display::Flex),
                flex_direction: Some(FlexDirection::Row),
                gap_px: Some(10.0),
                ..Style::default()
            },
            vec![kid("Hi"), kid("There")],
        );
        let root = layout(&row, &LayoutConfig::default());
        assert!(
            root.children[0].content.w < 50.0,
            "first child should size to text, got {}",
            root.children[0].content.w
        );
        assert!(
            root.children[1].content.x < 100.0,
            "second child should sit near the first, got {}",
            root.children[1].content.x
        );
    }

    #[test]
    fn flex_row_shrink_reduces_overflowing_children() {
        let kid = || {
            block(
                "div",
                Style {
                    display: Some(Display::Block),
                    width: Some(LengthSpec::Px(200.0)),
                    ..Style::default()
                },
                vec![text("x")],
            )
        };
        let row = block(
            "div",
            Style {
                display: Some(Display::Flex),
                flex_direction: Some(FlexDirection::Row),
                ..Style::default()
            },
            vec![kid(), kid()],
        );
        let cfg = LayoutConfig {
            viewport_w: 300.0,
            ..LayoutConfig::default()
        };
        let root = layout(&row, &cfg);
        assert!(
            root.children[0].content.w < 200.0,
            "first child should shrink, got {}",
            root.children[0].content.w
        );
        assert!(
            root.children[1].content.w < 200.0,
            "second child should shrink, got {}",
            root.children[1].content.w
        );
        assert!(
            root.children[1].content.x < 170.0,
            "second child should stay within the row, got {}",
            root.children[1].content.x
        );
    }

    #[test]
    fn flex_row_grow_expands_children_to_fill_available_space() {
        let kid = || {
            block(
                "div",
                Style {
                    display: Some(Display::Block),
                    width: Some(LengthSpec::Px(50.0)),
                    flex_grow: Some(1.0),
                    ..Style::default()
                },
                vec![text("x")],
            )
        };
        let row = block(
            "div",
            Style {
                display: Some(Display::Flex),
                flex_direction: Some(FlexDirection::Row),
                width: Some(LengthSpec::Px(300.0)),
                ..Style::default()
            },
            vec![kid(), kid()],
        );
        let root = layout(&row, &LayoutConfig::default());
        assert!(
            root.children[0].content.w > 140.0,
            "first child should grow, got {}",
            root.children[0].content.w
        );
        assert!(
            root.children[1].content.w > 140.0,
            "second child should grow, got {}",
            root.children[1].content.w
        );
        assert!(
            root.children[1].content.x > 140.0,
            "second child should be laid out after the grown first child, got {}",
            root.children[1].content.x
        );
    }

    #[test]
    fn flex_item_percent_width_fills_container() {
        // A `width:100%` flex item must fill the flex container — its flex base
        // size resolves the percentage against the container, NOT collapse to
        // content width. Regression: real grid/flex cards (mail.hyvechain.com's
        // login card) were crushed to ~content width, clipping "SIGN IN"→"SI IN".
        let kid = block(
            "div",
            Style {
                display: Some(Display::Block),
                width: Some(LengthSpec::Pct(100.0)),
                ..Style::default()
            },
            vec![text("x")],
        );
        let row = block(
            "div",
            Style {
                display: Some(Display::Flex),
                justify_content: Some(JustifyContent::Center),
                ..Style::default()
            },
            vec![kid],
        );
        let cfg = LayoutConfig::default(); // viewport 1024
        let root = layout(&row, &cfg);
        assert!(
            root.children[0].content.w > 1000.0,
            "width:100% flex item should fill ~1024, got {}",
            root.children[0].content.w
        );
    }

    #[test]
    fn grid_max_content_sums_columns_not_max() {
        // A grid's max-content (its flex base size when shrink-to-fit) is the SUM
        // of its column tracks, NOT the max of its items. Regression: a 2-column
        // grid nested in a centered flex container collapsed to one column's
        // width, crushing the auth card and clipping the tab labels
        // ("SIGN IN"→"SI IN") once mail.hyvechain.com's JS flipped the container
        // to flex.
        let cell_a = block(
            "div",
            Style {
                display: Some(Display::Block),
                ..Style::default()
            },
            vec![text("A")],
        );
        let cell_b = block(
            "div",
            Style {
                display: Some(Display::Block),
                ..Style::default()
            },
            vec![text("B")],
        );
        let grid = block(
            "div",
            Style {
                display: Some(Display::Grid),
                grid_template_columns: Some(vec![GridTrack::Px(200.0), GridTrack::Px(200.0)]),
                ..Style::default()
            },
            vec![cell_a, cell_b],
        );
        let row = block(
            "div",
            Style {
                display: Some(Display::Flex),
                justify_content: Some(JustifyContent::Center),
                ..Style::default()
            },
            vec![grid],
        );
        let cfg = LayoutConfig::default();
        let root = layout(&row, &cfg);
        let grid_w = root.children[0].content.w;
        assert!(
            grid_w >= 390.0 && grid_w <= 410.0,
            "grid flex item should be ~400 (sum of two 200px columns), got {grid_w}"
        );
    }

    #[test]
    fn flex_row_justify_center() {
        let kid = block(
            "div",
            Style {
                display: Some(Display::Block),
                width: Some(LengthSpec::Px(100.0)),
                ..Style::default()
            },
            vec![text("x")],
        );
        let row = block(
            "div",
            Style {
                display: Some(Display::Flex),
                justify_content: Some(JustifyContent::Center),
                ..Style::default()
            },
            vec![kid],
        );
        let cfg = LayoutConfig::default(); // viewport 1024
        let root = layout(&row, &cfg);
        let first = &root.children[0];
        // 100-wide child in 1024-wide container, centered → x ≈ 462.
        assert!(
            (first.content.x - 462.0).abs() < 2.0,
            "expected centered ≈462, got {}",
            first.content.x
        );
    }

    #[test]
    fn grid_three_fr_columns_split_viewport() {
        let kid = || {
            block(
                "div",
                Style {
                    display: Some(Display::Block),
                    ..Style::default()
                },
                vec![text("x")],
            )
        };
        let grid = block(
            "div",
            Style {
                display: Some(Display::Grid),
                grid_template_columns: Some(vec![
                    GridTrack::Fr(1.0),
                    GridTrack::Fr(1.0),
                    GridTrack::Fr(1.0),
                ]),
                ..Style::default()
            },
            vec![kid(), kid(), kid()],
        );
        let cfg = LayoutConfig::default(); // 1024 wide
        let root = layout(&grid, &cfg);
        assert_eq!(root.children.len(), 3);
        let xs: Vec<f32> = root.children.iter().map(|c| c.content.x).collect();
        // Each column ≈ 341 wide, so starts at 0, 341, 682.
        assert!((xs[0] - 0.0).abs() < 1.5, "col0 x = {}", xs[0]);
        assert!((xs[1] - 341.33).abs() < 1.5, "col1 x = {}", xs[1]);
        assert!((xs[2] - 682.66).abs() < 1.5, "col2 x = {}", xs[2]);
    }

    #[test]
    fn grid_fixed_and_fr_mix() {
        let kid = || {
            block(
                "div",
                Style {
                    display: Some(Display::Block),
                    ..Style::default()
                },
                vec![text("x")],
            )
        };
        let grid = block(
            "div",
            Style {
                display: Some(Display::Grid),
                grid_template_columns: Some(vec![
                    GridTrack::Px(100.0),
                    GridTrack::Fr(1.0),
                    GridTrack::Px(200.0),
                ]),
                ..Style::default()
            },
            vec![kid(), kid(), kid()],
        );
        let cfg = LayoutConfig::default(); // 1024 wide
        let root = layout(&grid, &cfg);
        // Middle column = 1024 - 100 - 200 = 724.
        let xs: Vec<f32> = root.children.iter().map(|c| c.content.x).collect();
        assert!((xs[0] - 0.0).abs() < 1.0, "col0 x = {}", xs[0]);
        assert!((xs[1] - 100.0).abs() < 1.0, "col1 x = {}", xs[1]);
        assert!((xs[2] - 824.0).abs() < 1.0, "col2 x = {}", xs[2]);
    }

    #[test]
    fn grid_gap_pushes_columns() {
        let kid = || {
            block(
                "div",
                Style {
                    display: Some(Display::Block),
                    ..Style::default()
                },
                vec![text("x")],
            )
        };
        let grid = block(
            "div",
            Style {
                display: Some(Display::Grid),
                grid_template_columns: Some(vec![
                    GridTrack::Px(100.0),
                    GridTrack::Px(100.0),
                    GridTrack::Px(100.0),
                ]),
                column_gap_px: Some(20.0),
                ..Style::default()
            },
            vec![kid(), kid(), kid()],
        );
        let cfg = LayoutConfig::default();
        let root = layout(&grid, &cfg);
        let xs: Vec<f32> = root.children.iter().map(|c| c.content.x).collect();
        assert!((xs[0] - 0.0).abs() < 1.0);
        assert!((xs[1] - 120.0).abs() < 1.0, "col1 x = {}", xs[1]);
        assert!((xs[2] - 240.0).abs() < 1.0, "col2 x = {}", xs[2]);
    }

    #[test]
    fn grid_autoflow_wraps_to_next_row() {
        let kid = || {
            block(
                "div",
                Style {
                    display: Some(Display::Block),
                    height: Some(LengthSpec::Px(50.0)),
                    ..Style::default()
                },
                vec![text("x")],
            )
        };
        let grid = block(
            "div",
            Style {
                display: Some(Display::Grid),
                grid_template_columns: Some(vec![GridTrack::Fr(1.0), GridTrack::Fr(1.0)]),
                row_gap_px: Some(10.0),
                ..Style::default()
            },
            vec![kid(), kid(), kid(), kid()],
        );
        let cfg = LayoutConfig::default();
        let root = layout(&grid, &cfg);
        let ys: Vec<f32> = root.children.iter().map(|c| c.content.y).collect();
        // Two children per row → rows at y=0 and y=row_h+gap.
        assert!((ys[0] - ys[1]).abs() < 1.0, "row0 same y");
        assert!((ys[2] - ys[3]).abs() < 1.0, "row1 same y");
        assert!(ys[2] > ys[0], "row1 below row0");
    }

    #[test]
    fn relative_position_shifts_box() {
        // Container with one normally-positioned child at y=0, then a
        // relatively-positioned child offset by top:50 left:30. The
        // *content* of the offset child should sit 50px below where it
        // would have landed without the offset.
        let normal = block(
            "div",
            Style {
                display: Some(Display::Block),
                height: Some(LengthSpec::Px(20.0)),
                ..Style::default()
            },
            vec![text("a")],
        );
        let shifted = block(
            "div",
            Style {
                display: Some(Display::Block),
                position: Some(Position::Relative),
                top_px: Some(LengthSpec::Px(50.0)),
                left_px: Some(LengthSpec::Px(30.0)),
                height: Some(LengthSpec::Px(20.0)),
                ..Style::default()
            },
            vec![text("b")],
        );
        let body = block(
            "body",
            Style {
                display: Some(Display::Block),
                ..Style::default()
            },
            vec![normal, shifted],
        );
        let cfg = LayoutConfig::default();
        let root = layout(&body, &cfg);
        let s = &root.children[1];
        // Without offset, s would start right below `normal` (which
        // takes the text height of ~22). With top:50 it lands ~22+50.
        assert!(s.content.x >= 30.0 - 0.5, "x shift, got {}", s.content.x);
        assert!(s.content.y >= 50.0, "y shift, got {}", s.content.y);
    }

    #[test]
    fn text_height_uses_font_size() {
        let doc = block(
            "body",
            Style {
                display: Some(Display::Block),
                font_size_px: Some(20.0),
                ..Style::default()
            },
            vec![text("hi")],
        );
        let cfg = LayoutConfig::default();
        let root = layout(&doc, &cfg);
        let t = &root.children[0];
        assert!((t.content.h - 28.0).abs() < 1.0);
    }

    #[test]
    fn preserved_whitespace_text_counts_explicit_newlines() {
        // white-space:normal collapses `\n` to a space, so the explicit
        // newline only forces a second line when the TEXT NODE itself
        // preserves whitespace (the real pipeline inherits this onto text
        // nodes inside <pre>). Build the text node with the flag set.
        let pre_text = StyledNode {
            kind: StyledKind::Text("a  b\nc".to_string()),
            style: Style {
                preserve_whitespace: true,
                ..Style::default()
            },
            children: Vec::new(),
        };
        let doc = block(
            "pre",
            Style {
                display: Some(Display::Block),
                preserve_whitespace: true,
                ..Style::default()
            },
            vec![pre_text],
        );
        let cfg = LayoutConfig::default();
        let root = layout(&doc, &cfg);
        let t = &root.children[0];
        let line_h = t.font_size_px * cfg.default_line_height;
        assert!(t.content.h >= line_h * 2.0);
    }

    #[test]
    fn normal_whitespace_collapses_source_newlines_to_one_line() {
        // A bare text node with source newlines but white-space:normal must
        // collapse to a single line — NOT split per newline. This is the
        // Wikipedia header/card regression: a "\n"-only run was emitting
        // tall empty line boxes.
        let normal_text = StyledNode {
            kind: StyledKind::Text("\n  a  b\n  c  \n".to_string()),
            style: Style::default(),
            children: Vec::new(),
        };
        let doc = block(
            "div",
            Style {
                display: Some(Display::Block),
                ..Style::default()
            },
            vec![normal_text],
        );
        let cfg = LayoutConfig::default();
        let root = layout(&doc, &cfg);
        let t = &root.children[0];
        let line_h = t.font_size_px * cfg.default_line_height;
        // One line (everything fits well within the default width).
        assert!(t.content.h <= line_h * 1.5);
    }

    #[test]
    fn nbsp_not_treated_as_collapsible_whitespace() {
        // U+00A0 (NBSP / `&nbsp;`) must NOT be collapsed or stripped by the
        // CSS white-space:normal normaliser — Chrome-divergence audit §nbsp.
        // css_normalize_whitespace should preserve it verbatim.
        let nbsp = '\u{00A0}';
        // 1. A string of only NBSP chars should pass through unchanged.
        let only_nbsp = format!("{nbsp}{nbsp}{nbsp}");
        assert_eq!(css_normalize_whitespace(&only_nbsp), only_nbsp);
        // 2. NBSP surrounded by collapsible whitespace: the whitespace collapses
        //    but the NBSP itself is preserved.
        let mixed = format!("foo  {nbsp}  bar");
        assert_eq!(css_normalize_whitespace(&mixed), format!("foo {nbsp} bar"));
        // 3. css_split_whitespace should NOT split on NBSP.
        let with_nbsp = format!("foo{nbsp}bar");
        let parts: Vec<&str> = css_split_whitespace(&with_nbsp).collect();
        assert_eq!(parts, vec![with_nbsp.as_str()], "NBSP must not split words");
    }

    #[test]
    fn flex_row_reverse_places_first_child_on_right() {
        let doc = block(
            "div",
            Style {
                display: Some(Display::Flex),
                flex_direction: Some(FlexDirection::RowReverse),
                ..Style::default()
            },
            vec![
                block(
                    "a",
                    Style {
                        display: Some(Display::Block),
                        width: Some(LengthSpec::Px(100.0)),
                        height: Some(LengthSpec::Px(20.0)),
                        ..Style::default()
                    },
                    vec![text("a")],
                ),
                block(
                    "b",
                    Style {
                        display: Some(Display::Block),
                        width: Some(LengthSpec::Px(100.0)),
                        height: Some(LengthSpec::Px(20.0)),
                        ..Style::default()
                    },
                    vec![text("b")],
                ),
            ],
        );
        let mut cfg = LayoutConfig::default();
        cfg.viewport_w = 400.0;
        let root = layout(&doc, &cfg);
        assert_eq!(root.children.len(), 2);
        assert!(root.children[0].content.x > root.children[1].content.x);
    }

    #[test]
    fn flex_column_reverse_places_first_child_at_bottom() {
        let doc = block(
            "div",
            Style {
                display: Some(Display::Flex),
                flex_direction: Some(FlexDirection::ColumnReverse),
                ..Style::default()
            },
            vec![
                block(
                    "a",
                    Style {
                        display: Some(Display::Block),
                        width: Some(LengthSpec::Px(100.0)),
                        height: Some(LengthSpec::Px(20.0)),
                        ..Style::default()
                    },
                    vec![text("a")],
                ),
                block(
                    "b",
                    Style {
                        display: Some(Display::Block),
                        width: Some(LengthSpec::Px(100.0)),
                        height: Some(LengthSpec::Px(20.0)),
                        ..Style::default()
                    },
                    vec![text("b")],
                ),
            ],
        );
        let root = layout(&doc, &LayoutConfig::default());
        assert_eq!(root.children.len(), 2);
        assert!(root.children[0].content.y > root.children[1].content.y);
    }

    /// Bug 4: `flex-direction: column-reverse; justify-content: flex-start`
    /// should pin items to the BOTTOM (main-start for column-reverse).
    /// Previously items were packed at the top (wrong edge).
    #[test]
    fn flex_column_reverse_flex_start_packs_at_bottom() {
        // Container: 200px tall, column-reverse, flex-start justify.
        // Two children each 30px tall — they should both be at the
        // bottom of the 200px container, not the top.
        let doc = block(
            "div",
            Style {
                display: Some(Display::Flex),
                flex_direction: Some(FlexDirection::ColumnReverse),
                justify_content: Some(JustifyContent::Start),
                height: Some(LengthSpec::Px(200.0)),
                ..Style::default()
            },
            vec![
                block(
                    "a",
                    Style {
                        display: Some(Display::Block),
                        height: Some(LengthSpec::Px(30.0)),
                        ..Style::default()
                    },
                    vec![text("a")],
                ),
                block(
                    "b",
                    Style {
                        display: Some(Display::Block),
                        height: Some(LengthSpec::Px(30.0)),
                        ..Style::default()
                    },
                    vec![text("b")],
                ),
            ],
        );
        let root = layout(&doc, &LayoutConfig::default());
        // In column-reverse, first DOM child is main-start = last visually
        // (near the bottom). Both children (60px total) should pack at the
        // bottom of 200px, so the lower one (child[0]) sits near y=170.
        let y0 = root.children[0].content.y;
        let y1 = root.children[1].content.y;
        assert!(
            y0 >= 140.0,
            "column-reverse flex-start: first DOM child should be near bottom (≥140), got y={}",
            y0
        );
        assert!(
            y1 >= 140.0,
            "column-reverse flex-start: second DOM child should be near bottom (≥140), got y={}",
            y1
        );
        // First child (main-start = bottom-most visual position) is below second.
        assert!(
            y0 > y1,
            "in column-reverse DOM child[0] renders below DOM child[1], got y0={} y1={}",
            y0,
            y1
        );
    }

    /// Bug 3: `align-self` on a flex item must override the container's
    /// `align-items`. With `align-items: flex-start` on the container and
    /// `align-self: flex-end` on one child, that child should stick to the
    /// cross-axis end (bottom of the row), while siblings default to start.
    #[test]
    fn flex_row_align_self_overrides_align_items() {
        // Container: 100px tall row with align-items: Start.
        // Two children both 30px tall, but child[1] has align-self: End.
        let doc = block(
            "div",
            Style {
                display: Some(Display::Flex),
                flex_direction: Some(FlexDirection::Row),
                align_items: Some(AlignItems::Start),
                height: Some(LengthSpec::Px(100.0)),
                ..Style::default()
            },
            vec![
                // Default: inherits container's align-items: Start → top.
                block(
                    "a",
                    Style {
                        display: Some(Display::Block),
                        height: Some(LengthSpec::Px(30.0)),
                        ..Style::default()
                    },
                    vec![text("a")],
                ),
                // align-self: End overrides → bottom of the 100px row.
                block(
                    "b",
                    Style {
                        display: Some(Display::Block),
                        height: Some(LengthSpec::Px(30.0)),
                        align_self: Some(AlignItems::End),
                        ..Style::default()
                    },
                    vec![text("b")],
                ),
            ],
        );
        let root = layout(&doc, &LayoutConfig::default());
        let y_start = root.children[0].content.y;
        let y_end = root.children[1].content.y;
        // Child[0] with align-items:Start should be near y=0 (top).
        assert!(
            y_start < 5.0,
            "align-items:Start child should be near top (y≈0), got y={}",
            y_start
        );
        // Child[1] with align-self:End should be near y=70 (100 - 30).
        assert!(
            y_end > 60.0,
            "align-self:End child should be near bottom (y>60), got y={}",
            y_end
        );
    }

    /// Bug 5: `position: absolute; right: R` (no left) should place the
    /// RIGHT EDGE of the box at `cb.w - R`, not the LEFT edge.
    /// Before the fix, dx = cb.w - R placed the LEFT edge there, pushing
    /// the box R pixels too far right by its own width.
    #[test]
    fn position_absolute_right_only_places_right_edge() {
        // Container: 400px wide, relative.  Child: 100px wide, abs, right:20.
        // Expected: right edge at 400-20=380, so left edge (content.x) ≈ 280.
        let abs_child = block(
            "div",
            Style {
                display: Some(Display::Block),
                position: Some(Position::Absolute),
                right_px: Some(LengthSpec::Px(20.0)),
                width: Some(LengthSpec::Px(100.0)),
                height: Some(LengthSpec::Px(20.0)),
                ..Style::default()
            },
            vec![text("abs")],
        );
        let container = block(
            "div",
            Style {
                display: Some(Display::Block),
                position: Some(Position::Relative),
                width: Some(LengthSpec::Px(400.0)),
                ..Style::default()
            },
            vec![abs_child],
        );
        let root = layout(&container, &LayoutConfig::default());
        let child = &root.children[0];
        // content.x = left edge ≈ 280 (= 400 - 20 - 100).
        assert!(
            (child.content.x - 280.0).abs() < 1.5,
            "abs right:20 width:100 in 400px cb → left edge should be ≈280, got {}",
            child.content.x
        );
    }

    /// Bug 5 (bottom variant): `position: absolute; bottom: B` (no top) should
    /// place the BOTTOM EDGE at `cb.h - B`, not the TOP edge.
    #[test]
    fn position_absolute_bottom_only_places_bottom_edge() {
        // Container: 300px tall, relative.  Child: 50px tall, abs, bottom:10.
        // Expected: bottom edge at 300-10=290, so top edge (content.y) ≈ 240.
        let abs_child = block(
            "div",
            Style {
                display: Some(Display::Block),
                position: Some(Position::Absolute),
                bottom_px: Some(LengthSpec::Px(10.0)),
                height: Some(LengthSpec::Px(50.0)),
                width: Some(LengthSpec::Px(50.0)),
                ..Style::default()
            },
            vec![text("abs")],
        );
        let container = block(
            "div",
            Style {
                display: Some(Display::Block),
                position: Some(Position::Relative),
                height: Some(LengthSpec::Px(300.0)),
                ..Style::default()
            },
            vec![abs_child],
        );
        let root = layout(&container, &LayoutConfig::default());
        let child = &root.children[0];
        // content.y = top edge ≈ 240 (= 300 - 10 - 50).
        assert!(
            (child.content.y - 240.0).abs() < 1.5,
            "abs bottom:10 height:50 in 300px cb → top edge should be ≈240, got {}",
            child.content.y
        );
    }

    #[test]
    fn flex_row_wrap_moves_overflowing_items_to_next_line() {
        let kid = |w| {
            block(
                "div",
                Style {
                    width: Some(LengthSpec::Px(w)),
                    height: Some(LengthSpec::Px(20.0)),
                    ..Style::default()
                },
                vec![text("k")],
            )
        };
        let doc = block(
            "div",
            Style {
                display: Some(Display::Flex),
                flex_wrap: Some(FlexWrap::Wrap),
                width: Some(LengthSpec::Px(220.0)),
                column_gap_px: Some(10.0),
                row_gap_px: Some(8.0),
                ..Style::default()
            },
            vec![kid(100.0), kid(100.0), kid(100.0)],
        );
        let root = layout(&doc, &LayoutConfig::default());
        assert!((root.children[0].content.y - root.children[1].content.y).abs() < 1.0);
        assert!(root.children[2].content.y > root.children[0].content.y + 20.0);
        assert!((root.children[2].content.x - root.children[0].content.x).abs() < 1.0);
    }

    #[test]
    fn flex_column_justify_end_pushes_children_down() {
        let doc = block(
            "div",
            Style {
                display: Some(Display::Flex),
                flex_direction: Some(FlexDirection::Column),
                justify_content: Some(JustifyContent::End),
                height: Some(LengthSpec::Px(200.0)),
                ..Style::default()
            },
            vec![
                block(
                    "a",
                    Style {
                        display: Some(Display::Block),
                        height: Some(LengthSpec::Px(20.0)),
                        ..Style::default()
                    },
                    vec![text("a")],
                ),
                block(
                    "b",
                    Style {
                        display: Some(Display::Block),
                        height: Some(LengthSpec::Px(20.0)),
                        ..Style::default()
                    },
                    vec![text("b")],
                ),
            ],
        );
        let root = layout(&doc, &LayoutConfig::default());
        assert!(root.children[0].content.y > 100.0);
    }

    #[test]
    fn flex_column_grow_expands_children_to_fill_available_height() {
        let kid = || {
            block(
                "div",
                Style {
                    display: Some(Display::Block),
                    height: Some(LengthSpec::Px(40.0)),
                    flex_grow: Some(1.0),
                    ..Style::default()
                },
                vec![text("x")],
            )
        };
        let doc = block(
            "div",
            Style {
                display: Some(Display::Flex),
                flex_direction: Some(FlexDirection::Column),
                height: Some(LengthSpec::Px(300.0)),
                ..Style::default()
            },
            vec![kid(), kid()],
        );
        let root = layout(&doc, &LayoutConfig::default());
        assert!(
            root.children[0].content.h > 140.0,
            "first child should grow, got {}",
            root.children[0].content.h
        );
        assert!(
            root.children[1].content.h > 140.0,
            "second child should grow, got {}",
            root.children[1].content.h
        );
        assert!(
            root.children[1].content.y > 140.0,
            "second child should be laid out below the grown first child, got {}",
            root.children[1].content.y
        );
    }

    #[test]
    fn flex_column_min_height_does_not_cap_content_height() {
        let kid = || {
            block(
                "div",
                Style {
                    display: Some(Display::Block),
                    height: Some(LengthSpec::Px(80.0)),
                    ..Style::default()
                },
                vec![text("x")],
            )
        };
        let doc = block(
            "div",
            Style {
                display: Some(Display::Flex),
                flex_direction: Some(FlexDirection::Column),
                min_height: Some(LengthSpec::Px(100.0)),
                ..Style::default()
            },
            vec![kid(), kid()],
        );
        let root = layout(&doc, &LayoutConfig::default());
        assert!(
            root.content.h >= 160.0,
            "container should grow past min-height to fit content, got {}",
            root.content.h
        );
        assert!(
            root.children[1].content.y >= 80.0,
            "second child should remain below the first instead of being squashed upward, got {}",
            root.children[1].content.y
        );
    }

    #[test]
    fn flex_column_with_min_height_does_not_shrink_flex_child_to_viewport() {
        let main_inner = block(
            "div",
            Style {
                display: Some(Display::Block),
                height: Some(LengthSpec::Px(180.0)),
                ..Style::default()
            },
            vec![text("content")],
        );
        let doc = block(
            "div",
            Style {
                display: Some(Display::Flex),
                flex_direction: Some(FlexDirection::Column),
                min_height: Some(LengthSpec::Px(100.0)),
                ..Style::default()
            },
            vec![
                block(
                    "header",
                    Style {
                        display: Some(Display::Block),
                        height: Some(LengthSpec::Px(20.0)),
                        ..Style::default()
                    },
                    vec![text("head")],
                ),
                block(
                    "main",
                    Style {
                        display: Some(Display::Block),
                        flex_grow: Some(1.0),
                        ..Style::default()
                    },
                    vec![main_inner],
                ),
                block(
                    "footer",
                    Style {
                        display: Some(Display::Block),
                        height: Some(LengthSpec::Px(20.0)),
                        ..Style::default()
                    },
                    vec![text("foot")],
                ),
            ],
        );
        let root = layout(&doc, &LayoutConfig::default());
        assert!(
            root.content.h >= 220.0,
            "container should grow to fit header + main content + footer, got {}",
            root.content.h
        );
        assert!(
            root.children[2].content.y >= 200.0,
            "footer should sit below the tall main content instead of being pulled upward, got {}",
            root.children[2].content.y
        );
    }

    #[test]
    fn indefinite_flex_column_ignores_zero_flex_basis_for_intrinsic_height() {
        let doc = block(
            "div",
            Style {
                display: Some(Display::Flex),
                flex_direction: Some(FlexDirection::Column),
                min_height: Some(LengthSpec::Px(100.0)),
                ..Style::default()
            },
            vec![
                block(
                    "header",
                    Style {
                        display: Some(Display::Block),
                        height: Some(LengthSpec::Px(20.0)),
                        ..Style::default()
                    },
                    vec![text("head")],
                ),
                block(
                    "main",
                    Style {
                        display: Some(Display::Block),
                        flex_grow: Some(1.0),
                        flex_shrink: Some(1.0),
                        flex_basis: Some(LengthSpec::Px(0.0)),
                        ..Style::default()
                    },
                    vec![block(
                        "inner",
                        Style {
                            display: Some(Display::Block),
                            height: Some(LengthSpec::Px(180.0)),
                            ..Style::default()
                        },
                        vec![text("content")],
                    )],
                ),
                block(
                    "footer",
                    Style {
                        display: Some(Display::Block),
                        height: Some(LengthSpec::Px(20.0)),
                        ..Style::default()
                    },
                    vec![text("foot")],
                ),
            ],
        );
        let root = layout(&doc, &LayoutConfig::default());
        assert!(
            root.children[1].content.h >= 180.0,
            "flex item should keep its intrinsic content height in an indefinite column, got {}",
            root.children[1].content.h
        );
        assert!(
            root.children[2].content.y >= 200.0,
            "footer should remain below the main content, got {}",
            root.children[2].content.y
        );
    }

    /// Bug 1: `align-self` on a flex item must override the container's
    /// `align-items`. With `align-items: flex-start` on the container and
    /// `align-self: flex-end` on a child, that child is placed at the
    /// cross-axis end (bottom of the row), not the start.
    #[test]
    fn flex_item_align_self_overrides_container_align_items() {
        // Container: 100px tall flex row, align-items: Start.
        // child[0]: no align-self → inherits Start → top (y ≈ 0).
        // child[1]: align-self: End → overrides → bottom (y ≈ 70).
        let doc = block(
            "div",
            Style {
                display: Some(Display::Flex),
                flex_direction: Some(FlexDirection::Row),
                align_items: Some(AlignItems::Start),
                height: Some(LengthSpec::Px(100.0)),
                ..Style::default()
            },
            vec![
                block(
                    "a",
                    Style {
                        display: Some(Display::Block),
                        height: Some(LengthSpec::Px(30.0)),
                        ..Style::default()
                    },
                    vec![text("a")],
                ),
                block(
                    "b",
                    Style {
                        display: Some(Display::Block),
                        height: Some(LengthSpec::Px(30.0)),
                        align_self: Some(AlignItems::End),
                        ..Style::default()
                    },
                    vec![text("b")],
                ),
            ],
        );
        let root = layout(&doc, &LayoutConfig::default());
        let y_start = root.children[0].content.y;
        let y_end = root.children[1].content.y;
        assert!(
            y_start < 5.0,
            "align-items:Start child should be near top (y<5), got y={}",
            y_start
        );
        assert!(
            y_end > 60.0,
            "align-self:End child should be near bottom (y>60) in 100px row, got y={}",
            y_end
        );
    }

    /// Bug 2: `position: absolute; right: R; width: W` in a `cb_width`-wide
    /// containing block should place the box's LEFT edge at
    /// `cb_width - R - W`, not at `cb_width - R`.
    #[test]
    fn absolute_right_only_places_right_edge_correctly() {
        // Containing block: 200px wide, position: relative.
        // Child: position: absolute; right: 10px; width: 50px.
        // Expected: left edge = 200 - 10 - 50 = 140px.
        let abs_child = block(
            "div",
            Style {
                display: Some(Display::Block),
                position: Some(Position::Absolute),
                right_px: Some(LengthSpec::Px(10.0)),
                width: Some(LengthSpec::Px(50.0)),
                height: Some(LengthSpec::Px(20.0)),
                ..Style::default()
            },
            vec![text("x")],
        );
        let container = block(
            "div",
            Style {
                display: Some(Display::Block),
                position: Some(Position::Relative),
                width: Some(LengthSpec::Px(200.0)),
                ..Style::default()
            },
            vec![abs_child],
        );
        let root = layout(&container, &LayoutConfig::default());
        let child = &root.children[0];
        assert!(
            (child.content.x - 140.0).abs() < 1.5,
            "right:10 width:50 in 200px cb → left edge should be 140, got {}",
            child.content.x
        );
    }

    /// Bug 3: `flex-direction: column-reverse` with two 50px-tall items in a
    /// 300px container should pack items at the bottom (main-start = bottom),
    /// so the first item (DOM index 0) appears at y ≈ 200, not y = 0.
    #[test]
    fn flex_column_reverse_packs_items_at_bottom() {
        let doc = block(
            "div",
            Style {
                display: Some(Display::Flex),
                flex_direction: Some(FlexDirection::ColumnReverse),
                justify_content: Some(JustifyContent::Start),
                height: Some(LengthSpec::Px(300.0)),
                ..Style::default()
            },
            vec![
                block(
                    "a",
                    Style {
                        display: Some(Display::Block),
                        height: Some(LengthSpec::Px(50.0)),
                        ..Style::default()
                    },
                    vec![text("a")],
                ),
                block(
                    "b",
                    Style {
                        display: Some(Display::Block),
                        height: Some(LengthSpec::Px(50.0)),
                        ..Style::default()
                    },
                    vec![text("b")],
                ),
            ],
        );
        let root = layout(&doc, &LayoutConfig::default());
        // In column-reverse the first DOM child is main-start = bottom-most.
        // Two 50px items in 300px → first item sits at y ≈ 250,
        // second item (main-end) sits at y ≈ 200. Both well above 140.
        let y0 = root.children[0].content.y;
        let y1 = root.children[1].content.y;
        assert!(
            y0 >= 200.0,
            "column-reverse first item (main-start) should be near bottom (y≥200 in 300px), got y={}",
            y0
        );
        assert!(
            y1 >= 140.0,
            "column-reverse second item should also be near bottom (y≥140), got y={}",
            y1
        );
        assert!(
            y0 > y1,
            "in column-reverse, DOM child[0] renders below DOM child[1], got y0={} y1={}",
            y0,
            y1
        );
    }

    // ── Element-level scrolling (overflow:auto/scroll) ──────────────────────

    /// A fixed-height `overflow:auto` div with a much taller child is a
    /// scroll container; scrollHeight reflects the full content, clientHeight
    /// the padding box, and the two yield a non-zero max scrollTop.
    fn scroller_with_tall_child() -> LayoutBox {
        // 200px-tall viewport with 20px padding all round, holding a
        // 1000px-tall child. box-sizing content-box.
        let child = block(
            "div",
            Style {
                display: Some(Display::Block),
                width: Some(LengthSpec::Px(100.0)),
                height: Some(LengthSpec::Px(1000.0)),
                ..Style::default()
            },
            vec![],
        );
        let scroller = block(
            "div",
            Style {
                display: Some(Display::Block),
                width: Some(LengthSpec::Px(160.0)),
                height: Some(LengthSpec::Px(160.0)),
                padding: EdgeSizes {
                    top: 20.0,
                    right: 20.0,
                    bottom: 20.0,
                    left: 20.0,
                },
                overflow_y: Overflow::Auto,
                overflow_x: Overflow::Auto,
                ..Style::default()
            },
            vec![child],
        );
        let root = layout(&scroller, &LayoutConfig::default());
        // The root wraps the page; descend to our scroller box.
        find_scroller(&root).expect("scroller present").clone()
    }

    fn find_scroller(b: &LayoutBox) -> Option<&LayoutBox> {
        if b.is_scroll_container() {
            return Some(b);
        }
        for c in &b.children {
            if let Some(s) = find_scroller(c) {
                return Some(s);
            }
        }
        None
    }

    #[test]
    fn scroll_container_geometry_matches_chrome_box_model() {
        let s = scroller_with_tall_child();
        assert!(s.is_scroll_container(), "overflow:auto box scrolls");
        // content-box sizing: height:160px sets the CONTENT height to 160.
        // clientHeight = padding box = content (160) + top+bottom padding
        // (40) = 200.
        assert!(
            (s.client_height() - 200.0).abs() < 0.5,
            "clientHeight should be the padding box (200), got {}",
            s.client_height()
        );
        // scrollHeight = padding-top + full content + padding-bottom. The
        // 1000px child plus 20px top + 20px bottom padding = 1040.
        assert!(
            (s.scroll_height() - 1040.0).abs() < 1.0,
            "scrollHeight should span the full content + padding (~1040), got {}",
            s.scroll_height()
        );
        // max scrollTop = scrollHeight - clientHeight = 1040 - 200 = 840.
        assert!(
            (s.max_scroll_top() - 840.0).abs() < 1.0,
            "max scrollTop should be 840, got {}",
            s.max_scroll_top()
        );
    }

    #[test]
    fn scroll_offset_clamps_to_legal_range() {
        let mut s = scroller_with_tall_child();
        let max_y = s.max_scroll_top();
        // Over-scroll past the bottom clamps to max.
        s.scroll_offset_y = 99999.0;
        s.clamp_scroll();
        assert!(
            (s.scroll_offset_y - max_y).abs() < 0.5,
            "over-scroll clamps to max ({}), got {}",
            max_y,
            s.scroll_offset_y
        );
        // Negative scroll clamps to 0.
        s.scroll_offset_y = -50.0;
        s.clamp_scroll();
        assert_eq!(s.scroll_offset_y, 0.0, "negative scroll clamps to 0");
        // A mid value is preserved.
        s.scroll_offset_y = 300.0;
        s.clamp_scroll();
        assert!(
            (s.scroll_offset_y - 300.0).abs() < 0.5,
            "in-range scroll preserved, got {}",
            s.scroll_offset_y
        );
    }

    #[test]
    fn non_scrollable_axis_pins_offset_to_zero() {
        let mut s = scroller_with_tall_child();
        // Make the horizontal axis non-scrollable; a horizontal offset must
        // be pinned even though there'd be room (child is narrower, so no
        // overflow anyway, but the pin must hold regardless).
        s.overflow_x = Overflow::Visible;
        s.scroll_offset_x = 40.0;
        s.clamp_scroll();
        assert_eq!(
            s.scroll_offset_x, 0.0,
            "overflow-x:visible pins scrollLeft to 0"
        );
    }

    #[test]
    fn visible_overflow_is_not_a_scroll_container() {
        let plain = block(
            "div",
            Style {
                display: Some(Display::Block),
                width: Some(LengthSpec::Px(100.0)),
                height: Some(LengthSpec::Px(50.0)),
                ..Style::default()
            },
            vec![text("hi")],
        );
        let root = layout(&plain, &LayoutConfig::default());
        assert!(
            find_scroller(&root).is_none(),
            "a plain overflow:visible box never scrolls"
        );
    }

    #[test]
    fn scroll_chain_finds_innermost_then_ancestor() {
        // Outer scroller (node_id 1) holds an inner scroller (node_id 2)
        // holding a tall child. A point inside the inner scroller's padding
        // box must return [inner, outer] — innermost first.
        let inner_child = block(
            "div",
            Style {
                display: Some(Display::Block),
                width: Some(LengthSpec::Px(80.0)),
                height: Some(LengthSpec::Px(2000.0)),
                ..Style::default()
            },
            vec![],
        );
        let inner = block(
            "div",
            Style {
                display: Some(Display::Block),
                width: Some(LengthSpec::Px(150.0)),
                height: Some(LengthSpec::Px(120.0)),
                overflow_y: Overflow::Scroll,
                ..Style::default()
            },
            vec![inner_child],
        );
        let outer = block(
            "div",
            Style {
                display: Some(Display::Block),
                width: Some(LengthSpec::Px(200.0)),
                height: Some(LengthSpec::Px(160.0)),
                overflow_y: Overflow::Scroll,
                ..Style::default()
            },
            vec![inner],
        );
        let mut root = layout(&outer, &LayoutConfig::default());
        // Tag the two scroll containers with stable node ids (the test
        // layout pipeline doesn't thread arena ids).
        fn tag(b: &mut LayoutBox, next: &mut u64) {
            if b.is_scroll_container() {
                *next += 1;
                b.node_id = Some(*next);
            }
            for c in &mut b.children {
                tag(c, next);
            }
        }
        let mut n = 0u64;
        tag(&mut root, &mut n);
        // node 1 = outer (visited first top-down), node 2 = inner.
        let outer_box = find_box_by_node_id(&root, 1).unwrap();
        let inner_box = find_box_by_node_id(&root, 2).unwrap();
        let p = inner_box.padding_rect();
        let (px, py) = (p.x + 5.0, p.y + 5.0);
        let chain = scroll_chain_at(&root, px, py);
        assert_eq!(chain.len(), 2, "point is inside both scrollers: {:?}", chain);
        assert_eq!(chain[0].node_id, 2, "innermost (inner) first");
        assert_eq!(chain[1].node_id, 1, "outer ancestor second");
        // Both report a positive max scrollTop.
        assert!(chain[0].max_top > 0.0 && chain[1].max_top > 0.0);
        let _ = outer_box;
    }
}
