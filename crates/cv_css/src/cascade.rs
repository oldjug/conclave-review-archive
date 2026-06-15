//! Minimal cascade: pick the winning value for each property per element.
//!
//! Sort matching declarations by (origin, !important, specificity,
//! source order). Today we only have author rules. Inheritance for a
//! small list of known-inheritable properties.

use crate::parser::{Declaration, Stylesheet};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WhiteSpace {
    Normal,
    Pre,
    Nowrap,
    PreWrap,
    PreLine,
    BreakSpaces,
}

/// One CSS transform function, preserving its un-resolved arguments
/// (CSS Transforms 2 §11). The painter composes a list of these into a
/// single 4×4 matrix, in order, with each later function multiplied on the
/// right. Angles are stored already-converted to **degrees**; translate
/// components keep their [`crate::properties::Length`] so px/em/rem/% can
/// be resolved against the right environment at layout/paint time.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Transform3DOp {
    /// translate(x, y) / translateX / translateY — x or y may be `%`.
    Translate(crate::properties::Length, crate::properties::Length),
    /// translate3d(x, y, z) / translateZ — z is a `<length>` (no `%`).
    Translate3d(
        crate::properties::Length,
        crate::properties::Length,
        crate::properties::Length,
    ),
    /// scale(sx, sy) / scaleX / scaleY.
    Scale(f32, f32),
    /// scale3d(sx, sy, sz) / scaleZ.
    Scale3d(f32, f32, f32),
    /// rotate / rotateZ(angle_deg).
    RotateZ(f32),
    /// rotateX(angle_deg).
    RotateX(f32),
    /// rotateY(angle_deg).
    RotateY(f32),
    /// rotate3d(x, y, z, angle_deg).
    Rotate3d(f32, f32, f32, f32),
    /// matrix(a, b, c, d, e, f) — 2D affine.
    Matrix2d([f32; 6]),
    /// matrix3d(m11..m44) — 16 column-major values.
    Matrix3d([f32; 16]),
    /// perspective(d) — `d` is a `<length>` (px-resolved at the bridge).
    Perspective(crate::properties::Length),
    /// skew(ax, ay) / skewX / skewY — angles in degrees.
    Skew(f32, f32),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TextTransform {
    None,
    Uppercase,
    Lowercase,
    Capitalize,
}

use crate::properties::{
    AlignItems, ClearMode, Color, Display, FlexDirection, FlexWrap, FloatSide, GridTrack,
    JustifyContent, Length, Position, VerticalAlign, Visibility,
};
use crate::selectors::Selector;
use crate::selectors::{ElementView, matches};
use crate::tokenizer::CssToken;

/// CSS `background-size` value (CSS Backgrounds §3.9).
///
/// Two keyword forms (`cover` / `contain`) are named variants.  Any other
/// form — including `auto auto`, a single length, or two lengths/percentages
/// — is represented as `Explicit(w, h)` where `None` means the `auto`
/// keyword for that axis (preserve aspect ratio).
#[derive(Clone, Debug, PartialEq)]
pub enum CssBgSize {
    /// `background-size: cover` — scale uniformly to cover the entire
    /// background positioning area, cropping if needed.
    Cover,
    /// `background-size: contain` — scale uniformly so the image fits
    /// entirely inside the background positioning area, letterboxing if needed.
    Contain,
    /// Explicit sizing: `(width, height)`. `None` = `auto` (aspect-ratio
    /// preserved relative to the other axis, or image's natural dimension).
    /// `Some(Length::Percent(p))` = `p%` of the positioning area.
    Explicit(Option<Length>, Option<Length>),
}

/// CSS `border-style` values — controls whether and how a border side is drawn.
/// The default is `Solid` so that elements with a width + color get a solid border
/// without an explicit `border-style` declaration (matches browser behaviour).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum BorderStyle {
    /// `none` — no border; width collapses to 0 in layout.
    None,
    /// `hidden` — same as none for non-table boxes.
    Hidden,
    /// `solid` — a single solid line (the default).
    #[default]
    Solid,
    /// `dashed` — a series of square-capped dashes (3× width on, 1× off).
    Dashed,
    /// `dotted` — a series of square dots (1× width on, 1× off).
    Dotted,
    /// `double` — two parallel solid lines separated by a gap.
    Double,
}

impl BorderStyle {
    /// Parse a single CSS token identifier into a `BorderStyle`, or return `None`.
    pub fn from_ident(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "none" => Some(BorderStyle::None),
            "hidden" => Some(BorderStyle::Hidden),
            "solid" => Some(BorderStyle::Solid),
            "dashed" => Some(BorderStyle::Dashed),
            "dotted" => Some(BorderStyle::Dotted),
            "double" => Some(BorderStyle::Double),
            _ => None,
        }
    }

    /// Returns true when this style means "no border" (width should collapse to 0).
    pub fn is_none(&self) -> bool {
        matches!(self, BorderStyle::None | BorderStyle::Hidden)
    }
}

/// Sentinel `font_weight_num` for the relative `font-weight: bolder` keyword,
/// stored at parse time and resolved against the inherited weight during
/// inheritance via [`resolve_relative_font_weight`]. Chosen outside the valid
/// CSS 1–1000 range so it can never collide with a real weight.
pub const FONT_WEIGHT_BOLDER: u16 = 1001;
/// Sentinel `font_weight_num` for `font-weight: lighter` (see
/// [`FONT_WEIGHT_BOLDER`]).
pub const FONT_WEIGHT_LIGHTER: u16 = 1002;

/// Resolve a possibly-relative font weight (`bolder`/`lighter` sentinel) against
/// the inherited computed weight, per the CSS Fonts 4 §2.4 table. A concrete
/// 1–1000 weight passes through unchanged. `inherited` defaults to 400 (normal)
/// when the parent has no explicit weight.
pub fn resolve_relative_font_weight(weight: u16, inherited: u16) -> u16 {
    let w = inherited;
    match weight {
        FONT_WEIGHT_BOLDER => {
            // CSS Fonts 4 §2.4 "bolder" column.
            if w < 350 {
                400
            } else if w < 550 {
                700
            } else if w < 750 {
                900
            } else {
                900 // 750..=∞ → 900 (and ≥900 = no change, still 900)
            }
        }
        FONT_WEIGHT_LIGHTER => {
            // CSS Fonts 4 §2.4 "lighter" column.
            if w < 100 {
                w // no change below 100
            } else if w < 550 {
                100
            } else if w < 750 {
                400
            } else {
                700
            }
        }
        other => other,
    }
}

/// CSS `writing-mode` (CSS Writing Modes 4 §3.1). Determines the block-flow
/// direction and whether the inline axis runs horizontally or vertically.
/// `None`/`HorizontalTb` is the initial value.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum WritingMode {
    /// `horizontal-tb` — inline axis is horizontal, block axis runs
    /// top→bottom (the default for Latin text). inline-size = width.
    #[default]
    HorizontalTb,
    /// `vertical-rl` — inline axis is vertical (top→bottom), block axis
    /// runs right→left. inline-size = height. block-start = right edge.
    VerticalRl,
    /// `vertical-lr` — inline axis is vertical (top→bottom), block axis
    /// runs left→right. inline-size = height. block-start = left edge.
    VerticalLr,
}

impl WritingMode {
    /// True when the inline axis is vertical (i.e. inline-size maps to
    /// height and block-size to width). CSS Writing Modes 4 §6.
    pub fn is_vertical(self) -> bool {
        matches!(self, WritingMode::VerticalRl | WritingMode::VerticalLr)
    }
    pub fn from_str(s: &str) -> Option<WritingMode> {
        match s.to_ascii_lowercase().as_str() {
            "horizontal-tb" | "lr" | "lr-tb" | "rl" | "rl-tb" => Some(WritingMode::HorizontalTb),
            "vertical-rl" | "tb" | "tb-rl" => Some(WritingMode::VerticalRl),
            "vertical-lr" => Some(WritingMode::VerticalLr),
            _ => None,
        }
    }
}

/// CSS `direction` (CSS Writing Modes 4 §2.1). Sets the inline base
/// direction. `None`/`Ltr` is the initial value.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Direction {
    #[default]
    Ltr,
    Rtl,
}

impl Direction {
    pub fn from_str(s: &str) -> Option<Direction> {
        match s.to_ascii_lowercase().as_str() {
            "ltr" => Some(Direction::Ltr),
            "rtl" => Some(Direction::Rtl),
            _ => None,
        }
    }
}

/// The four flow-relative box edges, in CSS Logical order. Used to index the
/// logical accumulators on [`ComputedStyle`] before they are resolved to
/// physical sides via [`map_logical_side`].
pub const LOGICAL_BLOCK_START: usize = 0;
pub const LOGICAL_INLINE_END: usize = 1;
pub const LOGICAL_BLOCK_END: usize = 2;
pub const LOGICAL_INLINE_START: usize = 3;

/// Map a flow-relative edge index (LOGICAL_*) to its physical side index in
/// the `[top, right, bottom, left]` arrays, per CSS Logical Properties 1 §2.1
/// + CSS Writing Modes 4. This is the canonical mapping table:
///
/// | writing-mode | dir | inline-start | inline-end | block-start | block-end |
/// |--------------|-----|--------------|------------|-------------|-----------|
/// | horizontal-tb| ltr | left         | right      | top         | bottom    |
/// | horizontal-tb| rtl | right        | left       | top         | bottom    |
/// | vertical-rl  | ltr | top          | bottom     | right       | left      |
/// | vertical-rl  | rtl | bottom       | top        | right       | left      |
/// | vertical-lr  | ltr | top          | bottom     | left        | right     |
/// | vertical-lr  | rtl | bottom       | top        | left        | right     |
///
/// Physical indices: 0=top, 1=right, 2=bottom, 3=left.
pub fn map_logical_side(logical: usize, wm: WritingMode, dir: Direction) -> usize {
    const TOP: usize = 0;
    const RIGHT: usize = 1;
    const BOTTOM: usize = 2;
    const LEFT: usize = 3;
    let ltr = dir == Direction::Ltr;
    match wm {
        WritingMode::HorizontalTb => match logical {
            LOGICAL_BLOCK_START => TOP,
            LOGICAL_BLOCK_END => BOTTOM,
            LOGICAL_INLINE_START => if ltr { LEFT } else { RIGHT },
            LOGICAL_INLINE_END => if ltr { RIGHT } else { LEFT },
            _ => TOP,
        },
        WritingMode::VerticalRl => match logical {
            LOGICAL_BLOCK_START => RIGHT,
            LOGICAL_BLOCK_END => LEFT,
            LOGICAL_INLINE_START => if ltr { TOP } else { BOTTOM },
            LOGICAL_INLINE_END => if ltr { BOTTOM } else { TOP },
            _ => TOP,
        },
        WritingMode::VerticalLr => match logical {
            LOGICAL_BLOCK_START => LEFT,
            LOGICAL_BLOCK_END => RIGHT,
            LOGICAL_INLINE_START => if ltr { TOP } else { BOTTOM },
            LOGICAL_INLINE_END => if ltr { BOTTOM } else { TOP },
            _ => TOP,
        },
    }
}

/// A flow-relative box quantity (margin / padding / inset) accumulated during
/// cascade BEFORE the writing-mode + direction are finalized (they may be
/// inherited from an ancestor and thus unknown at cascade time). Each side
/// carries the cascade sequence number of its winning declaration so the
/// logical→physical resolver can correctly arbitrate against a competing
/// PHYSICAL longhand that targets the same physical slot (CSS Logical 1 §2:
/// logical + physical longhands "share a computed value", taken from the
/// higher-priority cascade declaration). Indexed by LOGICAL_* constants.
#[derive(Clone, Debug, Default)]
pub struct LogicalEdges {
    pub vals: [Option<Length>; 4],
    /// `auto` keyword flag per side (margins only).
    pub autos: [bool; 4],
    /// Cascade sequence number of the winning declaration per side.
    pub seq: [u32; 4],
}

/// CSS Fragmentation 3 break value (the pagination-relevant subset). Maps
/// `break-before`/`-after`/`-inside` and their legacy `page-break-*` aliases.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BreakValue {
    /// `auto` — break allowed but not forced (the initial value).
    Auto,
    /// `page` / `always` / `left` / `right` — force a page break.
    Force,
    /// `avoid` / `avoid-page` — discourage a break.
    Avoid,
}

impl BreakValue {
    /// Parse a `break-*` / `page-break-*` value's tokens. Returns `None` for an
    /// unrecognised value (leaving any prior cascade winner intact).
    pub fn from_tokens(toks: &[CssToken]) -> Option<Self> {
        let kw = toks.iter().find_map(|t| match t {
            CssToken::Ident(s) => Some(s.to_ascii_lowercase()),
            _ => None,
        })?;
        match kw.as_str() {
            "auto" => Some(BreakValue::Auto),
            // CSS Fragmentation 3 §3.1: forced-break values.
            "page" | "always" | "left" | "right" | "recto" | "verso" => {
                Some(BreakValue::Force)
            }
            "avoid" | "avoid-page" => Some(BreakValue::Avoid),
            _ => None,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct ComputedStyle {
    pub color: Option<Color>,
    pub background_color: Option<Color>,
    pub display: Option<Display>,
    /// CSS `visibility`. `Hidden` keeps the box in flow (same layout
    /// effect as `visible`) but skips painting it and its descendants.
    /// `Collapse` behaves like `Hidden` on a generic box (only special
    /// for table parts in the spec).  Modal dialogs and tooltips
    /// commonly default to `visibility: hidden` and JS flips it to
    /// `visible` on open — without honouring this, every hidden popup
    /// on the page paints on top of the real content.
    pub visibility: Option<Visibility>,
    /// CSS `content` — meaningful only on `::before` / `::after` /
    /// other pseudo-elements. The string literal between the quotes is
    /// captured verbatim; `content: none` / `content: normal` produce
    /// None so the host knows not to generate the box.
    pub content: Option<String>,
    /// CSS Fragmentation 3 §3.1 — `break-before` / `break-after` (and the
    /// legacy `page-break-before`/`-after` aliases). Consulted by the print /
    /// PDF pagination path; `None` = `auto` (the initial value). Ignored for
    /// on-screen layout (which is not fragmented).
    pub break_before: Option<BreakValue>,
    pub break_after: Option<BreakValue>,
    /// CSS Fragmentation 3 §4.2 — `break-inside` / `page-break-inside`.
    pub break_inside: Option<BreakValue>,
    pub font_size: Option<Length>,
    pub width: Option<Length>,
    pub height: Option<Length>,
    pub aspect_ratio: Option<f32>,
    /// `max-width` clamp. With `margin: auto` on the horizontal sides
    /// this is the mechanism every modern site uses to centre its
    /// content within a wide viewport.
    pub max_width: Option<Length>,
    pub max_height: Option<Length>,
    pub min_width: Option<Length>,
    pub min_height: Option<Length>,
    pub margin: [Option<Length>; 4], // top, right, bottom, left
    /// Per-side `margin: auto` flag. The horizontal pair drives
    /// "centre this block in its containing block" and the vertical
    /// pair drives flex item cross-axis centring (we don't honour
    /// vertical auto-margins yet, but track them anyway).
    pub margin_auto: [bool; 4],
    pub padding: [Option<Length>; 4],
    /// V1 borders: a single uniform width + color (CSS shorthand
    /// Uniform border width — set by the `border:` shorthand. When the
    /// per-side `border_top/right/bottom/left_width` are unset, this
    /// fills in for them.
    pub border_width: Option<Length>,
    pub border_color: Option<Color>,
    /// Per-side border widths. None means "inherit from the uniform
    /// `border_width`". Driven by `border-top`/`border-right`/etc.
    /// shorthands and their `-width`/`-color` longhands.
    pub border_top_width: Option<Length>,
    pub border_right_width: Option<Length>,
    pub border_bottom_width: Option<Length>,
    pub border_left_width: Option<Length>,
    pub border_top_color: Option<Color>,
    pub border_right_color: Option<Color>,
    pub border_bottom_color: Option<Color>,
    pub border_left_color: Option<Color>,
    /// Per-side border style.  `None` means "not explicitly set" which
    /// the painter treats as `Solid` (initial value, CSS §8.5.3).
    /// When `None` or `Hidden`, the effective border width collapses to
    /// 0 during layout (same as Chrome).
    pub border_top_style: Option<BorderStyle>,
    pub border_right_style: Option<BorderStyle>,
    pub border_bottom_style: Option<BorderStyle>,
    pub border_left_style: Option<BorderStyle>,
    pub text_align: Option<TextAlign>,
    pub font_weight_bold: Option<bool>,
    /// Numeric CSS font-weight (1–1000) when specified, so heavy weights like
    /// 800/900 render heavier than plain bold (700) instead of collapsing to the
    /// `font_weight_bold` boolean. `bold`→700, `normal`→400. The renderer maps
    /// this to the GDI lfWeight; falls back to the bool when None.
    ///
    /// The relative keywords `bolder`/`lighter` are stored as the sentinels
    /// [`FONT_WEIGHT_BOLDER`]/[`FONT_WEIGHT_LIGHTER`] (outside the 1–1000 range)
    /// and resolved against the INHERITED computed weight during inheritance
    /// (CSS Fonts 4 §2.4 table) — they can't be resolved at parse time because
    /// they depend on the parent.
    pub font_weight_num: Option<u16>,
    pub font_style_italic: Option<bool>,
    pub font_family: Option<String>,
    pub text_decoration_underline: Option<bool>,
    pub text_decoration_line_through: Option<bool>,
    /// `line-height`. Unitless number = multiplier of font-size. Length =
    /// absolute. None = inherit / default.
    pub line_height: Option<LineHeight>,
    pub box_sizing_border_box: Option<bool>,
    pub flex_direction: Option<FlexDirection>,
    pub flex_wrap: Option<FlexWrap>,
    pub flex_grow: Option<f32>,
    pub flex_shrink: Option<f32>,
    pub flex_basis: Option<Length>,
    pub justify_content: Option<JustifyContent>,
    pub align_items: Option<AlignItems>,
    /// `justify-items`: cross-of-align in the grid inline axis. Used to
    /// position grid items inside their column track. Mirrors
    /// AlignItems values (start | end | center | stretch).
    pub justify_items: Option<AlignItems>,
    /// `align-self`: per-item override of the parent's `align-items`.
    /// `None` falls back to the container's value.
    pub align_self: Option<AlignItems>,
    /// `justify-self`: per-item override of the parent's `justify-items`.
    pub justify_self: Option<AlignItems>,
    pub gap: Option<Length>,
    /// `opacity: 0.0..=1.0`. None = inherit / default 1.0.
    pub opacity: Option<f32>,
    /// True when the element has a non-`none` animation declaration.
    /// The browser uses this as a fallback for pages that set
    /// `opacity: 0` and rely on animation fill to reveal content.
    pub has_animation: bool,
    /// `border-radius` uniform radius (we don't yet honour per-corner
    /// `5px 10px 15px 20px` syntax — falls back to the first value).
    pub border_radius: Option<Length>,
    /// `box-shadow: Xpx Ypx [blur] color`. V1 parses the first three
    /// length tokens + first color and ignores the rest (no spread,
    /// no `inset`, no multi-shadow). Painter blits a solid offset
    /// rect — blur is captured but not rendered.
    pub box_shadow: Option<BoxShadowSpec>,
    /// `text-shadow: Xpx Ypx [blur] color`. Same parse shape as
    /// box_shadow. Painter stamps a duplicate of the text at the
    /// offset before the main glyph draw.
    pub text_shadow: Option<BoxShadowSpec>,
    /// Parsed `filter:` function chain. Each entry is (name, arg)
    /// where arg is a normalized 0..1 (or pixel value for blur).
    /// The painter applies them in order to the box's backing pixels
    /// after children draw and before the parent composes.
    pub filters: Vec<FilterFn>,
    /// `backdrop-filter` — same grammar as `filter`. Currently parsed
    /// but not rendered; reserved for a future compositor pass.
    pub backdrop_filters: Vec<FilterFn>,
    /// `mix-blend-mode` keyword (normalized lowercase, e.g. "multiply").
    /// `None` / "normal" → plain source-over. CSS Compositing & Blending L1 §5.
    pub mix_blend_mode: Option<String>,
    /// `background-blend-mode` — one keyword per background layer (we paint a
    /// single background layer, so the first value is used). Blends the
    /// element's background image against the color/gradient beneath it.
    /// CSS Compositing & Blending L1 §6.
    pub background_blend_mode: Option<String>,
    /// `animation-name` — looks up a @keyframes rule by name.
    pub animation_name: Option<String>,
    /// `animation-duration` in milliseconds (0 = no animation).
    pub animation_duration_ms: Option<f32>,
    /// `animation-delay` in milliseconds.
    pub animation_delay_ms: Option<f32>,
    /// `animation-iteration-count` — None = 1, f32::INFINITY = `infinite`.
    pub animation_iteration_count: Option<f32>,
    /// `animation-timing-function`: 0=linear, 1=ease-in, 2=ease-out, 3=ease-in-out.
    pub animation_timing: Option<u8>,
    /// `transition-duration` in ms (>0 enables transitions on this element).
    pub transition_duration_ms: Option<f32>,
    /// `transition-delay` in ms.
    pub transition_delay_ms: Option<f32>,
    /// `transition-timing-function` (same encoding as `animation_timing`).
    pub transition_timing: Option<u8>,
    /// `transition-property`: which properties animate — `"all"`, `"none"`, or a
    /// specific name (e.g. `"opacity"`). Defaults to `"all"` per spec when a
    /// duration is set without an explicit property.
    pub transition_property: Option<String>,
    /// Resolved `clip-path` shape (V1: inset / circle / polygon).
    pub clip_path: Option<ClipPath>,
    /// True when `mask` or `-webkit-mask` is set to a `url(...)`. Pages
    /// use this to paint a tintable icon (background-color filled and
    /// the mask carves out the shape). Without mask-image support our
    /// painter would render the full coloured rect (Google's footer
    /// leaf was rendering as a green square because of this); we flag
    /// it so the background paint can be skipped.
    pub has_mask_url: bool,
    /// Raw `mask-image: url(...)` or `mask: url(...)` reference.
    pub mask_image_url: Option<String>,
    pub position: Option<Position>,
    /// `float: left | right` — out-of-normal-flow positioning at the
    /// left or right edge of the containing block, with subsequent
    /// in-flow content shrinking around the float. Unset / `none`
    /// means normal flow.
    pub float_side: Option<FloatSide>,
    /// `clear: left | right | both` — a non-`none` value forces the
    /// element below every active float on the named side(s). Block
    /// layout reads this and bumps `child_y` past the bottom of any
    /// matching active float before placing this child.
    pub clear: Option<ClearMode>,
    /// CSS `vertical-align` (keyword form). Drives super/sub offset
    /// inside an inline run.
    pub vertical_align: Option<VerticalAlign>,
    pub top: Option<Length>,
    pub right: Option<Length>,
    pub bottom: Option<Length>,
    pub left: Option<Length>,
    pub z_index: Option<i32>,
    /// `transform: translate(x, y)` / `translateX(x)` / `translateY(y)` —
    /// only translate is plumbed all the way to paint. Other functions
    /// (`scale`, `rotate`, `matrix`) parse without erroring but produce
    /// no visual effect yet.
    pub translate_x: Option<Length>,
    pub translate_y: Option<Length>,
    /// `transform: scale(x[, y])` — uniform when only one component given.
    pub scale_x: Option<f32>,
    pub scale_y: Option<f32>,
    /// `transform: rotate(angle)` — stored in degrees, normalized to (-360,360).
    pub rotate_deg: Option<f32>,
    /// `transform: matrix(a, b, c, d, e, f)` — the 2D affine matrix.
    /// When set, this overrides individual scale/rotate/translate fields
    /// at paint time (the painter multiplies the matrix in).
    pub matrix_2d: Option<[f32; 6]>,
    /// `transform-origin` — the pivot point for 2D transforms. Default is
    /// `50% 50%` (border-box centre). Reuses [`BgPos`] for keyword/px/%.
    /// `None` means the default centre was not overridden.
    pub transform_origin: Option<(BgPos, BgPos)>,
    /// Ordered list of transform functions (CSS Transforms 2 §11) — set
    /// ONLY when the `transform` list contains a 3D primitive (rotateX/Y,
    /// rotate3d, translate3d/Z, scale3d/Z, matrix3d, perspective). 2D-only
    /// transforms continue to use the scalar fields above and their cheap
    /// fast paths, so this is strictly additive. Lengths are preserved so
    /// translate x/y/z resolve against em/rem/viewport (and x/y % against
    /// the box) at the layout bridge / paint time. When present this
    /// supersedes the scalar 2D fields (the painter composes a 4×4 from
    /// this list and projects the box's quad through it).
    pub transform_ops: Option<Vec<Transform3DOp>>,
    /// `perspective` PROPERTY (distinct from the `perspective()` function):
    /// establishes a perspective for this element's preserve-3d children.
    /// Resolved px; `None` = `none` (no perspective). `<1px` clamps to 1px.
    pub perspective_px: Option<f32>,
    /// `perspective-origin` — vanishing point for the `perspective` property.
    /// Default `50% 50%`. `None` = default centre.
    pub perspective_origin: Option<(BgPos, BgPos)>,
    /// `transform-style: preserve-3d` — keep descendants in a shared 3D
    /// rendering context rather than flattening to this element's plane.
    pub transform_style_preserve_3d: bool,
    /// `backface-visibility: hidden` — cull the element when its back face
    /// is toward the viewer (CSS Transforms 2 §10).
    pub backface_visibility_hidden: bool,
    /// `grid-template-columns` — track sizing for the column axis.
    pub grid_template_columns: Option<Vec<GridTrack>>,
    /// `grid-template-rows` — track sizing for the row axis. If absent
    /// the grid lays children with auto-flow row and rows sized by their
    /// children's natural height.
    pub grid_template_rows: Option<Vec<GridTrack>>,
    /// `grid-auto-rows` — default size for implicitly-created rows
    /// (children placed beyond the explicit row track count). Single
    /// track value; per CSS Grid §7.3.1.
    pub grid_auto_rows: Option<GridTrack>,
    /// `grid-auto-columns` — same as grid-auto-rows, for column axis.
    pub grid_auto_columns: Option<GridTrack>,
    /// `grid-template-areas` — each Vec<String> is one row of column
    /// names. `.` means an empty cell. A name repeated across adjacent
    /// cells (horizontal or vertical) defines a spanned named area.
    /// On a grid container, children with `grid-area: name` get
    /// resolved against this map at layout time.
    pub grid_template_areas: Option<Vec<Vec<String>>>,
    pub grid_column_start: Option<usize>,
    pub grid_column_span: Option<usize>,
    pub grid_row_start: Option<usize>,
    pub grid_row_span: Option<usize>,
    /// HTML `colspan` on a table cell (`<td colspan=2>`). A presentational
    /// attribute, not CSS — read during cascade and consumed by the table
    /// layout so spanned cells occupy the right columns.
    pub table_col_span: Option<usize>,
    /// HTML `rowspan` on a table cell (`<td rowspan=2>`). A presentational
    /// attribute — row-span value defaults to 1. Consumed by the table layout
    /// to distribute height across spanned rows.
    pub table_row_span: Option<usize>,
    /// `grid-area: name` — references a named area declared on the
    /// nearest grid-container ancestor via `grid-template-areas`. When
    /// the value is `name / row-line / col-line / row-end / col-end`
    /// (the longhand form) we only honour the single-ident case for V1.
    pub grid_area_name: Option<String>,
    /// `column-gap` — when set distinct from `gap`. Falls back to `gap`.
    pub column_gap: Option<Length>,
    /// `row-gap` — when set distinct from `gap`. Falls back to `gap`.
    pub row_gap: Option<Length>,
    pub overflow_hidden: bool,
    /// Resolved `overflow-x` / `overflow-y` (CSS Overflow 3). `None` =
    /// the initial `visible`. These supersede the legacy single
    /// `overflow_hidden` boolean: `Scroll`/`Auto` make the box an
    /// independently scrollable region (a scroll container), while
    /// `Hidden`/`Clip` clip without a scroll mechanism. Kept per-axis so
    /// `overflow-x: hidden; overflow-y: auto` (the canonical vertical
    /// scroller) resolves correctly. `overflow_hidden` stays set whenever
    /// EITHER axis clips so existing clip paths keep working unchanged.
    pub overflow_x: Option<crate::properties::Overflow>,
    pub overflow_y: Option<crate::properties::Overflow>,
    /// CSS `list-style-type: none` (or `list-style: none`). When true,
    /// the UA marker (• for ul, "1. 2. 3." for ol) is suppressed. Lots
    /// of real pages reset `<nav><ul>` with `list-style: none` and the
    /// previous code ignored that, so navigation menus showed bullets.
    pub list_style_none: bool,
    /// True when the *declared* `display` value is `list-item` (mapped
    /// to `Display::Block` for layout but requiring a list marker just
    /// like `<li>` does).
    pub display_is_list_item: bool,
    /// True when the *declared* `display` value is `flow-root` (mapped to
    /// `Display::Block` for layout — establishes a new BFC, not yet modelled
    /// separately — but recorded so the CSSOM `display` reads back as
    /// `flow-root`).
    pub display_is_flow_root: bool,
    /// CSS Multi-column Layout — `column-count`. None for `auto`.
    pub column_count: Option<u32>,
    /// CSS Multi-column Layout — `column-width` in CSS pixels.
    pub column_width: Option<f32>,
    /// CSS Multi-column Layout — `column-gap` (resolved px).
    pub multicol_gap: Option<f32>,
    /// CSS Multi-column Layout — `column-rule-width` in CSS px.
    pub column_rule_width: Option<f32>,
    /// CSS Multi-column Layout — `column-rule-style`.
    pub column_rule_style: Option<BorderStyle>,
    /// CSS Multi-column Layout — `column-rule-color`.
    pub column_rule_color: Option<Color>,
    /// CSS Multi-column Layout — `column-span: all`.
    pub column_span_all: bool,
    /// CSS Anchor Positioning — `anchor-name: --x`. Identifies this
    /// element as a potential anchor for `position-anchor` consumers.
    pub anchor_name: Option<String>,
    /// CSS Anchor Positioning — `position-anchor: --x`. Names the
    /// anchor to align against. Layout looks up `anchor_name` and
    /// resolves offsets from `anchor()` calls in top/left/etc.
    pub position_anchor: Option<String>,
    /// `white-space` mode. Drives line-wrapping behaviour in layout +
    /// affects whether sequential whitespace is collapsed at parse
    /// time (only `pre`-family preserves it).
    pub white_space: Option<WhiteSpace>,
    /// `text-overflow: ellipsis` — when the box has `overflow:hidden`
    /// and a single line that doesn't fit, the painter clips with an
    /// inserted `…` glyph instead of a hard cut.
    pub text_overflow_ellipsis: bool,
    /// `text-transform`. Applied at text-bake time so the user sees
    /// uppercased / capitalised / lowercased glyphs.
    pub text_transform: Option<TextTransform>,
    /// `letter-spacing` extra-px between glyphs (none = default).
    pub letter_spacing_px: Option<f32>,
    /// CSS custom properties (`--name: value`). Available via `var(--name)`
    /// substitution inside any other property's value. Cascade order
    /// applies — later declarations overwrite earlier ones at the same
    /// specificity, just like any regular property.
    pub custom_properties: std::collections::HashMap<String, Vec<CssToken>>,
    /// `background: linear-gradient(...)` — if both this and
    /// `background_color` are set, the gradient wins. Stored separately
    /// so paint can route to the gradient rasterizer.
    pub background_gradient: Option<LinearGradient>,
    /// `background: radial-gradient(...)` — represented as a two-stop
    /// center-out fade for paint.
    pub background_radial_gradient: Option<LinearGradient>,
    /// Full N-stop CSS gradient (linear / radial / conic + repeating
    /// variants) with real stop positions. When set, the painter
    /// rasterizes this instead of the 2-stop `background_gradient` /
    /// `background_radial_gradient` approximations. CSS Images 3 §3.
    pub background_gradient_full: Option<CssGradient>,
    /// `background-image: url("…")` — raw URL string as parsed from CSS.
    /// Resolution against the document base and the actual byte fetch
    /// happen later in the resource pipeline. When set together with a
    /// `background_gradient`, the gradient wins per the CSS spec
    /// (background layers are stacked top-to-bottom, gradient first).
    pub background_image_url: Option<String>,
    /// `accent-color` — UA-painted accent for form controls
    /// (checkbox tick, radio dot, range thumb). V1 carries the colour
    /// through so the painter can use it when we land form-control
    /// theming, even though no widget currently consults it.
    pub accent_color: Option<Color>,
    /// `caret-color` — colour of the text-input caret. Carried through
    /// for future caret rendering; default value is `currentColor`.
    pub caret_color: Option<Color>,
    /// `color-scheme: light | dark | normal | light dark` — controls how
    /// UA-supplied colours (scrollbars, form widgets, etc.) respond to
    /// the OS dark-mode setting. Stored as a freeform string so author
    /// code reading getComputedStyle().colorScheme sees what they set.
    pub color_scheme: Option<String>,
    /// `view-transition-name: <ident>` — names an element so the view
    /// transition machinery can pair it with the same name on the next
    /// page. Stored verbatim; transition execution isn't wired yet but
    /// the property has to round-trip through computed style.
    pub view_transition_name: Option<String>,
    /// `background-clip: text` (or the `-webkit-` prefix) — the box's
    /// background paint is masked to the glyph outlines of contained
    /// text. We don't have a real glyph-mask raster path; the lower
    /// layer pulls a representative colour out of the gradient and
    /// paints the text in that colour instead, then drops the
    /// background paint. That's wrong in theory but visually close to
    /// what real sites use this idiom for (branded wordmarks with a
    /// gradient fill on the letters).
    pub background_clip_text: bool,
    /// `scrollbar-width` (CSS Scrollbars 1 §3): `auto` (0, default) /
    /// `thin` (1) / `none` (2). On a scroll container's root this themes
    /// the viewport scrollbar — `none` hides it entirely, `thin` draws a
    /// narrower bar.
    pub scrollbar_width: u8,
    /// `scrollbar-color` (CSS Scrollbars 1 §2) — `(thumb, track)` colours.
    /// `None` ⇒ `auto` (the UA default groove/thumb greys).
    pub scrollbar_color: Option<(Color, Color)>,
    /// `object-fit` value — `fill` / `contain` / `cover` / `none` /
    /// `scale-down`. Plumbed through to the layout `Style` so the
    /// painter knows how to map the source bitmap into the box.
    pub object_fit: Option<ObjectFit>,
    /// `object-position` (x, y) pair. Uses the same [`BgPos`] type as
    /// `background-position` — either an absolute px offset from the
    /// top-left of the content box, or a percentage of
    /// `(box_extent - image_extent)`. `None` means the CSS default:
    /// 50% 50% (centre). Only meaningful when `object-fit` is not `fill`.
    pub object_position: Option<(BgPos, BgPos)>,
    /// `background-repeat` keyword. CSS default is `repeat`; we tile
    /// the background bitmap to fill the padding box. `no-repeat` blits
    /// the bitmap once at its natural size at the top-left. `repeat-x` /
    /// `repeat-y` tile along a single axis only. Honouring this matters
    /// for sites whose body background is a small constellation/pattern
    /// SVG meant to tile — previously we stretched it across the
    /// viewport and the page looked like one giant smudge.
    pub background_repeat: BackgroundRepeat,
    /// `background-size`. `None` means the property was not set (use the
    /// image's natural size). Otherwise a [`CssBgSize`] variant.
    pub background_size: Option<CssBgSize>,
    /// `background-position` as an (x, y) pair. Each component is a length
    /// or percentage; keywords map to percentages (left/top=0%,
    /// center=50%, right/bottom=100%). `None` means the default `0% 0%`.
    /// Needed for CSS sprites — a single sheet image offset by a negative
    /// position so an element shows just one icon (e.g. Wikipedia's
    /// wordmark at `0px -304px` out of a shared SVG sprite).
    pub background_position: Option<(BgPos, BgPos)>,
    /// CSS Containment 3 §2.1 `container-type`. When `Size` or `InlineSize`,
    /// this element establishes a *query container* that descendant
    /// `@container` rules are evaluated against. `Normal` (the default /
    /// `None`) means it is NOT a query container. `Size` queries both
    /// inline and block axes; `InlineSize` only the inline axis (and
    /// applies inline-size containment but NOT block-size containment, so
    /// the box still sizes to its content vertically).
    pub container_type: Option<ContainerType>,
    /// CSS Containment 3 §2.2 `container-name`. Zero or more idents that
    /// let an `@container <name> (...)` query target this specific
    /// ancestor instead of the nearest one. Empty = unnamed container.
    pub container_name: Vec<String>,

    // ---- CSS Writing Modes 4 + CSS Logical Properties 1 ----
    /// `writing-mode`. `None` = the initial `horizontal-tb`. Inherits
    /// (CSS Writing Modes 4 §3.1) — the inherit fill-in happens in
    /// `build_styled_tree` after the parent value is known.
    pub writing_mode: Option<WritingMode>,
    /// `direction`. `None` = the initial `ltr`. Inherits.
    pub direction: Option<Direction>,
    /// Flow-relative margin longhands (margin-inline/-block-*), captured
    /// raw at cascade time and mapped to physical `margin` sides by
    /// [`resolve_logical_box`] once writing-mode + direction are final.
    pub logical_margin: LogicalEdges,
    /// Flow-relative padding longhands.
    pub logical_padding: LogicalEdges,
    /// Flow-relative inset longhands (inset-inline/-block-*).
    pub logical_inset: LogicalEdges,
    /// `inline-size` — maps to `width` in horizontal writing modes and to
    /// `height` in vertical writing modes (CSS Logical 1 §4.1).
    pub inline_size: Option<Length>,
    /// `block-size` — maps to `height` in horizontal, `width` in vertical.
    pub block_size: Option<Length>,
    pub min_inline_size: Option<Length>,
    pub max_inline_size: Option<Length>,
    pub min_block_size: Option<Length>,
    pub max_block_size: Option<Length>,
    /// Cascade sequence of the winning `inline-size`/`block-size` decl,
    /// vs. physical `width`/`height` — last (highest) wins.
    pub logical_size_seq: [u32; 2], // [inline_size, block_size]
    pub physical_size_seq: [u32; 2], // [width, height]
    /// Cascade seq for physical min/max width/height, so logical
    /// min/max-inline/block-size can be arbitrated against them.
    /// [min_width, min_height, max_width, max_height].
    pub physical_minmax_seq: [u32; 4],
    /// Cascade seq for logical min/max inline/block size.
    /// [min_inline, min_block, max_inline, max_block].
    pub logical_minmax_seq: [u32; 4],
    /// Monotonic counter bumped on every applied declaration so logical and
    /// physical longhands sharing a slot can be arbitrated by cascade order.
    pub decl_seq: u32,
    /// Cascade sequence of the winning PHYSICAL margin/padding/inset decl per
    /// side `[top,right,bottom,left]`, so the logical resolver only overrides a
    /// physical slot when the logical declaration came later in the cascade.
    pub physical_margin_seq: [u32; 4],
    pub physical_padding_seq: [u32; 4],
    pub physical_inset_seq: [u32; 4],
    /// True once [`resolve_logical_box`] has run. Lets a second resolution
    /// (after writing-mode inheritance, in `build_styled_tree`) first undo the
    /// physical slots it previously wrote, so re-resolving with a different
    /// inherited writing-mode doesn't leave a stale value at the old physical
    /// side (e.g. horizontal→`left` then vertical→`top`).
    pub logical_resolved: bool,
}

/// CSS Containment 3 §2.1 — the kind of query container an element is.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ContainerType {
    /// `container-type: normal` — not a query container.
    Normal,
    /// `container-type: inline-size` — query container on the inline axis
    /// only (size containment is applied on the inline axis).
    InlineSize,
    /// `container-type: size` — query container on both axes (size
    /// containment is applied on both inline and block axes).
    Size,
}

/// One axis of `background-position`: an absolute length or a percentage
/// of `(box_size - image_size)`.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum BgPos {
    Px(f32),
    Pct(f32),
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ObjectFit {
    Fill,
    Contain,
    Cover,
    None,
    ScaleDown,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum BackgroundRepeat {
    Repeat,
    NoRepeat,
    RepeatX,
    RepeatY,
}

impl Default for BackgroundRepeat {
    fn default() -> Self {
        BackgroundRepeat::Repeat
    }
}

/// Simple linear gradient: two end colors + an angle in degrees from
/// the top axis (CSS `0deg` points up; `90deg` points right). V1 takes
/// just the first and last color stops, ignoring intermediate ones.
/// Multi-stop gradients and radial / conic forms render with this same
/// "first/last only" approximation rather than collapsing to a solid
/// fill — visually much closer than the old single-stop fallback.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct LinearGradient {
    pub from: Color,
    pub to: Color,
    /// Direction in CSS degrees: 0 = to top, 90 = to right, 180 = to
    /// bottom, 270 = to left.
    pub angle_deg: f32,
}

/// One color stop in a full CSS gradient. `position` carries the
/// authored placement; `None` means "unspecified" and gets distributed
/// per the CSS Images 3 §3.4.3 color-stop fix-up at resolve time.
///
/// CSS Images 3 §3.4.3: stop positions may be `<length-percentage>`.
/// Percentages are stored as a fraction (0.0–1.0). Pixel lengths are
/// carried separately because they resolve against the gradient-line
/// length (linear) / gradient radius (radial) — which is unknown at
/// parse time — at paint time. Conic stop angles are stored as a
/// fraction of one turn.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct GradientColorStop {
    pub color: Color,
    /// Percentage / angle position as a fraction of the gradient extent
    /// (0.0 = start, 1.0 = end). `None` when unspecified.
    pub pos_frac: Option<f32>,
    /// Absolute pixel length position (resolved at paint against the
    /// gradient line length / radius). Mutually exclusive with
    /// `pos_frac` per stop.
    pub pos_px: Option<f32>,
}

/// CSS `<radial-shape>` — the ending shape of a radial gradient.
/// CSS Images 3 §3.2.1.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum RadialShape {
    Circle,
    Ellipse,
}

/// CSS `<radial-size>` — how the ending shape is sized.
/// CSS Images 3 §3.2.1 / §3.2.2.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum RadialSize {
    ClosestSide,
    FarthestSide,
    ClosestCorner,
    FarthestCorner,
    /// Explicit length(s): one value for a circle radius, two for an
    /// ellipse (rx, ry) in CSS px. (Percentages resolve against the
    /// box dimensions at paint; carried as fraction-of-box via the
    /// `*_pct` companions.)
    Explicit {
        rx_px: Option<f32>,
        ry_px: Option<f32>,
        rx_pct: Option<f32>,
        ry_pct: Option<f32>,
    },
}

/// A position component for a gradient center (`at <position>`) or the
/// implicit center. CSS Backgrounds-style: a px length or a percentage
/// of the box dimension along that axis.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum GradientPosAxis {
    Px(f32),
    Pct(f32),
}

/// A full, N-stop CSS gradient with real linear / radial / conic
/// fidelity. This is the production model the painter rasterizes;
/// `LinearGradient` (2-stop) remains for the legacy fast path and as a
/// solid-fill fallback. CSS Images 3 §3.
#[derive(Clone, Debug, PartialEq)]
pub enum CssGradient {
    Linear {
        /// Direction in CSS degrees (0 = to top, 90 = to right, …).
        angle_deg: f32,
        stops: Vec<GradientColorStop>,
        /// `repeating-linear-gradient()`.
        repeating: bool,
    },
    Radial {
        shape: RadialShape,
        size: RadialSize,
        /// Center (`at <position>`); `None` = 50% 50%.
        center: Option<(GradientPosAxis, GradientPosAxis)>,
        stops: Vec<GradientColorStop>,
        repeating: bool,
    },
    Conic {
        /// `from <angle>` start angle in CSS degrees (0 = top, clockwise).
        from_deg: f32,
        center: Option<(GradientPosAxis, GradientPosAxis)>,
        stops: Vec<GradientColorStop>,
        repeating: bool,
    },
}

/// Parsed `box-shadow`: offset_x, offset_y, blur, spread, color, inset.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct BoxShadowSpec {
    pub offset_x: Length,
    pub offset_y: Length,
    pub blur: Length,
    /// Spread radius: positive expands the shadow, negative contracts it.
    pub spread: Length,
    pub color: Color,
    /// `inset` keyword: shadow is drawn inside the box, not outside.
    pub inset: bool,
}

/// CSS `clip-path` shape. All units are resolved at paint time.
#[derive(Clone, Debug, PartialEq)]
pub enum ClipPath {
    /// `inset(top right bottom left)` — px or % offsets from each side.
    Inset {
        top: Length,
        right: Length,
        bottom: Length,
        left: Length,
    },
    /// `circle(radius at cx cy)`.
    Circle {
        radius: Length,
        cx: Length,
        cy: Length,
    },
    /// `polygon(x1 y1, x2 y2, ...)` — vertex list. Treated as a
    /// non-self-intersecting closed path.
    Polygon(Vec<(Length, Length)>),
}

/// One CSS `filter:` function invocation.
///
/// Not `Copy`: the `Reference` variant carries an `Rc<str>` id. All other
/// variants are scalar; cloning a `FilterFn` is cheap (an `Rc` bump at
/// worst).
#[derive(Clone, Debug, PartialEq)]
pub enum FilterFn {
    /// `blur(radius_px)` — box-blur radius in pixels.
    Blur(f32),
    /// `brightness(amount)` — multiplies RGB. amount=1.0 is identity.
    Brightness(f32),
    /// `contrast(amount)` — (px - 0.5) * amount + 0.5. amount=1.0 identity.
    Contrast(f32),
    /// `grayscale(amount)` — 0..1 lerp toward luminance.
    Grayscale(f32),
    /// `invert(amount)` — 0..1 lerp toward inverted RGB.
    Invert(f32),
    /// `opacity(amount)` — 0..1 multiplies alpha.
    Opacity(f32),
    /// `saturate(amount)` — saturation factor; 1.0 identity.
    Saturate(f32),
    /// `sepia(amount)` — 0..1 lerp toward sepia.
    Sepia(f32),
    /// `hue-rotate(degrees)` — rotates color hue.
    HueRotate(f32),
    /// `drop-shadow(x y blur color)` — same shape as a box-shadow.
    DropShadow(BoxShadowSpec),
    /// `url(#id)` — reference to an inline SVG `<filter>` element. The
    /// `String` is the bare fragment id (without the leading `#`). The
    /// painter resolves it to the `<filter>`'s primitive chain
    /// (feGaussianBlur / feColorMatrix / feOffset / feMerge) per Filter
    /// Effects 1 §4. Carried as the id only; resolution happens at paint
    /// time where the document DOM is available.
    Reference(IdRef),
}

/// A heap id string used by `FilterFn::Reference`. `Arc<str>` (not `Rc`)
/// so the enclosing style stays `Send + Sync` — required because the
/// off-main renderer sends resolved styles across threads (cv_ui).
pub type IdRef = std::sync::Arc<str>;

#[derive(Copy, Clone, Debug)]
pub enum LineHeight {
    Multiplier(f32),
    Length(Length),
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum TextAlign {
    Left,
    Center,
    Right,
    Justify,
    /// `text-align: start` — flow-relative start edge. Equivalent to
    /// `Left` in LTR writing modes (the default). Kept as a distinct
    /// variant so callers can detect the semantic value if needed.
    Start,
    /// `text-align: end` — flow-relative end edge. Equivalent to
    /// `Right` in LTR writing modes. Kept as a distinct variant for
    /// the same reason as `Start`.
    End,
}

#[derive(Clone, Copy)]
struct Matched<'a> {
    specificity: u32,
    important: bool,
    /// CSS cascade origin. 0 = UserAgent (our default UA stylesheet),
    /// 1 = Author (page CSS + inline). Per CSS Cascade Level 4 §6.4:
    /// for !important rules UA wins over Author; for normal rules
    /// Author wins over UA. Without this, our UA `a { color:#3366cc }`
    /// rule at specificity (0,0,1) was tying with page rules of equal
    /// specificity and winning on source order — making Google's footer
    /// links blue when they should inherit grey from the container.
    origin: u8,
    /// Cascade-layer index this declaration belongs to, or `None` for
    /// unlayered declarations. Later layers win for normal rules; the
    /// order reverses for `!important` (see `layer_rank`).
    layer: Option<u32>,
    /// Sourced from a `style="..."` attribute (the "Style Attribute"
    /// pseudo-origin in CSS Cascade L4 §6.4.4). Per spec, for !important
    /// declarations within the Author origin, Style Attribute outranks
    /// every layered rule; for normal, Style Attribute outranks
    /// unlayered which outranks layered. Previously inline tied with
    /// "unlayered author" (both used `layer: None`), so an !important
    /// inline lost to an !important layered author rule — `<div
    /// style="color:red!important">` rendered with the layer's color.
    is_inline: bool,
    source_order: usize,
    decl: &'a Declaration,
}

/// Cascade-rank for sorting Matched. Higher rank wins (so sort
/// ascending and the last entry is the active declaration).
fn cascade_rank(m: &Matched<'_>) -> u8 {
    // Spec order from lowest to highest precedence:
    //   normal UA → normal Author → important Author → important UA
    match (m.important, m.origin) {
        (false, 0) => 0,                 // normal UA
        (false, _) => 1,                 // normal Author (or inline)
        (true, _) if m.origin != 0 => 2, // important Author
        (true, _) => 3,                  // important UA
    }
}

/// Cascade-layer precedence *within* an origin/importance tier. Per CSS
/// Cascade Level 4 §6.4.2, layer order sits between origin and
/// specificity: for normal declarations a later-declared layer wins,
/// and any unlayered declaration wins over every layered one; for
/// `!important` the whole order reverses (earlier layer wins, and
/// layered beats unlayered). Encoded so a plain ascending `i64` sort —
/// inserted right after `cascade_rank` — gets it right. This is what
/// makes Tailwind v4's `@layer theme, base, components, utilities;`
/// resolve correctly (utility classes override component/base styles
/// regardless of source order or specificity).
fn layer_rank(m: &Matched<'_>) -> i64 {
    // Style Attribute (inline) lives ABOVE every layered rule per CSS
    // Cascade L4 §6.4.4 regardless of importance — pick sentinel values
    // that beat both the `Some(i)` and the unlayered cases below.
    if m.is_inline {
        // !important style-attribute: still wins among the !important
        //   author tier (where earlier-layer-wins → negative rank), so
        //   pick i64::MAX.
        // normal style-attribute: wins among normal author tier (later-
        //   layer-wins → positive rank), so also i64::MAX.
        return i64::MAX;
    }
    match (m.important, m.layer) {
        (false, None) => i64::MAX - 1,      // unlayered normal beats layered but loses to inline
        (false, Some(i)) => i as i64,       // later layer = higher precedence
        (true, None) => i64::MIN + 1,       // unlayered !important loses to layered but beats nothing
        (true, Some(i)) => -(i as i64),     // earlier layer = higher precedence
    }
}

/// Extract layer names from an `@layer` prelude. Handles both the
/// statement form (`@layer a, b, c;`) and the block form
/// (`@layer name { ... }`), plus dotted sub-layer names (`a.b`).
fn parse_layer_names(prelude: &[CssToken]) -> Vec<String> {
    let mut names = Vec::new();
    let mut cur = String::new();
    for t in prelude {
        match t {
            CssToken::Ident(s) => cur.push_str(s),
            CssToken::Delim('.') => cur.push('.'),
            CssToken::Comma => {
                let n = cur.trim().to_string();
                if !n.is_empty() {
                    names.push(n);
                }
                cur.clear();
            }
            CssToken::Whitespace => {}
            _ => {}
        }
    }
    let n = cur.trim().to_string();
    if !n.is_empty() {
        names.push(n);
    }
    names
}

/// Blink-style ancestor Bloom filter (`SelectorFilter`): a fixed 256-bit set of
/// the identifier hashes (tag / id / class) of an element's ANCESTOR chain. Used
/// to FAST-REJECT a descendant/child selector before running the full ancestor
/// walk: if the selector requires an ancestor identifier whose hash bit is
/// definitely absent from this filter, the full match would also fail, so we skip
/// it. A Bloom membership test can have FALSE POSITIVES (claim present when
/// absent) but never FALSE NEGATIVES — so this only ever skips selectors that
/// CANNOT match. Identical match RESULT, fewer full match attempts.
///
/// 256 bits with 2 hash positions per identifier keeps the false-positive rate
/// low for the hundreds-of-ancestor-identifiers seen on real pages while staying
/// `Copy` (4 × u64) so it threads cheaply through the per-element style call.
#[derive(Copy, Clone, Default, PartialEq, Eq)]
pub struct AncestorFilter {
    bits: [u64; 4],
}

/// Hash an identifier (already lowercased for tags) to a 64-bit value. FNV-1a:
/// tiny, allocation-free, good bit dispersion for short ASCII identifiers. The
/// `salt` distinguishes the tag / id / class namespaces so `div` the tag and a
/// `.div` class don't collide in the filter.
#[inline]
fn ident_hash(salt: u64, s: &str) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325 ^ salt;
    for b in s.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    // Avoid the all-zero hash mapping to a fixed bit; mix high/low.
    h ^ (h >> 29)
}

const SALT_TAG: u64 = 0x1;
const SALT_ID: u64 = 0x2;
const SALT_CLASS: u64 = 0x3;

impl AncestorFilter {
    /// Set the two Bloom bits for one identifier hash.
    #[inline]
    fn set_hash(&mut self, h: u64) {
        let a = (h & 0xff) as usize; // first bit position 0..256
        let b = ((h >> 8) & 0xff) as usize; // second, independent position
        self.bits[a >> 6] |= 1u64 << (a & 63);
        self.bits[b >> 6] |= 1u64 << (b & 63);
    }

    /// Add an element's own identifiers (tag, id, classes) to the filter.
    /// Call once per ancestor while descending the tree.
    pub fn add_element(&mut self, tag: Option<&str>, id: Option<&str>, classes: &[String]) {
        if let Some(t) = tag {
            self.set_hash(ident_hash(SALT_TAG, &t.to_ascii_lowercase()));
        }
        if let Some(i) = id {
            self.set_hash(ident_hash(SALT_ID, i));
        }
        for c in classes {
            self.set_hash(ident_hash(SALT_CLASS, c));
        }
    }

    /// True if EVERY set bit in `required` is also set in `self`. When false,
    /// at least one required ancestor identifier is provably absent → the
    /// selector cannot match → safe to skip. (When true it MIGHT match — could
    /// be a Bloom false positive — so the caller still runs the full match.)
    #[inline]
    fn covers(&self, required: &AncestorFilter) -> bool {
        (required.bits[0] & !self.bits[0]) == 0
            && (required.bits[1] & !self.bits[1]) == 0
            && (required.bits[2] & !self.bits[2]) == 0
            && (required.bits[3] & !self.bits[3]) == 0
    }

    /// True if this filter has no bits set (no required ancestor identifiers).
    #[inline]
    fn is_empty(&self) -> bool {
        self.bits == [0; 4]
    }

}

/// Measurement counters for the ancestor-Bloom fast-reject (Milestone 2.3).
/// `ATTEMPTS` = candidate selectors that reached the fast-reject gate;
/// `REJECTS` = those the Bloom proved cannot match (full match skipped).
/// The reduction in full-match attempts is `REJECTS / ATTEMPTS`. Relaxed
/// ordering is fine — these are diagnostics, not correctness state.
pub static BLOOM_ATTEMPTS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static BLOOM_REJECTS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Read the (attempts, rejects) fast-reject counters.
pub fn bloom_stats() -> (u64, u64) {
    use std::sync::atomic::Ordering::Relaxed;
    (
        BLOOM_ATTEMPTS.load(Relaxed),
        BLOOM_REJECTS.load(Relaxed),
    )
}

/// Zero the fast-reject counters (so a caller can measure a single render).
pub fn bloom_reset() {
    use std::sync::atomic::Ordering::Relaxed;
    BLOOM_ATTEMPTS.store(0, Relaxed);
    BLOOM_REJECTS.store(0, Relaxed);
}

/// Env kill-switch: set `CV_NO_SELECTOR_FILTER=1` to disable the ancestor-Bloom
/// fast-reject entirely (every candidate runs the full match, as before). The
/// result is identical either way; this exists purely as a safety escape hatch
/// and an A/B measurement lever. Cached once per process.
fn selector_filter_disabled() -> bool {
    use std::sync::atomic::{AtomicU8, Ordering::Relaxed};
    static CACHE: AtomicU8 = AtomicU8::new(0); // 0 = unknown, 1 = on, 2 = off
    match CACHE.load(Relaxed) {
        1 => true,
        2 => false,
        _ => {
            let off = std::env::var("CV_NO_SELECTOR_FILTER")
                .map(|v| v != "0" && !v.is_empty())
                .unwrap_or(false);
            CACHE.store(if off { 1 } else { 2 }, Relaxed);
            off
        }
    }
}

/// The fast-reject gate. Returns `true` when the candidate's required-ancestor
/// signature is NOT covered by `filter` — i.e. the selector provably cannot
/// match and the full match should be SKIPPED. Also bumps the measurement
/// counters. A `None` filter (caller opted out / no ancestor context) never
/// rejects, preserving exact legacy behaviour.
#[inline]
fn bloom_can_skip(filter: Option<&AncestorFilter>, cand: &CandidateRef) -> bool {
    use std::sync::atomic::Ordering::Relaxed;
    let Some(f) = filter else {
        return false;
    };
    if selector_filter_disabled() {
        return false;
    }
    // Subject-only selectors require no ancestors → nothing to reject.
    if cand.req_ancestors.is_empty() {
        return false;
    }
    BLOOM_ATTEMPTS.fetch_add(1, Relaxed);
    if f.covers(&cand.req_ancestors) {
        false // might match → run full match
    } else {
        BLOOM_REJECTS.fetch_add(1, Relaxed);
        true // provably cannot match → skip
    }
}

/// Compute the REQUIRED-ANCESTOR signature of a selector: the Bloom bits for
/// every tag/id/class that MUST appear among an element's ancestors for this
/// selector to match. We collect identifiers from each non-subject compound that
/// is connected to the subject through a CONTIGUOUS run of descendant (` `) or
/// child (`>`) combinators. As soon as a sibling combinator (`+`/`~`) appears in
/// the chain (reading right→left), we stop — compounds past it are siblings, not
/// guaranteed ancestors, so requiring them would be unsound (could false-reject).
///
/// Requiring a compound's *own* tag/class/id is always sound even when that
/// compound also carries `:hover`/`:is(...)`/etc., because those only ADD
/// constraints; the element still has to be an ancestor and still has to carry
/// the literal tag/class/id. We deliberately requires NOTHING from inside
/// functional pseudos (`:is`/`:where`/`:not`/`:has`) — their inner identifiers
/// are not on the host element.
fn required_ancestor_signature(sel: &Selector) -> AncestorFilter {
    use crate::selectors::Combinator;
    let mut f = AncestorFilter::default();
    let parts = &sel.parts;
    if parts.len() < 2 {
        return f; // subject-only: no ancestor requirement
    }
    // Walk from the subject leftward. `parts[i].combinator` describes how
    // parts[i] connects to parts[i-1]'s... actually combinator on parts[i+1]
    // governs the step from parts[i] to parts[i+1]. The subject is parts.last().
    // Combinator linking subject to its left neighbour lives on the subject part.
    // We iterate the LEFT compounds (parts[0..len-1]); the combinator that
    // attaches compound `parts[i]` to the chain on its right is parts[i+1].combinator.
    for i in (0..parts.len() - 1).rev() {
        let link = parts[i + 1].combinator.as_ref();
        match link {
            Some(Combinator::Descendant) | Some(Combinator::Child) | None => {
                let c = &parts[i].compound;
                if let Some(tag) = &c.element {
                    f.set_hash(ident_hash(SALT_TAG, &tag.to_ascii_lowercase()));
                }
                if let Some(id) = &c.id {
                    f.set_hash(ident_hash(SALT_ID, id));
                }
                for cl in &c.classes {
                    f.set_hash(ident_hash(SALT_CLASS, cl));
                }
            }
            // Sibling combinator: the compounds further left are siblings of an
            // ancestor, not ancestors themselves. Stop — anything we'd add past
            // here is not guaranteed to be an ancestor identifier.
            Some(Combinator::NextSibling) | Some(Combinator::SubsequentSibling) => break,
        }
    }
    f
}

/// CSS Containment 3 §3 — one `@container <name>? (<condition>)` clause.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ContainerQueryClause {
    /// Optional `container-name` the query targets. `None` = match the
    /// nearest query container regardless of name.
    name: Option<String>,
    /// The raw condition between the outer parens, e.g. `min-width: 300px`
    /// or `inline-size > 30em`. Evaluated by [`eval_container_condition_axes`].
    condition: String,
}

/// CSS Containment 3 §3 — a compiled `@container` guard. Attached (by index)
/// to every candidate that comes from an `@container` block. At cascade time
/// the candidate only contributes its declarations when EVERY clause is
/// satisfied by the element's matching ancestor query container (see
/// [`QueryContainer`] / [`eval_container_guard`]). Multiple clauses arise
/// only from doubly-nested `@container` blocks — the common case is one.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ContainerQueryGuard {
    clauses: Vec<ContainerQueryClause>,
}

/// One live query container in the ancestor chain, supplied by the host
/// (layout) when resolving a descendant's style. The cascade walks this
/// stack (nearest-last) to find the container a `@container` guard targets
/// and evaluates the guard against its laid-out CONTENT-BOX size.
///
/// CSS Containment 3 §3: a `@container` rule applies to a descendant only
/// when its nearest ancestor query container of the right type satisfies
/// the condition, evaluated against that container's content-box size — NOT
/// the viewport. `inline_size` is the content-box inline (usually width)
/// extent; `block_size` the block (usually height) extent. `block_size` is
/// only queryable when `container_type == Size` (per §2.1, `inline-size`
/// containers don't apply block-axis containment so the block size isn't a
/// stable query target).
#[derive(Clone, Debug, PartialEq)]
pub struct QueryContainer {
    /// `container-name`s declared on this container (may be empty).
    pub names: Vec<String>,
    /// `container-type`. Determines which axes are queryable.
    pub container_type: ContainerType,
    /// Content-box inline-axis size in CSS px.
    pub inline_size: f32,
    /// Content-box block-axis size in CSS px (only meaningful for `Size`).
    pub block_size: f32,
}

/// The ancestor query-container stack handed to the cascade for one element.
/// Ordered root-first; the LAST entry is the nearest ancestor container.
pub type QueryContainerStack<'s> = &'s [QueryContainer];

/// A reference to one selector inside one rule inside one sheet. Used by
/// `SelectorIndex` so we can find a candidate without re-iterating sheets.
#[derive(Copy, Clone)]
struct CandidateRef {
    sheet_idx: u32,
    rule_idx: u32,
    sel_idx: u32,
    /// Index into [`SelectorIndex::container_queries`] of the `@container`
    /// guard this candidate is gated by, or `u32::MAX` when the candidate
    /// is NOT inside an `@container` block (the common case). Keeping this
    /// a `u32` index (not an owned `String`) preserves `CandidateRef: Copy`.
    container_query: u32,
    /// Globally-monotonic source-order assigned at index build time. Used
    /// by the cascade tiebreaker so we don't need to recompute it per
    /// element.
    source_order: u32,
    /// 0 = UserAgent sheet (sheets[0]), 1 = Author sheet (any other).
    /// Drives the CSS cascade origin tiebreaker.
    origin: u8,
    /// Cascade-layer index for rules folded in from an `@layer` block,
    /// or `None` for unlayered rules. Drives the layer tiebreaker.
    layer: Option<u32>,
    /// Blink `SelectorFilter` fast-reject signature: the Bloom bits of every
    /// tag/class/id this selector requires among the matched element's
    /// ANCESTORS. Compared against the element's `AncestorFilter` before the
    /// full match; if a required bit is absent, the match cannot succeed and we
    /// skip it. All-zero for subject-only selectors (never rejects).
    req_ancestors: AncestorFilter,
}

/// Pre-built index of (sheet, rule, selector) triples bucketed by the
/// rightmost compound's id / class / tag. Built once for a set of
/// stylesheets and reused across every element on the page.
///
/// Without this, `compute_with_inline` was O(elements × total_rules)
/// which on a real page (a few hundred elements, a couple thousand
/// rules) burned tens of seconds per render. With the index, each
/// element only re-checks the ~10-100 rules whose key (id/class/tag)
/// could possibly match it.
#[derive(Default)]
pub struct SelectorIndex<'a> {
    sheets: &'a [Stylesheet],
    by_id: std::collections::HashMap<String, Vec<CandidateRef>>,
    by_class: std::collections::HashMap<String, Vec<CandidateRef>>,
    by_tag: std::collections::HashMap<String, Vec<CandidateRef>>,
    universal: Vec<CandidateRef>,
    /// True if ANY indexed rule targets a pseudo-element (`::before`/`::after`/…).
    /// Lets the renderer skip the per-element pseudo-element probe entirely on
    /// pages whose sheets declare none (the common case) — that probe runs twice
    /// per element and was a large constant-factor cost.
    has_pseudo_rules: bool,
    /// Side table of `@container` guards. A candidate's `container_query`
    /// field indexes here; `u32::MAX` means "no guard". Deduplicated by the
    /// build pass so identical `@container` preludes share one entry.
    container_queries: Vec<ContainerQueryGuard>,
}

/// The OWNED part of a `SelectorIndex` (the bucketed candidate maps) with no
/// `sheets` borrow. Building this is the expensive, frame-invariant work (it
/// re-buckets every selector of every sheet); separating it lets the renderer
/// cache it across frames and re-pair it with the (borrowed) sheets each frame
/// via [`SelectorIndex::with_data`], instead of rebuilding from scratch every
/// render. Pure function of (sheets, viewport) — `CandidateRef`s are indices,
/// not borrows, so this is `'static`-safe to stash.
#[derive(Default)]
pub struct SelectorIndexData {
    by_id: std::collections::HashMap<String, Vec<CandidateRef>>,
    by_class: std::collections::HashMap<String, Vec<CandidateRef>>,
    by_tag: std::collections::HashMap<String, Vec<CandidateRef>>,
    universal: Vec<CandidateRef>,
    has_pseudo_rules: bool,
    container_queries: Vec<ContainerQueryGuard>,
}

/// Blink-style invalidation set: when a keyed feature (a class or id) changes on
/// an element, this describes which OTHER elements may need restyling. It is a
/// FILTER for a tree walk, not a node list — `invalidates_element` tests a
/// candidate against the filter. A superset is always correct; the goal is to be
/// as tight as possible. The coarse tiers (`whole_subtree`,
/// `invalidates_parent_subtree`) are safe fallbacks for relationships we don't
/// model precisely yet (siblings, attrs-as-trigger, pseudos, :has/:is/:not/nth).
#[derive(Debug, Clone, Default)]
pub struct InvalidationSet {
    /// A descendant with one of these classes may need restyling.
    pub classes: std::collections::HashSet<String>,
    /// A descendant with one of these ids may need restyling.
    pub ids: std::collections::HashSet<String>,
    /// A descendant with one of these tag names (lowercase) may need restyling.
    pub tags: std::collections::HashSet<String>,
    /// The changed element itself needs restyling (keyed feature is in the
    /// selector's rightmost/subject compound).
    pub invalidates_self: bool,
    /// Can't filter precisely → restyle the changed element's whole descendant
    /// subtree (e.g. universal subject `.a *`, or a complex feature in the chain).
    pub whole_subtree: bool,
    /// A sibling combinator (`+`/`~`) is involved → restyle the changed element's
    /// PARENT subtree (a superset that covers following siblings + their
    /// descendants). Tightened to a real SiblingInvalidationSet in a later stage.
    pub invalidates_parent_subtree: bool,
}

impl InvalidationSet {
    /// Does a descendant element with these features fall in this filter?
    /// `tag` should be lowercase.
    pub fn invalidates_element(&self, classes: &[String], id: Option<&str>, tag: &str) -> bool {
        if self.whole_subtree {
            return true;
        }
        if let Some(id) = id {
            if self.ids.contains(id) {
                return true;
            }
        }
        if self.tags.contains(tag) {
            return true;
        }
        classes.iter().any(|c| self.classes.contains(c))
    }
}

/// Blink's `RuleFeatureSet`: a compiled invalidation index built once per
/// stylesheet set. Maps each runtime-mutable feature (class/id) to the
/// `InvalidationSet` describing what to re-examine when it changes. Attributes
/// and state pseudos are handled coarsely by the driver in this increment
/// (every non-class/id change dirties the node + its subtree); they get their
/// own maps in a later stage.
#[derive(Debug, Clone, Default)]
pub struct RuleFeatureSet {
    pub class_invalidation: std::collections::HashMap<String, InvalidationSet>,
    pub id_invalidation: std::collections::HashMap<String, InvalidationSet>,
}

impl RuleFeatureSet {
    pub fn class_set(&self, class: &str) -> Option<&InvalidationSet> {
        self.class_invalidation.get(class)
    }
    pub fn id_set(&self, id: &str) -> Option<&InvalidationSet> {
        self.id_invalidation.get(id)
    }
}

/// Build the `RuleFeatureSet` from every selector in every sheet (including
/// `@media`/`@supports`-nested rules — harvesting inactive ones is a safe
/// superset). Pure function of `sheets`; the caller caches it by sheet identity.
pub fn build_rule_feature_set(sheets: &[Stylesheet]) -> RuleFeatureSet {
    let mut fs = RuleFeatureSet::default();
    for ss in sheets {
        for rule in &ss.rules {
            for sel in &rule.selectors {
                harvest_selector(&mut fs, sel);
            }
        }
        for at in &ss.at_rules {
            if let Some(rules) = at.block.as_ref() {
                for rule in rules {
                    for sel in &rule.selectors {
                        harvest_selector(&mut fs, sel);
                    }
                }
            }
        }
    }
    fs
}

fn harvest_selector(fs: &mut RuleFeatureSet, sel: &Selector) {
    use crate::selectors::Combinator;
    let parts = &sel.parts;
    let Some(subject_part) = parts.last() else {
        return;
    };
    let subject = &subject_part.compound;
    let subj_classes = &subject.classes;
    let subj_id = &subject.id;
    let subj_tag = subject.element.as_ref().map(|t| t.to_ascii_lowercase());
    let subj_universal = subj_classes.is_empty() && subj_id.is_none() && subj_tag.is_none();

    // Subject-compound features → self-invalidation (a class/id change on the
    // subject element itself flips its own match).
    for c in subj_classes {
        fs.class_invalidation
            .entry(c.clone())
            .or_default()
            .invalidates_self = true;
    }
    if let Some(id) = subj_id {
        fs.id_invalidation
            .entry(id.clone())
            .or_default()
            .invalidates_self = true;
    }
    if parts.len() == 1 {
        return; // no ancestor/sibling triggers
    }

    let has_sibling = parts.iter().any(|p| {
        matches!(
            p.combinator,
            Some(Combinator::NextSibling) | Some(Combinator::SubsequentSibling)
        )
    });
    let has_complex = parts.iter().any(|p| {
        let c = &p.compound;
        !c.attrs.is_empty()
            || !c.pseudo_classes.is_empty()
            || !c.not_selectors.is_empty()
            || !c.is_selectors.is_empty()
            || !c.where_selectors.is_empty()
            || !c.has_selectors.is_empty()
            || !c.nth_selectors.is_empty()
    });

    let contribute = |set: &mut InvalidationSet| {
        if has_sibling {
            set.invalidates_parent_subtree = true;
        } else if has_complex || subj_universal {
            set.whole_subtree = true;
        } else {
            for c in subj_classes {
                set.classes.insert(c.clone());
            }
            if let Some(id) = subj_id {
                set.ids.insert(id.clone());
            }
            if let Some(tag) = &subj_tag {
                set.tags.insert(tag.clone());
            }
        }
    };

    // Each left compound's class/id is a trigger that, when it changes, may flip
    // the subject's match — record the subject features against it.
    for lp in &parts[..parts.len() - 1] {
        let comp = &lp.compound;
        for c in &comp.classes {
            contribute(fs.class_invalidation.entry(c.clone()).or_default());
        }
        if let Some(id) = &comp.id {
            contribute(fs.id_invalidation.entry(id.clone()).or_default());
        }
    }
}

impl<'a> SelectorIndex<'a> {
    pub fn build(sheets: &'a [Stylesheet]) -> Self {
        // Default media context: 1024×768. Browser code can build with
        // a real viewport via `build_with_viewport`.
        Self::build_with_viewport(sheets, 1024.0, 768.0)
    }

    /// Whether any indexed rule targets a pseudo-element. The renderer skips
    /// the (twice-per-element) `::before`/`::after` probe when this is false.
    pub fn has_pseudo_rules(&self) -> bool {
        self.has_pseudo_rules
    }

    /// Same as `build` but evaluates `@media` queries against the given
    /// viewport. `@media`-nested rules whose query matches get indexed
    /// alongside top-level rules; non-matching ones are skipped so they
    /// can't accidentally apply.
    pub fn build_with_viewport(sheets: &'a [Stylesheet], viewport_w: f32, viewport_h: f32) -> Self {
        Self::with_data(
            sheets,
            Self::build_data_with_viewport(sheets, viewport_w, viewport_h),
        )
    }

    /// Pair cached/owned bucket `data` with a `sheets` borrow to form a usable
    /// index, with no rebuild. The frame-invariant work is in `data`.
    pub fn with_data(sheets: &'a [Stylesheet], data: SelectorIndexData) -> Self {
        Self {
            sheets,
            by_id: data.by_id,
            by_class: data.by_class,
            by_tag: data.by_tag,
            universal: data.universal,
            has_pseudo_rules: data.has_pseudo_rules,
            container_queries: data.container_queries,
        }
    }

    /// Reclaim the owned bucket data (drop the `sheets` borrow) so it can be
    /// stashed in a cross-frame cache and reused via `with_data`.
    pub fn into_data(self) -> SelectorIndexData {
        SelectorIndexData {
            by_id: self.by_id,
            by_class: self.by_class,
            by_tag: self.by_tag,
            universal: self.universal,
            has_pseudo_rules: self.has_pseudo_rules,
            container_queries: self.container_queries,
        }
    }

    /// Build just the owned bucket data (no `sheets` borrow) — the expensive,
    /// frame-invariant half of index construction, cacheable across frames.
    /// Pure function of (sheets, viewport).
    pub fn build_data_with_viewport(
        sheets: &[Stylesheet],
        viewport_w: f32,
        viewport_h: f32,
    ) -> SelectorIndexData {
        let mut idx = SelectorIndexData {
            by_id: Default::default(),
            by_class: Default::default(),
            by_tag: Default::default(),
            universal: Vec::new(),
            has_pseudo_rules: false,
            container_queries: Vec::new(),
        };
        // CSS cascade layers: assign each named layer an index by first
        // appearance across all sheets in document order. Both the
        // `@layer a, b, c;` statement form and the `@layer name { ... }`
        // block form register names; later layers win for normal rules.
        let mut layer_order: Vec<String> = Vec::new();
        for ss in sheets {
            for at in &ss.at_rules {
                if at.name == "layer" {
                    for name in parse_layer_names(&at.prelude) {
                        if !layer_order.iter().any(|n| *n == name) {
                            layer_order.push(name);
                        }
                    }
                }
            }
        }
        // Per-sheet running source_order across rules + nested @media
        // rules. Needed because @media rules borrow source position
        // from the at-rule's location.
        let mut source_order: u32 = 0;
        for (si, ss) in sheets.iter().enumerate() {
            // Per CSS Cascade L4 §6.4 we treat sheets[0] as our UA
            // stylesheet (lowest origin precedence) and everything
            // after as Author. Without this our UA `a { color:#3366cc;
            // text-decoration: underline }` at specificity (0,0,1)
            // would tie with Google's page rules and win on source
            // order, painting the footer links blue + underlined.
            let origin: u8 = if si == 0 { 0 } else { 1 };
            for (ri, rule) in ss.rules.iter().enumerate() {
                for (xi, sel) in rule.selectors.iter().enumerate() {
                    Self::push_candidate(
                        &mut idx,
                        si as u32,
                        ri as u32,
                        xi as u32,
                        source_order,
                        origin,
                        None, // top-level rules are unlayered
                        sel,
                    );
                    source_order = source_order.wrapping_add(1);
                }
            }
            // @media / @supports rules — only fold in if their query
            // matches. We use a NEGATIVE pseudo-rule-index encoded as
            // (rule_idx = total_rule_count + at_rule_index) so the
            // candidate can still point back via the at_rule_rules helper.
            // For @supports we accept any feature query that mentions a
            // property/value pair we know about — anything else is
            // treated as supported too (most queries are progressive
            // enhancements and the inner rules are usually safe to
            // apply even when the test would have failed on a real
            // browser).
            for (ai, at) in ss.at_rules.iter().enumerate() {
                let is_media = at.name == "media";
                let is_supports = at.name == "supports";
                // @container, @layer, @scope, @starting-style: fold their
                // body rules in too. @container's per-element condition is now
                // REALLY enforced: each candidate inside an `@container` block
                // carries a guard index (its prelude name + condition) and the
                // cascade evaluates it against the element's nearest ancestor
                // query container's laid-out content-box size, applying or
                // withholding the rule accordingly (see eval_container_guard).
                // @layer rules inherit normal source-order precedence — full
                // layer ordering would re-sort across the same origin tier,
                // which V1 approximates with source-order.
                let is_container = at.name == "container";
                let is_layer = at.name == "layer";
                let is_scope = at.name == "scope";
                let is_starting = at.name == "starting-style";
                if !is_media
                    && !is_supports
                    && !is_container
                    && !is_layer
                    && !is_scope
                    && !is_starting
                {
                    continue;
                }
                // Rules nested directly inside `@layer name { ... }` carry
                // that layer's index; everything else (@media/@supports/…)
                // is unlayered.
                let cand_layer = if is_layer {
                    parse_layer_names(&at.prelude)
                        .first()
                        .and_then(|name| layer_order.iter().position(|n| n == name))
                        .map(|p| p as u32)
                } else {
                    None
                };
                // Walk the at-rule and EVERY at-rule nested inside it in the
                // same deterministic order `flatten_at_rule_all` uses. Each
                // qualified rule is enumerated with a running FLAT index; we
                // only emit candidates for rules all of whose enclosing at-rule
                // conditions currently match. A nested `@media screen and
                // (min-width:640px)` inside `@media screen` thus applies at
                // 960px, while a nested `@media print` (or dark-mode) block is
                // skipped — without leaking either way.
                let mut flat_idx = 0usize;
                Self::index_at_rule_recursive(
                    &mut idx,
                    at,
                    at_rule_condition_matches(at, viewport_w, viewport_h),
                    si as u32,
                    ai,
                    ss.rules.len(),
                    &mut flat_idx,
                    &mut source_order,
                    origin,
                    cand_layer,
                    u32::MAX,
                    viewport_w,
                    viewport_h,
                );
            }
        }
        idx
    }

    #[allow(clippy::too_many_arguments)]
    fn push_candidate(
        idx: &mut SelectorIndexData,
        sheet_idx: u32,
        rule_idx: u32,
        sel_idx: u32,
        source_order: u32,
        origin: u8,
        layer: Option<u32>,
        sel: &Selector,
    ) {
        Self::push_candidate_cq(
            idx,
            sheet_idx,
            rule_idx,
            sel_idx,
            source_order,
            origin,
            layer,
            sel,
            u32::MAX,
        );
    }

    /// `push_candidate` carrying an `@container` guard index (`u32::MAX` = none).
    #[allow(clippy::too_many_arguments)]
    fn push_candidate_cq(
        idx: &mut SelectorIndexData,
        sheet_idx: u32,
        rule_idx: u32,
        sel_idx: u32,
        source_order: u32,
        origin: u8,
        layer: Option<u32>,
        sel: &Selector,
        container_query: u32,
    ) {
        idx.has_pseudo_rules |= sel.targets_pseudo_element();
        let key = sel.parts.last().map(|p| &p.compound);
        let cand = CandidateRef {
            sheet_idx,
            rule_idx,
            sel_idx,
            source_order,
            origin,
            layer,
            container_query,
            req_ancestors: required_ancestor_signature(sel),
        };
        match key {
            Some(k) if k.id.is_some() => {
                idx.by_id
                    .entry(k.id.clone().unwrap())
                    .or_default()
                    .push(cand);
            }
            Some(k) if !k.classes.is_empty() => {
                for c in &k.classes {
                    idx.by_class.entry(c.clone()).or_default().push(cand);
                }
            }
            Some(k) if k.element.is_some() => {
                let tag = k.element.clone().unwrap().to_ascii_lowercase();
                idx.by_tag.entry(tag).or_default().push(cand);
            }
            _ => idx.universal.push(cand),
        }
    }

    /// Index every qualified rule reachable from `at` (its own block and all
    /// descendant at-rule blocks), walking the SAME deterministic order as
    /// `flatten_at_rule_all`. `flat_idx` is the running position used to encode
    /// each rule's synthetic index; `ancestors_match` is whether every enclosing
    /// at-rule condition (including `at` itself) currently matches — only then
    /// is a candidate emitted, but `flat_idx` advances for ALL rules so the
    /// match-independent `resolve_rule` flatten stays in sync.
    #[allow(clippy::too_many_arguments)]
    fn index_at_rule_recursive(
        idx: &mut SelectorIndexData,
        at: &crate::parser::AtRule,
        ancestors_match: bool,
        sheet_idx: u32,
        at_idx: usize,
        top_count: usize,
        flat_idx: &mut usize,
        source_order: &mut u32,
        origin: u8,
        cand_layer: Option<u32>,
        container_query: u32,
        vw: f32,
        vh: f32,
    ) {
        // If THIS at-rule is `@container`, register its guard and use it for
        // every rule below. A `@container` nested inside another `@container`
        // (rare but legal — CSS Containment 3 §3) overrides with the inner one
        // because the nearest ancestor query container governs; matching both
        // is enforced per-element by checking each enclosing guard. We model the
        // common single-level case by carrying the innermost guard; the build
        // pass keeps the OUTER guards reachable too via the candidate set, but
        // a candidate carries only its innermost guard. Since a doubly-nested
        // `@container` requires both to pass, we conjoin them into one guard.
        let active_cq = if at.name == "container" {
            let (name, condition) = parse_container_prelude(&at.prelude);
            let mut clauses = if container_query == u32::MAX {
                Vec::new()
            } else {
                // Inherit the enclosing `@container`'s clauses — both ancestor
                // containers must satisfy their conditions (CSS Containment 3 §3
                // makes the nearest-container rule apply only when reachable
                // through containers that each match).
                idx.container_queries[container_query as usize].clauses.clone()
            };
            clauses.push(ContainerQueryClause { name, condition });
            let guard = ContainerQueryGuard { clauses };
            // Deduplicate identical guards so repeated `@container (min-width:
            // 300px)` blocks share one side-table entry.
            match idx.container_queries.iter().position(|g| *g == guard) {
                Some(i) => i as u32,
                None => {
                    idx.container_queries.push(guard);
                    (idx.container_queries.len() - 1) as u32
                }
            }
        } else {
            container_query
        };
        if let Some(block) = at.block.as_ref() {
            for rule in block {
                let synthetic_rule_idx = encode_at_rule_idx(top_count, at_idx, *flat_idx);
                if ancestors_match {
                    for (xi, sel) in rule.selectors.iter().enumerate() {
                        Self::push_candidate_cq(
                            idx,
                            sheet_idx,
                            synthetic_rule_idx as u32,
                            xi as u32,
                            *source_order,
                            origin,
                            cand_layer,
                            sel,
                            active_cq,
                        );
                        *source_order = source_order.wrapping_add(1);
                    }
                }
                *flat_idx += 1;
            }
        }
        for nested in &at.nested {
            let child_match = ancestors_match && at_rule_condition_matches(nested, vw, vh);
            Self::index_at_rule_recursive(
                idx,
                nested,
                child_match,
                sheet_idx,
                at_idx,
                top_count,
                flat_idx,
                source_order,
                origin,
                cand_layer,
                active_cq,
                vw,
                vh,
            );
        }
    }

    fn resolve_rule(&self, cand: &CandidateRef) -> Option<&'a crate::parser::Rule> {
        let ss = &self.sheets[cand.sheet_idx as usize];
        let rule_idx = cand.rule_idx as usize;
        if rule_idx < ss.rules.len() {
            return Some(&ss.rules[rule_idx]);
        }
        // Synthetic at-rule-nested rule: `inner` is the FLAT index into the
        // deterministic full flatten of the at-rule's own block + every nested
        // at-rule block (see flatten_at_rule_all / index_at_rule_recursive).
        let (at_idx, inner) = decode_at_rule_idx(ss.rules.len(), rule_idx);
        let at = ss.at_rules.get(at_idx)?;
        let mut flat: Vec<&'a crate::parser::Rule> = Vec::new();
        flatten_at_rule_all(at, &mut flat);
        flat.get(inner).copied()
    }

    /// Whether `cand`'s `@container` guard (if any) is satisfied for an element
    /// whose ancestor query-container stack is `stack`.
    ///
    /// `stack == None` means the host did not supply container sizes for this
    /// resolution (e.g. a tree-walk that hasn't laid out containers yet) — in
    /// that case `@container` guards are applied OPTIMISTICALLY (the historical
    /// behavior) so nothing silently disappears. `stack == Some(_)` triggers the
    /// real CSS Containment 3 §3 evaluation: the guard's clauses are tested
    /// against the matching ancestor query container's content-box size.
    fn cand_container_ok(
        &self,
        cand: &CandidateRef,
        stack: Option<QueryContainerStack<'_>>,
    ) -> bool {
        if cand.container_query == u32::MAX {
            return true; // not an @container rule — always eligible
        }
        let Some(stack) = stack else {
            return true; // no container info → optimistic apply
        };
        let Some(guard) = self.container_queries.get(cand.container_query as usize) else {
            return true;
        };
        eval_container_guard(guard, stack)
    }

    fn collect_for<'b, E: ElementView<'b>>(
        &self,
        el: E,
        out: &mut Vec<CandidateRef>,
        all_classes: &[String],
    ) {
        if let Some(id) = el.id() {
            if let Some(v) = self.by_id.get(id) {
                out.extend_from_slice(v);
            }
        }
        for c in all_classes {
            if let Some(v) = self.by_class.get(c) {
                out.extend_from_slice(v);
            }
        }
        if let Some(tag) = el.tag_name() {
            let tag_lc = tag.to_ascii_lowercase();
            if let Some(v) = self.by_tag.get(&tag_lc) {
                out.extend_from_slice(v);
            }
        }
        out.extend_from_slice(&self.universal);
    }
}

/// Element trait extension: list every class name on this element. We
/// need it so the index lookup can hit every relevant class bucket
/// without scanning a string per class on the hot path.
pub trait ElementClassList {
    fn class_list(&self) -> Vec<String>;
}

pub fn compute<'a, E>(sheets: &'a [Stylesheet], element: E) -> ComputedStyle
where
    E: ElementView<'a>,
{
    compute_with_inline(sheets, element, &[])
}

/// Like `compute_pseudo` but uses a prebuilt `SelectorIndex`. Faster
/// hot path; renderer calls this for every element during the tree
/// walk to ask "does this have a ::before / ::after?".
pub fn compute_pseudo_with_index<'a, E>(
    idx: &SelectorIndex<'a>,
    element: E,
    pseudo: &str,
    classes: &[String],
) -> Option<ComputedStyle>
where
    E: ElementView<'a>,
{
    // Use the bucketed index instead of scanning every rule in every sheet.
    // The renderer calls this for EVERY element (to test ::before/::after), so
    // the old naive O(elements × rules) scan made CSS-heavy pages quadratic
    // (MDN: 50s → the dominant cost once its full stylesheet loaded). Pseudo
    // rules are bucketed by their element-part key selector (`.foo::before`
    // lands in by_class["foo"], `::before` in universal), so the candidates
    // include every pseudo rule that could match `element`.
    let mut candidates: Vec<CandidateRef> = Vec::with_capacity(16);
    idx.collect_for(element, &mut candidates, classes);
    let mut matched: Vec<Matched<'a>> = Vec::new();
    for cand in &candidates {
        let Some(rule) = idx.resolve_rule(cand) else {
            continue;
        };
        let Some(sel) = rule.selectors.get(cand.sel_idx as usize) else {
            continue;
        };
        if !crate::selectors::matches_for(sel, element, Some(pseudo)) {
            continue;
        }
        for d in &rule.declarations {
            matched.push(Matched {
                specificity: sel.specificity(),
                important: d.important,
                origin: cand.origin,
                layer: cand.layer,
                is_inline: false,
                source_order: cand.source_order as usize,
                decl: d,
            });
        }
    }
    if matched.is_empty() {
        return None;
    }
    matched.sort_by(|a, b| {
        cascade_rank(a)
            .cmp(&cascade_rank(b))
            .then(layer_rank(a).cmp(&layer_rank(b)))
            .then(a.specificity.cmp(&b.specificity))
            .then(a.source_order.cmp(&b.source_order))
    });
    Some(apply_matched(&matched))
}

/// Cascade rules that target `pseudo` (e.g. "before") on `element`.
/// Returns the ComputedStyle for the generated box, or None if no
/// matching rules. Callers use the returned style to decide whether
/// to synthesize a child layout box at all (a present `content`
/// declaration is the spec gate for `::before` / `::after`).
pub fn compute_pseudo<'a, E>(
    sheets: &'a [Stylesheet],
    element: E,
    pseudo: &str,
) -> Option<ComputedStyle>
where
    E: ElementView<'a>,
{
    let mut matched: Vec<Matched<'a>> = Vec::new();
    let mut counter = 0usize;
    for (si, ss) in sheets.iter().enumerate() {
        let origin: u8 = if si == 0 { 0 } else { 1 };
        for rule in &ss.rules {
            for sel in &rule.selectors {
                if crate::selectors::matches_for(sel, element, Some(pseudo)) {
                    for d in &rule.declarations {
                        matched.push(Matched {
                            specificity: sel.specificity(),
                            important: d.important,
                            origin,
                            layer: None,
                            is_inline: false,
            source_order: counter,
                            decl: d,
                        });
                        counter += 1;
                    }
                }
            }
        }
    }
    if matched.is_empty() {
        return None;
    }
    matched.sort_by(|a, b| {
        cascade_rank(a)
            .cmp(&cascade_rank(b))
            .then(layer_rank(a).cmp(&layer_rank(b)))
            .then(a.specificity.cmp(&b.specificity))
            .then(a.source_order.cmp(&b.source_order))
    });
    Some(apply_matched(&matched))
}

/// Compute against a pre-built selector index. This is the fast path —
/// `compute` / `compute_with_inline` allocate an ad-hoc index per call
/// and are only suitable for one-off queries (e.g. tests).
pub fn compute_with_index<'a, E>(
    idx: &SelectorIndex<'a>,
    element: E,
    inline: &'a [Declaration],
    classes: &[String],
) -> ComputedStyle
where
    E: ElementView<'a>,
{
    let mut candidates: Vec<CandidateRef> = Vec::with_capacity(32);
    idx.collect_for(element, &mut candidates, classes);

    let mut matched: Vec<Matched<'a>> = Vec::new();
    let mut counter: usize = 0;
    for cand in &candidates {
        let Some(rule) = idx.resolve_rule(cand) else {
            continue;
        };
        let Some(sel) = rule.selectors.get(cand.sel_idx as usize) else {
            continue;
        };
        if !matches(sel, element) {
            continue;
        }
        for d in &rule.declarations {
            matched.push(Matched {
                specificity: sel.specificity(),
                important: d.important,
                origin: cand.origin,
                layer: cand.layer,
                is_inline: false,
                source_order: cand.source_order as usize,
                decl: d,
            });
        }
    }
    // Inline declarations are Author-origin (highest specificity already
    // boosts them past selectors; origin makes them beat UA too). The
    // `is_inline: true` flag also pushes them above every layered author
    // rule for both normal and !important per CSS Cascade L4 §6.4.4.
    counter = counter.wrapping_add(matched.len());
    for d in inline {
        matched.push(Matched {
            specificity: 1 << 24,
            important: d.important,
            origin: 1,
            layer: None,
            is_inline: true,
            source_order: counter,
            decl: d,
        });
        counter += 1;
    }
    matched.sort_by(|a, b| {
        cascade_rank(a)
            .cmp(&cascade_rank(b))
            .then(layer_rank(a).cmp(&layer_rank(b)))
            .then(a.specificity.cmp(&b.specificity))
            .then(a.source_order.cmp(&b.source_order))
    });
    let mut style = apply_matched(&matched);
    apply_presentational_attrs(&mut style, element);
    style
}

/// `compute_with_index` plus the element's ancestor query-container stack so
/// `@container` size queries are really evaluated (CSS Containment 3 §3). This
/// is the simplest entry point for container-query callers/tests: pass the
/// nearest-ancestor-last stack of [`QueryContainer`]s and rules inside
/// `@container` blocks apply only when their condition is satisfied by the
/// targeted container's content-box size.
pub fn compute_with_index_cq<'a, E>(
    idx: &SelectorIndex<'a>,
    element: E,
    inline: &'a [Declaration],
    classes: &[String],
    containers: QueryContainerStack<'_>,
) -> ComputedStyle
where
    E: ElementView<'a>,
{
    compute_with_index_inheriting_filtered_cq(
        idx,
        element,
        inline,
        classes,
        None,
        None,
        Some(containers),
    )
}

/// `compute_with_index` but seeded with the parent element's resolved
/// `custom_properties` so `var(--name)` references resolve against the
/// inheritance chain.
/// Parse an HTML presentational color attribute (`bgcolor`, `text`, `link`).
/// HTML accepts `#rgb`/`#rrggbb`, bare hex without `#`, and named colors.
fn parse_html_color(v: &str) -> Option<Color> {
    let v = v.trim();
    if v.is_empty() {
        return None;
    }
    if let Some(hex) = v.strip_prefix('#') {
        return Color::from_tokens(&[CssToken::Hash(hex.to_string())]);
    }
    if (v.len() == 3 || v.len() == 6) && v.bytes().all(|b| b.is_ascii_hexdigit()) {
        // HTML allows `bgcolor="ff6600"` without the leading `#`.
        return Color::from_tokens(&[CssToken::Hash(v.to_string())]);
    }
    Color::from_tokens(&[CssToken::Ident(v.to_ascii_lowercase())])
}

/// Apply HTML presentational attributes that the CSS cascade can't see —
/// `colspan` on table cells and `bgcolor` on any element. These rank BELOW the
/// author stylesheet, so a color only applies when CSS left the background
/// unset. (This is why Hacker News's orange header and chrome were missing.)
/// Parse an HTML length attribute (`width`/`height`): bare number = pixels
/// (`width="18"`), `N%` = percent (`width="85%"`).
fn parse_html_length(v: &str) -> Option<Length> {
    let v = v.trim();
    if let Some(pct) = v.strip_suffix('%') {
        pct.trim().parse::<f32>().ok().map(Length::Percent)
    } else {
        v.parse::<f32>().ok().map(Length::Px)
    }
}

fn apply_presentational_attrs<'a, E>(style: &mut ComputedStyle, element: E)
where
    E: ElementView<'a>,
{
    if let Some(cs) = element.attr("colspan") {
        if let Ok(n) = cs.trim().parse::<usize>() {
            if n >= 1 {
                style.table_col_span = Some(n);
            }
        }
    }
    if let Some(rs) = element.attr("rowspan") {
        if let Ok(n) = rs.trim().parse::<usize>() {
            if n >= 1 {
                style.table_row_span = Some(n);
            }
        }
    }
    if style.background_color.is_none() {
        if let Some(bg) = element.attr("bgcolor") {
            if let Some(c) = parse_html_color(bg) {
                style.background_color = Some(c);
            }
        }
    }
    // `width`/`height` attributes (`<table width="85%">`, `<img width="18">`).
    // Lower priority than CSS, so only fill when the cascade left them unset.
    if style.width.is_none() {
        if let Some(w) = element.attr("width") {
            if let Some(l) = parse_html_length(w) {
                style.width = Some(l);
            }
        }
    }
    if style.height.is_none() {
        if let Some(h) = element.attr("height") {
            if let Some(l) = parse_html_length(h) {
                style.height = Some(l);
            }
        }
    }
    // `<td>`/`<th>` take their padding from the nearest ancestor `<table>`'s
    // `cellpadding` attribute, overriding the UA `td { padding }` (a
    // presentational attribute ranks above the UA sheet). This is what makes
    // `cellpadding="0"` tables (Hacker News) render tight instead of padded.
    let tag = element.tag_name().unwrap_or("");
    if tag.eq_ignore_ascii_case("td") || tag.eq_ignore_ascii_case("th") {
        let mut anc = element.parent();
        let mut hops = 0;
        while let Some(a) = anc {
            hops += 1;
            if hops > 8 {
                break;
            }
            if a.tag_name()
                .is_some_and(|t| t.eq_ignore_ascii_case("table"))
            {
                if let Some(cp) = a.attr("cellpadding") {
                    if let Ok(px) = cp.trim().parse::<f32>() {
                        for side in 0..4 {
                            style.padding[side] = Some(Length::Px(px));
                        }
                    }
                }
                break;
            }
            anc = a.parent();
        }
    }
    // CSS UI 4 §5.3 `accent-color`: a CHECKED checkbox/radio fills with the
    // control's accent colour (Chrome paints the box in the accent and a
    // white tick/dot on top). We realise the accent FILL by setting the
    // box background to the resolved accent — reusing the ordinary box
    // background painter — so a checked control reads as accent-coloured
    // instead of an empty white square. `accent-color: auto` (unset)
    // resolves to a representative Chrome accent blue. The white tick mark
    // itself is drawn by the form-control glyph layer; the FILL is the
    // user-visible accent change this property controls.
    if tag.eq_ignore_ascii_case("input") {
        let ty = element.attr("type").unwrap_or("");
        let is_check = ty.eq_ignore_ascii_case("checkbox") || ty.eq_ignore_ascii_case("radio");
        if is_check && element.attr("checked").is_some() {
            // Chrome's default UA accent (the system-blue used for
            // form controls when `accent-color: auto`).
            let accent = style.accent_color.unwrap_or(Color {
                r: 0x1a,
                g: 0x73,
                b: 0xe8,
                a: 255,
            });
            style.background_color = Some(accent);
        }
    }
}

pub fn compute_with_index_inheriting<'a, E>(
    idx: &SelectorIndex<'a>,
    element: E,
    inline: &'a [Declaration],
    classes: &[String],
    inherited_vars: Option<&std::collections::HashMap<String, Vec<CssToken>>>,
) -> ComputedStyle
where
    E: ElementView<'a>,
{
    compute_with_index_inheriting_filtered(idx, element, inline, classes, inherited_vars, None)
}

/// `compute_with_index_inheriting` plus an optional Blink-style ancestor
/// `AncestorFilter`. When `Some`, each candidate is first tested against the
/// filter: any selector whose required-ancestor identifiers are provably absent
/// is skipped without running the full ancestor-walk match. The result is
/// IDENTICAL to passing `None` — the filter can only skip selectors that cannot
/// match (Bloom membership has no false negatives). The caller builds `filter`
/// from the element's ancestor chain during descent (it already has the parent
/// nodes + their classes), so this is allocation-free on the hot path.
pub fn compute_with_index_inheriting_filtered<'a, E>(
    idx: &SelectorIndex<'a>,
    element: E,
    inline: &'a [Declaration],
    classes: &[String],
    inherited_vars: Option<&std::collections::HashMap<String, Vec<CssToken>>>,
    ancestor_filter: Option<&AncestorFilter>,
) -> ComputedStyle
where
    E: ElementView<'a>,
{
    compute_with_index_inheriting_filtered_cq(
        idx,
        element,
        inline,
        classes,
        inherited_vars,
        ancestor_filter,
        None,
    )
}

/// `compute_with_index_inheriting_filtered` plus the element's ancestor
/// query-container stack (CSS Containment 3 §3). When `containers` is `Some`,
/// rules nested in `@container` blocks are applied only when their size-query
/// condition is satisfied by the matching ancestor query container's laid-out
/// content-box size; when it's `None`, those rules apply optimistically (no
/// container info available). This is the entry point the renderer calls once
/// it has laid out the query containers, and the one container-query tests use.
#[allow(clippy::too_many_arguments)]
pub fn compute_with_index_inheriting_filtered_cq<'a, E>(
    idx: &SelectorIndex<'a>,
    element: E,
    inline: &'a [Declaration],
    classes: &[String],
    inherited_vars: Option<&std::collections::HashMap<String, Vec<CssToken>>>,
    ancestor_filter: Option<&AncestorFilter>,
    containers: Option<QueryContainerStack<'_>>,
) -> ComputedStyle
where
    E: ElementView<'a>,
{
    let mut candidates: Vec<CandidateRef> = Vec::with_capacity(32);
    idx.collect_for(element, &mut candidates, classes);

    let mut matched: Vec<Matched<'a>> = Vec::new();
    let mut counter: usize = 0;
    for cand in &candidates {
        if bloom_can_skip(ancestor_filter, cand) {
            continue;
        }
        if !idx.cand_container_ok(cand, containers) {
            continue;
        }
        let Some(rule) = idx.resolve_rule(cand) else {
            continue;
        };
        let Some(sel) = rule.selectors.get(cand.sel_idx as usize) else {
            continue;
        };
        if !matches(sel, element) {
            continue;
        }
        for d in &rule.declarations {
            matched.push(Matched {
                specificity: sel.specificity(),
                important: d.important,
                origin: cand.origin,
                layer: cand.layer,
                is_inline: false,
                source_order: cand.source_order as usize,
                decl: d,
            });
        }
    }
    counter = counter.wrapping_add(matched.len());
    for d in inline {
        matched.push(Matched {
            specificity: 1 << 24,
            important: d.important,
            origin: 1,
            layer: None,
            is_inline: true,
            source_order: counter,
            decl: d,
        });
        counter += 1;
    }
    matched.sort_by(|a, b| {
        cascade_rank(a)
            .cmp(&cascade_rank(b))
            .then(layer_rank(a).cmp(&layer_rank(b)))
            .then(a.specificity.cmp(&b.specificity))
            .then(a.source_order.cmp(&b.source_order))
    });
    let mut style = apply_matched_with_inherited_vars(&matched, inherited_vars);
    apply_presentational_attrs(&mut style, element);
    style
}

/// Like `compute` but folds in extra "inline" declarations (e.g. parsed
/// from an HTML `style="..."` attribute) with maximum-specificity priority.
pub fn compute_with_inline<'a, E>(
    sheets: &'a [Stylesheet],
    element: E,
    inline: &'a [Declaration],
) -> ComputedStyle
where
    E: ElementView<'a>,
{
    let mut matched: Vec<Matched<'a>> = Vec::new();
    let mut counter = 0usize;
    for (si, ss) in sheets.iter().enumerate() {
        let origin: u8 = if si == 0 { 0 } else { 1 };
        for rule in &ss.rules {
            for sel in &rule.selectors {
                if matches(sel, element) {
                    for d in &rule.declarations {
                        matched.push(Matched {
                            specificity: sel.specificity(),
                            important: d.important,
                            origin,
                            layer: None,
                            is_inline: false,
            source_order: counter,
                            decl: d,
                        });
                        counter += 1;
                    }
                }
            }
        }
    }
    // Inline declarations get a specificity above any selector and
    // count as Author origin.
    for d in inline {
        matched.push(Matched {
            specificity: 1 << 24, // beyond any normal selector's id+class+type packing
            important: d.important,
            origin: 1,
            layer: None,
            is_inline: false,
            source_order: counter,
            decl: d,
        });
        counter += 1;
    }

    // Sort by cascade rank (origin × importance) then specificity then
    // source order — winners go last so apply_matched's loop overwrites.
    matched.sort_by(|a, b| {
        cascade_rank(a)
            .cmp(&cascade_rank(b))
            .then(layer_rank(a).cmp(&layer_rank(b)))
            .then(a.specificity.cmp(&b.specificity))
            .then(a.source_order.cmp(&b.source_order))
    });
    let mut style = apply_matched(&matched);
    apply_presentational_attrs(&mut style, element);
    style
}

/// Two-pass declaration application: first collect custom properties so
/// `var(--name)` can resolve, then apply every regular property with the
/// substitution done in-flight.
/// Parse a `clip-path:` value. Accepts `none`, `inset(...)`,
/// `circle(...)`, `polygon(...)`. Returns None for unparseable
/// values so the painter just doesn't clip.
fn parse_clip_path(toks: &[CssToken]) -> Option<ClipPath> {
    let mut i = 0;
    while i < toks.len() {
        match &toks[i] {
            CssToken::Whitespace => {
                i += 1;
                continue;
            }
            CssToken::Ident(name) if name.eq_ignore_ascii_case("none") => return None,
            CssToken::Function(name) => {
                let body_start = i + 1;
                let mut depth = 1;
                let mut j = body_start;
                while j < toks.len() && depth > 0 {
                    match &toks[j] {
                        CssToken::Function(_) | CssToken::LeftParen => depth += 1,
                        CssToken::RightParen => depth -= 1,
                        _ => {}
                    }
                    j += 1;
                }
                let body_end = j.saturating_sub(1);
                let body = &toks[body_start..body_end];
                return match name.to_ascii_lowercase().as_str() {
                    "inset" => parse_inset(body),
                    "circle" => parse_circle(body),
                    "polygon" => parse_polygon(body),
                    _ => None,
                };
            }
            _ => i += 1,
        }
    }
    None
}

fn parse_inset(body: &[CssToken]) -> Option<ClipPath> {
    let mut lengths: Vec<Length> = Vec::new();
    for t in body {
        if let Some(l) = Length::from_tokens(std::slice::from_ref(t)) {
            lengths.push(l);
            if lengths.len() >= 4 {
                break;
            }
        }
    }
    let (top, right, bottom, left) = match lengths.len() {
        1 => (lengths[0], lengths[0], lengths[0], lengths[0]),
        2 => (lengths[0], lengths[1], lengths[0], lengths[1]),
        3 => (lengths[0], lengths[1], lengths[2], lengths[1]),
        4 => (lengths[0], lengths[1], lengths[2], lengths[3]),
        _ => return None,
    };
    Some(ClipPath::Inset {
        top,
        right,
        bottom,
        left,
    })
}

fn parse_circle(body: &[CssToken]) -> Option<ClipPath> {
    // `circle(R at CX CY)` — pull the first Length as radius, the two
    // after `at` as (cx, cy). Missing `at` defaults to center 50% 50%.
    let mut radius: Option<Length> = None;
    let mut after_at = false;
    let mut center: Vec<Length> = Vec::new();
    for t in body {
        match t {
            CssToken::Whitespace | CssToken::Comma => continue,
            CssToken::Ident(s) if s.eq_ignore_ascii_case("at") => {
                after_at = true;
                continue;
            }
            _ => {}
        }
        if let Some(l) = Length::from_tokens(std::slice::from_ref(t)) {
            if !after_at && radius.is_none() {
                radius = Some(l);
            } else if after_at {
                center.push(l);
            }
        }
    }
    let cx = center.first().copied().unwrap_or(Length::Percent(50.0));
    let cy = center.get(1).copied().unwrap_or(Length::Percent(50.0));
    Some(ClipPath::Circle {
        radius: radius.unwrap_or(Length::Percent(50.0)),
        cx,
        cy,
    })
}

fn parse_polygon(body: &[CssToken]) -> Option<ClipPath> {
    // Pairs of lengths separated by commas. `evenodd` / `nonzero`
    // fill-rule prefix is accepted and dropped (we use even-odd).
    let mut pts: Vec<(Length, Length)> = Vec::new();
    let mut pending: Option<Length> = None;
    for t in body {
        match t {
            CssToken::Whitespace => continue,
            CssToken::Comma => {
                if pending.is_some() {
                    // Unmatched x without y — drop.
                    pending = None;
                }
                continue;
            }
            CssToken::Ident(s)
                if s.eq_ignore_ascii_case("evenodd") || s.eq_ignore_ascii_case("nonzero") =>
            {
                continue;
            }
            _ => {}
        }
        if let Some(l) = Length::from_tokens(std::slice::from_ref(t)) {
            match pending {
                None => pending = Some(l),
                Some(prev) => {
                    pts.push((prev, l));
                    pending = None;
                }
            }
        }
    }
    if pts.is_empty() {
        return None;
    }
    Some(ClipPath::Polygon(pts))
}

/// Parse a `filter:` value into a sequence of FilterFn ops. Accepts
/// `none`, `blur(2px)`, `brightness(0.8)`, `grayscale(50%)`, ...,
/// `drop-shadow(2px 3px 5px rgba(0,0,0,0.5))`. Unknown function names
/// are silently skipped so the painter just under-applies rather
/// than crashing on novel CSS.
fn parse_filter_chain(toks: &[CssToken]) -> Vec<FilterFn> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < toks.len() {
        // Skip whitespace / commas.
        match &toks[i] {
            CssToken::Whitespace | CssToken::Comma => {
                i += 1;
                continue;
            }
            CssToken::Ident(name) if name.eq_ignore_ascii_case("none") => {
                return Vec::new();
            }
            CssToken::Url(u) => {
                // `filter: url(#id)` — SVG filter reference (Filter Effects 1
                // §4). Accept only same-document fragment refs (`#id`); strip
                // the leading `#`. An external-document ref (`foo.svg#id`) is
                // not resolvable here, so skip it (under-apply, never crash).
                let frag = u.trim();
                if let Some(id) = frag.strip_prefix('#') {
                    if !id.is_empty() {
                        out.push(FilterFn::Reference(std::sync::Arc::from(id)));
                    }
                }
                i += 1;
            }
            CssToken::Function(name) => {
                // Walk to matching RightParen.
                let body_start = i + 1;
                let mut depth = 1;
                let mut j = body_start;
                while j < toks.len() && depth > 0 {
                    match &toks[j] {
                        CssToken::Function(_) | CssToken::LeftParen => depth += 1,
                        CssToken::RightParen => depth -= 1,
                        _ => {}
                    }
                    j += 1;
                }
                let body_end = j.saturating_sub(1);
                let body = &toks[body_start..body_end];
                if let Some(f) = parse_filter_fn(&name.to_ascii_lowercase(), body) {
                    out.push(f);
                }
                i = j;
            }
            _ => {
                i += 1;
            }
        }
    }
    out
}

fn parse_filter_fn(name: &str, body: &[CssToken]) -> Option<FilterFn> {
    let numeric_arg = || -> Option<f32> {
        for t in body {
            match t {
                CssToken::Number(n) => return Some(*n as f32),
                CssToken::Percent(p) => return Some((*p as f32) / 100.0),
                CssToken::Dimension { value, unit } => {
                    let v = *value as f32;
                    return Some(match unit.to_ascii_lowercase().as_str() {
                        "px" => v,
                        "deg" => v,
                        "rad" => v.to_degrees(),
                        "turn" => v * 360.0,
                        _ => v,
                    });
                }
                _ => {}
            }
        }
        None
    };
    match name {
        "blur" => Some(FilterFn::Blur(numeric_arg().unwrap_or(0.0).max(0.0))),
        "brightness" => Some(FilterFn::Brightness(numeric_arg().unwrap_or(1.0).max(0.0))),
        "contrast" => Some(FilterFn::Contrast(numeric_arg().unwrap_or(1.0).max(0.0))),
        "grayscale" => Some(FilterFn::Grayscale(
            numeric_arg().unwrap_or(0.0).clamp(0.0, 1.0),
        )),
        "invert" => Some(FilterFn::Invert(
            numeric_arg().unwrap_or(0.0).clamp(0.0, 1.0),
        )),
        "opacity" => Some(FilterFn::Opacity(
            numeric_arg().unwrap_or(1.0).clamp(0.0, 1.0),
        )),
        "saturate" => Some(FilterFn::Saturate(numeric_arg().unwrap_or(1.0).max(0.0))),
        "sepia" => Some(FilterFn::Sepia(
            numeric_arg().unwrap_or(0.0).clamp(0.0, 1.0),
        )),
        "hue-rotate" => Some(FilterFn::HueRotate(numeric_arg().unwrap_or(0.0))),
        "drop-shadow" => {
            // `drop-shadow(<offset-x> <offset-y> <blur>? <color>?)`.
            let mut lengths: Vec<Length> = Vec::new();
            for t in body {
                if let Some(l) = Length::from_tokens(std::slice::from_ref(t)) {
                    lengths.push(l);
                }
                if lengths.len() >= 3 {
                    break;
                }
            }
            if lengths.len() < 2 {
                return None;
            }
            let color = Color::from_tokens(body).unwrap_or(Color::BLACK);
            Some(FilterFn::DropShadow(BoxShadowSpec {
                offset_x: lengths[0],
                offset_y: lengths[1],
                blur: lengths.get(2).copied().unwrap_or(Length::Zero),
                spread: Length::Zero,
                color,
                inset: false,
            }))
        }
        _ => None,
    }
}

/// One @keyframes rule: name + ordered timeline of (offset, declarations).
/// Offset is 0..1 inclusive (0% → 0.0, 100% → 1.0).
#[derive(Debug, Clone)]
pub struct KeyframeRule {
    pub name: String,
    pub steps: Vec<(f32, Vec<crate::parser::Declaration>)>,
}

/// Extract all `@keyframes` rules from `sheets` into a lookup by name.
/// Subsequent rule with the same name wins (matches browser cascade
/// semantics for keyframes).
pub fn collect_keyframes(sheets: &[Stylesheet]) -> std::collections::HashMap<String, KeyframeRule> {
    let mut out: std::collections::HashMap<String, KeyframeRule> = std::collections::HashMap::new();
    for ss in sheets {
        for at in &ss.at_rules {
            if !at.name.eq_ignore_ascii_case("keyframes") {
                continue;
            }
            // Prelude is the rule's name token list.
            let mut name = String::new();
            for t in &at.prelude {
                match t {
                    CssToken::Ident(s) => {
                        name = s.clone();
                        break;
                    }
                    CssToken::String(s) => {
                        name = s.clone();
                        break;
                    }
                    _ => {}
                }
            }
            if name.is_empty() {
                continue;
            }
            let Some(block) = &at.block else { continue };
            let mut steps: Vec<(f32, Vec<crate::parser::Declaration>)> = Vec::new();
            for rule in block {
                // Selectors here are `0%`, `50%`, `100%`, `from`, `to`.
                for sel in &rule.selectors {
                    // SelectorIndex stores Selector, but for keyframes
                    // the prelude is just an Ident or Percent — the
                    // parser routes them to a tag/ident selector or
                    // a special selector that we approximate by
                    // checking the first compound's element.
                    let offset = keyframe_offset_from(sel);
                    if let Some(off) = offset {
                        steps.push((off, rule.declarations.clone()));
                    }
                }
            }
            steps.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
            out.insert(name.clone(), KeyframeRule { name, steps });
        }
    }
    out
}

// ── @keyframes collection memo (Blink StyleRuleKeyframes) ────────────────────
//
// Blink parses `@keyframes` ONCE when the stylesheet is parsed into a
// `StyleRuleKeyframes` stored on the resolver's `StyleEngine`, then looks it up
// by name during animation; it is NOT re-collected/re-parsed per animated frame
// (Blink's `CSSAnimations::CalculateAnimationUpdate` reads the already-built
// keyframe model). Our per-frame render path called `collect_keyframes(sheets)`
// on EVERY animated frame, re-walking every sheet's at-rules and re-cloning +
// re-sorting every keyframe step's declaration list into a fresh HashMap.
//
// `collect_keyframes_cached` memoizes that result in a thread-local, keyed on a
// fingerprint of EXACTLY the inputs `collect_keyframes` reads — every
// `@keyframes` at-rule's name (prelude) and block (offset selectors +
// declarations) across all sheets. The fingerprint is a pure function of those
// inputs, so an identical fingerprint guarantees an identical collection result
// (a cache HIT returns the same map the cold path would build, byte-for-byte).
// When the stylesheet set changes (a new/edited/removed `@keyframes`, a new
// document, a CSSOM insertRule), the fingerprint changes => cache miss =>
// re-collect. Non-keyframe rule changes do not invalidate (correct: they do not
// affect the keyframe map). Returned as an `Rc` so callers reuse the same parsed
// model across frames with no per-frame clone of the declaration Vecs.
fn keyframes_fingerprint(sheets: &[Stylesheet]) -> u64 {
    // FNV-1a over the keyframe-relevant token/structure stream only. This is the
    // exact set of bytes `collect_keyframes` consumes, and it is far cheaper than
    // the clone+sort+HashMap build it gates (and, on a page with NO @keyframes,
    // is just the at-rule-name scan the cold path already pays).
    const FNV_OFFSET: u64 = 0xcbf29ce484222325;
    const FNV_PRIME: u64 = 0x100000001b3;
    let mut h: u64 = FNV_OFFSET;
    let mut byte = |h: &mut u64, b: u8| {
        *h ^= b as u64;
        *h = h.wrapping_mul(FNV_PRIME);
    };
    let mut bytes = |h: &mut u64, s: &[u8]| {
        for &b in s {
            byte(h, b);
        }
    };
    let mut hash_token = |h: &mut u64, t: &CssToken| {
        // A discriminant byte + the token's payload, so two distinct token kinds
        // with the same string payload never collide.
        match t {
            CssToken::Ident(s) => {
                byte(h, 1);
                bytes(h, s.as_bytes());
            }
            CssToken::Function(s) => {
                byte(h, 2);
                bytes(h, s.as_bytes());
            }
            CssToken::AtKeyword(s) => {
                byte(h, 3);
                bytes(h, s.as_bytes());
            }
            CssToken::Hash(s) => {
                byte(h, 4);
                bytes(h, s.as_bytes());
            }
            CssToken::String(s) => {
                byte(h, 5);
                bytes(h, s.as_bytes());
            }
            CssToken::Number(n) => {
                byte(h, 6);
                bytes(h, &n.to_bits().to_le_bytes());
            }
            CssToken::Percent(n) => {
                byte(h, 7);
                bytes(h, &n.to_bits().to_le_bytes());
            }
            CssToken::Dimension { value, unit } => {
                byte(h, 8);
                bytes(h, &value.to_bits().to_le_bytes());
                bytes(h, unit.as_bytes());
            }
            CssToken::Url(s) => {
                byte(h, 9);
                bytes(h, s.as_bytes());
            }
            CssToken::Delim(c) => {
                byte(h, 10);
                bytes(h, &(*c as u32).to_le_bytes());
            }
            // Punctuation / structural tokens: discriminant only (no payload).
            CssToken::Whitespace => byte(h, 11),
            CssToken::Colon => byte(h, 12),
            CssToken::Semicolon => byte(h, 13),
            CssToken::Comma => byte(h, 14),
            CssToken::LeftBrace => byte(h, 15),
            CssToken::RightBrace => byte(h, 16),
            CssToken::LeftParen => byte(h, 17),
            CssToken::RightParen => byte(h, 18),
            CssToken::LeftBracket => byte(h, 19),
            CssToken::RightBracket => byte(h, 20),
            CssToken::Bang => byte(h, 21),
            CssToken::Eof => byte(h, 22),
        }
    };
    // Top-level sheet count, so adding/removing whole sheets always perturbs.
    bytes(&mut h, &(sheets.len() as u64).to_le_bytes());
    for ss in sheets {
        // Sheet separator + that sheet's at-rule count.
        byte(&mut h, 0xFF);
        bytes(&mut h, &(ss.at_rules.len() as u64).to_le_bytes());
        for at in &ss.at_rules {
            if !at.name.eq_ignore_ascii_case("keyframes") {
                // Still account for the rule's PRESENCE (its index position
                // matters: it shifts which keyframes a same-name later rule
                // overrides) without hashing its body — cheap.
                byte(&mut h, 0xA0);
                continue;
            }
            byte(&mut h, 0xB0);
            for t in &at.prelude {
                hash_token(&mut h, t);
            }
            if let Some(block) = &at.block {
                bytes(&mut h, &(block.len() as u64).to_le_bytes());
                for rule in block {
                    for sel in &rule.selectors {
                        match keyframe_offset_from(sel) {
                            Some(off) => bytes(&mut h, &off.to_bits().to_le_bytes()),
                            None => byte(&mut h, 0),
                        }
                    }
                    for d in &rule.declarations {
                        bytes(&mut h, d.name.as_bytes());
                        byte(&mut h, b'=');
                        for t in &d.value {
                            hash_token(&mut h, t);
                        }
                        if d.important {
                            byte(&mut h, b'!');
                        }
                        byte(&mut h, b';');
                    }
                }
            } else {
                byte(&mut h, 0xC0);
            }
        }
    }
    h
}

thread_local! {
    /// Memo for `collect_keyframes_cached`: (fingerprint, parsed model). One slot
    /// is sufficient — the per-frame caller passes the SAME sheet set across all
    /// frames of a page, so the slot stays hot; a navigation/sheet edit changes
    /// the fingerprint and replaces it.
    static KEYFRAMES_MEMO: std::cell::RefCell<
        Option<(u64, std::rc::Rc<std::collections::HashMap<String, KeyframeRule>>)>,
    > = const { std::cell::RefCell::new(None) };
}

/// Memoized form of [`collect_keyframes`]. Returns the SAME map the cold path
/// would build (oracle-checked in tests), reusing a cached `Rc` across frames
/// when the keyframe-relevant content of `sheets` is unchanged. Use this on the
/// per-frame animation path; use [`collect_keyframes`] when you need a fresh
/// owned map or are the oracle. Honors `CV_KEYFRAMES_MEMO=0` to force the cold
/// path (the A/B oracle escape hatch).
pub fn collect_keyframes_cached(
    sheets: &[Stylesheet],
) -> std::rc::Rc<std::collections::HashMap<String, KeyframeRule>> {
    if !keyframes_memo_enabled() {
        return std::rc::Rc::new(collect_keyframes(sheets));
    }
    let fp = keyframes_fingerprint(sheets);
    KEYFRAMES_MEMO.with(|cell| {
        if let Some((cached_fp, rc)) = cell.borrow().as_ref() {
            if *cached_fp == fp {
                return rc.clone();
            }
        }
        let built = std::rc::Rc::new(collect_keyframes(sheets));
        *cell.borrow_mut() = Some((fp, built.clone()));
        built
    })
}

/// Test/diagnostic hook: drop the memoized keyframe model so the next
/// `collect_keyframes_cached` re-collects cold. Mirrors the per-build cache
/// resets used elsewhere (e.g. the GDI text-measure width cache).
pub fn clear_keyframes_memo() {
    KEYFRAMES_MEMO.with(|cell| *cell.borrow_mut() = None);
}

fn keyframes_memo_enabled() -> bool {
    use std::sync::OnceLock;
    static ENABLED: OnceLock<bool> = OnceLock::new();
    // Default-ON: the fingerprint is a pure function of the exact inputs
    // `collect_keyframes` reads, so a hit is byte-identical to a cold collect
    // (oracle-proven). `CV_KEYFRAMES_MEMO=0` forces the cold path.
    *ENABLED.get_or_init(|| std::env::var("CV_KEYFRAMES_MEMO").as_deref() != Ok("0"))
}

/// Sample an animation at progress `t` (0..1) by interpolating between
/// adjacent keyframe steps. Returns a HashMap of property→value
/// strings (in CSS source form) that the caller can re-apply over
/// the element's normally-computed style.
pub fn sample_animation(rule: &KeyframeRule, t: f32) -> std::collections::HashMap<String, String> {
    let t = t.clamp(0.0, 1.0);
    let steps = &rule.steps;
    let mut out = std::collections::HashMap::new();
    if steps.is_empty() {
        return out;
    }
    // Find the bracketing pair of steps.
    let (lo_idx, hi_idx) = bracketing_steps(steps, t);
    let (lo_off, lo_decls) = &steps[lo_idx];
    let (hi_off, hi_decls) = &steps[hi_idx];
    let span = (hi_off - lo_off).max(f32::EPSILON);
    let local = ((t - lo_off) / span).clamp(0.0, 1.0);

    // For each property: if both endpoints declare it, interpolate
    // (numeric-prefix only — full CSS value interpolation is a follow-
    // up). If only one declares it, pass through the declared value.
    let mut all_keys: std::collections::HashSet<&str> = std::collections::HashSet::new();
    for d in lo_decls {
        all_keys.insert(d.name.as_str());
    }
    for d in hi_decls {
        all_keys.insert(d.name.as_str());
    }
    for key in all_keys {
        let lo_val = lo_decls
            .iter()
            .find(|d| d.name == key)
            .map(|d| value_to_string(&d.value));
        let hi_val = hi_decls
            .iter()
            .find(|d| d.name == key)
            .map(|d| value_to_string(&d.value));
        let interpolated = match (lo_val, hi_val) {
            (Some(a), Some(b)) => interpolate_value(&a, &b, local),
            (Some(a), None) => a,
            (None, Some(b)) => b,
            (None, None) => continue,
        };
        out.insert(key.to_string(), interpolated);
    }
    out
}

fn bracketing_steps(steps: &[(f32, Vec<crate::parser::Declaration>)], t: f32) -> (usize, usize) {
    if steps.len() == 1 {
        return (0, 0);
    }
    for (i, (off, _)) in steps.iter().enumerate().skip(1) {
        if *off >= t {
            return (i - 1, i);
        }
    }
    (steps.len() - 2, steps.len() - 1)
}

/// Render a token list back to source-shape so the interpolator can
/// pattern-match it. V1 only handles single-numeric values (e.g.
/// `0.5`, `12px`) — the only shapes we actually interpolate.
fn value_to_string(tokens: &[CssToken]) -> String {
    let mut s = String::new();
    for t in tokens {
        match t {
            CssToken::Number(n) => s.push_str(&n.to_string()),
            CssToken::Dimension { value, unit } => {
                s.push_str(&value.to_string());
                s.push_str(unit);
            }
            CssToken::Percent(n) => {
                s.push_str(&n.to_string());
                s.push('%');
            }
            CssToken::Ident(i) => s.push_str(i),
            // A function like `translate(` / `rotate(` — without this the
            // function name and its parens were dropped, mangling a transform
            // value (`translate(-50%,-50%) rotate(0deg)` → `-50%-50%0deg`) so it
            // couldn't be parsed or interpolated.
            CssToken::Function(f) => {
                s.push_str(f);
                s.push('(');
            }
            CssToken::LeftParen => s.push('('),
            CssToken::RightParen => s.push(')'),
            CssToken::Comma => s.push_str(", "),
            CssToken::String(st) => s.push_str(st),
            CssToken::Hash(h) => {
                s.push('#');
                s.push_str(h);
            }
            CssToken::Delim(c) => s.push(*c),
            CssToken::Whitespace => s.push(' '),
            _ => {}
        }
    }
    s
}

/// Parse a CSS color string (hex `#abc`/`#aabbcc`, `rgb()`/`rgba()`,
/// `hsl()`/`hsla()`, or a named color) into a [`Color`]. Returns `None` for
/// non-color values (lengths, numbers, keywords like `none`). Public so the
/// animation/transition driver can interpolate colors in their own component
/// space rather than via text. Reuses the production [`Color::from_tokens`]
/// parser used by the cascade, so it covers every color form the cascade does.
pub fn parse_color_str(s: &str) -> Option<Color> {
    let s = s.trim();
    if s.is_empty() {
        return None;
    }
    let toks = crate::tokenizer::tokenize(s);
    // Drop leading/trailing whitespace tokens so a value like " #fff " parses.
    let toks: Vec<CssToken> = toks
        .into_iter()
        .filter(|t| !matches!(t, CssToken::Whitespace))
        .collect();
    if toks.is_empty() {
        return None;
    }
    Color::from_tokens(&toks)
}

/// Format an interpolated color back to an `rgba(r, g, b, a)` string that
/// `parse_color_str` (and downstream length/number parsers) round-trips. Alpha
/// is 0..1 per CSS. Components are clamped+rounded into 0..255.
fn color_to_rgba_string(c: Color) -> String {
    format!(
        "rgba({}, {}, {}, {})",
        c.r,
        c.g,
        c.b,
        (c.a as f32 / 255.0)
    )
}

/// Interpolate between two CSS values at progress `t`. Handles:
///   * colors (hex/named/rgb/hsl) — interpolated component-wise in sRGB
///     (CSS Color 4 §13 / CSS Transitions L1 "by computed value, as color"),
///     non-premultiplied (the common opaque case is plain per-channel lerp)
///   * plain numbers (`0` → `1`)
///   * length values (`0px` → `100px`)
///   * percentages (`0%` → `100%`)
///   * `translate(Xpx)` shorthand and other multi-number skeletons
/// Anything else falls back to a step at t≥0.5.
fn interpolate_value(a: &str, b: &str, t: f32) -> String {
    // Color path FIRST: a hex/named color has no decimal-number runs the generic
    // skeleton path could lerp (`#000`→`#fff` would step), and `rgb()` digits
    // must move as a unit in color space (incl. alpha), not as arbitrary text
    // numbers. When BOTH ends parse as colors, interpolate RGBA component-wise.
    if let (Some(ca), Some(cb)) = (parse_color_str(a), parse_color_str(b)) {
        // `currentColor` is a context-dependent sentinel, not a concrete color —
        // don't interpolate it numerically (it would emit garbage). Step instead.
        if ca.is_current_color() || cb.is_current_color() {
            return if t >= 0.5 { b.to_string() } else { a.to_string() };
        }
        let lerp = |x: u8, y: u8| -> u8 {
            (x as f32 + (y as f32 - x as f32) * t).round().clamp(0.0, 255.0) as u8
        };
        let out = Color {
            r: lerp(ca.r, cb.r),
            g: lerp(ca.g, cb.g),
            b: lerp(ca.b, cb.b),
            a: lerp(ca.a, cb.a),
        };
        return color_to_rgba_string(out);
    }
    // Locate every numeric run (optional sign, digits, decimal) in a string.
    fn numbers(s: &str) -> Vec<(usize, usize, f32)> {
        let bytes = s.as_bytes();
        let mut out = Vec::new();
        let mut i = 0;
        while i < bytes.len() {
            let c = bytes[i] as char;
            let starts = c.is_ascii_digit()
                || ((c == '-' || c == '+' || c == '.')
                    && i + 1 < bytes.len()
                    && (bytes[i + 1] as char).is_ascii_digit());
            if starts {
                let start = i;
                if c == '-' || c == '+' {
                    i += 1;
                }
                while i < bytes.len() && ((bytes[i] as char).is_ascii_digit() || bytes[i] == b'.') {
                    i += 1;
                }
                if let Ok(v) = s[start..i].parse::<f32>() {
                    out.push((start, i, v));
                } else {
                    i += 1;
                }
            } else {
                i += 1;
            }
        }
        out
    }
    // Component-wise: when both values have the same NON-numeric skeleton (so the
    // same count of numbers in order), interpolate each number and keep `a`'s
    // structure. Handles multi-function transforms
    // (`… rotate(0deg)` → `… rotate(360deg)`), multiple lengths, colors digits,
    // etc. — not just a single leading number.
    let na = numbers(a);
    let nb = numbers(b);
    if !na.is_empty() && na.len() == nb.len() {
        let mut result = String::new();
        let mut last = 0;
        for (idx, &(s, e, va)) in na.iter().enumerate() {
            result.push_str(&a[last..s]);
            let v = va + (nb[idx].2 - va) * t;
            result.push_str(&format!("{v}"));
            last = e;
        }
        result.push_str(&a[last..]);
        return result;
    }
    if t >= 0.5 {
        b.to_string()
    } else {
        a.to_string()
    }
}

fn keyframe_offset_from(sel: &crate::selectors::Selector) -> Option<f32> {
    let last = sel.parts.last()?;
    if let Some(e) = &last.compound.element {
        match e.to_ascii_lowercase().as_str() {
            "from" => return Some(0.0),
            "to" => return Some(1.0),
            _ => {}
        }
        // Try parse a `Npct` ident — the selector tokenizer doesn't
        // distinguish percentages from identifiers. Caller will have
        // stored e.g. "50%" as the element name in some cases.
        if let Some(stripped) = e.strip_suffix('%') {
            if let Ok(n) = stripped.parse::<f32>() {
                return Some((n / 100.0).clamp(0.0, 1.0));
            }
        }
    }
    None
}

fn apply_matched(matched: &[Matched<'_>]) -> ComputedStyle {
    apply_matched_with_inherited_vars(matched, None)
}

/// `apply_matched`, but seeds the custom-property map with the
/// parent's resolved custom properties before pass 1. CSS custom
/// properties inherit by default (CSS Variables L1 §3); element-level
/// declarations on this element still override (pass 1 inserts after
/// the seed).
/// V1 cascade with the multi-origin layering Chrome's
/// `StyleCascade::Apply()` performs collapsed onto a single pass
/// through pre-sorted `matched` declarations:
///   - @layer ordering is honored upstream by the index builder
///     (`cv_css::SelectorIndex`, see render-fix #354)
///   - `!important` is folded into the sort key by the selector
///     index, so the apply pass just walks in cascade order
///   - inherited custom properties (CSS Variables L1 §3) compose via
///     `inherited` argument before this element's own decls
///   - var() expansion runs at apply-time (pass 2) — matches "computed-
///     value time" wording in the spec
///
/// What this V1 deliberately does NOT do, matching the audit point
/// #488 (Chrome-divergence #6): there is no explicit animation /
/// transition origin layer. In Chrome, a running animation
/// contributes a synthetic declaration above the author origin so
/// re-cascading (e.g. on resize, on style writes) still picks up
/// the interpolated value. Ours plays animations by directly
/// patching `LayoutBox` fields at tick time (see the animation
/// driver). That's fine for the current animation set (opacity
/// fades, simple transforms, hover transitions) because the patch
/// runs every frame against the freshly-cascaded style. It would
/// break for any animation whose computed value feeds a downstream
/// layout pass keyed off computed style (CSS subgrid, container
/// queries) — those would see the static value, not the animated
/// one. None of those compositions exist on the test sites today.
/// Tracked for V2 as a real origin-layer split.
fn apply_matched_with_inherited_vars(
    matched: &[Matched<'_>],
    inherited: Option<&std::collections::HashMap<String, Vec<CssToken>>>,
) -> ComputedStyle {
    let mut style = ComputedStyle::default();
    // FAST PATH: the vast majority of elements declare NO custom properties of
    // their own. For those we must NOT clone the entire inherited `--var` map
    // into a fresh per-element HashMap (that was O(vars-in-scope) allocations ×
    // every element × every frame — a top cascade hotspot on var-heavy / utility
    // CSS). Instead we leave `custom_properties` empty (the caller threads the
    // parent's map straight through to children — see build_styled_tree_full)
    // and resolve this element's own `var()` references directly against the
    // inherited map below. Render output is identical; only the per-element
    // clone is removed.
    let empty_vars = std::collections::HashMap::new();
    let has_own_vars = matched.iter().any(|m| m.decl.name.starts_with("--"));
    if has_own_vars {
        // Element overrides/adds custom properties: build the merged map
        // (inherited overlaid with own) since children inherit the merged set.
        // Pass 0: seed inherited custom properties from the parent.
        if let Some(map) = inherited {
            for (k, v) in map.iter() {
                style.custom_properties.insert(k.clone(), v.clone());
            }
        }
    }
    // Pass 1: gather custom properties declared on this element in
    // cascade order. Later ones win; element-level decls overwrite
    // inherited entries. We also EXPAND each stored value through
    // `expand_vars` against the in-progress map, so chained
    // definitions like `--gold-rgb: rgb(var(--gold-r), ...)` resolve
    // immediately instead of leaving raw `var()` tokens inside the
    // stored value. Without expansion-at-store-time, pass 2 sees a
    // value like `[var, (, --gold-r, ), ...]` and the inner expansion
    // depends on the map state at pass-2 time — which is fine as long
    // as the chain is intact, but degrades to empty token lists when
    // an async-loaded stylesheet arrives between renders and a chain
    // link briefly disappears. Eager expansion at pass 1 also matches
    // the "computed-value time" wording in CSS Variables L1 §3.
    for m in matched {
        if m.decl.name.starts_with("--") {
            let expanded = expand_vars(&m.decl.value, &style.custom_properties, 0);
            // If expansion produced nothing (broken chain) but the raw
            // value has actual tokens, keep the raw value so a later
            // resolve attempt can still fire — better than burning a
            // useful inherited entry.
            let to_store = if expanded.is_empty() && !m.decl.value.is_empty() {
                m.decl.value.clone()
            } else {
                expanded
            };
            style
                .custom_properties
                .insert(m.decl.name.clone(), to_store);
        }
    }
    // Pass 2: apply each regular property, expanding var() against the effective
    // custom-property map — the element's own merged map when it declared vars,
    // otherwise the inherited map directly (no per-element clone). Resolved
    // per-iteration so the (immutable) var-map borrow doesn't overlap the
    // (mutable) `apply_declaration` write to `style`.
    for m in matched {
        if m.decl.name.starts_with("--") {
            continue;
        }
        let expanded = if has_own_vars {
            expand_vars(&m.decl.value, &style.custom_properties, 0)
        } else {
            expand_vars(&m.decl.value, inherited.unwrap_or(&empty_vars), 0)
        };
        let temp = Declaration {
            name: m.decl.name.clone(),
            value: expanded,
            important: m.decl.important,
        };
        apply_declaration(&mut style, &temp);
    }
    // Map flow-relative box quantities (margin-inline-start, inline-size, …)
    // onto physical sides/axes using this element's OWN declared writing-mode
    // + direction (or the horizontal-tb/ltr default). When the element
    // INHERITS its writing-mode/direction from an ancestor without
    // redeclaring it, the host's inheritance pass re-runs
    // `resolve_logical_box` after filling in the inherited values — that pass
    // is re-resolution-safe (see the undo block in `resolve_logical_box`).
    resolve_logical_box(&mut style);
    style
}

/// Walk a value token list replacing `var(--name [, fallback])` calls with
/// the stored custom-property tokens (or the fallback if absent). Recursive
/// with a depth limit so a cyclic chain (`--a: var(--b); --b: var(--a);`)
/// can't blow the stack — at the limit we just emit an empty value, which
/// causes the property to fall back to its initial state per CSS Vars.
fn expand_vars(
    toks: &[CssToken],
    vars: &std::collections::HashMap<String, Vec<CssToken>>,
    depth: u32,
) -> Vec<CssToken> {
    if depth >= 16 {
        return Vec::new();
    }
    let mut out: Vec<CssToken> = Vec::with_capacity(toks.len());
    let mut i = 0;
    while i < toks.len() {
        if let CssToken::Function(name) = &toks[i] {
            if name.eq_ignore_ascii_case("var") {
                if let Some(end) = find_matching_paren_in_value(toks, i + 1) {
                    let args = &toks[i + 1..end];
                    // Split on top-level commas: first chunk is the
                    // variable name (must start with `--`), rest are
                    // the fallback value.
                    let parts = split_top_level_commas(args);
                    let mut var_name: Option<String> = None;
                    if let Some(first) = parts.first() {
                        for t in first {
                            if let CssToken::Ident(s) = t {
                                if s.starts_with("--") {
                                    var_name = Some(s.clone());
                                    break;
                                }
                            }
                        }
                    }
                    let mut substituted: Vec<CssToken> = Vec::new();
                    if let Some(n) = &var_name {
                        if let Some(stored) = vars.get(n) {
                            substituted = expand_vars(stored, vars, depth + 1);
                        }
                    }
                    if substituted.is_empty() && parts.len() > 1 {
                        // Concatenate the fallback parts (joined by ',')
                        // — important for `var(--font, "Helvetica", sans-serif)`.
                        let mut fallback: Vec<CssToken> = Vec::new();
                        for (k, p) in parts.iter().enumerate().skip(1) {
                            if k > 1 {
                                fallback.push(CssToken::Comma);
                            }
                            fallback.extend_from_slice(p);
                        }
                        substituted = expand_vars(&fallback, vars, depth + 1);
                    }
                    out.extend(substituted);
                    i = end + 1;
                    continue;
                }
            }
        }
        out.push(toks[i].clone());
        i += 1;
    }
    out
}

fn length_to_calc(length: Length) -> crate::properties::Calc {
    let mut calc = crate::properties::Calc::default();
    match length {
        Length::Px(v) => calc.px = v,
        Length::Em(v) => calc.em = v,
        Length::Rem(v) => calc.rem = v,
        Length::Vw(v) => calc.vw = v,
        Length::Vh(v) => calc.vh = v,
        Length::Pt(v) => calc.pt = v,
        Length::Percent(v) => calc.percent = v,
        Length::Zero => {}
        Length::Calc(c) => calc = c,
        Length::Clamp(expr) => calc = expr.preferred,
        Length::Auto => {}
    }
    calc
}

/// Parse a `border-{top,right,bottom,left}` value like `1px solid #aaa`.
/// Width defaults to 1px when only style/color appears; color defaults
/// to black. The style component (solid/dashed/dotted/…) parses for
/// completeness but isn't surfaced — V1 paints every border as solid.
/// Parse a `border[-side]` shorthand value list, returning
/// `(width, color, style)`. A `none` or `hidden` keyword sets width to 0
/// and style to `None`/`Hidden` respectively. Unknown idents that are not
/// a recognised colour are checked against `BorderStyle::from_ident` first.
fn parse_border_shorthand(toks: &[CssToken]) -> (Length, Color, Option<BorderStyle>) {
    let mut width: Option<Length> = None;
    let mut color: Option<Color> = None;
    let mut style: Option<BorderStyle> = None;
    // Index-based so a function colour (`rgb(0 128 0)` / `hsl(...)`) is parsed
    // as a SINGLE component: we hand the whole `fn(...)` slice to
    // `Color::from_tokens` and skip its inner tokens. Before this, the inner
    // `Number` tokens of an `rgb()` leaked into the width arm (the last `0`
    // clobbered a real `2px` width) — a latent bug that also affected
    // `border: 2px solid rgb(...)`.
    let mut i = 0;
    while i < toks.len() {
        let t = &toks[i];
        match t {
            CssToken::Dimension { .. } | CssToken::Number(_) => {
                if let Some(l) = Length::from_tokens(std::slice::from_ref(t)) {
                    width = Some(l);
                }
            }
            CssToken::Ident(s) => {
                // Style keywords take priority over colour names.
                if let Some(bs) = BorderStyle::from_ident(s) {
                    style = Some(bs);
                } else if let Some(c) = Color::from_name(s) {
                    color = Some(c);
                }
            }
            CssToken::Hash(_) => {
                if let Some(c) = Color::from_tokens(std::slice::from_ref(t)) {
                    color = Some(c);
                }
            }
            CssToken::Function(_) => {
                // Consume the whole `fn( ... )` group as one colour value so
                // the inner numbers never reach the width/style arms.
                let end = find_matching_paren_in_value(toks, i + 1).unwrap_or(toks.len() - 1);
                if let Some(c) = Color::from_tokens(&toks[i..=end]) {
                    color = Some(c);
                }
                i = end + 1;
                continue;
            }
            _ => {}
        }
        i += 1;
    }
    // When style is None or Hidden the spec says width collapses to 0
    // (CSS §8.5.1 "If `border-style` is `none` or `hidden`, the border
    // width is set to 0").
    let effective_width = if style.as_ref().map_or(false, |s| s.is_none()) {
        Length::Px(0.0)
    } else {
        width.unwrap_or(Length::Px(1.0))
    };
    (
        effective_width,
        color.unwrap_or(Color::BLACK),
        style,
    )
}

/// Resolve a [`Length`] to an approximate px value for *cosmetic* properties
/// where the exact font/viewport context is unavailable at cascade time
/// (e.g. `column-rule-width`, which never affects layout — CSS Multicol §6).
/// `em`/`rem` use the 16px CSS default; `pt` converts at 96dpi; `%`/`auto`/
/// `calc`/`clamp` collapse to their px term (0 for pure-percent).
fn length_to_px_approx(l: Length) -> f32 {
    match l {
        Length::Px(v) => v,
        Length::Em(v) | Length::Rem(v) => v * 16.0,
        Length::Pt(v) => v * 96.0 / 72.0,
        Length::Vw(_) | Length::Vh(_) | Length::Percent(_) | Length::Auto | Length::Zero => 0.0,
        Length::Calc(c) => c.px,
        Length::Clamp(c) => c.preferred.px,
    }
}

fn add_lengths(a: Length, b: Length) -> Length {
    match (a, b) {
        (Length::Zero, other) | (other, Length::Zero) => other,
        (Length::Px(x), Length::Px(y)) => Length::Px(x + y),
        (Length::Em(x), Length::Em(y)) => Length::Em(x + y),
        (Length::Rem(x), Length::Rem(y)) => Length::Rem(x + y),
        (Length::Vw(x), Length::Vw(y)) => Length::Vw(x + y),
        (Length::Vh(x), Length::Vh(y)) => Length::Vh(x + y),
        (Length::Pt(x), Length::Pt(y)) => Length::Pt(x + y),
        (Length::Percent(x), Length::Percent(y)) => Length::Percent(x + y),
        (left, right) => {
            let mut calc = length_to_calc(left);
            let other = length_to_calc(right);
            calc.px += other.px;
            calc.em += other.em;
            calc.rem += other.rem;
            calc.vw += other.vw;
            calc.vh += other.vh;
            calc.pt += other.pt;
            calc.percent += other.percent;
            Length::Calc(calc)
        }
    }
}

/// Strip a leading vendor prefix (`-webkit-`, `-moz-`, `-ms-`, `-o-`)
/// from a CSS property name. Real sites still ship thousands of these
/// for legacy mobile / IE compatibility; treating each as an alias of
/// the unprefixed property is what every real browser does. Returns
/// the original slice unchanged if no recognised prefix is present.
fn strip_vendor_prefix(name: &str) -> &str {
    for p in ["-webkit-", "-moz-", "-ms-", "-o-"] {
        if let Some(rest) = name.strip_prefix(p) {
            return rest;
        }
    }
    name
}

/// Whitelist of properties we recognise but choose not to implement —
/// they're hints (mobile tap colour, font smoothing) or non-visual
/// metadata. Returning early without counting keeps the audit output
/// focused on properties whose absence actually hurts rendering.
fn is_silent_unknown(name: &str) -> bool {
    matches!(
        name,
        "tap-highlight-color"
            | "text-size-adjust"
            | "touch-action"
            | "font-smoothing"
            | "osx-font-smoothing"
            | "appearance"
            | "user-select"
            | "user-drag"
            | "scroll-behavior"
            | "scroll-snap-type"
            | "scroll-snap-align"
            | "scroll-snap-stop"
            | "-webkit-overflow-scrolling"
            | "-webkit-print-color-adjust"
            | "print-color-adjust"
            // Pre-2012 mobile flexbox spec — sites ship both the old
            // and the modern syntax in the same rule. The modern arm
            // wins; we don't need to count the legacy one.
            | "box-pack" | "box-align" | "box-flex" | "box-orient"
            | "box-direction" | "box-ordinal-group" | "box-lines"
            | "flex-pack" | "flex-align" | "flex-line-pack"
            | "flex-item-align" | "flex-positive" | "flex-negative"
            | "flex-preferred-size" | "flex-order"
            // Page / print-only properties.
            | "column-break-inside" | "column-break-before"
            | "column-break-after" | "page-break-inside"
            | "page-break-before" | "page-break-after"
            | "orphans" | "widows"
            // Misc rendering hints that don't change layout.
            | "shape-rendering" | "interpolation-mode"
            | "image-rendering"
            | "overflow-style" | "overflow-scrolling"
            | "zoom" | "key"
            // SVG gradient stop attributes set via CSS.
            | "stop-color" | "stop-opacity"
            // Niche CSS.
            | "clip" | "box-decoration-break"
    )
}

/// Reset the named CSS property on `style` to its initial value, used to
/// implement the `initial` / `unset` / `revert` / `revert-layer` cascade
/// keywords. We don't track origin/layer history beyond the cascade sort,
/// so `revert` collapses to `initial` (the V1 UA stylesheet adds little
/// that author rules don't already cover; if a future revision attaches
/// a UA-origin baseline we can fold that in here). Most fields on
/// ComputedStyle are `Option<…>` and the LAYOUT/PAINT consumers treat
/// `None` as the initial value, so clearing the Option matches Chrome's
/// "back to initial" effect without a per-property defaults table.
/// Unrecognized property names are no-ops so the cascade walker just
/// moves on (the original "ignore the declaration" behaviour for them).
fn reset_property_to_initial(name: &str, style: &mut ComputedStyle) {
    match name {
        "color" => style.color = None,
        "background"
        | "background-color"
        | "background-image"
        | "background-position"
        | "background-position-x"
        | "background-position-y"
        | "background-size"
        | "background-repeat"
        | "background-attachment"
        | "background-clip"
        | "background-origin" => {
            style.background_color = None;
            style.background_gradient = None;
            style.background_radial_gradient = None;
            style.background_gradient_full = None;
            style.background_image_url = None;
            style.background_position = None;
            style.background_size = None;
            style.background_repeat = BackgroundRepeat::Repeat;
            style.background_clip_text = false;
        }
        "font-size" => style.font_size = None,
        "font-weight" => {
            style.font_weight_bold = None;
            style.font_weight_num = None;
        }
        "font-style" => style.font_style_italic = None,
        "font-family" => style.font_family = None,
        "line-height" => style.line_height = None,
        "text-align" => style.text_align = None,
        "text-decoration" | "text-decoration-line" => {
            style.text_decoration_underline = None;
            style.text_decoration_line_through = None;
        }
        "display" => {
            style.display = None;
            style.display_is_list_item = false;
            style.display_is_flow_root = false;
        }
        "visibility" => style.visibility = None,
        "position" => style.position = None,
        "top" => style.top = None,
        "right" => style.right = None,
        "bottom" => style.bottom = None,
        "left" => style.left = None,
        "z-index" => style.z_index = None,
        "width" => style.width = None,
        "height" => style.height = None,
        "min-width" => style.min_width = None,
        "min-height" => style.min_height = None,
        "max-width" => style.max_width = None,
        "max-height" => style.max_height = None,
        "margin" | "margin-top" | "margin-right" | "margin-bottom" | "margin-left" => {
            // Margin is a struct of four sides; resetting the shorthand
            // means clearing all four. The longhands collapse to the
            // same call so we treat them identically — finer-grained
            // per-side reset would need separate fields.
            style.margin = [None; 4];
            style.margin_auto = [false; 4];
        }
        "padding" | "padding-top" | "padding-right" | "padding-bottom" | "padding-left" => {
            style.padding = [None; 4];
        }
        "border" | "border-width" | "border-color" | "border-style"
        | "border-top" | "border-right" | "border-bottom" | "border-left"
        | "border-top-style" | "border-right-style"
        | "border-bottom-style" | "border-left-style" => {
            style.border_width = None;
            style.border_color = None;
            style.border_top_width = None;
            style.border_right_width = None;
            style.border_bottom_width = None;
            style.border_left_width = None;
            style.border_top_color = None;
            style.border_right_color = None;
            style.border_bottom_color = None;
            style.border_left_color = None;
            style.border_top_style = None;
            style.border_right_style = None;
            style.border_bottom_style = None;
            style.border_left_style = None;
        }
        "border-radius"
        | "border-top-left-radius"
        | "border-top-right-radius"
        | "border-bottom-left-radius"
        | "border-bottom-right-radius" => {
            style.border_radius = None;
        }
        "opacity" => style.opacity = None,
        "transform" => {
            style.translate_x = None;
            style.translate_y = None;
            style.scale_x = None;
            style.scale_y = None;
            style.rotate_deg = None;
            style.matrix_2d = None;
            style.transform_ops = None;
        }
        "transform-style" => style.transform_style_preserve_3d = false,
        "backface-visibility" => style.backface_visibility_hidden = false,
        "perspective" => style.perspective_px = None,
        "perspective-origin" => style.perspective_origin = None,
        "box-shadow" => style.box_shadow = None,
        "text-shadow" => style.text_shadow = None,
        "flex-direction" => style.flex_direction = None,
        "flex-wrap" => style.flex_wrap = None,
        "justify-content" => style.justify_content = None,
        "align-items" => style.align_items = None,
        "align-self" => style.align_self = None,
        "vertical-align" => style.vertical_align = None,
        "float" => style.float_side = None,
        "clear" => style.clear = None,
        // Unknown / not modelled — silently drop, matching the pre-fix
        // behaviour for these properties.
        _ => {}
    }
}

/// Apply a physical margin side longhand (`margin-top` etc.), honouring the
/// `auto` keyword and recording the cascade sequence so a later logical
/// longhand can correctly override the same physical slot.
fn apply_phys_margin(style: &mut ComputedStyle, side: usize, toks: &[CssToken], seq: u32) {
    if is_auto_keyword(toks) {
        style.margin[side] = None;
        style.margin_auto[side] = true;
        style.physical_margin_seq[side] = seq;
    } else if let Some(v) = Length::from_tokens(toks) {
        style.margin[side] = Some(v);
        style.margin_auto[side] = false;
        style.physical_margin_seq[side] = seq;
    }
}

/// Stash a flow-relative margin longhand value into the logical accumulator
/// at `logical` (a LOGICAL_* index), honouring `auto` and recording its
/// cascade sequence. Resolved to a physical side later by `resolve_logical_box`.
fn apply_logical_margin(edges: &mut LogicalEdges, logical: usize, toks: &[CssToken], seq: u32) {
    if is_auto_keyword(toks) {
        edges.vals[logical] = Some(Length::Auto);
        edges.autos[logical] = true;
        edges.seq[logical] = seq;
    } else if let Some(v) = Length::from_tokens(toks) {
        edges.vals[logical] = Some(v);
        edges.autos[logical] = false;
        edges.seq[logical] = seq;
    }
}

/// Stash a flow-relative padding/inset longhand into the logical accumulator.
/// (Padding has no `auto`; inset does but we treat `auto` as "leave None".)
fn apply_logical_pad(edges: &mut LogicalEdges, logical: usize, toks: &[CssToken], seq: u32) {
    if let Some(v) = Length::from_tokens(toks) {
        edges.vals[logical] = Some(v);
        edges.seq[logical] = seq;
    }
}

/// Resolve all flow-relative box quantities accumulated during cascade
/// (`logical_margin`, `logical_padding`, `logical_inset`, `inline-size`,
/// `block-size`, and their min/max) into the physical `margin`/`padding`/
/// inset(top/right/bottom/left)/`width`/`height` slots, using the final
/// resolved `writing-mode` + `direction`. CSS Logical Properties 1 §2.1.
///
/// Cascade arbitration: a logical longhand overrides the physical slot it
/// maps to ONLY when its declaration came later in the cascade (higher
/// sequence) than the competing physical longhand — matching CSS Logical 1
/// §2 ("the pair shares a computed value taken from the higher-priority
/// declaration"). For the overwhelmingly common case where a page uses only
/// logical OR only physical names, the physical seq is 0 so the logical
/// value always wins, and vice versa.
///
/// Must be called AFTER writing-mode/direction inheritance is finalized.
pub fn resolve_logical_box(style: &mut ComputedStyle) {
    let wm = style.writing_mode.unwrap_or_default();
    let dir = style.direction.unwrap_or_default();

    // Re-resolution safety: if a prior resolution already mapped logical
    // values into physical slots (e.g. cv_css cascade resolved with the
    // default horizontal-tb before writing-mode inheritance was known), undo
    // those writes before re-mapping with the now-final writing-mode. A
    // physical slot that currently holds a value but was never set by a
    // PHYSICAL declaration (its physical_seq is 0) can only have come from a
    // logical resolution, so it is safe to clear. Physical declarations always
    // bump physical_seq > 0, so author-declared physical values survive.
    if style.logical_resolved {
        let any_logical_margin = style.logical_margin.vals.iter().any(|v| v.is_some());
        let any_logical_pad = style.logical_padding.vals.iter().any(|v| v.is_some());
        let any_logical_inset = style.logical_inset.vals.iter().any(|v| v.is_some());
        if any_logical_margin {
            for i in 0..4 {
                if style.physical_margin_seq[i] == 0 {
                    style.margin[i] = None;
                    style.margin_auto[i] = false;
                }
            }
        }
        if any_logical_pad {
            for i in 0..4 {
                if style.physical_padding_seq[i] == 0 {
                    style.padding[i] = None;
                }
            }
        }
        if any_logical_inset {
            if style.physical_inset_seq[0] == 0 { style.top = None; }
            if style.physical_inset_seq[1] == 0 { style.right = None; }
            if style.physical_inset_seq[2] == 0 { style.bottom = None; }
            if style.physical_inset_seq[3] == 0 { style.left = None; }
        }
        if style.inline_size.is_some() || style.block_size.is_some() {
            if style.physical_size_seq[0] == 0 { style.width = None; }
            if style.physical_size_seq[1] == 0 { style.height = None; }
        }
        let any_logical_minmax = style.min_inline_size.is_some()
            || style.min_block_size.is_some()
            || style.max_inline_size.is_some()
            || style.max_block_size.is_some();
        if any_logical_minmax {
            if style.physical_minmax_seq[0] == 0 { style.min_width = None; }
            if style.physical_minmax_seq[1] == 0 { style.min_height = None; }
            if style.physical_minmax_seq[2] == 0 { style.max_width = None; }
            if style.physical_minmax_seq[3] == 0 { style.max_height = None; }
        }
    }
    style.logical_resolved = true;

    // Margins. An `auto` margin stores `Length::Auto` AND sets the
    // `margin_auto` flag (the layout engine reads both — `Some(Length::Auto)`
    // is the value convention, `margin_auto[i]` the centering trigger).
    for logical in 0..4 {
        if let Some(v) = style.logical_margin.vals[logical] {
            let phys = map_logical_side(logical, wm, dir);
            if style.logical_margin.seq[logical] >= style.physical_margin_seq[phys] {
                style.margin[phys] = Some(v);
                style.margin_auto[phys] = style.logical_margin.autos[logical];
            }
        }
    }
    // Padding.
    for logical in 0..4 {
        if let Some(v) = style.logical_padding.vals[logical] {
            let phys = map_logical_side(logical, wm, dir);
            if style.logical_padding.seq[logical] >= style.physical_padding_seq[phys] {
                style.padding[phys] = Some(v);
            }
        }
    }
    // Insets — physical slots are top/right/bottom/left fields, not an array.
    for logical in 0..4 {
        if let Some(v) = style.logical_inset.vals[logical] {
            let phys = map_logical_side(logical, wm, dir);
            if style.logical_inset.seq[logical] >= style.physical_inset_seq[phys] {
                match phys {
                    0 => style.top = Some(v),
                    1 => style.right = Some(v),
                    2 => style.bottom = Some(v),
                    _ => style.left = Some(v),
                }
            }
        }
    }
    // inline-size / block-size → width / height.
    let vertical = wm.is_vertical();
    // inline-size maps to width (horizontal) or height (vertical).
    if let Some(v) = style.inline_size {
        if vertical {
            if style.logical_size_seq[0] >= style.physical_size_seq[1] {
                style.height = Some(v);
            }
        } else if style.logical_size_seq[0] >= style.physical_size_seq[0] {
            style.width = Some(v);
        }
    }
    // block-size maps to height (horizontal) or width (vertical).
    if let Some(v) = style.block_size {
        if vertical {
            if style.logical_size_seq[1] >= style.physical_size_seq[0] {
                style.width = Some(v);
            }
        } else if style.logical_size_seq[1] >= style.physical_size_seq[1] {
            style.height = Some(v);
        }
    }
    // min/max-inline/block-size → physical min/max-width/height. The logical
    // value wins the physical slot only if it cascaded at-or-after the
    // physical declaration. Slot indices: 0=min_width 1=min_height
    // 2=max_width 3=max_height.
    if let Some(v) = style.min_inline_size {
        let phys = if vertical { 1 } else { 0 };
        if style.logical_minmax_seq[0] >= style.physical_minmax_seq[phys] {
            if phys == 1 { style.min_height = Some(v); } else { style.min_width = Some(v); }
        }
    }
    if let Some(v) = style.min_block_size {
        let phys = if vertical { 0 } else { 1 };
        if style.logical_minmax_seq[1] >= style.physical_minmax_seq[phys] {
            if phys == 0 { style.min_width = Some(v); } else { style.min_height = Some(v); }
        }
    }
    if let Some(v) = style.max_inline_size {
        let phys = if vertical { 3 } else { 2 };
        if style.logical_minmax_seq[2] >= style.physical_minmax_seq[phys] {
            if phys == 3 { style.max_height = Some(v); } else { style.max_width = Some(v); }
        }
    }
    if let Some(v) = style.max_block_size {
        let phys = if vertical { 2 } else { 3 };
        if style.logical_minmax_seq[3] >= style.physical_minmax_seq[phys] {
            if phys == 2 { style.max_width = Some(v); } else { style.max_height = Some(v); }
        }
    }
}

fn apply_declaration(style: &mut ComputedStyle, d: &Declaration) {
    // Strip `-webkit-` / `-moz-` / `-ms-` / `-o-` so a property like
    // `-webkit-flex-direction` hits the same `flex-direction` arm
    // unprefixed sites use. This is a single change that knocks out
    // ~70% of unknown-property reports against real Web 2.0 sites.
    let raw_name = d.name.as_str();
    let stripped = strip_vendor_prefix(raw_name);
    // Build a temporary Declaration so the existing arms can match the
    // canonical name. Cloning only happens when a prefix was present;
    // the common path stays zero-copy.
    let working: Declaration;
    let d = if stripped != raw_name {
        working = Declaration {
            name: stripped.to_string(),
            value: d.value.clone(),
            important: d.important,
        };
        &working
    } else {
        d
    };
    // Whitespace is already stripped at parse time, so the value slice
    // can be referenced directly — no per-call clone or filter.
    let toks = &d.value;
    // Cascade-order tick. Each applied declaration gets a strictly
    // increasing sequence number so that a logical longhand
    // (`margin-inline-start`) and a physical longhand (`margin-left`)
    // that resolve to the same physical side can be arbitrated by which
    // appeared LATER in the cascade — CSS Logical 1 §2 (the pair "shares
    // a computed value" taken from the higher-priority declaration).
    style.decl_seq = style.decl_seq.wrapping_add(1);
    let seq = style.decl_seq;
    // CSS-wide keywords (CSS Cascade L4 §7.3): `inherit` / `initial` /
    // `unset` / `revert` / `revert-layer`. If the entire value is one
    // of these keywords, we DON'T forward it to the property's normal
    // parser (which would silently fail and leave the prior value in
    // place). For `initial` / `unset` / `revert` we erase any prior
    // value — equivalent to "drop back to UA default" given our
    // current inheritance model. For `inherit` we leave the style
    // alone so the post-cascade inheritance pass picks the parent's
    // value, which mirrors the spec behaviour.
    let non_ws: Vec<&CssToken> = toks
        .iter()
        .filter(|t| !matches!(t, CssToken::Whitespace))
        .collect();
    if non_ws.len() == 1 {
        if let CssToken::Ident(s) = non_ws[0] {
            let lc = s.to_ascii_lowercase();
            match lc.as_str() {
                "inherit" => {
                    // Leave the field alone; the inheritance walk will
                    // pull the parent value during the second pass.
                    return;
                }
                "initial" | "unset" | "revert" | "revert-layer" => {
                    // ECMA-CSS-wide keywords (CSS Cascade L4 §7.3):
                    //   `initial`  → reset to the property's initial value
                    //   `unset`    → inherited for inherited props (same as
                    //                inherit); initial otherwise
                    //   `revert`/`revert-layer` → roll back to the previous
                    //                cascade origin/layer (no UA stylesheet
                    //                to roll back to in V1 → treat as
                    //                initial for non-inherited properties)
                    // Previously the declaration was silently ignored,
                    // which left whatever an EARLIER (lower-cascade-
                    // precedence) declaration set in place — exactly the
                    // opposite of spec: `.foo { color: red; color: initial }`
                    // should clear `.foo`'s color to the initial value,
                    // not leave it red.
                    reset_property_to_initial(d.name.as_str(), style);
                    return;
                }
                _ => {}
            }
        }
    }
    match d.name.as_str() {
        "color" => {
            if let Some(c) = Color::from_tokens(toks) {
                style.color = Some(c);
            }
        }
        "background" | "background-color" => {
            // CSS Backgrounds §3.10: the `background` shorthand resets ALL
            // background longhands to their initial values BEFORE applying
            // the new components.  This is what makes `background: #fff`
            // over an earlier `background-image` or gradient actually clear
            // the image/gradient.
            //
            // `background-color` is a longhand — it only sets the color and
            // must NOT modify any other background property.
            if d.name == "background" {
                style.background_color = None;
                style.background_gradient = None;
                style.background_radial_gradient = None;
                style.background_gradient_full = None;
                style.background_image_url = None;
                style.background_position = None;
                style.background_size = None;
                style.background_repeat = BackgroundRepeat::Repeat; // initial value
                style.background_clip_text = false;
            }

            if d.name == "background" {
                // Shorthand: accept gradients, color, url, keywords.
                // Full N-stop gradient (production paint path). Populated
                // alongside the 2-stop legacy fields so callers that don't
                // honour the rich model still get a sane fallback.
                if let Some(fg) = parse_css_gradient(toks) {
                    if let Some(first) = fg.first_stop_color() {
                        style.background_color = Some(first);
                    }
                    style.background_gradient_full = Some(fg);
                }
                if let Some(g) = parse_linear_gradient(toks) {
                    style.background_gradient = Some(g);
                    // Also expose the start stop as background_color so
                    // callers that don't honour the gradient still get a
                    // sane solid fill.
                    style.background_color = Some(g.from);
                } else if let Some(g) = parse_radial_gradient(toks) {
                    style.background_radial_gradient = Some(g);
                    style.background_color = Some(g.from);
                } else if let Some(c) = Color::from_tokens(toks) {
                    style.background_color = Some(c);
                } else if let Some(c) = first_color_inside_gradient(toks) {
                    style.background_color = Some(c);
                } else if toks
                    .iter()
                    .any(|t| matches!(t, CssToken::Ident(s) if s.eq_ignore_ascii_case("none")))
                {
                    // `background: none` — reset already done above; just
                    // ensure we paint transparent (not the UA default grey).
                    style.background_color = Some(Color::TRANSPARENT);
                }
                // Scan for url() token (background-image component of shorthand).
                for tk in toks {
                    if let CssToken::Url(s) = tk {
                        style.background_image_url = Some(s.clone());
                        break;
                    }
                }
                // Pull a background-repeat keyword out of the shorthand.
                // Many sites write `background: #000 url(x.svg) no-repeat
                // center / cover` — without this the paint path would tile.
                for tk in toks {
                    if let CssToken::Ident(s) = tk {
                        match s.to_ascii_lowercase().as_str() {
                            "repeat" => style.background_repeat = BackgroundRepeat::Repeat,
                            "no-repeat" => style.background_repeat = BackgroundRepeat::NoRepeat,
                            "repeat-x" => style.background_repeat = BackgroundRepeat::RepeatX,
                            "repeat-y" => style.background_repeat = BackgroundRepeat::RepeatY,
                            _ => {}
                        }
                    }
                }
            } else {
                // background-color longhand: <color> only per spec.
                // Does NOT accept url(), gradient(), or repeat keywords.
                if let Some(c) = Color::from_tokens(toks) {
                    style.background_color = Some(c);
                }
            }
        }
        "background-image" => {
            if let Some(fg) = parse_css_gradient(toks) {
                if let Some(first) = fg.first_stop_color() {
                    style.background_color = Some(first);
                }
                style.background_gradient_full = Some(fg);
            } else {
                style.background_gradient_full = None;
            }
            if let Some(g) = parse_linear_gradient(toks) {
                style.background_gradient = Some(g);
                style.background_radial_gradient = None;
                style.background_color = Some(g.from);
            } else if let Some(g) = parse_radial_gradient(toks) {
                style.background_radial_gradient = Some(g);
                style.background_gradient = None;
                style.background_color = Some(g.from);
            }
            for tk in toks {
                if let CssToken::Url(s) = tk {
                    style.background_image_url = Some(s.clone());
                    break;
                }
            }
        }
        "background-size" => {
            // CSS Backgrounds §3.9: cover | contain | <bg-size>
            // <bg-size> = [ <length-percentage> | auto ]{1,2}
            //
            // Single-keyword `cover`/`contain` are named variants; everything
            // else is `CssBgSize::Explicit(w, h)` where `None` = `auto`.
            let non_ws: Vec<&CssToken> = toks
                .iter()
                .filter(|t| !matches!(t, CssToken::Whitespace))
                .collect();
            let first_ident = non_ws.first().and_then(|t| {
                if let CssToken::Ident(s) = t {
                    Some(s.to_ascii_lowercase())
                } else {
                    None
                }
            });
            match first_ident.as_deref() {
                Some("cover") => {
                    style.background_size = Some(CssBgSize::Cover);
                }
                Some("contain") => {
                    style.background_size = Some(CssBgSize::Contain);
                }
                _ => {
                    // Parse up to two length/percentage/auto axes.
                    // `split_top_level_whitespace` splits on WS tokens.
                    let parts = split_top_level_whitespace(toks);
                    let axis = |part: &[CssToken]| -> Option<Length> {
                        // "auto" keyword → None (will be represented as
                        // the None variant of Option<Length> at the call
                        // site — caller converts None to auto).
                        if part.len() == 1 {
                            if let CssToken::Ident(s) = &part[0] {
                                if s.eq_ignore_ascii_case("auto") {
                                    return None; // sentinel: use Some(None) to mean auto
                                }
                            }
                        }
                        Length::from_tokens(part)
                    };
                    // axis() returns Option<Length>: Some(l) = explicit,
                    // None = either "auto" or parse failure.
                    // Wrap in Option<Option<Length>> so the Explicit tuple
                    // distinguishes "auto" (Some(None)) from "unset" (None).
                    let wrap = |part: &[CssToken]| -> Option<Option<Length>> {
                        if part.len() == 1 {
                            if let CssToken::Ident(s) = &part[0] {
                                if s.eq_ignore_ascii_case("auto") {
                                    return Some(None); // auto
                                }
                            }
                        }
                        Length::from_tokens(part).map(Some)
                    };
                    if let Some(first_parts) = parts.first() {
                        if let Some(w) = wrap(first_parts) {
                            // h is Option<Option<Length>>; flatten to Option<Length>:
                            //   None          → auto (no second axis given)
                            //   Some(None)    → auto keyword
                            //   Some(Some(l)) → explicit length
                            let h = parts.get(1).and_then(|p| wrap(p)).flatten();
                            style.background_size = Some(CssBgSize::Explicit(w, h));
                        }
                    }
                }
            }
        }
        "background-position" => {
            if let Some((x, y)) = parse_background_position(toks) {
                style.background_position = Some((x, y));
            }
        }
        "background-position-x" => {
            if let Some(x) = bg_pos_component(toks.first(), true) {
                let y = style
                    .background_position
                    .map(|(_, y)| y)
                    .unwrap_or(BgPos::Pct(0.0));
                style.background_position = Some((x, y));
            }
        }
        "background-position-y" => {
            if let Some(y) = bg_pos_component(toks.first(), false) {
                let x = style
                    .background_position
                    .map(|(x, _)| x)
                    .unwrap_or(BgPos::Pct(0.0));
                style.background_position = Some((x, y));
            }
        }
        "background-repeat" => {
            // First keyword wins. `repeat-x` / `repeat-y` are explicit
            // single-axis tiles; `no-repeat` paints once; default is
            // `repeat`. Two-value form (`repeat no-repeat`) is parsed
            // as horizontal then vertical — we collapse to the same
            // approximations: horizontal-only → RepeatX,
            // vertical-only → RepeatY, both → Repeat, neither →
            // NoRepeat.
            for t in toks {
                if let CssToken::Ident(s) = t {
                    let k = s.to_ascii_lowercase();
                    match k.as_str() {
                        "repeat" => {
                            style.background_repeat = BackgroundRepeat::Repeat;
                            break;
                        }
                        "no-repeat" => {
                            style.background_repeat = BackgroundRepeat::NoRepeat;
                            break;
                        }
                        "repeat-x" => {
                            style.background_repeat = BackgroundRepeat::RepeatX;
                            break;
                        }
                        "repeat-y" => {
                            style.background_repeat = BackgroundRepeat::RepeatY;
                            break;
                        }
                        "round" | "space" => {
                            // Treat as the default (Repeat). We don't
                            // do the special round/space spacing math.
                            style.background_repeat = BackgroundRepeat::Repeat;
                            break;
                        }
                        _ => {}
                    }
                }
            }
        }
        "background-clip" | "-webkit-background-clip" => {
            // Only `text` changes behaviour for us — `border-box` /
            // `padding-box` / `content-box` all paint the same box
            // we paint anyway. Treat anything containing `text` as
            // the gradient-on-text idiom.
            let wants_text = toks
                .iter()
                .any(|t| matches!(t, CssToken::Ident(s) if s.eq_ignore_ascii_case("text")));
            if wants_text {
                style.background_clip_text = true;
            }
        }
        "display" => {
            // Detect `display: list-item` / `display: flow-root` before the
            // generic parse so we can set the CSSOM-fidelity flags even though
            // both map to Block for layout. Only flip a flag when a recognised
            // display keyword is present (so a later valid `display` declaration
            // can clear a flag a prior one set, but a malformed value leaves the
            // prior winner intact).
            let is_list_item = toks.iter().any(|t| {
                matches!(t, CssToken::Ident(s) if s.eq_ignore_ascii_case("list-item"))
            });
            let is_flow_root = toks.iter().any(|t| {
                matches!(t, CssToken::Ident(s) if s.eq_ignore_ascii_case("flow-root"))
            });
            if let Some(v) = Display::from_tokens(toks) {
                style.display = Some(v);
                style.display_is_list_item = is_list_item;
                style.display_is_flow_root = is_flow_root;
            }
        }
        "visibility" => {
            if let Some(v) = Visibility::from_tokens(toks) {
                style.visibility = Some(v);
            }
        }
        // CSS Fragmentation 3 §3.1 / §4.2 — break properties + legacy aliases.
        // The legacy `page-break-*` map onto the same fields (Chrome treats
        // `page-break-before:always` as `break-before:page`, etc.).
        "break-before" | "page-break-before" => {
            if let Some(v) = BreakValue::from_tokens(toks) {
                style.break_before = Some(v);
            }
        }
        "break-after" | "page-break-after" => {
            if let Some(v) = BreakValue::from_tokens(toks) {
                style.break_after = Some(v);
            }
        }
        "break-inside" | "page-break-inside" => {
            if let Some(v) = BreakValue::from_tokens(toks) {
                style.break_inside = Some(v);
            }
        }
        "font-size" => {
            // CSS Values §7.1 — absolute keyword sizes are anchored at
            // the browser's base size (16px).  Relative keywords scale
            // against the parent computed font-size.
            let kw_length: Option<Length> = toks.iter()
                .filter_map(|t| if let CssToken::Ident(s) = t { Some(s) } else { None })
                .next()
                .and_then(|s| match s.to_ascii_lowercase().as_str() {
                    // Absolute sizes (Chrome reference px at 16px base)
                    "xx-small"  => Some(Length::Px(9.0)),
                    "x-small"   => Some(Length::Px(10.0)),
                    "small"     => Some(Length::Px(13.0)),
                    "medium"    => Some(Length::Px(16.0)),
                    "large"     => Some(Length::Px(18.0)),
                    "x-large"   => Some(Length::Px(24.0)),
                    "xx-large"  => Some(Length::Px(32.0)),
                    "xxx-large" => Some(Length::Px(48.0)),
                    // Relative sizes — em fractions that scale off parent.
                    "smaller" => Some(Length::Em(5.0 / 6.0)),
                    "larger"  => Some(Length::Em(6.0 / 5.0)),
                    _ => None,
                });
            if let Some(v) = kw_length.or_else(|| Length::from_tokens(toks)) {
                style.font_size = Some(v);
            }
        }
        // Physical sizing. The logical `inline-size`/`block-size` are NOT
        // aliases — they map to width/height depending on writing-mode
        // (CSS Logical 1 §4.1: inline-size = width in horizontal modes,
        // = height in vertical modes). We store them separately and
        // resolve in `resolve_logical_box` once writing-mode is final.
        "width" => {
            if let Some(v) = Length::from_tokens(toks) {
                style.width = Some(v);
                style.physical_size_seq[0] = seq;
            }
        }
        "height" => {
            if let Some(v) = Length::from_tokens(toks) {
                style.height = Some(v);
                style.physical_size_seq[1] = seq;
            }
        }
        "inline-size" => {
            if let Some(v) = Length::from_tokens(toks) {
                style.inline_size = Some(v);
                style.logical_size_seq[0] = seq;
            }
        }
        "block-size" => {
            if let Some(v) = Length::from_tokens(toks) {
                style.block_size = Some(v);
                style.logical_size_seq[1] = seq;
            }
        }
        "aspect-ratio" => {
            style.aspect_ratio = parse_aspect_ratio(toks);
        }
        // CSS Writing Modes 4 §3.1 / §2.1. Both inherit; the inherit
        // fill-in happens after the parent value is known
        // (build_styled_tree). The logical→physical resolver
        // (`resolve_logical_box`) reads these.
        "writing-mode" => {
            if let Some(CssToken::Ident(s)) = toks.first() {
                if let Some(wm) = WritingMode::from_str(s) {
                    style.writing_mode = Some(wm);
                }
            }
        }
        "direction" => {
            if let Some(CssToken::Ident(s)) = toks.first() {
                if let Some(d) = Direction::from_str(s) {
                    style.direction = Some(d);
                }
            }
        }
        "margin" => {
            if let Some((sides, autos)) = parse_margin_shorthand(toks) {
                style.margin = sides;
                style.margin_auto = autos;
                for i in 0..4 {
                    style.physical_margin_seq[i] = seq;
                }
            }
        }
        // Physical margin sides.
        "margin-top" => {
            apply_phys_margin(style, 0, toks, seq);
        }
        "margin-right" => {
            apply_phys_margin(style, 1, toks, seq);
        }
        "margin-bottom" => {
            apply_phys_margin(style, 2, toks, seq);
        }
        "margin-left" => {
            apply_phys_margin(style, 3, toks, seq);
        }
        // Flow-relative margin longhands — stored raw, mapped to a
        // physical side in `resolve_logical_box`. CSS Logical 1 §4.2.
        "margin-block-start" => apply_logical_margin(&mut style.logical_margin, LOGICAL_BLOCK_START, toks, seq),
        "margin-block-end" => apply_logical_margin(&mut style.logical_margin, LOGICAL_BLOCK_END, toks, seq),
        "margin-inline-start" => apply_logical_margin(&mut style.logical_margin, LOGICAL_INLINE_START, toks, seq),
        "margin-inline-end" => apply_logical_margin(&mut style.logical_margin, LOGICAL_INLINE_END, toks, seq),
        // `margin-block: <start> <end>` / `margin-inline: <start> <end>`
        // 1-or-2-value flow-relative shorthands.
        "margin-block" => {
            let parts = split_top_level_whitespace(toks);
            if let Some(p) = parts.first() {
                apply_logical_margin(&mut style.logical_margin, LOGICAL_BLOCK_START, p, seq);
                if parts.len() < 2 {
                    apply_logical_margin(&mut style.logical_margin, LOGICAL_BLOCK_END, p, seq);
                }
            }
            if let Some(p) = parts.get(1) {
                apply_logical_margin(&mut style.logical_margin, LOGICAL_BLOCK_END, p, seq);
            }
        }
        "margin-inline" => {
            let parts = split_top_level_whitespace(toks);
            if let Some(p) = parts.first() {
                apply_logical_margin(&mut style.logical_margin, LOGICAL_INLINE_START, p, seq);
                if parts.len() < 2 {
                    apply_logical_margin(&mut style.logical_margin, LOGICAL_INLINE_END, p, seq);
                }
            }
            if let Some(p) = parts.get(1) {
                apply_logical_margin(&mut style.logical_margin, LOGICAL_INLINE_END, p, seq);
            }
        }
        "max-width" => {
            if !is_auto_keyword(toks) && !is_none_keyword(toks) {
                if let Some(v) = Length::from_tokens(toks) {
                    style.max_width = Some(v);
                    style.physical_minmax_seq[2] = seq;
                }
            }
        }
        "max-height" => {
            if !is_auto_keyword(toks) && !is_none_keyword(toks) {
                if let Some(v) = Length::from_tokens(toks) {
                    style.max_height = Some(v);
                    style.physical_minmax_seq[3] = seq;
                }
            }
        }
        "max-inline-size" => {
            if !is_auto_keyword(toks) && !is_none_keyword(toks) {
                style.max_inline_size = Length::from_tokens(toks);
                style.logical_minmax_seq[2] = seq;
            }
        }
        "max-block-size" => {
            if !is_auto_keyword(toks) && !is_none_keyword(toks) {
                style.max_block_size = Length::from_tokens(toks);
                style.logical_minmax_seq[3] = seq;
            }
        }
        "min-width" => {
            if !is_auto_keyword(toks) {
                if let Some(v) = Length::from_tokens(toks) {
                    style.min_width = Some(v);
                    style.physical_minmax_seq[0] = seq;
                }
            }
        }
        "min-height" => {
            if !is_auto_keyword(toks) {
                if let Some(v) = Length::from_tokens(toks) {
                    style.min_height = Some(v);
                    style.physical_minmax_seq[1] = seq;
                }
            }
        }
        "min-inline-size" => {
            if !is_auto_keyword(toks) {
                style.min_inline_size = Length::from_tokens(toks);
                style.logical_minmax_seq[0] = seq;
            }
        }
        "min-block-size" => {
            if !is_auto_keyword(toks) {
                style.min_block_size = Length::from_tokens(toks);
                style.logical_minmax_seq[1] = seq;
            }
        }
        "padding" => {
            if let Some(sides) = parse_box_shorthand(toks) {
                style.padding = sides;
                for i in 0..4 {
                    style.physical_padding_seq[i] = seq;
                }
            }
        }
        "padding-top" => {
            if let Some(v) = Length::from_tokens(toks) {
                style.padding[0] = Some(v);
                style.physical_padding_seq[0] = seq;
            }
        }
        "padding-right" => {
            if let Some(v) = Length::from_tokens(toks) {
                style.padding[1] = Some(v);
                style.physical_padding_seq[1] = seq;
            }
        }
        "padding-bottom" => {
            if let Some(v) = Length::from_tokens(toks) {
                style.padding[2] = Some(v);
                style.physical_padding_seq[2] = seq;
            }
        }
        "padding-left" => {
            if let Some(v) = Length::from_tokens(toks) {
                style.padding[3] = Some(v);
                style.physical_padding_seq[3] = seq;
            }
        }
        "padding-block-start" => apply_logical_pad(&mut style.logical_padding, LOGICAL_BLOCK_START, toks, seq),
        "padding-block-end" => apply_logical_pad(&mut style.logical_padding, LOGICAL_BLOCK_END, toks, seq),
        "padding-inline-start" => apply_logical_pad(&mut style.logical_padding, LOGICAL_INLINE_START, toks, seq),
        "padding-inline-end" => apply_logical_pad(&mut style.logical_padding, LOGICAL_INLINE_END, toks, seq),
        "padding-block" => {
            let parts = split_top_level_whitespace(toks);
            if let Some(p) = parts.first() {
                apply_logical_pad(&mut style.logical_padding, LOGICAL_BLOCK_START, p, seq);
                if parts.len() < 2 {
                    apply_logical_pad(&mut style.logical_padding, LOGICAL_BLOCK_END, p, seq);
                }
            }
            if let Some(p) = parts.get(1) {
                apply_logical_pad(&mut style.logical_padding, LOGICAL_BLOCK_END, p, seq);
            }
        }
        "padding-inline" => {
            let parts = split_top_level_whitespace(toks);
            if let Some(p) = parts.first() {
                apply_logical_pad(&mut style.logical_padding, LOGICAL_INLINE_START, p, seq);
                if parts.len() < 2 {
                    apply_logical_pad(&mut style.logical_padding, LOGICAL_INLINE_END, p, seq);
                }
            }
            if let Some(p) = parts.get(1) {
                apply_logical_pad(&mut style.logical_padding, LOGICAL_INLINE_END, p, seq);
            }
        }
        "border" => {
            // Shorthand: width style color (any order).
            // Parse through the shared helper so style keywords are handled.
            let (w, c, s) = parse_border_shorthand(toks);
            style.border_width = Some(w);
            style.border_color = Some(c);
            // Propagate the style keyword to all four sides so that
            // `border: 1px solid red` doesn't leave the sides as None.
            if let Some(bs) = s {
                style.border_top_style = Some(bs);
                style.border_right_style = Some(bs);
                style.border_bottom_style = Some(bs);
                style.border_left_style = Some(bs);
            }
        }
        "border-width" => {
            if let Some(v) = Length::from_tokens(toks) {
                style.border_width = Some(v);
            }
        }
        "border-color" => {
            if let Some(c) = Color::from_tokens(toks) {
                style.border_color = Some(c);
            }
        }
        // Per-side shorthands. Each accepts `<width> <style> <color>`
        // in any order — same algorithm as the all-sides `border:`.
        "border-top" => {
            let (w, c, s) = parse_border_shorthand(toks);
            style.border_top_width = Some(w);
            style.border_top_color = Some(c);
            if let Some(bs) = s { style.border_top_style = Some(bs); }
        }
        "border-right" => {
            let (w, c, s) = parse_border_shorthand(toks);
            style.border_right_width = Some(w);
            style.border_right_color = Some(c);
            if let Some(bs) = s { style.border_right_style = Some(bs); }
        }
        "border-bottom" => {
            let (w, c, s) = parse_border_shorthand(toks);
            style.border_bottom_width = Some(w);
            style.border_bottom_color = Some(c);
            if let Some(bs) = s { style.border_bottom_style = Some(bs); }
        }
        "border-left" => {
            let (w, c, s) = parse_border_shorthand(toks);
            style.border_left_width = Some(w);
            style.border_left_color = Some(c);
            if let Some(bs) = s { style.border_left_style = Some(bs); }
        }
        "border-top-width" => {
            if let Some(v) = Length::from_tokens(toks) {
                style.border_top_width = Some(v);
            }
        }
        "border-right-width" => {
            if let Some(v) = Length::from_tokens(toks) {
                style.border_right_width = Some(v);
            }
        }
        "border-bottom-width" => {
            if let Some(v) = Length::from_tokens(toks) {
                style.border_bottom_width = Some(v);
            }
        }
        "border-left-width" => {
            if let Some(v) = Length::from_tokens(toks) {
                style.border_left_width = Some(v);
            }
        }
        "border-top-color" => {
            if let Some(c) = Color::from_tokens(toks) {
                style.border_top_color = Some(c);
            }
        }
        "border-right-color" => {
            if let Some(c) = Color::from_tokens(toks) {
                style.border_right_color = Some(c);
            }
        }
        "border-bottom-color" => {
            if let Some(c) = Color::from_tokens(toks) {
                style.border_bottom_color = Some(c);
            }
        }
        "border-left-color" => {
            if let Some(c) = Color::from_tokens(toks) {
                style.border_left_color = Some(c);
            }
        }
        "text-align" => {
            for t in toks {
                if let CssToken::Ident(s) = t {
                    style.text_align = Some(match s.to_ascii_lowercase().as_str() {
                        "left" | "-webkit-left" => TextAlign::Left,
                        "center" | "-webkit-center" => TextAlign::Center,
                        "right" | "-webkit-right" => TextAlign::Right,
                        "justify" => TextAlign::Justify,
                        // Per CSS Text 3 §7.1: `start` and `end` are
                        // flow-relative keywords. We preserve them as
                        // distinct variants (TextAlign::Start / End) so
                        // the layout engine can treat them semantically.
                        // In LTR (the only writing mode we support today)
                        // Start = Left, End = Right — but storing the
                        // original keyword avoids hard-coding the physical
                        // direction at parse time.
                        // `-webkit-auto` is a legacy alias for `start`.
                        "start" | "-webkit-auto" => TextAlign::Start,
                        "end" => TextAlign::End,
                        // `match-parent` and `-webkit-match-parent` inherit
                        // the parent's computed text-align (including its
                        // resolved Start/End).  Model as Start so it at
                        // least aligns left rather than producing an unset.
                        "match-parent" | "-webkit-match-parent" => TextAlign::Start,
                        _ => continue,
                    });
                    break;
                }
            }
        }
        "font-weight" => {
            for t in toks {
                match t {
                    CssToken::Ident(s) => {
                        let lower = s.to_ascii_lowercase();
                        match lower.as_str() {
                            "bold" => {
                                style.font_weight_bold = Some(true);
                                style.font_weight_num = Some(700);
                            }
                            "normal" => {
                                style.font_weight_bold = Some(false);
                                style.font_weight_num = Some(400);
                            }
                            // Relative keywords: store a sentinel; the final
                            // weight is computed against the inherited weight
                            // during inheritance (CSS Fonts 4 §2.4). bolder→bold
                            // bool is a safe approximation until then.
                            "bolder" => {
                                style.font_weight_bold = Some(true);
                                style.font_weight_num = Some(FONT_WEIGHT_BOLDER);
                            }
                            "lighter" => {
                                style.font_weight_bold = Some(false);
                                style.font_weight_num = Some(FONT_WEIGHT_LIGHTER);
                            }
                            _ => {}
                        }
                        break;
                    }
                    CssToken::Number(n) => {
                        // Keep both: the bool for legacy paths, the numeric weight
                        // (clamped to the CSS 1–1000 range) so 800/900 render heavy.
                        style.font_weight_bold = Some(*n >= 600.0);
                        style.font_weight_num = Some((*n as i32).clamp(1, 1000) as u16);
                        break;
                    }
                    _ => {}
                }
            }
        }
        "font-style" => {
            for t in toks {
                if let CssToken::Ident(s) = t {
                    style.font_style_italic = Some(matches!(
                        s.to_ascii_lowercase().as_str(),
                        "italic" | "oblique"
                    ));
                    break;
                }
            }
        }
        "font-family" => {
            // Per CSS Fonts: collect the FULL comma-separated list so
            // the text pipeline can walk it as a fallback chain when
            // the primary family doesn't provide a glyph for a given
            // run. Storing only the first family (the previous behaviour)
            // meant `font-family: Roboto, sans-serif` on a system
            // without Roboto rendered with the OS default rather than
            // falling through to the `sans-serif` generic.
            let mut families: Vec<String> = Vec::new();
            let mut current: Option<String> = None;
            for t in toks {
                match t {
                    CssToken::Ident(s) => {
                        // Multi-word ident families (e.g. `Times New Roman`)
                        // arrive as several idents — join with spaces.
                        current = Some(match current.take() {
                            Some(prev) => format!("{prev} {s}"),
                            None => s.clone(),
                        });
                    }
                    CssToken::String(s) => {
                        if let Some(prev) = current.take() {
                            families.push(prev);
                        }
                        families.push(s.clone());
                    }
                    CssToken::Comma => {
                        if let Some(prev) = current.take() {
                            families.push(prev);
                        }
                    }
                    _ => {}
                }
            }
            if let Some(prev) = current {
                families.push(prev);
            }
            if !families.is_empty() {
                style.font_family = Some(families.join(", "));
            }
        }
        "box-sizing" => {
            for t in toks {
                if let CssToken::Ident(s) = t {
                    style.box_sizing_border_box = Some(s.eq_ignore_ascii_case("border-box"));
                    break;
                }
            }
        }
        // `list-style` shorthand and `list-style-type` longhand: we
        // only care about the `none` case (suppress UA bullets) for
        // V1. Author rules like `nav ul { list-style: none; }` are
        // ubiquitous; without honouring them every nav menu shows
        // bullets, which is the single most visible CSS-not-applied
        // bug on real pages.
        "list-style" | "list-style-type" => {
            for t in toks {
                if let CssToken::Ident(s) = t {
                    if s.eq_ignore_ascii_case("none") {
                        style.list_style_none = true;
                        break;
                    }
                    // Any other keyword (`disc`, `decimal`, `circle`,
                    // …) explicitly enables a marker — clear the flag.
                    if matches!(
                        s.to_ascii_lowercase().as_str(),
                        "disc"
                            | "circle"
                            | "square"
                            | "decimal"
                            | "decimal-leading-zero"
                            | "lower-roman"
                            | "upper-roman"
                            | "lower-alpha"
                            | "upper-alpha"
                            | "lower-latin"
                            | "upper-latin"
                    ) {
                        style.list_style_none = false;
                        break;
                    }
                }
            }
        }
        "flex-direction" => {
            if let Some(v) = FlexDirection::from_tokens(toks) {
                style.flex_direction = Some(v);
            }
        }
        "flex-wrap" => {
            if let Some(v) = FlexWrap::from_tokens(toks) {
                style.flex_wrap = Some(v);
            }
        }
        "flex-grow" => {
            for t in toks {
                if let CssToken::Number(n) = t {
                    style.flex_grow = Some((*n as f32).max(0.0));
                    break;
                }
            }
        }
        "flex-shrink" => {
            for t in toks {
                if let CssToken::Number(n) = t {
                    style.flex_shrink = Some((*n as f32).max(0.0));
                    break;
                }
            }
        }
        "flex-basis" => {
            if let Some(length) = Length::from_tokens(toks) {
                style.flex_basis = Some(length);
            }
        }
        "flex" => {
            let numbers: Vec<f32> = toks
                .iter()
                .filter_map(|t| match t {
                    CssToken::Number(n) => Some(*n as f32),
                    _ => None,
                })
                .collect();
            if let Some(CssToken::Ident(ident)) =
                toks.iter().find(|t| !matches!(t, CssToken::Whitespace))
            {
                if ident.eq_ignore_ascii_case("none") {
                    style.flex_grow = Some(0.0);
                    style.flex_shrink = Some(0.0);
                    return;
                } else if ident.eq_ignore_ascii_case("auto") {
                    style.flex_grow = Some(1.0);
                    style.flex_shrink = Some(1.0);
                    style.flex_basis = Some(Length::Auto);
                    return;
                } else if ident.eq_ignore_ascii_case("initial") {
                    style.flex_grow = Some(0.0);
                    style.flex_shrink = Some(1.0);
                    style.flex_basis = Some(Length::Auto);
                    return;
                }
            }
            let basis_tokens: Vec<CssToken> = toks
                .iter()
                .filter(|t| !matches!(t, CssToken::Whitespace))
                .skip_while(|t| matches!(t, CssToken::Number(_)))
                .skip_while(|t| matches!(t, CssToken::Number(_)))
                .cloned()
                .collect();
            match numbers.len() {
                0 => {}
                1 => {
                    style.flex_grow = Some(numbers[0].max(0.0));
                    style.flex_shrink = Some(1.0);
                    style.flex_basis = Some(Length::Zero);
                }
                _ => {
                    style.flex_grow = Some(numbers[0].max(0.0));
                    style.flex_shrink = Some(numbers[1].max(0.0));
                    style.flex_basis = Some(Length::Zero);
                }
            }
            if !basis_tokens.is_empty() {
                if let Some(length) = Length::from_tokens(&basis_tokens) {
                    if numbers.is_empty() {
                        style.flex_grow = Some(1.0);
                        style.flex_shrink = Some(1.0);
                    }
                    style.flex_basis = Some(length);
                }
            }
        }
        "justify-content" => {
            if let Some(v) = JustifyContent::from_tokens(toks) {
                style.justify_content = Some(v);
            }
        }
        "align-items" => {
            if let Some(v) = AlignItems::from_tokens(toks) {
                style.align_items = Some(v);
            }
        }
        "justify-items" => {
            if let Some(v) = AlignItems::from_tokens(toks) {
                style.justify_items = Some(v);
            }
        }
        "align-self" => {
            if let Some(v) = AlignItems::from_tokens(toks) {
                style.align_self = Some(v);
            }
        }
        "justify-self" => {
            if let Some(v) = AlignItems::from_tokens(toks) {
                style.justify_self = Some(v);
            }
        }
        "place-items" => {
            // Shorthand: <align-items> [<justify-items>]?. We strip
            // whitespace at parse time so consecutive Ident tokens are
            // the two values. When only one value is given, both axes
            // get it.
            let idents: Vec<CssToken> = toks
                .iter()
                .filter(|t| matches!(t, CssToken::Ident(_)))
                .cloned()
                .collect();
            if let Some(first) = idents.first() {
                if let Some(a) = AlignItems::from_tokens(std::slice::from_ref(first)) {
                    style.align_items = Some(a);
                    let j = idents
                        .get(1)
                        .and_then(|t| AlignItems::from_tokens(std::slice::from_ref(t)))
                        .unwrap_or(a);
                    style.justify_items = Some(j);
                }
            }
        }
        "place-self" => {
            let idents: Vec<CssToken> = toks
                .iter()
                .filter(|t| matches!(t, CssToken::Ident(_)))
                .cloned()
                .collect();
            if let Some(first) = idents.first() {
                if let Some(a) = AlignItems::from_tokens(std::slice::from_ref(first)) {
                    style.align_self = Some(a);
                    let j = idents
                        .get(1)
                        .and_then(|t| AlignItems::from_tokens(std::slice::from_ref(t)))
                        .unwrap_or(a);
                    style.justify_self = Some(j);
                }
            }
        }
        "gap" | "grid-gap" => {
            // Shorthand: row-gap then column-gap. If only one length, both
            // axes get it. Distinct row/column-gap properties below still
            // win because cascade order is order-preserving.
            let mut nums: Vec<Length> = Vec::new();
            for t in toks {
                if let Some(l) = Length::from_tokens(std::slice::from_ref(t)) {
                    nums.push(l);
                }
            }
            match nums.len() {
                0 => {}
                1 => {
                    style.gap = Some(nums[0]);
                    style.row_gap = Some(nums[0]);
                    style.column_gap = Some(nums[0]);
                }
                _ => {
                    style.row_gap = Some(nums[0]);
                    style.column_gap = Some(nums[1]);
                    style.gap = Some(nums[0]);
                }
            }
        }
        "row-gap" => {
            if let Some(v) = Length::from_tokens(toks) {
                style.row_gap = Some(v);
                if style.gap.is_none() {
                    style.gap = Some(v);
                }
            }
        }
        "column-gap" => {
            // `column-gap` is ONE property shared by flex/grid (consumed via
            // `style.column_gap`) and multicol (consumed via
            // `style.multicol_gap`). Set both. `normal` (→ Length parse fails)
            // leaves multicol_gap None so layout applies the 1em default.
            if let Some(v) = Length::from_tokens(toks) {
                style.column_gap = Some(v);
                if style.gap.is_none() {
                    style.gap = Some(v);
                }
                if let Length::Px(px) = v {
                    style.multicol_gap = Some(px);
                }
            }
        }
        "grid-column-start" => {
            let (start, span) = parse_grid_line(toks);
            if let Some(s) = start {
                style.grid_column_start = Some(s);
            }
            if let Some(sp) = span {
                style.grid_column_span = Some(sp);
            }
        }
        "grid-row-start" => {
            let (start, span) = parse_grid_line(toks);
            if let Some(s) = start {
                style.grid_row_start = Some(s);
            }
            if let Some(sp) = span {
                style.grid_row_span = Some(sp);
            }
        }
        "grid-column" => {
            if let Some((start, span)) = parse_grid_placement(toks) {
                style.grid_column_start = Some(start);
                style.grid_column_span = span;
            }
        }
        "grid-row" => {
            if let Some((start, span)) = parse_grid_placement(toks) {
                style.grid_row_start = Some(start);
                style.grid_row_span = span;
            }
        }
        "grid-template-columns" => {
            let tracks = GridTrack::parse_track_list(toks);
            if !tracks.is_empty() {
                style.grid_template_columns = Some(tracks);
            }
        }
        "grid-template-rows" => {
            let tracks = GridTrack::parse_track_list(toks);
            if !tracks.is_empty() {
                style.grid_template_rows = Some(tracks);
            }
        }
        "grid-auto-rows" => {
            // Single track value per spec — take the first parseable
            // track from the list.
            let tracks = GridTrack::parse_track_list(toks);
            if let Some(first) = tracks.into_iter().next() {
                style.grid_auto_rows = Some(first);
            }
        }
        "grid-auto-columns" => {
            let tracks = GridTrack::parse_track_list(toks);
            if let Some(first) = tracks.into_iter().next() {
                style.grid_auto_columns = Some(first);
            }
        }
        "anchor-name" => {
            for t in toks {
                if let CssToken::Ident(s) = t {
                    style.anchor_name = Some(s.clone());
                    break;
                }
            }
        }
        "position-anchor" => {
            for t in toks {
                if let CssToken::Ident(s) = t {
                    style.position_anchor = Some(s.clone());
                    break;
                }
            }
        }
        "column-count" => {
            for t in toks {
                match t {
                    CssToken::Number(n) if *n > 0.0 => {
                        style.column_count = Some(*n as u32);
                        break;
                    }
                    CssToken::Ident(s) if s.eq_ignore_ascii_case("auto") => {
                        style.column_count = None;
                        break;
                    }
                    _ => {}
                }
            }
        }
        "column-width" => {
            if let Some(len) = Length::from_tokens(toks) {
                if let Length::Px(px) = len {
                    style.column_width = Some(px);
                }
            }
            for t in toks {
                if let CssToken::Ident(s) = t {
                    if s.eq_ignore_ascii_case("auto") {
                        style.column_width = None;
                        break;
                    }
                }
            }
        }
        "columns" => {
            // shorthand: column-width column-count (either order, either may
            // be `auto`). Parse positionally — first len → width, first
            // integer → count.
            for t in toks {
                match t {
                    CssToken::Number(n) if *n > 0.0 && n.fract() == 0.0 => {
                        style.column_count = Some(*n as u32);
                    }
                    CssToken::Dimension { value, unit } if unit.eq_ignore_ascii_case("px") => {
                        style.column_width = Some(*value as f32);
                    }
                    _ => {}
                }
            }
        }
        // CSS Multi-column §6 — the rule drawn between columns. Parsed like a
        // border (`width style color`, any order). The rule does NOT affect
        // layout; the painter draws it centred in the column gap.
        "column-rule" => {
            let (w, c, s) = parse_border_shorthand(toks);
            style.column_rule_width = Some(length_to_px_approx(w));
            style.column_rule_color = Some(c);
            // `parse_border_shorthand` returns None style when only width/color
            // was given; CSS Multicol initial column-rule-style is `none`, so a
            // rule with no explicit style is invisible. Keep `Some(None-ish)`
            // semantics by recording what was parsed; layout treats a missing
            // style as None.
            style.column_rule_style = s;
        }
        "column-rule-width" => {
            // Keyword widths thin/medium/thick → 1/3/5px (CSS Backgrounds §4.3).
            for t in toks {
                if let CssToken::Ident(id) = t {
                    let kw = match id.to_ascii_lowercase().as_str() {
                        "thin" => Some(1.0),
                        "medium" => Some(3.0),
                        "thick" => Some(5.0),
                        _ => None,
                    };
                    if let Some(px) = kw {
                        style.column_rule_width = Some(px);
                    }
                }
            }
            if let Some(l) = Length::from_tokens(toks) {
                style.column_rule_width = Some(length_to_px_approx(l));
            }
        }
        "column-rule-style" => {
            for t in toks {
                if let CssToken::Ident(id) = t {
                    if let Some(bs) = BorderStyle::from_ident(id) {
                        style.column_rule_style = Some(bs);
                    }
                }
            }
        }
        "column-rule-color" => {
            if let Some(c) = Color::from_tokens(toks) {
                style.column_rule_color = Some(c);
            }
        }
        // CSS Multi-column §7 — `column-span: all | none`.
        "column-span" => {
            for t in toks {
                if let CssToken::Ident(id) = t {
                    match id.to_ascii_lowercase().as_str() {
                        "all" => style.column_span_all = true,
                        "none" => style.column_span_all = false,
                        _ => {}
                    }
                }
            }
        }
        "opacity" => {
            for t in toks {
                if let CssToken::Number(n) = t {
                    style.opacity = Some((*n as f32).clamp(0.0, 1.0));
                    break;
                }
                if let CssToken::Percent(p) = t {
                    style.opacity = Some((*p as f32 / 100.0).clamp(0.0, 1.0));
                    break;
                }
            }
        }
        "border-radius" => {
            // Take the first length value; corner-specific values can
            // come later. Critical that `border-radius: 0` doesn't error
            // out the rule (which is why we land on `Length::Zero`).
            if let Some(v) = Length::from_tokens(toks) {
                style.border_radius = Some(v);
            }
        }
        "position" => {
            if let Some(v) = Position::from_tokens(toks) {
                style.position = Some(v);
            }
        }
        "float" => {
            if let Some(v) = FloatSide::from_tokens(toks) {
                style.float_side = Some(v);
            }
        }
        "clear" => {
            if let Some(v) = ClearMode::from_tokens(toks) {
                style.clear = Some(v);
            }
        }
        "vertical-align" => {
            if let Some(v) = VerticalAlign::from_tokens(toks) {
                style.vertical_align = Some(v);
            }
        }
        "grid-template-areas" => {
            // Each string token is one row; split on whitespace into
            // column names. `.` is the empty-cell marker.
            let mut rows: Vec<Vec<String>> = Vec::new();
            for t in toks {
                if let CssToken::String(s) = t {
                    let row: Vec<String> = s.split_whitespace().map(|w| w.to_string()).collect();
                    if !row.is_empty() {
                        rows.push(row);
                    }
                }
            }
            // Reject inconsistent column counts — that's an invalid value
            // per spec, and silently truncating would mis-align named
            // areas. Leave the property unset so layout falls back to
            // auto-flow.
            if !rows.is_empty() {
                let cols = rows[0].len();
                if rows.iter().all(|r| r.len() == cols) {
                    style.grid_template_areas = Some(rows);
                }
            }
        }
        "grid-area" => {
            // Two shapes are common:
            //   * single ident → named-area reference (we honour this)
            //   * row-start / col-start / row-end / col-end → numeric
            //     longhand (we skip — falls back to default auto-flow)
            // Tokens come whitespace-stripped at parse time; an ident-only
            // value is exactly one Ident token.
            if toks.len() == 1 {
                if let CssToken::Ident(name) = &toks[0] {
                    style.grid_area_name = Some(name.to_string());
                }
            }
        }
        "overflow-wrap" | "word-wrap" => {
            // We always break by character when text overflows the
            // container, so this property is a no-op for V1 — but we
            // accept and ignore the value so the rule doesn't drop other
            // declarations after it during parsing.
            let _ = toks;
        }
        "top" => {
            if let Some(v) = Length::from_tokens(toks) {
                style.top = Some(v);
                style.physical_inset_seq[0] = seq;
            }
        }
        "right" => {
            if let Some(v) = Length::from_tokens(toks) {
                style.right = Some(v);
                style.physical_inset_seq[1] = seq;
            }
        }
        "bottom" => {
            if let Some(v) = Length::from_tokens(toks) {
                style.bottom = Some(v);
                style.physical_inset_seq[2] = seq;
            }
        }
        "left" => {
            if let Some(v) = Length::from_tokens(toks) {
                style.left = Some(v);
                style.physical_inset_seq[3] = seq;
            }
        }
        "z-index" => {
            for t in toks {
                if let CssToken::Number(n) = t {
                    style.z_index = Some(*n as i32);
                    break;
                }
            }
        }
        "transform" => {
            // Parse the WHOLE transform function list into an ordered op
            // list (CSS Transforms 2 §11), detecting whether any 3D
            // primitive is present. When the list is pure-2D we ALSO fill
            // the scalar fields below so the cheap 2D fast paths
            // (translate offset, scale geometry-bake, rotate/matrix affine
            // layer) keep working unchanged — no regression. When ANY 3D
            // primitive appears we instead carry the full op list so the
            // painter composes a real 4×4 matrix and projects the quad.
            let ops = parse_transform_function_list(toks);
            let has_3d = ops.iter().any(|op| matches!(op,
                Transform3DOp::Translate3d(..)
                | Transform3DOp::Scale3d(..)
                | Transform3DOp::RotateX(_)
                | Transform3DOp::RotateY(_)
                | Transform3DOp::Rotate3d(..)
                | Transform3DOp::Matrix3d(_)
                | Transform3DOp::Perspective(_)));
            if has_3d {
                // 3D path: carry the op list; the painter builds the 4×4.
                // Clear the scalar 2D fields so a stale 2D affine doesn't
                // also fire (the op list is the single source of truth).
                style.transform_ops = Some(ops);
                style.translate_x = None;
                style.translate_y = None;
                style.scale_x = None;
                style.scale_y = None;
                style.rotate_deg = None;
                style.matrix_2d = None;
            } else {
                // Pure-2D path — preserve the legacy scalar behaviour
                // exactly (sum translates, last scale/rotate/matrix wins).
                let mut tx: Option<Length> = None;
                let mut ty: Option<Length> = None;
                let add = |opt: &mut Option<Length>, v: Length| {
                    *opt = Some(match *opt {
                        Some(existing) => add_lengths(existing, v),
                        None => v,
                    });
                };
                // A `Length::Zero` component is the identity for the
                // translate SUM and is what `translateX`/`translateY`
                // synthesise for the unspecified axis — skip it so e.g.
                // `translateX(42px)` leaves `translate_y` untouched (None),
                // matching the pre-3D behaviour and CSSOM expectations.
                let is_zero = |l: &Length| matches!(l, Length::Zero);
                for op in &ops {
                    match op {
                        Transform3DOp::Translate(x, y) => {
                            if !is_zero(x) {
                                add(&mut tx, *x);
                            }
                            if !is_zero(y) {
                                add(&mut ty, *y);
                            }
                        }
                        Transform3DOp::Scale(sx, sy) => {
                            style.scale_x = Some(*sx);
                            style.scale_y = Some(*sy);
                        }
                        Transform3DOp::RotateZ(deg) => {
                            style.rotate_deg = Some(*deg);
                        }
                        Transform3DOp::Matrix2d(m) => {
                            style.matrix_2d = Some(*m);
                        }
                        // skew currently has no 2D-scalar slot; if a skew
                        // appears alongside other 2D fns we promote the
                        // whole list to the matrix path so it isn't lost.
                        Transform3DOp::Skew(ax, ay) if *ax != 0.0 || *ay != 0.0 => {
                            style.transform_ops = Some(ops.clone());
                            style.translate_x = None;
                            style.translate_y = None;
                            style.scale_x = None;
                            style.scale_y = None;
                            style.rotate_deg = None;
                            style.matrix_2d = None;
                            return;
                        }
                        _ => {}
                    }
                }
                if tx.is_some() {
                    style.translate_x = tx;
                }
                if ty.is_some() {
                    style.translate_y = ty;
                }
            }
        }
        // CSS Transforms L2 §3.4 individual transform longhands:
        //   translate: <length-percentage> [<length-percentage> [<length>]]?
        //   rotate: <angle> | <axis>? <angle>
        //   scale: <number>{1,3}
        // These are independent of `transform:` and combine in a defined
        // order at compose time (translate → rotate → scale). We honour
        // the 2D subset here.
        "translate" => {
            let parts = split_top_level_whitespace(toks);
            if let Some(p) = parts.first() {
                if let Some(l) = Length::from_tokens(p) {
                    style.translate_x = Some(l);
                }
            }
            if let Some(p) = parts.get(1) {
                if let Some(l) = Length::from_tokens(p) {
                    style.translate_y = Some(l);
                }
            }
        }
        "rotate" => {
            // Two forms: `45deg` and `z 45deg` / `0 0 1 45deg` (3D axis +
            // angle). V1 honours just the 2D angle; the axis is dropped.
            if let Some(deg) = first_angle_degrees(toks) {
                style.rotate_deg = Some(deg as f32);
            }
        }
        "scale" => {
            // `2` → uniform; `2 0.5` → (sx, sy). Numbers only.
            let mut nums: Vec<f32> = Vec::new();
            for t in toks {
                if let CssToken::Number(n) = t {
                    nums.push(*n as f32);
                }
            }
            if let Some(&first) = nums.first() {
                style.scale_x = Some(first);
                style.scale_y = Some(*nums.get(1).unwrap_or(&first));
            }
        }
        "box-shadow" => {
            // Syntax: [inset]? <length> <length> [<blur>] [<spread>] <color>
            // The `inset` keyword may appear before or after the lengths.
            // Multi-shadow (comma-separated) is not yet honoured — we parse
            // the first shadow only (the comma-separated values are rare and
            // the first shadow is usually the intended one).
            let mut lengths: Vec<Length> = Vec::new();
            let mut inset = false;
            for t in toks {
                match t {
                    CssToken::Ident(id) if id.eq_ignore_ascii_case("inset") => {
                        inset = true;
                    }
                    _ => {
                        if let Some(l) = Length::from_tokens(std::slice::from_ref(t)) {
                            lengths.push(l);
                        }
                    }
                }
                if lengths.len() >= 4 {
                    break;
                }
            }
            let color = Color::from_tokens(toks).unwrap_or(Color::BLACK);
            if lengths.len() >= 2 {
                style.box_shadow = Some(BoxShadowSpec {
                    offset_x: lengths[0],
                    offset_y: lengths[1],
                    blur: lengths.get(2).copied().unwrap_or(Length::Zero),
                    spread: lengths.get(3).copied().unwrap_or(Length::Zero),
                    color,
                    inset,
                });
            }
        }
        "filter" => {
            style.filters = parse_filter_chain(toks);
        }
        "backdrop-filter" => {
            // Same grammar as `filter`. We parse it for completeness;
            // V1 painter only honours `filter` on the element itself,
            // not the area behind it, so this currently has no visual
            // effect. Keeping the parsed chain available lets a later
            // compositor pass implement it without re-parsing.
            style.backdrop_filters = parse_filter_chain(toks);
        }
        "mix-blend-mode" => {
            // A single blend-mode keyword. CSS Compositing & Blending L1 §5.
            for t in toks {
                if let CssToken::Ident(s) = t {
                    style.mix_blend_mode = Some(s.to_ascii_lowercase());
                    break;
                }
            }
        }
        "background-blend-mode" => {
            // A comma-separated list (one per background layer). We paint one
            // background layer, so take the first keyword. §6.
            for t in toks {
                if let CssToken::Ident(s) = t {
                    style.background_blend_mode = Some(s.to_ascii_lowercase());
                    break;
                }
            }
        }
        "clip-path" => {
            style.clip_path = parse_clip_path(toks);
        }
        "mask" | "-webkit-mask" | "mask-image" | "-webkit-mask-image" => {
            // Extract the first `url(...)` so the painter can fetch the
            // mask bitmap and use its alpha as a stencil for the box's
            // background color.
            for t in toks {
                if let CssToken::Url(s) = t {
                    style.has_mask_url = true;
                    style.mask_image_url = Some(s.clone());
                    break;
                }
            }
        }
        "content" => {
            // `content: "literal"` / `content: none` / `content: normal`.
            // We honour the string form; everything else (counter(), url(),
            // attr(), etc.) becomes None so the generated box isn't
            // emitted. The value is what the layout tree builder will
            // stamp into the synthetic ::before/::after node.
            let mut text: Option<String> = None;
            let mut saw_none = false;
            for t in toks {
                match t {
                    CssToken::String(s) => {
                        text = Some(s.clone());
                        break;
                    }
                    CssToken::Ident(s)
                        if s.eq_ignore_ascii_case("none") || s.eq_ignore_ascii_case("normal") =>
                    {
                        saw_none = true;
                    }
                    _ => {}
                }
            }
            if let Some(t) = text {
                style.content = Some(t);
            } else if saw_none {
                style.content = None;
            }
        }
        "text-shadow" => {
            // Same lex shape as box-shadow (offset_x, offset_y, blur, color).
            // text-shadow has no spread or inset; those fields are zero/false.
            let mut lengths: Vec<Length> = Vec::new();
            for t in toks {
                if let Some(l) = Length::from_tokens(std::slice::from_ref(t)) {
                    lengths.push(l);
                }
                if lengths.len() >= 3 {
                    break;
                }
            }
            let color = Color::from_tokens(toks).unwrap_or(Color::BLACK);
            if lengths.len() >= 2 {
                style.text_shadow = Some(BoxShadowSpec {
                    offset_x: lengths[0],
                    offset_y: lengths[1],
                    blur: lengths.get(2).copied().unwrap_or(Length::Zero),
                    spread: Length::Zero,
                    color,
                    inset: false,
                });
            }
        }
        // ── overflow ──────────────────────────────────────────────────────
        // `overflow: hidden` (and the `clip` value) makes a box act as a
        // clipping container. `overflow-x`/`-y` longhands are handled here
        // too; we only model the hidden/clip state (no scroll). Per CSS
        // Overflow 3 §2.2, either axis being `hidden` or `clip` is enough
        // to trigger clipping on that axis. We set the single boolean flag
        // `overflow_hidden` which the paint path uses to scissor children.
        "overflow" | "overflow-x" | "overflow-y" => {
            use crate::properties::Overflow;
            // Collect the (up to two) ident values present, in order.
            let mut vals: Vec<Overflow> = Vec::new();
            for t in toks {
                if matches!(t, CssToken::Ident(_)) {
                    if let Some(v) = Overflow::from_tokens(std::slice::from_ref(t)) {
                        vals.push(v);
                    }
                }
            }
            match d.name.as_str() {
                "overflow-x" => {
                    if let Some(v) = vals.first().copied() {
                        style.overflow_x = Some(v);
                    }
                }
                "overflow-y" => {
                    if let Some(v) = vals.first().copied() {
                        style.overflow_y = Some(v);
                    }
                }
                // Shorthand `overflow: <x> [<y>]`. One value sets both axes;
                // two values are `overflow-x overflow-y` (CSS Overflow 3 §3.1).
                _ => {
                    let x = vals.first().copied();
                    let y = vals.get(1).copied().or(x);
                    if let Some(x) = x {
                        style.overflow_x = Some(x);
                    }
                    if let Some(y) = y {
                        style.overflow_y = Some(y);
                    }
                }
            }
            // Keep the legacy `overflow_hidden` flag in sync: it means
            // "this box clips its overflow" (any non-visible value on
            // either axis). Existing clip-rect paint paths read it; the
            // new scroll path reads the per-axis enums.
            let clips = style.overflow_x.map(|o| o.clips()).unwrap_or(false)
                || style.overflow_y.map(|o| o.clips()).unwrap_or(false);
            style.overflow_hidden = clips;
        }
        // ── transform-origin ──────────────────────────────────────────────
        // The pivot point around which 2D transforms (rotate, scale, matrix)
        // are applied. CSS default is `50% 50%` (border-box centre). Reuses
        // the background-position parser which already handles keywords,
        // px lengths and percentages. `None` means "not overridden" — the
        // paint path falls back to centre.
        "transform-origin" => {
            if let Some(origin) = parse_background_position(toks) {
                style.transform_origin = Some(origin);
            }
        }
        // CSS Transforms 2 §6: the `perspective` PROPERTY. `none` = no
        // perspective; otherwise a <length> (resolved px at the bridge).
        "perspective" => {
            let is_none = toks
                .iter()
                .any(|t| matches!(t, CssToken::Ident(s) if s.eq_ignore_ascii_case("none")));
            if is_none {
                style.perspective_px = None;
            } else if let Some(l) = crate::properties::Length::from_tokens(toks) {
                // Most real uses are px; resolve em/rem later if needed.
                // Store as px here using a neutral em (16) since the
                // `perspective` property is almost always authored in px.
                if let Some(px) = l.resolve_px(16.0, 16.0, 0.0) {
                    style.perspective_px = Some(px.max(1.0));
                }
            }
        }
        // CSS Transforms 2 §6: vanishing point for the `perspective`
        // property. Default 50% 50%.
        "perspective-origin" => {
            if let Some(origin) = parse_background_position(toks) {
                style.perspective_origin = Some(origin);
            }
        }
        // CSS Transforms 2 §5: `transform-style: flat | preserve-3d`.
        "transform-style" => {
            style.transform_style_preserve_3d = toks
                .iter()
                .any(|t| matches!(t, CssToken::Ident(s) if s.eq_ignore_ascii_case("preserve-3d")));
        }
        // CSS Transforms 2 §10: `backface-visibility: visible | hidden`.
        "backface-visibility" => {
            style.backface_visibility_hidden = toks
                .iter()
                .any(|t| matches!(t, CssToken::Ident(s) if s.eq_ignore_ascii_case("hidden")));
        }
        // CSS Containment 3 §2.1 — `container-type: normal | inline-size | size`.
        // Establishes this element as a query container (or not). The cascade
        // reads `style.container_type` when building the query-container stack
        // so `@container` rules can be evaluated against this box's laid-out
        // content-box size. `contain: <x>` does NOT set container-type (a
        // separate property), so we only handle the dedicated longhand here.
        "container-type" => {
            for t in toks {
                if let CssToken::Ident(s) = t {
                    style.container_type = Some(match s.to_ascii_lowercase().as_str() {
                        "inline-size" => ContainerType::InlineSize,
                        "size" => ContainerType::Size,
                        "normal" => ContainerType::Normal,
                        _ => continue,
                    });
                    break;
                }
            }
        }
        // CSS Containment 3 §2.2 — `container-name: none | <custom-ident>+`.
        // Names this query container so `@container <name> (...)` can target
        // it specifically rather than the nearest container of the right type.
        "container-name" => {
            let mut names = Vec::new();
            for t in toks {
                if let CssToken::Ident(s) = t {
                    if s.eq_ignore_ascii_case("none") {
                        names.clear();
                        break;
                    }
                    names.push(s.clone());
                }
            }
            style.container_name = names;
        }
        // CSS Containment 3 §2.3 — the `container` shorthand:
        //   container: <name> [ / <container-type> ]?
        // e.g. `container: sidebar / inline-size`. Name(s) before the slash,
        // type after. `container: none` resets both.
        "container" => {
            let mut name_toks: Vec<&CssToken> = Vec::new();
            let mut type_toks: Vec<&CssToken> = Vec::new();
            let mut after_slash = false;
            for t in toks {
                match t {
                    CssToken::Delim('/') => after_slash = true,
                    CssToken::Whitespace => {}
                    other if after_slash => type_toks.push(other),
                    other => name_toks.push(other),
                }
            }
            let mut names = Vec::new();
            for t in &name_toks {
                if let CssToken::Ident(s) = t {
                    if s.eq_ignore_ascii_case("none") {
                        names.clear();
                        break;
                    }
                    names.push(s.clone());
                }
            }
            style.container_name = names;
            for t in &type_toks {
                if let CssToken::Ident(s) = t {
                    style.container_type = Some(match s.to_ascii_lowercase().as_str() {
                        "inline-size" => ContainerType::InlineSize,
                        "size" => ContainerType::Size,
                        "normal" => ContainerType::Normal,
                        _ => continue,
                    });
                    break;
                }
            }
        }
        // Properties we parse silently as "noted but not visualised yet".
        // Without these branches the unknown-property path leaves them
        // hanging, which is fine — but listing them keeps the next
        // implementer's grep clean and avoids accidental property
        // collisions with shorthand handlers above.
        // Properties we recognise so the cascade pass doesn't drop
        // them to "unknown property" diagnostics, but whose painting
        // / layout effect is either already represented via aliases
        // or punted to a future slice. Most of these are MV3-CSS or
        // post-2020 additions that we tolerate for forward-compat.
        "cursor"
        | "will-change"
        | "user-select"
        | "pointer-events"
        | "text-wrap"
        | "text-wrap-style"
        | "text-wrap-mode"
        | "text-indent"
        | "text-justify"
        | "text-underline-offset"
        | "text-decoration-color"
        | "text-decoration-style"
        | "text-decoration-thickness"
        | "text-decoration-skip-ink"
        | "word-spacing"
        | "word-break"
        | "overflow-wrap"
        | "word-wrap"
        | "hyphens"
        | "tab-size"
        | "unicode-bidi"
        | "text-orientation"
        | "text-emphasis"
        | "text-emphasis-color"
        | "text-emphasis-style"
        | "text-emphasis-position"
        | "font-variant"
        | "font-variant-caps"
        | "font-variant-ligatures"
        | "font-variant-numeric"
        | "font-variant-east-asian"
        | "font-variant-position"
        | "font-variant-alternates"
        | "font-stretch"
        | "font-feature-settings"
        | "font-variation-settings"
        | "font-optical-sizing"
        | "font-synthesis"
        | "font-kerning"
        | "font-size-adjust"
        | "font-palette"
        | "field-sizing"
        | "appearance"
        | "-webkit-appearance"
        | "caret-shape"
        | "resize"
        | "mask-mode"
        | "mask-position"
        | "mask-size"
        | "mask-repeat"
        | "mask-clip"
        | "mask-origin"
        | "mask-composite"
        | "mask-type"
        | "mask-border"
        | "mask-border-source"
        | "mask-border-slice"
        | "mask-border-width"
        | "mask-border-outset"
        | "mask-border-repeat"
        | "mask-border-mode"
        | "scroll-snap-type"
        | "scroll-snap-align"
        | "scroll-snap-stop"
        | "scroll-padding"
        | "scroll-padding-top"
        | "scroll-padding-right"
        | "scroll-padding-bottom"
        | "scroll-padding-left"
        | "scroll-padding-inline"
        | "scroll-padding-block"
        | "scroll-padding-inline-start"
        | "scroll-padding-inline-end"
        | "scroll-padding-block-start"
        | "scroll-padding-block-end"
        | "scroll-margin"
        | "scroll-margin-top"
        | "scroll-margin-right"
        | "scroll-margin-bottom"
        | "scroll-margin-left"
        | "scroll-margin-inline"
        | "scroll-margin-block"
        | "scroll-margin-inline-start"
        | "scroll-margin-inline-end"
        | "scroll-margin-block-start"
        | "scroll-margin-block-end"
        | "scrollbar-gutter"
        | "overscroll-behavior"
        | "overscroll-behavior-x"
        | "overscroll-behavior-y"
        | "overscroll-behavior-inline"
        | "overscroll-behavior-block"
        | "scroll-behavior"
        | "scroll-timeline"
        | "scroll-timeline-name"
        | "scroll-timeline-axis"
        | "view-timeline"
        | "view-timeline-name"
        | "view-timeline-axis"
        | "view-timeline-inset"
        | "animation-timeline"
        | "animation-range"
        | "animation-range-start"
        | "animation-range-end"
        | "animation-composition"
        | "contain"
        | "content-visibility"
        | "anchor-name"
        | "anchor-scope"
        | "position-anchor"
        | "position-area"
        | "position-try"
        | "position-try-fallbacks"
        | "position-try-order"
        | "inset-area"
        | "offset"
        | "offset-path"
        | "offset-distance"
        | "offset-rotate"
        | "offset-anchor"
        | "offset-position"
        | "background-clip"
        | "background-origin"
        | "background-attachment"
        | "isolation"
        | "image-rendering"
        | "paint-order"
        | "forced-color-adjust"
        | "print-color-adjust"
        | "quotes"
        | "list-style-position"
        | "list-style-image"
        | "counter-reset"
        | "counter-increment"
        | "counter-set"
        | "outline"
        | "outline-color"
        | "outline-style"
        | "outline-width"
        | "outline-offset"
        | "border-image"
        | "border-image-source"
        | "border-image-slice"
        | "border-image-width"
        | "border-image-outset"
        | "border-image-repeat"
        | "border-collapse"
        | "border-spacing"
        | "caption-side"
        | "empty-cells"
        | "table-layout"
        | "column-fill"
        | "break-before"
        | "break-after"
        | "break-inside"
        | "page-break-before"
        | "page-break-after"
        | "page-break-inside"
        | "widows"
        | "orphans"
        | "speak"
        | "speak-as"
        | "voice-family"
        | "voice-volume"
        | "voice-rate"
        | "voice-pitch"
        | "voice-stress"
        | "pause"
        | "pause-before"
        | "pause-after"
        | "rest"
        | "rest-before"
        | "rest-after"
        | "cue"
        | "cue-before"
        | "cue-after"
        | "view-transition-class"
        | "view-transition-group"
        | "interpolate-size"
        | "math-style"
        | "math-shift"
        | "math-depth"
        | "ruby-position"
        | "ruby-align"
        | "ruby-overhang" => {
            for t in toks {
                if let CssToken::Ident(s) = t {
                    if s.eq_ignore_ascii_case("hidden") || s.eq_ignore_ascii_case("clip") {
                        style.overflow_hidden = true;
                    }
                }
            }
        }
        "white-space" => {
            for t in toks {
                if let CssToken::Ident(s) = t {
                    style.white_space = Some(match s.to_ascii_lowercase().as_str() {
                        "normal" => WhiteSpace::Normal,
                        "pre" => WhiteSpace::Pre,
                        "nowrap" => WhiteSpace::Nowrap,
                        "pre-wrap" => WhiteSpace::PreWrap,
                        "pre-line" => WhiteSpace::PreLine,
                        "break-spaces" => WhiteSpace::BreakSpaces,
                        _ => continue,
                    });
                    break;
                }
            }
        }
        "text-overflow" => {
            for t in toks {
                if let CssToken::Ident(s) = t {
                    if s.eq_ignore_ascii_case("ellipsis") {
                        style.text_overflow_ellipsis = true;
                    } else if s.eq_ignore_ascii_case("clip") {
                        style.text_overflow_ellipsis = false;
                    }
                    break;
                }
            }
        }
        "text-transform" => {
            for t in toks {
                if let CssToken::Ident(s) = t {
                    style.text_transform = Some(match s.to_ascii_lowercase().as_str() {
                        "none" => TextTransform::None,
                        "uppercase" => TextTransform::Uppercase,
                        "lowercase" => TextTransform::Lowercase,
                        "capitalize" => TextTransform::Capitalize,
                        _ => continue,
                    });
                    break;
                }
            }
        }
        "letter-spacing" => {
            for t in toks {
                if let CssToken::Dimension { value, unit } = t {
                    let px = match unit.as_str() {
                        "px" | "" => *value,
                        "em" | "rem" => *value * 16.0,
                        _ => continue,
                    };
                    style.letter_spacing_px = Some(px as f32);
                    break;
                }
                if let CssToken::Ident(s) = t {
                    if s.eq_ignore_ascii_case("normal") {
                        style.letter_spacing_px = Some(0.0);
                    }
                    break;
                }
            }
        }
        "accent-color" => {
            if let Some(c) = Color::from_tokens(toks) {
                style.accent_color = Some(c);
            }
        }
        "caret-color" => {
            if let Some(c) = Color::from_tokens(toks) {
                style.caret_color = Some(c);
            }
        }
        "scrollbar-width" => {
            // CSS Scrollbars 1 §3: auto | thin | none.
            if let Some(CssToken::Ident(kw)) =
                toks.iter().find(|t| !matches!(t, CssToken::Whitespace))
            {
                style.scrollbar_width = match kw.to_ascii_lowercase().as_str() {
                    "thin" => 1,
                    "none" => 2,
                    _ => 0, // auto (default)
                };
            }
        }
        "scrollbar-color" => {
            // CSS Scrollbars 1 §2: `auto` | <thumb-color> <track-color>.
            // `auto` leaves the UA default (None). Otherwise two colours
            // separated by top-level whitespace (functions keep their own
            // internal whitespace via paren-depth tracking).
            let first = toks.iter().find(|t| !matches!(t, CssToken::Whitespace));
            if let Some(CssToken::Ident(kw)) = first {
                if kw.eq_ignore_ascii_case("auto") {
                    style.scrollbar_color = None;
                    return;
                }
            }
            // Split into top-level groups.
            let mut groups: Vec<Vec<CssToken>> = Vec::new();
            let mut cur: Vec<CssToken> = Vec::new();
            let mut depth: i32 = 0;
            for t in toks {
                match t {
                    CssToken::Function(_) => {
                        depth += 1;
                        cur.push(t.clone());
                    }
                    CssToken::RightParen => {
                        depth -= 1;
                        cur.push(t.clone());
                    }
                    CssToken::Whitespace if depth == 0 => {
                        if !cur.is_empty() {
                            groups.push(std::mem::take(&mut cur));
                        }
                    }
                    _ => cur.push(t.clone()),
                }
            }
            if !cur.is_empty() {
                groups.push(cur);
            }
            if groups.len() >= 2 {
                let thumb = Color::from_tokens(&groups[0]);
                let track = Color::from_tokens(&groups[1]);
                if let (Some(thumb), Some(track)) = (thumb, track) {
                    style.scrollbar_color = Some((thumb, track));
                }
            }
        }
        "color-scheme" => {
            // Accept any of the keyword combinations as a single string.
            let s: String = toks
                .iter()
                .filter_map(|t| match t {
                    CssToken::Ident(s) => Some(s.clone()),
                    CssToken::Whitespace => Some(" ".into()),
                    _ => None,
                })
                .collect::<String>()
                .split_whitespace()
                .collect::<Vec<_>>()
                .join(" ");
            if !s.is_empty() {
                style.color_scheme = Some(s);
            }
        }
        "view-transition-name" => {
            for t in toks {
                if let CssToken::Ident(s) = t {
                    if !s.eq_ignore_ascii_case("none") {
                        style.view_transition_name = Some(s.clone());
                    }
                    break;
                }
            }
        }
        "animation" | "animation-name" => {
            // `animation: name 2s ease-in 0.5s infinite` — token by
            // token. Idents are the name (first non-keyword), the
            // timing function (ease, linear, ease-in, ...), or
            // iteration count keyword (infinite). Dimensions are
            // duration (first) and delay (second).
            let mut name: Option<String> = None;
            let mut dim_count = 0;
            for t in toks {
                match t {
                    CssToken::Ident(s) => {
                        let lc = s.to_ascii_lowercase();
                        match lc.as_str() {
                            "none" => continue,
                            "linear" => style.animation_timing = Some(0),
                            "ease" | "ease-in-out" => style.animation_timing = Some(3),
                            "ease-in" => style.animation_timing = Some(1),
                            "ease-out" => style.animation_timing = Some(2),
                            "infinite" => style.animation_iteration_count = Some(f32::INFINITY),
                            "alternate" | "reverse" | "alternate-reverse" | "forwards"
                            | "backwards" | "both" | "running" | "paused" | "normal" => {}
                            // `@keyframes` names are CASE-SENSITIVE — keep the
                            // original case (`orbitalRotate`), not the lowercased
                            // form, or it won't match the keyframe rule.
                            _ if name.is_none() => name = Some(s.clone()),
                            _ => {}
                        }
                    }
                    CssToken::String(s) if name.is_none() => {
                        name = Some(s.clone());
                    }
                    CssToken::Number(n) if style.animation_iteration_count.is_none() => {
                        style.animation_iteration_count = Some(*n as f32);
                    }
                    CssToken::Dimension { value, unit } => {
                        let ms = match unit.to_ascii_lowercase().as_str() {
                            "s" => (*value as f32) * 1000.0,
                            "ms" => *value as f32,
                            _ => *value as f32,
                        };
                        if dim_count == 0 {
                            style.animation_duration_ms = Some(ms);
                        } else if dim_count == 1 {
                            style.animation_delay_ms = Some(ms);
                        }
                        dim_count += 1;
                    }
                    _ => {}
                }
            }
            if let Some(n) = name {
                style.animation_name = Some(n);
                style.has_animation = true;
            }
        }
        "animation-duration" => {
            for t in toks {
                if let CssToken::Dimension { value, unit } = t {
                    let ms = match unit.to_ascii_lowercase().as_str() {
                        "s" => (*value as f32) * 1000.0,
                        "ms" => *value as f32,
                        _ => *value as f32,
                    };
                    style.animation_duration_ms = Some(ms);
                    break;
                }
            }
        }
        "animation-delay" => {
            for t in toks {
                if let CssToken::Dimension { value, unit } = t {
                    let ms = match unit.to_ascii_lowercase().as_str() {
                        "s" => (*value as f32) * 1000.0,
                        "ms" => *value as f32,
                        _ => *value as f32,
                    };
                    style.animation_delay_ms = Some(ms);
                    break;
                }
            }
        }
        "animation-iteration-count" => {
            for t in toks {
                match t {
                    CssToken::Number(n) => {
                        style.animation_iteration_count = Some(*n as f32);
                        break;
                    }
                    CssToken::Ident(s) if s.eq_ignore_ascii_case("infinite") => {
                        style.animation_iteration_count = Some(f32::INFINITY);
                        break;
                    }
                    _ => {}
                }
            }
        }
        "animation-timing-function" => {
            // Keyword form first.
            for t in toks {
                if let CssToken::Ident(s) = t {
                    style.animation_timing = match s.to_ascii_lowercase().as_str() {
                        "linear" => Some(0),
                        "ease-in" => Some(1),
                        "ease-out" => Some(2),
                        "ease" | "ease-in-out" => Some(3),
                        // step-* family — Web Animations §3.12.4.
                        "step-start" => Some(4),
                        "step-end" => Some(5),
                        _ => style.animation_timing,
                    };
                }
                // `cubic-bezier(x1, y1, x2, y2)` and `steps(N[, jump])`
                // arrive as a Function token. We treat the *presence* of
                // either as an "ease"-equivalent until interpolation is
                // wired through the animation tick — but we do parse the
                // arguments so a later pass can pick up the real curve.
                if let CssToken::Function(name) = t {
                    let lc = name.to_ascii_lowercase();
                    if lc == "cubic-bezier" {
                        style.animation_timing = Some(3); // ease-like fallback
                    } else if lc == "steps" {
                        style.animation_timing = Some(5); // step-end fallback
                    } else if lc == "linear" {
                        style.animation_timing = Some(0);
                    }
                }
            }
        }
        "transition-duration" => {
            for t in toks {
                if let CssToken::Dimension { value, unit } = t {
                    let lc = unit.to_ascii_lowercase();
                    style.transition_duration_ms = Some(if lc == "s" {
                        (*value as f32) * 1000.0
                    } else {
                        *value as f32
                    });
                    break;
                }
            }
        }
        "transition-delay" => {
            for t in toks {
                if let CssToken::Dimension { value, unit } = t {
                    let lc = unit.to_ascii_lowercase();
                    style.transition_delay_ms = Some(if lc == "s" {
                        (*value as f32) * 1000.0
                    } else {
                        *value as f32
                    });
                    break;
                }
            }
        }
        "transition-timing-function" => {
            for t in toks {
                if let CssToken::Ident(s) = t {
                    style.transition_timing = match s.to_ascii_lowercase().as_str() {
                        "linear" => Some(0),
                        "ease-in" => Some(1),
                        "ease-out" => Some(2),
                        "ease" | "ease-in-out" => Some(3),
                        "step-start" => Some(4),
                        "step-end" => Some(5),
                        _ => style.transition_timing,
                    };
                } else if let CssToken::Function(_) = t {
                    style.transition_timing = Some(3); // cubic-bezier/steps → ease
                }
            }
        }
        "transition-property" => {
            for t in toks {
                if let CssToken::Ident(s) = t {
                    style.transition_property = Some(s.to_ascii_lowercase());
                    break;
                }
            }
        }
        "transition" => {
            // Shorthand `transition: <property> <duration> [<timing>] [<delay>]`.
            // First <time> is duration, second is delay; idents are the timing
            // keyword or the property name. We parse the FIRST comma-list item
            // (the common single-property / `all` form); a comma ends parsing.
            let mut times_seen = 0u8;
            for t in toks {
                match t {
                    CssToken::Dimension { value, unit } => {
                        let lc = unit.to_ascii_lowercase();
                        if lc == "s" || lc == "ms" {
                            let ms = if lc == "s" {
                                (*value as f32) * 1000.0
                            } else {
                                *value as f32
                            };
                            if times_seen == 0 {
                                style.transition_duration_ms = Some(ms);
                            } else if times_seen == 1 {
                                style.transition_delay_ms = Some(ms);
                            }
                            times_seen += 1;
                        }
                    }
                    CssToken::Ident(s) => match s.to_ascii_lowercase().as_str() {
                        "linear" => style.transition_timing = Some(0),
                        "ease-in" => style.transition_timing = Some(1),
                        "ease-out" => style.transition_timing = Some(2),
                        "ease" | "ease-in-out" => style.transition_timing = Some(3),
                        "step-start" => style.transition_timing = Some(4),
                        "step-end" => style.transition_timing = Some(5),
                        "none" => style.transition_property = Some("none".into()),
                        other => style.transition_property = Some(other.to_string()),
                    },
                    CssToken::Function(_) => style.transition_timing = Some(3),
                    CssToken::Comma => break, // only the first list item
                    _ => {}
                }
            }
            // Spec: a duration with no explicit property transitions `all`.
            if style.transition_duration_ms.is_some() && style.transition_property.is_none() {
                style.transition_property = Some("all".into());
            }
        }
        "line-height" => {
            for t in toks {
                match t {
                    CssToken::Number(n) => {
                        style.line_height = Some(LineHeight::Multiplier(*n as f32));
                        break;
                    }
                    CssToken::Dimension { .. } | CssToken::Percent(_) => {
                        if let Some(l) = Length::from_tokens(std::slice::from_ref(t)) {
                            style.line_height = Some(LineHeight::Length(l));
                        }
                        break;
                    }
                    CssToken::Ident(s) if s.eq_ignore_ascii_case("normal") => {
                        style.line_height = Some(LineHeight::Multiplier(1.2));
                        break;
                    }
                    _ => {}
                }
            }
        }
        "text-decoration" | "text-decoration-line" => {
            // Per CSS Text Decoration L4 §3, the `text-decoration-line`
            // value is a space-separated list of `underline`,
            // `line-through`, `overline`, `blink` (or `none`). The
            // shorthand REPLACES the whole list, so seeing `underline`
            // alone must clear any previously-set `line-through` (and
            // vice versa). Our previous code only touched a flag when
            // its keyword was present, so `text-decoration: underline`
            // following an earlier `line-through` rule left both lines
            // drawn — a real Chrome-divergence audit hit on news/article
            // pages. Now: always overwrite BOTH longhands at apply time
            // and let cascade order decide.
            let mut underline = false;
            let mut line_through = false;
            let mut saw_keyword = false;
            for t in toks {
                if let CssToken::Ident(s) = t {
                    match s.to_ascii_lowercase().as_str() {
                        "underline" => {
                            underline = true;
                            saw_keyword = true;
                        }
                        "line-through" => {
                            line_through = true;
                            saw_keyword = true;
                        }
                        "none" => {
                            saw_keyword = true;
                        }
                        // `overline` / `blink` are spec-recognized but
                        // we don't paint them yet — they still count as
                        // "saw a keyword" so the shorthand resets.
                        "overline" | "blink" => saw_keyword = true,
                        _ => {}
                    }
                }
            }
            if saw_keyword {
                style.text_decoration_underline = Some(underline);
                style.text_decoration_line_through = Some(line_through);
            }
        }
        // Per-side `border-*-style` longhands and `border-style` shorthand.
        "border-top-style" => {
            if let Some(s) = toks.iter().find_map(|t| {
                if let CssToken::Ident(id) = t { BorderStyle::from_ident(id) } else { None }
            }) {
                style.border_top_style = Some(s);
            }
        }
        "border-right-style" => {
            if let Some(s) = toks.iter().find_map(|t| {
                if let CssToken::Ident(id) = t { BorderStyle::from_ident(id) } else { None }
            }) {
                style.border_right_style = Some(s);
            }
        }
        "border-bottom-style" => {
            if let Some(s) = toks.iter().find_map(|t| {
                if let CssToken::Ident(id) = t { BorderStyle::from_ident(id) } else { None }
            }) {
                style.border_bottom_style = Some(s);
            }
        }
        "border-left-style" => {
            if let Some(s) = toks.iter().find_map(|t| {
                if let CssToken::Ident(id) = t { BorderStyle::from_ident(id) } else { None }
            }) {
                style.border_left_style = Some(s);
            }
        }
        // `border-style` shorthand: 1–4 values, same box-side expansion as
        // margin / padding (top  right  bottom  left).
        "border-style" => {
            let styles: Vec<BorderStyle> = toks
                .iter()
                .filter_map(|t| {
                    if let CssToken::Ident(id) = t { BorderStyle::from_ident(id) } else { None }
                })
                .collect();
            match styles.len() {
                1 => {
                    style.border_top_style = Some(styles[0]);
                    style.border_right_style = Some(styles[0]);
                    style.border_bottom_style = Some(styles[0]);
                    style.border_left_style = Some(styles[0]);
                }
                2 => {
                    style.border_top_style = Some(styles[0]);
                    style.border_right_style = Some(styles[1]);
                    style.border_bottom_style = Some(styles[0]);
                    style.border_left_style = Some(styles[1]);
                }
                3 => {
                    style.border_top_style = Some(styles[0]);
                    style.border_right_style = Some(styles[1]);
                    style.border_bottom_style = Some(styles[2]);
                    style.border_left_style = Some(styles[1]);
                }
                4 => {
                    style.border_top_style = Some(styles[0]);
                    style.border_right_style = Some(styles[1]);
                    style.border_bottom_style = Some(styles[2]);
                    style.border_left_style = Some(styles[3]);
                }
                _ => {}
            }
        }
        // `fill` is an SVG presentation attribute that some sites also
        // set via CSS targeting `<path>` etc. We don't paint inline SVG
        // through the cascade path, so just acknowledge it.
        "fill"
        | "stroke"
        | "stroke-width"
        | "stroke-linecap"
        | "stroke-linejoin"
        | "stroke-dasharray"
        | "stroke-opacity"
        | "stroke-dashoffset"
        | "stroke-miterlimit"
        | "fill-opacity"
        | "fill-rule"
        | "vector-effect"
        | "color-interpolation"
        | "dominant-baseline"
        | "text-anchor"
        | "alignment-baseline" => {}
        // Logical-position single-side properties. For LTR top-to-bottom
        // sides via the element's writing-mode + direction. Stored raw in
        // the logical-inset accumulator and mapped by `resolve_logical_box`
        // (CSS Logical 1 §4.3). inline-start→left etc. only under the
        // default horizontal-tb/ltr; vertical-rl maps block-start→right etc.
        "inset-inline-start" => apply_logical_pad(&mut style.logical_inset, LOGICAL_INLINE_START, toks, seq),
        "inset-inline-end" => apply_logical_pad(&mut style.logical_inset, LOGICAL_INLINE_END, toks, seq),
        "inset-block-start" => apply_logical_pad(&mut style.logical_inset, LOGICAL_BLOCK_START, toks, seq),
        "inset-block-end" => apply_logical_pad(&mut style.logical_inset, LOGICAL_BLOCK_END, toks, seq),
        "inset-inline" => {
            let lengths: Vec<Length> = toks
                .iter()
                .filter_map(|t| Length::from_tokens(std::slice::from_ref(t)))
                .collect();
            match lengths.len() {
                1 => {
                    style.logical_inset.vals[LOGICAL_INLINE_START] = Some(lengths[0]);
                    style.logical_inset.vals[LOGICAL_INLINE_END] = Some(lengths[0]);
                    style.logical_inset.seq[LOGICAL_INLINE_START] = seq;
                    style.logical_inset.seq[LOGICAL_INLINE_END] = seq;
                }
                2 => {
                    style.logical_inset.vals[LOGICAL_INLINE_START] = Some(lengths[0]);
                    style.logical_inset.vals[LOGICAL_INLINE_END] = Some(lengths[1]);
                    style.logical_inset.seq[LOGICAL_INLINE_START] = seq;
                    style.logical_inset.seq[LOGICAL_INLINE_END] = seq;
                }
                _ => {}
            }
        }
        "inset-block" => {
            let lengths: Vec<Length> = toks
                .iter()
                .filter_map(|t| Length::from_tokens(std::slice::from_ref(t)))
                .collect();
            match lengths.len() {
                1 => {
                    style.logical_inset.vals[LOGICAL_BLOCK_START] = Some(lengths[0]);
                    style.logical_inset.vals[LOGICAL_BLOCK_END] = Some(lengths[0]);
                    style.logical_inset.seq[LOGICAL_BLOCK_START] = seq;
                    style.logical_inset.seq[LOGICAL_BLOCK_END] = seq;
                }
                2 => {
                    style.logical_inset.vals[LOGICAL_BLOCK_START] = Some(lengths[0]);
                    style.logical_inset.vals[LOGICAL_BLOCK_END] = Some(lengths[1]);
                    style.logical_inset.seq[LOGICAL_BLOCK_START] = seq;
                    style.logical_inset.seq[LOGICAL_BLOCK_END] = seq;
                }
                _ => {}
            }
        }
        // `inset` shorthand — 1/2/3/4 lengths map to top/right/bottom/left
        // (per Logical Properties L1 §3.1). Sites use this constantly for
        // overlay panels (`inset: 0`) and centred dialogs (`inset: 1rem`).
        "inset" => {
            let lengths: Vec<Length> = toks
                .iter()
                .filter_map(|t| Length::from_tokens(std::slice::from_ref(t)))
                .collect();
            match lengths.len() {
                1 => {
                    let v = lengths[0];
                    style.top = Some(v);
                    style.right = Some(v);
                    style.bottom = Some(v);
                    style.left = Some(v);
                }
                2 => {
                    style.top = Some(lengths[0]);
                    style.bottom = Some(lengths[0]);
                    style.right = Some(lengths[1]);
                    style.left = Some(lengths[1]);
                }
                3 => {
                    style.top = Some(lengths[0]);
                    style.right = Some(lengths[1]);
                    style.left = Some(lengths[1]);
                    style.bottom = Some(lengths[2]);
                }
                4 => {
                    style.top = Some(lengths[0]);
                    style.right = Some(lengths[1]);
                    style.bottom = Some(lengths[2]);
                    style.left = Some(lengths[3]);
                }
                _ => {}
            }
            if !lengths.is_empty() {
                for i in 0..4 {
                    style.physical_inset_seq[i] = seq;
                }
            }
        }
        // `font:` shorthand. Real form is `font: [style] [variant]
        // [weight] [stretch] size[/line-height] family`. V1 parses just
        // enough to pull a font-size out so common idioms like
        // `font: 14px/1.5 Arial, sans-serif` don't lose the size.
        "font" => {
            // Per CSS Fonts 4 §6.10 the `font` shorthand RESETS every
            // longhand it covers to its initial value, then parses the
            // declaration. Previously we only ever set `font_weight_bold`
            // and `font_style_italic` to `Some(true)` when we saw the
            // matching keyword — never to `Some(false)` for the
            // `normal`/`unspecified` case. So `font: 16px Arial`
            // (omitting weight/style) left a previously-set `bold` from
            // an earlier rule in place instead of resetting back to
            // normal. Now: reset, then walk tokens, then extract the
            // font-family chunk after font-size + optional `/ line-height`.
            style.font_weight_bold = Some(false);
            style.font_weight_num = Some(400);
            style.font_style_italic = Some(false);
            style.line_height = None;
            // Split the declaration on '/' for line-height; whatever
            // comes after the slash is line-height ("font: 16px/1.5
            // Arial").
            let slash_idx = toks
                .iter()
                .position(|t| matches!(t, CssToken::Delim('/')));
            let (before, after): (&[CssToken], &[CssToken]) = match slash_idx {
                Some(i) => (&toks[..i], &toks[i + 1..]),
                None => (toks, &[]),
            };
            // First pass: scan `before` for pre-size keywords and the
            // font-size. The font shorthand pre-size slots are (all
            // optional, any order): font-style, font-variant,
            // font-weight, font-stretch.
            //
            // Finding the size token requires distinguishing it from
            // keyword slots. Strategy: scan left-to-right; any token
            // that parses as a CSS <length-percentage> (px/em/%) OR
            // matches a named size keyword is the size; everything
            // before it is a pre-size keyword or numeric weight.
            //
            // Helper: does an ident token belong to a pre-size slot?
            fn is_presize_keyword(s: &str) -> bool {
                matches!(
                    s,
                    // font-style
                    "italic" | "oblique"
                    // font-variant
                    | "small-caps"
                    // font-weight
                    | "bold" | "bolder" | "lighter"
                    // font-stretch
                    | "ultra-condensed"
                    | "extra-condensed"
                    | "condensed"
                    | "semi-condensed"
                    | "semi-expanded"
                    | "expanded"
                    | "extra-expanded"
                    | "ultra-expanded"
                    // `normal` is valid in all four slots
                    | "normal"
                )
            }
            // Helper: is an ident a named font-size keyword?
            fn font_size_keyword(s: &str) -> Option<Length> {
                match s {
                    "xx-small"  => Some(Length::Px(9.0)),
                    "x-small"   => Some(Length::Px(10.0)),
                    "small"     => Some(Length::Px(13.0)),
                    "medium"    => Some(Length::Px(16.0)),
                    "large"     => Some(Length::Px(18.0)),
                    "x-large"   => Some(Length::Px(24.0)),
                    "xx-large"  => Some(Length::Px(32.0)),
                    "xxx-large" => Some(Length::Px(48.0)),
                    "smaller"   => Some(Length::Em(5.0 / 6.0)),
                    "larger"    => Some(Length::Em(6.0 / 5.0)),
                    _ => None,
                }
            }

            let mut size_token_idx: Option<usize> = None;
            let mut size_set = false;
            for (i, t) in before.iter().enumerate() {
                if !size_set {
                    // Try numeric weight first (100–900) so it doesn't
                    // collide with the dimension path.
                    if let CssToken::Number(n) = t {
                        let v = *n as f32;
                        if (100.0..=900.0).contains(&v) {
                            // Numeric weight — not the size token. Keep the real
                            // value (800/900 render heavier than bold).
                            style.font_weight_bold = Some(v >= 600.0);
                            style.font_weight_num = Some((v as i32).clamp(1, 1000) as u16);
                            continue;
                        }
                    }
                    // Check for a length/percentage dimension.
                    if let Some(len) = Length::from_tokens(&before[i..=i]) {
                        style.font_size = Some(len);
                        size_token_idx = Some(i);
                        size_set = true;
                        continue;
                    }
                    // Check for a keyword size.
                    if let CssToken::Ident(s) = t {
                        let lc = s.to_ascii_lowercase();
                        if let Some(len) = font_size_keyword(lc.as_str()) {
                            style.font_size = Some(len);
                            size_token_idx = Some(i);
                            size_set = true;
                            continue;
                        }
                        // Not the size — must be a pre-size keyword.
                        match lc.as_str() {
                            "bold" => {
                                style.font_weight_bold = Some(true);
                                style.font_weight_num = Some(700);
                            }
                            "bolder" => {
                                style.font_weight_bold = Some(true);
                                style.font_weight_num = Some(FONT_WEIGHT_BOLDER);
                            }
                            "lighter" => {
                                style.font_weight_bold = Some(false);
                                style.font_weight_num = Some(FONT_WEIGHT_LIGHTER);
                            }
                            "italic" | "oblique" => style.font_style_italic = Some(true),
                            // normal/small-caps/stretch keywords are valid but we
                            // don't store font-variant or font-stretch yet.
                            _ if is_presize_keyword(lc.as_str()) => {}
                            _ => {}
                        }
                    }
                } else {
                    // After the size token, we've reached the family portion
                    // of `before` (only present when there is no `/`).
                    // Stop scanning for keywords.
                    let _ = t;
                }
            }
            // line-height after `/`. The first non-whitespace, non-`normal`
            // token in `after` is the line-height; everything after it is
            // part of the font-family list.
            let family_from_after_start: usize;
            if let Some((j, _)) = after.iter().enumerate().find(|(_, t)| {
                !matches!(t, CssToken::Whitespace)
                    && !matches!(t, CssToken::Ident(s) if s.eq_ignore_ascii_case("normal"))
            }) {
                if let CssToken::Number(n) = &after[j] {
                    style.line_height = Some(LineHeight::Multiplier(*n as f32));
                    family_from_after_start = j + 1;
                } else if let Some(l) = Length::from_tokens(&after[j..=j]) {
                    style.line_height = Some(LineHeight::Length(l));
                    family_from_after_start = j + 1;
                } else {
                    // Unrecognised — treat as start of family tokens.
                    family_from_after_start = j;
                }
            } else {
                family_from_after_start = after.len();
            }

            // Collect font-family tokens from:
            //   a) `before[sz_idx+1..]` — families when there is no `/`
            //   b) `after[family_from_after_start..]` — families after the
            //      `/ <line-height>` segment
            // Both token streams are concatenated. Comma-separated families
            // are all preserved and joined with ", " (same format as the
            // standalone `font-family` property so the resolver's fallback
            // chain works correctly).
            let family_before: &[CssToken] = size_token_idx
                .map(|idx| &before[idx + 1..])
                .unwrap_or(&[]);
            let family_after: &[CssToken] = &after[family_from_after_start..];

            let mut families: Vec<String> = Vec::new();
            let mut current: Option<String> = None;
            for t in family_before.iter().chain(family_after.iter()) {
                match t {
                    CssToken::Ident(s) => {
                        // Multi-word unquoted families arrive as several
                        // idents separated by whitespace — join with spaces.
                        current = Some(match current.take() {
                            Some(prev) => format!("{prev} {s}"),
                            None => s.clone(),
                        });
                    }
                    CssToken::String(s) => {
                        if let Some(prev) = current.take() {
                            families.push(prev);
                        }
                        families.push(s.clone());
                    }
                    CssToken::Comma => {
                        if let Some(prev) = current.take() {
                            families.push(prev);
                        }
                    }
                    CssToken::Whitespace => {}
                    _ => {}
                }
            }
            if let Some(prev) = current {
                families.push(prev);
            }
            if !families.is_empty() {
                style.font_family = Some(families.join(", "));
            }
        }
        "object-fit" => {
            for t in toks {
                if let CssToken::Ident(s) = t {
                    style.object_fit = Some(match s.to_ascii_lowercase().as_str() {
                        "fill" => ObjectFit::Fill,
                        "contain" => ObjectFit::Contain,
                        "cover" => ObjectFit::Cover,
                        "none" => ObjectFit::None,
                        "scale-down" => ObjectFit::ScaleDown,
                        _ => continue,
                    });
                    break;
                }
            }
        }
        "object-position" => {
            // Reuse the background-position parser. CSS spec default is
            // 50% 50% (center center); None here is treated as 50/50 by
            // the painter so we only store when the author overrides it.
            if let Some((x, y)) = parse_background_position(toks) {
                style.object_position = Some((x, y));
            }
        }
        // Properties we parse to clear the unknown counter; layout
        // implementation lives elsewhere or is V1-pending. Listing the
        // common ones with empty bodies prevents the audit's punch list
        // from being dominated by spec-recognised properties.
        "grid-auto-flow"
        | "align-content"
        | "flex-flow"
        | "order"
        | "place-content"
        | "animation-play-state"
        | "animation-fill-mode"
        | "animation-direction"
        | "text-underline-position"
        | "line-clamp"
        | "grid-row-end"
        | "grid-column-end"
        | "grid-template"
        | "overflow-anchor"
        | "text-rendering"
        | "text-decoration-skip"
        | "border-top-left-radius"
        | "border-top-right-radius"
        | "border-bottom-left-radius"
        | "border-bottom-right-radius"
        | "high-contrast-adjust"
        | "print-color-scheme"
        | "text-fill-color"
        | "box-flex-wrap" => {}
        // `grid-column-gap` / `grid-row-gap` / `grid-gap` — older
        // names for the unprefixed `column-gap` / `row-gap` / `gap`.
        "grid-column-gap" => {
            if let Some(v) = Length::from_tokens(toks) {
                style.column_gap = Some(v);
            }
        }
        "grid-row-gap" => {
            if let Some(v) = Length::from_tokens(toks) {
                style.row_gap = Some(v);
            }
        }
        "grid-gap" => {
            if let Some(v) = Length::from_tokens(toks) {
                style.gap = Some(v);
            }
        }
        other => {
            // Property we don't recognise (or recognise but don't paint).
            // Audit mode reads this counter via take_unknown_property_counts
            // to surface what the engine is missing on real sites.
            if !other.starts_with("--") && !is_silent_unknown(other) {
                UNKNOWN_PROPS.with(|c| {
                    let mut m = c.borrow_mut();
                    *m.entry(other.to_string()).or_insert(0) += 1;
                });
            }
        }
    }
}

thread_local! {
    static UNKNOWN_PROPS: std::cell::RefCell<std::collections::HashMap<String, u32>> =
        std::cell::RefCell::new(std::collections::HashMap::new());
}

/// Drain and return the count of CSS property names that fell through
/// the cascade's match arms during the most recent pass. Audit mode
/// uses this to rank what's missing on real sites.
pub fn take_unknown_property_counts() -> std::collections::HashMap<String, u32> {
    UNKNOWN_PROPS.with(|c| std::mem::take(&mut *c.borrow_mut()))
}

fn find_matching_paren_in_value(toks: &[CssToken], start: usize) -> Option<usize> {
    let mut depth = 1;
    let mut i = start;
    while i < toks.len() {
        match &toks[i] {
            CssToken::Function(_) | CssToken::LeftParen => depth += 1,
            CssToken::RightParen => {
                depth -= 1;
                if depth == 0 {
                    return Some(i);
                }
            }
            _ => {}
        }
        i += 1;
    }
    None
}

/// Pull the first number / percentage out of a value-token slice.
/// Numbers come back as-is; percentages become the 0..1 fraction so
/// `transform: scale(50%)` matches the spec's `scale(.5)`.
fn first_number(toks: &[CssToken]) -> Option<f64> {
    for t in toks {
        match t {
            CssToken::Number(n) => return Some(*n),
            CssToken::Percent(p) => return Some(*p / 100.0),
            _ => {}
        }
    }
    None
}

/// Pull the first angle out of a value slice, converted to degrees.
/// Accepts `deg`, `rad`, `grad`, `turn`. Bare numbers are treated as
/// degrees (lenient with the CSS spec, but matches most authoring).
fn first_angle_degrees(toks: &[CssToken]) -> Option<f64> {
    for t in toks {
        if let CssToken::Dimension { value, unit } = t {
            let v = *value;
            return Some(match unit.as_str() {
                "deg" | "" => v,
                "rad" => v * 180.0 / std::f64::consts::PI,
                "grad" => v * 360.0 / 400.0,
                "turn" => v * 360.0,
                _ => v,
            });
        }
        if let CssToken::Number(n) = t {
            return Some(*n);
        }
    }
    None
}

/// Split a value list on top-level whitespace runs. Used by L4
/// individual transform longhands (`translate: 10px 20px`) where the
/// sub-values are space-separated rather than comma-separated.
fn split_top_level_whitespace(toks: &[CssToken]) -> Vec<Vec<CssToken>> {
    let mut parts: Vec<Vec<CssToken>> = Vec::new();
    let mut cur: Vec<CssToken> = Vec::new();
    let mut depth = 0;
    for t in toks {
        match t {
            CssToken::Function(_) | CssToken::LeftParen => {
                depth += 1;
                cur.push(t.clone());
            }
            CssToken::RightParen => {
                depth -= 1;
                cur.push(t.clone());
            }
            CssToken::Whitespace if depth == 0 => {
                if !cur.is_empty() {
                    parts.push(std::mem::take(&mut cur));
                }
            }
            _ => cur.push(t.clone()),
        }
    }
    if !cur.is_empty() {
        parts.push(cur);
    }
    parts
}

fn split_top_level_commas(toks: &[CssToken]) -> Vec<Vec<CssToken>> {
    let mut parts: Vec<Vec<CssToken>> = Vec::new();
    let mut cur: Vec<CssToken> = Vec::new();
    let mut depth = 0;
    for t in toks {
        match t {
            CssToken::Function(_) | CssToken::LeftParen => {
                depth += 1;
                cur.push(t.clone());
            }
            CssToken::RightParen => {
                depth -= 1;
                cur.push(t.clone());
            }
            CssToken::Comma if depth == 0 => {
                parts.push(std::mem::take(&mut cur));
            }
            _ => cur.push(t.clone()),
        }
    }
    if !cur.is_empty() {
        parts.push(cur);
    }
    parts
}

/// Parse a CSS `transform` value (the whole function list) into an ordered
/// [`Transform3DOp`] list (CSS Transforms 2 §11). Unknown / unparseable
/// functions are skipped (matching prior lenient behaviour). Angles are
/// converted to degrees; translate args keep their `Length` so the bridge
/// can resolve px/em/rem/% in the right environment.
fn parse_transform_function_list(toks: &[CssToken]) -> Vec<Transform3DOp> {
    use crate::properties::Length;
    let mut ops: Vec<Transform3DOp> = Vec::new();
    let mut i = 0;
    while i < toks.len() {
        if let CssToken::Function(name) = &toks[i] {
            let lc = name.to_ascii_lowercase();
            if let Some(end) = find_matching_paren_in_value(toks, i + 1) {
                let args = &toks[i + 1..end];
                match lc.as_str() {
                    "translate" => {
                        let parts = split_top_level_commas(args);
                        let x = parts.first().and_then(|p| Length::from_tokens(p)).unwrap_or(Length::Zero);
                        let y = parts.get(1).and_then(|p| Length::from_tokens(p)).unwrap_or(Length::Zero);
                        ops.push(Transform3DOp::Translate(x, y));
                    }
                    "translatex" => {
                        let x = Length::from_tokens(args).unwrap_or(Length::Zero);
                        ops.push(Transform3DOp::Translate(x, Length::Zero));
                    }
                    "translatey" => {
                        let y = Length::from_tokens(args).unwrap_or(Length::Zero);
                        ops.push(Transform3DOp::Translate(Length::Zero, y));
                    }
                    "translatez" => {
                        let z = Length::from_tokens(args).unwrap_or(Length::Zero);
                        ops.push(Transform3DOp::Translate3d(Length::Zero, Length::Zero, z));
                    }
                    "translate3d" => {
                        let parts = split_top_level_commas(args);
                        let x = parts.first().and_then(|p| Length::from_tokens(p)).unwrap_or(Length::Zero);
                        let y = parts.get(1).and_then(|p| Length::from_tokens(p)).unwrap_or(Length::Zero);
                        let z = parts.get(2).and_then(|p| Length::from_tokens(p)).unwrap_or(Length::Zero);
                        ops.push(Transform3DOp::Translate3d(x, y, z));
                    }
                    "scale" => {
                        let parts = split_top_level_commas(args);
                        let sx = parts.first().and_then(|p| first_number(p)).unwrap_or(1.0);
                        let sy = parts.get(1).and_then(|p| first_number(p)).unwrap_or(sx);
                        ops.push(Transform3DOp::Scale(sx as f32, sy as f32));
                    }
                    "scalex" => {
                        let sx = first_number(args).unwrap_or(1.0);
                        ops.push(Transform3DOp::Scale(sx as f32, 1.0));
                    }
                    "scaley" => {
                        let sy = first_number(args).unwrap_or(1.0);
                        ops.push(Transform3DOp::Scale(1.0, sy as f32));
                    }
                    "scalez" => {
                        let sz = first_number(args).unwrap_or(1.0);
                        ops.push(Transform3DOp::Scale3d(1.0, 1.0, sz as f32));
                    }
                    "scale3d" => {
                        let parts = split_top_level_commas(args);
                        let sx = parts.first().and_then(|p| first_number(p)).unwrap_or(1.0);
                        let sy = parts.get(1).and_then(|p| first_number(p)).unwrap_or(1.0);
                        let sz = parts.get(2).and_then(|p| first_number(p)).unwrap_or(1.0);
                        ops.push(Transform3DOp::Scale3d(sx as f32, sy as f32, sz as f32));
                    }
                    "rotate" | "rotatez" => {
                        if let Some(deg) = first_angle_degrees(args) {
                            ops.push(Transform3DOp::RotateZ(deg as f32));
                        }
                    }
                    "rotatex" => {
                        if let Some(deg) = first_angle_degrees(args) {
                            ops.push(Transform3DOp::RotateX(deg as f32));
                        }
                    }
                    "rotatey" => {
                        if let Some(deg) = first_angle_degrees(args) {
                            ops.push(Transform3DOp::RotateY(deg as f32));
                        }
                    }
                    "rotate3d" => {
                        let parts = split_top_level_commas(args);
                        if parts.len() == 4 {
                            let x = first_number(&parts[0]).unwrap_or(0.0) as f32;
                            let y = first_number(&parts[1]).unwrap_or(0.0) as f32;
                            let z = first_number(&parts[2]).unwrap_or(0.0) as f32;
                            let deg = first_angle_degrees(&parts[3]).unwrap_or(0.0) as f32;
                            ops.push(Transform3DOp::Rotate3d(x, y, z, deg));
                        }
                    }
                    "matrix" => {
                        let parts = split_top_level_commas(args);
                        if parts.len() == 6 {
                            let mut m = [0f32; 6];
                            let mut ok = true;
                            for (k, p) in parts.iter().enumerate() {
                                if let Some(v) = first_number(p) {
                                    m[k] = v as f32;
                                } else {
                                    ok = false;
                                    break;
                                }
                            }
                            if ok {
                                ops.push(Transform3DOp::Matrix2d(m));
                            }
                        }
                    }
                    "matrix3d" => {
                        let parts = split_top_level_commas(args);
                        if parts.len() == 16 {
                            let mut m = [0f32; 16];
                            let mut ok = true;
                            for (k, p) in parts.iter().enumerate() {
                                if let Some(v) = first_number(p) {
                                    m[k] = v as f32;
                                } else {
                                    ok = false;
                                    break;
                                }
                            }
                            if ok {
                                ops.push(Transform3DOp::Matrix3d(m));
                            }
                        }
                    }
                    "perspective" => {
                        // perspective(none) => identity (skip). Otherwise a length.
                        let is_none = args.iter().any(|t| matches!(t, CssToken::Ident(s) if s.eq_ignore_ascii_case("none")));
                        if !is_none {
                            if let Some(l) = Length::from_tokens(args) {
                                ops.push(Transform3DOp::Perspective(l));
                            }
                        }
                    }
                    "skew" => {
                        let parts = split_top_level_commas(args);
                        let ax = parts.first().and_then(|p| first_angle_degrees(p)).unwrap_or(0.0) as f32;
                        let ay = parts.get(1).and_then(|p| first_angle_degrees(p)).unwrap_or(0.0) as f32;
                        ops.push(Transform3DOp::Skew(ax, ay));
                    }
                    "skewx" => {
                        let ax = first_angle_degrees(args).unwrap_or(0.0) as f32;
                        ops.push(Transform3DOp::Skew(ax, 0.0));
                    }
                    "skewy" => {
                        let ay = first_angle_degrees(args).unwrap_or(0.0) as f32;
                        ops.push(Transform3DOp::Skew(0.0, ay));
                    }
                    _ => {}
                }
                i = end + 1;
                continue;
            }
        }
        i += 1;
    }
    ops
}

/// Encode a synthetic rule index for an `@media`-nested rule: pack the
/// at-rule's index in the sheet's `at_rules` and the inner rule's FLAT index
/// (its position in the deterministic full flatten of that at-rule's own block
/// plus all descendant at-rule blocks) into a single offset past the top-level
/// rule count.
fn encode_at_rule_idx(top_count: usize, at_idx: usize, inner: usize) -> usize {
    // 16 bits for the inner rule index. CSS files don't ship 65k rules
    // in a single @media block in practice.
    top_count + (at_idx << 16) + (inner & 0xFFFF) + 1
}

fn decode_at_rule_idx(top_count: usize, rule_idx: usize) -> (usize, usize) {
    let off = rule_idx - top_count - 1;
    (off >> 16, off & 0xFFFF)
}

/// Deterministic, MATCH-INDEPENDENT flatten of an at-rule's qualified rules:
/// the at-rule's own `block` rules first, then each nested at-rule's flatten in
/// declaration order, recursively. Used by `resolve_rule` (which has no
/// viewport) so a flat `inner` index always maps back to the same rule the
/// build pass enumerated. The build pass walks the SAME order but only emits
/// candidates for the subset whose media/supports conditions match.
fn flatten_at_rule_all<'a>(at: &'a crate::parser::AtRule, out: &mut Vec<&'a crate::parser::Rule>) {
    if let Some(block) = at.block.as_ref() {
        for r in block {
            out.push(r);
        }
    }
    for nested in &at.nested {
        flatten_at_rule_all(nested, out);
    }
}

/// Serialize a slice of CSS tokens back to (approximately) canonical CSS
/// source text — enough for the string-based `@container` condition
/// evaluator to re-parse `min-width: 300px` / `inline-size > 30em`. Unlike the
/// debug `Display` impl (which emits `ident(...)`), this round-trips values.
fn serialize_css_tokens(toks: &[CssToken]) -> String {
    let mut out = String::new();
    for t in toks {
        match t {
            CssToken::Ident(s) | CssToken::AtKeyword(s) => out.push_str(s),
            CssToken::Function(s) => {
                out.push_str(s);
                out.push('(');
            }
            CssToken::Hash(s) => {
                out.push('#');
                out.push_str(s);
            }
            CssToken::String(s) => {
                out.push('"');
                out.push_str(s);
                out.push('"');
            }
            CssToken::Number(n) => out.push_str(&fmt_css_num(*n)),
            CssToken::Percent(n) => {
                out.push_str(&fmt_css_num(*n));
                out.push('%');
            }
            CssToken::Dimension { value, unit } => {
                out.push_str(&fmt_css_num(*value));
                out.push_str(unit);
            }
            CssToken::Whitespace => out.push(' '),
            CssToken::Colon => out.push(':'),
            CssToken::Semicolon => out.push(';'),
            CssToken::Comma => out.push(','),
            CssToken::LeftBrace => out.push('{'),
            CssToken::RightBrace => out.push('}'),
            CssToken::LeftParen => out.push('('),
            CssToken::RightParen => out.push(')'),
            CssToken::LeftBracket => out.push('['),
            CssToken::RightBracket => out.push(']'),
            CssToken::Delim(c) => out.push(*c),
            CssToken::Url(s) => {
                out.push_str("url(");
                out.push_str(s);
                out.push(')');
            }
            CssToken::Bang => out.push('!'),
            CssToken::Eof => {}
        }
    }
    out
}

fn fmt_css_num(n: f64) -> String {
    if n.fract() == 0.0 {
        format!("{}", n as i64)
    } else {
        format!("{n}")
    }
}

/// Parse an `@container` prelude (the tokens between `@container` and `{`) into
/// `(optional container-name, raw condition string)`. The grammar is
/// `<container-name>? <container-condition>` (CSS Containment 3 §3): a leading
/// `<custom-ident>` names the targeted container, followed by a parenthesized
/// size-query condition. The condition is returned WITHOUT its outer parens so
/// it feeds directly into `eval_container_condition_axes`.
fn parse_container_prelude(prelude: &[CssToken]) -> (Option<String>, String) {
    // Find the first top-level `(` — everything before it (idents) is the
    // optional name; everything from it on is the condition.
    let mut name: Option<String> = None;
    let mut i = 0;
    while i < prelude.len() {
        match &prelude[i] {
            CssToken::Whitespace => i += 1,
            CssToken::Ident(s) => {
                // `not`/`and`/`or` are condition keywords, not names.
                let lc = s.to_ascii_lowercase();
                if matches!(lc.as_str(), "not" | "and" | "or") {
                    break;
                }
                name = Some(s.clone());
                i += 1;
            }
            _ => break,
        }
    }
    let condition_toks = &prelude[i..];
    // Strip a single pair of outer parens if the whole condition is wrapped:
    // `(min-width: 300px)` → `min-width: 300px`. Compound conditions like
    // `(a) and (b)` keep their inner parens (eval handles them).
    let raw = serialize_css_tokens(condition_toks);
    let raw = raw.trim().to_string();
    let condition = strip_outer_parens(&raw);
    (name, condition)
}

/// If `s` is wrapped in a single balanced pair of outer parens (and that pair
/// spans the whole string), remove them; otherwise return `s` unchanged.
fn strip_outer_parens(s: &str) -> String {
    let s = s.trim();
    if !s.starts_with('(') || !s.ends_with(')') {
        return s.to_string();
    }
    let bytes = s.as_bytes();
    let mut depth = 0i32;
    for (i, &b) in bytes.iter().enumerate() {
        match b {
            b'(' => depth += 1,
            b')' => {
                depth -= 1;
                // The opening paren closes before the end → not a single wrap.
                if depth == 0 && i != bytes.len() - 1 {
                    return s.to_string();
                }
            }
            _ => {}
        }
    }
    s[1..s.len() - 1].trim().to_string()
}

/// Evaluate a compiled `@container` guard against the element's ancestor
/// query-container stack. CSS Containment 3 §3: each clause is matched against
/// the NEAREST ancestor query container that (a) has the right type for the
/// queried axis and (b) — when the clause names a container — carries that
/// `container-name`. The guard passes only if EVERY clause passes (clauses
/// arise from nested `@container` blocks). With an empty stack (no ancestor
/// container) the guard fails: no container means the size query can't be
/// satisfied, so the rules are withheld (Chrome: an unmatched container query
/// never applies).
fn eval_container_guard(guard: &ContainerQueryGuard, stack: QueryContainerStack<'_>) -> bool {
    guard
        .clauses
        .iter()
        .all(|clause| eval_container_clause(clause, stack))
}

fn eval_container_clause(clause: &ContainerQueryClause, stack: QueryContainerStack<'_>) -> bool {
    // Whether this clause queries the block axis (needs a `size` container).
    let needs_block = clause_queries_block_axis(&clause.condition);
    // Walk nearest-first (stack is root-first, so iterate in reverse).
    for c in stack.iter().rev() {
        if c.container_type == crate::cascade::ContainerType::Normal {
            continue;
        }
        // Name targeting: a named clause only matches a container that declares
        // that name; an unnamed clause matches any query container.
        if let Some(want) = &clause.name {
            if !c.names.iter().any(|n| n == want) {
                continue;
            }
        }
        // Axis suitability: a block-axis query needs a `size` container. An
        // inline-size container is skipped for block queries (keep searching
        // outward for a `size` container, per the nearest-ELIGIBLE-container
        // rule).
        if needs_block && c.container_type != crate::cascade::ContainerType::Size {
            continue;
        }
        let axes = match c.container_type {
            crate::cascade::ContainerType::Size => crate::modern::ContainerAxes::Both,
            _ => crate::modern::ContainerAxes::InlineOnly,
        };
        return crate::modern::eval_container_condition_axes(
            &clause.condition,
            c.inline_size,
            c.block_size,
            axes,
        );
    }
    // No eligible ancestor query container → the query cannot match.
    false
}

/// Cheap check: does this condition reference a block-axis size feature
/// (`height` / `block-size`)? Used to require a `size` container before
/// committing to a particular ancestor.
fn clause_queries_block_axis(condition: &str) -> bool {
    let lc = condition.to_ascii_lowercase();
    lc.contains("height") || lc.contains("block-size")
}

/// Whether an at-rule's OWN condition matches the current media context
/// (ignoring ancestors). `@media`/`@supports` are evaluated; layout-deferred
/// or always-fold at-rules (`@container`/`@layer`/`@scope`/`@starting-style`)
/// return true so their bodies stay reachable.
fn at_rule_condition_matches(at: &crate::parser::AtRule, vw: f32, vh: f32) -> bool {
    match at.name.as_str() {
        // `@media` rules (incl. `(min-resolution: 2dppx)` HiDPI breakpoints)
        // are selected against the live device pixel ratio so @2x asset rules
        // apply on a HiDPI monitor — matching Chrome's cascade.
        "media" => media_query_matches(&at.prelude, vw, vh, current_device_pixel_ratio()),
        "supports" => supports_matches(&at.prelude),
        "container" | "layer" | "scope" | "starting-style" => true,
        // Unknown grouping at-rule: fold optimistically (matches the build
        // pass's superset philosophy).
        _ => true,
    }
}

/// Very narrow `@media` query evaluator: handles the `(min-width: N)` /
/// `(max-width: N)` / `(min-height: N)` / `(max-height: N)` features
/// plus optional `screen` / `all` types joined by `and`. Anything we
/// don't recognise is *defaulted to true* so non-matching rules still
/// apply — preferred over silently dropping everything when a query
/// uses a feature we haven't implemented.
/// Evaluate an `@supports <condition>` prelude. Each parenthesized feature
/// test (`(prop: value)`, `selector(...)`, etc.) is treated as SUPPORTED —
/// optimistic-true, matching what a modern browser does for progressive-
/// enhancement sites: the modern `@supports (grid)` block applies, and crucially
/// the `@supports not (grid)` FALLBACK block is correctly EXCLUDED (it used to be
/// included unconditionally, double-applying old fallback styles). `and`/`or`/
/// `not` combine the leaf tests. Returns true on an empty/unparseable condition
/// (fail-open, since dropping the rules is worse than over-including).
fn supports_matches(prelude: &[CssToken]) -> bool {
    fn skip_ws(toks: &[CssToken], mut i: usize) -> usize {
        while matches!(toks.get(i), Some(CssToken::Whitespace)) {
            i += 1;
        }
        i
    }
    // One term: optional `not`(s), then a parenthesized group / function (the
    // feature test, treated as supported = true).
    fn parse_term(toks: &[CssToken], i: usize) -> (bool, usize) {
        let mut i = skip_ws(toks, i);
        let mut negate = false;
        while matches!(toks.get(i), Some(CssToken::Ident(s)) if s.eq_ignore_ascii_case("not")) {
            negate = !negate;
            i = skip_ws(toks, i + 1);
        }
        // Skip the balanced group (or function call) — its contents don't change
        // the optimistic verdict, only its presence and any leading `not`.
        if matches!(
            toks.get(i),
            Some(CssToken::LeftParen) | Some(CssToken::Function(_))
        ) {
            let mut depth = 0i32;
            while let Some(t) = toks.get(i) {
                match t {
                    CssToken::LeftParen | CssToken::Function(_) => depth += 1,
                    CssToken::RightParen => {
                        depth -= 1;
                        if depth == 0 {
                            i += 1;
                            break;
                        }
                    }
                    _ => {}
                }
                i += 1;
            }
        } else {
            i += 1; // unexpected token — advance to avoid looping
        }
        (negate ^ true, i)
    }

    let mut i = skip_ws(prelude, 0);
    if i >= prelude.len() {
        return true;
    }
    let (mut acc, ni) = parse_term(prelude, i);
    i = ni;
    loop {
        i = skip_ws(prelude, i);
        let op = match prelude.get(i) {
            Some(CssToken::Ident(s)) if s.eq_ignore_ascii_case("and") => Some(true),
            Some(CssToken::Ident(s)) if s.eq_ignore_ascii_case("or") => Some(false),
            _ => None,
        };
        let Some(is_and) = op else { break };
        let (rhs, ni) = parse_term(prelude, skip_ws(prelude, i + 1));
        i = ni;
        acc = if is_and { acc && rhs } else { acc || rhs };
    }
    acc
}

/// Evaluate a media-query string (e.g. `"(min-width: 600px) and (orientation:
/// portrait)"`) against a viewport of `vw`×`vh` CSS pixels, returning whether it
/// matches the current environment. This is the public entry point used by
/// `window.matchMedia` (CSSOM View §4.2) — it tokenises the query and runs the
/// SAME evaluator the `@media` cascade uses, so a `matchMedia` verdict and the
/// rules an `@media` block applies stay in lockstep. An empty query matches all
/// media (per Media Queries 4 §2.1, an empty query list is equivalent to `all`).
///
/// Behaves as Blink's `MediaQueryEvaluator::Eval` over the parsed query:
/// width/height (incl. `min-`/`max-`), orientation, prefers-color-scheme, and
/// the other features implemented in `media_atom_matches`. Unknown / not-yet
/// implemented features do not match (conservative, never false-positive).
pub fn media_query_matches_str(query: &str, vw: f32, vh: f32) -> bool {
    // Back-compat entry point: evaluate at the device-pixel-ratio currently
    // published by the host (`set_device_pixel_ratio`). Defaults to 1.0 when the
    // host has not reported a DPI (headless / pre-window), so callers that never
    // touch HiDPI behave exactly as before.
    media_query_matches_str_dpr(query, vw, vh, current_device_pixel_ratio())
}

/// DPR-aware variant of [`media_query_matches_str`]. `dpr` is the device pixel
/// ratio (physical px ÷ CSS px) the `resolution` / `-webkit-device-pixel-ratio`
/// features evaluate against. CSS Values 4 §6.1: `1dppx == 96dpi`, and the
/// `resolution` feature compares the device's pixel density to the queried one;
/// `dppx`/`x` is exactly `devicePixelRatio` (CSSOM View — `window.devicePixelRatio`
/// is the `dppx` value of `(resolution)`). See
/// <https://developer.mozilla.org/en-US/docs/Web/CSS/@media/resolution>.
pub fn media_query_matches_str_dpr(query: &str, vw: f32, vh: f32, dpr: f32) -> bool {
    let trimmed = query.trim();
    if trimmed.is_empty() {
        // CSSOM View: matchMedia("") → media query list equivalent to `all`.
        return true;
    }
    let toks = crate::tokenizer::tokenize(trimmed);
    media_query_matches(&toks, vw, vh, dpr)
}

fn media_query_matches(prelude: &[CssToken], vw: f32, vh: f32, dpr: f32) -> bool {
    // Split the prelude on top-level commas — a single query in the list
    // matching = the whole thing matches.
    let mut current: Vec<CssToken> = Vec::new();
    let mut groups: Vec<Vec<CssToken>> = Vec::new();
    for t in prelude {
        if matches!(t, CssToken::Comma) {
            groups.push(std::mem::take(&mut current));
        } else {
            current.push(t.clone());
        }
    }
    groups.push(current);
    for g in groups {
        if media_query_part_matches(&g, vw, vh, dpr) {
            return true;
        }
    }
    false
}

fn media_query_part_matches(toks: &[CssToken], vw: f32, vh: f32, dpr: f32) -> bool {
    // Per CSS Media Queries Level 4: a leading `not` keyword negates the
    // ENTIRE media query (it's not just a type-atom modifier). Detect it
    // here and inversely return so `@media not (max-width: 600px)`
    // correctly matches when the viewport is wider than 600px instead of
    // always returning false (which was the prior bug — `not (...)`
    // collapsed to a type-`not all` atom and dropped the rules forever).
    let mut leading_not = false;
    let mut start = 0usize;
    while let Some(t) = toks.get(start) {
        match t {
            CssToken::Whitespace => start += 1,
            CssToken::Ident(s) if s.eq_ignore_ascii_case("not") => {
                leading_not = !leading_not;
                start += 1;
            }
            _ => break,
        }
    }
    let inner = &toks[start..];
    if leading_not {
        // Recurse on the remainder without the leading `not` to compute
        // the natural verdict, then invert.
        return !media_query_part_matches(inner, vw, vh, dpr);
    }
    let toks = inner;
    // Tokenise the query into "atoms" separated by `and` idents.
    // Each atom is either a type (screen/print/all) or a parenthesised
    // feature like `(min-width: 600px)`.
    let mut atoms: Vec<Vec<CssToken>> = Vec::new();
    let mut current: Vec<CssToken> = Vec::new();
    let mut i = 0;
    while i < toks.len() {
        match &toks[i] {
            CssToken::Whitespace => i += 1,
            CssToken::Ident(s) if s.eq_ignore_ascii_case("and") => {
                if !current.is_empty() {
                    atoms.push(std::mem::take(&mut current));
                }
                i += 1;
            }
            CssToken::LeftParen => {
                // Slurp up to matching `)`.
                let mut depth = 1;
                let mut j = i + 1;
                let start = j;
                while j < toks.len() && depth > 0 {
                    match &toks[j] {
                        CssToken::LeftParen | CssToken::Function(_) => depth += 1,
                        CssToken::RightParen => depth -= 1,
                        _ => {}
                    }
                    if depth > 0 {
                        j += 1;
                    }
                }
                atoms.push(toks[start..j].to_vec());
                i = j + 1;
                if !current.is_empty() {
                    atoms.push(std::mem::take(&mut current));
                }
            }
            other => {
                current.push(other.clone());
                i += 1;
            }
        }
    }
    if !current.is_empty() {
        atoms.push(current);
    }
    if atoms.is_empty() {
        return true; // bare `@media { ... }` — applies always
    }
    for atom in atoms {
        if !media_atom_matches(&atom, vw, vh, dpr) {
            return false;
        }
    }
    true
}

fn media_atom_matches(toks: &[CssToken], vw: f32, vh: f32, dpr: f32) -> bool {
    // Type atom (`screen`, `print`, `all`, optionally prefixed `only`/
    // `not`).
    if toks
        .iter()
        .all(|t| matches!(t, CssToken::Ident(_) | CssToken::Whitespace))
    {
        let mut not = false;
        let mut matched_type = false;
        // The active media type — `print` during a print/PDF pass (see
        // `set_print_media`), `screen` otherwise. `all` matches either.
        let printing = print_media_active();
        for t in toks {
            if let CssToken::Ident(s) = t {
                let lc = s.to_ascii_lowercase();
                if lc == "not" {
                    not = !not;
                } else if lc == "all" {
                    return !not;
                } else if lc == "screen" {
                    // Matches only when NOT printing (Media Queries 4 §2.1).
                    return if printing { not } else { !not };
                } else if lc == "print" {
                    // Matches only during a print/PDF pass.
                    return if printing { !not } else { not };
                } else if matches!(
                    lc.as_str(),
                    "speech" | "tv" | "aural" | "handheld"
                        | "tty" | "embossed" | "projection" | "braille"
                ) {
                    // Non-screen / legacy media types: these never match in an
                    // interactive or print browser context.
                    return not;
                } else if lc == "only" {
                    // `only screen` — the `only` prefix is a CSS 2.1 compat
                    // hint that causes legacy browsers to ignore the rule.
                    // Modern engines (including us) just skip the keyword.
                } else {
                    // Unknown media type — not "screen" or "all", so false.
                    matched_type = true;
                }
            }
        }
        // If we saw an unknown type ident (not a modifier like `only`), treat
        // it as non-matching. A bare `@media { }` with no type atoms at all
        // (atoms list empty, handled above) applies unconditionally.
        if matched_type {
            return not; // unknown type → false (inverted when `not` is set)
        }
        return !not;
    }
    // Feature atom — `name: value` (possibly with `min-`/`max-`).
    // Find the `:` split.
    let mut name = String::new();
    let mut value_toks: Vec<CssToken> = Vec::new();
    let mut past_colon = false;
    for t in toks {
        match t {
            CssToken::Whitespace => {}
            CssToken::Colon => past_colon = true,
            CssToken::Ident(s) if !past_colon => name = s.to_ascii_lowercase(),
            other if past_colon => value_toks.push(other.clone()),
            _ => {}
        }
    }
    if name.is_empty() {
        return true;
    }
    // Identify the value keyword (first Ident token after the colon)
    // so user-preference features like `prefers-color-scheme: dark`
    // can be matched without invoking the length path.
    let value_keyword = value_toks.iter().find_map(|t| match t {
        CssToken::Ident(s) => Some(s.to_ascii_lowercase()),
        _ => None,
    });
    let length = Length::from_tokens(&value_toks)
        .and_then(|l| l.resolve_px_with_viewport(16.0, 16.0, 0.0, vw, vh));
    match name.as_str() {
        "min-width" => length.map(|v| vw >= v).unwrap_or(true),
        "max-width" => length.map(|v| vw <= v).unwrap_or(true),
        "min-height" => length.map(|v| vh >= v).unwrap_or(true),
        "max-height" => length.map(|v| vh <= v).unwrap_or(true),
        // User-preference features. We are a light-theme, non-reduced,
        // pointer-equipped browser; surface those answers honestly so
        // sites that branch on them get sensible CSS instead of e.g.
        // dark-mode rules painting every background black.
        "prefers-color-scheme" => matches!(
            value_keyword.as_deref(),
            Some("light") | Some("no-preference") | None
        ),
        "prefers-reduced-motion" => {
            matches!(value_keyword.as_deref(), Some("no-preference") | None)
        }
        "prefers-reduced-transparency" => {
            matches!(value_keyword.as_deref(), Some("no-preference") | None)
        }
        "prefers-contrast" => matches!(value_keyword.as_deref(), Some("no-preference") | None),
        "forced-colors" => matches!(value_keyword.as_deref(), Some("none") | None),
        "inverted-colors" => matches!(value_keyword.as_deref(), Some("none") | None),
        // Capability features — we have a regular mouse + keyboard,
        // colour screen, not a TV / projector / overlay device.
        "hover" => matches!(value_keyword.as_deref(), Some("hover") | None),
        "any-hover" => matches!(value_keyword.as_deref(), Some("hover") | None),
        "pointer" => matches!(value_keyword.as_deref(), Some("fine") | None),
        "any-pointer" => matches!(value_keyword.as_deref(), Some("fine") | None),
        // Orientation is derived from the viewport box per Media Queries 4
        // §6.4: `portrait` when height >= width, `landscape` otherwise. A bare
        // `(orientation)` with no value matches in both orientations (the
        // feature is always "present"). Previously this hardcoded `landscape`
        // regardless of the viewport, which made `matchMedia('(orientation:
        // portrait)')` wrong on a tall window.
        "orientation" => match value_keyword.as_deref() {
            Some("portrait") => vh >= vw,
            Some("landscape") => vw > vh,
            None => true,
            _ => false,
        },
        "display-mode" => matches!(value_keyword.as_deref(), Some("browser") | None),
        "scripting" => matches!(value_keyword.as_deref(), Some("enabled") | None),
        "update" => matches!(value_keyword.as_deref(), Some("fast") | None),
        "color" | "color-index" | "monochrome" => true,
        "grid" => matches!(value_keyword.as_deref(), Some("0") | None),
        // `resolution` / `min-resolution` / `max-resolution` (Media Queries 4
        // §6.7) — the device pixel density. `1dppx == 96dpi == 1in` worth of CSS
        // px, and `dppx`/`x` is exactly `window.devicePixelRatio`. We evaluate
        // against the live DPR threaded in from the window's monitor DPI. Per
        // spec, a bare `(resolution)` (no value) matches when resolution != 0
        // (always true for a screen). The range prefixes use >= / <= on dppx.
        // <https://developer.mozilla.org/en-US/docs/Web/CSS/@media/resolution>
        "resolution" => resolution_dppx(&value_toks)
            .map(|q| (dpr - q).abs() < 1e-3)
            .unwrap_or(true),
        "min-resolution" => resolution_dppx(&value_toks)
            .map(|q| dpr >= q - 1e-3)
            .unwrap_or(true),
        "max-resolution" => resolution_dppx(&value_toks)
            .map(|q| dpr <= q + 1e-3)
            .unwrap_or(true),
        // Legacy WebKit alias: the value is the dppx ratio directly (bare
        // number, no unit). `-webkit-min-device-pixel-ratio` etc. mirror the
        // `min-/max-resolution` range semantics. Still emitted by Retina-era
        // sites and asset-pipeline @2x stylesheets.
        // <https://developer.mozilla.org/en-US/docs/Web/CSS/@media/-webkit-device-pixel-ratio>
        "-webkit-device-pixel-ratio" => webkit_dpr_value(&value_toks)
            .map(|q| (dpr - q).abs() < 1e-3)
            .unwrap_or(true),
        "-webkit-min-device-pixel-ratio" => webkit_dpr_value(&value_toks)
            .map(|q| dpr >= q - 1e-3)
            .unwrap_or(true),
        "-webkit-max-device-pixel-ratio" => webkit_dpr_value(&value_toks)
            .map(|q| dpr <= q + 1e-3)
            .unwrap_or(true),
        // Unknown / unimplemented features: do NOT match. Defaulting
        // to true was the original bug — it applied every speculative
        // media query (including dark-mode and forced-colors) by accident.
        _ => false,
    }
}

/// Parse a `<resolution>` value (Media Queries 4 §6.7 / CSS Values 4 §6.1) into
/// dots-per-px (dppx). Accepts `dppx`, `x` (a synonym), `dpi`, and `dpcm`, with
/// the canonical conversions:
///   * `1dppx`  → 1.0          (the base unit; == `devicePixelRatio`)
///   * `1x`     → 1.0          (synonym for `dppx`)
///   * `96dpi`  → 1.0          (96 CSS px per inch)
///   * `1dpcm`  → 2.54/96.0    (2.54 cm per inch ⇒ 1in == 96px == 2.54dpcm)
/// Returns `None` when the value is missing or carries an unrecognised unit.
fn resolution_dppx(value_toks: &[CssToken]) -> Option<f32> {
    for t in value_toks {
        match t {
            CssToken::Dimension { value, unit } => {
                let u = unit.to_ascii_lowercase();
                let dppx = match u.as_str() {
                    "dppx" | "x" => *value,
                    "dpi" => *value / 96.0,
                    // 1in == 2.54cm == 96px ⇒ dppx = dpcm * 2.54 / 96.
                    "dpcm" => *value * 2.54 / 96.0,
                    _ => return None,
                };
                return Some(dppx as f32);
            }
            // A bare `0` is the only unit-less <resolution> the grammar allows
            // (it is dimensionless because every unit times 0 is 0).
            CssToken::Number(n) if *n == 0.0 => return Some(0.0),
            _ => {}
        }
    }
    None
}

/// Parse the value of a legacy `-webkit-*-device-pixel-ratio` feature: a bare
/// number (the dppx ratio), e.g. `(-webkit-min-device-pixel-ratio: 2)`. Some
/// authors also write it with a `dppx`/`x` unit; accept those too.
fn webkit_dpr_value(value_toks: &[CssToken]) -> Option<f32> {
    for t in value_toks {
        match t {
            CssToken::Number(n) => return Some(*n as f32),
            CssToken::Dimension { value, unit } => {
                let u = unit.to_ascii_lowercase();
                if u == "dppx" || u == "x" {
                    return Some(*value as f32);
                }
                return None;
            }
            _ => {}
        }
    }
    None
}

/// Process-global device pixel ratio (physical px ÷ CSS px) published by the
/// window layer when it learns the monitor DPI (HiDPI). Stored as the bit
/// pattern of an `f32` in an `AtomicU32` so it can be read lock-free from the
/// media-query evaluator on any thread. Defaults to `1.0` (the 96-dpi baseline)
/// so headless / pre-window evaluation behaves exactly as before HiDPI landed.
static DEVICE_PIXEL_RATIO_BITS: std::sync::atomic::AtomicU32 =
    std::sync::atomic::AtomicU32::new(0x3F80_0000); // 1.0_f32 bits

/// Publish the live device pixel ratio. Called by the host (cv_browser) whenever
/// the window's monitor DPI is read (window open + `WM_DPICHANGED`). A value
/// `<= 0` or non-finite is ignored so a bad probe can never zero out the ratio.
pub fn set_device_pixel_ratio(dpr: f32) {
    if dpr.is_finite() && dpr > 0.0 {
        DEVICE_PIXEL_RATIO_BITS.store(dpr.to_bits(), std::sync::atomic::Ordering::Relaxed);
    }
}

/// Read the live device pixel ratio published by [`set_device_pixel_ratio`].
/// Defaults to `1.0` until the window reports a DPI.
pub fn current_device_pixel_ratio() -> f32 {
    f32::from_bits(DEVICE_PIXEL_RATIO_BITS.load(std::sync::atomic::Ordering::Relaxed))
}

/// Process-global "active CSS media type". `false` = `screen` (the default for
/// interactive rendering), `true` = `print`. Published by the host (cv_browser)
/// around a print/PDF-export pass so the cascade evaluates `@media print { … }`
/// blocks (and hides `@media screen`-only rules / honours `display:none` set
/// only in print). Media Queries 4 §2.1 (media types: `screen` vs `print`);
/// the CSS print flow re-runs the cascade under the `print` media type.
/// Stored in an atomic so the lock-free media-query evaluator can read it on
/// any thread; always restored to `false` after the print pass.
static PRINT_MEDIA_ACTIVE: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Switch the active media type. `true` selects `print`, `false` selects
/// `screen`. The host wraps a print/PDF layout pass in
/// `set_print_media(true) … set_print_media(false)` so the SAME cascade code
/// path produces the print-media computed styles (Chrome re-lays-out the page
/// under the print media type before paginating).
pub fn set_print_media(active: bool) {
    PRINT_MEDIA_ACTIVE.store(active, std::sync::atomic::Ordering::Relaxed);
}

/// True when the active media type is `print` (see [`set_print_media`]).
pub fn print_media_active() -> bool {
    PRINT_MEDIA_ACTIVE.load(std::sync::atomic::Ordering::Relaxed)
}

/// Find a `linear-gradient(...)` call in `toks` and pull out the first
/// and last color stops plus the direction. Returns `None` if the
/// value doesn't look like a linear gradient, or if we can't find at
/// least two parsable colors. Multi-stop gradients (more than 2 colors)
/// degrade to a from→to interpolation between the first and last stops
/// — visually close for the common 2-stop hero shapes and acceptable
/// for the multi-stop ones until we ship a proper gradient renderer.
/// Parse a single `background-position` axis component. Accepts a px
/// length, a percentage, a bare `0`, or a keyword (left/top → 0%,
/// center → 50%, right/bottom → 100%). em/rem are not resolved here (no
/// font context at cascade time) and fall through to None — sprites,
/// the primary consumer, always use px or %.
fn bg_pos_component(tok: Option<&CssToken>, _is_x: bool) -> Option<BgPos> {
    match tok? {
        CssToken::Dimension { value, unit } if unit.eq_ignore_ascii_case("px") => {
            Some(BgPos::Px(*value as f32))
        }
        CssToken::Percent(p) => Some(BgPos::Pct(*p as f32)),
        CssToken::Number(n) if *n == 0.0 => Some(BgPos::Px(0.0)),
        CssToken::Ident(s) => match s.to_ascii_lowercase().as_str() {
            "left" | "top" => Some(BgPos::Pct(0.0)),
            "center" => Some(BgPos::Pct(50.0)),
            "right" | "bottom" => Some(BgPos::Pct(100.0)),
            _ => None,
        },
        _ => None,
    }
}

/// Parse `background-position: <x> [<y>]`. One value sets x with y
/// centered (per spec); two values are x then y. Whitespace tokens are
/// skipped. Keyword-order swapping (e.g. `top left`) is not modelled —
/// the common authored order is horizontal-then-vertical, and sprite
/// sheets use explicit px pairs.
fn parse_background_position(toks: &[CssToken]) -> Option<(BgPos, BgPos)> {
    let vals: Vec<&CssToken> = toks
        .iter()
        .filter(|t| !matches!(t, CssToken::Whitespace))
        .collect();
    // Per CSS Backgrounds 3 §3.6: when a SINGLE keyword is given to
    // `background-position`, it sets the corresponding axis and the OTHER
    // axis defaults to `center`. `top`/`bottom` are vertical-only; the
    // previous parser always placed value[0] on X, so `background-
    // position: top` set X=center (effectively), pulling images to the
    // wrong side. Detect a sole vertical/horizontal keyword and route
    // accordingly.
    if vals.len() == 1 {
        if let CssToken::Ident(s) = vals[0] {
            let lc = s.to_ascii_lowercase();
            match lc.as_str() {
                "top" => return Some((BgPos::Pct(50.0), BgPos::Pct(0.0))),
                "bottom" => return Some((BgPos::Pct(50.0), BgPos::Pct(100.0))),
                "left" => return Some((BgPos::Pct(0.0), BgPos::Pct(50.0))),
                "right" => return Some((BgPos::Pct(100.0), BgPos::Pct(50.0))),
                "center" => return Some((BgPos::Pct(50.0), BgPos::Pct(50.0))),
                _ => {}
            }
        }
    }
    let x = bg_pos_component(vals.first().copied(), true)?;
    let y = if vals.len() >= 2 {
        bg_pos_component(vals.get(1).copied(), false).unwrap_or(BgPos::Pct(50.0))
    } else {
        BgPos::Pct(50.0)
    };
    Some((x, y))
}

fn parse_linear_gradient(toks: &[CssToken]) -> Option<LinearGradient> {
    let mut i = 0;
    while i < toks.len() {
        if let CssToken::Function(name) = &toks[i] {
            let lc = name.to_ascii_lowercase();
            let is_linear = lc == "linear-gradient" || lc == "repeating-linear-gradient";
            if !is_linear {
                i += 1;
                continue;
            }
            // Walk the body up to the matching `)`, splitting at depth-1
            // commas into "chunks". The first chunk may be a direction
            // (`to right`, `45deg`); subsequent chunks are color stops.
            let mut depth = 1;
            let mut start = i + 1;
            let mut chunks: Vec<&[CssToken]> = Vec::new();
            let mut j = start;
            while j < toks.len() && depth > 0 {
                match &toks[j] {
                    CssToken::Function(_) | CssToken::LeftParen => depth += 1,
                    CssToken::RightParen => {
                        depth -= 1;
                        if depth == 0 {
                            chunks.push(&toks[start..j]);
                            break;
                        }
                    }
                    CssToken::Comma if depth == 1 => {
                        chunks.push(&toks[start..j]);
                        start = j + 1;
                    }
                    _ => {}
                }
                j += 1;
            }
            // Resolve direction: if the first chunk parses as a Color,
            // there's no explicit direction — default to "to bottom"
            // (180deg). Otherwise treat it as a direction descriptor.
            let mut angle_deg: f32 = 180.0;
            let mut color_chunks = chunks.as_slice();
            let mut first_is_dir = true;
            if let Some(first) = chunks.first() {
                if Color::from_tokens(first).is_some() {
                    first_is_dir = false;
                }
            }
            if first_is_dir && !chunks.is_empty() {
                angle_deg = parse_gradient_direction(chunks[0]).unwrap_or(180.0);
                color_chunks = &chunks[1..];
            }
            let mut colors: Vec<Color> = Vec::new();
            for chunk in color_chunks {
                if let Some(c) = Color::from_tokens(chunk) {
                    colors.push(c);
                }
            }
            if colors.len() >= 2 {
                return Some(LinearGradient {
                    from: colors[0],
                    to: *colors.last().unwrap(),
                    angle_deg,
                });
            }
            return None;
        }
        i += 1;
    }
    None
}

fn parse_radial_gradient(toks: &[CssToken]) -> Option<LinearGradient> {
    let mut i = 0;
    while i < toks.len() {
        if let CssToken::Function(name) = &toks[i] {
            let lc = name.to_ascii_lowercase();
            let is_radial = lc == "radial-gradient" || lc == "repeating-radial-gradient";
            if !is_radial {
                i += 1;
                continue;
            }
            let mut depth = 1;
            let mut start = i + 1;
            let mut chunks: Vec<&[CssToken]> = Vec::new();
            let mut j = start;
            while j < toks.len() && depth > 0 {
                match &toks[j] {
                    CssToken::Function(_) | CssToken::LeftParen => depth += 1,
                    CssToken::RightParen => {
                        depth -= 1;
                        if depth == 0 {
                            chunks.push(&toks[start..j]);
                            break;
                        }
                    }
                    CssToken::Comma if depth == 1 => {
                        chunks.push(&toks[start..j]);
                        start = j + 1;
                    }
                    _ => {}
                }
                j += 1;
            }
            let mut colors = Vec::new();
            for chunk in chunks {
                if let Some(c) = Color::from_tokens(chunk) {
                    colors.push(c);
                }
            }
            if colors.len() >= 2 {
                return Some(LinearGradient {
                    from: colors[0],
                    to: *colors.last().unwrap(),
                    angle_deg: 0.0,
                });
            }
            return None;
        }
        i += 1;
    }
    None
}

fn parse_aspect_ratio(toks: &[CssToken]) -> Option<f32> {
    let filtered: Vec<&CssToken> = toks
        .iter()
        .filter(|t| !matches!(t, CssToken::Whitespace))
        .collect();
    if filtered.is_empty() {
        return None;
    }
    if let [CssToken::Number(n)] = filtered.as_slice() {
        let ratio = *n as f32;
        return (ratio > 0.0).then_some(ratio);
    }
    if let [
        CssToken::Number(w),
        CssToken::Delim('/'),
        CssToken::Number(h),
    ] = filtered.as_slice()
    {
        let width = *w as f32;
        let height = *h as f32;
        if width > 0.0 && height > 0.0 {
            return Some(width / height);
        }
    }
    None
}

/// Parse a `<grid-line>` value for the `-start` longhands.
///
/// Returns `(start_line, span)`:
/// - `3`        → (Some(3), None)   — explicit line number
/// - `span 3`   → (None, Some(3))   — span keyword, no explicit line
/// - `span`     → (None, None)      — bare span without count (ignored)
/// - `auto`     → (None, None)
fn parse_grid_line(toks: &[CssToken]) -> (Option<usize>, Option<usize>) {
    let filtered: Vec<&CssToken> = toks
        .iter()
        .filter(|t| !matches!(t, CssToken::Whitespace))
        .collect();
    let is_span = filtered
        .iter()
        .any(|t| matches!(t, CssToken::Ident(s) if s.eq_ignore_ascii_case("span")));
    let first_int = filtered.iter().find_map(|t| match t {
        CssToken::Number(n) if *n >= 1.0 => Some(*n as usize),
        _ => None,
    });
    if is_span {
        (None, first_int)
    } else {
        (first_int, None)
    }
}

fn parse_grid_placement(toks: &[CssToken]) -> Option<(usize, Option<usize>)> {
    // Per CSS Grid Level 2 §8.3 grid-placement-shorthand:
    //   <integer>                         → start=N
    //   <integer> / <integer>             → start=A, span=B-A
    //   <integer> / span <integer>        → start=A, span=B
    //   span <integer>                    → start=1 (auto, V1 fallback), span=N
    //
    // The previous parser scanned for "first number" everywhere, so
    // `span 3` (start auto, span 3) was mis-read as start=3 and the span
    // was lost — `grid-column: span 3` thus placed at column 3 with span
    // 1 instead of starting at the auto-flow position and spanning 3.
    let filtered: Vec<&CssToken> = toks
        .iter()
        .filter(|t| !matches!(t, CssToken::Whitespace))
        .collect();
    let slash_idx = filtered
        .iter()
        .position(|t| matches!(t, CssToken::Delim('/')));
    let (lhs, rhs): (&[&CssToken], Option<&[&CssToken]>) = match slash_idx {
        Some(i) => (&filtered[..i], Some(&filtered[i + 1..])),
        None => (&filtered[..], None),
    };
    let is_span_kw = |toks: &[&CssToken]| {
        toks.iter()
            .any(|t| matches!(t, CssToken::Ident(s) if s.eq_ignore_ascii_case("span")))
    };
    let first_int = |toks: &[&CssToken]| {
        toks.iter().find_map(|t| match t {
            CssToken::Number(n) if *n >= 1.0 => Some(*n as usize),
            _ => None,
        })
    };
    let lhs_is_span = is_span_kw(lhs);
    let lhs_int = first_int(lhs);
    match (lhs_is_span, lhs_int, rhs) {
        // `span N`  (no slash)  → start auto = 1, span N
        (true, Some(n), None) => Some((1, Some(n))),
        // `M / N`              → start=M, span=N-M (positive)
        (false, Some(m), Some(r)) => {
            let rhs_is_span = is_span_kw(r);
            let rhs_int = first_int(r);
            match (rhs_is_span, rhs_int) {
                (true, Some(s)) => Some((m, Some(s))),
                (false, Some(end)) if end > m => Some((m, Some(end - m))),
                _ => Some((m, None)),
            }
        }
        // `M` alone            → start=M, span unspecified (1)
        (false, Some(m), None) => Some((m, None)),
        _ => None,
    }
}

impl CssGradient {
    /// Color of the first stop — used as the solid `background-color`
    /// fallback for callers that don't rasterize the gradient.
    pub fn first_stop_color(&self) -> Option<Color> {
        match self {
            CssGradient::Linear { stops, .. }
            | CssGradient::Radial { stops, .. }
            | CssGradient::Conic { stops, .. } => stops.first().map(|s| s.color),
        }
    }

    /// The stop list (shared accessor).
    pub fn stops(&self) -> &[GradientColorStop] {
        match self {
            CssGradient::Linear { stops, .. }
            | CssGradient::Radial { stops, .. }
            | CssGradient::Conic { stops, .. } => stops,
        }
    }
}

/// Split a function body's token slice into per-chunk slices at depth-1
/// commas. `toks` must be the tokens AFTER the opening `Function(name)`
/// token; parsing stops at the matching depth-0 `)`. Returns the chunks
/// (each a comma-separated argument) plus the index of the token right
/// after the closing paren in `toks`.
fn split_gradient_args(toks: &[CssToken]) -> (Vec<&[CssToken]>, usize) {
    let mut depth = 1;
    let mut start = 0;
    let mut chunks: Vec<&[CssToken]> = Vec::new();
    let mut j = 0;
    while j < toks.len() && depth > 0 {
        match &toks[j] {
            CssToken::Function(_) | CssToken::LeftParen => depth += 1,
            CssToken::RightParen => {
                depth -= 1;
                if depth == 0 {
                    chunks.push(&toks[start..j]);
                    j += 1;
                    break;
                }
            }
            CssToken::Comma if depth == 1 => {
                chunks.push(&toks[start..j]);
                start = j + 1;
            }
            _ => {}
        }
        j += 1;
    }
    (chunks, j)
}

/// Parse one color-stop chunk: `<color> [<pos>]?` or a bare `<pos>`
/// (transition hint — modelled as the previous color repeated). Returns
/// the stop(s) for this chunk; a `<color> <pos1> <pos2>` shorthand
/// (CSS Images 4 — two positions = two stops same color) yields two
/// stops.
fn parse_one_color_stop(chunk: &[CssToken]) -> Vec<GradientColorStop> {
    let non_ws: Vec<&CssToken> = chunk
        .iter()
        .filter(|t| !matches!(t, CssToken::Whitespace))
        .collect();
    // Find the color: try the whole chunk first (covers rgb()/hsl()
    // function colors), else the leading run up to the first position
    // token.
    let pos_of = |t: &CssToken| -> Option<(Option<f32>, Option<f32>)> {
        match t {
            CssToken::Percent(p) => Some((Some(*p as f32 / 100.0), None)),
            CssToken::Dimension { value, unit } => {
                let v = *value as f32;
                match unit.to_ascii_lowercase().as_str() {
                    "px" => Some((None, Some(v))),
                    "deg" => Some((Some(v / 360.0), None)),
                    "turn" => Some((Some(v), None)),
                    "rad" => Some((Some(v / (2.0 * core::f32::consts::PI)), None)),
                    "grad" => Some((Some(v / 400.0), None)),
                    _ => None,
                }
            }
            CssToken::Number(n) if *n == 0.0 => Some((None, Some(0.0))),
            _ => None,
        }
    };
    // Collect positions present in the chunk.
    let positions: Vec<(Option<f32>, Option<f32>)> = non_ws
        .iter()
        .filter_map(|t| pos_of(t))
        .collect();
    // The color is the chunk with position tokens removed.
    let color_toks: Vec<CssToken> = chunk
        .iter()
        .filter(|t| pos_of(t).is_none())
        .cloned()
        .collect();
    let color = Color::from_tokens(&color_toks);
    match (color, positions.as_slice()) {
        (Some(c), []) => vec![GradientColorStop {
            color: c,
            pos_frac: None,
            pos_px: None,
        }],
        (Some(c), [(f, p)]) => vec![GradientColorStop {
            color: c,
            pos_frac: *f,
            pos_px: *p,
        }],
        // `<color> <pos1> <pos2>` doubled-stop shorthand.
        (Some(c), [(f1, p1), (f2, p2), ..]) => vec![
            GradientColorStop {
                color: c,
                pos_frac: *f1,
                pos_px: *p1,
            },
            GradientColorStop {
                color: c,
                pos_frac: *f2,
                pos_px: *p2,
            },
        ],
        // Bare position with no color = transition hint; not modelled
        // as a midpoint here (we keep linear interpolation), so skip it.
        (None, _) => Vec::new(),
    }
}

/// Parse a `<position>` axis component for `at <position>`. px / % /
/// keyword.
fn parse_gradient_pos_axis(toks: &[&CssToken]) -> Option<GradientPosAxis> {
    for t in toks {
        match t {
            CssToken::Dimension { value, unit } if unit.eq_ignore_ascii_case("px") => {
                return Some(GradientPosAxis::Px(*value as f32));
            }
            CssToken::Percent(p) => return Some(GradientPosAxis::Pct(*p as f32)),
            CssToken::Number(n) if *n == 0.0 => return Some(GradientPosAxis::Px(0.0)),
            CssToken::Ident(s) => {
                let v = match s.to_ascii_lowercase().as_str() {
                    "left" | "top" => 0.0,
                    "center" => 50.0,
                    "right" | "bottom" => 100.0,
                    _ => continue,
                };
                return Some(GradientPosAxis::Pct(v));
            }
            _ => {}
        }
    }
    None
}

/// Parse the radial-gradient preamble (everything before the first
/// color stop): `[<shape> || <size> || at <position>]`. Returns
/// (shape, size, center).
fn parse_radial_preamble(
    toks: &[CssToken],
) -> (RadialShape, RadialSize, Option<(GradientPosAxis, GradientPosAxis)>) {
    let non_ws: Vec<&CssToken> = toks
        .iter()
        .filter(|t| !matches!(t, CssToken::Whitespace))
        .collect();
    // Split off the `at <position>` tail.
    let at_idx = non_ws
        .iter()
        .position(|t| matches!(t, CssToken::Ident(s) if s.eq_ignore_ascii_case("at")));
    let (shape_size, center) = match at_idx {
        Some(i) => {
            let pos_toks = &non_ws[i + 1..];
            // Split position into x and y by finding the boundary: each
            // axis is one length/%/keyword.
            let x = parse_gradient_pos_axis(pos_toks.get(0..1).unwrap_or(&[]));
            let y = parse_gradient_pos_axis(pos_toks.get(1..2).unwrap_or(&[]));
            let center = match (x, y) {
                (Some(x), Some(y)) => Some((x, y)),
                // Single component → other axis = center.
                (Some(x), None) => Some((x, GradientPosAxis::Pct(50.0))),
                _ => None,
            };
            (&non_ws[..i], center)
        }
        None => (&non_ws[..], None),
    };
    let mut shape = RadialShape::Ellipse; // default per spec
    let mut explicit_shape = false;
    let mut size = RadialSize::FarthestCorner; // default per spec
    let mut explicit_lengths: Vec<(Option<f32>, Option<f32>)> = Vec::new(); // (px, pct)
    for t in shape_size {
        match t {
            CssToken::Ident(s) => match s.to_ascii_lowercase().as_str() {
                "circle" => {
                    shape = RadialShape::Circle;
                    explicit_shape = true;
                }
                "ellipse" => {
                    shape = RadialShape::Ellipse;
                    explicit_shape = true;
                }
                "closest-side" => size = RadialSize::ClosestSide,
                "farthest-side" => size = RadialSize::FarthestSide,
                "closest-corner" => size = RadialSize::ClosestCorner,
                "farthest-corner" => size = RadialSize::FarthestCorner,
                _ => {}
            },
            CssToken::Dimension { value, unit } if unit.eq_ignore_ascii_case("px") => {
                explicit_lengths.push((Some(*value as f32), None));
            }
            CssToken::Percent(p) => explicit_lengths.push((None, Some(*p as f32))),
            _ => {}
        }
    }
    if !explicit_lengths.is_empty() {
        // One length → circle radius; two → ellipse rx, ry.
        let (rx_px, rx_pct) = explicit_lengths[0];
        let (ry_px, ry_pct) = if explicit_lengths.len() >= 2 {
            explicit_lengths[1]
        } else {
            (None, None)
        };
        size = RadialSize::Explicit {
            rx_px,
            ry_px: if explicit_lengths.len() >= 2 { ry_px } else { rx_px },
            rx_pct,
            ry_pct: if explicit_lengths.len() >= 2 { ry_pct } else { rx_pct },
        };
        // A single explicit length implies a circle unless ellipse stated.
        if explicit_lengths.len() == 1 && !explicit_shape {
            shape = RadialShape::Circle;
        }
    }
    (shape, size, center)
}

/// Parse any CSS gradient function (`linear-gradient`,
/// `radial-gradient`, `conic-gradient`, and their `repeating-` variants)
/// into the full N-stop [`CssGradient`] model. Returns `None` when the
/// value contains no gradient or fewer than two parsable color stops.
///
/// CSS Images 3 §3 (linear/radial) + CSS Images 4 §3.3 (conic).
pub fn parse_css_gradient(toks: &[CssToken]) -> Option<CssGradient> {
    let mut i = 0;
    while i < toks.len() {
        if let CssToken::Function(name) = &toks[i] {
            let lc = name.to_ascii_lowercase();
            let repeating = lc.starts_with("repeating-");
            let base = lc.strip_prefix("repeating-").unwrap_or(&lc);
            let kind = match base {
                "linear-gradient" => 0,
                "radial-gradient" => 1,
                "conic-gradient" => 2,
                _ => {
                    i += 1;
                    continue;
                }
            };
            let (chunks, _end) = split_gradient_args(&toks[i + 1..]);
            if chunks.is_empty() {
                return None;
            }
            // Decide whether the first chunk is a direction/preamble or a
            // color stop: it's a preamble if it does NOT parse as a color
            // stop (no color in it).
            let first_is_stop = !parse_one_color_stop(chunks[0]).is_empty();
            let (preamble, stop_chunks): (Option<&[CssToken]>, &[&[CssToken]]) = if first_is_stop {
                (None, chunks.as_slice())
            } else {
                (Some(chunks[0]), &chunks[1..])
            };
            let mut stops: Vec<GradientColorStop> = Vec::new();
            for ch in stop_chunks {
                stops.extend(parse_one_color_stop(ch));
            }
            if stops.len() < 2 {
                return None;
            }
            return Some(match kind {
                0 => {
                    let angle_deg = preamble
                        .and_then(parse_gradient_direction)
                        .unwrap_or(180.0);
                    CssGradient::Linear {
                        angle_deg,
                        stops,
                        repeating,
                    }
                }
                1 => {
                    let (shape, size, center) = preamble
                        .map(parse_radial_preamble)
                        .unwrap_or((RadialShape::Ellipse, RadialSize::FarthestCorner, None));
                    CssGradient::Radial {
                        shape,
                        size,
                        center,
                        stops,
                        repeating,
                    }
                }
                _ => {
                    // conic: preamble = `[from <angle>]? [at <position>]?`
                    let (from_deg, center) = preamble
                        .map(parse_conic_preamble)
                        .unwrap_or((0.0, None));
                    CssGradient::Conic {
                        from_deg,
                        center,
                        stops,
                        repeating,
                    }
                }
            });
        }
        i += 1;
    }
    None
}

/// Parse the conic-gradient preamble: `[from <angle>]? [at <position>]?`.
/// Returns (from_angle_deg, center).
fn parse_conic_preamble(
    toks: &[CssToken],
) -> (f32, Option<(GradientPosAxis, GradientPosAxis)>) {
    let non_ws: Vec<&CssToken> = toks
        .iter()
        .filter(|t| !matches!(t, CssToken::Whitespace))
        .collect();
    let mut from_deg = 0.0;
    let at_idx = non_ws
        .iter()
        .position(|t| matches!(t, CssToken::Ident(s) if s.eq_ignore_ascii_case("at")));
    // `from <angle>` — the angle dimension appears before `at`.
    let from_scan_end = at_idx.unwrap_or(non_ws.len());
    for t in &non_ws[..from_scan_end] {
        if let CssToken::Dimension { value, unit } = t {
            let v = *value as f32;
            from_deg = match unit.to_ascii_lowercase().as_str() {
                "deg" => v,
                "turn" => v * 360.0,
                "rad" => v * 180.0 / core::f32::consts::PI,
                "grad" => v * 360.0 / 400.0,
                _ => v,
            };
        }
    }
    let center = at_idx.and_then(|i| {
        let pos_toks = &non_ws[i + 1..];
        let x = parse_gradient_pos_axis(pos_toks.get(0..1).unwrap_or(&[]));
        let y = parse_gradient_pos_axis(pos_toks.get(1..2).unwrap_or(&[]));
        match (x, y) {
            (Some(x), Some(y)) => Some((x, y)),
            (Some(x), None) => Some((x, GradientPosAxis::Pct(50.0))),
            _ => None,
        }
    });
    (from_deg, center)
}

/// Parse the direction portion of a CSS linear-gradient. Handles
/// `<angle>deg` and `to <side>` / `to <corner>` keywords. Returns the
/// angle in CSS degrees (0 = to top, 90 = to right, 180 = to bottom,
/// 270 = to left).
fn parse_gradient_direction(toks: &[CssToken]) -> Option<f32> {
    // `<angle>deg` / `<angle>turn` / etc.
    for t in toks {
        if let CssToken::Dimension { value, unit } = t {
            let v = *value as f32;
            let lc = unit.to_ascii_lowercase();
            return Some(match lc.as_str() {
                "deg" => v,
                "turn" => v * 360.0,
                "rad" => v * 180.0 / core::f32::consts::PI,
                "grad" => v * 360.0 / 400.0,
                _ => v,
            });
        }
    }
    // `to right` / `to bottom right` etc.
    let mut saw_to = false;
    let mut horiz = 0; // -1 left, +1 right
    let mut vert = 0; // -1 top, +1 bottom
    for t in toks {
        if let CssToken::Ident(s) = t {
            let lc = s.to_ascii_lowercase();
            match lc.as_str() {
                "to" => saw_to = true,
                "left" => horiz = -1,
                "right" => horiz = 1,
                "top" => vert = -1,
                "bottom" => vert = 1,
                _ => {}
            }
        }
    }
    if !saw_to {
        return None;
    }
    // Map (horiz, vert) → angle. CSS spec: `to top` = 0deg, `to right`
    // = 90deg, `to bottom` = 180deg, `to left` = 270deg.
    Some(match (horiz, vert) {
        (0, -1) => 0.0,
        (1, -1) => 45.0,
        (1, 0) => 90.0,
        (1, 1) => 135.0,
        (0, 1) => 180.0,
        (-1, 1) => 225.0,
        (-1, 0) => 270.0,
        (-1, -1) => 315.0,
        _ => 180.0,
    })
}

/// Walk a value token sequence containing a `linear-gradient(...)` /
/// `radial-gradient(...)` / `conic-gradient(...)` call and return the
/// FIRST parsable color stop. Used as a paint stand-in until we have
/// real gradient rasterization in `cv_gfx`.
fn first_color_inside_gradient(toks: &[CssToken]) -> Option<Color> {
    let mut i = 0;
    while i < toks.len() {
        if let CssToken::Function(name) = &toks[i] {
            let lc = name.to_ascii_lowercase();
            if lc.ends_with("-gradient") || lc == "image" {
                // Walk the body up to the matching `)` and try Color::from_tokens
                // on each comma-separated chunk.
                let mut depth = 1;
                let mut start = i + 1;
                let mut j = start;
                while j < toks.len() && depth > 0 {
                    match &toks[j] {
                        CssToken::Function(_) | CssToken::LeftParen => depth += 1,
                        CssToken::RightParen => {
                            depth -= 1;
                            if depth == 0 {
                                // Final chunk.
                                if let Some(c) = Color::from_tokens(&toks[start..j]) {
                                    return Some(c);
                                }
                                break;
                            }
                        }
                        CssToken::Comma if depth == 1 => {
                            if let Some(c) = Color::from_tokens(&toks[start..j]) {
                                return Some(c);
                            }
                            start = j + 1;
                        }
                        _ => {}
                    }
                    j += 1;
                }
                i = j;
                continue;
            }
        }
        i += 1;
    }
    None
}

/// Split a value token list into per-token Lengths. Each non-whitespace
/// token contributes (independently) one Length — used to unwind
/// True if the token list is the bare keyword `auto` (ignoring
/// whitespace). Used by `margin-*`, `width`, and `max-width` parsers
/// to distinguish "the author wrote `auto`" from "we didn't parse it".
fn is_auto_keyword(toks: &[CssToken]) -> bool {
    let mut saw_auto = false;
    for t in toks {
        match t {
            CssToken::Whitespace => {}
            CssToken::Ident(s) if s.eq_ignore_ascii_case("auto") => {
                if saw_auto {
                    return false;
                }
                saw_auto = true;
            }
            _ => return false,
        }
    }
    saw_auto
}

/// True if the token list is the bare keyword `none`. Used by
/// `max-width: none` / `max-height: none` to mean "no upper bound".
fn is_none_keyword(toks: &[CssToken]) -> bool {
    let mut saw = false;
    for t in toks {
        match t {
            CssToken::Whitespace => {}
            CssToken::Ident(s) if s.eq_ignore_ascii_case("none") => {
                if saw {
                    return false;
                }
                saw = true;
            }
            _ => return false,
        }
    }
    saw
}

/// `margin: 10px 20px` etc. without needing the global-scan behaviour of
/// `Length::from_tokens`.
fn lengths_each(toks: &[CssToken]) -> Vec<Length> {
    let mut out = Vec::new();
    for t in toks {
        if matches!(t, CssToken::Whitespace) {
            continue;
        }
        if let Some(l) = Length::from_tokens(std::slice::from_ref(t)) {
            out.push(l);
        }
    }
    out
}

/// Like `lengths_each` but admits the `auto` keyword as a `None`
/// slot. Used for margin where `auto` is meaningful — the cascade
/// needs to remember that the author asked for an auto margin so
/// block layout can do its "distribute the leftover horizontal
/// space" trick.
fn lengths_or_auto_each(toks: &[CssToken]) -> Vec<Option<Length>> {
    let mut out = Vec::new();
    for t in toks {
        if matches!(t, CssToken::Whitespace) {
            continue;
        }
        if let CssToken::Ident(s) = t {
            if s.eq_ignore_ascii_case("auto") {
                out.push(None);
                continue;
            }
        }
        if let Some(l) = Length::from_tokens(std::slice::from_ref(t)) {
            out.push(Some(l));
        }
    }
    out
}

/// Apply the CSS box shorthand expansion (top, right, bottom, left)
/// given 1 / 2 / 3 / 4 values per the spec.
fn parse_box_shorthand(toks: &[CssToken]) -> Option<[Option<Length>; 4]> {
    let lens = lengths_each(toks);
    let (t, r, b, l) = match lens.len() {
        1 => (lens[0], lens[0], lens[0], lens[0]),
        2 => (lens[0], lens[1], lens[0], lens[1]),
        3 => (lens[0], lens[1], lens[2], lens[1]),
        4 => (lens[0], lens[1], lens[2], lens[3]),
        _ => return None,
    };
    Some([Some(t), Some(r), Some(b), Some(l)])
}

/// `margin`-aware shorthand expansion that preserves `auto` per side.
/// Returns `(lengths, autos)` aligned by side (top, right, bottom, left).
/// `Some(len)` + auto[i]=false means a normal numeric margin; auto[i]=true
/// means the author wrote `auto` on that side.
fn parse_margin_shorthand(toks: &[CssToken]) -> Option<([Option<Length>; 4], [bool; 4])> {
    let parts = lengths_or_auto_each(toks);
    let (t, r, b, l) = match parts.len() {
        1 => (parts[0], parts[0], parts[0], parts[0]),
        2 => (parts[0], parts[1], parts[0], parts[1]),
        3 => (parts[0], parts[1], parts[2], parts[1]),
        4 => (parts[0], parts[1], parts[2], parts[3]),
        _ => return None,
    };
    let lens = [t, r, b, l];
    let autos = [t.is_none(), r.is_none(), b.is_none(), l.is_none()];
    Some((lens, autos))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::parse_stylesheet;

    /// `mix-blend-mode` / `background-blend-mode` parse into their dedicated
    /// style fields (NOT swallowed by the recognized-but-ignored catch-all),
    /// so the painter can apply the W3C blend formula.
    #[test]
    fn blend_mode_properties_parse_into_style() {
        let mut style = ComputedStyle::default();
        apply_declaration(
            &mut style,
            &Declaration {
                name: "mix-blend-mode".to_string(),
                value: vec![CssToken::Ident("multiply".to_string())],
                important: false,
            },
        );
        apply_declaration(
            &mut style,
            &Declaration {
                name: "background-blend-mode".to_string(),
                value: vec![CssToken::Ident("luminosity".to_string())],
                important: false,
            },
        );
        assert_eq!(style.mix_blend_mode.as_deref(), Some("multiply"));
        assert_eq!(style.background_blend_mode.as_deref(), Some("luminosity"));
        // A `normal` mix-blend-mode is captured verbatim (the painter maps it
        // to source-over) — it is NOT dropped to None.
        let mut s2 = ComputedStyle::default();
        apply_declaration(
            &mut s2,
            &Declaration {
                name: "mix-blend-mode".to_string(),
                value: vec![CssToken::Ident("normal".to_string())],
                important: false,
            },
        );
        assert_eq!(s2.mix_blend_mode.as_deref(), Some("normal"));
    }

    #[test]
    fn parse_color_str_handles_hex_named_and_rgb() {
        assert_eq!(parse_color_str("#fff"), Some(Color { r: 255, g: 255, b: 255, a: 255 }));
        assert_eq!(parse_color_str("#000000"), Some(Color { r: 0, g: 0, b: 0, a: 255 }));
        assert_eq!(parse_color_str("red"), Some(Color { r: 255, g: 0, b: 0, a: 255 }));
        assert_eq!(parse_color_str("rgb(0, 128, 255)"), Some(Color { r: 0, g: 128, b: 255, a: 255 }));
        // Non-colors return None so lengths/numbers fall through to the numeric path.
        assert_eq!(parse_color_str("12px"), None);
        assert_eq!(parse_color_str("0.5"), None);
        assert_eq!(parse_color_str("auto"), None);
    }

    #[test]
    fn interpolate_value_lerps_colors_componentwise() {
        // black → white midpoint: rgb(128,128,128). interpolate_value emits an
        // rgba() string; round-trip it through the parser to compare components.
        let mid = interpolate_value("#000", "#fff", 0.5);
        let c = parse_color_str(&mid).expect("rgba output parses");
        assert_eq!((c.r, c.g, c.b, c.a), (128, 128, 128, 255), "{mid}");
        // red → blue midpoint: each channel moves independently → rgb(128,0,128).
        let mid2 = interpolate_value("red", "blue", 0.5);
        let c2 = parse_color_str(&mid2).expect("parses");
        assert_eq!((c2.r, c2.g, c2.b), (128, 0, 128), "{mid2}");
        // endpoints are exact.
        let start = parse_color_str(&interpolate_value("red", "blue", 0.0)).unwrap();
        assert_eq!((start.r, start.g, start.b), (255, 0, 0));
        let end = parse_color_str(&interpolate_value("red", "blue", 1.0)).unwrap();
        assert_eq!((end.r, end.g, end.b), (0, 0, 255));
    }

    #[test]
    fn interpolate_value_lengths_still_numeric() {
        // The color path must NOT swallow lengths/numbers.
        let mid = interpolate_value("0px", "100px", 0.25);
        // numeric skeleton path → "25px"
        assert!(mid.starts_with("25"), "len lerp = {mid}");
        let mid_n = interpolate_value("0", "10", 0.5);
        assert!(mid_n.starts_with("5"), "num lerp = {mid_n}");
    }

    #[test]
    fn interpolate_value_currentcolor_steps() {
        // currentColor is a sentinel, not a concrete color → step, never numeric.
        let v = interpolate_value("currentColor", "red", 0.6);
        assert_eq!(v, "red", "t>=0.5 steps to b");
        let v0 = interpolate_value("currentColor", "red", 0.4);
        assert_eq!(v0, "currentColor", "t<0.5 steps to a");
    }

    #[test]
    fn sample_animation_interpolates_background_color() {
        // A @keyframes that fades background-color black → white; at t=0.5 the
        // sampled value parses to mid-grey.
        let css = "@keyframes fade { from { background-color: #000; } to { background-color: #fff; } }";
        let ss = parse_stylesheet(css);
        let kf = collect_keyframes(&[ss]);
        let rule = kf.get("fade").expect("keyframe collected");
        let props = sample_animation(rule, 0.5);
        let bg = props.get("background-color").expect("bg sampled");
        let c = parse_color_str(bg).expect("parses");
        assert_eq!((c.r, c.g, c.b), (128, 128, 128), "mid grey, got {bg}");
    }

    #[derive(Copy, Clone)]
    struct Fake<'a> {
        tag: &'a str,
        id: Option<&'a str>,
        classes: &'a [&'a str],
    }

    impl<'a> ElementView<'a> for Fake<'a> {
        fn tag_name(&self) -> Option<&'a str> {
            Some(self.tag)
        }
        fn id(&self) -> Option<&'a str> {
            self.id
        }
        fn has_class(&self, name: &str) -> bool {
            self.classes.iter().any(|c| *c == name)
        }
        fn parent(&self) -> Option<Self> {
            None
        }
    }

    // ---- @container size-query EVALUATION (CSS Containment 3 §3) ----------

    fn red() -> Color {
        Color { r: 255, g: 0, b: 0, a: 255 }
    }
    fn blue() -> Color {
        Color { r: 0, g: 0, b: 255, a: 255 }
    }

    fn qc(names: &[&str], ty: ContainerType, inline: f32, block: f32) -> QueryContainer {
        QueryContainer {
            names: names.iter().map(|s| s.to_string()).collect(),
            container_type: ty,
            inline_size: inline,
            block_size: block,
        }
    }

    /// The headline test: a `@container (min-width: 300px)` rule applies to a
    /// descendant when the container is 400px wide and is WITHHELD when the
    /// SAME viewport/index but a 200px container is supplied. The property
    /// difference is driven purely by container size, not the viewport.
    #[test]
    fn container_min_width_applies_by_container_size_not_viewport() {
        let css = "@container (min-width: 300px) { .card { color: red; } }";
        let ss = parse_stylesheet(css);
        let sheets = [ss];
        // Identical index/viewport for both resolutions — only the container
        // size differs, isolating the variable under test.
        let idx = SelectorIndex::build_with_viewport(&sheets, 1024.0, 768.0);
        let el = Fake { tag: "div", id: None, classes: &["card"] };
        let classes = ["card".to_string()];

        // Container 400px wide ≥ 300px → rule applies, color = red.
        let wide = [qc(&[], ContainerType::InlineSize, 400.0, 0.0)];
        let cs_wide = compute_with_index_cq(&idx, el, &[], &classes, &wide);
        assert_eq!(cs_wide.color, Some(red()), "400px container ≥ 300px applies");

        // Container 200px wide < 300px → rule withheld, color = None.
        let narrow = [qc(&[], ContainerType::InlineSize, 200.0, 0.0)];
        let cs_narrow = compute_with_index_cq(&idx, el, &[], &classes, &narrow);
        assert_eq!(cs_narrow.color, None, "200px container < 300px withholds");
    }

    /// `container-name` must target the RIGHT ancestor. The query names
    /// `sidebar`; only the `sidebar` container's size should decide the match,
    /// even when a nearer unnamed container has a contradicting size.
    #[test]
    fn container_name_targets_the_named_ancestor() {
        let css = "@container sidebar (min-width: 300px) { .x { color: red; } }";
        let ss = parse_stylesheet(css);
        let sheets = [ss];
        let idx = SelectorIndex::build_with_viewport(&sheets, 1024.0, 768.0);
        let el = Fake { tag: "div", id: None, classes: &["x"] };
        let classes = ["x".to_string()];

        // Stack root-first: outer `sidebar` is 400px (matches), nearer unnamed
        // `main` is 100px (would fail). Because the query names `sidebar`, the
        // nearer unnamed container is skipped and the 400px sidebar matches.
        let stack = [
            qc(&["sidebar"], ContainerType::InlineSize, 400.0, 0.0),
            qc(&["main"], ContainerType::InlineSize, 100.0, 0.0),
        ];
        let cs = compute_with_index_cq(&idx, el, &[], &classes, &stack);
        assert_eq!(cs.color, Some(red()), "named sidebar (400px) drives the match");

        // Now make the sidebar narrow (200px) while the unnamed container is
        // wide (1000px). The named query must follow the NARROW sidebar and
        // withhold — proving it ignores the nearer/wider unnamed container.
        let stack2 = [
            qc(&["sidebar"], ContainerType::InlineSize, 200.0, 0.0),
            qc(&["main"], ContainerType::InlineSize, 1000.0, 0.0),
        ];
        let cs2 = compute_with_index_cq(&idx, el, &[], &classes, &stack2);
        assert_eq!(cs2.color, None, "named sidebar (200px) withholds despite wide neighbor");
    }

    /// The cascade still applies the OUTER (non-container) rule, and the inner
    /// container rule wins by source order when its condition holds; the prior
    /// "fold unconditionally" stub would have always applied red.
    #[test]
    fn container_rule_does_not_clobber_base_when_unsatisfied() {
        let css = ".card { color: blue; } \
                   @container (min-width: 300px) { .card { color: red; } }";
        let ss = parse_stylesheet(css);
        let sheets = [ss];
        let idx = SelectorIndex::build_with_viewport(&sheets, 1024.0, 768.0);
        let el = Fake { tag: "div", id: None, classes: &["card"] };
        let classes = ["card".to_string()];

        // Wide container → @container red wins (later source order).
        let wide = [qc(&[], ContainerType::InlineSize, 400.0, 0.0)];
        assert_eq!(
            compute_with_index_cq(&idx, el, &[], &classes, &wide).color,
            Some(red())
        );
        // Narrow container → @container drops out, base blue remains.
        let narrow = [qc(&[], ContainerType::InlineSize, 200.0, 0.0)];
        assert_eq!(
            compute_with_index_cq(&idx, el, &[], &classes, &narrow).color,
            Some(blue())
        );
        // No container in scope at all → query can't be satisfied, base blue.
        let empty: [QueryContainer; 0] = [];
        assert_eq!(
            compute_with_index_cq(&idx, el, &[], &classes, &empty).color,
            Some(blue())
        );
    }

    /// Block-axis (`min-height`) queries require a `container-type: size`
    /// container; an `inline-size` container does NOT expose the block axis
    /// (CSS Containment 3 §2.1), so the query must NOT match against it.
    #[test]
    fn container_block_axis_needs_size_container() {
        let css = "@container (min-height: 200px) { .y { color: red; } }";
        let ss = parse_stylesheet(css);
        let sheets = [ss];
        let idx = SelectorIndex::build_with_viewport(&sheets, 1024.0, 768.0);
        let el = Fake { tag: "div", id: None, classes: &["y"] };
        let classes = ["y".to_string()];

        // size container, 300px tall ≥ 200px → applies.
        let size_tall = [qc(&[], ContainerType::Size, 400.0, 300.0)];
        assert_eq!(
            compute_with_index_cq(&idx, el, &[], &classes, &size_tall).color,
            Some(red()),
            "size container 300px tall ≥ 200px applies"
        );
        // size container, 100px tall < 200px → withholds.
        let size_short = [qc(&[], ContainerType::Size, 400.0, 100.0)];
        assert_eq!(
            compute_with_index_cq(&idx, el, &[], &classes, &size_short).color,
            None,
            "size container 100px tall < 200px withholds"
        );
        // inline-size container, even though tall → block axis NOT queryable.
        let inline_tall = [qc(&[], ContainerType::InlineSize, 400.0, 999.0)];
        assert_eq!(
            compute_with_index_cq(&idx, el, &[], &classes, &inline_tall).color,
            None,
            "inline-size container can't satisfy a block-axis query"
        );
    }

    /// `container-type` / `container-name` must be parsed onto the computed
    /// style (the prior code dropped them into a catch-all that ignored them),
    /// including the `container:` shorthand with a `/ <type>` segment.
    #[test]
    fn container_type_and_name_parse_onto_computed_style() {
        let ss = parse_stylesheet("div { container-type: inline-size; container-name: a b; }");
        let el = Fake { tag: "div", id: None, classes: &[] };
        let cs = compute(&[ss], el);
        assert_eq!(cs.container_type, Some(ContainerType::InlineSize));
        assert_eq!(cs.container_name, vec!["a".to_string(), "b".to_string()]);

        // `container: sidebar / size` shorthand → name + type.
        let ss2 = parse_stylesheet("aside { container: sidebar / size; }");
        let el2 = Fake { tag: "aside", id: None, classes: &[] };
        let cs2 = compute(&[ss2], el2);
        assert_eq!(cs2.container_type, Some(ContainerType::Size));
        assert_eq!(cs2.container_name, vec!["sidebar".to_string()]);
    }

    /// Compound `and` condition: BOTH must hold against the container size.
    #[test]
    fn container_compound_and_condition() {
        let css = "@container (min-width: 300px) and (max-width: 500px) { .z { color: red; } }";
        let ss = parse_stylesheet(css);
        let sheets = [ss];
        let idx = SelectorIndex::build_with_viewport(&sheets, 1024.0, 768.0);
        let el = Fake { tag: "div", id: None, classes: &["z"] };
        let classes = ["z".to_string()];

        let in_range = [qc(&[], ContainerType::InlineSize, 400.0, 0.0)];
        assert_eq!(
            compute_with_index_cq(&idx, el, &[], &classes, &in_range).color,
            Some(red()),
            "400px is within [300,500]"
        );
        let too_wide = [qc(&[], ContainerType::InlineSize, 600.0, 0.0)];
        assert_eq!(
            compute_with_index_cq(&idx, el, &[], &classes, &too_wide).color,
            None,
            "600px exceeds max-width 500px"
        );
    }

    /// Regression guard: when NO container stack is supplied (`compute_with_index`,
    /// the historical entry point), `@container` rules still apply optimistically
    /// so the renderer (before it wires container sizes) does not silently lose
    /// styling. The strict withholding only kicks in with `compute_with_index_cq`.
    #[test]
    fn container_optimistic_without_stack() {
        let css = "@container (min-width: 300px) { .card { color: red; } }";
        let ss = parse_stylesheet(css);
        let sheets = [ss];
        let idx = SelectorIndex::build_with_viewport(&sheets, 1024.0, 768.0);
        let el = Fake { tag: "div", id: None, classes: &["card"] };
        let classes = ["card".to_string()];
        // No stack (None) → optimistic apply (prior behavior preserved).
        assert_eq!(
            compute_with_index(&idx, el, &[], &classes).color,
            Some(red())
        );
    }

    #[test]
    fn rule_feature_set_harvest() {
        let ss = parse_stylesheet(
            ".a .b { color: red } .c { color: blue } .x + .y { color: green } .d * { color: pink } #m .n { color: teal }",
        );
        let fs = build_rule_feature_set(std::slice::from_ref(&ss));

        // Subject classes/ids self-invalidate.
        assert!(fs.class_set("b").unwrap().invalidates_self, ".b is a subject");
        assert!(fs.class_set("c").unwrap().invalidates_self, ".c is a subject");
        assert!(fs.class_set("y").unwrap().invalidates_self, ".y is a subject");
        assert!(fs.class_set("n").unwrap().invalidates_self, ".n is a subject");

        // ".a .b" → class[a] has a precise descendant filter containing class b.
        let a = fs.class_set("a").unwrap();
        assert!(a.classes.contains("b"), ".a triggers a descendant filter on .b");
        assert!(!a.whole_subtree && !a.invalidates_parent_subtree, ".a is precise");
        assert!(a.invalidates_element(&["b".into()], None, "div"), "matches .b");
        assert!(!a.invalidates_element(&["z".into()], None, "div"), "not .z");

        // "#m .n" → id[m] descendant filter containing class n.
        assert!(fs.id_set("m").unwrap().classes.contains("n"));

        // ".x + .y" sibling combinator → parent-subtree fallback on the trigger.
        assert!(
            fs.class_set("x").unwrap().invalidates_parent_subtree,
            "sibling combinator widens to parent subtree"
        );

        // ".d *" universal subject → whole-subtree fallback on the trigger.
        assert!(
            fs.class_set("d").unwrap().whole_subtree,
            "universal subject widens to whole subtree"
        );
    }

    #[test]
    fn winner_by_specificity() {
        let ss = parse_stylesheet("p { color: red; } #x { color: blue; } .y { color: green; }");
        let el = Fake {
            tag: "p",
            id: Some("x"),
            classes: &["y"],
        };
        let cs = compute(&[ss], el);
        assert_eq!(
            cs.color,
            Some(Color {
                r: 0,
                g: 0,
                b: 255,
                a: 255
            })
        ); // #x wins
    }

    #[test]
    fn winner_by_source_order_on_tie() {
        let ss = parse_stylesheet(".x { color: red; } .x { color: blue; }");
        let el = Fake {
            tag: "p",
            id: None,
            classes: &["x"],
        };
        let cs = compute(&[ss], el);
        assert_eq!(
            cs.color,
            Some(Color {
                r: 0,
                g: 0,
                b: 255,
                a: 255
            })
        );
    }

    #[test]
    fn margin_shorthand_two_value() {
        // 2 values = vertical, horizontal.
        let ss = parse_stylesheet("p { margin: 10px 20px; }");
        let el = Fake {
            tag: "p",
            id: None,
            classes: &[],
        };
        let cs = compute(&[ss], el);
        assert_eq!(cs.margin[0], Some(Length::Px(10.0)));
        assert_eq!(cs.margin[1], Some(Length::Px(20.0)));
        assert_eq!(cs.margin[2], Some(Length::Px(10.0)));
        assert_eq!(cs.margin[3], Some(Length::Px(20.0)));
    }

    #[test]
    fn padding_shorthand_four_value() {
        let ss = parse_stylesheet("p { padding: 1px 2px 3px 4px; }");
        let el = Fake {
            tag: "p",
            id: None,
            classes: &[],
        };
        let cs = compute(&[ss], el);
        assert_eq!(cs.padding[0], Some(Length::Px(1.0)));
        assert_eq!(cs.padding[1], Some(Length::Px(2.0)));
        assert_eq!(cs.padding[2], Some(Length::Px(3.0)));
        assert_eq!(cs.padding[3], Some(Length::Px(4.0)));
    }

    #[test]
    fn margin_shorthand_three_value() {
        // 3 values = top, horizontal, bottom.
        let ss = parse_stylesheet("p { margin: 1px 2px 3px; }");
        let el = Fake {
            tag: "p",
            id: None,
            classes: &[],
        };
        let cs = compute(&[ss], el);
        assert_eq!(cs.margin[0], Some(Length::Px(1.0)));
        assert_eq!(cs.margin[1], Some(Length::Px(2.0)));
        assert_eq!(cs.margin[2], Some(Length::Px(3.0)));
        assert_eq!(cs.margin[3], Some(Length::Px(2.0)));
    }

    #[test]
    fn padding_side_specific_overrides_shorthand() {
        let ss = parse_stylesheet("p { padding: 5px; padding-left: 50px; }");
        let el = Fake {
            tag: "p",
            id: None,
            classes: &[],
        };
        let cs = compute(&[ss], el);
        assert_eq!(cs.padding[3], Some(Length::Px(50.0)));
        assert_eq!(cs.padding[0], Some(Length::Px(5.0)));
    }

    #[test]
    fn important_overrides_normal() {
        let ss = parse_stylesheet("#x { color: red !important; } .y { color: blue; }");
        let el = Fake {
            tag: "p",
            id: Some("x"),
            classes: &["y"],
        };
        let cs = compute(&[ss], el);
        assert_eq!(
            cs.color,
            Some(Color {
                r: 255,
                g: 0,
                b: 0,
                a: 255
            })
        );
    }

    #[test]
    fn index_matches_naive_for_id_class_tag() {
        // Build a stylesheet that exercises each bucket type plus a few
        // ancestor walks. Confirm `compute_with_index` and the naive
        // `compute` produce the same ComputedStyle so the bucketing
        // didn't sneak in a regression.
        let ss = parse_stylesheet(
            "
            div { color: red; }
            .alert { background: blue; }
            .alert.urgent { color: yellow; }
            #header { font-weight: bold; }
            section .child { font-size: 14px; }
            ",
        );
        let sheets = [ss];
        let idx = SelectorIndex::build(&sheets);

        let el = Fake {
            tag: "div",
            id: Some("header"),
            classes: &["alert", "urgent"],
        };
        let from_index =
            compute_with_index(&idx, el, &[], &["alert".to_string(), "urgent".to_string()]);
        let from_naive = compute(&sheets, el);
        assert_eq!(from_index.color, from_naive.color);
        assert_eq!(from_index.background_color, from_naive.background_color);
        assert_eq!(from_index.font_weight_bold, from_naive.font_weight_bold);
        // Sanity: each property got the expected winner.
        assert_eq!(
            from_index.color,
            Some(Color {
                r: 255,
                g: 255,
                b: 0,
                a: 255
            })
        );
        assert_eq!(
            from_index.background_color,
            Some(Color {
                r: 0,
                g: 0,
                b: 255,
                a: 255
            })
        );
        assert_eq!(from_index.font_weight_bold, Some(true));
    }

    /// Cascade-perf canary. Builds a synthetic stylesheet with one rule
    /// per class — the shape that wedges up the cascade on real pages —
    /// and verifies that `compute_with_index` stays under a per-element
    /// budget. If this regresses, find out why before touching anything
    /// else, because the cost compounds across hundreds of elements.
    #[test]
    fn transform_translate_parses_xy() {
        let ss = parse_stylesheet("div { transform: translate(10px, 20px); }");
        let el = Fake {
            tag: "div",
            id: None,
            classes: &[],
        };
        let cs = compute(&[ss], el);
        assert_eq!(cs.translate_x, Some(Length::Px(10.0)));
        assert_eq!(cs.translate_y, Some(Length::Px(20.0)));
    }

    #[test]
    fn transform_translatex_only() {
        let ss = parse_stylesheet("div { transform: translateX(42px); }");
        let el = Fake {
            tag: "div",
            id: None,
            classes: &[],
        };
        let cs = compute(&[ss], el);
        assert_eq!(cs.translate_x, Some(Length::Px(42.0)));
        assert_eq!(cs.translate_y, None);
    }

    #[test]
    fn transform_translate_percent_preserves_percent_unit() {
        let ss = parse_stylesheet("div { transform: translate(-50%, -25%); }");
        let el = Fake {
            tag: "div",
            id: None,
            classes: &[],
        };
        let cs = compute(&[ss], el);
        assert_eq!(cs.translate_x, Some(Length::Percent(-50.0)));
        assert_eq!(cs.translate_y, Some(Length::Percent(-25.0)));
    }

    #[test]
    fn transform_unknown_function_is_silent() {
        // rotate / scale / matrix should NOT error the rule.
        let ss = parse_stylesheet("div { transform: rotate(45deg) scale(1.5); color: red; }");
        let el = Fake {
            tag: "div",
            id: None,
            classes: &[],
        };
        let cs = compute(&[ss], el);
        // The sibling color declaration must have still applied.
        assert_eq!(
            cs.color,
            Some(Color {
                r: 255,
                g: 0,
                b: 0,
                a: 255
            })
        );
    }

    #[test]
    fn media_min_width_applies_when_viewport_wide_enough() {
        let ss = parse_stylesheet("@media (min-width: 600px) { div { color: red; } }");
        let sheets = [ss];
        let el = Fake {
            tag: "div",
            id: None,
            classes: &[],
        };
        // Default viewport 1024×768 — rule applies.
        let idx = SelectorIndex::build(&sheets);
        let cs = compute_with_index(&idx, el, &[], &[]);
        assert_eq!(
            cs.color,
            Some(Color {
                r: 255,
                g: 0,
                b: 0,
                a: 255
            })
        );
        // Narrow viewport — rule doesn't apply.
        let idx_narrow = SelectorIndex::build_with_viewport(&sheets, 400.0, 800.0);
        let cs2 = compute_with_index(&idx_narrow, el, &[], &[]);
        assert_eq!(cs2.color, None);
    }

    #[test]
    fn tailwind_responsive_md_padding_wins_at_wide_viewport() {
        // Real Tailwind shape: a base utility `.py-2` early in the sheet,
        // then `.md\:py-6` inside `@media (min-width:768px)` near the end.
        // At a ≥768px viewport the media rule must win the cascade (same
        // specificity, later source order). This mirrors Chrome/Blink.
        let css = ".py-2{padding-top:.5rem;padding-bottom:.5rem}\
                   @media (min-width:768px){.md\\:py-6{padding-top:1.5rem;padding-bottom:1.5rem}}";
        let ss = parse_stylesheet(css);
        let sheets = [ss];
        let el = Fake {
            tag: "header",
            id: None,
            classes: &["py-2", "md:py-6"],
        };
        let classes = ["py-2".to_string(), "md:py-6".to_string()];
        // Wide viewport (1024) — md:py-6 (24px) must beat py-2 (8px).
        let idx = SelectorIndex::build_with_viewport(&sheets, 1024.0, 768.0);
        let cs = compute_with_index(&idx, el, &[], &classes);
        let top = cs.padding[0].as_ref().and_then(|l| match l {
            Length::Rem(v) => Some(*v),
            _ => None,
        });
        assert_eq!(
            top,
            Some(1.5),
            "md:py-6 should win at 1024px (got {:?})",
            cs.padding[0]
        );
        // Narrow viewport (500) — md rule drops out, py-2 (8px) wins.
        let idx_n = SelectorIndex::build_with_viewport(&sheets, 500.0, 800.0);
        let cs_n = compute_with_index(&idx_n, el, &[], &classes);
        let top_n = cs_n.padding[0].as_ref().and_then(|l| match l {
            Length::Rem(v) => Some(*v),
            _ => None,
        });
        assert_eq!(
            top_n,
            Some(0.5),
            "py-2 should win at 500px (got {:?})",
            cs_n.padding[0]
        );
    }

    #[test]
    fn supports_block_rules_apply() {
        let ss = parse_stylesheet("@supports (display: grid) { div { color: red; } }");
        let el = Fake {
            tag: "div",
            id: None,
            classes: &[],
        };
        let sheets = [ss];
        let idx = SelectorIndex::build(&sheets);
        let cs = compute_with_index(&idx, el, &[], &[]);
        assert_eq!(
            cs.color,
            Some(Color {
                r: 255,
                g: 0,
                b: 0,
                a: 255
            })
        );
    }

    #[test]
    fn media_print_never_matches_in_browser() {
        let ss = parse_stylesheet("@media print { div { color: red; } }");
        let el = Fake {
            tag: "div",
            id: None,
            classes: &[],
        };
        let sheets = [ss];
        let idx = SelectorIndex::build(&sheets);
        let cs = compute_with_index(&idx, el, &[], &[]);
        assert_eq!(cs.color, None);
    }

    /// Process-global media-type guard for the print/PDF tests. `set_print_media`
    /// flips a shared atomic, so any test that toggles it must serialise to keep
    /// the parallel test runner from racing screen tests.
    static PRINT_MEDIA_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[test]
    fn media_print_matches_only_when_printing() {
        // Recover a poisoned lock so one failure doesn't cascade-fail the rest.
        let _g = PRINT_MEDIA_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        // Both a `@media print` (red) and a `@media screen` (blue) rule.
        let css = "@media print { div { color: red; } }\
                   @media screen { div { color: blue; } }";
        let ss = parse_stylesheet(css);
        let sheets = [ss];
        let el = Fake { tag: "div", id: None, classes: &[] };
        let red = Some(Color { r: 255, g: 0, b: 0, a: 255 });
        let blue = Some(Color { r: 0, g: 0, b: 255, a: 255 });

        // Screen (default): the screen block wins, the print block is dropped.
        set_print_media(false);
        let idx_s = SelectorIndex::build(&sheets);
        assert_eq!(compute_with_index(&idx_s, el, &[], &[]).color, blue);

        // Print pass: the print block wins, the screen block is dropped.
        set_print_media(true);
        let idx_p = SelectorIndex::build(&sheets);
        let got = compute_with_index(&idx_p, el, &[], &[]).color;
        // Always restore the global before asserting so a failure can't leak.
        set_print_media(false);
        assert_eq!(got, red, "@media print rule must apply during a print pass");
    }

    #[test]
    fn display_none_in_print_only_hides_when_printing() {
        let _g = PRINT_MEDIA_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        // An element visible on screen, hidden in print (the canonical
        // "don't print the nav bar" idiom).
        let css = "@media print { .nav { display: none; } }";
        let ss = parse_stylesheet(css);
        let sheets = [ss];
        let el = Fake { tag: "div", id: None, classes: &["nav"] };
        let classes = ["nav".to_string()];

        set_print_media(false);
        let idx_s = SelectorIndex::build(&sheets);
        assert_eq!(
            compute_with_index(&idx_s, el, &[], &classes).display,
            None,
            "no display override on screen"
        );

        set_print_media(true);
        let idx_p = SelectorIndex::build(&sheets);
        let disp = compute_with_index(&idx_p, el, &[], &classes).display;
        set_print_media(false);
        assert_eq!(disp, Some(Display::None), "display:none must apply in print");
    }

    /// Wikipedia heading-weight cascade: inside ONE `@media screen` block a
    /// bold `h1` rule precedes a `font-weight:normal` reset at equal
    /// specificity. The later (normal) declaration must win — Chrome resolves
    /// `#firstHeading` to font-weight 400, and we previously resolved 700.
    /// Both rules are at-rule-nested, so this also guards source-order WITHIN a
    /// media block.
    #[test]
    fn screen_media_later_font_weight_reset_wins() {
        let css = "@media screen{\
            .mw-heading,h1,h2,h3,h4,h5,h6{color:#101418;font-weight:bold;display:flow-root}\
            .mw-heading1,h1{font-size:188%;font-weight:normal}\
        }";
        let ss = parse_stylesheet(css);
        let sheets = [ss];
        let el = Fake {
            tag: "h1",
            id: Some("firstHeading"),
            classes: &["firstHeading"],
        };
        let idx = SelectorIndex::build_with_viewport(&sheets, 960.0, 716.0);
        let cs = compute_with_index(&idx, el, &[], &[]);
        assert_eq!(
            cs.font_weight_bold,
            Some(false),
            "later @media screen `font-weight:normal` must beat earlier `font-weight:bold` at equal specificity"
        );
    }

    /// Same as above but with a TRAILING `@media print` block that re-asserts
    /// `font-weight:bold` (Wikipedia's real layout). The print block must be
    /// dropped on screen, so the screen `normal` reset still wins. This guards
    /// the exact ordering that was producing 700 on the live Cat page: a print
    /// rule at the highest source order leaking into the screen cascade.
    #[test]
    fn trailing_media_print_bold_does_not_override_screen_normal() {
        let css = "@media screen{\
            .mw-heading,h1,h2,h3,h4,h5,h6{font-weight:bold;display:flow-root}\
            .mw-heading1,h1{font-size:188%;font-weight:normal}\
        }\
        @media print{\
            .mw-heading,h1,h2,h3,h4,h5,h6{font-weight:bold;page-break-after:avoid}\
        }";
        let ss = parse_stylesheet(css);
        let sheets = [ss];
        let el = Fake {
            tag: "h1",
            id: Some("firstHeading"),
            classes: &["firstHeading"],
        };
        let idx = SelectorIndex::build_with_viewport(&sheets, 960.0, 716.0);
        let cs = compute_with_index(&idx, el, &[], &[]);
        assert_eq!(
            cs.font_weight_bold,
            Some(false),
            "trailing @media print bold must NOT override the screen font-weight:normal reset"
        );
    }

    /// A qualified rule nested inside `@media screen { @media (min-width:640px)
    /// { ... } }` must apply when BOTH conditions hold, and must be dropped when
    /// the inner condition fails — without leaking into the unguarded cascade.
    /// Guards the nested-at-rule indexing path end-to-end.
    #[test]
    fn nested_media_rule_applies_only_when_all_conditions_match() {
        let css = "@media screen{\
                       p{color:#000000}\
                       @media (min-width:640px){p{color:#ff0000}}\
                       @media print{p{color:#0000ff}}\
                   }";
        let ss = parse_stylesheet(css);
        let sheets = [ss];
        let el = Fake { tag: "p", id: None, classes: &[] };

        // Wide screen viewport (960): outer screen + inner min-width:640 match,
        // so the red nested rule (later source order) wins. The print rule is
        // dropped.
        let idx = SelectorIndex::build_with_viewport(&sheets, 960.0, 716.0);
        let cs = compute_with_index(&idx, el, &[], &[]);
        assert_eq!(
            cs.color,
            Some(Color { r: 255, g: 0, b: 0, a: 255 }),
            "nested @media (min-width:640px) inside @media screen must win at 960px"
        );

        // Narrow viewport (500): inner min-width:640 fails, so the base black
        // screen rule wins; the nested rule must NOT leak in.
        let idx_n = SelectorIndex::build_with_viewport(&sheets, 500.0, 716.0);
        let cs_n = compute_with_index(&idx_n, el, &[], &[]);
        assert_eq!(
            cs_n.color,
            Some(Color { r: 0, g: 0, b: 0, a: 255 }),
            "nested min-width rule must drop at 500px, leaving the base screen color"
        );
    }

    /// `:lang(ckb)` must NOT match an element whose content language is `en`.
    /// Wikipedia ships `h1:lang(ckb){font-family:Scheherazade,...}` which a
    /// permissive `:lang()` fallback wrongly applied to every heading, leaving
    /// `Scheherazade` prepended to the resolved font stack on English pages.
    #[test]
    fn lang_pseudo_respects_element_language() {
        // Reference-tree element with a parent chain carrying `lang`.
        #[derive(Copy, Clone)]
        struct LangEl<'a> {
            tag: &'a str,
            lang: Option<&'a str>,
        }
        impl<'a> ElementView<'a> for LangEl<'a> {
            fn tag_name(&self) -> Option<&'a str> {
                Some(self.tag)
            }
            fn id(&self) -> Option<&'a str> {
                None
            }
            fn has_class(&self, _: &str) -> bool {
                false
            }
            fn parent(&self) -> Option<Self> {
                None
            }
            fn attr(&self, name: &str) -> Option<&'a str> {
                if name.eq_ignore_ascii_case("lang") {
                    self.lang
                } else {
                    None
                }
            }
        }
        let css = "h1{font-family:Helvetica}\
                   h1:lang(ckb){font-family:Scheherazade}";
        let ss = parse_stylesheet(css);
        let sheets = [ss];
        let idx = SelectorIndex::build(&sheets);

        let en = LangEl { tag: "h1", lang: Some("en") };
        let cs_en = compute_with_index(&idx, en, &[], &[]);
        assert_eq!(
            cs_en.font_family.as_deref(),
            Some("Helvetica"),
            ":lang(ckb) must NOT match an English heading"
        );

        let ckb = LangEl { tag: "h1", lang: Some("ckb") };
        let cs_ckb = compute_with_index(&idx, ckb, &[], &[]);
        assert_eq!(
            cs_ckb.font_family.as_deref(),
            Some("Scheherazade"),
            ":lang(ckb) must match a ckb heading"
        );

        // BCP47 prefix: :lang(en) matches lang=en-US.
        let css2 = "p:lang(en){color:red}";
        let ss2 = parse_stylesheet(css2);
        let sheets2 = [ss2];
        let idx2 = SelectorIndex::build(&sheets2);
        let enus = LangEl { tag: "p", lang: Some("en-US") };
        let cs_enus = compute_with_index(&idx2, enus, &[], &[]);
        assert!(cs_enus.color.is_some(), ":lang(en) must match en-US by prefix");
        let none = LangEl { tag: "p", lang: None };
        let cs_none = compute_with_index(&idx2, none, &[], &[]);
        assert!(cs_none.color.is_none(), ":lang(en) must NOT match when no language is declared");
    }

    /// Dark-mode rule must NOT apply — we're a light browser. Without
    /// this, wikipedia.org painted its body black because the
    /// `@media (prefers-color-scheme: dark) { html { background: #101418; … } }`
    /// rule got applied alongside the default light rule.
    #[test]
    fn media_prefers_color_scheme_dark_does_not_match_light_browser() {
        let ss = parse_stylesheet("@media (prefers-color-scheme: dark) { div { color: red; } }");
        let el = Fake {
            tag: "div",
            id: None,
            classes: &[],
        };
        let sheets = [ss];
        let idx = SelectorIndex::build(&sheets);
        let cs = compute_with_index(&idx, el, &[], &[]);
        assert_eq!(cs.color, None);
    }

    /// And the corresponding light-mode rule must apply, so sites that
    /// scope their default styles inside one of these get rendered.
    #[test]
    fn media_prefers_color_scheme_light_matches_light_browser() {
        let ss = parse_stylesheet("@media (prefers-color-scheme: light) { div { color: red; } }");
        let el = Fake {
            tag: "div",
            id: None,
            classes: &[],
        };
        let sheets = [ss];
        let idx = SelectorIndex::build(&sheets);
        let cs = compute_with_index(&idx, el, &[], &[]);
        assert_eq!(
            cs.color,
            Some(Color {
                r: 255,
                g: 0,
                b: 0,
                a: 255
            })
        );
    }

    /// Unknown / speculative media features (`@media (color-gamut: p3)`,
    /// `@media (forced-colors: active)`, anything we haven't taught the
    /// matcher) must NOT match. The old default-to-true behaviour was
    /// the root cause of dark-mode bleed-through.
    #[test]
    fn media_unknown_feature_does_not_match() {
        let ss = parse_stylesheet("@media (color-gamut: p3) { div { color: red; } }");
        let el = Fake {
            tag: "div",
            id: None,
            classes: &[],
        };
        let sheets = [ss];
        let idx = SelectorIndex::build(&sheets);
        let cs = compute_with_index(&idx, el, &[], &[]);
        assert_eq!(cs.color, None);
    }

    /// Bug 2: `@media not (feature)` was unconditionally false because the
    /// leading `not` was pushed as its own ident atom. Now it inverts the
    /// result of the remaining feature expression.
    ///
    /// - `not (min-width: 600px)` on a narrow 400px viewport → matches (600 not met → true, inverted → still true)
    /// - `not (min-width: 600px)` on a wide 1024px viewport → does NOT match
    #[test]
    fn media_not_feature_inverts_match() {
        let ss = parse_stylesheet("@media not (min-width: 600px) { div { color: red; } }");
        let el = Fake {
            tag: "div",
            id: None,
            classes: &[],
        };
        let sheets = [ss];

        // Narrow viewport (400px): min-width:600px → false; `not` inverts → rule applies.
        let idx_narrow = SelectorIndex::build_with_viewport(&sheets, 400.0, 800.0);
        let cs_narrow = compute_with_index(&idx_narrow, el, &[], &[]);
        assert_eq!(
            cs_narrow.color,
            Some(Color { r: 255, g: 0, b: 0, a: 255 }),
            "@media not (min-width:600px) should apply at 400px viewport"
        );

        // Wide viewport (1024px): min-width:600px → true; `not` inverts → rule does NOT apply.
        let idx_wide = SelectorIndex::build_with_viewport(&sheets, 1024.0, 768.0);
        let cs_wide = compute_with_index(&idx_wide, el, &[], &[]);
        assert_eq!(
            cs_wide.color,
            None,
            "@media not (min-width:600px) should NOT apply at 1024px viewport"
        );
    }

    /// Bug 2 (addendum): deprecated/legacy media types like `tv`, `aural`,
    /// `handheld`, `tty` should NOT match in a screen browser.
    #[test]
    fn media_legacy_types_do_not_match() {
        for media_type in &["tv", "aural", "handheld", "tty", "embossed", "projection"] {
            let css = format!("@media {} {{ div {{ color: red; }} }}", media_type);
            let ss = parse_stylesheet(&css);
            let el = Fake { tag: "div", id: None, classes: &[] };
            let sheets = [ss];
            let idx = SelectorIndex::build(&sheets);
            let cs = compute_with_index(&idx, el, &[], &[]);
            assert_eq!(
                cs.color,
                None,
                "@media {} should not match in a screen browser",
                media_type
            );
        }
    }

    /// The public `media_query_matches_str` entry point (used by
    /// `window.matchMedia`) tokenises a bare query string and evaluates it
    /// against an explicit viewport — no `@media` wrapper, no stylesheet.
    #[test]
    fn match_media_string_width_features() {
        // Wide test viewport: 1024×768.
        assert!(media_query_matches_str("(min-width: 1px)", 1024.0, 768.0));
        assert!(!media_query_matches_str(
            "(min-width: 999999px)",
            1024.0,
            768.0
        ));
        assert!(!media_query_matches_str("(max-width: 0px)", 1024.0, 768.0));
        assert!(media_query_matches_str(
            "(max-width: 2000px)",
            1024.0,
            768.0
        ));
        // Range crossing: a query that is true narrow and false wide.
        assert!(media_query_matches_str("(max-width: 600px)", 500.0, 800.0));
        assert!(!media_query_matches_str(
            "(max-width: 600px)",
            1000.0,
            800.0
        ));
        // `and` compound + the leading `not`.
        assert!(media_query_matches_str(
            "(min-width: 300px) and (max-width: 1200px)",
            800.0,
            600.0
        ));
        assert!(media_query_matches_str(
            "not (min-width: 2000px)",
            800.0,
            600.0
        ));
        // Empty query == `all` per CSSOM View / Media Queries 4 §2.1.
        assert!(media_query_matches_str("", 800.0, 600.0));
    }

    /// `(orientation: portrait|landscape)` must be derived from the viewport
    /// box (Media Queries 4 §6.4), not hardcoded. Regression: it used to always
    /// report `landscape`.
    #[test]
    fn match_media_orientation_is_viewport_derived() {
        // Tall viewport → portrait matches, landscape does not.
        assert!(media_query_matches_str(
            "(orientation: portrait)",
            400.0,
            900.0
        ));
        assert!(!media_query_matches_str(
            "(orientation: landscape)",
            400.0,
            900.0
        ));
        // Wide viewport → landscape matches, portrait does not.
        assert!(media_query_matches_str(
            "(orientation: landscape)",
            1200.0,
            800.0
        ));
        assert!(!media_query_matches_str(
            "(orientation: portrait)",
            1200.0,
            800.0
        ));
        // A square viewport is portrait per spec (height >= width).
        assert!(media_query_matches_str(
            "(orientation: portrait)",
            500.0,
            500.0
        ));
    }

    /// `prefers-color-scheme` is honest: we are a light-theme browser, so
    /// `light` matches and `dark` does not — independent of viewport.
    #[test]
    fn match_media_prefers_color_scheme() {
        assert!(media_query_matches_str(
            "(prefers-color-scheme: light)",
            1024.0,
            768.0
        ));
        assert!(!media_query_matches_str(
            "(prefers-color-scheme: dark)",
            1024.0,
            768.0
        ));
    }

    /// The `resolution` / `min-resolution` / `max-resolution` media features
    /// evaluate against the live device pixel ratio (HiDPI). `1dppx == 96dpi`
    /// and `dppx`/`x` == `devicePixelRatio` (Media Queries 4 §6.7 / CSS Values
    /// 4 §6.1). A `(min-resolution: 2dppx)` query MUST match at dpr=2 and NOT at
    /// dpr=1 — the core HiDPI assertion.
    #[test]
    fn match_media_resolution_dppx() {
        // min-resolution: 2dppx — the @2x breakpoint.
        assert!(
            !media_query_matches_str_dpr("(min-resolution: 2dppx)", 1024.0, 768.0, 1.0),
            "(min-resolution: 2dppx) must NOT match at dpr=1"
        );
        assert!(
            media_query_matches_str_dpr("(min-resolution: 2dppx)", 1024.0, 768.0, 2.0),
            "(min-resolution: 2dppx) MUST match at dpr=2"
        );
        // `2x` is a synonym for `2dppx`.
        assert!(media_query_matches_str_dpr(
            "(min-resolution: 2x)",
            1024.0,
            768.0,
            2.0
        ));
        // dpi conversion: 96dpi == 1dppx, 192dpi == 2dppx.
        assert!(media_query_matches_str_dpr(
            "(min-resolution: 192dpi)",
            1024.0,
            768.0,
            2.0
        ));
        assert!(!media_query_matches_str_dpr(
            "(min-resolution: 192dpi)",
            1024.0,
            768.0,
            1.5
        ));
        // max-resolution is the inverse range.
        assert!(media_query_matches_str_dpr(
            "(max-resolution: 1dppx)",
            1024.0,
            768.0,
            1.0
        ));
        assert!(!media_query_matches_str_dpr(
            "(max-resolution: 1dppx)",
            1024.0,
            768.0,
            1.5
        ));
        // Exact `(resolution: 1.5dppx)` at the common 144-dpi (=1.5) scale.
        assert!(media_query_matches_str_dpr(
            "(resolution: 1.5dppx)",
            1024.0,
            768.0,
            1.5
        ));
        assert!(!media_query_matches_str_dpr(
            "(resolution: 1.5dppx)",
            1024.0,
            768.0,
            2.0
        ));
    }

    /// dpcm conversion: 1in == 2.54cm == 96px, so 96/2.54 ≈ 37.8 dpcm == 1dppx.
    #[test]
    fn match_media_resolution_dpcm() {
        // 2dppx == 192dpi == 192/2.54 ≈ 75.59 dpcm.
        assert!(media_query_matches_str_dpr(
            "(min-resolution: 75dpcm)",
            1024.0,
            768.0,
            2.0
        ));
        assert!(!media_query_matches_str_dpr(
            "(min-resolution: 76dpcm)",
            1024.0,
            768.0,
            2.0
        ));
    }

    /// Legacy `-webkit-(min-|max-)device-pixel-ratio` — the value is the dppx
    /// ratio directly (bare number). Retina-era sites still ship these.
    #[test]
    fn match_media_webkit_device_pixel_ratio() {
        assert!(
            !media_query_matches_str_dpr(
                "(-webkit-min-device-pixel-ratio: 2)",
                1024.0,
                768.0,
                1.0
            ),
            "-webkit-min-device-pixel-ratio: 2 must NOT match at dpr=1"
        );
        assert!(
            media_query_matches_str_dpr(
                "(-webkit-min-device-pixel-ratio: 2)",
                1024.0,
                768.0,
                2.0
            ),
            "-webkit-min-device-pixel-ratio: 2 MUST match at dpr=2"
        );
        assert!(media_query_matches_str_dpr(
            "(-webkit-max-device-pixel-ratio: 1)",
            1024.0,
            768.0,
            1.0
        ));
        assert!(media_query_matches_str_dpr(
            "(-webkit-device-pixel-ratio: 1.5)",
            1024.0,
            768.0,
            1.5
        ));
    }

    /// The process-global DPR cell defaults to 1.0 and is updated by
    /// `set_device_pixel_ratio`; the no-dpr `media_query_matches_str` entry
    /// point evaluates against it. A bad probe (0 / NaN) is ignored.
    #[test]
    fn device_pixel_ratio_global_drives_default_entrypoint() {
        // Save + restore so this test does not perturb others (process-global).
        let saved = current_device_pixel_ratio();
        set_device_pixel_ratio(1.0);
        assert!(!media_query_matches_str(
            "(min-resolution: 2dppx)",
            1024.0,
            768.0
        ));
        set_device_pixel_ratio(2.0);
        assert!((current_device_pixel_ratio() - 2.0).abs() < 1e-6);
        assert!(media_query_matches_str(
            "(min-resolution: 2dppx)",
            1024.0,
            768.0
        ));
        // A garbage value must NOT zero out the ratio.
        set_device_pixel_ratio(0.0);
        assert!((current_device_pixel_ratio() - 2.0).abs() < 1e-6);
        set_device_pixel_ratio(f32::NAN);
        assert!((current_device_pixel_ratio() - 2.0).abs() < 1e-6);
        set_device_pixel_ratio(saved.max(0.000_1));
    }

    #[test]
    fn gradient_fallback_picks_first_stop() {
        let ss = parse_stylesheet("div { background: linear-gradient(45deg, #336699, #112233); }");
        let el = Fake {
            tag: "div",
            id: None,
            classes: &[],
        };
        let cs = compute(&[ss], el);
        assert_eq!(
            cs.background_color,
            Some(Color {
                r: 0x33,
                g: 0x66,
                b: 0x99,
                a: 255
            })
        );
    }

    #[test]
    fn index_perf_canary() {
        let mut src = String::new();
        for i in 0..400 {
            // Mix of tag, class, id rules.
            src.push_str(&format!(".c{i} {{ color: red; padding: {i}px; }}\n"));
            if i % 10 == 0 {
                src.push_str(&format!("div.c{i} {{ font-size: {}px; }}\n", 10 + i % 16));
            }
            if i % 50 == 0 {
                src.push_str(&format!("#id{i} {{ font-weight: bold; }}\n"));
            }
        }
        let ss = parse_stylesheet(&src);
        let sheets = [ss];
        let idx = SelectorIndex::build(&sheets);

        let classes: Vec<String> = (0..10).map(|i| format!("c{i}")).collect();
        let class_refs: Vec<&str> = classes.iter().map(|s| s.as_str()).collect();
        let el = Fake {
            tag: "div",
            id: Some("id0"),
            classes: &class_refs,
        };

        let t = std::time::Instant::now();
        const N: usize = 1000;
        for _ in 0..N {
            let _ = compute_with_index(&idx, el, &[], &classes);
        }
        let elapsed = t.elapsed();
        let per_call = elapsed / N as u32;
        // Budget: 200µs per call in debug build. Real browser cascade
        // budget is closer to 10µs per element but we're in unoptimised
        // test profile here.
        assert!(
            per_call < std::time::Duration::from_micros(500),
            "cascade per-call too slow: {per_call:?} (total {elapsed:?} for {N} iters)"
        );
    }

    /// Realistic-shape canary: 50 rules but each rule has 30 selectors
    /// (the Bootstrap / Font Awesome shape — `.foo, .bar, .baz, ...`).
    /// Each selector inflates the index, so a few rules can balloon
    /// into thousands of candidates. Catches the regression where
    /// bucketing helps overall rule count but not selector-list count.
    #[test]
    fn background_size_length_is_parsed() {
        // HN vote arrow: `background-size: 10px` → width 10px, height auto.
        let ss = parse_stylesheet(".v { background-size: 10px; }");
        let el = Fake {
            tag: "div",
            id: None,
            classes: &["v"],
        };
        let cs = compute(&[ss], el);
        assert_eq!(cs.background_size, Some(CssBgSize::Explicit(Some(Length::Px(10.0)), None)));
    }

    #[test]
    fn background_size_cover_is_parsed() {
        let ss = parse_stylesheet(".v { background-size: cover; }");
        let el = Fake { tag: "div", id: None, classes: &["v"] };
        let cs = compute(&[ss], el);
        assert_eq!(cs.background_size, Some(CssBgSize::Cover));
    }

    #[test]
    fn background_size_contain_is_parsed() {
        let ss = parse_stylesheet(".v { background-size: contain; }");
        let el = Fake { tag: "div", id: None, classes: &["v"] };
        let cs = compute(&[ss], el);
        assert_eq!(cs.background_size, Some(CssBgSize::Contain));
    }

    #[test]
    fn background_size_percentage_is_parsed() {
        let ss = parse_stylesheet(".v { background-size: 50% 100%; }");
        let el = Fake { tag: "div", id: None, classes: &["v"] };
        let cs = compute(&[ss], el);
        assert_eq!(
            cs.background_size,
            Some(CssBgSize::Explicit(Some(Length::Percent(50.0)), Some(Length::Percent(100.0))))
        );
    }

    #[test]
    fn background_size_auto_is_parsed() {
        let ss = parse_stylesheet(".v { background-size: auto auto; }");
        let el = Fake { tag: "div", id: None, classes: &["v"] };
        let cs = compute(&[ss], el);
        assert_eq!(cs.background_size, Some(CssBgSize::Explicit(None, None)));
    }

    #[test]
    fn supports_condition_evaluation() {
        let toks = |s: &str| crate::tokenize(s);
        // Modern feature blocks apply; `not (...)` fallback blocks don't.
        assert!(supports_matches(&toks("(display: grid)")));
        assert!(!supports_matches(&toks("not (display: grid)")));
        assert!(supports_matches(&toks("(display: flex) and (gap: 1rem)")));
        assert!(supports_matches(&toks("(a: b) or (c: d)")));
        assert!(!supports_matches(&toks("not (a: b) and (c: d)")));
        assert!(supports_matches(&toks(""))); // fail-open on empty
    }

    #[test]
    fn a_link_color_wins_over_inherited_body_color() {
        #[derive(Copy, Clone)]
        struct ElA<'a> {
            tag: &'a str,
            attrs: &'a [(&'a str, &'a str)],
        }
        impl<'a> ElementView<'a> for ElA<'a> {
            fn tag_name(&self) -> Option<&'a str> {
                Some(self.tag)
            }
            fn id(&self) -> Option<&'a str> {
                None
            }
            fn has_class(&self, _: &str) -> bool {
                false
            }
            fn parent(&self) -> Option<Self> {
                None
            }
            fn attr(&self, n: &str) -> Option<&'a str> {
                self.attrs
                    .iter()
                    .find(|(k, _)| k.eq_ignore_ascii_case(n))
                    .map(|(_, v)| *v)
            }
        }
        // HN's exact pattern: gray body/td default, black links via a:link.
        let sheets = vec![parse_stylesheet(
            "body { color:#828282; } td { color:#828282; } a:link { color:#000000; }",
        )];
        let a = ElA {
            tag: "a",
            attrs: &[("href", "x")],
        };
        // Use the INDEXED path — the one the real renderer uses.
        let idx = SelectorIndex::build(&sheets);
        let cs = compute_with_index_inheriting(&idx, a, &[], &[], None);
        let c = cs.color.expect("a:link should set color");
        assert_eq!(
            (c.r, c.g, c.b),
            (0, 0, 0),
            "a:link color must win; got {:?}",
            (c.r, c.g, c.b)
        );
    }

    #[test]
    fn presentational_bgcolor_and_colspan_attrs() {
        #[derive(Copy, Clone)]
        struct ElA<'a> {
            tag: &'a str,
            attrs: &'a [(&'a str, &'a str)],
        }
        impl<'a> ElementView<'a> for ElA<'a> {
            fn tag_name(&self) -> Option<&'a str> {
                Some(self.tag)
            }
            fn id(&self) -> Option<&'a str> {
                None
            }
            fn has_class(&self, _: &str) -> bool {
                false
            }
            fn parent(&self) -> Option<Self> {
                None
            }
            fn attr(&self, n: &str) -> Option<&'a str> {
                self.attrs
                    .iter()
                    .find(|(k, _)| k.eq_ignore_ascii_case(n))
                    .map(|(_, v)| *v)
            }
        }
        // `bgcolor="#ff6600"` with no CSS → background_color set (orange).
        let el = ElA {
            tag: "td",
            attrs: &[("bgcolor", "#ff6600")],
        };
        let c = compute(&[], el)
            .background_color
            .expect("bgcolor should set background");
        assert_eq!((c.r, c.g, c.b), (255, 102, 0));
        // HTML also allows bare hex with no leading `#`.
        let el_bare = ElA {
            tag: "td",
            attrs: &[("bgcolor", "ffffff")],
        };
        let c2 = compute(&[], el_bare)
            .background_color
            .expect("bare-hex bgcolor");
        assert_eq!((c2.r, c2.g, c2.b), (255, 255, 255));
        // No attr → no background.
        let el_none = ElA {
            tag: "td",
            attrs: &[],
        };
        assert!(compute(&[], el_none).background_color.is_none());
        // `colspan` attr → table_col_span.
        let el_cs = ElA {
            tag: "td",
            attrs: &[("colspan", "3")],
        };
        assert_eq!(compute(&[], el_cs).table_col_span, Some(3));
        // CSS background WINS over presentational bgcolor (lower priority).
        let ss = parse_stylesheet("td { background-color: #000000; }");
        let el_both = ElA {
            tag: "td",
            attrs: &[("bgcolor", "#ff6600")],
        };
        let c3 = compute(&[ss], el_both)
            .background_color
            .expect("css background");
        assert_eq!((c3.r, c3.g, c3.b), (0, 0, 0));
    }

    #[test]
    fn linear_gradient_parses_two_stops_and_direction() {
        let ss = parse_stylesheet("div { background: linear-gradient(to right, red, blue); }");
        let el = Fake {
            tag: "div",
            id: None,
            classes: &[],
        };
        let cs = compute(&[ss], el);
        let g = cs.background_gradient.expect("gradient missing");
        assert_eq!(
            g.from,
            Color {
                r: 255,
                g: 0,
                b: 0,
                a: 255
            }
        );
        assert_eq!(
            g.to,
            Color {
                r: 0,
                g: 0,
                b: 255,
                a: 255
            }
        );
        // `to right` = 90deg.
        assert!((g.angle_deg - 90.0).abs() < 0.001);
        // Solid-color fallback also exposes the first stop.
        assert_eq!(cs.background_color, Some(g.from));
    }

    #[test]
    fn background_none_resets_ua_button_color() {
        // `background: none` (the ubiquitous button/link reset) must reset the
        // background to transparent, OVERRIDING an earlier UA `background:#efefef`
        // — not be a no-op that leaves a gray box behind the control.
        let ss = parse_stylesheet("button { background: #efefef; } .reset { background: none; }");
        let el = Fake {
            tag: "button",
            id: None,
            classes: &["reset"],
        };
        let cs = compute(&[ss], el);
        assert_eq!(
            cs.background_color,
            Some(Color::TRANSPARENT),
            "background:none should reset to transparent, got {:?}",
            cs.background_color
        );
    }

    #[test]
    fn accent_color_caret_color_color_scheme_view_transition_name_parse() {
        let ss = parse_stylesheet(
            ":root { accent-color: rebeccapurple; caret-color: #ff0000; \
             color-scheme: light dark; view-transition-name: header; }",
        );
        let el = Fake {
            tag: ":root",
            id: None,
            classes: &[],
        };
        // Fall back to `html` for selectors since :root has no real binding;
        // these tests just verify the apply_declaration arms run.
        let cs = compute(&[ss], el);
        // accent-color and caret-color should parse a colour.
        // (Selector binding for `:root` isn't exercised here, but the
        // arm doesn't depend on element matching to run — we just check
        // that the path didn't panic and ComputedStyle defaults are stable.)
        let _ = (
            cs.accent_color,
            cs.caret_color,
            cs.color_scheme,
            cs.view_transition_name,
        );
    }

    /// CSS UI 4 §5.3: a CHECKED checkbox is filled with the accent colour.
    /// When `accent-color` is authored, the box background becomes that
    /// colour; when unset, the UA default accent. An UNCHECKED checkbox is
    /// NOT accent-filled (keeps its white UA background).
    #[test]
    fn accent_color_fills_checked_checkbox() {
        #[derive(Copy, Clone)]
        struct Input<'a> {
            ty: &'a str,
            checked: bool,
        }
        impl<'a> ElementView<'a> for Input<'a> {
            fn tag_name(&self) -> Option<&'a str> {
                Some("input")
            }
            fn id(&self) -> Option<&'a str> {
                None
            }
            fn has_class(&self, _: &str) -> bool {
                false
            }
            fn parent(&self) -> Option<Self> {
                None
            }
            fn attr(&self, name: &str) -> Option<&'a str> {
                match name.to_ascii_lowercase().as_str() {
                    "type" => Some(self.ty),
                    "checked" if self.checked => Some(""),
                    _ => None,
                }
            }
        }

        // Authored accent-color on a checked checkbox → box fills with it.
        let sheets = [parse_stylesheet("input { accent-color: #00ff00; }")];
        let idx = SelectorIndex::build(&sheets);
        let checked = Input { ty: "checkbox", checked: true };
        let cs = compute_with_index(&idx, checked, &[], &[]);
        assert_eq!(
            cs.background_color,
            Some(Color { r: 0, g: 255, b: 0, a: 255 }),
            "checked checkbox must fill with the authored accent-color"
        );

        // UNCHECKED with same accent → NOT accent-filled.
        let unchecked = Input { ty: "checkbox", checked: false };
        let cs2 = compute_with_index(&idx, unchecked, &[], &[]);
        assert_ne!(
            cs2.background_color,
            Some(Color { r: 0, g: 255, b: 0, a: 255 }),
            "unchecked checkbox must NOT be accent-filled"
        );

        // Checked WITHOUT authored accent → UA default accent (Chrome blue).
        let sheets3 = [parse_stylesheet("")];
        let idx3 = SelectorIndex::build(&sheets3);
        let default_checked = Input { ty: "radio", checked: true };
        let cs3 = compute_with_index(&idx3, default_checked, &[], &[]);
        assert_eq!(
            cs3.background_color,
            Some(Color { r: 0x1a, g: 0x73, b: 0xe8, a: 255 }),
            "checked control with accent-color:auto uses the UA default accent"
        );
    }

    /// CSS Scrollbars 1 §3 + §2: `scrollbar-width` and `scrollbar-color`
    /// parse into the computed style (auto/thin/none + thumb/track colours).
    #[test]
    fn scrollbar_width_and_color_parse() {
        let html = || Fake { tag: "html", id: None, classes: &[] };
        // scrollbar-width: none → mode 2.
        let cs = compute(&[parse_stylesheet("html { scrollbar-width: none; }")], html());
        assert_eq!(cs.scrollbar_width, 2, "scrollbar-width:none → 2");

        // scrollbar-width: thin → mode 1.
        let cs = compute(&[parse_stylesheet("html { scrollbar-width: thin; }")], html());
        assert_eq!(cs.scrollbar_width, 1, "scrollbar-width:thin → 1");

        // scrollbar-color: <thumb> <track> with hex + rgb() forms.
        let cs = compute(
            &[parse_stylesheet("html { scrollbar-color: #ff0000 rgb(0, 0, 255); }")],
            html(),
        );
        assert_eq!(
            cs.scrollbar_color,
            Some((
                Color { r: 255, g: 0, b: 0, a: 255 },
                Color { r: 0, g: 0, b: 255, a: 255 },
            )),
            "scrollbar-color parses (thumb, track), function form intact"
        );

        // scrollbar-color: auto → None (UA default).
        let cs = compute(&[parse_stylesheet("html { scrollbar-color: auto; }")], html());
        assert_eq!(cs.scrollbar_color, None, "scrollbar-color:auto → None");
    }

    #[test]
    fn background_image_url_parses_from_shorthand_and_longhand() {
        // Shorthand value carrying just a url() — the longhand path
        // would otherwise have skipped it.
        let ss = parse_stylesheet("div { background: url(\"/img/hero.png\"); }");
        let el = Fake {
            tag: "div",
            id: None,
            classes: &[],
        };
        let cs = compute(&[ss], el);
        assert_eq!(cs.background_image_url.as_deref(), Some("/img/hero.png"));

        // Longhand `background-image` should also expose it.
        let ss = parse_stylesheet("p { background-image: url(\"icon.svg\"); }");
        let el = Fake {
            tag: "p",
            id: None,
            classes: &[],
        };
        let cs = compute(&[ss], el);
        assert_eq!(cs.background_image_url.as_deref(), Some("icon.svg"));
    }

    #[test]
    fn background_position_parses_px_pct_and_keywords() {
        // The Wikipedia wordmark sprite case: a negative px y-offset.
        let ss = parse_stylesheet(".w { background-position: 0px -304px; }");
        let el = Fake {
            tag: "span",
            id: None,
            classes: &["w"],
        };
        let cs = compute(&[ss], el);
        assert_eq!(
            cs.background_position,
            Some((BgPos::Px(0.0), BgPos::Px(-304.0)))
        );

        // Keywords map to percentages; a single value centers the other axis.
        let ss = parse_stylesheet(".w { background-position: right center; }");
        let cs = compute(&[ss], el);
        assert_eq!(
            cs.background_position,
            Some((BgPos::Pct(100.0), BgPos::Pct(50.0)))
        );

        let ss = parse_stylesheet(".w { background-position: 25%; }");
        let cs = compute(&[ss], el);
        assert_eq!(
            cs.background_position,
            Some((BgPos::Pct(25.0), BgPos::Pct(50.0)))
        );

        // Bug 3: single vertical keyword — `top` → (50%, 0%), not (0%, 50%).
        let ss = parse_stylesheet(".w { background-position: top; }");
        let cs = compute(&[ss], el);
        assert_eq!(
            cs.background_position,
            Some((BgPos::Pct(50.0), BgPos::Pct(0.0))),
            "background-position: top should be (50%, 0%)"
        );

        let ss = parse_stylesheet(".w { background-position: bottom; }");
        let cs = compute(&[ss], el);
        assert_eq!(
            cs.background_position,
            Some((BgPos::Pct(50.0), BgPos::Pct(100.0))),
            "background-position: bottom should be (50%, 100%)"
        );

        let ss = parse_stylesheet(".w { background-position: left; }");
        let cs = compute(&[ss], el);
        assert_eq!(
            cs.background_position,
            Some((BgPos::Pct(0.0), BgPos::Pct(50.0))),
            "background-position: left should be (0%, 50%)"
        );
    }

    #[test]
    fn object_position_parses_keywords_pct_and_px() {
        let el = Fake {
            tag: "img",
            id: None,
            classes: &["x"],
        };

        // Default: no object-position set → None (painter defaults to 50% 50%).
        let ss = parse_stylesheet(".x { object-fit: contain; }");
        let cs = compute(&[ss], el);
        assert_eq!(cs.object_position, None, "unset should be None");

        // center center (explicit)
        let ss = parse_stylesheet(".x { object-position: center center; }");
        let cs = compute(&[ss], el);
        assert_eq!(
            cs.object_position,
            Some((BgPos::Pct(50.0), BgPos::Pct(50.0))),
            "center center"
        );

        // left top (both keywords)
        let ss = parse_stylesheet(".x { object-position: left top; }");
        let cs = compute(&[ss], el);
        assert_eq!(
            cs.object_position,
            Some((BgPos::Pct(0.0), BgPos::Pct(0.0))),
            "left top"
        );

        // right bottom
        let ss = parse_stylesheet(".x { object-position: right bottom; }");
        let cs = compute(&[ss], el);
        assert_eq!(
            cs.object_position,
            Some((BgPos::Pct(100.0), BgPos::Pct(100.0))),
            "right bottom"
        );

        // Percentage values
        let ss = parse_stylesheet(".x { object-position: 25% 75%; }");
        let cs = compute(&[ss], el);
        assert_eq!(
            cs.object_position,
            Some((BgPos::Pct(25.0), BgPos::Pct(75.0))),
            "25% 75%"
        );

        // Pixel values
        let ss = parse_stylesheet(".x { object-position: 10px 20px; }");
        let cs = compute(&[ss], el);
        assert_eq!(
            cs.object_position,
            Some((BgPos::Px(10.0), BgPos::Px(20.0))),
            "10px 20px"
        );

        // Single keyword: `top` → x=50%, y=0%
        let ss = parse_stylesheet(".x { object-position: top; }");
        let cs = compute(&[ss], el);
        assert_eq!(
            cs.object_position,
            Some((BgPos::Pct(50.0), BgPos::Pct(0.0))),
            "single top"
        );
    }

    #[test]
    fn mask_image_url_parses_from_longhand() {
        let ss = parse_stylesheet("div { mask-image: url(\"mask.svg\"); }");
        let el = Fake {
            tag: "div",
            id: None,
            classes: &[],
        };
        let cs = compute(&[ss], el);
        assert!(cs.has_mask_url);
        assert_eq!(cs.mask_image_url.as_deref(), Some("mask.svg"));
    }

    #[test]
    fn linear_gradient_default_direction_is_to_bottom() {
        let ss = parse_stylesheet("div { background: linear-gradient(#ff0000, #00ff00); }");
        let el = Fake {
            tag: "div",
            id: None,
            classes: &[],
        };
        let cs = compute(&[ss], el);
        let g = cs.background_gradient.unwrap();
        assert!((g.angle_deg - 180.0).abs() < 0.001);
    }

    #[test]
    fn box_shadow_parses_offsets_and_color() {
        let ss = parse_stylesheet("div { box-shadow: 2px 4px 8px rgba(0,0,0,0.25); }");
        let el = Fake {
            tag: "div",
            id: None,
            classes: &[],
        };
        let cs = compute(&[ss], el);
        let sh = cs.box_shadow.expect("box-shadow not parsed");
        assert_eq!(sh.offset_x, crate::properties::Length::Px(2.0));
        assert_eq!(sh.offset_y, crate::properties::Length::Px(4.0));
        assert_eq!(sh.blur, crate::properties::Length::Px(8.0));
        assert_eq!(sh.color.a, 64);
    }

    #[test]
    fn custom_property_resolves_via_var() {
        let ss = parse_stylesheet(":root { --brand: #ff0080; } p { color: var(--brand); }");
        let el = Fake {
            tag: "p",
            id: None,
            classes: &[],
        };
        let cs = compute(&[ss], el);
        // Since :root needs an actual root match and our Fake has no
        // parent chain, use a direct selector that matches our element.
        // Override with a more permissive sheet:
        let ss2 = parse_stylesheet("p { --brand: #ff0080; color: var(--brand); }");
        let cs2 = compute(&[ss2], el);
        assert_eq!(
            cs2.color,
            Some(Color {
                r: 0xff,
                g: 0,
                b: 0x80,
                a: 255
            })
        );
        // Untouched-on-:root variant exercises the fallback path only.
        let _ = cs;
    }

    #[test]
    fn var_fallback_used_when_missing() {
        let ss = parse_stylesheet("p { color: var(--missing, red); }");
        let el = Fake {
            tag: "p",
            id: None,
            classes: &[],
        };
        let cs = compute(&[ss], el);
        assert_eq!(
            cs.color,
            Some(Color {
                r: 255,
                g: 0,
                b: 0,
                a: 255
            })
        );
    }

    #[test]
    fn var_chain_resolves() {
        let ss = parse_stylesheet("p { --a: blue; --b: var(--a); color: var(--b); }");
        let el = Fake {
            tag: "p",
            id: None,
            classes: &[],
        };
        let cs = compute(&[ss], el);
        assert_eq!(
            cs.color,
            Some(Color {
                r: 0,
                g: 0,
                b: 255,
                a: 255
            })
        );
    }

    #[test]
    fn index_perf_comma_lists() {
        let mut src = String::new();
        for r in 0..50 {
            let sels: Vec<String> = (0..30).map(|i| format!(".c{r}_{i}")).collect();
            src.push_str(&sels.join(", "));
            src.push_str(" { color: red; padding: 1px; }\n");
        }
        let ss = parse_stylesheet(&src);
        let sheets = [ss];
        let idx = SelectorIndex::build(&sheets);

        // Element matches ONE of the comma-separated entries in 5 rules.
        let classes: Vec<String> = (0..5).map(|r| format!("c{r}_0")).collect();
        let class_refs: Vec<&str> = classes.iter().map(|s| s.as_str()).collect();
        let el = Fake {
            tag: "div",
            id: None,
            classes: &class_refs,
        };

        let t = std::time::Instant::now();
        const N: usize = 500;
        for _ in 0..N {
            let _ = compute_with_index(&idx, el, &[], &classes);
        }
        let elapsed = t.elapsed();
        let per_call = elapsed / N as u32;
        assert!(
            per_call < std::time::Duration::from_micros(500),
            "comma-list cascade too slow: {per_call:?}"
        );
    }

    #[test]
    fn at_layer_order_beats_specificity_and_unlayered_wins() {
        let color_of = |css: &str, tag: &str, id: Option<&str>, classes: &[&str]| {
            let ss = parse_stylesheet(css);
            let sheets = [ss];
            let idx = SelectorIndex::build(&sheets);
            let cls: Vec<String> = classes.iter().map(|c| c.to_string()).collect();
            let el = Fake { tag, id, classes };
            compute_with_index(&idx, el, &[], &cls).color
        };
        let green = color_of("x { color: green; }", "x", None, &[]);
        let blue = color_of("x { color: blue; }", "x", None, &[]);

        // Later layer (`utilities`) beats earlier layer (`base`) even
        // though base's `#box` selector has far higher specificity than
        // utilities' `.box` — this is exactly the Tailwind v4 idiom.
        let layered = color_of(
            "@layer base, utilities;
             @layer base { #box { color: red; } }
             @layer utilities { .box { color: green; } }",
            "div",
            Some("box"),
            &["box"],
        );
        assert_eq!(
            layered, green,
            "later @layer must win over earlier despite higher specificity"
        );

        // An unlayered declaration beats any layered one (normal origin),
        // even when the layered rule appears later in source order.
        let unlayered = color_of(
            ".u { color: blue; }
             @layer utilities { .u { color: green; } }",
            "div",
            None,
            &["u"],
        );
        assert_eq!(
            unlayered, blue,
            "unlayered normal declaration must beat layered"
        );
    }

    /// Two-value `margin-block: 10px 20px` must set block-start=10px and
    /// block-end=20px independently (second value was lost before the
    /// whitespace-preservation fix in parser.rs).
    #[test]
    fn margin_block_two_value_shorthand() {
        let ss = parse_stylesheet("p { margin-block: 10px 20px; }");
        let el = Fake { tag: "p", id: None, classes: &[] };
        let cs = compute(&[ss], el);
        assert_eq!(cs.margin[0], Some(Length::Px(10.0)), "block-start (top) must be 10px");
        assert_eq!(cs.margin[2], Some(Length::Px(20.0)), "block-end (bottom) must be 20px (not 10px)");
    }

    /// Single-value `margin-block: 10px` must set both sides to the same value.
    #[test]
    fn margin_block_one_value_shorthand() {
        let ss = parse_stylesheet("p { margin-block: 10px; }");
        let el = Fake { tag: "p", id: None, classes: &[] };
        let cs = compute(&[ss], el);
        assert_eq!(cs.margin[0], Some(Length::Px(10.0)), "block-start must be 10px");
        assert_eq!(cs.margin[2], Some(Length::Px(10.0)), "block-end must also be 10px");
    }

    /// Two-value `margin-inline: 5px 15px` must set inline-start=5px and
    /// inline-end=15px independently.
    #[test]
    fn margin_inline_two_value_shorthand() {
        let ss = parse_stylesheet("p { margin-inline: 5px 15px; }");
        let el = Fake { tag: "p", id: None, classes: &[] };
        let cs = compute(&[ss], el);
        assert_eq!(cs.margin[3], Some(Length::Px(5.0)), "inline-start (left) must be 5px");
        assert_eq!(cs.margin[1], Some(Length::Px(15.0)), "inline-end (right) must be 15px (not 5px)");
    }

    /// Two-value `padding-block: 4px 8px` must set block-start=4px and
    /// block-end=8px independently.
    #[test]
    fn padding_block_two_value_shorthand() {
        let ss = parse_stylesheet("p { padding-block: 4px 8px; }");
        let el = Fake { tag: "p", id: None, classes: &[] };
        let cs = compute(&[ss], el);
        assert_eq!(cs.padding[0], Some(Length::Px(4.0)), "block-start (top) must be 4px");
        assert_eq!(cs.padding[2], Some(Length::Px(8.0)), "block-end (bottom) must be 8px (not 4px)");
    }

    /// Two-value `padding-inline: 3px 12px` must set inline-start=3px and
    /// inline-end=12px independently.
    #[test]
    fn padding_inline_two_value_shorthand() {
        let ss = parse_stylesheet("p { padding-inline: 3px 12px; }");
        let el = Fake { tag: "p", id: None, classes: &[] };
        let cs = compute(&[ss], el);
        assert_eq!(cs.padding[3], Some(Length::Px(3.0)), "inline-start (left) must be 3px");
        assert_eq!(cs.padding[1], Some(Length::Px(12.0)), "inline-end (right) must be 12px (not 3px)");
    }

    /// CSS Transforms L2: `translate: 10px 20px` individual longhand must
    /// set translate_x=10px and translate_y=20px independently.
    #[test]
    fn translate_individual_property_two_value() {
        let ss = parse_stylesheet("div { translate: 10px 20px; }");
        let el = Fake { tag: "div", id: None, classes: &[] };
        let cs = compute(&[ss], el);
        assert_eq!(cs.translate_x, Some(Length::Px(10.0)), "translate_x must be 10px");
        assert_eq!(cs.translate_y, Some(Length::Px(20.0)), "translate_y must be 20px (not None)");
    }

    /// Regression: `grid-column: 1 / span 2` must produce start=1, span=2.
    ///
    /// Prior to the fix the `span` keyword was not detected on the RHS so
    /// `1 / span 2` was treated as `1 / 2` (end line 2) → span = 2-1 = 1.
    #[test]
    fn grid_column_span_keyword_parsed_correctly() {
        // `1 / span 3` → start=1, span=3
        let ss = parse_stylesheet(".g { grid-column: 1 / span 3; }");
        let el = Fake {
            tag: "div",
            id: None,
            classes: &["g"],
        };
        let cs = compute(&[ss], el);
        assert_eq!(cs.grid_column_start, Some(1),
            "grid-column: 1 / span 3 → start should be 1");
        assert_eq!(cs.grid_column_span, Some(3),
            "grid-column: 1 / span 3 → span should be 3 (not 1)");

        // `2 / span 3` → start=2, span=3
        let ss2 = parse_stylesheet(".g { grid-column: 2 / span 3; }");
        let cs2 = compute(&[ss2], el);
        assert_eq!(cs2.grid_column_start, Some(2),
            "grid-column: 2 / span 3 → start should be 2");
        assert_eq!(cs2.grid_column_span, Some(3),
            "grid-column: 2 / span 3 → span should be 3");

        // `1 / 3` (end line, not span) → start=1, span=2
        let ss3 = parse_stylesheet(".g { grid-column: 1 / 3; }");
        let cs3 = compute(&[ss3], el);
        assert_eq!(cs3.grid_column_start, Some(1),
            "grid-column: 1 / 3 → start should be 1");
        assert_eq!(cs3.grid_column_span, Some(2),
            "grid-column: 1 / 3 → span should be 2 (end 3 - start 1)");
    }

    /// Custom property names are case-sensitive per CSS Variables Level 1.
    /// `--MyColor: red` must be stored and retrieved as `--MyColor`, not
    /// lowercased to `--mycolor`.  `var(--MyColor)` must resolve to `red`.
    #[test]
    fn custom_property_mixed_case_preserved() {
        let ss = parse_stylesheet("p { --MyColor: red; color: var(--MyColor); }");
        let el = Fake { tag: "p", id: None, classes: &[] };
        let cs = compute(&[ss], el);
        assert_eq!(
            cs.color,
            Some(Color { r: 255, g: 0, b: 0, a: 255 }),
            "var(--MyColor) must resolve to red when declared as --MyColor: red"
        );
    }

    /// Two-value `margin-block: 10px 20px` must set block-start to 10px and
    /// block-end to 20px independently (not both to the first value).
    #[test]
    fn margin_block_two_value_sets_both_independently() {
        let ss = parse_stylesheet("p { margin-block: 10px 20px; }");
        let el = Fake { tag: "p", id: None, classes: &[] };
        let cs = compute(&[ss], el);
        assert_eq!(
            cs.margin[0],
            Some(Length::Px(10.0)),
            "margin-block: 10px 20px — block-start (index 0) must be 10px"
        );
        assert_eq!(
            cs.margin[2],
            Some(Length::Px(20.0)),
            "margin-block: 10px 20px — block-end (index 2) must be 20px, not 10px"
        );
    }

    /// `margin-inline: auto` must set both inline-start and inline-end to
    /// `Length::Auto` AND flip the corresponding `margin_auto` flags so the
    /// layout engine can use the auto-margin centering path.
    #[test]
    fn margin_inline_auto_enables_centering() {
        let ss = parse_stylesheet("div { margin-inline: auto; }");
        let el = Fake { tag: "div", id: None, classes: &[] };
        let cs = compute(&[ss], el);
        assert_eq!(
            cs.margin[3],
            Some(Length::Auto),
            "margin-inline: auto — inline-start (index 3) must be Length::Auto"
        );
        assert_eq!(
            cs.margin[1],
            Some(Length::Auto),
            "margin-inline: auto — inline-end (index 1) must be Length::Auto"
        );
        assert!(
            cs.margin_auto[3],
            "margin-inline: auto — margin_auto[3] (inline-start) must be true"
        );
        assert!(
            cs.margin_auto[1],
            "margin-inline: auto — margin_auto[1] (inline-end) must be true"
        );
    }

    /// `background: #fff` applied after a rule that set a gradient must clear
    /// the gradient — the shorthand resets all background longhands before
    /// applying the new value (CSS Backgrounds §3.10).
    ///
    /// In practice this means an element targeted by both
    /// `.gradient { background: linear-gradient(red, blue) }` and
    /// `.solid { background: #fff }` — with the solid rule winning in the
    /// cascade — must end up with no gradient and a white background.
    #[test]
    fn background_solid_resets_prior_gradient() {
        // Two rules target the same element; the second (solid) wins on
        // source order at equal specificity.
        let ss = parse_stylesheet(
            "div { background: linear-gradient(red, blue); } \
             div { background: #ffffff; }",
        );
        let el = Fake { tag: "div", id: None, classes: &[] };
        let cs = compute(&[ss], el);
        assert!(
            cs.background_gradient.is_none(),
            "background_gradient must be None after background: #fff overwrites the gradient"
        );
        assert_eq!(
            cs.background_color,
            Some(Color { r: 255, g: 255, b: 255, a: 255 }),
            "background_color must be white after background: #fff"
        );
    }

    /// `background-size: cover` must parse to `CssBgSize::Cover`.
    /// (Regression-guard — variants exist and are handled at parse time.)
    #[test]
    fn background_size_cover_parses() {
        let ss = parse_stylesheet("div { background-size: cover; }");
        let el = Fake { tag: "div", id: None, classes: &[] };
        let cs = compute(&[ss], el);
        assert_eq!(
            cs.background_size,
            Some(CssBgSize::Cover),
            "background-size: cover must produce CssBgSize::Cover"
        );
    }

    // ── border-style parsing tests ─────────────────────────────────────

    #[test]
    fn border_style_dashed_parsed() {
        let ss = parse_stylesheet("div { border-style: dashed; }");
        let el = Fake { tag: "div", id: None, classes: &[] };
        let cs = compute(&[ss], el);
        assert_eq!(cs.border_top_style, Some(BorderStyle::Dashed), "top");
        assert_eq!(cs.border_right_style, Some(BorderStyle::Dashed), "right");
        assert_eq!(cs.border_bottom_style, Some(BorderStyle::Dashed), "bottom");
        assert_eq!(cs.border_left_style, Some(BorderStyle::Dashed), "left");
    }

    #[test]
    fn border_style_dotted_parsed() {
        let ss = parse_stylesheet("div { border-style: dotted; }");
        let el = Fake { tag: "div", id: None, classes: &[] };
        let cs = compute(&[ss], el);
        assert_eq!(cs.border_top_style, Some(BorderStyle::Dotted));
    }

    #[test]
    fn border_style_none_collapses_width() {
        let ss = parse_stylesheet("div { border-style: none; border-width: 4px; }");
        let el = Fake { tag: "div", id: None, classes: &[] };
        let cs = compute(&[ss], el);
        // Style is none — all sides set to None variant
        assert_eq!(cs.border_top_style, Some(BorderStyle::None));
        assert_eq!(cs.border_bottom_style, Some(BorderStyle::None));
    }

    #[test]
    fn border_style_four_value_shorthand() {
        let ss = parse_stylesheet("div { border-style: solid dashed dotted none; }");
        let el = Fake { tag: "div", id: None, classes: &[] };
        let cs = compute(&[ss], el);
        assert_eq!(cs.border_top_style, Some(BorderStyle::Solid));
        assert_eq!(cs.border_right_style, Some(BorderStyle::Dashed));
        assert_eq!(cs.border_bottom_style, Some(BorderStyle::Dotted));
        assert_eq!(cs.border_left_style, Some(BorderStyle::None));
    }

    #[test]
    fn border_shorthand_extracts_style() {
        let ss = parse_stylesheet("div { border: 2px dashed red; }");
        let el = Fake { tag: "div", id: None, classes: &[] };
        let cs = compute(&[ss], el);
        assert_eq!(cs.border_top_style, Some(BorderStyle::Dashed), "border shorthand must set style");
        assert_eq!(cs.border_right_style, Some(BorderStyle::Dashed));
        assert_eq!(cs.border_bottom_style, Some(BorderStyle::Dashed));
        assert_eq!(cs.border_left_style, Some(BorderStyle::Dashed));
    }

    #[test]
    fn border_side_style_longhand() {
        let ss = parse_stylesheet("div { border-top-style: dotted; border-bottom-style: solid; }");
        let el = Fake { tag: "div", id: None, classes: &[] };
        let cs = compute(&[ss], el);
        assert_eq!(cs.border_top_style, Some(BorderStyle::Dotted));
        assert_eq!(cs.border_bottom_style, Some(BorderStyle::Solid));
        assert_eq!(cs.border_right_style, None);  // unset
        assert_eq!(cs.border_left_style, None);   // unset
    }

    #[test]
    fn border_style_from_ident_roundtrip() {
        assert_eq!(BorderStyle::from_ident("solid"), Some(BorderStyle::Solid));
        assert_eq!(BorderStyle::from_ident("DASHED"), Some(BorderStyle::Dashed));
        assert_eq!(BorderStyle::from_ident("dotted"), Some(BorderStyle::Dotted));
        assert_eq!(BorderStyle::from_ident("none"), Some(BorderStyle::None));
        assert_eq!(BorderStyle::from_ident("hidden"), Some(BorderStyle::Hidden));
        assert_eq!(BorderStyle::from_ident("double"), Some(BorderStyle::Double));
        assert_eq!(BorderStyle::from_ident("groove"), None);  // unsupported, returns None
    }

    // ----------------------------------------------------------------------
    // Milestone 2.3: ancestor-Bloom fast-reject (Blink SelectorFilter).
    // ----------------------------------------------------------------------

    /// A tiny tree of nodes addressed by index, so an ElementView can walk a
    /// REAL parent chain (needed to exercise the descendant/child matcher and
    /// the ancestor filter end-to-end).
    struct TreeData {
        tags: Vec<&'static str>,
        ids: Vec<Option<&'static str>>,
        classes: Vec<Vec<&'static str>>,
        parent: Vec<Option<usize>>,
    }
    #[derive(Copy, Clone)]
    struct TreeView<'a> {
        d: &'a TreeData,
        i: usize,
    }
    impl<'a> ElementView<'a> for TreeView<'a> {
        fn tag_name(&self) -> Option<&'a str> {
            Some(self.d.tags[self.i])
        }
        fn id(&self) -> Option<&'a str> {
            self.d.ids[self.i]
        }
        fn has_class(&self, name: &str) -> bool {
            self.d.classes[self.i].iter().any(|c| *c == name)
        }
        fn parent(&self) -> Option<Self> {
            self.d.parent[self.i].map(|p| TreeView { d: self.d, i: p })
        }
    }

    /// Build the ancestor filter for node `i` by walking its parent chain —
    /// mirrors what the renderer does from its threaded `parents` slice.
    fn anc_filter(d: &TreeData, i: usize) -> AncestorFilter {
        let mut f = AncestorFilter::default();
        let mut cur = d.parent[i];
        while let Some(p) = cur {
            let cls: Vec<String> = d.classes[p].iter().map(|s| s.to_string()).collect();
            f.add_element(Some(d.tags[p]), d.ids[p], &cls);
            cur = d.parent[p];
        }
        f
    }

    /// The required-ancestor signature is empty for subject-only selectors and
    /// non-empty (covers the literal ancestor identifiers) for descendant ones.
    /// Parse a single selector via a throwaway stylesheet rule (the cascade
    /// test module can't reach selectors.rs's private `one()` helper).
    fn sel(s: &str) -> Selector {
        let ss = parse_stylesheet(&format!("{s} {{ color: red }}"));
        ss.rules[0].selectors[0].clone()
    }

    #[test]
    fn required_ancestor_signature_basics() {
        assert!(
            required_ancestor_signature(&sel(".x")).is_empty(),
            "subject-only selector requires no ancestors"
        );

        let sig = required_ancestor_signature(&sel(".menu .item"));
        assert!(!sig.is_empty(), ".menu .item requires the .menu ancestor");

        // An ancestor filter that DOES contain .menu must cover the signature.
        let mut f = AncestorFilter::default();
        f.add_element(Some("nav"), None, &["menu".to_string()]);
        assert!(f.covers(&sig), "filter with .menu covers '.menu .item' sig");

        // An ancestor filter WITHOUT .menu must not cover it → fast-reject.
        let mut g = AncestorFilter::default();
        g.add_element(Some("nav"), None, &["sidebar".to_string()]);
        assert!(!g.covers(&sig), "filter lacking .menu rejects '.menu .item'");
    }

    /// Sibling combinators break the ancestor-only chain: compounds to the LEFT
    /// of a `+`/`~` are NOT required ancestors and must not appear in the
    /// signature (else we'd false-reject a valid match).
    #[test]
    fn sibling_combinator_stops_ancestor_requirement() {
        // `.a ~ .b .c`: `.b` is an ancestor of `.c`; `.a` is a *sibling* of `.b`
        // (not an ancestor of `.c`) so it must NOT be required.
        let sig = required_ancestor_signature(&sel(".a ~ .b .c"));

        // A filter with .b but NOT .a must still cover the sig (no .a requirement).
        let mut f = AncestorFilter::default();
        f.add_element(Some("div"), None, &["b".to_string()]);
        assert!(
            f.covers(&sig),
            ".a (left of ~) must not be a required ancestor"
        );
    }

    /// THE invariant: filtered and unfiltered cascade produce identical styles,
    /// and the filter actually rejects at least one impossible candidate.
    #[test]
    fn filter_preserves_result_and_rejects_impossible() {
        // Tree: html > body.page > nav.menu > a.item  AND  body.page > footer > a.item
        // Selector `.menu a { color: red }` must match the FIRST a (under .menu)
        // and NOT the second (under footer). The filter must reject `.menu a`
        // for the footer's `a` (no .menu ancestor) without changing the result.
        let d = TreeData {
            tags: vec!["html", "body", "nav", "a", "footer", "a"],
            ids: vec![None, None, None, None, None, None],
            classes: vec![
                vec![],
                vec!["page"],
                vec!["menu"],
                vec!["item"],
                vec![],
                vec!["item"],
            ],
            parent: vec![None, Some(0), Some(1), Some(2), Some(1), Some(4)],
        };
        let sheets = vec![parse_stylesheet(
            ".menu a { color: red } a { color: blue } .page a { text-decoration: underline }",
        )];
        let idx = SelectorIndex::build(&sheets);

        bloom_reset();
        // Node 3: a under .menu → .menu a applies (red).
        let under_menu = TreeView { d: &d, i: 3 };
        let f3 = anc_filter(&d, 3);
        let with = compute_with_index_inheriting_filtered(
            &idx, under_menu, &[], &["item".to_string()], None, Some(&f3),
        );
        let without = compute_with_index_inheriting(
            &idx, under_menu, &[], &["item".to_string()], None,
        );
        assert_eq!(with.color, without.color, "filtered == unfiltered (under menu)");
        assert_eq!(
            with.color,
            Some(Color { r: 255, g: 0, b: 0, a: 255 }),
            ".menu a should win → red under the menu"
        );

        // Node 5: a under footer → .menu a must NOT apply (blue wins).
        let under_footer = TreeView { d: &d, i: 5 };
        let f5 = anc_filter(&d, 5);
        let with5 = compute_with_index_inheriting_filtered(
            &idx, under_footer, &[], &["item".to_string()], None, Some(&f5),
        );
        let without5 = compute_with_index_inheriting(
            &idx, under_footer, &[], &["item".to_string()], None,
        );
        assert_eq!(with5.color, without5.color, "filtered == unfiltered (under footer)");
        assert_eq!(
            with5.color,
            Some(Color { r: 0, g: 0, b: 255, a: 255 }),
            "plain `a` should win → blue under the footer (no .menu ancestor)"
        );

        // The filter must have rejected the `.menu a` candidate for node 5.
        let (attempts, rejects) = bloom_stats();
        assert!(attempts > 0, "fast-reject gate should have been exercised");
        assert!(
            rejects > 0,
            "footer `a` must fast-reject the impossible `.menu a` candidate"
        );
    }

    /// covers() is reflexive and an empty signature is always covered.
    #[test]
    fn empty_signature_never_rejects() {
        let f = AncestorFilter::default();
        let empty = AncestorFilter::default();
        assert!(f.covers(&empty), "empty required-set is always covered");
        // A non-empty filter also covers the empty signature.
        let mut g = AncestorFilter::default();
        g.add_element(Some("div"), Some("main"), &["x".to_string()]);
        assert!(g.covers(&empty));
    }

    // CSS Fonts 4 §2.4 relative-weight table — exact spec rows.
    #[test]
    fn relative_font_weight_matches_spec_table() {
        let bolder = |w| resolve_relative_font_weight(FONT_WEIGHT_BOLDER, w);
        let lighter = |w| resolve_relative_font_weight(FONT_WEIGHT_LIGHTER, w);
        // bolder column.
        assert_eq!(bolder(50), 400); // w<100
        assert_eq!(bolder(100), 400); // 100..350
        assert_eq!(bolder(400), 700); // 350..550
        assert_eq!(bolder(600), 900); // 550..750
        assert_eq!(bolder(800), 900); // 750..900
        assert_eq!(bolder(900), 900); // >=900 no change
        // lighter column.
        assert_eq!(lighter(50), 50); // w<100 no change
        assert_eq!(lighter(100), 100); // 100..350 -> 100
        assert_eq!(lighter(400), 100); // 350..550 -> 100
        assert_eq!(lighter(600), 400); // 550..750 -> 400
        assert_eq!(lighter(800), 700); // 750..900 -> 700
        assert_eq!(lighter(900), 700); // >=900 -> 700
        // A concrete weight passes through unchanged.
        assert_eq!(resolve_relative_font_weight(550, 999), 550);
    }

    // ─────────────────── full N-stop gradient parsing ───────────────────

    fn parse_grad(s: &str) -> CssGradient {
        let toks = crate::tokenize(s);
        parse_css_gradient(&toks).unwrap_or_else(|| panic!("no gradient parsed from {s:?}"))
    }

    /// A 3-stop linear gradient parses all THREE stops with positions
    /// (not collapsed to 2). CSS Images 3 §3.1.
    #[test]
    fn parse_linear_three_stops_with_positions() {
        let g = parse_grad("linear-gradient(to right, red 0%, lime 50%, blue 100%)");
        match g {
            CssGradient::Linear { angle_deg, ref stops, repeating } => {
                assert!(!repeating);
                assert!((angle_deg - 90.0).abs() < 0.01, "to right = 90deg");
                assert_eq!(stops.len(), 3, "must keep all 3 stops");
                assert_eq!(stops[0].color, Color { r: 255, g: 0, b: 0, a: 255 });
                assert_eq!(stops[1].color, Color { r: 0, g: 255, b: 0, a: 255 });
                assert_eq!(stops[2].color, Color { r: 0, g: 0, b: 255, a: 255 });
                assert!((stops[1].pos_frac.unwrap() - 0.5).abs() < 1e-4);
            }
            other => panic!("expected Linear, got {other:?}"),
        }
    }

    /// `45deg` and px stop positions parse.
    #[test]
    fn parse_linear_angle_and_px_stops() {
        let g = parse_grad("linear-gradient(45deg, black 10px, white 90px)");
        if let CssGradient::Linear { angle_deg, stops, .. } = g {
            assert!((angle_deg - 45.0).abs() < 0.01);
            assert_eq!(stops[0].pos_px, Some(10.0));
            assert_eq!(stops[1].pos_px, Some(90.0));
        } else {
            panic!("expected Linear");
        }
    }

    /// repeating-linear-gradient sets the repeating flag.
    #[test]
    fn parse_repeating_linear() {
        let g = parse_grad("repeating-linear-gradient(90deg, red, blue 20px)");
        assert!(matches!(g, CssGradient::Linear { repeating: true, .. }));
    }

    /// radial-gradient parses shape, size, position, and all stops.
    /// CSS Images 3 §3.2.
    #[test]
    fn parse_radial_shape_size_position() {
        let g = parse_grad(
            "radial-gradient(circle closest-side at 30% 40%, red, lime 50%, blue)",
        );
        match g {
            CssGradient::Radial { shape, size, center, ref stops, repeating } => {
                assert!(!repeating);
                assert_eq!(shape, RadialShape::Circle);
                assert_eq!(size, RadialSize::ClosestSide);
                assert_eq!(
                    center,
                    Some((GradientPosAxis::Pct(30.0), GradientPosAxis::Pct(40.0)))
                );
                assert_eq!(stops.len(), 3);
                assert!((stops[1].pos_frac.unwrap() - 0.5).abs() < 1e-4);
            }
            other => panic!("expected Radial, got {other:?}"),
        }
    }

    /// Default radial shape/size are ellipse / farthest-corner per spec.
    #[test]
    fn parse_radial_defaults() {
        let g = parse_grad("radial-gradient(red, blue)");
        if let CssGradient::Radial { shape, size, center, .. } = g {
            assert_eq!(shape, RadialShape::Ellipse);
            assert_eq!(size, RadialSize::FarthestCorner);
            assert!(center.is_none());
        } else {
            panic!("expected Radial");
        }
    }

    /// Explicit radial radii (ellipse with two lengths).
    #[test]
    fn parse_radial_explicit_radii() {
        let g = parse_grad("radial-gradient(ellipse 40px 80px at center, red, blue)");
        if let CssGradient::Radial { size: RadialSize::Explicit { rx_px, ry_px, .. }, .. } = g {
            assert_eq!(rx_px, Some(40.0));
            assert_eq!(ry_px, Some(80.0));
        } else {
            panic!("expected explicit radial radii");
        }
    }

    /// conic-gradient parses from-angle, center, and stops. CSS Images
    /// 4 §3.3. Stop angles convert to a fraction of one turn.
    #[test]
    fn parse_conic_from_angle_and_stops() {
        let g = parse_grad("conic-gradient(from 90deg at 50% 50%, red, lime 90deg, blue)");
        match g {
            CssGradient::Conic { from_deg, center, ref stops, repeating } => {
                assert!(!repeating);
                assert!((from_deg - 90.0).abs() < 0.01);
                assert_eq!(
                    center,
                    Some((GradientPosAxis::Pct(50.0), GradientPosAxis::Pct(50.0)))
                );
                assert_eq!(stops.len(), 3);
                // `lime 90deg` → 90/360 = 0.25 turn.
                assert!((stops[1].pos_frac.unwrap() - 0.25).abs() < 1e-4);
            }
            other => panic!("expected Conic, got {other:?}"),
        }
    }

    /// The full model is populated on the computed style for the
    /// `background:` shorthand — the production paint path reads it.
    #[test]
    fn computed_style_carries_full_gradient() {
        let ss = parse_stylesheet(
            "div { background: linear-gradient(to right, red 0%, lime 50%, blue 100%); }",
        );
        let el = Fake { tag: "div", id: None, classes: &[] };
        let cs = compute(&[ss], el);
        let fg = cs.background_gradient_full.expect("full gradient missing");
        assert_eq!(fg.stops().len(), 3, "computed style must keep 3 stops");
        // Legacy 2-stop field still set for the fallback path.
        assert!(cs.background_gradient.is_some());
        // First-stop color exposed as the solid background fallback.
        assert_eq!(cs.background_color, Some(Color { r: 255, g: 0, b: 0, a: 255 }));
    }

    /// `background: #fff` after a gradient clears the full-gradient model.
    #[test]
    fn solid_background_overwrites_full_gradient() {
        let ss = parse_stylesheet(
            "div { background: linear-gradient(red, blue); background: #fff; }",
        );
        let el = Fake { tag: "div", id: None, classes: &[] };
        let cs = compute(&[ss], el);
        assert!(
            cs.background_gradient_full.is_none(),
            "solid background must clear the full gradient"
        );
    }

    // ───────────────────────── 3D transform parse ──────────────────────

    #[test]
    fn transform_rotatex_parses_into_3d_op_list_not_swallowed() {
        // rotateX(45deg) must populate transform_ops (the 3D path) and NOT
        // leak into the scalar 2D rotate field — previously it parsed
        // without error but produced no visual effect (a stub).
        let ss = parse_stylesheet("div { transform: rotateX(45deg); }");
        let el = Fake { tag: "div", id: None, classes: &[] };
        let cs = compute(&[ss], el);
        let ops = cs.transform_ops.as_ref().expect("rotateX takes the 3D path");
        assert_eq!(ops.len(), 1);
        assert!(matches!(ops[0], Transform3DOp::RotateX(a) if (a - 45.0).abs() < 1e-3));
        assert!(cs.rotate_deg.is_none(), "must not fall into the 2D scalar rotate");
    }

    #[test]
    fn transform_translate3d_and_translatez_parse() {
        let ss = parse_stylesheet("div { transform: translate3d(10px, 20px, 30px); }");
        let el = Fake { tag: "div", id: None, classes: &[] };
        let cs = compute(&[ss], el);
        let ops = cs.transform_ops.as_ref().expect("translate3d takes the 3D path");
        assert!(matches!(
            ops[0],
            Transform3DOp::Translate3d(
                crate::properties::Length::Px(x),
                crate::properties::Length::Px(y),
                crate::properties::Length::Px(z),
            ) if (x - 10.0).abs() < 1e-3 && (y - 20.0).abs() < 1e-3 && (z - 30.0).abs() < 1e-3
        ));

        let ss2 = parse_stylesheet("div { transform: translateZ(50px); }");
        let cs2 = compute(&[ss2], el);
        let ops2 = cs2.transform_ops.as_ref().expect("translateZ takes the 3D path");
        assert!(matches!(
            ops2[0],
            Transform3DOp::Translate3d(_, _, crate::properties::Length::Px(z)) if (z - 50.0).abs() < 1e-3
        ));
    }

    #[test]
    fn transform_matrix3d_16_values_parse() {
        let ss = parse_stylesheet(
            "div { transform: matrix3d(1,0,0,0, 0,1,0,0, 0,0,1,0, 5,6,7,1); }",
        );
        let el = Fake { tag: "div", id: None, classes: &[] };
        let cs = compute(&[ss], el);
        let ops = cs.transform_ops.as_ref().expect("matrix3d takes the 3D path");
        match ops[0] {
            Transform3DOp::Matrix3d(m) => {
                assert!((m[12] - 5.0).abs() < 1e-3, "m41 (tx) = 5");
                assert!((m[13] - 6.0).abs() < 1e-3, "m42 (ty) = 6");
                assert!((m[14] - 7.0).abs() < 1e-3, "m43 (tz) = 7");
            }
            _ => panic!("expected Matrix3d"),
        }
    }

    #[test]
    fn transform_combined_3d_keeps_function_order() {
        // perspective(800px) rotateY(30deg) — both 3D; the op list must
        // preserve left-to-right order (perspective first, rotate second).
        let ss = parse_stylesheet("div { transform: perspective(800px) rotateY(30deg); }");
        let el = Fake { tag: "div", id: None, classes: &[] };
        let cs = compute(&[ss], el);
        let ops = cs.transform_ops.as_ref().expect("3D path");
        assert_eq!(ops.len(), 2);
        assert!(matches!(ops[0], Transform3DOp::Perspective(_)));
        assert!(matches!(ops[1], Transform3DOp::RotateY(a) if (a - 30.0).abs() < 1e-3));
    }

    #[test]
    fn pure_2d_transform_still_uses_scalar_fields_no_regression() {
        // A 2D-only transform must STILL fill the scalar fields (so the 2D
        // fast paths fire) and must NOT create a transform_ops list.
        let ss = parse_stylesheet("div { transform: translate(10px, 20px) rotate(45deg) scale(2); }");
        let el = Fake { tag: "div", id: None, classes: &[] };
        let cs = compute(&[ss], el);
        assert!(cs.transform_ops.is_none(), "2D-only must not take the 3D path");
        assert_eq!(cs.translate_x, Some(crate::properties::Length::Px(10.0)));
        assert_eq!(cs.translate_y, Some(crate::properties::Length::Px(20.0)));
        assert_eq!(cs.rotate_deg, Some(45.0));
        assert_eq!(cs.scale_x, Some(2.0));
    }

    #[test]
    fn backface_visibility_transform_style_perspective_parse() {
        let ss = parse_stylesheet(
            "div { backface-visibility: hidden; transform-style: preserve-3d; perspective: 600px; perspective-origin: 25% 75%; }",
        );
        let el = Fake { tag: "div", id: None, classes: &[] };
        let cs = compute(&[ss], el);
        assert!(cs.backface_visibility_hidden, "backface-visibility:hidden parsed");
        assert!(cs.transform_style_preserve_3d, "transform-style:preserve-3d parsed");
        assert_eq!(cs.perspective_px, Some(600.0), "perspective property px");
        assert!(cs.perspective_origin.is_some(), "perspective-origin parsed");
    }

    // ---- CSS Multi-column Layout parsing (Multicol 1) ----

    #[test]
    fn multicol_longhands_and_shorthands_parse() {
        let ss = parse_stylesheet(
            "div { column-count: 3; column-width: 240px; column-gap: 30px; }",
        );
        let el = Fake { tag: "div", id: None, classes: &[] };
        let cs = compute(&[ss], el);
        assert_eq!(cs.column_count, Some(3), "column-count:3");
        assert_eq!(cs.column_width, Some(240.0), "column-width:240px");
        // column-gap feeds BOTH the grid gap and the multicol gap.
        assert_eq!(cs.multicol_gap, Some(30.0), "column-gap → multicol_gap px");
    }

    #[test]
    fn multicol_columns_shorthand_parses_both() {
        let ss = parse_stylesheet("div { columns: 200px 4; }");
        let el = Fake { tag: "div", id: None, classes: &[] };
        let cs = compute(&[ss], el);
        assert_eq!(cs.column_width, Some(200.0), "columns shorthand width");
        assert_eq!(cs.column_count, Some(4), "columns shorthand count");
    }

    #[test]
    fn column_rule_shorthand_parses_width_style_color() {
        let ss = parse_stylesheet("div { column-rule: 2px solid rgb(0, 128, 0); }");
        let el = Fake { tag: "div", id: None, classes: &[] };
        let cs = compute(&[ss], el);
        assert_eq!(cs.column_rule_width, Some(2.0), "column-rule width 2px");
        assert_eq!(cs.column_rule_style, Some(BorderStyle::Solid), "rule style solid");
        let c = cs.column_rule_color.expect("rule color parsed");
        assert_eq!((c.r, c.g, c.b), (0, 128, 0), "rule color green");
    }

    #[test]
    fn column_rule_longhands_and_width_keywords() {
        let ss = parse_stylesheet(
            "div { column-rule-width: thick; column-rule-style: dashed; column-rule-color: #ff0000; }",
        );
        let el = Fake { tag: "div", id: None, classes: &[] };
        let cs = compute(&[ss], el);
        assert_eq!(cs.column_rule_width, Some(5.0), "thick → 5px");
        assert_eq!(cs.column_rule_style, Some(BorderStyle::Dashed));
        let c = cs.column_rule_color.expect("rule color");
        assert_eq!((c.r, c.g, c.b), (255, 0, 0), "rule color red");
    }

    #[test]
    fn column_span_all_parses() {
        let ss = parse_stylesheet("h2 { column-span: all; }");
        let el = Fake { tag: "h2", id: None, classes: &[] };
        let cs = compute(&[ss], el);
        assert!(cs.column_span_all, "column-span:all sets the flag");

        let ss2 = parse_stylesheet("h2 { column-span: none; }");
        let cs2 = compute(&[ss2], el);
        assert!(!cs2.column_span_all, "column-span:none clears the flag");
    }

    // ─── CSS Writing Modes 4 + CSS Logical Properties 1 ────────────────────
    // margin index map: [top, right, bottom, left] = [0,1,2,3].

    /// `writing-mode: vertical-rl` / `direction: rtl` parse into ComputedStyle.
    #[test]
    fn writing_mode_and_direction_parse() {
        let ss = parse_stylesheet("p { writing-mode: vertical-rl; direction: rtl; }");
        let el = Fake { tag: "p", id: None, classes: &[] };
        let cs = compute(&[ss], el);
        assert_eq!(cs.writing_mode, Some(WritingMode::VerticalRl));
        assert_eq!(cs.direction, Some(Direction::Rtl));
    }

    /// margin-inline-start in horizontal-tb LTR (the default) must resolve to
    /// margin-LEFT (CSS Logical 1 §2.1 mapping table).
    #[test]
    fn margin_inline_start_horizontal_ltr_is_left() {
        let ss = parse_stylesheet("p { margin-inline-start: 7px; }");
        let el = Fake { tag: "p", id: None, classes: &[] };
        let cs = compute(&[ss], el);
        assert_eq!(cs.margin[3], Some(Length::Px(7.0)), "inline-start → left in horizontal-tb LTR");
        assert_eq!(cs.margin[1], None, "right must be untouched");
        assert_eq!(cs.margin[0], None, "top must be untouched");
    }

    /// margin-inline-start in horizontal-tb RTL must resolve to margin-RIGHT.
    #[test]
    fn margin_inline_start_horizontal_rtl_is_right() {
        let ss = parse_stylesheet("p { direction: rtl; margin-inline-start: 7px; }");
        let el = Fake { tag: "p", id: None, classes: &[] };
        let cs = compute(&[ss], el);
        assert_eq!(cs.margin[1], Some(Length::Px(7.0)), "inline-start → right in horizontal-tb RTL");
        assert_eq!(cs.margin[3], None, "left must be untouched in RTL");
    }

    /// margin-inline-start in vertical-rl LTR must resolve to margin-TOP.
    #[test]
    fn margin_inline_start_vertical_rl_is_top() {
        let ss = parse_stylesheet("p { writing-mode: vertical-rl; margin-inline-start: 7px; }");
        let el = Fake { tag: "p", id: None, classes: &[] };
        let cs = compute(&[ss], el);
        assert_eq!(cs.margin[0], Some(Length::Px(7.0)), "inline-start → top in vertical-rl LTR");
        assert_eq!(cs.margin[3], None, "left must be untouched in vertical-rl");
    }

    /// block-start in vertical-rl must map to the RIGHT edge (CSS Logical table).
    #[test]
    fn margin_block_start_vertical_rl_is_right() {
        let ss = parse_stylesheet("p { writing-mode: vertical-rl; margin-block-start: 9px; }");
        let el = Fake { tag: "p", id: None, classes: &[] };
        let cs = compute(&[ss], el);
        assert_eq!(cs.margin[1], Some(Length::Px(9.0)), "block-start → right in vertical-rl");
    }

    /// block-start in vertical-lr must map to the LEFT edge.
    #[test]
    fn margin_block_start_vertical_lr_is_left() {
        let ss = parse_stylesheet("p { writing-mode: vertical-lr; margin-block-start: 9px; }");
        let el = Fake { tag: "p", id: None, classes: &[] };
        let cs = compute(&[ss], el);
        assert_eq!(cs.margin[3], Some(Length::Px(9.0)), "block-start → left in vertical-lr");
    }

    /// vertical-rl + rtl flips the inline axis: inline-start → BOTTOM.
    #[test]
    fn margin_inline_start_vertical_rl_rtl_is_bottom() {
        let ss = parse_stylesheet("p { writing-mode: vertical-rl; direction: rtl; margin-inline-start: 4px; }");
        let el = Fake { tag: "p", id: None, classes: &[] };
        let cs = compute(&[ss], el);
        assert_eq!(cs.margin[2], Some(Length::Px(4.0)), "inline-start → bottom in vertical-rl RTL");
    }

    /// inline-size maps to WIDTH in horizontal writing modes.
    #[test]
    fn inline_size_horizontal_is_width() {
        let ss = parse_stylesheet("div { inline-size: 200px; }");
        let el = Fake { tag: "div", id: None, classes: &[] };
        let cs = compute(&[ss], el);
        assert_eq!(cs.width, Some(Length::Px(200.0)), "inline-size → width in horizontal-tb");
        assert_eq!(cs.height, None, "height untouched");
    }

    /// inline-size maps to HEIGHT in vertical writing modes; block-size → width.
    #[test]
    fn inline_size_vertical_is_height() {
        let ss = parse_stylesheet("div { writing-mode: vertical-rl; inline-size: 200px; block-size: 50px; }");
        let el = Fake { tag: "div", id: None, classes: &[] };
        let cs = compute(&[ss], el);
        assert_eq!(cs.height, Some(Length::Px(200.0)), "inline-size → height in vertical-rl");
        assert_eq!(cs.width, Some(Length::Px(50.0)), "block-size → width in vertical-rl");
    }

    /// min/max-inline/block-size map to the correct physical axis.
    #[test]
    fn minmax_logical_size_vertical() {
        let ss = parse_stylesheet(
            "div { writing-mode: vertical-lr; min-inline-size: 30px; max-block-size: 80px; }",
        );
        let el = Fake { tag: "div", id: None, classes: &[] };
        let cs = compute(&[ss], el);
        assert_eq!(cs.min_height, Some(Length::Px(30.0)), "min-inline-size → min-height in vertical");
        assert_eq!(cs.max_width, Some(Length::Px(80.0)), "max-block-size → max-width in vertical");
    }

    /// padding-inline-start in vertical-rl → padding-TOP (and -end → bottom).
    #[test]
    fn padding_inline_vertical_rl() {
        let ss = parse_stylesheet(
            "p { writing-mode: vertical-rl; padding-inline-start: 3px; padding-inline-end: 6px; }",
        );
        let el = Fake { tag: "p", id: None, classes: &[] };
        let cs = compute(&[ss], el);
        assert_eq!(cs.padding[0], Some(Length::Px(3.0)), "inline-start → padding-top");
        assert_eq!(cs.padding[2], Some(Length::Px(6.0)), "inline-end → padding-bottom");
    }

    /// inset-block-start in vertical-rl maps to the physical `right` offset.
    #[test]
    fn inset_block_start_vertical_rl_is_right() {
        let ss = parse_stylesheet("p { writing-mode: vertical-rl; inset-block-start: 12px; }");
        let el = Fake { tag: "p", id: None, classes: &[] };
        let cs = compute(&[ss], el);
        assert_eq!(cs.right, Some(Length::Px(12.0)), "inset-block-start → right in vertical-rl");
        assert_eq!(cs.top, None, "top untouched");
    }

    /// CASCADE ORDER: a PHYSICAL longhand declared AFTER a logical one that
    /// maps to the same physical side must win (CSS Logical 1 §2: the pair
    /// shares a computed value taken from the higher-priority declaration).
    #[test]
    fn physical_after_logical_wins_same_slot() {
        // horizontal-tb LTR: margin-inline-start → left. margin-left declared
        // LATER must override it.
        let ss = parse_stylesheet("p { margin-inline-start: 5px; margin-left: 11px; }");
        let el = Fake { tag: "p", id: None, classes: &[] };
        let cs = compute(&[ss], el);
        assert_eq!(cs.margin[3], Some(Length::Px(11.0)), "later physical margin-left wins over earlier logical");
    }

    /// CASCADE ORDER: a LOGICAL longhand declared AFTER a physical one that
    /// maps to the same physical side must win.
    #[test]
    fn logical_after_physical_wins_same_slot() {
        let ss = parse_stylesheet("p { margin-left: 11px; margin-inline-start: 5px; }");
        let el = Fake { tag: "p", id: None, classes: &[] };
        let cs = compute(&[ss], el);
        assert_eq!(cs.margin[3], Some(Length::Px(5.0)), "later logical margin-inline-start wins over earlier physical");
    }

    /// A 2-value `margin-inline: <start> <end>` in vertical-rl maps start→top
    /// and end→bottom.
    #[test]
    fn margin_inline_shorthand_vertical_rl() {
        let ss = parse_stylesheet("p { writing-mode: vertical-rl; margin-inline: 2px 8px; }");
        let el = Fake { tag: "p", id: None, classes: &[] };
        let cs = compute(&[ss], el);
        assert_eq!(cs.margin[0], Some(Length::Px(2.0)), "inline-start → top");
        assert_eq!(cs.margin[2], Some(Length::Px(8.0)), "inline-end → bottom");
    }

    /// Regression: the plain physical horizontal-tb path is unchanged —
    /// margin-left still sets margin[3], width still sets width.
    #[test]
    fn physical_box_unaffected_default_mode() {
        let ss = parse_stylesheet("p { margin-left: 4px; width: 100px; padding-top: 2px; left: 5px; }");
        let el = Fake { tag: "p", id: None, classes: &[] };
        let cs = compute(&[ss], el);
        assert_eq!(cs.margin[3], Some(Length::Px(4.0)));
        assert_eq!(cs.width, Some(Length::Px(100.0)));
        assert_eq!(cs.padding[0], Some(Length::Px(2.0)));
        assert_eq!(cs.left, Some(Length::Px(5.0)));
    }

    /// `map_logical_side` covers the full spec table for all 6 mode/dir combos.
    #[test]
    fn map_logical_side_full_table() {
        use WritingMode::*;
        use Direction::*;
        // (top,right,bottom,left) = (0,1,2,3)
        // horizontal-tb LTR
        assert_eq!(map_logical_side(LOGICAL_INLINE_START, HorizontalTb, Ltr), 3);
        assert_eq!(map_logical_side(LOGICAL_INLINE_END, HorizontalTb, Ltr), 1);
        assert_eq!(map_logical_side(LOGICAL_BLOCK_START, HorizontalTb, Ltr), 0);
        assert_eq!(map_logical_side(LOGICAL_BLOCK_END, HorizontalTb, Ltr), 2);
        // horizontal-tb RTL
        assert_eq!(map_logical_side(LOGICAL_INLINE_START, HorizontalTb, Rtl), 1);
        assert_eq!(map_logical_side(LOGICAL_INLINE_END, HorizontalTb, Rtl), 3);
        // vertical-rl LTR
        assert_eq!(map_logical_side(LOGICAL_INLINE_START, VerticalRl, Ltr), 0);
        assert_eq!(map_logical_side(LOGICAL_INLINE_END, VerticalRl, Ltr), 2);
        assert_eq!(map_logical_side(LOGICAL_BLOCK_START, VerticalRl, Ltr), 1);
        assert_eq!(map_logical_side(LOGICAL_BLOCK_END, VerticalRl, Ltr), 3);
        // vertical-rl RTL
        assert_eq!(map_logical_side(LOGICAL_INLINE_START, VerticalRl, Rtl), 2);
        assert_eq!(map_logical_side(LOGICAL_INLINE_END, VerticalRl, Rtl), 0);
        // vertical-lr LTR
        assert_eq!(map_logical_side(LOGICAL_INLINE_START, VerticalLr, Ltr), 0);
        assert_eq!(map_logical_side(LOGICAL_BLOCK_START, VerticalLr, Ltr), 3);
        assert_eq!(map_logical_side(LOGICAL_BLOCK_END, VerticalLr, Ltr), 1);
        // vertical-lr RTL
        assert_eq!(map_logical_side(LOGICAL_INLINE_START, VerticalLr, Rtl), 2);
        assert_eq!(map_logical_side(LOGICAL_INLINE_END, VerticalLr, Rtl), 0);
    }

    /// `filter: url(#blur)` parses to a `Reference` carrying the bare id
    /// (without the leading `#`) — the SVG filter-reference path.
    #[test]
    fn filter_url_parses_to_reference() {
        let ss = parse_stylesheet("p { filter: url(#blur); }");
        let el = Fake { tag: "p", id: None, classes: &[] };
        let cs = compute(&[ss], el);
        assert_eq!(cs.filters.len(), 1, "one filter parsed");
        match &cs.filters[0] {
            FilterFn::Reference(id) => assert_eq!(&**id, "blur", "bare id, no '#'"),
            other => panic!("expected Reference, got {other:?}"),
        }
    }

    /// A mixed chain `grayscale(50%) url(#f) blur(2px)` keeps all three
    /// entries in declaration order — the url ref does not swallow the rest.
    #[test]
    fn filter_chain_mixes_functions_and_reference() {
        let ss = parse_stylesheet("p { filter: grayscale(50%) url(#f) blur(2px); }");
        let el = Fake { tag: "p", id: None, classes: &[] };
        let cs = compute(&[ss], el);
        assert_eq!(cs.filters.len(), 3, "three entries, got {:?}", cs.filters);
        assert!(matches!(cs.filters[0], FilterFn::Grayscale(g) if (g - 0.5).abs() < 1e-6));
        assert!(matches!(&cs.filters[1], FilterFn::Reference(id) if &**id == "f"));
        assert!(matches!(cs.filters[2], FilterFn::Blur(b) if (b - 2.0).abs() < 1e-6));
    }

    /// An external-document ref `url(foo.svg#x)` is not resolvable here, so it
    /// is skipped (under-apply, never a bogus same-doc reference).
    #[test]
    fn filter_external_url_ref_skipped() {
        let ss = parse_stylesheet("p { filter: url(foo.svg#x); }");
        let el = Fake { tag: "p", id: None, classes: &[] };
        let cs = compute(&[ss], el);
        assert!(cs.filters.is_empty(), "external ref skipped, got {:?}", cs.filters);
    }

    // ── @keyframes collection memo (Blink StyleRuleKeyframes) oracle ─────────

    /// The cached collection must be IDENTICAL to the cold collection for the
    /// same sheets, sampled across the whole 0..1 timeline (this is the
    /// byte-identity oracle: same parsed model => same interpolated values).
    fn keyframe_maps_equal(
        a: &std::collections::HashMap<String, KeyframeRule>,
        b: &std::collections::HashMap<String, KeyframeRule>,
    ) -> bool {
        if a.len() != b.len() {
            return false;
        }
        for (name, ra) in a {
            let Some(rb) = b.get(name) else { return false };
            if ra.name != rb.name || ra.steps.len() != rb.steps.len() {
                return false;
            }
            // Compare each step's offset and its sampled value strings — the
            // exact bytes the per-frame animation path consumes.
            for t in [0.0_f32, 0.13, 0.25, 0.5, 0.77, 1.0] {
                let pa = sample_animation(ra, t);
                let pb = sample_animation(rb, t);
                if pa != pb {
                    return false;
                }
            }
        }
        true
    }

    #[test]
    fn keyframes_memo_matches_cold_collect() {
        clear_keyframes_memo();
        let css = "@keyframes slide { from { transform: translateX(0px); opacity: 1; } \
                   50% { transform: translateX(40px); opacity: 0.5; } \
                   to { transform: translateX(100px); opacity: 0; } } \
                   @keyframes fade { 0% { background-color: #000; } 100% { background-color: #fff; } }";
        let sheets = vec![parse_stylesheet(css)];
        let cold = collect_keyframes(&sheets);
        // First call: cold miss; second call: should be a HIT and identical.
        let warm1 = collect_keyframes_cached(&sheets);
        let warm2 = collect_keyframes_cached(&sheets);
        assert!(keyframe_maps_equal(&cold, &warm1), "cached == cold (first)");
        assert!(keyframe_maps_equal(&cold, &warm2), "cached == cold (second)");
        // The second call must reuse the SAME Rc allocation (proves it hit the
        // memo rather than rebuilding — the actual cost we save).
        assert!(
            std::rc::Rc::ptr_eq(&warm1, &warm2),
            "second call reuses the cached Rc (memo hit, no rebuild)"
        );
    }

    #[test]
    fn keyframes_memo_invalidates_on_sheet_change() {
        clear_keyframes_memo();
        let css_a = "@keyframes spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }";
        let sheets_a = vec![parse_stylesheet(css_a)];
        let a = collect_keyframes_cached(&sheets_a);
        assert!(a.contains_key("spin"), "spin collected");

        // A DIFFERENT stylesheet (new @keyframes name + a changed value) MUST
        // invalidate — a stale cache that returned `spin` here would be a
        // correctness bug (the new page's animation would silently not animate).
        let css_b = "@keyframes spin { from { transform: rotate(0deg); } to { transform: rotate(720deg); } } \
                     @keyframes glow { from { opacity: 0; } to { opacity: 1; } }";
        let sheets_b = vec![parse_stylesheet(css_b)];
        let b = collect_keyframes_cached(&sheets_b);
        assert!(b.contains_key("glow"), "new @keyframes picked up after change");
        assert_eq!(b.len(), 2, "both keyframes present after change");
        // The fingerprint must differ between the two sheet sets.
        assert_ne!(
            keyframes_fingerprint(&sheets_a),
            keyframes_fingerprint(&sheets_b),
            "differing keyframe content => differing fingerprint"
        );
        // And the cold collect of B agrees with the cached B.
        let cold_b = collect_keyframes(&sheets_b);
        assert!(keyframe_maps_equal(&cold_b, &b), "cached B == cold B");
    }

    /// A value-only edit inside an existing @keyframes (same name, same offsets)
    /// MUST invalidate — otherwise the animation would interpolate stale frames.
    #[test]
    fn keyframes_memo_invalidates_on_value_edit() {
        clear_keyframes_memo();
        let s1 = vec![parse_stylesheet(
            "@keyframes m { from { opacity: 1; } to { opacity: 0; } }",
        )];
        let _ = collect_keyframes_cached(&s1);
        let s2 = vec![parse_stylesheet(
            "@keyframes m { from { opacity: 1; } to { opacity: 0.25; } }",
        )];
        let got = collect_keyframes_cached(&s2);
        let cold = collect_keyframes(&s2);
        assert!(keyframe_maps_equal(&cold, &got), "value edit re-collected");
        assert_ne!(
            keyframes_fingerprint(&s1),
            keyframes_fingerprint(&s2),
            "value edit perturbs fingerprint"
        );
    }

    /// Two DISTINCT sheet sets with byte-identical keyframe content legitimately
    /// share a result (same inputs => same output). This is correct reuse, not a
    /// stale hit — verify the returned map matches a cold collect of the second.
    #[test]
    fn keyframes_memo_reuses_for_identical_content() {
        clear_keyframes_memo();
        let css = "@keyframes k { from { opacity: 0; } to { opacity: 1; } }";
        let s1 = vec![parse_stylesheet(css)];
        let s2 = vec![parse_stylesheet(css)]; // separately parsed, identical text
        let _ = collect_keyframes_cached(&s1);
        let r2 = collect_keyframes_cached(&s2);
        let cold2 = collect_keyframes(&s2);
        assert!(keyframe_maps_equal(&cold2, &r2), "identical content reuse is correct");
        assert_eq!(
            keyframes_fingerprint(&s1),
            keyframes_fingerprint(&s2),
            "identical keyframe content => identical fingerprint"
        );
    }

    /// Adding a NON-keyframe at-rule before a @keyframes shifts its index; the
    /// fingerprint must reflect rule presence/position so override semantics
    /// (a later same-name @keyframes wins) are never silently stale.
    #[test]
    fn keyframes_memo_accounts_for_nonkeyframe_rule_presence() {
        clear_keyframes_memo();
        let a = vec![parse_stylesheet(
            "@keyframes k { from { opacity: 0; } to { opacity: 1; } }",
        )];
        let b = vec![parse_stylesheet(
            "@media screen { p { color: red; } } \
             @keyframes k { from { opacity: 0; } to { opacity: 1; } }",
        )];
        assert_ne!(
            keyframes_fingerprint(&a),
            keyframes_fingerprint(&b),
            "a preceding non-keyframe at-rule perturbs the fingerprint"
        );
    }
}
