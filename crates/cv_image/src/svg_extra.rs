//! SVG extras — text/tspan/use/defs/mask/clipPath/filter/animate
//! parsing surface.
//!
//! The existing SVG rasterizer (`svg.rs`) handles path/rect/circle/
//! polygon + linear gradients. This module surfaces the additional
//! element shapes the layout tree needs to traverse: text runs with
//! tspan children, use references to defs, masks, clip paths, filter
//! primitives, and animate elements. Each is a typed Rust struct the
//! upstream rasterizer can dispatch on.

#[derive(Debug, Clone, Default)]
pub struct SvgText {
    pub x: f32,
    pub y: f32,
    pub font_size: f32,
    pub fill_color: u32,
    pub text: String,
    pub tspans: Vec<SvgTspan>,
}

#[derive(Debug, Clone, Default)]
pub struct SvgTspan {
    pub dx: f32,
    pub dy: f32,
    pub text: String,
}

#[derive(Debug, Clone)]
pub struct SvgUse {
    pub href: String,
    pub x: f32,
    pub y: f32,
}

#[derive(Debug, Clone, Default)]
pub struct SvgDefs {
    pub items: Vec<(String, SvgDefItem)>,
}

#[derive(Debug, Clone)]
pub enum SvgDefItem {
    Mask(SvgMask),
    ClipPath(SvgClipPath),
    Filter(SvgFilter),
    LinearGradient(SvgLinearGradient),
    RadialGradient(SvgRadialGradient),
    Pattern(SvgPattern),
    Symbol(SvgSymbol),
}

#[derive(Debug, Clone, Default)]
pub struct SvgMask {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}

#[derive(Debug, Clone, Default)]
pub struct SvgClipPath {
    pub path_d: String,
}

#[derive(Debug, Clone, Default)]
pub struct SvgFilter {
    pub primitives: Vec<SvgFilterPrim>,
}

#[derive(Debug, Clone)]
pub enum SvgFilterPrim {
    GaussianBlur { std_dev: f32 },
    Offset { dx: f32, dy: f32 },
    ColorMatrix { values: Vec<f32> },
    Composite { op: String },
    Merge { nodes: Vec<String> },
}

#[derive(Debug, Clone, Default)]
pub struct SvgLinearGradient {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
    pub stops: Vec<(f32, u32)>,
}

#[derive(Debug, Clone, Default)]
pub struct SvgRadialGradient {
    pub cx: f32,
    pub cy: f32,
    pub r: f32,
    pub stops: Vec<(f32, u32)>,
}

#[derive(Debug, Clone, Default)]
pub struct SvgPattern {
    pub width: f32,
    pub height: f32,
}

#[derive(Debug, Clone, Default)]
pub struct SvgSymbol {
    pub viewbox: Option<(f32, f32, f32, f32)>,
    pub children_xml: String,
}

#[derive(Debug, Clone, Default)]
pub struct SvgAnimate {
    pub attribute_name: String,
    pub from: String,
    pub to: String,
    pub dur_s: f32,
    pub repeat_count: Option<u32>,
}
