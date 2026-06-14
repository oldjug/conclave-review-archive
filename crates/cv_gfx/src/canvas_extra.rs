//! Canvas2D extras — clip, drawImage, getImageData, gradients, dash,
//! shadow. These plug into the existing CanvasContext2D so V1 scripts
//! that use the methods round-trip without throwing.
//!
//! The geometry stays in software (Bitmap-backed); GPU acceleration is
//! a future swap behind the same surface.

use crate::Color;

#[derive(Debug, Clone)]
pub struct GradientStop {
    pub offset: f32,
    pub color: Color,
}

#[derive(Debug, Clone)]
pub enum GradientKind {
    Linear { x0: f32, y0: f32, x1: f32, y1: f32 },
    Radial { cx: f32, cy: f32, r0: f32, r1: f32 },
    Conic { cx: f32, cy: f32, start_angle: f32 },
}

#[derive(Debug, Clone)]
pub struct Gradient {
    pub kind: GradientKind,
    pub stops: Vec<GradientStop>,
}

impl Gradient {
    pub fn linear(x0: f32, y0: f32, x1: f32, y1: f32) -> Self {
        Self {
            kind: GradientKind::Linear { x0, y0, x1, y1 },
            stops: Vec::new(),
        }
    }
    pub fn radial(cx: f32, cy: f32, r0: f32, r1: f32) -> Self {
        Self {
            kind: GradientKind::Radial { cx, cy, r0, r1 },
            stops: Vec::new(),
        }
    }
    pub fn conic(cx: f32, cy: f32, start_angle: f32) -> Self {
        Self {
            kind: GradientKind::Conic {
                cx,
                cy,
                start_angle,
            },
            stops: Vec::new(),
        }
    }
    pub fn add_stop(&mut self, offset: f32, color: Color) {
        self.stops.push(GradientStop { offset, color });
    }
    /// Sample the gradient at offset t∈[0,1] → color via linear stop
    /// interpolation.
    pub fn sample(&self, t: f32) -> Color {
        if self.stops.is_empty() {
            return Color {
                r: 0,
                g: 0,
                b: 0,
                a: 0,
            };
        }
        let mut prev = &self.stops[0];
        for s in self.stops.iter() {
            if t <= s.offset {
                if s.offset == prev.offset {
                    return s.color;
                }
                let k = ((t - prev.offset) / (s.offset - prev.offset)).clamp(0.0, 1.0);
                return Color {
                    r: lerp_u8(prev.color.r, s.color.r, k),
                    g: lerp_u8(prev.color.g, s.color.g, k),
                    b: lerp_u8(prev.color.b, s.color.b, k),
                    a: lerp_u8(prev.color.a, s.color.a, k),
                };
            }
            prev = s;
        }
        prev.color
    }
}

fn lerp_u8(a: u8, b: u8, t: f32) -> u8 {
    (a as f32 + (b as f32 - a as f32) * t).round() as u8
}

#[derive(Debug, Clone)]
pub struct Shadow {
    pub offset_x: f32,
    pub offset_y: f32,
    pub blur: f32,
    pub color: Color,
}

#[derive(Debug, Clone)]
pub struct DashPattern {
    pub segments: Vec<f32>,
    pub offset: f32,
}

#[derive(Debug, Clone)]
pub struct ImageData {
    pub width: u32,
    pub height: u32,
    pub data: Vec<u8>,
}
