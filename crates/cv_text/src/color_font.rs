//! COLR / CPAL color font rendering.
//!
//! V1 implements the COLR v0 layer-list model — each glyph maps to
//! a list of (glyph_id, palette_index) tuples that the rasterizer
//! stacks bottom-to-top into a multi-color glyph. CPAL provides the
//! palette table (one or more named palettes of sRGB colors).
//!
//! COLR v1 (paint-tree with gradients, transforms, composite modes)
//! lands in a follow-up; the data model here is the substrate it
//! extends.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PaletteColor {
    pub r: u8,
    pub g: u8,
    pub b: u8,
    pub a: u8,
}

impl PaletteColor {
    pub fn to_bgra(self) -> u32 {
        ((self.a as u32) << 24) | ((self.r as u32) << 16) | ((self.g as u32) << 8) | self.b as u32
    }
}

/// One palette from the CPAL table.
#[derive(Debug, Clone)]
pub struct Palette {
    pub colors: Vec<PaletteColor>,
}

/// CPAL table — multiple palettes for light / dark / accessibility variants.
#[derive(Debug, Default)]
pub struct Cpal {
    pub palettes: Vec<Palette>,
}

impl Cpal {
    pub fn color(&self, palette_idx: usize, color_idx: u16) -> Option<PaletteColor> {
        let palette = self.palettes.get(palette_idx)?;
        palette.colors.get(color_idx as usize).copied()
    }
}

/// One color layer of a composite glyph.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ColorLayer {
    pub glyph_id: u16,
    pub palette_color_index: u16,
}

/// COLR v0 — glyph ID → ordered layer list.
#[derive(Debug, Default)]
pub struct Colr {
    layers: std::collections::HashMap<u16, Vec<ColorLayer>>,
}

impl Colr {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn set_layers(&mut self, base_glyph: u16, layers: Vec<ColorLayer>) {
        self.layers.insert(base_glyph, layers);
    }

    pub fn layers_for(&self, glyph: u16) -> Option<&[ColorLayer]> {
        self.layers.get(&glyph).map(|v| v.as_slice())
    }

    /// Render a COLR base glyph using `palette_idx` from `cpal`.
    /// Each entry in the returned vec is `(glyph_id, bgra_color)` —
    /// the rasterizer stacks them bottom-to-top using the standard
    /// monochrome glyph rasterizer for each layer.
    pub fn render_layers(
        &self,
        base_glyph: u16,
        cpal: &Cpal,
        palette_idx: usize,
    ) -> Vec<(u16, u32)> {
        let layers = match self.layers_for(base_glyph) {
            Some(v) => v,
            None => return Vec::new(),
        };
        layers
            .iter()
            .filter_map(|l| {
                let col = cpal.color(palette_idx, l.palette_color_index)?;
                Some((l.glyph_id, col.to_bgra()))
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rgb(r: u8, g: u8, b: u8) -> PaletteColor {
        PaletteColor { r, g, b, a: 255 }
    }

    fn fixture() -> (Colr, Cpal) {
        let mut cpal = Cpal::default();
        cpal.palettes.push(Palette {
            colors: vec![rgb(255, 0, 0), rgb(0, 255, 0), rgb(0, 0, 255)],
        });
        cpal.palettes.push(Palette {
            colors: vec![rgb(0, 0, 0), rgb(255, 255, 255), rgb(128, 128, 128)],
        });
        let mut colr = Colr::new();
        colr.set_layers(
            1000, // base glyph
            vec![
                ColorLayer {
                    glyph_id: 1001,
                    palette_color_index: 0,
                },
                ColorLayer {
                    glyph_id: 1002,
                    palette_color_index: 2,
                },
            ],
        );
        (colr, cpal)
    }

    #[test]
    fn unknown_glyph_yields_no_layers() {
        let (colr, _cpal) = fixture();
        assert!(colr.layers_for(9999).is_none());
    }

    #[test]
    fn render_layers_resolves_palette_colors() {
        let (colr, cpal) = fixture();
        let layers = colr.render_layers(1000, &cpal, 0);
        assert_eq!(layers.len(), 2);
        assert_eq!(layers[0].0, 1001);
        // Red = (255, 0, 0) in BGRA u32 = 0xFFFF0000.
        assert_eq!(layers[0].1, 0xFFFF_0000);
        // Blue.
        assert_eq!(layers[1].1, 0xFF00_00FF);
    }

    #[test]
    fn swapping_palette_changes_colors() {
        let (colr, cpal) = fixture();
        let dark = colr.render_layers(1000, &cpal, 1);
        // Palette 1 index 0 = black, index 2 = grey.
        assert_eq!(dark[0].1, 0xFF00_0000);
        assert_eq!(dark[1].1, 0xFF80_8080);
    }

    #[test]
    fn palette_color_out_of_range_drops_layer() {
        let mut cpal = Cpal::default();
        cpal.palettes.push(Palette {
            colors: vec![rgb(1, 2, 3)],
        });
        let mut colr = Colr::new();
        colr.set_layers(
            42,
            vec![
                ColorLayer {
                    glyph_id: 1,
                    palette_color_index: 0,
                },
                ColorLayer {
                    glyph_id: 2,
                    palette_color_index: 99,
                },
            ],
        );
        let layers = colr.render_layers(42, &cpal, 0);
        assert_eq!(layers.len(), 1);
        assert_eq!(layers[0].0, 1);
    }
}
