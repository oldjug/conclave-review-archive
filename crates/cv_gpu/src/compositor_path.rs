//! Hardware-accelerated compositor path — wires
//! `cv_compositor` tiles through a D3D11 texture upload + DComp visual
//! tree to the swap chain.
//!
//! The D3D11 device + DComp swap chain are already built in `cv_gpu`.
//! This layer exposes the per-frame submission API: upload N tile
//! bitmaps to GPU textures, attach each to a DComp visual at its
//! position, and present.

#[derive(Debug, Clone)]
pub struct TileUpload {
    pub layer_id: u32,
    pub x: i32,
    pub y: i32,
    pub width: u32,
    pub height: u32,
    pub bgra: Vec<u32>,
}

#[derive(Debug, Default)]
pub struct FrameSubmission {
    pub tiles: Vec<TileUpload>,
    pub root_transform: [f32; 16],
}

impl FrameSubmission {
    pub fn new() -> Self {
        Self {
            tiles: Vec::new(),
            root_transform: identity_matrix(),
        }
    }
    pub fn add_tile(&mut self, t: TileUpload) {
        self.tiles.push(t);
    }
}

fn identity_matrix() -> [f32; 16] {
    [
        1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    ]
}
