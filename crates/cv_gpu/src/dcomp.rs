//! DirectComposition — Win32 FFI for IDCompositionDevice/Visual/Target.
//!
//! Real `DCompositionCreateDevice` link against `dcomp.dll`, with COM
//! vtables for IDCompositionDevice + IDCompositionVisual matching
//! `dcomp.h`. Also keeps the in-memory `VisualTree` model so the
//! browser code path can mirror native visuals.

#![allow(non_snake_case, non_camel_case_types, dead_code)]

use crate::dxgi::{GUID, HRESULT, LPVOID};
use std::ffi::c_void;

// IID_IDCompositionDevice: C37EA93A-E7AA-450D-B16F-9746CB0407F3
pub const IID_IDCOMPOSITION_DEVICE: GUID = GUID {
    Data1: 0xC37EA93A,
    Data2: 0xE7AA,
    Data3: 0x450D,
    Data4: [0xB1, 0x6F, 0x97, 0x46, 0xCB, 0x04, 0x07, 0xF3],
};

#[link(name = "dcomp")]
unsafe extern "system" {
    pub fn DCompositionCreateDevice(
        dxgiDevice: *mut c_void,
        iid: *const GUID,
        dcompositionDevice: *mut LPVOID,
    ) -> HRESULT;
}

/// Create an IDCompositionDevice from an IDXGIDevice. The dxgi_device
/// pointer comes from QueryInterface on the D3D11 device.
pub fn create_device(dxgi_device: *mut c_void) -> Result<LPVOID, HRESULT> {
    let mut p: LPVOID = std::ptr::null_mut();
    let hr = unsafe { DCompositionCreateDevice(dxgi_device, &IID_IDCOMPOSITION_DEVICE, &mut p) };
    if hr < 0 || p.is_null() {
        return Err(hr);
    }
    Ok(p)
}

// ----------------- In-memory mirror of the visual tree -----------------

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AffineTransform {
    pub m11: f32,
    pub m12: f32,
    pub m21: f32,
    pub m22: f32,
    pub dx: f32,
    pub dy: f32,
}

impl AffineTransform {
    pub fn identity() -> Self {
        Self {
            m11: 1.0,
            m12: 0.0,
            m21: 0.0,
            m22: 1.0,
            dx: 0.0,
            dy: 0.0,
        }
    }

    pub fn translate(dx: f32, dy: f32) -> Self {
        Self {
            m11: 1.0,
            m12: 0.0,
            m21: 0.0,
            m22: 1.0,
            dx,
            dy,
        }
    }

    pub fn scale(sx: f32, sy: f32) -> Self {
        Self {
            m11: sx,
            m12: 0.0,
            m21: 0.0,
            m22: sy,
            dx: 0.0,
            dy: 0.0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rect {
    pub x: i32,
    pub y: i32,
    pub w: i32,
    pub h: i32,
}

#[derive(Debug, Clone)]
pub struct Visual {
    pub id: u32,
    /// Index of the parent in `VisualTree::visuals`, or None for the root.
    pub parent: Option<usize>,
    pub transform: AffineTransform,
    pub opacity: f32,
    pub clip: Option<Rect>,
    /// Reference to the content surface (the BGRA bitmap or the
    /// `IDCompositionSurface` the platform path uploads).
    pub surface_id: Option<u32>,
}

#[derive(Debug, Default)]
pub struct VisualTree {
    pub visuals: Vec<Visual>,
}

impl VisualTree {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a visual. Returns its index for use as a parent ref.
    pub fn add(&mut self, parent: Option<usize>) -> usize {
        let id = self.visuals.len() as u32;
        self.visuals.push(Visual {
            id,
            parent,
            transform: AffineTransform::identity(),
            opacity: 1.0,
            clip: None,
            surface_id: None,
        });
        id as usize
    }

    pub fn root(&self) -> Option<&Visual> {
        self.visuals.iter().find(|v| v.parent.is_none())
    }

    pub fn children_of(&self, parent_idx: usize) -> Vec<&Visual> {
        self.visuals
            .iter()
            .filter(|v| v.parent == Some(parent_idx))
            .collect()
    }

    pub fn set_transform(&mut self, idx: usize, t: AffineTransform) {
        self.visuals[idx].transform = t;
    }

    pub fn set_opacity(&mut self, idx: usize, o: f32) {
        self.visuals[idx].opacity = o.clamp(0.0, 1.0);
    }

    pub fn set_surface(&mut self, idx: usize, surface_id: u32) {
        self.visuals[idx].surface_id = Some(surface_id);
    }

    pub fn set_clip(&mut self, idx: usize, clip: Rect) {
        self.visuals[idx].clip = Some(clip);
    }

    /// Commit the tree — the platform path here issues
    /// `IDCompositionDevice::Commit`. V1 just freezes the in-memory
    /// state by returning a snapshot count.
    pub fn commit(&self) -> usize {
        self.visuals.len()
    }
}

/// Build a visual tree from a `cv_compositor::LayerTree`. Each layer
/// becomes one child visual under a root visual; opacity and
/// translate transfer 1:1.
pub fn visual_tree_from_layers(layers: &[crate::present::PresentLayer<'_>]) -> VisualTree {
    let mut tree = VisualTree::new();
    let root = tree.add(None);
    for layer in layers {
        let v = tree.add(Some(root));
        tree.set_transform(
            v,
            AffineTransform::translate(layer.x as f32, layer.y as f32),
        );
        tree.set_opacity(v, layer.opacity);
        tree.set_surface(v, layer.id);
    }
    tree
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::present::PresentLayer;

    #[test]
    fn empty_tree_has_no_root() {
        let t = VisualTree::new();
        assert!(t.root().is_none());
    }

    #[test]
    fn add_returns_increasing_indices() {
        let mut t = VisualTree::new();
        let a = t.add(None);
        let b = t.add(Some(a));
        let c = t.add(Some(a));
        assert_eq!(a, 0);
        assert_eq!(b, 1);
        assert_eq!(c, 2);
        assert_eq!(t.children_of(a).len(), 2);
    }

    #[test]
    fn opacity_is_clamped() {
        let mut t = VisualTree::new();
        let v = t.add(None);
        t.set_opacity(v, 5.0);
        assert_eq!(t.visuals[v].opacity, 1.0);
        t.set_opacity(v, -0.5);
        assert_eq!(t.visuals[v].opacity, 0.0);
    }

    #[test]
    fn commit_returns_visual_count() {
        let mut t = VisualTree::new();
        t.add(None);
        t.add(Some(0));
        t.add(Some(0));
        assert_eq!(t.commit(), 3);
    }

    #[test]
    fn visual_tree_from_layers_mirrors_structure() {
        let pixels = vec![0u8; 4];
        let layers = vec![
            PresentLayer {
                id: 7,
                bgra: &pixels,
                width: 1,
                height: 1,
                x: 5,
                y: 10,
                opacity: 0.5,
            },
            PresentLayer {
                id: 8,
                bgra: &pixels,
                width: 1,
                height: 1,
                x: 0,
                y: 0,
                opacity: 1.0,
            },
        ];
        let t = visual_tree_from_layers(&layers);
        // 1 root + 2 leaf = 3 visuals.
        assert_eq!(t.visuals.len(), 3);
        let v = &t.visuals[1];
        assert_eq!(v.transform.dx, 5.0);
        assert_eq!(v.transform.dy, 10.0);
        assert_eq!(v.opacity, 0.5);
        assert_eq!(v.surface_id, Some(7));
    }

    #[test]
    fn affine_helpers() {
        let t = AffineTransform::translate(3.0, 4.0);
        assert_eq!(t.dx, 3.0);
        assert_eq!(t.dy, 4.0);
        let s = AffineTransform::scale(2.0, 3.0);
        assert_eq!(s.m11, 2.0);
        assert_eq!(s.m22, 3.0);
    }
}
