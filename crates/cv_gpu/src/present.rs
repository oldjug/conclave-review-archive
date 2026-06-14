//! Swap-chain present interface.
//!
//! `SwapChain` is the type the compositor pushes per-frame work to.
//! V1 owns the back-buffer geometry and a software composite path so
//! the present API works on machines without a D3D11-capable adapter
//! (CI, headless test runs); the production backend swaps in the
//! DXGI flip-model + DirectComposition visual tree.

#[derive(Debug, Clone, Copy)]
pub struct PresentConfig {
    pub width: u32,
    pub height: u32,
    /// Number of back buffers in the swap chain — typically 2 for
    /// flip-discard, 3 for FlipSequential when smoothness matters
    /// more than memory.
    pub buffer_count: u8,
    /// When `true`, request an HDR-capable swap chain (R10G10B10A2 or
    /// R16G16B16A16_FLOAT format). Falls back transparently to SDR.
    pub want_hdr: bool,
}

impl Default for PresentConfig {
    fn default() -> Self {
        Self {
            width: 1280,
            height: 720,
            buffer_count: 2,
            want_hdr: false,
        }
    }
}

/// Errors from the swap chain. Boundary type; the real DXGI HRESULTs
/// are folded into these variants by the platform backend.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SwapChainError {
    NoAdapter,
    DeviceLost,
    ResizeFailed,
    CompositionFailed,
}

/// One layer's worth of pixel input to the present pipeline.
#[derive(Debug, Clone)]
pub struct PresentLayer<'a> {
    pub id: u32,
    pub bgra: &'a [u8], // length = width * height * 4
    pub width: u32,
    pub height: u32,
    /// Pixel-space placement on the back buffer.
    pub x: i32,
    pub y: i32,
    /// 0..1 opacity multiplier applied at present time.
    pub opacity: f32,
}

/// Per-frame present description — the back-buffer size plus the
/// fully-stacked layer list.
#[derive(Debug, Clone)]
pub struct PresentDescriptor<'a> {
    pub width: u32,
    pub height: u32,
    pub background: u32, // BGRA u32 clear color
    pub layers: Vec<PresentLayer<'a>>,
}

/// Present pipeline handle. V1 stores the configured back-buffer
/// geometry; `present_layers` runs the composite-and-output path,
/// which a future backend swaps for a HW path.
pub struct SwapChain {
    cfg: PresentConfig,
    /// Most-recently-presented back-buffer pixels — exposed via
    /// `last_frame_bgra()` so tests and the multi-process renderer
    /// can verify what would be displayed without owning a HWND.
    last_frame: Vec<u32>,
}

impl SwapChain {
    /// Construct a software-backed swap chain. The real DXGI device
    /// path lives on top of this — `new_for_hwnd(hwnd, cfg)` lands
    /// in the platform module.
    pub fn new_software(cfg: PresentConfig) -> Result<Self, SwapChainError> {
        Ok(Self {
            last_frame: vec![cfg_background(&cfg); (cfg.width as usize) * (cfg.height as usize)],
            cfg,
        })
    }

    pub fn config(&self) -> &PresentConfig {
        &self.cfg
    }

    /// Composite the layers into the back-buffer and "present". V1
    /// stores the result internally; an HW backend writes to the
    /// DXGI back buffer and calls `IDXGISwapChain1::Present`.
    pub fn present_layers(&mut self, desc: &PresentDescriptor<'_>) -> Result<(), SwapChainError> {
        if desc.width != self.cfg.width || desc.height != self.cfg.height {
            return Err(SwapChainError::ResizeFailed);
        }
        let n = (desc.width as usize) * (desc.height as usize);
        let mut buf = vec![desc.background; n];
        for layer in &desc.layers {
            blit_layer(&mut buf, desc.width, desc.height, layer);
        }
        self.last_frame = buf;
        Ok(())
    }

    /// Inspect what the last present put on screen. Used by the
    /// renderer-side pipe path to ship a BGRA bitmap to the browser.
    pub fn last_frame_bgra(&self) -> &[u32] {
        &self.last_frame
    }

    /// Resize the back buffer. The HW backend calls
    /// `ResizeBuffers`; the SW backend reallocates the cache.
    pub fn resize(&mut self, w: u32, h: u32) -> Result<(), SwapChainError> {
        self.cfg.width = w;
        self.cfg.height = h;
        self.last_frame = vec![cfg_background(&self.cfg); (w as usize) * (h as usize)];
        Ok(())
    }
}

fn cfg_background(cfg: &PresentConfig) -> u32 {
    if cfg.want_hdr {
        0xFF000000 // 0 luminance in the alpha-opaque sense
    } else {
        0xFFFFFFFF
    }
}

fn blit_layer(buf: &mut [u32], w: u32, h: u32, layer: &PresentLayer<'_>) {
    let opacity = layer.opacity.clamp(0.0, 1.0);
    if opacity <= 0.0 {
        return;
    }
    for ly in 0..layer.height as i32 {
        let dy = ly + layer.y;
        if dy < 0 || dy >= h as i32 {
            continue;
        }
        for lx in 0..layer.width as i32 {
            let dx = lx + layer.x;
            if dx < 0 || dx >= w as i32 {
                continue;
            }
            let src_off = ((ly as usize) * (layer.width as usize) + lx as usize) * 4;
            let b = layer.bgra[src_off] as u32;
            let g = layer.bgra[src_off + 1] as u32;
            let r = layer.bgra[src_off + 2] as u32;
            let a = layer.bgra[src_off + 3] as u32;
            let src = (a << 24) | (r << 16) | (g << 8) | b;
            let dst_idx = (dy as usize) * (w as usize) + dx as usize;
            buf[dst_idx] = blend(buf[dst_idx], src, opacity);
        }
    }
}

/// Source-over blend with a per-layer opacity multiplier. Operates on
/// BGRA u32 pixels (alpha in the high byte).
///
/// MUST stay byte-identical to `cv_gfx::blend_bgra` — the straight-alpha
/// (non-premultiplied) Porter-Duff source-over that is the faint-particle fix.
/// The previous form here produced PREMULTIPLIED output RGB
/// (`or = sr*sa + dr*da*(1-sa)` with no `/oa` normalize) and TRUNCATED
/// (`* 255.0 as u32`), diverging from the oracle and darkening colors composited
/// onto a transparent backing store. We now run the SAME normalize-by-`out_a`
/// + `.round()` math.
///
/// `cv_gpu` does not depend on `cv_gfx` (and must not gain that edge), so the
/// oracle body is replicated here verbatim — kept identical to the copy in
/// `cv_compositor::blend_with_opacity`. The shared unit test
/// `converged_blend_matches_straight_alpha_oracle` pins the byte-for-byte
/// equality against hand-computed oracle values.
///
/// Bridge from the oracle's `(dst, Color)` contract to this `(dst, u32, opacity)`
/// one: fold `opacity` into the source alpha (`sa = src_a/255 * opacity`) BEFORE
/// applying the formula. Early-out contract is preserved: return `dst` unchanged
/// when the effective source alpha is non-positive.
fn blend(dst: u32, src: u32, opacity: f32) -> u32 {
    // Effective source alpha in 0..1, with the per-layer opacity folded in.
    let sa = ((src >> 24) & 0xFF) as f32 / 255.0 * opacity;
    if sa <= 0.0 {
        return dst;
    }
    // Source channels in 0..255 (sRGB byte space, no gamma).
    let sr = ((src >> 16) & 0xFF) as f32;
    let sg = ((src >> 8) & 0xFF) as f32;
    let sb = (src & 0xFF) as f32;
    let da = ((dst >> 24) & 0xFF) as f32 / 255.0;
    let dr = ((dst >> 16) & 0xFF) as f32;
    let dg = ((dst >> 8) & 0xFF) as f32;
    let db = (dst & 0xFF) as f32;
    let inv = 1.0 - sa;
    let out_a = sa + da * inv;
    if out_a <= 0.0 {
        return dst;
    }
    // Straight-alpha source-over: alpha-weight the destination and normalize
    // the output RGB by the output alpha (NOT premultiplied), then round.
    let r = ((sr * sa + dr * da * inv) / out_a).round() as u32;
    let g = ((sg * sa + dg * da * inv) / out_a).round() as u32;
    let b = ((sb * sa + db * da * inv) / out_a).round() as u32;
    let a = (out_a * 255.0).round() as u32;
    (a << 24) | (r << 16) | (g << 8) | b
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_software_initializes_clear_buffer() {
        let cfg = PresentConfig {
            width: 4,
            height: 4,
            buffer_count: 2,
            want_hdr: false,
        };
        let sc = SwapChain::new_software(cfg).unwrap();
        assert_eq!(sc.last_frame_bgra().len(), 16);
        for &p in sc.last_frame_bgra() {
            assert_eq!(p, 0xFFFFFFFF);
        }
    }

    #[test]
    fn present_layers_rejects_size_mismatch() {
        let mut sc = SwapChain::new_software(PresentConfig {
            width: 4,
            height: 4,
            ..Default::default()
        })
        .unwrap();
        let desc = PresentDescriptor {
            width: 8,
            height: 8,
            background: 0,
            layers: Vec::new(),
        };
        assert_eq!(sc.present_layers(&desc), Err(SwapChainError::ResizeFailed));
    }

    #[test]
    fn present_blits_layer_to_back_buffer() {
        let mut sc = SwapChain::new_software(PresentConfig {
            width: 4,
            height: 4,
            ..Default::default()
        })
        .unwrap();
        // 2x2 red layer at (1,1), BGRA bytes per pixel.
        let pixels: Vec<u8> = (0..4)
            .flat_map(|_| [0x00, 0x00, 0xFF, 0xFF].iter().copied())
            .collect();
        let desc = PresentDescriptor {
            width: 4,
            height: 4,
            background: 0xFF000000,
            layers: vec![PresentLayer {
                id: 0,
                bgra: &pixels,
                width: 2,
                height: 2,
                x: 1,
                y: 1,
                opacity: 1.0,
            }],
        };
        sc.present_layers(&desc).unwrap();
        let f = sc.last_frame_bgra();
        assert_eq!(f[0], 0xFF000000);
        assert_eq!(f[1 * 4 + 1], 0xFFFF0000);
        assert_eq!(f[2 * 4 + 2], 0xFFFF0000);
    }

    #[test]
    fn resize_clears_back_buffer() {
        let mut sc = SwapChain::new_software(PresentConfig {
            width: 4,
            height: 4,
            ..Default::default()
        })
        .unwrap();
        sc.resize(8, 8).unwrap();
        assert_eq!(sc.last_frame_bgra().len(), 64);
        assert_eq!(sc.config().width, 8);
        assert_eq!(sc.config().height, 8);
    }

    #[test]
    fn opacity_zero_skips_blit() {
        let mut sc = SwapChain::new_software(PresentConfig {
            width: 2,
            height: 2,
            ..Default::default()
        })
        .unwrap();
        let pixels = vec![0xFFu8; 16];
        let desc = PresentDescriptor {
            width: 2,
            height: 2,
            background: 0xFF000000,
            layers: vec![PresentLayer {
                id: 0,
                bgra: &pixels,
                width: 2,
                height: 2,
                x: 0,
                y: 0,
                opacity: 0.0,
            }],
        };
        sc.present_layers(&desc).unwrap();
        for &p in sc.last_frame_bgra() {
            assert_eq!(p, 0xFF000000);
        }
    }

    /// `blend` MUST match the `cv_gfx::blend_bgra` straight-alpha oracle
    /// byte-for-byte (the faint-particle fix). We cannot depend on `cv_gfx`
    /// here, so the expected values below were produced by the oracle
    /// (straight-alpha source-over: normalize RGB by `out_a`, `.round()`),
    /// and are identical to the matching `cv_compositor` test. A regression
    /// toward the old premultiplied/truncating form would darken the
    /// gold-over-transparent case toward 0 and fail this test.
    #[test]
    fn converged_blend_matches_straight_alpha_oracle() {
        // (1) Opaque over opaque: red (FFFF0000) over blue → pure red.
        assert_eq!(blend(0xFF0000FF, 0xFFFF0000, 1.0), 0xFFFF0000, "opaque-over-opaque");

        // (2) Faint gold rgba(255,215,0, a=26) over transparent. The
        // straight-alpha oracle KEEPS the gold RGB (255,215,0) — the buggy
        // premultiplied form dragged it toward 0 (the faint-particle bug).
        assert_eq!(
            blend(0x00000000, 0x1AFFD700, 1.0),
            0x1AFFD700,
            "faint-gold-over-transparent must not darken toward 0"
        );

        // (3) Semi over semi: white (a=128) over blue (a=128).
        assert_eq!(blend(0x800000FF, 0x80FFFFFF, 1.0), 0xC0AAAAFF, "semi-over-semi");

        // (4) Per-layer opacity < 1: opaque red over opaque blue at 0.5 opacity.
        assert_eq!(
            blend(0xFF0000FF, 0xFFFF0000, 0.5),
            0xFF800080,
            "opacity=0.5 folds into source alpha"
        );

        // (5) Early-out contract: a fully-transparent source leaves dst untouched.
        assert_eq!(
            blend(0x280A141E, 0x00FF0000, 1.0),
            0x280A141E,
            "zero effective source alpha returns dst unchanged"
        );
    }
}
