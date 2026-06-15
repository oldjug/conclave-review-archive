//! `cv_gpu` — DXGI swap-chain + DirectComposition skeleton.
//!
//! V1 ships the type surfaces and platform-independent geometry
//! the compositor will hand to a present pipeline. The actual
//! `CreateDXGIFactory2` / `D3D11CreateDevice` / DComp visual tree
//! wiring requires the foreign DXGI / D3D11 type definitions; we
//! drop those in next, behind these types so the compositor side
//! can already target them.
//!
//! The split is deliberate: this crate owns the device, swap chain,
//! and per-frame back-buffer abstraction. `cv_compositor` produces
//! `LayerTree`s that fan into `SwapChain::present_layers`. The
//! present path will (in a follow-up) upload each layer's bitmap
//! into a D3D11 texture, attach it to a DComp visual, and
//! `IDCompositionDevice::Commit` to swap atomically without tearing.

#![allow(missing_debug_implementations)]

pub mod compositor_path;
pub mod d3d11;
pub mod dcomp;
pub mod dxgi;
pub(crate) mod hlsl;
pub mod hw_present;
pub mod present;
pub mod quad_draw;

pub use hw_present::{HwPresenter, HwPresentError};
pub use present::{PresentConfig, PresentDescriptor, PresentLayer, SwapChain, SwapChainError};
pub use quad_draw::{GpuQuad, QuadDrawError, QuadDrawer, QuadFill, Rgba, quad_raster_enabled};
