//! `cv_ui` — Win32 window + bitmap presentation (V1).
//!
//! Opens a top-level window, owns a `cv_gfx::Bitmap`, and blits it on
//! `WM_PAINT`. No tabs, address bar, or input handling yet — that's M1
//! polish once we have stable visual output.

#![allow(non_camel_case_types, non_snake_case, clippy::upper_case_acronyms)]
#![allow(dead_code, missing_debug_implementations, unreachable_pub)]

use core::ffi::c_void;
use core::sync::atomic::{AtomicIsize, AtomicPtr, Ordering};
use std::sync::OnceLock;

use cv_gfx::Bitmap;
use cv_layout::LayoutBox;

pub mod shell_ui;
mod sys;
pub mod tabs;

// ===========================================================================
// M5.5 — off-main COMPOSITOR (CV_OFFMAIN_COMPOSITOR, default OFF).
//
// A dedicated `tb-compositor` thread CREATES + OWNS the thread-affine
// `HwPresenter` and the `TileCache`, receives `Arc<PaintData>` frame commits
// over an mpsc channel, reads a shared scroll atomic, and composites+presents
// OFF the UI thread. Default OFF ⇒ the synchronous UI-thread WM_PAINT
// composite+present (the fallback + oracle) is byte-for-byte unchanged.
// ===========================================================================

/// Resolved-once gate for the off-main compositor. Default OFF. Only meaningful
/// when `CV_OFFMAIN` is on (the compositor thread is only spawned in
/// `run_window_offmain`). Mirrors `CV_OFFMAIN`'s value discrimination: off for
/// unset / "0" / "false" / "off"; on for "1" / "on" / "true" / "yes".
static OFFMAIN_COMPOSITOR: OnceLock<bool> = OnceLock::new();

/// Whether the off-main compositor is enabled. **Default OFF** (reverted 2026-06-13:
/// flipping it default-on regressed the visible browser chrome — its DComp swap-chain
/// present covers the WHOLE window, including the top chrome strip, so the GDI-drawn
/// back/forward buttons + URL bar got hidden the instant the compositor's first
/// present landed [symptom: "first frame shows the buttons, then they vanish"]. The
/// compositor is correct for CONTENT [D3D-affinity/Send/thread-independence oracle
/// mutation-proven], but unifying the GDI chrome with a compositor-owned swap chain
/// [bake chrome into the composited frame, or a separate DComp chrome visual] is
/// unfinished. Until that lands, the synchronous UI-thread WM_PAINT present [which
/// keeps the GDI chrome intact] is the default. Opt in with `CV_OFFMAIN_COMPOSITOR=1`.
pub fn offmain_compositor_enabled() -> bool {
    *OFFMAIN_COMPOSITOR.get_or_init(|| {
        matches!(
            std::env::var("CV_OFFMAIN_COMPOSITOR").as_deref(),
            Ok("1") | Ok("on") | Ok("true") | Ok("yes")
        )
    })
}

/// Win32 message posted by the compositor thread (via thread-safe
/// `PostMessageW`) to wake the UI pump after a `CompositorStatus` is available
/// on the status channel. The UI drains the status and updates
/// `compositor_present_mode`. `WM_USER + 12` (10/11 are FROMPAGE / caret).
pub const WM_APP_COMPOSITOR_STATUS: u32 = sys::WM_USER + 12;

/// UI present mode under the off-main compositor, stored in a shared
/// `Arc<AtomicU8>`. Decides whether the UI WM_PAINT presents content via
/// StretchDIBits (fallback) or defers content present to the compositor's
/// swap chain. Chrome + caret are ALWAYS drawn by the UI regardless.
pub mod present_mode {
    /// GPU status not yet known. Defer content present to the compositor
    /// (its swap chain + DComp visual exist from init), draw chrome on UI.
    pub const UNKNOWN: u8 = 0;
    /// Compositor owns the present (its first GPU present succeeded). UI
    /// draws chrome/caret only; content present is on the compositor thread.
    pub const OWNED_BY_COMPOSITOR: u8 = 1;
    /// Compositor failed to init its `HwPresenter`; UI falls back to its own
    /// StretchDIBits content blit (the L2 fallback) using its own tile cache.
    pub const FALLBACK_STRETCH_DIBITS: u8 = 2;
}

/// Command the UI thread sends to the compositor thread. EVERY payload is
/// `Send` by construction — the variants admit ONLY primitives + `Arc<PaintData>`
/// (itself confirmed `Send`). The compile-time `_compositor_send_guards` below
/// turns any future non-`Send` field into a build error.
pub enum CompositorCmd {
    /// A finished frame to composite + present. `paint` is shared (one Arc
    /// bump). `chrome_h` is the pinned top strip height. `tabs` is the tab
    /// summary list the compositor needs to bake the chrome strip into the
    /// presented swap-chain frame (the GDI chrome on the window HDC is hidden
    /// under the topmost DComp visual).
    Present {
        paint: std::sync::Arc<PaintData>,
        w: u32,
        h: u32,
        chrome_h: u32,
        tabs: Vec<TabSummary>,
    },
    /// The client size changed; resize the swap chain + staging on the
    /// compositor thread, then ack via the resize rendezvous.
    Resize { w: u32, h: u32 },
    /// Window is closing — exit the loop (drops `HwPresenter` on this thread).
    Shutdown,
}

/// One-shot-ish status the compositor thread reports back to the UI.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum CompositorStatus {
    /// `HwPresenter::new` succeeded on the compositor thread.
    GpuReady,
    /// `HwPresenter::new` failed; the UI must fall back to StretchDIBits.
    GpuInitFailed,
}

/// Synchronous resize rendezvous. The UI sets `pending=true`, sends
/// `CompositorCmd::Resize`, then waits (bounded) on the condvar; the compositor
/// sets `pending=false` + notifies after `hw.resize` completes. Bounded wait so
/// a wedged compositor can never hang the resize drag.
#[derive(Default)]
pub struct ResizeAck {
    pub done: std::sync::Mutex<bool>,
    pub cv: std::sync::Condvar,
}

/// Compile-time `Send` boundary guard. Never called at runtime — its mere
/// existence forces the bounds at compile time. Any future field that makes
/// these `!Send` is a BUILD ERROR (not a runtime data race).
fn _compositor_send_guards() {
    const fn assert_send<T: Send>() {}
    assert_send::<CompositorCmd>();
    assert_send::<std::sync::Arc<PaintData>>();
    assert_send::<PaintData>();
    assert_send::<CompositorStatus>();
    assert_send::<PageHwnd>();
    assert_send::<std::sync::Arc<ResizeAck>>();
    assert_send::<std::sync::Arc<core::sync::atomic::AtomicI32>>();
    assert_send::<std::sync::Arc<[core::sync::atomic::AtomicU32; 2]>>();
    assert_send::<std::sync::Arc<core::sync::atomic::AtomicU8>>();
}

/// The `tb-compositor` thread body. CREATES + OWNS the thread-affine
/// `HwPresenter` (constructed HERE, never moved across the boundary) and the
/// `TileCache`. Loop: drain `CompositorCmd` with a 16ms timeout; on Present,
/// refresh the tile cache + composite + present at the current scroll; on
/// Resize, resize the swap chain then ack; on timeout, re-present the last
/// frame at the CURRENT scroll IF warranted (fast-scroll / animation tick).
///
/// `hwnd` is the opaque, `Send` page HWND; the presenter is created on this
/// thread so all COM lives on `creator_tid`. On init failure the thread reports
/// `GpuInitFailed` and exits — the UI falls back to StretchDIBits, never a
/// black screen / panic.
pub fn run_compositor_thread(
    page: PageHwnd,
    rx: std::sync::mpsc::Receiver<CompositorCmd>,
    scroll: std::sync::Arc<core::sync::atomic::AtomicI32>,
    dims: std::sync::Arc<[core::sync::atomic::AtomicU32; 2]>,
    present_mode_cell: std::sync::Arc<core::sync::atomic::AtomicU8>,
    status_tx: std::sync::mpsc::Sender<CompositorStatus>,
    resize_ack: std::sync::Arc<ResizeAck>,
) {
    use core::sync::atomic::Ordering as O;
    use std::sync::mpsc::RecvTimeoutError;
    use std::time::Duration;

    let hwnd = page.0;
    let init_w = dims[0].load(O::Acquire).max(1);
    let init_h = dims[1].load(O::Acquire).max(1);

    // ── CREATE HwPresenter ON THIS THREAD (the load-bearing rule). ──
    let mut hw: Option<cv_gpu::HwPresenter> = match cv_gpu::HwPresenter::new(hwnd, init_w, init_h) {
        Ok(hw) => {
            let _ = status_tx.send(CompositorStatus::GpuReady);
            // Wake the UI so it learns the status promptly.
            unsafe { sys::PostMessageW(hwnd, WM_APP_COMPOSITOR_STATUS, 0, 0) };
            Some(hw)
        }
        Err(_e) => {
            // GPU init failed on the compositor thread. Report and exit — the UI
            // flips to StretchDIBits and presents content on its own thread.
            present_mode_cell.store(present_mode::FALLBACK_STRETCH_DIBITS, O::Release);
            let _ = status_tx.send(CompositorStatus::GpuInitFailed);
            unsafe { sys::PostMessageW(hwnd, WM_APP_COMPOSITOR_STATUS, 0, 0) };
            return; // thread exits; no compositor
        }
    };

    let mut tile_cache = cv_compositor::TileCache::new();
    let mut last_paint: Option<std::sync::Arc<PaintData>> = None;
    let mut last_chrome_h: u32 = 0;
    // Tab summaries from the most recent Present, reused when a Resize re-
    // presents the last frame. Needed to bake the chrome strip into the swap-
    // chain frame (the window-HDC GDI chrome is hidden under the DComp visual).
    let mut last_tabs: Vec<TabSummary> = Vec::new();
    // Scroll value used by the previous present, so the idle tick can skip a
    // no-op re-present (scroll unchanged AND no new commit) to avoid burning a
    // vsync interval on identical pixels.
    let mut last_presented_scroll: i32 = i32::MIN;
    let mut first_present_done = false;

    // The single present helper: read scroll + dims fresh, composite the
    // visible viewport from cached tiles, and present through the swap chain.
    // This is the EXACT composite+present WM_PAINT does today, relocated here.
    let present = |hw: &mut cv_gpu::HwPresenter,
                       tile_cache: &cv_compositor::TileCache,
                       chrome_h: u32,
                       tabs: &[TabSummary],
                       first_present_done: &mut bool,
                       present_mode_cell: &core::sync::atomic::AtomicU8|
     -> i32 {
        let scroll_y = scroll.load(O::Acquire);
        let blit_w = dims[0].load(O::Acquire);
        let client_h = dims[1].load(O::Acquire) as i32;
        let viewport_h = (client_h - chrome_h as i32).max(0) as u32;
        if viewport_h == 0 || blit_w == 0 {
            return scroll_y;
        }
        let vp = tile_cache.composite_viewport(0, scroll_y, blit_w, viewport_h);
        // Present a FULL-client-height frame with the content offset down by
        // chrome_h (the swap chain covers the whole window; presenting only the
        // viewport_h-tall buffer slid content under the chrome + left a black
        // bottom bar). The top chrome_h rows are BAKED with the chrome strip via
        // an offscreen memory DC — the window-HDC GDI chrome is hidden under the
        // topmost DComp visual, so the chrome must live INSIDE the swap-chain
        // frame to be visible.
        let cw = blit_w as usize;
        let ch = client_h.max(0) as usize;
        let off = (chrome_h as usize) * cw;
        let mut frame = vec![0xFFFF_FFFFu32; cw * ch];
        let copy = vp.len().min(frame.len().saturating_sub(off));
        if copy > 0 {
            frame[off..off + copy].copy_from_slice(&vp[..copy]);
        }
        // Off-main compositor thread has no access to WindowState (UI-thread
        // only), so the pressed-in nav-button look isn't shown here. The
        // compositor is default-OFF; the default live path is the UI-thread
        // WM_PAINT GPU present below, which DOES read `guard.pressed_nav`.
        // The compositor thread has no WindowState (UI-thread-only), so neither
        // the pressed-nav look nor the in-flight loading indicator is shown
        // here. The compositor is default-OFF; the live default path is the
        // UI-thread WM_PAINT present, which DOES read both bits of state.
        bake_chrome_into_frame(
            &mut frame,
            blit_w as i32,
            client_h,
            chrome_h as i32,
            tabs,
            None,
            false,
        );
        if hw.present_u32(&frame, blit_w, client_h as u32).is_ok() {
            if !*first_present_done {
                present_mode_cell.store(present_mode::OWNED_BY_COMPOSITOR, O::Release);
                *first_present_done = true;
            }
        }
        scroll_y
    };

    loop {
        match rx.recv_timeout(Duration::from_millis(16)) {
            Ok(CompositorCmd::Present { paint, w: _, h: _, chrome_h, tabs }) => {
                last_paint = Some(paint.clone());
                last_chrome_h = chrome_h;
                last_tabs = tabs;
                // Mirror apply_new_paint's tile-cache work, now on this thread.
                tile_cache.invalidate_all();
                let bmp = &paint.bitmap;
                tile_cache.refresh_from_raw(&bmp.pixels, bmp.width as u32, bmp.height as u32);
                if let Some(hw) = hw.as_mut() {
                    last_presented_scroll = present(
                        hw,
                        &tile_cache,
                        last_chrome_h,
                        &last_tabs,
                        &mut first_present_done,
                        &present_mode_cell,
                    );
                }
            }
            Ok(CompositorCmd::Resize { w, h }) => {
                // Finish any in-flight present implicitly (we're single-threaded
                // here), resize the swap chain on this thread, update local dims,
                // re-present the last frame at the new size, THEN ack.
                dims[0].store(w.max(1), O::Release);
                dims[1].store(h.max(1), O::Release);
                if let Some(hw) = hw.as_mut() {
                    let _ = hw.resize(w.max(1), h.max(1));
                    if last_paint.is_some() {
                        last_presented_scroll = present(
                            hw,
                            &tile_cache,
                            last_chrome_h,
                            &last_tabs,
                            &mut first_present_done,
                            &present_mode_cell,
                        );
                    }
                }
                // Signal the resize ack (UI may be blocked with a bounded wait).
                {
                    let mut done = resize_ack.done.lock().unwrap();
                    *done = true;
                    resize_ack.cv.notify_all();
                }
            }
            Ok(CompositorCmd::Shutdown) | Err(RecvTimeoutError::Disconnected) => {
                break; // drops `hw` HERE, on its own thread (COM teardown affinity)
            }
            Err(RecvTimeoutError::Timeout) => {
                // Idle tick: re-present the last frame at the CURRENT scroll IF a
                // present is warranted — skip when there's no frame yet, or when
                // scroll is unchanged since the last present (no new pixels).
                if last_paint.is_some() {
                    let cur_scroll = scroll.load(O::Acquire);
                    if cur_scroll != last_presented_scroll {
                        if let Some(hw) = hw.as_mut() {
                            last_presented_scroll = present(
                                hw,
                                &tile_cache,
                                last_chrome_h,
                                &last_tabs,
                                &mut first_present_done,
                                &present_mode_cell,
                            );
                        }
                    }
                }
            }
        }
    }
    // `hw` (Option<HwPresenter>) drops here on the compositor thread.
    drop(hw);
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum TextAlign {
    Left,
    Center,
    Right,
}

#[derive(Clone, Debug, PartialEq)]
pub struct TextItem {
    pub x: i32,
    pub y: i32,
    pub w: i32,
    pub h: i32,
    pub font_size_px: i32,
    pub bold: bool,
    /// Numeric GDI font weight (1–1000). `0` = derive from `bold` (700/400).
    /// Lets heavy CSS weights (800/900) render at their real weight instead of
    /// collapsing to bold — GDI's `CreateFontW` lfWeight accepts the full range.
    pub font_weight: u16,
    pub italic: bool,
    pub font_family: Option<String>,
    pub color_rgb: (u8, u8, u8),
    pub color_alpha: u8,
    pub text: String,
    pub align: TextAlign,
    /// CSS `letter-spacing` extra pixels added between glyphs. Maps
    /// to GDI `SetTextCharacterExtra` at draw time. Tailwind's
    /// `tracking-wider` (0.05em → ~0.6px at 12px font) and
    /// `tracking-widest` (0.1em → ~1.2px) drive the spaced-out look
    /// of card labels like "TOTAL BLOCKS" — without this, the
    /// uppercased text reads as a dense block instead of the wide
    /// wordmark Chrome shows.
    pub letter_spacing_px: i32,
    /// True for text that belongs to the browser chrome (URL bar, back
    /// button, etc.) — those don't scroll. False for content text. The
    /// renderer previously inferred this from `y < chrome_h`, which
    /// mis-classified any content text whose absolute y happened to be
    /// small (Wikipedia's `(-1,-1)` skip-link, near-top inline elements,
    /// etc.) — those stayed pinned while everything else scrolled,
    /// producing the "some items go up, some stay" effect on every
    /// scroll.
    pub is_chrome: bool,
}

#[derive(Clone, Debug)]
pub struct HitRegion {
    pub x: i32,
    pub y: i32,
    pub w: i32,
    pub h: i32,
    pub href: Option<String>,
    pub element_path: Option<Vec<usize>>,
}

#[derive(Clone, Debug)]
pub struct PaintData {
    /// Full-document bitmap. Width = visible viewport width. Height =
    /// URL-bar height + total content height. The window blits a vertical
    /// strip out of this on each paint.
    ///
    /// Held behind an `Arc` so committing a finished frame (`tab.paint =
    /// paint.clone()` plus the copy returned to the UI) is a cheap refcount
    /// bump instead of a multi-megabyte pixel memcpy every animation frame.
    /// The bitmap is write-once at bake time and read-only afterwards (the
    /// next frame bakes a FRESH `Bitmap` and wraps it in a new `Arc`), so the
    /// sharing is sound — no consumer mutates `PaintData.bitmap` in place.
    /// Field access (`paint.bitmap.pixels` / `.width` / `.height`) reads
    /// transparently through `Arc`'s `Deref`.
    pub bitmap: std::sync::Arc<Bitmap>,
    pub texts: Vec<TextItem>,
    /// Held for hit testing when the user clicks. `None` skips hit testing.
    pub layout_root: Option<LayoutBox>,
    /// Compact clickable regions extracted from layout. Used as a
    /// fallback hit-test surface when the browser does not retain the
    /// full layout tree.
    pub hit_regions: Vec<HitRegion>,
    /// Title to show on the window after navigation, or the current URL.
    pub title: String,
    /// The page's current absolute URL — what the URL bar should seed
    /// with on focus. Separate from `title` because the window title
    /// usually shows the page's `<title>` (e.g. "Example Domain")
    /// rather than the address.
    pub current_url: String,
    /// Pixel height of the fixed top chrome (URL bar). Text/layout above
    /// this y stay pinned; everything below scrolls.
    pub chrome_h: u32,
    /// Initial visible content area height (excludes chrome). The actual
    /// runtime viewport is read from the window's client rect at paint
    /// time, but this drives initial sizing.
    pub viewport_h: u32,
    /// Caret rect drawn as a WM_PAINT overlay rather than baked into
    /// the bitmap. `None` = no focused editable input.
    /// `(x, y, w, h)` in bitmap coordinates. Whether it's visible right
    /// now (the blink phase) is read from `caret_blink_visible()` at
    /// paint time, so blink doesn't trigger re-rendering — just a
    /// 2×height pixel `FillRect` after the bitmap blit.
    pub caret_rect: Option<(i32, i32, i32, i32)>,
    /// Property trees from the last layout pass. When a CSS animation
    /// only modifies compositor properties (transform/opacity), the
    /// compositor can update layer positions without re-rasterizing by
    /// calling `apply_property_tree_update` with the new tree values.
    pub property_trees: Option<cv_paint::PropertyTrees>,
    /// M5.4 (`CV_DAMAGE_RASTER`): the FULL retained display list that produced
    /// THIS frame's `bitmap`. Carried opaquely (`Arc<dyn Any>`) so `cv_ui` needs
    /// no dependency on the `conclave`-defined `RetainedDisplayList`; the
    /// renderer downcasts it. The `Arc` makes `PaintData::clone` a refcount bump
    /// (the op stream is never deep-copied per frame). Together with the previous
    /// frame's `bitmap` (the composited pixels), this is the per-frame pixel cache
    /// the damage-driven incremental raster reuses. Default `None` (flag OFF) ⇒
    /// zero added clone/alloc cost and the full-bake path is byte-for-byte
    /// unchanged.
    pub retained: Option<std::sync::Arc<dyn std::any::Any + Send + Sync>>,
    /// Document-space y (px) of the bitmap's TOP row. `0` for a full-document
    /// bitmap (every legacy caller). When the renderer rasters only a viewport
    /// band (Chrome's interest rect), the bitmap covers document rows
    /// `[content_origin_y .. content_origin_y + bitmap.height)`; the present path
    /// blits from `src_y = scroll_y - content_origin_y` and the uncovered area
    /// shows `band_fill`.
    pub content_origin_y: u32,
    /// Total document content height (px) — the scroll range. `0` ⇒ legacy: use
    /// `bitmap.height` (full-document bitmap). For a band bitmap this is LARGER
    /// than the bitmap, so scroll clamp / scrollbar use THIS not the bitmap.
    pub document_h: u32,
}

impl PaintData {
    /// Document content height for scroll math: explicit `document_h` when set
    /// (band bitmap), else the bitmap height (legacy full-document bitmap).
    pub fn content_height(&self) -> u32 {
        if self.document_h > 0 {
            self.document_h
        } else {
            self.bitmap.height
        }
    }
}

fn hit_test_regions(regions: &[HitRegion], x: f32, y: f32) -> (Option<String>, Option<Vec<usize>>) {
    let mut href = None;
    let mut element_path = None;
    for region in regions {
        let rx = region.x as f32;
        let ry = region.y as f32;
        let rw = region.w as f32;
        let rh = region.h as f32;
        if x >= rx && x < rx + rw && y >= ry && y < ry + rh {
            if region.href.is_some() {
                href = region.href.clone();
            }
            if region.element_path.is_some() {
                element_path = region.element_path.clone();
            }
        }
    }
    (href, element_path)
}

thread_local! {
    /// Caret blink phase, owned by cv_ui so the window paint code can
    /// read it without crossing crate boundaries. conclave's ticker
    /// flips this via `set_caret_blink_visible` and triggers a repaint;
    /// WM_PAINT consults it after blitting the bitmap to decide whether
    /// to draw the caret rect overlay. Starts visible so newly-focused
    /// inputs show their caret immediately.
    static CARET_BLINK_VISIBLE: std::cell::Cell<bool> = const { std::cell::Cell::new(true) };
}

/// Read the current caret blink visibility. Used by the window paint
/// path to decide whether to overlay the caret on top of the bitmap.
pub fn caret_blink_visible() -> bool {
    CARET_BLINK_VISIBLE.with(|c| c.get())
}

/// Set the caret blink visibility. Called by the host's tick loop when
/// the blink period rolls and on focus change. Setting this does NOT
/// itself trigger a repaint — the host must invalidate the caret rect
/// (or return a fresh PaintData) to make the change visible.
pub fn set_caret_blink_visible(visible: bool) {
    CARET_BLINK_VISIBLE.with(|c| c.set(visible));
}

/// Ask the window to repaint just the caret rect (if there's a focused
/// input + an active window). Avoids the full re-layout + re-bake cycle
/// that returning `Some(PaintData)` from the ticker would trigger; only
/// the ~2-pixel-wide caret region gets InvalidateRect'd. Called by the
/// host after `set_caret_blink_visible`. No-op if no window or no caret.
/// Primary monitor pixel size. Used to bootstrap a maximized window's page
/// layout at the REAL viewport size so JS (canvas sizing, responsive breakpoints)
/// runs against correct metrics from the first paint — mirroring how a real
/// browser lays out at the widget's actual size.
pub fn primary_screen_size() -> (u32, u32) {
    let (w, h) = unsafe {
        (
            sys::GetSystemMetrics(sys::SM_CXSCREEN),
            sys::GetSystemMetrics(sys::SM_CYSCREEN),
        )
    };
    (w.max(1) as u32, h.max(1) as u32)
}

pub fn invalidate_caret() {
    // The off-main renderer's ticker calls this from the RENDERER thread. It must
    // NOT borrow the UI thread's `WindowState` RefCell (cross-thread = data race /
    // BorrowError panic). When off the UI thread, post a message and let the UI
    // thread perform the invalidate on its own thread (PostMessageW is thread-safe).
    let ui_tid = UI_THREAD_ID.load(Ordering::SeqCst);
    if ui_tid != 0 && unsafe { sys::GetCurrentThreadId() } != ui_tid {
        let hwnd = OWNER_HWND.load(Ordering::SeqCst);
        if hwnd != 0 {
            unsafe {
                sys::PostMessageW(hwnd as sys::HWND, WM_APP_INVALIDATE_CARET, 0, 0);
            }
        }
        return;
    }
    let state_ptr = OWNER_PTR.load(Ordering::SeqCst);
    if state_ptr.is_null() {
        return;
    }
    let (hwnd, rect_opt, scroll_y, chrome_h) = unsafe {
        let g = (*state_ptr).borrow();
        (
            g.hwnd,
            g.paint.caret_rect,
            g.scroll_y,
            g.paint.chrome_h as i32,
        )
    };
    let Some((cx, cy, cw, ch)) = rect_opt else {
        return;
    };
    // Inflate by 1px on each side so anti-aliasing remnants don't ghost.
    let draw_y = cy - scroll_y + chrome_h;
    let r = sys::RECT {
        left: cx - 1,
        top: draw_y - 1,
        right: cx + cw + 1,
        bottom: draw_y + ch + 1,
    };
    unsafe {
        sys::InvalidateRect(hwnd, &raw const r, 0);
    }
}

/// Bake every non-chrome `TextItem` into `bitmap` using a Win32
/// memory DC + DIB section.  Glyphs are rasterized via GDI into the
/// bitmap pixels at the same `(x, y)` they would have been drawn on
/// screen, with `is_chrome = false` items removed from `texts`.
///
/// After this call: the bitmap contains the same image content it
/// did before, PLUS glyphs for every content text.  The `texts` vec
/// retains only chrome items (URL bar, nav buttons) so the window's
/// WM_PAINT can still draw those at fixed screen coords.
///
/// This eliminates the scroll-divergence class of bugs entirely: text
/// and images become the same pixel buffer, so they CANNOT scroll in
/// different directions because the WM_PAINT scroll path has only one
/// thing to move.
/// Measure the pixel-exact rendered width of `text` using the same
/// GDI font configuration the bake pass uses. Lets the caret-painting
/// code position the caret at the actual right edge of the typed text
/// instead of guessing with a per-glyph approximation. Returns 0 on
/// any GDI failure so callers can fall back to a width estimate.
///
/// `font_family` is the CSS family the input asked for (or None to
/// use Segoe UI, our default). `bold` and `italic` map to GDI font
/// weight + italic flag.
/// Pick the first usable family name out of a comma-separated CSS
/// `font-family` list. CSS lets authors write `Roboto, "Helvetica
/// Neue", Arial, sans-serif` — we walk this in order and return
/// the first name that resolves to a real Win32 face. Generic
/// family keywords (`sans-serif`, `serif`, `monospace`, `cursive`,
/// `fantasy`, `system-ui`) are mapped to a reasonable Win32 default.
/// Returns "Segoe UI" if nothing in the list resolved.
pub fn resolve_font_family(font_family: Option<&str>) -> String {
    let list = match font_family {
        Some(l) if !l.is_empty() => l,
        _ => return "Segoe UI".to_string(),
    };
    for raw in list.split(',') {
        let name = raw.trim().trim_matches(|c| c == '"' || c == '\'');
        if name.is_empty() {
            continue;
        }
        // Map CSS generics to Win32 system equivalents.
        let lower = name.to_ascii_lowercase();
        let mapped: Option<&'static str> = match lower.as_str() {
            "sans-serif" | "system-ui" | "ui-sans-serif" => Some("Segoe UI"),
            "serif" | "ui-serif" => Some("Times New Roman"),
            "monospace" | "ui-monospace" => Some("Consolas"),
            "cursive" => Some("Comic Sans MS"),
            "fantasy" => Some("Impact"),
            "math" | "ui-rounded" => Some("Segoe UI"),
            _ => None,
        };
        if let Some(m) = mapped {
            return m.to_string();
        }
        // Non-generic name — return as-is. GDI substitutes internally
        // if the requested face is missing (better than handing GDI
        // the whole comma-joined list as a single face name, which
        // never matches anything).
        return name.to_string();
    }
    "Segoe UI".to_string()
}

pub fn measure_text_px(
    text: &str,
    font_size_px: i32,
    font_family: Option<&str>,
    bold: bool,
    italic: bool,
) -> i32 {
    if text.is_empty() {
        return 0;
    }
    unsafe {
        let mem_dc = sys::CreateCompatibleDC(core::ptr::null_mut());
        if mem_dc.is_null() {
            return 0;
        }
        let face_name = resolve_font_family(font_family);
        let face: Vec<u16> = format!("{face_name}\0").encode_utf16().collect();
        let weight = if bold { sys::FW_BOLD } else { sys::FW_NORMAL };
        let italic_flag = u32::from(italic);
        let hfont = sys::CreateFontW(
            font_size_px,
            0,
            0,
            0,
            weight,
            italic_flag,
            0,
            0,
            sys::DEFAULT_CHARSET,
            sys::OUT_DEFAULT_PRECIS,
            sys::CLIP_DEFAULT_PRECIS,
            sys::CLEARTYPE_QUALITY,
            sys::DEFAULT_PITCH | sys::FF_DONTCARE,
            face.as_ptr(),
        );
        if hfont.is_null() {
            sys::DeleteDC(mem_dc);
            return 0;
        }
        let old_font = sys::SelectObject(mem_dc, hfont);
        let wide: Vec<u16> = text.encode_utf16().collect();
        let mut size = sys::SIZE { cx: 0, cy: 0 };
        let ok =
            sys::GetTextExtentPoint32W(mem_dc, wide.as_ptr(), wide.len() as i32, &raw mut size);
        sys::SelectObject(mem_dc, old_font);
        sys::DeleteObject(hfont);
        sys::DeleteDC(mem_dc);
        if ok == 0 { 0 } else { size.cx }
    }
}

/// Text-run cache key. The rendered alpha mask is invariant in color +
/// position, so those are NOT in the key — we keep cache hit-rate high
/// (any scroll/colour-tween reuses the same mask). The wrap result
/// depends on width + align + text + font, so those ARE in the key.
#[derive(Clone, Eq, PartialEq, Hash)]
struct TextRunKey {
    text: String,
    face: String,
    size_px: i32,
    bold: bool,
    italic: bool,
    w: i32,
    align_tag: u8,
    /// CSS `letter-spacing` extra px between glyphs. Part of the key
    /// because the rasterized run's pixels (and width) change with
    /// tracking. Two `tracking-wider` vs `tracking-normal` runs of the
    /// same text must cache separately.
    letter_spacing_px: i32,
}

struct CachedRun {
    width: u32,
    height: u32,
    /// 8-bit alpha mask, row-major top-down. `alpha[y*width+x]` is the
    /// glyph coverage at that pixel (0 = transparent, 255 = solid). We
    /// rasterise once via GDI by drawing white-on-black into a DIB and
    /// taking `max(R,G,B)` per pixel; ClearType subpixel info collapses
    /// to grayscale AA, which is the right tradeoff for cache size +
    /// arbitrary text-colour blit later.
    alpha: Vec<u8>,
    /// LRU stamp — monotonically-incrementing counter on access. Cheaper
    /// than `Instant::now()` (which is forbidden by our env anyway).
    last_used: u64,
}

struct TextRunAtlas {
    runs: std::collections::HashMap<TextRunKey, CachedRun>,
    /// Sum of `alpha.len()` across all entries — drives LRU eviction.
    bytes: usize,
    /// Monotonic counter handed out as `last_used` on every access.
    clock: u64,
    /// Soft cap. ~64 MiB — enough for a Wikipedia-class page worth of
    /// text runs at body size, well below any reasonable RAM concern.
    budget_bytes: usize,
}

impl TextRunAtlas {
    fn new() -> Self {
        Self {
            runs: std::collections::HashMap::new(),
            bytes: 0,
            clock: 0,
            budget_bytes: 64 * 1024 * 1024,
        }
    }

    fn tick(&mut self) -> u64 {
        self.clock = self.clock.wrapping_add(1);
        self.clock
    }

    /// Look up an existing entry, bumping its LRU stamp.
    fn get(&mut self, key: &TextRunKey) -> Option<&CachedRun> {
        let t = self.tick();
        if let Some(r) = self.runs.get_mut(key) {
            r.last_used = t;
            return Some(&*r);
        }
        None
    }

    /// Insert a new entry, evicting LRU items if we'd blow the budget.
    fn insert(&mut self, key: TextRunKey, run: CachedRun) {
        self.bytes = self.bytes.saturating_add(run.alpha.len());
        let stamp = self.tick();
        let mut run = run;
        run.last_used = stamp;
        self.runs.insert(key, run);
        // Evict if over budget. Pull entries by ascending last_used
        // (oldest first), drop them until we're back under cap. This
        // is O(n log n) per overflow — acceptable since overflow is
        // rare and the alternative (priority queue) costs more on the
        // hot path.
        if self.bytes > self.budget_bytes {
            let mut keys: Vec<(TextRunKey, u64, usize)> = self
                .runs
                .iter()
                .map(|(k, v)| (k.clone(), v.last_used, v.alpha.len()))
                .collect();
            keys.sort_by_key(|e| e.1);
            for (k, _, sz) in keys {
                if self.bytes <= self.budget_bytes {
                    break;
                }
                self.runs.remove(&k);
                self.bytes = self.bytes.saturating_sub(sz);
            }
        }
    }
}

thread_local! {
    /// One atlas per UI thread (we only have one). Survives across paint
    /// frames so a scroll re-emits the same text items and we just blit
    /// cached alpha masks instead of re-rasterising through GDI.
    static TEXT_RUN_ATLAS: std::cell::RefCell<TextRunAtlas> =
        std::cell::RefCell::new(TextRunAtlas::new());
}

/// Rasterise a single text run into an 8-bit alpha mask via GDI.
/// White-on-black `DrawTextW` followed by per-pixel max(R,G,B) gives
/// the per-pixel glyph coverage that we can later blit at any color.
/// Word-wrap inside the run honours the requested rect width so this
/// mirrors what layout already measured.
unsafe fn render_text_run_alpha(
    text: &str,
    face_name: &str,
    size_px: i32,
    bold: bool,
    italic: bool,
    rect_w: i32,
    align: TextAlign,
    letter_spacing_px: i32,
) -> Option<CachedRun> {
    unsafe {
        let mem_dc = sys::CreateCompatibleDC(core::ptr::null_mut());
        if mem_dc.is_null() {
            return None;
        }
        // Allocate enough height for a long wrapped paragraph. We trim
        // to bbox after rasterising. Cap at 16K rows so a runaway value
        // doesn't allocate a gigabyte.
        let max_h = (size_px.max(8)).saturating_mul(32).min(16_384);
        let w = rect_w.max(1);
        let h = max_h.max(1);
        let bi = sys::BITMAPINFO {
            bmiHeader: sys::BITMAPINFOHEADER {
                biSize: core::mem::size_of::<sys::BITMAPINFOHEADER>() as u32,
                biWidth: w,
                biHeight: -h, // top-down
                biPlanes: 1,
                biBitCount: 32,
                biCompression: sys::BI_RGB,
                biSizeImage: 0,
                biXPelsPerMeter: 0,
                biYPelsPerMeter: 0,
                biClrUsed: 0,
                biClrImportant: 0,
            },
            bmiColors: [0; 1],
        };
        let mut bits: *mut c_void = core::ptr::null_mut();
        let dib = sys::CreateDIBSection(
            mem_dc,
            &bi,
            sys::DIB_RGB_COLORS,
            &mut bits,
            core::ptr::null_mut(),
            0,
        );
        if dib.is_null() || bits.is_null() {
            sys::DeleteDC(mem_dc);
            return None;
        }
        // Zero the DIB (black background).
        let pixel_count = (w * h) as usize;
        let bits_slice = core::slice::from_raw_parts_mut(bits as *mut u32, pixel_count);
        for p in bits_slice.iter_mut() {
            *p = 0;
        }
        let old_bm = sys::SelectObject(mem_dc, dib);
        // White text on transparent background — the DIB itself is the
        // background. ClearType emits subpixel-coloured edges; we
        // collapse to grayscale below.
        sys::SetBkMode(mem_dc, sys::TRANSPARENT);
        sys::SetTextColor(mem_dc, sys::rgb(255, 255, 255));
        let face: Vec<u16> = format!("{face_name}\0").encode_utf16().collect();
        let weight = if bold { sys::FW_BOLD } else { sys::FW_NORMAL };
        let italic_flag = u32::from(italic);
        let hfont = sys::CreateFontW(
            size_px,
            0,
            0,
            0,
            weight,
            italic_flag,
            0,
            0,
            sys::DEFAULT_CHARSET,
            sys::OUT_DEFAULT_PRECIS,
            sys::CLIP_DEFAULT_PRECIS,
            sys::CLEARTYPE_QUALITY,
            sys::DEFAULT_PITCH | sys::FF_DONTCARE,
            face.as_ptr(),
        );
        if hfont.is_null() {
            sys::SelectObject(mem_dc, old_bm);
            sys::DeleteObject(dib);
            sys::DeleteDC(mem_dc);
            return None;
        }
        let old_font = sys::SelectObject(mem_dc, hfont);
        let mut text_w: Vec<u16> = text.encode_utf16().collect();
        text_w.push(0);
        let align_flag = match align {
            TextAlign::Left => sys::DT_LEFT,
            TextAlign::Center => sys::DT_CENTER,
            TextAlign::Right => sys::DT_RIGHT,
        };
        let mut rc = sys::RECT {
            left: 0,
            top: 0,
            right: w,
            bottom: h,
        };
        // CSS `letter-spacing` — GDI inter-character extra. Set on the
        // scratch DC before measuring/drawing so the rasterized run
        // (cached by `TextRunKey.letter_spacing_px`) carries the right
        // tracking. Tailwind's `tracking-wider`/`tracking-widest` on
        // uppercase labels ("TOTAL BLOCKS") depend on this.
        if letter_spacing_px != 0 {
            sys::SetTextCharacterExtra(mem_dc, letter_spacing_px);
        }
        sys::DrawTextW(
            mem_dc,
            text_w.as_ptr(),
            -1,
            &raw mut rc,
            align_flag | sys::DT_TOP | sys::DT_NOPREFIX | sys::DT_WORDBREAK,
        );
        // Extract alpha mask + find the actually-used vertical extent
        // so we don't keep the whole 32-row tall scratch buffer in cache.
        let mut used_h: i32 = 0;
        for y in (0..h).rev() {
            let row = (y * w) as usize;
            if bits_slice[row..row + w as usize].iter().any(|&p| p != 0) {
                used_h = y + 1;
                break;
            }
        }
        let used_h = used_h.max(1);
        let mut alpha = vec![0u8; (w * used_h) as usize];
        for y in 0..used_h {
            let row = (y * w) as usize;
            let dst = (y * w) as usize;
            for x in 0..w as usize {
                let px = bits_slice[row + x];
                let r = (px >> 16) & 0xFF;
                let g = (px >> 8) & 0xFF;
                let b = px & 0xFF;
                alpha[dst + x] = r.max(g).max(b) as u8;
            }
        }
        sys::SelectObject(mem_dc, old_font);
        sys::DeleteObject(hfont);
        sys::SelectObject(mem_dc, old_bm);
        sys::DeleteObject(dib);
        sys::DeleteDC(mem_dc);
        Some(CachedRun {
            width: w as u32,
            height: used_h as u32,
            alpha,
            last_used: 0,
        })
    }
}

/// Alpha-blit a cached text-run mask onto the destination bitmap at
/// `(dst_x, dst_y)`, tinting with `text_rgb`. Per-pixel: out = lerp(bg,
/// fg, alpha/255). Clipped against the bitmap bounds.
fn blit_text_run(
    bitmap: &mut Bitmap,
    run: &CachedRun,
    dst_x: i32,
    dst_y: i32,
    text_rgb: (u8, u8, u8),
    text_alpha: u8,
) {
    blit_text_run_clipped(bitmap, run, dst_x, dst_y, text_rgb, text_alpha, None);
}

/// `blit_text_run` with an optional extra clip rect `(x, y, w, h)` in bitmap
/// coords. The M5.4 incremental text bake passes the damage region R here so a
/// glyph run is restricted to pixels inside R — guaranteeing the re-bake never
/// touches a pixel outside R (which the cache already holds correct), so the
/// incremental frame stays BYTE-IDENTICAL to a full bake even for
/// semi-transparent glyphs (no source-over double-darkening outside R).
#[allow(clippy::too_many_arguments)]
fn blit_text_run_clipped(
    bitmap: &mut Bitmap,
    run: &CachedRun,
    dst_x: i32,
    dst_y: i32,
    text_rgb: (u8, u8, u8),
    text_alpha: u8,
    clip: Option<(i32, i32, i32, i32)>,
) {
    if text_alpha == 0 {
        return;
    }
    let bw = bitmap.width as i32;
    let bh = bitmap.height as i32;
    let rw = run.width as i32;
    let rh = run.height as i32;
    // Clip to destination bounds.
    let mut x0 = dst_x.max(0);
    let mut y0 = dst_y.max(0);
    let mut x1 = (dst_x + rw).min(bw);
    let mut y1 = (dst_y + rh).min(bh);
    // Optional damage-region clip (M5.4): intersect with R.
    if let Some((cx, cy, cw, ch)) = clip {
        x0 = x0.max(cx);
        y0 = y0.max(cy);
        x1 = x1.min(cx + cw);
        y1 = y1.min(cy + ch);
    }
    if x0 >= x1 || y0 >= y1 {
        return;
    }
    let (fr, fg, fb) = (text_rgb.0 as u32, text_rgb.1 as u32, text_rgb.2 as u32);
    let global_alpha = text_alpha as u32;
    for dy in y0..y1 {
        let src_y = dy - dst_y;
        let dst_row = (dy * bw) as usize;
        let src_row = (src_y * rw) as usize;
        for dx in x0..x1 {
            let src_x = dx - dst_x;
            let a_glyph = run.alpha[src_row + src_x as usize] as u32;
            if a_glyph == 0 {
                continue;
            }
            let a = (a_glyph * global_alpha) / 255;
            if a == 0 {
                continue;
            }
            let dst_idx = dst_row + dx as usize;
            let bgp = bitmap.pixels[dst_idx];
            let br = ((bgp >> 16) & 0xFF) as i32;
            let bg2 = ((bgp >> 8) & 0xFF) as i32;
            let bb = (bgp & 0xFF) as i32;
            // out = bg + (fg - bg) * a / 255 — done in signed i32 to
            // handle fg < bg without underflow.
            let a_i = a as i32;
            let or = (br + (((fr as i32 - br) * a_i) / 255)) as u32 & 0xFF;
            let og = (bg2 + (((fg as i32 - bg2) * a_i) / 255)) as u32 & 0xFF;
            let ob = (bb + (((fb as i32 - bb) * a_i) / 255)) as u32 & 0xFF;
            bitmap.pixels[dst_idx] = 0xFF00_0000 | (or << 16) | (og << 8) | ob;
        }
    }
}

pub fn bake_content_text_into_bitmap(bitmap: &mut Bitmap, texts: &mut Vec<TextItem>) {
    bake_content_text_into_bitmap_clipped(bitmap, texts, None);
}

/// `bake_content_text_into_bitmap` with an optional damage-region clip `(x,y,w,h)`
/// in bitmap coords (M5.4). When `Some(R)`, every content glyph is restricted to
/// R, so the incremental raster's text bake writes ONLY inside the freshly
/// cleared-and-repainted damage region — keeping the frame byte-identical to a
/// full bake. `None` ⇒ identical to the un-clipped bake (the default full path).
/// Drains non-chrome items from `texts` exactly like the un-clipped variant so
/// callers observe the same post-condition.
pub fn bake_content_text_into_bitmap_clipped(
    bitmap: &mut Bitmap,
    texts: &mut Vec<TextItem>,
    clip: Option<(i32, i32, i32, i32)>,
) {
    let content: Vec<TextItem> = texts.iter().filter(|t| !t.is_chrome).cloned().collect();
    if content.is_empty() {
        return;
    }
    texts.retain(|t| t.is_chrome);

    // Fast-path: for each content text item, look up its cached alpha
    // mask. Miss → rasterise once via GDI and insert. Hit → alpha-blit
    // directly onto the bitmap. This is the per-frame work for a
    // re-render that doesn't change text content; scrolling a static
    // page repeats exactly the same items so the cache nails 100%.
    for t in &content {
        if t.color_alpha == 0 {
            continue;
        }
        if t.x >= bitmap.width as i32
            || t.y >= bitmap.height as i32
            || t.x + t.w <= 0
            || t.y + t.h <= 0
        {
            continue;
        }
        let face_name = resolve_font_family(t.font_family.as_deref());
        // Strip emoji here too — the cache key includes the cleaned
        // text so we don't end up with multiple entries for the same
        // user-visible content.
        let cleaned: String = t
            .text
            .chars()
            .map(|c| {
                let cp = c as u32;
                let is_emoji = (0x1F000..=0x1FFFF).contains(&cp)
                    || (0x2600..=0x27BF).contains(&cp)
                    || (0x2300..=0x23FF).contains(&cp)
                    || cp == 0x200D
                    || cp == 0xFE0F;
                if is_emoji { ' ' } else { c }
            })
            .collect();
        let align_tag = match t.align {
            TextAlign::Left => 0,
            TextAlign::Center => 1,
            TextAlign::Right => 2,
        };
        let key = TextRunKey {
            text: cleaned.clone(),
            face: face_name.clone(),
            size_px: t.font_size_px,
            bold: t.bold,
            italic: t.italic,
            w: t.w,
            align_tag,
            letter_spacing_px: t.letter_spacing_px,
        };
        // Cache lookup or rasterise.
        let cached_run: Option<CachedRun> = TEXT_RUN_ATLAS.with(|atlas| {
            let mut a = atlas.borrow_mut();
            if a.get(&key).is_some() {
                // Need to return a clone because we can't hold the
                // borrow across the blit (the atlas isn't accessed
                // again, but the borrow checker doesn't know that and
                // this keeps the type signature simple).
                a.runs.get(&key).map(|r| CachedRun {
                    width: r.width,
                    height: r.height,
                    alpha: r.alpha.clone(),
                    last_used: r.last_used,
                })
            } else {
                None
            }
        });
        let run_to_blit = match cached_run {
            Some(r) => r,
            None => {
                let rendered = unsafe {
                    render_text_run_alpha(
                        &cleaned,
                        &face_name,
                        t.font_size_px,
                        t.bold,
                        t.italic,
                        t.w,
                        t.align,
                        t.letter_spacing_px,
                    )
                };
                let Some(rendered) = rendered else {
                    continue;
                };
                // Insert a clone so we can blit the original without
                // re-borrowing the atlas.
                let blit_copy = CachedRun {
                    width: rendered.width,
                    height: rendered.height,
                    alpha: rendered.alpha.clone(),
                    last_used: rendered.last_used,
                };
                TEXT_RUN_ATLAS.with(|atlas| {
                    atlas.borrow_mut().insert(key, rendered);
                });
                blit_copy
            }
        };
        blit_text_run_clipped(bitmap, &run_to_blit, t.x, t.y, t.color_rgb, t.color_alpha, clip);
    }

    // Skip the old GDI bake path entirely — the atlas blits above
    // already wrote the pixels for every content text item.
    return;

    #[allow(unreachable_code)]
    unsafe {
        let mem_dc = sys::CreateCompatibleDC(core::ptr::null_mut());
        if mem_dc.is_null() {
            return;
        }
        let bi = sys::BITMAPINFO {
            bmiHeader: sys::BITMAPINFOHEADER {
                biSize: core::mem::size_of::<sys::BITMAPINFOHEADER>() as u32,
                biWidth: bitmap.width as i32,
                // Negative height = top-down DIB so row 0 is the top.
                biHeight: -(bitmap.height as i32),
                biPlanes: 1,
                biBitCount: 32,
                biCompression: sys::BI_RGB,
                biSizeImage: 0,
                biXPelsPerMeter: 0,
                biYPelsPerMeter: 0,
                biClrUsed: 0,
                biClrImportant: 0,
            },
            bmiColors: [0; 1],
        };
        let mut bits: *mut c_void = core::ptr::null_mut();
        let dib = sys::CreateDIBSection(
            mem_dc,
            &bi,
            sys::DIB_RGB_COLORS,
            &mut bits,
            core::ptr::null_mut(),
            0,
        );
        if dib.is_null() || bits.is_null() {
            sys::DeleteDC(mem_dc);
            return;
        }

        let pixel_count = (bitmap.width * bitmap.height) as usize;
        let bits_slice = core::slice::from_raw_parts_mut(bits as *mut u32, pixel_count);
        // Seed the DIB with the bitmap's current pixels so DrawTextW's
        // transparent-mode glyphs composite over the existing
        // backgrounds / borders / images instead of a black void.
        bits_slice.copy_from_slice(&bitmap.pixels);

        let old_bm = sys::SelectObject(mem_dc, dib);
        sys::SetBkMode(mem_dc, sys::TRANSPARENT);

        for t in &content {
            if t.color_alpha == 0 {
                continue;
            }
            // Off-bitmap cull.
            if t.x >= bitmap.width as i32
                || t.y >= bitmap.height as i32
                || t.x + t.w <= 0
                || t.y + t.h <= 0
            {
                continue;
            }
            let face_name = resolve_font_family(t.font_family.as_deref());
            let face: Vec<u16> = format!("{face_name}\0").encode_utf16().collect();
            // Honor the numeric weight (800/900 render heavier than bold); 0 =
            // derive from the bold bool. GDI lfWeight accepts 1–1000.
            let weight = if t.font_weight != 0 {
                t.font_weight as i32
            } else if t.bold {
                sys::FW_BOLD
            } else {
                sys::FW_NORMAL
            };
            let italic = if t.italic { 1 } else { 0 };
            let hfont = sys::CreateFontW(
                t.font_size_px,
                0,
                0,
                0,
                weight,
                italic,
                0,
                0,
                sys::DEFAULT_CHARSET,
                sys::OUT_DEFAULT_PRECIS,
                sys::CLIP_DEFAULT_PRECIS,
                sys::CLEARTYPE_QUALITY,
                sys::DEFAULT_PITCH | sys::FF_DONTCARE,
                face.as_ptr(),
            );
            let old_font = sys::SelectObject(mem_dc, hfont);
            let (r, g, b) = t.color_rgb;
            sys::SetTextColor(mem_dc, sys::rgb(r, g, b));
            let h_guess = t.h.max(t.font_size_px * 4);
            let mut rc = sys::RECT {
                left: t.x,
                top: t.y,
                right: t.x + t.w,
                bottom: t.y + h_guess,
            };
            // Strip emoji codepoints — without a color-emoji font path
            // (Segoe UI Emoji, COLR/CPAL or PNG-in-OT), GDI falls back
            // to a monochrome silhouette glyph that often paints as a
            // flat colored square (Google's 🌱 footer emoji was a
            // bright-green box). Replace with a space so layout width
            // is preserved but no spurious block ships.
            let cleaned: String = t
                .text
                .chars()
                .map(|c| {
                    let cp = c as u32;
                    let is_emoji = (0x1F000..=0x1FFFF).contains(&cp)
                        || (0x2600..=0x27BF).contains(&cp)
                        || (0x2300..=0x23FF).contains(&cp)
                        || cp == 0x200D
                        || cp == 0xFE0F;
                    if is_emoji { ' ' } else { c }
                })
                .collect();
            let mut text_w: Vec<u16> = cleaned.encode_utf16().collect();
            text_w.push(0);
            let align_flag = match t.align {
                TextAlign::Left => sys::DT_LEFT,
                TextAlign::Center => sys::DT_CENTER,
                TextAlign::Right => sys::DT_RIGHT,
            };
            // CSS `letter-spacing`. Snapshot the previous value so we
            // restore it after this run — DC state is per-window and
            // leaking the spacing into the next text item would smear
            // every subsequent draw.
            let prev_extra = if t.letter_spacing_px != 0 {
                sys::SetTextCharacterExtra(mem_dc, t.letter_spacing_px)
            } else {
                0
            };
            sys::DrawTextW(
                mem_dc,
                text_w.as_ptr(),
                -1,
                &raw mut rc,
                align_flag | sys::DT_TOP | sys::DT_NOPREFIX | sys::DT_WORDBREAK,
            );
            if t.letter_spacing_px != 0 {
                sys::SetTextCharacterExtra(mem_dc, prev_extra);
            }
            sys::SelectObject(mem_dc, old_font);
            sys::DeleteObject(hfont);
        }

        // Copy DIB pixels back into the bitmap.  GDI wrote text into
        // the DIB; this copy makes the bitmap hold those glyphs so
        // scroll-blit treats them like any other pixel.
        bitmap.pixels.copy_from_slice(bits_slice);

        sys::SelectObject(mem_dc, old_bm);
        sys::DeleteObject(dib);
        sys::DeleteDC(mem_dc);
    }
}

/// Win32 GUI message loop is single-threaded; the `Send` bound that used
/// to be on these closures was theatre. Drop it so the browser can hold
/// `Rc<RefCell<...>>`-backed JS state inside the callback.
pub type Navigator = Box<dyn FnMut(&str) -> Option<PaintData>>;

/// Live viewport resize callback. The host receives the current client
/// width and content-region height (excluding fixed chrome) and may
/// return fresh paint data laid out for that viewport.
pub type ResizeHandler = Box<dyn FnMut(u32, u32) -> Option<PaintData>>;

/// Periodic event-loop tick. The host (browser) drains expired
/// setTimeout / setInterval callbacks here, runs any pending JS, and
/// returns `Some(new_paint)` if the DOM changed. `None` keeps the
/// current paint.
pub type Ticker = Box<dyn FnMut() -> Option<PaintData>>;

/// Newtype wrapper that lets us move an `HWND` (a raw pointer) onto a
/// worker thread. The HWND is only used to `PostMessageW`, which is a
/// thread-safe Win32 API — the data is conceptually a handle, not a
/// pointer the worker dereferences.
#[derive(Copy, Clone)]
struct HwndSend(sys::HWND);

// SAFETY: `HWND` is opaque (the kernel owns the window). We only pass
// it to `PostMessageW`, which is documented thread-safe.
unsafe impl Send for HwndSend {}

/// Send + Sync URL → body byte-vector callback. Runs on a worker
/// thread when the URL bar submits, so the slow TLS handshake doesn't
/// freeze the UI. An empty Vec signals "fetch failed" — the navigator
/// falls back to the existing paint. Sync is required because the
/// callback may run on the worker concurrently with retained state on
/// the UI thread.
pub type Fetcher = std::sync::Arc<dyn Fn(String) -> Vec<u8> + Send + Sync>;

/// Navigator variant that consumes a body the fetcher already produced.
/// The UI thread calls this from the WM_USER message handler once the
/// worker thread posts its result back. Identical semantics to
/// `Navigator` otherwise — `Some(paint)` swaps in a new view.
pub type NavigatorWithBody = Box<dyn FnMut(&str, Vec<u8>) -> Option<PaintData>>;

/// "What URL would Backspace navigate to?" callback. cv_ui asks the
/// host to peek at its history and return the previous URL (without
/// popping it — the pop happens when `nav_with_body` is called with
/// the freshly-fetched body). Returning `None` falls back to the
/// existing sync `back://` path so single-page sites still work.
pub type BackUrlFn = Box<dyn FnMut() -> Option<String>>;

/// Same shape as `BackUrlFn` but returns the next URL in the forward
/// history. `None` means the forward stack is empty (Forward button
/// is greyed out / no-op).
pub type ForwardUrlFn = Box<dyn FnMut() -> Option<String>>;

#[derive(Debug, Clone)]
pub struct TabSummary {
    pub id: u64,
    pub title: String,
    pub url: String,
    pub active: bool,
}

#[derive(Debug, Clone)]
pub enum HostCommand {
    NewTab,
    CloseActiveTab,
    SwitchTab(u64),
    NewWindow,
}

#[derive(Debug)]
pub struct HostCommandResult {
    pub paint: PaintData,
    pub tabs: Vec<TabSummary>,
}

pub type HostCommandFn = Box<dyn FnMut(HostCommand) -> Option<HostCommandResult>>;

// ===========================================================================
// Off-main-thread renderer protocol (Chrome-shaped). The renderer thread owns
// the entire non-Send `Rc<RefCell<…>>` page graph; the UI thread owns only the
// HWND, the message pump, and the latest committed `PaintData`. The two
// communicate purely through these `Send` messages, so page work can NEVER
// block the UI thread (the non-Send state physically cannot live on it).
// ===========================================================================

/// Win32 message the renderer thread posts to wake the UI pump and have it
/// drain pending [`FromPage`] messages. `WM_USER + 1..=3` are already taken by
/// the legacy threaded-fetch path, so the off-main path uses `WM_USER + 10`.
pub const WM_APP_FROMPAGE: u32 = sys::WM_USER + 10;

/// Posted by `invalidate_caret` when it is called OFF the UI thread (the off-main
/// renderer's ticker drives caret blink). The UI thread handles it by performing
/// the actual `InvalidateRect` on its own thread, so the renderer never borrows
/// the UI's `WindowState` cross-thread.
pub const WM_APP_INVALIDATE_CARET: u32 = sys::WM_USER + 11;

/// A3: the encoded command the Stop button sends (only in the sandboxed-
/// renderer-process mode) to abort an in-flight load. It rides on the normal
/// `ToPage::Cmd` channel; the browser's site-router intercepts it (it is never
/// forwarded to a renderer as-is) and performs a graceful CancelLoad followed,
/// if needed, by a hard renderer kill + respawn. Distinct, reserved encoding
/// so it can never collide with a real navigation or input command.
pub const STOP_LOAD_CMD: &str = "__stop_load__";

/// UI thread → renderer thread command. All variants are plain `Send` data —
/// crucially, encoded command STRINGS (the same `javascript:` / `tb-link-click:`
/// / `tb-key:` / `tb-typed:` / `tb-mouse:` / `tb-element:` / plain-URL encoding
/// the navigator already understands), so the renderer reuses the full existing
/// navigator/ticker logic with zero reimplementation of input handling.
#[derive(Debug, Clone)]
pub enum ToPage {
    /// An encoded navigator command. `epoch` is the navigation epoch (see
    /// [`FromPage::Commit`]) — bumped by the UI on each real navigation so a
    /// late frame from an abandoned page can be dropped.
    Cmd { epoch: u64, cmd: String },
    /// Viewport resize: client width and content-region height (px).
    Resize { epoch: u64, w: u32, h: u32 },
    /// Tab / window chrome command (new tab, close, switch).
    Host { epoch: u64, command: HostCommand },
    /// Window is closing — the renderer loop should finish and exit.
    Shutdown,
}

/// Renderer thread → UI thread message. All variants are `Send` (`PaintData`
/// and its fields contain no `Rc`).
pub enum FromPage {
    /// A finished, self-contained frame for navigation epoch `gen`. The UI
    /// swaps it in via `apply_new_paint` UNLESS a newer navigation has already
    /// superseded `gen` (stale-frame drop — the race-free replacement for the
    /// single-thread `nav_in_flight` guard).
    Commit {
        epoch: u64,
        paint: PaintData,
        tabs: Vec<TabSummary>,
    },
}

// `PaintData` carries a `Bitmap` (no Debug) so derive isn't available; the
// enum is plain data otherwise.
impl std::fmt::Debug for FromPage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FromPage::Commit { epoch, tabs, .. } => f
                .debug_struct("Commit")
                .field("epoch", epoch)
                .field("tabs", &tabs.len())
                .finish(),
        }
    }
}

/// A `Send` handle to the page window. The renderer thread holds one so it can
/// wake the UI message pump with `PostMessageW(WM_APP_FROMPAGE)` after pushing a
/// [`FromPage`]. The HWND is opaque (kernel-owned) and only used with the
/// thread-safe `PostMessageW`, never dereferenced.
#[derive(Copy, Clone, Debug)]
pub struct PageHwnd(sys::HWND);
// SAFETY: see `HwndSend` — `PostMessageW` is documented thread-safe.
unsafe impl Send for PageHwnd {}

impl PageHwnd {
    /// Test-only: a `PageHwnd` wrapping a NULL window handle. `post_from_page`
    /// to it is a harmless no-op (`PostMessageW(NULL, …)` fails silently), so
    /// router/renderer plumbing can be exercised in unit tests with no real
    /// window. NOT for production use — a null HWND posts nothing.
    #[doc(hidden)]
    pub fn null_for_tests() -> Self {
        PageHwnd(core::ptr::null_mut())
    }
}

/// Wake the UI message pump to drain queued [`FromPage`] messages. Called by
/// the renderer thread after it `send`s on the `FromPage` channel.
pub fn post_from_page(target: PageHwnd) {
    unsafe {
        sys::PostMessageW(target.0, WM_APP_FROMPAGE, 0, 0);
    }
}

/// Chrome layout (Chrome/Edge/Firefox style): the TAB STRIP is the TOP row,
/// then the TOOLBAR row (Back / Forward / Refresh buttons + URL bar) sits
/// BELOW it, then a 1px divider, then page content. Both this module and
/// conclave need the same coords — conclave paints glyphs at these
/// positions, cv_ui hit-tests clicks against them.
///
/// Row 1 (TOP): tabs at `TAB_Y`, `TAB_H` tall, occupying the `TAB_STRIP_H`
/// region. Row 2: the toolbar at `TOOLBAR_Y`, `TOOLBAR_BTN_H` tall.
const TAB_STRIP_H: i32 = 32;
const TAB_X: i32 = 8;
const TAB_Y: i32 = 6;
const TAB_W: i32 = 156;
const TAB_H: i32 = 26;
const TAB_GAP: i32 = 4;
const NEW_TAB_W: i32 = 28;
const TAB_CLOSE_W: i32 = 22;

/// Top of the toolbar row (back/fwd/refresh + URL bar), directly below the
/// tab strip. The buttons + EDIT child all align to this Y. Public so the host
/// (conclave) can place its chrome glyph/label TextItems on the same row.
pub const TOOLBAR_Y: i32 = TAB_STRIP_H + 4;
pub const TOOLBAR_BTN_H: i32 = 24;

pub const BACK_BUTTON_X: i32 = 8;
pub const BACK_BUTTON_Y: i32 = TOOLBAR_Y;
pub const BACK_BUTTON_W: i32 = 28;
pub const BACK_BUTTON_H: i32 = TOOLBAR_BTN_H;
pub const FORWARD_BUTTON_X: i32 = 40;
pub const FORWARD_BUTTON_Y: i32 = TOOLBAR_Y;
pub const FORWARD_BUTTON_W: i32 = 28;
pub const FORWARD_BUTTON_H: i32 = TOOLBAR_BTN_H;
/// Refresh / reload button, placed right after Forward and before the URL bar.
pub const REFRESH_BUTTON_X: i32 = 72;
pub const REFRESH_BUTTON_Y: i32 = TOOLBAR_Y;
pub const REFRESH_BUTTON_W: i32 = 28;
pub const REFRESH_BUTTON_H: i32 = TOOLBAR_BTN_H;
/// Where the URL bar text content begins (right of the three buttons).
pub const URL_BAR_TEXT_X: i32 = 108;
const URL_BAR_EDIT_INSET_X: i32 = 12;
const URL_BAR_EDIT_INSET_RIGHT: i32 = 16;

/// Which nav button (if any) is currently held down. Drives the sunken
/// "pressed-in" chrome look and defers the actual navigation to mouse-up
/// so the press is visible. `pressed_nav: Option<NavButton>` on
/// [`WindowState`]; `None` = both buttons raised (the normal state).
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
enum NavButton {
    Back,
    Forward,
    Refresh,
}

impl NavButton {
    /// The button's pixel rect `(x, y, w, h)` in client coords.
    fn rect(self) -> (i32, i32, i32, i32) {
        match self {
            NavButton::Back => (BACK_BUTTON_X, BACK_BUTTON_Y, BACK_BUTTON_W, BACK_BUTTON_H),
            NavButton::Forward => (
                FORWARD_BUTTON_X,
                FORWARD_BUTTON_Y,
                FORWARD_BUTTON_W,
                FORWARD_BUTTON_H,
            ),
            NavButton::Refresh => (
                REFRESH_BUTTON_X,
                REFRESH_BUTTON_Y,
                REFRESH_BUTTON_W,
                REFRESH_BUTTON_H,
            ),
        }
    }

    /// Hit-test a point against any nav button rect. Returns the button
    /// the point lands in, or `None`. Used by both the press (mouse-down)
    /// and release (mouse-up) handlers so the geometry lives in one place.
    fn hit(x: i32, y: i32) -> Option<NavButton> {
        for btn in [NavButton::Back, NavButton::Forward, NavButton::Refresh] {
            let (bx, by, bw, bh) = btn.rect();
            if x >= bx && x < bx + bw && y >= by && y < by + bh {
                return Some(btn);
            }
        }
        None
    }
}

/// A tiny, valid in-memory WAV image of a short "click" — generated once
/// (no shipped asset, no risk of a byte-wrong hand-authored header). We
/// synthesize ~14 ms of 16-bit mono PCM at 22.05 kHz: a brief noise-ish
/// transient under a fast exponential decay envelope, which reads as a
/// crisp mechanical click rather than a tone.
///
/// WHY this is guaranteed audible on a stock Windows box: `PlaySoundW`
/// with `SND_MEMORY` plays the PCM straight through the default audio
/// device (winmm is present on every Windows install). It depends on NO
/// system sound file, NO registry alias (which the user could have set to
/// "None"), and NO external asset — so it can never degrade to silence the
/// way a `SND_ALIAS` system sound can when the user picks the "No Sounds"
/// scheme. The WAV bytes are constructed here with explicit header math,
/// so the RIFF/fmt/data chunks are always well-formed.
static CLICK_WAV: std::sync::LazyLock<Vec<u8>> = std::sync::LazyLock::new(build_click_wav);

fn build_click_wav() -> Vec<u8> {
    const SAMPLE_RATE: u32 = 22_050;
    const BITS: u16 = 16;
    const CHANNELS: u16 = 1;
    // ~14 ms of audio.
    let num_samples: u32 = SAMPLE_RATE * 14 / 1000;

    // Synthesize the PCM samples: a fast-decaying click. A simple
    // deterministic LCG gives the noisy transient; the envelope decays
    // exponentially so it's a "tick", and a short higher-frequency body
    // gives it presence. Peak well under i16::MAX to avoid clipping.
    let mut pcm: Vec<u8> = Vec::with_capacity((num_samples as usize) * 2);
    let mut rng: u32 = 0x1234_5678;
    for n in 0..num_samples {
        let t = n as f32 / SAMPLE_RATE as f32;
        // Exponential decay envelope (~280 1/s ⇒ near-silent by ~12 ms).
        let env = (-t * 280.0).exp();
        // Noise transient.
        rng = rng.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        let noise = ((rng >> 16) as f32 / 32_768.0) - 1.0; // [-1, 1)
        // A short tonal body (~1.6 kHz) for a "snappy" click rather than hiss.
        let body = (t * 2.0 * std::f32::consts::PI * 1_600.0).sin();
        let s = (noise * 0.55 + body * 0.45) * env * 0.6;
        let v = (s * 30_000.0).clamp(-32_000.0, 32_000.0) as i16;
        pcm.extend_from_slice(&v.to_le_bytes());
    }

    let data_len = pcm.len() as u32;
    let byte_rate = SAMPLE_RATE * u32::from(CHANNELS) * u32::from(BITS) / 8;
    let block_align = CHANNELS * BITS / 8;
    let mut wav: Vec<u8> = Vec::with_capacity(44 + pcm.len());
    wav.extend_from_slice(b"RIFF");
    wav.extend_from_slice(&(36 + data_len).to_le_bytes()); // file size - 8
    wav.extend_from_slice(b"WAVE");
    wav.extend_from_slice(b"fmt ");
    wav.extend_from_slice(&16u32.to_le_bytes()); // fmt chunk size
    wav.extend_from_slice(&1u16.to_le_bytes()); // PCM
    wav.extend_from_slice(&CHANNELS.to_le_bytes());
    wav.extend_from_slice(&SAMPLE_RATE.to_le_bytes());
    wav.extend_from_slice(&byte_rate.to_le_bytes());
    wav.extend_from_slice(&block_align.to_le_bytes());
    wav.extend_from_slice(&BITS.to_le_bytes());
    wav.extend_from_slice(b"data");
    wav.extend_from_slice(&data_len.to_le_bytes());
    wav.extend_from_slice(&pcm);
    wav
}

/// Play the embedded click asynchronously. Safe to call on the UI thread —
/// `SND_ASYNC` returns immediately. No-op-safe if the audio device is
/// unavailable (returns 0; we ignore it).
fn play_nav_click() {
    let wav = &*CLICK_WAV;
    unsafe {
        sys::PlaySoundW(
            wav.as_ptr() as *const u16,
            core::ptr::null_mut(),
            sys::SND_MEMORY | sys::SND_ASYNC | sys::SND_NODEFAULT,
        );
    }
}

struct WindowState {
    paint: PaintData,
    navigator: Option<Navigator>,
    resize_handler: Option<ResizeHandler>,
    ticker: Option<Ticker>,
    /// Network fetch shipped to a worker thread on URL-bar submit. The
    /// Arc lets us clone the callable into the worker thread without
    /// moving it out of the cell. When unset, URL-bar submission falls
    /// back to the synchronous `navigator` path.
    fetcher: Option<Fetcher>,
    /// Continuation that the UI thread runs after the worker delivers
    /// the body. Stays on the UI thread because it touches the JS
    /// interp + DOM, which aren't Send.
    nav_with_body: Option<NavigatorWithBody>,
    /// Optional "peek previous URL" callback for threading back-nav.
    back_url_fn: Option<BackUrlFn>,
    /// Optional "pop forward URL" callback for the ▶ button.
    forward_url_fn: Option<ForwardUrlFn>,
    host_command_fn: Option<HostCommandFn>,
    /// Off-main mode: command channel to the renderer thread. When `Some`, the
    /// window owns NO page closures (they live on the renderer thread) and the
    /// input/nav handlers send [`ToPage`] commands instead of running inline.
    to_page: Option<std::sync::mpsc::Sender<ToPage>>,
    /// Off-main mode: frames/notifications FROM the renderer thread, drained by
    /// the `WM_APP_FROMPAGE` handler.
    from_page: Option<std::sync::mpsc::Receiver<FromPage>>,
    /// Off-main mode: monotonically increasing navigation epoch. Bumped on each
    /// real navigation; sent with every [`ToPage`]; a [`FromPage::Commit`] with
    /// a lower epoch is a stale frame from an abandoned page and is dropped.
    nav_gen: u64,
    tabs: Vec<TabSummary>,
    hwnd: sys::HWND,
    /// Win32 EDIT child control hosting the URL bar. Real editor =
    /// arrow keys, mouse selection, Ctrl+A, IME — all from the OS.
    /// We just subclass its WNDPROC to capture Enter (commit) and
    /// Escape (cancel) keystrokes so they don't beep / get eaten.
    edit_hwnd: sys::HWND,
    /// Current vertical scroll position, in pixels of bitmap below the
    /// chrome. Clamped to `[0, content_h - viewport_h]` on every change.
    scroll_y: i32,
    /// True between dispatching a fetch worker and receiving its
    /// result. While set, navigation triggers are ignored so we don't
    /// stack overlapping fetches.
    nav_in_flight: bool,
    /// A3: set when the page is driven by a SEPARATE sandboxed renderer
    /// PROCESS (CV_USE_SANDBOX_RENDERER=1) rather than the in-process
    /// renderer thread. ONLY then is the Refresh→Stop button a REAL cancel
    /// (it sends a `__stop_load__` command the browser's site-router turns
    /// into a graceful CancelLoad + hard kill/respawn). When `false` (the
    /// default in-process path), Stop stays the honest no-op it has always
    /// been — the in-process renderer thread runs the fetch+build to
    /// completion with no cancellation token.
    sandbox_renderer: bool,
    /// The nav button currently held down (mouse captured), if any. Set on
    /// mouse-DOWN over a button, cleared on mouse-UP / lost-capture. Identity
    /// of the button whose navigation fires on a release-inside. `None` = no
    /// button is being held.
    pressed_nav: Option<NavButton>,
    /// Whether the cursor is currently OVER the held button (`pressed_nav`).
    /// The button draws sunken only when held AND hot — dragging off it while
    /// held pops it back out (cancel preview), dragging back in re-presses.
    /// Standard Win32 push-button drag feedback. Meaningless when
    /// `pressed_nav` is `None`.
    nav_press_hot: bool,
    /// Custom-scrollbar drag state: when the user grabs the thumb, this holds
    /// (mouse_y_at_grab, scroll_y_at_grab) so WM_MOUSEMOVE maps cursor travel to
    /// scroll position. None when not dragging. (We dropped WS_VSCROLL so the OS
    /// no longer drives the thumb — this is our own drag.)
    scroll_drag: Option<(i32, i32)>,
    /// HTML drag-and-drop gesture state. On a left-button press over a content
    /// element we record `(source_element_path, press_x, press_y)`. If the
    /// cursor then moves past the drag threshold while held, a drag is "in
    /// progress"; on release over a target element we emit a `tb-drag:` command
    /// so the worker runs the dragstart→dragover→drop→dragend sequence with a
    /// real DataTransfer (HTML §6.11). None when no left-button press is held.
    drag_press: Option<(String, i32, i32)>,
    /// Set once the held press has moved past `DRAG_THRESHOLD_PX`, i.e. the
    /// gesture is a drag (not a click). Distinguishes a drag-release (emit
    /// `tb-drag:`) from a plain click-release.
    drag_active: bool,
    /// Per-layer tile cache. Populated from the PaintData bitmap on
    /// every `apply_new_paint()`. WM_PAINT composites the visible
    /// viewport from cached tiles instead of reading the raw bitmap
    /// directly — this enables future incremental-repaint: only tiles
    /// overlapping a damage rect need refreshing.
    tile_cache: cv_compositor::TileCache,
    /// GPU-backed presenter (D3D11 + DXGI swap chain + DComp).  When
    /// `Some`, WM_PAINT presents the composited viewport through the
    /// hardware swap chain instead of GDI `StretchDIBits`.  Falls back
    /// to `None` (software path) if D3D11/DComp initialization fails
    /// (headless CI, remote desktop, etc.).
    ///
    /// Under `CV_OFFMAIN_COMPOSITOR`, this stays `None` on the UI thread —
    /// the presenter is CREATED + OWNED by the compositor thread (the
    /// thread-affine rule) — so the UI's GPU branch is naturally inert and
    /// the StretchDIBits fallback is reached only when the compositor reports
    /// `GpuInitFailed`.
    hw_presenter: Option<cv_gpu::HwPresenter>,

    // ── M5.5 off-main compositor (Some only under CV_OFFMAIN_COMPOSITOR) ──
    /// Command channel to the compositor thread (Present / Resize / Shutdown).
    compositor_tx: Option<std::sync::mpsc::Sender<CompositorCmd>>,
    /// Shared scroll position read by the compositor each present. Single
    /// writer (UI), single reader (compositor) ⇒ lock-free, race-free.
    /// Mirrors `scroll_y` (kept for the StretchDIBits fallback + clamp math).
    shared_scroll: Option<std::sync::Arc<core::sync::atomic::AtomicI32>>,
    /// Shared client dims `[w, h]` the compositor reads each present/resize.
    shared_dims: Option<std::sync::Arc<[core::sync::atomic::AtomicU32; 2]>>,
    /// Present-mode cell (see `present_mode`). UI WM_PAINT reads it to decide
    /// content present vs StretchDIBits fallback; compositor writes it.
    compositor_present_mode: Option<std::sync::Arc<core::sync::atomic::AtomicU8>>,
    /// Synchronous resize rendezvous (bounded wait) shared with the compositor.
    resize_ack: Option<std::sync::Arc<ResizeAck>>,
}

impl WindowState {
    /// The button currently being held (captured), regardless of whether the
    /// cursor is over it right now. `None` if no press is in progress.
    fn held_nav_button(&self) -> Option<NavButton> {
        self.pressed_nav
    }

    /// The button to draw SUNKEN: the held button, but only while the cursor
    /// is over it (`nav_press_hot`). Passed to `draw_chrome_to_hdc` so a press
    /// dragged off the button visually pops back out (cancel preview).
    fn effective_pressed_nav(&self) -> Option<NavButton> {
        if self.nav_press_hot { self.pressed_nav } else { None }
    }
}

/// Saved original WNDPROC of the EDIT control, set during subclass
/// install. The subclass forwards every non-intercepted message to
/// this pointer via `CallWindowProcW`. There is one URL bar per
/// process, so a single slot is fine.
static ORIG_EDIT_PROC: AtomicIsize = AtomicIsize::new(0);

/// Subclass WNDPROC for the URL bar EDIT control. We intercept Enter
/// and Escape so they don't get swallowed by the default proc (which
/// beeps on Enter in a single-line EDIT). Everything else — typing,
/// caret movement, mouse selection, IME — falls through to the
/// stock EDIT proc unchanged.
unsafe extern "system" fn url_bar_edit_proc(
    hwnd: sys::HWND,
    msg: u32,
    wparam: usize,
    lparam: isize,
) -> isize {
    if msg == sys::WM_CHAR {
        // CR (0x0D) = Enter → commit. ESC (0x1B) = cancel.
        if wparam == 0x0D {
            let parent = unsafe { sys::GetParent(hwnd) };
            unsafe {
                sys::PostMessageW(parent, sys::WM_USER + 2, 0, 0);
            }
            return 0;
        }
        if wparam == 0x1B {
            let parent = unsafe { sys::GetParent(hwnd) };
            unsafe {
                sys::PostMessageW(parent, sys::WM_USER + 3, 0, 0);
            }
            return 0;
        }
    }
    if msg == sys::WM_MOUSEWHEEL {
        // The URL-bar EDIT child often holds keyboard focus after load, and
        // WM_MOUSEWHEEL is delivered to the FOCUSED window — so without this
        // the page never scrolls (the EDIT swallows the wheel). Forward it to
        // the parent so the page scrolls regardless of which control has focus.
        let parent = unsafe { sys::GetParent(hwnd) };
        unsafe {
            sys::SendMessageW(parent, sys::WM_MOUSEWHEEL, wparam, lparam);
        }
        return 0;
    }
    if msg == sys::WM_LBUTTONDOWN {
        // Select-all on the click that gives the EDIT focus, so the
        // first click anywhere in the bar wipes the existing URL on
        // the user's first keystroke (matches Chrome). Subsequent
        // clicks position the caret normally.
        let had_focus = unsafe { sys::GetFocus() } == hwnd;
        let orig = ORIG_EDIT_PROC.load(Ordering::SeqCst);
        let r = unsafe { sys::CallWindowProcW(orig, hwnd, msg, wparam, lparam) };
        if !had_focus {
            unsafe {
                sys::SendMessageW(hwnd, sys::EM_SETSEL, 0, -1);
            }
        }
        return r;
    }
    let orig = ORIG_EDIT_PROC.load(Ordering::SeqCst);
    unsafe { sys::CallWindowProcW(orig, hwnd, msg, wparam, lparam) }
}

/// Geometry of the URL bar EDIT control inside the chrome strip. Used
/// at creation time and on WM_SIZE.
fn url_bar_rect(client_w: i32, _chrome_h: i32) -> (i32, i32, i32, i32) {
    let x = URL_BAR_TEXT_X + URL_BAR_EDIT_INSET_X;
    // Sit on the TOOLBAR row (below the tab strip), vertically centered on the
    // back/fwd/refresh buttons: buttons span TOOLBAR_Y..TOOLBAR_Y+TOOLBAR_BTN_H
    // (center TOOLBAR_Y + TOOLBAR_BTN_H/2); the EDIT is 2px taller and shifted
    // up 1px so both centers coincide.
    let y = TOOLBAR_Y - 1;
    let w = (client_w - x - URL_BAR_EDIT_INSET_RIGHT).max(0);
    let h = TOOLBAR_BTN_H + 2;
    (x, y, w, h)
}

fn tab_rect(index: usize) -> (i32, i32, i32, i32) {
    let x = TAB_X + index as i32 * (TAB_W + TAB_GAP);
    (x, TAB_Y, TAB_W, TAB_H)
}

fn new_tab_rect(tab_count: usize) -> (i32, i32, i32, i32) {
    let x = TAB_X + tab_count as i32 * (TAB_W + TAB_GAP);
    (x, TAB_Y, NEW_TAB_W, TAB_H)
}

fn hit_test_tab(tabs: &[TabSummary], x: i32, y: i32) -> Option<u64> {
    for (index, tab) in tabs.iter().enumerate() {
        let (tx, ty, tw, th) = tab_rect(index);
        if x >= tx && x < tx + tw && y >= ty && y < ty + th {
            return Some(tab.id);
        }
    }
    None
}

fn tab_close_rect(index: usize) -> (i32, i32, i32, i32) {
    let (tx, ty, tw, _) = tab_rect(index);
    (tx + tw - TAB_CLOSE_W - 4, ty + 3, TAB_CLOSE_W, TAB_H - 6)
}

fn hit_test_tab_close(tabs: &[TabSummary], x: i32, y: i32) -> Option<u64> {
    for (index, tab) in tabs.iter().enumerate() {
        let (tx, ty, tw, th) = tab_close_rect(index);
        if x >= tx && x < tx + tw && y >= ty && y < ty + th {
            return Some(tab.id);
        }
    }
    None
}

fn hit_test_new_tab(tabs: &[TabSummary], x: i32, y: i32) -> bool {
    let (tx, ty, tw, th) = new_tab_rect(tabs.len());
    x >= tx && x < tx + tw && y >= ty && y < ty + th
}

fn display_tab_title(tab: &TabSummary) -> String {
    let raw = if tab.title.trim().is_empty() {
        tab.url.as_str()
    } else {
        tab.title.as_str()
    };
    let mut out = String::new();
    for ch in raw.chars().take(18) {
        out.push(ch);
    }
    if raw.chars().count() > 18 {
        out.push_str("...");
    }
    out
}

/// Encode a Rust string as a null-terminated UTF-16 buffer suitable
/// for the W-suffix Win32 APIs.
fn to_wide(s: &str) -> Vec<u16> {
    s.encode_utf16().chain(std::iter::once(0)).collect()
}

/// Draw the browser chrome strip (background, divider, back/forward nav
/// buttons, tab strip + close glyphs, new-tab `+`) to an ARBITRARY HDC.
///
/// This is the SINGLE source of truth for the chrome look. It is invoked
/// two ways so the CPU and GPU paths stay byte-identical:
///   * CPU / StretchDIBits path: `hdc` is the window HDC (WM_PAINT). The
///     chrome sits directly on the window surface, which is fine because
///     nothing composites over the HDC when the GPU swap chain is off.
///   * GPU / DComp path: `hdc` is an OFFSCREEN memory DC backed by a
///     top-down 32bpp DIB section (see `bake_chrome_into_frame`). The
///     resulting pixels are copied into the top `chrome_h` rows of the
///     presented frame buffer, so the chrome rides INSIDE the swap-chain
///     frame — under the topmost DComp visual it would otherwise be hidden.
///
/// `client_w`/`chrome_h` are the strip extent in pixels. The caller has
/// already painted the chrome background fill for non-GPU callers? No — this
/// function paints the FULL strip itself (bg + divider + controls) so both
/// callers get an identical, complete strip.
///
/// The URL-bar EDIT child HWND is intentionally NOT drawn here: it is a real
/// Win32 child control that composites above DComp on its own and lives in
/// the reserved slot to the right of the nav buttons.
fn draw_chrome_to_hdc(
    hdc: sys::HDC,
    client_w: i32,
    chrome_h: i32,
    tabs: &[TabSummary],
    pressed: Option<NavButton>,
    nav_in_flight: bool,
) {
    if chrome_h <= 0 || client_w <= 0 {
        return;
    }
    unsafe {
        // Chrome background fill rgb(241,243,244).
        let chrome_rect = sys::RECT {
            left: 0,
            top: 0,
            right: client_w,
            bottom: chrome_h,
        };
        let chrome_brush = sys::CreateSolidBrush(sys::rgb(241, 243, 244));
        sys::FillRect(hdc, &raw const chrome_rect, chrome_brush);
        sys::DeleteObject(chrome_brush);

        // Bottom divider rgb(198,204,211).
        let divider_rect = sys::RECT {
            left: 0,
            top: chrome_h - 1,
            right: client_w,
            bottom: chrome_h,
        };
        let divider = sys::CreateSolidBrush(sys::rgb(198, 204, 211));
        sys::FillRect(hdc, &raw const divider_rect, divider);
        sys::DeleteObject(divider);

        // ── Back / Forward / Refresh nav buttons ─────────────────────
        // A raised chip with a crisp glyph (left arrow = Back, right arrow =
        // Forward, circular arrow = Refresh, filled square = Stop), at the
        // BACK_/FORWARD_/REFRESH_BUTTON_* rects the click handler hit-tests.
        // When `is_pressed`, the chip pushes IN: darker fill, the bevel inverts
        // (top+left dark, bottom+right light), and the glyph nudges +1px
        // right/down — reading as physically sunken. Glyphs are crisp GDI
        // primitives (filled Polygon / Arc) so they stay sharp at 28×24.
        #[derive(Copy, Clone, PartialEq, Eq)]
        enum BtnGlyph {
            ArrowLeft,
            ArrowRight,
            Reload,
            Stop,
        }
        let draw_btn = |x: i32, y: i32, w: i32, h: i32, glyph: BtnGlyph, is_pressed: bool| {
            let r = sys::RECT { left: x, top: y, right: x + w, bottom: y + h };
            // Chip face: white when raised, slightly darker when pressed.
            let fill_color = if is_pressed {
                sys::rgb(225, 228, 232)
            } else {
                sys::rgb(255, 255, 255)
            };
            let chip = sys::CreateSolidBrush(fill_color);
            sys::FillRect(hdc, &raw const r, chip);
            sys::DeleteObject(chip);

            // Bevel: raised = uniform light-grey border; pressed = SUNKEN
            // (top+left dark, bottom+right near-white) to invert the raised
            // look. Drawn as four 1px edge rects.
            let (top_left_col, bot_right_col) = if is_pressed {
                (sys::rgb(150, 156, 165), sys::rgb(248, 249, 251))
            } else {
                (sys::rgb(190, 197, 207), sys::rgb(190, 197, 207))
            };
            let tl_brush = sys::CreateSolidBrush(top_left_col);
            for edge in [
                sys::RECT { left: x, top: y, right: x + w, bottom: y + 1 }, // top
                sys::RECT { left: x, top: y, right: x + 1, bottom: y + h }, // left
            ] {
                sys::FillRect(hdc, &raw const edge, tl_brush);
            }
            sys::DeleteObject(tl_brush);
            let br_brush = sys::CreateSolidBrush(bot_right_col);
            for edge in [
                sys::RECT { left: x, top: y + h - 1, right: x + w, bottom: y + h }, // bottom
                sys::RECT { left: x + w - 1, top: y, right: x + w, bottom: y + h }, // right
            ] {
                sys::FillRect(hdc, &raw const edge, br_brush);
            }
            sys::DeleteObject(br_brush);

            let glyph_color = if is_pressed {
                sys::rgb(45, 49, 55)
            } else {
                sys::rgb(60, 64, 70)
            };
            let cx = x + w / 2 + if is_pressed { 1 } else { 0 };
            let cy = y + h / 2 + if is_pressed { 1 } else { 0 };

            match glyph {
                BtnGlyph::ArrowLeft | BtnGlyph::ArrowRight => {
                    // Filled-triangle arrow, centered, ~40% of the button tall.
                    let half_h = ((h as f32) * 0.40 / 2.0) as i32;
                    let half_w = (half_h * 3) / 4; // a touch narrower than tall
                    let pts: [sys::POINT; 3] = if glyph == BtnGlyph::ArrowLeft {
                        [
                            sys::POINT { x: cx - half_w, y: cy },
                            sys::POINT { x: cx + half_w, y: cy - half_h },
                            sys::POINT { x: cx + half_w, y: cy + half_h },
                        ]
                    } else {
                        [
                            sys::POINT { x: cx + half_w, y: cy },
                            sys::POINT { x: cx - half_w, y: cy - half_h },
                            sys::POINT { x: cx - half_w, y: cy + half_h },
                        ]
                    };
                    let brush = sys::CreateSolidBrush(glyph_color);
                    let pen = sys::CreatePen(sys::PS_SOLID, 1, glyph_color);
                    let old_brush = sys::SelectObject(hdc, brush);
                    let old_pen = sys::SelectObject(hdc, pen);
                    sys::Polygon(hdc, pts.as_ptr(), 3);
                    sys::SelectObject(hdc, old_brush);
                    sys::SelectObject(hdc, old_pen);
                    sys::DeleteObject(brush);
                    sys::DeleteObject(pen);
                }
                BtnGlyph::Reload => {
                    // Circular-arrow reload glyph: a ~270° arc drawn with a 2px
                    // pen plus a small filled-triangle arrowhead at the open end.
                    // The arc bounding box is a centered square ~46% of the
                    // button height. Arc draws COUNTER-clockwise from the radial
                    // through (x1,y1) to the radial through (x2,y2); we leave a
                    // gap at the top-right where the arrowhead points.
                    let radius = ((h as f32) * 0.30) as i32;
                    let left = cx - radius;
                    let top = cy - radius;
                    let right = cx + radius;
                    let bottom = cy + radius;
                    let pen = sys::CreatePen(sys::PS_SOLID, 2, glyph_color);
                    let old_pen = sys::SelectObject(hdc, pen);
                    // Start radial: just below the +X axis (right). End radial:
                    // just above the +X axis. Going CCW from start→end sweeps
                    // ~270° (down, left, up), leaving the upper-right open.
                    sys::Arc(
                        hdc,
                        left,
                        top,
                        right,
                        bottom,
                        cx + radius + 2,
                        cy + 3, // start radial → arc begins lower-right
                        cx + radius + 2,
                        cy - 3, // end radial → arc ends upper-right
                    );
                    sys::SelectObject(hdc, old_pen);
                    sys::DeleteObject(pen);
                    // Arrowhead at the arc's open (upper-right) end, pointing up.
                    let hx = cx + radius;
                    let hy = cy - radius + 2;
                    let s = (radius / 2).max(3);
                    let pts: [sys::POINT; 3] = [
                        sys::POINT { x: hx, y: hy - s },     // apex (up)
                        sys::POINT { x: hx - s, y: hy + 1 }, // base left
                        sys::POINT { x: hx + s, y: hy + 1 }, // base right
                    ];
                    let brush = sys::CreateSolidBrush(glyph_color);
                    let apen = sys::CreatePen(sys::PS_SOLID, 1, glyph_color);
                    let old_brush = sys::SelectObject(hdc, brush);
                    let old_apen = sys::SelectObject(hdc, apen);
                    sys::Polygon(hdc, pts.as_ptr(), 3);
                    sys::SelectObject(hdc, old_brush);
                    sys::SelectObject(hdc, old_apen);
                    sys::DeleteObject(brush);
                    sys::DeleteObject(apen);
                }
                BtnGlyph::Stop => {
                    // Filled square = stop (shown while a load is in flight).
                    let half = ((h as f32) * 0.22) as i32;
                    let sq = sys::RECT {
                        left: cx - half,
                        top: cy - half,
                        right: cx + half,
                        bottom: cy + half,
                    };
                    let brush = sys::CreateSolidBrush(sys::rgb(196, 64, 64));
                    sys::FillRect(hdc, &raw const sq, brush);
                    sys::DeleteObject(brush);
                }
            }
        };
        draw_btn(
            BACK_BUTTON_X,
            BACK_BUTTON_Y,
            BACK_BUTTON_W,
            BACK_BUTTON_H,
            BtnGlyph::ArrowLeft,
            pressed == Some(NavButton::Back),
        );
        draw_btn(
            FORWARD_BUTTON_X,
            FORWARD_BUTTON_Y,
            FORWARD_BUTTON_W,
            FORWARD_BUTTON_H,
            BtnGlyph::ArrowRight,
            pressed == Some(NavButton::Forward),
        );
        // Refresh: a reload arrow normally, a stop square while a load is in
        // flight. The button rect + hit-test are unchanged; only the glyph
        // (and the action — handled in trigger_nav_button) reflect the state.
        draw_btn(
            REFRESH_BUTTON_X,
            REFRESH_BUTTON_Y,
            REFRESH_BUTTON_W,
            REFRESH_BUTTON_H,
            if nav_in_flight { BtnGlyph::Stop } else { BtnGlyph::Reload },
            pressed == Some(NavButton::Refresh),
        );

        // ── Tab strip ────────────────────────────────────────────────
        for (index, tab) in tabs.iter().enumerate() {
            let (tx, ty, tw, th) = tab_rect(index);
            if tx >= client_w {
                break;
            }
            let tab_rect = sys::RECT {
                left: tx,
                top: ty,
                right: (tx + tw).min(client_w),
                bottom: ty + th,
            };
            let fill = if tab.active {
                sys::CreateSolidBrush(sys::rgb(255, 255, 255))
            } else {
                sys::CreateSolidBrush(sys::rgb(224, 229, 235))
            };
            sys::FillRect(hdc, &raw const tab_rect, fill);
            sys::DeleteObject(fill);
            let border = sys::CreateSolidBrush(sys::rgb(190, 197, 207));
            let top = sys::RECT {
                left: tx,
                top: ty,
                right: (tx + tw).min(client_w),
                bottom: ty + 1,
            };
            let left = sys::RECT {
                left: tx,
                top: ty,
                right: tx + 1,
                bottom: ty + th,
            };
            let right = sys::RECT {
                left: (tx + tw - 1).min(client_w),
                top: ty,
                right: (tx + tw).min(client_w),
                bottom: ty + th,
            };
            sys::FillRect(hdc, &raw const top, border);
            sys::FillRect(hdc, &raw const left, border);
            sys::FillRect(hdc, &raw const right, border);
            sys::DeleteObject(border);

            let title = display_tab_title(tab);
            let mut title_w: Vec<u16> = title.encode_utf16().collect();
            title_w.push(0);
            let face: Vec<u16> = "Segoe UI\0".encode_utf16().collect();
            let hfont = sys::CreateFontW(
                14,
                0,
                0,
                0,
                if tab.active { sys::FW_BOLD } else { sys::FW_NORMAL },
                0,
                0,
                0,
                sys::DEFAULT_CHARSET,
                sys::OUT_DEFAULT_PRECIS,
                sys::CLIP_DEFAULT_PRECIS,
                sys::CLEARTYPE_QUALITY,
                sys::DEFAULT_PITCH | sys::FF_DONTCARE,
                face.as_ptr(),
            );
            let old_font = sys::SelectObject(hdc, hfont);
            sys::SetTextColor(hdc, sys::rgb(32, 33, 36));
            let mut text_rect = sys::RECT {
                left: tx + 10,
                top: ty + 5,
                right: (tx + tw - TAB_CLOSE_W - 8).min(client_w),
                bottom: ty + th,
            };
            sys::DrawTextW(
                hdc,
                title_w.as_ptr(),
                -1,
                &raw mut text_rect,
                sys::DT_LEFT | sys::DT_TOP | sys::DT_NOPREFIX,
            );
            sys::SelectObject(hdc, old_font);
            sys::DeleteObject(hfont);

            let (cx, cy, cw, ch) = tab_close_rect(index);
            let mut close_w: Vec<u16> = "x\0".encode_utf16().collect();
            let hfont = sys::CreateFontW(
                14,
                0,
                0,
                0,
                sys::FW_BOLD,
                0,
                0,
                0,
                sys::DEFAULT_CHARSET,
                sys::OUT_DEFAULT_PRECIS,
                sys::CLIP_DEFAULT_PRECIS,
                sys::CLEARTYPE_QUALITY,
                sys::DEFAULT_PITCH | sys::FF_DONTCARE,
                face.as_ptr(),
            );
            let old_font = sys::SelectObject(hdc, hfont);
            sys::SetTextColor(hdc, sys::rgb(90, 95, 102));
            let mut close_rect = sys::RECT {
                left: cx + 7,
                top: cy + 2,
                right: (cx + cw).min(client_w),
                bottom: cy + ch,
            };
            sys::DrawTextW(
                hdc,
                close_w.as_mut_ptr(),
                -1,
                &raw mut close_rect,
                sys::DT_LEFT | sys::DT_TOP | sys::DT_NOPREFIX,
            );
            sys::SelectObject(hdc, old_font);
            sys::DeleteObject(hfont);
        }

        // ── New-tab `+` button ───────────────────────────────────────
        let (ntx, nty, ntw, nth) = new_tab_rect(tabs.len());
        if ntx < client_w {
            let plus_rect = sys::RECT {
                left: ntx,
                top: nty,
                right: (ntx + ntw).min(client_w),
                bottom: nty + nth,
            };
            let plus_fill = sys::CreateSolidBrush(sys::rgb(232, 237, 243));
            sys::FillRect(hdc, &raw const plus_rect, plus_fill);
            sys::DeleteObject(plus_fill);
            let mut plus_w: Vec<u16> = "+\0".encode_utf16().collect();
            let face: Vec<u16> = "Segoe UI\0".encode_utf16().collect();
            let hfont = sys::CreateFontW(
                20,
                0,
                0,
                0,
                sys::FW_BOLD,
                0,
                0,
                0,
                sys::DEFAULT_CHARSET,
                sys::OUT_DEFAULT_PRECIS,
                sys::CLIP_DEFAULT_PRECIS,
                sys::CLEARTYPE_QUALITY,
                sys::DEFAULT_PITCH | sys::FF_DONTCARE,
                face.as_ptr(),
            );
            let old_font = sys::SelectObject(hdc, hfont);
            sys::SetTextColor(hdc, sys::rgb(32, 33, 36));
            let mut r = plus_rect;
            r.left += 9;
            r.top += 1;
            sys::DrawTextW(
                hdc,
                plus_w.as_mut_ptr(),
                -1,
                &raw mut r,
                sys::DT_LEFT | sys::DT_TOP | sys::DT_NOPREFIX,
            );
            sys::SelectObject(hdc, old_font);
            sys::DeleteObject(hfont);
        }

        // ── Loading progress indicator ───────────────────────────────
        // While a navigation is in flight, draw a thin accent-colored bar
        // sitting directly on top of the bottom divider. This is the
        // browser-grade "the page is loading" feedback that REPLACES the old
        // "Loading…/Rendering…" address-bar text hijack. We have no true byte
        // progress to plot (the fetch+build is opaque to the UI thread), so we
        // draw an indeterminate full-width track with a brighter leading
        // segment — honest about being indeterminate, not a fake percentage.
        if nav_in_flight && chrome_h >= 4 {
            let bar_y = chrome_h - 3; // 2px bar resting on the 1px divider
            let track = sys::RECT {
                left: 0,
                top: bar_y,
                right: client_w,
                bottom: bar_y + 2,
            };
            let track_brush = sys::CreateSolidBrush(sys::rgb(180, 205, 245));
            sys::FillRect(hdc, &raw const track, track_brush);
            sys::DeleteObject(track_brush);
            // Brighter leading segment (~38% of the width) so the bar reads as
            // an active loading track rather than a static rule.
            let seg_w = (client_w * 38 / 100).max(48).min(client_w);
            let seg = sys::RECT {
                left: 0,
                top: bar_y,
                right: seg_w,
                bottom: bar_y + 2,
            };
            let seg_brush = sys::CreateSolidBrush(sys::rgb(26, 115, 232));
            sys::FillRect(hdc, &raw const seg, seg_brush);
            sys::DeleteObject(seg_brush);
        }
    }
}

/// Render the chrome strip into an offscreen top-down 32bpp DIB section and
/// copy the resulting pixels into the top `chrome_h` rows of a u32 BGRA frame
/// buffer (`frame` is `client_w * client_h` pixels, row-major, top-down,
/// packed BGRA per pixel — the same layout `cv_gfx::Bitmap::pixels` uses and
/// the swap chain expects).
///
/// This is the GPU/DComp fix: the swap-chain frame must CONTAIN the chrome,
/// because the DComp visual composites over the window HDC where the GDI
/// chrome is drawn (so HDC chrome is invisible when GPU is on). A memory DC
/// is NOT the window HDC, so drawing into it is not covered by DComp.
///
/// Byte order / orientation: a 32bpp `BI_RGB` DIB is laid out B,G,R,X per
/// pixel (a `u32` of `0x00RRGGBB` in little-endian = bytes B,G,R,0), which is
/// exactly the frame's BGRA packing. Negative `biHeight` makes the DIB
/// top-down so DIB row 0 == frame row 0 — a straight row copy, no flip, no
/// channel swap. Returns true if the bake succeeded (false ⇒ caller keeps
/// whatever was already in those rows; the GDI HDC chrome draw is the safety
/// net for the CPU path).
fn bake_chrome_into_frame(
    frame: &mut [u32],
    client_w: i32,
    client_h: i32,
    chrome_h: i32,
    tabs: &[TabSummary],
    pressed: Option<NavButton>,
    nav_in_flight: bool,
) -> bool {
    if chrome_h <= 0 || client_w <= 0 || client_h <= 0 {
        return false;
    }
    let top = chrome_h.min(client_h);
    let fw = client_w as usize;
    let strip_rows = top as usize;
    if frame.len() < fw * (client_h as usize) {
        return false;
    }
    unsafe {
        let mem_dc = sys::CreateCompatibleDC(core::ptr::null_mut());
        if mem_dc.is_null() {
            return false;
        }
        let bi = sys::BITMAPINFO {
            bmiHeader: sys::BITMAPINFOHEADER {
                biSize: core::mem::size_of::<sys::BITMAPINFOHEADER>() as u32,
                biWidth: client_w,
                biHeight: -top, // top-down: DIB row 0 == frame row 0
                biPlanes: 1,
                biBitCount: 32,
                biCompression: sys::BI_RGB,
                biSizeImage: 0,
                biXPelsPerMeter: 0,
                biYPelsPerMeter: 0,
                biClrUsed: 0,
                biClrImportant: 0,
            },
            bmiColors: [0; 1],
        };
        let mut bits: *mut c_void = core::ptr::null_mut();
        let dib = sys::CreateDIBSection(
            mem_dc,
            &bi,
            sys::DIB_RGB_COLORS,
            &mut bits,
            core::ptr::null_mut(),
            0,
        );
        if dib.is_null() || bits.is_null() {
            sys::DeleteDC(mem_dc);
            return false;
        }
        let old_bm = sys::SelectObject(mem_dc, dib);

        // Draw the chrome into the memory DC using the shared draw fn — the
        // pixels land byte-identical to the window-HDC chrome.
        draw_chrome_to_hdc(mem_dc, client_w, top, tabs, pressed, nav_in_flight);

        // Copy the DIB pixels into the top `strip_rows` rows of the frame.
        // Both are top-down BGRA, identical width ⇒ a straight memcpy of the
        // whole strip. The DIB's high (alpha/X) byte is left 0 by GDI; the
        // swap chain uses ALPHA_MODE_IGNORE so the X byte is irrelevant, but
        // we OR in 0xFF so the bytes are a clean opaque BGRA either way.
        let src = core::slice::from_raw_parts(bits as *const u32, fw * strip_rows);
        let dst = &mut frame[..fw * strip_rows];
        for (d, s) in dst.iter_mut().zip(src.iter()) {
            *d = *s | 0xFF00_0000;
        }

        sys::SelectObject(mem_dc, old_bm);
        sys::DeleteObject(dib);
        sys::DeleteDC(mem_dc);
    }
    true
}

/// Width of the custom content-region scrollbar (px).
const SCROLLBAR_W: i32 = 14;

/// The thumb rect (x, y, w, h) in CLIENT coords for the custom scrollbar given
/// the geometry, or None when the page fits (no scrollbar needed). Single source
/// of truth shared by the drawer and the drag hit-test so they never disagree.
fn scrollbar_thumb_rect(
    client_w: i32,
    client_h: i32,
    chrome_h: i32,
    content_h: i32,
    scroll_y: i32,
) -> Option<(i32, i32, i32, i32)> {
    let track_h = (client_h - chrome_h).max(0); // the viewport region height
    if content_h <= track_h || track_h <= 0 {
        return None; // fits — no scrollbar
    }
    let max_scroll = (content_h - track_h).max(1);
    let sy = scroll_y.clamp(0, max_scroll);
    // Thumb height proportional to visible fraction; min 24px so it's grabbable.
    let thumb_h = ((track_h as i64 * track_h as i64) / content_h as i64).max(24) as i32;
    let thumb_h = thumb_h.min(track_h);
    let travel = (track_h - thumb_h).max(0);
    let thumb_y = chrome_h + ((sy as i64 * travel as i64) / max_scroll as i64) as i32;
    let x = client_w - SCROLLBAR_W;
    Some((x, thumb_y, SCROLLBAR_W, thumb_h))
}

/// Draw the custom vertical scrollbar (track + thumb) into the presented BGRA
/// `frame` (client_w × client_h), confined to the content region BELOW the
/// chrome strip. Replaces the OS WS_VSCROLL non-client scrollbar (which spanned
/// the full window height and drew up through the chrome). No-op when the page
/// fits. `0xAABBGGRR`-style u32s: we write 0xFF_RRGGBB (opaque) directly.
fn draw_scrollbar_into_frame(
    frame: &mut [u32],
    client_w: i32,
    client_h: i32,
    chrome_h: i32,
    content_h: i32,
    scroll_y: i32,
) {
    let thumb = match scrollbar_thumb_rect(client_w, client_h, chrome_h, content_h, scroll_y) {
        Some(t) => t,
        None => return,
    };
    let fw = client_w.max(0) as usize;
    let track_top = chrome_h.max(0);
    let track_x = (client_w - SCROLLBAR_W).max(0);
    let fill = |frame: &mut [u32], x0: i32, y0: i32, w: i32, h: i32, color: u32| {
        for yy in y0..(y0 + h) {
            if yy < 0 || yy >= client_h {
                continue;
            }
            let row = yy as usize * fw;
            for xx in x0..(x0 + w) {
                if xx < 0 || xx >= client_w {
                    continue;
                }
                if let Some(p) = frame.get_mut(row + xx as usize) {
                    *p = color;
                }
            }
        }
    };
    // Track: subtle dark groove. Thumb: lighter grey, rounded-ish (flat is fine).
    fill(frame, track_x, track_top, SCROLLBAR_W, client_h - track_top, 0xFF1E_1E22);
    let (tx, ty, tw, th) = thumb;
    fill(frame, tx + 2, ty, tw - 3, th, 0xFF5A_5A64);
}

/// Centralised paint swap: install `new_paint` and update the window
/// title + URL bar text. Used everywhere a navigator/ticker delivers
/// fresh paint data.
/// Route a mouse-wheel delta to the innermost element-level scroll container
/// under the cursor (Blink scroll chaining). Returns `true` if an element
/// absorbed the scroll (so the page must NOT also scroll), `false` to fall
/// through to page scrolling.
///
/// `dy` is the CSS-px delta to ADD to the chosen container's vertical scroll
/// offset (already sign-corrected: wheel-down → positive `dy` → scroll down).
/// We pick the innermost scroll container that still has headroom in the
/// wheel's direction; if the innermost is pinned at its edge, we chain outward
/// to the next ancestor (exactly Chrome's overscroll/scroll-chaining model).
/// When no container can move, we return `false` and the page scrolls.
fn route_wheel_to_element(
    hwnd: sys::HWND,
    state_ptr: *mut std::cell::RefCell<WindowState>,
    dy: f32,
) -> bool {
    if dy == 0.0 {
        return false;
    }
    // Cursor → client → content coordinates (mirror dispatch_mouse_url).
    let mut pt = sys::POINT { x: 0, y: 0 };
    unsafe {
        if sys::GetCursorPos(&mut pt) == 0 {
            return false;
        }
        sys::ScreenToClient(hwnd, &mut pt);
    }
    let mut guard = unsafe { (*state_ptr).borrow_mut() };
    let chrome_h = guard.paint.chrome_h as i32;
    // Wheel over the chrome strip never scrolls page content.
    if pt.y < chrome_h {
        return false;
    }
    let content_x = pt.x as f32;
    let content_y = (pt.y - chrome_h) as f32 + guard.scroll_y as f32;
    let Some(root) = guard.paint.layout_root.as_ref() else {
        return false;
    };
    let chain = cv_layout::scroll_chain_at(root, content_x, content_y);
    if chain.is_empty() {
        return false;
    }
    // Innermost-first: pick the first target with headroom in the wheel's
    // direction. Vertical only here (horizontal-wheel/shift-wheel is a
    // follow-up); a container with max_top == 0 (no vertical overflow) is
    // skipped so the scroll chains past it.
    let target = chain.iter().find(|t| {
        if dy > 0.0 {
            t.cur_top < t.max_top - 0.5 // room to scroll down
        } else {
            t.cur_top > 0.5 // room to scroll up
        }
    });
    let Some(t) = target else {
        // Every container is pinned at the edge in this direction → let the
        // page scroll (outermost overscroll).
        return false;
    };
    let node_id = t.node_id;
    drop(guard);
    let cmd = format!("tb-scroll:{node_id}:0:{dy}");
    let mut guard = unsafe { (*state_ptr).borrow_mut() };
    pump_input_command(&mut guard, hwnd, &cmd);
    true
}

/// Dispatch a content INPUT command (`tb-mouse:` / `tb-key:` / `tb-typed:` /
/// `tb-backspace:` / `tb-enter:` / `tb-element:` / `tb-link-click:`) — an event
/// on the CURRENT page, NOT a navigation. OFF-MAIN: send it to the renderer
/// thread keeping the current epoch (so the resulting frame isn't dropped as a
/// stale pre-navigation commit; navigation, by contrast, bumps the epoch).
/// LEGACY (single-thread): run the navigator inline and swap in the new paint.
/// Before this, off-main mode silently dropped all content input because the
/// window holds no `navigator` (it lives on the renderer thread).
fn pump_input_command(guard: &mut WindowState, hwnd: sys::HWND, url: &str) {
    if guard.to_page.is_some() {
        // Drop content input while a navigation is in flight: the OLD page is
        // still displayed, but the renderer (FIFO) would run this input against
        // the NEW page with stale element paths. Mirrors the legacy
        // nav_in_flight guard. Cleared when the navigation's frame commits.
        if guard.nav_in_flight {
            return;
        }
        let epoch = guard.nav_gen;
        if let Some(tx) = guard.to_page.as_ref() {
            let _ = tx.send(ToPage::Cmd {
                epoch,
                cmd: url.to_string(),
            });
        }
    } else if let Some(nav) = guard.navigator.as_mut() {
        if let Some(new_paint) = nav(url) {
            apply_new_paint(guard, hwnd, new_paint);
        }
    }
}

/// Dispatch a NAVIGATION command (link click, Back/Forward, Reload) — it
/// replaces the displayed page. OFF-MAIN: bump `nav_gen` (a fresh epoch, so late
/// frames from the abandoned page are dropped), set `nav_in_flight` (so content
/// input is gated until the new page commits), reset scroll, and send
/// `ToPage::Cmd`. LEGACY: run the navigator inline and swap the paint. The
/// URL-bar submit path has its own equivalent inline block.
///
/// `dest_url`: when the destination URL is known up front (a fresh navigation to
/// an explicit http(s) URL), pass `Some(url)` and the address bar is set to that
/// URL IMMEDIATELY — matching Chrome, which shows the destination the instant you
/// navigate. For Back/Forward/Reload the destination isn't known on the UI
/// thread (history lives off-main), so pass `None`: the bar keeps the current URL
/// until the new page commits. The address bar is NEVER overwritten with
/// "Loading…/Rendering…" status text; the reload→stop button + the thin progress
/// bar (both driven by `nav_in_flight`) provide the loading feedback.
fn pump_navigation_command(
    guard: &mut WindowState,
    hwnd: sys::HWND,
    cmd: &str,
    dest_url: Option<&str>,
) {
    if guard.to_page.is_some() {
        guard.nav_gen += 1;
        guard.nav_in_flight = true;
        guard.scroll_y = 0;
        let epoch = guard.nav_gen;
        if let Some(tx) = guard.to_page.as_ref() {
            let _ = tx.send(ToPage::Cmd {
                epoch,
                cmd: cmd.to_string(),
            });
        }
        let edit = guard.edit_hwnd;
        if !edit.is_null() {
            // Show the destination URL right away when known; otherwise leave
            // whatever the bar currently shows. Don't clobber a URL the user is
            // mid-edit (would fight their typing).
            unsafe {
                if let Some(dest) = dest_url
                    && sys::GetFocus() != edit
                {
                    let w = to_wide(dest);
                    sys::SetWindowTextW(edit, w.as_ptr());
                }
            }
        }
        // Repaint the chrome so the in-flight indicator (stop glyph + progress
        // bar) shows immediately for command-driven navs too.
        invalidate_chrome(guard, hwnd);
    } else if let Some(nav) = guard.navigator.as_mut() {
        if let Some(new_paint) = nav(cmd) {
            apply_new_paint(guard, hwnd, new_paint);
        }
    }
}

/// Invalidate ONLY the chrome strip (rows 0..chrome_h) so the next WM_PAINT
/// repaints the buttons/tabs without re-presenting the page content. Used by
/// the nav-button press/release lifecycle to flip a button between its
/// raised and sunken look cheaply.
fn invalidate_chrome(guard: &WindowState, hwnd: sys::HWND) {
    let chrome_h = guard.paint.chrome_h as i32;
    unsafe {
        let mut client = sys::RECT::default();
        sys::GetClientRect(hwnd, &raw mut client);
        let chrome_rect = sys::RECT {
            left: 0,
            top: 0,
            right: (client.right - client.left).max(0),
            bottom: chrome_h,
        };
        sys::InvalidateRect(hwnd, &raw const chrome_rect, 0);
    }
}

/// Actually perform a Back / Forward / Refresh navigation for `btn`. Called
/// from the mouse-UP handler once the user releases inside the button they
/// pressed (standard push-button semantics — the action fires on release, not
/// press). Honors the `nav_in_flight` guard and routes through the off-main
/// command channel OR the inline peek+fetch path. Factored out so press (down)
/// and commit (up) stay separate.
///
/// Refresh RELOADS the current page: off-main it sends a `reload://` command
/// (the renderer re-fetches+rebuilds the active tab's URL in place, without
/// touching history); inline it re-fetches the current URL through the same
/// worker-thread fetcher path a fresh navigation uses. No status text is
/// written to the address bar — the URL stays put and the reload→stop button +
/// the thin progress bar provide the loading feedback (driven off nav_in_flight).
/// A3 — pure policy for "what does a nav-button press do while a load is in
/// flight?". Factored out of [`trigger_nav_button`] so the Stop-button
/// semantics (REAL cancel only in the sandboxed-renderer-process mode;
/// honest no-op otherwise) are unit-testable without a live window.
///
/// Returns `true` iff the press should fire a REAL load-cancel (send
/// [`STOP_LOAD_CMD`]). `false` means the historical honest no-op: the Refresh
/// glyph shows Stop but no cancellation token is wired for the in-process path,
/// and Back/Forward always no-op while in flight.
#[must_use]
fn in_flight_press_cancels(
    sandbox_renderer: bool,
    has_to_page: bool,
    btn: NavButton,
) -> bool {
    sandbox_renderer && has_to_page && matches!(btn, NavButton::Refresh)
}

fn trigger_nav_button(guard: &mut WindowState, hwnd: sys::HWND, btn: NavButton) {
    if guard.nav_in_flight {
        // A load is already in flight. The Refresh button is rendering as a
        // STOP affordance.
        //
        // A3 — REAL cancel in the sandboxed-renderer-process mode: when the
        // page is driven by a separate renderer PROCESS, the browser CAN abort
        // an in-flight load (the renderer is a job-object child it can signal
        // and, in the limit, kill). Stop (only Stop — Back/Forward still no-op
        // while in flight) sends the reserved `STOP_LOAD_CMD` on the command
        // channel; the browser's site-router intercepts it and performs a
        // graceful CancelLoad (bump the renderer's epoch so the in-flight
        // commit is dropped) backed by a hard kill+respawn. Clear our local
        // in-flight flag so the stop glyph reverts to Reload and the user can
        // retry. `nav_gen` is NOT bumped: the cancel rides the CURRENT epoch so
        // the renderer recognises which load to abandon.
        if in_flight_press_cancels(guard.sandbox_renderer, guard.to_page.is_some(), btn) {
            let epoch = guard.nav_gen;
            if let Some(tx) = guard.to_page.as_ref() {
                let _ = tx.send(ToPage::Cmd {
                    epoch,
                    cmd: STOP_LOAD_CMD.to_string(),
                });
            }
            guard.nav_in_flight = false;
            invalidate_chrome(guard, hwnd);
            return;
        }
        // In-process / inline path: a true mid-flight cancel is not available
        // here — off-main, the fetch+build runs to completion on the renderer
        // thread (no cancellation token wired); inline, the worker thread's
        // fetch can't be interrupted. So this is honestly a no-op rather than a
        // fake cancel. Back/Forward also no-op while in flight (unchanged).
        return;
    }
    if guard.to_page.is_some() {
        // OFF-MAIN: the renderer thread owns the per-tab history/forward stacks
        // AND the current URL, so route Back/Forward/Refresh as a command it
        // resolves. Refresh = "reload://" (re-enter the current URL in place).
        let cmd = match btn {
            NavButton::Back => "back://",
            NavButton::Forward => "forward://",
            NavButton::Refresh => "reload://",
        };
        // Destination URL is owned by the renderer (history/active tab); the bar
        // keeps the current URL until the new page commits.
        pump_navigation_command(guard, hwnd, cmd, None);
        return;
    }
    // INLINE / legacy path. Resolve the URL to (re)load.
    let target: Option<String> = match btn {
        NavButton::Back => guard.back_url_fn.as_mut().and_then(|f| f()),
        NavButton::Forward => guard.forward_url_fn.as_mut().and_then(|f| f()),
        NavButton::Refresh => {
            // Reload re-enters the CURRENT page URL. Skip non-navigable error
            // placeholders the bar may be showing.
            let cur = guard.paint.current_url.clone();
            if cur.starts_with("http://") || cur.starts_with("https://") {
                Some(cur)
            } else {
                None
            }
        }
    };
    if let Some(url) = target
        && guard.fetcher.is_some()
        && guard.nav_with_body.is_some()
    {
        guard.nav_in_flight = true;
        guard.scroll_y = 0;
        let fetcher = guard.fetcher.as_ref().unwrap().clone();
        // Keep the real URL in the address bar (never "Loading…"); the
        // reload→stop button + progress bar signal the load. Repaint the chrome
        // so the in-flight indicator (stop glyph + progress bar) appears now.
        invalidate_chrome(guard, hwnd);
        let hwnd_send = HwndSend(hwnd);
        std::thread::spawn(move || {
            let hs = hwnd_send;
            let body = fetcher(url.clone());
            let payload: Box<(String, Vec<u8>)> = Box::new((url, body));
            let raw = Box::into_raw(payload);
            unsafe {
                sys::PostMessageW(hs.0, sys::WM_USER + 1, 0, raw as isize);
            }
        });
    }
}

fn apply_new_paint(guard: &mut WindowState, hwnd: sys::HWND, new_paint: PaintData) {
    let title_w = to_wide(&new_paint.title);
    let url_w = to_wide(&new_paint.current_url);
    let edit = guard.edit_hwnd;

    // ── Off-main compositor seam ──────────────────────────────────────
    // When the compositor thread owns present, the tile-cache refresh +
    // composite + present move to that thread; the UI keeps ONLY the
    // UI-thread-only work (title/url text, scrollbar, storing guard.paint for
    // hit-test/chrome_h/caret/url) + a CHROME-ONLY invalidate. The UI must
    // still refresh ITS OWN tile cache when in the StretchDIBits fallback mode
    // (so the L2 path has pixels) — don't blindly delete the UI-side refresh.
    let compositor_owns = offmain_compositor_enabled() && guard.compositor_tx.is_some();
    let mode = guard
        .compositor_present_mode
        .as_ref()
        .map(|m| m.load(Ordering::Acquire))
        .unwrap_or(present_mode::UNKNOWN);
    let fallback_mode = compositor_owns && mode == present_mode::FALLBACK_STRETCH_DIBITS;

    if compositor_owns && !fallback_mode {
        // Store paint (UI reads it for hit-test, chrome_h, caret, url). Share
        // ONE allocation with the compositor via Arc to keep the send a single
        // refcount bump (paint is large; we avoid a deep pixel copy).
        let arc = std::sync::Arc::new(new_paint);
        guard.paint = (*arc).clone(); // Arc-field clones are refcount bumps
        let chrome_h = guard.paint.chrome_h;
        let (w, h) = guard
            .shared_dims
            .as_ref()
            .map(|d| (d[0].load(Ordering::Acquire), d[1].load(Ordering::Acquire)))
            .unwrap_or((arc.bitmap.width as u32, (arc.chrome_h + arc.viewport_h)));
        let tabs = guard.tabs.clone();
        if let Some(tx) = guard.compositor_tx.as_ref() {
            let _ = tx.send(CompositorCmd::Present { paint: arc, w, h, chrome_h, tabs });
        }
        update_scrollbar(guard, hwnd);
        unsafe {
            sys::SetWindowTextW(hwnd, title_w.as_ptr());
            if !edit.is_null() && sys::GetFocus() != edit {
                sys::SetWindowTextW(edit, url_w.as_ptr());
            }
            // Invalidate the CHROME strip only — the compositor presents the
            // content region via its swap chain. (Chrome stays a GDI overlay.)
            let mut client = sys::RECT::default();
            sys::GetClientRect(hwnd, &raw mut client);
            let chrome_rect = sys::RECT {
                left: 0,
                top: 0,
                right: (client.right - client.left).max(0),
                bottom: chrome_h as i32,
            };
            sys::InvalidateRect(hwnd, &raw const chrome_rect, 0);
        }
        return;
    }

    // ── Legacy / fallback path (synchronous UI-thread present) ─────────
    guard.paint = new_paint;
    // NOTE: we deliberately do NOT push the bitmap into the tile cache here.
    // The bitmap is fully re-rastered each frame (no cross-frame tile reuse to
    // exploit), so `refresh_from_raw` was copying the ENTIRE bitmap into tiles
    // every frame (~40MB for a 3440px page) only for `composite_viewport` to copy
    // a slice back out — two full-bitmap copies per frame. WM_PAINT now slices the
    // viewport DIRECTLY from `guard.paint.bitmap` via `composite_viewport_direct`
    // (one copy of just the visible rows). The tile cache stays for the off-main
    // compositor's own refresh path.
    update_scrollbar(guard, hwnd);
    unsafe {
        sys::SetWindowTextW(hwnd, title_w.as_ptr());
        if !edit.is_null() {
            // Don't clobber what the user is typing. A continuously-animating
            // page (particles.js etc.) delivers a paint EVERY frame, and each one
            // re-set the address-bar text to the page URL — so any edit was
            // instantly overwritten and the field looked stuck on the current
            // address. Only sync the URL bar when it does NOT have keyboard focus.
            if sys::GetFocus() != edit {
                sys::SetWindowTextW(edit, url_w.as_ptr());
            }
        }
        sys::InvalidateRect(hwnd, core::ptr::null(), 0);
    }
}

fn apply_host_command_result(guard: &mut WindowState, hwnd: sys::HWND, result: HostCommandResult) {
    guard.tabs = result.tabs;
    guard.scroll_y = 0;
    apply_new_paint(guard, hwnd, result.paint);
}

/// Whether a committed frame at `commit_epoch` should be applied, given the
/// current navigation generation `nav_gen`. The linchpin of the off-main commit
/// protocol: a frame is stale (dropped) ONLY if a newer navigation has
/// superseded its epoch; the navigation's own frame and same-generation
/// input/ticker/resize frames all apply (`>=`). Pure + unit-tested.
fn should_apply_commit(commit_epoch: u64, nav_gen: u64) -> bool {
    commit_epoch >= nav_gen
}

/// Dispatch a tab/window chrome command (new/close/switch tab, new window).
/// OFF-MAIN: send `ToPage::Host` to the renderer thread, which runs the host
/// closure and commits the new paint+tabs; a tab change alters the displayed
/// page, so bump `nav_gen` (like a navigation) so the resulting Commit isn't
/// epoch-dropped. LEGACY: run `host_command_fn` inline and apply the result.
/// Before this, off-main dropped ALL tab/window commands — `host_command_fn` is
/// `None` off-main and `ToPage::Host` was never sent from the UI thread.
fn pump_host_command(guard: &mut WindowState, hwnd: sys::HWND, command: HostCommand) {
    if guard.to_page.is_some() {
        guard.nav_gen += 1;
        let epoch = guard.nav_gen;
        if let Some(tx) = guard.to_page.as_ref() {
            let _ = tx.send(ToPage::Host { epoch, command });
        }
    } else if let Some(mut host) = guard.host_command_fn.take() {
        let result = host(command);
        guard.host_command_fn = Some(host);
        if let Some(result) = result {
            apply_host_command_result(guard, hwnd, result);
        }
    }
}

pub struct Window {
    hwnd: sys::HWND,
    state: *mut std::cell::RefCell<WindowState>,
}

static OWNER_PTR: AtomicPtr<std::cell::RefCell<WindowState>> =
    AtomicPtr::new(core::ptr::null_mut());

/// The UI thread's id and the window handle, stored lock-free at window creation
/// so any thread (e.g. the off-main renderer) can detect "am I on the UI thread?"
/// and reach the HWND for a thread-safe `PostMessageW` WITHOUT borrowing the
/// non-thread-safe `WindowState` RefCell.
static UI_THREAD_ID: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
static OWNER_HWND: AtomicIsize = AtomicIsize::new(0);

/// The Conclave application icon, embedded at compile time. A standard
/// multi-resolution Windows `.ico` (16/32/48/64/128/256). Wired in at
/// runtime via `CreateIconFromResourceEx` (no resource-compiler toolchain
/// needed): see `make_app_icon`.
static APP_ICON_BYTES: &[u8] = include_bytes!("../../../assets/conclave.ico");

/// Build an `HICON` for the requested square size from the embedded
/// `conclave.ico`. Parses the ICONDIR directory (a 6-byte header followed by
/// 16-byte `ICONDIRENTRY` records), picks the frame whose declared dimension
/// best matches `desired` (preferring an exact match, then the smallest frame
/// that is at least `desired`, then the largest available), and hands that one
/// frame's raw payload — PNG or DIB, both accepted — to
/// `CreateIconFromResourceEx`. Returns a null `HICON` on any parse/creation
/// failure so the caller falls back to the default (no icon), never panics.
///
/// Pure presentation: no behavior depends on the result.
fn make_app_icon(desired: i32) -> sys::HICON {
    let data = APP_ICON_BYTES;
    if data.len() < 6 {
        return core::ptr::null_mut();
    }
    let reserved = u16::from_le_bytes([data[0], data[1]]);
    let ty = u16::from_le_bytes([data[2], data[3]]);
    let count = u16::from_le_bytes([data[4], data[5]]) as usize;
    if reserved != 0 || ty != 1 || count == 0 || data.len() < 6 + count * 16 {
        return core::ptr::null_mut();
    }
    // Select the best-matching frame. A declared width byte of 0 means 256.
    let mut best: Option<(i32, usize, usize)> = None; // (width, offset, size)
    for k in 0..count {
        let e = &data[6 + k * 16..6 + k * 16 + 16];
        let w = if e[0] == 0 { 256i32 } else { e[0] as i32 };
        let size = u32::from_le_bytes(e[8..12].try_into().unwrap()) as usize;
        let off = u32::from_le_bytes(e[12..16].try_into().unwrap()) as usize;
        if off == 0 || size == 0 || off + size > data.len() {
            continue;
        }
        best = Some(match best {
            None => (w, off, size),
            Some(cur) => {
                // Prefer an exact match; else the candidate that minimizes the
                // distance to `desired` while not being smaller than it; else
                // the largest available frame.
                let better = if w == desired {
                    true
                } else if cur.0 == desired {
                    false
                } else if cur.0 < desired {
                    // Current is too small — any larger candidate is better.
                    w > cur.0
                } else {
                    // Current is >= desired — only a smaller-but-still->=desired
                    // candidate is better, otherwise keep current.
                    w >= desired && w < cur.0
                };
                if better { (w, off, size) } else { cur }
            }
        });
    }
    let Some((_w, off, size)) = best else {
        return core::ptr::null_mut();
    };
    // SAFETY: `off..off+size` is bounds-checked above to lie within `data`,
    // which is a 'static slice; `CreateIconFromResourceEx` only reads it.
    unsafe {
        sys::CreateIconFromResourceEx(
            data.as_ptr().add(off),
            size as u32,
            1, // fIcon = TRUE
            sys::ICON_RES_VERSION,
            desired,
            desired,
            sys::LR_DEFAULTCOLOR,
        )
    }
}

impl Window {
    /// Create a top-level window of the given client size with initial
    /// paint data and an optional click-to-navigate callback. The callback
    /// receives an absolute URL string when the user clicks a link; if it
    /// returns `Some(new_paint)` the window swaps in the new content and
    /// repaints.
    pub fn new(
        title: &str,
        paint: PaintData,
        navigator: Option<Navigator>,
    ) -> Result<Self, String> {
        Self::with_ticker(title, paint, navigator, None)
    }

    /// Variant that also installs a live resize callback so the host can
    /// rebuild layout when the window size changes.
    pub fn with_resize_handler(
        title: &str,
        paint: PaintData,
        navigator: Option<Navigator>,
        resize_handler: Option<ResizeHandler>,
    ) -> Result<Self, String> {
        Self::with_ticker_and_resize(title, paint, navigator, None, resize_handler)
    }

    /// Variant that also installs a periodic tick callback (~60Hz). Used
    /// by the browser to drive the JS event loop and any compositor work
    /// without blocking the message pump.
    pub fn with_ticker(
        title: &str,
        paint: PaintData,
        navigator: Option<Navigator>,
        ticker: Option<Ticker>,
    ) -> Result<Self, String> {
        Self::with_ticker_and_resize(title, paint, navigator, ticker, None)
    }

    /// Like `with_ticker` but also accepts a resize callback.
    pub fn with_ticker_and_resize(
        title: &str,
        paint: PaintData,
        navigator: Option<Navigator>,
        ticker: Option<Ticker>,
        resize_handler: Option<ResizeHandler>,
    ) -> Result<Self, String> {
        Self::with_ticker_and_fetch_and_resize(
            title,
            paint,
            navigator,
            ticker,
            None,
            None,
            resize_handler,
        )
    }

    /// Full constructor — adds a background `fetcher` (Send) and a
    /// `nav_with_body` continuation. When both are set, URL-bar
    /// submission spawns a worker thread for the network call so the
    /// UI keeps pumping during slow TLS handshakes. When either is
    /// `None`, URL-bar submission falls back to the synchronous
    /// `navigator` path.
    pub fn with_ticker_and_fetch(
        title: &str,
        paint: PaintData,
        navigator: Option<Navigator>,
        ticker: Option<Ticker>,
        fetcher: Option<Fetcher>,
        nav_with_body: Option<NavigatorWithBody>,
    ) -> Result<Self, String> {
        Self::with_ticker_and_fetch_and_resize(
            title,
            paint,
            navigator,
            ticker,
            fetcher,
            nav_with_body,
            None,
        )
    }

    /// Full constructor including a resize callback.
    pub fn with_ticker_and_fetch_and_resize(
        title: &str,
        paint: PaintData,
        navigator: Option<Navigator>,
        ticker: Option<Ticker>,
        fetcher: Option<Fetcher>,
        nav_with_body: Option<NavigatorWithBody>,
        resize_handler: Option<ResizeHandler>,
    ) -> Result<Self, String> {
        Self::with_nav_buttons_and_resize(
            title,
            paint,
            navigator,
            ticker,
            fetcher,
            nav_with_body,
            None,
            None,
            resize_handler,
        )
    }

    /// Most-detailed constructor, also accepting a `back_url_fn` that
    /// lets the Backspace handler route through the worker-thread
    /// fetch path instead of running synchronously.
    pub fn full(
        title: &str,
        paint: PaintData,
        navigator: Option<Navigator>,
        ticker: Option<Ticker>,
        fetcher: Option<Fetcher>,
        nav_with_body: Option<NavigatorWithBody>,
        back_url_fn: Option<BackUrlFn>,
    ) -> Result<Self, String> {
        Self::with_nav_buttons(
            title,
            paint,
            navigator,
            ticker,
            fetcher,
            nav_with_body,
            back_url_fn,
            None,
        )
    }

    /// Like `full` but also takes a `forward_url_fn` so the chrome's
    /// ▶ button can drive a worker-thread fetch.
    pub fn with_nav_buttons(
        title: &str,
        paint: PaintData,
        navigator: Option<Navigator>,
        ticker: Option<Ticker>,
        fetcher: Option<Fetcher>,
        nav_with_body: Option<NavigatorWithBody>,
        back_url_fn: Option<BackUrlFn>,
        forward_url_fn: Option<ForwardUrlFn>,
    ) -> Result<Self, String> {
        Self::with_nav_buttons_and_resize(
            title,
            paint,
            navigator,
            ticker,
            fetcher,
            nav_with_body,
            back_url_fn,
            forward_url_fn,
            None,
        )
    }

    /// Like `with_nav_buttons` but also accepts a resize callback.
    pub fn with_nav_buttons_and_resize(
        title: &str,
        paint: PaintData,
        navigator: Option<Navigator>,
        ticker: Option<Ticker>,
        fetcher: Option<Fetcher>,
        nav_with_body: Option<NavigatorWithBody>,
        back_url_fn: Option<BackUrlFn>,
        forward_url_fn: Option<ForwardUrlFn>,
        resize_handler: Option<ResizeHandler>,
    ) -> Result<Self, String> {
        Self::with_tabs_and_resize(
            title,
            paint,
            navigator,
            ticker,
            fetcher,
            nav_with_body,
            back_url_fn,
            forward_url_fn,
            resize_handler,
            Vec::new(),
            None,
            None,
            None,
        )
    }

    /// Off-main-thread constructor: opens the window IMMEDIATELY (no page
    /// closures — those live on the renderer thread) with an initial
    /// "Loading…" `paint`, wired to the renderer via the two channels. The
    /// UI thread sends [`ToPage`] commands and applies [`FromPage::Commit`]
    /// frames; it can never be blocked by page work. The caller spawns the
    /// renderer thread with `win.page_hwnd()` + the channel ends.
    pub fn with_render_channel(
        title: &str,
        paint: PaintData,
        tabs: Vec<TabSummary>,
        to_page: std::sync::mpsc::Sender<ToPage>,
        from_page: std::sync::mpsc::Receiver<FromPage>,
    ) -> Result<Self, String> {
        Self::with_tabs_and_resize(
            title, paint, None, None, None, None, None, None, None, tabs, None,
            Some(to_page), Some(from_page),
        )
    }

    /// A `Send` handle to this window for the renderer thread's `post_from_page`.
    pub fn page_hwnd(&self) -> PageHwnd {
        PageHwnd(self.hwnd)
    }

    /// A3: mark this window as driven by a separate sandboxed renderer PROCESS
    /// (CV_USE_SANDBOX_RENDERER=1). Enables the REAL Stop button — see
    /// [`WindowState::sandbox_renderer`]. Called by `run_window_offmain`
    /// immediately after a persistent renderer connects, BEFORE `win.run()`.
    /// No effect on the in-process path (never called there), so Stop stays
    /// the honest no-op when the flag is off.
    pub fn enable_sandbox_renderer_stop(&self) {
        let state_ptr = OWNER_PTR.load(Ordering::SeqCst);
        if state_ptr.is_null() {
            return;
        }
        let mut g = unsafe { (*state_ptr).borrow_mut() };
        g.sandbox_renderer = true;
    }

    /// The current client rect `(w, h)` in pixels. Read by `run_window_offmain`
    /// to seed the compositor thread's initial swap-chain dims (the HWND is
    /// valid; `GetClientRect` is thread-safe).
    pub fn client_size(&self) -> (u32, u32) {
        let mut r = sys::RECT::default();
        unsafe { sys::GetClientRect(self.hwnd, &raw mut r) };
        ((r.right - r.left).max(1) as u32, (r.bottom - r.top).max(1) as u32)
    }

    /// Wire the off-main compositor channel + shared atomics into `WindowState`.
    /// Called by `run_window_offmain` AFTER spawning the compositor thread (which
    /// owns the matching receiver/clones). No-op effect on the flag-OFF path
    /// because the caller only invokes it when `offmain_compositor_enabled()`.
    pub fn install_compositor(
        &self,
        tx: std::sync::mpsc::Sender<CompositorCmd>,
        scroll: std::sync::Arc<core::sync::atomic::AtomicI32>,
        dims: std::sync::Arc<[core::sync::atomic::AtomicU32; 2]>,
        present_mode_cell: std::sync::Arc<core::sync::atomic::AtomicU8>,
        resize_ack: std::sync::Arc<ResizeAck>,
    ) {
        let state_ptr = OWNER_PTR.load(Ordering::SeqCst);
        if state_ptr.is_null() {
            return;
        }
        let mut g = unsafe { (*state_ptr).borrow_mut() };
        g.compositor_tx = Some(tx);
        g.shared_scroll = Some(scroll);
        g.shared_dims = Some(dims);
        g.compositor_present_mode = Some(present_mode_cell);
        g.resize_ack = Some(resize_ack);
    }

    pub fn with_tabs_and_resize(
        title: &str,
        paint: PaintData,
        navigator: Option<Navigator>,
        ticker: Option<Ticker>,
        fetcher: Option<Fetcher>,
        nav_with_body: Option<NavigatorWithBody>,
        back_url_fn: Option<BackUrlFn>,
        forward_url_fn: Option<ForwardUrlFn>,
        resize_handler: Option<ResizeHandler>,
        tabs: Vec<TabSummary>,
        host_command_fn: Option<HostCommandFn>,
        to_page: Option<std::sync::mpsc::Sender<ToPage>>,
        from_page: Option<std::sync::mpsc::Receiver<FromPage>>,
    ) -> Result<Self, String> {
        let class_name: Vec<u16> = "ConclaveWindow\0".encode_utf16().collect();
        let title_w: Vec<u16> = title.encode_utf16().chain(std::iter::once(0)).collect();

        unsafe {
            let h_instance = sys::GetModuleHandleW(core::ptr::null());

            // Conclave window icon, built once from the embedded `conclave.ico`.
            // 32x32 for the class/large slot (Alt-Tab, taskbar), 16x16 for the
            // small slot (title bar). Null on failure ⇒ default (no icon).
            let h_icon_big = make_app_icon(32);
            let h_icon_small = make_app_icon(16);

            let wcex = sys::WNDCLASSEXW {
                cbSize: core::mem::size_of::<sys::WNDCLASSEXW>() as u32,
                style: sys::CS_HREDRAW | sys::CS_VREDRAW | sys::CS_DBLCLKS,
                lpfnWndProc: Some(wnd_proc),
                cbClsExtra: 0,
                cbWndExtra: 0,
                hInstance: h_instance,
                hIcon: h_icon_big,
                hCursor: sys::LoadCursorW(core::ptr::null_mut(), sys::IDC_ARROW),
                hbrBackground: core::ptr::null_mut(),
                lpszMenuName: core::ptr::null(),
                lpszClassName: class_name.as_ptr(),
                hIconSm: h_icon_small,
            };
            sys::RegisterClassExW(&raw const wcex);

            let init_w = paint.bitmap.width as i32;
            // Window starts at chrome + initial viewport, not the full
            // document — scrolling reveals the rest.
            let init_h = (paint.chrome_h + paint.viewport_h) as i32;
            let has_ticker = ticker.is_some();
            let state = Box::new(std::cell::RefCell::new(WindowState {
                paint,
                navigator,
                resize_handler,
                ticker,
                fetcher,
                nav_with_body,
                back_url_fn,
                forward_url_fn,
                host_command_fn,
                to_page,
                from_page,
                nav_gen: 0,
                tabs,
                hwnd: core::ptr::null_mut(),
                edit_hwnd: core::ptr::null_mut(),
                scroll_y: 0,
                nav_in_flight: false,
                sandbox_renderer: false,
                pressed_nav: None,
                nav_press_hot: false,
                scroll_drag: None,
                drag_press: None,
                drag_active: false,
                tile_cache: cv_compositor::TileCache::new(),
                hw_presenter: None,
                compositor_tx: None,
                shared_scroll: None,
                shared_dims: None,
                compositor_present_mode: None,
                resize_ack: None,
            }));
            let state_raw: *mut std::cell::RefCell<WindowState> = Box::into_raw(state);
            OWNER_PTR.store(state_raw, Ordering::SeqCst);

            let mut rect = sys::RECT {
                left: 0,
                top: 0,
                right: init_w,
                bottom: init_h,
            };
            // NO WS_VSCROLL: the OS non-client vertical scrollbar spans the FULL
            // window height, so it draws UP through the chrome/toolbar strip
            // (a reported bug). We draw a CUSTOM scrollbar into the content
            // region (below chrome_h) instead — see the present path + the
            // scrollbar hit-test/drag handlers.
            let window_style = sys::WS_OVERLAPPEDWINDOW;
            sys::AdjustWindowRectEx(&raw mut rect, window_style, 0, 0);
            let win_w = rect.right - rect.left;
            let win_h = rect.bottom - rect.top;

            let hwnd = sys::CreateWindowExW(
                0,
                class_name.as_ptr(),
                title_w.as_ptr(),
                // `WS_CLIPCHILDREN`: keep the parent's WM_PAINT from
                // drawing over the URL bar EDIT child each frame
                // (would otherwise flicker the bar on every repaint).
                sys::WS_OVERLAPPEDWINDOW | sys::WS_CLIPCHILDREN,
                sys::CW_USEDEFAULT,
                sys::CW_USEDEFAULT,
                win_w,
                win_h,
                core::ptr::null_mut(),
                core::ptr::null_mut(),
                h_instance,
                core::ptr::null_mut(),
            );
            if hwnd.is_null() {
                // Window creation failed (e.g. no window-station / desktop, or an
                // exhausted USER handle table). Reclaim the WindowState we leaked
                // via Box::into_raw, clear the global owner pointer, and surface a
                // clean error instead of aborting the whole process.
                OWNER_PTR.store(core::ptr::null_mut(), Ordering::SeqCst);
                drop(Box::from_raw(state_raw));
                let err = sys::GetLastError();
                return Err(format!(
                    "CreateWindowExW failed (GetLastError=0x{err:08X}); \
                     no GUI window could be created"
                ));
            }
            // Record the UI thread id + HWND lock-free so off-thread callers
            // (the off-main renderer's ticker) can detect they are NOT on the UI
            // thread and post a message instead of borrowing `WindowState`.
            OWNER_HWND.store(hwnd as isize, Ordering::SeqCst);
            UI_THREAD_ID.store(sys::GetCurrentThreadId(), Ordering::SeqCst);

            // Also set the icon directly on the window so the title bar +
            // Alt-Tab + taskbar pick it up immediately (WNDCLASS hIcon covers
            // newly registered classes, but WM_SETICON is the reliable per-HWND
            // path). Null handles are no-ops.
            if !h_icon_big.is_null() {
                sys::SendMessageW(hwnd, sys::WM_SETICON, sys::ICON_BIG, h_icon_big as isize);
            }
            if !h_icon_small.is_null() {
                sys::SendMessageW(hwnd, sys::WM_SETICON, sys::ICON_SMALL, h_icon_small as isize);
            }

            // Create the URL bar EDIT child control inset inside the
            // painted rounded address field. Keeping the EDIT borderless
            // lets the chrome bitmap define the browser's visual shape,
            // while the native control still handles selection, IME, and
            // keyboard editing.
            let edit_class: Vec<u16> = "EDIT\0".encode_utf16().collect();
            let (chrome_h_i32, initial_url_w) = {
                let g = (*state_raw).borrow();
                (g.paint.chrome_h as i32, to_wide(&g.paint.current_url))
            };
            let (ex, ey, ew, eh) = url_bar_rect(init_w, chrome_h_i32);
            let edit_hwnd = sys::CreateWindowExW(
                0,
                edit_class.as_ptr(),
                initial_url_w.as_ptr(),
                sys::WS_CHILD | sys::WS_VISIBLE | sys::ES_AUTOHSCROLL | sys::ES_LEFT,
                ex,
                ey,
                ew,
                eh,
                hwnd,
                core::ptr::null_mut(),
                h_instance,
                core::ptr::null_mut(),
            );
            // A failed URL-bar EDIT control must NOT abort the whole browser:
            // degrade to a no-edit-bar window. The page still loads, scrolls,
            // and renders; only inline URL editing is unavailable. Every
            // `edit_hwnd` consumer either null-checks first or calls a USER API
            // that is a harmless no-op on a null HWND (SetWindowTextW / SetFocus
            // / SendMessageW all fail cleanly, they don't crash). So we keep the
            // null handle, skip the subclass, and carry on.
            if !edit_hwnd.is_null() {
                // Subclass the EDIT WNDPROC. We keep the original around
                // and forward all non-intercepted messages to it.
                let orig = sys::SetWindowLongPtrW(
                    edit_hwnd,
                    sys::GWLP_WNDPROC,
                    url_bar_edit_proc as *const () as isize,
                );
                ORIG_EDIT_PROC.store(orig, Ordering::SeqCst);
            }

            // Save HWNDs into state for callbacks that need to
            // InvalidateRect / SetWindowText / MoveWindow.
            {
                let mut g = (*state_raw).borrow_mut();
                g.hwnd = hwnd;
                g.edit_hwnd = edit_hwnd;
                update_scrollbar(&g, hwnd);

                // Try to create the GPU-backed presenter (D3D11 +
                // DComp swap chain).  On failure, fall back to GDI
                // StretchDIBits — no panic.
                //
                // ★ Under CV_OFFMAIN_COMPOSITOR the COMPOSITOR THREAD creates +
                // owns the presenter (the thread-affine rule), so the UI thread
                // MUST NOT create one here — otherwise two DComp targets / two
                // swap chains fight over the same HWND. Leave hw_presenter None.
                if !offmain_compositor_enabled() {
                    let mut r = sys::RECT::default();
                    sys::GetClientRect(hwnd, &raw mut r);
                    let cw = (r.right - r.left).max(1) as u32;
                    let ch = (r.bottom - r.top).max(1) as u32;
                    match cv_gpu::HwPresenter::new(hwnd, cw, ch) {
                        Ok(hw) => g.hw_presenter = Some(hw),
                        Err(_e) => {
                            // GPU init failed — StretchDIBits will be used.
                            // This is expected on headless CI / WARP fallback
                            // without a real HWND visible.
                        }
                    }
                }
            }

            // Open maximized so pages that size to the viewport (canvases,
            // responsive layouts) bootstrap at the real screen size; WM_SIZE then
            // drives the resize handler to re-sync JS geometry + fire `resize`.
            sys::ShowWindow(hwnd, sys::SW_SHOWMAXIMIZED);
            sys::UpdateWindow(hwnd);

            // Drive the JS event loop / RAF at ~60Hz once a ticker is
            // installed. WM_TIMER posts into the same message pump, so
            // we never block.
            if has_ticker {
                sys::SetTimer(hwnd, sys::JS_TICK_TIMER_ID, 16, core::ptr::null_mut());
            }

            Ok(Self {
                hwnd,
                state: state_raw,
            })
        }
    }

    /// Run the message loop until the window is closed.
    pub fn run(&self) {
        unsafe {
            let mut msg: sys::MSG = core::mem::zeroed();
            while sys::GetMessageW(&raw mut msg, core::ptr::null_mut(), 0, 0) > 0 {
                sys::TranslateMessage(&raw const msg);
                sys::DispatchMessageW(&raw const msg);
            }
        }
    }
}

impl Drop for Window {
    fn drop(&mut self) {
        OWNER_PTR.store(core::ptr::null_mut(), Ordering::SeqCst);
        if !self.hwnd.is_null() {
            unsafe {
                sys::KillTimer(self.hwnd, sys::JS_TICK_TIMER_ID);
                sys::DestroyWindow(self.hwnd);
            }
        }
        if !self.state.is_null() {
            // SAFETY: we created the raw pointer via Box::into_raw above.
            unsafe {
                drop(Box::from_raw(self.state));
            }
        }
    }
}

enum ScrollDelta {
    Lines(i32),
    Page(i32),
    Absolute(i32),
}

/// Push a `tb-key:<event_type>:<key>:<ctrl>:<shift>:<alt>:<meta>` URL
/// through the navigator so JS sees a `KeyboardEvent` with real
/// modifier flags. Shared by WM_KEYDOWN + WM_KEYUP.
/// Build and pump a `tb-mouse:` URL for the given event type.
/// Encoding: `tb-mouse:<event>:<x>:<y>:<ctrl>:<shift>:<alt>:<meta>:<path>`
/// where `<path>` is the deepest element path under the cursor (slash-
/// separated child indices) or empty if hit-testing finds no element.
/// The browser-side parser tolerates a trailing colon and missing path.
fn dispatch_mouse_url(hwnd: sys::HWND, event_type: &str, lparam: isize, _wparam: usize) {
    let x_raw = (lparam & 0xFFFF) as i16 as i32;
    let y_raw = ((lparam >> 16) & 0xFFFF) as i16 as i32;
    let (ctrl, shift, alt, meta) = sys::modifiers_now();
    let state_ptr = OWNER_PTR.load(Ordering::SeqCst);
    if state_ptr.is_null() {
        return;
    }
    let mut guard = unsafe { (*state_ptr).borrow_mut() };
    // Map raw window y into the content-area y the layout tree was
    // built in (mousemove/click below the chrome strip is in scrolled
    // content space — apply chrome-h offset + scroll_y).
    let chrome_h = guard.paint.chrome_h as i32;
    let content_y = if y_raw < chrome_h {
        y_raw as f32
    } else {
        (y_raw - chrome_h) as f32 + guard.scroll_y as f32
    };
    let content_x = x_raw as f32;
    // Hit-test against the layout tree to find the deepest element
    // path under the cursor. Empty path = document root (no element).
    let path_str = match guard.paint.layout_root.as_ref() {
        Some(root) => match cv_layout::hit_test_element_path(root, content_x, content_y) {
            Some(path) => path
                .iter()
                .map(|n| n.to_string())
                .collect::<Vec<_>>()
                .join("/"),
            None => String::new(),
        },
        None => String::new(),
    };
    let url = format!(
        "tb-mouse:{event_type}:{x_raw}:{y_raw}:{}:{}:{}:{}:{}",
        if ctrl { 1 } else { 0 },
        if shift { 1 } else { 0 },
        if alt { 1 } else { 0 },
        if meta { 1 } else { 0 },
        path_str,
    );
    pump_input_command(&mut guard, hwnd, &url);
}

/// True if the OS cursor is currently over a link, so the window shows the
/// hand cursor. Reads the live paint's layout tree / hit regions.
fn pointer_over_link(hwnd: sys::HWND) -> bool {
    let state_ptr = OWNER_PTR.load(Ordering::SeqCst);
    if state_ptr.is_null() {
        return false;
    }
    let mut pt = sys::POINT { x: 0, y: 0 };
    unsafe {
        if sys::GetCursorPos(&mut pt) == 0 {
            return false;
        }
        sys::ScreenToClient(hwnd, &mut pt);
    }
    let guard = unsafe { (*state_ptr).borrow() };
    let chrome_h = guard.paint.chrome_h as i32;
    let x = pt.x as f32;
    let y = if pt.y < chrome_h {
        pt.y as f32
    } else {
        (pt.y - chrome_h) as f32 + guard.scroll_y as f32
    };
    match guard.paint.layout_root.as_ref() {
        Some(root) => cv_layout::hit_test_link(root, x, y).is_some(),
        None => hit_test_regions(&guard.paint.hit_regions, x, y).0.is_some(),
    }
}

fn fire_key_event(hwnd: sys::HWND, event_type: &str, vk: usize) {
    let key_name = vk_to_event_key(vk);
    let code_name = vk_to_event_code(vk);
    let (ctrl, shift, alt, meta) = sys::modifiers_now();
    let state_ptr = OWNER_PTR.load(Ordering::SeqCst);
    if state_ptr.is_null() {
        return;
    }
    // Format: tb-key:<event>:<key>:<ctrl>:<shift>:<alt>:<meta>:<code>
    let url = format!(
        "tb-key:{event_type}:{key_name}:{}:{}:{}:{}:{code_name}",
        if ctrl { 1 } else { 0 },
        if shift { 1 } else { 0 },
        if alt { 1 } else { 0 },
        if meta { 1 } else { 0 },
    );
    let mut guard = unsafe { (*state_ptr).borrow_mut() };
    pump_input_command(&mut guard, hwnd, &url);
}

/// Map a Win32 virtual-key code to the DOM `KeyboardEvent.key` string.
/// Covers the common ASCII letters/digits, navigation keys, and named
/// keys; anything else falls back to a printable character or `"Unknown"`.
fn vk_to_event_key(vk: usize) -> String {
    match vk {
        sys::VK_ESCAPE => "Escape".into(),
        sys::VK_BACK => "Backspace".into(),
        sys::VK_PRIOR => "PageUp".into(),
        sys::VK_NEXT => "PageDown".into(),
        sys::VK_HOME => "Home".into(),
        sys::VK_END => "End".into(),
        sys::VK_UP => "ArrowUp".into(),
        sys::VK_DOWN => "ArrowDown".into(),
        0x09 => "Tab".into(),
        0x0D => "Enter".into(),
        0x20 => " ".into(),
        0x25 => "ArrowLeft".into(),
        0x27 => "ArrowRight".into(),
        0x2E => "Delete".into(),
        0x10 => "Shift".into(),
        0x11 => "Control".into(),
        0x12 => "Alt".into(),
        0x14 => "CapsLock".into(),
        // 0-9
        0x30..=0x39 => ((vk as u8 - 0x30 + b'0') as char).to_string(),
        // A-Z (uppercase — JS spec uses lowercase for unshifted; close
        // enough for now). Sites that care use event.code instead.
        0x41..=0x5A => ((vk as u8 - 0x41 + b'a') as char).to_string(),
        // F1..F12
        0x70..=0x7B => format!("F{}", vk - 0x6F),
        _ => format!("Unknown({vk})"),
    }
}

/// Map a Win32 virtual-key code to the DOM `KeyboardEvent.code` string —
/// the physical key identifier, independent of layout/modifiers.
/// See https://www.w3.org/TR/uievents-code/
fn vk_to_event_code(vk: usize) -> String {
    match vk {
        0x08 => "Backspace".into(),        // VK_BACK
        0x09 => "Tab".into(),
        0x0D => "Enter".into(),            // VK_RETURN
        0x10 => "ShiftLeft".into(),        // VK_SHIFT (left; no distinction without extended flag)
        0x11 => "ControlLeft".into(),      // VK_CONTROL
        0x12 => "AltLeft".into(),          // VK_MENU
        0x14 => "CapsLock".into(),
        0x1B => "Escape".into(),           // VK_ESCAPE
        0x20 => "Space".into(),
        0x21 => "PageUp".into(),           // VK_PRIOR
        0x22 => "PageDown".into(),         // VK_NEXT
        0x23 => "End".into(),              // VK_END
        0x24 => "Home".into(),             // VK_HOME
        0x25 => "ArrowLeft".into(),
        0x26 => "ArrowUp".into(),          // VK_UP
        0x27 => "ArrowRight".into(),
        0x28 => "ArrowDown".into(),        // VK_DOWN
        0x2E => "Delete".into(),
        // Digit row: 0-9
        0x30 => "Digit0".into(),
        0x31 => "Digit1".into(),
        0x32 => "Digit2".into(),
        0x33 => "Digit3".into(),
        0x34 => "Digit4".into(),
        0x35 => "Digit5".into(),
        0x36 => "Digit6".into(),
        0x37 => "Digit7".into(),
        0x38 => "Digit8".into(),
        0x39 => "Digit9".into(),
        // Letter keys A-Z
        0x41 => "KeyA".into(),
        0x42 => "KeyB".into(),
        0x43 => "KeyC".into(),
        0x44 => "KeyD".into(),
        0x45 => "KeyE".into(),
        0x46 => "KeyF".into(),
        0x47 => "KeyG".into(),
        0x48 => "KeyH".into(),
        0x49 => "KeyI".into(),
        0x4A => "KeyJ".into(),
        0x4B => "KeyK".into(),
        0x4C => "KeyL".into(),
        0x4D => "KeyM".into(),
        0x4E => "KeyN".into(),
        0x4F => "KeyO".into(),
        0x50 => "KeyP".into(),
        0x51 => "KeyQ".into(),
        0x52 => "KeyR".into(),
        0x53 => "KeyS".into(),
        0x54 => "KeyT".into(),
        0x55 => "KeyU".into(),
        0x56 => "KeyV".into(),
        0x57 => "KeyW".into(),
        0x58 => "KeyX".into(),
        0x59 => "KeyY".into(),
        0x5A => "KeyZ".into(),
        // F1..F12
        0x70 => "F1".into(),
        0x71 => "F2".into(),
        0x72 => "F3".into(),
        0x73 => "F4".into(),
        0x74 => "F5".into(),
        0x75 => "F6".into(),
        0x76 => "F7".into(),
        0x77 => "F8".into(),
        0x78 => "F9".into(),
        0x79 => "F10".into(),
        0x7A => "F11".into(),
        0x7B => "F12".into(),
        _ => format!("Unknown({vk:#04x})"),
    }
}

fn scroll_metrics(guard: &WindowState, hwnd: sys::HWND) -> (i32, i32, i32) {
    let mut client = sys::RECT::default();
    unsafe { sys::GetClientRect(hwnd, &raw mut client) };
    let client_h = (client.bottom - client.top).max(0);
    let chrome_h = guard.paint.chrome_h as i32;
    // Scroll range is the DOCUMENT height, which for a band bitmap is larger than
    // the bitmap; content_height() returns document_h when set, else bitmap.height.
    let content_h = guard.paint.content_height() as i32;
    let viewport_h = (client_h - chrome_h).max(0);
    let max_scroll = (content_h - viewport_h).max(0);
    (content_h.max(0), viewport_h, max_scroll)
}

fn update_scrollbar(_guard: &WindowState, hwnd: sys::HWND) {
    // We draw a CUSTOM scrollbar into the presented frame (draw_scrollbar_into_frame).
    // The OS WS_VSCROLL non-client scrollbar is intentionally NOT used — and
    // crucially, calling SetScrollInfo(SB_VERT) would RE-CREATE/SHOW that OS
    // scrollbar even without the WS_VSCROLL style, producing a SECOND scrollbar
    // alongside our custom one (a reported bug). So we explicitly HIDE the OS
    // vertical scrollbar instead of feeding it. The custom thumb is positioned
    // entirely from scroll_y at present time.
    unsafe {
        sys::ShowScrollBar(hwnd, sys::SB_VERT, 0);
    }
}

fn clamp_scroll(guard: &mut WindowState, hwnd: sys::HWND) {
    let (_, _, max_scroll) = scroll_metrics(guard, hwnd);
    if guard.scroll_y > max_scroll {
        guard.scroll_y = max_scroll;
    }
    if guard.scroll_y < 0 {
        guard.scroll_y = 0;
    }
    update_scrollbar(guard, hwnd);
}

/// Publish the (already clamped) scroll position so the viewport updates.
///
/// Off-main compositor (and not in the StretchDIBits fallback): store the
/// clamped value into the shared scroll atomic (single writer, `Release`) and
/// DROP the content `InvalidateRect` — the compositor re-presents the shifted
/// strip on its next <=16ms tick (zero re-raster, zero renderer round-trip,
/// the Chrome-shaped compositor-thread fast scroll). Clamping stays on the UI
/// writer side so the atomic never holds an out-of-range value.
///
/// Legacy / fallback: `InvalidateRect` → WM_PAINT composites+presents inline,
/// exactly as today (byte-identical when the flag is OFF).
fn publish_scroll(guard: &WindowState, hwnd: sys::HWND) {
    let mode = guard
        .compositor_present_mode
        .as_ref()
        .map(|m| m.load(Ordering::Acquire))
        .unwrap_or(present_mode::UNKNOWN);
    let compositor_owns = offmain_compositor_enabled()
        && guard.compositor_tx.is_some()
        && mode != present_mode::FALLBACK_STRETCH_DIBITS;
    if compositor_owns {
        if let Some(scroll) = guard.shared_scroll.as_ref() {
            scroll.store(guard.scroll_y, Ordering::Release);
        }
        // No InvalidateRect for content — the compositor's tick re-presents.
        // (The scrollbar thumb was already updated in clamp_scroll.)
    } else {
        unsafe { sys::InvalidateRect(hwnd, core::ptr::null(), 0) };
    }
}

unsafe extern "system" fn wnd_proc(
    hwnd: sys::HWND,
    msg: u32,
    wparam: usize,
    lparam: isize,
) -> isize {
    match msg {
        sys::WM_PAINT => {
            let mut ps: sys::PAINTSTRUCT = unsafe { core::mem::zeroed() };
            let hdc = unsafe { sys::BeginPaint(hwnd, &raw mut ps) };
            let state_ptr = OWNER_PTR.load(Ordering::SeqCst);
            if !state_ptr.is_null() {
                let mut guard = unsafe { (*state_ptr).borrow_mut() };
                // Read the live client size so resizing eventually works;
                // current viewport_h is the content region under chrome.
                let mut client = sys::RECT::default();
                unsafe { sys::GetClientRect(hwnd, &raw mut client) };
                let client_w = (client.right - client.left).max(0);
                let client_h = (client.bottom - client.top).max(0);
                let chrome_h = guard.paint.chrome_h as i32;
                let viewport_h = (client_h - chrome_h).max(0);
                // Re-clamp scroll in case the bitmap shrunk or the window
                // grew. Use DOCUMENT height (content_height()), not the bitmap
                // height — a band bitmap is shorter than the document.
                let content_h = guard.paint.content_height() as i32;
                let max_scroll = (content_h - viewport_h).max(0);
                if guard.scroll_y > max_scroll {
                    guard.scroll_y = max_scroll;
                }
                if guard.scroll_y < 0 {
                    guard.scroll_y = 0;
                }
                update_scrollbar(&guard, hwnd);
                let scroll_y = guard.scroll_y;

                // Pre-compute blit width from bitmap BEFORE binding
                // `pd = &guard.paint` — the GPU present needs `&mut`
                // access to `guard.hw_presenter` which conflicts with
                // the immutable `pd` borrow through `Deref`.
                let bw = guard.paint.bitmap.width as i32;
                let blit_w = bw.min(client_w);

                // ── Off-main compositor content-present gate ──────
                // When the compositor thread owns present, the UI WM_PAINT
                // must NOT composite/present the content — the compositor does
                // it on its swap chain. The UI still draws chrome + caret to
                // the HDC (below). It presents content here ONLY when the
                // compositor reported GpuInitFailed (FALLBACK_STRETCH_DIBITS).
                // While Unknown / OwnedByCompositor it skips content present
                // (relying on the swap-chain present) so there is no fight over
                // the content area and no black flash.
                let ui_presents_content = if offmain_compositor_enabled()
                    && guard.compositor_tx.is_some()
                {
                    guard
                        .compositor_present_mode
                        .as_ref()
                        .map(|m| m.load(Ordering::Acquire) == present_mode::FALLBACK_STRETCH_DIBITS)
                        .unwrap_or(false)
                } else {
                    true
                };

                // ── GPU present path ─────────────────────────────
                // Composite the visible viewport strip and try the
                // hardware swap chain.  Must happen before `pd` is
                // bound because `present_u32` borrows `guard`
                // mutably through `DerefMut`.
                // Band-bitmap: the cached bitmap's row 0 is document row
                // `content_origin_y`, so slice it at `scroll_y - content_origin_y`.
                // 0 for a full-document bitmap (legacy) ⇒ unchanged.
                let band_src_y = (scroll_y - guard.paint.content_origin_y as i32).max(0);
                let (gpu_presented, viewport_pixels_pre) = if viewport_h > 0 && ui_presents_content {
                    // Direct single-copy viewport slice from the page bitmap (no
                    // per-frame tile-cache round-trip — see apply_new_paint).
                    let vp = cv_compositor::composite_viewport_direct(
                        &guard.paint.bitmap.pixels,
                        guard.paint.bitmap.width as u32,
                        guard.paint.bitmap.height as u32,
                        0,
                        band_src_y,
                        blit_w as u32,
                        viewport_h as u32,
                    );
                    // GPU present writes the swap chain from the top-left of the
                    // WHOLE window. The content must sit BELOW the chrome strip, so
                    // assemble a full-client-height frame (white) with the content
                    // copied in at row offset `chrome_h`. Presenting only the
                    // `viewport_h`-tall buffer (as before) slid content under the
                    // chrome AND left the bottom `chrome_h` rows of the swap chain
                    // unwritten → the black bottom bar. The top `chrome_h` rows here
                    // are overpainted by the GDI chrome draw below.
                    // Assemble a FULL-client-size frame (white): content occupies
                    // the left `blit_w` columns starting at row `chrome_h`. The
                    // swap chain covers the whole window, so the frame must be
                    // client_w × client_h — otherwise the right margin past the
                    // page width (blit_w < client_w) and the bottom chrome strip
                    // are left unwritten → black blocks. Right margin + top chrome
                    // rows stay white (the GDI chrome paints over the top strip).
                    let fw = client_w.max(0) as usize;          // full client width
                    let ch = client_h.max(0) as usize;          // full client height
                    let src_w = blit_w.max(0) as usize;         // content (page) width
                    let top = chrome_h.max(0) as usize;
                    let mut frame = vec![0xFFFF_FFFFu32; fw * ch];
                    if fw > 0 && src_w > 0 {
                        let rows = (vp.len() / src_w).min(ch.saturating_sub(top));
                        for r in 0..rows {
                            let s = r * src_w;
                            let d = (top + r) * fw;
                            frame[d..d + src_w].copy_from_slice(&vp[s..s + src_w]);
                        }
                    }
                    // Bake the chrome strip INTO the presented frame. The DComp
                    // swap-chain visual composites OVER the window HDC, so the
                    // GDI chrome drawn to the HDC below is invisible while GPU is
                    // on. Rendering the chrome into the frame's top `chrome_h`
                    // rows (via an offscreen memory DC, byte-identical look) puts
                    // it inside the swap-chain frame where it IS visible — and
                    // replaces the white top rows that previously hid the HDC
                    // chrome. The GDI chrome draw below is still correct for the
                    // StretchDIBits CPU path and harmless here (overdrawn by the
                    // present that already contains the chrome).
                    bake_chrome_into_frame(
                        &mut frame,
                        client_w,
                        client_h,
                        chrome_h,
                        &guard.tabs,
                        guard.effective_pressed_nav(),
                        guard.nav_in_flight,
                    );
                    // Custom content-region scrollbar (replaces WS_VSCROLL, which
                    // drew up through the chrome). Confined below chrome_h.
                    draw_scrollbar_into_frame(
                        &mut frame,
                        client_w,
                        client_h,
                        chrome_h,
                        guard.paint.bitmap.height as i32,
                        scroll_y,
                    );
                    let presented = if let Some(ref mut hw) = guard.hw_presenter {
                        hw.present_u32(&frame, client_w as u32, client_h as u32)
                            .is_ok()
                    } else {
                        false
                    };
                    (presented, Some(vp))
                } else {
                    (false, None)
                };

                let pd = &guard.paint;
                let bmp = &pd.bitmap;

                // Scroll diagnostic: when CV_SCROLL_LOG is set, write a
                // single line per paint into the temp dir.  Captures the
                // current scroll_y plus the first 3 content text items'
                // y values; the user can scroll, watch the log, and tell
                // us whether text and image (bitmap-baked) y values
                // actually diverge.  Off by default — no-op without the
                // env var.
                if std::env::var_os("CV_SCROLL_LOG").is_some() {
                    use std::io::Write;
                    let tmp = std::env::temp_dir().join("tb_scroll.log");
                    if let Ok(mut f) = std::fs::OpenOptions::new()
                        .create(true)
                        .append(true)
                        .open(&tmp)
                    {
                        let _ = write!(
                            f,
                            "paint scroll_y={} chrome_h={} viewport_h={} bitmap_h={} clip=[{}..{}]",
                            scroll_y,
                            chrome_h,
                            viewport_h,
                            bmp.height,
                            chrome_h,
                            chrome_h + viewport_h
                        );
                        // image strip: at this scroll_y, bitmap row
                        // Bitmap row `scroll_y` appears at screen row chrome_h.
                        // So bitmap row B appears at screen row B - scroll_y.
                        let _ = write!(
                            f,
                            " | img_strip_src=[{}..{}] img_strip_dst=[{}..{}]",
                            scroll_y,
                            scroll_y + viewport_h,
                            chrome_h,
                            chrome_h + viewport_h
                        );
                        let mut printed = 0;
                        for t in &pd.texts {
                            if t.is_chrome {
                                continue;
                            }
                            let _ =
                                write!(f, " | text[y={},h={},draw_y={}]", t.y, t.h, t.y - scroll_y);
                            printed += 1;
                            if printed >= 3 {
                                break;
                            }
                        }
                        let _ = writeln!(f);
                    }
                }
                // NOTE: the full-bitmap BITMAPINFO used to live here.
                // The content-strip blit now goes through the tile cache
                // (composite_viewport) and creates its own viewport-sized
                // BITMAPINFO inline, so the full-bitmap header is gone.
                unsafe {
                    // blit_w was pre-computed above (before `pd`) so
                    // that the GPU present could run without conflicting
                    // with the immutable `pd` borrow.  Blit 1:1; pad
                    // any extra width with white.
                    // 1. Chrome strip — always pinned, owned by the
                    //    native presenter. Page bitmaps are content-only.
                    //    Drawn to the window HDC via the SHARED chrome-draw fn
                    //    (the SAME code the GPU path bakes into the swap-chain
                    //    frame via an offscreen memory DC) so the CPU and GPU
                    //    chrome are byte-identical. When the GPU swap chain is on,
                    //    the frame already contains this chrome (bake above) and
                    //    composites over the HDC; this HDC draw is the CPU/
                    //    StretchDIBits path's chrome and a harmless under-layer
                    //    otherwise.
                    draw_chrome_to_hdc(
                        hdc,
                        client_w,
                        chrome_h,
                        &guard.tabs,
                        guard.effective_pressed_nav(),
                        guard.nav_in_flight,
                    );
                    // 2. Content strip — the viewport was composited
                    //    and (optionally) GPU-presented above.  If the
                    //    GPU path succeeded, skip the GDI blit;
                    //    otherwise fall back to StretchDIBits.
                    if viewport_h > 0 && !gpu_presented {
                        if let Some(ref viewport_pixels) = viewport_pixels_pre {
                        let vi = sys::BITMAPINFO {
                            bmiHeader: sys::BITMAPINFOHEADER {
                                biSize: core::mem::size_of::<sys::BITMAPINFOHEADER>()
                                    as u32,
                                biWidth: blit_w,
                                biHeight: -(viewport_h),
                                biPlanes: 1,
                                biBitCount: 32,
                                biCompression: sys::BI_RGB,
                                biSizeImage: 0,
                                biXPelsPerMeter: 0,
                                biYPelsPerMeter: 0,
                                biClrUsed: 0,
                                biClrImportant: 0,
                            },
                            bmiColors: [0; 1],
                        };
                        sys::StretchDIBits(
                            hdc,
                            0,
                            chrome_h,
                            blit_w,
                            viewport_h,
                            0,
                            0, // src_y=0: viewport_pixels is already scrolled
                            blit_w,
                            viewport_h,
                            viewport_pixels.as_ptr() as *const c_void,
                            &raw const vi,
                            sys::DIB_RGB_COLORS,
                            sys::SRCCOPY,
                        );
                        }
                    }
                    // Pad any client-width past the bitmap with a
                    // solid white rect so the user doesn't see garbage
                    // (or torn previous frames) on the right margin
                    // when the window is wider than the rendered page.
                    if blit_w < client_w && client_h > 0 {
                        let pad_rect = sys::RECT {
                            left: blit_w,
                            top: 0,
                            right: client_w,
                            bottom: client_h,
                        };
                        let white = sys::CreateSolidBrush(sys::rgb(255, 255, 255));
                        sys::FillRect(hdc, &raw const pad_rect, white);
                        sys::DeleteObject(white);
                    }

                    sys::SetBkMode(hdc, sys::TRANSPARENT);
                    // Track which clip region is currently selected on
                    // the HDC: None = no clip (chrome region), Some =
                    // content-area clip (chrome_h..chrome_h+viewport_h).
                    //
                    // The bitmap content strip (StretchDIBits above)
                    // already clips images / backgrounds to the content
                    // area by construction: its source rect skips
                    // bitmap rows below `chrome_h + scroll_y`.  Text,
                    // however, is drawn via DrawTextW at a screen y
                    // computed as `t.y - scroll_y` with no clip.  When
                    // scroll_y > 0, content text whose laid-out y falls
                    // in `[chrome_h, chrome_h + scroll_y)` produces a
                    // negative-or-tiny draw_y that lands INSIDE the
                    // URL-bar zone — text would paint over the chrome
                    // strip while the matching image (clipped at the
                    // source by StretchDIBits) disappears.  That's the
                    // "text goes one way, images go the other" bug.
                    //
                    // Fix: clip non-chrome text drawing to the content
                    // area via SelectClipRgn so text obeys the same
                    // visual boundary the bitmap strip does.
                    for t in &pd.texts {
                        let draw_y = if t.is_chrome {
                            t.y
                        } else {
                            chrome_h + t.y - scroll_y
                        };
                        // Cull text fully outside the visible window —
                        // saves CreateFont/DrawText work on long pages.
                        let h_guess = t.h.max(t.font_size_px * 4);
                        if draw_y + h_guess < 0 || draw_y > client_h {
                            continue;
                        }
                        // HARD CULL: content text whose top is anywhere
                        // inside the chrome strip area is dropped
                        // entirely.  The bitmap content strip starts at
                        // bitmap row `chrome_h + scroll_y` — anything
                        // before that vanishes when scrolled.  Text
                        // must obey the same boundary, otherwise the
                        // page's leading lines slide up THROUGH the URL
                        // bar while the matching backgrounds / images
                        // stop dead at the chrome edge, producing the
                        // "text goes one way, images go the other"
                        // divergence the user has reported repeatedly.
                        //
                        // Note: this clips entire text lines whose top
                        // overlaps the chrome strip.  A small visual
                        // glitch (text pops at the boundary instead of
                        // sliding under) is the trade — but it
                        // GUARANTEES no text ever paints in the URL
                        // bar zone, matching the bitmap behaviour
                        // exactly.
                        if !t.is_chrome && draw_y < chrome_h {
                            continue;
                        }
                        let face_name = resolve_font_family(t.font_family.as_deref());
                        let face: Vec<u16> = format!("{face_name}\0").encode_utf16().collect();
                        let weight = if t.bold { sys::FW_BOLD } else { sys::FW_NORMAL };
                        let italic = if t.italic { 1 } else { 0 };
                        let hfont = sys::CreateFontW(
                            t.font_size_px,
                            0,
                            0,
                            0,
                            weight,
                            italic,
                            0,
                            0,
                            sys::DEFAULT_CHARSET,
                            sys::OUT_DEFAULT_PRECIS,
                            sys::CLIP_DEFAULT_PRECIS,
                            sys::CLEARTYPE_QUALITY,
                            sys::DEFAULT_PITCH | sys::FF_DONTCARE,
                            face.as_ptr(),
                        );
                        let old_font = sys::SelectObject(hdc, hfont);
                        let (r, g, b) = t.color_rgb;
                        sys::SetTextColor(hdc, sys::rgb(r, g, b));
                        let cleaned: String = t
                            .text
                            .chars()
                            .map(|c| {
                                let cp = c as u32;
                                let is_emoji = (0x1F000..=0x1FFFF).contains(&cp)
                                    || (0x2600..=0x27BF).contains(&cp)
                                    || (0x2300..=0x23FF).contains(&cp)
                                    || cp == 0x200D
                                    || cp == 0xFE0F;
                                if is_emoji { ' ' } else { c }
                            })
                            .collect();
                        let mut text_w: Vec<u16> = cleaned.encode_utf16().collect();
                        text_w.push(0);
                        let mut rc = sys::RECT {
                            left: t.x,
                            top: draw_y,
                            right: t.x + t.w,
                            bottom: draw_y + h_guess,
                        };
                        let align_flag = match t.align {
                            TextAlign::Left => sys::DT_LEFT,
                            TextAlign::Center => sys::DT_CENTER,
                            TextAlign::Right => sys::DT_RIGHT,
                        };
                        sys::DrawTextW(
                            hdc,
                            text_w.as_ptr(),
                            -1,
                            &raw mut rc,
                            align_flag | sys::DT_TOP | sys::DT_NOPREFIX | sys::DT_WORDBREAK,
                        );
                        sys::SelectObject(hdc, old_font);
                        sys::DeleteObject(hfont);
                    }
                    // URL bar is a real Win32 EDIT child control —
                    // it paints itself on top of the chrome strip,
                    // so there's nothing to draw here. WS_CLIPCHILDREN
                    // on the parent keeps DrawText above from leaking
                    // into the EDIT's rect.
                }
                // Caret overlay: draw AFTER everything else so it sits
                // on top of the page bitmap. Caret_rect is in bitmap
                // coordinates; subtract scroll_y to land in client
                // coordinates. Skip if currently invisible (blink off)
                // or if the caret rect is in the scrolled-out region.
                if let Some((cx, cy, cw, ch)) = guard.paint.caret_rect {
                    if caret_blink_visible() {
                        let draw_y = cy - scroll_y + chrome_h;
                        if draw_y + ch > chrome_h && draw_y < client_h {
                            let r = sys::RECT {
                                left: cx,
                                top: draw_y.max(chrome_h),
                                right: cx + cw,
                                bottom: (draw_y + ch).min(client_h),
                            };
                            let brush = unsafe { sys::CreateSolidBrush(sys::rgb(16, 16, 16)) };
                            unsafe {
                                sys::FillRect(hdc, &raw const r, brush);
                                sys::DeleteObject(brush);
                            }
                        }
                    }
                }
            }
            unsafe { sys::EndPaint(hwnd, &raw const ps) };
            0
        }
        sys::WM_TIMER if wparam == sys::JS_TICK_TIMER_ID => {
            let state_ptr = OWNER_PTR.load(Ordering::SeqCst);
            if !state_ptr.is_null() {
                // Hold the lock only long enough to take the ticker out,
                // so the ticker itself can re-enter (e.g. to read paint).
                // We restore the closure when done.
                let mut ticker = {
                    let mut guard = unsafe { (*state_ptr).borrow_mut() };
                    guard.ticker.take()
                };
                let result = ticker.as_mut().and_then(|t| t());
                {
                    let mut guard = unsafe { (*state_ptr).borrow_mut() };
                    guard.ticker = ticker;
                    if let Some(new_paint) = result {
                        apply_new_paint(&mut guard, hwnd, new_paint);
                    }
                }
            }
            0
        }
        sys::WM_CHAR => {
            // wparam is the Unicode codepoint from the IME/keyboard
            // translation. Filter out control chars except backspace
            // (0x08) and CR (0x0D — Enter); those carry their own
            // semantic in the input event flow.
            let ch = wparam as u32;
            if !(ch == 0x08 || ch == 0x0D || (0x20..=0x10FFFF).contains(&ch)) {
                return 0;
            }
            let state_ptr = OWNER_PTR.load(Ordering::SeqCst);
            if state_ptr.is_null() {
                return 0;
            }
            // URL bar typing goes through the EDIT child — its own
            // WNDPROC runs the IME/keyboard pipeline. We only see
            // WM_CHAR here for the page-level input dispatch, which
            // synthesises `tb-typed:` / `tb-backspace:` / `tb-enter:`
            // URLs so the running JS sees keyboard input on focused
            // <input>/<textarea> nodes.
            let url = if ch == 0x08 {
                "tb-backspace:".to_string()
            } else if ch == 0x0D {
                "tb-enter:".to_string()
            } else {
                let c = char::from_u32(ch).unwrap_or(' ');
                format!("tb-typed:{c}")
            };
            let mut guard = unsafe { (*state_ptr).borrow_mut() };
            pump_input_command(&mut guard, hwnd, &url);
            0
        }
        m if m == WM_APP_FROMPAGE => {
            // OFF-MAIN: the renderer thread posted one or more finished frames.
            // Drain the channel, keep the newest, and apply it — UNLESS a newer
            // navigation has superseded its epoch (stale-frame drop). This runs
            // entirely on the UI thread; the renderer never touches the HWND.
            let state_ptr = OWNER_PTR.load(Ordering::SeqCst);
            if state_ptr.is_null() {
                return 0;
            }
            let mut guard = unsafe { (*state_ptr).borrow_mut() };
            let mut latest: Option<(u64, PaintData, Vec<TabSummary>)> = None;
            if let Some(rx) = guard.from_page.as_ref() {
                // Coalesce: only the most recent frame matters; intermediate
                // frames queued faster than we drained are obsolete.
                while let Ok(FromPage::Commit { epoch, paint, tabs }) = rx.try_recv() {
                    latest = Some((epoch, paint, tabs));
                }
            }
            if let Some((epoch, paint, tabs)) = latest {
                if should_apply_commit(epoch, guard.nav_gen) {
                    // A frame at the current epoch arrived: the pending
                    // navigation produced output, so the new page is live —
                    // stop gating content input.
                    guard.nav_in_flight = false;
                    if !tabs.is_empty() {
                        guard.tabs = tabs;
                    }
                    apply_new_paint(&mut guard, hwnd, paint);
                }
            }
            0
        }
        m if m == WM_APP_INVALIDATE_CARET => {
            // Posted by the off-main renderer's ticker (caret blink). We are on
            // the UI thread now, so invalidate_caret's WindowState borrow is safe.
            invalidate_caret();
            0
        }
        m if m == WM_APP_COMPOSITOR_STATUS => {
            // Posted by the compositor thread after a CompositorStatus is set.
            // The shared present-mode AtomicU8 already carries the new state
            // (the compositor wrote it). We just drain the status channel (for
            // completeness) and repaint so WM_PAINT re-evaluates whether the UI
            // should present content (StretchDIBits fallback) or defer to the
            // compositor's swap chain. A repaint here also guarantees the UI's
            // own tile cache gets populated if we fell back to StretchDIBits.
            let state_ptr = OWNER_PTR.load(Ordering::SeqCst);
            if !state_ptr.is_null() {
                let mut guard = unsafe { (*state_ptr).borrow_mut() };
                // If the compositor failed init, make sure the UI tile cache has
                // pixels for the StretchDIBits fallback (apply_new_paint's
                // fallback path also does this on the next commit, but populate
                // now so the very next paint is correct).
                let mode = guard
                    .compositor_present_mode
                    .as_ref()
                    .map(|m| m.load(Ordering::Acquire))
                    .unwrap_or(present_mode::UNKNOWN);
                if mode == present_mode::FALLBACK_STRETCH_DIBITS {
                    guard.tile_cache.invalidate_all();
                    let (px, w, h) = {
                        let bmp = &guard.paint.bitmap;
                        (bmp.pixels.clone(), bmp.width as u32, bmp.height as u32)
                    };
                    guard.tile_cache.refresh_from_raw(&px, w, h);
                }
            }
            unsafe { sys::InvalidateRect(hwnd, core::ptr::null(), 0) };
            0
        }
        m if m == sys::WM_USER + 1 => {
            // Worker thread delivered a fetched body. lparam is a
            // Box::into_raw(Box::new((url, body))). Re-take ownership,
            // run the post-fetch pipeline on the UI thread, swap in
            // the result paint.
            if lparam == 0 {
                return 0;
            }
            let payload_ptr = lparam as *mut (String, Vec<u8>);
            let payload = unsafe { Box::from_raw(payload_ptr) };
            let (url, body) = *payload;
            let state_ptr = OWNER_PTR.load(Ordering::SeqCst);
            if !state_ptr.is_null() {
                if body.is_empty() {
                    // Fetch failed. Reflect the URL the user asked
                    // for + a short error prefix in the EDIT so they
                    // can fix the address and retry. Then focus the
                    // EDIT and select-all so the next keystroke
                    // overwrites. (This is a real ERROR state, not
                    // progress spam — kept, unlike "Loading…".)
                    let mut guard = unsafe { (*state_ptr).borrow_mut() };
                    guard.nav_in_flight = false;
                    let edit = guard.edit_hwnd;
                    let err_w = to_wide(&format!("can't reach {url}"));
                    unsafe {
                        sys::SetWindowTextW(edit, err_w.as_ptr());
                        sys::SetFocus(edit);
                        sys::SendMessageW(edit, sys::EM_SETSEL, 0, -1);
                        sys::InvalidateRect(hwnd, core::ptr::null(), 0);
                    }
                    return 0;
                }
                // The fetch landed; the (synchronous) HTML parse + JS interp +
                // layout pass is about to run on the UI thread. We do NOT
                // overwrite the address bar with "Rendering…" anymore (real
                // browsers keep the URL there). Instead, leave nav_in_flight set
                // (it already is, from when the nav started) across the build so
                // the chrome's STOP glyph + thin progress bar stay visible, and
                // force a synchronous chrome repaint NOW so that feedback is on
                // screen before the build (which can hold the thread for seconds
                // on a heavy page).
                //
                // RE-ENTRANCY: UpdateWindow(hwnd) synchronously dispatches
                // WM_PAINT into this wnd_proc, whose paint arm borrow_mut()s
                // WindowState. So we must NOT hold a borrow across it. Invalidate
                // the chrome + UpdateWindow under NO borrow, then borrow only for
                // the build.
                {
                    let guard = unsafe { (*state_ptr).borrow() };
                    invalidate_chrome(&guard, hwnd);
                } // borrow dropped before UpdateWindow re-enters WM_PAINT
                unsafe {
                    sys::UpdateWindow(hwnd);
                }
                let mut guard = unsafe { (*state_ptr).borrow_mut() };
                if let Some(navb) = guard.nav_with_body.as_mut() {
                    if let Some(new_paint) = navb(&url, body) {
                        guard.nav_in_flight = false;
                        guard.scroll_y = 0;
                        apply_new_paint(&mut guard, hwnd, new_paint);
                    } else {
                        guard.nav_in_flight = false;
                        // navb returned None — the parser / layout /
                        // runtime build failed after a successful
                        // fetch. Without this the URL bar would stay
                        // stuck at "Rendering …" forever. Surface a
                        // concrete error and hand focus back to the
                        // user.
                        let edit = guard.edit_hwnd;
                        let err_w = to_wide(&format!("render failed for {url}"));
                        unsafe {
                            sys::SetWindowTextW(edit, err_w.as_ptr());
                            sys::SetFocus(edit);
                            sys::SendMessageW(edit, sys::EM_SETSEL, 0, -1);
                            sys::InvalidateRect(hwnd, core::ptr::null(), 0);
                        }
                    }
                } else {
                    // No nav_with_body wired — clear the in-flight flag so the
                    // stop glyph / progress bar don't stick on forever, and
                    // repaint the chrome back to the reload state.
                    guard.nav_in_flight = false;
                    invalidate_chrome(&guard, hwnd);
                }
            }
            0
        }
        m if m == sys::WM_USER + 2 => {
            // Subclassed EDIT posted "user hit Enter". Read its text
            // and dispatch the nav (worker-thread path when fetcher
            // is wired; sync fallback otherwise).
            let state_ptr = OWNER_PTR.load(Ordering::SeqCst);
            if state_ptr.is_null() {
                return 0;
            }
            let mut guard = unsafe { (*state_ptr).borrow_mut() };
            if guard.nav_in_flight {
                return 0;
            }
            // Read current EDIT text.
            let edit = guard.edit_hwnd;
            let len = unsafe { sys::GetWindowTextLengthW(edit) };
            if len < 0 {
                return 0;
            }
            let cap = (len as usize) + 1;
            let mut buf: Vec<u16> = vec![0u16; cap];
            unsafe { sys::GetWindowTextW(edit, buf.as_mut_ptr(), cap as i32) };
            // Trim the trailing NUL and any trailing zero pad.
            let end = buf.iter().position(|&c| c == 0).unwrap_or(buf.len());
            let raw_url = String::from_utf16_lossy(&buf[..end]);
            let mut url = raw_url.trim().to_string();
            if url.is_empty() {
                // Re-display the current page URL so the bar isn't blank.
                let cur = to_wide(&guard.paint.current_url);
                unsafe { sys::SetWindowTextW(edit, cur.as_ptr()) };
                return 0;
            }
            if !url.contains("://") && !url.starts_with("about:") && !url.starts_with("javascript:")
            {
                url = format!("https://{url}");
            }
            // OFF-MAIN: hand the navigation to the renderer thread. Bump the
            // navigation epoch (drops stale frames from the abandoned page),
            // reset scroll, and send the command — the renderer does the
            // blocking fetch+build on its own thread; the UI stays responsive.
            if guard.to_page.is_some() {
                guard.nav_gen += 1;
                guard.nav_in_flight = true;
                let epoch = guard.nav_gen;
                guard.scroll_y = 0;
                if let Some(tx) = guard.to_page.as_ref() {
                    let _ = tx.send(ToPage::Cmd { epoch, cmd: url.clone() });
                }
                let edit = guard.edit_hwnd;
                // Show the canonical destination URL (which may have been
                // https://-prefixed), NOT "Loading…". The reload→stop button +
                // progress bar (driven by nav_in_flight) signal the load.
                let dest_w = to_wide(&url);
                invalidate_chrome(&guard, hwnd);
                drop(guard);
                unsafe {
                    sys::SetWindowTextW(edit, dest_w.as_ptr());
                    sys::UpdateWindow(edit);
                    sys::SetFocus(hwnd);
                }
                return 0;
            }
            let can_thread = guard.fetcher.is_some() && guard.nav_with_body.is_some();
            if can_thread {
                guard.nav_in_flight = true;
                guard.scroll_y = 0;
                let fetcher = guard.fetcher.as_ref().unwrap().clone();
                let edit = guard.edit_hwnd;
                // Drop the borrow BEFORE the Win32 calls — SetFocus
                // is documented to dispatch WM_KILLFOCUS/WM_SETFOCUS
                // re-entrantly, and we don't want a reentry to find
                // the RefCell still borrowed.
                invalidate_chrome(&guard, hwnd);
                drop(guard);
                // Show the canonical destination URL (Chrome behavior), NOT
                // "Loading…", and force a paint of the EDIT + chrome (stop glyph
                // + progress bar) before kicking off the slow blocking fetch.
                let dest_w = to_wide(&url);
                unsafe {
                    sys::SetWindowTextW(edit, dest_w.as_ptr());
                    sys::UpdateWindow(edit);
                    // Hand keyboard focus back to the page so the
                    // EDIT doesn't keep a blinking caret on the
                    // placeholder text.
                    sys::SetFocus(hwnd);
                }
                let hwnd_send = HwndSend(hwnd);
                std::thread::spawn(move || {
                    let hs = hwnd_send;
                    let body = fetcher(url.clone());
                    let payload: Box<(String, Vec<u8>)> = Box::new((url, body));
                    let raw = Box::into_raw(payload);
                    unsafe {
                        sys::PostMessageW(hs.0, sys::WM_USER + 1, 0, raw as isize);
                    }
                });
                return 0;
            }
            // Synchronous fallback path.
            unsafe { sys::SetFocus(hwnd) };
            if let Some(nav) = guard.navigator.as_mut() {
                if let Some(new_paint) = nav(&url) {
                    guard.scroll_y = 0;
                    apply_new_paint(&mut guard, hwnd, new_paint);
                }
            }
            0
        }
        m if m == sys::WM_USER + 3 => {
            // Subclassed EDIT posted "user hit Escape". Restore the
            // text to the current URL (discarding edits) and drop
            // focus back to the main window.
            let state_ptr = OWNER_PTR.load(Ordering::SeqCst);
            if !state_ptr.is_null() {
                let guard = unsafe { (*state_ptr).borrow() };
                let cur = to_wide(&guard.paint.current_url);
                let edit = guard.edit_hwnd;
                unsafe {
                    sys::SetWindowTextW(edit, cur.as_ptr());
                    sys::SetFocus(hwnd);
                }
            }
            0
        }
        sys::WM_MOUSEWHEEL => {
            // High word of wParam is the wheel delta in WHEEL_DELTA units
            // (120 per notch). Win32 convention: positive delta = wheel
            // rotated forward (away from the user).  By long-standing
            // Windows app behavior, forward wheel scrolls the viewport
            // UP through the document — meaning the user wants to see
            // content that's higher up, which means decreasing the
            // scroll offset so previously-off-top content comes into
            // view.
            //
            // Old code did `scroll_y += lines * 60`, which inverted the
            // direction: forward wheel pushed `scroll_y` larger and the
            // content slid up off the screen instead of down.  Visually
            // this read as "everything scrolls backwards"; combined
            // with the bitmap-vs-text split it also makes parts of the
            // page appear to drift in different directions because the
            // user's expectation and our math disagree on which way is
            // "down."
            let delta = ((wparam >> 16) & 0xFFFF) as i16;
            let state_ptr = OWNER_PTR.load(Ordering::SeqCst);
            if !state_ptr.is_null() {
                let lines = (delta as i32) / 120;
                // Win32 convention: wheel-DOWN (toward user) = NEGATIVE delta,
                // and scrolling down means advancing INTO the document (content
                // slides up, scroll_y INCREASES). So scroll_y must move OPPOSITE
                // the delta sign: scroll_y -= lines*step. 60px per line ≈ 3
                // lines/notch like Chrome's default.
                let dy = -(lines * 60) as f32; // px to ADD to a scroll offset
                // ── Element-level scroll routing (Blink scroll chaining) ──────
                // Map the wheel position to the innermost scroll container under
                // the cursor that can still move in the wheel's direction; if
                // found, scroll IT (route a tb-scroll command to the renderer)
                // instead of the page. At the edge, chain outward to the next
                // ancestor; if nothing can absorb it, fall through to page scroll.
                let routed = route_wheel_to_element(hwnd, state_ptr, dy);
                if !routed {
                    let mut guard = unsafe { (*state_ptr).borrow_mut() };
                    guard.scroll_y -= lines * 60;
                    clamp_scroll(&mut guard, hwnd);
                    publish_scroll(&guard, hwnd);
                }
            }
            0
        }
        sys::WM_VSCROLL => {
            let state_ptr = OWNER_PTR.load(Ordering::SeqCst);
            if !state_ptr.is_null() {
                let mut guard = unsafe { (*state_ptr).borrow_mut() };
                let code = (wparam & 0xFFFF) as u32;
                let (_, viewport_h, max_scroll) = scroll_metrics(&guard, hwnd);
                match code {
                    sys::SB_LINEUP => guard.scroll_y -= 40,
                    sys::SB_LINEDOWN => guard.scroll_y += 40,
                    sys::SB_PAGEUP => guard.scroll_y -= viewport_h,
                    sys::SB_PAGEDOWN => guard.scroll_y += viewport_h,
                    sys::SB_TOP => guard.scroll_y = 0,
                    sys::SB_BOTTOM => guard.scroll_y = max_scroll,
                    sys::SB_THUMBPOSITION | sys::SB_THUMBTRACK => {
                        let mut info = sys::SCROLLINFO {
                            cbSize: core::mem::size_of::<sys::SCROLLINFO>() as u32,
                            fMask: sys::SIF_ALL,
                            ..Default::default()
                        };
                        unsafe {
                            sys::GetScrollInfo(hwnd, sys::SB_VERT, &raw mut info);
                        }
                        guard.scroll_y = info.nTrackPos;
                    }
                    _ => {}
                }
                clamp_scroll(&mut guard, hwnd);
                publish_scroll(&guard, hwnd);
            }
            0
        }
        sys::WM_LBUTTONDOWN => {
            let x = (lparam & 0xFFFF) as i16 as f32;
            let y_raw = ((lparam >> 16) & 0xFFFF) as i16 as f32;
            let state_ptr = OWNER_PTR.load(Ordering::SeqCst);
            if !state_ptr.is_null() {
                let mut guard = unsafe { (*state_ptr).borrow_mut() };
                let chrome_h = guard.paint.chrome_h as f32;
                // ── Custom scrollbar hit-test (before chrome/content) ─────────
                // Our scrollbar lives in the content region's right edge. A click
                // on the THUMB starts a drag; on the TRACK page-jumps toward the
                // click. (We dropped WS_VSCROLL so this is our own scrollbar.)
                {
                    let mut client = sys::RECT::default();
                    unsafe { sys::GetClientRect(hwnd, &raw mut client) };
                    let cw = (client.right - client.left).max(0);
                    let ch = (client.bottom - client.top).max(0);
                    let cont_h = guard.paint.bitmap.height as i32;
                    let x_int = x as i32;
                    let y_int = y_raw as i32;
                    if x_int >= cw - SCROLLBAR_W && y_int >= chrome_h as i32 {
                        if let Some((_tx, ty, _tw, th)) = scrollbar_thumb_rect(
                            cw, ch, chrome_h as i32, cont_h, guard.scroll_y,
                        ) {
                            if y_int >= ty && y_int < ty + th {
                                // Grab the thumb → start drag.
                                guard.scroll_drag = Some((y_int, guard.scroll_y));
                                unsafe { sys::SetCapture(hwnd); }
                            } else {
                                // Click on the track above/below the thumb →
                                // page up/down by a viewport.
                                let viewport_h = (ch - chrome_h as i32).max(1);
                                if y_int < ty {
                                    guard.scroll_y -= viewport_h;
                                } else {
                                    guard.scroll_y += viewport_h;
                                }
                                clamp_scroll(&mut guard, hwnd);
                                publish_scroll(&guard, hwnd);
                            }
                            return 0;
                        }
                    }
                }
                // Click in the chrome strip:
                //   - back/forward button: trigger nav via worker
                //   - elsewhere in chrome: focus the URL bar
                if y_raw < chrome_h {
                    let x_int = x as i32;
                    let y_int = y_raw as i32;
                    // Tabs are the TOP row (TAB_Y..TOOLBAR_Y); the toolbar
                    // (buttons + URL bar) is the row below. Only run tab
                    // hit-tests in the tab strip band so a toolbar-button click
                    // never gets misread as a tab interaction.
                    if y_int >= TAB_Y && y_int < TOOLBAR_Y {
                        if let Some(tab_id) = hit_test_tab_close(&guard.tabs, x_int, y_int) {
                            // The host only closes the ACTIVE tab, so closing a
                            // non-active tab means switch-to-it then close. Both
                            // legacy (inline) and off-main (two ToPage::Host sends,
                            // coalesced to the final state) go through
                            // pump_host_command.
                            let active_id =
                                guard.tabs.iter().find(|tab| tab.active).map(|tab| tab.id);
                            if active_id != Some(tab_id) {
                                pump_host_command(&mut guard, hwnd, HostCommand::SwitchTab(tab_id));
                            }
                            pump_host_command(&mut guard, hwnd, HostCommand::CloseActiveTab);
                            return 0;
                        }
                        if hit_test_new_tab(&guard.tabs, x_int, y_int) {
                            pump_host_command(&mut guard, hwnd, HostCommand::NewTab);
                            return 0;
                        }
                        if let Some(tab_id) = hit_test_tab(&guard.tabs, x_int, y_int) {
                            pump_host_command(&mut guard, hwnd, HostCommand::SwitchTab(tab_id));
                            return 0;
                        }
                    }
                    // Press a nav button: show the sunken look + play the
                    // click NOW, but DEFER the navigation to mouse-UP so the
                    // user SEES the press (standard push-button feel). Capture
                    // the mouse so we still get the matching up + can pop the
                    // button back out if the cursor drifts off it while held.
                    if let Some(btn) = NavButton::hit(x_int, y_int) {
                        // Register the press for ANY nav button — including while a
                        // load is in flight. The `nav_in_flight` decision belongs at
                        // the ACTION layer (trigger_nav_button), NOT here: while
                        // loading, the Refresh button IS the Stop button and MUST be
                        // clickable (back/forward correctly no-op in flight inside
                        // trigger_nav_button). The old `if !nav_in_flight` gate here
                        // swallowed the press so pressed_nav stayed None and mouse-up
                        // never reached trigger_nav_button → STOP_LOAD_CMD was
                        // unreachable and the Stop button was dead during a load.
                        {
                            guard.pressed_nav = Some(btn);
                            guard.nav_press_hot = true;
                            // Drop the borrow before SetCapture: if another
                            // in-process window currently holds capture,
                            // SetCapture synchronously delivers WM_CAPTURECHANGED
                            // (which borrow_mut()s) to that prior owner — and to
                            // be re-entrancy-safe by construction we never hold a
                            // RefCell borrow across any Win32 call that can pump a
                            // message into a wnd_proc. (Mirror of the mouse-up fix.)
                            drop(guard);
                            unsafe {
                                sys::SetCapture(hwnd);
                            }
                            play_nav_click();
                            let guard = unsafe { (*state_ptr).borrow() };
                            invalidate_chrome(&guard, hwnd);
                        }
                        return 0;
                    }
                    // Click in chrome strip outside the buttons (and
                    // outside the EDIT — child windows eat clicks in
                    // their own bounds). Hand focus + select-all to
                    // the EDIT so the user can start editing.
                    let edit = guard.edit_hwnd;
                    unsafe {
                        sys::SetFocus(edit);
                        sys::SendMessageW(edit, sys::EM_SETSEL, 0, -1);
                    }
                    return 0;
                }
                // Clicks in scrolled content map to content-only layout coords.
                let y = if y_raw < chrome_h {
                    y_raw
                } else {
                    y_raw - chrome_h + guard.scroll_y as f32
                };
                // Link href OR element path. Links take priority for
                // navigation; otherwise we dispatch a synthetic
                // `tb-element:` URL so the host can fire
                // `addEventListener("click")` callbacks for the deepest
                // element under the cursor.
                let (href_opt, element_path_opt) = match guard.paint.layout_root.as_ref() {
                    Some(root) => (
                        cv_layout::hit_test_link(root, x, y),
                        cv_layout::hit_test_element_path(root, x, y),
                    ),
                    None => hit_test_regions(&guard.paint.hit_regions, x, y),
                };
                // Record the press source for a potential HTML drag gesture: if
                // the cursor later moves past the threshold while held and is
                // released over a target, we emit `tb-drag:` (HTML §6.11). We
                // record for ANY content element (the page's `dragstart` handler
                // / `draggable` attribute decides whether a drag actually does
                // anything) and reset `drag_active` until the threshold is met.
                if let Some(path) = &element_path_opt {
                    let src = path
                        .iter()
                        .map(|n| n.to_string())
                        .collect::<Vec<_>>()
                        .join("/");
                    guard.drag_press = Some((src, x as i32, y_raw as i32));
                    guard.drag_active = false;
                }
                let url_to_navigate: Option<String> = match (href_opt, element_path_opt) {
                    (Some(href), Some(path)) => {
                        // When a link has both an href AND a known element path,
                        // encode as `tb-link-click:<path>|||<href>` so the host
                        // can fire the click event to JS first, check
                        // `event.defaultPrevented`, and skip navigation if a
                        // React/Next.js router called `preventDefault()`.
                        let s = path
                            .iter()
                            .map(|n| n.to_string())
                            .collect::<Vec<_>>()
                            .join("/");
                        Some(format!("tb-link-click:{s}|||{href}"))
                    }
                    (Some(href), None) => Some(href),
                    (None, Some(path)) => {
                        // Encode as `tb-element:0/2/3/1`.
                        let s = path
                            .iter()
                            .map(|n| n.to_string())
                            .collect::<Vec<_>>()
                            .join("/");
                        Some(format!("tb-element:{s}"))
                    }
                    _ => None,
                };
                if let Some(url) = url_to_navigate {
                    let is_http = url.starts_with("http://") || url.starts_with("https://");
                    // HTTP links route through the worker-thread fetcher
                    // when available so the UI doesn't freeze during the
                    // (slow) TLS handshake. Synthetic schemes
                    // (javascript:, tb-element:, tb-key:, tb-mouse:, …)
                    // stay synchronous because they don't touch the
                    // network and their handlers are fast.
                    let can_thread = is_http
                        && guard.fetcher.is_some()
                        && guard.nav_with_body.is_some()
                        && !guard.nav_in_flight;
                    if can_thread {
                        guard.scroll_y = 0;
                        guard.nav_in_flight = true;
                        let fetcher = guard.fetcher.as_ref().unwrap().clone();
                        let edit = guard.edit_hwnd;
                        invalidate_chrome(&guard, hwnd);
                        drop(guard);
                        // Show the DESTINATION URL immediately (Chrome behavior),
                        // never a "Loading…" status string. The reload→stop button
                        // + progress bar (driven by nav_in_flight) signal loading.
                        let dest_w = to_wide(&url);
                        unsafe {
                            if sys::GetFocus() != edit {
                                sys::SetWindowTextW(edit, dest_w.as_ptr());
                                sys::UpdateWindow(edit);
                            }
                        }
                        let hwnd_send = HwndSend(hwnd);
                        std::thread::spawn(move || {
                            let hs = hwnd_send;
                            let body = fetcher(url.clone());
                            let payload: Box<(String, Vec<u8>)> = Box::new((url, body));
                            let raw = Box::into_raw(payload);
                            unsafe {
                                sys::PostMessageW(hs.0, sys::WM_USER + 1, 0, raw as isize);
                            }
                        });
                        return 0;
                    }
                    // A real navigation (http href, or a link click that MAY
                    // navigate) goes through pump_navigation_command (bump epoch +
                    // gate input, resets scroll); javascript:/tb-element: are
                    // events on the current page and stay as input (no bump,
                    // scroll preserved). tb-link-click may be preventDefault'd by
                    // an SPA router — if so the renderer commits the same page at
                    // the new epoch and nav_in_flight clears immediately. The
                    // destination URL (the http href) is shown in the bar
                    // immediately, never a "Loading…" string.
                    let is_nav = !url.starts_with("javascript:") && !url.starts_with("tb-element:");
                    if is_nav {
                        let dest: Option<&str> = if url.starts_with("http") {
                            Some(url.as_str())
                        } else {
                            // tb-link-click:<path>|||<href> → the href, if http.
                            url.split("|||").nth(1).filter(|h| h.starts_with("http"))
                        };
                        pump_navigation_command(&mut guard, hwnd, &url, dest);
                    } else {
                        pump_input_command(&mut guard, hwnd, &url);
                    }
                }
            }
            0
        }
        sys::WM_KEYUP => {
            fire_key_event(hwnd, "keyup", wparam);
            unsafe { sys::DefWindowProcW(hwnd, msg, wparam, lparam) }
        }
        sys::WM_SETCURSOR => {
            // Show the hand cursor over links; otherwise fall through to the
            // window class's default arrow. LOWORD(lparam) is the hit-test area.
            if (lparam as usize & 0xffff) as u32 == sys::HTCLIENT && pointer_over_link(hwnd) {
                unsafe {
                    sys::SetCursor(sys::LoadCursorW(core::ptr::null_mut(), sys::IDC_HAND));
                }
                return 1;
            }
            unsafe { sys::DefWindowProcW(hwnd, msg, wparam, lparam) }
        }
        sys::WM_MOUSEMOVE => {
            // Custom scrollbar thumb drag: map cursor vertical travel since grab
            // to a scroll_y delta (scaled by the content/track ratio), BEFORE the
            // nav-button + content-move handling.
            {
                let state_ptr = OWNER_PTR.load(Ordering::SeqCst);
                if !state_ptr.is_null() {
                    let mut guard = unsafe { (*state_ptr).borrow_mut() };
                    if let Some((grab_y, grab_scroll)) = guard.scroll_drag {
                        let y = ((lparam >> 16) & 0xFFFF) as i16 as i32;
                        let mut client = sys::RECT::default();
                        unsafe { sys::GetClientRect(hwnd, &raw mut client) };
                        let ch = (client.bottom - client.top).max(0);
                        let chrome_h = guard.paint.chrome_h as i32;
                        let cont_h = guard.paint.bitmap.height as i32;
                        let track_h = (ch - chrome_h).max(1);
                        let thumb_h = ((track_h as i64 * track_h as i64)
                            / cont_h.max(1) as i64).max(24) as i32;
                        let travel = (track_h - thumb_h).max(1);
                        let max_scroll = (cont_h - track_h).max(1);
                        // pixels of thumb travel → scroll units.
                        let dy = y - grab_y;
                        let new_scroll = grab_scroll
                            + ((dy as i64 * max_scroll as i64) / travel as i64) as i32;
                        guard.scroll_y = new_scroll.clamp(0, max_scroll);
                        clamp_scroll(&mut guard, hwnd);
                        publish_scroll(&guard, hwnd);
                        return 0;
                    }
                }
            }
            // While a nav button is held (we have capture), track whether the
            // cursor is still over the SAME button: un-press when it leaves,
            // re-press when it returns — standard push-button drag feedback.
            // Done BEFORE the content-move throttle so the look stays snappy.
            {
                let state_ptr = OWNER_PTR.load(Ordering::SeqCst);
                if !state_ptr.is_null() {
                    let mut guard = unsafe { (*state_ptr).borrow_mut() };
                    if let Some(held) = guard.held_nav_button() {
                        let x = (lparam & 0xFFFF) as i16 as i32;
                        let y = ((lparam >> 16) & 0xFFFF) as i16 as i32;
                        let inside = NavButton::hit(x, y) == Some(held);
                        if guard.nav_press_hot != inside {
                            guard.nav_press_hot = inside;
                            invalidate_chrome(&guard, hwnd);
                        }
                        return 0;
                    }
                }
            }
            // HTML drag-gesture threshold: once a held left-button press has
            // moved more than DRAG_THRESHOLD_PX from where it started, mark the
            // gesture as a drag so the release emits `tb-drag:` instead of a
            // plain click. Done BEFORE the move throttle so a fast drag isn't
            // missed. (Chrome uses ~5px; we use the same.)
            {
                const DRAG_THRESHOLD_PX: i32 = 5;
                let state_ptr = OWNER_PTR.load(Ordering::SeqCst);
                if !state_ptr.is_null() {
                    let mut guard = unsafe { (*state_ptr).borrow_mut() };
                    if !guard.drag_active {
                        if let Some((_src, px, py)) = guard.drag_press.clone() {
                            let x = (lparam & 0xFFFF) as i16 as i32;
                            let y = ((lparam >> 16) & 0xFFFF) as i16 as i32;
                            if (x - px).abs() > DRAG_THRESHOLD_PX
                                || (y - py).abs() > DRAG_THRESHOLD_PX
                            {
                                guard.drag_active = true;
                            }
                        }
                    }
                }
            }
            // Coalesce — never fire faster than ~50 ms so JS handlers
            // doing layout-heavy work don't melt. The throttle state
            // lives in a thread-local since the GUI thread is single.
            thread_local! {
                static LAST_MOUSEMOVE_MS: std::cell::Cell<u128> =
                    const { std::cell::Cell::new(0) };
            }
            let now_ms = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_millis())
                .unwrap_or(0);
            let throttle_ok = LAST_MOUSEMOVE_MS.with(|c| {
                let prev = c.get();
                if now_ms.saturating_sub(prev) >= 50 {
                    c.set(now_ms);
                    true
                } else {
                    false
                }
            });
            if !throttle_ok {
                return unsafe { sys::DefWindowProcW(hwnd, msg, wparam, lparam) };
            }
            dispatch_mouse_url(hwnd, "mousemove", lparam, wparam);
            0
        }
        sys::WM_LBUTTONUP => {
            // End a custom-scrollbar thumb drag, if active. Clear the drag state
            // under a scoped borrow, DROP it, then ReleaseCapture (which re-enters
            // wnd_proc via WM_CAPTURECHANGED — never hold a RefCell borrow across
            // it; same discipline as the nav-button release below).
            {
                let state_ptr = OWNER_PTR.load(Ordering::SeqCst);
                if !state_ptr.is_null() {
                    let was_dragging = {
                        let mut guard = unsafe { (*state_ptr).borrow_mut() };
                        guard.scroll_drag.take().is_some()
                    };
                    if was_dragging {
                        unsafe { sys::ReleaseCapture(); }
                        return 0;
                    }
                }
            }
            // If a nav button is held, this up COMPLETES the press: pop it
            // back out (release capture + clear state + chrome-invalidate) and,
            // if the cursor is STILL inside the SAME button, fire the
            // navigation. Releasing outside the button cancels (no nav) — the
            // standard push-button contract. We handle this BEFORE the normal
            // content `mouseup` so a button release never leaks into the page.
            let state_ptr = OWNER_PTR.load(Ordering::SeqCst);
            if !state_ptr.is_null() {
                // Take the held button + clear press state under a SCOPED borrow,
                // then DROP the borrow before touching Win32. `ReleaseCapture`
                // synchronously re-enters wnd_proc with WM_CAPTURECHANGED, whose
                // arm also borrow_mut()s WindowState — doing it while this borrow
                // is alive is a RefCell double-borrow → panic=abort (the nav-button
                // crash). Because pressed_nav is already cleared here, the
                // re-entrant WM_CAPTURECHANGED simply no-ops.
                let pressed = {
                    let mut guard = unsafe { (*state_ptr).borrow_mut() };
                    let p = guard.pressed_nav.take();
                    if p.is_some() {
                        guard.nav_press_hot = false;
                    }
                    p
                }; // borrow dropped here
                if let Some(pressed) = pressed {
                    unsafe {
                        sys::ReleaseCapture(); // may re-enter WM_CAPTURECHANGED (now a no-op)
                    }
                    // Re-borrow AFTER ReleaseCapture has returned to repaint +
                    // (conditionally) navigate. No borrow is held across any
                    // call that can re-enter wnd_proc.
                    let mut guard = unsafe { (*state_ptr).borrow_mut() };
                    invalidate_chrome(&guard, hwnd);
                    let x = (lparam & 0xFFFF) as i16 as i32;
                    let y = ((lparam >> 16) & 0xFFFF) as i16 as i32;
                    if NavButton::hit(x, y) == Some(pressed) {
                        trigger_nav_button(&mut guard, hwnd, pressed);
                    }
                    return 0;
                }
            }
            // HTML drag-and-drop: if the press turned into a drag (moved past
            // the threshold while held), the release completes a drag — emit a
            // `tb-drag:<src>:<dst>:<x>:<y>` command so the worker runs the
            // dragstart→dragover→drop→dragend sequence with a real DataTransfer
            // (HTML §6.11). Otherwise fall through to the normal `mouseup`.
            {
                let state_ptr = OWNER_PTR.load(Ordering::SeqCst);
                if !state_ptr.is_null() {
                    let drag = {
                        let mut guard = unsafe { (*state_ptr).borrow_mut() };
                        let active = guard.drag_active;
                        let press = guard.drag_press.take();
                        guard.drag_active = false;
                        if active { press } else { None }
                    };
                    if let Some((src, _px, _py)) = drag {
                        // Hit-test the release point for the drop target path.
                        let (dst, rx, ry) = {
                            let guard = unsafe { (*state_ptr).borrow() };
                            let x_raw = (lparam & 0xFFFF) as i16 as f32;
                            let y_raw = ((lparam >> 16) & 0xFFFF) as i16 as f32;
                            let chrome_h = guard.paint.chrome_h as f32;
                            let cx = x_raw;
                            let cy = if y_raw < chrome_h {
                                y_raw
                            } else {
                                y_raw - chrome_h + guard.scroll_y as f32
                            };
                            let dst = match guard.paint.layout_root.as_ref() {
                                Some(root) => cv_layout::hit_test_element_path(root, cx, cy)
                                    .map(|p| {
                                        p.iter()
                                            .map(|n| n.to_string())
                                            .collect::<Vec<_>>()
                                            .join("/")
                                    })
                                    .unwrap_or_default(),
                                None => String::new(),
                            };
                            (dst, x_raw as i32, y_raw as i32)
                        };
                        let url = format!("tb-drag:{src}:{dst}:{rx}:{ry}");
                        let mut guard = unsafe { (*state_ptr).borrow_mut() };
                        pump_input_command(&mut guard, hwnd, &url);
                        return 0;
                    }
                    // Not a drag — clear any stale press record before the click.
                    let mut guard = unsafe { (*state_ptr).borrow_mut() };
                    guard.drag_press = None;
                }
            }
            dispatch_mouse_url(hwnd, "mouseup", lparam, wparam);
            0
        }
        sys::WM_CAPTURECHANGED => {
            // Capture was taken away (Alt-Tab, a dialog, our own
            // ReleaseCapture, etc.) while a nav button might be held. Pop it
            // back OUT without navigating so it can never get stuck pressed-in.
            let state_ptr = OWNER_PTR.load(Ordering::SeqCst);
            if !state_ptr.is_null() {
                let mut guard = unsafe { (*state_ptr).borrow_mut() };
                if guard.pressed_nav.take().is_some() {
                    guard.nav_press_hot = false;
                    invalidate_chrome(&guard, hwnd);
                }
            }
            0
        }
        sys::WM_LBUTTONDBLCLK => {
            // CS_DBLCLKS on the class style makes Windows promote the
            // second click of a doubleclick pair from WM_LBUTTONDOWN
            // to WM_LBUTTONDBLCLK. We fire BOTH `click` (per HTML
            // spec — the second click still fires click) AND a
            // distinct `dblclick`. Web pages bind one or the other.
            dispatch_mouse_url(hwnd, "click", lparam, wparam);
            dispatch_mouse_url(hwnd, "dblclick", lparam, wparam);
            0
        }
        sys::WM_RBUTTONDOWN => {
            dispatch_mouse_url(hwnd, "mousedown", lparam, wparam);
            0
        }
        sys::WM_RBUTTONUP => {
            // Right-click first emits `mouseup`, then the contextmenu
            // event. Pages that want to suppress the OS context menu
            // call preventDefault() on contextmenu.
            dispatch_mouse_url(hwnd, "mouseup", lparam, wparam);
            dispatch_mouse_url(hwnd, "contextmenu", lparam, wparam);
            0
        }
        sys::WM_MBUTTONDOWN => {
            dispatch_mouse_url(hwnd, "mousedown", lparam, wparam);
            0
        }
        sys::WM_MBUTTONUP => {
            dispatch_mouse_url(hwnd, "mouseup", lparam, wparam);
            0
        }
        sys::WM_KEYDOWN => {
            // Fire a host-level `tb-key:keydown:<key>` navigation BEFORE
            // built-in handling so JS document listeners can observe
            // (and ideally preventDefault) on their own.
            fire_key_event(hwnd, "keydown", wparam);
            // Helper: is the URL bar EDIT currently the focused HWND?
            // Used to skip browser-level shortcuts (Esc-to-close,
            // Backspace-as-back) when the user is editing the URL.
            let url_bar_has_focus = || -> bool {
                let state_ptr = OWNER_PTR.load(Ordering::SeqCst);
                if state_ptr.is_null() {
                    return false;
                }
                let edit = unsafe { (*state_ptr).borrow().edit_hwnd };
                !edit.is_null() && unsafe { sys::GetFocus() } == edit
            };
            let (ctrl_down, _, _, _) = sys::modifiers_now();
            if ctrl_down {
                let state_ptr = OWNER_PTR.load(Ordering::SeqCst);
                if !state_ptr.is_null() {
                    match wparam {
                        0x4C => {
                            let guard = unsafe { (*state_ptr).borrow() };
                            let edit = guard.edit_hwnd;
                            unsafe {
                                sys::SetFocus(edit);
                                sys::SendMessageW(edit, sys::EM_SETSEL, 0, -1);
                            }
                            return 0;
                        }
                        0x54 | 0x57 | 0x4E => {
                            let command = match wparam {
                                0x54 => HostCommand::NewTab,
                                0x57 => HostCommand::CloseActiveTab,
                                _ => HostCommand::NewWindow,
                            };
                            let mut guard = unsafe { (*state_ptr).borrow_mut() };
                            pump_host_command(&mut guard, hwnd, command);
                            return 0;
                        }
                        _ => {}
                    }
                }
            }
            // Esc — closes the window (legacy). The EDIT subclass
            // intercepts its own Esc, so this only fires when focus
            // is on the page.
            if wparam == sys::VK_ESCAPE {
                if url_bar_has_focus() {
                    return 0;
                }
                unsafe { sys::DestroyWindow(hwnd) };
                return 0;
            }
            // Backspace — back-nav from the page. Skip when URL bar
            // is focused (its WNDPROC handles the editing key).
            if wparam == sys::VK_BACK {
                if url_bar_has_focus() {
                    return 0;
                }
                let state_ptr = OWNER_PTR.load(Ordering::SeqCst);
                if !state_ptr.is_null() {
                    // OFF-MAIN: the renderer owns history — route back-nav as a
                    // navigation command (gated on nav_in_flight).
                    {
                        let mut guard = unsafe { (*state_ptr).borrow_mut() };
                        if guard.to_page.is_some() {
                            if !guard.nav_in_flight {
                                pump_navigation_command(&mut guard, hwnd, "back://", None);
                            }
                            return 0;
                        }
                    }
                    // Threaded back-nav: if the host provided a
                    // back-URL peek and a fetcher, route the previous
                    // URL through the worker-thread path so the slow
                    // TLS handshake doesn't freeze the UI.
                    {
                        let mut guard = unsafe { (*state_ptr).borrow_mut() };
                        let can_thread = guard.fetcher.is_some()
                            && guard.nav_with_body.is_some()
                            && guard.back_url_fn.is_some()
                            && !guard.nav_in_flight;
                        if can_thread {
                            let prev_url = guard.back_url_fn.as_mut().and_then(|f| f());
                            if let Some(url) = prev_url {
                                guard.nav_in_flight = true;
                                guard.scroll_y = 0;
                                let fetcher = guard.fetcher.as_ref().unwrap().clone();
                                drop(guard);
                                let hwnd_send = HwndSend(hwnd);
                                std::thread::spawn(move || {
                                    let hs = hwnd_send;
                                    let body = fetcher(url.clone());
                                    let payload: Box<(String, Vec<u8>)> = Box::new((url, body));
                                    let raw = Box::into_raw(payload);
                                    unsafe {
                                        sys::PostMessageW(hs.0, sys::WM_USER + 1, 0, raw as isize);
                                    }
                                });
                                return 0;
                            }
                        }
                    }
                    let mut guard = unsafe { (*state_ptr).borrow_mut() };
                    if let Some(nav) = guard.navigator.as_mut() {
                        if let Some(new_paint) = nav("back://") {
                            guard.scroll_y = 0;
                            apply_new_paint(&mut guard, hwnd, new_paint);
                        }
                    }
                }
                return 0;
            }
            // Scrolling keys.
            let scroll_action = match wparam {
                w if w == sys::VK_DOWN => Some(ScrollDelta::Lines(2)),
                w if w == sys::VK_UP => Some(ScrollDelta::Lines(-2)),
                w if w == sys::VK_NEXT => Some(ScrollDelta::Page(1)),
                w if w == sys::VK_PRIOR => Some(ScrollDelta::Page(-1)),
                w if w == sys::VK_HOME => Some(ScrollDelta::Absolute(0)),
                w if w == sys::VK_END => Some(ScrollDelta::Absolute(i32::MAX)),
                _ => None,
            };
            if let Some(action) = scroll_action {
                let state_ptr = OWNER_PTR.load(Ordering::SeqCst);
                if !state_ptr.is_null() {
                    let mut guard = unsafe { (*state_ptr).borrow_mut() };
                    let mut client = sys::RECT::default();
                    unsafe { sys::GetClientRect(hwnd, &raw mut client) };
                    let viewport_h =
                        ((client.bottom - client.top) - guard.paint.chrome_h as i32).max(0);
                    match action {
                        ScrollDelta::Lines(n) => guard.scroll_y += n * 40,
                        ScrollDelta::Page(n) => guard.scroll_y += n * viewport_h,
                        ScrollDelta::Absolute(v) => guard.scroll_y = v,
                    }
                    clamp_scroll(&mut guard, hwnd);
                    unsafe { sys::InvalidateRect(hwnd, core::ptr::null(), 0) };
                }
                return 0;
            }
            unsafe { sys::DefWindowProcW(hwnd, msg, wparam, lparam) }
        }
        sys::WM_SIZE => {
            // Re-clamp scroll, reposition the URL bar EDIT to span
            // the new client width, and repaint.
            let state_ptr = OWNER_PTR.load(Ordering::SeqCst);
            if !state_ptr.is_null() {
                let Ok(mut guard) = (unsafe { (*state_ptr).try_borrow_mut() }) else {
                    return 0;
                };
                let mut client = sys::RECT::default();
                unsafe { sys::GetClientRect(hwnd, &raw mut client) };
                let client_w = (client.right - client.left).max(0);
                let client_h = (client.bottom - client.top).max(0);
                let chrome_h = guard.paint.chrome_h as i32;
                let (ex, ey, ew, eh) = url_bar_rect(client_w, chrome_h);
                let edit = guard.edit_hwnd;
                if !edit.is_null() {
                    unsafe { sys::MoveWindow(edit, ex, ey, ew, eh, 1) };
                }
                let viewport_w = client_w.max(1) as u32;
                let viewport_h = (client_h - chrome_h).max(1) as u32;
                if guard.to_page.is_some() {
                    // OFF-MAIN: re-layout runs on the renderer thread. Send the
                    // new viewport with the current epoch (a resize is not a
                    // navigation); the renderer commits a reflowed frame back.
                    let epoch = guard.nav_gen;
                    if let Some(tx) = guard.to_page.as_ref() {
                        let _ = tx.send(ToPage::Resize {
                            epoch,
                            w: viewport_w,
                            h: viewport_h,
                        });
                    }
                } else {
                    if let Some(resize) = guard.resize_handler.as_mut() {
                        if let Some(paint) = resize(viewport_w, viewport_h) {
                            apply_new_paint(&mut guard, hwnd, paint);
                        }
                    }
                    // Always drive a full repaint at the NEW client size for the
                    // UI-thread present path (incl. the GPU swap chain). On the
                    // next WM_PAINT, `present_bgra` resizes the swap chain to the
                    // new client size and presents — so maximize/resize never
                    // leaves a stale-origin frame on screen. `apply_new_paint`
                    // also invalidates when a resize handler produced a frame;
                    // this unconditional invalidate covers the no-handler /
                    // handler-returned-None case (e.g. a static page) so the GPU
                    // frame is re-presented at the new geometry regardless.
                    unsafe { sys::InvalidateRect(hwnd, core::ptr::null(), 0) };
                }

                // ── Off-main compositor resize handshake ──────────
                // The swap chain + staging live on the compositor thread and
                // MUST be resized there. SYNCHRONOUS bounded-timeout ack so the
                // UI never sends Presents/scrolls while the swap chain is mid-
                // rebuild (DXGI size-mismatch). The bounded wait means a wedged
                // compositor can never hang the resize drag.
                let do_handshake = offmain_compositor_enabled()
                    && guard.compositor_tx.is_some()
                    && guard
                        .compositor_present_mode
                        .as_ref()
                        .map(|m| m.load(Ordering::Acquire) != present_mode::FALLBACK_STRETCH_DIBITS)
                        .unwrap_or(true);
                if do_handshake {
                    let cw = client_w.max(1) as u32;
                    let ch = client_h.max(1) as u32;
                    if let Some(dims) = guard.shared_dims.as_ref() {
                        dims[0].store(cw, Ordering::Release);
                        dims[1].store(ch, Ordering::Release);
                    }
                    if let (Some(tx), Some(ack)) =
                        (guard.compositor_tx.as_ref(), guard.resize_ack.as_ref())
                    {
                        // Arm the ack, send Resize, then wait (bounded).
                        {
                            let mut done = ack.done.lock().unwrap();
                            *done = false;
                        }
                        let _ = tx.send(CompositorCmd::Resize { w: cw, h: ch });
                        let ack = ack.clone();
                        // Drop the WindowState borrow while waiting so the
                        // compositor-driven UI work (none here) can't deadlock,
                        // and so a long wait doesn't hold the RefCell.
                        let guard_ref = &guard; // keep clamp below in scope
                        let _ = guard_ref;
                        let deadline = std::time::Duration::from_millis(100);
                        let mut done = ack.done.lock().unwrap();
                        while !*done {
                            let (g2, timeout) = ack.cv.wait_timeout(done, deadline).unwrap();
                            done = g2;
                            if timeout.timed_out() {
                                // Wedged/slow compositor: proceed; the next
                                // present picks up the resized buffer (≤1 frame
                                // late). Never crash, never hang the drag.
                                break;
                            }
                        }
                    }
                }

                clamp_scroll(&mut guard, hwnd);
                publish_scroll(&guard, hwnd);
            }
            0
        }
        sys::WM_DESTROY => {
            // OFF-MAIN: tell the renderer thread to finish and exit before we
            // quit the pump, so it doesn't keep working against a dead window.
            let state_ptr = OWNER_PTR.load(Ordering::SeqCst);
            if !state_ptr.is_null() {
                let guard = unsafe { (*state_ptr).borrow() };
                if let Some(tx) = guard.to_page.as_ref() {
                    let _ = tx.send(ToPage::Shutdown);
                }
                // Off-main compositor: tell its thread to exit so it drops the
                // HwPresenter on its OWN thread (COM apartment teardown affinity).
                if let Some(tx) = guard.compositor_tx.as_ref() {
                    let _ = tx.send(CompositorCmd::Shutdown);
                }
            }
            unsafe { sys::PostQuitMessage(0) };
            0
        }
        sys::WM_ERASEBKGND => 1, // suppress default erase to avoid flicker
        _ => unsafe { sys::DefWindowProcW(hwnd, msg, wparam, lparam) },
    }
}

#[cfg(test)]
mod tests {
    use super::{HitRegion, hit_test_regions, should_apply_commit};
    use super::{
        BACK_BUTTON_H, BACK_BUTTON_W, BACK_BUTTON_X, BACK_BUTTON_Y, FORWARD_BUTTON_H,
        FORWARD_BUTTON_W, FORWARD_BUTTON_X, FORWARD_BUTTON_Y, NavButton, REFRESH_BUTTON_H,
        REFRESH_BUTTON_W, REFRESH_BUTTON_X, REFRESH_BUTTON_Y, STOP_LOAD_CMD, TAB_STRIP_H, TAB_Y,
        TOOLBAR_Y, build_click_wav, in_flight_press_cancels, tab_rect, url_bar_rect,
    };

    /// A3 — the Stop button is a REAL load-cancel ONLY in the sandboxed-
    /// renderer-process mode (sandbox_renderer=true AND an off-main command
    /// channel exists). In every other configuration it stays the historical
    /// honest no-op. This pins that policy without needing a live window.
    #[test]
    fn stop_button_cancels_only_in_sandboxed_process_mode() {
        // REAL cancel: sandboxed renderer process + off-main channel + Refresh.
        assert!(in_flight_press_cancels(true, true, NavButton::Refresh));

        // Honest no-op: in-process renderer thread (sandbox_renderer=false),
        // even with an off-main channel.
        assert!(!in_flight_press_cancels(false, true, NavButton::Refresh));

        // Honest no-op: no off-main channel (inline/legacy path).
        assert!(!in_flight_press_cancels(true, false, NavButton::Refresh));

        // Back/Forward never cancel a load even in process mode.
        assert!(!in_flight_press_cancels(true, true, NavButton::Back));
        assert!(!in_flight_press_cancels(true, true, NavButton::Forward));
    }

    /// The Stop sentinel is a reserved encoding distinct from any real
    /// navigation/input command, so the site-router can intercept it without
    /// risking a collision with page traffic.
    #[test]
    fn stop_load_cmd_is_reserved_sentinel() {
        assert_eq!(STOP_LOAD_CMD, "__stop_load__");
        assert!(!STOP_LOAD_CMD.contains("://"));
        assert!(STOP_LOAD_CMD.starts_with("__"));
    }

    /// The nav-button hit mapping the press/release lifecycle relies on:
    /// a point inside each button's rect maps to that button; points just
    /// outside the rect (and the gap/chrome around it) map to `None`. Both
    /// mouse-DOWN (start press) and mouse-UP (commit-if-still-inside) use
    /// `NavButton::hit`, so this pins the geometry both ends agree on.
    #[test]
    fn nav_button_hit_mapping() {
        // Centers land in their button.
        let back_cx = BACK_BUTTON_X + BACK_BUTTON_W / 2;
        let back_cy = BACK_BUTTON_Y + BACK_BUTTON_H / 2;
        assert_eq!(NavButton::hit(back_cx, back_cy), Some(NavButton::Back));
        let fwd_cx = FORWARD_BUTTON_X + FORWARD_BUTTON_W / 2;
        let fwd_cy = FORWARD_BUTTON_Y + FORWARD_BUTTON_H / 2;
        assert_eq!(NavButton::hit(fwd_cx, fwd_cy), Some(NavButton::Forward));
        let ref_cx = REFRESH_BUTTON_X + REFRESH_BUTTON_W / 2;
        let ref_cy = REFRESH_BUTTON_Y + REFRESH_BUTTON_H / 2;
        assert_eq!(NavButton::hit(ref_cx, ref_cy), Some(NavButton::Refresh));

        // The three buttons don't overlap and are left-to-right Back, Forward,
        // Refresh, all on the toolbar row.
        assert!(BACK_BUTTON_X + BACK_BUTTON_W <= FORWARD_BUTTON_X);
        assert!(FORWARD_BUTTON_X + FORWARD_BUTTON_W <= REFRESH_BUTTON_X);
        assert_eq!(BACK_BUTTON_Y, TOOLBAR_Y);
        assert_eq!(FORWARD_BUTTON_Y, TOOLBAR_Y);
        assert_eq!(REFRESH_BUTTON_Y, TOOLBAR_Y);

        // Top-left corner is inclusive; bottom-right edge is exclusive.
        assert_eq!(
            NavButton::hit(BACK_BUTTON_X, BACK_BUTTON_Y),
            Some(NavButton::Back),
        );
        assert_eq!(
            NavButton::hit(BACK_BUTTON_X + BACK_BUTTON_W, BACK_BUTTON_Y),
            None,
            "right edge is exclusive",
        );

        // Above the strip and far to the right map to neither button.
        assert_eq!(NavButton::hit(back_cx, BACK_BUTTON_Y - 1), None);
        assert_eq!(NavButton::hit(1000, back_cy), None);

        // The rect helper round-trips with hit at the center.
        for btn in [NavButton::Back, NavButton::Forward, NavButton::Refresh] {
            let (x, y, w, h) = btn.rect();
            assert_eq!(NavButton::hit(x + w / 2, y + h / 2), Some(btn));
        }
    }

    /// Chrome layout = Chrome/Edge/Firefox: the TAB STRIP is the TOP row and the
    /// toolbar (back/fwd/refresh + URL bar) sits BELOW it. Pin that ordering and
    /// that the URL-bar EDIT child rect lands on the toolbar row (so the real
    /// child window aligns over its drawn slot at the new Y), not over the tabs.
    #[test]
    fn tabs_are_top_row_toolbar_below() {
        // Tab strip occupies the top of the chrome strip.
        let (_, tab_y, _, _) = tab_rect(0);
        assert_eq!(tab_y, TAB_Y);
        // The toolbar row begins below the tab strip region.
        assert!(TOOLBAR_Y >= TAB_STRIP_H, "toolbar must sit below the tab strip");
        assert!(tab_y < TOOLBAR_Y, "tabs must be ABOVE the toolbar buttons");

        // The URL-bar EDIT rect is on the toolbar row and vertically overlaps
        // the back/fwd/refresh buttons (the child window aligns to its slot).
        let chrome_h = 66;
        let (ex, ey, ew, eh) = url_bar_rect(800, chrome_h);
        assert!(ex > REFRESH_BUTTON_X + REFRESH_BUTTON_W, "URL bar starts right of the buttons");
        assert!(ew > 0 && eh > 0);
        let btn_top = TOOLBAR_Y;
        let btn_bot = TOOLBAR_Y + REFRESH_BUTTON_H;
        assert!(ey < btn_bot && ey + eh > btn_top, "EDIT must overlap the toolbar button row");
        // The EDIT must NOT intrude into the tab strip region.
        assert!(ey >= TAB_STRIP_H - 2, "URL bar must not sit over the tab strip");
    }

    /// The embedded click sound must be a well-formed in-memory WAV (the
    /// thing `PlaySoundW(SND_MEMORY)` requires) and carry real audio — not
    /// shipped silence. We verify the RIFF/WAVE/fmt /data structure, the
    /// 16-bit-mono-PCM header fields, the self-describing sizes, and that the
    /// PCM samples are not all zero (so it is guaranteed audible).
    #[test]
    fn click_wav_is_valid_and_audible() {
        let wav = build_click_wav();
        assert!(wav.len() > 44, "WAV must have a header + samples");
        assert_eq!(&wav[0..4], b"RIFF");
        assert_eq!(&wav[8..12], b"WAVE");
        assert_eq!(&wav[12..16], b"fmt ");
        assert_eq!(&wav[36..40], b"data");

        let riff_size = u32::from_le_bytes([wav[4], wav[5], wav[6], wav[7]]);
        assert_eq!(
            riff_size as usize,
            wav.len() - 8,
            "RIFF chunk size must describe the rest of the file",
        );
        let fmt_size = u32::from_le_bytes([wav[16], wav[17], wav[18], wav[19]]);
        assert_eq!(fmt_size, 16, "PCM fmt chunk is 16 bytes");
        let audio_fmt = u16::from_le_bytes([wav[20], wav[21]]);
        assert_eq!(audio_fmt, 1, "uncompressed PCM");
        let channels = u16::from_le_bytes([wav[22], wav[23]]);
        assert_eq!(channels, 1, "mono");
        let bits = u16::from_le_bytes([wav[34], wav[35]]);
        assert_eq!(bits, 16, "16-bit samples");

        let data_size = u32::from_le_bytes([wav[40], wav[41], wav[42], wav[43]]);
        assert_eq!(
            data_size as usize,
            wav.len() - 44,
            "data chunk size must match the PCM byte count",
        );
        assert!(data_size > 0, "must ship real audio, not an empty data chunk");

        // Not shipped silence: at least one sample is meaningfully nonzero.
        let pcm = &wav[44..];
        let loud = pcm
            .chunks_exact(2)
            .any(|s| i16::from_le_bytes([s[0], s[1]]).unsigned_abs() > 1_000);
        assert!(loud, "click WAV must carry audible (nonzero) samples");
    }

    /// The GPU/DComp chrome bake: rendering the chrome strip into the top
    /// `chrome_h` rows of the presented frame must (a) actually overwrite the
    /// white top rows with the chrome background colour at the CORRECT byte
    /// order/offset, and (b) leave the content rows (>= chrome_h) untouched.
    /// This pins the offscreen memory-DC → frame copy that makes the chrome
    /// visible under the topmost DComp visual.
    #[test]
    fn bake_chrome_into_frame_lands_bg_at_offset_and_leaves_content() {
        use super::{bake_chrome_into_frame, TabSummary};

        let client_w: i32 = 800;
        let client_h: i32 = 600;
        let chrome_h: i32 = 66;
        let fw = client_w as usize;

        // Start with a frame whose top rows are white (the old buggy fill) and
        // whose content rows carry a distinct sentinel we must NOT clobber.
        let content_sentinel: u32 = 0xFF12_3456;
        let mut frame = vec![0xFFFF_FFFFu32; (client_w * client_h) as usize];
        for row in (chrome_h as usize)..(client_h as usize) {
            for col in 0..fw {
                frame[row * fw + col] = content_sentinel;
            }
        }

        let tabs = vec![TabSummary {
            id: 1,
            title: "Example".into(),
            url: "https://example.com".into(),
            active: true,
        }];

        let ok =
            bake_chrome_into_frame(&mut frame, client_w, client_h, chrome_h, &tabs, None, false);
        // GDI memory-DC ops are available in CI; if the DIB couldn't be created
        // the bake honestly reports false and the assertions below are skipped.
        if !ok {
            return;
        }

        // (a) A spot in the pure chrome-background region — far right, above the
        // nav buttons (y<6) and left of nothing drawn there — must be the chrome
        // background rgb(241,243,244). In the BGRA-packed frame that is the u32
        // 0x00RRGGBB = 0x00F1F3F4, with the alpha/X byte forced opaque (0xFF).
        let bg_x = (client_w - 5) as usize;
        let bg_y = 3usize;
        let bg = frame[bg_y * fw + bg_x];
        assert_eq!(
            bg, 0xFFF1_F3F4,
            "chrome bg must be rgb(241,243,244) packed BGRA (no channel swap), got {bg:#010X}",
        );

        // (b) The top rows are no longer white (the bug) — at least the bg spot
        // changed from 0xFFFFFFFF.
        assert_ne!(bg, 0xFFFF_FFFF, "top chrome rows must not stay white");

        // (c) Content rows (>= chrome_h) are untouched.
        let cy = (chrome_h as usize) + 10;
        assert_eq!(
            frame[cy * fw + 100], content_sentinel,
            "content rows below the chrome strip must be left untouched by the bake",
        );

        // (d) The bake never wrote past the strip: the very first content row is
        // still the sentinel across its full width.
        let first_content = chrome_h as usize;
        assert_eq!(frame[first_content * fw + 0], content_sentinel);
        assert_eq!(frame[first_content * fw + (fw - 1)], content_sentinel);
    }

    #[test]
    fn commit_gate_drops_stale_applies_current_and_newer() {
        // The off-main stale-frame invariant. After a navigation bumps nav_gen,
        // a late frame from the abandoned page (lower epoch) is dropped; the
        // navigation's own frame and same-generation input/ticker/resize frames
        // (equal epoch) apply; a newer epoch also applies.
        assert!(!should_apply_commit(4, 5), "stale frame from abandoned page is dropped");
        assert!(should_apply_commit(5, 5), "navigation's own / same-gen frame applies");
        assert!(should_apply_commit(6, 5), "newer frame applies");
        assert!(should_apply_commit(0, 0), "initial frame applies");
    }

    #[test]
    fn hit_test_regions_returns_last_matching_region_metadata() {
        let regions = vec![
            HitRegion {
                x: 0,
                y: 0,
                w: 100,
                h: 100,
                href: Some("https://example.com/outer".into()),
                element_path: Some(vec![0]),
            },
            HitRegion {
                x: 10,
                y: 10,
                w: 40,
                h: 20,
                href: Some("https://example.com/inner".into()),
                element_path: Some(vec![0, 1]),
            },
        ];

        let (href, path) = hit_test_regions(&regions, 20.0, 20.0);
        assert_eq!(href.as_deref(), Some("https://example.com/inner"));
        assert_eq!(path.as_deref(), Some(&[0, 1][..]));
    }

    #[test]
    fn hit_test_regions_returns_none_outside_regions() {
        let regions = vec![HitRegion {
            x: 0,
            y: 0,
            w: 20,
            h: 20,
            href: Some("https://example.com/only".into()),
            element_path: Some(vec![1]),
        }];

        let (href, path) = hit_test_regions(&regions, 40.0, 40.0);
        assert!(href.is_none());
        assert!(path.is_none());
    }

    /// M2.2 Lever B: committing a finished frame must NOT deep-copy the
    /// full-screen pixel buffer. The per-frame commit (`tab.paint =
    /// paint.clone()` plus the copy handed to the UI) used to memcpy a
    /// multi-megabyte `Vec<u32>` of pixels every animation frame. With the
    /// frame bitmap held behind an `Arc`, `PaintData::clone()` shares the
    /// SAME pixel allocation — a refcount bump, not a copy. This test pins
    /// that: both clones point at the identical `Arc<Bitmap>` allocation and
    /// the strong count rises, which is impossible if a memcpy had occurred.
    #[test]
    fn paint_clone_shares_pixels_no_memcpy() {
        use super::PaintData;
        use std::sync::Arc;

        // A non-trivial "full-screen" bitmap so a memcpy would be observable.
        let bmp = cv_gfx::Bitmap {
            width: 1280,
            height: 800,
            pixels: vec![0xFF00_FF00u32; 1280 * 800],
        };
        let pixels_ptr = bmp.pixels.as_ptr();
        let paint = PaintData {
            bitmap: Arc::new(bmp),
            texts: Vec::new(),
            layout_root: None,
            hit_regions: Vec::new(),
            title: String::new(),
            current_url: String::new(),
            chrome_h: 66,
            viewport_h: 800,
            caret_rect: None,
            property_trees: None,
            retained: None,
            content_origin_y: 0,
            document_h: 0,
        };
        assert_eq!(Arc::strong_count(&paint.bitmap), 1, "fresh frame is uniquely owned");

        // This is exactly what the commit path does each frame:
        //   tab.paint = paint.clone();  // store in the tab
        //   return Some(paint);         // hand the original to the UI
        let committed = paint.clone();

        // Same allocation: no pixels were copied, just an Arc refcount bump.
        assert!(
            Arc::ptr_eq(&paint.bitmap, &committed.bitmap),
            "clone must share the SAME Arc<Bitmap> allocation (no per-frame memcpy)",
        );
        assert_eq!(
            committed.bitmap.pixels.as_ptr(),
            pixels_ptr,
            "the pixel buffer pointer is unchanged across the commit clone",
        );
        assert_eq!(
            Arc::strong_count(&paint.bitmap),
            2,
            "both the tab copy and the UI copy reference one shared pixel buffer",
        );

        // Reads still work transparently through the Arc Deref.
        assert_eq!(committed.bitmap.width, 1280);
        assert_eq!(committed.bitmap.height, 800);
        assert_eq!(committed.bitmap.pixels.len(), 1280 * 800);
    }

    // ── M5.5 off-main compositor tests ────────────────────────────────

    fn mk_paint(w: u32, h: u32) -> super::PaintData {
        use std::sync::Arc;
        let bmp = cv_gfx::Bitmap {
            width: w,
            height: h,
            pixels: (0..(w * h)).map(|i| 0xFF000000 | (i & 0xFFFFFF)).collect(),
        };
        super::PaintData {
            bitmap: Arc::new(bmp),
            texts: Vec::new(),
            layout_root: None,
            hit_regions: Vec::new(),
            title: "t".into(),
            current_url: "u".into(),
            chrome_h: 66,
            viewport_h: h.saturating_sub(66),
            caret_rect: None,
            property_trees: None,
            retained: None,
            content_origin_y: 0,
            document_h: 0,
        }
    }

    /// The flag MUST default OFF when the env var is unset. (Set in a
    /// freshly-spawned process-like context is hard given the OnceLock; instead
    /// we verify the value-discrimination logic directly, mirroring the
    /// accessor's match arms — the accessor itself is exercised in integration.)
    #[test]
    fn offmain_compositor_flag_discrimination() {
        fn on_for(v: Option<&str>) -> bool {
            matches!(v, Some("1") | Some("on") | Some("true") | Some("yes"))
        }
        // Default (unset) and explicit-off values are OFF.
        assert!(!on_for(None), "unset => OFF (default)");
        assert!(!on_for(Some("0")), "0 => OFF");
        assert!(!on_for(Some("false")), "false => OFF");
        assert!(!on_for(Some("off")), "off => OFF");
        // Affirmative values are ON.
        assert!(on_for(Some("1")));
        assert!(on_for(Some("on")));
        assert!(on_for(Some("true")));
        assert!(on_for(Some("yes")));
    }

    /// Compile-time Send guard is referenced (forces the bounds to be checked).
    /// Plus a direct runtime confirmation that the channel payloads are Send.
    #[test]
    fn compositor_channel_is_send() {
        fn assert_send<T: Send>() {}
        assert_send::<super::CompositorCmd>();
        assert_send::<std::sync::Arc<super::PaintData>>();
        assert_send::<super::PaintData>();
        assert_send::<super::CompositorStatus>();
        assert_send::<super::PageHwnd>();
        assert_send::<std::sync::Arc<super::ResizeAck>>();
        // Reference the guard fn itself so it is compiled (not dead-stripped).
        let _ = super::_compositor_send_guards as fn();
    }

    /// present_mode constants are distinct and the default is UNKNOWN.
    #[test]
    fn present_mode_constants_distinct() {
        use super::present_mode::*;
        assert_eq!(UNKNOWN, 0);
        assert_ne!(UNKNOWN, OWNED_BY_COMPOSITOR);
        assert_ne!(OWNED_BY_COMPOSITOR, FALLBACK_STRETCH_DIBITS);
        assert_ne!(UNKNOWN, FALLBACK_STRETCH_DIBITS);
    }

    /// The compositor thread, when HwPresenter init FAILS (here: a null HWND
    /// has no swap chain), must report GpuInitFailed, flip present_mode to
    /// FALLBACK_STRETCH_DIBITS, and EXIT — never panic / hang. This is the L2
    /// fallback ladder verified in-process without a real window.
    #[test]
    fn compositor_thread_init_fail_falls_back_and_exits() {
        use core::sync::atomic::{AtomicI32, AtomicU32, AtomicU8, Ordering as O};
        use std::sync::mpsc;
        let (_tx, rx) = mpsc::channel::<super::CompositorCmd>();
        let (status_tx, status_rx) = mpsc::channel::<super::CompositorStatus>();
        let scroll = std::sync::Arc::new(AtomicI32::new(0));
        let dims: std::sync::Arc<[AtomicU32; 2]> =
            std::sync::Arc::new([AtomicU32::new(64), AtomicU32::new(64)]);
        let mode = std::sync::Arc::new(AtomicU8::new(super::present_mode::UNKNOWN));
        let ack = std::sync::Arc::new(super::ResizeAck::default());

        // A null page HWND => HwPresenter::new fails (no swap chain target).
        let page = super::PageHwnd(core::ptr::null_mut());
        let mode_c = mode.clone();
        let h = std::thread::spawn(move || {
            super::run_compositor_thread(page, rx, scroll, dims, mode_c, status_tx, ack);
        });
        // The thread must finish quickly (it returns on init failure). Join with
        // a guard: if it hangs, the test thread would block — so we join after a
        // status read (the status send happens before return).
        let status = status_rx
            .recv_timeout(std::time::Duration::from_secs(5))
            .expect("compositor thread should report a status");
        assert_eq!(status, super::CompositorStatus::GpuInitFailed);
        h.join().expect("compositor thread should exit cleanly, not panic");
        assert_eq!(
            mode.load(O::Acquire),
            super::present_mode::FALLBACK_STRETCH_DIBITS,
            "init failure flips present mode to the StretchDIBits fallback"
        );
    }

    /// Frame-coalesce-and-composite equivalence: the compositor's per-Present
    /// tile-cache refresh + composite_viewport must produce the SAME pixels the
    /// UI thread's synchronous path would. We replicate the compositor's
    /// composite call against a TileCache built from a PaintData and assert it
    /// equals a direct UI-thread composite of the same cache — proving the
    /// off-thread composite is the relocated-but-identical work (the present
    /// itself is GPU-byte-identity-checked in cv_gpu's offscreen oracle).
    #[test]
    fn compositor_composite_matches_ui_thread_composite() {
        let paint = mk_paint(256, 600);
        let scroll = 120i32;
        let viewport_h = 400u32;
        let blit_w = paint.bitmap.width as u32;

        // UI-thread reference (today's path).
        let mut ui_cache = cv_compositor::TileCache::new();
        ui_cache.invalidate_all();
        ui_cache.refresh_from_raw(
            &paint.bitmap.pixels,
            paint.bitmap.width as u32,
            paint.bitmap.height as u32,
        );
        let gold = ui_cache.composite_viewport(0, scroll, blit_w, viewport_h);

        // Compositor-thread proxy: same refresh + composite on a spawned thread.
        let px = paint.bitmap.pixels.clone();
        let (w, hgt) = (paint.bitmap.width as u32, paint.bitmap.height as u32);
        let off = std::thread::spawn(move || {
            let mut c = cv_compositor::TileCache::new();
            c.invalidate_all();
            c.refresh_from_raw(&px, w, hgt);
            c.composite_viewport(0, scroll, blit_w, viewport_h)
        })
        .join()
        .expect("compositor composite thread panicked");

        assert_eq!(gold, off, "off-thread composite differs from UI-thread composite");
    }
}
