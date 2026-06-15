//! Cross-process frame + command marshaling for the persistent sandboxed
//! renderer (Phase A2).
//!
//! The in-process off-main renderer hands the UI a `cv_ui::FromPage::Commit`
//! carrying a full `cv_ui::PaintData` and a `Vec<cv_ui::TabSummary>` — pure
//! `Send` data, but the `PaintData` contains an `Arc<Bitmap>` and several
//! `cv_ui`-typed sub-structures that `cv_ipc` (which has no `cv_ui`
//! dependency) cannot name. This module is the byte codec that lives in
//! `conclave` (the one crate that sees BOTH `cv_ipc` and `cv_ui`) and
//! turns the marshalable subset of a committed frame into the opaque
//! payload carried by `cv_ipc::Msg::CommitFrame`, and back.
//!
//! ## What crosses, and why a subset
//!
//! A `PaintData` has render-internal optimization fields the BROWSER never
//! consumes when compositing a renderer-produced frame:
//!   * `layout_root` / `property_trees` / `retained` are inputs to the
//!     renderer's own incremental raster + compositor fast paths. The
//!     browser composites the finished `bitmap` and uses `hit_regions` as
//!     the hit-test surface, so those heavy trees do not need to cross the
//!     pipe. (Dropping them also avoids serializing the entire layout tree
//!     every frame — the bitmap copy is already the dominant cost.)
//! What DOES cross is everything the browser-process compositor + input
//! router actually reads: the BGRA `bitmap`, the chrome `texts` overlays
//! (URL bar text, nav glyphs), the `hit_regions` (clickable rects +
//! element paths), `title` / `current_url`, `chrome_h` / `viewport_h`, and
//! the `caret_rect`. Plus the `tabs` summary and the navigation `epoch`.
//!
//! ## Frame transport choice (documented honestly)
//!
//! V1 is **pixels-over-pipe**: the BGRA buffer travels inside this payload,
//! over the named pipe, and the browser reconstructs an owned `Bitmap` from
//! it (one extra copy vs the in-process `Arc` refcount-bump). For a
//! full-document frame (e.g. 1920×5000 ≈ 38 MB) that is a real per-frame
//! cost the in-process path avoids; it is acceptable for V1 because commits
//! happen only on actual change (nav / input / animation frames), not every
//! idle 16 ms tick. The next sub-milestone replaces the in-payload pixels
//! with a shared-memory section (the pipe then carries only a small
//! `FrameReady { section, w, h, len }`); see the renderer-process design
//! notes. The codec is structured so that swap touches only `encode_bitmap`
//! / `decode_bitmap`.

use cv_ipc::{Decode, DecodeError, Encode, Reader, Writer};

/// Codec format version. Both ends are the same binary (the renderer child
/// is a re-exec of the browser), so a mismatch can only happen across an
/// upgrade-in-place; we bump this and let the protocol-version handshake
/// (`cv_ipc::PROTOCOL_VERSION`) gate compatibility. Carried at the front of
/// every frame payload so a stale decoder fails loudly rather than
/// misreading fields.
// v2: added the CSS UI 4 `caret_color` field after `caret_rect`.
const FRAME_CODEC_VERSION: u32 = 2;

/// Mirror of `cv_ipc::renderer_proto`'s pixel-buffer guard so an attacker
/// (or a corrupt frame) cannot drive an unbounded `Bitmap` allocation in
/// the browser process when reconstructing the frame. Matches the
/// `MAX_PAINT_DIMENSION` / `MAX_PAINT_BYTES` policy.
const MAX_FRAME_DIMENSION: u32 = 16_384;
const MAX_FRAME_BYTES: usize = 256 * 1024 * 1024;

fn frame_bytes_ok(width: u32, height: u32, declared_pixels: usize) -> bool {
    if width == 0 || height == 0 {
        return false;
    }
    if width > MAX_FRAME_DIMENSION || height > MAX_FRAME_DIMENSION {
        return false;
    }
    let Some(pixels) = (width as usize).checked_mul(height as usize) else {
        return false;
    };
    if pixels != declared_pixels {
        return false;
    }
    let Some(bytes) = pixels.checked_mul(4) else {
        return false;
    };
    bytes <= MAX_FRAME_BYTES
}

// ---------------------------------------------------------------------------
// HostCommand <-> bytes (Browser → Renderer)
// ---------------------------------------------------------------------------

/// Serialize a `cv_ui::HostCommand` into the opaque payload carried by
/// `cv_ipc::Msg::HostCmd`. A tiny 1-byte tag (+ optional u64) — hand-rolled
/// rather than routing through a general Value codec to keep `cv_ipc`
/// dependency-free and the wire trivially auditable.
pub fn encode_host_command(cmd: &cv_ui::HostCommand) -> Vec<u8> {
    let mut w = Writer::new();
    match cmd {
        cv_ui::HostCommand::NewTab => w.write_u8(0),
        cv_ui::HostCommand::CloseActiveTab => w.write_u8(1),
        cv_ui::HostCommand::SwitchTab(id) => {
            w.write_u8(2);
            w.write_u64(*id);
        }
        cv_ui::HostCommand::NewWindow => w.write_u8(3),
    }
    w.into_bytes()
}

/// Decode a `cv_ui::HostCommand` from the `Msg::HostCmd` payload.
pub fn decode_host_command(bytes: &[u8]) -> Result<cv_ui::HostCommand, DecodeError> {
    let mut r = Reader::new(bytes);
    let tag = r.read_u8()?;
    Ok(match tag {
        0 => cv_ui::HostCommand::NewTab,
        1 => cv_ui::HostCommand::CloseActiveTab,
        2 => cv_ui::HostCommand::SwitchTab(r.read_u64()?),
        3 => cv_ui::HostCommand::NewWindow,
        _ => return Err(DecodeError::OutOfRange),
    })
}

// ---------------------------------------------------------------------------
// Frame (PaintData subset + tabs) <-> bytes (Renderer → Browser)
// ---------------------------------------------------------------------------

fn encode_text_align(w: &mut Writer, a: cv_ui::TextAlign) {
    w.write_u8(match a {
        cv_ui::TextAlign::Left => 0,
        cv_ui::TextAlign::Center => 1,
        cv_ui::TextAlign::Right => 2,
    });
}

fn decode_text_align(r: &mut Reader<'_>) -> Result<cv_ui::TextAlign, DecodeError> {
    Ok(match r.read_u8()? {
        0 => cv_ui::TextAlign::Left,
        1 => cv_ui::TextAlign::Center,
        2 => cv_ui::TextAlign::Right,
        _ => return Err(DecodeError::OutOfRange),
    })
}

fn encode_text_item(w: &mut Writer, t: &cv_ui::TextItem) {
    w.write_i32(t.x);
    w.write_i32(t.y);
    w.write_i32(t.w);
    w.write_i32(t.h);
    w.write_i32(t.font_size_px);
    w.write_bool(t.bold);
    w.write_i32(i32::from(t.font_weight));
    w.write_bool(t.italic);
    match &t.font_family {
        Some(f) => {
            w.write_bool(true);
            w.write_str(f);
        }
        None => w.write_bool(false),
    }
    w.write_u8(t.color_rgb.0);
    w.write_u8(t.color_rgb.1);
    w.write_u8(t.color_rgb.2);
    w.write_u8(t.color_alpha);
    w.write_str(&t.text);
    encode_text_align(w, t.align);
    w.write_i32(t.letter_spacing_px);
    w.write_bool(t.is_chrome);
}

fn decode_text_item(r: &mut Reader<'_>) -> Result<cv_ui::TextItem, DecodeError> {
    let x = r.read_i32()?;
    let y = r.read_i32()?;
    let wv = r.read_i32()?;
    let h = r.read_i32()?;
    let font_size_px = r.read_i32()?;
    let bold = r.read_bool()?;
    let font_weight = r.read_i32()?.clamp(0, 1000) as u16;
    let italic = r.read_bool()?;
    let font_family = if r.read_bool()? {
        Some(r.read_str()?)
    } else {
        None
    };
    let cr = r.read_u8()?;
    let cg = r.read_u8()?;
    let cb = r.read_u8()?;
    let color_alpha = r.read_u8()?;
    let text = r.read_str()?;
    let align = decode_text_align(r)?;
    let letter_spacing_px = r.read_i32()?;
    let is_chrome = r.read_bool()?;
    Ok(cv_ui::TextItem {
        x,
        y,
        w: wv,
        h,
        font_size_px,
        bold,
        font_weight,
        italic,
        font_family,
        color_rgb: (cr, cg, cb),
        color_alpha,
        text,
        align,
        letter_spacing_px,
        is_chrome,
        // Only browser-CHROME text overlays (URL bar, nav glyphs) cross this
        // codec; content gradient text (`background-clip:text`) is already
        // baked into the BGRA `bitmap` upstream and never travels as a text
        // item, so `None` here is lossless. See module docs ("chrome texts").
        text_gradient: None,
    })
}

fn encode_hit_region(w: &mut Writer, hr: &cv_ui::HitRegion) {
    w.write_i32(hr.x);
    w.write_i32(hr.y);
    w.write_i32(hr.w);
    w.write_i32(hr.h);
    match &hr.href {
        Some(h) => {
            w.write_bool(true);
            w.write_str(h);
        }
        None => w.write_bool(false),
    }
    match &hr.element_path {
        Some(path) => {
            w.write_bool(true);
            w.write_u32(path.len() as u32);
            for &seg in path {
                w.write_u32(seg as u32);
            }
        }
        None => w.write_bool(false),
    }
}

fn decode_hit_region(r: &mut Reader<'_>) -> Result<cv_ui::HitRegion, DecodeError> {
    let x = r.read_i32()?;
    let y = r.read_i32()?;
    let wv = r.read_i32()?;
    let h = r.read_i32()?;
    let href = if r.read_bool()? {
        Some(r.read_str()?)
    } else {
        None
    };
    let element_path = if r.read_bool()? {
        let n = r.read_u32()? as usize;
        let mut path = Vec::with_capacity(n);
        for _ in 0..n {
            path.push(r.read_u32()? as usize);
        }
        Some(path)
    } else {
        None
    };
    Ok(cv_ui::HitRegion {
        x,
        y,
        w: wv,
        h,
        href,
        element_path,
    })
}

fn encode_tab_summary(w: &mut Writer, t: &cv_ui::TabSummary) {
    w.write_u64(t.id);
    w.write_str(&t.title);
    w.write_str(&t.url);
    w.write_bool(t.active);
}

fn decode_tab_summary(r: &mut Reader<'_>) -> Result<cv_ui::TabSummary, DecodeError> {
    Ok(cv_ui::TabSummary {
        id: r.read_u64()?,
        title: r.read_str()?,
        url: r.read_str()?,
        active: r.read_bool()?,
    })
}

/// Encode the bitmap. The renderer hands us `&[u32]` (0xAARRGGBB in the
/// in-memory `Bitmap`), which we serialize as raw little-endian words. This
/// is the single function the shared-memory follow-up will replace with a
/// section handle; everything else in the payload stays.
fn encode_bitmap(w: &mut Writer, bmp: &cv_gfx::Bitmap) {
    w.write_u32(bmp.width);
    w.write_u32(bmp.height);
    w.write_u32(bmp.pixels.len() as u32);
    // Bulk-append the pixel words. We reserve up front to avoid repeated
    // growth on multi-MB frames.
    w.bytes.reserve(bmp.pixels.len() * 4);
    for &px in &bmp.pixels {
        w.bytes.extend_from_slice(&px.to_le_bytes());
    }
}

fn decode_bitmap(r: &mut Reader<'_>) -> Result<cv_gfx::Bitmap, DecodeError> {
    let width = r.read_u32()?;
    let height = r.read_u32()?;
    let pixel_count = r.read_u32()? as usize;
    if !frame_bytes_ok(width, height, pixel_count) {
        return Err(DecodeError::OutOfRange);
    }
    let mut bmp = cv_gfx::Bitmap::new(width, height);
    // `Bitmap::new` allocates `width*height` pixels; the declared count was
    // already cross-checked equal to that by `frame_bytes_ok`.
    debug_assert_eq!(bmp.pixels.len(), pixel_count);
    for slot in bmp.pixels.iter_mut() {
        *slot = r.read_u32()?;
    }
    Ok(bmp)
}

/// The marshalable bundle the renderer commits per frame: the paint data
/// subset the browser composites + the tab summaries. Mirrors the in-process
/// `cv_ui::FromPage::Commit { paint, tabs }` (minus `epoch`, which rides on
/// the `Msg::CommitFrame` envelope so a stale frame is dropped without
/// decoding the payload).
#[derive(Debug)]
pub struct CommittedFrame {
    pub paint: cv_ui::PaintData,
    pub tabs: Vec<cv_ui::TabSummary>,
}

/// Serialize a committed frame into the `Msg::CommitFrame` opaque payload.
pub fn encode_frame(paint: &cv_ui::PaintData, tabs: &[cv_ui::TabSummary]) -> Vec<u8> {
    let mut w = Writer::with_capacity(
        // A generous pre-size: the bitmap dominates.
        paint.bitmap.pixels.len() * 4 + 256,
    );
    w.write_u32(FRAME_CODEC_VERSION);
    encode_bitmap(&mut w, &paint.bitmap);

    w.write_u32(paint.texts.len() as u32);
    for t in &paint.texts {
        encode_text_item(&mut w, t);
    }

    w.write_u32(paint.hit_regions.len() as u32);
    for hr in &paint.hit_regions {
        encode_hit_region(&mut w, hr);
    }

    w.write_str(&paint.title);
    w.write_str(&paint.current_url);
    w.write_u32(paint.chrome_h);
    w.write_u32(paint.viewport_h);

    match paint.caret_rect {
        Some((x, y, cw, ch)) => {
            w.write_bool(true);
            w.write_i32(x);
            w.write_i32(y);
            w.write_i32(cw);
            w.write_i32(ch);
        }
        None => w.write_bool(false),
    }
    // CSS UI 4 caret-color (the WM_PAINT overlay tints with this).
    match paint.caret_color {
        Some((cr, cg, cb)) => {
            w.write_bool(true);
            w.write_u8(cr);
            w.write_u8(cg);
            w.write_u8(cb);
        }
        None => w.write_bool(false),
    }
    // CSS Scrollbars 1 viewport theme (browser draws the scrollbar overlay).
    w.write_u8(paint.scrollbar_theme.width_mode);
    match paint.scrollbar_theme.colors {
        Some(((tr, tg, tb), (kr, kg, kb))) => {
            w.write_bool(true);
            w.write_u8(tr);
            w.write_u8(tg);
            w.write_u8(tb);
            w.write_u8(kr);
            w.write_u8(kg);
            w.write_u8(kb);
        }
        None => w.write_bool(false),
    }

    w.write_u32(tabs.len() as u32);
    for t in tabs {
        encode_tab_summary(&mut w, t);
    }

    w.into_bytes()
}

/// Decode a committed frame from the `Msg::CommitFrame` opaque payload.
/// Render-internal fields the browser does not consume (`layout_root`,
/// `property_trees`, `retained`) are reconstructed as `None` — the browser
/// composites the finished `bitmap` and hit-tests via `hit_regions`.
pub fn decode_frame(bytes: &[u8]) -> Result<CommittedFrame, DecodeError> {
    let mut r = Reader::new(bytes);
    let version = r.read_u32()?;
    if version != FRAME_CODEC_VERSION {
        return Err(DecodeError::OutOfRange);
    }
    let bitmap = decode_bitmap(&mut r)?;

    let text_count = r.read_u32()? as usize;
    let mut texts = Vec::with_capacity(text_count.min(4096));
    for _ in 0..text_count {
        texts.push(decode_text_item(&mut r)?);
    }

    let hit_count = r.read_u32()? as usize;
    let mut hit_regions = Vec::with_capacity(hit_count.min(65536));
    for _ in 0..hit_count {
        hit_regions.push(decode_hit_region(&mut r)?);
    }

    let title = r.read_str()?;
    let current_url = r.read_str()?;
    let chrome_h = r.read_u32()?;
    let viewport_h = r.read_u32()?;

    let caret_rect = if r.read_bool()? {
        Some((r.read_i32()?, r.read_i32()?, r.read_i32()?, r.read_i32()?))
    } else {
        None
    };
    let caret_color = if r.read_bool()? {
        Some((r.read_u8()?, r.read_u8()?, r.read_u8()?))
    } else {
        None
    };
    let sb_width_mode = r.read_u8()?;
    let sb_colors = if r.read_bool()? {
        Some((
            (r.read_u8()?, r.read_u8()?, r.read_u8()?),
            (r.read_u8()?, r.read_u8()?, r.read_u8()?),
        ))
    } else {
        None
    };
    let scrollbar_theme = cv_ui::ScrollbarTheme {
        width_mode: sb_width_mode,
        colors: sb_colors,
    };

    let tab_count = r.read_u32()? as usize;
    let mut tabs = Vec::with_capacity(tab_count.min(4096));
    for _ in 0..tab_count {
        tabs.push(decode_tab_summary(&mut r)?);
    }

    let paint = cv_ui::PaintData {
        bitmap: std::sync::Arc::new(bitmap),
        texts,
        layout_root: None,
        hit_regions,
        title,
        current_url,
        chrome_h,
        viewport_h,
        caret_rect,
        caret_color,
        scrollbar_theme,
        property_trees: None,
        retained: None,
        // Off-main IPC frames are always full-document bitmaps today (band
        // rastering is the in-process path); 0/0 = legacy full-doc.
        content_origin_y: 0,
        document_h: 0,
    };
    Ok(CommittedFrame { paint, tabs })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_bitmap() -> cv_gfx::Bitmap {
        // 3x2 with distinct pixels so a transposed/clipped decode is caught.
        let mut b = cv_gfx::Bitmap::new(3, 2);
        for (i, px) in b.pixels.iter_mut().enumerate() {
            *px = 0xFF00_0000 | (i as u32 * 0x0010_2030);
        }
        b
    }

    fn sample_paint() -> cv_ui::PaintData {
        cv_ui::PaintData {
            bitmap: std::sync::Arc::new(sample_bitmap()),
            texts: vec![
                cv_ui::TextItem {
                    x: 8,
                    y: -3,
                    w: 28,
                    h: 24,
                    font_size_px: 27,
                    bold: false,
                    font_weight: 0,
                    italic: true,
                    font_family: Some("Segoe UI".into()),
                    color_rgb: (10, 20, 30),
                    color_alpha: 255,
                    text: "\u{2039}".into(),
                    align: cv_ui::TextAlign::Center,
                    letter_spacing_px: 1,
                    is_chrome: true,
                    text_gradient: None,
                },
                cv_ui::TextItem {
                    x: 100,
                    y: 200,
                    w: 300,
                    h: 24,
                    font_size_px: 16,
                    bold: true,
                    font_weight: 900,
                    italic: false,
                    font_family: None,
                    color_rgb: (0, 0, 0),
                    color_alpha: 200,
                    text: "https://example.com/path".into(),
                    align: cv_ui::TextAlign::Left,
                    letter_spacing_px: 0,
                    is_chrome: false,
                    text_gradient: None,
                },
            ],
            layout_root: None,
            hit_regions: vec![
                cv_ui::HitRegion {
                    x: 1,
                    y: 2,
                    w: 3,
                    h: 4,
                    href: Some("/story".into()),
                    element_path: Some(vec![0, 3, 7]),
                },
                cv_ui::HitRegion {
                    x: 5,
                    y: 6,
                    w: 7,
                    h: 8,
                    href: None,
                    element_path: None,
                },
            ],
            title: "Example Domain".into(),
            current_url: "https://example.com/".into(),
            chrome_h: 64,
            viewport_h: 900,
            caret_rect: Some((12, 34, 2, 18)),
            caret_color: Some((220, 30, 90)),
            scrollbar_theme: cv_ui::ScrollbarTheme {
                width_mode: 1,
                colors: Some(((200, 100, 50), (20, 20, 20))),
            },
            property_trees: None,
            retained: None,
            content_origin_y: 0,
            document_h: 0,
        }
    }

    fn assert_paint_subset_eq(a: &cv_ui::PaintData, b: &cv_ui::PaintData) {
        assert_eq!(a.bitmap.width, b.bitmap.width);
        assert_eq!(a.bitmap.height, b.bitmap.height);
        assert_eq!(a.bitmap.pixels, b.bitmap.pixels, "pixels byte-identical");
        assert_eq!(a.texts, b.texts);
        assert_eq!(a.title, b.title);
        assert_eq!(a.current_url, b.current_url);
        assert_eq!(a.chrome_h, b.chrome_h);
        assert_eq!(a.viewport_h, b.viewport_h);
        assert_eq!(a.caret_rect, b.caret_rect);
        assert_eq!(a.caret_color, b.caret_color);
        assert_eq!(a.scrollbar_theme, b.scrollbar_theme);
        assert_eq!(a.hit_regions.len(), b.hit_regions.len());
        for (x, y) in a.hit_regions.iter().zip(&b.hit_regions) {
            assert_eq!((x.x, x.y, x.w, x.h), (y.x, y.y, y.w, y.h));
            assert_eq!(x.href, y.href);
            assert_eq!(x.element_path, y.element_path);
        }
    }

    #[test]
    fn frame_roundtrip_preserves_paint_subset_and_tabs() {
        let paint = sample_paint();
        let tabs = vec![
            cv_ui::TabSummary {
                id: 0,
                title: "Tab A".into(),
                url: "https://a.com/".into(),
                active: true,
            },
            cv_ui::TabSummary {
                id: 7,
                title: "Tab B".into(),
                url: "https://b.com/".into(),
                active: false,
            },
        ];
        let bytes = encode_frame(&paint, &tabs);
        let decoded = decode_frame(&bytes).expect("decode frame");
        assert_paint_subset_eq(&paint, &decoded.paint);
        assert_eq!(decoded.tabs.len(), 2);
        assert_eq!(decoded.tabs[0].id, 0);
        assert!(decoded.tabs[0].active);
        assert_eq!(decoded.tabs[1].url, "https://b.com/");
    }

    #[test]
    fn frame_decode_drops_render_internal_fields() {
        // The browser does not consume layout_root/property_trees/retained
        // off a cross-process frame; they must come back None.
        let bytes = encode_frame(&sample_paint(), &[]);
        let decoded = decode_frame(&bytes).expect("decode");
        assert!(decoded.paint.layout_root.is_none());
        assert!(decoded.paint.property_trees.is_none());
        assert!(decoded.paint.retained.is_none());
    }

    #[test]
    fn frame_decode_rejects_wrong_version() {
        let mut bytes = encode_frame(&sample_paint(), &[]);
        // Corrupt the leading version word.
        bytes[0] = 0xFF;
        assert!(matches!(decode_frame(&bytes), Err(DecodeError::OutOfRange)));
    }

    #[test]
    fn frame_decode_rejects_oversized_dimensions() {
        let mut w = Writer::new();
        w.write_u32(FRAME_CODEC_VERSION);
        w.write_u32(MAX_FRAME_DIMENSION + 1); // width
        w.write_u32(1); // height
        w.write_u32(1); // claimed pixel count
        w.write_u32(0xFF00_0000);
        assert!(matches!(
            decode_frame(&w.into_bytes()),
            Err(DecodeError::OutOfRange)
        ));
    }

    #[test]
    fn frame_decode_rejects_pixel_count_mismatch() {
        // width*height != declared pixel count must be refused (geometry
        // tripwire) before any allocation.
        let mut w = Writer::new();
        w.write_u32(FRAME_CODEC_VERSION);
        w.write_u32(2); // width
        w.write_u32(2); // height -> expects 4 pixels
        w.write_u32(9); // lies: 9 pixels
        assert!(matches!(
            decode_frame(&w.into_bytes()),
            Err(DecodeError::OutOfRange)
        ));
    }

    #[test]
    fn frame_decode_rejects_truncated_pixels() {
        let mut bytes = encode_frame(&sample_paint(), &[]);
        // Lop off the tail so the pixel array (and trailing metadata) is
        // incomplete; the decoder must report Truncated, never panic.
        bytes.truncate(bytes.len() - 4);
        assert!(matches!(
            decode_frame(&bytes),
            Err(DecodeError::Truncated | DecodeError::OutOfRange)
        ));
    }

    #[test]
    fn host_command_roundtrip_every_variant() {
        for cmd in [
            cv_ui::HostCommand::NewTab,
            cv_ui::HostCommand::CloseActiveTab,
            cv_ui::HostCommand::SwitchTab(42),
            cv_ui::HostCommand::NewWindow,
        ] {
            let bytes = encode_host_command(&cmd);
            let decoded = decode_host_command(&bytes).expect("decode host command");
            // HostCommand has no PartialEq; compare by debug discriminant +
            // payload.
            match (&cmd, &decoded) {
                (cv_ui::HostCommand::NewTab, cv_ui::HostCommand::NewTab)
                | (cv_ui::HostCommand::CloseActiveTab, cv_ui::HostCommand::CloseActiveTab)
                | (cv_ui::HostCommand::NewWindow, cv_ui::HostCommand::NewWindow) => {}
                (cv_ui::HostCommand::SwitchTab(a), cv_ui::HostCommand::SwitchTab(b)) => {
                    assert_eq!(a, b)
                }
                other => panic!("host command mismatch: {other:?}"),
            }
        }
    }

    #[test]
    fn host_command_decode_rejects_bad_tag() {
        assert!(matches!(
            decode_host_command(&[0xEE]),
            Err(DecodeError::OutOfRange)
        ));
    }
}
