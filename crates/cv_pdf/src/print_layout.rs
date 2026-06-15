//! Print layout + pagination — turns a laid-out box tree (already laid out
//! under the `print` media type, with `@page` size/margins resolved) into a
//! sequence of [`PrintPage`]s, each a display list of [`PrintCmd`]s ready for
//! the [`crate::writer`] to serialise to PDF.
//!
//! This crate stays decoupled from `cv_layout`: the host (`cv_browser`) walks
//! its live `LayoutBox` tree and maps each box into a neutral [`PrintBox`].
//! That mirrors how Chrome's print path re-uses the layout tree but renders it
//! through the PDF (Skia) backend rather than the screen compositor.
//!
//! Pagination model (CSS Fragmentation 3 / CSS Paged Media 3):
//!   * The content flows down a single tall column whose width is the page's
//!     printable width (page width minus the `@page` left/right margins).
//!   * A page holds `page_content_height` points of content (page height minus
//!     top/bottom margins).
//!   * A box whose top would fall on a later page-row starts that later page.
//!   * `break-before: page`/`always` forces the box to start a fresh page;
//!     `break-after: page` forces the FOLLOWING content onto a fresh page.
//!   * `break-inside: avoid` keeps a box that fits within one page-height from
//!     being split across a page boundary — it is pushed wholesale to the next
//!     page when it would straddle the break (CSS Fragmentation 3 §4.2).
//!   * Content taller than a page still splits (a box can't avoid a break it
//!     can't fit inside).

/// A primitive paint command on a page, in **points**, top-left origin.
/// The writer flips Y into PDF's bottom-left user space.
#[derive(Clone, Debug, PartialEq)]
pub enum PrintCmd {
    /// Filled rectangle (background / border fill). `(r,g,b)` 0..=255.
    Rect {
        x: f32,
        y: f32,
        w: f32,
        ht: f32,
        color: (u8, u8, u8),
    },
    /// A run of text on its baseline. `baseline_y` is the baseline's distance
    /// from the page top (points). `size` is the font size in points.
    Text {
        x: f32,
        baseline_y: f32,
        size: f32,
        color: (u8, u8, u8),
        text: String,
    },
    /// An RGB raster image scaled into the box `(x,y,w,ht)` (points).
    /// `rgb` is `width_px*height_px*3` raw bytes, row-major, top-to-bottom.
    Image {
        x: f32,
        y: f32,
        w: f32,
        ht: f32,
        width_px: u32,
        height_px: u32,
        rgb: Vec<u8>,
    },
}

/// One emitted page: its media box (points) and its display list.
#[derive(Clone, Debug)]
pub struct PrintPage {
    pub width_pt: f32,
    pub height_pt: f32,
    pub cmds: Vec<PrintCmd>,
}

/// CSS fragmentation break value for a box edge (`break-before`/`break-after`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum BreakValue {
    /// `auto` — a break is allowed but not forced (the default).
    #[default]
    Auto,
    /// `page` / `always` / `left` / `right` — force a page break here.
    Force,
    /// `avoid` — discourage a break here (we honour this for `break-inside`).
    Avoid,
}

/// What a box paints. Mirrors the subset of `cv_layout::BoxKind` the PDF
/// backend can render directly.
#[derive(Clone, Debug)]
pub enum PaintKind {
    /// A block/anonymous container — paints only its background (if any).
    Container,
    /// A text run — `text` is the already-shaped string; `size` its font
    /// size (points); `baseline_offset` is the baseline's distance below the
    /// box's content top (points).
    Text {
        text: String,
        size: f32,
        baseline_offset: f32,
    },
    /// An image — raw RGB bytes (`width_px*height_px*3`, top-to-bottom).
    Image {
        width_px: u32,
        height_px: u32,
        rgb: Vec<u8>,
    },
}

/// A neutral, post-layout box the host maps `cv_layout::LayoutBox` into.
/// Coordinates are in **points** (CSS px == pt for print; Chrome maps 96 CSS px
/// to 72 pt, but our layout already runs at the print page width so we treat
/// 1 CSS px = 1 pt for a faithful 1:1 paged layout). `x`/`y`/`w`/`h` are the
/// box's *content+padding+border* rect in the document's flow coordinate space
/// (y grows downward from the document top).
#[derive(Clone, Debug)]
pub struct PrintBox {
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
    /// Solid background fill, if any (`(r,g,b)`).
    pub background: Option<(u8, u8, u8)>,
    pub color: (u8, u8, u8),
    pub kind: PaintKind,
    pub break_before: BreakValue,
    pub break_after: BreakValue,
    pub break_inside: BreakValue,
    pub children: Vec<PrintBox>,
}

impl PrintBox {
    /// A bare container at the given rect (test/host convenience).
    pub fn container(x: f32, y: f32, w: f32, h: f32) -> Self {
        Self {
            x,
            y,
            w,
            h,
            background: None,
            color: (0, 0, 0),
            kind: PaintKind::Container,
            break_before: BreakValue::Auto,
            break_after: BreakValue::Auto,
            break_inside: BreakValue::Auto,
            children: Vec::new(),
        }
    }
}

/// Resolved `@page` geometry (CSS Paged Media 3 §6). All in points.
#[derive(Clone, Copy, Debug)]
pub struct PageGeometry {
    pub width_pt: f32,
    pub height_pt: f32,
    pub margin_top: f32,
    pub margin_right: f32,
    pub margin_bottom: f32,
    pub margin_left: f32,
}

impl PageGeometry {
    /// US Letter, default 0.5in (36pt) margins (Chrome's print default).
    pub const LETTER: Self = Self {
        width_pt: 612.0,
        height_pt: 792.0,
        margin_top: 36.0,
        margin_right: 36.0,
        margin_bottom: 36.0,
        margin_left: 36.0,
    };
    /// A4 with the same default margins.
    pub const A4: Self = Self {
        width_pt: 595.0,
        height_pt: 842.0,
        margin_top: 36.0,
        margin_right: 36.0,
        margin_bottom: 36.0,
        margin_left: 36.0,
    };

    /// Printable content height between the top and bottom margins.
    pub fn content_height(&self) -> f32 {
        (self.height_pt - self.margin_top - self.margin_bottom).max(1.0)
    }
}

/// A flattened paint atom positioned in the document flow (points, top-left),
/// tagged with the fragmentation hints that decide which page it lands on.
struct Atom {
    /// Top edge of the atom in document-flow coordinates.
    top: f32,
    /// Bottom edge (top + height).
    bottom: f32,
    cmd_at_origin: AtomCmd,
    /// `break-inside: avoid` (and the atom fits a page) — don't split across a
    /// page boundary; push wholesale to the next page instead. (Forced
    /// `break-before`/`-after` are collected separately into the page-break
    /// list by [`collect_forced_breaks`].)
    avoid_break_inside: bool,
}

/// An atom's paint command with its y expressed RELATIVE to the atom's `top`,
/// so it can be re-based onto whichever page it lands on.
enum AtomCmd {
    Rect {
        x: f32,
        w: f32,
        ht: f32,
        color: (u8, u8, u8),
    },
    Text {
        x: f32,
        /// Baseline offset below the atom top.
        baseline_off: f32,
        size: f32,
        color: (u8, u8, u8),
        text: String,
    },
    Image {
        x: f32,
        w: f32,
        ht: f32,
        width_px: u32,
        height_px: u32,
        rgb: Vec<u8>,
    },
}

/// Flatten the box tree into positioned paint atoms (document-flow order,
/// which is the order the boxes were laid out — preorder gives us the
/// background-then-content z-order the painter needs).
fn flatten(b: &PrintBox, out: &mut Vec<Atom>) {
    // 1) Background fill, if any (painted first, behind content).
    if let Some(bg) = b.background {
        if b.w > 0.0 && b.h > 0.0 {
            out.push(Atom {
                top: b.y,
                bottom: b.y + b.h,
                cmd_at_origin: AtomCmd::Rect {
                    x: b.x,
                    w: b.w,
                    ht: b.h,
                    color: bg,
                },
                avoid_break_inside: b.break_inside == BreakValue::Avoid,
            });
        }
    }

    // 2) The box's own content (text / image).
    match &b.kind {
        PaintKind::Container => {}
        PaintKind::Text {
            text,
            size,
            baseline_offset,
        } => {
            if !text.is_empty() {
                let h = (*baseline_offset).max(*size);
                out.push(Atom {
                    top: b.y,
                    bottom: b.y + h,
                    cmd_at_origin: AtomCmd::Text {
                        x: b.x,
                        baseline_off: *baseline_offset,
                        size: *size,
                        color: b.color,
                        text: text.clone(),
                    },
                    avoid_break_inside: b.break_inside == BreakValue::Avoid,
                });
            }
        }
        PaintKind::Image {
            width_px,
            height_px,
            rgb,
        } => {
            out.push(Atom {
                top: b.y,
                bottom: b.y + b.h,
                cmd_at_origin: AtomCmd::Image {
                    x: b.x,
                    w: b.w,
                    ht: b.h,
                    width_px: *width_px,
                    height_px: *height_px,
                    rgb: rgb.clone(),
                },
                avoid_break_inside: b.break_inside == BreakValue::Avoid,
            });
        }
    }

    // 3) Children (z-order: after this box's own background/content).
    for c in &b.children {
        flatten(c, out);
    }
}

/// Collect every forced-break y (the top of any box with `break-before: page`
/// and the bottom of any box with `break-after: page`), in document-flow
/// coordinates.
fn collect_forced_breaks(b: &PrintBox, out: &mut Vec<f32>) {
    if b.break_before == BreakValue::Force {
        out.push(b.y);
    }
    if b.break_after == BreakValue::Force {
        out.push(b.y + b.h);
    }
    for c in &b.children {
        collect_forced_breaks(c, out);
    }
}

/// Paginate the laid-out print tree into pages under `geom`.
///
/// `doc_top`/`doc_height` are the flow extent of the content (usually 0 and the
/// root box height). Returns at least one page.
pub fn paginate(root: &PrintBox, geom: PageGeometry) -> Vec<PrintPage> {
    let mut atoms: Vec<Atom> = Vec::new();
    flatten(root, &mut atoms);
    // Stable sort by top edge so page assignment is deterministic and z-order
    // within a page follows flow order (already preorder; ties keep order).
    atoms.sort_by(|a, b| a.top.partial_cmp(&b.top).unwrap_or(std::cmp::Ordering::Equal));

    let mut forced: Vec<f32> = Vec::new();
    collect_forced_breaks(root, &mut forced);
    forced.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    forced.dedup_by(|a, b| (*a - *b).abs() < 0.01);

    let page_h = geom.content_height();

    // The y (document-flow) of the current page's top. We grow pages until all
    // atoms are placed. For each page we capture atoms whose top falls in
    // [page_top, page_top + page_h), honouring forced + avoid breaks.
    let mut pages: Vec<PrintPage> = Vec::new();
    let mut page_top = 0.0f32;
    // Guard against pathological infinite loops (e.g. zero-height content).
    let mut guard = 0usize;
    let total_bottom = atoms
        .iter()
        .map(|a| a.bottom)
        .fold(0.0f32, f32::max)
        .max(forced.iter().copied().fold(0.0, f32::max));

    loop {
        guard += 1;
        if guard > 100_000 {
            break;
        }
        let mut page_bottom = page_top + page_h;

        // A forced break strictly inside this page shortens it to that break.
        if let Some(&fb) = forced
            .iter()
            .find(|&&f| f > page_top + 0.01 && f < page_bottom - 0.01)
        {
            page_bottom = fb;
        }

        // An `avoid-break-inside` atom that STRADDLES page_bottom but FITS a
        // full page is pushed wholly to the next page: shorten this page to the
        // atom's top so it starts the next page intact.
        let mut shortened = page_bottom;
        for a in &atoms {
            if !a.avoid_break_inside {
                continue;
            }
            let height = a.bottom - a.top;
            let straddles = a.top > page_top + 0.01 && a.top < page_bottom - 0.01 && a.bottom > page_bottom + 0.01;
            if straddles && height <= page_h {
                shortened = shortened.min(a.top);
            }
        }
        page_bottom = shortened.max(page_top + 1.0).min(page_top + page_h);
        // Re-apply the forced break in case `avoid` pushed past it.
        if let Some(&fb) = forced
            .iter()
            .find(|&&f| f > page_top + 0.01 && f < page_bottom - 0.01)
        {
            page_bottom = fb;
        }

        // Emit atoms whose top is within [page_top, page_bottom).
        let mut cmds: Vec<PrintCmd> = Vec::new();
        for a in &atoms {
            if a.top >= page_top - 0.01 && a.top < page_bottom - 0.01 {
                // Re-base into page-local coordinates: doc y → page y, then add
                // the top margin so content sits inside the printable area.
                let local_top = a.top - page_top + geom.margin_top;
                let local_left_origin = geom.margin_left; // page x offset
                match &a.cmd_at_origin {
                    AtomCmd::Rect { x, w, ht, color } => {
                        cmds.push(PrintCmd::Rect {
                            x: local_left_origin + *x,
                            y: local_top,
                            w: *w,
                            ht: *ht,
                            color: *color,
                        });
                    }
                    AtomCmd::Text {
                        x,
                        baseline_off,
                        size,
                        color,
                        text,
                    } => {
                        cmds.push(PrintCmd::Text {
                            x: local_left_origin + *x,
                            baseline_y: local_top + *baseline_off,
                            size: *size,
                            color: *color,
                            text: text.clone(),
                        });
                    }
                    AtomCmd::Image {
                        x,
                        w,
                        ht,
                        width_px,
                        height_px,
                        rgb,
                    } => {
                        cmds.push(PrintCmd::Image {
                            x: local_left_origin + *x,
                            y: local_top,
                            w: *w,
                            ht: *ht,
                            width_px: *width_px,
                            height_px: *height_px,
                            rgb: rgb.clone(),
                        });
                    }
                }
            }
        }

        pages.push(PrintPage {
            width_pt: geom.width_pt,
            height_pt: geom.height_pt,
            cmds,
        });

        page_top = page_bottom;
        if page_top >= total_bottom - 0.01 {
            break;
        }
    }

    if pages.is_empty() {
        // Always emit at least one (blank) page.
        pages.push(PrintPage {
            width_pt: geom.width_pt,
            height_pt: geom.height_pt,
            cmds: Vec::new(),
        });
    }
    pages
}

#[cfg(test)]
mod tests {
    use super::*;

    fn text_box(y: f32, h: f32, s: &str) -> PrintBox {
        let mut b = PrintBox::container(0.0, y, 400.0, h);
        b.kind = PaintKind::Text {
            text: s.to_string(),
            size: 12.0,
            baseline_offset: 10.0,
        };
        b
    }

    /// A document taller than one page splits into N pages, with content
    /// distributed across the boundary.
    #[test]
    fn tall_document_paginates_into_multiple_pages() {
        // content_height for Letter w/ 36pt margins = 792-72 = 720pt.
        let geom = PageGeometry::LETTER;
        let mut root = PrintBox::container(0.0, 0.0, 400.0, 2000.0);
        // Three text rows: at y=0, y=800 (page 2), y=1600 (page 3).
        root.children.push(text_box(0.0, 20.0, "First page line"));
        root.children.push(text_box(800.0, 20.0, "Second page line"));
        root.children.push(text_box(1600.0, 20.0, "Third page line"));

        let pages = paginate(&root, geom);
        assert!(pages.len() >= 3, "expected >=3 pages, got {}", pages.len());

        // The first line is on page 1, not on page 2.
        let has = |p: &PrintPage, needle: &str| {
            p.cmds.iter().any(|c| matches!(c, PrintCmd::Text { text, .. } if text == needle))
        };
        assert!(has(&pages[0], "First page line"), "line 1 on page 1");
        assert!(!has(&pages[0], "Second page line"), "line 2 NOT on page 1");
        // Find the page holding line 2; it must be a later page.
        let p2 = pages.iter().position(|p| has(p, "Second page line")).unwrap();
        assert!(p2 >= 1, "line 2 on a later page");
        let p3 = pages.iter().position(|p| has(p, "Third page line")).unwrap();
        assert!(p3 > p2, "line 3 after line 2's page");
    }

    /// `break-before: page` forces a new page even when content would have fit.
    #[test]
    fn forced_break_before_starts_new_page() {
        let geom = PageGeometry::LETTER;
        let mut root = PrintBox::container(0.0, 0.0, 400.0, 200.0);
        root.children.push(text_box(0.0, 20.0, "Before break"));
        let mut after = text_box(40.0, 20.0, "After break");
        after.break_before = BreakValue::Force;
        root.children.push(after);

        let pages = paginate(&root, geom);
        assert!(pages.len() >= 2, "forced break creates >=2 pages");
        let has = |p: &PrintPage, needle: &str| {
            p.cmds.iter().any(|c| matches!(c, PrintCmd::Text { text, .. } if text == needle))
        };
        assert!(has(&pages[0], "Before break"));
        assert!(!has(&pages[0], "After break"), "forced break pushed to page 2");
        assert!(has(&pages[1], "After break"));
    }

    /// `break-inside: avoid` pushes a straddling box wholesale to the next page.
    #[test]
    fn avoid_break_inside_pushes_block_to_next_page() {
        let geom = PageGeometry::LETTER; // content height 720
        let mut root = PrintBox::container(0.0, 0.0, 400.0, 1000.0);
        // A 100pt-tall block whose top is at 700 would straddle the 720 break.
        let mut block = PrintBox::container(0.0, 700.0, 400.0, 100.0);
        block.background = Some((200, 200, 200));
        block.break_inside = BreakValue::Avoid;
        // Mark it via a child text so we can find which page it lands on.
        block.children.push(text_box(700.0, 100.0, "Keep together"));
        root.children.push(block);

        let pages = paginate(&root, geom);
        let has = |p: &PrintPage, needle: &str| {
            p.cmds.iter().any(|c| matches!(c, PrintCmd::Text { text, .. } if text == needle))
        };
        // The block must NOT be on page 1 (it straddled), it's pushed to page 2.
        assert!(!has(&pages[0], "Keep together"), "avoided box not split onto page 1");
        assert!(pages.len() >= 2);
        assert!(has(&pages[1], "Keep together"), "avoided box intact on page 2");
    }

    /// Margins reduce printable area: a box at doc-y 0 paints at page-y =
    /// top margin (36pt), and x is offset by the left margin.
    #[test]
    fn page_margins_offset_content() {
        let geom = PageGeometry::LETTER;
        let mut root = PrintBox::container(0.0, 0.0, 400.0, 50.0);
        root.children.push(text_box(0.0, 20.0, "M"));
        let pages = paginate(&root, geom);
        let cmd = pages[0]
            .cmds
            .iter()
            .find(|c| matches!(c, PrintCmd::Text { .. }))
            .unwrap();
        if let PrintCmd::Text { x, baseline_y, .. } = cmd {
            assert!((*x - 36.0).abs() < 0.01, "x offset by left margin: {x}");
            // baseline = top margin (36) + local_top(0) + baseline_off(10) = 46.
            assert!((*baseline_y - 46.0).abs() < 0.01, "baseline w/ margin: {baseline_y}");
        }
    }

    /// Single short document yields exactly one page.
    #[test]
    fn short_document_is_one_page() {
        let geom = PageGeometry::LETTER;
        let mut root = PrintBox::container(0.0, 0.0, 400.0, 50.0);
        root.children.push(text_box(0.0, 20.0, "Only line"));
        let pages = paginate(&root, geom);
        assert_eq!(pages.len(), 1, "short doc = 1 page");
    }

    /// End-to-end: paginate then write a PDF, assert page count and text.
    #[test]
    fn paginate_then_write_pdf_roundtrip() {
        let geom = PageGeometry::LETTER;
        let mut root = PrintBox::container(0.0, 0.0, 400.0, 2000.0);
        root.children.push(text_box(0.0, 20.0, "Alpha"));
        root.children.push(text_box(900.0, 20.0, "Bravo"));
        root.children.push(text_box(1800.0, 20.0, "Charlie"));
        let pages = paginate(&root, geom);
        let n = pages.len();
        let pdf = crate::writer::write_pdf(&pages);
        let s = String::from_utf8_lossy(&pdf);
        assert!(s.contains(&format!("/Count {n}")), "PDF page count matches");
        assert!(s.contains("(Alpha) Tj"), "PDF contains text Alpha");
        assert!(s.contains("(Charlie) Tj"), "PDF contains text Charlie");
    }
}
