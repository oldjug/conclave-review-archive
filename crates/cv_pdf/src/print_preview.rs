//! Print preview state + page-break engine.
//!
//! Holds the paginated output ready for the existing PDF writer to
//! emit. The "preview" is a list of PreviewPage records, each
//! carrying the rasterized canvas and a label. @page CSS at-rules
//! land here as PageStyle records the engine consults during
//! pagination.

#[derive(Debug, Clone)]
pub struct PageSize {
    pub width_pt: f32,
    pub height_pt: f32,
}

impl PageSize {
    pub const LETTER: Self = Self {
        width_pt: 612.0,
        height_pt: 792.0,
    };
    pub const A4: Self = Self {
        width_pt: 595.0,
        height_pt: 842.0,
    };
}

#[derive(Debug, Clone, Default)]
pub struct PageStyle {
    pub margin_top_pt: f32,
    pub margin_right_pt: f32,
    pub margin_bottom_pt: f32,
    pub margin_left_pt: f32,
    pub orientation: Orientation,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Orientation {
    #[default]
    Portrait,
    Landscape,
}

#[derive(Debug, Clone)]
pub struct PreviewPage {
    pub width_px: u32,
    pub height_px: u32,
    pub bgra: Vec<u32>,
    pub label: String,
}

#[derive(Debug, Clone)]
pub struct PrintJob {
    pub pages: Vec<PreviewPage>,
    pub page_size: PageSize,
    pub style: PageStyle,
}

impl PrintJob {
    /// Paginate a tall canvas into N preview pages of `page_size_px`
    /// height each. CSS `break-before`/`break-after` opportunities
    /// (the `forced_breaks_at` set, indexed in pixel rows) cause an
    /// early page boundary.
    pub fn paginate(
        canvas: &[u32],
        canvas_w: u32,
        canvas_h: u32,
        page_h_px: u32,
        forced_breaks_at: &[u32],
        style: PageStyle,
    ) -> Self {
        let mut pages = Vec::new();
        let mut start = 0u32;
        let mut i = 1u32;
        while start < canvas_h {
            let mut end = (start + page_h_px).min(canvas_h);
            if let Some(&forced) = forced_breaks_at.iter().find(|f| **f > start && **f <= end) {
                end = forced;
            }
            let mut bgra = Vec::with_capacity((canvas_w * (end - start)) as usize);
            for y in start..end {
                let row_start = (y * canvas_w) as usize;
                let row_end = row_start + canvas_w as usize;
                bgra.extend_from_slice(&canvas[row_start..row_end]);
            }
            pages.push(PreviewPage {
                width_px: canvas_w,
                height_px: end - start,
                bgra,
                label: format!("Page {i}"),
            });
            start = end;
            i += 1;
        }
        Self {
            pages,
            page_size: PageSize::LETTER,
            style,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn paginate_three_full_pages() {
        let canvas = vec![0u32; 100 * 300];
        let job = PrintJob::paginate(&canvas, 100, 300, 100, &[], PageStyle::default());
        assert_eq!(job.pages.len(), 3);
    }

    #[test]
    fn forced_break_creates_extra_page() {
        let canvas = vec![0u32; 100 * 200];
        let job = PrintJob::paginate(&canvas, 100, 200, 100, &[50], PageStyle::default());
        assert!(job.pages.len() >= 3);
        assert_eq!(job.pages[0].height_px, 50);
    }
}
