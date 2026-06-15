//! PDF *writer* — emits a real, valid PDF 1.7 file from a paginated print
//! display list. This is the "Skia PDF backend" analog: the browser lays the
//! page out under the `print` media type, paginates it into [`crate::print_layout::PrintPage`]s,
//! and hands them here to serialise.
//!
//! The output is a genuine PDF: header, indirect objects, document catalog,
//! page tree, per-page content streams (real text via `BT`/`Tf`/`Td`/`Tj`,
//! filled rectangles via `re`/`f`, RGB images via `Do` on an `/Image` XObject),
//! a cross-reference table with correct byte offsets, and a trailer with
//! `/Root` + `/Size` + `startxref`. The text is real text operators (selectable
//! / searchable in a viewer), not a rasterised image of the page.
//!
//! Spec: ISO 32000-1 (PDF 1.7) §7 (file structure: header/body/xref/trailer),
//! §9.4 (text objects `BT`/`ET`, `Tf`, `Td`, `Tj`), §8.5.2.1 (path painting
//! `re`/`f`), §8.9.5 (image XObjects), §9.6.2.2 (standard 14 Type1 fonts —
//! Helvetica, no embedded font program required). Coordinate origin is the
//! bottom-left corner of the page (PDF default user space); our print layout
//! uses a top-left origin in points, so we flip Y when emitting.

use crate::print_layout::{PrintCmd, PrintPage};

/// One indirect object's serialised bytes, assembled then concatenated with
/// the running byte offset recorded for the xref table.
struct Obj {
    bytes: Vec<u8>,
}

/// RGB color, 0..=255 per channel — matches the layout `Color` channels.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Rgb {
    pub r: u8,
    pub g: u8,
    pub b: u8,
}

impl Rgb {
    pub const BLACK: Self = Self { r: 0, g: 0, b: 0 };
    /// Emit as a PDF `r g b` (each in 0..1) for `rg` / `RG`.
    fn pdf_components(self) -> String {
        format!(
            "{:.4} {:.4} {:.4}",
            f64::from(self.r) / 255.0,
            f64::from(self.g) / 255.0,
            f64::from(self.b) / 255.0
        )
    }
}

/// Build a complete PDF byte buffer from the paginated print pages.
///
/// Object layout (1-indexed object numbers):
///   1 = Catalog, 2 = Pages (the page tree root), 3 = the Helvetica font.
///   Then for each page: a Page dict, its Contents stream, and one Image
///   XObject per image command on that page.
pub fn write_pdf(pages: &[PrintPage]) -> Vec<u8> {
    let mut objs: Vec<Obj> = Vec::new();
    // Reserve object numbers 1..=3 for catalog / pages / font; fill later.
    objs.push(Obj { bytes: Vec::new() }); // 1 catalog (placeholder)
    objs.push(Obj { bytes: Vec::new() }); // 2 pages root (placeholder)
    objs.push(Obj { bytes: Vec::new() }); // 3 font (Helvetica)

    objs[2] = Obj {
        bytes: b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>"
            .to_vec(),
    };

    let mut page_obj_nums: Vec<u32> = Vec::new();

    for page in pages {
        // Image XObjects referenced by this page (emitted as their own objects).
        let mut img_resource_entries: Vec<String> = Vec::new();
        let mut img_objs: Vec<Obj> = Vec::new();

        // Build the content stream first so image XObject names line up.
        let content = build_content_stream(page, &mut img_resource_entries, &mut img_objs);

        // Allocate object numbers: page dict, content stream, then images.
        let page_num = (objs.len() + 1) as u32;
        let content_num = page_num + 1;
        // Reserve slots for page + content.
        objs.push(Obj { bytes: Vec::new() }); // page dict (filled below)
        objs.push(Obj { bytes: Vec::new() }); // content stream (filled below)

        // Append image objects and record their numbers for the resource dict.
        let mut img_name_to_num: Vec<(String, u32)> = Vec::new();
        for (i, img) in img_objs.into_iter().enumerate() {
            let num = (objs.len() + 1) as u32;
            objs.push(img);
            // img_resource_entries[i] is the XObject name we used in the stream.
            img_name_to_num.push((img_resource_entries[i].clone(), num));
        }

        // Content stream object (uncompressed — spec-valid; /Length exact).
        let stream = format!(
            "<< /Length {} >>\nstream\n{}\nendstream",
            content.len(),
            content
        );
        objs[(content_num - 1) as usize] = Obj {
            bytes: stream.into_bytes(),
        };

        // Resources dict: the font + any image XObjects.
        let mut xobject_dict = String::new();
        if !img_name_to_num.is_empty() {
            xobject_dict.push_str(" /XObject << ");
            for (name, num) in &img_name_to_num {
                xobject_dict.push_str(&format!("/{name} {num} 0 R "));
            }
            xobject_dict.push_str(">>");
        }
        let page_dict = format!(
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {:.2} {:.2}] \
             /Resources << /Font << /F1 3 0 R >>{xobject_dict} \
             /ProcSet [/PDF /Text /ImageC] >> /Contents {content_num} 0 R >>",
            page.width_pt, page.height_pt,
        );
        objs[(page_num - 1) as usize] = Obj {
            bytes: page_dict.into_bytes(),
        };
        page_obj_nums.push(page_num);
    }

    // Pages tree root (object 2).
    let kids = page_obj_nums
        .iter()
        .map(|n| format!("{n} 0 R"))
        .collect::<Vec<_>>()
        .join(" ");
    objs[1] = Obj {
        bytes: format!(
            "<< /Type /Pages /Kids [{kids}] /Count {} >>",
            page_obj_nums.len()
        )
        .into_bytes(),
    };

    // Catalog (object 1).
    objs[0] = Obj {
        bytes: b"<< /Type /Catalog /Pages 2 0 R >>".to_vec(),
    };

    serialize(&objs)
}

/// Build a single page's content stream (the PostScript-like operators).
/// Appends image XObject resource names + their objects via the out-params.
fn build_content_stream(
    page: &PrintPage,
    img_resource_entries: &mut Vec<String>,
    img_objs: &mut Vec<Obj>,
) -> String {
    let h = page.height_pt;
    let mut s = String::new();
    for cmd in &page.cmds {
        match cmd {
            PrintCmd::Rect {
                x,
                y,
                w,
                ht,
                color,
            } => {
                // Flip Y: PDF origin is bottom-left. A box at top-y..top-y+ht
                // occupies pdf-y (h - top_y - ht) .. (h - top_y).
                let py = h - y - ht;
                s.push_str(&format!(
                    "{} rg\n{:.2} {:.2} {:.2} {:.2} re\nf\n",
                    Rgb {
                        r: color.0,
                        g: color.1,
                        b: color.2
                    }
                    .pdf_components(),
                    x,
                    py,
                    w,
                    ht
                ));
            }
            PrintCmd::Text {
                x,
                baseline_y,
                size,
                color,
                text,
            } => {
                if text.is_empty() {
                    continue;
                }
                let py = h - baseline_y;
                s.push_str(&format!(
                    "BT\n{} rg\n/F1 {:.2} Tf\n{:.2} {:.2} Td\n({}) Tj\nET\n",
                    Rgb {
                        r: color.0,
                        g: color.1,
                        b: color.2
                    }
                    .pdf_components(),
                    size,
                    x,
                    py,
                    escape_pdf_string(text)
                ));
            }
            PrintCmd::Image {
                x,
                y,
                w,
                ht,
                width_px,
                height_px,
                rgb,
            } => {
                let name = format!("Im{}", img_resource_entries.len() + 1);
                // Image XObject: a DeviceRGB sample image, 8 bits/component.
                // Stream data is width*height*3 raw RGB bytes (uncompressed).
                let header = format!(
                    "<< /Type /XObject /Subtype /Image /Width {width_px} /Height {height_px} \
                     /ColorSpace /DeviceRGB /BitsPerComponent 8 /Length {} >>\nstream\n",
                    rgb.len()
                );
                let mut bytes = Vec::with_capacity(header.len() + rgb.len() + 10);
                bytes.extend_from_slice(header.as_bytes());
                bytes.extend_from_slice(rgb);
                bytes.extend_from_slice(b"\nendstream");
                img_objs.push(Obj { bytes });
                img_resource_entries.push(name.clone());

                // Draw it: CTM maps the unit image square to the box. Flip Y.
                let py = h - y - ht;
                s.push_str(&format!(
                    "q\n{:.2} 0 0 {:.2} {:.2} {:.2} cm\n/{} Do\nQ\n",
                    w, ht, x, py, name
                ));
            }
        }
    }
    s
}

/// Escape a string for a PDF literal `( … )` string: backslash-escape the
/// `\`, `(`, `)` delimiters and emit non-ASCII / control bytes as octal
/// (`\ddd`). ISO 32000-1 §7.3.4.2.
fn escape_pdf_string(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 8);
    for &b in s.as_bytes() {
        match b {
            b'\\' => out.push_str("\\\\"),
            b'(' => out.push_str("\\("),
            b')' => out.push_str("\\)"),
            b'\n' => out.push_str("\\n"),
            b'\r' => out.push_str("\\r"),
            b'\t' => out.push_str("\\t"),
            0x20..=0x7E => out.push(b as char),
            // Non-ASCII / control: octal escape (WinAnsi byte value).
            other => out.push_str(&format!("\\{other:03o}")),
        }
    }
    out
}

/// Concatenate the objects into a final PDF buffer with a correct xref table.
fn serialize(objs: &[Obj]) -> Vec<u8> {
    let mut out: Vec<u8> = Vec::new();
    // Header. The binary-comment line marks the file as containing binary data
    // (recommended by the spec so transport layers treat it as binary).
    out.extend_from_slice(b"%PDF-1.7\n");
    out.extend_from_slice(b"%\xE2\xE3\xCF\xD3\n");

    // Body: each object, recording its byte offset for the xref.
    let mut offsets: Vec<usize> = Vec::with_capacity(objs.len());
    for (i, obj) in objs.iter().enumerate() {
        offsets.push(out.len());
        let num = i + 1;
        out.extend_from_slice(format!("{num} 0 obj\n").as_bytes());
        out.extend_from_slice(&obj.bytes);
        out.extend_from_slice(b"\nendobj\n");
    }

    // Cross-reference table.
    let xref_offset = out.len();
    let count = objs.len() + 1; // +1 for the free object 0
    out.extend_from_slice(format!("xref\n0 {count}\n").as_bytes());
    // Object 0 is the head of the free list: offset 0, gen 65535, free.
    out.extend_from_slice(b"0000000000 65535 f \n");
    for off in &offsets {
        // Each entry is EXACTLY 20 bytes: "nnnnnnnnnn ggggg n \n".
        out.extend_from_slice(format!("{off:010} 00000 n \n").as_bytes());
    }

    // Trailer.
    out.extend_from_slice(
        format!(
            "trailer\n<< /Size {count} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n"
        )
        .as_bytes(),
    );
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::print_layout::{PrintCmd, PrintPage};

    fn text_page(strings: &[&str]) -> PrintPage {
        let cmds = strings
            .iter()
            .enumerate()
            .map(|(i, t)| PrintCmd::Text {
                x: 72.0,
                baseline_y: 72.0 + i as f32 * 20.0,
                size: 12.0,
                color: (0, 0, 0),
                text: (*t).to_string(),
            })
            .collect();
        PrintPage {
            width_pt: 612.0,
            height_pt: 792.0,
            cmds,
        }
    }

    #[test]
    fn emits_valid_header_and_eof() {
        let pdf = write_pdf(&[text_page(&["Hello"])]);
        assert!(pdf.starts_with(b"%PDF-1.7"), "PDF header present");
        assert!(pdf.ends_with(b"%%EOF\n"), "EOF marker present");
        assert!(
            pdf.windows(9).any(|w| w == b"startxref"),
            "startxref present"
        );
    }

    #[test]
    fn page_count_matches_input() {
        let pages = vec![text_page(&["A"]), text_page(&["B"]), text_page(&["C"])];
        let pdf = write_pdf(&pages);
        // The Pages root must declare /Count 3.
        let s = String::from_utf8_lossy(&pdf);
        assert!(s.contains("/Count 3"), "Pages /Count reflects 3 pages");
        // Three /Type /Page dicts.
        let page_dicts = s.matches("/Type /Page ").count();
        assert_eq!(page_dicts, 3, "three page objects emitted");
    }

    #[test]
    fn text_is_real_selectable_operators() {
        let pdf = write_pdf(&[text_page(&["Conclave"])]);
        let s = String::from_utf8_lossy(&pdf);
        // Real text operators, not a raster.
        assert!(s.contains("BT\n"), "text object begun");
        assert!(s.contains("/F1 12.00 Tf"), "font set");
        assert!(s.contains("(Conclave) Tj"), "show-text operator with content");
        assert!(s.contains("/BaseFont /Helvetica"), "standard font declared");
    }

    #[test]
    fn round_trips_through_the_pdf_reader() {
        // The writer's output must parse with our OWN reader (xref + page tree).
        let pdf = write_pdf(&[text_page(&["X"]), text_page(&["Y"])]);
        let sx = crate::find_startxref(&pdf).expect("startxref found");
        let xref = crate::parse_xref(&pdf, sx).expect("xref parses");
        // Object 1 (catalog) and 2 (pages) must be present & in-use.
        assert!(xref.lookup(1).unwrap().in_use);
        assert!(xref.lookup(2).unwrap().in_use);
    }

    #[test]
    fn rect_emits_fill_operators() {
        let page = PrintPage {
            width_pt: 612.0,
            height_pt: 792.0,
            cmds: vec![PrintCmd::Rect {
                x: 10.0,
                y: 10.0,
                w: 100.0,
                ht: 50.0,
                color: (255, 0, 0),
            }],
        };
        let pdf = write_pdf(&[page]);
        let s = String::from_utf8_lossy(&pdf);
        assert!(s.contains(" re\n"), "rectangle path op");
        assert!(s.contains("f\n"), "fill op");
        assert!(s.contains("1.0000 0.0000 0.0000 rg"), "red fill color");
    }

    #[test]
    fn image_emits_xobject() {
        let page = PrintPage {
            width_pt: 612.0,
            height_pt: 792.0,
            cmds: vec![PrintCmd::Image {
                x: 0.0,
                y: 0.0,
                w: 10.0,
                ht: 10.0,
                width_px: 2,
                height_px: 1,
                rgb: vec![255, 0, 0, 0, 255, 0], // 2 px: red, green
            }],
        };
        let pdf = write_pdf(&[page]);
        let s = String::from_utf8_lossy(&pdf);
        assert!(s.contains("/Subtype /Image"), "image XObject emitted");
        assert!(s.contains("/Im1 Do"), "image drawn via Do");
        assert!(s.contains("cm\n"), "CTM placement emitted");
    }

    #[test]
    fn special_chars_in_text_escaped() {
        let page = PrintPage {
            width_pt: 612.0,
            height_pt: 792.0,
            cmds: vec![PrintCmd::Text {
                x: 0.0,
                baseline_y: 100.0,
                size: 12.0,
                color: (0, 0, 0),
                text: "a(b)c\\d".to_string(),
            }],
        };
        let pdf = write_pdf(&[page]);
        let s = String::from_utf8_lossy(&pdf);
        assert!(
            s.contains("(a\\(b\\)c\\\\d) Tj"),
            "parens and backslash escaped: {s}"
        );
    }
}
