//! `cv_reader` — Readability-style article distiller.
//!
//! Given an HTML document, score each block-level element by
//! Mozilla Readability's heuristic (link density, paragraph count,
//! `<article>` boost, `byline`/`comment`/`sidebar` penalty) and emit a
//! reader-mode HTML that's just the main article body + title + byline.
//!
//! This is the same algorithm Chrome's "Distiller" / Firefox's
//! "Reader View" ship — we port the published Mozilla Readability.js
//! scoring model. No third-party crates.
//!
//! Input: a parsed HTML doc as plain text + minimal tag-block stream
//! (we don't import cv_html to keep this crate cycle-free; embedder
//! converts a real DOM into our intermediate form).
//!
//! Output: `Article { title, byline, content_html, lang, text_word_count }`.

#![allow(clippy::too_many_lines)]

/// Block-level node our scorer reasons about. Inline runs are flattened
/// into the parent block's `text`.
#[derive(Debug, Clone)]
pub struct Block {
    pub tag: String,
    pub id: String,
    pub class: String,
    pub text: String,
    /// Inline HTML children we preserve verbatim in the output (links,
    /// emphasis). The scorer only looks at `text`.
    pub inner_html: String,
    /// Direct child <p> count — a key positive signal in Readability.
    pub child_p_count: u32,
    /// Direct child <a> link character count — used for link density.
    pub link_text_chars: u32,
}

#[derive(Debug, Clone, Default)]
pub struct Article {
    pub title: String,
    pub byline: String,
    pub content_html: String,
    pub lang: String,
    pub text_word_count: u32,
}

/// Score a single block per Readability rules.
pub fn score_block(b: &Block) -> f32 {
    let mut score: f32 = 0.0;

    // Base by tag.
    match b.tag.as_str() {
        "div" => score += 5.0,
        "pre" | "td" | "blockquote" => score += 3.0,
        "article" | "section" | "main" => score += 8.0,
        "address" | "ol" | "ul" | "dl" | "dd" | "dt" | "li" | "form" => score -= 3.0,
        "h2" | "h3" | "h4" | "h5" | "h6" | "th" => score -= 5.0,
        _ => {}
    }

    // ID/class positive bumps.
    let idc = format!("{} {}", b.id, b.class).to_lowercase();
    let positive_re = [
        "article",
        "body",
        "content",
        "entry",
        "hentry",
        "main",
        "page",
        "pagination",
        "post",
        "text",
        "blog",
        "story",
    ];
    let negative_re = [
        "hidden",
        "banner",
        "combx",
        "comment",
        "com-",
        "contact",
        "foot",
        "footer",
        "footnote",
        "masthead",
        "media",
        "meta",
        "outbrain",
        "promo",
        "related",
        "scroll",
        "share",
        "shoutbox",
        "sidebar",
        "skyscraper",
        "sponsor",
        "shopping",
        "tags",
        "tool",
        "widget",
    ];
    for p in &positive_re {
        if idc.contains(p) {
            score += 25.0;
        }
    }
    for n in &negative_re {
        if idc.contains(n) {
            score -= 25.0;
        }
    }

    // Body text bonus: 1 point per 100 chars, capped at 3.
    let txt_len = b.text.chars().count();
    let len_bonus = (txt_len as f32 / 100.0).min(3.0);
    score += len_bonus;

    // Paragraph bonus.
    score += (b.child_p_count as f32).min(3.0);

    // Comma bonus — proxy for prose.
    let commas = b.text.matches(',').count() as f32;
    score += commas.min(10.0) * 0.5;

    // Link density penalty: linkText / textLen.
    if txt_len > 0 {
        let density = (b.link_text_chars as f32) / (txt_len as f32);
        score *= 1.0 - density.min(1.0);
    }

    score
}

/// Pick the top-scoring block out of a flat list. Returns its index +
/// final score.
pub fn pick_top(blocks: &[Block]) -> Option<(usize, f32)> {
    let mut best: Option<(usize, f32)> = None;
    for (i, b) in blocks.iter().enumerate() {
        let s = score_block(b);
        match best {
            None => best = Some((i, s)),
            Some((_, prev)) if s > prev => best = Some((i, s)),
            _ => {}
        }
    }
    best
}

/// Heuristically extract the article title from a list of candidate
/// strings (typically `<title>`, `<h1>`, OG meta tag).
pub fn pick_title(candidates: &[&str]) -> String {
    let mut best = "";
    let mut best_score = -1_i32;
    for c in candidates {
        let s = title_quality(c);
        if s > best_score {
            best_score = s;
            best = c;
        }
    }
    best.trim().to_string()
}

fn title_quality(s: &str) -> i32 {
    let t = s.trim();
    if t.is_empty() {
        return -1;
    }
    let mut q = 100;
    // Penalise very short titles (likely "Home") and excess separator
    // suffixes ("Site Name - News - Article").
    let len = t.chars().count() as i32;
    if len < 10 {
        q -= 50;
    }
    let pipes = t.matches('|').count() as i32;
    let dashes = t.matches(" - ").count() as i32;
    q -= (pipes + dashes) * 8;
    q
}

/// Assemble the final reader-mode HTML wrapped in standard markup.
pub fn render_article(article: &Article) -> String {
    format!(
        "<!doctype html><html lang=\"{lang}\"><head><meta charset=\"utf-8\"><title>{title}</title>\
<style>body{{max-width:42em;margin:2em auto;padding:0 1em;font:18px/1.6 Georgia,serif;color:#222;}}\
h1{{font:bold 28px/1.3 Georgia,serif;}}.byline{{color:#666;font-size:14px;margin-bottom:1.5em;}}\
img{{max-width:100%;height:auto;}}p{{margin:0 0 1em;}}</style></head>\
<body><h1>{title}</h1>{byline_block}{content}</body></html>",
        lang = article.lang,
        title = html_escape(&article.title),
        byline_block = if article.byline.is_empty() {
            String::new()
        } else {
            format!("<p class=\"byline\">{}</p>", html_escape(&article.byline))
        },
        content = article.content_html,
    )
}

pub fn html_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&#39;"),
            _ => out.push(c),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn blk(tag: &str, id: &str, class: &str, text: &str, pcount: u32, link_chars: u32) -> Block {
        Block {
            tag: tag.into(),
            id: id.into(),
            class: class.into(),
            text: text.into(),
            inner_html: format!("<p>{text}</p>", text = text),
            child_p_count: pcount,
            link_text_chars: link_chars,
        }
    }

    #[test]
    fn article_class_wins() {
        let blocks = vec![
            blk(
                "div",
                "main-content",
                "article",
                "This is the body of the article. It has commas, multiple sentences, and lots of text.",
                3,
                10,
            ),
            blk(
                "div",
                "sidebar",
                "promo widget",
                "Sign up for our newsletter! Click here. Sale ends today.",
                0,
                40,
            ),
        ];
        let (idx, _s) = pick_top(&blocks).unwrap();
        assert_eq!(idx, 0);
    }

    #[test]
    fn high_link_density_penalised() {
        let mostly_links = blk("div", "", "navlist", "Home About Contact Privacy", 0, 20);
        let prose = blk(
            "div",
            "story",
            "article",
            "Today we report on a major event. There were many witnesses, and they all said the same thing.",
            2,
            0,
        );
        assert!(score_block(&prose) > score_block(&mostly_links));
    }

    #[test]
    fn negative_class_wipes_score() {
        let bad = blk(
            "div",
            "comments-section",
            "comment-list",
            "Some comments thread here.",
            1,
            0,
        );
        assert!(score_block(&bad) < 0.0);
    }

    #[test]
    fn title_picks_longest_reasonable() {
        let t = pick_title(&["Home", "How rain forms — Science Today", "Science Today"]);
        assert_eq!(t, "How rain forms — Science Today");
    }

    #[test]
    fn renders_full_document() {
        let a = Article {
            title: "Hi".into(),
            byline: "By Author".into(),
            content_html: "<p>Body.</p>".into(),
            lang: "en".into(),
            text_word_count: 2,
        };
        let html = render_article(&a);
        assert!(html.contains("<h1>Hi</h1>"));
        assert!(html.contains("By Author"));
        assert!(html.contains("<p>Body.</p>"));
    }
}
