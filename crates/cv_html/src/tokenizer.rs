//! WHATWG HTML tokenizer — practical subset.
//!
//! Implements the most-traveled states:
//!   - Data, RCDATA, RAWTEXT, Script
//!   - Tag open / tag name / end tag open
//!   - Attribute name, before / after attribute name
//!   - Attribute value (double-quoted, single-quoted, unquoted)
//!   - Self-closing start tag
//!   - Markup declaration (Doctype, comments, CDATA marker)
//!   - Comment start / inside / end / end-bang
//!   - Doctype (name, public/system keywords + identifiers)
//!   - Numeric and named character references (limited table)
//!
//! Edge cases of the full WHATWG state list (CDATA inside SVG/MathML,
//! ambiguous ampersand, character-reference end with extra `;`) are
//! tracked as TODOs but tolerated as best-effort.

#![allow(clippy::too_many_lines)]

use crate::entities::lookup_named;
use crate::token::{Attribute, Token};

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum State {
    Data,
    RcData,
    RawText,
    ScriptData,
    TagOpen,
    EndTagOpen,
    TagName,
    BeforeAttributeName,
    AttributeName,
    AfterAttributeName,
    BeforeAttributeValue,
    AttributeValueDoubleQuoted,
    AttributeValueSingleQuoted,
    AttributeValueUnquoted,
    AfterAttributeValueQuoted,
    SelfClosingStartTag,
    MarkupDeclarationOpen,
    CommentStart,
    Comment,
    CommentLessThanSign,
    CommentEndDash,
    CommentEnd,
    DoctypeOpen,
    BeforeDoctypeName,
    DoctypeName,
    AfterDoctypeName,
    BogusComment,
}

#[derive(Debug)]
pub struct Tokenizer<'a> {
    src: &'a [u8],
    pos: usize,
    state: State,
    /// Where to return after RCDATA/RAWTEXT/Script's tag-open detour.
    raw_kind_tag: Option<&'static str>,

    text_run: String,
    current_tag_name: String,
    current_tag_attrs: Vec<Attribute>,
    current_tag_self_closing: bool,
    current_is_end_tag: bool,
    current_attr_name: String,
    current_attr_value: String,
    current_comment: String,
    current_doctype_name: String,

    out: Vec<Token>,
    done: bool,
}

impl<'a> Tokenizer<'a> {
    pub fn new(src: &'a str) -> Self {
        Self {
            src: src.as_bytes(),
            pos: 0,
            state: State::Data,
            raw_kind_tag: None,
            text_run: String::new(),
            current_tag_name: String::new(),
            current_tag_attrs: Vec::new(),
            current_tag_self_closing: false,
            current_is_end_tag: false,
            current_attr_name: String::new(),
            current_attr_value: String::new(),
            current_comment: String::new(),
            current_doctype_name: String::new(),
            out: Vec::new(),
            done: false,
        }
    }

    pub fn run(mut self) -> Vec<Token> {
        while !self.done {
            self.step();
        }
        self.out
    }

    fn step(&mut self) {
        let c = self.peek();
        match self.state {
            State::Data => self.s_data(c),
            State::RcData => self.s_rawish(c, State::RcData, true),
            State::RawText => self.s_rawish(c, State::RawText, false),
            State::ScriptData => self.s_rawish(c, State::ScriptData, false),
            State::TagOpen => self.s_tag_open(c),
            State::EndTagOpen => self.s_end_tag_open(c),
            State::TagName => self.s_tag_name(c),
            State::BeforeAttributeName => self.s_before_attr_name(c),
            State::AttributeName => self.s_attr_name(c),
            State::AfterAttributeName => self.s_after_attr_name(c),
            State::BeforeAttributeValue => self.s_before_attr_value(c),
            State::AttributeValueDoubleQuoted => self.s_attr_value_quoted(c, b'"'),
            State::AttributeValueSingleQuoted => self.s_attr_value_quoted(c, b'\''),
            State::AttributeValueUnquoted => self.s_attr_value_unquoted(c),
            State::AfterAttributeValueQuoted => self.s_after_attr_value_quoted(c),
            State::SelfClosingStartTag => self.s_self_closing_start_tag(c),
            State::MarkupDeclarationOpen => self.s_markup_decl_open(),
            State::CommentStart => self.s_comment_start(c),
            State::Comment => self.s_comment(c),
            State::CommentLessThanSign => self.s_comment_less_than_sign(c),
            State::CommentEndDash => self.s_comment_end_dash(c),
            State::CommentEnd => self.s_comment_end(c),
            State::DoctypeOpen => self.s_doctype_open(c),
            State::BeforeDoctypeName => self.s_before_doctype_name(c),
            State::DoctypeName => self.s_doctype_name(c),
            State::AfterDoctypeName => self.s_after_doctype_name(c),
            State::BogusComment => self.s_bogus_comment(c),
        }
    }

    fn peek(&self) -> Option<u8> {
        self.src.get(self.pos).copied()
    }

    fn consume(&mut self) -> Option<u8> {
        let c = self.peek();
        if c.is_some() {
            self.pos += 1;
        }
        c
    }

    fn flush_text(&mut self) {
        if !self.text_run.is_empty() {
            let s = std::mem::take(&mut self.text_run);
            self.out.push(Token::Text(s));
        }
    }

    fn emit_tag(&mut self) {
        let name = std::mem::take(&mut self.current_tag_name).to_ascii_lowercase();
        let attrs = std::mem::take(&mut self.current_tag_attrs);
        let self_closing = std::mem::take(&mut self.current_tag_self_closing);
        let end = std::mem::take(&mut self.current_is_end_tag);

        // Switch tokenizer mode for content of script/style/etc.
        if !end {
            match name.as_str() {
                "script" => {
                    self.raw_kind_tag = Some("script");
                    self.state = State::ScriptData;
                }
                "style" | "xmp" | "iframe" | "noembed" | "noframes" | "noscript" => {
                    self.raw_kind_tag = Some(Box::leak(name.clone().into_boxed_str()));
                    self.state = State::RawText;
                }
                "title" | "textarea" => {
                    self.raw_kind_tag = Some(Box::leak(name.clone().into_boxed_str()));
                    self.state = State::RcData;
                }
                _ => {}
            }
        }

        let tok = if end {
            Token::EndTag { name }
        } else {
            Token::StartTag {
                name,
                attrs,
                self_closing,
            }
        };
        self.out.push(tok);
    }

    fn s_data(&mut self, c: Option<u8>) {
        match c {
            None => {
                self.flush_text();
                self.out.push(Token::Eof);
                self.done = true;
            }
            Some(b'<') => {
                self.flush_text();
                self.consume();
                self.state = State::TagOpen;
            }
            Some(b'&') => {
                self.consume();
                self.tokenize_char_ref();
            }
            Some(_) => {
                let ch = self.consume_char();
                self.text_run.push(ch);
            }
        }
    }

    fn s_rawish(&mut self, c: Option<u8>, _kind: State, allow_refs: bool) {
        match c {
            None => {
                self.flush_text();
                self.out.push(Token::Eof);
                self.done = true;
            }
            Some(b'<') => {
                // Look for </kind followed by tag-terminator literally.
                let kind = self.raw_kind_tag.unwrap_or("");
                let needle: Vec<u8> = format!("</{kind}").bytes().collect();
                let pos = self.pos;
                if self.src.get(pos..pos + needle.len()).is_some_and(|s| {
                    // Case-insensitive ASCII match — HTML is case-insensitive for tag names.
                    s.iter()
                        .zip(needle.iter())
                        .all(|(a, b)| a.eq_ignore_ascii_case(b))
                }) {
                    let next = self.src.get(pos + needle.len()).copied();
                    if matches!(
                        next,
                        Some(b' ' | b'\t' | b'\n' | b'\r' | b'\x0C' | b'/' | b'>')
                    ) {
                        self.flush_text();
                        self.raw_kind_tag = None;
                        self.pos += 2; // consume "</"
                        self.current_tag_name.clear();
                        self.current_tag_attrs.clear();
                        self.current_tag_self_closing = false;
                        self.current_is_end_tag = true;
                        self.state = State::TagName;
                        return;
                    }
                }
                let ch = self.consume_char();
                self.text_run.push(ch);
            }
            Some(b'&') if allow_refs => {
                self.consume();
                self.tokenize_char_ref();
            }
            Some(_) => {
                let ch = self.consume_char();
                self.text_run.push(ch);
            }
        }
    }

    fn s_tag_open(&mut self, c: Option<u8>) {
        match c {
            None => {
                self.text_run.push('<');
                self.out.push(Token::Eof);
                self.done = true;
            }
            Some(b'!') => {
                self.consume();
                self.state = State::MarkupDeclarationOpen;
            }
            Some(b'/') => {
                self.consume();
                self.state = State::EndTagOpen;
            }
            Some(b'?') => {
                self.consume();
                self.current_comment.clear();
                self.current_comment.push('?');
                self.state = State::BogusComment;
            }
            Some(c) if c.is_ascii_alphabetic() => {
                self.current_tag_name.clear();
                self.current_tag_attrs.clear();
                self.current_tag_self_closing = false;
                self.current_is_end_tag = false;
                self.state = State::TagName;
            }
            Some(_) => {
                self.text_run.push('<');
                self.state = State::Data;
            }
        }
    }

    fn s_end_tag_open(&mut self, c: Option<u8>) {
        match c {
            None => {
                self.text_run.push('<');
                self.text_run.push('/');
                self.out.push(Token::Eof);
                self.done = true;
            }
            Some(c) if c.is_ascii_alphabetic() => {
                self.current_tag_name.clear();
                self.current_tag_attrs.clear();
                self.current_tag_self_closing = false;
                self.current_is_end_tag = true;
                self.state = State::TagName;
            }
            Some(b'>') => {
                self.consume();
                self.state = State::Data;
            }
            Some(_) => {
                self.current_comment.clear();
                self.state = State::BogusComment;
            }
        }
    }

    fn s_tag_name(&mut self, c: Option<u8>) {
        match c {
            None => self.eof_in_tag(),
            Some(b'\t' | b'\n' | b'\r' | b'\x0C' | b' ') => {
                self.consume();
                self.state = State::BeforeAttributeName;
            }
            Some(b'/') => {
                self.consume();
                self.state = State::SelfClosingStartTag;
            }
            Some(b'>') => {
                self.consume();
                self.emit_tag();
                // State might be changed to RAWTEXT/RCDATA/Script by emit_tag.
                if matches!(self.state, State::TagName) {
                    self.state = State::Data;
                }
            }
            Some(c) => {
                self.consume();
                self.current_tag_name.push(c.to_ascii_lowercase() as char);
            }
        }
    }

    fn s_before_attr_name(&mut self, c: Option<u8>) {
        match c {
            None => self.eof_in_tag(),
            Some(b'\t' | b'\n' | b'\r' | b'\x0C' | b' ') => {
                self.consume();
            }
            Some(b'/') => {
                self.consume();
                self.state = State::SelfClosingStartTag;
            }
            Some(b'>') => {
                self.consume();
                self.emit_tag();
                if matches!(self.state, State::BeforeAttributeName) {
                    self.state = State::Data;
                }
            }
            Some(_) => {
                self.current_attr_name.clear();
                self.current_attr_value.clear();
                self.state = State::AttributeName;
            }
        }
    }

    fn s_attr_name(&mut self, c: Option<u8>) {
        match c {
            None | Some(b'\t' | b'\n' | b'\r' | b'\x0C' | b' ' | b'/' | b'>') => {
                self.state = State::AfterAttributeName;
            }
            Some(b'=') => {
                self.consume();
                self.state = State::BeforeAttributeValue;
            }
            Some(c) => {
                self.consume();
                self.current_attr_name.push(c.to_ascii_lowercase() as char);
            }
        }
    }

    fn s_after_attr_name(&mut self, c: Option<u8>) {
        match c {
            None => self.eof_in_tag(),
            Some(b'\t' | b'\n' | b'\r' | b'\x0C' | b' ') => {
                self.consume();
            }
            Some(b'/') => {
                self.consume();
                self.commit_attr_no_value();
                self.state = State::SelfClosingStartTag;
            }
            Some(b'=') => {
                self.consume();
                self.state = State::BeforeAttributeValue;
            }
            Some(b'>') => {
                self.consume();
                self.commit_attr_no_value();
                self.emit_tag();
                if matches!(self.state, State::AfterAttributeName) {
                    self.state = State::Data;
                }
            }
            Some(_) => {
                self.commit_attr_no_value();
                self.current_attr_name.clear();
                self.current_attr_value.clear();
                self.state = State::AttributeName;
            }
        }
    }

    fn commit_attr_no_value(&mut self) {
        if !self.current_attr_name.is_empty() {
            self.current_tag_attrs.push(Attribute {
                name: std::mem::take(&mut self.current_attr_name),
                value: String::new(),
            });
        }
    }

    fn commit_attr_with_value(&mut self) {
        if !self.current_attr_name.is_empty() {
            self.current_tag_attrs.push(Attribute {
                name: std::mem::take(&mut self.current_attr_name),
                value: std::mem::take(&mut self.current_attr_value),
            });
        }
    }

    fn s_before_attr_value(&mut self, c: Option<u8>) {
        match c {
            Some(b'\t' | b'\n' | b'\r' | b'\x0C' | b' ') => {
                self.consume();
            }
            Some(b'"') => {
                self.consume();
                self.state = State::AttributeValueDoubleQuoted;
            }
            Some(b'\'') => {
                self.consume();
                self.state = State::AttributeValueSingleQuoted;
            }
            Some(b'>') => {
                self.consume();
                self.commit_attr_with_value();
                self.emit_tag();
                if matches!(self.state, State::BeforeAttributeValue) {
                    self.state = State::Data;
                }
            }
            _ => {
                self.state = State::AttributeValueUnquoted;
            }
        }
    }

    fn s_attr_value_quoted(&mut self, c: Option<u8>, quote: u8) {
        match c {
            None => self.eof_in_tag(),
            Some(b) if b == quote => {
                self.consume();
                self.commit_attr_with_value();
                self.state = State::AfterAttributeValueQuoted;
            }
            Some(b'&') => {
                self.consume();
                self.tokenize_char_ref_into_attr();
            }
            Some(_) => {
                let ch = self.consume_char();
                self.current_attr_value.push(ch);
            }
        }
    }

    fn s_attr_value_unquoted(&mut self, c: Option<u8>) {
        match c {
            None => self.eof_in_tag(),
            Some(b'\t' | b'\n' | b'\r' | b'\x0C' | b' ') => {
                self.consume();
                self.commit_attr_with_value();
                self.state = State::BeforeAttributeName;
            }
            Some(b'>') => {
                self.consume();
                self.commit_attr_with_value();
                self.emit_tag();
                if matches!(self.state, State::AttributeValueUnquoted) {
                    self.state = State::Data;
                }
            }
            Some(b'&') => {
                self.consume();
                self.tokenize_char_ref_into_attr();
            }
            Some(_) => {
                let ch = self.consume_char();
                self.current_attr_value.push(ch);
            }
        }
    }

    fn s_after_attr_value_quoted(&mut self, c: Option<u8>) {
        match c {
            None => self.eof_in_tag(),
            Some(b'\t' | b'\n' | b'\r' | b'\x0C' | b' ') => {
                self.consume();
                self.state = State::BeforeAttributeName;
            }
            Some(b'/') => {
                self.consume();
                self.state = State::SelfClosingStartTag;
            }
            Some(b'>') => {
                self.consume();
                self.emit_tag();
                if matches!(self.state, State::AfterAttributeValueQuoted) {
                    self.state = State::Data;
                }
            }
            Some(_) => {
                self.state = State::BeforeAttributeName;
            }
        }
    }

    fn s_self_closing_start_tag(&mut self, c: Option<u8>) {
        match c {
            None => self.eof_in_tag(),
            Some(b'>') => {
                self.consume();
                self.current_tag_self_closing = true;
                self.emit_tag();
                if matches!(self.state, State::SelfClosingStartTag) {
                    self.state = State::Data;
                }
            }
            Some(_) => {
                self.state = State::BeforeAttributeName;
            }
        }
    }

    fn s_markup_decl_open(&mut self) {
        if self.src.get(self.pos..self.pos + 2) == Some(b"--") {
            self.pos += 2;
            self.current_comment.clear();
            self.state = State::CommentStart;
        } else if self
            .src
            .get(self.pos..self.pos + 7)
            .map(|s| s.eq_ignore_ascii_case(b"DOCTYPE"))
            == Some(true)
        {
            self.pos += 7;
            self.state = State::DoctypeOpen;
        } else if self.src.get(self.pos..self.pos + 7) == Some(b"[CDATA[") {
            // Outside foreign content the spec says: bogus comment.
            self.pos += 7;
            self.current_comment.clear();
            self.current_comment.push_str("[CDATA[");
            self.state = State::BogusComment;
        } else {
            self.current_comment.clear();
            self.state = State::BogusComment;
        }
    }

    fn s_comment_start(&mut self, c: Option<u8>) {
        match c {
            Some(b'-') => {
                self.consume();
                self.state = State::CommentEndDash;
            }
            Some(b'>') => {
                self.consume();
                self.out
                    .push(Token::Comment(std::mem::take(&mut self.current_comment)));
                self.state = State::Data;
            }
            _ => self.state = State::Comment,
        }
    }

    fn s_comment(&mut self, c: Option<u8>) {
        match c {
            None => {
                self.out
                    .push(Token::Comment(std::mem::take(&mut self.current_comment)));
                self.out.push(Token::Eof);
                self.done = true;
            }
            Some(b'<') => {
                self.consume();
                self.current_comment.push('<');
                self.state = State::CommentLessThanSign;
            }
            Some(b'-') => {
                self.consume();
                self.state = State::CommentEndDash;
            }
            Some(_) => {
                let ch = self.consume_char();
                self.current_comment.push(ch);
            }
        }
    }

    fn s_comment_less_than_sign(&mut self, c: Option<u8>) {
        match c {
            Some(b'!' | b'<') => {
                self.consume();
                self.current_comment.push('!');
            }
            _ => self.state = State::Comment,
        }
    }

    fn s_comment_end_dash(&mut self, c: Option<u8>) {
        match c {
            Some(b'-') => {
                self.consume();
                self.state = State::CommentEnd;
            }
            _ => {
                self.current_comment.push('-');
                self.state = State::Comment;
            }
        }
    }

    fn s_comment_end(&mut self, c: Option<u8>) {
        match c {
            Some(b'>') => {
                self.consume();
                self.out
                    .push(Token::Comment(std::mem::take(&mut self.current_comment)));
                self.state = State::Data;
            }
            Some(b'-') => {
                self.consume();
                self.current_comment.push('-');
            }
            _ => {
                self.current_comment.push_str("--");
                self.state = State::Comment;
            }
        }
    }

    fn s_doctype_open(&mut self, c: Option<u8>) {
        match c {
            Some(b'\t' | b'\n' | b'\r' | b'\x0C' | b' ') => {
                self.consume();
                self.state = State::BeforeDoctypeName;
            }
            _ => self.state = State::BeforeDoctypeName,
        }
    }

    fn s_before_doctype_name(&mut self, c: Option<u8>) {
        match c {
            None => {
                self.out.push(Token::Doctype {
                    name: None,
                    public_id: None,
                    system_id: None,
                    force_quirks: true,
                });
                self.out.push(Token::Eof);
                self.done = true;
            }
            Some(b'\t' | b'\n' | b'\r' | b'\x0C' | b' ') => {
                self.consume();
            }
            Some(b'>') => {
                self.consume();
                self.out.push(Token::Doctype {
                    name: None,
                    public_id: None,
                    system_id: None,
                    force_quirks: true,
                });
                self.state = State::Data;
            }
            Some(_) => {
                self.current_doctype_name.clear();
                self.state = State::DoctypeName;
            }
        }
    }

    fn s_doctype_name(&mut self, c: Option<u8>) {
        match c {
            None => {
                self.out.push(Token::Doctype {
                    name: Some(std::mem::take(&mut self.current_doctype_name)),
                    public_id: None,
                    system_id: None,
                    force_quirks: true,
                });
                self.out.push(Token::Eof);
                self.done = true;
            }
            Some(b'\t' | b'\n' | b'\r' | b'\x0C' | b' ') => {
                self.consume();
                self.state = State::AfterDoctypeName;
            }
            Some(b'>') => {
                self.consume();
                self.out.push(Token::Doctype {
                    name: Some(std::mem::take(&mut self.current_doctype_name)),
                    public_id: None,
                    system_id: None,
                    force_quirks: false,
                });
                self.state = State::Data;
            }
            Some(c) => {
                self.consume();
                self.current_doctype_name
                    .push(c.to_ascii_lowercase() as char);
            }
        }
    }

    fn s_after_doctype_name(&mut self, c: Option<u8>) {
        // We don't fully parse PUBLIC/SYSTEM identifiers — just walk to '>'.
        match c {
            None => {
                self.out.push(Token::Doctype {
                    name: Some(std::mem::take(&mut self.current_doctype_name)),
                    public_id: None,
                    system_id: None,
                    force_quirks: true,
                });
                self.out.push(Token::Eof);
                self.done = true;
            }
            Some(b'>') => {
                self.consume();
                self.out.push(Token::Doctype {
                    name: Some(std::mem::take(&mut self.current_doctype_name)),
                    public_id: None,
                    system_id: None,
                    force_quirks: false,
                });
                self.state = State::Data;
            }
            Some(_) => {
                self.consume();
            }
        }
    }

    fn s_bogus_comment(&mut self, c: Option<u8>) {
        match c {
            None => {
                self.out
                    .push(Token::Comment(std::mem::take(&mut self.current_comment)));
                self.out.push(Token::Eof);
                self.done = true;
            }
            Some(b'>') => {
                self.consume();
                self.out
                    .push(Token::Comment(std::mem::take(&mut self.current_comment)));
                self.state = State::Data;
            }
            Some(_) => {
                let ch = self.consume_char();
                self.current_comment.push(ch);
            }
        }
    }

    fn eof_in_tag(&mut self) {
        // Drop the partial tag, emit text+EOF.
        self.flush_text();
        self.out.push(Token::Eof);
        self.done = true;
    }

    fn consume_char(&mut self) -> char {
        // UTF-8 lead-byte decode. Falls back to U+FFFD on malformed.
        let b = match self.consume() {
            Some(b) => b,
            None => return '\u{FFFD}',
        };
        if b < 0x80 {
            return b as char;
        }
        let extra = if b >= 0xF0 {
            3
        } else if b >= 0xE0 {
            2
        } else if b >= 0xC0 {
            1
        } else {
            return '\u{FFFD}';
        };
        let mut bytes = [0u8; 4];
        bytes[0] = b;
        for i in 0..extra {
            match self.peek() {
                Some(nb) if nb & 0xC0 == 0x80 => {
                    bytes[i + 1] = nb;
                    self.pos += 1;
                }
                _ => return '\u{FFFD}',
            }
        }
        std::str::from_utf8(&bytes[..=extra])
            .ok()
            .and_then(|s| s.chars().next())
            .unwrap_or('\u{FFFD}')
    }

    fn tokenize_char_ref(&mut self) {
        match self.read_char_ref() {
            Some(s) => self.text_run.push_str(&s),
            None => self.text_run.push('&'),
        }
    }

    fn tokenize_char_ref_into_attr(&mut self) {
        match self.read_char_ref() {
            Some(s) => self.current_attr_value.push_str(&s),
            None => self.current_attr_value.push('&'),
        }
    }

    fn read_char_ref(&mut self) -> Option<String> {
        // Numeric: &#1234; &#xABCD;
        if self.peek() == Some(b'#') {
            self.consume();
            let hex = matches!(self.peek(), Some(b'x' | b'X'));
            if hex {
                self.consume();
            }
            let mut v: u32 = 0;
            let mut any = false;
            loop {
                match self.peek() {
                    Some(c) if c.is_ascii_digit() => {
                        v = v.wrapping_mul(if hex { 16 } else { 10 }) + u32::from(c - b'0');
                        any = true;
                        self.consume();
                    }
                    Some(c) if hex && c.is_ascii_hexdigit() => {
                        let d = (c | 0x20) - b'a' + 10;
                        v = v.wrapping_mul(16) + u32::from(d);
                        any = true;
                        self.consume();
                    }
                    _ => break,
                }
            }
            if self.peek() == Some(b';') {
                self.consume();
            }
            if !any {
                return None;
            }
            // WHATWG HTML §13.2.5.72: numeric character references in the
            // Windows-1252 C1 range (0x80–0x9F) are remapped through the
            // Windows-1252 table rather than used as raw Unicode code points.
            // Code point 0, surrogate pairs, and values > U+10FFFF all
            // produce U+FFFD (replacement character).
            #[rustfmt::skip]
            const C1_REMAP: [u32; 32] = [
                0x20AC, 0x0081, 0x201A, 0x0192, 0x201E, 0x2026, 0x2020, 0x2021,
                0x02C6, 0x2030, 0x0160, 0x2039, 0x0152, 0x008D, 0x017D, 0x008F,
                0x0090, 0x2018, 0x2019, 0x201C, 0x201D, 0x2022, 0x2013, 0x2014,
                0x02DC, 0x2122, 0x0161, 0x203A, 0x0153, 0x009D, 0x017E, 0x0178,
            ];
            let v = if v >= 0x80 && v <= 0x9F {
                C1_REMAP[(v - 0x80) as usize]
            } else if v == 0 || (v >= 0xD800 && v <= 0xDFFF) || v > 0x10FFFF {
                0xFFFD
            } else {
                v
            };
            let ch = char::from_u32(v).unwrap_or('\u{FFFD}');
            return Some(ch.to_string());
        }
        // Named.
        let start = self.pos;
        while let Some(c) = self.peek() {
            if c.is_ascii_alphanumeric() {
                self.consume();
            } else {
                break;
            }
        }
        let name = std::str::from_utf8(&self.src[start..self.pos]).ok()?;
        if name.is_empty() {
            return None;
        }
        let semi = self.peek() == Some(b';');
        if semi {
            self.consume();
        }
        match lookup_named(name) {
            Some(s) => Some(s.to_string()),
            None => {
                // Not a known reference. We've already consumed the `&`
                // (in the caller) and the would-be name chars (in the
                // loop above). Emit them as literal text — don't rewind
                // self.pos, because the previous code did `pos = start-1`
                // and then sliced `[start..pos]`, which panicked whenever
                // `start > 0` because the range was reversed.
                let mut out = String::with_capacity(1 + name.len() + 1);
                out.push('&');
                out.push_str(name);
                if semi {
                    out.push(';');
                }
                Some(out)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn toks(s: &str) -> Vec<Token> {
        Tokenizer::new(s).run()
    }

    #[test]
    fn text_only() {
        let t = toks("hello world");
        assert_eq!(t, vec![Token::Text("hello world".into()), Token::Eof]);
    }

    #[test]
    fn simple_tag_pair() {
        let t = toks("<p>hi</p>");
        assert_eq!(
            t,
            vec![
                Token::StartTag {
                    name: "p".into(),
                    attrs: vec![],
                    self_closing: false,
                },
                Token::Text("hi".into()),
                Token::EndTag { name: "p".into() },
                Token::Eof,
            ]
        );
    }

    #[test]
    fn attributes_mixed_quoting() {
        let t = toks(r#"<a href="https://x.test" target='_blank' disabled>x</a>"#);
        let attrs = match &t[0] {
            Token::StartTag { attrs, .. } => attrs,
            _ => panic!("expected start tag, got {:?}", t[0]),
        };
        assert_eq!(attrs.len(), 3);
        assert_eq!(attrs[0].name, "href");
        assert_eq!(attrs[0].value, "https://x.test");
        assert_eq!(attrs[1].name, "target");
        assert_eq!(attrs[1].value, "_blank");
        assert_eq!(attrs[2].name, "disabled");
        assert_eq!(attrs[2].value, "");
    }

    #[test]
    fn comment() {
        let t = toks("<!-- hi there -->after");
        assert_eq!(
            t,
            vec![
                Token::Comment(" hi there ".into()),
                Token::Text("after".into()),
                Token::Eof,
            ]
        );
    }

    #[test]
    fn doctype() {
        let t = toks("<!DOCTYPE html><html></html>");
        assert!(
            matches!(t[0], Token::Doctype { name: Some(ref n), force_quirks: false, .. } if n == "html")
        );
    }

    #[test]
    fn entities() {
        let t = toks("a &amp; b &lt;c&gt;");
        assert_eq!(t[0], Token::Text("a & b <c>".into()));
    }

    #[test]
    fn numeric_entity() {
        let t = toks("&#65;&#x42;");
        assert_eq!(t[0], Token::Text("AB".into()));
    }

    /// WHATWG HTML §13.2.5.72: numeric character references in the
    /// Windows-1252 C1 range (0x80–0x9F) must be remapped to the correct
    /// Unicode code points — not emitted as raw C1 control characters.
    /// Legacy CMS HTML uses &#146;/&#147;/&#148;/&#151; for curly quotes
    /// and em dashes.
    #[test]
    fn numeric_entity_c1_windows1252_remap() {
        // &#151; = 0x97 → Windows-1252 → U+2014 EM DASH
        let t = toks("&#151;");
        assert_eq!(t[0], Token::Text("\u{2014}".into()), "&#151; must be em dash U+2014");

        // &#146; = 0x92 → Windows-1252 → U+2019 RIGHT SINGLE QUOTATION MARK
        let t = toks("&#146;");
        assert_eq!(t[0], Token::Text("\u{2019}".into()), "&#146; must be right single quote U+2019");

        // &#147; = 0x93 → Windows-1252 → U+201C LEFT DOUBLE QUOTATION MARK
        let t = toks("&#147;");
        assert_eq!(t[0], Token::Text("\u{201C}".into()), "&#147; must be left double quote U+201C");

        // &#148; = 0x94 → Windows-1252 → U+201D RIGHT DOUBLE QUOTATION MARK
        let t = toks("&#148;");
        assert_eq!(t[0], Token::Text("\u{201D}".into()), "&#148; must be right double quote U+201D");

        // &#128; = 0x80 → Windows-1252 → U+20AC EURO SIGN
        let t = toks("&#128;");
        assert_eq!(t[0], Token::Text("\u{20AC}".into()), "&#128; must be euro sign U+20AC");

        // &#0; → U+FFFD replacement character (code point 0 is forbidden)
        let t = toks("&#0;");
        assert_eq!(t[0], Token::Text("\u{FFFD}".into()), "&#0; must be U+FFFD");

        // Normal code point above 0x9F must pass through unchanged
        let t = toks("&#8364;");
        // U+20AC (euro) is also 8364 decimal — should pass through as euro sign
        assert_eq!(t[0], Token::Text("\u{20AC}".into()), "&#8364; (decimal euro) must pass through");
    }

    #[test]
    fn self_closing() {
        let t = toks("<br/>");
        assert!(matches!(
            &t[0],
            Token::StartTag { name, self_closing: true, .. } if name == "br"
        ));
    }

    #[test]
    fn script_content_is_raw() {
        let t = toks("<script>var x = 1 < 2 && 3;</script>");
        // Inside script, "<" is not a tag — should be a single text run between the start/end tag tokens.
        let texts: Vec<&str> = t
            .iter()
            .filter_map(|tok| match tok {
                Token::Text(s) => Some(s.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(texts.concat(), "var x = 1 < 2 && 3;");
    }

    #[test]
    fn real_world_snippet() {
        let html = r#"<!DOCTYPE html><html><head><title>X</title></head><body><h1>Hello</h1><p class="a">World</p></body></html>"#;
        let t = toks(html);
        assert!(!t.is_empty());
        assert!(t.iter().any(|tok| matches!(tok, Token::Doctype { .. })));
        assert!(
            t.iter()
                .any(|tok| matches!(tok, Token::StartTag { name, .. } if name == "h1"))
        );
    }
}
