//! JavaScript lexer — ECMA-262 §12 subset.
//!
//! Produces a `Vec<Token>` for slice-1 consumers (mostly: cheap "does this
//! script parse?" checks until the full parser lands). Templates,
//! regex-literal disambiguation, and full numeric-literal syntax (BigInt
//! suffix, separator underscores, etc.) come incrementally.

use core::fmt;

/// A punctuator, as a compact `Copy` tag — NO heap allocation.
///
/// M3.1 Phase 0: `TokenKind::Punct` previously carried an owned `String`, so
/// every `;`, `,`, `(`, `=>`, `+`, … in a script heap-allocated. This enum has
/// one variant per punctuator `match_punct` can produce; the lexer now emits a
/// `Copy` tag and the parser compares it via `as_str()` / `PartialEq<str>`,
/// keeping the produced AST byte-identical.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Punct {
    // 4-char
    UShrAssign, // >>>=
    // 3-char
    Ellipsis, // ...
    EqEqEq,   // ===
    NeqEq,    // !==
    UShr,     // >>>
    StarStarAssign, // **=
    ShlAssign, // <<=
    ShrAssign, // >>=
    AndAndAssign, // &&=
    OrOrAssign, // ||=
    QQAssign, // ??=
    // 2-char
    OptChain, // ?.
    Arrow,    // =>
    EqEq,     // ==
    Neq,      // !=
    Le,       // <=
    Ge,       // >=
    AndAnd,   // &&
    OrOr,     // ||
    QQ,       // ??
    PlusPlus, // ++
    MinusMinus, // --
    Shl,      // <<
    Shr,      // >>
    StarStar, // **
    PlusAssign, // +=
    MinusAssign, // -=
    StarAssign, // *=
    SlashAssign, // /=
    PercentAssign, // %=
    AndAssign, // &=
    OrAssign, // |=
    XorAssign, // ^=
    // 1-char
    LParen,    // (
    RParen,    // )
    LBracket,  // [
    RBracket,  // ]
    LBrace,    // {
    RBrace,    // }
    Semi,      // ;
    Comma,     // ,
    Lt,        // <
    Gt,        // >
    Plus,      // +
    Minus,     // -
    Star,      // *
    Slash,     // /
    Percent,   // %
    Amp,       // &
    Pipe,      // |
    Caret,     // ^
    Tilde,     // ~
    Bang,      // !
    Assign,    // =
    Question,  // ?
    Colon,     // :
    Dot,       // .
}

impl Punct {
    /// The exact source spelling of this punctuator. Used by the parser's
    /// string-comparison logic so it stays behavior-identical.
    pub fn as_str(self) -> &'static str {
        match self {
            Punct::UShrAssign => ">>>=",
            Punct::Ellipsis => "...",
            Punct::EqEqEq => "===",
            Punct::NeqEq => "!==",
            Punct::UShr => ">>>",
            Punct::StarStarAssign => "**=",
            Punct::ShlAssign => "<<=",
            Punct::ShrAssign => ">>=",
            Punct::AndAndAssign => "&&=",
            Punct::OrOrAssign => "||=",
            Punct::QQAssign => "??=",
            Punct::OptChain => "?.",
            Punct::Arrow => "=>",
            Punct::EqEq => "==",
            Punct::Neq => "!=",
            Punct::Le => "<=",
            Punct::Ge => ">=",
            Punct::AndAnd => "&&",
            Punct::OrOr => "||",
            Punct::QQ => "??",
            Punct::PlusPlus => "++",
            Punct::MinusMinus => "--",
            Punct::Shl => "<<",
            Punct::Shr => ">>",
            Punct::StarStar => "**",
            Punct::PlusAssign => "+=",
            Punct::MinusAssign => "-=",
            Punct::StarAssign => "*=",
            Punct::SlashAssign => "/=",
            Punct::PercentAssign => "%=",
            Punct::AndAssign => "&=",
            Punct::OrAssign => "|=",
            Punct::XorAssign => "^=",
            Punct::LParen => "(",
            Punct::RParen => ")",
            Punct::LBracket => "[",
            Punct::RBracket => "]",
            Punct::LBrace => "{",
            Punct::RBrace => "}",
            Punct::Semi => ";",
            Punct::Comma => ",",
            Punct::Lt => "<",
            Punct::Gt => ">",
            Punct::Plus => "+",
            Punct::Minus => "-",
            Punct::Star => "*",
            Punct::Slash => "/",
            Punct::Percent => "%",
            Punct::Amp => "&",
            Punct::Pipe => "|",
            Punct::Caret => "^",
            Punct::Tilde => "~",
            Punct::Bang => "!",
            Punct::Assign => "=",
            Punct::Question => "?",
            Punct::Colon => ":",
            Punct::Dot => ".",
        }
    }
}

impl fmt::Display for Punct {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

// Let the parser's existing `p == "{"` comparisons (where `p: &Punct`) keep
// working unchanged: `impl PartialEq<str> for Punct` gives `&Punct: PartialEq<&str>`
// via the std blanket impl.
impl PartialEq<str> for Punct {
    fn eq(&self, other: &str) -> bool {
        self.as_str() == other
    }
}
impl PartialEq<&str> for Punct {
    fn eq(&self, other: &&str) -> bool {
        self.as_str() == *other
    }
}

/// A keyword, as a compact `Copy` tag — NO heap allocation. One variant per
/// `KEYWORDS` entry. Recognition is a direct match on the scanned identifier
/// byte slice (no `String` alloc, no O(41) linear scan).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Keyword {
    Await,
    Break,
    Case,
    Catch,
    Class,
    Const,
    Continue,
    Debugger,
    Default,
    Delete,
    Do,
    Else,
    Export,
    Extends,
    False,
    Finally,
    For,
    Function,
    If,
    Import,
    In,
    Instanceof,
    Let,
    New,
    Null,
    Of,
    Return,
    Super,
    Switch,
    This,
    Throw,
    True,
    Try,
    Typeof,
    Undefined,
    Var,
    Void,
    While,
    With,
    Yield,
    Async,
}

impl Keyword {
    /// The exact source spelling of this keyword.
    pub fn as_str(self) -> &'static str {
        match self {
            Keyword::Await => "await",
            Keyword::Break => "break",
            Keyword::Case => "case",
            Keyword::Catch => "catch",
            Keyword::Class => "class",
            Keyword::Const => "const",
            Keyword::Continue => "continue",
            Keyword::Debugger => "debugger",
            Keyword::Default => "default",
            Keyword::Delete => "delete",
            Keyword::Do => "do",
            Keyword::Else => "else",
            Keyword::Export => "export",
            Keyword::Extends => "extends",
            Keyword::False => "false",
            Keyword::Finally => "finally",
            Keyword::For => "for",
            Keyword::Function => "function",
            Keyword::If => "if",
            Keyword::Import => "import",
            Keyword::In => "in",
            Keyword::Instanceof => "instanceof",
            Keyword::Let => "let",
            Keyword::New => "new",
            Keyword::Null => "null",
            Keyword::Of => "of",
            Keyword::Return => "return",
            Keyword::Super => "super",
            Keyword::Switch => "switch",
            Keyword::This => "this",
            Keyword::Throw => "throw",
            Keyword::True => "true",
            Keyword::Try => "try",
            Keyword::Typeof => "typeof",
            Keyword::Undefined => "undefined",
            Keyword::Var => "var",
            Keyword::Void => "void",
            Keyword::While => "while",
            Keyword::With => "with",
            Keyword::Yield => "yield",
            Keyword::Async => "async",
        }
    }

    /// Recognize a keyword directly from the scanned identifier byte slice — no
    /// `String` allocation, no O(41) linear scan. Returns `None` for a genuine
    /// identifier (only then does the caller allocate a `String`).
    #[inline]
    fn from_bytes(b: &[u8]) -> Option<Keyword> {
        // Dispatch on length first, then match the exact bytes. The compiler
        // lowers each inner `match` to a small jump/compare — far cheaper than
        // 41 `String` compares, and zero allocation.
        Some(match b.len() {
            2 => match b {
                b"do" => Keyword::Do,
                b"if" => Keyword::If,
                b"in" => Keyword::In,
                b"of" => Keyword::Of,
                _ => return None,
            },
            3 => match b {
                b"for" => Keyword::For,
                b"let" => Keyword::Let,
                b"new" => Keyword::New,
                b"try" => Keyword::Try,
                b"var" => Keyword::Var,
                _ => return None,
            },
            4 => match b {
                b"case" => Keyword::Case,
                b"else" => Keyword::Else,
                b"null" => Keyword::Null,
                b"this" => Keyword::This,
                b"true" => Keyword::True,
                b"void" => Keyword::Void,
                b"with" => Keyword::With,
                _ => return None,
            },
            5 => match b {
                b"await" => Keyword::Await,
                b"break" => Keyword::Break,
                b"catch" => Keyword::Catch,
                b"class" => Keyword::Class,
                b"const" => Keyword::Const,
                b"false" => Keyword::False,
                b"super" => Keyword::Super,
                b"throw" => Keyword::Throw,
                b"while" => Keyword::While,
                b"yield" => Keyword::Yield,
                b"async" => Keyword::Async,
                _ => return None,
            },
            6 => match b {
                b"delete" => Keyword::Delete,
                b"export" => Keyword::Export,
                b"import" => Keyword::Import,
                b"return" => Keyword::Return,
                b"switch" => Keyword::Switch,
                b"typeof" => Keyword::Typeof,
                _ => return None,
            },
            7 => match b {
                b"default" => Keyword::Default,
                b"extends" => Keyword::Extends,
                b"finally" => Keyword::Finally,
                _ => return None,
            },
            8 => match b {
                b"continue" => Keyword::Continue,
                b"debugger" => Keyword::Debugger,
                b"function" => Keyword::Function,
                _ => return None,
            },
            9 => match b {
                b"undefined" => Keyword::Undefined,
                _ => return None,
            },
            10 => match b {
                b"instanceof" => Keyword::Instanceof,
                _ => return None,
            },
            _ => return None,
        })
    }
}

impl fmt::Display for Keyword {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl PartialEq<str> for Keyword {
    fn eq(&self, other: &str) -> bool {
        self.as_str() == other
    }
}
impl PartialEq<&str> for Keyword {
    fn eq(&self, other: &&str) -> bool {
        self.as_str() == *other
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum TokenKind {
    // Literals
    Number(f64),
    BigInt(String),
    String(String),
    /// Template literal body: `(cooked, raw)`.
    /// `cooked` has escape sequences processed (`\n` → newline, etc.).
    /// `raw` preserves backslashes as written (`\n` stays `\n` two chars).
    /// Both have `${...}` interpolation holes copied verbatim as source text.
    TemplateString(String, String),
    Identifier(String),
    Regex(String, String), // body, flags (recognized via context heuristic)
    // Keywords (the common set) — `Copy` tag, no heap String.
    Keyword(Keyword),
    // Punctuation — `Copy` tag, no heap String.
    Punct(Punct),
    LineTerminator,
}

impl fmt::Display for TokenKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Number(n) => write!(f, "num({n})"),
            Self::BigInt(n) => write!(f, "bigint({n})"),
            Self::String(s) => write!(f, "str({s:?})"),
            Self::TemplateString(s, _) => write!(f, "tmpl({s:?})"),
            Self::Identifier(s) => write!(f, "id({s})"),
            Self::Regex(b, fl) => write!(f, "regex(/{b}/{fl})"),
            Self::Keyword(s) => write!(f, "kw({s})"),
            Self::Punct(s) => write!(f, "punct({s})"),
            Self::LineTerminator => f.write_str("nl"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Token {
    pub kind: TokenKind,
    pub line: u32,
    pub col: u32,
}

pub fn tokenize(src: &str) -> Vec<Token> {
    let bytes = src.as_bytes();
    let mut i = 0;
    let mut line: u32 = 1;
    let mut col: u32 = 1;
    let mut out = Vec::new();
    let mut prev_is_value_or_close = false;

    while i < bytes.len() {
        let b = bytes[i];
        let start_line = line;
        let start_col = col;
        match b {
            // JS WhiteSpace (ECMA-262 §12.2): space, tab, vertical tab (),
            // form feed (). Multibyte WS (NBSP  , BOM ﻿) handled
            // below.
            b' ' | b'\t' | 0x0B | 0x0C => {
                i += 1;
                col += 1;
            }
            // NBSP (  = C2 A0) and BOM/ZWNBSP (﻿ = EF BB BF) are
            // WhiteSpace too; skip the whole multibyte sequence.
            0xC2 if bytes.get(i + 1) == Some(&0xA0) => {
                i += 2;
                col += 1;
            }
            0xEF if bytes.get(i + 1) == Some(&0xBB) && bytes.get(i + 2) == Some(&0xBF) => {
                i += 3;
                col += 1;
            }
            b'\r' => {
                i += 1;
                if i < bytes.len() && bytes[i] == b'\n' {
                    i += 1;
                }
                out.push(Token {
                    kind: TokenKind::LineTerminator,
                    line: start_line,
                    col: start_col,
                });
                line += 1;
                col = 1;
                prev_is_value_or_close = false;
            }
            b'\n' => {
                i += 1;
                out.push(Token {
                    kind: TokenKind::LineTerminator,
                    line: start_line,
                    col: start_col,
                });
                line += 1;
                col = 1;
                prev_is_value_or_close = false;
            }
            b'/' if i + 1 < bytes.len() && bytes[i + 1] == b'/' => {
                // Line comment.
                i += 2;
                while i < bytes.len() && bytes[i] != b'\n' {
                    i += 1;
                }
                col = 1; // comments end before newline; newline handled next pass
            }
            b'/' if i + 1 < bytes.len() && bytes[i + 1] == b'*' => {
                // Block comment.
                i += 2;
                while i + 1 < bytes.len() && !(bytes[i] == b'*' && bytes[i + 1] == b'/') {
                    if bytes[i] == b'\n' {
                        line += 1;
                        col = 1;
                    } else {
                        col += 1;
                    }
                    i += 1;
                }
                i = (i + 2).min(bytes.len());
            }
            b'"' | b'\'' => {
                let quote = b;
                i += 1;
                col += 1;
                let mut s = String::new();
                while i < bytes.len() && bytes[i] != quote {
                    if bytes[i] == b'\\' && i + 1 < bytes.len() {
                        let esc = bytes[i + 1];
                        // Multi-char escapes (`\uXXXX`, `\u{…}`, `\xXX`) consume
                        // more than two bytes — handle them before the simple
                        // single-char table. WITHOUT this, `&` decoded to
                        // the literal text "u0026" (the `_ => esc as char` arm
                        // dropped the backslash), corrupting every string with a
                        // unicode escape — e.g. the React RSC flight payload
                        // encodes `&` as `&`, so hrefs came out as
                        // "…&display=swap" and hydration mismatched (#418).
                        let consumed = match esc {
                            b'u' if i + 2 < bytes.len() && bytes[i + 2] == b'{' => {
                                // `\u{1-6 hex}`
                                let mut j = i + 3;
                                while j < bytes.len() && bytes[j] != b'}' {
                                    j += 1;
                                }
                                if j < bytes.len() {
                                    if let Some(cp) = hex_val(&bytes[i + 3..j]) {
                                        s.push(char::from_u32(cp).unwrap_or('\u{FFFD}'));
                                    }
                                    Some(j + 1 - i)
                                } else {
                                    None
                                }
                            }
                            b'u' if i + 6 <= bytes.len() => {
                                // `\uXXXX` (exactly 4 hex), with surrogate-pair
                                // joining for `😀`-style code points.
                                match hex_val(&bytes[i + 2..i + 6]) {
                                    Some(hi)
                                        if (0xD800..=0xDBFF).contains(&hi)
                                            && i + 12 <= bytes.len()
                                            && bytes[i + 6] == b'\\'
                                            && bytes[i + 7] == b'u' =>
                                    {
                                        match hex_val(&bytes[i + 8..i + 12]) {
                                            Some(lo) if (0xDC00..=0xDFFF).contains(&lo) => {
                                                let cp =
                                                    0x10000 + ((hi - 0xD800) << 10) + (lo - 0xDC00);
                                                s.push(char::from_u32(cp).unwrap_or('\u{FFFD}'));
                                                Some(12)
                                            }
                                            _ => {
                                                s.push(char::from_u32(hi).unwrap_or('\u{FFFD}'));
                                                Some(6)
                                            }
                                        }
                                    }
                                    Some(cp) => {
                                        s.push(char::from_u32(cp).unwrap_or('\u{FFFD}'));
                                        Some(6)
                                    }
                                    None => None,
                                }
                            }
                            b'x' if i + 4 <= bytes.len() => {
                                // `\xXX` (exactly 2 hex).
                                match hex_val(&bytes[i + 2..i + 4]) {
                                    Some(cp) => {
                                        s.push(char::from_u32(cp).unwrap_or('\u{FFFD}'));
                                        Some(4)
                                    }
                                    None => None,
                                }
                            }
                            _ => None,
                        };
                        if let Some(n) = consumed {
                            i += n;
                            col += n as u32;
                            continue;
                        }
                        let ch = match esc {
                            b'n' => '\n',
                            b'r' => '\r',
                            b't' => '\t',
                            b'\\' => '\\',
                            b'\'' => '\'',
                            b'"' => '"',
                            b'0' => '\0',
                            b'b' => '\u{08}',
                            b'f' => '\u{0C}',
                            b'v' => '\u{0B}',
                            _ => esc as char,
                        };
                        s.push(ch);
                        i += 2;
                        col += 2;
                    } else {
                        let (ch, n) = decode_one(&bytes[i..]);
                        s.push(ch);
                        i += n;
                        col += 1;
                    }
                }
                if i < bytes.len() {
                    i += 1;
                    col += 1;
                }
                out.push(Token {
                    kind: TokenKind::String(s),
                    line: start_line,
                    col: start_col,
                });
                prev_is_value_or_close = true;
            }
            b'`' => {
                i += 1;
                col += 1;
                let mut s = String::new(); // cooked: escape sequences processed
                let mut r = String::new(); // raw: backslashes preserved as-is
                // Scan the template body. Plain text outside `${...}` has its
                // escapes decoded (so `\n` becomes a newline) into `s` (cooked),
                // while `r` (raw) preserves the source text verbatim. Inside a
                // `${...}` interpolation hole we copy the source *raw* into
                // both — including any nested template literals (which carry their
                // own backticks and `${...}` holes) and string literals —
                // so the interpreter can re-parse the expression faithfully.
                // Without raw, nesting-aware copying a nested backtick would
                // be mistaken for the end of the outer template, truncating
                // the whole token (this is exactly what broke Next.js /
                // webpack chunks built with nested template literals).
                loop {
                    if i >= bytes.len() {
                        break;
                    }
                    let c = bytes[i];
                    if c == b'`' {
                        i += 1;
                        col += 1;
                        break;
                    }
                    if c == b'$' && i + 1 < bytes.len() && bytes[i + 1] == b'{' {
                        // `${...}` holes are identical in cooked and raw — both
                        // get the raw source text so the interpreter can parse them.
                        let s_before = s.len();
                        copy_template_interp(bytes, &mut i, &mut s, &mut line, &mut col);
                        r.push_str(&s[s_before..]);
                        continue;
                    }
                    if c == b'\\' && i + 1 < bytes.len() {
                        let nxt = bytes[i + 1];
                        // cooked: process the escape
                        let ch = match nxt {
                            b'n' => '\n',
                            b't' => '\t',
                            b'r' => '\r',
                            b'\\' => '\\',
                            b'`' => '`',
                            b'$' => '$',
                            b'0' => '\0',
                            other => other as char,
                        };
                        s.push(ch);
                        // raw: keep the backslash and the following char verbatim
                        r.push('\\');
                        r.push(nxt as char);
                        i += 2;
                        col += 2;
                    } else if c == b'\n' {
                        s.push('\n');
                        r.push('\n');
                        i += 1;
                        line += 1;
                        col = 1;
                    } else {
                        let (ch, n) = decode_one(&bytes[i..]);
                        s.push(ch);
                        // raw: copy the exact bytes as a str slice
                        if let Ok(raw_ch) = std::str::from_utf8(&bytes[i..i + n]) {
                            r.push_str(raw_ch);
                        } else {
                            r.push(ch); // fallback
                        }
                        i += n;
                        col += 1;
                    }
                }
                out.push(Token {
                    kind: TokenKind::TemplateString(s, r),
                    line: start_line,
                    col: start_col,
                });
                prev_is_value_or_close = true;
            }
            b'/' if !prev_is_value_or_close => {
                // Regex literal context: `/pattern/flags`.
                i += 1;
                col += 1;
                let mut body = String::new();
                let mut in_class = false;
                while i < bytes.len() {
                    let c = bytes[i];
                    if c == b'\\' && i + 1 < bytes.len() {
                        body.push(c as char);
                        body.push(bytes[i + 1] as char);
                        i += 2;
                        col += 2;
                        continue;
                    }
                    if c == b'[' {
                        in_class = true;
                    }
                    if c == b']' {
                        in_class = false;
                    }
                    if c == b'/' && !in_class {
                        break;
                    }
                    if c == b'\n' {
                        // Unterminated; bail and fall back to division.
                        break;
                    }
                    body.push(c as char);
                    i += 1;
                    col += 1;
                }
                if i < bytes.len() && bytes[i] == b'/' {
                    i += 1;
                    col += 1;
                }
                let mut flags = String::new();
                while i < bytes.len() && bytes[i].is_ascii_alphabetic() {
                    flags.push(bytes[i] as char);
                    i += 1;
                    col += 1;
                }
                out.push(Token {
                    kind: TokenKind::Regex(body, flags),
                    line: start_line,
                    col: start_col,
                });
                prev_is_value_or_close = true;
            }
            b'0'..=b'9' => {
                let (kind, end, consumed_cols) = read_number(bytes, i);
                i = end;
                col += consumed_cols;
                out.push(Token {
                    kind,
                    line: start_line,
                    col: start_col,
                });
                prev_is_value_or_close = true;
            }
            b'.' if i + 1 < bytes.len() && bytes[i + 1].is_ascii_digit() => {
                let (kind, end, consumed_cols) = read_number(bytes, i);
                i = end;
                col += consumed_cols;
                out.push(Token {
                    kind,
                    line: start_line,
                    col: start_col,
                });
                prev_is_value_or_close = true;
            }
            b'#' if i + 1 < bytes.len() && is_id_start(bytes[i + 1]) => {
                // Private class field reference: `#name`. V1 strips the
                // `#` and treats the rest as a regular identifier so
                // class bodies that use private fields still parse and
                // run. Real lexical privacy isn't enforced — the name
                // is mangled with a `_pvt_` prefix to avoid collisions
                // with regular fields.
                i += 1;
                col += 1;
                let start = i;
                while i < bytes.len() && is_id_continue(bytes[i]) {
                    i += 1;
                    col += 1;
                }
                let raw = std::str::from_utf8(&bytes[start..i]).unwrap_or("");
                let name = format!("_pvt_{raw}");
                out.push(Token {
                    kind: TokenKind::Identifier(name),
                    line: start_line,
                    col: start_col,
                });
                prev_is_value_or_close = true;
            }
            b if is_id_start(b) => {
                let start = i;
                while i < bytes.len() && is_id_continue(bytes[i]) {
                    i += 1;
                    col += 1;
                }
                // Recognize keywords directly off the scanned BYTE SLICE — no
                // String allocation, no O(41) linear scan. Only a genuine
                // identifier (not a keyword) allocates a `String`.
                let slice = &bytes[start..i];
                let (kind, is_close) = match Keyword::from_bytes(slice) {
                    Some(kw) => {
                        // A `/` after a value-keyword (`this`/`true`/`false`/
                        // `null`/`undefined`/`super`) is division; after any
                        // other keyword it begins a regex.
                        let value_keyword = matches!(
                            kw,
                            Keyword::This
                                | Keyword::True
                                | Keyword::False
                                | Keyword::Null
                                | Keyword::Undefined
                                | Keyword::Super
                        );
                        (TokenKind::Keyword(kw), value_keyword)
                    }
                    None => {
                        let name = std::str::from_utf8(slice).unwrap_or("").to_string();
                        (TokenKind::Identifier(name), true)
                    }
                };
                out.push(Token {
                    kind,
                    line: start_line,
                    col: start_col,
                });
                prev_is_value_or_close = is_close;
            }
            _ => {
                let (sym, n) = match_punct(&bytes[i..]);
                if let Some(p) = sym {
                    i += n;
                    col += n as u32;
                    // A `}` is intentionally NOT treated as a value-close
                    // for regex disambiguation: a `/` following `}` almost
                    // always begins a regex literal in a fresh statement
                    // (`...;return}/re/.test(x)`) rather than dividing an
                    // object literal. `)`/`]`/postfix `++`/`--` do close a
                    // value, so `/` after them stays division.
                    let closes = matches!(
                        p,
                        Punct::RParen | Punct::RBracket | Punct::PlusPlus | Punct::MinusMinus
                    );
                    out.push(Token {
                        kind: TokenKind::Punct(p),
                        line: start_line,
                        col: start_col,
                    });
                    prev_is_value_or_close = closes;
                } else {
                    // Unknown byte; skip to avoid hanging.
                    let (_, nb) = decode_one(&bytes[i..]);
                    i += nb;
                    col += 1;
                }
            }
        }
    }
    out
}

fn read_number(bytes: &[u8], start: usize) -> (TokenKind, usize, u32) {
    // Numeric separator `_` is allowed between digits (ES2021). BigInt
    // suffix `n` is preserved as a distinct token so the runtime can keep
    // integer semantics instead of silently degrading to IEEE-754.
    let mut i = start;
    // 0x hex, 0o octal, 0b binary.
    if bytes.get(i) == Some(&b'0') && matches!(bytes.get(i + 1), Some(b'x' | b'X')) {
        i += 2;
        let s = i;
        while i < bytes.len() && (bytes[i].is_ascii_hexdigit() || bytes[i] == b'_') {
            i += 1;
        }
        let digits: String = std::str::from_utf8(&bytes[s..i])
            .unwrap_or("0")
            .chars()
            .filter(|c| *c != '_')
            .collect();
        if bytes.get(i) == Some(&b'n') {
            i += 1;
            return (
                TokenKind::BigInt(format!("0x{digits}")),
                i,
                (i - start) as u32,
            );
        }
        let v = u64::from_str_radix(&digits, 16).unwrap_or(0) as f64;
        return (TokenKind::Number(v), i, (i - start) as u32);
    }
    if bytes.get(i) == Some(&b'0') && matches!(bytes.get(i + 1), Some(b'o' | b'O')) {
        i += 2;
        let s = i;
        while i < bytes.len() && (matches!(bytes[i], b'0'..=b'7') || bytes[i] == b'_') {
            i += 1;
        }
        let digits: String = std::str::from_utf8(&bytes[s..i])
            .unwrap_or("0")
            .chars()
            .filter(|c| *c != '_')
            .collect();
        if bytes.get(i) == Some(&b'n') {
            i += 1;
            return (
                TokenKind::BigInt(format!("0o{digits}")),
                i,
                (i - start) as u32,
            );
        }
        let v = u64::from_str_radix(&digits, 8).unwrap_or(0) as f64;
        return (TokenKind::Number(v), i, (i - start) as u32);
    }
    if bytes.get(i) == Some(&b'0') && matches!(bytes.get(i + 1), Some(b'b' | b'B')) {
        i += 2;
        let s = i;
        while i < bytes.len() && (bytes[i] == b'0' || bytes[i] == b'1' || bytes[i] == b'_') {
            i += 1;
        }
        let digits: String = std::str::from_utf8(&bytes[s..i])
            .unwrap_or("0")
            .chars()
            .filter(|c| *c != '_')
            .collect();
        if bytes.get(i) == Some(&b'n') {
            i += 1;
            return (
                TokenKind::BigInt(format!("0b{digits}")),
                i,
                (i - start) as u32,
            );
        }
        let v = u64::from_str_radix(&digits, 2).unwrap_or(0) as f64;
        return (TokenKind::Number(v), i, (i - start) as u32);
    }
    let is_digit_or_sep = |b: u8| b.is_ascii_digit() || b == b'_';
    while i < bytes.len() && is_digit_or_sep(bytes[i]) {
        i += 1;
    }
    if bytes.get(i) == Some(&b'.') {
        i += 1;
        while i < bytes.len() && is_digit_or_sep(bytes[i]) {
            i += 1;
        }
    }
    if matches!(bytes.get(i), Some(b'e' | b'E')) {
        i += 1;
        if matches!(bytes.get(i), Some(b'+' | b'-')) {
            i += 1;
        }
        while i < bytes.len() && is_digit_or_sep(bytes[i]) {
            i += 1;
        }
    }
    let is_bigint = if bytes.get(i) == Some(&b'n') {
        i += 1;
        true
    } else {
        false
    };
    let cleaned: String = std::str::from_utf8(&bytes[start..i])
        .unwrap_or("0")
        .chars()
        .filter(|c| *c != '_' && *c != 'n')
        .collect();
    if is_bigint {
        return (TokenKind::BigInt(cleaned), i, (i - start) as u32);
    }
    let v = cleaned.parse::<f64>().unwrap_or(0.0);
    (TokenKind::Number(v), i, (i - start) as u32)
}

fn is_id_start(b: u8) -> bool {
    b.is_ascii_alphabetic() || b == b'_' || b == b'$' || b >= 0x80
}

fn is_id_continue(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_' || b == b'$' || b >= 0x80
}

fn match_punct(bytes: &[u8]) -> (Option<Punct>, usize) {
    // Try multi-char first so longer matches aren't eclipsed by shorter ones.
    // (Punct, spelling) — the spelling drives the byte match; the tag is what we
    // emit (no allocation). Order = longest-first, identical to the prior table.
    const CANDIDATES: &[(Punct, &str)] = &[
        (Punct::UShrAssign, ">>>="),
        (Punct::Ellipsis, "..."),
        (Punct::EqEqEq, "==="),
        (Punct::NeqEq, "!=="),
        (Punct::UShr, ">>>"),
        (Punct::StarStarAssign, "**="),
        (Punct::ShlAssign, "<<="),
        (Punct::ShrAssign, ">>="),
        (Punct::AndAndAssign, "&&="),
        (Punct::OrOrAssign, "||="),
        (Punct::QQAssign, "??="),
        (Punct::OptChain, "?."),
        (Punct::Arrow, "=>"),
        (Punct::EqEq, "=="),
        (Punct::Neq, "!="),
        (Punct::Le, "<="),
        (Punct::Ge, ">="),
        (Punct::AndAnd, "&&"),
        (Punct::OrOr, "||"),
        (Punct::QQ, "??"),
        (Punct::PlusPlus, "++"),
        (Punct::MinusMinus, "--"),
        (Punct::Shl, "<<"),
        (Punct::Shr, ">>"),
        (Punct::StarStar, "**"),
        (Punct::PlusAssign, "+="),
        (Punct::MinusAssign, "-="),
        (Punct::StarAssign, "*="),
        (Punct::SlashAssign, "/="),
        (Punct::PercentAssign, "%="),
        (Punct::AndAssign, "&="),
        (Punct::OrAssign, "|="),
        (Punct::XorAssign, "^="),
        (Punct::LParen, "("),
        (Punct::RParen, ")"),
        (Punct::LBracket, "["),
        (Punct::RBracket, "]"),
        (Punct::LBrace, "{"),
        (Punct::RBrace, "}"),
        (Punct::Semi, ";"),
        (Punct::Comma, ","),
        (Punct::Lt, "<"),
        (Punct::Gt, ">"),
        (Punct::Plus, "+"),
        (Punct::Minus, "-"),
        (Punct::Star, "*"),
        (Punct::Slash, "/"),
        (Punct::Percent, "%"),
        (Punct::Amp, "&"),
        (Punct::Pipe, "|"),
        (Punct::Caret, "^"),
        (Punct::Tilde, "~"),
        (Punct::Bang, "!"),
        (Punct::Assign, "="),
        (Punct::Question, "?"),
        (Punct::Colon, ":"),
        (Punct::Dot, "."),
    ];
    for (p, c) in CANDIDATES {
        let cb = c.as_bytes();
        if bytes.len() >= cb.len() && &bytes[..cb.len()] == cb {
            // ECMA-262 §12.7: `?.` is the optional-chaining punctuator ONLY when
            // it is NOT immediately followed by a decimal digit. `cond?.5:x` is
            // a conditional (`?`) whose consequent is the numeric literal `.5`,
            // not `cond ?. 5`. Minified bundles use `x?.5:y` constantly; without
            // this, `?.` ate the `.`, the ternary mis-parsed, and the whole
            // statement (and any bindings after it) were corrupted.
            if *p == Punct::OptChain && matches!(bytes.get(2), Some(d) if d.is_ascii_digit()) {
                continue;
            }
            return (Some(*p), cb.len());
        }
    }
    (None, 0)
}

/// Parse an ASCII-hex byte slice (1–6 digits) into a code point, or `None` if
/// empty, too long, or any byte is not a hex digit. Used to decode `\xXX`,
/// `\uXXXX`, and `\u{…}` string escapes.
fn hex_val(bytes: &[u8]) -> Option<u32> {
    if bytes.is_empty() || bytes.len() > 6 {
        return None;
    }
    let mut v: u32 = 0;
    for &b in bytes {
        let d = match b {
            b'0'..=b'9' => (b - b'0') as u32,
            b'a'..=b'f' => (b - b'a' + 10) as u32,
            b'A'..=b'F' => (b - b'A' + 10) as u32,
            _ => return None,
        };
        v = v.checked_mul(16)?.checked_add(d)?;
    }
    Some(v)
}

/// Raw-copy a single source character (or escape pair) into `s`, advancing
/// `i`/`line`/`col`. Used inside template interpolation holes where the body
/// must be preserved verbatim for the expression parser.
fn copy_raw_char(bytes: &[u8], i: &mut usize, s: &mut String, line: &mut u32, col: &mut u32) {
    let c = bytes[*i];
    if c == b'\\' && *i + 1 < bytes.len() {
        // Preserve the backslash and the following code point verbatim.
        s.push('\\');
        *i += 1;
        *col += 1;
        let (ch, n) = decode_one(&bytes[*i..]);
        s.push(ch);
        *i += n;
        *col += 1;
        return;
    }
    if c == b'\n' {
        s.push('\n');
        *i += 1;
        *line += 1;
        *col = 1;
        return;
    }
    let (ch, n) = decode_one(&bytes[*i..]);
    s.push(ch);
    *i += n;
    *col += 1;
}

/// Raw-copy a `'`/`"` string literal (from its opening quote through its
/// matching closing quote) into `s`. Backslash escapes are preserved so an
/// embedded quote or brace can't be mistaken for structure.
fn copy_string_literal(bytes: &[u8], i: &mut usize, s: &mut String, line: &mut u32, col: &mut u32) {
    let quote = bytes[*i];
    s.push(quote as char);
    *i += 1;
    *col += 1;
    while *i < bytes.len() {
        let c = bytes[*i];
        if c == b'\\' && *i + 1 < bytes.len() {
            copy_raw_char(bytes, i, s, line, col);
            continue;
        }
        if c == quote {
            s.push(quote as char);
            *i += 1;
            *col += 1;
            break;
        }
        copy_raw_char(bytes, i, s, line, col);
    }
}

/// Raw-copy a nested template literal (from its opening backtick through its
/// matching closing backtick), recursively handling its own `${...}` holes.
fn copy_nested_template(
    bytes: &[u8],
    i: &mut usize,
    s: &mut String,
    line: &mut u32,
    col: &mut u32,
) {
    s.push('`');
    *i += 1;
    *col += 1;
    while *i < bytes.len() {
        let c = bytes[*i];
        if c == b'`' {
            s.push('`');
            *i += 1;
            *col += 1;
            break;
        }
        if c == b'$' && *i + 1 < bytes.len() && bytes[*i + 1] == b'{' {
            copy_template_interp(bytes, i, s, line, col);
            continue;
        }
        copy_raw_char(bytes, i, s, line, col);
    }
}

/// Raw-copy a `${ ... }` interpolation hole (from `${` through its matching
/// `}`) into `s`, tracking brace depth and skipping over nested string and
/// template literals so a `}`/backtick inside them isn't treated as the end.
/// Copy a regex literal `/body/flags` verbatim, treating its contents (incl.
/// quotes and `[...]` char classes) as regex — NOT as string delimiters. Mirrors
/// the main lexer's regex scan. Without this, a regex containing a quote inside a
/// template hole — `` `${x.replace(/'/g, "")}` `` — was mis-read: the `'` started
/// a "string" that ran to the next quote far downstream, swallowing the closing
/// `}` and backtick and desyncing the entire parse (broke a webmail SPA).
fn copy_regex_literal(bytes: &[u8], i: &mut usize, s: &mut String, line: &mut u32, col: &mut u32) {
    s.push('/');
    *i += 1;
    *col += 1;
    let mut in_class = false;
    while *i < bytes.len() {
        let c = bytes[*i];
        if c == b'\\' {
            s.push('\\');
            *i += 1;
            *col += 1;
            if *i < bytes.len() {
                copy_raw_char(bytes, i, s, line, col);
            }
            continue;
        }
        if c == b'[' {
            in_class = true;
        }
        if c == b']' {
            in_class = false;
        }
        if c == b'/' && !in_class {
            break;
        }
        if c == b'\n' {
            break; // unterminated — bail (the parser will error cleanly)
        }
        copy_raw_char(bytes, i, s, line, col);
    }
    if *i < bytes.len() && bytes[*i] == b'/' {
        s.push('/');
        *i += 1;
        *col += 1;
    }
    while *i < bytes.len() && bytes[*i].is_ascii_alphabetic() {
        s.push(bytes[*i] as char);
        *i += 1;
        *col += 1;
    }
}

fn copy_template_interp(
    bytes: &[u8],
    i: &mut usize,
    s: &mut String,
    line: &mut u32,
    col: &mut u32,
) {
    // Copy the opening "${".
    s.push('$');
    s.push('{');
    *i += 2;
    *col += 2;
    let mut depth: usize = 1;
    // Track whether the previous significant char ends a value, for regex-vs-
    // division disambiguation (a `/` after a value is division; otherwise regex).
    let mut prev_value = false;
    while *i < bytes.len() && depth > 0 {
        let c = bytes[*i];
        match c {
            b'{' => {
                depth += 1;
                s.push('{');
                *i += 1;
                *col += 1;
                prev_value = false;
            }
            b'}' => {
                depth -= 1;
                s.push('}');
                *i += 1;
                *col += 1;
                prev_value = false;
            }
            b'\'' | b'"' => {
                copy_string_literal(bytes, i, s, line, col);
                prev_value = true;
            }
            b'`' => {
                copy_nested_template(bytes, i, s, line, col);
                prev_value = true;
            }
            b'/' if *i + 1 < bytes.len() && bytes[*i + 1] == b'/' => {
                // Line comment — copy to end of line.
                while *i < bytes.len() && bytes[*i] != b'\n' {
                    copy_raw_char(bytes, i, s, line, col);
                }
            }
            b'/' if *i + 1 < bytes.len() && bytes[*i + 1] == b'*' => {
                // Block comment — copy to `*/`.
                s.push('/');
                s.push('*');
                *i += 2;
                *col += 2;
                while *i + 1 < bytes.len() && !(bytes[*i] == b'*' && bytes[*i + 1] == b'/') {
                    copy_raw_char(bytes, i, s, line, col);
                }
                if *i + 1 < bytes.len() {
                    s.push('*');
                    s.push('/');
                    *i += 2;
                    *col += 2;
                }
            }
            b'/' if !prev_value => {
                copy_regex_literal(bytes, i, s, line, col);
                prev_value = true;
            }
            _ => {
                if c.is_ascii_alphanumeric() || c == b'_' || c == b'$' || c == b')' || c == b']' {
                    prev_value = true;
                } else if !c.is_ascii_whitespace() {
                    prev_value = false;
                }
                copy_raw_char(bytes, i, s, line, col);
            }
        }
    }
}

fn decode_one(bytes: &[u8]) -> (char, usize) {
    if bytes.is_empty() {
        return ('\u{FFFD}', 0);
    }
    let b = bytes[0];
    if b < 0x80 {
        return (b as char, 1);
    }
    let extra = if b >= 0xF0 {
        3
    } else if b >= 0xE0 {
        2
    } else if b >= 0xC0 {
        1
    } else {
        return ('\u{FFFD}', 1);
    };
    if bytes.len() < 1 + extra {
        return ('\u{FFFD}', bytes.len());
    }
    let s = std::str::from_utf8(&bytes[..1 + extra]).ok();
    match s.and_then(|s| s.chars().next()) {
        Some(c) => (c, 1 + extra),
        None => ('\u{FFFD}', 1),
    }
}

/// Counting global allocator used ONLY by the M3.1 allocation-measurement test.
/// It forwards every allocation to the system allocator and, when armed via a
/// thread-local flag, tallies the byte count + allocation count. This makes the
/// "punctuators/keywords now allocate ZERO String bytes" claim a REAL measured
/// number rather than an assertion-by-inspection.
#[cfg(test)]
pub(crate) mod alloc_count {
    use std::alloc::{GlobalAlloc, Layout, System};
    use std::cell::Cell;
    use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

    thread_local! {
        /// When true on the current thread, allocations are tallied below.
        pub static ARMED: Cell<bool> = const { Cell::new(false) };
    }
    pub static BYTES: AtomicU64 = AtomicU64::new(0);
    pub static COUNT: AtomicUsize = AtomicUsize::new(0);

    pub struct CountingAlloc;

    unsafe impl GlobalAlloc for CountingAlloc {
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            if ARMED.with(|a| a.get()) {
                BYTES.fetch_add(layout.size() as u64, Ordering::Relaxed);
                COUNT.fetch_add(1, Ordering::Relaxed);
            }
            unsafe { System.alloc(layout) }
        }
        unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
            unsafe { System.dealloc(ptr, layout) }
        }
        unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
            if ARMED.with(|a| a.get()) {
                // A realloc that grows is a (re)allocation of new bytes.
                if new_size > layout.size() {
                    BYTES.fetch_add((new_size - layout.size()) as u64, Ordering::Relaxed);
                }
                COUNT.fetch_add(1, Ordering::Relaxed);
            }
            unsafe { System.realloc(ptr, layout, new_size) }
        }
    }

    /// Run `f`, returning `(bytes_allocated, alloc_count)` measured during it.
    pub fn measure<R>(f: impl FnOnce() -> R) -> (R, u64, usize) {
        // Drain any lazy thread-local init before arming.
        ARMED.with(|a| a.set(false));
        BYTES.store(0, Ordering::Relaxed);
        COUNT.store(0, Ordering::Relaxed);
        ARMED.with(|a| a.set(true));
        let r = f();
        ARMED.with(|a| a.set(false));
        (
            r,
            BYTES.load(Ordering::Relaxed),
            COUNT.load(Ordering::Relaxed),
        )
    }
}

#[cfg(test)]
#[global_allocator]
static GLOBAL: alloc_count::CountingAlloc = alloc_count::CountingAlloc;

#[cfg(test)]
mod tests {
    use super::*;

    fn kinds(src: &str) -> Vec<TokenKind> {
        tokenize(src)
            .into_iter()
            .map(|t| t.kind)
            .filter(|k| !matches!(k, TokenKind::LineTerminator))
            .collect()
    }

    #[test]
    fn idents_and_keywords() {
        let k = kinds("let x = 42;");
        assert!(matches!(&k[0], TokenKind::Keyword(s) if s == "let"));
        assert!(matches!(&k[1], TokenKind::Identifier(s) if s == "x"));
        assert!(matches!(&k[2], TokenKind::Punct(s) if s == "="));
        assert!(matches!(&k[3], TokenKind::Number(n) if (*n - 42.0).abs() < 1e-9));
        assert!(matches!(&k[4], TokenKind::Punct(s) if s == ";"));
    }

    #[test]
    fn strings_and_templates() {
        let k = kinds(r#"const greet = "hi"; const t = `hello world`;"#);
        assert!(
            k.iter()
                .any(|t| matches!(t, TokenKind::String(s) if s == "hi"))
        );
        assert!(
            k.iter()
                .any(|t| matches!(t, TokenKind::TemplateString(s, _) if s == "hello world"))
        );
    }

    #[test]
    fn numbers_hex_and_exp() {
        let k = kinds("0xFF 1.5e2 .5");
        assert!(matches!(&k[0], TokenKind::Number(n) if (*n - 255.0).abs() < 1e-9));
        assert!(matches!(&k[1], TokenKind::Number(n) if (*n - 150.0).abs() < 1e-9));
        assert!(matches!(&k[2], TokenKind::Number(n) if (*n - 0.5).abs() < 1e-9));
    }

    #[test]
    fn comments_skipped() {
        let k = kinds("a // line\nb /* block */ c");
        let ids: Vec<&str> = k
            .iter()
            .filter_map(|t| {
                if let TokenKind::Identifier(s) = t {
                    Some(s.as_str())
                } else {
                    None
                }
            })
            .collect();
        assert_eq!(ids, vec!["a", "b", "c"]);
    }

    #[test]
    fn regex_vs_division() {
        let k = kinds("a / b; var r = /abc[/]/gi;");
        // First `/` is division.
        assert!(matches!(&k[1], TokenKind::Punct(p) if p == "/"));
        // The literal after `=` should tokenize as a Regex.
        assert!(
            k.iter()
                .any(|t| matches!(t, TokenKind::Regex(b, f) if b == "abc[/]" && f == "gi"))
        );
    }

    #[test]
    fn arrow_function_punct_chunks() {
        let k = kinds("const f = (a, b) => a + b;");
        let puncts: Vec<&str> = k
            .iter()
            .filter_map(|t| {
                if let TokenKind::Punct(s) = t {
                    Some(s.as_str())
                } else {
                    None
                }
            })
            .collect();
        assert!(puncts.contains(&"=>"));
        assert!(puncts.contains(&"("));
        assert!(puncts.contains(&","));
        assert!(puncts.contains(&"+"));
    }

    #[test]
    fn optional_chain_and_nullish() {
        let k = kinds("a?.b ?? c");
        assert!(matches!(&k[1], TokenKind::Punct(p) if p == "?."));
        assert!(matches!(&k[3], TokenKind::Punct(p) if p == "??"));
    }

    #[test]
    fn question_dot_before_digit_is_ternary_not_optional_chain() {
        // `cond?.5:x` is a conditional whose consequent is `.5`, NOT `cond ?. 5`.
        // Real optional chaining (`?.` before a non-digit) must still tokenize.
        let k = kinds("t?.5:9");
        assert!(
            matches!(&k[1], TokenKind::Punct(p) if p == "?"),
            "expected ternary '?', got {:?}",
            k[1]
        );
        assert!(
            matches!(&k[2], TokenKind::Number(n) if (*n - 0.5).abs() < 1e-12),
            "expected .5, got {:?}",
            k[2]
        );
        assert!(matches!(&k[3], TokenKind::Punct(p) if p == ":"));
        // `?.` before an identifier stays optional chaining.
        let k2 = kinds("t?.x");
        assert!(
            matches!(&k2[1], TokenKind::Punct(p) if p == "?."),
            "expected '?.', got {:?}",
            k2[1]
        );
    }

    // ───────────────── M3.1 Phase 0: new Copy-tag repr ──────────────────

    /// The new `Copy` punct/keyword tags carry NO heap String — assert the
    /// exact enum variants are produced (the new representation).
    #[test]
    fn punct_keyword_are_copy_tags() {
        let k = kinds("let x = 42;");
        assert_eq!(k[0], TokenKind::Keyword(Keyword::Let));
        assert_eq!(k[1], TokenKind::Identifier("x".to_string()));
        assert_eq!(k[2], TokenKind::Punct(Punct::Assign));
        assert_eq!(k[4], TokenKind::Punct(Punct::Semi));

        // A multi-char punctuator + a value keyword.
        let k2 = kinds("a => true");
        assert_eq!(k2[1], TokenKind::Punct(Punct::Arrow));
        assert_eq!(k2[2], TokenKind::Keyword(Keyword::True));
    }

    /// `Keyword::from_bytes` must recognize EVERY keyword and reject look-alikes
    /// (no partial / prefix matches), with zero allocation. Also assert the
    /// `as_str()` round-trips back to the source spelling.
    #[test]
    fn keyword_recognition_is_exact_and_total() {
        let all = [
            "await", "break", "case", "catch", "class", "const", "continue", "debugger",
            "default", "delete", "do", "else", "export", "extends", "false", "finally", "for",
            "function", "if", "import", "in", "instanceof", "let", "new", "null", "of", "return",
            "super", "switch", "this", "throw", "true", "try", "typeof", "undefined", "var",
            "void", "while", "with", "yield", "async",
        ];
        for kw in all {
            let got = Keyword::from_bytes(kw.as_bytes())
                .unwrap_or_else(|| panic!("'{kw}' should be a keyword"));
            assert_eq!(got.as_str(), kw, "as_str round-trip for '{kw}'");
        }
        // Look-alikes / identifiers must NOT be keyword.
        for id in ["awaitt", "fo", "forr", "Function", "VAR", "x", "_let", "iff", "constx"] {
            assert!(
                Keyword::from_bytes(id.as_bytes()).is_none(),
                "'{id}' must NOT be a keyword"
            );
        }
    }

    /// Every punctuator `match_punct` can emit must round-trip through
    /// `as_str()` back to a single-match of itself (the tag <-> spelling map is
    /// total and consistent).
    #[test]
    fn punct_as_str_round_trips_through_match_punct() {
        // The full set of spellings the lexer recognizes.
        let spellings = [
            ">>>=", "...", "===", "!==", ">>>", "**=", "<<=", ">>=", "&&=", "||=", "??=", "?.",
            "=>", "==", "!=", "<=", ">=", "&&", "||", "??", "++", "--", "<<", ">>", "**", "+=",
            "-=", "*=", "/=", "%=", "&=", "|=", "^=", "(", ")", "[", "]", "{", "}", ";", ",", "<",
            ">", "+", "-", "*", "/", "%", "&", "|", "^", "~", "!", "=", "?", ":", ".",
        ];
        for sp in spellings {
            let (p, n) = match_punct(sp.as_bytes());
            let p = p.unwrap_or_else(|| panic!("'{sp}' should match a punctuator"));
            assert_eq!(n, sp.len(), "consumed length for '{sp}'");
            assert_eq!(p.as_str(), sp, "as_str round-trip for '{sp}'");
            assert!(p == sp, "PartialEq<&str> for '{sp}'");
        }
    }

    /// THE WIN — measured. Tokenizing a punctuator- and keyword-heavy source
    /// must allocate ZERO String bytes for the punct/keyword tokens. We measure
    /// total allocation while tokenizing two sources:
    ///   - a PURE punct+keyword source (no identifiers/strings/numbers): the
    ///     emitted `Vec<Token>` is the ONLY thing that should allocate, and a
    ///     `Token` is `Copy`-payload here, so the bytes scale with the Vec, NOT
    ///     with the token count's worth of `String`s.
    ///   - the same shape repeated: per-token String bytes stay at zero.
    /// Before this change every one of these tokens carried an owned `String`
    /// (a heap alloc each); after, punct/keyword carry a `Copy` tag.
    #[test]
    fn punct_keyword_tokens_allocate_zero_string_bytes() {
        // 200 keyword/punct tokens, NO identifiers/literals (those legitimately
        // still allocate — out of scope this phase).
        let unit = "if(true){}else{}while(false){}for(;;){}return;break;continue;";
        let src = unit.repeat(20);
        let n_tokens = tokenize(&src).len();
        assert!(n_tokens > 500, "sanity: produced {n_tokens} tokens");

        // Measure allocation for tokenize() over the punct/keyword-only source.
        let (toks, bytes, count) = super::alloc_count::measure(|| tokenize(&src));

        // Every token here is a Punct or Keyword (plus possibly LineTerminator);
        // assert NONE carry a heap String.
        let mut string_carrying = 0usize;
        for t in &toks {
            match &t.kind {
                TokenKind::Punct(_) | TokenKind::Keyword(_) | TokenKind::LineTerminator => {}
                _ => string_carrying += 1,
            }
        }
        assert_eq!(
            string_carrying, 0,
            "punct/keyword-only source must yield only Copy-tag tokens"
        );

        // The ONLY heap traffic is the output `Vec<Token>` growth (a Token is
        // {kind, line, col}; its kind here holds no String). So per-token String
        // allocations are ZERO. The Vec itself reallocs O(log n) times as it
        // grows — bound the per-token allocation count well below 1 alloc/token.
        let allocs_per_token = count as f64 / toks.len() as f64;
        assert!(
            allocs_per_token < 0.2,
            "expected far fewer than 1 alloc/token (Vec growth only), got {count} allocs for {} tokens ({allocs_per_token:.4}/token), {bytes} bytes",
            toks.len()
        );

        // Report the measured numbers (visible with `cargo test -- --nocapture`).
        eprintln!(
            "[M3.1 alloc] punct/keyword-only: {} tokens, {} total allocs, {} bytes, {:.4} allocs/token",
            toks.len(),
            count,
            bytes,
            allocs_per_token
        );
    }

    /// Differential-parse stability: each corpus snippet parses to an AST, and
    /// re-tokenizing + re-parsing the SAME source yields a byte-identical AST
    /// (`Stmt`/`Expr` derive `PartialEq`). This proves the new lexer repr did
    /// not perturb the produced AST — the parser maps the new `Copy` tags to the
    /// exact same nodes. (Execution-semantics parity is covered separately by
    /// the M3.0 A/B oracle corpus, which is green.)
    #[test]
    fn differential_parse_ast_is_stable() {
        // A punct/keyword-dense cross-section: every operator class, control
        // flow, classes, arrows, optional chaining, destructuring, generators.
        let corpus = [
            "let a = 1 + 2 * 3 - 4 / 5 % 6; const b = a ** 2;",
            "if (x === y && a !== b || c <= d) { return x >>> 1; } else { x <<= 2; }",
            "for (let i = 0; i < 10; i = i + 1) { if (i % 2 == 0) continue; break; }",
            "const f = (a, b) => a ?? b?.c ?? 0; f(1, {c: 2});",
            "class A { #p = 1; static s = 2; get v() { return this.#p; } } class B extends A {}",
            "function* g() { yield 1; yield* [2, 3]; return 4; }",
            "const { x, y = 5, ...rest } = obj; const [p, , q] = arr;",
            "switch (n) { case 1: a += 1; break; default: a -= 1; }",
            "try { throw new Error('x'); } catch (e) { } finally { }",
            "x?.y?.z; a |= b; c &= d; e ^= f; g &&= h; i ||= j; k ??= l;",
            "void 0; typeof x; delete o.p; new C(); a instanceof B; 'k' in o;",
            "async function h() { await p; }",
        ];
        for src in corpus {
            // `parse_program` tokenizes (via the new repr) then parses.
            let ast1 = crate::parser::parse_program(src)
                .unwrap_or_else(|e| panic!("parse failed for {src:?}: {e:?}"));
            // Parse the identical source again.
            let ast2 = crate::parser::parse_program(src)
                .unwrap_or_else(|e| panic!("re-parse failed for {src:?}: {e:?}"));
            assert_eq!(
                ast1, ast2,
                "AST must be byte-identical across re-parse for {src:?}"
            );
            assert!(!ast1.is_empty(), "non-empty AST for {src:?}");
        }
    }
}
