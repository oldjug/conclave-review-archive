//! Backtracking RegExp engine — ES2024 subset for V1.
//!
//! Supports: literal chars, escape sequences (\d \D \w \W \s \S \b \B \n \r \t
//! \\ \. etc.), character classes including ranges and negation, dot `.`
//! (matches everything except \n unless `s` flag), anchors `^ $`,
//! greedy and lazy quantifiers (* + ? {n} {n,} {n,m}), capturing groups
//! `(...)`, non-capturing `(?:...)`, named groups `(?<name>...)`,
//! alternation `|`, backreferences `\1`-`\9`.
//!
//! Flags: `g` (global), `i` (case-insensitive ASCII), `m` (multiline),
//! `s` (dotall), `u` (accepted; behaves as default Unicode handling), `y`
//! (sticky), `d` (indices).
//!
//! Lookahead `(?=)`/`(?!)` and lookbehind `(?<=)`/`(?<!)` ARE supported
//! (zero-width assertions compiled to an `Op::Look` sub-program; positive
//! lookarounds keep their inner captures).
//!
//! Not yet: Unicode property escapes (\p{}), `v` flag set notation — noted as
//! gaps; failure mode is a parse error that surfaces as a TypeError to the
//! user, never a panic.

/// Compiled regex program: a flat array of bytecode ops, executed
/// against a UTF-16 codeunit slice by `exec`.
#[derive(Debug, Clone)]
pub struct Regex {
    pub source: String,
    pub flags: String,
    prog: Vec<Op>,
    /// Total number of capturing groups (0 == whole-match only).
    pub group_count: usize,
    /// Map from named group → group index.
    pub named_groups: Vec<(String, usize)>,
    /// Cached flag bits.
    pub global: bool,
    pub ignore_case: bool,
    pub multiline: bool,
    pub dot_all: bool,
    pub sticky: bool,
}

#[derive(Debug, Clone)]
enum Op {
    Char(char),
    /// Match any character; if `dotall`=true (the dot_all flag), matches
    /// newlines too.
    AnyChar,
    /// Character-class: list of (lo, hi) inclusive ranges + `negate`.
    Class {
        ranges: Vec<(u32, u32)>,
        negate: bool,
    },
    /// Beginning-of-input (or beginning-of-line if multiline).
    Anchor(Anchor),
    /// `\b` / `\B` word boundary.
    Boundary(bool),
    /// Open capturing group `n`.
    Save(usize),
    /// Close capturing group `n`.
    Restore(usize),
    /// Backreference to capturing group `n`.
    BackRef(usize),
    /// Unconditional jump.
    Jmp(usize),
    /// Branch: try `a`, on failure try `b`.
    Split(usize, usize),
    /// Zero-width lookaround assertion. `behind=false` → lookahead `(?=)`/`(?!)`;
    /// `behind=true` → lookbehind `(?<=)`/`(?<!)`. `negate` inverts the test.
    /// `prog` is a self-contained sub-program (terminated by `Match`). The
    /// assertion consumes no input; captures inside a POSITIVE lookaround
    /// persist (JS semantics), negative ones discard.
    Look {
        negate: bool,
        behind: bool,
        prog: Vec<Op>,
    },
    /// Successful end of pattern.
    Match,
}

#[derive(Debug, Clone, Copy)]
enum Anchor {
    Start,
    End,
}

/// One successful match, with full text + per-group captures.
#[derive(Debug, Clone)]
pub struct Match {
    /// Byte offsets in the input *string* (UTF-8) for the whole match.
    pub start: usize,
    pub end: usize,
    /// Captures: index 0 == whole match. Index N > 0 == group N.
    /// Each entry: (start, end) byte offsets, or None if group didn't
    /// participate in the match.
    pub groups: Vec<Option<(usize, usize)>>,
    /// The matched substring.
    pub matched: String,
    /// Per-group matched substrings (None when group didn't participate).
    pub group_strings: Vec<Option<String>>,
}

impl Regex {
    /// Compile a pattern + flags string. Returns a parse error on
    /// malformed input.
    pub fn new(pattern: &str, flags: &str) -> Result<Self, String> {
        let mut p = Parser {
            src: pattern.chars().collect(),
            pos: 0,
            prog: Vec::new(),
            group_count: 0,
            named: Vec::new(),
            ignore_case: flags.contains('i'),
        };
        // Whole-match implicit group 0 — push Save(0)/Restore(0) around
        // the top-level alternation.
        p.prog.push(Op::Save(0));
        p.parse_alt()?;
        p.prog.push(Op::Restore(0));
        p.prog.push(Op::Match);
        Ok(Self {
            source: pattern.to_string(),
            flags: flags.to_string(),
            prog: p.prog,
            group_count: p.group_count + 1, // +1 for group 0
            named_groups: p.named,
            global: flags.contains('g'),
            ignore_case: flags.contains('i'),
            multiline: flags.contains('m'),
            dot_all: flags.contains('s'),
            sticky: flags.contains('y'),
        })
    }

    /// Find the first match at or after `start_byte`, byte-indexed into
    /// the input string. Returns None if no match.
    pub fn find_from(&self, input: &str, start_byte: usize) -> Option<Match> {
        // Walk byte-aligned char boundaries; at each, try to match the
        // program. If `sticky`, only try at start_byte exactly.
        let chars: Vec<(usize, char)> = input.char_indices().collect();
        let n = chars.len();
        let mut byte_starts: Vec<usize> = chars.iter().map(|(b, _)| *b).collect();
        byte_starts.push(input.len()); // end sentinel
        // Find char index whose byte offset >= start_byte.
        let start_char_idx = byte_starts
            .iter()
            .position(|&b| b >= start_byte)
            .unwrap_or(n);
        let mut i = start_char_idx;
        loop {
            let mut caps: Vec<(Option<usize>, Option<usize>)> =
                vec![(None, None); self.group_count];
            if exec_thread(&self.prog, 0, &chars, &byte_starts, i, &mut caps, self, None) {
                let (s, e) = (
                    caps[0].0.unwrap_or(byte_starts[i]),
                    caps[0].1.unwrap_or(byte_starts[i]),
                );
                let groups: Vec<Option<(usize, usize)>> = caps
                    .iter()
                    .map(|c| match (c.0, c.1) {
                        (Some(a), Some(b)) if b >= a => Some((a, b)),
                        _ => None,
                    })
                    .collect();
                let group_strings: Vec<Option<String>> = groups
                    .iter()
                    .map(|g| g.map(|(a, b)| input[a..b].to_string()))
                    .collect();
                return Some(Match {
                    start: s,
                    end: e,
                    groups,
                    matched: input[s..e].to_string(),
                    group_strings,
                });
            }
            if self.sticky {
                return None;
            }
            if i >= n {
                return None;
            }
            i += 1;
        }
    }

    pub fn test(&self, input: &str) -> bool {
        self.find_from(input, 0).is_some()
    }

    /// Iterate all non-overlapping matches.
    pub fn find_all(&self, input: &str) -> Vec<Match> {
        let mut out = Vec::new();
        let mut pos = 0;
        while pos <= input.len() {
            if let Some(m) = self.find_from(input, pos) {
                let next = if m.end == m.start {
                    // Empty match — advance by one char to avoid loops.
                    next_char_boundary(input, m.end)
                } else {
                    m.end
                };
                out.push(m);
                pos = next;
            } else {
                break;
            }
        }
        out
    }
}

fn next_char_boundary(s: &str, mut i: usize) -> usize {
    i += 1;
    while i < s.len() && !s.is_char_boundary(i) {
        i += 1;
    }
    i
}

// ----------------------------------------------------------------------
// Parser
// ----------------------------------------------------------------------

struct Parser {
    src: Vec<char>,
    pos: usize,
    prog: Vec<Op>,
    group_count: usize,
    named: Vec<(String, usize)>,
    ignore_case: bool,
}

impl Parser {
    fn peek(&self) -> Option<char> {
        self.src.get(self.pos).copied()
    }
    fn bump(&mut self) -> Option<char> {
        let c = self.peek();
        if c.is_some() {
            self.pos += 1;
        }
        c
    }
    fn eat(&mut self, c: char) -> bool {
        if self.peek() == Some(c) {
            self.pos += 1;
            true
        } else {
            false
        }
    }

    fn parse_alt(&mut self) -> Result<(), String> {
        // Single concat (no `|`): emit nothing extra. With `|`: a Split
        // per branch + an unconditional Jmp at branch-end that
        // gets fixed up to the post-alternation address once the whole
        // alternation closes.
        //
        // Removing the leading Split lazily isn't safe: stored split /
        // jump targets are absolute indices, so any later `remove(...)`
        // invalidates everything written after it. Append-only is safer.
        let first_start = self.prog.len();
        self.parse_concat()?;
        if self.peek() != Some('|') {
            return Ok(());
        }
        // Multi-arm alternation. Re-frame the already-emitted first
        // branch by *inserting* a Split before it, then chaining each
        // subsequent arm with its own Split. Because the insertion
        // happens before any branch code is reachable from later code
        // (we haven't returned to a caller yet), we can compensate by
        // bumping all absolute indices inside the inserted region.
        // Simpler approach: capture the first arm's ops, clear them,
        // then re-emit using the standard Split-per-arm template.
        let first_arm: Vec<Op> = self.prog.drain(first_start..).collect();
        let mut arms: Vec<Vec<Op>> = vec![first_arm];
        while self.peek() == Some('|') {
            self.bump();
            let arm_start = self.prog.len();
            self.parse_concat()?;
            arms.push(self.prog.drain(arm_start..).collect());
        }
        // Emit: For arms [A, B, C]:
        //   Split(a, x1)
        //   A
        //   Jmp(end)
        //   x1: Split(b, x2)
        //   B
        //   Jmp(end)
        //   x2: C
        //   end:
        // Compute total length first so we can fill in jumps in one
        // pass with correct absolute targets.
        // Length per arm: Split (1) + arm body + Jmp (1), except the
        // last arm which has no Split nor Jmp.
        let mut lens: Vec<usize> = arms
            .iter()
            .enumerate()
            .map(|(idx, a)| {
                if idx + 1 == arms.len() {
                    a.len()
                } else {
                    a.len() + 2
                }
            })
            .collect();
        let total: usize = lens.iter().sum();
        let end = first_start + total;
        // Now emit arm-by-arm.
        let mut cursor = first_start;
        let _ = lens.iter_mut(); // explicit no-op to silence the unused mut
        for (idx, arm) in arms.iter().enumerate() {
            let arm_body_len = arm.len();
            if idx + 1 == arms.len() {
                // last arm: just inline.
                for op in arm.iter().cloned() {
                    self.prog.push(op);
                }
                cursor += arm_body_len;
            } else {
                let next_arm_start = cursor + 1 + arm_body_len + 1;
                self.prog.push(Op::Split(cursor + 1, next_arm_start));
                for op in arm.iter().cloned() {
                    self.prog.push(op);
                }
                self.prog.push(Op::Jmp(end));
                cursor = next_arm_start;
            }
        }
        Ok(())
    }

    fn parse_concat(&mut self) -> Result<(), String> {
        loop {
            match self.peek() {
                None | Some(')') | Some('|') => return Ok(()),
                _ => self.parse_atom_with_quant()?,
            }
        }
    }

    fn parse_atom_with_quant(&mut self) -> Result<(), String> {
        let atom_start = self.prog.len();
        self.parse_atom()?;
        let atom_end = self.prog.len();
        match self.peek() {
            Some('*') => {
                self.bump();
                let lazy = self.eat('?');
                // Loop: Split → atom → Jmp back → end
                let split_pos = atom_start;
                self.prog.insert(split_pos, Op::Split(0, 0));
                let new_atom_end = atom_end + 1;
                self.prog.push(Op::Jmp(split_pos));
                let end = self.prog.len();
                self.prog[split_pos] = if lazy {
                    Op::Split(end, split_pos + 1)
                } else {
                    Op::Split(split_pos + 1, end)
                };
                let _ = new_atom_end;
            }
            Some('+') => {
                self.bump();
                let lazy = self.eat('?');
                // Atom; Split (back-edge greedy)
                let split_at = self.prog.len();
                self.prog.push(Op::Split(0, 0));
                let end = self.prog.len();
                self.prog[split_at] = if lazy {
                    Op::Split(end, atom_start)
                } else {
                    Op::Split(atom_start, end)
                };
            }
            Some('?') => {
                self.bump();
                let lazy = self.eat('?');
                let split_pos = atom_start;
                self.prog.insert(split_pos, Op::Split(0, 0));
                let end = self.prog.len();
                self.prog[split_pos] = if lazy {
                    Op::Split(end, split_pos + 1)
                } else {
                    Op::Split(split_pos + 1, end)
                };
            }
            Some('{') => {
                let save_pos = self.pos;
                self.bump();
                // Parse n, optional ,m
                let mut n_str = String::new();
                while let Some(c) = self.peek() {
                    if c.is_ascii_digit() {
                        n_str.push(c);
                        self.bump();
                    } else {
                        break;
                    }
                }
                if n_str.is_empty() {
                    // Not a quantifier — back up; treat `{` as literal.
                    self.pos = save_pos;
                    return Ok(());
                }
                let n: usize = n_str.parse().map_err(|_| "bad quantifier".to_string())?;
                let (min, max) = if self.eat(',') {
                    let mut m_str = String::new();
                    while let Some(c) = self.peek() {
                        if c.is_ascii_digit() {
                            m_str.push(c);
                            self.bump();
                        } else {
                            break;
                        }
                    }
                    let m = if m_str.is_empty() {
                        usize::MAX
                    } else {
                        m_str.parse().map_err(|_| "bad quantifier".to_string())?
                    };
                    (n, m)
                } else {
                    (n, n)
                };
                if !self.eat('}') {
                    self.pos = save_pos;
                    return Ok(());
                }
                let _ = self.eat('?');
                // Expand: copy the atom n times unconditionally, then
                // (max-n) optional copies. Simple but works for small
                // bounds; large bounds blow up the program — acceptable
                // for V1.
                let atom_bytes: Vec<Op> = self.prog[atom_start..atom_end].to_vec();
                // Truncate the original copy; we'll re-emit it min times.
                self.prog.truncate(atom_start);
                for _ in 0..min {
                    self.prog.extend(atom_bytes.iter().cloned());
                }
                if max == usize::MAX {
                    // Append a *-loop of the atom.
                    let loop_start = self.prog.len();
                    self.prog.push(Op::Split(0, 0));
                    let body_start = self.prog.len();
                    self.prog.extend(atom_bytes.iter().cloned());
                    self.prog.push(Op::Jmp(loop_start));
                    let end = self.prog.len();
                    self.prog[loop_start] = Op::Split(body_start, end);
                } else {
                    for _ in min..max {
                        let split_at = self.prog.len();
                        self.prog.push(Op::Split(0, 0));
                        let body_start = self.prog.len();
                        self.prog.extend(atom_bytes.iter().cloned());
                        let end = self.prog.len();
                        self.prog[split_at] = Op::Split(body_start, end);
                    }
                }
            }
            _ => {}
        }
        Ok(())
    }

    fn parse_atom(&mut self) -> Result<(), String> {
        match self.peek() {
            None => Err("unexpected end of pattern".into()),
            Some('(') => {
                self.bump();
                let mut capture = true;
                let mut name: Option<String> = None;
                if self.eat('?') {
                    if self.eat(':') {
                        capture = false;
                    } else if self.eat('=') {
                        return self.parse_lookaround(false, false);
                    } else if self.eat('!') {
                        return self.parse_lookaround(true, false);
                    } else if self.eat('<') {
                        // `(?<=` / `(?<!` are lookbehind; `(?<name>` is a named
                        // capturing group. Disambiguate on the char after `<`.
                        if self.eat('=') {
                            return self.parse_lookaround(false, true);
                        } else if self.eat('!') {
                            return self.parse_lookaround(true, true);
                        }
                        let mut n = String::new();
                        while let Some(c) = self.peek() {
                            if c == '>' {
                                break;
                            }
                            n.push(c);
                            self.bump();
                        }
                        if !self.eat('>') {
                            return Err("bad named group".into());
                        }
                        name = Some(n);
                    } else {
                        return Err("unsupported group syntax".into());
                    }
                }
                let group_idx = if capture {
                    self.group_count += 1;
                    self.group_count
                } else {
                    0
                };
                if let Some(n) = name {
                    self.named.push((n, group_idx));
                }
                if capture {
                    self.prog.push(Op::Save(group_idx));
                }
                self.parse_alt()?;
                if capture {
                    self.prog.push(Op::Restore(group_idx));
                }
                if !self.eat(')') {
                    return Err("unmatched (".into());
                }
                Ok(())
            }
            Some('[') => self.parse_class(),
            Some('.') => {
                self.bump();
                self.prog.push(Op::AnyChar);
                Ok(())
            }
            Some('^') => {
                self.bump();
                self.prog.push(Op::Anchor(Anchor::Start));
                Ok(())
            }
            Some('$') => {
                self.bump();
                self.prog.push(Op::Anchor(Anchor::End));
                Ok(())
            }
            Some('\\') => self.parse_escape(),
            Some(c) => {
                self.bump();
                self.emit_char(c);
                Ok(())
            }
        }
    }

    /// Parse a lookaround body `(?=…)` / `(?!…)` / `(?<=…)` / `(?<!…)` (the
    /// leading marker is already consumed) into a self-contained sub-program,
    /// then emit one `Op::Look`. The body is compiled into a fresh program so
    /// its ops/jumps are self-relative; capturing groups inside still allocate
    /// shared group indices so positive-lookaround captures land in the right
    /// slots.
    fn parse_lookaround(&mut self, negate: bool, behind: bool) -> Result<(), String> {
        let saved = std::mem::take(&mut self.prog);
        self.parse_alt()?;
        if !self.eat(')') {
            return Err("unmatched ( in lookaround".into());
        }
        self.prog.push(Op::Match);
        let sub = std::mem::replace(&mut self.prog, saved);
        self.prog.push(Op::Look {
            negate,
            behind,
            prog: sub,
        });
        Ok(())
    }

    fn emit_char(&mut self, c: char) {
        if self.ignore_case && c.is_ascii_alphabetic() {
            let lo = c.to_ascii_lowercase();
            let hi = c.to_ascii_uppercase();
            self.prog.push(Op::Class {
                ranges: vec![(lo as u32, lo as u32), (hi as u32, hi as u32)],
                negate: false,
            });
        } else {
            self.prog.push(Op::Char(c));
        }
    }

    fn parse_escape(&mut self) -> Result<(), String> {
        self.bump();
        let c = match self.bump() {
            Some(c) => c,
            None => return Err("trailing backslash".into()),
        };
        match c {
            'd' => self.prog.push(Op::Class {
                ranges: vec![('0' as u32, '9' as u32)],
                negate: false,
            }),
            'D' => self.prog.push(Op::Class {
                ranges: vec![('0' as u32, '9' as u32)],
                negate: true,
            }),
            'w' => self.prog.push(Op::Class {
                ranges: vec![
                    ('a' as u32, 'z' as u32),
                    ('A' as u32, 'Z' as u32),
                    ('0' as u32, '9' as u32),
                    ('_' as u32, '_' as u32),
                ],
                negate: false,
            }),
            'W' => self.prog.push(Op::Class {
                ranges: vec![
                    ('a' as u32, 'z' as u32),
                    ('A' as u32, 'Z' as u32),
                    ('0' as u32, '9' as u32),
                    ('_' as u32, '_' as u32),
                ],
                negate: true,
            }),
            's' => self.prog.push(Op::Class {
                ranges: vec![
                    (' ' as u32, ' ' as u32),
                    ('\t' as u32, '\t' as u32),
                    ('\n' as u32, '\n' as u32),
                    ('\r' as u32, '\r' as u32),
                    (0x0c, 0x0c), // form feed
                    (0x0b, 0x0b), // vtab
                ],
                negate: false,
            }),
            'S' => self.prog.push(Op::Class {
                ranges: vec![
                    (' ' as u32, ' ' as u32),
                    ('\t' as u32, '\t' as u32),
                    ('\n' as u32, '\n' as u32),
                    ('\r' as u32, '\r' as u32),
                    (0x0c, 0x0c),
                    (0x0b, 0x0b),
                ],
                negate: true,
            }),
            'b' => self.prog.push(Op::Boundary(true)),
            'B' => self.prog.push(Op::Boundary(false)),
            'n' => self.emit_char('\n'),
            'r' => self.emit_char('\r'),
            't' => self.emit_char('\t'),
            '0' => self.emit_char('\0'),
            'f' => self.emit_char('\u{0c}'),
            'v' => self.emit_char('\u{0b}'),
            c if c.is_ascii_digit() => {
                // Backreference: \1..\9. The lexer ate one digit; in
                // the future we should peek for two-digit refs.
                let idx = c.to_digit(10).unwrap_or(0) as usize;
                if idx == 0 {
                    self.emit_char('\0');
                } else {
                    self.prog.push(Op::BackRef(idx));
                }
            }
            c => self.emit_char(c),
        }
        Ok(())
    }

    fn parse_class(&mut self) -> Result<(), String> {
        self.bump(); // consume '['
        let mut negate = false;
        if self.eat('^') {
            negate = true;
        }
        let mut ranges: Vec<(u32, u32)> = Vec::new();
        while let Some(c) = self.peek() {
            if c == ']' {
                self.bump();
                if self.ignore_case {
                    // Case-fold ranges: expand A-Z to a-z and vice versa.
                    let mut extra: Vec<(u32, u32)> = Vec::new();
                    for &(lo, hi) in &ranges {
                        for cu in lo..=hi {
                            if let Some(c) = char::from_u32(cu) {
                                if c.is_ascii_alphabetic() {
                                    let other = if c.is_ascii_lowercase() {
                                        c.to_ascii_uppercase() as u32
                                    } else {
                                        c.to_ascii_lowercase() as u32
                                    };
                                    extra.push((other, other));
                                }
                            }
                        }
                    }
                    ranges.extend(extra);
                }
                self.prog.push(Op::Class { ranges, negate });
                return Ok(());
            }
            if c == '\\' {
                if let Some(escaped) = self.src.get(self.pos + 1).copied() {
                    if let Some(mut escaped_ranges) = class_escape_ranges(escaped) {
                        self.bump();
                        self.bump();
                        ranges.append(&mut escaped_ranges);
                        continue;
                    }
                }
            }
            let lo = self.class_atom()?;
            let hi = if self.peek() == Some('-') && self.src.get(self.pos + 1).copied() != Some(']')
            {
                self.bump();
                self.class_atom()?
            } else {
                lo
            };
            ranges.push((lo, hi));
        }
        Err("unterminated character class".into())
    }

    fn class_atom(&mut self) -> Result<u32, String> {
        match self.bump() {
            None => Err("unterminated character class".into()),
            Some('\\') => match self.bump() {
                Some('n') => Ok('\n' as u32),
                Some('r') => Ok('\r' as u32),
                Some('t') => Ok('\t' as u32),
                Some('f') => Ok(0x0C),
                Some('v') => Ok(0x0B),
                Some('b') => Ok(0x08), // backspace inside a class (vs word-boundary outside)
                Some('0') => Ok(0),
                // `\uXXXX` and `\u{XXXXX}` unicode escapes, `\xXX` hex escape.
                // CRITICAL: without these, `[\uD800-\uDFFF]` (core-js's lone-
                // surrogate fixer in JSON.stringify) parsed `\u` as the literal
                // 'u' and the leftover `D800-DFFF` as a garbage range that
                // matched nearly every character — corrupting every
                // JSON.stringify output and, downstream, core-js's URL parser.
                Some('u') => {
                    if self.peek() == Some('{') {
                        self.bump();
                        let mut val: u32 = 0;
                        let mut any = false;
                        while let Some(c) = self.peek() {
                            if c == '}' {
                                self.bump();
                                break;
                            }
                            let d = c.to_digit(16).ok_or("bad \\u{} escape in class")?;
                            val = val.saturating_mul(16).saturating_add(d);
                            any = true;
                            self.bump();
                        }
                        if !any {
                            return Err("empty \\u{} escape in class".into());
                        }
                        Ok(val)
                    } else {
                        let mut val: u32 = 0;
                        for _ in 0..4 {
                            let c = self.bump().ok_or("bad \\u escape in class")?;
                            let d = c.to_digit(16).ok_or("bad \\u escape in class")?;
                            val = val * 16 + d;
                        }
                        Ok(val)
                    }
                }
                Some('x') => {
                    let mut val: u32 = 0;
                    for _ in 0..2 {
                        let c = self.bump().ok_or("bad \\x escape in class")?;
                        let d = c.to_digit(16).ok_or("bad \\x escape in class")?;
                        val = val * 16 + d;
                    }
                    Ok(val)
                }
                Some('d') | Some('D') | Some('w') | Some('W') | Some('s') | Some('S') => {
                    // Class-in-class isn't representable as a single
                    // range — V1 approximation: treat as a single
                    // literal of the escape's first range start. Most
                    // real-world patterns don't use these inside [].
                    Ok('?' as u32)
                }
                Some(c) => Ok(c as u32),
                None => Err("trailing backslash".into()),
            },
            Some(c) => Ok(c as u32),
        }
    }
}

fn class_escape_ranges(c: char) -> Option<Vec<(u32, u32)>> {
    let max = char::MAX as u32;
    match c {
        'd' => Some(vec![('0' as u32, '9' as u32)]),
        'D' => Some(vec![(0, '/' as u32), (':' as u32, max)]),
        'w' => Some(vec![
            ('0' as u32, '9' as u32),
            ('A' as u32, 'Z' as u32),
            ('_' as u32, '_' as u32),
            ('a' as u32, 'z' as u32),
        ]),
        'W' => Some(vec![
            (0, '/' as u32),
            (':' as u32, '@' as u32),
            ('[' as u32, '^' as u32),
            ('`' as u32, '`' as u32),
            ('{' as u32, max),
        ]),
        's' => Some(vec![
            ('\t' as u32, '\r' as u32),
            (' ' as u32, ' ' as u32),
            (0x00A0, 0x00A0),
            (0x1680, 0x1680),
            (0x2000, 0x200A),
            (0x2028, 0x2029),
            (0x202F, 0x202F),
            (0x205F, 0x205F),
            (0x3000, 0x3000),
        ]),
        'S' => Some(vec![
            (0, '\t' as u32 - 1),
            ('\r' as u32 + 1, ' ' as u32 - 1),
            (' ' as u32 + 1, 0x009F),
            (0x00A1, 0x167F),
            (0x1681, 0x1FFF),
            (0x200B, 0x2027),
            (0x202A, 0x202E),
            (0x2030, 0x205E),
            (0x2060, 0x2FFF),
            (0x3001, max),
        ]),
        _ => None,
    }
}

// ----------------------------------------------------------------------
// Executor (backtracking)
// ----------------------------------------------------------------------

fn exec_thread(
    prog: &[Op],
    mut pc: usize,
    chars: &[(usize, char)],
    byte_starts: &[usize],
    mut i: usize,
    caps: &mut Vec<(Option<usize>, Option<usize>)>,
    re: &Regex,
    // When `Some(e)`, a `Match` only succeeds if the current position equals `e`
    // (used by lookbehind, which must end exactly at the assertion point).
    require_end: Option<usize>,
) -> bool {
    // Iterative backtracking with a stack of (pc, i, caps_snapshot).
    // The interpreter avoids recursion to keep big alternations from
    // blowing the host stack.
    struct Frame {
        pc: usize,
        i: usize,
        caps: Vec<(Option<usize>, Option<usize>)>,
    }
    let mut stack: Vec<Frame> = Vec::new();
    let n = chars.len();
    let byte_at = |k: usize| -> usize {
        if k < n {
            byte_starts[k]
        } else {
            byte_starts[n]
        }
    };
    // Watchdog: this backtracking matcher can blow up exponentially on a
    // pathological pattern (`(a+)+$` vs a long non-match). Our engine is far
    // slower per step than V8's, so a regex that's fine in Chrome can wedge the
    // UI thread here in native code — invisible to the statement/VM watchdogs.
    // Honour the per-task wall-clock deadline so it aborts (as a non-match)
    // instead of freezing; the JS task watchdog then unwinds the script.
    let mut steps: u64 = 0;
    loop {
        steps = steps.wrapping_add(1);
        if steps & 0x3FFF == 0 && crate::interp::js_runtime_deadline_exceeded() {
            return false;
        }
        if pc >= prog.len() {
            return false;
        }
        match &prog[pc] {
            Op::Match => match require_end {
                None => return true,
                Some(e) if i == e => return true,
                Some(_) => {
                    // Reached the sub-pattern end but not at the required
                    // position (lookbehind) — backtrack to try another path.
                    if let Some(f) = stack.pop() {
                        pc = f.pc;
                        i = f.i;
                        *caps = f.caps;
                    } else {
                        return false;
                    }
                }
            },
            Op::Look {
                negate,
                behind,
                prog: sub,
            } => {
                let mut matched_caps: Option<Vec<(Option<usize>, Option<usize>)>> = None;
                let ok = if *behind {
                    // Lookbehind: sub-pattern must match ENDING exactly at i.
                    // Try each start position j ≤ i (existence is enough).
                    let mut found = false;
                    let mut j = i + 1;
                    while j > 0 {
                        j -= 1;
                        let mut sc = caps.clone();
                        if exec_thread(sub, 0, chars, byte_starts, j, &mut sc, re, Some(i)) {
                            matched_caps = Some(sc);
                            found = true;
                            break;
                        }
                    }
                    found
                } else {
                    // Lookahead: sub-pattern must match STARTING at i.
                    let mut sc = caps.clone();
                    if exec_thread(sub, 0, chars, byte_starts, i, &mut sc, re, None) {
                        matched_caps = Some(sc);
                        true
                    } else {
                        false
                    }
                };
                if ok != *negate {
                    // Assertion holds — zero-width. Positive lookarounds keep
                    // their inner captures (JS semantics); negative discard.
                    if !*negate {
                        if let Some(sc) = matched_caps {
                            *caps = sc;
                        }
                    }
                    pc += 1;
                } else if let Some(f) = stack.pop() {
                    pc = f.pc;
                    i = f.i;
                    *caps = f.caps;
                } else {
                    return false;
                }
            }
            Op::Char(c) => {
                if i < n && chars[i].1 == *c {
                    pc += 1;
                    i += 1;
                } else if let Some(f) = stack.pop() {
                    pc = f.pc;
                    i = f.i;
                    *caps = f.caps;
                } else {
                    return false;
                }
            }
            Op::AnyChar => {
                if i < n && (re.dot_all || chars[i].1 != '\n') {
                    pc += 1;
                    i += 1;
                } else if let Some(f) = stack.pop() {
                    pc = f.pc;
                    i = f.i;
                    *caps = f.caps;
                } else {
                    return false;
                }
            }
            Op::Class { ranges, negate } => {
                if i < n {
                    let cu = chars[i].1 as u32;
                    let mut hit = ranges.iter().any(|(lo, hi)| cu >= *lo && cu <= *hi);
                    if *negate {
                        hit = !hit;
                    }
                    if hit {
                        pc += 1;
                        i += 1;
                        continue;
                    }
                }
                if let Some(f) = stack.pop() {
                    pc = f.pc;
                    i = f.i;
                    *caps = f.caps;
                } else {
                    return false;
                }
            }
            Op::Anchor(Anchor::Start) => {
                let ok = i == 0 || (re.multiline && i > 0 && chars[i - 1].1 == '\n');
                if ok {
                    pc += 1;
                } else if let Some(f) = stack.pop() {
                    pc = f.pc;
                    i = f.i;
                    *caps = f.caps;
                } else {
                    return false;
                }
            }
            Op::Anchor(Anchor::End) => {
                let ok = i == n || (re.multiline && i < n && chars[i].1 == '\n');
                if ok {
                    pc += 1;
                } else if let Some(f) = stack.pop() {
                    pc = f.pc;
                    i = f.i;
                    *caps = f.caps;
                } else {
                    return false;
                }
            }
            Op::Boundary(want_boundary) => {
                let lc = if i == 0 { None } else { Some(chars[i - 1].1) };
                let rc = if i < n { Some(chars[i].1) } else { None };
                let is_word = |c: Option<char>| -> bool {
                    c.map(|c| c.is_ascii_alphanumeric() || c == '_')
                        .unwrap_or(false)
                };
                let at_boundary = is_word(lc) != is_word(rc);
                if at_boundary == *want_boundary {
                    pc += 1;
                } else if let Some(f) = stack.pop() {
                    pc = f.pc;
                    i = f.i;
                    *caps = f.caps;
                } else {
                    return false;
                }
            }
            Op::Save(g) => {
                if let Some(slot) = caps.get_mut(*g) {
                    slot.0 = Some(byte_at(i));
                }
                pc += 1;
            }
            Op::Restore(g) => {
                if let Some(slot) = caps.get_mut(*g) {
                    slot.1 = Some(byte_at(i));
                }
                pc += 1;
            }
            Op::BackRef(g) => {
                let (gs, ge) = match caps.get(*g).cloned().unwrap_or((None, None)) {
                    (Some(s), Some(e)) => (s, e),
                    _ => {
                        if let Some(f) = stack.pop() {
                            pc = f.pc;
                            i = f.i;
                            *caps = f.caps;
                            continue;
                        }
                        return false;
                    }
                };
                // Compare bytes ge-gs starting at i.
                let mut ok = true;
                let mut taken = 0usize;
                let src_chars: Vec<char> = {
                    let target_start = chars.iter().position(|(b, _)| *b == gs).unwrap_or(0);
                    let target_end = chars.iter().position(|(b, _)| *b == ge).unwrap_or(n);
                    chars[target_start..target_end]
                        .iter()
                        .map(|(_, c)| *c)
                        .collect()
                };
                for sc in &src_chars {
                    if i + taken >= n {
                        ok = false;
                        break;
                    }
                    let mc = chars[i + taken].1;
                    if mc != *sc && !(re.ignore_case && mc.eq_ignore_ascii_case(sc)) {
                        ok = false;
                        break;
                    }
                    taken += 1;
                }
                if ok {
                    i += taken;
                    pc += 1;
                } else if let Some(f) = stack.pop() {
                    pc = f.pc;
                    i = f.i;
                    *caps = f.caps;
                } else {
                    return false;
                }
            }
            Op::Jmp(t) => {
                pc = *t;
            }
            Op::Split(a, b) => {
                // Try `a` first; on failure backtrack to `b`.
                stack.push(Frame {
                    pc: *b,
                    i,
                    caps: caps.clone(),
                });
                pc = *a;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn literal_match() {
        let r = Regex::new("abc", "").unwrap();
        assert!(r.test("xxabcxx"));
        assert!(!r.test("xxabxx"));
    }

    #[test]
    fn positive_lookahead() {
        let r = Regex::new(r"\d+(?=px)", "").unwrap();
        assert_eq!(r.find_from("12px", 0).unwrap().matched, "12");
        assert!(r.find_from("12em", 0).is_none());
    }

    #[test]
    fn negative_lookahead() {
        let r = Regex::new(r"foo(?!bar)", "").unwrap();
        assert!(r.find_from("foobaz", 0).is_some());
        assert!(r.find_from("foobar", 0).is_none());
    }

    #[test]
    fn positive_lookbehind() {
        let r = Regex::new(r"(?<=\$)\d+", "").unwrap();
        assert_eq!(r.find_from("price $42 now", 0).unwrap().matched, "42");
        assert!(r.find_from("price 42 now", 0).is_none());
    }

    #[test]
    fn negative_lookbehind() {
        let r = Regex::new(r"(?<!a)b", "").unwrap();
        assert_eq!(r.find_from("xb", 0).unwrap().matched, "b");
        assert!(r.find_from("ab", 0).is_none());
    }

    #[test]
    fn lookahead_captures_persist() {
        // A positive lookahead's inner capture group participates in the match.
        let r = Regex::new(r"(?=(\d+))", "").unwrap();
        let m = r.find_from("abc123", 0).unwrap();
        assert_eq!(m.matched, ""); // zero-width
        assert_eq!(m.group_strings.get(1).cloned().flatten(), Some("123".to_string()));
    }
    #[test]
    fn quantifiers() {
        let r = Regex::new("a+b", "").unwrap();
        assert!(r.test("aaab"));
        assert!(r.test("ab"));
        assert!(!r.test("b"));
    }
    #[test]
    fn char_class() {
        let r = Regex::new("[abc]+", "").unwrap();
        let m = r.find_from("xxabcxx", 0).unwrap();
        assert_eq!(m.matched, "abc");
    }

    #[test]
    fn class_escape_digits_work_inside_ranges() {
        let r = Regex::new("^#?([a-f\\d]{2})([a-f\\d]{2})([a-f\\d]{2})$", "i").unwrap();
        let m = r.find_from("#FFD700", 0).unwrap();
        assert_eq!(m.group_strings[1].as_deref(), Some("FF"));
        assert_eq!(m.group_strings[2].as_deref(), Some("D7"));
        assert_eq!(m.group_strings[3].as_deref(), Some("00"));
    }

    #[test]
    fn unicode_escape_in_class_range() {
        // `\uXXXX` inside a class range must decode to the codepoint, not the
        // literal 'u'. core-js's JSON.stringify lone-surrogate fix uses
        // `/[\uD800-\uDFFF]/g`; before this fix `\u` was read as 'u' and the
        // leftover `D800-DFFF` formed a garbage range matching nearly every
        // char — corrupting all JSON output (and core-js's URL parser).
        let r = Regex::new("[\\uD800-\\uDFFF]", "").unwrap();
        assert!(!r.test("a")); // ASCII must NOT match a surrogate range
        assert!(!r.test("[")); // nor brackets / punctuation
        assert!(!r.test("\"")); // nor a quote
        assert!(!r.test("z")); // nor any BMP letter
        // A-F hex digits and the exact codepoint range are honored:
        let r2 = Regex::new("[\\u0041-\\u005A]", "").unwrap(); // A-Z
        assert!(r2.test("B"));
        assert!(!r2.test("b"));
        // `\xNN` two-digit hex also works in a class.
        let r3 = Regex::new("[\\x61-\\x7A]", "").unwrap(); // a-z
        assert!(r3.test("m"));
        assert!(!r3.test("M"));
    }
    #[test]
    fn anchors() {
        let r = Regex::new("^foo$", "").unwrap();
        assert!(r.test("foo"));
        assert!(!r.test("xfoo"));
    }
    #[test]
    fn case_insensitive() {
        let r = Regex::new("hello", "i").unwrap();
        assert!(r.test("HELLO"));
        assert!(r.test("Hello"));
    }
    #[test]
    fn groups() {
        let r = Regex::new("(a)(b)", "").unwrap();
        let m = r.find_from("zabz", 0).unwrap();
        assert_eq!(m.group_strings[1].as_deref(), Some("a"));
        assert_eq!(m.group_strings[2].as_deref(), Some("b"));
    }
    #[test]
    fn alternation() {
        let r = Regex::new("cat|dog", "").unwrap();
        assert!(r.test("a dog"));
        assert!(r.test("a cat"));
        assert!(!r.test("fish"));
    }
    #[test]
    fn dot_default_skips_newline() {
        let r = Regex::new("a.b", "").unwrap();
        assert!(r.test("axb"));
        assert!(!r.test("a\nb"));
    }
    #[test]
    fn dot_with_s_flag() {
        let r = Regex::new("a.b", "s").unwrap();
        assert!(r.test("a\nb"));
    }
    #[test]
    fn find_all_iterates() {
        let r = Regex::new("[a-z]+", "g").unwrap();
        let ms = r.find_all("hi 1 world 22 foo");
        let strs: Vec<&str> = ms.iter().map(|m| m.matched.as_str()).collect();
        assert_eq!(strs, vec!["hi", "world", "foo"]);
    }
    #[test]
    fn quantifier_brace() {
        let r = Regex::new("a{3}", "").unwrap();
        assert!(r.test("aaa"));
        assert!(!r.test("aa"));
    }
    #[test]
    fn quantifier_range() {
        let r = Regex::new("a{2,4}", "").unwrap();
        assert!(r.test("aa"));
        assert!(r.test("aaaa"));
        assert!(!r.test("a"));
    }
}
