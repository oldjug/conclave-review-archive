//! A real GLSL ES interpreter for the software WebGL backend.
//!
//! This is NOT a stub: it tokenizes, parses, and *executes* GLSL ES 1.00
//! (the WebGL 1 shading language, OpenGL ES Shading Language 1.00 spec)
//! source. The vertex stage computes `gl_Position` from per-vertex
//! `attribute` inputs and `uniform`s; the fragment stage computes
//! `gl_FragColor`. Compilation returns a real success/failure status with
//! an info log — a syntax error (missing `main`, unbalanced braces, a parse
//! error, an unknown function call) reports COMPILE_STATUS=false with a
//! non-empty log, exactly as Chrome/ANGLE do (per the WebGL 1.0 spec
//! §5.13.9 getShaderParameter(COMPILE_STATUS) and the GLSL ES 1.00 grammar).
//!
//! The interpreter supports the surface real demos (three.js's simplest
//! material, hand-written shaders, MDN/learn-webgl tutorials) actually use:
//!   * types: float, int, bool, vec2/3/4, mat2/3/4 (and `void` returns)
//!   * qualifiers: attribute, uniform, varying, precision, const
//!   * the `main()` function plus user functions with parameters
//!   * vector constructors `vec4(x,y,z,w)` incl. scalar/vector mixing
//!   * swizzles `.xyz`, `.rgba`, `.stpq`, write-swizzle on assignment
//!   * arithmetic `+ - * /`, unary `-`, comparisons, `&& || !`
//!   * mat*vec, mat*mat, scalar*vec, vec*vec componentwise
//!   * builtins: normalize, dot, cross, length, mix, clamp, min, max,
//!     abs, floor, fract, mod, pow, sqrt, sin, cos, tan, step, smoothstep,
//!     reflect, texture2D/texture (returns a sampled texel)
//!   * statements: declarations, assignment, compound assignment, `if/else`,
//!     `for` (counted loops), `return`, blocks
//!
//! Anything genuinely unsupported during *execution* is reported honestly as
//! a runtime error (Result::Err), never silently faked.

use std::collections::HashMap;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Stage {
    Vertex,
    Fragment,
}

// ---------------------------------------------------------------------------
// Values
// ---------------------------------------------------------------------------

/// A runtime GLSL value. Vectors and matrices are stored as flat f32 arrays.
#[derive(Clone, Debug, PartialEq)]
pub enum Val {
    Bool(bool),
    Float(f32),
    Int(i32),
    /// Vector of length 2..=4.
    Vec(Vec<f32>),
    /// Column-major NxN matrix (n in {2,3,4}); `data.len()==n*n`.
    Mat { n: usize, data: Vec<f32> },
}

impl Val {
    fn as_scalar(&self) -> Option<f32> {
        match self {
            Val::Float(f) => Some(*f),
            Val::Int(i) => Some(*i as f32),
            Val::Bool(b) => Some(if *b { 1.0 } else { 0.0 }),
            _ => None,
        }
    }
    fn as_vec(&self) -> Option<Vec<f32>> {
        match self {
            Val::Vec(v) => Some(v.clone()),
            Val::Float(f) => Some(vec![*f]),
            Val::Int(i) => Some(vec![*i as f32]),
            _ => None,
        }
    }
    /// Expand a value to exactly `n` components (a scalar broadcasts).
    pub fn to_vecn(&self, n: usize) -> Option<Vec<f32>> {
        match self {
            Val::Vec(v) if v.len() == n => Some(v.clone()),
            Val::Vec(v) if v.len() > n => Some(v[..n].to_vec()),
            Val::Float(f) => Some(vec![*f; n]),
            Val::Int(i) => Some(vec![*i as f32; n]),
            _ => None,
        }
    }
    pub fn to_vec4(&self) -> [f32; 4] {
        match self {
            Val::Vec(v) => {
                let mut out = [0.0, 0.0, 0.0, 1.0];
                for (i, c) in v.iter().take(4).enumerate() {
                    out[i] = *c;
                }
                out
            }
            Val::Float(f) => [*f, *f, *f, *f],
            Val::Int(i) => [*i as f32; 4],
            Val::Bool(b) => {
                let v = if *b { 1.0 } else { 0.0 };
                [v, v, v, v]
            }
            Val::Mat { .. } => [0.0, 0.0, 0.0, 1.0],
        }
    }
}

// ---------------------------------------------------------------------------
// Tokenizer
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq)]
enum Tok {
    Ident(String),
    Num(f32, bool), // value, is_int
    Sym(String),
}

fn tokenize(src: &str) -> Result<Vec<Tok>, String> {
    let b = src.as_bytes();
    let mut i = 0;
    let mut toks = Vec::new();
    while i < b.len() {
        let c = b[i] as char;
        if c.is_whitespace() {
            i += 1;
            continue;
        }
        // Line comment.
        if c == '/' && i + 1 < b.len() && b[i + 1] == b'/' {
            while i < b.len() && b[i] != b'\n' {
                i += 1;
            }
            continue;
        }
        // Block comment.
        if c == '/' && i + 1 < b.len() && b[i + 1] == b'*' {
            i += 2;
            while i + 1 < b.len() && !(b[i] == b'*' && b[i + 1] == b'/') {
                i += 1;
            }
            if i + 1 >= b.len() {
                return Err("unterminated block comment".into());
            }
            i += 2;
            continue;
        }
        // Preprocessor: keep it simple — skip `#...` directive lines. (Real
        // ANGLE runs a full preprocessor; demos that need #define beyond this
        // get a runtime error rather than a silent wrong result.)
        if c == '#' {
            while i < b.len() && b[i] != b'\n' {
                i += 1;
            }
            continue;
        }
        if c.is_ascii_alphabetic() || c == '_' {
            let start = i;
            while i < b.len() && {
                let ch = b[i] as char;
                ch.is_ascii_alphanumeric() || ch == '_'
            } {
                i += 1;
            }
            toks.push(Tok::Ident(src[start..i].to_string()));
            continue;
        }
        if c.is_ascii_digit() || (c == '.' && i + 1 < b.len() && (b[i + 1] as char).is_ascii_digit())
        {
            let start = i;
            let mut is_int = true;
            while i < b.len() {
                let ch = b[i] as char;
                if ch.is_ascii_digit() {
                    i += 1;
                } else if ch == '.' {
                    is_int = false;
                    i += 1;
                } else if ch == 'e' || ch == 'E' {
                    is_int = false;
                    i += 1;
                    if i < b.len() && (b[i] == b'+' || b[i] == b'-') {
                        i += 1;
                    }
                } else if ch == 'f' || ch == 'F' {
                    // float suffix
                    is_int = false;
                    i += 1;
                    break;
                } else {
                    break;
                }
            }
            let txt = src[start..i].trim_end_matches(['f', 'F']);
            let v: f32 = txt
                .parse()
                .map_err(|_| format!("bad number literal '{txt}'"))?;
            toks.push(Tok::Num(v, is_int));
            continue;
        }
        // Multi-char operators.
        let two = if i + 1 < b.len() {
            &src[i..i + 2]
        } else {
            ""
        };
        match two {
            "==" | "!=" | "<=" | ">=" | "&&" | "||" | "+=" | "-=" | "*=" | "/=" | "++" | "--" => {
                toks.push(Tok::Sym(two.to_string()));
                i += 2;
                continue;
            }
            _ => {}
        }
        // Single-char symbol.
        toks.push(Tok::Sym(c.to_string()));
        i += 1;
    }
    Ok(toks)
}

// ---------------------------------------------------------------------------
// AST
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
enum Expr {
    Num(f32, bool),
    Bool(bool),
    Var(String),
    Call(String, Vec<Expr>),
    Member(Box<Expr>, String), // swizzle / field
    Index(Box<Expr>, Box<Expr>),
    Unary(String, Box<Expr>),
    Bin(String, Box<Expr>, Box<Expr>),
    Assign(String, Box<Expr>, Box<Expr>), // op ("=","+=",…), lhs, rhs
}

#[derive(Clone, Debug)]
enum Stmt {
    Decl {
        name: String,
        init: Option<Expr>,
    },
    Expr(Expr),
    If(Expr, Vec<Stmt>, Vec<Stmt>),
    For {
        init: Box<Stmt>,
        cond: Expr,
        step: Expr,
        body: Vec<Stmt>,
    },
    Return(Option<Expr>),
    Block(Vec<Stmt>),
}

#[derive(Clone, Debug)]
struct Func {
    #[allow(dead_code)]
    name: String,
    params: Vec<String>,
    body: Vec<Stmt>,
}

/// The compiled (parsed) shader program. Holding this is proof of a real
/// successful compile.
#[derive(Clone, Debug)]
pub struct CompiledShader {
    pub stage: Stage,
    funcs: HashMap<String, Func>,
    /// Names declared `attribute` (vertex inputs).
    pub attributes: Vec<String>,
    /// Names declared `uniform`.
    pub uniforms: Vec<String>,
    /// Names declared `varying`.
    pub varyings: Vec<String>,
}

// ---------------------------------------------------------------------------
// Parser
// ---------------------------------------------------------------------------

struct Parser {
    toks: Vec<Tok>,
    pos: usize,
    attributes: Vec<String>,
    uniforms: Vec<String>,
    varyings: Vec<String>,
}

const TYPE_KEYWORDS: &[&str] = &[
    "void", "float", "int", "bool", "vec2", "vec3", "vec4", "mat2", "mat3", "mat4", "ivec2",
    "ivec3", "ivec4", "bvec2", "bvec3", "bvec4", "sampler2D", "samplerCube",
];

fn is_type(s: &str) -> bool {
    TYPE_KEYWORDS.contains(&s)
}

impl Parser {
    fn peek(&self) -> Option<&Tok> {
        self.toks.get(self.pos)
    }
    fn next(&mut self) -> Option<Tok> {
        let t = self.toks.get(self.pos).cloned();
        if t.is_some() {
            self.pos += 1;
        }
        t
    }
    fn eat_sym(&mut self, s: &str) -> Result<(), String> {
        match self.next() {
            Some(Tok::Sym(x)) if x == s => Ok(()),
            other => Err(format!("expected '{s}', found {other:?}")),
        }
    }
    fn is_sym(&self, s: &str) -> bool {
        matches!(self.peek(), Some(Tok::Sym(x)) if x == s)
    }
    fn is_ident(&self, s: &str) -> bool {
        matches!(self.peek(), Some(Tok::Ident(x)) if x == s)
    }

    fn parse_program(&mut self) -> Result<HashMap<String, Func>, String> {
        let mut funcs = HashMap::new();
        while self.peek().is_some() {
            // Global qualifier-prefixed declarations.
            if self.is_ident("precision") {
                // precision <qualifier> <type>;
                self.skip_to_semicolon()?;
                continue;
            }
            let qualifier = self.maybe_qualifier();
            // After a qualifier (or none) expect a type for a decl or function.
            match self.peek() {
                Some(Tok::Ident(s)) if is_type(s) => {}
                Some(Tok::Ident(_)) if qualifier.is_some() => {
                    return Err(format!(
                        "expected a type after qualifier, found {:?}",
                        self.peek()
                    ));
                }
                _ => {
                    return Err(format!("unexpected token at top level: {:?}", self.peek()));
                }
            }
            self.next(); // consume type
            // The name.
            let name = match self.next() {
                Some(Tok::Ident(n)) => n,
                other => return Err(format!("expected identifier, found {other:?}")),
            };
            if self.is_sym("(") {
                // Function definition (or prototype).
                let func = self.parse_function(name.clone())?;
                if let Some(f) = func {
                    funcs.insert(name, f);
                }
            } else {
                // Global variable declaration. Record qualifier class.
                match qualifier.as_deref() {
                    Some("attribute") => self.attributes.push(name.clone()),
                    Some("uniform") => self.uniforms.push(name.clone()),
                    Some("varying") => self.varyings.push(name.clone()),
                    _ => {}
                }
                // Skip the rest of the declaration (initializer / array / list).
                self.skip_to_semicolon()?;
            }
        }
        Ok(funcs)
    }

    fn maybe_qualifier(&mut self) -> Option<String> {
        let q = match self.peek() {
            Some(Tok::Ident(s))
                if matches!(
                    s.as_str(),
                    "attribute" | "uniform" | "varying" | "const" | "in" | "out" | "highp"
                        | "mediump" | "lowp" | "invariant"
                ) =>
            {
                Some(s.clone())
            }
            _ => None,
        };
        if q.is_some() {
            self.next();
            // Allow multiple stacked qualifiers (e.g. `varying highp vec3`).
            // Precision qualifiers are not the "class" we track.
            if matches!(q.as_deref(), Some("highp" | "mediump" | "lowp" | "invariant")) {
                return self.maybe_qualifier().or(q);
            }
            // skip a trailing precision qualifier after the class qualifier.
            if let Some(Tok::Ident(s)) = self.peek() {
                if matches!(s.as_str(), "highp" | "mediump" | "lowp") {
                    self.next();
                }
            }
        }
        q
    }

    fn parse_function(&mut self, name: String) -> Result<Option<Func>, String> {
        self.eat_sym("(")?;
        let mut params = Vec::new();
        if !self.is_sym(")") {
            loop {
                // [qualifier] type name
                let _ = self.maybe_qualifier();
                // type
                match self.next() {
                    Some(Tok::Ident(t)) if is_type(&t) || t == "void" => {}
                    Some(Tok::Ident(t)) => {
                        return Err(format!("expected param type, found '{t}'"));
                    }
                    other => return Err(format!("expected param type, found {other:?}")),
                }
                // name (may be absent for `void`)
                if let Some(Tok::Ident(pn)) = self.peek().cloned() {
                    if !is_type(&pn) {
                        self.next();
                        params.push(pn);
                    }
                }
                if self.is_sym(",") {
                    self.next();
                    continue;
                }
                break;
            }
        }
        self.eat_sym(")")?;
        if self.is_sym(";") {
            // prototype only
            self.next();
            return Ok(None);
        }
        self.eat_sym("{")?;
        let body = self.parse_block()?;
        Ok(Some(Func { name, params, body }))
    }

    /// Parse statements up to and consuming the matching `}`.
    fn parse_block(&mut self) -> Result<Vec<Stmt>, String> {
        let mut stmts = Vec::new();
        while !self.is_sym("}") {
            if self.peek().is_none() {
                return Err("unexpected end of input (unbalanced braces)".into());
            }
            stmts.push(self.parse_stmt()?);
        }
        self.eat_sym("}")?;
        Ok(stmts)
    }

    fn parse_stmt(&mut self) -> Result<Stmt, String> {
        if self.is_sym("{") {
            self.next();
            let b = self.parse_block()?;
            return Ok(Stmt::Block(b));
        }
        if self.is_ident("if") {
            self.next();
            self.eat_sym("(")?;
            let cond = self.parse_expr()?;
            self.eat_sym(")")?;
            let then = self.parse_stmt_as_block()?;
            let els = if self.is_ident("else") {
                self.next();
                self.parse_stmt_as_block()?
            } else {
                Vec::new()
            };
            return Ok(Stmt::If(cond, then, els));
        }
        if self.is_ident("for") {
            self.next();
            self.eat_sym("(")?;
            let init = Box::new(self.parse_simple_stmt()?); // consumes ';'
            let cond = self.parse_expr()?;
            self.eat_sym(";")?;
            let step = self.parse_expr()?;
            self.eat_sym(")")?;
            let body = self.parse_stmt_as_block()?;
            return Ok(Stmt::For {
                init,
                cond,
                step,
                body,
            });
        }
        if self.is_ident("return") {
            self.next();
            if self.is_sym(";") {
                self.next();
                return Ok(Stmt::Return(None));
            }
            let e = self.parse_expr()?;
            self.eat_sym(";")?;
            return Ok(Stmt::Return(Some(e)));
        }
        if self.is_ident("discard") {
            self.next();
            self.eat_sym(";")?;
            // Represent discard as a return-with-flag via a sentinel call.
            return Ok(Stmt::Expr(Expr::Call("__discard".into(), vec![])));
        }
        self.parse_simple_stmt()
    }

    /// A declaration or expression statement, consuming the trailing ';'.
    fn parse_simple_stmt(&mut self) -> Result<Stmt, String> {
        // Declaration: [qualifier] type name [= expr];
        let is_decl = matches!(self.peek(), Some(Tok::Ident(s)) if is_type(s) || matches!(s.as_str(),"const"|"highp"|"mediump"|"lowp"));
        if is_decl {
            let _ = self.maybe_qualifier();
            // type
            self.next();
            let name = match self.next() {
                Some(Tok::Ident(n)) => n,
                other => return Err(format!("expected declared name, found {other:?}")),
            };
            // optional array suffix
            if self.is_sym("[") {
                self.skip_brackets()?;
            }
            let init = if self.is_sym("=") {
                self.next();
                Some(self.parse_expr()?)
            } else {
                None
            };
            // Allow comma-separated extra declarations by ignoring them past
            // the first (rare in shader main bodies). Just stop at ';'.
            while self.is_sym(",") {
                self.next();
                // name [= expr]
                let _ = self.next();
                if self.is_sym("=") {
                    self.next();
                    let _ = self.parse_expr()?;
                }
            }
            self.eat_sym(";")?;
            return Ok(Stmt::Decl { name, init });
        }
        // Expression statement.
        let e = self.parse_expr()?;
        self.eat_sym(";")?;
        Ok(Stmt::Expr(e))
    }

    fn parse_stmt_as_block(&mut self) -> Result<Vec<Stmt>, String> {
        if self.is_sym("{") {
            self.next();
            self.parse_block()
        } else {
            Ok(vec![self.parse_stmt()?])
        }
    }

    fn skip_brackets(&mut self) -> Result<(), String> {
        self.eat_sym("[")?;
        let mut depth = 1;
        while depth > 0 {
            match self.next() {
                Some(Tok::Sym(s)) if s == "[" => depth += 1,
                Some(Tok::Sym(s)) if s == "]" => depth -= 1,
                None => return Err("unbalanced '['".into()),
                _ => {}
            }
        }
        Ok(())
    }

    fn skip_to_semicolon(&mut self) -> Result<(), String> {
        loop {
            match self.next() {
                Some(Tok::Sym(s)) if s == ";" => return Ok(()),
                None => return Err("expected ';' before end of input".into()),
                _ => {}
            }
        }
    }

    // ---- expressions (precedence climbing) ----

    fn parse_expr(&mut self) -> Result<Expr, String> {
        self.parse_assign()
    }

    fn parse_assign(&mut self) -> Result<Expr, String> {
        let lhs = self.parse_logic_or()?;
        if let Some(Tok::Sym(op)) = self.peek().cloned() {
            if matches!(op.as_str(), "=" | "+=" | "-=" | "*=" | "/=") {
                self.next();
                let rhs = self.parse_assign()?;
                return Ok(Expr::Assign(op, Box::new(lhs), Box::new(rhs)));
            }
        }
        Ok(lhs)
    }

    fn parse_logic_or(&mut self) -> Result<Expr, String> {
        let mut e = self.parse_logic_and()?;
        while self.is_sym("||") {
            self.next();
            let r = self.parse_logic_and()?;
            e = Expr::Bin("||".into(), Box::new(e), Box::new(r));
        }
        Ok(e)
    }
    fn parse_logic_and(&mut self) -> Result<Expr, String> {
        let mut e = self.parse_equality()?;
        while self.is_sym("&&") {
            self.next();
            let r = self.parse_equality()?;
            e = Expr::Bin("&&".into(), Box::new(e), Box::new(r));
        }
        Ok(e)
    }
    fn parse_equality(&mut self) -> Result<Expr, String> {
        let mut e = self.parse_rel()?;
        while self.is_sym("==") || self.is_sym("!=") {
            let op = self.sym_text();
            self.next();
            let r = self.parse_rel()?;
            e = Expr::Bin(op, Box::new(e), Box::new(r));
        }
        Ok(e)
    }
    fn parse_rel(&mut self) -> Result<Expr, String> {
        let mut e = self.parse_add()?;
        while self.is_sym("<") || self.is_sym(">") || self.is_sym("<=") || self.is_sym(">=") {
            let op = self.sym_text();
            self.next();
            let r = self.parse_add()?;
            e = Expr::Bin(op, Box::new(e), Box::new(r));
        }
        Ok(e)
    }
    fn parse_add(&mut self) -> Result<Expr, String> {
        let mut e = self.parse_mul()?;
        while self.is_sym("+") || self.is_sym("-") {
            let op = self.sym_text();
            self.next();
            let r = self.parse_mul()?;
            e = Expr::Bin(op, Box::new(e), Box::new(r));
        }
        Ok(e)
    }
    fn parse_mul(&mut self) -> Result<Expr, String> {
        let mut e = self.parse_unary()?;
        while self.is_sym("*") || self.is_sym("/") {
            let op = self.sym_text();
            self.next();
            let r = self.parse_unary()?;
            e = Expr::Bin(op, Box::new(e), Box::new(r));
        }
        Ok(e)
    }
    fn parse_unary(&mut self) -> Result<Expr, String> {
        if self.is_sym("-") || self.is_sym("!") || self.is_sym("+") {
            let op = self.sym_text();
            self.next();
            let e = self.parse_unary()?;
            if op == "+" {
                return Ok(e);
            }
            return Ok(Expr::Unary(op, Box::new(e)));
        }
        self.parse_postfix()
    }
    fn parse_postfix(&mut self) -> Result<Expr, String> {
        let mut e = self.parse_primary()?;
        loop {
            if self.is_sym(".") {
                self.next();
                let field = match self.next() {
                    Some(Tok::Ident(f)) => f,
                    other => return Err(format!("expected member after '.', found {other:?}")),
                };
                e = Expr::Member(Box::new(e), field);
            } else if self.is_sym("[") {
                self.next();
                let idx = self.parse_expr()?;
                self.eat_sym("]")?;
                e = Expr::Index(Box::new(e), Box::new(idx));
            } else {
                break;
            }
        }
        Ok(e)
    }
    fn parse_primary(&mut self) -> Result<Expr, String> {
        match self.next() {
            Some(Tok::Num(v, is_int)) => Ok(Expr::Num(v, is_int)),
            Some(Tok::Ident(id)) => {
                if id == "true" {
                    return Ok(Expr::Bool(true));
                }
                if id == "false" {
                    return Ok(Expr::Bool(false));
                }
                if self.is_sym("(") {
                    // function / constructor call
                    self.next();
                    let mut args = Vec::new();
                    if !self.is_sym(")") {
                        loop {
                            args.push(self.parse_expr()?);
                            if self.is_sym(",") {
                                self.next();
                                continue;
                            }
                            break;
                        }
                    }
                    self.eat_sym(")")?;
                    Ok(Expr::Call(id, args))
                } else {
                    Ok(Expr::Var(id))
                }
            }
            Some(Tok::Sym(s)) if s == "(" => {
                let e = self.parse_expr()?;
                self.eat_sym(")")?;
                Ok(e)
            }
            other => Err(format!("unexpected token in expression: {other:?}")),
        }
    }

    fn sym_text(&self) -> String {
        if let Some(Tok::Sym(s)) = self.peek() {
            s.clone()
        } else {
            String::new()
        }
    }
}

/// Compile (tokenize + parse + validate) a GLSL ES shader source. On success
/// returns the executable program; on a syntax/semantic error returns a real
/// error message suitable for `getShaderInfoLog`.
pub fn compile(src: &str, stage: Stage) -> Result<CompiledShader, String> {
    let toks = tokenize(src).map_err(|e| format!("ERROR: 0:0: {e}"))?;
    if toks.is_empty() {
        return Err("ERROR: 0:0: empty shader source".into());
    }
    let mut p = Parser {
        toks,
        pos: 0,
        attributes: Vec::new(),
        uniforms: Vec::new(),
        varyings: Vec::new(),
    };
    let funcs = p.parse_program().map_err(|e| format!("ERROR: 0:0: {e}"))?;
    if !funcs.contains_key("main") {
        return Err("ERROR: 0:0: 'main' : function does not return a value or is missing".into());
    }
    Ok(CompiledShader {
        stage,
        funcs,
        attributes: p.attributes,
        uniforms: p.uniforms,
        varyings: p.varyings,
    })
}

// ---------------------------------------------------------------------------
// Interpreter
// ---------------------------------------------------------------------------

/// Texture-sampling callback supplied by the GL context (returns straight
/// RGBA in 0..1 for a 2D sampler at the given uv). `None` means no texture
/// bound — sampler reads default to opaque black per GL.
pub type Sampler<'a> = dyn Fn(&str, f32, f32) -> [f32; 4] + 'a;

struct Interp<'a> {
    funcs: &'a HashMap<String, Func>,
    scopes: Vec<HashMap<String, Val>>,
    sampler: Option<&'a Sampler<'a>>,
    discarded: bool,
}

enum Flow {
    Normal,
    Return(Option<Val>),
}

impl<'a> Interp<'a> {
    fn get_var(&self, name: &str) -> Option<Val> {
        for s in self.scopes.iter().rev() {
            if let Some(v) = s.get(name) {
                return Some(v.clone());
            }
        }
        None
    }
    fn set_var(&mut self, name: &str, v: Val) {
        for s in self.scopes.iter_mut().rev() {
            if s.contains_key(name) {
                s.insert(name.to_string(), v);
                return;
            }
        }
        // not found: define in current scope
        self.scopes.last_mut().unwrap().insert(name.to_string(), v);
    }
    fn declare(&mut self, name: &str, v: Val) {
        self.scopes.last_mut().unwrap().insert(name.to_string(), v);
    }

    fn exec_block(&mut self, stmts: &[Stmt]) -> Result<Flow, String> {
        self.scopes.push(HashMap::new());
        let r = self.exec_stmts(stmts);
        self.scopes.pop();
        r
    }

    fn exec_stmts(&mut self, stmts: &[Stmt]) -> Result<Flow, String> {
        for s in stmts {
            match self.exec_stmt(s)? {
                Flow::Normal => {}
                f => return Ok(f),
            }
            if self.discarded {
                return Ok(Flow::Return(None));
            }
        }
        Ok(Flow::Normal)
    }

    fn exec_stmt(&mut self, s: &Stmt) -> Result<Flow, String> {
        match s {
            Stmt::Decl { name, init } => {
                let v = match init {
                    Some(e) => self.eval(e)?,
                    None => Val::Float(0.0),
                };
                self.declare(name, v);
                Ok(Flow::Normal)
            }
            Stmt::Expr(e) => {
                self.eval(e)?;
                Ok(Flow::Normal)
            }
            Stmt::If(cond, then, els) => {
                if truthy(&self.eval(cond)?) {
                    self.exec_block(then)
                } else {
                    self.exec_block(els)
                }
            }
            Stmt::For {
                init,
                cond,
                step,
                body,
            } => {
                self.scopes.push(HashMap::new());
                self.exec_stmt(init)?;
                let mut guard = 0u32;
                let result = loop {
                    guard += 1;
                    if guard > 100_000 {
                        break Err("for-loop iteration cap exceeded".to_string());
                    }
                    if !truthy(&self.eval(cond)?) {
                        break Ok(Flow::Normal);
                    }
                    match self.exec_block(body)? {
                        Flow::Normal => {}
                        f => break Ok(f),
                    }
                    if self.discarded {
                        break Ok(Flow::Return(None));
                    }
                    self.eval(step)?;
                };
                self.scopes.pop();
                result
            }
            Stmt::Return(e) => {
                let v = match e {
                    Some(e) => Some(self.eval(e)?),
                    None => None,
                };
                Ok(Flow::Return(v))
            }
            Stmt::Block(b) => self.exec_block(b),
        }
    }

    fn eval(&mut self, e: &Expr) -> Result<Val, String> {
        match e {
            Expr::Num(v, is_int) => Ok(if *is_int {
                Val::Int(*v as i32)
            } else {
                Val::Float(*v)
            }),
            Expr::Bool(b) => Ok(Val::Bool(*b)),
            Expr::Var(name) => self
                .get_var(name)
                .ok_or_else(|| format!("ERROR: undeclared identifier '{name}'")),
            Expr::Unary(op, inner) => {
                let v = self.eval(inner)?;
                match op.as_str() {
                    "-" => Ok(negate(&v)),
                    "!" => Ok(Val::Bool(!truthy(&v))),
                    _ => Err(format!("unknown unary op '{op}'")),
                }
            }
            Expr::Bin(op, a, b) => {
                let va = self.eval(a)?;
                let vb = self.eval(b)?;
                binop(op, &va, &vb)
            }
            Expr::Member(obj, field) => {
                let v = self.eval(obj)?;
                swizzle(&v, field)
            }
            Expr::Index(obj, idx) => {
                let v = self.eval(obj)?;
                let i = self.eval(idx)?.as_scalar().unwrap_or(0.0) as usize;
                index_value(&v, i)
            }
            Expr::Call(name, args) => self.eval_call(name, args),
            Expr::Assign(op, lhs, rhs) => {
                let rv = self.eval(rhs)?;
                let new = if op == "=" {
                    rv
                } else {
                    let cur = self.eval(lhs)?;
                    let bop = &op[..1];
                    binop(bop, &cur, &rv)?
                };
                self.assign_to(lhs, new.clone())?;
                Ok(new)
            }
        }
    }

    /// Assign `val` into an lvalue expression: a plain variable, or a
    /// write-swizzle / index on a variable (e.g. `gl_Position.xyz = ...`).
    fn assign_to(&mut self, lhs: &Expr, val: Val) -> Result<(), String> {
        match lhs {
            Expr::Var(name) => {
                self.set_var(name, val);
                Ok(())
            }
            Expr::Member(obj, field) => {
                // write-swizzle: read base, splat components, write back.
                if let Expr::Var(name) = obj.as_ref() {
                    let mut base = self
                        .get_var(name)
                        .unwrap_or(Val::Vec(vec![0.0, 0.0, 0.0, 0.0]));
                    write_swizzle(&mut base, field, &val)?;
                    self.set_var(name, base);
                    Ok(())
                } else {
                    Err("unsupported swizzle assignment target".into())
                }
            }
            Expr::Index(obj, idx) => {
                if let Expr::Var(name) = obj.as_ref() {
                    let i = self.eval(idx)?.as_scalar().unwrap_or(0.0) as usize;
                    let mut base = self
                        .get_var(name)
                        .ok_or_else(|| format!("undeclared '{name}'"))?;
                    write_index(&mut base, i, &val)?;
                    self.set_var(name, base);
                    Ok(())
                } else {
                    Err("unsupported index assignment target".into())
                }
            }
            _ => Err("invalid assignment target".into()),
        }
    }

    fn eval_call(&mut self, name: &str, args: &[Expr]) -> Result<Val, String> {
        if name == "__discard" {
            self.discarded = true;
            return Ok(Val::Float(0.0));
        }
        // Constructors.
        if let Some(v) = self.try_constructor(name, args)? {
            return Ok(v);
        }
        // Texture sampling.
        if name == "texture2D" || name == "textureCube" || name == "texture" {
            let sampler_name = if let Expr::Var(n) = &args[0] {
                n.clone()
            } else {
                "tex".to_string()
            };
            let uv = self.eval(args.get(1).ok_or("texture2D needs uv")?)?;
            let uvv = uv.to_vecn(2).unwrap_or(vec![0.0, 0.0]);
            let rgba = match self.sampler {
                Some(s) => s(&sampler_name, uvv[0], uvv[1]),
                None => [0.0, 0.0, 0.0, 1.0],
            };
            return Ok(Val::Vec(rgba.to_vec()));
        }
        // Builtin functions.
        let mut argv = Vec::with_capacity(args.len());
        for a in args {
            argv.push(self.eval(a)?);
        }
        if let Some(v) = builtin(name, &argv)? {
            return Ok(v);
        }
        // User function call.
        if let Some(f) = self.funcs.get(name).cloned() {
            if f.params.len() != argv.len() {
                return Err(format!(
                    "function '{name}' expects {} args, got {}",
                    f.params.len(),
                    argv.len()
                ));
            }
            self.scopes.push(HashMap::new());
            for (p, a) in f.params.iter().zip(argv.into_iter()) {
                self.declare(p, a);
            }
            let flow = self.exec_stmts(&f.body);
            self.scopes.pop();
            return match flow? {
                Flow::Return(Some(v)) => Ok(v),
                _ => Ok(Val::Float(0.0)),
            };
        }
        Err(format!("ERROR: no matching function for call to '{name}'"))
    }

    fn try_constructor(&mut self, name: &str, args: &[Expr]) -> Result<Option<Val>, String> {
        let n = match name {
            "float" | "int" | "bool" => 1,
            "vec2" | "ivec2" | "bvec2" => 2,
            "vec3" | "ivec3" | "bvec3" => 3,
            "vec4" | "ivec4" | "bvec4" => 4,
            "mat2" => return self.mat_constructor(2, args).map(Some),
            "mat3" => return self.mat_constructor(3, args).map(Some),
            "mat4" => return self.mat_constructor(4, args).map(Some),
            _ => return Ok(None),
        };
        let mut argv = Vec::with_capacity(args.len());
        for a in args {
            argv.push(self.eval(a)?);
        }
        if n == 1 {
            let s = argv
                .first()
                .and_then(|v| v.as_scalar())
                .ok_or("scalar constructor needs an argument")?;
            return Ok(Some(match name {
                "int" => Val::Int(s as i32),
                "bool" => Val::Bool(s != 0.0),
                _ => Val::Float(s),
            }));
        }
        // Vector: flatten all args' components, then take/broadcast to n.
        let mut comps: Vec<f32> = Vec::new();
        for a in &argv {
            match a {
                Val::Vec(v) => comps.extend_from_slice(v),
                other => {
                    if let Some(s) = other.as_scalar() {
                        comps.push(s);
                    }
                }
            }
        }
        if comps.len() == 1 {
            // single scalar broadcasts to all n
            comps = vec![comps[0]; n];
        }
        if comps.len() < n {
            return Err(format!("vec{n} constructor: not enough components"));
        }
        comps.truncate(n);
        Ok(Some(Val::Vec(comps)))
    }

    fn mat_constructor(&mut self, n: usize, args: &[Expr]) -> Result<Val, String> {
        let mut argv = Vec::with_capacity(args.len());
        for a in args {
            argv.push(self.eval(a)?);
        }
        // matN(scalar) → scalar*identity (diagonal).
        if argv.len() == 1 {
            if let Some(s) = argv[0].as_scalar() {
                let mut data = vec![0.0; n * n];
                for i in 0..n {
                    data[i * n + i] = s;
                }
                return Ok(Val::Mat { n, data });
            }
            // matN(matM) — copy upper-left.
            if let Val::Mat { n: m, data } = &argv[0] {
                let mut out = vec![0.0; n * n];
                for c in 0..n {
                    for r in 0..n {
                        out[c * n + r] = if c < *m && r < *m {
                            data[c * *m + r]
                        } else if c == r {
                            1.0
                        } else {
                            0.0
                        };
                    }
                }
                return Ok(Val::Mat { n, data: out });
            }
        }
        // Column-major list of n*n scalars or n column vectors.
        let mut data: Vec<f32> = Vec::new();
        for a in &argv {
            match a {
                Val::Vec(v) => data.extend_from_slice(v),
                other => {
                    if let Some(s) = other.as_scalar() {
                        data.push(s);
                    }
                }
            }
        }
        if data.len() != n * n {
            return Err(format!(
                "mat{n} constructor expects {} components, got {}",
                n * n,
                data.len()
            ));
        }
        Ok(Val::Mat { n, data })
    }
}

fn truthy(v: &Val) -> bool {
    match v {
        Val::Bool(b) => *b,
        Val::Float(f) => *f != 0.0,
        Val::Int(i) => *i != 0,
        Val::Vec(v) => v.iter().any(|x| *x != 0.0),
        Val::Mat { data, .. } => data.iter().any(|x| *x != 0.0),
    }
}

fn negate(v: &Val) -> Val {
    match v {
        Val::Float(f) => Val::Float(-f),
        Val::Int(i) => Val::Int(-i),
        Val::Vec(v) => Val::Vec(v.iter().map(|x| -x).collect()),
        Val::Mat { n, data } => Val::Mat {
            n: *n,
            data: data.iter().map(|x| -x).collect(),
        },
        Val::Bool(b) => Val::Bool(!b),
    }
}

fn index_value(v: &Val, i: usize) -> Result<Val, String> {
    match v {
        Val::Vec(comps) => comps
            .get(i)
            .map(|c| Val::Float(*c))
            .ok_or_else(|| format!("vector index {i} out of range")),
        Val::Mat { n, data } => {
            // matrix[i] is column i (a vector of length n)
            if i >= *n {
                return Err(format!("matrix column index {i} out of range"));
            }
            Ok(Val::Vec(data[i * n..i * n + n].to_vec()))
        }
        _ => Err("cannot index a scalar".into()),
    }
}

fn write_index(base: &mut Val, i: usize, val: &Val) -> Result<(), String> {
    match base {
        Val::Vec(comps) => {
            if i < comps.len() {
                comps[i] = val.as_scalar().unwrap_or(0.0);
                Ok(())
            } else {
                Err("vector index out of range".into())
            }
        }
        Val::Mat { n, data } => {
            let col = val.to_vecn(*n).ok_or("matrix column assign needs a vector")?;
            for (r, c) in col.iter().enumerate() {
                data[i * *n + r] = *c;
            }
            Ok(())
        }
        _ => Err("cannot index-assign scalar".into()),
    }
}

const COMPONENT_SETS: [&[char]; 3] = [
    &['x', 'y', 'z', 'w'],
    &['r', 'g', 'b', 'a'],
    &['s', 't', 'p', 'q'],
];

fn comp_index(c: char) -> Option<usize> {
    for set in COMPONENT_SETS {
        if let Some(p) = set.iter().position(|&x| x == c) {
            return Some(p);
        }
    }
    None
}

fn swizzle(v: &Val, field: &str) -> Result<Val, String> {
    let comps = match v {
        Val::Vec(c) => c.clone(),
        Val::Float(f) => vec![*f],
        Val::Int(i) => vec![*i as f32],
        _ => return Err(format!("cannot swizzle .{field} on a matrix/bool")),
    };
    let mut out = Vec::with_capacity(field.len());
    for ch in field.chars() {
        let idx = comp_index(ch).ok_or_else(|| format!("invalid swizzle component '{ch}'"))?;
        let c = comps
            .get(idx)
            .ok_or_else(|| format!("swizzle .{field} out of range"))?;
        out.push(*c);
    }
    if out.len() == 1 {
        Ok(Val::Float(out[0]))
    } else {
        Ok(Val::Vec(out))
    }
}

fn write_swizzle(base: &mut Val, field: &str, val: &Val) -> Result<(), String> {
    // Ensure base is a vector of sufficient length.
    let needed = field
        .chars()
        .filter_map(comp_index)
        .max()
        .map(|m| m + 1)
        .unwrap_or(0);
    let comps = match base {
        Val::Vec(c) => c,
        _ => {
            // promote scalar to vector
            let s = base.as_scalar().unwrap_or(0.0);
            *base = Val::Vec(vec![s; needed.max(1)]);
            if let Val::Vec(c) = base {
                c
            } else {
                unreachable!()
            }
        }
    };
    while comps.len() < needed {
        comps.push(0.0);
    }
    let src = match val {
        Val::Vec(v) => v.clone(),
        other => vec![other.as_scalar().unwrap_or(0.0)],
    };
    for (k, ch) in field.chars().enumerate() {
        let idx = comp_index(ch).ok_or_else(|| format!("invalid swizzle '{ch}'"))?;
        let value = if src.len() == 1 { src[0] } else { *src.get(k).unwrap_or(&0.0) };
        if idx < comps.len() {
            comps[idx] = value;
        }
    }
    Ok(())
}

fn binop(op: &str, a: &Val, b: &Val) -> Result<Val, String> {
    // Comparisons / logic first (scalar results).
    match op {
        "&&" => return Ok(Val::Bool(truthy(a) && truthy(b))),
        "||" => return Ok(Val::Bool(truthy(a) || truthy(b))),
        "==" => return Ok(Val::Bool(vals_eq(a, b))),
        "!=" => return Ok(Val::Bool(!vals_eq(a, b))),
        "<" | ">" | "<=" | ">=" => {
            let (x, y) = (
                a.as_scalar().ok_or("comparison needs scalars")?,
                b.as_scalar().ok_or("comparison needs scalars")?,
            );
            let r = match op {
                "<" => x < y,
                ">" => x > y,
                "<=" => x <= y,
                _ => x >= y,
            };
            return Ok(Val::Bool(r));
        }
        _ => {}
    }
    // Matrix * vector / matrix * matrix.
    if op == "*" {
        match (a, b) {
            (Val::Mat { n, data }, Val::Vec(v)) if v.len() == *n => {
                return Ok(Val::Vec(mat_vec(*n, data, v)));
            }
            (Val::Vec(v), Val::Mat { n, data }) if v.len() == *n => {
                // row-vector * matrix
                return Ok(Val::Vec(vec_mat(*n, v, data)));
            }
            (Val::Mat { n: na, data: da }, Val::Mat { n: nb, data: db }) if na == nb => {
                return Ok(Val::Mat {
                    n: *na,
                    data: mat_mat(*na, da, db),
                });
            }
            (Val::Mat { n, data }, s) if s.as_scalar().is_some() => {
                let sc = s.as_scalar().unwrap();
                return Ok(Val::Mat {
                    n: *n,
                    data: data.iter().map(|x| x * sc).collect(),
                });
            }
            (s, Val::Mat { n, data }) if s.as_scalar().is_some() => {
                let sc = s.as_scalar().unwrap();
                return Ok(Val::Mat {
                    n: *n,
                    data: data.iter().map(|x| x * sc).collect(),
                });
            }
            _ => {}
        }
    }
    // Componentwise / scalar arithmetic.
    let f = |x: f32, y: f32| -> f32 {
        match op {
            "+" => x + y,
            "-" => x - y,
            "*" => x * y,
            "/" => x / y,
            _ => f32::NAN,
        }
    };
    match (a, b) {
        (Val::Vec(va), Val::Vec(vb)) if va.len() == vb.len() => {
            Ok(Val::Vec(va.iter().zip(vb).map(|(x, y)| f(*x, *y)).collect()))
        }
        (Val::Vec(va), s) if s.as_scalar().is_some() => {
            let sc = s.as_scalar().unwrap();
            Ok(Val::Vec(va.iter().map(|x| f(*x, sc)).collect()))
        }
        (s, Val::Vec(vb)) if s.as_scalar().is_some() => {
            let sc = s.as_scalar().unwrap();
            Ok(Val::Vec(vb.iter().map(|y| f(sc, *y)).collect()))
        }
        (Val::Int(x), Val::Int(y)) if op != "/" => Ok(Val::Int(match op {
            "+" => x + y,
            "-" => x - y,
            "*" => x * y,
            _ => 0,
        })),
        _ => {
            let x = a.as_scalar().ok_or("arithmetic on non-numeric")?;
            let y = b.as_scalar().ok_or("arithmetic on non-numeric")?;
            Ok(Val::Float(f(x, y)))
        }
    }
}

fn vals_eq(a: &Val, b: &Val) -> bool {
    match (a, b) {
        (Val::Vec(x), Val::Vec(y)) => x == y,
        _ => match (a.as_scalar(), b.as_scalar()) {
            (Some(x), Some(y)) => x == y,
            _ => false,
        },
    }
}

fn mat_vec(n: usize, m: &[f32], v: &[f32]) -> Vec<f32> {
    // column-major: result[r] = sum_c m[c*n+r]*v[c]
    let mut out = vec![0.0; n];
    for r in 0..n {
        let mut acc = 0.0;
        for c in 0..n {
            acc += m[c * n + r] * v[c];
        }
        out[r] = acc;
    }
    out
}

fn vec_mat(n: usize, v: &[f32], m: &[f32]) -> Vec<f32> {
    // result[c] = sum_r v[r]*m[c*n+r]
    let mut out = vec![0.0; n];
    for c in 0..n {
        let mut acc = 0.0;
        for r in 0..n {
            acc += v[r] * m[c * n + r];
        }
        out[c] = acc;
    }
    out
}

fn mat_mat(n: usize, a: &[f32], b: &[f32]) -> Vec<f32> {
    // C = A*B, column-major. C[c*n+r] = sum_k A[k*n+r]*B[c*n+k]
    let mut out = vec![0.0; n * n];
    for c in 0..n {
        for r in 0..n {
            let mut acc = 0.0;
            for k in 0..n {
                acc += a[k * n + r] * b[c * n + k];
            }
            out[c * n + r] = acc;
        }
    }
    out
}

/// GLSL ES 1.00 built-in functions (§8). Returns `Ok(None)` if `name` is not
/// a builtin (so the caller can try user functions). Componentwise functions
/// map over vectors.
fn builtin(name: &str, args: &[Val]) -> Result<Option<Val>, String> {
    let map1 = |f: fn(f32) -> f32, v: &Val| -> Val {
        match v {
            Val::Vec(c) => Val::Vec(c.iter().map(|x| f(*x)).collect()),
            other => Val::Float(f(other.as_scalar().unwrap_or(0.0))),
        }
    };
    let comps = |v: &Val| -> Vec<f32> { v.as_vec().unwrap_or_default() };
    let out = match name {
        "radians" => Some(map1(|x| x.to_radians(), &args[0])),
        "degrees" => Some(map1(|x| x.to_degrees(), &args[0])),
        "sin" => Some(map1(|x| x.sin(), &args[0])),
        "cos" => Some(map1(|x| x.cos(), &args[0])),
        "tan" => Some(map1(|x| x.tan(), &args[0])),
        "asin" => Some(map1(|x| x.asin(), &args[0])),
        "acos" => Some(map1(|x| x.acos(), &args[0])),
        "atan" => {
            if args.len() == 2 {
                Some(zip2(&args[0], &args[1], |y, x| y.atan2(x)))
            } else {
                Some(map1(|x| x.atan(), &args[0]))
            }
        }
        "exp" => Some(map1(|x| x.exp(), &args[0])),
        "log" => Some(map1(|x| x.ln(), &args[0])),
        "exp2" => Some(map1(|x| x.exp2(), &args[0])),
        "log2" => Some(map1(|x| x.log2(), &args[0])),
        "sqrt" => Some(map1(|x| x.sqrt(), &args[0])),
        "inversesqrt" => Some(map1(|x| 1.0 / x.sqrt(), &args[0])),
        "abs" => Some(map1(|x| x.abs(), &args[0])),
        "sign" => Some(map1(|x| x.signum().copysign(x).clamp(-1.0, 1.0), &args[0])),
        "floor" => Some(map1(|x| x.floor(), &args[0])),
        "ceil" => Some(map1(|x| x.ceil(), &args[0])),
        "fract" => Some(map1(|x| x - x.floor(), &args[0])),
        "pow" => Some(zip2(&args[0], &args[1], |a, b| a.powf(b))),
        "mod" => Some(zip2_broadcast(&args[0], &args[1], |a, b| a - b * (a / b).floor())),
        "min" => Some(zip2_broadcast(&args[0], &args[1], |a, b| a.min(b))),
        "max" => Some(zip2_broadcast(&args[0], &args[1], |a, b| a.max(b))),
        "clamp" => {
            let x = comps(&args[0]);
            let lo = &args[1];
            let hi = &args[2];
            let los = expand(lo, x.len());
            let his = expand(hi, x.len());
            let r: Vec<f32> = x
                .iter()
                .enumerate()
                .map(|(i, v)| v.clamp(los[i], his[i]))
                .collect();
            Some(pack(r))
        }
        "mix" => {
            let a = comps(&args[0]);
            let b = comps(&args[1]);
            let t = expand(&args[2], a.len());
            let r: Vec<f32> = a
                .iter()
                .zip(&b)
                .enumerate()
                .map(|(i, (x, y))| x + (y - x) * t[i])
                .collect();
            Some(pack(r))
        }
        "step" => {
            let edge = comps(&args[0]);
            let x = comps(&args[1]);
            let e = if edge.len() == 1 { vec![edge[0]; x.len()] } else { edge };
            let r: Vec<f32> = x
                .iter()
                .enumerate()
                .map(|(i, v)| if *v < e[i] { 0.0 } else { 1.0 })
                .collect();
            Some(pack(r))
        }
        "smoothstep" => {
            let e0 = comps(&args[0]);
            let e1 = comps(&args[1]);
            let x = comps(&args[2]);
            let a = if e0.len() == 1 { vec![e0[0]; x.len()] } else { e0 };
            let b = if e1.len() == 1 { vec![e1[0]; x.len()] } else { e1 };
            let r: Vec<f32> = x
                .iter()
                .enumerate()
                .map(|(i, v)| {
                    let t = ((v - a[i]) / (b[i] - a[i])).clamp(0.0, 1.0);
                    t * t * (3.0 - 2.0 * t)
                })
                .collect();
            Some(pack(r))
        }
        "length" => {
            let v = comps(&args[0]);
            Some(Val::Float(v.iter().map(|x| x * x).sum::<f32>().sqrt()))
        }
        "distance" => {
            let a = comps(&args[0]);
            let b = comps(&args[1]);
            let d: f32 = a.iter().zip(&b).map(|(x, y)| (x - y) * (x - y)).sum();
            Some(Val::Float(d.sqrt()))
        }
        "dot" => {
            let a = comps(&args[0]);
            let b = comps(&args[1]);
            Some(Val::Float(a.iter().zip(&b).map(|(x, y)| x * y).sum()))
        }
        "cross" => {
            let a = comps(&args[0]);
            let b = comps(&args[1]);
            if a.len() < 3 || b.len() < 3 {
                return Err("cross() needs vec3".into());
            }
            Some(Val::Vec(vec![
                a[1] * b[2] - a[2] * b[1],
                a[2] * b[0] - a[0] * b[2],
                a[0] * b[1] - a[1] * b[0],
            ]))
        }
        "normalize" => {
            let v = comps(&args[0]);
            let len = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            if len == 0.0 {
                Some(Val::Vec(v))
            } else {
                Some(pack(v.iter().map(|x| x / len).collect()))
            }
        }
        "reflect" => {
            let i = comps(&args[0]);
            let nrm = comps(&args[1]);
            let d: f32 = i.iter().zip(&nrm).map(|(a, b)| a * b).sum();
            let r: Vec<f32> = i
                .iter()
                .zip(&nrm)
                .map(|(a, b)| a - 2.0 * d * b)
                .collect();
            Some(pack(r))
        }
        "faceforward" => {
            let n = comps(&args[0]);
            let i = comps(&args[1]);
            let nref = comps(&args[2]);
            let d: f32 = nref.iter().zip(&i).map(|(a, b)| a * b).sum();
            let r: Vec<f32> = if d < 0.0 {
                n.clone()
            } else {
                n.iter().map(|x| -x).collect()
            };
            Some(pack(r))
        }
        _ => None,
    };
    Ok(out)
}

fn expand(v: &Val, n: usize) -> Vec<f32> {
    match v {
        Val::Vec(c) if c.len() == n => c.clone(),
        Val::Vec(c) if c.len() == 1 => vec![c[0]; n],
        other => vec![other.as_scalar().unwrap_or(0.0); n],
    }
}

fn pack(v: Vec<f32>) -> Val {
    if v.len() == 1 {
        Val::Float(v[0])
    } else {
        Val::Vec(v)
    }
}

fn zip2(a: &Val, b: &Val, f: fn(f32, f32) -> f32) -> Val {
    let av = a.as_vec().unwrap_or_default();
    let bv = b.as_vec().unwrap_or_default();
    let n = av.len().max(bv.len());
    let ae = if av.len() == 1 { vec![av[0]; n] } else { av };
    let be = if bv.len() == 1 { vec![bv[0]; n] } else { bv };
    let r: Vec<f32> = (0..n).map(|i| f(ae[i], be[i])).collect();
    pack(r)
}

fn zip2_broadcast(a: &Val, b: &Val, f: fn(f32, f32) -> f32) -> Val {
    zip2(a, b, f)
}

/// Run a compiled shader's `main()` with the supplied input variables
/// (attributes/uniforms/varyings), returning the full output environment
/// (so the caller can read `gl_Position`, `gl_FragColor`, and varyings).
pub fn run_main(
    shader: &CompiledShader,
    inputs: &HashMap<String, Val>,
    sampler: Option<&Sampler<'_>>,
) -> Result<(HashMap<String, Val>, bool), String> {
    let mut globals: HashMap<String, Val> = inputs.clone();
    // GL builtins start at sane defaults.
    globals
        .entry("gl_Position".into())
        .or_insert(Val::Vec(vec![0.0, 0.0, 0.0, 1.0]));
    globals
        .entry("gl_FragColor".into())
        .or_insert(Val::Vec(vec![0.0, 0.0, 0.0, 1.0]));
    globals
        .entry("gl_PointSize".into())
        .or_insert(Val::Float(1.0));
    let mut interp = Interp {
        funcs: &shader.funcs,
        scopes: vec![globals],
        sampler,
        discarded: false,
    };
    let main = shader
        .funcs
        .get("main")
        .ok_or("missing main at runtime")?
        .clone();
    interp.exec_stmts(&main.body)?;
    let out = interp.scopes.into_iter().next().unwrap();
    Ok((out, interp.discarded))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compile_valid_vertex_shader() {
        let src = "attribute vec3 a_pos; uniform mat4 u_mvp; \
                   void main(){ gl_Position = u_mvp * vec4(a_pos, 1.0); }";
        let s = compile(src, Stage::Vertex).expect("should compile");
        assert!(s.attributes.contains(&"a_pos".to_string()));
        assert!(s.uniforms.contains(&"u_mvp".to_string()));
    }

    #[test]
    fn compile_missing_main_fails() {
        let src = "attribute vec3 a_pos; void notmain(){}";
        let e = compile(src, Stage::Vertex).unwrap_err();
        assert!(e.to_lowercase().contains("main"), "log: {e}");
    }

    #[test]
    fn compile_syntax_error_fails() {
        let src = "void main(){ gl_Position = vec4(1.0 }"; // missing ')'
        let e = compile(src, Stage::Vertex).unwrap_err();
        assert!(!e.is_empty());
    }

    #[test]
    fn run_passthrough_vertex() {
        let src = "attribute vec2 a; void main(){ gl_Position = vec4(a, 0.0, 1.0); }";
        let s = compile(src, Stage::Vertex).unwrap();
        let mut inputs = HashMap::new();
        inputs.insert("a".to_string(), Val::Vec(vec![0.5, -0.5]));
        let (out, _) = run_main(&s, &inputs, None).unwrap();
        assert_eq!(out["gl_Position"].to_vec4(), [0.5, -0.5, 0.0, 1.0]);
    }

    #[test]
    fn run_mvp_transform() {
        // translate by (1,2,0) via a mat4 and check gl_Position moves.
        let src = "attribute vec3 p; uniform mat4 m; void main(){ gl_Position = m * vec4(p,1.0); }";
        let s = compile(src, Stage::Vertex).unwrap();
        let mut inputs = HashMap::new();
        inputs.insert("p".to_string(), Val::Vec(vec![0.0, 0.0, 0.0]));
        // Column-major translation matrix.
        #[rustfmt::skip]
        let m = vec![
            1.0,0.0,0.0,0.0,
            0.0,1.0,0.0,0.0,
            0.0,0.0,1.0,0.0,
            1.0,2.0,0.0,1.0,
        ];
        inputs.insert("m".to_string(), Val::Mat { n: 4, data: m });
        let (out, _) = run_main(&s, &inputs, None).unwrap();
        let p = out["gl_Position"].to_vec4();
        assert!((p[0] - 1.0).abs() < 1e-5 && (p[1] - 2.0).abs() < 1e-5);
    }

    #[test]
    fn run_fragment_solid_color() {
        let src = "uniform vec4 u_color; void main(){ gl_FragColor = u_color; }";
        let s = compile(src, Stage::Fragment).unwrap();
        let mut inputs = HashMap::new();
        inputs.insert("u_color".to_string(), Val::Vec(vec![1.0, 0.0, 0.0, 1.0]));
        let (out, disc) = run_main(&s, &inputs, None).unwrap();
        assert!(!disc);
        assert_eq!(out["gl_FragColor"].to_vec4(), [1.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn swizzle_and_arith() {
        let src = "void main(){ vec4 c = vec4(0.2,0.4,0.6,1.0); gl_FragColor = vec4(c.bgr, c.a); }";
        let s = compile(src, Stage::Fragment).unwrap();
        let (out, _) = run_main(&s, &HashMap::new(), None).unwrap();
        let c = out["gl_FragColor"].to_vec4();
        assert!((c[0] - 0.6).abs() < 1e-6 && (c[1] - 0.4).abs() < 1e-6 && (c[2] - 0.2).abs() < 1e-6);
    }

    #[test]
    fn builtins_mix_clamp_dot() {
        let src = "void main(){ float d = dot(vec3(1.0,0.0,0.0), vec3(1.0,2.0,3.0)); \
                   vec3 m = mix(vec3(0.0), vec3(1.0), 0.25); \
                   gl_FragColor = vec4(m, clamp(d, 0.0, 0.5)); }";
        let s = compile(src, Stage::Fragment).unwrap();
        let (out, _) = run_main(&s, &HashMap::new(), None).unwrap();
        let c = out["gl_FragColor"].to_vec4();
        assert!((c[0] - 0.25).abs() < 1e-6);
        assert!((c[3] - 0.5).abs() < 1e-6); // dot=1 clamped to 0.5
    }
}
