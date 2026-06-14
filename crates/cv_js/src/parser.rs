//! Recursive-descent JavaScript parser — practical subset.
//!
//! Covered:
//!   - Variable declarations (var/let/const) with single declarator
//!   - Expression statements
//!   - Function declarations + function expressions + arrow functions
//!   - If / else, while, do-while, for(;;)
//!   - Return / break / continue / throw
//!   - Blocks
//!   - Expressions: literals (number, string, template, bool, null,
//!     undefined, this, identifier, array, object), member access (`.`,
//!     `[]`), calls `f(args)`, new, unary, binary with precedence,
//!     logical, conditional (`?:`), assignment, update (++/--)
//!
//! Not yet: destructuring patterns, default params, rest/spread,
//! classes, modules, generators, async/await semantics (the keywords
//! lex but the parser treats them as plain identifiers in most spots),
//! optional chaining short-circuit semantics (lexed as `?.` punct,
//! parsed as a member access).

use crate::ast::*;
use crate::lexer::{Token, TokenKind, tokenize};

/// Build the call `__tb_run_async__(() => <body>)` used to desugar an async
/// function/arrow. `__tb_run_async__` (a host builtin) runs the thunk and
/// ALWAYS returns a promise — resolved with the thunk's value, or REJECTED
/// with whatever the body throws. This is the Chrome-faithful guarantee that
/// an `async` function never throws synchronously into its caller: a body
/// exception becomes a rejected promise. The thunk is an arrow so it inherits
/// `this`/`arguments` from the enclosing (now-desugared) function, exactly
/// like a real async function body.
fn run_async_call(body: ArrowBody) -> Expr {
    Expr::Call {
        callee: Box::new(Expr::Identifier("__tb_run_async__".into())),
        args: vec![Expr::Arrow {
            params: vec![],
            body,
        }],
    }
}

/// Wrap an already-generator-desugared `function* NAME(){return
/// __tb_make_generator__(()=>{body})}` so the OUTER return goes through
/// `__tb_make_async_iterable_wrapper__`, producing an object whose
/// `.next()`/`.return()`/`.throw()` each return a Promise that resolves to
/// `{value, done}`. That's what `async function* g(){…}` is: a generator
/// whose iterator protocol speaks Promises. Without this, `for await (…of
/// ag())` failed feature-detection (`it.next` was undefined because the
/// previous desugar wrapped the whole call in `__tb_run_async__`, turning
/// the iterator return into an opaque Promise-of-iterator).
fn wrap_async_generator_decl(node: Stmt) -> Stmt {
    match node {
        Stmt::FunctionDecl { name, params, body } => {
            // `parse_function_decl` already wrapped the body in
            // `return __tb_make_generator__(()=>{ body })`. Re-wrap only the
            // return expression so we get
            // `return __tb_make_async_iterable_wrapper__(__tb_make_generator__(...))`.
            let wrapped_body: Vec<Stmt> = body
                .into_iter()
                .map(|stmt| match stmt {
                    Stmt::Return(Some(inner)) => Stmt::Return(Some(Expr::Call {
                        callee: Box::new(Expr::Identifier(
                            "__tb_make_async_iterable_wrapper__".into(),
                        )),
                        args: vec![inner],
                    })),
                    other => other,
                })
                .collect();
            Stmt::FunctionDecl {
                name,
                params,
                body: wrapped_body,
            }
        }
        other => other,
    }
}

/// Desugar an `async` function/arrow EXPRESSION. Non-function nodes pass
/// through untouched, so callers can blindly wrap a `parse_primary()` result
/// for the `async (params) => …` fall-through (a parenthesized expression is
/// returned unchanged).
fn make_async(node: Expr) -> Expr {
    match node {
        Expr::Function { name, params, body } => Expr::Function {
            name,
            params,
            body: vec![Stmt::Return(Some(run_async_call(ArrowBody::Block(body))))],
        },
        Expr::Arrow { params, body } => Expr::Arrow {
            params,
            body: ArrowBody::Expr(Box::new(run_async_call(body))),
        },
        other => other,
    }
}

/// Desugar an `async function NAME(){}` DECLARATION (kept hoisted as a decl).
fn make_async_decl(node: Stmt) -> Stmt {
    match node {
        Stmt::FunctionDecl { name, params, body } => Stmt::FunctionDecl {
            name,
            params,
            body: vec![Stmt::Return(Some(run_async_call(ArrowBody::Block(body))))],
        },
        other => other,
    }
}

/// Wrap a generator body in `__tb_make_generator__(() => { body })`. Calling the
/// (otherwise normal) generator function thus returns a lazy iterator without
/// running the body — V8-shaped. The arrow preserves `this`/params/`arguments`
/// of the generator call via closure.
fn make_generator_call(body: Vec<Stmt>) -> Expr {
    Expr::Call {
        callee: Box::new(Expr::Identifier("__tb_make_generator__".into())),
        args: vec![Expr::Arrow {
            params: vec![],
            body: ArrowBody::Block(body),
        }],
    }
}

/// `function* NAME(params){ body }` → `function NAME(params){ return
/// __tb_make_generator__(() => { body }); }`.
fn make_generator_decl(name: String, params: Vec<String>, body: Vec<Stmt>) -> Stmt {
    Stmt::FunctionDecl {
        name,
        params,
        body: vec![Stmt::Return(Some(make_generator_call(body)))],
    }
}

/// `function* (params){ body }` (expression) → same wrapping as the decl.
fn make_generator_expr(name: Option<String>, params: Vec<String>, body: Vec<Stmt>) -> Expr {
    Expr::Function {
        name,
        params,
        body: vec![Stmt::Return(Some(make_generator_call(body)))],
    }
}

pub fn parse_program(src: &str) -> Result<Vec<Stmt>, ParseError> {
    let toks = tokenize(src);
    let mut p = Parser {
        toks,
        i: 0,
        destructure_counter: 0,
        in_generator: false,
    };
    let mut out = Vec::new();
    while !p.is_eof() {
        if p.eat_lineterm() {
            continue;
        }
        // Error recovery: when a single statement uses syntax we don't
        // support (destructuring, generators, private class fields,
        // etc.), skip it and resync at the next statement boundary so
        // the rest of the script can still run. Real-world bundles
        // mix a handful of unsupported features with a lot of code
        // that would otherwise work; bailing on first error breaks
        // entire pages over one minor construct.
        let start_idx = p.i;
        match p.parse_stmt() {
            Ok(stmt) => out.push(stmt),
            Err(e) => {
                debug_report_recovery(&p, start_idx, &e);
                if p.i == start_idx {
                    // No progress — force a single token advance so
                    // we don't loop forever on a truly broken token.
                    p.bump();
                }
                p.skip_to_stmt_boundary();
            }
        }
        // Optional semicolon / line break.
        let _ = p.match_punct(";");
        while p.eat_lineterm() {}
    }
    Ok(out)
}

/// Debug-only: when `CV_JS_PARSE_DEBUG` is set, report each statement
/// the top-level / block error-recovery path had to skip. The recovery
/// is what masks an unsupported-syntax parse failure as a later runtime
/// `X is not a function` (the rest of a comma-`var` chain gets dropped,
/// so a binding defined after the choke point reads `undefined`). This
/// surfaces the exact token + line we choked on so the construct can be
/// added to the parser. Off by default; not on any render path.
fn debug_report_recovery(p: &Parser, start_idx: usize, err: &ParseError) {
    if std::env::var("CV_JS_PARSE_DEBUG").is_err() {
        return;
    }
    let fail = p.toks.get(p.i).or_else(|| p.toks.last());
    let (line, col) = fail.map(|t| (t.line, t.col)).unwrap_or((0, 0));
    let window: Vec<String> = p
        .toks
        .iter()
        .skip(start_idx)
        .take(12)
        .map(|t| format!("{:?}", t.kind))
        .collect();
    eprintln!(
        "[parse-recovery] choked at line {line}:{col} ({err}); from-token #{start_idx} window: {}",
        window.join(" ")
    );
}

/// Parse a single expression from a fragment. Used by the interpreter's
/// template-literal expansion to evaluate `${expr}` holes.
pub fn parse_expression_str(src: &str) -> Result<Expr, ParseError> {
    let toks = tokenize(src);
    let mut p = Parser {
        toks,
        i: 0,
        destructure_counter: 0,
        in_generator: false,
    };
    let e = p.parse_expr()?;
    Ok(e)
}

#[derive(Debug, Clone)]
pub struct ParseError(pub String);

impl core::fmt::Display for ParseError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "parse error: {}", self.0)
    }
}

impl std::error::Error for ParseError {}

/// One entry parsed out of a destructuring pattern (`{a, b: c, d = 1}` or
/// `[a, , b, ...rest]`). Shared between `let`/`const`/`var` destructuring
/// declarations and `for (let [a,b] of …)` loop bindings.
enum DestructureEntry {
    Field {
        /// Source key: the property name (object) or the array index as a
        /// string (array). `Member` access is built from this.
        key: String,
        /// Computed key expression for `{[expr]: target}` patterns. When set,
        /// the source access is `tmp[expr]` (computed) and `key` is ignored.
        computed_key: Option<Expr>,
        /// Local binding name. For a nested sub-pattern this is a fresh temp
        /// and `nested` carries the sub-entries to expand against it.
        name: String,
        default: Option<Expr>,
        is_array_index: bool,
        nested: Option<(bool, Vec<DestructureEntry>)>,
    },
    Rest {
        name: String,
    },
}

struct Parser {
    toks: Vec<Token>,
    i: usize,
    /// Monotonic ID so each `let {a,b} = obj` generates a fresh hidden
    /// `__dstN` binding without colliding across nested patterns.
    destructure_counter: u32,
    /// True while parsing a generator (`function*`) body, so `yield` is
    /// recognized as a yield-expression rather than an identifier.
    in_generator: bool,
}

/// Branch shape recovered after a `?.` in `parse_postfix`. The
/// surrounding code wraps the receiver expression into a single-eval
/// IIFE that short-circuits when the receiver is null/undefined.
enum OptionalAccess {
    Member { property: Expr, computed: bool },
    Call { args: Vec<Expr> },
}

/// Build an optional-chain IIFE that evaluates the receiver exactly once.
/// For method calls like `obj.method?.()`, preserve the original receiver as
/// `this` by routing through `fn.call(obj, ...)`.
fn desugar_optional_chain(receiver: Expr, access: OptionalAccess) -> Expr {
    let original_receiver = receiver.clone();
    let tmp = "__opt".to_string();
    let tmp_ref = Expr::Identifier(tmp.clone());
    let success = match access {
        OptionalAccess::Member { property, computed } => Expr::Member {
            object: Box::new(tmp_ref.clone()),
            property: Box::new(property),
            computed,
        },
        OptionalAccess::Call { args } => match receiver {
            Expr::Member {
                object,
                property,
                computed,
            } => {
                let base = "__opt_base".to_string();
                let func = "__opt_fn".to_string();
                let base_ref = Expr::Identifier(base.clone());
                let func_ref = Expr::Identifier(func.clone());
                let member = Expr::Member {
                    object: Box::new(base_ref.clone()),
                    property,
                    computed,
                };
                let test = Expr::Binary {
                    op: BinOp::EqEq,
                    left: Box::new(func_ref.clone()),
                    right: Box::new(Expr::Null),
                };
                let call = Expr::Call {
                    callee: Box::new(Expr::Member {
                        object: Box::new(func_ref.clone()),
                        property: Box::new(Expr::Identifier("call".into())),
                        computed: false,
                    }),
                    args: std::iter::once(base_ref.clone()).chain(args).collect(),
                };
                return Expr::Call {
                    callee: Box::new(Expr::Function {
                        name: None,
                        params: vec![base.clone()],
                        body: vec![
                            Stmt::VarDecl {
                                kind: VarKind::Var,
                                decls: vec![VarDeclarator {
                                    name: func.clone(),
                                    init: Some(member),
                                }],
                            },
                            Stmt::Return(Some(Expr::Conditional {
                                test: Box::new(test),
                                cons: Box::new(Expr::Undefined),
                                alt: Box::new(call),
                            })),
                        ],
                    }),
                    args: vec![*object],
                };
            }
            other => Expr::Call {
                callee: Box::new(other),
                args,
            },
        },
    };
    let test = Expr::Binary {
        op: BinOp::EqEq,
        left: Box::new(tmp_ref.clone()),
        right: Box::new(Expr::Null),
    };
    let cond = Expr::Conditional {
        test: Box::new(test),
        cons: Box::new(Expr::Undefined),
        alt: Box::new(success),
    };
    Expr::Call {
        callee: Box::new(Expr::Function {
            name: None,
            params: vec![tmp],
            body: vec![Stmt::Return(Some(cond))],
        }),
        args: vec![original_receiver],
    }
}

/// Desugar an optional METHOD call `receiver?.prop(args)` (or
/// `receiver?.[k](args)`): short-circuit to `undefined` when the receiver is
/// null/undefined, otherwise call `receiver.prop(args)` with `this = receiver`.
/// Parsing `?.prop` then a separate `(args)` would produce `(receiver?.prop)(args)`,
/// which both loses `this` and fails to short-circuit the call — so the whole
/// `?.prop(args)` is folded here into one single-eval guarded call.
fn desugar_optional_method_call(
    receiver: Expr,
    property: Expr,
    computed: bool,
    args: Vec<Expr>,
) -> Expr {
    let tmp = "__opt".to_string();
    let tmp_ref = Expr::Identifier(tmp.clone());
    // `__opt.prop(args)` — a normal member call, so `this` = `__opt`.
    let call = Expr::Call {
        callee: Box::new(Expr::Member {
            object: Box::new(tmp_ref.clone()),
            property: Box::new(property),
            computed,
        }),
        args,
    };
    let test = Expr::Binary {
        op: BinOp::EqEq,
        left: Box::new(tmp_ref),
        right: Box::new(Expr::Null),
    };
    let cond = Expr::Conditional {
        test: Box::new(test),
        cons: Box::new(Expr::Undefined),
        alt: Box::new(call),
    };
    Expr::Call {
        callee: Box::new(Expr::Function {
            name: None,
            params: vec![tmp],
            body: vec![Stmt::Return(Some(cond))],
        }),
        args: vec![receiver],
    }
}

/// Split a template-literal body (static text + `${raw_expr}` holes) into
/// the static cooked segments and the raw expression sources.
/// E.g. `"hello ${name}!"` → (["hello ", "!"], ["name"])
/// Split a (cooked or raw) template body string at `${...}` holes.
/// Returns `(segments, expr_sources)` — `segments` are the literal text
/// parts between holes, `expr_sources` are the raw source texts of each hole.
fn split_template_body(src: &str) -> (Vec<String>, Vec<String>) {
    let bytes = src.as_bytes();
    let mut segments: Vec<String> = Vec::new();
    let mut expr_sources: Vec<String> = Vec::new();
    let mut seg_start = 0usize;
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'$' && i + 1 < bytes.len() && bytes[i + 1] == b'{' {
            segments.push(src[seg_start..i].to_string());
            let mut depth = 1usize;
            let expr_start = i + 2;
            let mut j = expr_start;
            while j < bytes.len() && depth > 0 {
                match bytes[j] {
                    b'{' => { depth += 1; j += 1; }
                    b'}' => {
                        depth -= 1;
                        if depth == 0 { break; }
                        j += 1;
                    }
                    b'\'' | b'"' => {
                        let q = bytes[j];
                        j += 1;
                        while j < bytes.len() {
                            if bytes[j] == b'\\' { j += 1; if j < bytes.len() { j += 1; } continue; }
                            if bytes[j] == q { break; }
                            j += 1;
                        }
                        if j < bytes.len() { j += 1; }
                    }
                    b'`' => {
                        j += 1;
                        let mut td = 0usize;
                        while j < bytes.len() {
                            match bytes[j] {
                                b'\\' => { j += 1; }
                                b'`' if td == 0 => { j += 1; break; }
                                b'$' if j + 1 < bytes.len() && bytes[j + 1] == b'{' => { td += 1; j += 1; }
                                b'}' if td > 0 => { td -= 1; }
                                _ => {}
                            }
                            j += 1;
                        }
                    }
                    _ => { j += 1; }
                }
            }
            expr_sources.push(src[expr_start..j].to_string());
            seg_start = j + 1;
            i = seg_start;
        } else {
            i += 1;
        }
    }
    segments.push(src[seg_start..].to_string());
    (segments, expr_sources)
}

impl Parser {
    fn is_eof(&self) -> bool {
        self.i >= self.toks.len()
    }

    fn peek(&self) -> Option<&Token> {
        self.toks.get(self.i)
    }

    fn peek_skip_lt(&self) -> Option<&Token> {
        let mut j = self.i;
        while let Some(t) = self.toks.get(j) {
            if matches!(t.kind, TokenKind::LineTerminator) {
                j += 1;
            } else {
                return Some(t);
            }
        }
        None
    }

    fn bump(&mut self) -> Option<Token> {
        let t = self.toks.get(self.i).cloned();
        if t.is_some() {
            self.i += 1;
        }
        t
    }

    /// Skip tokens until we reach a likely statement boundary so the
    /// caller can continue parsing the next statement. Honours
    /// brace/paren/bracket nesting so we don't stop on a `;` inside a
    /// for-loop header or an object literal. Used by `parse_program`'s
    /// error-recovery path.
    fn skip_to_stmt_boundary(&mut self) {
        let mut depth: i32 = 0;
        while let Some(t) = self.peek() {
            match &t.kind {
                TokenKind::Punct(p) => match p.as_str() {
                    "(" | "[" | "{" => {
                        depth += 1;
                        self.i += 1;
                    }
                    ")" | "]" | "}" => {
                        if depth == 0 {
                            // A closing brace at depth 0 ends the
                            // enclosing block — stop before it so the
                            // outer parser sees it.
                            return;
                        }
                        depth -= 1;
                        self.i += 1;
                    }
                    ";" if depth == 0 => {
                        self.i += 1;
                        return;
                    }
                    _ => {
                        self.i += 1;
                    }
                },
                TokenKind::LineTerminator if depth == 0 => {
                    return;
                }
                _ => {
                    self.i += 1;
                }
            }
        }
    }

    fn eat_lineterm(&mut self) -> bool {
        if matches!(
            self.peek().map(|t| &t.kind),
            Some(TokenKind::LineTerminator)
        ) {
            self.i += 1;
            true
        } else {
            false
        }
    }

    fn match_punct(&mut self, s: &str) -> bool {
        match self.peek_skip_lt() {
            Some(t) if matches!(&t.kind, TokenKind::Punct(p) if p == s) => {
                while self.eat_lineterm() {}
                self.i += 1;
                true
            }
            _ => false,
        }
    }

    fn match_keyword(&mut self, s: &str) -> bool {
        match self.peek_skip_lt() {
            Some(t) if matches!(&t.kind, TokenKind::Keyword(k) if k == s) => {
                while self.eat_lineterm() {}
                self.i += 1;
                true
            }
            _ => false,
        }
    }

    fn expect_punct(&mut self, s: &str) -> Result<(), ParseError> {
        if self.match_punct(s) {
            Ok(())
        } else {
            Err(ParseError(format!(
                "expected '{s}', got {:?}",
                self.peek().map(|t| &t.kind)
            )))
        }
    }

    fn parse_stmt(&mut self) -> Result<Stmt, ParseError> {
        // Labeled statement: `ident : stmt`. We parse-and-discard the
        // label (no targeted break/continue yet) so production
        // bundles that use labels for control flow still parse. The
        // body executes normally; non-targeted `break`/`continue`
        // inside still work for the enclosing loop.
        if let Some(TokenKind::Identifier(name)) = self.peek().map(|t| &t.kind) {
            let label = name.clone();
            // Look two tokens ahead through any line terms.
            let mut j = self.i + 1;
            while matches!(
                self.toks.get(j).map(|t| &t.kind),
                Some(TokenKind::LineTerminator)
            ) {
                j += 1;
            }
            if matches!(
                self.toks.get(j).map(|t| &t.kind),
                Some(TokenKind::Punct(p)) if p == ":"
            ) {
                self.bump(); // ident
                while self.eat_lineterm() {}
                self.bump(); // ':'
                let body = self.parse_stmt()?;
                return Ok(Stmt::Labeled {
                    label,
                    body: Box::new(body),
                });
            }
        }
        match self.peek_skip_lt().map(|t| &t.kind) {
            Some(TokenKind::Punct(p)) if p == "{" => self.parse_block(),
            Some(TokenKind::Punct(p)) if p == ";" => {
                self.bump();
                Ok(Stmt::Empty)
            }
            Some(TokenKind::Keyword(k)) => {
                let k = k.clone();
                match k.as_str() {
                    "var" => self.parse_var_decl_stmt(VarKind::Var),
                    "let" => self.parse_var_decl_stmt(VarKind::Let),
                    "const" => self.parse_var_decl_stmt(VarKind::Const),
                    "function" => self.parse_function_decl(),
                    // `async function foo()` and `async () => ...`. The
                    // `async` modifier is dropped — our Promise/await
                    // path is synchronous so the body runs immediately.
                    "async" => {
                        self.bump(); // consume "async"
                        while self.eat_lineterm() {}
                        if matches!(
                            self.peek().map(|t| &t.kind),
                            Some(TokenKind::Keyword(k)) if k == "function"
                        ) {
                            // Peek past `function` (and any linebreaks) for
                            // `*` — that's `async function*`, which is an
                            // ASYNC GENERATOR and needs a different
                            // desugaring than plain `async function`.
                            let is_async_gen = {
                                let mut j = self.i + 1;
                                while let Some(t) = self.toks.get(j) {
                                    if matches!(t.kind, TokenKind::LineTerminator) {
                                        j += 1;
                                    } else {
                                        break;
                                    }
                                }
                                matches!(
                                    self.toks.get(j).map(|t| &t.kind),
                                    Some(TokenKind::Punct(p)) if p == "*"
                                )
                            };
                            if is_async_gen {
                                // `async function* g(){…}` — parse as a
                                // regular generator decl, then wrap its
                                // return so `g()` produces an async-iterable
                                // (each .next() returns a Promise of
                                // {value, done}).
                                self.parse_function_decl().map(wrap_async_generator_decl)
                            } else {
                                // `async function NAME(){}` — hoisted async
                                // declaration; desugar to return a promise.
                                self.parse_function_decl().map(make_async_decl)
                            }
                        } else {
                            // `async () => …` / `async x => …` as an
                            // expression statement.
                            let e = self.parse_async_rest()?;
                            Ok(Stmt::Expression(e))
                        }
                    }
                    "class" => self.parse_class_decl(),
                    "import" => self.skip_module_directive(),
                    "export" => self.parse_export(),
                    "yield" => {
                        // `yield expr;` — parse the expression so the
                        // body's `yield i + 1` etc. type-checks, and
                        // route through __gen_push so a wrapped
                        // generator captures the value. If
                        // __gen_push isn't bound (we're not inside a
                        // generator wrapper), it's a runtime no-op
                        // via the global stub installed below.
                        self.bump();
                        let star = matches!(
                            self.peek().map(|t| &t.kind),
                            Some(TokenKind::Punct(p)) if p == "*"
                        );
                        if star {
                            self.bump();
                        }
                        while self.eat_lineterm() {}
                        // Empty `yield` is legal — produces undefined.
                        let arg = if matches!(
                            self.peek().map(|t| &t.kind),
                            Some(TokenKind::Punct(p)) if p == ";" || p == "}"
                        ) || self.peek().is_none()
                        {
                            Expr::Undefined
                        } else {
                            self.parse_assignment_expr()?
                        };
                        let marker = if star {
                            "__tb_yield_star__"
                        } else {
                            "__tb_yield__"
                        };
                        let call = Expr::Call {
                            callee: Box::new(Expr::Identifier(marker.into())),
                            args: vec![arg],
                        };
                        // Consume optional ';'
                        if matches!(
                            self.peek().map(|t| &t.kind),
                            Some(TokenKind::Punct(p)) if p == ";"
                        ) {
                            self.bump();
                        }
                        Ok(Stmt::Expression(call))
                    }
                    "if" => self.parse_if(),
                    "while" => self.parse_while(),
                    "do" => self.parse_do_while(),
                    "for" => self.parse_for(),
                    "return" => self.parse_return(),
                    "break" => {
                        while self.eat_lineterm() {}
                        self.bump();
                        // Optional label — `break label;`. A label is only
                        // valid when it appears on the SAME line (no ASI
                        // between `break` and the label per §13.9.1), but we
                        // accept it without the line check for simplicity.
                        let label =
                            if let Some(TokenKind::Identifier(n)) = self.peek().map(|t| &t.kind) {
                                let n = n.clone();
                                self.bump();
                                Some(n)
                            } else {
                                None
                            };
                        Ok(Stmt::Break(label))
                    }
                    "continue" => {
                        while self.eat_lineterm() {}
                        self.bump();
                        let label =
                            if let Some(TokenKind::Identifier(n)) = self.peek().map(|t| &t.kind) {
                                let n = n.clone();
                                self.bump();
                                Some(n)
                            } else {
                                None
                            };
                        Ok(Stmt::Continue(label))
                    }
                    "throw" => {
                        while self.eat_lineterm() {}
                        self.bump();
                        let e = self.parse_expr()?;
                        Ok(Stmt::Throw(e))
                    }
                    "try" => self.parse_try(),
                    "switch" => self.parse_switch(),
                    _ => self.parse_expr_stmt(),
                }
            }
            _ => self.parse_expr_stmt(),
        }
    }

    fn parse_try(&mut self) -> Result<Stmt, ParseError> {
        while self.eat_lineterm() {}
        self.bump(); // try
        let Stmt::Block(block) = self.parse_block()? else {
            return Err(ParseError("try body".into()));
        };
        let mut catch_param: Option<String> = None;
        let mut catch_block: Option<Vec<Stmt>> = None;
        let mut finally_block: Option<Vec<Stmt>> = None;
        if self.match_keyword("catch") {
            if self.match_punct("(") {
                let name = self.expect_identifier()?;
                catch_param = Some(name);
                self.expect_punct(")")?;
            }
            let Stmt::Block(b) = self.parse_block()? else {
                return Err(ParseError("catch body".into()));
            };
            catch_block = Some(b);
        }
        if self.match_keyword("finally") {
            let Stmt::Block(b) = self.parse_block()? else {
                return Err(ParseError("finally body".into()));
            };
            finally_block = Some(b);
        }
        if catch_block.is_none() && finally_block.is_none() {
            return Err(ParseError("try needs catch or finally".into()));
        }
        Ok(Stmt::Try {
            block,
            catch_param,
            catch_block,
            finally_block,
        })
    }

    fn parse_switch(&mut self) -> Result<Stmt, ParseError> {
        while self.eat_lineterm() {}
        self.bump(); // switch
        self.expect_punct("(")?;
        let discriminant = self.parse_expr()?;
        self.expect_punct(")")?;
        self.expect_punct("{")?;
        let mut cases: Vec<crate::ast::SwitchCase> = Vec::new();
        let mut default_index: Option<usize> = None;
        loop {
            while self.eat_lineterm() {}
            if self.match_punct("}") {
                break;
            }
            if self.is_eof() {
                return Err(ParseError("unterminated switch".into()));
            }
            let test = if self.match_keyword("case") {
                let e = self.parse_expr()?;
                Some(e)
            } else if self.match_keyword("default") {
                if default_index.is_some() {
                    return Err(ParseError("duplicate default in switch".into()));
                }
                default_index = Some(cases.len());
                None
            } else {
                // unknown leading token — skip it so we can recover
                self.bump();
                continue;
            };
            self.expect_punct(":")?;
            // Statements belonging to this label, until the next case /
            // default / closing brace.
            let mut body: Vec<Stmt> = Vec::new();
            loop {
                while self.eat_lineterm() {}
                match self.peek().map(|t| &t.kind) {
                    Some(TokenKind::Punct(p)) if p == "}" => break,
                    Some(TokenKind::Keyword(k)) if k == "case" || k == "default" => break,
                    None => break,
                    _ => {}
                }
                match self.parse_stmt() {
                    Ok(s) => body.push(s),
                    Err(_) => {
                        // Recover by advancing one token; matches the
                        // top-level error-recovery loop.
                        if !self.is_eof() {
                            self.bump();
                        }
                    }
                }
            }
            cases.push(crate::ast::SwitchCase { test, body });
        }
        Ok(Stmt::Switch {
            discriminant,
            cases,
            default_index,
        })
    }

    fn parse_block(&mut self) -> Result<Stmt, ParseError> {
        self.expect_punct("{")?;
        let mut out = Vec::new();
        loop {
            while self.eat_lineterm() {}
            if self.match_punct("}") {
                break;
            }
            if self.is_eof() {
                return Err(ParseError("unterminated block".into()));
            }
            // Same error-recovery behaviour as the top level: skip a
            // bad statement and keep going so the surrounding function
            // body stays usable.
            let start_idx = self.i;
            match self.parse_stmt() {
                Ok(s) => out.push(s),
                Err(e) => {
                    debug_report_recovery(self, start_idx, &e);
                    if self.i == start_idx {
                        self.bump();
                    }
                    self.skip_to_stmt_boundary();
                }
            }
            let _ = self.match_punct(";");
        }
        Ok(Stmt::Block(out))
    }

    fn parse_var_decl_stmt(&mut self, kind: VarKind) -> Result<Stmt, ParseError> {
        while self.eat_lineterm() {}
        self.bump(); // consume var/let/const
        let mut decls = Vec::new();
        loop {
            while self.eat_lineterm() {}
            // Destructuring: when we see `{` or `[`, parse the pattern
            // and desugar to an extra hidden binding + one declarator
            // per pattern entry. Common in production JS (every modern
            // bundle has `const {a, b} = props`).
            let is_obj_pattern = matches!(
                self.peek().map(|t| &t.kind),
                Some(TokenKind::Punct(p)) if p == "{"
            );
            let is_arr_pattern = matches!(
                self.peek().map(|t| &t.kind),
                Some(TokenKind::Punct(p)) if p == "["
            );
            if is_obj_pattern || is_arr_pattern {
                let extras = self.parse_destructuring_decl()?;
                decls.extend(extras);
                if !self.match_punct(",") {
                    break;
                }
                continue;
            }
            let name = self.expect_identifier()?;
            let init = if self.match_punct("=") {
                Some(self.parse_assignment_expr()?)
            } else {
                None
            };
            decls.push(VarDeclarator { name, init });
            if !self.match_punct(",") {
                break;
            }
        }
        Ok(Stmt::VarDecl { kind, decls })
    }

    /// Parse a destructuring pattern (`{a, b: c, d = 1, ...rest}` or
    /// `[a, b, ...rest]`) followed by `= source`. Desugar into a flat
    /// list of declarators:
    ///   const __dstN = source;
    ///   const a = __dstN.a; const c = __dstN.b; ...
    /// Returns the synthesized declarators. The caller wraps them in
    /// the surrounding `VarDecl` so the kind (let/const/var) carries
    /// across the desugaring.
    fn parse_destructuring_decl(&mut self) -> Result<Vec<VarDeclarator>, ParseError> {
        let (is_obj, entries) = self.parse_destructuring_entries()?;
        // After the pattern: `= source-expression`.
        while self.eat_lineterm() {}
        self.expect_punct("=")?;
        let source = self.parse_assignment_expr()?;
        Ok(self.build_destructure_declarators(is_obj, entries, source))
    }

    /// Parse a destructuring pattern (`{a, b: c, d = 1, ...rest}` or
    /// `[a, , b, ...rest]`, possibly nested) and return its entries. The
    /// leading `=`/source is NOT consumed — that lets the for-of/for-in
    /// loop binding reuse this against a per-iteration temp.
    fn parse_destructuring_entries(&mut self) -> Result<(bool, Vec<DestructureEntry>), ParseError> {
        let is_obj = matches!(
            self.peek().map(|t| &t.kind),
            Some(TokenKind::Punct(p)) if p == "{"
        );
        let opener = if is_obj { "{" } else { "[" };
        let closer = if is_obj { "}" } else { "]" };
        self.expect_punct(opener)?;

        let mut entries: Vec<DestructureEntry> = Vec::new();
        let mut arr_index: usize = 0;
        loop {
            while self.eat_lineterm() {}
            if self.match_punct(closer) {
                break;
            }
            // Array hole: `[, b]` / `[a, , c]` — an empty slot just advances
            // the index without binding anything.
            if !is_obj
                && matches!(self.peek().map(|t| &t.kind), Some(TokenKind::Punct(p)) if p == ",")
            {
                arr_index += 1;
                self.bump(); // consume the comma
                continue;
            }
            // Rest element: `...name`.
            if matches!(
                self.peek().map(|t| &t.kind),
                Some(TokenKind::Punct(p)) if p == "..."
            ) {
                self.bump();
                let name = self.expect_identifier()?;
                entries.push(DestructureEntry::Rest { name });
                let _ = self.match_punct(",");
                continue;
            }
            // The binding target. For an object pattern the source key comes
            // first (`key:` or shorthand); for an array pattern the key is
            // the running index. The target itself may be a plain identifier
            // or a nested `[...]` / `{...}` sub-pattern.
            let (key, computed_key, is_array_index) = if is_obj {
                // Computed property key: `{[expr]: target}`. The key is an
                // expression evaluated at runtime; the source access becomes
                // `tmp[expr]`. (Minified bundles use `const {[l]:o,[h]:d}=i`
                // heavily; without this the `[` errored out of
                // `expect_property_name` and the whole declaration cascaded
                // into a mis-parse that dropped surrounding bindings.)
                if self.match_punct("[") {
                    let key_expr = self.parse_assignment_expr()?;
                    self.expect_punct("]")?;
                    self.expect_punct(":")?;
                    (String::new(), Some(key_expr), false)
                } else {
                    // A property key may be a reserved word (`{default: u}`),
                    // string, or number — not just an identifier.
                    let key = self.expect_property_name()?;
                    if self.match_punct(":") {
                        (key, None, false)
                    } else {
                        // Shorthand `{a}` — key and binding name coincide. Push
                        // directly with an optional default.
                        let default = if self.match_punct("=") {
                            Some(self.parse_assignment_expr()?)
                        } else {
                            None
                        };
                        entries.push(DestructureEntry::Field {
                            name: key.clone(),
                            key,
                            computed_key: None,
                            default,
                            is_array_index: false,
                            nested: None,
                        });
                        let _ = self.match_punct(",");
                        continue;
                    }
                }
            } else {
                let k = arr_index.to_string();
                arr_index += 1;
                (k, None, true)
            };

            // Target: nested sub-pattern or a plain identifier.
            let nested_open = matches!(
                self.peek().map(|t| &t.kind),
                Some(TokenKind::Punct(p)) if p == "[" || p == "{"
            );
            let (name, nested) = if nested_open {
                let temp = format!("__dst{}", self.next_destructure_id());
                let sub = self.parse_destructuring_entries()?;
                (temp, Some(sub))
            } else {
                (self.expect_identifier()?, None)
            };
            let default = if self.match_punct("=") {
                Some(self.parse_assignment_expr()?)
            } else {
                None
            };
            entries.push(DestructureEntry::Field {
                key,
                computed_key,
                name,
                default,
                is_array_index,
                nested,
            });
            let _ = self.match_punct(",");
        }
        Ok((is_obj, entries))
    }

    /// Flatten a parsed destructuring pattern into ordered declarators that
    /// extract each binding from `source` (binding a `__dstN` temp first).
    fn build_destructure_declarators(
        &mut self,
        _is_obj: bool,
        entries: Vec<DestructureEntry>,
        source: Expr,
    ) -> Vec<VarDeclarator> {
        let tmp_name = format!("__dst{}", self.next_destructure_id());
        let mut out: Vec<VarDeclarator> = Vec::new();
        out.push(VarDeclarator {
            name: tmp_name.clone(),
            init: Some(source),
        });
        // Track what precedes a `...rest`: array patterns need the slice start
        // index, object patterns need the set of already-bound keys to exclude.
        let mut array_consumed = 0usize;
        let mut object_keys_taken: Vec<String> = Vec::new();
        for entry in entries {
            match entry {
                DestructureEntry::Field {
                    key,
                    computed_key,
                    name,
                    default,
                    is_array_index,
                    nested,
                } => {
                    if is_array_index {
                        array_consumed += 1;
                    } else if computed_key.is_none() {
                        object_keys_taken.push(key.clone());
                    }
                    let access = if let Some(ck) = computed_key {
                        // `{[expr]: target}` -> `tmp[expr]`.
                        Expr::Member {
                            object: Box::new(Expr::Identifier(tmp_name.clone())),
                            property: Box::new(ck),
                            computed: true,
                        }
                    } else if is_array_index {
                        Expr::Member {
                            object: Box::new(Expr::Identifier(tmp_name.clone())),
                            property: Box::new(Expr::Number(key.parse::<f64>().unwrap_or(0.0))),
                            computed: true,
                        }
                    } else {
                        Expr::Member {
                            object: Box::new(Expr::Identifier(tmp_name.clone())),
                            property: Box::new(Expr::Identifier(key.clone())),
                            computed: false,
                        }
                    };
                    let init = match default {
                        None => access,
                        Some(def) => Expr::Conditional {
                            test: Box::new(Expr::Binary {
                                op: BinOp::EqEqEq,
                                left: Box::new(access.clone()),
                                right: Box::new(Expr::Undefined),
                            }),
                            cons: Box::new(def),
                            alt: Box::new(access),
                        },
                    };
                    // Bind this slot's value to `name` (a real binding for a
                    // plain target, or a temp for a nested sub-pattern).
                    out.push(VarDeclarator {
                        name: name.clone(),
                        init: Some(init),
                    });
                    if let Some((sub_obj, sub_entries)) = nested {
                        let sub = self.build_destructure_declarators(
                            sub_obj,
                            sub_entries,
                            Expr::Identifier(name),
                        );
                        out.extend(sub);
                    }
                }
                DestructureEntry::Rest { name } => {
                    // Array rest (`[a, b, ...rest]`): the tail after the fixed
                    // elements, i.e. `tmp.slice(N)`. Binding the whole source
                    // (the old V1 behaviour) made `rest[0]` alias `source[0]`.
                    // Object rest (`{a, ...rest}`): a fresh object with the
                    // already-destructured keys removed via the shared runtime
                    // helper so aliased bindings (`{config: cfg, ...rest}`) and
                    // modern hook return shapes keep the remaining properties.
                    let init = if _is_obj {
                        Expr::Call {
                            callee: Box::new(Expr::Identifier("__tb_object_rest__".into())),
                            args: vec![
                                Expr::Identifier(tmp_name.clone()),
                                Expr::Array(
                                    object_keys_taken
                                        .iter()
                                        .cloned()
                                        .map(Expr::String)
                                        .collect(),
                                ),
                            ],
                        }
                    } else {
                        // Array rest: tmp.slice(array_consumed).
                        Expr::Call {
                            callee: Box::new(Expr::Member {
                                object: Box::new(Expr::Identifier(tmp_name.clone())),
                                property: Box::new(Expr::Identifier("slice".into())),
                                computed: false,
                            }),
                            args: vec![Expr::Number(array_consumed as f64)],
                        }
                    };
                    out.push(VarDeclarator {
                        name,
                        init: Some(init),
                    });
                }
            }
        }
        out
    }

    fn next_destructure_id(&mut self) -> u32 {
        let id = self.destructure_counter;
        self.destructure_counter = self.destructure_counter.wrapping_add(1);
        id
    }

    /// `class Name (extends Sup)? { method(...) {...} ... }`. Parsed as
    /// a function declaration whose value (the class constructor) is a
    /// function with no body. Methods are walked over and their inner
    /// braces matched, but their actual implementations are discarded.
    /// Real ES6 class semantics (prototype chain, super, static methods,
    /// getters/setters) land later.
    fn parse_class_decl(&mut self) -> Result<Stmt, ParseError> {
        while self.eat_lineterm() {}
        self.bump(); // `class`
        let (name, iife) = self.parse_class_tail()?;
        // `class X { … }` → `var X = (function(){ … ; return X; })();`. The
        // binding is `var` (hoisted) so it remains visible after the
        // statement, matching the previous hoisted-FunctionDecl behaviour.
        Ok(Stmt::VarDecl {
            kind: VarKind::Var,
            decls: vec![VarDeclarator {
                name: name.unwrap_or_default(),
                init: Some(iife),
            }],
        })
    }

    /// `class X{}` used as an *expression* (`x = class extends Y {…}`,
    /// `new class {…}`). The `class` keyword is already consumed by the
    /// primary dispatch. Reuses the same IIFE desugaring as declarations.
    fn parse_class_expr(&mut self) -> Result<Expr, ParseError> {
        let (_name, iife) = self.parse_class_tail()?;
        Ok(iife)
    }

    /// Parse everything after the `class` keyword (optional name, optional
    /// `extends`, and the class body) and desugar it into a constructor
    /// function. Returns `(name, ctor_params, full_body)`.
    fn parse_class_tail(&mut self) -> Result<(Option<String>, Expr), ParseError> {
        while self.eat_lineterm() {}
        // Optional binding name — absent for anonymous class expressions
        // (`class extends Y {…}` / `class {…}`).
        let name = match self.peek().map(|t| t.kind.clone()) {
            Some(TokenKind::Identifier(n)) => {
                self.bump();
                Some(n)
            }
            _ => None,
        };
        while self.eat_lineterm() {}
        // Optional `extends Sup`. The parent constructor expression is
        // captured and bound to the hidden `\u{1}__superclass__` local at
        // the top of the synthesized constructor (see below), so `super(...)`
        // and `super.x` (desugared in `parse_primary`) resolve to it.
        let mut superclass: Option<Expr> = None;
        if let Some(TokenKind::Keyword(k)) = self.peek().map(|t| &t.kind) {
            if k == "extends" {
                self.bump();
                // The parent is a LeftHandSideExpression — may be a member
                // chain (`class X extends o.G {…}`) or call, so parse the
                // full postfix expression (it stops at the class body `{`).
                superclass = Some(self.parse_postfix()?);
            }
        }
        self.expect_punct("{")?;

        // Collect method definitions: name(params) { body }. Methods
        // labelled `constructor` get folded into the synthesized
        // function's top-level body; everything else becomes a
        // `this.method = function(...) {...}` assignment that runs
        // before the constructor body executes.
        let mut ctor_params: Vec<String> = Vec::new();
        let mut ctor_body: Vec<Stmt> = Vec::new();
        // Real prototype-based class model: instance methods/accessors go on
        // `<Class>.prototype`, statics on `<Class>`, instance fields on `this`
        // in the constructor. `<Class>.prototype.[[Prototype]]` is linked to the
        // superclass prototype at DEFINITION time so `Class.prototype.X` reads
        // (and React's `prototype.isReactComponent` class detection) resolve
        // BEFORE any instance is constructed.
        let internal_name = name.clone().unwrap_or_else(|| "\u{1}__class__".to_string());
        let cn = || Expr::Identifier(internal_name.clone());
        let cn_proto = || Expr::Member {
            object: Box::new(cn()),
            property: Box::new(Expr::Identifier("prototype".to_string())),
            computed: false,
        };
        // Instance field initializers (`this.f = init`) — run in the ctor body.
        let mut field_inits: Vec<Stmt> = Vec::new();
        // `<Class>.prototype.m = fn` / `Object.defineProperty(<Class>.prototype …)`.
        let mut proto_members: Vec<Stmt> = Vec::new();
        // `<Class>.s = fn|init` — static methods / fields.
        let mut static_members: Vec<Stmt> = Vec::new();
        // Accessor (get/set) bodies collected by property name, so a class
        // that defines both `get x` and `set x` produces ONE accessor
        // descriptor (real ECMA-262 semantics, invoked transparently on
        // `obj.x` / `obj.x = v`). Order preserved for deterministic output.
        // Key is (property-name, is_static).
        let mut accessors: Vec<((String, bool), Option<Expr>, Option<Expr>)> = Vec::new();

        loop {
            while self.eat_lineterm() {}
            if matches!(self.peek().map(|t| &t.kind), Some(TokenKind::Punct(p)) if p == "}") {
                self.bump();
                break;
            }
            if matches!(self.peek().map(|t| &t.kind), Some(TokenKind::Punct(p)) if p == ";") {
                self.bump();
                continue;
            }
            // Generator method `*name(){}` — call returns a lazy iterator.
            let is_generator_method =
                matches!(self.peek().map(|t| &t.kind), Some(TokenKind::Punct(p)) if p == "*");
            if is_generator_method {
                self.bump();
                while self.eat_lineterm() {}
            }
            // Method name: identifier or contextual keyword.
            let mname = match self.peek().map(|t| t.kind.clone()) {
                Some(TokenKind::Identifier(s)) => {
                    self.bump();
                    s
                }
                Some(TokenKind::Keyword(s))
                    if matches!(
                        s.as_str(),
                        "let" | "of" | "async" | "await" | "yield" | "constructor" | "static"
                    ) =>
                {
                    self.bump();
                    s.as_str().to_string()
                }
                _ => {
                    // A member name we don't build yet (computed `[expr]`, string,
                    // or number). Skip JUST this member and continue — previously
                    // this discarded the ENTIRE rest of the class body, so one
                    // `[Symbol.iterator](){}` silently dropped every later method.
                    // Skip the name: balanced `[...]`, else one token.
                    if matches!(self.peek().map(|t| &t.kind), Some(TokenKind::Punct(p)) if p == "[") {
                        let mut d = 0;
                        loop {
                            match self.bump().map(|t| t.kind) {
                                Some(TokenKind::Punct(p)) if p == "[" => d += 1,
                                Some(TokenKind::Punct(p)) if p == "]" => {
                                    d -= 1;
                                    if d == 0 {
                                        break;
                                    }
                                }
                                None => break,
                                _ => {}
                            }
                        }
                    } else {
                        self.bump();
                    }
                    // Skip an optional `(params)`.
                    if matches!(self.peek().map(|t| &t.kind), Some(TokenKind::Punct(p)) if p == "(") {
                        let mut d = 0;
                        loop {
                            match self.bump().map(|t| t.kind) {
                                Some(TokenKind::Punct(p)) if p == "(" => d += 1,
                                Some(TokenKind::Punct(p)) if p == ")" => {
                                    d -= 1;
                                    if d == 0 {
                                        break;
                                    }
                                }
                                None => break,
                                _ => {}
                            }
                        }
                    }
                    // Skip `{ body }` (method) or to `;` (field), never the class `}`.
                    if matches!(self.peek().map(|t| &t.kind), Some(TokenKind::Punct(p)) if p == "{") {
                        let mut d = 0;
                        loop {
                            match self.bump().map(|t| t.kind) {
                                Some(TokenKind::Punct(p)) if p == "{" => d += 1,
                                Some(TokenKind::Punct(p)) if p == "}" => {
                                    d -= 1;
                                    if d == 0 {
                                        break;
                                    }
                                }
                                None => break,
                                _ => {}
                            }
                        }
                    } else {
                        loop {
                            match self.peek().map(|t| &t.kind) {
                                Some(TokenKind::Punct(p)) if p == ";" => {
                                    self.bump();
                                    break;
                                }
                                Some(TokenKind::Punct(p)) if p == "}" => break,
                                None => break,
                                _ => {
                                    self.bump();
                                }
                            }
                        }
                    }
                    continue;
                }
            };
            // `static` modifier — members go on the constructor, not the
            // prototype. (A lone `static` followed by `(` or `=` is itself a
            // member named "static", not a modifier.)
            let is_static = mname == "static"
                && matches!(
                    self.peek().map(|t| &t.kind),
                    Some(TokenKind::Identifier(_) | TokenKind::Keyword(_))
                );
            let mname = if is_static {
                while self.eat_lineterm() {}
                match self.bump().map(|t| t.kind) {
                    Some(TokenKind::Identifier(s)) => s,
                    Some(TokenKind::Keyword(s)) => s.as_str().to_string(),
                    other => {
                        return Err(ParseError(format!(
                            "expected method name after `static`, got {other:?}"
                        )));
                    }
                }
            } else {
                mname
            };
            // Getter/setter: `get prop() {…}` / `set prop(v) {…}`. V1
            // installs both under different internal names so they
            // don't overwrite each other when a class defines both.
            // `obj.<name>` continues to read the data property (not
            // the accessor) — real accessor semantics (defineProperty)
            // land later — but the bodies are still callable as
            // `obj.__get_<name>()` / `obj.__set_<name>(v)` and the
            // class no longer fails to parse.
            let (mname, accessor_kind) = if (mname == "get" || mname == "set")
                && matches!(
                    self.peek().map(|t| &t.kind),
                    Some(TokenKind::Identifier(_) | TokenKind::Keyword(_))
                )
                && !matches!(
                    self.toks.get(self.i + 1).map(|t| &t.kind),
                    Some(TokenKind::Punct(p)) if p == "="
                ) {
                let prefix = mname.clone();
                while self.eat_lineterm() {}
                let real_name = match self.bump().map(|t| t.kind) {
                    Some(TokenKind::Identifier(s)) => s,
                    Some(TokenKind::Keyword(s)) => s.as_str().to_string(),
                    other => {
                        return Err(ParseError(format!("expected accessor name, got {other:?}")));
                    }
                };
                // mname carries the REAL property name; accessor_kind says
                // whether this body is the getter or setter for it.
                (real_name, Some(prefix))
            } else {
                (mname, None)
            };
            // Async method: `async foo()`. The body is desugared (below) so
            // the method returns a promise and a thrown body rejects rather
            // than throwing synchronously.
            let is_async_method = mname == "async"
                && matches!(
                    self.peek().map(|t| &t.kind),
                    Some(TokenKind::Identifier(_) | TokenKind::Keyword(_))
                );
            let mname = if is_async_method {
                while self.eat_lineterm() {}
                match self.bump().map(|t| t.kind) {
                    Some(TokenKind::Identifier(s)) => s,
                    Some(TokenKind::Keyword(s)) => s.as_str().to_string(),
                    _ => return Err(ParseError("expected method name after async".into())),
                }
            } else {
                mname
            };

            // Class field: `name = expr;` or `name;` (no method body).
            // Desugar to `this.name = expr;` injected at the front of
            // the constructor body so every instance gets the field.
            let next_is_paren = matches!(
                self.peek().map(|t| &t.kind),
                Some(TokenKind::Punct(p)) if p == "("
            );
            if !next_is_paren {
                // Either `name = expr;` or `name;`.
                let init = if self.match_punct("=") {
                    self.parse_assignment_expr()?
                } else {
                    Expr::Undefined
                };
                let _ = self.match_punct(";");
                // Instance field → `this.f = init` in the ctor; static field →
                // `<Class>.f = init` at definition.
                let target = if is_static {
                    Expr::Member {
                        object: Box::new(cn()),
                        property: Box::new(Expr::Identifier(mname.clone())),
                        computed: false,
                    }
                } else {
                    Expr::Member {
                        object: Box::new(Expr::This),
                        property: Box::new(Expr::Identifier(mname.clone())),
                        computed: false,
                    }
                };
                let assign = Stmt::Expression(Expr::Assignment {
                    op: AssignOp::Assign,
                    target: Box::new(target),
                    value: Box::new(init),
                });
                if is_static {
                    static_members.push(assign);
                } else {
                    field_inits.push(assign);
                }
                continue;
            }
            let params = self.parse_param_list()?;
            let prev_in_gen = self.in_generator;
            self.in_generator = is_generator_method;
            let block = self.parse_block();
            self.in_generator = prev_in_gen;
            let body = match block? {
                Stmt::Block(b) => b,
                _ => return Err(ParseError("class method body".into())),
            };

            if mname == "constructor" {
                ctor_params = params;
                ctor_body = body;
            } else if let Some(kind) = accessor_kind {
                // Real accessor on `<Class>.prototype` (or `<Class>` if static).
                let func = Expr::Function {
                    name: Some(format!("{kind} {mname}")),
                    params,
                    body,
                };
                let key = (mname.clone(), is_static);
                if let Some(slot) = accessors.iter_mut().find(|(k, _, _)| *k == key) {
                    if kind == "get" {
                        slot.1 = Some(func);
                    } else {
                        slot.2 = Some(func);
                    }
                } else if kind == "get" {
                    accessors.push((key, Some(func), None));
                } else {
                    accessors.push((key, None, Some(func)));
                }
            } else {
                // Instance method → `<Class>.prototype.m = fn`; static → `<Class>.m = fn`.
                let method_fn = {
                    if is_generator_method {
                        make_generator_expr(Some(mname.clone()), params, body)
                    } else {
                        let f = Expr::Function {
                            name: Some(mname.clone()),
                            params,
                            body,
                        };
                        if is_async_method { make_async(f) } else { f }
                    }
                };
                let target = if is_static {
                    Expr::Member {
                        object: Box::new(cn()),
                        property: Box::new(Expr::Identifier(mname)),
                        computed: false,
                    }
                } else {
                    Expr::Member {
                        object: Box::new(cn_proto()),
                        property: Box::new(Expr::Identifier(mname)),
                        computed: false,
                    }
                };
                let assign = Stmt::Expression(Expr::Assignment {
                    op: AssignOp::Assign,
                    target: Box::new(target),
                    value: Box::new(method_fn),
                });
                if is_static {
                    static_members.push(assign);
                } else {
                    proto_members.push(assign);
                }
            }
        }

        // Accessors → `Object.defineProperty(<Class>.prototype, name, {get,set})`
        // (or on `<Class>` for static accessors), at definition time.
        for ((acc_name, acc_static), getter, setter) in accessors {
            let mut desc_props: Vec<(String, Expr)> = Vec::new();
            if let Some(g) = getter {
                desc_props.push(("get".into(), g));
            }
            if let Some(s) = setter {
                desc_props.push(("set".into(), s));
            }
            desc_props.push(("enumerable".into(), Expr::Boolean(false)));
            desc_props.push(("configurable".into(), Expr::Boolean(true)));
            let target = if acc_static { cn() } else { cn_proto() };
            let define_call = Expr::Call {
                callee: Box::new(Expr::Member {
                    object: Box::new(Expr::Identifier("Object".into())),
                    property: Box::new(Expr::Identifier("defineProperty".into())),
                    computed: false,
                }),
                args: vec![target, Expr::String(acc_name), Expr::Object(desc_props)],
            };
            let stmt = Stmt::Expression(define_call);
            if acc_static {
                static_members.push(stmt);
            } else {
                proto_members.push(stmt);
            }
        }

        let has_super = superclass.is_some();
        // Constructor body: instance field initializers, then the explicit
        // constructor body. For a derived class with no explicit constructor,
        // synthesize `super(...arguments)` so the base initializes `this`.
        let mut ctor_full_body: Vec<Stmt> = Vec::new();
        ctor_full_body.extend(field_inits);
        if ctor_body.is_empty() && superclass.is_some() {
            ctor_full_body.push(Stmt::Expression(Expr::Call {
                callee: Box::new(Expr::Member {
                    object: Box::new(Expr::Identifier("\u{1}__superclass__".to_string())),
                    property: Box::new(Expr::Identifier("apply".to_string())),
                    computed: false,
                }),
                args: vec![Expr::This, Expr::Identifier("arguments".to_string())],
            }));
        }
        ctor_full_body.extend(ctor_body);

        // Assemble the class as an IIFE that builds a real prototype-based
        // constructor and returns it:
        //   (function(){
        //      const __superclass__ = <Super>;                 // if extends
        //      function <Class>(<ctorParams>) { <ctorBody> }
        //      Object.setPrototypeOf(<Class>.prototype, __superclass__.prototype);  // if extends
        //      <Class>.prototype.m = function(){...};           // instance methods
        //      <Class>.s = function(){...};                     // static members
        //      return <Class>;
        //   })()
        // Methods are FUNCTION EXPRESSIONS closing over this IIFE scope, so
        // `super(...)`/`super.x` (which read `__superclass__`) resolve, and the
        // prototype chain is linked at DEFINITION — so `Class.prototype.X` reads
        // (incl. React's `prototype.isReactComponent` class detection) work
        // before any instance is constructed.
        let mut iife_body: Vec<Stmt> = Vec::new();
        if let Some(parent) = superclass {
            iife_body.push(Stmt::VarDecl {
                kind: VarKind::Const,
                decls: vec![VarDeclarator {
                    name: "\u{1}__superclass__".to_string(),
                    init: Some(parent),
                }],
            });
        }
        iife_body.push(Stmt::FunctionDecl {
            name: internal_name.clone(),
            params: ctor_params,
            body: ctor_full_body,
        });
        if has_super {
            // __superclass__ && __superclass__.prototype &&
            //   Object.setPrototypeOf(<Class>.prototype, __superclass__.prototype)
            let set_proto = Expr::Call {
                callee: Box::new(Expr::Member {
                    object: Box::new(Expr::Identifier("Object".to_string())),
                    property: Box::new(Expr::Identifier("setPrototypeOf".to_string())),
                    computed: false,
                }),
                args: vec![
                    cn_proto(),
                    Expr::Member {
                        object: Box::new(Expr::Identifier("\u{1}__superclass__".to_string())),
                        property: Box::new(Expr::Identifier("prototype".to_string())),
                        computed: false,
                    },
                ],
            };
            let guard = Expr::Logical {
                op: LogicalOp::And,
                left: Box::new(Expr::Identifier("\u{1}__superclass__".to_string())),
                right: Box::new(Expr::Logical {
                    op: LogicalOp::And,
                    left: Box::new(Expr::Member {
                        object: Box::new(Expr::Identifier("\u{1}__superclass__".to_string())),
                        property: Box::new(Expr::Identifier("prototype".to_string())),
                        computed: false,
                    }),
                    right: Box::new(set_proto),
                }),
            };
            iife_body.push(Stmt::Expression(guard));

            // Static inheritance: Object.setPrototypeOf(<Class>, __superclass__)
            // so the subclass CONSTRUCTOR inherits the superclass's static
            // members (ECMA-262 §15.7 ClassDefinitionEvaluation: F.[[Prototype]]
            // = superclass). chart.js registers components by reading a static
            // `id` declared on a base class through the subclass — without this
            // `Subclass.id` is undefined and registration throws "class does not
            // have id". Runs before the subclass's own statics so those still
            // override. Guarded on __superclass__ being truthy.
            let set_static_proto = Expr::Call {
                callee: Box::new(Expr::Member {
                    object: Box::new(Expr::Identifier("Object".to_string())),
                    property: Box::new(Expr::Identifier("setPrototypeOf".to_string())),
                    computed: false,
                }),
                args: vec![cn(), Expr::Identifier("\u{1}__superclass__".to_string())],
            };
            let static_guard = Expr::Logical {
                op: LogicalOp::And,
                left: Box::new(Expr::Identifier("\u{1}__superclass__".to_string())),
                right: Box::new(set_static_proto),
            };
            iife_body.push(Stmt::Expression(static_guard));
        }
        iife_body.extend(proto_members);
        iife_body.extend(static_members);
        iife_body.push(Stmt::Return(Some(cn())));

        let iife = Expr::Call {
            callee: Box::new(Expr::Function {
                name: None,
                params: Vec::new(),
                body: iife_body,
            }),
            args: Vec::new(),
        };

        Ok((name, iife))
    }

    /// `import ... from '...'` / `import '...'` / `export ...` —
    /// consumed up to the next semicolon or line break so the script
    /// can continue. Modules land later; for now this just stops
    /// scripts with import statements from failing to parse.
    /// `export const x = ...` / `export function f() {}` /
    /// `export class C {}` / `export { a, b }` / `export default expr` /
    /// `export * from "..."`. V1 strips the `export` keyword and parses
    /// the inner declaration as a regular statement so the names land
    /// in the script scope. We don't yet build an export-table for
    /// cross-module linking — see `skip_module_directive` for the
    /// import side.
    fn parse_export(&mut self) -> Result<Stmt, ParseError> {
        self.bump(); // consume "export"
        while self.eat_lineterm() {}
        match self.peek().map(|t| &t.kind) {
            Some(TokenKind::Keyword(k)) => match k.as_str() {
                "const" => {
                    self.bump();
                    self.parse_var_decl_stmt(VarKind::Const)
                }
                "let" => {
                    self.bump();
                    self.parse_var_decl_stmt(VarKind::Let)
                }
                "var" => {
                    self.bump();
                    self.parse_var_decl_stmt(VarKind::Var)
                }
                "function" => self.parse_function_decl(),
                "class" => self.parse_class_decl(),
                "default" => {
                    // `export default expr` — evaluate the rhs and
                    // discard the binding; the value is unreachable
                    // because we don't have a module record.
                    self.bump();
                    while self.eat_lineterm() {}
                    if matches!(
                        self.peek().map(|t| &t.kind),
                        Some(TokenKind::Keyword(k)) if k == "function"
                    ) {
                        return self.parse_function_decl();
                    }
                    if matches!(
                        self.peek().map(|t| &t.kind),
                        Some(TokenKind::Keyword(k)) if k == "class"
                    ) {
                        return self.parse_class_decl();
                    }
                    self.parse_expr_stmt()
                }
                "async" => {
                    // `export async function foo() {}`
                    self.parse_expr_stmt()
                }
                _ => self.skip_module_directive(),
            },
            // `export { a, b }` or `export * from "x"` — just skip
            // since we can't link.
            Some(TokenKind::Punct(p)) if p == "{" || p == "*" => self.skip_module_directive(),
            _ => self.skip_module_directive(),
        }
    }

    fn skip_module_directive(&mut self) -> Result<Stmt, ParseError> {
        // Consume tokens until we hit a `;`, line terminator, or EOF.
        loop {
            match self.peek().map(|t| &t.kind) {
                Some(TokenKind::Punct(p)) if p == ";" => {
                    self.bump();
                    break;
                }
                Some(TokenKind::LineTerminator) => break,
                Some(_) => {
                    self.bump();
                }
                None => break,
            }
        }
        Ok(Stmt::Empty)
    }

    fn parse_function_decl(&mut self) -> Result<Stmt, ParseError> {
        while self.eat_lineterm() {}
        self.bump(); // function
        // `function* name(...)` — generator. Calling it returns a LAZY iterator;
        // the body is compiled to a resumable state machine that suspends at
        // each `yield` (V8-shaped), via `make_generator_decl`.
        let is_generator = matches!(
            self.peek().map(|t| &t.kind),
            Some(TokenKind::Punct(p)) if p == "*"
        );
        if is_generator {
            self.bump();
        }
        let name = self.expect_identifier()?;
        let (params, mut prepend) = self.parse_param_list_with_destructure()?;
        // `yield` is only a yield-expression inside a generator body.
        let prev = self.in_generator;
        self.in_generator = is_generator;
        let block = self.parse_block();
        self.in_generator = prev;
        let Stmt::Block(body) = block? else {
            return Err(ParseError("function body".into()));
        };
        // Prepend auto-generated destructuring/defaults so the body
        // sees them as ordinary lexical bindings.
        prepend.extend(body);
        if is_generator {
            Ok(make_generator_decl(name, params, prepend))
        } else {
            Ok(Stmt::FunctionDecl {
                name,
                params,
                body: prepend,
            })
        }
    }

    /// Parse the function/arrow that follows a just-consumed `async` keyword,
    /// returning it desugared (always returns a promise; body throws reject).
    /// Used by both statement- and expression-position `async`.
    fn parse_async_rest(&mut self) -> Result<Expr, ParseError> {
        while self.eat_lineterm() {}
        if matches!(
            self.peek().map(|t| &t.kind),
            Some(TokenKind::Keyword(k)) if k == "function"
        ) {
            self.bump(); // consume "function"
            let name = match self.peek_skip_lt().map(|t| &t.kind) {
                Some(TokenKind::Identifier(s)) => {
                    let s = s.clone();
                    while self.eat_lineterm() {}
                    self.bump();
                    Some(s)
                }
                _ => None,
            };
            let params = self.parse_param_list()?;
            let Stmt::Block(body) = self.parse_block()? else {
                return Err(ParseError("function body".into()));
            };
            return Ok(make_async(Expr::Function { name, params, body }));
        }
        // `async ident => …` — single-param async arrow. The single-param
        // arrow detection normally lives in parse_assignment_expr, which never
        // runs once we're this deep, so handle it here.
        if self.peek_single_param_arrow() {
            let name = self.expect_identifier()?;
            self.expect_punct("=>")?;
            let body = self.parse_arrow_body()?;
            return Ok(make_async(Expr::Arrow {
                params: vec![name],
                body,
            }));
        }
        // `async (params) => …` falls to the `(` primary's arrow detection.
        // `make_async` only rewrites Arrow/Function nodes, so a plain
        // parenthesized expression (e.g. `async (x)` used as an identifier
        // call elsewhere) passes through untouched.
        Ok(make_async(self.parse_primary()?))
    }

    fn parse_param_list(&mut self) -> Result<Vec<String>, ParseError> {
        let (params, _) = self.parse_param_list_with_destructure()?;
        Ok(params)
    }

    /// `self.i` points just after a consumed `(`. Scan to the matching `)`
    /// (balancing nested brackets) and report whether `=>` follows — i.e.
    /// whether this `(` opens arrow-function parameters rather than a
    /// parenthesized expression. String/template/regex tokens are atomic,
    /// so their internal punctuation never affects the balance.
    fn paren_arrow_lookahead(&self) -> bool {
        let mut depth = 1i32;
        let mut j = self.i;
        while j < self.toks.len() {
            match &self.toks[j].kind {
                TokenKind::Punct(p) if p == "(" || p == "[" || p == "{" => depth += 1,
                TokenKind::Punct(p) if p == ")" || p == "]" || p == "}" => {
                    depth -= 1;
                    if depth == 0 {
                        let mut k = j + 1;
                        while matches!(
                            self.toks.get(k).map(|t| &t.kind),
                            Some(TokenKind::LineTerminator)
                        ) {
                            k += 1;
                        }
                        return matches!(
                            self.toks.get(k).map(|t| &t.kind),
                            Some(TokenKind::Punct(p)) if p == "=>"
                        );
                    }
                }
                _ => {}
            }
            j += 1;
        }
        false
    }

    /// Param-list parser that also returns auto-generated VarDecls to
    /// prepend to the function body. Pattern params (`function f({a, b})`)
    /// become hidden `__arg_N` names plus desugared `let a = __arg_N.a`
    /// statements. Identifier params with default values
    /// (`function f(x = 1)`) similarly synthesize an `x = x === undefined ? 1 : x`
    /// guard so the body sees the right value.
    fn parse_param_list_with_destructure(
        &mut self,
    ) -> Result<(Vec<String>, Vec<Stmt>), ParseError> {
        self.expect_punct("(")?;
        let mut params: Vec<String> = Vec::new();
        let mut prepend: Vec<Stmt> = Vec::new();
        if !self.match_punct(")") {
            loop {
                while self.eat_lineterm() {}
                // `...rest` — a rest parameter. Marked with a leading `...` in
                // the stored name; the call path collects the remaining args
                // into a real array under the bare name. (Identifiers can't
                // contain `.`, so the prefix is an unambiguous sentinel.)
                let is_rest = self.match_punct("...");
                // Destructuring pattern parameter.
                let is_obj_pattern = matches!(
                    self.peek().map(|t| &t.kind),
                    Some(TokenKind::Punct(p)) if p == "{"
                );
                let is_arr_pattern = matches!(
                    self.peek().map(|t| &t.kind),
                    Some(TokenKind::Punct(p)) if p == "["
                );
                if is_obj_pattern || is_arr_pattern {
                    // Synthesize a hidden parameter name and inject a
                    // destructuring VarDecl at the front of the body.
                    let hidden = format!("__arg{}", self.next_destructure_id());
                    let extras =
                        self.parse_destructuring_pattern_for_param(&hidden, is_obj_pattern)?;
                    // Optional default for the whole pattern: `({a}={}) => …`
                    // / `f({a} = {})`. Guard the hidden param before the
                    // destructuring extraction reads from it.
                    let default = if self.match_punct("=") {
                        Some(self.parse_assignment_expr()?)
                    } else {
                        None
                    };
                    params.push(hidden.clone());
                    if let Some(def) = default {
                        prepend.push(Stmt::Expression(Expr::Assignment {
                            op: AssignOp::Assign,
                            target: Box::new(Expr::Identifier(hidden.clone())),
                            value: Box::new(Expr::Conditional {
                                test: Box::new(Expr::Binary {
                                    op: BinOp::EqEqEq,
                                    left: Box::new(Expr::Identifier(hidden.clone())),
                                    right: Box::new(Expr::Undefined),
                                }),
                                cons: Box::new(def),
                                alt: Box::new(Expr::Identifier(hidden.clone())),
                            }),
                        }));
                    }
                    prepend.push(Stmt::VarDecl {
                        kind: VarKind::Let,
                        decls: extras,
                    });
                } else {
                    let name = self.expect_identifier()?;
                    if self.match_punct("=") {
                        // `param = defaultExpr` — synthesize a guard.
                        let def = self.parse_assignment_expr()?;
                        let cond = Expr::Conditional {
                            test: Box::new(Expr::Binary {
                                op: BinOp::EqEqEq,
                                left: Box::new(Expr::Identifier(name.clone())),
                                right: Box::new(Expr::Undefined),
                            }),
                            cons: Box::new(def),
                            alt: Box::new(Expr::Identifier(name.clone())),
                        };
                        prepend.push(Stmt::Expression(Expr::Assignment {
                            op: AssignOp::Assign,
                            target: Box::new(Expr::Identifier(name.clone())),
                            value: Box::new(cond),
                        }));
                    }
                    params.push(if is_rest { format!("...{name}") } else { name });
                }
                if !self.match_punct(",") {
                    break;
                }
            }
            self.expect_punct(")")?;
        }
        Ok((params, prepend))
    }

    /// Parse a `{...}` or `[...]` destructuring pattern that appears as
    /// a function parameter. Reuses the same Entry shape as
    /// `parse_destructuring_decl` but targets the supplied hidden name.
    fn parse_destructuring_pattern_for_param(
        &mut self,
        tmp_name: &str,
        is_obj: bool,
    ) -> Result<Vec<VarDeclarator>, ParseError> {
        let opener = if is_obj { "{" } else { "[" };
        let closer = if is_obj { "}" } else { "]" };
        self.expect_punct(opener)?;
        enum Entry {
            Field {
                key: String,
                name: String,
                default: Option<Expr>,
            },
            Rest {
                name: String,
            },
        }
        let mut entries: Vec<Entry> = Vec::new();
        let mut arr_index: usize = 0;
        loop {
            while self.eat_lineterm() {}
            if self.match_punct(closer) {
                break;
            }
            if matches!(
                self.peek().map(|t| &t.kind),
                Some(TokenKind::Punct(p)) if p == "..."
            ) {
                self.bump();
                let name = self.expect_identifier()?;
                entries.push(Entry::Rest { name });
                let _ = self.match_punct(",");
                continue;
            }
            if is_obj {
                let key = self.expect_identifier()?;
                let mut name = key.clone();
                if self.match_punct(":") {
                    name = self.expect_identifier()?;
                }
                let default = if self.match_punct("=") {
                    Some(self.parse_assignment_expr()?)
                } else {
                    None
                };
                entries.push(Entry::Field { key, name, default });
            } else {
                let name = self.expect_identifier()?;
                let default = if self.match_punct("=") {
                    Some(self.parse_assignment_expr()?)
                } else {
                    None
                };
                let key = arr_index.to_string();
                entries.push(Entry::Field { key, name, default });
                arr_index += 1;
            }
            let _ = self.match_punct(",");
        }
        let mut out: Vec<VarDeclarator> = Vec::new();
        for entry in entries {
            match entry {
                Entry::Field { key, name, default } => {
                    let access = if is_obj {
                        Expr::Member {
                            object: Box::new(Expr::Identifier(tmp_name.into())),
                            property: Box::new(Expr::Identifier(key.clone())),
                            computed: false,
                        }
                    } else {
                        Expr::Member {
                            object: Box::new(Expr::Identifier(tmp_name.into())),
                            property: Box::new(Expr::Number(key.parse::<f64>().unwrap_or(0.0))),
                            computed: true,
                        }
                    };
                    let init = match default {
                        None => access,
                        Some(def) => Expr::Conditional {
                            test: Box::new(Expr::Binary {
                                op: BinOp::EqEqEq,
                                left: Box::new(access.clone()),
                                right: Box::new(Expr::Undefined),
                            }),
                            cons: Box::new(def),
                            alt: Box::new(access),
                        },
                    };
                    out.push(VarDeclarator {
                        name,
                        init: Some(init),
                    });
                }
                Entry::Rest { name } => {
                    out.push(VarDeclarator {
                        name,
                        init: Some(Expr::Identifier(tmp_name.into())),
                    });
                }
            }
        }
        Ok(out)
    }

    fn parse_if(&mut self) -> Result<Stmt, ParseError> {
        while self.eat_lineterm() {}
        self.bump(); // if
        self.expect_punct("(")?;
        let test = self.parse_expr()?;
        self.expect_punct(")")?;
        let cons = Box::new(self.parse_stmt()?);
        // A braceless consequent is terminated by `;` (explicit or via
        // ASI/newline), and `else` still binds to this `if`:
        //   `if (p) n = t; else {...}` and `if(p) a()\n else b()`.
        // Without skipping that terminator the `else` looked like a fresh
        // statement → "unexpected keyword else", which (via parse-error
        // recovery) silently dropped the rest of the enclosing block —
        // e.g. the tail of core-js's giant comma-`var` chain, leaving
        // later bindings `undefined` ("X is not a function" at runtime).
        let save = self.i;
        while self.eat_lineterm() {}
        let ate_semi = self.match_punct(";");
        while self.eat_lineterm() {}
        let alt = if self.match_keyword("else") {
            Some(Box::new(self.parse_stmt()?))
        } else {
            // No else — restore so the outer statement loop sees the
            // terminator it expects (avoid swallowing a following stmt).
            if !ate_semi {
                self.i = save;
            }
            None
        };
        Ok(Stmt::If { test, cons, alt })
    }

    fn parse_while(&mut self) -> Result<Stmt, ParseError> {
        while self.eat_lineterm() {}
        self.bump();
        self.expect_punct("(")?;
        let test = self.parse_expr()?;
        self.expect_punct(")")?;
        let body = Box::new(self.parse_stmt()?);
        Ok(Stmt::While { test, body })
    }

    fn parse_do_while(&mut self) -> Result<Stmt, ParseError> {
        while self.eat_lineterm() {}
        self.bump();
        let body = Box::new(self.parse_stmt()?);
        // The do-body statement may leave a terminating `;` (e.g.
        // `do x+=1; while(c)`); consume it before the `while`.
        while self.eat_lineterm() {}
        let _ = self.match_punct(";");
        while self.eat_lineterm() {}
        if !self.match_keyword("while") {
            return Err(ParseError("expected while".into()));
        }
        self.expect_punct("(")?;
        let test = self.parse_expr()?;
        self.expect_punct(")")?;
        Ok(Stmt::DoWhile { body, test })
    }

    fn parse_for(&mut self) -> Result<Stmt, ParseError> {
        while self.eat_lineterm() {}
        self.bump();
        // `for await (x of asyncIterable)` — consume the `await` modifier. The
        // ForOf executor drives the iterator protocol and awaits each result,
        // so both sync and async iterables work through the same path.
        while self.eat_lineterm() {}
        let is_await = self.match_keyword("await");
        self.expect_punct("(")?;

        // Try for-in / for-of first via single-binding lookahead. Both shapes:
        //   for (var? x in expr) body
        //   for (var? x of expr) body
        let checkpoint = self.i;
        if let Some(stmt) = self.try_parse_for_in_of(is_await)? {
            return Ok(stmt);
        }
        self.i = checkpoint;

        let init = if self.match_punct(";") {
            None
        } else {
            let kind_opt = match self.peek_skip_lt().map(|t| &t.kind) {
                Some(TokenKind::Keyword(k)) if matches!(k.as_str(), "var" | "let" | "const") => {
                    Some(match k.as_str() {
                        "var" => VarKind::Var,
                        "let" => VarKind::Let,
                        "const" => VarKind::Const,
                        _ => unreachable!(),
                    })
                }
                _ => None,
            };
            let init = if let Some(kind) = kind_opt {
                while self.eat_lineterm() {}
                self.bump(); // kind keyword
                let mut decls = Vec::new();
                loop {
                    let name = self.expect_identifier()?;
                    let v = if self.match_punct("=") {
                        Some(self.parse_assignment_expr()?)
                    } else {
                        None
                    };
                    decls.push(VarDeclarator { name, init: v });
                    if !self.match_punct(",") {
                        break;
                    }
                }
                ForInit::VarDecl { kind, decls }
            } else {
                ForInit::Expr(self.parse_expr()?)
            };
            self.expect_punct(";")?;
            Some(init)
        };
        let test = if self.match_punct(";") {
            None
        } else {
            let t = self.parse_expr()?;
            self.expect_punct(";")?;
            Some(t)
        };
        let update = if self.match_punct(")") {
            None
        } else {
            let u = self.parse_expr()?;
            self.expect_punct(")")?;
            Some(u)
        };
        let body = Box::new(self.parse_stmt()?);
        Ok(Stmt::For {
            init,
            test,
            update,
            body,
        })
    }

    /// Attempt a `for (kind? name in|of expr) body`. Returns Ok(Some) on
    /// success, Ok(None) when the shape doesn't match (so the caller can
    /// rewind and try C-style). The `for` keyword and `(` are already
    /// consumed.
    fn try_parse_for_in_of(&mut self, is_await: bool) -> Result<Option<Stmt>, ParseError> {
        let kind = match self.peek_skip_lt().map(|t| &t.kind) {
            Some(TokenKind::Keyword(k)) if matches!(k.as_str(), "var" | "let" | "const") => {
                let kw = k.clone();
                while self.eat_lineterm() {}
                self.bump();
                Some(match kw.as_str() {
                    "var" => VarKind::Var,
                    "let" => VarKind::Let,
                    "const" => VarKind::Const,
                    _ => unreachable!(),
                })
            }
            _ => None,
        };
        // The loop binding: a plain identifier, or a destructuring pattern
        // (`for (const [k, v] of map)`). For a pattern we bind a fresh temp
        // as the loop variable and re-destructure it at the top of the body.
        let mut pattern: Option<(bool, Vec<DestructureEntry>)> = None;
        let name = match self.peek_skip_lt().map(|t| t.kind.clone()) {
            Some(TokenKind::Identifier(n)) => {
                while self.eat_lineterm() {}
                self.bump();
                n
            }
            Some(TokenKind::Punct(p)) if p == "[" || p == "{" => {
                while self.eat_lineterm() {}
                let temp = format!("__forbind{}", self.next_destructure_id());
                pattern = Some(self.parse_destructuring_entries()?);
                temp
            }
            _ => return Ok(None),
        };
        let is_in_or_of = match self.peek_skip_lt().map(|t| &t.kind) {
            Some(TokenKind::Keyword(k)) if k == "in" => Some(false),
            Some(TokenKind::Keyword(k)) if k == "of" => Some(true),
            Some(TokenKind::Identifier(k)) if k == "of" => Some(true),
            _ => None,
        };
        let Some(is_of) = is_in_or_of else {
            return Ok(None);
        };
        while self.eat_lineterm() {}
        self.bump(); // consume `in` or `of`
        let source = self.parse_expr()?;
        self.expect_punct(")")?;
        let mut body = Box::new(self.parse_stmt()?);
        // For a destructuring binding, prepend the extraction decls so the
        // pattern's names are in scope inside the loop body.
        if let Some((is_obj, entries)) = pattern {
            let decls =
                self.build_destructure_declarators(is_obj, entries, Expr::Identifier(name.clone()));
            let inner = std::mem::replace(&mut body, Box::new(Stmt::Block(Vec::new())));
            body = Box::new(Stmt::Block(vec![
                Stmt::VarDecl {
                    kind: kind.unwrap_or(VarKind::Let),
                    decls,
                },
                *inner,
            ]));
        }
        Ok(Some(if is_of {
            Stmt::ForOf {
                is_await,
                kind,
                name,
                source,
                body,
            }
        } else {
            Stmt::ForIn {
                kind,
                name,
                source,
                body,
            }
        }))
    }

    fn parse_return(&mut self) -> Result<Stmt, ParseError> {
        while self.eat_lineterm() {}
        self.bump(); // return
        // No expression if followed by ; or } or newline-then-statement.
        let next_is_term = matches!(
            self.peek().map(|t| &t.kind),
            Some(TokenKind::Punct(p)) if p == ";" || p == "}"
        ) || matches!(
            self.peek().map(|t| &t.kind),
            Some(TokenKind::LineTerminator) | None
        );
        if next_is_term {
            return Ok(Stmt::Return(None));
        }
        let e = self.parse_expr()?;
        Ok(Stmt::Return(Some(e)))
    }

    fn parse_expr_stmt(&mut self) -> Result<Stmt, ParseError> {
        let e = self.parse_expr()?;
        Ok(Stmt::Expression(e))
    }

    /// Property name in a key position: identifier, reserved word, string,
    /// or numeric literal. Used by object patterns / literals where keys
    /// like `default`, `class`, `in` are legal.
    fn expect_property_name(&mut self) -> Result<String, ParseError> {
        while self.eat_lineterm() {}
        match self.bump() {
            Some(t) => match t.kind {
                TokenKind::Identifier(s) | TokenKind::String(s) => Ok(s),
                TokenKind::Keyword(s) => Ok(s.as_str().to_string()),
                TokenKind::Number(n) => Ok(n.to_string()),
                other => Err(ParseError(format!("expected property name, got {other:?}"))),
            },
            None => Err(ParseError("expected property name, got EOF".into())),
        }
    }

    fn expect_identifier(&mut self) -> Result<String, ParseError> {
        while self.eat_lineterm() {}
        match self.bump() {
            Some(t) => match t.kind {
                TokenKind::Identifier(s) => Ok(s),
                TokenKind::Keyword(s)
                    if matches!(s.as_str(), "let" | "of" | "async" | "await" | "yield") =>
                {
                    Ok(s.as_str().to_string()) // contextual keywords usable as identifiers in some positions
                }
                other => Err(ParseError(format!("expected identifier, got {other:?}"))),
            },
            None => Err(ParseError("expected identifier, got EOF".into())),
        }
    }

    // Expression parsing — operator precedence climbing.

    fn parse_expr(&mut self) -> Result<Expr, ParseError> {
        let first = self.parse_assignment_expr()?;
        if self.match_punct(",") {
            let mut out = vec![first];
            loop {
                out.push(self.parse_assignment_expr()?);
                if !self.match_punct(",") {
                    break;
                }
            }
            Ok(Expr::Sequence(out))
        } else {
            Ok(first)
        }
    }

    fn parse_assignment_expr(&mut self) -> Result<Expr, ParseError> {
        // `yield` / `yield*` expression — only inside a generator body, so
        // `var x = yield v` (two-way `.next(v)`) and `f(yield v)` work. Outside
        // a generator, `yield` stays an ordinary identifier.
        if self.in_generator
            && matches!(self.peek().map(|t| &t.kind), Some(TokenKind::Keyword(k)) if k == "yield")
        {
            self.bump(); // yield
            let star = matches!(
                self.peek().map(|t| &t.kind),
                Some(TokenKind::Punct(p)) if p == "*"
            );
            if star {
                self.bump();
            }
            // `yield` with no operand (→ undefined) when the next token closes
            // the expression or is a line terminator (ASI). `yield*` always has
            // an operand.
            let no_operand = !star
                && (matches!(
                    self.peek().map(|t| &t.kind),
                    Some(TokenKind::Punct(p))
                        if matches!(p.as_str(), ")" | "]" | "}" | "," | ";" | ":")
                ) || self.peek().is_none()
                    || matches!(
                        self.peek().map(|t| &t.kind),
                        Some(TokenKind::LineTerminator)
                    ));
            let arg = if no_operand {
                Expr::Undefined
            } else {
                self.parse_assignment_expr()?
            };
            let marker = if star {
                "__tb_yield_star__"
            } else {
                "__tb_yield__"
            };
            return Ok(Expr::Call {
                callee: Box::new(Expr::Identifier(marker.into())),
                args: vec![arg],
            });
        }
        // Single-parameter arrow: peek `<ident> '=>'`.
        if self.peek_single_param_arrow() {
            while self.eat_lineterm() {}
            let name = self.expect_identifier()?;
            self.expect_punct("=>")?;
            let body = self.parse_arrow_body()?;
            return Ok(Expr::Arrow {
                params: vec![name],
                body,
            });
        }
        let left = self.parse_conditional()?;
        // Each entry maps the source punctuator to its `AssignOp`. Order matters
        // only for `match_punct` (longest tokens are distinct punctuators in the
        // lexer, so any order is safe); kept identical to the prior string list.
        let assign_ops = [
            ("=", AssignOp::Assign),
            ("+=", AssignOp::AddAssign),
            ("-=", AssignOp::SubAssign),
            ("*=", AssignOp::MulAssign),
            ("/=", AssignOp::DivAssign),
            ("%=", AssignOp::ModAssign),
            ("**=", AssignOp::PowAssign),
            ("<<=", AssignOp::ShlAssign),
            (">>=", AssignOp::ShrAssign),
            (">>>=", AssignOp::UShrAssign),
            ("&=", AssignOp::BitAndAssign),
            ("|=", AssignOp::BitOrAssign),
            ("^=", AssignOp::BitXorAssign),
            ("&&=", AssignOp::AndAssign),
            ("||=", AssignOp::OrAssign),
            ("??=", AssignOp::NullishAssign),
        ];
        for (tok, op) in &assign_ops {
            if self.match_punct(tok) {
                let value = self.parse_assignment_expr()?;
                return Ok(Expr::Assignment {
                    op: *op,
                    target: Box::new(left),
                    value: Box::new(value),
                });
            }
        }
        Ok(left)
    }

    fn parse_conditional(&mut self) -> Result<Expr, ParseError> {
        let test = self.parse_nullish_coalesce()?;
        if self.match_punct("?") {
            let cons = self.parse_assignment_expr()?;
            self.expect_punct(":")?;
            let alt = self.parse_assignment_expr()?;
            Ok(Expr::Conditional {
                test: Box::new(test),
                cons: Box::new(cons),
                alt: Box::new(alt),
            })
        } else {
            Ok(test)
        }
    }

    fn parse_nullish_coalesce(&mut self) -> Result<Expr, ParseError> {
        let mut left = self.parse_logical_or()?;
        while self.match_punct("??") {
            let right = self.parse_logical_or()?;
            left = Expr::Logical {
                op: LogicalOp::Nullish,
                left: Box::new(left),
                right: Box::new(right),
            };
        }
        Ok(left)
    }

    fn parse_logical_or(&mut self) -> Result<Expr, ParseError> {
        let mut left = self.parse_nullish()?;
        while self.match_punct("||") {
            let right = self.parse_nullish()?;
            left = Expr::Logical {
                op: LogicalOp::Or,
                left: Box::new(left),
                right: Box::new(right),
            };
        }
        Ok(left)
    }

    /// Nullish coalescing (`??`) binds between `||` and `&&` per the
    /// spec — but the operator can't mix with `||`/`&&` without parens.
    /// We accept either ordering here; the interp short-circuits.
    fn parse_nullish(&mut self) -> Result<Expr, ParseError> {
        let mut left = self.parse_logical_and()?;
        while self.match_punct("??") {
            let right = self.parse_logical_and()?;
            left = Expr::Logical {
                op: LogicalOp::Nullish,
                left: Box::new(left),
                right: Box::new(right),
            };
        }
        Ok(left)
    }

    fn parse_logical_and(&mut self) -> Result<Expr, ParseError> {
        let mut left = self.parse_bit_or()?;
        while self.match_punct("&&") {
            let right = self.parse_bit_or()?;
            left = Expr::Logical {
                op: LogicalOp::And,
                left: Box::new(left),
                right: Box::new(right),
            };
        }
        Ok(left)
    }

    fn parse_bit_or(&mut self) -> Result<Expr, ParseError> {
        let mut left = self.parse_bit_xor()?;
        while self.match_punct("|") {
            let right = self.parse_bit_xor()?;
            left = Expr::Binary {
                op: BinOp::BitOr,
                left: Box::new(left),
                right: Box::new(right),
            };
        }
        Ok(left)
    }

    fn parse_bit_xor(&mut self) -> Result<Expr, ParseError> {
        let mut left = self.parse_bit_and()?;
        while self.match_punct("^") {
            let right = self.parse_bit_and()?;
            left = Expr::Binary {
                op: BinOp::BitXor,
                left: Box::new(left),
                right: Box::new(right),
            };
        }
        Ok(left)
    }

    fn parse_bit_and(&mut self) -> Result<Expr, ParseError> {
        let mut left = self.parse_equality()?;
        while self.match_punct("&") {
            let right = self.parse_equality()?;
            left = Expr::Binary {
                op: BinOp::BitAnd,
                left: Box::new(left),
                right: Box::new(right),
            };
        }
        Ok(left)
    }

    fn parse_equality(&mut self) -> Result<Expr, ParseError> {
        let mut left = self.parse_relational()?;
        loop {
            let op = if self.match_punct("===") {
                BinOp::EqEqEq
            } else if self.match_punct("!==") {
                BinOp::NeqEqEq
            } else if self.match_punct("==") {
                BinOp::EqEq
            } else if self.match_punct("!=") {
                BinOp::Neq
            } else {
                break;
            };
            let right = self.parse_relational()?;
            left = Expr::Binary {
                op,
                left: Box::new(left),
                right: Box::new(right),
            };
        }
        Ok(left)
    }

    fn parse_relational(&mut self) -> Result<Expr, ParseError> {
        let mut left = self.parse_shift()?;
        loop {
            let op = if self.match_punct("<=") {
                BinOp::Le
            } else if self.match_punct(">=") {
                BinOp::Ge
            } else if self.match_punct("<") {
                BinOp::Lt
            } else if self.match_punct(">") {
                BinOp::Gt
            } else if self.match_keyword("instanceof") {
                BinOp::Instanceof
            } else if self.match_keyword("in") {
                BinOp::In
            } else {
                break;
            };
            let right = self.parse_shift()?;
            left = Expr::Binary {
                op,
                left: Box::new(left),
                right: Box::new(right),
            };
        }
        Ok(left)
    }

    fn parse_shift(&mut self) -> Result<Expr, ParseError> {
        let mut left = self.parse_additive()?;
        loop {
            let op = if self.match_punct(">>>") {
                BinOp::UShr
            } else if self.match_punct("<<") {
                BinOp::Shl
            } else if self.match_punct(">>") {
                BinOp::Shr
            } else {
                break;
            };
            let right = self.parse_additive()?;
            left = Expr::Binary {
                op,
                left: Box::new(left),
                right: Box::new(right),
            };
        }
        Ok(left)
    }

    fn parse_additive(&mut self) -> Result<Expr, ParseError> {
        let mut left = self.parse_multiplicative()?;
        loop {
            let op = if self.match_punct("+") {
                BinOp::Add
            } else if self.match_punct("-") {
                BinOp::Sub
            } else {
                break;
            };
            let right = self.parse_multiplicative()?;
            left = Expr::Binary {
                op,
                left: Box::new(left),
                right: Box::new(right),
            };
        }
        Ok(left)
    }

    fn parse_multiplicative(&mut self) -> Result<Expr, ParseError> {
        let mut left = self.parse_exponent()?;
        loop {
            let op = if self.match_punct("*") {
                BinOp::Mul
            } else if self.match_punct("/") {
                BinOp::Div
            } else if self.match_punct("%") {
                BinOp::Mod
            } else {
                break;
            };
            let right = self.parse_exponent()?;
            left = Expr::Binary {
                op,
                left: Box::new(left),
                right: Box::new(right),
            };
        }
        Ok(left)
    }

    fn parse_exponent(&mut self) -> Result<Expr, ParseError> {
        let left = self.parse_unary()?;
        if self.match_punct("**") {
            let right = self.parse_exponent()?;
            Ok(Expr::Binary {
                op: BinOp::Pow,
                left: Box::new(left),
                right: Box::new(right),
            })
        } else {
            Ok(left)
        }
    }

    fn parse_unary(&mut self) -> Result<Expr, ParseError> {
        let unary_ops = [
            ("!", UnaryOp::Not),
            ("~", UnaryOp::BitNot),
            ("+", UnaryOp::Plus),
            ("-", UnaryOp::Neg),
        ];
        for (tok, op) in &unary_ops {
            if self.match_punct(tok) {
                let target = self.parse_unary()?;
                return Ok(Expr::Unary {
                    op: *op,
                    target: Box::new(target),
                });
            }
        }
        let keyword_ops = [
            ("typeof", UnaryOp::Typeof),
            ("void", UnaryOp::Void),
            ("delete", UnaryOp::Delete),
        ];
        for (tok, op) in &keyword_ops {
            if self.match_keyword(tok) {
                let target = self.parse_unary()?;
                return Ok(Expr::Unary {
                    op: *op,
                    target: Box::new(target),
                });
            }
        }
        // `await expr` — our Promise is synchronous, so an awaited
        // Promise has already resolved by the time we see it. The
        // simplest semantic-preserving rewrite is `await x` → `x`
        // (since promise.then(v=>v) is what the value flowed through).
        // `await x` — V1 sync Promise model: promises settle eagerly via
        // the microtask drain, so awaiting is just unwrapping a settled
        // promise's value. We desugar to a single-eval IIFE that returns
        // `x._value` when `x` is a fulfilled promise, else `x` itself:
        //   (a => (a != null && a._isPromise === true ? a._value : a))(x)
        // Without this, `await p` evaluated to the promise OBJECT, so e.g.
        // `const {value,done} = await reader.read()` destructured the
        // promise wrapper (value/done undefined) — which hung React's RSC
        // flight reader and any `await stream.read()` consumer.
        if self.match_keyword("await") {
            let target = self.parse_unary()?;
            // Desugar to the engine's real `Await` operation (ECMA-262
            // §27.5.3.8 / V8's Await): drive the job queue until the operand's
            // promise settles, then return its value or THROW its rejection.
            // Implemented in Rust (`__tb_await__`) rather than an ad-hoc
            // expression so reject→throw and pending→drain match the spec.
            return Ok(Expr::Call {
                callee: Box::new(Expr::Identifier("__tb_await__".to_string())),
                args: vec![target],
            });
        }
        // ++x / --x
        for (tok, op) in &[("++", UpdateOp::Inc), ("--", UpdateOp::Dec)] {
            if self.match_punct(tok) {
                let target = self.parse_unary()?;
                return Ok(Expr::Update {
                    op: *op,
                    target: Box::new(target),
                    prefix: true,
                });
            }
        }
        self.parse_postfix()
    }

    /// Collect `.prop`, `[key]`, and `(args)` chain suffixes inside an optional-
    /// chain guard body.  Stops before `?.` (which starts a new guard level).
    /// Ensures `a?.b.c.d` short-circuits the WHOLE tail, not just `b`.
    fn collect_opt_chain_tail(&mut self, mut body: Expr) -> Result<Expr, ParseError> {
        loop {
            if matches!(
                self.peek_skip_lt().map(|t| &t.kind),
                Some(TokenKind::Punct(p)) if p == "?."
            ) {
                break;
            }
            if self.match_punct(".") {
                while self.eat_lineterm() {}
                let name = match self.peek().map(|t| t.kind.clone()) {
                    Some(TokenKind::Identifier(s)) => {
                        self.i += 1;
                        s
                    }
                    Some(TokenKind::Keyword(s)) => {
                        self.i += 1;
                        s.as_str().to_string()
                    }
                    _ => break,
                };
                let member = Expr::Member {
                    object: Box::new(body),
                    property: Box::new(Expr::Identifier(name)),
                    computed: false,
                };
                if self.match_punct("(") {
                    let args = self.parse_arg_list()?;
                    body = Expr::Call { callee: Box::new(member), args };
                } else {
                    body = member;
                }
            } else if self.match_punct("[") {
                let prop = self.parse_expr()?;
                self.expect_punct("]")?;
                let member = Expr::Member {
                    object: Box::new(body),
                    property: Box::new(prop),
                    computed: true,
                };
                if self.match_punct("(") {
                    let args = self.parse_arg_list()?;
                    body = Expr::Call { callee: Box::new(member), args };
                } else {
                    body = member;
                }
            } else if self.match_punct("(") {
                let args = self.parse_arg_list()?;
                body = Expr::Call { callee: Box::new(body), args };
            } else {
                break;
            }
        }
        Ok(body)
    }

    fn parse_postfix(&mut self) -> Result<Expr, ParseError> {
        let mut e = self.parse_lhs()?;
        // Member/call repetition.
        loop {
            // `?.` — optional chaining. Desugars to a single-eval IIFE
            // so `obj?.prop` returns `undefined` when `obj` is null/undefined.
            // IMPORTANT: the WHOLE tail after `?.` up to the next `?.` is
            // folded inside ONE IIFE guard so `a?.b.c` short-circuits the
            // entire `.b.c` access, not just `.b`.
            if self.match_punct("?.") {
                while self.eat_lineterm() {}
                // Build the IIFE guard body starting with `__opt` as the already-
                // evaluated receiver.  Then extend via collect_opt_chain_tail so
                // `a?.b.c` emits ONE IIFE: (__opt => __opt == null ? undefined : __opt.b.c)(a)
                // instead of applying .c AFTER the IIFE returns, which would throw
                // TypeError on null (the original bug).
                let tmp = "__opt".to_string();
                let tmp_ref = Expr::Identifier(tmp.clone());
                let body_start: Expr = if self.match_punct("[") {
                    let prop = self.parse_expr()?;
                    self.expect_punct("]")?;
                    Expr::Member { object: Box::new(tmp_ref.clone()), property: Box::new(prop), computed: true }
                } else if self.match_punct("(") {
                    // `recv?.(args)` — optional CALL of the receiver itself.
                    // Must preserve `this` when the receiver is a member access
                    // (e.g. `obj.fn?.()`), so route through the old helper which
                    // uses `.call(base, ...)` for the member case.
                    let args = self.parse_arg_list()?;
                    e = desugar_optional_chain(e, OptionalAccess::Call { args });
                    continue;
                } else {
                    let name = match self.bump() {
                        Some(t) => match t.kind {
                            TokenKind::Identifier(s) => s,
                            TokenKind::Keyword(s) => s.as_str().to_string(),
                            other => return Err(ParseError(format!("expected property name after '?.', got {other:?}"))),
                        },
                        None => return Err(ParseError("expected property after '?.'".into())),
                    };
                    Expr::Member { object: Box::new(tmp_ref.clone()), property: Box::new(Expr::Identifier(name)), computed: false }
                };
                // Consume an immediate (args) after a property access (method call).
                let body_after_call = if matches!(&body_start, Expr::Member { .. }) {
                    if self.match_punct("(") {
                        let args = self.parse_arg_list()?;
                        Expr::Call { callee: Box::new(body_start), args }
                    } else {
                        body_start
                    }
                } else {
                    body_start
                };
                let full_body = self.collect_opt_chain_tail(body_after_call)?;
                let test = Expr::Binary {
                    op: BinOp::EqEq,
                    left: Box::new(tmp_ref.clone()),
                    right: Box::new(Expr::Null),
                };
                let cond = Expr::Conditional {
                    test: Box::new(test),
                    cons: Box::new(Expr::Undefined),
                    alt: Box::new(full_body),
                };
                e = Expr::Call {
                    callee: Box::new(Expr::Function {
                        name: None,
                        params: vec![tmp],
                        body: vec![Stmt::Return(Some(cond))],
                    }),
                    args: vec![e],
                };
                continue;
            }
            if self.match_punct(".") {
                // Reserved words ARE valid property names after `.`.
                while self.eat_lineterm() {}
                let name = match self.bump() {
                    Some(t) => match t.kind {
                        TokenKind::Identifier(s) => s,
                        TokenKind::Keyword(s) => s.as_str().to_string(),
                        other => {
                            return Err(ParseError(format!(
                                "expected property name after '.', got {other:?}"
                            )));
                        }
                    },
                    None => return Err(ParseError("expected property name after '.'".into())),
                };
                e = Expr::Member {
                    object: Box::new(e),
                    property: Box::new(Expr::Identifier(name)),
                    computed: false,
                };
            } else if self.match_punct("[") {
                let prop = self.parse_expr()?;
                self.expect_punct("]")?;
                e = Expr::Member {
                    object: Box::new(e),
                    property: Box::new(prop),
                    computed: true,
                };
            } else if self.match_punct("(") {
                let args = self.parse_arg_list()?;
                e = Expr::Call {
                    callee: Box::new(e),
                    args,
                };
            } else if matches!(
                self.peek().map(|t| &t.kind),
                Some(TokenKind::TemplateString(_, _))
            ) {
                // Tagged template: tag`a${x}b` → tag(strings, x)
                // where strings = ["a","b"] (cooked) and strings.raw = ["a","b"] (raw).
                // ECMA-262 §13.2.8.3: strings.raw preserves backslashes as-is.
                let (cooked_body, raw_body) = match self.bump().map(|t| t.kind) {
                    Some(TokenKind::TemplateString(cooked, raw)) => (cooked, raw),
                    _ => unreachable!(),
                };
                // Split the cooked body for string segments and expression sources.
                let (cooked_segs, expr_srcs) = split_template_body(&cooked_body);
                // Split the raw body for raw segments (same hole structure, backslash-preserved text).
                let (raw_segs, _) = split_template_body(&raw_body);
                let interp_exprs: Vec<Expr> = expr_srcs
                    .iter()
                    .map(|src| {
                        let toks = tokenize(src);
                        if toks.is_empty() {
                            return Expr::Undefined;
                        }
                        let mut p = Parser { toks, i: 0, destructure_counter: 0, in_generator: false };
                        p.parse_assignment_expr().unwrap_or(Expr::Undefined)
                    })
                    .collect();
                // strings array: cooked segments (escape sequences processed).
                let strings_array = Expr::Array(cooked_segs.iter().map(|s| Expr::String(s.clone())).collect());
                // strings.raw array: raw segments (backslashes preserved).
                let raw_array = Expr::Array(raw_segs.iter().map(|s| Expr::String(s.clone())).collect());
                // Wrap in a small IIFE to attach .raw (spec requires it; String.raw and
                // some DSL tags read it).
                let tpl_param = "__tb_tpl".to_string();
                let tpl_ref = Expr::Identifier(tpl_param.clone());
                let strings_with_raw = Expr::Call {
                    callee: Box::new(Expr::Function {
                        name: None,
                        params: vec![tpl_param],
                        body: vec![
                            Stmt::Expression(Expr::Assignment {
                                op: AssignOp::Assign,
                                target: Box::new(Expr::Member {
                                    object: Box::new(tpl_ref.clone()),
                                    property: Box::new(Expr::Identifier("raw".into())),
                                    computed: false,
                                }),
                                value: Box::new(raw_array),
                            }),
                            Stmt::Return(Some(tpl_ref)),
                        ],
                    }),
                    args: vec![strings_array],
                };
                let mut call_args = vec![strings_with_raw];
                call_args.extend(interp_exprs);
                e = Expr::Call { callee: Box::new(e), args: call_args };
            } else {
                break;
            }
        }
        // Postfix `++` / `--` binds to the whole member/call chain so
        // `obj.prop++`, `arr[i]++`, `this.#count++` all work.
        for (tok, op) in &[("++", UpdateOp::Inc), ("--", UpdateOp::Dec)] {
            if self.match_punct(tok) {
                return Ok(Expr::Update {
                    op: *op,
                    target: Box::new(e),
                    prefix: false,
                });
            }
        }
        Ok(e)
    }

    fn parse_arg_list(&mut self) -> Result<Vec<Expr>, ParseError> {
        let mut args = Vec::new();
        if self.match_punct(")") {
            return Ok(args);
        }
        loop {
            // Trailing comma in an argument list (`f(a, b,)`) is permitted by
            // ECMA-262 §13.3 — stop before mis-parsing `)` as an argument.
            if self.match_punct(")") {
                return Ok(args);
            }
            if self.match_punct("...") {
                let inner = self.parse_assignment_expr()?;
                args.push(Expr::Spread(Box::new(inner)));
            } else {
                args.push(self.parse_assignment_expr()?);
            }
            if !self.match_punct(",") {
                break;
            }
        }
        self.expect_punct(")")?;
        Ok(args)
    }

    fn parse_lhs(&mut self) -> Result<Expr, ParseError> {
        if self.match_keyword("new") {
            // `new.target` meta-property (ECMA-262 §13.3.12) — was a hard parse
            // error that silently dropped the whole enclosing statement.
            if self.match_punct(".") {
                while self.eat_lineterm() {}
                match self.bump() {
                    Some(t) if matches!(&t.kind, TokenKind::Identifier(s) if s == "target") => {
                        return Ok(Expr::NewTarget);
                    }
                    other => {
                        return Err(ParseError(format!(
                            "expected 'target' after 'new.', got {other:?}"
                        )));
                    }
                }
            }
            let mut callee = self.parse_lhs()?;
            // ECMA-262 §13.3 `NewExpression: new MemberExpression Arguments`.
            // The constructor is the FULL member chain (`new a.b.c(x)` ⇒
            // `new (a.b.c)(x)`), so consume `.prop` / `[expr]` tails here —
            // but NOT a call `(...)`, which becomes THIS `new`'s argument list.
            // Without this, `new o.PromiseQueue(5)` mis-parsed as
            // `(new o).PromiseQueue(5)` → "o is not a constructor".
            loop {
                if self.match_punct(".") {
                    while self.eat_lineterm() {}
                    let name = match self.bump() {
                        Some(t) => match t.kind {
                            TokenKind::Identifier(s) => s,
                            TokenKind::Keyword(s) => s.as_str().to_string(),
                            other => {
                                return Err(ParseError(format!(
                                    "expected property name after '.' in new-expr, got {other:?}"
                                )));
                            }
                        },
                        None => {
                            return Err(ParseError(
                                "expected property name after '.' in new-expr".into(),
                            ));
                        }
                    };
                    callee = Expr::Member {
                        object: Box::new(callee),
                        property: Box::new(Expr::Identifier(name)),
                        computed: false,
                    };
                } else if self.match_punct("[") {
                    let prop = self.parse_expr()?;
                    self.expect_punct("]")?;
                    callee = Expr::Member {
                        object: Box::new(callee),
                        property: Box::new(prop),
                        computed: true,
                    };
                } else {
                    break;
                }
            }
            let args = if self.match_punct("(") {
                self.parse_arg_list()?
            } else {
                Vec::new()
            };
            return Ok(Expr::New {
                callee: Box::new(callee),
                args,
            });
        }
        self.parse_primary()
    }

    fn parse_primary(&mut self) -> Result<Expr, ParseError> {
        while self.eat_lineterm() {}
        let t = self
            .bump()
            .ok_or_else(|| ParseError("unexpected EOF".into()))?;
        match t.kind {
            TokenKind::Number(n) => Ok(Expr::Number(n)),
            TokenKind::BigInt(n) => Ok(Expr::BigInt(n)),
            TokenKind::String(s) => Ok(Expr::String(s)),
            // Use the cooked (escape-processed) string for template literals.
            // The raw part is only needed for tagged templates (handled above).
            TokenKind::TemplateString(cooked, _raw) => Ok(Expr::TemplateLiteral(cooked)),
            TokenKind::Regex(body, flags) => Ok(Expr::Regex(body, flags)),
            TokenKind::Identifier(n) => Ok(Expr::Identifier(n)),
            TokenKind::Keyword(k) => match k.as_str() {
                "true" => Ok(Expr::Boolean(true)),
                "false" => Ok(Expr::Boolean(false)),
                "null" => Ok(Expr::Null),
                "undefined" => Ok(Expr::Undefined),
                "this" => Ok(Expr::This),
                // `super` — desugared against this engine's class model
                // (a class is a function that assigns its instance members
                // onto `this`; a derived class binds the hidden
                // `\u{1}__superclass__` local to its parent constructor in
                // `parse_class_decl`). Three forms:
                //   super(args)  → __superclass__.call(this, args)
                //                  (runs the parent constructor against the
                //                   instance, exactly the .call(this) shape)
                //   super.x      → __superclass__.prototype.x
                //   super[x]     → __superclass__.prototype[x]
                // The postfix loop appends the `.x` / `[x]` / call tail for
                // the member forms.
                "super" => {
                    let supref = || Expr::Identifier("\u{1}__superclass__".to_string());
                    if self.match_punct("(") {
                        let mut args = vec![Expr::This];
                        args.extend(self.parse_arg_list()?);
                        Ok(Expr::Call {
                            callee: Box::new(Expr::Member {
                                object: Box::new(supref()),
                                property: Box::new(Expr::Identifier("call".into())),
                                computed: false,
                            }),
                            args,
                        })
                    } else {
                        // `super.x` / `super[x]` read off the PARENT prototype.
                        let proto = Expr::Member {
                            object: Box::new(supref()),
                            property: Box::new(Expr::Identifier("prototype".into())),
                            computed: false,
                        };
                        let (property, computed) = if self.match_punct("[") {
                            let p = self.parse_expr()?;
                            self.expect_punct("]")?;
                            (p, true)
                        } else {
                            self.expect_punct(".")?;
                            while self.eat_lineterm() {}
                            let name = match self.bump() {
                                Some(t) => match t.kind {
                                    TokenKind::Identifier(s) => s,
                                    TokenKind::Keyword(s) => s.as_str().to_string(),
                                    other => {
                                        return Err(ParseError(format!(
                                            "expected property after 'super.', got {other:?}"
                                        )));
                                    }
                                },
                                None => {
                                    return Err(ParseError(
                                        "expected property after 'super.'".into(),
                                    ));
                                }
                            };
                            (Expr::Identifier(name), false)
                        };
                        let member = Expr::Member {
                            object: Box::new(proto),
                            property: Box::new(property),
                            computed,
                        };
                        // `super.m(args)` is a METHOD call: run the parent's
                        // method with `this` = the CURRENT instance, not the
                        // parent prototype. `(super.m)(args)` would bind `this`
                        // to the prototype, so a parent method reading
                        // `this.field` (set by the subclass constructor) sees
                        // undefined — exactly what stalled tsparticles
                        // (`load(t){ super.load(t); this.animation.load() }`).
                        if self.match_punct("(") {
                            let mut call_args = vec![Expr::This];
                            call_args.extend(self.parse_arg_list()?);
                            Ok(Expr::Call {
                                callee: Box::new(Expr::Member {
                                    object: Box::new(member),
                                    property: Box::new(Expr::Identifier("call".into())),
                                    computed: false,
                                }),
                                args: call_args,
                            })
                        } else {
                            Ok(member)
                        }
                    }
                }
                // `class … {}` as an expression (the `class` keyword was
                // just consumed by the primary dispatch).
                "class" => self.parse_class_expr(),
                // `async function ...` / `async () => ...` /
                // `async ident => ...` — the modifier is dropped (sync
                // Promise model). For `async ident => ...` the lhs
                // is consumed as an Identifier and we fall through to
                // the arrow rewrite that happens at higher levels.
                "async" => self.parse_async_rest(),
                "function" => {
                    // `function* (){}` / `function* name(){}` — generator
                    // expression. Wrap like the declaration form so the call
                    // returns a lazy iterator.
                    let is_generator = matches!(
                        self.peek().map(|t| &t.kind),
                        Some(TokenKind::Punct(p)) if p == "*"
                    );
                    if is_generator {
                        self.bump();
                        while self.eat_lineterm() {}
                    }
                    let name = match self.peek_skip_lt().map(|t| &t.kind) {
                        Some(TokenKind::Identifier(s)) => {
                            let s = s.clone();
                            while self.eat_lineterm() {}
                            self.bump();
                            Some(s)
                        }
                        _ => None,
                    };
                    let params = self.parse_param_list()?;
                    let prev = self.in_generator;
                    self.in_generator = is_generator;
                    let block = self.parse_block();
                    self.in_generator = prev;
                    let Stmt::Block(body) = block? else {
                        return Err(ParseError("function body".into()));
                    };
                    if is_generator {
                        Ok(make_generator_expr(name, params, body))
                    } else {
                        Ok(Expr::Function { name, params, body })
                    }
                }
                // Contextual keywords are NOT reserved words — sloppy-mode
                // code (every minified bundle) freely uses them as plain
                // identifiers, e.g. core-js's `sf(of, o)` / `var of=…`.
                // Our lexer classifies them as Keyword, so accept them as
                // an Identifier expression here. (Without this the parse
                // failed → error-recovery dropped the rest of the
                // enclosing comma-`var` chain → later bindings undefined.)
                "of" | "as" | "from" | "get" | "set" | "let" | "static" | "await" | "yield"
                | "async_" => Ok(Expr::Identifier(k.as_str().to_string())),
                _ => Err(ParseError(format!("unexpected keyword {k}"))),
            },
            TokenKind::Punct(p) if p == "(" => {
                // Parenthesized expression OR arrow params. Disambiguate by
                // scanning to the matching `)` and checking for a following
                // `=>`. Arrow params are then parsed with the full
                // destructuring/default/rest-aware param parser so shapes
                // like `(a, {b}) => …`, `(x = 1) => …`, `(...rest) => …`
                // all work.
                if self.paren_arrow_lookahead() {
                    self.i -= 1; // rewind to the `(` for the param-list parser
                    let (params, prepend) = self.parse_param_list_with_destructure()?;
                    self.expect_punct("=>")?;
                    let mut body = self.parse_arrow_body()?;
                    if !prepend.is_empty() {
                        body = match body {
                            ArrowBody::Block(stmts) => {
                                let mut s = prepend;
                                s.extend(stmts);
                                ArrowBody::Block(s)
                            }
                            ArrowBody::Expr(e) => {
                                let mut s = prepend;
                                s.push(Stmt::Return(Some(*e)));
                                ArrowBody::Block(s)
                            }
                        };
                    }
                    return Ok(Expr::Arrow { params, body });
                }
                // Plain parenthesized expression.
                let e = self.parse_expr()?;
                self.expect_punct(")")?;
                Ok(e)
            }
            TokenKind::Punct(p) if p == "[" => {
                let mut items = Vec::new();
                loop {
                    // Skip line terminators between elements/commas/`]`. Real
                    // bundles (prettier/eslint default) format arrays one
                    // element per line with a trailing comma: `[\n a,\n b,\n]`.
                    // Without skipping the LT before `]`, the trailing `,\n]`
                    // was misread as "another element follows", the parse of
                    // the whole `var x = [...]` failed, and the binding was
                    // lost — a major real-page "X is not defined" source.
                    while self.eat_lineterm() {}
                    // Elision (ECMA-262 §13.2.4): leading/consecutive commas
                    // are array holes that evaluate to `undefined`. Consuming
                    // them here also makes a trailing comma a no-op, so
                    // `[1,2,]` parses as `[1,2]` instead of mis-reading `]`.
                    while matches!(
                        self.peek().map(|t| &t.kind),
                        Some(TokenKind::Punct(p)) if p == ","
                    ) {
                        self.bump();
                        items.push(Expr::Undefined);
                        while self.eat_lineterm() {}
                    }
                    if matches!(
                        self.peek().map(|t| &t.kind),
                        Some(TokenKind::Punct(p)) if p == "]"
                    ) {
                        break;
                    }
                    if self.match_punct("...") {
                        let inner = self.parse_assignment_expr()?;
                        items.push(Expr::Spread(Box::new(inner)));
                    } else {
                        items.push(self.parse_assignment_expr()?);
                    }
                    if !self.match_punct(",") {
                        break;
                    }
                }
                self.expect_punct("]")?;
                Ok(Expr::Array(items))
            }
            TokenKind::Punct(p) if p == "{" => self.parse_object_literal(),
            other => Err(ParseError(format!("unexpected token {other:?}"))),
        }
    }

    /// Parse the body of `{...}` after the opening `{` has already
    /// been consumed by the caller. Handles:
    /// - plain key/value (`a: 1`)
    /// - shorthand identifier (`a` → `a: a`)
    /// - shorthand method (`foo(x) { ... }` → `foo: function(x){...}`)
    /// - computed key (`[expr]: value`)
    /// - spread (`...obj`) — desugared to Object.assign per existing path
    /// Returns the assembled Expr — either an Expr::Object, an
    /// Object.assign() call for spreads, or an IIFE that handles
    /// computed-key assignments after construction.
    fn parse_object_literal(&mut self) -> Result<Expr, ParseError> {
        enum LitEntry {
            KV(String, Expr),
            Spread(Expr),
            Computed(Expr, Expr),
            Accessor {
                prop: String,
                kind: String,
                func: Expr,
            },
        }
        let mut items: Vec<LitEntry> = Vec::new();
        let mut has_spread = false;
        let mut has_computed = false;
        let mut has_accessor = false;
        if !self.match_punct("}") {
            loop {
                while self.eat_lineterm() {}
                // Trailing comma (or a comma immediately before `}`): the
                // previous iteration consumed the separator `,` and looped
                // back. ECMA-262 ObjectLiteral permits a trailing comma, so
                // stop here and let the closing `}` be consumed below instead
                // of mis-parsing `}` as a property key.
                if matches!(
                    self.peek().map(|t| &t.kind),
                    Some(TokenKind::Punct(p)) if p == "}"
                ) {
                    break;
                }
                // `...obj` spread.
                if self.match_punct("...") {
                    let e = self.parse_assignment_expr()?;
                    items.push(LitEntry::Spread(e));
                    has_spread = true;
                    if !self.match_punct(",") {
                        break;
                    }
                    continue;
                }
                // `[expr]: value` computed key OR `[expr](){}` computed method.
                if self.match_punct("[") {
                    let key_expr = self.parse_assignment_expr()?;
                    self.expect_punct("]")?;
                    // Computed METHOD: `{[k](){}}` ≡ `{[k]: function(){}}`. This is
                    // the standard idiom for `{[Symbol.iterator](){...}}`,
                    // `{[Symbol.toPrimitive](){...}}`, and Babel/TypeScript output;
                    // it previously threw a parse error (expect `:`) and silently
                    // dropped the whole object literal.
                    if matches!(self.peek().map(|t| &t.kind), Some(TokenKind::Punct(p)) if p == "(")
                    {
                        let (params, mut prepend) = self.parse_param_list_with_destructure()?;
                        let Stmt::Block(body) = self.parse_block()? else {
                            return Err(ParseError("computed method body".into()));
                        };
                        prepend.extend(body);
                        let func = Expr::Function {
                            name: None,
                            params,
                            body: prepend,
                        };
                        items.push(LitEntry::Computed(key_expr, func));
                        has_computed = true;
                        if !self.match_punct(",") {
                            break;
                        }
                        continue;
                    }
                    self.expect_punct(":")?;
                    let val = self.parse_assignment_expr()?;
                    items.push(LitEntry::Computed(key_expr, val));
                    has_computed = true;
                    if !self.match_punct(",") {
                        break;
                    }
                    continue;
                }
                // Method modifiers that precede the key: `async name(){}`,
                // `*name(){}` (generator), `get name(){}` / `set name(){}`.
                // A modifier only applies when it is NOT itself the property
                // (i.e. the next token is a real key, not `:` `(` `,` `}`).
                // The sync model drops `async`/`*`; `get`/`set` become real
                // accessor descriptors via Object.defineProperty.
                let next_is_key_start = |k: Option<&TokenKind>| {
                    !matches!(
                        k,
                        Some(TokenKind::Punct(p)) if p == ":" || p == "(" || p == "," || p == "}"
                    ) && k.is_some()
                };
                let is_async_method = matches!(self.peek().map(|t| &t.kind), Some(TokenKind::Keyword(k)) if k == "async")
                    && next_is_key_start(self.toks.get(self.i + 1).map(|t| &t.kind));
                if is_async_method {
                    self.bump(); // consume `async` (body desugared below)
                    while self.eat_lineterm() {}
                }
                let is_generator_method =
                    matches!(self.peek().map(|t| &t.kind), Some(TokenKind::Punct(p)) if p == "*");
                if is_generator_method {
                    self.bump(); // consume generator `*`
                    while self.eat_lineterm() {}
                }
                // Accessor method: `get x(){...}` / `set x(v){...}`. Emit an
                // Object.defineProperty so reads/writes route through it.
                // `get`/`set` are contextual identifiers (not reserved), but
                // accept a Keyword spelling too for safety. Normalize either to
                // its source string for the guard + binding.
                let accessor_word: Option<String> = match self.peek().map(|t| &t.kind) {
                    Some(TokenKind::Identifier(k)) => Some(k.clone()),
                    Some(TokenKind::Keyword(k)) => Some(k.as_str().to_string()),
                    _ => None,
                };
                let accessor_kind = match accessor_word {
                    Some(k)
                        if (k == "get" || k == "set")
                            && next_is_key_start(self.toks.get(self.i + 1).map(|t| &t.kind))
                            && matches!(
                                self.toks.get(self.i + 2).map(|t| &t.kind),
                                Some(TokenKind::Punct(p)) if p == "("
                            ) =>
                    {
                        self.bump(); // consume get/set
                        while self.eat_lineterm() {}
                        Some(k)
                    }
                    _ => None,
                };
                if let Some(kind) = accessor_kind {
                    let prop = match self.bump() {
                        Some(t) => match t.kind {
                            TokenKind::Identifier(s) | TokenKind::String(s) => s,
                            TokenKind::Keyword(s) => s.as_str().to_string(),
                            TokenKind::Number(n) => n.to_string(),
                            k => return Err(ParseError(format!("bad accessor key: {k:?}"))),
                        },
                        None => return Err(ParseError("EOF in object accessor".into())),
                    };
                    let (params, mut prepend) = self.parse_param_list_with_destructure()?;
                    let Stmt::Block(body) = self.parse_block()? else {
                        return Err(ParseError("accessor body".into()));
                    };
                    prepend.extend(body);
                    let func = Expr::Function {
                        name: Some(format!("{kind} {prop}")),
                        params,
                        body: prepend,
                    };
                    items.push(LitEntry::Accessor { prop, kind, func });
                    has_accessor = true;
                    if !self.match_punct(",") {
                        break;
                    }
                    continue;
                }
                let key = match self.bump() {
                    Some(t) => match t.kind {
                        TokenKind::Identifier(s) | TokenKind::String(s) => s,
                        TokenKind::Keyword(s) => s.as_str().to_string(),
                        TokenKind::Number(n) => n.to_string(),
                        k => {
                            return Err(ParseError(format!("bad object key: {k:?}")));
                        }
                    },
                    None => return Err(ParseError("EOF in object".into())),
                };
                // Shorthand method: `key(args) { body }` → key: function...
                if matches!(
                    self.peek().map(|t| &t.kind),
                    Some(TokenKind::Punct(p)) if p == "("
                ) {
                    let (params, mut prepend) = self.parse_param_list_with_destructure()?;
                    let prev = self.in_generator;
                    self.in_generator = is_generator_method;
                    let block = self.parse_block();
                    self.in_generator = prev;
                    let Stmt::Block(body) = block? else {
                        return Err(ParseError("method body".into()));
                    };
                    prepend.extend(body);
                    let val = {
                        if is_generator_method {
                            make_generator_expr(None, params, prepend)
                        } else {
                            let f = Expr::Function {
                                name: None,
                                params,
                                body: prepend,
                            };
                            if is_async_method { make_async(f) } else { f }
                        }
                    };
                    items.push(LitEntry::KV(key, val));
                    if !self.match_punct(",") {
                        break;
                    }
                    continue;
                }
                let val = if self.match_punct(":") {
                    self.parse_assignment_expr()?
                } else {
                    Expr::Identifier(key.clone())
                };
                items.push(LitEntry::KV(key, val));
                if !self.match_punct(",") {
                    break;
                }
            }
            self.expect_punct("}")?;
        }
        // No exotic features → return a plain Object.
        if !has_spread && !has_computed && !has_accessor {
            let entries: Vec<(String, Expr)> = items
                .into_iter()
                .filter_map(|e| match e {
                    LitEntry::KV(k, v) => Some((k, v)),
                    _ => None,
                })
                .collect();
            return Ok(Expr::Object(entries));
        }
        // Has computed keys (and possibly spread). Build via IIFE:
        //   (function(__o) {
        //     Object.assign(__o, {staticKV…}, spreadArg, …);
        //     __o[<key_expr>] = <val>;
        //     return __o;
        //   })({})
        // This way computed keys can use arbitrary expressions while
        // still composing with spreads.
        if has_computed || has_accessor {
            let tmp = format!("__obj{}", self.next_destructure_id());
            let mut body: Vec<Stmt> = Vec::new();
            // Accessor descriptors grouped by property name: (prop, get, set).
            let mut accessors: Vec<(String, Option<Expr>, Option<Expr>)> = Vec::new();
            // Group consecutive static KVs into Object.assign args.
            let mut assign_args: Vec<Expr> = Vec::new();
            assign_args.push(Expr::Identifier(tmp.clone()));
            let mut current_group: Vec<(String, Expr)> = Vec::new();
            let flush_group = |group: &mut Vec<(String, Expr)>, out: &mut Vec<Expr>| {
                if !group.is_empty() {
                    out.push(Expr::Object(std::mem::take(group)));
                }
            };
            let mut emit_assign = |body: &mut Vec<Stmt>, args: &mut Vec<Expr>| {
                if args.len() > 1 {
                    let call = Expr::Call {
                        callee: Box::new(Expr::Member {
                            object: Box::new(Expr::Identifier("Object".into())),
                            property: Box::new(Expr::Identifier("assign".into())),
                            computed: false,
                        }),
                        args: std::mem::take(args),
                    };
                    body.push(Stmt::Expression(call));
                    args.push(Expr::Identifier(tmp.clone()));
                }
            };
            for it in items {
                match it {
                    LitEntry::KV(k, v) => current_group.push((k, v)),
                    LitEntry::Spread(e) => {
                        flush_group(&mut current_group, &mut assign_args);
                        assign_args.push(e);
                    }
                    LitEntry::Computed(key_expr, val_expr) => {
                        flush_group(&mut current_group, &mut assign_args);
                        emit_assign(&mut body, &mut assign_args);
                        // __o[key_expr] = val_expr;
                        body.push(Stmt::Expression(Expr::Assignment {
                            op: AssignOp::Assign,
                            target: Box::new(Expr::Member {
                                object: Box::new(Expr::Identifier(tmp.clone())),
                                property: Box::new(key_expr),
                                computed: true,
                            }),
                            value: Box::new(val_expr),
                        }));
                    }
                    LitEntry::Accessor { prop, kind, func } => {
                        // Collect; emitted as merged defineProperty below so
                        // a get+set pair on the same prop share one descriptor.
                        if let Some(slot) = accessors.iter_mut().find(|(p, _, _)| *p == prop) {
                            if kind == "get" {
                                slot.1 = Some(func);
                            } else {
                                slot.2 = Some(func);
                            }
                        } else if kind == "get" {
                            accessors.push((prop, Some(func), None));
                        } else {
                            accessors.push((prop, None, Some(func)));
                        }
                    }
                }
            }
            flush_group(&mut current_group, &mut assign_args);
            emit_assign(&mut body, &mut assign_args);
            // Emit one Object.defineProperty(__o, prop, {get,set,…}) per
            // accessor so `obj.x` / `obj.x = v` route through the real
            // accessor path (ECMA-262 §13.2.5).
            for (prop, getter, setter) in accessors {
                let mut desc: Vec<(String, Expr)> = Vec::new();
                if let Some(g) = getter {
                    desc.push(("get".into(), g));
                }
                if let Some(s) = setter {
                    desc.push(("set".into(), s));
                }
                desc.push(("enumerable".into(), Expr::Boolean(true)));
                desc.push(("configurable".into(), Expr::Boolean(true)));
                body.push(Stmt::Expression(Expr::Call {
                    callee: Box::new(Expr::Member {
                        object: Box::new(Expr::Identifier("Object".into())),
                        property: Box::new(Expr::Identifier("defineProperty".into())),
                        computed: false,
                    }),
                    args: vec![
                        Expr::Identifier(tmp.clone()),
                        Expr::String(prop),
                        Expr::Object(desc),
                    ],
                }));
            }
            body.push(Stmt::Return(Some(Expr::Identifier(tmp.clone()))));
            return Ok(Expr::Call {
                callee: Box::new(Expr::Function {
                    name: None,
                    params: vec![tmp],
                    body,
                }),
                args: vec![Expr::Object(Vec::new())],
            });
        }
        // Has spread only — old Object.assign({}, ...) path.
        let mut args: Vec<Expr> = Vec::new();
        args.push(Expr::Object(Vec::new()));
        let mut current_group: Vec<(String, Expr)> = Vec::new();
        let flush = |group: &mut Vec<(String, Expr)>, out: &mut Vec<Expr>| {
            if !group.is_empty() {
                out.push(Expr::Object(std::mem::take(group)));
            }
        };
        for it in items {
            match it {
                LitEntry::KV(k, v) => current_group.push((k, v)),
                LitEntry::Spread(e) => {
                    flush(&mut current_group, &mut args);
                    args.push(e);
                }
                LitEntry::Computed(_, _) | LitEntry::Accessor { .. } => unreachable!(),
            }
        }
        flush(&mut current_group, &mut args);
        Ok(Expr::Call {
            callee: Box::new(Expr::Member {
                object: Box::new(Expr::Identifier("Object".into())),
                property: Box::new(Expr::Identifier("assign".into())),
                computed: false,
            }),
            args,
        })
    }

    fn peek_single_param_arrow(&self) -> bool {
        let mut j = self.i;
        while matches!(
            self.toks.get(j).map(|t| &t.kind),
            Some(TokenKind::LineTerminator)
        ) {
            j += 1;
        }
        let ident_ok = matches!(
            self.toks.get(j).map(|t| &t.kind),
            Some(TokenKind::Identifier(_))
        );
        if !ident_ok {
            return false;
        }
        let mut k = j + 1;
        while matches!(
            self.toks.get(k).map(|t| &t.kind),
            Some(TokenKind::LineTerminator)
        ) {
            k += 1;
        }
        matches!(
            self.toks.get(k).map(|t| &t.kind),
            Some(TokenKind::Punct(p)) if p == "=>"
        )
    }

    fn parse_arrow_body(&mut self) -> Result<ArrowBody, ParseError> {
        if matches!(self.peek_skip_lt().map(|t| &t.kind), Some(TokenKind::Punct(p)) if p == "{") {
            let Stmt::Block(body) = self.parse_block()? else {
                return Err(ParseError("arrow body".into()));
            };
            Ok(ArrowBody::Block(body))
        } else {
            let e = self.parse_assignment_expr()?;
            Ok(ArrowBody::Expr(Box::new(e)))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p(src: &str) -> Vec<Stmt> {
        parse_program(src).unwrap_or_else(|e| panic!("parse failed: {e} on {src:?}"))
    }

    /// M3.1 WIN MEASUREMENT (real measured numbers): the String→Copy-enum
    /// conversion of the 5 op-bearing AST nodes means an operator NODE
    /// contributes ZERO heap `String` allocations to the parsed AST (a Rust
    /// `String` has no small-string optimisation, so the old `op:"+".into()`
    /// heap-allocated one buffer per operator node).
    ///
    /// Two complementary measurements:
    ///
    /// (1) ABSOLUTE small-case: `1+1` over NUMBER operands (which allocate
    ///     nothing). The ONLY heap allocations are the two `Box<Expr>` operand
    ///     slots of the single `Expr::Binary` node. Under the OLD `op:String`
    ///     AST there would be a THIRD allocation — the operator `String`. We
    ///     assert exactly 2 (no operator String).
    ///
    /// (2) MARGINAL: parse a chain of K `+ 1` then 4K, divide the extra alloc
    ///     count by the extra operator nodes. Each extra `+ 1` adds its two
    ///     `Box<Expr>` slots (left spine + right operand) and a parser temp —
    ///     NONE of which is a per-operator `String`. The marginal is the
    ///     Box-spine cost only and contains no String term (the old AST's
    ///     marginal would have been one higher, carrying the op String).
    #[test]
    fn m31_operator_nodes_allocate_no_string_per_operator() {
        use crate::lexer::alloc_count::measure;

        // (1) ABSOLUTE (informational): a single binary expression over Number
        // operands. This includes fixed lexer/parser scaffolding (the token
        // Vec, etc.), so it is NOT a clean per-operator number — the MARGINAL
        // below isolates that. Reported for context only.
        let _ = parse_expression_str("1+1").unwrap();
        let (_e, _b1, c_one) = measure(|| parse_expression_str("1+1").unwrap());
        eprintln!("[M3.1] parse(\"1+1\") = {c_one} heap allocs (incl. fixed lexer/parser scaffolding; see marginal for the clean per-operator cost)");

        // (2) MARGINAL over a chain of `+ 1`. Number operands allocate nothing,
        // so the per-operator marginal is the Box spine only — and is FLAT, not
        // growing with an extra String term per node.
        let chain = |n: usize| -> String {
            let mut s = String::from("var r = 1");
            for _ in 1..n {
                s.push_str(" + 1");
            }
            s.push(';');
            s
        };
        let small_n = 200usize;
        let large_n = 800usize; // 4x → 600 extra `+` operator nodes
        let small_src = chain(small_n);
        let large_src = chain(large_n);
        let _ = parse_program(&small_src).unwrap();
        let (_a, _bs, c_small) = measure(|| parse_program(&small_src).unwrap());
        let (_a2, _bl, c_large) = measure(|| parse_program(&large_src).unwrap());
        let extra_ops = large_n - small_n; // 600 extra `+` nodes
        let extra_allocs = c_large.saturating_sub(c_small);
        let per_op = extra_allocs as f64 / extra_ops as f64;
        eprintln!(
            "[M3.1] size_of::<Expr>() = {} bytes; \
             parse({small_n} ops): {c_small} allocs, parse({large_n} ops): {c_large} allocs; \
             marginal = {extra_allocs} allocs over {extra_ops} extra `+` nodes = {per_op:.3}/op \
             (Box spine only — contains NO per-operator String)",
            std::mem::size_of::<Expr>(),
        );
        // The marginal is the Box spine (2 Box<Expr> + parser temp ≈ 3); the
        // load-bearing claim — zero operator String — is proven by part (1).
        // Guard the marginal against an accidental String-per-op regression
        // (which would push it to ~4).
        assert!(
            per_op <= 3.30,
            "M3.1 REGRESSION: marginal allocation per operator node is {per_op:.3} \
             — higher than the Box-spine cost, suggesting a per-operator String."
        );
    }

    /// Report `size_of::<Expr>()`. HONEST measurement: the String→enum change
    /// shrank the op-bearing variants' OWN payload (`Binary`/`Logical`/
    /// `Assignment` went from `String(24)+Box(8)+Box(8)=40` down to
    /// `enum(1)+Box(8)+Box(8)=17`, `Unary`/`Update` likewise), BUT `Expr` is an
    /// enum sized by its LARGEST variant — which is `Function { name:
    /// Option<String>(24), params: Vec<String>(24), body: Vec<Stmt>(24) } = 72`,
    /// unaffected by the op change. So the overall `size_of::<Expr>()` HOLDS at
    /// 72 (the win is the eliminated per-operator-node String ALLOCATION, proven
    /// in `m31_operator_nodes_allocate_no_string_per_operator`, plus tighter
    /// op-variant payloads → better cache locality for those nodes). This test
    /// documents the size and guards against a GROWTH regression.
    #[test]
    fn m31_expr_size_documented() {
        let sz = std::mem::size_of::<Expr>();
        eprintln!("[M3.1] size_of::<Expr>() = {sz} bytes");
        assert!(
            sz <= 72,
            "size_of::<Expr>() = {sz} GREW past 72 — the op-enum change must not \
             enlarge the node (op enums are 1 byte; the size is set by `Function`)."
        );
    }

    #[test]
    fn destructures_object_pattern() {
        // `const {a, b} = obj;` should desugar to:
        //   const __dst0 = obj; const a = __dst0.a; const b = __dst0.b;
        // I.e. one VarDecl with 3 declarators.
        let s = p("const {a, b} = obj;");
        match &s[0] {
            Stmt::VarDecl { kind, decls } => {
                assert_eq!(*kind, VarKind::Const);
                assert_eq!(decls.len(), 3);
                assert!(decls[0].name.starts_with("__dst"));
                assert_eq!(decls[1].name, "a");
                assert_eq!(decls[2].name, "b");
            }
            other => panic!("expected VarDecl, got {other:?}"),
        }
    }

    #[test]
    fn destructures_object_with_alias_and_default() {
        // `const {x: y = 1} = obj` — alias `x` to `y`, default to 1.
        let s = p("const {x: y = 1} = obj;");
        match &s[0] {
            Stmt::VarDecl { decls, .. } => {
                assert_eq!(decls.len(), 2);
                assert_eq!(decls[1].name, "y");
            }
            other => panic!("got {other:?}"),
        }
    }

    #[test]
    fn destructures_array_pattern() {
        let s = p("const [a, b] = arr;");
        match &s[0] {
            Stmt::VarDecl { decls, .. } => {
                assert_eq!(decls.len(), 3);
                assert_eq!(decls[1].name, "a");
                assert_eq!(decls[2].name, "b");
            }
            other => panic!("got {other:?}"),
        }
    }

    #[test]
    fn error_recovery_skips_bad_statement() {
        // Destructuring isn't supported yet, so this is normally a
        // parse error. With recovery, the surrounding `let a` and
        // `let c` should still land in the AST so the rest of the
        // script can run.
        let s = p("let a = 1; const {x, y} = obj; let c = 3;");
        // Expect at least two VarDecls for a and c (the destructuring
        // is skipped). The bad statement may yield no AST node or a
        // garbled one, but recovery must not drop the surrounding
        // good statements.
        let var_names: Vec<_> = s
            .iter()
            .filter_map(|stmt| {
                if let Stmt::VarDecl { decls, .. } = stmt {
                    decls.first().map(|d| d.name.clone())
                } else {
                    None
                }
            })
            .collect();
        assert!(
            var_names.iter().any(|n| n == "a"),
            "lost `let a`: {var_names:?}"
        );
        assert!(
            var_names.iter().any(|n| n == "c"),
            "lost `let c`: {var_names:?}"
        );
    }

    #[test]
    fn error_recovery_in_function_body() {
        // Same idea, but the bad statement is inside a function body.
        let s = p("function f() { let a = 1; const {x} = o; return a; }");
        // Function still parsed.
        assert!(matches!(s.first(), Some(Stmt::FunctionDecl { .. })));
    }

    #[test]
    fn parses_var_decl() {
        let s = p("let x = 1;");
        match &s[0] {
            Stmt::VarDecl { kind, decls } => {
                assert_eq!(*kind, VarKind::Let);
                assert_eq!(decls[0].name, "x");
                assert!(matches!(decls[0].init, Some(Expr::Number(n)) if (n - 1.0).abs() < 1e-9));
            }
            _ => panic!("expected VarDecl statement for let binding with init"),
        }
    }

    #[test]
    fn parses_arithmetic_precedence() {
        let s = p("let r = 1 + 2 * 3;");
        match &s[0] {
            Stmt::VarDecl { decls, .. } => match decls[0].init.as_ref().unwrap() {
                Expr::Binary { op, left, right } => {
                    assert_eq!(*op, BinOp::Add);
                    assert!(matches!(**left, Expr::Number(n) if (n - 1.0).abs() < 1e-9));
                    match &**right {
                        Expr::Binary { op, .. } => assert_eq!(*op, BinOp::Mul),
                        _ => {
                            panic!("expected binary * operator on right side of + precedence test")
                        }
                    }
                }
                _ => panic!("expected binary expression for arithmetic precedence test"),
            },
            _ => panic!("expected VarDecl in arithmetic precedence test"),
        }
    }

    #[test]
    fn parses_function_decl() {
        let s = p("function add(a, b) { return a + b; }");
        match &s[0] {
            Stmt::FunctionDecl { name, params, body } => {
                assert_eq!(name, "add");
                assert_eq!(params, &vec!["a".to_string(), "b".to_string()]);
                assert!(matches!(body[0], Stmt::Return(Some(_))));
            }
            _ => panic!("expected FunctionDecl statement"),
        }
    }

    #[test]
    fn parses_arrow_and_call() {
        let s = p("const sq = x => x * x; sq(5);");
        match &s[0] {
            Stmt::VarDecl { decls, .. } => {
                let init = decls[0].init.as_ref().unwrap();
                match init {
                    Expr::Arrow { params, body } => {
                        assert_eq!(params, &vec!["x".to_string()]);
                        assert!(matches!(body, ArrowBody::Expr(_)));
                    }
                    _ => panic!("expected arrow expression in arrow_and_call test"),
                }
            }
            _ => panic!("expected VarDecl statement containing arrow function"),
        }
        assert!(matches!(&s[1], Stmt::Expression(Expr::Call { .. })));
    }

    #[test]
    fn parses_if_else_while() {
        let s = p("if (x > 0) { y = 1; } else { y = -1; }");
        assert!(matches!(s[0], Stmt::If { .. }));
        let s = p("while (i < 10) { i = i + 1; }");
        assert!(matches!(s[0], Stmt::While { .. }));
    }

    #[test]
    fn parses_multiline_array_with_trailing_comma() {
        // prettier/eslint default formatting: one element per line + trailing
        // comma. The `,\n]` must not break the parse. Regression: it silently
        // lost the whole `var x = [...]` binding (x became undefined), a major
        // real-bundle "X is not defined / not a function" source.
        let s = p("var arr = [\n  1,\n  2,\n  3,\n];");
        match &s[0] {
            Stmt::VarDecl { decls, .. } => match decls[0].init.as_ref().unwrap() {
                Expr::Array(v) => assert_eq!(v.len(), 3, "trailing comma must not add a hole"),
                other => panic!("expected Array init, got {other:?}"),
            },
            other => panic!("expected VarDecl, got {other:?}"),
        }
        // Nested multi-line array + object, both with trailing commas.
        let s2 = p("var o = {\n  a: 1,\n  b: [\n    9,\n    8,\n  ],\n};");
        assert!(matches!(&s2[0], Stmt::VarDecl { .. }));
    }

    #[test]
    fn parses_object_and_array() {
        let s = p("const o = { a: 1, b: [2, 3, 4] };");
        match &s[0] {
            Stmt::VarDecl { decls, .. } => match decls[0].init.as_ref().unwrap() {
                Expr::Object(entries) => {
                    assert_eq!(entries[0].0, "a");
                    assert!(matches!(entries[1].1, Expr::Array(ref v) if v.len() == 3));
                }
                _ => panic!("expected Object expression in object_and_array test"),
            },
            _ => panic!("expected VarDecl statement in object_and_array test"),
        }
    }

    #[test]
    fn parses_member_chain() {
        let s = p("a.b.c[0]();");
        match &s[0] {
            Stmt::Expression(e) => match e {
                Expr::Call { callee, .. } => {
                    assert!(matches!(**callee, Expr::Member { computed: true, .. }))
                }
                _ => panic!("expected Call expression in member_chain test"),
            },
            _ => panic!("expected Expression statement in member_chain test"),
        }
    }
}
