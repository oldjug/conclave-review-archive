//! Chrome-parity async/await via a state-machine lowering.
//!
//! The parser desugars `await E` to `Call(__tb_await__, [E])`. To make an async
//! function TRULY suspend — yield to the event loop and resume when its awaited
//! promise settles, exactly as V8 does — we compile its body into a resumable
//! op-list (`AsyncOp`) run by a driver in the interpreter (`interp::AsyncState`).
//! Because the engine's scopes are `Rc<RefCell<Scope>>`, locals survive
//! suspension for free, so the resumable state is just `(scope, program-counter,
//! handler stack)`.
//!
//! Stages:
//!   1. **Hoist** (`Hoister`): lift every `await` out of sub-expressions into
//!      top-level `let $tmp = await E;` statements so the only suspension points
//!      are whole statements. Short-circuit (`&&`/`||`/`??`) and conditional
//!      (`?:`) operators are desugared so laziness is preserved.
//!   2. **Lower** (`Lowerer`): compile the hoisted statement tree into a flat
//!      `Vec<AsyncOp>` with explicit jumps + a try-handler stack. Statements that
//!      contain NO suspension run wholesale via the normal interpreter
//!      (`AsyncOp::Exec`) — only suspension-containing control flow is lowered.
//!
//! Coverage grows by milestone; `is_lowerable` gates what we compile so any
//! not-yet-lowered construct keeps the legacy synchronous path (no regression).
//! Milestone 1 (this file): sequential statements, `if/else`, `try/catch/finally`,
//! blocks, `return`, `throw` — i.e. the explorer's particle loader shape. Loops,
//! `switch`, `for-of/in`, and generators are added in later milestones.

use crate::ast::{AssignOp, BinOp, Expr, ForInit, LogicalOp, Stmt, UnaryOp, VarDeclarator, VarKind};

/// Identifier the parser emits for `await` operands.
pub const AWAIT_FN: &str = "__tb_await__";
/// Identifiers the parser emits for `yield` / `yield*` operands inside a
/// generator. The same hoist+lower machinery handles all suspension markers;
/// only the lowerer's op emission distinguishes await from yield.
pub const YIELD_FN: &str = "__tb_yield__";
pub const YIELD_STAR_FN: &str = "__tb_yield_star__";

#[derive(Debug, Clone, Copy, PartialEq)]
enum Suspend {
    Await,
    Yield,
    YieldStar,
}

/// If `e` is a suspension-marker call, return its operand and which kind.
fn suspend_call(e: &Expr) -> Option<(&Expr, Suspend)> {
    if let Expr::Call { callee, args } = e {
        if let Expr::Identifier(name) = callee.as_ref() {
            let kind = match name.as_str() {
                AWAIT_FN => Suspend::Await,
                YIELD_FN => Suspend::Yield,
                YIELD_STAR_FN => Suspend::YieldStar,
                _ => return None,
            };
            return Some((args.first().unwrap_or(&Expr::Undefined), kind));
        }
    }
    None
}

/// A compiled async function body: a flat instruction list run by the
/// interpreter's resumable driver.
#[derive(Debug, Clone, PartialEq)]
pub struct Machine {
    pub ops: Vec<AsyncOp>,
}

/// One instruction of a lowered async body. The driver keeps a program counter,
/// the function scope, and a stack of active try-handlers.
#[derive(Debug, Clone, PartialEq)]
pub enum AsyncOp {
    /// Run a fully-synchronous statement via the normal interpreter. Contains no
    /// suspension point and no async-relevant control flow (those are lowered).
    Exec(Stmt),
    /// Evaluate `0` (the await operand) and suspend: the driver resolves it to a
    /// promise, subscribes, returns to the event loop, and on settle resumes at
    /// the NEXT op with the resolved value as the pending "sent" value (a
    /// rejection resumes by throwing here).
    Await(Expr),
    /// `yield expr` (generator-only). Evaluate `expr` and suspend, surfacing the
    /// value to the `.next()` caller. `delegate` = `yield*` (iterate the operand,
    /// yielding each element). On resume, the value passed to `.next(v)` becomes
    /// the pending "sent" value (the result of the `yield` expression).
    Yield { expr: Expr, delegate: bool },
    /// Bind the pending sent value into `target` (an lvalue) — declaring it with
    /// `decl` (let/const/var) or assigning when `decl` is None. Consumes the sent value.
    StoreResume { target: Expr, decl: Option<VarKind> },
    /// Discard the pending sent value (a bare `await E;`).
    DropResume,
    /// Unconditional jump.
    Goto(usize),
    /// Evaluate `test`; jump to `target` when FALSY.
    JumpIfFalsy { test: Expr, target: usize },
    /// `return expr?;` — settle the result promise with the value (or undefined).
    Return(Option<Expr>),
    /// `throw expr;` — routed to the nearest handler, else rejects the result.
    Throw(Expr),
    /// Push a try-handler active until the matching `PopHandler`. A throw in
    /// between transfers to `catch_pc` (binding the error to `catch_param`),
    /// otherwise propagates to the next outer handler / rejects the result.
    PushHandler {
        catch_pc: usize,
        catch_param: Option<String>,
    },
    /// Pop the most-recent handler (its try block completed without throwing).
    PopHandler,
    /// Enter a fresh lexical block scope (child of the current). Keeps `let`/
    /// `const` (incl. loop variables) block-scoped across suspension, so a
    /// loop variable can't leak and shadow an enclosing binding.
    PushScope,
    /// Leave the current block scope (restore its parent).
    PopScope,
}

/// Hoist + lower an async function body into a `Machine`, or `None` if it uses a
/// construct not yet lowered (caller keeps the legacy synchronous path).
pub fn compile(body: &[Stmt]) -> Option<Machine> {
    if !body.iter().any(stmt_has_await) {
        return None; // no await at all — no state machine needed
    }
    if !body.iter().all(lowerable_stmt) {
        return None; // contains an await inside a not-yet-lowered construct
    }
    let mut hoister = Hoister::default();
    let hoisted = hoister.hoist_stmts(body.to_vec());
    let mut lowerer = Lowerer::default();
    lowerer.lower_stmts(&hoisted);
    lowerer.ops.push(AsyncOp::Return(None));
    Some(Machine { ops: lowerer.ops })
}

/// Compile a GENERATOR body into a resumable `Machine`. Unlike `compile`
/// (async), this always produces a machine — a generator must run lazily even
/// with no `yield` (its body runs on the first `.next()`, not at call time) —
/// or `None` if the body uses a construct not yet lowerable (the caller then
/// falls back to an eager iterator). `yield` markers lower to `Yield` ops.
pub fn compile_generator(body: &[Stmt]) -> Option<Machine> {
    if !body.iter().all(lowerable_stmt) {
        return None;
    }
    let mut hoister = Hoister::default();
    let hoisted = hoister.hoist_stmts(body.to_vec());
    let mut lowerer = Lowerer::default();
    lowerer.lower_stmts(&hoisted);
    lowerer.ops.push(AsyncOp::Return(None));
    Some(Machine { ops: lowerer.ops })
}

// ---------------------------------------------------------------------------
// await detection
// ---------------------------------------------------------------------------

/// If `e` is any suspension marker (`await`/`yield`/`yield*`), return its
/// operand. Named `as_await` historically; the detection/hoisting machinery is
/// kind-agnostic (only the lowerer's emission distinguishes them).
fn as_await(e: &Expr) -> Option<&Expr> {
    suspend_call(e).map(|(operand, _)| operand)
}

fn expr_has_await(e: &Expr) -> bool {
    if as_await(e).is_some() {
        return true;
    }
    match e {
        Expr::Number(_)
        | Expr::BigInt(_)
        | Expr::String(_)
        | Expr::TemplateLiteral(_)
        | Expr::Boolean(_)
        | Expr::Null
        | Expr::Undefined
        | Expr::This
        | Expr::NewTarget
        | Expr::Identifier(_)
        | Expr::Regex(_, _) => false,
        Expr::Array(items) => items.iter().any(expr_has_await),
        Expr::Object(props) => props.iter().any(|(_, v)| expr_has_await(v)),
        Expr::Unary { target, .. } | Expr::Update { target, .. } | Expr::Spread(target) => {
            expr_has_await(target)
        }
        Expr::Binary { left, right, .. } | Expr::Logical { left, right, .. } => {
            expr_has_await(left) || expr_has_await(right)
        }
        Expr::Conditional { test, cons, alt } => {
            expr_has_await(test) || expr_has_await(cons) || expr_has_await(alt)
        }
        Expr::Assignment { target, value, .. } => expr_has_await(target) || expr_has_await(value),
        Expr::Member {
            object, property, ..
        } => expr_has_await(object) || expr_has_await(property),
        Expr::Call { callee, args } | Expr::New { callee, args } => {
            expr_has_await(callee) || args.iter().any(expr_has_await)
        }
        Expr::Sequence(items) => items.iter().any(expr_has_await),
        // `import.meta` has no sub-expressions; `import(spec)` may await in its
        // specifier (`import(await urlFor(x))`).
        Expr::ImportMeta => false,
        Expr::DynamicImport(spec) => expr_has_await(spec),
        // Nested functions are their own async context — don't descend.
        Expr::Function { .. } | Expr::Arrow { .. } => false,
    }
}

fn opt_await(e: &Option<Expr>) -> bool {
    e.as_ref().is_some_and(expr_has_await)
}

fn stmt_has_await(s: &Stmt) -> bool {
    match s {
        Stmt::Empty | Stmt::Break(_) | Stmt::Continue(_) | Stmt::FunctionDecl { .. } => false,
        // Module declarations don't introduce await points relevant to async
        // function lowering (top-level await is handled by the module graph,
        // not the async-function lowerer). `export <decl>` could carry an
        // initializer with await, so descend into the inner declaration.
        Stmt::Import { .. } => false,
        Stmt::Export { decl, .. } => decl.as_ref().is_some_and(|d| stmt_has_await(d)),
        Stmt::Expression(e) | Stmt::Throw(e) => expr_has_await(e),
        Stmt::Return(e) => opt_await(e),
        Stmt::VarDecl { decls, .. } => decls.iter().any(|d| opt_await(&d.init)),
        Stmt::Block(stmts) => stmts.iter().any(stmt_has_await),
        Stmt::If { test, cons, alt } => {
            expr_has_await(test)
                || stmt_has_await(cons)
                || alt.as_ref().is_some_and(|a| stmt_has_await(a))
        }
        Stmt::While { test, body } | Stmt::DoWhile { body, test } => {
            expr_has_await(test) || stmt_has_await(body)
        }
        Stmt::For {
            init,
            test,
            update,
            body,
        } => {
            init.as_ref().is_some_and(|i| match i {
                crate::ast::ForInit::Expr(e) => expr_has_await(e),
                crate::ast::ForInit::VarDecl { decls, .. } => {
                    decls.iter().any(|d| opt_await(&d.init))
                }
            }) || opt_await(test)
                || opt_await(update)
                || stmt_has_await(body)
        }
        Stmt::ForIn { source, body, .. } => expr_has_await(source) || stmt_has_await(body),
        Stmt::ForOf {
            source,
            body,
            is_await,
            ..
        } => *is_await || expr_has_await(source) || stmt_has_await(body),
        Stmt::Labeled { body, .. } => stmt_has_await(body),
        Stmt::Try {
            block,
            catch_block,
            finally_block,
            ..
        } => {
            block.iter().any(stmt_has_await)
                || catch_block
                    .as_ref()
                    .is_some_and(|b| b.iter().any(stmt_has_await))
                || finally_block
                    .as_ref()
                    .is_some_and(|b| b.iter().any(stmt_has_await))
        }
        Stmt::Switch {
            discriminant,
            cases,
            ..
        } => {
            expr_has_await(discriminant)
                || cases.iter().any(|c| {
                    c.test.as_ref().is_some_and(expr_has_await) || c.body.iter().any(stmt_has_await)
                })
        }
    }
}

/// Whether a statement's await points are all inside constructs this milestone
/// can lower. A non-suspending statement is always fine (it runs via `Exec`).
fn lowerable_stmt(s: &Stmt) -> bool {
    if !stmt_has_await(s) {
        return true;
    }
    match s {
        Stmt::Expression(_) | Stmt::Throw(_) | Stmt::Return(_) | Stmt::VarDecl { .. } => true,
        Stmt::Block(stmts) => stmts.iter().all(lowerable_stmt),
        Stmt::If { cons, alt, .. } => {
            lowerable_stmt(cons) && alt.as_ref().is_none_or(|a| lowerable_stmt(a))
        }
        Stmt::Try {
            block,
            catch_block,
            finally_block,
            ..
        } => {
            // `finally` through a suspension needs the pending-completion model
            // (its own milestone). Until then a suspending try/finally keeps the
            // legacy path; try/catch is fully lowered here.
            finally_block.is_none()
                && block.iter().all(lowerable_stmt)
                && catch_block
                    .as_ref()
                    .is_none_or(|b| b.iter().all(lowerable_stmt))
        }
        // Loops: while/do-while/for/for-of/for-in with await are lowered (the
        // hoister desugars for/for-of/for-in to a `while`). break/continue
        // (incl. labeled) resolve to loop edges.
        Stmt::While { body, .. } | Stmt::DoWhile { body, .. } => lowerable_stmt(body),
        Stmt::For { body, .. } | Stmt::ForIn { body, .. } | Stmt::Labeled { body, .. } => {
            lowerable_stmt(body)
        }
        Stmt::ForOf { is_await, body, .. } => !*is_await && lowerable_stmt(body),
        Stmt::Break(_) | Stmt::Continue(_) | Stmt::Empty => true,
        // `switch` with await: a later milestone.
        _ => false,
    }
}

// ---------------------------------------------------------------------------
// Stage 1: hoist await out to statement level
// ---------------------------------------------------------------------------

#[derive(Default)]
struct Hoister {
    counter: usize,
}

impl Hoister {
    fn fresh(&mut self) -> String {
        let n = self.counter;
        self.counter += 1;
        format!("__tb_aw{n}")
    }

    fn hoist_stmts(&mut self, stmts: Vec<Stmt>) -> Vec<Stmt> {
        let mut out = Vec::with_capacity(stmts.len());
        for s in stmts {
            out.extend(self.hoist_stmt(s));
        }
        out
    }

    fn hoist_box(&mut self, body: Box<Stmt>) -> Box<Stmt> {
        match *body {
            Stmt::Block(stmts) => Box::new(Stmt::Block(self.hoist_stmts(stmts))),
            other => {
                let mut hoisted = self.hoist_stmt(other);
                if hoisted.len() == 1 {
                    Box::new(hoisted.pop().unwrap())
                } else {
                    Box::new(Stmt::Block(hoisted))
                }
            }
        }
    }

    fn hoist_stmt(&mut self, s: Stmt) -> Vec<Stmt> {
        if !stmt_has_await(&s) {
            return vec![s];
        }
        match s {
            Stmt::Expression(e) => {
                let mut prelude = Vec::new();
                let e2 = self.hoist_expr(e, &mut prelude);
                prelude.push(Stmt::Expression(e2));
                prelude
            }
            Stmt::Throw(e) => {
                let mut prelude = Vec::new();
                let e2 = self.hoist_expr(e, &mut prelude);
                prelude.push(Stmt::Throw(e2));
                prelude
            }
            Stmt::Return(Some(e)) => {
                let mut prelude = Vec::new();
                let e2 = self.hoist_expr(e, &mut prelude);
                prelude.push(Stmt::Return(Some(e2)));
                prelude
            }
            Stmt::Return(None) => vec![Stmt::Return(None)],
            Stmt::VarDecl { kind, decls } => {
                let mut out = Vec::new();
                for d in decls {
                    let mut prelude = Vec::new();
                    let init = d.init.map(|e| self.hoist_expr(e, &mut prelude));
                    out.extend(prelude);
                    out.push(Stmt::VarDecl {
                        kind,
                        decls: vec![VarDeclarator { name: d.name, init }],
                    });
                }
                out
            }
            Stmt::Block(stmts) => vec![Stmt::Block(self.hoist_stmts(stmts))],
            Stmt::If { test, cons, alt } => {
                let mut prelude = Vec::new();
                let test2 = self.hoist_expr(test, &mut prelude);
                let cons2 = self.hoist_box(cons);
                let alt2 = alt.map(|a| self.hoist_box(a));
                prelude.push(Stmt::If {
                    test: test2,
                    cons: cons2,
                    alt: alt2,
                });
                prelude
            }
            Stmt::Try {
                block,
                catch_param,
                catch_block,
                finally_block,
            } => vec![Stmt::Try {
                block: self.hoist_stmts(block),
                catch_param,
                catch_block: catch_block.map(|b| self.hoist_stmts(b)),
                finally_block: finally_block.map(|b| self.hoist_stmts(b)),
            }],
            Stmt::Labeled { label, body } => vec![Stmt::Labeled {
                label,
                body: self.hoist_box(body),
            }],
            Stmt::While { test, body } => {
                if expr_has_await(&test) {
                    // `while (await c) body` → re-evaluate the condition each
                    // iteration: `while (true) { let $c = c; if (!$c) break; body }`.
                    let mut head = Vec::new();
                    let tmp = self.fresh();
                    let cond = self.hoist_expr(test, &mut head);
                    head.push(decl_let(&tmp, Some(cond)));
                    head.push(break_if_falsy(&tmp));
                    head.push(*self.hoist_box(body));
                    vec![while_true(head)]
                } else {
                    vec![Stmt::While {
                        test,
                        body: self.hoist_box(body),
                    }]
                }
            }
            Stmt::DoWhile { body, test } => {
                // `do body while (c)` → flag-guarded: test runs AFTER the body.
                let first = self.fresh();
                let mut out = vec![decl_let(&first, Some(Expr::Boolean(true)))];
                let mut loop_body = Vec::new();
                let mut t_prelude = Vec::new();
                let cond = self.hoist_expr(test, &mut t_prelude);
                let tmp = self.fresh();
                let mut cond_block = t_prelude;
                cond_block.push(decl_let(&tmp, Some(cond)));
                cond_block.push(break_if_falsy(&tmp));
                loop_body.push(Stmt::If {
                    test: not_expr(Expr::Identifier(first.clone())),
                    cons: Box::new(Stmt::Block(cond_block)),
                    alt: None,
                });
                loop_body.push(assign_stmt(&first, Expr::Boolean(false)));
                loop_body.push(*self.hoist_box(body));
                out.push(while_true(loop_body));
                out
            }
            Stmt::For {
                init,
                test,
                update,
                body,
            } => self.hoist_for(init, test, update, body),
            Stmt::ForOf {
                kind,
                name,
                source,
                body,
                ..
            } => self.hoist_for_of(kind, name, source, body),
            Stmt::ForIn {
                kind,
                name,
                source,
                body,
            } => self.hoist_for_in(kind, name, source, body),
            // `lowerable_stmt` guarantees we never reach a suspending construct
            // outside the set above.
            other => vec![other],
        }
    }

    /// `for (init; test; update) body` → `init; let $first=true; while(true){
    /// if(!$first) update; $first=false; if(!test) break; body }`. Running
    /// `update` at the loop TOP (skipped the first time) makes `continue` — which
    /// jumps to the top — correctly run `update` before re-testing.
    ///
    /// Per-iteration binding (ECMA-262 §14.7.4.2): when `init` is `let`/`const`,
    /// each loop iteration must have its own copy of the bindings so that closures
    /// capture the value at the time of creation, not the final loop value.  We
    /// achieve this by wrapping the body in a block that re-declares every loop
    /// variable as `let name = name` — the RHS reads the outer (updated) value,
    /// the LHS creates a fresh block-scoped binding that shadows it.
    fn hoist_for(
        &mut self,
        init: Option<ForInit>,
        test: Option<Expr>,
        update: Option<Expr>,
        body: Box<Stmt>,
    ) -> Vec<Stmt> {
        let mut out = Vec::new();
        // Collect `let`/`const` variable names for per-iteration copies.
        let mut per_iter_names: Vec<String> = Vec::new();
        match init {
            Some(ForInit::VarDecl { kind, decls }) => {
                if matches!(kind, VarKind::Let | VarKind::Const) {
                    for d in &decls {
                        per_iter_names.push(d.name.clone());
                    }
                }
                out.extend(self.hoist_stmt(Stmt::VarDecl { kind, decls }));
            }
            Some(ForInit::Expr(e)) => out.extend(self.hoist_stmt(Stmt::Expression(e))),
            None => {}
        }
        let first = self.fresh();
        out.push(decl_let(&first, Some(Expr::Boolean(true))));
        let mut loop_body = Vec::new();
        if let Some(update) = update {
            let mut upd_prelude = Vec::new();
            let upd = self.hoist_expr(update, &mut upd_prelude);
            let mut upd_block = upd_prelude;
            upd_block.push(Stmt::Expression(upd));
            loop_body.push(Stmt::If {
                test: not_expr(Expr::Identifier(first.clone())),
                cons: Box::new(Stmt::Block(upd_block)),
                alt: None,
            });
        }
        loop_body.push(assign_stmt(&first, Expr::Boolean(false)));
        if let Some(test) = test {
            let mut t_prelude = Vec::new();
            let t = self.hoist_expr(test, &mut t_prelude);
            loop_body.extend(t_prelude);
            let tmp = self.fresh();
            loop_body.push(decl_let(&tmp, Some(t)));
            loop_body.push(break_if_falsy(&tmp));
        }
        if per_iter_names.is_empty() {
            loop_body.push(*self.hoist_box(body));
        } else {
            // Wrap the body in a block: `{ let i = i; let j = j; ... BODY }`.
            // Each `let name = name` creates a fresh block-scoped binding
            // initialised from the outer (shared) loop variable.  Closures
            // inside BODY then close over this per-iteration copy.
            let mut iter_block: Vec<Stmt> = per_iter_names
                .into_iter()
                .map(|name| Stmt::VarDecl {
                    kind: VarKind::Let,
                    decls: vec![VarDeclarator {
                        init: Some(Expr::Identifier(name.clone())),
                        name,
                    }],
                })
                .collect();
            iter_block.push(*self.hoist_box(body));
            loop_body.push(Stmt::Block(iter_block));
        }
        out.push(while_true(loop_body));
        out
    }

    /// `for (kind name of src) body` — lazy iterator protocol desugaring:
    ///
    /// ```text
    /// let $iter = __tb_get_iterator__(src);
    /// let $step;
    /// while (!($step = $iter.next()).done) {
    ///     kind name = $step.value;
    ///     body
    /// }
    /// ```
    ///
    /// This drives the iterator lazily (one `next()` per iteration) rather than
    /// eagerly draining `[...src]` up front, so generators and other
    /// single-pass iterables work correctly.
    fn hoist_for_of(
        &mut self,
        kind: Option<VarKind>,
        name: String,
        source: Expr,
        body: Box<Stmt>,
    ) -> Vec<Stmt> {
        let mut out = Vec::new();
        let mut src_prelude = Vec::new();
        let src = self.hoist_expr(source, &mut src_prelude);
        out.extend(src_prelude);
        let iter = self.fresh();
        let step = self.fresh();
        // let $iter = __tb_get_iterator__(src);
        out.push(decl_let(
            &iter,
            Some(Expr::Call {
                callee: Box::new(Expr::Identifier("__tb_get_iterator__".into())),
                args: vec![src],
            }),
        ));
        // let $step;  (declared outside the loop so it's accessible in the test)
        out.push(decl_let(&step, None));
        // while loop test: `!($step = $iter.next()).done`
        // i.e.: assign result of iter.next() to $step, then check !$step.done
        let next_call = Expr::Call {
            callee: Box::new(Expr::Member {
                object: Box::new(Expr::Identifier(iter.clone())),
                property: Box::new(Expr::Identifier("next".into())),
                computed: false,
            }),
            args: vec![],
        };
        let assign_step = Expr::Assignment {
            op: AssignOp::Assign,
            target: Box::new(Expr::Identifier(step.clone())),
            value: Box::new(next_call),
        };
        let done_prop = Expr::Member {
            object: Box::new(assign_step),
            property: Box::new(Expr::Identifier("done".into())),
            computed: false,
        };
        let loop_test = not_expr(done_prop);
        // loop body: `kind name = $step.value; BODY`
        let value_expr = Expr::Member {
            object: Box::new(Expr::Identifier(step.clone())),
            property: Box::new(Expr::Identifier("value".into())),
            computed: false,
        };
        let mut loop_body = vec![Stmt::VarDecl {
            kind: kind.unwrap_or(VarKind::Let),
            decls: vec![VarDeclarator {
                name,
                init: Some(value_expr),
            }],
        }];
        loop_body.push(*self.hoist_box(body));
        out.push(Stmt::While {
            test: loop_test,
            body: Box::new(Stmt::Block(loop_body)),
        });
        out
    }

    /// `for (kind name in src) body` → `let $k=__tb_enum_keys__(src); let $i=0;
    /// while($i<$k.length){ kind name=$k[$i]; $i=$i+1; body }`.
    ///
    /// Uses `__tb_enum_keys__` instead of `Object.keys` so that inherited
    /// enumerable properties are included, matching ECMA-262 §14.7.5.6
    /// `EnumerateObjectProperties` (which walks the prototype chain).
    fn hoist_for_in(
        &mut self,
        kind: Option<VarKind>,
        name: String,
        source: Expr,
        body: Box<Stmt>,
    ) -> Vec<Stmt> {
        let mut out = Vec::new();
        let mut src_prelude = Vec::new();
        let src = self.hoist_expr(source, &mut src_prelude);
        out.extend(src_prelude);
        let keys = self.fresh();
        let idx = self.fresh();
        out.push(decl_let(
            &keys,
            Some(Expr::Call {
                callee: Box::new(Expr::Identifier("__tb_enum_keys__".into())),
                args: vec![src],
            }),
        ));
        out.push(decl_let(&idx, Some(Expr::Number(0.0))));
        let mut loop_body = vec![Stmt::VarDecl {
            kind: kind.unwrap_or(VarKind::Let),
            decls: vec![VarDeclarator {
                name,
                init: Some(index_expr(&keys, &idx)),
            }],
        }];
        loop_body.push(incr_stmt(&idx));
        loop_body.push(*self.hoist_box(body));
        out.push(Stmt::While {
            test: less_than_len(&idx, &keys),
            body: Box::new(Stmt::Block(loop_body)),
        });
        out
    }

    /// Hoist all awaits out of `e`, appending statements to `prelude`, returning
    /// an equivalent await-free expression.
    fn hoist_expr(&mut self, e: Expr, prelude: &mut Vec<Stmt>) -> Expr {
        if !expr_has_await(&e) {
            return e;
        }
        if as_await(&e).is_some() {
            if let Expr::Call { callee, mut args } = e {
                let operand = if args.is_empty() {
                    Expr::Undefined
                } else {
                    args.remove(0)
                };
                let operand = self.hoist_expr(operand, prelude);
                let tmp = self.fresh();
                prelude.push(Stmt::VarDecl {
                    kind: VarKind::Let,
                    decls: vec![VarDeclarator {
                        name: tmp.clone(),
                        init: Some(Expr::Call {
                            callee,
                            args: vec![operand],
                        }),
                    }],
                });
                return Expr::Identifier(tmp);
            }
            unreachable!();
        }
        match e {
            Expr::Unary { op, target } => Expr::Unary {
                op,
                target: Box::new(self.hoist_expr(*target, prelude)),
            },
            Expr::Update { op, target, prefix } => Expr::Update {
                op,
                target: Box::new(self.hoist_expr(*target, prelude)),
                prefix,
            },
            Expr::Spread(inner) => Expr::Spread(Box::new(self.hoist_expr(*inner, prelude))),
            Expr::Binary { op, left, right } => {
                let l = self.hoist_expr(*left, prelude);
                let r = self.hoist_expr(*right, prelude);
                Expr::Binary {
                    op,
                    left: Box::new(l),
                    right: Box::new(r),
                }
            }
            Expr::Member {
                object,
                property,
                computed,
            } => {
                let o = self.hoist_expr(*object, prelude);
                let p = self.hoist_expr(*property, prelude);
                Expr::Member {
                    object: Box::new(o),
                    property: Box::new(p),
                    computed,
                }
            }
            Expr::Assignment { op, target, value } => {
                let t = self.hoist_expr(*target, prelude);
                let v = self.hoist_expr(*value, prelude);
                Expr::Assignment {
                    op,
                    target: Box::new(t),
                    value: Box::new(v),
                }
            }
            Expr::Call { callee, args } => {
                let c = self.hoist_expr(*callee, prelude);
                let args = args
                    .into_iter()
                    .map(|a| self.hoist_expr(a, prelude))
                    .collect();
                Expr::Call {
                    callee: Box::new(c),
                    args,
                }
            }
            Expr::New { callee, args } => {
                let c = self.hoist_expr(*callee, prelude);
                let args = args
                    .into_iter()
                    .map(|a| self.hoist_expr(a, prelude))
                    .collect();
                Expr::New {
                    callee: Box::new(c),
                    args,
                }
            }
            Expr::Array(items) => Expr::Array(
                items
                    .into_iter()
                    .map(|i| self.hoist_expr(i, prelude))
                    .collect(),
            ),
            Expr::Object(props) => Expr::Object(
                props
                    .into_iter()
                    .map(|(k, v)| (k, self.hoist_expr(v, prelude)))
                    .collect(),
            ),
            Expr::Sequence(items) => {
                let n = items.len();
                let mut last = Expr::Undefined;
                for (i, it) in items.into_iter().enumerate() {
                    let e2 = self.hoist_expr(it, prelude);
                    if i + 1 == n {
                        last = e2;
                    } else {
                        prelude.push(Stmt::Expression(e2));
                    }
                }
                last
            }
            Expr::Logical { op, left, right } => {
                let l = self.hoist_expr(*left, prelude);
                if !expr_has_await(&right) {
                    return Expr::Logical {
                        op,
                        left: Box::new(l),
                        right,
                    };
                }
                let tmp = self.fresh();
                prelude.push(decl_let(&tmp, Some(l)));
                let cond = match op {
                    LogicalOp::And => Expr::Identifier(tmp.clone()),
                    LogicalOp::Or => not_expr(Expr::Identifier(tmp.clone())),
                    LogicalOp::Nullish => Expr::Binary {
                        op: BinOp::EqEq,
                        left: Box::new(Expr::Identifier(tmp.clone())),
                        right: Box::new(Expr::Null),
                    },
                };
                let mut branch = Vec::new();
                let r = self.hoist_expr(*right, &mut branch);
                branch.push(assign_stmt(&tmp, r));
                prelude.push(Stmt::If {
                    test: cond,
                    cons: Box::new(Stmt::Block(branch)),
                    alt: None,
                });
                Expr::Identifier(tmp)
            }
            Expr::Conditional { test, cons, alt } => {
                let t = self.hoist_expr(*test, prelude);
                if !expr_has_await(&cons) && !expr_has_await(&alt) {
                    return Expr::Conditional {
                        test: Box::new(t),
                        cons,
                        alt,
                    };
                }
                let tmp = self.fresh();
                prelude.push(decl_let(&tmp, None));
                let mut cons_b = Vec::new();
                let c = self.hoist_expr(*cons, &mut cons_b);
                cons_b.push(assign_stmt(&tmp, c));
                let mut alt_b = Vec::new();
                let a = self.hoist_expr(*alt, &mut alt_b);
                alt_b.push(assign_stmt(&tmp, a));
                prelude.push(Stmt::If {
                    test: t,
                    cons: Box::new(Stmt::Block(cons_b)),
                    alt: Some(Box::new(Stmt::Block(alt_b))),
                });
                Expr::Identifier(tmp)
            }
            other => other,
        }
    }
}

fn not_expr(e: Expr) -> Expr {
    Expr::Unary {
        op: UnaryOp::Not,
        target: Box::new(e),
    }
}

fn decl_let(name: &str, init: Option<Expr>) -> Stmt {
    Stmt::VarDecl {
        kind: VarKind::Let,
        decls: vec![VarDeclarator {
            name: name.to_string(),
            init,
        }],
    }
}

fn assign_stmt(name: &str, value: Expr) -> Stmt {
    Stmt::Expression(Expr::Assignment {
        op: AssignOp::Assign,
        target: Box::new(Expr::Identifier(name.to_string())),
        value: Box::new(value),
    })
}

fn break_if_falsy(name: &str) -> Stmt {
    Stmt::If {
        test: not_expr(Expr::Identifier(name.to_string())),
        cons: Box::new(Stmt::Break(None)),
        alt: None,
    }
}

fn while_true(body: Vec<Stmt>) -> Stmt {
    Stmt::While {
        test: Expr::Boolean(true),
        body: Box::new(Stmt::Block(body)),
    }
}

/// `arr[idx]` (computed member).
fn index_expr(arr: &str, idx: &str) -> Expr {
    Expr::Member {
        object: Box::new(Expr::Identifier(arr.to_string())),
        property: Box::new(Expr::Identifier(idx.to_string())),
        computed: true,
    }
}

/// `idx = idx + 1`.
fn incr_stmt(idx: &str) -> Stmt {
    assign_stmt(
        idx,
        Expr::Binary {
            op: BinOp::Add,
            left: Box::new(Expr::Identifier(idx.to_string())),
            right: Box::new(Expr::Number(1.0)),
        },
    )
}

/// `idx < arr.length`.
fn less_than_len(idx: &str, arr: &str) -> Expr {
    Expr::Binary {
        op: BinOp::Lt,
        left: Box::new(Expr::Identifier(idx.to_string())),
        right: Box::new(Expr::Member {
            object: Box::new(Expr::Identifier(arr.to_string())),
            property: Box::new(Expr::Identifier("length".into())),
            computed: false,
        }),
    }
}

// ---------------------------------------------------------------------------
// Stage 2: lower hoisted statements → flat op list
// ---------------------------------------------------------------------------

/// Whether `s` contains a `break`/`continue` that would escape `s` to an
/// enclosing loop (i.e. not captured by a loop nested inside `s`). Such a
/// statement can't be `Exec`'d wholesale even when it has no `await` — the jump
/// must become a `Goto` to the lowered loop's edge.
fn stmt_has_loop_jump(s: &Stmt) -> bool {
    match s {
        Stmt::Break(_) | Stmt::Continue(_) => true,
        Stmt::Block(stmts) => stmts.iter().any(stmt_has_loop_jump),
        Stmt::If { cons, alt, .. } => {
            stmt_has_loop_jump(cons) || alt.as_ref().is_some_and(|a| stmt_has_loop_jump(a))
        }
        Stmt::Labeled { body, .. } => stmt_has_loop_jump(body),
        Stmt::Try {
            block,
            catch_block,
            finally_block,
            ..
        } => {
            block.iter().any(stmt_has_loop_jump)
                || catch_block
                    .as_ref()
                    .is_some_and(|b| b.iter().any(stmt_has_loop_jump))
                || finally_block
                    .as_ref()
                    .is_some_and(|b| b.iter().any(stmt_has_loop_jump))
        }
        // A nested loop/switch captures unlabeled break/continue. (Labeled jumps
        // escaping a nested loop are rare and not on the lowered paths here.)
        _ => false,
    }
}

#[derive(Default)]
struct Lowerer {
    ops: Vec<AsyncOp>,
    /// Active loops, innermost last — for resolving break/continue to op edges.
    loops: Vec<LoopFrame>,
    /// Label attached to the next loop (from an enclosing `label:` statement).
    pending_label: Option<String>,
}

struct LoopFrame {
    /// Where `continue` jumps (the loop's re-test / top).
    continue_target: usize,
    label: Option<String>,
    /// `Goto` op indices emitted for `break`s; patched to the loop end.
    breaks: Vec<usize>,
}

impl Lowerer {
    fn emit(&mut self, op: AsyncOp) -> usize {
        let i = self.ops.len();
        self.ops.push(op);
        i
    }

    /// Emit the suspension op for `operand` of the given kind: `Await` for
    /// async, `Yield {delegate}` for generator `yield` / `yield*`.
    fn emit_suspend(&mut self, operand: &Expr, kind: Suspend) {
        match kind {
            Suspend::Await => {
                self.emit(AsyncOp::Await(operand.clone()));
            }
            Suspend::Yield => {
                self.emit(AsyncOp::Yield {
                    expr: operand.clone(),
                    delegate: false,
                });
            }
            Suspend::YieldStar => {
                self.emit(AsyncOp::Yield {
                    expr: operand.clone(),
                    delegate: true,
                });
            }
        }
    }

    fn here(&self) -> usize {
        self.ops.len()
    }

    fn patch(&mut self, at: usize, target: usize) {
        match &mut self.ops[at] {
            AsyncOp::Goto(t) | AsyncOp::JumpIfFalsy { target: t, .. } => *t = target,
            _ => unreachable!("patched a non-jump op"),
        }
    }

    /// Resolve a break/continue target loop frame (innermost, or by label).
    fn find_loop(&self, label: Option<&str>) -> Option<usize> {
        match label {
            None => self.loops.iter().rposition(|_| true),
            Some(l) => self
                .loops
                .iter()
                .rposition(|f| f.label.as_deref() == Some(l)),
        }
    }

    fn lower_stmts(&mut self, stmts: &[Stmt]) {
        for s in stmts {
            self.lower_stmt(s);
        }
    }

    fn lower_stmt(&mut self, s: &Stmt) {
        // break/continue inside a lowered loop become jumps to its edges. (They
        // reach here only when lowered structurally — a break inside an `Exec`'d
        // synchronous loop is handled by that loop's own `exec_stmt`.)
        match s {
            Stmt::Break(label) => {
                if let Some(i) = self.find_loop(label.as_deref()) {
                    let g = self.emit(AsyncOp::Goto(usize::MAX));
                    self.loops[i].breaks.push(g);
                }
                return;
            }
            Stmt::Continue(label) => {
                if let Some(i) = self.find_loop(label.as_deref()) {
                    let target = self.loops[i].continue_target;
                    self.emit(AsyncOp::Goto(target));
                }
                return;
            }
            _ => {}
        }
        // Fully-synchronous statement with no escaping loop jump → run wholesale.
        if !stmt_has_await(s) && !stmt_has_loop_jump(s) {
            self.emit(AsyncOp::Exec(s.clone()));
            return;
        }
        match s {
            Stmt::Block(stmts) => {
                self.emit(AsyncOp::PushScope);
                self.lower_stmts(stmts);
                self.emit(AsyncOp::PopScope);
            }
            Stmt::Labeled { label, body } => {
                self.pending_label = Some(label.clone());
                self.lower_stmt(body);
                self.pending_label = None;
            }
            Stmt::While { test, body } => {
                let label = self.pending_label.take();
                let top = self.here();
                let exit = if matches!(test, Expr::Boolean(true)) {
                    None
                } else {
                    Some(self.emit(AsyncOp::JumpIfFalsy {
                        test: test.clone(),
                        target: usize::MAX,
                    }))
                };
                self.loops.push(LoopFrame {
                    continue_target: top,
                    label,
                    breaks: Vec::new(),
                });
                self.lower_stmt(body);
                self.emit(AsyncOp::Goto(top));
                let end = self.here();
                let frame = self.loops.pop().unwrap();
                for b in frame.breaks {
                    self.patch(b, end);
                }
                if let Some(jf) = exit {
                    self.patch(jf, end);
                }
            }
            Stmt::Expression(e) => {
                if let Some((operand, kind)) = suspend_call(e) {
                    self.emit_suspend(operand, kind);
                    self.emit(AsyncOp::DropResume);
                } else {
                    self.emit(AsyncOp::Exec(Stmt::Expression(e.clone())));
                }
            }
            Stmt::VarDecl { kind, decls } => {
                // After hoisting: exactly `let x = <suspend> E`.
                let d = &decls[0];
                if let Some(init) = &d.init {
                    if let Some((operand, skind)) = suspend_call(init) {
                        self.emit_suspend(operand, skind);
                        self.emit(AsyncOp::StoreResume {
                            target: Expr::Identifier(d.name.clone()),
                            decl: Some(*kind),
                        });
                        return;
                    }
                }
                self.emit(AsyncOp::Exec(s.clone()));
            }
            Stmt::Return(e) => {
                self.emit(AsyncOp::Return(e.clone()));
            }
            Stmt::Throw(e) => {
                self.emit(AsyncOp::Throw(e.clone()));
            }
            Stmt::If { test, cons, alt } => {
                let jf = self.emit(AsyncOp::JumpIfFalsy {
                    test: test.clone(),
                    target: usize::MAX,
                });
                self.lower_stmt(cons);
                if let Some(alt) = alt {
                    let goto_end = self.emit(AsyncOp::Goto(usize::MAX));
                    let else_pc = self.here();
                    self.patch(jf, else_pc);
                    self.lower_stmt(alt);
                    let end = self.here();
                    self.patch(goto_end, end);
                } else {
                    let end = self.here();
                    self.patch(jf, end);
                }
            }
            Stmt::Try {
                block,
                catch_param,
                catch_block,
                finally_block,
            } => self.lower_try(block, catch_param, catch_block, finally_block),
            // lowerable_stmt guarantees nothing else suspending reaches here.
            other => {
                self.emit(AsyncOp::Exec(other.clone()));
            }
        }
    }

    fn lower_try(
        &mut self,
        block: &[Stmt],
        catch_param: &Option<String>,
        catch_block: &Option<Vec<Stmt>>,
        _finally_block: &Option<Vec<Stmt>>, // gated to None by `lowerable_stmt`
    ) {
        // try/catch layout (no finally — gated):
        //   push:   PushHandler{catch_pc}
        //           <try ops>
        //           PopHandler            ; try completed without throwing
        //           Goto after
        //   catch:                        ; driver popped the handler + bound err
        //           <catch ops>
        //   after:
        let catch_body = catch_block
            .as_ref()
            .expect("try without catch is non-suspending or gated");
        let push = self.emit(AsyncOp::PushHandler {
            catch_pc: usize::MAX,
            catch_param: catch_param.clone(),
        });
        self.lower_stmts(block);
        self.emit(AsyncOp::PopHandler);
        let goto_after = self.emit(AsyncOp::Goto(usize::MAX));

        let catch_pc = self.here();
        self.lower_stmts(catch_body);
        let after = self.here();

        self.patch(goto_after, after);
        self.ops[push] = AsyncOp::PushHandler {
            catch_pc,
            catch_param: catch_param.clone(),
        };
    }
}
