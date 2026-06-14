//! ECMAScript AST node types. We keep the AST close to the source grammar
//! (ESTree-ish) so the interpreter can walk it directly without lowering.

use core::fmt;

/// A binary (value-producing, non-short-circuit) operator, as a compact `Copy`
/// tag — NO heap allocation.
///
/// M3.1 Phase 1: `Expr::Binary` previously carried an owned `String` (`"+"`,
/// `"==="`, `"instanceof"`, …), so every operator node in a parsed bundle
/// heap-allocated. This enum has one variant per operator the parser's
/// arithmetic/relational/equality/bitwise/shift precedence ladder (plus the
/// keyword ops `instanceof`/`in`) can produce; dispatch is now an integer match
/// in BOTH execution tiers. `as_str()` stays available for error text and for
/// the runtime `binary_op(&str, …)` host hook (the VM↔tree-walk BigInt bridge).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BinOp {
    Add,        // +
    Sub,        // -
    Mul,        // *
    Div,        // /
    Mod,        // %
    Pow,        // **
    EqEqEq,     // ===
    NeqEqEq,    // !==
    EqEq,       // ==
    Neq,        // !=
    Lt,         // <
    Le,         // <=
    Gt,         // >
    Ge,         // >=
    Shl,        // <<
    Shr,        // >>
    UShr,       // >>>
    BitAnd,     // &
    BitOr,      // |
    BitXor,     // ^
    Instanceof, // instanceof
    In,         // in
}

impl BinOp {
    pub fn as_str(self) -> &'static str {
        match self {
            BinOp::Add => "+",
            BinOp::Sub => "-",
            BinOp::Mul => "*",
            BinOp::Div => "/",
            BinOp::Mod => "%",
            BinOp::Pow => "**",
            BinOp::EqEqEq => "===",
            BinOp::NeqEqEq => "!==",
            BinOp::EqEq => "==",
            BinOp::Neq => "!=",
            BinOp::Lt => "<",
            BinOp::Le => "<=",
            BinOp::Gt => ">",
            BinOp::Ge => ">=",
            BinOp::Shl => "<<",
            BinOp::Shr => ">>",
            BinOp::UShr => ">>>",
            BinOp::BitAnd => "&",
            BinOp::BitOr => "|",
            BinOp::BitXor => "^",
            BinOp::Instanceof => "instanceof",
            BinOp::In => "in",
        }
    }
}

impl fmt::Display for BinOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl PartialEq<str> for BinOp {
    fn eq(&self, other: &str) -> bool {
        self.as_str() == other
    }
}
impl PartialEq<&str> for BinOp {
    fn eq(&self, other: &&str) -> bool {
        self.as_str() == *other
    }
}

/// A short-circuiting logical operator (`&&`, `||`, `??`), kept SEPARATE from
/// `BinOp` because the engine treats these specially: the right operand is only
/// evaluated conditionally. Mirrors `Expr::Logical`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LogicalOp {
    And, // &&
    Or,  // ||
    Nullish, // ??
}

impl LogicalOp {
    pub fn as_str(self) -> &'static str {
        match self {
            LogicalOp::And => "&&",
            LogicalOp::Or => "||",
            LogicalOp::Nullish => "??",
        }
    }
}

impl fmt::Display for LogicalOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl PartialEq<str> for LogicalOp {
    fn eq(&self, other: &str) -> bool {
        self.as_str() == other
    }
}
impl PartialEq<&str> for LogicalOp {
    fn eq(&self, other: &&str) -> bool {
        self.as_str() == *other
    }
}

/// A prefix unary operator. `Expr::Update` (`++`/`--`) is separate (it mutates a
/// reference); these are pure value-producing prefix operators plus `delete`
/// (which operates on a reference but the parser still models it as a unary).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum UnaryOp {
    Neg,    // -
    Plus,   // +
    Not,    // !
    BitNot, // ~
    Typeof, // typeof
    Void,   // void
    Delete, // delete
}

impl UnaryOp {
    pub fn as_str(self) -> &'static str {
        match self {
            UnaryOp::Neg => "-",
            UnaryOp::Plus => "+",
            UnaryOp::Not => "!",
            UnaryOp::BitNot => "~",
            UnaryOp::Typeof => "typeof",
            UnaryOp::Void => "void",
            UnaryOp::Delete => "delete",
        }
    }
}

impl fmt::Display for UnaryOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl PartialEq<str> for UnaryOp {
    fn eq(&self, other: &str) -> bool {
        self.as_str() == other
    }
}
impl PartialEq<&str> for UnaryOp {
    fn eq(&self, other: &&str) -> bool {
        self.as_str() == *other
    }
}

/// The increment/decrement operator on `Expr::Update`. The prefix-vs-postfix
/// distinction stays a separate `prefix: bool` field on the node.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum UpdateOp {
    Inc, // ++
    Dec, // --
}

impl UpdateOp {
    pub fn as_str(self) -> &'static str {
        match self {
            UpdateOp::Inc => "++",
            UpdateOp::Dec => "--",
        }
    }
}

impl fmt::Display for UpdateOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl PartialEq<str> for UpdateOp {
    fn eq(&self, other: &str) -> bool {
        self.as_str() == other
    }
}
impl PartialEq<&str> for UpdateOp {
    fn eq(&self, other: &&str) -> bool {
        self.as_str() == *other
    }
}

/// An assignment operator on `Expr::Assignment`. `Assign` is plain `=`; the rest
/// are compound. The compound forms map to an underlying `BinOp` (`+=` → `+`)
/// via [`AssignOp::base_binop`]; the logical-assignment forms (`&&=`/`||=`/`??=`)
/// short-circuit and have NO `BinOp` (they map to a `LogicalOp` instead).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AssignOp {
    Assign,       // =
    AddAssign,    // +=
    SubAssign,    // -=
    MulAssign,    // *=
    DivAssign,    // /=
    ModAssign,    // %=
    PowAssign,    // **=
    ShlAssign,    // <<=
    ShrAssign,    // >>=
    UShrAssign,   // >>>=
    BitAndAssign, // &=
    BitOrAssign,  // |=
    BitXorAssign, // ^=
    AndAssign,    // &&=
    OrAssign,     // ||=
    NullishAssign, // ??=
}

impl AssignOp {
    pub fn as_str(self) -> &'static str {
        match self {
            AssignOp::Assign => "=",
            AssignOp::AddAssign => "+=",
            AssignOp::SubAssign => "-=",
            AssignOp::MulAssign => "*=",
            AssignOp::DivAssign => "/=",
            AssignOp::ModAssign => "%=",
            AssignOp::PowAssign => "**=",
            AssignOp::ShlAssign => "<<=",
            AssignOp::ShrAssign => ">>=",
            AssignOp::UShrAssign => ">>>=",
            AssignOp::BitAndAssign => "&=",
            AssignOp::BitOrAssign => "|=",
            AssignOp::BitXorAssign => "^=",
            AssignOp::AndAssign => "&&=",
            AssignOp::OrAssign => "||=",
            AssignOp::NullishAssign => "??=",
        }
    }

    /// The underlying arithmetic/bitwise `BinOp` for a compound assignment
    /// (`+=` → `BinOp::Add`). Returns `None` for plain `=` and for the
    /// short-circuiting logical assignments (`&&=`/`||=`/`??=`), which combine
    /// via a `LogicalOp` and so have no plain binary semantics.
    pub fn base_binop(self) -> Option<BinOp> {
        Some(match self {
            AssignOp::AddAssign => BinOp::Add,
            AssignOp::SubAssign => BinOp::Sub,
            AssignOp::MulAssign => BinOp::Mul,
            AssignOp::DivAssign => BinOp::Div,
            AssignOp::ModAssign => BinOp::Mod,
            AssignOp::PowAssign => BinOp::Pow,
            AssignOp::ShlAssign => BinOp::Shl,
            AssignOp::ShrAssign => BinOp::Shr,
            AssignOp::UShrAssign => BinOp::UShr,
            AssignOp::BitAndAssign => BinOp::BitAnd,
            AssignOp::BitOrAssign => BinOp::BitOr,
            AssignOp::BitXorAssign => BinOp::BitXor,
            AssignOp::Assign
            | AssignOp::AndAssign
            | AssignOp::OrAssign
            | AssignOp::NullishAssign => return None,
        })
    }

    /// The `LogicalOp` for a short-circuiting logical assignment
    /// (`&&=` → `LogicalOp::And`), else `None`.
    pub fn logical_op(self) -> Option<LogicalOp> {
        Some(match self {
            AssignOp::AndAssign => LogicalOp::And,
            AssignOp::OrAssign => LogicalOp::Or,
            AssignOp::NullishAssign => LogicalOp::Nullish,
            _ => return None,
        })
    }
}

impl fmt::Display for AssignOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl PartialEq<str> for AssignOp {
    fn eq(&self, other: &str) -> bool {
        self.as_str() == other
    }
}
impl PartialEq<&str> for AssignOp {
    fn eq(&self, other: &&str) -> bool {
        self.as_str() == *other
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    Number(f64),
    BigInt(String),
    String(String),
    TemplateLiteral(String),
    Boolean(bool),
    Null,
    Undefined,
    This,
    /// `new.target` meta-property: the constructor when the enclosing function
    /// was invoked via `new`, else `undefined`.
    NewTarget,
    Identifier(String),
    Array(Vec<Expr>),
    Object(Vec<(String, Expr)>),
    Unary {
        op: UnaryOp,
        target: Box<Expr>,
    },
    Binary {
        op: BinOp,
        left: Box<Expr>,
        right: Box<Expr>,
    },
    Logical {
        op: LogicalOp,
        left: Box<Expr>,
        right: Box<Expr>,
    },
    Conditional {
        test: Box<Expr>,
        cons: Box<Expr>,
        alt: Box<Expr>,
    },
    Assignment {
        op: AssignOp,
        target: Box<Expr>,
        value: Box<Expr>,
    },
    Update {
        op: UpdateOp,
        target: Box<Expr>,
        prefix: bool,
    },
    Member {
        object: Box<Expr>,
        property: Box<Expr>,
        computed: bool,
    },
    Call {
        callee: Box<Expr>,
        args: Vec<Expr>,
    },
    New {
        callee: Box<Expr>,
        args: Vec<Expr>,
    },
    Function {
        name: Option<String>,
        params: Vec<String>,
        body: Vec<Stmt>,
    },
    Arrow {
        params: Vec<String>,
        body: ArrowBody,
    },
    Sequence(Vec<Expr>),
    /// `...expr` inside an array literal or call argument list.
    /// The interpreter flattens this into the surrounding list when
    /// it evaluates the value (expected to be array-like).
    Spread(Box<Expr>),
    /// `/pattern/flags` regex literal. The lexer recognises this via
    /// context. Evaluated to a runtime RegExp object.
    Regex(String, String),
}

#[derive(Debug, Clone, PartialEq)]
pub enum ArrowBody {
    Expr(Box<Expr>),
    Block(Vec<Stmt>),
}

#[derive(Debug, Clone, PartialEq)]
pub enum Stmt {
    Block(Vec<Stmt>),
    Expression(Expr),
    VarDecl {
        kind: VarKind,
        decls: Vec<VarDeclarator>,
    },
    FunctionDecl {
        name: String,
        params: Vec<String>,
        body: Vec<Stmt>,
    },
    If {
        test: Expr,
        cons: Box<Stmt>,
        alt: Option<Box<Stmt>>,
    },
    While {
        test: Expr,
        body: Box<Stmt>,
    },
    DoWhile {
        body: Box<Stmt>,
        test: Expr,
    },
    For {
        init: Option<ForInit>,
        test: Option<Expr>,
        update: Option<Expr>,
        body: Box<Stmt>,
    },
    /// `for (kind? name in source) body` — iterates enumerable string keys.
    ForIn {
        kind: Option<VarKind>,
        name: String,
        source: Expr,
        body: Box<Stmt>,
    },
    /// `for (kind? name of source) body` — iterates iterable values.
    /// Only array iteration is supported in V1.
    ForOf {
        is_await: bool,
        kind: Option<VarKind>,
        name: String,
        source: Expr,
        body: Box<Stmt>,
    },
    Return(Option<Expr>),
    /// `break;` or `break label;` — the optional label targets an
    /// enclosing labeled statement (loop or switch).
    Break(Option<String>),
    /// `continue;` or `continue label;`.
    Continue(Option<String>),
    /// `label: stmt` — a labeled statement. A `break label` exits it; a
    /// `continue label` continues the labeled loop. This is load-bearing:
    /// minified bundles (e.g. React's `createFiberFromTypeAndProps`) use
    /// `getTag: switch(...){ ... break getTag; }` to break out of an outer
    /// switch from an inner one. Without targeted labels the inner switch
    /// swallows the break and execution falls through to a throw.
    Labeled {
        label: String,
        body: Box<Stmt>,
    },
    Throw(Expr),
    Try {
        block: Vec<Stmt>,
        catch_param: Option<String>,
        catch_block: Option<Vec<Stmt>>,
        finally_block: Option<Vec<Stmt>>,
    },
    /// `switch (disc) { case A: ...; case B: ...; default: ... }` — the
    /// discriminant value is compared (strict equality) against each
    /// `case` expr in source order until one matches; from there we
    /// fall through to subsequent labels (including `default`) until
    /// a `break` or end of switch.  `default_index` marks which entry
    /// of `cases` is the `default:` label (None when no default).
    Switch {
        discriminant: Expr,
        cases: Vec<SwitchCase>,
        default_index: Option<usize>,
    },
    Empty,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SwitchCase {
    /// `None` indicates the `default:` label. `Some(e)` is a `case e:`.
    pub test: Option<Expr>,
    /// Statements belonging to this label, terminating implicitly at
    /// the next label or `}` (a `break` inside still bubbles out).
    pub body: Vec<Stmt>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ForInit {
    VarDecl {
        kind: VarKind,
        decls: Vec<VarDeclarator>,
    },
    Expr(Expr),
}

#[derive(Debug, Clone, PartialEq)]
pub struct VarDeclarator {
    pub name: String,
    pub init: Option<Expr>,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum VarKind {
    Var,
    Let,
    Const,
}

impl fmt::Display for VarKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Var => f.write_str("var"),
            Self::Let => f.write_str("let"),
            Self::Const => f.write_str("const"),
        }
    }
}
