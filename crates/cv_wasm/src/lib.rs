//! `cv_wasm` — WebAssembly 1.0 (MVP) module parser, validator, and
//! stack-based interpreter. From-scratch, no third-party crates.
//!
//! Scope (V1):
//!   * Binary module decoder (sections 0–11).
//!   * Type validator (function signatures + locals + globals).
//!   * Stack interpreter for the full MVP opcode set (control flow,
//!     numeric, memory, table, calls, locals/globals, parametric).
//!   * Linear memory with `memory.grow` / page = 64 KiB.
//!   * One table with `call_indirect`.
//!   * Imports + exports (function / table / memory / global).
//!   * JS-binding surface — `WebAssembly.Module` / `Instance` /
//!     `Memory` / `Table` / `Global` (wired in cv_js via host hooks).
//!
//! Not yet (V2):
//!   * SIMD, threads, bulk-memory ops, reference types, GC proposal.
//!   * Streaming compilation (we decode the whole module up front).
//!   * Optimizing JIT — we run the interpreter; cv_js JIT can take
//!     over via the same shape later.

#![allow(clippy::too_many_lines)]

use std::collections::HashMap;

/// WebAssembly value types (MVP).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValType {
    I32,
    I64,
    F32,
    F64,
}

impl ValType {
    fn from_byte(b: u8) -> Result<Self, WasmError> {
        match b {
            0x7f => Ok(Self::I32),
            0x7e => Ok(Self::I64),
            0x7d => Ok(Self::F32),
            0x7c => Ok(Self::F64),
            other => Err(WasmError::BadValType(other)),
        }
    }
}

/// Runtime values pushed on the operand stack.
#[derive(Debug, Clone, Copy)]
pub enum Value {
    I32(i32),
    I64(i64),
    F32(f32),
    F64(f64),
}

impl Value {
    fn ty(&self) -> ValType {
        match self {
            Self::I32(_) => ValType::I32,
            Self::I64(_) => ValType::I64,
            Self::F32(_) => ValType::F32,
            Self::F64(_) => ValType::F64,
        }
    }

    fn default_for(ty: ValType) -> Self {
        match ty {
            ValType::I32 => Self::I32(0),
            ValType::I64 => Self::I64(0),
            ValType::F32 => Self::F32(0.0),
            ValType::F64 => Self::F64(0.0),
        }
    }

    /// JS-side: surface as f64 (everything coerces through Number).
    pub fn to_f64(self) -> f64 {
        match self {
            Self::I32(v) => f64::from(v),
            Self::I64(v) => v as f64,
            Self::F32(v) => f64::from(v),
            Self::F64(v) => v,
        }
    }
}

/// A function signature — params then results. MVP allows ≤1 result.
#[derive(Debug, Clone)]
pub struct FuncType {
    pub params: Vec<ValType>,
    pub results: Vec<ValType>,
}

#[derive(Debug, Clone)]
pub struct Limits {
    pub min: u32,
    pub max: Option<u32>,
}

#[derive(Debug, Clone)]
pub struct MemoryType {
    pub limits: Limits,
}

#[derive(Debug, Clone)]
pub struct TableType {
    pub limits: Limits,
    // MVP: element type is always funcref (0x70).
}

#[derive(Debug, Clone)]
pub struct GlobalType {
    pub ty: ValType,
    pub mutable: bool,
}

#[derive(Debug, Clone)]
pub enum ImportDesc {
    Func(u32), // type idx
    Table(TableType),
    Memory(MemoryType),
    Global(GlobalType),
}

#[derive(Debug, Clone)]
pub struct Import {
    pub module: String,
    pub name: String,
    pub desc: ImportDesc,
}

#[derive(Debug, Clone)]
pub enum ExportDesc {
    Func(u32),
    Table(u32),
    Memory(u32),
    Global(u32),
}

#[derive(Debug, Clone)]
pub struct Export {
    pub name: String,
    pub desc: ExportDesc,
}

#[derive(Debug, Clone)]
pub struct FuncBody {
    pub locals: Vec<ValType>,
    pub code: Vec<u8>,
    pub code_start: usize,
}

#[derive(Debug, Clone)]
pub struct DataSegment {
    pub mem_idx: u32,
    pub offset_expr: Vec<u8>,
    pub init: Vec<u8>,
}

#[derive(Debug, Clone)]
pub struct ElementSegment {
    pub table_idx: u32,
    pub offset_expr: Vec<u8>,
    pub func_indices: Vec<u32>,
}

/// Decoded module — everything needed to instantiate.
#[derive(Debug, Default)]
pub struct Module {
    pub types: Vec<FuncType>,
    pub imports: Vec<Import>,
    pub funcs: Vec<u32>, // type idx per function defined in this module
    pub tables: Vec<TableType>,
    pub memories: Vec<MemoryType>,
    pub globals: Vec<(GlobalType, Vec<u8>)>, // type + init expr
    pub exports: Vec<Export>,
    pub start: Option<u32>,
    pub elements: Vec<ElementSegment>,
    pub code: Vec<FuncBody>,
    pub data: Vec<DataSegment>,
}

// ============================================================
// Errors
// ============================================================

#[derive(Debug)]
pub enum WasmError {
    UnexpectedEof,
    BadMagic([u8; 4]),
    BadVersion(u32),
    BadValType(u8),
    BadSection(u8),
    BadOpcode(u8),
    BadAlignment,
    BadImportDesc(u8),
    BadExportDesc(u8),
    InvalidUtf8,
    StackUnderflow,
    TypeMismatch,
    UnknownImport(String, String),
    MemoryOutOfBounds,
    TableOutOfBounds,
    IntegerOverflow,
    IntegerDivByZero,
    InvalidConversion,
    Unreachable,
    CallIndirectTypeMismatch,
    GlobalImmutable,
    StackOverflow,
    Trap(String),
}

impl std::fmt::Display for WasmError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

// ============================================================
// Binary reader
// ============================================================

struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }

    fn remaining(&self) -> usize {
        self.buf.len().saturating_sub(self.pos)
    }

    fn eof(&self) -> bool {
        self.pos >= self.buf.len()
    }

    fn u8(&mut self) -> Result<u8, WasmError> {
        if self.pos >= self.buf.len() {
            return Err(WasmError::UnexpectedEof);
        }
        let b = self.buf[self.pos];
        self.pos += 1;
        Ok(b)
    }

    fn bytes(&mut self, n: usize) -> Result<&'a [u8], WasmError> {
        if self.pos + n > self.buf.len() {
            return Err(WasmError::UnexpectedEof);
        }
        let slice = &self.buf[self.pos..self.pos + n];
        self.pos += n;
        Ok(slice)
    }

    /// LEB128-encoded u32 (≤5 bytes per spec).
    fn u32_leb(&mut self) -> Result<u32, WasmError> {
        let mut result: u32 = 0;
        let mut shift: u32 = 0;
        for _ in 0..5 {
            let b = self.u8()?;
            result |= u32::from(b & 0x7f) << shift;
            if b & 0x80 == 0 {
                return Ok(result);
            }
            shift += 7;
        }
        Err(WasmError::IntegerOverflow)
    }

    /// LEB128-encoded i32 (signed, ≤5 bytes).
    fn i32_leb(&mut self) -> Result<i32, WasmError> {
        let mut result: i64 = 0;
        let mut shift: u32 = 0;
        loop {
            let b = self.u8()?;
            result |= i64::from(b & 0x7f) << shift;
            shift += 7;
            if b & 0x80 == 0 {
                if shift < 64 && (b & 0x40) != 0 {
                    result |= -(1_i64 << shift);
                }
                return i32::try_from(result).map_err(|_| WasmError::IntegerOverflow);
            }
            if shift >= 35 {
                return Err(WasmError::IntegerOverflow);
            }
        }
    }

    fn i64_leb(&mut self) -> Result<i64, WasmError> {
        let mut result: i64 = 0;
        let mut shift: u32 = 0;
        loop {
            let b = self.u8()?;
            result |= i64::from(b & 0x7f) << shift;
            shift += 7;
            if b & 0x80 == 0 {
                if shift < 64 && (b & 0x40) != 0 {
                    result |= -(1_i64 << shift);
                }
                return Ok(result);
            }
            if shift >= 70 {
                return Err(WasmError::IntegerOverflow);
            }
        }
    }

    fn f32(&mut self) -> Result<f32, WasmError> {
        let bytes = self.bytes(4)?;
        Ok(f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    fn f64(&mut self) -> Result<f64, WasmError> {
        let bytes = self.bytes(8)?;
        let mut a = [0u8; 8];
        a.copy_from_slice(bytes);
        Ok(f64::from_le_bytes(a))
    }

    fn name(&mut self) -> Result<String, WasmError> {
        let n = self.u32_leb()? as usize;
        let bytes = self.bytes(n)?;
        std::str::from_utf8(bytes)
            .map(str::to_owned)
            .map_err(|_| WasmError::InvalidUtf8)
    }

    fn limits(&mut self) -> Result<Limits, WasmError> {
        let flag = self.u8()?;
        let min = self.u32_leb()?;
        let max = if flag & 1 == 1 {
            Some(self.u32_leb()?)
        } else {
            None
        };
        Ok(Limits { min, max })
    }
}

// ============================================================
// Module decoding
// ============================================================

const MAGIC: [u8; 4] = [0x00, 0x61, 0x73, 0x6d]; // \0asm
const VERSION: u32 = 1;

/// Decode a WASM binary into a `Module`.
pub fn decode(bytes: &[u8]) -> Result<Module, WasmError> {
    let mut r = Reader::new(bytes);
    let magic = r.bytes(4)?;
    if magic != MAGIC {
        let mut m = [0u8; 4];
        m.copy_from_slice(magic);
        return Err(WasmError::BadMagic(m));
    }
    let version = u32::from_le_bytes([r.u8()?, r.u8()?, r.u8()?, r.u8()?]);
    if version != VERSION {
        return Err(WasmError::BadVersion(version));
    }

    let mut m = Module::default();

    while !r.eof() {
        let section_id = r.u8()?;
        let section_size = r.u32_leb()? as usize;
        let end = r.pos + section_size;
        if end > r.buf.len() {
            return Err(WasmError::UnexpectedEof);
        }
        match section_id {
            0 => {
                /* custom — skip */
                r.pos = end;
            }
            1 => decode_type_section(&mut r, &mut m)?,
            2 => decode_import_section(&mut r, &mut m)?,
            3 => decode_function_section(&mut r, &mut m)?,
            4 => decode_table_section(&mut r, &mut m)?,
            5 => decode_memory_section(&mut r, &mut m)?,
            6 => decode_global_section(&mut r, &mut m)?,
            7 => decode_export_section(&mut r, &mut m)?,
            8 => m.start = Some(r.u32_leb()?),
            9 => decode_element_section(&mut r, &mut m)?,
            10 => decode_code_section(&mut r, &mut m)?,
            11 => decode_data_section(&mut r, &mut m)?,
            other => return Err(WasmError::BadSection(other)),
        }
        r.pos = end;
    }

    Ok(m)
}

fn decode_type_section(r: &mut Reader, m: &mut Module) -> Result<(), WasmError> {
    let n = r.u32_leb()?;
    for _ in 0..n {
        let form = r.u8()?;
        if form != 0x60 {
            return Err(WasmError::BadValType(form));
        }
        let pn = r.u32_leb()?;
        let mut params = Vec::with_capacity(pn as usize);
        for _ in 0..pn {
            params.push(ValType::from_byte(r.u8()?)?);
        }
        let rn = r.u32_leb()?;
        let mut results = Vec::with_capacity(rn as usize);
        for _ in 0..rn {
            results.push(ValType::from_byte(r.u8()?)?);
        }
        m.types.push(FuncType { params, results });
    }
    Ok(())
}

fn decode_import_section(r: &mut Reader, m: &mut Module) -> Result<(), WasmError> {
    let n = r.u32_leb()?;
    for _ in 0..n {
        let module = r.name()?;
        let name = r.name()?;
        let kind = r.u8()?;
        let desc = match kind {
            0x00 => ImportDesc::Func(r.u32_leb()?),
            0x01 => {
                let et = r.u8()?;
                if et != 0x70 {
                    return Err(WasmError::BadImportDesc(et));
                }
                ImportDesc::Table(TableType {
                    limits: r.limits()?,
                })
            }
            0x02 => ImportDesc::Memory(MemoryType {
                limits: r.limits()?,
            }),
            0x03 => {
                let ty = ValType::from_byte(r.u8()?)?;
                let mutable = r.u8()? != 0;
                ImportDesc::Global(GlobalType { ty, mutable })
            }
            other => return Err(WasmError::BadImportDesc(other)),
        };
        m.imports.push(Import { module, name, desc });
    }
    Ok(())
}

fn decode_function_section(r: &mut Reader, m: &mut Module) -> Result<(), WasmError> {
    let n = r.u32_leb()?;
    for _ in 0..n {
        m.funcs.push(r.u32_leb()?);
    }
    Ok(())
}

fn decode_table_section(r: &mut Reader, m: &mut Module) -> Result<(), WasmError> {
    let n = r.u32_leb()?;
    for _ in 0..n {
        let et = r.u8()?;
        if et != 0x70 {
            return Err(WasmError::BadImportDesc(et));
        }
        m.tables.push(TableType {
            limits: r.limits()?,
        });
    }
    Ok(())
}

fn decode_memory_section(r: &mut Reader, m: &mut Module) -> Result<(), WasmError> {
    let n = r.u32_leb()?;
    for _ in 0..n {
        m.memories.push(MemoryType {
            limits: r.limits()?,
        });
    }
    Ok(())
}

fn decode_global_section(r: &mut Reader, m: &mut Module) -> Result<(), WasmError> {
    let n = r.u32_leb()?;
    for _ in 0..n {
        let ty = ValType::from_byte(r.u8()?)?;
        let mutable = r.u8()? != 0;
        let init = read_const_expr(r)?;
        m.globals.push((GlobalType { ty, mutable }, init));
    }
    Ok(())
}

fn decode_export_section(r: &mut Reader, m: &mut Module) -> Result<(), WasmError> {
    let n = r.u32_leb()?;
    for _ in 0..n {
        let name = r.name()?;
        let kind = r.u8()?;
        let idx = r.u32_leb()?;
        let desc = match kind {
            0 => ExportDesc::Func(idx),
            1 => ExportDesc::Table(idx),
            2 => ExportDesc::Memory(idx),
            3 => ExportDesc::Global(idx),
            other => return Err(WasmError::BadExportDesc(other)),
        };
        m.exports.push(Export { name, desc });
    }
    Ok(())
}

fn decode_element_section(r: &mut Reader, m: &mut Module) -> Result<(), WasmError> {
    let n = r.u32_leb()?;
    for _ in 0..n {
        let table_idx = r.u32_leb()?;
        let offset_expr = read_const_expr(r)?;
        let cn = r.u32_leb()?;
        let mut indices = Vec::with_capacity(cn as usize);
        for _ in 0..cn {
            indices.push(r.u32_leb()?);
        }
        m.elements.push(ElementSegment {
            table_idx,
            offset_expr,
            func_indices: indices,
        });
    }
    Ok(())
}

fn decode_code_section(r: &mut Reader, m: &mut Module) -> Result<(), WasmError> {
    let n = r.u32_leb()?;
    for _ in 0..n {
        let body_size = r.u32_leb()? as usize;
        let body_end = r.pos + body_size;
        let ln = r.u32_leb()?;
        let mut locals: Vec<ValType> = Vec::new();
        for _ in 0..ln {
            let count = r.u32_leb()?;
            let ty = ValType::from_byte(r.u8()?)?;
            for _ in 0..count {
                locals.push(ty);
            }
        }
        let code_start = r.pos;
        let code_len = body_end - r.pos;
        let code = r.bytes(code_len)?.to_vec();
        m.code.push(FuncBody {
            locals,
            code,
            code_start,
        });
    }
    Ok(())
}

fn decode_data_section(r: &mut Reader, m: &mut Module) -> Result<(), WasmError> {
    let n = r.u32_leb()?;
    for _ in 0..n {
        let mem_idx = r.u32_leb()?;
        let offset_expr = read_const_expr(r)?;
        let dn = r.u32_leb()? as usize;
        let init = r.bytes(dn)?.to_vec();
        m.data.push(DataSegment {
            mem_idx,
            offset_expr,
            init,
        });
    }
    Ok(())
}

/// Read a constant expression up to (and including) the `end` opcode.
fn read_const_expr(r: &mut Reader) -> Result<Vec<u8>, WasmError> {
    let start = r.pos;
    loop {
        let b = r.u8()?;
        if b == 0x0b {
            break;
        }
        // Const expressions in MVP: i32.const / i64.const / f32.const /
        // f64.const / global.get. Each has one immediate.
        match b {
            0x41 => {
                let _ = r.i32_leb()?;
            }
            0x42 => {
                let _ = r.i64_leb()?;
            }
            0x43 => {
                let _ = r.f32()?;
            }
            0x44 => {
                let _ = r.f64()?;
            }
            0x23 => {
                let _ = r.u32_leb()?;
            }
            _ => return Err(WasmError::BadOpcode(b)),
        }
    }
    Ok(r.buf[start..r.pos].to_vec())
}

/// Evaluate a const-expr against the current globals snapshot.
fn eval_const_expr(expr: &[u8], globals: &[Value]) -> Result<Value, WasmError> {
    let mut r = Reader::new(expr);
    let b = r.u8()?;
    let v = match b {
        0x41 => Value::I32(r.i32_leb()?),
        0x42 => Value::I64(r.i64_leb()?),
        0x43 => Value::F32(r.f32()?),
        0x44 => Value::F64(r.f64()?),
        0x23 => {
            let idx = r.u32_leb()? as usize;
            globals.get(idx).copied().ok_or(WasmError::TypeMismatch)?
        }
        _ => return Err(WasmError::BadOpcode(b)),
    };
    let end = r.u8()?;
    if end != 0x0b {
        return Err(WasmError::BadOpcode(end));
    }
    Ok(v)
}

// ============================================================
// Runtime — Instance, Memory, Table, Globals, hosted imports
// ============================================================

pub const PAGE_SIZE: usize = 65_536;

#[derive(Debug)]
pub struct Memory {
    pub data: Vec<u8>,
    pub max_pages: Option<u32>,
}

impl Memory {
    pub fn new(min_pages: u32, max_pages: Option<u32>) -> Self {
        Self {
            data: vec![0u8; (min_pages as usize) * PAGE_SIZE],
            max_pages,
        }
    }

    pub fn pages(&self) -> u32 {
        (self.data.len() / PAGE_SIZE) as u32
    }

    pub fn grow(&mut self, delta_pages: u32) -> i32 {
        let cur = self.pages();
        let new = cur.saturating_add(delta_pages);
        if let Some(mx) = self.max_pages {
            if new > mx {
                return -1;
            }
        }
        if (new as usize).saturating_mul(PAGE_SIZE) > (u32::MAX as usize) {
            return -1;
        }
        self.data.resize((new as usize) * PAGE_SIZE, 0);
        cur as i32
    }

    fn check(&self, ea: u32, len: usize) -> Result<(), WasmError> {
        if (ea as usize)
            .checked_add(len)
            .map_or(true, |x| x > self.data.len())
        {
            return Err(WasmError::MemoryOutOfBounds);
        }
        Ok(())
    }
}

/// A host function — implemented by the embedder (cv_js).
pub type HostFn = std::rc::Rc<dyn Fn(&[Value]) -> Result<Option<Value>, WasmError>>;

#[derive(Clone)]
pub enum FuncInst {
    Local(u32), // index into module.code
    Host(HostFn),
}

impl std::fmt::Debug for FuncInst {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Local(idx) => write!(f, "FuncInst::Local({})", idx),
            Self::Host(_) => write!(f, "FuncInst::Host"),
        }
    }
}

#[derive(Debug)]
pub struct Table {
    pub entries: Vec<Option<u32>>, // function indices in `funcs`
    pub max: Option<u32>,
}

#[derive(Debug, Clone)]
pub struct Global {
    /// Interior mutability so `global.set` can mutate a global at runtime
    /// while the interpreter holds only `&Instance`. `Value` is `Copy`, so a
    /// plain `Cell` (no `RefCell`) suffices and stays alloc-free.
    pub value: std::cell::Cell<Value>,
    pub mutable: bool,
}

/// Imports the embedder supplies — keyed by (module, name).
#[derive(Default, Clone)]
pub struct Imports {
    pub funcs: HashMap<(String, String), HostFn>,
    pub memories: HashMap<(String, String), std::rc::Rc<std::cell::RefCell<Memory>>>,
    pub tables: HashMap<(String, String), std::rc::Rc<std::cell::RefCell<Table>>>,
    pub globals: HashMap<(String, String), Global>,
}

impl std::fmt::Debug for Imports {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Imports")
            .field("funcs", &self.funcs.keys().collect::<Vec<_>>())
            .field("memories", &self.memories.keys().collect::<Vec<_>>())
            .field("tables", &self.tables.keys().collect::<Vec<_>>())
            .field("globals", &self.globals.keys().collect::<Vec<_>>())
            .finish()
    }
}

#[derive(Debug)]
pub struct Instance {
    pub module: Module,
    pub funcs: Vec<FuncInst>, // imports first, then locals
    pub func_types: Vec<u32>, // parallel to funcs
    pub memory: Option<std::rc::Rc<std::cell::RefCell<Memory>>>,
    pub table: Option<std::rc::Rc<std::cell::RefCell<Table>>>,
    pub globals: Vec<Global>,
    pub exports: HashMap<String, ExportDesc>,
}

impl Instance {
    pub fn instantiate(module: Module, imports: &Imports) -> Result<Self, WasmError> {
        let mut funcs: Vec<FuncInst> = Vec::new();
        let mut func_types: Vec<u32> = Vec::new();
        let mut memory: Option<std::rc::Rc<std::cell::RefCell<Memory>>> = None;
        let mut table: Option<std::rc::Rc<std::cell::RefCell<Table>>> = None;
        let mut globals: Vec<Global> = Vec::new();

        // Resolve imports first.
        for imp in &module.imports {
            let key = (imp.module.clone(), imp.name.clone());
            match &imp.desc {
                ImportDesc::Func(type_idx) => {
                    let host =
                        imports.funcs.get(&key).cloned().ok_or_else(|| {
                            WasmError::UnknownImport(key.0.clone(), key.1.clone())
                        })?;
                    funcs.push(FuncInst::Host(host));
                    func_types.push(*type_idx);
                }
                ImportDesc::Memory(_) => {
                    memory =
                        Some(imports.memories.get(&key).cloned().ok_or_else(|| {
                            WasmError::UnknownImport(key.0.clone(), key.1.clone())
                        })?);
                }
                ImportDesc::Table(_) => {
                    table =
                        Some(imports.tables.get(&key).cloned().ok_or_else(|| {
                            WasmError::UnknownImport(key.0.clone(), key.1.clone())
                        })?);
                }
                ImportDesc::Global(_) => {
                    let g = imports
                        .globals
                        .get(&key)
                        .ok_or_else(|| WasmError::UnknownImport(key.0.clone(), key.1.clone()))?;
                    globals.push(Global {
                        value: std::cell::Cell::new(g.value.get()),
                        mutable: g.mutable,
                    });
                }
            }
        }

        // Module-defined globals (evaluated against globals seen so far).
        for (gt, init_expr) in &module.globals {
            let snapshot: Vec<Value> = globals.iter().map(|g| g.value.get()).collect();
            let value = eval_const_expr(init_expr, &snapshot)?;
            if value.ty() != gt.ty {
                return Err(WasmError::TypeMismatch);
            }
            globals.push(Global {
                value: std::cell::Cell::new(value),
                mutable: gt.mutable,
            });
        }

        // Module-defined memory (only one in MVP).
        if memory.is_none() {
            if let Some(mt) = module.memories.first() {
                memory = Some(std::rc::Rc::new(std::cell::RefCell::new(Memory::new(
                    mt.limits.min,
                    mt.limits.max,
                ))));
            }
        }

        // Module-defined table.
        if table.is_none() {
            if let Some(tt) = module.tables.first() {
                table = Some(std::rc::Rc::new(std::cell::RefCell::new(Table {
                    entries: vec![None; tt.limits.min as usize],
                    max: tt.limits.max,
                })));
            }
        }

        // Module-defined functions.
        let import_func_count = funcs.len();
        for (i, type_idx) in module.funcs.iter().enumerate() {
            funcs.push(FuncInst::Local((import_func_count + i) as u32));
            func_types.push(*type_idx);
        }

        // Init data segments.
        if let Some(mem) = &memory {
            for seg in &module.data {
                let snapshot: Vec<Value> = globals.iter().map(|g| g.value.get()).collect();
                let off = match eval_const_expr(&seg.offset_expr, &snapshot)? {
                    Value::I32(v) => v as u32,
                    _ => return Err(WasmError::TypeMismatch),
                };
                let mut mb = mem.borrow_mut();
                mb.check(off, seg.init.len())?;
                mb.data[off as usize..off as usize + seg.init.len()].copy_from_slice(&seg.init);
            }
        }

        // Init element segments (function table fill).
        if let Some(tbl) = &table {
            for seg in &module.elements {
                let snapshot: Vec<Value> = globals.iter().map(|g| g.value.get()).collect();
                let off = match eval_const_expr(&seg.offset_expr, &snapshot)? {
                    Value::I32(v) => v as usize,
                    _ => return Err(WasmError::TypeMismatch),
                };
                let mut t = tbl.borrow_mut();
                if off + seg.func_indices.len() > t.entries.len() {
                    return Err(WasmError::TableOutOfBounds);
                }
                for (i, fi) in seg.func_indices.iter().enumerate() {
                    t.entries[off + i] = Some(*fi);
                }
            }
        }

        // Build exports map.
        let mut exports = HashMap::new();
        for e in &module.exports {
            exports.insert(e.name.clone(), e.desc.clone());
        }

        let inst = Self {
            module,
            funcs,
            func_types,
            memory,
            table,
            globals,
            exports,
        };

        // Run start function if present.
        if let Some(start) = inst.module.start {
            let mut interp = Interp::new(&inst);
            interp.call(start as usize, &[])?;
        }

        Ok(inst)
    }

    /// Look up an exported function by name and invoke it.
    pub fn invoke(&self, export_name: &str, args: &[Value]) -> Result<Option<Value>, WasmError> {
        let desc = self
            .exports
            .get(export_name)
            .ok_or_else(|| WasmError::Trap(format!("no export {export_name}")))?;
        let idx = match desc {
            ExportDesc::Func(i) => *i as usize,
            _ => return Err(WasmError::Trap(format!("{export_name} is not a function"))),
        };
        let mut interp = Interp::new(self);
        interp.call(idx, args)
    }
}

// ============================================================
// Interpreter
// ============================================================

const STACK_LIMIT: usize = 8 * 1024;
const CALL_DEPTH_LIMIT: usize = 1024;

struct Frame<'a> {
    code: &'a [u8],
    pc: usize,
    locals: Vec<Value>,
    /// Label stack: each entry is (continuation pc, arity, kind).
    /// kind: 0=block, 1=loop, 2=if.
    labels: Vec<(usize, usize, u8)>,
    /// Number of return values expected.
    arity: usize,
    /// Stack height when this frame was entered — used to truncate
    /// on return so we leave only the result(s).
    base: usize,
}

pub struct Interp<'a> {
    inst: &'a Instance,
    stack: Vec<Value>,
}

impl<'a> Interp<'a> {
    fn new(inst: &'a Instance) -> Self {
        Self {
            inst,
            stack: Vec::with_capacity(256),
        }
    }

    fn push(&mut self, v: Value) -> Result<(), WasmError> {
        if self.stack.len() >= STACK_LIMIT {
            return Err(WasmError::StackOverflow);
        }
        self.stack.push(v);
        Ok(())
    }

    fn pop(&mut self) -> Result<Value, WasmError> {
        self.stack.pop().ok_or(WasmError::StackUnderflow)
    }

    fn pop_i32(&mut self) -> Result<i32, WasmError> {
        match self.pop()? {
            Value::I32(v) => Ok(v),
            _ => Err(WasmError::TypeMismatch),
        }
    }

    fn pop_i64(&mut self) -> Result<i64, WasmError> {
        match self.pop()? {
            Value::I64(v) => Ok(v),
            _ => Err(WasmError::TypeMismatch),
        }
    }

    fn pop_f32(&mut self) -> Result<f32, WasmError> {
        match self.pop()? {
            Value::F32(v) => Ok(v),
            _ => Err(WasmError::TypeMismatch),
        }
    }

    fn pop_f64(&mut self) -> Result<f64, WasmError> {
        match self.pop()? {
            Value::F64(v) => Ok(v),
            _ => Err(WasmError::TypeMismatch),
        }
    }

    pub fn call(&mut self, func_idx: usize, args: &[Value]) -> Result<Option<Value>, WasmError> {
        self.call_with_depth(func_idx, args, 0)
    }

    fn call_with_depth(
        &mut self,
        func_idx: usize,
        args: &[Value],
        depth: usize,
    ) -> Result<Option<Value>, WasmError> {
        if depth >= CALL_DEPTH_LIMIT {
            return Err(WasmError::StackOverflow);
        }
        let f = self
            .inst
            .funcs
            .get(func_idx)
            .ok_or(WasmError::TypeMismatch)?;
        let type_idx = self.inst.func_types[func_idx] as usize;
        let ft = &self.inst.module.types[type_idx];
        if args.len() != ft.params.len() {
            return Err(WasmError::TypeMismatch);
        }
        match f {
            FuncInst::Host(h) => h(args),
            FuncInst::Local(local_idx) => {
                // local_idx counts across imports; index into module.code
                // by subtracting import-func count.
                let import_count = self
                    .inst
                    .module
                    .imports
                    .iter()
                    .filter(|i| matches!(i.desc, ImportDesc::Func(_)))
                    .count();
                let body_idx = (*local_idx as usize)
                    .checked_sub(import_count)
                    .ok_or(WasmError::TypeMismatch)?;
                let body = &self.inst.module.code[body_idx];

                let mut locals: Vec<Value> =
                    Vec::with_capacity(ft.params.len() + body.locals.len());
                for a in args {
                    locals.push(*a);
                }
                for lt in &body.locals {
                    locals.push(Value::default_for(*lt));
                }

                let base = self.stack.len();
                let arity = ft.results.len();
                let mut frame = Frame {
                    code: &body.code,
                    pc: 0,
                    locals,
                    labels: Vec::new(),
                    arity,
                    base,
                };
                self.execute(&mut frame, depth)?;
                // Result truncation: we expect `arity` results on top.
                let result = if arity == 1 { Some(self.pop()?) } else { None };
                self.stack.truncate(base);
                Ok(result)
            }
        }
    }

    fn execute(&mut self, frame: &mut Frame, depth: usize) -> Result<(), WasmError> {
        // We work on a borrowed slice — clone the code into a local Vec to
        // dodge the borrow conflict with `self.inst.module.code`.
        let code: Vec<u8> = frame.code.to_vec();
        let inst = self.inst;
        let mut pc = frame.pc;

        loop {
            if pc >= code.len() {
                return Ok(()); // function fell off end
            }
            let op = code[pc];
            pc += 1;
            match op {
                0x00 => return Err(WasmError::Unreachable),
                0x01 => {} // nop
                0x02 => {
                    // block bt
                    let _bt = read_block_type(&code, &mut pc)?;
                    // Find matching `end` to set continuation.
                    let end = find_end(&code, pc)?;
                    frame.labels.push((end + 1, 0, 0));
                }
                0x03 => {
                    // loop bt
                    let _bt = read_block_type(&code, &mut pc)?;
                    let loop_start = pc;
                    frame.labels.push((loop_start, 0, 1));
                }
                0x04 => {
                    // if bt
                    let _bt = read_block_type(&code, &mut pc)?;
                    let cond = self.pop_i32()?;
                    let else_pos = find_else_or_end(&code, pc)?;
                    let end_pos = find_end(&code, pc)?;
                    frame.labels.push((end_pos + 1, 0, 2));
                    if cond == 0 {
                        // jump past else (or to end)
                        if code[else_pos] == 0x05 {
                            pc = else_pos + 1;
                        } else {
                            pc = end_pos;
                        }
                    }
                }
                0x05 => {
                    // else — when reached in then-branch, skip to end.
                    let end_pos = find_end(&code, pc)?;
                    pc = end_pos;
                }
                0x0b => {
                    // end — pop label
                    if let Some(_) = frame.labels.pop() {
                        // nothing else to do
                    } else {
                        // end of function body
                        return Ok(());
                    }
                }
                0x0c => {
                    // br l
                    let l = read_u32(&code, &mut pc)? as usize;
                    do_branch(frame, l, &mut pc)?;
                }
                0x0d => {
                    // br_if l
                    let l = read_u32(&code, &mut pc)? as usize;
                    let c = self.pop_i32()?;
                    if c != 0 {
                        do_branch(frame, l, &mut pc)?;
                    }
                }
                0x0e => {
                    // br_table
                    let n = read_u32(&code, &mut pc)? as usize;
                    let mut targets = Vec::with_capacity(n);
                    for _ in 0..n {
                        targets.push(read_u32(&code, &mut pc)? as usize);
                    }
                    let default = read_u32(&code, &mut pc)? as usize;
                    let i = self.pop_i32()? as usize;
                    let l = if i < n { targets[i] } else { default };
                    do_branch(frame, l, &mut pc)?;
                }
                0x0f => return Ok(()), // return
                0x10 => {
                    // call
                    let f = read_u32(&code, &mut pc)? as usize;
                    let type_idx = inst.func_types[f] as usize;
                    let ft = &inst.module.types[type_idx];
                    let nparams = ft.params.len();
                    if self.stack.len() < nparams {
                        return Err(WasmError::StackUnderflow);
                    }
                    let args: Vec<Value> = self.stack.split_off(self.stack.len() - nparams);
                    if let Some(v) = self.call_with_depth(f, &args, depth + 1)? {
                        self.push(v)?;
                    }
                }
                0x11 => {
                    // call_indirect type_idx, 0x00
                    let type_idx = read_u32(&code, &mut pc)? as usize;
                    let _reserved = read_u32(&code, &mut pc)?;
                    let i = self.pop_i32()? as usize;
                    let tbl = inst
                        .table
                        .as_ref()
                        .ok_or(WasmError::Trap("no table".into()))?;
                    let func_idx =
                        tbl.borrow()
                            .entries
                            .get(i)
                            .copied()
                            .flatten()
                            .ok_or(WasmError::TableOutOfBounds)? as usize;
                    if inst.func_types[func_idx] as usize != type_idx {
                        return Err(WasmError::CallIndirectTypeMismatch);
                    }
                    let ft = &inst.module.types[type_idx];
                    let nparams = ft.params.len();
                    if self.stack.len() < nparams {
                        return Err(WasmError::StackUnderflow);
                    }
                    let args: Vec<Value> = self.stack.split_off(self.stack.len() - nparams);
                    if let Some(v) = self.call_with_depth(func_idx, &args, depth + 1)? {
                        self.push(v)?;
                    }
                }
                0x1a => {
                    let _ = self.pop()?;
                }
                0x1b => {
                    // select
                    let c = self.pop_i32()?;
                    let v2 = self.pop()?;
                    let v1 = self.pop()?;
                    self.push(if c != 0 { v1 } else { v2 })?;
                }
                0x20 => {
                    // local.get
                    let i = read_u32(&code, &mut pc)? as usize;
                    let v = *frame.locals.get(i).ok_or(WasmError::TypeMismatch)?;
                    self.push(v)?;
                }
                0x21 => {
                    // local.set
                    let i = read_u32(&code, &mut pc)? as usize;
                    let v = self.pop()?;
                    *frame.locals.get_mut(i).ok_or(WasmError::TypeMismatch)? = v;
                }
                0x22 => {
                    // local.tee
                    let i = read_u32(&code, &mut pc)? as usize;
                    let v = *self.stack.last().ok_or(WasmError::StackUnderflow)?;
                    *frame.locals.get_mut(i).ok_or(WasmError::TypeMismatch)? = v;
                }
                0x23 => {
                    // global.get
                    let i = read_u32(&code, &mut pc)? as usize;
                    let v = self
                        .inst
                        .globals
                        .get(i)
                        .ok_or(WasmError::TypeMismatch)?
                        .value
                        .get();
                    self.push(v)?;
                }
                0x24 => {
                    // global.set — real interior-mutability write. Setting an
                    // immutable global is a wasm VALIDATION error; with no
                    // separate validator pass we surface it as a runtime trap
                    // (correct result, later phase than the spec).
                    let i = read_u32(&code, &mut pc)? as usize;
                    let g = self.inst.globals.get(i).ok_or(WasmError::TypeMismatch)?;
                    if !g.mutable {
                        return Err(WasmError::GlobalImmutable);
                    }
                    let v = self.pop()?;
                    g.value.set(v);
                }
                // ===== Memory loads =====
                0x28 => {
                    let (_a, o) = read_memarg(&code, &mut pc)?;
                    let base = self.pop_i32()? as u32;
                    let mem = inst
                        .memory
                        .as_ref()
                        .ok_or(WasmError::Trap("no mem".into()))?;
                    let m = mem.borrow();
                    let ea = base.wrapping_add(o);
                    m.check(ea, 4)?;
                    let v = i32::from_le_bytes([
                        m.data[ea as usize],
                        m.data[ea as usize + 1],
                        m.data[ea as usize + 2],
                        m.data[ea as usize + 3],
                    ]);
                    self.push(Value::I32(v))?;
                }
                0x29 => {
                    let (_a, o) = read_memarg(&code, &mut pc)?;
                    let base = self.pop_i32()? as u32;
                    let mem = inst
                        .memory
                        .as_ref()
                        .ok_or(WasmError::Trap("no mem".into()))?;
                    let m = mem.borrow();
                    let ea = base.wrapping_add(o);
                    m.check(ea, 8)?;
                    let mut buf = [0u8; 8];
                    buf.copy_from_slice(&m.data[ea as usize..ea as usize + 8]);
                    self.push(Value::I64(i64::from_le_bytes(buf)))?;
                }
                0x2a => {
                    let (_a, o) = read_memarg(&code, &mut pc)?;
                    let base = self.pop_i32()? as u32;
                    let mem = inst
                        .memory
                        .as_ref()
                        .ok_or(WasmError::Trap("no mem".into()))?;
                    let m = mem.borrow();
                    let ea = base.wrapping_add(o);
                    m.check(ea, 4)?;
                    let v = f32::from_le_bytes([
                        m.data[ea as usize],
                        m.data[ea as usize + 1],
                        m.data[ea as usize + 2],
                        m.data[ea as usize + 3],
                    ]);
                    self.push(Value::F32(v))?;
                }
                0x2b => {
                    let (_a, o) = read_memarg(&code, &mut pc)?;
                    let base = self.pop_i32()? as u32;
                    let mem = inst
                        .memory
                        .as_ref()
                        .ok_or(WasmError::Trap("no mem".into()))?;
                    let m = mem.borrow();
                    let ea = base.wrapping_add(o);
                    m.check(ea, 8)?;
                    let mut buf = [0u8; 8];
                    buf.copy_from_slice(&m.data[ea as usize..ea as usize + 8]);
                    self.push(Value::F64(f64::from_le_bytes(buf)))?;
                }
                0x2c => {
                    // i32.load8_s
                    let (_a, o) = read_memarg(&code, &mut pc)?;
                    let base = self.pop_i32()? as u32;
                    let mem = inst
                        .memory
                        .as_ref()
                        .ok_or(WasmError::Trap("no mem".into()))?;
                    let m = mem.borrow();
                    let ea = base.wrapping_add(o);
                    m.check(ea, 1)?;
                    let v = m.data[ea as usize] as i8 as i32;
                    self.push(Value::I32(v))?;
                }
                0x2d => {
                    // i32.load8_u
                    let (_a, o) = read_memarg(&code, &mut pc)?;
                    let base = self.pop_i32()? as u32;
                    let mem = inst
                        .memory
                        .as_ref()
                        .ok_or(WasmError::Trap("no mem".into()))?;
                    let m = mem.borrow();
                    let ea = base.wrapping_add(o);
                    m.check(ea, 1)?;
                    self.push(Value::I32(i32::from(m.data[ea as usize])))?;
                }
                0x2e => {
                    let (_a, o) = read_memarg(&code, &mut pc)?;
                    let base = self.pop_i32()? as u32;
                    let mem = inst
                        .memory
                        .as_ref()
                        .ok_or(WasmError::Trap("no mem".into()))?;
                    let m = mem.borrow();
                    let ea = base.wrapping_add(o);
                    m.check(ea, 2)?;
                    let v = i16::from_le_bytes([m.data[ea as usize], m.data[ea as usize + 1]]);
                    self.push(Value::I32(v as i32))?;
                }
                0x2f => {
                    let (_a, o) = read_memarg(&code, &mut pc)?;
                    let base = self.pop_i32()? as u32;
                    let mem = inst
                        .memory
                        .as_ref()
                        .ok_or(WasmError::Trap("no mem".into()))?;
                    let m = mem.borrow();
                    let ea = base.wrapping_add(o);
                    m.check(ea, 2)?;
                    let v = u16::from_le_bytes([m.data[ea as usize], m.data[ea as usize + 1]]);
                    self.push(Value::I32(i32::from(v)))?;
                }
                0x30 => {
                    // i64.load8_s — sign-extend a byte to i64
                    let (_a, o) = read_memarg(&code, &mut pc)?;
                    let base = self.pop_i32()? as u32;
                    let mem = inst
                        .memory
                        .as_ref()
                        .ok_or(WasmError::Trap("no mem".into()))?;
                    let m = mem.borrow();
                    let ea = base.wrapping_add(o);
                    m.check(ea, 1)?;
                    let v = m.data[ea as usize] as i8 as i64;
                    self.push(Value::I64(v))?;
                }
                0x31 => {
                    // i64.load8_u — zero-extend a byte to i64
                    let (_a, o) = read_memarg(&code, &mut pc)?;
                    let base = self.pop_i32()? as u32;
                    let mem = inst
                        .memory
                        .as_ref()
                        .ok_or(WasmError::Trap("no mem".into()))?;
                    let m = mem.borrow();
                    let ea = base.wrapping_add(o);
                    m.check(ea, 1)?;
                    let v = m.data[ea as usize] as u64 as i64;
                    self.push(Value::I64(v))?;
                }
                0x32 => {
                    // i64.load16_s
                    let (_a, o) = read_memarg(&code, &mut pc)?;
                    let base = self.pop_i32()? as u32;
                    let mem = inst
                        .memory
                        .as_ref()
                        .ok_or(WasmError::Trap("no mem".into()))?;
                    let m = mem.borrow();
                    let ea = base.wrapping_add(o);
                    m.check(ea, 2)?;
                    let v = i16::from_le_bytes([m.data[ea as usize], m.data[ea as usize + 1]]) as i64;
                    self.push(Value::I64(v))?;
                }
                0x33 => {
                    // i64.load16_u
                    let (_a, o) = read_memarg(&code, &mut pc)?;
                    let base = self.pop_i32()? as u32;
                    let mem = inst
                        .memory
                        .as_ref()
                        .ok_or(WasmError::Trap("no mem".into()))?;
                    let m = mem.borrow();
                    let ea = base.wrapping_add(o);
                    m.check(ea, 2)?;
                    let v = u16::from_le_bytes([m.data[ea as usize], m.data[ea as usize + 1]]) as u64
                        as i64;
                    self.push(Value::I64(v))?;
                }
                0x34 => {
                    // i64.load32_s
                    let (_a, o) = read_memarg(&code, &mut pc)?;
                    let base = self.pop_i32()? as u32;
                    let mem = inst
                        .memory
                        .as_ref()
                        .ok_or(WasmError::Trap("no mem".into()))?;
                    let m = mem.borrow();
                    let ea = base.wrapping_add(o);
                    m.check(ea, 4)?;
                    let v = i32::from_le_bytes([
                        m.data[ea as usize],
                        m.data[ea as usize + 1],
                        m.data[ea as usize + 2],
                        m.data[ea as usize + 3],
                    ]) as i64;
                    self.push(Value::I64(v))?;
                }
                0x35 => {
                    // i64.load32_u
                    let (_a, o) = read_memarg(&code, &mut pc)?;
                    let base = self.pop_i32()? as u32;
                    let mem = inst
                        .memory
                        .as_ref()
                        .ok_or(WasmError::Trap("no mem".into()))?;
                    let m = mem.borrow();
                    let ea = base.wrapping_add(o);
                    m.check(ea, 4)?;
                    let v = u32::from_le_bytes([
                        m.data[ea as usize],
                        m.data[ea as usize + 1],
                        m.data[ea as usize + 2],
                        m.data[ea as usize + 3],
                    ]) as u64 as i64;
                    self.push(Value::I64(v))?;
                }
                // ===== Memory stores =====
                0x36 => {
                    let (_a, o) = read_memarg(&code, &mut pc)?;
                    let v = self.pop_i32()?;
                    let base = self.pop_i32()? as u32;
                    let mem = inst
                        .memory
                        .as_ref()
                        .ok_or(WasmError::Trap("no mem".into()))?;
                    let mut m = mem.borrow_mut();
                    let ea = base.wrapping_add(o);
                    m.check(ea, 4)?;
                    let bytes = v.to_le_bytes();
                    m.data[ea as usize..ea as usize + 4].copy_from_slice(&bytes);
                }
                0x37 => {
                    let (_a, o) = read_memarg(&code, &mut pc)?;
                    let v = self.pop_i64()?;
                    let base = self.pop_i32()? as u32;
                    let mem = inst
                        .memory
                        .as_ref()
                        .ok_or(WasmError::Trap("no mem".into()))?;
                    let mut m = mem.borrow_mut();
                    let ea = base.wrapping_add(o);
                    m.check(ea, 8)?;
                    let bytes = v.to_le_bytes();
                    m.data[ea as usize..ea as usize + 8].copy_from_slice(&bytes);
                }
                0x38 => {
                    let (_a, o) = read_memarg(&code, &mut pc)?;
                    let v = self.pop_f32()?;
                    let base = self.pop_i32()? as u32;
                    let mem = inst
                        .memory
                        .as_ref()
                        .ok_or(WasmError::Trap("no mem".into()))?;
                    let mut m = mem.borrow_mut();
                    let ea = base.wrapping_add(o);
                    m.check(ea, 4)?;
                    let bytes = v.to_le_bytes();
                    m.data[ea as usize..ea as usize + 4].copy_from_slice(&bytes);
                }
                0x39 => {
                    let (_a, o) = read_memarg(&code, &mut pc)?;
                    let v = self.pop_f64()?;
                    let base = self.pop_i32()? as u32;
                    let mem = inst
                        .memory
                        .as_ref()
                        .ok_or(WasmError::Trap("no mem".into()))?;
                    let mut m = mem.borrow_mut();
                    let ea = base.wrapping_add(o);
                    m.check(ea, 8)?;
                    let bytes = v.to_le_bytes();
                    m.data[ea as usize..ea as usize + 8].copy_from_slice(&bytes);
                }
                0x3a => {
                    // i32.store8
                    let (_a, o) = read_memarg(&code, &mut pc)?;
                    let v = self.pop_i32()? as u8;
                    let base = self.pop_i32()? as u32;
                    let mem = inst
                        .memory
                        .as_ref()
                        .ok_or(WasmError::Trap("no mem".into()))?;
                    let mut m = mem.borrow_mut();
                    let ea = base.wrapping_add(o);
                    m.check(ea, 1)?;
                    m.data[ea as usize] = v;
                }
                0x3b => {
                    let (_a, o) = read_memarg(&code, &mut pc)?;
                    let v = self.pop_i32()? as u16;
                    let base = self.pop_i32()? as u32;
                    let mem = inst
                        .memory
                        .as_ref()
                        .ok_or(WasmError::Trap("no mem".into()))?;
                    let mut m = mem.borrow_mut();
                    let ea = base.wrapping_add(o);
                    m.check(ea, 2)?;
                    m.data[ea as usize..ea as usize + 2].copy_from_slice(&v.to_le_bytes());
                }
                0x3c => {
                    // i64.store8 — write the low byte of an i64
                    let (_a, o) = read_memarg(&code, &mut pc)?;
                    let v = self.pop_i64()? as u8;
                    let base = self.pop_i32()? as u32;
                    let mem = inst
                        .memory
                        .as_ref()
                        .ok_or(WasmError::Trap("no mem".into()))?;
                    let mut m = mem.borrow_mut();
                    let ea = base.wrapping_add(o);
                    m.check(ea, 1)?;
                    m.data[ea as usize] = v;
                }
                0x3d => {
                    // i64.store16 — write the low 2 bytes of an i64
                    let (_a, o) = read_memarg(&code, &mut pc)?;
                    let v = self.pop_i64()? as u16;
                    let base = self.pop_i32()? as u32;
                    let mem = inst
                        .memory
                        .as_ref()
                        .ok_or(WasmError::Trap("no mem".into()))?;
                    let mut m = mem.borrow_mut();
                    let ea = base.wrapping_add(o);
                    m.check(ea, 2)?;
                    m.data[ea as usize..ea as usize + 2].copy_from_slice(&v.to_le_bytes());
                }
                0x3e => {
                    // i64.store32 — write the low 4 bytes of an i64
                    let (_a, o) = read_memarg(&code, &mut pc)?;
                    let v = self.pop_i64()? as u32;
                    let base = self.pop_i32()? as u32;
                    let mem = inst
                        .memory
                        .as_ref()
                        .ok_or(WasmError::Trap("no mem".into()))?;
                    let mut m = mem.borrow_mut();
                    let ea = base.wrapping_add(o);
                    m.check(ea, 4)?;
                    m.data[ea as usize..ea as usize + 4].copy_from_slice(&v.to_le_bytes());
                }
                0x3f => {
                    // memory.size
                    let _reserved = read_u32(&code, &mut pc)?;
                    let pages = inst
                        .memory
                        .as_ref()
                        .map(|m| m.borrow().pages())
                        .unwrap_or(0);
                    self.push(Value::I32(pages as i32))?;
                }
                0x40 => {
                    // memory.grow
                    let _reserved = read_u32(&code, &mut pc)?;
                    let n = self.pop_i32()?;
                    let result = inst
                        .memory
                        .as_ref()
                        .map(|m| m.borrow_mut().grow(n as u32))
                        .unwrap_or(-1);
                    self.push(Value::I32(result))?;
                }
                // ===== Const =====
                0x41 => {
                    let v = read_i32(&code, &mut pc)?;
                    self.push(Value::I32(v))?;
                }
                0x42 => {
                    let v = read_i64(&code, &mut pc)?;
                    self.push(Value::I64(v))?;
                }
                0x43 => {
                    let v = read_f32(&code, &mut pc)?;
                    self.push(Value::F32(v))?;
                }
                0x44 => {
                    let v = read_f64(&code, &mut pc)?;
                    self.push(Value::F64(v))?;
                }
                // ===== i32 comparisons =====
                0x45 => {
                    let a = self.pop_i32()?;
                    self.push(Value::I32(i32::from(a == 0)))?;
                }
                0x46 => {
                    let b = self.pop_i32()?;
                    let a = self.pop_i32()?;
                    self.push(Value::I32(i32::from(a == b)))?;
                }
                0x47 => {
                    let b = self.pop_i32()?;
                    let a = self.pop_i32()?;
                    self.push(Value::I32(i32::from(a != b)))?;
                }
                0x48 => {
                    let b = self.pop_i32()?;
                    let a = self.pop_i32()?;
                    self.push(Value::I32(i32::from(a < b)))?;
                }
                0x49 => {
                    let b = self.pop_i32()?;
                    let a = self.pop_i32()?;
                    self.push(Value::I32(i32::from((a as u32) < (b as u32))))?;
                }
                0x4a => {
                    let b = self.pop_i32()?;
                    let a = self.pop_i32()?;
                    self.push(Value::I32(i32::from(a > b)))?;
                }
                0x4b => {
                    let b = self.pop_i32()?;
                    let a = self.pop_i32()?;
                    self.push(Value::I32(i32::from((a as u32) > (b as u32))))?;
                }
                0x4c => {
                    let b = self.pop_i32()?;
                    let a = self.pop_i32()?;
                    self.push(Value::I32(i32::from(a <= b)))?;
                }
                0x4d => {
                    let b = self.pop_i32()?;
                    let a = self.pop_i32()?;
                    self.push(Value::I32(i32::from((a as u32) <= (b as u32))))?;
                }
                0x4e => {
                    let b = self.pop_i32()?;
                    let a = self.pop_i32()?;
                    self.push(Value::I32(i32::from(a >= b)))?;
                }
                0x4f => {
                    let b = self.pop_i32()?;
                    let a = self.pop_i32()?;
                    self.push(Value::I32(i32::from((a as u32) >= (b as u32))))?;
                }
                // ===== i64 comparisons (push i32 0/1) =====
                0x50 => {
                    // i64.eqz
                    let a = self.pop_i64()?;
                    self.push(Value::I32(i32::from(a == 0)))?;
                }
                0x51 => {
                    let b = self.pop_i64()?;
                    let a = self.pop_i64()?;
                    self.push(Value::I32(i32::from(a == b)))?;
                }
                0x52 => {
                    let b = self.pop_i64()?;
                    let a = self.pop_i64()?;
                    self.push(Value::I32(i32::from(a != b)))?;
                }
                0x53 => {
                    let b = self.pop_i64()?;
                    let a = self.pop_i64()?;
                    self.push(Value::I32(i32::from(a < b)))?;
                }
                0x54 => {
                    let b = self.pop_i64()?;
                    let a = self.pop_i64()?;
                    self.push(Value::I32(i32::from((a as u64) < (b as u64))))?;
                }
                0x55 => {
                    let b = self.pop_i64()?;
                    let a = self.pop_i64()?;
                    self.push(Value::I32(i32::from(a > b)))?;
                }
                0x56 => {
                    let b = self.pop_i64()?;
                    let a = self.pop_i64()?;
                    self.push(Value::I32(i32::from((a as u64) > (b as u64))))?;
                }
                0x57 => {
                    let b = self.pop_i64()?;
                    let a = self.pop_i64()?;
                    self.push(Value::I32(i32::from(a <= b)))?;
                }
                0x58 => {
                    let b = self.pop_i64()?;
                    let a = self.pop_i64()?;
                    self.push(Value::I32(i32::from((a as u64) <= (b as u64))))?;
                }
                0x59 => {
                    let b = self.pop_i64()?;
                    let a = self.pop_i64()?;
                    self.push(Value::I32(i32::from(a >= b)))?;
                }
                0x5a => {
                    let b = self.pop_i64()?;
                    let a = self.pop_i64()?;
                    self.push(Value::I32(i32::from((a as u64) >= (b as u64))))?;
                }
                // ===== f32 comparisons (IEEE-754; NaN compares unordered) =====
                0x5b => {
                    let b = self.pop_f32()?;
                    let a = self.pop_f32()?;
                    self.push(Value::I32(i32::from(a == b)))?;
                }
                0x5c => {
                    let b = self.pop_f32()?;
                    let a = self.pop_f32()?;
                    self.push(Value::I32(i32::from(a != b)))?;
                }
                0x5d => {
                    let b = self.pop_f32()?;
                    let a = self.pop_f32()?;
                    self.push(Value::I32(i32::from(a < b)))?;
                }
                0x5e => {
                    let b = self.pop_f32()?;
                    let a = self.pop_f32()?;
                    self.push(Value::I32(i32::from(a > b)))?;
                }
                0x5f => {
                    let b = self.pop_f32()?;
                    let a = self.pop_f32()?;
                    self.push(Value::I32(i32::from(a <= b)))?;
                }
                0x60 => {
                    let b = self.pop_f32()?;
                    let a = self.pop_f32()?;
                    self.push(Value::I32(i32::from(a >= b)))?;
                }
                // ===== f64 comparisons (IEEE-754; NaN compares unordered) =====
                0x61 => {
                    let b = self.pop_f64()?;
                    let a = self.pop_f64()?;
                    self.push(Value::I32(i32::from(a == b)))?;
                }
                0x62 => {
                    let b = self.pop_f64()?;
                    let a = self.pop_f64()?;
                    self.push(Value::I32(i32::from(a != b)))?;
                }
                0x63 => {
                    let b = self.pop_f64()?;
                    let a = self.pop_f64()?;
                    self.push(Value::I32(i32::from(a < b)))?;
                }
                0x64 => {
                    let b = self.pop_f64()?;
                    let a = self.pop_f64()?;
                    self.push(Value::I32(i32::from(a > b)))?;
                }
                0x65 => {
                    let b = self.pop_f64()?;
                    let a = self.pop_f64()?;
                    self.push(Value::I32(i32::from(a <= b)))?;
                }
                0x66 => {
                    let b = self.pop_f64()?;
                    let a = self.pop_f64()?;
                    self.push(Value::I32(i32::from(a >= b)))?;
                }
                // ===== i32 arithmetic =====
                0x67 => {
                    let a = self.pop_i32()?;
                    self.push(Value::I32(a.leading_zeros() as i32))?;
                }
                0x68 => {
                    let a = self.pop_i32()?;
                    self.push(Value::I32(a.trailing_zeros() as i32))?;
                }
                0x69 => {
                    let a = self.pop_i32()?;
                    self.push(Value::I32(a.count_ones() as i32))?;
                }
                0x6a => {
                    let b = self.pop_i32()?;
                    let a = self.pop_i32()?;
                    self.push(Value::I32(a.wrapping_add(b)))?;
                }
                0x6b => {
                    let b = self.pop_i32()?;
                    let a = self.pop_i32()?;
                    self.push(Value::I32(a.wrapping_sub(b)))?;
                }
                0x6c => {
                    let b = self.pop_i32()?;
                    let a = self.pop_i32()?;
                    self.push(Value::I32(a.wrapping_mul(b)))?;
                }
                0x6d => {
                    let b = self.pop_i32()?;
                    let a = self.pop_i32()?;
                    if b == 0 {
                        return Err(WasmError::IntegerDivByZero);
                    }
                    if a == i32::MIN && b == -1 {
                        return Err(WasmError::IntegerOverflow);
                    }
                    self.push(Value::I32(a / b))?;
                }
                0x6e => {
                    let b = self.pop_i32()? as u32;
                    let a = self.pop_i32()? as u32;
                    if b == 0 {
                        return Err(WasmError::IntegerDivByZero);
                    }
                    self.push(Value::I32((a / b) as i32))?;
                }
                0x6f => {
                    let b = self.pop_i32()?;
                    let a = self.pop_i32()?;
                    if b == 0 {
                        return Err(WasmError::IntegerDivByZero);
                    }
                    self.push(Value::I32(a.wrapping_rem(b)))?;
                }
                0x70 => {
                    let b = self.pop_i32()? as u32;
                    let a = self.pop_i32()? as u32;
                    if b == 0 {
                        return Err(WasmError::IntegerDivByZero);
                    }
                    self.push(Value::I32((a % b) as i32))?;
                }
                0x71 => {
                    let b = self.pop_i32()?;
                    let a = self.pop_i32()?;
                    self.push(Value::I32(a & b))?;
                }
                0x72 => {
                    let b = self.pop_i32()?;
                    let a = self.pop_i32()?;
                    self.push(Value::I32(a | b))?;
                }
                0x73 => {
                    let b = self.pop_i32()?;
                    let a = self.pop_i32()?;
                    self.push(Value::I32(a ^ b))?;
                }
                0x74 => {
                    let b = self.pop_i32()?;
                    let a = self.pop_i32()?;
                    self.push(Value::I32(a.wrapping_shl(b as u32 & 31)))?;
                }
                0x75 => {
                    let b = self.pop_i32()?;
                    let a = self.pop_i32()?;
                    self.push(Value::I32(a.wrapping_shr(b as u32 & 31)))?;
                }
                0x76 => {
                    let b = self.pop_i32()? as u32;
                    let a = self.pop_i32()? as u32;
                    self.push(Value::I32((a.wrapping_shr(b & 31)) as i32))?;
                }
                0x77 => {
                    let b = self.pop_i32()?;
                    let a = self.pop_i32()?;
                    self.push(Value::I32(a.rotate_left(b as u32 & 31)))?;
                }
                0x78 => {
                    let b = self.pop_i32()?;
                    let a = self.pop_i32()?;
                    self.push(Value::I32(a.rotate_right(b as u32 & 31)))?;
                }
                // ===== i64 arithmetic / bitwise =====
                0x79 => {
                    // i64.clz
                    let a = self.pop_i64()?;
                    self.push(Value::I64(a.leading_zeros() as i64))?;
                }
                0x7a => {
                    // i64.ctz
                    let a = self.pop_i64()?;
                    self.push(Value::I64(a.trailing_zeros() as i64))?;
                }
                0x7b => {
                    // i64.popcnt
                    let a = self.pop_i64()?;
                    self.push(Value::I64(a.count_ones() as i64))?;
                }
                0x7c => {
                    let b = self.pop_i64()?;
                    let a = self.pop_i64()?;
                    self.push(Value::I64(a.wrapping_add(b)))?;
                }
                0x7d => {
                    let b = self.pop_i64()?;
                    let a = self.pop_i64()?;
                    self.push(Value::I64(a.wrapping_sub(b)))?;
                }
                0x7e => {
                    let b = self.pop_i64()?;
                    let a = self.pop_i64()?;
                    self.push(Value::I64(a.wrapping_mul(b)))?;
                }
                0x7f => {
                    // i64.div_s
                    let b = self.pop_i64()?;
                    let a = self.pop_i64()?;
                    if b == 0 {
                        return Err(WasmError::IntegerDivByZero);
                    }
                    if a == i64::MIN && b == -1 {
                        return Err(WasmError::IntegerOverflow);
                    }
                    self.push(Value::I64(a / b))?;
                }
                0x80 => {
                    // i64.div_u
                    let b = self.pop_i64()? as u64;
                    let a = self.pop_i64()? as u64;
                    if b == 0 {
                        return Err(WasmError::IntegerDivByZero);
                    }
                    self.push(Value::I64((a / b) as i64))?;
                }
                0x81 => {
                    // i64.rem_s — wrapping_rem avoids the MIN%-1 overflow trap
                    // (rem of i64::MIN % -1 is defined to be 0, no trap).
                    let b = self.pop_i64()?;
                    let a = self.pop_i64()?;
                    if b == 0 {
                        return Err(WasmError::IntegerDivByZero);
                    }
                    self.push(Value::I64(a.wrapping_rem(b)))?;
                }
                0x82 => {
                    // i64.rem_u
                    let b = self.pop_i64()? as u64;
                    let a = self.pop_i64()? as u64;
                    if b == 0 {
                        return Err(WasmError::IntegerDivByZero);
                    }
                    self.push(Value::I64((a % b) as i64))?;
                }
                0x83 => {
                    let b = self.pop_i64()?;
                    let a = self.pop_i64()?;
                    self.push(Value::I64(a & b))?;
                }
                0x84 => {
                    let b = self.pop_i64()?;
                    let a = self.pop_i64()?;
                    self.push(Value::I64(a | b))?;
                }
                0x85 => {
                    let b = self.pop_i64()?;
                    let a = self.pop_i64()?;
                    self.push(Value::I64(a ^ b))?;
                }
                0x86 => {
                    // i64.shl — shift count masked mod 64
                    let b = self.pop_i64()?;
                    let a = self.pop_i64()?;
                    self.push(Value::I64(a.wrapping_shl((b as u64 & 63) as u32)))?;
                }
                0x87 => {
                    // i64.shr_s — arithmetic right shift
                    let b = self.pop_i64()?;
                    let a = self.pop_i64()?;
                    self.push(Value::I64(a.wrapping_shr((b as u64 & 63) as u32)))?;
                }
                0x88 => {
                    // i64.shr_u — logical right shift
                    let b = self.pop_i64()?;
                    let a = self.pop_i64()? as u64;
                    self.push(Value::I64(a.wrapping_shr((b as u64 & 63) as u32) as i64))?;
                }
                0x89 => {
                    // i64.rotl
                    let b = self.pop_i64()?;
                    let a = self.pop_i64()?;
                    self.push(Value::I64(a.rotate_left((b as u64 & 63) as u32)))?;
                }
                0x8a => {
                    // i64.rotr
                    let b = self.pop_i64()?;
                    let a = self.pop_i64()?;
                    self.push(Value::I64(a.rotate_right((b as u64 & 63) as u32)))?;
                }
                // ===== f32 unary =====
                0x8b => {
                    let a = self.pop_f32()?;
                    self.push(Value::F32(a.abs()))?;
                }
                0x8c => {
                    let a = self.pop_f32()?;
                    self.push(Value::F32(-a))?;
                }
                0x8d => {
                    let a = self.pop_f32()?;
                    self.push(Value::F32(a.ceil()))?;
                }
                0x8e => {
                    let a = self.pop_f32()?;
                    self.push(Value::F32(a.floor()))?;
                }
                0x8f => {
                    let a = self.pop_f32()?;
                    self.push(Value::F32(a.trunc()))?;
                }
                0x90 => {
                    // f32.nearest — round-ties-to-even (banker's rounding)
                    let a = self.pop_f32()?;
                    self.push(Value::F32(a.round_ties_even()))?;
                }
                0x91 => {
                    let a = self.pop_f32()?;
                    self.push(Value::F32(a.sqrt()))?;
                }
                // ===== f32/f64 arithmetic (subset) =====
                0x92 => {
                    let b = self.pop_f32()?;
                    let a = self.pop_f32()?;
                    self.push(Value::F32(a + b))?;
                }
                0x93 => {
                    let b = self.pop_f32()?;
                    let a = self.pop_f32()?;
                    self.push(Value::F32(a - b))?;
                }
                0x94 => {
                    let b = self.pop_f32()?;
                    let a = self.pop_f32()?;
                    self.push(Value::F32(a * b))?;
                }
                0x95 => {
                    let b = self.pop_f32()?;
                    let a = self.pop_f32()?;
                    self.push(Value::F32(a / b))?;
                }
                // ===== f32 binary (min/max propagate NaN; copysign) =====
                0x96 => {
                    let b = self.pop_f32()?;
                    let a = self.pop_f32()?;
                    self.push(Value::F32(wasm_fmin_f32(a, b)))?;
                }
                0x97 => {
                    let b = self.pop_f32()?;
                    let a = self.pop_f32()?;
                    self.push(Value::F32(wasm_fmax_f32(a, b)))?;
                }
                0x98 => {
                    let b = self.pop_f32()?;
                    let a = self.pop_f32()?;
                    self.push(Value::F32(a.copysign(b)))?;
                }
                // ===== f64 unary =====
                0x99 => {
                    let a = self.pop_f64()?;
                    self.push(Value::F64(a.abs()))?;
                }
                0x9a => {
                    let a = self.pop_f64()?;
                    self.push(Value::F64(-a))?;
                }
                0x9b => {
                    let a = self.pop_f64()?;
                    self.push(Value::F64(a.ceil()))?;
                }
                0x9c => {
                    let a = self.pop_f64()?;
                    self.push(Value::F64(a.floor()))?;
                }
                0x9d => {
                    let a = self.pop_f64()?;
                    self.push(Value::F64(a.trunc()))?;
                }
                0x9e => {
                    // f64.nearest — round-ties-to-even
                    let a = self.pop_f64()?;
                    self.push(Value::F64(a.round_ties_even()))?;
                }
                0x9f => {
                    let a = self.pop_f64()?;
                    self.push(Value::F64(a.sqrt()))?;
                }
                0xa0 => {
                    let b = self.pop_f64()?;
                    let a = self.pop_f64()?;
                    self.push(Value::F64(a + b))?;
                }
                0xa1 => {
                    let b = self.pop_f64()?;
                    let a = self.pop_f64()?;
                    self.push(Value::F64(a - b))?;
                }
                0xa2 => {
                    let b = self.pop_f64()?;
                    let a = self.pop_f64()?;
                    self.push(Value::F64(a * b))?;
                }
                0xa3 => {
                    let b = self.pop_f64()?;
                    let a = self.pop_f64()?;
                    self.push(Value::F64(a / b))?;
                }
                // ===== f64 binary (min/max propagate NaN; copysign) =====
                0xa4 => {
                    let b = self.pop_f64()?;
                    let a = self.pop_f64()?;
                    self.push(Value::F64(wasm_fmin_f64(a, b)))?;
                }
                0xa5 => {
                    let b = self.pop_f64()?;
                    let a = self.pop_f64()?;
                    self.push(Value::F64(wasm_fmax_f64(a, b)))?;
                }
                0xa6 => {
                    let b = self.pop_f64()?;
                    let a = self.pop_f64()?;
                    self.push(Value::F64(a.copysign(b)))?;
                }
                // ===== Conversions (most common) =====
                0xa7 => {
                    // i32.wrap_i64
                    let v = self.pop_i64()?;
                    self.push(Value::I32(v as i32))?;
                }
                0xac => {
                    // i64.extend_i32_s
                    let v = self.pop_i32()?;
                    self.push(Value::I64(i64::from(v)))?;
                }
                0xad => {
                    // i64.extend_i32_u
                    let v = self.pop_i32()? as u32;
                    self.push(Value::I64(i64::from(v)))?;
                }
                _ => return Err(WasmError::BadOpcode(op)),
            }

            // Persist pc back so future iterations resume at the right spot.
            frame.pc = pc;
        }
    }
}

// --- Helpers for the interpreter loop ---

fn read_u32(code: &[u8], pc: &mut usize) -> Result<u32, WasmError> {
    let mut r = Reader::new(&code[*pc..]);
    let v = r.u32_leb()?;
    *pc += r.pos;
    Ok(v)
}

fn read_i32(code: &[u8], pc: &mut usize) -> Result<i32, WasmError> {
    let mut r = Reader::new(&code[*pc..]);
    let v = r.i32_leb()?;
    *pc += r.pos;
    Ok(v)
}

fn read_i64(code: &[u8], pc: &mut usize) -> Result<i64, WasmError> {
    let mut r = Reader::new(&code[*pc..]);
    let v = r.i64_leb()?;
    *pc += r.pos;
    Ok(v)
}

fn read_f32(code: &[u8], pc: &mut usize) -> Result<f32, WasmError> {
    if *pc + 4 > code.len() {
        return Err(WasmError::UnexpectedEof);
    }
    let v = f32::from_le_bytes([code[*pc], code[*pc + 1], code[*pc + 2], code[*pc + 3]]);
    *pc += 4;
    Ok(v)
}

fn read_f64(code: &[u8], pc: &mut usize) -> Result<f64, WasmError> {
    if *pc + 8 > code.len() {
        return Err(WasmError::UnexpectedEof);
    }
    let mut buf = [0u8; 8];
    buf.copy_from_slice(&code[*pc..*pc + 8]);
    *pc += 8;
    Ok(f64::from_le_bytes(buf))
}

fn read_memarg(code: &[u8], pc: &mut usize) -> Result<(u32, u32), WasmError> {
    let a = read_u32(code, pc)?;
    let o = read_u32(code, pc)?;
    Ok((a, o))
}

fn read_block_type(code: &[u8], pc: &mut usize) -> Result<i32, WasmError> {
    // Block types in MVP: 0x40 (empty) or a single valtype byte.
    if *pc >= code.len() {
        return Err(WasmError::UnexpectedEof);
    }
    let b = code[*pc];
    *pc += 1;
    Ok(b as i32)
}

/// Find the matching `end` opcode for a structured-control block,
/// walking past any nested blocks. `start` points to the byte AFTER
/// the opening opcode and its blocktype.
fn find_end(code: &[u8], start: usize) -> Result<usize, WasmError> {
    let mut depth = 1;
    let mut pc = start;
    while pc < code.len() {
        let op = code[pc];
        pc += 1;
        match op {
            0x02 | 0x03 | 0x04 => {
                // Skip block type.
                if pc < code.len() {
                    pc += 1;
                }
                depth += 1;
            }
            0x0b => {
                depth -= 1;
                if depth == 0 {
                    return Ok(pc - 1);
                }
            }
            // Skip immediates of common opcodes so we don't mis-scan.
            0x0c | 0x0d | 0x10 | 0x20 | 0x21 | 0x22 | 0x23 | 0x24 | 0x3f | 0x40 => {
                let _ = read_u32(code, &mut pc).ok();
            }
            0x11 => {
                let _ = read_u32(code, &mut pc).ok();
                let _ = read_u32(code, &mut pc).ok();
            }
            0x28..=0x3e => {
                let _ = read_u32(code, &mut pc).ok();
                let _ = read_u32(code, &mut pc).ok();
            }
            0x41 => {
                let _ = read_i32(code, &mut pc).ok();
            }
            0x42 => {
                let _ = read_i64(code, &mut pc).ok();
            }
            0x43 => {
                pc += 4;
            }
            0x44 => {
                pc += 8;
            }
            0x0e => {
                let n = read_u32(code, &mut pc).unwrap_or(0) as usize;
                for _ in 0..n {
                    let _ = read_u32(code, &mut pc).ok();
                }
                let _ = read_u32(code, &mut pc).ok();
            }
            _ => {}
        }
    }
    Err(WasmError::UnexpectedEof)
}

fn find_else_or_end(code: &[u8], start: usize) -> Result<usize, WasmError> {
    let mut depth = 1;
    let mut pc = start;
    while pc < code.len() {
        let op = code[pc];
        let here = pc;
        pc += 1;
        match op {
            0x02 | 0x03 | 0x04 => {
                if pc < code.len() {
                    pc += 1;
                }
                depth += 1;
            }
            0x05 if depth == 1 => return Ok(here),
            0x0b => {
                depth -= 1;
                if depth == 0 {
                    return Ok(here);
                }
            }
            0x0c | 0x0d | 0x10 | 0x20 | 0x21 | 0x22 | 0x23 | 0x24 | 0x3f | 0x40 => {
                let _ = read_u32(code, &mut pc).ok();
            }
            0x11 => {
                let _ = read_u32(code, &mut pc).ok();
                let _ = read_u32(code, &mut pc).ok();
            }
            0x28..=0x3e => {
                let _ = read_u32(code, &mut pc).ok();
                let _ = read_u32(code, &mut pc).ok();
            }
            0x41 => {
                let _ = read_i32(code, &mut pc).ok();
            }
            0x42 => {
                let _ = read_i64(code, &mut pc).ok();
            }
            0x43 => {
                pc += 4;
            }
            0x44 => {
                pc += 8;
            }
            0x0e => {
                let n = read_u32(code, &mut pc).unwrap_or(0) as usize;
                for _ in 0..n {
                    let _ = read_u32(code, &mut pc).ok();
                }
                let _ = read_u32(code, &mut pc).ok();
            }
            _ => {}
        }
    }
    Err(WasmError::UnexpectedEof)
}

// --- IEEE-754 fmin/fmax with wasm semantics ---
//
// Rust's `f64::min`/`f64::max` DIVERGE from wasm: they IGNORE NaN (returning
// the non-NaN operand) and mishandle signed zeros. wasm `fmin`/`fmax` must
// PROPAGATE NaN (any NaN operand → a NaN result) and distinguish ±0.0
// (`min(-0,+0) = -0`, `max(-0,+0) = +0`). Hand-rolled so the divergence is
// impossible.

fn wasm_fmin_f64(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        return f64::NAN;
    }
    if a == 0.0 && b == 0.0 {
        // Both zero: the negative zero wins.
        return if a.is_sign_negative() || b.is_sign_negative() {
            -0.0
        } else {
            0.0
        };
    }
    if a < b { a } else { b }
}

fn wasm_fmax_f64(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        return f64::NAN;
    }
    if a == 0.0 && b == 0.0 {
        // Both zero: the positive zero wins.
        return if a.is_sign_positive() || b.is_sign_positive() {
            0.0
        } else {
            -0.0
        };
    }
    if a > b { a } else { b }
}

fn wasm_fmin_f32(a: f32, b: f32) -> f32 {
    if a.is_nan() || b.is_nan() {
        return f32::NAN;
    }
    if a == 0.0 && b == 0.0 {
        return if a.is_sign_negative() || b.is_sign_negative() {
            -0.0
        } else {
            0.0
        };
    }
    if a < b { a } else { b }
}

fn wasm_fmax_f32(a: f32, b: f32) -> f32 {
    if a.is_nan() || b.is_nan() {
        return f32::NAN;
    }
    if a == 0.0 && b == 0.0 {
        return if a.is_sign_positive() || b.is_sign_positive() {
            0.0
        } else {
            -0.0
        };
    }
    if a > b { a } else { b }
}

fn do_branch(frame: &mut Frame, label: usize, pc: &mut usize) -> Result<(), WasmError> {
    if label >= frame.labels.len() {
        return Err(WasmError::TypeMismatch);
    }
    let target_idx = frame.labels.len() - 1 - label;
    let (cont, _arity, _kind) = frame.labels[target_idx];
    frame.labels.truncate(target_idx);
    *pc = cont;
    Ok(())
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Smallest valid module: just the magic + version.
    #[test]
    fn empty_module() {
        let bytes = [0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00];
        let m = decode(&bytes).unwrap();
        assert_eq!(m.types.len(), 0);
        assert_eq!(m.funcs.len(), 0);
    }

    /// A module exporting a function `add(i32, i32) -> i32 { local.get 0; local.get 1; i32.add }`.
    #[test]
    fn add_i32() {
        let bytes: &[u8] = &[
            0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00, // header
            // type section: 1 type, (i32,i32)->i32
            0x01, 0x07, 0x01, 0x60, 0x02, 0x7f, 0x7f, 0x01, 0x7f,
            // function section: 1 function of type 0
            0x03, 0x02, 0x01, 0x00, // export section: "add" -> func 0
            0x07, 0x07, 0x01, 0x03, b'a', b'd', b'd', 0x00, 0x00, // code section
            0x0a, 0x09, 0x01, // 1 body
            0x07, // body size
            0x00, // 0 local decls
            0x20, 0x00, // local.get 0
            0x20, 0x01, // local.get 1
            0x6a, // i32.add
            0x0b, // end
        ];
        let m = decode(bytes).unwrap();
        let inst = Instance::instantiate(m, &Imports::default()).unwrap();
        let r = inst
            .invoke("add", &[Value::I32(40), Value::I32(2)])
            .unwrap()
            .unwrap();
        match r {
            Value::I32(v) => assert_eq!(v, 42),
            _ => panic!("expected i32"),
        }
    }

    #[test]
    fn memory_grow() {
        let mut m = Memory::new(1, Some(4));
        assert_eq!(m.pages(), 1);
        assert_eq!(m.grow(2), 1);
        assert_eq!(m.pages(), 3);
        assert_eq!(m.grow(2), -1); // would exceed max
    }

    #[test]
    fn leb_roundtrip() {
        // -64 encoded as i32 LEB is [0x40].
        let mut r = Reader::new(&[0x40]);
        assert_eq!(r.i32_leb().unwrap(), -64);

        // 624485 = 0xe5, 0x8e, 0x26.
        let mut r = Reader::new(&[0xe5, 0x8e, 0x26]);
        assert_eq!(r.u32_leb().unwrap(), 624_485);
    }

    #[test]
    fn limits_parse() {
        let mut r = Reader::new(&[0x01, 0x02, 0x10]);
        let l = r.limits().unwrap();
        assert_eq!(l.min, 2);
        assert_eq!(l.max, Some(16));
    }

    // ===== Layer-A: hand-built opcode coverage =====

    /// LEB128 unsigned encode (for hand-built module section sizes / counts).
    fn uleb(mut n: u32) -> Vec<u8> {
        let mut out = Vec::new();
        loop {
            let mut byte = (n & 0x7f) as u8;
            n >>= 7;
            if n != 0 {
                byte |= 0x80;
            }
            out.push(byte);
            if n == 0 {
                break;
            }
        }
        out
    }

    /// Build a minimal single-export-function module. `params`/`results` are
    /// valtype bytes (0x7f i32, 0x7e i64, 0x7d f32, 0x7c f64). `body` is the
    /// function body WITHOUT the local-decl-count or trailing `end` (we append
    /// `0x00` local decls + `0x0b`).
    fn build_func_module(
        name: &str,
        params: &[u8],
        results: &[u8],
        body: &[u8],
    ) -> Vec<u8> {
        let mut out: Vec<u8> = vec![0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00];

        // Type section (id 1): one functype.
        let mut ty: Vec<u8> = Vec::new();
        ty.push(0x01); // 1 type
        ty.push(0x60); // functype tag
        ty.extend(uleb(params.len() as u32));
        ty.extend_from_slice(params);
        ty.extend(uleb(results.len() as u32));
        ty.extend_from_slice(results);
        out.push(0x01);
        out.extend(uleb(ty.len() as u32));
        out.extend(ty);

        // Function section (id 3): one func of type 0.
        let func = vec![0x01u8, 0x00];
        out.push(0x03);
        out.extend(uleb(func.len() as u32));
        out.extend(func);

        // Export section (id 7): name -> func 0.
        let mut ex: Vec<u8> = Vec::new();
        ex.push(0x01); // 1 export
        ex.extend(uleb(name.len() as u32));
        ex.extend_from_slice(name.as_bytes());
        ex.push(0x00); // func kind
        ex.push(0x00); // func index 0
        out.push(0x07);
        out.extend(uleb(ex.len() as u32));
        out.extend(ex);

        // Code section (id 10): one body.
        let mut fb: Vec<u8> = Vec::new();
        fb.push(0x00); // 0 local decls
        fb.extend_from_slice(body);
        fb.push(0x0b); // end
        let mut code: Vec<u8> = Vec::new();
        code.push(0x01); // 1 body
        code.extend(uleb(fb.len() as u32));
        code.extend(fb);
        out.push(0x0a);
        out.extend(uleb(code.len() as u32));
        out.extend(code);

        out
    }

    fn invoke1(bytes: &[u8], name: &str, args: &[Value]) -> Result<Option<Value>, WasmError> {
        let m = decode(bytes).unwrap();
        let inst = Instance::instantiate(m, &Imports::default()).unwrap();
        inst.invoke(name, args)
    }

    #[test]
    fn i64_compares() {
        // f(a,b) = a < b (signed) -> i32
        let lt = build_func_module(
            "f",
            &[0x7e, 0x7e],
            &[0x7f],
            &[0x20, 0x00, 0x20, 0x01, 0x53],
        );
        assert!(matches!(
            invoke1(&lt, "f", &[Value::I64(-5), Value::I64(3)]).unwrap(),
            Some(Value::I32(1))
        ));
        assert!(matches!(
            invoke1(&lt, "f", &[Value::I64(7), Value::I64(3)]).unwrap(),
            Some(Value::I32(0))
        ));

        // eqz
        let eqz = build_func_module("f", &[0x7e], &[0x7f], &[0x20, 0x00, 0x50]);
        assert!(matches!(
            invoke1(&eqz, "f", &[Value::I64(0)]).unwrap(),
            Some(Value::I32(1))
        ));
        assert!(matches!(
            invoke1(&eqz, "f", &[Value::I64(9)]).unwrap(),
            Some(Value::I32(0))
        ));

        // unsigned lt: -1 (= u64::MAX) is NOT < 1
        let ltu = build_func_module(
            "f",
            &[0x7e, 0x7e],
            &[0x7f],
            &[0x20, 0x00, 0x20, 0x01, 0x54],
        );
        assert!(matches!(
            invoke1(&ltu, "f", &[Value::I64(-1), Value::I64(1)]).unwrap(),
            Some(Value::I32(0))
        ));
    }

    #[test]
    fn i64_arith_bitwise_shift() {
        // shl with count masked mod 64: a << (b & 63)
        let shl = build_func_module(
            "f",
            &[0x7e, 0x7e],
            &[0x7e],
            &[0x20, 0x00, 0x20, 0x01, 0x86],
        );
        // 1 << 64 must be 1 << 0 = 1 (mask &63), proving the modulus is 64 not 32.
        assert!(matches!(
            invoke1(&shl, "f", &[Value::I64(1), Value::I64(64)]).unwrap(),
            Some(Value::I64(1))
        ));
        assert!(matches!(
            invoke1(&shl, "f", &[Value::I64(1), Value::I64(40)]).unwrap(),
            Some(Value::I64(v)) if v == 1i64 << 40
        ));

        // rotl by 1
        let rotl = build_func_module(
            "f",
            &[0x7e, 0x7e],
            &[0x7e],
            &[0x20, 0x00, 0x20, 0x01, 0x89],
        );
        assert!(matches!(
            invoke1(&rotl, "f", &[Value::I64(i64::MIN), Value::I64(1)]).unwrap(),
            Some(Value::I64(1))
        ));

        // and / or / xor
        let and = build_func_module(
            "f",
            &[0x7e, 0x7e],
            &[0x7e],
            &[0x20, 0x00, 0x20, 0x01, 0x83],
        );
        assert!(matches!(
            invoke1(&and, "f", &[Value::I64(0b1100), Value::I64(0b1010)]).unwrap(),
            Some(Value::I64(0b1000))
        ));
    }

    #[test]
    fn i64_div_rem_traps() {
        // div_s by zero -> IntegerDivByZero (no panic)
        let div = build_func_module(
            "f",
            &[0x7e, 0x7e],
            &[0x7e],
            &[0x20, 0x00, 0x20, 0x01, 0x7f],
        );
        assert!(matches!(
            invoke1(&div, "f", &[Value::I64(10), Value::I64(0)]),
            Err(WasmError::IntegerDivByZero)
        ));
        // i64::MIN / -1 -> IntegerOverflow
        assert!(matches!(
            invoke1(&div, "f", &[Value::I64(i64::MIN), Value::I64(-1)]),
            Err(WasmError::IntegerOverflow)
        ));
        // ordinary division works
        assert!(matches!(
            invoke1(&div, "f", &[Value::I64(20), Value::I64(5)]).unwrap(),
            Some(Value::I64(4))
        ));

        // rem_s of i64::MIN % -1 must NOT trap and yields 0
        let rem = build_func_module(
            "f",
            &[0x7e, 0x7e],
            &[0x7e],
            &[0x20, 0x00, 0x20, 0x01, 0x81],
        );
        assert!(matches!(
            invoke1(&rem, "f", &[Value::I64(i64::MIN), Value::I64(-1)]).unwrap(),
            Some(Value::I64(0))
        ));
        assert!(matches!(
            invoke1(&rem, "f", &[Value::I64(10), Value::I64(0)]),
            Err(WasmError::IntegerDivByZero)
        ));
    }

    #[test]
    fn i64_add_beyond_2_53() {
        // (2^53 + 1) computed in i64 must be exact (would be lossy via f64).
        let add = build_func_module(
            "f",
            &[0x7e, 0x7e],
            &[0x7e],
            &[0x20, 0x00, 0x20, 0x01, 0x7c],
        );
        let r = invoke1(
            &add,
            "f",
            &[Value::I64(9_007_199_254_740_992), Value::I64(1)],
        )
        .unwrap();
        assert!(matches!(r, Some(Value::I64(9_007_199_254_740_993))));
    }

    #[test]
    fn f64_compares_and_advanced() {
        // sqrt(2)
        let sqrt = build_func_module("f", &[0x7c], &[0x7c], &[0x20, 0x00, 0x9f]);
        if let Some(Value::F64(v)) = invoke1(&sqrt, "f", &[Value::F64(2.0)]).unwrap() {
            assert!((v - 2.0_f64.sqrt()).abs() < 1e-12);
        } else {
            panic!("expected f64");
        }

        // f64.eq(NaN, NaN) == 0 ; f64.ne(NaN, NaN) == 1
        let eq = build_func_module(
            "f",
            &[0x7c, 0x7c],
            &[0x7f],
            &[0x20, 0x00, 0x20, 0x01, 0x61],
        );
        assert!(matches!(
            invoke1(&eq, "f", &[Value::F64(f64::NAN), Value::F64(f64::NAN)]).unwrap(),
            Some(Value::I32(0))
        ));
        let ne = build_func_module(
            "f",
            &[0x7c, 0x7c],
            &[0x7f],
            &[0x20, 0x00, 0x20, 0x01, 0x62],
        );
        assert!(matches!(
            invoke1(&ne, "f", &[Value::F64(f64::NAN), Value::F64(f64::NAN)]).unwrap(),
            Some(Value::I32(1))
        ));

        // min/max
        let min = build_func_module(
            "f",
            &[0x7c, 0x7c],
            &[0x7c],
            &[0x20, 0x00, 0x20, 0x01, 0xa4],
        );
        assert!(matches!(
            invoke1(&min, "f", &[Value::F64(3.0), Value::F64(5.0)]).unwrap(),
            Some(Value::F64(v)) if v == 3.0
        ));
        let max = build_func_module(
            "f",
            &[0x7c, 0x7c],
            &[0x7c],
            &[0x20, 0x00, 0x20, 0x01, 0xa5],
        );
        assert!(matches!(
            invoke1(&max, "f", &[Value::F64(3.0), Value::F64(5.0)]).unwrap(),
            Some(Value::F64(v)) if v == 5.0
        ));
    }

    #[test]
    fn fmin_fmax_nan_and_signed_zero() {
        // NaN propagation — Rust's f64::min would WRONGLY return the non-NaN.
        assert!(wasm_fmin_f64(f64::NAN, 5.0).is_nan());
        assert!(wasm_fmin_f64(5.0, f64::NAN).is_nan());
        assert!(wasm_fmax_f64(f64::NAN, 5.0).is_nan());
        assert!(wasm_fmax_f32(5.0, f32::NAN).is_nan());

        // signed-zero discrimination
        assert!(wasm_fmin_f64(-0.0, 0.0).is_sign_negative());
        assert!(wasm_fmax_f64(-0.0, 0.0).is_sign_positive());
        assert!(wasm_fmin_f64(0.0, -0.0).is_sign_negative());

        // f32.nearest is round-ties-to-EVEN, not Rust's round() (ties-away).
        // 2.5 -> 2 (even), 0.5 -> 0, 3.5 -> 4.
        let near = build_func_module("f", &[0x7d], &[0x7d], &[0x20, 0x00, 0x90]);
        assert!(matches!(
            invoke1(&near, "f", &[Value::F32(2.5)]).unwrap(),
            Some(Value::F32(v)) if v == 2.0
        ));
        assert!(matches!(
            invoke1(&near, "f", &[Value::F32(0.5)]).unwrap(),
            Some(Value::F32(v)) if v == 0.0
        ));
        assert!(matches!(
            invoke1(&near, "f", &[Value::F32(3.5)]).unwrap(),
            Some(Value::F32(v)) if v == 4.0
        ));
    }

    #[test]
    fn unreachable_traps() {
        let m = build_func_module("f", &[], &[], &[0x00]);
        assert!(matches!(
            invoke1(&m, "f", &[]),
            Err(WasmError::Unreachable)
        ));
    }

    /// Build a module with a 1-page memory, a passive-into-active data segment
    /// writing the given bytes at offset 0, and a single exported nullary
    /// function with `body`.
    fn build_mem_module(name: &str, results: &[u8], data: &[u8], body: &[u8]) -> Vec<u8> {
        let mut out: Vec<u8> = vec![0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00];

        // Type section: one functype () -> results.
        let mut ty: Vec<u8> = Vec::new();
        ty.push(0x01);
        ty.push(0x60);
        ty.extend(uleb(0)); // 0 params
        ty.extend(uleb(results.len() as u32));
        ty.extend_from_slice(results);
        out.push(0x01);
        out.extend(uleb(ty.len() as u32));
        out.extend(ty);

        // Function section.
        out.push(0x03);
        out.extend(uleb(2));
        out.extend([0x01, 0x00]);

        // Memory section (id 5): 1 memory, limits min=1 no max.
        out.push(0x05);
        out.extend(uleb(3));
        out.extend([0x01, 0x00, 0x01]);

        // Export section.
        let mut ex: Vec<u8> = Vec::new();
        ex.push(0x01);
        ex.extend(uleb(name.len() as u32));
        ex.extend_from_slice(name.as_bytes());
        ex.push(0x00);
        ex.push(0x00);
        out.push(0x07);
        out.extend(uleb(ex.len() as u32));
        out.extend(ex);

        // Code section.
        let mut fb: Vec<u8> = Vec::new();
        fb.push(0x00);
        fb.extend_from_slice(body);
        fb.push(0x0b);
        let mut code: Vec<u8> = Vec::new();
        code.push(0x01);
        code.extend(uleb(fb.len() as u32));
        code.extend(fb);
        out.push(0x0a);
        out.extend(uleb(code.len() as u32));
        out.extend(code);

        // Data section (id 11): 1 active segment, mem 0, offset i32.const 0.
        let mut ds: Vec<u8> = Vec::new();
        ds.push(0x01); // 1 segment
        ds.push(0x00); // mem idx 0
        ds.extend([0x41, 0x00, 0x0b]); // i32.const 0, end
        ds.extend(uleb(data.len() as u32));
        ds.extend_from_slice(data);
        out.push(0x0b);
        out.extend(uleb(ds.len() as u32));
        out.extend(ds);

        out
    }

    #[test]
    fn i64_load8_sign_zero_ext() {
        // memory byte 0 = 0xFF; load8_u -> 255, load8_s -> -1
        // body: i32.const 0; i64.load8_u (0x31)  [memarg align=0 offset=0]
        let lu = build_mem_module(
            "f",
            &[0x7e],
            &[0xFF],
            &[0x41, 0x00, 0x31, 0x00, 0x00],
        );
        assert!(matches!(
            invoke1(&lu, "f", &[]).unwrap(),
            Some(Value::I64(255))
        ));
        let ls = build_mem_module(
            "f",
            &[0x7e],
            &[0xFF],
            &[0x41, 0x00, 0x30, 0x00, 0x00],
        );
        assert!(matches!(
            invoke1(&ls, "f", &[]).unwrap(),
            Some(Value::I64(-1))
        ));
    }

    #[test]
    fn i64_store_then_load_roundtrip() {
        // store32 the low 4 bytes of an i64 const, then load32_u back.
        // body: i32.const 0; i64.const 0x1_0000_00AB; i64.store32; i32.const 0; i64.load32_u
        // 0x1_0000_00AB low-32 = 0x000000AB = 171.
        let mut body: Vec<u8> = Vec::new();
        body.extend([0x41, 0x00]); // i32.const 0 (addr for store)
        body.push(0x42); // i64.const
        // LEB128 of 0x1_0000_00AB = 4294967467
        body.extend(i64_sleb(0x1_0000_00AB));
        body.extend([0x3e, 0x00, 0x00]); // i64.store32 align=0 off=0
        body.extend([0x41, 0x00]); // i32.const 0 (addr for load)
        body.extend([0x35, 0x00, 0x00]); // i64.load32_u
        let m = build_mem_module("f", &[0x7e], &[], &body);
        assert!(matches!(
            invoke1(&m, "f", &[]).unwrap(),
            Some(Value::I64(0xAB))
        ));
    }

    #[test]
    fn memory_oob_traps() {
        // load i64 at a huge offset -> MemoryOutOfBounds, no panic.
        // body: i32.const 70000; i64.load8_u
        let mut body: Vec<u8> = Vec::new();
        body.push(0x41);
        body.extend(i32_sleb(70000));
        body.extend([0x31, 0x00, 0x00]);
        let m = build_mem_module("f", &[0x7e], &[], &body);
        assert!(matches!(
            invoke1(&m, "f", &[]),
            Err(WasmError::MemoryOutOfBounds)
        ));
    }

    #[test]
    fn mutable_global_set_get() {
        // global 0: mutable i32 = 0
        // export "set"(i32) = local.get 0; global.set 0
        // export "get"() -> i32 = global.get 0
        let mut out: Vec<u8> = vec![0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00];

        // Type section: type0 (i32)->(), type1 ()->i32
        let mut ty: Vec<u8> = Vec::new();
        ty.push(0x02); // 2 types
        ty.extend([0x60, 0x01, 0x7f, 0x00]); // (i32)->()
        ty.extend([0x60, 0x00, 0x01, 0x7f]); // ()->i32
        out.push(0x01);
        out.extend(uleb(ty.len() as u32));
        out.extend(ty);

        // Function section: func0 type0, func1 type1
        out.push(0x03);
        out.extend(uleb(3));
        out.extend([0x02, 0x00, 0x01]);

        // Global section (id 6): 1 global, mutable i32 = 0
        out.push(0x06);
        out.extend(uleb(6));
        out.extend([0x01, 0x7f, 0x01, 0x41, 0x00, 0x0b]);

        // Export section: "set"->func0, "get"->func1
        let mut ex: Vec<u8> = Vec::new();
        ex.push(0x02);
        ex.extend([0x03, b's', b'e', b't', 0x00, 0x00]);
        ex.extend([0x03, b'g', b'e', b't', 0x00, 0x01]);
        out.push(0x07);
        out.extend(uleb(ex.len() as u32));
        out.extend(ex);

        // Code section: 2 bodies
        let body_set = vec![0x00u8, 0x20, 0x00, 0x24, 0x00, 0x0b];
        let body_get = vec![0x00u8, 0x23, 0x00, 0x0b];
        let mut code: Vec<u8> = Vec::new();
        code.push(0x02);
        code.extend(uleb(body_set.len() as u32));
        code.extend(body_set);
        code.extend(uleb(body_get.len() as u32));
        code.extend(body_get);
        out.push(0x0a);
        out.extend(uleb(code.len() as u32));
        out.extend(code);

        let m = decode(&out).unwrap();
        let inst = Instance::instantiate(m, &Imports::default()).unwrap();
        // initially 0
        assert!(matches!(
            inst.invoke("get", &[]).unwrap(),
            Some(Value::I32(0))
        ));
        // set to 99 (the Cell persists across separate invoke calls)
        inst.invoke("set", &[Value::I32(99)]).unwrap();
        assert!(matches!(
            inst.invoke("get", &[]).unwrap(),
            Some(Value::I32(99))
        ));
    }

    #[test]
    fn immutable_global_set_traps() {
        // global 0: immutable i32 = 7; func tries global.set 0 -> GlobalImmutable
        let mut out: Vec<u8> = vec![0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00];
        // Type: (i32)->()
        out.push(0x01);
        out.extend(uleb(5));
        out.extend([0x01, 0x60, 0x01, 0x7f, 0x00]);
        // Function: func0 type0
        out.push(0x03);
        out.extend(uleb(2));
        out.extend([0x01, 0x00]);
        // Global: immutable i32 = 7
        out.push(0x06);
        out.extend(uleb(6));
        out.extend([0x01, 0x7f, 0x00, 0x41, 0x07, 0x0b]);
        // Export "f"->func0
        let mut ex: Vec<u8> = Vec::new();
        ex.push(0x01); // 1 export
        ex.extend([0x01, b'f', 0x00, 0x00]); // name len 1 "f", func kind, idx 0
        out.push(0x07);
        out.extend(uleb(ex.len() as u32));
        out.extend(ex);
        // Code: local.get 0; global.set 0
        let body = vec![0x00u8, 0x20, 0x00, 0x24, 0x00, 0x0b];
        let mut code: Vec<u8> = Vec::new();
        code.push(0x01);
        code.extend(uleb(body.len() as u32));
        code.extend(body);
        out.push(0x0a);
        out.extend(uleb(code.len() as u32));
        out.extend(code);

        let m = decode(&out).unwrap();
        let inst = Instance::instantiate(m, &Imports::default()).unwrap();
        assert!(matches!(
            inst.invoke("f", &[Value::I32(1)]),
            Err(WasmError::GlobalImmutable)
        ));
    }

    // signed-LEB helpers for hand-built const immediates
    fn i64_sleb(mut value: i64) -> Vec<u8> {
        let mut out = Vec::new();
        loop {
            let byte = (value & 0x7f) as u8;
            value >>= 7;
            let sign_bit = byte & 0x40;
            if (value == 0 && sign_bit == 0) || (value == -1 && sign_bit != 0) {
                out.push(byte);
                break;
            }
            out.push(byte | 0x80);
        }
        out
    }

    fn i32_sleb(value: i32) -> Vec<u8> {
        i64_sleb(value as i64)
    }
}
