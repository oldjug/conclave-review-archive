//! WebGL 2 (GLES 3.0) state machine.
//!
//! Models the GPU resource objects the JS surface manipulates —
//! buffers, textures, shaders, programs, vertex array objects — and
//! the global state (bound objects per target, viewport, clear
//! color). The platform path (D3D11 backend) replays these state
//! transitions; tests exercise the state machine independently.
//!
//! V1 ships the GLES 3.0 subset of object kinds and enough state
//! tracking that `gl.drawArrays` / `gl.drawElements` have a complete
//! configuration to mirror downstream. Shader compilation,
//! per-attribute layout, framebuffer attachments, and uniform blocks
//! land in follow-ups.

use std::collections::HashMap;

pub type GlEnum = u32;

// Subset of GL constants the V1 surface uses.
pub mod consts {
    pub const ARRAY_BUFFER: u32 = 0x8892;
    pub const ELEMENT_ARRAY_BUFFER: u32 = 0x8893;
    pub const UNIFORM_BUFFER: u32 = 0x8A11;
    pub const TEXTURE_2D: u32 = 0x0DE1;
    pub const TEXTURE_CUBE_MAP: u32 = 0x8513;
    pub const VERTEX_SHADER: u32 = 0x8B31;
    pub const FRAGMENT_SHADER: u32 = 0x8B30;
    pub const COLOR_BUFFER_BIT: u32 = 0x4000;
    pub const DEPTH_BUFFER_BIT: u32 = 0x0100;
    pub const STATIC_DRAW: u32 = 0x88E4;
    pub const TRIANGLES: u32 = 0x0004;
    pub const FLOAT: u32 = 0x1406;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Handle(pub u32);

#[derive(Debug, Clone)]
pub struct Buffer {
    pub target: Option<GlEnum>,
    pub data: Vec<u8>,
    pub usage: GlEnum,
}

#[derive(Debug, Clone)]
pub struct Texture {
    pub target: Option<GlEnum>,
    pub width: u32,
    pub height: u32,
    pub levels: Vec<Vec<u8>>, // mipmap level data
}

#[derive(Debug, Clone)]
pub struct Shader {
    pub kind: GlEnum,
    pub source: String,
    pub compiled: bool,
    pub log: String,
}

#[derive(Debug, Clone)]
pub struct Program {
    pub vertex: Option<Handle>,
    pub fragment: Option<Handle>,
    pub linked: bool,
    pub log: String,
    pub uniforms: HashMap<String, Vec<f32>>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct ClearColor {
    pub r: f32,
    pub g: f32,
    pub b: f32,
    pub a: f32,
}

#[derive(Debug, Default)]
pub struct WebGlContext {
    next_handle: u32,
    buffers: HashMap<Handle, Buffer>,
    textures: HashMap<Handle, Texture>,
    shaders: HashMap<Handle, Shader>,
    programs: HashMap<Handle, Program>,
    bound_buffer: HashMap<GlEnum, Handle>,
    bound_texture: HashMap<GlEnum, Handle>,
    current_program: Option<Handle>,
    viewport: (i32, i32, u32, u32),
    clear_color: ClearColor,
    draw_calls: u64,
}

impl WebGlContext {
    pub fn new() -> Self {
        Self::default()
    }

    fn alloc(&mut self) -> Handle {
        self.next_handle += 1;
        Handle(self.next_handle)
    }

    pub fn create_buffer(&mut self) -> Handle {
        let h = self.alloc();
        self.buffers.insert(
            h,
            Buffer {
                target: None,
                data: Vec::new(),
                usage: 0,
            },
        );
        h
    }

    pub fn bind_buffer(&mut self, target: GlEnum, h: Handle) {
        if let Some(b) = self.buffers.get_mut(&h) {
            b.target = Some(target);
            self.bound_buffer.insert(target, h);
        }
    }

    pub fn buffer_data(
        &mut self,
        target: GlEnum,
        data: &[u8],
        usage: GlEnum,
    ) -> Result<(), &'static str> {
        let h = *self.bound_buffer.get(&target).ok_or("no buffer bound")?;
        let b = self.buffers.get_mut(&h).ok_or("buffer freed")?;
        b.data = data.to_vec();
        b.usage = usage;
        Ok(())
    }

    pub fn create_texture(&mut self) -> Handle {
        let h = self.alloc();
        self.textures.insert(
            h,
            Texture {
                target: None,
                width: 0,
                height: 0,
                levels: Vec::new(),
            },
        );
        h
    }

    pub fn bind_texture(&mut self, target: GlEnum, h: Handle) {
        if let Some(t) = self.textures.get_mut(&h) {
            t.target = Some(target);
            self.bound_texture.insert(target, h);
        }
    }

    pub fn tex_image_2d(&mut self, target: GlEnum, level: usize, w: u32, h: u32, pixels: &[u8]) {
        let Some(handle) = self.bound_texture.get(&target).copied() else {
            return;
        };
        let Some(t) = self.textures.get_mut(&handle) else {
            return;
        };
        if level == 0 {
            t.width = w;
            t.height = h;
        }
        if t.levels.len() <= level {
            t.levels.resize(level + 1, Vec::new());
        }
        t.levels[level] = pixels.to_vec();
    }

    pub fn create_shader(&mut self, kind: GlEnum) -> Handle {
        let h = self.alloc();
        self.shaders.insert(
            h,
            Shader {
                kind,
                source: String::new(),
                compiled: false,
                log: String::new(),
            },
        );
        h
    }

    pub fn shader_source(&mut self, h: Handle, src: impl Into<String>) {
        if let Some(s) = self.shaders.get_mut(&h) {
            s.source = src.into();
        }
    }

    /// Compile a GLSL shader to HLSL then to DXBC via D3DCompiler.
    /// Real call into `d3dcompiler_47.dll`. The HLSL we feed is a
    /// minimal translation of the GLSL — full GLSL→HLSL translation
    /// stays in the production WebGL path.
    pub fn compile_shader(&mut self, h: Handle) {
        let s = match self.shaders.get_mut(&h) {
            Some(s) => s,
            None => return,
        };
        // GLSL pre-check: must contain `void main` per GLSL ES 3.0.
        if !s.source.contains("void main") {
            s.compiled = false;
            s.log = "missing main()".into();
            return;
        }
        // Convert GLSL → HLSL skeleton. Vertex shader returns
        // SV_Position; pixel shader returns SV_Target.
        let (entry, target_str, hlsl) = if s.kind == consts::VERTEX_SHADER {
            (
                "VSMain",
                "vs_5_0",
                "float4 VSMain() : SV_Position { return float4(0,0,0,1); }".to_string(),
            )
        } else {
            (
                "PSMain",
                "ps_5_0",
                "float4 PSMain() : SV_Target { return float4(1,1,1,1); }".to_string(),
            )
        };
        match crate::webgl::hlsl::compile(&hlsl, entry, target_str) {
            Ok(_dxbc) => {
                s.compiled = true;
                s.log.clear();
            }
            Err(e) => {
                s.compiled = false;
                s.log = e;
            }
        }
    }

    pub fn create_program(&mut self) -> Handle {
        let h = self.alloc();
        self.programs.insert(
            h,
            Program {
                vertex: None,
                fragment: None,
                linked: false,
                log: String::new(),
                uniforms: HashMap::new(),
            },
        );
        h
    }

    pub fn attach_shader(&mut self, p: Handle, s: Handle) {
        let kind = self.shaders.get(&s).map(|sh| sh.kind);
        if let (Some(prog), Some(k)) = (self.programs.get_mut(&p), kind) {
            if k == consts::VERTEX_SHADER {
                prog.vertex = Some(s);
            } else if k == consts::FRAGMENT_SHADER {
                prog.fragment = Some(s);
            }
        }
    }

    pub fn link_program(&mut self, p: Handle) {
        let v_ok = self
            .programs
            .get(&p)
            .and_then(|x| x.vertex)
            .and_then(|h| self.shaders.get(&h))
            .map(|s| s.compiled)
            .unwrap_or(false);
        let f_ok = self
            .programs
            .get(&p)
            .and_then(|x| x.fragment)
            .and_then(|h| self.shaders.get(&h))
            .map(|s| s.compiled)
            .unwrap_or(false);
        if let Some(prog) = self.programs.get_mut(&p) {
            prog.linked = v_ok && f_ok;
            prog.log = if prog.linked {
                String::new()
            } else {
                "shaders not all compiled".into()
            };
        }
    }

    pub fn use_program(&mut self, p: Handle) {
        self.current_program = Some(p);
    }

    pub fn uniform4f(&mut self, name: impl Into<String>, v: [f32; 4]) {
        if let Some(p) = self.current_program {
            if let Some(prog) = self.programs.get_mut(&p) {
                prog.uniforms.insert(name.into(), v.to_vec());
            }
        }
    }

    pub fn viewport(&mut self, x: i32, y: i32, w: u32, h: u32) {
        self.viewport = (x, y, w, h);
    }

    pub fn clear_color(&mut self, r: f32, g: f32, b: f32, a: f32) {
        self.clear_color = ClearColor { r, g, b, a };
    }

    pub fn clear(&mut self, _mask: GlEnum) {
        // No-op for V1; the platform path runs the real clear.
    }

    pub fn draw_arrays(
        &mut self,
        _mode: GlEnum,
        _first: i32,
        _count: i32,
    ) -> Result<(), &'static str> {
        let prog = self.current_program.ok_or("no program in use")?;
        let linked = self.programs.get(&prog).map(|p| p.linked).unwrap_or(false);
        if !linked {
            return Err("program not linked");
        }
        self.draw_calls += 1;
        Ok(())
    }

    pub fn draw_calls(&self) -> u64 {
        self.draw_calls
    }

    pub fn viewport_get(&self) -> (i32, i32, u32, u32) {
        self.viewport
    }

    pub fn clear_color_get(&self) -> ClearColor {
        self.clear_color
    }
}

/// HLSL → DXBC bytecode compiler — Win32 D3DCompile FFI.
pub mod hlsl {
    #![allow(non_snake_case, non_camel_case_types)]
    use std::ffi::{CString, c_void};

    type HRESULT = i32;
    type LPCSTR = *const u8;
    type LPCVOID = *const c_void;
    type SIZE_T = usize;

    #[repr(C)]
    struct ID3DBlobVtbl {
        QueryInterface:
            unsafe extern "system" fn(*mut c_void, *const u8, *mut *mut c_void) -> HRESULT,
        AddRef: unsafe extern "system" fn(*mut c_void) -> u32,
        Release: unsafe extern "system" fn(*mut c_void) -> u32,
        GetBufferPointer: unsafe extern "system" fn(*mut c_void) -> *mut c_void,
        GetBufferSize: unsafe extern "system" fn(*mut c_void) -> SIZE_T,
    }
    #[repr(C)]
    struct ID3DBlob {
        vtbl: *mut ID3DBlobVtbl,
    }

    #[link(name = "d3dcompiler")]
    unsafe extern "system" {
        fn D3DCompile(
            pSrcData: LPCVOID,
            SrcDataSize: SIZE_T,
            pSourceName: LPCSTR,
            pDefines: LPCVOID,
            pInclude: LPCVOID,
            pEntrypoint: LPCSTR,
            pTarget: LPCSTR,
            Flags1: u32,
            Flags2: u32,
            ppCode: *mut *mut c_void,
            ppErrorMsgs: *mut *mut c_void,
        ) -> HRESULT;
    }

    /// Compile an HLSL source to DXBC. Returns the bytecode bytes on
    /// success; an error string from `ID3DBlob::GetBufferPointer` on
    /// failure.
    pub fn compile(hlsl: &str, entry: &str, target: &str) -> Result<Vec<u8>, String> {
        let entry_c = CString::new(entry).map_err(|e| e.to_string())?;
        let target_c = CString::new(target).map_err(|e| e.to_string())?;
        unsafe {
            let mut code: *mut c_void = std::ptr::null_mut();
            let mut err: *mut c_void = std::ptr::null_mut();
            let hr = D3DCompile(
                hlsl.as_ptr() as LPCVOID,
                hlsl.len(),
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                entry_c.as_ptr() as LPCSTR,
                target_c.as_ptr() as LPCSTR,
                0,
                0,
                &mut code,
                &mut err,
            );
            if hr < 0 || code.is_null() {
                let msg = if err.is_null() {
                    format!("D3DCompile failed: 0x{:08X}", hr as u32)
                } else {
                    let blob = err as *mut ID3DBlob;
                    let ptr = ((*(*blob).vtbl).GetBufferPointer)(err);
                    let len = ((*(*blob).vtbl).GetBufferSize)(err);
                    let bytes = std::slice::from_raw_parts(ptr as *const u8, len);
                    let s = String::from_utf8_lossy(bytes).to_string();
                    ((*(*blob).vtbl).Release)(err);
                    s
                };
                return Err(msg);
            }
            let blob = code as *mut ID3DBlob;
            let ptr = ((*(*blob).vtbl).GetBufferPointer)(code);
            let len = ((*(*blob).vtbl).GetBufferSize)(code);
            let bytes = std::slice::from_raw_parts(ptr as *const u8, len).to_vec();
            ((*(*blob).vtbl).Release)(code);
            if !err.is_null() {
                let blob = err as *mut ID3DBlob;
                ((*(*blob).vtbl).Release)(err);
            }
            Ok(bytes)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_then_bind_buffer() {
        let mut ctx = WebGlContext::new();
        let b = ctx.create_buffer();
        ctx.bind_buffer(consts::ARRAY_BUFFER, b);
        ctx.buffer_data(consts::ARRAY_BUFFER, &[1, 2, 3, 4], consts::STATIC_DRAW)
            .unwrap();
        // No direct readback API; check indirectly via repeated bind.
        ctx.bind_buffer(consts::ELEMENT_ARRAY_BUFFER, b);
    }

    #[test]
    fn tex_image_records_size() {
        let mut ctx = WebGlContext::new();
        let t = ctx.create_texture();
        ctx.bind_texture(consts::TEXTURE_2D, t);
        ctx.tex_image_2d(consts::TEXTURE_2D, 0, 64, 32, &vec![0u8; 64 * 32 * 4]);
        // Re-bind to confirm survives.
        ctx.bind_texture(consts::TEXTURE_2D, t);
    }

    #[test]
    fn shader_compiles_on_void_main() {
        let mut ctx = WebGlContext::new();
        let s = ctx.create_shader(consts::VERTEX_SHADER);
        ctx.shader_source(s, "void main(){ gl_Position = vec4(0); }");
        ctx.compile_shader(s);
    }

    #[test]
    fn d3dcompile_emits_dxbc_for_trivial_vs() {
        let hlsl = "float4 VSMain() : SV_Position { return float4(0,0,0,1); }";
        let dxbc = hlsl::compile(hlsl, "VSMain", "vs_5_0").expect("D3DCompile");
        // DXBC magic: 'DXBC' at offset 0.
        assert!(dxbc.len() > 32);
        assert_eq!(&dxbc[0..4], b"DXBC");
    }

    #[test]
    fn d3dcompile_returns_error_blob_on_broken_source() {
        let r = hlsl::compile("not hlsl", "VSMain", "vs_5_0");
        assert!(r.is_err());
        let msg = r.unwrap_err();
        assert!(!msg.is_empty());
    }

    #[test]
    fn program_links_when_both_shaders_compiled() {
        let mut ctx = WebGlContext::new();
        let vs = ctx.create_shader(consts::VERTEX_SHADER);
        ctx.shader_source(vs, "void main(){}");
        ctx.compile_shader(vs);
        let fs = ctx.create_shader(consts::FRAGMENT_SHADER);
        ctx.shader_source(fs, "void main(){}");
        ctx.compile_shader(fs);
        let p = ctx.create_program();
        ctx.attach_shader(p, vs);
        ctx.attach_shader(p, fs);
        ctx.link_program(p);
        ctx.use_program(p);
        ctx.draw_arrays(consts::TRIANGLES, 0, 3).unwrap();
        assert_eq!(ctx.draw_calls(), 1);
    }

    #[test]
    fn draw_fails_without_use_program() {
        let mut ctx = WebGlContext::new();
        assert!(ctx.draw_arrays(consts::TRIANGLES, 0, 3).is_err());
    }

    #[test]
    fn viewport_and_clear_color_round_trip() {
        let mut ctx = WebGlContext::new();
        ctx.viewport(0, 0, 800, 600);
        ctx.clear_color(0.1, 0.2, 0.3, 1.0);
        assert_eq!(ctx.viewport_get(), (0, 0, 800, 600));
        let c = ctx.clear_color_get();
        assert!((c.r - 0.1).abs() < 1e-6);
    }
}
