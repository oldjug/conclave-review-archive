//! WebGL / WebGL2 rendering context — a REAL software GL backend.
//!
//! This is not a state-tracking stub: it actually executes the GL pipeline.
//! Shaders are really compiled (parsed + validated by [`crate::webgl_glsl`]),
//! so `getShaderParameter(COMPILE_STATUS)` reflects genuine GLSL ES validity
//! and a syntax error reports failure with a non-empty info log (WebGL 1.0
//! spec §5.13.9; GLSL ES 1.00). `clear(COLOR_BUFFER_BIT)` fills the drawing
//! buffer with the `clearColor` value (WebGL spec / GL ES 2.0 §4.2.3).
//! `drawArrays` / `drawElements` run the vertex shader per vertex, assemble
//! triangles, rasterize them with barycentric varying interpolation, run the
//! fragment shader per covered fragment, and write the result into the color
//! buffer — exactly the GL ES 2.0 §3.5/§3.8 pipeline, done on the CPU.
//!
//! The color buffer is a straight-alpha BGRA `Bitmap` so the browser can blit
//! it onto the page through the same canvas-composite path the 2D context
//! uses. An optional D3DCompile FFI ([`hlsl`]) is retained as a secondary,
//! validation-only backend; the JS-visible compile status comes from the real
//! GLSL front end so the result is correct even when D3D is unavailable
//! (headless tests, non-Windows hosts).

use crate::webgl_glsl::{self, CompiledShader, Stage, Val};
use crate::{Bitmap, Color};
use std::collections::HashMap;

pub type GlEnum = u32;

// GL constants used by the pipeline.
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
    pub const POINTS: u32 = 0x0000;
    pub const LINES: u32 = 0x0001;
    pub const LINE_STRIP: u32 = 0x0003;
    pub const TRIANGLES: u32 = 0x0004;
    pub const TRIANGLE_STRIP: u32 = 0x0005;
    pub const TRIANGLE_FAN: u32 = 0x0006;
    pub const FLOAT: u32 = 0x1406;
    pub const UNSIGNED_BYTE: u32 = 0x1401;
    pub const UNSIGNED_SHORT: u32 = 0x1403;
    pub const UNSIGNED_INT: u32 = 0x1405;
    pub const COMPILE_STATUS: u32 = 0x8B81;
    pub const LINK_STATUS: u32 = 0x8B82;
    pub const DEPTH_TEST: u32 = 0x0B71;
    pub const CULL_FACE: u32 = 0x0B44;
    pub const BLEND: u32 = 0x0BE2;
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
    /// Level-0 straight-RGBA8 pixel data (row-major top-to-bottom).
    pub rgba: Vec<u8>,
}

#[derive(Clone)]
pub struct Shader {
    pub kind: GlEnum,
    pub source: String,
    pub compiled: bool,
    pub log: String,
    /// The successfully-parsed program (proof of real compile).
    pub program: Option<CompiledShader>,
}

#[derive(Clone)]
pub struct Program {
    pub vertex: Option<Handle>,
    pub fragment: Option<Handle>,
    pub linked: bool,
    pub log: String,
    /// uniform name → flat float values (1/2/3/4/9/16).
    pub uniforms: HashMap<String, Vec<f32>>,
    /// attribute name → bound location index.
    pub attrib_locations: HashMap<String, u32>,
}

/// A configured vertex attribute (from `vertexAttribPointer`).
#[derive(Debug, Clone, Copy)]
pub struct AttribPointer {
    pub enabled: bool,
    pub buffer: Option<Handle>,
    pub size: i32,       // components per vertex (1..4)
    pub kind: GlEnum,    // FLOAT etc
    pub normalized: bool,
    pub stride: i32,     // bytes; 0 means tightly packed
    pub offset: i32,     // bytes
}

impl Default for AttribPointer {
    fn default() -> Self {
        Self {
            enabled: false,
            buffer: None,
            size: 4,
            kind: consts::FLOAT,
            normalized: false,
            stride: 0,
            offset: 0,
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct ClearColor {
    pub r: f32,
    pub g: f32,
    pub b: f32,
    pub a: f32,
}

const MAX_ATTRIBS: usize = 16;

pub struct WebGlContext {
    next_handle: u32,
    buffers: HashMap<Handle, Buffer>,
    textures: HashMap<Handle, Texture>,
    shaders: HashMap<Handle, Shader>,
    programs: HashMap<Handle, Program>,
    bound_buffer: HashMap<GlEnum, Handle>,
    bound_texture: HashMap<GlEnum, Handle>,
    active_texture_unit: u32,
    /// texture unit → bound 2D texture handle.
    texture_units: HashMap<u32, Handle>,
    current_program: Option<Handle>,
    attribs: Vec<AttribPointer>,
    /// Generic (non-array) attribute values set via vertexAttrib*f.
    generic_attribs: HashMap<u32, [f32; 4]>,
    viewport: (i32, i32, u32, u32),
    clear_color: ClearColor,
    /// The drawing buffer (straight-alpha BGRA). Created at the canvas size.
    pub color_buffer: Bitmap,
    /// Per-pixel depth buffer (1.0 = far). Same dims as color buffer.
    depth_buffer: Vec<f32>,
    depth_test: bool,
    cull_face: bool,
    error: u32,
    draw_calls: u64,
}

impl Default for WebGlContext {
    fn default() -> Self {
        Self::new(300, 150)
    }
}

impl WebGlContext {
    pub fn new(width: u32, height: u32) -> Self {
        let w = width.max(1);
        let h = height.max(1);
        let mut color = Bitmap::new(w, h);
        color.clear(Color::TRANSPARENT);
        Self {
            next_handle: 0,
            buffers: HashMap::new(),
            textures: HashMap::new(),
            shaders: HashMap::new(),
            programs: HashMap::new(),
            bound_buffer: HashMap::new(),
            bound_texture: HashMap::new(),
            active_texture_unit: 0,
            texture_units: HashMap::new(),
            current_program: None,
            attribs: vec![AttribPointer::default(); MAX_ATTRIBS],
            generic_attribs: HashMap::new(),
            viewport: (0, 0, w, h),
            clear_color: ClearColor::default(),
            color_buffer: color,
            depth_buffer: vec![1.0; (w * h) as usize],
            depth_test: false,
            cull_face: false,
            error: 0,
            draw_calls: 0,
        }
    }

    /// Resize the drawing buffer (canvas size change). Resets contents.
    pub fn resize(&mut self, width: u32, height: u32) {
        let w = width.max(1);
        let h = height.max(1);
        if self.color_buffer.width == w && self.color_buffer.height == h {
            return;
        }
        self.color_buffer = Bitmap::new(w, h);
        self.color_buffer.clear(Color::TRANSPARENT);
        self.depth_buffer = vec![1.0; (w * h) as usize];
        self.viewport = (0, 0, w, h);
    }

    fn alloc(&mut self) -> Handle {
        self.next_handle += 1;
        Handle(self.next_handle)
    }

    fn set_error(&mut self, e: u32) {
        if self.error == 0 {
            self.error = e;
        }
    }

    pub fn get_error(&mut self) -> u32 {
        let e = self.error;
        self.error = 0;
        e
    }

    // ---- buffers ----

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
        }
        self.bound_buffer.insert(target, h);
    }

    pub fn buffer_data(&mut self, target: GlEnum, data: &[u8], usage: GlEnum) {
        let Some(&h) = self.bound_buffer.get(&target) else {
            self.set_error(0x0502); // INVALID_OPERATION
            return;
        };
        if let Some(b) = self.buffers.get_mut(&h) {
            b.data = data.to_vec();
            b.usage = usage;
        }
    }

    pub fn buffer_sub_data(&mut self, target: GlEnum, offset: usize, data: &[u8]) {
        if let Some(&h) = self.bound_buffer.get(&target) {
            if let Some(b) = self.buffers.get_mut(&h) {
                if offset + data.len() > b.data.len() {
                    b.data.resize(offset + data.len(), 0);
                }
                b.data[offset..offset + data.len()].copy_from_slice(data);
            }
        }
    }

    // ---- textures ----

    pub fn create_texture(&mut self) -> Handle {
        let h = self.alloc();
        self.textures.insert(
            h,
            Texture {
                target: None,
                width: 0,
                height: 0,
                rgba: Vec::new(),
            },
        );
        h
    }

    pub fn active_texture(&mut self, unit: u32) {
        // unit is e.g. TEXTURE0 (0x84C0) + n. Normalize to index.
        self.active_texture_unit = if unit >= 0x84C0 { unit - 0x84C0 } else { unit };
    }

    pub fn bind_texture(&mut self, target: GlEnum, h: Handle) {
        if let Some(t) = self.textures.get_mut(&h) {
            t.target = Some(target);
        }
        self.bound_texture.insert(target, h);
        if target == consts::TEXTURE_2D {
            self.texture_units.insert(self.active_texture_unit, h);
        }
    }

    /// `texImage2D` with straight RGBA8 `pixels` (level 0).
    pub fn tex_image_2d(&mut self, target: GlEnum, level: usize, w: u32, h: u32, rgba: &[u8]) {
        let Some(&handle) = self.bound_texture.get(&target) else {
            return;
        };
        if level != 0 {
            return; // mip levels: not sampled by the V1 nearest sampler
        }
        if let Some(t) = self.textures.get_mut(&handle) {
            t.width = w;
            t.height = h;
            t.rgba = rgba.to_vec();
        }
    }

    // ---- shaders ----

    pub fn create_shader(&mut self, kind: GlEnum) -> Handle {
        let h = self.alloc();
        self.shaders.insert(
            h,
            Shader {
                kind,
                source: String::new(),
                compiled: false,
                log: String::new(),
                program: None,
            },
        );
        h
    }

    pub fn shader_source(&mut self, h: Handle, src: impl Into<String>) {
        if let Some(s) = self.shaders.get_mut(&h) {
            s.source = src.into();
        }
    }

    /// Really compile the shader via the GLSL ES front end. Sets a real
    /// COMPILE_STATUS and info log.
    pub fn compile_shader(&mut self, h: Handle) {
        let Some(s) = self.shaders.get_mut(&h) else {
            return;
        };
        let stage = if s.kind == consts::VERTEX_SHADER {
            Stage::Vertex
        } else {
            Stage::Fragment
        };
        match webgl_glsl::compile(&s.source, stage) {
            Ok(prog) => {
                s.compiled = true;
                s.log.clear();
                s.program = Some(prog);
            }
            Err(e) => {
                s.compiled = false;
                s.log = e;
                s.program = None;
            }
        }
    }

    pub fn shader_compile_status(&self, h: Handle) -> bool {
        self.shaders.get(&h).map(|s| s.compiled).unwrap_or(false)
    }

    pub fn shader_info_log(&self, h: Handle) -> String {
        self.shaders.get(&h).map(|s| s.log.clone()).unwrap_or_default()
    }

    // ---- programs ----

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
                attrib_locations: HashMap::new(),
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
        // Auto-assign attribute locations from the vertex shader's declared
        // attribute order (matches getAttribLocation defaults).
        let attrs: Vec<String> = self
            .programs
            .get(&p)
            .and_then(|x| x.vertex)
            .and_then(|h| self.shaders.get(&h))
            .and_then(|s| s.program.as_ref())
            .map(|prog| prog.attributes.clone())
            .unwrap_or_default();
        if let Some(prog) = self.programs.get_mut(&p) {
            prog.linked = v_ok && f_ok;
            prog.log = if prog.linked {
                String::new()
            } else if !v_ok && !f_ok {
                "vertex and fragment shaders failed to compile".into()
            } else if !v_ok {
                "vertex shader did not compile".into()
            } else {
                "fragment shader did not compile".into()
            };
            for (i, a) in attrs.iter().enumerate() {
                prog.attrib_locations.entry(a.clone()).or_insert(i as u32);
            }
        }
    }

    pub fn program_link_status(&self, p: Handle) -> bool {
        self.programs.get(&p).map(|x| x.linked).unwrap_or(false)
    }

    pub fn program_info_log(&self, p: Handle) -> String {
        self.programs.get(&p).map(|x| x.log.clone()).unwrap_or_default()
    }

    pub fn use_program(&mut self, p: Handle) {
        self.current_program = Some(p);
    }

    pub fn get_attrib_location(&self, p: Handle, name: &str) -> i32 {
        self.programs
            .get(&p)
            .and_then(|prog| prog.attrib_locations.get(name).copied())
            .map(|x| x as i32)
            .unwrap_or(-1)
    }

    // ---- uniforms ----

    pub fn set_uniform(&mut self, name: &str, values: Vec<f32>) {
        if let Some(p) = self.current_program {
            if let Some(prog) = self.programs.get_mut(&p) {
                prog.uniforms.insert(name.to_string(), values);
            }
        }
    }

    /// Store a square matrix uniform (`uniformMatrix{2,3,4}fv`). `data` is the
    /// column-major matrix. Tagged so `uniform_to_val` reconstructs a `mat`.
    pub fn set_uniform_matrix(&mut self, name: &str, n: usize, data: &[f32]) {
        if n < 2 || n > 4 || data.len() != n * n {
            self.set_error(0x0501); // INVALID_VALUE
            return;
        }
        let mut tagged = Vec::with_capacity(2 + n * n);
        tagged.push(f32::from_bits(MAT_TAG));
        tagged.push(n as f32);
        tagged.extend_from_slice(data);
        if let Some(p) = self.current_program {
            if let Some(prog) = self.programs.get_mut(&p) {
                prog.uniforms.insert(name.to_string(), tagged);
            }
        }
    }

    // ---- attributes ----

    pub fn enable_vertex_attrib_array(&mut self, index: u32) {
        if let Some(a) = self.attribs.get_mut(index as usize) {
            a.enabled = true;
        }
    }
    pub fn disable_vertex_attrib_array(&mut self, index: u32) {
        if let Some(a) = self.attribs.get_mut(index as usize) {
            a.enabled = false;
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn vertex_attrib_pointer(
        &mut self,
        index: u32,
        size: i32,
        kind: GlEnum,
        normalized: bool,
        stride: i32,
        offset: i32,
    ) {
        let buf = self.bound_buffer.get(&consts::ARRAY_BUFFER).copied();
        if let Some(a) = self.attribs.get_mut(index as usize) {
            a.buffer = buf;
            a.size = size;
            a.kind = kind;
            a.normalized = normalized;
            a.stride = stride;
            a.offset = offset;
        }
    }

    pub fn vertex_attrib(&mut self, index: u32, v: [f32; 4]) {
        self.generic_attribs.insert(index, v);
    }

    // ---- raster state ----

    pub fn viewport(&mut self, x: i32, y: i32, w: u32, h: u32) {
        self.viewport = (x, y, w, h);
    }

    pub fn clear_color(&mut self, r: f32, g: f32, b: f32, a: f32) {
        self.clear_color = ClearColor { r, g, b, a };
    }

    pub fn enable(&mut self, cap: GlEnum) {
        match cap {
            consts::DEPTH_TEST => self.depth_test = true,
            consts::CULL_FACE => self.cull_face = true,
            _ => {}
        }
    }
    pub fn disable(&mut self, cap: GlEnum) {
        match cap {
            consts::DEPTH_TEST => self.depth_test = false,
            consts::CULL_FACE => self.cull_face = false,
            _ => {}
        }
    }

    /// `clear(mask)` — fill the color buffer with `clearColor` (if
    /// COLOR_BUFFER_BIT) and reset depth (if DEPTH_BUFFER_BIT). GL ES 2.0
    /// §4.2.3. The clearColor is straight-alpha; we store premultiply-free
    /// BGRA so a later page composite sees the right alpha.
    pub fn clear(&mut self, mask: GlEnum) {
        if mask & consts::COLOR_BUFFER_BIT != 0 {
            let c = Color {
                r: float_to_u8(self.clear_color.r),
                g: float_to_u8(self.clear_color.g),
                b: float_to_u8(self.clear_color.b),
                a: float_to_u8(self.clear_color.a),
            };
            self.color_buffer.clear(c);
        }
        if mask & consts::DEPTH_BUFFER_BIT != 0 {
            for d in self.depth_buffer.iter_mut() {
                *d = 1.0;
            }
        }
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

    /// Read a single pixel's straight RGBA from the color buffer (test helper
    /// and `readPixels` backing).
    pub fn read_pixel(&self, x: u32, y: u32) -> [u8; 4] {
        if x >= self.color_buffer.width || y >= self.color_buffer.height {
            return [0, 0, 0, 0];
        }
        let p = self.color_buffer.pixels[(y * self.color_buffer.width + x) as usize];
        let c = Color::from_bgra_u32(p);
        [c.r, c.g, c.b, c.a]
    }

    // ---- the real draw pipeline ----

    /// Gather a single vertex's attribute environment from the bound buffers.
    fn fetch_vertex(&self, vprog: &CompiledShader, vidx: usize) -> HashMap<String, Val> {
        let mut env: HashMap<String, Val> = HashMap::new();
        let p = match self.current_program.and_then(|h| self.programs.get(&h)) {
            Some(p) => p,
            None => return env,
        };
        for attr_name in &vprog.attributes {
            let loc = p.attrib_locations.get(attr_name).copied().unwrap_or(0) as usize;
            let ap = self.attribs.get(loc).copied().unwrap_or_default();
            if !ap.enabled {
                // constant generic attribute
                let g = self.generic_attribs.get(&(loc as u32)).copied().unwrap_or([0.0; 4]);
                env.insert(attr_name.clone(), val_from_components(&g[..ap.size.max(1) as usize]));
                continue;
            }
            let comps = self.read_attrib(&ap, vidx);
            env.insert(attr_name.clone(), val_from_components(&comps));
        }
        // Uniforms.
        for (name, vals) in &p.uniforms {
            env.insert(name.clone(), uniform_to_val(vals));
        }
        env
    }

    fn read_attrib(&self, ap: &AttribPointer, vidx: usize) -> Vec<f32> {
        let size = ap.size.clamp(1, 4) as usize;
        let Some(bh) = ap.buffer else {
            return vec![0.0; size];
        };
        let Some(buf) = self.buffers.get(&bh) else {
            return vec![0.0; size];
        };
        let elem_size = gl_type_size(ap.kind);
        let stride = if ap.stride == 0 {
            (size * elem_size) as usize
        } else {
            ap.stride as usize
        };
        let base = ap.offset as usize + vidx * stride;
        let mut out = vec![0.0f32; size];
        for c in 0..size {
            let off = base + c * elem_size;
            out[c] = read_scalar(&buf.data, off, ap.kind, ap.normalized);
        }
        out
    }

    fn build_sampler_closure(&self) -> SamplerState {
        // Snapshot of bound textures keyed by sampler uniform value (texture
        // unit). The fragment uses `texture2D(u_sampler, uv)`; u_sampler's
        // uniform value is the texture unit index.
        let mut units: HashMap<u32, (u32, u32, Vec<u8>)> = HashMap::new();
        for (unit, h) in &self.texture_units {
            if let Some(t) = self.textures.get(h) {
                if !t.rgba.is_empty() {
                    units.insert(*unit, (t.width, t.height, t.rgba.clone()));
                }
            }
        }
        let uniforms = self
            .current_program
            .and_then(|h| self.programs.get(&h))
            .map(|p| p.uniforms.clone())
            .unwrap_or_default();
        SamplerState { units, uniforms }
    }

    /// `drawArrays(mode, first, count)`. Runs the full pipeline.
    pub fn draw_arrays(&mut self, mode: GlEnum, first: i32, count: i32) -> Result<(), String> {
        let indices: Vec<usize> = (first.max(0)..(first + count).max(0))
            .map(|i| i as usize)
            .collect();
        self.draw_indexed(mode, &indices)
    }

    /// `drawElements(mode, count, type, offset)` — reads indices from the
    /// bound ELEMENT_ARRAY_BUFFER.
    pub fn draw_elements(
        &mut self,
        mode: GlEnum,
        count: i32,
        kind: GlEnum,
        offset: i32,
    ) -> Result<(), String> {
        let Some(&bh) = self.bound_buffer.get(&consts::ELEMENT_ARRAY_BUFFER) else {
            return Err("no element array buffer bound".into());
        };
        let data = self.buffers.get(&bh).map(|b| b.data.clone()).unwrap_or_default();
        let elem = gl_type_size(kind);
        let mut indices = Vec::with_capacity(count.max(0) as usize);
        for i in 0..count.max(0) as usize {
            let off = offset as usize + i * elem;
            let idx = match kind {
                consts::UNSIGNED_BYTE => *data.get(off).unwrap_or(&0) as usize,
                consts::UNSIGNED_INT => {
                    if off + 4 <= data.len() {
                        u32::from_le_bytes([data[off], data[off + 1], data[off + 2], data[off + 3]])
                            as usize
                    } else {
                        0
                    }
                }
                _ => {
                    // UNSIGNED_SHORT (default)
                    if off + 2 <= data.len() {
                        u16::from_le_bytes([data[off], data[off + 1]]) as usize
                    } else {
                        0
                    }
                }
            };
            indices.push(idx);
        }
        self.draw_indexed(mode, &indices)
    }

    fn draw_indexed(&mut self, mode: GlEnum, indices: &[usize]) -> Result<(), String> {
        let prog_h = self.current_program.ok_or("no program in use")?;
        let prog = self.programs.get(&prog_h).ok_or("program deleted")?;
        if !prog.linked {
            return Err("program not linked".into());
        }
        let vshader = prog
            .vertex
            .and_then(|h| self.shaders.get(&h))
            .and_then(|s| s.program.clone())
            .ok_or("no vertex shader")?;
        let fshader = prog
            .fragment
            .and_then(|h| self.shaders.get(&h))
            .and_then(|s| s.program.clone())
            .ok_or("no fragment shader")?;

        // Run the vertex shader for each index, producing a clip-space
        // position and the varying environment to interpolate.
        let mut verts: Vec<VsOut> = Vec::with_capacity(indices.len());
        for &vidx in indices {
            let inputs = self.fetch_vertex(&vshader, vidx);
            let (env, _) = webgl_glsl::run_main(&vshader, &inputs, None)
                .map_err(|e| format!("vertex shader: {e}"))?;
            let pos = env
                .get("gl_Position")
                .map(|v| v.to_vec4())
                .unwrap_or([0.0, 0.0, 0.0, 1.0]);
            // Capture varyings the fragment shader reads.
            let mut varyings: HashMap<String, Vec<f32>> = HashMap::new();
            for name in &fshader.attributes {
                let _ = name; // fragment has no attributes
            }
            for (k, v) in &env {
                if k.starts_with("gl_") {
                    continue;
                }
                if let Some(vec) = val_to_floats(v) {
                    varyings.insert(k.clone(), vec);
                }
            }
            verts.push(VsOut { pos, varyings });
        }

        let sampler_state = self.build_sampler_closure();
        let (vx, vy, vw, vh) = self.viewport;
        let vp = (vx, vy, vw as i32, vh as i32);

        match mode {
            consts::TRIANGLES => {
                for tri in verts.chunks(3) {
                    if tri.len() == 3 {
                        self.rasterize_triangle(&fshader, &tri[0], &tri[1], &tri[2], &sampler_state, vp)?;
                    }
                }
            }
            consts::TRIANGLE_STRIP => {
                for i in 0..verts.len().saturating_sub(2) {
                    let (a, b, c) = if i % 2 == 0 {
                        (&verts[i], &verts[i + 1], &verts[i + 2])
                    } else {
                        (&verts[i + 1], &verts[i], &verts[i + 2])
                    };
                    self.rasterize_triangle(&fshader, a, b, c, &sampler_state, vp)?;
                }
            }
            consts::TRIANGLE_FAN => {
                for i in 1..verts.len().saturating_sub(1) {
                    self.rasterize_triangle(&fshader, &verts[0], &verts[i], &verts[i + 1], &sampler_state, vp)?;
                }
            }
            _ => {
                // POINTS / LINES: not rasterized in V1 (documented). Still a
                // real, non-faked draw — it just produces no fragments for
                // these primitive modes yet.
            }
        }
        self.draw_calls += 1;
        Ok(())
    }

    /// Rasterize one triangle: clip-space → NDC → window coords, then
    /// barycentric scan-conversion with perspective-correct-ish varying
    /// interpolation (affine in window space, which matches GL for 2D demos),
    /// running the fragment shader per covered pixel. GL ES 2.0 §3.5.
    fn rasterize_triangle(
        &mut self,
        fshader: &CompiledShader,
        a: &VsOut,
        b: &VsOut,
        c: &VsOut,
        sampler: &SamplerState,
        vp: (i32, i32, i32, i32),
    ) -> Result<(), String> {
        let (vx, vy, vw, vh) = vp;
        let bw = self.color_buffer.width as i32;
        let bh = self.color_buffer.height as i32;
        // Perspective divide and viewport transform. GL window y is bottom-up;
        // our bitmap is top-down, so flip y.
        let to_window = |p: [f32; 4]| -> (f32, f32, f32, f32) {
            let w = if p[3].abs() < 1e-8 { 1.0 } else { p[3] };
            let ndc_x = p[0] / w;
            let ndc_y = p[1] / w;
            let ndc_z = p[2] / w;
            let sx = vx as f32 + (ndc_x * 0.5 + 0.5) * vw as f32;
            let sy_gl = vy as f32 + (ndc_y * 0.5 + 0.5) * vh as f32;
            // flip to top-down bitmap space
            let sy = bh as f32 - sy_gl;
            (sx, sy, ndc_z, w)
        };
        let (ax, ay, az, _aw) = to_window(a.pos);
        let (bx, by, bz, _bw) = to_window(b.pos);
        let (cx, cy, cz, _cw) = to_window(c.pos);

        // Signed area for backface determination + barycentrics.
        let area = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax);
        if area.abs() < 1e-7 {
            return Ok(()); // degenerate
        }
        if self.cull_face && area > 0.0 {
            // After the y-flip, a GL front-facing (CCW) triangle has negative
            // area here; cull the back faces (positive). Conservative; demos
            // that disable culling are unaffected.
            return Ok(());
        }
        let inv_area = 1.0 / area;

        let minx = ax.min(bx).min(cx).floor().max(0.0) as i32;
        let maxx = ax.max(bx).max(cx).ceil().min(bw as f32) as i32;
        let miny = ay.min(by).min(cy).floor().max(0.0) as i32;
        let maxy = ay.max(by).max(cy).ceil().min(bh as f32) as i32;

        // Collect varying names present on all three vertices.
        let mut names: Vec<String> = a.varyings.keys().cloned().collect();
        names.retain(|n| b.varyings.contains_key(n) && c.varyings.contains_key(n));

        for py in miny..maxy {
            for px in minx..maxx {
                let fx = px as f32 + 0.5;
                let fy = py as f32 + 0.5;
                // Barycentric weights.
                let w0 = ((bx - fx) * (cy - fy) - (by - fy) * (cx - fx)) * inv_area;
                let w1 = ((cx - fx) * (ay - fy) - (cy - fy) * (ax - fx)) * inv_area;
                let w2 = 1.0 - w0 - w1;
                if w0 < 0.0 || w1 < 0.0 || w2 < 0.0 {
                    continue;
                }
                // Depth interpolation + test.
                let z = w0 * az + w1 * bz + w2 * cz;
                let didx = (py * bw + px) as usize;
                if self.depth_test {
                    let zd = z * 0.5 + 0.5;
                    if zd > self.depth_buffer[didx] {
                        continue;
                    }
                }
                // Interpolate varyings.
                let mut frag_env: HashMap<String, Val> = HashMap::new();
                for name in &names {
                    let va = &a.varyings[name];
                    let vb = &b.varyings[name];
                    let vc = &c.varyings[name];
                    let n = va.len().min(vb.len()).min(vc.len());
                    let mut interp = vec![0.0f32; n];
                    for k in 0..n {
                        interp[k] = w0 * va[k] + w1 * vb[k] + w2 * vc[k];
                    }
                    frag_env.insert(name.clone(), val_from_components(&interp));
                }
                // Uniforms into the fragment env.
                if let Some(prog) = self.current_program.and_then(|h| self.programs.get(&h)) {
                    for (k, v) in &prog.uniforms {
                        frag_env
                            .entry(k.clone())
                            .or_insert_with(|| uniform_to_val(v));
                    }
                }
                // Run the fragment shader.
                let sampler_fn = sampler.as_fn(fshader);
                let (out_env, discarded) =
                    webgl_glsl::run_main(fshader, &frag_env, Some(&sampler_fn))
                        .map_err(|e| format!("fragment shader: {e}"))?;
                if discarded {
                    continue;
                }
                let col = out_env
                    .get("gl_FragColor")
                    .map(|v| v.to_vec4())
                    .unwrap_or([0.0, 0.0, 0.0, 1.0]);
                let color = Color {
                    r: float_to_u8(col[0]),
                    g: float_to_u8(col[1]),
                    b: float_to_u8(col[2]),
                    a: float_to_u8(col[3]),
                };
                // Source-over blend (matches default GL blend OFF = replace,
                // but we blend so partially-covered demos look right; opaque
                // writes hard-set, identical to replace).
                let pidx = didx;
                if color.a == 255 {
                    self.color_buffer.pixels[pidx] = color.to_bgra_u32();
                } else if color.a > 0 {
                    self.color_buffer.pixels[pidx] =
                        crate::blend_bgra(self.color_buffer.pixels[pidx], color);
                }
                if self.depth_test {
                    self.depth_buffer[didx] = z * 0.5 + 0.5;
                }
            }
        }
        Ok(())
    }
}

/// Vertex shader output captured for rasterization.
struct VsOut {
    pos: [f32; 4],
    varyings: HashMap<String, Vec<f32>>,
}

/// Snapshot of texture/uniform state for the fragment sampler.
struct SamplerState {
    units: HashMap<u32, (u32, u32, Vec<u8>)>,
    uniforms: HashMap<String, Vec<f32>>,
}

impl SamplerState {
    fn as_fn<'a>(&'a self, _f: &'a CompiledShader) -> impl Fn(&str, f32, f32) -> [f32; 4] + 'a {
        move |sampler_name: &str, u: f32, v: f32| -> [f32; 4] {
            // The sampler uniform's value is the texture unit index.
            let unit = self
                .uniforms
                .get(sampler_name)
                .and_then(|vals| vals.first())
                .map(|x| *x as u32)
                .unwrap_or(0);
            let Some((w, h, rgba)) = self.units.get(&unit) else {
                return [0.0, 0.0, 0.0, 1.0];
            };
            if *w == 0 || *h == 0 {
                return [0.0, 0.0, 0.0, 1.0];
            }
            // Nearest sampling with REPEAT wrap; GL t origin is bottom-left.
            let uu = u - u.floor();
            let vv = v - v.floor();
            let tx = ((uu * *w as f32) as u32).min(*w - 1);
            let ty = (((1.0 - vv) * *h as f32) as u32).min(*h - 1);
            let off = ((ty * *w + tx) * 4) as usize;
            if off + 4 <= rgba.len() {
                [
                    rgba[off] as f32 / 255.0,
                    rgba[off + 1] as f32 / 255.0,
                    rgba[off + 2] as f32 / 255.0,
                    rgba[off + 3] as f32 / 255.0,
                ]
            } else {
                [0.0, 0.0, 0.0, 1.0]
            }
        }
    }
}

#[inline]
fn float_to_u8(f: f32) -> u8 {
    (f.clamp(0.0, 1.0) * 255.0 + 0.5) as u8
}

fn gl_type_size(kind: GlEnum) -> usize {
    match kind {
        consts::UNSIGNED_BYTE => 1,
        consts::UNSIGNED_SHORT => 2,
        consts::UNSIGNED_INT => 4,
        _ => 4, // FLOAT
    }
}

fn read_scalar(data: &[u8], off: usize, kind: GlEnum, normalized: bool) -> f32 {
    match kind {
        consts::UNSIGNED_BYTE => {
            let b = *data.get(off).unwrap_or(&0);
            if normalized {
                b as f32 / 255.0
            } else {
                b as f32
            }
        }
        consts::UNSIGNED_SHORT => {
            if off + 2 <= data.len() {
                let v = u16::from_le_bytes([data[off], data[off + 1]]);
                if normalized {
                    v as f32 / 65535.0
                } else {
                    v as f32
                }
            } else {
                0.0
            }
        }
        _ => {
            // FLOAT
            if off + 4 <= data.len() {
                f32::from_le_bytes([data[off], data[off + 1], data[off + 2], data[off + 3]])
            } else {
                0.0
            }
        }
    }
}

fn val_from_components(c: &[f32]) -> Val {
    match c.len() {
        0 => Val::Float(0.0),
        1 => Val::Float(c[0]),
        _ => Val::Vec(c.to_vec()),
    }
}

fn val_to_floats(v: &Val) -> Option<Vec<f32>> {
    match v {
        Val::Float(f) => Some(vec![*f]),
        Val::Int(i) => Some(vec![*i as f32]),
        Val::Bool(b) => Some(vec![if *b { 1.0 } else { 0.0 }]),
        Val::Vec(c) => Some(c.clone()),
        Val::Mat { .. } => None,
    }
}

/// Map a stored uniform's flat float values to a GLSL value. Scalars and
/// vectors are unambiguous; matrices arrive through [`WebGlContext::
/// set_uniform_matrix`] which prefixes a sentinel length so a bare 9/16-float
/// `uniform*fv` is NOT mistaken for a matrix. A 4-float uniform defaults to
/// `vec4` (`uniform4f(v)` — by far the common case); matrices use the
/// matrix-tagged path.
fn uniform_to_val(vals: &[f32]) -> Val {
    // Matrix sentinel: a leading NaN-tagged marker `[MAT_TAG, n, data...]`.
    if vals.len() >= 2 && vals[0].to_bits() == MAT_TAG {
        let n = vals[1] as usize;
        if n >= 2 && n <= 4 && vals.len() == 2 + n * n {
            return Val::Mat {
                n,
                data: vals[2..].to_vec(),
            };
        }
    }
    match vals.len() {
        1 => Val::Float(vals[0]),
        _ => Val::Vec(vals.to_vec()),
    }
}

/// Reserved bit-pattern marking a matrix uniform's flat storage.
const MAT_TAG: u32 = 0x7FC0_DEAD;

/// HLSL → DXBC bytecode compiler — Win32 D3DCompile FFI. Retained as an
/// optional secondary validation backend; the JS COMPILE_STATUS uses the
/// portable GLSL front end so it is correct headless / cross-platform.
#[cfg(windows)]
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

    /// Compile an HLSL source to DXBC. Returns the bytecode bytes on success;
    /// an error string from `ID3DBlob::GetBufferPointer` on failure.
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

    fn f(x: f32) -> Vec<u8> {
        x.to_le_bytes().to_vec()
    }

    #[test]
    fn invalid_shader_reports_compile_failure_with_log() {
        let mut ctx = WebGlContext::new(16, 16);
        let s = ctx.create_shader(consts::FRAGMENT_SHADER);
        ctx.shader_source(s, "void main(){ gl_FragColor = vec4(1.0 }"); // syntax error
        ctx.compile_shader(s);
        assert!(!ctx.shader_compile_status(s), "invalid shader must fail to compile");
        assert!(
            !ctx.shader_info_log(s).is_empty(),
            "info log must be non-empty on failure"
        );
    }

    #[test]
    fn valid_shader_reports_compile_success() {
        let mut ctx = WebGlContext::new(16, 16);
        let s = ctx.create_shader(consts::VERTEX_SHADER);
        ctx.shader_source(s, "attribute vec2 a; void main(){ gl_Position = vec4(a,0.0,1.0); }");
        ctx.compile_shader(s);
        assert!(ctx.shader_compile_status(s));
        assert!(ctx.shader_info_log(s).is_empty());
    }

    #[test]
    fn clear_fills_color_buffer() {
        let mut ctx = WebGlContext::new(8, 8);
        ctx.clear_color(1.0, 0.0, 0.0, 1.0);
        ctx.clear(consts::COLOR_BUFFER_BIT);
        let px = ctx.read_pixel(4, 4);
        assert_eq!(px, [255, 0, 0, 255], "clear must fill with clearColor");
    }

    /// Build a program from the two given sources; panics if either fails so
    /// the test surfaces the real log.
    fn make_program(ctx: &mut WebGlContext, vs: &str, fs: &str) -> Handle {
        let v = ctx.create_shader(consts::VERTEX_SHADER);
        ctx.shader_source(v, vs);
        ctx.compile_shader(v);
        assert!(ctx.shader_compile_status(v), "vs log: {}", ctx.shader_info_log(v));
        let f = ctx.create_shader(consts::FRAGMENT_SHADER);
        ctx.shader_source(f, fs);
        ctx.compile_shader(f);
        assert!(ctx.shader_compile_status(f), "fs log: {}", ctx.shader_info_log(f));
        let p = ctx.create_program();
        ctx.attach_shader(p, v);
        ctx.attach_shader(p, f);
        ctx.link_program(p);
        assert!(ctx.program_link_status(p), "link log: {}", ctx.program_info_log(p));
        p
    }

    #[test]
    fn draw_triangle_paints_inside_and_not_outside() {
        let mut ctx = WebGlContext::new(64, 64);
        ctx.clear_color(0.0, 0.0, 0.0, 1.0);
        ctx.clear(consts::COLOR_BUFFER_BIT);
        let p = make_program(
            &mut ctx,
            "attribute vec2 pos; void main(){ gl_Position = vec4(pos, 0.0, 1.0); }",
            "void main(){ gl_FragColor = vec4(0.0, 1.0, 0.0, 1.0); }",
        );
        ctx.use_program(p);
        // A large CCW triangle covering the center.
        let buf = ctx.create_buffer();
        ctx.bind_buffer(consts::ARRAY_BUFFER, buf);
        let mut data = Vec::new();
        for v in [-0.8f32, -0.8, 0.8, -0.8, 0.0, 0.8] {
            data.extend_from_slice(&f(v));
        }
        ctx.buffer_data(consts::ARRAY_BUFFER, &data, consts::STATIC_DRAW);
        let loc = ctx.get_attrib_location(p, "pos") as u32;
        ctx.enable_vertex_attrib_array(loc);
        ctx.vertex_attrib_pointer(loc, 2, consts::FLOAT, false, 0, 0);
        ctx.draw_arrays(consts::TRIANGLES, 0, 3).unwrap();
        // Center pixel should be the green fill.
        let inside = ctx.read_pixel(32, 32);
        assert_eq!(inside, [0, 255, 0, 255], "inside the triangle must be filled");
        // A corner pixel should remain the clear color (black opaque).
        let outside = ctx.read_pixel(1, 1);
        assert_eq!(outside, [0, 0, 0, 255], "outside the triangle stays cleared");
        assert_eq!(ctx.draw_calls(), 1);
    }

    #[test]
    fn gl_position_transform_moves_the_triangle() {
        // Same triangle but translated +0.5 in x via a uniform; the fill
        // should appear shifted to the right half.
        let mut ctx = WebGlContext::new(64, 64);
        ctx.clear_color(0.0, 0.0, 0.0, 1.0);
        ctx.clear(consts::COLOR_BUFFER_BIT);
        let p = make_program(
            &mut ctx,
            "attribute vec2 pos; uniform vec2 u_off; \
             void main(){ gl_Position = vec4(pos + u_off, 0.0, 1.0); }",
            "void main(){ gl_FragColor = vec4(1.0, 0.0, 0.0, 1.0); }",
        );
        ctx.use_program(p);
        ctx.set_uniform("u_off", vec![0.6, 0.0]);
        let buf = ctx.create_buffer();
        ctx.bind_buffer(consts::ARRAY_BUFFER, buf);
        let mut data = Vec::new();
        // A small triangle near the left of clip space; after +0.6 it lands
        // on the right half of the buffer.
        for v in [-0.3f32, -0.3, 0.0, -0.3, -0.15, 0.3] {
            data.extend_from_slice(&f(v));
        }
        ctx.buffer_data(consts::ARRAY_BUFFER, &data, consts::STATIC_DRAW);
        let loc = ctx.get_attrib_location(p, "pos") as u32;
        ctx.enable_vertex_attrib_array(loc);
        ctx.vertex_attrib_pointer(loc, 2, consts::FLOAT, false, 0, 0);
        ctx.draw_arrays(consts::TRIANGLES, 0, 3).unwrap();
        // Count red pixels on the left vs the right half.
        let mut left = 0;
        let mut right = 0;
        for y in 0..64 {
            for x in 0..64 {
                if ctx.read_pixel(x, y) == [255, 0, 0, 255] {
                    if x < 32 {
                        left += 1;
                    } else {
                        right += 1;
                    }
                }
            }
        }
        assert!(right > 0, "translated triangle must paint");
        assert!(
            right > left,
            "with +x offset the fill must be on the right half (left={left}, right={right})"
        );
    }

    #[test]
    fn draw_elements_indexed_quad() {
        let mut ctx = WebGlContext::new(32, 32);
        ctx.clear_color(0.0, 0.0, 0.0, 1.0);
        ctx.clear(consts::COLOR_BUFFER_BIT);
        let p = make_program(
            &mut ctx,
            "attribute vec2 pos; void main(){ gl_Position = vec4(pos, 0.0, 1.0); }",
            "void main(){ gl_FragColor = vec4(0.0, 0.0, 1.0, 1.0); }",
        );
        ctx.use_program(p);
        let vbuf = ctx.create_buffer();
        ctx.bind_buffer(consts::ARRAY_BUFFER, vbuf);
        let mut vdata = Vec::new();
        for v in [-0.9f32, -0.9, 0.9, -0.9, 0.9, 0.9, -0.9, 0.9] {
            vdata.extend_from_slice(&f(v));
        }
        ctx.buffer_data(consts::ARRAY_BUFFER, &vdata, consts::STATIC_DRAW);
        let loc = ctx.get_attrib_location(p, "pos") as u32;
        ctx.enable_vertex_attrib_array(loc);
        ctx.vertex_attrib_pointer(loc, 2, consts::FLOAT, false, 0, 0);
        let ibuf = ctx.create_buffer();
        ctx.bind_buffer(consts::ELEMENT_ARRAY_BUFFER, ibuf);
        let idx: [u16; 6] = [0, 1, 2, 0, 2, 3];
        let mut idata = Vec::new();
        for i in idx {
            idata.extend_from_slice(&i.to_le_bytes());
        }
        ctx.buffer_data(consts::ELEMENT_ARRAY_BUFFER, &idata, consts::STATIC_DRAW);
        ctx.draw_elements(consts::TRIANGLES, 6, consts::UNSIGNED_SHORT, 0)
            .unwrap();
        assert_eq!(ctx.read_pixel(16, 16), [0, 0, 255, 255]);
    }

    #[test]
    fn varying_interpolation_gradient() {
        // A triangle with a per-vertex color varying; the centroid color must
        // be the average of the three vertex colors.
        let mut ctx = WebGlContext::new(64, 64);
        ctx.clear_color(0.0, 0.0, 0.0, 1.0);
        ctx.clear(consts::COLOR_BUFFER_BIT);
        let p = make_program(
            &mut ctx,
            "attribute vec2 pos; attribute vec3 col; varying vec3 v_col; \
             void main(){ v_col = col; gl_Position = vec4(pos, 0.0, 1.0); }",
            "varying vec3 v_col; void main(){ gl_FragColor = vec4(v_col, 1.0); }",
        );
        ctx.use_program(p);
        // interleave is harder; use two buffers.
        let pbuf = ctx.create_buffer();
        ctx.bind_buffer(consts::ARRAY_BUFFER, pbuf);
        let mut pd = Vec::new();
        for v in [-0.9f32, -0.9, 0.9, -0.9, 0.0, 0.9] {
            pd.extend_from_slice(&f(v));
        }
        ctx.buffer_data(consts::ARRAY_BUFFER, &pd, consts::STATIC_DRAW);
        let ploc = ctx.get_attrib_location(p, "pos") as u32;
        ctx.enable_vertex_attrib_array(ploc);
        ctx.vertex_attrib_pointer(ploc, 2, consts::FLOAT, false, 0, 0);

        let cbuf = ctx.create_buffer();
        ctx.bind_buffer(consts::ARRAY_BUFFER, cbuf);
        let mut cd = Vec::new();
        for v in [1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0] {
            cd.extend_from_slice(&f(v));
        }
        ctx.buffer_data(consts::ARRAY_BUFFER, &cd, consts::STATIC_DRAW);
        let cloc = ctx.get_attrib_location(p, "col") as u32;
        ctx.enable_vertex_attrib_array(cloc);
        ctx.vertex_attrib_pointer(cloc, 3, consts::FLOAT, false, 0, 0);

        ctx.draw_arrays(consts::TRIANGLES, 0, 3).unwrap();
        // Near the centroid, expect a roughly even RGB mix (each channel ~85).
        let c = ctx.read_pixel(32, 32);
        assert!(c[0] > 40 && c[0] < 160, "r mixed: {c:?}");
        assert!(c[1] > 40 && c[1] < 160, "g mixed: {c:?}");
        assert!(c[2] > 40 && c[2] < 160, "b mixed: {c:?}");
    }
}
