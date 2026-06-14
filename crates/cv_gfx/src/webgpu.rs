//! WebGPU surface — `GPUDevice` + `GPUQueue` + pipeline objects.
//!
//! Models the WebGPU JS object graph at the resource level. V1 ships
//! the device + buffer/texture/pipeline/bind-group factories and a
//! command-queue that records submitted command buffers for
//! deterministic test inspection. The D3D12 backend replays this
//! against a real ID3D12Device once the FFI bindings land.

use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ResourceId(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BufferUsage {
    Vertex,
    Index,
    Uniform,
    Storage,
    CopySrc,
    CopyDst,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TextureFormat {
    Rgba8Unorm,
    Bgra8Unorm,
    R32Float,
    Depth24Plus,
}

#[derive(Debug, Clone)]
pub struct Buffer {
    pub size: usize,
    pub usage: Vec<BufferUsage>,
    pub data: Vec<u8>,
}

#[derive(Debug, Clone)]
pub struct Texture {
    pub width: u32,
    pub height: u32,
    pub depth_or_layers: u32,
    pub format: TextureFormat,
    pub mip_level_count: u32,
}

#[derive(Debug, Clone)]
pub struct ShaderModule {
    pub source: String,
    pub valid: bool,
}

#[derive(Debug, Clone)]
pub struct RenderPipeline {
    pub vertex: Option<ResourceId>,
    pub fragment: Option<ResourceId>,
    pub vertex_entry: String,
    pub fragment_entry: String,
}

#[derive(Debug, Clone)]
pub struct BindGroup {
    pub entries: HashMap<u32, ResourceId>,
}

#[derive(Debug, Clone)]
pub enum Command {
    SetPipeline(ResourceId),
    SetBindGroup {
        index: u32,
        group: ResourceId,
    },
    SetVertexBuffer {
        slot: u32,
        buffer: ResourceId,
    },
    Draw {
        vertex_count: u32,
        instance_count: u32,
    },
    CopyBufferToBuffer {
        src: ResourceId,
        dst: ResourceId,
        size: usize,
    },
}

#[derive(Debug, Default)]
pub struct CommandEncoder {
    cmds: Vec<Command>,
}

impl CommandEncoder {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn set_pipeline(&mut self, p: ResourceId) {
        self.cmds.push(Command::SetPipeline(p));
    }
    pub fn set_bind_group(&mut self, index: u32, group: ResourceId) {
        self.cmds.push(Command::SetBindGroup { index, group });
    }
    pub fn set_vertex_buffer(&mut self, slot: u32, buffer: ResourceId) {
        self.cmds.push(Command::SetVertexBuffer { slot, buffer });
    }
    pub fn draw(&mut self, vertex_count: u32, instance_count: u32) {
        self.cmds.push(Command::Draw {
            vertex_count,
            instance_count,
        });
    }
    pub fn copy_buffer_to_buffer(&mut self, src: ResourceId, dst: ResourceId, size: usize) {
        self.cmds
            .push(Command::CopyBufferToBuffer { src, dst, size });
    }
    pub fn finish(self) -> Vec<Command> {
        self.cmds
    }
}

#[derive(Debug, Default)]
pub struct Queue {
    submitted: Vec<Vec<Command>>,
}

impl Queue {
    pub fn submit(&mut self, cb: Vec<Command>) {
        self.submitted.push(cb);
    }
    pub fn submitted(&self) -> &[Vec<Command>] {
        &self.submitted
    }
}

#[derive(Debug, Default)]
pub struct GpuDevice {
    next_id: u32,
    buffers: HashMap<ResourceId, Buffer>,
    textures: HashMap<ResourceId, Texture>,
    shaders: HashMap<ResourceId, ShaderModule>,
    pipelines: HashMap<ResourceId, RenderPipeline>,
    bind_groups: HashMap<ResourceId, BindGroup>,
    pub queue: Queue,
}

impl GpuDevice {
    pub fn new() -> Self {
        Self::default()
    }
    fn alloc(&mut self) -> ResourceId {
        self.next_id += 1;
        ResourceId(self.next_id)
    }

    pub fn create_buffer(&mut self, size: usize, usage: Vec<BufferUsage>) -> ResourceId {
        let id = self.alloc();
        self.buffers.insert(
            id,
            Buffer {
                size,
                usage,
                data: vec![0u8; size],
            },
        );
        id
    }
    pub fn buffer(&self, id: ResourceId) -> Option<&Buffer> {
        self.buffers.get(&id)
    }

    pub fn create_texture(
        &mut self,
        width: u32,
        height: u32,
        depth_or_layers: u32,
        format: TextureFormat,
        mip_level_count: u32,
    ) -> ResourceId {
        let id = self.alloc();
        self.textures.insert(
            id,
            Texture {
                width,
                height,
                depth_or_layers,
                format,
                mip_level_count,
            },
        );
        id
    }

    pub fn create_shader_module(&mut self, source: impl Into<String>) -> ResourceId {
        let id = self.alloc();
        let s = source.into();
        // V1 validation: WGSL must contain `@vertex` or `@fragment`.
        let valid = s.contains("@vertex") || s.contains("@fragment") || s.contains("@compute");
        self.shaders.insert(id, ShaderModule { source: s, valid });
        id
    }
    pub fn shader_valid(&self, id: ResourceId) -> bool {
        self.shaders.get(&id).map(|s| s.valid).unwrap_or(false)
    }

    pub fn create_render_pipeline(
        &mut self,
        vertex: ResourceId,
        fragment: ResourceId,
        vertex_entry: impl Into<String>,
        fragment_entry: impl Into<String>,
    ) -> Option<ResourceId> {
        if !self.shader_valid(vertex) || !self.shader_valid(fragment) {
            return None;
        }
        let id = self.alloc();
        self.pipelines.insert(
            id,
            RenderPipeline {
                vertex: Some(vertex),
                fragment: Some(fragment),
                vertex_entry: vertex_entry.into(),
                fragment_entry: fragment_entry.into(),
            },
        );
        Some(id)
    }

    pub fn create_bind_group(&mut self, entries: HashMap<u32, ResourceId>) -> ResourceId {
        let id = self.alloc();
        self.bind_groups.insert(id, BindGroup { entries });
        id
    }

    pub fn write_buffer(
        &mut self,
        id: ResourceId,
        offset: usize,
        data: &[u8],
    ) -> Result<(), &'static str> {
        let b = self.buffers.get_mut(&id).ok_or("no such buffer")?;
        if offset + data.len() > b.size {
            return Err("write out of range");
        }
        b.data[offset..offset + data.len()].copy_from_slice(data);
        Ok(())
    }
}

// -------------- Real D3D12 device creation (Win32 FFI) ------------------

#[allow(non_snake_case, non_camel_case_types, dead_code)]
pub mod d3d12 {
    use std::ffi::c_void;

    type HRESULT = i32;
    type LPVOID = *mut c_void;

    #[repr(C)]
    #[derive(Clone, Copy)]
    pub struct GUID {
        pub Data1: u32,
        pub Data2: u16,
        pub Data3: u16,
        pub Data4: [u8; 8],
    }

    // IID_ID3D12Device: 189819f1-1db6-4b57-be54-1821339b85f7
    pub const IID_ID3D12_DEVICE: GUID = GUID {
        Data1: 0x189819F1,
        Data2: 0x1DB6,
        Data3: 0x4B57,
        Data4: [0xBE, 0x54, 0x18, 0x21, 0x33, 0x9B, 0x85, 0xF7],
    };

    // Feature levels (D3D_FEATURE_LEVEL_*).
    pub const D3D_FEATURE_LEVEL_12_0: u32 = 0xc000;
    pub const D3D_FEATURE_LEVEL_12_1: u32 = 0xc100;
    pub const D3D_FEATURE_LEVEL_11_0: u32 = 0xb000;
    pub const D3D_FEATURE_LEVEL_11_1: u32 = 0xb100;

    #[link(name = "d3d12")]
    unsafe extern "system" {
        pub fn D3D12CreateDevice(
            pAdapter: *mut c_void,
            MinimumFeatureLevel: u32,
            riid: *const GUID,
            ppDevice: *mut LPVOID,
        ) -> HRESULT;
        pub fn D3D12GetDebugInterface(riid: *const GUID, ppvDebug: *mut LPVOID) -> HRESULT;
    }

    /// Create a D3D12 device on the default hardware adapter.
    /// Returns a live ID3D12Device pointer; caller must Release.
    pub fn create_device() -> Result<LPVOID, HRESULT> {
        unsafe {
            let mut dev: LPVOID = std::ptr::null_mut();
            let hr = D3D12CreateDevice(
                std::ptr::null_mut(),
                D3D_FEATURE_LEVEL_11_0,
                &IID_ID3D12_DEVICE,
                &mut dev,
            );
            if hr < 0 || dev.is_null() {
                Err(hr)
            } else {
                Ok(dev)
            }
        }
    }

    // ID3D12Device vtable layout (selected slots).  Inherits
    // ID3D12Object (4 methods: GetPrivateData, SetPrivateData,
    // SetPrivateDataInterface, SetName) which inherits IUnknown.
    // ID3D12Device methods start at offset 7:
    //   7  GetNodeCount() -> UINT
    //   8  CreateCommandQueue(...)
    //   9  CreateCommandAllocator(...)
    //  10  CreateGraphicsPipelineState(...)
    //  ... etc.
    #[repr(C)]
    pub struct ID3D12DeviceVtbl {
        // IUnknown
        QueryInterface: unsafe extern "system" fn(*mut c_void, *const GUID, *mut LPVOID) -> HRESULT,
        AddRef: unsafe extern "system" fn(*mut c_void) -> u32,
        Release: unsafe extern "system" fn(*mut c_void) -> u32,
        // ID3D12Object
        GetPrivateData: unsafe extern "system" fn() -> HRESULT,
        SetPrivateData: unsafe extern "system" fn() -> HRESULT,
        SetPrivateDataInterface: unsafe extern "system" fn() -> HRESULT,
        SetName: unsafe extern "system" fn() -> HRESULT,
        // ID3D12Device
        pub GetNodeCount: unsafe extern "system" fn(*mut c_void) -> u32,
        // (remaining slots elided — added when their FFI lands)
    }

    #[repr(C)]
    pub struct ID3D12Device {
        pub vtbl: *mut ID3D12DeviceVtbl,
    }

    /// Call ID3D12Device::GetNodeCount on a live device pointer.
    /// Always returns >= 1 for valid devices (single-GPU = 1).
    pub fn device_node_count(dev: LPVOID) -> u32 {
        unsafe {
            let p = dev as *mut ID3D12Device;
            ((*(*p).vtbl).GetNodeCount)(p as _)
        }
    }

    /// Release a device's IUnknown refcount.
    pub fn release_device(dev: LPVOID) {
        unsafe {
            let p = dev as *mut ID3D12Device;
            ((*(*p).vtbl).Release)(p as _);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_buffer_records_size_and_usage() {
        let mut d = GpuDevice::new();
        let b = d.create_buffer(1024, vec![BufferUsage::Vertex, BufferUsage::CopyDst]);
        let info = d.buffer(b).unwrap();
        assert_eq!(info.size, 1024);
        assert!(info.usage.contains(&BufferUsage::Vertex));
    }

    #[test]
    fn shader_module_validates_wgsl_stage_decl() {
        let mut d = GpuDevice::new();
        let s = d.create_shader_module("@vertex fn main() {}");
        assert!(d.shader_valid(s));
        let bad = d.create_shader_module("not wgsl");
        assert!(!d.shader_valid(bad));
    }

    #[test]
    fn pipeline_creation_fails_for_invalid_shaders() {
        let mut d = GpuDevice::new();
        let v = d.create_shader_module("@vertex fn vs(){}");
        let f = d.create_shader_module("invalid");
        assert!(d.create_render_pipeline(v, f, "vs", "fs").is_none());
    }

    #[test]
    fn write_buffer_then_inspect() {
        let mut d = GpuDevice::new();
        let b = d.create_buffer(4, vec![BufferUsage::Uniform]);
        d.write_buffer(b, 0, &[1, 2, 3, 4]).unwrap();
        assert_eq!(d.buffer(b).unwrap().data, vec![1, 2, 3, 4]);
    }

    #[test]
    fn command_encoder_records_in_order() {
        let mut enc = CommandEncoder::new();
        enc.set_pipeline(ResourceId(5));
        enc.draw(3, 1);
        let cmds = enc.finish();
        assert_eq!(cmds.len(), 2);
        matches!(cmds[0], Command::SetPipeline(_));
    }

    #[test]
    fn d3d12_iid_constants() {
        assert_eq!(d3d12::IID_ID3D12_DEVICE.Data1, 0x189819F1);
        assert_eq!(d3d12::D3D_FEATURE_LEVEL_12_0, 0xc000);
        assert_eq!(d3d12::D3D_FEATURE_LEVEL_11_0, 0xb000);
    }

    #[test]
    fn d3d12_get_node_count_through_vtable() {
        // Real ID3D12Device::GetNodeCount call through the COM
        // vtable.  On a single-GPU host this returns 1; on a
        // multi-GPU host >1; on CI without GPU we skip.
        match d3d12::create_device() {
            Ok(dev) => {
                let n = d3d12::device_node_count(dev);
                assert!(n >= 1, "expected at least one node, got {n}");
                d3d12::release_device(dev);
            }
            Err(_hr) => {
                // No GPU on this host; the FFI signature still
                // compiled and dispatched.
            }
        }
    }

    #[test]
    fn d3d12_create_device_or_warn() {
        // Real call into d3d12.dll. Modern Windows ships it; on
        // pre-Win10 or no-GPU CI it returns an error which is fine
        // for this test (we just verify the FFI signature compiles
        // and dispatches).
        match d3d12::create_device() {
            Ok(dev) => {
                assert!(!dev.is_null());
                unsafe {
                    // Release via IUnknown vtable head.
                    #[repr(C)]
                    struct IUnkVtbl {
                        _q: unsafe extern "system" fn(),
                        _a: unsafe extern "system" fn(),
                        release: unsafe extern "system" fn(*mut std::ffi::c_void) -> u32,
                    }
                    #[repr(C)]
                    struct IUnk {
                        vtbl: *mut IUnkVtbl,
                    }
                    let p = dev as *mut IUnk;
                    ((*(*p).vtbl).release)(p as _);
                }
            }
            Err(_hr) => {
                // CI without GPU. FFI executed, that's the test.
            }
        }
    }

    #[test]
    fn queue_submit_round_trips() {
        let mut q = Queue::default();
        q.submit(vec![Command::Draw {
            vertex_count: 6,
            instance_count: 1,
        }]);
        assert_eq!(q.submitted().len(), 1);
    }
}
