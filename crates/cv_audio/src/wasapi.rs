//! WASAPI shared-mode output — real Win32 COM bindings.
//!
//! Calls into mmdevapi.dll + ole32.dll. We define the COM interface
//! vtables directly (no third-party `windows` crate) and dispatch
//! through them.

#![allow(non_snake_case, non_camel_case_types, dead_code)]

use std::sync::Mutex;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamCategory {
    Media,
    Communication,
    GameEffects,
}

#[derive(Debug, Clone)]
pub struct OutputFormat {
    pub sample_rate: u32,
    pub channels: u8,
    pub bits_per_sample: u8,
}

impl Default for OutputFormat {
    fn default() -> Self {
        Self {
            sample_rate: 48_000,
            channels: 2,
            bits_per_sample: 32,
        }
    }
}

#[derive(Debug)]
pub struct OutputStream {
    format: OutputFormat,
    category: StreamCategory,
    buffer: Mutex<Vec<f32>>,
    underrun_count: std::sync::atomic::AtomicU64,
}

impl OutputStream {
    pub fn new(format: OutputFormat, category: StreamCategory) -> Self {
        Self {
            format,
            category,
            buffer: Mutex::new(Vec::new()),
            underrun_count: std::sync::atomic::AtomicU64::new(0),
        }
    }
    pub fn format(&self) -> &OutputFormat {
        &self.format
    }
    pub fn category(&self) -> StreamCategory {
        self.category
    }
    pub fn push_pcm(&self, samples: &[f32]) {
        self.buffer.lock().unwrap().extend_from_slice(samples);
    }
    pub fn pull_pcm_for_device(&self, frames: usize) -> Vec<f32> {
        let needed = frames * (self.format.channels as usize);
        let mut buf = self.buffer.lock().unwrap();
        if buf.len() >= needed {
            buf.drain(..needed).collect()
        } else {
            self.underrun_count
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let mut out = Vec::with_capacity(needed);
            out.extend(buf.drain(..));
            out.resize(needed, 0.0);
            out
        }
    }
    pub fn queued_frames(&self) -> usize {
        self.buffer.lock().unwrap().len() / (self.format.channels as usize)
    }
    pub fn underrun_count(&self) -> u64 {
        self.underrun_count
            .load(std::sync::atomic::Ordering::Relaxed)
    }
}

pub fn mix_streams(streams: &[OutputStream], frames: usize) -> Vec<f32> {
    if streams.is_empty() {
        return vec![0.0; frames * 2];
    }
    let channels = streams[0].format.channels as usize;
    let mut out = vec![0.0f32; frames * channels];
    for s in streams {
        let pulled = s.pull_pcm_for_device(frames);
        for (i, v) in pulled.iter().enumerate() {
            out[i] += v;
        }
    }
    for v in out.iter_mut() {
        *v = v.clamp(-1.0, 1.0);
    }
    out
}

// ----------------------------------------------------------------------
// Win32 COM types
// ----------------------------------------------------------------------

type HRESULT = i32;
type DWORD = u32;
type REFCLSID = *const GUID;
type REFIID = *const GUID;
type LPVOID = *mut std::ffi::c_void;
type HANDLE = *mut std::ffi::c_void;
type REFERENCE_TIME = i64;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct GUID {
    pub Data1: u32,
    pub Data2: u16,
    pub Data3: u16,
    pub Data4: [u8; 8],
}

// CLSID_MMDeviceEnumerator: BCDE0395-E52F-467C-8E3D-C4579291692E
pub const CLSID_MMDEVICE_ENUMERATOR: GUID = GUID {
    Data1: 0xBCDE0395,
    Data2: 0xE52F,
    Data3: 0x467C,
    Data4: [0x8E, 0x3D, 0xC4, 0x57, 0x92, 0x91, 0x69, 0x2E],
};
// IID_IMMDeviceEnumerator: A95664D2-9614-4F35-A746-DE8DB63617E6
pub const IID_IMMDEVICE_ENUMERATOR: GUID = GUID {
    Data1: 0xA95664D2,
    Data2: 0x9614,
    Data3: 0x4F35,
    Data4: [0xA7, 0x46, 0xDE, 0x8D, 0xB6, 0x36, 0x17, 0xE6],
};
// IID_IAudioClient: 1CB9AD4C-DBFA-4C32-B178-C2F568A703B2
pub const IID_IAUDIO_CLIENT: GUID = GUID {
    Data1: 0x1CB9AD4C,
    Data2: 0xDBFA,
    Data3: 0x4C32,
    Data4: [0xB1, 0x78, 0xC2, 0xF5, 0x68, 0xA7, 0x03, 0xB2],
};
// IID_IAudioRenderClient: F294ACFC-3146-4483-A7BF-ADDCA7C260E2
pub const IID_IAUDIO_RENDER_CLIENT: GUID = GUID {
    Data1: 0xF294ACFC,
    Data2: 0x3146,
    Data3: 0x4483,
    Data4: [0xA7, 0xBF, 0xAD, 0xDC, 0xA7, 0xC2, 0x60, 0xE2],
};

const COINIT_MULTITHREADED: u32 = 0x0;
const CLSCTX_INPROC_SERVER: u32 = 0x1;
pub const AUDCLNT_SHAREMODE_SHARED: u32 = 0;
pub const AUDCLNT_STREAMFLAGS_EVENTCALLBACK: u32 = 0x00040000;
pub const eRender: u32 = 0;
pub const eConsole: u32 = 0;

#[link(name = "ole32")]
unsafe extern "system" {
    fn CoInitializeEx(reserved: LPVOID, dwCoInit: u32) -> HRESULT;
    fn CoCreateInstance(
        rclsid: REFCLSID,
        pUnkOuter: LPVOID,
        dwClsContext: u32,
        riid: REFIID,
        ppv: *mut LPVOID,
    ) -> HRESULT;
    fn CoTaskMemFree(pv: LPVOID);
}

// Minimal IMMDeviceEnumerator vtable: we only need GetDefaultAudioEndpoint.
#[repr(C)]
struct IMMDeviceEnumeratorVtbl {
    QueryInterface:
        unsafe extern "system" fn(*mut std::ffi::c_void, REFIID, *mut LPVOID) -> HRESULT,
    AddRef: unsafe extern "system" fn(*mut std::ffi::c_void) -> u32,
    Release: unsafe extern "system" fn(*mut std::ffi::c_void) -> u32,
    EnumAudioEndpoints:
        unsafe extern "system" fn(*mut std::ffi::c_void, u32, u32, *mut LPVOID) -> HRESULT,
    GetDefaultAudioEndpoint:
        unsafe extern "system" fn(*mut std::ffi::c_void, u32, u32, *mut LPVOID) -> HRESULT,
}
#[repr(C)]
struct IMMDeviceEnumerator {
    vtbl: *mut IMMDeviceEnumeratorVtbl,
}

#[repr(C)]
struct IMMDeviceVtbl {
    QueryInterface:
        unsafe extern "system" fn(*mut std::ffi::c_void, REFIID, *mut LPVOID) -> HRESULT,
    AddRef: unsafe extern "system" fn(*mut std::ffi::c_void) -> u32,
    Release: unsafe extern "system" fn(*mut std::ffi::c_void) -> u32,
    Activate: unsafe extern "system" fn(
        *mut std::ffi::c_void,
        REFIID,
        u32,
        LPVOID,
        *mut LPVOID,
    ) -> HRESULT,
}
#[repr(C)]
struct IMMDevice {
    vtbl: *mut IMMDeviceVtbl,
}

#[repr(C)]
pub struct WAVEFORMATEX {
    pub wFormatTag: u16,
    pub nChannels: u16,
    pub nSamplesPerSec: u32,
    pub nAvgBytesPerSec: u32,
    pub nBlockAlign: u16,
    pub wBitsPerSample: u16,
    pub cbSize: u16,
}

#[repr(C)]
struct IAudioClientVtbl {
    QueryInterface:
        unsafe extern "system" fn(*mut std::ffi::c_void, REFIID, *mut LPVOID) -> HRESULT,
    AddRef: unsafe extern "system" fn(*mut std::ffi::c_void) -> u32,
    Release: unsafe extern "system" fn(*mut std::ffi::c_void) -> u32,
    Initialize: unsafe extern "system" fn(
        *mut std::ffi::c_void,
        u32,
        u32,
        REFERENCE_TIME,
        REFERENCE_TIME,
        *const WAVEFORMATEX,
        *const GUID,
    ) -> HRESULT,
    GetBufferSize: unsafe extern "system" fn(*mut std::ffi::c_void, *mut u32) -> HRESULT,
    GetStreamLatency:
        unsafe extern "system" fn(*mut std::ffi::c_void, *mut REFERENCE_TIME) -> HRESULT,
    GetCurrentPadding: unsafe extern "system" fn(*mut std::ffi::c_void, *mut u32) -> HRESULT,
    IsFormatSupported: unsafe extern "system" fn(
        *mut std::ffi::c_void,
        u32,
        *const WAVEFORMATEX,
        *mut *mut WAVEFORMATEX,
    ) -> HRESULT,
    GetMixFormat:
        unsafe extern "system" fn(*mut std::ffi::c_void, *mut *mut WAVEFORMATEX) -> HRESULT,
    GetDevicePeriod: unsafe extern "system" fn(
        *mut std::ffi::c_void,
        *mut REFERENCE_TIME,
        *mut REFERENCE_TIME,
    ) -> HRESULT,
    Start: unsafe extern "system" fn(*mut std::ffi::c_void) -> HRESULT,
    Stop: unsafe extern "system" fn(*mut std::ffi::c_void) -> HRESULT,
    Reset: unsafe extern "system" fn(*mut std::ffi::c_void) -> HRESULT,
    SetEventHandle: unsafe extern "system" fn(*mut std::ffi::c_void, HANDLE) -> HRESULT,
    GetService: unsafe extern "system" fn(*mut std::ffi::c_void, REFIID, *mut LPVOID) -> HRESULT,
}
#[repr(C)]
struct IAudioClient {
    vtbl: *mut IAudioClientVtbl,
}

// IAudioRenderClient — GetBuffer/ReleaseBuffer let us hand decoded PCM
// straight into the shared-mode endpoint buffer (the real device push path).
#[repr(C)]
struct IAudioRenderClientVtbl {
    QueryInterface:
        unsafe extern "system" fn(*mut std::ffi::c_void, REFIID, *mut LPVOID) -> HRESULT,
    AddRef: unsafe extern "system" fn(*mut std::ffi::c_void) -> u32,
    Release: unsafe extern "system" fn(*mut std::ffi::c_void) -> u32,
    GetBuffer:
        unsafe extern "system" fn(*mut std::ffi::c_void, u32, *mut *mut u8) -> HRESULT,
    ReleaseBuffer: unsafe extern "system" fn(*mut std::ffi::c_void, u32, u32) -> HRESULT,
}
#[repr(C)]
struct IAudioRenderClient {
    vtbl: *mut IAudioRenderClientVtbl,
}

const AUDCLNT_BUFFERFLAGS_SILENT: u32 = 0x2;

/// Open the system default audio render endpoint and return its
/// preferred (mix-format) WAVEFORMATEX along with the IAudioClient
/// pointer for callers that want to push real PCM into the system.
pub struct WasapiDevice {
    /// Raw IAudioClient COM pointer. Owned — released on drop.
    audio_client: *mut IAudioClient,
    /// Raw IAudioRenderClient COM pointer (from `GetService`). Owned —
    /// released on drop. Used by [`WasapiDevice::write_frames`].
    render_client: *mut IAudioRenderClient,
    pub mix_format: OutputFormat,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WasapiError {
    CoInitFailed(i32),
    CoCreateInstanceFailed(i32),
    GetDefaultEndpointFailed(i32),
    ActivateFailed(i32),
    GetMixFormatFailed(i32),
    InitializeFailed(i32),
    GetServiceFailed(i32),
    StartFailed(i32),
}

impl WasapiDevice {
    /// Open the default render endpoint. Returns a pointer to a live
    /// IAudioClient + the mix format the system prefers.
    pub fn open_default() -> Result<Self, WasapiError> {
        unsafe {
            let hr = CoInitializeEx(std::ptr::null_mut(), COINIT_MULTITHREADED);
            // S_OK == 0; S_FALSE == 1 (already initialized) — both fine.
            if hr < 0 {
                return Err(WasapiError::CoInitFailed(hr));
            }
            // Create IMMDeviceEnumerator.
            let mut enumr: LPVOID = std::ptr::null_mut();
            let hr = CoCreateInstance(
                &CLSID_MMDEVICE_ENUMERATOR,
                std::ptr::null_mut(),
                CLSCTX_INPROC_SERVER,
                &IID_IMMDEVICE_ENUMERATOR,
                &mut enumr,
            );
            if hr < 0 || enumr.is_null() {
                return Err(WasapiError::CoCreateInstanceFailed(hr));
            }
            let enumr = enumr as *mut IMMDeviceEnumerator;
            // GetDefaultAudioEndpoint.
            let mut dev: LPVOID = std::ptr::null_mut();
            let hr =
                ((*(*enumr).vtbl).GetDefaultAudioEndpoint)(enumr as _, eRender, eConsole, &mut dev);
            ((*(*enumr).vtbl).Release)(enumr as _);
            if hr < 0 || dev.is_null() {
                return Err(WasapiError::GetDefaultEndpointFailed(hr));
            }
            let dev = dev as *mut IMMDevice;
            // Activate IAudioClient.
            let mut client: LPVOID = std::ptr::null_mut();
            let hr = ((*(*dev).vtbl).Activate)(
                dev as _,
                &IID_IAUDIO_CLIENT,
                CLSCTX_INPROC_SERVER,
                std::ptr::null_mut(),
                &mut client,
            );
            ((*(*dev).vtbl).Release)(dev as _);
            if hr < 0 || client.is_null() {
                return Err(WasapiError::ActivateFailed(hr));
            }
            let audio_client = client as *mut IAudioClient;
            // GetMixFormat.
            let mut wfx: *mut WAVEFORMATEX = std::ptr::null_mut();
            let hr = ((*(*audio_client).vtbl).GetMixFormat)(audio_client as _, &mut wfx);
            if hr < 0 || wfx.is_null() {
                ((*(*audio_client).vtbl).Release)(audio_client as _);
                return Err(WasapiError::GetMixFormatFailed(hr));
            }
            let mix_format = OutputFormat {
                sample_rate: (*wfx).nSamplesPerSec,
                channels: (*wfx).nChannels as u8,
                bits_per_sample: (*wfx).wBitsPerSample as u8,
            };
            // Initialize the client in shared mode at the mix format.
            let hns_buffer: REFERENCE_TIME = 10_000_000; // 1 second
            let hr = ((*(*audio_client).vtbl).Initialize)(
                audio_client as _,
                AUDCLNT_SHAREMODE_SHARED,
                0,
                hns_buffer,
                0,
                wfx,
                std::ptr::null(),
            );
            CoTaskMemFree(wfx as _);
            if hr < 0 {
                ((*(*audio_client).vtbl).Release)(audio_client as _);
                return Err(WasapiError::InitializeFailed(hr));
            }
            // Acquire the IAudioRenderClient — the interface we push PCM
            // through. Must happen after Initialize, before Start.
            let mut rc: LPVOID = std::ptr::null_mut();
            let hr = ((*(*audio_client).vtbl).GetService)(
                audio_client as _,
                &IID_IAUDIO_RENDER_CLIENT,
                &mut rc,
            );
            if hr < 0 || rc.is_null() {
                ((*(*audio_client).vtbl).Release)(audio_client as _);
                return Err(WasapiError::GetServiceFailed(hr));
            }
            let render_client = rc as *mut IAudioRenderClient;
            // Start the stream.
            let hr = ((*(*audio_client).vtbl).Start)(audio_client as _);
            if hr < 0 {
                ((*(*render_client).vtbl).Release)(render_client as _);
                ((*(*audio_client).vtbl).Release)(audio_client as _);
                return Err(WasapiError::StartFailed(hr));
            }
            Ok(Self {
                audio_client,
                render_client,
                mix_format,
            })
        }
    }

    /// Push interleaved `f32` PCM into the endpoint buffer. `samples` must
    /// be `frames * channels` long. Returns the number of frames actually
    /// written (limited by the free space in the shared-mode buffer). This
    /// is the REAL device push: `GetBuffer` → memcpy → `ReleaseBuffer`.
    ///
    /// Note: callers run this on the audio render thread, pacing by
    /// `buffer_size_frames() - current_padding_frames()` (the standard
    /// WASAPI shared-mode event loop).
    pub fn write_frames(&self, samples: &[f32]) -> u32 {
        let channels = self.mix_format.channels.max(1) as usize;
        let want_frames = (samples.len() / channels) as u32;
        if want_frames == 0 {
            return 0;
        }
        unsafe {
            let buffer_frames = self.buffer_size_frames();
            let padding = self.current_padding_frames();
            let free = buffer_frames.saturating_sub(padding);
            let frames = want_frames.min(free);
            if frames == 0 {
                return 0;
            }
            let mut ptr: *mut u8 = std::ptr::null_mut();
            let hr =
                ((*(*self.render_client).vtbl).GetBuffer)(self.render_client as _, frames, &mut ptr);
            if hr < 0 || ptr.is_null() {
                return 0;
            }
            // The mix format is float (we initialized at the device mix
            // format, which is virtually always 32-bit float in shared mode).
            // Copy `frames * channels` f32 samples.
            let n = (frames as usize) * channels;
            let dst = ptr as *mut f32;
            for i in 0..n {
                *dst.add(i) = samples.get(i).copied().unwrap_or(0.0);
            }
            ((*(*self.render_client).vtbl).ReleaseBuffer)(self.render_client as _, frames, 0);
            frames
        }
    }

    /// Write `frames` of silence into the endpoint (used to prime the
    /// buffer or recover from underrun without clicks).
    pub fn write_silence(&self, frames: u32) -> u32 {
        unsafe {
            let mut ptr: *mut u8 = std::ptr::null_mut();
            let hr =
                ((*(*self.render_client).vtbl).GetBuffer)(self.render_client as _, frames, &mut ptr);
            if hr < 0 || ptr.is_null() {
                return 0;
            }
            ((*(*self.render_client).vtbl).ReleaseBuffer)(
                self.render_client as _,
                frames,
                AUDCLNT_BUFFERFLAGS_SILENT,
            );
            frames
        }
    }

    pub fn buffer_size_frames(&self) -> u32 {
        let mut n: u32 = 0;
        unsafe {
            ((*(*self.audio_client).vtbl).GetBufferSize)(self.audio_client as _, &mut n);
        }
        n
    }

    pub fn current_padding_frames(&self) -> u32 {
        let mut n: u32 = 0;
        unsafe {
            ((*(*self.audio_client).vtbl).GetCurrentPadding)(self.audio_client as _, &mut n);
        }
        n
    }
}

impl Drop for WasapiDevice {
    fn drop(&mut self) {
        unsafe {
            if !self.render_client.is_null() {
                ((*(*self.render_client).vtbl).Release)(self.render_client as _);
            }
            if !self.audio_client.is_null() {
                let _ = ((*(*self.audio_client).vtbl).Stop)(self.audio_client as _);
                ((*(*self.audio_client).vtbl).Release)(self.audio_client as _);
            }
        }
    }
}

// SAFETY: COM pointer lifetime is owned by this struct; we never
// alias it across threads simultaneously. Marking Send so the
// rendering thread can take ownership; not Sync because we don't
// guard concurrent calls.
unsafe impl Send for WasapiDevice {}

#[cfg(test)]
mod tests {
    use super::*;

    fn stream() -> OutputStream {
        OutputStream::new(OutputFormat::default(), StreamCategory::Media)
    }

    #[test]
    fn push_pull_round_trips() {
        let s = stream();
        s.push_pcm(&[0.1, 0.2, 0.3, 0.4]);
        let out = s.pull_pcm_for_device(2);
        assert_eq!(out, vec![0.1, 0.2, 0.3, 0.4]);
        assert_eq!(s.queued_frames(), 0);
    }

    #[test]
    fn underflow_pads_and_counts() {
        let s = stream();
        s.push_pcm(&[0.5; 4]);
        let out = s.pull_pcm_for_device(4);
        assert_eq!(out.len(), 8);
        assert_eq!(out[4..8], [0.0; 4]);
        assert_eq!(s.underrun_count(), 1);
    }

    #[test]
    fn mixer_sums() {
        let a = stream();
        let b = stream();
        a.push_pcm(&[0.1, 0.1, 0.2, 0.2]);
        b.push_pcm(&[0.3, 0.3, 0.4, 0.4]);
        let mixed = mix_streams(&[a, b], 2);
        assert!((mixed[0] - 0.4).abs() < 1e-6);
        assert!((mixed[3] - 0.6).abs() < 1e-6);
    }

    #[test]
    fn mixer_clips_to_unit_range() {
        let a = stream();
        let b = stream();
        a.push_pcm(&[0.9; 2]);
        b.push_pcm(&[0.9; 2]);
        let mixed = mix_streams(&[a, b], 1);
        for v in mixed {
            assert!(v <= 1.0);
        }
    }

    #[test]
    fn queued_frames_tracks_pending() {
        let s = stream();
        s.push_pcm(&[0.0; 10]);
        assert_eq!(s.queued_frames(), 5);
    }
}
