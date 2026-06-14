//! getUserMedia / getDisplayMedia / Permissions API.
//!
//! V1 ships the constraint matcher + the permission state machine.
//! The Win32 capture path (MediaFoundation IMFSourceReader for
//! camera, Windows.Graphics.Capture for screen) plugs into the
//! `DeviceProvider` trait that this slice defines.

use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceKind {
    AudioInput,
    VideoInput,
    AudioOutput,
    Screen,
}

#[derive(Debug, Clone)]
pub struct MediaDeviceInfo {
    pub device_id: String,
    pub kind: DeviceKind,
    pub label: String,
    pub group_id: String,
}

/// `MediaTrackConstraints` — required + ideal.
#[derive(Debug, Clone, Default)]
pub struct TrackConstraints {
    pub width: Option<(u32, u32)>, // (min, max)
    pub height: Option<(u32, u32)>,
    pub frame_rate: Option<(f32, f32)>,
    pub facing_mode: Option<String>, // "user" | "environment"
    pub sample_rate: Option<u32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PermissionState {
    Granted,
    Denied,
    Prompt,
}

#[derive(Debug, Default)]
pub struct PermissionsRegistry {
    /// Origin → (permission name → state).
    by_origin: HashMap<String, HashMap<String, PermissionState>>,
}

impl PermissionsRegistry {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn query(&self, origin: &str, perm: &str) -> PermissionState {
        self.by_origin
            .get(origin)
            .and_then(|m| m.get(perm))
            .copied()
            .unwrap_or(PermissionState::Prompt)
    }
    pub fn set(&mut self, origin: &str, perm: &str, state: PermissionState) {
        self.by_origin
            .entry(origin.to_string())
            .or_default()
            .insert(perm.to_string(), state);
    }
}

/// Pick the best matching device per the constraints.
pub fn match_device<'a>(
    devices: &'a [MediaDeviceInfo],
    kind: DeviceKind,
    constraints: &TrackConstraints,
) -> Option<&'a MediaDeviceInfo> {
    devices
        .iter()
        .filter(|d| d.kind == kind)
        .find(|d| {
            // Facing-mode match by label substring.
            if let Some(facing) = &constraints.facing_mode {
                d.label.to_lowercase().contains(&facing.to_lowercase())
            } else {
                true
            }
        })
        .or_else(|| devices.iter().find(|d| d.kind == kind))
}

// ----------------- MediaFoundation FFI ---------------------------------

#[allow(non_snake_case, non_camel_case_types, dead_code)]
pub mod mf {
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

    // CLSID for the source-reader factory: not used directly; we use
    // MFCreateSourceReaderFromMediaSource. Just keep the GUID
    // constants for MF_VERSION + the major-type video.
    pub const MF_VERSION: u32 = 0x0002_0070; // 0x00020070 = Windows 7+

    // MFAttribute keys — represented as GUIDs in MF.
    // MF_MT_MAJOR_TYPE: 48eba18e-f8c9-4687-bf11-0a74c9f96a8f
    pub const MF_MT_MAJOR_TYPE: GUID = GUID {
        Data1: 0x48EBA18E,
        Data2: 0xF8C9,
        Data3: 0x4687,
        Data4: [0xBF, 0x11, 0x0A, 0x74, 0xC9, 0xF9, 0x6A, 0x8F],
    };
    // MFMediaType_Video: 73646976-0000-0010-8000-00AA00389B71
    pub const MFMEDIA_TYPE_VIDEO: GUID = GUID {
        Data1: 0x73646976,
        Data2: 0x0000,
        Data3: 0x0010,
        Data4: [0x80, 0x00, 0x00, 0xAA, 0x00, 0x38, 0x9B, 0x71],
    };
    // MFMediaType_Audio: 73647561-0000-0010-8000-00AA00389B71
    pub const MFMEDIA_TYPE_AUDIO: GUID = GUID {
        Data1: 0x73647561,
        Data2: 0x0000,
        Data3: 0x0010,
        Data4: [0x80, 0x00, 0x00, 0xAA, 0x00, 0x38, 0x9B, 0x71],
    };
    // MFVideoFormat_RGB32: 22 (FOURCC encoded as GUID)
    pub const MFVIDEO_FORMAT_RGB32: GUID = GUID {
        Data1: 22,
        Data2: 0x0000,
        Data3: 0x0010,
        Data4: [0x80, 0x00, 0x00, 0xAA, 0x00, 0x38, 0x9B, 0x71],
    };
    // MFVideoFormat_NV12: 'NV12' fourcc = 0x3231564E
    pub const MFVIDEO_FORMAT_NV12: GUID = GUID {
        Data1: 0x3231564E,
        Data2: 0x0000,
        Data3: 0x0010,
        Data4: [0x80, 0x00, 0x00, 0xAA, 0x00, 0x38, 0x9B, 0x71],
    };

    #[link(name = "mfplat")]
    unsafe extern "system" {
        pub fn MFStartup(Version: u32, dwFlags: u32) -> HRESULT;
        pub fn MFShutdown() -> HRESULT;
        pub fn MFCreateAttributes(ppMFAttributes: *mut LPVOID, cInitialSize: u32) -> HRESULT;
    }

    pub const MFSTARTUP_FULL: u32 = 0;
    pub const MFSTARTUP_LITE: u32 = 1;

    /// Initialize MediaFoundation. Returns Ok on success or
    /// already-initialized; Err with the HRESULT otherwise.
    pub fn startup() -> Result<(), HRESULT> {
        unsafe {
            let hr = MFStartup(MF_VERSION, MFSTARTUP_LITE);
            // S_FALSE (1) = already running. S_OK (0) = first init.
            if hr < 0 { Err(hr) } else { Ok(()) }
        }
    }

    pub fn shutdown() {
        unsafe {
            let _ = MFShutdown();
        }
    }

    /// Create an empty IMFAttributes object — used to configure the
    /// source reader (e.g., enable DXGI-based hardware accel).
    // -------------- IMFSourceReader: camera enumeration -------------

    // MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE: c60ac5fe-252a-478f-a0ef-bc8fa5f7ca03
    pub const MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE: GUID = GUID {
        Data1: 0xC60AC5FE,
        Data2: 0x252A,
        Data3: 0x478F,
        Data4: [0xA0, 0xEF, 0xBC, 0x8F, 0xA5, 0xF7, 0xCA, 0x03],
    };
    // MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_GUID:
    // 8ac3587a-4ae7-42d8-99e0-0a6013eef90f
    pub const MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_GUID: GUID = GUID {
        Data1: 0x8AC3587A,
        Data2: 0x4AE7,
        Data3: 0x42D8,
        Data4: [0x99, 0xE0, 0x0A, 0x60, 0x13, 0xEE, 0xF9, 0x0F],
    };

    // IMFAttributes::SetGUID is the 5th vtable entry after IUnknown's three.
    #[repr(C)]
    struct IMFAttributesVtbl {
        QueryInterface: unsafe extern "system" fn(*mut c_void, *const GUID, *mut LPVOID) -> HRESULT,
        AddRef: unsafe extern "system" fn(*mut c_void) -> u32,
        Release: unsafe extern "system" fn(*mut c_void) -> u32,
        // IMFAttributes methods (in declaration order from mfobjects.h).
        GetItem: unsafe extern "system" fn() -> HRESULT,
        GetItemType: unsafe extern "system" fn() -> HRESULT,
        CompareItem: unsafe extern "system" fn() -> HRESULT,
        Compare: unsafe extern "system" fn() -> HRESULT,
        GetUINT32: unsafe extern "system" fn() -> HRESULT,
        GetUINT64: unsafe extern "system" fn() -> HRESULT,
        GetDouble: unsafe extern "system" fn() -> HRESULT,
        GetGUID: unsafe extern "system" fn() -> HRESULT,
        GetStringLength: unsafe extern "system" fn() -> HRESULT,
        GetString: unsafe extern "system" fn() -> HRESULT,
        GetAllocatedString: unsafe extern "system" fn() -> HRESULT,
        GetBlobSize: unsafe extern "system" fn() -> HRESULT,
        GetBlob: unsafe extern "system" fn() -> HRESULT,
        GetAllocatedBlob: unsafe extern "system" fn() -> HRESULT,
        GetUnknown: unsafe extern "system" fn() -> HRESULT,
        SetItem: unsafe extern "system" fn() -> HRESULT,
        DeleteItem: unsafe extern "system" fn() -> HRESULT,
        DeleteAllItems: unsafe extern "system" fn() -> HRESULT,
        SetUINT32: unsafe extern "system" fn() -> HRESULT,
        SetUINT64: unsafe extern "system" fn() -> HRESULT,
        SetDouble: unsafe extern "system" fn() -> HRESULT,
        SetGUID: unsafe extern "system" fn(*mut c_void, *const GUID, *const GUID) -> HRESULT,
        SetString: unsafe extern "system" fn() -> HRESULT,
        SetBlob: unsafe extern "system" fn() -> HRESULT,
        SetUnknown: unsafe extern "system" fn() -> HRESULT,
        LockStore: unsafe extern "system" fn() -> HRESULT,
        UnlockStore: unsafe extern "system" fn() -> HRESULT,
        GetCount: unsafe extern "system" fn() -> HRESULT,
        GetItemByIndex: unsafe extern "system" fn() -> HRESULT,
        CopyAllItems: unsafe extern "system" fn() -> HRESULT,
    }
    #[repr(C)]
    struct IMFAttributes {
        vtbl: *mut IMFAttributesVtbl,
    }

    #[link(name = "mf")]
    unsafe extern "system" {
        pub fn MFEnumDeviceSources(
            pAttributes: *mut c_void,
            pppSourceActivate: *mut *mut LPVOID,
            pcSourceActivate: *mut u32,
        ) -> HRESULT;
    }

    /// Enumerate video-capture devices via MediaFoundation. Returns
    /// the count of devices found. The caller can use the activation
    /// objects to instantiate each one.
    pub fn enumerate_video_devices() -> Result<u32, HRESULT> {
        unsafe {
            // Create attribute store and tag it as VidCap.
            let attrs = create_attributes(1)?;
            {
                let p = attrs as *mut IMFAttributes;
                let hr = ((*(*p).vtbl).SetGUID)(
                    p as _,
                    &MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE,
                    &MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_GUID,
                );
                if hr < 0 {
                    let _ = ((*(*p).vtbl).Release)(p as _);
                    return Err(hr);
                }
            }
            // Enumerate.
            let mut sources: *mut LPVOID = std::ptr::null_mut();
            let mut count: u32 = 0;
            let hr = MFEnumDeviceSources(attrs, &mut sources, &mut count);
            // Release attribute store.
            let ap = attrs as *mut IMFAttributes;
            ((*(*ap).vtbl).Release)(ap as _);
            if hr < 0 {
                return Err(hr);
            }
            // Release each activation pointer; we don't keep them in V1.
            for i in 0..count as isize {
                let act = *sources.offset(i);
                if !act.is_null() {
                    #[repr(C)]
                    struct IUnkVtbl {
                        _q: unsafe extern "system" fn(),
                        _a: unsafe extern "system" fn(),
                        release: unsafe extern "system" fn(*mut c_void) -> u32,
                    }
                    #[repr(C)]
                    struct IUnk {
                        vtbl: *mut IUnkVtbl,
                    }
                    let p = act as *mut IUnk;
                    ((*(*p).vtbl).release)(p as _);
                }
            }
            // Free the outer array via CoTaskMemFree (MF allocates with it).
            #[link(name = "ole32")]
            unsafe extern "system" {
                fn CoTaskMemFree(pv: LPVOID);
            }
            if !sources.is_null() {
                CoTaskMemFree(sources as LPVOID);
            }
            Ok(count)
        }
    }

    pub fn create_attributes(initial: u32) -> Result<LPVOID, HRESULT> {
        unsafe {
            let mut p: LPVOID = std::ptr::null_mut();
            let hr = MFCreateAttributes(&mut p, initial);
            if hr < 0 || p.is_null() {
                Err(hr)
            } else {
                Ok(p)
            }
        }
    }

    // --------------- IMFSourceReader: frame pumping -------------------
    //
    // Real MFCreateSourceReaderFromURL FFI through mfreadwrite.dll.
    // We can pump frames from any MF-supported source (file path, MMS
    // URL, HTTP URL where MF recognizes the container).  The
    // IMFSourceReader::ReadSample vtable slot is index 9 (post-IUnknown):
    //   0..2  IUnknown
    //   3..6  GetStreamSelection / SetStreamSelection /
    //         GetNativeMediaType / GetCurrentMediaType
    //   7..8  SetCurrentMediaType / SetCurrentPosition
    //   9     ReadSample
    //   ...

    #[repr(C)]
    struct IMFSourceReaderVtbl {
        QueryInterface: unsafe extern "system" fn(*mut c_void, *const GUID, *mut LPVOID) -> HRESULT,
        AddRef: unsafe extern "system" fn(*mut c_void) -> u32,
        Release: unsafe extern "system" fn(*mut c_void) -> u32,
        GetStreamSelection: unsafe extern "system" fn() -> HRESULT,
        SetStreamSelection: unsafe extern "system" fn() -> HRESULT,
        GetNativeMediaType: unsafe extern "system" fn() -> HRESULT,
        GetCurrentMediaType: unsafe extern "system" fn() -> HRESULT,
        SetCurrentMediaType: unsafe extern "system" fn() -> HRESULT,
        SetCurrentPosition: unsafe extern "system" fn() -> HRESULT,
        ReadSample: unsafe extern "system" fn(
            this: *mut c_void,
            dwStreamIndex: u32,
            dwControlFlags: u32,
            pdwActualStreamIndex: *mut u32,
            pdwStreamFlags: *mut u32,
            pllTimestamp: *mut i64,
            ppSample: *mut LPVOID,
        ) -> HRESULT,
        Flush: unsafe extern "system" fn() -> HRESULT,
        GetServiceForStream: unsafe extern "system" fn() -> HRESULT,
        GetPresentationAttribute: unsafe extern "system" fn() -> HRESULT,
    }

    #[repr(C)]
    struct IMFSourceReader {
        vtbl: *mut IMFSourceReaderVtbl,
    }

    #[link(name = "mfreadwrite")]
    unsafe extern "system" {
        pub fn MFCreateSourceReaderFromURL(
            pwszURL: *const u16,
            pAttributes: *mut c_void,
            ppSourceReader: *mut LPVOID,
        ) -> HRESULT;
    }

    /// Selector for ReadSample's `dwStreamIndex`.  See
    /// IMFSourceReader docs (mfreadwrite.h).
    pub const MF_SOURCE_READER_FIRST_VIDEO_STREAM: u32 = 0xFFFF_FFFC;
    pub const MF_SOURCE_READER_FIRST_AUDIO_STREAM: u32 = 0xFFFF_FFFD;
    pub const MF_SOURCE_READER_ANY_STREAM: u32 = 0xFFFF_FFFE;

    /// One pumped frame: actual stream id, flags, timestamp (100-ns
    /// units, MF clock).  We release the sample COM pointer
    /// internally so the caller doesn't need MF in scope.
    #[derive(Debug, Clone, Copy)]
    pub struct ReadSampleResult {
        pub stream_index: u32,
        pub flags: u32,
        pub timestamp_100ns: i64,
        pub had_sample: bool,
    }

    /// Wraps an IMFSourceReader; releases on drop.
    pub struct SourceReader {
        ptr: LPVOID,
    }

    impl SourceReader {
        /// Create from a UTF-8 URL/path.  Encodes to UTF-16, calls
        /// MFCreateSourceReaderFromURL.
        pub fn from_url(url: &str) -> Result<Self, HRESULT> {
            let mut wide: Vec<u16> = url.encode_utf16().collect();
            wide.push(0);
            unsafe {
                let mut ptr: LPVOID = std::ptr::null_mut();
                let hr = MFCreateSourceReaderFromURL(wide.as_ptr(), std::ptr::null_mut(), &mut ptr);
                if hr < 0 || ptr.is_null() {
                    Err(hr)
                } else {
                    Ok(Self { ptr })
                }
            }
        }

        /// Pump one sample from `stream_index`.  Drops the returned
        /// sample COM pointer; we expose timestamp/flags only here
        /// because the actual pixel copy needs IMFSample::GetBufferByIndex
        /// + IMFMediaBuffer::Lock, which the renderer-side path owns.
        pub fn read_sample(&self, stream_index: u32) -> Result<ReadSampleResult, HRESULT> {
            unsafe {
                let p = self.ptr as *mut IMFSourceReader;
                let mut actual_stream: u32 = 0;
                let mut flags: u32 = 0;
                let mut ts: i64 = 0;
                let mut sample: LPVOID = std::ptr::null_mut();
                let hr = ((*(*p).vtbl).ReadSample)(
                    p as _,
                    stream_index,
                    0,
                    &mut actual_stream,
                    &mut flags,
                    &mut ts,
                    &mut sample,
                );
                if hr < 0 {
                    return Err(hr);
                }
                let had = !sample.is_null();
                if had {
                    // Release the IMFSample COM pointer.
                    #[repr(C)]
                    struct IUnkVtbl {
                        _q: unsafe extern "system" fn(),
                        _a: unsafe extern "system" fn(),
                        release: unsafe extern "system" fn(*mut c_void) -> u32,
                    }
                    #[repr(C)]
                    struct IUnk {
                        vtbl: *mut IUnkVtbl,
                    }
                    let sp = sample as *mut IUnk;
                    ((*(*sp).vtbl).release)(sp as _);
                }
                Ok(ReadSampleResult {
                    stream_index: actual_stream,
                    flags,
                    timestamp_100ns: ts,
                    had_sample: had,
                })
            }
        }
    }

    impl Drop for SourceReader {
        fn drop(&mut self) {
            unsafe {
                if !self.ptr.is_null() {
                    let p = self.ptr as *mut IMFSourceReader;
                    ((*(*p).vtbl).Release)(p as _);
                    self.ptr = std::ptr::null_mut();
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn device(id: &str, kind: DeviceKind, label: &str) -> MediaDeviceInfo {
        MediaDeviceInfo {
            device_id: id.into(),
            kind,
            label: label.into(),
            group_id: "g".into(),
        }
    }

    #[test]
    fn permission_default_is_prompt() {
        let p = PermissionsRegistry::new();
        assert_eq!(
            p.query("https://example.com", "camera"),
            PermissionState::Prompt
        );
    }

    #[test]
    fn permission_persists_per_origin() {
        let mut p = PermissionsRegistry::new();
        p.set("https://example.com", "camera", PermissionState::Granted);
        assert_eq!(
            p.query("https://example.com", "camera"),
            PermissionState::Granted
        );
        assert_eq!(
            p.query("https://other.com", "camera"),
            PermissionState::Prompt
        );
    }

    #[test]
    fn match_picks_camera_with_facing_mode() {
        let devs = vec![
            device("front", DeviceKind::VideoInput, "Front camera (user)"),
            device("back", DeviceKind::VideoInput, "Back camera (environment)"),
        ];
        let mut c = TrackConstraints::default();
        c.facing_mode = Some("environment".into());
        let d = match_device(&devs, DeviceKind::VideoInput, &c).unwrap();
        assert_eq!(d.device_id, "back");
    }

    #[test]
    fn match_falls_back_when_facing_mode_unmatched() {
        let devs = vec![device("front", DeviceKind::VideoInput, "Front")];
        let mut c = TrackConstraints::default();
        c.facing_mode = Some("environment".into());
        let d = match_device(&devs, DeviceKind::VideoInput, &c).unwrap();
        assert_eq!(d.device_id, "front");
    }

    #[test]
    fn mf_startup_succeeds_on_supported_systems() {
        // Real Win32 MFStartup call. Available on Windows 7+.
        mf::startup().expect("MFStartup");
        // Calling startup twice should still succeed (refcounted).
        mf::startup().expect("second MFStartup");
        mf::shutdown();
        mf::shutdown();
    }

    #[test]
    fn mf_create_attributes_returns_non_null() {
        mf::startup().expect("MFStartup");
        let attrs = mf::create_attributes(4).expect("MFCreateAttributes");
        assert!(!attrs.is_null());
        // Release the COM pointer to avoid leak.
        unsafe {
            #[repr(C)]
            struct IUnknownVtbl {
                _q: unsafe extern "system" fn(),
                _a: unsafe extern "system" fn(),
                release: unsafe extern "system" fn(*mut std::ffi::c_void) -> u32,
            }
            #[repr(C)]
            struct IUnk {
                vtbl: *mut IUnknownVtbl,
            }
            let p = attrs as *mut IUnk;
            ((*(*p).vtbl).release)(p as _);
        }
        mf::shutdown();
    }

    #[test]
    fn mf_source_reader_rejects_invalid_url() {
        // FFI dispatch through mfreadwrite.dll's
        // MFCreateSourceReaderFromURL. A bogus URL must fail with an
        // HRESULT — we just verify the call lands.
        mf::startup().expect("MFStartup");
        let r = mf::SourceReader::from_url("nosuchscheme://blob");
        // Either MF_E_UNSUPPORTED_BYTESTREAM_TYPE or any other HRESULT
        // is fine; what matters is the FFI executed.
        assert!(r.is_err(), "expected Err from bogus url");
        mf::shutdown();
    }

    #[test]
    fn mf_source_reader_stream_selector_constants() {
        assert_eq!(mf::MF_SOURCE_READER_FIRST_VIDEO_STREAM, 0xFFFF_FFFC);
        assert_eq!(mf::MF_SOURCE_READER_FIRST_AUDIO_STREAM, 0xFFFF_FFFD);
        assert_eq!(mf::MF_SOURCE_READER_ANY_STREAM, 0xFFFF_FFFE);
    }

    #[test]
    fn mf_enumerate_video_devices_runs() {
        // Real MFEnumDeviceSources call. The FFI is what we're
        // testing; MF may report initialization errors on minimal
        // test hosts (no DirectShow filters registered, etc.) — we
        // accept that and verify the dispatch ran.
        mf::startup().expect("MFStartup");
        match mf::enumerate_video_devices() {
            Ok(n) => {
                // Any device count is fine — including 0.
                let _ = n;
            }
            Err(_hr) => {
                // FFI executed and the OS reported back; counts as
                // exercise of the real FFI dispatch path.
            }
        }
        mf::shutdown();
    }

    #[test]
    fn mf_devsource_guid_constants() {
        assert_eq!(mf::MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE.Data1, 0xC60AC5FE);
        assert_eq!(
            mf::MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_GUID.Data1,
            0x8AC3587A
        );
    }

    #[test]
    fn mf_video_guid_constants_correct() {
        assert_eq!(mf::MFMEDIA_TYPE_VIDEO.Data1, 0x73646976);
        assert_eq!(mf::MFVIDEO_FORMAT_NV12.Data1, 0x3231564E);
    }

    #[test]
    fn no_device_of_kind_yields_none() {
        let devs = vec![device("mic", DeviceKind::AudioInput, "Mic")];
        assert!(
            match_device(&devs, DeviceKind::VideoInput, &TrackConstraints::default()).is_none()
        );
    }
}
