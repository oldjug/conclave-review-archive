//! Live COM bridge: a real `IRawElementProviderSimple` /
//! `IRawElementProviderFragment` / `IRawElementProviderFragmentRoot` object that
//! the OS UI Automation runtime (and thus Narrator) calls into, served from a
//! captured [`crate::provider::PublishedTree`] snapshot.
//!
//! Windows COM object layout: a heap object whose first field is a pointer to a
//! static vtable of `extern "system"` function pointers laid out exactly per
//! UIAutomationCore's IDL (IUnknown's 3 methods, then the interface's methods in
//! declaration order). The OS receives an `*mut IRawElementProviderSimple` and
//! calls through `(*vtbl).Method(this, ...)`. We hand the OS our object pointer;
//! every method recovers `&Provider` from `this`.
//!
//! Threading: UIA marshals all calls to the object's apartment. We construct the
//! object on the UI thread inside `WM_GETOBJECT` and only ever touch it there,
//! so a non-atomic refcount + `Cell` interior mutability is sound. The snapshot
//! it serves is a `PublishedTree` clone taken at construction (immutable for the
//! object's lifetime), so there is no cross-thread aliasing of the renderer's
//! live tree.
//!
//! GATED: this module's entry point is only reached when `CV_A11Y_UIA=1`
//! (`crate::provider::a11y_uia_enabled`). It links UIAutomationCore.

#![allow(non_snake_case, non_camel_case_types)]
#![cfg(target_os = "windows")]

use crate::provider::PublishedTree;
use crate::uia::{GUID, NavigateDirection, UiaPropertyId, VariantValue};
use core::ffi::c_void;
use std::cell::Cell;

use crate::uia::{
    IID_IRAW_ELEMENT_PROVIDER_FRAGMENT, IID_IRAW_ELEMENT_PROVIDER_FRAGMENT_ROOT,
    IID_IRAW_ELEMENT_PROVIDER_SIMPLE, IID_IUNKNOWN, PROVIDER_OPTIONS_SERVER_SIDE_PROVIDER,
};

// HRESULTs.
const S_OK: i32 = 0;
const E_POINTER: i32 = -2147467261; // 0x80004003
const E_NOINTERFACE: i32 = -2147467262; // 0x80004002

// COM VARIANT vt codes used by UIA GetPropertyValue.
const VT_EMPTY: u16 = 0;
const VT_I4: u16 = 3;
const VT_R8: u16 = 5;
const VT_BSTR: u16 = 8;
const VT_BOOL: u16 = 11;
const VT_R8_ARRAY: u16 = VT_ARRAY | VT_R8;
const VT_I4_ARRAY: u16 = VT_ARRAY | VT_I4;
const VT_ARRAY: u16 = 0x2000;

/// COM `VARIANT` (16 bytes on x64: vt + 3 pad words + an 8-byte union payload).
#[repr(C)]
struct Variant {
    vt: u16,
    _r1: u16,
    _r2: u16,
    _r3: u16,
    payload: u64,
}

#[link(name = "oleaut32")]
unsafe extern "system" {
    fn SysAllocStringLen(psz: *const u16, len: u32) -> *mut u16;
    fn SafeArrayCreateVector(vt: u16, lbound: i32, celems: u32) -> *mut c_void;
    fn SafeArrayPutElement(psa: *mut c_void, indices: *const i32, pv: *const c_void) -> i32;
}

#[link(name = "uiautomationcore")]
unsafe extern "system" {
    fn UiaReturnRawElementProvider(
        hwnd: *mut c_void,
        wparam: usize,
        lparam: isize,
        el: *mut c_void,
    ) -> isize;
}

/// The three vtables share a header (IUnknown). We expose one fat vtable that is
/// a superset (Simple + Fragment + FragmentRoot in IDL order) and return the SAME
/// object pointer for all three IIDs from QueryInterface — legal because each
/// interface's slots are a prefix of the next in the UIA provider hierarchy:
///   IRawElementProviderSimple : IUnknown
///   IRawElementProviderFragment : IUnknown          (separate, but we co-host)
///   IRawElementProviderFragmentRoot : IRawElementProviderFragment
/// To keep slot layout exact per-interface we instead vend THREE distinct vtable
/// pointers via three sub-objects embedded in the provider (classic C++ multiple
/// inheritance COM layout).
#[repr(C)]
struct SimpleVtbl {
    QueryInterface: extern "system" fn(*mut c_void, *const GUID, *mut *mut c_void) -> i32,
    AddRef: extern "system" fn(*mut c_void) -> u32,
    Release: extern "system" fn(*mut c_void) -> u32,
    get_ProviderOptions: extern "system" fn(*mut c_void, *mut i32) -> i32,
    GetPatternProvider: extern "system" fn(*mut c_void, i32, *mut *mut c_void) -> i32,
    GetPropertyValue: extern "system" fn(*mut c_void, i32, *mut Variant) -> i32,
    get_HostRawElementProvider: extern "system" fn(*mut c_void, *mut *mut c_void) -> i32,
}

#[repr(C)]
struct FragmentVtbl {
    QueryInterface: extern "system" fn(*mut c_void, *const GUID, *mut *mut c_void) -> i32,
    AddRef: extern "system" fn(*mut c_void) -> u32,
    Release: extern "system" fn(*mut c_void) -> u32,
    Navigate: extern "system" fn(*mut c_void, i32, *mut *mut c_void) -> i32,
    GetRuntimeId: extern "system" fn(*mut c_void, *mut *mut c_void) -> i32,
    get_BoundingRectangle: extern "system" fn(*mut c_void, *mut [f64; 4]) -> i32,
    GetEmbeddedFragmentRoots: extern "system" fn(*mut c_void, *mut *mut c_void) -> i32,
    SetFocus: extern "system" fn(*mut c_void) -> i32,
    get_FragmentRoot: extern "system" fn(*mut c_void, *mut *mut c_void) -> i32,
}

// The single concrete object. The two interface sub-pointers let the OS call
// either vtable; both recover the same `Provider` by subtracting the sub-object
// offset. We keep it simple: store both vtable pointers as the first two fields
// and dispatch QueryInterface to hand back the matching one.
#[repr(C)]
struct Provider {
    simple_vtbl: *const SimpleVtbl,
    fragment_vtbl: *const FragmentVtbl,
    refcount: Cell<u32>,
    tree: PublishedTree,
    /// The AX id this provider represents (the root, for the object returned to
    /// WM_GETOBJECT). Navigation creates new Provider objects for other ids.
    ax_id: u32,
    hwnd: *mut c_void,
}

static SIMPLE_VTBL: SimpleVtbl = SimpleVtbl {
    QueryInterface: qi,
    AddRef: add_ref,
    Release: release,
    get_ProviderOptions: get_provider_options,
    GetPatternProvider: get_pattern_provider,
    GetPropertyValue: get_property_value,
    get_HostRawElementProvider: get_host_raw,
};

static FRAGMENT_VTBL: FragmentVtbl = FragmentVtbl {
    QueryInterface: qi_fragment,
    AddRef: add_ref_fragment,
    Release: release_fragment,
    Navigate: navigate,
    GetRuntimeId: get_runtime_id,
    get_BoundingRectangle: get_bounding,
    GetEmbeddedFragmentRoots: get_embedded_roots,
    SetFocus: set_focus,
    get_FragmentRoot: get_fragment_root,
};

impl Provider {
    fn alloc(tree: PublishedTree, ax_id: u32, hwnd: *mut c_void) -> *mut Provider {
        Box::into_raw(Box::new(Provider {
            simple_vtbl: &SIMPLE_VTBL,
            fragment_vtbl: &FRAGMENT_VTBL,
            refcount: Cell::new(1),
            tree,
            ax_id,
            hwnd,
        }))
    }
}

fn guid_eq(a: &GUID, b: &GUID) -> bool {
    a.Data1 == b.Data1 && a.Data2 == b.Data2 && a.Data3 == b.Data3 && a.Data4 == b.Data4
}

// ---- IUnknown (shared logic, two thin wrappers per vtable) -----------------

unsafe fn provider_from_simple<'a>(this: *mut c_void) -> &'a Provider {
    unsafe { &*(this as *const Provider) }
}
// The fragment vtable pointer is the SECOND field, so a `this` pointing at it is
// `provider + size_of::<*const ()>()`. Recover the Provider by subtracting one
// pointer width.
unsafe fn provider_from_fragment<'a>(this: *mut c_void) -> &'a Provider {
    let base = (this as *const u8).wrapping_sub(core::mem::size_of::<*const c_void>());
    unsafe { &*(base as *const Provider) }
}

extern "system" fn qi(this: *mut c_void, iid: *const GUID, out: *mut *mut c_void) -> i32 {
    if out.is_null() || iid.is_null() {
        return E_POINTER;
    }
    let iid = unsafe { &*iid };
    let p = unsafe { provider_from_simple(this) };
    if guid_eq(iid, &IID_IUNKNOWN) || guid_eq(iid, &IID_IRAW_ELEMENT_PROVIDER_SIMPLE) {
        p.refcount.set(p.refcount.get() + 1);
        unsafe { *out = this };
        S_OK
    } else if guid_eq(iid, &IID_IRAW_ELEMENT_PROVIDER_FRAGMENT)
        || guid_eq(iid, &IID_IRAW_ELEMENT_PROVIDER_FRAGMENT_ROOT)
    {
        p.refcount.set(p.refcount.get() + 1);
        // Hand back the fragment sub-object pointer (address of fragment_vtbl).
        unsafe { *out = (&raw const p.fragment_vtbl) as *mut c_void };
        S_OK
    } else {
        unsafe { *out = core::ptr::null_mut() };
        E_NOINTERFACE
    }
}

extern "system" fn qi_fragment(this: *mut c_void, iid: *const GUID, out: *mut *mut c_void) -> i32 {
    // Delegate to the simple QI using the recovered base object pointer.
    let p = unsafe { provider_from_fragment(this) };
    qi((p as *const Provider) as *mut c_void, iid, out)
}

extern "system" fn add_ref(this: *mut c_void) -> u32 {
    let p = unsafe { provider_from_simple(this) };
    let n = p.refcount.get() + 1;
    p.refcount.set(n);
    n
}
extern "system" fn add_ref_fragment(this: *mut c_void) -> u32 {
    let p = unsafe { provider_from_fragment(this) };
    add_ref((p as *const Provider) as *mut c_void)
}

extern "system" fn release(this: *mut c_void) -> u32 {
    let p = unsafe { provider_from_simple(this) };
    let n = p.refcount.get().saturating_sub(1);
    p.refcount.set(n);
    if n == 0 {
        // Reclaim the box.
        unsafe { drop(Box::from_raw(this as *mut Provider)) };
    }
    n
}
extern "system" fn release_fragment(this: *mut c_void) -> u32 {
    let p = unsafe { provider_from_fragment(this) };
    release((p as *const Provider) as *mut c_void)
}

// ---- IRawElementProviderSimple ---------------------------------------------

extern "system" fn get_provider_options(_this: *mut c_void, out: *mut i32) -> i32 {
    if out.is_null() {
        return E_POINTER;
    }
    unsafe { *out = PROVIDER_OPTIONS_SERVER_SIDE_PROVIDER };
    S_OK
}

extern "system" fn get_pattern_provider(
    _this: *mut c_void,
    _pattern_id: i32,
    out: *mut *mut c_void,
) -> i32 {
    // Pattern objects (Toggle/Value/...) are a follow-up; report "unsupported"
    // by returning a null provider with S_OK, which is the documented contract
    // (UIA treats a null pattern provider as "pattern not supported").
    if out.is_null() {
        return E_POINTER;
    }
    unsafe { *out = core::ptr::null_mut() };
    S_OK
}

extern "system" fn get_property_value(this: *mut c_void, prop_id: i32, out: *mut Variant) -> i32 {
    if out.is_null() {
        return E_POINTER;
    }
    let p = unsafe { provider_from_simple(this) };
    let prop = match prop_id_to_enum(prop_id) {
        Some(p) => p,
        None => {
            unsafe { *out = empty_variant() };
            return S_OK;
        }
    };
    let v = p.tree.property(p.ax_id, prop);
    unsafe { *out = variant_from(v) };
    S_OK
}

extern "system" fn get_host_raw(this: *mut c_void, out: *mut *mut c_void) -> i32 {
    if out.is_null() {
        return E_POINTER;
    }
    let p = unsafe { provider_from_simple(this) };
    // Only the fragment ROOT supplies a host provider (the HWND's own provider),
    // per UIA: this fuses our content tree onto the window's non-client tree.
    if Some(p.ax_id) == p.tree.root {
        let mut host: *mut c_void = core::ptr::null_mut();
        let hr = unsafe { UiaHostProviderFromHwnd(p.hwnd, &raw mut host) };
        if hr == S_OK {
            unsafe { *out = host };
            return S_OK;
        }
    }
    unsafe { *out = core::ptr::null_mut() };
    S_OK
}

#[link(name = "uiautomationcore")]
unsafe extern "system" {
    fn UiaHostProviderFromHwnd(hwnd: *mut c_void, ppProvider: *mut *mut c_void) -> i32;
}

// ---- IRawElementProviderFragment -------------------------------------------

extern "system" fn navigate(this: *mut c_void, dir: i32, out: *mut *mut c_void) -> i32 {
    if out.is_null() {
        return E_POINTER;
    }
    let p = unsafe { provider_from_fragment(this) };
    let dir = match dir {
        0 => NavigateDirection::Parent,
        1 => NavigateDirection::NextSibling,
        2 => NavigateDirection::PreviousSibling,
        3 => NavigateDirection::FirstChild,
        4 => NavigateDirection::LastChild,
        _ => {
            unsafe { *out = core::ptr::null_mut() };
            return S_OK;
        }
    };
    match p.tree.navigate(p.ax_id, dir) {
        Some(dest) => {
            // Vend a NEW provider object for the destination, sharing the
            // snapshot. The OS releases it when done.
            let obj = Provider::alloc(p.tree.clone(), dest, p.hwnd);
            unsafe { *out = obj as *mut c_void };
        }
        None => unsafe { *out = core::ptr::null_mut() },
    }
    S_OK
}

extern "system" fn get_runtime_id(this: *mut c_void, out: *mut *mut c_void) -> i32 {
    if out.is_null() {
        return E_POINTER;
    }
    let p = unsafe { provider_from_fragment(this) };
    let ids = match p.tree.find(p.ax_id) {
        Some(n) => n.provider.runtime_id.clone(),
        None => Vec::new(),
    };
    unsafe { *out = i4_safearray(&ids) };
    S_OK
}

extern "system" fn get_bounding(this: *mut c_void, out: *mut [f64; 4]) -> i32 {
    if out.is_null() {
        return E_POINTER;
    }
    let p = unsafe { provider_from_fragment(this) };
    let r = p
        .tree
        .find(p.ax_id)
        .map(|n| n.provider.bounding)
        .unwrap_or((0.0, 0.0, 0.0, 0.0));
    unsafe { *out = [r.0, r.1, r.2, r.3] };
    S_OK
}

extern "system" fn get_embedded_roots(_this: *mut c_void, out: *mut *mut c_void) -> i32 {
    if out.is_null() {
        return E_POINTER;
    }
    // No embedded fragment roots (iframes-as-OOPIF would populate this later).
    unsafe { *out = core::ptr::null_mut() };
    S_OK
}

extern "system" fn set_focus(_this: *mut c_void) -> i32 {
    // Programmatic focus from an AT (e.g. Narrator activating an item) is a
    // follow-up that must route back to the renderer to move DOM focus; until
    // that channel exists we honestly report S_OK without moving focus (no fake
    // state change). The renderer remains the focus source of truth.
    S_OK
}

extern "system" fn get_fragment_root(this: *mut c_void, out: *mut *mut c_void) -> i32 {
    if out.is_null() {
        return E_POINTER;
    }
    let p = unsafe { provider_from_fragment(this) };
    let root_id = p.tree.root.unwrap_or(p.ax_id);
    let obj = Provider::alloc(p.tree.clone(), root_id, p.hwnd);
    // Return the fragment-root sub-object pointer.
    let obj_ref = unsafe { &*obj };
    unsafe { *out = (&raw const obj_ref.fragment_vtbl) as *mut c_void };
    S_OK
}

// ---- VARIANT / SAFEARRAY marshaling ----------------------------------------

fn empty_variant() -> Variant {
    Variant {
        vt: VT_EMPTY,
        _r1: 0,
        _r2: 0,
        _r3: 0,
        payload: 0,
    }
}

fn variant_from(v: VariantValue) -> Variant {
    match v {
        VariantValue::Empty => empty_variant(),
        VariantValue::Bool(b) => Variant {
            vt: VT_BOOL,
            _r1: 0,
            _r2: 0,
            _r3: 0,
            // VARIANT_BOOL: -1 (0xFFFF) true, 0 false.
            payload: if b { 0xFFFF } else { 0 },
        },
        VariantValue::I4(i) => Variant {
            vt: VT_I4,
            _r1: 0,
            _r2: 0,
            _r3: 0,
            payload: (i as u32) as u64,
        },
        VariantValue::R8(f) => Variant {
            vt: VT_R8,
            _r1: 0,
            _r2: 0,
            _r3: 0,
            payload: f.to_bits(),
        },
        VariantValue::BStr(s) => {
            let bstr = alloc_bstr(&s);
            Variant {
                vt: VT_BSTR,
                _r1: 0,
                _r2: 0,
                _r3: 0,
                payload: bstr as u64,
            }
        }
        VariantValue::I4Array(arr) => Variant {
            vt: VT_I4_ARRAY,
            _r1: 0,
            _r2: 0,
            _r3: 0,
            payload: i4_safearray(&arr) as u64,
        },
        VariantValue::R8Array(arr) => Variant {
            vt: VT_R8_ARRAY,
            _r1: 0,
            _r2: 0,
            _r3: 0,
            payload: r8_safearray(&arr) as u64,
        },
    }
}

fn alloc_bstr(s: &str) -> *mut u16 {
    let utf16: Vec<u16> = s.encode_utf16().collect();
    unsafe { SysAllocStringLen(utf16.as_ptr(), utf16.len() as u32) }
}

fn i4_safearray(vals: &[i32]) -> *mut c_void {
    let sa = unsafe { SafeArrayCreateVector(VT_I4, 0, vals.len() as u32) };
    if sa.is_null() {
        return sa;
    }
    for (i, v) in vals.iter().enumerate() {
        let idx = i as i32;
        unsafe {
            SafeArrayPutElement(sa, &raw const idx, (v as *const i32) as *const c_void);
        }
    }
    sa
}

fn r8_safearray(vals: &[f64]) -> *mut c_void {
    let sa = unsafe { SafeArrayCreateVector(VT_R8, 0, vals.len() as u32) };
    if sa.is_null() {
        return sa;
    }
    for (i, v) in vals.iter().enumerate() {
        let idx = i as i32;
        unsafe {
            SafeArrayPutElement(sa, &raw const idx, (v as *const f64) as *const c_void);
        }
    }
    sa
}

fn prop_id_to_enum(id: i32) -> Option<UiaPropertyId> {
    use UiaPropertyId::*;
    Some(match id {
        30003 => ControlType,
        30005 => Name,
        30013 => HelpText,
        30001 => BoundingRectangle,
        30008 => HasKeyboardFocus,
        30009 => IsKeyboardFocusable,
        30010 => IsEnabled,
        30019 => IsPassword,
        30025 => IsRequiredForForm,
        30011 => AutomationId,
        30012 => ClassName,
        30024 => FrameworkId,
        30000 => RuntimeId,
        30016 => IsControlElement,
        30017 => IsContentElement,
        30022 => IsOffscreen,
        30086 => ToggleToggleState,
        30070 => ExpandCollapseState,
        30045 => ValueValue,
        30046 => ValueIsReadOnly,
        30154 => Level,
        30101 => AriaRole,
        _ => return None,
    })
}

const UIA_ROOT_OBJECT_ID: isize = -25;

/// `WM_GETOBJECT` handler. When the OS asks for the UIA root object
/// (`lParam == UiaRootObjectId`), construct a fragment-root provider serving the
/// currently-published AX snapshot and return it via
/// `UiaReturnRawElementProvider`. Returns the LRESULT for the window procedure,
/// or `None` if this message is not a UIA root request (caller falls through to
/// `DefWindowProc`).
///
/// SAFETY: must be called on the UI thread inside the window procedure; `hwnd`
/// must be the live window handle.
pub fn handle_wm_getobject(
    hwnd: *mut c_void,
    wparam: usize,
    lparam: isize,
) -> Option<isize> {
    if lparam != UIA_ROOT_OBJECT_ID {
        return None;
    }
    // Capture an immutable snapshot of the published tree for this provider's
    // lifetime. If nothing is published yet, serve an empty (document-only)
    // tree rather than failing — UIA tolerates an empty content root.
    let snapshot = crate::provider::with_published(|pt| pt.clone()).unwrap_or_default();
    let root_id = snapshot.root.unwrap_or(0);
    let obj = Provider::alloc(snapshot, root_id, hwnd);
    let lr = unsafe { UiaReturnRawElementProvider(hwnd, wparam, lparam, obj as *mut c_void) };
    // UiaReturnRawElementProvider takes its own reference; drop ours.
    let _ = release(obj as *mut c_void);
    Some(lr)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prop_id_roundtrip_covers_core_props() {
        // The ids the COM bridge must translate for screen readers.
        assert!(matches!(prop_id_to_enum(30005), Some(UiaPropertyId::Name)));
        assert!(matches!(
            prop_id_to_enum(30003),
            Some(UiaPropertyId::ControlType)
        ));
        assert!(matches!(
            prop_id_to_enum(30086),
            Some(UiaPropertyId::ToggleToggleState)
        ));
        assert!(prop_id_to_enum(999999).is_none());
    }

    #[test]
    fn variant_from_bool_uses_variant_bool_encoding() {
        let t = variant_from(VariantValue::Bool(true));
        assert_eq!(t.vt, VT_BOOL);
        assert_eq!(t.payload, 0xFFFF);
        let f = variant_from(VariantValue::Bool(false));
        assert_eq!(f.payload, 0);
    }

    #[test]
    fn variant_from_i4_packs_value() {
        let v = variant_from(VariantValue::I4(50000));
        assert_eq!(v.vt, VT_I4);
        assert_eq!(v.payload as u32 as i32, 50000);
    }

    #[test]
    fn guid_eq_matches_iunknown() {
        assert!(guid_eq(&IID_IUNKNOWN, &IID_IUNKNOWN));
        assert!(!guid_eq(&IID_IUNKNOWN, &IID_IRAW_ELEMENT_PROVIDER_SIMPLE));
    }
}
