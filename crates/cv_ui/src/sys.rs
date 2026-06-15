//! Win32 FFI for `cv_ui`.

#![allow(non_camel_case_types, non_snake_case, clippy::upper_case_acronyms)]
#![allow(unreachable_pub, missing_debug_implementations)]

use core::ffi::c_void;

pub type HWND = *mut c_void;
pub type HDC = *mut c_void;
pub type HBRUSH = *mut c_void;
pub type HICON = *mut c_void;
pub type HCURSOR = *mut c_void;
pub type HMENU = *mut c_void;
pub type HINSTANCE = *mut c_void;
pub type LPCWSTR = *const u16;
pub type WNDPROC =
    Option<unsafe extern "system" fn(hwnd: HWND, msg: u32, wparam: usize, lparam: isize) -> isize>;

pub const CW_USEDEFAULT: i32 = 0x8000_0000_u32 as i32;
pub const WS_OVERLAPPEDWINDOW: u32 = 0x00CF_0000;
pub const WS_CHILD: u32 = 0x4000_0000;
pub const WS_VISIBLE: u32 = 0x1000_0000;
pub const WS_CLIPCHILDREN: u32 = 0x0200_0000;
pub const WS_VSCROLL: u32 = 0x0020_0000;
pub const ES_AUTOHSCROLL: u32 = 0x0080;
pub const ES_LEFT: u32 = 0x0000;
pub const SW_SHOW: i32 = 5;
pub const SW_SHOWMAXIMIZED: i32 = 3;
pub const SM_CXSCREEN: i32 = 0;
pub const SM_CYSCREEN: i32 = 1;

/// `EM_SETSEL(start, end)` — selects characters [start, end). `wparam`
/// is start, `lparam` is end. `(0, -1)` selects all.
pub const EM_SETSEL: u32 = 0x00B1;

/// `SetWindowLongPtrW` index for the window procedure. Negative
/// because Win32 reserves the negative half of the index space for
/// these special slots.
pub const GWLP_WNDPROC: i32 = -4;

pub const CS_HREDRAW: u32 = 0x0002;
pub const CS_VREDRAW: u32 = 0x0001;
/// CS_DBLCLKS — class style asking Windows to translate the second
/// click of a double-click pair into `WM_LBUTTONDBLCLK` (or the right-/
/// middle-button equivalents) instead of a second `WM_LBUTTONDOWN`.
/// Without this style Windows treats two clicks as two independent
/// down events and the browser never receives a "dblclick" signal.
pub const CS_DBLCLKS: u32 = 0x0008;

pub const WM_PAINT: u32 = 0x000F;
pub const WM_DESTROY: u32 = 0x0002;
/// `WM_SETICON` — assign the window's title-bar / Alt-Tab / taskbar icon.
/// `wparam` selects the size class (`ICON_BIG` / `ICON_SMALL`).
pub const WM_SETICON: u32 = 0x0080;
/// Large icon (Alt-Tab, taskbar): `wparam` for `WM_SETICON`.
pub const ICON_BIG: usize = 1;
/// Small icon (title bar): `wparam` for `WM_SETICON`.
pub const ICON_SMALL: usize = 0;
/// `LR_DEFAULTCOLOR` — load the icon image using the system default color
/// format (no monochrome / shrink transforms). Flags for
/// `CreateIconFromResourceEx`.
pub const LR_DEFAULTCOLOR: u32 = 0x0000;
/// Icon resource version expected by `CreateIconFromResourceEx`.
pub const ICON_RES_VERSION: u32 = 0x0003_0000;
pub const WM_ERASEBKGND: u32 = 0x0014;
pub const WM_LBUTTONDOWN: u32 = 0x0201;
pub const WM_LBUTTONUP: u32 = 0x0202;
pub const WM_LBUTTONDBLCLK: u32 = 0x0203;
pub const WM_RBUTTONDOWN: u32 = 0x0204;
pub const WM_RBUTTONUP: u32 = 0x0205;
pub const WM_MBUTTONDOWN: u32 = 0x0207;
pub const WM_MBUTTONUP: u32 = 0x0208;
pub const WM_MOUSEWHEEL: u32 = 0x020A;
pub const WM_VSCROLL: u32 = 0x0115;
pub const WM_SIZE: u32 = 0x0005;
pub const WM_KEYDOWN: u32 = 0x0100;
pub const WM_KEYUP: u32 = 0x0101;
pub const WM_MOUSEMOVE: u32 = 0x0200;
pub const WM_SETCURSOR: u32 = 0x0020;
/// Sent to the window that HAD the mouse capture when capture is taken
/// away (by `SetCapture` elsewhere, `ReleaseCapture`, an Alt-Tab, etc.).
/// We use it to pop a stuck pressed-in nav button back out without
/// triggering its navigation. `lParam` is the HWND gaining capture.
pub const WM_CAPTURECHANGED: u32 = 0x0215;
/// `WM_DPICHANGED` — posted to a per-monitor-DPI-aware top-level window when
/// the DPI changes (the window moved to a monitor with a different scale, or
/// the user changed the scale). `wParam` LOWORD = the new DPI (e.g. 144 = 150%);
/// `lParam` points to the OS-suggested new window RECT. We read the new DPI,
/// republish `devicePixelRatio`, resize to the suggested rect, and repaint.
/// <https://learn.microsoft.com/en-us/windows/win32/hidpi/wm-dpichanged>
pub const WM_DPICHANGED: u32 = 0x02E0;

/// `WM_GETOBJECT` — sent by the OS accessibility runtime (and Narrator/other
/// ATs) to request an accessibility interface for the window. When
/// `lParam == UiaRootObjectId` (-25) the window returns an
/// `IRawElementProviderSimple` for its UI Automation content root via
/// `UiaReturnRawElementProvider`. We answer it from the published AX snapshot
/// (cv_a11y), gated behind `CV_A11Y_UIA`.
/// <https://learn.microsoft.com/en-us/windows/win32/winauto/wm-getobject>
pub const WM_GETOBJECT: u32 = 0x003D;

/// Opaque DPI-awareness pseudo-handle passed to `SetProcessDpiAwarenessContext`.
/// These are NOT real handles — they are small negative sentinel values the OS
/// recognises. Modelled as a pointer-sized C handle.
pub type DPI_AWARENESS_CONTEXT = *mut c_void;
/// `DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2` (-4). The recommended mode:
/// the window receives `WM_DPICHANGED` and reports the per-monitor DPI, child
/// windows / non-client area / dialogs scale correctly. (Windows 10 1703+.)
/// <https://learn.microsoft.com/en-us/windows/win32/hidpi/dpi-awareness-context>
pub const DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2: DPI_AWARENESS_CONTEXT = -4_isize as _;
/// `DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE` (-3). Fallback for OS builds
/// predating the V2 context (pre-1703); still per-monitor but without the V2
/// improvements (no automatic non-client scaling).
pub const DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE: DPI_AWARENESS_CONTEXT = -3_isize as _;
/// Baseline DPI for the 100% scale factor. `devicePixelRatio = dpi / 96`.
pub const USER_DEFAULT_SCREEN_DPI: u32 = 96;

pub const HTCLIENT: u32 = 1;
pub const IDC_HAND: LPCWSTR = 32649 as LPCWSTR;
pub const WM_TIMER: u32 = 0x0113;
pub const WM_CHAR: u32 = 0x0102;
/// Arbitrary, unique within this process — used for the JS event-loop
/// tick. WM_TIMER's `wParam` is the timer id we set here.
pub const JS_TICK_TIMER_ID: usize = 0xB17B;
pub const VK_ESCAPE: usize = 0x1B;
pub const VK_BACK: usize = 0x08;
pub const VK_PRIOR: usize = 0x21; // PageUp
pub const VK_NEXT: usize = 0x22; // PageDown
pub const VK_HOME: usize = 0x24;
pub const VK_END: usize = 0x23;
pub const VK_UP: usize = 0x26;
pub const VK_DOWN: usize = 0x28;

pub const SB_VERT: i32 = 1;
pub const SB_LINEUP: u32 = 0;
pub const SB_LINEDOWN: u32 = 1;
pub const SB_PAGEUP: u32 = 2;
pub const SB_PAGEDOWN: u32 = 3;
pub const SB_THUMBPOSITION: u32 = 4;
pub const SB_THUMBTRACK: u32 = 5;
pub const SB_TOP: u32 = 6;
pub const SB_BOTTOM: u32 = 7;
pub const SIF_RANGE: u32 = 0x0001;
pub const SIF_PAGE: u32 = 0x0002;
pub const SIF_POS: u32 = 0x0004;
pub const SIF_TRACKPOS: u32 = 0x0010;
pub const SIF_ALL: u32 = SIF_RANGE | SIF_PAGE | SIF_POS | SIF_TRACKPOS;

pub const IDC_ARROW: LPCWSTR = 32512 as LPCWSTR;

pub const BI_RGB: u32 = 0;
pub const DIB_RGB_COLORS: u32 = 0;
pub const SRCCOPY: u32 = 0x00CC_0020;

#[repr(C)]
pub struct WNDCLASSEXW {
    pub cbSize: u32,
    pub style: u32,
    pub lpfnWndProc: WNDPROC,
    pub cbClsExtra: i32,
    pub cbWndExtra: i32,
    pub hInstance: HINSTANCE,
    pub hIcon: HICON,
    pub hCursor: HCURSOR,
    pub hbrBackground: HBRUSH,
    pub lpszMenuName: LPCWSTR,
    pub lpszClassName: LPCWSTR,
    pub hIconSm: HICON,
}

#[repr(C)]
#[derive(Default, Copy, Clone)]
pub struct RECT {
    pub left: i32,
    pub top: i32,
    pub right: i32,
    pub bottom: i32,
}

#[repr(C)]
#[derive(Default, Copy, Clone)]
pub struct POINT {
    pub x: i32,
    pub y: i32,
}

#[repr(C)]
#[derive(Default, Copy, Clone)]
pub struct MSG {
    pub hwnd: HWND,
    pub message: u32,
    pub wParam: usize,
    pub lParam: isize,
    pub time: u32,
    pub pt: POINT,
}

#[repr(C)]
pub struct PAINTSTRUCT {
    pub hdc: HDC,
    pub fErase: i32,
    pub rcPaint: RECT,
    pub fRestore: i32,
    pub fIncUpdate: i32,
    pub rgbReserved: [u8; 32],
}

#[repr(C)]
pub struct BITMAPINFOHEADER {
    pub biSize: u32,
    pub biWidth: i32,
    pub biHeight: i32,
    pub biPlanes: u16,
    pub biBitCount: u16,
    pub biCompression: u32,
    pub biSizeImage: u32,
    pub biXPelsPerMeter: i32,
    pub biYPelsPerMeter: i32,
    pub biClrUsed: u32,
    pub biClrImportant: u32,
}

#[repr(C)]
pub struct BITMAPINFO {
    pub bmiHeader: BITMAPINFOHEADER,
    pub bmiColors: [u32; 1],
}

#[repr(C)]
#[derive(Default, Copy, Clone)]
pub struct SCROLLINFO {
    pub cbSize: u32,
    pub fMask: u32,
    pub nMin: i32,
    pub nMax: i32,
    pub nPage: u32,
    pub nPos: i32,
    pub nTrackPos: i32,
}

#[link(name = "user32")]
unsafe extern "system" {
    pub fn GetModuleHandleW(module: LPCWSTR) -> HINSTANCE;
    /// Per-monitor DPI of the monitor `hwnd` is on (Windows 10 1607+). Returns
    /// 96 for the 100% baseline, 120 for 125%, 144 for 150%, 192 for 200%, etc.
    /// Requires the process to be per-monitor-DPI-aware; otherwise returns the
    /// system DPI. <https://learn.microsoft.com/en-us/windows/win32/api/winuser/nf-winuser-getdpiforwindow>
    pub fn GetDpiForWindow(hwnd: HWND) -> u32;
    /// Set the process default DPI awareness to one of the
    /// `DPI_AWARENESS_CONTEXT_*` pseudo-handles (Windows 10 1703+). Must be
    /// called before any window is created. Returns nonzero on success.
    /// <https://learn.microsoft.com/en-us/windows/win32/api/winuser/nf-winuser-setprocessdpiawarenesscontext>
    pub fn SetProcessDpiAwarenessContext(value: DPI_AWARENESS_CONTEXT) -> i32;
    pub fn LoadCursorW(hInstance: HINSTANCE, lpCursorName: LPCWSTR) -> HCURSOR;
    pub fn SetCursor(hCursor: HCURSOR) -> HCURSOR;
    pub fn GetCursorPos(lpPoint: *mut POINT) -> i32;
    pub fn ScreenToClient(hWnd: HWND, lpPoint: *mut POINT) -> i32;
    pub fn RegisterClassExW(lpwcx: *const WNDCLASSEXW) -> u16;
    pub fn CreateWindowExW(
        dwExStyle: u32,
        lpClassName: LPCWSTR,
        lpWindowName: LPCWSTR,
        dwStyle: u32,
        X: i32,
        Y: i32,
        nWidth: i32,
        nHeight: i32,
        hWndParent: HWND,
        hMenu: HMENU,
        hInstance: HINSTANCE,
        lpParam: *mut c_void,
    ) -> HWND;
    pub fn DestroyWindow(hwnd: HWND) -> i32;
    pub fn GetSystemMetrics(nIndex: i32) -> i32;
    pub fn ShowWindow(hwnd: HWND, nCmdShow: i32) -> i32;
    pub fn UpdateWindow(hwnd: HWND) -> i32;
    pub fn AdjustWindowRectEx(lpRect: *mut RECT, dwStyle: u32, bMenu: i32, dwExStyle: u32) -> i32;
    pub fn DefWindowProcW(hwnd: HWND, msg: u32, wparam: usize, lparam: isize) -> isize;
    pub fn GetMessageW(lpMsg: *mut MSG, hwnd: HWND, wMsgFilterMin: u32, wMsgFilterMax: u32) -> i32;
    pub fn TranslateMessage(lpMsg: *const MSG) -> i32;
    pub fn DispatchMessageW(lpMsg: *const MSG) -> isize;
    pub fn PostQuitMessage(nExitCode: i32);
    pub fn BeginPaint(hwnd: HWND, lpPaint: *mut PAINTSTRUCT) -> HDC;
    pub fn EndPaint(hwnd: HWND, lpPaint: *const PAINTSTRUCT) -> i32;
    pub fn InvalidateRect(hwnd: HWND, lpRect: *const RECT, bErase: i32) -> i32;
    pub fn SetWindowTextW(hwnd: HWND, lpString: LPCWSTR) -> i32;
    pub fn GetClientRect(hwnd: HWND, lpRect: *mut RECT) -> i32;
    pub fn SetScrollInfo(hwnd: HWND, nBar: i32, lpsi: *const SCROLLINFO, redraw: i32) -> i32;
    pub fn GetScrollInfo(hwnd: HWND, nBar: i32, lpsi: *mut SCROLLINFO) -> i32;
    pub fn ShowScrollBar(hwnd: HWND, wBar: i32, bShow: i32) -> i32;
    pub fn SetTimer(hwnd: HWND, nIDEvent: usize, uElapse: u32, lpTimerFunc: *mut c_void) -> usize;
    pub fn KillTimer(hwnd: HWND, uIDEvent: usize) -> i32;
    pub fn GetKeyState(nVirtKey: i32) -> i16;
    pub fn SetFocus(hwnd: HWND) -> HWND;
    pub fn GetFocus() -> HWND;
    /// Capture all mouse input to `hwnd` until `ReleaseCapture`. Used so a
    /// nav button keeps its pressed-in look (and we receive the matching
    /// mouse-up) even if the cursor drifts off the button while held.
    pub fn SetCapture(hwnd: HWND) -> HWND;
    /// Release the mouse capture taken by `SetCapture`. Returns nonzero on
    /// success. Releasing capture posts `WM_CAPTURECHANGED` to the prior
    /// capture window.
    pub fn ReleaseCapture() -> i32;
    pub fn GetParent(hwnd: HWND) -> HWND;
    pub fn GetWindowTextW(hwnd: HWND, lpString: *mut u16, nMaxCount: i32) -> i32;
    pub fn GetWindowTextLengthW(hwnd: HWND) -> i32;
    pub fn MoveWindow(hwnd: HWND, X: i32, Y: i32, nWidth: i32, nHeight: i32, bRepaint: i32) -> i32;
    pub fn SendMessageW(hwnd: HWND, msg: u32, wparam: usize, lparam: isize) -> isize;
    /// Build an `HICON` from a single in-memory icon image frame (the bytes
    /// of one ICONDIRENTRY's payload — either a PNG or a DIB). Used to embed
    /// the Conclave window icon at runtime from `assets/conclave.ico` without
    /// a resource-compiler toolchain. `dwVer` must be `0x0003_0000`.
    pub fn CreateIconFromResourceEx(
        presbits: *const u8,
        dwResSize: u32,
        fIcon: i32,
        dwVer: u32,
        cxDesired: i32,
        cyDesired: i32,
        flags: u32,
    ) -> HICON;
    /// Replace a window's WNDPROC (or other long pointers, indexed by
    /// `nIndex`). Returns the previous value — we stash that and
    /// chain to it via `CallWindowProcW` so EDIT control behavior
    /// stays intact aside from our Enter/Esc intercept.
    pub fn SetWindowLongPtrW(hwnd: HWND, nIndex: i32, dwNewLong: isize) -> isize;
    /// Invoke a saved WNDPROC. Used by the subclass to forward
    /// unhandled messages back to the standard EDIT proc.
    pub fn CallWindowProcW(
        lpPrevWndFunc: isize,
        hwnd: HWND,
        msg: u32,
        wparam: usize,
        lparam: isize,
    ) -> isize;
}

/// Read live modifier state for `KeyboardEvent.ctrlKey` etc.
pub fn modifiers_now() -> (bool, bool, bool, bool) {
    unsafe {
        let down = |vk: i32| (GetKeyState(vk) as u16) & 0x8000 != 0;
        (
            down(0x11),               // VK_CONTROL
            down(0x10),               // VK_SHIFT
            down(0x12),               // VK_MENU (Alt)
            down(0x5B) || down(0x5C), // VK_LWIN / VK_RWIN (meta)
        )
    }
}

pub type HFONT = *mut c_void;
pub type HGDIOBJ = *mut c_void;
pub type COLORREF = u32;

pub const DT_LEFT: u32 = 0x0;
pub const DT_TOP: u32 = 0x0;
pub const DT_NOPREFIX: u32 = 0x800;
pub const DT_WORDBREAK: u32 = 0x10;
pub const DT_CENTER: u32 = 0x1;
pub const DT_RIGHT: u32 = 0x2;
pub const DT_VCENTER: u32 = 0x4;
pub const DT_SINGLELINE: u32 = 0x20;
pub const TRANSPARENT: i32 = 1;

pub const FW_NORMAL: i32 = 400;
pub const FW_BOLD: i32 = 700;
pub const DEFAULT_CHARSET: u32 = 1;
pub const OUT_DEFAULT_PRECIS: u32 = 0;
pub const CLIP_DEFAULT_PRECIS: u32 = 0;
pub const CLEARTYPE_QUALITY: u32 = 5;
pub const DEFAULT_PITCH: u32 = 0;
pub const FF_DONTCARE: u32 = 0;

#[inline]
pub fn rgb(r: u8, g: u8, b: u8) -> COLORREF {
    u32::from(r) | (u32::from(g) << 8) | (u32::from(b) << 16)
}

#[link(name = "gdi32")]
unsafe extern "system" {
    pub fn StretchDIBits(
        hdc: HDC,
        xDest: i32,
        yDest: i32,
        DestWidth: i32,
        DestHeight: i32,
        xSrc: i32,
        ySrc: i32,
        SrcWidth: i32,
        SrcHeight: i32,
        lpBits: *const c_void,
        lpbmi: *const BITMAPINFO,
        iUsage: u32,
        rop: u32,
    ) -> i32;
    pub fn CreateFontW(
        cHeight: i32,
        cWidth: i32,
        cEscapement: i32,
        cOrientation: i32,
        cWeight: i32,
        bItalic: u32,
        bUnderline: u32,
        bStrikeOut: u32,
        iCharSet: u32,
        iOutPrecision: u32,
        iClipPrecision: u32,
        iQuality: u32,
        iPitchAndFamily: u32,
        pszFaceName: LPCWSTR,
    ) -> HFONT;
    pub fn SelectObject(hdc: HDC, h: HGDIOBJ) -> HGDIOBJ;
    pub fn DeleteObject(h: HGDIOBJ) -> i32;
    pub fn SetTextColor(hdc: HDC, color: COLORREF) -> COLORREF;
    /// GDI extra-spacing between character cells. Honors CSS
    /// `letter-spacing` at draw time. Default is 0; positive values
    /// spread the run, negative tighten it. Per-DC state so set/reset
    /// around each `DrawTextW` call to avoid bleed into the next.
    pub fn SetTextCharacterExtra(hdc: HDC, extra: i32) -> i32;
    pub fn SetBkMode(hdc: HDC, mode: i32) -> i32;
    pub fn CreateSolidBrush(color: COLORREF) -> HGDIOBJ;
    /// Create a rectangular clip region.  Returns an HRGN owned by the
    /// caller; freed via `DeleteObject`.
    pub fn CreateRectRgn(x1: i32, y1: i32, x2: i32, y2: i32) -> HGDIOBJ;
    /// Create a memory device context compatible with the supplied
    /// HDC.  Pass null to get a screen-compatible memory DC (most
    /// common use — off-screen drawing target).  Free via DeleteDC.
    pub fn CreateCompatibleDC(hdc: HDC) -> HDC;
    /// Free a memory DC.
    pub fn DeleteDC(hdc: HDC) -> i32;
    /// Create a DIB section bound to GDI-allocated memory.  On return
    /// `ppvBits` points to the pixel buffer the caller can read/write
    /// directly.  `DIB_RGB_COLORS = 0`.  Free via DeleteObject.
    pub fn CreateDIBSection(
        hdc: HDC,
        pbmi: *const BITMAPINFO,
        usage: u32,
        ppvBits: *mut *mut c_void,
        hSection: *mut c_void,
        offset: u32,
    ) -> HGDIOBJ;
    /// Set the HDC's clip region.  Pass null to clear the clip.  The
    /// HDC stores a COPY of the region, so the caller still owns the
    /// HRGN and should free it after this call.
    pub fn SelectClipRgn(hdc: HDC, hrgn: HGDIOBJ) -> i32;
    /// Measure the pixel width/height of a UTF-16 string in the HDC's
    /// currently selected font. `size_out` receives a `SIZE { cx, cy }`
    /// — `cx` is the advance-width sum across the string (what we
    /// want for caret placement); `cy` is the line height. Used by
    /// `cv_ui::measure_text_px` to position the caret at the
    /// pixel-exact end of the input's typed text.
    pub fn GetTextExtentPoint32W(hdc: HDC, lpString: *const u16, c: i32, psizl: *mut SIZE) -> i32;
    /// Draw + fill a closed polygon using the HDC's current pen (edge) and
    /// brush (interior). `points` is an array of `count` `POINT`s. Used to
    /// paint the crisp filled triangle arrows on the nav buttons.
    pub fn Polygon(hdc: HDC, points: *const POINT, count: i32) -> i32;
    /// Draw an elliptical arc outline using the HDC's current pen. The arc is
    /// bounded by the rectangle `(left,top)-(right,bottom)`; it is drawn
    /// counter-clockwise from the radial vector through `(x1,y1)` to the radial
    /// vector through `(x2,y2)`. No fill (interior is untouched). Used to paint
    /// the reload/refresh circular-arrow glyph crisply at small sizes.
    pub fn Arc(
        hdc: HDC,
        left: i32,
        top: i32,
        right: i32,
        bottom: i32,
        x1: i32,
        y1: i32,
        x2: i32,
        y2: i32,
    ) -> i32;
    /// Create a cosmetic pen (line style/width/color). `style` is e.g.
    /// `PS_SOLID`. Returns an HPEN (an HGDIOBJ) the caller frees via
    /// `DeleteObject`.
    pub fn CreatePen(style: i32, width: i32, color: COLORREF) -> HPEN;
}

/// Handle to a GDI pen. Aliases `HGDIOBJ` so it threads through
/// `SelectObject` / `DeleteObject` like every other GDI object.
pub type HPEN = *mut c_void;

/// Solid pen style for `CreatePen`.
pub const PS_SOLID: i32 = 0;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct SIZE {
    pub cx: i32,
    pub cy: i32,
}

#[link(name = "user32")]
unsafe extern "system" {
    pub fn DrawTextW(
        hdc: HDC,
        lpchText: LPCWSTR,
        cchText: i32,
        lprc: *mut RECT,
        format: u32,
    ) -> i32;
    pub fn FillRect(hdc: HDC, lprc: *const RECT, hbr: HGDIOBJ) -> i32;
    /// Asynchronously post a message to a window's queue. Used by the
    /// background-fetch thread to hand a completed body back to the UI
    /// thread without blocking either.
    pub fn PostMessageW(hwnd: HWND, msg: u32, wparam: usize, lparam: isize) -> i32;
}

#[link(name = "kernel32")]
unsafe extern "system" {
    /// Id of the calling thread. Used to detect cross-thread calls into
    /// UI-thread-only state: the off-main renderer thread must never borrow the
    /// UI thread's `WindowState`, so `invalidate_caret` posts a message instead.
    pub fn GetCurrentThreadId() -> u32;

    /// Last-error code for the calling thread. Used to surface a precise reason
    /// when a USER/GDI object creation (e.g. `CreateWindowExW`) returns null, so
    /// a launch failure becomes a clean error message instead of a panic-abort.
    pub fn GetLastError() -> u32;
}

/// First app-defined message. We use this for "fetch completed —
/// here's the body" messages from the background thread.
pub const WM_USER: u32 = 0x0400;

// ── winmm: click sound for the nav buttons ──────────────────────────────
/// Play the sound asynchronously (return immediately, don't block the UI
/// thread while the click plays).
pub const SND_ASYNC: u32 = 0x0001;
/// `sound` points at an in-memory WAV image (RIFF bytes), not a filename
/// or alias. Lets us play an embedded WAV with zero file/asset dependency.
pub const SND_MEMORY: u32 = 0x0004;
/// Don't emit the default system "beep" if the requested sound can't be
/// played — silence-on-failure rather than a jarring fallback ding. (We
/// embed a guaranteed-valid WAV, so this is just defensive.)
pub const SND_NODEFAULT: u32 = 0x0002;

#[link(name = "winmm")]
unsafe extern "system" {
    /// Play a waveform sound. With `SND_MEMORY` the `sound` pointer is an
    /// in-memory WAV image (cast from `*const u8`); `hmod` is null. Returns
    /// nonzero on success. We pass `SND_ASYNC` so it never blocks input.
    pub fn PlaySoundW(sound: *const u16, hmod: *mut c_void, flags: u32) -> i32;
}
