//! Raw Win32 / WinSock FFI declarations for `cv_net`.
//!
//! Declared inline rather than pulling `windows-sys`, per the strict
//! third-party policy.

#![allow(non_camel_case_types, non_snake_case, clippy::upper_case_acronyms)]
#![allow(missing_debug_implementations)]

use core::ffi::c_void;

pub type BOOL = i32;
pub type DWORD = u32;
pub type WORD = u16;
pub type INT = i32;
pub type SOCKET = usize;
pub type CHAR = u8;
pub type LPCSTR = *const u8;
pub type LPVOID = *mut c_void;

pub const INVALID_SOCKET: SOCKET = !0_usize;
pub const SOCKET_ERROR: i32 = -1;
pub const AF_UNSPEC: i32 = 0;
pub const AF_INET: i32 = 2;
pub const AF_INET6: i32 = 23;
pub const SOCK_STREAM: i32 = 1;
pub const SOCK_DGRAM: i32 = 2;
pub const IPPROTO_TCP: i32 = 6;
pub const IPPROTO_UDP: i32 = 17;
/// Disable Nagle's algorithm. Critical for our TLS 1.3 client: BoringSSL
/// servers (Cloudflare's edges) fire `unexpected_message` if our client
/// Finished and first application data record arrive in the same TCP
/// segment — `tls_open_record` checks
/// `tls_has_unprocessed_handshake_data` before accepting an app-data
/// record, and if Nagle batches Finished + AppData into one packet, the
/// server's record loop sees the app data with the Finished still
/// buffered for the state machine.
pub const TCP_NODELAY: i32 = 0x0001;
pub const SOL_SOCKET: i32 = 0xffff;
pub const SO_RCVTIMEO: i32 = 0x1006;
pub const SO_SNDTIMEO: i32 = 0x1005;
pub const SO_ERROR: i32 = 0x1007;
pub const FIONBIO: u32 = 0x8004_667e;
pub const WSAEWOULDBLOCK: i32 = 10035;
pub const FD_SETSIZE: usize = 64;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct fd_set {
    pub fd_count: u32,
    pub fd_array: [SOCKET; FD_SETSIZE],
}

impl fd_set {
    pub fn zeroed() -> Self {
        unsafe { core::mem::zeroed() }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct timeval {
    pub tv_sec: i32,
    pub tv_usec: i32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct WSADATA {
    pub wVersion: WORD,
    pub wHighVersion: WORD,
    pub iMaxSockets: u16,
    pub iMaxUdpDg: u16,
    pub lpVendorInfo: *mut u8,
    pub szDescription: [CHAR; 257],
    pub szSystemStatus: [CHAR; 129],
}

impl WSADATA {
    pub fn zeroed() -> Self {
        // SAFETY: all fields are POD or raw pointers; zero is a valid
        // representation of every field (NULL ptr is fine to overwrite).
        unsafe { core::mem::zeroed() }
    }
}

#[repr(C)]
pub struct sockaddr {
    pub sa_family: u16,
    pub sa_data: [u8; 14],
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct sockaddr_in {
    pub sin_family: u16,
    pub sin_port: u16,
    pub sin_addr: [u8; 4],
    pub sin_zero: [u8; 8],
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct sockaddr_in6 {
    pub sin6_family: u16,
    pub sin6_port: u16,
    pub sin6_flowinfo: u32,
    pub sin6_addr: [u8; 16],
    pub sin6_scope_id: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct addrinfoW {
    pub ai_flags: i32,
    pub ai_family: i32,
    pub ai_socktype: i32,
    pub ai_protocol: i32,
    pub ai_addrlen: usize,
    pub ai_canonname: *mut u16,
    pub ai_addr: *mut sockaddr,
    pub ai_next: *mut addrinfoW,
}

unsafe extern "system" {
    pub fn WSAStartup(wVersionRequested: WORD, lpWSAData: *mut WSADATA) -> INT;
    pub fn WSACleanup() -> INT;
    pub fn WSAGetLastError() -> INT;

    pub fn socket(af: i32, type_: i32, protocol: i32) -> SOCKET;
    pub fn closesocket(s: SOCKET) -> INT;
    pub fn connect(s: SOCKET, name: *const sockaddr, namelen: i32) -> INT;
    pub fn send(s: SOCKET, buf: *const u8, len: i32, flags: i32) -> INT;
    pub fn recv(s: SOCKET, buf: *mut u8, len: i32, flags: i32) -> INT;
    pub fn setsockopt(s: SOCKET, level: i32, optname: i32, optval: *const u8, optlen: i32) -> INT;
    pub fn shutdown(s: SOCKET, how: i32) -> INT;

    pub fn ioctlsocket(s: SOCKET, cmd: i32, argp: *mut u32) -> INT;
    pub fn select(
        nfds: i32,
        readfds: *mut fd_set,
        writefds: *mut fd_set,
        exceptfds: *mut fd_set,
        timeout: *const timeval,
    ) -> INT;
    pub fn getsockopt(
        s: SOCKET,
        level: i32,
        optname: i32,
        optval: *mut u8,
        optlen: *mut i32,
    ) -> INT;

    pub fn GetAddrInfoW(
        node: *const u16,
        service: *const u16,
        hints: *const addrinfoW,
        result: *mut *mut addrinfoW,
    ) -> i32;
    pub fn FreeAddrInfoW(info: *mut addrinfoW);
}

pub const SHUTDOWN_BOTH: i32 = 2;
