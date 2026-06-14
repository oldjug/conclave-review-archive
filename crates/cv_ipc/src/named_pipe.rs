//! Windows named-pipe transport.
//!
//! Sends and receives the same wire-framed messages as the in-process
//! `Endpoint`, but the bytes cross a kernel pipe instead of an in-RAM
//! `VecDeque`. The transport is **byte-stream** (PIPE_TYPE_BYTE) so the
//! framing header is what tells the receiver where a message ends —
//! that mirrors what the future TCP/QUIC transport will look like and
//! keeps a single decoder serving both.
//!
//! V1 is synchronous (blocking ReadFile/WriteFile). Async via
//! `IO_COMPLETION_PORT` lands when the renderer ↔ browser process
//! split needs it.

#![cfg(target_os = "windows")]
#![allow(non_camel_case_types, non_snake_case, clippy::upper_case_acronyms)]
#![allow(unreachable_pub, missing_debug_implementations, dead_code)]

use core::ffi::c_void;

use crate::codec::{Decode, Encode, Reader, Writer};
use crate::{MAGIC, RecvError, SendError};

type HANDLE = *mut c_void;
type BOOL = i32;
type DWORD = u32;
type LPCWSTR = *const u16;
type LPVOID = *mut c_void;
type LPCVOID = *const c_void;
type LPSECURITY_ATTRIBUTES = *mut c_void;
type LPOVERLAPPED = *mut c_void;

const INVALID_HANDLE_VALUE: HANDLE = -1isize as HANDLE;

const PIPE_ACCESS_DUPLEX: DWORD = 0x0000_0003;
const PIPE_TYPE_BYTE: DWORD = 0x0000_0000;
const PIPE_READMODE_BYTE: DWORD = 0x0000_0000;
const PIPE_WAIT: DWORD = 0x0000_0000;

const GENERIC_READ: DWORD = 0x8000_0000;
const GENERIC_WRITE: DWORD = 0x4000_0000;
const OPEN_EXISTING: DWORD = 3;
const FILE_ATTRIBUTE_NORMAL: DWORD = 0x0000_0080;

const ERROR_PIPE_CONNECTED: DWORD = 535;
const ERROR_BROKEN_PIPE: DWORD = 109;
const ERROR_PIPE_LISTENING: DWORD = 536;
const ERROR_PIPE_BUSY: DWORD = 231;
const ERROR_IO_PENDING: DWORD = 997;

// Overlapped-accept constants (timed `ConnectNamedPipe`).
const FILE_FLAG_OVERLAPPED: DWORD = 0x4000_0000;
const WAIT_OBJECT_0: DWORD = 0;
const WAIT_TIMEOUT: DWORD = 258;
const INFINITE_MS: DWORD = 0xFFFF_FFFF;

/// Win32 `OVERLAPPED` layout. We only ever touch `hEvent`; the rest must
/// be present (and zeroed) for the kernel.
#[repr(C)]
struct OVERLAPPED {
    Internal: usize,
    InternalHigh: usize,
    Offset: DWORD,
    OffsetHigh: DWORD,
    hEvent: HANDLE,
}

#[link(name = "kernel32")]
unsafe extern "system" {
    fn CreateNamedPipeW(
        lpName: LPCWSTR,
        dwOpenMode: DWORD,
        dwPipeMode: DWORD,
        nMaxInstances: DWORD,
        nOutBufferSize: DWORD,
        nInBufferSize: DWORD,
        nDefaultTimeOut: DWORD,
        lpSecurityAttributes: LPSECURITY_ATTRIBUTES,
    ) -> HANDLE;

    fn CreateEventW(
        lpEventAttributes: LPSECURITY_ATTRIBUTES,
        bManualReset: BOOL,
        bInitialState: BOOL,
        lpName: LPCWSTR,
    ) -> HANDLE;

    fn WaitForSingleObject(hHandle: HANDLE, dwMilliseconds: DWORD) -> DWORD;

    fn GetOverlappedResult(
        hFile: HANDLE,
        lpOverlapped: *mut OVERLAPPED,
        lpNumberOfBytesTransferred: *mut DWORD,
        bWait: BOOL,
    ) -> BOOL;

    fn CancelIoEx(hFile: HANDLE, lpOverlapped: *mut OVERLAPPED) -> BOOL;

    fn CreateFileW(
        lpFileName: LPCWSTR,
        dwDesiredAccess: DWORD,
        dwShareMode: DWORD,
        lpSecurityAttributes: LPSECURITY_ATTRIBUTES,
        dwCreationDisposition: DWORD,
        dwFlagsAndAttributes: DWORD,
        hTemplateFile: HANDLE,
    ) -> HANDLE;

    fn ConnectNamedPipe(hNamedPipe: HANDLE, lpOverlapped: LPOVERLAPPED) -> BOOL;

    fn DisconnectNamedPipe(hNamedPipe: HANDLE) -> BOOL;

    fn ReadFile(
        hFile: HANDLE,
        lpBuffer: LPVOID,
        nNumberOfBytesToRead: DWORD,
        lpNumberOfBytesRead: *mut DWORD,
        lpOverlapped: LPOVERLAPPED,
    ) -> BOOL;

    fn WriteFile(
        hFile: HANDLE,
        lpBuffer: LPCVOID,
        nNumberOfBytesToWrite: DWORD,
        lpNumberOfBytesWritten: *mut DWORD,
        lpOverlapped: LPOVERLAPPED,
    ) -> BOOL;

    fn CloseHandle(hObject: HANDLE) -> BOOL;

    fn GetLastError() -> DWORD;

    fn WaitNamedPipeW(lpNamedPipeName: LPCWSTR, nTimeOut: DWORD) -> BOOL;
}

#[derive(Debug, PartialEq, Eq)]
pub enum TransportError {
    /// CreateNamedPipeW failed (last-error in payload).
    Create(u32),
    /// CreateFileW (client connect) failed.
    Connect(u32),
    /// Read/Write returned 0 or got a Win32 error.
    Io(u32),
    /// Peer closed the pipe.
    Closed,
    /// Frame header malformed.
    BadFrame,
    /// Encoder/decoder rejected the payload.
    Codec,
    /// A bounded accept/connect (`create_server_timeout`) expired before a
    /// client connected — a renderer that never came up fails honestly here
    /// instead of wedging the caller forever.
    Timeout,
}

impl std::fmt::Display for TransportError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Create(e) => write!(f, "named pipe create failed ({e})"),
            Self::Connect(e) => write!(f, "named pipe connect failed ({e})"),
            Self::Io(e) => write!(f, "named pipe io failed ({e})"),
            Self::Closed => f.write_str("named pipe closed by peer"),
            Self::BadFrame => f.write_str("bad frame"),
            Self::Codec => f.write_str("codec rejected"),
            Self::Timeout => f.write_str("named pipe accept timed out"),
        }
    }
}

impl std::error::Error for TransportError {}

/// Convert a Rust pipe name (no `\\.\pipe\` prefix) into a UTF-16
/// null-terminated path Windows expects.
fn pipe_path_utf16(name: &str) -> Vec<u16> {
    let full = format!("\\\\.\\pipe\\{name}");
    full.encode_utf16().chain(std::iter::once(0)).collect()
}

/// One side of a kernel pipe. RAII: dropping the struct closes the
/// handle. Both client and server end up with one of these once
/// connected; the API after that point is symmetric.
pub struct PipeHandle {
    h: HANDLE,
    /// True when the handle was opened with `FILE_FLAG_OVERLAPPED` (the
    /// timed-accept server path). Such a handle REQUIRES every `ReadFile`/
    /// `WriteFile` to pass an `OVERLAPPED`; we issue them overlapped and
    /// then block on the event for completion (`sync_read`/`sync_write`),
    /// which preserves the exact blocking semantics of the non-overlapped
    /// path. A non-overlapped handle (`false`) keeps the original direct
    /// blocking `ReadFile`/`WriteFile` with a null overlapped pointer.
    overlapped: bool,
}

unsafe impl Send for PipeHandle {}

// SAFETY: a duplex (`PIPE_ACCESS_DUPLEX`) byte-mode named pipe has two
// independent kernel I/O buffers — one per direction. A blocking
// `ReadFile` on one thread and a blocking `WriteFile` on another thread,
// both against the same handle, operate on those disjoint directions and
// do not race: the Windows I/O manager serialises requests per direction.
// This crate only ever shares a handle as exactly one reader + one writer
// (the persistent renderer's reader thread `recv`s while its main thread
// `send`s a committed frame); it never issues two concurrent reads or two
// concurrent writes on the same handle. `&self` `recv`/`send` therefore
// stay sound to call from two threads at once under that contract.
unsafe impl Sync for PipeHandle {}

impl Drop for PipeHandle {
    fn drop(&mut self) {
        if !self.h.is_null() && self.h != INVALID_HANDLE_VALUE {
            unsafe {
                CloseHandle(self.h);
            }
        }
    }
}

impl PipeHandle {
    /// Server side: publish a pipe with the given name and wait for a
    /// client to connect. Blocks until connection is established.
    pub fn create_server(name: &str) -> Result<Self, TransportError> {
        let path = pipe_path_utf16(name);
        let h = unsafe {
            CreateNamedPipeW(
                path.as_ptr(),
                PIPE_ACCESS_DUPLEX,
                PIPE_TYPE_BYTE | PIPE_READMODE_BYTE | PIPE_WAIT,
                1,
                64 * 1024,
                64 * 1024,
                0,
                std::ptr::null_mut(),
            )
        };
        if h == INVALID_HANDLE_VALUE {
            return Err(TransportError::Create(unsafe { GetLastError() }));
        }
        let ok = unsafe { ConnectNamedPipe(h, std::ptr::null_mut()) };
        if ok == 0 {
            let err = unsafe { GetLastError() };
            if err != ERROR_PIPE_CONNECTED {
                unsafe { CloseHandle(h) };
                return Err(TransportError::Connect(err));
            }
        }
        Ok(Self { h, overlapped: false })
    }

    /// Server side with a BOUNDED accept: publish the pipe, then wait at most
    /// `accept_timeout_ms` for a client to connect. Returns
    /// `Err(TransportError::Timeout)` if no client connects in time (so a
    /// renderer that never launches/connects fails honestly instead of
    /// wedging the spawn forever). The accept uses overlapped I/O purely to
    /// make the wait abortable; once a client is connected the returned
    /// `PipeHandle` is opened in OVERLAPPED mode, so all subsequent
    /// `ReadFile`/`WriteFile` go through the synchronous-completion helper
    /// (`sync_read`/`sync_write`) which blocks to completion exactly like the
    /// non-overlapped path — the on-wire behaviour is identical.
    pub fn create_server_timeout(
        name: &str,
        accept_timeout_ms: u32,
    ) -> Result<Self, TransportError> {
        let path = pipe_path_utf16(name);
        let h = unsafe {
            CreateNamedPipeW(
                path.as_ptr(),
                PIPE_ACCESS_DUPLEX | FILE_FLAG_OVERLAPPED,
                PIPE_TYPE_BYTE | PIPE_READMODE_BYTE | PIPE_WAIT,
                1,
                64 * 1024,
                64 * 1024,
                0,
                std::ptr::null_mut(),
            )
        };
        if h == INVALID_HANDLE_VALUE {
            return Err(TransportError::Create(unsafe { GetLastError() }));
        }

        // Manual-reset event the kernel signals when the overlapped connect
        // completes.
        let event = unsafe { CreateEventW(std::ptr::null_mut(), 1, 0, std::ptr::null()) };
        if event.is_null() {
            let e = unsafe { GetLastError() };
            unsafe { CloseHandle(h) };
            return Err(TransportError::Create(e));
        }
        let mut ov: OVERLAPPED = unsafe { core::mem::zeroed() };
        ov.hEvent = event;

        let ok = unsafe { ConnectNamedPipe(h, &mut ov as *mut OVERLAPPED as LPOVERLAPPED) };
        // Overlapped ConnectNamedPipe returns 0; the real status is in
        // GetLastError: ERROR_IO_PENDING (wait on the event) or
        // ERROR_PIPE_CONNECTED (a client beat us to it — already connected).
        if ok == 0 {
            let err = unsafe { GetLastError() };
            match err {
                ERROR_PIPE_CONNECTED => {
                    // Already connected before the overlapped call posted.
                    unsafe { CloseHandle(event) };
                    return Ok(Self { h, overlapped: true });
                }
                ERROR_IO_PENDING => {
                    let wr = unsafe { WaitForSingleObject(event, accept_timeout_ms) };
                    match wr {
                        WAIT_OBJECT_0 => {
                            let mut transferred: DWORD = 0;
                            let gor = unsafe {
                                GetOverlappedResult(h, &mut ov, &mut transferred, 0)
                            };
                            unsafe { CloseHandle(event) };
                            if gor == 0 {
                                let e = unsafe { GetLastError() };
                                unsafe { CloseHandle(h) };
                                return Err(TransportError::Connect(e));
                            }
                            Ok(Self { h, overlapped: true })
                        }
                        WAIT_TIMEOUT => {
                            // Cancel the pending accept, then tear down.
                            unsafe {
                                CancelIoEx(h, &mut ov);
                                CloseHandle(event);
                                CloseHandle(h);
                            }
                            Err(TransportError::Timeout)
                        }
                        _ => {
                            let e = unsafe { GetLastError() };
                            unsafe {
                                CancelIoEx(h, &mut ov);
                                CloseHandle(event);
                                CloseHandle(h);
                            }
                            Err(TransportError::Io(e))
                        }
                    }
                }
                other => {
                    unsafe {
                        CloseHandle(event);
                        CloseHandle(h);
                    }
                    Err(TransportError::Connect(other))
                }
            }
        } else {
            // Rare synchronous success.
            unsafe { CloseHandle(event) };
            Ok(Self { h, overlapped: true })
        }
    }

    /// Client side: open a connection to an existing server pipe. If
    /// the server hasn't created the pipe yet, retries via
    /// `WaitNamedPipeW` for up to `timeout_ms` milliseconds.
    pub fn connect_client(name: &str, timeout_ms: u32) -> Result<Self, TransportError> {
        let path = pipe_path_utf16(name);
        // CreateFileW retries the open if the pipe is busy; we add an
        // optional WaitNamedPipeW for the case where the server side
        // hasn't yet posted ConnectNamedPipe.
        for _ in 0..3 {
            let h = unsafe {
                CreateFileW(
                    path.as_ptr(),
                    GENERIC_READ | GENERIC_WRITE,
                    0,
                    std::ptr::null_mut(),
                    OPEN_EXISTING,
                    FILE_ATTRIBUTE_NORMAL,
                    std::ptr::null_mut(),
                )
            };
            if h != INVALID_HANDLE_VALUE {
                return Ok(Self { h, overlapped: false });
            }
            let err = unsafe { GetLastError() };
            if err == ERROR_PIPE_BUSY {
                let _ = unsafe { WaitNamedPipeW(path.as_ptr(), timeout_ms) };
                continue;
            }
            return Err(TransportError::Connect(err));
        }
        Err(TransportError::Connect(ERROR_PIPE_BUSY))
    }

    /// One overlapped I/O op (read when `is_read`, else write) that BLOCKS to
    /// completion — the synchronous-completion shim for a handle opened with
    /// `FILE_FLAG_OVERLAPPED`. Each call uses its OWN stack `OVERLAPPED` +
    /// event, so a concurrent reader and writer (disjoint pipe directions)
    /// never share overlapped state. Returns the bytes transferred (0 ==
    /// peer closed). Infinite wait — completion bound is enforced upstream by
    /// killing the child, exactly as for the blocking non-overlapped path.
    fn overlapped_io(&self, ptr: *mut c_void, len: DWORD, is_read: bool) -> Result<DWORD, TransportError> {
        let event = unsafe { CreateEventW(std::ptr::null_mut(), 1, 0, std::ptr::null()) };
        if event.is_null() {
            return Err(TransportError::Io(unsafe { GetLastError() }));
        }
        let mut ov: OVERLAPPED = unsafe { core::mem::zeroed() };
        ov.hEvent = event;
        let ok = if is_read {
            unsafe {
                ReadFile(self.h, ptr as LPVOID, len, std::ptr::null_mut(), &mut ov as *mut OVERLAPPED as LPOVERLAPPED)
            }
        } else {
            unsafe {
                WriteFile(self.h, ptr as LPCVOID, len, std::ptr::null_mut(), &mut ov as *mut OVERLAPPED as LPOVERLAPPED)
            }
        };
        if ok == 0 {
            let err = unsafe { GetLastError() };
            if err != ERROR_IO_PENDING {
                unsafe { CloseHandle(event) };
                if err == ERROR_BROKEN_PIPE {
                    return Err(TransportError::Closed);
                }
                return Err(TransportError::Io(err));
            }
        }
        // Block until the kernel signals completion.
        let _ = unsafe { WaitForSingleObject(event, INFINITE_MS) };
        let mut transferred: DWORD = 0;
        let gor = unsafe { GetOverlappedResult(self.h, &mut ov, &mut transferred, 1) };
        unsafe { CloseHandle(event) };
        if gor == 0 {
            let err = unsafe { GetLastError() };
            if err == ERROR_BROKEN_PIPE {
                return Err(TransportError::Closed);
            }
            return Err(TransportError::Io(err));
        }
        Ok(transferred)
    }

    /// Block until all `buf` bytes have been written, or fail. The pipe
    /// is in byte mode so we don't have to worry about message
    /// boundaries on the OS layer; framing comes from the wire header.
    fn write_all(&self, buf: &[u8]) -> Result<(), TransportError> {
        let mut sent: usize = 0;
        while sent < buf.len() {
            let chunk = &buf[sent..];
            let written = if self.overlapped {
                self.overlapped_io(chunk.as_ptr() as *mut c_void, chunk.len() as DWORD, false)?
            } else {
                let mut written: DWORD = 0;
                let ok = unsafe {
                    WriteFile(
                        self.h,
                        chunk.as_ptr() as LPCVOID,
                        chunk.len() as DWORD,
                        &mut written,
                        std::ptr::null_mut(),
                    )
                };
                if ok == 0 {
                    let err = unsafe { GetLastError() };
                    if err == ERROR_BROKEN_PIPE {
                        return Err(TransportError::Closed);
                    }
                    return Err(TransportError::Io(err));
                }
                written
            };
            if written == 0 {
                return Err(TransportError::Closed);
            }
            sent += written as usize;
        }
        Ok(())
    }

    fn read_exact(&self, buf: &mut [u8]) -> Result<(), TransportError> {
        let mut got: usize = 0;
        while got < buf.len() {
            let slice = &mut buf[got..];
            let n = if self.overlapped {
                self.overlapped_io(slice.as_mut_ptr() as *mut c_void, slice.len() as DWORD, true)?
            } else {
                let mut n: DWORD = 0;
                let ok = unsafe {
                    ReadFile(
                        self.h,
                        slice.as_mut_ptr() as LPVOID,
                        slice.len() as DWORD,
                        &mut n,
                        std::ptr::null_mut(),
                    )
                };
                if ok == 0 {
                    let err = unsafe { GetLastError() };
                    if err == ERROR_BROKEN_PIPE {
                        return Err(TransportError::Closed);
                    }
                    return Err(TransportError::Io(err));
                }
                n
            };
            if n == 0 {
                return Err(TransportError::Closed);
            }
            got += n as usize;
        }
        Ok(())
    }

    /// Send a typed message — encode payload, frame with the same
    /// header as the in-process transport, write the whole frame.
    pub fn send<T: Encode>(&self, message_id: u32, payload: &T) -> Result<(), TransportError> {
        let mut payload_w = Writer::new();
        payload.encode(&mut payload_w);
        let payload_bytes = payload_w.into_bytes();
        let mut frame = Writer::with_capacity(12 + payload_bytes.len());
        frame.write_u32(MAGIC);
        frame.write_u32(message_id);
        frame.write_u32(payload_bytes.len() as u32);
        frame.bytes.extend_from_slice(&payload_bytes);
        self.write_all(&frame.into_bytes())
    }

    /// Block until a full framed message arrives. Returns (id, value).
    pub fn recv<T: Decode>(&self) -> Result<(u32, T), TransportError> {
        let mut header = [0u8; 12];
        self.read_exact(&mut header)?;
        let mut r = Reader::new(&header);
        let magic = r.read_u32().map_err(|_| TransportError::BadFrame)?;
        if magic != MAGIC {
            return Err(TransportError::BadFrame);
        }
        let id = r.read_u32().map_err(|_| TransportError::BadFrame)?;
        let payload_len = r.read_u32().map_err(|_| TransportError::BadFrame)? as usize;
        let mut payload = vec![0u8; payload_len];
        self.read_exact(&mut payload)?;
        let mut pr = Reader::new(&payload);
        let v = T::decode(&mut pr).map_err(|_| TransportError::Codec)?;
        Ok((id, v))
    }
}

fn map_transport_recv_error(err: TransportError) -> RecvError {
    match err {
        TransportError::Closed => RecvError::Closed,
        TransportError::BadFrame => RecvError::BadFrame,
        TransportError::Codec => RecvError::BadMessage,
        _ => RecvError::Closed,
    }
}

/// Bridge a `PipeHandle` to the same shape as the in-process
/// `Endpoint`: `send` and `try_recv` mirror that API. Caller picks
/// `try_recv` over `recv` when the surrounding loop is event-driven
/// and would deadlock on a blocking read. **For now `try_recv` is a
/// trivial wrapper around the blocking call** — a real non-blocking
/// variant lands when we wire up `IO_COMPLETION_PORT`.
pub struct NamedPipeEndpoint {
    handle: PipeHandle,
}

impl NamedPipeEndpoint {
    pub fn server(name: &str) -> Result<Self, TransportError> {
        Ok(Self {
            handle: PipeHandle::create_server(name)?,
        })
    }

    /// Server side with a BOUNDED accept (`accept_timeout_ms`). Returns
    /// `Err(TransportError::Timeout)` when no client connects in time, so a
    /// renderer that never launches fails honestly instead of hanging the
    /// spawn forever.
    pub fn server_timeout(name: &str, accept_timeout_ms: u32) -> Result<Self, TransportError> {
        Ok(Self {
            handle: PipeHandle::create_server_timeout(name, accept_timeout_ms)?,
        })
    }

    pub fn client(name: &str, timeout_ms: u32) -> Result<Self, TransportError> {
        Ok(Self {
            handle: PipeHandle::connect_client(name, timeout_ms)?,
        })
    }

    pub fn send<T: Encode>(&self, message_id: u32, payload: &T) -> Result<(), SendError> {
        // Cross-process send errors collapse to PeerClosed at the
        // public boundary; callers can inspect TransportError via the
        // raw `PipeHandle` if they need finer detail.
        self.handle
            .send(message_id, payload)
            .map_err(|_| SendError::PeerClosed)
    }

    pub fn recv<T: Decode>(&self) -> Result<(u32, T), RecvError> {
        self.handle.recv::<T>().map_err(map_transport_recv_error)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pipe_path_format() {
        // Spot-check the UTF-16 prefix is exactly r"\\.\pipe\".
        let utf16 = pipe_path_utf16("foo");
        let prefix: Vec<u16> = "\\\\.\\pipe\\foo".encode_utf16().collect();
        assert_eq!(&utf16[..prefix.len()], &prefix[..]);
        assert_eq!(*utf16.last().unwrap(), 0u16);
    }

    #[test]
    fn server_client_round_trip_in_threads() {
        // Spawn a server thread; main thread acts as client. End-to-end
        // proves the FFI is correct without launching a real second
        // process. Pipe name is uniquified by PID to avoid colliding
        // with stale pipes from earlier crashed test runs.
        let pid = std::process::id();
        let name = format!("tbrm_test_{pid}_round_trip");
        let server_name = name.clone();
        let server = std::thread::spawn(move || {
            let ep = NamedPipeEndpoint::server(&server_name).unwrap();
            let (id, msg): (u32, String) = ep.recv().unwrap();
            assert_eq!(id, 7);
            assert_eq!(msg, "ping");
            ep.send(8, &"pong".to_string()).unwrap();
        });
        // Give the server a beat to publish the pipe.
        std::thread::sleep(std::time::Duration::from_millis(50));
        let client = NamedPipeEndpoint::client(&name, 2_000).unwrap();
        client.send(7, &"ping".to_string()).unwrap();
        let (id, msg): (u32, String) = client.recv().unwrap();
        assert_eq!(id, 8);
        assert_eq!(msg, "pong");
        server.join().unwrap();
    }

    #[test]
    fn codec_errors_map_to_bad_message() {
        assert_eq!(
            map_transport_recv_error(TransportError::Codec),
            RecvError::BadMessage
        );
        assert_eq!(
            map_transport_recv_error(TransportError::BadFrame),
            RecvError::BadFrame
        );
    }

    /// A BOUNDED accept with no client connecting must return `Timeout`
    /// promptly — NOT hang forever. This is the IPC-level root-cause guard for
    /// the cross-origin-swap deadlock: a renderer that never connects fails
    /// honestly instead of wedging the spawn.
    #[test]
    fn bounded_accept_times_out_when_no_client_connects() {
        let pid = std::process::id();
        let name = format!("tbrm_test_{pid}_accept_timeout");
        let start = std::time::Instant::now();
        let result = NamedPipeEndpoint::server_timeout(&name, 300);
        let elapsed = start.elapsed();
        let timed_out = matches!(result, Err(TransportError::Timeout));
        assert!(timed_out, "no client → Timeout expected");
        // Returned within a small multiple of the budget (not hung).
        assert!(
            elapsed < std::time::Duration::from_secs(5),
            "bounded accept must return promptly, took {elapsed:?}"
        );
    }

    /// A bounded-accept server whose client DOES connect in time completes the
    /// accept AND round-trips messages correctly. Proves the overlapped handle
    /// the timed path returns reads/writes byte-identically to the blocking
    /// path (the `sync_read`/`sync_write` shim is wire-correct).
    #[test]
    fn bounded_accept_round_trips_when_client_connects() {
        let pid = std::process::id();
        let name = format!("tbrm_test_{pid}_accept_ok");
        let server_name = name.clone();
        let server = std::thread::spawn(move || {
            let ep = NamedPipeEndpoint::server_timeout(&server_name, 5_000).unwrap();
            let (id, msg): (u32, String) = ep.recv().unwrap();
            assert_eq!(id, 11);
            assert_eq!(msg, "hello-overlapped");
            ep.send(12, &"ack-overlapped".to_string()).unwrap();
        });
        std::thread::sleep(std::time::Duration::from_millis(50));
        let client = NamedPipeEndpoint::client(&name, 2_000).unwrap();
        client.send(11, &"hello-overlapped".to_string()).unwrap();
        let (id, msg): (u32, String) = client.recv().unwrap();
        assert_eq!(id, 12);
        assert_eq!(msg, "ack-overlapped");
        server.join().unwrap();
    }
}
