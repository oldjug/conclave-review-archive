//! Blocking TCP socket over WinSock 2.

use crate::dns::{IpAddr, ResolvedAddr};
use crate::sys;
use std::sync::Once;

#[derive(Debug)]
pub enum SocketError {
    StartupFailed(i32),
    SocketFailed(i32),
    ConnectFailed(i32),
    SendFailed(i32),
    RecvFailed(i32),
    Closed,
}

static WSA_INIT: Once = Once::new();

pub fn ensure_wsa_started() {
    WSA_INIT.call_once(|| {
        let mut data = sys::WSADATA::zeroed();
        let rc = unsafe { sys::WSAStartup(0x0202, &raw mut data) }; // 2.2
        assert!(rc == 0, "WSAStartup failed: {rc}");
    });
}

#[derive(Debug)]
pub struct Socket {
    s: sys::SOCKET,
}

impl Socket {
    pub fn connect(addr: &ResolvedAddr) -> Result<Self, SocketError> {
        Self::connect_with_timeout(addr, 30_000)
    }

    /// Connect with `timeout_ms` covering both the TCP connect step and
    /// subsequent recv/send operations. Used by stylesheet/image fetches
    /// that need to give up quickly when a CDN doesn't respond.
    pub fn connect_with_timeout(addr: &ResolvedAddr, timeout_ms: u32) -> Result<Self, SocketError> {
        ensure_wsa_started();
        let family = match addr.ip {
            IpAddr::V4(_) => sys::AF_INET,
            IpAddr::V6(_) => sys::AF_INET6,
        };
        let s = unsafe { sys::socket(family, sys::SOCK_STREAM, sys::IPPROTO_TCP) };
        if s == sys::INVALID_SOCKET {
            return Err(SocketError::SocketFailed(unsafe { sys::WSAGetLastError() }));
        }
        let sock = Self { s };

        // Recv/send timeouts. These apply AFTER connect.
        let to_bytes = timeout_ms.to_le_bytes();
        unsafe {
            sys::setsockopt(s, sys::SOL_SOCKET, sys::SO_RCVTIMEO, to_bytes.as_ptr(), 4);
            sys::setsockopt(s, sys::SOL_SOCKET, sys::SO_SNDTIMEO, to_bytes.as_ptr(), 4);
        }

        // TCP_NODELAY — disable Nagle's algorithm. See sys.rs comment for
        // the BoringSSL handshake-interleave issue this avoids. Without
        // this, our TLS 1.3 client Finished and the first HTTP/1.1 GET
        // record can arrive in a single TCP segment on a Cloudflare edge,
        // tripping `tls_open_record`'s
        // `tls_has_unprocessed_handshake_data` guard → fatal
        // `unexpected_message` alert. thehindu.com is the canonical
        // reproducer.
        let one_bytes: [u8; 4] = 1u32.to_le_bytes();
        unsafe {
            sys::setsockopt(s, sys::IPPROTO_TCP, sys::TCP_NODELAY, one_bytes.as_ptr(), 4);
        }

        // Connect with timeout: flip to non-blocking, call connect (which
        // returns WSAEWOULDBLOCK immediately), select() for write-ready,
        // check SO_ERROR, then flip back to blocking.
        let mut nonblock: u32 = 1;
        unsafe { sys::ioctlsocket(s, sys::FIONBIO as i32, &raw mut nonblock) };

        let rc = match addr.ip {
            IpAddr::V4(ip) => {
                let sa = sys::sockaddr_in {
                    sin_family: sys::AF_INET as u16,
                    sin_port: addr.port.to_be(),
                    sin_addr: ip,
                    sin_zero: [0; 8],
                };
                unsafe {
                    sys::connect(
                        s,
                        (&raw const sa).cast::<sys::sockaddr>(),
                        std::mem::size_of::<sys::sockaddr_in>() as i32,
                    )
                }
            }
            IpAddr::V6(ip) => {
                let sa = sys::sockaddr_in6 {
                    sin6_family: sys::AF_INET6 as u16,
                    sin6_port: addr.port.to_be(),
                    sin6_flowinfo: 0,
                    sin6_addr: ip,
                    sin6_scope_id: 0,
                };
                unsafe {
                    sys::connect(
                        s,
                        (&raw const sa).cast::<sys::sockaddr>(),
                        std::mem::size_of::<sys::sockaddr_in6>() as i32,
                    )
                }
            }
        };

        if rc == sys::SOCKET_ERROR {
            let err = unsafe { sys::WSAGetLastError() };
            if err != sys::WSAEWOULDBLOCK {
                return Err(SocketError::ConnectFailed(err));
            }
            // Wait for the socket to become writable, or time out.
            let mut wfds = sys::fd_set::zeroed();
            wfds.fd_count = 1;
            wfds.fd_array[0] = s;
            let mut efds = sys::fd_set::zeroed();
            efds.fd_count = 1;
            efds.fd_array[0] = s;
            let tv = sys::timeval {
                tv_sec: (timeout_ms / 1000) as i32,
                tv_usec: ((timeout_ms % 1000) * 1000) as i32,
            };
            let n = unsafe {
                sys::select(
                    0,
                    core::ptr::null_mut(),
                    &raw mut wfds,
                    &raw mut efds,
                    &raw const tv,
                )
            };
            if n == 0 {
                return Err(SocketError::ConnectFailed(-1)); // timed out
            }
            if n == sys::SOCKET_ERROR {
                return Err(SocketError::ConnectFailed(unsafe {
                    sys::WSAGetLastError()
                }));
            }
            // Writable + non-zero SO_ERROR = async connect failure
            // (refused, unreachable, etc.).
            let mut so_err: i32 = 0;
            let mut len: i32 = 4;
            unsafe {
                sys::getsockopt(
                    s,
                    sys::SOL_SOCKET,
                    sys::SO_ERROR,
                    (&raw mut so_err).cast::<u8>(),
                    &raw mut len,
                );
            }
            if so_err != 0 {
                return Err(SocketError::ConnectFailed(so_err));
            }
        }

        // Back to blocking — recv/send timeouts take over from here.
        let mut block: u32 = 0;
        unsafe { sys::ioctlsocket(s, sys::FIONBIO as i32, &raw mut block) };

        Ok(sock)
    }

    pub fn write_all(&mut self, mut data: &[u8]) -> Result<(), SocketError> {
        while !data.is_empty() {
            let n = unsafe {
                sys::send(
                    self.s,
                    data.as_ptr(),
                    data.len().min(i32::MAX as usize) as i32,
                    0,
                )
            };
            if n == sys::SOCKET_ERROR {
                return Err(SocketError::SendFailed(unsafe { sys::WSAGetLastError() }));
            }
            if n == 0 {
                return Err(SocketError::Closed);
            }
            data = &data[n as usize..];
        }
        Ok(())
    }

    pub fn read(&mut self, buf: &mut [u8]) -> Result<usize, SocketError> {
        let n = unsafe {
            sys::recv(
                self.s,
                buf.as_mut_ptr(),
                buf.len().min(i32::MAX as usize) as i32,
                0,
            )
        };
        if n == sys::SOCKET_ERROR {
            return Err(SocketError::RecvFailed(unsafe { sys::WSAGetLastError() }));
        }
        Ok(n as usize)
    }

    /// Read up to `buf.len()` bytes with a short SO_RCVTIMEO. Returns 0
    /// if the timeout expires with no data. Used by the TLS 1.3 client
    /// to drain server post-handshake messages (NewSessionTicket) before
    /// sending app data — guarantees the server's record loop has fully
    /// processed our Finished so a subsequent app-data record doesn't
    /// trip BoringSSL's `tls_has_unprocessed_handshake_data` guard.
    pub fn read_with_timeout(
        &mut self,
        buf: &mut [u8],
        timeout_ms: u32,
    ) -> Result<usize, SocketError> {
        let to_bytes = timeout_ms.to_le_bytes();
        unsafe {
            sys::setsockopt(
                self.s,
                sys::SOL_SOCKET,
                sys::SO_RCVTIMEO,
                to_bytes.as_ptr(),
                4,
            );
        }
        let n = unsafe {
            sys::recv(
                self.s,
                buf.as_mut_ptr(),
                buf.len().min(i32::MAX as usize) as i32,
                0,
            )
        };
        // Whatever happens, restore a long-ish default timeout.
        let default_to: [u8; 4] = 30_000_u32.to_le_bytes();
        unsafe {
            sys::setsockopt(
                self.s,
                sys::SOL_SOCKET,
                sys::SO_RCVTIMEO,
                default_to.as_ptr(),
                4,
            );
        }
        if n == sys::SOCKET_ERROR {
            let e = unsafe { sys::WSAGetLastError() };
            // WSAETIMEDOUT is 10060 — treat as "no data" rather than error.
            if e == 10060 {
                return Ok(0);
            }
            return Err(SocketError::RecvFailed(e));
        }
        Ok(n as usize)
    }

    /// Override the receive timeout (SO_RCVTIMEO) on an open socket. Used
    /// to give a reused keep-alive socket a short leash: a warm connection
    /// answers fast, so if it doesn't we want to give up quickly and
    /// re-dial rather than stall for the full per-request budget.
    pub fn set_read_timeout_ms(&self, ms: u32) {
        let to = ms.to_le_bytes();
        unsafe {
            sys::setsockopt(self.s, sys::SOL_SOCKET, sys::SO_RCVTIMEO, to.as_ptr(), 4);
        }
    }

    /// Cheap, non-blocking liveness check before reusing a pooled
    /// keep-alive socket. A connection we left at a clean HTTP message
    /// boundary has NOTHING for us to read while idle, so `select()` with
    /// a zero timeout must report it not-readable. If it IS readable the
    /// peer has either closed (a pending FIN makes the socket readable
    /// with a 0-byte recv), reset it (RST → readable error), or pushed
    /// unsolicited data — all reasons to discard rather than write a
    /// request into a dead/confused socket and stall until recv times
    /// out. Returns true only when the socket looks safe to reuse.
    pub fn is_reuse_safe(&self) -> bool {
        let mut rfds = sys::fd_set::zeroed();
        rfds.fd_count = 1;
        rfds.fd_array[0] = self.s;
        // Zero timeout → poll, never block.
        let tv = sys::timeval {
            tv_sec: 0,
            tv_usec: 0,
        };
        let rc = unsafe {
            // nfds is ignored on WinSock; the fd_set count is what matters.
            sys::select(
                1,
                &raw mut rfds,
                core::ptr::null_mut(),
                core::ptr::null_mut(),
                &raw const tv,
            )
        };
        // 0 → nothing readable → healthy idle socket. Anything else
        // (1 = readable = FIN/RST/stray data, or SOCKET_ERROR) → unsafe.
        rc == 0
    }

    /// Read until EOF (the server closes the connection or `read` returns 0).
    pub fn read_to_end(&mut self, dst: &mut Vec<u8>) -> Result<(), SocketError> {
        let mut buf = [0u8; 16 * 1024];
        loop {
            let n = self.read(&mut buf)?;
            if n == 0 {
                return Ok(());
            }
            dst.extend_from_slice(&buf[..n]);
        }
    }
}

impl Drop for Socket {
    fn drop(&mut self) {
        if self.s != sys::INVALID_SOCKET {
            unsafe {
                sys::shutdown(self.s, sys::SHUTDOWN_BOTH);
                sys::closesocket(self.s);
            }
        }
    }
}
