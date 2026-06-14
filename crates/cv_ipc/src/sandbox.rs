//! Windows Job-Object based sandbox primitives.
//!
//! A `JobObject` is a kernel container for one or more processes that
//! the parent can use to enforce hard limits:
//!
//! - `kill_on_close = true` — when the parent process exits or drops
//!   the job handle, the kernel forcibly terminates every process
//!   still inside the job. This is the "renderer dies if browser
//!   crashes" guarantee Chromium relies on.
//! - `active_process_limit` — caps the number of processes the job
//!   may hold. Setting it to 1 prevents the child from `fork()`-ing
//!   helper processes via `CreateProcessW`.
//! - `memory_limit_bytes` — kernel-enforced upper bound on the job's
//!   working-set + commit charge.
//!
//! Use: build the policy with `JobObjectBuilder`, attach a freshly
//! spawned `ChildProcess` via `attach`. Drop the `JobObject` to take
//! down the children (if `kill_on_close` was set).

#![cfg(target_os = "windows")]
#![allow(non_camel_case_types, non_snake_case, clippy::upper_case_acronyms)]
#![allow(unreachable_pub, missing_debug_implementations, dead_code)]

use core::ffi::c_void;

use crate::spawn::ChildProcess;

type HANDLE = *mut c_void;
type BOOL = i32;
type DWORD = u32;
type LPCWSTR = *const u16;

const INVALID_HANDLE_VALUE: HANDLE = -1isize as HANDLE;

const JOB_OBJECT_LIMIT_ACTIVE_PROCESS: DWORD = 0x0000_0008;
const JOB_OBJECT_LIMIT_PROCESS_MEMORY: DWORD = 0x0000_0100;
const JOB_OBJECT_LIMIT_JOB_MEMORY: DWORD = 0x0000_0200;
const JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE: DWORD = 0x0000_2000;
const JOB_OBJECT_LIMIT_BREAKAWAY_OK: DWORD = 0x0000_0800;

#[repr(C)]
#[derive(Copy, Clone, Default)]
struct IO_COUNTERS {
    ReadOperationCount: u64,
    WriteOperationCount: u64,
    OtherOperationCount: u64,
    ReadTransferCount: u64,
    WriteTransferCount: u64,
    OtherTransferCount: u64,
}

#[repr(C)]
#[derive(Copy, Clone, Default)]
struct JOBOBJECT_BASIC_LIMIT_INFORMATION {
    PerProcessUserTimeLimit: i64,
    PerJobUserTimeLimit: i64,
    LimitFlags: DWORD,
    MinimumWorkingSetSize: usize,
    MaximumWorkingSetSize: usize,
    ActiveProcessLimit: DWORD,
    Affinity: usize,
    PriorityClass: DWORD,
    SchedulingClass: DWORD,
}

#[repr(C)]
#[derive(Copy, Clone, Default)]
struct JOBOBJECT_EXTENDED_LIMIT_INFORMATION {
    BasicLimitInformation: JOBOBJECT_BASIC_LIMIT_INFORMATION,
    IoInfo: IO_COUNTERS,
    ProcessMemoryLimit: usize,
    JobMemoryLimit: usize,
    PeakProcessMemoryUsed: usize,
    PeakJobMemoryUsed: usize,
}

// JobObjectExtendedLimitInformation = 9
const JOB_OBJECT_INFO_EXTENDED_LIMIT_INFORMATION: i32 = 9;

#[link(name = "kernel32")]
unsafe extern "system" {
    fn CreateJobObjectW(lpJobAttributes: *mut c_void, lpName: LPCWSTR) -> HANDLE;

    fn SetInformationJobObject(
        hJob: HANDLE,
        JobObjectInformationClass: i32,
        lpJobObjectInformation: *const c_void,
        cbJobObjectInformationLength: DWORD,
    ) -> BOOL;

    fn AssignProcessToJobObject(hJob: HANDLE, hProcess: HANDLE) -> BOOL;

    fn CloseHandle(hObject: HANDLE) -> BOOL;
    fn GetLastError() -> DWORD;
}

#[derive(Debug, PartialEq, Eq)]
pub enum SandboxError {
    Create(u32),
    SetInfo(u32),
    Assign(u32),
}

impl std::fmt::Display for SandboxError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Create(e) => write!(f, "CreateJobObject failed ({e})"),
            Self::SetInfo(e) => write!(f, "SetInformationJobObject failed ({e})"),
            Self::Assign(e) => write!(f, "AssignProcessToJobObject failed ({e})"),
        }
    }
}

impl std::error::Error for SandboxError {}

/// Builder for a Job Object's policy. Sensible defaults — assemble
/// the constraints you actually need, then `.build()`.
#[derive(Debug, Default)]
pub struct JobObjectBuilder {
    kill_on_close: bool,
    active_process_limit: Option<u32>,
    memory_limit_bytes: Option<usize>,
}

impl JobObjectBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    /// Terminate every process in the job when the last handle is
    /// closed. This is the primary "browser dies → renderer dies"
    /// guarantee.
    pub fn kill_on_close(mut self, yes: bool) -> Self {
        self.kill_on_close = yes;
        self
    }

    /// Cap the number of processes the job may contain. Use 1 to
    /// prevent the child from spawning further processes.
    pub fn active_process_limit(mut self, limit: u32) -> Self {
        self.active_process_limit = Some(limit);
        self
    }

    /// Per-job committed-memory ceiling. Kernel kills the offending
    /// process if it pushes past this.
    pub fn memory_limit_bytes(mut self, bytes: usize) -> Self {
        self.memory_limit_bytes = Some(bytes);
        self
    }

    /// Create the kernel object and apply the policy.
    pub fn build(self) -> Result<JobObject, SandboxError> {
        let h = unsafe { CreateJobObjectW(core::ptr::null_mut(), core::ptr::null()) };
        if h.is_null() || h == INVALID_HANDLE_VALUE {
            return Err(SandboxError::Create(unsafe { GetLastError() }));
        }
        let mut info: JOBOBJECT_EXTENDED_LIMIT_INFORMATION = unsafe { core::mem::zeroed() };
        let mut flags: DWORD = 0;
        if self.kill_on_close {
            flags |= JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
        }
        if let Some(n) = self.active_process_limit {
            flags |= JOB_OBJECT_LIMIT_ACTIVE_PROCESS;
            info.BasicLimitInformation.ActiveProcessLimit = n;
        }
        if let Some(n) = self.memory_limit_bytes {
            flags |= JOB_OBJECT_LIMIT_JOB_MEMORY;
            info.JobMemoryLimit = n;
        }
        info.BasicLimitInformation.LimitFlags = flags;
        let ok = unsafe {
            SetInformationJobObject(
                h,
                JOB_OBJECT_INFO_EXTENDED_LIMIT_INFORMATION,
                &info as *const _ as *const c_void,
                core::mem::size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as DWORD,
            )
        };
        if ok == 0 {
            let err = unsafe { GetLastError() };
            unsafe { CloseHandle(h) };
            return Err(SandboxError::SetInfo(err));
        }
        Ok(JobObject { h })
    }
}

pub struct JobObject {
    h: HANDLE,
}

unsafe impl Send for JobObject {}

impl std::fmt::Debug for JobObject {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("JobObject").finish_non_exhaustive()
    }
}

impl Drop for JobObject {
    fn drop(&mut self) {
        if !self.h.is_null() && self.h != INVALID_HANDLE_VALUE {
            unsafe { CloseHandle(self.h) };
        }
    }
}

impl JobObject {
    /// Add a freshly spawned process to this job. Most browser
    /// deployments do this *immediately* after spawn so the child has
    /// no chance to fork before the limits engage.
    pub fn attach(&self, child: &ChildProcess) -> Result<(), SandboxError> {
        // We need the child's process handle. The `ChildProcess` API
        // hides it for safety; expose just enough via a method on the
        // spawn module so we don't leak `HANDLE` across the public
        // boundary unnecessarily.
        let ok = unsafe { AssignProcessToJobObject(self.h, crate::spawn::process_handle(child)) };
        if ok == 0 {
            return Err(SandboxError::Assign(unsafe { GetLastError() }));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_kill_on_close_job() {
        let job = JobObjectBuilder::new()
            .kill_on_close(true)
            .active_process_limit(1)
            .memory_limit_bytes(64 * 1024 * 1024)
            .build()
            .unwrap();
        // Just ensure the build succeeded and Drop works.
        drop(job);
    }

    #[test]
    fn attach_then_kill_via_drop() {
        // Spawn a child that would otherwise run for ~3s, attach it to
        // a kill_on_close job, drop the job, then wait. The child
        // should exit promptly (within the grace period, definitely
        // under the natural ping duration).
        let child = ChildProcess::spawn("cmd.exe /c ping localhost -n 5", false).unwrap();
        let job = JobObjectBuilder::new().kill_on_close(true).build().unwrap();
        job.attach(&child).unwrap();
        drop(job);
        // Give Windows a moment to enact the kill.
        let r = child.try_wait_for(1_000).unwrap();
        // Either the kernel killed it during the wait window, or we
        // poll once more — `kill_on_close` is supposed to be prompt,
        // so this should generally come back as Some.
        assert!(r.is_some(), "child should have exited after job drop");
    }
}
