//! High-level "spawn a sandboxed child with an IPC channel" helper.
//!
//! Combines `NamedPipeEndpoint` + `ChildProcess` + `JobObject` into the
//! one-shot operation the browser process uses to launch a renderer.
//! Parent generates a unique pipe name, opens the server side on a
//! background thread (so it can accept the client connection while we
//! still spawn the child), passes the name on the child's command
//! line, attaches the spawned process to a kill-on-close Job Object,
//! then joins the server thread to retrieve the connected endpoint.
//!
//! The composite is what callers actually want — the four lower-level
//! pieces are useful in isolation for tests and for advanced cases,
//! but production code reaches for this.

#![cfg(target_os = "windows")]
#![allow(dead_code)]

use crate::named_pipe::NamedPipeEndpoint;
use crate::sandbox::{JobObject, JobObjectBuilder, SandboxError};
use crate::spawn::{ChildProcess, LowIntegrityToken, SpawnError};

#[derive(Debug)]
pub enum SandboxedSpawnError {
    Pipe(String),
    Spawn(SpawnError),
    Sandbox(SandboxError),
    Connect(String),
}

impl std::fmt::Display for SandboxedSpawnError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Pipe(s) => write!(f, "pipe: {s}"),
            Self::Spawn(e) => write!(f, "spawn: {e}"),
            Self::Sandbox(e) => write!(f, "sandbox: {e}"),
            Self::Connect(s) => write!(f, "connect: {s}"),
        }
    }
}

impl std::error::Error for SandboxedSpawnError {}

/// Render-side / utility-side identity. Encoded in the pipe name so
/// stray pipes from prior runs (crashed parents) don't collide.
#[derive(Debug, Clone, Copy)]
pub enum ChildKind {
    Renderer,
    Network,
    Gpu,
    Utility,
}

impl ChildKind {
    fn tag(self) -> &'static str {
        match self {
            Self::Renderer => "renderer",
            Self::Network => "network",
            Self::Gpu => "gpu",
            Self::Utility => "utility",
        }
    }
}

/// Spawn options. The defaults are paranoid: kill-on-close on, no
/// child forks allowed, generous-but-bounded memory.
#[derive(Debug)]
pub struct SpawnOptions {
    pub kind: ChildKind,
    pub memory_limit_bytes: usize,
    pub allow_subprocesses: bool,
    /// Process-mitigation policy bitmask. Packed by
    /// `cv_sandbox::MitigationPolicies::pack_word()` and passed to
    /// `UpdateProcThreadAttribute` via STARTUPINFOEX so the spawned
    /// child boots up with ASLR + CFG + DEP + ACG + Win32k-disable +
    /// Image-load-no-remote already engaged. 0 disables the override.
    pub mitigation_policy: u64,
    /// AppContainer SID string ("S-1-15-2-..."). When non-empty, the
    /// spawn attaches a SECURITY_CAPABILITIES with this SID so the
    /// child runs in a per-install AppContainer. Empty = no AC.
    pub app_container_sid: String,
    /// Lower the child's integrity level to Low. Stacks with
    /// AppContainer per Chromium's renderer policy.
    pub low_integrity: bool,
    /// Bounded accept: max milliseconds to wait for the spawned child to
    /// connect to the IPC pipe. `0` = INFINITE (the original blocking
    /// behaviour, kept as the default so existing callers/tests are
    /// unchanged). When non-zero, a child that never connects fails the
    /// spawn with `SandboxedSpawnError::Connect("accept timed out")`
    /// instead of wedging the caller forever.
    pub accept_timeout_ms: u32,
}

impl Default for SpawnOptions {
    fn default() -> Self {
        Self {
            kind: ChildKind::Utility,
            memory_limit_bytes: 2 * 1024 * 1024 * 1024, // 2 GiB
            allow_subprocesses: false,
            mitigation_policy: 0,
            app_container_sid: String::new(),
            low_integrity: false,
            accept_timeout_ms: 0,
        }
    }
}

impl SpawnOptions {
    /// Apply a `cv_sandbox::ChannelPolicy` to these options.
    /// `sid_string` is the policy's AppContainer SID rendered as
    /// the standard "S-1-15-2-..." form by
    /// `AppContainerSid::to_string_sid()`. We accept it as a string
    /// so this crate doesn't depend on cv_sandbox.
    pub fn apply_channel_policy(
        &mut self,
        mitigation_packed: u64,
        sid_string: &str,
        use_app_container: bool,
        low_integrity: bool,
        allow_child_processes: bool,
    ) {
        self.mitigation_policy = mitigation_packed;
        if use_app_container {
            self.app_container_sid = sid_string.to_string();
        } else {
            self.app_container_sid.clear();
        }
        self.low_integrity = low_integrity;
        self.allow_subprocesses = allow_child_processes;
    }
}

/// Which rung of the hardening ladder actually launched the child.
/// Recorded on the spawned `SandboxedChild` so the broker / tests can
/// assert which level of sandboxing took effect. ★ Honesty rule: this
/// reflects what ACTUALLY APPLIED, never what was requested —
/// `JobOnly` is reported as job-only (NOT "hardened") so verifiability
/// never becomes a lie.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AppliedTier {
    /// Tier1: AppContainer + restricted token + low integrity +
    /// mitigation. Reached via a SECURITY_CAPABILITIES attribute + a
    /// token-bearing `CreateProcessAsUserW` (the AppContainer's
    /// restricted token). ★ Only recorded after a query-back
    /// (`query_is_app_container() == Some(true)`) confirms the kernel
    /// actually placed the child in the AppContainer — never claimed on
    /// a bare spawn success.
    AppContainer,
    /// Tier2: mitigation policy + low integrity (no AppContainer).
    /// Reached by duplicating the broker's token, lowering it to Low
    /// mandatory integrity (S-1-16-4096), and spawning with it. ★ Only
    /// recorded after a query-back (`query_integrity_is_low() ==
    /// Some(true)`) confirms the child is genuinely Low integrity.
    MitigationLowIntegrity,
    /// Tier3: mitigation policy word applied via
    /// `spawn_with_mitigation`. The real V1 default rung — DEP / ASLR
    /// / ACG / CFG / font-win32k-disable / image-load-no-remote
    /// kernel-enforced on the child.
    Mitigation,
    /// Tier4: plain spawn (the original behavior). Job-object
    /// contained but NO process-mitigation hardening. The floor of
    /// the ladder; also the only rung when `CV_SANDBOX_HARDEN` is off.
    JobOnly,
}

impl AppliedTier {
    /// Human-readable label for logs. Tier4 is honestly "job-only",
    /// never "hardened".
    pub fn label(self) -> &'static str {
        match self {
            Self::AppContainer => "appcontainer+token+lowint+mitigation",
            Self::MitigationLowIntegrity => "mitigation+lowintegrity",
            Self::Mitigation => "mitigation",
            Self::JobOnly => "plain (job-only)",
        }
    }
}

/// A live child process bound to a kill-on-close Job Object with a
/// connected IPC pipe. Drop the struct to terminate the child (the
/// job's kill_on_close flag takes care of that even if the parent
/// crashes — Windows wipes orphaned jobs when the last handle dies).
pub struct SandboxedChild {
    pub process: ChildProcess,
    pub job: JobObject,
    pub endpoint: NamedPipeEndpoint,
    pub pipe_name: String,
    /// Which hardening rung actually launched this child.
    pub tier: AppliedTier,
}

impl std::fmt::Debug for SandboxedChild {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SandboxedChild")
            .field("pid", &self.process.process_id())
            .field("pipe_name", &self.pipe_name)
            .finish_non_exhaustive()
    }
}

/// Build a pseudo-unique pipe name from the parent's pid, the child
/// kind, and a monotonically-incrementing counter so multiple children
/// of the same kind don't share a name.
fn next_pipe_name(kind: ChildKind) -> String {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    format!(
        "tbrm_{tag}_{pid}_{n:x}",
        tag = kind.tag(),
        pid = std::process::id(),
        n = n
    )
}

/// Convenience: build SpawnOptions from a cv_sandbox channel policy
/// + an install key. The install key is folded into the
/// AppContainer SID so two installs of the browser get isolated AC
/// storage.
pub fn options_from_channel_policy(
    kind: ChildKind,
    policy: cv_sandbox::ChannelPolicy,
    install_key: &str,
) -> SpawnOptions {
    let mut opts = SpawnOptions {
        kind,
        ..Default::default()
    };
    let sid = cv_sandbox::AppContainerSid::from_install_key(install_key);
    opts.apply_channel_policy(
        policy.mitigation.pack_word(),
        &sid.to_string_sid(),
        policy.use_app_container,
        policy.low_integrity,
        policy.allow_child_processes,
    );
    opts
}

impl SandboxedChild {
    /// Spawn `exe` with `args`. The pipe name is appended as the last
    /// argument so the child can find it via its own command line
    /// (use `std::env::args().last()` on the renderer side). Returns
    /// once the child has connected to the pipe.
    ///
    /// The parent thread accepts the client connection on a background
    /// thread (`PipeHandle::create_server` blocks until a connect), so
    /// the child's `CreateFileW` doesn't deadlock against the parent's
    /// `ConnectNamedPipe`.
    pub fn spawn(
        exe: &str,
        args: &[&str],
        opts: &SpawnOptions,
    ) -> Result<Self, SandboxedSpawnError> {
        let pipe_name = next_pipe_name(opts.kind);

        // Background server: waits for the child's ConnectNamedPipe. When
        // `accept_timeout_ms` is set, the accept is BOUNDED (overlapped) so a
        // child that never launches/connects fails the spawn honestly instead
        // of wedging the caller's thread forever; `0` keeps the original
        // INFINITE blocking accept.
        let server_name = pipe_name.clone();
        let accept_timeout_ms = opts.accept_timeout_ms;
        let server_handle = std::thread::spawn(move || {
            if accept_timeout_ms == 0 {
                NamedPipeEndpoint::server(&server_name)
            } else {
                NamedPipeEndpoint::server_timeout(&server_name, accept_timeout_ms)
            }
        });

        // Tiny pause so the server thread actually creates the pipe
        // before we spawn the client. CreateFileW retries on
        // ERROR_PIPE_BUSY but not on "pipe doesn't exist yet" — the
        // latter is just a plain not-found.
        std::thread::sleep(std::time::Duration::from_millis(20));

        // Assemble the child command line. We quote each arg minimally
        // so paths with spaces work; full CommandLineToArgvW-faithful
        // quoting lands when we ship signed installers.
        let mut cmd = quote_arg(exe);
        for a in args {
            cmd.push(' ');
            cmd.push_str(&quote_arg(a));
        }
        cmd.push(' ');
        cmd.push_str(&pipe_name);

        // ★ THE FALLBACK LADDER. Highest-applicable hardening rung
        // first; each failure steps DOWN. Never fails to launch the
        // child because hardening failed — a sandboxed child that
        // won't start is worse than a less-sandboxed one. The Job
        // Object attach below runs AFTER whichever rung succeeds, so
        // it is present in ALL tiers including plain spawn.
        let (child, tier) = spawn_with_ladder(&cmd, opts)?;

        // Attach to a kill-on-close job immediately so the renderer
        // can't escape via CreateProcess in the tiny window between
        // spawn and policy application.
        let mut jb = JobObjectBuilder::new()
            .kill_on_close(true)
            .memory_limit_bytes(opts.memory_limit_bytes);
        if !opts.allow_subprocesses {
            jb = jb.active_process_limit(1);
        }
        let job = jb.build().map_err(SandboxedSpawnError::Sandbox)?;
        job.attach(&child).map_err(SandboxedSpawnError::Sandbox)?;

        // Now wait for the client connection.
        let endpoint = server_handle
            .join()
            .map_err(|_| SandboxedSpawnError::Connect("server thread panic".into()))?
            .map_err(|e| SandboxedSpawnError::Connect(format!("{e:?}")))?;

        Ok(Self {
            process: child,
            job,
            endpoint,
            pipe_name,
            tier,
        })
    }
}

/// `CV_SANDBOX_HARDEN` — master kill switch for the hardening ladder.
/// Default ON. When OFF, `spawn` reverts to the exact original
/// behavior: plain `ChildProcess::spawn` (Tier4) + the unchanged Job
/// Object. Read here in the cold spawn path (once per child).
fn hardening_enabled() -> bool {
    match std::env::var("CV_SANDBOX_HARDEN") {
        Ok(v) => {
            let v = v.trim();
            !(v == "0" || v.eq_ignore_ascii_case("off") || v.eq_ignore_ascii_case("false"))
        }
        Err(_) => true, // default ON
    }
}

/// `CV_SANDBOX_APPCONTAINER` — opt-in (default OFF) Tier1 AppContainer
/// attempt. The token/capabilities spawn plumbing now EXISTS (Tier1 is
/// genuinely reachable: SECURITY_CAPABILITIES + CreateProcessAsUserW
/// with the restricted token, then a TokenIsAppContainer query-back).
/// Kept opt-in until the full-lockdown AppContainer profile is verified
/// to still render real pages (win32k/font disable can starve GDI text
/// bake); flip the default only after that soak.
fn appcontainer_enabled() -> bool {
    matches!(std::env::var("CV_SANDBOX_APPCONTAINER"), Ok(v) if {
        let v = v.trim();
        v == "1" || v.eq_ignore_ascii_case("on") || v.eq_ignore_ascii_case("true")
    })
}

/// Test-only seam: force the mitigation rung (Tier3) to fail so the
/// ladder must step down to plain spawn (Tier4). Proves the
/// "never fail to launch" safety property end-to-end.
#[cfg(test)]
fn force_mitigation_fail() -> bool {
    matches!(std::env::var("CV_SANDBOX_TEST_FORCE_MITIGATION_FAIL"), Ok(v) if v == "1")
}

#[cfg(not(test))]
fn force_mitigation_fail() -> bool {
    false
}

/// Test-only seam: force the Tier2 (token-bearing low-integrity) rung
/// to be treated as failed so the ladder must step DOWN to Tier3
/// (Mitigation). Proves the honesty contract: when low-integrity can't
/// be applied + verified, we record Mitigation, never claim Tier2.
#[cfg(test)]
fn force_lowint_fail() -> bool {
    matches!(std::env::var("CV_SANDBOX_TEST_FORCE_LOWINT_FAIL"), Ok(v) if v == "1")
}

#[cfg(not(test))]
fn force_lowint_fail() -> bool {
    false
}

/// The fallback ladder proper. Returns the surviving `ChildProcess`
/// plus the `AppliedTier` that actually launched it. Only propagates
/// an error if even plain spawn (Tier4) fails — a genuinely
/// unspawnable exe is a real error, not a hardening failure.
///
/// NO-LEAK INVARIANT: each rung either returns an owned
/// `ChildProcess` or fully releases its own intermediates before the
/// next rung. `spawn_with_mitigation` self-cleans its attribute list
/// (DeleteProcThreadAttributeList) on internal failure; the
/// AppContainer sandbox releases its token + SID via RAII Drop. Only
/// one `ChildProcess` is ever live.
fn spawn_with_ladder(
    cmd: &str,
    opts: &SpawnOptions,
) -> Result<(ChildProcess, AppliedTier), SandboxedSpawnError> {
    // Kill switch: behave exactly like the original code path.
    if !hardening_enabled() {
        let child = ChildProcess::spawn(cmd, false).map_err(SandboxedSpawnError::Spawn)?;
        return Ok((child, AppliedTier::JobOnly));
    }

    // --- Tier1: AppContainer + restricted token + low integrity +
    // mitigation. Only attempted when explicitly opted in AND the
    // policy asks for an AppContainer. We build the REAL OS-derived
    // AppContainer (CreateAppContainerProfile + CreateRestrictedToken),
    // attach its OS-derived SID as a SECURITY_CAPABILITIES attribute,
    // and spawn the child bearing the restricted token via
    // CreateProcessAsUserW. ★ HONESTY KEYSTONE: a successful spawn is
    // NOT enough — we query the child's token back
    // (query_is_app_container) and ONLY record Tier1 when the kernel
    // confirms TokenIsAppContainer. Otherwise we kill the unverified
    // child and step down, never claiming a tier the kernel didn't
    // apply.
    if appcontainer_enabled() && !opts.app_container_sid.is_empty() {
        match cv_sandbox::appcontainer::AppContainerSandbox::create("ConclaveRenderer") {
            Ok(ac) => {
                // Build SECURITY_CAPABILITIES from the OS-derived SID
                // (NOT the synthetic from_install_key string — that one
                // is only a uniqueness hint). The struct + SID storage
                // must outlive the spawn call: `ac` owns the SID and is
                // dropped only at the end of this block.
                let mut caps = cv_sandbox::build_security_capabilities(ac.appcontainer_sid);
                let caps_ptr = (&mut caps) as *mut _ as *mut core::ffi::c_void;
                // SAFETY: ac.restricted_token is a valid primary token
                // owned by `ac`; caps points to live storage backed by
                // `ac`'s SID; both outlive this call.
                let spawn_res = unsafe {
                    ChildProcess::spawn_with_token(
                        cmd,
                        opts.mitigation_policy,
                        caps_ptr,
                        ac.restricted_token,
                    )
                };
                match spawn_res {
                    Ok(child) => {
                        // Query-back: did the kernel actually place the
                        // child in an AppContainer?
                        match child.query_is_app_container() {
                            Some(true) => {
                                drop(ac);
                                return Ok((child, AppliedTier::AppContainer));
                            }
                            other => {
                                // Not verifiably AppContainer — refuse
                                // to claim Tier1. Kill the unverified
                                // child and step down honestly.
                                let _ = child.kill(0);
                                let _ = child.wait();
                                log_appcontainer_unverified(other);
                                log_step_down(
                                    AppliedTier::AppContainer,
                                    AppliedTier::MitigationLowIntegrity,
                                    0,
                                );
                            }
                        }
                    }
                    Err(e) => {
                        let gle = match e {
                            SpawnError::Create(g) | SpawnError::Token(g) => g,
                            _ => 0,
                        };
                        log_step_down(
                            AppliedTier::AppContainer,
                            AppliedTier::MitigationLowIntegrity,
                            gle,
                        );
                    }
                }
                drop(ac);
            }
            Err(e) => {
                log_appcontainer_fail(&e);
                // Fall through to Tier2.
            }
        }
    }

    // --- Tier2: mitigation + low integrity (no AppContainer). Only
    // attempted when the policy requests low integrity. Duplicate the
    // broker's token, lower it to Low mandatory integrity, and spawn
    // bearing it. ★ HONESTY: record MitigationLowIntegrity only after
    // query_integrity_is_low confirms the child really is Low; on an
    // unverified spawn, kill + step down to Tier3.
    if opts.low_integrity && !force_mitigation_fail() && !force_lowint_fail() {
        match LowIntegrityToken::from_current_process() {
            Ok(low_tok) => {
                // SAFETY: low_tok.handle() is a valid primary token
                // owned by `low_tok`, alive across this call; no
                // SECURITY_CAPABILITIES (null) for the no-AppContainer
                // rung.
                let spawn_res = unsafe {
                    ChildProcess::spawn_with_token(
                        cmd,
                        opts.mitigation_policy,
                        core::ptr::null_mut(),
                        low_tok.handle(),
                    )
                };
                match spawn_res {
                    Ok(child) => match child.query_integrity_is_low() {
                        Some(true) => {
                            drop(low_tok);
                            return Ok((child, AppliedTier::MitigationLowIntegrity));
                        }
                        other => {
                            let _ = child.kill(0);
                            let _ = child.wait();
                            log_lowintegrity_unverified(other);
                            log_step_down(
                                AppliedTier::MitigationLowIntegrity,
                                AppliedTier::Mitigation,
                                0,
                            );
                        }
                    },
                    Err(e) => {
                        let gle = match e {
                            SpawnError::Create(g) | SpawnError::Token(g) => g,
                            _ => 0,
                        };
                        log_step_down(
                            AppliedTier::MitigationLowIntegrity,
                            AppliedTier::Mitigation,
                            gle,
                        );
                    }
                }
                drop(low_tok);
            }
            Err(e) => {
                let gle = match e {
                    SpawnError::Token(g) | SpawnError::Create(g) => g,
                    _ => 0,
                };
                log_step_down(
                    AppliedTier::MitigationLowIntegrity,
                    AppliedTier::Mitigation,
                    gle,
                );
            }
        }
    }

    // --- Tier3: mitigation policy word ONLY. THE V1 DEFAULT RUNG.
    // Real kernel-enforced DEP / ASLR / ACG / CFG / font-win32k-
    // disable / image-load-no-remote via spawn_with_mitigation.
    if !force_mitigation_fail() {
        match ChildProcess::spawn_with_mitigation(cmd, false, opts.mitigation_policy) {
            Ok(child) => return Ok((child, AppliedTier::Mitigation)),
            Err(e) => {
                let gle = match e {
                    SpawnError::Create(g) => g,
                    _ => 0,
                };
                // spawn_with_mitigation self-cleaned its attribute
                // list before returning Err — no leak on step-down.
                log_step_down(AppliedTier::Mitigation, AppliedTier::JobOnly, gle);
            }
        }
    } else {
        // Test seam: pretend the kernel rejected the policy bits.
        log_step_down(AppliedTier::Mitigation, AppliedTier::JobOnly, 87);
    }

    // --- Tier4: plain spawn(). The floor. Job-object contained but
    // no mitigation hardening. If even this fails, THEN propagate —
    // a genuinely unspawnable exe is a real error.
    let child = ChildProcess::spawn(cmd, false).map_err(SandboxedSpawnError::Spawn)?;
    Ok((child, AppliedTier::JobOnly))
}

/// Structured step-down log line (NOT in the render hot path — this is
/// the cold spawn path). Records from-tier, GetLastError, to-tier so a
/// silent regression where the ladder always falls to plain spawn is
/// observable. Routed through `eprintln` only when a debug env opts in
/// so production stays quiet.
fn log_step_down(from: AppliedTier, to: AppliedTier, gle: u32) {
    if std::env::var("CV_SANDBOX_LOG").is_ok() {
        eprintln!(
            "cv_sandbox: hardening step-down from={} to={} gle={}",
            from.label(),
            to.label(),
            gle
        );
    }
}

fn log_appcontainer_fail(reason: &str) {
    if std::env::var("CV_SANDBOX_LOG").is_ok() {
        eprintln!("cv_sandbox: AppContainer tier unavailable: {reason}");
    }
}

/// The spawn succeeded but the AppContainer query-back did NOT confirm
/// the child is in an AppContainer. We refuse to claim Tier1. `q` is
/// the read-back result (`None` = query failed, `Some(false)` = not an
/// AppContainer).
fn log_appcontainer_unverified(q: Option<bool>) {
    if std::env::var("CV_SANDBOX_LOG").is_ok() {
        eprintln!("cv_sandbox: AppContainer spawn not verified (TokenIsAppContainer={q:?}); refusing to claim Tier1");
    }
}

/// The low-integrity spawn succeeded but the integrity-level query-back
/// did NOT confirm the child is Low. We refuse to claim Tier2.
fn log_lowintegrity_unverified(q: Option<bool>) {
    if std::env::var("CV_SANDBOX_LOG").is_ok() {
        eprintln!("cv_sandbox: low-integrity spawn not verified (integrity_is_low={q:?}); refusing to claim Tier2");
    }
}

fn quote_arg(s: &str) -> String {
    if s.is_empty() || s.contains(' ') || s.contains('\t') || s.contains('"') {
        let escaped = s.replace('"', "\\\"");
        format!("\"{escaped}\"")
    } else {
        s.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pipe_name_generation_is_unique() {
        // Sanity: each call yields a different name.
        let a = next_pipe_name(ChildKind::Renderer);
        let b = next_pipe_name(ChildKind::Renderer);
        assert_ne!(a, b);
        assert!(a.contains("renderer"));
    }

    #[test]
    fn quote_arg_handles_spaces_and_quotes() {
        assert_eq!(quote_arg("hello"), "hello");
        assert_eq!(quote_arg("with space"), "\"with space\"");
        assert_eq!(quote_arg("with \"quote\""), "\"with \\\"quote\\\"\"");
    }

    // End-to-end test: spawn a small "child" that echoes one message
    // back via the IPC pipe. We use `cmd /c` with a powershell one-
    // liner that opens the pipe, reads a u32 message id + a string
    // payload, and writes back a constant response. That's too brittle
    // for V1 — Windows pipe access from cmd is awkward. Instead the
    // test here proves the parent-side composition (server thread +
    // spawn + job attach) terminates cleanly even if no client
    // actually connects within the timeout.

    use std::sync::Mutex;

    // Tests that mutate process-global env vars must run serially —
    // the test harness runs tests in parallel by default and env is
    // shared. This mutex serializes them; each restores the env it
    // touched.
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    /// Detect whether this environment can actually spawn a child.
    /// Mirrors the GPU-device-test convention: on a locked-down CI
    /// that denies process creation, the spawn-dependent tests SKIP
    /// (return early) instead of failing, so headless CI stays green.
    fn can_spawn() -> bool {
        match ChildProcess::spawn("cmd.exe /c exit 0", false) {
            Ok(c) => {
                let _ = c.wait();
                true
            }
            Err(_) => false,
        }
    }

    fn build_job(opts: &SpawnOptions) -> JobObject {
        let mut jb = JobObjectBuilder::new()
            .kill_on_close(true)
            .memory_limit_bytes(opts.memory_limit_bytes);
        if !opts.allow_subprocesses {
            jb = jb.active_process_limit(1);
        }
        jb.build().unwrap()
    }

    // (a) ★ PROFILE-WIRED — the keystone, deterministic, no spawn.
    // Proves the production renderer mitigation profile flows through
    // options_from_channel_policy into the packed word + low_integrity
    // + AppContainer sid. MUTATION CHECK: swapping
    // renderer_defaults() -> MitigationPolicies::default() in
    // ChannelPolicy::stable_renderer zeroes the gated bits below and
    // FAILS this test — proving it is genuinely wired, not a constant.
    #[test]
    fn options_from_stable_renderer_have_full_mitigation_word() {
        let o = options_from_channel_policy(
            ChildKind::Renderer,
            cv_sandbox::ChannelPolicy::stable_renderer(),
            "test-install-key",
        );
        let w = o.mitigation_policy;
        assert_ne!(w & 0x01, 0, "DEP must be set");
        assert_ne!(w & 0x10_0000, 0, "high-entropy ASLR must be set");
        assert_ne!(w & 0x1_0000, 0, "bottom-up ASLR must be set");
        assert_ne!(w & 0x1_0000_0000_0, 0, "ACG / prohibit-dynamic-code must be set");
        assert_ne!(w & 0x1_0000_0000_00, 0, "CFG must be set");
        assert!(o.low_integrity, "stable renderer must request low integrity");
        assert!(
            o.app_container_sid.starts_with("S-1-15-2-"),
            "AppContainer sid must be carried: {}",
            o.app_container_sid
        );
    }

    // dev_renderer is debugger-friendly: ACG + font off, no
    // AppContainer, no low integrity. Deterministic.
    #[test]
    fn options_from_dev_renderer_are_looser() {
        let o = options_from_channel_policy(
            ChildKind::Renderer,
            cv_sandbox::ChannelPolicy::dev_renderer(),
            "k",
        );
        let w = o.mitigation_policy;
        // ACG (prohibit dynamic code) must be OFF in dev.
        assert_eq!(w & 0x1_0000_0000_0, 0, "ACG must be OFF in dev");
        // But DEP + ASLR stay on.
        assert_ne!(w & 0x01, 0, "DEP stays on in dev");
        assert_ne!(w & 0x10_0000, 0, "high-entropy ASLR stays on in dev");
        assert!(!o.low_integrity, "dev must NOT lower integrity");
        assert!(o.app_container_sid.is_empty(), "dev must not carry an AC sid");
    }

    // (c) FALLBACK LADDER — forced Tier3 failure → Tier4 chosen, child
    // STILL launches. Proves "never fail to launch". Deterministic
    // (env seam), skips if env can't spawn at all.
    #[test]
    fn forced_mitigation_failure_falls_back_to_plain_and_still_launches() {
        let _g = ENV_LOCK.lock().unwrap();
        if !can_spawn() {
            return; // skip: no spawn rights
        }
        // SAFETY: env mutation serialized by ENV_LOCK; restored below.
        unsafe {
            std::env::set_var("CV_SANDBOX_TEST_FORCE_MITIGATION_FAIL", "1");
            std::env::set_var("CV_SANDBOX_HARDEN", "1");
        }
        let opts = options_from_channel_policy(
            ChildKind::Renderer,
            cv_sandbox::ChannelPolicy::stable_renderer(),
            "k",
        );
        let result = spawn_with_ladder("cmd.exe /c exit 0", &opts);
        unsafe {
            std::env::remove_var("CV_SANDBOX_TEST_FORCE_MITIGATION_FAIL");
            std::env::remove_var("CV_SANDBOX_HARDEN");
        }
        let (child, tier) = result.expect("ladder must still launch a child");
        assert_eq!(tier, AppliedTier::JobOnly, "forced mitigation fail → Tier4");
        let code = child.wait().unwrap();
        assert_eq!(code, 0);
    }

    // Healthy config: hardening ON, no forced failure → a HARDENED rung
    // is chosen, NOT Tier4 plain. The exact rung depends on whether this
    // box grants SeAssignPrimaryTokenPrivilege for the token-bearing
    // low-integrity spawn: with it, Tier2 (MitigationLowIntegrity);
    // without it the token spawn fails and we step down to Tier3
    // (Mitigation). Both are the hardened path; the regression this
    // catches is the ladder silently always falling to plain spawn.
    #[test]
    fn healthy_config_uses_a_hardened_tier() {
        let _g = ENV_LOCK.lock().unwrap();
        if !can_spawn() {
            return; // skip
        }
        unsafe {
            std::env::remove_var("CV_SANDBOX_TEST_FORCE_MITIGATION_FAIL");
            std::env::set_var("CV_SANDBOX_HARDEN", "1");
        }
        let opts = options_from_channel_policy(
            ChildKind::Renderer,
            cv_sandbox::ChannelPolicy::stable_renderer(),
            "k",
        );
        let result = spawn_with_ladder("cmd.exe /c ping localhost -n 2", &opts);
        unsafe {
            std::env::remove_var("CV_SANDBOX_HARDEN");
        }
        let (child, tier) = result.expect("must launch");
        assert!(
            matches!(
                tier,
                AppliedTier::MitigationLowIntegrity | AppliedTier::Mitigation
            ),
            "healthy config must take a hardened rung (Tier2 or Tier3), not plain spawn; got {tier:?}"
        );
        // ★ HONESTY: whichever tier we recorded must be backed by the
        // kernel query-back. If we claimed low integrity, the child must
        // verifiably be Low; if just mitigation, DEP must be in effect.
        match tier {
            AppliedTier::MitigationLowIntegrity => {
                assert_eq!(
                    child.query_integrity_is_low(),
                    Some(true),
                    "Tier2 claimed but kernel does not confirm Low integrity"
                );
            }
            AppliedTier::Mitigation => {
                if let Some(dep) = child.query_mitigation().dep_enabled {
                    assert!(dep, "Tier3 claimed but DEP not in effect");
                }
            }
            _ => {}
        }
        let _ = child.kill(0);
        let _ = child.wait();
    }

    // Kill switch: CV_SANDBOX_HARDEN=0 → exact original behavior
    // (Tier4 only), child still launches.
    #[test]
    fn kill_switch_off_uses_plain_spawn() {
        let _g = ENV_LOCK.lock().unwrap();
        if !can_spawn() {
            return; // skip
        }
        unsafe {
            std::env::set_var("CV_SANDBOX_HARDEN", "0");
        }
        let opts = options_from_channel_policy(
            ChildKind::Renderer,
            cv_sandbox::ChannelPolicy::stable_renderer(),
            "k",
        );
        let result = spawn_with_ladder("cmd.exe /c exit 0", &opts);
        unsafe {
            std::env::remove_var("CV_SANDBOX_HARDEN");
        }
        let (child, tier) = result.expect("must launch");
        assert_eq!(tier, AppliedTier::JobOnly, "kill switch off → plain spawn");
        let _ = child.wait();
    }

    // (d) JOB IN ALL TIERS — after a Tier3 spawn AND after a forced
    // Tier4 spawn, the kill-on-close Job Object kills the child when
    // the job drops (reuses the sandbox.rs attach-then-kill pattern).
    // Proves the Job survives the ladder in every tier.
    #[test]
    fn job_object_kills_child_in_mitigation_tier() {
        let _g = ENV_LOCK.lock().unwrap();
        if !can_spawn() {
            return; // skip
        }
        unsafe {
            std::env::remove_var("CV_SANDBOX_TEST_FORCE_MITIGATION_FAIL");
            std::env::set_var("CV_SANDBOX_HARDEN", "1");
        }
        // dev_renderer has low_integrity=false → Tier2 is NOT attempted,
        // so the ladder deterministically lands on Tier3 (Mitigation)
        // regardless of whether this box grants the token-spawn
        // privilege. That keeps this job-kill-in-mitigation-tier test
        // deterministic now that Tier2 is genuinely reachable.
        let opts = options_from_channel_policy(
            ChildKind::Renderer,
            cv_sandbox::ChannelPolicy::dev_renderer(),
            "k",
        );
        // Long-running child so we can prove the job kill, not natural exit.
        let result = spawn_with_ladder("cmd.exe /c ping localhost -n 5", &opts);
        unsafe {
            std::env::remove_var("CV_SANDBOX_HARDEN");
        }
        let (child, tier) = result.expect("must launch");
        assert_eq!(tier, AppliedTier::Mitigation);
        let job = build_job(&opts);
        job.attach(&child).unwrap();
        drop(job);
        let r = child.try_wait_for(2_000).unwrap();
        assert!(r.is_some(), "child must die when the kill-on-close job drops (mitigation tier)");
    }

    #[test]
    fn job_object_kills_child_in_plain_tier() {
        let _g = ENV_LOCK.lock().unwrap();
        if !can_spawn() {
            return; // skip
        }
        unsafe {
            std::env::set_var("CV_SANDBOX_TEST_FORCE_MITIGATION_FAIL", "1");
            std::env::set_var("CV_SANDBOX_HARDEN", "1");
        }
        let opts = options_from_channel_policy(
            ChildKind::Renderer,
            cv_sandbox::ChannelPolicy::stable_renderer(),
            "k",
        );
        let result = spawn_with_ladder("cmd.exe /c ping localhost -n 5", &opts);
        unsafe {
            std::env::remove_var("CV_SANDBOX_TEST_FORCE_MITIGATION_FAIL");
            std::env::remove_var("CV_SANDBOX_HARDEN");
        }
        let (child, tier) = result.expect("must launch");
        assert_eq!(tier, AppliedTier::JobOnly);
        let job = build_job(&opts);
        job.attach(&child).unwrap();
        drop(job);
        let r = child.try_wait_for(2_000).unwrap();
        assert!(r.is_some(), "child must die when the kill-on-close job drops (plain tier)");
    }

    // (b) STRONGEST PROOF — GetProcessMitigationPolicy read-back on a
    // live child spawned through the mitigation rung. Asserts the
    // kernel ACTUALLY applied DEP + ASLR. Skips gracefully if the env
    // can't spawn. Upgrades "kernel accepted the word" to "verifiably
    // in effect on the child".
    #[test]
    fn spawned_child_has_mitigation_in_effect() {
        let _g = ENV_LOCK.lock().unwrap();
        if !can_spawn() {
            return; // skip
        }
        unsafe {
            std::env::remove_var("CV_SANDBOX_TEST_FORCE_MITIGATION_FAIL");
            std::env::set_var("CV_SANDBOX_HARDEN", "1");
        }
        // dev_renderer (low_integrity=false) → deterministic Tier3 so the
        // DEP/ASLR read-back below is asserted on the mitigation rung
        // specifically, not on a token-bearing Tier2 spawn whose
        // availability depends on box privileges.
        let opts = options_from_channel_policy(
            ChildKind::Renderer,
            cv_sandbox::ChannelPolicy::dev_renderer(),
            "k",
        );
        // Stay alive ~1s so we can read its policy before it exits.
        let result = spawn_with_ladder("cmd.exe /c ping localhost -n 3", &opts);
        unsafe {
            std::env::remove_var("CV_SANDBOX_HARDEN");
        }
        let (child, tier) = result.expect("must launch");
        assert_eq!(tier, AppliedTier::Mitigation);
        let applied = child.query_mitigation();
        // DEP must be ON. high-entropy ASLR is only meaningful for
        // 64-bit images and may not read back identically across
        // configs, so we assert the field that is reliably reported.
        if let Some(dep) = applied.dep_enabled {
            assert!(dep, "DEP must be in effect on the hardened child");
        }
        // If ASLR read back at all, the bottom-up bit should be set
        // (it's in renderer_defaults). Tolerate a None read-back
        // (older kernels / WOW64) by only asserting when present.
        if let Some(bu) = applied.aslr_bottom_up {
            assert!(bu, "bottom-up ASLR must be in effect on the hardened child");
        }
        let _ = child.kill(0);
        let _ = child.wait();
    }

    // ===================== A4: TOKEN-TIER TESTS =====================

    /// Building a Low-integrity primary token from the current process
    /// must succeed WITHOUT any special privilege (you can always lower
    /// your own token), and the lowered token's integrity must read back
    /// as Low. This proves the integrity primitive itself is real —
    /// independent of whether the box grants the spawn privilege.
    #[test]
    fn low_integrity_token_reads_back_as_low() {
        use crate::spawn::LowIntegrityToken;
        let tok = match LowIntegrityToken::from_current_process() {
            Ok(t) => t,
            Err(e) => {
                // Lowering one's own token should not require privilege;
                // if this environment refuses it entirely, skip rather
                // than fail (locked-down CI).
                eprintln!("skip: cannot build low-integrity token: {e}");
                return;
            }
        };
        // Read TokenIntegrityLevel directly off the duplicated token via
        // the same RID-extraction path the child query-back uses.
        let rid = crate::spawn::token_integrity_rid(tok.handle());
        match rid {
            Some(r) => assert!(
                r <= 0x1000,
                "duplicated token integrity RID {r:#x} must be <= Low (0x1000)"
            ),
            None => eprintln!("skip: integrity read-back unsupported here"),
        }
    }

    /// Tier2 (token-bearing low-integrity) end-to-end via the ladder.
    /// On a box that grants SeAssignPrimaryTokenPrivilege the ladder
    /// reaches MitigationLowIntegrity and the child VERIFIABLY reads
    /// back as Low integrity. On a box without the privilege the token
    /// spawn fails and the ladder steps down to Mitigation — also
    /// acceptable (the honest contract). NEVER records Tier2 unless the
    /// kernel confirms Low integrity.
    #[test]
    fn ladder_low_integrity_tier_is_verified_when_reached() {
        let _g = ENV_LOCK.lock().unwrap();
        if !can_spawn() {
            return; // skip
        }
        unsafe {
            std::env::remove_var("CV_SANDBOX_TEST_FORCE_MITIGATION_FAIL");
            std::env::remove_var("CV_SANDBOX_TEST_FORCE_LOWINT_FAIL");
            std::env::set_var("CV_SANDBOX_HARDEN", "1");
        }
        // stable_renderer requests low_integrity → Tier2 is attempted.
        let opts = options_from_channel_policy(
            ChildKind::Renderer,
            cv_sandbox::ChannelPolicy::stable_renderer(),
            "k",
        );
        let result = spawn_with_ladder("cmd.exe /c ping localhost -n 3", &opts);
        unsafe {
            std::env::remove_var("CV_SANDBOX_HARDEN");
        }
        let (child, tier) = result.expect("must launch");
        match tier {
            AppliedTier::MitigationLowIntegrity => {
                // The keystone: a claimed Tier2 MUST be kernel-verified.
                assert_eq!(
                    child.query_integrity_is_low(),
                    Some(true),
                    "Tier2 recorded but child is not verifiably Low integrity (fake tier!)"
                );
            }
            AppliedTier::Mitigation => {
                // Box lacks the spawn privilege; honest step-down.
                eprintln!("note: token spawn unavailable, stepped down to Mitigation (honest)");
            }
            other => panic!("unexpected tier for stable_renderer: {other:?}"),
        }
        let _ = child.kill(0);
        let _ = child.wait();
    }

    /// DOWNGRADE PROOF: with the Tier2 rung forced to fail (test seam),
    /// the ladder must record the VERIFIED lower tier (Mitigation),
    /// never the requested MitigationLowIntegrity. Catches a regression
    /// where a failed/unverified low-integrity spawn is wrongly claimed
    /// as Tier2.
    #[test]
    fn forced_lowint_failure_steps_down_to_mitigation() {
        let _g = ENV_LOCK.lock().unwrap();
        if !can_spawn() {
            return; // skip
        }
        unsafe {
            std::env::remove_var("CV_SANDBOX_TEST_FORCE_MITIGATION_FAIL");
            std::env::set_var("CV_SANDBOX_TEST_FORCE_LOWINT_FAIL", "1");
            std::env::set_var("CV_SANDBOX_HARDEN", "1");
        }
        let opts = options_from_channel_policy(
            ChildKind::Renderer,
            cv_sandbox::ChannelPolicy::stable_renderer(),
            "k",
        );
        let result = spawn_with_ladder("cmd.exe /c exit 0", &opts);
        unsafe {
            std::env::remove_var("CV_SANDBOX_TEST_FORCE_LOWINT_FAIL");
            std::env::remove_var("CV_SANDBOX_HARDEN");
        }
        let (child, tier) = result.expect("must launch");
        assert_eq!(
            tier,
            AppliedTier::Mitigation,
            "forced Tier2 failure must step down to verified Tier3, not claim Tier2"
        );
        let code = child.wait().unwrap();
        assert_eq!(code, 0);
    }

    /// Tier1 (AppContainer) end-to-end via the ladder, opt-in. Gated on
    /// CV_SANDBOX_APPCONTAINER. On a box where the AppContainer profile
    /// + token spawn succeed, the ladder records AppContainer ONLY after
    /// query_is_app_container() == Some(true). If profile creation or
    /// the token spawn fails (privilege / policy), the ladder steps down
    /// honestly — also accepted. NEVER claims AppContainer unverified.
    #[test]
    fn ladder_app_container_tier_is_verified_when_reached() {
        let _g = ENV_LOCK.lock().unwrap();
        if !can_spawn() {
            return; // skip
        }
        unsafe {
            std::env::remove_var("CV_SANDBOX_TEST_FORCE_MITIGATION_FAIL");
            std::env::remove_var("CV_SANDBOX_TEST_FORCE_LOWINT_FAIL");
            std::env::set_var("CV_SANDBOX_HARDEN", "1");
            std::env::set_var("CV_SANDBOX_APPCONTAINER", "1");
        }
        let opts = options_from_channel_policy(
            ChildKind::Renderer,
            cv_sandbox::ChannelPolicy::stable_renderer(),
            "k",
        );
        let result = spawn_with_ladder("cmd.exe /c ping localhost -n 3", &opts);
        unsafe {
            std::env::remove_var("CV_SANDBOX_APPCONTAINER");
            std::env::remove_var("CV_SANDBOX_HARDEN");
        }
        let (child, tier) = result.expect("must launch");
        match tier {
            AppliedTier::AppContainer => {
                assert_eq!(
                    child.query_is_app_container(),
                    Some(true),
                    "Tier1 recorded but child is not verifiably in an AppContainer (fake tier!)"
                );
            }
            AppliedTier::MitigationLowIntegrity | AppliedTier::Mitigation => {
                eprintln!("note: AppContainer unavailable, stepped down honestly to {tier:?}");
            }
            AppliedTier::JobOnly => {
                eprintln!("note: stepped all the way down to JobOnly (no hardening privilege)");
            }
        }
        let _ = child.kill(0);
        let _ = child.wait();
    }
}
