# Conclave

![Conclave](assets/conclave-256.png)

A from-scratch, Windows-native web browser. **Not** a fork of Chromium,
WebKit, Gecko, or Servo — and it pulls in **zero third-party crates**.

The application icon (`assets/conclave.ico`, multi-resolution 16–256px) is
embedded into the window at runtime and shown in the title bar, Alt-Tab, and
taskbar.

The HTML/CSS engine, JavaScript runtime (parser, bytecode VM, and optimizing
JIT), layout engine, networking stack, TLS 1.3 implementation, image decoders,
font shaping, GPU/CPU graphics rasterizer, and Chrome-extension-compatible
runtime are all written by hand in Rust (edition 2024) against the raw Win32 /
Direct3D 11 APIs.

## System requirements

- **Windows 10 or 11, x64.**
- **No redistributables required.** The binary is self-contained: it statically
  links the MSVC C runtime and only calls stock OS DLLs that ship with every
  Windows 10/11 install (`user32`, `gdi32`, `kernel32`, `d3d11`, `dxgi`,
  `dcomp`, `ole32`, `crypt32`, `bcrypt`, `imm32`, `winmm`). The GPU present
  path uses pre-compiled shaders embedded in the executable, so even
  `d3dcompiler_47.dll` is **not** needed. Copy `conclave.exe` to a clean
  machine and it runs.

## Running

The browser is a single executable, `conclave.exe`, that dispatches on a
`--type` flag. To open the interactive browser window:

```
conclave.exe --type window https://example.com
```

The release executable is a windowless GUI app (no console flashes behind the
window). Other useful, headless modes (handy for scripting/diagnostics):

| Command                                            | What it does                                              |
| -------------------------------------------------- | --------------------------------------------------------- |
| `conclave.exe --type window <url>`               | Open the full browser window (URL bar, back/fwd, tabs).   |
| `conclave.exe --type render <url>`               | Fetch + lay out a page and print a text dump (no window). |
| `conclave.exe --type screenshot <url> <out.png>` | Render a page to a PNG (no JS, no window).                |
| `conclave.exe --type run-js <file.js> [...]`     | Execute one or more JS files in the runtime (no window).  |
| `conclave.exe --type fetch <url>`                | Fetch a URL over HTTP(S) and print the response.          |

When launched from an existing console (e.g. `cmd`/PowerShell), the headless
modes inherit that console and write to stdout/stderr as usual.

## Building from source

```
cargo build --release -p cv_browser
```

Targets `x86_64-pc-windows-msvc` (pinned in `.cargo/config.toml`); requires the
MSVC toolchain. The shipping binary lands at
`target/x86_64-pc-windows-msvc/release/conclave.exe`. Run the test suite with:

```
cargo test --workspace
```

## Feature flags

Everything needed to load and render real pages is **on by default**. The
following `CV_*` environment variables gate optional/experimental paths and
diagnostics; set a variable to `1` (or `0`/`off` to force-disable a default-on
feature) before launching:

**Runtime / engine (default-on; flags below mostly toggle or disable):**

- `CV_OFFMAIN` — off-main-thread renderer (default on; set `0`/`off`/`false`
  to fall back to the legacy single-thread path).
- `CV_DOM` — arena-backed DOM.
- `CV_T2` — optimizing (T2) JIT for hot JS.
- `CV_BYTECODE` — bytecode VM execution path.

**Opt-in experimental (default-off):**

- `CV_GPU_PIPELINE` — GPU textured-quad render path (vs. the CPU rasterizer).
- `CV_OFFMAIN_COMPOSITOR` — off-main compositor thread.
- `CV_RETAINED_DL` — retained, node-keyed display list.
- `CV_DAMAGE_RASTER` — incremental damage-only re-raster (with
  `CV_DAMAGE_VERIFY`, `CV_DAMAGE_AREA_FRAC`, `CV_DAMAGE_CHUNK_FRAC`).
- `CV_PAINT_CACHE` / `CV_PAINT_CACHE_DISK` — persistent paint cache (with
  `CV_PAINT_CACHE_BUDGET_MB`, `CV_PAINT_CACHE_DISK_BUDGET_MB`).
- `CV_REAL_WORKERS` — real Web Workers.
- `CV_SERVICE_WORKERS` — service-worker support.
- `CV_IDB_PERSIST` / `CV_CACHES_PERSIST` — persist IndexedDB / Cache Storage.
- `CV_GC` — tracing GC for JS reference cycles.

**Diagnostics (default-off; logging/verification):**

- `CV_LOGFILE` — redirect diagnostic output to a file.
- `CV_DIAG`, `CV_SCHEDLOG`, `CV_RENDERLOG`, `CV_FETCHLOG`, `CV_SCRIPTLOG`,
  `CV_POLLLOG`, `CV_PAINTTIME` — subsystem trace logs.
- `CV_LAYOUT_VERIFY`, `CV_RENDER_VERIFY`, `CV_DAMAGE_ORACLE`,
  `CV_RETAINED_ORACLE` — internal correctness oracles.
- `CV_T2_STATS`, `CV_LAYOUT_STATS`, `CV_BLOOM_STATS` — counters/stats.
- `CV_JS_TIME_BUDGET_MS` — per-task JS wall-clock watchdog (default 8000ms).
