//! Device output driver — bridges an [`crate::graph::AudioGraph`] to the
//! system audio endpoint via [`crate::wasapi`].
//!
//! Chrome runs the audio graph on a dedicated real-time audio thread that
//! the OS calls back (or that polls the device) to keep the endpoint buffer
//! fed in 128-frame render quanta. This module is that driver: it opens the
//! default WASAPI render endpoint, then on each `pump()` renders as many
//! quanta as the endpoint has free space for and writes them through
//! `IAudioRenderClient` (the real device push — `GetBuffer`/`ReleaseBuffer`).
//!
//! The driver is gated by the host behind `CV_WEBAUDIO` (default OFF) because
//! opening the system endpoint can fail or regress on some machines; the
//! graph + decode + render-buffer path is always real and used by
//! `OfflineAudioContext` and the unit tests regardless.
//!
//! Reference: WASAPI rendering loop (Microsoft "Rendering a Stream"):
//!   GetBufferSize → loop { GetCurrentPadding; avail = bufSize - padding;
//!   GetBuffer(avail); fill; ReleaseBuffer(avail); sleep ~half buffer }.

use crate::graph::AudioGraph;
use crate::wasapi::{WasapiDevice, WasapiError};

/// A live device-output driver. Holds the open endpoint; the host calls
/// [`AudioOutput::pump`] from its audio/render tick to keep audio flowing.
pub struct AudioOutput {
    device: WasapiDevice,
}

impl std::fmt::Debug for AudioOutput {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AudioOutput")
            .field("sample_rate", &self.device.mix_format.sample_rate)
            .field("channels", &self.device.mix_format.channels)
            .finish()
    }
}

impl AudioOutput {
    /// Open the default render endpoint. Errors propagate so the host can
    /// fall back to silent (offline-only) operation without crashing.
    pub fn open() -> Result<Self, WasapiError> {
        let device = WasapiDevice::open_default()?;
        Ok(Self { device })
    }

    /// The device mix sample rate (the graph should match this).
    pub fn sample_rate(&self) -> u32 {
        self.device.mix_format.sample_rate
    }

    /// The device channel count.
    pub fn channels(&self) -> u8 {
        self.device.mix_format.channels
    }

    /// Render from `graph` and push into the endpoint until the endpoint
    /// buffer is full. Returns the number of frames written this pump.
    /// Call this from the host's audio tick (e.g. every animation/render
    /// frame, or a dedicated thread).
    pub fn pump(&mut self, graph: &mut AudioGraph) -> u32 {
        let buffer_frames = self.device.buffer_size_frames();
        let padding = self.device.current_padding_frames();
        let mut free = buffer_frames.saturating_sub(padding);
        let mut written = 0u32;
        while free > 0 {
            // Render one quantum and write as much of it as fits.
            let q = graph.render_quantum();
            let n = self.device.write_frames(&q);
            if n == 0 {
                break;
            }
            written += n;
            free = free.saturating_sub(n);
        }
        written
    }
}

#[cfg(test)]
mod tests {
    // Device output is intentionally NOT unit-tested against real hardware
    // (no audible side effects in CI; opening the endpoint is environment-
    // dependent). The DSP that feeds it IS fully tested in `graph` and
    // `decode`. The contract here — "render quanta and push to the device" —
    // is exercised through `WasapiDevice::write_frames` (real GetBuffer/
    // ReleaseBuffer) when `CV_WEBAUDIO=1` on a machine with an audio device.
    //
    // We DO assert the bridge wiring compiles and that the type surface is
    // what the host expects.
    #[test]
    fn audio_output_type_surface_exists() {
        fn _assert_send<T: Send>() {}
        // WasapiDevice is Send (audio thread takes ownership); AudioOutput
        // wraps it, so it must remain movable to a render thread.
        _assert_send::<crate::wasapi::WasapiDevice>();
    }
}
