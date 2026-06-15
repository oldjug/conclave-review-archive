//! Web Audio API render graph — real DSP, processed in 128-sample
//! render quanta exactly as Chrome/Blink does.
//!
//! This is the engine behind `AudioContext`. It models the node graph
//! (sources → effects → destination) and pulls audio through it one
//! *render quantum* (128 frames) at a time, summing every input edge
//! into each node and applying that node's transfer function. The
//! output of the `Destination` node is the buffer that gets handed to
//! the system audio device (WASAPI, see [`crate::wasapi`]).
//!
//! Spec: <https://www.w3.org/TR/webaudio/>
//!   * §1.4 "The destination node" — single sink.
//!   * §2 render quantum is 128 sample-frames.
//!     (`AudioContext` "renders audio in blocks of 128 sample-frames".)
//!   * §1.7.x OscillatorNode periodic waveforms (sine/square/sawtooth/
//!     triangle), GainNode (output = input · gain), AudioBufferSourceNode
//!     (plays an `AudioBuffer`).
//!
//! Blink reference: `third_party/blink/renderer/modules/webaudio/`
//!   (`audio_node.cc`, `gain_node.cc`, `oscillator_node.cc`,
//!    `audio_destination_node.cc`, `offline_audio_context.cc`). Blink
//!   pulls from the destination; each `AudioHandler::Process` mixes its
//!   inputs then writes its outputs. We mirror that pull model.

use std::f32::consts::PI;

/// The Web Audio render quantum: audio is processed 128 frames at a
/// time. (Spec: "Rendering an audio graph … is done in blocks of
/// `RENDER_QUANTUM_FRAMES` (128) sample-frames".)
pub const RENDER_QUANTUM_FRAMES: usize = 128;

/// Periodic waveform types for an `OscillatorNode`.
/// Spec §1.7.10 `OscillatorType`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OscillatorType {
    Sine,
    Square,
    Sawtooth,
    Triangle,
}

impl OscillatorType {
    pub fn from_str(s: &str) -> Self {
        match s {
            "square" => Self::Square,
            "sawtooth" => Self::Sawtooth,
            "triangle" => Self::Triangle,
            _ => Self::Sine,
        }
    }
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Sine => "sine",
            Self::Square => "square",
            Self::Sawtooth => "sawtooth",
            Self::Triangle => "triangle",
        }
    }
}

/// Opaque handle to a node inside an [`AudioGraph`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeId(pub usize);

/// One decoded PCM buffer (the `AudioBuffer` interface). Channels are
/// stored de-interleaved (planar) — one `Vec<f32>` per channel — which
/// is exactly what `AudioBuffer.getChannelData(c)` returns.
#[derive(Debug, Clone, Default)]
pub struct AudioBuffer {
    pub sample_rate: u32,
    /// `channels[c][frame]`. `channels.len()` == numberOfChannels.
    pub channels: Vec<Vec<f32>>,
}

impl AudioBuffer {
    pub fn new(num_channels: usize, length: usize, sample_rate: u32) -> Self {
        Self {
            sample_rate,
            channels: vec![vec![0.0; length]; num_channels.max(1)],
        }
    }
    pub fn number_of_channels(&self) -> usize {
        self.channels.len()
    }
    pub fn length(&self) -> usize {
        self.channels.first().map_or(0, Vec::len)
    }
    pub fn duration(&self) -> f64 {
        if self.sample_rate == 0 {
            0.0
        } else {
            self.length() as f64 / f64::from(self.sample_rate)
        }
    }
}

/// The transfer function of a node. The graph processing pass dispatches
/// on this. Each variant owns just enough live state to be sample-accurate
/// across render quanta (e.g. an oscillator's running phase).
#[derive(Debug, Clone)]
enum NodeKind {
    /// The single sink. Its rendered output is the device output.
    Destination,
    /// `output = sum(inputs) * gain`. Spec §1.7.7.
    Gain { gain: f32 },
    /// A periodic waveform generator. `phase` is in cycles [0,1).
    /// Spec §1.7.10.
    Oscillator {
        osc_type: OscillatorType,
        frequency: f32,
        detune: f32,
        phase: f32,
        /// Sample frame at which `start()` takes effect (`when * sampleRate`).
        start_frame: u64,
        /// Sample frame at which `stop()` silences it (`u64::MAX` = never).
        stop_frame: u64,
        playing: bool,
    },
    /// Plays back an `AudioBuffer`. Spec §1.7.3 `AudioBufferSourceNode`.
    BufferSource {
        buffer: Option<AudioBuffer>,
        /// Read cursor, in source frames (fractional for playbackRate≠1).
        cursor: f64,
        playback_rate: f32,
        detune: f32,
        loop_: bool,
        start_frame: u64,
        stop_frame: u64,
        playing: bool,
    },
    /// `output = sum(inputs)` plus per-edge gain — a pure summing bus.
    /// Used for any node whose only job is to pass audio through (and for
    /// the analyser tap).
    PassThrough,
}

#[derive(Debug, Clone)]
struct Node {
    kind: NodeKind,
    /// Output channel count this node produces.
    channels: usize,
    /// Cached output of the last rendered quantum (planar). Other nodes
    /// read this when they pull. `out[c][frame]`.
    out: Vec<[f32; RENDER_QUANTUM_FRAMES]>,
}

impl Node {
    fn new(kind: NodeKind, channels: usize) -> Self {
        Self {
            kind,
            channels: channels.max(1),
            out: vec![[0.0; RENDER_QUANTUM_FRAMES]; channels.max(1)],
        }
    }
}

/// A directed edge `from.output -> to.input`. We model a simple single
/// input/output topology (sufficient for the core node set); fan-in is
/// supported by summing every edge that targets a node.
#[derive(Debug, Clone, Copy)]
struct Edge {
    from: NodeId,
    to: NodeId,
}

/// The audio render graph owned by one `AudioContext`.
#[derive(Debug)]
pub struct AudioGraph {
    sample_rate: u32,
    nodes: Vec<Node>,
    edges: Vec<Edge>,
    destination: NodeId,
    /// Total frames rendered so far (drives `currentTime` + node scheduling).
    frame_clock: u64,
}

impl AudioGraph {
    /// Create a graph with a single destination node.
    pub fn new(sample_rate: u32) -> Self {
        let sample_rate = if sample_rate == 0 { 48_000 } else { sample_rate };
        let dest = Node::new(NodeKind::Destination, 2);
        Self {
            sample_rate,
            nodes: vec![dest],
            edges: Vec::new(),
            destination: NodeId(0),
            frame_clock: 0,
        }
    }

    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    pub fn destination(&self) -> NodeId {
        self.destination
    }

    pub fn current_time(&self) -> f64 {
        self.frame_clock as f64 / f64::from(self.sample_rate)
    }

    fn add(&mut self, kind: NodeKind, channels: usize) -> NodeId {
        let id = NodeId(self.nodes.len());
        self.nodes.push(Node::new(kind, channels));
        id
    }

    pub fn create_oscillator(&mut self) -> NodeId {
        self.add(
            NodeKind::Oscillator {
                osc_type: OscillatorType::Sine,
                frequency: 440.0,
                detune: 0.0,
                phase: 0.0,
                start_frame: 0,
                stop_frame: u64::MAX,
                playing: false,
            },
            1,
        )
    }

    pub fn create_gain(&mut self) -> NodeId {
        self.add(NodeKind::Gain { gain: 1.0 }, 2)
    }

    pub fn create_buffer_source(&mut self) -> NodeId {
        self.add(
            NodeKind::BufferSource {
                buffer: None,
                cursor: 0.0,
                playback_rate: 1.0,
                detune: 0.0,
                loop_: false,
                start_frame: 0,
                stop_frame: u64::MAX,
                playing: false,
            },
            2,
        )
    }

    /// An analyser is, audio-path-wise, a pass-through that also exposes
    /// the most recent time-domain samples (see [`Self::analyser_data`]).
    pub fn create_analyser(&mut self) -> NodeId {
        self.add(NodeKind::PassThrough, 2)
    }

    /// Connect `from`'s output to `to`'s input. Returns false if either id
    /// is invalid.
    pub fn connect(&mut self, from: NodeId, to: NodeId) -> bool {
        if from.0 >= self.nodes.len() || to.0 >= self.nodes.len() {
            return false;
        }
        // Dedup identical edges (Web Audio collapses duplicate connects).
        if self.edges.iter().any(|e| e.from == from && e.to == to) {
            return true;
        }
        self.edges.push(Edge { from, to });
        true
    }

    /// Remove all outgoing edges from `from`.
    pub fn disconnect(&mut self, from: NodeId) {
        self.edges.retain(|e| e.from != from);
    }

    // ---- node parameter setters (called from the JS binding layer) ----

    pub fn set_gain(&mut self, id: NodeId, value: f32) {
        if let Some(Node {
            kind: NodeKind::Gain { gain },
            ..
        }) = self.nodes.get_mut(id.0)
        {
            *gain = value;
        }
    }

    pub fn set_osc_type(&mut self, id: NodeId, t: OscillatorType) {
        if let Some(Node {
            kind: NodeKind::Oscillator { osc_type, .. },
            ..
        }) = self.nodes.get_mut(id.0)
        {
            *osc_type = t;
        }
    }

    pub fn set_osc_frequency(&mut self, id: NodeId, hz: f32) {
        if let Some(Node {
            kind: NodeKind::Oscillator { frequency, .. },
            ..
        }) = self.nodes.get_mut(id.0)
        {
            *frequency = hz;
        }
    }

    pub fn set_osc_detune(&mut self, id: NodeId, cents: f32) {
        if let Some(Node {
            kind: NodeKind::Oscillator { detune, .. },
            ..
        }) = self.nodes.get_mut(id.0)
        {
            *detune = cents;
        }
    }

    pub fn set_buffer(&mut self, id: NodeId, buf: AudioBuffer) {
        if let Some(Node {
            kind: NodeKind::BufferSource { buffer, .. },
            channels,
            out,
            ..
        }) = self.nodes.get_mut(id.0)
        {
            *channels = buf.number_of_channels().max(1);
            *out = vec![[0.0; RENDER_QUANTUM_FRAMES]; *channels];
            *buffer = Some(buf);
        }
    }

    pub fn set_buffer_playback_rate(&mut self, id: NodeId, rate: f32) {
        if let Some(Node {
            kind: NodeKind::BufferSource { playback_rate, .. },
            ..
        }) = self.nodes.get_mut(id.0)
        {
            *playback_rate = rate;
        }
    }

    pub fn set_buffer_loop(&mut self, id: NodeId, on: bool) {
        if let Some(Node {
            kind: NodeKind::BufferSource { loop_, .. },
            ..
        }) = self.nodes.get_mut(id.0)
        {
            *loop_ = on;
        }
    }

    /// Schedule a source (oscillator or buffer source) to begin at `when`
    /// seconds (context time). `0.0` means "now".
    pub fn start(&mut self, id: NodeId, when: f64) {
        let frame = (when.max(0.0) * f64::from(self.sample_rate)).round() as u64;
        match self.nodes.get_mut(id.0).map(|n| &mut n.kind) {
            Some(NodeKind::Oscillator {
                start_frame,
                playing,
                ..
            }) => {
                *start_frame = frame;
                *playing = true;
            }
            Some(NodeKind::BufferSource {
                start_frame,
                playing,
                ..
            }) => {
                *start_frame = frame;
                *playing = true;
            }
            _ => {}
        }
    }

    /// Schedule a source to stop at `when` seconds (context time).
    pub fn stop(&mut self, id: NodeId, when: f64) {
        let frame = (when.max(0.0) * f64::from(self.sample_rate)).round() as u64;
        match self.nodes.get_mut(id.0).map(|n| &mut n.kind) {
            Some(NodeKind::Oscillator { stop_frame, .. })
            | Some(NodeKind::BufferSource { stop_frame, .. }) => {
                *stop_frame = frame;
            }
            _ => {}
        }
    }

    // ---- rendering ----

    /// Render exactly one render quantum (128 frames) of the destination,
    /// interleaved by channel. Returns `frames*channels` samples in
    /// `[L0,R0,L1,R1,…]` order (the device wants interleaved).
    pub fn render_quantum(&mut self) -> Vec<f32> {
        self.process_quantum();
        let dest = &self.nodes[self.destination.0];
        let ch = dest.channels;
        let mut out = vec![0.0f32; RENDER_QUANTUM_FRAMES * ch];
        for c in 0..ch {
            for f in 0..RENDER_QUANTUM_FRAMES {
                out[f * ch + c] = dest.out[c][f];
            }
        }
        self.frame_clock += RENDER_QUANTUM_FRAMES as u64;
        out
    }

    /// Render `frames` frames (rounded up to whole render quanta) of the
    /// destination, returning the PLANAR per-channel buffers — convenient
    /// for tests and for `OfflineAudioContext.startRendering()`.
    pub fn render_planar(&mut self, frames: usize) -> Vec<Vec<f32>> {
        let ch = self.nodes[self.destination.0].channels;
        let mut planar = vec![Vec::with_capacity(frames); ch];
        let quanta = frames.div_ceil(RENDER_QUANTUM_FRAMES);
        for _ in 0..quanta {
            self.process_quantum();
            let dest = &self.nodes[self.destination.0];
            for c in 0..ch {
                planar[c].extend_from_slice(&dest.out[c]);
            }
            self.frame_clock += RENDER_QUANTUM_FRAMES as u64;
        }
        for chan in &mut planar {
            chan.truncate(frames);
        }
        planar
    }

    /// Read the most recently rendered time-domain samples of a node's
    /// output channel 0 (the data an `AnalyserNode.getFloatTimeDomainData`
    /// or `getByteTimeDomainData` would return). Must be called after a
    /// render pass.
    pub fn analyser_data(&self, id: NodeId) -> Vec<f32> {
        self.nodes
            .get(id.0)
            .map(|n| n.out[0].to_vec())
            .unwrap_or_default()
    }

    /// The pull-based processing pass. Topologically evaluate every node
    /// (sources first, destination last) so that when a node mixes its
    /// inputs they already hold this quantum's output.
    fn process_quantum(&mut self) {
        let order = self.topo_order();
        for id in order {
            self.process_node(id);
        }
    }

    /// Kahn topological sort over the edge set. Nodes with no resolved
    /// predecessor (sources) come first; the destination comes last.
    /// Cycles (illegal in Web Audio without a DelayNode) are broken by
    /// appending any leftover nodes in id order.
    fn topo_order(&self) -> Vec<NodeId> {
        let n = self.nodes.len();
        let mut indeg = vec![0usize; n];
        for e in &self.edges {
            indeg[e.to.0] += 1;
        }
        let mut queue: Vec<usize> = (0..n).filter(|&i| indeg[i] == 0).collect();
        let mut order = Vec::with_capacity(n);
        let mut visited = vec![false; n];
        let mut head = 0;
        while head < queue.len() {
            let u = queue[head];
            head += 1;
            if visited[u] {
                continue;
            }
            visited[u] = true;
            order.push(NodeId(u));
            for e in &self.edges {
                if e.from.0 == u {
                    indeg[e.to.0] = indeg[e.to.0].saturating_sub(1);
                    if indeg[e.to.0] == 0 && !visited[e.to.0] {
                        queue.push(e.to.0);
                    }
                }
            }
        }
        // Any node not reached (part of a cycle) — append deterministically.
        for u in 0..n {
            if !visited[u] {
                order.push(NodeId(u));
            }
        }
        order
    }

    /// Mix every edge that targets `id` into a fresh planar input buffer,
    /// up-/down-mixing predecessors to `id`'s channel count (spec §4 mixing:
    /// mono→stereo duplicates; extra channels are summed/dropped — we use
    /// the common "speakers" up-mix: mono fans out to all channels).
    fn gather_inputs(&self, id: NodeId, channels: usize) -> Vec<[f32; RENDER_QUANTUM_FRAMES]> {
        let mut acc = vec![[0.0f32; RENDER_QUANTUM_FRAMES]; channels];
        for e in &self.edges {
            if e.to != id {
                continue;
            }
            let src = &self.nodes[e.from.0];
            for c in 0..channels {
                // Up-mix: if the source is mono, read its single channel for
                // every output channel; otherwise read the matching channel.
                let sc = if src.channels == 1 { 0 } else { c.min(src.channels - 1) };
                for f in 0..RENDER_QUANTUM_FRAMES {
                    acc[c][f] += src.out[sc][f];
                }
            }
        }
        acc
    }

    fn process_node(&mut self, id: NodeId) {
        let channels = self.nodes[id.0].channels;
        // Sources don't read inputs; effects/destination do.
        let kind = self.nodes[id.0].kind.clone();
        let base_frame = self.frame_clock;
        let sample_rate_f = self.sample_rate as f32;
        match kind {
            NodeKind::Oscillator {
                osc_type,
                frequency,
                detune,
                mut phase,
                start_frame,
                stop_frame,
                playing,
            } => {
                // Effective frequency including detune (cents → ratio).
                let freq = frequency * 2.0f32.powf(detune / 1200.0);
                let inc = freq / sample_rate_f; // cycles per sample
                let node = &mut self.nodes[id.0];
                for f in 0..RENDER_QUANTUM_FRAMES {
                    let frame = base_frame + f as u64;
                    let active = playing && frame >= start_frame && frame < stop_frame;
                    let s = if active {
                        osc_sample(osc_type, phase)
                    } else {
                        0.0
                    };
                    node.out[0][f] = s;
                    // Phase advances only while the oscillator is running so a
                    // not-yet-started oscillator begins at phase 0.
                    if playing && frame >= start_frame {
                        phase += inc;
                        if phase >= 1.0 {
                            phase -= phase.floor();
                        }
                    }
                }
                // Persist running phase back into the node.
                if let NodeKind::Oscillator { phase: p, .. } = &mut node.kind {
                    *p = phase;
                }
            }
            NodeKind::BufferSource {
                buffer,
                mut cursor,
                playback_rate,
                detune,
                loop_,
                start_frame,
                stop_frame,
                playing,
            } => {
                let step =
                    f64::from(playback_rate * 2.0f32.powf(detune / 1200.0)).max(0.0);
                let node = &mut self.nodes[id.0];
                for arr in node.out.iter_mut() {
                    *arr = [0.0; RENDER_QUANTUM_FRAMES];
                }
                if let Some(buf) = &buffer {
                    let len = buf.length();
                    for f in 0..RENDER_QUANTUM_FRAMES {
                        let frame = base_frame + f as u64;
                        if !playing || frame < start_frame || frame >= stop_frame {
                            continue;
                        }
                        if len == 0 {
                            continue;
                        }
                        let idx = cursor as usize;
                        if idx >= len {
                            if loop_ {
                                cursor = 0.0;
                            } else {
                                continue;
                            }
                        }
                        let idx = (cursor as usize).min(len.saturating_sub(1));
                        for c in 0..channels {
                            let sc = c.min(buf.number_of_channels().saturating_sub(1));
                            node.out[c][f] = buf.channels[sc][idx];
                        }
                        cursor += step;
                        if loop_ && cursor as usize >= len {
                            cursor -= len as f64;
                        }
                    }
                }
                if let NodeKind::BufferSource { cursor: cur, .. } = &mut node.kind {
                    *cur = cursor;
                }
            }
            NodeKind::Gain { gain } => {
                let inputs = self.gather_inputs(id, channels);
                let node = &mut self.nodes[id.0];
                for c in 0..channels {
                    for f in 0..RENDER_QUANTUM_FRAMES {
                        node.out[c][f] = inputs[c][f] * gain;
                    }
                }
            }
            NodeKind::PassThrough | NodeKind::Destination => {
                let inputs = self.gather_inputs(id, channels);
                let node = &mut self.nodes[id.0];
                for c in 0..channels {
                    node.out[c] = inputs[c];
                }
            }
        }
    }
}

impl AudioGraph {
    /// Render `frames` and push the interleaved result into an
    /// [`crate::wasapi::OutputStream`]. This is the bridge between the JS
    /// node graph and the device feeder: the audio render thread calls this
    /// to keep the endpoint buffer fed, and the device pulls from the same
    /// stream via `pull_pcm_for_device`. Returns the number of interleaved
    /// samples pushed.
    pub fn render_into_stream(
        &mut self,
        stream: &crate::wasapi::OutputStream,
        frames: usize,
    ) -> usize {
        let quanta = frames.div_ceil(RENDER_QUANTUM_FRAMES);
        let mut total = 0;
        for _ in 0..quanta {
            let q = self.render_quantum();
            stream.push_pcm(&q);
            total += q.len();
        }
        total
    }
}

/// One sample of a band-unlimited periodic waveform at `phase` (cycles,
/// in [0,1)). These are the *ideal* (non-bandlimited) shapes; Chrome uses
/// a `PeriodicWave` wavetable for alias suppression, but the ideal shapes
/// are correct in waveform and amplitude — which is what we assert.
/// Spec §1.7.10 defines the waveforms; amplitudes are in [-1, 1].
fn osc_sample(t: OscillatorType, phase: f32) -> f32 {
    let p = phase - phase.floor();
    match t {
        OscillatorType::Sine => (2.0 * PI * p).sin(),
        OscillatorType::Square => {
            if p < 0.5 {
                1.0
            } else {
                -1.0
            }
        }
        OscillatorType::Sawtooth => 2.0 * p - 1.0,
        OscillatorType::Triangle => {
            // 0→1 ramps up to +1 at 0.25, down to -1 at 0.75, back to 0.
            if p < 0.25 {
                4.0 * p
            } else if p < 0.75 {
                2.0 - 4.0 * p
            } else {
                4.0 * p - 4.0
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Count zero crossings (sign changes) in a signal and convert to Hz.
    fn zero_crossing_freq(samples: &[f32], sample_rate: u32) -> f64 {
        let mut crossings = 0usize;
        for w in samples.windows(2) {
            if (w[0] <= 0.0 && w[1] > 0.0) || (w[0] >= 0.0 && w[1] < 0.0) {
                crossings += 1;
            }
        }
        // Two zero crossings per cycle.
        let cycles = crossings as f64 / 2.0;
        let seconds = samples.len() as f64 / f64::from(sample_rate);
        cycles / seconds
    }

    #[test]
    fn oscillator_440hz_produces_440hz_sine() {
        let sr = 48_000;
        let mut g = AudioGraph::new(sr);
        let osc = g.create_oscillator();
        g.set_osc_frequency(osc, 440.0);
        g.connect(osc, g.destination());
        g.start(osc, 0.0);
        // Render ~1 second.
        let planar = g.render_planar(sr as usize);
        let mono = &planar[0];
        let f = zero_crossing_freq(mono, sr);
        assert!(
            (f - 440.0).abs() < 1.0,
            "expected ~440Hz, measured {f}Hz from zero crossings"
        );
        // Amplitude must reach near ±1 (it's a real sine, not silence).
        let peak = mono.iter().cloned().fold(0.0f32, |a, b| a.max(b.abs()));
        assert!(peak > 0.99, "sine peak {peak} should be ~1.0");
    }

    #[test]
    fn oscillator_sine_matches_analytic_shape() {
        let sr = 48_000;
        let mut g = AudioGraph::new(sr);
        let osc = g.create_oscillator();
        g.set_osc_frequency(osc, 1000.0);
        g.connect(osc, g.destination());
        g.start(osc, 0.0);
        let planar = g.render_planar(RENDER_QUANTUM_FRAMES);
        let out = &planar[0];
        let inc = 1000.0f32 / sr as f32;
        for (i, &v) in out.iter().enumerate() {
            let expected = (2.0 * PI * (inc * i as f32)).sin();
            assert!(
                (v - expected).abs() < 1e-3,
                "sample {i}: got {v}, expected {expected}"
            );
        }
    }

    #[test]
    fn gain_half_halves_amplitude() {
        let sr = 48_000;
        let mut g = AudioGraph::new(sr);
        let osc = g.create_oscillator();
        g.set_osc_frequency(osc, 440.0);
        let gain = g.create_gain();
        g.set_gain(gain, 0.5);
        // source -> gain -> destination
        g.connect(osc, gain);
        g.connect(gain, g.destination());
        g.start(osc, 0.0);

        let with_gain = g.render_planar(sr as usize);
        let peak_g = with_gain[0]
            .iter()
            .cloned()
            .fold(0.0f32, |a, b| a.max(b.abs()));

        // Reference: same osc straight to destination (gain 1.0).
        let mut g2 = AudioGraph::new(sr);
        let osc2 = g2.create_oscillator();
        g2.set_osc_frequency(osc2, 440.0);
        g2.connect(osc2, g2.destination());
        g2.start(osc2, 0.0);
        let no_gain = g2.render_planar(sr as usize);
        let peak_ref = no_gain[0]
            .iter()
            .cloned()
            .fold(0.0f32, |a, b| a.max(b.abs()));

        assert!(
            (peak_g - peak_ref * 0.5).abs() < 1e-3,
            "gain 0.5 peak {peak_g} should be half of {peak_ref}"
        );
    }

    #[test]
    fn chain_source_gain_destination_reflects_chain() {
        // Destination output must equal source output * gain, sample for
        // sample — proves the chain actually carries audio through.
        let sr = 8_000;
        let mut g = AudioGraph::new(sr);
        let osc = g.create_oscillator();
        g.set_osc_type(osc, OscillatorType::Sawtooth);
        g.set_osc_frequency(osc, 100.0);
        let gain = g.create_gain();
        g.set_gain(gain, 0.25);
        g.connect(osc, gain);
        g.connect(gain, g.destination());
        g.start(osc, 0.0);
        let planar = g.render_planar(RENDER_QUANTUM_FRAMES);
        // Independently regenerate the oscillator and scale by 0.25. Use the
        // SAME phase accumulation the node uses (`phase += inc` per sample,
        // wrapping at 1.0) so we compare DSP-for-DSP rather than fighting
        // float drift between `inc*i` and an accumulator near a wrap.
        let inc = 100.0f32 / sr as f32;
        let mut phase = 0.0f32;
        for (i, &v) in planar[0].iter().enumerate() {
            let expected = osc_sample(OscillatorType::Sawtooth, phase) * 0.25;
            assert!(
                (v - expected).abs() < 1e-4,
                "frame {i}: chain {v} != expected {expected}"
            );
            phase += inc;
            if phase >= 1.0 {
                phase -= phase.floor();
            }
        }
    }

    #[test]
    fn disconnected_source_is_silent_at_destination() {
        let sr = 48_000;
        let mut g = AudioGraph::new(sr);
        let osc = g.create_oscillator();
        g.start(osc, 0.0);
        // Never connected to destination.
        let planar = g.render_planar(RENDER_QUANTUM_FRAMES);
        for &v in &planar[0] {
            assert_eq!(v, 0.0);
        }
    }

    #[test]
    fn buffer_source_plays_pcm_through_destination() {
        let sr = 8_000;
        let mut g = AudioGraph::new(sr);
        let src = g.create_buffer_source();
        let mut buf = AudioBuffer::new(1, 200, sr);
        for (i, s) in buf.channels[0].iter_mut().enumerate() {
            *s = (i as f32) / 200.0; // ramp 0 → ~1
        }
        g.set_buffer(src, buf);
        g.connect(src, g.destination());
        g.start(src, 0.0);
        let planar = g.render_planar(RENDER_QUANTUM_FRAMES);
        // First 128 frames should be the ramp values 0..128/200.
        for (i, &v) in planar[0].iter().enumerate().take(128) {
            let expected = (i as f32) / 200.0;
            assert!((v - expected).abs() < 1e-5, "frame {i}: {v} != {expected}");
        }
    }

    #[test]
    fn stop_silences_oscillator() {
        let sr = 8_000;
        let mut g = AudioGraph::new(sr);
        let osc = g.create_oscillator();
        g.set_osc_frequency(osc, 200.0);
        g.connect(osc, g.destination());
        g.start(osc, 0.0);
        // Stop after 64 frames.
        g.stop(osc, 64.0 / f64::from(sr));
        let planar = g.render_planar(RENDER_QUANTUM_FRAMES);
        // Frames >= 64 must be silent.
        for (i, &v) in planar[0].iter().enumerate() {
            if i >= 64 {
                assert_eq!(v, 0.0, "frame {i} should be silent after stop");
            }
        }
    }

    #[test]
    fn render_quantum_is_128_frames_interleaved() {
        let mut g = AudioGraph::new(48_000);
        let q = g.render_quantum();
        // destination is stereo → 128 * 2 = 256 samples.
        assert_eq!(q.len(), RENDER_QUANTUM_FRAMES * 2);
    }

    #[test]
    fn current_time_advances_by_quantum() {
        let mut g = AudioGraph::new(48_000);
        assert_eq!(g.current_time(), 0.0);
        g.render_quantum();
        let expect = RENDER_QUANTUM_FRAMES as f64 / 48_000.0;
        assert!((g.current_time() - expect).abs() < 1e-12);
    }

    #[test]
    fn render_into_stream_feeds_outputstream() {
        use crate::wasapi::{OutputFormat, OutputStream, StreamCategory};
        let sr = 48_000;
        let mut g = AudioGraph::new(sr);
        let osc = g.create_oscillator();
        g.set_osc_frequency(osc, 440.0);
        g.connect(osc, g.destination());
        g.start(osc, 0.0);
        let stream = OutputStream::new(
            OutputFormat {
                sample_rate: sr,
                channels: 2,
                bits_per_sample: 32,
            },
            StreamCategory::Media,
        );
        // Render two quanta worth.
        let pushed = g.render_into_stream(&stream, RENDER_QUANTUM_FRAMES * 2);
        assert_eq!(pushed, RENDER_QUANTUM_FRAMES * 2 * 2); // *2 channels
        assert_eq!(stream.queued_frames(), RENDER_QUANTUM_FRAMES * 2);
        // The pulled audio is non-silent (real sine on both channels via
        // mono→stereo up-mix).
        let pulled = stream.pull_pcm_for_device(RENDER_QUANTUM_FRAMES);
        let peak = pulled.iter().cloned().fold(0.0f32, |a, b| a.max(b.abs()));
        assert!(peak > 0.0, "stream-fed audio must not be silence");
    }

    #[test]
    fn square_wave_is_plus_minus_one() {
        let mut g = AudioGraph::new(48_000);
        let osc = g.create_oscillator();
        g.set_osc_type(osc, OscillatorType::Square);
        g.set_osc_frequency(osc, 500.0);
        g.connect(osc, g.destination());
        g.start(osc, 0.0);
        let planar = g.render_planar(RENDER_QUANTUM_FRAMES);
        for &v in &planar[0] {
            assert!(v == 1.0 || v == -1.0, "square sample {v} not ±1");
        }
    }
}
