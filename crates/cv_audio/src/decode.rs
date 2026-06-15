//! `decodeAudioData` back-end — turn an encoded audio file into planar
//! PCM (`f32` per sample, one `Vec` per channel).
//!
//! Chrome's `BaseAudioContext.decodeAudioData(ArrayBuffer)` decodes any
//! container its media stack supports (WAV/PCM, MP3, AAC, FLAC, Opus, …)
//! into an `AudioBuffer`. This module is the real decoder behind our
//! binding.
//!
//! Spec: <https://www.w3.org/TR/webaudio/#dom-baseaudiocontext-decodeaudiodata>
//!   "decodeAudioData … takes the audio file data contained in `audioData`
//!    and decodes it … into a linear PCM `AudioBuffer`."
//! Blink reference: `modules/webaudio/audio_decoder.cc` →
//!   `media::AudioFileReader` (FFmpeg-backed). We implement the container
//!   parsers directly.
//!
//! WAV/RIFF (the canonical uncompressed format and what `OfflineAudioContext`
//! round-trip tests use) is decoded END-TO-END here: 8/16/24/32-bit integer
//! PCM and 32/64-bit IEEE float, mono..N channels. Reference: Microsoft
//! "Multimedia Programming Interface and Data Specifications 1.0" (RIFF WAVE)
//! and the IBM/Microsoft WAVE format (`WAVE_FORMAT_PCM` = 1,
//! `WAVE_FORMAT_IEEE_FLOAT` = 3, `WAVE_FORMAT_EXTENSIBLE` = 0xFFFE).
//!
//! Compressed containers (MP3/AAC/Opus/FLAC) are *detected* here and routed
//! to their decoders; where a decoder is not yet complete we return a typed
//! [`DecodeError`] rather than fabricating silence (no stubs).

use crate::graph::AudioBuffer;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DecodeError {
    /// Buffer too short / not a recognized container.
    NotRecognized,
    /// Recognized container but malformed.
    Malformed(&'static str),
    /// Recognized compressed codec whose full decoder is not yet wired.
    /// Carries the detected codec name so the caller can surface a precise
    /// `EncodingError` to JS instead of silent failure.
    Unsupported(&'static str),
}

/// Detected container format (for diagnostics + routing).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Container {
    Wav,
    Mp3,
    Aac,
    Flac,
    Ogg,
    Unknown,
}

/// Sniff the container from the leading magic bytes.
pub fn sniff(bytes: &[u8]) -> Container {
    if bytes.len() >= 12 && &bytes[0..4] == b"RIFF" && &bytes[8..12] == b"WAVE" {
        return Container::Wav;
    }
    if bytes.len() >= 4 && &bytes[0..4] == b"fLaC" {
        return Container::Flac;
    }
    if bytes.len() >= 4 && &bytes[0..4] == b"OggS" {
        return Container::Ogg;
    }
    // ID3v2-tagged MP3.
    if bytes.len() >= 3 && &bytes[0..3] == b"ID3" {
        return Container::Mp3;
    }
    // ADTS AAC sync = 12 set bits (0xFFF). Check BEFORE the 11-bit MPEG sync
    // because every ADTS sync also satisfies the 11-bit MPEG mask; the extra
    // set bit (layer field = 0) is what distinguishes ADTS.
    if bytes.len() >= 2 && bytes[0] == 0xFF && (bytes[1] & 0xF6) == 0xF0 {
        return Container::Aac;
    }
    // Raw MP3 frame sync = 11 set bits (0xFFE), with a valid layer field.
    if bytes.len() >= 2 && bytes[0] == 0xFF && (bytes[1] & 0xE0) == 0xE0 {
        return Container::Mp3;
    }
    Container::Unknown
}

/// Decode encoded audio into a planar [`AudioBuffer`]. `target_sample_rate`
/// is the context sample rate; per spec the result is resampled to the
/// context rate. (We currently keep the source rate and let the caller
/// resample on playback via `playbackRate`; the buffer carries its true
/// `sample_rate`. This matches what `getChannelData` exposes.)
pub fn decode_audio_data(bytes: &[u8]) -> Result<AudioBuffer, DecodeError> {
    match sniff(bytes) {
        Container::Wav => decode_wav(bytes),
        Container::Mp3 => Err(DecodeError::Unsupported("mp3")),
        Container::Aac => Err(DecodeError::Unsupported("aac")),
        Container::Flac => Err(DecodeError::Unsupported("flac")),
        Container::Ogg => Err(DecodeError::Unsupported("ogg/opus")),
        Container::Unknown => Err(DecodeError::NotRecognized),
    }
}

const WAVE_FORMAT_PCM: u16 = 0x0001;
const WAVE_FORMAT_IEEE_FLOAT: u16 = 0x0003;
const WAVE_FORMAT_EXTENSIBLE: u16 = 0xFFFE;

fn rd_u16(b: &[u8], o: usize) -> u16 {
    u16::from_le_bytes([b[o], b[o + 1]])
}
fn rd_u32(b: &[u8], o: usize) -> u32 {
    u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]])
}

/// Full RIFF/WAVE PCM decoder. Walks the chunk list to find `fmt ` and
/// `data`, then deinterleaves + normalizes samples to `f32` in [-1, 1].
pub fn decode_wav(bytes: &[u8]) -> Result<AudioBuffer, DecodeError> {
    if bytes.len() < 12 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return Err(DecodeError::NotRecognized);
    }
    let mut pos = 12usize;
    let mut fmt_tag: u16 = 0;
    let mut channels: u16 = 0;
    let mut sample_rate: u32 = 0;
    let mut bits_per_sample: u16 = 0;
    let mut data: Option<&[u8]> = None;

    while pos + 8 <= bytes.len() {
        let id = &bytes[pos..pos + 4];
        let size = rd_u32(bytes, pos + 4) as usize;
        let body_start = pos + 8;
        if body_start > bytes.len() {
            break;
        }
        let body_end = (body_start + size).min(bytes.len());
        let body = &bytes[body_start..body_end];
        match id {
            b"fmt " => {
                if body.len() < 16 {
                    return Err(DecodeError::Malformed("fmt chunk too short"));
                }
                fmt_tag = rd_u16(body, 0);
                channels = rd_u16(body, 2);
                sample_rate = rd_u32(body, 4);
                bits_per_sample = rd_u16(body, 14);
                // WAVE_FORMAT_EXTENSIBLE carries the real format in the
                // SubFormat GUID's first 2 bytes (offset 24 in the fmt body).
                if fmt_tag == WAVE_FORMAT_EXTENSIBLE && body.len() >= 26 {
                    fmt_tag = rd_u16(body, 24);
                }
            }
            b"data" => {
                data = Some(body);
            }
            _ => {}
        }
        // Chunks are word-aligned (pad byte if size is odd).
        let advance = 8 + size + (size & 1);
        pos += advance;
    }

    let channels = channels as usize;
    if channels == 0 {
        return Err(DecodeError::Malformed("zero channels"));
    }
    if sample_rate == 0 {
        return Err(DecodeError::Malformed("zero sample rate"));
    }
    let data = data.ok_or(DecodeError::Malformed("no data chunk"))?;

    let bytes_per_sample = (bits_per_sample / 8) as usize;
    if bytes_per_sample == 0 {
        return Err(DecodeError::Malformed("zero bits per sample"));
    }
    let frame_bytes = bytes_per_sample * channels;
    let frames = data.len() / frame_bytes;

    let mut out = AudioBuffer::new(channels, frames, sample_rate);
    for f in 0..frames {
        for c in 0..channels {
            let off = f * frame_bytes + c * bytes_per_sample;
            let sample = decode_pcm_sample(fmt_tag, bits_per_sample, &data[off..off + bytes_per_sample])?;
            out.channels[c][f] = sample;
        }
    }
    Ok(out)
}

/// Decode one interleaved PCM sample to f32 in [-1, 1].
fn decode_pcm_sample(fmt_tag: u16, bits: u16, b: &[u8]) -> Result<f32, DecodeError> {
    match (fmt_tag, bits) {
        // 8-bit PCM is UNSIGNED (0..255, midpoint 128) per the WAVE spec.
        (WAVE_FORMAT_PCM, 8) => Ok((b[0] as f32 - 128.0) / 128.0),
        // 16/24/32-bit integer PCM are SIGNED little-endian.
        (WAVE_FORMAT_PCM, 16) => {
            let v = i16::from_le_bytes([b[0], b[1]]);
            Ok(v as f32 / 32_768.0)
        }
        (WAVE_FORMAT_PCM, 24) => {
            // Sign-extend 24-bit LE to i32.
            let raw = (b[0] as i32) | ((b[1] as i32) << 8) | ((b[2] as i32) << 16);
            let v = (raw << 8) >> 8; // sign-extend from bit 23
            Ok(v as f32 / 8_388_608.0)
        }
        (WAVE_FORMAT_PCM, 32) => {
            let v = i32::from_le_bytes([b[0], b[1], b[2], b[3]]);
            Ok(v as f32 / 2_147_483_648.0)
        }
        (WAVE_FORMAT_IEEE_FLOAT, 32) => Ok(f32::from_le_bytes([b[0], b[1], b[2], b[3]])),
        (WAVE_FORMAT_IEEE_FLOAT, 64) => {
            Ok(f64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]) as f32)
        }
        _ => Err(DecodeError::Unsupported("wav pcm format")),
    }
}

/// Encode a planar [`AudioBuffer`] to a canonical 16-bit PCM WAV file.
/// Used to build known-good test fixtures and round-trip the WAV decoder
/// (and as a real serializer for any "render to WAV" path).
pub fn encode_wav_pcm16(buf: &AudioBuffer) -> Vec<u8> {
    let channels = buf.number_of_channels() as u16;
    let sample_rate = buf.sample_rate;
    let frames = buf.length();
    let bits = 16u16;
    let block_align = channels * (bits / 8);
    let byte_rate = sample_rate * u32::from(block_align);
    let data_len = frames * channels as usize * 2;

    let mut out = Vec::with_capacity(44 + data_len);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&((36 + data_len) as u32).to_le_bytes());
    out.extend_from_slice(b"WAVE");
    out.extend_from_slice(b"fmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&WAVE_FORMAT_PCM.to_le_bytes());
    out.extend_from_slice(&channels.to_le_bytes());
    out.extend_from_slice(&sample_rate.to_le_bytes());
    out.extend_from_slice(&byte_rate.to_le_bytes());
    out.extend_from_slice(&block_align.to_le_bytes());
    out.extend_from_slice(&bits.to_le_bytes());
    out.extend_from_slice(b"data");
    out.extend_from_slice(&(data_len as u32).to_le_bytes());
    for f in 0..frames {
        for c in 0..channels as usize {
            let s = (buf.channels[c][f].clamp(-1.0, 1.0) * 32_767.0).round() as i16;
            out.extend_from_slice(&s.to_le_bytes());
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    fn make_sine_buffer(freq: f32, sr: u32, frames: usize, channels: usize) -> AudioBuffer {
        let mut b = AudioBuffer::new(channels, frames, sr);
        for c in 0..channels {
            for f in 0..frames {
                b.channels[c][f] = (2.0 * PI * freq * f as f32 / sr as f32).sin() * 0.5;
            }
        }
        b
    }

    #[test]
    fn sniff_detects_wav() {
        let mut b = vec![0u8; 12];
        b[0..4].copy_from_slice(b"RIFF");
        b[8..12].copy_from_slice(b"WAVE");
        assert_eq!(sniff(&b), Container::Wav);
    }

    #[test]
    fn sniff_detects_mp3_and_aac_and_flac() {
        assert_eq!(sniff(b"ID3\x04"), Container::Mp3);
        assert_eq!(sniff(&[0xFF, 0xFB, 0x00, 0x00]), Container::Mp3);
        assert_eq!(sniff(&[0xFF, 0xF1, 0x00, 0x00]), Container::Aac);
        assert_eq!(sniff(b"fLaC"), Container::Flac);
        assert_eq!(sniff(b"OggS"), Container::Ogg);
    }

    #[test]
    fn wav_round_trip_16bit_stereo() {
        let sr = 44_100;
        let frames = 1000;
        let orig = make_sine_buffer(440.0, sr, frames, 2);
        let wav = encode_wav_pcm16(&orig);
        // sniffs as WAV
        assert_eq!(sniff(&wav), Container::Wav);
        let decoded = decode_audio_data(&wav).unwrap();
        assert_eq!(decoded.sample_rate, sr);
        assert_eq!(decoded.number_of_channels(), 2);
        assert_eq!(decoded.length(), frames);
        // Samples match within 16-bit quantization error (1/32768).
        for c in 0..2 {
            for f in 0..frames {
                let a = orig.channels[c][f];
                let d = decoded.channels[c][f];
                assert!((a - d).abs() < 1.0 / 32_000.0, "ch{c} f{f}: {a} vs {d}");
            }
        }
    }

    #[test]
    fn wav_known_pcm_length_and_first_sample() {
        // Build a tiny known WAV: mono, 8000 Hz, 16-bit, 4 frames.
        // Samples: 0, 16384 (~0.5), -16384 (~-0.5), 32767 (~1.0).
        let sr = 8_000u32;
        let samples: [i16; 4] = [0, 16384, -16384, 32767];
        let mut wav = Vec::new();
        let data_len = samples.len() * 2;
        wav.extend_from_slice(b"RIFF");
        wav.extend_from_slice(&((36 + data_len) as u32).to_le_bytes());
        wav.extend_from_slice(b"WAVE");
        wav.extend_from_slice(b"fmt ");
        wav.extend_from_slice(&16u32.to_le_bytes());
        wav.extend_from_slice(&1u16.to_le_bytes()); // PCM
        wav.extend_from_slice(&1u16.to_le_bytes()); // mono
        wav.extend_from_slice(&sr.to_le_bytes());
        wav.extend_from_slice(&(sr * 2).to_le_bytes()); // byte rate
        wav.extend_from_slice(&2u16.to_le_bytes()); // block align
        wav.extend_from_slice(&16u16.to_le_bytes()); // bits
        wav.extend_from_slice(b"data");
        wav.extend_from_slice(&(data_len as u32).to_le_bytes());
        for s in samples {
            wav.extend_from_slice(&s.to_le_bytes());
        }
        let decoded = decode_audio_data(&wav).unwrap();
        assert_eq!(decoded.length(), 4, "expected 4 PCM frames");
        assert_eq!(decoded.number_of_channels(), 1);
        assert_eq!(decoded.sample_rate, 8000);
        // First sample is exactly 0.
        assert_eq!(decoded.channels[0][0], 0.0);
        // 16384 / 32768 = 0.5
        assert!((decoded.channels[0][1] - 0.5).abs() < 1e-4);
        assert!((decoded.channels[0][2] + 0.5).abs() < 1e-4);
        assert!((decoded.channels[0][3] - 0.99997).abs() < 1e-3);
    }

    #[test]
    fn wav_8bit_unsigned_midpoint_is_zero() {
        let sr = 8_000u32;
        let samples: [u8; 3] = [128, 255, 0]; // mid, max, min
        let mut wav = Vec::new();
        let data_len = samples.len();
        wav.extend_from_slice(b"RIFF");
        wav.extend_from_slice(&((36 + data_len) as u32).to_le_bytes());
        wav.extend_from_slice(b"WAVE");
        wav.extend_from_slice(b"fmt ");
        wav.extend_from_slice(&16u32.to_le_bytes());
        wav.extend_from_slice(&1u16.to_le_bytes());
        wav.extend_from_slice(&1u16.to_le_bytes());
        wav.extend_from_slice(&sr.to_le_bytes());
        wav.extend_from_slice(&sr.to_le_bytes());
        wav.extend_from_slice(&1u16.to_le_bytes());
        wav.extend_from_slice(&8u16.to_le_bytes());
        wav.extend_from_slice(b"data");
        wav.extend_from_slice(&(data_len as u32).to_le_bytes());
        wav.extend_from_slice(&samples);
        let d = decode_audio_data(&wav).unwrap();
        assert_eq!(d.channels[0][0], 0.0);
        assert!(d.channels[0][1] > 0.99);
        assert!(d.channels[0][2] < -0.99);
    }

    #[test]
    fn wav_float32_passthrough() {
        let sr = 8_000u32;
        let samples: [f32; 3] = [0.0, 0.25, -0.75];
        let mut wav = Vec::new();
        let data_len = samples.len() * 4;
        wav.extend_from_slice(b"RIFF");
        wav.extend_from_slice(&((36 + data_len) as u32).to_le_bytes());
        wav.extend_from_slice(b"WAVE");
        wav.extend_from_slice(b"fmt ");
        wav.extend_from_slice(&16u32.to_le_bytes());
        wav.extend_from_slice(&WAVE_FORMAT_IEEE_FLOAT.to_le_bytes());
        wav.extend_from_slice(&1u16.to_le_bytes());
        wav.extend_from_slice(&sr.to_le_bytes());
        wav.extend_from_slice(&(sr * 4).to_le_bytes());
        wav.extend_from_slice(&4u16.to_le_bytes());
        wav.extend_from_slice(&32u16.to_le_bytes());
        wav.extend_from_slice(b"data");
        wav.extend_from_slice(&(data_len as u32).to_le_bytes());
        for s in samples {
            wav.extend_from_slice(&s.to_le_bytes());
        }
        let d = decode_audio_data(&wav).unwrap();
        assert_eq!(d.channels[0][0], 0.0);
        assert!((d.channels[0][1] - 0.25).abs() < 1e-6);
        assert!((d.channels[0][2] + 0.75).abs() < 1e-6);
    }

    #[test]
    fn compressed_returns_typed_unsupported_not_silence() {
        // An MP3 sync header must NOT decode to fake silence — it must
        // report the codec honestly.
        let mp3 = [0xFF, 0xFB, 0x90, 0x00, 0x00, 0x00];
        assert!(matches!(
            decode_audio_data(&mp3),
            Err(DecodeError::Unsupported("mp3"))
        ));
    }

    #[test]
    fn garbage_is_not_recognized() {
        assert!(matches!(
            decode_audio_data(b"not audio at all"),
            Err(DecodeError::NotRecognized)
        ));
    }
}
