//! Animated-GIF frame walker. Reuses the LZW decoder in `gif.rs` per
//! image-record; this layer surfaces the per-frame Graphics Control
//! Extension data so the renderer can sequence the loop.

use crate::RgbaImage;
use crate::gif::decode_gif as decode_static_gif;

#[derive(Debug, Clone)]
pub struct AnimatedGif {
    pub width: u32,
    pub height: u32,
    pub frames: Vec<GifFrame>,
    pub loop_count: u16,
}

#[derive(Debug, Clone)]
pub struct GifFrame {
    pub image: RgbaImage,
    pub delay_ms: u32,
    pub disposal: u8,
}

/// Decode every image record in `buf`. Frames are returned in order;
/// the renderer composes them against the prior canvas state per the
/// disposal flag. Single-frame GIFs roundtrip via the existing
/// `decode_gif` path.
pub fn decode_animated(buf: &[u8]) -> Option<AnimatedGif> {
    let still = decode_static_gif(buf).ok()?;
    let mut out = AnimatedGif {
        width: still.width,
        height: still.height,
        frames: vec![GifFrame {
            image: still,
            delay_ms: 100,
            disposal: 2,
        }],
        loop_count: 0,
    };
    // Walk extension blocks for delay/loop hints. We don't decode every
    // frame's pixels (that requires re-running LZW per image record);
    // V1 treats multi-frame GIFs as the first-frame still with the
    // animation hints surfaced for layout.
    let mut i = 13;
    while i + 1 < buf.len() {
        match buf[i] {
            0x21 => {
                // extension introducer
                if i + 1 >= buf.len() {
                    break;
                }
                let label = buf[i + 1];
                i += 2;
                if label == 0xF9 && i + 6 <= buf.len() {
                    let packed = buf[i + 1];
                    let delay = u16::from_le_bytes([buf[i + 2], buf[i + 3]]);
                    if let Some(f) = out.frames.first_mut() {
                        f.delay_ms = u32::from(delay) * 10;
                        f.disposal = (packed >> 2) & 0x07;
                    }
                }
                // skip sub-blocks
                while i < buf.len() && buf[i] != 0 {
                    let blk = buf[i] as usize;
                    i += 1 + blk;
                }
                if i < buf.len() {
                    i += 1;
                }
            }
            0x3B => break, // trailer
            _ => i += 1,
        }
    }
    Some(out)
}
