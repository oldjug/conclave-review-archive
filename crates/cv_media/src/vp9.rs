//! VP9 uncompressed-header parser. Surfaces frame geometry, profile,
//! and key-frame flag so the demuxer can dispatch to the right driver.

#[derive(Debug, Clone, Default)]
pub struct Vp9Header {
    pub profile: u8,
    pub show_existing_frame: bool,
    pub key_frame: bool,
    pub width_minus_1: u16,
    pub height_minus_1: u16,
}

pub fn parse_uncompressed_header(buf: &[u8]) -> Option<Vp9Header> {
    if buf.len() < 3 {
        return None;
    }
    let mut r = BitReader::new(buf);
    let frame_marker = r.bits(2)?;
    if frame_marker != 2 {
        return None;
    }
    let p0 = r.bits(1)?;
    let p1 = r.bits(1)?;
    let profile = (p0 | (p1 << 1)) as u8;
    if profile == 3 {
        let _ = r.bits(1)?;
    }
    let show_existing_frame = r.bits(1)? == 1;
    if show_existing_frame {
        return Some(Vp9Header {
            profile,
            show_existing_frame: true,
            ..Default::default()
        });
    }
    let key_frame = r.bits(1)? == 0;
    let _show_frame = r.bits(1)?;
    let _error_resilient = r.bits(1)?;
    if !key_frame {
        return Some(Vp9Header {
            profile,
            show_existing_frame: false,
            key_frame: false,
            ..Default::default()
        });
    }
    let sync = r.bits(24)?;
    if sync != 0x49_83_42 {
        return None;
    }
    // Skip color config.
    let bit_depth = if profile >= 2 {
        if r.bits(1)? == 1 { 12 } else { 10 }
    } else {
        8
    };
    let _ = bit_depth;
    let _color_space = r.bits(3)?;
    if _color_space != 7 {
        let _color_range = r.bits(1)?;
        if profile == 1 || profile == 3 {
            let _ss_x = r.bits(1)?;
            let _ss_y = r.bits(1)?;
            let _ = r.bits(1)?;
        }
    } else if profile == 1 || profile == 3 {
        let _ = r.bits(1)?;
    }
    let frame_width_minus_1 = r.bits(16)? as u16;
    let frame_height_minus_1 = r.bits(16)? as u16;
    Some(Vp9Header {
        profile,
        show_existing_frame: false,
        key_frame: true,
        width_minus_1: frame_width_minus_1,
        height_minus_1: frame_height_minus_1,
    })
}

struct BitReader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> BitReader<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }
    fn bits(&mut self, n: u32) -> Option<u32> {
        let mut out = 0u32;
        for _ in 0..n {
            if self.pos >= self.buf.len() * 8 {
                return None;
            }
            let byte = self.buf[self.pos / 8];
            let bit = (byte >> (7 - (self.pos % 8))) & 1;
            out = (out << 1) | bit as u32;
            self.pos += 1;
        }
        Some(out)
    }
}
