//! AV1 (AOMediaCodec/AV1) — OBU walker + sequence-header surfacer.

#[derive(Debug, Clone, Copy)]
pub struct Av1Header {
    pub profile: u8,
    pub width: u32,
    pub height: u32,
}

pub fn parse_sequence_header(buf: &[u8]) -> Option<Av1Header> {
    // OBU framing: obu_header(1) + obu_size(leb128) + payload
    if buf.len() < 2 {
        return None;
    }
    let obu_type = (buf[0] >> 3) & 0x0F;
    if obu_type != 1 {
        return None;
    }
    let mut i = 1;
    let (size, n) = read_leb128(&buf[i..])?;
    i += n;
    if i + size as usize > buf.len() {
        return None;
    }
    let p = &buf[i..i + size as usize];
    if p.len() < 4 {
        return None;
    }
    let profile = (p[0] >> 5) & 0x07;
    let width = u16::from_be_bytes([p[2], p[3]]) as u32 + 1;
    let height = u16::from_be_bytes([p[0] & 0xFF, p[1] & 0xFF]) as u32 + 1;
    Some(Av1Header {
        profile,
        width,
        height,
    })
}

fn read_leb128(buf: &[u8]) -> Option<(u64, usize)> {
    let mut value = 0u64;
    let mut shift = 0;
    for (i, &b) in buf.iter().enumerate().take(8) {
        value |= ((b & 0x7F) as u64) << shift;
        if b & 0x80 == 0 {
            return Some((value, i + 1));
        }
        shift += 7;
    }
    None
}
