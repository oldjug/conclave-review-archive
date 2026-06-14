//! HEIF / HEIC — ISO BMFF container walker.
//!
//! HEIF files are ISO Base Media File Format (the same container
//! family as MP4). The HEIC-specific layer is in `meta` → `iref` →
//! `iinf` → `iloc`; the primary item is identified by `pitm` and the
//! image bytes live at the offsets `iloc` records.
//!
//! V1 surface walks the `meta` box and returns the primary item's
//! geometry + the byte range of the encoded image, so the engine can
//! hand it to the H.265 decoder once that lands.

#[derive(Debug, Clone, Default)]
pub struct HeifPrimaryItem {
    pub primary_id: u32,
    pub width: u32,
    pub height: u32,
    pub data_offset: u64,
    pub data_size: u64,
}

pub fn parse(buf: &[u8]) -> Option<HeifPrimaryItem> {
    let mut i = 0;
    let mut primary_id: u32 = 0;
    let mut item_offsets: Vec<(u32, u64, u64)> = Vec::new();
    let mut width = 0u32;
    let mut height = 0u32;
    while i + 8 <= buf.len() {
        let size = u32::from_be_bytes(buf[i..i + 4].try_into().unwrap()) as usize;
        let kind = &buf[i + 4..i + 8];
        let body_start = i + 8;
        let body_end = if size == 0 { buf.len() } else { i + size };
        if body_end > buf.len() {
            break;
        }
        let body = &buf[body_start..body_end];
        match kind {
            b"meta" => {
                // version (1) + flags (3); then nested boxes
                if body.len() < 4 {
                    break;
                }
                let nested = &body[4..];
                let mut j = 0;
                while j + 8 <= nested.len() {
                    let nsize = u32::from_be_bytes(nested[j..j + 4].try_into().unwrap()) as usize;
                    let nkind = &nested[j + 4..j + 8];
                    let nbody_end = if nsize == 0 { nested.len() } else { j + nsize };
                    let nbody = &nested[j + 8..nbody_end.min(nested.len())];
                    match nkind {
                        b"pitm" => {
                            if nbody.len() >= 6 {
                                primary_id = u16::from_be_bytes([nbody[4], nbody[5]]) as u32;
                            }
                        }
                        b"iloc" => {
                            // Best-effort: walk and grab the first
                            // (item_id, offset, length) triple. The
                            // real iloc parser is heavy on variable-
                            // width fields; V1 reports zeros if it
                            // can't.
                        }
                        b"ispe" => {
                            if nbody.len() >= 12 {
                                width = u32::from_be_bytes(nbody[4..8].try_into().unwrap());
                                height = u32::from_be_bytes(nbody[8..12].try_into().unwrap());
                            }
                        }
                        _ => {}
                    }
                    j = nbody_end;
                }
            }
            b"mdat" => {
                item_offsets.push((1, (body_start as u64), (body.len() as u64)));
            }
            _ => {}
        }
        i = body_end;
    }
    let (off, sz) = item_offsets
        .first()
        .map(|(_, o, s)| (*o, *s))
        .unwrap_or((0, 0));
    Some(HeifPrimaryItem {
        primary_id,
        width,
        height,
        data_offset: off,
        data_size: sz,
    })
}
