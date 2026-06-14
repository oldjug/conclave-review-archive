//! WOFF2 → SFNT decoder.
//!
//! Implements the W3C "WOFF File Format 2.0" Recommendation
//! (<https://www.w3.org/TR/WOFF2/>):
//!
//! - 48-byte header: signature `wOF2`, original sfnt flavor, lengths.
//! - Table directory: per-table flags + tag + UIntBase128 sizes. Tag
//!   may be packed into the flags byte via a 64-entry "known tags"
//!   lookup; flag bits 6-7 carry the transform version.
//! - Brotli-compressed table data block (decoded via `cv_compression`).
//! - For the `glyf` table the spec defines a special transformation
//!   (TripletEncoding for coordinates, 255UInt16 for counts, per-glyph
//!   bbox bitmap, separate composite + instruction streams) which we
//!   un-transform back to standard TrueType `glyf`. The `loca` table
//!   is regenerated from the resulting per-glyph offsets.
//! - Output is repacked as a standard SFNT (TrueType-flavoured)
//!   container that Windows GDI accepts via `AddFontMemResourceEx`.
//!
//! Untransformed glyf (transform_version 3 on TTF / always for CFF
//! fonts) is passed through unchanged.

#![allow(non_snake_case)]

use core::convert::TryInto;

#[derive(Debug)]
pub enum Woff2Error {
    Truncated(&'static str),
    BadSignature,
    Brotli(String),
    TooManyTables,
    BadGlyf(&'static str),
    Reserialize(&'static str),
}

impl core::fmt::Display for Woff2Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Truncated(s) => write!(f, "WOFF2 truncated: {s}"),
            Self::BadSignature => f.write_str("not a WOFF2 file (signature mismatch)"),
            Self::Brotli(s) => write!(f, "Brotli decompress failed: {s}"),
            Self::TooManyTables => f.write_str("WOFF2 numTables out of range"),
            Self::BadGlyf(s) => write!(f, "transformed glyf malformed: {s}"),
            Self::Reserialize(s) => write!(f, "SFNT reserialise: {s}"),
        }
    }
}

impl std::error::Error for Woff2Error {}

/// Decode a WOFF2 byte stream into a standard SFNT (TrueType /
/// OpenType) container. Returns the SFNT bytes ready to hand to
/// `AddFontMemResourceEx`.
pub fn decode_woff2(input: &[u8]) -> Result<Vec<u8>, Woff2Error> {
    let mut p = Cursor::new(input);
    if p.read_bytes(4)? != b"wOF2" {
        return Err(Woff2Error::BadSignature);
    }
    let flavor = p.u32()?;
    let _length = p.u32()?;
    let num_tables = p.u16()?;
    let _reserved = p.u16()?;
    let _total_sfnt_size = p.u32()?;
    let total_compressed_size = p.u32()?;
    let _major = p.u16()?;
    let _minor = p.u16()?;
    let _meta_offset = p.u32()?;
    let _meta_length = p.u32()?;
    let _meta_orig_length = p.u32()?;
    let _priv_offset = p.u32()?;
    let _priv_length = p.u32()?;

    if num_tables == 0 || (num_tables as usize) > 4096 {
        return Err(Woff2Error::TooManyTables);
    }

    // Read table directory.
    let mut tables: Vec<TableEntry> = Vec::with_capacity(num_tables as usize);
    for _ in 0..num_tables {
        tables.push(TableEntry::read(&mut p)?);
    }

    // Collection directory (TTC) — skip; we only handle single-face.
    let is_collection = flavor == u32::from_be_bytes(*b"ttcf");
    if is_collection {
        // Read collection header but bail — single-face fonts are
        // overwhelmingly common in @font-face use.
        return Err(Woff2Error::Reserialize("TTC collections not yet supported"));
    }

    // Brotli-decompress the compressed font data block.
    let compressed = p.read_bytes(total_compressed_size as usize)?;
    let decompressed = cv_compression::decode_brotli(compressed)
        .map_err(|e| Woff2Error::Brotli(format!("{e:?}")))?;

    // Slice out per-table raw bytes from the decompressed block.
    let mut data_cursor = 0usize;
    let mut raw_tables: Vec<&[u8]> = Vec::with_capacity(num_tables as usize);
    for t in &tables {
        let len = t.transform_length.unwrap_or(t.orig_length) as usize;
        let end = data_cursor
            .checked_add(len)
            .ok_or(Woff2Error::Truncated("table size overflow"))?;
        if end > decompressed.len() {
            return Err(Woff2Error::Truncated("table data past end of stream"));
        }
        raw_tables.push(&decompressed[data_cursor..end]);
        data_cursor = end;
    }

    // Find and untransform glyf if present.
    let mut untransformed_glyf: Option<Vec<u8>> = None;
    let mut regenerated_loca: Option<Vec<u8>> = None;
    let mut index_format: i16 = 0; // 0 = short loca, 1 = long loca
    for (i, t) in tables.iter().enumerate() {
        if t.tag == tag(b"glyf") && t.is_transformed() {
            let (glyf, loca, idx_fmt) = untransform_glyf(raw_tables[i])?;
            untransformed_glyf = Some(glyf);
            regenerated_loca = Some(loca);
            index_format = idx_fmt;
        }
    }

    // Build per-table final bytes + tags for SFNT packaging.
    // SFNT requires tables sorted by tag.
    struct OutTable {
        tag: u32,
        bytes: Vec<u8>,
    }
    let mut out: Vec<OutTable> = Vec::with_capacity(num_tables as usize);
    for (i, t) in tables.iter().enumerate() {
        if t.tag == tag(b"glyf") && t.is_transformed() {
            let g = untransformed_glyf
                .as_ref()
                .ok_or(Woff2Error::Reserialize("missing transformed glyf"))?
                .clone();
            out.push(OutTable {
                tag: t.tag,
                bytes: g,
            });
        } else if t.tag == tag(b"loca") && regenerated_loca.is_some() {
            // Spec: a transformed glyf implies the loca data is
            // empty in the compressed stream — we regenerate it
            // from glyf offsets above.
            out.push(OutTable {
                tag: t.tag,
                bytes: regenerated_loca.clone().unwrap(),
            });
        } else {
            out.push(OutTable {
                tag: t.tag,
                bytes: raw_tables[i].to_vec(),
            });
        }
    }
    out.sort_by_key(|t| t.tag);

    // If loca was regenerated and head.indexToLocFormat needs to
    // match `index_format`, patch the head table.
    if regenerated_loca.is_some() {
        if let Some(head) = out.iter_mut().find(|t| t.tag == tag(b"head")) {
            if head.bytes.len() >= 54 {
                head.bytes[50] = ((index_format >> 8) & 0xFF) as u8;
                head.bytes[51] = (index_format & 0xFF) as u8;
            }
        }
    }
    // Zero head.checkSumAdjustment before packing so the head table's
    // directory checksum is computed per spec (with the field == 0);
    // the real adjustment is written into the assembled font below.
    if let Some(head) = out.iter_mut().find(|t| t.tag == tag(b"head")) {
        if head.bytes.len() >= 12 {
            head.bytes[8..12].copy_from_slice(&[0, 0, 0, 0]);
        }
    }
    let head_index = out.iter().position(|t| t.tag == tag(b"head"));

    // Pack as SFNT.
    let n = out.len() as u16;
    let mut sfnt: Vec<u8> = Vec::new();
    // Offset table (12 bytes): sfnt version + numTables + searchRange + entrySelector + rangeShift.
    sfnt.extend_from_slice(&flavor.to_be_bytes());
    sfnt.extend_from_slice(&n.to_be_bytes());
    let entry_selector = (n as f32).log2().floor() as u16;
    let search_range = (1u16 << entry_selector) * 16;
    let range_shift = n * 16 - search_range;
    sfnt.extend_from_slice(&search_range.to_be_bytes());
    sfnt.extend_from_slice(&entry_selector.to_be_bytes());
    sfnt.extend_from_slice(&range_shift.to_be_bytes());

    // Reserve table directory (16 bytes per table).
    let dir_offset = sfnt.len();
    sfnt.resize(dir_offset + 16 * out.len(), 0);

    // Append each table aligned to 4 bytes, fill in the directory.
    for (i, t) in out.iter().enumerate() {
        // Align table start to 4 bytes.
        while sfnt.len() % 4 != 0 {
            sfnt.push(0);
        }
        let table_offset = sfnt.len() as u32;
        let table_len = t.bytes.len() as u32;
        let checksum = sfnt_checksum(&t.bytes);
        // Fill directory entry for this table.
        let de = dir_offset + i * 16;
        sfnt[de..de + 4].copy_from_slice(&t.tag.to_be_bytes());
        sfnt[de + 4..de + 8].copy_from_slice(&checksum.to_be_bytes());
        sfnt[de + 8..de + 12].copy_from_slice(&table_offset.to_be_bytes());
        sfnt[de + 12..de + 16].copy_from_slice(&table_len.to_be_bytes());
        sfnt.extend_from_slice(&t.bytes);
    }
    // Final pad to 4 bytes (GDI is happier).
    while sfnt.len() % 4 != 0 {
        sfnt.push(0);
    }

    // head.checkSumAdjustment = 0xB1B0AFBA - checksum(entire font with
    // the field treated as zero). The field was already zeroed in the
    // head table bytes above, so we can sum the assembled font as-is.
    if let Some(hi) = head_index {
        let de = dir_offset + hi * 16;
        let head_offset =
            u32::from_be_bytes([sfnt[de + 8], sfnt[de + 9], sfnt[de + 10], sfnt[de + 11]]) as usize;
        if head_offset + 12 <= sfnt.len() {
            let total = sfnt_checksum(&sfnt);
            let adj = 0xB1B0_AFBAu32.wrapping_sub(total);
            sfnt[head_offset + 8..head_offset + 12].copy_from_slice(&adj.to_be_bytes());
        }
    }

    Ok(sfnt)
}

// ===========================================================================
// Internals: header / dir parsing, cursor, table entry.
// ===========================================================================

struct Cursor<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, pos: 0 }
    }
    fn read_bytes(&mut self, n: usize) -> Result<&'a [u8], Woff2Error> {
        let end = self
            .pos
            .checked_add(n)
            .ok_or(Woff2Error::Truncated("overflow"))?;
        if end > self.bytes.len() {
            return Err(Woff2Error::Truncated("short read"));
        }
        let s = &self.bytes[self.pos..end];
        self.pos = end;
        Ok(s)
    }
    fn u8(&mut self) -> Result<u8, Woff2Error> {
        Ok(self.read_bytes(1)?[0])
    }
    fn u16(&mut self) -> Result<u16, Woff2Error> {
        let b = self.read_bytes(2)?;
        Ok(u16::from_be_bytes([b[0], b[1]]))
    }
    fn u32(&mut self) -> Result<u32, Woff2Error> {
        let b = self.read_bytes(4)?;
        Ok(u32::from_be_bytes([b[0], b[1], b[2], b[3]]))
    }
    fn uint_base_128(&mut self) -> Result<u32, Woff2Error> {
        // 1-5 bytes; high bit set means "another byte follows".
        // First byte 0x80 is reserved (no leading zero).
        let mut result: u32 = 0;
        for i in 0..5 {
            let b = self.u8()?;
            if i == 0 && b == 0x80 {
                return Err(Woff2Error::Truncated("UIntBase128 leading zero"));
            }
            // Overflow check before shifting in 7 more bits.
            if result & 0xFE00_0000 != 0 {
                return Err(Woff2Error::Truncated("UIntBase128 overflow"));
            }
            result = (result << 7) | (b as u32 & 0x7F);
            if b & 0x80 == 0 {
                return Ok(result);
            }
        }
        Err(Woff2Error::Truncated("UIntBase128 too long"))
    }
}

/// The 64 known table tags WOFF2 packs into 6-bit indices.
const KNOWN_TAGS: [&[u8; 4]; 63] = [
    b"cmap", b"head", b"hhea", b"hmtx", b"maxp", b"name", b"OS/2", b"post", b"cvt ", b"fpgm",
    b"glyf", b"loca", b"prep", b"CFF ", b"VORG", b"EBDT", b"EBLC", b"gasp", b"hdmx", b"kern",
    b"LTSH", b"PCLT", b"VDMX", b"vhea", b"vmtx", b"BASE", b"GDEF", b"GPOS", b"GSUB", b"EBSC",
    b"JSTF", b"MATH", b"CBDT", b"CBLC", b"COLR", b"CPAL", b"SVG ", b"sbix", b"acnt", b"avar",
    b"bdat", b"bloc", b"bsln", b"cvar", b"fdsc", b"feat", b"fmtx", b"fvar", b"gvar", b"hsty",
    b"just", b"lcar", b"mort", b"morx", b"opbd", b"prop", b"trak", b"Zapf", b"Silf", b"Glat",
    b"Gloc", b"Feat", b"Sill",
];

struct TableEntry {
    tag: u32,
    /// transform_version 0..3. For glyf/loca: 0 = transformed,
    /// 3 = not transformed. For other tables: 0 = not transformed,
    /// 3 = transformed.
    transform_version: u8,
    orig_length: u32,
    transform_length: Option<u32>,
}

impl TableEntry {
    fn read(c: &mut Cursor) -> Result<Self, Woff2Error> {
        let flags = c.u8()?;
        let tag_idx = flags & 0x3F;
        let transform_version = (flags >> 6) & 0x03;
        let tag = if tag_idx == 0x3F {
            c.u32()?
        } else if (tag_idx as usize) < KNOWN_TAGS.len() {
            u32::from_be_bytes(*KNOWN_TAGS[tag_idx as usize])
        } else {
            // Indices >= 63 are reserved per spec; treat as truncation.
            return Err(Woff2Error::Truncated("unknown known-tag index"));
        };
        let orig_length = c.uint_base_128()?;
        // Whether a transformLength is present:
        //   - glyf / loca: present iff transformed (transform_version == 0)
        //   - other tables: present iff transformed (transform_version != 0)
        // We accept the simpler rule "present iff transform applies",
        // which matches the spec for every realistic font.
        let is_glyf_or_loca = tag == tag_bytes(b"glyf") || tag == tag_bytes(b"loca");
        let transformed = if is_glyf_or_loca {
            transform_version == 0
        } else {
            transform_version != 0
        };
        let transform_length = if transformed {
            Some(c.uint_base_128()?)
        } else {
            None
        };
        Ok(Self {
            tag,
            transform_version,
            orig_length,
            transform_length,
        })
    }

    fn is_transformed(&self) -> bool {
        let is_glyf_or_loca = self.tag == tag_bytes(b"glyf") || self.tag == tag_bytes(b"loca");
        if is_glyf_or_loca {
            self.transform_version == 0
        } else {
            self.transform_version != 0
        }
    }
}

fn tag(s: &[u8; 4]) -> u32 {
    u32::from_be_bytes(*s)
}
const fn tag_bytes(s: &[u8; 4]) -> u32 {
    ((s[0] as u32) << 24) | ((s[1] as u32) << 16) | ((s[2] as u32) << 8) | s[3] as u32
}

/// SFNT table checksum: u32 big-endian word sum, with trailing
/// partial-word treated as if zero-padded.
fn sfnt_checksum(table: &[u8]) -> u32 {
    let mut sum: u32 = 0;
    let mut i = 0;
    while i + 4 <= table.len() {
        let w = u32::from_be_bytes(table[i..i + 4].try_into().unwrap());
        sum = sum.wrapping_add(w);
        i += 4;
    }
    if i < table.len() {
        let mut last = [0u8; 4];
        let rem = table.len() - i;
        last[..rem].copy_from_slice(&table[i..]);
        sum = sum.wrapping_add(u32::from_be_bytes(last));
    }
    sum
}

// ===========================================================================
// glyf transformation — see WOFF2 §5.1.
// ===========================================================================

/// 255UInt16 — variable-length unsigned integer encoding used for
/// per-contour point counts and per-glyph instruction lengths.
fn read_255_u16(c: &mut Cursor) -> Result<u16, Woff2Error> {
    const ONE_MORE_BYTE_CODE1: u8 = 255;
    const ONE_MORE_BYTE_CODE2: u8 = 254;
    const WORD_CODE: u8 = 253;
    const LOWEST_U_CODE: u16 = 253;
    let b0 = c.u8()?;
    match b0 {
        WORD_CODE => {
            let v = c.u16()?;
            Ok(v)
        }
        ONE_MORE_BYTE_CODE1 => {
            let b1 = c.u8()?;
            Ok(b1 as u16 + LOWEST_U_CODE)
        }
        ONE_MORE_BYTE_CODE2 => {
            let b1 = c.u8()?;
            Ok(b1 as u16 + LOWEST_U_CODE * 2)
        }
        _ => Ok(b0 as u16),
    }
}

/// Untransform a transformed glyf table per WOFF2 §5.1. Returns
/// `(glyf_bytes, loca_bytes, index_to_loc_format)`.
fn untransform_glyf(input: &[u8]) -> Result<(Vec<u8>, Vec<u8>, i16), Woff2Error> {
    let mut c = Cursor::new(input);
    // Header (32 bytes).
    let _reserved = c.u16()?;
    let option_flags = c.u16()?;
    let num_glyphs = c.u16()? as usize;
    let index_format = c.u16()? as i16; // 0 short, 1 long
    let n_contour_stream_size = c.u32()? as usize;
    let n_points_stream_size = c.u32()? as usize;
    let flag_stream_size = c.u32()? as usize;
    let glyph_stream_size = c.u32()? as usize;
    let composite_stream_size = c.u32()? as usize;
    let bbox_stream_size = c.u32()? as usize;
    let instruction_stream_size = c.u32()? as usize;

    // Optional "overlap simple bitmap" preceding the streams when
    // option_flags bit 0 is set (WOFF2 amendment for overlap-simple
    // flag preservation). Each glyph occupies one bit; align to byte.
    let overlap_bitmap_size = if option_flags & 1 != 0 {
        (num_glyphs + 7) / 8
    } else {
        0
    };
    let _overlap_bitmap = if overlap_bitmap_size > 0 {
        c.read_bytes(overlap_bitmap_size)?
    } else {
        &[]
    };

    // The stream order in the spec.
    let n_contour_stream = c.read_bytes(n_contour_stream_size)?;
    let n_points_stream = c.read_bytes(n_points_stream_size)?;
    let flag_stream = c.read_bytes(flag_stream_size)?;
    let glyph_stream = c.read_bytes(glyph_stream_size)?;
    let composite_stream = c.read_bytes(composite_stream_size)?;
    let bbox_stream = c.read_bytes(bbox_stream_size)?;
    let instruction_stream = c.read_bytes(instruction_stream_size)?;

    // The first (numGlyphs + 7) / 8 bytes of bboxStream are the
    // bbox bitmap; the rest are bbox values (8 bytes each set bit).
    let bbox_bitmap_size = (num_glyphs + 7) / 8;
    if bbox_bitmap_size > bbox_stream.len() {
        return Err(Woff2Error::BadGlyf("bbox bitmap larger than bbox stream"));
    }
    let bbox_bitmap = &bbox_stream[..bbox_bitmap_size];
    let bbox_values = &bbox_stream[bbox_bitmap_size..];

    // Per-glyph: produce reconstructed bytes; track offsets for loca.
    let mut glyf_out: Vec<u8> = Vec::new();
    let mut loca_offsets: Vec<u32> = Vec::with_capacity(num_glyphs + 1);
    loca_offsets.push(0);

    let mut nc_cur = Cursor::new(n_contour_stream);
    let mut np_cur = Cursor::new(n_points_stream);
    let mut flag_cur = Cursor::new(flag_stream);
    let mut glyph_cur = Cursor::new(glyph_stream);
    let mut comp_cur = Cursor::new(composite_stream);
    let mut bbox_cur = Cursor::new(bbox_values);
    let mut inst_cur = Cursor::new(instruction_stream);
    let mut bbox_idx = 0usize;

    for gi in 0..num_glyphs {
        let n_contours_raw = nc_cur.u16()? as i16;
        let glyph_start = glyf_out.len();
        let has_bbox = (bbox_bitmap[gi / 8] >> (7 - (gi % 8))) & 1 == 1;

        if n_contours_raw == 0 {
            // Empty glyph (no outline, no header).
            // (loca offset unchanged length 0.)
        } else if n_contours_raw < 0 {
            // Composite glyph — copy bytes from compositeStream
            // until end of components, then maybe instructions.
            // The WOFF2 composite stream is byte-identical to the
            // TrueType composite-glyph encoding.
            if !has_bbox {
                return Err(Woff2Error::BadGlyf("composite glyph missing bbox"));
            }
            // Glyph header: numberOfContours, xMin, yMin, xMax, yMax (10 bytes).
            glyf_out.extend_from_slice(&n_contours_raw.to_be_bytes());
            let bb = bbox_cur.read_bytes(8)?;
            glyf_out.extend_from_slice(bb);
            bbox_idx += 1;

            // Walk composite components until end-of-components flag.
            // Each component starts with a 2-byte flag word. We need
            // to know how many bytes the component consumes so we
            // can copy it through.
            let mut have_instructions = false;
            loop {
                let flag_word = comp_cur.u16()?;
                glyf_out.extend_from_slice(&flag_word.to_be_bytes());
                // glyphIndex (2 bytes)
                glyf_out.extend_from_slice(comp_cur.read_bytes(2)?);
                // ARG_1_AND_2_ARE_WORDS (bit 0)
                let words = flag_word & 0x0001 != 0;
                let arg_size = if words { 4 } else { 2 };
                glyf_out.extend_from_slice(comp_cur.read_bytes(arg_size)?);
                // Scale variants:
                if flag_word & 0x0008 != 0 {
                    glyf_out.extend_from_slice(comp_cur.read_bytes(2)?);
                } else if flag_word & 0x0040 != 0 {
                    glyf_out.extend_from_slice(comp_cur.read_bytes(4)?);
                } else if flag_word & 0x0080 != 0 {
                    glyf_out.extend_from_slice(comp_cur.read_bytes(8)?);
                }
                // WE_HAVE_INSTRUCTIONS
                if flag_word & 0x0100 != 0 {
                    have_instructions = true;
                }
                // MORE_COMPONENTS
                if flag_word & 0x0020 == 0 {
                    break;
                }
            }
            if have_instructions {
                let inst_len = read_255_u16(&mut glyph_cur)?;
                glyf_out.extend_from_slice(&inst_len.to_be_bytes());
                if inst_len > 0 {
                    let inst = inst_cur.read_bytes(inst_len as usize)?;
                    glyf_out.extend_from_slice(inst);
                }
            }
        } else {
            // Simple glyph.
            let n_contours = n_contours_raw as usize;
            // Per-contour point counts.
            let mut end_pts: Vec<u16> = Vec::with_capacity(n_contours);
            let mut running = 0u16;
            for _ in 0..n_contours {
                let np = read_255_u16(&mut np_cur)?;
                running = running
                    .checked_add(np)
                    .ok_or(Woff2Error::BadGlyf("point count overflow"))?;
                // endPtsOfContours is "last point index", so subtract 1.
                if running == 0 {
                    return Err(Woff2Error::BadGlyf("contour with zero points"));
                }
                end_pts.push(running - 1);
            }
            let total_pts = running as usize;
            // Read flags + (dx, dy) deltas per point.
            let mut flags: Vec<u8> = Vec::with_capacity(total_pts);
            let mut xs: Vec<i16> = Vec::with_capacity(total_pts);
            let mut ys: Vec<i16> = Vec::with_capacity(total_pts);
            for _ in 0..total_pts {
                let flag = flag_cur.u8()?;
                flags.push(flag);
                let (dx, dy) = decode_triplet(flag, &mut glyph_cur)?;
                xs.push(dx);
                ys.push(dy);
            }
            // Instruction length follows (255UInt16) for simple glyphs.
            let inst_len = read_255_u16(&mut glyph_cur)?;
            let inst = if inst_len > 0 {
                inst_cur.read_bytes(inst_len as usize)?
            } else {
                &[]
            };
            // Compute / read bbox.
            let (xmin, ymin, xmax, ymax) = if has_bbox {
                let bb = bbox_cur.read_bytes(8)?;
                bbox_idx += 1;
                (
                    i16::from_be_bytes([bb[0], bb[1]]),
                    i16::from_be_bytes([bb[2], bb[3]]),
                    i16::from_be_bytes([bb[4], bb[5]]),
                    i16::from_be_bytes([bb[6], bb[7]]),
                )
            } else {
                // Spec: derive from absolute coordinates.
                let mut ax: i32 = 0;
                let mut ay: i32 = 0;
                let mut min_x = i32::MAX;
                let mut min_y = i32::MAX;
                let mut max_x = i32::MIN;
                let mut max_y = i32::MIN;
                for i in 0..total_pts {
                    ax += xs[i] as i32;
                    ay += ys[i] as i32;
                    min_x = min_x.min(ax);
                    max_x = max_x.max(ax);
                    min_y = min_y.min(ay);
                    max_y = max_y.max(ay);
                }
                if total_pts == 0 {
                    (0, 0, 0, 0)
                } else {
                    (
                        min_x.clamp(i16::MIN as i32, i16::MAX as i32) as i16,
                        min_y.clamp(i16::MIN as i32, i16::MAX as i32) as i16,
                        max_x.clamp(i16::MIN as i32, i16::MAX as i32) as i16,
                        max_y.clamp(i16::MIN as i32, i16::MAX as i32) as i16,
                    )
                }
            };
            // Glyph header.
            glyf_out.extend_from_slice(&(n_contours as i16).to_be_bytes());
            glyf_out.extend_from_slice(&xmin.to_be_bytes());
            glyf_out.extend_from_slice(&ymin.to_be_bytes());
            glyf_out.extend_from_slice(&xmax.to_be_bytes());
            glyf_out.extend_from_slice(&ymax.to_be_bytes());
            // endPtsOfContours
            for ep in &end_pts {
                glyf_out.extend_from_slice(&ep.to_be_bytes());
            }
            // instructionLength + instructions
            glyf_out.extend_from_slice(&inst_len.to_be_bytes());
            glyf_out.extend_from_slice(inst);
            // Reconstruct simple-glyph flag stream. We currently
            // emit one byte per point with the on-curve bit copied
            // from `flag` bit 7 and use the long-coordinate form
            // for the x/y delta encoding (no repeat / dual-byte
            // compression — readable, correct, slightly bigger
            // than optimal).
            for &fl in &flags {
                // google/woff2 `woff2_dec.cc` TripletDecode:
                //   `bool on_curve = !(flag >> 7);`
                // Bit 7 SET = OFF-curve, CLEAR = ON-curve. We previously
                // inverted this (treated bit-7-set as on-curve), which
                // swapped every curve anchor with its control point and
                // mangled the outline of every curved glyph.
                let on_curve = (fl >> 7) & 1 == 0;
                let mut tt_flag: u8 = 0;
                if on_curve {
                    tt_flag |= 0x01; // ON_CURVE_POINT
                }
                glyf_out.push(tt_flag);
            }
            // x deltas (i16 each).
            for &dx in &xs {
                glyf_out.extend_from_slice(&dx.to_be_bytes());
            }
            // y deltas (i16 each).
            for &dy in &ys {
                glyf_out.extend_from_slice(&dy.to_be_bytes());
            }
        }
        // Pad each glyph to even length (TrueType alignment).
        if glyf_out.len() % 2 != 0 {
            glyf_out.push(0);
        }
        loca_offsets.push(
            glyf_out.len() as u32 - glyph_start as u32 + loca_offsets.last().copied().unwrap_or(0),
        );
        // Wait — loca offsets are cumulative byte positions in glyf,
        // not per-glyph deltas. Recompute properly: the *next* slot
        // should be the current `glyf_out.len()`. Fix:
        *loca_offsets.last_mut().unwrap() = glyf_out.len() as u32;
    }
    let _ = bbox_idx;

    // Build loca table — short or long depending on indexFormat.
    let loca_bytes = if index_format == 0 {
        // Short loca: each entry is offset / 2 as u16. Requires
        // every glyph offset to be even (we padded — good).
        let mut out = Vec::with_capacity((loca_offsets.len()) * 2);
        for &off in &loca_offsets {
            let v = (off / 2) as u16;
            out.extend_from_slice(&v.to_be_bytes());
        }
        out
    } else {
        let mut out = Vec::with_capacity(loca_offsets.len() * 4);
        for &off in &loca_offsets {
            out.extend_from_slice(&off.to_be_bytes());
        }
        out
    };

    Ok((glyf_out, loca_bytes, index_format))
}

/// One entry in the WOFF2 §5.1 triplet table. `byte_count` is the
/// *total* bytes per encoded point (flag + data); subtract 1 to get
/// the extra bytes to read after the flag.
#[derive(Copy, Clone)]
struct Triplet {
    byte_count: u8,
    x_bits: u8,
    y_bits: u8,
    delta_x: u16,
    delta_y: u16,
    x_sign: i8,
    y_sign: i8,
}

/// Canonical 128-entry triplet decoding table per W3C WOFF2 §5.1
/// table "T_glyph". The earlier pattern-derived version had groups
/// A and B swapped (y-only vs x-only) and incorrect byte counts for
/// groups D-F, which made every simple glyph come out distorted.
/// This table is generated below by `build_triplet_table()` at
/// const-evaluation time so every value is auditable side by side
/// with the spec.
static TRIPLET_TABLE: [Triplet; 128] = build_triplet_table();

const fn build_triplet_table() -> [Triplet; 128] {
    let mut t = [Triplet {
        byte_count: 0,
        x_bits: 0,
        y_bits: 0,
        delta_x: 0,
        delta_y: 0,
        x_sign: 0,
        y_sign: 0,
    }; 128];
    // Canonical WOFF2 §5.2 "Triplet Encoding" table. `byte_count`
    // INCLUDES the flag byte, so data bytes = byte_count - 1. The
    // sign code `s` packs xSign in bit 0 and ySign in bit 1, with
    // 0 = negative and 1 = positive (so s: 0=(-,-) 1=(+,-) 2=(-,+)
    // 3=(+,+)). For the packed groups the iteration is dy outer,
    // dx middle, sign inner (verified: index 35 = dx 49, dy 1).

    // Group A (0..9): 1 data byte, Y only (xBits=0, yBits=8).
    // delta_y = (n>>1)*256, y_sign alternates -/+.
    let mut n = 0;
    while n < 10 {
        t[n] = Triplet {
            byte_count: 2,
            x_bits: 0,
            y_bits: 8,
            delta_x: 0,
            delta_y: ((n >> 1) * 256) as u16,
            x_sign: 0,
            y_sign: if n & 1 == 1 { 1 } else { -1 },
        };
        n += 1;
    }
    // Group B (10..19): 1 data byte, X only (xBits=8, yBits=0).
    // delta_x = (n>>1)*256, x_sign alternates -/+.
    let mut n = 0;
    while n < 10 {
        t[10 + n] = Triplet {
            byte_count: 2,
            x_bits: 8,
            y_bits: 0,
            delta_x: ((n >> 1) * 256) as u16,
            delta_y: 0,
            x_sign: if n & 1 == 1 { 1 } else { -1 },
            y_sign: 0,
        };
        n += 1;
    }
    // Group C (20..83): 1 data byte, 4+4 bits. delta in {1,17,33,49}.
    // 64 entries. Per WOFF2 §5.2 / fonttools `_decodeTriplets`, with
    // `b0 = flag-20`: dx_base = 1 + (b0 & 0x30), dy_base = 1 + ((b0 &
    // 0x0c) << 2). With the loop order (outer `dy_i`, middle `dx_i`,
    // inner sign), b0 = dy_i*16 + dx_i*4 + s, so b0&0x30 isolates the
    // OUTER index → delta_x indexes by `dy_i`, and (b0&0x0c)<<2 isolates
    // the MIDDLE → delta_y indexes by `dx_i`. (They are CROSSED — a
    // straight `delta_x=d4[dx_i]` mangles ~half of all glyph points.)
    let d4 = [1u16, 17, 33, 49];
    let mut idx = 20;
    let mut dy_i = 0;
    while dy_i < 4 {
        let mut dx_i = 0;
        while dx_i < 4 {
            let mut s = 0;
            while s < 4 {
                t[idx] = Triplet {
                    byte_count: 2,
                    x_bits: 4,
                    y_bits: 4,
                    delta_x: d4[dy_i],
                    delta_y: d4[dx_i],
                    x_sign: if s & 1 == 1 { 1 } else { -1 },
                    y_sign: if s & 2 == 2 { 1 } else { -1 },
                };
                idx += 1;
                s += 1;
            }
            dx_i += 1;
        }
        dy_i += 1;
    }
    // Group D (84..119): 2 data bytes, 8+8 bits. delta in {1,257,513}.
    // 36 entries. Same crossed indexing as Group C: with `b0 = flag-84`,
    // dx_base = 1 + ((b0/12)<<8) → OUTER `dy_i`; dy_base = 1 +
    // (((b0%12)>>2)<<8) → MIDDLE `dx_i`.
    let d8 = [1u16, 257, 513];
    let mut dy_i = 0;
    while dy_i < 3 {
        let mut dx_i = 0;
        while dx_i < 3 {
            let mut s = 0;
            while s < 4 {
                t[idx] = Triplet {
                    byte_count: 3,
                    x_bits: 8,
                    y_bits: 8,
                    delta_x: d8[dy_i],
                    delta_y: d8[dx_i],
                    x_sign: if s & 1 == 1 { 1 } else { -1 },
                    y_sign: if s & 2 == 2 { 1 } else { -1 },
                };
                idx += 1;
                s += 1;
            }
            dx_i += 1;
        }
        dy_i += 1;
    }
    // Group E (120..123): 3 data bytes, 12+12 bits, no delta.
    let mut s = 0;
    while s < 4 {
        t[120 + s] = Triplet {
            byte_count: 4,
            x_bits: 12,
            y_bits: 12,
            delta_x: 0,
            delta_y: 0,
            x_sign: if s & 1 == 1 { 1 } else { -1 },
            y_sign: if s & 2 == 2 { 1 } else { -1 },
        };
        s += 1;
    }
    // Group F (124..127): 4 data bytes, 16+16 bits, no delta.
    let mut s = 0;
    while s < 4 {
        t[124 + s] = Triplet {
            byte_count: 5,
            x_bits: 16,
            y_bits: 16,
            delta_x: 0,
            delta_y: 0,
            x_sign: if s & 1 == 1 { 1 } else { -1 },
            y_sign: if s & 2 == 2 { 1 } else { -1 },
        };
        s += 1;
    }
    t
}

/// WOFF2 triplet encoding for simple-glyph (dx, dy) deltas, per
/// the canonical `TRIPLET_TABLE` above.
fn decode_triplet(flag: u8, c: &mut Cursor) -> Result<(i16, i16), Woff2Error> {
    let entry = TRIPLET_TABLE[(flag & 0x7F) as usize];
    // Read (byte_count - 1) extra data bytes into a shift register.
    let extra = entry.byte_count.saturating_sub(1);
    let mut buf: u32 = 0;
    let mut i = 0;
    while i < extra {
        buf = (buf << 8) | (c.u8()? as u32);
        i += 1;
    }
    let x_mask = if entry.x_bits == 0 {
        0
    } else {
        (1u32 << entry.x_bits) - 1
    };
    let y_mask = if entry.y_bits == 0 {
        0
    } else {
        (1u32 << entry.y_bits) - 1
    };
    let x_val = ((buf >> entry.y_bits) & x_mask) as i32 + entry.delta_x as i32;
    let y_val = (buf & y_mask) as i32 + entry.delta_y as i32;
    let dx = (x_val * entry.x_sign as i32).clamp(i16::MIN as i32, i16::MAX as i32) as i16;
    let dy = (y_val * entry.y_sign as i32).clamp(i16::MIN as i32, i16::MAX as i32) as i16;
    Ok((dx, dy))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn signature_check_rejects_non_woff2() {
        let bad = b"OTTOxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx";
        let r = decode_woff2(bad);
        assert!(matches!(r, Err(Woff2Error::BadSignature)));
    }

    #[test]
    fn header_truncation_rejected() {
        let r = decode_woff2(b"wOF2tooshort");
        assert!(matches!(r, Err(Woff2Error::Truncated(_))));
    }

    #[test]
    fn uintbase128_reads_simple_values() {
        let mut c = Cursor::new(&[0x05]);
        assert_eq!(c.uint_base_128().unwrap(), 5);
        // 128 = 0x81 0x00
        let mut c = Cursor::new(&[0x81, 0x00]);
        assert_eq!(c.uint_base_128().unwrap(), 128);
        // 0x80 leading is illegal.
        let mut c = Cursor::new(&[0x80, 0x00]);
        assert!(c.uint_base_128().is_err());
    }

    #[test]
    fn read_255_u16_simple_byte() {
        let mut c = Cursor::new(&[0x7B]);
        assert_eq!(read_255_u16(&mut c).unwrap(), 123);
    }

    #[test]
    fn read_255_u16_word_code() {
        let mut c = Cursor::new(&[253, 0x01, 0x00]);
        assert_eq!(read_255_u16(&mut c).unwrap(), 256);
    }

    #[test]
    fn triplet_decode_matches_woff2_spec_vectors() {
        // Ground-truth (flag, data-bytes) → (dx, dy) computed by hand from
        // the WOFF2 §5.2 algorithm (matches fonttools `_decodeTriplets`).
        // The Group-C cross-indexing bug made flag 35 decode as (49,1)
        // instead of the correct (1,49) — mangling ~half of all points.
        let cases: &[(u8, &[u8], i16, i16)] = &[
            // Group A: dy sign is NEGATIVE when flag&1 is clear (the
            // `withSign` convention), positive when set.
            (0, &[5], 0, -5),
            (1, &[5], 0, 5),
            // Group C: flag 20 (b0=0) → dx=-(1+hi), dy=-(1+lo).
            (20, &[0x00], -1, -1),
            // Group C: flag 35 (b0=15) → dx=+(1+hi), dy=+(49+lo). MUST be
            // (1,49) not (49,1).
            (35, &[0x00], 1, 49),
            (35, &[0x12], 1 + 1, 49 + 2), // hi=1, lo=2
            // Group D: flag 84 (b0=0) → dx=-(1+b1), dy=-(1+b2).
            (84, &[0x00, 0x00], -1, -1),
        ];
        for &(flag, bytes, want_dx, want_dy) in cases {
            let mut c = Cursor::new(bytes);
            let (dx, dy) = decode_triplet(flag, &mut c).unwrap();
            assert_eq!((dx, dy), (want_dx, want_dy), "flag {flag} bytes {bytes:?}");
        }
    }
}
