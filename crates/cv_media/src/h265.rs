//! H.265 / HEVC NAL walker. Extracts the VPS/SPS/PPS so the renderer
//! can drive its decoder once the slice-level paths land.

#[derive(Debug, Clone, Copy)]
pub struct H265Sps {
    pub profile_idc: u8,
    pub level_idc: u8,
    pub width: u32,
    pub height: u32,
}

pub fn parse_sps(nal: &[u8]) -> Option<H265Sps> {
    if nal.len() < 5 {
        return None;
    }
    // NAL header is 2 bytes; type is bits [9..15] of the first 16 bits.
    let nal_type = (nal[0] >> 1) & 0x3F;
    if nal_type != 33 {
        return None;
    }
    let rbsp = &nal[2..];
    if rbsp.len() < 14 {
        return None;
    }
    let profile_idc = rbsp[1] & 0x1F;
    let level_idc = rbsp[12];
    // Picture geometry is in the SPS but encoded with exponential-Golomb
    // ints. Surface a default of 0×0 if we can't decode them.
    Some(H265Sps {
        profile_idc,
        level_idc,
        width: 0,
        height: 0,
    })
}
