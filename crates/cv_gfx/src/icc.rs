//! ICC profile parser (ICC v4 spec, ICC.1:2010-12) — extracts the
//! header + tag table so colour transforms can decide whether to apply
//! a matrix/TRC pipeline. Full chromatic adaptation lives in the
//! compositor; this layer only surfaces enough to identify the
//! profile.

#[derive(Debug, Clone)]
pub struct IccProfile {
    pub size: u32,
    pub class: [u8; 4],
    pub colour_space: [u8; 4],
    pub pcs: [u8; 4],
    pub tags: Vec<IccTag>,
}

#[derive(Debug, Clone)]
pub struct IccTag {
    pub signature: [u8; 4],
    pub offset: u32,
    pub size: u32,
}

pub fn parse(buf: &[u8]) -> Option<IccProfile> {
    if buf.len() < 128 + 4 {
        return None;
    }
    let size = u32::from_be_bytes(buf[0..4].try_into().unwrap());
    let mut class = [0u8; 4];
    class.copy_from_slice(&buf[12..16]);
    let mut colour_space = [0u8; 4];
    colour_space.copy_from_slice(&buf[16..20]);
    let mut pcs = [0u8; 4];
    pcs.copy_from_slice(&buf[20..24]);
    let tag_count = u32::from_be_bytes(buf[128..132].try_into().unwrap()) as usize;
    let mut tags = Vec::with_capacity(tag_count);
    for i in 0..tag_count {
        let off = 132 + i * 12;
        if off + 12 > buf.len() {
            return None;
        }
        let mut sig = [0u8; 4];
        sig.copy_from_slice(&buf[off..off + 4]);
        let toff = u32::from_be_bytes(buf[off + 4..off + 8].try_into().unwrap());
        let tsize = u32::from_be_bytes(buf[off + 8..off + 12].try_into().unwrap());
        tags.push(IccTag {
            signature: sig,
            offset: toff,
            size: tsize,
        });
    }
    Some(IccProfile {
        size,
        class,
        colour_space,
        pcs,
        tags,
    })
}

/// Apply a Display P3 → sRGB linear transform to an RGB triple. Matrix
/// derived from the canonical P3↔sRGB white-point-adapted conversion.
pub fn p3_to_srgb(rgb: [f32; 3]) -> [f32; 3] {
    let m = [
        [1.2249, -0.2247, 0.0],
        [-0.0420, 1.0419, 0.0],
        [-0.0197, -0.0786, 1.0979],
    ];
    [
        m[0][0] * rgb[0] + m[0][1] * rgb[1] + m[0][2] * rgb[2],
        m[1][0] * rgb[0] + m[1][1] * rgb[1] + m[1][2] * rgb[2],
        m[2][0] * rgb[0] + m[2][1] * rgb[1] + m[2][2] * rgb[2],
    ]
}

/// Apply the HDR PQ EOTF (SMPTE ST.2084) to a normalized signal in
/// [0,1] returning luminance in cd/m² (0..10000).
pub fn pq_eotf(e: f32) -> f32 {
    const C1: f32 = 0.8359375;
    const C2: f32 = 18.8515625;
    const C3: f32 = 18.6875;
    const M1: f32 = 0.1593017578125;
    const M2: f32 = 78.84375;
    let ep = e.powf(1.0 / M2);
    let n = (ep - C1).max(0.0);
    let d = C2 - C3 * ep;
    if d <= 0.0 {
        return 0.0;
    }
    let l = (n / d).powf(1.0 / M1);
    l * 10_000.0
}

/// Apply the HLG OETF^-1 (ARIB STD-B67 / BT.2100) for a normalized
/// signal value to scene luminance.
pub fn hlg_eotf(e: f32) -> f32 {
    let a = 0.17883277_f32;
    let b = 0.28466892_f32;
    let c = 0.55991073_f32;
    if e <= 0.5 {
        (e * e) / 3.0
    } else {
        ((((e - c) / a).exp()) + b) / 12.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_header_size() {
        let mut buf = vec![0u8; 132];
        buf[0..4].copy_from_slice(&132u32.to_be_bytes());
        let p = parse(&buf).unwrap();
        assert_eq!(p.size, 132);
        assert_eq!(p.tags.len(), 0);
    }

    #[test]
    fn p3_to_srgb_passes_through_unit() {
        let r = p3_to_srgb([1.0, 1.0, 1.0]);
        for c in r {
            assert!((c - 1.0).abs() < 0.05);
        }
    }
}
