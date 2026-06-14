use crate::png::ImageError;

#[derive(Debug, Clone, Copy)]
pub struct AvifInfo {
    pub width: u32,
    pub height: u32,
}

pub fn parse_avif_info(input: &[u8]) -> Result<AvifInfo, ImageError> {
    if !looks_like_avif(input) {
        return Err(ImageError::BadSignature);
    }
    let mut dims = None;
    walk_boxes(input, false, &mut dims)?;
    dims.ok_or(ImageError::Malformed("AVIF: missing ispe"))
}

fn looks_like_avif(input: &[u8]) -> bool {
    if input.len() < 16 {
        return false;
    }
    let Some((kind, payload, _)) = next_box(input, 0) else {
        return false;
    };
    if kind != *b"ftyp" || payload.len() < 8 {
        return false;
    }
    payload[0..4] == *b"avif"
        || payload[0..4] == *b"avis"
        || payload[8..]
            .chunks_exact(4)
            .any(|b| b == b"avif" || b == b"avis")
}

fn walk_boxes(
    input: &[u8],
    skip_fullbox: bool,
    dims: &mut Option<AvifInfo>,
) -> Result<(), ImageError> {
    let mut offset = 0usize;
    let body = if skip_fullbox {
        input.get(4..).ok_or(ImageError::Truncated)?
    } else {
        input
    };
    while let Some((kind, payload, next)) = next_box(body, offset) {
        match &kind {
            b"meta" => walk_boxes(payload, true, dims)?,
            b"iprp" | b"ipco" => walk_boxes(payload, false, dims)?,
            b"ispe" => {
                if payload.len() < 12 {
                    return Err(ImageError::Truncated);
                }
                let width = u32::from_be_bytes([payload[4], payload[5], payload[6], payload[7]]);
                let height = u32::from_be_bytes([payload[8], payload[9], payload[10], payload[11]]);
                *dims = Some(AvifInfo { width, height });
            }
            _ => {}
        }
        offset = next;
    }
    Ok(())
}

fn next_box(input: &[u8], offset: usize) -> Option<([u8; 4], &[u8], usize)> {
    let header = input.get(offset..offset + 8)?;
    let size32 = u32::from_be_bytes([header[0], header[1], header[2], header[3]]) as usize;
    let kind = [header[4], header[5], header[6], header[7]];
    let (header_len, size) = if size32 == 1 {
        let ext = input.get(offset + 8..offset + 16)?;
        let size64 = u64::from_be_bytes([
            ext[0], ext[1], ext[2], ext[3], ext[4], ext[5], ext[6], ext[7],
        ]) as usize;
        (16usize, size64)
    } else {
        (8usize, size32)
    };
    if size < header_len {
        return None;
    }
    let end = if size == 0 {
        input.len()
    } else {
        offset.checked_add(size)?
    };
    let payload = input.get(offset + header_len..end)?;
    Some((kind, payload, end))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn boxed(kind: &[u8; 4], payload: &[u8]) -> Vec<u8> {
        let mut out = Vec::with_capacity(payload.len() + 8);
        out.extend_from_slice(&((payload.len() + 8) as u32).to_be_bytes());
        out.extend_from_slice(kind);
        out.extend_from_slice(payload);
        out
    }

    #[test]
    fn parses_avif_dimensions_from_ispe() {
        let ftyp = boxed(b"ftyp", b"avif\0\0\0\0avifmif1");
        let ispe = boxed(b"ispe", &[0, 0, 0, 0, 0, 0, 2, 128, 0, 0, 1, 224]);
        let ipco = boxed(b"ipco", &ispe);
        let iprp = boxed(b"iprp", &ipco);
        let mut meta_payload = vec![0, 0, 0, 0];
        meta_payload.extend_from_slice(&iprp);
        let meta = boxed(b"meta", &meta_payload);
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&ftyp);
        bytes.extend_from_slice(&meta);

        let info = parse_avif_info(&bytes).unwrap();
        assert_eq!(info.width, 640);
        assert_eq!(info.height, 480);
    }

    #[test]
    fn rejects_non_avif() {
        assert!(matches!(
            parse_avif_info(b"not avif"),
            Err(ImageError::BadSignature)
        ));
    }
}
