//! MSIX installer — AppxManifest scaffolding + package layout.

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PackageIdentity {
    pub name: String,
    pub publisher: String,
    pub version: String,
}

/// Emit the AppxManifest.xml body for a desktop app entry point.
pub fn render_appx_manifest(
    id: &PackageIdentity,
    app_id: &str,
    exe: &str,
    display_name: &str,
) -> String {
    format!(
        r#"<?xml version="1.0" encoding="UTF-8"?>
<Package xmlns="http://schemas.microsoft.com/appx/manifest/foundation/windows10"
         xmlns:uap="http://schemas.microsoft.com/appx/manifest/uap/windows10"
         xmlns:rescap="http://schemas.microsoft.com/appx/manifest/foundation/windows10/restrictedcapabilities">
  <Identity Name="{name}" Publisher="{publisher}" Version="{version}" ProcessorArchitecture="x64"/>
  <Properties>
    <DisplayName>{display}</DisplayName>
    <PublisherDisplayName>{publisher}</PublisherDisplayName>
    <Logo>Assets/StoreLogo.png</Logo>
  </Properties>
  <Resources>
    <Resource Language="en-us"/>
  </Resources>
  <Dependencies>
    <TargetDeviceFamily Name="Windows.Desktop" MinVersion="10.0.19041.0" MaxVersionTested="10.0.22631.0"/>
  </Dependencies>
  <Applications>
    <Application Id="{app_id}" Executable="{exe}" EntryPoint="Windows.FullTrustApplication">
      <uap:VisualElements DisplayName="{display}" Description="{display}"
                          Square150x150Logo="Assets/Square150x150Logo.png"
                          Square44x44Logo="Assets/Square44x44Logo.png"
                          BackgroundColor="transparent"/>
    </Application>
  </Applications>
  <Capabilities>
    <rescap:Capability Name="runFullTrust"/>
  </Capabilities>
</Package>
"#,
        name = id.name,
        publisher = id.publisher,
        version = id.version,
        display = display_name,
        app_id = app_id,
        exe = exe,
    )
}

/// Build the AppxBlockMap.xml content per the OPC packaging spec.
/// Each file contributes a `<File>` with a `<Block>` whose Hash is the
/// base64 SHA-256 of its bytes (chunked at 64KiB blocks for files
/// larger than that limit).
pub fn render_block_map(files: &[(String, &[u8])]) -> String {
    let mut s = String::from(
        r#"<?xml version="1.0" encoding="UTF-8"?>
<BlockMap xmlns="http://schemas.microsoft.com/appx/2010/blockmap" HashMethod="http://www.w3.org/2001/04/xmlenc#sha256">
"#,
    );
    for (path, bytes) in files {
        s.push_str(&format!(
            "  <File Name=\"{}\" Size=\"{}\" LfhSize=\"30\">\n",
            xml_escape(path),
            bytes.len()
        ));
        const CHUNK: usize = 64 * 1024;
        let mut offset = 0;
        while offset < bytes.len() {
            let end = (offset + CHUNK).min(bytes.len());
            let mut h = cv_crypto::sha256::Sha256::new();
            h.update(&bytes[offset..end]);
            let digest = h.finalize();
            let b64 = base64_encode(&digest);
            s.push_str(&format!("    <Block Hash=\"{}\"/>\n", b64));
            offset = end;
        }
        s.push_str("  </File>\n");
    }
    s.push_str("</BlockMap>\n");
    s
}

fn xml_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

fn base64_encode(input: &[u8]) -> String {
    const ALPHA: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity((input.len() + 2) / 3 * 4);
    let mut i = 0;
    while i + 3 <= input.len() {
        let n = ((input[i] as u32) << 16) | ((input[i + 1] as u32) << 8) | input[i + 2] as u32;
        out.push(ALPHA[((n >> 18) & 0x3F) as usize] as char);
        out.push(ALPHA[((n >> 12) & 0x3F) as usize] as char);
        out.push(ALPHA[((n >> 6) & 0x3F) as usize] as char);
        out.push(ALPHA[(n & 0x3F) as usize] as char);
        i += 3;
    }
    let rem = input.len() - i;
    if rem == 1 {
        let n = (input[i] as u32) << 16;
        out.push(ALPHA[((n >> 18) & 0x3F) as usize] as char);
        out.push(ALPHA[((n >> 12) & 0x3F) as usize] as char);
        out.push_str("==");
    } else if rem == 2 {
        let n = ((input[i] as u32) << 16) | ((input[i + 1] as u32) << 8);
        out.push(ALPHA[((n >> 18) & 0x3F) as usize] as char);
        out.push(ALPHA[((n >> 12) & 0x3F) as usize] as char);
        out.push(ALPHA[((n >> 6) & 0x3F) as usize] as char);
        out.push('=');
    }
    out
}

/// Write the manifest + block map + payload files into a directory
/// the user can hand to `makeappx pack`. The PKCS#7 signing pass is
/// the next step (we model the layout, signing lands in publisher
/// tooling).
pub fn write_package_layout(
    dir: &std::path::Path,
    id: &PackageIdentity,
    payload: &[(&str, &[u8])],
) -> std::io::Result<()> {
    std::fs::create_dir_all(dir)?;
    let manifest = render_appx_manifest(id, "Conclave", "conclave.exe", "Conclave");
    std::fs::write(dir.join("AppxManifest.xml"), manifest)?;
    let block_input: Vec<(String, &[u8])> = payload
        .iter()
        .map(|(n, b)| (String::from(*n), *b))
        .collect();
    let blockmap = render_block_map(&block_input);
    std::fs::write(dir.join("AppxBlockMap.xml"), blockmap)?;
    for (name, bytes) in payload {
        let p = dir.join(name);
        if let Some(parent) = p.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(p, bytes)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn manifest_includes_identity_block() {
        let id = PackageIdentity {
            name: "Conclave".into(),
            publisher: "CN=Conclave Authors".into(),
            version: "1.2.3.4".into(),
        };
        let xml = render_appx_manifest(&id, "Conclave", "conclave.exe", "Conclave");
        assert!(xml.contains(r#"Name="Conclave""#));
        assert!(xml.contains(r#"Publisher="CN=Conclave Authors""#));
        assert!(xml.contains(r#"Version="1.2.3.4""#));
        assert!(xml.contains("conclave.exe"));
    }

    #[test]
    fn manifest_requests_full_trust_capability() {
        let id = PackageIdentity {
            name: "X".into(),
            publisher: "CN=X".into(),
            version: "1.0.0.0".into(),
        };
        let xml = render_appx_manifest(&id, "Y", "y.exe", "Y");
        assert!(xml.contains("runFullTrust"));
    }
}
