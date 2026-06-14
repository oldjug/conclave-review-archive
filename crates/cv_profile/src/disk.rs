//! Persistent on-disk profile root. Stores cookies, history,
//! bookmarks, localStorage, IndexedDB blobs under
//! `%LOCALAPPDATA%\Conclave\<profile>`. Incognito mode flips
//! `incognito = true` and short-circuits every disk write.

use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone)]
pub struct ProfileRoot {
    pub root: PathBuf,
    pub incognito: bool,
}

impl ProfileRoot {
    pub fn default_root() -> PathBuf {
        if let Ok(local) = std::env::var("LOCALAPPDATA") {
            return PathBuf::from(local).join("Conclave").join("Default");
        }
        if let Ok(home) = std::env::var("USERPROFILE") {
            return PathBuf::from(home)
                .join("AppData")
                .join("Local")
                .join("Conclave")
                .join("Default");
        }
        PathBuf::from(".toasty-blum")
    }

    pub fn open() -> Self {
        let root = Self::default_root();
        let _ = fs::create_dir_all(&root);
        let _ = fs::create_dir_all(root.join("cookies"));
        let _ = fs::create_dir_all(root.join("history"));
        let _ = fs::create_dir_all(root.join("bookmarks"));
        let _ = fs::create_dir_all(root.join("localStorage"));
        let _ = fs::create_dir_all(root.join("indexeddb"));
        let _ = fs::create_dir_all(root.join("downloads"));
        Self {
            root,
            incognito: false,
        }
    }

    pub fn incognito() -> Self {
        Self {
            root: std::env::temp_dir().join("tb_incognito"),
            incognito: true,
        }
    }

    pub fn cookies_path(&self) -> PathBuf {
        self.root.join("cookies").join("jar.txt")
    }

    pub fn history_path(&self) -> PathBuf {
        self.root.join("history").join("urls.txt")
    }

    pub fn bookmarks_path(&self) -> PathBuf {
        self.root.join("bookmarks").join("marks.json")
    }

    pub fn local_storage_path(&self, origin: &str) -> PathBuf {
        let safe: String = origin
            .chars()
            .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
            .collect();
        self.root.join("localStorage").join(format!("{safe}.json"))
    }

    pub fn indexed_db_path(&self, origin: &str, db: &str) -> PathBuf {
        let safe_o: String = origin
            .chars()
            .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
            .collect();
        let safe_db: String = db
            .chars()
            .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
            .collect();
        self.root
            .join("indexeddb")
            .join(format!("{safe_o}_{safe_db}.bin"))
    }

    pub fn write_string(&self, path: &Path, content: &str) -> std::io::Result<()> {
        if self.incognito {
            return Ok(());
        }
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(path, content)
    }

    pub fn read_string(&self, path: &Path) -> Option<String> {
        if self.incognito {
            return None;
        }
        fs::read_to_string(path).ok()
    }

    pub fn append_line(&self, path: &Path, line: &str) -> std::io::Result<()> {
        if self.incognito {
            return Ok(());
        }
        use std::io::Write;
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let mut f = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)?;
        writeln!(f, "{line}")
    }
}

/// Local-only profile sync — export the profile root to a JSON blob
/// the user can hand-carry to another install.
pub fn export_profile_json(profile: &ProfileRoot) -> String {
    use std::fmt::Write;
    let mut s = String::from("{\"v\":1,\n");
    if let Some(h) = profile.read_string(&profile.history_path()) {
        let _ = write!(s, "\"history\":{},\n", serialize_str(&h));
    }
    if let Some(b) = profile.read_string(&profile.bookmarks_path()) {
        let _ = write!(s, "\"bookmarks\":{},\n", serialize_str(&b));
    }
    if let Some(c) = profile.read_string(&profile.cookies_path()) {
        let _ = write!(s, "\"cookies\":{}\n", serialize_str(&c));
    }
    s.push('}');
    s
}

pub fn import_profile_json(profile: &ProfileRoot, json: &str) -> Result<(), String> {
    // Very thin: extract our three sections and write them back.
    let pairs = json_top_strings(json);
    for (k, v) in pairs {
        let target = match k.as_str() {
            "history" => profile.history_path(),
            "bookmarks" => profile.bookmarks_path(),
            "cookies" => profile.cookies_path(),
            _ => continue,
        };
        profile
            .write_string(&target, &v)
            .map_err(|e| e.to_string())?;
    }
    Ok(())
}

fn serialize_str(s: &str) -> String {
    let mut out = String::from("\"");
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

fn json_top_strings(json: &str) -> Vec<(String, String)> {
    let mut out = Vec::new();
    let bytes = json.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'"' {
            let mut j = i + 1;
            while j < bytes.len() && bytes[j] != b'"' {
                if bytes[j] == b'\\' {
                    j += 1;
                }
                j += 1;
            }
            if j >= bytes.len() {
                break;
            }
            let key = String::from_utf8_lossy(&bytes[i + 1..j]).into_owned();
            let mut k = j + 1;
            while k < bytes.len() && bytes[k] != b':' {
                k += 1;
            }
            k += 1;
            while k < bytes.len() && bytes[k].is_ascii_whitespace() {
                k += 1;
            }
            if k < bytes.len() && bytes[k] == b'"' {
                let mut e = k + 1;
                while e < bytes.len() && bytes[e] != b'"' {
                    if bytes[e] == b'\\' {
                        e += 1;
                    }
                    e += 1;
                }
                let val = String::from_utf8_lossy(&bytes[k + 1..e]).into_owned();
                out.push((key, val));
                i = e + 1;
                continue;
            }
            i = k;
            continue;
        }
        i += 1;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn incognito_skips_disk_writes() {
        let p = ProfileRoot::incognito();
        let tmp = std::env::temp_dir().join("tb_incognito_test.txt");
        p.write_string(&tmp, "hi").unwrap();
        assert!(!tmp.exists());
    }

    #[test]
    fn export_import_roundtrip() {
        let mut p = ProfileRoot::open();
        // Use a sandbox path so we don't tread on the real profile.
        p.root = std::env::temp_dir().join("tb_profile_test_export");
        let _ = std::fs::remove_dir_all(&p.root);
        let _ = std::fs::create_dir_all(p.root.join("bookmarks"));
        let _ = std::fs::create_dir_all(p.root.join("history"));
        let _ = std::fs::create_dir_all(p.root.join("cookies"));
        p.write_string(&p.bookmarks_path(), "[\"a\"]").unwrap();
        let blob = export_profile_json(&p);
        let _ = std::fs::remove_dir_all(&p.root);
        let _ = std::fs::create_dir_all(p.root.join("bookmarks"));
        import_profile_json(&p, &blob).unwrap();
        let got = p.read_string(&p.bookmarks_path()).unwrap_or_default();
        assert!(got.contains("a"));
    }
}
