//! Password vault — AES-GCM encrypted at rest, keyed off the
//! profile's master key. V1 stores in-memory; the on-disk binding
//! routes through `cv_crypto::aes_gcm` once the FFI integration
//! lands.

use std::collections::HashMap;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Credential {
    pub origin: String,
    pub username: String,
    pub password: String,
    pub created_ms: u64,
    pub last_used_ms: u64,
}

#[derive(Debug, Default)]
pub struct PasswordVault {
    by_origin: HashMap<String, Vec<Credential>>,
    /// Master-key fingerprint — caller derives via KDF before
    /// constructing the vault.
    pub master_key_fingerprint: [u8; 32],
}

impl PasswordVault {
    pub fn new(master_key_fingerprint: [u8; 32]) -> Self {
        Self {
            by_origin: HashMap::new(),
            master_key_fingerprint,
        }
    }

    pub fn save(&mut self, cred: Credential) {
        self.by_origin
            .entry(cred.origin.clone())
            .or_default()
            .push(cred);
    }

    pub fn list_for(&self, origin: &str) -> &[Credential] {
        self.by_origin
            .get(origin)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    pub fn touch(&mut self, origin: &str, username: &str, ts: u64) {
        if let Some(list) = self.by_origin.get_mut(origin) {
            if let Some(c) = list.iter_mut().find(|c| c.username == username) {
                c.last_used_ms = ts;
            }
        }
    }

    pub fn delete(&mut self, origin: &str, username: &str) -> bool {
        if let Some(list) = self.by_origin.get_mut(origin) {
            let before = list.len();
            list.retain(|c| c.username != username);
            if list.is_empty() {
                self.by_origin.remove(origin);
            }
            before != self.by_origin.get(origin).map(|v| v.len()).unwrap_or(0)
        } else {
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cred(o: &str, u: &str, p: &str) -> Credential {
        Credential {
            origin: o.into(),
            username: u.into(),
            password: p.into(),
            created_ms: 0,
            last_used_ms: 0,
        }
    }

    #[test]
    fn save_then_list_round_trip() {
        let mut v = PasswordVault::new([0; 32]);
        v.save(cred("https://a.com", "alice", "secret"));
        v.save(cred("https://a.com", "bob", "other"));
        assert_eq!(v.list_for("https://a.com").len(), 2);
    }

    #[test]
    fn touch_updates_last_used() {
        let mut v = PasswordVault::new([0; 32]);
        v.save(cred("a.com", "alice", "x"));
        v.touch("a.com", "alice", 12345);
        assert_eq!(v.list_for("a.com")[0].last_used_ms, 12345);
    }

    #[test]
    fn delete_removes_credential() {
        let mut v = PasswordVault::new([0; 32]);
        v.save(cred("a.com", "alice", "x"));
        assert!(v.delete("a.com", "alice"));
        assert_eq!(v.list_for("a.com").len(), 0);
    }
}
