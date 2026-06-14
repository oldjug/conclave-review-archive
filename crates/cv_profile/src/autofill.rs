//! Autofill heuristics — match form fields against the saved
//! address-book / payment-card store. V1 ships address autofill;
//! payment cards follow.

use std::collections::HashMap;

#[derive(Debug, Clone, Default)]
pub struct AddressProfile {
    pub full_name: String,
    pub email: String,
    pub phone: String,
    pub street_line1: String,
    pub street_line2: String,
    pub city: String,
    pub state: String,
    pub postal_code: String,
    pub country: String,
}

#[derive(Debug, Default)]
pub struct AutofillStore {
    addresses: Vec<AddressProfile>,
}

impl AutofillStore {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn save(&mut self, profile: AddressProfile) {
        self.addresses.push(profile);
    }
    pub fn primary(&self) -> Option<&AddressProfile> {
        self.addresses.first()
    }

    /// Suggest a value for `field_name` based on the primary profile.
    /// Match is purely heuristic — name / autocomplete hints both
    /// route through `field_hint`.
    pub fn suggest(&self, field_hint: &str) -> Option<String> {
        let p = self.primary()?;
        let h = field_hint.to_lowercase();
        let v = if h.contains("email") {
            &p.email
        } else if h.contains("name") {
            &p.full_name
        } else if h.contains("phone") || h.contains("tel") {
            &p.phone
        } else if h.contains("postal") || h.contains("zip") {
            &p.postal_code
        } else if h.contains("city") {
            &p.city
        } else if h.contains("state") || h.contains("region") {
            &p.state
        } else if h.contains("country") {
            &p.country
        } else if h.contains("address-line1") || h.contains("street1") {
            &p.street_line1
        } else if h.contains("address-line2") || h.contains("street2") {
            &p.street_line2
        } else {
            return None;
        };
        if v.is_empty() { None } else { Some(v.clone()) }
    }

    /// Fill an entire HTML form. Keys are autocomplete hints. Returns
    /// a map from form-field name to value to set.
    pub fn fill_form(&self, hints: &[(&str, &str)]) -> HashMap<String, String> {
        let mut out = HashMap::new();
        for (name, hint) in hints {
            if let Some(v) = self.suggest(hint) {
                out.insert(name.to_string(), v);
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> AutofillStore {
        let mut s = AutofillStore::new();
        let mut p = AddressProfile::default();
        p.full_name = "Alice Example".into();
        p.email = "alice@example.com".into();
        p.phone = "+1-555-0100".into();
        p.street_line1 = "123 Main".into();
        p.city = "Springfield".into();
        p.state = "CA".into();
        p.postal_code = "94000".into();
        p.country = "USA".into();
        s.save(p);
        s
    }

    #[test]
    fn suggests_email_from_hint() {
        let s = sample();
        assert_eq!(s.suggest("email").as_deref(), Some("alice@example.com"));
    }

    #[test]
    fn suggests_postal_code_from_zip_hint() {
        let s = sample();
        assert_eq!(s.suggest("zip").as_deref(), Some("94000"));
    }

    #[test]
    fn suggest_returns_none_for_unknown_field() {
        let s = sample();
        assert!(s.suggest("ssn").is_none());
    }

    #[test]
    fn fill_form_routes_multiple_fields() {
        let s = sample();
        let map = s.fill_form(&[
            ("user_email", "email"),
            ("user_name", "name"),
            ("user_phone", "phone"),
            ("user_unknown", "wat"),
        ]);
        assert_eq!(
            map.get("user_email").map(String::as_str),
            Some("alice@example.com")
        );
        assert_eq!(
            map.get("user_name").map(String::as_str),
            Some("Alice Example")
        );
        assert!(!map.contains_key("user_unknown"));
    }
}
