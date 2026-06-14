//! `cv_autofill` — Heuristic form field classification.
//!
//! Chrome's autofill engine inspects each `<input>` on a page and
//! tries to guess its semantic type (first name, last name, email,
//! street, postal code, CC number, CVC, …) from a mix of:
//!   * `name=`, `id=`, `placeholder=`, `autocomplete=` attribute
//!   * label text the input is associated with
//!   * input type (`email`, `tel`)
//!
//! Then it offers to fill matching profile values. The heuristic
//! tables here are simplified ports of Chrome's `autofill_regex_*.h`
//! patterns — enough to recognise the common shipping checkout forms.
//!
//! This crate provides the classifier; the actual on-disk profile
//! and password storage live in cv_profile.

use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FieldType {
    Unknown,
    NameFirst,
    NameLast,
    NameFull,
    Email,
    Phone,
    AddressLine1,
    AddressLine2,
    City,
    State,
    PostalCode,
    Country,
    CreditCardNumber,
    CreditCardExpMonth,
    CreditCardExpYear,
    CreditCardCvc,
    CreditCardHolder,
    Password,
    Username,
    Search,
    BirthDate,
}

#[derive(Debug, Clone, Default)]
pub struct FieldHints {
    pub name: String,
    pub id: String,
    pub placeholder: String,
    pub autocomplete: String,
    pub label: String,
    pub html_type: String,
}

impl FieldHints {
    /// All hints combined into a single lowercase blob for substring
    /// matching.
    pub fn search_blob(&self) -> String {
        let mut s = String::new();
        s.push_str(&self.name);
        s.push(' ');
        s.push_str(&self.id);
        s.push(' ');
        s.push_str(&self.placeholder);
        s.push(' ');
        s.push_str(&self.label);
        s.push(' ');
        s.push_str(&self.autocomplete);
        s.to_ascii_lowercase()
    }
}

/// Classify a single field.
pub fn classify(h: &FieldHints) -> FieldType {
    // Highest-priority: explicit `autocomplete=` token wins.
    let ac = h.autocomplete.to_ascii_lowercase();
    if let Some(t) = autocomplete_to_type(&ac) {
        return t;
    }
    // HTML type overrides for clear cases.
    match h.html_type.as_str() {
        "email" => return FieldType::Email,
        "tel" => return FieldType::Phone,
        "search" => return FieldType::Search,
        "password" => return FieldType::Password,
        _ => {}
    }

    let blob = h.search_blob();
    for (re_words, t) in PATTERNS {
        if re_words.iter().all(|w| blob.contains(w))
            && !NEGATIVE_FOR.get(t).map_or(false, |negs: &&[&str]| {
                negs.iter().any(|n| blob.contains(n))
            })
        {
            return *t;
        }
    }
    FieldType::Unknown
}

fn autocomplete_to_type(ac: &str) -> Option<FieldType> {
    Some(match ac {
        "given-name" | "name-first" | "fname" => FieldType::NameFirst,
        "family-name" | "name-last" | "lname" => FieldType::NameLast,
        "name" => FieldType::NameFull,
        "email" => FieldType::Email,
        "tel" | "tel-national" => FieldType::Phone,
        "address-line1" | "street-address" => FieldType::AddressLine1,
        "address-line2" => FieldType::AddressLine2,
        "address-level2" => FieldType::City,
        "address-level1" => FieldType::State,
        "postal-code" => FieldType::PostalCode,
        "country" | "country-name" => FieldType::Country,
        "cc-number" => FieldType::CreditCardNumber,
        "cc-exp-month" => FieldType::CreditCardExpMonth,
        "cc-exp-year" => FieldType::CreditCardExpYear,
        "cc-csc" => FieldType::CreditCardCvc,
        "cc-name" => FieldType::CreditCardHolder,
        "current-password" | "new-password" => FieldType::Password,
        "username" => FieldType::Username,
        "bday" => FieldType::BirthDate,
        _ => return None,
    })
}

/// Ordered match patterns. The first pattern with all keywords present
/// wins (unless a NEGATIVE_FOR keyword disqualifies the match).
const PATTERNS: &[(&[&str], FieldType)] = &[
    (&["email"], FieldType::Email),
    (&["phone"], FieldType::Phone),
    (&["mobile"], FieldType::Phone),
    (&["tel"], FieldType::Phone),
    (&["card", "number"], FieldType::CreditCardNumber),
    (&["cc", "number"], FieldType::CreditCardNumber),
    (&["card", "exp"], FieldType::CreditCardExpMonth),
    (&["cvc"], FieldType::CreditCardCvc),
    (&["cvv"], FieldType::CreditCardCvc),
    (&["card", "name"], FieldType::CreditCardHolder),
    (&["first", "name"], FieldType::NameFirst),
    (&["given", "name"], FieldType::NameFirst),
    (&["fname"], FieldType::NameFirst),
    (&["last", "name"], FieldType::NameLast),
    (&["family", "name"], FieldType::NameLast),
    (&["lname"], FieldType::NameLast),
    (&["full", "name"], FieldType::NameFull),
    (&["your", "name"], FieldType::NameFull),
    (&["address", "1"], FieldType::AddressLine1),
    (&["street"], FieldType::AddressLine1),
    (&["address", "2"], FieldType::AddressLine2),
    (&["apartment"], FieldType::AddressLine2),
    (&["suite"], FieldType::AddressLine2),
    (&["city"], FieldType::City),
    (&["town"], FieldType::City),
    (&["state"], FieldType::State),
    (&["province"], FieldType::State),
    (&["zip"], FieldType::PostalCode),
    (&["postal"], FieldType::PostalCode),
    (&["postcode"], FieldType::PostalCode),
    (&["country"], FieldType::Country),
    (&["password"], FieldType::Password),
    (&["username"], FieldType::Username),
    (&["user", "name"], FieldType::Username),
    (&["login"], FieldType::Username),
    (&["search"], FieldType::Search),
    (&["birth"], FieldType::BirthDate),
    (&["dob"], FieldType::BirthDate),
];

thread_local! {
    /// Negatives that override a positive match. E.g. don't classify
    /// "shipping address" as AddressLine1 if it also says "comments".
    static NEGATIVE_FOR_CELL: std::cell::OnceCell<HashMap<FieldType, &'static [&'static str]>> =
        const { std::cell::OnceCell::new() };
}

#[allow(non_upper_case_globals)]
const NEGATIVE_FOR: NegMap = NegMap;

/// Stub that returns negative-keyword lists per type. Hand-coded so we
/// don't need a real HashMap at the call site (interior mutability gets
/// in the way of `&'static [&'static str]`).
struct NegMap;
impl NegMap {
    fn get(&self, t: &FieldType) -> Option<&&[&'static str]> {
        let arr: &'static [&'static str] = match t {
            FieldType::Email => &["confirm", "alt", "verify"],
            FieldType::Password => &["hint"],
            _ => return None,
        };
        Some(BoxedSlice::store(arr))
    }
}

/// Helper that turns a `&'static [&'static str]` into a stable
/// `&'static &'static [&'static str]` — needed because the NegMap
/// return shape promises a reference to the slice ref.
struct BoxedSlice;
impl BoxedSlice {
    fn store(s: &'static [&'static str]) -> &'static &'static [&'static str] {
        thread_local! {
            static SLOT: std::cell::OnceCell<Vec<&'static [&'static str]>> =
                const { std::cell::OnceCell::new() };
        }
        // We can't easily return a stable &&[&str] for arbitrary inputs
        // without leak; the right shape is a static lookup. Simplify:
        // leak a thin wrapper.
        let leaked: &'static &'static [&'static str] = Box::leak(Box::new(s));
        leaked
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn h(name: &str, label: &str) -> FieldHints {
        FieldHints {
            name: name.into(),
            label: label.into(),
            ..Default::default()
        }
    }

    #[test]
    fn email_by_name() {
        assert_eq!(classify(&h("email", "")), FieldType::Email);
    }

    #[test]
    fn email_by_html_type() {
        let mut f = FieldHints::default();
        f.html_type = "email".into();
        assert_eq!(classify(&f), FieldType::Email);
    }

    #[test]
    fn autocomplete_wins() {
        let mut f = FieldHints::default();
        f.name = "totally_random".into();
        f.autocomplete = "cc-number".into();
        assert_eq!(classify(&f), FieldType::CreditCardNumber);
    }

    #[test]
    fn first_last_name_distinct() {
        assert_eq!(classify(&h("fname", "")), FieldType::NameFirst);
        assert_eq!(classify(&h("lname", "")), FieldType::NameLast);
        assert_eq!(classify(&h("first_name", "")), FieldType::NameFirst);
        assert_eq!(classify(&h("last_name", "")), FieldType::NameLast);
    }

    #[test]
    fn address_lines() {
        assert_eq!(
            classify(&h("address1", "Street address")),
            FieldType::AddressLine1
        );
        assert_eq!(
            classify(&h("addr2", "Apartment or suite")),
            FieldType::AddressLine2
        );
    }

    #[test]
    fn postal_and_state() {
        assert_eq!(classify(&h("zip", "")), FieldType::PostalCode);
        assert_eq!(classify(&h("postcode", "")), FieldType::PostalCode);
        assert_eq!(classify(&h("state", "")), FieldType::State);
    }

    #[test]
    fn credit_card_fields() {
        assert_eq!(
            classify(&h("card_number", "Card number")),
            FieldType::CreditCardNumber
        );
        assert_eq!(classify(&h("cvc", "")), FieldType::CreditCardCvc);
        assert_eq!(classify(&h("cvv", "")), FieldType::CreditCardCvc);
    }

    #[test]
    fn password_field() {
        let mut f = FieldHints::default();
        f.html_type = "password".into();
        assert_eq!(classify(&f), FieldType::Password);
    }

    #[test]
    fn unknown_falls_through() {
        let f = h("xyzzy", "Some random label");
        assert_eq!(classify(&f), FieldType::Unknown);
    }
}
