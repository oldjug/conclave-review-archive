//! chrome.* API registry.
//!
//! Routes JS-side `chrome.tabs.create(...)` etc. through the extension
//! service worker into the browser process. V1 models the API surface
//! as a typed registry; the JS binding generator references this to
//! produce the per-method stubs.

use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Namespace {
    Tabs,
    Windows,
    Runtime,
    Storage,
    Scripting,
    WebNavigation,
    DeclarativeNetRequest,
    Cookies,
    Alarms,
    Notifications,
    Commands,
    ContextMenus,
    Permissions,
    I18n,
    Action,
}

impl Namespace {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Tabs => "tabs",
            Self::Windows => "windows",
            Self::Runtime => "runtime",
            Self::Storage => "storage",
            Self::Scripting => "scripting",
            Self::WebNavigation => "webNavigation",
            Self::DeclarativeNetRequest => "declarativeNetRequest",
            Self::Cookies => "cookies",
            Self::Alarms => "alarms",
            Self::Notifications => "notifications",
            Self::Commands => "commands",
            Self::ContextMenus => "contextMenus",
            Self::Permissions => "permissions",
            Self::I18n => "i18n",
            Self::Action => "action",
        }
    }
}

#[derive(Debug, Clone)]
pub struct ApiMethod {
    pub name: String,
    pub required_permissions: Vec<String>,
}

#[derive(Debug, Default)]
pub struct ApiRegistry {
    by_ns: HashMap<Namespace, Vec<ApiMethod>>,
}

impl ApiRegistry {
    pub fn standard() -> Self {
        let mut r = Self::default();
        // Hand-wired Chrome MV3 surface — what extensions commonly use.
        r.add(Namespace::Tabs, "create", &["tabs"]);
        r.add(Namespace::Tabs, "query", &[]);
        r.add(Namespace::Tabs, "update", &["tabs"]);
        r.add(Namespace::Tabs, "remove", &["tabs"]);
        r.add(Namespace::Storage, "get", &["storage"]);
        r.add(Namespace::Storage, "set", &["storage"]);
        r.add(Namespace::Runtime, "sendMessage", &[]);
        r.add(Namespace::Runtime, "getURL", &[]);
        r.add(Namespace::Runtime, "id", &[]);
        r.add(Namespace::Scripting, "executeScript", &["scripting"]);
        r.add(
            Namespace::DeclarativeNetRequest,
            "updateDynamicRules",
            &["declarativeNetRequest"],
        );
        r.add(Namespace::Cookies, "get", &["cookies"]);
        r.add(Namespace::Cookies, "set", &["cookies"]);
        r.add(Namespace::Alarms, "create", &["alarms"]);
        r.add(Namespace::Notifications, "create", &["notifications"]);
        r.add(Namespace::Action, "setBadgeText", &[]);
        r.add(Namespace::I18n, "getMessage", &[]);
        r
    }

    pub fn add(&mut self, ns: Namespace, name: &str, perms: &[&str]) {
        self.by_ns.entry(ns).or_default().push(ApiMethod {
            name: name.to_string(),
            required_permissions: perms.iter().map(|s| s.to_string()).collect(),
        });
    }

    pub fn method(&self, ns: Namespace, name: &str) -> Option<&ApiMethod> {
        self.by_ns.get(&ns)?.iter().find(|m| m.name == name)
    }

    /// Returns true if the extension's permissions cover what
    /// `ns.name` requires.
    pub fn is_authorized(&self, ns: Namespace, name: &str, granted: &[String]) -> bool {
        let m = match self.method(ns, name) {
            Some(m) => m,
            None => return false,
        };
        m.required_permissions
            .iter()
            .all(|p| granted.iter().any(|g| g == p))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn standard_registry_has_tabs_create() {
        let r = ApiRegistry::standard();
        assert!(r.method(Namespace::Tabs, "create").is_some());
    }

    #[test]
    fn authorization_requires_listed_permission() {
        let r = ApiRegistry::standard();
        let no_perms: Vec<String> = vec![];
        assert!(!r.is_authorized(Namespace::Tabs, "create", &no_perms));
        let with: Vec<String> = vec!["tabs".into()];
        assert!(r.is_authorized(Namespace::Tabs, "create", &with));
    }

    #[test]
    fn method_with_no_required_perm_is_always_authorized() {
        let r = ApiRegistry::standard();
        let no_perms: Vec<String> = vec![];
        assert!(r.is_authorized(Namespace::Runtime, "sendMessage", &no_perms));
    }

    #[test]
    fn unknown_method_is_not_authorized() {
        let r = ApiRegistry::standard();
        let with: Vec<String> = vec!["tabs".into()];
        assert!(!r.is_authorized(Namespace::Tabs, "doesNotExist", &with));
    }
}
