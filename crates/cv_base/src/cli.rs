//! Tiny argv parser. Supports `--key=value`, `--key value`, `--flag`,
//! and positional args. No `clap` — strict-deps policy.

use std::collections::HashMap;

#[derive(Debug, Default, Clone)]
pub struct Cli {
    flags: HashMap<String, String>,
    positional: Vec<String>,
    program: String,
}

impl Cli {
    pub fn parse() -> Self {
        let raw: Vec<String> = std::env::args().collect();
        Self::parse_from(raw)
    }

    pub fn parse_from<I, S>(args: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let mut iter = args.into_iter().map(Into::into);
        let mut cli = Self::default();
        cli.program = iter.next().unwrap_or_default();
        let argv: Vec<String> = iter.collect();
        let mut i = 0;
        while i < argv.len() {
            let a = &argv[i];
            if let Some(rest) = a.strip_prefix("--") {
                if let Some((k, v)) = rest.split_once('=') {
                    cli.flags.insert(k.to_string(), v.to_string());
                } else if i + 1 < argv.len() && !argv[i + 1].starts_with("--") {
                    cli.flags.insert(rest.to_string(), argv[i + 1].clone());
                    i += 1;
                } else {
                    cli.flags.insert(rest.to_string(), String::new());
                }
            } else {
                cli.positional.push(a.clone());
            }
            i += 1;
        }
        cli
    }

    pub fn program(&self) -> &str {
        &self.program
    }

    pub fn flag(&self, name: &str) -> Option<&str> {
        self.flags.get(name).map(String::as_str)
    }

    pub fn has(&self, name: &str) -> bool {
        self.flags.contains_key(name)
    }

    pub fn positional(&self) -> &[String] {
        &self.positional
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn equals_form() {
        let c = Cli::parse_from(["prog", "--type=fetch", "https://example.com"]);
        assert_eq!(c.flag("type"), Some("fetch"));
        assert_eq!(c.positional(), &["https://example.com".to_string()]);
    }

    #[test]
    fn space_form() {
        let c = Cli::parse_from(["prog", "--type", "fetch", "url"]);
        assert_eq!(c.flag("type"), Some("fetch"));
        assert_eq!(c.positional(), &["url".to_string()]);
    }

    #[test]
    fn lone_flag() {
        let c = Cli::parse_from(["prog", "--headless"]);
        assert!(c.has("headless"));
    }
}
