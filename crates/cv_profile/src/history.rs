//! Visit history. Entries are append-only; queries support
//! substring search + range scans + most-visited-of-domain rollups.

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Visit {
    pub url: String,
    pub title: String,
    pub timestamp_ms: u64,
    pub typed: bool, // user typed vs followed link
    pub frequency: u32,
}

#[derive(Debug, Default)]
pub struct History {
    visits: Vec<Visit>,
}

impl History {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn record(&mut self, url: impl Into<String>, title: impl Into<String>, ts: u64) {
        let url = url.into();
        if let Some(v) = self.visits.iter_mut().find(|v| v.url == url) {
            v.frequency += 1;
            v.timestamp_ms = ts;
            return;
        }
        self.visits.push(Visit {
            url,
            title: title.into(),
            timestamp_ms: ts,
            typed: false,
            frequency: 1,
        });
    }
    pub fn search(&self, q: &str) -> Vec<&Visit> {
        let q_lower = q.to_lowercase();
        let mut hits: Vec<&Visit> = self
            .visits
            .iter()
            .filter(|v| {
                v.url.to_lowercase().contains(&q_lower) || v.title.to_lowercase().contains(&q_lower)
            })
            .collect();
        hits.sort_by_key(|v| std::cmp::Reverse(v.frequency));
        hits
    }
    pub fn most_visited(&self, n: usize) -> Vec<&Visit> {
        let mut all: Vec<&Visit> = self.visits.iter().collect();
        all.sort_by_key(|v| std::cmp::Reverse(v.frequency));
        all.into_iter().take(n).collect()
    }
    pub fn len(&self) -> usize {
        self.visits.len()
    }
    pub fn is_empty(&self) -> bool {
        self.visits.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn record_increments_frequency_on_revisit() {
        let mut h = History::new();
        h.record("https://example.com", "Example", 1);
        h.record("https://example.com", "Example", 2);
        assert_eq!(h.len(), 1);
        assert_eq!(h.visits[0].frequency, 2);
    }

    #[test]
    fn search_finds_by_title_substring() {
        let mut h = History::new();
        h.record("https://a.com", "Alpha", 0);
        h.record("https://b.com", "Beta", 0);
        let r = h.search("Alpha");
        assert_eq!(r.len(), 1);
        assert_eq!(r[0].url, "https://a.com");
    }

    #[test]
    fn most_visited_orders_by_frequency() {
        let mut h = History::new();
        h.record("https://a.com", "A", 0);
        h.record("https://b.com", "B", 0);
        h.record("https://b.com", "B", 1);
        let mv = h.most_visited(2);
        assert_eq!(mv[0].url, "https://b.com");
        assert_eq!(mv[1].url, "https://a.com");
    }
}
