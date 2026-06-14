//! Page tree walker: resolve catalog → pages → individual page dicts.

use crate::object::PdfObj;
use std::collections::HashMap;

#[derive(Debug, Default)]
pub struct PageTree {
    /// Object num → resolved object.
    pub objects: HashMap<u32, PdfObj>,
    /// Resolved catalog id.
    pub catalog_id: Option<u32>,
}

impl PageTree {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_object(&mut self, num: u32, obj: PdfObj) {
        self.objects.insert(num, obj);
    }

    pub fn resolve(&self, obj: &PdfObj) -> Option<PdfObj> {
        match obj {
            PdfObj::Ref(n, _) => self.objects.get(n).cloned(),
            _ => Some(obj.clone()),
        }
    }

    /// Walk the page tree starting from the catalog. Returns the list
    /// of leaf Page objects.
    pub fn pages(&self) -> Vec<PdfObj> {
        let cat = match self.catalog_id.and_then(|id| self.objects.get(&id)) {
            Some(c) => c,
            None => return Vec::new(),
        };
        let pages_root = match cat {
            PdfObj::Dict(d) => d.get("Pages").and_then(|r| self.resolve(r)),
            _ => None,
        };
        let mut out = Vec::new();
        if let Some(root) = pages_root {
            self.collect_pages(&root, &mut out);
        }
        out
    }

    fn collect_pages(&self, node: &PdfObj, out: &mut Vec<PdfObj>) {
        let dict = match node {
            PdfObj::Dict(d) => d,
            _ => return,
        };
        let typ = match dict.get("Type") {
            Some(PdfObj::Name(n)) => n.as_str(),
            _ => return,
        };
        match typ {
            "Pages" => {
                if let Some(PdfObj::Array(kids)) = dict.get("Kids") {
                    for k in kids {
                        if let Some(resolved) = self.resolve(k) {
                            self.collect_pages(&resolved, out);
                        }
                    }
                }
            }
            "Page" => {
                out.push(node.clone());
            }
            _ => {}
        }
    }

    /// Get the content stream(s) for a page.
    pub fn page_contents(&self, page: &PdfObj) -> Vec<PdfObj> {
        let dict = match page {
            PdfObj::Dict(d) => d,
            _ => return Vec::new(),
        };
        let contents = match dict.get("Contents") {
            Some(c) => c,
            None => return Vec::new(),
        };
        match self.resolve(contents) {
            Some(PdfObj::Stream { .. }) => vec![self.resolve(contents).unwrap()],
            Some(PdfObj::Array(arr)) => arr.iter().filter_map(|c| self.resolve(c)).collect(),
            _ => Vec::new(),
        }
    }

    pub fn page_media_box(&self, page: &PdfObj) -> Option<(f64, f64, f64, f64)> {
        let dict = match page {
            PdfObj::Dict(d) => d,
            _ => return None,
        };
        let arr = match dict.get("MediaBox") {
            Some(PdfObj::Array(a)) => a,
            _ => return None,
        };
        if arr.len() != 4 {
            return None;
        }
        let f = |o: &PdfObj| -> Option<f64> {
            match o {
                PdfObj::Int(n) => Some(*n as f64),
                PdfObj::Real(f) => Some(*f),
                _ => None,
            }
        };
        Some((f(&arr[0])?, f(&arr[1])?, f(&arr[2])?, f(&arr[3])?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn dict_with(pairs: &[(&str, PdfObj)]) -> PdfObj {
        let mut d = HashMap::new();
        for (k, v) in pairs {
            d.insert(k.to_string(), v.clone());
        }
        PdfObj::Dict(d)
    }

    #[test]
    fn pages_walks_single_page_tree() {
        let mut t = PageTree::new();
        let page = dict_with(&[
            ("Type", PdfObj::Name("Page".into())),
            (
                "MediaBox",
                PdfObj::Array(vec![
                    PdfObj::Int(0),
                    PdfObj::Int(0),
                    PdfObj::Int(612),
                    PdfObj::Int(792),
                ]),
            ),
        ]);
        let pages = dict_with(&[
            ("Type", PdfObj::Name("Pages".into())),
            ("Kids", PdfObj::Array(vec![PdfObj::Ref(2, 0)])),
            ("Count", PdfObj::Int(1)),
        ]);
        let catalog = dict_with(&[
            ("Type", PdfObj::Name("Catalog".into())),
            ("Pages", PdfObj::Ref(3, 0)),
        ]);
        t.add_object(1, catalog);
        t.add_object(2, page);
        t.add_object(3, pages);
        t.catalog_id = Some(1);
        let leaves = t.pages();
        assert_eq!(leaves.len(), 1);
        let mb = t.page_media_box(&leaves[0]).unwrap();
        assert_eq!(mb, (0.0, 0.0, 612.0, 792.0));
    }

    #[test]
    fn pages_walks_nested_pages_node() {
        let mut t = PageTree::new();
        let p1 = dict_with(&[("Type", PdfObj::Name("Page".into()))]);
        let p2 = dict_with(&[("Type", PdfObj::Name("Page".into()))]);
        let sub = dict_with(&[
            ("Type", PdfObj::Name("Pages".into())),
            (
                "Kids",
                PdfObj::Array(vec![PdfObj::Ref(2, 0), PdfObj::Ref(3, 0)]),
            ),
        ]);
        let pages = dict_with(&[
            ("Type", PdfObj::Name("Pages".into())),
            ("Kids", PdfObj::Array(vec![PdfObj::Ref(5, 0)])),
        ]);
        let catalog = dict_with(&[
            ("Type", PdfObj::Name("Catalog".into())),
            ("Pages", PdfObj::Ref(4, 0)),
        ]);
        t.add_object(1, catalog);
        t.add_object(2, p1);
        t.add_object(3, p2);
        t.add_object(4, pages);
        t.add_object(5, sub);
        t.catalog_id = Some(1);
        assert_eq!(t.pages().len(), 2);
    }
}
