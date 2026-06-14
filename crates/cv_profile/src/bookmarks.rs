//! Bookmarks store — tree of folders + items.

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BookmarkKind {
    Folder {
        name: String,
        children: Vec<Bookmark>,
    },
    Item {
        title: String,
        url: String,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Bookmark {
    pub id: u32,
    pub kind: BookmarkKind,
}

#[derive(Debug, Default)]
pub struct BookmarksStore {
    next_id: u32,
    root_children: Vec<Bookmark>,
}

impl BookmarksStore {
    pub fn new() -> Self {
        Self::default()
    }
    fn alloc(&mut self) -> u32 {
        self.next_id += 1;
        self.next_id
    }
    pub fn add_folder(&mut self, name: impl Into<String>) -> u32 {
        let id = self.alloc();
        self.root_children.push(Bookmark {
            id,
            kind: BookmarkKind::Folder {
                name: name.into(),
                children: Vec::new(),
            },
        });
        id
    }
    pub fn add_item(&mut self, title: impl Into<String>, url: impl Into<String>) -> u32 {
        let id = self.alloc();
        self.root_children.push(Bookmark {
            id,
            kind: BookmarkKind::Item {
                title: title.into(),
                url: url.into(),
            },
        });
        id
    }
    pub fn add_item_to_folder(
        &mut self,
        folder_id: u32,
        title: impl Into<String>,
        url: impl Into<String>,
    ) -> Option<u32> {
        let item_id = self.alloc();
        let item = Bookmark {
            id: item_id,
            kind: BookmarkKind::Item {
                title: title.into(),
                url: url.into(),
            },
        };
        for b in self.root_children.iter_mut() {
            if b.id == folder_id {
                if let BookmarkKind::Folder { children, .. } = &mut b.kind {
                    children.push(item);
                    return Some(item_id);
                }
            }
        }
        None
    }
    pub fn len(&self) -> usize {
        self.root_children.len()
    }
    pub fn root(&self) -> &[Bookmark] {
        &self.root_children
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_item_returns_increasing_ids() {
        let mut s = BookmarksStore::new();
        let a = s.add_item("A", "https://a.com");
        let b = s.add_item("B", "https://b.com");
        assert_eq!(a + 1, b);
    }

    #[test]
    fn add_to_folder_nests_item() {
        let mut s = BookmarksStore::new();
        let f = s.add_folder("Tech");
        s.add_item_to_folder(f, "Hacker News", "https://news.ycombinator.com")
            .unwrap();
        let folder = &s.root_children[0];
        match &folder.kind {
            BookmarkKind::Folder { children, .. } => assert_eq!(children.len(), 1),
            _ => panic!("expected folder"),
        }
    }

    #[test]
    fn add_to_nonexistent_folder_returns_none() {
        let mut s = BookmarksStore::new();
        assert!(s.add_item_to_folder(999, "Nope", "x").is_none());
    }
}
