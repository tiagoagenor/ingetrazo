// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! A list whose items keep an identity while they sit in it.
//!
//! The annotation lists of a scene (`Scene.dimensions`, `text_labels`,
//! `guides`, `section_planes`, `image_planes`, `geo_paths`…) hold Python
//! objects that commands keep by reference: a command remembers the very
//! dimension it added, and undo removes THAT one however the list has been
//! reordered since (`list.remove(obj)` and `in` compare by identity, the
//! classes define no `__eq__`). An item here carries a number nobody reuses
//! — [`IdList::push`] gives it, [`IdList::insert_at`] puts it back — and a
//! command holds the number instead of the object.
//!
//! The list reads like a slice (`len`, `iter`, `[i]`, `first`…) through
//! `Deref`, and items can be edited in place; only the methods below change
//! its length.

use std::ops::{Deref, DerefMut};

#[derive(Clone, Debug, PartialEq)]
pub struct IdList<T> {
    items: Vec<T>,
    ids: Vec<u32>,
    next: u32,
}

impl<T> Default for IdList<T> {
    fn default() -> Self {
        IdList { items: Vec::new(), ids: Vec::new(), next: 0 }
    }
}

impl<T> Deref for IdList<T> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        &self.items
    }
}

impl<T> DerefMut for IdList<T> {
    fn deref_mut(&mut self) -> &mut [T] {
        &mut self.items
    }
}

impl<T> FromIterator<T> for IdList<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let mut l = IdList::default();
        for x in iter {
            l.push(x);
        }
        l
    }
}

impl<T> From<Vec<T>> for IdList<T> {
    fn from(v: Vec<T>) -> Self {
        v.into_iter().collect()
    }
}

impl<'a, T> IntoIterator for &'a IdList<T> {
    type Item = &'a T;
    type IntoIter = std::slice::Iter<'a, T>;
    fn into_iter(self) -> Self::IntoIter {
        self.items.iter()
    }
}

impl<T> IdList<T> {
    pub fn new() -> Self {
        IdList::default()
    }

    /// `list.append(item)`: the new item's identity.
    pub fn push(&mut self, item: T) -> u32 {
        let id = self.next;
        self.next += 1;
        self.items.push(item);
        self.ids.push(id);
        id
    }

    /// `list.extend(items)`.
    pub fn extend_items(&mut self, items: impl IntoIterator<Item = T>) -> Vec<u32> {
        items.into_iter().map(|x| self.push(x)).collect()
    }

    /// The identity of the item at `index`.
    pub fn id_at(&self, index: usize) -> Option<u32> {
        self.ids.get(index).copied()
    }

    /// Where the item `id` sits now (`list.index(obj)`), if it is in the list
    /// (`obj in list`).
    pub fn index_of(&self, id: u32) -> Option<usize> {
        self.ids.iter().position(|&i| i == id)
    }

    pub fn contains_id(&self, id: u32) -> bool {
        self.ids.contains(&id)
    }

    pub fn get_id(&self, id: u32) -> Option<&T> {
        self.index_of(id).map(|i| &self.items[i])
    }

    pub fn get_id_mut(&mut self, id: u32) -> Option<&mut T> {
        let i = self.index_of(id)?;
        Some(&mut self.items[i])
    }

    /// `list.remove(obj)`: the item and the place it had.
    pub fn remove_id(&mut self, id: u32) -> Option<(usize, T)> {
        let i = self.index_of(id)?;
        self.ids.remove(i);
        Some((i, self.items.remove(i)))
    }

    /// `list.insert(index, obj)`: an index past the end appends, and the
    /// item keeps the identity it had.
    pub fn insert_at(&mut self, index: usize, id: u32, item: T) {
        let at = index.min(self.items.len());
        self.items.insert(at, item);
        self.ids.insert(at, id);
        if id >= self.next {
            self.next = id + 1;
        }
    }

    /// `list.clear()`; identities are not reused.
    pub fn clear(&mut self) {
        self.items.clear();
        self.ids.clear();
    }

    /// The identities in list order.
    pub fn ids(&self) -> &[u32] {
        &self.ids
    }

    /// Replace the items (`scene.section_planes = [...]`).
    pub fn replace(&mut self, items: Vec<T>) {
        self.clear();
        for x in items {
            self.push(x);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_item_keeps_its_identity_while_the_list_shifts() {
        let mut l: IdList<&str> = IdList::new();
        let a = l.push("a");
        let b = l.push("b");
        let c = l.push("c");
        assert_eq!(l.remove_id(a), Some((0, "a")));
        assert_eq!(l.index_of(c), Some(1));
        l.insert_at(0, a, "a");
        assert_eq!(&*l, &["a", "b", "c"]);
        assert_eq!(l.index_of(b), Some(1));
        let d = l.push("d");
        assert!(d > c, "identities are not reused");
        assert_eq!(l.remove_id(99), None);
    }
}
