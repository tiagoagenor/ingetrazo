// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! CPython's `set` of small-integer tuples, to the slot: hash table, linear
//! probing, perturbation, resize policy, dummies, and so the ITERATION ORDER
//! (`Objects/setobject.c`, 3.11/3.12).
//!
//! Python code that loops over a set of `(int, int)` keys sees them in table
//! order. That order is a function of the values and the insertion history,
//! not of memory addresses, so it can be reproduced — and the arrangement
//! (`core/arrangement.py`) emits its edges and faces in it. Elements here are
//! caller-chosen `u32` ids; the hash given with each is the hash of the tuple
//! Python would hold.

const LINEAR_PROBES: usize = 9;
const PERTURB_SHIFT: u32 = 5;
const MIN_SIZE: usize = 8;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Slot {
    Empty,
    Dummy,
    Active(u32),
}

#[derive(Clone, Copy, Debug)]
struct Entry {
    slot: Slot,
    hash: i64,
}

const EMPTY: Entry = Entry { slot: Slot::Empty, hash: 0 };

/// `hash(int)` for an integer far below 2**61: itself, but -1 hashes to -2.
pub fn hash_int(x: i64) -> i64 {
    if x == -1 {
        -2
    } else {
        x
    }
}

/// `hash((a, b))` of two ints (tuple hash, xxHash-derived, 64-bit).
pub fn hash_pair(a: i64, b: i64) -> i64 {
    const P1: u64 = 11400714785074694791;
    const P2: u64 = 14029467366897019727;
    const P5: u64 = 2870177450012600261;
    let mut acc = P5;
    for item in [a, b] {
        let lane = hash_int(item) as u64;
        acc = acc.wrapping_add(lane.wrapping_mul(P2));
        acc = acc.rotate_left(31);
        acc = acc.wrapping_mul(P1);
    }
    acc = acc.wrapping_add(2u64 ^ (P5 ^ 3527539));
    if acc == u64::MAX {
        1546275796
    } else {
        acc as i64
    }
}

/// A Python `set` of ids.
#[derive(Clone, Debug)]
pub struct PySet {
    table: Vec<Entry>,
    mask: usize,
    fill: usize,
    used: usize,
}

impl Default for PySet {
    fn default() -> Self {
        PySet::new()
    }
}

impl PySet {
    pub fn new() -> PySet {
        PySet { table: vec![EMPTY; MIN_SIZE], mask: MIN_SIZE - 1, fill: 0, used: 0 }
    }

    pub fn len(&self) -> usize {
        self.used
    }

    pub fn is_empty(&self) -> bool {
        self.used == 0
    }

    /// The elements in iteration order.
    pub fn iter(&self) -> Vec<u32> {
        self.table
            .iter()
            .filter_map(|e| match e.slot {
                Slot::Active(k) => Some(k),
                _ => None,
            })
            .collect()
    }

    /// `set_insert_clean`: put an element into a table that has no dummies
    /// and does not hold it.
    fn insert_clean(table: &mut [Entry], mask: usize, key: u32, hash: i64) {
        let mut perturb = hash as u64;
        let mut i = (hash as u64 as usize) & mask;
        loop {
            if table[i].slot == Slot::Empty {
                table[i] = Entry { slot: Slot::Active(key), hash };
                return;
            }
            if i + LINEAR_PROBES <= mask {
                for j in 1..=LINEAR_PROBES {
                    if table[i + j].slot == Slot::Empty {
                        table[i + j] = Entry { slot: Slot::Active(key), hash };
                        return;
                    }
                }
            }
            perturb >>= PERTURB_SHIFT;
            i = (i.wrapping_mul(5).wrapping_add(1).wrapping_add(perturb as usize)) & mask;
        }
    }

    /// `set_table_resize`.
    fn resize(&mut self, minused: usize) {
        let mut newsize = MIN_SIZE;
        while newsize <= minused {
            newsize <<= 1;
        }
        let old = std::mem::replace(&mut self.table, vec![EMPTY; newsize]);
        self.mask = newsize - 1;
        for e in &old {
            if let Slot::Active(k) = e.slot {
                Self::insert_clean(&mut self.table, self.mask, k, e.hash);
            }
        }
        self.fill = self.used;
    }

    /// `set.add(key)`.
    pub fn add(&mut self, key: u32, hash: i64) {
        let mask = self.mask;
        let mut i = (hash as u64 as usize) & mask;
        let mut freeslot: Option<usize> = None;
        let mut perturb = hash as u64;
        loop {
            let probes = if i + LINEAR_PROBES <= mask { LINEAR_PROBES } else { 0 };
            for j in i..=i + probes {
                let e = self.table[j];
                match e.slot {
                    Slot::Empty => {
                        // found_unused_or_dummy
                        match freeslot {
                            Some(f) => {
                                self.used += 1;
                                self.table[f] = Entry { slot: Slot::Active(key), hash };
                            }
                            None => {
                                self.fill += 1;
                                self.used += 1;
                                self.table[j] = Entry { slot: Slot::Active(key), hash };
                                if self.fill * 5 >= mask * 3 {
                                    let want = if self.used > 50000 { self.used * 2 } else { self.used * 4 };
                                    self.resize(want);
                                }
                            }
                        }
                        return;
                    }
                    Slot::Active(k) => {
                        if e.hash == hash && k == key {
                            return; // found_active
                        }
                    }
                    // The LAST dummy of the probe chain takes the new key,
                    // as measured against CPython 3.12 (tests/pyset_parity.rs).
                    Slot::Dummy => freeslot = Some(j),
                }
            }
            perturb >>= PERTURB_SHIFT;
            i = (i.wrapping_mul(5).wrapping_add(1).wrapping_add(perturb as usize)) & mask;
        }
    }

    /// `set.discard(key)`.
    pub fn discard(&mut self, key: u32, hash: i64) {
        let mask = self.mask;
        let mut i = (hash as u64 as usize) & mask;
        let mut perturb = hash as u64;
        loop {
            let probes = if i + LINEAR_PROBES <= mask { LINEAR_PROBES } else { 0 };
            for j in i..=i + probes {
                let e = self.table[j];
                match e.slot {
                    Slot::Empty => return,
                    Slot::Active(k) if e.hash == hash && k == key => {
                        self.table[j] = Entry { slot: Slot::Dummy, hash: -1 };
                        self.used -= 1;
                        return;
                    }
                    _ => {}
                }
            }
            perturb >>= PERTURB_SHIFT;
            i = (i.wrapping_mul(5).wrapping_add(1).wrapping_add(perturb as usize)) & mask;
        }
    }

    /// `set(other)`: `set_merge` into an empty set.
    pub fn copy_of(other: &PySet) -> PySet {
        let mut so = PySet::new();
        if other.used == 0 {
            return so;
        }
        if (so.fill + other.used) * 5 >= so.mask * 3 {
            so.resize((so.used + other.used) * 2);
        }
        if so.fill == 0 && so.mask == other.mask && other.fill == other.used {
            so.table = other.table.clone();
            so.fill = other.fill;
            so.used = other.used;
            return so;
        }
        if so.fill == 0 {
            so.fill = other.used;
            so.used = other.used;
            for e in &other.table {
                if let Slot::Active(k) = e.slot {
                    Self::insert_clean(&mut so.table, so.mask, k, e.hash);
                }
            }
        }
        so
    }
}

