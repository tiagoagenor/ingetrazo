// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! `PySet` against CPython 3.12's `set` of integer pairs
//! (scripts/golden/make_pyset_fixtures.py): the tuple hash, and the
//! iteration order after a history of adds and discards and after
//! `set(other)`.

use std::path::Path;

use ingetrazo_formats::pyjson::{loads, Value};
use ingetrazo_ops::pyset::{hash_pair, PySet};

fn arr(v: &Value) -> &Vec<Value> {
    match v {
        Value::Array(a) => a,
        _ => panic!("{v:?}"),
    }
}

fn int(v: &Value) -> i64 {
    v.as_i64().unwrap_or_else(|| panic!("{v:?}"))
}

#[test]
fn set_order_is_cpythons() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/pyset.json");
    let cases = loads(&std::fs::read_to_string(path).unwrap()).unwrap();
    let mut bad = Vec::new();
    for (n, case) in arr(&cases).iter().enumerate() {
        let keys: Vec<(i64, i64)> = arr(case.get("keys").unwrap()).iter().map(|k| (int(&arr(k)[0]), int(&arr(k)[1]))).collect();
        let hashes: Vec<i64> = arr(case.get("hash").unwrap()).iter().map(int).collect();
        for (i, &(a, b)) in keys.iter().enumerate() {
            if hash_pair(a, b) != hashes[i] {
                bad.push(format!("case {n}: hash of {:?} is {} not {}", keys[i], hash_pair(a, b), hashes[i]));
            }
        }
        let mut s = PySet::new();
        for step in arr(case.get("hist").unwrap()) {
            let (op, k) = match (&arr(step)[0], int(&arr(step)[1])) {
                (Value::Str(o), k) => (o.as_str(), k as usize),
                _ => panic!(),
            };
            let (a, b) = keys[k];
            match op {
                "a" => s.add(k as u32, hash_pair(a, b)),
                _ => s.discard(k as u32, hash_pair(a, b)),
            }
        }
        let want: Vec<u32> = arr(case.get("order").unwrap()).iter().map(|v| int(v) as u32).collect();
        if s.iter() != want {
            bad.push(format!("case {n}: order {:?} not {:?}", s.iter(), want));
        }
        let want: Vec<u32> = arr(case.get("copy").unwrap()).iter().map(|v| int(v) as u32).collect();
        if PySet::copy_of(&s).iter() != want {
            bad.push(format!("case {n}: copy order {:?} not {:?}", PySet::copy_of(&s).iter(), want));
        }
    }
    assert!(bad.is_empty(), "{} problems:\n{}", bad.len(), bad.iter().take(8).cloned().collect::<Vec<_>>().join("\n"));
}
