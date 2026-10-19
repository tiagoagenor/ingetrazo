// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! `triangulate` against `core.triangulate.triangulate`
//! (scripts/golden/make_triangulate_fixtures.py): the same triangles, in the
//! same order, the same float32 corners.

use std::path::Path;

use ingetrazo_formats::pyjson::{loads, Value};
use ingetrazo_geom::triangulate::triangulate;
use ingetrazo_math::{Hypot, V3};

fn arr(v: &Value) -> &Vec<Value> {
    match v {
        Value::Array(a) => a,
        _ => panic!("{v:?}"),
    }
}

fn f(v: &Value) -> f32 {
    match v {
        Value::Str(h) => f32::from_bits(u32::from_str_radix(h, 16).unwrap()),
        _ => panic!("{v:?}"),
    }
}

fn p(v: &Value) -> V3 {
    let a = arr(v);
    V3(f(&a[0]), f(&a[1]), f(&a[2]))
}

fn bits(v: V3) -> [u32; 3] {
    [v.0.to_bits(), v.1.to_bits(), v.2.to_bits()]
}

#[test]
fn triangulate_matches_python() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/triangulate.jsonl");
    let text = std::fs::read_to_string(path).unwrap();
    let mut fails = Vec::new();
    let mut n = 0;
    for (i, line) in text.lines().enumerate() {
        let c = loads(line).unwrap();
        let outer: Vec<V3> = arr(c.get("outer").unwrap()).iter().map(p).collect();
        let holes: Vec<Vec<V3>> = arr(c.get("holes").unwrap()).iter().map(|h| arr(h).iter().map(p).collect()).collect();
        let normal = match c.get("normal") {
            Some(Value::Null) | None => None,
            Some(v) => Some(p(v)),
        };
        let got: Vec<[[u32; 3]; 3]> =
            triangulate(&outer, &holes, normal, Hypot::Msvc).into_iter().map(|t| [bits(t[0]), bits(t[1]), bits(t[2])]).collect();
        let want: Vec<[[u32; 3]; 3]> = arr(c.get("tris").unwrap())
            .iter()
            .map(|t| {
                let t = arr(t);
                [bits(p(&t[0])), bits(p(&t[1])), bits(p(&t[2]))]
            })
            .collect();
        if got != want {
            fails.push(format!("#{i} {:?}: {} vs {} triangles", c.get("kind"), got.len(), want.len()));
        }
        n += 1;
    }
    assert!(fails.is_empty(), "{} of {n} differ:\n{}", fails.len(), fails.iter().take(10).cloned().collect::<Vec<_>>().join("\n"));
}
