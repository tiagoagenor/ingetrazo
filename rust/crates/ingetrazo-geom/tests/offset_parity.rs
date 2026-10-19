// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! `offset` and `bevel_profiles` against what the Python `core.offset` and
//! `core.bevel_profiles` computed (scripts/golden/make_offset_fixtures.py):
//! every float, exactly.

use std::cell::RefCell;
use std::path::Path;

use ingetrazo_formats::pyjson::{loads, Value};
use ingetrazo_geom::bevel_profiles::{self as bp, Control, Shape};
use ingetrazo_geom::offset::{self, Face};
use ingetrazo_math::{Hypot, V3};

/// The fixtures were written on Windows: MSVC's hypot.
const H: Hypot = Hypot::Msvc;

fn rows(name: &str) -> Vec<Value> {
    let p = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data").join(name);
    let text = std::fs::read_to_string(p).unwrap();
    text.lines().skip(1).map(|l| loads(l).unwrap()).collect()
}

fn s(v: &Value) -> &str {
    match v {
        Value::Str(x) => x,
        _ => panic!("{v:?}"),
    }
}

fn arr(v: &Value) -> &Vec<Value> {
    match v {
        Value::Array(a) => a,
        _ => panic!("{v:?}"),
    }
}

fn b(v: &Value) -> bool {
    match v {
        Value::Bool(x) => *x,
        _ => panic!("{v:?}"),
    }
}

fn f32b(v: &Value) -> f32 {
    f32::from_bits(u32::from_str_radix(s(v), 16).unwrap())
}

fn f64b(v: &Value) -> f64 {
    f64::from_bits(u64::from_str_radix(s(v), 16).unwrap())
}

fn v3(v: &Value) -> V3 {
    let a = arr(v);
    V3(f32b(&a[0]), f32b(&a[1]), f32b(&a[2]))
}

fn vs(v: &Value) -> Vec<V3> {
    arr(v).iter().map(v3).collect()
}

fn same3(a: V3, b: V3) -> bool {
    a.0.to_bits() == b.0.to_bits() && a.1.to_bits() == b.1.to_bits() && a.2.to_bits() == b.2.to_bits()
}

fn same_loop(a: &[V3], b: &[V3]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| same3(*x, *y))
}

fn faces(v: &Value) -> Vec<Face> {
    arr(v)
        .iter()
        .map(|f| {
            let f = arr(f);
            (vs(&f[0]), arr(&f[1]).iter().map(vs).collect())
        })
        .collect()
}

fn same_faces(a: &[Face], b: &[Face]) -> bool {
    a.len() == b.len()
        && a.iter().zip(b).all(|(x, y)| {
            same_loop(&x.0, &y.0)
                && x.1.len() == y.1.len()
                && x.1.iter().zip(&y.1).all(|(p, q)| same_loop(p, q))
        })
}

fn tris(v: &Value) -> Option<Vec<[V3; 3]>> {
    match v {
        Value::Null => None,
        _ => Some(arr(v).iter().map(|t| {
            let t = arr(t);
            [v3(&t[0]), v3(&t[1]), v3(&t[2])]
        }).collect()),
    }
}

#[test]
fn primitives_match_python() {
    let mut seen = std::collections::HashMap::new();
    for (i, r) in rows("offset_primitives.jsonl").iter().enumerate() {
        let kind = s(r.get("k").unwrap());
        *seen.entry(kind.to_string()).or_insert(0) += 1;
        let ctx = format!("row {i} ({kind})");
        match kind {
            "line" => {
                let out = offset::offset_line_intersection(
                    v3(r.get("p0").unwrap()), v3(r.get("u").unwrap()),
                    v3(r.get("p1").unwrap()), v3(r.get("v").unwrap()), v3(r.get("n").unwrap()));
                let want = r.get("out").unwrap();
                match (out, want) {
                    (None, Value::Null) => {}
                    (Some(o), w) if !matches!(w, Value::Null) => assert!(same3(o, v3(w)), "{ctx}"),
                    _ => panic!("{ctx}"),
                }
            }
            "basis" => {
                let (u, w) = offset::plane_basis(v3(r.get("n").unwrap()), H);
                assert!(same3(u, v3(r.get("u").unwrap())), "{ctx} u");
                assert!(same3(w, v3(r.get("w").unwrap())), "{ctx} w");
            }
            "pip" => {
                let poly: Vec<(f64, f64)> = arr(r.get("poly").unwrap()).iter()
                    .map(|p| (f64b(&arr(p)[0]), f64b(&arr(p)[1]))).collect();
                let p = arr(r.get("p").unwrap());
                let got = offset::point_in_polygon((f64b(&p[0]), f64b(&p[1])), &poly);
                assert_eq!(got, b(r.get("out").unwrap()), "{ctx}");
            }
            "dist" => {
                let got = offset::dist_to_boundary(v3(r.get("p").unwrap()), &vs(r.get("loop").unwrap()), H);
                assert_eq!(got.to_bits(), f64b(r.get("out").unwrap()).to_bits(), "{ctx}");
            }
            "inside" => {
                let got = offset::inside(v3(r.get("p").unwrap()), &vs(r.get("loop").unwrap()),
                                         v3(r.get("n").unwrap()), H);
                assert_eq!(got, b(r.get("out").unwrap()), "{ctx}");
            }
            "interior" => {
                let lp = vs(r.get("loop").unwrap());
                let n = v3(r.get("n").unwrap());
                let recs = arr(r.get("tri").unwrap());
                let mut next = 0;
                let got = offset::interior_point(&lp, n, |l, nn| {
                    let rec = &recs[next];
                    next += 1;
                    assert!(same_loop(l, &vs(rec.get("loop").unwrap())), "{ctx} loop");
                    assert!(same3(nn, v3(rec.get("normal").unwrap())), "{ctx} normal");
                    tris(rec.get("tris").unwrap())
                });
                assert_eq!(next, recs.len(), "{ctx}: triangulate calls");
                match (got, r.get("out").unwrap()) {
                    (None, Value::Null) => {}
                    (Some(g), w) if !matches!(w, Value::Null) => assert!(same3(g, v3(w)), "{ctx}"),
                    _ => panic!("{ctx}"),
                }
            }
            k => panic!("{k}"),
        }
    }
    eprintln!("primitives: {seen:?}");
}

#[test]
fn offset_segments_match_python() {
    let all = rows("offset_segments.jsonl");
    for (i, r) in all.iter().enumerate() {
        let got = offset::offset_segments(
            &vs(r.get("loop").unwrap()), v3(r.get("n").unwrap()), f64b(r.get("d").unwrap()), H);
        let want = arr(r.get("out").unwrap());
        assert_eq!(got.len(), want.len(), "row {i}");
        for (g, w) in got.iter().zip(want) {
            let w = arr(w);
            assert!(same3(g.0, v3(&w[0])) && same3(g.1, v3(&w[1])), "row {i}");
        }
    }
    eprintln!("offset_segments: {} cases", all.len());
}

#[test]
fn offset_chain_match_python() {
    let all = rows("offset_chain.jsonl");
    for (i, r) in all.iter().enumerate() {
        let got = offset::offset_chain(
            &vs(r.get("pts").unwrap()), v3(r.get("n").unwrap()), f64b(r.get("d").unwrap()), H);
        match (got, r.get("out").unwrap()) {
            (None, Value::Null) => {}
            (Some(g), w) if !matches!(w, Value::Null) => {
                assert!(same_loop(&g, &vs(w)), "row {i}");
            }
            _ => panic!("row {i}: None vs Some"),
        }
    }
    eprintln!("offset_chain: {} cases", all.len());
}

#[test]
fn offset_regions_match_python() {
    let all = rows("offset_regions.jsonl");
    for (i, r) in all.iter().enumerate() {
        let lp = vs(r.get("loop").unwrap());
        let arr_rec = r.get("arr").unwrap();
        let tri_recs = arr(r.get("tri").unwrap());
        let next_tri = RefCell::new(0usize);
        let arranged = RefCell::new(0usize);
        let got = offset::offset_regions(
            &lp, v3(r.get("n").unwrap()), f64b(r.get("d").unwrap()), H,
            |segs, origin, normal| {
                *arranged.borrow_mut() += 1;
                assert!(!matches!(arr_rec, Value::Null), "row {i}: arrangement called");
                let want: Vec<&Value> = arr(arr_rec.get("segs").unwrap()).iter().collect();
                assert_eq!(segs.len(), want.len(), "row {i}: segments");
                for (g, w) in segs.iter().zip(want) {
                    let w = arr(w);
                    assert!(same3(g.0, v3(&w[0])) && same3(g.1, v3(&w[1])), "row {i}: segment");
                }
                assert!(same3(origin, v3(arr_rec.get("origin").unwrap())), "row {i}: origin");
                assert!(same3(normal, v3(arr_rec.get("normal").unwrap())), "row {i}: normal");
                match arr_rec.get("faces").unwrap() {
                    Value::Null => None,
                    f => Some(faces(f)),
                }
            },
            |l, n| {
                let rec = &tri_recs[*next_tri.borrow()];
                *next_tri.borrow_mut() += 1;
                assert!(same_loop(l, &vs(rec.get("loop").unwrap())), "row {i}: tri loop");
                assert!(same3(n, v3(rec.get("normal").unwrap())), "row {i}: tri normal");
                tris(rec.get("tris").unwrap())
            },
        );
        assert_eq!(*arranged.borrow(), usize::from(!matches!(arr_rec, Value::Null)), "row {i}");
        assert_eq!(*next_tri.borrow(), tri_recs.len(), "row {i}: triangulate calls");
        assert!(same_faces(&got, &faces(r.get("out").unwrap())), "row {i}: regions");
    }
    eprintln!("offset_regions: {} cases", all.len());
}

fn controls_of(v: &Value) -> Vec<Control> {
    arr(v).iter().map(|c| {
        let c = arr(c);
        (f64b(&c[0]), f64b(&c[1]), b(&c[2]))
    }).collect()
}

fn points_of(v: &Value) -> Vec<(f64, f64)> {
    arr(v).iter().map(|p| (f64b(&arr(p)[0]), f64b(&arr(p)[1]))).collect()
}

fn same_points(a: &[(f64, f64)], b: &[(f64, f64)]) -> bool {
    a.len() == b.len()
        && a.iter().zip(b).all(|(p, q)| p.0.to_bits() == q.0.to_bits() && p.1.to_bits() == q.1.to_bits())
}

fn same_controls(a: &[Control], b: &[Control]) -> bool {
    a.len() == b.len()
        && a.iter().zip(b).all(|(p, q)| {
            p.0.to_bits() == q.0.to_bits() && p.1.to_bits() == q.1.to_bits() && p.2 == q.2
        })
}

#[test]
fn bevel_profiles_match_python() {
    let all = rows("bevel_profiles.jsonl");
    let mut seen = std::collections::HashMap::new();
    for (i, r) in all.iter().enumerate() {
        let kind = s(r.get("k").unwrap());
        *seen.entry(kind.to_string()).or_insert(0) += 1;
        let ctx = format!("row {i} ({kind})");
        let shape_owned;
        let shape = |r: &Value, store: &mut Vec<Control>| -> bool {
            let sh = r.get("shape").unwrap();
            if let Some(Value::Str(_)) = sh.get("name") {
                false
            } else {
                *store = controls_of(sh.get("controls").unwrap());
                true
            }
        };
        match kind {
            "steps_count" => {
                let sg = r.get("s").unwrap().as_i64().unwrap();
                assert_eq!(bp::steps_count(sg), r.get("out").unwrap().as_i64().unwrap(), "{ctx}");
            }
            "valid" => {
                let c = controls_of(r.get("c").unwrap());
                assert_eq!(bp::valid(&c), b(r.get("out").unwrap()), "{ctx}");
            }
            "hypot" => {
                let got = bp::py_hypot2(f64b(r.get("x").unwrap()), f64b(r.get("y").unwrap()));
                assert_eq!(got.to_bits(), f64b(r.get("out").unwrap()).to_bits(), "{ctx}");
            }
            "mirrored" => {
                let got = bp::mirrored(&points_of(r.get("pts").unwrap()));
                assert!(same_points(&got, &points_of(r.get("out").unwrap())), "{ctx}");
            }
            "sample" | "controls" => {
                let sg = r.get("s").unwrap().as_i64().unwrap();
                let mut store = Vec::new();
                let drawn = shape(r, &mut store);
                shape_owned = r.get("shape").unwrap().get("name").map(|n| s(n).to_string());
                let sh = if drawn {
                    Shape::Controls(&store)
                } else {
                    Shape::Name(shape_owned.as_deref().unwrap())
                };
                if kind == "sample" {
                    match (bp::sample(sh, sg), r.get("out").unwrap()) {
                        (Err(_), Value::Null) => {}
                        (Ok(p), w) if !matches!(w, Value::Null) => {
                            assert!(same_points(&p, &points_of(w)), "{ctx}");
                        }
                        _ => panic!("{ctx}: error mismatch"),
                    }
                } else {
                    let got = bp::controls(sh, sg);
                    assert!(same_controls(&got, &controls_of(r.get("out").unwrap())), "{ctx}");
                }
            }
            k => panic!("{k}"),
        }
    }
    eprintln!("bevel_profiles: {} rows {seen:?}", all.len());
}
