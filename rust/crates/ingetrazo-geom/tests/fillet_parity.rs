// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! Fillet and vertex-bevel planning against what the Python planners made
//! (scripts/golden/make_fillet_fixtures.py): every point of every loop,
//! every index, every refusal message, exactly.

use std::collections::HashMap;
use std::path::Path;

use ingetrazo_formats::pyjson::{loads, Value};
use ingetrazo_geom::fillet::{
    plan_fillet, profile_sample, py_hypot2, Corners, FilletPlan, Measure, Shape, TopoEdge,
    TopoFace, TopoMesh, TopoVertex,
};
use ingetrazo_geom::vertex_bevel::plan_vertex_bevel;
use ingetrazo_math::{Hypot, V3};

/// The fixtures were written on Windows: MSVC's hypot.
const H: Hypot = Hypot::Msvc;

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

fn int(v: &Value) -> usize {
    v.as_i64().unwrap() as usize
}

fn f32h(h: &str) -> f32 {
    f32::from_bits(u32::from_str_radix(h, 16).unwrap())
}

fn f64h(h: &str) -> f64 {
    f64::from_bits(u64::from_str_radix(h, 16).unwrap())
}

fn pt(h: &str) -> V3 {
    V3(f32h(&h[0..8]), f32h(&h[8..16]), f32h(&h[16..24]))
}

fn ptv(p: V3) -> Value {
    Value::Str(format!("{:08x}{:08x}{:08x}", p.0.to_bits(), p.1.to_bits(), p.2.to_bits()))
}

fn loopv(l: &[V3]) -> Value {
    Value::Array(l.iter().map(|&p| ptv(p)).collect())
}

fn mesh_of(row: &Value) -> TopoMesh {
    let ve = arr(row.get("ve").unwrap());
    let vertices = arr(row.get("v").unwrap())
        .iter()
        .zip(ve.iter())
        .map(|(p, es)| TopoVertex { position: pt(s(p)), edges: arr(es).iter().map(int).collect() })
        .collect();
    let edges = arr(row.get("e").unwrap())
        .iter()
        .map(|e| {
            let e = arr(e);
            TopoEdge { v0: int(&e[0]), v1: int(&e[1]), faces: arr(&e[2]).iter().map(int).collect() }
        })
        .collect();
    let faces = arr(row.get("f").unwrap())
        .iter()
        .map(|f| TopoFace {
            loop_: arr(f.get("l").unwrap()).iter().map(int).collect(),
            holes: arr(f.get("h").unwrap()).iter().map(|h| arr(h).iter().map(int).collect()).collect(),
            paint: arr(f.get("p").unwrap())
                .iter()
                .map(|kv| (s(&arr(kv)[0]).to_string(), s(&arr(kv)[1]).to_string()))
                .collect(),
        })
        .collect();
    TopoMesh { vertices, edges, faces }
}

fn shape_of(v: &Value) -> Option<Shape> {
    match v {
        Value::Null => None,
        Value::Str(x) => Shape::parse(x),
        Value::Array(a) => Some(Shape::Custom(
            a.iter()
                .map(|c| {
                    let c = arr(c);
                    (f64h(s(&c[0])), f64h(s(&c[1])), c[2] == Value::Bool(true))
                })
                .collect(),
        )),
        _ => panic!("{v:?}"),
    }
}

fn plan_value(p: &FilletPlan) -> Value {
    let ints = |v: &[usize]| Value::Array(v.iter().map(|&i| Value::int(i as i64)).collect());
    let loops = |v: &[Vec<V3>]| Value::Array(v.iter().map(|l| loopv(l)).collect());
    let strip_attrs = match &p.strip_attrs {
        None => Value::Null,
        Some(a) => Value::Array(
            a.iter().map(|(k, v)| Value::Array(vec![Value::Str(k.clone()), Value::Str(v.clone())])).collect(),
        ),
    };
    Value::Object(vec![
        ("removed_faces".into(), ints(&p.removed_faces)),
        ("removed_edges".into(), ints(&p.removed_edges)),
        (
            "rebuilt".into(),
            Value::Array(
                p.rebuilt
                    .iter()
                    .map(|r| {
                        Value::Array(vec![
                            Value::int(r.face as i64),
                            loopv(&r.loop_),
                            Value::Array(r.holes.iter().map(|h| loopv(h)).collect()),
                        ])
                    })
                    .collect(),
            ),
        ),
        ("strips".into(), loops(&p.strips)),
        ("patches".into(), loops(&p.patches)),
        (
            "soft".into(),
            Value::Array(p.soft.iter().map(|&(a, b)| Value::Array(vec![ptv(a), ptv(b)])).collect()),
        ),
        ("curves".into(), loops(&p.curves)),
        ("strip_attrs".into(), strip_attrs),
        (
            "flags".into(),
            Value::Array(
                p.flags.iter().map(|&(a, b, e)| Value::Array(vec![ptv(a), ptv(b), Value::int(e as i64)])).collect(),
            ),
        ),
    ])
}

fn first_diff(a: &Value, b: &Value, path: &mut String) -> bool {
    match (a, b) {
        (Value::Array(x), Value::Array(y)) => {
            if x.len() != y.len() {
                path.push_str(&format!("[len {} vs {}]", x.len(), y.len()));
                return true;
            }
            for (i, (p, q)) in x.iter().zip(y.iter()).enumerate() {
                path.push_str(&format!("[{i}]"));
                if first_diff(p, q, path) {
                    return true;
                }
                path.truncate(path.rfind('[').unwrap());
            }
            false
        }
        (Value::Object(x), Value::Object(y)) => {
            for ((k, p), (_, q)) in x.iter().zip(y.iter()) {
                path.push_str(&format!(".{k}"));
                if first_diff(p, q, path) {
                    return true;
                }
                path.truncate(path.rfind('.').unwrap());
            }
            false
        }
        _ => {
            if a != b {
                path.push_str(&format!(" {a:?} vs {b:?}"));
                true
            } else {
                false
            }
        }
    }
}

fn check(i: usize, kind: &str, got: Value, want: &Value) {
    if &got != want {
        let mut path = String::new();
        first_diff(&got, want, &mut path);
        panic!("{kind} case {i} differs at {path}");
    }
}

fn rows() -> Vec<Value> {
    let p = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/fillet.jsonl");
    let text = std::fs::read_to_string(p).unwrap();
    text.lines().skip(1).map(|l| loads(l).unwrap()).collect()
}

#[test]
fn plans_match_python_bit_for_bit() {
    let mut meshes: HashMap<usize, TopoMesh> = HashMap::new();
    let (mut n_fillet, mut n_bevel, mut n_ok, mut n_err) = (0, 0, 0, 0);
    for (i, row) in rows().iter().enumerate() {
        match s(row.get("t").unwrap()) {
            "mesh" => {
                meshes.insert(int(row.get("id").unwrap()), mesh_of(row));
            }
            "fillet" => {
                n_fillet += 1;
                let m = &meshes[&int(row.get("mesh").unwrap())];
                let edges: Vec<usize> = arr(row.get("edges").unwrap()).iter().map(int).collect();
                let shape = shape_of(row.get("shape").unwrap());
                let res = plan_fillet(
                    m,
                    &edges,
                    f64h(s(row.get("radius").unwrap())),
                    row.get("segments").unwrap().as_i64().unwrap(),
                    f64h(s(row.get("profile").unwrap())),
                    Measure::parse(s(row.get("measure").unwrap())),
                    Corners::parse(s(row.get("corners").unwrap())),
                    shape.as_ref(),
                    H,
                );
                let want = row.get("out").unwrap();
                match res {
                    Ok(p) => {
                        n_ok += 1;
                        check(i, "fillet", Value::Object(vec![("ok".into(), plan_value(&p))]), want);
                    }
                    Err(e) => {
                        n_err += 1;
                        check(i, "fillet", Value::Object(vec![("err".into(), Value::Str(e.message()))]), want);
                    }
                }
            }
            "bevel" => {
                n_bevel += 1;
                let m = &meshes[&int(row.get("mesh").unwrap())];
                let verts: Vec<usize> = arr(row.get("verts").unwrap()).iter().map(int).collect();
                let res = plan_vertex_bevel(
                    m,
                    &verts,
                    f64h(s(row.get("size").unwrap())),
                    row.get("segments").unwrap().as_i64().unwrap(),
                    f64h(s(row.get("profile").unwrap())),
                    H,
                );
                let want = row.get("out").unwrap();
                match res {
                    Ok(p) => {
                        n_ok += 1;
                        check(i, "bevel", Value::Object(vec![("ok".into(), plan_value(&p))]), want);
                    }
                    Err(e) => {
                        n_err += 1;
                        check(i, "bevel", Value::Object(vec![("err".into(), Value::Str(e.message()))]), want);
                    }
                }
            }
            _ => {}
        }
    }
    eprintln!("fillet {n_fillet}, bevel {n_bevel}: {n_ok} plans, {n_err} refusals");
    assert!(n_fillet > 200 && n_bevel > 80 && n_ok > 200);
}

#[test]
fn hypot_and_profile_samples_match_python() {
    let (mut n_h, mut n_s) = (0, 0);
    for row in rows() {
        match s(row.get("t").unwrap()) {
            "hypot" => {
                n_h += 1;
                let got = py_hypot2(f64h(s(row.get("x").unwrap())), f64h(s(row.get("y").unwrap())));
                assert_eq!(got.to_bits(), f64h(s(row.get("r").unwrap())).to_bits(), "hypot {n_h}");
            }
            "sample" => {
                n_s += 1;
                let shape = shape_of(row.get("shape").unwrap()).unwrap();
                let got = profile_sample(&shape, row.get("segments").unwrap().as_i64().unwrap());
                let want = arr(row.get("pts").unwrap());
                assert_eq!(got.len(), want.len(), "sample {n_s}");
                for (g, w) in got.iter().zip(want.iter()) {
                    let w = arr(w);
                    assert_eq!(g.0.to_bits(), f64h(s(&w[0])).to_bits(), "sample {n_s}");
                    assert_eq!(g.1.to_bits(), f64h(s(&w[1])).to_bits(), "sample {n_s}");
                }
            }
            _ => {}
        }
    }
    assert!(n_h >= 400 && n_s >= 40);
}
