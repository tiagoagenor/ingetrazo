// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! Solid booleans against `core.solids` (scripts/golden/make_solids_fixtures.py):
//! pairs and triples of closed solids go through the Rust operations and
//! must give what the real Python ones gave, with the native engine off:
//! the same groups removed, the same result groups, and for each the L2
//! mesh signature and the whole mesh state in order (vertices, edges with
//! their soft flag and faces, faces with their loops and attrs hash).
//!
//! The fixtures come from a Windows machine: `length()` is MSVC's `hypot`.

#![cfg(feature = "manifold")]

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use ingetrazo_formats::igz;
use ingetrazo_formats::pyjson::{loads, Value};
use ingetrazo_golden::model_sig::mesh_signature;
use ingetrazo_golden::sig::{canon_dumps, sha, Files};
use ingetrazo_math::Hypot;
use ingetrazo_model::mesh::Mesh;
use ingetrazo_model::scene::Scene;
use ingetrazo_ops::solids::{self, Op, SolidError};

const H: Hypot = Hypot::Msvc;

fn data() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/solids")
}

fn open(name: &str) -> Scene {
    let raw = std::fs::read(data().join(format!("{name}.igz.json"))).unwrap();
    let doc = igz::read_document(&raw).unwrap();
    Scene::from_document(&doc).unwrap_or_else(|e| panic!("{name}: {e:?}"))
}

fn bits32(x: f32) -> String {
    format!("{:08x}", x.to_bits())
}

fn uint(v: &Value) -> usize {
    v.as_i64().unwrap() as usize
}

fn arr(v: &Value) -> &[Value] {
    match v {
        Value::Array(a) => a,
        other => panic!("not an array: {other:?}"),
    }
}

fn text(v: &Value) -> &str {
    match v {
        Value::Str(s) => s,
        other => panic!("not a string: {other:?}"),
    }
}

/// The state Python recorded (`mesh_state`), from the Rust mesh, as JSON.
fn mesh_state(m: &Mesh) -> Value {
    let vi: HashMap<_, usize> = m.vertices.iter().enumerate().map(|(i, &v)| (v, i)).collect();
    let fi: HashMap<_, usize> = m.faces.iter().enumerate().map(|(i, &f)| (f, i)).collect();
    let int = |i: usize| Value::int(i as i64);
    let s = |x: String| Value::Str(x);
    let files = Files { zip: None };
    let v = m
        .vertices
        .iter()
        .map(|&v| {
            let p = m.position(v);
            Value::Array(vec![s(bits32(p.0)), s(bits32(p.1)), s(bits32(p.2))])
        })
        .collect();
    let e = m
        .edges
        .iter()
        .map(|&e| {
            let ed = m.edge(e);
            Value::Array(vec![int(vi[&ed.v0]), int(vi[&ed.v1]), int(ed.soft as usize), Value::Array(ed.faces.iter().map(|f| int(fi[f])).collect())])
        })
        .collect();
    let f = m
        .faces
        .iter()
        .map(|&f| {
            let fc = m.face(f);
            let attrs = if fc.attrs.is_empty() { String::new() } else { sha(&canon_dumps(&files.canonical(&Value::Object(fc.attrs.clone()))), 16) };
            Value::Array(vec![
                Value::Array(fc.outer.iter().map(|v| int(vi[v])).collect()),
                Value::Array(fc.holes.iter().map(|h| Value::Array(h.iter().map(|v| int(vi[v])).collect())).collect()),
                int(fc.interior as usize),
                s(attrs),
            ])
        })
        .collect();
    Value::Object(vec![("v".into(), Value::Array(v)), ("e".into(), Value::Array(e)), ("f".into(), Value::Array(f))])
}

fn first_difference(path: &str, a: &Value, b: &Value) -> Option<String> {
    match (a, b) {
        (Value::Array(x), Value::Array(y)) => {
            for (i, (p, q)) in x.iter().zip(y).enumerate() {
                if let Some(d) = first_difference(&format!("{path}[{i}]"), p, q) {
                    return Some(d);
                }
            }
            (x.len() != y.len()).then(|| format!("{path}: {} entries, Python {}", x.len(), y.len()))
        }
        (Value::Object(x), Value::Object(y)) => {
            for (k, p) in x {
                match y.iter().find(|(kk, _)| kk == k) {
                    Some((_, q)) => {
                        if let Some(d) = first_difference(&format!("{path}.{k}"), p, q) {
                            return Some(d);
                        }
                    }
                    None => return Some(format!("{path}.{k} missing in Python")),
                }
            }
            None
        }
        _ => (a != b).then(|| format!("{path}: {a:?} vs Python {b:?}")),
    }
}

fn bits64(x: f64) -> String {
    format!("{:016x}", x.to_bits())
}

#[test]
fn solid_operations_match_python() {
    let cases = loads(&std::fs::read_to_string(data().join("cases.json")).unwrap()).unwrap();
    let mut scenes: HashMap<String, Scene> = HashMap::new();
    let mut checked = 0;
    let mut errors = 0;
    for case in arr(&cases) {
        let name = text(case.get("scene").unwrap()).to_string();
        let op_id = text(case.get("op").unwrap());
        let op = Op::from_id(op_id).unwrap();
        let picks: Vec<usize> = arr(case.get("picks").unwrap()).iter().map(uint).collect();
        let label = format!("{name} {op_id} {picks:?}");
        let scene = scenes.entry(name.clone()).or_insert_with(|| open(&name));
        let groups: Vec<_> = picks.iter().map(|&i| &scene.groups[i]).collect();

        for (g, want) in groups.iter().zip(arr(case.get("inputs").unwrap())) {
            let vol = solids::solid_volume(scene, g, H);
            assert_eq!(vol.is_some(), want.get("solid") == Some(&Value::Bool(true)), "{label}: is_solid");
            if let Some(v) = vol {
                assert_eq!(bits64(v), text(want.get("volume").unwrap()), "{label}: volume");
            }
        }

        let got = solids::run(scene, op, &groups, &|n| n.to_string(), H);
        match (got, case.get("error")) {
            (Err(e), Some(want)) => {
                assert_eq!(e.to_string(), text(want), "{label}");
                errors += 1;
            }
            (Err(e), None) => panic!("{label}: Rust raised {e}, Python did not"),
            (Ok(_), Some(want)) => panic!("{label}: Python raised {want:?}, Rust did not"),
            (Ok(res), None) => {
                let removed: Vec<usize> = arr(case.get("removed").unwrap()).iter().map(uint).collect();
                assert_eq!(res.removed, removed, "{label}: removed");
                let want = arr(case.get("created").unwrap());
                assert_eq!(res.created.len(), want.len(), "{label}: groups created");
                for (k, (g, w)) in res.created.iter().zip(want).enumerate() {
                    let at = format!("{label} group {k}");
                    assert_eq!(g.name, text(w.get("name").unwrap()), "{at}: name");
                    let layer = match w.get("layer") {
                        Some(Value::Str(s)) => Some(s.clone()),
                        _ => None,
                    };
                    assert_eq!(g.layer, layer, "{at}: layer");
                    let state = mesh_state(&g.mesh);
                    if let Some(d) = first_difference("mesh", &state, w.get("mesh").unwrap()) {
                        let sig = sha(&canon_dumps(&mesh_signature(&g.mesh, &Files { zip: None })), 32);
                        panic!("{at}: first difference {d} (L2 signature {})", if sig == text(w.get("sig").unwrap()) { "equal" } else { "differs" });
                    }
                    let sig = sha(&canon_dumps(&mesh_signature(&g.mesh, &Files { zip: None })), 32);
                    assert_eq!(sig, text(w.get("sig").unwrap()), "{at}: L2 signature");
                }
            }
        }
        checked += 1;
    }
    assert!(checked > 150 && errors > 20, "{checked} cases, {errors} errors");
}

#[test]
fn applying_an_operation_replaces_the_inputs_and_undo_restores_them() {
    let mut scene = open("overlap");
    let before: Vec<String> = scene.groups.iter().map(|g| g.name.clone()).collect();
    let applied = solids::apply(&mut scene, Op::Split, &[0, 1], &|n| n.to_string(), H).unwrap();
    let names: Vec<&str> = scene.groups.iter().map(|g| g.name.as_str()).collect();
    assert_eq!(names, ["Split", "Split", "Split"]);
    solids::undo(&mut scene, applied);
    let after: Vec<String> = scene.groups.iter().map(|g| g.name.clone()).collect();
    assert_eq!(before, after);

    // Trim keeps the cutter.
    let applied = solids::apply(&mut scene, Op::Trim, &[0, 1], &|n| n.to_string(), H).unwrap();
    let names: Vec<&str> = scene.groups.iter().map(|g| g.name.as_str()).collect();
    assert_eq!(names, ["A", "Trimmed"]);
    solids::undo(&mut scene, applied);
    assert_eq!(scene.groups.len(), 2);
}

#[test]
fn errors_carry_the_python_messages() {
    let scene = open("overlap");
    let one = [&scene.groups[0]];
    assert_eq!(solids::run(&scene, Op::Union, &one, &|n| n.to_string(), H).unwrap_err(), SolidError::TwoSolids);
    assert_eq!(SolidError::NoOverlap.to_string(), "no overlap");
}
