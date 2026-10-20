// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! Intersect Faces against the Python app (scripts/golden/make_intersect_fixtures.py):
//! 65 cases over 17 scenes (boxes, slanted planes, rotated and nested
//! component instances, faces with holes, coplanar and touching faces, a
//! crossing shorter than the weld, a lattice, random scatters; the three
//! modes; loose or inside an open group) must give, with the native engine
//! off in Python, the very segments `core.intersect.segments_for` found, in
//! its order, and the ones `_on_intersect_faces` keeps (f32 bits).
//!
//! The scene each case starts from is checked too: the mesh Rust reopens
//! from the .igz is the state Python recorded before the apply, so the rest
//! of the fixture (`post`, `undone`: the whole ordered mesh after
//! `build_add_edges` through `History` and after its undo) is the state the
//! edits planner (M3.1) must reproduce when it lands: the apply is that one
//! command, `build_add_edges(scene, &kept, true)`.
//!
//! The fixtures come from a Windows machine: `length()` is MSVC's `hypot`.

use std::path::{Path, PathBuf};

use ingetrazo_formats::igz;
use ingetrazo_formats::pyjson::{loads, Value};
use ingetrazo_golden::sig::{canon_dumps, sha, Files};
use ingetrazo_math::Hypot;
use ingetrazo_model::mesh::Mesh;
use ingetrazo_model::scene::Scene;
use ingetrazo_ops::command::{Cx, History};
use ingetrazo_ops::intersect::{apply_intersect,intersect_faces, segments_for, Entity, Mode};

const H: Hypot = Hypot::Msvc;

fn data() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/intersect")
}

fn open(name: &str) -> Scene {
    let raw = std::fs::read(data().join(format!("{name}.igz.json"))).unwrap();
    let doc = igz::read_document(&raw).unwrap();
    Scene::from_document(&doc).unwrap_or_else(|e| panic!("{name}: {e:?}"))
}

fn bits32(x: f32) -> String {
    format!("{:08x}", x.to_bits())
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

/// `mesh_state` of the script: the whole ordered state of a mesh.
fn mesh_state(m: &Mesh) -> Value {
    let vi: std::collections::HashMap<_, usize> = m.vertices.iter().enumerate().map(|(i, &v)| (v, i)).collect();
    let fi: std::collections::HashMap<_, usize> = m.faces.iter().enumerate().map(|(i, &f)| (f, i)).collect();
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
            Value::Array(vec![
                int(vi[&ed.v0]),
                int(vi[&ed.v1]),
                int(ed.soft as usize),
                int(ed.hidden as usize),
                ed.layer.clone().map(Value::Str).unwrap_or(Value::Null),
                ed.curve.map(Value::int).unwrap_or(Value::Null),
                Value::Array(ed.faces.iter().map(|f| int(fi[f])).collect()),
            ])
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

fn seg_bits(segs: &[(ingetrazo_math::V3, ingetrazo_math::V3)]) -> Vec<[[String; 3]; 2]> {
    let b = |p: ingetrazo_math::V3| [bits32(p.0), bits32(p.1), bits32(p.2)];
    segs.iter().map(|&(p, q)| [b(p), b(q)]).collect()
}

fn want_segs(v: &Value) -> Vec<[[String; 3]; 2]> {
    let pt = |v: &Value| {
        let a = arr(v);
        [text(&a[0]).to_string(), text(&a[1]).to_string(), text(&a[2]).to_string()]
    };
    arr(v)
        .iter()
        .map(|s| {
            let s = arr(s);
            [pt(&s[0]), pt(&s[1])]
        })
        .collect()
}

#[test]
fn intersect_segments_match_python() {
    let cases = loads(&std::fs::read_to_string(data().join("cases.json")).unwrap()).unwrap();
    let (mut applied, mut checked, mut with_segs, mut filtered, mut total) = (0, 0, 0, 0, 0);
    for (n, case) in arr(&cases).iter().enumerate() {
        let name = text(case.get("scene").unwrap());
        let mode_s = text(case.get("mode").unwrap());
        let mode = match mode_s {
            "model" => Mode::Model,
            "selection" => Mode::Selection,
            "context" => Mode::Context,
            other => panic!("mode {other}"),
        };
        let edit: Option<Vec<usize>> = match case.get("edit").unwrap() {
            Value::Null => None,
            v => Some(arr(v).iter().map(|i| i.as_i64().unwrap() as usize).collect()),
        };
        let label = format!("case {n} {name} {mode_s} edit={edit:?}");
        let scene = open(name);
        let active = match &edit {
            Some(p) => scene.groups[p[0]].mesh,
            None => ingetrazo_model::scene::MeshId(0),
        };
        let mesh = scene.mesh(active);

        // The scene reopens to the state Python recorded.
        if let Some(d) = first_difference("pre", &mesh_state(mesh), case.get("pre").unwrap()) {
            panic!("{label}: reopened scene differs: {d}");
        }

        let sel: Vec<Entity> = arr(case.get("sel").unwrap())
            .iter()
            .map(|s| {
                let s = arr(s);
                let i = s[1].as_i64().unwrap() as usize;
                if text(&s[0]) == "f" {
                    Entity::Face(mesh.faces[i])
                } else {
                    Entity::Group(i)
                }
            })
            .collect();

        let raw = segments_for(&scene, active, edit.as_deref(), &sel, mode, H);
        let kept = intersect_faces(&scene, active, edit.as_deref(), &sel, mode, H);
        let (want_raw, want_kept) = (want_segs(case.get("raw").unwrap()), want_segs(case.get("kept").unwrap()));
        let got_raw = seg_bits(&raw);
        if got_raw != want_raw {
            let i = got_raw.iter().zip(&want_raw).position(|(a, b)| a != b).unwrap_or(got_raw.len().min(want_raw.len()));
            panic!("{label}: raw segments differ at {i}: {} found, Python {}\n rust   {:?}\n python {:?}", got_raw.len(), want_raw.len(), got_raw.get(i), want_raw.get(i));
        }
        assert_eq!(seg_bits(&kept), want_kept, "{label}: kept segments");
        if let Some(post) = case.get("post") {
            let mut scene = scene;
            let mut cx = Cx { hypot: H, mesh: active, ..Cx::default() };
            let mut hist = History::new();
            apply_intersect(&mut scene, &mut hist, &mut cx, &kept);
            if let Some(d) = first_difference("post", &mesh_state(scene.mesh(active)), post) {
                panic!("{label}: after apply differs: {d}");
            }
            assert!(hist.undo(&mut scene, &mut cx), "{label}: nothing to undo");
            if let Some(d) = first_difference("undone", &mesh_state(scene.mesh(active)), case.get("undone").unwrap()) {
                panic!("{label}: after undo differs: {d}");
            }
            applied += 1;
        }
        checked += 1;
        total += raw.len();
        with_segs += (!raw.is_empty()) as usize;
        filtered += raw.len() - kept.len();
    }
    assert!(applied >= 50, "{applied} applied");
    assert!(checked >= 60 && with_segs >= 50 && total > 1400 && filtered >= 7, "{checked} cases, {with_segs} with crossings, {total} segments, {filtered} filtered");
}
