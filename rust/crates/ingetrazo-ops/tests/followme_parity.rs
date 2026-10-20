// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! Follow Me against the Python app (scripts/golden/make_followme_fixtures.py):
//! 83 cases of the REAL `tools.followme.FollowMeTool` over straight and bent
//! paths, a profile standing mid-edge, closed loops, lathes, profiles with
//! holes, near-collinear and declined paths, curved sweeps (softening), a boss
//! and tunnels over existing geometry, a profile inside an open group and
//! random profiles on random planes.
//!
//! For the selected-path flows (`edges`, `face`) the path read from the
//! selection and the path the profile orients it into must be Python's bits;
//! for the dragged flow the path `_commit` was handed is replayed. Then the
//! sweep runs as one history step and the active mesh's whole ordered state
//! after the commit, after the undo, after the redo and after a second undo
//! is compared (positions as f32 bits, lists in order, flags, links, attrs).
//!
//! The fixtures come from a Windows machine: `length()` is MSVC's `hypot`.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use ingetrazo_formats::igz;
use ingetrazo_formats::pyjson::{loads, Value};
use ingetrazo_golden::sig::{canon_dumps, sha, Files};
use ingetrazo_math::{Hypot, V3};
use ingetrazo_model::mesh::{EdgeId, FaceId, Mesh};
use ingetrazo_model::scene::{MeshId, Scene};
use ingetrazo_ops::command::{Cx, History};
use ingetrazo_ops::followme::{commit_follow_me, orient_path, path_of_edges, path_of_face, profile_of};

const H: Hypot = Hypot::Msvc;

fn data() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/followme")
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

fn pbits(p: V3) -> [String; 3] {
    [bits32(p.0), bits32(p.1), bits32(p.2)]
}

fn want_path(v: &Value) -> Vec<[String; 3]> {
    arr(v)
        .iter()
        .map(|p| {
            let a = arr(p);
            [text(&a[0]).to_string(), text(&a[1]).to_string(), text(&a[2]).to_string()]
        })
        .collect()
}

fn idx(v: &Value) -> usize {
    v.as_i64().unwrap() as usize
}

#[test]
fn follow_me_matches_python() {
    let cases = loads(&std::fs::read_to_string(data().join("cases.json")).unwrap()).unwrap();
    let (mut swept, mut declined, mut no_commit, mut soft, mut paths) = (0, 0, 0, 0, 0);
    for (n, case) in arr(&cases).iter().enumerate() {
        let name = text(case.get("scene").unwrap());
        let flow = text(case.get("flow").unwrap());
        let edit: Option<usize> = match case.get("edit").unwrap() {
            Value::Null => None,
            v => Some(idx(v)),
        };
        let label = format!("case {n} {name} {flow} edit={edit:?}");
        let mut scene = open(name);
        let active = match edit {
            Some(g) => scene.groups[g].mesh,
            None => MeshId(0),
        };
        if let Some(d) = first_difference("pre", &mesh_state(scene.mesh(active)), case.get("pre").unwrap()) {
            panic!("{label}: reopened scene differs: {d}");
        }
        let spec = case.get("spec").unwrap();
        let faces: Vec<FaceId> = scene.mesh(active).faces.clone();
        let edges: Vec<EdgeId> = scene.mesh(active).edges.clone();
        let profile_ix = idx(spec.get("profile").unwrap());

        // The path the tool read from the selection, and the one it commits.
        let commits = arr(case.get("commits").unwrap());
        let mut replay: Option<(Vec<V3>, bool)> = None;
        if flow != "drag" {
            let mesh = scene.mesh(active);
            let got = if flow == "edges" {
                let sel: Vec<EdgeId> = arr(spec.get("edges").unwrap()).iter().map(|i| edges[idx(i)]).collect();
                path_of_edges(mesh, &sel)
            } else {
                Some(path_of_face(mesh, faces[idx(spec.get("path_face").unwrap())]))
            };
            match (case.get("path0").unwrap(), &got) {
                (Value::Null, None) => {}
                (Value::Null, Some(_)) => panic!("{label}: Python found no path, Rust did"),
                (_, None) => panic!("{label}: Python read a path, Rust found none"),
                (w, Some((pts, closed))) => {
                    let w = arr(w);
                    assert_eq!(pts.iter().map(|&p| pbits(p)).collect::<Vec<_>>(), want_path(&w[0]), "{label}: selected path");
                    assert_eq!(*closed, matches!(w[1], Value::Bool(true)), "{label}: selected path closed");
                    paths += 1;
                }
            }
            if let Some((pts, closed)) = got {
                // Clicking the profile commits at once.
                let prof = profile_of(mesh, faces[profile_ix]);
                replay = Some(orient_path(&pts, closed, &prof, H));
            }
        }
        if commits.is_empty() {
            if flow != "drag" {
                // A flow that ends without a commit never gets a path.
                assert!(replay.is_none(), "{label}: Rust has a path to commit, Python did not commit");
            }
            assert!(case.get("undone").is_none(), "{label}: nothing committed");
            if let Some(d) = first_difference("post", &mesh_state(scene.mesh(active)), case.get("post").unwrap()) {
                panic!("{label}: untouched scene differs: {d}");
            }
            no_commit += 1;
            continue;
        }
        let commit = &commits[0];
        let py_path = want_path(commit.get("path").unwrap());
        let py_closed = matches!(commit.get("closed").unwrap(), Value::Bool(true));
        let face = faces[idx(commit.get("face").unwrap())];
        assert_eq!(idx(commit.get("face").unwrap()), profile_ix, "{label}: profile");
        let (path, closed) = match replay {
            Some((p, c)) => {
                assert_eq!(p.iter().map(|&q| pbits(q)).collect::<Vec<_>>(), py_path, "{label}: oriented path");
                assert_eq!(c, py_closed, "{label}: oriented closed");
                (p, c)
            }
            None => {
                let f = |s: &String| f32::from_bits(u32::from_str_radix(s, 16).unwrap());
                let pts = py_path.iter().map(|p| V3(f(&p[0]), f(&p[1]), f(&p[2]))).collect();
                (pts, py_closed)
            }
        };

        let mut cx = Cx { hypot: H, mesh: active, ..Cx::default() };
        let mut hist = History::new();
        let ok = commit_follow_me(&mut scene, &mut hist, &mut cx, face, path, closed);
        let sweeps = arr(case.get("sweeps").unwrap());
        assert_eq!(sweeps.len(), 1, "{label}: one sweep call");
        assert_eq!(ok, matches!(sweeps[0], Value::Bool(true)), "{label}: sweep accepted");
        assert_eq!(hist.undo_len(), idx(case.get("undo_len").unwrap()), "{label}: undo depth");
        assert_eq!(hist.redo_len(), idx(case.get("redo_len").unwrap()), "{label}: redo depth");
        if let Some(d) = first_difference("post", &mesh_state(scene.mesh(active)), case.get("post").unwrap()) {
            panic!("{label}: after the commit differs: {d}");
        }
        if ok {
            swept += 1;
            if scene.mesh(active).edges.iter().any(|&e| scene.mesh(active).edge(e).soft) {
                soft += 1;
            }
            for (step, key) in [("undo", "undone"), ("redo", "redone"), ("undo", "undone2")] {
                let did = if step == "undo" { hist.undo(&mut scene, &mut cx) } else { hist.redo(&mut scene, &mut cx) };
                assert!(did, "{label}: nothing to {step}");
                if let Some(d) = first_difference(key, &mesh_state(scene.mesh(active)), case.get(key).unwrap()) {
                    panic!("{label}: after the {step} differs: {d}");
                }
            }
        } else {
            declined += 1;
            assert!(case.get("undone").is_none(), "{label}: a declined sweep leaves no step");
        }
    }
    assert!(swept >= 70 && declined >= 2 && no_commit >= 6 && soft >= 30 && paths >= 50, "{swept} swept, {declined} declined, {no_commit} not committed, {soft} soft, {paths} paths");
}
