// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! The L2 signature of a loaded Rust `Scene` — `tests/golden/signature.py`
//! over the model instead of over the document. Equal signatures for a
//! start scene mean the Rust loader built the scene Python had.

use ingetrazo_formats::pyjson::Value;
use ingetrazo_math::M4;
use ingetrazo_model::mesh::Mesh;
use ingetrazo_model::scene::{Group, MeshId, Scene};

use crate::sig::{canon_dumps, f32hex, sha, Files};

fn s(x: impl Into<String>) -> Value {
    Value::Str(x.into())
}

fn obj(items: Vec<(&str, Value)>) -> Value {
    Value::Object(items.into_iter().map(|(k, v)| (k.to_string(), v)).collect())
}

fn strs(v: Vec<String>) -> Value {
    Value::Array(v.into_iter().map(Value::Str).collect())
}

fn vkey(m: &Mesh, v: ingetrazo_model::VertexId) -> String {
    let p = m.position(v);
    format!("{},{},{}", f32hex(p.0 as f64), f32hex(p.1 as f64), f32hex(p.2 as f64))
}

fn rotated(mut keys: Vec<String>) -> Vec<String> {
    if let Some((i, _)) = keys.iter().enumerate().min_by(|a, b| a.1.cmp(b.1)) {
        keys.rotate_left(i);
    }
    keys
}

/// `signature.mesh_signature` of a live mesh.
pub fn mesh_signature(m: &Mesh, files: &Files) -> Value {
    let mut verts: Vec<String> = m.vertices.iter().map(|&v| vkey(m, v)).collect();
    verts.sort();
    verts.dedup();
    let mut raw: Vec<(String, String, &ingetrazo_model::Edge)> = m
        .edges
        .iter()
        .map(|&e| {
            let ed = m.edge(e);
            let (a, b) = (vkey(m, ed.v0), vkey(m, ed.v1));
            if a <= b {
                (a, b, ed)
            } else {
                (b, a, ed)
            }
        })
        .collect();
    raw.sort_by(|x, y| (&x.0, &x.1).cmp(&(&y.0, &y.1)));
    let mut curves: Vec<i64> = Vec::new();
    let mut edges: Vec<(String, String, String)> = Vec::new();
    for (a, b, ed) in raw {
        let mut flags = String::new();
        if ed.soft {
            flags.push('s');
        }
        if ed.hidden {
            flags.push('h');
        }
        if let Some(l) = &ed.layer {
            if !l.is_empty() {
                flags.push('@');
                flags.push_str(l);
            }
        }
        if let Some(c) = ed.curve {
            let k = match curves.iter().position(|&x| x == c) {
                Some(k) => k,
                None => {
                    curves.push(c);
                    curves.len() - 1
                }
            };
            flags.push_str(&format!("c{k}"));
        }
        edges.push((a, b, flags));
    }
    edges.sort();
    let mut faces: Vec<Value> = Vec::new();
    let mut keyed: Vec<(Vec<String>, Vec<Vec<String>>, String, bool)> = m
        .faces
        .iter()
        .map(|&f| {
            let fc = m.face(f);
            let lp = rotated(fc.outer.iter().map(|&v| vkey(m, v)).collect());
            let mut hs: Vec<Vec<String>> = fc.holes.iter().map(|h| rotated(h.iter().map(|&v| vkey(m, v)).collect())).collect();
            hs.sort();
            let attrs = if fc.attrs.is_empty() {
                String::new()
            } else {
                sha(&canon_dumps(&files.canonical(&Value::Object(fc.attrs.clone()))), 16)
            };
            (lp, hs, attrs, fc.interior)
        })
        .collect();
    // Python sorts the lists [loop, holes, attrs(, "i")]: an interior face
    // sorts after its twin by the extra element.
    keyed.sort();
    for (lp, hs, attrs, interior) in keyed {
        let mut entry = vec![strs(lp), Value::Array(hs.into_iter().map(strs).collect()), s(attrs)];
        if interior {
            entry.push(s("i"));
        }
        faces.push(Value::Array(entry));
    }
    obj(vec![
        ("v", strs(verts)),
        ("e", Value::Array(edges.into_iter().map(|(a, b, f)| Value::Array(vec![s(a), s(b), s(f)])).collect())),
        ("f", Value::Array(faces)),
    ])
}

fn m16(m: &M4) -> Value {
    Value::Array(m.data().iter().map(|&x| s(f32hex(x as f64))).collect())
}

struct Walk<'a> {
    scene: &'a Scene,
    files: &'a Files<'a>,
    defs: Vec<(String, Value)>,
    shares: Vec<(MeshId, Vec<String>)>,
}

impl Walk<'_> {
    fn group(&mut self, g: &Group) -> Value {
        let is_comp = g.is_component();
        let mut rest: Vec<(String, Value)> = Vec::new();
        if let Some(a) = &g.axes {
            rest.push(("axes".into(), Value::Array(a.data().iter().map(|&x| Value::Float(x as f64)).collect())));
        }
        for (k, v) in [("ifc", &g.ifc), ("text3d", &g.text3d), ("exploded", &g.exploded)] {
            if let Some(v) = v {
                if !matches!(v, Value::Object(o) if o.is_empty()) {
                    rest.push((k.into(), v.clone()));
                }
            }
        }
        if let Some(b) = &g.billboard {
            rest.push(("billboard".into(), b.clone()));
        }
        if let Some(e) = &g.ext {
            if !matches!(e, Value::Object(o) if o.is_empty()) {
                rest.push(("ext".into(), e.clone()));
            }
        }
        if let Some(o) = g.explode_offset {
            rest.push(("explode_offset".into(), Value::Array(o.iter().map(|&x| Value::Float(x)).collect())));
        }
        let mut out = vec![
            ("name", s(g.name.clone())),
            ("kind", s(if is_comp { "component" } else { "group" })),
            ("m", g.xform.as_ref().map(m16).unwrap_or(Value::Null)),
            ("material", g.material.as_ref().map(|m| self.files.canonical(m)).unwrap_or(Value::Null)),
            ("layer", g.layer.clone().map(Value::Str).unwrap_or(Value::Null)),
            ("locked", Value::Bool(g.locked)),
            ("hidden", Value::Bool(g.hidden)),
            ("rest", s(sha(&canon_dumps(&self.files.canonical(&Value::Object(rest))), 16))),
        ];
        let mesh_sig = mesh_signature(self.scene.mesh(g.mesh), self.files);
        if is_comp {
            let key = sha(&canon_dumps(&mesh_sig), 16);
            if !self.defs.iter().any(|(k, _)| *k == key) {
                self.defs.push((key.clone(), mesh_sig));
            }
            out.push(("def", s(key)));
        } else {
            out.push(("mesh", mesh_sig));
        }
        let mut kids: Vec<Value> = g.children.iter().map(|c| self.group(c)).collect();
        kids.sort_by_key(canon_dumps);
        out.push(("children", Value::Array(kids)));
        let out = obj(out);
        if is_comp {
            let member = sha(&canon_dumps(&out), 16);
            match self.shares.iter_mut().find(|(m, _)| *m == g.mesh) {
                Some((_, ms)) => ms.push(member),
                None => self.shares.push((g.mesh, vec![member])),
            }
        }
        out
    }
}

/// `signature.scene_signature` of a loaded scene.
pub fn scene_signature(scene: &Scene, files: &Files) -> Value {
    let mut w = Walk { scene, files, defs: Vec::new(), shares: Vec::new() };
    let mut groups: Vec<Value> = scene.groups.iter().map(|g| w.group(g)).collect();
    groups.sort_by_key(canon_dumps);
    let listed = |items: Vec<&Value>| {
        let mut v: Vec<String> = items.into_iter().map(|x| canon_dumps(&files.canonical(x))).collect();
        v.sort();
        strs(v)
    };
    let mut shares: Vec<Vec<String>> = w
        .shares
        .into_iter()
        .map(|(_, mut ms)| {
            ms.sort();
            ms
        })
        .collect();
    shares.sort();
    let count = |n: usize| Value::int(n as i64);
    obj(vec![
        ("mesh", mesh_signature(scene.loose(), files)),
        ("groups", Value::Array(groups)),
        ("defs", Value::Object(w.defs)),
        ("shares", Value::Array(shares.into_iter().map(strs).collect())),
        ("materials", listed(scene.materials.iter().map(|(_, v)| v).collect())),
        ("layers", listed(scene.layers.iter().collect())),
        (
            "counts",
            obj(vec![
                ("dimensions", count(scene.dimensions.len())),
                ("text_labels", count(scene.text_labels.len())),
                ("guides", count(scene.guides.len())),
                ("section_planes", count(scene.section_planes.len())),
                ("image_planes", count(scene.image_planes.len())),
                ("geo_paths", count(scene.geo_paths.len())),
            ]),
        ),
    ])
}
