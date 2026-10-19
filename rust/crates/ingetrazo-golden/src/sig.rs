// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! The L2 canonical signature (golden/SCHEMA.md, section 5), computed from
//! an .igz document — what `tests/golden/signature.py` computes from the
//! live Python scene. Until the Rust model exists (M2) this is how a
//! script's start state is checked: if the document reopens, in Rust, to
//! the signature Python saw, the two apps agree on what the file holds.

use ingetrazo_formats::igz::Zip;
use ingetrazo_formats::pyjson::{dumps, Options, Value};
use sha2::{Digest, Sha256};

/// `json.dumps(x, sort_keys=True, separators=(",", ":"), allow_nan=False)`.
pub fn canon_dumps(v: &Value) -> String {
    dumps(v, &Options::compact().sort_keys(true).allow_nan(false)).unwrap_or_else(|e| format!("<{e}>"))
}

pub fn sha_hex(bytes: &[u8], n: usize) -> String {
    let d = Sha256::digest(bytes);
    let mut s = String::with_capacity(64);
    for b in d {
        s.push_str(&format!("{b:02x}"));
    }
    s.truncate(n);
    s
}

fn sha(text: &str, n: usize) -> String {
    sha_hex(text.as_bytes(), n)
}

/// The 8 hex digits of a float32 (`struct.pack(">f", x)`).
pub fn f32hex(x: f64) -> String {
    format!("{:08x}", (x as f32).to_bits())
}

fn num(v: &Value) -> f64 {
    match v {
        Value::Float(x) => *x,
        Value::Int(s) => s.parse().unwrap_or(f64::NAN),
        _ => f64::NAN,
    }
}

fn vkey(p: &Value) -> String {
    match p {
        Value::Array(c) if c.len() == 3 => format!("{},{},{}", f32hex(num(&c[0])), f32hex(num(&c[1])), f32hex(num(&c[2]))),
        _ => String::from("?"),
    }
}

fn s(x: impl Into<String>) -> Value {
    Value::Str(x.into())
}

fn obj(items: Vec<(&str, Value)>) -> Value {
    Value::Object(items.into_iter().map(|(k, v)| (k.to_string(), v)).collect())
}

fn truthy(v: Option<&Value>) -> bool {
    match v {
        None | Some(Value::Null) | Some(Value::Bool(false)) => false,
        Some(Value::Str(x)) => !x.is_empty(),
        Some(Value::Int(x)) => x != "0",
        Some(Value::Float(x)) => *x != 0.0,
        Some(Value::Array(a)) => !a.is_empty(),
        Some(Value::Object(o)) => !o.is_empty(),
        Some(Value::Bool(true)) => true,
    }
}

/// Where texture paths point: the archive the document came from.
pub struct Files<'a> {
    pub zip: Option<&'a Zip<'a>>,
}

impl Files<'_> {
    /// `signature.canonical`: a `"path"` or `"base"` naming a file becomes
    /// the SHA-256 of its bytes. An embedded texture is written as
    /// `"embed"`/`"base_embed"` (an archive member) and reopens as
    /// `"path"`/`"base"` (formats/igz.py `_unpack_textures`); anything else
    /// is looked up on disk, as Python does.
    fn canonical(&self, v: &Value) -> Value {
        match v {
            Value::Object(items) => Value::Object(
                items
                    .iter()
                    .map(|(k, x)| match (k.as_str(), x) {
                        ("path" | "base", Value::Str(p)) => (k.clone(), Value::Str(self.file_hash(p))),
                        ("embed", Value::Str(m)) => ("path".to_string(), Value::Str(self.file_hash(m))),
                        ("base_embed", Value::Str(m)) => ("base".to_string(), Value::Str(self.file_hash(m))),
                        _ => (k.clone(), self.canonical(x)),
                    })
                    .collect(),
            ),
            Value::Array(a) => Value::Array(a.iter().map(|x| self.canonical(x)).collect()),
            _ => v.clone(),
        }
    }

    fn file_hash(&self, path: &str) -> String {
        if let Some(zip) = self.zip {
            if let Ok(bytes) = zip.read(path) {
                return format!("sha256:{}", sha_hex(&bytes, 64));
            }
        }
        match std::fs::read(path) {
            Ok(bytes) => format!("sha256:{}", sha_hex(&bytes, 64)),
            Err(_) => path.to_string(),
        }
    }
}

fn rotated(mut keys: Vec<String>) -> Vec<String> {
    if let Some((i, _)) = keys.iter().enumerate().min_by(|a, b| a.1.cmp(b.1)) {
        keys.rotate_left(i);
    }
    keys
}

fn loop_keys(l: Option<&Value>) -> Vec<String> {
    match l {
        Some(Value::Array(ps)) => ps.iter().map(vkey).collect(),
        _ => Vec::new(),
    }
}

fn list_of_str(v: Vec<String>) -> Value {
    Value::Array(v.into_iter().map(Value::Str).collect())
}

/// `signature.mesh_signature` of a mesh block (`edges` + `faces`).
pub fn mesh_signature(mesh: &Value, files: &Files) -> Result<Value, String> {
    if mesh.get("geom").is_some() {
        return Err("format-3 geometry block (not read yet)".into());
    }
    let empty = Vec::new();
    let edges = match mesh.get("edges") {
        Some(Value::Array(a)) => a,
        _ => &empty,
    };
    let faces = match mesh.get("faces") {
        Some(Value::Array(a)) => a,
        _ => &empty,
    };
    let mut verts: Vec<String> = Vec::new();
    let mut raw: Vec<(String, String, &Value)> = Vec::new();
    for e in edges {
        let (a, b) = (vkey(e.get("a").unwrap_or(&Value::Null)), vkey(e.get("b").unwrap_or(&Value::Null)));
        verts.push(a.clone());
        verts.push(b.clone());
        let (a, b) = if a <= b { (a, b) } else { (b, a) };
        raw.push((a, b, e));
    }
    raw.sort_by(|x, y| (&x.0, &x.1).cmp(&(&y.0, &y.1))); // stable, as Python's sort
    let mut curves: Vec<String> = Vec::new();
    let mut out_edges: Vec<(String, String, String)> = Vec::new();
    for (a, b, e) in raw {
        let mut flags = String::new();
        if truthy(e.get("soft")) {
            flags.push('s');
        }
        if truthy(e.get("hidden")) {
            flags.push('h');
        }
        if let Some(Value::Str(layer)) = e.get("layer") {
            if !layer.is_empty() {
                flags.push('@');
                flags.push_str(layer);
            }
        }
        if let Some(c) = e.get("curve") {
            if !matches!(c, Value::Null) {
                let key = canon_dumps(c);
                let k = match curves.iter().position(|x| *x == key) {
                    Some(k) => k,
                    None => {
                        curves.push(key);
                        curves.len() - 1
                    }
                };
                flags.push_str(&format!("c{k}"));
            }
        }
        out_edges.push((a, b, flags));
    }
    out_edges.sort();
    let mut out_faces: Vec<(Vec<String>, Vec<Vec<String>>, String)> = Vec::new();
    for f in faces {
        let lp = loop_keys(f.get("vertices"));
        verts.extend(lp.iter().cloned());
        let mut holes: Vec<Vec<String>> = match f.get("holes") {
            Some(Value::Array(hs)) => hs.iter().map(|h| loop_keys(Some(h))).collect(),
            _ => Vec::new(),
        };
        for h in &holes {
            verts.extend(h.iter().cloned());
        }
        holes = holes.into_iter().map(rotated).collect();
        holes.sort();
        let attrs: Vec<(String, Value)> = match f {
            Value::Object(items) => items.iter().filter(|(k, _)| k != "vertices" && k != "holes").cloned().collect(),
            _ => Vec::new(),
        };
        let attrs = if attrs.is_empty() { String::new() } else { sha(&canon_dumps(&files.canonical(&Value::Object(attrs))), 16) };
        out_faces.push((rotated(lp), holes, attrs));
    }
    out_faces.sort();
    verts.sort();
    verts.dedup();
    Ok(obj(vec![
        ("v", list_of_str(verts)),
        (
            "e",
            Value::Array(out_edges.into_iter().map(|(a, b, f)| Value::Array(vec![s(a), s(b), s(f)])).collect()),
        ),
        (
            "f",
            Value::Array(
                out_faces
                    .into_iter()
                    .map(|(l, h, a)| Value::Array(vec![list_of_str(l), Value::Array(h.into_iter().map(list_of_str).collect()), s(a)]))
                    .collect(),
            ),
        ),
    ]))
}

const REST_KEYS: [&str; 7] = ["axes", "ifc", "text3d", "exploded", "billboard", "ext", "explode_offset"];

struct Walk<'a> {
    protos: &'a [Value],
    files: &'a Files<'a>,
    defs: Vec<(String, Value)>,
    shares: Vec<(i64, Vec<String>)>,
}

impl Walk<'_> {
    fn group(&mut self, g: &Value) -> Result<Value, String> {
        let proto = g.get("proto").and_then(Value::as_i64);
        let xform = g.get("xform");
        let is_comp = proto.is_some() && xform.is_some() && !matches!(g.get("component"), Some(Value::Bool(false)));
        let m = match (proto, xform) {
            (Some(_), Some(Value::Array(xs))) => Value::Array(xs.iter().map(|x| s(f32hex(num(x)))).collect()),
            _ => Value::Null,
        };
        let rest: Vec<(String, Value)> =
            REST_KEYS.iter().filter_map(|k| g.get(k).map(|v| (k.to_string(), v.clone()))).collect();
        let mut out = vec![
            ("name", g.get("name").cloned().unwrap_or(Value::Null)),
            ("kind", s(if is_comp { "component" } else { "group" })),
            ("m", m),
            ("material", g.get("material").map(|v| self.files.canonical(v)).unwrap_or(Value::Null)),
            ("layer", g.get("layer").cloned().unwrap_or(Value::Null)),
            ("locked", Value::Bool(truthy(g.get("locked")))),
            ("hidden", Value::Bool(truthy(g.get("hidden")))),
            ("rest", s(sha(&canon_dumps(&self.files.canonical(&Value::Object(rest))), 16))),
        ];
        let mesh_src = match proto {
            Some(i) => self.protos.get(i as usize).ok_or_else(|| format!("proto {i} missing"))?,
            None => g,
        };
        let mesh_sig = mesh_signature(mesh_src, self.files)?;
        if is_comp {
            let key = sha(&canon_dumps(&mesh_sig), 16);
            if !self.defs.iter().any(|(k, _)| *k == key) {
                self.defs.push((key.clone(), mesh_sig));
            }
            out.push(("def", s(key)));
        } else {
            out.push(("mesh", mesh_sig));
        }
        let mut kids = Vec::new();
        if let Some(Value::Array(cs)) = g.get("children") {
            for c in cs {
                kids.push(self.group(c)?);
            }
        }
        kids.sort_by_key(canon_dumps);
        out.push(("children", Value::Array(kids)));
        let out = obj(out);
        if is_comp {
            let member = sha(&canon_dumps(&out), 16);
            let p = proto.unwrap_or(-1);
            match self.shares.iter_mut().find(|(q, _)| *q == p) {
                Some((_, ms)) => ms.push(member),
                None => self.shares.push((p, vec![member])),
            }
        }
        Ok(out)
    }
}

/// `signature.scene_signature` of a whole document (`{"scene": …}`).
pub fn scene_signature(doc: &Value, files: &Files) -> Result<Value, String> {
    let scene = doc.get("scene").ok_or("no scene")?;
    let empty = Vec::new();
    let protos = match scene.get("protos") {
        Some(Value::Array(a)) => a,
        _ => &empty,
    };
    let mut walk = Walk { protos, files, defs: Vec::new(), shares: Vec::new() };
    let mut groups = Vec::new();
    if let Some(Value::Array(gs)) = scene.get("groups") {
        for g in gs {
            groups.push(walk.group(g)?);
        }
    }
    groups.sort_by_key(canon_dumps);
    let listed = |name: &str, default: Option<Value>| -> Value {
        let items: Vec<Value> = match (scene.get(name), default) {
            (Some(Value::Array(a)), _) => a.clone(),
            (_, Some(d)) => vec![d],
            _ => Vec::new(),
        };
        let mut out: Vec<String> = items.iter().map(|x| canon_dumps(&files.canonical(x))).collect();
        out.sort();
        list_of_str(out)
    };
    let count = |name: &str| -> Value {
        Value::int(match scene.get(name) {
            Some(Value::Array(a)) => a.len() as i64,
            _ => 0,
        })
    };
    let mut shares: Vec<Vec<String>> = walk
        .shares
        .into_iter()
        .map(|(_, mut ms)| {
            ms.sort();
            ms
        })
        .collect();
    shares.sort();
    Ok(obj(vec![
        ("mesh", mesh_signature(scene, files)?),
        ("groups", Value::Array(groups)),
        ("defs", Value::Object(walk.defs)),
        ("shares", Value::Array(shares.into_iter().map(list_of_str).collect())),
        ("materials", listed("materials", None)),
        ("layers", listed("layers", Some(obj(vec![("name", s("Layer 0"))])))),
        (
            "counts",
            obj(vec![
                ("dimensions", count("dimensions")),
                ("text_labels", count("text_labels")),
                ("guides", count("guides")),
                ("section_planes", count("section_planes")),
                ("image_planes", count("image_planes")),
                ("geo_paths", count("geo_paths")),
            ]),
        ),
    ]))
}

/// `signature.signature_hash`.
pub fn signature_hash(sig: &Value) -> String {
    sha(&canon_dumps(sig), 64)
}
