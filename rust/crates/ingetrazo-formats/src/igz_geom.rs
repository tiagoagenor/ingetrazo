// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! Format 3 of the .igz (`formats/igz.py`, `_pack_geometry` and
//! `_geometry_arrays`): a big mesh is not JSON lists but a binary block
//! under `geom/` — every distinct point once as float32, edges and face
//! rings as indices into them, and a table of face attributes kept as JSON.
//!
//! [`expand_geometry`] reads a block exactly as Python does (same dtypes,
//! same checks, same refusals); [`GeomMesh::to_json`] spells it out as the
//! `edges`/`faces` block `_mesh_json` writes for a small mesh, so every
//! consumer of the JSON form reads format 3 unchanged.

use crate::igz::{IgzError, Zip};
use crate::pyjson::Value;

/// The dtypes a block may declare (`igz._GEOM_DTYPES`); anything else is
/// refused.
const DTYPES: [&str; 4] = ["<f4", "<i4", "<i8", "|u1"];

/// A mesh read from a geometry block. Indices point into `positions`.
#[derive(Clone, Debug, PartialEq)]
pub struct GeomMesh {
    /// Every distinct point, float32 as the live mesh holds it.
    pub positions: Vec<[f32; 3]>,
    pub edges: Vec<GeomEdge>,
    pub faces: Vec<GeomFace>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct GeomEdge {
    pub a: usize,
    pub b: usize,
    pub soft: bool,
    pub hidden: bool,
    pub curve: Option<i64>,
    pub layer: Option<String>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct GeomFace {
    pub outer: Vec<usize>,
    pub holes: Vec<Vec<usize>>,
    /// The attribute-table row (a JSON object, `_face_attrs_json`), if any.
    pub attrs: Option<Value>,
}

fn damaged(why: &str) -> IgzError {
    IgzError::Container(format!("damaged geometry block ({why})"))
}

/// One array of the blob.
struct Arr {
    float: bool,
    shape: Vec<usize>,
    /// Integers widened to i64 (empty for a float array).
    ints: Vec<i64>,
    floats: Vec<f32>,
}

impl Arr {
    fn len(&self) -> usize {
        self.shape.first().copied().unwrap_or(1)
    }

}

fn int_of(v: &Value) -> Option<i64> {
    v.as_i64()
}

fn read_array(name: &str, spec: &Value, blob: &[u8]) -> Result<Arr, IgzError> {
    let bad = || damaged(name);
    let Value::Array(spec) = spec else { return Err(bad()) };
    let [dtype, shape, offset] = spec.as_slice() else { return Err(bad()) };
    let Value::Str(dtype) = dtype else { return Err(bad()) };
    let Value::Array(shape) = shape else { return Err(bad()) };
    let offset = int_of(offset).ok_or_else(bad)?;
    let shape: Vec<i64> = shape.iter().map(|n| int_of(n).ok_or_else(bad)).collect::<Result<_, _>>()?;
    if !DTYPES.contains(&dtype.as_str()) || offset < 0 || shape.iter().any(|&n| n < 0) {
        return Err(bad());
    }
    let shape: Vec<usize> = shape.iter().map(|&n| n as usize).collect();
    let count = shape.iter().try_fold(1usize, |a, &n| a.checked_mul(n)).ok_or_else(bad)?;
    let width = match dtype.as_str() {
        "<f4" | "<i4" => 4,
        "<i8" => 8,
        _ => 1,
    };
    let start = offset as usize;
    let end = count.checked_mul(width).and_then(|s| s.checked_add(start)).ok_or_else(bad)?;
    let bytes = blob.get(start..end).ok_or_else(bad)?;
    let mut arr = Arr { float: dtype == "<f4", shape, ints: Vec::new(), floats: Vec::new() };
    match dtype.as_str() {
        "<f4" => arr.floats = bytes.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
        "<i4" => arr.ints = bytes.as_chunks::<4>().0.iter().map(|c| i64::from(i32::from_le_bytes([c[0], c[1], c[2], c[3]]))).collect(),
        "<i8" => {
            arr.ints = bytes.as_chunks::<8>().0.iter().map(|c| i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]])).collect()
        }
        _ => arr.ints = bytes.iter().map(|&b| i64::from(b)).collect(),
    }
    Ok(arr)
}

/// An integer array that must be one-dimensional.
fn ints1<'a>(a: &'a std::collections::HashMap<String, Arr>, name: &str) -> Result<&'a [i64], IgzError> {
    let arr = a.get(name).ok_or_else(|| damaged(&format!("missing {name}")))?;
    if arr.float || arr.shape.len() != 1 {
        return Err(damaged(name));
    }
    Ok(&arr.ints)
}

/// The mesh of a `{"geom": meta}` reference, read from the archive's
/// binary member (`_document_geometry_arrays` + `_load_geometry`).
pub fn expand_geometry(meta: &Value, zip: Option<&Zip>) -> Result<GeomMesh, IgzError> {
    let member = match meta.get("member") {
        Some(Value::Str(m)) => m,
        _ => return Err(IgzError::Container("the document lost its geometry block None".into())),
    };
    let zip = zip.ok_or_else(|| IgzError::Container("a geometry block in a plain JSON file".into()))?;
    let blob = zip
        .read(member)
        .map_err(|_| IgzError::Container(format!("the document lost its geometry block {member:?}")))?;
    mesh_from_blob(meta, &blob)
}

/// `igz._geometry_arrays` and the decoding of its result.
pub fn mesh_from_blob(meta: &Value, blob: &[u8]) -> Result<GeomMesh, IgzError> {
    let Some(Value::Object(layout)) = meta.get("arrays") else { return Err(damaged("no layout")) };
    let mut a = std::collections::HashMap::new();
    for (name, spec) in layout {
        a.insert(name.clone(), read_array(name, spec, blob)?);
    }
    for name in ["pos", "edges", "rings", "ring_sizes", "ring_counts", "face_attr", "edge_flags"] {
        if !a.contains_key(name) {
            return Err(damaged(&format!("missing {name}")));
        }
    }
    let pos = &a["pos"];
    if pos.shape.len() != 2 || pos.shape[1] != 3 {
        return Err(damaged("pos"));
    }
    let positions: Vec<[f32; 3]> = if pos.float {
        pos.floats.as_chunks::<3>().0.iter().map(|c| [c[0], c[1], c[2]]).collect()
    } else {
        pos.ints.as_chunks::<3>().0.iter().map(|c| [c[0] as f32, c[1] as f32, c[2] as f32]).collect()
    };
    let nv = positions.len() as i64;
    let edges_arr = &a["edges"];
    if edges_arr.float || edges_arr.shape.len() != 2 || edges_arr.shape[1] != 2 {
        return Err(damaged("edges"));
    }
    let ne = edges_arr.len();
    let in_range = |xs: &[i64]| xs.iter().all(|&i| (0..nv).contains(&i));
    if !in_range(&edges_arr.ints) {
        return Err(damaged("edges"));
    }
    let rings = ints1(&a, "rings")?;
    if !in_range(rings) {
        return Err(damaged("rings"));
    }
    let sizes = ints1(&a, "ring_sizes")?;
    let counts = ints1(&a, "ring_counts")?;
    if sizes.iter().any(|&n| n < 1) || counts.iter().any(|&n| n < 1) {
        return Err(damaged("rings"));
    }
    if sizes.iter().sum::<i64>() != rings.len() as i64 || counts.iter().sum::<i64>() != sizes.len() as i64 {
        return Err(damaged("rings"));
    }
    let table: &[Value] = match meta.get("face_attrs") {
        Some(Value::Array(t)) => t,
        _ => &[],
    };
    let face_attr = ints1(&a, "face_attr")?;
    if face_attr.len() != counts.len() || face_attr.iter().any(|&k| k < -1 || k >= table.len() as i64) {
        return Err(damaged("face_attr"));
    }
    let layer_names: &[Value] = match meta.get("edge_layers") {
        Some(Value::Array(t)) => t,
        _ => &[],
    };
    let flags = ints1(&a, "edge_flags")?;
    let curves = match a.get("edge_curve") {
        Some(_) => Some(ints1(&a, "edge_curve")?),
        None => None,
    };
    let layers = match a.get("edge_layer") {
        Some(_) => Some(ints1(&a, "edge_layer")?),
        None => None,
    };
    if flags.len() != ne {
        return Err(damaged("edge_flags"));
    }
    if curves.is_some_and(|c| c.len() != ne) {
        return Err(damaged("edge_curve"));
    }
    if let Some(l) = layers {
        if l.len() != ne || l.iter().any(|&k| k < -1 || k >= layer_names.len() as i64) {
            return Err(damaged("edge_layer"));
        }
    }

    // Non-finite points are never written, but a damaged block opens like
    // a JSON one: the pieces touching them are left out.
    let bad: Vec<bool> = positions.iter().map(|p| !p.iter().all(|x| x.is_finite())).collect();
    let mut edges = Vec::with_capacity(ne);
    for i in 0..ne {
        let (ea, eb) = (edges_arr.ints[2 * i] as usize, edges_arr.ints[2 * i + 1] as usize);
        if bad[ea] || bad[eb] {
            continue;
        }
        let layer = match layers.map(|l| l[i]) {
            Some(k) if k >= 0 => match &layer_names[k as usize] {
                Value::Str(s) => Some(s.clone()),
                _ => return Err(damaged("edge_layers")),
            },
            _ => None,
        };
        edges.push(GeomEdge {
            a: ea,
            b: eb,
            soft: flags[i] & 1 != 0,
            hidden: flags[i] & 2 != 0,
            curve: curves.map(|c| c[i]).filter(|&c| c >= 0),
            layer,
        });
    }
    let mut faces = Vec::with_capacity(counts.len());
    let (mut at, mut ring) = (0usize, 0usize);
    for (f, &nr) in counts.iter().enumerate() {
        let mut loops: Vec<Vec<usize>> = Vec::with_capacity(nr as usize);
        for _ in 0..nr {
            let n = sizes[ring] as usize;
            loops.push(rings[at..at + n].iter().map(|&i| i as usize).collect());
            at += n;
            ring += 1;
        }
        if loops.iter().flatten().any(|&i| bad[i]) {
            continue;
        }
        let k = face_attr[f];
        let mut it = loops.into_iter();
        let outer = it.next().unwrap_or_default();
        faces.push(GeomFace { outer, holes: it.collect(), attrs: (k >= 0).then(|| table[k as usize].clone()) });
    }
    Ok(GeomMesh { positions, edges, faces })
}

impl GeomMesh {
    fn point(&self, i: usize) -> Value {
        Value::Array(self.positions[i].iter().map(|&x| Value::Float(f64::from(x))).collect())
    }

    fn ring(&self, ring: &[usize]) -> Value {
        Value::Array(ring.iter().map(|&i| self.point(i)).collect())
    }

    /// The mesh as `_mesh_json` writes a small one: `{"edges": [...],
    /// "faces": [...]}`, edges as `_edge_json` (`a`, `b`, then `soft`,
    /// `hidden`, `curve`, `layer` where set) and faces as `_face_json`
    /// (`vertices`, `holes` when present, then the attributes).
    pub fn to_json(&self) -> Value {
        let edges = self
            .edges
            .iter()
            .map(|e| {
                let mut o = vec![("a".to_string(), self.point(e.a)), ("b".to_string(), self.point(e.b))];
                if e.soft {
                    o.push(("soft".into(), Value::Bool(true)));
                }
                if e.hidden {
                    o.push(("hidden".into(), Value::Bool(true)));
                }
                if let Some(c) = e.curve {
                    o.push(("curve".into(), Value::int(c)));
                }
                if let Some(l) = &e.layer {
                    o.push(("layer".into(), Value::Str(l.clone())));
                }
                Value::Object(o)
            })
            .collect();
        let faces = self
            .faces
            .iter()
            .map(|f| {
                let mut o = vec![("vertices".to_string(), self.ring(&f.outer))];
                if !f.holes.is_empty() {
                    o.push(("holes".into(), Value::Array(f.holes.iter().map(|h| self.ring(h)).collect())));
                }
                if let Some(Value::Object(attrs)) = &f.attrs {
                    o.extend(attrs.iter().filter(|(k, _)| k != "vertices" && k != "holes").cloned());
                }
                Value::Object(o)
            })
            .collect();
        Value::Object(vec![("edges".into(), Value::Array(edges)), ("faces".into(), Value::Array(faces))])
    }
}

/// A mesh block as plain `edges`/`faces` JSON: unchanged unless it holds a
/// format-3 `"geom"` reference, which is read from `zip` and spelled out.
/// The other keys of the object (a group's name, its matrix…) are kept.
pub fn expand_mesh(mesh: &Value, zip: Option<&Zip>) -> Result<Value, IgzError> {
    let Some(geom) = mesh.get("geom") else { return Ok(mesh.clone()) };
    let Value::Object(plain) = expand_geometry(geom, zip)?.to_json() else { unreachable!() };
    let Value::Object(items) = mesh else { return Ok(mesh.clone()) };
    let mut out: Vec<(String, Value)> = items.iter().filter(|(k, _)| k != "geom").cloned().collect();
    out.extend(plain);
    Ok(Value::Object(out))
}

/// The whole document with every format-3 mesh (the scene's own, the
/// prototypes', each group's and its children's) expanded in place.
pub fn expand_document(doc: &Value, zip: Option<&Zip>) -> Result<Value, IgzError> {
    fn group(g: &Value, zip: Option<&Zip>) -> Result<Value, IgzError> {
        let Value::Object(items) = expand_mesh(g, zip)? else { return Ok(g.clone()) };
        let mut out = Vec::with_capacity(items.len());
        for (k, v) in items {
            let v = match (k.as_str(), v) {
                ("children", Value::Array(cs)) => Value::Array(cs.iter().map(|c| group(c, zip)).collect::<Result<_, _>>()?),
                (_, v) => v,
            };
            out.push((k, v));
        }
        Ok(Value::Object(out))
    }
    let Some(scene) = doc.get("scene") else { return Ok(doc.clone()) };
    let Value::Object(items) = expand_mesh(scene, zip)? else { return Ok(doc.clone()) };
    let mut new_scene = Vec::with_capacity(items.len());
    for (k, v) in items {
        let v = match (k.as_str(), v) {
            ("protos", Value::Array(ps)) => Value::Array(ps.iter().map(|p| expand_mesh(p, zip)).collect::<Result<_, _>>()?),
            ("groups", Value::Array(gs)) => Value::Array(gs.iter().map(|g| group(g, zip)).collect::<Result<_, _>>()?),
            (_, v) => v,
        };
        new_scene.push((k, v));
    }
    let Value::Object(top) = doc else { return Ok(doc.clone()) };
    Ok(Value::Object(top.iter().map(|(k, v)| if k == "scene" { (k.clone(), Value::Object(new_scene.clone())) } else { (k.clone(), v.clone()) }).collect()))
}
