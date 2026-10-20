// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! `formats/fuse.py` on the editable mesh: the fast coplanar fusion the big
//! imports use, `simplify_mesh`, and the two smoothing passes.
//!
//! `ingetrazo_geom::fuse` is the O(F) pass itself, on arrays and attribute
//! signature ids; this module is the part that reads the attribute dicts
//! (`_attrs_sig`, `_freeze_attr`, the texture groups of `_fuse_native`) and
//! builds meshes from what comes out (`formats.dae._add_fused`).
//!
//! Python equality of attribute values is reproduced by [`Fz`]: ints, floats
//! and bools that are equal compare equal, a dict is a tuple of its sorted
//! `(key, value)` pairs.

use std::collections::HashMap;

use ingetrazo_formats::pyjson::Value;
use ingetrazo_geom::fuse::{self, Region};
use ingetrazo_math::{Hypot, V3};
use ingetrazo_model::mesh::{EdgeFlags as MeshEdgeFlags, FaceId, Mesh};
use ingetrazo_model::pyval;
use ingetrazo_model::scene::{Group, MeshId, Scene};

use crate::pyutil;

/// Attributes of a loop, in Python's insertion order; empty is `None`.
pub type Attrs = Vec<(String, Value)>;

/// One polygon going in: `(points, attrs)`.
pub type RawLoop = (Vec<V3>, Attrs);

/// One region coming out: `(outer, holes, attrs, originals)` of
/// `fuse_coplanar_loops`.
#[derive(Clone, Debug)]
pub struct Fused {
    pub outer: Vec<V3>,
    pub holes: Vec<Vec<V3>>,
    pub attrs: Attrs,
    /// The source loops of the region, for a caller whose `add_face` rejects
    /// the fused polygon.
    pub originals: Vec<Vec<V3>>,
}

// ---- attribute signatures ----------------------------------------------------------

/// `_freeze_attr(value)` as something hashable with Python's equality.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum Fz {
    None,
    Num(u64),
    Big(String),
    Str(String),
    Tuple(Vec<Fz>),
}

fn num(x: f64) -> Fz {
    Fz::Num((x + 0.0).to_bits())
}

fn freeze(v: &Value) -> Fz {
    match v {
        Value::Null => Fz::None,
        Value::Bool(b) => num(f64::from(u8::from(*b))),
        Value::Int(_) => match pyval::as_int(v) {
            Some(i) if i.unsigned_abs() < (1 << 53) => num(i as f64),
            _ => match v {
                Value::Int(s) => Fz::Big(s.clone()),
                _ => unreachable!(),
            },
        },
        Value::Float(x) => num(*x),
        Value::Str(s) => Fz::Str(s.clone()),
        Value::Array(a) => Fz::Tuple(a.iter().map(freeze).collect()),
        Value::Object(items) => {
            let mut sorted: Vec<&(String, Value)> = items.iter().collect();
            sorted.sort_by(|a, b| a.0.cmp(&b.0));
            Fz::Tuple(sorted.into_iter().map(|(k, x)| Fz::Tuple(vec![Fz::Str(k.clone()), freeze(x)])).collect())
        }
    }
}

fn as_f64(v: &Value) -> Option<f64> {
    match v {
        Value::Float(x) => Some(*x),
        Value::Int(_) | Value::Bool(_) => pyval::float(v).ok(),
        _ => None,
    }
}

/// `(path, sw, sh, rot, uvw)` of `_attrs_sig`; the uvw rounded to 4 places.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct TSig {
    path: Fz,
    sw: Fz,
    sh: Fz,
    rot: Fz,
    uvw: Option<Vec<u64>>,
}

/// `_attrs_sig(attrs)`: `(color, texture sig, extras)`.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct Sig {
    color: Option<Fz>,
    tex: Option<TSig>,
    extras: Vec<(String, Fz)>,
}

fn attrs_sig(attrs: &[(String, Value)]) -> Sig {
    let get = |k: &str| attrs.iter().find(|(x, _)| x == k).map(|(_, v)| v);
    let color = match get("color") {
        None | Some(Value::Null) => None,
        Some(Value::Array(c)) => Some(Fz::Tuple(c.iter().map(freeze).collect())),
        Some(other) => Some(freeze(other)),
    };
    let tex = get("texture").filter(|t| pyval::truthy(t)).map(|t| {
        let tg = |k: &str| t.get(k);
        let uvw = match tg("uvw") {
            Some(Value::Array(u)) if !u.is_empty() => Some(u.iter().map(|x| pyutil::round4(as_f64(x).unwrap_or(f64::NAN)).to_bits()).collect()),
            _ => None,
        };
        TSig {
            path: tg("path").map_or(Fz::None, freeze),
            sw: tg("sw").map_or(Fz::None, freeze),
            sh: tg("sh").map_or(Fz::None, freeze),
            rot: tg("rot").map_or(num(0.0), freeze),
            uvw,
        }
    });
    let mut extras: Vec<(String, Fz)> = attrs.iter().filter(|(k, _)| k != "color" && k != "texture").map(|(k, v)| (k.clone(), freeze(v))).collect();
    extras.sort_by(|a, b| a.0.cmp(&b.0));
    Sig { color, tex, extras }
}

// ---- fuse_coplanar_loops ------------------------------------------------------------

/// `fuse_coplanar_loops(loops, cos_tol, principal_planes)` (the pure Python
/// path: its twin in `ingetrazo_geom::fuse` runs it op for op).
pub fn fuse_coplanar_loops(loops: &[RawLoop], cos_tol: f64, principal: bool, h: Hypot) -> Vec<Fused> {
    let mut flat: Vec<V3> = Vec::new();
    let mut starts = vec![0usize];
    let mut sigs: Vec<i64> = Vec::new();
    let mut groups: Vec<i64> = Vec::new();
    let mut ranks: Vec<u8> = Vec::new();
    let mut uvws: Vec<Vec<f64>> = Vec::new();
    let mut sig_ids: HashMap<Sig, i64> = HashMap::new();
    let mut group_ids: HashMap<(Option<Fz>, Vec<(String, Fz)>, Fz), i64> = HashMap::new();
    for (pts, attrs) in loops {
        flat.extend_from_slice(pts);
        starts.push(flat.len());
        let s = attrs_sig(attrs);
        let next = sig_ids.len() as i64;
        let id = *sig_ids.entry(s.clone()).or_insert(next);
        sigs.push(id);
        match &s.tex {
            Some(t) => {
                ranks.push(2);
                match &t.uvw {
                    Some(u) => {
                        let next = group_ids.len() as i64;
                        let g = *group_ids.entry((s.color.clone(), s.extras.clone(), t.path.clone())).or_insert(next);
                        groups.push(g);
                        uvws.push(u.iter().map(|&b| f64::from_bits(b)).collect());
                    }
                    None => {
                        groups.push(-1);
                        uvws.push(Vec::new());
                    }
                }
            }
            None => {
                ranks.push(1);
                groups.push(-1);
                uvws.push(Vec::new());
            }
        }
    }
    let inp = fuse::Input { pts: &flat, loop_start: &starts, sig: &sigs, tex_group: &groups, rank: &ranks, uvw: &uvws };
    let ring = |lp: usize| flat[starts[lp]..starts[lp + 1]].to_vec();
    let mut out = Vec::new();
    for region in fuse::fuse(&inp, cos_tol, principal, h) {
        match region {
            Region::Pass(lps) => {
                for lp in lps {
                    let pts = ring(lp);
                    out.push(Fused { outer: pts.clone(), holes: Vec::new(), attrs: loops[lp].1.clone(), originals: vec![pts] });
                }
            }
            Region::Traced(rings, lps) => {
                let mut rs: Vec<Vec<V3>> = rings.iter().map(|r| r.iter().map(|&i| flat[i]).collect()).collect();
                let outer = rs.remove(0);
                out.push(Fused { outer, holes: rs, attrs: loops[lps[0]].1.clone(), originals: lps.iter().map(|&l| ring(l)).collect() });
            }
        }
    }
    out
}

// ---- building ----------------------------------------------------------------------

fn finite(p: V3) -> bool {
    p.0.is_finite() && p.1.is_finite() && p.2.is_finite()
}

/// `mesh.add_face(outer, holes)` where Python raises on a NaN or infinite
/// corner (its weld key is an int) after welding the corners before it:
/// `None` then.
pub fn add_face_py(mesh: &mut Mesh, outer: &[V3], holes: &[Vec<V3>]) -> Option<FaceId> {
    let ok = outer.iter().all(|&c| finite(c)) && holes.iter().all(|h| h.iter().all(|&c| finite(c)));
    if ok {
        return Some(mesh.add_face(outer, holes));
    }
    for &c in outer {
        if !finite(c) {
            return None;
        }
        mesh.vertex(c);
    }
    for hole in holes {
        for &c in hole {
            if !finite(c) {
                return None;
            }
            mesh.vertex(c);
        }
    }
    None
}

fn update_attrs(mesh: &mut Mesh, f: FaceId, attrs: &[(String, Value)]) {
    for (k, v) in attrs {
        pyval::set(&mut mesh.face_mut(f).attrs, k, v.clone());
    }
}

/// `formats.dae._add_fused(mesh, fused)`: add each region, falling back to
/// its original loops when the fused polygon is rejected.
pub fn add_fused(mesh: &mut Mesh, fused: &[Fused]) {
    for item in fused {
        if let Some(f) = add_face_py(mesh, &item.outer, &item.holes) {
            update_attrs(mesh, f, &item.attrs);
            continue;
        }
        for pts in &item.originals {
            if let Some(f) = add_face_py(mesh, pts, &[]) {
                update_attrs(mesh, f, &item.attrs);
            }
        }
    }
}

// ---- smoothing --------------------------------------------------------------------

/// `_smooth_sig(attrs)`: colour and texture image only.
fn smooth_sig(attrs: &[(String, Value)]) -> Option<(Option<Fz>, Option<Fz>)> {
    if attrs.is_empty() {
        return None;
    }
    let get = |k: &str| attrs.iter().find(|(x, _)| x == k).map(|(_, v)| v);
    let color = match get("color") {
        None | Some(Value::Null) => None,
        Some(Value::Array(c)) => Some(Fz::Tuple(c.iter().map(freeze).collect())),
        Some(other) => Some(freeze(other)),
    };
    let path = get("texture").filter(|t| pyval::truthy(t)).map(|t| t.get("path").map_or(Fz::None, freeze));
    Some((color, path))
}

/// `soften_smooth_edges(mesh, cos_threshold=0.85)`: the edges between two
/// same-material faces at a shallow dihedral are soft.
pub fn soften_smooth_edges(mesh: &mut Mesh, h: Hypot) {
    let mut normals: HashMap<FaceId, V3> = HashMap::new();
    for e in mesh.edges.clone() {
        let ed = mesh.edge(e);
        if ed.faces.len() != 2 {
            continue;
        }
        let (f0, f1) = (ed.faces[0], ed.faces[1]);
        if f0 == f1 {
            continue;
        }
        if smooth_sig(&mesh.face(f0).attrs) != smooth_sig(&mesh.face(f1).attrs) {
            continue;
        }
        let n0 = *normals.entry(f0).or_insert_with(|| mesh.face_normal(f0, h));
        let n1 = *normals.entry(f1).or_insert_with(|| mesh.face_normal(f1, h));
        if (n0.dot(n1) as f64).abs() > 0.85 {
            mesh.edge_mut(e).soft = true;
        }
    }
}

// ---- simplify_mesh -----------------------------------------------------------------

/// `fuse._key(p)`: the rounded coordinates as a dict key (an exact integer
/// cell for a float32).
fn fkey(p: V3) -> [i64; 3] {
    let r = |x: f32| (x as f64 * 1e4).round_ties_even() as i64;
    [r(p.0), r(p.1), r(p.2)]
}

/// `_collapse_all(mesh)`: collapse redundant collinear vertices, always the
/// first collapsible one in vertex-list order, without the rescan.
fn collapse_all(mesh: &mut Mesh, h: Hypot) {
    let order = mesh.vertices.clone();
    let place: HashMap<_, usize> = order.iter().enumerate().map(|(i, &v)| (v, i)).collect();
    let mut gone = vec![false; order.len()];
    let mut i = 0;
    while i < order.len() {
        let v = order[i];
        if gone[i] || !mesh.collapsible_vertex(v, h) {
            i += 1;
            continue;
        }
        let es = mesh.sorted_edges(v);
        let (n1, n2) = (mesh.other(es[0], v), mesh.other(es[1], v));
        mesh.collapse_vertex(v);
        gone[i] = true;
        i = i.min(place.get(&n1).copied().unwrap_or(i)).min(place.get(&n2).copied().unwrap_or(i));
    }
}

/// `simplify_mesh(mesh, max_angle_degrees, principal_planes)`: merge the
/// connected near-coplanar faces of `mesh` in place; returns the faces
/// removed. The new mesh replaces the old one's contents (Python's
/// `mesh.restore_state(simplified.capture_state())` adopts its objects).
pub fn simplify_mesh(mesh: &mut Mesh, max_angle_degrees: f64, principal: bool, h: Hypot) -> Result<usize, String> {
    if !max_angle_degrees.is_finite() || !(0.0..=5.0).contains(&max_angle_degrees) {
        return Err("Simplification angle must be between 0 and 5 degrees".to_string());
    }
    if mesh.faces.is_empty() {
        return Ok(0);
    }
    let original_count = mesh.faces.len();
    let positions = |m: &Mesh, l: &[ingetrazo_model::mesh::VertexId]| -> Vec<V3> { l.iter().map(|&v| m.position(v)).collect() };
    let mut loops: Vec<RawLoop> = Vec::new();
    let mut holed: Vec<FaceId> = Vec::new();
    for &f in &mesh.faces {
        let face = mesh.face(f);
        if !face.holes.is_empty() || face.interior {
            holed.push(f);
        } else {
            loops.push((positions(mesh, &face.outer), face.attrs.clone()));
        }
    }
    let tolerance = max_angle_degrees.to_radians().max(0.00045);
    let cos_tol = tolerance.cos();
    let fused = fuse_coplanar_loops(&loops, cos_tol, principal, h);
    let mut simplified = Mesh::new();
    simplified.hypot = mesh.hypot;
    add_fused(&mut simplified, &fused);
    for &f in &holed {
        let face = mesh.face(f);
        let holes: Vec<Vec<V3>> = face.holes.iter().map(|l| positions(mesh, l)).collect();
        let rebuilt = simplified.add_face(&positions(mesh, &face.outer), &holes);
        update_attrs(&mut simplified, rebuilt, &face.attrs);
        simplified.face_mut(rebuilt).interior = face.interior;
    }
    // The flags of the old edges, by the rounded ends; the last one wins.
    let pair = |a: V3, b: V3| {
        let (ka, kb) = (fkey(a), fkey(b));
        if ka <= kb {
            (ka, kb)
        } else {
            (kb, ka)
        }
    };
    let mut original: HashMap<([i64; 3], [i64; 3]), MeshEdgeFlags> = HashMap::new();
    for &e in &mesh.edges {
        let ed = mesh.edge(e);
        original.insert(
            pair(mesh.position(ed.v0), mesh.position(ed.v1)),
            MeshEdgeFlags { soft: ed.soft, curve: ed.curve, layer: ed.layer.clone(), hidden: ed.hidden },
        );
    }
    for e in simplified.edges.clone() {
        let (a, b) = (simplified.edge(e).v0, simplified.edge(e).v1);
        if let Some(fl) = original.get(&pair(simplified.position(a), simplified.position(b))) {
            let ed = simplified.edge_mut(e);
            ed.soft = fl.soft;
            ed.curve = fl.curve;
            ed.layer = fl.layer.clone();
            ed.hidden = fl.hidden;
        }
    }
    collapse_all(&mut simplified, h);
    soften_smooth_edges(&mut simplified, h);
    let removed = original_count.saturating_sub(simplified.faces.len());
    mesh.adopt(simplified);
    Ok(removed)
}

// ---- groups -----------------------------------------------------------------------

/// `Group.__init__`'s name: the one given, else "Group N" from the process
/// counter (`core.group._counter`, which `counter` stands for).
pub fn group_name(counter: &mut i64, name: &str) -> String {
    if name.is_empty() {
        let n = *counter;
        *counter += 1;
        format!("Group {n}")
    } else {
        name.to_string()
    }
}

/// Puts `mesh` in the scene's arena.
pub fn push_mesh(scene: &mut Scene, mesh: Mesh) -> MeshId {
    scene.meshes.push(mesh);
    MeshId(scene.meshes.len() as u32 - 1)
}

/// `Group(mesh, name)`: a classic group on a mesh of its own.
pub fn new_group(scene: &mut Scene, mesh: Mesh, name: &str, counter: &mut i64) -> Group {
    let id = push_mesh(scene, mesh);
    Group::new(id, group_name(counter, name))
}
