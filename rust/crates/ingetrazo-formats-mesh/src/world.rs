// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! The faces an exporter walks, in world space: `formats/stl.py` and
//! `formats/obj.py` `_faces(scene)` over `core/group.py` `world_mesh`,
//! `transformed_mesh`, `_merged_placements` and `iter_placements`.
//!
//! A component's faces live in its prototype's coordinates; Python builds a
//! transformed copy (welding the moved points again, which can merge two
//! corners that land within the weld tolerance) and an exporter reads that
//! copy's faces. The copy is rebuilt here the same way, through the same
//! weld ([`Mesh::vertex`]), but only as far as a face's corners and attrs go.

use ingetrazo_formats::pyjson::Value;
use ingetrazo_math::{M4, V3};
use ingetrazo_model::mesh::Mesh;
use ingetrazo_model::pyval;
use ingetrazo_model::scene::{Group, Scene, DEFAULT_LAYER};

/// One face as an exporter reads it: corners in world space.
#[derive(Clone, Debug)]
pub struct WFace {
    pub outer: Vec<V3>,
    pub holes: Vec<Vec<V3>>,
    pub attrs: Vec<(String, Value)>,
}

impl WFace {
    /// `face.attrs.get(key)`.
    pub fn attr(&self, key: &str) -> Option<&Value> {
        attr(&self.attrs, key)
    }
}

pub fn attr<'a>(attrs: &'a [(String, Value)], key: &str) -> Option<&'a Value> {
    attrs.iter().rev().find(|(k, _)| k == key).map(|(_, v)| v)
}

fn truthy(v: Option<&Value>) -> bool {
    v.is_some_and(pyval::truthy)
}

/// The faces of a mesh as they are stored.
pub fn mesh_faces(mesh: &Mesh) -> Vec<WFace> {
    mesh.faces
        .iter()
        .map(|&f| {
            let face = mesh.face(f);
            WFace {
                outer: face.outer.iter().map(|&v| mesh.position(v)).collect(),
                holes: face.holes.iter().map(|h| h.iter().map(|&v| mesh.position(v)).collect()).collect(),
                attrs: face.attrs.clone(),
            }
        })
        .collect()
}

// ---- visibility (core/scene.py entity_visible) -----------------------------

fn layer_visible(scene: &Scene, name: &Value) -> bool {
    let Value::Str(name) = name else { return true };
    for layer in &scene.layers {
        if matches!(layer.get("name"), Some(Value::Str(n)) if n == name) {
            return layer.get("visible").map(pyval::truthy).unwrap_or(true);
        }
    }
    true
}

fn default_layer() -> Value {
    Value::Str(DEFAULT_LAYER.to_string())
}

/// `scene.entity_visible(face)`: not hidden, and its layer shown.
pub fn face_visible(scene: &Scene, f: &WFace) -> bool {
    if truthy(f.attr("hidden")) {
        return false;
    }
    let layer = match f.attr("layer") {
        Some(l) if pyval::truthy(l) => l.clone(),
        _ => default_layer(),
    };
    layer_visible(scene, &layer)
}

/// `scene.entity_visible(group)`.
pub fn group_visible(scene: &Scene, g: &Group) -> bool {
    if g.hidden {
        return false;
    }
    match &g.layer {
        Some(l) if !l.is_empty() => layer_visible(scene, &Value::Str(l.clone())),
        _ => true,
    }
}

pub fn billboard(g: &Group) -> bool {
    g.billboard.as_ref().is_some_and(pyval::truthy)
}

// ---- transformed copies -----------------------------------------------------

/// `_remap_uvws` / `transformed_attrs` on one face: the texture's world to
/// UV map follows the transform, g' = L⁻ᵀ·g, c' = c − g'·t.
pub(crate) fn remap_uvw(attrs: &[(String, Value)], inv: &M4, t: [f64; 3]) -> Vec<(String, Value)> {
    let mut out = attrs.to_vec();
    let Some(Value::Object(tex)) = attr(attrs, "texture") else { return out };
    let Some(Value::Array(uvw)) = tex.iter().rev().find(|(k, _)| k == "uvw").map(|(_, v)| v) else { return out };
    if uvw.len() != 8 {
        return out;
    }
    let Ok(u): Result<Vec<f64>, _> = uvw.iter().map(pyval::float).collect() else { return out };
    let minv = |r: usize, c: usize| inv.row(r)[c] as f64;
    let mut new = u.clone();
    for base in [0usize, 4] {
        let (gx, gy, gz, c) = (u[base], u[base + 1], u[base + 2], u[base + 3]);
        let gpx = minv(0, 0) * gx + minv(1, 0) * gy + minv(2, 0) * gz;
        let gpy = minv(0, 1) * gx + minv(1, 1) * gy + minv(2, 1) * gz;
        let gpz = minv(0, 2) * gx + minv(1, 2) * gy + minv(2, 2) * gz;
        new[base] = gpx;
        new[base + 1] = gpy;
        new[base + 2] = gpz;
        new[base + 3] = c - (gpx * t[0] + gpy * t[1] + gpz * t[2]);
    }
    let mut tex = tex.clone();
    pyval::set(&mut tex, "uvw", Value::Array(new.into_iter().map(Value::Float).collect()));
    pyval::set(&mut out, "texture", Value::Object(tex));
    out
}

/// The translation column `m(0, 3), m(1, 3), m(2, 3)`.
pub(crate) fn translation(m: &M4) -> [f64; 3] {
    [m.row(0)[3] as f64, m.row(1)[3] as f64, m.row(2)[3] as f64]
}

/// `transformed_attrs(attrs, m)`.
fn transformed_attrs(attrs: &[(String, Value)], m: &M4) -> Vec<(String, Value)> {
    let (inv, ok) = m.inverted();
    if !ok {
        return attrs.to_vec();
    }
    remap_uvw(attrs, &inv, translation(m))
}

/// `np_affine(m)`: `(rot, trans)` as doubles, from the column-major data.
pub(crate) fn np_affine(m: &M4) -> ([[f64; 3]; 3], [f64; 3]) {
    let d = m.data().map(|x| x as f64);
    ([[d[0], d[4], d[8]], [d[1], d[5], d[9]], [d[2], d[6], d[10]]], [d[12], d[13], d[14]])
}

/// `pts @ rot.T + trans` for one point, as NumPy's float64 matmul (OpenBLAS
/// on x86-64, checked against it for N = 1..1000 rows) computes a row: the
/// first product, then each next one fused into the running sum, then the
/// translation added.
pub(crate) fn affine_point(rot: &[[f64; 3]; 3], trans: &[f64; 3], p: V3) -> V3 {
    let (x, y, z) = (p.0 as f64, p.1 as f64, p.2 as f64);
    let row = |i: usize| z.mul_add(rot[i][2], y.mul_add(rot[i][1], x * rot[i][0])) + trans[i];
    V3(row(0) as f32, row(1) as f32, row(2) as f32)
}

/// A mesh's rings in the order `transformed_mesh` flattens them: the end
/// points of every edge, then each face's outer ring and its holes.
pub(crate) struct Flat {
    pub(crate) edge_points: Vec<V3>,
    pub(crate) faces: Vec<(Vec<V3>, Vec<Vec<V3>>, Vec<(String, Value)>)>,
}

pub(crate) fn flatten(src: &Mesh) -> Flat {
    let mut edge_points = Vec::with_capacity(src.edges.len() * 2);
    for &e in &src.edges {
        let ed = src.edge(e);
        edge_points.push(src.position(ed.v0));
        edge_points.push(src.position(ed.v1));
    }
    let faces = mesh_faces(src).into_iter().map(|f| (f.outer, f.holes, f.attrs)).collect();
    Flat { edge_points, faces }
}

/// One `transformed_mesh(src, m)` of any size, as faces.
fn transformed_faces(src: &Mesh, m: &M4) -> Vec<WFace> {
    let (inv, ok) = m.inverted();
    let t = translation(m);
    let remap = |attrs: Vec<(String, Value)>| if ok { remap_uvw(&attrs, &inv, t) } else { attrs };
    if src.edges.len() * 2 + src.faces.len() * 4 < 1024 {
        // The per-face walk: `add_face` of the moved corners.
        let mut new = Mesh::new();
        let mut ids = Vec::new();
        for f in mesh_faces(src) {
            let outer: Vec<V3> = f.outer.iter().map(|&p| m.map(p)).collect();
            let holes: Vec<Vec<V3>> = f.holes.iter().map(|h| h.iter().map(|&p| m.map(p)).collect()).collect();
            ids.push((new.add_face(&outer, &holes), f.attrs));
        }
        return ids
            .into_iter()
            .map(|(id, attrs)| {
                let face = new.face(id);
                WFace {
                    outer: face.outer.iter().map(|&v| new.position(v)).collect(),
                    holes: face.holes.iter().map(|h| h.iter().map(|&v| new.position(v)).collect()).collect(),
                    attrs: remap(attrs),
                }
            })
            .collect();
    }
    // The bulk pass: every point moved in double, then welded in order.
    let (rot, trans) = np_affine(m);
    let flat = flatten(src);
    let mut new = Mesh::new();
    for &p in &flat.edge_points {
        new.vertex(affine_point(&rot, &trans, p));
    }
    flat.faces
        .into_iter()
        .map(|(outer, holes, attrs)| {
            let mut weld = |ring: &[V3]| -> Vec<V3> {
                ring.iter()
                    .map(|&p| {
                        let id = new.vertex(affine_point(&rot, &trans, p));
                        new.position(id)
                    })
                    .collect()
            };
            let outer = weld(&outer);
            let holes = holes.iter().map(|h| weld(h)).collect();
            WFace { outer, holes, attrs: remap(attrs) }
        })
        .collect()
}

/// `iter_placements(group)`: `(group, world matrix)` depth first; the matrix
/// is `None` while the whole chain is untransformed.
fn placements<'a>(g: &'a Group, base: Option<M4>, out: &mut Vec<(&'a Group, Option<M4>)>) {
    let m = match (g.xform, base) {
        (Some(x), Some(b)) => Some(b.mul(&x)),
        (Some(x), None) => Some(x),
        (None, b) => b,
    };
    out.push((g, m));
    for c in &g.children {
        placements(c, m, out);
    }
}

/// `_merged_placements(group)`: every placement welded into one mesh, the
/// edges of all of them first, then the faces.
fn merged_faces(scene: &Scene, group: &Group) -> Vec<WFace> {
    let mut list = Vec::new();
    placements(group, None, &mut list);
    let mut new = Mesh::new();
    let moved = |affine: &Option<([[f64; 3]; 3], [f64; 3])>, p: V3| match affine {
        Some((rot, trans)) => affine_point(rot, trans, p),
        None => p,
    };
    for (g, m) in &list {
        let affine = m.as_ref().map(np_affine);
        for p in flatten(scene.mesh(g.mesh)).edge_points {
            new.vertex(moved(&affine, p));
        }
    }
    let mut out = Vec::new();
    for (g, m) in &list {
        let affine = m.as_ref().map(np_affine);
        for f in mesh_faces(scene.mesh(g.mesh)) {
            let mut weld = |ring: &[V3]| -> Vec<V3> {
                ring.iter()
                    .map(|&p| {
                        let id = new.vertex(moved(&affine, p));
                        new.position(id)
                    })
                    .collect()
            };
            let outer = weld(&f.outer);
            let holes = f.holes.iter().map(|h| weld(h)).collect();
            let attrs = match m {
                Some(m) if !f.attrs.is_empty() => transformed_attrs(&f.attrs, m),
                _ => f.attrs,
            };
            out.push(WFace { outer, holes, attrs });
        }
    }
    out
}

/// `world_mesh(group).faces`.
fn world_mesh_faces(scene: &Scene, g: &Group) -> Vec<WFace> {
    if g.children.is_empty() {
        let src = scene.mesh(g.mesh);
        return match &g.xform {
            None => mesh_faces(src),
            Some(m) => transformed_faces(src, m),
        };
    }
    merged_faces(scene, g)
}

/// `_faces(scene)`: every renderable face in world space, loose mesh first.
pub fn world_faces(scene: &Scene) -> Vec<WFace> {
    let mut out: Vec<WFace> = mesh_faces(scene.loose()).into_iter().filter(|f| face_visible(scene, f)).collect();
    // Without a placed group `render_faces` reads each group's own mesh
    // (nested placements cannot exist: owning one makes a group an instance).
    for g in &scene.groups {
        if !group_visible(scene, g) || billboard(g) {
            continue;
        }
        if scene.groups.iter().any(|x| x.xform.is_some()) {
            out.extend(world_mesh_faces(scene, g));
        } else {
            out.extend(mesh_faces(scene.mesh(g.mesh)));
        }
    }
    out
}
