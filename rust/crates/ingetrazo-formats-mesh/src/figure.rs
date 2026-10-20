// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! Face-me figures for a file: `formats/meshexport.py` `_feet`,
//! `_turned_toward` and `_facing_quad`, over `core.group.transformed_mesh`.
//!
//! A file has no camera, so a figure (a photo sprite that always looks at
//! the viewer) is turned toward the Front view and written as a node of its
//! own; the exporter needs the figure's feet, its heading and its turned
//! faces.

use ingetrazo_formats::pyjson::Value;
use ingetrazo_geom::sweep::py_hypot3;
use ingetrazo_math::{Hypot, M4, V3};
use ingetrazo_model::mesh::{EdgeFlags, Mesh};
use ingetrazo_model::pyval;

use crate::tri::face_area;
use crate::uv::fit_uv_affine;
use crate::world::{affine_point, attr, flatten, mesh_faces, np_affine, remap_uvw, translation};

/// `remap_uvws(mesh, m)`: `core.group._remap_uvws` over every face.
fn remap_uvws(mesh: &mut Mesh, m: &M4) {
    let (inv, ok) = m.inverted();
    if !ok {
        return;
    }
    let t = translation(m);
    for f in mesh.faces.clone() {
        let attrs = mesh.face(f).attrs.clone();
        if !attrs.is_empty() {
            mesh.face_mut(f).attrs = remap_uvw(&attrs, &inv, t);
        }
    }
}

/// `transformed_mesh(src, m)`: the deep copy (faces, edges, texture maps),
/// not only its faces as [`crate::world::world_faces`] needs them: a figure
/// turns about its feet, which takes every vertex of the copy, those of loose
/// edges included.
///
/// Holes nested in a bigger hole of one face are not dropped
/// (`_maximal_holes`) in the bulk pass, as in `world_faces`, and curve ids
/// are not re-split (`resplit_curves`): neither moves a position.
pub fn transformed_mesh(src: &Mesh, m: &M4) -> Mesh {
    let mut new = Mesh::new();
    if src.edges.len() * 2 + src.faces.len() * 4 < 1024 {
        for f in mesh_faces(src) {
            let outer: Vec<V3> = f.outer.iter().map(|&p| m.map(p)).collect();
            let holes: Vec<Vec<V3>> = f.holes.iter().map(|h| h.iter().map(|&p| m.map(p)).collect()).collect();
            let nf = new.add_face(&outer, &holes);
            for (k, v) in f.attrs {
                pyval::set(&mut new.face_mut(nf).attrs, &k, v);
            }
        }
        for &e in &src.edges {
            let ed = src.edge(e);
            let (a, b) = (m.map(src.position(ed.v0)), m.map(src.position(ed.v1)));
            let found = match (new.vertex_at(a), new.vertex_at(b)) {
                (Some(v0), Some(v1)) => new.find_edge(v0, v1),
                _ => None,
            };
            let ne = match found {
                Some(ne) => ne,
                None => match new.add_edge(a, b) {
                    Ok(ne) => ne,
                    Err(_) => continue,
                },
            };
            new.edge_mut(ne).soft = ed.soft;
            new.edge_mut(ne).curve = ed.curve;
        }
    } else {
        let (rot, trans) = np_affine(m);
        let flat = flatten(src);
        let n_edge_pts = flat.edge_points.len();
        let mut pts: Vec<V3> = flat.edge_points.iter().map(|&p| affine_point(&rot, &trans, p)).collect();
        let mut sizes = Vec::new();
        let mut counts = Vec::new();
        let mut attrs_list = Vec::new();
        for (outer, holes, attrs) in &flat.faces {
            counts.push(1 + holes.len());
            sizes.push(outer.len());
            pts.extend(outer.iter().map(|&p| affine_point(&rot, &trans, p)));
            for h in holes {
                sizes.push(h.len());
                pts.extend(h.iter().map(|&p| affine_point(&rot, &trans, p)));
            }
            attrs_list.push(attrs.clone());
        }
        if pts.is_empty() {
            return new;
        }
        let (vobjs, inverse) = new.bulk_weld(&pts);
        let mut emap = None;
        if !src.edges.is_empty() {
            let ia: Vec<usize> = (0..n_edge_pts).step_by(2).map(|i| inverse[i]).collect();
            let ib: Vec<usize> = (1..n_edge_pts).step_by(2).map(|i| inverse[i]).collect();
            let flags: Vec<EdgeFlags> =
                src.edges.iter().map(|&e| EdgeFlags { soft: src.edge(e).soft, curve: src.edge(e).curve, layer: None, hidden: false }).collect();
            emap = Some(new.add_edges_welded(&vobjs, &ia, &ib, Some(&flags)));
        }
        if !counts.is_empty() {
            new.add_faces_welded(&vobjs, &inverse[n_edge_pts..], &sizes, &counts, Some(&attrs_list), emap);
        }
    }
    remap_uvws(&mut new, m);
    new
}

fn extent(mesh: &Mesh) -> Option<([f64; 3], [f64; 3])> {
    if mesh.vertices.is_empty() {
        return None;
    }
    let (mut lo, mut hi) = ([f64::INFINITY; 3], [f64::NEG_INFINITY; 3]);
    let mut first = true;
    for &v in &mesh.vertices {
        let p = mesh.position(v);
        let c = [p.0 as f64, p.1 as f64, p.2 as f64];
        for k in 0..3 {
            // Python's min()/max(): the first item, then replaced on < / >.
            if first || c[k] < lo[k] {
                lo[k] = c[k];
            }
            if first || c[k] > hi[k] {
                hi[k] = c[k];
            }
        }
        first = false;
    }
    Some((lo, hi))
}

fn anchor_of(lo: [f64; 3], hi: [f64; 3]) -> V3 {
    V3(((lo[0] + hi[0]) / 2.0) as f32, ((lo[1] + hi[1]) / 2.0) as f32, lo[2] as f32)
}

/// `_feet(mesh)`: the middle of the footprint at the lowest point.
pub fn feet(mesh: &Mesh) -> Option<V3> {
    extent(mesh).map(|(lo, hi)| anchor_of(lo, hi))
}

/// `math.hypot(a, b)`.
fn hypot2(a: f64, b: f64) -> f64 {
    py_hypot3(a, b, 0.0)
}

/// `_turned_toward(mesh, toward)`; `None` stands for "the very same mesh".
pub fn turned_toward(mesh: &Mesh, toward: V3, h: Hypot) -> Option<Mesh> {
    if mesh.faces.is_empty() || mesh.vertices.is_empty() {
        return None;
    }
    let mut big = mesh.faces[0];
    let mut best = face_area(&outer_of(mesh, big), h);
    for &f in &mesh.faces[1..] {
        let a = face_area(&outer_of(mesh, f), h);
        if a > best {
            best = a;
            big = f;
        }
    }
    let n = crate::tri::face_normal(&outer_of(mesh, big), h);
    let (mut nx, mut ny) = (n.0 as f64, n.1 as f64);
    let ln = hypot2(nx, ny);
    if ln < 1e-9 {
        return None;
    }
    let (lo, hi) = extent(mesh)?;
    let anchor = anchor_of(lo, hi);
    let (mut dx, mut dy) = (toward.0 as f64, toward.1 as f64);
    let ld = hypot2(dx, dy);
    if ld < 1e-9 {
        return None;
    }
    nx /= ln;
    ny /= ln;
    dx /= ld;
    dy /= ld;
    let angle = (nx * dy - ny * dx).atan2(nx * dx + ny * dy).to_degrees();
    let mut m = M4::identity();
    m.translate(anchor.0, anchor.1, anchor.2);
    m.rotate(angle as f32, 0.0, 0.0, 1.0);
    let back = anchor.neg();
    m.translate(back.0, back.1, back.2);
    Some(transformed_mesh(mesh, &m))
}

fn outer_of(mesh: &Mesh, f: ingetrazo_model::mesh::FaceId) -> Vec<V3> {
    mesh.face(f).outer.iter().map(|&v| mesh.position(v)).collect()
}

/// `_facing_quad(mesh, toward)`; `None` stands for "the very same mesh".
pub fn facing_quad(mesh: &Mesh, toward: V3, h: Hypot) -> Option<Mesh> {
    let tex = mesh.faces.iter().find_map(|&f| {
        let attrs = &mesh.face(f).attrs;
        match attr(attrs, "texture") {
            Some(t @ Value::Object(items)) if t.get("path").is_some_and(pyval::truthy) => Some(items.clone()),
            _ => None,
        }
    });
    let (Some(mut tex), Some((lo, hi))) = (tex, extent(mesh)) else {
        return turned_toward(mesh, toward, h);
    };
    let anchor = anchor_of(lo, hi);
    let w = hypot2(hi[0] - lo[0], hi[1] - lo[1]);
    let height = hi[2] - lo[2];
    let d = V3(toward.0, toward.1, 0.0);
    if w < 1e-9 || height < 1e-9 || (d.length(h) as f64) < 1e-6 {
        return None;
    }
    let d = d.normalized(h);
    let r = V3(-d.1, d.0, 0.0);
    let up = V3(0.0, 0.0, 1.0);
    let half = (w / 2.0) as f32;
    let (c0, c1) = (anchor.sub(r.mul(half)), anchor.add(r.mul(half)));
    let hh = height as f32;
    let corners = [c0, c1, c1.add(up.mul(hh)), c0.add(up.mul(hh))];
    let uvw = fit_uv_affine(&corners, &[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]);
    pyval::set(&mut tex, "uvw", uvw.map_or(Value::Null, |u| Value::Array(u.iter().map(|&x| Value::Float(x)).collect())));
    let mut out = Mesh::new();
    let face = out.add_face(&corners, &[]);
    let first = mesh.face(mesh.faces[0]).attrs.clone();
    for (k, v) in first {
        if k != "texture" {
            pyval::set(&mut out.face_mut(face).attrs, &k, v);
        }
    }
    pyval::set(&mut out.face_mut(face).attrs, "texture", Value::Object(tex));
    Some(out)
}
