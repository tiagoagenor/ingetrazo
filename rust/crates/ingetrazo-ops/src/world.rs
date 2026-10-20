// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! A group's geometry in world space as a mesh: `core/group.py`
//! `world_mesh` and `transformed_mesh`, for a group without nested
//! placements (the only kind a solid can be).
//!
//! A classic group's world mesh is its mesh. A component instance's is a
//! DEEP copy with every position mapped through its matrix, welding the
//! moved points again (which can merge two corners that land within the
//! weld tolerance), carrying face attrs, soft and curve flags and the
//! re-fitted texture UV maps. Small meshes are copied face by face, big ones
//! through the bulk weld, as Python does.

use std::borrow::Cow;

use ingetrazo_formats::pyjson::Value;
use ingetrazo_formats_mesh::world::{affine_point, np_affine, transformed_attrs};
use ingetrazo_math::{Hypot, M4, V3};
use ingetrazo_model::mesh::{EdgeFlags, Mesh};
use ingetrazo_model::scene::{Group, Scene};
use ingetrazo_model::topology::{add_face_py, maximal_holes};

/// `world_mesh(group)` for a group with no children.
pub fn world_mesh<'a>(scene: &'a Scene, g: &Group, h: Hypot) -> Cow<'a, Mesh> {
    let src = scene.mesh(g.mesh);
    match &g.xform {
        None => Cow::Borrowed(src),
        Some(m) => Cow::Owned(transformed_mesh(src, m, h)),
    }
}

fn positions(mesh: &Mesh, ring: &[ingetrazo_model::VertexId]) -> Vec<V3> {
    ring.iter().map(|&v| mesh.position(v)).collect()
}

/// `transformed_mesh(src, m)`.
pub fn transformed_mesh(src: &Mesh, m: &M4, h: Hypot) -> Mesh {
    if src.edges.len() * 2 + src.faces.len() * 4 < 1024 {
        return transformed_mesh_walk(src, m, h);
    }
    let mut new = Mesh::new();
    let (rot, trans) = np_affine(m);
    let mut flat: Vec<V3> = Vec::new();
    for &e in &src.edges {
        let ed = src.edge(e);
        flat.push(affine_point(&rot, &trans, src.position(ed.v0)));
        flat.push(affine_point(&rot, &trans, src.position(ed.v1)));
    }
    let n_edge_pts = flat.len();
    let mut ring_sizes: Vec<usize> = Vec::new();
    let mut ring_counts: Vec<usize> = Vec::new();
    let mut attrs_list: Vec<Vec<(String, Value)>> = Vec::new();
    for &f in &src.faces {
        let face = src.face(f);
        let hole_pos: Vec<Vec<V3>> = face.holes.iter().map(|hl| positions(src, hl)).collect();
        let keep: Vec<usize> = if hole_pos.len() > 1 { maximal_holes(&hole_pos, h) } else { (0..hole_pos.len()).collect() };
        ring_counts.push(1 + keep.len());
        ring_sizes.push(face.outer.len());
        flat.extend(face.outer.iter().map(|&v| affine_point(&rot, &trans, src.position(v))));
        for &k in &keep {
            ring_sizes.push(hole_pos[k].len());
            flat.extend(hole_pos[k].iter().map(|&p| affine_point(&rot, &trans, p)));
        }
        attrs_list.push(face.attrs.clone());
    }
    if flat.is_empty() {
        return new;
    }
    let (vobjs, inverse) = new.bulk_weld(&flat);
    let mut emap = None;
    if !src.edges.is_empty() {
        let flags: Vec<EdgeFlags> = src
            .edges
            .iter()
            .map(|&e| {
                let ed = src.edge(e);
                EdgeFlags { soft: ed.soft, curve: ed.curve, layer: None, hidden: false }
            })
            .collect();
        let ia: Vec<usize> = (0..n_edge_pts).step_by(2).map(|i| inverse[i]).collect();
        let ib: Vec<usize> = (1..n_edge_pts).step_by(2).map(|i| inverse[i]).collect();
        emap = Some(new.add_edges_welded(&vobjs, &ia, &ib, Some(&flags)));
    }
    if !ring_counts.is_empty() {
        new.add_faces_welded(&vobjs, &inverse[n_edge_pts..], &ring_sizes, &ring_counts, Some(&attrs_list), emap);
    }
    remap_uvws(&mut new, m);
    new
}

/// `_transformed_mesh_walk`: the per-entity copy.
fn transformed_mesh_walk(src: &Mesh, m: &M4, h: Hypot) -> Mesh {
    let mut new = Mesh::new();
    let w = |p: V3| m.map(p);
    for &f in &src.faces {
        let face = src.face(f);
        let outer: Vec<V3> = face.outer.iter().map(|&v| w(src.position(v))).collect();
        let holes: Vec<Vec<V3>> = face.holes.iter().map(|hl| hl.iter().map(|&v| w(src.position(v))).collect()).collect();
        let nf = add_face_py(&mut new, &outer, &holes, h);
        if !face.attrs.is_empty() {
            new.face_mut(nf).attrs = face.attrs.clone();
        }
    }
    for &e in &src.edges {
        let ed = src.edge(e);
        let (a, b) = (w(src.position(ed.v0)), w(src.position(ed.v1)));
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
        let out = new.edge_mut(ne);
        out.soft = ed.soft;
        out.curve = ed.curve;
    }
    remap_uvws(&mut new, m);
    new
}

/// `_remap_uvws`: each face's world to UV map follows the transform.
fn remap_uvws(mesh: &mut Mesh, m: &M4) {
    let (_, ok) = m.inverted();
    if !ok {
        return;
    }
    for f in mesh.faces.clone() {
        if mesh.face(f).attrs.is_empty() {
            continue;
        }
        let moved = transformed_attrs(&mesh.face(f).attrs, m);
        mesh.face_mut(f).attrs = moved;
    }
}
