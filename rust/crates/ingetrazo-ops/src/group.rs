// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! The functions of `core/group.py` the transform commands use: placements,
//! world meshes, transformed copies, copies of groups, local axes.
//!
//! Python's groups hold their `Mesh` by reference; here a group names its
//! mesh by [`MeshId`] into `Scene::meshes`, so the functions that make a mesh
//! take the arena (`&mut Vec<Mesh>`) rather than the scene, which lets a
//! caller hold a group of `scene.groups` at the same time.
//!
//! NumPy. `transformed_mesh` and `_merged_placements` map their points with
//! `pts @ rot.T + trans` in float64, and NumPy hands that product to
//! OpenBLAS. Checked against NumPy 2.5.4 on Haswell-class AVX2 + FMA (see
//! `hlr_draw.rs`, same machine), each output is
//!
//! * `N >= 2` rows (`dgemm`): `fma(a2, b2, fma(a1, b1, a0*b0)) + t`,
//! * `N == 1` (`dgemv`): `fma(a2, b2, fma(a0, b0, a1*b1)) + t`.
//!
//! A nested placement always has at least one edge or one face, so `N >= 2`
//! in `_merged_placements`; `N == 1` is kept for completeness.

use std::borrow::Cow;

use ingetrazo_formats::pyjson::Value;
use ingetrazo_math::{Hypot, M4, V3};
use ingetrazo_model::mesh::{EdgeFlags, Mesh};
use ingetrazo_model::scene::{Group, MeshId};
use ingetrazo_model::topology::maximal_holes;

use crate::command::Counters;

/// Where a group is in the scene tree: indexes from `scene.groups` through
/// `children` (the `$group` selector of the golden schema).
pub type Path = Vec<usize>;

/// The group at `path`, if there is one.
pub fn group_at<'a>(groups: &'a [Group], path: &[usize]) -> Option<&'a Group> {
    let (first, rest) = path.split_first()?;
    let mut g = groups.get(*first)?;
    for &i in rest {
        g = g.children.get(i)?;
    }
    Some(g)
}

pub fn group_at_mut<'a>(groups: &'a mut [Group], path: &[usize]) -> Option<&'a mut Group> {
    let (first, rest) = path.split_first()?;
    let mut g = groups.get_mut(*first)?;
    for &i in rest {
        g = g.children.get_mut(i)?;
    }
    Some(g)
}

// ---- local axes (issue #44) ---------------------------------------------------

/// `carry_axes`: a classic group's geometry was just transformed by `m` in
/// world space; its axes go with it. Instances are left alone.
pub fn carry_axes(group: &mut Group, m: &M4) {
    if group.xform.is_some() {
        return;
    }
    let base = group.axes.unwrap_or_default();
    group.axes = Some(m.mul(&base));
}

/// `group_frame`: the world matrix of the group's own axes (`xform * axes`),
/// or `None` for the world axes.
pub fn group_frame(group: &Group) -> Option<M4> {
    match (&group.xform, &group.axes) {
        (None, axes) => *axes,
        (Some(x), None) => Some(*x),
        (Some(x), Some(a)) => Some(x.mul(a)),
    }
}

/// `frame_axes`: `(origin, x, y, z)` of a frame matrix, the three axes made
/// unit and square to each other. `None` is the world.
pub fn frame_axes(m: Option<&M4>, h: Hypot) -> (V3, V3, V3, V3) {
    let Some(m) = m else {
        return (V3(0.0, 0.0, 0.0), V3(1.0, 0.0, 0.0), V3(0.0, 1.0, 0.0), V3(0.0, 0.0, 1.0));
    };
    let o = m.map(V3(0.0, 0.0, 0.0));
    let mut x = m.map_vector(V3(1.0, 0.0, 0.0));
    let mut y = m.map_vector(V3(0.0, 1.0, 0.0));
    let z = m.map_vector(V3(0.0, 0.0, 1.0));
    if (x.length(h) as f64) < 1e-12 {
        x = V3(1.0, 0.0, 0.0);
    }
    x = x.normalized(h);
    y = y.sub(x.mul(y.dot(x)));
    if (y.length(h) as f64) < 1e-12 {
        y = z.cross(x);
    }
    y = y.normalized(h);
    let mut zz = x.cross(y).normalized(h);
    if (zz.dot(z) as f64) < 0.0 {
        zz = zz.neg(); // a mirrored placement
    }
    (o, x, y, zz)
}

/// `core.axes.frame_matrix`: the matrix whose columns are `x`, `y`, `z` and
/// whose translation is `o`. `QMatrix4x4(16 floats)`: General.
pub fn frame_matrix(o: V3, x: V3, y: V3, z: V3) -> M4 {
    M4::from_rows([x.0, y.0, z.0, o.0, x.1, y.1, z.1, o.1, x.2, y.2, z.2, o.2, 0.0, 0.0, 0.0, 1.0])
}

// ---- face attrs and texture maps ----------------------------------------------

/// A JSON number as a Python float (`int` or `float`).
pub fn num(v: &Value) -> Option<f64> {
    match v {
        Value::Float(f) => Some(*f),
        Value::Int(s) => s.parse::<f64>().ok(),
        Value::Bool(b) => Some(f64::from(u8::from(*b))),
        _ => None,
    }
}

/// `texture.uvw` of an attrs dict when it is a list of 8 numbers.
fn uvw_of(attrs: &[(String, Value)]) -> Option<([f64; 8], usize)> {
    let idx = attrs.iter().position(|(k, _)| k == "texture")?;
    let Value::Object(t) = &attrs[idx].1 else { return None };
    if t.is_empty() {
        return None;
    }
    let Some(Value::Array(a)) = t.iter().rev().find(|(k, _)| k == "uvw").map(|(_, v)| v) else { return None };
    if a.len() != 8 {
        return None;
    }
    let mut out = [0.0; 8];
    for (i, x) in a.iter().enumerate() {
        out[i] = num(x)?;
    }
    Some((out, idx))
}

/// Put `new` in place of `texture.uvw` (`{**t, "uvw": new}` keeps the key's place).
fn set_uvw(attrs: &mut [(String, Value)], idx: usize, new: [f64; 8]) {
    if let Value::Object(t) = &mut attrs[idx].1 {
        let arr = Value::Array(new.iter().map(|&x| Value::Float(x)).collect());
        match t.iter_mut().rev().find(|(k, _)| k == "uvw") {
            Some(slot) => slot.1 = arr,
            None => t.push(("uvw".into(), arr)),
        }
    }
}

/// g' = L⁻ᵀ·g, c' = c − g'·t, for both maps of an 8-number `uvw`.
fn refit(uvw: &[f64; 8], minv: &M4, t: [f64; 3]) -> [f64; 8] {
    // `minv(row, col)` read as Python floats (float32 widened).
    let e = |r: usize, c: usize| minv.m[c][r] as f64;
    let mut new = *uvw;
    for base in [0usize, 4] {
        let (gx, gy, gz, c) = (uvw[base], uvw[base + 1], uvw[base + 2], uvw[base + 3]);
        let gpx = e(0, 0) * gx + e(1, 0) * gy + e(2, 0) * gz;
        let gpy = e(0, 1) * gx + e(1, 1) * gy + e(2, 1) * gz;
        let gpz = e(0, 2) * gx + e(1, 2) * gy + e(2, 2) * gz;
        new[base] = gpx;
        new[base + 1] = gpy;
        new[base + 2] = gpz;
        new[base + 3] = c - (gpx * t[0] + gpy * t[1] + gpz * t[2]);
    }
    new
}

/// `m(0, 3), m(1, 3), m(2, 3)`.
fn translation_of(m: &M4) -> [f64; 3] {
    [m.m[3][0] as f64, m.m[3][1] as f64, m.m[3][2] as f64]
}

/// PySide6 reads `m(r, c)` through Qt's NON-const `operator()`, which sets
/// the matrix's type flag to General: the caller's matrix changes with it,
/// and later products, maps and inverses take the General paths.
fn mark_read(m: &mut M4) {
    m.flags = ingetrazo_math::matrix::GENERAL;
}

/// `_remap_uvws`: rewrite each face's world→UV affine map so that
/// evaluating it at the TRANSFORMED positions reproduces the prototype's UVs.
pub fn remap_uvws(mesh: &mut Mesh, m: &mut M4) {
    let (minv, ok) = m.inverted();
    if !ok {
        return;
    }
    let t = translation_of(m);
    mark_read(m);
    for f in mesh.faces.clone() {
        let Some((uvw, idx)) = uvw_of(&mesh.face(f).attrs) else { continue };
        let new = refit(&uvw, &minv, t);
        set_uvw(&mut mesh.face_mut(f).attrs, idx, new);
    }
}

/// `transformed_attrs`: a copy of face `attrs` with the texture map re-fitted
/// for geometry mapped through `m`.
pub fn transformed_attrs(attrs: &[(String, Value)], m: &mut M4) -> Vec<(String, Value)> {
    let mut out = attrs.to_vec();
    let Some((uvw, idx)) = uvw_of(&out) else { return out };
    let (minv, ok) = m.inverted();
    if !ok {
        return out;
    }
    let t = translation_of(m);
    mark_read(m);
    let new = refit(&uvw, &minv, t);
    set_uvw(&mut out, idx, new);
    out
}

/// `translated_attrs`: a copy of face `attrs` re-anchored for geometry moved
/// by `delta` (the constant term of each map follows the translation).
pub fn translated_attrs(attrs: &[(String, Value)], delta: V3) -> Vec<(String, Value)> {
    let mut out = attrs.to_vec();
    let Some((uvw, idx)) = uvw_of(&out) else { return out };
    let mut new = uvw;
    for base in [0usize, 4] {
        let (gx, gy, gz, c) = (uvw[base], uvw[base + 1], uvw[base + 2], uvw[base + 3]);
        new[base + 3] = c - (gx * delta.0 as f64 + gy * delta.1 as f64 + gz * delta.2 as f64);
    }
    set_uvw(&mut out, idx, new);
    out
}

// ---- placements -----------------------------------------------------------------

/// `iter_placements`: `(group, world_matrix)` for the group and every nested
/// placement below it, depth first, each matrix composed from the root down.
/// The matrix is `None` only when the whole chain is untransformed.
pub fn iter_placements(group: &Group, base: Option<M4>) -> Vec<(&Group, Option<M4>)> {
    let mut out = Vec::new();
    placements_into(group, base, &mut out);
    out
}

fn placements_into<'a>(group: &'a Group, base: Option<M4>, out: &mut Vec<(&'a Group, Option<M4>)>) {
    let m = match (group.xform, base) {
        (m, None) => m,
        (None, Some(b)) => Some(b),
        (Some(x), Some(b)) => Some(b.mul(&x)),
    };
    out.push((group, m));
    for c in &group.children {
        placements_into(c, m, out);
    }
}

/// The paths of the groups of `iter_placements(scene.groups[top])`, depth
/// first (the order is the same).
pub fn placement_paths(groups: &[Group]) -> Vec<Path> {
    fn walk(g: &Group, path: &mut Path, out: &mut Vec<Path>) {
        out.push(path.clone());
        for (i, c) in g.children.iter().enumerate() {
            path.push(i);
            walk(c, path, out);
            path.pop();
        }
    }
    let mut out = Vec::new();
    for (i, g) in groups.iter().enumerate() {
        let mut p = vec![i];
        walk(g, &mut p, &mut out);
    }
    out
}

// ---- copies of meshes -------------------------------------------------------------

/// `np_affine`: `(rot, trans)` with `pts @ rot.T + trans` = `m.map(p)`.
pub fn np_affine(m: &M4) -> ([[f64; 3]; 3], [f64; 3]) {
    let d = m.data().map(|x| x as f64);
    ([[d[0], d[4], d[8]], [d[1], d[5], d[9]], [d[2], d[6], d[10]]], [d[12], d[13], d[14]])
}

/// `np.array(pts) @ rot.T + trans` over float64 rows (see the module notes
/// for the BLAS order).
pub fn affine_rows(pts: &[[f64; 3]], rot: &[[f64; 3]; 3], trans: &[f64; 3]) -> Vec<[f64; 3]> {
    let single = pts.len() == 1;
    pts.iter()
        .map(|a| {
            let mut o = [0.0; 3];
            for r in 0..3 {
                let b = &rot[r];
                let s = if single {
                    a[2].mul_add(b[2], a[0].mul_add(b[0], a[1] * b[1]))
                } else {
                    a[2].mul_add(b[2], a[1].mul_add(b[1], a[0] * b[0]))
                };
                o[r] = s + trans[r];
            }
            o
        })
        .collect()
}

fn pos3(m: &Mesh, v: ingetrazo_model::VertexId) -> [f64; 3] {
    let p = m.position(v);
    [p.0 as f64, p.1 as f64, p.2 as f64]
}

fn positions(m: &Mesh, ids: &[ingetrazo_model::VertexId]) -> Vec<V3> {
    ids.iter().map(|&v| m.position(v)).collect()
}

/// One face's rings as float64 rows, nested holes dropped when there are
/// two or more (`_maximal_holes`).
fn face_rings(src: &Mesh, f: ingetrazo_model::FaceId) -> (Vec<[f64; 3]>, Vec<Vec<[f64; 3]>>) {
    let face = src.face(f);
    let outer: Vec<[f64; 3]> = face.outer.iter().map(|&v| pos3(src, v)).collect();
    let mut holes: Vec<Vec<V3>> = face.holes.iter().map(|h| positions(src, h)).collect();
    if holes.len() > 1 {
        let keep = maximal_holes(&holes, src.hypot);
        holes = keep.into_iter().map(|i| holes[i].clone()).collect();
    }
    let holes = holes.iter().map(|h| h.iter().map(|p| [p.0 as f64, p.1 as f64, p.2 as f64]).collect()).collect();
    (outer, holes)
}

fn to_v3(p: &[f64; 3]) -> V3 {
    V3(p[0] as f32, p[1] as f32, p[2] as f32)
}

/// What a bulk build collects from one mesh or one placement.
#[derive(Default)]
struct Bulk {
    edge_pts: Vec<[f64; 3]>,
    face_pts: Vec<[f64; 3]>,
    edge_flags: Vec<EdgeFlags>,
    ring_sizes: Vec<usize>,
    ring_counts: Vec<usize>,
    attrs: Vec<Vec<(String, Value)>>,
}

/// `bulk_weld` + `add_edges_welded` + `add_faces_welded` + `resplit_curves`
/// over collected points, as `transformed_mesh` and `_merged_placements` end.
fn build_bulk(new: &mut Mesh, b: Bulk, curve: &mut i64) {
    let n_edge_pts = b.edge_pts.len();
    if n_edge_pts == 0 && b.face_pts.is_empty() {
        return;
    }
    // `bulk_weld` rounds to float32 first.
    let pts: Vec<V3> = b.edge_pts.iter().chain(b.face_pts.iter()).map(to_v3).collect();
    let (vobjs, inverse) = new.bulk_weld(&pts);
    let mut emap = None;
    if n_edge_pts > 0 {
        let ia: Vec<usize> = (0..n_edge_pts).step_by(2).map(|i| inverse[i]).collect();
        let ib: Vec<usize> = (1..n_edge_pts).step_by(2).map(|i| inverse[i]).collect();
        emap = Some(new.add_edges_welded(&vobjs, &ia, &ib, Some(&b.edge_flags)));
    }
    if !b.ring_counts.is_empty() {
        new.add_faces_welded(&vobjs, &inverse[n_edge_pts..], &b.ring_sizes, &b.ring_counts, Some(&b.attrs), emap);
    }
    new.resplit_curves(curve);
}

/// `transformed_mesh`: a DEEP copy of `src` with every position mapped
/// through `m`, carrying face attrs, soft/curve flags and re-fitted texture
/// maps. Big meshes go through one bulk weld, small ones keep the walk.
pub fn transformed_mesh(src: &Mesh, m: &mut M4, curve: &mut i64) -> Mesh {
    if src.edges.len() * 2 + src.faces.len() * 4 < 1024 {
        return transformed_mesh_walk(src, m, curve);
    }
    let mut new = Mesh::new();
    new.hypot = src.hypot;
    let mut b = Bulk::default();
    for &e in &src.edges {
        let ed = src.edge(e);
        b.edge_pts.push(pos3(src, ed.v0));
        b.edge_pts.push(pos3(src, ed.v1));
    }
    for &f in &src.faces {
        let (outer, holes) = face_rings(src, f);
        b.ring_counts.push(1 + holes.len());
        b.ring_sizes.push(outer.len());
        b.face_pts.extend(outer);
        for h in holes {
            b.ring_sizes.push(h.len());
            b.face_pts.extend(h);
        }
        b.attrs.push(src.face(f).attrs.clone());
    }
    if b.edge_pts.is_empty() && b.face_pts.is_empty() {
        return new;
    }
    let (rot, trans) = np_affine(m);
    b.edge_pts = affine_rows(&b.edge_pts, &rot, &trans);
    b.face_pts = affine_rows(&b.face_pts, &rot, &trans);
    // The flat array of Python is one matmul over edges and faces together;
    // N >= 2 either way, so splitting it changes nothing.
    if !src.edges.is_empty() {
        b.edge_flags = src.edges.iter().map(|&e| EdgeFlags { soft: src.edge(e).soft, curve: src.edge(e).curve, layer: None, hidden: false }).collect();
    }
    build_bulk(&mut new, b, curve);
    remap_uvws(&mut new, m);
    new
}

/// `_transformed_mesh_walk`: the sequential per-entity copy.
fn transformed_mesh_walk(src: &Mesh, m: &mut M4, curve: &mut i64) -> Mesh {
    let mut new = Mesh::new();
    new.hypot = src.hypot;
    let w = |p: V3| m.map(p);
    for &f in &src.faces {
        let face = src.face(f);
        let outer: Vec<V3> = positions(src, &face.outer).into_iter().map(w).collect();
        let holes: Vec<Vec<V3>> = face.holes.iter().map(|h| positions(src, h).into_iter().map(w).collect()).collect();
        let nf = new.add_face(&outer, &holes);
        if !face.attrs.is_empty() {
            let fa = &mut new.face_mut(nf).attrs;
            for (k, v) in &face.attrs {
                match fa.iter_mut().find(|(x, _)| x == k) {
                    Some(slot) => slot.1 = v.clone(),
                    None => fa.push((k.clone(), v.clone())),
                }
            }
        }
    }
    for &e in &src.edges {
        let ed = src.edge(e);
        let (a, b) = (w(src.position(ed.v0)), w(src.position(ed.v1)));
        let (v0, v1) = (new.vertex_at(a), new.vertex_at(b));
        let mut ne = match (v0, v1) {
            (Some(x), Some(y)) => new.find_edge(x, y),
            _ => None,
        };
        if ne.is_none() {
            match new.add_edge(a, b) {
                Ok(x) => ne = Some(x),
                Err(_) => continue, // degenerate under the transform
            }
        }
        let ne = ne.unwrap_or_else(|| unreachable!());
        let edge = new.edge_mut(ne);
        edge.soft = ed.soft;
        edge.curve = ed.curve;
    }
    new.resplit_curves(curve);
    remap_uvws(&mut new, m);
    new
}

/// `_merged_placements`: every placement of `group` welded into ONE mesh in
/// a single vectorised pass.
fn merged_placements(meshes: &[Mesh], group: &mut Group, curve: &mut i64) -> Mesh {
    let mut new = Mesh::new();
    new.hypot = meshes[group.mesh.0 as usize].hypot;
    let mut b = Bulk::default();
    walk_merge(meshes, group, &mut None, &mut b);
    if b.edge_pts.is_empty() && b.face_pts.is_empty() {
        return new;
    }
    build_bulk(&mut new, b, curve);
    new
}

/// One step of `iter_placements` and the loop body of `_merged_placements`.
/// The matrix of a placement is a Python object that the body mutates (the
/// `m(r, c)` reads of `transformed_attrs` flag it General) and that the
/// generator then hands on as the base of the children — so it is walked
/// by reference: a placement without a matrix of its own shares its
/// parent's, and the group's own `xform` is the object when nothing is
/// above it.
fn walk_merge(meshes: &[Mesh], g: &mut Group, base: &mut Option<M4>, b: &mut Bulk) {
    let mesh_id = g.mesh;
    let (xform, children) = (&mut g.xform, &mut g.children);
    let mut product: Option<M4>;
    let m: &mut Option<M4> = match (base.is_some(), xform.is_some()) {
        (false, _) => xform,
        (true, false) => base,
        (true, true) => {
            product = Some(base.unwrap_or_default().mul(&xform.unwrap_or_default()));
            &mut product
        }
    };
    {
        let src = &meshes[mesh_id.0 as usize];
        let mut epts: Vec<[f64; 3]> = Vec::new();
        let mut fpts: Vec<[f64; 3]> = Vec::new();
        for &e in &src.edges {
            let ed = src.edge(e);
            epts.push(pos3(src, ed.v0));
            epts.push(pos3(src, ed.v1));
            b.edge_flags.push(EdgeFlags { soft: ed.soft, curve: ed.curve, layer: None, hidden: false });
        }
        for &f in &src.faces {
            let (outer, holes) = face_rings(src, f);
            b.ring_counts.push(1 + holes.len());
            b.ring_sizes.push(outer.len());
            fpts.extend(outer);
            for h in holes {
                b.ring_sizes.push(h.len());
                fpts.extend(h);
            }
            let attrs = &src.face(f).attrs;
            b.attrs.push(match m.as_mut() {
                Some(m) if !attrs.is_empty() => transformed_attrs(attrs, m),
                _ => attrs.clone(),
            });
        }
        if let Some(m) = m.as_ref() {
            let (rot, trans) = np_affine(m);
            epts = affine_rows(&epts, &rot, &trans);
            fpts = affine_rows(&fpts, &rot, &trans);
        }
        b.edge_pts.extend(epts);
        b.face_pts.extend(fpts);
    }
    for c in children.iter_mut() {
        walk_merge(meshes, c, m, b);
    }
}

/// `world_mesh`: the mesh itself for a classic group, a transformed DEEP copy
/// for a component instance, the nested placements folded in.
pub fn world_mesh<'a>(meshes: &'a [Mesh], group: &mut Group, curve: &mut i64) -> Cow<'a, Mesh> {
    let src = &meshes[group.mesh.0 as usize];
    if group.children.is_empty() {
        return match group.xform.as_mut() {
            None => Cow::Borrowed(src),
            Some(m) => Cow::Owned(transformed_mesh(src, m, curve)),
        };
    }
    Cow::Owned(merged_placements(meshes, group, curve))
}

pub fn push_mesh(meshes: &mut Vec<Mesh>, mesh: Mesh) -> MeshId {
    meshes.push(mesh);
    MeshId(meshes.len() as u32 - 1)
}

/// `Group.materialize`: bake an instance into its OWN world-space mesh.
pub fn materialize(meshes: &mut Vec<Mesh>, group: &mut Group, counters: &mut Counters) {
    if group.xform.is_none() && group.children.is_empty() {
        return;
    }
    let mut frame = group_frame(group);
    // Without axes the frame IS the xform object, which the world mesh's
    // `m(r, c)` reads flag General.
    let aliased = group.axes.is_none() && group.xform.is_some();
    let world = world_mesh(meshes, group, &mut counters.curve).into_owned();
    if aliased {
        frame = group.xform;
    }
    group.mesh = push_mesh(meshes, world);
    group.xform = None;
    group.axes = frame; // baked, but it still faces its way
    group.children = Vec::new();
}

/// `Group.make_unique`: this instance stops sharing with its siblings. Returns
/// the group's former children — the list Python's undo keeps, its groups
/// touched by the copies made of them.
pub fn make_unique(meshes: &mut Vec<Mesh>, group: &mut Group, counters: &mut Counters) -> Vec<Group> {
        if group.is_component() {
        let copy = transformed_mesh(&meshes[group.mesh.0 as usize], &mut M4::identity(), &mut counters.curve);
        group.mesh = push_mesh(meshes, copy);
        let mut old = std::mem::take(&mut group.children);
        group.children = old.iter_mut().map(|c| independent_copy(meshes, c, counters)).collect();
        return old;
    }
    if group.children.is_empty() {
        materialize(meshes, group, counters);
        return Vec::new();
    }
    let copy = transformed_mesh(&meshes[group.mesh.0 as usize], &mut M4::identity(), &mut counters.curve);
    group.mesh = push_mesh(meshes, copy);
    let mut old = std::mem::take(&mut group.children);
    group.children = old.iter_mut().map(|c| independent_copy(meshes, c, counters)).collect();
    // Nothing shared any more: what is left is a group of groups.
    group.component = false;
    old
}

pub fn truthy_dict(v: &Option<Value>) -> Option<Value> {
    match v {
        Some(Value::Object(o)) if !o.is_empty() => v.clone(),
        _ => None,
    }
}

/// `copy_group`: a pastable duplicate of `group`, optionally translated by
/// `delta`. An instance stays an instance sharing the prototype; a classic
/// group gets a deep mesh copy. Inside a component everything is shared, so
/// a classic child becomes an identity placement first — in the SOURCE, as
/// Python does.
pub fn copy_group(meshes: &mut Vec<Mesh>, group: &mut Group, delta: Option<V3>, in_definition: bool, counters: &mut Counters) -> Group {
    let share = in_definition || (group.xform.is_some() && group.component);
    if share {
        for child in &mut group.children {
            if child.xform.is_none() {
                child.xform = Some(M4::identity());
                child.component = false; // it was a group, it stays one
            }
        }
    }
    let mut t = M4::identity();
    if let Some(d) = delta {
        t.translate(d.0, d.1, d.2);
    }
    let mut g;
    if let Some(x) = group.xform {
        g = Group::new(group.mesh, group.name.clone());
        g.xform = Some(t.mul(&x));
        // The local axes live in the mesh's space, which the copy shares.
        if let Some(a) = group.axes {
            g.axes = Some(a);
        }
    } else {
        let copy = transformed_mesh(&meshes[group.mesh.0 as usize], &mut t, &mut counters.curve);
        let id = push_mesh(meshes, copy);
        g = Group::new(id, group.name.clone());
        if let Some(a) = group.axes {
            g.axes = Some(t.mul(&a));
        }
    }
    g.uid = counters.new_uid();
    g.layer = group.layer.clone();
    g.ifc = truthy_dict(&group.ifc);
    g.billboard = group.billboard.clone();
    g.text3d = truthy_dict(&group.text3d);
    g.ext = truthy_dict(&group.ext);
    g.hidden = group.hidden;
    g.locked = group.locked;
    g.material = truthy_dict(&group.material);
    g.component = group.component;
    g.exploded = truthy_dict(&group.exploded);
    g.explode_offset = group.explode_offset;
    // Nested placements ride along untranslated: `delta` already moved the
    // parent, and a child's transform is relative to it.
    g.children = group.children.iter_mut().map(|c| copy_group(meshes, c, None, share, counters)).collect();
    g
}

/// `_independent_copy`: `copy_group` without the sharing; the copy keeps the
/// uid — it REPLACES the original in its parent.
pub fn independent_copy(meshes: &mut Vec<Mesh>, group: &mut Group, counters: &mut Counters) -> Group {
    let mut g = copy_group(meshes, group, None, false, counters);
    if group.xform.is_some() {
        let copy = transformed_mesh(&meshes[group.mesh.0 as usize], &mut M4::identity(), &mut counters.curve);
        g.mesh = push_mesh(meshes, copy);
    }
    g.uid = group.uid.clone();
    g.children = group.children.iter_mut().map(|c| independent_copy(meshes, c, counters)).collect();
    g
}

/// `effective_material`: the paint a placement's default faces show — its
/// own, else the top-level object's it belongs to (`owner`, set only on the
/// viewport's placement proxies). `None` for an unpainted one.
pub fn effective_material<'a>(group: Option<&'a Group>, owner: Option<&'a Group>) -> Option<&'a Value> {
    let group = group?;
    let own = |g: &'a Group| match &g.material {
        Some(Value::Object(o)) if !o.is_empty() => g.material.as_ref(),
        _ => None,
    };
    if let Some(m) = own(group) {
        return Some(m);
    }
    match owner {
        Some(o) if !std::ptr::eq(o, group) => own(o),
        _ => None,
    }
}
