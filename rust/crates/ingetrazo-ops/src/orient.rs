// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! `core/orient.py`: consistent outward orientation for closed solids.
//!
//! Every face of a closed solid is wound so its normal points outward from
//! the enclosed volume. The mesh is a shared-vertex, non-manifold B-rep, so
//! orientation is decided per face by parity ray casting: a point just off
//! a face along its normal is outside the solid iff a ray from it crosses
//! the rest of the mesh an even number of times, voted over seven jittered
//! rays (`random.Random(12345)`, reproduced by `PyRandom`). Interior
//! partitions are peeled off iteratively (`face.interior`).
//!
//! The crossing counts come from `ingetrazo_geom::parity::Tris`, the exact
//! Möller–Trumbore test of `_ParityQuery.crossings_many` over a ray index
//! that only skips triangles the exact test refuses. Judged against its own
//! edge-connected component only, as in Python.

use std::collections::{HashMap, HashSet};

use ingetrazo_geom::meshface;
use ingetrazo_geom::parity::Tris;
use ingetrazo_math::{Hypot, V3};
use ingetrazo_model::mesh::{FaceId, Mesh};
use ingetrazo_model::topology::PFace;

use crate::arrangement::{plane_basis, point_in_polygon};
use crate::pyrandom::PyRandom;

/// Number of jittered rays voted over, and their angular spread.
const RAYS: usize = 7;
const JITTER: f64 = 3e-4;

/// `_all_coplanar`: every face on one plane (a 2D drawing).
pub fn all_coplanar(m: &Mesh, h: Hypot) -> bool {
    let pos = |f| m.face(f).outer.iter().map(|&v| m.position(v)).collect::<Vec<V3>>();
    let first = pos(m.faces[0]);
    let n = meshface::normal(&first, h);
    let o = meshface::centroid(&first);
    for &f in &m.faces {
        let lp = pos(f);
        let fnormal = meshface::normal(&lp, h);
        if (fnormal.dot(n) as f64).abs() < 0.999 {
            return false;
        }
        if (meshface::centroid(&lp).sub(o).dot(n) as f64).abs() > 1e-4 {
            return false;
        }
    }
    true
}

/// `is_closed`: no boundary, every edge borders at least two faces.
pub fn is_closed(m: &Mesh) -> bool {
    !m.faces.is_empty() && m.edges.iter().all(|&e| m.edge(e).faces.len() >= 2)
}

/// `_basis`: two unit vectors perpendicular to `n`.
fn basis(n: V3, h: Hypot) -> (V3, V3) {
    let mut reference = V3(1.0, 0.0, 0.0);
    if (reference.dot(n) as f64).abs() > 0.9 {
        reference = V3(0.0, 1.0, 0.0);
    }
    let u = n.cross(reference).normalized(h);
    let v = n.cross(u).normalized(h);
    (u, v)
}

/// `_face_components`: face index -> edge-connected component (union-find
/// through shared edges); the label is the root's index.
pub fn face_components(m: &Mesh) -> HashMap<FaceId, usize> {
    let idx: HashMap<FaceId, usize> = m.faces.iter().enumerate().map(|(i, &f)| (f, i)).collect();
    let mut parent: Vec<usize> = (0..m.faces.len()).collect();
    fn find(parent: &mut [usize], mut x: usize) -> usize {
        while parent[x] != x {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        x
    }
    for &e in &m.edges {
        let mut it = m.edge(e).faces.iter().filter_map(|f| idx.get(f).copied());
        let Some(first) = it.next() else { continue };
        for j in it {
            let (ra, rb) = (find(&mut parent, first), find(&mut parent, j));
            if ra != rb {
                parent[ra] = rb;
            }
        }
    }
    idx.iter().map(|(&f, &i)| (f, find(&mut parent, i))).collect()
}

/// `_on_face`: whether `p` (on the face's plane) lies inside its outer loop.
fn on_face(m: &Mesh, f: FaceId, p: V3, h: Hypot) -> bool {
    let n = m.face_normal(f, h);
    if (n.length(h) as f64) < 1e-12 {
        return true;
    }
    let (u, v) = plane_basis(n.normalized(h), h);
    let lp: Vec<(f64, f64)> = m
        .face(f)
        .outer
        .iter()
        .map(|&q| {
            let d = m.position(q).sub(p);
            let (x, y, z) = (d.0 as f64, d.1 as f64, d.2 as f64);
            (x * u.0 as f64 + y * u.1 as f64 + z * u.2 as f64, x * v.0 as f64 + y * v.1 as f64 + z * v.2 as f64)
        })
        .collect();
    point_in_polygon((0.0, 0.0), &lp)
}

/// `_interior_probe`: a point on the face's solid material.
fn interior_probe(m: &Mesh, f: FaceId, h: Hypot) -> V3 {
    let face = m.face(f);
    let centroid = |m: &Mesh| meshface::centroid(&face.outer.iter().map(|&v| m.position(v)).collect::<Vec<_>>());
    if face.holes.is_empty() {
        let c = centroid(m);
        if on_face(m, f, c, h) {
            return c;
        }
    }
    let tris = PFace::of(m, f).triangulate(h);
    if tris.is_empty() {
        return centroid(m);
    }
    let area2 = |t: &[V3; 3]| t[1].sub(t[0]).cross(t[2].sub(t[0])).length_squared();
    let mut big = &tris[0];
    for t in &tris[1..] {
        if area2(t) > area2(big) {
            big = t;
        }
    }
    big[0].add(big[1]).add(big[2]).div(3.0)
}

/// What the packed triangles of one component say, per face.
struct Component {
    tris: Tris,
    /// Row -> face index.
    row_face: Vec<i64>,
}

/// `ray_parity_outside`: whether `origin` is outside the volume bounded by
/// the (masked) triangles, by crossing parity along jittered rays around
/// `direction`; `None` on a tie.
fn ray_parity_outside(origin: V3, direction: V3, comp: &Component, mask: &[bool], skip: i64, rng: &mut PyRandom, h: Hypot) -> Option<bool> {
    ray_parity_outside_tris(origin, direction, &comp.tris, Some(mask), Some(skip), rng, h)
}

/// `ray_parity_outside` over a prebuilt triangle set (`packed=`), with the
/// optional `mask` over its rows and the `skip`ped face index. The per-plane
/// rebuild of a push/pull (`cap_rebuild`) asks it from sample points.
pub fn ray_parity_outside_tris(origin: V3, direction: V3, tris: &Tris, mask: Option<&[bool]>, skip: Option<i64>, rng: &mut PyRandom, h: Hypot) -> Option<bool> {
    let n = direction.normalized(h);
    let (u, v) = basis(n, h);
    let mut rays: Vec<V3> = Vec::with_capacity(RAYS);
    for r in 0..RAYS {
        if r == 0 {
            rays.push(n);
        } else {
            let a = rng.uniform(-JITTER, JITTER) as f32;
            let b = rng.uniform(-JITTER, JITTER) as f32;
            rays.push(n.add(u.mul(a)).add(v.mul(b)).normalized(h));
        }
    }
    let dirs: Vec<[f64; 3]> = rays.iter().map(|d| [d.0 as f64, d.1 as f64, d.2 as f64]).collect();
    let o = [origin.0 as f64, origin.1 as f64, origin.2 as f64];
    let counts = tris.crossings(o, &dirs, mask, skip);
    let (mut outside, mut inside) = (0, 0);
    for c in counts {
        if c % 2 == 0 {
            outside += 1;
        } else {
            inside += 1;
        }
    }
    if outside == inside {
        None
    } else {
        Some(outside > inside)
    }
}

#[derive(PartialEq, Eq, Clone, Copy)]
enum Side {
    Outward,
    Inward,
    Interior,
}

/// `_face_side_state`: `Outward` / `Inward` for a boundary face (by which
/// side is empty), `Interior` for a partition with material on both sides,
/// `None` when undecidable.
fn face_side_state(m: &Mesh, f: FaceId, comp: &Component, mask: &[bool], skip: i64, rng: &mut PyRandom, h: Hypot) -> Option<Side> {
    let n = m.face_normal(f, h);
    if (n.length(h) as f64) < 1e-9 {
        return None;
    }
    let probe = interior_probe(m, f, h);
    let ahead = ray_parity_outside(probe, n, comp, mask, skip, rng, h);
    if ahead != Some(false) {
        return if ahead == Some(true) { Some(Side::Outward) } else { None };
    }
    let behind = ray_parity_outside(probe, n.neg(), comp, mask, skip, rng, h);
    match behind {
        Some(true) => Some(Side::Inward),
        Some(false) => Some(Side::Interior),
        None => None,
    }
}

/// `_incoherent_edges`: how many edges two of `faces` both walk in the same
/// direction.
fn incoherent_edges(m: &Mesh, faces: &[FaceId]) -> usize {
    use ingetrazo_model::mesh::VertexId;
    let mut uses: HashMap<(VertexId, VertexId), Vec<(VertexId, VertexId)>> = HashMap::new();
    for &f in faces {
        let face = m.face(f);
        for lp in std::iter::once(&face.outer).chain(face.holes.iter()) {
            let n = lp.len();
            for i in 0..n {
                let (a, b) = (lp[i], lp[(i + 1) % n]);
                let key = if a <= b { (a, b) } else { (b, a) };
                uses.entry(key).or_default().push((a, b));
            }
        }
    }
    uses.values().filter(|u| u.len() == 2 && u[0] == u[1]).count()
}

/// `orient_outward(mesh, seed, only)`: flip the faces of a closed solid so
/// every boundary normal points outward, and mark the interior partitions
/// (`face.interior`). Returns the faces that were flipped. `only` restricts
/// the pass to the edge-connected components those faces belong to.
pub fn orient_outward_with(m: &mut Mesh, h: Hypot, seed: u64, only: Option<&[FaceId]>) -> Vec<FaceId> {
    if m.faces.is_empty() {
        return Vec::new();
    }
    if all_coplanar(m, h) {
        // A flat drawing has no volume at all: no partitions, nothing to flip.
        for f in m.faces.clone() {
            m.face_mut(f).interior = false;
        }
        return Vec::new();
    }
    let faces: Vec<FaceId> = m.faces.clone();
    let fidx: HashMap<FaceId, usize> = faces.iter().enumerate().map(|(i, &f)| (f, i)).collect();
    // Triangulate every face once and pack the rows per component.
    let comp_of = face_components(m);
    let mut comp_ids: Vec<usize> = comp_of.values().copied().collect();
    comp_ids.sort();
    comp_ids.dedup();
    let mut rows: HashMap<usize, (Vec<[f64; 3]>, Vec<[f64; 3]>, Vec<[f64; 3]>, Vec<i64>)> = HashMap::new();
    for (i, &f) in faces.iter().enumerate() {
        let entry = rows.entry(comp_of[&f]).or_default();
        for t in PFace::of(m, f).triangulate(h) {
            let p = |v: V3| [v.0 as f64, v.1 as f64, v.2 as f64];
            let (a, b, c) = (p(t[0]), p(t[1]), p(t[2]));
            entry.0.push(a);
            entry.1.push([b[0] - a[0], b[1] - a[1], b[2] - a[2]]);
            entry.2.push([c[0] - a[0], c[1] - a[1], c[2] - a[2]]);
            entry.3.push(i as i64);
        }
    }
    let comps: HashMap<usize, Component> = comp_ids
        .iter()
        .map(|&c| {
            let (a, e1, e2, face) = rows.remove(&c).unwrap_or_default();
            (c, Component { tris: Tris::new(a, e1, e2, face.clone()), row_face: face })
        })
        .collect();
    // The mask of one component's rows: the triangles of boundary faces.
    let boundary_mask = |comp: &Component, boundary: &HashSet<FaceId>| -> Vec<bool> {
        comp.row_face.iter().map(|&fi| boundary.contains(&faces[fi as usize])).collect()
    };

    // Faces this call decides.
    let mut scope: Vec<FaceId> = match only {
        None => faces.clone(),
        Some(only) => {
            let live: HashSet<usize> = only.iter().filter_map(|f| comp_of.get(f).copied()).collect();
            if live.is_empty() {
                return Vec::new();
            }
            faces.iter().copied().filter(|f| live.contains(&comp_of[f])).collect()
        }
    };

    let all: HashSet<FaceId> = faces.iter().copied().collect();
    let mut boundary: HashSet<FaceId> = all.clone();
    for _ in 0..4 {
        let mut rng = PyRandom::new(seed);
        let masks: HashMap<usize, Vec<bool>> = comps.iter().map(|(&c, comp)| (c, boundary_mask(comp, &boundary))).collect();
        let mut interior_now: HashSet<FaceId> = HashSet::new();
        for &f in &scope {
            let c = comp_of[&f];
            let state = face_side_state(m, f, &comps[&c], &masks[&c], fidx[&f] as i64, &mut rng, h);
            if state == Some(Side::Interior) {
                interior_now.insert(f);
            }
        }
        let prev_interior: HashSet<FaceId> = scope.iter().copied().filter(|f| !boundary.contains(f)).collect();
        if interior_now == prev_interior {
            break;
        }
        boundary = all.iter().copied().filter(|f| !interior_now.contains(f)).collect();
    }
    for &f in &scope {
        m.face_mut(f).interior = !boundary.contains(&f);
    }

    // Flipping needs a real outside, which only a closed mesh has.
    let mesh_closed = is_closed(m);
    if !mesh_closed {
        if only.is_none() {
            return Vec::new();
        }
        // A scoped pass judges closure per component: an open component has
        // no outside, so its faces are never turned.
        let mut open_comps: HashSet<usize> = HashSet::new();
        for &e in &m.edges {
            if m.edge(e).faces.len() == 1 {
                for f in &m.edge(e).faces {
                    if let Some(&c) = comp_of.get(f) {
                        open_comps.insert(c);
                    }
                }
            }
        }
        scope.retain(|f| !open_comps.contains(&comp_of[f]));
        if scope.is_empty() {
            return Vec::new();
        }
    }
    let mut rng = PyRandom::new(seed);
    let masks: HashMap<usize, Vec<bool>> = comps.iter().map(|(&c, comp)| (c, boundary_mask(comp, &boundary))).collect();
    let mut to_flip: Vec<FaceId> = Vec::new();
    for &f in &scope {
        if !boundary.contains(&f) {
            continue;
        }
        let c = comp_of[&f];
        if face_side_state(m, f, &comps[&c], &masks[&c], fidx[&f] as i64, &mut rng, h) == Some(Side::Inward) {
            to_flip.push(f);
        }
    }
    // Through an open mesh the flips are checked before they stay: a solid
    // whose shared edges disagree MORE after the flips than before was
    // misread by the parity rays, and keeps the windings it had.
    type Saved = (Vec<(String, ingetrazo_formats::pyjson::Value)>, Vec<ingetrazo_model::VertexId>, Vec<Vec<ingetrazo_model::VertexId>>);
    let mut saved: HashMap<FaceId, Saved> = HashMap::new();
    let mut before_bad: HashMap<usize, usize> = HashMap::new();
    let mut members: HashMap<usize, Vec<FaceId>> = HashMap::new();
    if !mesh_closed && !to_flip.is_empty() {
        for &f in &to_flip {
            let face = m.face(f);
            saved.insert(f, (face.attrs.clone(), face.outer.clone(), face.holes.clone()));
        }
        let flip_comps: HashSet<usize> = to_flip.iter().map(|f| comp_of[f]).collect();
        for &f in &scope {
            if flip_comps.contains(&comp_of[&f]) {
                members.entry(comp_of[&f]).or_default().push(f);
            }
        }
        for (&c, fs) in &members {
            before_bad.insert(c, incoherent_edges(m, fs));
        }
    }
    for &f in &to_flip {
        m.turn_face_over(f);
    }
    if !before_bad.is_empty() {
        let worse: HashSet<usize> = members.iter().filter(|(c, fs)| incoherent_edges(m, fs) > before_bad[c]).map(|(&c, _)| c).collect();
        if !worse.is_empty() {
            for &f in &to_flip {
                if worse.contains(&comp_of[&f]) {
                    let (attrs, lp, holes) = saved.remove(&f).expect("saved");
                    let face = m.face_mut(f);
                    face.attrs = attrs;
                    face.outer = lp;
                    face.holes = holes;
                }
            }
            to_flip.retain(|f| !worse.contains(&comp_of[f]));
        }
    }
    to_flip
}

/// `orient_outward(mesh)` with the default seed over the whole mesh.
pub fn orient_outward(m: &mut Mesh, h: Hypot) -> Vec<FaceId> {
    orient_outward_with(m, h, 12345, None)
}
