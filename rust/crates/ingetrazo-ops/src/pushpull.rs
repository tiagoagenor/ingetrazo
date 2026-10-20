// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! The Push/Pull operation (`tools/pushpull.py`, the commit pipeline).
//!
//! Python commits it as `SnapshotMutation(self._mutate_or_top_out)`, a
//! closure over the tool's drag state; here it is a proper [`Command`],
//! [`PushPull`], carrying exactly the drag state the mutation reads. Its
//! undo restores the snapshot taken before (`Mesh::capture_state` /
//! `restore_state`, identity-preserving), and its redo restores the
//! snapshot taken after, as `SnapshotMutation` does.
//!
//! The pipeline (the module docstring of `tools/pushpull.py`):
//!
//! 1. orient the mesh outward (entry invariant; no-op on open sheets);
//! 2. build the *naive* extrusion: consumed/kept base, moved cap (with
//!    holes), one wall quad per boundary/hole edge;
//! 3. stitch connectivity (T-junctions, collinear vertices);
//! 4. on a solid, rebuild every touched plane from its edges
//!    ([`crate::cap_rebuild`]); on a flat sheet, the seeded coplanar merge;
//! 5. orient outward again.
//!
//! A **BIM-grade refusal guard** wraps it: a push on a closed solid must
//! leave a closed solid (or collapse it flat), else the mesh is restored
//! and the push refused; a refused distance is bisected down to the
//! furthest one the guard accepts (`_mutate_or_top_out`).
//!
//! ORDER (reported). Python iterates sets of `Face`/`Edge` objects in
//! address order in a few places; every one of them is order-free here
//! except where noted in `cap_rebuild` (which face's centroid stands as a
//! plane's origin). Python's `before_verts = {id(v) ...}` can see a freed
//! vertex's address reused by a new one; ids here are never reused.

use std::cell::{Cell, RefCell};
use std::collections::{HashMap, HashSet};
use std::rc::Rc;

use ingetrazo_formats::pyjson::Value;
use ingetrazo_geom::meshface;
use ingetrazo_math::{Hypot, V3};
use ingetrazo_model::mesh::{key as mesh_key, EdgeId, FaceId, Mesh, State, VertexId, PAINT_KEYS};
use ingetrazo_model::pyval;
use ingetrazo_model::scene::Scene;
use ingetrazo_model::stitch::{run_stitch, PruneOrphanEdgesCommand};
use ingetrazo_model::topology::{self, classify_push_edge, key as topo_key, loop_inside_face, mesh_is_flat, positions, refine_loop_with_points, subtract_loop_from_face, PFace, PushEdge};

use crate::arrangement::{interior_point, plane_basis, point_in_polygon};
use crate::cap_rebuild::{apply_rebuild, crack_planes, edges_along, key_bits, key_cmp, plane_key, seam_planes, PlaneKey, RebuildCache};
use crate::command::{Command, Cx, Fail};
use crate::draw::{AddEdgeCommand, AddFaceCommand, DeleteFaceCommand};
use crate::orient::{is_closed, orient_outward_with};
use crate::transform::translate_points;

/// The smallest extrusion that still moves geometry: anything below the
/// mesh's weld resolution would weld the top loop onto the base loop and
/// degenerate the wall quads. Treated as a no-op push.
pub const MIN_EXTRUDE: f64 = 2e-4;

/// How many halvings the commit tries when the guard refuses the distance
/// the user dragged to.
pub const CLAMP_TRIES: usize = 6;

/// The dihedral above which a swept seam reads as a *curve* facet (a
/// cylinder's side) rather than a real corner. cos(31.8 deg).
pub const CURVE_FACET_COS: f64 = 0.85;

fn dotf(a: V3, b: V3) -> f64 {
    a.dot(b) as f64
}

fn lenf(a: V3, h: Hypot) -> f64 {
    a.length(h) as f64
}

/// `Face.normal()` of a mesh face.
fn fnormal(m: &Mesh, f: FaceId, h: Hypot) -> V3 {
    m.face_normal(f, h)
}

/// `Face.centroid()`.
fn fcentroid(m: &Mesh, f: FaceId) -> V3 {
    meshface::centroid(&positions(m, &m.face(f).outer))
}

// ---- helpers ported from the module level of tools/pushpull.py -------------

/// `_project_loop_2d`.
pub fn project_loop_2d(lp: &[V3], normal: V3, h: Hypot) -> Vec<(f64, f64)> {
    let (u, v) = plane_basis(normal, h);
    lp.iter().map(|&p| (dotf(p, u), dotf(p, v))).collect()
}

/// `_loops_overlap_2d`: whether two loops projected to the same plane
/// overlap laterally (holes ignored).
pub fn loops_overlap_2d(a: &[(f64, f64)], b: &[(f64, f64)]) -> bool {
    if point_in_polygon(interior_point(a), b) {
        return true;
    }
    if point_in_polygon(interior_point(b), a) {
        return true;
    }
    a.iter().any(|&p| point_in_polygon(p, b)) || b.iter().any(|&p| point_in_polygon(p, a))
}

/// `_component_is_closed`: whether `face`'s edge-connected component is
/// watertight.
pub fn component_is_closed(m: &Mesh, face: FaceId) -> bool {
    let mut seen: HashSet<FaceId> = HashSet::new();
    seen.insert(face);
    let mut stack = vec![face];
    while let Some(f) = stack.pop() {
        let fc = m.face(f);
        for lp in std::iter::once(&fc.outer).chain(fc.holes.iter()) {
            let n = lp.len();
            for i in 0..n {
                let Some(e) = m.find_edge(lp[i], lp[(i + 1) % n]) else { continue };
                if m.edge(e).faces.len() < 2 {
                    return false;
                }
                for &g in &m.edge(e).faces {
                    if seen.insert(g) {
                        stack.push(g);
                    }
                }
            }
        }
    }
    true
}

/// `_swept_by_push`: whether `neighbour` has an edge parallel to the push.
pub fn swept_by_push(neighbour: &PFace, push_normal: V3, h: Hypot) -> bool {
    let n = push_normal.normalized(h);
    let verts = &neighbour.vertices;
    let m = verts.len();
    for i in 0..m {
        let e = verts[(i + 1) % m].sub(verts[i]);
        if lenf(e, h) > 1e-9 && dotf(e.normalized(h), n).abs() > 0.999 {
            return true;
        }
    }
    false
}

/// `_paint_of`: the base face's paint, ready to stamp on the geometry a push
/// creates (the texture's per-face `uvw` is dropped and `planar` says so).
pub fn paint_of(attrs: &[(String, Value)]) -> Vec<(String, Value)> {
    let mut paint: Vec<(String, Value)> = Vec::new();
    for k in PAINT_KEYS {
        if let Some((_, v)) = attrs.iter().find(|(n, _)| n == k) {
            paint.push((k.to_string(), v.clone()));
        }
    }
    if let Some(slot) = paint.iter_mut().find(|(k, _)| k == "texture") {
        if pyval::truthy(&slot.1) {
            if let Value::Object(items) = &slot.1 {
                let mut out: Vec<(String, Value)> = items.iter().filter(|(k, _)| k != "uvw").cloned().collect();
                pyval::set(&mut out, "planar", Value::Bool(true));
                slot.1 = Value::Object(out);
            }
        }
    }
    paint
}

/// `_seam_planes_along`: `(origin, normal)` of each plane where an edge along
/// `segments` parts two coplanar faces facing the same way with the same paint.
pub fn seam_planes_along(m: &Mesh, segments: &[(V3, V3)], h: Hypot) -> Vec<(V3, V3)> {
    let mut out = Vec::new();
    for e in edges_along(m, segments, h) {
        let faces = &m.edge(e).faces;
        if faces.len() != 2 {
            continue;
        }
        let (fa, fb) = (faces[0], faces[1]);
        if m.face(fa).interior || m.face(fb).interior {
            continue;
        }
        if !pyval::eq(&Value::Object(m.face(fa).attrs.clone()), &Value::Object(m.face(fb).attrs.clone())) {
            continue;
        }
        if dotf(fnormal(m, fa, h).normalized(h), fnormal(m, fb, h).normalized(h)) < 0.9999 {
            continue;
        }
        out.push((fcentroid(m, fa), fnormal(m, fa, h)));
    }
    out
}

/// `_faces_window`: the box around `faces` grown by 1 cm (plus a relative
/// term far from the origin), or `None` for none.
pub fn faces_window(m: &Mesh, faces: &[FaceId]) -> Option<[f64; 6]> {
    let (mut xs, mut ys, mut zs) = (Vec::new(), Vec::new(), Vec::new());
    for &f in faces {
        let fc = m.face(f);
        for lp in std::iter::once(&fc.outer).chain(fc.holes.iter()) {
            for &v in lp {
                let p = m.position(v);
                xs.push(p.0 as f64);
                ys.push(p.1 as f64);
                zs.push(p.2 as f64);
            }
        }
    }
    if xs.is_empty() {
        return None;
    }
    let mn = |v: &[f64]| v.iter().copied().fold(f64::INFINITY, f64::min);
    let mx = |v: &[f64]| v.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let lo = [mn(&xs), mn(&ys), mn(&zs)];
    let hi = [mx(&xs), mx(&ys), mx(&zs)];
    let big = lo.iter().chain(hi.iter()).map(|c| c.abs()).fold(0.0, f64::max);
    let mg = 0.01 + 1e-6 * big;
    Some([lo[0] - mg, lo[1] - mg, lo[2] - mg, hi[0] + mg, hi[1] + mg, hi[2] + mg])
}

/// `_curve_runs`: the base's curves as runs of positions.
pub fn curve_runs(m: &Mesh, face: FaceId) -> Vec<(Vec<V3>, bool)> {
    let mut runs: Vec<(Vec<V3>, bool)> = Vec::new();
    let fc = m.face(face);
    for lp in std::iter::once(&fc.outer).chain(fc.holes.iter()) {
        let n = lp.len();
        if n < 2 {
            continue;
        }
        let ids: Vec<Option<i64>> = (0..n).map(|i| m.find_edge(lp[i], lp[(i + 1) % n]).and_then(|e| m.edge(e).curve)).collect();
        if ids[0].is_some() && ids.iter().all(|c| *c == ids[0]) {
            runs.push((lp.iter().map(|&v| m.position(v)).collect(), true));
            continue;
        }
        let Some(start) = (0..n).find(|&i| ids[i] != ids[(i + n - 1) % n]) else { continue };
        let mut i = start;
        for _ in 0..n {
            if ids[i].is_some() && ids[i] != ids[(i + n - 1) % n] {
                let mut pts = vec![m.position(lp[i])];
                let mut j = i;
                while ids[j] == ids[i] && pts.len() <= n {
                    j = (j + 1) % n;
                    pts.push(m.position(lp[j]));
                }
                runs.push((pts, false));
            }
            i = (i + 1) % n;
        }
    }
    runs
}

/// `_touched_faces`: the faces carrying one of the operation's seed
/// positions; `None` (the whole mesh) when the op left none.
pub fn touched_faces(m: &Mesh, seedkeys: &HashSet<[i64; 3]>) -> Option<Vec<FaceId>> {
    let hit: Vec<FaceId> = m.faces.iter().copied().filter(|&f| m.face(f).outer.iter().any(|&v| seedkeys.contains(&topo_key(m.position(v))))).collect();
    if hit.is_empty() {
        None
    } else {
        Some(hit)
    }
}

/// `_rebuild_planes_fixpoint`: recompute each touched plane's faces from its
/// edges, rebuilt to a fixpoint (see the Python docstring).
pub fn rebuild_planes_fixpoint(m: &mut Mesh, mut fresh: HashSet<FaceId>, seedkeys: &HashSet<[i64; 3]>, keep_mode: bool, removing: bool, h: Hypot) {
    // The push's own rims, captured by *position* before any rebuild.
    let mut op_rims: Vec<(V3, V3)> = Vec::new();
    let mut fresh_sorted: Vec<FaceId> = fresh.iter().copied().collect();
    fresh_sorted.sort();
    for &f in &fresh_sorted {
        let fc = m.face(f);
        for lp in std::iter::once(&fc.outer).chain(fc.holes.iter()) {
            let cnt = lp.len();
            for i in 0..cnt {
                op_rims.push((m.position(lp[i]), m.position(lp[(i + 1) % cnt])));
            }
        }
    }
    let mut cache = RebuildCache::new();
    for _ in 0..4 {
        cache.seed_faces = fresh.clone();
        let mut planes: Vec<(PlaneKey, V3, V3)> = Vec::new();
        let mut seen: HashSet<[u64; 4]> = HashSet::new();
        let mut add = |planes: &mut Vec<(PlaneKey, V3, V3)>, origin: V3, n: V3| {
            let k = plane_key(origin, n, h).0;
            if seen.insert(key_bits(&k)) {
                planes.push((k, origin, n));
            }
        };
        for (_, origin, n) in seam_planes(m, &fresh, h) {
            add(&mut planes, origin, n);
        }
        let mut fl: Vec<FaceId> = fresh.iter().copied().collect();
        fl.sort();
        for (_, origin, n) in crack_planes(m, faces_window(m, &fl), h) {
            add(&mut planes, origin, n);
        }
        // A strip the last round dropped as inside the solid (#94) leaves its
        // rim as a crease in the plane beside it.
        for (origin, n) in seam_planes_along(m, &cache.dropped, h) {
            add(&mut planes, origin, n);
        }
        cache.dropped.clear();
        planes.sort_by(|a, b| key_cmp(&a.0, &b.0));
        let mut changed = false;
        for (_, origin, plane_n) in planes {
            let before_faces: HashSet<FaceId> = m.faces.iter().copied().collect();
            if apply_rebuild(m, origin, plane_n, &fresh, keep_mode, removing, true, &mut op_rims, &mut cache, h) {
                changed = true;
                for &f in &m.faces {
                    if !before_faces.contains(&f) {
                        fresh.insert(f);
                    }
                }
            }
        }
        if !changed {
            break;
        }
        let live: HashSet<FaceId> = m.faces.iter().copied().collect();
        fresh.retain(|f| live.contains(f));
    }
    // One more connectivity pass collapses the redundant collinear vertices
    // the dissolved seams leave. No merge: rebuild already did.
    run_stitch(m, seedkeys, None, false, true, h);
}

// ---- tool-side classification (used by the drag and by the command) --------

/// `(attached, prism_cap)` of `_classify_base`.
pub fn classify_base(m: &Mesh, face: FaceId, h: Hypot) -> (bool, bool) {
    let faces = m.faces.clone();
    let pf = PFace::of(m, face);
    let mut kinds: Vec<PushEdge> = Vec::new();
    for lp in std::iter::once(&pf.vertices).chain(pf.holes.iter()) {
        let n = lp.len();
        for i in 0..n {
            kinds.push(classify_push_edge(m, face, lp[i], lp[(i + 1) % n], &faces, h));
        }
    }
    let normal = pf.normal(h);
    let attached = kinds.iter().all(|k| !matches!(k, PushEdge::Free));
    let prism_cap = !kinds.is_empty()
        && kinds.iter().all(|k| match k {
            PushEdge::Perp(nb) => swept_by_push(&PFace::of(m, *nb), normal, h),
            _ => false,
        });
    (attached, prism_cap)
}

/// `_surface_normal`: `face`'s normal, turned to agree with its coplanar
/// neighbours.
pub fn surface_normal(m: &Mesh, face: FaceId, h: Hypot) -> V3 {
    let n = fnormal(m, face, h).normalized(h);
    let on_loop: HashSet<VertexId> = m.face(face).outer.iter().copied().collect();
    let (mut agree, mut disagree) = (0, 0);
    for &v in &m.face(face).outer {
        for &e in &m.vertex_ref(v).edges {
            let ed = m.edge(e);
            if !on_loop.contains(&ed.v0) || !on_loop.contains(&ed.v1) {
                continue; // not a boundary edge
            }
            for &other in &ed.faces {
                if other == face {
                    continue;
                }
                let d = dotf(fnormal(m, other, h).normalized(h), n);
                if d.abs() > 0.999 {
                    if d > 0.0 {
                        agree += 1;
                    } else {
                        disagree += 1;
                    }
                }
            }
        }
    }
    if disagree > agree {
        n.neg()
    } else {
        n
    }
}

/// `_outward_normal`: the normal a drag signs `extrusion` along.
pub fn outward_normal(m: &Mesh, face: FaceId, h: Hypot) -> V3 {
    if is_closed(m) || mesh_is_flat(m, h) {
        return fnormal(m, face, h);
    }
    surface_normal(m, face, h)
}

/// `_cap_loop_positions`: every boundary position of `face`.
pub fn cap_loop_positions(m: &Mesh, face: FaceId) -> Vec<V3> {
    let fc = m.face(face);
    let mut pts: Vec<V3> = fc.outer.iter().map(|&v| m.position(v)).collect();
    for hl in &fc.holes {
        pts.extend(hl.iter().map(|&v| m.position(v)));
    }
    pts
}

/// Moller-Trumbore, `core.orient._ray_triangle`: the hit distance or `None`.
pub fn ray_triangle(origin: V3, direction: V3, tri: &[V3; 3]) -> Option<f64> {
    const EPS: f64 = 1e-7;
    let e1 = tri[1].sub(tri[0]);
    let e2 = tri[2].sub(tri[0]);
    let p = direction.cross(e2);
    let det = dotf(e1, p);
    if det.abs() < EPS {
        return None;
    }
    let inv = 1.0 / det;
    let t = origin.sub(tri[0]);
    let u = dotf(t, p) * inv;
    if u < -EPS || u > 1.0 + EPS {
        return None;
    }
    let q = t.cross(e1);
    let v = dotf(direction, q) * inv;
    if v < -EPS || u + v > 1.0 + EPS {
        return None;
    }
    let dist = dotf(e2, q) * inv;
    if dist > 1e-6 {
        Some(dist)
    } else {
        None
    }
}

/// `_compute_inward_limit`: how far the face can be pushed *into* the solid
/// before the sweep would shoot past blocking geometry; `None` = unbounded.
pub fn compute_inward_limit(m: &Mesh, base: FaceId, normal: Option<V3>, anchor: Option<V3>, attached: bool, h: Hypot) -> Option<f64> {
    if !attached {
        return None;
    }
    let mut limit: Option<f64> = None;
    let n = normal.unwrap_or_else(|| fnormal(m, base, h)).normalized(h);
    let anchor = anchor.unwrap_or_else(|| fcentroid(m, base));
    let pb = PFace::of(m, base);
    let base_xy = project_loop_2d(&pb.vertices, n, h);
    for &g in &m.faces {
        if g == base {
            continue;
        }
        let gn = fnormal(m, g, h).normalized(h);
        if dotf(gn, n).abs() < 0.999 {
            continue; // not parallel
        }
        let dist = dotf(anchor.sub(fcentroid(m, g)), n);
        if dist <= 1e-6 {
            continue; // coplanar or on the outward side
        }
        let back_loop: Vec<V3> = pb.vertices.iter().map(|&v| v.sub(n.mul(dist as f32))).collect();
        let pg = PFace::of(m, g);
        if loop_inside_face(&pg, &back_loop, false, h) {
            continue; // a through-hole target, not a blocker
        }
        if !loops_overlap_2d(&base_xy, &project_loop_2d(&pg.vertices, n, h)) {
            continue; // laterally elsewhere
        }
        if limit.is_none_or(|l| dist < l) {
            limit = Some(dist);
        }
    }

    // Immediate lateral exit: a backing wall that leans away from the push.
    let direction = n.neg();
    let fc = m.face(base);
    for lp in std::iter::once(&fc.outer).chain(fc.holes.iter()) {
        let cnt = lp.len();
        for i in 0..cnt {
            let Some(ed) = m.find_edge(lp[i], lp[(i + 1) % cnt]) else { continue };
            for &g in &m.edge(ed).faces {
                if g == base || m.face(g).interior {
                    continue;
                }
                let gn = fnormal(m, g, h).normalized(h);
                if dotf(gn, n).abs() > 0.999 {
                    continue; // parallel/coplanar: not a lateral wall
                }
                if dotf(direction, gn) > 1e-6 {
                    return Some(0.0);
                }
            }
        }
    }

    // Lateral exit bound: march inward from points just inside the base loop
    // and take the first crossing of any non-parallel boundary face.
    let loop3 = pb.vertices.clone();
    let (u, w) = plane_basis(n, h);
    let origin3 = loop3[0];
    let loop_xy: Vec<(f64, f64)> = loop3.iter().map(|&p| (dotf(p.sub(origin3), u), dotf(p.sub(origin3), w))).collect();
    let ip_xy = interior_point(&loop_xy);
    let ip3 = origin3.add(u.mul(ip_xy.0 as f32)).add(w.mul(ip_xy.1 as f32));
    let mut samples = vec![ip3];
    for (i, &p) in loop3.iter().enumerate() {
        samples.push(p.add(ip3.sub(p).mul(1e-3)));
        let mid = p.add(loop3[(i + 1) % loop3.len()]).mul(0.5);
        samples.push(mid.add(ip3.sub(mid).mul(1e-3)));
    }
    let mut lateral: Option<f64> = None;
    for &g in &m.faces {
        if g == base || m.face(g).interior {
            continue;
        }
        if dotf(fnormal(m, g, h).normalized(h), n).abs() > 0.999 {
            continue; // parallel: blocker/through logic owns it
        }
        for tri in PFace::of(m, g).triangulate(h) {
            for &q in &samples {
                if let Some(t) = ray_triangle(q, direction, &tri) {
                    if t > 1e-4 && lateral.is_none_or(|l| t < l) {
                        lateral = Some(t);
                    }
                }
            }
        }
    }
    if let Some(l) = lateral {
        // A lateral contact is an edge-on graze: stop a hair short.
        let l = ingetrazo_math::pyfloat::py_max2(l - 1e-3, 0.0);
        if limit.is_none_or(|x| l < x) {
            limit = Some(l);
        }
    }
    limit
}

// ---- the command ---------------------------------------------------------

/// The drag state `_mutate` reads, and the verdict it leaves.
#[derive(Clone, Debug)]
pub struct PushPull {
    pub base_face: FaceId,
    /// Signed distance along the normal.
    pub extrusion: f64,
    /// `tool._normal`; `None` falls back to the face's own.
    pub normal: Option<V3>,
    pub keep_base: bool,
    pub prism_cap: bool,
    pub drag_pre_oriented: bool,
    pub cap_positions: Vec<V3>,
    /// The guard rolled the push back (or it digested itself).
    pub refused: bool,
    /// The dragged distance was reduced to the furthest one the guard took.
    pub topped_out: bool,
}

impl PushPull {
    pub fn new(base_face: FaceId, extrusion: f64) -> PushPull {
        PushPull { base_face, extrusion, normal: None, keep_base: false, prism_cap: false, drag_pre_oriented: false, cap_positions: Vec::new(), refused: false, topped_out: false }
    }

    /// `_mutate_or_top_out`.
    pub fn mutate_or_top_out(&mut self, scene: &mut Scene, cx: &mut Cx) {
        self.mutate(scene, cx);
        if !self.refused {
            return;
        }
        let wanted = self.extrusion;
        let (mut lo, mut hi) = (0.0f64, 1.0f64);
        let mut best: Option<f64> = None;
        for _ in 0..CLAMP_TRIES {
            let mid = (lo + hi) / 2.0;
            let snap = cx.mesh(scene).capture_state();
            self.extrusion = wanted * mid;
            self.mutate(scene, cx);
            if self.refused || self.extrusion.abs() < MIN_EXTRUDE {
                cx.mesh_mut(scene).restore_state(&snap);
                hi = mid;
            } else {
                best = Some(self.extrusion);
                cx.mesh_mut(scene).restore_state(&snap);
                lo = mid;
            }
        }
        let Some(best) = best else {
            self.extrusion = wanted;
            self.refused = true;
            return;
        };
        self.extrusion = best;
        self.mutate(scene, cx);
        self.topped_out = true;
    }

    /// `_mesh_fingerprint`.
    fn fingerprint(m: &Mesh) -> (usize, usize, Vec<(i64, i64, i64)>) {
        let mut keys: Vec<(i64, i64, i64)> = m.vertices.iter().map(|&v| mesh_key(m.position(v))).collect();
        keys.sort();
        (m.faces.len(), m.edges.len(), keys)
    }

    /// `_mutate`: apply the push and stitch watertight, with the refusal
    /// guard.
    pub fn mutate(&mut self, scene: &mut Scene, cx: &mut Cx) {
        let h = cx.hypot;
        self.refused = false;
        let m = cx.mesh(scene);
        let before_edges: HashSet<EdgeId> = m.edges.iter().copied().collect();
        // Pushing a tagged face EXTENDS its BIM object. Capture before the
        // mutation: the base is consumed.
        let base_attrs = m.face(self.base_face).attrs.clone();
        let base_tag: Option<Value> = base_attrs.iter().find(|(k, _)| k == "ifc").filter(|(_, v)| pyval::truthy(v)).map(|(_, v)| v.clone());
        let base_paint = paint_of(&base_attrs);
        let before_verts: Option<HashSet<VertexId>> = if base_tag.is_some() || !base_paint.is_empty() { Some(m.vertices.iter().copied().collect()) } else { None };
        let runs = curve_runs(m, self.base_face);
        let guard: Option<State> = if is_closed(m) && !mesh_is_flat(m, h) { Some(m.capture_state()) } else { None };
        let fp_before = guard.as_ref().map(|_| PushPull::fingerprint(m));
        self.mutate_inner(scene, cx);
        let m = cx.mesh_mut(scene);
        if let Some(g) = &guard {
            if !(is_closed(m) || mesh_is_flat(m, h)) {
                m.restore_state(g);
                self.refused = true;
                return;
            }
        }
        if let Some(fp) = &fp_before {
            if self.extrusion.abs() >= MIN_EXTRUDE && PushPull::fingerprint(m) == *fp {
                // The rebuild digested the push back to the starting state:
                // geometrically a no-op. Flag it.
                self.refused = true;
                return;
            }
        }
        self.soften_curve_facets(m, &before_edges, h);
        self.tag_moved_curves(m, &runs, &mut cx.counters.curve, h);
        if base_tag.is_some() || !base_paint.is_empty() {
            // Only faces carrying at least one vertex this push created: the
            // extrusion's own walls and cap.
            let before_verts = before_verts.unwrap();
            let m = cx.mesh_mut(scene);
            let mut fresh: Vec<FaceId> = Vec::new();
            for &v in &m.vertices.clone() {
                if !before_verts.contains(&v) {
                    fresh.extend(m.faces_at(v));
                }
            }
            fresh.sort();
            fresh.dedup();
            for f in fresh {
                let attrs = &mut m.face_mut(f).attrs;
                if let Some(t) = &base_tag {
                    let has = attrs.iter().find(|(k, _)| k == "ifc").is_some_and(|(_, v)| pyval::truthy(v));
                    if !has {
                        pyval::set(attrs, "ifc", t.clone());
                    }
                }
                if !base_paint.is_empty() && !PAINT_KEYS.iter().any(|k| attrs.iter().any(|(n, _)| n == k)) {
                    for (k, v) in &base_paint {
                        pyval::set(attrs, k, v.clone());
                    }
                }
            }
        }
    }

    /// `_soften_curve_facets`.
    fn soften_curve_facets(&self, m: &mut Mesh, before_edges: &HashSet<EdgeId>, h: Hypot) {
        for e in m.edges.clone() {
            let ed = m.edge(e);
            if before_edges.contains(&e) || ed.soft || ed.faces.len() != 2 {
                continue;
            }
            let d = dotf(fnormal(m, ed.faces[0], h).normalized(h), fnormal(m, ed.faces[1], h).normalized(h));
            if CURVE_FACET_COS < d && d < 0.99995 {
                m.edge_mut(e).soft = true;
            }
        }
    }

    /// `_tag_moved_curves`: the base's curves, carried to the moved cap.
    fn tag_moved_curves(&self, m: &mut Mesh, runs: &[(Vec<V3>, bool)], counter: &mut i64, _h: Hypot) {
        let Some(normal) = self.normal else { return };
        if runs.is_empty() {
            return;
        }
        let shift = normal.mul(self.extrusion as f32);
        for (pts, closed) in runs {
            let moved: Vec<Option<VertexId>> = pts.iter().map(|&p| m.vertex_at(p.add(shift))).collect();
            if moved.iter().any(|v| v.is_none()) {
                continue;
            }
            let moved: Vec<VertexId> = moved.into_iter().flatten().collect();
            let mut pairs: Vec<(VertexId, VertexId)> = moved.windows(2).map(|w| (w[0], w[1])).collect();
            if *closed {
                pairs.push((moved[moved.len() - 1], moved[0]));
            }
            let edges: Vec<Option<EdgeId>> = pairs.iter().map(|&(a, b)| m.find_edge(a, b)).collect();
            if edges.iter().any(|e| e.is_none_or(|e| m.edge(e).curve.is_some())) {
                continue;
            }
            let cid = Mesh::next_curve_id(counter);
            for e in edges.into_iter().flatten() {
                m.edge_mut(e).curve = Some(cid);
            }
        }
    }

    /// `_mutate_inner`: apply the push to the target mesh and stitch it
    /// watertight.
    fn mutate_inner(&mut self, scene: &mut Scene, cx: &mut Cx) {
        let h = cx.hypot;
        let face = self.base_face;
        let mut d = self.extrusion;

        // A prism cap is exactly a translation of the cap along its normal;
        // the walls follow through shared vertices.
        if self.prism_cap && !self.keep_base {
            let m = cx.mesh_mut(scene);
            let normal = self.normal.unwrap_or_else(|| fnormal(m, face, h));
            let keys: HashSet<ingetrazo_model::mesh::Key> = self.cap_positions.iter().map(|&p| mesh_key(p)).collect();
            translate_points(m, &keys, normal.mul(d as f32));
            let mut seed: Vec<V3> = self.cap_positions.clone();
            seed.extend(self.cap_positions.iter().map(|&p| p.add(normal.mul(d as f32))));
            let seedkeys: HashSet<[i64; 3]> = seed.iter().map(|&p| topo_key(p)).collect();
            let mut none: HashSet<FaceId> = HashSet::new();
            run_stitch(m, &seedkeys, Some(&mut none), false, true, h);
            // A shrink can land the cap flush on a coplanar neighbour: run the
            // same fixpoint plane rebuild as the extrude path.
            if !mesh_is_flat(m, h) {
                let fresh: HashSet<FaceId> = m.faces.iter().copied().filter(|&f| m.face(f).outer.iter().any(|&v| seedkeys.contains(&topo_key(m.position(v))))).collect();
                rebuild_planes_fixpoint(m, fresh, &seedkeys, false, d < 0.0, h);
            } else {
                // A full collapse flattened the solid: the flat-drawing heal.
                topology::heal_overlapping_faces(m, 0.5, None, None, h, &mut |mm: &mut Mesh| {
                    crate::orient::orient_outward(mm, h);
                });
            }
            let touched = touched_faces(m, &seedkeys);
            orient_outward_with(m, h, 12345, touched.as_deref());
            return;
        }

        // Establish the outward invariant up front.
        let m = cx.mesh_mut(scene);
        let entry_closed = is_closed(m);
        let pre_normal = self.normal.unwrap_or_else(|| fnormal(m, face, h));
        if entry_closed && !self.drag_pre_oriented {
            orient_outward_with(m, h, 12345, Some(&[face]));
        }
        let normal = fnormal(m, face, h);
        if dotf(normal, pre_normal) < 0.0 {
            d = -d;
        }
        let allverts: Vec<V3> = m.vertices.iter().map(|&v| m.position(v)).collect();
        let pface = PFace::of(m, face);
        // Split the base loop at any existing vertex sitting on one of its
        // edges so each sub-edge is carried by one face and classifies right.
        let base = refine_loop_with_points(&pface.vertices, &allverts, topology::SPLIT_TOLERANCE, h);
        let off = normal.mul(d as f32);
        let top: Vec<V3> = base.iter().map(|&v| v.add(off)).collect();
        let base_holes: Vec<Vec<V3>> = pface.holes.iter().map(|hl| refine_loop_with_points(hl, &allverts, topology::SPLIT_TOLERANCE, h)).collect();
        let top_holes: Vec<Vec<V3>> = base_holes.iter().map(|hl| hl.iter().map(|&v| v.add(off)).collect()).collect();
        let count = base.len();
        let faces = m.faces.clone();
        let kinds: Vec<PushEdge> = (0..count).map(|i| classify_push_edge(m, face, base[i], base[(i + 1) % count], &faces, h)).collect();
        // Ctrl ("push/pull a copy") forces the free-extrusion semantics.
        let attached_any = kinds.iter().all(|k| !matches!(k, PushEdge::Free));
        let attached = attached_any && !self.keep_base;
        let through = if attached { self.find_through_face(m, face, d, &faces, h) } else { None };
        // "Solid" must be a property of the REGION being pushed: a base whose
        // rim backs onto ONLY coplanar faces and whose component has open
        // edges is pure sheet.
        let mut sheet_fast = !entry_closed && kinds.iter().all(|k| matches!(k, PushEdge::Coplanar(_)));
        if sheet_fast {
            'holes: for hl in &base_holes {
                let n = hl.len();
                for i in 0..n {
                    if !matches!(classify_push_edge(m, face, hl[i], hl[(i + 1) % n], &faces, h), PushEdge::Coplanar(_)) {
                        sheet_fast = false;
                        break 'holes;
                    }
                }
            }
        }
        if sheet_fast && component_is_closed(m, face) {
            sheet_fast = false;
        }
        let was_solid = !mesh_is_flat(m, h) && !sheet_fast;
        if was_solid && !entry_closed && !self.drag_pre_oriented {
            // Open mixed mesh taking the solid path: mark interior partitions.
            orient_outward_with(m, h, 12345, Some(&[face]));
        }

        let before: HashSet<FaceId> = m.faces.iter().copied().collect();
        let base_attrs = m.face(face).attrs.clone();
        let mut cap_face: Option<Rc<Cell<Option<FaceId>>>> = None;
        let mut far: Option<(Rc<Cell<Option<FaceId>>>, Vec<(String, Value)>)> = None;
        let commands: Vec<Box<dyn Command>> = if let Some((far_face, back_loop, remainder)) = through {
            let (cmds, far_cmd) = Self::through_commands(m, face, &base, far_face, &back_loop, remainder);
            far = Some((far_cmd, m.face(far_face).attrs.clone()));
            cmds
        } else {
            let (cmds, cap) = Self::extrude_commands(&pface, &base, &top, &base_holes, &top_holes, count, attached);
            cap_face = cap;
            cmds
        };
        for mut cmd in commands {
            let _ = cmd.apply(scene, cx);
        }
        let m = cx.mesh_mut(scene);
        if !base_attrs.is_empty() {
            if let Some(cf) = cap_face.as_ref().and_then(|c| c.get()) {
                if m.faces.contains(&cf) {
                    // The moved cap continues the (consumed) base: same attrs.
                    m.face_mut(cf).attrs = base_attrs.clone();
                }
            }
        }
        if let Some((ff, attrs)) = &far {
            if !attrs.is_empty() {
                if let Some(f) = ff.get() {
                    if m.faces.contains(&f) {
                        m.face_mut(f).attrs = attrs.clone();
                    }
                }
            }
        }
        let mut new_faces: HashSet<FaceId> = m.faces.iter().copied().filter(|f| !before.contains(f)).collect();
        if self.keep_base && attached_any && d < 0.0 {
            // The Ctrl-stack grows into the solid: its cap is a deliberate
            // interior division.
            let mut nf: Vec<FaceId> = new_faces.iter().copied().collect();
            nf.sort();
            for f in nf {
                let fnr = fnormal(m, f, h).normalized(h);
                if dotf(fnr, normal).abs() > 0.999 && dotf(fcentroid(m, f).sub(top[0]), normal).abs() < 1e-4 {
                    m.face_mut(f).interior = true;
                }
            }
        }
        let mut seed: Vec<V3> = base.clone();
        seed.extend(top.iter().copied());
        for (hb, ht) in base_holes.iter().zip(top_holes.iter()) {
            seed.extend(hb.iter().copied());
            seed.extend(ht.iter().copied());
        }
        let seedkeys: HashSet<[i64; 3]> = seed.iter().map(|&p| topo_key(p)).collect();
        // On a solid, the naive extrude's seams and overlaps are dissolved by
        // the per-plane rebuild, so the winding-tolerant coplanar merge stays
        // off.
        let solid = attached_any && was_solid;
        run_stitch(m, &seedkeys, Some(&mut new_faces), !solid && !self.keep_base, !solid, h);
        if solid {
            rebuild_planes_fixpoint(m, new_faces, &seedkeys, self.keep_base, d < 0.0, h);
        }
        // Give any closed solid a consistent outward orientation.
        let touched = touched_faces(m, &seedkeys);
        if !sheet_fast || is_closed(m) || touched.as_ref().is_some_and(|t| component_is_closed(m, t[0])) {
            orient_outward_with(m, h, 12345, touched.as_deref());
        }
    }

    /// `_extrude_commands`: the naive extrusion of `face` to `top`.
    #[allow(clippy::type_complexity)]
    fn extrude_commands(
        face: &PFace,
        base: &[V3],
        top: &[V3],
        base_holes: &[Vec<V3>],
        top_holes: &[Vec<V3>],
        count: usize,
        attached: bool,
    ) -> (Vec<Box<dyn Command>>, Option<Rc<Cell<Option<FaceId>>>>) {
        let mut commands: Vec<Box<dyn Command>> = Vec::new();
        if attached {
            commands.push(Box::new(DeleteFaceCommand::new(face.vertices.clone()))); // base consumed into the solid
        }
        for i in 0..count {
            commands.push(Box::new(AddEdgeCommand::new(top[i], top[(i + 1) % count], None, None)));
        }
        for i in 0..count {
            commands.push(Box::new(AddEdgeCommand::new(base[i], top[i], None, None)));
        }
        let holes: Vec<Vec<V3>> = top_holes.to_vec();
        let cap_cmd = AddFaceCommand::new(top.to_vec(), !attached, if holes.is_empty() { None } else { Some(holes) }, None);
        let cap = cap_cmd.face.clone();
        commands.push(Box::new(cap_cmd));
        // Inner walls: each hole edge raises a wall to the moved cap's opening.
        for (hb, ht) in base_holes.iter().zip(top_holes.iter()) {
            let hn = hb.len();
            for i in 0..hn {
                commands.push(Box::new(AddEdgeCommand::new(ht[i], ht[(i + 1) % hn], None, None)));
            }
            for i in 0..hn {
                commands.push(Box::new(AddEdgeCommand::new(hb[i], ht[i], None, None)));
            }
            for i in 0..hn {
                let j = (i + 1) % hn;
                commands.push(Box::new(AddFaceCommand::new(vec![hb[i], hb[j], ht[j], ht[i]], false, None, None)));
            }
        }
        for i in 0..count {
            let j = (i + 1) % count;
            commands.push(Box::new(AddFaceCommand::new(vec![base[i], base[j], top[j], top[i]], false, None, None)));
        }
        if attached {
            commands.push(Box::new(PruneOrphanEdgesCommand::new(base)));
        }
        (commands, Some(cap))
    }

    /// `_find_through_face`: if pushing `face` by `d` reaches a face parallel
    /// to it on the far side, `(far_face, back_loop, remainder)`.
    #[allow(clippy::type_complexity)]
    fn find_through_face(&self, m: &Mesh, face: FaceId, d: f64, faces: &[FaceId], h: Hypot) -> Option<(FaceId, Vec<V3>, Option<Vec<V3>>)> {
        let fnr = fnormal(m, face, h).normalized(h);
        let push = if d > 0.0 { fnr } else { fnr.neg() };
        let origin = fcentroid(m, face);
        let pf = PFace::of(m, face);
        let mut best: Option<(f64, FaceId, Vec<V3>, Option<Vec<V3>>)> = None;
        let pushd = push.mul(d.abs() as f32);
        let mut swept: Vec<V3> = pf.vertices.clone();
        swept.extend(pf.vertices.iter().map(|&v| v.add(pushd)));
        let mx = |f: fn(&V3) -> f32| swept.iter().map(|p| f(p) as f64).fold(f64::NEG_INFINITY, f64::max);
        let mn = |f: fn(&V3) -> f32| swept.iter().map(|p| f(p) as f64).fold(f64::INFINITY, f64::min);
        let big = swept.iter().flat_map(|p| [p.0, p.1, p.2]).map(|c| (c as f64).abs()).fold(0.0, f64::max);
        let mg = 0.01 + 1e-6 * big;
        let w0 = [mn(|p| p.0) - mg, mn(|p| p.1) - mg, mn(|p| p.2) - mg];
        let w1 = [mx(|p| p.0) + mg, mx(|p| p.1) + mg, mx(|p| p.2) + mg];
        for &g in faces {
            if g == face {
                continue;
            }
            let gv = positions(m, &m.face(g).outer);
            let gmax = |f: fn(&V3) -> f32| gv.iter().map(|p| f(p) as f64).fold(f64::NEG_INFINITY, f64::max);
            let gmin = |f: fn(&V3) -> f32| gv.iter().map(|p| f(p) as f64).fold(f64::INFINITY, f64::min);
            if gmax(|p| p.0) < w0[0] || gmin(|p| p.0) > w1[0] || gmax(|p| p.1) < w0[1] || gmin(|p| p.1) > w1[1] || gmax(|p| p.2) < w0[2] || gmin(|p| p.2) > w1[2] {
                continue; // nowhere near what the push sweeps
            }
            if dotf(fnr, fnormal(m, g, h).normalized(h)).abs() < 0.999 {
                continue; // not parallel
            }
            let dist = dotf(fcentroid(m, g).sub(origin), push);
            if dist <= 1e-4 || dist > d.abs() + 1e-4 {
                continue; // coplanar / behind / farther than the push reaches
            }
            let pd = push.mul(dist as f32);
            let back_loop: Vec<V3> = pf.vertices.iter().map(|&v| v.add(pd)).collect();
            let pg = PFace::of(m, g);
            if loop_inside_face(&pg, &back_loop, false, h) {
                if best.as_ref().is_none_or(|b| dist < b.0) {
                    best = Some((dist, g, back_loop, None));
                }
                continue;
            }
            // Landing FLUSH on the far face with the opening on its rim: the
            // far face is trimmed to what is left of it and the cap goes.
            if (dist - d.abs()).abs() <= 1e-4 {
                if let Some(rem) = subtract_loop_from_face(&pg, &back_loop, h) {
                    if best.as_ref().is_none_or(|b| dist < b.0) {
                        best = Some((dist, g, back_loop, Some(rem)));
                    }
                }
            }
        }
        best.map(|b| (b.1, b.2, b.3))
    }

    /// `_through_commands`: punch the far face, join it to the front opening
    /// with a tunnel, and sweep the dangling base edges.
    #[allow(clippy::type_complexity)]
    fn through_commands(m: &Mesh, face: FaceId, base: &[V3], far_face: FaceId, back_loop: &[V3], remainder: Option<Vec<V3>>) -> (Vec<Box<dyn Command>>, Rc<Cell<Option<FaceId>>>) {
        let count = base.len();
        let pf = PFace::of(m, face);
        let pfar = PFace::of(m, far_face);
        let mut commands: Vec<Box<dyn Command>> = vec![
            Box::new(DeleteFaceCommand::new(pf.vertices.clone())), // remove the pushed cap (window pane)
            Box::new(DeleteFaceCommand::new(pfar.vertices.clone())), // re-add the far face with a new hole
        ];
        let far_cmd;
        match remainder {
            None => {
                let mut holes: Vec<Vec<V3>> = pfar.holes.clone();
                holes.push(back_loop.to_vec());
                let c = AddFaceCommand::new(pfar.vertices.clone(), false, Some(holes), None);
                far_cmd = c.face.clone();
                commands.push(Box::new(c));
                for i in 0..count {
                    // back opening boundary
                    commands.push(Box::new(AddEdgeCommand::new(back_loop[i], back_loop[(i + 1) % count], None, None)));
                }
            }
            Some(rem) => {
                // A notch on the rim: the far face becomes what is left of it.
                let holes = pfar.holes.clone();
                let c = AddFaceCommand::new(rem, false, if holes.is_empty() { None } else { Some(holes) }, None);
                far_cmd = c.face.clone();
                commands.push(Box::new(c));
            }
        }
        for i in 0..count {
            // tunnel verticals
            commands.push(Box::new(AddEdgeCommand::new(base[i], back_loop[i], None, None)));
        }
        for i in 0..count {
            // tunnel walls
            let j = (i + 1) % count;
            commands.push(Box::new(AddFaceCommand::new(vec![base[i], base[j], back_loop[j], back_loop[i]], false, None, None)));
        }
        let mut pts = base.to_vec();
        pts.extend(back_loop.iter().copied());
        commands.push(Box::new(PruneOrphanEdgesCommand::new(&pts)));
        (commands, far_cmd)
    }
}

/// The commit of a push: `SnapshotMutation(self._mutate_or_top_out)`.
pub struct PushPullCommand {
    /// Shared with the tool that committed it: the verdict (refused, topped
    /// out, the distance finally pushed) is read back from here.
    pub push: Rc<RefCell<PushPull>>,
    before: Option<State>,
    after: Option<State>,
}

impl PushPullCommand {
    pub fn new(push: Rc<RefCell<PushPull>>) -> PushPullCommand {
        PushPullCommand { push, before: None, after: None }
    }
}

impl Command for PushPullCommand {
    fn name(&self) -> &'static str {
        "SnapshotMutation"
    }

    fn apply(&mut self, scene: &mut Scene, cx: &mut Cx) -> Result<(), Fail> {
        if self.after.is_none() {
            self.before = Some(cx.mesh(scene).capture_state());
            self.push.borrow_mut().mutate_or_top_out(scene, cx);
            self.after = Some(cx.mesh(scene).capture_state());
        } else {
            let s = self.after.clone().expect("after");
            cx.mesh_mut(scene).restore_state(&s);
        }
        Ok(())
    }

    fn undo(&mut self, scene: &mut Scene, cx: &mut Cx) {
        if let Some(b) = &self.before {
            cx.mesh_mut(scene).restore_state(b);
        }
    }

    fn as_any(&self) -> Option<&dyn std::any::Any> {
        Some(self)
    }
}

#[allow(dead_code)]
fn _unused(_: &HashMap<u8, u8>) {}

/// `SnapshotMutation.from_states`: a mutation already applied to the model,
/// given its two snapshots - `apply` takes the redo branch (restores
/// `after`) and nothing is re-run. Push/Pull commits its staged preview this
/// way: the model already shows the result.
pub struct SnapshotStates {
    before: State,
    after: State,
}

impl SnapshotStates {
    pub fn new(before: State, after: State) -> SnapshotStates {
        SnapshotStates { before, after }
    }
}

impl Command for SnapshotStates {
    fn name(&self) -> &'static str {
        "SnapshotMutation"
    }

    fn apply(&mut self, scene: &mut Scene, cx: &mut Cx) -> Result<(), Fail> {
        cx.mesh_mut(scene).restore_state(&self.after);
        Ok(())
    }

    fn undo(&mut self, scene: &mut Scene, cx: &mut Cx) {
        cx.mesh_mut(scene).restore_state(&self.before);
    }
}

/// `_compute_plane_levels`: the levels (along the normal, both sides) of
/// every face parallel to the base that overlaps it laterally - where a
/// staged preview's topology changes. Empty for a base that is never staged.
pub fn compute_plane_levels(m: &Mesh, base: FaceId, normal: V3, anchor: V3, attached: bool, prism_cap: bool, h: Hypot) -> Vec<f64> {
    let mut levels = Vec::new();
    if !attached || prism_cap {
        return levels; // never staged: the overlay or the prism
    }
    let nn = normal.normalized(h);
    let base_xy = project_loop_2d(&PFace::of(m, base).vertices, nn, h);
    for &g in &m.faces {
        if g == base {
            continue;
        }
        if dotf(fnormal(m, g, h).normalized(h), nn).abs() <= 0.999 {
            continue; // not parallel
        }
        let level = dotf(fcentroid(m, g).sub(anchor), normal);
        if level.abs() <= topology::STITCH_TOL {
            continue; // the base's own plane: level 0 is always an event
        }
        if loops_overlap_2d(&base_xy, &project_loop_2d(&PFace::of(m, g).vertices, nn, h)) {
            levels.push(level);
        }
    }
    levels
}

/// `_slide_rate`: how a moving vertex travels per unit of push, keeping
/// every face in `faces` (the ones it drags that keep a fixed vertex) flat:
/// the push normal when they all contain it (a wall), otherwise the line the
/// slanted face shares with the others, scaled to keep pace with the cap.
/// `None` when no such line exists.
pub fn slide_rate(m: &Mesh, faces: &[FaceId], n: V3, h: Hypot) -> Option<V3> {
    let mut normals: Vec<V3> = Vec::new();
    for &f in faces {
        let mm = fnormal(m, f, h).normalized(h);
        if normals.iter().all(|&o| lenf(mm.cross(o), h) > 1e-3) {
            normals.push(mm);
        }
    }
    let nn = n.normalized(h);
    let slanted: Vec<V3> = normals.iter().copied().filter(|&mm| dotf(mm, nn).abs() >= 1e-3).collect();
    if slanted.is_empty() {
        return Some(n); // walls only: straight along n
    }
    let line = if normals.len() == 1 {
        let mm = normals[0];
        nn.sub(mm.mul(dotf(nn, mm) as f32))
    } else {
        normals[0].cross(normals[1])
    };
    let along = dotf(line, nn);
    if lenf(line, h) < 1e-9 || along.abs() < 1e-3 * lenf(line, h) {
        return None; // the line runs in the cap plane
    }
    let rate = line.mul((dotf(n, nn) / along) as f32);
    if normals.iter().any(|&mm| dotf(mm, rate).abs() > 1e-6 * lenf(rate, h)) {
        return None; // 3+ faces with no common line
    }
    Some(rate)
}
