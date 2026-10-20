// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! The stitch pass (`core.history.run_stitch` and the commands around it) on
//! the arena [`Mesh`]: weld coincident vertices, drop stacked faces, resolve
//! T-junctions, collapse redundant collinear vertices, fuse coplanar faces.
//!
//! The mesh-level primitives live in `mesh.rs`; this module is the
//! orchestration and the three commands that drive it —
//! `StitchSolidCommand`, `CoplanarMergeCommand`, `PruneOrphanEdgesCommand`
//! — as plain structs with `apply`/`undo` over a `&mut Mesh` (the scene's
//! `version`/`selection` bookkeeping belongs to the Command layer).
//!
//! Seeds. Python passes the operation's positions as `core.topology._key`
//! tuples (`round(x, 4)` floats); here a key is the exact integer pair of
//! that float, `round_ties_even(x · 1e4)` (`meshface::ikey`): two points have
//! equal keys exactly when their float keys are equal.
//!
//! Order. Python iterates sets of faces (the coplanar component) and of
//! edges (`Vertex.edges`) in address order; here both go in ascending id
//! (creation order) — see the note on `Mesh::sorted_edges`.

use std::collections::{HashMap, HashSet};

use ingetrazo_geom::meshface::ikey;
use ingetrazo_math::{Hypot, V3};

use crate::mesh::{EdgeId, FaceId, Mesh, State, VertexId, STITCH_TOL};

pub type SeedKey = [i64; 3];

// ---- topology helpers ----------------------------------------------------------
// TODO(M3.2 merge): use crate::topology (`_point_on_seg_incl`, `same_position`,
// `_key`, `sweep_tjunctions`, `orphaned_edges_at`) instead of these copies.

/// `core.topology.same_position`: equal weld keys.
pub(crate) fn same_position(p: V3, q: V3) -> bool {
    ikey(p) == ikey(q)
}

/// `core.topology._point_on_seg_incl` (default tolerance): whether `pt` lies
/// on the segment `p`-`q`, endpoints included.
pub(crate) fn point_on_seg_incl(pt: V3, p: V3, q: V3, h: Hypot) -> bool {
    let pq = q.sub(p);
    let length = pq.length(h) as f64;
    if length < STITCH_TOL {
        return same_position(pt, p);
    }
    let t = (pt.sub(p).dot(pq) as f64) / (length * length);
    if t < -STITCH_TOL || t > 1.0 + STITCH_TOL {
        return false;
    }
    (pt.sub(p.add(pq.mul(t as f32))).length(h) as f64) < STITCH_TOL
}

/// `{_key(p) for p in positions}`.
pub fn seed_keys(positions: &[V3]) -> HashSet<SeedKey> {
    positions.iter().map(|&p| ikey(p)).collect()
}

fn unordered(a: SeedKey, b: SeedKey) -> (SeedKey, SeedKey) {
    if a <= b {
        (a, b)
    } else {
        (b, a)
    }
}

/// `core.topology.orphaned_edges_at`: edges with an end at one of `vertices`
/// that border no face.
pub fn orphaned_edges_at(mesh: &Mesh, vertices: &[V3]) -> Vec<EdgeId> {
    let vkeys = seed_keys(vertices);
    let mut face_edges: HashSet<(SeedKey, SeedKey)> = HashSet::new();
    for &f in &mesh.faces {
        let face = mesh.face(f);
        let loops: Vec<Vec<SeedKey>> = std::iter::once(&face.outer)
            .chain(face.holes.iter())
            .map(|l| l.iter().map(|&v| ikey(mesh.position(v))).collect())
            .collect();
        if !loops.iter().any(|lk| lk.iter().any(|k| vkeys.contains(k))) {
            continue;
        }
        for lk in &loops {
            let n = lk.len();
            for i in 0..n {
                face_edges.insert(unordered(lk[i], lk[(i + 1) % n]));
            }
        }
    }
    let mut out = Vec::new();
    for &e in &mesh.edges {
        let ed = mesh.edge(e);
        let (ka, kb) = (ikey(mesh.position(ed.v0)), ikey(mesh.position(ed.v1)));
        if (vkeys.contains(&ka) || vkeys.contains(&kb)) && !face_edges.contains(&unordered(ka, kb)) {
            out.push(e);
        }
    }
    out
}

/// `core.topology.sweep_tjunctions`: split every edge that runs past a
/// vertex of another face, round after round until none does. `seedkeys`
/// bound the sweep to the operation's box (`None`: the whole mesh).
///
/// Python does this as numpy array passes; this is the same test per
/// (edge, vertex) pair, in float64, on the same broad-phase bounds, and the
/// lowest-indexed qualifying vertex (in `mesh.vertices` order) splits each
/// edge. The hits of a round split in edge order.
pub fn sweep_tjunctions(mesh: &mut Mesh, seedkeys: Option<&HashSet<SeedKey>>) {
    let tol = STITCH_TOL;
    let tol2 = tol * tol;
    let seed: Option<([f64; 3], [f64; 3])> = seedkeys.filter(|s| !s.is_empty()).map(|s| {
        let mut lo = [f64::INFINITY; 3];
        let mut hi = [f64::NEG_INFINITY; 3];
        for k in s {
            for a in 0..3 {
                let c = k[a] as f64 / 1e4;
                lo[a] = lo[a].min(c);
                hi[a] = hi[a].max(c);
            }
        }
        ([lo[0] - tol, lo[1] - tol, lo[2] - tol], [hi[0] + tol, hi[1] + tol, hi[2] + tol])
    });
    loop {
        let verts = mesh.vertices.clone();
        let edges = mesh.edges.clone();
        if verts.is_empty() || edges.is_empty() {
            break;
        }
        let vpos: Vec<[f64; 3]> = verts
            .iter()
            .map(|&v| {
                let p = mesh.position(v);
                [p.0 as f64, p.1 as f64, p.2 as f64]
            })
            .collect();
        let slot: HashMap<VertexId, usize> = verts.iter().enumerate().map(|(i, &v)| (v, i)).collect();
        let mut best: Vec<Option<usize>> = vec![None; edges.len()];
        let mut any_live = false;
        for (ei, &e) in edges.iter().enumerate() {
            let ed = mesh.edge(e);
            let (i0, i1) = (slot[&ed.v0], slot[&ed.v1]);
            let (va, vb) = (vpos[i0], vpos[i1]);
            let mut elo = [0.0f64; 3];
            let mut ehi = [0.0f64; 3];
            let mut seg = [0.0f64; 3];
            for a in 0..3 {
                elo[a] = va[a].min(vb[a]);
                ehi[a] = va[a].max(vb[a]);
                seg[a] = vb[a] - va[a];
            }
            if let Some((slo, shi)) = &seed {
                let overlap = (0..3).all(|a| ehi[a] >= slo[a]) && (0..3).all(|a| elo[a] <= shi[a]);
                if !overlap {
                    continue;
                }
            }
            let l2 = seg[0] * seg[0] + seg[1] * seg[1] + seg[2] * seg[2];
            if !(l2 > tol2) {
                continue;
            }
            any_live = true;
            let tol_t = tol / l2.sqrt();
            for (j, p) in vpos.iter().enumerate() {
                if j == i0 || j == i1 {
                    continue;
                }
                if !(0..3).all(|a| p[a] >= elo[a] - tol && p[a] <= ehi[a] + tol) {
                    continue;
                }
                let rel = [p[0] - va[0], p[1] - va[1], p[2] - va[2]];
                let t = (rel[0] * seg[0] + rel[1] * seg[1] + rel[2] * seg[2]) / l2;
                if !(t > tol_t && t < 1.0 - tol_t) {
                    continue;
                }
                let d = [rel[0] - seg[0] * t, rel[1] - seg[1] * t, rel[2] - seg[2] * t];
                if d[0] * d[0] + d[1] * d[1] + d[2] * d[2] < tol2 {
                    best[ei] = Some(j);
                    break;
                }
            }
        }
        if !any_live {
            break;
        }
        let mut hit = false;
        for (ei, b) in best.iter().enumerate() {
            if let Some(j) = *b {
                hit = true;
                mesh.split_edge_at(edges[ei], verts[j]);
            }
        }
        if !hit {
            break;
        }
    }
}

// ---- the pass ------------------------------------------------------------------

/// `core.history._coplanar_component`: the maximal set of coplanar,
/// edge-connected faces touching the seed (a crease — an edge that also
/// carries a perpendicular face — is never crossed). Python returns a set;
/// this is ascending id.
fn coplanar_component(
    mesh: &Mesh,
    f0: FaceId,
    seedkeys: &HashSet<SeedKey>,
    ncache: &mut HashMap<FaceId, V3>,
    scache: &mut HashMap<FaceId, bool>,
    h: Hypot,
) -> Vec<FaceId> {
    let nrm = |f: FaceId, ncache: &mut HashMap<FaceId, V3>| -> V3 {
        *ncache.entry(f).or_insert_with(|| mesh.face_normal(f, h).normalized(h))
    };
    let seeded = |f: FaceId, scache: &mut HashMap<FaceId, bool>| -> bool {
        *scache.entry(f).or_insert_with(|| mesh.face(f).outer.iter().any(|&v| seedkeys.contains(&ikey(mesh.position(v)))))
    };
    if !seeded(f0, scache) {
        return Vec::new();
    }
    let n0 = nrm(f0, ncache);
    let mut comp: HashSet<FaceId> = HashSet::from([f0]);
    let mut stack = vec![f0];
    while let Some(f) = stack.pop() {
        let face = mesh.face(f);
        for l in std::iter::once(&face.outer).chain(face.holes.iter()) {
            let n = l.len();
            for i in 0..n {
                let (a, b) = (l[i], l[(i + 1) % n]);
                let e = match mesh.find_edge(a, b) {
                    Some(e) => e,
                    None => continue,
                };
                let faces = &mesh.edge(e).faces;
                if faces.iter().any(|&g| (n0.dot(nrm(g, ncache)) as f64).abs() < 0.999) {
                    continue;
                }
                for &g in faces {
                    if comp.contains(&g) || !seeded(g, scache) {
                        continue;
                    }
                    if (n0.dot(mesh.face_normal(g, h)) as f64).abs() > 0.999 {
                        comp.insert(g);
                        stack.push(g);
                    }
                }
            }
        }
    }
    let mut out: Vec<FaceId> = comp.into_iter().collect();
    out.sort();
    out
}

/// `core.history.run_stitch`: the three-phase watertight cleanup (no undo
/// bookkeeping — the caller snapshots or journals).
///
/// `new_faces` (when given) are the faces the operation created; phase 3
/// only fuses a component containing one of them, and the set is updated as
/// components merge. `None` merges any seeded component.
pub fn run_stitch(
    mesh: &mut Mesh,
    seedkeys: &HashSet<SeedKey>,
    mut new_faces: Option<&mut HashSet<FaceId>>,
    coplanar_merge: bool,
    dedupe: bool,
    h: Hypot,
) {
    // Phase 0 — weld coincident vertices, then drop stacked identical faces.
    mesh.weld_coincident();
    if dedupe {
        mesh.dedupe_faces();
    }
    // Phase 1 — resolve T-junctions, scoped to the operation's box.
    sweep_tjunctions(mesh, Some(seedkeys));
    // Phase 2 — collapse redundant valence-2 collinear vertices.
    loop {
        let mut collapsed = false;
        for v in mesh.vertices.clone() {
            if mesh.collapsible_vertex(v, h) {
                mesh.collapse_vertex(v);
                collapsed = true;
                break;
            }
        }
        if !collapsed {
            break;
        }
    }
    if !coplanar_merge {
        return;
    }
    // Phase 3 — coplanar-merge, seeded to the operation.
    loop {
        let mut merged = false;
        let mut ncache: HashMap<FaceId, V3> = HashMap::new();
        let mut scache: HashMap<FaceId, bool> = HashMap::new();
        for f0 in mesh.faces.clone() {
            let comp = coplanar_component(mesh, f0, seedkeys, &mut ncache, &mut scache, h);
            if comp.len() < 2 {
                continue;
            }
            if let Some(nf) = new_faces.as_deref() {
                if !comp.iter().any(|f| nf.contains(f)) {
                    continue;
                }
            }
            if let Some(region) = mesh.dissolve_coplanar_region(&comp, h) {
                if let Some(nf) = new_faces.as_deref_mut() {
                    for f in &comp {
                        nf.remove(f);
                    }
                    nf.insert(region);
                }
                merged = true;
                break;
            }
        }
        if !merged {
            break;
        }
    }
}

// ---- commands --------------------------------------------------------------------

/// `StitchSolidCommand`: make a solid watertight again after push/pull; undo
/// restores an identity-preserving snapshot taken before the pass.
#[derive(Clone, Debug)]
pub struct StitchSolidCommand {
    pub seed: Vec<V3>,
    snapshot: Option<State>,
}

impl StitchSolidCommand {
    pub fn new(seed: &[V3]) -> Self {
        StitchSolidCommand { seed: seed.to_vec(), snapshot: None }
    }

    pub fn apply(&mut self, mesh: &mut Mesh, h: Hypot) {
        self.snapshot = Some(mesh.capture_state());
        run_stitch(mesh, &seed_keys(&self.seed), None, true, true, h);
    }

    pub fn undo(&mut self, mesh: &mut Mesh) {
        if let Some(s) = &self.snapshot {
            mesh.restore_state(s);
        }
    }
}

/// `CoplanarMergeCommand`: dissolve the coplanar seams around the seed.
#[derive(Clone, Debug)]
pub struct CoplanarMergeCommand {
    pub seed: Vec<V3>,
    /// `(face_a, face_b, edge, merged_face)` per merge.
    pub merges: Vec<(FaceId, FaceId, EdgeId, FaceId)>,
}

impl CoplanarMergeCommand {
    pub fn new(seed: &[V3]) -> Self {
        CoplanarMergeCommand { seed: seed.to_vec(), merges: Vec::new() }
    }

    pub fn apply(&mut self, mesh: &mut Mesh, h: Hypot) {
        let seedkeys = seed_keys(&self.seed);
        let mut progress = true;
        while progress {
            progress = false;
            for edge in mesh.edges.clone() {
                let ed = mesh.edge(edge);
                if ed.faces.len() != 2 {
                    continue;
                }
                if !seedkeys.contains(&ikey(mesh.position(ed.v0))) && !seedkeys.contains(&ikey(mesh.position(ed.v1))) {
                    continue;
                }
                let (face_a, face_b) = (ed.faces[0], ed.faces[1]);
                let merged = match mesh.dissolve_edge(edge, h) {
                    Some(m) => m,
                    None => continue,
                };
                self.merges.push((face_a, face_b, edge, merged));
                progress = true;
                break;
            }
        }
    }

    pub fn undo(&mut self, mesh: &mut Mesh) {
        for &(face_a, face_b, edge, merged) in self.merges.iter().rev() {
            mesh.remove_face(merged);
            mesh.relink_edge(edge);
            mesh.relink_face(face_a);
            mesh.relink_face(face_b);
        }
        self.merges.clear();
    }
}

/// `PruneOrphanEdgesCommand`: remove the dangling edges at the seed vertices.
#[derive(Clone, Debug)]
pub struct PruneOrphanEdgesCommand {
    pub vertices: Vec<V3>,
    pub removed: Vec<EdgeId>,
}

impl PruneOrphanEdgesCommand {
    pub fn new(vertices: &[V3]) -> Self {
        PruneOrphanEdgesCommand { vertices: vertices.to_vec(), removed: Vec::new() }
    }

    pub fn apply(&mut self, mesh: &mut Mesh) {
        self.removed = orphaned_edges_at(mesh, &self.vertices);
        for &e in &self.removed {
            mesh.remove_edge(e);
        }
    }

    pub fn undo(&mut self, mesh: &mut Mesh) {
        for &e in &self.removed {
            mesh.relink_edge(e);
        }
        self.removed.clear();
    }
}
