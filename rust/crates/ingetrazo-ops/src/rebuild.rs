// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! `RebuildPlanarFacesCommand` and `RebuildPlaneFacesCommand`
//! (`core/history.py`): rebuild the faces of a flat drawing, or of one plane
//! of a 3D mesh, from the arrangement of its edges. Snapshot undo.

use std::collections::HashMap;

use ingetrazo_geom::triangulate::{newell, triangulate};
use ingetrazo_math::{Hypot, V3};
use ingetrazo_model::mesh::{EdgeFlags, FaceId, Mesh, State, VertexId};
use ingetrazo_model::pyval;
use ingetrazo_model::scene::Scene;
use ingetrazo_model::topology::{self, key, PFace};

use crate::arrangement::{coplanar_plane, planar_rebuild};
use crate::command::{CmdResult, Command, CommandError, Env};

const IN_TRI_SLACK: f64 = 1e-7;

/// `_point_in_tri`: coplanar point-in-triangle by consistent cross-product
/// orientation, measured as distances.
pub fn point_in_tri(p: V3, t0: V3, t1: V3, t2: V3, h: Hypot) -> bool {
    let n = t1.sub(t0).cross(t2.sub(t0));
    let nl = n.length(h) as f64;
    let ls = |a: V3| a.length_squared() as f64;
    let (l0, l1, l2) = (ls(t1.sub(t0)), ls(t2.sub(t1)), ls(t0.sub(t2)));
    // Python's `max(a, b, c)` keeps the first of equals.
    let mut longest = l0;
    if l1 > longest {
        longest = l1;
    }
    if l2 > longest {
        longest = l2;
    }
    if nl < 1e-12 || nl < 1e-9 * longest {
        return false;
    }
    for (a, b) in [(t0, t1), (t1, t2), (t2, t0)] {
        let c = b.sub(a).cross(p.sub(a));
        let side = b.sub(a).length(h) as f64;
        if side > 0.0 && (c.dot(n) as f64) / (nl * side) < -IN_TRI_SLACK {
            return false;
        }
    }
    true
}

fn edge_is_plain(m: &Mesh, e: ingetrazo_model::EdgeId) -> bool {
    let ed = m.edge(e);
    !ed.soft && ed.curve.is_none() && ed.layer.is_none() && !ed.hidden
}

fn stamp(m: &mut Mesh, e: ingetrazo_model::EdgeId, f: &EdgeFlags) {
    let ed = m.edge_mut(e);
    ed.soft = f.soft;
    ed.curve = f.curve;
    ed.layer = f.layer.clone();
    ed.hidden = f.hidden;
}

/// `RebuildPlanarFacesCommand`: rebuild the minimal faces of a flat
/// drawing from its edge graph. Only runs on a single-plane mesh; 3D is
/// left untouched.
#[derive(Default)]
pub struct RebuildPlanarFacesCommand {
    snapshot: Option<State>,
    pub rebuilt: usize,
    pub flat: bool,
}

impl RebuildPlanarFacesCommand {
    pub fn new() -> Self {
        RebuildPlanarFacesCommand { snapshot: None, rebuilt: 0, flat: true }
    }
}

impl Command for RebuildPlanarFacesCommand {
    fn apply(&mut self, scene: &mut Scene, env: &mut Env) -> CmdResult {
        let h = env.hypot;
        self.snapshot = Some(env.mesh(scene).capture_state());
        let m = env.mesh_mut(scene);
        let footprint = topology::winding_footprint(m, h);
        if m.edges.is_empty() {
            self.flat = false;
            return Ok(());
        }
        // Every edge endpoint must share one plane; 3D models are left alone.
        let verts: Vec<V3> = m.vertices.iter().map(|&v| m.position(v)).collect();
        let Some((mut origin, mut normal)) = coplanar_plane(&verts, 1e-4, h) else {
            self.flat = false;
            return Ok(());
        };
        if let Some(&f0) = m.faces.first() {
            normal = m.face_normal(f0, h);
            origin = m.position(m.face(f0).outer[0]);
        }
        let segments: Vec<(V3, V3)> = m.edges.iter().map(|&e| (m.position(m.edge(e).v0), m.position(m.edge(e).v1))).collect();
        // Remember flagged edges and face attrs so the rebuild preserves them.
        let flagged: Vec<(V3, V3, EdgeFlags)> = m
            .edges
            .iter()
            .copied()
            .filter(|&e| !edge_is_plain(m, e))
            .map(|e| {
                let ed = m.edge(e);
                (
                    m.position(ed.v0),
                    m.position(ed.v1),
                    EdgeFlags { soft: ed.soft, curve: ed.curve, layer: ed.layer.clone(), hidden: ed.hidden },
                )
            })
            .collect();
        let old_attrs: Vec<(Vec<[V3; 3]>, Vec<(String, ingetrazo_formats::pyjson::Value)>)> = m
            .faces
            .iter()
            .copied()
            .filter(|&f| !m.face(f).attrs.is_empty())
            .map(|f| (PFace::of(m, f).triangulate(h), m.face(f).attrs.clone()))
            .collect();
        let (edges, faces) = planar_rebuild(&segments, origin, normal, h);
        m.clear();
        for (a, b) in &edges {
            let e = m.add_edge(*a, *b).map_err(|_| CommandError::Failed("degenerate edge".into()))?;
            let (ea, eb) = (m.position(m.edge(e).v0), m.position(m.edge(e).v1));
            for (fa, fb, flags) in &flagged {
                if topology::point_on_seg_incl(ea, *fa, *fb, topology::SPLIT_TOLERANCE, h)
                    && topology::point_on_seg_incl(eb, *fa, *fb, topology::SPLIT_TOLERANCE, h)
                {
                    stamp(m, e, flags);
                    break;
                }
            }
        }
        for (outer, holes) in &faces {
            let f = m.add_face(outer, holes);
            if old_attrs.is_empty() {
                continue;
            }
            let tris = PFace::of(m, f).triangulate(h);
            let Some(t0) = tris.first() else { continue };
            let probe = t0[0].add(t0[1]).add(t0[2]).div(3.0);
            for (old_tris, attrs) in &old_attrs {
                if old_tris.iter().any(|t| point_in_tri(probe, t[0], t[1], t[2], h)) {
                    for (k, v) in attrs {
                        pyval::set(&mut m.face_mut(f).attrs, k, v.clone());
                    }
                    break;
                }
            }
        }
        // Every region comes out facing `normal`; each takes back the way
        // the face it lies in was facing.
        if let Some(fp) = &footprint {
            topology::orient_coplanar_faces(m, Some(fp), h);
        }
        let mut counter = env.curve;
        m.resplit_curves(&mut counter);
        env.curve = counter;
        self.rebuilt = faces.len();
        Ok(())
    }

    fn undo(&mut self, scene: &mut Scene, env: &mut Env) {
        if let Some(s) = &self.snapshot {
            env.mesh_mut(scene).restore_state(s);
        }
    }

    fn name(&self) -> &'static str {
        "RebuildPlanarFacesCommand"
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// `RebuildPlaneFacesCommand`: rebuild the faces of ONE plane of a
/// (possibly 3D) mesh from that plane's edge subgraph. A minimal region
/// keeps a face only if its interior was covered by an existing on-plane
/// face; uncovered regions stay empty.
pub struct RebuildPlaneFacesCommand {
    pub origin: V3,
    pub normal: V3,
    snapshot: Option<State>,
    pub rebuilt: usize,
}

const PLANE_TOL: f64 = 1e-4;

impl RebuildPlaneFacesCommand {
    /// Normalised by hand, as Python does: `QVector3D.normalized()` nulls
    /// anything shorter than 1e-5, and a zero normal here means "rebuild
    /// the plane of everything".
    pub fn new(origin: V3, normal: V3, h: Hypot) -> Self {
        let length = normal.length(h);
        let normal = if (length as f64) > 1e-9 { normal.div(length) } else { V3(0.0, 0.0, 0.0) };
        RebuildPlaneFacesCommand { origin, normal, snapshot: None, rebuilt: 0 }
    }

    /// The command as the recorder stores it: constructor arguments.
    pub fn from_args(origin: V3, normal: V3, h: Hypot) -> Self {
        Self::new(origin, normal, h)
    }

    fn on_plane(&self, p: V3) -> bool {
        (p.sub(self.origin).dot(self.normal) as f64).abs() < PLANE_TOL
    }
}

impl Command for RebuildPlaneFacesCommand {
    fn apply(&mut self, scene: &mut Scene, env: &mut Env) -> CmdResult {
        let h = env.hypot;
        self.snapshot = Some(env.mesh(scene).capture_state());
        if (self.normal.length_squared() as f64) < 0.5 {
            // No plane, nothing to rebuild.
            return Ok(());
        }
        let m = env.mesh_mut(scene);
        let plane_edges: Vec<ingetrazo_model::EdgeId> = m
            .edges
            .iter()
            .copied()
            .filter(|&e| self.on_plane(m.position(m.edge(e).v0)) && self.on_plane(m.position(m.edge(e).v1)))
            .collect();
        if plane_edges.is_empty() {
            return Ok(());
        }
        let on = |l: &[VertexId], m: &Mesh| l.iter().all(|&v| self.on_plane(m.position(v)));
        let plane_faces: Vec<FaceId> = m
            .faces
            .iter()
            .copied()
            .filter(|&f| on(&m.face(f).outer, m) && m.face(f).holes.iter().all(|hl| on(hl, m)))
            .collect();
        struct Old {
            tris: Vec<[V3; 3]>,
            normal: V3,
            attrs: Vec<(String, ingetrazo_formats::pyjson::Value)>,
        }
        let old: Vec<Old> = plane_faces
            .iter()
            .map(|&f| Old { tris: PFace::of(m, f).triangulate(h), normal: m.face_normal(f, h), attrs: m.face(f).attrs.clone() })
            .collect();
        let segments: Vec<(V3, V3)> = plane_edges.iter().map(|&e| (m.position(m.edge(e).v0), m.position(m.edge(e).v1))).collect();
        let (_, regions) = planar_rebuild(&segments, self.origin, self.normal, h);

        let ring = |l: &[V3]| -> Vec<topology::Key> {
            let mut v: Vec<topology::Key> = l.iter().map(|&p| key(p)).collect();
            v.sort();
            v.dedup();
            v
        };
        // The openings the plane's faces already have, and the outlines a
        // face was drawn exactly along.
        let mut outlines: HashMap<Vec<topology::Key>, usize> = HashMap::new();
        for &f in &plane_faces {
            *outlines.entry(ring(&PFace::of(m, f).vertices)).or_default() += 1;
        }
        let mut openings: HashMap<Vec<topology::Key>, Vec<Vec<topology::Key>>> = HashMap::new();
        for &f in &plane_faces {
            let pf = PFace::of(m, f);
            for hl in &pf.holes {
                let owners = openings.entry(ring(hl)).or_default();
                let r = ring(&pf.vertices);
                if !owners.contains(&r) {
                    owners.push(r);
                }
            }
        }
        let stays_open = |outer: &[V3]| -> bool {
            let r = ring(outer);
            let Some(owners) = openings.get(&r) else { return false };
            if outlines.get(&r).copied().unwrap_or(0) > 0 {
                return false;
            }
            !owners.iter().any(|o| outlines.get(o).copied().unwrap_or(0) > 1)
        };
        let covering = |outer: &[V3], holes: &[Vec<V3>]| -> Option<(V3, Vec<(String, ingetrazo_formats::pyjson::Value)>)> {
            let tris = triangulate(outer, holes, Some(self.normal), h);
            if tris.is_empty() {
                return None;
            }
            // The centroid of the largest triangle, not the first.
            let area2 = |t: &[V3; 3]| t[1].sub(t[0]).cross(t[2].sub(t[0])).length_squared();
            let mut big = &tris[0];
            for t in &tris[1..] {
                if area2(t) > area2(big) {
                    big = t;
                }
            }
            let probe = big[0].add(big[1]).add(big[2]).div(3.0);
            for o in &old {
                if o.tris.iter().any(|t| point_in_tri(probe, t[0], t[1], t[2], h)) {
                    return Some((o.normal, o.attrs.clone()));
                }
            }
            None
        };
        let mut keep: Vec<(Vec<V3>, Vec<Vec<V3>>, Vec<(String, ingetrazo_formats::pyjson::Value)>)> = Vec::new();
        for (mut outer, mut holes) in regions {
            if holes.is_empty() && stays_open(&outer) {
                continue;
            }
            let Some((nrm, attrs)) = covering(&outer, &holes) else { continue };
            // The arrangement emits CCW around self.normal; match the old face.
            if (newell(&outer, h).dot(nrm) as f64) < 0.0 {
                outer.reverse();
                for hl in holes.iter_mut() {
                    hl.reverse();
                }
            }
            keep.push((outer, holes, attrs));
        }
        for &f in &plane_faces {
            m.remove_face(f);
        }
        for (outer, holes, attrs) in &keep {
            let f = m.add_face(outer, holes);
            for (k, v) in attrs {
                pyval::set(&mut m.face_mut(f).attrs, k, v.clone());
            }
        }
        // Prune the collinear duplicates the re-added polygons left.
        topology::prune_collinear_orphan_edges(m, h);
        let mut counter = env.curve;
        m.resplit_curves(&mut counter);
        env.curve = counter;
        self.rebuilt = keep.len();
        Ok(())
    }

    fn undo(&mut self, scene: &mut Scene, env: &mut Env) {
        if let Some(s) = &self.snapshot {
            env.mesh_mut(scene).restore_state(s);
        }
    }

    fn name(&self) -> &'static str {
        "RebuildPlaneFacesCommand"
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}
