// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! `core/edits.py`: the planner that turns drawn segments into history
//! commands — split the existing edges the segments cross, break each
//! segment at those crossings, weld duplicates, auto-face the planar cycles
//! the new pieces close — and the mesh-level Divide and Join Segments.
//!
//! The planner reads the mesh and emits a command list; it mutates nothing.
//! It mirrors what each emitted command will do in a local `simulated` edge
//! list and `faces_snapshot` face list, so planning and execution stay in
//! step. Python's edges and faces there are objects compared by identity: a
//! real one is named by its arena id, a simulated one by a fresh id.
//!
//! Python narrows every search to what can touch the segment (`_Nearby`, a
//! NumPy box filter); the plan it gives is the one the full lists give
//! (`tests/test_planner_box_filter.py`), so here the searches go over the
//! full lists and the filter is left out. That costs time on a model of
//! thousands of faces, never a different plan.

use std::collections::HashSet;

use ingetrazo_formats::pyjson::Value;
use ingetrazo_model::mesh::{EdgeFlags, EdgeId, Mesh};
use ingetrazo_model::scene::Scene;
use ingetrazo_model::topology::{self, key, CycleGraph, PFace, Seg};

use ingetrazo_math::{Hypot, V3};

use crate::arrangement::coplanar_plane;
use crate::command::{Command, Env};
use crate::draw::{AddEdgeCommand, AddFaceCommand, AutoTagDrawnFacesCommand, DeleteEdgesCommand, DeleteFaceCommand, SnapshotCompound};
use crate::rebuild::{RebuildPlanarFacesCommand, RebuildPlaneFacesCommand};

/// An edge of the planner's simulated edge list. A real mesh edge keeps its
/// arena id; a piece the plan makes gets a fresh one.
#[derive(Clone, Debug)]
struct SimEdge {
    id: u32,
    a: V3,
    b: V3,
    /// A simulated piece has neither flag (`getattr(edge, "soft", False)`).
    soft: bool,
    curve: Option<i64>,
}

/// A face of `faces_snapshot`: positions only (`core.geometry.Face` or the
/// mesh face read through `.vertices` / `.holes`).
#[derive(Clone, Debug)]
struct SimFace {
    vertices: Vec<V3>,
    holes: Vec<Vec<V3>>,
    attrs: Option<Vec<(String, Value)>>,
}

impl SimFace {
    fn pface(&self) -> PFace {
        PFace { vertices: self.vertices.clone(), holes: self.holes.clone() }
    }
}

type Segment = (V3, V3);

fn keyset(l: &[V3]) -> Vec<topology::Key> {
    let mut v: Vec<topology::Key> = l.iter().map(|&p| key(p)).collect();
    v.sort();
    v.dedup();
    v
}

/// `face_exists` over the snapshot's faces.
fn face_exists(faces: &[SimFace], cycle: &[V3]) -> bool {
    let want = keyset(cycle);
    faces.iter().any(|f| keyset(&f.vertices) == want)
}

/// `_loop_with_point`: a copy of `lp` with `point` inserted between the
/// first consecutive pair whose keys are `ka`/`kb` (either order); `None`
/// if there is no such boundary edge or `point` already is one of them.
fn loop_with_point(lp: &[V3], ka: topology::Key, kb: topology::Key, point: V3) -> Option<Vec<V3>> {
    let kp = key(point);
    let n = lp.len();
    for i in 0..n {
        let j = (i + 1) % n;
        let (ki, kj) = (key(lp[i]), key(lp[j]));
        if (ki == ka && kj == kb) || (ki == kb && kj == ka) {
            if kp == ki || kp == kj {
                return None;
            }
            let mut new = lp.to_vec();
            new.insert(i + 1, point);
            return Some(new);
        }
    }
    None
}

/// `split_edge_in_faces` over the snapshot: `(index, new_vertices)` for the
/// faces that gain `point` on the cut edge, leaving out those that carry all
/// of `skip` on their boundary (the face the drawn segment chord-splits).
fn split_edge_in_faces(faces: &[SimFace], edge_a: V3, edge_b: V3, point: V3, skip: &[V3], h: Hypot) -> Vec<(usize, Vec<V3>)> {
    let (ka, kb) = (key(edge_a), key(edge_b));
    let mut out = Vec::new();
    for (i, f) in faces.iter().enumerate() {
        if !skip.is_empty() && skip.iter().all(|&p| topology::locate_on_loop(&f.vertices, p, h).is_some()) {
            continue;
        }
        if let Some(nv) = loop_with_point(&f.vertices, ka, kb, point) {
            out.push((i, nv));
        }
    }
    out
}

// ---- face orientation ------------------------------------------------------------

/// `_newell`: the Newell vector of a loop, each term in double over the
/// float32 coordinates, rounded once to float32 by `QVector3D(...)`.
fn newell(lp: &[V3]) -> V3 {
    let (mut nx, mut ny, mut nz) = (0.0f64, 0.0f64, 0.0f64);
    let n = lp.len();
    for i in 0..n {
        let (p, q) = (lp[i], lp[(i + 1) % n]);
        let (px, py, pz) = (p.0 as f64, p.1 as f64, p.2 as f64);
        let (qx, qy, qz) = (q.0 as f64, q.1 as f64, q.2 as f64);
        nx += (py - qy) * (pz + qz);
        ny += (pz - qz) * (px + qx);
        nz += (px - qx) * (py + qy);
    }
    V3(nx as f32, ny as f32, nz as f32)
}

/// `face_up_or_toward`: the winding a brand-new face should take. A
/// horizontal one shows its front upwards, any other one faces the `eye`
/// that drew it. Untouched when nothing decides.
pub fn face_up_or_toward(lp: &[V3], eye: Option<V3>, h: Hypot) -> Vec<V3> {
    let pts = lp.to_vec();
    if pts.len() < 3 {
        return pts;
    }
    let n = newell(&pts);
    let len = n.length(h) as f64;
    if len < 1e-12 {
        return pts;
    }
    let want = if (n.2 as f64).abs() > 0.5 * len {
        n.2 as f64 > 0.0
    } else if let Some(eye) = eye {
        let mut c = V3(0.0, 0.0, 0.0);
        for &q in &pts {
            c = c.add(q);
        }
        let c = c.div(pts.len() as f32);
        (n.dot(eye.sub(c)) as f64) > 0.0
    } else {
        return pts;
    };
    if want {
        pts
    } else {
        let mut out = vec![pts[0]];
        out.extend(pts[1..].iter().rev().copied());
        out
    }
}

// ---- the planner -----------------------------------------------------------------------

struct Plan {
    h: Hypot,
    eye: Option<V3>,
    detect_faces: bool,
    simulated: Vec<SimEdge>,
    /// `simulated` as segments, kept in step with it.
    segs: Vec<Seg>,
    faces: Vec<SimFace>,
    graph: Option<CycleGraph>,
    next_id: u32,
    commands: Vec<Box<dyn Command>>,
}

impl Plan {
    fn push_edge(&mut self, a: V3, b: V3) {
        let id = self.next_id;
        self.next_id += 1;
        self.simulated.push(SimEdge { id, a, b, soft: false, curve: None });
        self.segs.push((a, b));
        if let Some(g) = &mut self.graph {
            g.add(EdgeId(id), a, b);
        }
    }

    fn drop_edge(&mut self, id: u32) {
        if let Some(i) = self.simulated.iter().position(|e| e.id == id) {
            self.simulated.remove(i);
            self.segs.remove(i);
            if let Some(g) = &mut self.graph {
                g.remove(EdgeId(id));
            }
        }
    }

    /// `_plan_faces`: the face commands for a freshly added edge `sa`-`sb`:
    /// a chord split of the face it divides, else the new minimal cycles.
    fn plan_faces(&mut self, sa: V3, sb: V3) {
        let h = self.h;
        let mut chord = None;
        for (i, f) in self.faces.iter().enumerate() {
            if let Some((x, y)) = topology::split_face_by_chord(&f.pface(), sa, sb, h) {
                chord = Some((i, x, y));
                break;
            }
        }
        if let Some((i, loop_a, loop_b)) = chord {
            // Both halves ARE the mother, cut in two: they keep her paint.
            let mother = self.faces.remove(i);
            let keep = mother.attrs.clone().filter(|a| !a.is_empty());
            self.commands.push(Box::new(DeleteFaceCommand::new(mother.vertices.clone())));
            for lp in [loop_a, loop_b] {
                if topology::is_planar(&lp, topology::PLANAR_TOLERANCE, h) && !face_exists(&self.faces, &lp) {
                    self.commands.push(Box::new(AddFaceCommand::new(lp.clone(), true, None, keep.clone())));
                    self.faces.push(SimFace { vertices: lp, holes: Vec::new(), attrs: keep.clone() });
                }
            }
            return;
        }
        let cycles = match &self.graph {
            Some(g) => topology::find_cycles_through_graph(g, sa, sb, 2),
            None => topology::find_cycles_through(&self.segs, sa, sb, 2),
        };
        for cycle in cycles {
            if topology::is_planar(&cycle, topology::PLANAR_TOLERANCE, h) && !face_exists(&self.faces, &cycle) {
                let cycle = face_up_or_toward(&cycle, self.eye, h);
                self.commands.push(Box::new(AddFaceCommand::new(cycle.clone(), true, None, None)));
                self.faces.push(SimFace { vertices: cycle, holes: Vec::new(), attrs: None });
            }
        }
    }
}

/// `plan_edge_commands`: the ordered command list that adds `segments` to
/// the mesh `m`. Segments are processed in order against a running
/// simulation of the mesh's edge list, so a batch (a rectangle's four edges)
/// splits correctly even when later segments cross edges earlier ones made.
pub fn plan_edge_commands(m: &Mesh, segments: &[Segment], detect_faces: bool, eye: Option<V3>, h: Hypot) -> Vec<Box<dyn Command>> {
    let simulated: Vec<SimEdge> = m
        .edges
        .iter()
        .map(|&e| {
            let ed = m.edge(e);
            SimEdge { id: e.0, a: m.position(ed.v0), b: m.position(ed.v1), soft: ed.soft, curve: ed.curve }
        })
        .collect();
    let segs: Vec<Seg> = simulated.iter().map(|e| (e.a, e.b)).collect();
    let faces: Vec<SimFace> = m
        .faces
        .iter()
        .map(|&f| {
            let p = PFace::of(m, f);
            SimFace { vertices: p.vertices, holes: p.holes, attrs: Some(m.face(f).attrs.clone()) }
        })
        .collect();
    // The cycle search's graph of `simulated`, kept in step with it.
    let graph = if detect_faces {
        let edges: Vec<(EdgeId, V3, V3)> = simulated.iter().map(|e| (EdgeId(e.id), e.a, e.b)).collect();
        Some(CycleGraph::from_edges(&edges))
    } else {
        None
    };
    let mut plan = Plan {

        h,
        eye,
        detect_faces,
        simulated,
        segs,
        faces,
        graph,
        next_id: m.arena_sizes().1 as u32,
        commands: Vec::new(),
    };


    for &(a, b) in segments {
        let (new_segments, edge_cuts) = topology::plan_edge_split(&plan.segs, a, b, h);
        let cuts: Vec<(SimEdge, V3)> = edge_cuts.iter().map(|&(i, p)| (plan.simulated[i].clone(), p)).collect();

        // Replace each crossed existing edge with its two sub-edges.
        for (edge, point) in cuts {
            // Splitting an edge, not erasing it: no cascade to faces.
            plan.commands.push(Box::new(DeleteEdgesCommand::new(vec![(edge.a, edge.b)], false)));
            plan.drop_edge(edge.id);
            // Sub-edges inherit the split edge's curve/soft flags.
            let e_soft = if edge.soft { Some(true) } else { None };
            let e_curve = edge.curve;
            for (sa, sb) in [(edge.a, point), (point, edge.b)] {
                if topology::find_duplicate_edge(&plan.segs, sa, sb).is_none() {
                    plan.commands.push(Box::new(AddEdgeCommand::new(sa, sb, e_soft, e_curve)));
                    plan.push_edge(sa, sb);
                }
            }
            // Carry the split into the faces sharing this edge, but not the
            // face the drawn segment chord-splits. NOT gated on
            // `detect_faces`: that flag is about FACING.
            for (i, new_verts) in split_edge_in_faces(&plan.faces, edge.a, edge.b, point, &[a, b], h) {
                let f = plan.faces[i].clone();
                // Same face, one collinear vertex richer: it keeps its paint.
                let keep = f.attrs.clone().filter(|x| !x.is_empty());
                plan.commands.push(Box::new(DeleteFaceCommand::new(f.vertices.clone())));
                let holes = if f.holes.is_empty() { None } else { Some(f.holes.clone()) };
                plan.commands.push(Box::new(AddFaceCommand::new(new_verts.clone(), false, holes, keep.clone())));
                plan.faces[i] = SimFace { vertices: new_verts, holes: f.holes.clone(), attrs: keep };
            }
        }

        // Add the new segment's pieces (welding duplicates), auto-facing.
        for (sa, sb) in new_segments {
            let already = topology::find_duplicate_edge(&plan.segs, sa, sb);
            // Always emit the add: AddEdgeCommand welds, so a duplicate is a
            // no-op, but face detection must still run.
            plan.commands.push(Box::new(AddEdgeCommand::new(sa, sb, None, None)));
            if already.is_none() {
                plan.push_edge(sa, sb);
            }
            if plan.detect_faces {
                plan.plan_faces(sa, sb);
            }
        }
    }
    plan.commands
}

fn scene_has_curves(m: &Mesh) -> bool {
    m.edges.iter().any(|&e| m.edge(e).curve.is_some())
}

fn has_rebuild(commands: &[Box<dyn Command>]) -> bool {
    commands.iter().any(|c| c.as_any().is::<RebuildPlanarFacesCommand>() || c.as_any().is::<RebuildPlaneFacesCommand>())
}

/// `_append_flat_curve_rebuild`: the deterministic planar-arrangement
/// rebuild after a draw, when the cycle planner alone cannot produce the
/// regions expected (a square over a circle, overlapping rectangles).
fn append_flat_curve_rebuild(m: &Mesh, h: Hypot, commands: &mut Vec<Box<dyn Command>>, points: &[V3]) {
    if has_rebuild(commands) {
        return;
    }
    let has_curves = scene_has_curves(m);
    let mut verts: Vec<V3> = m.vertices.iter().map(|&v| m.position(v)).collect();
    verts.extend_from_slice(points);
    if coplanar_plane(&verts, 1e-4, h).is_some() {
        if has_curves {
            // Whole-flat drawing with curves: the full rebuild.
            commands.push(Box::new(RebuildPlanarFacesCommand::new()));
        } else if let Some(&f0) = m.faces.first() {
            // Whole-flat, straight edges only, faces present: the scoped
            // rebuild with coverage semantics.
            let normal = m.face_normal(f0, h);
            let origin = m.position(m.face(f0).outer[0]);
            commands.push(Box::new(RebuildPlaneFacesCommand::new(origin, normal, h)));
        }
        return;
    }
    if !has_curves {
        return;
    }
    let Some((origin, normal)) = coplanar_plane(points, 1e-4, h) else { return };
    let on_plane = |p: V3| (p.sub(origin).dot(normal) as f64).abs() < 1e-4;
    if m.edges.iter().any(|&e| {
        let ed = m.edge(e);
        ed.curve.is_some() && on_plane(m.position(ed.v0)) && on_plane(m.position(ed.v1))
    }) {
        commands.push(Box::new(RebuildPlaneFacesCommand::new(origin, normal, h)));
    }
}

/// `_append_face_plane_rebuild`: the scoped plane rebuild when straight
/// edges land on a plane that already carries faces.
fn append_face_plane_rebuild(m: &Mesh, h: Hypot, commands: &mut Vec<Box<dyn Command>>, points: &[V3]) {
    if has_rebuild(commands) {
        return;
    }
    let tol = 1e-4;
    for &f in &m.faces {
        let origin = m.position(m.face(f).outer[0]);
        let normal = m.face_normal(f, h);
        // A degenerate normal is a wildcard: it would claim every line.
        if (normal.length_squared() as f64) < 0.5 {
            continue;
        }
        if points.iter().all(|&p| (p.sub(origin).dot(normal) as f64).abs() < tol) {
            // Reference-scale guard: past it, keep the naive path.
            let on_plane = m
                .faces
                .iter()
                .filter(|&&g| {
                    (m.position(m.face(g).outer[0]).sub(origin).dot(normal) as f64).abs() < tol
                        && (m.face_normal(g, h).dot(normal) as f64).abs() > 0.999
                })
                .count();
            if on_plane <= 400 {
                commands.push(Box::new(RebuildPlaneFacesCommand::new(origin, normal, h)));
            }
            return;
        }
    }
}

/// `_append_active_tag`: tag-as-you-draw. When the BIM active class is set,
/// stamp it on the plan's own drawn faces, BEFORE the plane rebuilds.
fn append_active_tag(env: &Env, commands: &mut Vec<Box<dyn Command>>) {
    let Some(Value::Object(tag)) = &env.active_ifc else { return };
    if tag.is_empty() {
        return;
    }
    let face_cmds: Vec<_> = commands
        .iter()
        .filter_map(|c| c.as_any().downcast_ref::<AddFaceCommand>())
        .filter(|c| c.auto)
        .map(|c| c.face.clone())
        .collect();
    if !face_cmds.is_empty() {
        commands.push(Box::new(AutoTagDrawnFacesCommand { face_commands: face_cmds, tag: tag.clone() }));
    }
}

/// `build_add_edge`: one drawn edge as one atomic command.
pub fn build_add_edge(scene: &Scene, env: &Env, a: V3, b: V3, detect_faces: bool, eye: Option<V3>) -> Box<dyn Command> {
    let m = env.mesh(scene);
    let h = env.hypot;
    let mut commands = plan_edge_commands(m, &[(a, b)], detect_faces, eye, h);
    append_active_tag(env, &mut commands);
    append_flat_curve_rebuild(m, h, &mut commands, &[a, b]);
    if detect_faces {
        append_face_plane_rebuild(m, h, &mut commands, &[a, b]);
    }
    // When curves exist even a single edge can break a circle into contours,
    // so it goes through SnapshotCompound, which runs the contour re-split.
    if commands.len() == 1 && !scene_has_curves(m) {
        return commands.pop().expect("one command");
    }
    Box::new(SnapshotCompound::new(commands))
}

/// `build_add_edges`: many drawn edges (plus `extra` commands, such as a
/// tool-managed face) as one atomic command. `eye` decides which way a new
/// vertical face looks; `plane_rebuild` asks for the plane the tool lands
/// on to be rebuilt (the Rectangle tool).
pub fn build_add_edges(
    scene: &Scene,
    env: &Env,
    segments: &[Segment],
    detect_faces: bool,
    extra: Vec<Box<dyn Command>>,
    eye: Option<V3>,
    plane_rebuild: bool,
) -> Box<dyn Command> {
    let m = env.mesh(scene);
    let h = env.hypot;
    let mut commands = plan_edge_commands(m, segments, detect_faces, eye, h);
    commands.extend(extra);
    append_active_tag(env, &mut commands);
    let points: Vec<V3> = segments.iter().flat_map(|&(a, b)| [a, b]).collect();
    append_flat_curve_rebuild(m, h, &mut commands, &points);
    if plane_rebuild {
        append_face_plane_rebuild(m, h, &mut commands, &points);
    }
    if commands.len() == 1 && !scene_has_curves(m) {
        return commands.pop().expect("one command");
    }
    Box::new(SnapshotCompound::new(commands))
}

// ---- Divide and Join Segments -------------------------------------------------------------

fn edge_flags(m: &Mesh, e: EdgeId) -> EdgeFlags {
    let ed = m.edge(e);
    EdgeFlags { soft: ed.soft, curve: ed.curve, layer: ed.layer.clone(), hidden: ed.hidden }
}

fn stamp_edge_flags(m: &mut Mesh, e: EdgeId, f: &EdgeFlags) {
    let ed = m.edge_mut(e);
    ed.soft = f.soft;
    ed.curve = f.curve;
    ed.layer = f.layer.clone();
    ed.hidden = f.hidden;
}

type Walk = Vec<(EdgeId, ingetrazo_model::VertexId, ingetrazo_model::VertexId)>;

/// `_order_chain`: the edges of a curve in walking order as `(edge, from,
/// to)`, starting at one end (or at `seed.v0` for a closed loop).
fn order_chain(m: &Mesh, chain: &[EdgeId], seed: EdgeId) -> Walk {
    if chain.len() <= 1 {
        let e = chain[0];
        return vec![(e, m.edge(e).v0, m.edge(e).v1)];
    }
    // by_vertex: vertex -> edges, in first-seen order.
    let mut by_vertex: Vec<(ingetrazo_model::VertexId, Vec<EdgeId>)> = Vec::new();
    for &e in chain {
        for v in [m.edge(e).v0, m.edge(e).v1] {
            match by_vertex.iter_mut().find(|(x, _)| *x == v) {
                Some((_, es)) => es.push(e),
                None => by_vertex.push((v, vec![e])),
            }
        }
    }
    let ends: Vec<_> = by_vertex.iter().filter(|(_, es)| es.len() == 1).map(|(v, _)| *v).collect();
    let mut v = ends.first().copied().unwrap_or(m.edge(seed).v0);
    let mut ordered: Walk = Vec::new();
    let mut seen: HashSet<EdgeId> = HashSet::new();
    loop {
        let es = &by_vertex.iter().find(|(x, _)| *x == v).expect("vertex of the chain").1;
        let Some(&nxt) = es.iter().find(|e| !seen.contains(e)) else { break };
        seen.insert(nxt);
        let w = m.other(nxt, v);
        ordered.push((nxt, v, w));
        v = w;
        if ordered.len() == chain.len() {
            break;
        }
    }
    ordered
}

/// Python's float `//` (`float_floor_div`).
fn py_floordiv(x: f64, y: f64) -> f64 {
    let modulus = x % y;
    let mut div = (x - modulus) / y;
    if modulus != 0.0 {
        if (y < 0.0) != (modulus < 0.0) {
            div -= 1.0;
        }
    } else {
        div = 0.0f64.copysign(x / y);
    }
    if div != 0.0 {
        let mut floordiv = div.floor();
        if div - floordiv > 0.5 {
            floordiv += 1.0;
        }
        floordiv
    } else {
        0.0f64.copysign(x / y)
    }
}

/// `divide_edges`: split each selected edge (or, for an edge of a curve, the
/// whole curve) into `n` pieces of equal length. A curve comes out as `n`
/// independent arcs, each its own curve. Returns the number of cuts made.
/// `curve_counter` is `core.mesh._CURVE_COUNTER`.
pub fn divide_edges(m: &mut Mesh, edges: &[EdgeId], n: i64, curve_counter: &mut i64, h: Hypot) -> usize {
    if n < 2 {
        return 0;
    }
    let n = n as usize;
    let mut done: HashSet<EdgeId> = HashSet::new();
    let mut cuts = 0;
    for &edge in edges {
        if done.contains(&edge) {
            continue;
        }
        let is_curve = m.edge(edge).curve.is_some();
        let chain = if is_curve { m.curve_edges(edge) } else { vec![edge] };
        let walk = order_chain(m, &chain, edge);
        for (e, _, _) in &walk {
            done.insert(*e);
        }
        // `sum()` of a generator of floats: the 3.12 compensated sum.
        let total = ingetrazo_math::pyfloat::py_sum(
            walk.iter().map(|&(_, vf, vt)| m.position(vt).sub(m.position(vf)).length(h) as f64),
            true,
        );
        if total < 1e-9 {
            continue;
        }
        let step = total / n as f64;
        // The facets of each of the n pieces, in walking order.
        let mut pieces: Vec<Vec<EdgeId>> = vec![Vec::new(); n];
        let piece_of = |s0: f64, s1: f64| -> usize {
            let k = py_floordiv((s0 + s1) * 0.5, step) as i64;
            k.clamp(0, n as i64 - 1) as usize
        };
        let mut walked = 0.0f64;
        for (e, v_from, v_to) in walk {
            let a = m.position(v_from);
            let b = m.position(v_to);
            let ln = b.sub(a).length(h) as f64;
            if ln < 1e-12 {
                continue;
            }
            let flags = edge_flags(m, e);
            let inside: Vec<usize> = (1..n).filter(|&k| walked + 1e-9 < step * k as f64 && step * (k as f64) < walked + ln - 1e-9).collect();
            let (mut rest, mut rest_from, mut rest_start) = (e, v_from, walked);
            for k in inside {
                let at = step * k as f64;
                let point = a.add(b.sub(a).mul(((at - walked) / ln) as f32));
                let (e0, e1) = m.split_edge(rest, point);
                if e0 == e1 {
                    break;
                }
                // split_edge hands the halves back in v0->v1 order of the
                // facet, which need not be the walking order.
                let first = if m.edge(e0).v0 == rest_from || m.edge(e0).v1 == rest_from { e0 } else { e1 };
                let second = if first == e0 { e1 } else { e0 };
                for half in [first, second] {
                    stamp_edge_flags(m, half, &flags);
                }
                pieces[piece_of(rest_start, at)].push(first);
                rest_from = m.other(first, rest_from);
                rest = second;
                rest_start = at;
                cuts += 1;
            }
            pieces[piece_of(rest_start, walked + ln)].push(rest);
            walked += ln;
        }
        if is_curve {
            for facets in &pieces {
                if facets.is_empty() {
                    continue;
                }
                let cid = Mesh::next_curve_id(curve_counter);
                for &f in facets {
                    m.edge_mut(f).curve = Some(cid);
                }
            }
        }
    }
    cuts
}

// How far a point may sit off the line through its neighbours and still go.
const JOIN_TOL_FLOOR: f64 = 1e-6;
const JOIN_TOL_REL: f64 = 1e-6;
const JOIN_TOL_MAX: f64 = 1e-4;

/// `_off_line`: the distance from `p` to the line through `a` and `b`, and
/// the tolerance the size of those coordinates allows it, in double.
fn off_line(p: V3, a: V3, b: V3) -> (f64, f64) {
    let d = |v: V3| [v.0 as f64, v.1 as f64, v.2 as f64];
    let (p, a, b) = (d(p), d(a), d(b));
    let ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    let ap = [p[0] - a[0], p[1] - a[1], p[2] - a[2]];
    let cross = [ap[1] * ab[2] - ap[2] * ab[1], ap[2] * ab[0] - ap[0] * ab[2], ap[0] * ab[1] - ap[1] * ab[0]];
    use ingetrazo_math::pyfloat::py_sum;
    let off = py_sum(cross.iter().map(|c| c * c), true).sqrt() / py_sum(ab.iter().map(|c| c * c), true).sqrt();
    let mut size = 0.0f64;
    for q in [p, a, b] {
        for c in q {
            // Python's max(abs(c) ...) keeps the first of equals.
            if c.abs() > size {
                size = c.abs();
            }
        }
    }
    let tol = JOIN_TOL_MAX.min(JOIN_TOL_FLOOR + JOIN_TOL_REL * size);
    (off, tol)
}

/// Under this angle a corner kept by Join Segments is worth pointing out.
const JOIN_KINK_DEG: f64 = 1.0;

/// `_kink_deg`: the turn, in degrees, a path a -> v -> b takes at `v`.
fn kink_deg(m: &Mesh, v: ingetrazo_model::VertexId, a: ingetrazo_model::VertexId, b: ingetrazo_model::VertexId, h: Hypot) -> f64 {
    let d1 = m.position(v).sub(m.position(a));
    let d2 = m.position(b).sub(m.position(v));
    if d1.length(h) == 0.0 || d2.length(h) == 0.0 {
        return 180.0;
    }
    let c = d1.normalized(h).dot(d2.normalized(h)) as f64;
    // max(-1.0, min(1.0, c)), then math.degrees(math.acos(...)).
    let c = c.clamp(-1.0, 1.0);
    c.acos().to_degrees()
}

/// `_straight_through`: whether two of the `picked` edges at `v` run on in
/// one straight line through it.
fn straight_through(m: &Mesh, v: ingetrazo_model::VertexId, picked: &HashSet<EdgeId>, h: Hypot) -> bool {
    let ends: Vec<_> = m.sorted_edges(v).into_iter().filter(|e| picked.contains(e)).map(|e| m.other(e, v)).collect();
    for (i, &a) in ends.iter().enumerate() {
        for &b in &ends[i + 1..] {
            let (off, tol) = off_line(m.position(v), m.position(a), m.position(b));
            if off <= tol && kink_deg(m, v, a, b, h) < 90.0 {
                return true;
            }
        }
    }
    false
}

/// Why Join Segments leaves a point where it is (`JOIN_REASONS`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum JoinVerdict {
    Join,
    Kink,
    Branch,
    Flags,
    Curve,
}

/// `join_edges`: the inverse of Divide. Every vertex where two of `edges`
/// meet in a straight line, with nothing else attached, goes, and the two
/// segments become one. Returns the edges the selection is made of
/// afterwards. `verdicts`, when given, is filled with what happened to each
/// point on the way.
pub fn join_edges(
    m: &mut Mesh,
    edges: &[EdgeId],
    mut verdicts: Option<&mut Vec<(ingetrazo_model::VertexId, JoinVerdict)>>,
    h: Hypot,
) -> Vec<EdgeId> {
    let mut live: Vec<EdgeId> = edges.iter().copied().filter(|e| m.edges.contains(e)).collect();
    let mut picked: HashSet<EdgeId> = live.iter().copied().collect();
    let mut seen: HashSet<ingetrazo_model::VertexId> = HashSet::new();
    let mut candidates = Vec::new();
    for &e in &live {
        for v in [m.edge(e).v0, m.edge(e).v1] {
            if seen.insert(v) {
                candidates.push(v);
            }
        }
    }
    let mut keep = |v: ingetrazo_model::VertexId, r: JoinVerdict| {
        if let Some(vs) = verdicts.as_deref_mut() {
            match vs.iter_mut().find(|(x, _)| *x == v) {
                Some(slot) => slot.1 = r,
                None => vs.push((v, r)),
            }
        }
    };
    for v in candidates {
        let nedges = m.vertex_ref(v).edges.len();
        if nedges != 2 {
            if nedges > 2 && straight_through(m, v, &picked, h) {
                keep(v, JoinVerdict::Branch);
            }
            continue;
        }
        let es = m.sorted_edges(v);
        let (e1, e2) = (es[0], es[1]);
        if !picked.contains(&e1) || !picked.contains(&e2) {
            continue;
        }
        let (a, b) = (m.other(e1, v), m.other(e2, v));
        let (off, tol) = off_line(m.position(v), m.position(a), m.position(b));
        if off > tol {
            if kink_deg(m, v, a, b, h) < JOIN_KINK_DEG {
                keep(v, JoinVerdict::Kink);
            }
            continue;
        }
        let flags = edge_flags(m, e1);
        if flags != edge_flags(m, e2) {
            keep(v, JoinVerdict::Flags);
            continue;
        }
        if m.edge(e1).curve.is_some() || m.edge(e2).curve.is_some() {
            keep(v, JoinVerdict::Curve);
            continue;
        }
        if !m.collapsible_vertex(v, h) {
            continue;
        }
        if m.find_edge(a, b).is_some() {
            continue; // the whole line is there already
        }
        let joined = match m.collapse_vertex(v) {
            ingetrazo_model::mesh::StitchRec::Collapse { new_edge, .. } => new_edge,
            _ => continue,
        };
        stamp_edge_flags(m, joined, &flags);
        keep(v, JoinVerdict::Join);
        picked.remove(&e1);
        picked.remove(&e2);
        picked.insert(joined);
        live.push(joined);
    }
    live.into_iter().filter(|e| picked.contains(e) && m.edges.contains(e)).collect()
}

/// `join_preview`: what Join Segments would do to `edges`, without doing it
/// (the mesh is put back as it was, the same objects included).
pub fn join_preview(m: &mut Mesh, edges: &[EdgeId], h: Hypot) -> Vec<(ingetrazo_model::VertexId, JoinVerdict)> {
    let mut verdicts = Vec::new();
    let snap = m.capture_state();
    join_edges(m, edges, Some(&mut verdicts), h);
    m.restore_state(&snap);
    verdicts
}

