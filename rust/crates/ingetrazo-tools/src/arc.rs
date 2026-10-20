// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! The Arc tools (`tools/arc.py`): the 2-point [`ArcTool`] (the two ends,
//! then the bulge - with the tangent-to-edge, tangent-at-vertex, half-circle
//! and fillet inferences), the [`ThreePointArcTool`], the compass
//! [`CenterArcTool`] and its closed variant, the [`PieTool`]. All commit
//! through [`commit_arc`], the shared curve pipeline.
//!
//! Where it differs from Python (reported): the tangent an existing arc
//! leaves its end vertex with (`_tangent_at_start`) looks in the loose mesh
//! only, not in the groups and components placed in the scene; and the
//! circle fit behind it is `ingetrazo-snap`'s, which is not LAPACK (a few
//! ulps of float64, hidden by the float32 of the result).

use ingetrazo_geom::bevel_profiles::py_hypot2;
use ingetrazo_math::pyfloat::{py_max2, py_min2};
use ingetrazo_math::V3;
use ingetrazo_model::mesh::{EdgeId, VertexId};
use ingetrazo_model::topology::point_on_seg_incl;
use ingetrazo_ops::command::Command;
use ingetrazo_ops::draw::{DeleteEdgesCommand, TagCurveCommand};
use ingetrazo_ops::edits::build_add_edges;
use ingetrazo_ops::rebuild::{RebuildPlanarFacesCommand, RebuildPlaneFacesCommand};
use ingetrazo_snap::circle::fit_circle;
use ingetrazo_snap::types::COLOR_MIDPOINT;
use ingetrazo_snap::{SnapKind, ToolState};

use crate::base::*;
use crate::circle::{busy_plane, flat_drawing};
use crate::host::Core;
use crate::units::fmt_angle_deg;
use crate::vcb::VcbValue;

/// `_SEGMENTS`: polyline segments approximating the arc.
const SEGMENTS: i64 = 16;
/// `COLOR_FILLET`: the arc that is tangent to BOTH edges of a corner.
const COLOR_FILLET: [f64; 3] = [0.85, 0.30, 0.80];
/// `core.snap.COLOR_TANGENT`.
const COLOR_TANGENT: [f64; 3] = [0.20, 0.66, 0.74];

fn rgba(c: [f64; 3]) -> [f64; 4] {
    [c[0], c[1], c[2], 1.0]
}

// ---- the arc maths -------------------------------------------------------------------------

type P2 = (f64, f64);

fn circumcenter2(a: P2, b: P2, c: P2) -> Option<P2> {
    let d = 2.0 * (a.0 * (b.1 - c.1) + b.0 * (c.1 - a.1) + c.0 * (a.1 - b.1));
    if d.abs() < 1e-12 {
        return None;
    }
    let (a2, b2, c2) = (a.0 * a.0 + a.1 * a.1, b.0 * b.0 + b.1 * b.1, c.0 * c.0 + c.1 * c.1);
    let ux = (a2 * (b.1 - c.1) + b2 * (c.1 - a.1) + c2 * (a.1 - b.1)) / d;
    let uy = (a2 * (c.0 - b.0) + b2 * (a.0 - c.0) + c2 * (b.0 - a.0)) / d;
    Some((ux, uy))
}

fn wrap(mut a: f64) -> f64 {
    let pi = std::f64::consts::PI;
    while a <= -pi {
        a += 2.0 * pi;
    }
    while a > pi {
        a -= 2.0 * pi;
    }
    a
}

/// `_arc_3pts_2d`: 2D points of the circular arc from `s` to `e` through
/// `m`, sampled into `segments` spans; the straight chord when collinear.
pub fn arc_3pts_2d(s: P2, m: P2, e: P2, segments: i64) -> Vec<P2> {
    let Some((cx, cy)) = circumcenter2(s, m, e) else { return vec![s, e] };
    let pi = std::f64::consts::PI;
    let r = py_hypot2(s.0 - cx, s.1 - cy);
    let a0 = (s.1 - cy).atan2(s.0 - cx);
    let a1 = (e.1 - cy).atan2(e.0 - cx);
    let am = (m.1 - cy).atan2(m.0 - cx);
    let mut d = wrap(a1 - a0);
    let da = wrap(am - a0);
    if d >= 0.0 && !(0.0 <= da && da <= d) {
        d -= 2.0 * pi;
    } else if d < 0.0 && !(d <= da && da <= 0.0) {
        d += 2.0 * pi;
    }
    (0..=segments).map(|k| (cx + r * (a0 + d * (k as f64 / segments as f64)).cos(), cy + r * (a0 + d * (k as f64 / segments as f64)).sin())).collect()
}

/// `commit_arc(viewport, pts, close_to, trim)`: an arc polyline with the
/// shared curve pipeline - planar arrangement on flat drawings, the scoped
/// per-plane arrangement when the drawing plane already carries content in
/// a 3D scene, naive otherwise. Returns the identity of the command when it
/// landed on the undo stack.
pub fn commit_arc(core: &mut Core, pts: &[V3], close_to: Option<V3>, trim: Option<&[(V3, V3)]>) -> Option<u64> {
    let mut segments: Vec<(V3, V3)> = (0..pts.len().saturating_sub(1)).map(|i| (pts[i], pts[i + 1])).collect();
    if let Some(c) = close_to {
        let mut s = vec![(c, pts[0])];
        s.extend(segments);
        s.push((pts[pts.len() - 1], c));
        segments = s;
    }
    let after = || -> Vec<Box<dyn Command>> {
        match trim {
            Some(t) if !t.is_empty() => vec![Box::new(DeleteEdgesCommand::new(t.to_vec(), true))],
            _ => Vec::new(),
        }
    };
    let tag = || -> Box<dyn Command> { Box::new(TagCurveCommand { pts: pts.to_vec(), closed: false }) };
    let cmd = if flat_drawing(core, pts) {
        let mut extra: Vec<Box<dyn Command>> = vec![tag(), Box::new(RebuildPlanarFacesCommand::new())];
        extra.extend(after());
        build_add_edges(&core.scene, &core.cx, &segments, false, extra, None, false)
    } else if let Some((o, n)) = busy_plane(core, pts) {
        let mut extra: Vec<Box<dyn Command>> = vec![tag(), Box::new(RebuildPlaneFacesCommand::new(o, n, core.h()))];
        extra.extend(after());
        build_add_edges(&core.scene, &core.cx, &segments, true, extra, None, false)
    } else {
        let mut extra: Vec<Box<dyn Command>> = vec![tag()];
        extra.extend(after());
        build_add_edges(&core.scene, &core.cx, &segments, true, extra, None, false)
    };
    let before = core.undo_len();
    core.exec(cmd);
    if core.undo_len() > before {
        core.history.top_serial()
    } else {
        None
    }
}

/// `SegmentCount._set_segments`: the count of the next arc.
fn set_segments(core: &mut Core, segments: &mut Option<i64>, n: f64) -> bool {
    let n = n.round_ties_even() as i64;
    if n < 2 {
        return false;
    }
    *segments = Some(n);
    core.flash(format!("{n} segments"));
    true
}

// ---- Arc -----------------------------------------------------------------------------------

/// What `Ns` right after an arc needs to rebuild it: the apex in the arc's
/// own 2D frame, the chord's end, and the frame.
#[derive(Clone, Copy, Debug)]
struct Apex {
    apex: P2,
    e2: P2,
    u: V3,
    v: V3,
    origin: V3,
}

type Corner = (VertexId, EdgeId, EdgeId);

pub struct ArcTool {
    pub start_point: Option<V3>,
    pub end_point: Option<V3>,
    pub hover_point: Option<V3>,
    segments: i64,
    tangent_dir: Option<V3>,
    snap_bulge: Option<f64>,
    bulge_kind: Option<&'static str>,
    /// The edge the start point sits on (interior), with the start point.
    start_edge: Option<(EdgeId, V3)>,
    /// `(V, P2, edge B)`: the equidistant point on an adjacent edge.
    equidistant: Option<(V3, V3, EdgeId)>,
    /// `(V, P1, P2)`: the corner this arc rounds.
    fillet: Option<(V3, V3, V3)>,
    fillet_edge_b: Option<EdgeId>,
    last_cmd: Option<u64>,
    last_params: Option<(Apex, Option<Vec<(V3, V3)>>)>,
    last_fillet_d: Option<f64>,
    pl: PlaneState,
    viewport_seen: bool,
    wireframe_color: Option<[f64; 4]>,
}

impl Default for ArcTool {
    fn default() -> Self {
        ArcTool::new()
    }
}

impl ArcTool {
    /// Within this many screen pixels of the tangent bulge, the arc snaps to
    /// it ("Tangent at Vertex", cyan).
    const TANGENT_PX: f64 = 8.0;
    /// Within this many pixels of the point on the adjacent edge at the same
    /// distance from the corner, the end point snaps to it (a fillet).
    const FILLET_PX: f64 = 10.0;
    /// A double-click this close (px) to a two-edge corner repeats the last
    /// rounding on it.
    const CORNER_PX: f64 = 20.0;

    pub fn new() -> ArcTool {
        ArcTool {
            start_point: None,
            end_point: None,
            hover_point: None,
            segments: SEGMENTS,
            tangent_dir: None,
            snap_bulge: None,
            bulge_kind: None,
            start_edge: None,
            equidistant: None,
            fillet: None,
            fillet_edge_b: None,
            last_cmd: None,
            last_params: None,
            last_fillet_d: None,
            pl: PlaneState::default(),
            viewport_seen: false,
            wireframe_color: None,
        }
    }

    fn reset(&mut self) {
        self.start_point = None;
        self.end_point = None;
        self.pl.work_plane = None;
        self.pl.hover_plane = None;
        clear_plane_lock(&mut self.pl);
        self.tangent_dir = None;
        self.snap_bulge = None;
        self.bulge_kind = None;
        self.start_edge = None;
        self.equidistant = None;
        self.fillet = None;
        self.fillet_edge_b = None;
        self.wireframe_color = None;
    }

    fn axes(&self, core: &Core) -> (V3, V3) {
        plane_axes(&core.axes, drawing_plane(self, core).1, core.h())
    }

    fn to2(&self, p: V3, u: V3, v: V3) -> P2 {
        let d = p.sub(self.start_point.expect("a start"));
        (d.dot(u) as f64, d.dot(v) as f64)
    }

    /// `_bulge_for(cursor)`: signed perpendicular distance from the chord
    /// midpoint to the cursor.
    fn bulge_for(&self, core: &Core, cursor: V3) -> f64 {
        let (u, v) = self.axes(core);
        let e2 = self.to2(self.end_point.expect("an end"), u, v);
        let length = py_hypot2(e2.0, e2.1);
        if length < 1e-9 {
            return 0.0;
        }
        let (px, py) = (-e2.1 / length, e2.0 / length);
        let mid = (e2.0 / 2.0, e2.1 / 2.0);
        let c2 = self.to2(cursor, u, v);
        (c2.0 - mid.0) * px + (c2.1 - mid.1) * py
    }

    /// `_points(cursor, bulge, end)`: the arc polyline, and the apex it was
    /// built with (`_last_apex`).
    fn points(&self, core: &Core, cursor: Option<V3>, bulge: Option<f64>, end: Option<V3>) -> (Vec<V3>, Option<Apex>) {
        let end = end.or(self.end_point).expect("an end");
        let start = self.start_point.expect("a start");
        let (u, v) = self.axes(core);
        let e2 = self.to2(end, u, v);
        let length = py_hypot2(e2.0, e2.1);
        if length < 1e-9 {
            return (Vec::new(), None);
        }
        let h = if let Some(b) = bulge {
            b
        } else if let Some(sb) = self.snap_bulge {
            sb // tangent to the arc before
        } else {
            self.bulge_for(core, cursor.expect("a cursor"))
        };
        if h.abs() < 1e-4 {
            return (vec![start, end], None); // flat: a straight chord
        }
        let (px, py) = (-e2.1 / length, e2.0 / length);
        let mid = (e2.0 / 2.0, e2.1 / 2.0);
        let apex = (mid.0 + px * h, mid.1 + py * h);
        let pts2 = arc_3pts_2d((0.0, 0.0), apex, e2, self.segments);
        let pts = pts2.iter().map(|&(x, y)| start.add(u.mul(x as f32)).add(v.mul(y as f32))).collect();
        (pts, Some(Apex { apex, e2, u, v, origin: start }))
    }

    /// `_tangent_bulge(end, tangent)`: the signed bulge that makes this arc
    /// leave the start along `tangent` toward `end`.
    fn tangent_bulge(&self, core: &Core, end: Option<V3>, tangent: Option<V3>) -> Option<f64> {
        let tangent = tangent.or(self.tangent_dir)?;
        let end = end.or(self.end_point)?;
        let (u, v) = self.axes(core);
        let e2 = self.to2(end, u, v);
        let length = py_hypot2(e2.0, e2.1);
        if length < 1e-9 {
            return None;
        }
        let (cx, cy) = (e2.0 / length, e2.1 / length);
        let (tx, ty) = (tangent.dot(u) as f64, tangent.dot(v) as f64);
        let tl = py_hypot2(tx, ty);
        if tl < 1e-6 {
            return None; // tangent leaves the plane
        }
        let (tx, ty) = (tx / tl, ty / tl);
        let angle = (cx * ty - cy * tx).atan2(cx * tx + cy * ty);
        if (angle.abs() - std::f64::consts::PI).abs() < 1e-3 {
            return None; // doubling back: no arc
        }
        Some((length / 2.0) * (angle / 2.0).tan())
    }

    /// `_edge_under(viewport, P)`: `(edge, P)` when `P` lies on the interior
    /// of an edge of the mesh being drawn on.
    fn edge_under(core: &Core, p: V3) -> Option<(EdgeId, V3)> {
        let h = core.h();
        let m = core.mesh();
        let interior = |e: EdgeId| -> bool {
            let ed = m.edge(e);
            let (a, b) = (m.position(ed.v0), m.position(ed.v1));
            if !point_on_seg_incl(p, a, b, 1e-4, h) {
                return false;
            }
            (p.sub(a).length(h) as f64) > 1e-4 && (p.sub(b).length(h) as f64) > 1e-4 && (b.sub(a).length(h) as f64) > 1e-6
        };
        if let Some(e) = core.hover_loose_edge {
            if m.edges.contains(&e) && m.vertex_ref(m.edge(e).v0).edges.contains(&e) && interior(e) {
                return Some((e, p));
            }
        }
        m.edges.iter().copied().find(|&e| interior(e)).map(|e| (e, p))
    }

    /// `_edge_tangent(toward)`: the direction the arc leaves the start point
    /// along its edge, on the side of `toward`.
    fn edge_tangent(&self, core: &Core, toward: V3) -> Option<V3> {
        let h = core.h();
        let (edge, p1) = self.start_edge?;
        let m = core.mesh();
        let ed = m.edge(edge);
        let d = m.position(ed.v1).sub(m.position(ed.v0));
        if (d.length(h) as f64) < 1e-9 {
            return None;
        }
        let mut d = d.normalized(h);
        if (d.dot(toward.sub(p1)) as f64) < 0.0 {
            d = d.neg();
        }
        Some(d)
    }

    /// `_pixel_distance`: cursor to `P` in screen pixels.
    fn pixel_distance(core: &Core, screen: (f64, f64), p: V3) -> Option<f64> {
        let q = core.world_to_pixel(p)?;
        Some(py_hypot2(q.0 - screen.0, q.1 - screen.1))
    }

    /// `_equidistant_near`: `(V, P2, B)`, the point on an edge `B` adjacent to
    /// the start edge at the start point's distance from their shared corner
    /// `V`, when the cursor is within `FILLET_PX` of it.
    fn equidistant_near(&self, core: &Core, screen: (f64, f64)) -> Option<(V3, V3, EdgeId)> {
        let h = core.h();
        let (edge, p1) = self.start_edge?;
        let m = core.mesh();
        let mut best: Option<(f64, V3, V3, EdgeId)> = None;
        let ed = m.edge(edge);
        for vtx in [ed.v0, ed.v1] {
            let vp = m.position(vtx);
            let d = p1.sub(vp).length(h) as f64;
            if d < 1e-6 {
                continue;
            }
            for b in m.edges_at(vtx) {
                if b == edge {
                    continue;
                }
                let w = m.position(m.other(b, vtx));
                let span = w.sub(vp).length(h) as f64;
                if span < 1e-9 || d > span + 1e-6 {
                    continue;
                }
                let p2 = vp.add(w.sub(vp).mul((d / span) as f32));
                let Some(dist) = Self::pixel_distance(core, screen, p2) else { continue };
                if dist > Self::FILLET_PX {
                    continue;
                }
                if best.is_none() || dist < best.unwrap().0 {
                    best = Some((dist, vp, p2, b));
                }
            }
        }
        best.map(|(_, v, p2, b)| (v, p2, b))
    }

    /// `_corner_near`: the two-edge corner within `CORNER_PX` of the cursor
    /// whose edges are both long enough for the last fillet.
    fn corner_near(&self, core: &Core, screen: (f64, f64)) -> Option<Corner> {
        let h = core.h();
        let d = self.last_fillet_d.filter(|d| *d != 0.0)?;
        let m = core.mesh();
        let mut best: Option<(f64, Corner)> = None;
        for &vtx in &m.vertices {
            let edges = m.edges_at(vtx);
            if edges.len() != 2 {
                continue;
            }
            let (a, b) = (edges[0], edges[1]);
            if m.edge(a).faces.is_empty() && m.edge(b).faces.is_empty() {
                continue;
            }
            let len = |e: EdgeId| (m.position(m.edge(e).v1).sub(m.position(m.edge(e).v0)).length(h)) as f64;
            if len(a) < d - 1e-6 || len(b) < d - 1e-6 {
                continue;
            }
            let Some(dist) = Self::pixel_distance(core, screen, m.position(vtx)) else { continue };
            if dist > Self::CORNER_PX {
                continue;
            }
            if best.is_none() || dist < best.unwrap().0 {
                best = Some((dist, (vtx, a, b)));
            }
        }
        best.map(|b| b.1)
    }

    /// `_corner_trim(modifiers)`: the two corner stubs to erase once a
    /// fillet is committed.
    fn corner_trim(&self, core: &Core, mods: Option<u32>) -> Option<Vec<(V3, V3)>> {
        if self.fillet.is_none() || self.bulge_kind != Some("fillet") {
            return None;
        }
        if mods.is_some_and(|m| m & MOD_ALT != 0) {
            return None;
        }
        let (v, p1, p2) = self.fillet?;
        let (edge, _) = self.start_edge?;
        let m = core.mesh();
        if m.edge(edge).faces.is_empty() {
            return None;
        }
        let vert = m.vertex_at(v)?;
        if m.vertex_ref(vert).edges.len() != 2 {
            return None;
        }
        Some(vec![(v, p1), (v, p2)])
    }

    /// `_tangent_at_start`: the direction an existing arc leaves its END
    /// vertex when that vertex is our start point.
    fn tangent_at_start(&self, core: &Core) -> Option<V3> {
        let p = self.start_point?;
        Self::tangent_in_mesh(core, p)
    }

    fn tangent_in_mesh(core: &Core, p: V3) -> Option<V3> {
        let h = core.h();
        let m = core.mesh();
        for &edge in &m.edges {
            let ed = m.edge(edge);
            if ed.curve.is_none() {
                continue;
            }
            let (a, b) = (m.position(ed.v0), m.position(ed.v1));
            if (a.sub(p).length(h) as f64) > 1e-6 && (b.sub(p).length(h) as f64) > 1e-6 {
                continue;
            }
            let chain = m.curve_edges(edge);
            let incident: Vec<EdgeId> = chain
                .iter()
                .copied()
                .filter(|&e| {
                    let ee = m.edge(e);
                    (m.position(ee.v0).sub(p).length(h) as f64) <= 1e-6 || (m.position(ee.v1).sub(p).length(h) as f64) <= 1e-6
                })
                .collect();
            if incident.len() != 1 {
                continue; // not the arc's end
            }
            let ee = m.edge(incident[0]);
            let (ea, eb) = (m.position(ee.v0), m.position(ee.v1));
            let q = if (ea.sub(p).length(h) as f64) <= 1e-6 { eb } else { ea };
            let mut verts: Vec<VertexId> = Vec::new();
            for &c in &chain {
                for vv in [m.edge(c).v0, m.edge(c).v1] {
                    if !verts.contains(&vv) {
                        verts.push(vv);
                    }
                }
            }
            let pts: Vec<V3> = verts.iter().map(|&v| m.position(v)).collect();
            let fit = if pts.len() >= 3 { fit_circle(&pts) } else { None };
            let Some((center, _r)) = fit else { continue };
            let radial = p.sub(center);
            if (radial.length(h) as f64) < 1e-9 {
                continue;
            }
            let r_hat = radial.normalized(h);
            let leaving = p.sub(q);
            let t = leaving.sub(r_hat.mul(leaving.dot(r_hat)));
            if (t.length(h) as f64) < 1e-9 {
                continue;
            }
            return Some(t.normalized(h));
        }
        None
    }

    /// `_snap_bulge_for`: bulge-phase inferences - tangent to the edge / arc
    /// the start sits on (cyan, magenta for a fillet), and the half circle.
    fn snap_bulge_for(&mut self, core: &Core, ctx: &Ctx) {
        let (u, v) = self.axes(core);
        let (start, end) = (self.start_point.expect("a start"), self.end_point.expect("an end"));
        let e2 = self.to2(end, u, v);
        let length = py_hypot2(e2.0, e2.1);
        if length < 1e-9 {
            return;
        }
        let perp = u.mul((-e2.1 / length) as f32).add(v.mul((e2.0 / length) as f32));
        let mid = start.add(end).mul(0.5f32);
        let Some(scale) = world_per_pixel(core, mid, perp) else { return };
        let tol = scale * Self::TANGENT_PX;
        let b = self.bulge_for(core, ctx.world);
        if self.tangent_dir.is_some() {
            if let Some(h_t) = self.tangent_bulge(core, None, None) {
                if (b - h_t).abs() <= tol {
                    self.snap_bulge = Some(h_t);
                    self.bulge_kind = Some(if self.fillet.is_some() { "fillet" } else { "tangent" });
                    self.wireframe_color = Some(rgba(if self.fillet.is_some() { COLOR_FILLET } else { COLOR_TANGENT }));
                    return;
                }
            }
        }
        let half = length / 2.0;
        if (b.abs() - half).abs() <= tol {
            self.snap_bulge = Some(half.copysign(if b != 0.0 { b } else { 1.0 }));
            self.bulge_kind = Some("half");
            self.wireframe_color = Some(rgba(COLOR_MIDPOINT));
        }
    }

    /// `_commit(viewport, pts, trim)`.
    fn commit(&mut self, core: &mut Core, pts: &[V3], apex: Option<Apex>, trim: Option<Vec<(V3, V3)>>) {
        let trim_on = trim.as_ref().is_some_and(|t| !t.is_empty());
        if trim_on {
            if let Some((v, p1, _p2)) = self.fillet {
                self.last_fillet_d = Some(p1.sub(v).length(core.h()) as f64);
            }
        }
        self.last_cmd = commit_arc(core, pts, None, trim.as_deref());
        self.last_params = None;
        if let Some(a) = apex {
            if pts.len() >= 3 {
                self.last_params = Some((a, trim));
            }
        }
        self.reset();
    }

    /// `_round_corner`: draw the fillet of distance `d` on `corner` and trim it.
    fn round_corner(&mut self, core: &mut Core, corner: Corner, d: f64, mods: u32) {
        let h = core.h();
        let (vtx, a, b) = corner;
        let (vpos, apos, bpos) = {
            let m = core.mesh();
            (m.position(vtx), m.position(core.mesh().other(a, vtx)), m.position(core.mesh().other(b, vtx)))
        };
        let p1 = vpos.add(apos.sub(vpos).normalized(h).mul(d as f32));
        let p2 = vpos.add(bpos.sub(vpos).normalized(h).mul(d as f32));
        self.start_point = Some(p1);
        self.end_point = Some(p2);
        self.start_edge = Some((a, p1));
        self.fillet = Some((vpos, p1, p2));
        // `_corner_plane`
        let av = apos.sub(vpos);
        let bv = bpos.sub(vpos);
        let n = av.cross(bv);
        self.pl.hover_plane = if (n.length(h) as f64) < 1e-9 { None } else { Some((vpos, n.normalized(h))) };
        self.tangent_dir = Some(vpos.sub(p1).normalized(h));
        let Some(bulge) = self.tangent_bulge(core, None, None) else {
            self.reset();
            return;
        };
        self.snap_bulge = Some(bulge);
        self.bulge_kind = Some("fillet");
        let (pts, apex) = self.points(core, None, Some(bulge), None);
        if pts.len() >= 2 {
            let trim = self.corner_trim(core, Some(mods));
            self.commit(core, &pts, apex, trim);
        } else {
            self.reset();
        }
    }

    /// `_fillet_with_radius`: redraw the armed fillet with `radius`.
    fn fillet_with_radius(&mut self, core: &mut Core, radius: f64) -> bool {
        let h = core.h();
        let (vpos, p1, p2) = self.fillet.expect("a fillet");
        let a = p1.sub(vpos);
        let b = p2.sub(vpos);
        if radius <= 0.0 || (a.length(h) as f64) < 1e-9 || (b.length(h) as f64) < 1e-9 {
            return false;
        }
        let (ua, ub) = (a.normalized(h), b.normalized(h));
        let cos_t = py_max2(-1.0, py_min2(1.0, ua.dot(ub) as f64));
        let theta = cos_t.acos();
        if theta < 1e-3 || (theta - std::f64::consts::PI).abs() < 1e-3 {
            return false; // no corner to round
        }
        let d = radius / (theta / 2.0).tan();
        let m = core.mesh();
        let reach: Vec<f64> = [self.start_edge.map(|s| s.0), self.fillet_edge_b]
            .into_iter()
            .flatten()
            .map(|e| m.position(m.edge(e).v0).sub(m.position(m.edge(e).v1)).length(h) as f64)
            .collect();
        if !reach.is_empty() {
            let shortest = reach.iter().copied().fold(f64::INFINITY, py_min2);
            if d > shortest + 1e-6 {
                core.flash(format!(
                    "Radius {} needs {} of edge on each side of the corner; the shorter one is {}",
                    core.fmt_len(radius),
                    core.fmt_len(d),
                    core.fmt_len(shortest)
                ));
                return true;
            }
        }
        let sp = vpos.add(ua.mul(d as f32));
        let ep = vpos.add(ub.mul(d as f32));
        self.start_point = Some(sp);
        self.end_point = Some(ep);
        if let Some((e, _)) = self.start_edge {
            self.start_edge = Some((e, sp));
        }
        self.fillet = Some((vpos, sp, ep));
        self.tangent_dir = Some(vpos.sub(sp).normalized(h));
        let Some(bulge) = self.tangent_bulge(core, None, None) else { return false };
        self.snap_bulge = Some(bulge);
        self.bulge_kind = Some("fillet");
        let (pts, apex) = self.points(core, None, Some(bulge), None);
        if pts.len() < 2 {
            return false;
        }
        let trim = self.corner_trim(core, None);
        self.commit(core, &pts, apex, trim);
        true
    }
}

impl Planar for ArcTool {
    fn pl(&self) -> &PlaneState {
        &self.pl
    }
    fn pl_mut(&mut self) -> &mut PlaneState {
        &mut self.pl
    }
    /// Start and end: an end snapped off the plane turns the arc to the axis
    /// plane holding both.
    fn plane_points(&self) -> Vec<Option<V3>> {
        match self.start_point {
            None => Vec::new(),
            Some(s) => vec![Some(s), if self.end_point.is_some() { self.end_point } else { self.hover_point }],
        }
    }
}

impl DrawTool for ArcTool {
    fn name(&self) -> &'static str {
        "Arc"
    }

    fn snap_state(&self, _core: &Core) -> ToolState {
        // The chord end is a direction from the start; the bulge is not, so
        // the magnet is off once the end is down.
        ToolState {
            magnetic_axis_deg: if self.end_point.is_none() { Some(MAGNET_DEG) } else { None },
            screen_axis_px: None,
            magnet_off_plane: false,
            axis_source: true,
            axis_source_empty_only: true,
            ..planar_snap_state(&self.pl, self.start_point)
        }
    }

    fn start_point(&self) -> Option<V3> {
        self.start_point
    }
    fn work_plane(&self) -> Option<(V3, V3)> {
        self.pl.work_plane
    }
    fn set_work_plane(&mut self, plane: Option<(V3, V3)>) {
        self.pl.work_plane = plane;
    }
    fn planar(&mut self) -> Option<&mut PlaneState> {
        Some(&mut self.pl)
    }
    fn set_snap_kind(&mut self, kind: Option<SnapKind>) {
        self.pl.snap_kind = kind;
    }
    fn vcb_label(&self) -> String {
        if self.start_point.is_none() {
            return "Segments".into();
        }
        if self.fillet.is_some() { "Radius" } else { "Bulge" }.into()
    }

    fn on_activate(&mut self, _core: &mut Core) {
        self.reset();
        self.last_cmd = None;
        self.last_params = None;
    }

    fn on_deactivate(&mut self, _core: &mut Core) {
        self.reset();
        self.hover_point = None;
        self.last_cmd = None;
        self.last_params = None;
    }

    fn on_click(&mut self, core: &mut Core, ctx: &Ctx) {
        note_plane(&mut self.pl, core);
        self.viewport_seen = true;
        let h = core.h();
        let Some(start) = self.start_point else {
            self.last_cmd = None;
            self.last_params = None;
            self.start_point = Some(ctx.world);
            if self.pl.work_plane.is_none() {
                self.pl.work_plane = locked_work_plane(&self.pl, ctx.world, &core.axes);
            }
            self.start_edge = Self::edge_under(core, ctx.world);
            return;
        };
        if self.end_point.is_none() {
            let end = self.equidistant.map(|e| e.1).unwrap_or(ctx.world);
            if (end.sub(start).length(h) as f64) < 1e-6 {
                return;
            }
            self.end_point = Some(end);
            adopt_snapped_plane(self, core); // the bulge is read on the arc's plane
            self.fillet = None;
            self.fillet_edge_b = None;
            if let Some((vv, p2, b)) = self.equidistant {
                self.fillet = Some((vv, start, p2));
                self.fillet_edge_b = Some(b);
            }
            self.equidistant = None;
            self.tangent_dir = self.tangent_at_start(core);
            if self.tangent_dir.is_none() {
                self.tangent_dir = self.edge_tangent(core, end);
            }
            return;
        }
        let (pts, apex) = self.points(core, Some(ctx.world), None, None);
        if pts.len() >= 2 {
            let trim = self.corner_trim(core, Some(ctx.mods));
            self.commit(core, &pts, apex, trim);
        }
    }

    /// A double-click on the magenta point draws the fillet with its tangent
    /// bulge and trims the corner in one gesture; a double-click near another
    /// two-edge corner repeats the last distance. Anything else is a plain
    /// click.
    fn on_double_click(&mut self, core: &mut Core, ctx: &Ctx) {
        self.viewport_seen = true;
        if self.start_point.is_some() && self.end_point.is_some() && self.fillet.is_some() {
            if let Some(hb) = self.tangent_bulge(core, None, None) {
                self.snap_bulge = Some(hb);
                self.bulge_kind = Some("fillet");
                let (pts, apex) = self.points(core, None, Some(hb), None);
                if pts.len() >= 2 {
                    let trim = self.corner_trim(core, Some(ctx.mods));
                    self.commit(core, &pts, apex, trim);
                    return;
                }
            }
        }
        if self.start_point.is_some() && self.end_point.is_none() && self.last_fillet_d.is_some_and(|d| d != 0.0) {
            if let Some(corner) = self.corner_near(core, ctx.screen) {
                self.reset();
                let d = self.last_fillet_d.expect("a distance");
                self.round_corner(core, corner, d, ctx.mods);
                return;
            }
        }
        self.on_click(core, ctx);
    }

    fn on_hover(&mut self, core: &mut Core, ctx: &Ctx) {
        note_plane(&mut self.pl, core);
        self.viewport_seen = true;
        self.hover_point = Some(ctx.world);
        self.snap_bulge = None;
        self.bulge_kind = None;
        self.equidistant = None;
        self.wireframe_color = lock_color(&self.pl, &core.axes, core.h());
        if self.start_point.is_some() && self.end_point.is_none() {
            // End-point phase: from an edge, the preview is the arc tangent
            // to it (cyan); at the equidistant point of the adjacent edge it
            // turns magenta.
            if self.start_edge.is_some() {
                self.wireframe_color = Some(rgba(COLOR_TANGENT));
                self.equidistant = self.equidistant_near(core, ctx.screen);
                if let Some(eq) = self.equidistant {
                    self.hover_point = Some(eq.1);
                    self.wireframe_color = Some(rgba(COLOR_FILLET));
                }
            }
        } else if self.end_point.is_some() {
            self.snap_bulge_for(core, ctx);
        }
    }

    fn on_value(&mut self, core: &mut Core, value: &VcbValue) -> bool {
        let VcbValue::Num(value) = value else { return false };
        let value = *value;
        if self.start_point.is_none() {
            let mut seg = None;
            let ok = set_segments(core, &mut seg, value);
            if let Some(n) = seg {
                self.segments = n;
            }
            return ok;
        }
        let (Some(_), Some(hover)) = (self.end_point, self.hover_point) else { return false };
        if self.fillet.is_some() {
            // Both tangencies fixed - the corner is being rounded - so the
            // number is the fillet's RADIUS (issue #43).
            return self.fillet_with_radius(core, value);
        }
        let sign = if self.bulge_for(core, hover) < 0.0 { -1.0 } else { 1.0 };
        self.bulge_kind = None;
        let (pts, apex) = self.points(core, None, Some(sign * value), None);
        if pts.len() >= 2 {
            self.commit(core, &pts, apex, None);
        }
        true
    }

    /// The usual "2r": type the RADIUS instead of the bulge.
    fn on_radius_value(&mut self, core: &mut Core, radius: f64) -> Option<bool> {
        let h = core.h();
        let (Some(start), Some(end)) = (self.start_point, self.end_point) else { return Some(false) };
        if radius <= 0.0 {
            return Some(false);
        }
        let half = end.sub(start).length(h) as f64 * 0.5;
        if half < 1e-9 {
            return Some(false);
        }
        if radius < half - 1e-9 {
            core.flash(format!("Radius {} is smaller than half the chord ({})", core.fmt_len(radius), core.fmt_len(half)));
            return Some(true);
        }
        let radius = py_max2(radius, half);
        let hh = radius - py_max2(radius * radius - half * half, 0.0).sqrt();
        let mut sign = 1.0;
        if let Some(hover) = self.hover_point {
            if self.bulge_for(core, hover) < 0.0 {
                sign = -1.0;
            }
        }
        // A typed radius while the arc is magenta keeps the fillet's trim
        // only when it IS the fillet radius.
        let mut trim = None;
        if self.fillet.is_some() {
            if let Some(h_t) = self.tangent_bulge(core, None, None) {
                if (h_t.abs() - hh).abs() <= 1e-6 * py_max2(1.0, radius) {
                    sign = 1.0f64.copysign(h_t);
                    self.bulge_kind = Some("fillet");
                    trim = self.corner_trim(core, None);
                }
            }
        }
        let (pts, apex) = self.points(core, None, Some(sign * hh), None);
        if pts.len() >= 2 {
            self.commit(core, &pts, apex, trim);
        }
        Some(true)
    }

    /// The usual "Ns": the segment count of the arc - the one being drawn,
    /// or the one just drawn, which is rebuilt in place.
    fn on_segments_value(&mut self, core: &mut Core, n: i64) -> Option<bool> {
        let mut seg = None;
        if !set_segments(core, &mut seg, n as f64) {
            return Some(false);
        }
        self.segments = seg.expect("a count");
        let rebuild = match (self.last_cmd, core.history.top_serial(), &self.last_params) {
            (Some(c), Some(top), Some(_)) => c == top,
            _ => false,
        };
        if rebuild {
            core.history_undo();
            let (a, trim) = self.last_params.clone().expect("params");
            let pts2 = arc_3pts_2d((0.0, 0.0), a.apex, a.e2, self.segments);
            let points: Vec<V3> = pts2.iter().map(|&(x, y)| a.origin.add(a.u.mul(x as f32)).add(a.v.mul(y as f32))).collect();
            self.last_cmd = commit_arc(core, &points, None, trim.as_deref());
            self.last_params = Some((a, trim));
        }
        Some(true)
    }

    fn on_cancel(&mut self, _core: &mut Core) {
        self.reset();
    }

    fn on_key(&mut self, core: &mut Core, key: i32, _mods: u32) -> bool {
        plane_lock_key(self, core, key)
    }

    fn rubber_band_lines(&self, core: &Core) -> Vec<(V3, V3)> {
        let (Some(start), Some(hover)) = (self.start_point, self.hover_point) else { return Vec::new() };
        let pairs = |pts: &[V3]| -> Vec<(V3, V3)> { (0..pts.len().saturating_sub(1)).map(|i| (pts[i], pts[i + 1])).collect() };
        if self.end_point.is_none() {
            if self.start_edge.is_some() {
                // Tangent to the start edge, through the cursor.
                let tangent = self.edge_tangent(core, hover);
                if let Some(h) = self.tangent_bulge(core, Some(hover), tangent) {
                    let (pts, _) = self.points(core, None, Some(h), Some(hover));
                    if pts.len() >= 2 {
                        return pairs(&pts);
                    }
                }
            }
            return vec![(start, hover)]; // the chord
        }
        let (pts, _) = self.points(core, Some(hover), None, None);
        pairs(&pts)
    }

    fn value_label(&self, core: &Core) -> Option<(String, V3)> {
        let (start, hover) = (self.start_point?, self.hover_point?);
        let Some(end) = self.end_point else {
            if self.start_edge.is_some() {
                return Some(("Tangent to edge".into(), start.add(hover).mul(0.5f32)));
            }
            return None;
        };
        let mid = start.add(end).mul(0.5f32);
        if let Some(sb) = self.snap_bulge {
            let len = core.fmt_len(sb.abs());
            return Some(match self.bulge_kind {
                Some("half") => (format!("Half circle  {len}"), mid),
                Some("fillet") => (format!("Tangent to edge  {len}"), mid),
                _ => (format!("Tangent at vertex  {len}"), mid),
            });
        }
        let b = self.bulge_for(core, hover);
        Some((format!("Bulge {}", core.fmt_len(b.abs())), mid))
    }

    fn fields(&self) -> Vec<(&'static str, Field)> {
        let mut f = vec![
            ("start_point", Field::point(self.start_point)),
            ("end_point", Field::point(self.end_point)),
            ("hover_point", Field::point(self.hover_point)),
            ("segments", Field::Int(self.segments)),
            ("_tangent_dir", Field::point(self.tangent_dir)),
            ("_snap_bulge", Field::float(self.snap_bulge)),
            ("_bulge_kind", Field::string(self.bulge_kind)),
            ("_fillet", self.fillet.map(|(a, b, c)| Field::List(vec![Field::P(a), Field::P(b), Field::P(c)])).unwrap_or(Field::Null)),
            ("_equidistant", self.equidistant.map(|(a, b, _)| Field::List(vec![Field::P(a), Field::P(b)])).unwrap_or(Field::Null)),
        ];
        f.extend(plane_fields(&self.pl, self.wireframe_color));
        f
    }
}

// ---- 3-Point Arc ----------------------------------------------------------------------------

/// The 3-point arc: the arc passes through all three clicked points.
pub struct ThreePointArcTool {
    pub start_point: Option<V3>,
    pub mid_point: Option<V3>,
    pub hover_point: Option<V3>,
    segments: i64,
    pl: PlaneState,
}

impl Default for ThreePointArcTool {
    fn default() -> Self {
        ThreePointArcTool::new()
    }
}

impl ThreePointArcTool {
    pub fn new() -> ThreePointArcTool {
        ThreePointArcTool { start_point: None, mid_point: None, hover_point: None, segments: SEGMENTS, pl: PlaneState::default() }
    }

    fn reset(&mut self) {
        self.start_point = None;
        self.mid_point = None;
        self.pl.work_plane = None;
        self.pl.hover_plane = None;
        clear_plane_lock(&mut self.pl);
    }

    fn axes(&self, core: &Core) -> (V3, V3) {
        plane_axes(&core.axes, drawing_plane(self, core).1, core.h())
    }

    fn points(&self, core: &Core, end: V3) -> Vec<V3> {
        let (u, v) = self.axes(core);
        let start = self.start_point.expect("a start");
        let to2 = |p: V3| {
            let d = p.sub(start);
            (d.dot(u) as f64, d.dot(v) as f64)
        };
        let (m2, e2) = (to2(self.mid_point.expect("a mid")), to2(end));
        if py_hypot2(e2.0, e2.1) < 1e-9 || py_hypot2(m2.0, m2.1) < 1e-9 {
            return Vec::new();
        }
        arc_3pts_2d((0.0, 0.0), m2, e2, self.segments).iter().map(|&(x, y)| start.add(u.mul(x as f32)).add(v.mul(y as f32))).collect()
    }

    fn set_segments(&mut self, core: &mut Core, n: f64) -> bool {
        let mut seg = None;
        let ok = set_segments(core, &mut seg, n);
        if let Some(s) = seg {
            self.segments = s;
        }
        ok
    }
}

impl Planar for ThreePointArcTool {
    fn pl(&self) -> &PlaneState {
        &self.pl
    }
    fn pl_mut(&mut self) -> &mut PlaneState {
        &mut self.pl
    }
    /// The arc's own points: with all three, a snapped one off the plane
    /// takes the plane through them.
    fn plane_points(&self) -> Vec<Option<V3>> {
        match (self.start_point, self.mid_point) {
            (None, _) => Vec::new(),
            (Some(s), None) => vec![Some(s), self.hover_point],
            (Some(s), Some(m)) => vec![Some(s), Some(m), self.hover_point],
        }
    }
}

impl DrawTool for ThreePointArcTool {
    fn name(&self) -> &'static str {
        "3-Point Arc"
    }

    fn snap_state(&self, _core: &Core) -> ToolState {
        ToolState { magnetic_axis_deg: Some(MAGNET_DEG), screen_axis_px: None, magnet_off_plane: false, axis_source: true, axis_source_empty_only: true, ..planar_snap_state(&self.pl, self.start_point) }
    }

    fn start_point(&self) -> Option<V3> {
        self.start_point
    }
    fn work_plane(&self) -> Option<(V3, V3)> {
        self.pl.work_plane
    }
    fn set_work_plane(&mut self, plane: Option<(V3, V3)>) {
        self.pl.work_plane = plane;
    }
    fn planar(&mut self) -> Option<&mut PlaneState> {
        Some(&mut self.pl)
    }
    fn set_snap_kind(&mut self, kind: Option<SnapKind>) {
        self.pl.snap_kind = kind;
    }
    fn vcb_label(&self) -> String {
        "Segments".into()
    }

    fn on_activate(&mut self, _core: &mut Core) {
        self.reset();
    }

    fn on_deactivate(&mut self, _core: &mut Core) {
        self.reset();
        self.hover_point = None;
    }

    fn on_click(&mut self, core: &mut Core, ctx: &Ctx) {
        note_plane(&mut self.pl, core);
        let Some(start) = self.start_point else {
            self.start_point = Some(ctx.world);
            if self.pl.work_plane.is_none() {
                self.pl.work_plane = locked_work_plane(&self.pl, ctx.world, &core.axes);
            }
            return;
        };
        if self.mid_point.is_none() {
            if (ctx.world.sub(start).length(core.h()) as f64) < 1e-6 {
                return;
            }
            self.mid_point = Some(ctx.world);
            adopt_snapped_plane(self, core);
            return;
        }
        let pts = self.points(core, ctx.world);
        if pts.len() >= 2 {
            commit_arc(core, &pts, None, None);
            self.reset();
        }
    }

    fn on_hover(&mut self, core: &mut Core, ctx: &Ctx) {
        note_plane(&mut self.pl, core);
        self.hover_point = Some(ctx.world);
    }

    fn on_value(&mut self, core: &mut Core, value: &VcbValue) -> bool {
        let VcbValue::Num(v) = value else { return false };
        if self.start_point.is_some() {
            return false;
        }
        self.set_segments(core, *v)
    }

    fn on_segments_value(&mut self, core: &mut Core, n: i64) -> Option<bool> {
        Some(self.set_segments(core, n as f64))
    }

    fn on_cancel(&mut self, _core: &mut Core) {
        self.reset();
    }

    fn on_key(&mut self, core: &mut Core, key: i32, _mods: u32) -> bool {
        plane_lock_key(self, core, key)
    }

    fn rubber_band_lines(&self, core: &Core) -> Vec<(V3, V3)> {
        let (Some(start), Some(hover)) = (self.start_point, self.hover_point) else { return Vec::new() };
        if self.mid_point.is_none() {
            return vec![(start, hover)];
        }
        let pts = self.points(core, hover);
        (0..pts.len().saturating_sub(1)).map(|i| (pts[i], pts[i + 1])).collect()
    }

    fn fields(&self) -> Vec<(&'static str, Field)> {
        let mut f = vec![
            ("start_point", Field::point(self.start_point)),
            ("mid_point", Field::point(self.mid_point)),
            ("hover_point", Field::point(self.hover_point)),
            ("segments", Field::Int(self.segments)),
        ];
        f.extend(plane_fields(&self.pl, None));
        f
    }
}

// ---- Center Arc and Pie -------------------------------------------------------------------------

/// Compass arc: centre -> start point (the radius and 0 degree arm) -> sweep
/// angle. The polyline samples at a 15 degree pitch unless a segment count
/// was typed. With `pie`, the two radii close the wedge into a face.
pub struct CenterArcTool {
    pie: bool,
    pub start_point: Option<V3>,
    pub arm_point: Option<V3>,
    pub hover_point: Option<V3>,
    segments: Option<i64>,
    pl: PlaneState,
}

impl CenterArcTool {
    const PITCH_DEG: f64 = 15.0;

    pub fn center_arc() -> CenterArcTool {
        CenterArcTool { pie: false, start_point: None, arm_point: None, hover_point: None, segments: None, pl: PlaneState::default() }
    }

    /// `PieTool`: the centre arc whose wedge CLOSES.
    pub fn pie() -> CenterArcTool {
        CenterArcTool { pie: true, ..CenterArcTool::center_arc() }
    }

    fn reset(&mut self) {
        self.start_point = None;
        self.arm_point = None;
        self.pl.work_plane = None;
        self.pl.hover_plane = None;
        clear_plane_lock(&mut self.pl);
    }

    fn axes(&self, core: &Core) -> (V3, V3) {
        plane_axes(&core.axes, drawing_plane(self, core).1, core.h())
    }

    /// `_sweep_to(cursor)`: signed sweep (degrees) from the 0 degree arm.
    fn sweep_to(&self, core: &Core, cursor: V3) -> f64 {
        let (u, v) = self.axes(core);
        let start = self.start_point.expect("a centre");
        let a = self.arm_point.expect("an arm").sub(start);
        let b = cursor.sub(start);
        let a0 = (a.dot(v) as f64).atan2(a.dot(u) as f64);
        let b0 = (b.dot(v) as f64).atan2(b.dot(u) as f64);
        let mut deg = (b0 - a0) * (180.0 / std::f64::consts::PI);
        while deg <= -180.0 {
            deg += 360.0;
        }
        while deg > 180.0 {
            deg -= 360.0;
        }
        deg
    }

    fn points(&self, core: &Core, sweep_deg: f64) -> Vec<V3> {
        if sweep_deg.abs() < 1e-6 {
            return Vec::new();
        }
        let (u, v) = self.axes(core);
        let start = self.start_point.expect("a centre");
        let a = self.arm_point.expect("an arm").sub(start);
        let r = py_hypot2(a.dot(u) as f64, a.dot(v) as f64);
        if r < 1e-6 {
            return Vec::new();
        }
        let a0 = (a.dot(v) as f64).atan2(a.dot(u) as f64);
        let steps = match self.segments {
            Some(s) if s != 0 => s,
            _ => ((sweep_deg.abs() / Self::PITCH_DEG).round_ties_even() as i64).max(1),
        };
        (0..=steps)
            .map(|k| {
                let t = a0 + (sweep_deg * (std::f64::consts::PI / 180.0)) * (k as f64) / (steps as f64);
                start.add(u.mul(t.cos() as f32).add(v.mul(t.sin() as f32)).mul(r as f32))
            })
            .collect()
    }

    fn commit(&mut self, core: &mut Core, pts: &[V3]) {
        let close = if self.pie { self.start_point } else { None };
        commit_arc(core, pts, close, None);
        self.reset();
    }

    fn set_segments(&mut self, core: &mut Core, n: f64) -> bool {
        let mut seg = self.segments;
        let ok = set_segments(core, &mut seg, n);
        self.segments = seg;
        ok
    }
}

impl Planar for CenterArcTool {
    fn pl(&self) -> &PlaneState {
        &self.pl
    }
    fn pl_mut(&mut self) -> &mut PlaneState {
        &mut self.pl
    }
    /// Centre and arm: an arm snapped off the plane turns the arc to the axis
    /// plane holding both. The sweep point is only an angle.
    fn plane_points(&self) -> Vec<Option<V3>> {
        match self.start_point {
            None => Vec::new(),
            Some(s) => vec![Some(s), if self.arm_point.is_some() { self.arm_point } else { self.hover_point }],
        }
    }
}

impl DrawTool for CenterArcTool {
    fn name(&self) -> &'static str {
        if self.pie {
            "Pie"
        } else {
            "Center Arc"
        }
    }

    fn snap_state(&self, _core: &Core) -> ToolState {
        ToolState { magnetic_axis_deg: Some(MAGNET_DEG), screen_axis_px: None, magnet_off_plane: false, axis_source: true, axis_source_empty_only: true, ..planar_snap_state(&self.pl, self.start_point) }
    }

    fn start_point(&self) -> Option<V3> {
        self.start_point
    }
    fn work_plane(&self) -> Option<(V3, V3)> {
        self.pl.work_plane
    }
    fn set_work_plane(&mut self, plane: Option<(V3, V3)>) {
        self.pl.work_plane = plane;
    }
    fn planar(&mut self) -> Option<&mut PlaneState> {
        Some(&mut self.pl)
    }
    fn set_snap_kind(&mut self, kind: Option<SnapKind>) {
        self.pl.snap_kind = kind;
    }
    fn vcb_label(&self) -> String {
        if self.start_point.is_none() {
            return "Segments".into();
        }
        if self.arm_point.is_some() { "Angle" } else { "Radius" }.into()
    }

    fn on_activate(&mut self, _core: &mut Core) {
        self.reset();
    }

    fn on_deactivate(&mut self, _core: &mut Core) {
        self.reset();
        self.hover_point = None;
    }

    fn on_click(&mut self, core: &mut Core, ctx: &Ctx) {
        note_plane(&mut self.pl, core);
        let Some(start) = self.start_point else {
            self.start_point = Some(ctx.world);
            if self.pl.work_plane.is_none() {
                self.pl.work_plane = locked_work_plane(&self.pl, ctx.world, &core.axes);
            }
            return;
        };
        if self.arm_point.is_none() {
            if (ctx.world.sub(start).length(core.h()) as f64) < 1e-6 {
                return;
            }
            self.arm_point = Some(ctx.world);
            adopt_snapped_plane(self, core); // the sweep is read on the arc's plane
            return;
        }
        let sweep = self.sweep_to(core, ctx.world);
        let pts = self.points(core, sweep);
        if pts.len() >= 2 {
            self.commit(core, &pts);
        }
    }

    fn on_hover(&mut self, core: &mut Core, ctx: &Ctx) {
        note_plane(&mut self.pl, core);
        self.hover_point = Some(ctx.world);
    }

    fn on_value(&mut self, core: &mut Core, value: &VcbValue) -> bool {
        let VcbValue::Num(value) = value else { return false };
        if self.start_point.is_none() {
            return self.set_segments(core, *value);
        }
        if self.arm_point.is_none() {
            return false;
        }
        let mut sign = -1.0;
        if let Some(hover) = self.hover_point {
            let sweep = self.sweep_to(core, hover);
            sign = if sweep < 0.0 { -1.0 } else { 1.0 };
        }
        let pts = self.points(core, sign * value.abs());
        if pts.len() >= 2 {
            self.commit(core, &pts);
        }
        true
    }

    fn on_segments_value(&mut self, core: &mut Core, n: i64) -> Option<bool> {
        Some(self.set_segments(core, n as f64))
    }

    fn on_cancel(&mut self, _core: &mut Core) {
        self.reset();
    }

    fn on_key(&mut self, core: &mut Core, key: i32, _mods: u32) -> bool {
        plane_lock_key(self, core, key)
    }

    fn rubber_band_lines(&self, core: &Core) -> Vec<(V3, V3)> {
        let (Some(start), Some(hover)) = (self.start_point, self.hover_point) else { return Vec::new() };
        let Some(arm) = self.arm_point else { return vec![(start, hover)] }; // the radius arm
        let mut segments = vec![(start, arm), (start, hover)];
        let pts = self.points(core, self.sweep_to(core, hover));
        segments.extend((0..pts.len().saturating_sub(1)).map(|i| (pts[i], pts[i + 1])));
        segments
    }

    fn value_label(&self, core: &Core) -> Option<(String, V3)> {
        let (hover, start) = (self.hover_point?, self.start_point?);
        if self.arm_point.is_none() {
            let (u, v) = self.axes(core); // the radius drawn, as the circle's
            let d = hover.sub(start);
            let r = py_hypot2(d.dot(u) as f64, d.dot(v) as f64);
            return Some((format!("R {}", core.fmt_len(r)), hover));
        }
        Some((fmt_angle_deg(self.sweep_to(core, hover), true, &core.units), hover))
    }

    fn fields(&self) -> Vec<(&'static str, Field)> {
        let mut f = vec![
            ("start_point", Field::point(self.start_point)),
            ("arm_point", Field::point(self.arm_point)),
            ("hover_point", Field::point(self.hover_point)),
            ("segments", self.segments.map(Field::Int).unwrap_or(Field::Null)),
        ];
        f.extend(plane_fields(&self.pl, None));
        f
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn three_points_make_a_semicircle() {
        let pts = arc_3pts_2d((0.0, 0.0), (1.0, 1.0), (2.0, 0.0), 8);
        assert_eq!(pts.len(), 9);
        assert!((pts[0].0 - 0.0).abs() < 1e-12 && (pts[0].1 - 0.0).abs() < 1e-12);
        assert!((pts[8].0 - 2.0).abs() < 1e-12 && pts[8].1.abs() < 1e-12);
        assert!((pts[4].0 - 1.0).abs() < 1e-12 && (pts[4].1 - 1.0).abs() < 1e-12);
    }

    #[test]
    fn collinear_points_give_the_chord() {
        assert_eq!(arc_3pts_2d((0.0, 0.0), (1.0, 0.0), (2.0, 0.0), 16), vec![(0.0, 0.0), (2.0, 0.0)]);
    }

    #[test]
    fn the_arc_goes_the_way_the_middle_point_says() {
        // Through a point below the chord the arc dips below it.
        let pts = arc_3pts_2d((0.0, 0.0), (1.0, -1.0), (2.0, 0.0), 4);
        assert!(pts.iter().skip(1).take(3).all(|p| p.1 < 0.0));
    }
}
