// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! The Offset tool (`tools/offset.py`): offset a face's boundary in its plane
//! (walls with thickness), or - Edges mode, issue #40 - a run of connected
//! coplanar edges.
//!
//! Offset (F): pick a face, drag (or type a distance) and a parallel loop
//! appears offset from the boundary. The face splits into a ring and an inner
//! face. Inward is the common case; dragging the cursor outside offsets
//! outward. The distance is measured as the cursor's perpendicular distance
//! from the boundary edge nearest the first click, so crossing it flips the
//! side. Edges selected BEFORE the tool is picked up arm it (the first click
//! starts the drag); with nothing selected an edge click takes the run of
//! edges it belongs to. The second click (or a typed distance) commits; a
//! retype right after the commit replaces the offset just made.
//!
//! Where it differs from Python (reported): nothing known. The pick of a face
//! is `pick_face` (loose geometry only, as in Python); the snap engine is not
//! used (`uses_snap` is false).

use std::hint::black_box;

use ingetrazo_formats::pyjson::Value;
use ingetrazo_geom::offset::{offset_chain, offset_regions};
use ingetrazo_geom::triangulate::triangulate;
use ingetrazo_math::pyfloat::{py_max2, py_min2};
use ingetrazo_math::{Hypot, V3};
use ingetrazo_model::mesh::{EdgeId, FaceId, Mesh, VertexId};
use ingetrazo_model::topology::{max_offset_distance, offset_loop};
use ingetrazo_model::units::fmt_len;
use ingetrazo_ops::arrangement::{arrange, plane_basis, point_in_polygon, XY};
use ingetrazo_ops::command::Command;
use ingetrazo_ops::draw::{AddFaceCommand, DeleteFaceCommand, SnapshotCompound, TagCurveCommand};
use ingetrazo_ops::edits::build_add_edges;

use crate::modify_base::{Env, ModifyTool, Probe, Sel, ToolCx, Vcb};

type Loop = Vec<V3>;
type Run = (Vec<V3>, bool);

/// `_last`: the offset just made, for the hot retype.
#[derive(Clone)]
struct Last {
    serial: Option<u64>,
    base_face: Option<FaceId>,
    loop_: Loop,
    normal: Option<V3>,
    chain: bool,
    closed: bool,
    ref_point: Option<V3>,
    ref_inward: Option<V3>,
    sign: f64,
}

#[derive(Default)]
pub struct OffsetTool {
    hovered_face: Option<FaceId>,
    hovered_edge: Option<EdgeId>,
    base_face: Option<FaceId>,
    /// Signed; > 0 inward, < 0 outward.
    distance: f64,
    dragging: bool,
    loop_: Loop,
    normal: Option<V3>,
    ref_point: Option<V3>,
    ref_inward: Option<V3>,
    chain: bool,
    closed: bool,
    armed: bool,
    last: Option<Last>,
}

fn pow2(x: f64) -> f64 {
    x.powf(black_box(2.0))
}

/// `_point_segment_distance`.
fn point_segment_distance(p: V3, a: V3, b: V3, h: Hypot) -> f64 {
    let ab = b.sub(a);
    let length_sq = ab.dot(ab) as f64;
    if length_sq < 1e-12 {
        return p.sub(a).length(h) as f64;
    }
    let t = py_max2(0.0, py_min2(1.0, p.sub(a).dot(ab) as f64 / length_sq));
    p.sub(a.add(ab.mul(t as f32))).length(h) as f64
}

/// `_segments`: consecutive pairs, wrapping when closed.
fn segments(points: &[V3], closed: bool) -> Vec<(V3, V3)> {
    let n = points.len();
    if n < 2 {
        return Vec::new();
    }
    let last = if closed { n } else { n - 1 };
    (0..last).map(|i| (points[i], points[(i + 1) % n])).collect()
}

/// `_newell`: the plane normal of a polyline, `None` when in line.
fn newell(points: &[V3], h: Hypot) -> Option<V3> {
    let n = points.len();
    if n < 3 {
        return None;
    }
    let (mut nx, mut ny, mut nz) = (0.0f64, 0.0f64, 0.0f64);
    for i in 0..n {
        let (p, q) = (points[i], points[(i + 1) % n]);
        let (px, py, pz) = (p.0 as f64, p.1 as f64, p.2 as f64);
        let (qx, qy, qz) = (q.0 as f64, q.1 as f64, q.2 as f64);
        nx += (py - qy) * (pz + qz);
        ny += (pz - qz) * (px + qx);
        nz += (px - qx) * (py + qy);
    }
    let normal = V3(nx as f32, ny as f32, nz as f32);
    let mut span = 0.0f64;
    let mut first = true;
    for p in points {
        let l = p.sub(points[0]).length(h) as f64;
        if first || l > span {
            span = l;
            first = false;
        }
    }
    if span == 0.0 {
        span = 1.0;
    }
    if (normal.length(h) as f64) < 1e-6 * span * span {
        return None;
    }
    Some(normal.normalized(h))
}

/// `_run_from_edges`: order a set of edges into one run.
fn run_from_edges(m: &Mesh, edges: &[EdgeId]) -> Option<Run> {
    let mut nbrs: Vec<(VertexId, Vec<EdgeId>)> = Vec::new();
    for &e in edges {
        let ed = m.edge(e);
        for v in [ed.v0, ed.v1] {
            match nbrs.iter_mut().find(|(x, _)| *x == v) {
                Some((_, es)) => es.push(e),
                None => nbrs.push((v, vec![e])),
            }
        }
    }
    if nbrs.iter().any(|(_, es)| es.len() > 2) {
        return None;
    }
    let ends: Vec<VertexId> = nbrs.iter().filter(|(_, es)| es.len() == 1).map(|(v, _)| *v).collect();
    if ends.len() != 0 && ends.len() != 2 {
        return None;
    }
    let closed = ends.is_empty();
    let start = if closed { m.edge(edges[0]).v0 } else { ends[0] };
    let mut points = vec![m.position(start)];
    let mut seen: Vec<EdgeId> = Vec::new();
    let mut v = start;
    loop {
        let es = &nbrs.iter().find(|(x, _)| *x == v).expect("vertex of the run").1;
        let Some(&nxt) = es.iter().find(|e| !seen.contains(e)) else { break };
        seen.push(nxt);
        v = m.other(nxt, v);
        if closed && v == start {
            break;
        }
        points.push(m.position(v));
    }
    if seen.len() != edges.len() {
        return None;
    }
    Some((points, closed))
}

/// `_run_from_edge`: the run `edge` belongs to, walked both ways through the
/// vertices where exactly two edges meet.
fn run_from_edge(m: &Mesh, edge: EdgeId) -> Run {
    let walk = |start: VertexId| -> (Vec<V3>, bool) {
        let mut out = Vec::new();
        let mut v = start;
        let mut e = edge;
        while m.vertex_ref(v).edges.len() == 2 {
            e = *m.vertex_ref(v).edges.iter().find(|&&x| x != e).expect("two edges");
            if e == edge {
                return (out, true);
            }
            v = m.other(e, v);
            out.push(m.position(v));
        }
        (out, false)
    };
    let ed = m.edge(edge);
    let (fwd, closed) = walk(ed.v1);
    if closed {
        let mut pts = vec![m.position(ed.v0), m.position(ed.v1)];
        pts.extend_from_slice(&fwd[..fwd.len() - 1]);
        return (pts, true);
    }
    let (back, _) = walk(ed.v0);
    let mut pts: Vec<V3> = back.into_iter().rev().collect();
    pts.push(m.position(ed.v0));
    pts.push(m.position(ed.v1));
    pts.extend(fwd);
    (pts, false)
}

/// `_outlines_meet`: whether two closed 2D outlines cross, touch or overlap.
fn outlines_meet(a: &[XY], b: &[XY]) -> bool {
    let eps = 1e-6;
    let orient = |p: XY, q: XY, r: XY| -> i32 {
        let v = (q.0 - p.0) * (r.1 - p.1) - (q.1 - p.1) * (r.0 - p.0);
        if v.abs() < eps {
            0
        } else if v > 0.0 {
            1
        } else {
            -1
        }
    };
    let on_seg = |p: XY, q: XY, r: XY| -> bool {
        py_min2(p.0, r.0) - eps <= q.0 && q.0 <= py_max2(p.0, r.0) + eps && py_min2(p.1, r.1) - eps <= q.1 && q.1 <= py_max2(p.1, r.1) + eps
    };
    for i in 0..a.len() {
        let (p1, p2) = (a[i], a[(i + 1) % a.len()]);
        for j in 0..b.len() {
            let (q1, q2) = (b[j], b[(j + 1) % b.len()]);
            let (o1, o2) = (orient(p1, p2, q1), orient(p1, p2, q2));
            let (o3, o4) = (orient(q1, q2, p1), orient(q1, q2, p2));
            if o1 != o2 && o3 != o4 {
                return true;
            }
            if (o1 == 0 && on_seg(p1, q1, p2)) || (o2 == 0 && on_seg(p1, q2, p2)) || (o3 == 0 && on_seg(q1, p1, q2)) || (o4 == 0 && on_seg(q1, p2, q2)) {
                return true;
            }
        }
    }
    false
}

/// `_place_holes`: `(per_region, strip)`, or `None` when the offset reaches
/// into a hole.
fn place_holes(holes: &[Loop], regions: &[Loop], normal: V3, h: Hypot) -> Option<(Vec<Vec<Loop>>, Vec<Loop>)> {
    let (u, w) = plane_basis(normal, h);
    let origin = regions.first().map(|r| r[0]).unwrap_or(V3(0.0, 0.0, 0.0));
    let flat = |lp: &Loop| -> Vec<XY> { lp.iter().map(|&p| (p.sub(origin).dot(u) as f64, p.sub(origin).dot(w) as f64)).collect() };
    let flat_regions: Vec<Vec<XY>> = regions.iter().map(flat).collect();
    let mut per_region: Vec<Vec<Loop>> = vec![Vec::new(); regions.len()];
    let mut strip: Vec<Loop> = Vec::new();
    for hole in holes {
        let fh = flat(hole);
        if flat_regions.iter().any(|fr| outlines_meet(&fh, fr)) {
            return None;
        }
        let mut placed = false;
        for (i, fr) in flat_regions.iter().enumerate() {
            if point_in_polygon(fh[0], fr) {
                per_region[i].push(hole.clone());
                placed = true;
                break;
            }
            if point_in_polygon(fr[0], &fh) {
                return None;
            }
        }
        if !placed {
            strip.push(hole.clone());
        }
    }
    Some((per_region, strip))
}

/// `_loop_extent`: the bounding box's squared diagonal.
fn loop_extent(lp: &[V3]) -> f64 {
    let ext = |f: &dyn Fn(&V3) -> f32| -> f64 {
        let mut lo = f(&lp[0]) as f64;
        let mut hi = lo;
        for p in lp {
            let x = f(p) as f64;
            lo = py_min2(lo, x);
            hi = py_max2(hi, x);
        }
        hi - lo
    };
    pow2(ext(&|p| p.0)) + pow2(ext(&|p| p.1)) + pow2(ext(&|p| p.2))
}

fn bits(x: f32) -> Value {
    Value::Str(format!("{:08x}", x.to_bits()))
}

fn v3v(p: Option<V3>) -> Value {
    match p {
        Some(q) => Value::Array(vec![bits(q.0), bits(q.1), bits(q.2)]),
        None => Value::Null,
    }
}

impl OffsetTool {
    pub fn new() -> OffsetTool {
        OffsetTool { closed: true, ..Default::default() }
    }

    fn reset(&mut self) {
        self.hovered_face = None;
        self.hovered_edge = None;
        self.base_face = None;
        self.distance = 0.0;
        self.dragging = false;
        self.loop_ = Vec::new();
        self.normal = None;
        self.ref_point = None;
        self.ref_inward = None;
        self.chain = false;
        self.closed = true;
        self.armed = false;
    }

    fn face_loop(env: &Env, f: FaceId) -> Loop {
        let m = env.cx.mesh(env.scene);
        m.face(f).outer.iter().map(|&v| m.position(v)).collect()
    }

    /// `_offset_points`: the slid boundary / run at the current distance.
    fn offset_points(&self, h: Hypot) -> Option<Vec<V3>> {
        let normal = self.normal.expect("normal");
        if self.closed {
            offset_loop(&self.loop_, normal, self.distance, h)
        } else {
            offset_chain(&self.loop_, normal, self.distance, h)
        }
    }

    /// `_cursor_on_plane`.
    fn cursor_on_plane(&self, env: &Env, sx: f64, sy: f64) -> Option<V3> {
        let h = env.cx.hypot;
        let (origin, direction) = env.vp.pixel_to_ray(sx, sy)?;
        let n = self.normal.expect("normal");
        let denom = direction.dot(n) as f64;
        if denom.abs() < 1e-9 {
            return None;
        }
        let anchor = match self.base_face {
            Some(f) => ingetrazo_geom::meshface::centroid(&Self::face_loop(env, f)),
            None => self.loop_[0],
        };
        let _ = h;
        let t = anchor.sub(origin).dot(n) as f64 / denom;
        Some(origin.add(direction.mul(t as f32)))
    }

    /// `_pick_reference`.
    fn pick_reference(&mut self, env: &Env, sx: f64, sy: f64) {
        let h = env.cx.hypot;
        let cursor = self.cursor_on_plane(env, sx, sy);
        let segs = segments(&self.loop_, self.closed);
        let (mut best_i, mut best_d) = (0usize, f64::INFINITY);
        for (i, &(a, b)) in segs.iter().enumerate() {
            let d = match cursor {
                Some(c) => point_segment_distance(c, a, b, h),
                None => 0.0,
            };
            if d < best_d {
                best_d = d;
                best_i = i;
            }
        }
        let (a, b) = segs[best_i];
        let e = b.sub(a).normalized(h);
        self.ref_point = Some(a);
        self.ref_inward = Some(self.normal.expect("normal").normalized(h).cross(e).normalized(h));
    }

    /// `_take_run`: adopt `run` as the thing to offset, or say why not.
    fn take_run(&mut self, env: &mut Env, run: Option<Run>) -> bool {
        let h = env.cx.hypot;
        let Some((points, closed)) = run else {
            env.flash_for("Offset needs one connected run of edges — a polyline, or a closed loop.", 5000);
            return false;
        };
        let Some(normal) = newell(&points, h) else {
            if points.len() == 2 && !closed {
                env.flash_for(
                    "This line is on its own — nothing is joined to its ends, so there is no chain to offset. Draw the missing line, or select the edges to offset.",
                    6000,
                );
                if let Some(e) = self.hovered_edge {
                    env.select(&[Sel::Edge(e)], "replace");
                }
                return false;
            }
            env.flash_for("Offset needs at least two edges that are not in line — a straight run has no plane to offset in.", 5000);
            return false;
        };
        self.base_face = None;
        self.loop_ = points;
        self.normal = Some(normal);
        self.chain = true;
        self.closed = closed;
        true
    }

    /// `_arm_from_selection`.
    fn arm_from_selection(&mut self, env: &mut Env) {
        let edges: Vec<EdgeId> = env.sel.edges();
        if edges.len() < 2 || edges.len() != env.sel.len() {
            return;
        }
        let run = run_from_edges(env.cx.mesh(env.scene), &edges);
        if run.is_none() {
            return;
        }
        if self.take_run(env, run) {
            self.armed = true;
            env.flash_for(format!("{} edges selected — click to start the offset", edges.len()), 4000);
        }
    }

    fn sign(&self) -> f64 {
        if self.distance < 0.0 {
            -1.0
        } else {
            1.0
        }
    }

    fn remember(&self, serial: Option<u64>, chain: bool) -> Last {
        Last {
            serial,
            base_face: if chain { None } else { self.base_face },
            loop_: self.loop_.clone(),
            normal: self.normal,
            chain,
            closed: if chain { self.closed } else { true },
            ref_point: self.ref_point,
            ref_inward: self.ref_inward,
            sign: self.sign(),
        }
    }

    fn commit(&mut self, env: &mut Env) {
        if self.chain {
            self.commit_chain(env);
            return;
        }
        let h = env.cx.hypot;
        let normal = self.normal.expect("normal");
        let mut regions = offset_regions(
            &self.loop_,
            normal,
            self.distance,
            h,
            |segs, origin, nrm| {
                let a = arrange(segs, origin, nrm, h);
                Some(
                    a.faces
                        .iter()
                        .map(|(outer, holes)| (outer.iter().map(|&p| a.to3d(p)).collect(), holes.iter().map(|hl| hl.iter().map(|&p| a.to3d(p)).collect()).collect()))
                        .collect(),
                )
            },
            |l, n| Some(triangulate(l, &[], Some(n), h)),
        );
        if regions.is_empty() {
            if let Some(fallback) = offset_loop(&self.loop_, normal, self.distance, h) {
                regions = vec![(fallback, Vec::new())];
            }
        }
        if regions.is_empty() {
            self.refuse(env);
            return;
        }
        let base = self.base_face.expect("base face");
        let (attrs, own_holes) = {
            let m = env.cx.mesh(env.scene);
            let f = m.face(base);
            (f.attrs.clone(), f.holes.iter().map(|hl| hl.iter().map(|&v| m.position(v)).collect::<Loop>()).collect::<Vec<_>>())
        };
        let (outer, inners, inner_holes, strip_holes): (Loop, Vec<Loop>, Vec<Vec<Loop>>, Vec<Loop>);
        if self.distance > 0.0 {
            outer = self.loop_.clone();
            inners = regions.iter().map(|(l, _)| l.clone()).collect();
            match place_holes(&own_holes, &inners, normal, h) {
                None => {
                    env.flash_for(format!("{} runs into a hole of this face — try a smaller offset", fmt_len(self.distance.abs(), env.units)), 5000);
                    self.reset();
                    return;
                }
                Some((ih, sh)) => {
                    inner_holes = ih;
                    strip_holes = sh;
                }
            }
        } else {
            let mut best = 0usize;
            let mut best_k = loop_extent(&regions[0].0);
            for (i, (l, _)) in regions.iter().enumerate().skip(1) {
                let k = loop_extent(l);
                if k > best_k {
                    best_k = k;
                    best = i;
                }
            }
            outer = regions[best].0.clone();
            inners = vec![self.loop_.clone()];
            inner_holes = vec![own_holes];
            strip_holes = Vec::new();
        }
        let mut commands: Vec<Box<dyn Command>> = vec![Box::new(DeleteFaceCommand::new(Self::face_loop(env, base)))];
        let mut ring_holes: Vec<Loop> = inners.clone();
        ring_holes.extend(strip_holes);
        commands.push(Box::new(AddFaceCommand::new(outer, false, Some(ring_holes), Some(attrs.clone()))));
        for (lp, holes) in inners.iter().zip(inner_holes) {
            commands.push(Box::new(AddFaceCommand::new(lp.clone(), false, if holes.is_empty() { None } else { Some(holes) }, Some(attrs.clone()))));
        }
        if regions.len() == 1 && regions[0].0.len() == self.loop_.len() {
            let tags = self.curve_tags(env.cx.mesh(env.scene), &regions[0].0);
            commands.extend(tags);
        }
        let cmd = SnapshotCompound::new(commands);
        let serial = env.execute_serial(Box::new(cmd));
        self.last = Some(self.remember(serial, false));
        self.reset();
    }

    fn commit_chain(&mut self, env: &mut Env) {
        let h = env.cx.hypot;
        let Some(off) = self.offset_points(h) else {
            env.flash_for(format!("{} collapses this run — try a smaller offset", fmt_len(self.distance.abs(), env.units)), 5000);
            self.reset();
            return;
        };
        let eye = env.vp.eye();
        let cmd = build_add_edges(env.scene, env.cx, &segments(&off, self.closed), true, Vec::new(), Some(eye), false);
        let serial = env.execute_serial(cmd);
        self.last = Some(self.remember(serial, true));
        self.reset();
    }

    fn refuse(&mut self, env: &mut Env) {
        self.last = None;
        let h = env.cx.hypot;
        let sign = self.sign();
        let room = max_offset_distance(&self.loop_, self.normal.expect("normal"), sign, 10.0, h);
        let side = if sign > 0.0 { "inward" } else { "outward" };
        let d = fmt_len(self.distance.abs(), env.units);
        if room <= 1e-4 {
            env.flash_for(format!("{d} closes this face — it takes no offset {side}"), 5000);
        } else {
            env.flash_for(format!("{d} closes this face — {side} it takes at most {}", fmt_len(room, env.units)), 5000);
        }
        self.reset();
    }

    /// `_curve_tags`.
    fn curve_tags(&self, m: &Mesh, off: &[V3]) -> Vec<Box<dyn Command>> {
        let lp = &self.loop_;
        let n = lp.len();
        let mut ids: Vec<Option<i64>> = Vec::new();
        for i in 0..n {
            let va = m.vertex_at(lp[i]);
            let vb = m.vertex_at(lp[(i + 1) % n]);
            let e = match (va, vb) {
                (Some(a), Some(b)) => m.find_edge(a, b),
                _ => None,
            };
            ids.push(e.and_then(|e| m.edge(e).curve));
        }
        if ids[0].is_some() && ids.iter().all(|c| *c == ids[0]) {
            return vec![Box::new(TagCurveCommand { pts: off.to_vec(), closed: true })];
        }
        let mut tags: Vec<Box<dyn Command>> = Vec::new();
        for k in 0..n {
            if ids[k].is_none() || ids[(k + n - 1) % n] == ids[k] {
                continue;
            }
            let mut pts = vec![off[k]];
            let mut j = k;
            while ids[j] == ids[k] {
                pts.push(off[(j + 1) % n]);
                j = (j + 1) % n;
            }
            tags.push(Box::new(TagCurveCommand { pts, closed: false }));
        }
        tags
    }
}

impl ModifyTool for OffsetTool {
    fn on_activate(&mut self, env: &mut Env) {
        self.reset();
        self.last = None;
        self.arm_from_selection(env);
    }

    fn on_deactivate(&mut self, env: &mut Env) {
        env.vp.set_hover(None, None);
        self.reset();
        self.last = None;
    }

    fn on_hover(&mut self, env: &mut Env, cx: &ToolCx) {
        if !self.dragging {
            if self.armed {
                return;
            }
            self.hovered_face = env.vp.pick_face(env.scene, env.cx, cx.screen.0, cx.screen.1);
            self.hovered_edge = None;
            if self.hovered_face.is_none() {
                self.hovered_edge = env.vp.pick_edge(env.scene, env.cx, cx.screen.0, cx.screen.1);
            }
            let hov = self.hovered_face.map(Sel::Face).or(self.hovered_edge.map(Sel::Edge));
            env.vp.set_hover(hov, None);
            return;
        }
        let cursor = self.cursor_on_plane(env, cx.screen.0, cx.screen.1);
        if let (Some(c), Some(rp), Some(ri)) = (cursor, self.ref_point, self.ref_inward) {
            self.distance = c.sub(rp).dot(ri) as f64;
        }
    }

    fn on_click(&mut self, env: &mut Env, cx: &ToolCx) {
        let h = env.cx.hypot;
        if !self.dragging {
            self.last = None;
            if self.armed {
                self.armed = false;
            } else if let Some(face) = self.hovered_face {
                let lp = Self::face_loop(env, face);
                if lp.len() < 3 {
                    return;
                }
                self.base_face = Some(face);
                self.loop_ = lp;
                self.normal = Some(env.cx.mesh(env.scene).face_normal(face, h));
                self.chain = false;
                self.closed = true;
            } else if let Some(edge) = self.hovered_edge {
                let run = run_from_edge(env.cx.mesh(env.scene), edge);
                if !self.take_run(env, Some(run)) {
                    return;
                }
            } else {
                return;
            }
            self.distance = 0.0;
            self.dragging = true;
            self.pick_reference(env, cx.screen.0, cx.screen.1);
            env.vp.set_hover(None, None);
            return;
        }
        if self.distance.abs() < 1e-6 {
            return;
        }
        self.commit(env);
    }

    fn on_value(&mut self, env: &mut Env, value: &Vcb) -> bool {
        let v = match value {
            Vcb::Num(v) => *v,
            _ => return false,
        };
        if v == 0.0 {
            return false;
        }
        if self.dragging && !self.loop_.is_empty() {
            let sign = if self.distance < 0.0 { -1.0 } else { 1.0 };
            self.distance = sign * v;
            self.commit(env);
            return true;
        }
        let Some(last) = self.last.clone() else { return false };
        if !env.is_top(last.serial) {
            self.last = None;
            return false;
        }
        env.undo();
        self.base_face = last.base_face;
        self.loop_ = last.loop_.clone();
        self.normal = last.normal;
        self.chain = last.chain;
        self.closed = last.closed;
        self.ref_point = last.ref_point;
        self.ref_inward = last.ref_inward;
        self.distance = last.sign * v;
        self.dragging = true;
        let depth = env.history.undo_len();
        self.commit(env);
        if env.history.undo_len() == depth {
            env.redo();
            self.last = Some(last);
        }
        true
    }

    fn on_cancel(&mut self, env: &mut Env) {
        env.vp.set_hover(None, None);
        self.reset();
        self.last = None;
    }

    fn probe(&self, env: &Env) -> Probe {
        let h = env.cx.hypot;
        let mut p = Probe::default();
        p.rb = Some(if !self.dragging || self.loop_.is_empty() || self.distance.abs() < 1e-6 {
            Vec::new()
        } else {
            match self.offset_points(h) {
                Some(off) => segments(&off, self.closed),
                None => Vec::new(),
            }
        });
        p.vl = Some(match (self.dragging, self.ref_point) {
            (true, Some(rp)) => {
                let inward = self.ref_inward.unwrap_or(V3(0.0, 0.0, 0.0));
                let mid = rp.add(inward.mul((self.distance * 0.5) as f32));
                Some((fmt_len(self.distance.abs(), env.units), mid))
            }
            _ => None,
        });
        p.wf = Some([0.13, 0.17, 0.23, 1.0]);
        p.uses_snap = false;
        let m = env.cx.mesh(env.scene);
        let face_ref = |f: Option<FaceId>| match f {
            None => Value::Null,
            Some(f) => match m.faces.iter().position(|x| *x == f) {
                Some(i) => Value::Array(vec![Value::Str("face".into()), Value::int(i as i64)]),
                None => Value::Array(vec![Value::Str("unknown".into()), Value::Str("Face".into())]),
            },
        };
        let edge_ref = |e: Option<EdgeId>| match e {
            None => Value::Null,
            Some(e) => match m.edges.iter().position(|x| *x == e) {
                Some(i) => Value::Array(vec![Value::Str("edge".into()), Value::int(i as i64)]),
                None => Value::Array(vec![Value::Str("unknown".into()), Value::Str("Edge".into())]),
            },
        };
        p.x = Some(Value::Object(vec![
            ("hovered_face".into(), face_ref(self.hovered_face)),
            ("hovered_edge".into(), edge_ref(self.hovered_edge)),
            ("base_face".into(), face_ref(self.base_face)),
            ("distance".into(), Value::Float(self.distance)),
            ("dragging".into(), Value::Bool(self.dragging)),
            ("n_loop".into(), Value::int(self.loop_.len() as i64)),
            ("normal".into(), v3v(self.normal)),
            ("ref_point".into(), v3v(self.ref_point)),
            ("ref_inward".into(), v3v(self.ref_inward)),
            ("chain".into(), Value::Bool(self.chain)),
            ("closed".into(), Value::Bool(self.closed)),
            ("armed".into(), Value::Bool(self.armed)),
            ("last".into(), Value::Bool(self.last.is_some())),
            ("last_chain".into(), self.last.as_ref().map(|l| Value::Bool(l.chain)).unwrap_or(Value::Null)),
            ("last_sign".into(), self.last.as_ref().map(|l| Value::Float(l.sign)).unwrap_or(Value::Null)),
        ]));
        p
    }
}
