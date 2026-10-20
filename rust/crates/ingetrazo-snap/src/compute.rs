// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! `core/snap.py`: the inference rules, in the order and with the
//! arithmetic of the Python.
//!
//! Number kinds are kept apart on purpose. A `QVector3D` is float32 and
//! every vector operation is `V3`'s; `length()`, `dotProduct()` and `x()`
//! hand Python a float32 widened to a float, so the scalar arithmetic that
//! follows (`bb * bb`, `cc - bb * bb`, `s / ma`, `math.hypot`) is float64 of
//! those widened values; a vector times a Python float narrows the float
//! back to float32 first (`.mul(x as f32)`). Python's iteration orders are
//! kept: the axes are walked x, y, z; `max(key=...)` keeps the first of
//! equal keys; `sort()` is stable.

use ingetrazo_geom::bevel_profiles::py_hypot2 as hypot;
use ingetrazo_math::pyfloat::{py_max2, py_min2};
use ingetrazo_math::V3;
use ingetrazo_model::topology::{segment_intersection, SPLIT_TOLERANCE};

use crate::types::*;

/// `math.radians`.
#[inline]
pub fn radians(d: f64) -> f64 {
    d * (std::f64::consts::PI / 180.0)
}

/// `_detect_axis_alignment`: the axis the start -> candidate direction is
/// within `angle_deg` of, if any.
pub fn detect_axis_alignment(cx: &Cx, start: V3, candidate: V3, angle_deg: f64) -> Option<Axis> {
    let mut delta = candidate.sub(start);
    let length = cx.len(delta);
    if length < 1e-6 {
        return None;
    }
    let cos_thresh = radians(angle_deg).cos();
    if !cx.axes.world {
        delta = cx.axes.to_local(delta);
    }
    let nx = (delta.0.abs() as f64) / length;
    let ny = (delta.1.abs() as f64) / length;
    let nz = (delta.2.abs() as f64) / length;
    // `candidates.sort(reverse=True)` on (value, name) tuples: a tie on the
    // value goes to the greater name.
    let mut c = [(nx, Axis::X), (ny, Axis::Y), (nz, Axis::Z)];
    c.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    if c[0].0 >= cos_thresh {
        return Some(c[0].1);
    }
    None
}

/// How far an axis may sit out of a plane and still count as IN it.
const IN_PLANE_TOL: f64 = 1e-3;
/// An axis whose screen image is shorter than this points at the camera.
const MIN_AXIS_SCREEN_PX: f64 = 12.0;
/// How far along an axis the resolved point may land, in cursor distances.
const MAX_AXIS_REACH: f64 = 5.0;
/// Two point candidates closer than this on screen are a tie.
const TIE_PX: f64 = 0.5;

/// The Alt toggle (`linear_inference_mode`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LinearMode {
    All,
    Off,
    ParallelPerp,
}

/// `reference_mode`: the Down arrow's lock. Anything but "perpendicular"
/// locks parallel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RefMode {
    Parallel,
    Perpendicular,
}

/// `compute_snap`'s arguments (the callbacks live in the [`SnapEnv`]).
#[derive(Clone, Debug)]
pub struct SnapParams<'a> {
    pub candidate_world: V3,
    pub candidate_pixel: (f64, f64),
    pub edges: &'a [SnapEdge],
    pub threshold_px: f64,
    /// `project_onto_line is not None`.
    pub has_project: bool,
    pub chain_first_point: Option<V3>,
    /// `chain_first_point is start_point` (object identity in Python).
    pub chain_first_is_start: bool,
    pub start_point: Option<V3>,
    pub axis_lock: Option<Axis>,
    pub shift_held: bool,
    pub reference_edge: Option<EdgeRef>,
    pub reference_mode: Option<RefMode>,
    pub inference_angle_deg: f64,
    pub screen_axis_px: Option<f64>,
    /// `is_occluded is not None`.
    pub has_occlusion: bool,
    pub face_under_cursor: bool,
    pub edge_threshold_px: Option<f64>,
    pub magnetic_axis_deg: Option<f64>,
    pub acquired_edge: Option<EdgeRef>,
    pub acquired_point: Option<V3>,
    pub acquired_face_normal: Option<V3>,
    pub acquired_points: &'a [V3],
    pub shift_lock_dir: Option<V3>,
    pub shift_lock_color: Option<Rgb>,
    pub linear_mode: LinearMode,
    pub work_plane_normal: Option<V3>,
    pub radial_arm: bool,
    pub axis_in_plane: bool,
}

impl<'a> SnapParams<'a> {
    /// The defaults of `compute_snap`'s keyword arguments.
    pub fn new(candidate_world: V3, candidate_pixel: (f64, f64), edges: &'a [SnapEdge], threshold_px: f64) -> Self {
        SnapParams {
            candidate_world,
            candidate_pixel,
            edges,
            threshold_px,
            has_project: false,
            chain_first_point: None,
            chain_first_is_start: false,
            start_point: None,
            axis_lock: None,
            shift_held: false,
            reference_edge: None,
            reference_mode: None,
            inference_angle_deg: 3.0,
            screen_axis_px: None,
            has_occlusion: false,
            face_under_cursor: false,
            edge_threshold_px: None,
            magnetic_axis_deg: None,
            acquired_edge: None,
            acquired_point: None,
            acquired_face_normal: None,
            acquired_points: &[],
            shift_lock_dir: None,
            shift_lock_color: None,
            linear_mode: LinearMode::All,
            work_plane_normal: None,
            radial_arm: false,
            axis_in_plane: false,
        }
    }
}

/// The engine's working set: numeric context, the callbacks and which of
/// them Python was given.
struct E<'a> {
    cx: Cx,
    env: &'a dyn SnapEnv,
    has_project: bool,
    has_occ: bool,
}

type Px = Option<(f64, f64)>;

impl<'a> E<'a> {
    #[inline]
    fn len(&self, v: V3) -> f64 {
        self.cx.len(v)
    }
    #[inline]
    fn dot(&self, a: V3, b: V3) -> f64 {
        self.cx.dot(a, b)
    }
    #[inline]
    fn norm(&self, v: V3) -> V3 {
        self.cx.norm(v)
    }
    #[inline]
    fn w2p(&self, w: V3) -> Px {
        self.env.world_to_pixel(w)
    }
    #[inline]
    fn proj(&self, s: V3, d: V3) -> V3 {
        self.env.project_onto_line(s, d)
    }
    /// `is_occluded is not None and is_occluded(p)`.
    #[inline]
    fn occ(&self, p: V3) -> bool {
        self.has_occ && self.env.is_occluded(p)
    }
    #[inline]
    fn ax(&self, a: Axis) -> V3 {
        self.cx.axes.axis(a)
    }

    // ---- helpers ---------------------------------------------------------

    /// `_detect_axis_alignment`.
    fn detect_axis_alignment(&self, start: V3, candidate: V3, angle_deg: f64) -> Option<Axis> {
        detect_axis_alignment(&self.cx, start, candidate, angle_deg)
    }

    /// `_detect_axis_on_screen`.
    fn detect_axis_on_screen(&self, start: V3, candidate_world: V3, cpix: (f64, f64), threshold_px: f64) -> Option<Axis> {
        let reach = self.len(candidate_world.sub(start));
        if reach < 1e-6 {
            return None;
        }
        let (sx, sy) = self.w2p(start)?;
        let (cx, cy) = cpix;
        let mut best: Option<(f64, Axis)> = None;
        for name in Axis::ALL {
            let Some((ax, ay)) = self.w2p(start.add(self.ax(name).mul(reach as f32))) else {
                continue;
            };
            let (dx, dy) = (ax - sx, ay - sy);
            let span = hypot(dx, dy);
            if span < MIN_AXIS_SCREEN_PX {
                continue;
            }
            let dist = ((cx - sx) * dy - (cy - sy) * dx).abs() / span;
            if dist <= threshold_px && (best.is_none() || dist < best.unwrap().0) {
                best = Some((dist, name));
            }
        }
        best.map(|b| b.1)
    }

    /// `_axis_in_plane`.
    fn axis_in_plane(&self, name: Option<Axis>, normal: Option<V3>) -> Option<Axis> {
        let (name, normal) = match (name, normal) {
            (Some(a), Some(n)) => (a, n),
            (n, _) => return n,
        };
        if self.len(normal) < 1e-9 {
            return Some(name);
        }
        let n = self.norm(normal);
        if self.dot(self.ax(name), n).abs() > IN_PLANE_TOL {
            return None;
        }
        Some(name)
    }

    /// `_engaged_axis`.
    #[allow(clippy::too_many_arguments)]
    fn engaged_axis(
        &self,
        start_point: Option<V3>,
        candidate_world: V3,
        cpix: (f64, f64),
        magnetic_axis_deg: Option<f64>,
        screen_axis_px: Option<f64>,
    ) -> Option<Axis> {
        let start_point = start_point?;
        if !self.has_project {
            return None;
        }
        if let Some(px) = screen_axis_px {
            if let Some(axis) = self.detect_axis_on_screen(start_point, candidate_world, cpix, px) {
                let reach = self.len(candidate_world.sub(start_point));
                let locked = self.proj(start_point, self.ax(axis));
                if self.len(locked.sub(start_point)) <= MAX_AXIS_REACH * reach {
                    return Some(axis);
                }
            }
        }
        if let Some(deg) = magnetic_axis_deg {
            return self.detect_axis_alignment(start_point, candidate_world, deg);
        }
        None
    }

    /// `_direction_from_edge`.
    fn direction_from_edge(&self, edge: EdgeRef, mode: RefMode, plane_normal: Option<V3>) -> Option<V3> {
        let direction = edge.b.sub(edge.a);
        if self.len(direction) < 1e-6 {
            return None;
        }
        if mode == RefMode::Perpendicular {
            let normal = plane_normal.unwrap_or(self.ax(Axis::Z));
            let perp = normal.cross(direction);
            if self.len(perp) < 1e-9 {
                return None;
            }
            return Some(perp.div(self.len(perp) as f32));
        }
        Some(self.norm(direction))
    }

    /// `_on_edge_point`.
    fn on_edge_point(&self, a: V3, b: V3, t: f64) -> V3 {
        let ab = b.sub(a);
        if self.has_project && self.len(ab) > 1e-9 {
            let proj = self.proj(a, ab);
            let tt = self.dot(proj.sub(a), ab) / self.dot(ab, ab);
            return a.add(ab.mul(py_max2(0.0, py_min2(1.0, tt)) as f32));
        }
        a.add(ab.mul(t as f32))
    }

    /// `_line_segment_intersection` with `tol = 1e-3`.
    fn line_segment_intersection(&self, p: V3, u: V3, a: V3, b: V3) -> Option<V3> {
        let tol = 1e-3;
        let v = b.sub(a);
        if self.len(v) < 1e-9 {
            return None;
        }
        let w0 = p.sub(a);
        let bb = self.dot(u, v);
        let cc = self.dot(v, v);
        let dd = self.dot(u, w0);
        let ee = self.dot(v, w0);
        let denom = cc - bb * bb;
        if denom.abs() < 1e-12 {
            return None;
        }
        let sc = (bb * ee - cc * dd) / denom;
        let tc = (ee - bb * dd) / denom;
        if tc < -1e-6 || tc > 1.0 + 1e-6 {
            return None;
        }
        let on_line = p.add(u.mul(sc as f32));
        let on_seg = a.add(v.mul(tc as f32));
        if self.len(on_line.sub(on_seg)) > tol {
            return None;
        }
        Some(on_seg)
    }

    /// `_point_on_segment_world` with `tol = 1e-3`.
    fn point_on_segment_world(&self, p: V3, a: V3, b: V3) -> bool {
        let tol = 1e-3;
        let ab = b.sub(a);
        let length_sq = self.dot(ab, ab);
        if length_sq < 1e-12 {
            return self.len(p.sub(a)) < tol;
        }
        let t = self.dot(p.sub(a), ab) / length_sq;
        if t < -1e-6 || t > 1.0 + 1e-6 {
            return false;
        }
        self.len(p.sub(a.add(ab.mul(t as f32)))) < tol
    }

    /// `_vertex_on_line` with `tol = 1e-4`.
    fn vertex_on_line(&self, vertex: V3, line_start: V3, line_dir: V3) -> bool {
        let rel = vertex.sub(line_start);
        let proj = line_dir.mul(self.dot(rel, line_dir) as f32);
        self.len(rel.sub(proj)) < 1e-4
    }

    /// `max(AXES, key=lambda k: abs(dot(u, AXES[k])))`: the first of equal
    /// keys.
    fn most_aligned_axis(&self, u: V3) -> Axis {
        let mut best = Axis::X;
        let mut best_key = self.dot(u, self.ax(Axis::X)).abs();
        for k in [Axis::Y, Axis::Z] {
            let key = self.dot(u, self.ax(k)).abs();
            if key > best_key {
                best = k;
                best_key = key;
            }
        }
        best
    }

    // ---- inferences ------------------------------------------------------

    /// `_two_point_snap`.
    fn two_point_snap(&self, refs: &[V3], cpix: (f64, f64), threshold_px: f64) -> Option<SnapResult> {
        let pts = refs;
        if pts.len() < 2 {
            return None;
        }
        let (cx, cy) = cpix;
        type Best = (f64, V3, V3, Axis, V3, Axis);
        let mut best: Option<Best> = None;
        for i in 0..pts.len() {
            for j in i + 1..pts.len() {
                let (p, q) = (pts[i], pts[j]);
                for ka in Axis::ALL {
                    let a = self.ax(ka);
                    for kb in Axis::ALL {
                        if ka == kb {
                            continue;
                        }
                        let b = self.ax(kb);
                        let w = p.sub(q);
                        let ab = self.dot(a, b);
                        let denom = 1.0 - ab * ab;
                        if denom < 1e-9 {
                            continue;
                        }
                        let wa = self.dot(w, a);
                        let wb = self.dot(w, b);
                        let s_ = (ab * wb - wa) / denom;
                        let t_ = (wb - ab * wa) / denom;
                        let pa = p.add(a.mul(s_ as f32));
                        let pb = q.add(b.mul(t_ as f32));
                        if self.len(pa.sub(pb)) > 1e-3 {
                            continue;
                        }
                        if s_.abs() < 1e-3 || t_.abs() < 1e-3 {
                            continue;
                        }
                        let Some(px) = self.w2p(pa) else { continue };
                        let d = hypot(px.0 - cx, px.1 - cy);
                        if d > threshold_px {
                            continue;
                        }
                        if self.occ(pa) {
                            continue;
                        }
                        if best.is_none() || d < best.unwrap().0 {
                            best = Some((d, pa, p, ka, q, kb));
                        }
                    }
                }
            }
        }
        let (_, cross, p, ka, q, kb) = best?;
        let mut r = SnapResult::new(cross, SnapKind::FromPoint, COLOR_ENDPOINT);
        r.guide = Some((p, cross));
        r.guide_color = Some(ka.color());
        r.guides = Some(vec![(q, cross, kb.color())]);
        Some(r)
    }

    /// `_first_point_from_point`.
    fn first_point_from_point(
        &self,
        r: V3,
        candidate: V3,
        cpix: (f64, f64),
        threshold_px: f64,
        plane_normal: Option<V3>,
    ) -> Option<SnapResult> {
        let (cx, cy) = cpix;
        let mut best: Option<(f64, V3, Axis)> = None;
        let n = match plane_normal {
            Some(pn) if self.len(pn) > 1e-9 => Some(self.norm(pn)),
            _ => None,
        };
        for axis in Axis::ALL {
            let a = self.ax(axis);
            let an = match n {
                Some(n) => self.dot(a, n),
                None => 0.0,
            };
            let foot = if an.abs() < 1e-6 {
                let s = self.dot(candidate.sub(r), a);
                if s.abs() < 1e-3 {
                    continue;
                }
                if let Some(n) = n {
                    if self.dot(r.sub(candidate), n).abs() > 1e-4 {
                        continue;
                    }
                }
                r.add(a.mul(s as f32))
            } else {
                let t = self.dot(candidate.sub(r), n.unwrap()) / an;
                if t.abs() < 1e-3 {
                    continue;
                }
                r.add(a.mul(t as f32))
            };
            let Some(fp) = self.w2p(foot) else { continue };
            let d = hypot(fp.0 - cx, fp.1 - cy);
            if d > threshold_px {
                continue;
            }
            if self.occ(foot) {
                continue;
            }
            if best.is_none() || d < best.unwrap().0 {
                best = Some((d, foot, axis));
            }
        }
        let (_, foot, axis) = best?;
        let mut res = SnapResult::new(foot, SnapKind::FromPoint, COLOR_ENDPOINT);
        res.guide = Some((r, foot));
        res.guide_color = Some(axis.color());
        Some(res)
    }

    /// `_edge_from_point_crossing`.
    fn edge_from_point_crossing(&self, edge: &SnapEdge, r: V3, cpix: (f64, f64), threshold_px: f64) -> Option<SnapResult> {
        let (cx, cy) = cpix;
        let mut best: Option<(f64, V3, Axis)> = None;
        for axis in Axis::ALL {
            let a = self.ax(axis);
            let Some(hit) = self.line_segment_intersection(r, a, edge.a, edge.b) else { continue };
            if self.len(hit.sub(r)) < 1e-4 {
                continue;
            }
            let Some(hp) = self.w2p(hit) else { continue };
            let d = hypot(hp.0 - cx, hp.1 - cy);
            if d > threshold_px {
                continue;
            }
            if self.occ(hit) {
                continue;
            }
            if best.is_none() || d < best.unwrap().0 {
                best = Some((d, hit, axis));
            }
        }
        let (_, hit, axis) = best?;
        let mut res = SnapResult::new(hit, SnapKind::FromPoint, COLOR_ENDPOINT);
        res.guide = Some((r, hit));
        res.guide_color = Some(axis.color());
        res.context = edge.context;
        Some(res)
    }

    /// `_in_plane_with_point_snap`.
    fn in_plane_with_point_snap(
        &self,
        r: V3,
        candidate: V3,
        plane_normal: V3,
        cpix: (f64, f64),
        threshold_px: f64,
    ) -> Option<SnapResult> {
        if self.len(plane_normal) < 1e-9 {
            return None;
        }
        let (cx, cy) = cpix;
        let n = self.norm(plane_normal);
        let mut best: Option<(f64, V3, Axis, V3)> = None;
        for axis in Axis::ALL {
            let a = self.ax(axis);
            let mut d = a.cross(n);
            if self.len(d) < 1e-6 {
                continue;
            }
            d = self.norm(d);
            let m = n.cross(d);
            let ma = self.dot(m, a);
            let s = self.dot(candidate.sub(r), a);
            let foot = candidate.sub(m.mul((s / ma) as f32));
            let Some(fp) = self.w2p(foot) else { continue };
            let dist = hypot(fp.0 - cx, fp.1 - cy);
            if dist > threshold_px {
                continue;
            }
            if self.occ(foot) {
                continue;
            }
            if best.is_none() || dist < best.unwrap().0 {
                best = Some((dist, foot, axis, d));
            }
        }
        let (_, foot, axis, d) = best?;
        let elbow = foot.add(d.mul(self.dot(r.sub(foot), d) as f32));
        let colour_along = |v: V3| -> Rgb {
            if self.len(v) < 1e-6 {
                return COLOR_EXTENSION;
            }
            self.most_aligned_axis(self.norm(v)).color()
        };
        let mut guides = Vec::new();
        if self.len(elbow.sub(r)) > 1e-4 {
            guides.push((r, elbow, colour_along(elbow.sub(r))));
        }
        let mut res = SnapResult::new(foot, SnapKind::Aligned, COLOR_ENDPOINT);
        res.axis = Some(axis);
        res.guide = Some((elbow, foot));
        res.guide_color = Some(colour_along(foot.sub(elbow)));
        res.guides = if guides.is_empty() { None } else { Some(guides) };
        Some(res)
    }

    /// `_extension_snap`.
    #[allow(clippy::too_many_arguments)]
    fn extension_snap(
        &self,
        candidate_world: V3,
        cpix: (f64, f64),
        edges: &[SnapEdge],
        et: f64,
        start_point: Option<V3>,
        plane_normal: Option<V3>,
    ) -> Option<SnapResult> {
        let start_point = start_point?;
        let (cx, cy) = cpix;
        let plane_normal = plane_normal.and_then(|n| if self.len(n) > 1e-9 { Some(self.norm(n)) } else { None });
        let draw = candidate_world.sub(start_point);
        if self.len(draw) < 1e-6 {
            return None;
        }
        let draw = self.norm(draw);
        let sp = if self.has_project { self.w2p(start_point) } else { None };
        let (mut sdx, mut sdy) = (0.0, 0.0);
        if let Some(sp) = sp {
            sdx = cx - sp.0;
            sdy = cy - sp.1;
        }
        let sdl = hypot(sdx, sdy);
        // (dist, proj, from_end, edge index, dir)
        let mut best_ext: Option<(f64, V3, V3, usize, V3)> = None;
        for (ei, edge) in edges.iter().enumerate() {
            if edge.soft || edge.hidden || edge.curve.is_some() {
                continue;
            }
            let ab = edge.b.sub(edge.a);
            if self.len(ab) < 1e-9 {
                continue;
            }
            let u = self.norm(ab);
            let mut at = candidate_world;
            if self.dot(draw, u).abs() < 0.966 {
                if sdl < 4.0 {
                    continue;
                }
                let off = start_point.sub(edge.a);
                if self.len(off.sub(u.mul(self.dot(off, u) as f32))) > 1e-3 {
                    continue;
                }
                let (Some(pa), Some(pb)) = (self.w2p(edge.a), self.w2p(edge.b)) else {
                    continue;
                };
                let (edx, edy) = (pb.0 - pa.0, pb.1 - pa.1);
                let edl = hypot(edx, edy);
                if edl < 20.0 || (sdx * edx + sdy * edy).abs() < 0.966 * sdl * edl {
                    continue;
                }
                at = self.proj(edge.a, u);
            }
            let mut t = self.dot(at.sub(edge.a), u);
            if let Some(pn) = plane_normal {
                let denom = self.dot(pn, u);
                if denom.abs() > 1e-6 {
                    t = self.dot(pn, candidate_world.sub(edge.a)) / denom;
                }
            }
            if -1e-6 <= t && t <= self.len(ab) + 1e-6 {
                continue;
            }
            let proj = edge.a.add(u.mul(t as f32));
            let Some(pp) = self.w2p(proj) else { continue };
            let d = hypot(pp.0 - cx, pp.1 - cy);
            if d > et {
                continue;
            }
            if self.occ(proj) {
                continue;
            }
            if best_ext.is_none() || d < best_ext.unwrap().0 {
                let from_end = if t < 0.0 { edge.a } else { edge.b };
                best_ext = Some((d, proj, from_end, ei, u));
            }
        }
        let (_, proj, from_end, src, u) = best_ext?;
        let mut best_hit: Option<(f64, V3)> = None;
        for (oi, other) in edges.iter().enumerate() {
            if oi == src {
                continue;
            }
            let Some(hit) = self.line_segment_intersection(proj, u, other.a, other.b) else { continue };
            let Some(hp) = self.w2p(hit) else { continue };
            let dh = hypot(hp.0 - cx, hp.1 - cy);
            if dh > et {
                continue;
            }
            if self.occ(hit) {
                continue;
            }
            if best_hit.is_none() || dh < best_hit.unwrap().0 {
                best_hit = Some((dh, hit));
            }
        }
        if let Some((_, hit)) = best_hit {
            if (self.len_sq(hit.sub(from_end))) < 1e-12 {
                return Some(SnapResult::new(hit, SnapKind::Endpoint, COLOR_ENDPOINT));
            }
            let mut r = SnapResult::new(hit, SnapKind::Intersection, COLOR_ENDPOINT);
            r.guide = Some((from_end, hit));
            return Some(r);
        }
        let mut r = SnapResult::new(proj, SnapKind::Extension, COLOR_EXTENSION);
        r.guide = Some((from_end, proj));
        Some(r)
    }

    /// `v.lengthSquared()` as a Python float.
    #[inline]
    fn len_sq(&self, v: V3) -> f64 {
        v.length_squared() as f64
    }

    /// `_from_point_snap`.
    #[allow(clippy::too_many_arguments)]
    fn snap_from_point(
        &self,
        edges: &[SnapEdge],
        start_point: Option<V3>,
        draw_dir: V3,
        cpix: (f64, f64),
        threshold_px: f64,
        extra_point: Option<V3>,
        axis_deg: f64,
        hovered_refs: bool,
        line_dir: Option<V3>,
    ) -> Option<SnapResult> {
        let start_point = start_point?;
        if self.len(draw_dir) < 1e-6 {
            return None;
        }
        let (cx, cy) = cpix;
        let adir: V3;
        if let Some(ld) = line_dir {
            if self.len(ld) < 1e-6 {
                return None;
            }
            adir = self.norm(ld);
        } else {
            let u = self.norm(draw_dir);
            let mut found = None;
            let cos_axis = radians(axis_deg).cos();
            for k in Axis::ALL {
                let a = self.ax(k);
                let dp = self.dot(u, a);
                if dp.abs() >= cos_axis {
                    found = Some(if dp > 0.0 { a } else { a.neg() });
                    break;
                }
            }
            adir = found?;
        }

        // (point, kind, colour, is the edge body)
        let mut refs: Vec<(V3, SnapKind, Rgb, bool)> = Vec::new();
        if let Some(ep) = extra_point {
            refs.push((ep, SnapKind::FromPoint, COLOR_ENDPOINT, false));
        }
        for edge in edges {
            if edge.figure {
                continue;
            }
            refs.push((edge.a, SnapKind::FromPoint, COLOR_ENDPOINT, false));
            refs.push((edge.b, SnapKind::FromPoint, COLOR_ENDPOINT, false));
            refs.push((edge.a.add(edge.b).mul(0.5), SnapKind::Midpoint, COLOR_MIDPOINT, false));
        }
        refs.push((V3(0.0, 0.0, 0.0), SnapKind::Origin, COLOR_ORIGIN, false));

        if hovered_refs {
            let mut best_edge: Option<(f64, V3)> = None;
            for edge in edges {
                let (Some(pa), Some(pb)) = (self.w2p(edge.a), self.w2p(edge.b)) else {
                    continue;
                };
                let (d, t) = closest_on_segment_2d((cx, cy), pa, pb);
                if d <= threshold_px && (best_edge.is_none() || d < best_edge.unwrap().0) {
                    best_edge = Some((d, self.on_edge_point(edge.a, edge.b, t)));
                }
            }
            if let Some((_, p)) = best_edge {
                refs.push((p, SnapKind::FromPoint, COLOR_ENDPOINT, true));
            }
        }

        let mut best: Option<(f64, V3, V3, SnapKind, Rgb)> = None;
        for &(r, kind, color, body) in &refs {
            if body && best.is_some() {
                continue;
            }
            let s = self.dot(r.sub(start_point), adir);
            if !hovered_refs && s <= 1e-6 {
                continue;
            }
            let foot = start_point.add(adir.mul(s as f32));
            if self.len(foot.sub(r)) < 1e-4 {
                continue;
            }
            let Some(qp) = self.w2p(foot) else { continue };
            let mut d = hypot(qp.0 - cx, qp.1 - cy);
            if hovered_refs {
                if let Some(rp) = self.w2p(r) {
                    d = py_min2(d, hypot(rp.0 - cx, rp.1 - cy));
                }
            }
            if d > threshold_px {
                continue;
            }
            if self.occ(foot) {
                continue;
            }
            if best.is_none() || d < best.unwrap().0 {
                best = Some((d, foot, r, kind, color));
            }
        }
        let (_, foot, r, kind, color) = best?;
        let mut guide_color = None;
        if kind == SnapKind::FromPoint {
            let gdn = self.norm(foot.sub(r));
            guide_color = Some(self.most_aligned_axis(gdn).color());
        }
        let mut res = SnapResult::new(foot, kind, color);
        res.guide = Some((r, foot));
        res.guide_color = guide_color;
        Some(res)
    }

    /// `_through_point_snap`.
    fn through_point_snap(&self, start_point: Option<V3>, through_point: Option<V3>, draw_dir: V3, inference_deg: f64) -> Option<SnapResult> {
        let (start_point, through_point) = (start_point?, through_point?);
        if !self.has_project || self.len(draw_dir) < 1e-6 {
            return None;
        }
        let dirv = through_point.sub(start_point);
        if self.len(dirv) < 1e-6 {
            return None;
        }
        let dir_u = self.norm(dirv);
        if self.dot(self.norm(draw_dir), dir_u).abs() < radians(inference_deg).cos() {
            return None;
        }
        let locked = self.proj(start_point, dir_u);
        let mut r = SnapResult::new(locked, SnapKind::ThroughPoint, COLOR_REFERENCE);
        r.guide = Some((through_point, locked));
        Some(r)
    }

    /// `_perpendicular_face_snap`.
    fn perpendicular_face_snap(&self, start_point: Option<V3>, face_normal: Option<V3>, draw_dir: V3, inference_deg: f64) -> Option<SnapResult> {
        let (start_point, face_normal) = (start_point?, face_normal?);
        if !self.has_project || self.len(draw_dir) < 1e-6 || self.len(face_normal) < 1e-6 {
            return None;
        }
        let n_u = self.norm(face_normal);
        if self.dot(self.norm(draw_dir), n_u).abs() < radians(inference_deg).cos() {
            return None;
        }
        let locked = self.proj(start_point, n_u);
        Some(SnapResult::new(locked, SnapKind::PerpFace, COLOR_REFERENCE))
    }

    /// `_intersection_snap`.
    fn intersection_snap(&self, cpix: (f64, f64), edges: &[SnapEdge], et: f64) -> Option<SnapResult> {
        let (cx, cy) = cpix;
        let mut near: Vec<&SnapEdge> = Vec::new();
        for edge in edges {
            if edge.center {
                continue;
            }
            let ab = edge.b.sub(edge.a);
            if self.dot(ab, ab) < 1e-12 {
                continue;
            }
            let (Some(pa), Some(pb)) = (self.w2p(edge.a), self.w2p(edge.b)) else {
                continue;
            };
            let (d, _t) = closest_on_segment_2d((cx, cy), pa, pb);
            if d <= et {
                near.push(edge);
            }
        }
        if near.len() < 2 {
            return None;
        }
        let mut best: Option<(f64, V3, usize, usize)> = None;
        for i in 0..near.len() {
            let e1 = near[i];
            for j in i + 1..near.len() {
                let e2 = near[j];
                let Some(hit) = segment_intersection(e1.a, e1.b, e2.a, e2.b, SPLIT_TOLERANCE, self.cx.h) else {
                    continue;
                };
                let Some(hp) = self.w2p(hit) else { continue };
                let dh = hypot(hp.0 - cx, hp.1 - cy);
                if dh > et {
                    continue;
                }
                if self.occ(hit) {
                    continue;
                }
                if best.is_none() || dh < best.unwrap().0 {
                    best = Some((dh, hit, i, j));
                }
            }
        }
        let (_, hit, i, j) = best?;
        let corners = [near[i].a, near[i].b, near[j].a, near[j].b];
        if corners.iter().any(|&c| self.len_sq(hit.sub(c)) < 1e-12) {
            return Some(SnapResult::new(hit, SnapKind::Endpoint, COLOR_ENDPOINT));
        }
        Some(SnapResult::new(hit, SnapKind::Intersection, COLOR_ENDPOINT))
    }

    /// `_lock_line_snaps`.
    #[allow(clippy::too_many_arguments)]
    fn lock_line_snaps(
        &self,
        edges: &[SnapEdge],
        start_point: V3,
        line_dir: V3,
        cpix: (f64, f64),
        threshold_px: f64,
        acquired_point: Option<V3>,
        chain_first_point: Option<V3>,
    ) -> Option<SnapResult> {
        let (cx, cy) = cpix;
        if let Some(cf) = chain_first_point {
            if self.vertex_on_line(cf, start_point, line_dir) {
                if let Some(fp) = self.w2p(cf) {
                    if hypot(fp.0 - cx, fp.1 - cy) <= threshold_px {
                        return Some(SnapResult::new(cf, SnapKind::Close, COLOR_CLOSE));
                    }
                }
            }
        }
        for edge in edges {
            for vertex in [edge.a, edge.b] {
                if !self.vertex_on_line(vertex, start_point, line_dir) {
                    continue;
                }
                let Some(vp) = self.w2p(vertex) else { continue };
                if hypot(vp.0 - cx, vp.1 - cy) <= threshold_px {
                    return Some(SnapResult::new(vertex, SnapKind::Endpoint, COLOR_ENDPOINT));
                }
            }
        }
        let mut best_hit: Option<(f64, V3)> = None;
        for edge in edges {
            let Some(hit) = self.line_segment_intersection(start_point, line_dir, edge.a, edge.b) else {
                continue;
            };
            let Some(hp) = self.w2p(hit) else { continue };
            let dh = hypot(hp.0 - cx, hp.1 - cy);
            if dh > threshold_px {
                continue;
            }
            if self.occ(hit) {
                continue;
            }
            if best_hit.is_none() || dh < best_hit.unwrap().0 {
                best_hit = Some((dh, hit));
            }
        }
        if let Some((_, hit)) = best_hit {
            let mut r = SnapResult::new(hit, SnapKind::Intersection, COLOR_ENDPOINT);
            r.guide = Some((start_point, hit));
            return Some(r);
        }
        self.snap_from_point(
            edges,
            Some(start_point),
            line_dir,
            cpix,
            threshold_px,
            acquired_point,
            10.0,
            true,
            Some(line_dir),
        )
    }

    /// `_first_visible`: the index of the first point not occluded.
    fn first_visible(&self, points: &[V3]) -> Option<usize> {
        if points.is_empty() {
            return None;
        }
        if !self.has_occ {
            return Some(0);
        }
        points.iter().position(|&p| !self.env.is_occluded(p))
    }
}

/// `_closest_on_segment_2d`: `(distance, t)`.
pub fn closest_on_segment_2d(p: (f64, f64), a: (f64, f64), b: (f64, f64)) -> (f64, f64) {
    let (ax, ay) = a;
    let (bx, by) = b;
    let (px, py) = p;
    let dx = bx - ax;
    let dy = by - ay;
    let denom = dx * dx + dy * dy;
    if denom == 0.0 {
        return (hypot(px - ax, py - ay), 0.0);
    }
    let mut t = ((px - ax) * dx + (py - ay) * dy) / denom;
    t = py_max2(0.0, py_min2(1.0, t));
    let qx = ax + t * dx;
    let qy = ay + t * dy;
    (hypot(px - qx, py - qy), t)
}

/// A point candidate waiting to be resolved: `(d, world, kind, colour,
/// context, occludable)`.
type Pending = (f64, V3, SnapKind, Rgb, Option<Context>, bool);

struct Resolver<'e, 'a> {
    e: &'e E<'a>,
    cx: f64,
    cy: f64,
    threshold_px: f64,
    pending: Vec<Pending>,
}

impl Resolver<'_, '_> {
    /// `_consider`.
    fn consider(&mut self, world: V3, kind: SnapKind, color: Rgb, occludable: bool, context: Option<Context>) {
        let Some(px) = self.e.w2p(world) else { return };
        let d = hypot(px.0 - self.cx, px.1 - self.cy);
        if d > self.threshold_px {
            return;
        }
        self.pending.push((d, world, kind, color, context, occludable));
    }

    /// `_resolve`: the nearest visible candidate; near-ties go to the
    /// context being drawn in (a stable sort on the tie bucket and the
    /// context flag). The batched occlusion asks of Python change how many
    /// rays are cast, not which one is chosen.
    fn resolve(&mut self) -> Option<(f64, V3, SnapKind, Rgb, Option<Context>)> {
        self.pending.sort_by(|a, b| {
            let ka = ((a.0 / TIE_PX).floor(), u8::from(a.4.is_some()));
            let kb = ((b.0 / TIE_PX).floor(), u8::from(b.4.is_some()));
            ka.partial_cmp(&kb).unwrap_or(std::cmp::Ordering::Equal)
        });
        let mut chosen = None;
        for &(d, world, kind, color, context, occludable) in &self.pending {
            if occludable && self.e.has_occ && self.e.env.is_occluded(world) {
                continue;
            }
            chosen = Some((d, world, kind, color, context));
            break;
        }
        self.pending.clear();
        chosen
    }
}

/// `compute_snap`.
pub fn compute_snap(env: &dyn SnapEnv, cx: Cx, p: &SnapParams) -> SnapResult {
    let e = E { cx, env, has_project: p.has_project, has_occ: p.has_occlusion };
    let edges = p.edges;
    let candidate_world = p.candidate_world;
    let start_point = p.start_point;
    let allow_axis = p.linear_mode == LinearMode::All;
    let allow_parperp = p.linear_mode != LinearMode::Off;
    let line_inferences = allow_axis && !p.radial_arm;
    let on_plane = if p.face_under_cursor { p.work_plane_normal } else { None };

    let mut acquired_point = p.acquired_point;
    if let (Some(ap), Some(sp)) = (acquired_point, start_point) {
        if e.len_sq(ap.sub(sp)) < 1e-12 {
            acquired_point = None;
        }
    }

    let mut axis_engaged: Option<Axis> = None;
    if allow_axis && p.has_project {
        axis_engaged = e.engaged_axis(start_point, candidate_world, p.candidate_pixel, p.magnetic_axis_deg, p.screen_axis_px);
        if p.axis_in_plane {
            axis_engaged = e.axis_in_plane(axis_engaged, p.work_plane_normal);
        }
    }

    let (cx_, cy_) = p.candidate_pixel;

    // 1. Explicit axis lock (arrow keys).
    if let (Some(lock), Some(sp), true) = (p.axis_lock, start_point, p.has_project) {
        let mut axis_dir = e.ax(lock);
        let locked = e.proj(sp, axis_dir);
        if e.dot(locked.sub(sp), axis_dir) < 0.0 {
            axis_dir = axis_dir.neg();
        }
        if let Some(hit) = e.lock_line_snaps(edges, sp, axis_dir, p.candidate_pixel, p.threshold_px, acquired_point, p.chain_first_point) {
            return hit;
        }
        let mut r = SnapResult::new(locked, SnapKind::Axis, lock.color());
        r.axis = Some(lock);
        return r;
    }

    // 1.5 Sticky inference lock (Shift captured an active inference).
    if let (Some(sld), Some(sp), true) = (p.shift_lock_dir, start_point, p.has_project) {
        let mut lock_dir = sld;
        let locked = e.proj(sp, lock_dir);
        if e.dot(locked.sub(sp), lock_dir) < 0.0 {
            lock_dir = lock_dir.neg();
        }
        if let Some(hit) = e.lock_line_snaps(edges, sp, lock_dir, p.candidate_pixel, p.threshold_px, acquired_point, p.chain_first_point) {
            return hit;
        }
        let color = p.shift_lock_color.unwrap_or(COLOR_REFERENCE);
        return SnapResult::new(locked, SnapKind::Reference, color);
    }

    // 2. Reference edge lock (Down arrow + edge under cursor).
    if let (Some(re), Some(mode), Some(sp), true) = (p.reference_edge, p.reference_mode, start_point, p.has_project) {
        if let Some(mut direction) = e.direction_from_edge(re, mode, p.work_plane_normal) {
            let locked = e.proj(sp, direction);
            if e.dot(locked.sub(sp), direction) < 0.0 {
                direction = direction.neg();
            }
            if let Some(hit) = e.lock_line_snaps(edges, sp, direction, p.candidate_pixel, p.threshold_px, acquired_point, p.chain_first_point) {
                return hit;
            }
            return SnapResult::new(locked, SnapKind::Reference, COLOR_REFERENCE);
        }
    }

    // 3. Shift held + auto axis inference -> lock to that axis.
    if let (true, true, Some(sp), true) = (allow_axis, p.shift_held, start_point, p.has_project) {
        if let Some(inferred) = e.detect_axis_alignment(sp, candidate_world, p.inference_angle_deg) {
            let mut axis_dir = e.ax(inferred);
            let locked = e.proj(sp, axis_dir);
            if e.dot(locked.sub(sp), axis_dir) < 0.0 {
                axis_dir = axis_dir.neg();
            }
            if let Some(hit) = e.lock_line_snaps(edges, sp, axis_dir, p.candidate_pixel, p.threshold_px, acquired_point, p.chain_first_point) {
                return hit;
            }
            let mut r = SnapResult::new(locked, SnapKind::Axis, inferred.color());
            r.axis = Some(inferred);
            return r;
        }
    }

    let et = p.edge_threshold_px.unwrap_or(p.threshold_px);
    let mut rs = Resolver { e: &e, cx: cx_, cy: cy_, threshold_px: p.threshold_px, pending: Vec::new() };
    let mut best: Option<(f64, V3, SnapKind, Rgb, Option<Context>)> = None;

    // 4. Vertex snaps (close, endpoint).
    if let (Some(cf), Some(_), false) = (p.chain_first_point, start_point, p.chain_first_is_start) {
        rs.consider(cf, SnapKind::Close, COLOR_CLOSE, false, None);
        best = rs.resolve();
    }
    if best.is_none() || best.unwrap().2 != SnapKind::Close {
        let mut plain: Vec<&SnapEdge> = Vec::new();
        for edge in edges {
            if edge.center {
                rs.consider(edge.a, SnapKind::Center, COLOR_ENDPOINT, true, None);
            } else if edge.component_origin {
                rs.consider(edge.a, SnapKind::ComponentOrigin, COLOR_ORIGIN, true, None);
            } else if edge.arc_midpoint {
                rs.consider(edge.a, SnapKind::ArcMidpoint, COLOR_MIDPOINT, true, None);
            } else if edge.guide {
                continue;
            } else {
                plain.push(edge);
            }
        }
        rs.consider(V3(0.0, 0.0, 0.0), SnapKind::Origin, COLOR_ORIGIN, true, None);
        for edge in &plain {
            rs.consider(edge.a, SnapKind::Endpoint, COLOR_ENDPOINT, true, edge.context);
            rs.consider(edge.b, SnapKind::Endpoint, COLOR_ENDPOINT, true, edge.context);
        }
        if best.is_none() {
            best = rs.resolve();
        }
    }
    if let Some(b) = best {
        let mut r = SnapResult::new(b.1, b.2, b.3);
        r.context = b.4;
        return r;
    }

    // 4b. Perpendicular to a wall you started on.
    if let (true, None, Some(sp), true) = (allow_parperp, axis_engaged, start_point, p.has_project) {
        let draw = candidate_world.sub(sp);
        if e.len(draw) > 1e-6 {
            let draw_u = e.norm(draw);
            let cos_tol = radians(8.0).cos();
            for edge in edges {
                if !e.point_on_segment_world(sp, edge.a, edge.b) {
                    continue;
                }
                let Some(perp) = e.direction_from_edge(EdgeRef { a: edge.a, b: edge.b }, RefMode::Perpendicular, p.work_plane_normal) else {
                    continue;
                };
                if e.dot(draw_u, perp).abs() < cos_tol {
                    continue;
                }
                let mut best_hit: Option<(f64, V3)> = None;
                for other in edges {
                    if e.point_on_segment_world(sp, other.a, other.b) {
                        continue;
                    }
                    let Some(hit) = e.line_segment_intersection(sp, perp, other.a, other.b) else { continue };
                    if e.dot(hit.sub(sp), draw_u) <= 0.0 {
                        continue;
                    }
                    let Some(hp) = e.w2p(hit) else { continue };
                    let dh = hypot(hp.0 - cx_, hp.1 - cy_);
                    if dh > et {
                        continue;
                    }
                    if e.occ(hit) {
                        continue;
                    }
                    if best_hit.is_none() || dh < best_hit.unwrap().0 {
                        best_hit = Some((dh, hit));
                    }
                }
                if let Some((_, hit)) = best_hit {
                    let mut r = SnapResult::new(hit, SnapKind::Intersection, COLOR_ENDPOINT);
                    r.guide = Some((sp, hit));
                    return r;
                }
                if line_inferences && axis_engaged.is_none() {
                    if let Some(fp) = e.snap_from_point(
                        edges,
                        Some(sp),
                        candidate_world.sub(sp),
                        p.candidate_pixel,
                        p.threshold_px,
                        acquired_point,
                        10.0,
                        false,
                        None,
                    ) {
                        return fp;
                    }
                }
                let locked = e.proj(sp, perp);
                return SnapResult::new(locked, SnapKind::Reference, COLOR_REFERENCE);
            }
        }
    }

    // 4e. 'Through point'.
    if let (true, Some(sp)) = (line_inferences, start_point) {
        let mut sobre_el_punto = false;
        if let Some(ap) = acquired_point {
            if let Some(apx) = e.w2p(ap) {
                sobre_el_punto = hypot(apx.0 - cx_, apx.1 - cy_) <= p.threshold_px;
            }
        }
        if !sobre_el_punto {
            if let Some(tp) = e.through_point_snap(Some(sp), acquired_point, candidate_world.sub(sp), p.inference_angle_deg) {
                return tp;
            }
        }
    }

    // 5. Extension.
    if line_inferences && axis_engaged.is_none() {
        if let Some(ext) = e.extension_snap(candidate_world, p.candidate_pixel, edges, et, start_point, on_plane) {
            return ext;
        }
    }

    // 5b. 'From point'.
    if let (true, Some(sp), None) = (line_inferences, start_point, axis_engaged) {
        if let Some(fp) = e.snap_from_point(
            edges,
            Some(sp),
            candidate_world.sub(sp),
            p.candidate_pixel,
            p.threshold_px,
            acquired_point,
            10.0,
            false,
            None,
        ) {
            return fp;
        }
    }

    // 5c. Edge / guide intersection.
    if axis_engaged.is_none() {
        if let Some(inter) = e.intersection_snap(p.candidate_pixel, edges, et) {
            return inter;
        }
    }

    // 6. Midpoint + origin.
    for edge in edges {
        if edge.guide || e.len(edge.a.sub(edge.b)) < 1e-9 {
            continue;
        }
        rs.consider(edge.a.add(edge.b).mul(0.5), SnapKind::Midpoint, COLOR_MIDPOINT, true, edge.context);
    }
    rs.consider(V3(0.0, 0.0, 0.0), SnapKind::Origin, COLOR_ORIGIN, true, None);
    if let Some(b) = rs.resolve() {
        let mut r = SnapResult::new(b.1, b.2, b.3);
        r.context = b.4;
        return r;
    }

    // 7. On-edge.
    let mut on_edges: Vec<(f64, V3, usize)> = Vec::new();
    for (ei, edge) in edges.iter().enumerate() {
        let (Some(pa), Some(pb)) = (e.w2p(edge.a), e.w2p(edge.b)) else {
            continue;
        };
        let (d, t) = closest_on_segment_2d((cx_, cy_), pa, pb);
        if d > et {
            continue;
        }
        let ab = edge.b.sub(edge.a);
        let on_pt = if p.has_project && e.len(ab) > 1e-9 {
            let proj = e.proj(edge.a, ab);
            let tt = e.dot(proj.sub(edge.a), ab) / e.dot(ab, ab);
            edge.a.add(ab.mul(py_max2(0.0, py_min2(1.0, tt)) as f32))
        } else {
            edge.a.add(ab.mul(t as f32))
        };
        on_edges.push((d, on_pt, ei));
    }
    on_edges.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    let pts: Vec<V3> = on_edges.iter().map(|c| c.1).collect();
    let best_edge = e.first_visible(&pts).map(|k| on_edges[k]);
    // 7a. With an axis engaged, the crossing of the axis line with the edge.
    if let (Some((_d, _on_pt, ei)), Some(ax), Some(sp)) = (best_edge, axis_engaged, start_point) {
        let edge = &edges[ei];
        if let Some(hit) = e.line_segment_intersection(sp, e.ax(ax), edge.a, edge.b) {
            if let Some(hp) = e.w2p(hit) {
                if hypot(hp.0 - cx_, hp.1 - cy_) <= et && !e.occ(hit) {
                    let mut r = SnapResult::new(hit, SnapKind::Intersection, COLOR_ENDPOINT);
                    r.guide = Some((sp, hit));
                    return r;
                }
            }
        }
    }
    if let (Some((_d, on_pt, ei)), None) = (best_edge, axis_engaged) {
        let edge = &edges[ei];
        if line_inferences {
            if let Some(ap) = acquired_point {
                if let Some(cross) = e.edge_from_point_crossing(edge, ap, p.candidate_pixel, p.threshold_px) {
                    return cross;
                }
            }
        }
        if edge.guide {
            return SnapResult::new(on_pt, SnapKind::OnLine, COLOR_ON_EDGE);
        }
        let mut r = SnapResult::new(on_pt, SnapKind::OnEdge, COLOR_ON_EDGE);
        r.context = edge.context;
        return r;
    }

    // 8b. Acquired-edge parallel inference.
    if let (true, None, Some(ae), Some(sp), true) = (allow_parperp, axis_engaged, p.acquired_edge, start_point, p.has_project) {
        let ev = ae.b.sub(ae.a);
        let draw = candidate_world.sub(sp);
        if e.len(ev) > 1e-9 && e.len(draw) > 1e-6 && e.detect_axis_alignment(ae.a, ae.b, p.inference_angle_deg).is_none() {
            let ev_u = e.norm(ev);
            let par_cos = radians(p.inference_angle_deg).cos();
            if e.dot(e.norm(draw), ev_u).abs() >= par_cos {
                let locked = e.proj(sp, ev_u);
                return SnapResult::new(locked, SnapKind::Reference, COLOR_REFERENCE);
            }
        }
    }

    // 8c. Perpendicular to an encouraged face.
    if let (true, None, Some(sp), true) = (allow_parperp, axis_engaged, start_point, p.has_project) {
        if let Some(pf) = e.perpendicular_face_snap(Some(sp), p.acquired_face_normal, candidate_world.sub(sp), p.inference_angle_deg) {
            return pf;
        }
    }

    // 8d. 'From point' from an encouraged point.
    if line_inferences && axis_engaged.is_none() {
        if !p.acquired_points.is_empty() {
            if let Some(tp) = e.two_point_snap(p.acquired_points, p.candidate_pixel, p.threshold_px) {
                return tp;
            }
        }
        if let Some(ap) = acquired_point {
            if let Some(fp) = e.first_point_from_point(ap, candidate_world, p.candidate_pixel, p.threshold_px, on_plane) {
                return fp;
            }
            // 8e. Level with / in line with it across this face.
            if let Some(pn) = on_plane {
                if let Some(ip) = e.in_plane_with_point_snap(ap, candidate_world, pn, p.candidate_pixel, p.threshold_px) {
                    return ip;
                }
            }
        }
    }

    // 9. The axis that took command, then the soft cue.
    if let (Some(ax), Some(sp)) = (axis_engaged, start_point) {
        let mut r = SnapResult::new(e.proj(sp, e.ax(ax)), SnapKind::Axis, ax.color());
        r.axis = Some(ax);
        return r;
    }
    if let (Some(sp), true) = (start_point, allow_axis) {
        let mut inferred = e.detect_axis_alignment(sp, candidate_world, p.inference_angle_deg);
        if p.axis_in_plane {
            inferred = e.axis_in_plane(inferred, p.work_plane_normal);
        }
        if let Some(ax) = inferred {
            let mut r = SnapResult::new(candidate_world, SnapKind::AxisInference, ax.color());
            r.axis = Some(ax);
            return r;
        }
    }

    // 10. On-face.
    if p.face_under_cursor {
        return SnapResult::new(candidate_world, SnapKind::OnFace, COLOR_ON_FACE);
    }
    SnapResult::new(candidate_world, SnapKind::None, COLOR_NONE)
}
