// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! The drawing plane and the cursor's point on it (`Viewport.
//! _world_from_pixel`, `_current_work_plane`, `_planar_face_plane`,
//! `_hover_face_plane`, `_camera_planar_work_plane`, `_start_point_plane`,
//! `_near_horizon_vertical`) and the first-click fallback
//! (`core.snap.first_point_work_plane`).
//!
//! The tool's side of the questions (`tool.work_plane`, `plane_lock`,
//! `drag_plane`, `start_point`, whether it is a `PlaneLock`…) arrives in
//! [`ToolState`]; the viewport's (the arrow lock, the camera, the faces) in
//! the arguments.

use ingetrazo_math::V3;

use crate::compute::radians;
use crate::types::{Axes, Axis, Cx};
use crate::world::World;
use crate::View;

/// `HORIZON_PITCH_THRESHOLD_DEG`.
const HORIZON_PITCH_THRESHOLD_DEG: f64 = 15.0;
const GROUND_PREFERENCE_ANGLE_DEG: f64 = 5.0;
const GROUND_DEPTH_PREFERENCE_RATIO: f64 = 1.25;
const PERSPECTIVE_GROUND_DEPTH_PREFERENCE_RATIO: f64 = 10.0;
const PLANE_ALIGNMENT_TIE_MARGIN: f64 = 0.05;

/// What the active tool tells the engine (attributes the viewport reads
/// with `getattr`).
#[derive(Clone, Debug, PartialEq)]
pub struct ToolState {
    pub uses_snap: bool,
    /// `isinstance(tool, PlaneLock)`: a planar shape.
    pub is_plane_lock: bool,
    pub prefers_vertical_drag: bool,
    pub radial_arm: bool,
    pub magnetic_axis_deg: Option<f64>,
    pub screen_axis_px: Option<f64>,
    /// `AxisMagnet.magnet_off_plane()` (true for tools without a magnet).
    pub magnet_off_plane: bool,
    pub axis_source: bool,
    pub axis_source_empty_only: bool,
    /// `hover_group_edges`: the tool infers to a group's edges even when it
    /// does not snap (Push/Pull).
    pub hover_group_edges: bool,
    pub start_point: Option<V3>,
    pub chain_first_point: Option<V3>,
    /// `chain_first_point is start_point`.
    pub chain_first_is_start: bool,
    /// `tool.work_plane`: `(point, normal)` captured at the first click.
    pub work_plane: Option<(V3, V3)>,
    /// `tool.plane_lock`: an arrow-key plane.
    pub plane_lock: Option<Axis>,
    /// `tool.plane_ref[0]`: the Down-arrow reference plane's normal.
    pub plane_ref: Option<V3>,
    /// `tool.drag_plane(viewport)`.
    pub own_plane: Option<(V3, V3)>,
}

impl Default for ToolState {
    fn default() -> Self {
        ToolState {
            uses_snap: true,
            is_plane_lock: false,
            prefers_vertical_drag: false,
            radial_arm: false,
            magnetic_axis_deg: None,
            screen_axis_px: None,
            magnet_off_plane: true,
            axis_source: false,
            axis_source_empty_only: false,
            hover_group_edges: false,
            start_point: None,
            chain_first_point: None,
            chain_first_is_start: false,
            work_plane: None,
            plane_lock: None,
            plane_ref: None,
            own_plane: None,
        }
    }
}

/// `core.snap.first_point_work_plane(forward, scene_center, threshold)`.
pub fn first_point_work_plane(cx: &Cx, forward: V3, scene_center: V3) -> Option<(V3, V3)> {
    let threshold = 0.97;
    if cx.len(forward) < 1e-9 {
        return None;
    }
    let f = cx.norm(forward);
    if !cx.axes.world {
        let lf = cx.axes.to_local(f);
        // max over (abs, name) tuples: a tie goes to the greater name.
        let mut comps = [(lf.0.abs() as f64, Axis::X), (lf.1.abs() as f64, Axis::Y), (lf.2.abs() as f64, Axis::Z)];
        comps.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        let (best, name) = comps[0];
        if best < threshold {
            return None;
        }
        let a = cx.axes.axis(name);
        let sign = if f.dot(a) as f64 > 0.0 { 1.0 } else { -1.0 };
        return Some((scene_center, a.mul(sign)));
    }
    let (ax, ay, az) = (f.0.abs() as f64, f.1.abs() as f64, f.2.abs() as f64);
    let m = if ax > ay { if ax > az { ax } else { az } } else if ay > az { ay } else { az };
    if m < threshold {
        return None;
    }
    let normal = if ax >= ay && ax >= az {
        V3(if (f.0 as f64) > 0.0 { 1.0 } else { -1.0 }, 0.0, 0.0)
    } else if ay >= az {
        V3(0.0, if (f.1 as f64) > 0.0 { 1.0 } else { -1.0 }, 0.0)
    } else {
        V3(0.0, 0.0, if (f.2 as f64) > 0.0 { 1.0 } else { -1.0 })
    };
    Some((scene_center, normal))
}

/// The pieces `world_from_pixel` reads.
pub struct PlaneCx<'a> {
    pub view: &'a View,
    pub world: &'a World,
    pub axes: &'a Axes,
    pub tool: &'a ToolState,
    /// `viewport.axis_lock is not None`.
    pub axis_locked: bool,
    pub inference_angle_deg: f64,
}

impl PlaneCx<'_> {
    fn cx(&self) -> Cx {
        Cx { h: self.view.h, axes: *self.axes }
    }

    fn forward(&self) -> V3 {
        self.view.cam.target.sub(self.view.cam.eye())
    }

    /// `_near_horizon_vertical(start)`.
    fn near_horizon_vertical(&self, start: V3) -> Option<(V3, V3)> {
        let forward = self.forward();
        if self.cx().len(forward) < 1e-9 {
            return None;
        }
        let forward = self.cx().norm(forward);
        if (forward.2.abs() as f64) >= radians(HORIZON_PITCH_THRESHOLD_DEG).sin() {
            return None;
        }
        if (forward.0.abs() as f64) >= (forward.1.abs() as f64) {
            Some((start, V3(1.0, 0.0, 0.0)))
        } else {
            Some((start, V3(0.0, 1.0, 0.0)))
        }
    }

    /// `_start_point_plane(tool, start)`.
    fn start_point_plane(&self, start: V3) -> (V3, V3) {
        let cx = self.cx();
        let forward = self.forward();
        if cx.len(forward) < 1e-9 {
            return (start, V3(0.0, 0.0, 1.0));
        }
        let forward = cx.norm(forward);
        if self.tool.prefers_vertical_drag {
            let horiz = V3(forward.0, forward.1, 0.0);
            if cx.len(horiz) >= radians(HORIZON_PITCH_THRESHOLD_DEG).sin() {
                return (start, cx.norm(horiz));
            }
            return (start, V3(0.0, 0.0, 1.0));
        }
        if let Some(v) = self.near_horizon_vertical(start) {
            return v;
        }
        (start, V3(0.0, 0.0, 1.0))
    }

    /// `_ray_hit_plane(cursor, plane)`.
    fn ray_hit_plane(&self, cursor: (f64, f64), plane: (V3, V3)) -> Option<V3> {
        let (origin, direction) = self.view.pixel_to_ray(cursor.0, cursor.1)?;
        let (point, normal) = plane;
        let denom = normal.dot(direction) as f64;
        if denom.abs() < 1e-6 {
            return None;
        }
        let t = normal.dot(point.sub(origin)) as f64 / denom;
        if t < 0.0 {
            return None;
        }
        Some(origin.add(direction.mul(t as f32)))
    }

    /// `_hover_face_plane(cursor)` (no selection, no face being bent).
    fn hover_face_plane(&self, cursor: (f64, f64)) -> Option<(V3, V3)> {
        let e = self.world.face_under(self.view, cursor.0, cursor.1)?;
        Some(self.world.ents[e].plane_place)
    }

    /// `_same_plane(a, b)`.
    fn same_plane(&self, a: (V3, V3), b: (V3, V3)) -> bool {
        let cx = self.cx();
        let na = cx.norm(a.1);
        let nb = cx.norm(b.1);
        if cx.dot(na, nb).abs() < 1e-3 {
            return false;
        }
        cx.dot(b.0.sub(a.0), na).abs() < 1e-6
    }

    /// `_planar_face_plane(tool, cursor, captured)`.
    fn planar_face_plane(&self, cursor: (f64, f64), captured: Option<(V3, V3)>) -> Option<(V3, V3)> {
        if self.axis_locked {
            return None;
        }
        if self.tool.plane_lock.is_some() || self.tool.plane_ref.is_some() {
            return None;
        }
        if !self.tool.is_plane_lock {
            return None;
        }
        let face_plane = self.hover_face_plane(cursor)?;
        if let Some(c) = captured {
            if self.same_plane(c, face_plane) {
                return None;
            }
        }
        Some(face_plane)
    }

    /// `_camera_planar_work_plane(cursor)`.
    fn camera_planar_work_plane(&self, cursor: (f64, f64)) -> Option<(V3, V3)> {
        let cx = self.cx();
        let origin = self.axes.origin;
        let view = self.forward();
        if view.length_squared() as f64 <= 1e-18 {
            // `view.lengthSquared() < 1e-18`
        }
        if (view.length_squared() as f64) < 1e-18 {
            return None;
        }
        let view = {
            let mut v = view;
            v = v.normalize(cx.h);
            v
        };
        let mut ray_origin = self.view.cam.eye();
        let mut ray_direction = view;
        let mut cursor_origin: Option<V3> = None;
        let mut cursor_direction: Option<V3> = None;
        if let Some((o, d)) = self.view.pixel_to_ray(cursor.0, cursor.1) {
            cursor_origin = Some(o);
            cursor_direction = Some(d);
            ray_origin = o;
            ray_direction = cx.norm(d);
        }
        let names = [Axis::X, Axis::Y, Axis::Z];
        let normals: Vec<V3> = names.iter().map(|&n| self.axes.axis(n)).collect();
        // hits: name -> depth
        let mut hits: [Option<f64>; 3] = [None; 3];
        for (i, &normal) in normals.iter().enumerate() {
            let Some(hit) = self.view.ray_plane(ray_origin, ray_direction, origin, normal, true) else {
                continue;
            };
            let depth = if self.view.cam.perspective {
                cx.len(hit.sub(self.view.cam.eye()))
            } else {
                hit.sub(self.view.cam.eye()).dot(view) as f64
            };
            if depth >= 0.0 {
                hits[i] = Some(depth);
            }
        }
        let horizontal_angle = (cx.dot(view, normals[2]).abs().min(1.0)).asin().to_degrees_py();
        if let Some(zd) = hits[2] {
            let vertical_depths: Vec<f64> = [hits[0], hits[1]].into_iter().flatten().collect();
            let depth_ratio = if horizontal_angle <= GROUND_PREFERENCE_ANGLE_DEG {
                GROUND_DEPTH_PREFERENCE_RATIO
            } else {
                PERSPECTIVE_GROUND_DEPTH_PREFERENCE_RATIO
            };
            let min_v = vertical_depths.iter().copied().fold(f64::INFINITY, |a, b| if b < a { b } else { a });
            if vertical_depths.is_empty() || zd <= min_v * depth_ratio {
                return Some((origin, normals[2]));
            }
        }
        let alignment = [cx.dot(ray_direction, normals[0]).abs(), cx.dot(ray_direction, normals[1]).abs()];
        let mut cand = [true, true];
        if let (Some(co), Some(cd)) = (cursor_origin, cursor_direction) {
            for k in 0..2 {
                cand[k] = self.view.ray_plane(co, cd, origin, normals[k], true).is_some();
            }
        }
        if !cand[0] && !cand[1] {
            cand = [true, true];
        }
        // `max(candidates, key=alignment.__getitem__)` over a set of one-letter
        // strings: with both present a tie would depend on the hash seed, and
        // the margin below settles every tie.
        let mut name = if cand[0] && cand[1] {
            if alignment[1] > alignment[0] {
                1
            } else {
                0
            }
        } else if cand[0] {
            0
        } else {
            1
        };
        if cand[0] && cand[1] && (alignment[0] - alignment[1]).abs() <= PLANE_ALIGNMENT_TIE_MARGIN {
            name = 1;
        }
        Some((origin, normals[name]))
    }

    /// `_current_work_plane(cursor)`: `(point, normal)`.
    pub fn current_work_plane(&self, cursor: (f64, f64)) -> (V3, V3) {
        let tool = self.tool;
        let cx = self.cx();
        if let Some(own) = tool.own_plane {
            return own;
        }
        if let Some(captured) = tool.work_plane {
            if let Some(fp) = self.planar_face_plane(cursor, Some(captured)) {
                return fp;
            }
            if let Some(start) = tool.start_point {
                if (cx.norm(captured.1).2.abs() as f64) > 0.94 && tool.plane_lock.is_none() && !tool.is_plane_lock {
                    if let Some(v) = self.near_horizon_vertical(start) {
                        return v;
                    }
                }
            }
            return captured;
        }
        let Some(start) = tool.start_point else {
            if tool.is_plane_lock {
                // `locked_work_plane(axes.origin())`
                if let Some(a) = tool.plane_lock {
                    return (self.axes.origin, self.axes.axis(a));
                }
                if let Some(n) = tool.plane_ref {
                    return (self.axes.origin, n);
                }
            }
            if let Some(e) = self.world.face_under(self.view, cursor.0, cursor.1) {
                return self.world.ents[e].plane_place;
            }
            if tool.is_plane_lock {
                if let Some(p) = self.camera_planar_work_plane(cursor) {
                    return p;
                }
            }
            let forward = self.forward();
            let center = match self.world.bounds {
                Some((lo, hi)) => lo.add(hi).mul(0.5),
                None => V3(0.0, 0.0, 0.0),
            };
            if let Some(p) = first_point_work_plane(&cx, forward, center) {
                return p;
            }
            return (V3(0.0, 0.0, 0.0), V3(0.0, 0.0, 1.0));
        };
        let base = self.start_point_plane(start);
        if !self.axis_locked {
            if let Some(fp) = self.hover_face_plane(cursor) {
                if tool.is_plane_lock {
                    return fp;
                }
                let cand = self.ray_hit_plane(cursor, base);
                let angle = match tool.magnetic_axis_deg {
                    Some(a) if a != 0.0 => a,
                    _ => self.inference_angle_deg,
                };
                match cand {
                    None => return fp,
                    Some(c) => {
                        if crate::compute::detect_axis_alignment(&cx, start, c, angle).is_none() {
                            return fp;
                        }
                    }
                }
            }
        }
        base
    }

    /// `_world_from_pixel(x, y)` and the plane it used (`_last_work_plane`).
    pub fn world_from_pixel(&self, x: f64, y: f64) -> (Option<V3>, Option<(V3, V3)>) {
        let Some((origin, direction)) = self.view.pixel_to_ray(x, y) else {
            return (None, None);
        };
        let cx = self.cx();
        let (plane_point, plane_normal) = self.current_work_plane((x, y));
        let tool = self.tool;
        let mut allow_planar_grazing = false;
        if tool.start_point.is_none() && tool.work_plane.is_none() && tool.plane_lock.is_none() && tool.plane_ref.is_none() && tool.is_plane_lock {
            let normal = cx.norm(plane_normal);
            let through_origin = cx.len(plane_point.sub(self.axes.origin)) < 1e-5;
            let principal = Axis::ALL.iter().any(|&n| cx.dot(normal, self.axes.axis(n)).abs() > 0.999);
            allow_planar_grazing = through_origin && principal;
        }
        let plane = Some((plane_point, plane_normal));
        if let Some(hit) = self.view.ray_plane(origin, direction, plane_point, plane_normal, allow_planar_grazing) {
            return (Some(hit), plane);
        }
        // The ray only grazes the work plane: believe the face under it.
        if let Some(e) = self.world.face_under(self.view, x, y) {
            let (fpoint, fnormal) = self.world.ents[e].plane_owner;
            return (self.view.ray_plane(origin, direction, fpoint, fnormal, false), plane);
        }
        (None, plane)
    }
}

trait ToDegreesPy {
    fn to_degrees_py(self) -> f64;
}
impl ToDegreesPy for f64 {
    /// `math.degrees`.
    fn to_degrees_py(self) -> f64 {
        self * (180.0 / std::f64::consts::PI)
    }
}
