// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! `core/inference.py::InferenceEngine.snap` and what it asks of the
//! viewport: the snap scene (`Viewport._snap_scene`), the cursor's world
//! point, the face under it, the occlusion rays and the axis-source cue.
//!
//! The reference state the viewport keeps between moves (the acquired
//! edge/point/face normal, the two encouraged points, the locks, the centre
//! under the cursor) is the caller's: [`InferenceState`].

use ingetrazo_geom::bevel_profiles::py_hypot2 as hypot;
use ingetrazo_math::V3;

use crate::camera::View;
use crate::compute::{compute_snap, LinearMode, RefMode, SnapParams};
use crate::types::*;
use crate::workplane::{PlaneCx, ToolState};
use crate::world::World;

/// Distances the engine reads from the viewport (`snap_threshold_px`…).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Settings {
    pub snap_threshold_px: f64,
    pub edge_snap_threshold_px: f64,
    pub inference_angle_deg: f64,
    pub pick_threshold_px: f64,
    pub center_min_radius_px: f64,
    pub center_hint_px: f64,
}

impl Default for Settings {
    fn default() -> Self {
        Settings {
            snap_threshold_px: 9.0,
            edge_snap_threshold_px: 14.0,
            inference_angle_deg: 3.0,
            pick_threshold_px: 8.0,
            center_min_radius_px: 12.0,
            center_hint_px: 40.0,
        }
    }
}

/// The inference engine's state (`InferenceEngine`'s attributes and the
/// viewport properties that delegate to it).
#[derive(Clone, Debug, Default)]
pub struct InferenceState {
    pub axis_lock: Option<Axis>,
    pub reference_edge: Option<EdgeRef>,
    pub reference_mode: Option<RefMode>,
    /// `_shift_lock`: direction and colour captured from an inference.
    pub shift_lock: Option<(V3, Rgb)>,
    pub acquired_edge: Option<EdgeRef>,
    pub acquired_point: Option<V3>,
    pub acquired_face_normal: Option<V3>,
    pub encouraged: Vec<V3>,
    pub linear_mode: Option<LinearMode>,
    /// The acquired circle/arc centre: `(centre, radius)`.
    pub center_ref: Option<(V3, f32)>,
    /// `_hover_edge`: the edge under the cursor at the last `acquire`.
    pub hover_edge: Option<EdgeRef>,
    /// `_hover_center`: the centre `acquire` found under the cursor.
    pub hover_center: Option<(V3, f32)>,
}

/// The cursor-bound callbacks of `compute_snap`.
pub struct ViewEnv<'a> {
    pub view: &'a View,
    pub world: &'a World,
    /// Cursor pixel (`project_onto_line` closes over it).
    pub px: f64,
    pub py: f64,
}

impl SnapEnv for ViewEnv<'_> {
    fn world_to_pixel(&self, w: V3) -> Option<(f64, f64)> {
        self.view.world_to_pixel(w)
    }
    fn project_onto_line(&self, start: V3, dir: V3) -> V3 {
        self.view.project_to_lock_line(start, dir, self.px, self.py)
    }
    fn is_occluded(&self, w: V3) -> bool {
        self.world.is_occluded(self.view.cam.eye(), w, self.view.h)
    }
}

/// `Viewport._center_hint_px`.
pub fn center_hint_px(view: &View, settings: &Settings, ref_: (V3, f32), pc: (f64, f64)) -> f64 {
    let rim = view.world_to_pixel(ref_.0.add(V3(ref_.1, 0.0, 0.0)));
    let rim2 = view.world_to_pixel(ref_.0.add(V3(0.0, ref_.1, 0.0)));
    let mut r_px = 0.0f64;
    for q in [rim, rim2].into_iter().flatten() {
        let d = hypot(q.0 - pc.0, q.1 - pc.1);
        if d > r_px {
            r_px = d;
        }
    }
    if r_px < settings.center_min_radius_px {
        return 0.0;
    }
    let half = 0.5 * r_px;
    if half < settings.center_hint_px {
        half
    } else {
        settings.center_hint_px
    }
}

/// `Viewport._np_mvp()` as float64 rows: `m[r][c]` of the row-major view of
/// the column-major `data()`.
fn np_mvp(view: &View) -> [[f64; 4]; 4] {
    let d = view.mvp().data();
    let mut m = [[0.0; 4]; 4];
    for r in 0..4 {
        for c in 0..4 {
            m[r][c] = d[r + 4 * c] as f64;
        }
    }
    m
}

/// `Viewport._project_px` of float64 points: `(px, py, in_front)`.
pub(crate) fn project_px(view: &View, pts: &[[f64; 3]]) -> Vec<(f64, f64, bool)> {
    let m = np_mvp(view);
    // pts @ M[:, :3].T + M[:, 3]: rot rows are M's rows.
    let rot = [[m[0][0], m[0][1], m[0][2]], [m[1][0], m[1][1], m[1][2]], [m[2][0], m[2][1], m[2][2]]];
    // The fourth clip component is a fourth row of the same product.
    let rot4 = [rot[0], rot[1], rot[2], [m[3][0], m[3][1], m[3][2]]];
    let single = pts.len() == 1;
    pts.iter()
        .map(|a| {
            let mut c = [0.0; 4];
            for r in 0..4 {
                let b = &rot4[r];
                let s = if single {
                    a[2].mul_add(b[2], a[0].mul_add(b[0], a[1] * b[1]))
                } else {
                    a[2].mul_add(b[2], a[1].mul_add(b[1], a[0] * b[0]))
                };
                c[r] = s + m[r][3];
            }
            let ok = c[3] > 0.0;
            let safe = if ok { c[3] } else { 1.0 };
            let px = (c[0] / safe * 0.5 + 0.5) * view.width as f64;
            let py = (1.0 - (c[1] / safe * 0.5 + 0.5)) * view.height as f64;
            (px, py, ok)
        })
        .collect()
}

/// `core.edge_grid.segment_distances` for one segment.
pub(crate) fn segment_distance(a: (f64, f64, bool), b: (f64, f64, bool), px: f64, py: f64) -> f64 {
    let ok = a.2 && b.2;
    let (dx, dy) = (b.0 - a.0, b.1 - a.1);
    let l2 = dx * dx + dy * dy;
    let safe = if l2 > 1e-12 { l2 } else { 1.0 };
    let mut t = ((px - a.0) * dx + (py - a.1) * dy) / safe;
    t = t.clamp(0.0, 1.0);
    let d = (a.0 + t * dx - px).hypot(a.1 + t * dy - py);
    if ok {
        d
    } else {
        f64::INFINITY
    }
}

/// `POINT_INFERENCE_PX`.
const POINT_INFERENCE_PX: f64 = 40.0;

/// `Viewport._snap_scene(px, py)`: the edges the rules see.
pub fn snap_scene(view: &View, world: &World, inf: &InferenceState, settings: &Settings, px: f64, py: f64) -> Vec<SnapEdge> {
    let mut lines: Vec<SnapEdge> = Vec::new();
    for g in &world.guides {
        if g.is_line() {
            if let Some((a, b)) = view.guide_snap_segment(g.point, g.direction.unwrap(), 1.5) {
                let mut e = SnapEdge::new(a, b);
                e.guide = true;
                lines.push(e);
            }
        } else {
            lines.push(SnapEdge::new(g.point, g.point));
        }
    }
    let mut near: Vec<SnapEdge> = Vec::new();
    // Group hard edges near the cursor, in index order (at most 48 within
    // 48 px; the fixtures never reach the cap).
    if !world.gedges.is_empty() {
        let pts: Vec<[f64; 3]> = world
            .gedges
            .iter()
            .flat_map(|g| [[g.a.0 as f64, g.a.1 as f64, g.a.2 as f64], [g.b.0 as f64, g.b.1 as f64, g.b.2 as f64]])
            .collect();
        let proj = project_px(view, &pts);
        let mut cand: Vec<(f64, usize)> = Vec::new();
        for (i, _) in world.gedges.iter().enumerate() {
            let d = segment_distance(proj[2 * i], proj[2 * i + 1], px, py);
            if d < 48.0 {
                cand.push((d, i));
            }
        }
        if cand.len() > 48 {
            cand.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
            cand.truncate(48);
        }
        for (_, i) in cand {
            let g = &world.gedges[i];
            let mut e = SnapEdge::new(g.a, g.b);
            e.context = Some(g.context);
            near.push(e);
        }
    }
    // Component origins within 40 px.
    if !world.origins.is_empty() {
        let pts: Vec<[f64; 3]> = world.origins.iter().map(|o| [o.0 as f64, o.1 as f64, o.2 as f64]).collect();
        let proj = project_px(view, &pts);
        for (i, p) in proj.iter().enumerate() {
            let d = if p.2 { (p.0 - px).hypot(p.1 - py) } else { f64::INFINITY };
            if d <= POINT_INFERENCE_PX {
                let o = world.origins[i];
                let mut e = SnapEdge::new(o, o);
                e.component_origin = true;
                near.push(e);
            }
        }
    }
    // Arc midpoints within 40 px.
    for m in &world.arc_mids {
        let Some(pc) = view.world_to_pixel(*m) else { continue };
        if hypot(pc.0 - px, pc.1 - py) > POINT_INFERENCE_PX {
            continue;
        }
        let mut e = SnapEdge::new(*m, *m);
        e.arc_midpoint = true;
        near.push(e);
    }
    // The centre of the last circle the cursor visited.
    if let Some(ref_) = inf.center_ref {
        if let Some(pc) = view.world_to_pixel(ref_.0) {
            if center_hint_px(view, settings, ref_, pc) > 0.0 {
                let mut e = SnapEdge::new(ref_.0, ref_.0);
                e.center = true;
                near.push(e);
            }
        }
    }
    if lines.is_empty() && near.is_empty() {
        // `return self.scene`: every loose edge, hidden ones included.
        return world.loose_edges.clone();
    }
    let mut out: Vec<SnapEdge> = world.loose_edges.iter().filter(|e| !e.hidden).cloned().collect();
    out.extend(lines);
    out.extend(near);
    out
}

/// The tip's key: the English text `SnapResult` is shown with
/// (`Viewport._SNAP_LABELS`, `"Level with point"` for a level `aligned`),
/// or `None`.
pub fn label_key(kind: SnapKind, axis: Option<Axis>) -> Option<&'static str> {
    let key = match kind {
        SnapKind::Endpoint => "Endpoint",
        SnapKind::Midpoint => "Midpoint",
        SnapKind::ArcMidpoint => "Arc midpoint",
        SnapKind::ComponentOrigin => "Component origin",
        SnapKind::OnLine => "On line",
        SnapKind::OnAxis => "On axis",
        SnapKind::OnEdge => "On edge",
        SnapKind::OnFace => "On face",
        SnapKind::Origin => "Origin",
        SnapKind::Extension => "Extension",
        SnapKind::Intersection => "Intersection",
        SnapKind::FromPoint => "From point",
        SnapKind::Aligned => {
            if axis == Some(Axis::Z) {
                "Level with point"
            } else {
                "In line with point"
            }
        }
        SnapKind::ThroughPoint => "Through point",
        SnapKind::PerpFace => "Perpendicular to face",
        SnapKind::Center => "Center",
        _ => return None,
    };
    Some(key)
}

/// The ScreenTip text for a result: the translated label plus the "in
/// group" / "in component" tail (`Viewport._draw_snap_indicator`).
pub fn tooltip(snap: &SnapResult, tr: &dyn Fn(&str) -> String) -> Option<String> {
    let mut label = label_key(snap.kind, snap.axis).map(tr);
    if let Some(l) = &snap.label {
        label = Some(l.clone());
    }
    let mut label = label?;
    match snap.context {
        Some(Context::Component) => {
            label.push(' ');
            label.push_str(&tr("in component"));
        }
        Some(Context::Group) => {
            label.push(' ');
            label.push_str(&tr("in group"));
        }
        None => {}
    }
    Some(label)
}

/// `Viewport._clip_segment_front` at the pick distance used by `pick_axis`.
fn point_to_segment_distance_2d(p: (f64, f64), a: (f64, f64), b: (f64, f64)) -> f64 {
    let (dx, dy) = (b.0 - a.0, b.1 - a.1);
    if dx == 0.0 && dy == 0.0 {
        return hypot(p.0 - a.0, p.1 - a.1);
    }
    let t = ((p.0 - a.0) * dx + (p.1 - a.1) * dy) / (dx * dx + dy * dy);
    let t = ingetrazo_math::pyfloat::py_max2(0.0, ingetrazo_math::pyfloat::py_min2(1.0, t));
    let (qx, qy) = (a.0 + t * dx, a.1 + t * dy);
    hypot(p.0 - qx, p.1 - qy)
}

/// `Viewport.pick_axis`: the model axis line under the cursor.
pub fn pick_axis(view: &View, axes: &Axes, pick_threshold_px: f64, x: f64, y: f64) -> Option<Axis> {
    let half = ingetrazo_model::guide::GUIDE_HALF_LEN;
    let mut best = None;
    let mut best_d = pick_threshold_px;
    let o = axes.origin;
    for name in Axis::ALL {
        let d = axes.axis(name).mul(half);
        let Some(seg) = view.clip_segment_front(o.sub(d), o.add(d)) else { continue };
        let (Some(pa), Some(pb)) = (view.world_to_pixel(seg.0), view.world_to_pixel(seg.1)) else {
            continue;
        };
        let dist = point_to_segment_distance_2d((x, y), pa, pb);
        if dist < best_d {
            best_d = dist;
            best = Some(name);
        }
    }
    best
}

/// `InferenceEngine._axis_source_cue`.
#[allow(clippy::too_many_arguments)]
fn axis_source_cue(
    snap: SnapResult,
    view: &View,
    world: &World,
    axes: &Axes,
    tool: &ToolState,
    settings: &Settings,
    px: f64,
    py: f64,
) -> SnapResult {
    let empty_only = tool.axis_source_empty_only;
    let allowed = |k: SnapKind| k == SnapKind::None || (!empty_only && k == SnapKind::OnFace);
    if !tool.axis_source || tool.start_point.is_some() || !allowed(snap.kind) {
        return snap;
    }
    let Some(name) = pick_axis(view, axes, settings.pick_threshold_px, px, py) else {
        return snap;
    };
    let axis = axes.axis(name);
    let o = axes.origin;
    let foot;
    if empty_only {
        foot = view.project_to_lock_line(o, axis, px, py);
        // `_axis_source_visible`: a step along the axis must show on screen.
        let before = view.world_to_pixel(foot.sub(axis.mul(0.25)));
        let after = view.world_to_pixel(foot.add(axis.mul(0.25)));
        let visible = match (before, after) {
            (Some(b), Some(a)) => hypot(a.0 - b.0, a.1 - b.1) >= 0.5,
            _ => false,
        };
        if !visible || world.is_occluded(view.cam.eye(), foot, view.h) {
            return snap;
        }
    } else {
        foot = o.add(axis.mul(snap.point.sub(o).dot(axis)));
    }
    let mut r = SnapResult::new(foot, SnapKind::OnAxis, name.color());
    r.axis = Some(name);
    r
}

/// Everything one `snap` call reads.
pub struct SnapInput<'a> {
    pub view: &'a View,
    pub world: &'a World,
    pub axes: &'a Axes,
    pub tool: &'a ToolState,
    pub state: &'a InferenceState,
    pub settings: &'a Settings,
    pub shift_held: bool,
}

/// What `snap` worked out on the way (for tests and the tool layer).
#[derive(Clone, Debug, Default)]
pub struct SnapTrace {
    /// `_world_from_pixel`.
    pub world_raw: Option<V3>,
    /// `_work_plane_normal()` after it.
    pub work_plane_normal: Option<V3>,
    /// `_still_face_under`.
    pub face_under: bool,
    /// The snap scene's edges.
    pub edges: Vec<SnapEdge>,
    /// `compute_snap`'s own answer, before the axis-source cue.
    pub core: Option<SnapResult>,
}

/// The `compute_snap(...)` call of `InferenceEngine.snap`, argument by
/// argument.
#[allow(clippy::too_many_arguments)]
pub fn snap_params<'a>(
    inp: &'a SnapInput,
    edges: &'a [SnapEdge],
    world_raw: V3,
    work_plane_normal: Option<V3>,
    face_under: bool,
    px: f64,
    py: f64,
) -> SnapParams<'a> {
    let mut p = SnapParams::new(world_raw, (px, py), edges, inp.settings.snap_threshold_px);
    p.has_project = true;
    p.has_occlusion = true;
    p.chain_first_point = inp.tool.chain_first_point;
    p.chain_first_is_start = inp.tool.chain_first_is_start;
    p.start_point = inp.tool.start_point;
    p.axis_lock = inp.state.axis_lock;
    p.shift_held = inp.shift_held;
    p.reference_edge = inp.state.reference_edge;
    p.reference_mode = inp.state.reference_mode;
    p.inference_angle_deg = inp.settings.inference_angle_deg;
    p.screen_axis_px = inp.tool.screen_axis_px;
    p.face_under_cursor = face_under;
    p.edge_threshold_px = Some(inp.settings.edge_snap_threshold_px);
    p.magnetic_axis_deg = inp.tool.magnetic_axis_deg;
    p.acquired_edge = inp.state.acquired_edge;
    p.acquired_point = inp.state.acquired_point;
    p.acquired_face_normal = inp.state.acquired_face_normal;
    p.acquired_points = &inp.state.encouraged;
    p.shift_lock_dir = inp.state.shift_lock.map(|s| s.0);
    p.shift_lock_color = inp.state.shift_lock.map(|s| s.1);
    p.linear_mode = inp.state.linear_mode.unwrap_or(LinearMode::All);
    p.work_plane_normal = work_plane_normal;
    p.radial_arm = inp.tool.radial_arm;
    p.axis_in_plane = !inp.tool.magnet_off_plane;
    p
}

/// `InferenceEngine.snap(px, py, modifiers)`.
pub fn snap(inp: &SnapInput, px: f64, py: f64) -> (Option<SnapResult>, SnapTrace) {
    let mut trace = SnapTrace::default();
    let cx = Cx { h: inp.view.h, axes: *inp.axes };
    let plane_cx = PlaneCx {
        view: inp.view,
        world: inp.world,
        axes: inp.axes,
        tool: inp.tool,
        axis_locked: inp.state.axis_lock.is_some(),
        inference_angle_deg: inp.settings.inference_angle_deg,
    };
    let (world_raw, plane) = plane_cx.world_from_pixel(px, py);
    trace.world_raw = world_raw;
    trace.work_plane_normal = plane.map(|p| p.1);
    let Some(world_raw) = world_raw else {
        return (None, trace);
    };
    if !inp.tool.uses_snap {
        return (Some(SnapResult::new(world_raw, SnapKind::None, COLOR_NONE)), trace);
    }
    let edges = snap_scene(inp.view, inp.world, inp.state, inp.settings, px, py);
    let face_under = inp.world.face_under(inp.view, px, py).is_some();
    trace.face_under = face_under;
    let env = ViewEnv { view: inp.view, world: inp.world, px, py };
    let p = snap_params(inp, &edges, world_raw, trace.work_plane_normal, face_under, px, py);
    let core = compute_snap(&env, cx, &p);
    trace.core = Some(core.clone());
    trace.edges = edges;
    let out = axis_source_cue(core, inp.view, inp.world, inp.axes, inp.tool, inp.settings, px, py);
    (Some(out), trace)
}

/// The screen distance from the cursor to every group hard edge
/// (`Viewport._gedge_dist`), in the order of `world.gedges`.
pub fn gedge_distances(view: &View, world: &World, px: f64, py: f64) -> Vec<f64> {
    let pts: Vec<[f64; 3]> = world
        .gedges
        .iter()
        .flat_map(|g| [[g.a.0 as f64, g.a.1 as f64, g.a.2 as f64], [g.b.0 as f64, g.b.1 as f64, g.b.2 as f64]])
        .collect();
    let proj = project_px(view, &pts);
    (0..world.gedges.len()).map(|i| segment_distance(proj[2 * i], proj[2 * i + 1], px, py)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn labels_follow_the_viewport_table() {
        assert_eq!(label_key(SnapKind::Endpoint, None), Some("Endpoint"));
        assert_eq!(label_key(SnapKind::Aligned, Some(Axis::Z)), Some("Level with point"));
        assert_eq!(label_key(SnapKind::Aligned, Some(Axis::X)), Some("In line with point"));
        assert_eq!(label_key(SnapKind::Axis, Some(Axis::X)), None);
        assert_eq!(label_key(SnapKind::None, None), None);
    }

    #[test]
    fn the_tip_adds_where_the_point_lives() {
        let mut r = SnapResult::new(V3(0.0, 0.0, 0.0), SnapKind::Midpoint, COLOR_MIDPOINT);
        r.context = Some(Context::Component);
        let tr = |s: &str| s.to_string();
        assert_eq!(tooltip(&r, &tr).as_deref(), Some("Midpoint in component"));
        r.context = Some(Context::Group);
        assert_eq!(tooltip(&r, &tr).as_deref(), Some("Midpoint in group"));
        r.label = Some("Level PA".into());
        assert_eq!(tooltip(&r, &tr).as_deref(), Some("Level PA in group"));
    }

    #[test]
    fn a_far_circle_has_no_hint_and_a_big_one_is_capped() {
        let cam = crate::OrbitCamera::default();
        let view = View::new(cam, 800, 600, ingetrazo_math::Hypot::Msvc);
        let s = Settings::default();
        let pc = view.world_to_pixel(V3(0.0, 0.0, 0.0)).unwrap();
        assert_eq!(center_hint_px(&view, &s, (V3(0.0, 0.0, 0.0), 0.01), pc), 0.0);
        assert_eq!(center_hint_px(&view, &s, (V3(0.0, 0.0, 0.0), 40.0), pc), s.center_hint_px);
    }
}

