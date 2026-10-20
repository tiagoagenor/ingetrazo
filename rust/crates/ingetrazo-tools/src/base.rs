// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! What the drawing tools share (`tools/base.py`): the [`DrawTool`] contract
//! the viewport calls, the plane lock every planar shape carries
//! (`PlaneLock`), the axis magnet (`AxisMagnet`) and the small helpers
//! (`face_the_plane`, `loop_normal`, the drawing-axes `plane_axes`).
//!
//! The viewport side of the dialogue - the pointer, the keys, the snap, the
//! value box - lives in [`crate::host`]; a tool here gets a [`Ctx`] (the
//! cursor's snapped point, its pixel, the modifiers, the snap) and the
//! [`Core`] (the document, the history, the camera and what the viewport
//! remembers), and answers with preview geometry and the commands it
//! executes.

use ingetrazo_geom::bevel_profiles::py_hypot2;
use ingetrazo_geom::facegeom::plane_axes as world_plane_axes;
use ingetrazo_math::{Hypot, V3};
use ingetrazo_snap::types::{Axes, Axis, COLOR_REFERENCE};
use ingetrazo_snap::{SnapKind, SnapResult, ToolState};

use crate::host::Core;
use crate::vcb::VcbValue;

// ---- Qt's key and modifier codes -----------------------------------------------------------

pub const KEY_ESCAPE: i32 = 0x0100_0000;
pub const KEY_BACKSPACE: i32 = 0x0100_0003;
pub const KEY_RETURN: i32 = 0x0100_0004;
pub const KEY_ENTER: i32 = 0x0100_0005;
pub const KEY_DELETE: i32 = 0x0100_0007;
pub const KEY_LEFT: i32 = 0x0100_0012;
pub const KEY_UP: i32 = 0x0100_0013;
pub const KEY_RIGHT: i32 = 0x0100_0014;
pub const KEY_DOWN: i32 = 0x0100_0015;
pub const KEY_SHIFT: i32 = 0x0100_0020;
pub const KEY_CONTROL: i32 = 0x0100_0021;
pub const KEY_META: i32 = 0x0100_0022;
pub const KEY_ALT: i32 = 0x0100_0023;
pub const KEY_ALTGR: i32 = 0x0100_1103;

pub const MOD_SHIFT: u32 = 0x0200_0000;
pub const MOD_CONTROL: u32 = 0x0400_0000;
pub const MOD_ALT: u32 = 0x0800_0000;
pub const MOD_META: u32 = 0x1000_0000;

/// `PLANE_LOCK_KEYS`: the arrow -> the axis a locked plane is NORMAL to.
pub fn plane_lock_axis(key: i32) -> Option<Axis> {
    match key {
        KEY_RIGHT => Some(Axis::X),
        KEY_LEFT => Some(Axis::Y),
        KEY_UP => Some(Axis::Z),
        _ => None,
    }
}

/// `PLANE_LOCK_NAMES`.
pub fn plane_lock_name(a: Axis) -> &'static str {
    match a {
        Axis::X => "YZ",
        Axis::Y => "XZ",
        Axis::Z => "XY",
    }
}

// ---- the call --------------------------------------------------------------------------------

/// `ToolContext`: a hover or a click as the tool sees it.
#[derive(Clone, Debug)]
pub struct Ctx {
    pub world: V3,
    /// `ctx.screen`: the exact pointer position (a `QPointF`).
    pub screen: (f64, f64),
    pub mods: u32,
    pub snap: SnapResult,
}

/// One value a test reads back from a tool's state.
#[derive(Clone, Debug, PartialEq)]
pub enum Field {
    Null,
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(String),
    P(V3),
    Plane(V3, V3),
    Color(Vec<f64>),
    List(Vec<Field>),
}

impl Field {
    pub fn point(p: Option<V3>) -> Field {
        p.map(Field::P).unwrap_or(Field::Null)
    }
    pub fn plane(p: Option<(V3, V3)>) -> Field {
        p.map(|(a, b)| Field::Plane(a, b)).unwrap_or(Field::Null)
    }
    pub fn float(f: Option<f64>) -> Field {
        f.map(Field::Float).unwrap_or(Field::Null)
    }
    pub fn string(s: Option<&str>) -> Field {
        s.map(|s| Field::Str(s.to_string())).unwrap_or(Field::Null)
    }
    pub fn color(c: Option<[f64; 4]>) -> Field {
        c.map(|c| Field::Color(c.to_vec())).unwrap_or(Field::Null)
    }
}

/// A tool the viewport drives (`tools.base.Tool` for the drawing tools).
pub trait DrawTool {
    fn name(&self) -> &'static str;
    fn uses_snap(&self) -> bool {
        true
    }
    /// What the snap engine reads off the tool (`start_point`,
    /// `work_plane`, `plane_lock`, the magnets...).
    fn snap_state(&self, core: &Core) -> ToolState;
    fn start_point(&self) -> Option<V3>;
    fn work_plane(&self) -> Option<(V3, V3)>;
    fn set_work_plane(&mut self, plane: Option<(V3, V3)>);
    /// The tool's plane lock, when it is a `PlaneLock`.
    fn planar(&mut self) -> Option<&mut PlaneState> {
        None
    }
    fn plane_ref_kind(&self) -> Option<RefKind> {
        None
    }
    /// `Tool.snap_kind`, pushed by the viewport before every call.
    fn set_snap_kind(&mut self, _kind: Option<SnapKind>) {}
    /// `Viewport._tool_busy`: an operation Esc should cancel.
    fn busy(&self) -> bool {
        self.start_point().is_some()
    }
    fn vcb_comma_lists(&self) -> bool {
        false
    }
    /// `vcb_label` (or `vcb_caption()` when the tool has one).
    fn vcb_label(&self) -> String;

    fn on_activate(&mut self, _core: &mut Core) {}
    fn on_deactivate(&mut self, _core: &mut Core) {}
    fn on_click(&mut self, core: &mut Core, ctx: &Ctx);
    fn on_double_click(&mut self, core: &mut Core, ctx: &Ctx) {
        self.on_click(core, ctx)
    }
    fn on_hover(&mut self, core: &mut Core, ctx: &Ctx);
    fn on_release(&mut self, _core: &mut Core) {}
    fn on_cancel(&mut self, core: &mut Core);
    /// Ctrl+Z first goes to the tool; `true` = it took it.
    fn on_undo(&mut self, _core: &mut Core) -> bool {
        false
    }
    fn on_key(&mut self, _core: &mut Core, _key: i32, _mods: u32) -> bool {
        false
    }
    fn on_key_release(&mut self, _core: &mut Core, _key: i32) -> bool {
        false
    }
    fn on_value(&mut self, _core: &mut Core, _value: &VcbValue) -> bool {
        false
    }
    /// `on_segments_value`: `None` when the tool has no such handler.
    fn on_segments_value(&mut self, _core: &mut Core, _n: i64) -> Option<bool> {
        None
    }
    /// `on_radius_value`: `None` when the tool has no such handler.
    fn on_radius_value(&mut self, _core: &mut Core, _r: f64) -> Option<bool> {
        None
    }
    fn rubber_band_lines(&self, core: &Core) -> Vec<(V3, V3)>;
    fn value_label(&self, _core: &Core) -> Option<(String, V3)> {
        None
    }
    /// The tool's state, by Python attribute name, for the parity test.
    fn fields(&self) -> Vec<(&'static str, Field)>;
}

// ---- drawing axes ------------------------------------------------------------------------------

/// `core.axes.plane_axes(normal)`: the in-plane `(u, v)` of a drawing.
pub fn plane_axes(axes: &Axes, normal: V3, h: Hypot) -> (V3, V3) {
    if axes.world {
        return world_plane_axes(normal, h);
    }
    let n = normal.normalized(h);
    let mut r = axes.x;
    let mut u = r.sub(n.mul(r.dot(n)));
    if (u.length(h) as f64) < 0.1 {
        r = axes.y;
        u = r.sub(n.mul(r.dot(n)));
    }
    let u = u.normalized(h);
    let v = n.cross(u).normalized(h);
    (u, v)
}

/// `QVector3D` truthiness: a null vector is falsy in PySide.
pub fn truthy(p: Option<V3>) -> Option<V3> {
    p.filter(|v| !v.is_null())
}

/// `loop_normal`: Newell normal of a closed loop of points.
pub fn loop_normal(corners: &[V3]) -> V3 {
    let mut n = V3(0.0, 0.0, 0.0);
    for i in 0..corners.len() {
        n = n.add(corners[i].cross(corners[(i + 1) % corners.len()]));
    }
    n
}

/// `face_the_plane`: wind `corners` so the face looks the way its plane does.
pub fn face_the_plane(corners: &[V3], plane_normal: Option<V3>) -> Vec<V3> {
    let Some(pn) = plane_normal else { return corners.to_vec() };
    if corners.len() < 3 {
        return corners.to_vec();
    }
    let dot = loop_normal(corners).dot(pn) as f64;
    if dot < -1e-12 {
        let mut out = vec![corners[0]];
        out.extend(corners[1..].iter().rev());
        return out;
    }
    corners.to_vec()
}

// ---- PlaneLock -----------------------------------------------------------------------------------

/// What the Down arrow (or a held Shift) holds the plane to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RefKind {
    Edge,
    Face,
    /// Marked by a held Shift: not locked.
    Plane,
}

impl RefKind {
    pub fn name(self) -> &'static str {
        match self {
            RefKind::Edge => "edge",
            RefKind::Face => "face",
            RefKind::Plane => "plane",
        }
    }
}

/// The state `PlaneLock` adds to a tool.
#[derive(Clone, Debug, Default)]
pub struct PlaneState {
    pub plane_lock: Option<Axis>,
    pub plane_ref: Option<(V3, RefKind)>,
    pub hover_plane: Option<(V3, V3)>,
    pub work_plane: Option<(V3, V3)>,
    pub snap_kind: Option<SnapKind>,
}

/// `PlaneLock.PREVIEW_PX`.
pub const PREVIEW_PX: f64 = 22.0;
/// `PlaneLock.OFF_PLANE_TOL`.
pub const OFF_PLANE_TOL: f64 = 1e-3;

/// A tool that is a `PlaneLock`.
pub trait Planar: DrawTool {
    fn pl(&self) -> &PlaneState;
    fn pl_mut(&mut self) -> &mut PlaneState;
    /// `plane_points()`: the points the shape must pass through.
    fn plane_points(&self) -> Vec<Option<V3>> {
        Vec::new()
    }
    /// `plane_lock_after_the_first_point`.
    fn lock_after_first_point(&self) -> bool {
        true
    }
}

fn own_plane(st: &PlaneState, axes: &Axes) -> (V3, V3) {
    st.work_plane.or(st.hover_plane).unwrap_or((axes.origin, axes.axis(Axis::Z)))
}

/// `PlaneLock.note_plane`: remember the plane the viewport just hit.
pub fn note_plane(st: &mut PlaneState, core: &Core) {
    if let Some(p) = core.last_work_plane {
        st.hover_plane = Some(p);
    }
}

/// `PlaneLock.drawing_plane`.
pub fn drawing_plane<T: Planar + ?Sized>(t: &T, core: &Core) -> (V3, V3) {
    let plane = own_plane(t.pl(), &core.axes);
    snapped_plane(t, plane, core).unwrap_or(plane)
}

/// `PlaneLock.snapped_plane`.
pub fn snapped_plane<T: Planar + ?Sized>(t: &T, plane: (V3, V3), core: &Core) -> Option<(V3, V3)> {
    let st = t.pl();
    let h = core.h();
    if st.plane_lock.is_some() || st.plane_ref.is_some() {
        return None;
    }
    if let Some(k) = st.snap_kind {
        if !k.is_point() {
            return None;
        }
    }
    let pts: Vec<V3> = t.plane_points().into_iter().flatten().collect();
    if pts.len() < 2 {
        return None;
    }
    let (origin, normal) = plane;
    let n = normal.normalized(h);
    let tol = OFF_PLANE_TOL;
    if pts.iter().all(|p| (p.sub(origin).dot(n) as f64).abs() < tol) {
        return None;
    }
    let a = pts[0];
    if pts.len() >= 3 {
        let through = pts[1].sub(a).cross(pts[2].sub(a));
        if (through.length(h) as f64) > tol * tol {
            return Some((a, through.normalized(h)));
        }
    }
    let span = pts[pts.len() - 1].sub(a);
    let flat: Vec<V3> = Axis::ALL
        .iter()
        .map(|&k| core.axes.axis(k))
        .filter(|ax| (span.dot(*ax) as f64).abs() < tol)
        .collect();
    if flat.len() != 1 {
        return None;
    }
    Some((a, flat[0]))
}

/// `PlaneLock.adopt_snapped_plane`.
pub fn adopt_snapped_plane<T: Planar + ?Sized>(t: &mut T, core: &Core) {
    let own = own_plane(t.pl(), &core.axes);
    if let Some(p) = snapped_plane(&*t, own, core) {
        t.pl_mut().work_plane = Some(p);
    }
}

/// `PlaneLock.locked_work_plane(point)`.
pub fn locked_work_plane(st: &PlaneState, point: V3, axes: &Axes) -> Option<(V3, V3)> {
    if let Some(a) = st.plane_lock {
        return Some((point, axes.axis(a)));
    }
    st.plane_ref.map(|(n, _)| (point, n))
}

/// `PlaneLock.plane_color(normal)`: RGBA of the axis the plane is normal to.
pub fn plane_color(normal: V3, axes: &Axes, h: Hypot) -> Option<[f64; 4]> {
    let mut n = normal.normalized(h);
    if !axes.world {
        n = axes.to_local(n);
    }
    for (axis, comp) in [(Axis::X, n.0), (Axis::Y, n.1), (Axis::Z, n.2)] {
        if (comp as f64).abs() > 0.99 {
            let c = axis.color();
            return Some([c[0], c[1], c[2], 1.0]);
        }
    }
    None
}

/// `PlaneLock.lock_color()`: the rubber band colour for a locked or
/// camera-selected plane.
pub fn lock_color(st: &PlaneState, axes: &Axes, h: Hypot) -> Option<[f64; 4]> {
    if st.plane_ref.is_some() {
        return Some([COLOR_REFERENCE[0], COLOR_REFERENCE[1], COLOR_REFERENCE[2], 1.0]);
    }
    if let Some(a) = st.plane_lock {
        return plane_color(axes.axis(a), axes, h);
    }
    if let Some(p) = st.hover_plane {
        return plane_color(p.1, axes, h);
    }
    None
}

/// `PlaneLock.world_per_pixel`: metres per screen pixel at `point` along
/// `along`.
pub fn world_per_pixel(core: &Core, point: V3, along: V3) -> Option<f64> {
    let a = core.world_to_pixel(point)?;
    let b = core.world_to_pixel(point.add(along.normalized(core.h()).mul(0.25f32)))?;
    let d = py_hypot2(b.0 - a.0, b.1 - a.1);
    if d > 1e-6 {
        Some(0.25 / d)
    } else {
        None
    }
}

/// `PlaneLock.plane_preview_scale`.
pub fn plane_preview_scale(core: &Core, point: V3, u: V3, v: V3) -> Option<f64> {
    let mut best: Option<f64> = None;
    for axis in [u, v] {
        if let Some(s) = world_per_pixel(core, point, axis) {
            best = Some(match best {
                Some(b) if b <= s => b,
                _ => s,
            });
        }
    }
    best
}

/// `PlaneLock.preview_plane(viewport, point)`: the plane the shape WOULD
/// take at `point` before the first click.
pub fn preview_plane(st: &PlaneState, core: &Core, point: V3) -> (V3, V3) {
    if let Some(l) = locked_work_plane(st, point, &core.axes) {
        return l;
    }
    match core.last_work_plane {
        Some(p) => (point, p.1),
        None => (point, core.axes.axis(Axis::Z)),
    }
}

/// `PlaneLock._anchor_lock`.
fn anchor_lock<T: Planar + ?Sized>(t: &mut T, core: &Core) {
    let Some(start) = t.start_point() else { return };
    let w = locked_work_plane(t.pl(), start, &core.axes);
    t.pl_mut().work_plane = w;
}

/// `PlaneLock.plane_lock_key`: an arrow (or Down) locks the drawing plane.
pub fn plane_lock_key<T: Planar + ?Sized>(t: &mut T, core: &mut Core, key: i32) -> bool {
    if t.start_point().is_some() && !t.lock_after_first_point() {
        return false;
    }
    if key == KEY_DOWN {
        return reference_lock_key(t, core);
    }
    let Some(axis) = plane_lock_axis(key) else { return false };
    toggle_plane_lock(t, core, axis);
    true
}

/// The body shared by `plane_lock_key` and the Rectangle's
/// `_plane_lock_after_start`: toggle the arrow lock, re-anchor, say it.
pub fn toggle_plane_lock<T: Planar + ?Sized>(t: &mut T, core: &mut Core, axis: Axis) {
    let st = t.pl_mut();
    st.plane_ref = None;
    st.plane_lock = if st.plane_lock == Some(axis) { None } else { Some(axis) };
    anchor_lock(t, core);
    let msg = match t.pl().plane_lock {
        Some(a) => format!("Drawing plane locked: {}", plane_lock_name(a)),
        None => "Drawing plane free".to_string(),
    };
    core.flash(msg);
}

/// `PlaneLock.reference_lock_key`: the Down arrow.
pub fn reference_lock_key<T: Planar + ?Sized>(t: &mut T, core: &mut Core) -> bool {
    if t.pl().plane_ref.is_some() {
        t.pl_mut().plane_ref = None;
        anchor_lock(t, core);
        core.flash("Drawing plane free".to_string());
    } else {
        let Some(r) = hovered_reference(core) else { return true };
        let st = t.pl_mut();
        st.plane_ref = Some(r);
        st.plane_lock = None;
        anchor_lock(t, core);
        core.flash(
            if r.1 == RefKind::Edge { "Drawing plane locked: perpendicular to the edge" } else { "Drawing plane locked: parallel to the face" }.to_string(),
        );
    }
    true
}

/// `PlaneLock.hovered_reference`: `(normal, kind)` of the reference under the
/// cursor: the edge beats the face.
pub fn hovered_reference(core: &mut Core) -> Option<(V3, RefKind)> {
    let h = core.h();
    if let Some(edge) = core.inference.hover_edge {
        let d = edge.b.sub(edge.a);
        if (d.length(h) as f64) > 1e-9 {
            return Some((d.normalized(h), RefKind::Edge));
        }
    }
    if let Some((x, y)) = core.last_mouse_pos {
        if let Some((_p, n)) = core.pick_face_placement(x, y) {
            if (n.length(h) as f64) > 1e-9 {
                return Some((n.normalized(h), RefKind::Face));
            }
        }
    }
    None
}

/// `PlaneLock.mark_plane`: the held-Shift mark.
pub fn mark_plane(st: &mut PlaneState, h: Hypot) -> bool {
    if st.plane_lock.is_some() || st.plane_ref.is_some() {
        return false;
    }
    let Some(plane) = st.work_plane.or(st.hover_plane) else { return false };
    st.plane_ref = Some((plane.1.normalized(h), RefKind::Plane));
    true
}

/// `PlaneLock.clear_plane_lock`.
pub fn clear_plane_lock(st: &mut PlaneState) {
    st.plane_lock = None;
    st.plane_ref = None;
}

/// The fields every `PlaneLock` has, for [`DrawTool::fields`].
pub fn plane_fields(st: &PlaneState, wireframe: Option<[f64; 4]>) -> Vec<(&'static str, Field)> {
    vec![
        ("work_plane", Field::plane(st.work_plane)),
        ("hover_plane", Field::plane(st.hover_plane)),
        ("plane_lock", Field::string(st.plane_lock.map(|a| a.name()))),
        ("plane_ref", st.plane_ref.map(|(n, k)| Field::List(vec![Field::P(n), Field::Str(k.name().into())])).unwrap_or(Field::Null)),
        ("snap_kind", Field::string(st.snap_kind.map(|k| k.as_str()))),
        ("wireframe_color", Field::color(wireframe)),
    ]
}

// ---- AxisMagnet ---------------------------------------------------------------------------------

/// `AxisMagnet`: within 3 degrees of an axis the point lands ON it, and the
/// axis the work plane cannot hold is found in pixels.
pub const MAGNET_DEG: f64 = 3.0;
pub const MAGNET_PX: f64 = 9.0;

/// The part of a [`ToolState`] a planar tool's plane lock decides.
pub fn planar_snap_state(st: &PlaneState, start: Option<V3>) -> ToolState {
    ToolState {
        uses_snap: true,
        is_plane_lock: true,
        start_point: start,
        work_plane: st.work_plane,
        plane_lock: st.plane_lock,
        plane_ref: st.plane_ref.map(|(n, _)| n),
        ..ToolState::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn square() -> Vec<V3> {
        vec![V3(0.0, 0.0, 0.0), V3(1.0, 0.0, 0.0), V3(1.0, 1.0, 0.0), V3(0.0, 1.0, 0.0)]
    }

    #[test]
    fn a_loop_is_wound_to_look_the_way_its_plane_does() {
        let up = Some(V3(0.0, 0.0, 1.0));
        assert_eq!(face_the_plane(&square(), up), square());
        let down = face_the_plane(&square(), Some(V3(0.0, 0.0, -1.0)));
        assert_eq!(down, vec![V3(0.0, 0.0, 0.0), V3(0.0, 1.0, 0.0), V3(1.0, 1.0, 0.0), V3(1.0, 0.0, 0.0)]);
        // No plane, or a loop square to it: left as it came.
        assert_eq!(face_the_plane(&square(), None), square());
        assert_eq!(face_the_plane(&square(), Some(V3(1.0, 0.0, 0.0))), square());
    }

    #[test]
    fn the_arrows_name_the_axis_their_plane_is_normal_to() {
        assert_eq!(plane_lock_axis(KEY_RIGHT), Some(Axis::X));
        assert_eq!(plane_lock_axis(KEY_LEFT), Some(Axis::Y));
        assert_eq!(plane_lock_axis(KEY_UP), Some(Axis::Z));
        assert_eq!(plane_lock_axis(KEY_DOWN), None);
        assert_eq!(plane_lock_name(Axis::X), "YZ");
    }

    #[test]
    fn a_null_vector_is_falsy_as_in_pyside() {
        assert_eq!(truthy(Some(V3(0.0, 0.0, 0.0))), None);
        assert_eq!(truthy(Some(V3(0.0, 0.0, 1.0))), Some(V3(0.0, 0.0, 1.0)));
    }

    #[test]
    fn the_drawing_axes_of_a_plane_are_the_worlds() {
        let (u, v) = plane_axes(&Axes::world(), V3(0.0, 0.0, 1.0), Hypot::Msvc);
        assert_eq!((u, v), (V3(1.0, 0.0, 0.0), V3(0.0, 1.0, 0.0)));
    }
}
