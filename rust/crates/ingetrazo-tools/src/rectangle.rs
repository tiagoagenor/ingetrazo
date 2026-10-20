// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! The Rectangle tool (`tools/rectangle.py`): two clicks define opposite
//! corners on the work plane - or, with Ctrl, a centre and a corner. The
//! face and its four edges are one command (one undo step). A rectangle
//! within 4% of a square snaps to one ("Cuadrado"); Shift (or Down) holds the
//! square; an arrow chooses the drawing plane, before and after the first
//! corner.

use ingetrazo_geom::bevel_profiles::py_hypot2;
use ingetrazo_geom::facegeom::plane_axes as world_plane_axes;
use ingetrazo_math::V3;
use ingetrazo_ops::draw::AddFaceCommand;
use ingetrazo_ops::edits::build_add_edges;
use ingetrazo_snap::types::Axis;
use ingetrazo_snap::ToolState;

use crate::base::*;
use crate::host::Core;
use crate::units::fmt_pair;
use crate::vcb::VcbValue;

/// `RectangleTool.SQUARE_TOL`.
const SQUARE_TOL: f64 = 0.04;

/// The side the cursor is stretching.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Side {
    U,
    V,
}

pub struct RectangleTool {
    pub start_point: Option<V3>,
    pub hover_point: Option<V3>,
    from_center: bool,
    pl: PlaneState,
    shift_square_lock: bool,
    arrow_square_lock: bool,
    live_side: Option<Side>,
    ctrl_toggled: bool,
    wireframe_color: Option<[f64; 4]>,
    /// `_viewport is not None`: set by the first hover.
    viewport_seen: bool,
}

impl Default for RectangleTool {
    fn default() -> Self {
        RectangleTool::new()
    }
}

impl RectangleTool {
    pub fn new() -> RectangleTool {
        RectangleTool {
            start_point: None,
            hover_point: None,
            from_center: false,
            pl: PlaneState::default(),
            shift_square_lock: false,
            arrow_square_lock: false,
            live_side: None,
            ctrl_toggled: false,
            wireframe_color: None,
            viewport_seen: false,
        }
    }

    fn reset(&mut self) {
        self.start_point = None;
        self.pl.work_plane = None;
        self.pl.hover_plane = None;
        self.wireframe_color = None;
        self.shift_square_lock = false;
        self.arrow_square_lock = false;
        self.live_side = None;
        clear_plane_lock(&mut self.pl);
    }

    fn square_locked(&self) -> bool {
        self.shift_square_lock || self.arrow_square_lock
    }

    /// `_axes`: in-plane horizontal / vertical axes for the drawing plane.
    fn axes(&self, core: &Core) -> (V3, V3) {
        let (_, normal) = drawing_plane(self, core);
        plane_axes_of(core, normal)
    }

    /// `_dimensions(a, b)`: signed width and height along the plane's axes.
    fn dimensions(&self, core: &Core, a: V3, b: V3) -> (f64, f64) {
        let (u, v) = self.axes(core);
        let delta = b.sub(a);
        (delta.dot(u) as f64, delta.dot(v) as f64)
    }

    /// `_square_corner(a, b, force)`.
    fn square_corner(&self, core: &Core, a: V3, b: V3, force: bool) -> (V3, bool) {
        let (u, v) = self.axes(core);
        let (du, dv) = self.dimensions(core, a, b);
        let (adu, adv) = (du.abs(), dv.abs());
        if adu < 1e-6 || adv < 1e-6 {
            return (b, false);
        }
        if force || (adu - adv).abs() <= SQUARE_TOL * ingetrazo_math::pyfloat::py_max2(adu, adv) {
            let side = ingetrazo_math::pyfloat::py_max2(adu, adv);
            let far = a.add(u.mul(side.copysign(du) as f32)).add(v.mul(side.copysign(dv) as f32));
            return (far, true);
        }
        (b, false)
    }

    /// `_span(cursor)`: the two opposite corners the cursor asks for.
    fn span(&self, core: &Core, cursor: V3) -> (V3, V3) {
        let start = self.start_point.expect("a start");
        let (far, _sq) = self.square_corner(core, start, cursor, self.square_locked());
        if self.from_center {
            return (start.mul(2.0f32).sub(far), far);
        }
        (start, far)
    }

    /// `_corners(a, b)`.
    fn corners(&self, core: &Core, a: V3, b: V3) -> Vec<V3> {
        let (u, v) = self.axes(core);
        let delta = b.sub(a);
        let du = delta.dot(u);
        let dv = delta.dot(v);
        vec![a, a.add(u.mul(du)), a.add(u.mul(du)).add(v.mul(dv)), a.add(v.mul(dv))]
    }

    /// `_note_live_side(prev, new)`.
    fn note_live_side(&mut self, core: &Core, prev: V3, new: V3) {
        let (u, v) = self.axes(core);
        let delta = new.sub(prev);
        let du = (delta.dot(u) as f64).abs();
        let dv = (delta.dot(v) as f64).abs();
        if du > 2.0 * dv && du > 1e-6 {
            self.live_side = Some(Side::U);
        } else if dv > 2.0 * du && dv > 1e-6 {
            self.live_side = Some(Side::V);
        }
    }

    /// `_cursor_preview`: the little square on the cursor before the first
    /// corner.
    fn cursor_preview(&self, core: &Core) -> Vec<(V3, V3)> {
        if !self.viewport_seen {
            return Vec::new();
        }
        let Some(hover) = self.hover_point else { return Vec::new() };
        let (_point, normal) = preview_plane(&self.pl, core, hover);
        let (u, v) = rect_plane_axes(core, normal);
        let Some(scale) = plane_preview_scale(core, hover, u, v) else { return Vec::new() };
        let h = (0.5 * PREVIEW_PX * scale) as f32;
        let c = hover;
        let pts = [
            c.add(u.mul(h)).add(v.mul(h)),
            c.sub(u.mul(h)).add(v.mul(h)),
            c.sub(u.mul(h)).sub(v.mul(h)),
            c.add(u.mul(h)).sub(v.mul(h)),
        ];
        (0..4).map(|i| (pts[i], pts[(i + 1) % 4])).collect()
    }

    /// `_degenerate_reason`: why the rectangle came out flat.
    fn degenerate_reason(&self, core: &Core, world: V3) -> String {
        let plain = "Rectangle needs two sides \u{2014} pick the opposite corner".to_string();
        let (Some(wp), Some(start)) = (self.pl.work_plane, self.start_point) else { return plain };
        let h = core.h();
        let (origin, normal) = wp;
        let n = normal.normalized(h);
        let off = (world.sub(origin).dot(n) as f64).abs();
        if off < 1e-3 {
            return plain;
        }
        let span = world.sub(start);
        // `min(("x","y","z"), key=|span . axis|)`: the first of the smallest.
        let mut axis = Axis::X;
        let mut best = (span.dot(core.axes.axis(Axis::X)) as f64).abs();
        for a in [Axis::Y, Axis::Z] {
            let d = (span.dot(core.axes.axis(a)) as f64).abs();
            if d < best {
                best = d;
                axis = a;
            }
        }
        let arrow = match axis {
            Axis::X => "\u{2192}",
            Axis::Y => "\u{2190}",
            Axis::Z => "\u{2191}",
        };
        format!("That corner is off the drawing plane, so the rectangle has no width. Press {arrow} for the {} plane.", plane_lock_name(axis))
    }

    /// `_commit_rect`.
    fn commit_rect(&mut self, core: &mut Core, corners: Vec<V3>) {
        let (u, v) = self.axes(core);
        let corners = face_the_plane(&corners, Some(u.cross(v)));
        let segments: Vec<(V3, V3)> = (0..4).map(|i| (corners[i], corners[(i + 1) % 4])).collect();
        let extra: Vec<Box<dyn ingetrazo_ops::command::Command>> = vec![Box::new(AddFaceCommand::new(corners.clone(), true, None, None))];
        let eye = None;
        let cmd = build_add_edges(&core.scene, &core.cx, &segments, false, extra, eye, true);
        core.exec(cmd);
        self.reset();
    }

    fn set_from_center(&mut self, on: bool) {
        self.from_center = on;
    }

    /// `_toggle_center`.
    fn toggle_center(&mut self, core: &mut Core, say: bool) {
        let on = !self.from_center;
        self.set_from_center(on);
        if say {
            core.flash(if self.from_center { "Rectangle from the centre" } else { "Rectangle from a corner" }.to_string());
        }
    }

    /// `_plane_lock_after_start`: an arrow AFTER the first corner still
    /// chooses the drawing plane (issue #543).
    fn plane_lock_after_start(&mut self, core: &mut Core, key: i32) -> bool {
        let axis = plane_lock_axis(key).expect("an arrow");
        self.pl.plane_ref = None;
        self.pl.plane_lock = if self.pl.plane_lock == Some(axis) { None } else { Some(axis) };
        let start = self.start_point.expect("a start");
        self.pl.work_plane = if self.pl.plane_lock.is_some() { locked_work_plane(&self.pl, start, &core.axes) } else { None };
        core.flash(match self.pl.plane_lock {
            Some(a) => format!("Drawing plane locked: {}", plane_lock_name(a)),
            None => "Drawing plane free".to_string(),
        });
        true
    }
}

/// `tools.rectangle._plane_axes`.
fn rect_plane_axes(core: &Core, normal: V3) -> (V3, V3) {
    if !core.axes.world {
        return plane_axes(&core.axes, normal, core.h());
    }
    world_plane_axes(normal, core.h())
}

fn plane_axes_of(core: &Core, normal: V3) -> (V3, V3) {
    rect_plane_axes(core, normal)
}

impl Planar for RectangleTool {
    fn pl(&self) -> &PlaneState {
        &self.pl
    }
    fn pl_mut(&mut self) -> &mut PlaneState {
        &mut self.pl
    }
    /// First corner and far corner.
    fn plane_points(&self) -> Vec<Option<V3>> {
        match self.start_point {
            None => Vec::new(),
            Some(s) => vec![Some(s), self.hover_point],
        }
    }
}

impl DrawTool for RectangleTool {
    fn name(&self) -> &'static str {
        "Rectangle"
    }

    fn snap_state(&self, _core: &Core) -> ToolState {
        // No AxisMagnet here (issue #52): the second click is the OPPOSITE
        // CORNER, and a corner pulled onto the axis through the first is a
        // rectangle of zero height. `chain_first_point` is never set.
        ToolState { axis_source: true, axis_source_empty_only: true, ..planar_snap_state(&self.pl, self.start_point) }
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
    fn set_snap_kind(&mut self, kind: Option<ingetrazo_snap::SnapKind>) {
        self.pl.snap_kind = kind;
    }
    fn vcb_comma_lists(&self) -> bool {
        true
    }
    fn vcb_label(&self) -> String {
        "Dimensions".into()
    }

    fn on_activate(&mut self, _core: &mut Core) {
        self.reset();
        self.set_from_center(false); // a pick-up starts corner-wise
    }

    fn on_deactivate(&mut self, _core: &mut Core) {
        self.reset();
        self.hover_point = None;
    }

    fn on_click(&mut self, core: &mut Core, ctx: &Ctx) {
        note_plane(&mut self.pl, core);
        let Some(_start) = self.start_point else {
            self.start_point = Some(ctx.world);
            if self.pl.work_plane.is_none() {
                self.pl.work_plane = locked_work_plane(&self.pl, ctx.world, &core.axes);
            }
            return;
        };
        // The plane may follow the far corner (`drawing_plane`), so it must
        // be judged on the point being clicked, not the last hover.
        self.hover_point = Some(ctx.world);
        let (anchor, far) = self.span(core, ctx.world);
        let (du, dv) = self.dimensions(core, anchor, far);
        if du.abs() < 1e-6 || dv.abs() < 1e-6 {
            // A side of zero: nothing is drawn.
            let why = self.degenerate_reason(core, ctx.world);
            core.flash(why);
            return;
        }
        let corners = self.corners(core, anchor, far);
        self.commit_rect(core, corners);
    }

    fn on_hover(&mut self, core: &mut Core, ctx: &Ctx) {
        note_plane(&mut self.pl, core);
        self.viewport_seen = true;
        // First: the plane follows a snapped far corner (`plane_points`), so
        // the square test below must see this point, not the last one.
        if let Some(start) = self.start_point {
            // `self.hover_point or self.start_point`: a null vector is falsy.
            let prev = truthy(self.hover_point).unwrap_or(start);
            self.note_live_side(core, prev, ctx.world);
        }
        self.hover_point = Some(ctx.world);
        let square_inference = match self.start_point {
            Some(s) => self.square_corner(core, s, ctx.world, false).1,
            None => false,
        };
        if ctx.mods & MOD_SHIFT == 0 {
            self.shift_square_lock = false;
        } else if square_inference {
            self.shift_square_lock = true;
        }
        self.wireframe_color = lock_color(&self.pl, &core.axes, core.h());
    }

    fn on_value(&mut self, core: &mut Core, value: &VcbValue) -> bool {
        let (Some(start), Some(hover)) = (self.start_point, self.hover_point) else { return false };
        let (w, h);
        match value {
            VcbValue::Num(v) => {
                if self.square_locked() {
                    w = *v;
                    h = *v;
                } else {
                    // One side already reads right: the number is the side
                    // being stretched, the other stays.
                    let (span_a, span_b) = self.span(core, hover);
                    let (dw, dh) = self.dimensions(core, span_a, span_b);
                    let (mut ww, mut hh) = (dw.abs(), dh.abs());
                    match self.live_side {
                        Some(Side::U) => ww = *v,
                        Some(Side::V) => hh = *v,
                        None => return false,
                    }
                    w = ww;
                    h = hh;
                }
            }
            VcbValue::Pair(a, b) => {
                w = *a;
                h = *b;
            }
            _ => return false,
        }
        if w <= 0.0 || h <= 0.0 {
            return false;
        }
        let (u, v) = self.axes(core);
        let (du, dv) = self.dimensions(core, start, hover);
        let su = if du < 0.0 { -1.0 } else { 1.0 }; // keep the side the cursor is heading to
        let sv = if dv < 0.0 { -1.0 } else { 1.0 };
        let (anchor, far);
        if self.from_center {
            // Typed sizes are the WHOLE width and height, centred here.
            let half = u.mul((su * w * 0.5) as f32).add(v.mul((sv * h * 0.5) as f32));
            anchor = start.sub(half);
            far = start.add(half);
        } else {
            anchor = start;
            far = start.add(u.mul((su * w) as f32)).add(v.mul((sv * h) as f32));
        }
        let corners = self.corners(core, anchor, far);
        self.commit_rect(core, corners);
        true
    }

    fn on_cancel(&mut self, _core: &mut Core) {
        self.reset();
    }

    fn on_key(&mut self, core: &mut Core, key: i32, _mods: u32) -> bool {
        if key == KEY_CONTROL {
            // Ctrl = corner <-> centre, like Move's Ctrl = copy (issue #39).
            self.toggle_center(core, true);
            self.ctrl_toggled = true;
            return true;
        }
        if key == KEY_DOWN {
            if let Some(start) = self.start_point {
                if self.arrow_square_lock {
                    self.arrow_square_lock = false;
                    return true;
                }
                if let Some(hover) = self.hover_point {
                    if self.square_corner(core, start, hover, false).1 {
                        self.arrow_square_lock = true;
                        return true;
                    }
                }
            }
        }
        if self.start_point.is_some() && plane_lock_axis(key).is_some() {
            return self.plane_lock_after_start(core, key);
        }
        plane_lock_key(self, core, key)
    }

    /// Toggled on the PRESS so the pencil's badge changes at once; a Ctrl that
    /// was part of a shortcut (Ctrl+Z...) takes it back on the release (#342).
    fn on_key_release(&mut self, core: &mut Core, key: i32) -> bool {
        if key != KEY_CONTROL || !self.ctrl_toggled {
            return false;
        }
        self.ctrl_toggled = false;
        if !core.ctrl_alone {
            self.toggle_center(core, false);
        }
        true
    }

    fn rubber_band_lines(&self, core: &Core) -> Vec<(V3, V3)> {
        let Some(hover) = self.hover_point else { return Vec::new() };
        if self.start_point.is_none() {
            return self.cursor_preview(core);
        }
        let (anchor, far) = self.span(core, hover);
        let is_square = self.square_corner(core, anchor, far, false).1;
        let c = self.corners(core, anchor, far);
        let mut lines = vec![(c[0], c[1]), (c[1], c[2]), (c[2], c[3]), (c[3], c[0])];
        // A diagonal across the square is the "Square" cue (preview only).
        if is_square {
            lines.push((c[0], c[2]));
        }
        lines
    }

    fn value_label(&self, core: &Core) -> Option<(String, V3)> {
        let (Some(_), Some(hover)) = (self.start_point, self.hover_point) else { return None };
        let (anchor, far) = self.span(core, hover);
        let is_square = self.square_corner(core, anchor, far, false).1;
        let (du, dv) = self.dimensions(core, anchor, far);
        let mut text = fmt_pair(du.abs(), dv.abs(), &core.units);
        if is_square {
            text.push_str("  (Cuadrado)");
        }
        let mid = anchor.add(far).mul(0.5f32);
        Some((text, mid))
    }

    fn fields(&self) -> Vec<(&'static str, Field)> {
        let mut f = vec![
            ("start_point", Field::point(self.start_point)),
            ("hover_point", Field::point(self.hover_point)),
            ("_from_center", Field::Bool(self.from_center)),
            ("_shift_square_lock", Field::Bool(self.shift_square_lock)),
            ("_arrow_square_lock", Field::Bool(self.arrow_square_lock)),
            ("_live_side", Field::string(self.live_side.map(|s| if s == Side::U { "u" } else { "v" }))),
            ("_ctrl_toggled", Field::Bool(self.ctrl_toggled)),
        ];
        f.extend(plane_fields(&self.pl, self.wireframe_color));
        f
    }
}

#[allow(dead_code)]
fn _hypot(a: f64, b: f64) -> f64 {
    py_hypot2(a, b)
}
