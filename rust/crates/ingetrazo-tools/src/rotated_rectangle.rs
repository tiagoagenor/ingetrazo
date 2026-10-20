// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! The Rotated Rectangle tool (`tools/rotated_rectangle.py`): a rectangle at
//! any angle IN ANY PLANE. Three clicks: the first corner (which captures the
//! plane: the arrow lock, else the face under the click), the second corner
//! (the base edge's direction and length, with a protractor at the first
//! corner: near its rim the direction snaps to 15 degree ticks, Shift holds
//! the direction), and the width with its angle around the base edge (a
//! second protractor square to the edge; Shift holds the angle). The value
//! box takes `length` or `length;angle`, then `width` or `width;angle`.

use ingetrazo_geom::bevel_profiles::py_hypot2;
use ingetrazo_math::V3;
use ingetrazo_ops::command::Command;
use ingetrazo_ops::draw::AddFaceCommand;
use ingetrazo_ops::edits::build_add_edges;
use ingetrazo_snap::types::Axis;
use ingetrazo_snap::{SnapKind, ToolState};

use crate::base::*;
use crate::host::Core;
use crate::units::fmt_pair;
use crate::vcb::VcbValue;

/// `tools.protractor.DISC_PX`: the protractor's radius on screen.
pub const DISC_PX: f64 = 60.0;
/// `tools.protractor.TICK_DEG`.
pub const TICK_DEG: f64 = 15.0;
/// The width below which there is no rectangle.
const MIN_WIDTH: f64 = 1e-6;
/// Degrees within which the angle sticks to flat / perpendicular.
const ANGLE_SNAP: f64 = 3.0;

/// Python's float `%` for a positive divisor.
fn py_mod(x: f64, y: f64) -> f64 {
    let m = x % y;
    if m == 0.0 {
        return 0.0f64.copysign(y);
    }
    if (y < 0.0) != (m < 0.0) {
        m + y
    } else {
        m
    }
}

/// `_degrees`: an angle typed in the value box (a metre document: unscaled).
fn degrees_of(value: f64) -> f64 {
    value
}

fn radians(d: f64) -> f64 {
    d * (std::f64::consts::PI / 180.0)
}

fn degrees(r: f64) -> f64 {
    r * (180.0 / std::f64::consts::PI)
}

pub struct RotatedRectangleTool {
    pub start_point: Option<V3>,
    pub base_point: Option<V3>,
    pub hover_point: Option<V3>,
    angle: f64,
    locked: Option<V3>,
    disc_r: f64,
    near_disc: bool,
    held_angle: Option<f64>,
    held_dir: Option<V3>,
    pl: PlaneState,
}

impl Default for RotatedRectangleTool {
    fn default() -> Self {
        RotatedRectangleTool::new()
    }
}

impl RotatedRectangleTool {
    pub fn new() -> RotatedRectangleTool {
        RotatedRectangleTool { start_point: None, base_point: None, hover_point: None, angle: 0.0, locked: None, disc_r: 1.0, near_disc: false, held_angle: None, held_dir: None, pl: PlaneState::default() }
    }

    fn reset(&mut self) {
        self.start_point = None;
        self.base_point = None;
        self.pl.work_plane = None;
        clear_plane_lock(&mut self.pl);
        self.angle = 0.0;
        self.locked = None;
        self.near_disc = false;
        self.held_angle = None;
        self.held_dir = None;
    }

    fn normal(&self, core: &Core) -> V3 {
        match self.pl.work_plane {
            Some((_, n)) => n.normalized(core.h()),
            None => V3(0.0, 0.0, 1.0),
        }
    }

    /// `_perp`: in-plane unit vector perpendicular to the base edge.
    fn perp(&self, core: &Core) -> V3 {
        let h = core.h();
        let e = self.base_point.expect("a base").sub(self.start_point.expect("a start"));
        if (e.length(h) as f64) < 1e-9 {
            return V3(0.0, 0.0, 0.0);
        }
        self.normal(core).cross(e.normalized(h)).normalized(h)
    }

    /// `_width_dir(angle)`.
    fn width_dir(&self, core: &Core, angle: Option<f64>) -> V3 {
        let h = core.h();
        let perp = self.perp(core);
        if (perp.length(h) as f64) < 1e-6 {
            return V3(0.0, 0.0, 0.0);
        }
        let a = radians(angle.unwrap_or(self.angle));
        perp.mul(a.cos() as f32).add(self.normal(core).mul(a.sin() as f32)).normalized(h)
    }

    /// `_width_and_angle(cursor, forced)`.
    fn width_and_angle(&self, core: &Core, cursor: V3, forced: Option<V3>) -> (f64, f64) {
        let h = core.h();
        let perp = self.perp(core);
        if (perp.length(h) as f64) < 1e-6 {
            return (0.0, 0.0);
        }
        let (start, base) = (self.start_point.expect("a start"), self.base_point.expect("a base"));
        let edge = base.sub(start).normalized(h);
        let mut d = cursor.sub(base);
        d = d.sub(edge.mul(d.dot(edge))); // off-edge part only
        if let Some(f) = forced {
            d = f.mul(d.dot(f));
        }
        let width = d.length(h) as f64;
        if width < 1e-9 {
            return (0.0, self.angle);
        }
        let mut angle = degrees((d.dot(self.normal(core)) as f64).atan2(d.dot(perp) as f64));
        if self.near_disc && forced.is_none() {
            angle = (angle / TICK_DEG).round_ties_even() * TICK_DEG; // the 15 degree ticks
        }
        for target in [-180.0, -90.0, 0.0, 90.0, 180.0] {
            if (angle - target).abs() <= ANGLE_SNAP {
                angle = target;
                break;
            }
        }
        (width, angle)
    }

    /// `locked_dir`: the direction the viewport's axis lock imposes on the
    /// width, or `None`.
    fn locked_dir(&self, core: &Core) -> Option<V3> {
        let h = core.h();
        let lock = core.axis_lock()?;
        let (start, base) = (self.start_point?, self.base_point?);
        let edge = base.sub(start);
        if (edge.length(h) as f64) < 1e-9 {
            return None;
        }
        let edge = edge.normalized(h);
        let axis = core.axes.axis(lock);
        let d = axis.sub(edge.mul(axis.dot(edge)));
        if (d.length(h) as f64) > 1e-6 {
            Some(d.normalized(h))
        } else {
            None
        }
    }

    /// `_corners(width, angle)`.
    fn corners(&self, core: &Core, width: f64, angle: Option<f64>) -> Vec<V3> {
        let h = core.h();
        let direction = self.width_dir(core, angle);
        if (direction.length(h) as f64) < 1e-6 {
            return Vec::new();
        }
        let off = direction.mul(width as f32);
        let (start, base) = (self.start_point.expect("a start"), self.base_point.expect("a base"));
        vec![start, base, base.add(off), start.add(off)]
    }

    /// `_capture_plane`: the arrow-key lock wins; otherwise the face under
    /// the click decides.
    fn capture_plane(&self, core: &mut Core, ctx: &Ctx) -> Option<(V3, V3)> {
        if let Some(l) = locked_work_plane(&self.pl, ctx.world, &core.axes) {
            return Some(l);
        }
        let (_p, normal) = core.pick_face_placement(ctx.screen.0, ctx.screen.1)?;
        if (normal.length_squared() as f64) < 0.5 {
            return None;
        }
        Some((ctx.world, normal))
    }

    /// `_width_cursor`: where the cursor is for the width, on the second
    /// protractor's plane.
    fn width_cursor(&self, core: &Core, ctx: &Ctx) -> V3 {
        let h = core.h();
        let point_snap = matches!(
            ctx.snap.kind,
            SnapKind::Endpoint | SnapKind::Midpoint | SnapKind::Intersection | SnapKind::OnEdge | SnapKind::Close | SnapKind::Origin | SnapKind::Center | SnapKind::OnLine
        );
        if point_snap {
            return ctx.world;
        }
        let Some((origin, direction)) = core.pixel_to_ray(ctx.screen.0, ctx.screen.1) else { return ctx.world };
        let (start, base) = (self.start_point.expect("a start"), self.base_point.expect("a base"));
        let n = base.sub(start).normalized(h);
        let denom = direction.dot(n) as f64;
        if denom.abs() < 0.02 {
            // The protractor seen edge-on: the drawing plane's answer is the
            // best there is.
            return ctx.world;
        }
        let t = (base.sub(origin).dot(n) as f64) / denom;
        if t <= 0.0 {
            return ctx.world;
        }
        origin.add(direction.mul(t as f32))
    }

    fn in_plane(&self, core: &Core, d: V3) -> V3 {
        let n = self.normal(core);
        d.sub(n.mul(d.dot(n)))
    }

    /// `_edge_angle(d)`: direction of the base edge in the plane, degrees
    /// from its first axis (0..360).
    fn edge_angle(&self, core: &Core, d: V3) -> f64 {
        let (u, v) = plane_axes(&core.axes, self.normal(core), core.h());
        py_mod(degrees((d.dot(v) as f64).atan2(d.dot(u) as f64)), 360.0)
    }

    fn dir_at(&self, core: &Core, deg: f64) -> V3 {
        let (u, v) = plane_axes(&core.axes, self.normal(core), core.h());
        let a = radians(deg);
        u.mul(a.cos() as f32).add(v.mul(a.sin() as f32)).normalized(core.h())
    }

    /// `_disc_metrics`: fixed screen size for the disc, and whether the
    /// cursor is near enough to snap to its ticks.
    fn disc_metrics(&mut self, core: &Core, ctx: &Ctx, centre: V3, axis: V3) {
        self.near_disc = false;
        let (u, _v) = plane_axes(&core.axes, axis, core.h());
        let (Some(p0), Some(p1)) = (core.world_to_pixel(centre), core.world_to_pixel(centre.add(u))) else { return };
        let px = py_hypot2(p1.0 - p0.0, p1.1 - p0.1);
        if px > 1e-6 {
            self.disc_r = DISC_PX / px;
        }
        self.near_disc = py_hypot2(ctx.screen.0 - p0.0, ctx.screen.1 - p0.1) <= DISC_PX * 1.25;
    }

    /// `_disc(centre, u, v)`: a protractor disc, rim and ticks every 15
    /// degrees, long ones every 90, zero on `u`.
    fn disc(&self, core: &Core, centre: V3, u: V3, v: V3) -> Vec<(V3, V3)> {
        let h = core.h();
        if (u.length(h) as f64) < 1e-9 || (v.length(h) as f64) < 1e-9 {
            return Vec::new();
        }
        let r = self.disc_r;
        let rim = |t: f64, k: f64| centre.add(u.mul(t.cos() as f32).add(v.mul(t.sin() as f32)).mul((r * k) as f32));
        let n = 48usize;
        let pts: Vec<V3> = (0..n).map(|k| rim(2.0 * std::f64::consts::PI * (k as f64) / (n as f64), 1.0)).collect();
        let mut segs: Vec<(V3, V3)> = (0..n).map(|k| (pts[k], pts[(k + 1) % n])).collect();
        for k in 0..((360.0 / TICK_DEG) as i64) {
            let t = radians(k as f64 * TICK_DEG);
            let inner = if py_mod(k as f64 * TICK_DEG, 90.0) == 0.0 { 0.75 } else { 0.86 };
            segs.push((rim(t, inner), rim(t, 1.0)));
        }
        segs
    }

    /// `_commit`.
    fn commit(&mut self, core: &mut Core, corners: Vec<V3>) {
        let plane_normal = match self.pl.work_plane {
            Some((_, n)) => Some(n),
            None => core.last_work_plane.map(|p| p.1),
        };
        let corners = face_the_plane(&corners, plane_normal);
        let segments: Vec<(V3, V3)> = (0..4).map(|i| (corners[i], corners[(i + 1) % 4])).collect();
        let extra: Vec<Box<dyn Command>> = vec![Box::new(AddFaceCommand::new(corners.clone(), true, None, None))];
        let cmd = build_add_edges(&core.scene, &core.cx, &segments, false, extra, None, true);
        core.exec(cmd);
        self.reset();
    }

    /// `_on_edge_value`: the base edge typed.
    fn on_edge_value(&mut self, core: &mut Core, value: &VcbValue) -> bool {
        let h = core.h();
        let start = self.start_point.expect("a start");
        let (mut length, mut direction);
        match value {
            VcbValue::Pair(l, a) => {
                length = *l;
                let angle = degrees_of(*a);
                direction = self.dir_at(core, angle);
            }
            VcbValue::Num(l) => {
                length = *l;
                let d = match self.hover_point {
                    Some(hp) => hp.sub(start),
                    None => V3(0.0, 0.0, 0.0),
                };
                direction = if (d.length(h) as f64) > 1e-9 { d.normalized(h) } else { self.dir_at(core, 0.0) };
            }
            _ => return false,
        }
        if length == 0.0 {
            return false;
        }
        if length < 0.0 {
            length = -length;
            direction = direction.neg();
        }
        self.base_point = Some(start.add(direction.mul(length as f32)));
        if (self.perp(core).length_squared() as f64) < 1e-12 {
            self.base_point = None;
            return false;
        }
        true
    }
}

impl Planar for RotatedRectangleTool {
    fn pl(&self) -> &PlaneState {
        &self.pl
    }
    fn pl_mut(&mut self) -> &mut PlaneState {
        &mut self.pl
    }
    /// The arrows answer before the first click (choosing the plane of the
    /// base edge); once the base is down they are the viewport's linear axis
    /// lock again, which the width of the third click reads.
    fn lock_after_first_point(&self) -> bool {
        false
    }
}

impl DrawTool for RotatedRectangleTool {
    fn name(&self) -> &'static str {
        "Rotated Rect"
    }

    fn snap_state(&self, _core: &Core) -> ToolState {
        // AxisMagnet on a PlaneLock; `magnet_on()` = no base edge yet.
        ToolState {
            magnetic_axis_deg: if self.base_point.is_none() { Some(MAGNET_DEG) } else { None },
            screen_axis_px: None,
            magnet_off_plane: false,
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
        if self.base_point.is_none() { "Length; angle" } else { "Width; angle" }.into()
    }

    fn on_activate(&mut self, _core: &mut Core) {
        self.reset();
    }

    fn on_deactivate(&mut self, _core: &mut Core) {
        self.reset();
        self.hover_point = None;
    }

    fn on_click(&mut self, core: &mut Core, ctx: &Ctx) {
        let h = core.h();
        let Some(start) = self.start_point else {
            // BEFORE start_point is set: the plane comes from the arrow-key
            // lock if there is one, else from the face under this very click.
            self.pl.work_plane = self.capture_plane(core, ctx);
            self.start_point = Some(ctx.world);
            return;
        };
        if self.base_point.is_none() {
            if (ctx.world.sub(start).length(h) as f64) < 1e-6 {
                return;
            }
            // With Shift holding the direction the corner is where the
            // preview shows it - on the held line, not under the cursor.
            self.base_point = Some(match (self.held_dir, self.hover_point) {
                (Some(_), Some(hp)) => hp,
                _ => ctx.world,
            });
            self.held_dir = None;
            if (self.perp(core).length_squared() as f64) < 1e-12 {
                // The base edge is perpendicular to the drawing plane, so
                // there is no width direction left. Say it.
                self.base_point = None;
                core.flash("That edge is perpendicular to the drawing plane \u{2014} lock another one with the arrow keys, or start on the face you want to draw on".into());
            }
            return;
        }
        let locked = self.locked_dir(core);
        let cursor = self.width_cursor(core, ctx);
        let (mut width, angle) = self.width_and_angle(core, cursor, locked);
        self.angle = angle;
        if locked.is_some() && width < MIN_WIDTH {
            if let Some(hp) = self.hover_point {
                // With an axis locked the snap engine can still hand over, AT
                // the click, a point off that axis: what the user accepted
                // by clicking is what they were seeing, the hover's point.
                let r = self.width_and_angle(core, hp, locked);
                width = r.0;
                self.angle = r.1;
            }
        }
        if width < MIN_WIDTH {
            core.flash("The width is zero \u{2014} move away from the base edge, or type the width (and the angle after a comma)".into());
            return;
        }
        let corners = self.corners(core, width, None);
        if !corners.is_empty() {
            self.commit(core, corners);
        }
    }

    fn on_hover(&mut self, core: &mut Core, ctx: &Ctx) {
        let h = core.h();
        self.hover_point = Some(if self.base_point.is_some() { self.width_cursor(core, ctx) } else { ctx.world });
        // Cached for the preview and the label, which get no viewport.
        self.locked = self.locked_dir(core);
        if let (Some(start), None) = (self.start_point, self.base_point) {
            let n = self.normal(core);
            self.disc_metrics(core, ctx, start, n);
            if self.near_disc {
                // Near the first protractor the edge turns in 15 degree steps.
                let d = self.in_plane(core, ctx.world.sub(start));
                if (d.length(h) as f64) > 1e-9 {
                    let deg = (self.edge_angle(core, d) / TICK_DEG).round_ties_even() * TICK_DEG;
                    self.hover_point = Some(start.add(self.dir_at(core, deg).mul(d.length(h))));
                }
            }
            if ctx.mods & MOD_SHIFT == 0 {
                self.held_dir = None;
            } else {
                let d = self.in_plane(core, self.hover_point.expect("hover").sub(start));
                if self.held_dir.is_none() && (d.length(h) as f64) > 1e-9 {
                    self.held_dir = Some(d.normalized(h)); // Shift takes it
                }
                if let Some(hd) = self.held_dir {
                    // Only the length follows the cursor now - along the held
                    // line, either way along it.
                    let along = ctx.world.sub(start).dot(hd);
                    self.hover_point = Some(start.add(hd.mul(along)));
                }
            }
        }
        if let Some(base) = self.base_point {
            let start = self.start_point.expect("a start");
            let edge = base.sub(start).normalized(h);
            self.disc_metrics(core, ctx, base, edge);
            let shift = ctx.mods & MOD_SHIFT != 0;
            if !shift {
                self.held_angle = None;
            } else if self.held_angle.is_none() {
                self.held_angle = Some(self.angle); // Shift holds the angle
            }
            if let Some(ha) = self.held_angle {
                self.locked = Some(self.width_dir(core, Some(ha)));
            }
            // The angle being SHOWN is kept: typing the width in the box
            // builds the rectangle with it.
            let (w, angulo) = self.width_and_angle(core, self.hover_point.expect("hover"), self.locked);
            if w > MIN_WIDTH {
                self.angle = angulo;
            }
        }
    }

    fn on_value(&mut self, core: &mut Core, value: &VcbValue) -> bool {
        if self.start_point.is_none() {
            return false;
        }
        if self.base_point.is_none() {
            return self.on_edge_value(core, value);
        }
        if self.hover_point.is_none() {
            return false;
        }
        let corners = match value {
            VcbValue::Pair(width, angle) => {
                if *width <= 0.0 {
                    return false;
                }
                self.angle = degrees_of(*angle);
                self.corners(core, *width, None)
            }
            VcbValue::Num(v) => {
                if *v == 0.0 {
                    return false;
                }
                // The SIDE already travels inside the angle the hover keeps.
                self.corners(core, *v, None)
            }
            _ => return false,
        };
        if !corners.is_empty() {
            self.commit(core, corners);
        }
        true
    }

    fn on_cancel(&mut self, _core: &mut Core) {
        self.reset();
    }

    fn on_key(&mut self, core: &mut Core, key: i32, _mods: u32) -> bool {
        plane_lock_key(self, core, key)
    }

    fn rubber_band_lines(&self, core: &Core) -> Vec<(V3, V3)> {
        let (Some(start), Some(hover)) = (self.start_point, self.hover_point) else { return Vec::new() };
        let Some(base) = self.base_point else {
            // Drawing the base edge, over the first protractor.
            let (u, v) = plane_axes(&core.axes, self.normal(core), core.h());
            let mut out = vec![(start, hover)];
            out.extend(self.disc(core, start, u, v));
            return out;
        };
        let (width, angle) = self.width_and_angle(core, hover, self.locked);
        let c = self.corners(core, width, Some(angle));
        // The second protractor: square to the base edge at its end.
        let disc = self.disc(core, base, self.perp(core), self.normal(core));
        if c.is_empty() {
            let mut out = vec![(start, base)];
            out.extend(disc);
            return out;
        }
        let mut out: Vec<(V3, V3)> = (0..4).map(|i| (c[i], c[(i + 1) % 4])).collect();
        out.extend(disc);
        out
    }

    fn value_label(&self, core: &Core) -> Option<(String, V3)> {
        let h = core.h();
        let (start, hover) = (self.start_point?, self.hover_point?);
        let Some(base) = self.base_point else {
            let d = hover.sub(start);
            let mid = start.add(hover).mul(0.5f32);
            let mut texto = core.fmt_len(d.length(h) as f64);
            let flat = self.in_plane(core, d);
            if (flat.length(h) as f64) > 1e-9 {
                texto.push_str(&format!("   {:.0}\u{b0}", self.edge_angle(core, flat)));
            }
            return Some((texto, mid));
        };
        let (w, angle) = self.width_and_angle(core, hover, self.locked);
        let length = base.sub(start).length(h) as f64;
        let c = self.corners(core, w, Some(angle));
        let mid = if c.is_empty() { base } else { start.add(c[2]).mul(0.5f32) };
        let mut texto = fmt_pair(length, w.abs(), &core.units);
        if angle.abs() > 0.05 {
            texto.push_str(&format!("   {:.0}\u{b0}", angle));
        }
        Some((texto, mid))
    }

    fn fields(&self) -> Vec<(&'static str, Field)> {
        let mut f = vec![
            ("start_point", Field::point(self.start_point)),
            ("base_point", Field::point(self.base_point)),
            ("hover_point", Field::point(self.hover_point)),
            ("angle", Field::Float(self.angle)),
            ("_locked", Field::point(self.locked)),
            ("_disc_r", Field::Float(self.disc_r)),
            ("_near_disc", Field::Bool(self.near_disc)),
            ("_held_angle", Field::float(self.held_angle)),
            ("_held_dir", Field::point(self.held_dir)),
        ];
        f.extend(plane_fields(&self.pl, None));
        f
    }
}

#[allow(dead_code)]
fn _axis_unused(_: Axis) {}
