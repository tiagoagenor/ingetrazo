// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! The Protractor tool (`tools/protractor.py`, Shift+H) and the shared
//! protractor mechanics ([`ProtractorBase`], which Rotate (Q) shows too).
//!
//! * Before the first click the disc follows the cursor, aligned to the face
//!   underneath (empty ground measures in plan) and coloured by the axis it
//!   rotates about. Arrow keys lock the plane to an axis (Right=red, Left=
//!   green, Up=blue; the same arrow again releases); Shift freezes the plane;
//!   it fixes once the centre is placed.
//! * The disc keeps a fixed SCREEN size with ticks every 15 degrees; near the
//!   disc the cursor snaps to the ticks, farther out the angle is free at 0.1
//!   degree. A cursor held by an inference (anything but none/on_face)
//!   reaches that point exactly.
//! * Click-drag from the centre tilts the instrument along the drag, with the
//!   disc tilting live past 8 px.
//! * The Protractor leaves ANGLED guide lines (centre, base arm, sweep and
//!   click); Ctrl toggles measure-only; the angle stays hot so a typed value
//!   re-aims the guide just created.
//!
//! Differences from Python (reported): none known. The drawing axes
//! (`core.axes`) are the world's: the tool is exercised with no group open.
//! `ProtractorBase` keeps every field public so Move's rotation-grip entry
//! (`MoveTool._start_grip_rotation` sets `_custom_axis`, `start_point`...) can
//! drive a Rotate tool.

use ingetrazo_formats::pyjson::Value;
use ingetrazo_geom::facegeom::plane_axes;
use ingetrazo_geom::sweep::py_hypot2;
use ingetrazo_math::{Hypot, V3};
use ingetrazo_model::guide::Guide;
use ingetrazo_model::units::fmt_angle_deg;
use ingetrazo_ops::annotate::{AddGuide, ChangeGuide};
use ingetrazo_snap::SnapKind;

use crate::modify_base::{keys, Env, FaceRef, ModifyTool, Mods, Probe, ToolCx, Vcb};

/// The disc keeps a fixed SCREEN size; ticks every 15 degrees.
pub const DISC_PX: f64 = 60.0;
pub const TICK_DEG: f64 = 15.0;
/// How far the cursor must travel from the centre for the press to read as a
/// DRAG that tilts the instrument.
pub const AXIS_DRAG_PX: f64 = 8.0;
const OFF_AXIS_RGBA: [f64; 4] = [0.24, 0.27, 0.32, 1.0];

/// `AXES[name]` of the world's drawing axes.
pub fn axis_vec(c: char) -> V3 {
    match c {
        'x' => V3(1.0, 0.0, 0.0),
        'y' => V3(0.0, 1.0, 0.0),
        _ => V3(0.0, 0.0, 1.0),
    }
}

fn axis_rgba(c: char) -> [f64; 4] {
    match c {
        'x' => [0.86, 0.22, 0.27, 1.0],
        'y' => [0.16, 0.62, 0.36, 1.0],
        _ => [0.20, 0.40, 0.78, 1.0],
    }
}

/// `round(x, 1)` of a Python float.
fn round1(x: f64) -> f64 {
    if !x.is_finite() {
        return x;
    }
    format!("{x:.1}").parse().unwrap_or(x)
}

/// A point as float32 bits, the way the fixtures record it.
pub fn v3_value(p: V3) -> Value {
    let b = |x: f32| Value::Str(format!("{:08x}", x.to_bits()));
    Value::Array(vec![b(p.0), b(p.1), b(p.2)])
}

/// A Python float as float64 bits.
pub fn f64_value(x: f64) -> Value {
    Value::Str(format!("{:016x}", x.to_bits()))
}

pub fn opt_v3_value(p: Option<V3>) -> Value {
    p.map(v3_value).unwrap_or(Value::Null)
}

/// Shared protractor state + behaviour.
#[derive(Clone, Debug)]
pub struct ProtractorBase {
    /// The protractor centre.
    pub start_point: Option<V3>,
    /// The base (0 degree) direction.
    pub ref_point: Option<V3>,
    pub hover_point: Option<V3>,
    pub work_plane: Option<(V3, V3)>,
    /// Arrow-key plane lock: `'x'`, `'y'` or `'z'`.
    pub axis_pick: Option<char>,
    pub shift_normal: Option<V3>,
    pub custom_axis: Option<V3>,
    pub axis_drag_armed: bool,
    pub axis_drag_live: Option<V3>,
    pub disc_r: f64,
    pub snap_ticks: bool,
    pub exact_snap: bool,
    /// `hasattr(type(self), "_guides")`: the tool leaves a guide behind.
    pub has_guides: bool,
    pub guides: bool,
    pub ctrl_toggled: bool,
}

impl ProtractorBase {
    pub fn new(has_guides: bool) -> ProtractorBase {
        ProtractorBase {
            start_point: None,
            ref_point: None,
            hover_point: None,
            work_plane: None,
            axis_pick: None,
            shift_normal: None,
            custom_axis: None,
            axis_drag_armed: false,
            axis_drag_live: None,
            disc_r: 1.0,
            snap_ticks: false,
            exact_snap: false,
            has_guides,
            guides: true,
            ctrl_toggled: false,
        }
    }

    // ---- click-drag axis -----------------------------------------------------

    /// `_track_axis_drag`.
    pub fn track_axis_drag(&mut self, env: &Env) {
        if !self.axis_drag_armed {
            self.axis_drag_live = None;
            return;
        }
        let (Some(hover), Some(start)) = (self.hover_point, self.start_point) else {
            self.axis_drag_live = None;
            return;
        };
        let (p0, p1) = (env.vp.world_to_pixel(start), env.vp.world_to_pixel(hover));
        match (p0, p1) {
            (Some(p0), Some(p1)) if py_hypot2(p1.0 - p0.0, p1.1 - p0.1) > AXIS_DRAG_PX => {}
            _ => {
                self.axis_drag_live = None;
                return;
            }
        }
        let h = env.cx.hypot;
        let d = hover.sub(start);
        self.axis_drag_live = if (d.length(h) as f64) > 1e-9 { Some(d.normalized(h)) } else { None };
    }

    /// `_release_axis_drag`: true when an axis was set.
    pub fn release_axis_drag(&mut self, env: &Env) -> bool {
        if !self.axis_drag_armed {
            return false;
        }
        self.track_axis_drag(env);
        self.axis_drag_armed = false;
        let axis = self.axis_drag_live.take();
        match axis {
            None => false,
            Some(a) => {
                self.custom_axis = Some(a);
                true
            }
        }
    }

    // ---- keyboard ------------------------------------------------------------------

    /// `_toggle_guides`.
    pub fn toggle_guides(&mut self, env: &mut Env, say: bool) {
        self.guides = !self.guides;
        if say {
            env.flash(if self.guides { "Create guides: on" } else { "Create guides: off \u{2014} measure only" });
        }
    }

    /// `ProtractorBase.on_key_release`.
    pub fn on_key_release(&mut self, env: &mut Env, key: i32) -> bool {
        if key != keys::CONTROL || !self.has_guides || !self.ctrl_toggled {
            return false;
        }
        self.ctrl_toggled = false;
        if !env.vp.ctrl_tapped() {
            self.toggle_guides(env, false);
        }
        true
    }

    /// `ProtractorBase.on_key`.
    pub fn on_key(&mut self, env: &mut Env, key: i32) -> bool {
        if key == keys::CONTROL && self.has_guides {
            self.toggle_guides(env, true);
            self.ctrl_toggled = true;
            return true;
        }
        let axis = match key {
            keys::RIGHT => 'x',
            keys::LEFT => 'y',
            keys::UP => 'z',
            _ => return false,
        };
        self.axis_pick = if self.axis_pick == Some(axis) { None } else { Some(axis) };
        if self.axis_pick.is_some() && self.start_point.is_none() {
            let anchor = self.hover_point.unwrap_or(V3(0.0, 0.0, 0.0));
            self.work_plane = Some((anchor, axis_vec(axis)));
        }
        true
    }

    // ---- plane / colour --------------------------------------------------------------

    /// `wireframe_color`.
    pub fn wireframe_color(&self, h: Hypot) -> [f64; 4] {
        let n = self.axis(h);
        for c in ['x', 'y', 'z'] {
            if (n.dot(axis_vec(c)) as f64).abs() > 0.999 {
                return axis_rgba(c);
            }
        }
        OFF_AXIS_RGBA
    }

    /// `_axis`.
    pub fn axis(&self, h: Hypot) -> V3 {
        if let Some(a) = self.custom_axis {
            return a;
        }
        if let Some(a) = self.shift_normal {
            return a;
        }
        if let Some(c) = self.axis_pick {
            return axis_vec(c);
        }
        if let Some((_, n)) = self.work_plane {
            return n.normalized(h);
        }
        V3(0.0, 0.0, 1.0)
    }

    /// `_infer_plane`.
    pub fn infer_plane(&mut self, env: &Env, cx: &ToolCx) {
        let h = env.cx.hypot;
        let shift = cx.mods.0 & Mods::SHIFT != 0;
        if !shift {
            self.shift_normal = None;
        }
        if self.start_point.is_some() {
            return;
        }
        if shift {
            if self.shift_normal.is_none() {
                self.shift_normal = Some(self.axis(h));
            }
            return;
        }
        if let Some(c) = self.axis_pick {
            self.work_plane = Some((cx.world, axis_vec(c)));
            return;
        }
        // The face's LOCAL normal (a placed face is the prototype's), as
        // `face.normal().normalized()` reads it.
        let hit = env.vp.pick_face_any(env.scene, env.cx, cx.screen.0, cx.screen.1);
        self.work_plane = hit.and_then(|hit| {
            let normal = match &hit.face {
                FaceRef::Loose(f) => env.cx.mesh(env.scene).face_normal(*f, h),
                FaceRef::Placed { placement, index } => {
                    let g = ingetrazo_ops::group::group_at(&env.scene.groups, placement)?;
                    let mesh = env.scene.mesh(g.mesh);
                    mesh.face_normal(mesh.faces[*index], h)
                }
            };
            Some((cx.world, normal.normalized(h)))
        });
    }

    // ---- screen metrics / snapping ------------------------------------------------------

    /// `_update_screen_metrics`.
    pub fn update_screen_metrics(&mut self, env: &Env, cx: &ToolCx) {
        let h = env.cx.hypot;
        // NOT `start or hover`: a zero vector (the origin!) is falsy in Python.
        let Some(centre) = self.start_point.or(self.hover_point) else { return };
        let (u, _v) = plane_axes(self.axis(h), h);
        let (Some(p0), Some(p1)) = (env.vp.world_to_pixel(centre), env.vp.world_to_pixel(centre.add(u))) else { return };
        let px_per_unit = py_hypot2(p1.0 - p0.0, p1.1 - p0.1);
        if px_per_unit > 1e-6 {
            self.disc_r = DISC_PX / px_per_unit;
        }
        if self.start_point.is_some() {
            let dist = py_hypot2(cx.screen.0 - p0.0, cx.screen.1 - p0.1);
            self.snap_ticks = dist <= DISC_PX * 1.25;
        } else {
            self.snap_ticks = false;
        }
    }

    // ---- angles -------------------------------------------------------------------------

    /// `_angle_to`.
    pub fn angle_to(&self, point: V3, h: Hypot) -> Option<f64> {
        let (u, v) = plane_axes(self.axis(h), h);
        let start = self.start_point.expect("start point");
        let a = self.ref_point.expect("reference point").sub(start);
        let b = point.sub(start);
        let a2 = (a.dot(u) as f64, a.dot(v) as f64);
        let b2 = (b.dot(u) as f64, b.dot(v) as f64);
        if py_hypot2(a2.0, a2.1) < 1e-9 || py_hypot2(b2.0, b2.1) < 1e-9 {
            return None;
        }
        let mut deg = (b2.1.atan2(b2.0) - a2.1.atan2(a2.0)).to_degrees();
        while deg <= -180.0 {
            deg += 360.0;
        }
        while deg > 180.0 {
            deg -= 360.0;
        }
        Some(deg)
    }

    /// `_display_deg`: snapped to the 15 degree ticks near the disc, 0.1
    /// degree farther out.
    pub fn display_deg(&self, point: V3, h: Hypot) -> Option<f64> {
        let mut deg = self.angle_to(point, h)?;
        if self.snap_ticks {
            deg = (deg / TICK_DEG).round_ties_even() * TICK_DEG;
            if deg <= -180.0 {
                deg += 360.0;
            }
        }
        Some(round1(deg))
    }

    /// `_note_snap`: the cursor is held by an inference.
    pub fn note_snap(&mut self, cx: &ToolCx) {
        let kind = cx.snap_kind();
        self.exact_snap = !matches!(kind, SnapKind::None | SnapKind::OnFace);
    }

    /// `_commit_deg`: the angle to APPLY.
    pub fn commit_deg(&self, point: V3, h: Hypot) -> Option<f64> {
        if self.exact_snap && !self.snap_ticks {
            return self.angle_to(point, h);
        }
        self.display_deg(point, h)
    }

    /// `_direction_at`: unit direction of the base arm rotated by `deg`.
    pub fn direction_at(&self, deg: f64, h: Hypot) -> V3 {
        let (u, v) = plane_axes(self.axis(h), h);
        let a = self.ref_point.expect("ref").sub(self.start_point.expect("start"));
        let a0 = (a.dot(v) as f64).atan2(a.dot(u) as f64);
        let t = a0 + deg.to_radians();
        u.mul(t.cos() as f32).add(v.mul(t.sin() as f32)).normalized(h)
    }

    /// `_protractor_disc`: the fixed-screen-size disc with ticks, zero on the
    /// base arm once set. `axis` borrows a different one (the tilt preview).
    pub fn protractor_disc(&self, centre: V3, axis: Option<V3>, h: Hypot) -> Vec<(V3, V3)> {
        let (u, v) = plane_axes(axis.unwrap_or_else(|| self.axis(h)), h);
        let r = self.disc_r;
        let mut base = 0.0;
        if let (Some(s), Some(rf)) = (self.start_point, self.ref_point) {
            let a = rf.sub(s);
            base = (a.dot(v) as f64).atan2(a.dot(u) as f64);
        }
        let rim = |t: f64, k: f64| centre.add(u.mul(t.cos() as f32).add(v.mul(t.sin() as f32)).mul((r * k) as f32));
        let n = 48usize;
        let pts: Vec<V3> = (0..n).map(|k| rim(base + 2.0 * std::f64::consts::PI * k as f64 / n as f64, 1.0)).collect();
        let mut segments: Vec<(V3, V3)> = (0..n).map(|k| (pts[k], pts[(k + 1) % n])).collect();
        for k in 0..(360.0 / TICK_DEG) as i64 {
            let deg = k as f64 * TICK_DEG;
            let t = base + deg.to_radians();
            let inner = if deg % 90.0 == 0.0 { 0.75 } else { 0.86 };
            segments.push((rim(t, inner), rim(t, 1.0)));
        }
        segments
    }

    /// `_reset_protractor`.
    pub fn reset_protractor(&mut self) {
        self.start_point = None;
        self.ref_point = None;
        self.work_plane = None;
        self.shift_normal = None;
        self.custom_axis = None;
        self.axis_drag_live = None;
        self.axis_drag_armed = false;
        self.snap_ticks = false;
        self.exact_snap = false;
    }

    /// `(centre, tilt)` -> the disc and the arms, as `rubber_band_lines` of
    /// both tools start.
    pub fn disc_and_arms(&self, h: Hypot) -> Option<Vec<(V3, V3)>> {
        let centre = self.start_point.or(self.hover_point)?;
        let tilt = if self.axis_drag_armed { self.axis_drag_live } else { None };
        let mut segments = self.protractor_disc(centre, tilt, h);
        if let (Some(s), Some(hv)) = (self.start_point, self.hover_point) {
            segments.push((s, hv));
        }
        Some(segments)
    }

    /// The shared part of `extra_probe`.
    pub fn probe_fields(&self, h: Hypot) -> Vec<(String, Value)> {
        let s = |k: &str, v: Value| (k.to_string(), v);
        let _ = h;
        vec![
            s("ref", opt_v3_value(self.ref_point)),
            s(
                "work_plane",
                match self.work_plane {
                    None => Value::Null,
                    Some((p, n)) => Value::Array(vec![v3_value(p), v3_value(n)]),
                },
            ),
            s("axis_pick", self.axis_pick.map(|c| Value::Str(c.to_string())).unwrap_or(Value::Null)),
            s("shift_normal", opt_v3_value(self.shift_normal)),
            s("custom_axis", opt_v3_value(self.custom_axis)),
            s("drag_armed", Value::Bool(self.axis_drag_armed)),
            s("drag_live", opt_v3_value(self.axis_drag_live)),
            s("disc_r", f64_value(self.disc_r)),
            s("snap_ticks", Value::Bool(self.snap_ticks)),
            s("exact_snap", Value::Bool(self.exact_snap)),
            s("ctrl_toggled", Value::Bool(self.ctrl_toggled)),
        ]
    }
}

/// `ProtractorTool._last`: the angle stays hot.
#[derive(Clone, Debug)]
struct LastGuide {
    guide: Option<u32>,
    u: V3,
    v: V3,
    a0: f64,
    sign: f64,
}

pub struct ProtractorTool {
    pub base: ProtractorBase,
    last: Option<LastGuide>,
}

impl Default for ProtractorTool {
    fn default() -> Self {
        ProtractorTool::new()
    }
}

impl ProtractorTool {
    pub fn new() -> ProtractorTool {
        ProtractorTool { base: ProtractorBase::new(true), last: None }
    }

    /// `status_clause`.
    pub fn status_clause(&self) -> String {
        if self.base.guides {
            "Ctrl = [guide] / measure".to_string()
        } else {
            "Ctrl = guide / [measure]".to_string()
        }
    }

    fn reset(&mut self) {
        self.base.reset_protractor();
    }

    /// `_commit`.
    fn commit(&mut self, env: &mut Env, deg: f64) {
        let h = env.cx.hypot;
        let b = &self.base;
        let d = b.direction_at(deg, h);
        if !b.guides {
            // Measure only: report the angle and leave nothing behind.
            let txt = format!("Angle: {}", fmt_angle_deg(deg, true, env.units));
            env.flash_for(txt, 4000);
            self.reset();
            return;
        }
        let start = b.start_point.expect("start");
        let guide = Guide::new(start, Some(d), None, h);
        if !guide.is_finite() || !guide.is_line() {
            self.reset(); // a degenerate angle: leave nothing (#723)
            return;
        }
        let before = env.scene.guides.len();
        env.execute(Box::new(AddGuide::new(guide)));
        let id = if env.scene.guides.len() > before { env.scene.guides.id_at(env.scene.guides.len() - 1) } else { None };
        // The tool resets for the next measurement, but the angle stays hot.
        let (u, v) = plane_axes(self.base.axis(h), h);
        let a = self.base.ref_point.expect("ref").sub(start);
        self.last = Some(LastGuide { guide: id, u, v, a0: (a.dot(v) as f64).atan2(a.dot(u) as f64), sign: if deg < 0.0 { -1.0 } else { 1.0 } });
        self.reset();
    }

    pub fn rubber_band_lines(&self, h: Hypot) -> Vec<(V3, V3)> {
        let Some(mut segments) = self.base.disc_and_arms(h) else { return Vec::new() };
        if let (Some(s), Some(_), Some(r)) = (self.base.start_point, self.base.hover_point, self.base.ref_point) {
            segments.push((s, r));
        }
        segments
    }

    pub fn guide_preview_lines(&self, h: Hypot) -> Vec<(V3, V3)> {
        let b = &self.base;
        let (true, Some(start), Some(_), Some(hover)) = (b.guides, b.start_point, b.ref_point, b.hover_point) else { return Vec::new() };
        let Some(deg) = b.commit_deg(hover, h) else { return Vec::new() };
        vec![Guide::new(start, Some(b.direction_at(deg, h)), None, h).segment()]
    }

    pub fn value_label(&self, env: &Env) -> Option<(String, V3)> {
        let (Some(_), Some(hover)) = (self.base.ref_point, self.base.hover_point) else { return None };
        let deg = self.base.display_deg(hover, env.cx.hypot)?;
        Some((fmt_angle_deg(deg, true, env.units), hover))
    }
}

impl ModifyTool for ProtractorTool {
    fn on_activate(&mut self, _env: &mut Env) {
        self.base.guides = true;
        self.reset();
        self.last = None;
        self.base.axis_pick = None;
    }

    fn on_deactivate(&mut self, _env: &mut Env) {
        self.reset();
        self.last = None;
        self.base.axis_pick = None;
        self.base.hover_point = None;
    }

    fn on_click(&mut self, env: &mut Env, cx: &ToolCx) {
        self.last = None; // a click ends the retype window
        let h = env.cx.hypot;
        let Some(start) = self.base.start_point else {
            self.base.start_point = Some(cx.world);
            self.base.axis_drag_armed = true; // a DRAG from here tilts the disc
            return;
        };
        if self.base.ref_point.is_none() {
            if (cx.world.sub(start).length(h) as f64) < 1e-6 {
                return;
            }
            self.base.ref_point = Some(cx.world);
            return;
        }
        self.base.note_snap(cx);
        if let Some(deg) = self.base.commit_deg(cx.world, h) {
            self.commit(env, deg);
        }
    }

    fn on_hover(&mut self, env: &mut Env, cx: &ToolCx) {
        self.base.hover_point = Some(cx.world);
        self.base.track_axis_drag(env);
        self.base.infer_plane(env, cx);
        self.base.update_screen_metrics(env, cx);
        self.base.note_snap(cx);
    }

    fn on_release(&mut self, env: &mut Env) {
        if self.base.release_axis_drag(env) {
            env.flash("Protractor axis set along the drag");
        }
    }

    fn on_key(&mut self, env: &mut Env, key: i32, _mods: Mods) -> bool {
        self.base.on_key(env, key)
    }

    fn on_key_release(&mut self, env: &mut Env, key: i32) -> bool {
        self.base.on_key_release(env, key)
    }

    fn on_value(&mut self, env: &mut Env, value: &Vcb) -> bool {
        let Vcb::Num(value) = value else { return false };
        let value = *value;
        let h = env.cx.hypot;
        if self.base.ref_point.is_some() {
            // Mid-flow: the typed angle turns the way the cursor is sweeping.
            let mut sign = 1.0;
            if let Some(hover) = self.base.hover_point {
                if let Some(cur) = self.base.angle_to(hover, h) {
                    if cur < 0.0 {
                        sign = -1.0;
                    }
                }
            }
            self.commit(env, sign * value.abs());
            return true;
        }
        let Some(last) = self.last.clone() else { return false };
        // Hot retype: re-aim the guide just created.
        let Some(id) = last.guide.filter(|&g| env.scene.guides.contains_id(g)) else {
            self.last = None;
            return false;
        };
        let side = last.sign * if value >= 0.0 { 1.0 } else { -1.0 };
        let t = last.a0 + (side * value.abs()).to_radians();
        let d = last.u.mul(t.cos() as f32).add(last.v.mul(t.sin() as f32));
        let cmd = ChangeGuide::new(env.scene, id, d);
        env.execute(Box::new(cmd));
        true
    }

    fn on_cancel(&mut self, _env: &mut Env) {
        self.reset();
        self.last = None;
        self.base.axis_pick = None; // Esc and the reload both let it go (#48)
    }

    fn probe(&self, env: &Env) -> Probe {
        let h = env.cx.hypot;
        let mut p = Probe::default();
        p.rb = Some(self.rubber_band_lines(h));
        p.gl = Some(self.guide_preview_lines(h));
        p.vl = Some(self.value_label(env));
        p.sc = Some(self.status_clause());
        p.plus = Some(self.base.guides);
        p.wf = Some(self.base.wireframe_color(h));
        p.cap = Some("Angle".to_string());
        p.unitless = Some(true);
        p.start = self.base.start_point;
        p.hover = self.base.hover_point;
        let mut x = self.base.probe_fields(h);
        let s = |k: &str, v: Value| (k.to_string(), v);
        x.push(s("guides", Value::Bool(self.base.guides)));
        x.push(s("last", Value::Bool(self.last.is_some())));
        x.push(s("last_sign", self.last.as_ref().map(|l| f64_value(l.sign)).unwrap_or(Value::Null)));
        x.push(s("last_a0", self.last.as_ref().map(|l| f64_value(l.a0)).unwrap_or(Value::Null)));
        x.push(s("last_u", opt_v3_value(self.last.as_ref().map(|l| l.u))));
        x.push(s("last_v", opt_v3_value(self.last.as_ref().map(|l| l.v))));
        x.push(s(
            "last_live",
            Value::Bool(self.last.as_ref().and_then(|l| l.guide).is_some_and(|g| env.scene.guides.contains_id(g))),
        ));
        p.x = Some(Value::Object(x));
        p
    }
}
