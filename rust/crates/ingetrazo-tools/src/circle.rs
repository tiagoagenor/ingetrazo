// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! The Circle and Polygon tools (`tools/circle.py`): a centre click, then a
//! radius. Both draw a regular N-gon on the work plane (a circle is just a
//! many-sided one); one vertex points toward the cursor, so a hexagon's
//! orientation follows the mouse. The side count is typed before the centre
//! (`24` Enter) or at any moment (`24s`); right after a shape is drawn, `Ns`
//! rebuilds that shape in place (one undo step).

use ingetrazo_geom::bevel_profiles::py_hypot2;
use ingetrazo_math::V3;
use ingetrazo_ops::arrangement::coplanar_plane;
use ingetrazo_ops::command::Command;
use ingetrazo_ops::draw::{AddFaceCommand, TagCurveCommand};
use ingetrazo_ops::edits::build_add_edges;
use ingetrazo_ops::rebuild::{RebuildPlanarFacesCommand, RebuildPlaneFacesCommand};
use ingetrazo_snap::types::Axis;
use ingetrazo_snap::ToolState;

use crate::base::*;
use crate::host::Core;
use crate::vcb::VcbValue;

/// `flat_drawing`: whether the whole drawing (existing mesh + the new loop)
/// shares one plane - the gate for the classic planar arrangement.
pub fn flat_drawing(core: &Core, new_points: &[V3]) -> bool {
    let m = core.mesh();
    let mut verts: Vec<V3> = m.vertices.iter().map(|&v| m.position(v)).collect();
    verts.extend_from_slice(new_points);
    coplanar_plane(&verts, 1e-4, core.h()).is_some()
}

/// `busy_plane`: `(origin, normal)` of the plane the new loop lies on, when
/// that plane already carries mesh content (edges).
pub fn busy_plane(core: &Core, new_points: &[V3]) -> Option<(V3, V3)> {
    let (origin, normal) = coplanar_plane(new_points, 1e-4, core.h())?;
    let tol = 1e-4;
    let m = core.mesh();
    for &e in &m.edges {
        let ed = m.edge(e);
        let (a, b) = (m.position(ed.v0), m.position(ed.v1));
        if (a.sub(origin).dot(normal) as f64).abs() < tol && (b.sub(origin).dot(normal) as f64).abs() < tol {
            return Some((origin, normal));
        }
    }
    None
}

/// What `Ns` needs to rebuild the last shape.
struct Last {
    /// The command's identity on the undo stack (`None` = it never landed).
    serial: Option<u64>,
    center: V3,
    rim: V3,
    u: V3,
    v: V3,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum RadialKind {
    Circle,
    Polygon,
}

/// `_RadialTool`.
pub struct RadialTool {
    kind: RadialKind,
    pub start_point: Option<V3>,
    pub hover_point: Option<V3>,
    pub sides: i64,
    pl: PlaneState,
    viewport_seen: bool,
    last: Option<Last>,
    wireframe_color: Option<[f64; 4]>,
}

impl RadialTool {
    pub fn circle() -> RadialTool {
        RadialTool::new(RadialKind::Circle, 24)
    }

    pub fn polygon() -> RadialTool {
        RadialTool::new(RadialKind::Polygon, 6)
    }

    fn new(kind: RadialKind, sides: i64) -> RadialTool {
        RadialTool { kind, start_point: None, hover_point: None, sides, pl: PlaneState::default(), viewport_seen: false, last: None, wireframe_color: None }
    }

    fn reset(&mut self) {
        self.start_point = None;
        self.pl.work_plane = None;
        self.pl.hover_plane = None;
        self.wireframe_color = None;
        clear_plane_lock(&mut self.pl);
    }

    fn axes(&self, core: &Core) -> (V3, V3) {
        plane_axes(&core.axes, drawing_plane(self, core).1, core.h())
    }

    /// `_points(center, rim)`: the N-gon, one vertex toward the rim.
    fn points(&self, core: &Core, center: V3, rim: V3) -> Vec<V3> {
        let (u, v) = self.axes(core);
        let d = rim.sub(center);
        let (du, dv) = (d.dot(u) as f64, d.dot(v) as f64);
        let r = py_hypot2(du, dv);
        if r < 1e-6 {
            return Vec::new();
        }
        let a0 = dv.atan2(du); // one vertex toward the cursor
        let mut out = Vec::new();
        for k in 0..self.sides {
            let a = a0 + 2.0 * std::f64::consts::PI * (k as f64) / (self.sides as f64);
            out.push(center.add(u.mul(a.cos() as f32).add(v.mul(a.sin() as f32)).mul(r as f32)));
        }
        out
    }

    /// `_commit(viewport, pts, axes, center)`.
    fn commit(&mut self, core: &mut Core, pts: Vec<V3>, axes: Option<(V3, V3)>, center: Option<V3>) {
        let n = pts.len();
        let axes = axes.unwrap_or_else(|| self.axes(core));
        let center = center.unwrap_or_else(|| {
            let mut s = V3(0.0, 0.0, 0.0);
            for p in &pts {
                s = s.add(*p);
            }
            s.div(n as f32)
        });
        let segments: Vec<(V3, V3)> = (0..n).map(|i| (pts[i], pts[(i + 1) % n])).collect();
        let tag = || -> Box<dyn Command> { Box::new(TagCurveCommand { pts: pts.clone(), closed: true }) };
        let face = || -> Box<dyn Command> { Box::new(AddFaceCommand::new(pts.clone(), true, None, None)) };
        // The outline is drawn (a 24-segment circle reads round).
        let extra: Vec<Box<dyn Command>> = if flat_drawing(core, &pts) {
            // Flat drawing: rebuild the plane's faces from the edge graph.
            vec![tag(), Box::new(RebuildPlanarFacesCommand::new())]
        } else if let Some((o, nrm)) = busy_plane(core, &pts) {
            // 3D scene, but the drawing plane already carries content.
            vec![face(), tag(), Box::new(RebuildPlaneFacesCommand::new(o, nrm, core.h()))]
        } else {
            vec![face(), tag()]
        };
        let cmd = build_add_edges(&core.scene, &core.cx, &segments, false, extra, None, false);
        let before = core.undo_len();
        core.exec(cmd);
        let landed = core.undo_len() > before;
        self.last = Some(Last { serial: if landed { core.history.top_serial() } else { None }, center, rim: pts[0], u: axes.0, v: axes.1 });
        self.reset();
    }

    /// `_cursor_preview`: the ring on the cursor before the centre is placed.
    fn cursor_preview(&self, core: &Core) -> Vec<(V3, V3)> {
        if !self.viewport_seen {
            return Vec::new();
        }
        let Some(hover) = self.hover_point else { return Vec::new() };
        let (_p, normal) = preview_plane(&self.pl, core, hover);
        let (u, v) = plane_axes(&core.axes, normal, core.h());
        let Some(scale) = plane_preview_scale(core, hover, u, v) else { return Vec::new() };
        let r = PREVIEW_PX * scale;
        let n = self.sides.max(3);
        let pts: Vec<V3> = (0..n)
            .map(|i| {
                let a = 2.0 * std::f64::consts::PI * (i as f64) / (n as f64);
                hover.add(u.mul((r * a.cos()) as f32)).add(v.mul((r * a.sin()) as f32))
            })
            .collect();
        (0..n as usize).map(|i| (pts[i], pts[(i + 1) % n as usize])).collect()
    }
}

impl Planar for RadialTool {
    fn pl(&self) -> &PlaneState {
        &self.pl
    }
    fn pl_mut(&mut self) -> &mut PlaneState {
        &mut self.pl
    }
    /// Centre and rim: a rim snapped off the plane turns the circle to the
    /// axis plane holding both.
    fn plane_points(&self) -> Vec<Option<V3>> {
        match self.start_point {
            None => Vec::new(),
            Some(s) => vec![Some(s), self.hover_point],
        }
    }
}

impl DrawTool for RadialTool {
    fn name(&self) -> &'static str {
        match self.kind {
            RadialKind::Circle => "Circle",
            RadialKind::Polygon => "Polygon",
        }
    }

    fn snap_state(&self, _core: &Core) -> ToolState {
        let circle = self.kind == RadialKind::Circle;
        // AxisMagnet on a PlaneLock: the magnet works along the plane's axes
        // only (`magnet_off_plane` is False), so no screen detector.
        ToolState {
            magnetic_axis_deg: Some(MAGNET_DEG),
            screen_axis_px: None,
            magnet_off_plane: false,
            axis_source: circle,
            axis_source_empty_only: circle,
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
    fn set_snap_kind(&mut self, kind: Option<ingetrazo_snap::SnapKind>) {
        self.pl.snap_kind = kind;
    }
    /// `vcb_caption`: 'Sides' before the centre, 'Radius' after.
    fn vcb_label(&self) -> String {
        if self.start_point.is_some() { "Radius" } else { "Sides" }.into()
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
        let pts = self.points(core, start, ctx.world);
        if !pts.is_empty() {
            self.commit(core, pts, None, None);
        }
    }

    fn on_hover(&mut self, core: &mut Core, ctx: &Ctx) {
        note_plane(&mut self.pl, core);
        self.viewport_seen = true;
        self.hover_point = Some(ctx.world);
        self.wireframe_color = lock_color(&self.pl, &core.axes, core.h());
    }

    /// The classic "24s": the side count, typed at any moment - and, right
    /// after a shape is drawn, that shape is rebuilt in place with the new
    /// count, one undo step (#305).
    fn on_segments_value(&mut self, core: &mut Core, n: i64) -> Option<bool> {
        if n < 3 {
            return Some(false);
        }
        self.sides = n;
        core.flash(format!("{n} sides"));
        let rebuild = match (&self.last, core.history.top_serial()) {
            (Some(l), Some(top)) => self.start_point.is_none() && l.serial == Some(top),
            _ => false,
        };
        if rebuild {
            let l = self.last.take().expect("last");
            core.history_undo();
            let h = core.h();
            let d = l.rim.sub(l.center);
            let (du, dv) = (d.dot(l.u) as f64, d.dot(l.v) as f64);
            let r = py_hypot2(du, dv);
            let a0 = dv.atan2(du);
            let pts: Vec<V3> = (0..n)
                .map(|k| {
                    let a = a0 + 2.0 * std::f64::consts::PI * (k as f64) / (n as f64);
                    l.center.add(l.u.mul(a.cos() as f32).add(l.v.mul(a.sin() as f32)).mul(r as f32))
                })
                .collect();
            let _ = h;
            self.commit(core, pts, Some((l.u, l.v)), Some(l.center));
        }
        Some(true)
    }

    fn on_value(&mut self, core: &mut Core, value: &VcbValue) -> bool {
        let VcbValue::Num(value) = value else { return false };
        let value = *value;
        let Some(start) = self.start_point else {
            // Before the centre is placed, a typed number sets the side count.
            let n = value.round_ties_even() as i64;
            if n >= 3 {
                self.sides = n;
                core.flash(format!("{n} sides"));
                return true;
            }
            return false;
        };
        let Some(hover) = self.hover_point else { return false };
        if value <= 0.0 {
            return false;
        }
        // Keep the cursor's direction, override only the radius.
        let (u, v) = self.axes(core);
        let d = hover.sub(start);
        let ang = (d.dot(v) as f64).atan2(d.dot(u) as f64);
        let rim = start.add(u.mul(ang.cos() as f32).add(v.mul(ang.sin() as f32)).mul(value as f32));
        let pts = self.points(core, start, rim);
        if !pts.is_empty() {
            self.commit(core, pts, None, None);
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
        let Some(hover) = self.hover_point else { return Vec::new() };
        let Some(start) = self.start_point else { return self.cursor_preview(core) };
        let pts = self.points(core, start, hover);
        let n = pts.len();
        (0..n).map(|i| (pts[i], pts[(i + 1) % n])).collect()
    }

    fn value_label(&self, core: &Core) -> Option<(String, V3)> {
        let (start, hover) = (self.start_point?, self.hover_point?);
        // The radius DRAWN - the rim projected onto the plane - not the
        // straight distance to the cursor.
        let (u, v) = self.axes(core);
        let d = hover.sub(start);
        let r = py_hypot2(d.dot(u) as f64, d.dot(v) as f64);
        Some((format!("R {}  ({} sides)", core.fmt_len(r), self.sides), hover))
    }

    fn fields(&self) -> Vec<(&'static str, Field)> {
        let mut f = vec![
            ("start_point", Field::point(self.start_point)),
            ("hover_point", Field::point(self.hover_point)),
            ("sides", Field::Int(self.sides)),
        ];
        f.extend(plane_fields(&self.pl, self.wireframe_color));
        f
    }
}

#[allow(dead_code)]
fn _axis_unused(_: Axis) {}
