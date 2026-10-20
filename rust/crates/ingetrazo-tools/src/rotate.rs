// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! The Rotate tool (`tools/rotate.py`, Q): turn geometry with the classic
//! protractor ([`ProtractorBase`], shared with the Protractor tool).
//!
//! The first click is the centre (a click-drag from it sets a custom axis
//! along the drag), the second the reference arm; the geometry then swings
//! live with the cursor (15 degree ticks near the disc, free 0.1 degree farther
//! out) and a third click, a typed angle or a `3:12` slope commits. A tap of
//! Ctrl toggles COPY mode; right after a copy `3x` / `/3` lay a polar array.
//! The angle stays hot: typing a value redoes the rotation at the new angle.
//!
//! The live preview mutates the scene directly (no history entry), from a
//! snapshot of the rotating set (issue #163); a groups-only rotation leaves
//! the geometry alone and asks the viewport to draw the turned copy
//! (`begin_groups_preview` ...).
//!
//! Where it differs from Python (reported):
//!
//! * Section planes and reference images are not rotated (the selection of
//!   the Rust model carries them, Rotate does not act on them yet).
//! * `MoveTool._start_grip_rotation` (the rotation grips of Move) is not
//!   wired; `base` keeps its fields public (`custom_axis`, `start_point`...).
//! * The drawing axes are the world's (no group open).

use ingetrazo_formats::pyjson::Value;
use ingetrazo_math::{Hypot, M4, V3};
use ingetrazo_model::mesh::{EdgeId, FaceId, Mesh, VertexId};
use ingetrazo_model::scene::Scene;
use ingetrazo_model::units::fmt_angle_deg;
use ingetrazo_ops::command::Command;
use ingetrazo_ops::draw::{AddEdgeCommand, AddFaceCommand, CompoundCommand};
use ingetrazo_ops::group::{copy_group, group_at, group_at_mut, iter_placements, transformed_attrs, Path};
use ingetrazo_ops::grouping::InsertGroup;
use ingetrazo_ops::transform::{rotation_matrix, RotateGroup, RotateVertices};
use ingetrazo_geom::facegeom::plane_axes;

use crate::modify_base::{face_loops, keys, Env, ModifyTool, Mods, Probe, Sel, ToolCx, Vcb};
use crate::move_tool::gather_targets;
use crate::protractor::{f64_value, ProtractorBase};

/// The preview's snapshot: the rotating set exactly as it was.
struct Snapshot {
    /// Instances: the matrix they had.
    xf: Vec<(Path, M4)>,
    /// Classic groups: every vertex of their mesh.
    gv: Vec<(usize, VertexId, V3)>,
    /// Loose vertices.
    v: Vec<(VertexId, V3)>,
}

/// What the commit builder closes over, kept by the hot-retype window so the
/// rotation can be redone at a new angle after the tool has reset.
#[derive(Clone)]
struct Builder {
    start: V3,
    axis: V3,
    copy: bool,
    groups: Vec<Path>,
    positions: Vec<V3>,
    faces: Vec<FaceId>,
    edges: Vec<EdgeId>,
}

impl Builder {
    /// `build(degs)`: one command for the whole operation (or none).
    fn build(&self, env: &mut Env, degs: &[f64]) -> Option<Box<dyn Command>> {
        let h = env.cx.hypot;
        let mut cmds: Vec<Box<dyn Command>> = Vec::new();
        if self.copy {
            let mut id_map: Vec<((i64, u64), i64)> = Vec::new();
            let mut inserted = 0usize;
            let base = env.scene.groups.len();
            for &deg in degs {
                let mut m = rotation_matrix(self.start, self.axis, deg, h);
                for path in &self.groups {
                    let Scene { meshes, groups, .. } = &mut *env.scene;
                    let Some(g) = group_at_mut(groups, path) else { continue };
                    let copy = copy_group(meshes, g, None, false, &mut env.cx.counters);
                    cmds.push(Box::new(InsertGroup::new(copy, None)));
                    cmds.push(Box::new(RotateGroup::new(&[base + inserted], self.start, self.axis, deg)));
                    inserted += 1;
                }
                let mesh = env.cx.mesh(env.scene);
                for &f in &self.faces {
                    let loops = face_loops(mesh, f);
                    let outer: Vec<V3> = loops[0].iter().map(|&v| m.map(v)).collect();
                    let holes: Vec<Vec<V3>> = loops[1..].iter().map(|hl| hl.iter().map(|&v| m.map(v)).collect()).collect();
                    let attrs = transformed_attrs(&mesh.face(f).attrs, &mut m);
                    cmds.push(Box::new(AddFaceCommand::new(outer, false, if holes.is_empty() { None } else { Some(holes) }, Some(attrs))));
                }
                for &e in &self.edges {
                    let (curve, soft, a, b) = {
                        let ed = mesh.edge(e);
                        (ed.curve, ed.soft, mesh.position(ed.v0), mesh.position(ed.v1))
                    };
                    // One fresh curve id per (curve, copy): the arms of an
                    // array are separate curves, not one long one.
                    let key = curve.map(|c| (c, deg.to_bits()));
                    if let Some(k) = key {
                        if !id_map.iter().any(|&(kk, _)| kk == k) {
                            let fresh = Mesh::next_curve_id(&mut env.cx.counters.curve);
                            id_map.push((k, fresh));
                        }
                    }
                    let new_curve = key.and_then(|k| id_map.iter().find(|&&(kk, _)| kk == k).map(|&(_, v)| v));
                    cmds.push(Box::new(AddEdgeCommand::new(m.map(a), m.map(b), if soft { Some(true) } else { None }, new_curve)));
                }
            }
        } else {
            // Nothing is copied, so there is only ever one angle.
            let deg = *degs.last()?;
            for g in &self.groups {
                cmds.push(Box::new(RotateGroup::new(g, self.start, self.axis, deg)));
            }
            if !self.positions.is_empty() {
                cmds.push(Box::new(RotateVertices::new(&self.positions, self.start, self.axis, deg)));
            }
        }
        if cmds.is_empty() {
            return None;
        }
        Some(if cmds.len() == 1 { cmds.pop().expect("one") } else { Box::new(CompoundCommand { commands: cmds }) })
    }
}

/// `_last`: the rotation just made, for the hot retype and the polar array.
struct Last {
    serial: Option<u64>,
    builder: Builder,
    deg: f64,
    copy: bool,
    sign: f64,
}

pub struct RotateTool {
    pub base: ProtractorBase,
    groups: Vec<Path>,
    positions: Vec<V3>,
    verts: Vec<VertexId>,
    sel_faces: Vec<FaceId>,
    sel_edges: Vec<EdgeId>,
    base_segments: Vec<(V3, V3)>,
    preview_deg: f64,
    orig: Option<Snapshot>,
    copy: bool,
    vp_preview: bool,
    last: Option<Last>,
}

impl Default for RotateTool {
    fn default() -> Self {
        RotateTool::new()
    }
}

/// CPython's `float_floor_div` for a positive divisor.
fn floor_div(vx: f64, wx: f64) -> f64 {
    let m = vx % wx;
    let div = (vx - m) / wx;
    let mut fd = div.floor();
    if div - fd > 0.5 {
        fd += 1.0;
    }
    fd
}

impl RotateTool {
    pub fn new() -> RotateTool {
        RotateTool {
            base: ProtractorBase::new(false),
            groups: Vec::new(),
            positions: Vec::new(),
            verts: Vec::new(),
            sel_faces: Vec::new(),
            sel_edges: Vec::new(),
            base_segments: Vec::new(),
            preview_deg: 0.0,
            orig: None,
            copy: false,
            vp_preview: false,
            last: None,
        }
    }

    pub fn is_copy(&self) -> bool {
        self.copy
    }

    // ---- keyboard ------------------------------------------------------------------------

    fn toggle_copy(&mut self, env: &mut Env, say: bool) {
        self.copy = !self.copy;
        if self.copy {
            self.revert_preview(env); // the original stops swinging
            if say {
                env.flash("Rotate a copy: on");
            }
        } else if say {
            env.flash("Rotate a copy: off");
        }
    }

    // ---- gathering -------------------------------------------------------------------------

    /// `_gather_copy_entities`.
    fn gather_copy_entities(&mut self, env: &Env, cx: &ToolCx) {
        self.sel_faces.clear();
        self.sel_edges.clear();
        self.base_segments.clear();
        for path in &self.groups {
            let Some(group) = group_at(&env.scene.groups, path) else { continue };
            for (pg, xf) in iter_placements(group, None) {
                let mesh = env.scene.mesh(pg.mesh);
                for &e in &mesh.edges {
                    let ed = mesh.edge(e);
                    let (mut a, mut b) = (mesh.position(ed.v0), mesh.position(ed.v1));
                    if let Some(x) = xf {
                        a = x.map(a);
                        b = x.map(b);
                    }
                    self.base_segments.push((a, b));
                }
            }
        }
        let mut sel: Vec<Sel> = env.sel.iter().cloned().collect();
        if sel.is_empty() && self.groups.is_empty() {
            if let Some(e) = env.vp.pick_edge(env.scene, env.cx, cx.screen.0, cx.screen.1) {
                sel = vec![Sel::Edge(e)];
            } else if let Some(f) = env.vp.pick_face(env.scene, env.cx, cx.screen.0, cx.screen.1) {
                sel = vec![Sel::Face(f)];
            }
        }
        for s in &sel {
            match s {
                Sel::Face(f) => self.sel_faces.push(*f),
                Sel::Edge(e) => self.sel_edges.push(*e),
                _ => {}
            }
        }
        let mesh = env.cx.mesh(env.scene);
        for &f in &self.sel_faces {
            for lp in face_loops(mesh, f) {
                let n = lp.len();
                for i in 0..n {
                    self.base_segments.push((lp[i], lp[(i + 1) % n]));
                }
            }
        }
        for &e in &self.sel_edges {
            let ed = mesh.edge(e);
            self.base_segments.push((mesh.position(ed.v0), mesh.position(ed.v1)));
        }
    }

    // ---- the live preview -------------------------------------------------------------------

    /// `_snapshot`, made once per preview.
    fn ensure_snapshot(&mut self, env: &Env) {
        if self.orig.is_some() {
            return;
        }
        let mut snap = Snapshot { xf: Vec::new(), gv: Vec::new(), v: Vec::new() };
        if !self.vp_preview {
            for path in &self.groups {
                let Some(g) = group_at(&env.scene.groups, path) else { continue };
                if let Some(x) = g.xform {
                    snap.xf.push((path.clone(), x));
                } else {
                    let mesh = env.scene.mesh(g.mesh);
                    for &vx in &mesh.vertices {
                        snap.gv.push((g.mesh.0 as usize, vx, mesh.position(vx)));
                    }
                }
            }
            let mesh = env.cx.mesh(env.scene);
            snap.v = self.verts.iter().map(|&vx| (vx, mesh.position(vx))).collect();
        }
        self.orig = Some(snap);
    }

    /// `_pose`: place the rotating set at `m` applied to its snapshot
    /// (`None` puts the snapshot back as it was).
    fn pose(&mut self, env: &mut Env, m: Option<M4>) {
        self.ensure_snapshot(env);
        let snap = self.orig.as_ref().expect("snapshot");
        for (path, xf) in &snap.xf {
            if let Some(g) = group_at_mut(&mut env.scene.groups, path) {
                g.xform = Some(match m {
                    Some(m) => m.mul(xf),
                    None => *xf,
                });
            }
        }
        for &(mesh, vx, pos) in &snap.gv {
            env.scene.meshes[mesh].place_vertex(vx, m.map_or(pos, |m| m.map(pos)));
        }
        for &(vx, pos) in &snap.v {
            env.cx.mesh_mut(env.scene).place_vertex(vx, m.map_or(pos, |m| m.map(pos)));
        }
        if !snap.xf.is_empty() || !snap.gv.is_empty() || !snap.v.is_empty() {
            *env.version += 1;
        }
    }

    /// `_apply_preview`.
    fn apply_preview(&mut self, env: &mut Env, target_deg: f64) {
        let h = env.cx.hypot;
        let m = rotation_matrix(self.base.start_point.expect("start"), self.base.axis(h), target_deg, h);
        self.pose(env, Some(m));
        self.preview_deg = target_deg;
        if self.vp_preview {
            env.vp.set_groups_preview_matrix(&m);
        }
    }

    /// `_revert_preview`.
    fn revert_preview(&mut self, env: &mut Env) {
        if self.orig.is_some() {
            self.pose(env, None); // the exact values from before
            self.orig = None;
        }
        self.preview_deg = 0.0;
        if self.vp_preview {
            env.vp.set_groups_preview_matrix(&M4::identity()); // park, keep freeze
        }
    }

    /// `_end_vp_preview`.
    fn end_vp_preview(&mut self, env: &mut Env) {
        if self.vp_preview {
            env.vp.end_groups_preview();
        }
        self.vp_preview = false;
    }

    // ---- commit --------------------------------------------------------------------------------

    fn make_builder(&self, h: Hypot) -> Builder {
        Builder {
            start: self.base.start_point.expect("start"),
            axis: self.base.axis(h),
            copy: self.copy,
            groups: self.groups.clone(),
            positions: self.positions.clone(),
            faces: self.sel_faces.clone(),
            edges: self.sel_edges.clone(),
        }
    }

    fn commit(&mut self, env: &mut Env, deg: f64) {
        let h = env.cx.hypot;
        self.revert_preview(env);
        self.end_vp_preview(env);
        if deg.abs() > 1e-9 {
            let builder = self.make_builder(h);
            let copy = self.copy;
            if let Some(cmd) = builder.build(env, &[deg]) {
                let serial = env.execute_serial(cmd);
                // The angle stays hot - typing a value redoes this rotation,
                // and after a COPY `3x` / `/3` fan it into a polar array.
                self.last = Some(Last { serial, builder, deg, copy, sign: if deg < 0.0 { -1.0 } else { 1.0 } });
            }
        }
        self.reset();
    }

    fn reset(&mut self) {
        self.base.reset_protractor();
        self.groups.clear();
        self.positions.clear();
        self.verts.clear();
        self.sel_faces.clear();
        self.sel_edges.clear();
        self.base_segments.clear();
        self.preview_deg = 0.0;
        self.orig = None;
        self.copy = false; // the Ctrl modifier arms ONE operation
    }

    // ---- what the viewport reads ---------------------------------------------------------------

    /// `array_pending`.
    pub fn array_pending(&self, env: &Env) -> bool {
        match &self.last {
            Some(l) if l.copy && self.base.ref_point.is_none() => env.is_top(l.serial),
            _ => false,
        }
    }

    /// `_arc_segments`: the arc between the two arms, at the reference radius.
    fn arc_segments(&self, deg: f64, h: Hypot) -> Vec<(V3, V3)> {
        let (u, v) = plane_axes(self.base.axis(h), h);
        let start = self.base.start_point.expect("start");
        let a = self.base.ref_point.expect("ref").sub(start);
        let r = a.length(h) as f64 * 0.75;
        let a0 = (a.dot(v) as f64).atan2(a.dot(u) as f64);
        let steps = std::cmp::max(2, floor_div(deg.abs(), 10.0) as i64 + 1);
        let pts: Vec<V3> = (0..=steps)
            .map(|k| {
                let t = a0 + deg.to_radians() * k as f64 / steps as f64;
                start.add(u.mul(t.cos() as f32).add(v.mul(t.sin() as f32)).mul(r as f32))
            })
            .collect();
        pts.windows(2).map(|w| (w[0], w[1])).collect()
    }

    pub fn rubber_band_lines(&self, h: Hypot) -> Vec<(V3, V3)> {
        let Some(mut segments) = self.base.disc_and_arms(h) else { return Vec::new() };
        let (Some(start), Some(hover)) = (self.base.start_point, self.base.hover_point) else { return segments };
        if let Some(rf) = self.base.ref_point {
            segments.push((start, rf));
            if let Some(deg) = self.base.commit_deg(hover, h) {
                segments.extend(self.arc_segments(deg, h));
                if self.copy && !self.base_segments.is_empty() {
                    // Copy mode: the original stays put - preview the rotated
                    // COPY as a wireframe at the cursor angle.
                    let m = rotation_matrix(start, self.base.axis(h), deg, h);
                    segments.extend(self.base_segments.iter().map(|&(a, b)| (m.map(a), m.map(b))));
                }
            }
        }
        segments
    }

    pub fn value_label(&self, env: &Env) -> Option<(String, V3)> {
        let (Some(_), Some(hover)) = (self.base.ref_point, self.base.hover_point) else { return None };
        let deg = self.base.display_deg(hover, env.cx.hypot)?;
        Some((fmt_angle_deg(deg, true, env.units), hover))
    }
}

impl ModifyTool for RotateTool {
    fn on_activate(&mut self, _env: &mut Env) {
        self.reset();
        self.copy = false;
        self.last = None;
    }

    fn on_deactivate(&mut self, env: &mut Env) {
        self.revert_preview(env);
        self.end_vp_preview(env);
        self.reset();
        self.copy = false;
        self.last = None;
        self.base.axis_pick = None;
        self.base.hover_point = None;
    }

    fn on_key(&mut self, env: &mut Env, key: i32, _mods: Mods) -> bool {
        // Ctrl toggles copy mode (rotate a copy, original stays).
        if key == keys::CONTROL {
            self.toggle_copy(env, true);
            self.base.ctrl_toggled = true;
            return true;
        }
        self.base.on_key(env, key)
    }

    fn on_key_release(&mut self, env: &mut Env, key: i32) -> bool {
        // The copy toggle happens on the Ctrl PRESS; a Ctrl that turns out to
        // be part of a shortcut takes it back on the release, silently.
        if key != keys::CONTROL || !self.base.ctrl_toggled {
            return false;
        }
        self.base.ctrl_toggled = false;
        if !env.vp.ctrl_tapped() {
            self.toggle_copy(env, false);
        }
        true
    }

    fn on_click(&mut self, env: &mut Env, cx: &ToolCx) {
        let h = env.cx.hypot;
        self.last = None; // a click ends the retype window
        // The viewport's follow-up (`_dispatch_tool_click`): a start point
        // placed on a face locks the plane to it.
        let hit = self.base.face_at_click(env, cx);
        let Some(start) = self.base.start_point else {
            let (groups, positions) = gather_targets(env, cx);
            if groups.is_empty() && positions.is_empty() {
                env.flash("Select (or click) the geometry to rotate first");
                return;
            }
            self.groups = groups;
            self.positions = positions;
            let mesh = env.cx.mesh(env.scene);
            self.verts = self.positions.iter().filter_map(|&p| mesh.vertex_at(p)).collect();
            self.vp_preview = false;
            if !self.groups.is_empty() && self.verts.is_empty() && self.positions.is_empty() {
                // Groups-only rotate: viewport-side preview (see MoveTool).
                env.vp.begin_groups_preview(&self.groups, false);
                self.vp_preview = true;
            }
            self.gather_copy_entities(env, cx);
            self.base.start_point = Some(cx.world);
            self.base.axis_drag_armed = true; // a DRAG from here sets the axis
            self.base.lock_plane_to_face(env, hit);
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
        let h = env.cx.hypot;
        self.base.hover_point = Some(cx.world);
        self.base.track_axis_drag(env); // live tilt preview (issue #25)
        self.base.infer_plane(env, cx);
        self.base.update_screen_metrics(env, cx);
        self.base.note_snap(cx);
        if self.base.ref_point.is_some() && !self.copy {
            if let Some(deg) = self.base.commit_deg(cx.world, h) {
                self.apply_preview(env, deg);
            }
        }
    }

    fn on_release(&mut self, env: &mut Env) {
        if self.base.release_axis_drag(env) {
            env.flash("Rotation axis set along the drag");
        }
    }

    fn on_value(&mut self, env: &mut Env, value: &Vcb) -> bool {
        let Vcb::Num(value) = value else { return false };
        let value = *value;
        let h = env.cx.hypot;
        if self.base.ref_point.is_some() {
            // The typed angle turns the way the user is currently dragging.
            let mut sign = if self.preview_deg < 0.0 { -1.0 } else { 1.0 };
            if let (true, Some(hover)) = (self.copy, self.base.hover_point) {
                let cur = self.base.angle_to(hover, h);
                sign = if cur.is_some_and(|c| c < 0.0) { -1.0 } else { 1.0 };
            }
            self.commit(env, sign * value.abs());
            return true;
        }
        let Some(last) = &self.last else { return false };
        // Hot retype: redo the rotation just made at the new angle.
        if !env.is_top(last.serial) {
            self.last = None;
            return false;
        }
        let side = last.sign * if value >= 0.0 { 1.0 } else { -1.0 };
        let deg = side * value.abs();
        let builder = last.builder.clone();
        env.undo();
        let Some(cmd) = builder.build(env, &[deg]) else {
            self.last = None;
            return false;
        };
        let serial = env.execute_serial(cmd);
        let last = self.last.as_mut().expect("last");
        last.serial = serial;
        last.deg = deg;
        true
    }

    fn close_array_window(&mut self) {
        // Esc right after a rotate-copy: `3x` / `/3` stop applying.
        self.last = None;
    }

    fn on_array_value(&mut self, env: &mut Env, count: i64, mode: char, _step: Option<f64>) -> bool {
        let Some(last) = &self.last else { return false };
        if !last.copy || self.base.ref_point.is_some() || count < 1 {
            return false;
        }
        if !env.is_top(last.serial) {
            self.last = None;
            return false;
        }
        let deg = last.deg;
        let degs: Vec<f64> = if mode == '/' {
            (1..=count).map(|k| deg * (k as f64 / count as f64)).collect()
        } else {
            (1..=count).map(|k| deg * k as f64).collect()
        };
        let builder = last.builder.clone();
        env.undo();
        let Some(cmd) = builder.build(env, &degs) else {
            self.last = None;
            return false;
        };
        let serial = env.execute_serial(cmd);
        self.last.as_mut().expect("last").serial = serial;
        env.flash(format!("{count} copies"));
        true
    }

    fn on_cancel(&mut self, env: &mut Env) {
        self.revert_preview(env);
        self.end_vp_preview(env);
        self.reset();
        self.last = None;
    }

    fn probe(&self, env: &Env) -> Probe {
        let h = env.cx.hypot;
        let mut p = Probe::default();
        p.rb = Some(self.rubber_band_lines(h));
        p.vl = Some(self.value_label(env));
        p.plus = Some(self.copy);
        p.wf = Some(self.base.wireframe_color(h));
        p.cap = Some("Angle".to_string());
        p.unitless = Some(true);
        p.ap = Some(self.array_pending(env));
        p.start = self.base.start_point;
        p.hover = self.base.hover_point;
        // `snap_excluded`: the geometry swinging live.
        p.excl = Some(if self.base.ref_point.is_none() || self.copy {
            None
        } else {
            let mesh = env.cx.mesh(env.scene);
            let mut seen = std::collections::HashSet::new();
            let mut edges: Vec<usize> = Vec::new();
            for &v in &self.verts {
                for e in mesh.edges_at(v) {
                    if seen.insert(e) {
                        if let Some(i) = mesh.edges.iter().position(|x| *x == e) {
                            edges.push(i);
                        }
                    }
                }
            }
            if edges.is_empty() && self.groups.is_empty() {
                None
            } else {
                Some((edges, self.groups.clone()))
            }
        });
        let i = |n: usize| Value::int(n as i64);
        let s = |k: &str, v: Value| (k.to_string(), v);
        let mut x = self.base.probe_fields(h);
        x.extend([
            s("copy", Value::Bool(self.copy)),
            s("n_groups", i(self.groups.len())),
            s("n_pos", i(self.positions.len())),
            s("n_verts", i(self.verts.len())),
            s("sel_faces", i(self.sel_faces.len())),
            s("sel_edges", i(self.sel_edges.len())),
            s("base_segments", i(self.base_segments.len())),
            s("preview_deg", f64_value(self.preview_deg)),
            s("has_orig", Value::Bool(self.orig.is_some())),
            s("vp_preview", Value::Bool(self.vp_preview)),
            s("last", Value::Bool(self.last.is_some())),
            s("last_copy", self.last.as_ref().map(|l| Value::Bool(l.copy)).unwrap_or(Value::Null)),
            s("last_deg", self.last.as_ref().map(|l| f64_value(l.deg)).unwrap_or(Value::Null)),
            s("last_sign", self.last.as_ref().map(|l| f64_value(l.sign)).unwrap_or(Value::Null)),
            s("last_top", Value::Bool(self.last.as_ref().is_some_and(|l| env.is_top(l.serial)))),
        ]);
        p.x = Some(Value::Object(x));
        p
    }
}
