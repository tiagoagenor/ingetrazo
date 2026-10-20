// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! The Flip tool (`tools/flip.py`): mirror the selection across its red,
//! green or blue plane, with no UI.
//!
//! With a selection three planes appear over it, one per axis; hovering
//! picks one (the projected quad under the cursor, `QPolygonF.containsPoint`
//! with the odd-even rule; ties go to the plane whose centre is nearest on
//! screen) and ONE CLICK flips the selection about it. The arrow keys pick a
//! plane (Right red, Left green, Up blue); a Ctrl tap toggles COPY mode,
//! which leaves the original and creates the mirrored duplicate (for one
//! flip).
//!
//! [`FlipTool::flip`] is also an API: the right-click menu's Flip Along.
//!
//! Where it differs from Python (reported): the overlay drawing
//! (`draw_overlay`) is the host's; the drawing axes are the world's.

use ingetrazo_formats::pyjson::Value;
use ingetrazo_math::V3;
use ingetrazo_model::mesh::{EdgeId, FaceId, Mesh};
use ingetrazo_model::scene::Scene;
use ingetrazo_ops::command::Command;
use ingetrazo_ops::draw::{AddEdgeCommand, AddFaceCommand, CompoundCommand};
use ingetrazo_ops::group::{affine_rows, copy_group, group_at, group_at_mut, iter_placements, np_affine, transformed_attrs, Path};
use ingetrazo_ops::grouping::InsertGroup;
use ingetrazo_ops::transform::{mirror_matrix, FlipGroups, FlipVertices};

use crate::modify_base::{arrow_axis, key6, keys, Env, ModifyTool, Mods, Probe, Sel, ToolCx};
use ingetrazo_snap::Axis;

const AXES: [(char, V3); 3] = [('x', V3(1.0, 0.0, 0.0)), ('y', V3(0.0, 1.0, 0.0)), ('z', V3(0.0, 0.0, 1.0))];

fn axis_char(a: Axis) -> char {
    match a {
        Axis::X => 'x',
        Axis::Y => 'y',
        Axis::Z => 'z',
    }
}

fn axis_vec(c: char) -> V3 {
    AXES.iter().find(|(k, _)| *k == c).map(|(_, v)| *v).expect("axis x, y or z")
}

/// What `_targets(viewport, full=True)` answers.
struct Targets {
    ok: bool,
    groups: Vec<Path>,
    positions: Vec<V3>,
    faces: Vec<FaceId>,
    edges: Vec<EdgeId>,
    centre: V3,
}

#[derive(Default)]
pub struct FlipTool {
    pub hover_point: Option<V3>,
    hover_axis: Option<char>,
    lock_axis: Option<char>,
    copy: bool,
    ctrl_toggled: bool,
}

/// `qFuzzyCompare(double, double)`.
fn fuzzy_compare(a: f64, b: f64) -> bool {
    (a - b).abs() * 1_000_000_000_000.0 <= a.abs().min(b.abs())
}

/// `qt_polygon_isect_line`: count the crossing of the scan line through `pos`.
fn isect_line(p1: (f64, f64), p2: (f64, f64), pos: (f64, f64), winding: &mut i32) {
    let (mut x1, mut y1) = p1;
    let (mut x2, mut y2) = p2;
    let y = pos.1;
    let mut dir = 1;
    if fuzzy_compare(y1, y2) {
        return; // horizontal lines are ignored by the scan conversion rule
    } else if y2 < y1 {
        std::mem::swap(&mut x1, &mut x2);
        std::mem::swap(&mut y1, &mut y2);
        dir = -1;
    }
    if y >= y1 && y < y2 {
        let x = x1 + ((x2 - x1) / (y2 - y1)) * (y - y1);
        if x <= pos.0 {
            *winding += dir;
        }
    }
}

/// `QPolygonF::containsPoint(pt, Qt::OddEvenFill)`.
fn contains_odd_even(poly: &[(f64, f64)], pt: (f64, f64)) -> bool {
    if poly.is_empty() {
        return false;
    }
    let mut winding = 0;
    let first = poly[0];
    let mut last = first;
    for &e in &poly[1..] {
        isect_line(last, e, pt, &mut winding);
        last = e;
    }
    // implicitly close the last subpath (`QPointF !=` is a fuzzy compare)
    let same = |a: f64, b: f64| if a == 0.0 || b == 0.0 { (a + b).abs() <= 1e-12 } else { fuzzy_compare(a, b) };
    if !(same(last.0, first.0) && same(last.1, first.1)) {
        isect_line(last, first, pt, &mut winding);
    }
    winding % 2 != 0
}

impl FlipTool {
    pub fn new() -> FlipTool {
        FlipTool::default()
    }

    pub fn is_copy(&self) -> bool {
        self.copy
    }

    // ---- copy toggle ------------------------------------------------------------

    fn toggle_copy(&mut self, env: &mut Env, say: bool) {
        self.copy = !self.copy;
        if say {
            env.flash(if self.copy { "Flip a copy: on" } else { "Flip a copy: off" });
        }
    }

    // ---- the selection ---------------------------------------------------------------

    /// `_bounds`: numpy float64 min/max of the group placement points and the
    /// loose points.
    fn bounds(env: &Env) -> Option<(V3, V3)> {
        let mut arrays: Vec<Vec<[f64; 3]>> = Vec::new();
        let mut pts: Vec<[f64; 3]> = Vec::new();
        let mesh = env.cx.mesh(env.scene);
        for s in env.sel.iter() {
            match s {
                Sel::Group(path) => {
                    let Some(g) = group_at(&env.scene.groups, path) else { continue };
                    let arr = placement_points(env.scene, g);
                    if !arr.is_empty() {
                        arrays.push(arr);
                    }
                }
                Sel::Face(f) => {
                    for &v in &mesh.face(*f).outer {
                        let p = mesh.position(v);
                        pts.push([p.0 as f64, p.1 as f64, p.2 as f64]);
                    }
                }
                Sel::Edge(e) => {
                    let ed = mesh.edge(*e);
                    for v in [ed.v0, ed.v1] {
                        let p = mesh.position(v);
                        pts.push([p.0 as f64, p.1 as f64, p.2 as f64]);
                    }
                }
                _ => {}
            }
        }
        if !pts.is_empty() {
            arrays.push(pts);
        }
        if arrays.is_empty() {
            return None;
        }
        let mut lo = [f64::INFINITY; 3];
        let mut hi = [f64::NEG_INFINITY; 3];
        for p in arrays.iter().flatten() {
            for i in 0..3 {
                if p[i] < lo[i] {
                    lo[i] = p[i];
                }
                if p[i] > hi[i] {
                    hi[i] = p[i];
                }
            }
        }
        Some((V3(lo[0] as f32, lo[1] as f32, lo[2] as f32), V3(hi[0] as f32, hi[1] as f32, hi[2] as f32)))
    }

    fn ok(env: &Env) -> bool {
        env.sel.iter().any(|s| matches!(s, Sel::Group(_) | Sel::Face(_) | Sel::Edge(_)))
    }

    /// `_targets(viewport, full=True)`.
    fn targets(env: &Env) -> Targets {
        let groups = env.sel.groups();
        let faces = env.sel.faces();
        let edges = env.sel.edges();
        let ok = !groups.is_empty() || !faces.is_empty() || !edges.is_empty();
        let mesh = env.cx.mesh(env.scene);
        let mut positions: Vec<V3> = Vec::new();
        let mut seen: Vec<[f64; 3]> = Vec::new();
        let mut take = |p: V3| {
            let k = key6(p);
            if !seen.contains(&k) {
                seen.push(k);
                positions.push(p);
            }
        };
        for &f in &faces {
            for &v in &mesh.face(f).outer {
                take(mesh.position(v));
            }
        }
        for &e in &edges {
            let ed = mesh.edge(e);
            for v in [ed.v0, ed.v1] {
                take(mesh.position(v));
            }
        }
        let centre = match Self::bounds(env) {
            Some((lo, hi)) => lo.add(hi).mul(0.5),
            None => V3(0.0, 0.0, 0.0),
        };
        Targets { ok, groups, positions, faces, edges, centre }
    }

    // ---- the flip ----------------------------------------------------------------------

    /// Mirror the selection across its own centre, along the world `axis`
    /// ('x' red, 'y' green, 'z' blue), as one undo step - what a click on a
    /// plane does, and what the right-click menu's Flip Along asks for.
    /// `copy` leaves the original in place. False when nothing is selected.
    pub fn flip(&mut self, env: &mut Env, axis: char, copy: bool) -> bool {
        let t = Self::targets(env);
        if !t.ok {
            env.flash("Select the geometry to flip first");
            return false;
        }
        let n = axis_vec(axis);
        let mut cmds: Vec<Box<dyn Command>> = Vec::new();
        if copy {
            let h = env.cx.hypot;
            let mut m = mirror_matrix(t.centre, n, h);
            let n_groups = env.scene.groups.len();
            for (k, path) in t.groups.iter().enumerate() {
                let copy = {
                    let Scene { meshes, groups, .. } = &mut *env.scene;
                    let Some(g) = group_at_mut(groups, path) else { continue };
                    copy_group(meshes, g, None, false, &mut env.cx.counters)
                };
                cmds.push(Box::new(InsertGroup::new(copy, None)));
                // the copy lands at the end of the top-level groups
                cmds.push(Box::new(FlipGroups::new(&[vec![n_groups + k]], t.centre, n)));
            }
            let mesh = env.cx.mesh(env.scene);
            let mut id_map: Vec<(i64, i64)> = Vec::new();
            for &f in &t.faces {
                let face = mesh.face(f);
                // Mirrored loops are re-reversed so the copy faces OUT.
                let loop_: Vec<V3> = face.outer.iter().map(|&v| m.map(mesh.position(v))).rev().collect();
                let holes: Vec<Vec<V3>> = face.holes.iter().map(|hl| hl.iter().map(|&v| m.map(mesh.position(v))).rev().collect()).collect();
                let attrs = transformed_attrs(&face.attrs, &mut m);
                cmds.push(Box::new(AddFaceCommand::new(loop_, false, if holes.is_empty() { None } else { Some(holes) }, Some(attrs))));
            }
            for &e in &t.edges {
                let (curve, soft, a, b) = {
                    let ed = mesh.edge(e);
                    (ed.curve, ed.soft, mesh.position(ed.v0), mesh.position(ed.v1))
                };
                if let Some(c) = curve {
                    if !id_map.iter().any(|&(k, _)| k == c) {
                        let fresh = Mesh::next_curve_id(&mut env.cx.counters.curve);
                        id_map.push((c, fresh));
                    }
                }
                let new_curve = curve.and_then(|c| id_map.iter().find(|&&(k, _)| k == c).map(|&(_, v)| v));
                cmds.push(Box::new(AddEdgeCommand::new(m.map(a), m.map(b), if soft { Some(true) } else { None }, new_curve)));
            }
        } else {
            if !t.groups.is_empty() {
                cmds.push(Box::new(FlipGroups::new(&t.groups, t.centre, n)));
            }
            if !t.positions.is_empty() {
                cmds.push(Box::new(FlipVertices::new(&t.positions, t.centre, n, &t.faces)));
            }
        }
        if cmds.is_empty() {
            return false;
        }
        let cmd = if cmds.len() == 1 { cmds.pop().expect("one") } else { Box::new(CompoundCommand { commands: cmds }) };
        env.execute(cmd);
        true
    }

    // ---- the planes ---------------------------------------------------------------------

    /// `_plane_quad`.
    fn plane_quad(axis: char, c: V3, half: V3, margin: f64) -> [V3; 4] {
        let hx = (half.0 as f64).max(0.3) * margin;
        let hy = (half.1 as f64).max(0.3) * margin;
        let hz = (half.2 as f64).max(0.3) * margin;
        let (cx, cy, cz) = (c.0 as f64, c.1 as f64, c.2 as f64);
        let f = |x: f64| x as f32;
        match axis {
            'x' => [V3(c.0, f(cy - hy), f(cz - hz)), V3(c.0, f(cy + hy), f(cz - hz)), V3(c.0, f(cy + hy), f(cz + hz)), V3(c.0, f(cy - hy), f(cz + hz))],
            'y' => [V3(f(cx - hx), c.1, f(cz - hz)), V3(f(cx + hx), c.1, f(cz - hz)), V3(f(cx + hx), c.1, f(cz + hz)), V3(f(cx - hx), c.1, f(cz + hz))],
            _ => [V3(f(cx - hx), f(cy - hy), c.2), V3(f(cx + hx), f(cy - hy), c.2), V3(f(cx + hx), f(cy + hy), c.2), V3(f(cx - hx), f(cy + hy), c.2)],
        }
    }

    /// `_plane_under_cursor`: which of the three planes the cursor is inside
    /// (projected); ties break toward the plane whose centre is nearest.
    fn plane_under_cursor(env: &Env, sx: f64, sy: f64) -> Option<char> {
        let (lo, hi) = Self::bounds(env)?;
        let centre = lo.add(hi).mul(0.5);
        let half = hi.sub(lo).mul(0.5);
        let mut best = None;
        let mut best_d = f64::INFINITY;
        for (axis, _) in AXES {
            let quad = Self::plane_quad(axis, centre, half, 1.15);
            let pts: Option<Vec<(f64, f64)>> = quad.iter().map(|&q| env.vp.world_to_pixel(q)).collect();
            let Some(pts) = pts else { continue };
            if !contains_odd_even(&pts, (sx, sy)) {
                continue;
            }
            let cpx = env.vp.world_to_pixel(centre);
            let mid = ((((pts[0].0 + pts[1].0) + pts[2].0) + pts[3].0) / 4.0, (((pts[0].1 + pts[1].1) + pts[2].1) + pts[3].1) / 4.0);
            let d = (mid.0 - sx) * (mid.0 - sx) + (mid.1 - sy) * (mid.1 - sy);
            if cpx.is_some() && d < best_d {
                best_d = d;
                best = Some(axis);
            }
        }
        best
    }
}

/// `placement_points`: every vertex of the group and its nested placements in
/// world space, as float64 rows (`arr @ rot.T + trans`, NumPy's order).
fn placement_points(scene: &Scene, group: &ingetrazo_model::scene::Group) -> Vec<[f64; 3]> {
    let mut out: Vec<[f64; 3]> = Vec::new();
    for (g, m) in iter_placements(group, None) {
        let mesh = scene.mesh(g.mesh);
        let arr: Vec<[f64; 3]> = mesh.vertices.iter().map(|&v| {
            let p = mesh.position(v);
            [p.0 as f64, p.1 as f64, p.2 as f64]
        }).collect();
        if arr.is_empty() {
            continue;
        }
        match m {
            Some(m) => {
                let (rot, trans) = np_affine(&m);
                out.extend(affine_rows(&arr, &rot, &trans));
            }
            None => out.extend(arr),
        }
    }
    out
}


fn axis_value(a: Option<char>) -> Value {
    match a {
        Some(c) => Value::Str(c.to_string()),
        None => Value::Null,
    }
}

impl ModifyTool for FlipTool {
    fn on_activate(&mut self, env: &mut Env) {
        self.hover_axis = None;
        self.lock_axis = None;
        self.copy = false;
        if !Self::ok(env) {
            env.flash("Select the geometry to flip first");
        }
    }

    fn on_deactivate(&mut self, _env: &mut Env) {
        self.hover_axis = None;
    }

    fn on_key_release(&mut self, env: &mut Env, key: i32) -> bool {
        // The copy toggle happens on the Ctrl PRESS; a Ctrl that turns out to
        // be part of a shortcut takes it back on the release, silently.
        if key != keys::CONTROL || !self.ctrl_toggled {
            return false;
        }
        self.ctrl_toggled = false;
        if !env.vp.ctrl_tapped() {
            self.toggle_copy(env, false);
        }
        true
    }

    fn on_key(&mut self, env: &mut Env, key: i32, _mods: Mods) -> bool {
        if key == keys::CONTROL {
            self.toggle_copy(env, true);
            self.ctrl_toggled = true;
            return true;
        }
        let Some(axis) = arrow_axis(key).map(axis_char) else { return false };
        self.lock_axis = if self.lock_axis == Some(axis) { None } else { Some(axis) };
        true
    }

    fn on_hover(&mut self, env: &mut Env, cx: &ToolCx) {
        self.hover_point = Some(cx.world);
        self.hover_axis = Self::plane_under_cursor(env, cx.screen.0, cx.screen.1);
    }

    fn on_click(&mut self, env: &mut Env, _cx: &ToolCx) {
        let Some(axis) = self.lock_axis.or(self.hover_axis) else { return };
        let copy = self.copy;
        self.flip(env, axis, copy);
        self.copy = false; // the modifier arms ONE flip
    }

    fn on_cancel(&mut self, _env: &mut Env) {
        self.lock_axis = None;
    }

    fn probe(&self, _env: &Env) -> Probe {
        let mut p = Probe::default();
        p.sc = Some(format!("Ctrl = {}", if self.copy { "[copy] / original" } else { "copy / [original]" }));
        p.plus = Some(self.copy);
        p.uses_snap = false;
        p.hover = self.hover_point;
        p.x = Some(Value::Object(vec![
            ("hover_axis".into(), axis_value(self.hover_axis)),
            ("lock_axis".into(), axis_value(self.lock_axis)),
            ("copy".into(), Value::Bool(self.copy)),
            ("ctrl_toggled".into(), Value::Bool(self.ctrl_toggled)),
        ]));
        p
    }
}

