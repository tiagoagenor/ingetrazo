// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! The Move tool (`tools/move.py`): the state machine, with no UI.
//!
//! UX (the classic push/pull modeller behaviour): with a selection Move acts
//! on it, otherwise the first click grabs the edge/face/group under the
//! cursor (or, on an Endpoint snap, that one vertex). The second click drops
//! at the current offset; a typed length moves exactly that far along the
//! current direction, `X;Y;Z` is a 3D delta. A tap of Ctrl toggles COPY mode
//! (the original stays, a translated copy is made); right after a copy `3x`
//! makes an external array and `/3` an internal one, and `5x10m` carries the
//! spacing. Esc (or switching tools) cancels and snaps the geometry back.
//!
//! The live preview mutates the scene directly (no history entry): loose
//! vertices and classic groups are moved vertex by vertex, an instance's
//! matrix is composed; a groups-only drag leaves the geometry alone and asks
//! the viewport to draw the translated copy (`begin_groups_preview`...).
//!
//! Where it differs from Python (reported):
//!
//! * The rotation grips of a group under the cursor (issue #115, taken
//!   with a click to hand the turn to a Rotate tool) are not ported: they
//!   need the viewport's oriented box of a group.
//! * Leader texts, dimensions, section planes and reference images are not
//!   moved (Move acts on faces, edges and groups; the selection has no
//!   such items here for it to carry).

use std::cell::Cell;
use std::collections::HashSet;
use std::rc::Rc;

use ingetrazo_formats::pyjson::Value;
use ingetrazo_math::{M4, V3};
use ingetrazo_model::mesh::{EdgeId, FaceId, VertexId};
use ingetrazo_model::scene::Scene;
use ingetrazo_model::units::fmt_len;
use ingetrazo_ops::command::{Command, Cx, Fail};
use ingetrazo_ops::draw::{AddEdgeCommand, AddFaceCommand, CompoundCommand};
use ingetrazo_ops::edits::{build_add_edges, PlannedWhenRun};
use ingetrazo_ops::group::{copy_group, group_at_mut, iter_placements, transformed_attrs, Path};
use ingetrazo_ops::grouping::InsertGroup;
use ingetrazo_ops::transform::{MoveGroup, MoveVertices};
use ingetrazo_model::mesh::Mesh;
use ingetrazo_snap::SnapKind;

use crate::modify_base::{dedup, face_loops, key6, keys, Env, ModifyTool, Mods, Probe, Sel, ToolCx, Vcb};

/// A `MoveVerticesCommand` that tells the tool whether the move joined
/// vertices (`command.welded`, which Python reads off the command it kept).
struct MoveVerticesWelded {
    inner: MoveVertices,
    welded: Rc<Cell<bool>>,
}

impl Command for MoveVerticesWelded {
    fn name(&self) -> &'static str {
        self.inner.name()
    }
    fn apply(&mut self, scene: &mut Scene, cx: &mut Cx) -> Result<(), Fail> {
        let r = self.inner.apply(scene, cx);
        self.welded.set(self.inner.welded);
        r
    }
    fn undo(&mut self, scene: &mut Scene, cx: &mut Cx) {
        self.inner.undo(scene, cx);
    }
}

/// What the copy builder of `_make_copy_builder` closes over, kept by the
/// array window so `3x` / `/3` can re-lay the copies after the tool reset.
#[derive(Clone)]
struct CopyBuilder {
    groups: Vec<Path>,
    faces: Vec<FaceId>,
    edges: Vec<EdgeId>,
    /// The sides of the copied faces, as unordered pairs of rounded points.
    face_sides: Vec<([f64; 3], [f64; 3])>,
}

fn side(a: V3, b: V3) -> ([f64; 3], [f64; 3]) {
    let (a, b) = (key6(a), key6(b));
    if a <= b {
        (a, b)
    } else {
        (b, a)
    }
}

impl CopyBuilder {
    fn new(env: &Env, groups: &[Path], faces: &[FaceId], edges: &[EdgeId]) -> CopyBuilder {
        let m = env.cx.mesh(env.scene);
        let mut face_sides = Vec::new();
        for &f in faces {
            for lp in face_loops(m, f) {
                for i in 0..lp.len() {
                    face_sides.push(side(lp[i], lp[(i + 1) % lp.len()]));
                }
            }
        }
        CopyBuilder { groups: groups.to_vec(), faces: faces.to_vec(), edges: edges.to_vec(), face_sides }
    }

    /// `build(deltas)`: one copy per offset, as one command (or none).
    fn build(&self, env: &mut Env, deltas: &[V3]) -> Option<Box<dyn Command>> {
        let mut cmds: Vec<Box<dyn Command>> = Vec::new();
        let mut plain: Vec<(V3, V3)> = Vec::new();
        let mut bound: Vec<(V3, V3)> = Vec::new();
        for &d in deltas {
            let mut m = M4::identity();
            m.translate(d.0, d.1, d.2);
            for path in &self.groups {
                let Scene { meshes, groups, .. } = &mut *env.scene;
                let Some(g) = group_at_mut(groups, path) else { continue };
                let copy = copy_group(meshes, g, Some(d), false, &mut env.cx.counters);
                cmds.push(Box::new(InsertGroup::new(copy, None)));
            }
            let mesh = env.cx.mesh(env.scene);
            for &f in &self.faces {
                let loops = face_loops(mesh, f);
                let outer: Vec<V3> = loops[0].iter().map(|&v| v.add(d)).collect();
                let holes: Vec<Vec<V3>> = loops[1..].iter().map(|h| h.iter().map(|&v| v.add(d)).collect()).collect();
                let attrs = transformed_attrs(&mesh.face(f).attrs, &mut m);
                cmds.push(Box::new(AddFaceCommand::new(outer, false, if holes.is_empty() { None } else { Some(holes) }, Some(attrs))));
            }
            let mut id_map: Vec<(i64, i64)> = Vec::new();
            for &e in &self.edges {
                let (curve, soft, a, b) = {
                    let ed = mesh.edge(e);
                    (ed.curve, ed.soft, mesh.position(ed.v0), mesh.position(ed.v1))
                };
                if curve.is_none() && !soft {
                    let target = if self.face_sides.contains(&side(a, b)) { &mut bound } else { &mut plain };
                    target.push((a.add(d), b.add(d)));
                    continue;
                }
                // Soft edges and curves keep their flags on the copy.
                if let Some(c) = curve {
                    if !id_map.iter().any(|&(k, _)| k == c) {
                        let fresh = Mesh::next_curve_id(&mut env.cx.counters.curve);
                        id_map.push((c, fresh));
                    }
                }
                let new_curve = curve.and_then(|c| id_map.iter().find(|&&(k, _)| k == c).map(|&(_, v)| v));
                cmds.push(Box::new(AddEdgeCommand::new(a.add(d), b.add(d), if soft { Some(true) } else { None }, new_curve)));
            }
        }
        if !plain.is_empty() {
            cmds.push(build_add_edges(env.scene, env.cx, &plain, true, Vec::new(), None, false));
        }
        if !bound.is_empty() {
            cmds.push(Box::new(PlannedWhenRun::new(bound)));
        }
        if cmds.is_empty() {
            return None;
        }
        Some(if cmds.len() == 1 { cmds.pop().expect("one") } else { Box::new(CompoundCommand { commands: cmds }) })
    }
}

/// `_last`: the copy just made, for `3x` / `/3`.
struct Last {
    serial: Option<u64>,
    builder: CopyBuilder,
    delta: V3,
}

pub struct MoveTool {
    pub start_point: Option<V3>,
    pub grab: Option<V3>,
    pub hover_point: Option<V3>,
    positions: Vec<V3>,
    verts: Vec<VertexId>,
    groups: Vec<Path>,
    preview_delta: V3,
    copy: bool,
    ctrl_toggled: bool,
    sel_faces: Vec<FaceId>,
    sel_edges: Vec<EdgeId>,
    base_segments: Vec<(V3, V3)>,
    last: Option<Last>,
    vp_preview: bool,
}

impl Default for MoveTool {
    fn default() -> Self {
        MoveTool::new()
    }
}

const ZERO: V3 = V3(0.0, 0.0, 0.0);

impl MoveTool {
    pub fn new() -> MoveTool {
        MoveTool {
            start_point: None,
            grab: None,
            hover_point: None,
            positions: Vec::new(),
            verts: Vec::new(),
            groups: Vec::new(),
            preview_delta: ZERO,
            copy: false,
            ctrl_toggled: false,
            sel_faces: Vec::new(),
            sel_edges: Vec::new(),
            base_segments: Vec::new(),
            last: None,
            vp_preview: false,
        }
    }

    pub fn is_copy(&self) -> bool {
        self.copy
    }

    // ---- keyboard ------------------------------------------------------------------------

    fn toggle_copy(&mut self, env: &mut Env, say: bool) {
        self.copy = !self.copy;
        if self.copy {
            // The original stops following the cursor; the copy's wireframe
            // takes over in rubber_band_lines.
            self.revert_preview(env);
            if self.vp_preview {
                env.vp.set_groups_preview_offset(ZERO);
            }
            if say {
                env.flash("Move a copy: on");
            }
        } else {
            if let (Some(grab), Some(hover)) = (self.grab, self.hover_point) {
                self.apply_preview(env, hover.sub(grab));
            }
            if say {
                env.flash("Move a copy: off");
            }
        }
    }

    // ---- gathering what moves ----------------------------------------------------------------

    /// `_gather`: a grab ON a corner takes that one vertex, else
    /// `gather_targets`.
    fn gather(&self, env: &Env, cx: &ToolCx) -> (Vec<Path>, Vec<V3>) {
        if let Some(v) = self.hovered_vertex(env, cx) {
            return (Vec::new(), vec![env.cx.mesh(env.scene).position(v)]);
        }
        gather_targets(env, cx)
    }

    /// `_hovered_vertex`: with nothing selected, a grab on an Endpoint snap
    /// of loose geometry takes that one vertex.
    fn hovered_vertex(&self, env: &Env, cx: &ToolCx) -> Option<VertexId> {
        let snap = cx.snap.as_ref()?;
        if !env.sel.is_empty() || snap.kind != SnapKind::Endpoint || snap.context.is_some() {
            return None;
        }
        env.cx.mesh(env.scene).vertex_at(snap.point)
    }

    /// `_gather_copy_entities`: the faces/edges copy mode duplicates, and the
    /// wireframe segments the copy preview carries.
    fn gather_copy_entities(&mut self, env: &Env, cx: &ToolCx) {
        self.sel_faces.clear();
        self.sel_edges.clear();
        self.base_segments.clear();
        for path in &self.groups {
            let Some(group) = ingetrazo_ops::group::group_at(&env.scene.groups, path) else { continue };
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
            // Nothing selected and no group hover-picked: the click grabbed
            // the loose entity under the cursor (mirror of gather_targets).
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

    // ---- the live preview ---------------------------------------------------------------------

    /// `_shift`: translate the live geometry by `step` - every grabbed group
    /// (an instance composes into its matrix, a classic group walks its
    /// vertices) and the grabbed loose vertices.
    fn shift(&self, env: &mut Env, step: V3) {
        {
            let Scene { meshes, groups, .. } = &mut *env.scene;
            for path in &self.groups {
                let Some(g) = group_at_mut(groups, path) else { continue };
                if let Some(x) = g.xform {
                    let mut t = M4::identity();
                    t.translate(step.0, step.1, step.2);
                    g.xform = Some(t.mul(&x)); // instance: O(1)
                } else {
                    let mesh = &mut meshes[g.mesh.0 as usize];
                    for v in mesh.vertices.clone() {
                        mesh.move_vertex(v, step);
                    }
                }
            }
        }
        let mesh = env.cx.mesh_mut(env.scene);
        for &v in &self.verts {
            mesh.move_vertex(v, step);
        }
        *env.version += 1;
    }

    /// `_apply_preview`: live-deform the scene so the current offset is
    /// `target_delta` from the grab point, by translating the incremental
    /// step.
    fn apply_preview(&mut self, env: &mut Env, target: V3) {
        let step = target.sub(self.preview_delta);
        if (step.length(env.cx.hypot) as f64) < 1e-12 {
            return;
        }
        if self.vp_preview {
            // Groups-only drag: the viewport draws the frozen scratch copy
            // at the offset - no geometry mutation.
            self.preview_delta = target;
            env.vp.set_groups_preview_offset(target);
            return;
        }
        self.shift(env, step);
        self.preview_delta = target;
    }

    /// `_revert_preview`: undo the live deformation, returning geometry to
    /// its grab-time spot.
    fn revert_preview(&mut self, env: &mut Env) {
        if self.vp_preview {
            self.preview_delta = ZERO;
            return;
        }
        if (self.preview_delta.length(env.cx.hypot) as f64) < 1e-12 {
            return;
        }
        let back = self.preview_delta.neg();
        self.shift(env, back);
        self.preview_delta = ZERO;
    }

    fn end_freeze(&self, env: &mut Env) {
        env.vp.end_groups_preview();
    }

    // ---- commit -------------------------------------------------------------------------------------

    fn commit(&mut self, env: &mut Env, delta: V3) {
        let h = env.cx.hypot;
        self.end_freeze(env);
        // Revert the live preview, then apply the move as one undoable
        // command so the history holds a single clean entry.
        self.revert_preview(env);
        if self.copy && (delta.length(h) as f64) > 1e-9 {
            // Copy mode: the original never moved; stamp the copies.
            let builder = CopyBuilder::new(env, &self.groups, &self.sel_faces, &self.sel_edges);
            if let Some(cmd) = builder.build(env, &[delta]) {
                let serial = env.execute_serial(cmd);
                // Right after the copy, "3x" / "/3" make an array.
                self.last = Some(Last { serial, builder, delta });
            }
            self.reset();
            return;
        }
        if (delta.length(h) as f64) > 1e-9 {
            let welded = Rc::new(Cell::new(false));
            let mut commands: Vec<Box<dyn Command>> = Vec::new();
            for g in &self.groups {
                commands.push(Box::new(MoveGroup::new(g, delta)));
            }
            if !self.positions.is_empty() {
                commands.push(Box::new(MoveVerticesWelded { inner: MoveVertices::new(&self.positions, delta), welded: welded.clone() }));
            }
            if commands.len() == 1 {
                env.execute(commands.pop().expect("one"));
            } else if !commands.is_empty() {
                env.execute(Box::new(CompoundCommand { commands }));
            }
            // Say what became of a landing on other vertices: joined, or
            // refused (and the move undone) because the join would break a
            // face.
            if let Some(refused) = env.history.last_rejected.clone() {
                env.flash_for(refused, 8000);
            } else if welded.get() {
                env.flash("Vertices joined");
            }
        }
        self.reset();
    }

    fn reset(&mut self) {
        self.start_point = None;
        self.grab = None;
        self.hover_point = None;
        self.positions.clear();
        self.verts.clear();
        self.groups.clear();
        self.preview_delta = ZERO;
        self.sel_faces.clear();
        self.sel_edges.clear();
        self.base_segments.clear();
        self.vp_preview = false;
        self.copy = false; // the Ctrl modifier arms ONE operation
    }

    /// The edges of the grabbed vertices, for the snap exclusion.
    fn grabbed_edges(&self, env: &Env) -> Vec<EdgeId> {
        let mesh = env.cx.mesh(env.scene);
        let mut seen = HashSet::new();
        let mut out = Vec::new();
        for &v in &self.verts {
            for e in mesh.edges_at(v) {
                if seen.insert(e) {
                    out.push(e);
                }
            }
        }
        out
    }

    /// `moving_face_ids`: the loose faces the drag bends.
    pub fn moving_faces(&self, env: &Env) -> usize {
        if self.grab.is_none() || self.copy {
            return 0;
        }
        let mesh = env.cx.mesh(env.scene);
        let mut faces: HashSet<FaceId> = HashSet::new();
        for &v in &self.verts {
            for e in mesh.edges_at(v) {
                faces.extend(mesh.edge(e).faces.iter().copied());
            }
        }
        faces.len()
    }

    /// `array_pending`: a copy was just made, so `3x` / `/3` would lay an
    /// array.
    pub fn array_pending(&self, env: &Env) -> bool {
        match &self.last {
            Some(last) if self.start_point.is_none() => env.is_top(last.serial),
            _ => false,
        }
    }
}

/// `gather_targets`: what a transform tool acts on, `(groups, positions)`.
/// Every selected (or hovered) group transforms as a unit; the positions are
/// the unique loose coordinates of the selection or of the entity under the
/// cursor.
pub fn gather_targets(env: &Env, cx: &ToolCx) -> (Vec<Path>, Vec<V3>) {
    let mut entities: Vec<Sel> = env.sel.iter().cloned().collect();
    if entities.is_empty() {
        if let Some(group) = env.vp.pick_group(env.scene, env.cx, cx.screen.0, cx.screen.1) {
            return (vec![group], Vec::new());
        }
        if let Some(e) = env.vp.pick_edge(env.scene, env.cx, cx.screen.0, cx.screen.1) {
            entities = vec![Sel::Edge(e)];
        } else if let Some(f) = env.vp.pick_face(env.scene, env.cx, cx.screen.0, cx.screen.1) {
            entities = vec![Sel::Face(f)];
        }
    }
    let groups: Vec<Path> = entities.iter().filter_map(|s| if let Sel::Group(p) = s { Some(p.clone()) } else { None }).collect();
    let mesh = env.cx.mesh(env.scene);
    let mut positions: Vec<V3> = Vec::new();
    for s in &entities {
        match s {
            Sel::Edge(e) => {
                let ed = mesh.edge(*e);
                positions.push(mesh.position(ed.v0));
                positions.push(mesh.position(ed.v1));
            }
            Sel::Face(f) => positions.extend(mesh.face(*f).outer.iter().map(|&v| mesh.position(v))),
            _ => {}
        }
    }
    (groups, dedup(&positions))
}

impl ModifyTool for MoveTool {
    fn on_activate(&mut self, _env: &mut Env) {
        self.reset();
        self.copy = false;
        self.last = None;
    }

    fn on_deactivate(&mut self, env: &mut Env) {
        self.end_freeze(env);
        self.revert_preview(env);
        self.reset();
        self.copy = false;
        self.last = None;
    }

    fn on_key(&mut self, env: &mut Env, key: i32, _mods: Mods) -> bool {
        // Ctrl toggles copy mode (move a copy, the original stays).
        if key == keys::CONTROL {
            self.toggle_copy(env, true);
            self.ctrl_toggled = true;
            return true;
        }
        false
    }

    fn on_key_release(&mut self, env: &mut Env, key: i32) -> bool {
        // The copy toggle happens on the Ctrl PRESS, so the + shows at once;
        // a Ctrl that turns out to be part of a shortcut (Ctrl+Z, Ctrl+C...)
        // takes it back on the release, silently.
        if key != keys::CONTROL || !self.ctrl_toggled {
            return false;
        }
        self.ctrl_toggled = false;
        if !env.vp.ctrl_tapped() {
            self.toggle_copy(env, false);
        }
        true
    }

    fn on_click(&mut self, env: &mut Env, cx: &ToolCx) {
        if self.start_point.is_none() {
            let (groups, positions) = self.gather(env, cx);
            if groups.is_empty() && positions.is_empty() {
                return; // nothing under the cursor / selected to move
            }
            let mesh = env.cx.mesh(env.scene);
            let verts: Vec<VertexId> = positions.iter().filter_map(|&p| mesh.vertex_at(p)).collect();
            self.last = None; // a new move ends the array window
            self.start_point = Some(cx.world);
            self.grab = Some(cx.world);
            self.groups = groups;
            self.positions = positions;
            // The grabbed positions resolved to vertex OBJECTS once: the live
            // preview then moves these identities directly, so dragging
            // through (or onto) a coincident vertex never drags the innocent
            // one along.
            self.verts = verts;
            self.vp_preview = false;
            if !self.groups.is_empty() && self.verts.is_empty() && self.positions.is_empty() {
                // Groups-only drag: viewport-side preview - the movers
                // render from one-time scratch buffers at a translated MVP.
                env.vp.begin_groups_preview(&self.groups, false);
                self.vp_preview = true;
            }
            self.gather_copy_entities(env, cx);
            return;
        }
        let delta = cx.world.sub(self.grab.expect("grab"));
        self.commit(env, delta);
    }

    fn on_hover(&mut self, env: &mut Env, cx: &ToolCx) {
        self.hover_point = Some(cx.world);
        if let (Some(grab), false) = (self.grab, self.copy) {
            self.apply_preview(env, cx.world.sub(grab));
        }
    }

    fn on_value(&mut self, env: &mut Env, value: &Vcb) -> bool {
        let (Some(_), Some(grab)) = (self.start_point, self.grab) else { return false };
        let h = env.cx.hypot;
        let delta = match value {
            Vcb::Tuple(t) => {
                if t.len() != 3 {
                    return false; // a 2-tuple is a rectangle's W x H, not a move delta
                }
                V3(t[0] as f32, t[1] as f32, t[2] as f32)
            }
            // `("abs_len", v)` is a 2-tuple to a tool that does not know it.
            Vcb::AbsLen(_) => return false,
            Vcb::Num(n) => {
                let Some(hover) = self.hover_point else { return false };
                let direction = hover.sub(grab);
                if (direction.length(h) as f64) < 1e-9 {
                    return false;
                }
                direction.normalized(h).mul(*n as f32)
            }
        };
        self.commit(env, delta);
        true
    }

    fn close_array_window(&mut self) {
        // Esc right after a copy: "3x" / "/3" stop applying.
        self.last = None;
    }

    fn on_array_value(&mut self, env: &mut Env, count: i64, mode: char, step: Option<f64>) -> bool {
        let h = env.cx.hypot;
        if let (Some(step), Some(_)) = (step, self.start_point) {
            let (Some(grab), Some(hover)) = (self.grab, self.hover_point) else { return false };
            if !self.copy {
                return false;
            }
            let direction = hover.sub(grab);
            if (direction.length(h) as f64) < 1e-9 {
                return false;
            }
            self.commit(env, direction.normalized(h).mul(step as f32));
            if self.last.is_none() {
                return false;
            }
            return self.on_array_value(env, count, mode, None);
        }
        if let (Some(step), Some(last)) = (step, self.last.as_mut()) {
            let d = last.delta;
            if (d.length(h) as f64) < 1e-9 {
                return false;
            }
            last.delta = d.normalized(h).mul(step as f32);
        }
        if self.last.is_none() || self.start_point.is_some() || count < 1 {
            return false;
        }
        let (on_top, delta) = {
            let last = self.last.as_ref().expect("last");
            (env.is_top(last.serial), last.delta)
        };
        if !on_top {
            self.last = None;
            return false;
        }
        let deltas: Vec<V3> = if mode == '/' {
            (1..=count).map(|k| delta.mul((k as f64 / count as f64) as f32)).collect()
        } else {
            (1..=count).map(|k| delta.mul(k as f32)).collect()
        };
        env.undo();
        let builder = self.last.as_ref().expect("last").builder.clone();
        let Some(cmd) = builder.build(env, &deltas) else {
            self.last = None;
            return false;
        };
        let serial = env.execute_serial(cmd);
        self.last.as_mut().expect("last").serial = serial;
        env.flash(format!("{count} copies"));
        true
    }

    fn on_cancel(&mut self, env: &mut Env) {
        self.end_freeze(env);
        self.revert_preview(env);
        self.reset();
        self.last = None;
    }

    fn probe(&self, env: &Env) -> Probe {
        let h = env.cx.hypot;
        let mut p = Probe::default();
        // The geometry deforms live, so the only extra cue is the move
        // vector from the grab point to the cursor.
        p.rb = Some(match (self.grab, self.hover_point) {
            (Some(grab), Some(hover)) => {
                let mut segs = vec![(grab, hover)];
                if self.copy && !self.base_segments.is_empty() {
                    // Copy mode: the original stays put - preview the
                    // translated COPY as a wireframe at the cursor offset.
                    let d = hover.sub(grab);
                    segs.extend(self.base_segments.iter().map(|&(a, b)| (a.add(d), b.add(d))));
                }
                segs
            }
            _ => Vec::new(),
        });
        p.vl = Some(match (self.grab, self.hover_point) {
            (Some(grab), Some(hover)) => {
                let delta = hover.sub(grab);
                let mid = grab.add(hover).mul(0.5);
                Some((fmt_len(delta.length(h) as f64, env.units), mid))
            }
            _ => None,
        });
        p.plus = Some(self.copy);
        p.unitless = Some(false);
        p.excl = Some(if self.grab.is_none() || self.copy {
            None
        } else {
            let mesh = env.cx.mesh(env.scene);
            let edges: Vec<usize> = self.grabbed_edges(env).iter().filter_map(|e| mesh.edges.iter().position(|x| x == e)).collect();
            if edges.is_empty() && self.groups.is_empty() {
                None
            } else {
                Some((edges, self.groups.clone()))
            }
        });
        p.ap = Some(self.array_pending(env));
        p.start = self.start_point;
        p.hover = self.hover_point;
        let i = |n: usize| Value::int(n as i64);
        let v3 = |q: V3| Value::Array(vec![Value::Str(bits(q.0)), Value::Str(bits(q.1)), Value::Str(bits(q.2))]);
        p.x = Some(Value::Object(vec![
            ("copy".into(), Value::Bool(self.copy)),
            ("grab".into(), self.grab.map(v3).unwrap_or(Value::Null)),
            ("n_pos".into(), i(self.positions.len())),
            ("n_verts".into(), i(self.verts.len())),
            ("n_groups".into(), i(self.groups.len())),
            ("vp_preview".into(), Value::Bool(self.vp_preview)),
            ("preview_delta".into(), v3(self.preview_delta)),
            ("sel_faces".into(), i(self.sel_faces.len())),
            ("sel_edges".into(), i(self.sel_edges.len())),
            ("base_segments".into(), i(self.base_segments.len())),
            ("last".into(), Value::Bool(self.last.is_some())),
            ("hot_grip".into(), Value::Null),
            ("grip_rot".into(), Value::Bool(false)),
            ("moving_faces".into(), i(self.moving_faces(env))),
            ("ctrl_toggled".into(), Value::Bool(self.ctrl_toggled)),
            // The generator cuts a program where a held group left the scene.
            ("stale".into(), Value::Bool(false)),
        ]));
        p
    }
}

fn bits(x: f32) -> String {
    format!("{:08x}", x.to_bits())
}
