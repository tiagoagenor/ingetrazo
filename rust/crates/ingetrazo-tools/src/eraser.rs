// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! The Eraser tool (`tools/eraser.py`): erase by clicking or by dragging over
//! geometry.
//!
//! Press marks the edge under the cursor, dragging keeps marking everything
//! the cursor sweeps over (shown red), and release erases the whole stroke as
//! ONE undo step. A segment of a drawn curve erases its whole contour; a group
//! or component is erased whole; guides, dimensions, leader texts and georef
//! paths are erased too. Esc cancels the stroke. Shift at press starts a HIDE
//! stroke: the swept edges and groups are hidden instead (still one step), and
//! a hide stroke ignores guides, dimensions, texts and paths.
//!
//! `marked` is a Python `set` of objects (address order, which no run
//! repeats): here, as in the fixtures, it keeps insertion order.
//!
//! Where it differs from Python (reported):
//!
//! * The red box a marked GROUP draws on the stroke (`_group_box_segments`,
//!   the viewport's `_group_obb`: a numpy convex-hull fit of the group's
//!   points) is not ported: [`rubber_band_lines`](EraserTool::probe) leaves a
//!   group out. The fixtures patch the Python tool the same way.

use ingetrazo_formats::pyjson::Value;
use ingetrazo_math::V3;
use ingetrazo_model::scene::Scene;
use ingetrazo_ops::annotate::{DeleteDimensions, DeleteGuides, DeleteTextLabels};
use ingetrazo_ops::command::Command;
use ingetrazo_ops::draw::{CompoundCommand, EraseSelectionCommand};
use ingetrazo_ops::edit_cmds::{Entity, Hide};
use ingetrazo_ops::group::{group_at, Path};
use ingetrazo_ops::grouping::DeleteGroup;
use ingetrazo_ops::reference::DeleteGeoPaths;

use crate::modify_base::{Env, ModifyTool, Probe, Sel, ToolCx};
use ingetrazo_model::mesh::EdgeId;

/// One marked thing. Annotations are held by their identity in the scene's
/// list, as Python holds the object.
#[derive(Clone, Debug, PartialEq)]
enum Marked {
    Edge(EdgeId),
    Group(Path),
    Guide(u32),
    Dim(u32),
    Geo(u32),
    Label(u32),
}

#[derive(Default)]
pub struct EraserTool {
    stroke: bool,
    hide: bool,
    marked: Vec<Marked>,
    /// `_viewport is not None`.
    has_viewport: bool,
}

impl EraserTool {
    pub fn new() -> EraserTool {
        EraserTool::default()
    }

    fn reset(&mut self) {
        self.stroke = false;
        self.hide = false;
        self.marked.clear();
        self.has_viewport = false;
    }

    fn add(&mut self, m: Marked) {
        if !self.marked.contains(&m) {
            self.marked.push(m);
        }
    }

    /// `_pick`: only an edge the user can see.
    fn pick(env: &Env, x: f64, y: f64) -> Option<EdgeId> {
        env.vp.pick_visible_edge(env.scene, env.cx, x, y)
    }

    /// `_mark`.
    fn mark(&mut self, env: &Env, x: f64, y: f64) {
        self.has_viewport = true;
        if !self.hide {
            // The text block overdraws all geometry: its glyphs outrank every 3D pick.
            if let Some(i) = env.vp.pick_text_label(env.scene, env.cx, x, y, true) {
                if let Some(id) = env.scene.text_labels.id_at(i) {
                    self.add(Marked::Label(id));
                    return;
                }
            }
        }
        if let Some(edge) = EraserTool::pick(env, x, y) {
            // A curve is one entity: marking a segment marks its contour.
            for e in env.cx.mesh(env.scene).curve_edges(edge) {
                self.add(Marked::Edge(e));
            }
            return;
        }
        if let Some(path) = env.vp.pick_group(env.scene, env.cx, x, y) {
            self.add(Marked::Group(path));
            return;
        }
        if self.hide {
            return; // only edges and objects can hide
        }
        if let Some(i) = env.vp.pick_guide(env.scene, env.cx, x, y) {
            if let Some(id) = env.scene.guides.id_at(i) {
                self.add(Marked::Guide(id));
            }
            return;
        }
        if let Some(i) = env.vp.pick_dimension(env.scene, env.cx, x, y) {
            if let Some(id) = env.scene.dimensions.id_at(i) {
                self.add(Marked::Dim(id));
            }
            return;
        }
        // The label's thin leader keeps the normal, post-edge priority.
        if let Some(i) = env.vp.pick_text_label(env.scene, env.cx, x, y, false) {
            if let Some(id) = env.scene.text_labels.id_at(i) {
                self.add(Marked::Label(id));
            }
            return;
        }
        if let Some(i) = env.vp.pick_geopath(env.scene, env.cx, x, y) {
            if let Some(id) = env.scene.geo_paths.id_at(i) {
                self.add(Marked::Geo(id));
            }
        }
    }

    fn run_hide(&self, env: &mut Env, edges: Vec<EdgeId>, groups: Vec<Path>) {
        let mesh = env.cx.mesh;
        let mut cmds: Vec<Box<dyn Command>> = Vec::new();
        let hidden_edges: Vec<Entity> = edges.iter().map(|&e| Entity::Edge(mesh, e)).collect();
        let hidden_groups: Vec<Entity> = groups.iter().map(|p| Entity::Group(p.clone())).collect();
        if !hidden_edges.is_empty() {
            cmds.push(Box::new(Hide::edges(hidden_edges.clone(), true)));
        }
        if !hidden_groups.is_empty() {
            cmds.push(Box::new(Hide::new(hidden_groups.clone(), true)));
        }
        let Some(cmd) = one_or_compound(cmds) else { return };
        env.execute(cmd);
        if env.history.last_error.is_none() {
            // `HideCommand.do`: an invisible entity must not linger in the selection.
            for e in edges {
                env.sel.discard(&Sel::Edge(e));
            }
            for p in groups {
                env.sel.discard(&Sel::Group(p));
            }
        }
    }

    /// Execute the delete commands, keeping the selection the way Python's
    /// `discard`s do: the deleted items leave it and the later siblings of a
    /// deleted group / annotation keep their identity (their path / index
    /// moves).
    fn run_delete(&self, env: &mut Env, cmd: Box<dyn Command>, deleted_groups: &[Path]) {
        let before: Vec<Sel> = env.sel.items.clone();
        let ids: Vec<Option<u32>> = before.iter().map(|s| annotation_id(env.scene, s)).collect();
        env.execute(cmd);
        if env.history.last_error.is_some() {
            return;
        }
        let mut out: Vec<Sel> = Vec::new();
        for (s, id) in before.iter().zip(ids) {
            match s {
                Sel::Face(_) | Sel::Edge(_) => {
                    if env.sel.contains(s) {
                        out.push(s.clone());
                    }
                }
                Sel::Group(p) => {
                    if let Some(np) = remap_group_path(p, deleted_groups) {
                        out.push(Sel::Group(np));
                    }
                }
                Sel::Guide(_) => {
                    if let Some(i) = id.and_then(|id| env.scene.guides.index_of(id)) {
                        out.push(Sel::Guide(i));
                    }
                }
                Sel::Dimension(_) => {
                    if let Some(i) = id.and_then(|id| env.scene.dimensions.index_of(id)) {
                        out.push(Sel::Dimension(i));
                    }
                }
                Sel::Label(_) => {
                    if let Some(i) = id.and_then(|id| env.scene.text_labels.index_of(id)) {
                        out.push(Sel::Label(i));
                    }
                }
                Sel::GeoPath(_) => {
                    if let Some(i) = id.and_then(|id| env.scene.geo_paths.index_of(id)) {
                        out.push(Sel::GeoPath(i));
                    }
                }
                other => out.push(other.clone()),
            }
        }
        env.sel.items = out;
    }
}

fn one_or_compound(mut cmds: Vec<Box<dyn Command>>) -> Option<Box<dyn Command>> {
    match cmds.len() {
        0 => None,
        1 => cmds.pop(),
        _ => Some(Box::new(CompoundCommand { commands: cmds })),
    }
}

/// The identity of a selected annotation.
fn annotation_id(scene: &Scene, s: &Sel) -> Option<u32> {
    match s {
        Sel::Guide(i) => scene.guides.id_at(*i),
        Sel::Dimension(i) => scene.dimensions.id_at(*i),
        Sel::Label(i) => scene.text_labels.id_at(*i),
        Sel::GeoPath(i) => scene.geo_paths.id_at(*i),
        _ => None,
    }
}

/// Where a selected group's path lands once `deleted` are gone; `None` when
/// it is one of them (or lives inside one).
fn remap_group_path(p: &[usize], deleted: &[Path]) -> Option<Path> {
    let mut out = p.to_vec();
    for d in deleted {
        let k = d.len() - 1;
        if p.len() > k && p[..k] == d[..k] {
            if p[k] == d[k] {
                return None;
            }
            if p[k] > d[k] {
                out[k] -= 1;
            }
        }
    }
    Some(out)
}

impl ModifyTool for EraserTool {
    fn on_activate(&mut self, _env: &mut Env) {
        self.reset();
    }

    fn on_deactivate(&mut self, _env: &mut Env) {
        self.reset();
    }

    /// Press: start a stroke and mark whatever is under the cursor. The
    /// modifier latches here: a whole stroke is one gesture, erase or hide.
    fn on_click(&mut self, env: &mut Env, cx: &ToolCx) {
        self.stroke = true;
        self.hide = cx.mods.shift();
        self.mark(env, cx.screen.0, cx.screen.1);
    }

    fn on_hover(&mut self, env: &mut Env, cx: &ToolCx) {
        let (x, y) = cx.screen;
        if self.stroke {
            self.mark(env, x, y);
        } else {
            // Preview what a click would take; a text's glyphs first.
            let label = env.vp.pick_text_label(env.scene, env.cx, x, y, true);
            let ent = match label {
                Some(i) => Some(Sel::Label(i)),
                None => EraserTool::pick(env, x, y).map(Sel::Edge),
            };
            env.vp.set_hover(ent, None);
        }
    }

    /// Release: erase everything the stroke marked, as one undo step.
    fn on_release(&mut self, env: &mut Env) {
        if !self.stroke {
            return;
        }
        let marked = std::mem::take(&mut self.marked);
        self.stroke = false;
        if marked.is_empty() {
            return;
        }
        let edges: Vec<EdgeId> = marked.iter().filter_map(|m| if let Marked::Edge(e) = m { Some(*e) } else { None }).collect();
        let groups: Vec<Path> = marked.iter().filter_map(|m| if let Marked::Group(p) = m { Some(p.clone()) } else { None }).collect();
        if self.hide {
            let mesh = env.cx.mesh(env.scene);
            let edges: Vec<EdgeId> = edges.into_iter().filter(|&e| !mesh.edge(e).hidden).collect();
            let groups: Vec<Path> = groups.into_iter().filter(|p| !group_at(&env.scene.groups, p).is_some_and(|g| g.hidden)).collect();
            self.run_hide(env, edges, groups);
            return;
        }
        let ids = |f: &dyn Fn(&Marked) -> Option<u32>| -> Vec<u32> { marked.iter().filter_map(f).collect() };
        let guides = ids(&|m| if let Marked::Guide(i) = m { Some(*i) } else { None });
        let dims = ids(&|m| if let Marked::Dim(i) = m { Some(*i) } else { None });
        let paths = ids(&|m| if let Marked::Geo(i) = m { Some(*i) } else { None });
        let labels = ids(&|m| if let Marked::Label(i) = m { Some(*i) } else { None });
        let mut cmds: Vec<Box<dyn Command>> = Vec::new();
        if !edges.is_empty() {
            let mesh = env.cx.mesh(env.scene);
            let ends: Vec<(V3, V3)> = edges
                .iter()
                .map(|&e| {
                    let ed = mesh.edge(e);
                    (mesh.position(ed.v0), mesh.position(ed.v1))
                })
                .collect();
            cmds.push(Box::new(EraseSelectionCommand::new(ends, Vec::new())));
        }
        // Each `DeleteGroupCommand` runs after the ones before it: the group
        // it holds sits where the earlier deletions left it.
        let mut done: Vec<Path> = Vec::new();
        for p in &groups {
            let path = remap_group_path(p, &done).unwrap_or_else(|| vec![usize::MAX]);
            cmds.push(Box::new(DeleteGroup::new(&path)));
            done.push(p.clone());
        }
        if !guides.is_empty() {
            cmds.push(Box::new(DeleteGuides::new(guides)));
        }
        if !dims.is_empty() {
            cmds.push(Box::new(DeleteDimensions::new(dims)));
        }
        if !paths.is_empty() {
            cmds.push(Box::new(DeleteGeoPaths::new(paths)));
        }
        if !labels.is_empty() {
            cmds.push(Box::new(DeleteTextLabels::new(labels)));
        }
        if let Some(cmd) = one_or_compound(cmds) {
            self.run_delete(env, cmd, &groups);
        }
    }

    fn on_cancel(&mut self, _env: &mut Env) {
        self.reset();
    }

    fn probe(&self, env: &Env) -> Probe {
        let h = env.cx.hypot;
        let mut p = Probe::default();
        let mesh = env.cx.mesh(env.scene);
        let mut segs: Vec<(V3, V3)> = Vec::new();
        for m in &self.marked {
            match m {
                Marked::Edge(e) => {
                    let ed = mesh.edge(*e);
                    segs.push((mesh.position(ed.v0), mesh.position(ed.v1)));
                }
                Marked::Group(_) => {}
                Marked::Guide(id) => {
                    if let Some(g) = env.scene.guides.get_id(*id) {
                        segs.push(g.segment());
                    }
                }
                Marked::Dim(id) => {
                    if let Some(d) = env.scene.dimensions.get_id(*id) {
                        segs.push(d.line_points(h));
                    }
                }
                Marked::Geo(id) => {
                    if let Some(g) = env.scene.geo_paths.get_id(*id) {
                        segs.extend(g.segments());
                    }
                }
                Marked::Label(id) => {
                    if let Some(t) = env.scene.text_labels.get_id(*id) {
                        segs.push((t.anchor, t.position()));
                    }
                }
            }
        }
        p.rb = Some(segs);
        p.wf = Some([0.90, 0.20, 0.15, 1.0]);
        p.uses_snap = false;
        let count = |f: fn(&Marked) -> bool| Value::int(self.marked.iter().filter(|m| f(m)).count() as i64);
        p.x = Some(Value::Object(vec![
            ("stroke".into(), Value::Bool(self.stroke)),
            ("hide".into(), Value::Bool(self.hide)),
            ("n_marked".into(), Value::int(self.marked.len() as i64)),
            ("n_edges".into(), count(|m| matches!(m, Marked::Edge(_)))),
            ("n_groups".into(), count(|m| matches!(m, Marked::Group(_)))),
            ("n_guides".into(), count(|m| matches!(m, Marked::Guide(_)))),
            ("n_dims".into(), count(|m| matches!(m, Marked::Dim(_)))),
            ("n_geo".into(), count(|m| matches!(m, Marked::Geo(_)))),
            ("n_labels".into(), count(|m| matches!(m, Marked::Label(_)))),
            ("viewport".into(), Value::Bool(self.has_viewport)),
        ]));
        p
    }
}
