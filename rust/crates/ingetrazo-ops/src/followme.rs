// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! Follow Me applied to the mesh: `core.sweep.sweep_profile` and the commit
//! of `tools/followme.py` (`FollowMeTool._commit`, `_orient_path` and the
//! preselected-path reading of `on_activate`).
//!
//! The geometry (rings, wall quads, caps, the stitch seed) is
//! `ingetrazo_geom::sweep`; this is the part that needs the mesh: consuming
//! the profile face, `Mesh.add_face` with the plan, clearing the profile's
//! orphan boundary edges, `run_stitch` (no coplanar merge), softening the
//! seams of a curved sweep and `orient_outward`.
//!
//! The tool commits through `SnapshotMutation(mutate)`: `capture_state`
//! before, the mutation, `capture_state` after; undo restores the first and
//! redo the second (the mutation is never re-run — it closes over tool
//! state). [`SweepProfileCommand`] is that command. When the sweep declines
//! (degenerate path: a 180 degree reversal, a segment parallel to a joint
//! plane, a zero-length span) the mesh is untouched but the tool still
//! executed a command: it undoes it and clears the redo stack
//! ([`commit_follow_me`] does the same).
//!
//! Python's `set`s in play: the soften pass walks `mesh.edges` (a list;
//! each edge is decided on its own), `before_edges`/`before_faces` are only
//! membership tests (arena ids here; Python's `id(face)` could only collide
//! with a face freed in between, and the only face removed before the walls
//! exist — the profile — is kept alive by the caller), and the selected
//! edges `on_activate` reads come from `scene.selection`, a set of objects:
//! the callers pass them as an ordered list (the fixtures pin creation
//! order, and `order_path_edges` only depends on it for which end a chain
//! starts at, which `orient_path` then settles unless the profile is
//! equidistant to both ends).

use std::cell::Cell;
use std::collections::HashSet;
use std::rc::Rc;

use ingetrazo_geom::meshface::centroid;
use ingetrazo_geom::sweep::{curve_facet_soft, order_path_edges, orient_closed_path, sweep_plan, Profile};
use ingetrazo_math::{Hypot, V3};
use ingetrazo_model::mesh::{EdgeId, FaceId, Mesh, State, VertexId};
use ingetrazo_model::scene::Scene;
use ingetrazo_model::stitch;

use crate::command::{Command, Cx, Fail, History};
use crate::orient;

/// `Profile` of a face: `face.vertices` and `face.holes` as positions.
pub fn profile_of(mesh: &Mesh, face: FaceId) -> Profile {
    let f = mesh.face(face);
    let pos = |l: &[VertexId]| l.iter().map(|&v| mesh.position(v)).collect::<Vec<V3>>();
    Profile { outer: pos(&f.outer), holes: f.holes.iter().map(|h| pos(h)).collect() }
}

/// `sweep_profile(mesh, face, path, closed)`: sweep `face` (with holes) along
/// `path`, in place. `false` (mesh untouched) on degenerate input.
pub fn sweep_profile(mesh: &mut Mesh, face: FaceId, path: &[V3], closed: bool, h: Hypot) -> bool {
    let profile = profile_of(mesh, face);
    let Some(plan) = sweep_plan(&profile, path, closed, h) else {
        return false;
    };
    let before_edges: HashSet<EdgeId> = mesh.edges.iter().copied().collect();
    let before_faces: HashSet<FaceId> = mesh.faces.iter().copied().collect();
    let fc = mesh.face(face);
    let profile_loops: Vec<Vec<VertexId>> = std::iter::once(fc.outer.clone()).chain(fc.holes.iter().cloned()).collect();
    mesh.remove_face(face); // the profile is consumed
    for p in &plan.faces {
        mesh.add_face(&p.outer, &p.holes);
    }
    // The profile's own edges go with it: when station 0 is a mitre (the
    // profile at a corner of a closed path) they no longer bound anything
    // and would stay behind as loose lines.
    for lp in &profile_loops {
        let n = lp.len();
        for i in 0..n {
            if let Some(e) = mesh.find_edge(lp[i], lp[(i + 1) % n]) {
                if mesh.edge(e).faces.is_empty() {
                    mesh.remove_edge(e);
                }
            }
        }
    }
    // `{_key(p) ...}`: the plan carries `round(x, 4)` floats; they are the
    // integer keys over 1e4 (`round4` = round_ties_even(x * 1e4) / 1e4).
    let seed: HashSet<stitch::SeedKey> = plan.seed.iter().map(|k| [0, 1, 2].map(|i| (k[i] * 1e4).round_ties_even() as i64)).collect();
    stitch::run_stitch(mesh, &seed, None, false, true, h);
    // Soften the seams of a curved sweep (shallow dihedral between successive
    // spans) so a moulding around a circle reads smooth; real corners stay.
    // An edge that existed before counts too when the sweep built BOTH of
    // its faces.
    for e in mesh.edges.clone() {
        let ed = mesh.edge(e);
        if ed.soft || ed.faces.len() != 2 {
            continue;
        }
        if before_edges.contains(&e) && ed.faces.iter().any(|f| before_faces.contains(f)) {
            continue;
        }
        let (f0, f1) = (ed.faces[0], ed.faces[1]);
        if curve_facet_soft(mesh.face_normal(f0, h), mesh.face_normal(f1, h), h) {
            mesh.edge_mut(e).soft = true;
        }
    }
    orient::orient_outward(mesh, h);
    true
}

/// `SnapshotMutation(lambda scene: sweep_profile(scene.mesh, ...))`.
pub struct SweepProfileCommand {
    pub face: FaceId,
    pub path: Vec<V3>,
    pub closed: bool,
    /// What `sweep_profile` answered (the tool's `result["ok"]`).
    pub ok: Rc<Cell<bool>>,
    before: Option<State>,
    after: Option<State>,
}

impl SweepProfileCommand {
    pub fn new(face: FaceId, path: Vec<V3>, closed: bool) -> Self {
        SweepProfileCommand { face, path, closed, ok: Rc::new(Cell::new(false)), before: None, after: None }
    }
}

impl Command for SweepProfileCommand {
    fn apply(&mut self, scene: &mut Scene, cx: &mut Cx) -> Result<(), Fail> {
        if self.after.is_none() {
            let h = cx.hypot;
            self.before = Some(cx.mesh(scene).capture_state());
            let ok = sweep_profile(cx.mesh_mut(scene), self.face, &self.path, self.closed, h);
            self.ok.set(ok);
            self.after = Some(cx.mesh(scene).capture_state());
        } else if let Some(s) = &self.after {
            cx.mesh_mut(scene).restore_state(s);
        }
        Ok(())
    }
    fn undo(&mut self, scene: &mut Scene, cx: &mut Cx) {
        if let Some(s) = &self.before {
            cx.mesh_mut(scene).restore_state(s);
        }
    }
    fn name(&self) -> &'static str {
        "SnapshotMutation"
    }
    fn as_any(&self) -> Option<&dyn std::any::Any> {
        Some(self)
    }
}

/// `FollowMeTool._commit` without the viewport: run the sweep as one history
/// step; when it declines, drop the no-op step (undo it, clear redo).
/// Returns whether the sweep happened. The tool then clears the selection.
pub fn commit_follow_me(scene: &mut Scene, history: &mut History, cx: &mut Cx, face: FaceId, path: Vec<V3>, closed: bool) -> bool {
    let cmd = SweepProfileCommand::new(face, path, closed);
    let ok = cmd.ok.clone();
    history.execute(scene, cx, Box::new(cmd));
    if !ok.get() {
        history.undo(scene, cx);
        history.clear_redo();
    }
    ok.get()
}

/// `on_activate` with edges selected: `order_path_edges` of the selected
/// edges (in the order given). `None`: they do not form one simple path.
pub fn path_of_edges(mesh: &Mesh, edges: &[EdgeId]) -> Option<(Vec<V3>, bool)> {
    let n = mesh.arena_sizes().0;
    let pos: Vec<V3> = (0..n).map(|i| mesh.position(VertexId(i as u32))).collect();
    let pairs: Vec<(usize, usize)> = edges
        .iter()
        .map(|&e| {
            let ed = mesh.edge(e);
            (ed.v0.0 as usize, ed.v1.0 as usize)
        })
        .collect();
    order_path_edges(&pairs, &pos)
}

/// `on_activate` with one face selected: its outer boundary, a closed path.
pub fn path_of_face(mesh: &Mesh, face: FaceId) -> (Vec<V3>, bool) {
    (mesh.face(face).outer.iter().map(|&v| mesh.position(v)).collect(), true)
}

/// `FollowMeTool._orient_path`: start the sweep at the path point nearest the
/// profile: reverse an open chain whose far end is closer, rotate a closed
/// loop so station 0 sits next to the profile.
pub fn orient_path(path: &[V3], closed: bool, profile: &Profile, h: Hypot) -> (Vec<V3>, bool) {
    let mut path = path.to_vec();
    if closed {
        return (orient_closed_path(&path, profile, h), true);
    }
    let c = centroid(&profile.outer);
    let last = path[path.len() - 1].sub(c).length(h);
    let first = path[0].sub(c).length(h);
    if last < first {
        path.reverse();
    }
    (path, false)
}
