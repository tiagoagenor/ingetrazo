// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! The group and component commands of `core/history.py` (M3.10): Make
//! Group / Nested Group / Component, Group in Context, Reparent, Merge,
//! Explode, Split into Pieces, Insert, Delete, Rename, Lock, group paint,
//! Replace Component, Keep in File and Reshare, plus `core/kept_components.py`.
//!
//! Python's commands hold their groups by reference and move them between
//! lists; here a group is a value in the tree. A command that takes a
//! group out of the tree keeps it (or the pieces of it) and puts it back
//! on undo, so what `undo` finds is what `do` left.
//!
//! Lists. The commands that Make Group / Explode inside an open group run
//! on the group's `children` instead of `scene.groups`
//! (`GroupInContextCommand` swaps the two); here they carry a `ctx` path
//! (see [`InContext`]) and index into the list it names.
//!
//! The Qt matrix-flag rule (see `group.rs`): every `m(r, c)` read of a
//! matrix Python holds flags it General, so the reads that touch a matrix
//! that outlives the call (`part_points`, the `transformed_mesh` of
//! `ExplodeGroup`) are mirrored on the stored matrix.

use ingetrazo_formats::pyjson::Value;
use ingetrazo_math::{Hypot, M4, V3};
use ingetrazo_model::mesh::{key, EdgeFlags, Key, Mesh, State};
use ingetrazo_model::scene::{Group, MeshId, Scene};
use ingetrazo_model::{EdgeId, FaceId};

use crate::command::{Command, Counters, Cx, Fail};
use crate::group::{
    affine_rows, carry_axes, copy_group, frame_axes, frame_matrix, group_at, group_at_mut, np_affine, push_mesh, transformed_mesh, truthy_dict,
    world_mesh, Path,
};

fn gone() -> Fail {
    Fail::Failed("the group is no longer in the scene".into())
}

/// `Group.__init__`'s name: the one given, else "Group N" from the counter.
pub fn group_name(counters: &mut Counters, name: Option<&str>) -> String {
    match name {
        Some(n) if !n.is_empty() => n.to_string(),
        _ => {
            let n = counters.group;
            counters.group += 1;
            format!("Group {n}")
        }
    }
}

/// `Group(mesh, name)`: a fresh classic group on `mesh`.
pub fn new_group(mesh: MeshId, name: Option<&str>, counters: &mut Counters) -> Group {
    let mut g = Group::new(mesh, group_name(counters, name));
    g.uid = counters.new_uid();
    g
}

fn group_at_list_mut<'a>(groups: &'a mut Vec<Group>, ctx: &Option<Path>) -> Result<&'a mut Vec<Group>, Fail> {
    match ctx {
        None => Ok(groups),
        Some(p) => group_at_mut(groups, p).map(|g| &mut g.children).ok_or_else(|| Fail::Failed("the open group is no longer in the scene".into())),
    }
}

/// A command that can run on the children of an open group.
pub trait InContext: Command {
    /// `None`: the model's own groups; `Some(path)`: that group's children.
    fn set_context(&mut self, ctx: Option<Path>);
}

fn empty_mesh(meshes: &mut Vec<Mesh>, hypot: Hypot) -> MeshId {
    let mut m = Mesh::new();
    m.hypot = hypot;
    push_mesh(meshes, m)
}

/// `min(...)` over Python floats: the first of equals stays.
fn py_min(vals: impl Iterator<Item = f32>) -> f32 {
    let mut it = vals;
    let mut cur = it.next().unwrap_or(0.0);
    for v in it {
        if v < cur {
            cur = v;
        }
    }
    cur
}

// ---- MakeGroupCommand -------------------------------------------------------------

type FaceLoop = (Vec<V3>, Vec<Vec<V3>>, Vec<(String, Value)>);

/// `MakeGroupCommand`: the selected faces and edges become a new group with
/// a mesh of its own; a component builds it in LOCAL coordinates (origin at
/// the selection's min corner) and is an instance.
pub struct MakeGroup {
    face_loops: Vec<FaceLoop>,
    edge_ends: Vec<(V3, V3)>,
    flagged: Vec<(V3, V3, EdgeFlags)>,
    component: bool,
    name: Option<String>,
    /// `scene.drawing_frame`: the open group's axes, `None` at the top.
    pub frame: Option<M4>,
    ctx: Option<Path>,
    snapshot: Option<State>,
}

fn edge_is_plain(f: &EdgeFlags) -> bool {
    *f == EdgeFlags::default()
}

fn flags_of(m: &Mesh, e: EdgeId) -> EdgeFlags {
    let ed = m.edge(e);
    EdgeFlags { soft: ed.soft, curve: ed.curve, layer: ed.layer.clone(), hidden: ed.hidden }
}

fn stamp(m: &mut Mesh, e: EdgeId, f: &EdgeFlags) {
    let ed = m.edge_mut(e);
    ed.soft = f.soft;
    ed.curve = f.curve;
    ed.layer = f.layer.clone();
    ed.hidden = f.hidden;
}

impl MakeGroup {
    pub fn new(scene: &Scene, faces: &[(MeshId, FaceId)], edges: &[(MeshId, EdgeId)], component: bool, name: Option<String>) -> Self {
        let mut face_loops: Vec<FaceLoop> = Vec::new();
        let mut flagged: Vec<(V3, V3, EdgeFlags)> = Vec::new();
        for &(mid, f) in faces {
            let m = scene.mesh(mid);
            let face = m.face(f);
            let pos = |l: &[ingetrazo_model::VertexId]| -> Vec<V3> { l.iter().map(|&v| m.position(v)).collect() };
            face_loops.push((pos(&face.outer), face.holes.iter().map(|h| pos(h)).collect(), face.attrs.clone()));
        }
        let mut note = |m: &Mesh, e: Option<EdgeId>| {
            if let Some(e) = e {
                let fl = flags_of(m, e);
                if !edge_is_plain(&fl) {
                    let ed = m.edge(e);
                    flagged.push((m.position(ed.v0), m.position(ed.v1), fl));
                }
            }
        };
        for &(mid, f) in faces {
            let m = scene.mesh(mid);
            let face = m.face(f);
            for lp in std::iter::once(&face.outer).chain(face.holes.iter()) {
                let n = lp.len();
                for i in 0..n {
                    note(m, m.find_edge(lp[i], lp[(i + 1) % n]));
                }
            }
        }
        let mut edge_ends = Vec::new();
        for &(mid, e) in edges {
            let m = scene.mesh(mid);
            let ed = m.edge(e);
            edge_ends.push((m.position(ed.v0), m.position(ed.v1)));
            note(m, Some(e));
        }
        MakeGroup { face_loops, edge_ends, flagged, component, name, frame: None, ctx: None, snapshot: None }
    }

    /// `do`, on the list `ctx` names.
    fn apply_in(&mut self, scene: &mut Scene, cx: &mut Cx) -> Result<(), Fail> {
        let Scene { meshes, groups, .. } = scene;
        let active = cx.mesh.0 as usize;
        let h = meshes[active].hypot;
        self.snapshot = Some(meshes[active].capture_state());
        let mut origin = V3(0.0, 0.0, 0.0);
        let mut pts: Vec<V3> = Vec::new();
        for (lp, holes, _) in &self.face_loops {
            pts.extend(lp.iter().copied());
            for hl in holes {
                pts.extend(hl.iter().copied());
            }
        }
        for (a, b) in &self.edge_ends {
            pts.push(*a);
            pts.push(*b);
        }
        let mut rot: Option<(V3, V3, V3)> = None;
        let mut corner = V3(0.0, 0.0, 0.0);
        if let (Some(frame), false) = (&self.frame, pts.is_empty()) {
            let (fo, fx, fy, fz) = frame_axes(Some(frame), h);
            let ls: Vec<V3> = pts
                .iter()
                .map(|&p| {
                    let d = p.sub(fo);
                    V3(d.dot(fx), d.dot(fy), d.dot(fz))
                })
                .collect();
            let lo = (py_min(ls.iter().map(|q| q.0)), py_min(ls.iter().map(|q| q.1)), py_min(ls.iter().map(|q| q.2)));
            corner = fo.add(fx.mul(lo.0)).add(fy.mul(lo.1)).add(fz.mul(lo.2));
            rot = Some((fx, fy, fz));
        }
        if self.component && !pts.is_empty() {
            origin = match rot {
                Some(_) => corner,
                None => V3(py_min(pts.iter().map(|p| p.0)), py_min(pts.iter().map(|p| p.1)), py_min(pts.iter().map(|p| p.2))),
            };
        }
        let component = self.component;
        let local = move |p: V3| -> V3 {
            match (component, rot) {
                (true, Some((x, y, z))) => {
                    let d = p.sub(origin);
                    V3(d.dot(x), d.dot(y), d.dot(z))
                }
                _ => p.sub(origin),
            }
        };
        let mut gmesh = Mesh::new();
        gmesh.hypot = h;
        for (lp, holes, attrs) in &self.face_loops {
            let outer: Vec<V3> = lp.iter().map(|&p| local(p)).collect();
            let hs: Vec<Vec<V3>> = holes.iter().map(|hl| hl.iter().map(|&p| local(p)).collect()).collect();
            let gf = gmesh.add_face(&outer, &hs);
            if !attrs.is_empty() {
                let fa = &mut gmesh.face_mut(gf).attrs;
                for (k, v) in attrs {
                    ingetrazo_model::pyval::set(fa, k, v.clone());
                }
            }
        }
        for (a, b) in &self.edge_ends {
            let (la, lb) = (local(*a), local(*b));
            if key(la) == key(lb) {
                continue; // a zero-length edge is no edge (#341)
            }
            let _ = gmesh.add_edge(la, lb);
        }
        for (a, b, fl) in &self.flagged {
            let (va, vb) = (gmesh.vertex_at(local(*a)), gmesh.vertex_at(local(*b)));
            let e = match (va, vb) {
                (Some(x), Some(y)) => gmesh.find_edge(x, y),
                _ => None,
            };
            if let Some(e) = e {
                stamp(&mut gmesh, e, fl);
            }
        }
        let gid = push_mesh(meshes, gmesh);
        let mut group = new_group(gid, self.name.as_deref(), &mut cx.counters);
        if self.component {
            group.xform = Some(match rot {
                Some((x, y, z)) => frame_matrix(origin, x, y, z),
                None => {
                    let mut t = M4::identity();
                    t.translate(origin.0, origin.1, origin.2);
                    t
                }
            });
        } else if let Some((x, y, z)) = rot {
            group.axes = Some(frame_matrix(corner, x, y, z));
        }
        // Remove the grouped geometry from the loose mesh.
        let m = &mut meshes[active];
        let face_keysets: Vec<std::collections::HashSet<Key>> = self.face_loops.iter().map(|(lp, _, _)| lp.iter().map(|&p| key(p)).collect()).collect();
        for f in m.faces.clone() {
            let ks: std::collections::HashSet<Key> = m.face(f).outer.iter().map(|&v| key(m.position(v))).collect();
            if face_keysets.contains(&ks) {
                m.remove_face(f);
            }
        }
        let grouped: std::collections::HashSet<Key> =
            self.face_loops.iter().flat_map(|(lp, holes, _)| lp.iter().chain(holes.iter().flatten())).map(|&p| key(p)).collect();
        let sel_edges: Vec<std::collections::HashSet<Key>> = self.edge_ends.iter().map(|(a, b)| [key(*a), key(*b)].into_iter().collect()).collect();
        for e in m.edges.clone() {
            let ed = m.edge(e);
            let (ka, kb) = (key(m.position(ed.v0)), key(m.position(ed.v1)));
            let ek: std::collections::HashSet<Key> = [ka, kb].into_iter().collect();
            if ed.faces.is_empty() && (sel_edges.contains(&ek) || (grouped.contains(&ka) && grouped.contains(&kb))) {
                m.remove_edge(e);
            }
        }
        group_at_list_mut(groups, &self.ctx)?.push(group);
        Ok(())
    }

    /// `undo`: the group out of its list, the mesh as it was.
    fn undo_in(&mut self, scene: &mut Scene, cx: &Cx) {
        let Scene { meshes, groups, .. } = scene;
        if let Ok(list) = group_at_list_mut(groups, &self.ctx) {
            list.pop();
        }
        if let Some(s) = &self.snapshot {
            meshes[cx.mesh.0 as usize].restore_state(s);
        }
    }
}

impl Command for MakeGroup {
    fn name(&self) -> &'static str {
        "make_group"
    }
    fn apply(&mut self, scene: &mut Scene, cx: &mut Cx) -> Result<(), Fail> {
        self.apply_in(scene, cx)
    }
    fn undo(&mut self, scene: &mut Scene, cx: &mut Cx) {
        self.undo_in(scene, cx);
    }
}

impl InContext for MakeGroup {
    fn set_context(&mut self, ctx: Option<Path>) {
        self.ctx = ctx;
    }
}

// ---- MakeNestedGroupCommand / MakeComponentOfCommand ------------------------------------

/// `MakeNestedGroupCommand`: Make Group when the selection already holds
/// groups. `groups` are indices into the list the command runs on.
pub struct MakeNestedGroup {
    inner: Option<MakeGroup>,
    groups: Vec<usize>,
    name: Option<String>,
    ctx: Option<Path>,
    container: Option<Group>,
    indices: Vec<usize>,
    xform0: Option<M4>,
}

impl MakeNestedGroup {
    pub fn new(scene: &Scene, faces: &[(MeshId, FaceId)], edges: &[(MeshId, EdgeId)], groups: &[usize], name: Option<String>) -> Self {
        let inner = if !faces.is_empty() || !edges.is_empty() { Some(MakeGroup::new(scene, faces, edges, false, name.clone())) } else { None };
        MakeNestedGroup { inner, groups: groups.to_vec(), name, ctx: None, container: None, indices: Vec::new(), xform0: None }
    }

    fn apply_in(&mut self, scene: &mut Scene, cx: &mut Cx) -> Result<(), Fail> {
        let mut container = if let Some(inner) = &mut self.inner {
            inner.ctx = self.ctx.clone();
            inner.apply_in(scene, cx)?;
            let Scene { groups, .. } = scene;
            group_at_list_mut(groups, &self.ctx)?.pop().ok_or_else(gone)?
        } else if let Some(c) = self.container.take() {
            c
        } else {
            let Scene { meshes, .. } = scene;
            let h = meshes[0].hypot;
            let id = empty_mesh(meshes, h);
            new_group(id, self.name.as_deref(), &mut cx.counters)
        };
        let Scene { groups, .. } = scene;
        let list = group_at_list_mut(groups, &self.ctx)?;
        self.xform0 = container.xform;
        let mut idx: Vec<usize> = self.groups.clone();
        idx.sort_unstable();
        idx.dedup();
        if idx.iter().any(|&i| i >= list.len()) {
            list.push(container);
            return Err(Fail::Failed("group is not in the list".into()));
        }
        // In document order, not selection order.
        let mut taken: Vec<Group> = idx.iter().rev().map(|&i| list.remove(i)).collect();
        taken.reverse();
        self.indices = idx;
        container.adopt(taken);
        // A GROUP of groups: `adopt` gives it a matrix, which does not make
        // it a component (issue #90).
        container.component = false;
        list.push(container);
        Ok(())
    }

    fn undo_in(&mut self, scene: &mut Scene, cx: &Cx) {
        let Scene { groups, .. } = scene;
        let Ok(list) = group_at_list_mut(groups, &self.ctx) else { return };
        let Some(last) = list.last_mut() else { return };
        let kids = std::mem::take(&mut last.children);
        last.xform = self.xform0;
        for (idx, g) in self.indices.iter().zip(kids) {
            let at = (*idx).min(list.len());
            list.insert(at, g);
        }
        if let Some(inner) = &mut self.inner {
            inner.undo_in(scene, cx); // loose geometry back, group gone
        } else {
            self.container = list.pop();
        }
    }
}

impl Command for MakeNestedGroup {
    fn name(&self) -> &'static str {
        "make_nested_group"
    }
    fn apply(&mut self, scene: &mut Scene, cx: &mut Cx) -> Result<(), Fail> {
        self.apply_in(scene, cx)
    }
    fn undo(&mut self, scene: &mut Scene, cx: &mut Cx) {
        self.undo_in(scene, cx);
    }
}

impl InContext for MakeNestedGroup {
    fn set_context(&mut self, ctx: Option<Path>) {
        self.ctx = ctx;
    }
}

/// `MakeComponentOfCommand`: ONE component holding the selection, each
/// group still a group inside.
pub struct MakeComponentOf {
    nest: MakeNestedGroup,
}

impl MakeComponentOf {
    pub fn new(scene: &Scene, faces: &[(MeshId, FaceId)], edges: &[(MeshId, EdgeId)], groups: &[usize], name: Option<String>) -> Self {
        MakeComponentOf { nest: MakeNestedGroup::new(scene, faces, edges, groups, name) }
    }
}

impl Command for MakeComponentOf {
    fn name(&self) -> &'static str {
        "make_component_of"
    }
    fn apply(&mut self, scene: &mut Scene, cx: &mut Cx) -> Result<(), Fail> {
        self.nest.apply_in(scene, cx)?;
        let Scene { groups, .. } = scene;
        if let Some(c) = group_at_list_mut(groups, &self.nest.ctx)?.last_mut() {
            c.component = true;
        }
        Ok(())
    }
    fn undo(&mut self, scene: &mut Scene, cx: &mut Cx) {
        self.nest.undo_in(scene, cx);
    }
}

impl InContext for MakeComponentOf {
    fn set_context(&mut self, ctx: Option<Path>) {
        self.nest.ctx = ctx;
    }
}

// ---- GroupInContextCommand -------------------------------------------------------------------

/// `GroupInContextCommand`: Make Group, Make Component or Explode while a
/// group is open: it runs on the open group's children, and the open group
/// ends up holding groups (identity matrix, a group, not a component).
pub struct GroupInContext {
    inner: Box<dyn InContext>,
    context: Path,
    state: Option<(Option<M4>, bool)>,
}

impl GroupInContext {
    pub fn new(inner: Box<dyn InContext>, context: &[usize]) -> Self {
        GroupInContext { inner, context: context.to_vec(), state: None }
    }
}

impl Command for GroupInContext {
    fn name(&self) -> &'static str {
        "group_in_context"
    }
    fn apply(&mut self, scene: &mut Scene, cx: &mut Cx) -> Result<(), Fail> {
        let g = group_at(&scene.groups, &self.context).ok_or_else(gone)?;
        self.state = Some((g.xform, g.component));
        self.inner.set_context(Some(self.context.clone()));
        self.inner.apply(scene, cx)?;
        let g = group_at_mut(&mut scene.groups, &self.context).ok_or_else(gone)?;
        if !g.children.is_empty() && g.xform.is_none() {
            g.xform = Some(M4::identity());
            g.component = false;
        }
        Ok(())
    }
    fn undo(&mut self, scene: &mut Scene, cx: &mut Cx) {
        self.inner.undo(scene, cx);
        if let (Some(g), Some((x, c))) = (group_at_mut(&mut scene.groups, &self.context), self.state) {
            g.xform = x;
            g.component = c;
        }
    }
}

// ---- ReparentGroupsCommand -----------------------------------------------------------------------

/// `outliner.world_matrix`: the matrix that takes the group's coordinates to
/// the world, `None` when nothing on its chain is transformed.
pub fn world_matrix(groups: &[Group], path: &[usize]) -> Option<M4> {
    let mut m: Option<M4> = None;
    let mut level = groups;
    for &i in path {
        let g = level.get(i)?;
        m = match (g.xform, m) {
            (x, None) => x,
            (None, b) => b,
            (Some(x), Some(b)) => Some(b.mul(&x)),
        };
        level = &g.children;
    }
    m
}

/// A path after the group at `removed` left its list: later siblings (and
/// whatever is under them) move one place up.
fn shift_after_removal(path: &mut [usize], removed: &[usize]) {
    let n = removed.len();
    if n > 0 && path.len() >= n && path[..n - 1] == removed[..n - 1] && removed[n - 1] < path[n - 1] {
        path[n - 1] -= 1;
    }
}

struct Move {
    /// The parent it came from (`None`: the model) and its index there.
    from: Option<Path>,
    index: usize,
    /// Where it sits after the move: the target's path (after the removal).
    into: Option<Path>,
    xform: Option<M4>,
    mesh: MeshId,
}

/// `ReparentGroupsCommand`: move groups into another group (or out to the
/// model with `target = None`) without moving them in the world.
pub struct ReparentGroups {
    groups: Vec<Path>,
    target: Option<Path>,
    moves: Vec<Move>,
    target_state: Option<(Option<M4>, bool)>,
    /// The target's path before anything moved, to restore it on undo.
    target_at: Option<Path>,
}

impl ReparentGroups {
    pub fn new(groups: &[Path], target: Option<Path>) -> Self {
        ReparentGroups { groups: groups.to_vec(), target, moves: Vec::new(), target_state: None, target_at: None }
    }
}

impl Command for ReparentGroups {
    fn name(&self) -> &'static str {
        "reparent_groups"
    }

    fn apply(&mut self, scene: &mut Scene, cx: &mut Cx) -> Result<(), Fail> {
        let Scene { meshes, groups, .. } = scene;
        // `topmost`: the chosen groups not inside another chosen one, in
        // document order (depth first = lexicographic by path).
        let mut chosen = self.groups.clone();
        chosen.sort();
        chosen.dedup();
        let mut moving: Vec<Path> =
            chosen.iter().filter(|p| !chosen.iter().any(|q| q.len() < p.len() && p[..q.len()] == q[..])).cloned().collect();
        self.target_state = None;
        self.target_at = self.target.clone();
        if let Some(t) = &self.target {
            let g = group_at_mut(groups, t).ok_or_else(gone)?;
            self.target_state = Some((g.xform, g.component));
            if g.xform.is_none() {
                g.xform = Some(M4::identity()); // same place: identity
                g.component = false;
            }
        }
        let mut target = self.target.clone();
        let to_new = target.as_ref().and_then(|t| world_matrix(groups, t));
        let inv_new = to_new.map(|m| m.inverted().0);
        self.moves.clear();
        let mut i = 0;
        while i < moving.len() {
            let path = moving[i].clone();
            let (index, from) = (*path.last().ok_or_else(gone)?, if path.len() > 1 { Some(path[..path.len() - 1].to_vec()) } else { None });
            let from_old = from.as_ref().and_then(|p| world_matrix(groups, p));
            let delta = match (inv_new, from_old) {
                (None, b) => b,
                (a, None) => a,
                (Some(a), Some(b)) => Some(a.mul(&b)),
            };
            let g = group_at_mut(groups, &path).ok_or_else(gone)?;
            let (old_x, old_mesh) = (g.xform, g.mesh);
            if let Some(mut d) = delta {
                if !d.is_identity() {
                    match g.xform {
                        Some(x) => g.xform = Some(d.mul(&x)),
                        None => {
                            let copy = transformed_mesh(&meshes[g.mesh.0 as usize], &mut d, &mut cx.counters.curve);
                            g.mesh = push_mesh(meshes, copy);
                        }
                    }
                }
            }
            // siblings.remove(g)
            let siblings = match &from {
                None => &mut *groups,
                Some(p) => &mut group_at_mut(groups, p).ok_or_else(gone)?.children,
            };
            let moved = siblings.remove(index);
            for q in moving.iter_mut().skip(i + 1) {
                shift_after_removal(q, &path);
            }
            if let Some(t) = target.as_mut() {
                shift_after_removal(t, &path);
            }
            match &target {
                Some(t) => group_at_mut(groups, t).ok_or_else(gone)?.children.push(moved),
                None => groups.push(moved),
            }
            self.moves.push(Move { from, index, into: target.clone(), xform: old_x, mesh: old_mesh });
            i += 1;
        }
        Ok(())
    }

    fn undo(&mut self, scene: &mut Scene, _cx: &mut Cx) {
        let Scene { groups, .. } = scene;
        for mv in self.moves.iter().rev() {
            let list = match &mv.into {
                Some(t) => match group_at_mut(groups, t) {
                    Some(g) => &mut g.children,
                    None => continue,
                },
                None => &mut *groups,
            };
            let Some(mut g) = list.pop() else { continue };
            g.xform = mv.xform;
            g.mesh = mv.mesh;
            let dest = match &mv.from {
                Some(p) => match group_at_mut(groups, p) {
                    Some(parent) => &mut parent.children,
                    None => continue,
                },
                None => &mut *groups,
            };
            let at = mv.index.min(dest.len());
            dest.insert(at, g);
        }
        if let (Some(t), Some((x, c))) = (&self.target_at, self.target_state) {
            if let Some(g) = group_at_mut(groups, t) {
                g.xform = x;
                g.component = c;
            }
        }
    }
}

// ---- LockCommand / RenameGroupCommand / SetGroupMaterialCommand / DeleteGroupCommand ----------------

/// `LockCommand`.
pub struct Lock {
    groups: Vec<Path>,
    locked: bool,
    prev: Vec<bool>,
}

impl Lock {
    pub fn new(groups: &[Path], locked: bool) -> Self {
        Lock { groups: groups.to_vec(), locked, prev: Vec::new() }
    }
}

impl Command for Lock {
    fn name(&self) -> &'static str {
        "lock"
    }
    fn apply(&mut self, scene: &mut Scene, _cx: &mut Cx) -> Result<(), Fail> {
        self.prev = self.groups.iter().map(|p| group_at(&scene.groups, p).map(|g| g.locked).unwrap_or(false)).collect();
        for p in &self.groups {
            group_at_mut(&mut scene.groups, p).ok_or_else(gone)?.locked = self.locked;
        }
        Ok(())
    }
    fn undo(&mut self, scene: &mut Scene, _cx: &mut Cx) {
        for (p, prev) in self.groups.iter().zip(&self.prev) {
            if let Some(g) = group_at_mut(&mut scene.groups, p) {
                g.locked = *prev;
            }
        }
    }
}

/// `RenameGroupCommand`.
pub struct RenameGroup {
    path: Path,
    name: String,
    old: Option<String>,
}

impl RenameGroup {
    pub fn new(path: &[usize], name: String) -> Self {
        RenameGroup { path: path.to_vec(), name, old: None }
    }
}

impl Command for RenameGroup {
    fn name(&self) -> &'static str {
        "rename_group"
    }
    fn apply(&mut self, scene: &mut Scene, _cx: &mut Cx) -> Result<(), Fail> {
        let g = group_at_mut(&mut scene.groups, &self.path).ok_or_else(gone)?;
        self.old = Some(std::mem::replace(&mut g.name, self.name.clone()));
        Ok(())
    }
    fn undo(&mut self, scene: &mut Scene, _cx: &mut Cx) {
        if let (Some(g), Some(old)) = (group_at_mut(&mut scene.groups, &self.path), &self.old) {
            g.name = old.clone();
        }
    }
}

/// `SetGroupMaterialCommand`: the container's own paint (`None` unpaints).
pub struct SetGroupMaterial {
    path: Path,
    material: Option<Value>,
    old: Option<Option<Value>>,
}

impl SetGroupMaterial {
    pub fn new(path: &[usize], material: Option<Value>) -> Self {
        // `dict(material) if material else None`
        let material = material.filter(|m| !matches!(m, Value::Object(o) if o.is_empty()) && !matches!(m, Value::Null));
        SetGroupMaterial { path: path.to_vec(), material, old: None }
    }
}

impl Command for SetGroupMaterial {
    fn name(&self) -> &'static str {
        "set_group_material"
    }
    fn apply(&mut self, scene: &mut Scene, _cx: &mut Cx) -> Result<(), Fail> {
        let g = group_at_mut(&mut scene.groups, &self.path).ok_or_else(gone)?;
        self.old = Some(std::mem::replace(&mut g.material, self.material.clone()));
        Ok(())
    }
    fn undo(&mut self, scene: &mut Scene, _cx: &mut Cx) {
        if let (Some(g), Some(old)) = (group_at_mut(&mut scene.groups, &self.path), &self.old) {
            g.material = old.clone();
        }
    }
}

/// `DeleteGroupCommand`: the group out of the list it lives in (the model's,
/// or a container's `children`); undo puts it back at its place.
pub struct DeleteGroup {
    path: Path,
    removed: Option<Group>,
}

impl DeleteGroup {
    pub fn new(path: &[usize]) -> Self {
        DeleteGroup { path: path.to_vec(), removed: None }
    }

    fn owner<'a>(groups: &'a mut Vec<Group>, path: &[usize]) -> Option<&'a mut Vec<Group>> {
        if path.len() <= 1 {
            Some(groups)
        } else {
            group_at_mut(groups, &path[..path.len() - 1]).map(|g| &mut g.children)
        }
    }
}

impl Command for DeleteGroup {
    fn name(&self) -> &'static str {
        "delete_group"
    }
    fn apply(&mut self, scene: &mut Scene, _cx: &mut Cx) -> Result<(), Fail> {
        let i = *self.path.last().ok_or_else(|| Fail::Failed("group is not in the scene".into()))?;
        if group_at(&scene.groups, &self.path).is_none() {
            return Err(Fail::Failed("group is not in the scene".into()));
        }
        let owner = DeleteGroup::owner(&mut scene.groups, &self.path).ok_or_else(gone)?;
        self.removed = Some(owner.remove(i));
        Ok(())
    }
    fn undo(&mut self, scene: &mut Scene, _cx: &mut Cx) {
        let (Some(g), Some(&i)) = (self.removed.take(), self.path.last()) else { return };
        if let Some(owner) = DeleteGroup::owner(&mut scene.groups, &self.path) {
            let at = i.min(owner.len());
            owner.insert(at, g);
        }
    }
}

// ---- MergeGroupsCommand ---------------------------------------------------------------------------------

/// `MergeGroupsCommand`: several groups fused into ONE new group holding
/// their combined world-space geometry, through the bulk weld.
pub struct MergeGroups {
    groups: Vec<usize>,
    merged: Option<Group>,
    indices: Vec<usize>,
    originals: Vec<Group>,
}

impl MergeGroups {
    /// `groups`: indices into `scene.groups`.
    pub fn new(groups: &[usize]) -> Self {
        MergeGroups { groups: groups.to_vec(), merged: None, indices: Vec::new(), originals: Vec::new() }
    }
}

impl Command for MergeGroups {
    fn name(&self) -> &'static str {
        "merge_groups"
    }

    fn apply(&mut self, scene: &mut Scene, cx: &mut Cx) -> Result<(), Fail> {
        let Scene { meshes, groups, .. } = scene;
        if self.groups.is_empty() || self.groups.iter().any(|&i| i >= groups.len()) {
            return Err(Fail::Failed("group is not in the scene".into()));
        }
        if self.merged.is_none() {
            let mut flat: Vec<V3> = Vec::new();
            let mut e_flat: Vec<V3> = Vec::new();
            let mut e_flags: Vec<EdgeFlags> = Vec::new();
            let mut ring_sizes: Vec<usize> = Vec::new();
            let mut ring_counts: Vec<usize> = Vec::new();
            let mut attrs_list: Vec<Vec<(String, Value)>> = Vec::new();
            for &gi in &self.groups {
                // `world_mesh` flags the matrices it reads, on the group itself.
                let src = world_mesh(meshes, &mut groups[gi], &mut cx.counters.curve);
                for &f in &src.faces {
                    let face = src.face(f);
                    ring_counts.push(1 + face.holes.len());
                    ring_sizes.push(face.outer.len());
                    flat.extend(face.outer.iter().map(|&v| src.position(v)));
                    for hl in &face.holes {
                        ring_sizes.push(hl.len());
                        flat.extend(hl.iter().map(|&v| src.position(v)));
                    }
                    attrs_list.push(face.attrs.clone());
                }
                for &e in &src.edges {
                    let ed = src.edge(e);
                    e_flat.push(src.position(ed.v0));
                    e_flat.push(src.position(ed.v1));
                    e_flags.push(EdgeFlags { soft: ed.soft, curve: ed.curve, layer: ed.layer.clone(), hidden: ed.hidden });
                }
            }
            let mut mesh = Mesh::new();
            mesh.hypot = meshes[0].hypot;
            let n_edge_pts = e_flat.len();
            if !e_flat.is_empty() || !flat.is_empty() {
                let all: Vec<V3> = e_flat.iter().chain(flat.iter()).copied().collect();
                let (vobjs, inverse) = mesh.bulk_weld(&all);
                let mut emap = None;
                if !e_flat.is_empty() {
                    let ia: Vec<usize> = (0..n_edge_pts).step_by(2).map(|i| inverse[i]).collect();
                    let ib: Vec<usize> = (1..n_edge_pts).step_by(2).map(|i| inverse[i]).collect();
                    emap = Some(mesh.add_edges_welded(&vobjs, &ia, &ib, Some(&e_flags)));
                }
                if !ring_counts.is_empty() {
                    mesh.add_faces_welded(&vobjs, &inverse[n_edge_pts..], &ring_sizes, &ring_counts, Some(&attrs_list), emap);
                }
            }
            let id = push_mesh(meshes, mesh);
            let first = &groups[self.groups[0]];
            let mut merged = new_group(id, Some(&first.name), &mut cx.counters);
            merged.layer = first.layer.clone();
            self.merged = Some(merged);
        }
        let mut order: Vec<usize> = self.groups.clone();
        order.sort_unstable();
        order.dedup();
        let mut removed: Vec<(usize, Group)> = order.iter().rev().map(|&i| (i, groups.remove(i))).collect();
        removed.reverse();
        self.originals = removed.into_iter().map(|(_, g)| g).collect();
        self.indices = order;
        groups.push(self.merged.clone().ok_or_else(gone)?);
        Ok(())
    }

    fn undo(&mut self, scene: &mut Scene, _cx: &mut Cx) {
        let groups = &mut scene.groups;
        if let Some(m) = groups.pop() {
            self.merged = Some(m);
        }
        for (idx, g) in self.indices.iter().zip(std::mem::take(&mut self.originals)) {
            let at = (*idx).min(groups.len());
            groups.insert(at, g);
        }
    }
}

// ---- ExplodeGroupCommand -------------------------------------------------------------------------------------

/// `effective_attrs` + the copy `ExplodeGroup` makes of what a face showed:
/// its own paint, or — a default face — the group's on both sides.
fn drawn_attrs(attrs: &[(String, Value)], paint: Option<&Value>) -> Vec<(String, Value)> {
    let own = |a: &[(String, Value)]| a.iter().any(|(k, v)| (k == "texture" || k == "color") && !matches!(v, Value::Null));
    let paint = match paint {
        Some(Value::Object(o)) if !o.is_empty() => o,
        _ => return attrs.to_vec(),
    };
    if own(attrs) {
        return attrs.to_vec();
    }
    let mut merged = attrs.to_vec();
    for key in ["color", "texture", "opacity", "mat"] {
        if let Some((_, v)) = paint.iter().rev().find(|(k, _)| k == key) {
            if !matches!(v, Value::Null) {
                ingetrazo_model::pyval::set(&mut merged, key, v.clone());
            }
        }
    }
    let has_back = merged.iter().any(|(k, v)| k == "back" && ingetrazo_model::pyval::truthy(v));
    if !has_back {
        ingetrazo_model::pyval::set(&mut merged, "back", Value::Bool(true));
    }
    merged
}

/// `ExplodeGroupCommand`: dissolve ONE level of a group. Its own geometry
/// merges back into the loose mesh and the groups nested in it come out
/// whole, as top-level objects in its place.
pub struct ExplodeGroup {
    index: usize,
    ctx: Option<Path>,
    snapshot: Option<State>,
    /// The group, as it was taken out (with its matrix as `do` left it).
    group: Option<Group>,
    lifted: usize,
}

impl ExplodeGroup {
    /// `index`: the group's place in the list it runs on.
    pub fn new(index: usize) -> Self {
        ExplodeGroup { index, ctx: None, snapshot: None, group: None, lifted: 0 }
    }
}

impl Command for ExplodeGroup {
    fn name(&self) -> &'static str {
        "explode_group"
    }

    fn apply(&mut self, scene: &mut Scene, cx: &mut Cx) -> Result<(), Fail> {
        let Scene { meshes, groups, .. } = scene;
        let active = cx.mesh.0 as usize;
        self.snapshot = Some(meshes[active].capture_state());
        let list = group_at_list_mut(groups, &self.ctx)?;
        if self.index >= list.len() {
            return Err(Fail::Failed("group is not in the scene".into()));
        }
        let mut g = list.remove(self.index);
        let mut p = g.xform;
        let paint = g.material.clone();
        let w = |p: &Option<M4>, v: V3| match p {
            Some(m) => m.map(v),
            None => v,
        };
        // Read the source geometry first: the active mesh may be written to.
        let src = &meshes[g.mesh.0 as usize];
        let faces: Vec<(Vec<V3>, Vec<Vec<V3>>, Vec<(String, Value)>)> = src
            .faces
            .iter()
            .map(|&f| {
                let face = src.face(f);
                (
                    face.outer.iter().map(|&v| src.position(v)).collect(),
                    face.holes.iter().map(|hl| hl.iter().map(|&v| src.position(v)).collect()).collect(),
                    face.attrs.clone(),
                )
            })
            .collect();
        let edges: Vec<(V3, V3, EdgeFlags)> = src
            .edges
            .iter()
            .map(|&e| {
                let ed = src.edge(e);
                (src.position(ed.v0), src.position(ed.v1), flags_of(src, e))
            })
            .collect();
        let m = &mut meshes[active];
        for (outer, holes, attrs) in &faces {
            let o: Vec<V3> = outer.iter().map(|&v| w(&p, v)).collect();
            let hs: Vec<Vec<V3>> = holes.iter().map(|hl| hl.iter().map(|&v| w(&p, v)).collect()).collect();
            let nf = m.add_face(&o, &hs);
            // What the face showed inside is what it keeps outside.
            for (k, v) in drawn_attrs(attrs, paint.as_ref()) {
                let fa = &mut m.face_mut(nf).attrs;
                ingetrazo_model::pyval::set(fa, &k, v);
            }
        }
        for (a, b, _) in &edges {
            let (wa, wb) = (w(&p, *a), w(&p, *b));
            let have = match (m.vertex_at(wa), m.vertex_at(wb)) {
                (Some(v0), Some(v1)) => m.find_edge(v0, v1).is_some(),
                _ => false,
            };
            if !have {
                let _ = m.add_edge(wa, wb);
            }
        }
        // Edge flags travel back out of the group.
        for (a, b, fl) in &edges {
            if edge_is_plain(fl) {
                continue;
            }
            let (wa, wb) = (w(&p, *a), w(&p, *b));
            let k = match (m.vertex_at(wa), m.vertex_at(wb)) {
                (Some(v0), Some(v1)) => m.find_edge(v0, v1),
                _ => None,
            };
            if let Some(k) = k {
                stamp(m, k, fl);
            }
        }
        m.resplit_curves(&mut cx.counters.curve);
        // The nested objects come out one level up, whole.
        let mut kids: Vec<Group> = g.children.clone();
        for c in &mut kids {
            c.explode_offset = None;
            if let Some(pm) = p.as_mut() {
                match c.xform {
                    Some(x) => c.xform = Some(pm.mul(&x)),
                    None => {
                        let moved = transformed_mesh(&meshes[c.mesh.0 as usize], pm, &mut cx.counters.curve);
                        c.mesh = push_mesh(meshes, moved);
                        carry_axes(c, pm);
                    }
                }
            }
            if c.material.is_none() {
                if let Some(pt) = &paint {
                    if ingetrazo_model::pyval::truthy(pt) {
                        c.material = Some(pt.clone());
                    }
                }
            }
        }
        g.xform = p; // the matrix the reads above left flagged
        self.lifted = kids.len();
        let list = group_at_list_mut(groups, &self.ctx)?;
        let tail = list.split_off(self.index);
        list.extend(kids);
        list.extend(tail);
        self.group = Some(g);
        Ok(())
    }

    fn undo(&mut self, scene: &mut Scene, cx: &mut Cx) {
        let Scene { meshes, groups, .. } = scene;
        if let Ok(list) = group_at_list_mut(groups, &self.ctx) {
            let end = (self.index + self.lifted).min(list.len());
            list.drain(self.index.min(end)..end);
            if let Some(g) = self.group.take() {
                let at = self.index.min(list.len());
                list.insert(at, g);
            }
        }
        if let Some(s) = &self.snapshot {
            meshes[cx.mesh.0 as usize].restore_state(s);
        }
    }
}

impl InContext for ExplodeGroup {
    fn set_context(&mut self, ctx: Option<Path>) {
        self.ctx = ctx;
    }
}

// ---- SplitIntoPiecesCommand / InsertGroupCommand -----------------------------------------------------------------

/// `SplitIntoPiecesCommand`: the group's contents replaced by `pieces`, one
/// child group per physical piece.
pub struct SplitIntoPieces {
    path: Path,
    pieces: Vec<Group>,
    before: Option<(MeshId, Vec<Group>, Option<M4>, Option<Value>)>,
}

impl SplitIntoPieces {
    pub fn new(path: &[usize], pieces: Vec<Group>) -> Self {
        SplitIntoPieces { path: path.to_vec(), pieces, before: None }
    }
}

impl Command for SplitIntoPieces {
    fn name(&self) -> &'static str {
        "split_into_pieces"
    }
    fn apply(&mut self, scene: &mut Scene, _cx: &mut Cx) -> Result<(), Fail> {
        let Scene { meshes, groups, .. } = scene;
        let h = meshes[0].hypot;
        let id = empty_mesh(meshes, h);
        let g = group_at_mut(groups, &self.path).ok_or_else(gone)?;
        self.before = Some((g.mesh, g.children.clone(), g.xform, g.exploded.clone()));
        g.mesh = id;
        g.adopt(self.pieces.clone());
        // The pieces are cut from the geometry as it stands: assembled.
        g.exploded = None;
        Ok(())
    }
    fn undo(&mut self, scene: &mut Scene, _cx: &mut Cx) {
        let (Some(g), Some((mesh, children, xform, exploded))) = (group_at_mut(&mut scene.groups, &self.path), self.before.clone()) else { return };
        g.mesh = mesh;
        g.children = children;
        g.xform = xform;
        g.exploded = exploded;
    }
}

/// `InsertGroupCommand`: a ready-made group into the model, or into the
/// open container (`context`), expressed in the container's own space.
pub struct InsertGroup {
    group: Group,
    context: Option<Path>,
}

impl InsertGroup {
    pub fn new(group: Group, context: Option<Path>) -> Self {
        InsertGroup { group, context }
    }
}

impl Command for InsertGroup {
    fn name(&self) -> &'static str {
        "insert_group"
    }
    fn apply(&mut self, scene: &mut Scene, cx: &mut Cx) -> Result<(), Fail> {
        let Scene { meshes, groups, .. } = scene;
        if let Some(ctx) = &self.context {
            let cx_xform = group_at(groups, ctx).ok_or_else(gone)?.xform;
            if let Some(x) = cx_xform {
                let (mut inv, ok) = x.inverted();
                if ok {
                    match self.group.xform {
                        Some(gx) => self.group.xform = Some(inv.mul(&gx)),
                        None => {
                            let moved = transformed_mesh(&meshes[self.group.mesh.0 as usize], &mut inv, &mut cx.counters.curve);
                            self.group.mesh = push_mesh(meshes, moved);
                            carry_axes(&mut self.group, &inv);
                        }
                    }
                }
            }
        }
        group_at_list_mut(groups, &self.context)?.push(self.group.clone());
        Ok(())
    }
    fn undo(&mut self, scene: &mut Scene, _cx: &mut Cx) {
        if let Ok(list) = group_at_list_mut(&mut scene.groups, &self.context) {
            if let Some(g) = list.pop() {
                self.group = g;
            }
        }
    }
}

// ---- ReplaceComponentCommand ----------------------------------------------------------------------------------------

struct Shared {
    mesh: MeshId,
    children: Vec<Group>,
    name: String,
    axes: Option<M4>,
    component: bool,
    text3d: Option<Value>,
    ifc: Option<Value>,
    ext: Option<Value>,
    billboard: Option<Value>,
    exploded: Option<Value>,
}

/// `ReplaceComponentCommand`: each target takes `source`'s definition and
/// keeps its own placement.
pub struct ReplaceComponent {
    source: Path,
    targets: Vec<Path>,
    before: Vec<Shared>,
}

impl ReplaceComponent {
    pub fn new(scene: &Scene, targets: &[Path], source: &[usize]) -> Result<Self, String> {
        let src = group_at(&scene.groups, source).ok_or("source is not in the scene")?;
        let mut kept = Vec::new();
        for t in targets {
            let g = group_at(&scene.groups, t).ok_or("target is not in the scene")?;
            if t.as_slice() != source && g.is_component() && g.mesh != src.mesh {
                kept.push(t.clone());
            }
        }
        Ok(ReplaceComponent { source: source.to_vec(), targets: kept, before: Vec::new() })
    }
}

impl Command for ReplaceComponent {
    fn name(&self) -> &'static str {
        "replace_component"
    }

    fn apply(&mut self, scene: &mut Scene, cx: &mut Cx) -> Result<(), Fail> {
        let Scene { meshes, groups, .. } = scene;
        let src = group_at(groups, &self.source).ok_or_else(gone)?.clone();
        // `copy_group` shares the source's subgroups, touching them in the
        // SOURCE as Python does; the touched children go back to it below.
        let mut src_children = src.children.clone();
        self.before.clear();
        for t in &self.targets {
            let g = group_at_mut(groups, t).ok_or_else(gone)?;
            self.before.push(Shared {
                mesh: g.mesh,
                children: g.children.clone(),
                name: g.name.clone(),
                axes: g.axes,
                component: g.component,
                text3d: g.text3d.clone(),
                ifc: g.ifc.clone(),
                ext: g.ext.clone(),
                billboard: g.billboard.clone(),
                exploded: g.exploded.clone(),
            });
            g.mesh = src.mesh;
            g.children = src_children.iter_mut().map(|c| copy_group(meshes, c, None, true, &mut cx.counters)).collect();
            g.name = src.name.clone();
            g.axes = src.axes;
            g.component = true;
            g.text3d = truthy_dict(&src.text3d);
            g.ifc = truthy_dict(&src.ifc);
            g.ext = truthy_dict(&src.ext);
            g.billboard = src.billboard.clone();
            g.exploded = truthy_dict(&src.exploded);
        }
        group_at_mut(groups, &self.source).ok_or_else(gone)?.children = src_children;
        Ok(())
    }

    fn undo(&mut self, scene: &mut Scene, _cx: &mut Cx) {
        for (t, b) in self.targets.iter().zip(self.before.drain(..)) {
            let Some(g) = group_at_mut(&mut scene.groups, t) else { continue };
            g.mesh = b.mesh;
            g.children = b.children;
            g.name = b.name;
            g.axes = b.axes;
            g.component = b.component;
            g.text3d = b.text3d;
            g.ifc = b.ifc;
            g.ext = b.ext;
            g.billboard = b.billboard;
            g.exploded = b.exploded;
        }
    }
}

// ---- kept components ----------------------------------------------------------------------------------------------------

/// `kept_template`: the index of the template kept for the definition whose
/// prototype is `mesh`.
pub fn kept_template(scene: &Scene, mesh: MeshId) -> Option<usize> {
    scene.kept_components.iter().position(|t| t.mesh == mesh)
}

/// `make_template`: a placement-free copy of `instance`'s definition.
pub fn make_template(meshes: &mut Vec<Mesh>, instance: &mut Group, counters: &mut Counters) -> Group {
    let mut t = copy_group(meshes, instance, None, false, counters);
    t.xform = Some(M4::identity());
    t.layer = None;
    t.material = None;
    t.hidden = false;
    t.explode_offset = None;
    t.component = true;
    t
}

/// A group named by where it is: in the model, or a kept template.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GroupRef {
    Scene(Path),
    Kept(usize),
}

pub fn group_ref<'a>(scene: &'a Scene, r: &GroupRef) -> Option<&'a Group> {
    match r {
        GroupRef::Scene(p) => group_at(&scene.groups, p),
        GroupRef::Kept(i) => scene.kept_components.get(*i),
    }
}

fn group_ref_mut<'a>(scene: &'a mut Scene, r: &GroupRef) -> Option<&'a mut Group> {
    match r {
        GroupRef::Scene(p) => group_at_mut(&mut scene.groups, p),
        GroupRef::Kept(i) => scene.kept_components.get_mut(*i),
    }
}

/// `KeepComponentCommand`: Keep in File, on or off, for the definition
/// `instance` uses.
pub struct KeepComponent {
    instance: GroupRef,
    keep: bool,
    template: Option<Group>,
    index: Option<usize>,
    /// Whether `do` found something to change (Python: `_template` set).
    acted: bool,
}

impl KeepComponent {
    pub fn new(instance: GroupRef, keep: bool) -> Self {
        KeepComponent { instance, keep, template: None, index: None, acted: false }
    }
}

impl Command for KeepComponent {
    fn name(&self) -> &'static str {
        "keep_component"
    }
    fn apply(&mut self, scene: &mut Scene, cx: &mut Cx) -> Result<(), Fail> {
        let mesh = group_ref(scene, &self.instance).ok_or_else(gone)?.mesh;
        let current = kept_template(scene, mesh);
        if self.keep && current.is_none() {
            if self.template.is_none() {
                let mut inst = group_ref(scene, &self.instance).ok_or_else(gone)?.clone();
                let t = make_template(&mut scene.meshes, &mut inst, &mut cx.counters);
                // The copy touched the instance's subgroups (Python: in place).
                if let Some(g) = group_ref_mut(scene, &self.instance) {
                    g.children = inst.children;
                }
                self.template = Some(t);
            }
            scene.kept_components.push(self.template.clone().ok_or_else(gone)?);
            self.index = None;
            self.acted = true;
        } else if let (false, Some(i)) = (self.keep, current) {
            self.template = Some(scene.kept_components.remove(i));
            self.index = Some(i);
            self.acted = true;
        } else {
            self.acted = false; // already as asked: a no-op
        }
        Ok(())
    }
    fn undo(&mut self, scene: &mut Scene, _cx: &mut Cx) {
        if !self.acted {
            return;
        }
        if self.keep {
            if let Some(t) = scene.kept_components.pop() {
                self.template = Some(t);
            }
        } else if let (Some(t), Some(i)) = (self.template.clone(), self.index) {
            let at = i.min(scene.kept_components.len());
            scene.kept_components.insert(at, t);
        }
    }
}

/// `AddKeptComponentsCommand`: ready-made definitions into the document as
/// kept components (no copy placed).
pub struct AddKeptComponents {
    templates: Vec<Group>,
}

impl AddKeptComponents {
    pub fn new(templates: Vec<Group>) -> Self {
        AddKeptComponents { templates }
    }
}

impl Command for AddKeptComponents {
    fn name(&self) -> &'static str {
        "add_kept_components"
    }
    fn apply(&mut self, scene: &mut Scene, _cx: &mut Cx) -> Result<(), Fail> {
        scene.kept_components.extend(self.templates.iter().cloned());
        Ok(())
    }
    fn undo(&mut self, scene: &mut Scene, _cx: &mut Cx) {
        let n = self.templates.len().min(scene.kept_components.len());
        let at = scene.kept_components.len() - n;
        scene.kept_components.truncate(at);
    }
}

// ---- ReshareInstanceCommand -----------------------------------------------------------------------------------------------------

/// `ReshareInstanceCommand`: leaving a component instance edited in place,
/// the session's edits go into the shared definition. `edited` is the
/// instance's world-space copy (the group's mesh while it was edited).
///
/// Not bit-exact in one place: Python writes the edit into the prototype's
/// arrays with `restore_state(local.capture_state())`; here the prototype's
/// arena slot is replaced by the local copy and the old one kept for undo.
pub struct ReshareInstance {
    path: Path,
    proto: MeshId,
    xform: M4,
    edited: MeshId,
    before: Option<Mesh>,
}

impl ReshareInstance {
    pub fn new(path: &[usize], proto: MeshId, xform: M4, edited: MeshId) -> Self {
        ReshareInstance { path: path.to_vec(), proto, xform, edited, before: None }
    }
}

impl Command for ReshareInstance {
    fn name(&self) -> &'static str {
        "reshare_instance"
    }
    fn apply(&mut self, scene: &mut Scene, cx: &mut Cx) -> Result<(), Fail> {
        let Scene { meshes, groups, .. } = scene;
        if self.before.is_none() {
            self.before = Some(meshes[self.proto.0 as usize].clone());
        }
        let (inverse, ok) = self.xform.inverted();
        let mut m = if ok { inverse } else { self.xform };
        let local = transformed_mesh(&meshes[self.edited.0 as usize], &mut m, &mut cx.counters.curve);
        meshes[self.proto.0 as usize] = local;
        let g = group_at_mut(groups, &self.path).ok_or_else(gone)?;
        g.mesh = self.proto;
        g.xform = Some(self.xform);
        Ok(())
    }
    fn undo(&mut self, scene: &mut Scene, _cx: &mut Cx) {
        if let Some(b) = &self.before {
            scene.meshes[self.proto.0 as usize] = b.clone();
        }
        if let Some(g) = group_at_mut(&mut scene.groups, &self.path) {
            g.mesh = self.proto;
            g.xform = Some(self.xform);
        }
    }
}

// ---- exploded view (core/explode.py, core/parts.py) ---------------------------------------------------------------------------------

/// `parts.part_points`: the part's vertex positions in its container's
/// coordinates (nested placements included). Reading `m(r, c)` flags the
/// matrix General — the part's own `xform` when it is the one read.
pub fn part_points(meshes: &[Mesh], part: &mut Group) -> Vec<[f64; 3]> {
    fn walk(meshes: &[Mesh], g: &mut Group, base: &mut Option<M4>, out: &mut Vec<[f64; 3]>) {
        let mesh_id = g.mesh;
        let (xform, children) = (&mut g.xform, &mut g.children);
        let mut product: Option<M4>;
        let m: &mut Option<M4> = match (base.is_some(), xform.is_some()) {
            (false, _) => xform,
            (true, false) => base,
            (true, true) => {
                product = Some(base.unwrap_or_default().mul(&xform.unwrap_or_default()));
                &mut product
            }
        };
        let src = &meshes[mesh_id.0 as usize];
        if !src.vertices.is_empty() {
            let mut pos: Vec<[f64; 3]> = src
                .vertices
                .iter()
                .map(|&v| {
                    let p = src.position(v);
                    [p.0 as f64, p.1 as f64, p.2 as f64]
                })
                .collect();
            if let Some(mm) = m.as_mut() {
                // `m(r, c)` reads: the non-const operator() flags General.
                mm.flags = ingetrazo_math::matrix::GENERAL;
                let (rot, trans) = np_affine(mm);
                pos = affine_rows(&pos, &rot, &trans);
            }
            out.extend(pos);
        }
        for c in children.iter_mut() {
            walk(meshes, c, m, out);
        }
    }
    let mut out = Vec::new();
    walk(meshes, part, &mut None, &mut out);
    out
}

/// `explode._offset_of`.
fn offset_of(part: &Group) -> V3 {
    match part.explode_offset {
        Some(o) => V3(o[0] as f32, o[1] as f32, o[2] as f32),
        None => V3(0.0, 0.0, 0.0),
    }
}

/// `explode.assembled_centres`.
pub fn assembled_centres(meshes: &[Mesh], container: &mut Group) -> Vec<V3> {
    let mut out = Vec::new();
    for kid in &mut container.children {
        let pts = part_points(meshes, kid);
        let c = if pts.is_empty() {
            V3(0.0, 0.0, 0.0)
        } else {
            let (mut lo, mut hi) = (pts[0], pts[0]);
            for p in &pts[1..] {
                for i in 0..3 {
                    lo[i] = lo[i].min(p[i]);
                    hi[i] = hi[i].max(p[i]);
                }
            }
            V3(((lo[0] + hi[0]) / 2.0) as f32, ((lo[1] + hi[1]) / 2.0) as f32, ((lo[2] + hi[2]) / 2.0) as f32)
        };
        out.push(c.sub(offset_of(kid)));
    }
    out
}

/// `explode.explode_offsets`.
pub fn explode_offsets(centres: &[V3], factor: f64, mode: &str) -> Vec<V3> {
    if centres.is_empty() {
        return Vec::new();
    }
    let (xs, ys, zs): (Vec<f64>, Vec<f64>, Vec<f64>) = (
        centres.iter().map(|c| c.0 as f64).collect(),
        centres.iter().map(|c| c.1 as f64).collect(),
        centres.iter().map(|c| c.2 as f64).collect(),
    );
    let fmin = |v: &[f64]| v.iter().copied().fold(f64::INFINITY, f64::min);
    let fmax = |v: &[f64]| v.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mid = V3(((fmin(&xs) + fmax(&xs)) / 2.0) as f32, ((fmin(&ys) + fmax(&ys)) / 2.0) as f32, ((fmin(&zs) + fmax(&zs)) / 2.0) as f32);
    let keep: [f64; 3] = match mode {
        "x" => [1.0, 0.0, 0.0],
        "y" => [0.0, 1.0, 0.0],
        "z" => [0.0, 0.0, 1.0],
        _ => [1.0, 1.0, 1.0],
    };
    centres
        .iter()
        .map(|c| {
            let d = c.sub(mid).mul(factor as f32);
            V3((d.0 as f64 * keep[0]) as f32, (d.1 as f64 * keep[1]) as f32, (d.2 as f64 * keep[2]) as f32)
        })
        .collect()
}

/// `explode.apply_explode`.
pub fn apply_explode(meshes: &[Mesh], container: &mut Group, factor: f64, mode: &str) {
    if container.children.is_empty() {
        return;
    }
    let centres = assembled_centres(meshes, container);
    let factor = factor.max(0.0);
    let new: Vec<Option<V3>> = if factor > 0.0 { explode_offsets(&centres, factor, mode).into_iter().map(Some).collect() } else { vec![None; container.children.len()] };
    for (kid, target) in container.children.iter_mut().zip(new) {
        let old = offset_of(kid);
        let mut step = M4::identity();
        let t = match target {
            Some(t) => t.sub(old),
            None => old.mul(-1.0),
        };
        step.translate(t.0, t.1, t.2);
        kid.xform = Some(step.mul(&kid.xform.unwrap_or_default()));
        kid.explode_offset = match target {
            Some(t) if !t.is_null() => Some([t.0 as f64, t.1 as f64, t.2 as f64]),
            _ => None,
        };
    }
    container.exploded = if factor > 0.0 {
        Some(Value::Object(vec![("factor".into(), Value::Float(factor)), ("mode".into(), Value::Str(mode.to_string()))]))
    } else {
        None
    };
}

type Snap = (Option<Value>, Vec<(Option<M4>, Option<[f64; 3]>)>);

/// `explode.snapshot`.
fn explode_snapshot(container: &Group) -> Snap {
    (truthy_dict(&container.exploded), container.children.iter().map(|k| (k.xform, k.explode_offset)).collect())
}

/// `explode.restore`.
fn explode_restore(container: &mut Group, snap: &Snap) {
    container.exploded = snap.0.clone();
    for (kid, (x, o)) in container.children.iter_mut().zip(&snap.1) {
        kid.xform = *x;
        kid.explode_offset = *o;
    }
}

/// `ExplodeViewCommand`: pull a component's parts apart to `factor` along
/// `mode` — or put them back with `factor = 0`.
pub struct ExplodeView {
    path: Path,
    factor: f64,
    mode: String,
    before: Option<Snap>,
}

impl ExplodeView {
    pub fn new(path: &[usize], factor: f64, mode: &str) -> Self {
        ExplodeView { path: path.to_vec(), factor, mode: mode.to_string(), before: None }
    }
}

impl Command for ExplodeView {
    fn name(&self) -> &'static str {
        "explode_view"
    }
    fn apply(&mut self, scene: &mut Scene, _cx: &mut Cx) -> Result<(), Fail> {
        let Scene { meshes, groups, .. } = scene;
        let g = group_at_mut(groups, &self.path).ok_or_else(gone)?;
        self.before = Some(explode_snapshot(g));
        apply_explode(meshes, g, self.factor, &self.mode);
        Ok(())
    }
    fn undo(&mut self, scene: &mut Scene, _cx: &mut Cx) {
        if let (Some(g), Some(b)) = (group_at_mut(&mut scene.groups, &self.path), &self.before) {
            explode_restore(g, b);
        }
    }
}
