// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! The drawing commands of `core/history.py`: `AddEdgeCommand`,
//! `DeleteEdgesCommand`, `EraseSelectionCommand`, `TagCurveCommand`,
//! `AddFaceCommand`, `DeleteFaceCommand`, `FlipFacesCommand`,
//! `SnapshotCompound`, `CompoundCommand`, `MeshSnapshotCommand`,
//! `AutoTagDrawnFacesCommand`, the small face-paint and edge-flag commands
//! and the three stitch commands as `Command`s.
//!
//! Each command resolves what it acts on by POSITION at `do` time and
//! reverses itself the way Python's does (see `command.rs`). Where Python
//! holds a live object between `do` and `undo` this holds its arena id.
//!
//! Order: nothing here iterates a Python set whose order shows, except
//! where noted.

use std::cell::Cell;
use std::collections::{HashMap, HashSet};
use std::rc::Rc;

use ingetrazo_formats::pyjson::Value;
use ingetrazo_math::{Hypot, V3};
use ingetrazo_model::mesh::{key as mesh_key, EdgeId, FaceId, Mesh, State, VertexId, PAINT_KEYS};
use ingetrazo_model::pyval;
use ingetrazo_model::scene::{MeshId, Scene};
use ingetrazo_model::stitch::{self, CoplanarMergeCommand, PruneOrphanEdgesCommand, StitchSolidCommand};
use ingetrazo_model::topology::{self, key, PFace};

use crate::command::{CmdResult, Command, CommandError, Env};
use crate::orient;

type Key = topology::Key;

fn unordered(a: Key, b: Key) -> (Key, Key) {
    if a <= b {
        (a, b)
    } else {
        (b, a)
    }
}

/// `core.topology._loop_edges` of a position loop.
fn loop_edges(l: &[V3]) -> Vec<(Key, Key)> {
    let n = l.len();
    (0..n).map(|i| unordered(key(l[i]), key(l[(i + 1) % n]))).collect()
}

fn loop_keyset(l: &[V3]) -> Vec<Key> {
    let mut v: Vec<Key> = l.iter().map(|&p| key(p)).collect();
    v.sort();
    v.dedup();
    v
}

fn positions(m: &Mesh, l: &[VertexId]) -> Vec<V3> {
    l.iter().map(|&v| m.position(v)).collect()
}

/// `_find_face_by_loop`: the first mesh face whose outer loop has these
/// corners (as a set, by weld key).
pub fn find_face_by_loop(m: &Mesh, loop_positions: &[V3]) -> Option<FaceId> {
    let target = loop_keyset(loop_positions);
    m.faces.iter().copied().find(|&f| loop_keyset(&positions(m, &m.face(f).outer)) == target)
}

/// `_inherit_paint`: give `face` the paint of `mother` for the keys it does
/// not carry itself.
fn inherit_paint(m: &mut Mesh, face: FaceId, mother: FaceId) {
    let src = m.face(mother).attrs.clone();
    if src.is_empty() {
        return;
    }
    for k in PAINT_KEYS {
        if let Some((_, v)) = src.iter().find(|(n, _)| n == k) {
            if !m.face(face).attrs.iter().any(|(n, _)| n == k) {
                m.face_mut(face).attrs.push((k.to_string(), v.clone()));
            }
        }
    }
}

// ---- AddEdge --------------------------------------------------------------------

/// `AddEdgeCommand`: add a single edge (welded and deduplicated); undo only
/// removes an edge this command created.
pub struct AddEdgeCommand {
    pub a: V3,
    pub b: V3,
    pub soft: Option<bool>,
    pub curve: Option<i64>,
    pub edge: Option<EdgeId>,
    owned: bool,
}

impl AddEdgeCommand {
    pub fn new(a: V3, b: V3, soft: Option<bool>, curve: Option<i64>) -> Self {
        AddEdgeCommand { a, b, soft, curve, edge: None, owned: false }
    }

    fn stamp(&self, m: &mut Mesh, e: EdgeId) {
        if let Some(s) = self.soft {
            m.edge_mut(e).soft = s;
        }
        if let Some(c) = self.curve {
            m.edge_mut(e).curve = Some(c);
        }
    }
}

impl Command for AddEdgeCommand {
    fn apply(&mut self, scene: &mut Scene, env: &mut Env) -> CmdResult {
        let m = env.mesh_mut(scene);
        if self.owned {
            if let Some(e) = self.edge {
                m.relink_edge(e);
                self.stamp(m, e);
                return Ok(());
            }
        }
        let v0 = m.vertex_at(self.a);
        let v1 = m.vertex_at(self.b);
        let pre = match (v0, v1) {
            (Some(x), Some(y)) => m.find_edge(x, y),
            _ => None,
        };
        if let Some(pre) = pre {
            self.stamp(m, pre);
            return Ok(());
        }
        let e = m.add_edge(self.a, self.b).map_err(|_| CommandError::Failed("degenerate edge: endpoints weld to one vertex".into()))?;
        self.edge = Some(e);
        self.owned = true;
        self.stamp(m, e);
        Ok(())
    }

    fn undo(&mut self, scene: &mut Scene, env: &mut Env) {
        if !self.owned {
            return;
        }
        if let Some(e) = self.edge {
            env.mesh_mut(scene).remove_edge(e);
        }
    }

    fn name(&self) -> &'static str {
        "AddEdgeCommand"
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

// ---- DeleteEdges ----------------------------------------------------------------

/// `DeleteEdgesCommand`: erase edges (found by endpoint position); with
/// `cascade_faces` also the faces that used one on their outer boundary.
pub struct DeleteEdgesCommand {
    pub(crate) endpoints: Vec<(V3, V3)>,
    pub cascade_faces: bool,
    removed_edges: Vec<EdgeId>,
    removed_faces: Vec<FaceId>,
}

impl DeleteEdgesCommand {
    pub fn new(endpoints: Vec<(V3, V3)>, cascade_faces: bool) -> Self {
        DeleteEdgesCommand { endpoints, cascade_faces, removed_edges: Vec::new(), removed_faces: Vec::new() }
    }
}

/// The edge of the mesh whose ends weld to `a` and `b`.
pub fn edge_at(m: &Mesh, a: V3, b: V3) -> Option<EdgeId> {
    let v0 = m.vertex_at(a)?;
    let v1 = m.vertex_at(b)?;
    m.find_edge(v0, v1)
}

impl Command for DeleteEdgesCommand {
    fn apply(&mut self, scene: &mut Scene, env: &mut Env) -> CmdResult {
        let m = env.mesh_mut(scene);
        self.removed_edges.clear();
        for &(a, b) in &self.endpoints {
            if let Some(e) = edge_at(m, a, b) {
                self.removed_edges.push(e);
            }
        }
        if self.cascade_faces {
            let gone: HashSet<(Key, Key)> = self.endpoints.iter().map(|&(a, b)| unordered(key(a), key(b))).collect();
            self.removed_faces = m
                .faces
                .iter()
                .copied()
                .filter(|&f| loop_edges(&positions(m, &m.face(f).outer)).iter().any(|k| gone.contains(k)))
                .collect();
            for &f in &self.removed_faces {
                m.remove_face(f);
            }
        }
        for &e in &self.removed_edges {
            m.remove_edge(e);
        }
        Ok(())
    }

    fn undo(&mut self, scene: &mut Scene, env: &mut Env) {
        let m = env.mesh_mut(scene);
        for &e in &self.removed_edges {
            m.relink_edge(e);
        }
        for &f in &self.removed_faces {
            m.relink_face(f);
        }
        self.removed_faces.clear();
    }

    fn name(&self) -> &'static str {
        "DeleteEdgesCommand"
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

// ---- EraseSelection ---------------------------------------------------------------

/// `EraseSelectionCommand`: erase edges and faces; an edge between two
/// coplanar faces dissolves them into one, any other takes its faces with
/// it. Undo restores a snapshot.
pub struct EraseSelectionCommand {
    pub(crate) edge_endpoints: Vec<(V3, V3)>,
    pub(crate) face_loops: Vec<Vec<V3>>,
    snapshot: Option<State>,
}

impl EraseSelectionCommand {
    pub fn new(edge_endpoints: Vec<(V3, V3)>, face_loops: Vec<Vec<V3>>) -> Self {
        EraseSelectionCommand { edge_endpoints, face_loops, snapshot: None }
    }
}

impl Command for EraseSelectionCommand {
    fn apply(&mut self, scene: &mut Scene, env: &mut Env) -> CmdResult {
        let h = env.hypot;
        self.snapshot = Some(env.mesh(scene).capture_state());
        let footprint = topology::winding_footprint(env.mesh(scene), h);
        // Loop -> faces through one index (list order; `pop(0)` takes the first).
        let mut by_loop: HashMap<Vec<Key>, Vec<FaceId>> = HashMap::new();
        {
            let m = env.mesh(scene);
            for &f in &m.faces {
                by_loop.entry(loop_keyset(&positions(m, &m.face(f).outer))).or_default().push(f);
            }
        }
        for lp in &self.face_loops {
            let bucket = by_loop.get_mut(&loop_keyset(lp));
            let f = match bucket {
                Some(b) if !b.is_empty() => Some(b.remove(0)),
                _ => None,
            };
            if let Some(f) = f {
                env.mesh_mut(scene).remove_face(f);
            }
        }
        // One segment -> faces index for the cascade, kept live as faces merge.
        let mut seg_faces: HashMap<(Key, Key), Vec<FaceId>> = HashMap::new();
        fn index_face(m: &Mesh, seg_faces: &mut HashMap<(Key, Key), Vec<FaceId>>, f: FaceId) {
            for ek in loop_edges(&positions(m, &m.face(f).outer)) {
                seg_faces.entry(ek).or_default().push(f);
            }
        }
        {
            let m = env.mesh(scene);
            for &f in &m.faces {
                index_face(m, &mut seg_faces, f);
            }
        }
        let mut dead_faces: HashSet<FaceId> = HashSet::new();
        let mut slit_planes: Vec<(V3, V3)> = Vec::new();
        for &(a, b) in &self.edge_endpoints {
            let m = env.mesh_mut(scene);
            let Some(edge) = edge_at(m, a, b) else { continue };
            let faces = m.edge(edge).faces.clone();
            let mut merged: Option<FaceId> = None;
            if faces.len() == 2 && faces[0] == faces[1] {
                // Slit edge: one face walks it twice. Drop only the edge and
                // re-derive the plane's regions below.
                let f = faces[0];
                slit_planes.push((m.position(m.face(f).outer[0]), m.face_normal(f, h)));
                m.remove_edge(edge);
                continue;
            }
            if faces.len() == 2 && (m.face_normal(faces[0], h).dot(m.face_normal(faces[1], h)) as f64) > 0.999 {
                merged = m.dissolve_coplanar_region(&faces, h);
                if let Some(mf) = merged {
                    dead_faces.extend(faces.iter().copied());
                    index_face(m, &mut seg_faces, mf);
                }
            }
            if merged.is_none() {
                // Cascade: the edge and every face bounding it go away.
                let ed = m.edge(edge);
                let ekey = unordered(key(m.position(ed.v0)), key(m.position(ed.v1)));
                let hit: Vec<FaceId> = seg_faces.get(&ekey).cloned().unwrap_or_default();
                for f in hit {
                    if dead_faces.contains(&f) {
                        continue;
                    }
                    dead_faces.insert(f);
                    m.remove_face(f);
                }
                m.remove_edge(edge);
            }
        }
        // Slit removals re-derive their plane from the arrangement while the
        // cut-ring face is still present to provide coverage.
        let mut done_planes: Vec<(V3, V3)> = Vec::new();
        for (origin, normal) in slit_planes {
            let nn = normal.normalized(h);
            if done_planes
                .iter()
                .any(|&(o, n)| (nn.dot(n) as f64).abs() > 0.999 && (origin.sub(o).dot(n) as f64).abs() < 1e-4)
            {
                continue;
            }
            done_planes.push((origin, nn));
            crate::rebuild::RebuildPlaneFacesCommand::new(origin, nn, h).apply(scene, env)?;
        }
        let m = env.mesh_mut(scene);
        // Deleting a curved surface takes its hidden seams with it.
        for e in m.edges.clone() {
            if m.edge(e).soft && m.edge(e).faces.is_empty() {
                m.remove_edge(e);
            }
        }
        // A merge can leave the big enclosing face overlapping its subdivisions.
        topology::heal_overlapping_faces(m, 0.5, None, footprint.as_deref(), h, &mut |mm: &mut Mesh| {
            orient::orient_outward(mm, h);
        });
        m.prune_orphan_vertices();
        Ok(())
    }

    fn undo(&mut self, scene: &mut Scene, env: &mut Env) {
        if let Some(s) = &self.snapshot {
            env.mesh_mut(scene).restore_state(s);
        }
    }

    fn name(&self) -> &'static str {
        "EraseSelectionCommand"
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

// ---- TagCurve ----------------------------------------------------------------------

/// `TagCurveCommand`: mark the edges along a drawn loop as one curve. Its
/// undo is a no-op: the enclosing snapshot restores the pre-draw state.
pub struct TagCurveCommand {
    pub pts: Vec<V3>,
    pub closed: bool,
}

impl Command for TagCurveCommand {
    fn apply(&mut self, scene: &mut Scene, env: &mut Env) -> CmdResult {
        let h = env.hypot;
        let mut counter = env.curve;
        env.mesh_mut(scene).tag_curve(&mut counter, &self.pts, self.closed, h);
        env.curve = counter;
        Ok(())
    }
    fn undo(&mut self, _scene: &mut Scene, _env: &mut Env) {}
    fn name(&self) -> &'static str {
        "TagCurveCommand"
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

// ---- AddFace -----------------------------------------------------------------------

/// `AddFaceCommand`: add a face, dividing any coplanar face it lands
/// strictly inside (or is drawn against).
pub struct AddFaceCommand {
    pub vertices: Vec<V3>,
    pub preset_holes: Option<Vec<Vec<V3>>>,
    pub auto: bool,
    pub attrs: Option<Vec<(String, Value)>>,
    /// The face once made. Shared so `AutoTagDrawnFacesCommand` can read the
    /// faces its sibling commands made (`face_commands`).
    pub face: Rc<Cell<Option<FaceId>>>,
    punches: Vec<(FaceId, Vec<VertexId>)>,
    subdiv_mother: Option<FaceId>,
    subdiv_remainder: Option<Vec<FaceId>>,
    made_pairs: HashSet<(VertexId, VertexId)>,
    made_vertices: Vec<VertexId>,
}

fn vpair(a: VertexId, b: VertexId) -> (VertexId, VertexId) {
    if a <= b {
        (a, b)
    } else {
        (b, a)
    }
}

impl AddFaceCommand {
    pub fn new(vertices: Vec<V3>, auto: bool, holes: Option<Vec<Vec<V3>>>, attrs: Option<Vec<(String, Value)>>) -> Self {
        let preset_holes = holes.filter(|h| !h.is_empty());
        let attrs = attrs.filter(|a| !a.is_empty());
        AddFaceCommand {
            vertices,
            preset_holes,
            auto,
            attrs,
            face: Rc::new(Cell::new(None)),
            punches: Vec::new(),
            subdiv_mother: None,
            subdiv_remainder: None,
            made_pairs: HashSet::new(),
            made_vertices: Vec::new(),
        }
    }
}

impl Command for AddFaceCommand {
    fn apply(&mut self, scene: &mut Scene, env: &mut Env) -> CmdResult {
        let h = env.hypot;
        let m = env.mesh_mut(scene);
        let face = match self.face.get() {
            None => {
                let holes: Vec<Vec<V3>> = self.preset_holes.clone().unwrap_or_default();
                // What add_face is about to create besides the face: its
                // boundary edges and corners that were not there yet.
                let mut pairs: Vec<(V3, V3)> = Vec::new();
                for lp in std::iter::once(&self.vertices).chain(holes.iter()) {
                    for i in 0..lp.len() {
                        pairs.push((lp[i], lp[(i + 1) % lp.len()]));
                    }
                }
                let had_v: HashSet<VertexId> =
                    self.vertices.iter().chain(holes.iter().flatten()).filter_map(|&p| m.vertex_at(p)).collect();
                let mut had_e: HashSet<EdgeId> = HashSet::new();
                for &(a, b) in &pairs {
                    if let Some(e) = edge_at(m, a, b) {
                        had_e.insert(e);
                    }
                }
                let f = m.add_face(&self.vertices, &holes);
                self.face.set(Some(f));
                self.made_pairs.clear();
                self.made_vertices.clear();
                let loops: Vec<Vec<VertexId>> =
                    std::iter::once(m.face(f).outer.clone()).chain(m.face(f).holes.iter().cloned()).collect();
                for lp in &loops {
                    for i in 0..lp.len() {
                        let (a, b) = (lp[i], lp[(i + 1) % lp.len()]);
                        if let Some(e) = m.find_edge(a, b) {
                            if !had_e.contains(&e) {
                                self.made_pairs.insert(vpair(a, b));
                            }
                        }
                    }
                    for &v in lp {
                        if !had_v.contains(&v) {
                            self.made_vertices.push(v);
                        }
                    }
                }
                if let Some(attrs) = &self.attrs {
                    for (k, v) in attrs {
                        pyval::set(&mut m.face_mut(f).attrs, k, v.clone());
                    }
                }
                f
            }
            Some(f) => {
                m.relink_vertices(&self.made_vertices);
                m.relink_face(f);
                f
            }
        };
        if !self.auto {
            return Ok(());
        }
        let face_pos = positions(m, &m.face(face).outer);
        // Direction A: the new face falls inside an existing mother.
        let faces = m.faces.clone();
        if let Some(mother) = topology::find_containing_face(m, &faces, &face_pos, Some(face), h) {
            let lp = m.add_hole(mother, &face_pos);
            self.punches.push((mother, lp));
            inherit_paint(m, face, mother);
        }
        // Direction B: the new face encloses existing smaller faces.
        for other in m.faces.clone() {
            if other == face {
                continue;
            }
            let other_pos = positions(m, &m.face(other).outer);
            if topology::loop_inside_face(&PFace::of(m, face), &other_pos, false, h) {
                let lp = m.add_hole(face, &other_pos);
                self.punches.push((face, lp));
            }
        }
        // Direction C: drawn against an existing face's boundary.
        if self.punches.is_empty() {
            for other in m.faces.clone() {
                if other == face {
                    continue;
                }
                let pf = PFace::of(m, other);
                let remainder = topology::subtract_loop_from_face(&pf, &face_pos, h);
                let pieces: Vec<Vec<V3>> = match remainder {
                    Some(r) => vec![r],
                    None => topology::carve_loop_by_chords(&pf, &face_pos, h).unwrap_or_default(),
                };
                if pieces.is_empty() {
                    continue;
                }
                // Each hole of the mother goes to the piece that holds it; a
                // hole straddling a cut leaves the mother alone.
                let mut piece_holes: Vec<Vec<Vec<V3>>> = vec![Vec::new(); pieces.len()];
                let mut straddle = false;
                for hole in &pf.holes {
                    let mut placed = false;
                    for (k, piece) in pieces.iter().enumerate() {
                        let fake = PFace { vertices: piece.clone(), holes: Vec::new() };
                        if topology::loop_inside_face(&fake, hole, false, h) {
                            piece_holes[k].push(hole.clone());
                            placed = true;
                            break;
                        }
                    }
                    if !placed {
                        straddle = true;
                        break;
                    }
                }
                if straddle {
                    continue;
                }
                m.remove_face(other);
                let mut rem = Vec::new();
                let other_attrs = m.face(other).attrs.clone();
                for (piece, holes) in pieces.iter().zip(piece_holes.iter()) {
                    let rf = m.add_face(piece, holes);
                    m.face_mut(rf).attrs = other_attrs.clone();
                    rem.push(rf);
                }
                self.subdiv_remainder = Some(rem);
                inherit_paint(m, face, other);
                self.subdiv_mother = Some(other);
                break;
            }
        }
        Ok(())
    }

    fn undo(&mut self, scene: &mut Scene, env: &mut Env) {
        let m = env.mesh_mut(scene);
        if let Some(mother) = self.subdiv_mother.take() {
            for rem in self.subdiv_remainder.take().unwrap_or_default() {
                m.remove_face(rem);
            }
            m.relink_face(mother);
        }
        for (face, lp) in std::mem::take(&mut self.punches) {
            m.remove_hole(face, &lp);
        }
        if let Some(face) = self.face.get() {
            m.remove_face(face);
            // The edges and corners this face brought with it go too, when
            // nothing else uses them.
            let loops: Vec<Vec<VertexId>> =
                std::iter::once(m.face(face).outer.clone()).chain(m.face(face).holes.iter().cloned()).collect();
            for lp in &loops {
                for i in 0..lp.len() {
                    let (a, b) = (lp[i], lp[(i + 1) % lp.len()]);
                    if !self.made_pairs.contains(&vpair(a, b)) {
                        continue;
                    }
                    if let Some(e) = m.find_edge(a, b) {
                        if m.edge(e).faces.is_empty() {
                            m.remove_edge(e);
                        }
                    }
                }
            }
            m.drop_orphan_vertices(&self.made_vertices);
        }
    }

    fn name(&self) -> &'static str {
        "AddFaceCommand"
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

// ---- DeleteFace -----------------------------------------------------------------------

/// `DeleteFaceCommand`: remove a face, found by its outer loop.
pub struct DeleteFaceCommand {
    pub(crate) loop_: Vec<V3>,
    face: Option<FaceId>,
}

impl DeleteFaceCommand {
    pub fn new(loop_: Vec<V3>) -> Self {
        DeleteFaceCommand { loop_, face: None }
    }
}

impl Command for DeleteFaceCommand {
    fn apply(&mut self, scene: &mut Scene, env: &mut Env) -> CmdResult {
        let m = env.mesh_mut(scene);
        self.face = find_face_by_loop(m, &self.loop_);
        if let Some(f) = self.face {
            m.remove_face(f);
        }
        Ok(())
    }
    fn undo(&mut self, scene: &mut Scene, env: &mut Env) {
        if let Some(f) = self.face {
            env.mesh_mut(scene).relink_face(f);
        }
    }
    fn name(&self) -> &'static str {
        "DeleteFaceCommand"
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

// ---- FlipFaces ---------------------------------------------------------------------------

/// `FlipFacesCommand`: reverse the winding of a set of faces, which may
/// live in the loose mesh or in a group's. An involution: undo flips again.
pub struct FlipFacesCommand {
    pub faces: Vec<(MeshId, FaceId)>,
}

impl FlipFacesCommand {
    fn flip(&self, scene: &mut Scene) {
        for &(mesh, f) in &self.faces {
            let face = scene.meshes[mesh.0 as usize].face_mut(f);
            face.outer.reverse();
            for h in &mut face.holes {
                h.reverse();
            }
        }
    }
}

impl Command for FlipFacesCommand {
    fn apply(&mut self, scene: &mut Scene, _env: &mut Env) -> CmdResult {
        self.flip(scene);
        Ok(())
    }
    fn undo(&mut self, scene: &mut Scene, _env: &mut Env) {
        self.flip(scene);
    }
    fn name(&self) -> &'static str {
        "FlipFacesCommand"
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

// ---- Compounds ---------------------------------------------------------------------------

/// `CompoundCommand`: a list of commands executed and reverted as one step.
pub struct CompoundCommand {
    pub commands: Vec<Box<dyn Command>>,
}

impl Command for CompoundCommand {
    fn apply(&mut self, scene: &mut Scene, env: &mut Env) -> CmdResult {
        let mut done = 0;
        let mut failure = None;
        for cmd in self.commands.iter_mut() {
            match cmd.apply(scene, env) {
                Ok(()) => done += 1,
                Err(e) => {
                    failure = Some(e);
                    break;
                }
            }
        }
        match failure {
            None => Ok(()),
            Some(CommandError::Rejected(why)) => {
                // One part declined: the rest goes back too.
                for cmd in self.commands[..done].iter_mut().rev() {
                    cmd.undo(scene, env);
                }
                Err(CommandError::Rejected(why))
            }
            Some(e) => Err(e),
        }
    }
    fn undo(&mut self, scene: &mut Scene, env: &mut Env) {
        for cmd in self.commands.iter_mut().rev() {
            cmd.undo(scene, env);
        }
    }
    fn name(&self) -> &'static str {
        "CompoundCommand"
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// `SnapshotCompound`: run sub-commands under one identity-preserving
/// snapshot, undone by restoring it. After the plan, the coplanar overlap it
/// made is healed and split curves get their own ids, before the snapshot
/// of the result that redo restores.
pub struct SnapshotCompound {
    pub inner: Vec<Box<dyn Command>>,
    before: Option<State>,
    after: Option<State>,
}

impl SnapshotCompound {
    pub fn new(inner: Vec<Box<dyn Command>>) -> Self {
        SnapshotCompound { inner, before: None, after: None }
    }
}

impl Command for SnapshotCompound {
    fn apply(&mut self, scene: &mut Scene, env: &mut Env) -> CmdResult {
        if self.after.is_none() {
            let h = env.hypot;
            self.before = Some(env.mesh(scene).capture_state());
            // Which way every face looked before the draw: the heal may align
            // the faces the draw MADE, never the ones already there.
            let footprint = topology::winding_footprint(env.mesh(scene), h);
            for cmd in self.inner.iter_mut() {
                cmd.apply(scene, env)?;
            }
            let m = env.mesh_mut(scene);
            topology::heal_overlapping_faces(m, 0.5, None, footprint.as_deref(), h, &mut |mm: &mut Mesh| {
                orient::orient_outward(mm, h);
            });
            let mut counter = env.curve;
            m.resplit_curves(&mut counter);
            env.curve = counter;
            self.after = Some(m.capture_state());
        } else if let Some(s) = &self.after {
            env.mesh_mut(scene).restore_state(s);
        }
        Ok(())
    }
    fn undo(&mut self, scene: &mut Scene, env: &mut Env) {
        if let Some(s) = &self.before {
            env.mesh_mut(scene).restore_state(s);
        }
    }
    fn name(&self) -> &'static str {
        "SnapshotCompound"
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// `MeshSnapshotCommand`: sub-commands plus the stitch pass under one
/// snapshot (push/pull's shape).
pub struct MeshSnapshotCommand {
    pub inner: Vec<Box<dyn Command>>,
    pub seed: Vec<V3>,
    snapshot: Option<State>,
}

impl MeshSnapshotCommand {
    pub fn new(inner: Vec<Box<dyn Command>>, seed: Vec<V3>) -> Self {
        MeshSnapshotCommand { inner, seed, snapshot: None }
    }
}

impl Command for MeshSnapshotCommand {
    fn apply(&mut self, scene: &mut Scene, env: &mut Env) -> CmdResult {
        let h = env.hypot;
        self.snapshot = Some(env.mesh(scene).capture_state());
        let before: HashSet<FaceId> = env.mesh(scene).faces.iter().copied().collect();
        for cmd in self.inner.iter_mut() {
            cmd.apply(scene, env)?;
        }
        let m = env.mesh_mut(scene);
        let mut new_faces: HashSet<FaceId> = m.faces.iter().copied().filter(|f| !before.contains(f)).collect();
        stitch::run_stitch(m, &stitch::seed_keys(&self.seed), Some(&mut new_faces), true, true, h);
        Ok(())
    }
    fn undo(&mut self, scene: &mut Scene, env: &mut Env) {
        if let Some(s) = &self.snapshot {
            env.mesh_mut(scene).restore_state(s);
        }
    }
    fn name(&self) -> &'static str {
        "MeshSnapshotCommand"
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

// ---- AutoTagDrawnFaces ----------------------------------------------------------------------

/// `AutoTagDrawnFacesCommand`: the last step of a draw plan when a BIM class
/// is active: stamps it, with a fresh object id, on the faces the plan's own
/// `AddFaceCommand`s made. Its undo is a no-op (the snapshot restore
/// reverts the attrs).
pub struct AutoTagDrawnFacesCommand {
    pub face_commands: Vec<Rc<Cell<Option<FaceId>>>>,
    pub tag: Vec<(String, Value)>,
}

impl Command for AutoTagDrawnFacesCommand {
    fn apply(&mut self, scene: &mut Scene, env: &mut Env) -> CmdResult {
        // `next_object_id(scene)`: over the LOOSE mesh's faces.
        let mut top: i64 = 0;
        {
            let loose = scene.loose();
            for &f in &loose.faces {
                let attrs = &loose.face(f).attrs;
                if let Some((_, tag)) = attrs.iter().find(|(k, _)| k == "ifc") {
                    if pyval::truthy(tag) {
                        if let Some(id) = tag.get("id").and_then(|v| match v {
                            Value::Int(_) => v.as_i64(),
                            _ => None,
                        }) {
                            top = top.max(id);
                        }
                    }
                }
            }
        }
        let mut tag = self.tag.clone();
        pyval::set(&mut tag, "id", Value::int(top + 1));
        let m = env.mesh_mut(scene);
        let live: HashSet<FaceId> = m.faces.iter().copied().collect();
        for fc in &self.face_commands {
            if let Some(f) = fc.get() {
                if live.contains(&f) {
                    let has = m.face(f).attrs.iter().find(|(k, _)| k == "ifc").is_some_and(|(_, v)| pyval::truthy(v));
                    if !has {
                        pyval::set(&mut m.face_mut(f).attrs, "ifc", Value::Object(tag.clone()));
                    }
                }
            }
        }
        Ok(())
    }
    fn undo(&mut self, _scene: &mut Scene, _env: &mut Env) {}
    fn name(&self) -> &'static str {
        "AutoTagDrawnFacesCommand"
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

// ---- The stitch commands as Commands ------------------------------------------------------------

impl Command for StitchSolidCommand {
    fn apply(&mut self, scene: &mut Scene, env: &mut Env) -> CmdResult {
        let h = env.hypot;
        StitchSolidCommand::apply(self, env.mesh_mut(scene), h);
        Ok(())
    }
    fn undo(&mut self, scene: &mut Scene, env: &mut Env) {
        StitchSolidCommand::undo(self, env.mesh_mut(scene));
    }
    fn name(&self) -> &'static str {
        "StitchSolidCommand"
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl Command for CoplanarMergeCommand {
    fn apply(&mut self, scene: &mut Scene, env: &mut Env) -> CmdResult {
        let h = env.hypot;
        CoplanarMergeCommand::apply(self, env.mesh_mut(scene), h);
        Ok(())
    }
    fn undo(&mut self, scene: &mut Scene, env: &mut Env) {
        CoplanarMergeCommand::undo(self, env.mesh_mut(scene));
    }
    fn name(&self) -> &'static str {
        "CoplanarMergeCommand"
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl Command for PruneOrphanEdgesCommand {
    fn apply(&mut self, scene: &mut Scene, env: &mut Env) -> CmdResult {
        PruneOrphanEdgesCommand::apply(self, env.mesh_mut(scene));
        Ok(())
    }
    fn undo(&mut self, scene: &mut Scene, env: &mut Env) {
        PruneOrphanEdgesCommand::undo(self, env.mesh_mut(scene));
    }
    fn name(&self) -> &'static str {
        "PruneOrphanEdgesCommand"
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

// ---- Small attribute commands ----------------------------------------------------------------------

fn attr(m: &Mesh, f: FaceId, k: &str) -> Option<Value> {
    m.face(f).attrs.iter().find(|(n, _)| n == k).map(|(_, v)| v.clone())
}

fn attr_set(m: &mut Mesh, f: FaceId, k: &str, v: Option<Value>) {
    match v {
        Some(v) => pyval::set(&mut m.face_mut(f).attrs, k, v),
        None => m.face_mut(f).attrs.retain(|(n, _)| n != k),
    }
}

/// `SetFaceColorCommand` / `SetFaceOpacityCommand` / the material tag: one
/// attrs key swapped on a set of faces; undo puts each face's old value back.
pub struct SetFaceAttrCommand {
    class: &'static str,
    key: &'static str,
    pub faces: Vec<(MeshId, FaceId)>,
    pub value: Option<Value>,
    old: Option<Vec<Option<Value>>>,
    /// `SetFaceMaterialTagCommand` only: the registry entry to add.
    material: Option<(String, Value)>,
    registered: bool,
}

impl SetFaceAttrCommand {
    pub fn color(faces: Vec<(MeshId, FaceId)>, color: Option<Value>) -> Self {
        Self::make("SetFaceColorCommand", "color", faces, color)
    }
    pub fn opacity(faces: Vec<(MeshId, FaceId)>, opacity: Option<Value>) -> Self {
        Self::make("SetFaceOpacityCommand", "opacity", faces, opacity)
    }
    pub fn texture(faces: Vec<(MeshId, FaceId)>, texture: Option<Value>) -> Self {
        Self::make("SetFaceTextureCommand", "texture", faces, texture)
    }
    pub fn material_tag(faces: Vec<(MeshId, FaceId)>, name: Option<Value>, material: Option<(String, Value)>) -> Self {
        let mut c = Self::make("SetFaceMaterialTagCommand", "mat", faces, name);
        c.material = material;
        c
    }
    fn make(class: &'static str, key: &'static str, faces: Vec<(MeshId, FaceId)>, value: Option<Value>) -> Self {
        SetFaceAttrCommand { class, key, faces, value, old: None, material: None, registered: false }
    }
}

impl Command for SetFaceAttrCommand {
    fn apply(&mut self, scene: &mut Scene, _env: &mut Env) -> CmdResult {
        if self.old.is_none() {
            self.old = Some(self.faces.iter().map(|&(m, f)| attr(&scene.meshes[m.0 as usize], f, self.key)).collect());
        }
        if let (Some((name, mat)), Some(Value::Str(n))) = (&self.material, &self.value) {
            if !n.is_empty() && !scene.materials.iter().any(|(k, _)| k == name) && name == n {
                scene.materials.push((name.clone(), mat.clone()));
                self.registered = true;
            }
        }
        for &(m, f) in &self.faces {
            attr_set(&mut scene.meshes[m.0 as usize], f, self.key, self.value.clone());
        }
        Ok(())
    }
    fn undo(&mut self, scene: &mut Scene, _env: &mut Env) {
        let old = self.old.clone().unwrap_or_default();
        for (&(m, f), o) in self.faces.iter().zip(old) {
            attr_set(&mut scene.meshes[m.0 as usize], f, self.key, o);
        }
        if self.registered {
            if let Some((name, _)) = &self.material {
                scene.materials.retain(|(k, _)| k != name);
            }
            self.registered = false;
        }
    }
    fn name(&self) -> &'static str {
        self.class
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// `SetFaceBackCommand`: paint the BACK side of faces, one value per face
/// (`attrs["back"]`; `None` clears it). The old value may be `true` (a
/// two-sided face) and comes back as such.
pub struct SetFaceBackCommand {
    pub faces: Vec<(MeshId, FaceId)>,
    pub backs: Vec<Option<Value>>,
    old: Option<Vec<Option<Value>>>,
}

impl SetFaceBackCommand {
    pub fn new(faces: Vec<(MeshId, FaceId)>, backs: Vec<Option<Value>>) -> Self {
        SetFaceBackCommand { faces, backs, old: None }
    }
}

impl Command for SetFaceBackCommand {
    fn apply(&mut self, scene: &mut Scene, _env: &mut Env) -> CmdResult {
        if self.old.is_none() {
            self.old = Some(self.faces.iter().map(|&(m, f)| attr(&scene.meshes[m.0 as usize], f, "back")).collect());
        }
        for (&(m, f), b) in self.faces.iter().zip(self.backs.iter()) {
            attr_set(&mut scene.meshes[m.0 as usize], f, "back", b.clone());
        }
        Ok(())
    }
    fn undo(&mut self, scene: &mut Scene, _env: &mut Env) {
        let old = self.old.clone().unwrap_or_default();
        for (&(m, f), o) in self.faces.iter().zip(old) {
            attr_set(&mut scene.meshes[m.0 as usize], f, "back", o);
        }
    }
    fn name(&self) -> &'static str {
        "SetFaceBackCommand"
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// `HideEdgesCommand`: hide or unhide edges (a hidden edge stays in the
/// topology). Previous flags are captured at `do` time.
pub struct HideEdgesCommand {
    pub edges: Vec<(MeshId, EdgeId)>,
    pub hidden: bool,
    prev: Vec<bool>,
}

impl HideEdgesCommand {
    pub fn new(edges: Vec<(MeshId, EdgeId)>, hidden: bool) -> Self {
        HideEdgesCommand { edges, hidden, prev: Vec::new() }
    }
}

impl Command for HideEdgesCommand {
    fn apply(&mut self, scene: &mut Scene, _env: &mut Env) -> CmdResult {
        self.prev = self.edges.iter().map(|&(m, e)| scene.meshes[m.0 as usize].edge(e).hidden).collect();
        for &(m, e) in &self.edges {
            scene.meshes[m.0 as usize].edge_mut(e).hidden = self.hidden;
        }
        Ok(())
    }
    fn undo(&mut self, scene: &mut Scene, _env: &mut Env) {
        for (&(m, e), &p) in self.edges.iter().zip(self.prev.iter()) {
            scene.meshes[m.0 as usize].edge_mut(e).hidden = p;
        }
    }
    fn name(&self) -> &'static str {
        "HideEdgesCommand"
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// `ExplodeCurvesCommand`: the curves the given edges belong to become
/// plain segments; undo gives every edge its id back.
pub struct ExplodeCurvesCommand {
    /// The curve ids of the edges given (taken when the command is built).
    pub curve_ids: Vec<i64>,
    ids: Vec<(EdgeId, i64)>,
}

impl ExplodeCurvesCommand {
    pub fn new(curve_ids: Vec<i64>) -> Self {
        ExplodeCurvesCommand { curve_ids, ids: Vec::new() }
    }
}

impl Command for ExplodeCurvesCommand {
    fn apply(&mut self, scene: &mut Scene, env: &mut Env) -> CmdResult {
        let m = env.mesh_mut(scene);
        self.ids = m
            .edges
            .iter()
            .copied()
            .filter_map(|e| m.edge(e).curve.filter(|c| self.curve_ids.contains(c)).map(|c| (e, c)))
            .collect();
        for &(e, _) in &self.ids {
            m.edge_mut(e).curve = None;
        }
        Ok(())
    }
    fn undo(&mut self, scene: &mut Scene, env: &mut Env) {
        let m = env.mesh_mut(scene);
        for &(e, c) in &self.ids {
            m.edge_mut(e).curve = Some(c);
        }
    }
    fn name(&self) -> &'static str {
        "ExplodeCurvesCommand"
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// `_PruneOnUndo` (tools/paste.py): first in a stamp's compound, so its undo
/// runs last and takes the vertices nothing uses any more.
#[derive(Default)]
pub struct PruneOnUndo {
    gone: Vec<VertexId>,
}

impl Command for PruneOnUndo {
    fn apply(&mut self, scene: &mut Scene, env: &mut Env) -> CmdResult {
        let m = env.mesh_mut(scene);
        m.relink_vertices(&self.gone);
        self.gone.clear();
        Ok(())
    }
    fn undo(&mut self, scene: &mut Scene, env: &mut Env) {
        let m = env.mesh_mut(scene);
        let before = m.vertices.clone();
        m.prune_orphan_vertices();
        let kept: HashSet<VertexId> = m.vertices.iter().copied().collect();
        // Python's dict keeps one entry per vertex, in list order.
        let mut seen = HashSet::new();
        self.gone = before.into_iter().filter(|v| seen.insert(*v)).filter(|v| !kept.contains(v)).collect();
    }
    fn name(&self) -> &'static str {
        "_PruneOnUndo"
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[allow(unused_imports)]
use mesh_key as _;
#[allow(dead_code)]
fn _hypot_marker(_: Hypot) {}
