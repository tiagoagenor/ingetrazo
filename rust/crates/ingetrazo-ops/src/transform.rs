// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! The transform commands of `core/history.py`: move, rotate, scale and flip
//! (loose vertices and whole groups) and Change Axes.
//!
//! Matrices follow Qt's type flags: every `QMatrix4x4` Python builds with
//! `translate`/`rotate`/`scale`/`*` is built here with the same calls in
//! the same order, and a matrix read from a file is `M4::from_data`
//! (General). Python floats (`degrees`, scale factors) reach Qt as float32
//! (`as f32`), and `-self.degrees` is negated as a double first.
//!
//! Snapshots. `MoveVertices`, `RotateVertices`, `ScaleVertices` and
//! `FlipVertices` undo and redo by `capture_state`/`restore_state` of the
//! mesh, like Python; the group commands compose into `xform` for an
//! instance (their undo is the inverse composition, not a restore) and walk
//! the group's own mesh for a classic group (snapshot undo).

use std::collections::HashSet;

use ingetrazo_math::{M4, V3};
use ingetrazo_model::mesh::{key, Key, Mesh, State};
use ingetrazo_model::scene::{Group, MeshId, Scene};
use ingetrazo_model::topology::{self, new_defects, PLANAR_TOLERANCE};
use ingetrazo_model::{FaceId, VertexId};

use crate::command::{Command, Counters, Cx, Fail};
use crate::group::{carry_axes, frame_matrix, group_at, group_at_mut, placement_paths, remap_uvws, transformed_mesh, Path};

fn mesh_mut<'a>(scene: &'a mut Scene, cx: &Cx) -> &'a mut Mesh {
    &mut scene.meshes[cx.mesh.0 as usize]
}

// ---- matrices ----------------------------------------------------------------------

/// `rotation_matrix`: rigid rotation of `degrees` around `axis` through `center`.
pub fn rotation_matrix(center: V3, axis: V3, degrees: f64, h: ingetrazo_math::Hypot) -> M4 {
    let mut m = M4::identity();
    m.translate(center.0, center.1, center.2);
    let a = axis.normalized(h);
    m.rotate(degrees as f32, a.0, a.1, a.2);
    let c = center.neg();
    m.translate(c.0, c.1, c.2);
    m
}

/// `mirror_matrix`: reflection about the plane through `centre` with normal
/// `axis` (Householder). The entries are Python float arithmetic over the
/// float32 components, converted to float32 by the matrix constructor.
pub fn mirror_matrix(centre: V3, axis: V3, h: ingetrazo_math::Hypot) -> M4 {
    let n = axis.normalized(h);
    let (x, y, z) = (n.0 as f64, n.1 as f64, n.2 as f64);
    let r = M4::from_rows([
        (1.0 - 2.0 * x * x) as f32,
        (-2.0 * x * y) as f32,
        (-2.0 * x * z) as f32,
        0.0,
        (-2.0 * y * x) as f32,
        (1.0 - 2.0 * y * y) as f32,
        (-2.0 * y * z) as f32,
        0.0,
        (-2.0 * z * x) as f32,
        (-2.0 * z * y) as f32,
        (1.0 - 2.0 * z * z) as f32,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
    ]);
    let mut t_in = M4::identity();
    let c = centre.neg();
    t_in.translate(c.0, c.1, c.2);
    let mut t_out = M4::identity();
    t_out.translate(centre.0, centre.1, centre.2);
    t_out.mul(&r).mul(&t_in)
}

/// A scale factor: one number, or one per axis.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Factor {
    Uniform(f64),
    Axes([f64; 3]),
}

/// `scale_matrix`: scale about `center`; with `axes` (red, green, blue unit
/// vectors) the three factors act along those.
pub fn scale_matrix(center: V3, factor: Factor, axes: Option<[V3; 3]>) -> M4 {
    let neg = center.neg();
    if let (Some([x, y, z]), Factor::Axes(f)) = (axes, factor) {
        let rot = frame_matrix(V3(0.0, 0.0, 0.0), x, y, z);
        let inv = rot.transposed();
        let mut s = M4::identity();
        s.scale(f[0] as f32, f[1] as f32, f[2] as f32);
        let mut m = M4::identity();
        m.translate(center.0, center.1, center.2);
        let m = m.mul(&rot).mul(&s).mul(&inv);
        let mut back = M4::identity();
        back.translate(neg.0, neg.1, neg.2);
        return m.mul(&back);
    }
    let mut m = M4::identity();
    m.translate(center.0, center.1, center.2);
    match factor {
        Factor::Axes(f) => m.scale(f[0] as f32, f[1] as f32, f[2] as f32),
        Factor::Uniform(f) => m.scale1(f as f32),
    }
    m.translate(neg.0, neg.1, neg.2);
    m
}

/// `invert_scale_factor`: the factor that undoes `factor`; zero factors
/// invert to 1.0.
pub fn invert_scale_factor(factor: Factor) -> Factor {
    let inv = |f: f64| if f.abs() > 1e-12 { 1.0 / f } else { 1.0 };
    match factor {
        Factor::Axes(f) => Factor::Axes([inv(f[0]), inv(f[1]), inv(f[2])]),
        Factor::Uniform(f) => Factor::Uniform(inv(f)),
    }
}

// ---- moving points -------------------------------------------------------------------

fn keys_of(src: &[V3]) -> HashSet<Key> {
    src.iter().map(|&p| key(p)).collect()
}

fn vertices_with_keys(mesh: &Mesh, keys: &HashSet<Key>) -> Vec<VertexId> {
    mesh.vertices.iter().copied().filter(|&v| keys.contains(&key(mesh.position(v)))).collect()
}

/// `translate_points`: move every shared vertex whose position key is in
/// `keys` by `delta`.
pub fn translate_points(mesh: &mut Mesh, keys: &HashSet<Key>, delta: V3) {
    for v in vertices_with_keys(mesh, keys) {
        mesh.move_vertex(v, delta);
    }
}

/// `rotate_points`: move every shared vertex whose key is in `keys` through
/// the matrix (`move_vertex(v, m.map(p) - p)`).
pub fn rotate_points(mesh: &mut Mesh, keys: &HashSet<Key>, m: &M4) {
    for v in vertices_with_keys(mesh, keys) {
        let p = mesh.position(v);
        mesh.move_vertex(v, m.map(p).sub(p));
    }
}

/// `_weld_landed`: merge a moved vertex into the one it landed on.
fn weld_landed(mesh: &mut Mesh, moving: &HashSet<VertexId>) -> bool {
    let moved_keys: HashSet<Key> = mesh.vertices.iter().filter(|v| moving.contains(v)).map(|&v| key(mesh.position(v))).collect();
    if moved_keys.is_empty() {
        return false;
    }
    if !mesh.vertices.iter().any(|v| !moving.contains(v) && moved_keys.contains(&key(mesh.position(*v)))) {
        return false;
    }
    mesh.weld_coincident();
    true
}

fn fold(mesh: &mut Mesh) {
    let h = mesh.hypot;
    topology::fold_nonplanar_faces(mesh, PLANAR_TOLERANCE, h);
}

/// The words a refused weld uses for each defect kind (`defect_names`).
fn defect_name(kind: &str) -> &'static str {
    match kind {
        "duplicate_face" => "a face doubled over another",
        "overlapping_face" => "two faces overlapping in one plane",
        "pinched_face" => "a face that touches itself",
        "crossed_face" => "a face whose sides cross",
        "flat_face" => "a face with no area",
        _ => "an edge of zero length",
    }
}

/// `MoveVerticesCommand`: translate every shared vertex at a set of
/// positions, then autofold; a move that lands on other vertices joins them
/// and is refused when the join leaves a defect that was not there before.
pub struct MoveVertices {
    pub src: Vec<V3>,
    pub delta: V3,
    before: Option<State>,
    after: Option<State>,
    /// Whether the move joined moved vertices into ones it landed on.
    pub welded: bool,
}

impl MoveVertices {
    pub fn new(positions: &[V3], delta: V3) -> Self {
        MoveVertices { src: positions.to_vec(), delta, before: None, after: None, welded: false }
    }
}

impl Command for MoveVertices {
    fn name(&self) -> &'static str {
        "move_vertices"
    }

    fn apply(&mut self, scene: &mut Scene, cx: &mut Cx) -> Result<(), Fail> {
        let mesh = mesh_mut(scene, cx);
        if let Some(after) = &self.after {
            mesh.restore_state(after); // redo
            return Ok(());
        }
        let h = mesh.hypot;
        self.before = Some(mesh.capture_state());
        let keys = keys_of(&self.src);
        let moving: HashSet<VertexId> = vertices_with_keys(mesh, &keys).into_iter().collect();
        let landing: HashSet<Key> = self.src.iter().map(|&p| key(p.add(self.delta))).collect();
        let mut before = None;
        if mesh.vertices.iter().any(|v| !moving.contains(v) && landing.contains(&key(mesh.position(*v)))) {
            // Something stands where the move lands: note how healthy the
            // neighbourhood is now, to judge the join against it.
            let near: Vec<VertexId> = mesh
                .vertices
                .iter()
                .copied()
                .filter(|&v| {
                    let k = key(mesh.position(v));
                    keys.contains(&k) || landing.contains(&k)
                })
                .collect();
            before = Some(topology::defects_near(mesh, &near, 1e-10, h));
        }
        translate_points(mesh, &keys, self.delta);
        self.welded = weld_landed(mesh, &moving);
        fold(mesh);
        if self.welded {
            if let Some(before) = before {
                let near: Vec<VertexId> = mesh.vertices.iter().copied().filter(|&v| landing.contains(&key(mesh.position(v)))).collect();
                let after = topology::defects_near(mesh, &near, 1e-10, h);
                let worse = new_defects(&before, &after);
                if let Some(first) = worse.first() {
                    if let Some(b) = &self.before {
                        mesh.restore_state(b);
                    }
                    self.welded = false;
                    return Err(Fail::Rejected(format!("Not joined: it would leave {}. The move was undone.", defect_name(first))));
                }
            }
        }
        self.after = Some(mesh.capture_state());
        Ok(())
    }

    fn undo(&mut self, scene: &mut Scene, cx: &mut Cx) {
        if let Some(b) = &self.before {
            mesh_mut(scene, cx).restore_state(b);
        }
    }
}

/// The snapshot commands of loose vertices share their shape: capture,
/// map the points, autofold, capture.
macro_rules! snapshot_undo {
    () => {
        fn undo(&mut self, scene: &mut Scene, cx: &mut Cx) {
            if let Some(b) = &self.before {
                mesh_mut(scene, cx).restore_state(b);
            }
        }
    };
}

/// `RotateVerticesCommand`.
pub struct RotateVertices {
    pub src: Vec<V3>,
    pub center: V3,
    pub axis: V3,
    pub degrees: f64,
    before: Option<State>,
    after: Option<State>,
}

impl RotateVertices {
    pub fn new(positions: &[V3], center: V3, axis: V3, degrees: f64) -> Self {
        RotateVertices { src: positions.to_vec(), center, axis, degrees, before: None, after: None }
    }
}

impl Command for RotateVertices {
    fn name(&self) -> &'static str {
        "rotate_vertices"
    }

    fn apply(&mut self, scene: &mut Scene, cx: &mut Cx) -> Result<(), Fail> {
        let mesh = mesh_mut(scene, cx);
        if let Some(after) = &self.after {
            mesh.restore_state(after);
            return Ok(());
        }
        self.before = Some(mesh.capture_state());
        let m = rotation_matrix(self.center, self.axis, self.degrees, mesh.hypot);
        rotate_points(mesh, &keys_of(&self.src), &m);
        fold(mesh);
        self.after = Some(mesh.capture_state());
        Ok(())
    }

    snapshot_undo!();
}

/// `ScaleVerticesCommand`.
pub struct ScaleVertices {
    pub src: Vec<V3>,
    pub center: V3,
    pub factor: Factor,
    pub axes: Option<[V3; 3]>,
    before: Option<State>,
    after: Option<State>,
}

impl ScaleVertices {
    pub fn new(positions: &[V3], center: V3, factor: Factor, axes: Option<[V3; 3]>) -> Self {
        ScaleVertices { src: positions.to_vec(), center, factor, axes, before: None, after: None }
    }
}

impl Command for ScaleVertices {
    fn name(&self) -> &'static str {
        "scale_vertices"
    }

    fn apply(&mut self, scene: &mut Scene, cx: &mut Cx) -> Result<(), Fail> {
        let mesh = mesh_mut(scene, cx);
        if let Some(after) = &self.after {
            mesh.restore_state(after);
            return Ok(());
        }
        self.before = Some(mesh.capture_state());
        let m = scale_matrix(self.center, self.factor, self.axes);
        rotate_points(mesh, &keys_of(&self.src), &m); // the generic mapper
        fold(mesh);
        self.after = Some(mesh.capture_state());
        Ok(())
    }

    snapshot_undo!();
}

/// `FlipVerticesCommand`: mirror loose positions about an axis plane and
/// re-reverse the fully selected faces' windings.
pub struct FlipVertices {
    pub src: Vec<V3>,
    pub centre: V3,
    pub axis: V3,
    pub faces: Vec<FaceId>,
    before: Option<State>,
    after: Option<State>,
}

impl FlipVertices {
    pub fn new(positions: &[V3], centre: V3, axis: V3, faces: &[FaceId]) -> Self {
        FlipVertices { src: positions.to_vec(), centre, axis, faces: faces.to_vec(), before: None, after: None }
    }
}

fn reverse_loops(mesh: &mut Mesh, f: FaceId) {
    let face = mesh.face_mut(f);
    face.outer.reverse();
    for h in &mut face.holes {
        h.reverse();
    }
}

impl Command for FlipVertices {
    fn name(&self) -> &'static str {
        "flip_vertices"
    }

    fn apply(&mut self, scene: &mut Scene, cx: &mut Cx) -> Result<(), Fail> {
        let mesh = mesh_mut(scene, cx);
        if let Some(after) = &self.after {
            mesh.restore_state(after);
            return Ok(());
        }
        self.before = Some(mesh.capture_state());
        let m = mirror_matrix(self.centre, self.axis, mesh.hypot);
        rotate_points(mesh, &keys_of(&self.src), &m);
        for &f in &self.faces {
            reverse_loops(mesh, f);
        }
        fold(mesh);
        self.after = Some(mesh.capture_state());
        Ok(())
    }

    snapshot_undo!();
}

// ---- groups --------------------------------------------------------------------------

/// The arena and the group tree of a scene, borrowed apart.
fn split(scene: &mut Scene) -> (&mut Vec<Mesh>, &mut Vec<Group>) {
    (&mut scene.meshes, &mut scene.groups)
}

fn target<'a>(groups: &'a mut [Group], path: &[usize]) -> Result<&'a mut Group, Fail> {
    group_at_mut(groups, path).ok_or_else(|| Fail::Failed("the group is no longer in the scene".into()))
}

/// Map every vertex of `mesh` through `m` (`move_vertex(v, m.map(p) - p)`).
fn map_all_vertices(mesh: &mut Mesh, m: &M4) {
    for v in mesh.vertices.clone() {
        let p = mesh.position(v);
        mesh.move_vertex(v, m.map(p).sub(p));
    }
}

/// `MoveGroupCommand`: translate a whole group. An instance composes the
/// translation into its `xform`; a classic group walks its vertices, its
/// texture maps and its axes. Undo shifts by `-delta` (not a restore).
pub struct MoveGroup {
    pub path: Path,
    pub delta: V3,
}

impl MoveGroup {
    pub fn new(path: &[usize], delta: V3) -> Self {
        MoveGroup { path: path.to_vec(), delta }
    }

    fn shift(&self, scene: &mut Scene, delta: V3) -> Result<(), Fail> {
        let (meshes, groups) = split(scene);
        let g = target(groups, &self.path)?;
        let mut t = M4::identity();
        t.translate(delta.0, delta.1, delta.2);
        if let Some(x) = g.xform {
            // Component instance: compose into the transform — O(1), and
            // the shared prototype mesh (siblings!) is never touched.
            g.xform = Some(t.mul(&x));
        } else {
            let mesh = &mut meshes[g.mesh.0 as usize];
            for v in mesh.vertices.clone() {
                mesh.move_vertex(v, delta);
            }
            remap_uvws(mesh, &mut t); // the texture travels along
            carry_axes(g, &t); // and its axes (#44)
        }
        Ok(())
    }
}

impl Command for MoveGroup {
    fn name(&self) -> &'static str {
        "move_group"
    }
    fn apply(&mut self, scene: &mut Scene, _cx: &mut Cx) -> Result<(), Fail> {
        self.shift(scene, self.delta)
    }
    fn undo(&mut self, scene: &mut Scene, _cx: &mut Cx) {
        let _ = self.shift(scene, self.delta.neg());
    }
}

/// What the classic-group commands remember: the group's mesh and axes
/// before and after.
#[derive(Default)]
struct GroupSnap {
    before: Option<State>,
    after: Option<State>,
    axes_before: Option<M4>,
    axes_after: Option<M4>,
}

/// Apply `m` to a classic group: every vertex, the texture maps, the axes;
/// snapshot before and after. A redo restores the second snapshot.
fn transform_classic(meshes: &mut [Mesh], g: &mut Group, snap: &mut GroupSnap, m: &mut M4) {
    let mesh = &mut meshes[g.mesh.0 as usize];
    if let Some(after) = &snap.after {
        mesh.restore_state(after); // redo
        g.axes = snap.axes_after;
        return;
    }
    snap.before = Some(mesh.capture_state());
    snap.axes_before = g.axes;
    map_all_vertices(mesh, m);
    remap_uvws(mesh, m); // the texture turns with it
    carry_axes(g, m); // and so do its axes (#44)
    snap.axes_after = g.axes;
    snap.after = Some(mesh.capture_state());
}

fn restore_classic(meshes: &mut [Mesh], g: &mut Group, snap: &GroupSnap) {
    if let Some(b) = &snap.before {
        meshes[g.mesh.0 as usize].restore_state(b);
        g.axes = snap.axes_before;
    }
}

/// `RotateGroupCommand`.
pub struct RotateGroup {
    pub path: Path,
    pub center: V3,
    pub axis: V3,
    pub degrees: f64,
    snap: GroupSnap,
}

impl RotateGroup {
    pub fn new(path: &[usize], center: V3, axis: V3, degrees: f64) -> Self {
        RotateGroup { path: path.to_vec(), center, axis, degrees, snap: GroupSnap::default() }
    }
}

impl Command for RotateGroup {
    fn name(&self) -> &'static str {
        "rotate_group"
    }

    fn apply(&mut self, scene: &mut Scene, _cx: &mut Cx) -> Result<(), Fail> {
        let (meshes, groups) = split(scene);
        let h = meshes[0].hypot;
        let g = target(groups, &self.path)?;
        let mut m = rotation_matrix(self.center, self.axis, self.degrees, h);
        if let Some(x) = g.xform {
            g.xform = Some(m.mul(&x));
            return Ok(());
        }
        transform_classic(meshes, g, &mut self.snap, &mut m);
        Ok(())
    }

    fn undo(&mut self, scene: &mut Scene, _cx: &mut Cx) {
        let (meshes, groups) = split(scene);
        let h = meshes[0].hypot;
        let Ok(g) = target(groups, &self.path) else { return };
        if let Some(x) = g.xform {
            let m = rotation_matrix(self.center, self.axis, -self.degrees, h);
            g.xform = Some(m.mul(&x));
            return;
        }
        restore_classic(meshes, g, &self.snap);
    }
}

/// `ScaleGroupCommand`.
pub struct ScaleGroup {
    pub path: Path,
    pub center: V3,
    pub factor: Factor,
    pub axes: Option<[V3; 3]>,
    snap: GroupSnap,
}

impl ScaleGroup {
    pub fn new(path: &[usize], center: V3, factor: Factor, axes: Option<[V3; 3]>) -> Self {
        ScaleGroup { path: path.to_vec(), center, factor, axes, snap: GroupSnap::default() }
    }
}

impl Command for ScaleGroup {
    fn name(&self) -> &'static str {
        "scale_group"
    }

    fn apply(&mut self, scene: &mut Scene, _cx: &mut Cx) -> Result<(), Fail> {
        let (meshes, groups) = split(scene);
        let g = target(groups, &self.path)?;
        let mut m = scale_matrix(self.center, self.factor, self.axes);
        if let Some(x) = g.xform {
            g.xform = Some(m.mul(&x));
            return Ok(());
        }
        transform_classic(meshes, g, &mut self.snap, &mut m);
        Ok(())
    }

    fn undo(&mut self, scene: &mut Scene, _cx: &mut Cx) {
        let (meshes, groups) = split(scene);
        let Ok(g) = target(groups, &self.path) else { return };
        if let Some(x) = g.xform {
            let m = scale_matrix(self.center, invert_scale_factor(self.factor), self.axes);
            g.xform = Some(m.mul(&x));
            return;
        }
        restore_classic(meshes, g, &self.snap);
    }
}

/// `FlipGroupsCommand`: flip whole groups about an axis plane. Instances
/// compose the reflection into their transform; classic groups map their
/// vertices and re-reverse every face loop. A reflection is an involution:
/// undo flips again.
pub struct FlipGroups {
    pub paths: Vec<Path>,
    pub centre: V3,
    pub axis: V3,
}

impl FlipGroups {
    pub fn new(paths: &[Path], centre: V3, axis: V3) -> Self {
        FlipGroups { paths: paths.to_vec(), centre, axis }
    }

    fn flip(&self, scene: &mut Scene) {
        let (meshes, groups) = split(scene);
        let m = mirror_matrix(self.centre, self.axis, meshes[0].hypot);
        for path in &self.paths {
            let Some(g) = group_at_mut(groups, path) else { continue };
            if let Some(x) = g.xform {
                g.xform = Some(m.mul(&x));
                continue;
            }
            let mesh = &mut meshes[g.mesh.0 as usize];
            map_all_vertices(mesh, &m);
            carry_axes(g, &m); // an involution, like the rest
            for f in mesh.faces.clone() {
                reverse_loops(mesh, f);
            }
        }
    }
}

impl Command for FlipGroups {
    fn name(&self) -> &'static str {
        "flip_groups"
    }
    fn apply(&mut self, scene: &mut Scene, _cx: &mut Cx) -> Result<(), Fail> {
        self.flip(scene);
        Ok(())
    }
    fn undo(&mut self, scene: &mut Scene, _cx: &mut Cx) {
        self.flip(scene);
    }
}

// ---- Change Axes ----------------------------------------------------------------------

/// `ChangeAxesCommand` (issue #44): give a group a new frame while nothing
/// moves in the world. A classic group only swaps its `axes`; a component
/// instance re-expresses its SHARED definition in the new axes (`D =
/// frame⁻¹·xform`), takes `frame` as its placement, and every other
/// instance of the definition takes `D⁻¹` on the right.
pub struct ChangeAxes {
    pub path: Path,
    pub frame: M4,
    old_axes: Option<M4>,
    d: Option<M4>,
    instances: Vec<Path>,
}

impl ChangeAxes {
    pub fn new(path: &[usize], frame: M4) -> Self {
        ChangeAxes { path: path.to_vec(), frame, old_axes: None, d: None, instances: Vec::new() }
    }

    /// `_siblings`: every placed instance of the group's definition, then the
    /// group itself when it is not among them.
    fn siblings(&self, scene: &Scene, proto: MeshId) -> Vec<Path> {
        let mut out: Vec<Path> = Vec::new();
        for p in placement_paths(&scene.groups) {
            if let Some(g) = group_at(&scene.groups, &p) {
                if g.mesh == proto && g.xform.is_some() && !out.contains(&p) {
                    out.push(p);
                }
            }
        }
        if !out.contains(&self.path) {
            out.push(self.path.clone());
        }
        out
    }

    /// `_reexpress`: map the definition's geometry (and the group's nested
    /// placements) by `d`. The definition's mesh is replaced by the
    /// transformed copy, which is what `proto.restore_state(moved.capture_state())`
    /// leaves behind (the copy's own objects).
    fn reexpress(&self, scene: &mut Scene, counters: &mut Counters, d: &mut M4) {
        let (meshes, groups) = split(scene);
        let Some(g) = group_at_mut(groups, &self.path) else { return };
        let moved = transformed_mesh(&meshes[g.mesh.0 as usize], d, &mut counters.curve);
        meshes[g.mesh.0 as usize] = moved;
        for child in &mut g.children {
            if let Some(x) = child.xform {
                child.xform = Some(d.mul(&x));
            }
        }
    }
}

impl Command for ChangeAxes {
    fn name(&self) -> &'static str {
        "change_axes"
    }

    fn apply(&mut self, scene: &mut Scene, cx: &mut Cx) -> Result<(), Fail> {
        let g = target(&mut scene.groups, &self.path)?;
        let Some(xform) = g.xform else {
            self.old_axes = g.axes;
            g.axes = Some(self.frame);
            return Ok(());
        };
        let proto = g.mesh;
        let (inv, ok) = self.frame.inverted();
        if !ok {
            return Ok(());
        }
        let mut d = inv.mul(&xform);
        let (d_inv, ok2) = d.inverted();
        if !ok2 {
            self.d = None;
            return Ok(());
        }
        self.instances = self.siblings(scene, proto);
        // `transformed_mesh` reads `d(r, c)`: the stored matrix comes out General.
        self.reexpress(scene, &mut cx.counters, &mut d);
        self.d = Some(d);
        for p in &self.instances {
            if let Some(inst) = group_at_mut(&mut scene.groups, p) {
                if let Some(x) = inst.xform {
                    inst.xform = Some(x.mul(&d_inv));
                }
            }
        }
        Ok(())
    }

    fn undo(&mut self, scene: &mut Scene, cx: &mut Cx) {
        let Some(g) = group_at_mut(&mut scene.groups, &self.path) else { return };
        if g.xform.is_none() && self.d.is_none() {
            g.axes = self.old_axes;
            return;
        }
        let Some(d) = self.d else { return };
        let (mut d_inv, _ok) = d.inverted();
        self.reexpress(scene, &mut cx.counters, &mut d_inv);
        for p in &self.instances {
            if let Some(inst) = group_at_mut(&mut scene.groups, p) {
                if let Some(x) = inst.xform {
                    inst.xform = Some(x.mul(&d));
                }
            }
        }
    }
}
