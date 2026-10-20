// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! The Scale tool (`tools/scale.py`): the classic grip box, with no UI.
//!
//! Activating with a selection wraps it in a box with 26 grips (3D) or 8
//! (flat); with nothing selected the first click picks the entity under the
//! cursor. Corner grips scale uniformly, edge midpoints two axes, face
//! centres one; a tap of Ctrl toggles About Center, a tap of Shift Scale
//! Uniformly; the VCB takes a factor, `a;b;c`, a negative factor or an
//! absolute size (`2m`), and a typed value right after a commit redoes it.
//!
//! The box is kept in the coordinates of its frame (`_lo`/`_hi`): the
//! group's own axes when exactly one group is selected, the world's
//! otherwise. It is rebuilt when `Env::version` (`scene.version`) moved.
//!
//! Where it differs from Python (reported):
//!
//! * Reference images (`ImagePlane`) are not scaled (the box ignores them and
//!   the commit builds no `ScaleImagePlanesCommand`); a click with nothing
//!   selected still selects an image the picker finds, as Python does.
//! * The drawing axes of an open group (`core.axes`) are the world's: a tool
//!   here never runs with a non-world drawing frame.

use ingetrazo_formats::pyjson::Value;
use ingetrazo_math::{M4, V3};
use ingetrazo_model::mesh::VertexId;
use ingetrazo_model::scene::MeshId;
use ingetrazo_ops::command::Command;
use ingetrazo_ops::draw::CompoundCommand;
use ingetrazo_ops::group::{frame_axes, group_at, group_at_mut, group_frame, iter_placements, Path};
use ingetrazo_ops::transform::{scale_matrix, Factor, ScaleGroup, ScaleVertices};
use ingetrazo_snap::SnapKind;

use crate::modify_base::{keys, Env, ModifyTool, Mods, Probe, Sel, ToolCx, Vcb};
use crate::move_tool::gather_targets;

const MIN_FACTOR: f64 = 1e-4;
/// Below this extent an axis is flat: no grips, never scales.
const FLAT: f64 = 1e-6;
const AXIS_NAMES: [&str; 3] = ["Red", "Green", "Blue"];

/// What a held grip may land on (`_TARGET_KINDS`).
fn is_target_kind(k: SnapKind) -> bool {
    matches!(
        k,
        SnapKind::Endpoint
            | SnapKind::Midpoint
            | SnapKind::Intersection
            | SnapKind::Center
            | SnapKind::Origin
            | SnapKind::OnEdge
            | SnapKind::OnLine
            | SnapKind::FromPoint
            | SnapKind::Close
            | SnapKind::ComponentOrigin
            | SnapKind::ArcMidpoint
    )
}

/// `_Grip`: box-parameter position + the axes it scales.
#[derive(Clone, Debug)]
struct Grip {
    params: [f64; 3],
    mask: Vec<usize>,
}

impl Grip {
    fn kind(&self, active_axes: usize) -> &'static str {
        let n = self.mask.len();
        if n >= active_axes {
            "corner"
        } else if n == 2 {
            "edge"
        } else {
            "face"
        }
    }
}

type Ext = [f64; 3];

/// The `spec` of `_last`: `(uniform, mask, ext, active)`.
#[derive(Clone, Debug)]
struct Spec {
    uniform: bool,
    mask: Vec<usize>,
    ext: Ext,
    active: Vec<usize>,
}

/// `_last`: what `build` closes over, to redo the scale at a new value.
struct Last {
    serial: Option<u64>,
    spec: Spec,
    anchor: V3,
    axes: Option<[V3; 3]>,
    groups: Vec<Path>,
    positions: Vec<V3>,
}

/// `_orig_v`/`_orig_x`: where everything that scales stood at the grab.
struct Originals {
    verts: Vec<(MeshId, VertexId, V3)>,
    xforms: Vec<(Path, M4)>,
}

pub struct ScaleTool {
    lo: Option<V3>,
    hi: Option<V3>,
    axes: Option<[V3; 3]>,
    orig: Option<V3>,
    grips: Vec<Grip>,
    box_version: i64,
    groups: Vec<Path>,
    positions: Vec<V3>,
    verts: Vec<VertexId>,
    grip: Option<usize>,
    hover_grip: Option<usize>,
    anchor: Option<V3>,
    factors: [f64; 3],
    grabbed_px: Option<(f64, f64)>,
    moved: bool,
    pub about_center: bool,
    pub uniform: bool,
    last: Option<Last>,
    originals: Option<Originals>,
    ctrl_toggled: bool,
}

impl Default for ScaleTool {
    fn default() -> Self {
        ScaleTool::new()
    }
}

fn f64s(p: V3) -> [f64; 3] {
    [p.0 as f64, p.1 as f64, p.2 as f64]
}

fn comp(p: V3, i: usize) -> f64 {
    f64s(p)[i]
}

impl ScaleTool {
    pub fn new() -> ScaleTool {
        ScaleTool {
            lo: None,
            hi: None,
            axes: None,
            orig: None,
            grips: Vec::new(),
            box_version: -1,
            groups: Vec::new(),
            positions: Vec::new(),
            verts: Vec::new(),
            grip: None,
            hover_grip: None,
            anchor: None,
            factors: [1.0; 3],
            grabbed_px: None,
            moved: false,
            about_center: false,
            uniform: false,
            last: None,
            originals: None,
            ctrl_toggled: false,
        }
    }

    // ---- box + grips ---------------------------------------------------------

    /// `_pick_frame`: one group selected: its own axes; else the world's.
    fn pick_frame(&mut self, env: &Env) {
        let mut frame: Option<M4> = None;
        if env.sel.len() == 1 {
            if let Some(Sel::Group(p)) = env.sel.iter().next() {
                if let Some(g) = group_at(&env.scene.groups, p) {
                    frame = group_frame(g);
                }
            }
        }
        if frame.is_none() {
            self.axes = None;
            self.orig = None;
            return;
        }
        let (o, x, y, z) = frame_axes(frame.as_ref(), env.cx.hypot);
        if x == V3(1.0, 0.0, 0.0) && y == V3(0.0, 1.0, 0.0) && z == V3(0.0, 0.0, 1.0) {
            self.axes = None;
            self.orig = None;
            return;
        }
        self.axes = Some([x, y, z]);
        self.orig = Some(o);
    }

    fn to_local(&self, p: V3) -> V3 {
        match (&self.axes, self.orig) {
            (Some([x, y, z]), Some(o)) => {
                let d = p.sub(o);
                V3(d.dot(*x), d.dot(*y), d.dot(*z))
            }
            _ => p,
        }
    }

    fn to_world(&self, p: V3) -> V3 {
        match (&self.axes, self.orig) {
            (Some([x, y, z]), Some(o)) => o.add(x.mul(p.0)).add(y.mul(p.1)).add(z.mul(p.2)),
            _ => p,
        }
    }

    /// `_selection_bounds`, in the box frame.
    fn selection_bounds(&mut self, env: &Env) -> (Option<V3>, Option<V3>) {
        self.pick_frame(env);
        let mut pts: Vec<V3> = Vec::new();
        let mesh = env.cx.mesh(env.scene);
        for s in env.sel.iter() {
            match s {
                Sel::Edge(e) => {
                    let ed = mesh.edge(*e);
                    pts.push(mesh.position(ed.v0));
                    pts.push(mesh.position(ed.v1));
                }
                Sel::Face(f) => pts.extend(mesh.face(*f).outer.iter().map(|&v| mesh.position(v))),
                Sel::Group(p) => {
                    let Some(g) = group_at(&env.scene.groups, p) else { continue };
                    for (pg, m) in iter_placements(g, None) {
                        let gm = env.scene.mesh(pg.mesh);
                        for &v in &gm.vertices {
                            let q = gm.position(v);
                            pts.push(if let Some(m) = m { m.map(q) } else { q });
                        }
                    }
                }
                _ => {}
            }
        }
        if pts.is_empty() {
            return (None, None);
        }
        let mut lo = [f32::INFINITY; 3];
        let mut hi = [f32::NEG_INFINITY; 3];
        for p in pts {
            let p = self.to_local(p);
            for (i, c) in [p.0, p.1, p.2].into_iter().enumerate() {
                if c < lo[i] {
                    lo[i] = c;
                }
                if c > hi[i] {
                    hi[i] = c;
                }
            }
        }
        (Some(V3(lo[0], lo[1], lo[2])), Some(V3(hi[0], hi[1], hi[2])))
    }

    /// `_refresh_box`: keyed on `scene.version`.
    fn refresh_box(&mut self, env: &Env) {
        if self.grip.is_some() {
            return;
        }
        if *env.version as i64 == self.box_version {
            return;
        }
        self.box_version = *env.version as i64;
        let (lo, hi) = self.selection_bounds(env);
        self.lo = lo;
        self.hi = hi;
        self.grips.clear();
        self.hover_grip = None;
        if self.lo.is_none() {
            return;
        }
        let ext = self.extents();
        let active: Vec<usize> = (0..3).filter(|&i| ext[i] > FLAT).collect();
        if active.is_empty() {
            self.lo = None;
            self.hi = None;
            return;
        }
        let choices: Vec<Vec<f64>> = (0..3).map(|i| if active.contains(&i) { vec![0.0, 0.5, 1.0] } else { vec![0.5] }).collect();
        for &a in &choices[0] {
            for &b in &choices[1] {
                for &c in &choices[2] {
                    let params = [a, b, c];
                    let mask: Vec<usize> = active.iter().copied().filter(|&i| params[i] != 0.5).collect();
                    if mask.is_empty() {
                        continue; // the centre is not a grip
                    }
                    self.grips.push(Grip { params, mask });
                }
            }
        }
    }

    fn extents(&self) -> Ext {
        let (lo, hi) = (f64s(self.lo.expect("box")), f64s(self.hi.expect("box")));
        [hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]]
    }

    /// The extents, `(1, 1, 1)` where the box is missing (`_commit`).
    fn extents_or_unit(&self) -> Ext {
        if self.lo.is_some() {
            self.extents()
        } else {
            [1.0; 3]
        }
    }

    fn active_axes(&self) -> usize {
        let ext = self.extents();
        (0..3).filter(|&i| ext[i] > FLAT).count()
    }

    fn pos_of(&self, params: &[f64; 3]) -> V3 {
        let (lo, hi) = (f64s(self.lo.expect("box")), f64s(self.hi.expect("box")));
        let c = |i: usize| (lo[i] + (hi[i] - lo[i]) * params[i]) as f32;
        self.to_world(V3(c(0), c(1), c(2)))
    }

    fn grip_pos(&self, g: &Grip) -> V3 {
        self.pos_of(&g.params)
    }

    /// `_anchor_for`.
    fn anchor_for(&self, g: &Grip) -> V3 {
        if self.about_center {
            return self.to_world(self.lo.expect("box").add(self.hi.expect("box")).mul(0.5));
        }
        let mut params = [0.5; 3];
        for (i, t) in g.params.iter().enumerate() {
            if g.mask.contains(&i) {
                params[i] = 1.0 - t;
            }
        }
        self.pos_of(&params)
    }

    fn grip_under(&self, env: &Env, sx: f64, sy: f64) -> Option<usize> {
        let threshold = env.vp.pick_threshold_px();
        let (mut best, mut best_d) = (None, threshold);
        for (i, g) in self.grips.iter().enumerate() {
            let Some(px) = env.vp.world_to_pixel(self.grip_pos(g)) else { continue };
            let d = ((px.0 - sx) * (px.0 - sx) + (px.1 - sy) * (px.1 - sy)).sqrt();
            if d < best_d {
                best = Some(i);
                best_d = d;
            }
        }
        best
    }

    // ---- cursor -> factors -----------------------------------------------------

    /// `_closest_on_line`: the point on line `(a, u)` closest to ray `(o, d)`.
    fn closest_on_line(o: V3, d: V3, a: V3, u: V3) -> Option<V3> {
        let b = d.dot(u) as f64;
        let denom = 1.0 - b * b;
        if denom.abs() < 1e-9 {
            return None;
        }
        let w = o.sub(a);
        let d0 = d.dot(w) as f64;
        let e = u.dot(w) as f64;
        let s = (e - b * d0) / denom;
        Some(a.add(u.mul(s as f32)))
    }

    fn snapped_point(cx: &ToolCx) -> Option<V3> {
        let snap = cx.snap.as_ref()?;
        if !is_target_kind(snap.kind) {
            return None;
        }
        Some(snap.point)
    }

    /// `_factors_from_point`.
    fn factors_from_point(&self, env: &Env, p: V3) -> Option<[f64; 3]> {
        let h = env.cx.hypot;
        let grip = &self.grips[self.grip.expect("grip")];
        let anchor = self.anchor.expect("anchor");
        let g0 = self.grip_pos(grip);
        let uniform = self.uniform || grip.kind(self.active_axes()) == "corner";
        if uniform || grip.mask.len() == 1 {
            let axis_dir = g0.sub(anchor);
            let length = axis_dir.length(h) as f64;
            if length < 1e-12 {
                return None;
            }
            let f = p.sub(anchor).dot(axis_dir) as f64 / (length * length);
            if uniform {
                let ext = self.extents();
                return Some(std::array::from_fn(|i| if ext[i] > FLAT { f } else { 1.0 }));
            }
            let mut factors = [1.0; 3];
            factors[grip.mask[0]] = f;
            return Some(factors);
        }
        let gl = self.to_local(g0).sub(self.to_local(anchor));
        let pl = self.to_local(p).sub(self.to_local(anchor));
        let mut factors = [1.0; 3];
        for &i in &grip.mask {
            let ga = comp(gl, i);
            if ga.abs() < 1e-12 {
                return None;
            }
            factors[i] = comp(pl, i) / ga;
        }
        Some(factors)
    }

    fn factors_for(&self, env: &Env, cx: &ToolCx, sx: f64, sy: f64) -> Option<[f64; 3]> {
        if let Some(p) = Self::snapped_point(cx) {
            if let Some(got) = self.factors_from_point(env, p) {
                return Some(got);
            }
        }
        self.factors_from_cursor(env, sx, sy)
    }

    /// `_factors_from_cursor`.
    fn factors_from_cursor(&self, env: &Env, sx: f64, sy: f64) -> Option<[f64; 3]> {
        let h = env.cx.hypot;
        let grip = &self.grips[self.grip.expect("grip")];
        let anchor = self.anchor.expect("anchor");
        let g0 = self.grip_pos(grip);
        let (o, d) = env.vp.pixel_to_ray(sx, sy)?;
        let uniform = self.uniform || grip.kind(self.active_axes()) == "corner";
        if uniform || grip.mask.len() == 1 {
            let axis_dir = g0.sub(anchor);
            let length = axis_dir.length(h);
            if (length as f64) < 1e-12 {
                return None;
            }
            let u = axis_dir.div(length);
            let p = Self::closest_on_line(o, d, anchor, u)?;
            let f = p.sub(anchor).dot(u) as f64 / length as f64;
            if uniform {
                let ext = self.extents();
                return Some(std::array::from_fn(|i| if ext[i] > FLAT { f } else { 1.0 }));
            }
            let mut factors = [1.0; 3];
            factors[grip.mask[0]] = f;
            return Some(factors);
        }
        // Two axes (edge midpoint): track on the grip's own box plane.
        let normal_axis = (0..3).find(|i| !grip.mask.contains(i))?;
        let n = match &self.axes {
            None => {
                let mut c = [0.0f32; 3];
                c[normal_axis] = 1.0;
                V3(c[0], c[1], c[2])
            }
            Some(ax) => ax[normal_axis],
        };
        let denom = n.dot(d) as f64;
        if denom.abs() < 1e-9 {
            return None;
        }
        let t = n.dot(g0.sub(o)) as f64 / denom;
        if t < 0.0 {
            return None;
        }
        let p = o.add(d.mul(t as f32));
        let mut factors = [1.0; 3];
        if self.axes.is_some() {
            let gl = self.to_local(g0).sub(self.to_local(anchor));
            let pl = self.to_local(p).sub(self.to_local(anchor));
            for &i in &grip.mask {
                let ga = comp(gl, i);
                if ga.abs() < 1e-12 {
                    return None;
                }
                factors[i] = comp(pl, i) / ga;
            }
            return Some(factors);
        }
        for &i in &grip.mask {
            let ga = comp(g0, i) - comp(anchor, i);
            if ga.abs() < 1e-12 {
                return None;
            }
            let pa = comp(p, i) - comp(anchor, i);
            factors[i] = pa / ga;
        }
        Some(factors)
    }

    // ---- grab / toggles --------------------------------------------------------

    fn grab(&mut self, env: &Env, grip: usize, screen: (f64, f64)) {
        let h = env.cx.hypot;
        let mesh = env.cx.mesh(env.scene);
        self.groups = env.sel.groups();
        let mut positions: Vec<V3> = Vec::new();
        for s in env.sel.iter() {
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
        let mut seen: Vec<V3> = Vec::new();
        for p in positions {
            if !seen.iter().any(|&q| (p.sub(q).length(h) as f64) < 1e-9) {
                seen.push(p);
            }
        }
        self.verts = seen.iter().filter_map(|&p| mesh.vertex_at(p)).collect();
        self.positions = seen;
        self.grip = Some(grip);
        let g = self.grips[grip].clone();
        self.anchor = Some(self.anchor_for(&g));
        self.factors = [1.0; 3];
        self.capture_originals(env);
        self.grabbed_px = Some(screen);
        self.moved = false;
    }

    /// `_capture_originals`.
    fn capture_originals(&mut self, env: &Env) {
        let mut verts = Vec::new();
        let mut xforms = Vec::new();
        for path in &self.groups {
            let Some(g) = group_at(&env.scene.groups, path) else { continue };
            if let Some(x) = g.xform {
                xforms.push((path.clone(), x));
            } else {
                let gm = env.scene.mesh(g.mesh);
                verts.extend(gm.vertices.iter().map(|&v| (g.mesh, v, gm.position(v))));
            }
        }
        let mesh = env.cx.mesh(env.scene);
        verts.extend(self.verts.iter().map(|&v| (env.cx.mesh, v, mesh.position(v))));
        self.originals = Some(Originals { verts, xforms });
    }

    fn toggle_center(&mut self, env: &mut Env, say: bool) {
        self.about_center = !self.about_center;
        self.reanchor(env);
        if say {
            env.flash(if self.about_center { "About Center: on" } else { "About Center: off" });
        }
    }

    /// `_reanchor`: Ctrl mid-operation moves the anchor without losing the
    /// factors already dragged.
    fn reanchor(&mut self, env: &mut Env) {
        let Some(grip) = self.grip else { return };
        let current = self.factors;
        self.apply_preview(env, [1.0; 3]);
        let g = self.grips[grip].clone();
        self.anchor = Some(self.anchor_for(&g));
        self.apply_preview(env, current);
    }

    // ---- live preview ------------------------------------------------------------

    /// `_set_preview`: everything at `factors` about the anchor, from the
    /// originals; at (1, 1, 1) an exact restore.
    fn set_preview(&mut self, env: &mut Env, factors: [f64; 3]) {
        let Some(orig) = &self.originals else { return };
        let identity = factors.iter().all(|f| (f - 1.0).abs() < 1e-12);
        let m = if identity { None } else { Some(scale_matrix(self.anchor.expect("anchor"), Factor::Axes(factors), self.axes)) };
        for &(mid, v, p0) in &orig.verts {
            let target = match &m {
                None => p0,
                Some(m) => m.map(p0),
            };
            let mesh = &mut env.scene.meshes[mid.0 as usize];
            let cur = mesh.position(v);
            mesh.move_vertex(v, target.sub(cur));
        }
        for (path, x0) in &orig.xforms {
            if let Some(g) = group_at_mut(&mut env.scene.groups, path) {
                g.xform = Some(match &m {
                    None => *x0,
                    Some(m) => m.mul(x0),
                });
            }
        }
        *env.version += 1;
    }

    fn apply_preview(&mut self, env: &mut Env, target: [f64; 3]) {
        if target.iter().zip(self.factors.iter()).all(|(t, f)| (t - f).abs() < 1e-12) {
            return;
        }
        self.set_preview(env, target);
        self.factors = target;
    }

    fn revert_preview(&mut self, env: &mut Env) {
        if self.factors.iter().any(|f| (f - 1.0).abs() > 1e-12) {
            self.set_preview(env, [1.0; 3]);
            self.factors = [1.0; 3];
        }
    }

    // ---- commit ------------------------------------------------------------------

    /// `build(fs)` of `_commit`.
    fn build(groups: &[Path], positions: &[V3], anchor: V3, axes: Option<[V3; 3]>, fs: [f64; 3]) -> Option<Box<dyn Command>> {
        let packed = if fs[0] == fs[1] && fs[1] == fs[2] { Factor::Uniform(fs[0]) } else { Factor::Axes(fs) };
        let mut cmds: Vec<Box<dyn Command>> = Vec::new();
        for g in groups {
            cmds.push(Box::new(ScaleGroup::new(g, anchor, packed, axes)));
        }
        if !positions.is_empty() {
            cmds.push(Box::new(ScaleVertices::new(positions, anchor, packed, axes)));
        }
        if cmds.is_empty() {
            return None;
        }
        Some(if cmds.len() == 1 { cmds.pop().expect("one") } else { Box::new(CompoundCommand { commands: cmds }) })
    }

    fn commit(&mut self, env: &mut Env, factors: [f64; 3]) {
        self.revert_preview(env);
        let grip = self.grip.map(|i| self.grips[i].clone());
        let uniform = self.uniform || grip.as_ref().is_some_and(|g| g.kind(self.active_axes()) == "corner");
        let mask = grip.as_ref().map(|g| g.mask.clone()).unwrap_or_else(|| vec![0, 1, 2]);
        let ext = self.extents_or_unit();
        let active: Vec<usize> = (0..3).filter(|&i| ext[i] > FLAT).collect();
        let spec = Spec { uniform, mask, ext, active };
        let anchor = self.anchor.expect("anchor");
        let box_axes = self.axes;
        let groups = self.groups.clone();
        let positions = self.positions.clone();
        let changed = factors.iter().any(|f| (f - 1.0).abs() > 1e-9);
        if changed && factors.iter().all(|f| f.abs() >= MIN_FACTOR) {
            if let Some(cmd) = Self::build(&groups, &positions, anchor, box_axes, factors) {
                let serial = env.execute_serial(cmd);
                self.last = Some(Last { serial, spec, anchor, axes: box_axes, groups, positions });
            }
        }
        self.grip = None;
        self.anchor = None;
        self.factors = [1.0; 3];
        self.originals = None;
        self.box_version = -1;
        self.refresh_box(env);
    }

    fn reset(&mut self) {
        self.lo = None;
        self.hi = None;
        self.grips.clear();
        self.grip = None;
        self.hover_grip = None;
        self.anchor = None;
        self.factors = [1.0; 3];
        self.groups.clear();
        self.positions.clear();
        self.verts.clear();
        self.originals = None;
        self.box_version = -1;
        self.grabbed_px = None;
        self.moved = false;
    }

    /// `_typed_factors`.
    fn typed_factors(&self, value: &Vcb, absolute: bool, spec: Option<&Spec>) -> Option<[f64; 3]> {
        let (uniform, mask, ext, active) = match spec {
            None => {
                let grip = &self.grips[self.grip.expect("grip")];
                let uniform = self.uniform || grip.kind(self.active_axes()) == "corner";
                let ext = self.extents();
                (uniform, grip.mask.clone(), ext, (0..3).filter(|&i| ext[i] > FLAT).collect::<Vec<_>>())
            }
            Some(s) => (s.uniform, s.mask.clone(), s.ext, s.active.clone()),
        };
        let values: Vec<f64> = match value {
            Vcb::Num(n) => vec![*n],
            Vcb::Tuple(t) => t.clone(),
            Vcb::AbsLen(_) => return None,
        };
        if values.iter().any(|v| v.abs() < MIN_FACTOR) {
            return None;
        }
        let axes = if uniform { active } else { mask };
        let mut factors = [1.0; 3];
        if values.len() == 1 {
            let mut f = values[0];
            if absolute {
                let base = ext[*axes.first()?];
                if base <= FLAT {
                    return None;
                }
                f /= base;
            }
            for &i in &axes {
                factors[i] = f;
            }
            return Some(factors);
        }
        if values.len() != axes.len() {
            return None;
        }
        for (&i, &v) in axes.iter().zip(values.iter()) {
            let mut v = v;
            if absolute {
                if ext[i] <= FLAT {
                    return None;
                }
                v /= ext[i];
            }
            factors[i] = v;
        }
        Some(factors)
    }

    // ---- feedback ------------------------------------------------------------------

    fn vcb_caption(&self) -> String {
        let Some(gi) = self.grip.or(self.hover_grip) else { return "Scale".to_string() };
        let grip = &self.grips[gi];
        if self.uniform || grip.kind(self.active_axes()) == "corner" {
            return "Scale".to_string();
        }
        let names: Vec<&str> = grip.mask.iter().map(|&i| AXIS_NAMES[i]).collect();
        format!("{} Scale", names.join(", "))
    }

    fn value_label(&self) -> Option<(String, V3)> {
        let grip = &self.grips[self.grip?];
        let text = if self.uniform || grip.kind(self.active_axes()) == "corner" {
            format!("{:.2}", self.factors[grip.mask[0]])
        } else {
            grip.mask.iter().map(|&i| format!("{:.2}", self.factors[i])).collect::<Vec<_>>().join("; ")
        };
        let pos = if self.lo.is_some() { self.grip_pos(grip) } else { self.anchor.expect("anchor") };
        Some((text, pos))
    }

    /// `scale_box_state`: the (live-scaled) box segments and each grip with its
    /// role.
    #[allow(clippy::type_complexity)]
    pub fn scale_box_state(&self) -> Option<(Vec<(V3, V3)>, Vec<(V3, &'static str)>)> {
        let (lo, hi) = (f64s(self.lo?), f64s(self.hi?));
        let m = if self.grip.is_some() { Some(scale_matrix(self.anchor.expect("anchor"), Factor::Axes(self.factors), self.axes)) } else { None };
        let corner = |tx: f64, ty: f64, tz: f64| {
            let t = [tx, ty, tz];
            let c = |i: usize| (lo[i] + (hi[i] - lo[i]) * t[i]) as f32;
            let p = self.to_world(V3(c(0), c(1), c(2)));
            match &m {
                Some(m) => m.map(p),
                None => p,
            }
        };
        let ext = self.extents();
        let axes: Vec<usize> = (0..3).filter(|&i| ext[i] > FLAT).collect();
        let mut segments = Vec::new();
        if axes.len() >= 2 {
            for &axis in &axes {
                let others: Vec<usize> = (0..3).filter(|&i| i != axis).collect();
                let tas: &[f64] = if ext[others[0]] <= FLAT { &[0.0] } else { &[0.0, 1.0] };
                let tbs: &[f64] = if ext[others[1]] <= FLAT { &[0.0] } else { &[0.0, 1.0] };
                for &ta in tas {
                    for &tb in tbs {
                        let mut t0 = [0.0; 3];
                        t0[others[0]] = ta;
                        t0[others[1]] = tb;
                        let mut t1 = t0;
                        t1[axis] = 1.0;
                        segments.push((corner(t0[0], t0[1], t0[2]), corner(t1[0], t1[1], t1[2])));
                    }
                }
            }
        } else if axes.len() == 1 {
            let mut t0 = [0.5; 3];
            let mut t1 = t0;
            t0[axes[0]] = 0.0;
            t1[axes[0]] = 1.0;
            segments.push((corner(t0[0], t0[1], t0[2]), corner(t1[0], t1[1], t1[2])));
        }
        let mut grips = Vec::new();
        for (i, g) in self.grips.iter().enumerate() {
            let mut p = self.grip_pos(g);
            if let Some(m) = &m {
                p = m.map(p);
            }
            let role = if Some(i) == self.grip {
                "active"
            } else if Some(i) == self.hover_grip && self.grip.is_none() {
                "hover"
            } else {
                "idle"
            };
            grips.push((p, role));
        }
        if self.grip.is_some() {
            if let Some(a) = self.anchor {
                grips.push((a, "anchor"));
            }
        }
        Some((segments, grips))
    }
}

/// FNV-1a over 32-bit words (little endian), the checksum `modify_scale.py`
/// records, truncated to 60 bits.
fn fnv(words: &[u32]) -> i64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for w in words {
        for b in w.to_le_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x0100_0000_01b3);
        }
    }
    (h >> 4) as i64
}

fn bits(x: f32) -> String {
    format!("{:08x}", x.to_bits())
}

impl ModifyTool for ScaleTool {
    fn on_activate(&mut self, env: &mut Env) {
        self.reset();
        self.about_center = false;
        self.uniform = false;
        self.last = None;
        self.refresh_box(env);
    }

    fn on_deactivate(&mut self, env: &mut Env) {
        self.revert_preview(env);
        self.reset();
        self.last = None;
    }

    fn on_click(&mut self, env: &mut Env, cx: &ToolCx) {
        self.last = None; // a click closes the retype window
        self.refresh_box(env);
        let (sx, sy) = cx.screen;
        if self.grip.is_some() {
            // second click commits
            if let Some(factors) = self.factors_for(env, cx, sx, sy) {
                self.commit(env, factors);
            }
            return;
        }
        if self.lo.is_some() {
            if let Some(grip) = self.grip_under(env, sx, sy) {
                self.grab(env, grip, (sx, sy));
                return;
            }
        }
        // No grip hit: with nothing selected, a click picks the entity under
        // the cursor and boxes it (click-to-scale).
        if env.sel.is_empty() {
            let (groups, positions) = gather_targets(env, cx);
            let image = env.vp.pick_image_plane(env.scene, env.cx, sx, sy);
            let mut picked: Vec<Sel> = groups.into_iter().map(Sel::Group).collect();
            picked.extend(image.map(Sel::Image));
            if picked.is_empty() && !positions.is_empty() {
                let edge = env.vp.pick_edge(env.scene, env.cx, sx, sy);
                let face = if edge.is_none() { env.vp.pick_face(env.scene, env.cx, sx, sy) } else { None };
                picked.extend(edge.map(Sel::Edge));
                picked.extend(face.map(Sel::Face));
            }
            if !picked.is_empty() {
                env.select(&picked, "replace");
                self.refresh_box(env);
                return;
            }
            env.flash("Select (or click) the geometry to scale first");
        }
    }

    fn on_hover(&mut self, env: &mut Env, cx: &ToolCx) {
        let (sx, sy) = cx.screen;
        if self.grip.is_none() {
            self.refresh_box(env);
            let hover = if self.lo.is_some() { self.grip_under(env, sx, sy) } else { None };
            if hover != self.hover_grip {
                self.hover_grip = hover;
            }
            return;
        }
        let Some(factors) = self.factors_for(env, cx, sx, sy) else { return };
        if factors.iter().any(|f| f.abs() < MIN_FACTOR) {
            return;
        }
        if let Some(g) = self.grabbed_px {
            let (dx, dy) = (sx - g.0, sy - g.1);
            if (dx * dx + dy * dy).sqrt() > 4.0 {
                self.moved = true;
            }
        }
        self.apply_preview(env, factors);
    }

    fn on_release(&mut self, env: &mut Env) {
        if self.grip.is_none() || !self.moved {
            return;
        }
        if self.factors.iter().any(|f| (f - 1.0).abs() > 1e-9) {
            let f = self.factors;
            self.commit(env, f);
        }
    }

    fn on_key_release(&mut self, env: &mut Env, key: i32) -> bool {
        if key != keys::CONTROL || !self.ctrl_toggled {
            return false;
        }
        self.ctrl_toggled = false;
        if !env.vp.ctrl_tapped() {
            self.toggle_center(env, false);
        }
        true
    }

    fn on_key(&mut self, env: &mut Env, key: i32, _mods: Mods) -> bool {
        if key == keys::CONTROL {
            self.toggle_center(env, true);
            self.ctrl_toggled = true;
            return true;
        }
        if key == keys::SHIFT {
            self.uniform = !self.uniform;
            env.flash(if self.uniform { "Scale Uniformly: on" } else { "Scale Uniformly: off" });
            return true;
        }
        false
    }

    fn on_value(&mut self, env: &mut Env, value: &Vcb) -> bool {
        let (absolute, value) = match value {
            Vcb::AbsLen(inner) => (true, inner.as_ref()),
            v => (false, v),
        };
        if self.grip.is_some() {
            let Some(factors) = self.typed_factors(value, absolute, None) else { return false };
            self.commit(env, factors);
            return true;
        }
        if self.last.is_some() {
            // Hot retype: redo the scale just made at the new value.
            let top = self.last.as_ref().is_some_and(|l| env.is_top(l.serial));
            if !top {
                self.last = None;
                return false;
            }
            let spec = self.last.as_ref().expect("last").spec.clone();
            let Some(factors) = self.typed_factors(value, absolute, Some(&spec)) else { return false };
            env.undo();
            let last = self.last.take().expect("last");
            let cmd = Self::build(&last.groups, &last.positions, last.anchor, last.axes, factors);
            let serial = match cmd {
                Some(c) => env.execute_serial(c),
                None => None,
            };
            self.last = Some(Last { serial, ..last });
            return true;
        }
        false
    }

    fn on_cancel(&mut self, env: &mut Env) {
        self.revert_preview(env);
        self.grip = None;
        self.anchor = None;
        self.last = None;
    }

    fn probe(&self, env: &Env) -> Probe {
        let mut p = Probe::default();
        p.vl = Some(self.value_label());
        p.cap = Some(self.vcb_caption());
        p.unitless = Some(true);
        p.uses_snap = self.grip.is_some();
        p.excl = Some(if self.grip.is_none() {
            None
        } else {
            let mesh = env.cx.mesh(env.scene);
            let mut edges: Vec<usize> = Vec::new();
            for &v in &self.verts {
                for e in mesh.edges_at(v) {
                    if let Some(i) = mesh.edges.iter().position(|x| *x == e) {
                        if !edges.contains(&i) {
                            edges.push(i);
                        }
                    }
                }
            }
            Some((edges, self.groups.clone()))
        });
        let i = |n: usize| Value::int(n as i64);
        let v3 = |q: V3| Value::Array(vec![Value::Str(bits(q.0)), Value::Str(bits(q.1)), Value::Str(bits(q.2))]);
        let opt3 = |q: Option<V3>| q.map(v3).unwrap_or(Value::Null);
        let state = match self.scale_box_state() {
            None => Value::Null,
            Some((segs, grips)) => {
                let mut words: Vec<u32> = Vec::new();
                for (a, b) in &segs {
                    for q in [a, b] {
                        words.extend([q.0.to_bits(), q.1.to_bits(), q.2.to_bits()]);
                    }
                }
                let mut gwords: Vec<u32> = Vec::new();
                for (q, _) in &grips {
                    gwords.extend([q.0.to_bits(), q.1.to_bits(), q.2.to_bits()]);
                }
                Value::Object(vec![
                    ("n_seg".into(), i(segs.len())),
                    ("seg_sum".into(), Value::int(fnv(&words))),
                    ("grip_sum".into(), Value::int(fnv(&gwords))),
                    ("roles".into(), Value::Array(grips.iter().map(|(_, r)| Value::Str((*r).to_string())).collect())),
                ])
            }
        };
        let tuple = |t: [f64; 3]| Value::Array(t.iter().map(|&x| Value::Float(x)).collect());
        p.x = Some(Value::Object(vec![
            ("lo".into(), opt3(self.lo)),
            ("hi".into(), opt3(self.hi)),
            ("axes".into(), self.axes.map(|a| Value::Array(a.iter().map(|&q| v3(q)).collect())).unwrap_or(Value::Null)),
            ("orig".into(), opt3(self.orig)),
            ("n_grips".into(), i(self.grips.len())),
            ("box_fresh".into(), Value::Bool(self.box_version == *env.version as i64)),
            ("n_groups".into(), i(self.groups.len())),
            ("n_pos".into(), i(self.positions.len())),
            ("n_verts".into(), i(self.verts.len())),
            ("grip".into(), self.grip.map(i).unwrap_or(Value::Null)),
            ("hover_grip".into(), self.hover_grip.map(i).unwrap_or(Value::Null)),
            ("anchor".into(), opt3(self.anchor)),
            ("factors".into(), tuple(self.factors)),
            ("grabbed_px".into(), self.grabbed_px.map(|(x, y)| Value::Array(vec![Value::Float(x), Value::Float(y)])).unwrap_or(Value::Null)),
            ("moved".into(), Value::Bool(self.moved)),
            ("about_center".into(), Value::Bool(self.about_center)),
            ("uniform".into(), Value::Bool(self.uniform)),
            ("last".into(), Value::Bool(self.last.is_some())),
            ("orig_loaded".into(), Value::Bool(self.originals.is_some())),
            ("ctrl_toggled".into(), Value::Bool(self.ctrl_toggled)),
            ("state".into(), state),
        ]));
        p
    }
}
