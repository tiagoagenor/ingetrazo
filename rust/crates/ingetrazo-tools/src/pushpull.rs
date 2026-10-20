// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! The Push/Pull tool (`tools/pushpull.py`): the state machine, with no UI.
//!
//! In: pointer pixels (the camera ray is derived from [`View`]), Ctrl, typed
//! text (a number: the VCB). Out: the preview (`preview_faces`,
//! `rubber_band_lines`, the inference marker and guide, the distance label)
//! and the [`ingetrazo_ops::pushpull::PushPullCommand`] it executes through
//! the host's [`History`].
//!
//! UX (the usual push/pull convention): hover shades the face under the
//! cursor; the first click locks on it and starts a drag along its normal,
//! measured from the clicked spot; a second click commits at the current
//! distance; a number typed (VCB) commits at exactly that distance,
//! preserving the direction's sign; Ctrl pushes a copy (the start face stays
//! as a slab division); a double-click repeats the last distance on the face
//! under the cursor; a press-drag-release stroke commits on release; Esc
//! cancels; typing on the idle tool right after a push adjusts it (the hot
//! retype).
//!
//! Where it differs from Python (reported):
//!
//! * The *staged* preview of an attached face (the real pipeline applied
//!   once to the model, then only its cap slid) is not carried: such a face
//!   previews as the overlay sweep, like a free one, and the commit runs the
//!   pipeline. The prism preview (the cap translated in the model, undone
//!   by arithmetic) is, since its float32 round trip is part of the state a
//!   commit starts from.
//! * Inference skips the references the viewport hides (`_is_occluded`, the
//!   active section cut) unless the host passes `hidden_reference`.
//! * Picking is the host's: a [`Picker`].

use std::cell::RefCell;
use std::collections::HashSet;
use std::rc::Rc;

use ingetrazo_formats::pyjson::Value;
use ingetrazo_math::{Hypot, V3};
use ingetrazo_model::mesh::{key as mesh_key, EdgeId, FaceId, Mesh, State, VertexId, STITCH_TOL};
use ingetrazo_model::scene::{MeshId, Scene};
use ingetrazo_model::topology::{key as topo_key, PFace};
use ingetrazo_model::units::{fmt_len, ModelUnits};
use ingetrazo_ops::command::{Cx, History};
use ingetrazo_ops::orient::orient_outward_with;
use ingetrazo_ops::pushpull::{cap_loop_positions, classify_base, compute_inward_limit, compute_plane_levels, outward_normal, paint_of, slide_rate, PushPull, PushPullCommand, SnapshotStates, CURVE_FACET_COS, MIN_EXTRUDE};

use crate::pick::Picker;
use crate::view::View;

/// Below this many pixels between press and release, the "drag" was a
/// jittery click, not a stroke.
pub const DRAG_PX: f64 = 6.0;

/// What a tool step needs of the host.
pub struct Env<'a> {
    pub scene: &'a mut Scene,
    /// The mesh being edited (`cx.mesh`: the loose one, or an open group's)
    /// and the counters.
    pub cx: &'a mut Cx,
    pub history: &'a mut History,
    /// The camera; `None` for a headless host that only types distances.
    pub view: Option<&'a View>,
    pub picker: &'a dyn Picker,
    /// `viewport._hover_edge`: the edge under the cursor.
    pub hover_edge: Option<HoverEdge>,
    /// `viewport.snap_threshold_px`.
    pub snap_threshold_px: f64,
    /// The references the eye cannot reach (`_reference_hidden`).
    pub hidden_reference: Option<&'a dyn Fn(V3) -> bool>,
    pub units: &'a ModelUnits,
}

/// The edge under the cursor: its ends and, when it is a mesh edge, which
/// (a staged preview's own edges are not references).
#[derive(Clone, Copy, Debug)]
pub struct HoverEdge {
    pub a: V3,
    pub b: V3,
    pub edge: Option<EdgeId>,
}

/// A staged result left behind when the cursor leaves its interval
/// (`_park_stage`): one per side of the base plane.
#[derive(Clone)]
struct StageRec {
    d: f64,
    lo: f64,
    hi: f64,
    verts: Vec<VertexId>,
    pos: Vec<(V3, V3)>,
    new_faces: HashSet<FaceId>,
    new_edges: HashSet<EdgeId>,
    applied: f64,
    state: State,
}

/// A hover or click: the cursor in screen pixels and Ctrl.
#[derive(Clone, Copy, Debug, Default)]
pub struct Pointer {
    pub x: f64,
    pub y: f64,
    pub ctrl: bool,
}

/// A polygon of the overlay sweep (`Face(...)` of `_build_light_faces`).
#[derive(Clone, Debug, PartialEq)]
pub struct LightFace {
    pub vertices: Vec<V3>,
    pub holes: Vec<Vec<V3>>,
    pub attrs: Option<Vec<(String, Value)>>,
}

/// `_last`: everything a hot retype needs.
#[derive(Clone, Debug)]
struct Retype {
    serial: u64,
    face: FaceId,
    keep_base: bool,
    prism_cap: bool,
    normal: Option<V3>,
    cap_positions: Vec<V3>,
    pre_oriented: bool,
    limit_in: Option<f64>,
    sign: f64,
}

pub struct PushPullTool {
    pub hovered_face: Option<FaceId>,
    pub base_face: Option<FaceId>,
    /// Signed distance along the normal.
    pub extrusion: f64,
    pub dragging: bool,
    keep_base: bool,
    keep_mode: bool,
    ctrl_used: bool,
    limit_in: Option<f64>,
    /// `_hover_group is not None`: the hovered face lives in a closed group.
    pub hover_closed_group: bool,
    selected_arm: bool,
    attached: bool,
    prism_cap: bool,
    drag_pre_oriented: bool,
    anchor: Option<V3>,
    grab: Option<V3>,
    normal: Option<V3>,
    cap_positions: Vec<V3>,
    prism_applied: f64,
    prism_verts: Vec<VertexId>,
    inference_point: Option<V3>,
    inference_kind: Option<&'static str>,
    refused: bool,
    topped_out: bool,
    tiny_hint_shown: bool,
    press_screen: Option<(f64, f64)>,
    last_screen: Option<(f64, f64)>,
    light_faces: Vec<LightFace>,
    light_rings: Vec<(Vec<V3>, Vec<V3>)>,
    last: Option<Retype>,
    infer_cache: Option<Vec<(MeshId, Vec<VertexId>)>>,
    // ---- the staged preview (see `show_staged_preview`)
    /// The clean model under a staged preview, captured before the drag's
    /// first stage and kept until the drag ends.
    preview_snapshot: Option<State>,
    /// The distance the staged result was built at (`None` = the model is
    /// NOT staged), where its cap sits, the new cap-plane vertices a frame
    /// slides (with their staged positions and slide rates) and the open
    /// interval of distances the slide stays exact in.
    stage_d: Option<f64>,
    stage_applied: f64,
    stage_verts: Vec<VertexId>,
    stage_pos: Vec<(V3, V3)>,
    stage_lo: f64,
    stage_hi: f64,
    /// The faces/edges the stage created: the inference skips them.
    stage_new_faces: HashSet<FaceId>,
    stage_new_edges: HashSet<EdgeId>,
    /// Stages left behind, one per side of the base plane (`[+1, -1]`), and
    /// whether the current stage came from a parked record.
    stage_cache: [Option<StageRec>; 2],
    stage_parked: bool,
    /// `(sign, abs(d))` of the last refused or failed stage.
    stage_refusal: Option<(f64, f64)>,
    /// Levels (along the normal) of the parallel faces over/under the base.
    plane_levels: Vec<f64>,
    /// `PushPullTool.last_distance`: the last committed distance, shared
    /// across activations (a double-click repeats it).
    pub last_distance: Option<f64>,
    // ---- what the host reads back (set_hover, set_suppressed_faces, flash_status)
    /// The face to shade (`viewport.set_hover`).
    pub hover: Option<FaceId>,
    /// The faces to hide while an overlay stands in for them.
    pub suppressed: Vec<FaceId>,
    /// Every status message flashed, in order.
    pub flashes: Vec<String>,
}

impl Default for PushPullTool {
    fn default() -> Self {
        PushPullTool::new()
    }
}

impl PushPullTool {
    pub fn new() -> PushPullTool {
        PushPullTool {
            hovered_face: None,
            base_face: None,
            extrusion: 0.0,
            dragging: false,
            keep_base: false,
            keep_mode: false,
            ctrl_used: false,
            limit_in: None,
            hover_closed_group: false,
            selected_arm: false,
            attached: false,
            prism_cap: false,
            drag_pre_oriented: false,
            anchor: None,
            grab: None,
            normal: None,
            cap_positions: Vec::new(),
            prism_applied: 0.0,
            prism_verts: Vec::new(),
            inference_point: None,
            inference_kind: None,
            refused: false,
            topped_out: false,
            tiny_hint_shown: false,
            press_screen: None,
            last_screen: None,
            light_faces: Vec::new(),
            light_rings: Vec::new(),
            last: None,
            infer_cache: None,
            preview_snapshot: None,
            stage_d: None,
            stage_applied: 0.0,
            stage_verts: Vec::new(),
            stage_pos: Vec::new(),
            stage_lo: 0.0,
            stage_hi: 0.0,
            stage_new_faces: HashSet::new(),
            stage_new_edges: HashSet::new(),
            stage_cache: [None, None],
            stage_parked: false,
            stage_refusal: None,
            plane_levels: Vec::new(),
            last_distance: None,
            hover: None,
            suppressed: Vec::new(),
            flashes: Vec::new(),
        }
    }

    pub fn keep_base(&self) -> bool {
        self.keep_base
    }

    pub fn keep_mode(&self) -> bool {
        self.keep_mode
    }

    pub fn limit_in(&self) -> Option<f64> {
        self.limit_in
    }

    pub fn anchor(&self) -> Option<V3> {
        self.anchor
    }

    pub fn normal(&self) -> Option<V3> {
        self.normal
    }

    pub fn is_attached(&self) -> bool {
        self.attached
    }

    pub fn is_prism_cap(&self) -> bool {
        self.prism_cap
    }

    pub fn was_refused(&self) -> bool {
        self.refused
    }

    pub fn last_flash(&self) -> Option<&str> {
        self.flashes.last().map(|s| s.as_str())
    }

    fn flash(&mut self, text: String) {
        self.flashes.push(text);
    }

    // ---- lifecycle -----------------------------------------------------------

    /// `on_activate`: with exactly one loose-mesh Face selected (`selected`),
    /// the tool arms on it - the SketchUp preselection workflow.
    pub fn on_activate(&mut self, env: &mut Env, selected: Option<FaceId>) {
        self.reset();
        self.keep_mode = false; // the modifier arms ONE operation
        self.ctrl_used = false;
        let Some(item) = selected else { return };
        if !env.cx.mesh(env.scene).faces.contains(&item) {
            return;
        }
        self.hovered_face = Some(item);
        self.hover_closed_group = false;
        self.selected_arm = true;
        self.hover = Some(item);
    }

    pub fn on_deactivate(&mut self, env: &mut Env) {
        self.revert_preview(env);
        self.hover = None;
        self.suppressed.clear();
        self.reset();
    }

    /// `drag_plane`: mid-drag, the plane through the anchor that contains the
    /// push axis and faces the camera, as `(point, normal)`.
    pub fn drag_plane(&self, env: &Env) -> Option<(V3, V3)> {
        if !self.dragging {
            return None;
        }
        let (anchor, n) = (self.anchor?, self.normal?);
        let view = env.view?;
        let h = view.hypot;
        let forward = view.camera.target.sub(view.camera.eye());
        let side = forward.sub(n.mul(forward.dot(n)));
        if (side.length(h) as f64) < 1e-6 {
            return None;
        }
        Some((anchor, side.normalized(h)))
    }

    // ---- spatial input ---------------------------------------------------------

    pub fn on_hover(&mut self, env: &mut Env, ptr: Pointer) {
        if !self.dragging {
            self.inference_point = None;
            self.inference_kind = None;
            // The armed preselection outranks the cursor.
            if self.selected_arm {
                if let Some(f) = self.hovered_face {
                    if env.cx.mesh(env.scene).faces.contains(&f) {
                        self.hover = Some(f);
                        return;
                    }
                    self.selected_arm = false;
                }
            }
            let pick = env.picker.pick_face_any(env.view, env.cx.mesh(env.scene), ptr.x, ptr.y);
            self.hovered_face = pick.map(|p| p.face);
            self.hover_closed_group = pick.is_some_and(|p| p.closed_group);
            // A face inside a group that is not open gets NO shading.
            self.hover = if self.hover_closed_group { None } else { self.hovered_face };
            return;
        }

        // Tracked before any guard: the release's "was this a stroke" test.
        self.last_screen = Some((ptr.x, ptr.y));
        if self.base_face.is_none() || self.anchor.is_none() {
            return;
        }
        // Ctrl can be pressed/released mid-drag; the live preview follows.
        self.sync_keep_base(ptr.ctrl, false);
        // Only the prism translation is undone per frame - arithmetic.
        self.revert_prism_preview(env);
        let inferred = self.infer_reference_distance(env, ptr);
        if let Some(d) = inferred {
            self.extrusion = d;
        } else if let Some(view) = env.view {
            let (anchor, normal) = (self.anchor.unwrap(), self.normal.unwrap());
            let grab = self.grab.unwrap_or(anchor);
            let projected = view.project_to_lock_line(grab, normal, ptr.x, ptr.y);
            self.extrusion = projected.sub(grab).dot(normal) as f64;
        }
        self.clamp_extrusion(env);
        // A clean prism extend/shrink is previewed by MOVING the cap, so the
        // model itself shows the result. Any other attached face is STAGED:
        // the real commit applied once, then stretched. A free face and a
        // Ctrl-copy show the sweep as an overlay.
        if self.prism_cap && !self.keep_base {
            self.show_prism_preview(env);
        } else if self.attached && !self.keep_base {
            self.show_staged_preview(env);
        } else {
            self.show_light_preview(env);
        }
    }

    pub fn on_click(&mut self, env: &mut Env, ptr: Pointer) {
        self.last = None; // a click ends the retype window
        // A CLICK latches Ctrl as the gesture's own.
        self.sync_keep_base(ptr.ctrl, true);
        if !self.dragging {
            let Some(face) = self.hovered_face else { return };
            if self.hover_closed_group {
                self.refuse_closed_group();
                return;
            }
            self.begin_drag(env, face, Some(ptr));
            self.press_screen = Some((ptr.x, ptr.y));
            self.last_screen = self.press_screen;
            self.hover = None; // the live preview takes over from the hover shade
            return;
        }
        // Already dragging - second click commits.
        if self.extrusion.abs() < MIN_EXTRUDE {
            self.tiny_hint();
            return;
        }
        self.commit(env);
    }

    /// The drag state a click on `face` establishes (`on_click` and the
    /// first press of `on_double_click`).
    fn begin_drag(&mut self, env: &mut Env, face: FaceId, ptr: Option<Pointer>) {
        let h = env.cx.hypot;
        self.base_face = Some(face);
        self.extrusion = 0.0;
        self.dragging = true;
        self.selected_arm = false; // a drag exists; the arm is spent
        // Establish the outward invariant ONCE per drag, before anchor and
        // normal are captured, scoped to the pushed solid.
        let m = env.cx.mesh_mut(env.scene);
        orient_outward_with(m, h, 12345, Some(&[face]));
        self.drag_pre_oriented = true;
        let anchor = ingetrazo_geom::meshface::centroid(&ingetrazo_model::topology::positions(m, &m.face(face).outer));
        self.anchor = Some(anchor);
        let normal = outward_normal(m, face, h);
        self.normal = Some(normal);
        self.grab = self.grab_point(env, ptr);
        let m = env.cx.mesh(env.scene);
        let (attached, prism_cap) = classify_base(m, face, h);
        self.attached = attached;
        self.prism_cap = prism_cap;
        self.cap_positions = cap_loop_positions(m, face);
        self.prism_verts = self.cap_vertices(m);
        self.limit_in = compute_inward_limit(m, face, self.normal, self.anchor, self.attached, h);
        self.plane_levels = compute_plane_levels(m, face, normal, anchor, self.attached, self.prism_cap, h);
        self.preview_snapshot = None;
    }

    /// `_grab_point`: the clicked spot on the face's plane, or `None` (the
    /// centroid then stands in).
    fn grab_point(&self, env: &Env, ptr: Option<Pointer>) -> Option<V3> {
        let (view, ptr) = (env.view?, ptr?);
        let (anchor, normal) = (self.anchor?, self.normal?);
        view.ray_hit_plane((ptr.x, ptr.y), (anchor, normal))
    }

    pub fn on_double_click(&mut self, env: &mut Env, ptr: Pointer) {
        let Some(last) = self.last_distance else {
            self.on_click(env, ptr);
            return;
        };
        self.sync_keep_base(ptr.ctrl, true);
        if !self.dragging {
            let pick = env.picker.pick_face_any(env.view, env.cx.mesh(env.scene), ptr.x, ptr.y);
            let Some(pick) = pick else { return };
            if pick.closed_group {
                self.refuse_closed_group();
                return;
            }
            self.begin_drag(env, pick.face, Some(ptr));
        } else if self.extrusion.abs() > MIN_EXTRUDE {
            // Mid-drag with a real distance: treat as the commit click.
            self.commit(env);
            return;
        }
        self.extrusion = last;
        self.clamp_extrusion(env);
        self.commit(env);
    }

    /// `on_release`: left button up mid-drag. A press-drag-release stroke
    /// commits the push at the distance the drag built; `live_ctrl` is Ctrl
    /// read live (`QApplication.queryKeyboardModifiers`).
    pub fn on_release(&mut self, env: &mut Env, live_ctrl: bool) {
        if !self.dragging {
            return;
        }
        let (Some(a), Some(b)) = (self.press_screen, self.last_screen) else { return };
        if ingetrazo_geom::sweep::py_hypot2(b.0 - a.0, b.1 - a.1) < DRAG_PX {
            return; // jitter, not a stroke: never a commit
        }
        self.sync_keep_base(live_ctrl, true);
        if self.extrusion.abs() < MIN_EXTRUDE {
            self.tiny_hint();
            return;
        }
        self.commit(env);
    }

    /// `on_value`: a number typed in the VCB. Mid-drag it commits at that
    /// distance (a negative value reverses the direction); idle it is the hot
    /// retype of the push just committed. `false` when nothing took it.
    pub fn on_value(&mut self, env: &mut Env, value: f64) -> bool {
        if self.dragging && self.base_face.is_some() {
            if value == 0.0 {
                return false;
            }
            // A positive value goes the way the user is dragging.
            let sign = if self.extrusion < 0.0 { -1.0 } else { 1.0 };
            self.extrusion = sign * value;
            self.clamp_extrusion(env);
            self.commit(env);
            return true;
        }
        // Idle: the hot retype. The typed distance is the TOTAL from the base.
        let Some(last) = self.last.clone() else { return false };
        if value == 0.0 {
            return false;
        }
        if env.history.top_serial() != Some(last.serial) {
            self.last = None;
            return false;
        }
        env.history.undo(env.scene, env.cx);
        // Replay the push as a drag at the typed distance.
        self.base_face = Some(last.face);
        self.keep_base = last.keep_base;
        self.prism_cap = last.prism_cap;
        self.normal = last.normal;
        self.cap_positions = last.cap_positions.clone();
        self.drag_pre_oriented = last.pre_oriented;
        self.limit_in = last.limit_in;
        self.dragging = true;
        self.extrusion = 0.0;
        self.prism_applied = 0.0;
        self.prism_verts.clear();
        self.light_faces.clear();
        self.light_rings.clear();
        self.infer_cache = None;
        self.preview_snapshot = None;
        self.refused = false;
        self.topped_out = false;
        self.extrusion = last.sign * value;
        self.clamp_extrusion(env);
        self.commit(env);
        true
    }

    pub fn on_cancel(&mut self, env: &mut Env) {
        self.light_faces.clear();
        self.light_rings.clear();
        self.revert_preview(env);
        self.hover = None;
        self.suppressed.clear();
        self.reset();
        self.spend_keep_mode();
    }

    /// Ctrl+Z mid-drag abandons the push, like Esc.
    pub fn on_undo(&mut self, env: &mut Env) -> bool {
        if !self.dragging {
            return false;
        }
        self.on_cancel(env);
        true
    }

    /// Redo mid-drag abandons the push first, then lets the redo run (false).
    pub fn on_redo(&mut self, env: &mut Env) -> bool {
        if self.dragging {
            self.on_cancel(env);
        }
        false
    }

    // ---- visual preview ---------------------------------------------------------

    /// `rubber_band_lines`: the sweep's edges (the moved ring and one riser
    /// per corner), leaving out the risers the commit will soften.
    pub fn rubber_band_lines(&self, h: Hypot) -> Vec<(V3, V3)> {
        let mut segs: Vec<(V3, V3)> = Vec::new();
        for (low, high) in &self.light_rings {
            let n = low.len();
            if n < 2 {
                continue;
            }
            // Wall i spans low[i] -> low[i+1]; its normal is that edge crossed
            // with the sweep. The riser at corner i divides walls i-1 and i.
            let off = high[0].sub(low[0]);
            let normals: Vec<Option<V3>> = (0..n)
                .map(|i| {
                    let nv = low[(i + 1) % n].sub(low[i]).cross(off);
                    if (nv.length(h) as f64) > 1e-12 {
                        Some(nv.normalized(h))
                    } else {
                        None
                    }
                })
                .collect();
            for i in 0..n {
                let j = (i + 1) % n;
                segs.push((high[i], high[j]));
                if let (Some(a), Some(b)) = (normals[(i + n - 1) % n], normals[i]) {
                    let d = a.dot(b) as f64;
                    if CURVE_FACET_COS < d && d < 0.99995 {
                        continue; // curve facet: soft once committed
                    }
                }
                segs.push((low[i], high[i]));
            }
        }
        segs
    }

    pub fn preview_faces(&self) -> &[LightFace] {
        &self.light_faces
    }

    /// `_build_light_faces`: the naive sweep of the base face as plain
    /// polygons - the moved cap (holes carried) and one quad per edge.
    fn build_light_faces(&mut self, env: &Env) -> Vec<LightFace> {
        self.light_rings.clear();
        let (Some(face), Some(n)) = (self.base_face, self.normal) else { return Vec::new() };
        let d = self.extrusion;
        if d.abs() < MIN_EXTRUDE {
            return Vec::new();
        }
        let m = env.cx.mesh(env.scene);
        let off = n.mul(d as f32);
        let attrs = if m.face(face).attrs.is_empty() { None } else { Some(m.face(face).attrs.clone()) };
        // The walls take the base's PAINT.
        let wall_paint = paint_of(&m.face(face).attrs);
        let wall_attrs = if wall_paint.is_empty() { None } else { Some(wall_paint) };
        let pf = PFace::of(m, face);
        let mut rings: Vec<(Vec<V3>, Vec<V3>)> = vec![(pf.vertices.clone(), pf.vertices.iter().map(|&p| p.add(off)).collect())];
        for hl in &pf.holes {
            rings.push((hl.clone(), hl.iter().map(|&p| p.add(off)).collect()));
        }
        self.light_rings = rings.clone();
        let cap = LightFace { vertices: rings[0].1.clone(), holes: rings[1..].iter().map(|r| r.1.clone()).collect(), attrs };
        let mut out = vec![cap];
        for (low, high) in &rings {
            let count = low.len();
            for i in 0..count {
                let j = (i + 1) % count;
                out.push(LightFace { vertices: vec![low[i], low[j], high[j], high[i]], holes: Vec::new(), attrs: wall_attrs.clone() });
            }
        }
        out
    }

    /// `_show_light_preview`: show the sweep as an overlay; the mesh is left
    /// untouched.
    fn show_light_preview(&mut self, env: &mut Env) {
        // A held stage is parked first (a no-op once parked): a Ctrl tap
        // mid-drag must find it again, not pay for a new one.
        self.park_stage(env);
        self.revert_preview(env);
        self.light_faces = self.build_light_faces(env);
        // Hiding the base face is what lets a drag AWAY FROM THE EYE read as
        // a recess opening.
        let recessing = !self.light_faces.is_empty() && self.base_face.is_some() && self.attached && !self.keep_base && self.sweeps_away_from_eye(env);
        self.suppressed = if recessing { vec![self.base_face.unwrap()] } else { Vec::new() };
    }

    /// `_sweeps_away_from_eye`: does the sweep move the cap away from the
    /// camera? Without a camera the normal's sign answers.
    fn sweeps_away_from_eye(&self, env: &Env) -> bool {
        let (Some(view), Some(anchor), Some(normal)) = (env.view, self.anchor, self.normal) else { return self.extrusion < 0.0 };
        let off = normal.mul(self.extrusion as f32);
        (off.dot(anchor.sub(view.camera.eye())) as f64) > 0.0
    }

    fn refuse_closed_group(&mut self) {
        self.flash("That face belongs to a group — double-click to open it first, then push inside".to_string());
    }

    fn tiny_hint(&mut self) {
        if !self.tiny_hint_shown {
            self.flash("Too small a push — drag further or type a distance".to_string());
            self.tiny_hint_shown = true;
        }
    }

    // ---- the prism preview: the cap translated in the model ---------------------

    /// `_show_prism_preview`: translate the cap in the MODEL, reversible by
    /// arithmetic (the frame moves the delta, the revert moves the total).
    fn show_prism_preview(&mut self, env: &mut Env) {
        self.light_faces.clear();
        self.light_rings.clear();
        self.suppressed.clear();
        let delta = self.extrusion - self.prism_applied;
        if delta.abs() > 1e-12 {
            self.move_cap(env, delta, self.extrusion);
        }
    }

    /// `_cap_vertices`: the Vertex objects the cap loop stands on.
    fn cap_vertices(&self, m: &Mesh) -> Vec<VertexId> {
        let keys: HashSet<_> = self.cap_positions.iter().map(|&p| mesh_key(p)).collect();
        m.vertices.iter().copied().filter(|&v| keys.contains(&mesh_key(m.position(v)))).collect()
    }

    /// `_move_cap`: slide the cap's own vertices by `delta` along the normal,
    /// by IDENTITY, never by position.
    fn move_cap(&mut self, env: &mut Env, delta: f64, applied: f64) {
        let Some(normal) = self.normal else { return };
        let step = normal.mul(delta as f32);
        let m = env.cx.mesh_mut(env.scene);
        for &v in &self.prism_verts {
            m.move_vertex(v, step);
        }
        // Booked as the TARGET, not as a running sum.
        self.prism_applied = applied;
    }

    fn revert_prism_preview(&mut self, env: &mut Env) {
        if self.prism_applied != 0.0 {
            self.move_cap(env, -self.prism_applied, 0.0);
        }
    }

    /// `_revert_preview`: back to the clean model - the prism slid back, a
    /// staged model restored (its stage dropped, not parked).
    fn revert_preview(&mut self, env: &mut Env) {
        self.revert_prism_preview(env);
        if self.stage_d.is_some() {
            self.restore_clean(env);
        }
    }

    // ---- the staged preview -----------------------------------------------------
    //
    // An attached push is previewed by showing the REAL result: the commit's
    // own mutation applied to the model once ("stage"), then only its new cap
    // vertices slid along the normal per frame ("stretch"). The slide is
    // exact while the cap crosses no "event level" (`resolve_stage` works
    // them out). Leaving that interval parks the stage and restores the one
    // parked on the cursor's side, or stages anew. A refused stage shows the
    // overlay instead. A stage slower than Python's 0.25 s guard is not
    // carried (it depends on the clock, never on the model).

    fn show_staged_preview(&mut self, env: &mut Env) {
        self.light_faces.clear();
        self.light_rings.clear();
        let d = self.extrusion;
        if d.abs() < MIN_EXTRUDE {
            // Nothing to show, as the overlay. Parked, not dropped: the cursor
            // is on its way to the other side and may come back.
            self.park_stage(env);
            self.revert_preview(env);
            self.suppressed.clear();
            return;
        }
        let sign = if d > 0.0 { 1.0 } else { -1.0 };
        if self.stage_refusal.is_some_and(|r| r.0 != sign) {
            self.stage_refusal = None;
        }
        if !self.stage_holds(d) && !self.enter_stage(env, sign) {
            self.show_light_preview(env);
            return;
        }
        self.slide_stage(env, d);
        // The base face is genuinely gone from the staged model.
        self.suppressed.clear();
    }

    /// The cursor is outside the current stage's interval (or nothing is
    /// staged): park the current stage, then restore the one parked on this
    /// side if it holds the distance, else stage anew. `false` = show the
    /// overlay (the model is left as it was; the overlay's revert cleans it).
    fn enter_stage(&mut self, env: &mut Env, sign: f64) -> bool {
        let d = self.extrusion;
        self.park_stage(env);
        let slot = if sign > 0.0 { 0 } else { 1 };
        if let Some(rec) = self.stage_cache[slot].clone() {
            if Self::interval_holds(d, Some(rec.d), rec.lo, rec.hi) {
                self.unpark_stage(env, rec);
                return true;
            }
        }
        if self.stage_refusal.is_some_and(|r| d.abs() < 2.0 * r.1) {
            return false;
        }
        if self.stage(env) {
            return true;
        }
        self.stage_refusal = Some((sign, d.abs()));
        false
    }

    /// Slide the stage's moving vertices to distance `d`: placed at the staged
    /// position + rate x offset, not moved by deltas (float32 deltas drift).
    fn slide_stage(&mut self, env: &mut Env, d: f64) {
        let Some(stage_d) = self.stage_d else { return };
        if self.stage_verts.is_empty() || (d - self.stage_applied).abs() <= 1e-12 {
            return;
        }
        let off = (d - stage_d) as f32;
        let m = env.cx.mesh_mut(env.scene);
        for (&v, &(p, rate)) in self.stage_verts.iter().zip(self.stage_pos.iter()) {
            m.place_vertex(v, p.add(rate.mul(off)));
        }
        self.stage_applied = d;
    }

    /// Whether a stage built at `stage_d` with interval `(lo, hi)` shows the
    /// commit at distance `d`: its own distance, or strictly inside - kept a
    /// weld tolerance clear of the levels, where the commit welds the cap.
    fn interval_holds(d: f64, stage_d: Option<f64>, lo: f64, hi: f64) -> bool {
        let Some(sd) = stage_d else { return false };
        if (d - sd).abs() <= 1e-9 {
            return true;
        }
        lo + STITCH_TOL < d && d < hi - STITCH_TOL
    }

    fn stage_holds(&self, d: f64) -> bool {
        Self::interval_holds(d, self.stage_d, self.stage_lo, self.stage_hi)
    }

    /// The push at the tool's current state, as the commit would run it.
    fn current_push(&self, face: FaceId) -> PushPull {
        let mut push = PushPull::new(face, self.extrusion);
        push.normal = self.normal;
        push.keep_base = self.keep_base;
        push.prism_cap = self.prism_cap;
        push.drag_pre_oriented = self.drag_pre_oriented;
        push.cap_positions = self.cap_positions.clone();
        push
    }

    /// Apply the commit's mutation at `extrusion` to the clean model
    /// (capturing it on the drag's first stage, restoring it when the model is
    /// staged) and resolve what a frame may slide. `false` - with the clean
    /// model back - when the push is refused or would not stretch cleanly.
    fn stage(&mut self, env: &mut Env) -> bool {
        let Some(face) = self.base_face else { return false };
        if self.preview_snapshot.is_none() {
            self.preview_snapshot = Some(env.cx.mesh(env.scene).capture_state());
        } else if self.stage_d.is_some() {
            self.restore_clean(env);
        }
        let mut push = self.current_push(face);
        push.mutate(env.scene, env.cx);
        let ok = !push.refused && self.resolve_stage(env);
        // The commit decides refusal for itself.
        if !ok {
            if self.preview_snapshot.is_some() {
                self.restore_clean(env);
            }
            return false;
        }
        self.stage_d = Some(self.extrusion);
        self.stage_applied = self.extrusion;
        true
    }

    /// After a stage: the moving vertices (new, on the cap plane), the open
    /// interval `(lo, hi)` the slide is exact in, and the new faces/edges.
    /// `false` when a face holding a moving vertex would bend.
    fn resolve_stage(&mut self, env: &Env) -> bool {
        let h = env.cx.hypot;
        let m = env.cx.mesh(env.scene);
        let snap = self.preview_snapshot.as_ref().expect("a stage has a clean snapshot");
        let (Some(n), Some(anchor)) = (self.normal, self.anchor) else { return false };
        let d = self.extrusion;
        let before: HashSet<VertexId> = snap.vertex_ids().iter().copied().collect();
        let movers: Vec<VertexId> = m.vertices.iter().copied().filter(|v| !before.contains(v) && ((m.position(*v).sub(anchor).dot(n) as f64) - d).abs() < STITCH_TOL).collect();
        let moving: HashSet<VertexId> = movers.iter().copied().collect();
        let mut levels: Vec<f64> = vec![0.0];
        levels.extend(self.plane_levels.iter().copied());
        if let Some(l) = self.limit_in {
            levels.push(-l);
        }
        // The faces each mover drags that keep a fixed vertex (the cap and
        // anything else wholly on the cap plane just translate with it).
        let mut anchored: std::collections::HashMap<VertexId, Vec<FaceId>> = movers.iter().map(|&v| (v, Vec::new())).collect();
        let mut touched: Vec<FaceId> = movers.iter().flat_map(|&v| m.faces_at(v)).collect();
        touched.sort();
        touched.dedup();
        for f in touched {
            let fc = m.face(f);
            let rest: Vec<VertexId> = std::iter::once(&fc.outer).chain(fc.holes.iter()).flatten().copied().filter(|w| !moving.contains(w)).collect();
            if rest.is_empty() {
                continue;
            }
            levels.extend(rest.iter().map(|&w| m.position(w).sub(anchor).dot(n) as f64));
            for lp in std::iter::once(&fc.outer).chain(fc.holes.iter()) {
                for w in lp {
                    if moving.contains(w) {
                        anchored.get_mut(w).expect("a mover").push(f);
                    }
                }
            }
        }
        if levels.iter().any(|lv| (lv - d).abs() <= STITCH_TOL) {
            // On a level: valid at exactly this distance, nothing to slide.
            self.stage_lo = d;
            self.stage_hi = d;
        } else {
            let rates: Vec<Option<V3>> = movers.iter().map(|v| slide_rate(m, &anchored[v], n, h)).collect();
            if rates.iter().any(|r| r.is_none()) {
                return false; // a face it drags would bend
            }
            self.stage_lo = levels.iter().copied().filter(|&lv| lv < d).fold(f64::NEG_INFINITY, f64::max);
            self.stage_hi = levels.iter().copied().filter(|&lv| lv > d).fold(f64::INFINITY, f64::min);
            self.stage_pos = movers.iter().zip(rates.iter()).map(|(&v, r)| (m.position(v), r.expect("a rate"))).collect();
            self.stage_verts = movers;
        }
        let old_f: HashSet<FaceId> = snap.face_ids().iter().copied().collect();
        let old_e: HashSet<EdgeId> = snap.edge_ids().iter().copied().collect();
        self.stage_new_faces = m.faces.iter().copied().filter(|f| !old_f.contains(f)).collect();
        self.stage_new_edges = m.edges.iter().copied().filter(|e| !old_e.contains(e)).collect();
        true
    }

    /// Keep the current stage for a return trip: its fields plus a snapshot of
    /// the staged model, captured the first time it is parked. One per side of
    /// the base plane; a newer stage replaces the older one.
    fn park_stage(&mut self, env: &mut Env) {
        let Some(stage_d) = self.stage_d else { return };
        if self.stage_parked {
            return;
        }
        let rec = StageRec {
            d: stage_d,
            lo: self.stage_lo,
            hi: self.stage_hi,
            verts: self.stage_verts.clone(),
            pos: self.stage_pos.clone(),
            new_faces: self.stage_new_faces.clone(),
            new_edges: self.stage_new_edges.clone(),
            applied: self.stage_applied,
            state: env.cx.mesh(env.scene).capture_state(),
        };
        self.stage_cache[if stage_d > 0.0 { 0 } else { 1 }] = Some(rec);
        self.stage_parked = true;
    }

    /// Show a parked stage again: one restore. `restore_state` resets every
    /// captured object and the lists, so whatever the model holds now needs no
    /// clean restore first.
    fn unpark_stage(&mut self, env: &mut Env, rec: StageRec) {
        env.cx.mesh_mut(env.scene).restore_state(&rec.state);
        self.stage_d = Some(rec.d);
        self.stage_lo = rec.lo;
        self.stage_hi = rec.hi;
        self.stage_verts = rec.verts;
        self.stage_pos = rec.pos;
        self.stage_new_faces = rec.new_faces;
        self.stage_new_edges = rec.new_edges;
        self.stage_applied = rec.applied;
        self.stage_parked = true;
    }

    /// Back to the clean model the stages are built on. The current stage is
    /// dropped; a parked copy survives.
    fn restore_clean(&mut self, env: &mut Env) {
        if let Some(s) = &self.preview_snapshot {
            env.cx.mesh_mut(env.scene).restore_state(s);
        }
        self.clear_stage();
    }

    /// Forget the current stage (the model's state is the caller's).
    fn clear_stage(&mut self) {
        self.stage_d = None;
        self.stage_applied = 0.0;
        self.stage_verts.clear();
        self.stage_pos.clear();
        self.stage_new_faces.clear();
        self.stage_new_edges.clear();
        self.stage_parked = false;
    }

    /// The commit of a held stage that already IS the result at `extrusion`:
    /// the cap slid there, the staged model captured as the command's result
    /// and the clean snapshot as its undo state. `None` when the pipeline must
    /// run: nothing staged, Ctrl, outside the interval.
    fn staged_command(&mut self, env: &mut Env) -> Option<Box<dyn ingetrazo_ops::command::Command>> {
        if self.stage_d.is_none() || self.keep_base || self.extrusion.abs() < MIN_EXTRUDE || !self.stage_holds(self.extrusion) {
            return None;
        }
        self.slide_stage(env, self.extrusion);
        let before = self.preview_snapshot.take()?;
        let cmd = SnapshotStates::new(before, env.cx.mesh(env.scene).capture_state());
        // The command owns the snapshots now: nothing is restored.
        self.stage_cache = [None, None];
        self.clear_stage();
        self.refused = false;
        Some(Box::new(cmd))
    }

    // ---- inference ----------------------------------------------------------

    pub fn inference_marker(&self) -> Option<(V3, &'static str)> {
        if !self.dragging {
            return None;
        }
        Some((self.inference_point?, self.inference_kind?))
    }

    /// `inference_guide_lines`: the "level with that point" segment, as
    /// `(reference, moved_anchor, kind)`.
    pub fn inference_guide_lines(&self) -> Vec<(V3, V3, &'static str)> {
        let (Some(p), true) = (self.inference_point, self.dragging) else { return Vec::new() };
        let (Some(anchor), Some(normal), Some(kind)) = (self.anchor, self.normal, self.inference_kind) else { return Vec::new() };
        vec![(p, anchor.add(normal.mul(self.extrusion as f32)), kind)]
    }

    /// `value_label`: the floating distance label and its midpoint.
    pub fn value_label(&self, units: &ModelUnits) -> Option<(String, V3)> {
        if !self.dragging {
            return None;
        }
        let (anchor, normal) = (self.anchor?, self.normal?);
        let midpoint = anchor.add(normal.mul((self.extrusion * 0.5) as f32));
        Some((fmt_len(self.extrusion.abs(), units), midpoint))
    }

    /// `_clamp_extrusion`: the inward limit ("Push limited to").
    fn clamp_extrusion(&mut self, env: &Env) {
        if let Some(limit) = self.limit_in {
            if self.extrusion < -limit {
                self.extrusion = -limit;
                self.flash(format!("Push limited to {} — deeper would leave the solid", fmt_len(limit, env.units)));
            }
        }
    }

    fn reference_hidden(&self, env: &Env, world: V3) -> bool {
        env.hidden_reference.is_some_and(|f| f(world))
    }

    /// `_infer_reference_distance`: the distance making the moved face level
    /// with the model geometry under the cursor. Records the engaged point
    /// for the overlay marker; `None` when nothing engages.
    fn infer_reference_distance(&mut self, env: &Env, ptr: Pointer) -> Option<f64> {
        let view = env.view?;
        let (anchor, normal) = (self.anchor?, self.normal?);
        let (sx, sy) = (ptr.x, ptr.y);
        let thr = env.snap_threshold_px;
        let exclude: HashSet<_> = self.cap_positions.iter().map(|&p| topo_key(p)).collect();
        // (dd, distance, point, kind)
        let mut best: Option<(f64, f64, V3, &'static str)> = None;
        if self.infer_cache.is_none() {
            let mut cache: Vec<(MeshId, Vec<VertexId>)> = Vec::new();
            let mut ids = vec![env.cx.mesh];
            ids.extend(env.scene.groups.iter().map(|g| g.mesh));
            let mut seen: HashSet<MeshId> = HashSet::new();
            for id in ids {
                if !seen.insert(id) {
                    continue; // shared prototypes: project once
                }
                let verts = env.scene.mesh(id).vertices.clone();
                if verts.is_empty() {
                    continue;
                }
                cache.push((id, verts));
            }
            self.infer_cache = Some(cache);
        }
        for (id, verts) in self.infer_cache.as_ref().unwrap() {
            let m = env.scene.mesh(*id);
            for &v in verts {
                let p = m.position(v);
                let Some(pix) = view.world_to_pixel(p) else { continue };
                let dd = (pix.0 - sx).powi(2) + (pix.1 - sy).powi(2);
                if dd > thr * thr {
                    continue;
                }
                if exclude.contains(&topo_key(p)) {
                    continue; // the base's own corners would pin the drag to 0
                }
                if best.as_ref().is_none_or(|b| dd < b.0) && !self.reference_hidden(env, p) {
                    best = Some((dd, p.sub(anchor).dot(normal) as f64, p, "vertex"));
                }
            }
        }
        // Tape Measure guides: a guide POINT is a corner like any vertex.
        let guides: Vec<_> = env.scene.guides.iter().collect();
        for g in &guides {
            if g.direction.is_some() {
                continue;
            }
            let Some(pix) = view.world_to_pixel(g.point) else { continue };
            let dd = (pix.0 - sx).powi(2) + (pix.1 - sy).powi(2);
            if dd <= thr * thr && best.as_ref().is_none_or(|b| dd < b.0) {
                best = Some((dd, g.point.sub(anchor).dot(normal) as f64, g.point, "guide_point"));
            }
        }
        // The origin is a point like a corner (#407).
        let origin = V3(0.0, 0.0, 0.0);
        if let Some(pix) = view.world_to_pixel(origin) {
            let dd = (pix.0 - sx).powi(2) + (pix.1 - sy).powi(2);
            if dd <= thr * thr && best.as_ref().is_none_or(|b| dd < b.0) {
                best = Some((dd, origin.sub(anchor).dot(normal) as f64, origin, "origin"));
            }
        }
        if let Some(b) = best {
            self.inference_point = Some(b.2);
            self.inference_kind = Some(b.3);
            return Some(b.1);
        }
        if let Some((on, dist)) = self.infer_guide_line(view, &guides, sx, sy, thr, anchor, normal) {
            self.inference_point = Some(on);
            self.inference_kind = Some("guide_line");
            return Some(dist);
        }
        // No corner nearby: the EDGE under the cursor.
        if let Some(he) = env.hover_edge.filter(|he| he.edge.is_none_or(|e| !self.stage_new_edges.contains(&e))) {
            let (a, b) = (he.a, he.b);
            let ab = b.sub(a);
            let h = view.hypot;
            if (!exclude.contains(&topo_key(a)) || !exclude.contains(&topo_key(b))) && (ab.length(h) as f64) > 1e-9 {
                let on = view.project_to_lock_line(a, ab, sx, sy);
                let t = (on.sub(a).dot(ab) as f64) / (ab.dot(ab) as f64);
                let on = a.add(ab.mul(ingetrazo_math::pyfloat::py_max2(0.0, ingetrazo_math::pyfloat::py_min2(1.0, t)) as f32));
                let dist = on.sub(anchor).dot(normal) as f64;
                if dist.abs() >= MIN_EXTRUDE && !self.reference_hidden(env, on) {
                    self.inference_point = Some(on);
                    self.inference_kind = Some("edge");
                    return Some(dist);
                }
            }
        }
        // No edge either: align to the face under the cursor (its plane).
        let pick = env.picker.pick_face_any(env.view, env.cx.mesh(env.scene), sx, sy);
        if let Some(p) = pick {
            if Some(p.face) != self.base_face && !self.stage_new_faces.contains(&p.face) {
                if let Some((origin, direction)) = view.pixel_to_ray(sx, sy) {
                    let m = env.cx.mesh(env.scene);
                    let h = view.hypot;
                    let pf = PFace::of(m, p.face);
                    let fpt = ingetrazo_geom::meshface::centroid(&pf.vertices);
                    let fnr = pf.normal(h).normalized(h);
                    let denom = fnr.dot(direction) as f64;
                    if denom.abs() >= 1e-6 {
                        let t = (fnr.dot(fpt.sub(origin)) as f64) / denom;
                        if t > 0.0 {
                            let hit = origin.add(direction.mul(t as f32));
                            let dist = hit.sub(anchor).dot(normal) as f64;
                            if dist.abs() >= MIN_EXTRUDE {
                                self.inference_point = Some(hit);
                                self.inference_kind = Some("face");
                                return Some(dist);
                            }
                        }
                    }
                }
            }
        }
        self.inference_point = None;
        self.inference_kind = None;
        None
    }

    /// `_infer_guide_line`: the guide LINE under the cursor, as `(point on it,
    /// push distance)`.
    fn infer_guide_line(&self, view: &View, guides: &[&ingetrazo_model::guide::Guide], sx: f64, sy: f64, thr: f64, anchor: V3, normal: V3) -> Option<(V3, f64)> {
        let lines: Vec<_> = guides.iter().filter(|g| g.direction.is_some()).collect();
        if lines.is_empty() {
            return None;
        }
        let (origin, direction) = view.pixel_to_ray(sx, sy)?;
        let mut best: Option<(f64, V3, f64)> = None;
        for g in lines {
            let gd = g.direction?;
            if (gd.dot(normal) as f64).abs() > 0.999 {
                continue;
            }
            // closest points between the guide line and the cursor ray
            let w0 = g.point.sub(origin);
            let a = gd.dot(gd) as f64;
            let b = gd.dot(direction) as f64;
            let c = direction.dot(direction) as f64;
            let d = gd.dot(w0) as f64;
            let e = direction.dot(w0) as f64;
            let den = a * c - b * b;
            if den.abs() < 1e-12 {
                continue;
            }
            let s = (b * e - c * d) / den;
            let on = g.point.add(gd.mul(s as f32));
            let Some(pix) = view.world_to_pixel(on) else { continue };
            let dd = (pix.0 - sx).powi(2) + (pix.1 - sy).powi(2);
            if dd > thr * thr {
                continue;
            }
            let dist = on.sub(anchor).dot(normal) as f64;
            if dist.abs() < MIN_EXTRUDE {
                continue;
            }
            if best.as_ref().is_none_or(|b| dd < b.0) {
                best = Some((dd, on, dist));
            }
        }
        best.map(|b| (b.1, b.2))
    }

    // ---- commit -----------------------------------------------------------------

    fn commit(&mut self, env: &mut Env) {
        self.hover = None; // the hovered face is about to be replaced
        self.light_faces.clear();
        self.light_rings.clear();
        self.suppressed.clear();
        // A held stage showing this very distance is kept as the result;
        // otherwise drop the live preview and redo it for real.
        let staged = self.staged_command(env);
        if staged.is_none() {
            self.revert_preview(env);
        }
        let Some(face) = self.base_face else {
            self.reset();
            self.spend_keep_mode();
            return;
        };
        if self.extrusion.abs() < MIN_EXTRUDE {
            self.reset();
            self.spend_keep_mode();
            return;
        }
        // One snapshot wraps the edit *and* the watertight stitch.
        self.topped_out = false;
        let shared = Rc::new(RefCell::new(self.current_push(face)));
        let cmd: Box<dyn ingetrazo_ops::command::Command> = match staged {
            Some(c) => c,
            None => Box::new(PushPullCommand::new(shared.clone())),
        };
        env.history.execute(env.scene, env.cx, cmd);
        let serial = env.history.last_serial;
        {
            let p = shared.borrow();
            self.extrusion = p.extrusion; // the reduced distance on a top-out
            self.refused = p.refused;
            self.topped_out = p.topped_out;
        }
        // The retype window's payload, captured BEFORE reset clears the drag
        // state (the sign is the direction actually committed).
        let mut last: Option<Retype> = None;
        let make = |t: &PushPullTool| Retype {
            serial: serial.unwrap_or(0),
            face,
            keep_base: t.keep_base,
            prism_cap: t.prism_cap,
            normal: t.normal,
            cap_positions: t.cap_positions.clone(),
            pre_oriented: t.drag_pre_oriented,
            limit_in: t.limit_in,
            sign: if t.extrusion >= 0.0 { 1.0 } else { -1.0 },
        };
        if self.topped_out {
            self.flash(format!("Push stopped at {}: going further would break the solid", fmt_len(self.extrusion.abs(), env.units)));
            self.last_distance = Some(self.extrusion);
            last = Some(make(self));
        } else if self.refused {
            // The guard rolled the push back to keep the solid watertight.
            self.flash("Push refused: would break the solid (left it unchanged)".to_string());
        } else {
            self.last_distance = Some(self.extrusion); // double-click repeats it
            last = Some(make(self));
        }
        self.reset();
        self.spend_keep_mode();
        self.last = last; // arms the retype window, if one opened
    }

    fn reset(&mut self) {
        self.hovered_face = None;
        self.base_face = None;
        self.extrusion = 0.0;
        self.dragging = false;
        self.attached = false;
        self.prism_cap = false;
        self.prism_applied = 0.0;
        self.prism_verts.clear();
        self.keep_base = false;
        self.limit_in = None;
        self.hover_closed_group = false;
        self.selected_arm = false;
        self.anchor = None;
        self.grab = None;
        self.normal = None;
        self.cap_positions.clear();
        self.drag_pre_oriented = false;
        self.inference_point = None;
        self.inference_kind = None;
        self.infer_cache = None;
        self.preview_snapshot = None;
        self.clear_stage();
        self.stage_cache = [None, None];
        self.stage_refusal = None;
        self.plane_levels.clear();
        self.refused = false;
        self.tiny_hint_shown = false;
        self.press_screen = None;
        self.last_screen = None;
        self.light_faces.clear();
        self.light_rings.clear();
        self.last = None; // the retype window closes with the drag
    }

    // ---- modifier: Ctrl keeps the starting face -----------------------------

    /// `on_key` for the Ctrl press: swallowed - the tap acts on the RELEASE.
    pub fn on_key_ctrl_press(&self) -> bool {
        true
    }

    /// `on_key_release(Ctrl)`: `tapped` is whether the released Ctrl was a tap
    /// of its own (`ctrl_tapped`). `true` when handled.
    pub fn on_key_ctrl_release(&mut self, tapped: bool) -> bool {
        if self.ctrl_used {
            self.ctrl_used = false;
            return true; // the gesture's own Ctrl, not a tap
        }
        if !tapped {
            return false; // Ctrl was part of a shortcut
        }
        self.keep_mode = !self.keep_mode;
        self.keep_base = self.keep_mode;
        self.flash(if self.keep_mode { "Keep the base: on" } else { "Keep the base: off" }.to_string());
        true
    }

    /// `status_clause`: the Ctrl clause of the status bar.
    pub fn status_clause(&self) -> (bool, &'static str, &'static str) {
        (self.keep_mode, "keep the base", "move the base")
    }

    /// `_sync_keep_base`: read Ctrl where it is HELD; `latch_ctrl` is true
    /// only where a CLICK happens.
    fn sync_keep_base(&mut self, ctrl: bool, latch_ctrl: bool) {
        if latch_ctrl {
            self.ctrl_used = ctrl;
        }
        self.keep_base = ctrl || self.keep_mode;
    }

    /// `_spend_keep_mode`: the Ctrl mode is spent by the push it armed.
    fn spend_keep_mode(&mut self) {
        if !self.keep_mode {
            return;
        }
        self.keep_mode = false;
        self.keep_base = false;
    }

    // ---- test harness hooks --------------------------------------------------

    /// `_push` of `tests/test_fuzz_engine.py`: the drag state set up by hand
    /// (anchor, the face's own normal, classification, inward limit), the
    /// extrusion clamped and committed. `false` when the clamp left nothing
    /// to push. For headless callers that have no pointer.
    pub fn push_direct(&mut self, env: &mut Env, face: FaceId, dist: f64, keep_base: bool) -> bool {
        let h = env.cx.hypot;
        self.base_face = Some(face);
        self.dragging = true;
        let m = env.cx.mesh(env.scene);
        self.anchor = Some(ingetrazo_geom::meshface::centroid(&ingetrazo_model::topology::positions(m, &m.face(face).outer)));
        self.normal = Some(m.face_normal(face, h));
        let (attached, prism_cap) = classify_base(m, face, h);
        self.attached = attached;
        self.prism_cap = prism_cap;
        self.cap_positions = cap_loop_positions(m, face);
        self.keep_base = keep_base;
        self.limit_in = compute_inward_limit(m, face, self.normal, self.anchor, self.attached, h);
        self.extrusion = dist;
        self.clamp_extrusion(env);
        if self.extrusion.abs() < 1e-6 {
            self.reset();
            return false;
        }
        self.commit(env);
        true
    }
}
