// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! The viewport as the drawing tools meet it, with no window: the pointer,
//! the keys and the value box in; the snap, the work plane and the tool's
//! calls out (`views/viewport.py`: `_process_hover`, `_build_ctx`,
//! `_snap_at`, `_note_snap`, `mousePressEvent` / `_dispatch_tool_click`,
//! `mouseReleaseEvent`, `keyPressEvent` / `keyReleaseEvent`, `escape`,
//! `_handle_value_key`, `_refresh_snap`, the Shift lock and the arrow axis
//! locks, and `MainWindow._on_undo` / `_on_redo`).
//!
//! [`Core`] is what a tool reaches through its `viewport` argument: the
//! document, its history, the camera, and what the viewport remembers
//! between events (the last work plane, the last snap, the last pointer, the
//! messages flashed, the typed buffer). [`Host`] owns it with the active
//! tool and runs the events.
//!
//! Not carried: the hover coalescing timer (`mouseMoveEvent`'s gate - every
//! hover is processed), the dwell timer (a pause becomes an encouraged point
//! through [`Host::dwell`]), camera navigation (the application's), the
//! section cut, the status bar and the cursor icons. A box the Select tool
//! was dragged is handed over by [`Host::box_select`]. Drawing happens in the
//! world axes (no group frame is synced).

use ingetrazo_math::{Hypot, V3};
use ingetrazo_model::mesh::Mesh;
use ingetrazo_model::scene::Scene;
use ingetrazo_model::units::{fmt_len, ModelUnits};
use ingetrazo_ops::command::{Command, Cx, History};
use ingetrazo_snap::engine::{self, InferenceState, Settings, SnapInput};
use ingetrazo_snap::hover::HoverEdge as SnapHover;
use ingetrazo_snap::types::{Axes, Axis};
use ingetrazo_snap::workplane::{PlaneCx, ToolState};
use ingetrazo_snap::{LinearMode, RefMode, SnapKind, SnapResult, View, World};

use crate::base::*;
use crate::modify_base::Selection;
use crate::vcb::{parse_value_buffer, VcbValue};

/// `Viewport._ENCOURAGING_KINDS`.
fn encouraging(k: SnapKind) -> bool {
    matches!(
        k,
        SnapKind::Endpoint | SnapKind::Midpoint | SnapKind::ArcMidpoint | SnapKind::Center | SnapKind::ComponentOrigin | SnapKind::Origin | SnapKind::Intersection | SnapKind::Close
    )
}

/// `_SHIFT_LOCKABLE`.
fn shift_lockable(k: SnapKind) -> bool {
    matches!(k, SnapKind::Axis | SnapKind::AxisInference | SnapKind::Reference | SnapKind::ThroughPoint | SnapKind::PerpFace | SnapKind::Extension)
}

/// `QPointF.toPoint()` for one coordinate: Qt's `qRound`.
pub fn qround(d: f64) -> f64 {
    if d >= 0.0 {
        (d + 0.5).floor()
    } else {
        (d - 0.5).ceil()
    }
}

/// What a tool reaches through `viewport`.
pub struct Core {
    pub scene: Scene,
    pub cx: Cx,
    pub history: History,
    pub view: View,
    pub inference: InferenceState,
    pub settings: Settings,
    pub axes: Axes,
    pub units: ModelUnits,
    /// `_last_work_plane`.
    pub last_work_plane: Option<(V3, V3)>,
    /// `last_snap`.
    pub last_snap: Option<SnapResult>,
    /// `_last_mouse_pos`.
    pub last_mouse_pos: Option<(f64, f64)>,
    /// `QGuiApplication.keyboardModifiers()`.
    pub kb_mods: u32,
    /// Every `flash_status` text, in order.
    pub flashes: Vec<String>,
    /// `_value_buffer`.
    pub value_buffer: String,
    /// `_ctrl_alone`.
    pub ctrl_alone: bool,
    /// `inference._dwell_point`.
    pub dwell_point: Option<V3>,
    /// The mesh edge `_hover_edge` is, when it is a loose-mesh edge.
    pub hover_loose_edge: Option<ingetrazo_model::EdgeId>,
    /// `scene.version`: bumped by every change of the model.
    pub version: u64,
    /// `scene.selection`: what Select picked, and what the modify tools act on.
    pub selection: Selection,
    /// What the acquired circle centre (`inference.center_ref`) was read
    /// from, and the model version it was read at (`_center_ref[2..5]`).
    center_src: Option<CenterSrc>,
    center_version: u64,
    world: Option<World>,
    /// What the snap world keeps between its rebuilds (see `WorldCache`).
    world_cache: ingetrazo_snap::world_cache::WorldCache,
}

/// Where a circle's centre was found: a curve edge, or a face's boundary run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CenterSrc {
    Edge(ingetrazo_model::EdgeId),
    Face(ingetrazo_model::FaceId, (usize, usize)),
}

impl Core {
    pub fn h(&self) -> Hypot {
        self.view.h
    }

    pub fn flash(&mut self, text: String) {
        self.flashes.push(text);
    }

    /// `scene.mesh`.
    pub fn mesh(&self) -> &Mesh {
        self.cx.mesh(&self.scene)
    }

    /// The snap world, built on demand and dropped when the model changes.
    pub fn world(&mut self) -> &World {
        if self.world.is_none() {
            self.world = Some(World::build_cached(&self.scene, self.view.h, &mut self.world_cache));
        }
        self.world.as_ref().unwrap()
    }

    pub fn invalidate(&mut self) {
        self.world = None;
    }

    /// The model changed behind the core's back (a tool executed through its
    /// own environment): the snap world is rebuilt on the next ask.
    pub fn touch(&mut self) {
        self.world = None;
    }

    /// What no longer exists leaves the selection (`Env::reconcile_selection`).
    pub fn reconcile_selection(&mut self) {
        let scene = &self.scene;
        let mesh = self.cx.mesh(scene);
        self.selection.items.retain(|s| match s {
            crate::modify_base::Sel::Face(f) => mesh.faces.contains(f),
            crate::modify_base::Sel::Edge(e) => mesh.edges.contains(e),
            crate::modify_base::Sel::Group(p) => ingetrazo_ops::group::group_at(&scene.groups, p).is_some(),
            crate::modify_base::Sel::Dimension(i) => *i < scene.dimensions.len(),
            crate::modify_base::Sel::Label(i) => *i < scene.text_labels.len(),
            crate::modify_base::Sel::Guide(i) => *i < scene.guides.len(),
            crate::modify_base::Sel::Section(i) => *i < scene.section_planes.len(),
            crate::modify_base::Sel::Image(i) => *i < scene.image_planes.len(),
            crate::modify_base::Sel::GeoPath(i) => *i < scene.geo_paths.len(),
        });
    }

    pub fn world_to_pixel(&self, p: V3) -> Option<(f64, f64)> {
        self.view.world_to_pixel(p)
    }

    pub fn pixel_to_ray(&self, x: f64, y: f64) -> Option<(V3, V3)> {
        self.view.pixel_to_ray(x, y)
    }

    pub fn eye(&self) -> V3 {
        self.view.cam.eye()
    }

    pub fn fmt_len(&self, metres: f64) -> String {
        fmt_len(metres, &self.units)
    }

    /// `history.execute(cmd)`.
    pub fn exec(&mut self, cmd: Box<dyn Command>) {
        self.history.execute(&mut self.scene, &mut self.cx, cmd);
        self.world = None;
        self.version += 1;
    }

    pub fn history_undo(&mut self) -> bool {
        let r = self.history.undo(&mut self.scene, &mut self.cx);
        self.world = None;
        self.version += 1;
        r
    }

    pub fn history_redo(&mut self) -> bool {
        let r = self.history.redo(&mut self.scene, &mut self.cx);
        self.world = None;
        self.version += 1;
        r
    }

    pub fn undo_len(&self) -> usize {
        self.history.undo_len()
    }

    /// `axis_lock`.
    pub fn axis_lock(&self) -> Option<Axis> {
        self.inference.axis_lock
    }

    /// `pick_face_placement(x, y)`: the plane `(point, normal)` of the face
    /// under the pixel, through the placement it was hit by.
    pub fn pick_face_placement(&mut self, x: f64, y: f64) -> Option<(V3, V3)> {
        let view = self.view;
        let w = self.world();
        let e = w.face_under(&view, x, y)?;
        Some(w.ents[e].plane_place)
    }

    /// `pick_face_any(x, y)` read for its plane: through the owner.
    pub fn pick_face_owner(&mut self, x: f64, y: f64) -> Option<(V3, V3)> {
        let view = self.view;
        let w = self.world();
        let e = w.face_under(&view, x, y)?;
        Some(w.ents[e].plane_owner)
    }

    /// `inference.clear_references()` as the viewport reads it.
    pub fn clear_references(&mut self) {
        self.inference.clear_references();
        self.inference.hover_center = None;
        self.dwell_point = None;
        self.last_snap = None;
        self.hover_loose_edge = None;
        self.center_src = None;
    }

    /// `Viewport._valid_center_ref`: the acquired centre, re-checked against
    /// the model once it has changed - the circle may be gone, or moved, and
    /// the centre follows.
    pub fn valid_center_ref(&mut self) {
        if self.inference.center_ref.is_none() {
            return;
        }
        if self.center_version == self.version {
            return;
        }
        let h = self.view.h;
        let m = self.cx.mesh(&self.scene);
        let fresh = match self.center_src {
            Some(CenterSrc::Face(f, key)) => {
                if m.faces.contains(&f) {
                    ingetrazo_snap::circle::curve_centers_of_face(m, f, None, h).into_iter().find(|c| c.2 == key).map(|c| (c.0, c.1))
                } else {
                    None
                }
            }
            Some(CenterSrc::Edge(e)) => {
                if m.edges.contains(&e) {
                    ingetrazo_snap::circle::center_of_edge(m, e)
                } else {
                    None
                }
            }
            None => None,
        };
        self.inference.center_ref = fresh.map(|(c, r)| (c, r as f32));
        self.center_version = self.version;
        if fresh.is_none() {
            self.center_src = None;
        }
    }

    /// What `_update_center_ref` found the centre through: the hovered
    /// curve edge, else a curve on the face under the cursor.
    fn find_center_src(&mut self, x: f64, y: f64) -> Option<CenterSrc> {
        let view = self.view;
        let pick = self.settings.pick_threshold_px;
        let h = view.h;
        let w = self.world.as_ref()?;
        if let Some(SnapHover::Loose(i)) = w.pick_edge_any(&view, pick, x, y) {
            if let Some(cid) = w.loose_edges[i].curve {
                if w.curve_centers.get(&cid).copied().flatten().is_some() {
                    return Some(CenterSrc::Edge(w.loose_edge_ids[i]));
                }
            }
        }
        let e = w.face_under(&view, x, y)?;
        let ingetrazo_snap::world::FaceRef::Loose(f) = w.ents[e].face else { return None };
        let centers = ingetrazo_snap::circle::curve_centers_of_face(&w.meshes[w.loose_id as usize], f, None, h);
        if centers.is_empty() {
            return None;
        }
        if centers.len() == 1 {
            return Some(CenterSrc::Face(f, centers[0].2));
        }
        let mut best: Option<(f64, (usize, usize))> = None;
        for (c, _r, key) in &centers {
            let Some(pc) = view.world_to_pixel(*c) else { continue };
            let d = ingetrazo_geom::bevel_profiles::py_hypot2(pc.0 - x, pc.1 - y);
            if best.is_none() || d < best.unwrap().0 {
                best = Some((d, *key));
            }
        }
        best.map(|b| CenterSrc::Face(f, b.1))
    }

    /// `_set_value_buffer`.
    pub fn set_value_buffer(&mut self, text: &str) {
        if self.value_buffer != text {
            self.value_buffer = text.to_string();
        }
    }

    /// `_current_token_tail`: the buffer after the last `;` or space.
    fn token_tail(&self) -> String {
        let normalized = self.value_buffer.replace(';', " ");
        match normalized.rfind(' ') {
            None => self.value_buffer.clone(),
            Some(i) => self.value_buffer[i + 1..].to_string(),
        }
    }
}

/// The document, the camera and the tool: a viewport with no window.
pub struct Host {
    pub core: Core,
    tool: Option<Box<dyn DrawTool>>,
    alt_tap: bool,
    /// `_shift_plane_mark is tool`.
    shift_plane_mark: bool,
    /// `_last_double`: where the last double-click was.
    last_double: Option<(f64, f64)>,
}

impl Host {
    pub fn new(scene: Scene, cx: Cx, view: View) -> Host {
        Host {
            core: Core {
                scene,
                cx,
                history: History::new(),
                view,
                inference: InferenceState::default(),
                settings: Settings::default(),
                axes: Axes::world(),
                units: ModelUnits::default(),
                last_work_plane: None,
                last_snap: None,
                last_mouse_pos: None,
                kb_mods: 0,
                flashes: Vec::new(),
                value_buffer: String::new(),
                ctrl_alone: false,
                dwell_point: None,
                hover_loose_edge: None,
                version: 1,
                selection: Selection::default(),
                center_src: None,
                center_version: 0,
                world: None,
                world_cache: Default::default(),
            },
            tool: None,
            alt_tap: false,
            shift_plane_mark: false,
            last_double: None,
        }
    }

    pub fn tool(&self) -> &dyn DrawTool {
        self.tool.as_deref().expect("an active tool")
    }

    /// Run `f` with the tool and the core apart (the tool calls back into
    /// the core, not into the host).
    fn with_tool<R>(&mut self, f: impl FnOnce(&mut dyn DrawTool, &mut Core) -> R) -> R {
        let mut tool = self.tool.take().expect("an active tool");
        let r = f(tool.as_mut(), &mut self.core);
        self.tool = Some(tool);
        r
    }

    /// `Viewport.set_active_tool(tool)` for a different tool.
    pub fn set_active_tool(&mut self, tool: Box<dyn DrawTool>) {
        if self.tool.is_some() {
            self.with_tool(|t, c| t.on_deactivate(c));
        }
        self.tool = Some(tool);
        self.core.inference.linear_mode = None;
        self.core.inference.axis_lock = None;
        self.core.clear_references();
        self.with_tool(|t, c| t.on_activate(c));
    }

    /// `Viewport.set_active_tool(tool)` for the tool already held: it starts
    /// over.
    pub fn reselect_tool(&mut self) {
        self.with_tool(|t, c| t.on_cancel(c));
        self.core.clear_references();
        self.core.inference.axis_lock = None;
        self.core.inference.shift_lock = None;
        self.core.inference.reference_edge = None;
        self.core.inference.reference_mode = None;
        self.core.inference.linear_mode = None;
        self.refresh_snap();
    }

    // ---- the snap ------------------------------------------------------------------------

    fn tool_state(&self) -> ToolState {
        self.tool().snap_state(&self.core)
    }

    /// `_snap_at(px, py, modifiers)`.
    fn snap_at(&mut self, px: f64, py: f64, mods: u32) -> Option<SnapResult> {
        let ts = self.tool_state();
        if ts.uses_snap {
            // `_snap_scene` asks `_valid_center_ref()`.
            self.core.valid_center_ref();
        }
        let view = self.core.view;
        let axes = self.core.axes;
        let settings = self.core.settings;
        let inference = self.core.inference.clone();
        // The plane `_world_from_pixel` stashes as `_last_work_plane`.
        let plane = {
            let world = self.core.world();
            let pcx = PlaneCx { view: &view, world, axes: &axes, tool: &ts, axis_locked: inference.axis_lock.is_some(), inference_angle_deg: settings.inference_angle_deg };
            pcx.world_from_pixel(px, py).1
        };
        if plane.is_some() {
            self.core.last_work_plane = plane;
        }
        // `_current_work_plane` hands a PLANAR shape's capture to the face the
        // cursor is over (#545): `tool.work_plane = face_plane`.
        if let (true, Some(captured), Some(p)) = (ts.is_plane_lock, ts.work_plane, plane) {
            if p != captured && ts.own_plane.is_none() {
                self.with_tool(|t, _| t.set_work_plane(Some(p)));
            }
        }
        let world = self.core.world();
        let inp = SnapInput { view: &view, world, axes: &axes, tool: &ts, state: &inference, settings: &settings, shift_held: mods & MOD_SHIFT != 0 };
        engine::snap(&inp, px, py).0
    }

    /// `_build_ctx(ev)` for a pointer at `(x, y)`.
    fn build_ctx(&mut self, x: f64, y: f64, mods: u32) -> Option<Ctx> {
        let snap = self.snap_at(qround(x), qround(y), mods)?;
        Some(Ctx { world: snap.point, screen: (x, y), mods, snap })
    }

    /// `_note_snap`.
    fn note_snap(&mut self, snap: &SnapResult) {
        self.core.last_snap = Some(snap.clone());
        let k = snap.kind;
        self.with_tool(|t, _| t.set_snap_kind(Some(k)));
    }

    /// `inference.acquire(x, y)` and the hover edge it leaves.
    fn acquire(&mut self, x: f64, y: f64) {
        let ts = self.tool_state();
        let view = self.core.view;
        let settings = self.core.settings;
        // The world is built first: acquire borrows it beside the state.
        self.core.world();
        let world = self.core.world.as_ref().unwrap();
        self.core.inference.acquire(&view, world, &ts, &settings, x, y);
        let pick = settings.pick_threshold_px;
        let hov = if ts.uses_snap || ts.hover_group_edges { world.pick_edge_any(&view, pick, x, y) } else { None };
        self.core.hover_loose_edge = match hov {
            Some(SnapHover::Loose(i)) => world.loose_edge_ids.get(i).copied(),
            _ => None,
        };
        if ts.uses_snap {
            if self.core.inference.hover_center.is_some() {
                // `_center_ref = found`
                self.core.center_src = self.core.find_center_src(x, y);
                self.core.center_version = self.core.version;
            } else {
                // Nothing under the cursor keeps the last centre, re-checked.
                self.core.valid_center_ref();
            }
        }
    }

    /// `Viewport._process_hover(pos, modifiers)`: the pointer moved.
    pub fn hover(&mut self, x: f64, y: f64, mods: u32) {
        self.core.last_mouse_pos = Some((x, y));
        self.acquire(x, y);
        let Some(ctx) = self.build_ctx(x, y, mods) else { return };
        self.note_snap(&ctx.snap);
        if self.tool().uses_snap() {
            // Encouraging: the point the snap already found, or the centre
            // of the circle under the cursor.
            let mut point = None;
            if encouraging(ctx.snap.kind) {
                point = Some(ctx.snap.point);
            }
            if let Some(center) = self.core.inference.hover_center {
                let by_edge = self.center_found_by_edge(x, y);
                if point.is_none() || by_edge {
                    point = Some(center.0);
                }
            }
            self.dwell_on(point);
        }
        self.with_tool(|t, c| t.on_hover(c, &ctx));
    }

    /// Whether the circle centre under the cursor came through the hovered
    /// edge (`center[2][0] == "edge"`) rather than through the face.
    fn center_found_by_edge(&mut self, x: f64, y: f64) -> bool {
        let view = self.core.view;
        let pick = self.core.settings.pick_threshold_px;
        let h = view.h;
        let w = self.core.world();
        let Some(hov) = w.pick_edge_any(&view, pick, x, y) else { return false };
        match hov {
            // A loose curve edge gives its own curve's centre (`_center_of_edge`).
            SnapHover::Loose(i) => w.loose_edges[i].curve.is_some_and(|cid| w.curve_centers.get(&cid).copied().flatten().is_some()),
            SnapHover::Group(_) => {
                let with_edge = w.update_center_ref(&view, Some(hov), x, y, h);
                let by_face = w.update_center_ref(&view, None, x, y, h);
                with_edge != by_face
            }
        }
    }

    fn dwell_on(&mut self, point: Option<V3>) {
        let h = self.core.h();
        match point {
            None => self.core.dwell_point = None,
            Some(p) => {
                if let Some(d) = self.core.dwell_point {
                    if (d.sub(p).length(h) as f64) < 1e-6 {
                        return;
                    }
                }
                self.core.dwell_point = Some(p);
            }
        }
    }

    /// The dwell timer fired: the point the cursor rested on is encouraged.
    pub fn dwell(&mut self) {
        if let Some(p) = self.core.dwell_point {
            let h = self.core.h();
            self.core.inference.encourage_point(p, h);
        }
    }

    /// `_refresh_snap`: the snap again at the last pointer, after a modifier
    /// or a lock changed with the hand still.
    pub fn refresh_snap(&mut self) {
        let Some((x, y)) = self.core.last_mouse_pos else { return };
        if !self.tool().uses_snap() {
            return;
        }
        let mods = self.core.kb_mods;
        let Some(snap) = self.snap_at(qround(x), qround(y), mods) else { return };
        self.note_snap(&snap);
        let ctx = Ctx { world: snap.point, screen: (x, y), mods, snap };
        self.with_tool(|t, c| t.on_hover(c, &ctx));
    }

    // ---- the buttons ------------------------------------------------------------------------

    /// `mousePressEvent`, left button.
    pub fn press(&mut self, x: f64, y: f64, mods: u32) {
        self.release_linear_mode();
        if let Some((lx, ly)) = self.last_double {
            if (x - lx).abs() + (y - ly).abs() < 8.0 {
                // A press right after a double-click in place: a triple
                // click, which these tools take as a click.
                self.last_double = None;
                if let Some(ctx) = self.build_ctx(x, y, mods) {
                    self.with_tool(|t, c| t.on_click(c, &ctx));
                }
                return;
            }
        }
        self.dispatch_tool_click(x, y, mods, false);
    }

    /// `mouseDoubleClickEvent`, left button.
    pub fn double_click(&mut self, x: f64, y: f64, mods: u32) {
        self.last_double = Some((x, y));
        self.dispatch_tool_click(x, y, mods, true);
    }

    /// `_dispatch_tool_click`.
    fn dispatch_tool_click(&mut self, x: f64, y: f64, mods: u32, double: bool) {
        let had_start = self.tool().start_point().is_some();
        let had_plane = self.tool().work_plane().is_some();
        let mut face_plane = None;
        if !had_start && !had_plane {
            face_plane = self.core.pick_face_placement(x, y);
        }
        let Some(ctx) = self.build_ctx(x, y, mods) else { return };
        self.note_snap(&ctx.snap);
        let before = self.core.undo_len();
        self.with_tool(|t, c| if double { t.on_double_click(c, &ctx) } else { t.on_click(c, &ctx) });
        let drew = self.core.undo_len() > before;
        let now_start = self.tool().start_point();
        if !had_start && now_start.is_some() && face_plane.is_some() && self.tool().work_plane().is_none() {
            let fp = face_plane;
            self.with_tool(|t, _| t.set_work_plane(fp));
        }
        self.core.set_value_buffer("");
        if let Some(err) = self.core.history.last_error.clone() {
            self.core.flash(format!("Operation failed and was undone: {err}"));
        }
        self.release_axis_lock_after_operation(drew);
    }

    /// `mouseReleaseEvent` of a box that was dragged: the Select tool's
    /// rubber band (`rect` is `(x0, y0, x1, y1)`, `crossing` a right-to-left drag).
    pub fn box_select(&mut self, rect: [f64; 4], crossing: bool, mods: u32) {
        self.with_tool(|t, c| t.on_box_select(c, rect, crossing, mods));
    }

    /// The active tool, for the UI to read and to key by.
    pub fn tool_mut(&mut self) -> &mut dyn DrawTool {
        self.tool.as_deref_mut().expect("an active tool")
    }

    /// Whether a tool is active.
    pub fn has_tool(&self) -> bool {
        self.tool.is_some()
    }

    /// `mouseReleaseEvent`, left button.
    pub fn release(&mut self) {
        self.with_tool(|t, c| t.on_release(c));
        self.release_axis_lock_after_operation(false);
    }

    // ---- locks ------------------------------------------------------------------------------

    fn tool_busy(&self) -> bool {
        self.tool().busy()
    }

    /// `_release_linear_mode`.
    fn release_linear_mode(&mut self) {
        if matches!(self.core.inference.linear_mode, None | Some(LinearMode::All)) {
            return;
        }
        if self.tool_busy() {
            return;
        }
        self.core.inference.linear_mode = None;
        self.refresh_snap();
    }

    /// `_release_axis_lock_after_operation`.
    fn release_axis_lock_after_operation(&mut self, committed: bool) {
        if self.core.inference.axis_lock.is_none() {
            return;
        }
        if !committed && self.tool_busy() {
            return;
        }
        self.core.inference.axis_lock = None;
        self.refresh_snap();
    }

    /// `_cycle_linear_inference_mode`.
    fn cycle_linear_inference_mode(&mut self) {
        self.core.inference.linear_mode = match self.core.inference.linear_mode.unwrap_or(LinearMode::All) {
            LinearMode::All => Some(LinearMode::Off),
            LinearMode::Off => Some(LinearMode::ParallelPerp),
            LinearMode::ParallelPerp => Some(LinearMode::All),
        };
        self.refresh_snap();
    }

    /// `_cycle_reference_mode`: Down.
    fn cycle_reference_mode(&mut self) {
        let inf = &mut self.core.inference;
        match inf.reference_mode {
            None => {
                let Some(e) = inf.hover_edge else { return };
                inf.reference_edge = Some(e);
                inf.reference_mode = Some(RefMode::Parallel);
            }
            Some(RefMode::Parallel) => inf.reference_mode = Some(RefMode::Perpendicular),
            Some(RefMode::Perpendicular) => {
                inf.reference_edge = None;
                inf.reference_mode = None;
            }
        }
    }

    /// `release_constraints`: the first stop of the Esc cascade.
    fn release_constraints(&mut self) -> bool {
        let inf = &mut self.core.inference;
        if inf.axis_lock.is_none() && inf.reference_mode.is_none() {
            return false;
        }
        inf.axis_lock = None;
        inf.reference_edge = None;
        inf.reference_mode = None;
        self.refresh_snap();
        true
    }

    /// `Viewport.escape`.
    pub fn escape(&mut self) {
        if !self.core.value_buffer.is_empty() {
            self.core.set_value_buffer("");
            return;
        }
        if self.release_constraints() {
            return;
        }
        // Right after a copy the array window is what is open: Esc closes it.
        if self.tool().accepts_array() && self.tool().array_pending(&self.core) {
            self.with_tool(|t, _| t.close_array_window());
            return;
        }
        if self.tool_busy() {
            self.with_tool(|t, c| t.on_cancel(c));
            self.release_linear_mode();
            return;
        }
        // Then the selection.
        if !self.core.selection.is_empty() {
            self.core.selection.clear();
            self.core.version += 1;
            return;
        }
        // Nothing selected, no group open: the tool's own cancel.
        self.with_tool(|t, c| t.on_cancel(c));
    }

    /// `_capture_shift_lock`.
    fn capture_shift_lock(&mut self) {
        self.core.inference.shift_lock = None;
        self.mark_plane_for_shift();
        let start = self.tool().start_point();
        let h = self.core.h();
        let (Some(snap), Some(start)) = (self.core.last_snap.clone(), start) else { return };
        if !shift_lockable(snap.kind) {
            return;
        }
        let d = if let Some(a) = snap.axis {
            let mut d = self.core.axes.axis(a);
            if (snap.point.sub(start).dot(d) as f64) < 0.0 {
                d = d.neg();
            }
            d
        } else {
            snap.point.sub(start)
        };
        if (d.length(h) as f64) > 1e-6 {
            self.core.inference.shift_lock = Some((d.normalized(h), snap.color));
        }
    }

    /// `_mark_plane_for_shift`.
    fn mark_plane_for_shift(&mut self) {
        self.shift_plane_mark = false;
        let h = self.core.h();
        let marked = self.with_tool(|t, _| t.planar().map(|st| mark_plane(st, h)).unwrap_or(false));
        if marked {
            self.shift_plane_mark = true;
        }
    }

    /// `_release_shift_plane_mark`.
    fn release_shift_plane_mark(&mut self) {
        let had = std::mem::replace(&mut self.shift_plane_mark, false);
        if !had {
            return;
        }
        self.with_tool(|t, _| {
            if let Some(st) = t.planar() {
                if st.plane_ref.is_some_and(|r| r.1 == RefKind::Plane) {
                    st.plane_ref = None;
                }
            }
        });
    }

    // ---- the keys -------------------------------------------------------------------------

    /// `QWidget.event` for a key press, then `keyPressEvent`.
    pub fn key_press(&mut self, key: i32, mods: u32, text: &str) -> bool {
        let is_modifier = matches!(key, KEY_CONTROL | KEY_SHIFT | KEY_ALT | KEY_META | KEY_ALTGR);
        if !is_modifier {
            self.core.ctrl_alone = false;
        }
        // The held state the application reports: a modifier is down from
        // its own press.
        match key {
            KEY_SHIFT => self.core.kb_mods |= MOD_SHIFT,
            KEY_CONTROL => self.core.kb_mods |= MOD_CONTROL,
            KEY_ALT => self.core.kb_mods |= MOD_ALT,
            KEY_META => self.core.kb_mods |= MOD_META,
            _ => {}
        }
        if key == KEY_CONTROL {
            self.core.ctrl_alone = true;
        }
        if key == KEY_DOWN {
            self.shift_plane_mark = false;
        }
        if key == KEY_SHIFT {
            self.capture_shift_lock();
            self.refresh_snap();
        }
        if key == KEY_ALT {
            // (the eyedropper cursor is Paint's)
        } else if self.alt_tap {
            self.alt_tap = false;
        }
        if self.handle_value_key(key, mods, text) {
            return true;
        }
        if self.with_tool(|t, c| t.on_key(c, key, mods)) {
            return true;
        }
        if key == KEY_ESCAPE {
            self.escape();
            return true;
        }
        if key == KEY_ALT {
            if self.tool_busy() {
                self.alt_tap = true;
                self.cycle_linear_inference_mode();
            }
            return true;
        }
        match key {
            KEY_RIGHT | KEY_LEFT | KEY_UP => {
                let a = plane_lock_axis(key).expect("an arrow");
                let inf = &mut self.core.inference;
                inf.axis_lock = if inf.axis_lock == Some(a) { None } else { Some(a) };
                self.refresh_snap();
                true
            }
            KEY_DOWN => {
                self.cycle_reference_mode();
                self.refresh_snap();
                true
            }
            _ => false,
        }
    }

    /// Whether a printable character goes to the value box (a digit, a
    /// separator or a unit letter after a digit...): what `keyPressEvent`
    /// does first with a character, before any shortcut is looked at.
    pub fn type_text(&mut self, mods: u32, text: &str) -> bool {
        self.core.ctrl_alone = false;
        self.handle_value_key(0, mods, text)
    }

    /// `keyReleaseEvent`.
    pub fn key_release(&mut self, key: i32) {
        match key {
            KEY_SHIFT => self.core.kb_mods &= !MOD_SHIFT,
            KEY_CONTROL => self.core.kb_mods &= !MOD_CONTROL,
            KEY_ALT => self.core.kb_mods &= !MOD_ALT,
            KEY_META => self.core.kb_mods &= !MOD_META,
            _ => {}
        }
        if self.with_tool(|t, c| t.on_key_release(c, key)) {
            return;
        }
        if key == KEY_ALT && self.alt_tap {
            self.alt_tap = false;
            return;
        }
        if key == KEY_SHIFT {
            self.core.inference.shift_lock = None;
            self.release_shift_plane_mark();
            self.refresh_snap();
        }
    }

    /// `_is_chord`: Ctrl or Alt (or Meta) held, but not Ctrl and Alt together
    /// (AltGr, which types).
    fn is_chord(mods: u32) -> bool {
        let ctrl = mods & MOD_CONTROL != 0;
        let alt = mods & MOD_ALT != 0;
        (ctrl || alt || mods & MOD_META != 0) && !(ctrl && alt)
    }

    /// `_handle_value_key`: buffer a digit / separator / unit letter; Enter
    /// applies the buffer through the tool.
    fn handle_value_key(&mut self, key: i32, mods: u32, text: &str) -> bool {
        if key == KEY_RETURN || key == KEY_ENTER {
            if self.core.value_buffer.is_empty() {
                return false;
            }
            // A tool with a grammar of its own reads the buffer itself.
            let buffer = self.core.value_buffer.clone();
            let before = self.core.undo_len();
            if self.with_tool(|t, c| t.on_buffer(c, &buffer)) {
                let drew = self.core.undo_len() > before;
                self.core.set_value_buffer("");
                self.release_axis_lock_after_operation(drew);
                return true;
            }
            let comma_lists = self.tool().vcb_comma_lists();
            let value = parse_value_buffer(&self.core.value_buffer, comma_lists);
            let Some(value) = value else {
                self.core.set_value_buffer("");
                return true;
            };
            match value {
                VcbValue::Other => {
                    self.core.set_value_buffer("");
                    return true;
                }
                VcbValue::Segments(n) => {
                    self.with_tool(|t, c| t.on_segments_value(c, n));
                    self.core.set_value_buffer("");
                    return true;
                }
                VcbValue::Radius(r) => {
                    self.with_tool(|t, c| t.on_radius_value(c, r));
                    self.core.set_value_buffer("");
                    return true;
                }
                _ => {}
            }
            let before = self.core.undo_len();
            self.with_tool(|t, c| t.on_value(c, &value));
            let drew = self.core.undo_len() > before;
            self.core.set_value_buffer("");
            self.release_axis_lock_after_operation(drew);
            return true;
        }
        if key == KEY_BACKSPACE {
            if self.core.value_buffer.is_empty() {
                return false;
            }
            let mut b = self.core.value_buffer.clone();
            b.pop();
            self.core.set_value_buffer(&b);
            return true;
        }
        if Self::is_chord(mods) {
            return false;
        }
        let text = crate::vcb::plain_marks(text);
        // Arrays after a copy: `3x`, `*3`, `/3` (only a tool that takes them).
        if self.tool().accepts_array() {
            if text == "*" {
                let b = format!("{}{}", self.core.value_buffer, text);
                self.core.set_value_buffer(&b);
                return true;
            }
            if text.eq_ignore_ascii_case("x") {
                let pending = self.core.value_buffer.is_empty() && self.with_tool(|t, c| t.array_pending(c));
                if !self.core.token_tail().is_empty() || pending {
                    let b = format!("{}{}", self.core.value_buffer, text);
                    self.core.set_value_buffer(&b);
                    return true;
                }
            }
            if text == "/" && self.core.value_buffer.is_empty() {
                self.core.set_value_buffer(&text);
                return true;
            }
        }
        let letters = ['m', 'c', 'r', 'i', 'n', 'f', 't', 's'];
        let single = text.chars().count() == 1;
        let ch = text.chars().next();
        let is_digit = single && ch.is_some_and(|c| c.is_ascii_digit());
        let is_sep = single && ch.is_some_and(|c| matches!(c, '.' | ',' | ';' | ' ' | '-' | ':' | '"' | '\'' | '/'));
        let is_letter = single && ch.is_some_and(|c| letters.contains(&c.to_ascii_lowercase()));
        if !text.is_empty() && (is_digit || is_sep || is_letter) {
            let ch = ch.unwrap();
            let mut text = text;
            if (ch == ';' || ch == ' ') && self.core.value_buffer.is_empty() {
                return false;
            }
            let tail = self.core.token_tail();
            if is_letter && tail.is_empty() {
                return false;
            }
            if matches!(ch, '"' | '\'' | '/') && tail.is_empty() {
                return true;
            }
            if ch == '-' && !tail.is_empty() {
                return true;
            }
            if ch == ':' && (tail.is_empty() || self.core.value_buffer.contains(':')) {
                return true;
            }
            if ch == '.' || ch == ',' {
                let comma_fields = self.tool().vcb_comma_lists() && !self.core.value_buffer.contains(';') && !self.core.value_buffer.contains(' ');
                let mut tail = tail;
                if comma_fields {
                    tail = tail.rsplit(',').next().unwrap_or("").to_string();
                }
                if tail.contains('.') || tail.contains(',') {
                    if ch == '.' {
                        return true;
                    }
                    if !comma_fields {
                        text = ";".to_string();
                    }
                }
            }
            let b = format!("{}{}", self.core.value_buffer, text);
            self.core.set_value_buffer(&b);
            return true;
        }
        false
    }

    // ---- undo -----------------------------------------------------------------------------

    /// `MainWindow._on_undo`: the tool answers first.
    pub fn undo(&mut self) -> bool {
        if self.with_tool(|t, c| t.on_undo(c)) {
            self.core.clear_references();
            self.core.invalidate();
            return true;
        }
        if self.core.history_undo() {
            self.core.clear_references();
            return true;
        }
        false
    }

    /// `MainWindow._on_redo`.
    pub fn redo(&mut self) -> bool {
        if self.with_tool(|t, c| t.on_redo(c)) {
            self.core.clear_references();
            self.core.invalidate();
            return true;
        }
        if self.core.history_redo() {
            self.core.clear_references();
            return true;
        }
        false
    }

    // ---- what the screen would show -----------------------------------------------------------

    /// `_measurement_text`: the box under the viewport.
    pub fn measurement_text(&self) -> String {
        if let Some((text, _)) = self.tool().value_label(&self.core) {
            return text;
        }
        let segs = self.tool().rubber_band_lines(&self.core);
        if segs.len() == 1 {
            let (a, b) = segs[0];
            return self.core.fmt_len(b.sub(a).length(self.core.h()) as f64);
        }
        String::new()
    }
}
