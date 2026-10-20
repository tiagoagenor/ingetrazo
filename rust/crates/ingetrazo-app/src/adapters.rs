// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! The headless tools behind the window's one host (M4.5, M4.6).
//!
//! The drawing tools already are `DrawTool`s the `Host` drives. The modify and
//! measure tools (`ModifyTool` on an `Env`) and Push/Pull (its own `Env`) were
//! ported against a `Viewport` of their own; here each is wrapped in a
//! `DrawTool`, so that the pointer, the snap, the keys, the value box and the
//! undo of the host reach them exactly as they reach a drawing tool, the way
//! Python's viewport hands every tool the same events. The Select tool, which
//! has no state machine of its own, is written here on the picks of
//! `ingetrazo-snap` (`tools/select.py`).

use ingetrazo_geom::hlr_draw::OrbitCamera as PickOrbit;
use ingetrazo_math::V3;
use ingetrazo_model::mesh::FaceId;
use ingetrazo_model::scene::Scene;
use ingetrazo_snap::picking::{Camera as PickCamera, Picked, View as PickView};
use ingetrazo_snap::workplane::ToolState;
use ingetrazo_snap::Context;
use ingetrazo_tools::base::{Ctx, DrawTool, Field, Overlay};
use ingetrazo_tools::host::Core;
use ingetrazo_tools::modify_base::{Env, FaceRef, ModifyTool, Mods, Probe, Sel, SnapInfo, ToolCx, Vcb};
use ingetrazo_tools::modify_viewport::PickViewport;
use ingetrazo_tools::pick::{FixedPick, Pick, Picker, RayPicker};
use ingetrazo_tools::pushpull::{Env as PushEnv, HoverEdge, Pointer, PushPullTool};
use ingetrazo_tools::vcb::{parse_value_buffer, plain_marks, VcbValue};
use ingetrazo_tools::view::{Camera as ToolCamera, View as ToolView};

/// The snap engine's view as the picks read it.
pub fn pick_view(v: &ingetrazo_snap::View) -> PickView {
    let c = v.cam;
    let mut cam = PickCamera::new(PickOrbit { target: c.target, distance: c.distance, yaw: c.yaw, pitch: c.pitch, up: c.up });
    cam.perspective = c.perspective;
    cam.fov_deg = c.fov_deg;
    cam.aspect = c.aspect;
    cam.znear = c.znear;
    cam.zfar = c.zfar;
    PickView::new(cam, v.width, v.height, v.h)
}

/// The snap engine's view as Push/Pull reads it.
fn tool_view(v: &ingetrazo_snap::View) -> ToolView {
    let c = v.cam;
    let cam = ToolCamera {
        target: c.target,
        distance: c.distance,
        yaw: c.yaw,
        pitch: c.pitch,
        up: c.up,
        fov_deg: c.fov_deg,
        aspect: c.aspect,
        znear: c.znear,
        zfar: c.zfar,
        perspective: c.perspective,
    };
    ToolView::new(cam, v.width.max(1) as u32, v.height.max(1) as u32, v.h)
}

/// A number that changes whenever a mesh of the scene does (the meshes'
/// serials, which every mutation bumps): a tool's step that moved geometry
/// without a command shows here.
pub fn scene_stamp(scene: &Scene) -> u64 {
    scene.meshes.iter().fold(scene.meshes.len() as u64, |a, m| a.wrapping_mul(1_000_003).wrapping_add(m.serial))
}

/// Text sizes for a label pick: there is no font here yet, so a label is
/// measured as a fixed-pitch 9 pt face.
pub struct ApproxText;

impl ingetrazo_snap::picking::TextMetrics for ApproxText {
    fn height(&self) -> i32 {
        15
    }
    fn ascent(&self) -> i32 {
        12
    }
    fn descent(&self) -> i32 {
        3
    }
    fn advance(&self, line: &str) -> i32 {
        7 * line.chars().count() as i32
    }
}

/// A step of a tool that edits the scene on its own: the host's caches must
/// notice.
fn after_step(core: &mut Core, before: u64, serial: Option<u64>) {
    if scene_stamp(&core.scene) != before || core.history.top_serial() != serial {
        core.version += 1;
        core.touch();
    }
    core.reconcile_selection();
}

// ---- the modify and measure tools ---------------------------------------------------------------

/// Which modify or measure tool an adapter wraps: what the host asks of it
/// that the tool does not say itself (the class attributes of the Python
/// tool).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ModifyKind {
    Move,
    Rotate,
    Scale,
    Offset,
    Eraser,
    Paint,
    Tape,
    Protractor,
    Flip,
}

impl ModifyKind {
    pub fn name(self) -> &'static str {
        match self {
            ModifyKind::Move => "Move",
            ModifyKind::Rotate => "Rotate",
            ModifyKind::Scale => "Scale",
            ModifyKind::Offset => "Offset",
            ModifyKind::Eraser => "Eraser",
            ModifyKind::Paint => "Paint",
            ModifyKind::Tape => "Tape Measure",
            ModifyKind::Protractor => "Protractor",
            ModifyKind::Flip => "Flip",
        }
    }

    /// `vcb_label`: the box's caption (none for a tool without a box).
    fn caption(self) -> &'static str {
        match self {
            ModifyKind::Move | ModifyKind::Tape => "Distance",
            ModifyKind::Rotate | ModifyKind::Protractor => "Angle",
            ModifyKind::Scale => "Scale",
            ModifyKind::Offset => "Offset",
            ModifyKind::Eraser | ModifyKind::Paint | ModifyKind::Flip => "",
        }
    }

    /// `accepts_array`: `3x` and `/3` after a copy.
    fn accepts_array(self) -> bool {
        matches!(self, ModifyKind::Move | ModifyKind::Rotate)
    }

    /// `accepts_angle_ratio`: `3:12` arrives as degrees.
    fn accepts_angle_ratio(self) -> bool {
        matches!(self, ModifyKind::Rotate | ModifyKind::Protractor)
    }

    /// The engine's view of the tool (`magnetic_axis_deg`, `screen_axis_px`,
    /// `prefers_vertical_drag`, `radial_arm`, `axis_source`).
    fn snap_profile(self, state: &mut ToolState) {
        match self {
            ModifyKind::Move => {
                state.prefers_vertical_drag = true;
                state.magnetic_axis_deg = Some(15.0);
                state.screen_axis_px = Some(9.0);
            }
            ModifyKind::Rotate | ModifyKind::Protractor => {
                state.magnetic_axis_deg = Some(3.0);
                state.radial_arm = true;
            }
            ModifyKind::Tape => {
                state.magnetic_axis_deg = Some(3.0);
                state.screen_axis_px = Some(9.0);
                state.axis_source = true;
            }
            ModifyKind::Scale | ModifyKind::Offset | ModifyKind::Eraser | ModifyKind::Paint | ModifyKind::Flip => {}
        }
    }
}

/// A `ModifyTool` as a `DrawTool`.
pub struct ModifyAdapter {
    kind: ModifyKind,
    tool: Box<dyn ModifyTool>,
    vp: PickViewport,
    /// What the tool said about itself after its last step (`probe`).
    probe: Probe,
    hover: Option<Sel>,
}

impl ModifyAdapter {
    pub fn new(kind: ModifyKind, tool: Box<dyn ModifyTool>) -> ModifyAdapter {
        let view = pick_view(&ingetrazo_snap::View::new(Default::default(), 1, 1, Default::default()));
        let mut vp = PickViewport::new(view, Default::default());
        vp.use_cache = true;
        vp.metrics = Box::new(ApproxText);
        ModifyAdapter { kind, tool, vp, probe: Probe::default(), hover: None }
    }

    /// Run `f` on the tool with the environment of the host.
    fn step<R>(&mut self, core: &mut Core, f: impl FnOnce(&mut dyn ModifyTool, &mut Env) -> R) -> R {
        self.vp.view = pick_view(&core.view);
        self.vp.hypot = core.view.h;
        self.vp.hint = core.version;
        self.vp.ctrl_alone = core.ctrl_alone;
        self.vp.threshold = core.settings.pick_threshold_px;
        let (stamp, serial) = (scene_stamp(&core.scene), core.history.top_serial());
        let mut version = core.version;
        let (r, probe) = {
            let mut env = Env {
                scene: &mut core.scene,
                cx: &mut core.cx,
                history: &mut core.history,
                sel: &mut core.selection,
                vp: &mut self.vp,
                units: &core.units,
                version: &mut version,
            };
            let r = f(self.tool.as_mut(), &mut env);
            let probe = self.tool.probe(&env);
            (r, probe)
        };
        self.probe = probe;
        core.version = version;
        after_step(core, stamp, serial);
        for (text, _msec) in self.vp.flashes.drain(..) {
            core.flash(text);
        }
        for g in self.vp.gfx.drain(..) {
            if let ingetrazo_tools::modify_base::Gfx::Hover(entity, _placement) = g {
                self.hover = entity;
            }
        }
        r
    }

    fn tool_cx(ctx: &Ctx) -> ToolCx {
        ToolCx {
            world: ctx.world,
            screen: ctx.screen,
            mods: Mods(ctx.mods),
            snap: Some(SnapInfo { kind: ctx.snap.kind, point: ctx.snap.point, axis: ctx.snap.axis, context: ctx.snap.context.map(|c| match c {
                Context::Group => Context::Group,
                Context::Component => Context::Component,
            }) }),
        }
    }
}

/// `Viewport._parse_value_buffer`'s arrays: `3x`, `x3`, `3*`, `/3`, `3/` and
/// `5x10m` (count, mode and the spacing).
fn parse_array(stripped: &str) -> Option<(i64, char, Option<f64>)> {
    let t = stripped.trim().to_lowercase();
    let digits = |s: &str| (!s.is_empty() && s.chars().all(|c| c.is_ascii_digit())).then(|| s.parse::<i64>().ok()).flatten();
    if let Some(rest) = t.strip_suffix(['x', '*']) {
        if let Some(n) = digits(rest.trim()) {
            return Some((n, 'x', None));
        }
    }
    if let Some(rest) = t.strip_prefix(['x', '*']) {
        if let Some(n) = digits(rest.trim()) {
            return Some((n, 'x', None));
        }
    }
    if let Some(i) = t.find(['x', '*']) {
        if let Some(n) = digits(t[..i].trim()) {
            let rest = &t[i + 1..];
            if !rest.trim().is_empty() {
                return match parse_value_buffer(rest, false) {
                    Some(VcbValue::Num(s)) if s > 0.0 => Some((n, 'x', Some(s))),
                    _ => None,
                };
            }
        }
    }
    if let Some(rest) = t.strip_prefix('/') {
        if let Some(n) = digits(rest.trim()) {
            return Some((n, '/', None));
        }
    }
    if let Some(rest) = t.strip_suffix('/') {
        if let Some(n) = digits(rest.trim()) {
            return Some((n, '/', None));
        }
    }
    None
}

/// Whether the buffer carries a unit (`2m`, `1'`, `3"`): Scale reads it as a
/// size, not a factor.
fn has_unit(buffer: &str) -> bool {
    let t = buffer.to_lowercase();
    let chars: Vec<char> = t.chars().collect();
    for (i, &c) in chars.iter().enumerate() {
        if !(c.is_ascii_digit() || c == '"' || c == '\'') {
            continue;
        }
        let mut j = i + 1;
        while j < chars.len() && chars[j].is_whitespace() {
            j += 1;
        }
        if c.is_ascii_digit() && j < chars.len() && (chars[j] == '"' || chars[j] == '\'') {
            return true;
        }
        let rest: String = chars[j..].iter().collect();
        for unit in ["mm", "cm", "in", "ft", "m"] {
            if let Some(after) = rest.strip_prefix(unit) {
                if !after.chars().next().is_some_and(|n| n.is_alphanumeric() || n == '_') {
                    return true;
                }
            }
        }
    }
    false
}

impl DrawTool for ModifyAdapter {
    fn name(&self) -> &'static str {
        self.kind.name()
    }
    fn uses_snap(&self) -> bool {
        self.probe.uses_snap
    }
    fn snap_state(&self, _core: &Core) -> ToolState {
        let mut state = ToolState { uses_snap: self.probe.uses_snap, start_point: self.probe.start, ..ToolState::default() };
        self.kind.snap_profile(&mut state);
        state
    }
    fn start_point(&self) -> Option<V3> {
        self.probe.start
    }
    fn work_plane(&self) -> Option<(V3, V3)> {
        None
    }
    fn set_work_plane(&mut self, _plane: Option<(V3, V3)>) {}
    fn busy(&self) -> bool {
        self.probe.start.is_some() || self.probe.rb.as_ref().is_some_and(|r| !r.is_empty())
    }
    fn vcb_label(&self) -> String {
        self.probe.cap.clone().unwrap_or_else(|| self.kind.caption().to_string())
    }
    fn accepts_array(&self) -> bool {
        self.kind.accepts_array()
    }
    fn array_pending(&self, _core: &Core) -> bool {
        self.probe.ap.unwrap_or(false)
    }
    fn hover(&self) -> Option<Sel> {
        self.hover.clone()
    }

    fn on_activate(&mut self, core: &mut Core) {
        self.step(core, |t, e| t.on_activate(e));
    }
    fn on_deactivate(&mut self, core: &mut Core) {
        self.step(core, |t, e| t.on_deactivate(e));
        self.hover = None;
    }
    fn on_click(&mut self, core: &mut Core, ctx: &Ctx) {
        let cx = Self::tool_cx(ctx);
        self.step(core, |t, e| t.on_click(e, &cx));
    }
    fn on_double_click(&mut self, core: &mut Core, ctx: &Ctx) {
        let cx = Self::tool_cx(ctx);
        self.step(core, |t, e| t.on_double_click(e, &cx));
    }
    fn on_hover(&mut self, core: &mut Core, ctx: &Ctx) {
        let cx = Self::tool_cx(ctx);
        self.step(core, |t, e| t.on_hover(e, &cx));
    }
    fn on_release(&mut self, core: &mut Core) {
        self.step(core, |t, e| t.on_release(e));
    }
    fn on_cancel(&mut self, core: &mut Core) {
        self.step(core, |t, e| t.on_cancel(e));
    }
    fn close_array_window(&mut self) {
        self.tool.close_array_window();
        self.probe.ap = Some(false);
    }
    fn on_key(&mut self, core: &mut Core, key: i32, mods: u32) -> bool {
        self.step(core, |t, e| t.on_key(e, key, Mods(mods)))
    }
    fn on_key_release(&mut self, core: &mut Core, key: i32) -> bool {
        self.step(core, |t, e| t.on_key_release(e, key))
    }

    /// The value box of the modify tools: a number, a tuple, a `3x` array, a
    /// `3:12` slope for the angle tools, and for Scale a size when it carries
    /// a unit (`Viewport._handle_value_key` and `_parse_value_buffer`).
    fn on_buffer(&mut self, core: &mut Core, buffer: &str) -> bool {
        let text = plain_marks(buffer.trim());
        let normalized = text.replace(',', ".").replace(';', " ");
        let stripped = normalized.trim();
        if let Some((n, mode, step)) = parse_array(stripped) {
            if self.kind.accepts_array() {
                self.step(core, |t, e| t.on_array_value(e, n, mode, step));
            }
            return true;
        }
        if stripped.contains(':') {
            // A slope as rise:run, read as the degrees it makes.
            let mut parts = stripped.splitn(2, ':');
            let rise = parts.next().and_then(|s| s.trim().parse::<f64>().ok());
            let run = parts.next().and_then(|s| s.trim().parse::<f64>().ok());
            if let (Some(rise), Some(run), true) = (rise, run, self.kind.accepts_angle_ratio()) {
                if run > 0.0 && rise.is_finite() && run.is_finite() {
                    let deg = rise.atan2(run).to_degrees();
                    self.step(core, |t, e| t.on_value(e, &Vcb::Num(deg)));
                }
            }
            return true;
        }
        let value = match parse_value_buffer(&text, false) {
            Some(VcbValue::Num(v)) => Vcb::Num(v),
            Some(VcbValue::Pair(a, b)) => Vcb::Tuple(vec![a, b]),
            Some(VcbValue::Triple(a, b, c)) => Vcb::Tuple(vec![a, b, c]),
            // A segment count, a radius, a profile: nothing here takes them.
            _ => return true,
        };
        let value = if self.kind == ModifyKind::Scale && has_unit(&text) { Vcb::AbsLen(Box::new(value)) } else { value };
        self.step(core, |t, e| t.on_value(e, &value));
        true
    }

    fn rubber_band_lines(&self, _core: &Core) -> Vec<(V3, V3)> {
        self.probe.rb.clone().unwrap_or_default()
    }
    fn value_label(&self, _core: &Core) -> Option<(String, V3)> {
        self.probe.vl.clone().flatten()
    }
    fn overlay(&self, _core: &Core) -> Overlay {
        Overlay {
            rubber: self.probe.rb.clone().unwrap_or_default(),
            guide: self.probe.gl.clone().unwrap_or_default(),
            label: self.probe.vl.clone().flatten(),
            wireframe: self.probe.wf,
            clause: self.probe.sc.clone().unwrap_or_default(),
            plus: self.probe.plus.unwrap_or(false),
            ..Overlay::default()
        }
    }
    fn fields(&self) -> Vec<(&'static str, Field)> {
        Vec::new()
    }
}

// ---- Push/Pull -------------------------------------------------------------------------------------

/// `PushPullTool` as a `DrawTool`: its pointer is the cursor's pixel and
/// Ctrl, its picks are the pick index of `ingetrazo-snap` while the tool is
/// idle (groups, hidden things and layers as Python's `pick_face_any` reads
/// them) and rays against the active mesh (`RayPicker`) while it drags, when
/// the model is its own live preview; the edge under the cursor is the one
/// the host acquired.
pub struct PushPullAdapter {
    tool: PushPullTool,
    vp: PickViewport,
}

impl PushPullAdapter {
    pub fn new() -> PushPullAdapter {
        let view = pick_view(&ingetrazo_snap::View::new(Default::default(), 1, 1, Default::default()));
        let mut vp = PickViewport::new(view, Default::default());
        vp.use_cache = true;
        PushPullAdapter { tool: PushPullTool::new(), vp }
    }

    /// `viewport.pick_face_any(x, y)` as the idle tool asks it: a face of the
    /// open mesh, or one inside a closed group (which the tool refuses).
    fn pick_idle(&mut self, core: &Core, at: (f64, f64)) -> Option<Pick> {
        self.vp.view = pick_view(&core.view);
        self.vp.hypot = core.view.h;
        self.vp.hint = core.version;
        self.vp.threshold = core.settings.pick_threshold_px;
        let hit = ingetrazo_tools::modify_base::Viewport::pick_face_any(&self.vp, &core.scene, &core.cx, at.0, at.1)?;
        Some(match hit.face {
            FaceRef::Loose(face) => Pick { face, closed_group: false },
            // A face of a group's mesh: no face of this one, only the refusal.
            FaceRef::Placed { .. } => Pick { face: FaceId(u32::MAX), closed_group: true },
        })
    }

    fn step<R>(&mut self, core: &mut Core, at: Option<(f64, f64)>, f: impl FnOnce(&mut PushPullTool, &mut PushEnv) -> R) -> R {
        let view = tool_view(&core.view);
        let hover_edge = core.inference.hover_edge.map(|e| HoverEdge { a: e.a, b: e.b, edge: core.hover_loose_edge });
        let (stamp, serial) = (scene_stamp(&core.scene), core.history.top_serial());
        let idle_pick = match at {
            Some(at) if !self.tool.dragging => Some(FixedPick(self.pick_idle(core, at))),
            _ => None,
        };
        let ray = RayPicker;
        let picker: &dyn Picker = match &idle_pick {
            Some(p) => p,
            None => &ray,
        };
        let r = {
            let mut env = PushEnv {
                scene: &mut core.scene,
                cx: &mut core.cx,
                history: &mut core.history,
                view: Some(&view),
                picker,
                hover_edge,
                snap_threshold_px: core.settings.snap_threshold_px,
                hidden_reference: None,
                units: &core.units,
            };
            f(&mut self.tool, &mut env)
        };
        after_step(core, stamp, serial);
        for text in self.tool.flashes.drain(..) {
            core.flash(text);
        }
        r
    }

    fn pointer(ctx: &Ctx) -> Pointer {
        Pointer { x: ctx.screen.0, y: ctx.screen.1, ctrl: ctx.mods & ingetrazo_tools::base::MOD_CONTROL != 0 }
    }
}

impl Default for PushPullAdapter {
    fn default() -> Self {
        PushPullAdapter::new()
    }
}

impl DrawTool for PushPullAdapter {
    fn name(&self) -> &'static str {
        "Push / Pull"
    }
    fn uses_snap(&self) -> bool {
        false
    }
    fn snap_state(&self, _core: &Core) -> ToolState {
        ToolState { uses_snap: false, hover_group_edges: true, start_point: self.start_point(), ..ToolState::default() }
    }
    fn start_point(&self) -> Option<V3> {
        if self.tool.dragging {
            self.tool.anchor()
        } else {
            None
        }
    }
    fn work_plane(&self) -> Option<(V3, V3)> {
        None
    }
    fn set_work_plane(&mut self, _plane: Option<(V3, V3)>) {}
    fn vcb_label(&self) -> String {
        "Distance".into()
    }
    fn hover(&self) -> Option<Sel> {
        self.tool.hover.map(Sel::Face)
    }

    fn on_activate(&mut self, core: &mut Core) {
        // Armed with exactly one loose face selected.
        let faces: Vec<FaceId> = core.selection.faces();
        let selected = if core.selection.len() == 1 && faces.len() == 1 { Some(faces[0]) } else { None };
        self.step(core, None, |t, e| t.on_activate(e, selected));
    }
    fn on_deactivate(&mut self, core: &mut Core) {
        self.step(core, None, |t, e| t.on_deactivate(e));
    }
    fn on_click(&mut self, core: &mut Core, ctx: &Ctx) {
        let ptr = Self::pointer(ctx);
        self.step(core, Some((ptr.x, ptr.y)), |t, e| t.on_click(e, ptr));
    }
    fn on_double_click(&mut self, core: &mut Core, ctx: &Ctx) {
        let ptr = Self::pointer(ctx);
        self.step(core, Some((ptr.x, ptr.y)), |t, e| t.on_double_click(e, ptr));
    }
    fn on_hover(&mut self, core: &mut Core, ctx: &Ctx) {
        let ptr = Self::pointer(ctx);
        self.step(core, Some((ptr.x, ptr.y)), |t, e| t.on_hover(e, ptr));
    }
    fn on_release(&mut self, core: &mut Core) {
        // Ctrl read live, as `QApplication.queryKeyboardModifiers` does.
        let ctrl = core.kb_mods & ingetrazo_tools::base::MOD_CONTROL != 0;
        self.step(core, None, |t, e| t.on_release(e, ctrl));
    }
    fn on_cancel(&mut self, core: &mut Core) {
        self.step(core, None, |t, e| t.on_cancel(e));
    }
    fn on_undo(&mut self, core: &mut Core) -> bool {
        self.step(core, None, |t, e| t.on_undo(e))
    }
    fn on_redo(&mut self, core: &mut Core) -> bool {
        self.step(core, None, |t, e| t.on_redo(e))
    }
    fn on_key(&mut self, _core: &mut Core, key: i32, _mods: u32) -> bool {
        // The Ctrl press is swallowed: the tap acts on the release.
        key == ingetrazo_tools::base::KEY_CONTROL && self.tool.on_key_ctrl_press()
    }
    fn on_key_release(&mut self, core: &mut Core, key: i32) -> bool {
        if key != ingetrazo_tools::base::KEY_CONTROL {
            return false;
        }
        let tapped = core.ctrl_alone;
        let r = self.tool.on_key_ctrl_release(tapped);
        for text in self.tool.flashes.drain(..) {
            core.flash(text);
        }
        r
    }
    fn on_value(&mut self, core: &mut Core, value: &VcbValue) -> bool {
        let VcbValue::Num(v) = value else { return false };
        let v = *v;
        self.step(core, None, |t, e| t.on_value(e, v))
    }

    fn rubber_band_lines(&self, core: &Core) -> Vec<(V3, V3)> {
        self.tool.rubber_band_lines(core.h())
    }
    fn value_label(&self, core: &Core) -> Option<(String, V3)> {
        self.tool.value_label(&core.units)
    }
    fn overlay(&self, core: &Core) -> Overlay {
        let (on, a, b) = self.tool.status_clause();
        Overlay {
            rubber: self.tool.rubber_band_lines(core.h()),
            label: self.tool.value_label(&core.units),
            wireframe: Some([0.13, 0.17, 0.23, 1.0]),
            clause: if on { format!("Ctrl = [{a}] / {b}") } else { format!("Ctrl = {a} / [{b}]") },
            marker: self.tool.inference_marker(),
            inference_guides: self.tool.inference_guide_lines(),
            ..Overlay::default()
        }
    }
    fn fields(&self) -> Vec<(&'static str, Field)> {
        vec![("dragging", Field::Bool(self.tool.dragging))]
    }
}

// ---- Select ---------------------------------------------------------------------------------------------

/// `tools/select.py`: a click picks what is under the cursor (a group first,
/// then an edge, a face...) and joins it to the selection by the modifiers;
/// a box takes what it encloses (or touches, dragged right to left).
pub struct SelectAdapter {
    vp: PickViewport,
    hover: Option<Sel>,
}

/// `selection_mode(modifiers)`.
pub fn selection_mode(mods: u32) -> &'static str {
    let shift = mods & ingetrazo_tools::base::MOD_SHIFT != 0;
    let ctrl = mods & ingetrazo_tools::base::MOD_CONTROL != 0;
    match (shift, ctrl) {
        (true, true) => "remove",
        (_, true) => "add",
        (true, false) => "toggle",
        _ => "replace",
    }
}

impl SelectAdapter {
    pub fn new() -> SelectAdapter {
        let view = pick_view(&ingetrazo_snap::View::new(Default::default(), 1, 1, Default::default()));
        let mut vp = PickViewport::new(view, Default::default());
        vp.use_cache = true;
        vp.metrics = Box::new(ApproxText);
        SelectAdapter { vp, hover: None }
    }

    fn sync(&mut self, core: &Core) {
        self.vp.view = pick_view(&core.view);
        self.vp.hypot = core.view.h;
        self.vp.hint = core.version;
        self.vp.threshold = core.settings.pick_threshold_px;
    }

    /// Name a pick by identity.
    fn sel_of(core: &Core, p: &Picked) -> Sel {
        let mesh = core.mesh();
        match p {
            Picked::Face(i) => Sel::Face(mesh.faces[*i]),
            Picked::Edge(i) => Sel::Edge(mesh.edges[*i]),
            Picked::Group(path) => Sel::Group(path.clone()),
            Picked::Dimension(i) => Sel::Dimension(*i),
            Picked::Label(i) => Sel::Label(*i),
            Picked::GeoPath(i) => Sel::GeoPath(*i),
            Picked::Section(i) => Sel::Section(*i),
            Picked::Guide(i) => Sel::Guide(*i),
            Picked::Image(i) => Sel::Image(*i),
        }
    }

    fn pick(&mut self, core: &Core, x: f64, y: f64) -> Option<Sel> {
        self.sync(core);
        let p = self.vp.with_picker(&core.scene, &core.cx, |p| p.select_pick(x, y, &ApproxText))?;
        Some(Self::sel_of(core, &p))
    }

    /// `SelectTool._expand`: a face is its whole curved surface, an edge of a
    /// curve is the whole curve.
    fn expand(core: &Core, entity: Sel) -> Vec<Sel> {
        let mesh = core.mesh();
        match entity {
            Sel::Face(f) => mesh.surface_of(f).into_iter().map(Sel::Face).collect(),
            Sel::Edge(e) if mesh.edge(e).curve.is_some() => mesh.curve_edges(e).into_iter().map(Sel::Edge).collect(),
            other => vec![other],
        }
    }
}

impl Default for SelectAdapter {
    fn default() -> Self {
        SelectAdapter::new()
    }
}

impl DrawTool for SelectAdapter {
    fn name(&self) -> &'static str {
        "Select"
    }
    fn uses_snap(&self) -> bool {
        false
    }
    fn snap_state(&self, _core: &Core) -> ToolState {
        ToolState { uses_snap: false, ..ToolState::default() }
    }
    fn start_point(&self) -> Option<V3> {
        None
    }
    fn work_plane(&self) -> Option<(V3, V3)> {
        None
    }
    fn set_work_plane(&mut self, _plane: Option<(V3, V3)>) {}
    fn busy(&self) -> bool {
        false
    }
    fn vcb_label(&self) -> String {
        String::new()
    }
    fn box_select(&self) -> bool {
        true
    }
    fn hover(&self) -> Option<Sel> {
        self.hover.clone()
    }

    fn on_deactivate(&mut self, _core: &mut Core) {
        self.hover = None;
    }

    fn on_click(&mut self, core: &mut Core, ctx: &Ctx) {
        let mode = selection_mode(ctx.mods);
        let entity = self.pick(core, ctx.screen.0, ctx.screen.1);
        match entity {
            None => {
                if mode == "replace" && !core.selection.is_empty() {
                    core.selection.clear();
                    core.version += 1;
                }
            }
            Some(e) => {
                let picked = Self::expand(core, e);
                core.selection.select(&picked, mode);
                core.version += 1;
            }
        }
    }

    /// A face selects itself and the edges around it, an edge itself and its
    /// faces.
    fn on_double_click(&mut self, core: &mut Core, ctx: &Ctx) {
        let entity = self.pick(core, ctx.screen.0, ctx.screen.1);
        let Some(entity) = entity else {
            self.on_click(core, ctx);
            return;
        };
        if !matches!(entity, Sel::Face(_) | Sel::Edge(_)) {
            self.on_click(core, ctx);
            return;
        }
        let mut picked = Self::expand(core, entity.clone());
        let mesh = core.mesh();
        let mut more: Vec<Sel> = Vec::new();
        match entity {
            Sel::Face(_) => {
                for s in &picked {
                    let Sel::Face(f) = s else { continue };
                    let face = mesh.face(*f);
                    for lp in std::iter::once(&face.outer).chain(face.holes.iter()) {
                        let n = lp.len();
                        for i in 0..n {
                            if let Some(e) = mesh.find_edge(lp[i], lp[(i + 1) % n]) {
                                more.push(Sel::Edge(e));
                            }
                        }
                    }
                }
            }
            _ => {
                for s in &picked {
                    let Sel::Edge(e) = s else { continue };
                    more.extend(mesh.edge(*e).faces.iter().map(|f| Sel::Face(*f)));
                }
            }
        }
        picked.extend(more);
        core.selection.select(&picked, selection_mode(ctx.mods));
        core.version += 1;
    }

    fn on_hover(&mut self, core: &mut Core, ctx: &Ctx) {
        self.hover = self.pick(core, ctx.screen.0, ctx.screen.1);
    }

    fn on_cancel(&mut self, _core: &mut Core) {}

    fn on_box_select(&mut self, core: &mut Core, rect: [f64; 4], crossing: bool, mods: u32) {
        self.sync(core);
        let r = (rect[0], rect[1], rect[2], rect[3]);
        let picked = self.vp.with_picker(&core.scene, &core.cx, |p| p.box_select(r, crossing));
        let sels: Vec<Sel> = picked.iter().map(|p| Self::sel_of(core, p)).collect();
        core.selection.select(&sels, selection_mode(mods));
        core.version += 1;
    }

    fn rubber_band_lines(&self, _core: &Core) -> Vec<(V3, V3)> {
        Vec::new()
    }
    fn fields(&self) -> Vec<(&'static str, Field)> {
        Vec::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arrays_are_read_as_the_viewport_reads_them() {
        assert_eq!(parse_array("3x"), Some((3, 'x', None)));
        assert_eq!(parse_array("3*"), Some((3, 'x', None)));
        assert_eq!(parse_array("x5"), Some((5, 'x', None)));
        assert_eq!(parse_array("/4"), Some((4, '/', None)));
        assert_eq!(parse_array("4/"), Some((4, '/', None)));
        assert_eq!(parse_array("5x10m"), Some((5, 'x', Some(10.0))));
        assert_eq!(parse_array("5x-2"), None, "a spacing is positive");
        assert_eq!(parse_array("2"), None);
        assert_eq!(parse_array("2.5"), None);
    }

    #[test]
    fn a_unit_makes_a_size_of_a_factor() {
        assert!(has_unit("2m"));
        assert!(has_unit("30cm"));
        assert!(has_unit("1500 mm"));
        assert!(has_unit("3'"));
        assert!(has_unit("2\""));
        assert!(!has_unit("2"));
        assert!(!has_unit("1.5"));
        assert!(!has_unit("2;3"));
    }

    #[test]
    fn the_modifiers_pick_how_a_click_joins_the_selection() {
        use ingetrazo_tools::base::{MOD_CONTROL, MOD_SHIFT};
        assert_eq!(selection_mode(0), "replace");
        assert_eq!(selection_mode(MOD_SHIFT), "toggle");
        assert_eq!(selection_mode(MOD_CONTROL), "add");
        assert_eq!(selection_mode(MOD_SHIFT | MOD_CONTROL), "remove");
    }
}
