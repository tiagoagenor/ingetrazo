// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! The model on the screen, with no window (M4.5 to M4.8): the document, the
//! camera, the active tool and what the user did to them. Events come in as
//! [`Input`] - the pointer, the keys, the typed text, the same ones the
//! egui layer translates its own into - and the tools of `ingetrazo-tools`
//! answer through the one `Host` (the Python viewport: pointer, snap,
//! inference locks, value box, undo), so a test drives the application
//! exactly as a hand does.
//!
//! The session knows what egui does not: whether the document is dirty (the
//! history is not where the last save left it), what to tell the status bar
//! (the tool's hint, the value box), what to highlight, and how to save and
//! open the file.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use ingetrazo_math::{Hypot, V3};
use ingetrazo_model::scene::Scene;
use ingetrazo_model::units::ModelUnits;
use ingetrazo_ops::command::Cx;
use ingetrazo_render::options::{Hover as RenderHover, Selection as RenderSelection};
use ingetrazo_render::Camera;
use ingetrazo_snap::{OrbitCamera, SnapResult, View};
use ingetrazo_tools::base::{Overlay, KEY_ALT, KEY_CONTROL, KEY_SHIFT, MOD_ALT, MOD_CONTROL, MOD_SHIFT};
use ingetrazo_tools::host::Host;
use ingetrazo_tools::modify_base::Sel;

use crate::document::{self, Document};
use crate::nav::{self, Button, NavMode, NavSettings, Navigator};
use crate::probe::{PickCache, ScenePick};
use crate::toolbox::{self, ToolId};
use crate::viewport::Textures;

/// `Viewport.BOX_DRAG_THRESHOLD_PX`: a drag shorter than this is a click.
pub const BOX_DRAG_THRESHOLD_PX: f64 = 4.0;

/// A mouse button.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Btn {
    Left,
    Middle,
}

/// One thing that happened to the viewport: what the egui layer reports,
/// in the viewport's own pixels. `mods` are the modifier bits `ingetrazo_tools::base::MOD_*`
/// (Qt's), `key` a Qt key code (`KEY_*`).
#[derive(Clone, Debug, PartialEq)]
pub enum Input {
    Move { x: f64, y: f64, mods: u32 },
    Press { button: Btn, x: f64, y: f64, mods: u32 },
    Release { button: Btn, x: f64, y: f64, mods: u32 },
    DoubleClick { x: f64, y: f64, mods: u32 },
    /// The wheel: notches, positive away from the user.
    Wheel { x: f64, y: f64, notches: f64 },
    Key { key: i32, mods: u32, pressed: bool },
    /// Typed characters (digits for the value box, letters for the shortcuts).
    Text { text: String, mods: u32 },
}

/// A rubber-band box being dragged by the Select tool.
#[derive(Clone, Copy, Debug)]
pub struct BoxDrag {
    pub start: (f64, f64),
    pub cur: (f64, f64),
}

/// The state of the window's document and tools.
pub struct Session {
    pub host: Host,
    pub cam: Camera,
    pub nav: Navigator,
    pub path: Option<PathBuf>,
    /// The file the document was opened from (embedded textures).
    raw: Arc<Vec<u8>>,
    pub tool: ToolId,
    pub size: (i32, i32),
    /// Frame the model once the viewport has a size (a document just opened).
    pub fit_pending: bool,
    /// Build what the first pointer move would build (the snap world, the
    /// pick index) once the model is on the screen: [`Session::warm`].
    pub warm_pending: bool,
    pub box_drag: Option<BoxDrag>,
    saved_serial: Option<u64>,
    pick: PickCache,
    /// The camera and the size the host's view was last built for.
    view_key: Option<[f64; 12]>,
    /// What the pick cache and the renderer were last told.
    seen: (u64, u64),
    scene_changed: bool,
    buttons: u8,
    left_to_tool: bool,
    hypot: Hypot,
}

impl Default for Session {
    fn default() -> Self {
        Session::new()
    }
}

/// `max curve id + 1` and the largest uid the scene holds, so what is drawn
/// next does not collide with what was opened.
fn continue_counters(scene: &Scene, cx: &mut Cx) {
    let mut curve = 0i64;
    for m in &scene.meshes {
        for &e in &m.edges {
            if let Some(c) = m.edge(e).curve {
                curve = curve.max(c);
            }
        }
    }
    cx.counters.curve = cx.counters.curve.max(curve + 1);
    fn uids(groups: &[ingetrazo_model::scene::Group], max: &mut u64) {
        for g in groups {
            if let Ok(n) = u64::from_str_radix(&g.uid, 16) {
                *max = (*max).max(n);
            }
            uids(&g.children, max);
        }
    }
    let mut max = 0u64;
    uids(&scene.groups, &mut max);
    uids(&scene.kept_components, &mut max);
    cx.counters.uid = cx.counters.uid.max(max + 1);
}

impl Session {
    /// A new, empty model with the Select tool.
    pub fn new() -> Session {
        let hypot = Hypot::default();
        let mut s = Session {
            host: Host::new(Scene::new(), Cx { hypot, ..Cx::default() }, View::new(OrbitCamera::default(), 1, 1, hypot)),
            cam: Camera::default(),
            nav: Navigator::new(NavSettings::default()),
            path: None,
            raw: Arc::new(Vec::new()),
            tool: ToolId::Select,
            size: (1, 1),
            fit_pending: false,
            warm_pending: false,
            box_drag: None,
            saved_serial: None,
            pick: PickCache::default(),
            view_key: None,
            seen: (0, 0),
            scene_changed: true,
            buttons: 0,
            left_to_tool: false,
            hypot,
        };
        s.host.set_active_tool(toolbox::make(ToolId::Select));
        s
    }

    // ---- the document ---------------------------------------------------------------------------

    /// Start over with an empty model (File > New); the camera stays.
    pub fn new_document(&mut self) {
        let (cam, size) = (self.cam, self.size);
        let nav = std::mem::take(&mut self.nav);
        *self = Session::new();
        self.cam = cam;
        self.nav = nav;
        self.nav.end_drag();
        self.set_size(size.0, size.1);
    }

    /// Replace the model with an opened document.
    pub fn set_document(&mut self, doc: Document) {
        let scene = Arc::try_unwrap(doc.scene).unwrap_or_else(|a| (*a).clone());
        let units = ModelUnits::of(scene.extra.units.as_ref());
        let mut cx = Cx { hypot: self.hypot, ..Cx::default() };
        continue_counters(&scene, &mut cx);
        let view = self.host_view();
        self.host = Host::new(scene, cx, view);
        self.host.core.units = units;
        self.host.set_active_tool(toolbox::make(ToolId::Select));
        self.tool = ToolId::Select;
        self.path = doc.path;
        self.raw = doc.raw;
        self.saved_serial = None;
        self.cam = Camera::default();
        self.set_size(self.size.0, self.size.1);
        self.nav.end_drag();
        self.box_drag = None;
        self.fit_pending = true;
        self.warm_pending = true;
        self.view_key = None;
        self.scene_changed = true;
        self.seen = (0, 0);
        self.pick.reset();
    }

    /// Open a file: the textures for the renderer come back.
    pub fn open(&mut self, path: &Path) -> Result<Textures, String> {
        let (doc, textures) = document::open_path(path)?;
        self.set_document(doc);
        Ok(textures)
    }

    pub fn scene(&self) -> &Scene {
        &self.host.core.scene
    }

    /// Whether the model changed since the last call: the renderer needs it.
    pub fn take_scene_changed(&mut self) -> bool {
        self.sync();
        std::mem::replace(&mut self.scene_changed, false)
    }

    /// The history is not where the last save left it.
    pub fn dirty(&self) -> bool {
        self.host.core.history.top_serial() != self.saved_serial
    }

    pub fn title(&self) -> String {
        document::window_title(self.path.as_deref(), self.dirty())
    }

    /// `File > Save` to `path`: the bytes and what the file embeds.
    pub fn save_to(&mut self, path: &Path) -> Result<document::Written, String> {
        // The camera is part of the file (`scene.camera_home`).
        if let Some(block) = document::camera_block(&self.cam) {
            self.host.core.scene.extra.camera = Some(block);
        }
        let written = document::write_bytes(&mut self.host.core.scene, &self.raw)?;
        document::write_file(path, &written.bytes)?;
        // The new file is what a later save packs the textures from.
        if !written.bytes.is_empty() {
            self.raw = Arc::new(written.bytes.clone());
        }
        self.path = Some(path.to_path_buf());
        self.saved_serial = self.host.core.history.top_serial();
        Ok(written)
    }

    // ---- the camera ------------------------------------------------------------------------------

    /// The viewport is `w` x `h` pixels.
    pub fn set_size(&mut self, w: i32, h: i32) {
        self.size = (w.max(1), h.max(1));
        self.cam.set_aspect(self.size.0 as u32, self.size.1 as u32);
    }

    fn host_view(&self) -> View {
        let c = &self.cam;
        let cam = OrbitCamera {
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
        View::new(cam, self.size.0, self.size.1, self.hypot)
    }

    /// Bring the host up to date with the camera and the model before an
    /// event reaches a tool.
    pub fn sync(&mut self) {
        let c = &self.cam;
        let key = [
            c.target.0 as f64,
            c.target.1 as f64,
            c.target.2 as f64,
            c.distance,
            c.yaw,
            c.pitch,
            c.fov_deg,
            c.aspect,
            f64::from(c.perspective),
            self.size.0 as f64,
            self.size.1 as f64,
            f64::from(c.two_point),
        ];
        if self.view_key != Some(key) {
            self.view_key = Some(key);
            self.host.core.view = self.host_view();
        }
        let now = (self.host.core.version, crate::adapters::scene_stamp(&self.host.core.scene));
        if now != self.seen {
            self.seen = now;
            self.pick.invalidate();
            self.scene_changed = true;
        }
    }

    /// Build what the first pointer move over a big model would build (the
    /// snap world and the Select tool's pick index: some hundreds of
    /// milliseconds), so the user's first move is not the one that stalls.
    /// A move to a pixel outside the viewport touches nothing visible.
    pub fn warm(&mut self) {
        self.warm_pending = false;
        self.sync();
        let last = self.host.core.last_mouse_pos;
        self.host.hover(-1.0, -1.0, 0);
        self.host.core.last_mouse_pos = last;
        self.sync();
    }

    /// `MainWindow._on_zoom_extents`.
    pub fn zoom_extents(&mut self) {
        self.sync();
        let mut probe = ScenePick::new(&self.host.core.scene, &mut self.pick, self.cam.hypot);
        if let Some((lo, hi)) = probe.bounds() {
            nav::fit_box(&mut self.cam, lo, hi, nav::FIT_MARGIN);
        }
    }

    // ---- tools --------------------------------------------------------------------------------------

    /// `MainWindow._activate_tool`: picking the tool the host already holds
    /// starts it over.
    pub fn set_tool(&mut self, id: ToolId) {
        self.sync();
        self.nav.set_nav_mode(None);
        self.box_drag = None;
        if id == self.tool {
            self.host.reselect_tool();
        } else {
            self.host.set_active_tool(toolbox::make(id));
            self.tool = id;
        }
        self.sync();
    }

    pub fn undo(&mut self) -> bool {
        self.sync();
        let done = self.host.undo();
        self.host.core.reconcile_selection();
        self.sync();
        done
    }

    pub fn redo(&mut self) -> bool {
        self.sync();
        let done = self.host.redo();
        self.host.core.reconcile_selection();
        self.sync();
        done
    }

    /// Whether Edit > Undo has something to do (a tool in the middle of
    /// something answers Ctrl+Z first).
    pub fn can_undo(&self) -> bool {
        self.host.core.history.undo_len() > 0 || self.host.tool().busy()
    }

    pub fn can_redo(&self) -> bool {
        self.host.core.history.redo_len() > 0
    }

    // ---- events ---------------------------------------------------------------------------------------

    /// One event of the viewport. Returns whether it was taken (a key no tool
    /// and no shortcut wanted is not).
    pub fn handle(&mut self, input: &Input) -> bool {
        self.sync();
        let taken = self.dispatch(input);
        self.sync();
        taken
    }

    fn shift(mods: u32) -> bool {
        mods & MOD_SHIFT != 0
    }

    fn dispatch(&mut self, input: &Input) -> bool {
        match *input {
            Input::Move { x, y, mods } => {
                if self.nav.dragging() {
                    let any = self.buttons > 0;
                    if self.nav.drag(&mut self.cam, self.size, x, y, Self::shift(mods), any) {
                        self.sync();
                    }
                    return true;
                }
                if let Some(b) = &mut self.box_drag {
                    b.cur = (x, y);
                    return true;
                }
                self.host.hover(x, y, mods);
                true
            }
            Input::Press { button, x, y, mods } => {
                self.buttons = self.buttons.saturating_add(1);
                match button {
                    Btn::Middle => {
                        let mut probe = ScenePick::new(&self.host.core.scene, &mut self.pick, self.cam.hypot);
                        self.nav.press(&self.cam, &mut probe, self.size, Button::Middle, x, y, Self::shift(mods));
                    }
                    Btn::Left => {
                        if self.nav.nav_mode.is_some() {
                            let mut probe = ScenePick::new(&self.host.core.scene, &mut self.pick, self.cam.hypot);
                            self.nav.press(&self.cam, &mut probe, self.size, Button::Left, x, y, Self::shift(mods));
                        } else if self.host.tool().box_select() {
                            self.box_drag = Some(BoxDrag { start: (x, y), cur: (x, y) });
                            self.left_to_tool = false;
                        } else {
                            self.left_to_tool = true;
                            self.host.press(x, y, mods);
                        }
                    }
                }
                true
            }
            Input::DoubleClick { x, y, mods } => {
                self.buttons = self.buttons.saturating_add(1);
                if self.nav.nav_mode.is_none() {
                    self.left_to_tool = true;
                    self.host.double_click(x, y, mods);
                }
                true
            }
            Input::Release { button, x, y, mods } => {
                self.buttons = self.buttons.saturating_sub(1);
                match button {
                    Btn::Middle => self.nav.release(Button::Middle),
                    Btn::Left => {
                        if self.nav.nav_mode.is_some() {
                            self.nav.release(Button::Left);
                        } else if let Some(b) = self.box_drag.take() {
                            let (dx, dy) = (x - b.start.0, y - b.start.1);
                            if (dx * dx + dy * dy).sqrt() < BOX_DRAG_THRESHOLD_PX {
                                // A click: pick the one thing under the cursor.
                                self.host.press(x, y, mods);
                            } else {
                                let rect = [b.start.0.min(x), b.start.1.min(y), b.start.0.max(x), b.start.1.max(y)];
                                self.host.box_select(rect, dx < 0.0, mods);
                            }
                        } else if self.left_to_tool {
                            self.left_to_tool = false;
                            self.host.release();
                        }
                    }
                }
                true
            }
            Input::Wheel { x, y, notches } => {
                let mut probe = ScenePick::new(&self.host.core.scene, &mut self.pick, self.cam.hypot);
                self.nav.wheel(&mut self.cam, &mut probe, self.size, x, y, notches * 120.0);
                true
            }
            Input::Key { key, mods, pressed } => {
                if pressed {
                    // Esc leaves a navigation mode before it is the tool's.
                    if key == ingetrazo_tools::base::KEY_ESCAPE && self.nav.nav_mode.is_some() {
                        self.nav.set_nav_mode(None);
                        return true;
                    }
                    self.host.key_press(key, mods, "")
                } else {
                    self.host.key_release(key);
                    true
                }
            }
            Input::Text { ref text, mods } => {
                if self.host.type_text(mods, text) {
                    return true;
                }
                self.shortcut(text)
            }
        }
    }

    /// The characters nobody typed into the value box are shortcuts.
    fn shortcut(&mut self, text: &str) -> bool {
        let mut any = false;
        for c in text.chars() {
            any |= match c {
                'o' => self.nav_mode(Some(NavMode::Orbit)),
                'h' => self.nav_mode(Some(NavMode::Pan)),
                'z' => self.nav_mode(Some(NavMode::Zoom)),
                'Z' => {
                    self.zoom_extents();
                    true
                }
                'P' => {
                    nav::toggle_projection(&mut self.cam);
                    true
                }
                c => match ToolId::from_shortcut(c) {
                    Some(id) => {
                        self.set_tool(id);
                        true
                    }
                    None => false,
                },
            };
        }
        any
    }

    pub fn nav_mode(&mut self, mode: Option<NavMode>) -> bool {
        self.nav.set_nav_mode(mode);
        true
    }

    // ---- what the window shows ---------------------------------------------------------------------------

    /// The tool's preview: rubber band, guides, label.
    pub fn overlay(&self) -> Overlay {
        self.host.tool().overlay(&self.host.core)
    }

    /// The snap the cursor is on (the marker and the tip).
    pub fn snap(&self) -> Option<&SnapResult> {
        self.host.core.last_snap.as_ref()
    }

    /// The Select rubber band, as `(start, current)` pixels.
    pub fn box_rect(&self) -> Option<BoxDrag> {
        self.box_drag.filter(|b| ((b.cur.0 - b.start.0).powi(2) + (b.cur.1 - b.start.1).powi(2)).sqrt() >= BOX_DRAG_THRESHOLD_PX)
    }

    /// The status bar's hint: the tool's name and what to do next.
    pub fn status_line(&self) -> String {
        let mut line = toolbox::status_line(self.tool, self.host.tool(), &self.host.core, self.nav.nav_mode);
        if let Some(a) = self.host.core.inference.axis_lock {
            line.push_str(&format!(" | axis lock: {}", a.name()));
        }
        line
    }

    /// The value box: its caption (`None` when the tool has none) and what
    /// it holds - what is being typed, or the live reading.
    pub fn vcb(&self) -> Option<(String, String, bool)> {
        let caption = self.host.tool().vcb_label();
        if caption.is_empty() {
            return None;
        }
        let typed = &self.host.core.value_buffer;
        if typed.is_empty() {
            Some((caption, self.host.measurement_text(), false))
        } else {
            Some((caption, typed.clone(), true))
        }
    }

    /// The texts the tools flashed since the last call (`flash_status`).
    pub fn take_flashes(&mut self) -> Vec<String> {
        std::mem::take(&mut self.host.core.flashes)
    }

    /// What the renderer highlights: the selection, and the one under the
    /// cursor.
    pub fn render_cues(&self) -> (RenderSelection, RenderHover) {
        let mut sel = RenderSelection::default();
        for s in &self.host.core.selection.items {
            match s {
                Sel::Face(f) => sel.faces.push((Vec::new(), *f)),
                Sel::Edge(e) => sel.edges.push((Vec::new(), *e)),
                Sel::Group(p) => sel.groups.push(p.clone()),
                _ => {}
            }
        }
        let mut hover = RenderHover::default();
        match self.host.tool().hover() {
            Some(Sel::Face(f)) => hover.face = Some((Vec::new(), f)),
            Some(Sel::Edge(e)) => hover.edge = Some((Vec::new(), e)),
            _ => {}
        }
        (sel, hover)
    }

    /// Where a world point is on the screen.
    pub fn to_pixel(&self, p: V3) -> Option<(f64, f64)> {
        self.host.core.view.world_to_pixel(p)
    }

    /// Whether the pointer is up (no button held).
    pub fn idle(&self) -> bool {
        self.buttons == 0
    }

    /// The modifier keys by themselves are keys of the host too (Shift locks
    /// the inference, Ctrl copies, Alt cycles the inference mode): the egui
    /// layer reports them through [`Input::Key`] with these codes.
    pub fn modifier_keys() -> [(i32, u32); 3] {
        [(KEY_SHIFT, MOD_SHIFT), (KEY_CONTROL, MOD_CONTROL), (KEY_ALT, MOD_ALT)]
    }
}
