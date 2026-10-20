// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! What the MODIFY and MEASURE tools (Move, Rotate, Scale, Flip, Offset,
//! Eraser, Paint, Tape Measure, Protractor) share: the vocabulary of
//! `tools/base.py` (`Tool`, `ToolContext`), the selection the Python scene
//! keeps (`scene.selection`), and the part of the viewport a tool talks to.
//!
//! A tool here is a state machine with no UI: the host hands it what the
//! viewport hands the Python one - a [`ToolCx`] (the snapped world point, the
//! cursor pixel, the modifiers, the snap) per click/hover, a key, a typed
//! [`Vcb`] value - and an [`Env`] (the scene, the history, the selection and
//! the [`Viewport`] it asks to pick and to flash). The tool answers by
//! executing `Command`s through the history, by editing the selection, and
//! through [`ModifyTool::probe`], what the Python tool offers the viewport to
//! draw and to word (rubber band, guide preview, label, status clause...).
//!
//! The snap engine runs in the host (`ingetrazo-snap`): `ToolCx::world` is
//! its answer, exactly as `ctx.world` is the viewport's in Python.
//!
//! `scene.version`: the Python scene counts every change, and a few tools
//! cache by it (the Scale grip box). [`Env::version`] is that counter; the
//! tools bump it where Python's tools do, and [`Env::execute`], [`Env::undo`]
//! and [`Env::redo`] bump it as a command does.

use ingetrazo_formats::pyjson::Value;
use ingetrazo_math::{M4, V3};
use ingetrazo_model::mesh::{EdgeId, FaceId};
use ingetrazo_model::scene::Scene;
use ingetrazo_model::units::ModelUnits;
use ingetrazo_ops::command::{Command, Cx, History};
use ingetrazo_ops::draw::{CompoundCommand, SnapshotCompound};
use ingetrazo_ops::group::{group_at, Path};
use ingetrazo_snap::{Axis, Context, SnapKind};

// ---- keys and modifiers (Qt's codes) ------------------------------------------------------

/// `Qt.Key_*` values the tools look at.
pub mod keys {
    pub const ESCAPE: i32 = 0x0100_0000;
    pub const RETURN: i32 = 0x0100_0004;
    pub const DELETE: i32 = 0x0100_0007;
    pub const LEFT: i32 = 0x0100_0012;
    pub const UP: i32 = 0x0100_0013;
    pub const RIGHT: i32 = 0x0100_0014;
    pub const DOWN: i32 = 0x0100_0015;
    pub const SHIFT: i32 = 0x0100_0020;
    pub const CONTROL: i32 = 0x0100_0021;
    pub const ALT: i32 = 0x0100_0023;
}

/// `Qt.KeyboardModifiers`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Mods(pub u32);

impl Mods {
    pub const SHIFT: u32 = 0x0200_0000;
    pub const CTRL: u32 = 0x0400_0000;
    pub const ALT: u32 = 0x0800_0000;

    pub fn shift(self) -> bool {
        self.0 & Mods::SHIFT != 0
    }
    pub fn ctrl(self) -> bool {
        self.0 & Mods::CTRL != 0
    }
    pub fn alt(self) -> bool {
        self.0 & Mods::ALT != 0
    }
}

// ---- what a click or a hover carries ----------------------------------------------------------

/// `SnapResult` as a tool reads it: the kind, the snapped point, the axis of
/// an axis inference and where the point lives.
#[derive(Clone, Debug, PartialEq)]
pub struct SnapInfo {
    pub kind: SnapKind,
    pub point: V3,
    pub axis: Option<Axis>,
    pub context: Option<Context>,
}

/// `ToolContext`: what the viewport hands a tool per click/hover.
#[derive(Clone, Debug, PartialEq)]
pub struct ToolCx {
    /// `ctx.world`: the snapped world point.
    pub world: V3,
    /// `ctx.screen`: the cursor pixel.
    pub screen: (f64, f64),
    pub mods: Mods,
    /// `ctx.snap`; `None` where the viewport had none to give.
    pub snap: Option<SnapInfo>,
}

impl ToolCx {
    /// The snap kind as a string, `"none"` without a snap (the way the
    /// tools read `ctx.snap.kind if ctx.snap is not None else "none"`).
    pub fn snap_kind(&self) -> SnapKind {
        self.snap.as_ref().map(|s| s.kind).unwrap_or(SnapKind::None)
    }
}

/// A typed value, as the viewport's VCB parser hands it to `on_value`: a
/// number, a tuple of numbers (`3;4;5`), or - for a tool that
/// `accepts_absolute_length` - the same tagged as a length with a unit
/// (`("abs_len", v)`, which to a tool that does not know it is a 2-tuple).
#[derive(Clone, Debug, PartialEq)]
pub enum Vcb {
    Num(f64),
    Tuple(Vec<f64>),
    AbsLen(Box<Vcb>),
}

// ---- the selection --------------------------------------------------------------------------------

/// One selected entity. Faces and edges by their stable ids, groups by
/// their path, annotations by their index in the scene's lists.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Sel {
    Face(FaceId),
    Edge(EdgeId),
    Group(Path),
    Dimension(usize),
    Label(usize),
    Guide(usize),
    Section(usize),
    Image(usize),
    GeoPath(usize),
}

/// `scene.selection`: in insertion order (Python's is a `set` of objects, an
/// order nothing can repeat; the fixtures pin it to insertion order).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Selection {
    pub items: Vec<Sel>,
}

impl Selection {
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    pub fn len(&self) -> usize {
        self.items.len()
    }

    pub fn iter(&self) -> std::slice::Iter<'_, Sel> {
        self.items.iter()
    }

    pub fn contains(&self, s: &Sel) -> bool {
        self.items.contains(s)
    }

    pub fn add(&mut self, s: Sel) {
        if !self.items.contains(&s) {
            self.items.push(s);
        }
    }

    pub fn discard(&mut self, s: &Sel) {
        self.items.retain(|x| x != s);
    }

    pub fn clear(&mut self) {
        self.items.clear();
    }

    /// `Scene.select(edges, mode=...)`: `"replace"`, `"add"`, `"toggle"` or
    /// `"remove"`.
    pub fn select(&mut self, ents: &[Sel], mode: &str) {
        if mode == "replace" {
            self.clear();
        }
        match mode {
            "toggle" => {
                for e in ents {
                    if self.contains(e) {
                        self.discard(e);
                    } else {
                        self.add(e.clone());
                    }
                }
            }
            "remove" => {
                for e in ents {
                    self.discard(e);
                }
            }
            _ => {
                for e in ents {
                    self.add(e.clone());
                }
            }
        }
    }

    pub fn faces(&self) -> Vec<FaceId> {
        self.items.iter().filter_map(|s| if let Sel::Face(f) = s { Some(*f) } else { None }).collect()
    }

    pub fn edges(&self) -> Vec<EdgeId> {
        self.items.iter().filter_map(|s| if let Sel::Edge(e) = s { Some(*e) } else { None }).collect()
    }

    pub fn groups(&self) -> Vec<Path> {
        self.items.iter().filter_map(|s| if let Sel::Group(p) = s { Some(p.clone()) } else { None }).collect()
    }
}

// ---- the viewport a tool talks to -------------------------------------------------------------------

/// A face found by `pick_face_any`/`pick_face_placement`: where it lives
/// and what a click on it selects.
#[derive(Clone, Debug, PartialEq)]
pub struct FaceHit {
    pub face: FaceRef,
    /// The group a click on it selects (`None` for a loose face).
    pub owner: Option<Path>,
    /// The placement it was hit through (`None` for a loose face).
    pub placement: Option<Path>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FaceRef {
    Loose(FaceId),
    /// A face of a placement's mesh, by its index in that mesh's `faces`.
    Placed { placement: Path, index: usize },
}

/// The edge under the cursor as `pick_edge_any` answers it: a loose edge, or
/// the world ends of a group's edge.
#[derive(Clone, Debug, PartialEq)]
pub enum EdgeAny {
    Loose(EdgeId),
    Group { a: V3, b: V3 },
}

/// A call into the viewport's group-preview hooks (the graphics the Python
/// viewport draws during a groups-only drag), recorded by the host.
#[derive(Clone, Debug, PartialEq)]
pub enum Gfx {
    Hover(Option<Sel>, Option<Path>),
    Begin(Vec<Path>, bool),
    Offset(V3),
    Matrix(M4),
    End,
}

/// The viewport, as a tool sees it: pixel/world maps, the picks (against the
/// scene as it is NOW, live previews included) and the things it asks the
/// viewport to show. `ingetrazo_tools::modify_viewport::PickViewport` is the
/// headless one; the UI's will implement it too.
pub trait Viewport {
    // -- the camera
    fn world_to_pixel(&self, p: V3) -> Option<(f64, f64)>;
    /// `_pixel_to_ray`: `(origin, unit direction)`.
    fn pixel_to_ray(&self, x: f64, y: f64) -> Option<(V3, V3)>;
    /// `camera.eye()`.
    fn eye(&self) -> V3;
    /// `viewport.pick_threshold_px`.
    fn pick_threshold_px(&self) -> f64;

    // -- the picks (`Viewport.pick_*`)
    fn pick_face(&self, scene: &Scene, cx: &Cx, x: f64, y: f64) -> Option<FaceId>;
    fn pick_face_any(&self, scene: &Scene, cx: &Cx, x: f64, y: f64) -> Option<FaceHit>;
    fn pick_edge(&self, scene: &Scene, cx: &Cx, x: f64, y: f64) -> Option<EdgeId>;
    fn pick_visible_edge(&self, scene: &Scene, cx: &Cx, x: f64, y: f64) -> Option<EdgeId>;
    /// `pick_edge_any`: a loose edge, or a group's as world ends.
    fn pick_edge_any(&self, scene: &Scene, cx: &Cx, x: f64, y: f64) -> Option<EdgeAny>;
    fn pick_group(&self, scene: &Scene, cx: &Cx, x: f64, y: f64) -> Option<Path>;
    fn pick_guide(&self, scene: &Scene, cx: &Cx, x: f64, y: f64) -> Option<usize>;
    fn pick_dimension(&self, scene: &Scene, cx: &Cx, x: f64, y: f64) -> Option<usize>;
    fn pick_text_label(&self, scene: &Scene, cx: &Cx, x: f64, y: f64, rect_only: bool) -> Option<usize>;
    fn pick_geopath(&self, scene: &Scene, cx: &Cx, x: f64, y: f64) -> Option<usize>;
    fn pick_section_plane(&self, scene: &Scene, cx: &Cx, x: f64, y: f64) -> Option<usize>;
    fn pick_image_plane(&self, scene: &Scene, cx: &Cx, x: f64, y: f64) -> Option<usize>;
    fn pick_axis(&self, scene: &Scene, cx: &Cx, x: f64, y: f64) -> Option<Axis>;

    // -- what a tool asks the viewport to do
    /// `flash_status(text, msec)`.
    fn flash_status(&mut self, text: &str, msec: u32);
    /// `set_hover(entity, placement)`.
    fn set_hover(&mut self, entity: Option<Sel>, placement: Option<Path>);
    /// The groups-only drag preview hooks.
    fn begin_groups_preview(&mut self, groups: &[Path], external: bool);
    fn set_groups_preview_offset(&mut self, delta: V3);
    fn set_groups_preview_matrix(&mut self, m: &M4);
    fn end_groups_preview(&mut self);
    /// `ctrl_tapped()`: whether the Ctrl being released was pressed alone.
    fn ctrl_tapped(&self) -> bool;
}

// ---- the environment of a tool step ----------------------------------------------------------------------

/// What a tool step needs of the host.
pub struct Env<'a> {
    pub scene: &'a mut Scene,
    /// `cx.mesh`: the mesh being edited (the loose one, or an open group's).
    pub cx: &'a mut Cx,
    pub history: &'a mut History,
    pub sel: &'a mut Selection,
    pub vp: &'a mut dyn Viewport,
    pub units: &'a ModelUnits,
    /// `scene.version`.
    pub version: &'a mut u64,
}

impl Env<'_> {
    /// `viewport.history.execute(cmd)`. A command that fails is rolled back
    /// and takes the selection with it, as in `History.execute`; one that
    /// inserts a group selects it (`InsertGroupCommand.do`).
    pub fn execute(&mut self, cmd: Box<dyn Command>) {
        let inserts = count_inserted(cmd.as_ref());
        let before = self.history.last_serial;
        self.history.execute(self.scene, self.cx, cmd);
        if self.history.last_error.is_some() {
            self.sel.clear();
        } else if inserts > 0 && self.history.last_serial != before {
            self.select_last_group();
        }
        self.reconcile_selection();
        *self.version += 1;
    }

    /// `execute`, answering the serial of the command if it landed on the
    /// undo stack (for a tool that keeps `undo_stack[-1] is cmd`).
    pub fn execute_serial(&mut self, cmd: Box<dyn Command>) -> Option<u64> {
        let before = self.history.last_serial;
        self.execute(cmd);
        if self.history.last_serial != before {
            self.history.last_serial
        } else {
            None
        }
    }

    /// `viewport.history.undo()`.
    pub fn undo(&mut self) -> bool {
        let done = self.history.undo(self.scene, self.cx);
        if done {
            self.reconcile_selection();
            *self.version += 1;
        }
        done
    }

    /// `viewport.history.redo()`.
    pub fn redo(&mut self) -> bool {
        let inserts = self.history.top_redo().map(count_inserted).unwrap_or(0);
        let done = self.history.redo(self.scene, self.cx);
        if done {
            if inserts > 0 {
                self.select_last_group();
            }
            self.reconcile_selection();
            *self.version += 1;
        }
        done
    }

    /// `InsertGroupCommand.do`: the group that arrives is the selection.
    fn select_last_group(&mut self) {
        if let Some(last) = self.scene.groups.len().checked_sub(1) {
            self.sel.clear();
            self.sel.add(Sel::Group(vec![last]));
        }
    }

    /// The commands `discard` what they remove from the selection: whatever
    /// no longer exists leaves it.
    pub fn reconcile_selection(&mut self) {
        let scene = &*self.scene;
        let mesh = self.cx.mesh(scene);
        self.sel.items.retain(|s| match s {
            Sel::Face(f) => mesh.faces.contains(f),
            Sel::Edge(e) => mesh.edges.contains(e),
            Sel::Group(p) => group_at(&scene.groups, p).is_some(),
            Sel::Dimension(i) => *i < scene.dimensions.len(),
            Sel::Label(i) => *i < scene.text_labels.len(),
            Sel::Guide(i) => *i < scene.guides.len(),
            Sel::Section(i) => *i < scene.section_planes.len(),
            Sel::Image(i) => *i < scene.image_planes.len(),
            Sel::GeoPath(i) => *i < scene.geo_paths.len(),
        });
    }

    /// `undo_stack[-1] is cmd`: whether the command of `serial` is the last
    /// one done.
    pub fn is_top(&self, serial: Option<u64>) -> bool {
        serial.is_some() && self.history.top_serial() == serial
    }

    /// `viewport.flash_status(text, msec)` (2500 ms is Python's default).
    pub fn flash(&mut self, text: impl AsRef<str>) {
        self.vp.flash_status(text.as_ref(), 2500);
    }

    pub fn flash_for(&mut self, text: impl AsRef<str>, msec: u32) {
        self.vp.flash_status(text.as_ref(), msec);
    }

    /// `scene.select(ents, mode)`: bumps the version like `bump_view`.
    pub fn select(&mut self, ents: &[Sel], mode: &str) {
        self.sel.select(ents, mode);
        *self.version += 1;
    }

    /// The group at `path`.
    pub fn group(&self, path: &[usize]) -> Option<&ingetrazo_model::scene::Group> {
        group_at(&self.scene.groups, path)
    }
}

/// How many `InsertGroupCommand`s a command holds (a compound's included): each
/// one selects the group it inserts.
pub fn count_inserted(cmd: &dyn Command) -> usize {
    let mut n = usize::from(cmd.name() == "insert_group");
    if let Some(any) = cmd.as_any() {
        if let Some(c) = any.downcast_ref::<CompoundCommand>() {
            n += c.commands.iter().map(|c| count_inserted(c.as_ref())).sum::<usize>();
        } else if let Some(c) = any.downcast_ref::<SnapshotCompound>() {
            n += c.inner.iter().map(|c| count_inserted(c.as_ref())).sum::<usize>();
        }
    }
    n
}

// ---- the probe ------------------------------------------------------------------------------------------------

/// What the viewport reads off a tool every frame, as the harness records it
/// (`generic_probe` of `scripts/golden/_modify_harness.py`). A field left
/// `None` is an attribute the Python tool does not have. The defaults are
/// the ones `tools/base.py::Tool` gives every tool.
#[derive(Clone, Debug)]
pub struct Probe {
    /// `rubber_band_lines()`.
    pub rb: Option<Vec<(V3, V3)>>,
    /// `guide_preview_lines()`.
    pub gl: Option<Vec<(V3, V3)>>,
    /// `value_label()`: `Some(None)` is a tool that has one and has none now.
    pub vl: Option<Option<(String, V3)>>,
    /// `status_clause()`.
    pub sc: Option<String>,
    /// `cursor_plus`.
    pub plus: Option<bool>,
    /// `wireframe_color`.
    pub wf: Option<[f64; 4]>,
    /// `vcb_caption()`.
    pub cap: Option<String>,
    /// `value_is_unitless()`.
    pub unitless: Option<bool>,
    /// `uses_snap`.
    pub uses_snap: bool,
    /// `snap_excluded()`: edges (by index in the active mesh) and groups.
    pub excl: Option<Option<(Vec<usize>, Vec<Path>)>>,
    /// `array_pending(viewport)`.
    pub ap: Option<bool>,
    /// `start_point`, `hover_point`.
    pub start: Option<V3>,
    pub hover: Option<V3>,
    /// The tool's own state, as the module's `extra_probe` records it.
    pub x: Option<Value>,
}

impl Default for Probe {
    fn default() -> Self {
        Probe {
            rb: Some(Vec::new()),
            gl: Some(Vec::new()),
            vl: None,
            sc: Some(String::new()),
            plus: None,
            wf: None,
            cap: None,
            unitless: None,
            uses_snap: true,
            excl: None,
            ap: None,
            start: None,
            hover: None,
            x: None,
        }
    }
}

/// The entry points of `tools.base.Tool` a viewport calls.
pub trait ModifyTool {
    fn on_activate(&mut self, env: &mut Env);
    fn on_deactivate(&mut self, env: &mut Env);
    fn on_click(&mut self, _env: &mut Env, _cx: &ToolCx) {}
    fn on_hover(&mut self, _env: &mut Env, _cx: &ToolCx) {}
    /// Defaults to a plain click, as `Tool.on_double_click` does.
    fn on_double_click(&mut self, env: &mut Env, cx: &ToolCx) {
        self.on_click(env, cx);
    }
    /// Defaults to a plain click, as `Tool.on_triple_click` does.
    fn on_triple_click(&mut self, env: &mut Env, cx: &ToolCx) {
        self.on_click(env, cx);
    }
    fn on_cancel(&mut self, _env: &mut Env) {}
    fn on_release(&mut self, _env: &mut Env) {}
    fn on_key(&mut self, _env: &mut Env, _key: i32, _mods: Mods) -> bool {
        false
    }
    fn on_key_release(&mut self, _env: &mut Env, _key: i32) -> bool {
        false
    }
    fn on_value(&mut self, _env: &mut Env, _value: &Vcb) -> bool {
        false
    }
    /// `close_array_window()`: Esc right after a copy.
    fn close_array_window(&mut self) {}
    /// `on_array_value(viewport, count, mode, step=None)`.
    fn on_array_value(&mut self, _env: &mut Env, _count: i64, _mode: char, _step: Option<f64>) -> bool {
        false
    }
    /// `on_box_select(viewport, rect, crossing, ...)`.
    fn on_box_select(&mut self, _env: &mut Env, _rect: [f64; 4], _crossing: bool) {}
    fn probe(&self, env: &Env) -> Probe;
}

// ---- helpers the tools share ----------------------------------------------------------------------------------

/// `Qt.Key_*` for an arrow key -> the axis it locks (`Right` red, `Left`
/// green, `Up` blue).
pub fn arrow_axis(key: i32) -> Option<Axis> {
    match key {
        keys::RIGHT => Some(Axis::X),
        keys::LEFT => Some(Axis::Y),
        keys::UP => Some(Axis::Z),
        _ => None,
    }
}

/// `round(x, 6)` of a Python float: the double nearest to `x` rounded to six
/// decimals, ties to even on the exact binary value.
pub fn round6(x: f64) -> f64 {
    if !x.is_finite() {
        return x;
    }
    format!("{x:.6}").parse().unwrap_or(x)
}

/// `(round(x, 6), round(y, 6), round(z, 6))` of a point: the key the copy
/// builders of Move and Rotate compare loop sides by.
pub fn key6(p: V3) -> [f64; 3] {
    [round6(p.0 as f64), round6(p.1 as f64), round6(p.2 as f64)]
}

/// The loops of a face as positions: the outer one, then each hole
/// (`face.vertices` and `face.holes`).
pub fn face_loops(m: &ingetrazo_model::mesh::Mesh, f: FaceId) -> Vec<Vec<V3>> {
    let face = m.face(f);
    std::iter::once(&face.outer).chain(face.holes.iter()).map(|lp| lp.iter().map(|&v| m.position(v)).collect()).collect()
}

/// `_dedup` of `tools/move.py`: the positions once each (by `topology._key`).
pub fn dedup(positions: &[V3]) -> Vec<V3> {
    let mut seen = std::collections::HashSet::new();
    let mut out = Vec::new();
    for &p in positions {
        if seen.insert(ingetrazo_model::topology::key(p)) {
            out.push(p);
        }
    }
    out
}
