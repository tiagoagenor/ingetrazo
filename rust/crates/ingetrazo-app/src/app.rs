// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! The window: a menu bar (File, Edit, Tools, View), a toolbar with the
//! tools, the viewport in the middle and a status bar with the tool's hint
//! and the value box.
//!
//! Everything the viewport does is the [`Session`]'s: this file turns egui
//! into [`Input`] events, draws what the tools preview over the picture, and
//! holds the window's own business (menus, file dialogs, the question about
//! unsaved work).
//!
//! Navigation is the Python viewport's: middle-drag orbits, Shift+middle-drag
//! pans, the wheel zooms towards the cursor, Shift+Z or F2 zooms extents,
//! Shift+P toggles perspective and parallel; O, H and Z make the left button
//! orbit, pan or zoom (Space or Esc gives it back). The tools are picked by
//! their shortcut (L, R, C, A, P, M, Q, S, F, E, B, T, Space for Select).

use std::collections::VecDeque;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use eframe::egui::{self, Color32, CursorIcon, Key, Pos2, Rect, Stroke};
use eframe::egui_wgpu;
use ingetrazo_math::V3;
use ingetrazo_snap::{SnapKind, SnapResult};


use crate::bench::{Bench, Frame};
use crate::input::Translator;
use crate::nav::{self, NavMode};
use crate::session::Session;
use crate::toolbox::{self, ToolId};
use crate::viewport::{GpuView, ViewCallback};

/// How many frame intervals the fps averages.
const FPS_WINDOW: usize = 60;
/// How long a flashed message stays (`flash_status`'s 2500 ms).
const FLASH: Duration = Duration::from_millis(2500);

/// What is waiting for an answer to "save the changes?".
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Pending {
    New,
    /// `None`: ask for the file after the question.
    Open(Option<PathBuf>),
    Quit,
}

pub struct App {
    rs: egui_wgpu::RenderState,
    sess: Session,
    translator: Translator,
    confirm: Option<Pending>,
    allow_close: bool,
    sent_title: String,
    error: Option<String>,
    flash: Option<(String, Instant)>,
    backend: String,
    last_frame: Option<Instant>,
    dts: VecDeque<f32>,
    continuous: bool,
    bench: Option<Bench>,
    last_ui_ms: f32,
    /// Frames to wait before [`Session::warm`] after a document is opened.
    warm_delay: u8,
}

impl App {
    /// `path`: the document to open, or `None` for an empty scene.
    pub fn new(cc: &eframe::CreationContext<'_>, path: Option<PathBuf>) -> App {
        App::with_bench(cc, path, None)
    }

    /// Like [`App::new`]; with `bench_orbit` the window runs the alpha
    /// benchmark (`--bench-orbit`, see [`crate::bench`]) and quits.
    pub fn with_bench(cc: &eframe::CreationContext<'_>, path: Option<PathBuf>, bench_orbit: Option<f64>) -> App {
        let mut open_ms = 0.0;
        let rs = cc.wgpu_render_state.clone().expect("the wgpu backend gives a render state");
        let info = rs.adapter.get_info();
        let backend = format!("{:?}: {}", info.backend, info.name);
        let view = GpuView::new(&rs.device, &rs.queue, rs.target_format);
        rs.renderer.write().callback_resources.insert(view);
        let mut app = App {
            rs,
            sess: Session::new(),
            translator: Translator::new(),
            confirm: None,
            allow_close: false,
            sent_title: String::new(),
            error: None,
            flash: None,
            backend,
            last_frame: None,
            dts: VecDeque::new(),
            continuous: false,
            bench: None,
            last_ui_ms: 0.0,
            warm_delay: 2,
        };
        if let Some(p) = path {
            let t = Instant::now();
            app.open(p);
            open_ms = t.elapsed().as_secs_f64() * 1000.0;
        }
        if let Some(secs) = bench_orbit {
            let mut b = Bench::new(secs);
            b.open_ms = open_ms;
            app.bench = Some(b);
            app.continuous = true;
        }
        app
    }

    fn gpu<R>(&self, f: impl FnOnce(&mut GpuView) -> R) -> Option<R> {
        self.rs.renderer.write().callback_resources.get_mut::<GpuView>().map(f)
    }

    // ---- the file -----------------------------------------------------------------------------

    /// Open a document and frame it.
    pub fn open(&mut self, path: PathBuf) {
        match self.sess.open(&path) {
            Ok(textures) => {
                self.gpu(|v| v.open(textures));
                self.sess.take_scene_changed();
                self.warm_delay = 2;
                self.error = None;
            }
            Err(e) => self.error = Some(e),
        }
    }

    fn open_dialog(&mut self) {
        let picked = rfd::FileDialog::new().set_title("Open").add_filter("IngeTrazo model (*.igz)", &["igz"]).pick_file();
        if let Some(path) = picked {
            self.open(path);
        }
    }

    fn new_document(&mut self) {
        self.sess.new_document();
        self.gpu(|v| v.open(Box::new(crate::paths::InstallTextures(ingetrazo_render::NoTextures))));
        self.sess.take_scene_changed();
        self.error = None;
    }

    /// `MainWindow._on_save` / `_on_save_as`. `true` when the file was written.
    fn save(&mut self, as_new: bool) -> bool {
        let path = match (&self.sess.path, as_new) {
            (Some(p), false) => p.clone(),
            _ => {
                let name = self.sess.path.as_deref().and_then(|p| p.file_name()).map(|n| n.to_string_lossy().into_owned()).unwrap_or_else(|| "untitled.igz".into());
                let Some(mut p) =
                    rfd::FileDialog::new().set_title("Save IngeTrazo document").set_file_name(name).add_filter("IngeTrazo model (*.igz)", &["igz"]).save_file()
                else {
                    return false;
                };
                if p.extension().is_none_or(|e| !e.eq_ignore_ascii_case("igz")) {
                    p.set_extension("igz");
                }
                p
            }
        };
        match self.sess.save_to(&path) {
            Ok(w) => {
                self.error = None;
                if w.missing > 0 {
                    self.flash = Some((format!("Saved with {} texture(s) that could not be read.", w.missing), Instant::now()));
                } else if w.embedded > 0 {
                    self.flash = Some((format!("Saved - {} texture(s) packed into the document.", w.embedded), Instant::now()));
                }
                true
            }
            Err(e) => {
                self.error = Some(format!("Save failed: {e}"));
                false
            }
        }
    }

    /// Do `what`, asking first when the work is unsaved (`_confirm_discard`).
    fn request(&mut self, ctx: &egui::Context, what: Pending) {
        if self.sess.dirty() {
            self.confirm = Some(what);
        } else {
            self.perform(ctx, what);
        }
    }

    fn perform(&mut self, ctx: &egui::Context, what: Pending) {
        match what {
            Pending::New => self.new_document(),
            Pending::Open(Some(p)) => self.open(p),
            Pending::Open(None) => self.open_dialog(),
            Pending::Quit => {
                self.allow_close = true;
                ctx.send_viewport_cmd(egui::ViewportCommand::Close);
            }
        }
    }

    // ---- shortcuts and menus -----------------------------------------------------------------

    /// The chords of the window (the viewport's own keys go through the
    /// session).
    fn chords(&mut self, ctx: &egui::Context) {
        enum Cmd {
            New,
            Open,
            Save,
            SaveAs,
            Undo,
            Redo,
            Extents,
        }
        let cmd = ctx.input(|i| {
            let m = i.modifiers;
            if m.command && m.shift && i.key_pressed(Key::S) {
                Some(Cmd::SaveAs)
            } else if m.command && m.shift && i.key_pressed(Key::Z) {
                Some(Cmd::Redo)
            } else if m.command && i.key_pressed(Key::Z) {
                Some(Cmd::Undo)
            } else if m.command && i.key_pressed(Key::Y) {
                Some(Cmd::Redo)
            } else if m.command && i.key_pressed(Key::S) {
                Some(Cmd::Save)
            } else if m.command && i.key_pressed(Key::O) {
                Some(Cmd::Open)
            } else if m.command && i.key_pressed(Key::N) {
                Some(Cmd::New)
            } else if i.key_pressed(Key::F2) {
                Some(Cmd::Extents)
            } else {
                None
            }
        });
        match cmd {
            Some(Cmd::New) => self.request(ctx, Pending::New),
            Some(Cmd::Open) => self.request(ctx, Pending::Open(None)),
            Some(Cmd::Save) => {
                self.save(false);
            }
            Some(Cmd::SaveAs) => {
                self.save(true);
            }
            Some(Cmd::Undo) => {
                self.sess.undo();
            }
            Some(Cmd::Redo) => {
                self.sess.redo();
            }
            Some(Cmd::Extents) => self.sess.zoom_extents(),
            None => {}
        }
    }

    fn menu(&mut self, ui: &mut egui::Ui) {
        let ctx = ui.ctx().clone();
        egui::MenuBar::new().ui(ui, |ui| {
            ui.menu_button("File", |ui| {
                if ui.add(egui::Button::new("New").shortcut_text("Ctrl+N")).clicked() {
                    ui.close();
                    self.request(&ctx, Pending::New);
                }
                if ui.add(egui::Button::new("Open...").shortcut_text("Ctrl+O")).clicked() {
                    ui.close();
                    self.request(&ctx, Pending::Open(None));
                }
                ui.separator();
                if ui.add(egui::Button::new("Save").shortcut_text("Ctrl+S")).clicked() {
                    ui.close();
                    self.save(false);
                }
                if ui.add(egui::Button::new("Save As...").shortcut_text("Ctrl+Shift+S")).clicked() {
                    ui.close();
                    self.save(true);
                }
                ui.separator();
                if ui.button("Quit").clicked() {
                    ui.close();
                    self.request(&ctx, Pending::Quit);
                }
            });
            ui.menu_button("Edit", |ui| {
                if ui.add_enabled(self.sess.can_undo(), egui::Button::new("Undo").shortcut_text("Ctrl+Z")).clicked() {
                    self.sess.undo();
                    ui.close();
                }
                if ui.add_enabled(self.sess.can_redo(), egui::Button::new("Redo").shortcut_text("Ctrl+Y")).clicked() {
                    self.sess.redo();
                    ui.close();
                }
            });
            ui.menu_button("Tools", |ui| {
                for group in toolbox::GROUPS {
                    for &id in group.tools {
                        let mut b = egui::Button::new(id.name()).selected(self.sess.tool == id);
                        if let Some(s) = id.shortcut_label() {
                            b = b.shortcut_text(s);
                        }
                        if ui.add(b).clicked() {
                            self.sess.set_tool(id);
                            ui.close();
                        }
                    }
                    ui.separator();
                }
            });
            ui.menu_button("View", |ui| {
                if ui.add(egui::Button::new("Zoom Extents").shortcut_text("Shift+Z")).clicked() {
                    self.sess.zoom_extents();
                    ui.close();
                }
                let label = if self.sess.cam.perspective { "Parallel projection" } else { "Perspective projection" };
                if ui.add(egui::Button::new(label).shortcut_text("Shift+P")).clicked() {
                    nav::toggle_projection(&mut self.sess.cam);
                    ui.close();
                }
                let mut two_point = self.sess.cam.two_point;
                if ui.checkbox(&mut two_point, "Two-point perspective").clicked() {
                    nav::toggle_two_point(&mut self.sess.cam);
                    ui.close();
                }
                ui.menu_button("Standard views", |ui| {
                    for (name, label) in
                        [("top", "Top"), ("bottom", "Bottom"), ("front", "Front"), ("back", "Back"), ("left", "Left"), ("right", "Right"), ("iso", "Iso")]
                    {
                        if ui.button(label).clicked() {
                            nav::set_view(&mut self.sess.cam, name);
                            ui.close();
                        }
                    }
                });
                ui.separator();
                for (mode, label, key) in [(NavMode::Orbit, "Orbit", "O"), (NavMode::Pan, "Pan", "H"), (NavMode::Zoom, "Zoom", "Z")] {
                    let on = self.sess.nav.nav_mode == Some(mode);
                    if ui.add(egui::Button::new(label).selected(on).shortcut_text(key)).clicked() {
                        self.sess.nav_mode(if on { None } else { Some(mode) });
                        ui.close();
                    }
                }
            });
            ui.checkbox(&mut self.continuous, "Continuous repaint");
        });
    }

    fn toolbar(&mut self, ui: &mut egui::Ui) {
        ui.spacing_mut().item_spacing = egui::vec2(3.0, 3.0);
        for group in toolbox::GROUPS {
            ui.label(egui::RichText::new(group.title).small().weak());
            ui.horizontal_wrapped(|ui| {
                for &id in group.tools {
                    let on = self.sess.tool == id && self.sess.nav.nav_mode.is_none();
                    let tip = match id.shortcut_label() {
                        Some(s) => format!("{} ({s})", id.name()),
                        None => id.name().to_string(),
                    };
                    let b = egui::Button::new(egui::RichText::new(id.glyph()).monospace()).selected(on).min_size(egui::vec2(34.0, 26.0));
                    if ui.add(b).on_hover_text(tip).clicked() {
                        self.sess.set_tool(id);
                    }
                }
            });
            ui.add_space(4.0);
        }
    }

    fn status(&mut self, ui: &mut egui::Ui) {
        let hint = self.flash.as_ref().filter(|(_, t)| t.elapsed() < FLASH).map(|(s, _)| s.clone());
        let hint = hint.unwrap_or_else(|| self.sess.status_line());
        ui.horizontal(|ui| {
            if let Some(e) = &self.error {
                ui.colored_label(Color32::LIGHT_RED, e);
            } else {
                ui.label(hint);
            }
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                // The value box: the caption, and what is typed (or the live reading).
                if let Some((caption, value, typing)) = self.sess.vcb() {
                    let text = if value.is_empty() { " ".to_string() } else { value };
                    let rich = egui::RichText::new(format!(" {text} ")).monospace();
                    let rich = if typing { rich.background_color(Color32::from_rgb(255, 232, 160)).color(Color32::BLACK) } else { rich.background_color(ui.visuals().extreme_bg_color) };
                    ui.label(rich);
                    ui.label(caption);
                }
            });
        });
        let mean = self.dts.iter().sum::<f32>() / self.dts.len().max(1) as f32;
        let stats = self.gpu(|v| v.stats).unwrap_or_default();
        ui.horizontal(|ui| {
            if self.continuous && mean > 0.0 {
                ui.label(format!("{:.0} fps ({mean:.1} ms)", 1000.0 / mean));
                ui.separator();
            }
            ui.label(format!("{} triangles, {} draw calls", stats.triangles, stats.draw_calls));
            ui.separator();
            ui.label(format!("{} | distance {:.2} m", self.sess.cam.projection_mode(), self.sess.cam.distance));
            ui.separator();
            ui.label(&self.backend);
        });
    }

    // ---- the viewport -------------------------------------------------------------------------------

    fn viewport(&mut self, ui: &mut egui::Ui) {
        let (rect, resp) = ui.allocate_exact_size(ui.available_size(), egui::Sense::click_and_drag());
        self.sess.set_size(rect.width().round() as i32, rect.height().round() as i32);
        if self.sess.fit_pending && rect.width() > 1.0 && rect.height() > 1.0 {
            self.sess.fit_pending = false;
            self.sess.zoom_extents();
        }

        // While the question about unsaved work is open the viewport sleeps.
        if self.confirm.is_none() {
            let (events, mods, time) = ui.input(|i| (i.events.clone(), i.modifiers, i.time));
            for input in self.translator.translate(&events, mods, time, rect, resp.hovered()) {
                self.sess.handle(&input);
            }
        }
        for text in self.sess.take_flashes() {
            self.flash = Some((text, Instant::now()));
        }

        // The model and the highlights for the renderer.
        self.sess.take_scene_changed();
        let (sel, hover) = self.sess.render_cues();
        // Draw the frame now, from the session's own scene (nothing is
        // copied: the renderer rebakes the meshes whose serial moved); the
        // paint callback below only blits it.
        let ppp = ui.ctx().pixels_per_point();
        let px = ((rect.width() * ppp).round() as u32, (rect.height() * ppp).round() as u32);
        {
            let (device, queue, scene, cam) = (&self.rs.device, &self.rs.queue, self.sess.scene(), self.sess.cam);
            if let Some(v) = self.rs.renderer.write().callback_resources.get_mut::<GpuView>() {
                v.set_cues(sel, hover);
                v.render_now(device, queue, scene, &cam, px);
            }
        }

        if resp.hovered() {
            ui.ctx().set_cursor_icon(match (self.sess.nav.nav_mode, self.sess.tool) {
                (Some(_), _) => CursorIcon::Grab,
                (None, ToolId::Select) => CursorIcon::Default,
                (None, ToolId::Move) => CursorIcon::Move,
                (None, ToolId::Eraser | ToolId::Paint) => CursorIcon::PointingHand,
                _ => CursorIcon::Crosshair,
            });
        }

        ui.painter().add(egui_wgpu::Callback::new_paint_callback(rect, ViewCallback));
        self.paint_overlay(&ui.painter_at(rect), rect);

        // A model just opened: once it has been on the screen for a couple of
        // frames, build what the first pointer move would (see `Session::warm`).
        if self.sess.warm_pending && !self.sess.fit_pending {
            if self.warm_delay == 0 {
                self.sess.warm();
            } else {
                self.warm_delay -= 1;
                ui.ctx().request_repaint();
            }
        }
    }

    /// What the tool previews over the picture: the rubber band, the guide,
    /// the snap marker and its tip, the floating value, the selection box
    /// (`Viewport._draw_overlay`).
    fn paint_overlay(&self, painter: &egui::Painter, rect: Rect) {
        let to = |p: V3| -> Option<Pos2> { self.sess.to_pixel(p).map(|(x, y)| rect.min + egui::vec2(x as f32, y as f32)) };
        let snap = self.sess.snap();
        let tool_snaps = self.sess.host.tool().uses_snap();
        let ov = self.sess.overlay();

        // The rubber band, in the colour of the inference that holds it.
        let (color, width) = rubber_pen(ov.wireframe, snap);
        for (a, b) in &ov.rubber {
            if let (Some(p), Some(q)) = (to(*a), to(*b)) {
                painter.line_segment([p, q], Stroke::new(width, color));
            }
        }
        // The guide the Tape or the Protractor is about to leave.
        for (a, b) in &ov.guide {
            if let (Some(p), Some(q)) = (to(*a), to(*b)) {
                painter.extend(egui::Shape::dashed_line(&[p, q], Stroke::new(1.5, Color32::from_rgb(70, 90, 120)), 10.0, 7.0));
            }
        }
        // Push/Pull's "level with that point".
        for (a, b, _kind) in &ov.inference_guides {
            if let (Some(p), Some(q)) = (to(*a), to(*b)) {
                painter.extend(egui::Shape::dashed_line(&[p, q], Stroke::new(1.5, Color32::from_rgb(217, 77, 204)), 6.0, 4.0));
            }
        }
        if let Some((p, kind)) = ov.marker {
            if let Some(c) = to(p) {
                let col = Color32::from_rgb(41, 158, 92);
                painter.circle_stroke(c, 6.5, Stroke::new(2.6, col));
                painter.text(c + egui::vec2(10.0, 16.0), egui::Align2::LEFT_BOTTOM, kind.replace('_', " "), egui::FontId::proportional(12.0), col);
            }
        }
        if tool_snaps {
            if let Some(s) = snap {
                paint_snap(painter, s, &to);
            }
        }
        // The floating value: the typed text while there is one.
        if let Some((text, at)) = &ov.label {
            if let Some(c) = to(*at) {
                let typed = &self.sess.host.core.value_buffer;
                let (text, shadow) = if typed.is_empty() { (text.clone(), Color32::from_white_alpha(220)) } else { (typed.clone(), Color32::from_rgba_unmultiplied(255, 220, 130, 235)) };
                let font = egui::FontId::proportional(14.0);
                painter.text(c + egui::vec2(12.0, -7.0), egui::Align2::LEFT_BOTTOM, &text, font.clone(), shadow);
                painter.text(c + egui::vec2(11.0, -8.0), egui::Align2::LEFT_BOTTOM, text, font, Color32::from_rgb(15, 20, 27));
            }
        }
        // The Select tool's box: solid blue for a window, dashed green for a crossing.
        if let Some(b) = self.sess.box_rect() {
            let (a, c) = (rect.min + egui::vec2(b.start.0 as f32, b.start.1 as f32), rect.min + egui::vec2(b.cur.0 as f32, b.cur.1 as f32));
            let r = Rect::from_two_pos(a, c);
            if b.cur.0 < b.start.0 {
                let col = Color32::from_rgb(40, 158, 92);
                painter.rect_filled(r, 0.0, Color32::from_rgba_unmultiplied(40, 158, 92, 28));
                let corners = [r.left_top(), r.right_top(), r.right_bottom(), r.left_bottom(), r.left_top()];
                painter.extend(egui::Shape::dashed_line(&corners, Stroke::new(1.5, col), 6.0, 4.0));
            } else {
                let col = Color32::from_rgb(51, 102, 199);
                painter.rect_filled(r, 0.0, Color32::from_rgba_unmultiplied(51, 102, 199, 28));
                painter.rect_stroke(r, 0.0, Stroke::new(1.5, col), egui::StrokeKind::Inside);
            }
        }
    }

    /// The question before work is thrown away (`_confirm_discard`).
    fn confirm_window(&mut self, ctx: &egui::Context) {
        let Some(what) = self.confirm.clone() else { return };
        let prompt = match what {
            Pending::New => "Discard current drawing?",
            Pending::Open(_) => "Discard current drawing and open another?",
            Pending::Quit => "Quit IngeTrazo?",
        };
        let mut answer: Option<Option<bool>> = None; // Some(Some(save)) | Some(None) cancel
        egui::Window::new("Unsaved changes").collapsible(false).resizable(false).anchor(egui::Align2::CENTER_CENTER, egui::vec2(0.0, 0.0)).show(ctx, |ui| {
            ui.label(prompt);
            ui.label("Unsaved changes will be lost.");
            ui.horizontal(|ui| {
                if ui.button("Save").clicked() {
                    answer = Some(Some(true));
                }
                if ui.button("Don't Save").clicked() {
                    answer = Some(Some(false));
                }
                if ui.button("Cancel").clicked() {
                    answer = Some(None);
                }
            });
        });
        match answer {
            Some(Some(save)) => {
                self.confirm = None;
                if !save || (self.save(false) && !self.sess.dirty()) {
                    self.perform(ctx, what);
                }
            }
            Some(None) => self.confirm = None,
            None => {}
        }
    }
}

/// `_draw_rubber_band`: a tool's own colour, else the snap's.
fn rubber_pen(forced: Option<[f64; 4]>, snap: Option<&SnapResult>) -> (Color32, f32) {
    let rgba = |c: [f64; 4]| Color32::from_rgba_unmultiplied((c[0] * 255.0) as u8, (c[1] * 255.0) as u8, (c[2] * 255.0) as u8, (c[3] * 255.0) as u8);
    if let Some(f) = forced {
        return (rgba(f), 2.0);
    }
    let from_snap = |s: &SnapResult, a: f64| rgba([s.color[0], s.color[1], s.color[2], a]);
    match snap {
        Some(s) if s.kind == SnapKind::Axis => (from_snap(s, 1.0), 2.5),
        Some(s) if s.kind == SnapKind::AxisInference => (from_snap(s, 0.5), 2.5),
        Some(s) if matches!(s.kind, SnapKind::Reference | SnapKind::ThroughPoint | SnapKind::PerpFace) => (from_snap(s, 1.0), 2.5),
        Some(s) if s.kind == SnapKind::Extension => (rgba([0.95, 0.45, 0.16, 0.85]), 2.5),
        Some(s) if s.kind == SnapKind::Close => (rgba([0.20, 0.40, 0.78, 0.95]), 2.0),
        _ => (rgba([0.95, 0.45, 0.16, 0.85]), 2.0),
    }
}

/// `Viewport._SNAP_LABELS`.
fn snap_label(s: &SnapResult) -> Option<&'static str> {
    Some(match s.kind {
        SnapKind::Endpoint => "Endpoint",
        SnapKind::Midpoint => "Midpoint",
        SnapKind::ArcMidpoint => "Arc midpoint",
        SnapKind::ComponentOrigin => "Component origin",
        SnapKind::OnLine => "On line",
        SnapKind::OnAxis => "On axis",
        SnapKind::OnEdge => "On edge",
        SnapKind::OnFace => "On face",
        SnapKind::Origin => "Origin",
        SnapKind::Extension => "Extension",
        SnapKind::Intersection => "Intersection",
        SnapKind::FromPoint => "From point",
        SnapKind::Aligned => {
            if s.axis == Some(ingetrazo_snap::Axis::Z) {
                "Level with point"
            } else {
                "In line with point"
            }
        }
        SnapKind::ThroughPoint => "Through point",
        SnapKind::PerpFace => "Perpendicular to face",
        SnapKind::Center => "Center",
        _ => return None,
    })
}

/// `Viewport._draw_snap_indicator`: the dashed guides and the marker of a
/// discrete snap, with its tip.
fn paint_snap(painter: &egui::Painter, s: &SnapResult, to: &dyn Fn(V3) -> Option<Pos2>) {
    if matches!(s.kind, SnapKind::AxisInference | SnapKind::Axis | SnapKind::None) {
        return;
    }
    let Some(c) = to(s.point) else { return };
    let rgb = |c: [f64; 3], a: u8| Color32::from_rgba_unmultiplied((c[0] * 255.0) as u8, (c[1] * 255.0) as u8, (c[2] * 255.0) as u8, a);
    let color = rgb(s.color, 255);
    if let Some((a, b)) = s.guide {
        if let (Some(p), Some(q)) = (to(a), to(b)) {
            let gc = rgb(s.guide_color.unwrap_or(s.color), 230);
            painter.extend(egui::Shape::dashed_line(&[p, q], Stroke::new(2.0, gc), 6.0, 4.0));
        }
    }
    for (a, b, gc) in s.guides.iter().flatten() {
        if let (Some(p), Some(q)) = (to(*a), to(*b)) {
            painter.extend(egui::Shape::dashed_line(&[p, q], Stroke::new(2.0, rgb(*gc, 230)), 6.0, 4.0));
        }
    }
    let halo = Stroke::new(4.5, Color32::from_white_alpha(230));
    let mark = Stroke::new(2.6, color);
    let fill = rgb(s.color, 77);
    match s.kind {
        SnapKind::Intersection => {
            for st in [halo, mark] {
                painter.line_segment([c + egui::vec2(-8.0, -8.0), c + egui::vec2(8.0, 8.0)], st);
                painter.line_segment([c + egui::vec2(-8.0, 8.0), c + egui::vec2(8.0, -8.0)], st);
            }
        }
        SnapKind::Center => {
            painter.circle_stroke(c, 6.5, halo);
            painter.circle(c, 6.5, rgb(s.color, 217), mark);
        }
        SnapKind::Endpoint | SnapKind::Origin | SnapKind::ComponentOrigin | SnapKind::OnEdge | SnapKind::OnAxis | SnapKind::OnLine | SnapKind::Extension | SnapKind::FromPoint | SnapKind::Aligned => {
            let r = Rect::from_center_size(c, egui::vec2(14.0, 14.0));
            painter.rect_stroke(r, 0.0, halo, egui::StrokeKind::Middle);
            painter.rect(r, 0.0, fill, mark, egui::StrokeKind::Middle);
        }
        SnapKind::Midpoint | SnapKind::ArcMidpoint => {
            let pts = vec![c + egui::vec2(0.0, -9.0), c + egui::vec2(9.0, 0.0), c + egui::vec2(0.0, 9.0), c + egui::vec2(-9.0, 0.0)];
            painter.add(egui::Shape::closed_line(pts.clone(), halo));
            painter.add(egui::Shape::convex_polygon(pts, fill, mark));
        }
        SnapKind::OnFace => {
            painter.circle_stroke(c, 5.5, halo);
            painter.circle_stroke(c, 5.5, mark);
        }
        SnapKind::Close => {
            painter.circle_stroke(c, 9.0, halo);
            painter.circle_stroke(c, 9.0, mark);
        }
        SnapKind::Reference | SnapKind::ThroughPoint | SnapKind::PerpFace => {
            painter.circle_stroke(c, 6.5, halo);
            painter.circle_stroke(c, 6.5, mark);
        }
        _ => {}
    }
    if let Some(label) = snap_label(s) {
        let mut text = label.to_string();
        match s.context {
            Some(ingetrazo_snap::Context::Component) => text.push_str(" in component"),
            Some(ingetrazo_snap::Context::Group) => text.push_str(" in group"),
            None => {}
        }
        let font = egui::FontId::proportional(12.0);
        painter.text(c + egui::vec2(11.0, 17.0), egui::Align2::LEFT_BOTTOM, &text, font.clone(), Color32::from_white_alpha(220));
        painter.text(c + egui::vec2(10.0, 16.0), egui::Align2::LEFT_BOTTOM, text, font, color);
    }
}

impl eframe::App for App {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        let now = Instant::now();
        if let Some(last) = self.last_frame {
            self.dts.push_back(now.duration_since(last).as_secs_f32() * 1000.0);
            if self.dts.len() > FPS_WINDOW {
                self.dts.pop_front();
            }
        }
        self.last_frame = Some(now);
        if let Some(mut b) = self.bench.take() {
            b.on_frame_start(now);
            let frame = Frame { dt: self.dts.back().copied().unwrap_or(0.0), ui: self.last_ui_ms, prepare: self.gpu(|v| v.prepare_ms).unwrap_or(0.0) };
            b.step(&mut self.sess, now, frame);
            if b.done {
                let stats = self.gpu(|v| v.stats).unwrap_or_default();
                let faces: usize = self.sess.scene().meshes.iter().map(|m| m.faces.len()).sum();
                print!("{}", b.report(&format!("faces {faces} triangles {} draw_calls {} backend \"{}\"", stats.triangles, stats.draw_calls, self.backend)));
                self.allow_close = true;
                ctx.send_viewport_cmd(egui::ViewportCommand::Close);
            } else {
                self.bench = Some(b);
            }
        }
        if self.continuous {
            ctx.request_repaint();
        }
        if let Some((_, t)) = &self.flash {
            match FLASH.checked_sub(t.elapsed()) {
                Some(left) => ctx.request_repaint_after(left),
                None => self.flash = None,
            }
        }

        // The window's close button: ask first when the work is unsaved.
        if ctx.input(|i| i.viewport().close_requested()) && !self.allow_close && self.sess.dirty() {
            ctx.send_viewport_cmd(egui::ViewportCommand::CancelClose);
            self.confirm = Some(Pending::Quit);
        }

        if self.confirm.is_none() {
            self.chords(&ctx);
        }
        egui::Panel::top("menu").show(ui, |ui| self.menu(ui));
        egui::Panel::bottom("status").show(ui, |ui| self.status(ui));
        egui::Panel::left("tools").resizable(false).exact_size(124.0).show(ui, |ui| self.toolbar(ui));
        egui::CentralPanel::default().show(ui, |ui| self.viewport(ui));
        self.confirm_window(&ctx);

        let title = self.sess.title();
        if title != self.sent_title {
            ctx.send_viewport_cmd(egui::ViewportCommand::Title(title.clone()));
            self.sent_title = title;
        }
        self.last_ui_ms = now.elapsed().as_secs_f32() * 1000.0;
    }
}
