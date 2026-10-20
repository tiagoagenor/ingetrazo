// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! The window: a top bar (File, View), the viewport in the middle and a
//! status line with the fps.
//!
//! Navigation is the Python viewport's: middle-drag orbits, Shift+middle-drag
//! pans, the wheel zooms towards the cursor, Shift+Z or F2 zooms extents,
//! Shift+P toggles perspective and parallel; O, H and Z make the left button
//! orbit, pan or zoom (Space or Esc gives it back).

use std::collections::VecDeque;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use eframe::egui::{self, Event, Key, MouseWheelUnit, PointerButton};
use eframe::egui_wgpu;
use ingetrazo_render::Camera;

use crate::document::{self, Document};
use crate::nav::{self, Button, NavMode, NavSettings, Navigator};
use crate::probe::{PickCache, ScenePick};
use crate::viewport::{GpuView, ViewCallback};

/// Wheel notches in egui's smooth-scroll points (`egui` turns a notch into
/// 50 points).
const POINTS_PER_NOTCH: f32 = 50.0;
/// How many frame intervals the fps averages.
const FPS_WINDOW: usize = 60;

pub struct App {
    rs: egui_wgpu::RenderState,
    doc: Document,
    cam: Camera,
    nav: Navigator,
    pick: PickCache,
    /// Frame the model once the viewport has a size (a document just opened).
    fit_pending: bool,
    error: Option<String>,
    backend: String,
    last_frame: Option<Instant>,
    dts: VecDeque<f32>,
    continuous: bool,
}

impl App {
    /// `path`: the document to open, or `None` for an empty scene.
    pub fn new(cc: &eframe::CreationContext<'_>, path: Option<PathBuf>) -> App {
        let rs = cc.wgpu_render_state.clone().expect("the wgpu backend gives a render state");
        let info = rs.adapter.get_info();
        let backend = format!("{:?}: {}", info.backend, info.name);
        let view = GpuView::new(&rs.device, &rs.queue, rs.target_format);
        rs.renderer.write().callback_resources.insert(view);
        let mut app = App {
            rs,
            doc: Document::empty(),
            cam: Camera::default(),
            nav: Navigator::new(NavSettings::default()),
            pick: PickCache::default(),
            fit_pending: false,
            error: None,
            backend,
            last_frame: None,
            dts: VecDeque::new(),
            continuous: false,
        };
        if let Some(p) = path {
            app.open(&cc.egui_ctx, p);
        }
        app
    }

    /// Open a document and frame it.
    pub fn open(&mut self, ctx: &egui::Context, path: PathBuf) {
        match document::open_path(&path) {
            Ok((doc, textures)) => {
                if let Some(v) = self.rs.renderer.write().callback_resources.get_mut::<GpuView>() {
                    v.set_scene(doc.scene.clone(), textures);
                }
                self.doc = doc;
                self.pick.invalidate();
                self.cam = Camera::default();
                self.nav.end_drag();
                self.fit_pending = true;
                self.error = None;
                ctx.send_viewport_cmd(egui::ViewportCommand::Title(self.doc.title()));
            }
            Err(e) => self.error = Some(e),
        }
    }

    fn open_dialog(&mut self, ctx: &egui::Context) {
        let picked = rfd::FileDialog::new().set_title("Open").add_filter("IngeTrazo model (*.igz)", &["igz"]).pick_file();
        if let Some(path) = picked {
            self.open(ctx, path);
        }
    }

    /// `MainWindow._on_zoom_extents`.
    fn zoom_extents(&mut self) {
        let mut probe = ScenePick::new(&self.doc.scene, &mut self.pick, self.cam.hypot);
        if let Some((lo, hi)) = probe.bounds() {
            nav::fit_box(&mut self.cam, lo, hi, nav::FIT_MARGIN);
        }
    }

    fn stats(&self) -> ingetrazo_render::FrameStats {
        self.rs.renderer.read().callback_resources.get::<GpuView>().map(|v| v.stats).unwrap_or_default()
    }

    fn shortcuts(&mut self, ctx: &egui::Context) {
        if ctx.egui_wants_keyboard_input() {
            return;
        }
        let (open, ze, projection, orbit, pan, zoom, none) = ctx.input(|i| {
            let shift = i.modifiers.shift;
            let plain = !i.modifiers.command && !i.modifiers.alt && !shift;
            (
                i.modifiers.command && i.key_pressed(Key::O),
                (shift && i.key_pressed(Key::Z)) || i.key_pressed(Key::F2),
                shift && i.key_pressed(Key::P),
                plain && i.key_pressed(Key::O),
                plain && i.key_pressed(Key::H),
                plain && i.key_pressed(Key::Z),
                plain && (i.key_pressed(Key::Space) || i.key_pressed(Key::Escape)),
            )
        });
        if open {
            self.open_dialog(ctx);
        }
        if ze {
            self.zoom_extents();
        }
        if projection {
            nav::toggle_projection(&mut self.cam);
        }
        for (hit, mode) in [(orbit, Some(NavMode::Orbit)), (pan, Some(NavMode::Pan)), (zoom, Some(NavMode::Zoom)), (none, None)] {
            if hit {
                self.nav.set_nav_mode(mode);
            }
        }
    }

    fn menu(&mut self, ui: &mut egui::Ui) {
        egui::MenuBar::new().ui(ui, |ui| {
            ui.menu_button("File", |ui| {
                if ui.add(egui::Button::new("Open...").shortcut_text("Ctrl+O")).clicked() {
                    ui.close();
                    self.open_dialog(&ui.ctx().clone());
                }
                ui.separator();
                if ui.button("Quit").clicked() {
                    ui.ctx().send_viewport_cmd(egui::ViewportCommand::Close);
                }
            });
            ui.menu_button("View", |ui| {
                if ui.add(egui::Button::new("Zoom Extents").shortcut_text("Shift+Z")).clicked() {
                    self.zoom_extents();
                    ui.close();
                }
                let label = if self.cam.perspective { "Parallel projection" } else { "Perspective projection" };
                if ui.add(egui::Button::new(label).shortcut_text("Shift+P")).clicked() {
                    nav::toggle_projection(&mut self.cam);
                    ui.close();
                }
                let mut two_point = self.cam.two_point;
                if ui.checkbox(&mut two_point, "Two-point perspective").clicked() {
                    nav::toggle_two_point(&mut self.cam);
                    ui.close();
                }
                ui.menu_button("Standard views", |ui| {
                    for (name, label) in
                        [("top", "Top"), ("bottom", "Bottom"), ("front", "Front"), ("back", "Back"), ("left", "Left"), ("right", "Right"), ("iso", "Iso")]
                    {
                        if ui.button(label).clicked() {
                            nav::set_view(&mut self.cam, name);
                            ui.close();
                        }
                    }
                });
                ui.separator();
                for (mode, label, key) in
                    [(NavMode::Orbit, "Orbit", "O"), (NavMode::Pan, "Pan", "H"), (NavMode::Zoom, "Zoom", "Z")]
                {
                    let on = self.nav.nav_mode == Some(mode);
                    if ui.add(egui::Button::new(label).selected(on).shortcut_text(key)).clicked() {
                        self.nav.set_nav_mode(if on { None } else { Some(mode) });
                        ui.close();
                    }
                }
            });
            ui.checkbox(&mut self.continuous, "Continuous repaint");
        });
    }

    fn status(&self, ui: &mut egui::Ui) {
        let mean = self.dts.iter().sum::<f32>() / self.dts.len().max(1) as f32;
        let live = self.continuous || self.last_frame.is_some_and(|t| t.elapsed() < Duration::from_millis(500));
        let stats = self.stats();
        ui.horizontal(|ui| {
            if let Some(e) = &self.error {
                ui.colored_label(egui::Color32::LIGHT_RED, e);
            } else if live && mean > 0.0 {
                ui.label(format!("{:.0} fps ({mean:.1} ms)", 1000.0 / mean));
            } else {
                ui.label("idle (repaints on input)");
            }
            ui.separator();
            ui.label(format!("{} triangles, {} draw calls", stats.triangles, stats.draw_calls));
            ui.separator();
            ui.label(format!("{} | distance {:.2} m", self.cam.projection_mode(), self.cam.distance));
            if let Some(mode) = self.nav.nav_mode {
                ui.separator();
                ui.label(format!("nav: {mode:?}"));
            }
            ui.separator();
            ui.label(&self.backend);
        });
    }

    fn viewport(&mut self, ui: &mut egui::Ui) {
        let (rect, resp) = ui.allocate_exact_size(ui.available_size(), egui::Sense::click_and_drag());
        let size = (rect.width().round() as i32, rect.height().round() as i32);
        self.cam.set_aspect(size.0.max(1) as u32, size.1.max(1) as u32);
        if self.fit_pending && size.0 > 0 && size.1 > 0 {
            self.fit_pending = false;
            self.zoom_extents();
        }

        let (events, shift, any_down) = ui.input(|i| (i.events.clone(), i.modifiers.shift, i.pointer.any_down()));
        let hovered = resp.hovered();
        let cam_hypot = self.cam.hypot;
        let rel = |p: egui::Pos2| ((p.x - rect.min.x) as f64, (p.y - rect.min.y) as f64);
        for event in events {
            match event {
                Event::PointerButton { pos, button, pressed, modifiers } => {
                    let b = match button {
                        PointerButton::Middle => Button::Middle,
                        PointerButton::Primary => Button::Left,
                        _ => continue,
                    };
                    if pressed {
                        if rect.contains(pos) && hovered {
                            let (x, y) = rel(pos);
                            let mut probe = ScenePick::new(&self.doc.scene, &mut self.pick, cam_hypot);
                            self.nav.press(&self.cam, &mut probe, size, b, x, y, modifiers.shift);
                        }
                    } else {
                        self.nav.release(b);
                    }
                }
                Event::PointerMoved(pos) if self.nav.dragging() => {
                    let (x, y) = rel(pos);
                    self.nav.drag(&mut self.cam, size, x, y, shift, any_down);
                }
                Event::MouseWheel { unit, delta, .. } if hovered => {
                    let notches = match unit {
                        MouseWheelUnit::Line => delta.y,
                        MouseWheelUnit::Point => delta.y / POINTS_PER_NOTCH,
                        MouseWheelUnit::Page => delta.y * 10.0,
                    };
                    if let Some(pos) = ui.input(|i| i.pointer.latest_pos()) {
                        let (x, y) = rel(pos);
                        let mut probe = ScenePick::new(&self.doc.scene, &mut self.pick, cam_hypot);
                        self.nav.wheel(&mut self.cam, &mut probe, size, x, y, notches as f64 * 120.0);
                    }
                }
                _ => {}
            }
        }

        let ppp = ui.ctx().pixels_per_point();
        ui.painter().add(egui_wgpu::Callback::new_paint_callback(
            rect,
            ViewCallback { camera: self.cam, px: ((rect.width() * ppp).round() as u32, (rect.height() * ppp).round() as u32) },
        ));
        ui.painter().text(
            rect.left_top() + egui::vec2(8.0, 6.0),
            egui::Align2::LEFT_TOP,
            "orbit: middle drag   pan: Shift + middle drag   zoom: wheel   extents: Shift+Z   projection: Shift+P",
            egui::FontId::proportional(12.0),
            egui::Color32::from_gray(90),
        );
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
        if self.continuous {
            ctx.request_repaint();
        }

        self.shortcuts(&ctx);
        egui::Panel::top("menu").show(ui, |ui| self.menu(ui));
        egui::Panel::bottom("status").show(ui, |ui| self.status(ui));
        egui::CentralPanel::default().show(ui, |ui| self.viewport(ui));
    }
}
