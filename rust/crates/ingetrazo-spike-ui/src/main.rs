//! IngeTrazo M0.8 UI spike: eframe (wgpu) + egui_dock + a custom wgpu paint
//! callback drawing an .igz (or a synthetic stress scene). Throwaway code.
//!
//! Usage: ingetrazo-spike-ui [--model PATH] [--stress] [--novsync]
//!                           [--bench SECONDS] [--accesskit-dump]

mod gpu;
mod model;

use eframe::egui::{self, PointerButton};
use eframe::egui_wgpu::{self, wgpu};
use egui_dock::{DockArea, DockState, NodeIndex, Style, TabViewer};
use std::collections::VecDeque;
use std::time::Instant;

const DEFAULT_MODEL: &str = "examples/arco-yanque.igz";

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum Tab {
    Viewport,
    Outliner,
    EntityInfo,
}

struct Args {
    model: String,
    stress: bool,
    novsync: bool,
    bench: Option<f32>,
    accesskit_dump: bool,
}

fn parse_args() -> Args {
    let mut a = Args {
        model: DEFAULT_MODEL.into(),
        stress: false,
        novsync: false,
        bench: None,
        accesskit_dump: false,
    };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--model" => a.model = it.next().unwrap_or_default(),
            "--stress" => a.stress = true,
            "--novsync" => a.novsync = true,
            "--bench" => a.bench = it.next().and_then(|s| s.parse().ok()),
            "--accesskit-dump" => a.accesskit_dump = true,
            other => eprintln!("ignoring unknown argument {other}"),
        }
    }
    a
}

struct State {
    rs: egui_wgpu::RenderState,
    mesh: model::Mesh,
    source: String,
    cam: gpu::Camera,
    selected: Option<usize>,
    text: String,
    backend: String,
    // timing
    last_frame: Option<Instant>,
    dts: VecDeque<f32>,
    cpu_ms: f32,
    continuous: bool,
    // bench
    bench_secs: Option<f32>,
    bench_start: Option<Instant>,
    bench_frames: usize,
    bench_dts: Vec<f32>,
    bench_cpu: Vec<f32>,
}

impl State {
    fn set_mesh(&mut self, mesh: model::Mesh, source: String) {
        let (c, r) = mesh.bounds();
        self.cam = gpu::Camera::fit(c, r);
        self.selected = None;
        let mut renderer = self.rs.renderer.write();
        if let Some(g) = renderer.callback_resources.get_mut::<gpu::Gpu>() {
            g.upload(&self.rs, &mesh);
        }
        self.mesh = mesh;
        self.source = source;
    }

    fn load_file(&mut self, path: &str) {
        let t = Instant::now();
        match model::load_igz(path) {
            Ok(m) => {
                eprintln!("loaded {path} in {:.0} ms", t.elapsed().as_secs_f32() * 1e3);
                self.set_mesh(m, path.to_string());
            }
            Err(e) => eprintln!("load failed: {e}"),
        }
    }

    fn load_stress(&mut self) {
        let t = Instant::now();
        let m = model::stress_scene(45, 128);
        eprintln!(
            "stress scene built in {:.0} ms ({} tris)",
            t.elapsed().as_secs_f32() * 1e3,
            m.tri_count()
        );
        self.set_mesh(m, "synthetic stress scene".into());
    }
}

struct App {
    dock: DockState<Tab>,
    s: State,
}

struct Viewer<'a> {
    s: &'a mut State,
}

impl TabViewer for Viewer<'_> {
    type Tab = Tab;

    fn id(&mut self, tab: &mut Tab) -> egui::Id {
        egui::Id::new(match tab {
            Tab::Viewport => "viewport",
            Tab::Outliner => "outliner",
            Tab::EntityInfo => "entity_info",
        })
    }

    fn title(&mut self, tab: &mut Tab) -> egui::WidgetText {
        match tab {
            Tab::Viewport => "3D Viewport",
            Tab::Outliner => "Outliner",
            Tab::EntityInfo => "Entity info",
        }
        .into()
    }

    fn is_closeable(&self, tab: &Tab) -> bool {
        *tab != Tab::Viewport
    }

    fn scroll_bars(&self, tab: &Tab) -> [bool; 2] {
        if *tab == Tab::Viewport {
            [false, false]
        } else {
            [true, true]
        }
    }

    fn ui(&mut self, ui: &mut egui::Ui, tab: &mut Tab) {
        match tab {
            Tab::Viewport => viewport_ui(self.s, ui),
            Tab::Outliner => outliner_ui(self.s, ui),
            Tab::EntityInfo => info_ui(self.s, ui),
        }
    }
}

fn viewport_ui(s: &mut State, ui: &mut egui::Ui) {
    let (rect, resp) = ui.allocate_exact_size(ui.available_size(), egui::Sense::click_and_drag());
    let (shift, scroll) = ui.input(|i| (i.modifiers.shift, i.smooth_scroll_delta.y));
    let d = resp.drag_delta();
    if resp.dragged_by(PointerButton::Middle) || (resp.dragged_by(PointerButton::Primary) && shift) {
        let (r, u) = s.cam.right_up();
        let k = s.cam.dist * 2.0 * (45.0f32.to_radians() * 0.5).tan() / rect.height().max(1.0);
        for i in 0..3 {
            s.cam.target[i] += (-r[i] * d.x + u[i] * d.y) * k;
        }
    } else if resp.dragged_by(PointerButton::Primary) {
        s.cam.yaw -= d.x * 0.008;
        s.cam.pitch = (s.cam.pitch + d.y * 0.008).clamp(-1.55, 1.55);
    }
    if resp.hovered() && scroll != 0.0 {
        s.cam.dist = (s.cam.dist * (-scroll * 0.002).exp()).max(1e-3);
    }

    let aspect = rect.width().max(1.0) / rect.height().max(1.0);
    let eye = s.cam.eye();
    let mut l = [
        eye[0] - s.cam.target[0],
        eye[1] - s.cam.target[1],
        eye[2] - s.cam.target[2] + s.cam.dist * 0.5,
    ];
    let len = (l[0] * l[0] + l[1] * l[1] + l[2] * l[2]).sqrt().max(1e-9);
    l = [l[0] / len, l[1] / len, l[2] / len];
    ui.painter().add(egui_wgpu::Callback::new_paint_callback(
        rect,
        gpu::ViewportCallback {
            view_proj: s.cam.view_proj(aspect),
            light: l,
        },
    ));
    ui.painter().text(
        rect.left_top() + egui::vec2(8.0, 6.0),
        egui::Align2::LEFT_TOP,
        "orbit: left drag   pan: middle drag / shift+drag   zoom: wheel",
        egui::FontId::proportional(12.0),
        egui::Color32::from_gray(90),
    );
}

fn outliner_ui(s: &mut State, ui: &mut egui::Ui) {
    ui.label(format!("{} groups", s.mesh.groups.len()));
    ui.separator();
    let row_h = ui.text_style_height(&egui::TextStyle::Body);
    let n = s.mesh.groups.len();
    egui::ScrollArea::vertical().auto_shrink([false, false]).show_rows(ui, row_h, n, |ui, range| {
        for i in range {
            let g = &s.mesh.groups[i];
            let label = format!("{}  ({} tris)", g.name, g.tris);
            if ui.selectable_label(s.selected == Some(i), label).clicked() {
                s.selected = Some(i);
            }
        }
    });
}

fn info_ui(s: &mut State, ui: &mut egui::Ui) {
    match s.selected.and_then(|i| s.mesh.groups.get(i)) {
        Some(g) => {
            ui.heading(&g.name);
            egui::Grid::new("info").show(ui, |ui| {
                let mut row = |k: &str, v: String| {
                    ui.label(k);
                    ui.label(v);
                    ui.end_row();
                };
                row("Triangles", g.tris.to_string());
                row("Edges", g.edges.to_string());
                row(
                    "Prototype",
                    g.proto.map_or("none".into(), |p| format!("#{p}")),
                );
                row("Min", format!("{:.2}, {:.2}, {:.2}", g.min[0], g.min[1], g.min[2]));
                row("Max", format!("{:.2}, {:.2}, {:.2}", g.max[0], g.max[1], g.max[2]));
            });
        }
        None => {
            ui.label("Select a group in the Outliner.");
        }
    }
    ui.separator();
    ui.label("Text / IME test (type here, try an IME such as zh-CN pinyin):");
    let resp = ui.text_edit_singleline(&mut s.text);
    ui.label(format!(
        "{} chars; focused: {}",
        s.text.chars().count(),
        resp.has_focus()
    ));
    ui.label("Backend / adapter:");
    ui.label(&s.backend);
    ui.label("Model source:");
    ui.label(&s.source);
}

fn percentile(sorted: &[f32], p: f32) -> f32 {
    if sorted.is_empty() {
        return 0.0;
    }
    sorted[((sorted.len() - 1) as f32 * p) as usize]
}

impl App {
    fn finish_bench(&self) {
        let mut d = self.s.bench_dts.clone();
        d.sort_by(|a, b| a.total_cmp(b));
        let mean = d.iter().sum::<f32>() / d.len().max(1) as f32;
        let cpu = self.s.bench_cpu.iter().sum::<f32>() / self.s.bench_cpu.len().max(1) as f32;
        println!(
            "BENCH source={:?} tris={} edges={} backend={:?} frames={} frame_ms mean={:.2} p50={:.2} p95={:.2} max={:.2} fps_mean={:.1} ui_cpu_ms_mean={:.3}",
            self.s.source,
            self.s.mesh.tri_count(),
            self.s.mesh.edge_count(),
            self.s.backend,
            d.len(),
            mean,
            percentile(&d, 0.5),
            percentile(&d, 0.95),
            d.last().copied().unwrap_or(0.0),
            1000.0 / mean.max(1e-6),
            cpu
        );
        std::process::exit(0);
    }
}

impl eframe::App for App {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let t0 = Instant::now();
        let ctx = ui.ctx().clone();

        // Frame timing (wall-clock interval between two ui() calls).
        if let Some(last) = self.s.last_frame {
            let dt = t0.duration_since(last).as_secs_f32() * 1000.0;
            self.s.dts.push_back(dt);
            if self.s.dts.len() > 60 {
                self.s.dts.pop_front();
            }
            if let Some(secs) = self.s.bench_secs {
                self.s.bench_frames += 1;
                if self.s.bench_frames > 30 {
                    let start = *self.s.bench_start.get_or_insert(t0);
                    self.s.bench_dts.push(dt);
                    self.s.bench_cpu.push(self.s.cpu_ms);
                    if t0.duration_since(start).as_secs_f32() >= secs {
                        self.finish_bench();
                    }
                }
                self.s.cam.yaw += 0.01; // auto-orbit
            }
        }
        self.s.last_frame = Some(t0);
        if self.s.continuous || self.s.bench_secs.is_some() {
            ctx.request_repaint();
        }

        egui::Panel::top("menu").show(ui, |ui| {
            egui::MenuBar::new().ui(ui, |ui| {
                ui.menu_button("Model", |ui| {
                    if ui.button("Load examples/arco-yanque.igz").clicked() {
                        self.s.load_file("examples/arco-yanque.igz");
                        ui.close();
                    }
                    if ui.button("Synthetic stress scene (~1M tris)").clicked() {
                        self.s.load_stress();
                        ui.close();
                    }
                });
                ui.menu_button("Window", |ui| {
                    for (tab, name) in [(Tab::Outliner, "Outliner"), (Tab::EntityInfo, "Entity info")] {
                        if ui.button(format!("Show {name}")).clicked() {
                            if self.dock.find_tab(&tab).is_none() {
                                self.dock.push_to_focused_leaf(tab);
                            }
                            ui.close();
                        }
                    }
                });
                ui.checkbox(&mut self.s.continuous, "Continuous repaint");
            });
        });

        egui::Panel::bottom("status").show(ui, |ui| {
            let n = self.s.dts.len().max(1) as f32;
            let ms = self.s.dts.iter().sum::<f32>() / n;
            ui.horizontal(|ui| {
                if self.s.continuous || self.s.bench_secs.is_some() {
                    ui.label(format!("{:.1} fps ({ms:.2} ms/frame)", 1000.0 / ms.max(1e-6)));
                } else {
                    ui.label(format!("last frame interval {ms:.1} ms (idle: repaints on input only)"));
                }
                ui.separator();
                ui.label(format!("UI cpu {:.2} ms", self.s.cpu_ms));
                ui.separator();
                ui.label(format!(
                    "{} triangles, {} edges, {} groups",
                    self.s.mesh.tri_count(),
                    self.s.mesh.edge_count(),
                    self.s.mesh.groups.len()
                ));
                ui.separator();
                ui.label(&self.s.backend);
            });
        });

        egui::CentralPanel::default().show(ui, |ui| {
            DockArea::new(&mut self.dock)
                .style(Style::from_egui(ui.style().as_ref()))
                .show_inside(ui, &mut Viewer { s: &mut self.s });
        });

        self.s.cpu_ms = t0.elapsed().as_secs_f32() * 1000.0;
    }
}

/// Headless AccessKit check: run one egui pass with AccessKit enabled and
/// print the tree that would be handed to the platform adapter.
fn accesskit_dump() {
    let ctx = egui::Context::default();
    ctx.enable_accesskit();
    let mut text = String::from("hello");
    let out = ctx.run_ui(egui::RawInput::default(), |ui| {
        ui.heading("Entity info");
        let _ = ui.button("Click me");
        ui.text_edit_singleline(&mut text);
        ui.checkbox(&mut true, "Continuous repaint");
    });
    match out.platform_output.accesskit_update {
        Some(update) => {
            println!("AccessKit tree update: {} nodes", update.nodes.len());
            for (id, node) in &update.nodes {
                println!("  {:?} role={:?} label={:?}", id, node.role(), node.label());
            }
        }
        None => println!("no AccessKit update produced"),
    }
}

fn main() -> eframe::Result {
    let args = parse_args();
    if args.accesskit_dump {
        accesskit_dump();
        return Ok(());
    }

    let mut wgpu_options = egui_wgpu::WgpuConfiguration::default();
    if args.novsync {
        wgpu_options.surface.present_mode = wgpu::PresentMode::AutoNoVsync;
    }
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1280.0, 800.0])
            .with_title("IngeTrazo UI spike (M0.8)"),
        renderer: eframe::Renderer::Wgpu,
        depth_buffer: 32, // Depth32Float, matches gpu::DEPTH_FORMAT
        wgpu_options,
        ..Default::default()
    };

    eframe::run_native(
        "ingetrazo-spike-ui",
        options,
        Box::new(move |cc| {
            cc.egui_ctx.enable_accesskit();
            let rs = cc.wgpu_render_state.clone().expect("wgpu render state");
            let info = rs.adapter.get_info();
            let backend = format!("{:?}: {} ({:?})", info.backend, info.name, info.device_type);
            eprintln!("wgpu adapter: {backend}");
            rs.renderer.write().callback_resources.insert(gpu::Gpu::new(&rs));

            let mut dock = DockState::new(vec![Tab::Viewport]);
            let [main, _left] = dock
                .main_surface_mut()
                .split_left(NodeIndex::root(), 0.2, vec![Tab::Outliner]);
            let [_main, _right] = dock
                .main_surface_mut()
                .split_right(main, 0.78, vec![Tab::EntityInfo]);

            let mut s = State {
                rs,
                mesh: model::Mesh::default(),
                source: String::new(),
                cam: gpu::Camera::fit([0.0; 3], 1.0),
                selected: None,
                text: String::new(),
                backend,
                last_frame: None,
                dts: VecDeque::new(),
                cpu_ms: 0.0,
                continuous: false,
                bench_secs: args.bench,
                bench_start: None,
                bench_frames: 0,
                bench_dts: Vec::new(),
                bench_cpu: Vec::new(),
            };
            if args.stress {
                s.load_stress();
            } else {
                s.load_file(&args.model);
            }
            Ok(Box::new(App { dock, s }))
        }),
    )
}
