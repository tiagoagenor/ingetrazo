// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! Shared helpers of the render tests.
#![allow(dead_code)]

use ingetrazo_formats::pyjson::Value;
use ingetrazo_math::V3;
use ingetrazo_model::mesh::Mesh;
use ingetrazo_model::scene::{Group, MeshId, Scene};
use ingetrazo_render::offscreen::Offscreen;
use ingetrazo_render::{Camera, NoTextures, RenderOptions, Renderer, RendererConfig, Style};

/// One GPU test at a time: several devices of one process polling the same
/// driver concurrently is how a test run hung once.
static GPU: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// A headless device that holds the GPU lock while it lives.
pub struct Device {
    off: Offscreen,
    _guard: std::sync::MutexGuard<'static, ()>,
}

impl std::ops::Deref for Device {
    type Target = Offscreen;
    fn deref(&self) -> &Offscreen {
        &self.off
    }
}

/// The headless device, or `None` (the test then returns early and says so).
pub fn device() -> Option<Device> {
    let guard = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let o = Offscreen::new();
    if o.is_none() {
        eprintln!("SKIP: no wgpu adapter on this machine (set INGETRAZO_RENDER_FALLBACK=1 for the software one)");
    }
    o.map(|off| Device { off, _guard: guard })
}

pub fn cube_into(mesh: &mut Mesh, origin: V3, size: f32) {
    let o = origin;
    let p = |x: f32, y: f32, z: f32| V3(o.0 + x * size, o.1 + y * size, o.2 + z * size);
    let q = [
        p(0., 0., 0.), p(1., 0., 0.), p(1., 1., 0.), p(0., 1., 0.),
        p(0., 0., 1.), p(1., 0., 1.), p(1., 1., 1.), p(0., 1., 1.),
    ];
    // Outward windings.
    let faces: [[usize; 4]; 6] = [
        [0, 3, 2, 1], [4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7],
    ];
    for f in faces {
        let ring: Vec<V3> = f.iter().map(|&i| q[i]).collect();
        mesh.add_face(&ring, &[]);
    }
    for (a, b) in [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)] {
        let _ = mesh.add_edge(q[a], q[b]);
    }
}

pub fn cube_scene() -> Scene {
    let mut sc = Scene::new();
    cube_into(&mut sc.meshes[0], V3(0.0, 0.0, 0.0), 2.0);
    sc
}

pub fn paint_all(mesh: &mut Mesh, rgb: [f64; 3]) {
    let ids: Vec<_> = mesh.faces.clone();
    for f in ids {
        mesh.face_mut(f).attrs.push((
            "color".into(),
            Value::Array(rgb.iter().map(|&c| Value::Float(c)).collect()),
        ));
    }
}

/// A component instance of mesh `proto` placed by `m`.
pub fn instance(proto: MeshId, m: ingetrazo_math::M4, name: &str) -> Group {
    let mut g = Group::new(proto, name.into());
    g.xform = Some(m);
    g
}

pub fn translation(x: f32, y: f32, z: f32) -> ingetrazo_math::M4 {
    let mut m = ingetrazo_math::M4::identity();
    m.translate(x, y, z);
    m
}

pub fn view_of(sc: &Scene, r: &mut Renderer) -> Camera {
    let _ = sc;
    let (lo, hi) = r.scene_bounds().expect("bounds");
    let mut cam = Camera::default();
    cam.fit_to(V3(lo[0], lo[1], lo[2]), V3(hi[0], hi[1], hi[2]), 1.3);
    cam
}

pub fn render(
    dev: &Offscreen,
    r: &mut Renderer,
    sc: &Scene,
    cam: &Camera,
    opts: &RenderOptions,
    w: u32,
    h: u32,
) -> ingetrazo_render::offscreen::Image {
    // The scene may be another one with the same serials as the last.
    r.invalidate();
    let (img, _) = dev.render_image(r, sc, &mut NoTextures, cam, opts, w, h);
    img
}

pub fn style(name: &str) -> RenderOptions {
    RenderOptions { style: Some(Style::builtin(name).expect("builtin style")), ..Default::default() }
}

pub fn new_renderer(dev: &Offscreen) -> Renderer {
    dev.renderer(RendererConfig::default())
}

pub fn out_dir() -> std::path::PathBuf {
    let d = std::env::var("INGETRAZO_RENDER_OUT").map(std::path::PathBuf::from).unwrap_or_else(|_| {
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/render-out")
    });
    std::fs::create_dir_all(&d).ok();
    d
}
