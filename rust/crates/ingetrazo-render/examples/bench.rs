// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! Frame-time measurements of the renderer on a headless device.
//!
//!     cargo run -p ingetrazo-render --release --example bench -- \
//!         [--model examples/arco-yanque.igz] [--stress] [--frames 300] \
//!         [--size 1280x800] [--png out.png]
//!
//! A frame here is what the renderer does per viewport repaint: scene sync
//! (a no-op after the first), the frame plan, encoding, submission and the
//! GPU executing it (the wait is the end of the measurement), with the
//! camera orbiting. Nothing is read back except in the last frame when
//! `--png` is given.

use std::time::Instant;

use ingetrazo_math::{M4, V3};
use ingetrazo_model::mesh::Mesh;
use ingetrazo_model::scene::{Group, MeshId, Scene};
use ingetrazo_render::document::open_igz;
use ingetrazo_render::offscreen::Offscreen;
use ingetrazo_render::{Camera, NoTextures, RenderOptions, RendererConfig, TextureSource};

/// The spike's synthetic scene: 45 x 45 extruded 128-gon prisms
/// (1,028,700 triangles, 777,600 edges), here as 2025 placements of one
/// component.
fn stress_scene(soft_sides: bool) -> Scene {
    let mut sc = Scene::new();
    let mut proto = Mesh::new();
    let n = 128usize;
    let ring: Vec<V3> = (0..n)
        .map(|i| {
            let a = std::f32::consts::TAU * i as f32 / n as f32;
            V3(a.cos(), a.sin(), 0.0)
        })
        .collect();
    let top: Vec<V3> = ring.iter().map(|p| V3(p.0, p.1, 2.0)).collect();
    let bottom: Vec<V3> = ring.iter().rev().copied().collect();
    proto.add_face(&bottom, &[]);
    proto.add_face(&top, &[]);
    for i in 0..n {
        let j = (i + 1) % n;
        proto.add_face(&[ring[i], ring[j], top[j], top[i]], &[]);
    }
    if soft_sides {
        let ids: Vec<_> = proto.edges.clone();
        for e in ids {
            let (a, b) = (proto.edge(e).v0, proto.edge(e).v1);
            if (proto.position(a).2 - proto.position(b).2).abs() > 1e-6 {
                proto.edge_mut(e).soft = true;
            }
        }
    }
    sc.meshes.push(proto);
    for gx in 0..45 {
        for gy in 0..45 {
            let mut m = M4::identity();
            m.translate(gx as f32 * 3.0, gy as f32 * 3.0, 0.0);
            let mut g = Group::new(MeshId(1), format!("p{gx}-{gy}"));
            g.xform = Some(m);
            sc.groups.push(g);
        }
    }
    sc
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let flag = |name: &str| args.iter().any(|a| a == name);
    let value = |name: &str| args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).cloned();
    let frames: usize = value("--frames").and_then(|v| v.parse().ok()).unwrap_or(300);
    let (w, h) = value("--size")
        .and_then(|s| {
            let (a, b) = s.split_once('x')?;
            Some((a.parse().ok()?, b.parse().ok()?))
        })
        .unwrap_or((1280u32, 800u32));

    let Some(dev) = Offscreen::new() else {
        eprintln!("no wgpu adapter");
        return;
    };
    println!("adapter: {} ({:?})", dev.info.name, dev.info.backend);

    let t0 = Instant::now();
    let (scene, mut textures): (Scene, Box<dyn TextureSource>) = if flag("--stress") {
        (stress_scene(flag("--soft")), Box::new(NoTextures))
    } else {
        let path = value("--model").unwrap_or_else(|| "examples/arco-yanque.igz".to_string());
        let raw = std::fs::read(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
        let (sc, tex) = open_igz(&raw).unwrap_or_else(|e| panic!("{path}: {e}"));
        (sc, Box::new(tex))
    };
    println!("scene ready in {:.0} ms ({} meshes, {} groups)", t0.elapsed().as_secs_f64() * 1e3, scene.meshes.len(), scene.groups.len());

    let mut r = dev.renderer(RendererConfig::default());
    let opts = RenderOptions::default();
    let t1 = Instant::now();
    let stats = r.sync(&scene, textures.as_mut(), &opts);
    println!(
        "first sync: {:.0} ms, {} prototypes rebuilt, {} instances, {:.1} MB uploaded",
        t1.elapsed().as_secs_f64() * 1e3,
        stats.rebuilt,
        stats.instances,
        stats.uploaded_bytes as f64 / 1e6
    );
    let (lo, hi) = r.scene_bounds().expect("empty scene");
    let mut cam = Camera::default();
    cam.fit_to(V3(lo[0], lo[1], lo[2]), V3(hi[0], hi[1], hi[2]), 1.3);

    if let Some(z) = value("--znear").and_then(|v| v.parse().ok()) {
        cam.znear = z;
    }
    if let Some(z) = value("--zfar").and_then(|v| v.parse().ok()) {
        cam.zfar = z;
    }
    let target = r.create_target(w, h);
    let view = target.create_view(&wgpu::TextureViewDescriptor::default());
    let mut times = Vec::with_capacity(frames);
    let mut last = None;
    for i in 0..frames + 30 {
        cam.yaw = -0.8 + i as f64 * 0.02;
        let t = Instant::now();
        let mut enc = dev.device.create_command_encoder(&Default::default());
        let st = r.render(&mut enc, &view, (w, h), &scene, textures.as_mut(), &cam, &opts);
        dev.queue.submit([enc.finish()]);
        let _ = dev.device.poll(wgpu::PollType::wait_indefinitely());
        let ms = t.elapsed().as_secs_f64() * 1e3;
        if i >= 30 {
            times.push(ms);
        }
        last = Some(st);
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mean = times.iter().sum::<f64>() / times.len() as f64;
    let p95 = times[(times.len() as f64 * 0.95) as usize];
    let st = last.unwrap();
    println!(
        "{w}x{h}, {frames} frames: mean {mean:.2} ms, p95 {p95:.2} ms, max {:.2} ms ({:.0} fps); {} draw calls, {} triangles submitted",
        times.last().unwrap(),
        1000.0 / mean,
        st.draw_calls,
        st.triangles
    );

    if let Some(path) = value("--png") {
        let mut enc = dev.device.create_command_encoder(&Default::default());
        r.render(&mut enc, &view, (w, h), &scene, textures.as_mut(), &cam, &opts);
        let img = dev.read_back(&mut enc, &target, w, h);
        img.write_png(std::path::Path::new(&path)).expect("png");
        println!("wrote {path}");
    }
}
