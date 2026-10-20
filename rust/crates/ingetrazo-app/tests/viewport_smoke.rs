// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! One frame of an `examples/` model through the application's viewport
//! path: the document is opened as the app opens it, framed with Zoom
//! Extents, drawn by `GpuView::prepare` and copied by `GpuView::paint` into
//! a pass of another texture, as the egui paint callback does. Skipped when
//! the machine has no wgpu adapter (INGETRAZO_RENDER_FALLBACK=1 asks for the
//! software one).

#![allow(clippy::chunks_exact_to_as_chunks)]

use std::path::Path;

use ingetrazo_app::document::open_path;
use ingetrazo_app::nav;
use ingetrazo_app::probe::{PickCache, ScenePick};
use ingetrazo_render::offscreen::Offscreen;
use ingetrazo_render::Camera;

#[test]
fn an_example_renders_through_the_app_viewport() {
    let Some(dev) = Offscreen::new() else {
        eprintln!("SKIP: no wgpu adapter on this machine");
        return;
    };
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../examples/arco-yanque.igz");
    let (doc, textures) = open_path(&path).expect("the example opens");

    let (w, h) = (640u32, 400u32);
    let mut cam = Camera::default();
    cam.set_aspect(w, h);
    let mut cache = PickCache::default();
    let (lo, hi) = ScenePick::new(&doc.scene, &mut cache, cam.hypot).bounds().expect("the example has geometry");
    nav::fit_box(&mut cam, lo, hi, nav::FIT_MARGIN);

    let format = wgpu_format();
    let mut view = ingetrazo_app::viewport::GpuView::new(&dev.device, &dev.queue, format);
    view.set_scene(doc.scene.clone(), textures);
    let mut encoder = dev.device.create_command_encoder(&Default::default());
    view.prepare(&dev.device, &mut encoder, &cam, (w, h));
    assert!(view.stats.triangles > 0, "the frame drew nothing: {:?}", view.stats);

    // The host's pass: clear to black, then the callback's paint.
    let host = dev.device.create_texture(&eframe::egui_wgpu::wgpu::TextureDescriptor {
        label: Some("host"),
        size: eframe::egui_wgpu::wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: eframe::egui_wgpu::wgpu::TextureDimension::D2,
        format,
        usage: eframe::egui_wgpu::wgpu::TextureUsages::RENDER_ATTACHMENT | eframe::egui_wgpu::wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let host_view = host.create_view(&Default::default());
    {
        use eframe::egui_wgpu::wgpu as g;
        let mut pass = encoder.begin_render_pass(&g::RenderPassDescriptor {
            label: Some("host pass"),
            color_attachments: &[Some(g::RenderPassColorAttachment {
                view: &host_view,
                depth_slice: None,
                resolve_target: None,
                ops: g::Operations { load: g::LoadOp::Clear(g::Color::BLACK), store: g::StoreOp::Store },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        view.paint(&mut pass);
    }
    let image = dev.read_back(&mut encoder, &host, w, h);

    // Something was drawn: more than the background's colours, and not black.
    let mut colours = std::collections::HashSet::new();
    for px in image.rgba.chunks_exact(4) {
        colours.insert([px[0], px[1], px[2]]);
    }
    assert!(colours.len() > 8, "the picture is flat: {} colours", colours.len());
    assert!(image.rgba.chunks_exact(4).any(|p| p[0] != 0 || p[1] != 0 || p[2] != 0), "the picture is black");
}

fn wgpu_format() -> eframe::egui_wgpu::wgpu::TextureFormat {
    eframe::egui_wgpu::wgpu::TextureFormat::Rgba8Unorm
}
