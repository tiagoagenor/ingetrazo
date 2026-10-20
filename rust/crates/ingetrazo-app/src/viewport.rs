// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! The viewport on top of `ingetrazo-render`: the renderer draws the scene
//! into its own (multisampled) target while the frame is prepared, and the
//! paint step blits that texture into the host's pass. [`GpuView`] knows
//! nothing about egui (the tests drive it over a headless device);
//! [`ViewCallback`] is the egui_wgpu glue.
//!
//! The scene is never copied: [`GpuView::prepare`] reads the session's own
//! `&Scene`, and the renderer rebakes only the meshes whose serial moved
//! (M4.9; the first version cloned the whole scene on every edit). That is
//! why the frame is drawn from the window's `ui` pass (where the scene can be
//! borrowed) with [`GpuView::render_now`], not from the paint callback's
//! `prepare` (which cannot borrow it): the callback only blits.

use eframe::egui;
use eframe::egui_wgpu::{self, wgpu};
use ingetrazo_model::scene::Scene;
use ingetrazo_render::blit::{BlitSource, Blitter};
use ingetrazo_render::{Camera, FrameStats, NoTextures, RenderOptions, Renderer, RendererConfig, TextureSource};

struct Target {
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    source: BlitSource,
    size: (u32, u32),
}

/// A texture source the callback may own.
pub type Textures = Box<dyn TextureSource + Send + Sync>;

/// Everything the draw needs, kept in egui's `callback_resources`.
pub struct GpuView {
    pub renderer: Renderer,
    pub textures: Textures,
    pub opts: RenderOptions,
    pub stats: FrameStats,
    /// CPU milliseconds of the last [`GpuView::prepare`].
    pub prepare_ms: f32,
    blitter: Blitter,
    target: Option<Target>,
}

impl GpuView {
    /// `target_format` is the format of the pass [`GpuView::paint`] draws in.
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, target_format: wgpu::TextureFormat) -> GpuView {
        GpuView {
            renderer: Renderer::new(device, queue, RendererConfig::default()),
            textures: Box::new(NoTextures),
            opts: RenderOptions::default(),
            stats: FrameStats::default(),
            prepare_ms: 0.0,
            blitter: Blitter::new(device, target_format),
            target: None,
        }
    }

    /// A document was opened (or the scene object was replaced): the
    /// renderer forgets every baked mesh, the textures are the new ones.
    pub fn open(&mut self, textures: Textures) {
        self.textures = textures;
        self.renderer.invalidate();
    }

    /// What to highlight: the selection and the one under the cursor.
    pub fn set_cues(&mut self, selection: ingetrazo_render::Selection, hover: ingetrazo_render::Hover) {
        self.opts.selection = selection;
        self.opts.hover = hover;
    }

    /// Draw `scene` as `camera` sees it, into the view's own target of `px`
    /// pixels (re-created when the size changes). The renderer rebakes what
    /// the meshes' serials say changed.
    pub fn prepare(&mut self, device: &wgpu::Device, encoder: &mut wgpu::CommandEncoder, scene: &Scene, camera: &Camera, px: (u32, u32)) {
        let px = (px.0.max(1), px.1.max(1));
        if self.target.as_ref().is_none_or(|t| t.size != px) {
            let texture = self.renderer.create_target(px.0, px.1);
            let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
            let source = self.blitter.source(device, &view);
            self.target = Some(Target { texture, view, source, size: px });
        }
        let t0 = std::time::Instant::now();
        if let Some(t) = &self.target {
            self.stats = self.renderer.render(encoder, &t.view, px, scene, self.textures.as_mut(), camera, &self.opts);
            let _ = &t.texture;
        }
        self.prepare_ms = t0.elapsed().as_secs_f32() * 1000.0;
    }

    /// [`GpuView::prepare`] with its own encoder, submitted at once: the
    /// picture is ready before the host's pass is recorded.
    pub fn render_now(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, scene: &Scene, camera: &Camera, px: (u32, u32)) {
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("ingetrazo viewport") });
        self.prepare(device, &mut encoder, scene, camera, px);
        queue.submit(std::iter::once(encoder.finish()));
    }

    /// Copy the prepared picture into the host's pass.
    pub fn paint(&self, pass: &mut wgpu::RenderPass<'_>) {
        if let Some(t) = &self.target {
            self.blitter.draw(pass, &t.source);
        }
    }
}

/// The paint callback of one frame: it blits what [`GpuView::render_now`]
/// drew.
pub struct ViewCallback;

impl egui_wgpu::CallbackTrait for ViewCallback {
    fn paint(&self, _info: egui::PaintCallbackInfo, pass: &mut wgpu::RenderPass<'static>, resources: &egui_wgpu::CallbackResources) {
        if let Some(v) = resources.get::<GpuView>() {
            v.paint(pass);
        }
    }
}
