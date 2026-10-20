// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! The viewport on top of `ingetrazo-render`: the renderer draws the scene
//! into its own (multisampled) target while the frame is prepared, and the
//! paint step blits that texture into the host's pass. [`GpuView`] knows
//! nothing about egui (the tests drive it over a headless device);
//! [`ViewCallback`] is the egui_wgpu glue.

use std::sync::Arc;

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
    pub scene: Arc<Scene>,
    pub textures: Textures,
    pub opts: RenderOptions,
    pub stats: FrameStats,
    blitter: Blitter,
    target: Option<Target>,
}

impl GpuView {
    /// `target_format` is the format of the pass [`GpuView::paint`] draws in.
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, target_format: wgpu::TextureFormat) -> GpuView {
        GpuView {
            renderer: Renderer::new(device, queue, RendererConfig::default()),
            scene: Arc::new(Scene::new()),
            textures: Box::new(NoTextures),
            opts: RenderOptions::default(),
            stats: FrameStats::default(),
            blitter: Blitter::new(device, target_format),
            target: None,
        }
    }

    /// A document was opened (or the scene was replaced).
    pub fn set_scene(&mut self, scene: Arc<Scene>, textures: Textures) {
        self.scene = scene;
        self.textures = textures;
        self.renderer.invalidate();
    }

    /// The model was edited: the renderer rebakes what a mesh's serial says
    /// changed (no textures or cache are dropped).
    pub fn update_scene(&mut self, scene: Arc<Scene>) {
        self.scene = scene;
    }

    /// What to highlight: the selection and the one under the cursor.
    pub fn set_cues(&mut self, selection: ingetrazo_render::Selection, hover: ingetrazo_render::Hover) {
        self.opts.selection = selection;
        self.opts.hover = hover;
    }

    /// Draw the scene as `camera` sees it, into the view's own target of
    /// `px` pixels (re-created when the size changes).
    pub fn prepare(&mut self, device: &wgpu::Device, encoder: &mut wgpu::CommandEncoder, camera: &Camera, px: (u32, u32)) {
        let px = (px.0.max(1), px.1.max(1));
        if self.target.as_ref().is_none_or(|t| t.size != px) {
            let texture = self.renderer.create_target(px.0, px.1);
            let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
            let source = self.blitter.source(device, &view);
            self.target = Some(Target { texture, view, source, size: px });
        }
        if let Some(t) = &self.target {
            self.stats = self.renderer.render(encoder, &t.view, px, &self.scene, self.textures.as_mut(), camera, &self.opts);
            let _ = &t.texture;
        }
    }

    /// Copy the prepared picture into the host's pass.
    pub fn paint(&self, pass: &mut wgpu::RenderPass<'_>) {
        if let Some(t) = &self.target {
            self.blitter.draw(pass, &t.source);
        }
    }
}

/// The paint callback of one frame.
pub struct ViewCallback {
    pub camera: Camera,
    /// The viewport rectangle in physical pixels.
    pub px: (u32, u32),
}

impl egui_wgpu::CallbackTrait for ViewCallback {
    fn prepare(
        &self,
        device: &wgpu::Device,
        _queue: &wgpu::Queue,
        _screen: &egui_wgpu::ScreenDescriptor,
        encoder: &mut wgpu::CommandEncoder,
        resources: &mut egui_wgpu::CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        if let Some(v) = resources.get_mut::<GpuView>() {
            v.prepare(device, encoder, &self.camera, self.px);
        }
        Vec::new()
    }

    fn paint(&self, _info: egui::PaintCallbackInfo, pass: &mut wgpu::RenderPass<'static>, resources: &egui_wgpu::CallbackResources) {
        if let Some(v) = resources.get::<GpuView>() {
            v.paint(pass);
        }
    }
}
