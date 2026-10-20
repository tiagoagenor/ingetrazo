//! The viewport on top of `ingetrazo-render`: the renderer draws the scene
//! into its own (multisampled) target during the callback's `prepare`, and
//! `paint` blits that texture into egui's pass. The renderer itself knows
//! nothing about egui.

use eframe::egui_wgpu::{self, wgpu};
use eframe::egui;
use ingetrazo_model::scene::Scene;
use ingetrazo_render::blit::{BlitSource, Blitter};
use ingetrazo_render::{Camera, FrameStats, RenderOptions, Renderer, RendererConfig, TextureSource};

struct Target {
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    source: BlitSource,
    size: (u32, u32),
}

/// Everything the callback needs, kept in egui's `callback_resources`.
pub struct View {
    pub renderer: Renderer,
    pub scene: Scene,
    pub textures: Box<dyn TextureSource + Send + Sync>,
    pub opts: RenderOptions,
    pub stats: FrameStats,
    blitter: Blitter,
    target: Option<Target>,
}

impl View {
    pub fn new(rs: &egui_wgpu::RenderState) -> View {
        View {
            renderer: Renderer::new(&rs.device, &rs.queue, RendererConfig::default()),
            scene: Scene::new(),
            textures: Box::new(ingetrazo_render::NoTextures),
            opts: RenderOptions::default(),
            stats: FrameStats::default(),
            blitter: Blitter::new(&rs.device, rs.target_format),
            target: None,
        }
    }

    /// Replace the scene (a document was opened).
    pub fn set_scene(&mut self, scene: Scene, textures: Box<dyn TextureSource + Send + Sync>) -> Camera {
        self.scene = scene;
        self.textures = textures;
        self.renderer.invalidate();
        self.stats = self.renderer.sync(&self.scene, self.textures.as_mut(), &self.opts);
        let mut cam = Camera::default();
        if let Some((lo, hi)) = self.renderer.scene_bounds() {
            cam.fit_to(
                ingetrazo_math::V3(lo[0], lo[1], lo[2]),
                ingetrazo_math::V3(hi[0], hi[1], hi[2]),
                1.3,
            );
        }
        cam
    }

    fn prepare(&mut self, device: &wgpu::Device, encoder: &mut wgpu::CommandEncoder, camera: &Camera, px: (u32, u32)) {
        let px = (px.0.max(1), px.1.max(1));
        if self.target.as_ref().is_none_or(|t| t.size != px) {
            let texture = self.renderer.create_target(px.0, px.1);
            let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
            let source = self.blitter.source(device, &view);
            self.target = Some(Target { texture, view, source, size: px });
        }
        let t = self.target.as_ref().unwrap();
        self.stats = self.renderer.render(encoder, &t.view, px, &self.scene, self.textures.as_mut(), camera, &self.opts);
        let _ = &t.texture;
    }

    fn paint(&self, pass: &mut wgpu::RenderPass<'_>) {
        if let Some(t) = &self.target {
            self.blitter.draw(pass, &t.source);
        }
    }
}

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
        if let Some(v) = resources.get_mut::<View>() {
            v.prepare(device, encoder, &self.camera, self.px);
        }
        Vec::new()
    }

    fn paint(
        &self,
        _info: egui::PaintCallbackInfo,
        pass: &mut wgpu::RenderPass<'static>,
        resources: &egui_wgpu::CallbackResources,
    ) {
        if let Some(v) = resources.get::<View>() {
            v.paint(pass);
        }
    }
}
