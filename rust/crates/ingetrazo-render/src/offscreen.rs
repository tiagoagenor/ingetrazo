// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! Headless rendering: a device without a window, a render into a texture and
//! a read-back into an [`Image`]. The tests use it, and so will the PNG
//! export.
//!
//! No adapter is not an error: [`Offscreen::new`] returns `None` and callers
//! (the tests) skip. `INGETRAZO_RENDER_FALLBACK=1` asks for the software
//! adapter (WARP on Windows, llvmpipe on Linux) instead of the GPU.

use ingetrazo_model::scene::Scene;

use crate::camera::Camera;
use crate::options::{FrameStats, RenderOptions};
use crate::renderer::{Renderer, RendererConfig};
use crate::textures::TextureSource;

/// An RGBA8 image, rows top to bottom.
#[derive(Clone, Debug)]
pub struct Image {
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<u8>,
}

/// How two images differ (the corpus' `render_pixels` rule: a pixel is equal
/// when every channel is within `channel`; the images match when
/// `fraction_equal` of the pixels are).
#[derive(Clone, Copy, Debug)]
pub struct Comparison {
    pub fraction_equal: f64,
    pub max_diff: u8,
    pub pixels: usize,
}

impl Image {
    pub fn pixel(&self, x: u32, y: u32) -> [u8; 4] {
        let i = ((y * self.width + x) * 4) as usize;
        [self.rgba[i], self.rgba[i + 1], self.rgba[i + 2], self.rgba[i + 3]]
    }

    pub fn compare(&self, other: &Image, channel: u8) -> Comparison {
        assert_eq!((self.width, self.height), (other.width, other.height), "sizes differ");
        let mut equal = 0usize;
        let mut max_diff = 0u8;
        for (a, b) in self.rgba.chunks_exact(4).zip(other.rgba.chunks_exact(4)) {
            let d = (0..3).map(|k| a[k].abs_diff(b[k])).max().unwrap_or(0);
            max_diff = max_diff.max(d);
            if d <= channel {
                equal += 1;
            }
        }
        let n = self.rgba.len() / 4;
        Comparison { fraction_equal: equal as f64 / n.max(1) as f64, max_diff, pixels: n }
    }

    pub fn write_png(&self, path: &std::path::Path) -> std::io::Result<()> {
        let file = std::fs::File::create(path)?;
        let mut enc = png::Encoder::new(std::io::BufWriter::new(file), self.width, self.height);
        enc.set_color(png::ColorType::Rgba);
        enc.set_depth(png::BitDepth::Eight);
        let mut w = enc.write_header().map_err(std::io::Error::other)?;
        w.write_image_data(&self.rgba).map_err(std::io::Error::other)
    }

    pub fn read_png(path: &std::path::Path) -> std::io::Result<Image> {
        let dec = png::Decoder::new(std::io::BufReader::new(std::fs::File::open(path)?));
        let mut reader = dec.read_info().map_err(std::io::Error::other)?;
        let mut buf = vec![0; reader.output_buffer_size().unwrap_or(0)];
        let info = reader.next_frame(&mut buf).map_err(std::io::Error::other)?;
        buf.truncate(info.buffer_size());
        let rgba = match info.color_type {
            png::ColorType::Rgba => buf,
            png::ColorType::Rgb => buf.chunks_exact(3).flat_map(|p| [p[0], p[1], p[2], 255]).collect(),
            other => return Err(std::io::Error::other(format!("unsupported PNG colour type {other:?}"))),
        };
        Ok(Image { width: info.width, height: info.height, rgba })
    }
}

/// A headless wgpu device.
pub struct Offscreen {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub info: wgpu::AdapterInfo,
}

impl Offscreen {
    /// `None` when the machine has no adapter at all.
    pub fn new() -> Option<Offscreen> {
        let fallback = std::env::var("INGETRAZO_RENDER_FALLBACK").is_ok_and(|v| v == "1");
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: fallback,
            compatible_surface: None,
            ..Default::default()
        }))
        .ok()?;
        let info = adapter.get_info();
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("ingetrazo-render offscreen"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            ..Default::default()
        }))
        .ok()?;
        Some(Offscreen { device, queue, info })
    }

    pub fn renderer(&self, cfg: RendererConfig) -> Renderer {
        Renderer::new(&self.device, &self.queue, cfg)
    }

    /// Render one frame and read it back.
    #[allow(clippy::too_many_arguments)]
    pub fn render_image(
        &self,
        renderer: &mut Renderer,
        scene: &Scene,
        textures: &mut dyn TextureSource,
        camera: &Camera,
        opts: &RenderOptions,
        width: u32,
        height: u32,
    ) -> (Image, FrameStats) {
        let target = renderer.create_target(width, height);
        let view = target.create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("frame") });
        let stats = renderer.render(&mut encoder, &view, (width, height), scene, textures, camera, opts);
        let image = self.read_back(&mut encoder, &target, width, height);
        (image, stats)
    }

    /// Copy `texture` (Rgba8 or Bgra8) to the CPU; submits `encoder`.
    pub fn read_back(&self, encoder: &mut wgpu::CommandEncoder, texture: &wgpu::Texture, width: u32, height: u32) -> Image {
        let bpr = (width * 4).div_ceil(256) * 256;
        let buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: (bpr * height) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &buf,
                layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(bpr), rows_per_image: Some(height) },
            },
            wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
        );
        // `finish` consumes the encoder, which the caller still owns: swap in a fresh one.
        let fresh = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("spare") });
        let done = std::mem::replace(encoder, fresh);
        self.queue.submit([done.finish()]);
        let slice = buf.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
        rx.recv().expect("map callback").expect("map readback");
        let data = slice.get_mapped_range().expect("mapped range");
        let mut rgba = Vec::with_capacity((width * height * 4) as usize);
        for y in 0..height as usize {
            let row = &data[y * bpr as usize..y * bpr as usize + (width * 4) as usize];
            rgba.extend_from_slice(row);
        }
        drop(data);
        buf.unmap();
        Image { width, height, rgba }
    }
}
