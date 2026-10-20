// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! Drawing a rendered target into somebody else's render pass.
//!
//! egui's pass is single-sampled and owns its depth buffer, so the viewport
//! cannot share it: the renderer draws into its own target (during the
//! callback's `prepare`) and the callback's `paint` copies that texture to
//! the viewport rectangle with a [`Blitter`]. No egui types are involved.

use crate::renderer::texture_bgl;

pub struct Blitter {
    pipeline: wgpu::RenderPipeline,
    bgl: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
}

/// A target prepared for blitting.
pub struct BlitSource {
    bind_group: wgpu::BindGroup,
}

impl Blitter {
    /// `target_format` is the format of the pass the blit is drawn into.
    pub fn new(device: &wgpu::Device, target_format: wgpu::TextureFormat) -> Blitter {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ingetrazo-render blit"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
        });
        let bgl = texture_bgl(device);
        // The shader reads the texture at group 1; group 0 stays empty.
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("blit layout"),
            bind_group_layouts: &[None, Some(&bgl)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("blit"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vs_blit"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("fs_blit"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: target_format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });
        Blitter { pipeline, bgl, sampler }
    }

    /// Bind a rendered texture (re-create when the target is re-created).
    pub fn source(&self, device: &wgpu::Device, view: &wgpu::TextureView) -> BlitSource {
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("blit source"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.sampler) },
            ],
        });
        BlitSource { bind_group }
    }

    /// Draw the source over the pass's current viewport.
    pub fn draw(&self, pass: &mut wgpu::RenderPass<'_>, source: &BlitSource) {
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(1, &source.bind_group, &[]);
        pass.draw(0..3, 0..1);
    }
}
