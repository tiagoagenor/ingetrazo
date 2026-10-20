// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! The renderer: GPU resources, the scene sync and the frame.
//!
//! One [`Renderer`] draws one viewport. It owns its colour (multisampled),
//! depth and resolve targets, so it never depends on the host's render pass:
//! it renders into any texture view of the configured format, and a host that
//! cannot hand over a pass (egui) blits the result with [`crate::blit`].
//!
//! Components are instanced: every distinct (mesh, paint) pair is baked once
//! into its own vertex buffers, and each placement is one entry of a per
//! prototype instance buffer (a world matrix plus a few flags).

use std::cell::RefCell;
use std::collections::HashMap;

use ingetrazo_math::{M4, V3};
use ingetrazo_model::scene::{MeshId, Scene};
use wgpu::util::DeviceExt;

use crate::bake::{
    bake_mesh, face_tris, selection_dot_colors, BakeEnv, BakeKind, EdgeVertex, FaceVertex, LayerTable, MeshBake,
    Run, RunClass, SoftVertex,
};
use crate::billboard::{self, Billboard};
use crate::camera::{gl_to_wgpu, Camera};
use crate::frame::*;
use crate::options::{FrameStats, RenderOptions};
use crate::overlay::{self, AXIS_COLORS};
use crate::plan::{collect_placements, GroupPath, PlanOptions, Placement};
use crate::section;
use crate::style::{
    edit_fade_color, rgb_of, FaceMode, Rgb, Style, EDIT_REST_FADE, XRAY_FACE_OPACITY, XRAY_HIDDEN_EDGE_FADE,
};
use crate::textures::{mip_chain, TexInfo, TextureSource};

/// Depth format of the renderer's own depth buffer.
pub const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

/// The orange of the selection cues.
pub const SELECTION_COLOR: Rgb = [0.95, 0.45, 0.16];
/// The light blue of the hovered edge.
pub const HOVER_COLOR: Rgb = [0.30, 0.55, 0.95];

#[derive(Clone, Copy, Debug)]
pub struct RendererConfig {
    /// Format of the targets handed to [`Renderer::render`].
    pub color_format: wgpu::TextureFormat,
    /// MSAA samples (1 or 4); the Python viewport defaults to 4.
    pub samples: u32,
}

impl Default for RendererConfig {
    fn default() -> RendererConfig {
        RendererConfig { color_format: wgpu::TextureFormat::Rgba8Unorm, samples: 4 }
    }
}

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
struct ProtoKey {
    mesh: MeshId,
    paint: String,
    ghost: bool,
    ghost_geometry: bool,
}

#[derive(Clone, PartialEq, Eq, Debug)]
struct Stamp {
    serial: u64,
    sizes: (usize, usize, usize),
    env: u64,
}

struct ProtoEntry {
    stamp: Stamp,
    runs: Vec<Run>,
    tex_paths: Vec<String>,
    tex_slot: Vec<Option<usize>>,
    faces: Option<wgpu::Buffer>,
    hard: Option<wgpu::Buffer>,
    soft: Option<wgpu::Buffer>,
    ghost: Option<wgpu::Buffer>,
    n_hard: u32,
    n_soft: u32,
    n_ghost: u32,
    tris: usize,
    lo: [f32; 3],
    hi: [f32; 3],
}

/// One prototype's instances in this frame.
struct ProtoDraw {
    proto: usize,
    inst: (u32, u32),
    /// World centres of its placements, for the translucent sort.
    centers: Vec<V3>,
}

struct TexGpu {
    bind_group: wgpu::BindGroup,
}

/// A buffer that grows when the data does not fit.
#[derive(Default)]
struct GrowBuf {
    buf: Option<wgpu::Buffer>,
    cap: u64,
}

impl GrowBuf {
    fn write(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, usage: wgpu::BufferUsages, bytes: &[u8]) {
        let need = (bytes.len() as u64).max(16);
        if self.buf.is_none() || self.cap < need {
            let cap = need.next_power_of_two();
            self.buf = Some(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("ingetrazo-render dynamic"),
                size: cap,
                usage: usage | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
            self.cap = cap;
        }
        if !bytes.is_empty() {
            queue.write_buffer(self.buf.as_ref().unwrap(), 0, bytes);
        }
    }
}

struct Targets {
    size: (u32, u32),
    msaa: Option<wgpu::TextureView>,
    depth: wgpu::TextureView,
}

pub struct Renderer {
    device: wgpu::Device,
    queue: wgpu::Queue,
    cfg: RendererConfig,
    module: wgpu::ShaderModule,
    params_bgl: wgpu::BindGroupLayout,
    tex_bgl: wgpu::BindGroupLayout,
    layout: wgpu::PipelineLayout,
    pipelines: HashMap<PipeKey, wgpu::RenderPipeline>,
    stride: u64,
    params_buf: GrowBuf,
    params_bg: Option<(wgpu::BindGroup, u64)>,
    inst_buf: GrowBuf,
    eye_buf: GrowBuf,
    ident_inst: wgpu::Buffer,
    ident_eye: wgpu::Buffer,
    dyn_face: GrowBuf,
    dyn_edge: GrowBuf,
    sky_tri: GrowBuf,
    sky_line: GrowBuf,
    targets: Option<Targets>,
    protos: Vec<ProtoEntry>,
    proto_index: HashMap<ProtoKey, usize>,
    tex_cache: HashMap<String, TexInfo>,
    tex_gpu: Vec<TexGpu>,
    tex_index: HashMap<String, usize>,
    dummy_bg: wgpu::BindGroup,
    sampler_mip: wgpu::Sampler,
    sampler_nomip: wgpu::Sampler,
    draws: Vec<ProtoDraw>,
    placements: Vec<Placement>,
    billboards: Vec<Billboard>,
    chord_cache: std::sync::Mutex<Option<(u64, std::sync::Arc<Vec<crate::section::Chord>>)>>,
    inst_matrices: Vec<M4>,
    sync_stats: FrameStats,
    last_plan: FramePlan,
}

fn size_of_params() -> u64 {
    std::mem::size_of::<ParamsData>() as u64
}

impl Renderer {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, cfg: RendererConfig) -> Renderer {
        let device = device.clone();
        let queue = queue.clone();
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ingetrazo-render"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
        });
        let params_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("params"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: true,
                    min_binding_size: wgpu::BufferSize::new(size_of_params()),
                },
                count: None,
            }],
        });
        let tex_bgl = texture_bgl(&device);
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("ingetrazo-render layout"),
            bind_group_layouts: &[Some(&params_bgl), Some(&tex_bgl)],
            immediate_size: 0,
        });
        let align = device.limits().min_uniform_buffer_offset_alignment as u64;
        let stride = size_of_params().div_ceil(align) * align;
        let sampler = |mip: bool| {
            device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some(if mip { "texture sampler" } else { "cutout sampler" }),
                address_mode_u: wgpu::AddressMode::Repeat,
                address_mode_v: wgpu::AddressMode::Repeat,
                address_mode_w: wgpu::AddressMode::Repeat,
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                mipmap_filter: wgpu::MipmapFilterMode::Linear,
                lod_max_clamp: if mip { 32.0 } else { 0.0 },
                ..Default::default()
            })
        };
        let sampler_mip = sampler(true);
        let sampler_nomip = sampler(false);
        let white = device.create_texture_with_data(
            &queue,
            &wgpu::TextureDescriptor {
                label: Some("white"),
                size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            },
            wgpu::util::TextureDataOrder::LayerMajor,
            &[255, 255, 255, 255],
        );
        let dummy_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("dummy texture"),
            layout: &tex_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(
                        &white.create_view(&wgpu::TextureViewDescriptor::default()),
                    ),
                },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&sampler_mip) },
            ],
        });
        let identity: [f32; 20] = {
            let mut a = [0.0f32; 20];
            a[0] = 1.0;
            a[5] = 1.0;
            a[10] = 1.0;
            a[15] = 1.0;
            a
        };
        let ident_inst = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("identity instance"),
            contents: bytemuck::cast_slice(&identity),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let ident_eye = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("identity eye"),
            contents: bytemuck::cast_slice(&[0.0f32; 4]),
            usage: wgpu::BufferUsages::VERTEX,
        });
        Renderer {
            device,
            queue,
            cfg,
            module,
            params_bgl,
            tex_bgl,
            layout,
            pipelines: HashMap::new(),
            stride,
            params_buf: GrowBuf::default(),
            params_bg: None,
            inst_buf: GrowBuf::default(),
            eye_buf: GrowBuf::default(),
            ident_inst,
            ident_eye,
            dyn_face: GrowBuf::default(),
            dyn_edge: GrowBuf::default(),
            sky_tri: GrowBuf::default(),
            sky_line: GrowBuf::default(),
            targets: None,
            protos: Vec::new(),
            proto_index: HashMap::new(),
            tex_cache: HashMap::new(),
            tex_gpu: Vec::new(),
            tex_index: HashMap::new(),
            dummy_bg,
            sampler_mip,
            sampler_nomip,
            draws: Vec::new(),
            placements: Vec::new(),
            billboards: Vec::new(),
            chord_cache: std::sync::Mutex::new(None),
            inst_matrices: Vec::new(),
            sync_stats: FrameStats::default(),
            last_plan: FramePlan::default(),
        }
    }

    pub fn config(&self) -> RendererConfig {
        self.cfg
    }

    /// The placements of the last sync.
    pub fn placements(&self) -> &[Placement] {
        &self.placements
    }

    pub(crate) fn chord_cache(&self) -> &std::sync::Mutex<Option<(u64, std::sync::Arc<Vec<crate::section::Chord>>)>> {
        &self.chord_cache
    }

    /// Forget every baked mesh and texture (call when the scene object is
    /// replaced, e.g. a document was opened: serials restart from zero).
    pub fn invalidate(&mut self) {
        self.protos.clear();
        self.proto_index.clear();
        self.tex_cache.clear();
        self.tex_gpu.clear();
        self.tex_index.clear();
        self.draws.clear();
        self.placements.clear();
        self.billboards.clear();
    }

    /// The command list of the last frame (for tests and diagnostics).
    pub fn last_plan(&self) -> &FramePlan {
        &self.last_plan
    }

    /// Triangle and edge counts of the baked prototypes, in bake order.
    pub fn prototype_stats(&self) -> Vec<(usize, usize, usize)> {
        self.protos.iter().map(|p| (p.tris, (p.n_hard / 2) as usize, (p.n_soft / 2) as usize)).collect()
    }

    /// A target texture the renderer can draw into and a host can sample.
    pub fn create_target(&self, width: u32, height: u32) -> wgpu::Texture {
        self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("ingetrazo-render target"),
            size: wgpu::Extent3d { width: width.max(1), height: height.max(1), depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: self.cfg.color_format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        })
    }

    // ---- sync ------------------------------------------------------------------------

    /// Bring the GPU copies up to date with the scene. Cheap when nothing
    /// changed: a prototype is rebaked only when its mesh's serial (or the
    /// layer table) moved.
    pub fn sync(&mut self, scene: &Scene, textures: &mut dyn TextureSource, opts: &RenderOptions) -> FrameStats {
        let mut stats = FrameStats::default();
        let layers = LayerTable::from_scene(scene);
        let show_objects = scene.extra.show_hidden_objects;
        let ghost_geometry = scene.extra.show_hidden_geometry;
        let plan_opts = PlanOptions {
            show_hidden_objects: show_objects,
            edit_path: opts.edit_path.clone(),
            edit_rest: opts.edit_rest,
        };
        self.placements = collect_placements(scene, &layers, &plan_opts);
        self.billboards = billboard::collect(scene, &layers);

        // The environment a bake depends on.
        let env_hash = {
            let mut h: u64 = ghost_geometry as u64;
            for l in &scene.layers {
                if let (Some(ingetrazo_formats::pyjson::Value::Str(n)), Some(v)) = (l.get("name"), l.get("visible")) {
                    if !ingetrazo_model::pyval::truthy(v) {
                        for b in n.bytes() {
                            h = h.wrapping_mul(1_099_511_628_211).wrapping_add(b as u64 + 1);
                        }
                    }
                }
            }
            h
        };

        // Group the placements by prototype.
        let mut order: Vec<ProtoKey> = Vec::new();
        let mut members: HashMap<ProtoKey, Vec<usize>> = HashMap::new();
        for (i, p) in self.placements.iter().enumerate() {
            let key = ProtoKey {
                mesh: p.mesh,
                paint: p.paint.as_ref().map(|v| format!("{v:?}")).unwrap_or_default(),
                ghost: p.ghost,
                ghost_geometry: ghost_geometry && p.path.is_empty() && !p.ghost,
            };
            members.entry(key.clone()).or_insert_with(|| {
                order.push(key.clone());
                Vec::new()
            });
            members.get_mut(&key).unwrap().push(i);
        }

        // Bake what changed.
        let loader = RefCell::new(TexLoader { cache: &mut self.tex_cache, src: textures });
        let mut used_protos: Vec<usize> = Vec::new();
        let mut new_protos: Vec<(usize, ProtoEntry)> = Vec::new();
        for key in &order {
            let mesh = scene.mesh(key.mesh);
            let stamp = Stamp { serial: mesh.serial, sizes: mesh.arena_sizes(), env: env_hash };
            if let Some(&ix) = self.proto_index.get(key) {
                if self.protos[ix].stamp == stamp {
                    used_protos.push(ix);
                    continue;
                }
            }
            let paint = members[key].first().and_then(|&i| self.placements[i].paint.clone());
            let cutout = |p: &str| loader.borrow_mut().info(p).has_cutout;
            let env = BakeEnv { layers: &layers, cutout: &cutout, ghost_geometry: key.ghost_geometry };
            let kind = if key.ghost { BakeKind::GhostObject } else { BakeKind::Normal };
            let bk = bake_mesh(mesh, paint.as_ref(), &env, kind);
            let entry = upload_proto(&self.device, stamp, bk, &mut stats);
            stats.rebuilt += 1;
            match self.proto_index.get(key) {
                Some(&ix) => {
                    new_protos.push((ix, entry));
                    used_protos.push(ix);
                }
                None => {
                    self.protos.push(entry);
                    let ix = self.protos.len() - 1;
                    self.proto_index.insert(key.clone(), ix);
                    used_protos.push(ix);
                }
            }
        }

        for (ix, e) in new_protos {
            self.protos[ix] = e;
        }

        // Textures the protos name.
        let mut wanted: Vec<String> = Vec::new();
        for &ix in &used_protos {
            for p in &self.protos[ix].tex_paths {
                if !wanted.contains(p) {
                    wanted.push(p.clone());
                }
            }
        }
        // ... and the ones the billboards' faces name.
        for b in &self.billboards {
            for &f in &scene.mesh(b.mesh).faces {
                for (k, v) in &scene.mesh(b.mesh).face(f).attrs {
                    if k == "texture" {
                        if let Some(ingetrazo_formats::pyjson::Value::Str(p)) = v.get("path") {
                            if !p.is_empty() && !wanted.contains(p) {
                                wanted.push(p.clone());
                            }
                        }
                    }
                }
            }
        }
        for path in wanted {
            if !self.tex_cache.contains_key(&path) {
                let info = TexInfo::from_image(textures.load(&path));
                self.tex_cache.insert(path.clone(), info);
            }
            self.ensure_texture(&path);
        }
        for &ix in &used_protos {
            let slots: Vec<Option<usize>> =
                self.protos[ix].tex_paths.iter().map(|p| self.tex_index.get(p).copied()).collect();
            self.protos[ix].tex_slot = slots;
        }

        // Instances.
        self.draws.clear();
        self.inst_matrices.clear();
        let mut inst_data: Vec<f32> = Vec::new();
        for key in &order {
            let ix = self.proto_index[key];
            let start = self.inst_matrices.len() as u32;
            let mut centers = Vec::new();
            let pe = &self.protos[ix];
            let c = V3(
                (pe.lo[0] + pe.hi[0]) * 0.5,
                (pe.lo[1] + pe.hi[1]) * 0.5,
                (pe.lo[2] + pe.hi[2]) * 0.5,
            );
            for &pi in &members[key] {
                let p = &self.placements[pi];
                let m = p.matrix.unwrap_or_else(M4::identity);
                inst_data.extend_from_slice(&m.data());
                inst_data.extend_from_slice(&[
                    if p.faded { EDIT_REST_FADE } else { 0.0 },
                    if p.mirrored { 1.0 } else { 0.0 },
                    0.0,
                    0.0,
                ]);
                centers.push(m.map(c));
                self.inst_matrices.push(m);
            }
            let count = self.inst_matrices.len() as u32 - start;
            self.draws.push(ProtoDraw { proto: ix, inst: (start, count), centers });
        }
        self.inst_buf.write(&self.device, &self.queue, wgpu::BufferUsages::VERTEX, bytemuck::cast_slice(&inst_data));
        stats.instances = self.inst_matrices.len();
        stats.prototypes = self.draws.len();
        stats.uploaded_bytes += (inst_data.len() * 4) as u64;
        self.sync_stats = stats;
        stats
    }

    fn ensure_texture(&mut self, path: &str) {
        if self.tex_index.contains_key(path) {
            return;
        }
        let Some(info) = self.tex_cache.get(path) else { return };
        let Some(img) = info.image.as_ref() else { return };
        let chain = mip_chain(img);
        let levels = if info.hard_cutout { 1 } else { chain.len() as u32 };
        let tex = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("face texture"),
            size: wgpu::Extent3d { width: img.width, height: img.height, depth_or_array_layers: 1 },
            mip_level_count: levels,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        for (level, (w, h, data)) in chain.iter().enumerate().take(levels as usize) {
            self.queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &tex,
                    mip_level: level as u32,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                data,
                wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(w * 4), rows_per_image: Some(*h) },
                wgpu::Extent3d { width: *w, height: *h, depth_or_array_layers: 1 },
            );
        }
        let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
        let sampler = if info.hard_cutout { &self.sampler_nomip } else { &self.sampler_mip };
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("face texture"),
            layout: &self.tex_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(sampler) },
            ],
        });
        self.tex_gpu.push(TexGpu { bind_group });
        self.tex_index.insert(path.to_string(), self.tex_gpu.len() - 1);
    }

    // ---- the frame ---------------------------------------------------------------------

    /// Sync the scene and render it into `target` (a view of a texture with
    /// the configured format and `size`), recording on `encoder`.
    ///
    /// The queue writes this makes (buffers, uniforms) must be submitted
    /// before the encoder's command buffer, which is the normal order. The
    /// per-frame buffers are reused: one `render` per submission (a second
    /// viewport takes a second `Renderer`).
    ///
    /// The target's format is the configured one, non-sRGB: the colours are
    /// written as the OpenGL viewport wrote them, with no gamma conversion.
    #[allow(clippy::too_many_arguments)]
    pub fn render(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
        size: (u32, u32),
        scene: &Scene,
        textures: &mut dyn TextureSource,
        camera: &Camera,
        opts: &RenderOptions,
    ) -> FrameStats {
        let mut stats = self.sync(scene, textures, opts);
        let mut cam = *camera;
        cam.set_aspect(size.0, size.1);
        let style = opts
            .style
            .clone()
            .or_else(|| scene.extra.display_style.as_ref().map(Style::from_value))
            .unwrap_or_default();
        let plan = self.plan_frame(scene, &cam, size, opts, &style);
        stats.draw_calls = plan.draw_count();
        stats.triangles = plan.triangle_count();
        self.encode(encoder, target, size, &cam, &plan);
        self.last_plan = plan;
        stats
    }

    /// Like [`Renderer::render`] but returns the plan without encoding it.
    pub fn plan(
        &mut self,
        scene: &Scene,
        textures: &mut dyn TextureSource,
        camera: &Camera,
        size: (u32, u32),
        opts: &RenderOptions,
    ) -> FramePlan {
        self.sync(scene, textures, opts);
        let mut cam = *camera;
        cam.set_aspect(size.0, size.1);
        let style = opts
            .style
            .clone()
            .or_else(|| scene.extra.display_style.as_ref().map(Style::from_value))
            .unwrap_or_default();
        self.plan_frame(scene, &cam, size, opts, &style)
    }

    fn tex_for(&self, pe: &ProtoEntry, run: &Run) -> (Option<usize>, [f32; 3], bool, bool) {
        // (gpu texture, average colour, hard cutout, image present)
        match run.tex {
            None => (None, [0.78; 3], false, true),
            Some(t) => {
                let path = &pe.tex_paths[t as usize];
                let info = self.tex_cache.get(path);
                (
                    pe.tex_slot[t as usize],
                    info.map(|i| i.average).unwrap_or([0.78; 3]),
                    info.is_some_and(|i| i.hard_cutout),
                    info.is_some_and(|i| i.image.is_some()),
                )
            }
        }
    }

    fn world_of(&self, path: &GroupPath) -> (Option<MeshId>, M4, Option<&Placement>) {
        match self.placements.iter().find(|p| &p.path == path) {
            Some(p) => (Some(p.mesh), p.matrix.unwrap_or_else(M4::identity), Some(p)),
            None => (None, M4::identity(), None),
        }
    }

    /// Build the command list of a frame.
    fn plan_frame(&self, scene: &Scene, cam: &Camera, size: (u32, u32), opts: &RenderOptions, style: &Style) -> FramePlan {
        let (w, h) = (size.0.max(1), size.1.max(1));
        let mode = style.face_mode;
        let xray = mode == FaceMode::Xray;
        let flat_mode = matches!(mode, FaceMode::HiddenLine | FaceMode::Monochrome);
        let exporting = opts.export.is_some();
        let scene_back = rgb_of(scene.extra.back_face_color.as_ref());
        let tint = style.effective_back_color(scene_back);

        let mvp_gl = cam.mvp();
        let mvp = gl_to_wgpu(&mvp_gl);
        let fade_rgb = edit_fade_color(style.background);
        let sect = section::active_cut(scene);
        let clip = sect.as_ref().map(|s| s.clip);

        let mut b = Builder::new(
            Ctx { mvp, clip, fade: [fade_rgb[0], fade_rgb[1], fade_rgb[2], 1.0], vp: (w as f32, h as f32) },
            style.background,
        );

        // Sky and ground.
        if opts.backdrop && style.sky {
            let sky = overlay::sky(style, cam, &mvp_gl);
            b.plan.sky_tris = sky.tris;
            b.plan.sky_line = sky.line;
            let key = PipeKey {
                kind: PipeKind::SkyTri,
                blend: false,
                depth_write: false,
                cmp: DepthCmp::Always,
                bias: 0,
                color_write: true,
            };
            let slot = b.slot(P::default());
            if !b.plan.sky_tris.is_empty() {
                let n = b.plan.sky_tris.len() as u32;
                b.cmd(key, slot, None, Geo::SkyTris, (0, n), None);
            }
            if !b.plan.sky_line.is_empty() {
                let key = PipeKey { kind: PipeKind::SkyLine, ..key };
                b.cmd(key, slot, None, Geo::SkyLine, (0, 2), None);
            }
        }

        // ---- faces ---------------------------------------------------------------------
        if mode != FaceMode::Wireframe {
            for d in &self.draws {
                let pe = &self.protos[d.proto];
                for run in pe.runs.iter().filter(|r| r.class == RunClass::Main) {
                    let (tex, avg, hard, present) = self.tex_for(pe, run);
                    let verts = (run.start, run.start + run.count);
                    if flat_mode {
                        let fr = style.front_color;
                        let back = if mode == FaceMode::HiddenLine || run.tex.is_some() { fr } else { tint };
                        let slot = b.slot(P {
                            color: [fr[0], fr[1], fr[2], 1.0],
                            back: [back[0], back[1], back[2], 1.0],
                            mode: ColMode::Flat,
                            clip: true,
                            ..P::default()
                        });
                        b.cmd(PipeKey::face(), slot, None, Geo::Faces(d.proto), verts, Some(d.inst));
                    } else if run.tex.is_none() {
                        let slot = b.slot(P {
                            back: [tint[0], tint[1], tint[2], 1.0],
                            mode: ColMode::Vertex,
                            tint: !xray,
                            opacity: if xray { XRAY_FACE_OPACITY } else { 1.0 },
                            clip: true,
                            ..P::default()
                        });
                        let pipe = PipeKey { blend: xray, depth_write: !xray, ..PipeKey::face() };
                        b.cmd(pipe, slot, None, Geo::Faces(d.proto), verts, Some(d.inst));
                    } else if mode == FaceMode::Shaded {
                        // Textured faces in their average colour, with the
                        // back tint on unpainted backs.
                        let slot = b.slot(P {
                            color: [avg[0], avg[1], avg[2], 1.0],
                            back: [tint[0], tint[1], tint[2], 1.0],
                            mode: ColMode::FlatTint,
                            tint: true,
                            clip: true,
                            ..P::default()
                        });
                        b.cmd(PipeKey::face(), slot, None, Geo::Faces(d.proto), verts, Some(d.inst));
                    } else if present && tex.is_some() {
                        let slot = b.slot(P {
                            back: [tint[0], tint[1], tint[2], 1.0],
                            mode: ColMode::Texture,
                            tint: true,
                            opacity: if xray { XRAY_FACE_OPACITY } else { 1.0 },
                            hard,
                            clip: true,
                            ..P::default()
                        });
                        let pipe = PipeKey { blend: true, depth_write: !xray, ..PipeKey::face() };
                        b.cmd(pipe, slot, tex, Geo::Faces(d.proto), verts, Some(d.inst));
                    }
                }
            }
            // The back-side paint: only seen from behind, no polygon offset.
            if !flat_mode {
                for d in &self.draws {
                    let pe = &self.protos[d.proto];
                    for run in pe.runs.iter().filter(|r| r.class == RunClass::Override) {
                        let (tex, _avg, hard, present) = self.tex_for(pe, run);
                        let verts = (run.start, run.start + run.count);
                        let base = PipeKey { bias: 0, ..PipeKey::face() };
                        if run.tex.is_none() {
                            let slot = b.slot(P { mode: ColMode::Vertex, clip: true, ..P::default() });
                            b.cmd(base, slot, None, Geo::Faces(d.proto), verts, Some(d.inst));
                        } else if present && tex.is_some() {
                            let slot = b.slot(P { mode: ColMode::Texture, hard, clip: true, ..P::default() });
                            b.cmd(PipeKey { blend: true, ..base }, slot, tex, Geo::Faces(d.proto), verts, Some(d.inst));
                        }
                    }
                }
            }
            // Section fill: the cut-through areas of solids.
            if let Some(s) = &sect {
                if !matches!(mode, FaceMode::Xray) {
                    section::add_fill(&mut b, self, scene, s, style, mode);
                }
            }
            // Translucent runs, far to near.
            if !flat_mode {
                let eye = cam.eye();
                let mut order: Vec<(usize, f32)> = self
                    .draws
                    .iter()
                    .enumerate()
                    .map(|(i, d)| {
                        let far = d
                            .centers
                            .iter()
                            .map(|c| {
                                let v = c.sub(eye);
                                v.0 * v.0 + v.1 * v.1 + v.2 * v.2
                            })
                            .fold(0.0f32, f32::max);
                        (i, far)
                    })
                    .collect();
                order.sort_by(|a, c| c.1.total_cmp(&a.1));
                for (i, _) in order {
                    let d = &self.draws[i];
                    let pe = &self.protos[d.proto];
                    for run in pe.runs.iter().filter(|r| r.class == RunClass::Translucent) {
                        let (tex, _avg, hard, present) = self.tex_for(pe, run);
                        let verts = (run.start, run.start + run.count);
                        let pipe = PipeKey { blend: true, depth_write: false, ..PipeKey::face() };
                        if run.tex.is_none() {
                            let slot = b.slot(P { mode: ColMode::Vertex, clip: true, ..P::default() });
                            b.cmd(pipe, slot, None, Geo::Faces(d.proto), verts, Some(d.inst));
                        } else if present && tex.is_some() {
                            let slot = b.slot(P { mode: ColMode::Texture, hard, clip: true, ..P::default() });
                            b.cmd(pipe, slot, tex, Geo::Faces(d.proto), verts, Some(d.inst));
                        }
                    }
                }
            }
            // Face-me billboards (2D people), in the shaded looks only.
            if opts.billboards && !flat_mode {
                self.add_billboards(&mut b, scene, cam);
            }
            // Selection and hover dots on faces (never in exports).
            if !exporting {
                self.add_face_highlights(&mut b, scene, opts, tint);
            }
        }

        // Hidden things as a see-through weave (any style).
        for d in &self.draws {
            let pe = &self.protos[d.proto];
            for run in pe.runs.iter().filter(|r| r.class == RunClass::Ghost) {
                let slot = b.slot(P { mode: ColMode::Vertex, stipple: Stipple::Grid, clip: true, ..P::default() });
                b.cmd(PipeKey::face(), slot, None, Geo::Faces(d.proto), (run.start, run.start + run.count), Some(d.inst));
            }
            if pe.n_ghost > 0 {
                let ec = style.edge_color;
                let slot = b.slot(P {
                    color: [ec[0], ec[1], ec[2], 1.0],
                    stipple: Stipple::Dotted,
                    clip: true,
                    ..P::default()
                });
                b.cmd(PipeKey::line(), slot, None, Geo::GhostEdges(d.proto), (0, pe.n_ghost), Some(d.inst));
            }
        }

        // ---- scaffolding: axes and guides ---------------------------------------------
        if opts.overlays {
            let spacing = (cam.distance as f32 * 0.008).max(1e-4);
            let axes = overlay::axes(spacing, opts.axes_origin);
            let pipe = PipeKey { depth_write: false, ..PipeKey::line() };
            for (k, segs) in axes.iter().enumerate() {
                let c = AXIS_COLORS[k];
                let slot = b.slot(P { color: [c[0], c[1], c[2], 1.0], ..P::default() });
                let r = b.dyn_edges(segs);
                b.cmd(pipe, slot, None, Geo::DynEdges, r, None);
            }
            let guides = section::parse_guides(scene, cam.hypot);
            if !guides.is_empty() {
                let dist = (cam.distance as f32).max(1e-3);
                let runs = overlay::guides(&guides, dist * 0.03, &opts.guides_selected, cam.target, dist * 15.0);
                for (segs, selected) in runs {
                    let c = if selected && !exporting { SELECTION_COLOR } else { [0.27, 0.35, 0.47] };
                    let slot = b.slot(P { color: [c[0], c[1], c[2], 1.0], ..P::default() });
                    let r = b.dyn_edges(&segs);
                    b.cmd(pipe, slot, None, Geo::DynEdges, r, None);
                }
            }
        }

        // ---- edges ------------------------------------------------------------------------
        let show_edges = style.edges || mode == FaceMode::Wireframe;
        let ec = style.edge_color;
        let (jit_e, jit_p) = match opts.export {
            Some(e) => (overlay::line_jitter(e.edge_px, w, h), overlay::line_jitter(e.profile_px, w, h)),
            None => (vec![(0.0, 0.0)], vec![(0.0, 0.0)]),
        };
        let xray_edges = xray && show_edges;
        if xray_edges {
            // The faces' depth, with no colour: hidden edges are told apart.
            for d in &self.draws {
                let pe = &self.protos[d.proto];
                for run in pe.runs.iter().filter(|r| r.class == RunClass::Main) {
                    let slot = b.slot(P { mode: ColMode::Vertex, clip: true, ..P::default() });
                    let pipe = PipeKey { color_write: false, ..PipeKey::face() };
                    b.cmd(pipe, slot, None, Geo::Faces(d.proto), (run.start, run.start + run.count), Some(d.inst));
                }
            }
        }
        let mut edge_passes: Vec<(Rgb, bool)> = Vec::new();
        if xray_edges {
            let bg = style.background;
            let k = XRAY_HIDDEN_EDGE_FADE;
            edge_passes.push(([ec[0] + (bg[0] - ec[0]) * k, ec[1] + (bg[1] - ec[1]) * k, ec[2] + (bg[2] - ec[2]) * k], true));
        }
        edge_passes.push((ec, false));
        if show_edges {
            for (col, hidden) in &edge_passes {
                for &(dx, dy) in &jit_e {
                    let pipe = PipeKey {
                        cmp: if *hidden { DepthCmp::Greater } else { DepthCmp::LessEqual },
                        depth_write: !*hidden,
                        ..PipeKey::line()
                    };
                    let slot = b.slot(P {
                        color: [col[0], col[1], col[2], 1.0],
                        clip: true,
                        shift: (dx, dy),
                        ..P::default()
                    });
                    for d in &self.draws {
                        let pe = &self.protos[d.proto];
                        if pe.n_hard > 0 {
                            b.cmd(pipe, slot, None, Geo::Hard(d.proto), (0, pe.n_hard), Some(d.inst));
                        }
                    }
                }
            }
        }
        if xray_edges {
            // glClear(DEPTH): what follows sees through the faces.
            b.new_pass();
        } else if show_edges && style.back_edges && mode != FaceMode::Wireframe {
            // Back Edges: the edges a face hides, dashed.
            let pipe = PipeKey { cmp: DepthCmp::Greater, depth_write: false, ..PipeKey::line() };
            let slot = b.slot(P {
                color: [ec[0], ec[1], ec[2], 1.0],
                stipple: Stipple::Dashed,
                dash: 4.0,
                clip: true,
                ..P::default()
            });
            for d in &self.draws {
                let pe = &self.protos[d.proto];
                if pe.n_hard > 0 {
                    b.cmd(pipe, slot, None, Geo::Hard(d.proto), (0, pe.n_hard), Some(d.inst));
                }
            }
        }
        // Profile (silhouette) edges.
        if show_edges && style.profiles {
            for &(dx, dy) in &jit_p {
                let slot = b.slot(P { color: [ec[0], ec[1], ec[2], 1.0], clip: true, shift: (dx, dy), ..P::default() });
                let pipe = PipeKey { kind: PipeKind::Soft, ..PipeKey::line() };
                for d in &self.draws {
                    let pe = &self.protos[d.proto];
                    if pe.n_soft > 0 {
                        b.cmd(pipe, slot, None, Geo::Soft(d.proto), (0, pe.n_soft), Some(d.inst));
                    }
                }
            }
        }
        // Selected and hovered edges, on top.
        if !exporting {
            self.add_edge_highlights(&mut b, scene, opts);
        }
        // The cut line of the active section, on the plane.
        if let Some(s) = &sect {
            let logical_h = opts.logical_height.unwrap_or(size.1 as f32);
            section::add_cut_edges(&mut b, self, scene, s, cam, logical_h, style);
        }

        // Tool previews and rubber bands.
        if !opts.overlay_lines.is_empty() {
            for l in &opts.overlay_lines {
                let pipe = PipeKey {
                    cmp: if l.depth_test { DepthCmp::LessEqual } else { DepthCmp::Always },
                    depth_write: false,
                    ..PipeKey::line()
                };
                let slot = b.slot(P { color: [l.color[0], l.color[1], l.color[2], 1.0], ..P::default() });
                let r = b.dyn_edges(&[
                    EdgeVertex { pos: [l.a.0, l.a.1, l.a.2], anchor: [l.b.0, l.b.1, l.b.2] },
                    EdgeVertex { pos: [l.b.0, l.b.1, l.b.2], anchor: [l.b.0, l.b.1, l.b.2] },
                ]);
                b.cmd(pipe, slot, None, Geo::DynEdges, r, None);
            }
        }
        b.finish()
    }

    fn add_billboards(&self, b: &mut Builder, scene: &Scene, cam: &Camera) {
        let eye = cam.eye();
        for bb in &self.billboards {
            let Some(g) = billboard::build(scene, bb, eye, cam.target, cam.perspective) else { continue };
            for (path, verts) in &g.textured {
                let Some(&tex) = self.tex_index.get(path) else { continue };
                let r = b.dyn_faces(verts);
                // A hard alpha cut: the mipmapped edge alpha under the dither
                // would stipple dots around the silhouette.
                let slot = b.slot(P { mode: ColMode::Texture, hard: true, clip: true, ..P::default() });
                let pipe = PipeKey { blend: true, bias: 0, ..PipeKey::face() };
                b.cmd(pipe, slot, Some(tex), Geo::DynFaces, r, None);
            }
            for (col, verts) in &g.colored {
                let r = b.dyn_faces(verts);
                let slot = b.slot(P {
                    color: [col[0], col[1], col[2], 1.0],
                    back: [col[0], col[1], col[2], 1.0],
                    mode: ColMode::Flat,
                    clip: true,
                    ..P::default()
                });
                b.cmd(PipeKey { bias: 0, ..PipeKey::face() }, slot, None, Geo::DynFaces, r, None);
            }
        }
    }

    fn add_face_highlights(&self, b: &mut Builder, scene: &Scene, opts: &RenderOptions, tint: Rgb) {
        let mut runs: Vec<([f32; 3], [f32; 3], Vec<FaceVertex>)> = Vec::new();
        let add = |path: &GroupPath, f: ingetrazo_model::FaceId, runs: &mut Vec<([f32; 3], [f32; 3], Vec<FaceVertex>)>| {
            let (mesh_id, m, placement) = self.world_of(path);
            let Some(mid) = mesh_id else { return };
            let mesh = scene.mesh(mid);
            let paint = placement.and_then(|p| p.paint.as_ref());
            let (front, back) = selection_dot_colors(&mesh.face(f).attrs, paint, tint);
            let mut verts = Vec::new();
            for t in face_tris(mesh, f) {
                for p in t {
                    let q = m.map(p);
                    verts.push(FaceVertex { pos: [q.0, q.1, q.2], uv: [0.0; 2], rgba: [255; 4], misc: 0 });
                }
            }
            if verts.is_empty() {
                return;
            }
            match runs.iter_mut().find(|r| r.0 == front && r.1 == back) {
                Some(r) => r.2.extend(verts),
                None => runs.push((front, back, verts)),
            }
        };
        for (path, f) in &opts.selection.faces {
            add(path, *f, &mut runs);
        }
        if let Some((path, f)) = &opts.hover.face {
            if !opts.selection.faces.iter().any(|(p, g)| p == path && g == f) {
                add(path, *f, &mut runs);
            }
        }
        for (front, back, verts) in runs {
            let r = b.dyn_faces(&verts);
            let slot = b.slot(P {
                color: [front[0], front[1], front[2], 1.0],
                back: [back[0], back[1], back[2], 1.0],
                mode: ColMode::Flat,
                stipple: Stipple::Dots,
                clip: true,
                ..P::default()
            });
            let pipe = PipeKey { depth_write: false, ..PipeKey::face() };
            b.cmd(pipe, slot, None, Geo::DynFaces, r, None);
        }
    }

    fn add_edge_highlights(&self, b: &mut Builder, scene: &Scene, opts: &RenderOptions) {
        let seg = |a: V3, c: V3| {
            [
                EdgeVertex { pos: [a.0, a.1, a.2], anchor: [c.0, c.1, c.2] },
                EdgeVertex { pos: [c.0, c.1, c.2], anchor: [c.0, c.1, c.2] },
            ]
        };
        let mut sel: Vec<EdgeVertex> = Vec::new();
        for (path, e) in &opts.selection.edges {
            let (mesh_id, m, _) = self.world_of(path);
            let Some(mid) = mesh_id else { continue };
            let mesh = scene.mesh(mid);
            let ed = mesh.edge(*e);
            sel.extend(seg(m.map(mesh.position(ed.v0)), m.map(mesh.position(ed.v1))));
        }
        // A selected group reads as the box around it.
        for path in &opts.selection.groups {
            if let Some((lo, hi)) = self.group_world_bounds(path) {
                sel.extend(overlay::box_edges(&overlay::aabb_corners(lo, hi)));
            }
        }
        if !sel.is_empty() {
            let r = b.dyn_edges(&sel);
            let slot = b.slot(P {
                color: [SELECTION_COLOR[0], SELECTION_COLOR[1], SELECTION_COLOR[2], 1.0],
                clip: true,
                ..P::default()
            });
            b.cmd(PipeKey::line(), slot, None, Geo::DynEdges, r, None);
        }
        if let Some((path, e)) = &opts.hover.edge {
            let (mesh_id, m, _) = self.world_of(path);
            if let Some(mid) = mesh_id {
                let mesh = scene.mesh(mid);
                // A curve segment highlights its whole contour.
                let mut ids = mesh.curve_edges(*e);
                if ids.is_empty() {
                    ids.push(*e);
                }
                let mut v = Vec::new();
                for id in ids {
                    let ed = mesh.edge(id);
                    v.extend(seg(m.map(mesh.position(ed.v0)), m.map(mesh.position(ed.v1))));
                }
                let r = b.dyn_edges(&v);
                let slot = b.slot(P {
                    color: [HOVER_COLOR[0], HOVER_COLOR[1], HOVER_COLOR[2], 1.0],
                    clip: true,
                    ..P::default()
                });
                b.cmd(PipeKey::line(), slot, None, Geo::DynEdges, r, None);
            }
        }
    }

    /// World AABB of a group's placements (itself and everything under it).
    fn group_world_bounds(&self, path: &GroupPath) -> Option<([f32; 3], [f32; 3])> {
        let mut lo = [f32::MAX; 3];
        let mut hi = [f32::MIN; 3];
        let mut any = false;
        for p in self.placements.iter().filter(|p| p.path.starts_with(path) && !p.path.is_empty()) {
            let key = ProtoKey {
                mesh: p.mesh,
                paint: p.paint.as_ref().map(|v| format!("{v:?}")).unwrap_or_default(),
                ghost: p.ghost,
                ghost_geometry: false,
            };
            let Some(&ix) = self.proto_index.get(&key) else { continue };
            let pe = &self.protos[ix];
            if pe.tris == 0 && pe.n_hard == 0 {
                continue;
            }
            let m = p.matrix.unwrap_or_else(M4::identity);
            for c in overlay::aabb_corners(pe.lo, pe.hi) {
                let q = m.map(c);
                lo = [lo[0].min(q.0), lo[1].min(q.1), lo[2].min(q.2)];
                hi = [hi[0].max(q.0), hi[1].max(q.1), hi[2].max(q.2)];
                any = true;
            }
        }
        any.then_some((lo, hi))
    }

    /// World bounds of everything drawn (zoom extents).
    pub fn scene_bounds(&self) -> Option<([f32; 3], [f32; 3])> {
        let mut lo = [f32::MAX; 3];
        let mut hi = [f32::MIN; 3];
        let mut any = false;
        for p in &self.placements {
            if p.ghost {
                continue;
            }
            let key = ProtoKey {
                mesh: p.mesh,
                paint: p.paint.as_ref().map(|v| format!("{v:?}")).unwrap_or_default(),
                ghost: false,
                ghost_geometry: false,
            };
            let pe = self.proto_index.get(&key).map(|&i| &self.protos[i]).or_else(|| {
                let key = ProtoKey { ghost_geometry: true, ..key };
                self.proto_index.get(&key).map(|&i| &self.protos[i])
            });
            let Some(pe) = pe else { continue };
            if pe.tris == 0 && pe.n_hard == 0 && pe.n_soft == 0 {
                continue;
            }
            let m = p.matrix.unwrap_or_else(M4::identity);
            for c in overlay::aabb_corners(pe.lo, pe.hi) {
                let q = m.map(c);
                lo = [lo[0].min(q.0), lo[1].min(q.1), lo[2].min(q.2)];
                hi = [hi[0].max(q.0), hi[1].max(q.1), hi[2].max(q.2)];
                any = true;
            }
        }
        any.then_some((lo, hi))
    }

    // ---- encoding ------------------------------------------------------------------------

    fn ensure_pipeline(&mut self, key: PipeKey) {
        if self.pipelines.contains_key(&key) {
            return;
        }
        let p = build_pipeline(&self.device, &self.module, &self.layout, &self.cfg, key);
        self.pipelines.insert(key, p);
    }

    fn ensure_targets(&mut self, size: (u32, u32)) {
        if self.targets.as_ref().is_some_and(|t| t.size == size) {
            return;
        }
        let extent = wgpu::Extent3d { width: size.0, height: size.1, depth_or_array_layers: 1 };
        let samples = self.cfg.samples;
        let depth = self
            .device
            .create_texture(&wgpu::TextureDescriptor {
                label: Some("ingetrazo-render depth"),
                size: extent,
                mip_level_count: 1,
                sample_count: samples,
                dimension: wgpu::TextureDimension::D2,
                format: DEPTH_FORMAT,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            })
            .create_view(&wgpu::TextureViewDescriptor::default());
        let msaa = (samples > 1).then(|| {
            self.device
                .create_texture(&wgpu::TextureDescriptor {
                    label: Some("ingetrazo-render msaa"),
                    size: extent,
                    mip_level_count: 1,
                    sample_count: samples,
                    dimension: wgpu::TextureDimension::D2,
                    format: self.cfg.color_format,
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                    view_formats: &[],
                })
                .create_view(&wgpu::TextureViewDescriptor::default())
        });
        self.targets = Some(Targets { size, msaa, depth });
    }

    fn encode(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
        size: (u32, u32),
        cam: &Camera,
        plan: &FramePlan,
    ) {
        self.ensure_targets(size);
        // Per-frame buffers.
        let stride = self.stride as usize;
        let mut bytes = vec![0u8; plan.params.len().max(1) * stride];
        for (i, p) in plan.params.iter().enumerate() {
            bytes[i * stride..i * stride + std::mem::size_of::<ParamsData>()].copy_from_slice(bytemuck::bytes_of(p));
        }
        let need = bytes.len() as u64;
        let had = self.params_buf.cap;
        self.params_buf.write(&self.device, &self.queue, wgpu::BufferUsages::UNIFORM, &bytes);
        if self.params_bg.is_none() || self.params_buf.cap != had {
            let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("params"),
                layout: &self.params_bgl,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: self.params_buf.buf.as_ref().unwrap(),
                        offset: 0,
                        size: wgpu::BufferSize::new(size_of_params()),
                    }),
                }],
            });
            self.params_bg = Some((bg, need));
        }
        self.dyn_face.write(&self.device, &self.queue, wgpu::BufferUsages::VERTEX, bytemuck::cast_slice(&plan.dyn_faces));
        self.dyn_edge.write(&self.device, &self.queue, wgpu::BufferUsages::VERTEX, bytemuck::cast_slice(&plan.dyn_edges));
        self.sky_tri.write(&self.device, &self.queue, wgpu::BufferUsages::VERTEX, bytemuck::cast_slice(&plan.sky_tris));
        self.sky_line.write(&self.device, &self.queue, wgpu::BufferUsages::VERTEX, bytemuck::cast_slice(&plan.sky_line));
        // The eye in each placement's frame, for the profile test.
        let eye = cam.eye();
        let mut eye_data: Vec<f32> = Vec::with_capacity(self.inst_matrices.len() * 4);
        for m in &self.inst_matrices {
            let e = if m.is_identity() {
                eye
            } else {
                let (inv, ok) = m.inverted();
                if ok {
                    inv.map(eye)
                } else {
                    V3(f32::NAN, f32::NAN, f32::NAN)
                }
            };
            eye_data.extend_from_slice(&[e.0, e.1, e.2, 1.0]);
        }
        self.eye_buf.write(&self.device, &self.queue, wgpu::BufferUsages::VERTEX, bytemuck::cast_slice(&eye_data));

        // Pipelines first (the pass borrows them).
        let keys: Vec<PipeKey> = plan.cmds().map(|c| c.pipe).collect();
        for k in keys {
            self.ensure_pipeline(k);
        }

        let targets = self.targets.as_ref().unwrap();
        let (color_view, resolve) = match &targets.msaa {
            Some(m) => (m, Some(target)),
            None => (target, None),
        };
        let multi = plan.passes.len() > 1;
        let params_bg = &self.params_bg.as_ref().unwrap().0;
        for (pi, pass) in plan.passes.iter().enumerate() {
            let c = plan.clear;
            let color_op = if pi == 0 {
                wgpu::LoadOp::Clear(wgpu::Color { r: c[0] as f64, g: c[1] as f64, b: c[2] as f64, a: 1.0 })
            } else {
                wgpu::LoadOp::Load
            };
            let last = pi + 1 == plan.passes.len();
            let mut rp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("ingetrazo-render"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: color_view,
                    depth_slice: None,
                    resolve_target: resolve,
                    ops: wgpu::Operations {
                        load: color_op,
                        store: if multi && !last { wgpu::StoreOp::Store } else if resolve.is_some() { wgpu::StoreOp::Discard } else { wgpu::StoreOp::Store },
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &targets.depth,
                    depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Discard }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            let mut cur: Option<PipeKey> = None;
            for cmd in &pass.cmds {
                if cur != Some(cmd.pipe) {
                    rp.set_pipeline(&self.pipelines[&cmd.pipe]);
                    cur = Some(cmd.pipe);
                }
                rp.set_bind_group(0, params_bg, &[(cmd.slot as u64 * self.stride) as u32]);
                match cmd.tex {
                    Some(t) => rp.set_bind_group(1, &self.tex_gpu[t].bind_group, &[]),
                    None => rp.set_bind_group(1, &self.dummy_bg, &[]),
                }
                let buf = match cmd.geo {
                    Geo::Faces(p) => self.protos[p].faces.as_ref(),
                    Geo::Hard(p) => self.protos[p].hard.as_ref(),
                    Geo::Soft(p) => self.protos[p].soft.as_ref(),
                    Geo::GhostEdges(p) => self.protos[p].ghost.as_ref(),
                    Geo::DynFaces => self.dyn_face.buf.as_ref(),
                    Geo::DynEdges => self.dyn_edge.buf.as_ref(),
                    Geo::SkyTris => self.sky_tri.buf.as_ref(),
                    Geo::SkyLine => self.sky_line.buf.as_ref(),
                };
                let Some(buf) = buf else { continue };
                rp.set_vertex_buffer(0, buf.slice(..));
                let sky = matches!(cmd.geo, Geo::SkyTris | Geo::SkyLine);
                let insts = match cmd.insts {
                    Some((s, n)) => {
                        let ib = self.inst_buf.buf.as_ref().unwrap();
                        rp.set_vertex_buffer(1, ib.slice(..));
                        if cmd.pipe.kind == PipeKind::Soft {
                            rp.set_vertex_buffer(2, self.eye_buf.buf.as_ref().unwrap().slice(..));
                        }
                        s..s + n
                    }
                    None => {
                        if !sky {
                            rp.set_vertex_buffer(1, self.ident_inst.slice(..));
                            if cmd.pipe.kind == PipeKind::Soft {
                                rp.set_vertex_buffer(2, self.ident_eye.slice(..));
                            }
                        }
                        0..1
                    }
                };
                rp.draw(cmd.verts.0..cmd.verts.1, insts);
            }
        }
    }
}

// ---- frame builder ------------------------------------------------------------------------

/// The constants of a frame.
pub(crate) struct Ctx {
    pub mvp: [f32; 16],
    pub clip: Option<[f32; 4]>,
    pub fade: [f32; 4],
    pub vp: (f32, f32),
}

/// One draw's uniforms, with the defaults of a plain opaque flat line.
pub(crate) struct P {
    pub color: [f32; 4],
    pub back: [f32; 4],
    pub mode: ColMode,
    pub stipple: Stipple,
    pub opacity: f32,
    pub dash: f32,
    pub tint: bool,
    pub hard: bool,
    /// Section clipping applies.
    pub clip: bool,
    /// Clip-space nudge of the mvp (sub-pixel line thickening).
    pub shift: (f32, f32),
}

impl Default for P {
    fn default() -> P {
        P {
            color: [0.0, 0.0, 0.0, 1.0],
            back: [0.0, 0.0, 0.0, 1.0],
            mode: ColMode::Vertex,
            stipple: Stipple::None,
            opacity: 1.0,
            dash: 0.0,
            tint: false,
            hard: false,
            clip: false,
            shift: (0.0, 0.0),
        }
    }
}

pub(crate) struct Builder {
    ctx: Ctx,
    pub plan: FramePlan,
    cur: Vec<DrawCmd>,
}

impl Builder {
    fn new(ctx: Ctx, clear: Rgb) -> Builder {
        Builder { ctx, plan: FramePlan { clear, ..Default::default() }, cur: Vec::new() }
    }

    pub(crate) fn slot(&mut self, p: P) -> u32 {
        let mvp = if p.shift != (0.0, 0.0) { overlay::shifted_mvp(&self.ctx.mvp, p.shift.0, p.shift.1) } else { self.ctx.mvp };
        let (clip, on) = match (p.clip, self.ctx.clip) {
            (true, Some(c)) => (c, 1.0),
            _ => ([0.0; 4], 0.0),
        };
        let data = ParamsData {
            mvp,
            clip,
            color: p.color,
            back_color: p.back,
            fade_color: self.ctx.fade,
            v0: [on, p.mode as u32 as f32, p.stipple as u32 as f32, p.opacity],
            v1: [self.ctx.vp.0, self.ctx.vp.1, p.dash, if p.tint { 1.0 } else { 0.0 }],
            v2: [if p.hard { 1.0 } else { 0.0 }, 0.0, 0.0, 0.0],
        };
        self.plan.params.push(data);
        self.plan.params.len() as u32 - 1
    }

    pub(crate) fn cmd(
        &mut self,
        pipe: PipeKey,
        slot: u32,
        tex: Option<usize>,
        geo: Geo,
        verts: (u32, u32),
        insts: Option<(u32, u32)>,
    ) {
        if verts.1 <= verts.0 || insts.is_some_and(|i| i.1 == 0) {
            return;
        }
        self.cur.push(DrawCmd { pipe, slot, tex, geo, verts, insts });
    }

    pub(crate) fn dyn_edges(&mut self, v: &[EdgeVertex]) -> (u32, u32) {
        let s = self.plan.dyn_edges.len() as u32;
        self.plan.dyn_edges.extend_from_slice(v);
        (s, self.plan.dyn_edges.len() as u32)
    }

    pub(crate) fn dyn_faces(&mut self, v: &[FaceVertex]) -> (u32, u32) {
        let s = self.plan.dyn_faces.len() as u32;
        self.plan.dyn_faces.extend_from_slice(v);
        (s, self.plan.dyn_faces.len() as u32)
    }

    /// Start a new render pass: colour kept, depth cleared.
    pub(crate) fn new_pass(&mut self) {
        let cmds = std::mem::take(&mut self.cur);
        self.plan.passes.push(PassPlan { cmds });
    }

    fn finish(mut self) -> FramePlan {
        let cmds = std::mem::take(&mut self.cur);
        self.plan.passes.push(PassPlan { cmds });
        self.plan
    }
}

// ---- GPU helpers -----------------------------------------------------------------------------

struct TexLoader<'a> {
    cache: &'a mut HashMap<String, TexInfo>,
    src: &'a mut dyn TextureSource,
}

impl TexLoader<'_> {
    fn info(&mut self, path: &str) -> &TexInfo {
        if !self.cache.contains_key(path) {
            let info = TexInfo::from_image(self.src.load(path));
            self.cache.insert(path.to_string(), info);
        }
        &self.cache[path]
    }
}

pub(crate) fn texture_bgl(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("texture"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
        ],
    })
}

fn upload_proto(device: &wgpu::Device, stamp: Stamp, bk: MeshBake, stats: &mut FrameStats) -> ProtoEntry {
    let mk = |data: &[u8], label: &str, stats: &mut FrameStats| -> Option<wgpu::Buffer> {
        (!data.is_empty()).then(|| {
            stats.uploaded_bytes += data.len() as u64;
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: data,
                usage: wgpu::BufferUsages::VERTEX,
            })
        })
    };
    let faces = mk(bytemuck::cast_slice::<FaceVertex, u8>(&bk.verts), "faces", stats);
    let hard = mk(bytemuck::cast_slice::<EdgeVertex, u8>(&bk.hard), "edges", stats);
    let soft = mk(bytemuck::cast_slice::<SoftVertex, u8>(&bk.soft), "soft edges", stats);
    let ghost = mk(bytemuck::cast_slice::<EdgeVertex, u8>(&bk.ghost_edges), "ghost edges", stats);
    ProtoEntry {
        stamp,
        tris: bk.tri_count(),
        n_hard: bk.hard.len() as u32,
        n_soft: bk.soft.len() as u32,
        n_ghost: bk.ghost_edges.len() as u32,
        tex_slot: vec![None; bk.tex_paths.len()],
        runs: bk.runs,
        tex_paths: bk.tex_paths,
        faces,
        hard,
        soft,
        ghost,
        lo: bk.lo,
        hi: bk.hi,
    }
}

const FACE_ATTRS: [wgpu::VertexAttribute; 4] = wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x2, 2 => Unorm8x4, 3 => Uint32];
const EDGE_ATTRS: [wgpu::VertexAttribute; 2] = wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3];
const SOFT_ATTRS: [wgpu::VertexAttribute; 6] = wgpu::vertex_attr_array![
    0 => Float32x3, 1 => Float32x3, 2 => Float32, 3 => Float32x3, 4 => Float32, 5 => Float32
];
const SKY_ATTRS: [wgpu::VertexAttribute; 2] = wgpu::vertex_attr_array![0 => Float32x2, 1 => Float32x3];
const INST_ATTRS: [wgpu::VertexAttribute; 5] = wgpu::vertex_attr_array![
    10 => Float32x4, 11 => Float32x4, 12 => Float32x4, 13 => Float32x4, 14 => Float32x4
];
const EYE_ATTRS: [wgpu::VertexAttribute; 1] = wgpu::vertex_attr_array![15 => Float32x4];

fn build_pipeline(
    device: &wgpu::Device,
    module: &wgpu::ShaderModule,
    layout: &wgpu::PipelineLayout,
    cfg: &RendererConfig,
    key: PipeKey,
) -> wgpu::RenderPipeline {
    let (vs, fs, stride, attrs, topology): (&str, &str, u64, &[wgpu::VertexAttribute], wgpu::PrimitiveTopology) =
        match key.kind {
            PipeKind::Face => ("vs_face", "fs_face", 28, &FACE_ATTRS, wgpu::PrimitiveTopology::TriangleList),
            PipeKind::Line => ("vs_line", "fs_line", 24, &EDGE_ATTRS, wgpu::PrimitiveTopology::LineList),
            PipeKind::Soft => ("vs_soft", "fs_line", 48, &SOFT_ATTRS, wgpu::PrimitiveTopology::LineList),
            PipeKind::SkyTri => ("vs_sky", "fs_sky", 20, &SKY_ATTRS, wgpu::PrimitiveTopology::TriangleList),
            PipeKind::SkyLine => ("vs_sky", "fs_sky", 20, &SKY_ATTRS, wgpu::PrimitiveTopology::LineList),
        };
    let sky = matches!(key.kind, PipeKind::SkyTri | PipeKind::SkyLine);
    let inst_layout = wgpu::VertexBufferLayout {
        array_stride: 80,
        step_mode: wgpu::VertexStepMode::Instance,
        attributes: &INST_ATTRS,
    };
    let eye_layout = wgpu::VertexBufferLayout {
        array_stride: 16,
        step_mode: wgpu::VertexStepMode::Instance,
        attributes: &EYE_ATTRS,
    };
    let vtx = wgpu::VertexBufferLayout { array_stride: stride, step_mode: wgpu::VertexStepMode::Vertex, attributes: attrs };
    let mut buffers: Vec<Option<wgpu::VertexBufferLayout>> = vec![Some(vtx)];
    if !sky {
        buffers.push(Some(inst_layout));
    }
    if key.kind == PipeKind::Soft {
        buffers.push(Some(eye_layout));
    }
    let bias = match key.bias {
        0 => wgpu::DepthBiasState::default(),
        k => wgpu::DepthBiasState {
            constant: FACE_BIAS_CONSTANT * k as i32,
            slope_scale: FACE_BIAS_SLOPE * k as f32,
            clamp: 0.0,
        },
    };
    let compare = match key.cmp {
        DepthCmp::LessEqual => wgpu::CompareFunction::LessEqual,
        DepthCmp::Greater => wgpu::CompareFunction::Greater,
        DepthCmp::Always => wgpu::CompareFunction::Always,
    };
    let blend = key.blend.then_some(wgpu::BlendState {
        color: wgpu::BlendComponent {
            src_factor: wgpu::BlendFactor::SrcAlpha,
            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
            operation: wgpu::BlendOperation::Add,
        },
        alpha: wgpu::BlendComponent {
            src_factor: wgpu::BlendFactor::One,
            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
            operation: wgpu::BlendOperation::Add,
        },
    });
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("ingetrazo-render"),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module,
            entry_point: Some(vs),
            compilation_options: Default::default(),
            buffers: &buffers,
        },
        primitive: wgpu::PrimitiveState { topology, ..Default::default() },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: DEPTH_FORMAT,
            depth_write_enabled: Some(key.depth_write),
            depth_compare: Some(compare),
            stencil: Default::default(),
            bias,
        }),
        multisample: wgpu::MultisampleState { count: cfg.samples, mask: !0, alpha_to_coverage_enabled: false },
        fragment: Some(wgpu::FragmentState {
            module,
            entry_point: Some(fs),
            compilation_options: Default::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format: cfg.color_format,
                blend,
                write_mask: if key.color_write { wgpu::ColorWrites::ALL } else { wgpu::ColorWrites::empty() },
            })],
        }),
        multiview_mask: None,
        cache: None,
    })
}

/// Polygon offset of the face passes (GL: factor 1, units 1).
pub const FACE_BIAS_CONSTANT: i32 = 1;
pub const FACE_BIAS_SLOPE: f32 = 1.0;
