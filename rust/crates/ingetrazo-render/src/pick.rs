// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! GPU picking (docs/rust-rewrite-plan.md, M4.2): the ID buffer.
//!
//! [`IdPicker`] renders the scene a second time, not to colours but to
//! identities: three integer layers (`Rg32Uint`, placement + entity id per
//! pixel) that say which face, edge and vertex covers each pixel. A pick is
//! then a 1-pixel read-back and a rubber band is a read-back of its
//! rectangle: no ray is cast against the model, so the cost does not grow
//! with the triangle count of the scene.
//!
//! # How the app is meant to combine it with the CPU picker
//!
//! The ID buffer is a fast *candidate source*, not the arbiter. The Select
//! tool's rules live in `ingetrazo_snap::picking` (the M3.12 port of
//! `Viewport.pick_*` and `tools/select.py`, bit-exact to Python): the
//! priority between the pick kinds (label, group, edge, dimension, guide,
//! face, image ...), the thresholds in pixels (8 px for edges and vertices,
//! the nearest edge winning), the tie-breaks and the annotations (guides,
//! dimensions, text, section planes) that are not geometry at all. The
//! intended use is:
//!
//! 1. every frame whose camera or scene moved, call [`IdPicker::prepare`]
//!    (cheap when nothing changed: it returns `false` without touching the
//!    GPU);
//! 2. on mouse move, ask [`IdPicker::pick_layers`] for the face, edge and
//!    vertex under the cursor (one read-back of a few bytes). That is
//!    enough to drive hover highlights at any scene size;
//! 3. on click, run the CPU `Picker` for the final answer when the scene
//!    is small, or when the ID layers say there is something to refine (an
//!    edge or vertex candidate: the CPU applies the exact distance
//!    thresholds and the priority), and take the ID face directly when it
//!    is the only candidate on a heavy model, where the CPU ray cast is the
//!    slow part;
//! 4. for the rubber band, [`IdPicker::pick_rect`] gives what is *visible*
//!    in the rectangle (SketchUp's behaviour); the CPU `box_select` keeps
//!    the Python semantics (everything whose projection falls in the band,
//!    occluded or not).
//!
//! [`IdPicker::pick_pixel`] returns the topmost layer (vertex, then edge,
//! then face) as a convenience; it is not the Select tool's priority.
//!
//! # What the ID pass respects
//!
//! It mirrors the CPU index, not the colour frame: hidden faces and groups
//! (unless View > Hidden Objects/Geometry), hidden and locked layers, locked
//! groups, the open group (outside it nothing can be picked and the loose
//! mesh is left out, as in the CPU index), the active section cut (clipped
//! per fragment) and x-ray (faces are pickable through each other either
//! way; x-ray and wireframe styles also stop faces occluding edges). The
//! face layer ignores the display style entirely.
//!
//! # The layers
//!
//! | layer | holds | CPU counterpart |
//! |---|---|---|
//! | faces | the front-most selectable face, smaller wins ties | `pick_face_any` |
//! | edges | the nearest edge of the loose part within the band, not hidden | `pick_visible_edge` |
//! | vertices | the nearest edge end within the disc, not hidden | `pick_vertex` |
//! | groups | the nearest face of a placement, unless the loose part is in front by the bias | `pick_group` |
//! | group edges | the nearest hard edge of a placement (the fallback for groups of lines) | `pick_group`'s edge fallback |
//!
//! Faces are always built; the others can be switched off through
//! [`PickOptions::layers`] (the edge and vertex layers cost fragment work
//! proportional to the edges on screen).
//!
//! # Differences from the CPU picker (measured by `tests/id_pick.rs`)
//!
//! Over the 14 fixture scenes and 94 views of `pick_parity.rs` (24,490
//! recorded pixels, the CPU asked at each pixel's centre): faces agree on
//! 100 % of the pixels away from an ID border, groups on 99.5 %, edges on
//! 92 %. What differs:
//!
//! * Rasterisation: a pixel belongs to a triangle when its centre does, so
//!   answers at triangle borders can differ from a ray through the exact
//!   cursor position (about 14 % of the sampled pixels sit within 1 px of a
//!   border and are skipped), and a face smaller than a pixel may not be
//!   there at all.
//! * Coplanar faces: the CPU takes the smallest face within a relative
//!   `1e-4` of the nearest hit. The GPU nudges each face toward the eye by
//!   a fraction (growing as the face gets smaller) of a relative
//!   [`PickOptions::coplanar_bias`], so the smaller of coplanar faces wins.
//!   Faces separated by less than that bias, in depth, are ordered by size,
//!   not by depth; exactly coincident faces of the same size are a tie the
//!   CPU settles by index and the GPU by draw order.
//! * Edges are capsules of [`PickOptions::edge_tolerance_px`] pixels measured
//!   from the pixel's centre, the nearest wins. Near-equal distances (two
//!   edges meeting at a vertex, an edge and its collinear neighbour) can go
//!   either way, and in an axis-aligned parallel view the foot of a vertical
//!   wall projects onto its top edge, so which of the two is hidden is a
//!   coin toss in both pickers (the largest class of misses).
//! * Edges with an end behind the camera are clipped to their visible part
//!   here and skipped there; edges cut by the section plane are picked only
//!   where they are kept here, and wherever one end is kept there.
//! * The groups differ from the CPU in the cut scenes only, in a handful of
//!   pixels where a face on the cut is both taken out and not.
//! * Occlusion of edges and vertices uses an unbiased depth of the visible
//!   faces, a tolerance of 1 mm or a relative `3e-4`, and the slope of the
//!   face under the pixel (a grazing face moves a lot across half a pixel).

use std::collections::HashMap;
use std::hash::{Hash, Hasher};

use bytemuck::{Pod, Zeroable};
use ingetrazo_geom::meshface;
use ingetrazo_geom::triangulate::triangulate;
use ingetrazo_math::{M4, V3};
use ingetrazo_model::mesh::{EdgeId, FaceId, Mesh, VertexId};
use ingetrazo_model::pyval::truthy;
use ingetrazo_model::scene::{Group, MeshId, Scene};
use ingetrazo_snap::picking::index::{group_selectable, placements};
use ingetrazo_snap::picking::{FaceMode as SnapFaceMode, State};
use wgpu::util::DeviceExt;

use crate::camera::{gl_to_wgpu, Camera};
use crate::options::RenderOptions;
use crate::plan::{EditRest, GroupPath};
use crate::section;
use crate::style::{FaceMode, Style};

const ID_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rg32Uint;
const DEPTH: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

// Instance flags (see pick.wgsl).
const IF_LOOSE: u32 = 1;
const IF_SEL: u32 = 2;
const IF_VIS: u32 = 4;
// Face flags.
const FF_LOOSE_SEL: u32 = 1;
const FF_LOOSE_VIS: u32 = 2;
const FF_PLACED: u32 = 4;
// Edge flags.
const EF_LOOSE_SEL: u32 = 1;
const EF_SOFT: u32 = 2;
const EF_HARD: u32 = 4;

// ---- what a pick answers ------------------------------------------------------------

/// A face under the cursor.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FaceHit {
    /// The placement the face was hit through (empty = the loose mesh; while
    /// a group is open, the open group's own path: its mesh is the loose
    /// part of the CPU picker).
    pub placement: GroupPath,
    /// The group a click on it selects; `None` for loose faces.
    pub owner: Option<GroupPath>,
    pub mesh: MeshId,
    pub face: FaceId,
}

/// An edge within the pick tolerance of the cursor.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EdgeHit {
    pub placement: GroupPath,
    pub owner: Option<GroupPath>,
    pub mesh: MeshId,
    pub edge: EdgeId,
}

/// A vertex within the pick tolerance of the cursor.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VertexHit {
    pub placement: GroupPath,
    pub owner: Option<GroupPath>,
    pub mesh: MeshId,
    pub vertex: VertexId,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Hit {
    Face(FaceHit),
    Edge(EdgeHit),
    Vertex(VertexHit),
}

/// What each layer holds at one pixel.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PixelHits {
    pub face: Option<FaceHit>,
    pub edge: Option<EdgeHit>,
    pub vertex: Option<VertexHit>,
    /// The group a click on the group layer selects (see [`IdPicker::pick_group`]).
    pub group: Option<GroupPath>,
}

/// Which layers a query reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Layers {
    pub faces: bool,
    pub edges: bool,
    pub vertices: bool,
    /// The group layer: what `pick_group` reads.
    pub groups: bool,
}

impl Layers {
    pub const ALL: Layers = Layers { faces: true, edges: true, vertices: true, groups: true };
    pub const FACES: Layers = Layers { faces: true, edges: false, vertices: false, groups: false };
}

/// How a rubber band takes things.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RectMode {
    /// Only what lies entirely inside the rectangle.
    Window,
    /// Anything visible inside it.
    Crossing,
}

/// A raw layer read-back: `(placement + 1, entity id + 1)` per pixel, rows
/// top to bottom; zeros are empty.
#[derive(Clone, Debug)]
pub struct IdImage {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
    pub ids: Vec<[u32; 2]>,
}

impl IdImage {
    pub fn at(&self, x: u32, y: u32) -> [u32; 2] {
        self.ids[((y - self.y) * self.width + (x - self.x)) as usize]
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Layer {
    Faces,
    Edges,
    Vertices,
    Groups,
    /// The hard edges of the placements the cursor can touch.
    GroupEdges,
    /// Where the loose part hides the placements in the group layer.
    GroupBlockers,
}

#[derive(Clone, Copy, Debug)]
pub struct PickOptions {
    /// Half-width of the edge band in pixels (`Viewport.pick_threshold_px`).
    pub edge_tolerance_px: f32,
    /// Radius of the vertex disc in pixels.
    pub vertex_tolerance_px: f32,
    /// `pick_visible_edge`: soft edges are not pickable unless View >
    /// Hidden Geometry is on, and faces occlude edges (not in x-ray or
    /// wireframe). Off = `pick_edge`: every edge counts.
    pub visible_edges_only: bool,
    /// Relative depth nudge that tells coplanar faces apart (see the module
    /// documentation).
    pub coplanar_bias: f32,
    /// Which layers the pass builds (faces are always built). The edge and
    /// vertex layers cost fragment work proportional to the number of edges
    /// on screen: an app that picks vertices rarely can turn them off on
    /// heavy scenes.
    pub layers: Layers,
}

impl Default for PickOptions {
    fn default() -> PickOptions {
        PickOptions { edge_tolerance_px: 8.0, vertex_tolerance_px: 8.0, visible_edges_only: true, coplanar_bias: 1e-3, layers: Layers::ALL }
    }
}

/// Counters for diagnostics and benches.
#[derive(Clone, Copy, Debug, Default)]
pub struct PickStats {
    /// ID passes rendered.
    pub renders: u64,
    /// `prepare` calls answered from the last render.
    pub reused: u64,
    /// Meshes (re)baked for the pass.
    pub baked_meshes: u64,
    pub placements: usize,
    pub instances: usize,
}

// ---- GPU data ---------------------------------------------------------------------------

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
struct PickVert {
    pos: [f32; 3],
    face: u32,
    flags: u32,
    area: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
struct EdgeRec {
    a: [f32; 3],
    v0: u32,
    b: [f32; 3],
    v1: u32,
    id: u32,
    flags: u32,
    pad: [u32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
struct InstRec {
    m: [f32; 16],
    info: [u32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
struct ParamsData {
    mvp: [f32; 16],
    clip: [f32; 4],
    eye: [f32; 4],
    fwd: [f32; 4],
    zrow: [f32; 4],
    a: [f32; 4],
    b: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
struct DrawInfo {
    n_edges: u32,
    edge_base: u32,
    inst_first: u32,
    pad: u32,
}

const FACE_ATTRS: [wgpu::VertexAttribute; 4] =
    wgpu::vertex_attr_array![0 => Float32x3, 1 => Uint32, 2 => Uint32, 3 => Float32];
const INST_ATTRS: [wgpu::VertexAttribute; 5] =
    wgpu::vertex_attr_array![4 => Float32x4, 5 => Float32x4, 6 => Float32x4, 7 => Float32x4, 8 => Uint32x4];

/// One mesh baked for the pass.
struct PickMesh {
    stamp: (u64, (usize, usize, usize), u64),
    tris: Option<wgpu::Buffer>,
    n_verts: u32,
    edges: Option<wgpu::Buffer>,
    n_edges: u32,
    /// Face id to its outer ring in `ring_pos`, for window selection.
    face_rings: HashMap<u32, (u32, u32)>,
    ring_pos: Vec<[f32; 3]>,
    /// Edge id to its ends.
    edge_ends: HashMap<u32, ([f32; 3], [f32; 3])>,
}

/// A placement the pass draws.
#[derive(Clone, Debug)]
struct PlaceInfo {
    path: GroupPath,
    owner: Option<GroupPath>,
    mesh: MeshId,
    xform: Option<M4>,
    loose: bool,
    sel: bool,
    vis: bool,
}

struct MeshDraw {
    mesh: MeshId,
    first: u32,
    count: u32,
    /// Some instance is the loose part / a placement the cursor can touch.
    loose: bool,
    placed: bool,
    bind_group: Option<wgpu::BindGroup>,
}

struct Targets {
    size: (u32, u32),
    faces: wgpu::Texture,
    edges: wgpu::Texture,
    verts: wgpu::Texture,
    groups: wgpu::Texture,
    gedges: wgpu::Texture,
    gloose: wgpu::Texture,
    depth_pick: wgpu::TextureView,
    depth_occ: wgpu::TextureView,
    depth_line: wgpu::TextureView,
    group0_plain: wgpu::BindGroup,
    group0_depth: wgpu::BindGroup,
}

/// The ID-buffer picker. One per viewport.
pub struct IdPicker {
    device: wgpu::Device,
    queue: wgpu::Queue,
    /// Tunables; changing them re-renders on the next `prepare`.
    pub options: PickOptions,
    bgl_plain: wgpu::BindGroupLayout,
    bgl_depth: wgpu::BindGroupLayout,
    bgl_mesh: wgpu::BindGroupLayout,
    pipe_face: wgpu::RenderPipeline,
    pipe_occ: wgpu::RenderPipeline,
    pipe_group: wgpu::RenderPipeline,
    pipe_gloose: wgpu::RenderPipeline,
    pipe_edge: wgpu::RenderPipeline,
    pipe_gedge: wgpu::RenderPipeline,
    pipe_vertex: wgpu::RenderPipeline,
    params_buf: wgpu::Buffer,
    meshes: HashMap<MeshId, PickMesh>,
    targets: Option<Targets>,
    inst_buf: Option<wgpu::Buffer>,
    draws: Vec<MeshDraw>,
    places: Vec<PlaceInfo>,
    key: Option<u64>,
    view: Option<ViewState>,
    staging: Option<(wgpu::Buffer, u64)>,
    stats: PickStats,
}

/// What the last render saw, for decoding and window tests.
#[derive(Clone)]
struct ViewState {
    size: (u32, u32),
    mvp: M4,
}

impl IdPicker {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> IdPicker {
        let device = device.clone();
        let queue = queue.clone();
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ingetrazo-render pick"),
            source: wgpu::ShaderSource::Wgsl(include_str!("pick.wgsl").into()),
        });
        let uniform = |binding: u32, vis: wgpu::ShaderStages| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: vis,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let storage = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::VERTEX,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let both = wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT;
        let bgl_plain = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("pick params"),
            entries: &[uniform(0, both)],
        });
        let bgl_depth = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("pick params + occluder"),
            entries: &[
                uniform(0, both),
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        });
        let bgl_mesh = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("pick mesh"),
            entries: &[storage(0), storage(1), uniform(2, wgpu::ShaderStages::VERTEX)],
        });
        let layout_face = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("pick faces"),
            bind_group_layouts: &[Some(&bgl_plain)],
            immediate_size: 0,
        });
        let layout_line = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("pick lines"),
            bind_group_layouts: &[Some(&bgl_depth), Some(&bgl_mesh)],
            immediate_size: 0,
        });
        let face_buffers = [
            Some(wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<PickVert>() as u64,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &FACE_ATTRS,
            }),
            Some(wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<InstRec>() as u64,
                step_mode: wgpu::VertexStepMode::Instance,
                attributes: &INST_ATTRS,
            }),
        ];
        let color_target =
            [Some(wgpu::ColorTargetState { format: ID_FORMAT, blend: None, write_mask: wgpu::ColorWrites::ALL })];
        let pipe = |label: &str,
                    layout: &wgpu::PipelineLayout,
                    vs: &str,
                    fs: &str,
                    buffers: &[Option<wgpu::VertexBufferLayout>],
                    color: bool,
                    consts: &[(&str, f64)]| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(layout),
                vertex: wgpu::VertexState {
                    module: &module,
                    entry_point: Some(vs),
                    compilation_options: wgpu::PipelineCompilationOptions { constants: consts, ..Default::default() },
                    buffers,
                },
                primitive: wgpu::PrimitiveState { topology: wgpu::PrimitiveTopology::TriangleList, ..Default::default() },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: DEPTH,
                    depth_write_enabled: Some(true),
                    depth_compare: Some(wgpu::CompareFunction::Less),
                    stencil: Default::default(),
                    bias: Default::default(),
                }),
                multisample: wgpu::MultisampleState::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &module,
                    entry_point: Some(fs),
                    compilation_options: Default::default(),
                    targets: if color { &color_target } else { &[] },
                }),
                multiview_mask: None,
                cache: None,
            })
        };
        let pipe_face = pipe("pick faces", &layout_face, "vs_face", "fs_face", &face_buffers, true, &[]);
        let pipe_occ = pipe("pick occluders", &layout_face, "vs_occluder", "fs_occluder", &face_buffers, false, &[]);
        let pipe_group = pipe("pick groups", &layout_face, "vs_group_placed", "fs_face", &face_buffers, true, &[]);
        let pipe_gloose = pipe("pick group blockers", &layout_face, "vs_group_loose", "fs_face", &face_buffers, true, &[]);
        let pipe_edge = pipe("pick edges", &layout_line, "vs_edge", "fs_edge", &[], true, &[("EDGE_MODE", 0.0)]);
        let pipe_gedge = pipe("pick group edges", &layout_line, "vs_edge", "fs_edge", &[], true, &[("EDGE_MODE", 1.0)]);
        let pipe_vertex = pipe("pick vertices", &layout_line, "vs_vertex", "fs_vertex", &[], true, &[]);
        let params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pick params"),
            size: std::mem::size_of::<ParamsData>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        IdPicker {
            device,
            queue,
            options: PickOptions::default(),

            bgl_plain,
            bgl_depth,
            bgl_mesh,
            pipe_face,
            pipe_occ,
            pipe_group,
            pipe_gloose,
            pipe_edge,
            pipe_gedge,
            pipe_vertex,
            params_buf,
            meshes: HashMap::new(),
            targets: None,
            inst_buf: None,
            draws: Vec::new(),
            places: Vec::new(),
            key: None,
            view: None,
            staging: None,
            stats: PickStats::default(),
        }
    }

    pub fn stats(&self) -> PickStats {
        self.stats
    }

    /// Forget every baked mesh and the last render (call when the scene
    /// object is replaced: mesh serials restart from zero).
    pub fn invalidate(&mut self) {
        self.meshes.clear();
        self.key = None;
        self.view = None;
    }

    /// Render the ID layers for this scene, camera and viewport size unless
    /// the last render already answers it. Returns whether the pass ran.
    ///
    /// `opts` carries what the colour frame was given: the open group
    /// (`edit_path`, `edit_rest`) and the style (x-ray and wireframe stop
    /// faces occluding edges).
    pub fn prepare(&mut self, scene: &Scene, camera: &Camera, size: (u32, u32), opts: &RenderOptions) -> bool {
        let size = (size.0.max(1), size.1.max(1));
        let mut cam = *camera;
        cam.set_aspect(size.0, size.1);
        let style = opts
            .style
            .clone()
            .or_else(|| scene.extra.display_style.as_ref().map(Style::from_value))
            .unwrap_or_default();
        let see_through = matches!(style.face_mode, FaceMode::Xray | FaceMode::Wireframe);
        let key = {
            let mut h = std::collections::hash_map::DefaultHasher::new();
            scene_key(scene, &mut h);
            camera_key(&cam, &mut h);
            size.hash(&mut h);
            opts.edit_path.hash(&mut h);
            (opts.edit_rest == EditRest::Hide).hash(&mut h);
            see_through.hash(&mut h);
            let o = self.options;
            [o.edge_tolerance_px.to_bits(), o.vertex_tolerance_px.to_bits(), o.coplanar_bias.to_bits(), o.visible_edges_only as u32, o.layers.edges as u32 + 2 * o.layers.vertices as u32 + 4 * o.layers.groups as u32]
                .hash(&mut h);
            h.finish()
        };
        if self.key == Some(key) && self.targets.is_some() {
            self.stats.reused += 1;
            return false;
        }
        self.render(scene, &cam, size, opts, see_through);
        self.key = Some(key);
        self.stats.renders += 1;
        true
    }

    fn render(&mut self, scene: &Scene, cam: &Camera, size: (u32, u32), opts: &RenderOptions, see_through: bool) {
        // The scene as the CPU picks see it.
        let ctx = opts.edit_path.clone();
        let rest_hidden = ctx.is_some() && opts.edit_rest == EditRest::Hide;
        let active = match &ctx {
            None => MeshId(0),
            Some(c) => ingetrazo_ops::group::group_at(&scene.groups, c).map_or(MeshId(0), |g| g.mesh),
        };
        let st = State::new(scene, active, MeshId(0), ctx, rest_hidden, SnapFaceMode::Solid, scene.mesh(MeshId(0)).hypot);
        let places = build_places(&st);

        // Bake what changed.
        let env = env_key(scene);
        let mut used: Vec<MeshId> = Vec::new();
        let mut by_mesh: HashMap<MeshId, Vec<usize>> = HashMap::new();
        for (pi, p) in places.iter().enumerate() {
            by_mesh
                .entry(p.mesh)
                .or_insert_with(|| {
                    used.push(p.mesh);
                    Vec::new()
                })
                .push(pi);
        }
        for &id in &used {
            let mesh = scene.mesh(id);
            let stamp = (mesh.serial, mesh.arena_sizes(), env);
            if self.meshes.get(&id).is_some_and(|m| m.stamp == stamp) {
                continue;
            }
            let baked = bake_pick_mesh(&self.device, mesh, &st, stamp);
            self.meshes.insert(id, baked);
            self.stats.baked_meshes += 1;
        }

        // Instances, grouped by mesh.
        let mut insts: Vec<InstRec> = Vec::new();
        let mut draws: Vec<MeshDraw> = Vec::new();
        for &id in &used {
            let first = insts.len() as u32;
            let (mut loose, mut placed) = (false, false);
            for &pi in &by_mesh[&id] {
                let p = &places[pi];
                loose |= p.loose;
                placed |= !p.loose && p.sel;
                let m = p.xform.unwrap_or_else(M4::identity);
                let mut flags = 0;
                if p.loose {
                    flags |= IF_LOOSE;
                }
                if p.sel {
                    flags |= IF_SEL;
                }
                if p.vis {
                    flags |= IF_VIS;
                }
                insts.push(InstRec { m: m.data(), info: [pi as u32 + 1, flags, 0, 0] });
            }
            draws.push(MeshDraw { mesh: id, first, count: insts.len() as u32 - first, loose, placed, bind_group: None });
        }
        self.stats.placements = places.len();
        self.stats.instances = insts.len();
        let empty = [InstRec::default()];
        let inst_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("pick instances"),
            contents: bytemuck::cast_slice(if insts.is_empty() { &empty[..] } else { &insts[..] }),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::STORAGE,
        });
        for d in &mut draws {
            let pm = &self.meshes[&d.mesh];
            let Some(edges) = pm.edges.as_ref() else { continue };
            let info = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("pick draw"),
                contents: bytemuck::bytes_of(&DrawInfo { n_edges: pm.n_edges, edge_base: 0, inst_first: d.first, pad: 0 }),
                usage: wgpu::BufferUsages::UNIFORM,
            });
            d.bind_group = Some(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("pick mesh"),
                layout: &self.bgl_mesh,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: edges.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: inst_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: info.as_entire_binding() },
                ],
            }));
        }

        // Uniforms.
        let mvp_gl = cam.mvp();
        let proj = gl_to_wgpu(&cam.projection_matrix());
        let eye = cam.eye();
        let fwd = cam.forward();
        let clip = section::active_cut(scene).map_or([0.0; 4], |c| c.clip);
        let o = self.options;
        let occlude = o.visible_edges_only && !see_through;
        let params = ParamsData {
            mvp: gl_to_wgpu(&mvp_gl),
            clip,
            eye: [eye.0, eye.1, eye.2, if cam.perspective { 1.0 } else { 0.0 }],
            fwd: [fwd.0, fwd.1, fwd.2, 0.0],
            zrow: [proj[2 * 4 + 2], proj[3 * 4 + 2], proj[2 * 4 + 3], proj[3 * 4 + 3]],
            a: [
                if section::active_cut(scene).is_some() { 1.0 } else { 0.0 },
                size.0 as f32,
                size.1 as f32,
                o.coplanar_bias,
            ],
            b: [
                o.edge_tolerance_px,
                o.vertex_tolerance_px,
                if !o.visible_edges_only || scene.extra.show_hidden_geometry { 1.0 } else { 0.0 },
                if occlude { 1.0 } else { 0.0 },
            ],
        };
        self.queue.write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&params));
        self.ensure_targets(size);
        let targets = self.targets.as_ref().unwrap();

        let any_edges = draws.iter().any(|d| d.bind_group.is_some());
        let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("pick ids") });
        let face_pass = |enc: &mut wgpu::CommandEncoder,
                         label: &str,
                         color: Option<&wgpu::TextureView>,
                         depth: &wgpu::TextureView,
                         load_depth: bool,
                         store_depth: bool,
                         draw_geometry: bool,
                         pipeline: &wgpu::RenderPipeline| {
            let attachments = [color.and_then(color_attach)];
            let mut rp = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some(label),
                color_attachments: if color.is_some() { &attachments } else { &[] },
                depth_stencil_attachment: Some(depth_attach(depth, load_depth, store_depth)),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            rp.set_pipeline(pipeline);
            rp.set_bind_group(0, &targets.group0_plain, &[]);
            for d in &draws {
                let pm = &self.meshes[&d.mesh];
                let Some(tris) = pm.tris.as_ref() else { continue };
                if d.count == 0 || !draw_geometry {
                    continue;
                }
                rp.set_vertex_buffer(0, tris.slice(..));
                rp.set_vertex_buffer(1, inst_buf.slice(..));
                rp.draw(0..pm.n_verts, d.first..d.first + d.count);
            }
        };
        let faces_view = targets.faces.create_view(&Default::default());
        let edges_view = targets.edges.create_view(&Default::default());
        let verts_view = targets.verts.create_view(&Default::default());
        let gedges_view = targets.gedges.create_view(&Default::default());
        let groups_view = targets.groups.create_view(&Default::default());
        let layers = o.layers;
        let has_groups = layers.groups && places.iter().any(|p| !p.loose && p.sel);
        let gloose_view = targets.gloose.create_view(&Default::default());
        face_pass(&mut enc, "pick faces", Some(&faces_view), &targets.depth_pick, false, false, true, &self.pipe_face);
        face_pass(&mut enc, "pick groups", Some(&groups_view), &targets.depth_pick, false, true, has_groups, &self.pipe_group);
        face_pass(&mut enc, "pick group blockers", Some(&gloose_view), &targets.depth_pick, true, false, has_groups, &self.pipe_gloose);
        if occlude && any_edges && (layers.edges || layers.groups || layers.vertices) {
            face_pass(&mut enc, "pick occluders", None, &targets.depth_occ, false, true, true, &self.pipe_occ);
        }
        for (label, view, pipeline, per_edge) in
            [
                ("pick edges", &edges_view, &self.pipe_edge, 1u32),
                ("pick group edges", &gedges_view, &self.pipe_gedge, 1u32),
                ("pick vertices", &verts_view, &self.pipe_vertex, 2u32),
            ]
        {
            let mut rp = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some(label),
                color_attachments: &[color_attach(view)],
                depth_stencil_attachment: Some(depth_attach(&targets.depth_line, false, false)),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            rp.set_pipeline(pipeline);
            rp.set_bind_group(0, &targets.group0_depth, &[]);
            for d in &draws {
                let Some(bg) = d.bind_group.as_ref() else { continue };
                let wanted = match label {
                    "pick edges" => d.loose && layers.edges,
                    "pick group edges" => d.placed && layers.groups,
                    _ => layers.vertices,
                };
                if d.count == 0 || !wanted {
                    continue;
                }
                rp.set_bind_group(1, bg, &[]);
                let n = self.meshes[&d.mesh].n_edges * d.count * per_edge;
                rp.draw(0..6, 0..n);
            }
        }
        self.queue.submit([enc.finish()]);

        self.places = places;
        self.draws = draws;
        self.inst_buf = Some(inst_buf);
        self.view = Some(ViewState { size, mvp: mvp_gl });
    }

    fn ensure_targets(&mut self, size: (u32, u32)) {
        if self.targets.as_ref().is_some_and(|t| t.size == size) {
            return;
        }
        let extent = wgpu::Extent3d { width: size.0, height: size.1, depth_or_array_layers: 1 };
        let id_tex = |label: &str| {
            self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: extent,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: ID_FORMAT,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            })
        };
        let depth_tex = |label: &str, sampled: bool| {
            self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: extent,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: DEPTH,
                usage: if sampled {
                    wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING
                } else {
                    wgpu::TextureUsages::RENDER_ATTACHMENT
                },
                view_formats: &[],
            })
        };
        let view = |t: &wgpu::Texture| t.create_view(&Default::default());
        let depth_occ = view(&depth_tex("pick occluder depth", true));
        let group0_plain = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("pick params"),
            layout: &self.bgl_plain,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: self.params_buf.as_entire_binding() }],
        });
        let group0_depth = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("pick params + occluder"),
            layout: &self.bgl_depth,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.params_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&depth_occ) },
            ],
        });
        self.targets = Some(Targets {
            size,
            faces: id_tex("pick face ids"),
            edges: id_tex("pick edge ids"),
            verts: id_tex("pick vertex ids"),
            groups: id_tex("pick group ids"),
            gedges: id_tex("pick group edge ids"),
            gloose: id_tex("pick group blocker ids"),
            depth_pick: view(&depth_tex("pick depth", false)),
            depth_occ,
            depth_line: view(&depth_tex("pick line depth", false)),
            group0_plain,
            group0_depth,
        });
    }

    // ---- read-back ------------------------------------------------------------------------

    /// Read a rectangle of one layer (clamped to the viewport).
    pub fn read_layer(&mut self, layer: Layer, x: u32, y: u32, width: u32, height: u32) -> Option<IdImage> {
        let k = match layer {
            Layer::Faces => 0,
            Layer::Edges => 1,
            Layer::Vertices => 2,
            Layer::Groups => 3,
            Layer::GroupEdges => 4,
            Layer::GroupBlockers => 5,
        };
        let mut mask = [false; 6];
        mask[k] = true;
        self.read_mask(x, y, width, height, mask)?.swap_remove(k)
    }

    fn read_layers(&mut self, x: u32, y: u32, width: u32, height: u32, which: Layers) -> Option<Vec<Option<IdImage>>> {
        self.read_mask(x, y, width, height, [which.faces, which.edges, which.vertices, which.groups, false, which.groups])
    }

    /// Read the same rectangle of several layers with one submission. The
    /// result is `[faces, edges, vertices, groups, group edges]`-ordered,
    /// `None` for the layers not asked for.
    fn read_mask(&mut self, x: u32, y: u32, width: u32, height: u32, mask: [bool; 6]) -> Option<Vec<Option<IdImage>>> {
        let t = self.targets.as_ref()?;
        self.view.as_ref()?;
        let (w, h) = t.size;
        if x >= w || y >= h {
            return Some(vec![None, None, None, None, None, None]);
        }
        let (width, height) = (width.min(w - x).max(1), height.min(h - y).max(1));
        let bpr = (width * 8).div_ceil(256) * 256;
        let per = bpr as u64 * height as u64;
        let layers = [
            (mask[0], &t.faces),
            (mask[1], &t.edges),
            (mask[2], &t.verts),
            (mask[3], &t.groups),
            (mask[4], &t.gedges),
            (mask[5], &t.gloose),
        ];
        let n = layers.iter().filter(|l| l.0).count() as u64;
        let need = per * n;
        if self.staging.as_ref().is_none_or(|s| s.1 < need) {
            let cap = need.next_power_of_two().max(1024);
            self.staging = Some((
                self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("pick readback"),
                    size: cap,
                    usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                    mapped_at_creation: false,
                }),
                cap,
            ));
        }
        let buf = &self.staging.as_ref().unwrap().0;
        let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("pick readback") });
        let mut slot = 0u64;
        for (on, tex) in layers {
            if !on {
                continue;
            }
            enc.copy_texture_to_buffer(
                wgpu::TexelCopyTextureInfo { texture: tex, mip_level: 0, origin: wgpu::Origin3d { x, y, z: 0 }, aspect: wgpu::TextureAspect::All },
                wgpu::TexelCopyBufferInfo {
                    buffer: buf,
                    layout: wgpu::TexelCopyBufferLayout { offset: slot * per, bytes_per_row: Some(bpr), rows_per_image: Some(height) },
                },
                wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            );
            slot += 1;
        }
        self.queue.submit([enc.finish()]);
        let slice = buf.slice(..need);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
        rx.recv().ok()?.ok()?;
        let data = slice.get_mapped_range().ok()?;
        let mut out: Vec<Option<IdImage>> = vec![None, None, None, None, None, None];
        let mut slot = 0usize;
        for (k, on) in mask.into_iter().enumerate() {
            if !on {
                continue;
            }
            let base = slot * per as usize;
            let mut ids = Vec::with_capacity((width * height) as usize);
            for row in 0..height as usize {
                let line = &data[base + row * bpr as usize..base + row * bpr as usize + (width * 8) as usize];
                for px in line.chunks_exact(8) {
                    ids.push([
                        u32::from_le_bytes([px[0], px[1], px[2], px[3]]),
                        u32::from_le_bytes([px[4], px[5], px[6], px[7]]),
                    ]);
                }
            }
            out[k] = Some(IdImage { x, y, width, height, ids });
            slot += 1;
        }
        drop(data);
        buf.unmap();
        Some(out)
    }

    fn place(&self, id: u32) -> Option<&PlaceInfo> {
        self.places.get(id.checked_sub(1)? as usize)
    }

    fn face_hit(&self, ids: [u32; 2]) -> Option<FaceHit> {
        let p = self.place(ids[0])?;
        (ids[1] != 0).then(|| FaceHit { placement: p.path.clone(), owner: p.owner.clone(), mesh: p.mesh, face: FaceId(ids[1] - 1) })
    }

    fn edge_hit(&self, ids: [u32; 2]) -> Option<EdgeHit> {
        let p = self.place(ids[0])?;
        (ids[1] != 0).then(|| EdgeHit { placement: p.path.clone(), owner: p.owner.clone(), mesh: p.mesh, edge: EdgeId(ids[1] - 1) })
    }

    fn vertex_hit(&self, ids: [u32; 2]) -> Option<VertexHit> {
        let p = self.place(ids[0])?;
        (ids[1] != 0).then(|| VertexHit { placement: p.path.clone(), owner: p.owner.clone(), mesh: p.mesh, vertex: VertexId(ids[1] - 1) })
    }

    /// What every layer holds at the pixel containing `(x, y)` (viewport
    /// pixels, origin top-left). Needs a prior [`IdPicker::prepare`].
    pub fn pick_layers(&mut self, x: f64, y: f64) -> PixelHits {
        if !(x >= 0.0 && y >= 0.0) {
            return PixelHits::default();
        }
        let Some(layers) = self.read_layers(x.floor() as u32, y.floor() as u32, 1, 1, Layers::ALL) else {
            return PixelHits::default();
        };
        let first = |i: usize| layers[i].as_ref().map(|l| l.ids[0]);
        PixelHits {
            face: first(0).and_then(|v| self.face_hit(v)),
            edge: first(1).and_then(|v| self.edge_hit(v)),
            vertex: first(2).and_then(|v| self.vertex_hit(v)),
            group: if first(5).is_some_and(|b| b[1] != 0) { None } else { first(3).and_then(|v| self.face_hit(v)).and_then(|f| f.owner) },
        }
    }


    /// The group a click on the pixel selects (`Viewport.pick_group`): the
    /// owner of the nearest placed face, unless the loose part lies in front
    /// of it by more than the coplanar bias; with no placed face at all,
    /// the owner of the group edge in the band (a group of lines only).
    /// Needs a prior [`IdPicker::prepare`].
    ///
    /// One approximation next to the CPU: the group-edge fallback sees the
    /// nearest group edge of the band only, which is what the CPU does too,
    /// but both layers are built from what is visible at the edge's pixel.
    pub fn pick_group(&mut self, x: f64, y: f64) -> Option<GroupPath> {
        if !(x >= 0.0 && y >= 0.0) {
            return None;
        }
        let (px, py) = (x.floor() as u32, y.floor() as u32);
        let l = self.read_mask(px, py, 1, 1, [false, false, false, true, true, true])?;
        let placed = l[3].as_ref().and_then(|l| self.face_hit(l.ids[0]));
        let blocked = l[5].as_ref().is_some_and(|l| l.ids[0][1] != 0);
        match placed {
            Some(_) if blocked => None,
            Some(f) => f.owner,
            None => l[4].as_ref().and_then(|l| self.edge_hit(l.ids[0])).and_then(|e| e.owner),
        }
    }

    /// The topmost entity at the pixel: vertex, then edge, then face. This
    /// is a convenience; the Select tool's priority is the CPU picker's (see
    /// the module documentation).
    pub fn pick_pixel(&mut self, x: f64, y: f64) -> Option<Hit> {
        let h = self.pick_layers(x, y);
        h.vertex.map(Hit::Vertex).or(h.edge.map(Hit::Edge)).or(h.face.map(Hit::Face))
    }

    /// The face under the pixel, reading only the face layer.
    pub fn pick_face(&mut self, x: f64, y: f64) -> Option<FaceHit> {
        if !(x >= 0.0 && y >= 0.0) {
            return None;
        }
        let l = self.read_layers(x.floor() as u32, y.floor() as u32, 1, 1, Layers::FACES)?;
        l[0].as_ref().and_then(|l| self.face_hit(l.ids[0]))
    }

    /// What a rubber band takes from the *visible* picture (layers
    /// `faces` and `edges`; use [`IdPicker::pick_rect_layers`] for others).
    /// `rect` is `(x0, y0, x1, y1)` in viewport pixels.
    pub fn pick_rect(&mut self, rect: (f64, f64, f64, f64), mode: RectMode) -> Vec<Hit> {
        self.pick_rect_layers(rect, mode, Layers { faces: true, edges: true, vertices: false, groups: false })
    }

    /// [`IdPicker::pick_rect`] with the layers chosen. Crossing: every
    /// entity with a visible pixel (edge band included) in the rectangle.
    /// Window: of those, the ones whose whole outline projects inside it.
    /// Hits come in the order of first appearance, rows top to bottom,
    /// faces then edges then vertices.
    pub fn pick_rect_layers(&mut self, rect: (f64, f64, f64, f64), mode: RectMode, which: Layers) -> Vec<Hit> {
        let (x0, y0) = (rect.0.min(rect.2).max(0.0), rect.1.min(rect.3).max(0.0));
        let (x1, y1) = (rect.0.max(rect.2), rect.1.max(rect.3));
        if !(x1 > 0.0 && y1 > 0.0) {
            return Vec::new();
        }
        let (px, py) = (x0.floor() as u32, y0.floor() as u32);
        let (w, h) = ((x1.ceil() as u32).saturating_sub(px), (y1.ceil() as u32).saturating_sub(py));
        let Some(layers) = self.read_layers(px, py, w.max(1), h.max(1), which) else { return Vec::new() };
        let mut hits: Vec<Hit> = Vec::new();
        let mut seen: std::collections::HashSet<(u8, u32, u32)> = std::collections::HashSet::new();
        for (k, img) in layers.iter().enumerate() {
            let Some(img) = img else { continue };
            for &ids in &img.ids {
                if ids[1] == 0 || !seen.insert((k as u8, ids[0], ids[1])) {
                    continue;
                }
                let hit = match k {
                    0 => self.face_hit(ids).map(Hit::Face),
                    1 => self.edge_hit(ids).map(Hit::Edge),
                    2 => self.vertex_hit(ids).map(Hit::Vertex),
                    // Groups are not entities of a band.
                    _ => None,
                };
                if let Some(hit) = hit {
                    hits.push(hit);
                }
            }
        }
        if mode == RectMode::Window {
            let r = (x0, y0, x1, y1);
            hits.retain(|h| self.inside_window(h, r));
        }
        hits
    }

    /// Whether the whole outline of a hit projects inside `rect`.
    fn inside_window(&self, hit: &Hit, rect: (f64, f64, f64, f64)) -> bool {
        let Some(view) = &self.view else { return false };
        let project = |m: &Option<M4>, p: [f32; 3]| -> Option<(f64, f64)> {
            let q = m.map_or(V3(p[0], p[1], p[2]), |m| m.map(V3(p[0], p[1], p[2])));
            let v = [q.0, q.1, q.2, 1.0];
            let a = &view.mvp.m;
            let row = |r: usize| (0..4).map(|k| a[k][r] * v[k]).sum::<f32>();
            let c = [row(0), row(1), row(2), row(3)];
            if c[3] <= 0.0 {
                return None;
            }
            let (nx, ny) = (c[0] as f64 / c[3] as f64, c[1] as f64 / c[3] as f64);
            Some(((nx * 0.5 + 0.5) * view.size.0 as f64, (1.0 - (ny * 0.5 + 0.5)) * view.size.1 as f64))
        };
        let inside = |p: Option<(f64, f64)>| p.is_some_and(|p| rect.0 <= p.0 && p.0 <= rect.2 && rect.1 <= p.1 && p.1 <= rect.3);
        let xform_of = |path: &GroupPath, mesh: MeshId| {
            self.places.iter().find(|p| &p.path == path && p.mesh == mesh).and_then(|p| p.xform)
        };
        match hit {
            Hit::Face(f) => {
                let (Some(pm), m) = (self.meshes.get(&f.mesh), xform_of(&f.placement, f.mesh)) else { return false };
                let Some(&(start, n)) = pm.face_rings.get(&f.face.0) else { return false };
                pm.ring_pos[start as usize..(start + n) as usize].iter().all(|&p| inside(project(&m, p)))
            }
            Hit::Edge(e) => {
                let (Some(pm), m) = (self.meshes.get(&e.mesh), xform_of(&e.placement, e.mesh)) else { return false };
                let Some(&(a, b)) = pm.edge_ends.get(&e.edge.0) else { return false };
                inside(project(&m, a)) && inside(project(&m, b))
            }
            Hit::Vertex(_) => true,
        }
    }
}

// ---- keys -----------------------------------------------------------------------------------

fn env_key(scene: &Scene) -> u64 {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    format!("{:?}", scene.layers).hash(&mut h);
    scene.extra.show_hidden_geometry.hash(&mut h);
    h.finish()
}

fn camera_key(c: &Camera, h: &mut impl Hasher) {
    for v in [c.distance, c.yaw, c.pitch, c.fov_deg, c.aspect, c.znear, c.zfar] {
        v.to_bits().hash(h);
    }
    for v in [c.target, c.up] {
        [v.0.to_bits(), v.1.to_bits(), v.2.to_bits()].hash(h);
    }
    (c.perspective, c.two_point).hash(h);
}

fn group_key(g: &Group, h: &mut impl Hasher) {
    g.mesh.0.hash(h);
    (g.hidden, g.locked, g.billboard.as_ref().is_some_and(truthy)).hash(h);
    g.layer.hash(h);
    g.xform.map(|m| m.data().map(f32::to_bits)).hash(h);
    g.children.len().hash(h);
    for c in &g.children {
        group_key(c, h);
    }
}

/// Everything in the scene the ID layers depend on, hashed: mesh serials,
/// the group tree and the switches. Cheap next to a render.
fn scene_key(scene: &Scene, h: &mut impl Hasher) {
    scene.meshes.len().hash(h);
    for m in &scene.meshes {
        (m.serial, m.arena_sizes()).hash(h);
    }
    scene.groups.len().hash(h);
    for g in &scene.groups {
        group_key(g, h);
    }
    env_key(scene).hash(h);
    (scene.extra.show_hidden_objects, scene.extra.cuts_hidden).hash(h);
    for sp in &scene.section_planes {
        (sp.active, sp.normal.0.to_bits(), sp.normal.1.to_bits(), sp.normal.2.to_bits()).hash(h);
        (sp.point.0.to_bits(), sp.point.1.to_bits(), sp.point.2.to_bits()).hash(h);
    }
}

// ---- placements ---------------------------------------------------------------------------------

/// The placements the pass draws and what the cursor may do with each: the
/// loose part (or the open group's own mesh), then the group placements the
/// CPU index lists (`PickIndex::build`), with their selectable/visible
/// state.
fn build_places(st: &State) -> Vec<PlaceInfo> {
    let ctx = st.ctx.as_deref();
    let all = placements(st);
    let mut out: Vec<PlaceInfo> = Vec::new();
    match ctx {
        None => out.push(PlaceInfo {
            path: Vec::new(),
            owner: None,
            mesh: MeshId(0),
            xform: None,
            loose: true,
            sel: true,
            vis: true,
        }),
        Some(c) => {
            if let Some(p) = all.iter().find(|p| p.path == c) {
                out.push(PlaceInfo {
                    path: c.to_vec(),
                    owner: None,
                    mesh: p.mesh,
                    xform: p.xform,
                    loose: true,
                    sel: true,
                    vis: true,
                });
            }
        }
    }
    let context: Vec<&_> = match ctx {
        None => all.iter().collect(),
        Some(_) => all.iter().filter(|g| g.in_context && ctx != Some(g.owner.as_slice())).collect(),
    };
    let candidates: Vec<&_> = if ctx.is_none() || st.rest_hidden {
        context.clone()
    } else {
        all.iter().filter(|g| !g.is_ctx && ctx != Some(g.owner.as_slice())).collect()
    };
    for g in candidates {
        if g.billboard {
            continue;
        }
        let gvis = !g.hidden && st.layer_visible(g.layer.as_deref());
        let gsnap = group_selectable(st, g);
        let gsel = gsnap && context.iter().any(|c| c.path == g.path);
        if !(gvis || gsnap) {
            continue;
        }
        out.push(PlaceInfo {
            path: g.path.clone(),
            owner: Some(g.owner.clone()),
            mesh: g.mesh,
            xform: g.xform,
            loose: false,
            sel: gsel,
            vis: gvis,
        });
    }
    out
}

// ---- baking ----------------------------------------------------------------------------------------

fn bake_pick_mesh(device: &wgpu::Device, mesh: &Mesh, st: &State, stamp: (u64, (usize, usize, usize), u64)) -> PickMesh {
    let h = mesh.hypot;
    let mut verts: Vec<PickVert> = Vec::new();
    let mut face_rings: HashMap<u32, (u32, u32)> = HashMap::new();
    let mut ring_pos: Vec<[f32; 3]> = Vec::new();
    for &fid in &mesh.faces {
        let face = mesh.face(fid);
        let outer: Vec<V3> = face.outer.iter().map(|&v| mesh.position(v)).collect();
        if outer.len() < 3 {
            continue;
        }
        let mut flags = 0;
        if st.face_selectable(&face.attrs) {
            flags |= FF_LOOSE_SEL;
        }
        if st.face_visible(&face.attrs) {
            flags |= FF_LOOSE_VIS;
        }
        if !State::face_hidden(&face.attrs) {
            flags |= FF_PLACED;
        }
        if flags == 0 {
            continue;
        }
        let n = meshface::newell(&outer);
        let ln = n.length(h);
        let normal = if (ln as f64) > 1e-9 { n.normalized(h) } else { V3(0.0, 0.0, 1.0) };
        let area = 0.5 * ln;
        let holes: Vec<Vec<V3>> = face.holes.iter().map(|hl| hl.iter().map(|&v| mesh.position(v)).collect()).collect();
        let tris = triangulate(&outer, &holes, Some(normal), h);
        for t in &tris {
            for p in t {
                verts.push(PickVert { pos: [p.0, p.1, p.2], face: fid.0, flags, area });
            }
        }
        face_rings.insert(fid.0, (ring_pos.len() as u32, outer.len() as u32));
        ring_pos.extend(outer.iter().map(|p| [p.0, p.1, p.2]));
    }
    let mut edges: Vec<EdgeRec> = Vec::new();
    let mut edge_ends: HashMap<u32, ([f32; 3], [f32; 3])> = HashMap::new();
    for &eid in &mesh.edges {
        let e = mesh.edge(eid);
        let (a, b) = (mesh.position(e.v0), mesh.position(e.v1));
        let mut flags = 0;
        if st.layer_selectable(e.layer.as_deref()) && (!e.hidden || st.show_hidden_geometry()) {
            flags |= EF_LOOSE_SEL;
        }
        if e.soft {
            flags |= EF_SOFT;
        }
        if !e.hidden && !e.soft {
            flags |= EF_HARD;
        }
        edges.push(EdgeRec { a: [a.0, a.1, a.2], v0: e.v0.0, b: [b.0, b.1, b.2], v1: e.v1.0, id: eid.0, flags, pad: [0; 2] });
        edge_ends.insert(eid.0, ([a.0, a.1, a.2], [b.0, b.1, b.2]));
    }
    let buffer = |data: &[u8], usage: wgpu::BufferUsages, label: &str| {
        (!data.is_empty()).then(|| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some(label), contents: data, usage })
        })
    };
    PickMesh {
        stamp,
        n_verts: verts.len() as u32,
        tris: buffer(bytemuck::cast_slice(&verts), wgpu::BufferUsages::VERTEX, "pick triangles"),
        n_edges: edges.len() as u32,
        edges: buffer(bytemuck::cast_slice(&edges), wgpu::BufferUsages::STORAGE, "pick edges"),
        face_rings,
        ring_pos,
        edge_ends,
    }
}

fn depth_attach(view: &wgpu::TextureView, load: bool, store: bool) -> wgpu::RenderPassDepthStencilAttachment<'_> {
    wgpu::RenderPassDepthStencilAttachment {
        view,
        depth_ops: Some(wgpu::Operations {
            load: if load { wgpu::LoadOp::Load } else { wgpu::LoadOp::Clear(1.0) },
            store: if store { wgpu::StoreOp::Store } else { wgpu::StoreOp::Discard },
        }),
        stencil_ops: None,
    }
}

fn color_attach(view: &wgpu::TextureView) -> Option<wgpu::RenderPassColorAttachment<'_>> {
    Some(wgpu::RenderPassColorAttachment {
        view,
        depth_slice: None,
        resolve_target: None,
        ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT), store: wgpu::StoreOp::Store },
    })
}

