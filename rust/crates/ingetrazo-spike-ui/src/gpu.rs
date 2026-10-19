//! wgpu side of the viewport: pipelines, buffers and the egui_wgpu callback,
//! plus a tiny orbit camera (Z up, column-major matrices, depth 0..1).

use crate::model::{Mesh, Vertex};
use eframe::egui_wgpu::{self, wgpu, wgpu::util::DeviceExt};
use eframe::egui;

pub const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

#[derive(Clone, Copy)]
pub struct Camera {
    pub target: [f32; 3],
    pub dist: f32,
    pub yaw: f32,
    pub pitch: f32,
    pub radius: f32, // scene radius, for near/far
}

type M4 = [[f32; 4]; 4]; // m[col][row]

fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn norm(a: [f32; 3]) -> [f32; 3] {
    let l = dot(a, a).sqrt().max(1e-12);
    [a[0] / l, a[1] / l, a[2] / l]
}
fn mul(a: &M4, b: &M4) -> M4 {
    let mut r = [[0.0; 4]; 4];
    for c in 0..4 {
        for row in 0..4 {
            r[c][row] = (0..4).map(|k| a[k][row] * b[c][k]).sum();
        }
    }
    r
}

impl Camera {
    pub fn fit(center: [f32; 3], radius: f32) -> Self {
        Camera {
            target: center,
            dist: radius * 2.6,
            yaw: -0.8,
            pitch: 0.5,
            radius,
        }
    }

    pub fn eye(&self) -> [f32; 3] {
        let (cp, sp) = (self.pitch.cos(), self.pitch.sin());
        [
            self.target[0] + self.dist * cp * self.yaw.cos(),
            self.target[1] + self.dist * cp * self.yaw.sin(),
            self.target[2] + self.dist * sp,
        ]
    }

    /// Camera right and up vectors in world space.
    pub fn right_up(&self) -> ([f32; 3], [f32; 3]) {
        let f = norm(sub(self.target, self.eye()));
        let r = norm(cross(f, [0.0, 0.0, 1.0]));
        (r, cross(r, f))
    }

    pub fn view_proj(&self, aspect: f32) -> M4 {
        let eye = self.eye();
        let f = norm(sub(self.target, eye));
        let r = norm(cross(f, [0.0, 0.0, 1.0]));
        let u = cross(r, f);
        let view: M4 = [
            [r[0], u[0], -f[0], 0.0],
            [r[1], u[1], -f[1], 0.0],
            [r[2], u[2], -f[2], 0.0],
            [-dot(r, eye), -dot(u, eye), dot(f, eye), 1.0],
        ];
        let near = (self.dist - self.radius * 1.5).max(self.dist * 0.01).max(1e-3);
        let far = self.dist + self.radius * 2.0 + 1.0;
        let t = (45.0f32.to_radians() * 0.5).tan();
        let proj: M4 = [
            [1.0 / (t * aspect), 0.0, 0.0, 0.0],
            [0.0, 1.0 / t, 0.0, 0.0],
            [0.0, 0.0, far / (near - far), -1.0],
            [0.0, 0.0, near * far / (near - far), 0.0],
        ];
        mul(&proj, &view)
    }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    view_proj: [[f32; 4]; 4],
    light: [f32; 4],
}

const SHADER: &str = r#"
struct U { vp: mat4x4<f32>, light: vec4<f32> };
@group(0) @binding(0) var<uniform> u: U;

struct TriOut { @builtin(position) pos: vec4<f32>, @location(0) n: vec3<f32> };

@vertex fn vs_tri(@location(0) p: vec3<f32>, @location(1) n: vec3<f32>) -> TriOut {
    var o: TriOut;
    o.pos = u.vp * vec4<f32>(p, 1.0);
    o.n = n;
    return o;
}
@fragment fn fs_tri(i: TriOut) -> @location(0) vec4<f32> {
    let l = abs(dot(normalize(i.n), u.light.xyz));
    let c = vec3<f32>(0.78, 0.81, 0.86) * (0.30 + 0.70 * l);
    return vec4<f32>(c, 1.0);
}
@vertex fn vs_line(@location(0) p: vec3<f32>) -> @builtin(position) vec4<f32> {
    return u.vp * vec4<f32>(p, 1.0);
}
@fragment fn fs_line() -> @location(0) vec4<f32> {
    return vec4<f32>(0.08, 0.09, 0.12, 1.0);
}
"#;

pub struct Gpu {
    tri_pipe: wgpu::RenderPipeline,
    line_pipe: wgpu::RenderPipeline,
    uniform: wgpu::Buffer,
    bind: wgpu::BindGroup,
    tri_buf: Option<wgpu::Buffer>,
    tri_count: u32,
    line_buf: Option<wgpu::Buffer>,
    line_count: u32,
}

impl Gpu {
    pub fn new(rs: &egui_wgpu::RenderState) -> Self {
        let dev = &rs.device;
        let module = dev.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("spike"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });
        let bgl = dev.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("spike_bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let uniform = dev.create_buffer(&wgpu::BufferDescriptor {
            label: Some("spike_uniform"),
            size: std::mem::size_of::<Uniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind = dev.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("spike_bind"),
            layout: &bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform.as_entire_binding(),
            }],
        });
        let layout = dev.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("spike_layout"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });
        let make = |label: &str,
                    vs: &str,
                    fs: &str,
                    stride: u64,
                    attrs: &[wgpu::VertexAttribute],
                    topology: wgpu::PrimitiveTopology,
                    bias: wgpu::DepthBiasState| {
            dev.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module: &module,
                    entry_point: Some(vs),
                    compilation_options: Default::default(),
                    buffers: &[Some(wgpu::VertexBufferLayout {
                        array_stride: stride,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: attrs,
                    })],
                },
                primitive: wgpu::PrimitiveState {
                    topology,
                    ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: DEPTH_FORMAT,
                    depth_write_enabled: Some(true),
                    depth_compare: Some(wgpu::CompareFunction::LessEqual),
                    stencil: Default::default(),
                    bias,
                }),
                multisample: Default::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &module,
                    entry_point: Some(fs),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: rs.target_format,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                multiview_mask: None,
                cache: None,
            })
        };
        let tri_pipe = make(
            "spike_tris",
            "vs_tri",
            "fs_tri",
            std::mem::size_of::<Vertex>() as u64,
            &wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3],
            wgpu::PrimitiveTopology::TriangleList,
            // push faces slightly back so edges win the depth test
            wgpu::DepthBiasState {
                constant: 2,
                slope_scale: 1.5,
                clamp: 0.0,
            },
        );
        let line_pipe = make(
            "spike_lines",
            "vs_line",
            "fs_line",
            12,
            &wgpu::vertex_attr_array![0 => Float32x3],
            wgpu::PrimitiveTopology::LineList,
            wgpu::DepthBiasState::default(),
        );
        Gpu {
            tri_pipe,
            line_pipe,
            uniform,
            bind,
            tri_buf: None,
            tri_count: 0,
            line_buf: None,
            line_count: 0,
        }
    }

    pub fn upload(&mut self, rs: &egui_wgpu::RenderState, mesh: &Mesh) {
        let mk = |data: &[u8], label: &str| {
            (!data.is_empty()).then(|| {
                rs.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(label),
                    contents: data,
                    usage: wgpu::BufferUsages::VERTEX,
                })
            })
        };
        self.tri_buf = mk(bytemuck::cast_slice(&mesh.verts), "tris");
        self.tri_count = mesh.verts.len() as u32;
        self.line_buf = mk(bytemuck::cast_slice(&mesh.lines), "lines");
        self.line_count = mesh.lines.len() as u32;
    }
}

pub struct ViewportCallback {
    pub view_proj: [[f32; 4]; 4],
    pub light: [f32; 3],
}

impl egui_wgpu::CallbackTrait for ViewportCallback {
    fn prepare(
        &self,
        _device: &wgpu::Device,
        queue: &wgpu::Queue,
        _screen: &egui_wgpu::ScreenDescriptor,
        _encoder: &mut wgpu::CommandEncoder,
        resources: &mut egui_wgpu::CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        if let Some(gpu) = resources.get::<Gpu>() {
            let u = Uniforms {
                view_proj: self.view_proj,
                light: [self.light[0], self.light[1], self.light[2], 0.0],
            };
            queue.write_buffer(&gpu.uniform, 0, bytemuck::bytes_of(&u));
        }
        Vec::new()
    }

    fn paint(
        &self,
        _info: egui::PaintCallbackInfo,
        pass: &mut wgpu::RenderPass<'static>,
        resources: &egui_wgpu::CallbackResources,
    ) {
        let Some(gpu) = resources.get::<Gpu>() else {
            return;
        };
        pass.set_bind_group(0, &gpu.bind, &[]);
        if let Some(buf) = &gpu.tri_buf {
            pass.set_pipeline(&gpu.tri_pipe);
            pass.set_vertex_buffer(0, buf.slice(..));
            pass.draw(0..gpu.tri_count, 0..1);
        }
        if let Some(buf) = &gpu.line_buf {
            pass.set_pipeline(&gpu.line_pipe);
            pass.set_vertex_buffer(0, buf.slice(..));
            pass.draw(0..gpu.line_count, 0..1);
        }
    }
}
