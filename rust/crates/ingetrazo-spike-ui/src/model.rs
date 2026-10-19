//! Spike model loading: .igz -> flat triangle soup + line list, plus a
//! synthetic stress scene. Fan triangulation, holes ignored (spike only).

use bytemuck::{Pod, Zeroable};
use ingetrazo_formats::igz;
use ingetrazo_formats::pyjson::Value;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct Vertex {
    pub pos: [f32; 3],
    pub normal: [f32; 3],
}

pub type Mat4 = [f32; 16]; // column-major, as in the .igz "xform"

pub struct GroupInfo {
    pub name: String,
    pub proto: Option<usize>,
    pub tris: usize,
    pub edges: usize,
    pub min: [f32; 3],
    pub max: [f32; 3],
}

#[derive(Default)]
pub struct Mesh {
    pub verts: Vec<Vertex>,   // 3 per triangle, flat normal
    pub lines: Vec<[f32; 3]>, // 2 per edge
    pub groups: Vec<GroupInfo>,
}

impl Mesh {
    pub fn tri_count(&self) -> usize {
        self.verts.len() / 3
    }
    pub fn edge_count(&self) -> usize {
        self.lines.len() / 2
    }

    /// Bounding sphere (centre, radius) over all vertices and lines.
    pub fn bounds(&self) -> ([f32; 3], f32) {
        let mut min = [f32::MAX; 3];
        let mut max = [f32::MIN; 3];
        let pts = self
            .verts
            .iter()
            .map(|v| v.pos)
            .chain(self.lines.iter().copied());
        for p in pts {
            for i in 0..3 {
                min[i] = min[i].min(p[i]);
                max[i] = max[i].max(p[i]);
            }
        }
        if min[0] > max[0] {
            return ([0.0; 3], 1.0);
        }
        let c = [
            (min[0] + max[0]) * 0.5,
            (min[1] + max[1]) * 0.5,
            (min[2] + max[2]) * 0.5,
        ];
        let d = [max[0] - min[0], max[1] - min[1], max[2] - min[2]];
        (c, (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt() * 0.5)
    }

    /// Fan-triangulate a polygon (assumed convex/planar) with a Newell normal.
    fn add_face(&mut self, poly: &[[f32; 3]]) -> usize {
        if poly.len() < 3 {
            return 0;
        }
        let mut n = [0.0f32; 3];
        for i in 0..poly.len() {
            let a = poly[i];
            let b = poly[(i + 1) % poly.len()];
            n[0] += (a[1] - b[1]) * (a[2] + b[2]);
            n[1] += (a[2] - b[2]) * (a[0] + b[0]);
            n[2] += (a[0] - b[0]) * (a[1] + b[1]);
        }
        let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
        if len > 0.0 {
            n = [n[0] / len, n[1] / len, n[2] / len];
        }
        for i in 1..poly.len() - 1 {
            for p in [poly[0], poly[i], poly[i + 1]] {
                self.verts.push(Vertex { pos: p, normal: n });
            }
        }
        poly.len() - 2
    }
}

fn num(v: &Value) -> f32 {
    match v {
        Value::Float(f) => *f as f32,
        Value::Int(s) => s.parse::<f32>().unwrap_or(0.0),
        _ => 0.0,
    }
}

fn point(v: &Value) -> Option<[f32; 3]> {
    match v {
        Value::Array(a) if a.len() >= 3 => Some([num(&a[0]), num(&a[1]), num(&a[2])]),
        _ => None,
    }
}

fn xform(v: Option<&Value>) -> Mat4 {
    let mut m = [0.0f32; 16];
    m[0] = 1.0;
    m[5] = 1.0;
    m[10] = 1.0;
    m[15] = 1.0;
    if let Some(Value::Array(a)) = v {
        if a.len() == 16 {
            for (i, x) in a.iter().enumerate() {
                m[i] = num(x);
            }
        }
    }
    m
}

fn apply(m: &Mat4, p: [f32; 3]) -> [f32; 3] {
    [
        m[0] * p[0] + m[4] * p[1] + m[8] * p[2] + m[12],
        m[1] * p[0] + m[5] * p[1] + m[9] * p[2] + m[13],
        m[2] * p[0] + m[6] * p[1] + m[10] * p[2] + m[14],
    ]
}

/// Adds one container's own edges/faces (not nested groups); returns (tris, edges).
fn add_geometry(mesh: &mut Mesh, node: &Value, m: &Mat4) -> (usize, usize) {
    let mut tris = 0;
    let mut edges = 0;
    if let Some(Value::Array(es)) = node.get("edges") {
        for e in es {
            if let (Some(a), Some(b)) = (
                e.get("a").and_then(point),
                e.get("b").and_then(point),
            ) {
                mesh.lines.push(apply(m, a));
                mesh.lines.push(apply(m, b));
                edges += 1;
            }
        }
    }
    if let Some(Value::Array(fs)) = node.get("faces") {
        for f in fs {
            if let Some(Value::Array(vs)) = f.get("vertices") {
                let poly: Vec<[f32; 3]> = vs.iter().filter_map(point).map(|p| apply(m, p)).collect();
                tris += mesh.add_face(&poly);
            }
        }
    }
    (tris, edges)
}

fn group_bounds(mesh: &Mesh, v0: usize, l0: usize) -> ([f32; 3], [f32; 3]) {
    let mut min = [f32::MAX; 3];
    let mut max = [f32::MIN; 3];
    let pts = mesh.verts[v0..]
        .iter()
        .map(|v| v.pos)
        .chain(mesh.lines[l0..].iter().copied());
    for p in pts {
        for i in 0..3 {
            min[i] = min[i].min(p[i]);
            max[i] = max[i].max(p[i]);
        }
    }
    (min, max)
}

pub fn load_igz(path: &str) -> Result<Mesh, String> {
    let raw = std::fs::read(path).map_err(|e| format!("{path}: {e}"))?;
    let doc = igz::read_document(&raw).map_err(|e| format!("{path}: {e:?}"))?;
    let scene = doc.get("scene").ok_or("no scene")?;
    let mut mesh = Mesh::default();
    let ident = xform(None);

    let (v0, l0) = (mesh.verts.len(), mesh.lines.len());
    let (t, e) = add_geometry(&mut mesh, scene, &ident);
    if t + e > 0 {
        let (min, max) = group_bounds(&mesh, v0, l0);
        mesh.groups.push(GroupInfo {
            name: "(scene root)".into(),
            proto: None,
            tris: t,
            edges: e,
            min,
            max,
        });
    }

    let protos: &[Value] = match scene.get("protos") {
        Some(Value::Array(p)) => p,
        _ => &[],
    };
    if let Some(Value::Array(groups)) = scene.get("groups") {
        for (i, g) in groups.iter().enumerate() {
            let (v0, l0) = (mesh.verts.len(), mesh.lines.len());
            let m = xform(g.get("xform"));
            let proto = g.get("proto").and_then(Value::as_i64).map(|p| p as usize);
            let (t, e) = match proto.and_then(|p| protos.get(p)) {
                Some(p) => add_geometry(&mut mesh, p, &m),
                None => add_geometry(&mut mesh, g, &m),
            };
            let (min, max) = group_bounds(&mesh, v0, l0);
            let name = match g.get("name") {
                Some(Value::Str(s)) if !s.is_empty() => s.clone(),
                _ => format!("Group {i}"),
            };
            mesh.groups.push(GroupInfo {
                name,
                proto,
                tris: t,
                edges: e,
                min,
                max,
            });
        }
    }
    Ok(mesh)
}

/// `n x n` extruded regular `sides`-gons (prisms) on a grid: with n = 45 and
/// sides = 128 that is 2025 prisms ~ 1.03M triangles.
pub fn stress_scene(n: usize, sides: usize) -> Mesh {
    let mut mesh = Mesh::default();
    let spacing = 3.0f32;
    for gy in 0..n {
        for gx in 0..n {
            let (v0, l0) = (mesh.verts.len(), mesh.lines.len());
            let cx = gx as f32 * spacing;
            let cy = gy as f32 * spacing;
            let h = 1.0 + ((gx * 7 + gy * 13) % 5) as f32 * 0.6;
            let ring = |z: f32| -> Vec<[f32; 3]> {
                (0..sides)
                    .map(|k| {
                        let a = k as f32 / sides as f32 * std::f32::consts::TAU;
                        [cx + a.cos(), cy + a.sin(), z]
                    })
                    .collect()
            };
            let (b, t) = (ring(0.0), ring(h));
            let mut tris = 0;
            for k in 0..sides {
                let k2 = (k + 1) % sides;
                tris += mesh.add_face(&[b[k], b[k2], t[k2], t[k]]);
                mesh.lines.extend_from_slice(&[b[k], b[k2], t[k], t[k2], b[k], t[k]]);
            }
            let rb: Vec<_> = b.iter().rev().copied().collect();
            tris += mesh.add_face(&rb);
            tris += mesh.add_face(&t);
            let (min, max) = group_bounds(&mesh, v0, l0);
            mesh.groups.push(GroupInfo {
                name: format!("Prism {gx},{gy}"),
                proto: None,
                tris,
                edges: sides * 3,
                min,
                max,
            });
        }
    }
    mesh
}
