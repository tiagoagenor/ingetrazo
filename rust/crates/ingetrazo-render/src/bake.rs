// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! Mesh to vertex data: what `Viewport._sync_edges` and `_group_chunk` put in
//! the VBOs, without any GPU in the way (so it is unit-testable).
//!
//! Faces are triangulated by `ingetrazo_geom::facegeom::face_geometry`, the
//! kernel that reproduces the viewport's `_face_geometry`. Each triangle
//! corner carries its colour already shaded the way `_shaded_color` does
//! (a diffuse term against the fixed world light, quantised), so the shader
//! only picks between vertex colour, texture and flat colour.
//!
//! The vertices are in the MESH's own coordinates: a component's prototype
//! is baked once and every placement draws it through its own matrix, as the
//! Python viewport's instanced path does (its shade is computed in local
//! coordinates too).

use std::collections::{BTreeMap, HashMap, HashSet};

use bytemuck::{Pod, Zeroable};
use ingetrazo_formats::pyjson::Value;
use ingetrazo_geom::facegeom::face_geometry;
use ingetrazo_math::{Hypot, V3};
use ingetrazo_model::materials::effective_attrs;
use ingetrazo_model::mesh::Mesh;
use ingetrazo_model::pyval::truthy;
use ingetrazo_model::scene::Scene;

use crate::style::DEFAULT_FACE_COLOR;

// ---- vertex formats ------------------------------------------------------------

/// A face vertex: 28 bytes.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct FaceVertex {
    pub pos: [f32; 3],
    pub uv: [f32; 2],
    /// Shaded colour (or white for textures), alpha unused.
    pub rgba: [u8; 4],
    /// `FLAG_*` bits 0..7, opacity 8..15 (255 = opaque), texture shade in
    /// 1/32 steps 16..21.
    pub misc: u32,
}

/// The face is seen from behind: the style's back tint replaces its colour
/// (a face whose back is unpainted, `back_is_default`).
pub const FLAG_BACK_DEFAULT: u32 = 1;
/// Draw only where the face is seen from the front (opaque front copy of a
/// face with a translucent back).
pub const FLAG_FRONT_ONLY: u32 = 2;
/// Draw only where the face is seen from behind (the back-side paint).
pub const FLAG_BACK_ONLY: u32 = 4;

pub fn pack_misc(flags: u32, opacity: f64, shade32: u32) -> u32 {
    let op = (opacity.clamp(0.0, 1.0) * 255.0).round() as u32;
    (flags & 0xff) | (op << 8) | ((shade32 & 0x3f) << 16)
}

/// An edge vertex: position plus the segment's last endpoint (the vertex
/// GL's default provoking convention would use for the dash origin).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct EdgeVertex {
    pub pos: [f32; 3],
    pub anchor: [f32; 3],
}

/// A soft edge vertex, with the planes of its two faces for the profile
/// test (`n.p = d` for each; `single` = 1 for an open boundary).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct SoftVertex {
    pub pos: [f32; 3],
    pub n0: [f32; 3],
    pub d0: f32,
    pub n1: [f32; 3],
    pub d1: f32,
    pub single: f32,
}

// ---- runs ------------------------------------------------------------------------

/// Which draw pass a run belongs to. The derived order is the draw order.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum RunClass {
    /// The model's faces (colour, then textured), with the glass-backed
    /// front copies.
    Main,
    /// The back-side paint, shown only from behind.
    Override,
    /// Faces with opacity below 1, blended after everything opaque.
    Translucent,
    /// Hidden things drawn as a see-through weave.
    Ghost,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Run {
    pub class: RunClass,
    /// Index into [`MeshBake::tex_paths`]; `None` = vertex colour.
    pub tex: Option<u32>,
    pub start: u32,
    pub count: u32,
}

#[derive(Clone, Debug, Default)]
pub struct MeshBake {
    pub verts: Vec<FaceVertex>,
    pub runs: Vec<Run>,
    pub tex_paths: Vec<String>,
    /// Hard (visible, non-soft) edges: 2 vertices per segment.
    pub hard: Vec<EdgeVertex>,
    /// Soft edges, the candidates of the profile pass.
    pub soft: Vec<SoftVertex>,
    /// Hidden edges, drawn dotted when hidden geometry is shown.
    pub ghost_edges: Vec<EdgeVertex>,
    pub lo: [f32; 3],
    pub hi: [f32; 3],
}

impl MeshBake {
    pub fn tri_count(&self) -> usize {
        self.verts.len() / 3
    }
    pub fn edge_count(&self) -> usize {
        self.hard.len() / 2
    }
    pub fn is_empty(&self) -> bool {
        self.verts.is_empty() && self.hard.is_empty() && self.soft.is_empty() && self.ghost_edges.is_empty()
    }
}

// ---- environment -----------------------------------------------------------------------

/// Layer visibility (`Scene._layer_state`): unknown names are visible.
#[derive(Clone, Debug, Default)]
pub struct LayerTable {
    hidden: HashSet<String>,
}

impl LayerTable {
    pub fn from_scene(scene: &Scene) -> LayerTable {
        let mut hidden = HashSet::new();
        for l in &scene.layers {
            if let (Some(Value::Str(name)), Some(vis)) = (l.get("name"), l.get("visible")) {
                if !truthy(vis) {
                    hidden.insert(name.clone());
                }
            }
        }
        LayerTable { hidden }
    }

    pub fn visible(&self, layer: Option<&str>) -> bool {
        match layer {
            Some(n) => !self.hidden.contains(n),
            None => true,
        }
    }
}

pub struct BakeEnv<'a> {
    pub layers: &'a LayerTable,
    /// `image_has_cutout(path)`: a texture with see-through pixels is
    /// translucent.
    pub cutout: &'a dyn Fn(&str) -> bool,
    /// View > Hidden Geometry: hidden faces and edges of this mesh are baked
    /// as ghosts.
    pub ghost_geometry: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BakeKind {
    Normal,
    /// A hidden object (View > Hidden Objects): everything is a ghost.
    GhostObject,
}

// ---- shading ---------------------------------------------------------------------------

/// `Viewport._LIGHT`: the fixed world light.
pub fn light(h: Hypot) -> V3 {
    V3(0.35, 0.25, 1.0).normalized(h)
}

/// `_shade_factor(normal)`: 0.62..1.0 against the fixed light, quantised to
/// 1/32.
pub fn shade_factor(normal: V3, h: Hypot) -> f64 {
    if (normal.length(h) as f64) < 1e-9 {
        return 0.84375;
    }
    let d = normal.normalized(h).dot(light(h)).abs() as f64;
    ((0.62 + 0.38 * d) * 32.0).round_ties_even() / 32.0
}

/// `_shaded_color(base, normal)` given the shade: clamped, quantised to 1/64.
pub fn shaded_color(base: [f64; 3], shade: f64) -> [f64; 3] {
    let q = |c: f64| ((c * shade).min(1.0) * 64.0).round_ties_even() / 64.0;
    [q(base[0]), q(base[1]), q(base[2])]
}

fn to_u8(c: f64) -> u8 {
    (c.clamp(0.0, 1.0) * 255.0).round() as u8
}

// ---- attrs -----------------------------------------------------------------------------

struct Attrs(Vec<(String, Value)>);

impl Attrs {
    fn get(&self, k: &str) -> Option<&Value> {
        self.0.iter().find(|(n, _)| n == k).map(|(_, v)| v)
    }
}

fn num(v: &Value) -> Option<f64> {
    match v {
        Value::Float(f) => Some(*f),
        Value::Int(s) => s.parse().ok(),
        Value::Bool(b) => Some(*b as i64 as f64),
        _ => None,
    }
}

fn rgb3(v: Option<&Value>) -> Option<[f64; 3]> {
    match v {
        Some(Value::Array(a)) if a.len() >= 3 => Some([num(&a[0])?, num(&a[1])?, num(&a[2])?]),
        _ => None,
    }
}

fn opacity_of(v: Option<&Value>) -> f64 {
    v.and_then(num).unwrap_or(1.0)
}

/// The texture dict of a material (`{"path", ...}`) when it names an image.
fn texture_of(v: Option<&Value>) -> Option<(&Value, &str)> {
    match v {
        Some(t @ Value::Object(_)) => match t.get("path") {
            Some(Value::Str(p)) if !p.is_empty() => Some((t, p.as_str())),
            _ => None,
        },
        _ => None,
    }
}

fn is_translucent(a: &Attrs, cutout: &dyn Fn(&str) -> bool) -> bool {
    if opacity_of(a.get("opacity")) < 0.999 {
        return true;
    }
    match texture_of(a.get("texture")) {
        Some((_, p)) => cutout(p),
        None => false,
    }
}

/// `back_is_default(attrs)`.
fn back_is_default(a: &Attrs, cutout: &dyn Fn(&str) -> bool) -> bool {
    if a.get("back").is_some_and(truthy) {
        return false;
    }
    !is_translucent(a, cutout)
}

// ---- texture mapping ----------------------------------------------------------------------

const VERTICAL_TOLERANCE: f64 = 1e-3;

/// `core.texture.projection_basis`: the .skp in-plane axes for a normal.
pub fn projection_basis(normal: [f64; 3]) -> ([f64; 3], [f64; 3]) {
    let (mut nx, mut ny, mut nz) = (normal[0], normal[1], normal[2]);
    let ln = (nx * nx + ny * ny + nz * nz).sqrt();
    if ln > 1e-30 {
        nx /= ln;
        ny /= ln;
        nz /= ln;
    }
    let (mut xx, mut xy) = (-ny, nx);
    let lx = (xx * xx + xy * xy).sqrt();
    if lx < VERTICAL_TOLERANCE {
        return if nz > 0.0 {
            ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0])
        } else {
            ([-1.0, 0.0, 0.0], [0.0, 1.0, 0.0])
        };
    }
    xx /= lx;
    xy /= lx;
    ([xx, xy, 0.0], [-nz * xy, nz * xx, nx * xy - ny * xx])
}

fn v3d(a: [f64; 3]) -> V3 {
    V3(a[0] as f32, a[1] as f32, a[2] as f32)
}

/// `core.texture.face_uv_axes(tex, normal)`: `u = gu.p + cu`, `v = gv.p + cv`.
pub fn face_uv_axes(tex: &Value, normal: V3) -> (V3, f64, V3, f64) {
    if let Some(Value::Array(u)) = tex.get("uvw") {
        if u.len() >= 8 {
            let n = |i: usize| num(&u[i]).unwrap_or(0.0);
            return (
                V3(n(0) as f32, n(1) as f32, n(2) as f32),
                n(3),
                V3(n(4) as f32, n(5) as f32, n(6) as f32),
                n(7),
            );
        }
    }
    let rot = tex.get("rot").and_then(num).unwrap_or(0.0);
    let (xr, yr) = projection_basis([normal.0 as f64, normal.1 as f64, normal.2 as f64]);
    let (mut u_axis, mut v_axis) = (v3d(xr), v3d(yr));
    if rot != 0.0 {
        let a = rot.to_radians();
        let (c, s) = (a.cos() as f32, a.sin() as f32);
        let nu = u_axis.mul(c).add(v_axis.mul(s));
        let nv = v_axis.mul(c).sub(u_axis.mul(s));
        u_axis = nu;
        v_axis = nv;
    }
    // `tex.get("sw", 1.0) or 1.0`
    let pick = |k: &str| match tex.get(k).and_then(num) {
        Some(x) if x != 0.0 => x,
        _ => 1.0,
    };
    let (sw, sh) = (pick("sw") as f32, pick("sh") as f32);
    (u_axis.div(sw), 0.0, v_axis.div(sh), 0.0)
}

// ---- the bake -------------------------------------------------------------------------------

struct Builder {
    buckets: BTreeMap<(RunClass, Option<u32>), Vec<FaceVertex>>,
    tex_ids: HashMap<String, u32>,
    tex_paths: Vec<String>,
}

impl Builder {
    fn tex_id(&mut self, path: &str) -> u32 {
        if let Some(&i) = self.tex_ids.get(path) {
            return i;
        }
        let i = self.tex_paths.len() as u32;
        self.tex_paths.push(path.to_string());
        self.tex_ids.insert(path.to_string(), i);
        i
    }

    fn push_tris(
        &mut self,
        key: (RunClass, Option<u32>),
        tris: &[[u32; 3]],
        pos: &[V3],
        rgb: [u8; 3],
        misc: u32,
        uv: Option<&dyn Fn(V3) -> [f32; 2]>,
    ) {
        let out = self.buckets.entry(key).or_default();
        for t in tris {
            for &i in t {
                let p = pos[i as usize];
                out.push(FaceVertex {
                    pos: [p.0, p.1, p.2],
                    uv: uv.map_or([0.0, 0.0], |f| f(p)),
                    rgba: [rgb[0], rgb[1], rgb[2], 255],
                    misc,
                });
            }
        }
    }
}

/// Bake a mesh. `paint` is the container material of the placement
/// (`effective_material`), which dresses the faces still wearing the
/// default material.
pub fn bake_mesh(mesh: &Mesh, paint: Option<&Value>, env: &BakeEnv, kind: BakeKind) -> MeshBake {
    let h = mesh.hypot;
    let mut b = Builder { buckets: BTreeMap::new(), tex_ids: HashMap::new(), tex_paths: Vec::new() };
    let mut lo = [f32::MAX; 3];
    let mut hi = [f32::MIN; 3];
    let mut grow = |p: V3| {
        lo = [lo[0].min(p.0), lo[1].min(p.1), lo[2].min(p.2)];
        hi = [hi[0].max(p.0), hi[1].max(p.1), hi[2].max(p.2)];
    };
    let ghost_object = kind == BakeKind::GhostObject;

    for &fid in &mesh.faces {
        let face = mesh.face(fid);
        let own = Attrs(face.attrs.clone());
        let layer = match own.get("layer") {
            Some(Value::Str(s)) => Some(s.as_str()),
            _ => None,
        };
        if !env.layers.visible(layer) {
            continue;
        }
        let hidden = own.get("hidden").is_some_and(truthy);
        let ghost = ghost_object || (hidden && env.ghost_geometry);
        if hidden && !ghost {
            continue;
        }
        if face.outer.len() < 3 {
            continue;
        }
        // Ring positions and the triangulation.
        let mut pos: Vec<V3> = Vec::with_capacity(face.outer.len() + 4);
        let mut rings_idx: Vec<Vec<u32>> = Vec::new();
        for ring in std::iter::once(&face.outer).chain(face.holes.iter()) {
            let start = pos.len() as u32;
            for &v in ring {
                pos.push(mesh.position(v));
            }
            rings_idx.push((start..pos.len() as u32).collect());
        }
        let rings: Vec<&[u32]> = rings_idx.iter().map(|r| r.as_slice()).collect();
        let geom = face_geometry(&pos, &rings, h, true);
        let tris: &[[u32; 3]] = geom.tris_unit.as_deref().unwrap_or(&geom.tris);
        if tris.is_empty() {
            continue;
        }
        for &i in rings[0] {
            grow(pos[i as usize]);
        }
        let shade = shade_factor(geom.unit, h);

        // The attrs the face is DRAWN with inside its container.
        let attrs = match paint {
            Some(m) if truthy(m) => {
                let merged = effective_attrs(&Value::Object(face.attrs.clone()), m)
                    .unwrap_or_else(|_| Value::Object(face.attrs.clone()));
                match merged {
                    Value::Object(items) => Attrs(items),
                    _ => own,
                }
            }
            _ => own,
        };

        if ghost {
            // A see-through weave: the face's shaded colour; a textured one
            // is plain grey.
            let rgb = if texture_of(attrs.get("texture")).is_some() {
                [to_u8(0.62); 3]
            } else {
                let base = rgb3(attrs.get("color")).unwrap_or(rgb_f64(DEFAULT_FACE_COLOR));
                let c = shaded_color(base, shade);
                [to_u8(c[0]), to_u8(c[1]), to_u8(c[2])]
            };
            b.push_tris((RunClass::Ghost, None), tris, &pos, rgb, pack_misc(0, 1.0, 0), None);
            continue;
        }

        let cutout = env.cutout;
        let normal_for_uv = geom.normal;
        // Back-side paint (`bucket_back`).
        let mut fcull = false;
        if let Some(back @ Value::Object(_)) = attrs.get("back") {
            let op = opacity_of(back.get("opacity"));
            if let Some((tex, path)) = texture_of(back.get("texture")) {
                let (gu, cu, gv, cv) = face_uv_axes(tex, normal_for_uv);
                let id = b.tex_id(path);
                let uvf = move |p: V3| [(gu.dot(p) as f64 + cu) as f32, (gv.dot(p) as f64 + cv) as f32];
                let class = if op < 0.999 { RunClass::Translucent } else { RunClass::Override };
                fcull = op < 0.999;
                let misc = pack_misc(FLAG_BACK_ONLY, op, (shade * 32.0).round() as u32);
                b.push_tris((class, Some(id)), tris, &pos, [255; 3], misc, Some(&uvf));
            } else if let Some(col) = rgb3(back.get("color")) {
                let c = shaded_color(col, shade);
                let class = if op < 0.999 { RunClass::Translucent } else { RunClass::Override };
                fcull = op < 0.999;
                let misc = pack_misc(FLAG_BACK_ONLY, op, 0);
                b.push_tris((class, None), tris, &pos, [to_u8(c[0]), to_u8(c[1]), to_u8(c[2])], misc, None);
            }
        }
        let default_back = back_is_default(&attrs, cutout);
        let op = opacity_of(attrs.get("opacity"));
        let tex = texture_of(attrs.get("texture"));
        if let Some((tex, path)) = tex {
            let (gu, cu, gv, cv) = face_uv_axes(tex, normal_for_uv);
            let id = b.tex_id(path);
            let uvf = move |p: V3| [(gu.dot(p) as f64 + cu) as f32, (gv.dot(p) as f64 + cv) as f32];
            let mut flags = if default_back { FLAG_BACK_DEFAULT } else { 0 };
            let (class, opacity) = if op < 0.999 {
                if fcull {
                    flags |= FLAG_FRONT_ONLY;
                }
                (RunClass::Translucent, op)
            } else {
                (RunClass::Main, 1.0)
            };
            let misc = pack_misc(flags, opacity, (shade * 32.0).round() as u32);
            b.push_tris((class, Some(id)), tris, &pos, [255; 3], misc, Some(&uvf));
        } else {
            let base = rgb3(attrs.get("color")).unwrap_or(rgb_f64(DEFAULT_FACE_COLOR));
            let c = shaded_color(base, shade);
            let rgb = [to_u8(c[0]), to_u8(c[1]), to_u8(c[2])];
            if op < 0.999 {
                let flags = if fcull { FLAG_FRONT_ONLY } else { 0 };
                b.push_tris((RunClass::Translucent, None), tris, &pos, rgb, pack_misc(flags, op, 0), None);
            } else if fcull {
                b.push_tris((RunClass::Main, None), tris, &pos, rgb, pack_misc(FLAG_FRONT_ONLY, 1.0, 0), None);
            } else {
                let flags = if default_back { FLAG_BACK_DEFAULT } else { 0 };
                b.push_tris((RunClass::Main, None), tris, &pos, rgb, pack_misc(flags, 1.0, 0), None);
            }
        }
    }

    // Edges.
    let mut hard: Vec<EdgeVertex> = Vec::new();
    let mut soft: Vec<SoftVertex> = Vec::new();
    let mut ghost_edges: Vec<EdgeVertex> = Vec::new();
    for &eid in &mesh.edges {
        let e = mesh.edge(eid);
        if !env.layers.visible(e.layer.as_deref()) {
            continue;
        }
        let (a, bb) = (mesh.position(e.v0), mesh.position(e.v1));
        let seg = [
            EdgeVertex { pos: [a.0, a.1, a.2], anchor: [bb.0, bb.1, bb.2] },
            EdgeVertex { pos: [bb.0, bb.1, bb.2], anchor: [bb.0, bb.1, bb.2] },
        ];
        if ghost_object {
            if !e.soft && !e.hidden {
                ghost_edges.extend(seg);
            }
            continue;
        }
        if e.hidden {
            if env.ghost_geometry {
                ghost_edges.extend(seg);
            }
            continue;
        }
        grow(a);
        grow(bb);
        if !e.soft {
            hard.extend(seg);
            continue;
        }
        if e.faces.is_empty() {
            continue;
        }
        let plane = |f: ingetrazo_model::FaceId| -> ([f32; 3], f32) {
            let outer = &mesh.face(f).outer;
            if outer.len() < 3 {
                return ([0.0; 3], 0.0);
            }
            let p: Vec<[f64; 3]> = outer[..3]
                .iter()
                .map(|&v| {
                    let q = mesh.position(v);
                    [q.0 as f64, q.1 as f64, q.2 as f64]
                })
                .collect();
            let u = [p[1][0] - p[0][0], p[1][1] - p[0][1], p[1][2] - p[0][2]];
            let w = [p[2][0] - p[0][0], p[2][1] - p[0][1], p[2][2] - p[0][2]];
            let mut n = [u[1] * w[2] - u[2] * w[1], u[2] * w[0] - u[0] * w[2], u[0] * w[1] - u[1] * w[0]];
            // Normalised: the sign test is unchanged and f32 keeps its bits.
            let l = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
            if l > 0.0 {
                n = [n[0] / l, n[1] / l, n[2] / l];
            }
            let d = n[0] * p[0][0] + n[1] * p[0][1] + n[2] * p[0][2];
            ([n[0] as f32, n[1] as f32, n[2] as f32], d as f32)
        };
        let (n0, d0) = plane(e.faces[0]);
        let single = e.faces.len() != 2;
        let (n1, d1) = plane(if e.faces.len() == 2 { e.faces[1] } else { e.faces[0] });
        for p in [a, bb] {
            soft.push(SoftVertex {
                pos: [p.0, p.1, p.2],
                n0,
                d0,
                n1,
                d1,
                single: if single { 1.0 } else { 0.0 },
            });
        }
    }

    // Lay the runs out in draw order.
    let mut verts: Vec<FaceVertex> = Vec::new();
    let mut runs: Vec<Run> = Vec::new();
    for ((class, tex), v) in b.buckets {
        if v.is_empty() {
            continue;
        }
        runs.push(Run { class, tex, start: verts.len() as u32, count: v.len() as u32 });
        verts.extend(v);
    }
    if lo[0] > hi[0] {
        lo = [0.0; 3];
        hi = [0.0; 3];
    }
    MeshBake { verts, runs, tex_paths: b.tex_paths, hard, soft, ghost_edges, lo, hi }
}

fn rgb_f64(c: [f32; 3]) -> [f64; 3] {
    [c[0] as f64, c[1] as f64, c[2] as f64]
}

/// The triangles of one face in its mesh's coordinates (the viewport's
/// `_tris_of`), for the selection cues.
pub fn face_tris(mesh: &Mesh, f: ingetrazo_model::FaceId) -> Vec<[V3; 3]> {
    let face = mesh.face(f);
    if face.outer.len() < 3 {
        return Vec::new();
    }
    let mut pos: Vec<V3> = Vec::new();
    let mut rings_idx: Vec<Vec<u32>> = Vec::new();
    for ring in std::iter::once(&face.outer).chain(face.holes.iter()) {
        let start = pos.len() as u32;
        for &v in ring {
            pos.push(mesh.position(v));
        }
        rings_idx.push((start..pos.len() as u32).collect());
    }
    let rings: Vec<&[u32]> = rings_idx.iter().map(|r| r.as_slice()).collect();
    let geom = face_geometry(&pos, &rings, mesh.hypot, true);
    let tris: &[[u32; 3]] = geom.tris_unit.as_deref().unwrap_or(&geom.tris);
    tris.iter().map(|t| [pos[t[0] as usize], pos[t[1] as usize], pos[t[2] as usize]]).collect()
}

/// The dot colour of the selection cue for a surface painted `surface`
/// (`selection_dot_color`): orange, unless the surface is itself orange or
/// red, where the dots turn blue.
pub fn selection_dot_color(surface: Option<[f64; 3]>) -> [f32; 3] {
    const ORANGE: [f32; 3] = [0.95, 0.45, 0.16];
    const BLUE: [f32; 3] = [0.10, 0.25, 0.85];
    let Some(s) = surface else { return ORANGE };
    let (r, g, b) = (s[0], s[1], s[2]);
    let maxc = r.max(g).max(b);
    let minc = r.min(g).min(b);
    if minc == maxc {
        return ORANGE;
    }
    let sat = (maxc - minc) / maxc;
    let d = maxc - minc;
    let (rc, gc, bc) = ((maxc - r) / d, (maxc - g) / d, (maxc - b) / d);
    let hh = if r == maxc {
        bc - gc
    } else if g == maxc {
        2.0 + rc - bc
    } else {
        4.0 + gc - rc
    };
    let hue = (hh / 6.0).rem_euclid(1.0);
    if sat >= 0.35 && maxc >= 0.35 && (hue <= 0.14 || hue >= 0.96) {
        BLUE
    } else {
        ORANGE
    }
}

/// `(front_dot, back_dot)` for a face: each side's dot colour picked against
/// the paint that side shows (`_selection_dot_colors`).
pub fn selection_dot_colors(
    attrs: &[(String, Value)],
    paint: Option<&Value>,
    back_tint: [f32; 3],
) -> ([f32; 3], [f32; 3]) {
    let merged = match paint {
        Some(m) if truthy(m) => match effective_attrs(&Value::Object(attrs.to_vec()), m) {
            Ok(Value::Object(items)) => items,
            _ => attrs.to_vec(),
        },
        _ => attrs.to_vec(),
    };
    let a = Attrs(merged);
    let front = if texture_of(a.get("texture")).is_some() {
        None
    } else {
        Some(rgb3(a.get("color")).unwrap_or(rgb_f64(DEFAULT_FACE_COLOR)))
    };
    let back = match a.get("back") {
        Some(Value::Bool(true)) => front,
        Some(b @ Value::Object(_)) => rgb3(b.get("color")),
        _ => Some([back_tint[0] as f64, back_tint[1] as f64, back_tint[2] as f64]),
    };
    (selection_dot_color(front), selection_dot_color(back))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn env<'a>(layers: &'a LayerTable, cut: &'a dyn Fn(&str) -> bool) -> BakeEnv<'a> {
        BakeEnv { layers, cutout: cut, ghost_geometry: false }
    }

    fn cube(mesh: &mut Mesh) {
        let p = |x: f32, y: f32, z: f32| V3(x, y, z);
        let q = [
            p(0., 0., 0.), p(1., 0., 0.), p(1., 1., 0.), p(0., 1., 0.),
            p(0., 0., 1.), p(1., 0., 1.), p(1., 1., 1.), p(0., 1., 1.),
        ];
        let faces: [[usize; 4]; 6] = [
            [0, 3, 2, 1], [4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7],
        ];
        for f in faces {
            let ring: Vec<V3> = f.iter().map(|&i| q[i]).collect();
            mesh.add_face(&ring, &[]);
        }
        for (a, bb) in [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)] {
            let _ = mesh.add_edge(q[a], q[bb]);
        }
    }

    #[test]
    fn a_cube_bakes_to_twelve_triangles_and_twelve_edges() {
        let mut m = Mesh::new();
        cube(&mut m);
        let layers = LayerTable::default();
        let cut = |_: &str| false;
        let bk = bake_mesh(&m, None, &env(&layers, &cut), BakeKind::Normal);
        assert_eq!(bk.tri_count(), 12);
        assert_eq!(bk.edge_count(), 12);
        assert_eq!(bk.runs.len(), 1);
        assert_eq!(bk.runs[0].class, RunClass::Main);
        assert!(bk.verts.iter().all(|v| v.misc & FLAG_BACK_DEFAULT != 0));
        assert_eq!(bk.lo, [0.0; 3]);
        assert_eq!(bk.hi, [1.0; 3]);
    }

    #[test]
    fn top_faces_are_lighter_than_sides() {
        let h = Hypot::default();
        let top = shade_factor(V3(0.0, 0.0, 1.0), h);
        let side = shade_factor(V3(1.0, 0.0, 0.0), h);
        assert!(top > side);
        assert!(top <= 1.0 && side >= 0.62);
        // Quantised to 1/32.
        assert_eq!((top * 32.0).fract(), 0.0);
    }

    #[test]
    fn painted_and_translucent_faces_land_in_their_classes() {
        let mut m = Mesh::new();
        let f = m.add_face(&[V3(0., 0., 0.), V3(1., 0., 0.), V3(1., 1., 0.)], &[]);
        m.face_mut(f).attrs = vec![
            ("color".into(), Value::Array(vec![Value::Float(1.0), Value::Float(0.0), Value::Float(0.0)])),
            ("opacity".into(), Value::Float(0.5)),
        ];
        let layers = LayerTable::default();
        let cut = |_: &str| false;
        let bk = bake_mesh(&m, None, &env(&layers, &cut), BakeKind::Normal);
        assert_eq!(bk.runs.len(), 1);
        assert_eq!(bk.runs[0].class, RunClass::Translucent);
        assert_eq!((bk.verts[0].misc >> 8) & 0xff, 128);
        assert_eq!(bk.verts[0].misc & FLAG_BACK_DEFAULT, 0, "translucent: paint on both sides");
    }

    #[test]
    fn projection_basis_matches_the_skp_rule() {
        assert_eq!(projection_basis([0.0, 0.0, 1.0]), ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0]));
        assert_eq!(projection_basis([0.0, 0.0, -1.0]), ([-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]));
        let (u, v) = projection_basis([0.0, 1.0, 0.0]);
        assert_eq!(u, [-1.0, 0.0, 0.0]);
        assert!((v[2] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn hidden_layers_and_faces_are_skipped() {
        let mut m = Mesh::new();
        let f = m.add_face(&[V3(0., 0., 0.), V3(1., 0., 0.), V3(1., 1., 0.)], &[]);
        m.face_mut(f).attrs = vec![("hidden".into(), Value::Bool(true))];
        let layers = LayerTable::default();
        let cut = |_: &str| false;
        let bk = bake_mesh(&m, None, &env(&layers, &cut), BakeKind::Normal);
        assert!(bk.verts.is_empty());
        let e = BakeEnv { layers: &layers, cutout: &cut, ghost_geometry: true };
        let bk = bake_mesh(&m, None, &e, BakeKind::Normal);
        assert_eq!(bk.runs[0].class, RunClass::Ghost);
    }
}
