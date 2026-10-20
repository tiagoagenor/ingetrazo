// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! The Paint (bucket) tool (`tools/paint.py`): assign a material to a face.
//!
//! A click paints the face under the cursor with the current paint (a solid
//! colour, or a texture; the translucency and the material identity ride
//! along), the whole current face selection when the face belongs to it, or
//! the whole curved surface (faces joined by soft edges) it sits on. The
//! side under the cursor is the one painted (`attrs["back"]` for a back).
//! A face inside a group or component, clicked from outside, paints the
//! container as a whole (`SetGroupMaterial`). Alt (held, tapped, or the
//! armed eyedropper) samples the face's material into the current paint
//! instead. The "default material" removes the paint.
//!
//! The current paint is CLASS-LEVEL state in Python (shared across tool
//! instances); here it is [`PaintState`], a field of the tool, with setters
//! for the toolbar's choices ([`PaintTool::set_color`], `set_texture`...).
//!
//! The texture maths the tool needs (`placement_of`, `flattened_texture`
//! with a normal, `continuous_maps`, `face_uv_axes`) is ported here:
//! `ingetrazo-model` keeps only the pure-data part of `core/texture.py`.
//!
//! Where it differs from Python (reported):
//!
//! * `_alt_down`, the viewport's own Alt state, is tracked from what the
//!   tool is handed: the Alt press (`on_key`), its release (`on_key_release`,
//!   which the fixtures' Python tool exposes as an observer that answers
//!   `False`) and the modifiers of every hover (the viewport re-reads them
//!   on each mouse move).
//! * Hovering a face INSIDE a group asks the viewport to highlight it
//!   through its placement; a placed face cannot be named by a `Sel`, so
//!   the tool passes only the placement (the fixtures keep such hovers out).
//! * Painting while a group is open for editing is not covered.
//! * The tray (its swatch, the recent row) and the main window's eyedropper
//!   release are UI; `recent` is kept as data.

use ingetrazo_formats::pyjson::Value;
use ingetrazo_math::{Hypot, M4, V3};
use ingetrazo_model::materials::{effective_attrs, has_own_material};
use ingetrazo_model::mesh::{FaceId, Mesh};
use ingetrazo_model::pyval;
use ingetrazo_model::scene::MeshId;
use ingetrazo_ops::command::Command;
use ingetrazo_ops::draw::CompoundCommand;
use ingetrazo_ops::edit_cmds::{SetFaceBack, SetFaceColor, SetFaceMaterialTag, SetFaceOpacity, SetFaceTexture};
use ingetrazo_ops::group::group_at;
use ingetrazo_ops::grouping::SetGroupMaterial;
use ingetrazo_snap::world::face_plane_world;

use crate::modify_base::{keys, round6, Env, FaceHit, FaceRef, ModifyTool, Mods, Probe, Sel, ToolCx};

/// The default cream the viewport paints unpainted faces with.
pub const DEFAULT_FACE_COLOR: [f64; 3] = [0.96, 0.95, 0.925];

/// How many materials the "Recent" row keeps.
pub const RECENT_MAX: usize = 6;

/// A named material (`core.materials.Material`): its name and its
/// `to_dict()`, which is what the scene registry keeps.
#[derive(Clone, Debug, PartialEq)]
pub struct PaintMaterial {
    pub name: String,
    pub dict: Value,
}

/// `PaintTool`'s class-level attributes.
#[derive(Clone, Debug, PartialEq)]
pub struct PaintState {
    /// `current_color`: RGB 0..1.
    pub color: Vec<Value>,
    /// `current_texture`: `{"path", "sw", "sh"...}`.
    pub texture: Option<Value>,
    /// `current_material`.
    pub material: Option<PaintMaterial>,
    /// `current_opacity`.
    pub opacity: Option<Value>,
    /// `sample_armed`: the next click samples, then disarms.
    pub sample_armed: bool,
    /// `current_texture_plane`: `(normal, offset)` of the plane the texture's
    /// own `uvw` belongs to.
    pub texture_plane: Option<(V3, f64)>,
    /// `current_is_default`: painting removes the paint.
    pub is_default: bool,
    /// `recent`: the last materials painted with, newest first.
    pub recent: Vec<Value>,
}

impl Default for PaintState {
    fn default() -> Self {
        PaintState {
            color: vec![Value::Float(0.80), Value::Float(0.45), Value::Float(0.30)],
            texture: None,
            material: None,
            opacity: None,
            sample_armed: false,
            texture_plane: None,
            is_default: false,
            recent: Vec::new(),
        }
    }
}

pub struct PaintTool {
    pub state: PaintState,
    /// `viewport._alt_down`.
    alt_down: bool,
}

impl Default for PaintTool {
    fn default() -> Self {
        PaintTool::new()
    }
}

// ---- small helpers -------------------------------------------------------------------------

fn s(t: &str) -> Value {
    Value::Str(t.to_string())
}

fn fl(x: f64) -> Value {
    Value::Float(x)
}

fn num(v: &Value) -> Option<f64> {
    match v {
        Value::Float(x) => Some(*x),
        Value::Int(_) | Value::Bool(_) => pyval::float(v).ok(),
        _ => None,
    }
}

/// `d.get(key)` where a `None` value reads as absent.
fn get_nn<'a>(d: &'a Value, key: &str) -> Option<&'a Value> {
    d.get(key).filter(|v| !matches!(v, Value::Null))
}

/// `round(x, 4)`.
fn round4(x: f64) -> f64 {
    if !x.is_finite() {
        return x;
    }
    format!("{x:.4}").parse().unwrap_or(x)
}

fn dict_without(tex: &Value, drop: &[&str]) -> Value {
    match tex {
        Value::Object(items) => Value::Object(items.iter().filter(|(k, _)| !drop.contains(&k.as_str())).cloned().collect()),
        other => other.clone(),
    }
}

// ---- recent materials ----------------------------------------------------------------------

/// `recent_key`: what makes two recent entries THE SAME material.
fn recent_key(entry: &Value) -> Value {
    if let Some(m) = entry.get("mat").filter(|v| pyval::truthy(v)) {
        return Value::Array(vec![s("m"), m.clone()]);
    }
    if let Some(tex) = entry.get("texture").filter(|v| pyval::truthy(v)) {
        let tint = match tex.get("tint") {
            Some(t) if pyval::truthy(t) => t.clone(),
            _ => Value::Null,
        };
        let g = |k: &str| tex.get(k).cloned().unwrap_or(Value::Null);
        return Value::Array(vec![
            s("t"),
            g("path"),
            g("sw"),
            g("sh"),
            tex.get("rot").cloned().unwrap_or(fl(0.0)),
            tint,
            entry.get("opacity").cloned().unwrap_or(Value::Null),
        ]);
    }
    let col: Vec<Value> = match entry.get("color") {
        Some(Value::Array(c)) => c.iter().map(|c| fl(round4(num(c).unwrap_or(0.0)))).collect(),
        _ => Vec::new(),
    };
    Value::Array(vec![s("c"), Value::Array(col), entry.get("opacity").cloned().unwrap_or(Value::Null)])
}

/// `push_recent`: `entry` moved (or added) to the front, without duplicates
/// and at most `cap` long.
pub fn push_recent(entries: &[Value], entry: Value, cap: usize) -> Vec<Value> {
    let key = recent_key(&entry);
    let mut out = vec![entry];
    out.extend(entries.iter().filter(|e| !pyval::eq(&recent_key(e), &key)).take(cap.saturating_sub(1)).cloned());
    out
}

// ---- texture maths (core/texture.py) ----------------------------------------------------------

const VERTICAL_TOLERANCE: f64 = 1e-3;

/// `projection_basis`.
fn projection_basis(normal: [f64; 3]) -> ([f64; 3], [f64; 3]) {
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
        return if nz > 0.0 { ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0]) } else { ([-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]) };
    }
    xx /= lx;
    xy /= lx;
    ([xx, xy, 0.0], [-nz * xy, nz * xx, nx * xy - ny * xx])
}

/// `projection_axes(normal, rot)`.
fn projection_axes(normal: V3, rot: f64) -> (V3, V3) {
    let (xr, yr) = projection_basis([normal.0 as f64, normal.1 as f64, normal.2 as f64]);
    let mut u = V3(xr[0] as f32, xr[1] as f32, xr[2] as f32);
    let mut v = V3(yr[0] as f32, yr[1] as f32, yr[2] as f32);
    if rot != 0.0 {
        let a = rot.to_radians();
        let (cos_a, sin_a) = (a.cos() as f32, a.sin() as f32);
        let nu = u.mul(cos_a).add(v.mul(sin_a));
        let nv = v.mul(cos_a).sub(u.mul(sin_a));
        u = nu;
        v = nv;
    }
    (u, v)
}

/// `face_uv_axes(tex, normal)`: `(gu, cu, gv, cv)`.
fn face_uv_axes(tex: &Value, normal: V3) -> (V3, f64, V3, f64) {
    if let Some(Value::Array(u)) = tex.get("uvw") {
        if !u.is_empty() {
            let n = |i: usize| u.get(i).and_then(num).unwrap_or(0.0);
            return (V3(n(0) as f32, n(1) as f32, n(2) as f32), n(3), V3(n(4) as f32, n(5) as f32, n(6) as f32), n(7));
        }
    }
    let rot = tex.get("rot").and_then(num).unwrap_or(0.0);
    let (u_axis, v_axis) = projection_axes(normal, if rot != 0.0 { rot } else { 0.0 });
    // `tex.get("sw", 1.0) or 1.0`
    let pick = |k: &str| match tex.get(k).and_then(num) {
        Some(x) if x != 0.0 => x,
        _ => 1.0,
    };
    (u_axis.div(pick("sw") as f32), 0.0, v_axis.div(pick("sh") as f32), 0.0)
}

/// `placement_of(tex, normal)`: `(sw, sh, rot)` a positioned texture shows on
/// the face of `normal`.
fn placement_of(tex: &Value, normal: V3, h: Hypot) -> Option<(f64, f64, f64)> {
    let uvw = match tex.get("uvw") {
        Some(Value::Array(u)) if u.len() == 8 => u,
        _ => return None,
    };
    if (normal.length_squared() as f64) < 1e-18 {
        return None;
    }
    let n = normal.normalized(h);
    let g = |i: usize| num(&uvw[i]).unwrap_or(0.0) as f32;
    let mut gu = V3(g(0), g(1), g(2));
    let mut gv = V3(g(4), g(5), g(6));
    gu = gu.sub(n.mul(gu.dot(n)));
    gv = gv.sub(n.mul(gv.dot(n)));
    let g11 = gu.dot(gu) as f64;
    let g12 = gu.dot(gv) as f64;
    let g22 = gv.dot(gv) as f64;
    let det = g11 * g22 - g12 * g12;
    if det.abs() < 1e-24 {
        return None;
    }
    let e_u = gu.mul(g22 as f32).sub(gv.mul(g12 as f32)).div(det as f32);
    let e_v = gv.mul(g11 as f32).sub(gu.mul(g12 as f32)).div(det as f32);
    let (sw, sh) = (e_u.length(h) as f64, e_v.length(h) as f64);
    if sw < 1e-9 || sh < 1e-9 {
        return None;
    }
    let (u_axis, v_axis) = projection_axes(n, 0.0);
    let rot = (e_u.dot(v_axis) as f64).atan2(e_u.dot(u_axis) as f64).to_degrees();
    Some((sw, sh, rot))
}

/// `flattened_texture(tex, normal)`.
fn flattened_texture(tex: &Value, normal: Option<V3>, h: Hypot) -> Value {
    let mut flat: Vec<(String, Value)> = match dict_without(tex, &["uvw"]) {
        Value::Object(o) => o,
        _ => Vec::new(),
    };
    let Some(normal) = normal else { return Value::Object(flat) };
    if !tex.get("uvw").is_some_and(pyval::truthy) {
        return Value::Object(flat);
    }
    let Some((sw, sh, rot)) = placement_of(tex, normal, h) else { return Value::Object(flat) };
    pyval::set(&mut flat, "sw", fl(sw));
    pyval::set(&mut flat, "sh", fl(sh));
    if rot.abs() > 1e-6 {
        pyval::set(&mut flat, "rot", fl(round6(rot)));
    } else {
        flat.retain(|(k, _)| k != "rot");
    }
    Value::Object(flat)
}

/// `_rotated_map`: the map carried across a hinge.
#[allow(clippy::too_many_arguments)]
fn rotated_map(gu: V3, cu: f64, gv: V3, cv: f64, axis_point: V3, axis_dir: V3, cos_t: f64, sin_t: f64) -> (V3, f64, V3, f64) {
    let rot = |v: V3| {
        v.mul(cos_t as f32)
            .add(axis_dir.cross(v).mul(sin_t as f32))
            .add(axis_dir.mul((axis_dir.dot(v) as f64 * (1.0 - cos_t)) as f32))
    };
    let (gu2, gv2) = (rot(gu), rot(gv));
    let cu2 = cu + gu.dot(axis_point) as f64 - gu2.dot(axis_point) as f64;
    let cv2 = cv + gv.dot(axis_point) as f64 - gv2.dot(axis_point) as f64;
    (gu2, cu2, gv2, cv2)
}

/// `continuous_maps(mesh, faces, seed, tex)`: per-face texture dicts that make
/// `tex` run continuously over a curved surface.
fn continuous_maps(mesh: &Mesh, faces: &[FaceId], seed: FaceId, tex: &Value, h: Hypot) -> Vec<(FaceId, Value)> {
    if faces.is_empty() {
        return Vec::new();
    }
    let seed_n = mesh.face_normal(seed, h);
    let mut out: Vec<(FaceId, (V3, f64, V3, f64))> = vec![(seed, face_uv_axes(tex, seed_n))];
    let mut stack = vec![seed];
    while let Some(f) = stack.pop() {
        let (g_u, c_u, g_v, c_v) = out.iter().find(|(k, _)| *k == f).map(|(_, m)| *m).expect("reached");
        let n1 = mesh.face_normal(f, h);
        if (n1.length_squared() as f64) < 1e-18 {
            continue;
        }
        let n1 = n1.normalized(h);
        let face = mesh.face(f);
        for lp in std::iter::once(&face.outer).chain(face.holes.iter()) {
            let n = lp.len();
            for i in 0..n {
                let Some(e) = mesh.find_edge(lp[i], lp[(i + 1) % n]) else { continue };
                let ed = mesh.edge(e);
                if !ed.soft {
                    continue;
                }
                for &g in &ed.faces.clone() {
                    if g == f || !faces.contains(&g) || out.iter().any(|(k, _)| *k == g) {
                        continue;
                    }
                    let n2 = mesh.face_normal(g, h);
                    let (ea, eb) = (mesh.position(ed.v0), mesh.position(ed.v1));
                    let d = eb.sub(ea);
                    if (n2.length_squared() as f64) < 1e-18 || (d.length_squared() as f64) < 1e-18 {
                        continue;
                    }
                    let n2 = n2.normalized(h);
                    let d = d.normalized(h);
                    let cos_t = n1.dot(n2) as f64;
                    let sin_t = d.dot(n1.cross(n2)) as f64;
                    if cos_t < -0.999 {
                        continue; // folded back: no hinge
                    }
                    out.push((g, rotated_map(g_u, c_u, g_v, c_v, ea, d, cos_t, sin_t)));
                    stack.push(g);
                }
            }
        }
    }
    let base = dict_without(tex, &["uvw", "rot"]);
    let mut result = Vec::new();
    for &f in faces {
        let Some((_, (gu, cu, gv, cv))) = out.iter().find(|(k, _)| *k == f) else { continue };
        let mut d = match &base {
            Value::Object(o) => o.clone(),
            _ => Vec::new(),
        };
        let uvw = vec![gu.0 as f64, gu.1 as f64, gu.2 as f64, *cu, gv.0 as f64, gv.1 as f64, gv.2 as f64, *cv];
        pyval::set(&mut d, "uvw", Value::Array(uvw.into_iter().map(fl).collect()));
        result.push((f, Value::Object(d)));
    }
    result
}

// ---- planes ------------------------------------------------------------------------------------

/// `_face_plane(face)`: `(unit normal, offset)`.
fn face_plane(mesh: &Mesh, f: FaceId, h: Hypot) -> (V3, f64) {
    let n = mesh.face_normal(f, h);
    let p = mesh.face(f).outer.first().map(|&v| mesh.position(v)).unwrap_or_default();
    (n, n.dot(p) as f64)
}

/// `_same_plane(face, plane)`.
fn same_plane(mesh: &Mesh, f: FaceId, plane: (V3, f64), h: Hypot) -> bool {
    let (n, d) = plane;
    let fnorm = mesh.face_normal(f, h);
    if ((fnorm.dot(n) as f64).abs() - 1.0).abs() > 1e-3 {
        return false;
    }
    let p = mesh.face(f).outer.first().map(|&v| mesh.position(v)).unwrap_or_default();
    (n.dot(p) as f64 - d).abs() <= 1e-4
}

type FaceKey = (MeshId, FaceId);

/// `_texture_commands(faces, tex, plane)`.
fn texture_commands(mesh: &Mesh, mid: MeshId, faces: &[FaceId], tex: &Value, plane: Option<(V3, f64)>, h: Hypot) -> Vec<Box<dyn Command>> {
    let keyed = |fs: &[FaceId]| -> Vec<FaceKey> { fs.iter().map(|&f| (mid, f)).collect() };
    let plane = match plane {
        Some(p) if tex.get("uvw").is_some_and(pyval::truthy) => p,
        _ => return vec![Box::new(SetFaceTexture::new(&keyed(faces), Some(tex.clone())))],
    };
    let same: Vec<FaceId> = faces.iter().copied().filter(|&f| same_plane(mesh, f, plane, h)).collect();
    let other: Vec<FaceId> = faces.iter().copied().filter(|&f| !same_plane(mesh, f, plane, h)).collect();
    let mut cmds: Vec<Box<dyn Command>> = Vec::new();
    if !same.is_empty() {
        cmds.push(Box::new(SetFaceTexture::new(&keyed(&same), Some(tex.clone()))));
    }
    if !other.is_empty() {
        cmds.push(Box::new(SetFaceTexture::new(&keyed(&other), Some(flattened_texture(tex, Some(plane.0), h)))));
    }
    cmds
}

/// `_surface_commands(mesh, faces, seed, tex, plane)`.
fn surface_commands(mesh: &Mesh, mid: MeshId, faces: &[FaceId], seed: FaceId, tex: &Value, plane: Option<(V3, f64)>, h: Hypot) -> Option<Vec<Box<dyn Command>>> {
    if faces.len() < 2 {
        return None;
    }
    let look = if tex.get("uvw").is_some_and(pyval::truthy) && plane.is_none_or(|p| !same_plane(mesh, seed, p, h)) {
        flattened_texture(tex, plane.map(|p| p.0), h)
    } else {
        tex.clone()
    };
    let maps = continuous_maps(mesh, faces, seed, &look, h);
    if maps.len() < 2 {
        return None;
    }
    let mut cmds: Vec<Box<dyn Command>> = Vec::new();
    for &f in faces {
        if let Some((_, m)) = maps.iter().find(|(k, _)| *k == f) {
            cmds.push(Box::new(SetFaceTexture::new(&[(mid, f)], Some(m.clone()))));
        }
    }
    let rest: Vec<FaceId> = faces.iter().copied().filter(|f| !maps.iter().any(|(k, _)| k == f)).collect();
    if !rest.is_empty() {
        cmds.extend(texture_commands(mesh, mid, &rest, tex, plane, h));
    }
    Some(cmds)
}

fn bits(x: f32) -> Value {
    Value::Str(format!("{:08x}", x.to_bits()))
}

// ---- the tool -------------------------------------------------------------------------------------

impl PaintTool {
    pub fn new() -> PaintTool {
        PaintTool { state: PaintState::default(), alt_down: false }
    }

    pub fn with_state(state: PaintState) -> PaintTool {
        PaintTool { state, alt_down: false }
    }

    /// The toolbar swatch.
    pub fn set_color(&mut self, color: Vec<Value>) {
        self.state.color = color;
    }

    /// The toolbar's texture (`None` paints a colour).
    pub fn set_texture(&mut self, texture: Option<Value>, plane: Option<(V3, f64)>) {
        self.state.texture = texture;
        self.state.texture_plane = plane;
    }

    pub fn set_material(&mut self, material: Option<PaintMaterial>) {
        self.state.material = material;
    }

    pub fn set_opacity(&mut self, opacity: Option<Value>) {
        self.state.opacity = opacity;
    }

    pub fn set_default(&mut self, is_default: bool) {
        self.state.is_default = is_default;
    }

    /// The toolbar's eyedropper button.
    pub fn arm_eyedropper(&mut self, armed: bool) {
        self.state.sample_armed = armed;
    }

    /// `_current_as_material`: the current paint as a container material.
    fn current_as_material(&self) -> Value {
        let mut paint: Vec<(String, Value)> = Vec::new();
        if let Some(tex) = &self.state.texture {
            paint.push(("texture".into(), dict_without(tex, &["uvw"])));
        } else {
            paint.push(("color".into(), Value::Array(self.state.color.clone())));
        }
        if let Some(o) = &self.state.opacity {
            paint.push(("opacity".into(), fl(pyval::float(o).unwrap_or(0.0))));
        }
        if let Some(m) = &self.state.material {
            paint.push(("mat".into(), s(&m.name)));
        }
        Value::Object(paint)
    }

    /// `_back_material_for(face)`.
    fn back_material_for(&self, mesh: &Mesh, f: FaceId, h: Hypot) -> Value {
        let mut back: Vec<(String, Value)> = Vec::new();
        if let Some(tex) = &self.state.texture {
            let mut tex = tex.clone();
            if tex.get("uvw").is_some_and(pyval::truthy) && self.state.texture_plane.is_none_or(|p| !same_plane(mesh, f, p, h)) {
                tex = flattened_texture(&tex, self.state.texture_plane.map(|p| p.0), h);
            }
            back.push(("texture".into(), tex));
        } else {
            back.push(("color".into(), Value::Array(self.state.color.clone())));
        }
        if let Some(o) = &self.state.opacity {
            back.push(("opacity".into(), fl(pyval::float(o).unwrap_or(0.0))));
        }
        if let Some(m) = &self.state.material {
            back.push(("mat".into(), s(&m.name)));
        }
        Value::Object(back)
    }

    /// `_remember_current`.
    fn remember_current(&mut self) {
        if self.state.is_default {
            return;
        }
        let entry = self.current_as_material();
        self.state.recent = push_recent(&self.state.recent, entry, RECENT_MAX);
    }

    fn material_tag(&self, faces: &[FaceKey]) -> Box<dyn Command> {
        let m = self.state.material.as_ref();
        Box::new(SetFaceMaterialTag::new(faces, m.map(|m| m.name.clone()), m.map(|m| m.dict.clone())))
    }

    /// The mesh a hit's face lives in, and the face.
    fn hit_face<'a>(env: &'a Env, hit: &FaceHit) -> Option<(&'a Mesh, FaceId)> {
        match &hit.face {
            FaceRef::Loose(f) => Some((env.cx.mesh(env.scene), *f)),
            FaceRef::Placed { placement, index } => {
                let g = group_at(&env.scene.groups, placement)?;
                let mesh = env.scene.mesh(g.mesh);
                Some((mesh, *mesh.faces.get(*index)?))
            }
        }
    }

    /// `clicked_back_side`: whether the cursor sees the BACK of the face.
    fn clicked_back_side(env: &Env, hit: &FaceHit, x: f64, y: f64) -> bool {
        let Some((origin, direction)) = env.vp.pixel_to_ray(x, y) else { return false };
        let _ = origin;
        let Some((mesh, f)) = PaintTool::hit_face(env, hit) else { return false };
        let xform: Option<&M4> = hit.owner.as_ref().and_then(|p| group_at(&env.scene.groups, p)).and_then(|g| g.xform.as_ref());
        let (_p, normal) = face_plane_world(mesh, f, xform, env.cx.hypot);
        normal.dot(direction) > 0.0
    }

    /// The eyedropper half of a click.
    fn sample(&mut self, env: &mut Env, hit: &FaceHit, back_side: bool) {
        let h = env.cx.hypot;
        let (attrs_v, back_attr, plane, material) = {
            let Some((mesh, f)) = PaintTool::hit_face(env, hit) else { return };
            let attrs = Value::Object(mesh.face(f).attrs.clone());
            let back_attr = mesh.face(f).attrs.iter().find(|(k, _)| k == "back").map(|(_, v)| v.clone());
            let owner_mat = hit.owner.as_ref().and_then(|p| group_at(&env.scene.groups, p)).and_then(|g| g.material.clone()).filter(pyval::truthy);
            (attrs, back_attr, face_plane(mesh, f, h), owner_mat)
        };
        // a default face inside a painted container wears the container's paint
        let mut src = effective_attrs(&attrs_v, &material.unwrap_or(Value::Null)).unwrap_or(attrs_v);
        if back_side {
            match back_attr {
                Some(b @ Value::Object(_)) => src = b,
                Some(Value::Bool(true)) => {}
                _ => src = Value::Object(Vec::new()),
            }
        }
        self.state.is_default = !has_own_material(&src).unwrap_or(false);
        match get_nn(&src, "texture") {
            Some(tex) => {
                self.state.texture = Some(tex.clone());
                self.state.texture_plane = if tex.get("uvw").is_some_and(pyval::truthy) { Some(plane) } else { None };
            }
            None => {
                self.state.texture = None;
                self.state.texture_plane = None;
                match get_nn(&src, "color") {
                    Some(Value::Array(c)) => self.state.color = c.clone(),
                    Some(_) => {}
                    None => self.state.color = DEFAULT_FACE_COLOR.iter().map(|&c| fl(c)).collect(),
                }
            }
        }
        self.state.opacity = get_nn(&src, "opacity").cloned();
        self.state.material = match src.get("mat").filter(|v| pyval::truthy(v)) {
            Some(Value::Str(name)) => env.scene.materials.iter().find(|(k, _)| k == name).map(|(k, d)| PaintMaterial { name: k.clone(), dict: d.clone() }),
            _ => None,
        };
        if self.state.sample_armed {
            self.state.sample_armed = false;
        }
        env.flash("Material sampled \u{2014} click a face to paint it");
    }
}

impl ModifyTool for PaintTool {
    fn on_activate(&mut self, _env: &mut Env) {}

    fn on_deactivate(&mut self, env: &mut Env) {
        env.vp.set_hover(None, None);
    }

    fn on_hover(&mut self, env: &mut Env, cx: &ToolCx) {
        // The viewport re-reads Alt from every mouse move.
        self.alt_down = cx.mods.alt();
        let (x, y) = cx.screen;
        match env.vp.pick_face_any(env.scene, env.cx, x, y) {
            None => env.vp.set_hover(None, None),
            Some(hit) => match hit.face {
                FaceRef::Loose(f) => env.vp.set_hover(Some(Sel::Face(f)), hit.placement),
                FaceRef::Placed { .. } => env.vp.set_hover(None, hit.placement),
            },
        }
    }

    fn on_key(&mut self, _env: &mut Env, key: i32, _mods: Mods) -> bool {
        if key == keys::ALT {
            self.alt_down = true;
        }
        false
    }

    fn on_key_release(&mut self, _env: &mut Env, key: i32) -> bool {
        if key == keys::ALT {
            self.alt_down = false;
        }
        false
    }

    fn on_click(&mut self, env: &mut Env, cx: &ToolCx) {
        let (x, y) = cx.screen;
        let h = env.cx.hypot;
        let Some(hit) = env.vp.pick_face_any(env.scene, env.cx, x, y) else { return };
        let back_side = PaintTool::clicked_back_side(env, &hit, x, y);

        // The pointer is the promise: whenever it shows the eyedropper the click samples.
        if cx.mods.alt() || self.alt_down || self.state.sample_armed {
            self.sample(env, &hit, back_side);
            return;
        }

        // A face inside a container clicked from OUTSIDE it paints the container.
        if hit.owner.is_some() {
            if let Some(path) = env.vp.pick_group(env.scene, env.cx, x, y) {
                let group_material = if self.state.is_default { None } else { Some(self.current_as_material()) };
                let cmds: Vec<Box<dyn Command>> = vec![self.material_tag(&[]), Box::new(SetGroupMaterial::new(&path, group_material))];
                env.execute(Box::new(CompoundCommand { commands: cmds }));
                let name = group_at(&env.scene.groups, &path).map(|g| g.name.clone()).unwrap_or_default();
                let name = if name.is_empty() { "Group".to_string() } else { name };
                env.flash(format!("{name} painted as a whole; a face that already had its own material keeps it"));
                self.remember_current();
                return;
            }
        }

        // Paint the clicked face - or, if it is part of the face selection, the whole selection.
        let FaceRef::Loose(face) = hit.face else { return };
        let mid = env.cx.mesh;
        let sel_faces = env.sel.faces();
        let faces: Vec<FaceId> = if sel_faces.contains(&face) { sel_faces } else { env.cx.mesh(env.scene).surface_of(face) };
        let keyed: Vec<FaceKey> = faces.iter().map(|&f| (mid, f)).collect();
        if self.state.is_default {
            // "No material": the clicked side goes back to the default.
            let cmd: Box<dyn Command> = if back_side {
                Box::new(SetFaceBack::same(&keyed, None))
            } else {
                Box::new(CompoundCommand {
                    commands: vec![
                        Box::new(SetFaceColor::new(&keyed, None)),
                        Box::new(SetFaceTexture::new(&keyed, None)),
                        Box::new(SetFaceOpacity::new(&keyed, None)),
                        Box::new(SetFaceMaterialTag::new(&keyed, None, None)),
                    ],
                })
            };
            env.execute(cmd);
            return;
        }
        if back_side {
            let backs: Vec<Option<Value>> = {
                let mesh = env.cx.mesh(env.scene);
                faces.iter().map(|&f| Some(self.back_material_for(mesh, f, h))).collect()
            };
            env.execute(Box::new(CompoundCommand { commands: vec![self.material_tag(&[]), Box::new(SetFaceBack::new(&keyed, backs))] }));
            self.remember_current();
            return;
        }
        let tag = self.material_tag(&keyed);
        let opacity: Box<dyn Command> = Box::new(SetFaceOpacity::new(&keyed, self.state.opacity.as_ref().and_then(|o| pyval::float(o).ok())));
        let cmd: Box<dyn Command> = if let Some(tex) = self.state.texture.clone() {
            let mesh = env.cx.mesh(env.scene);
            let mut cmds = surface_commands(mesh, mid, &faces, face, &tex, self.state.texture_plane, h)
                .unwrap_or_else(|| texture_commands(mesh, mid, &faces, &tex, self.state.texture_plane, h));
            cmds.push(opacity);
            cmds.push(tag);
            Box::new(CompoundCommand { commands: cmds })
        } else {
            // Painting a solid colour clears any texture on those faces, in one undoable step.
            Box::new(CompoundCommand {
                commands: vec![Box::new(SetFaceColor::new(&keyed, Some(self.state.color.clone()))), Box::new(SetFaceTexture::new(&keyed, None)), opacity, tag],
            })
        };
        env.execute(cmd);
        self.remember_current();
    }

    fn probe(&self, env: &Env) -> Probe {
        let mut p = Probe::default();
        p.uses_snap = false;
        let st = &self.state;
        let plane = st.texture_plane.map(|(n, d)| Value::Array(vec![Value::Array(vec![bits(n.0), bits(n.1), bits(n.2)]), fl(d)]));
        p.x = Some(Value::Object(vec![
            ("color".into(), Value::Array(st.color.clone())),
            ("texture".into(), st.texture.clone().unwrap_or(Value::Null)),
            ("material".into(), st.material.as_ref().map(|m| s(&m.name)).unwrap_or(Value::Null)),
            ("opacity".into(), st.opacity.clone().unwrap_or(Value::Null)),
            ("armed".into(), Value::Bool(st.sample_armed)),
            ("plane".into(), plane.unwrap_or(Value::Null)),
            ("default".into(), Value::Bool(st.is_default)),
            ("recent".into(), Value::Array(st.recent.clone())),
            ("alt_down".into(), Value::Bool(self.alt_down)),
            ("mats".into(), Value::Array(env.scene.materials.iter().map(|(k, _)| s(k)).collect())),
        ]));
        p
    }
}
