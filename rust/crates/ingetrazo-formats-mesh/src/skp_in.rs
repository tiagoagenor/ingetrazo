// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! SketchUp import: `formats/skp.py` and `formats/skp_openskp.py`.
//!
//! [`parse_skp`] reads the file with `ingetrazo-skp` and adapts the model to
//! a [`Payload`] (the dict Python builds): world-space face loops per group,
//! shared prototypes with a matrix per placement, the file's materials, tags,
//! scenes, dimensions and texts. [`apply_payload`] adds it to a scene. Like
//! Python, the heavy half touches no scene.
//!
//! No `run_stitch`, no `orient_outward`: the `.skp` import never runs them (it
//! keeps the file's polygons and its soft edges), so there is no hook.
//!
//! An Image entity or an "always face camera" component becomes a group of
//! its own, built through `fuse_coplanar_loops` (the engine's `fuse`) and the
//! smooth-edge pass of the DAE import, as in Python.
//!
//! Not ported, an [`SkpLoadError::NotPorted`]: a colourized texture in a
//! format other than PNG or JPEG (whether it needs re-tinting takes Qt to
//! decode it).
//!
//! A colourized texture is re-tinted by [`crate::colorize`], with Python's
//! arithmetic: the pixels of a PNG source are Python's, the file is not (Qt's
//! PNG encoder wrote it there), and a JPEG source is decoded by `zune-jpeg`,
//! which may differ from Qt's decoder by a level.
//!
//! The model's own face tags that are numbers (a face directly in the model
//! root keeps the tag id, not its name, in openskp) come out as Python has
//! them: an integer `layer` in the face's attrs.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::rc::Rc;

use ingetrazo_formats::pyjson::Value;
use ingetrazo_math::{Hypot, M4, V3};
use ingetrazo_model::dimension::Dimension;
use ingetrazo_model::layers::Layer as SceneLayer;
use ingetrazo_model::materials::{register, Material as SceneMaterial, Registry};
use ingetrazo_model::mesh::Mesh;
use ingetrazo_model::scene::{Group, MeshId, Scene};
use ingetrazo_model::textlabel::TextLabel;
use ingetrazo_skp::{Definition, Face, Model, OMap};

use crate::colorize::colorize_image;
use crate::glb::sha1_hex;
use crate::img;
use crate::uv::{fit_uv_affine, projection_basis};
use crate::world::np_affine;

const INCH: f64 = 0.0254;
const MAX_DEPTH: usize = 32;
/// `formats.dae._INST_MIN_POLYS` / `_INST_MIN_SAVED`.
const INST_MIN_POLYS: usize = 60;
const INST_MIN_SAVED: usize = 400;

pub type Attrs = Vec<(String, Value)>;

#[derive(Debug, Clone, PartialEq)]
pub enum SkpLoadError {
    /// `NeedsConverter`: the reader could not use the file.
    Unreadable(String),
    /// A path of the Python importer that is not ported (see the module doc).
    NotPorted(String),
}

impl std::fmt::Display for SkpLoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SkpLoadError::Unreadable(m) => write!(f, "unreadable: {m}"),
            SkpLoadError::NotPorted(m) => write!(f, "not ported: {m}"),
        }
    }
}

type R<T> = Result<T, SkpLoadError>;

fn unreadable<T>(m: impl Into<String>) -> R<T> {
    Err(SkpLoadError::Unreadable(m.into()))
}

fn s(x: &str) -> Value {
    Value::Str(x.to_string())
}

fn fl(x: f64) -> Value {
    Value::Float(x)
}

// ---- the payload ---------------------------------------------------------------------

/// One face as the payload carries it: corners in the group's space.
#[derive(Clone, Debug)]
pub struct PFace {
    pub outer: Vec<V3>,
    pub holes: Vec<Vec<V3>>,
    pub attrs: Option<Attrs>,
}

#[derive(Clone, Debug)]
pub struct PGroup {
    pub name: String,
    pub faces: Vec<PFace>,
    /// The file's own edge flags as segments; `None` for an image or a
    /// face-me component, which Python builds with `fuse_coplanar_loops`.
    pub soft_edges: Option<Vec<(V3, V3)>>,
    pub axes: Option<[f32; 16]>,
    pub layer: Option<String>,
    /// Turns toward the camera (a cutout image or a face-me component).
    pub billboard: bool,
}

#[derive(Clone, Debug)]
pub struct PKid {
    pub proto: usize,
    pub xform: M4,
    pub layer: Option<String>,
}

#[derive(Clone, Debug)]
pub struct PProto {
    pub name: String,
    pub faces: Vec<PFace>,
    pub soft_edges: Vec<(V3, V3)>,
    pub children: Vec<PKid>,
    pub instances: Vec<M4>,
    pub instance_layers: Option<Vec<Option<String>>>,
}

#[derive(Clone, Debug)]
pub struct PScene {
    pub name: String,
    pub eye: [f64; 3],
    pub target: [f64; 3],
    pub up: [f64; 3],
    pub fov: f64,
    pub parallel: bool,
    pub ortho_height: f64,
    pub hidden_layers: Vec<String>,
}

#[derive(Clone, Debug)]
pub struct PDim {
    pub a: [f64; 3],
    pub b: [f64; 3],
    pub offset: f64,
    pub normal: Option<[f64; 3]>,
}

#[derive(Clone, Debug)]
pub struct PText {
    pub anchor: [f64; 3],
    pub label: Option<[f64; 3]>,
    pub text: String,
    pub hidden: bool,
}

/// What `parse_skp` returns (Python's payload dict).
#[derive(Clone, Debug, Default)]
pub struct Payload {
    pub groups: Vec<PGroup>,
    pub protos: Vec<PProto>,
    /// The registry as `.igz` material dicts.
    pub materials: Vec<Value>,
    /// `(name, visible)`.
    pub layers: Vec<(String, bool)>,
    pub scenes: Vec<PScene>,
    pub dimensions: Vec<PDim>,
    pub texts: Vec<PText>,
    pub back_color: Option<[f64; 3]>,
    pub empty: bool,
}

// ---- format detection -------------------------------------------------------------------

/// `detect_format(path)`: `true` for a `.skp` by its first bytes.
pub fn detect_skp(head: &[u8]) -> bool {
    let head = &head[..head.len().min(64)];
    head.starts_with(b"PK\x03\x04") || head.windows(11).any(|w| w == b"S\0k\0e\0t\0c\0h")
}

// ---- materials ----------------------------------------------------------------------------

/// An entry of `_material_attrs`'s map: what a face painted with the
/// material wears. Shared by `Rc`, because Python tells entries apart by
/// identity (`back_src is front_src`).
#[derive(Debug)]
struct Entry {
    /// The attrs dict: `texture` or `color`, `opacity`, `mat`.
    base: Attrs,
    /// `(path, sw, sh)` of the texture, when it has one.
    tex: Option<(String, f64, f64)>,
}

fn sane_tile(v: f64) -> Option<f64> {
    (1e-6 < v && v < 1e5).then_some(v)
}

/// `_tile_size(width, height)` in metres.
fn tile_size(width: f64, height: f64) -> (f64, f64) {
    let (mut w, mut h) = (sane_tile(width), sane_tile(height));
    match (w, h) {
        (None, None) => {
            w = Some(1.0 / INCH);
            h = w;
        }
        (None, Some(hh)) => w = Some(hh),
        (Some(ww), None) => h = Some(ww),
        _ => {}
    }
    (w.unwrap() * INCH, h.unwrap() * INCH)
}

fn path_text(p: &Path) -> String {
    let t = p.to_string_lossy().into_owned();
    if cfg!(windows) {
        let t = t.strip_prefix("\\\\?\\").unwrap_or(&t).to_string();
        t.replace('/', "\\")
    } else {
        t
    }
}

/// `_texture_dir(skp_path)`: the app's cache folder for one `.skp`'s images.
fn texture_dir(skp_path: &Path, cache_root: &Path) -> PathBuf {
    let stamp = match std::fs::metadata(skp_path) {
        Ok(meta) => {
            let resolved = std::fs::canonicalize(skp_path).unwrap_or_else(|_| skp_path.to_path_buf());
            let mtime_ns = meta.modified().ok().and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok()).map(|d| d.as_nanos()).unwrap_or(0);
            format!("{}|{}|{}", path_text(&resolved), meta.len(), mtime_ns)
        }
        Err(_) => path_text(skp_path),
    };
    let key = &sha1_hex(stamp.as_bytes())[..10];
    let stem = skp_path.file_stem().map(|x| x.to_string_lossy().into_owned()).unwrap_or_default();
    let stem: String = stem.chars().filter(|&c| ingetrazo_model::pyalnum::is_alnum(c) || " ._-".contains(c)).collect();
    let stem = stem.trim().to_string();
    let d = cache_root.join("skp").join(format!("{}-{key}", if stem.is_empty() { "model" } else { &stem }));
    if std::fs::create_dir_all(&d).is_ok() {
        return d;
    }
    let tmp = std::env::temp_dir().join(format!("ingetrazo-skp-tex-{}-{}", std::process::id(), key));
    let _ = std::fs::create_dir_all(&tmp);
    tmp
}

/// `_material_attrs(model, skp_path)`: the entry of each material id, and the
/// registry payload.
fn material_attrs(model: &Model, skp_path: &Path, cache_root: &Path) -> R<(HashMap<i64, Rc<Entry>>, Vec<Value>)> {
    struct Build {
        mid: i64,
        name: String,
        color: Option<[f64; 3]>,
        tex: Option<(String, f64, f64)>,
        opacity: Option<f64>,
    }
    let mut builds: Vec<Build> = Vec::new();
    let mut tex_dir: Option<PathBuf> = None;
    for &(mid, mi) in &model.materials_by_id {
        let mat = &model.materials[mi];
        if let Some(tex) = &mat.texture {
            if let Some(data) = tex.data.as_ref().filter(|d| !d.is_empty()) {
                let dir = tex_dir.get_or_insert_with(|| texture_dir(skp_path, cache_root)).clone();
                let mut safe = tex.filename.replace('\\', "/").rsplit('/').next().unwrap_or("").to_string();
                let mut bytes: std::borrow::Cow<[u8]> = std::borrow::Cow::Borrowed(data.as_slice());
                if mat.colorized {
                    match needs_tint(data, &mat.color) {
                        Some(false) => {}
                        Some(true) => {
                            // the shared image, re-tinted toward the colour and
                            // kept under the material's own name
                            bytes = std::borrow::Cow::Owned(colorize_image(data, &mat.color, mat.colorize_type));
                            safe = format!("{mid}_{}", if safe.is_empty() { "material.png" } else { &safe });
                        }
                        None => return Err(SkpLoadError::NotPorted("a colourized texture in a format other than PNG or JPEG (whether it needs a tint takes Qt to decode it)".into())),
                    }
                }
                if safe.is_empty() {
                    safe = format!("material_{mid}.png");
                }
                let mut img_path = dir.join(&safe);
                let write = || -> std::io::Result<Option<PathBuf>> {
                    let mut p = img_path.clone();
                    if p.exists() && std::fs::metadata(&p)?.len() != bytes.len() as u64 {
                        p = dir.join(format!("{mid}_{}", p.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default()));
                    }
                    if !p.exists() || std::fs::metadata(&p)?.len() != bytes.len() as u64 {
                        std::fs::write(&p, bytes.as_ref())?;
                    }
                    Ok(Some(p))
                };
                match write() {
                    Ok(Some(p)) => img_path = p,
                    _ => img_path = PathBuf::new(),
                }
                if !img_path.as_os_str().is_empty() {
                    let (sw, sh) = tile_size(tex.width, tex.height);
                    let op = mat.transparency;
                    builds.push(Build { mid, name: mat.name.clone(), color: None, tex: Some((path_text(&img_path), sw, sh)), opacity: (op < 0.999).then_some(op) });
                    continue;
                }
            }
        }
        let c = mat.color;
        let op = mat.transparency;
        builds.push(Build {
            mid,
            name: mat.name.clone(),
            color: Some([c[0] as f64 / 255.0, c[1] as f64 / 255.0, c[2] as f64 / 255.0]),
            tex: None,
            opacity: (op < 0.999).then_some(op),
        });
    }
    // Registry: every named entry gets its identity (`register` dedups).
    let mut registry = Registry::new();
    let mut names: Vec<Option<String>> = Vec::new();
    for b in &builds {
        if b.name.trim().is_empty() {
            names.push(None);
            continue;
        }
        let color = b.color.map(|c| c.iter().map(|&x| fl(x)).collect::<Vec<_>>());
        let texture = b.tex.as_ref().map(|(p, sw, sh)| vec![("path".to_string(), s(p)), ("sw".to_string(), fl(*sw)), ("sh".to_string(), fl(*sh))]);
        let mat = SceneMaterial::new(b.name.trim(), color, texture, b.opacity.map(fl));
        let final_name = register(&mut registry, mat).map_err(|e| SkpLoadError::Unreadable(format!("{}: {}", e.kind, e.msg)))?;
        names.push(Some(match final_name {
            Value::Str(n) => n,
            _ => String::new(),
        }));
    }
    let mut map = HashMap::new();
    for (b, name) in builds.iter().zip(names) {
        let mut base: Attrs = Vec::new();
        if let Some((p, sw, sh)) = &b.tex {
            base.push(("texture".into(), Value::Object(vec![("path".into(), s(p)), ("sw".into(), fl(*sw)), ("sh".into(), fl(*sh))])));
        } else if let Some(c) = b.color {
            base.push(("color".into(), Value::Array(c.iter().map(|&x| fl(x)).collect())));
        }
        if let Some(op) = b.opacity {
            base.push(("opacity".into(), fl(op)));
        }
        if let Some(n) = name {
            base.push(("mat".into(), Value::Str(n)));
        }
        map.insert(b.mid, Rc::new(Entry { base, tex: b.tex.clone() }));
    }
    let mut dicts = Vec::new();
    for (_, m) in &registry.entries {
        dicts.push(m.to_dict().map_err(|e| SkpLoadError::Unreadable(format!("{}: {}", e.kind, e.msg)))?);
    }
    Ok((map, dicts))
}

/// `_needs_tint(data, declared_rgb)` for a PNG or JPEG; `None` when the image
/// is in another format. An image that will not decode needs no tint, as for
/// Qt (`isNull()`).
fn needs_tint(data: &[u8], declared: &[i64; 4]) -> Option<bool> {
    let (w, h, px) = match img::decode_rgba8(data) {
        img::Decoded::Image(w, h, px) => (w, h, px),
        img::Decoded::Null => return Some(false),
        img::Decoded::Unknown => return None,
    };
    if w == 0 || h == 0 {
        return Some(false);
    }
    let (mut sr, mut sg, mut sb, mut sa) = (0i64, 0i64, 0i64, 0i64);
    for y in 0..16 {
        let sy = img::nearest_n(y, 16, h).min(h - 1);
        for x in 0..16 {
            let sx = img::nearest_n(x, 16, w).min(w - 1);
            let [r, g, b, a] = px[sy * w + sx];
            sr += r as i64 * a as i64;
            sg += g as i64 * a as i64;
            sb += b as i64 * a as i64;
            sa += a as i64;
        }
    }
    if sa == 0 {
        return Some(false);
    }
    let avg = [sr as f64 / sa as f64, sg as f64 / sa as f64, sb as f64 / sa as f64];
    let d = (0..3).map(|i| (avg[i] - declared[i] as f64).abs()).fold(f64::MIN, f64::max);
    Some(d > 24.0)
}

// ---- geometry ---------------------------------------------------------------------------

/// `_matrix(m)`: an instance's 13 numbers as a matrix in metres.
fn instance_matrix(m: &[f64]) -> R<M4> {
    if m.len() < 12 {
        return unreadable("IndexError: list index out of range");
    }
    Ok(M4::from_rows([
        m[0] as f32,
        m[1] as f32,
        m[2] as f32,
        (m[9] * INCH) as f32,
        m[3] as f32,
        m[4] as f32,
        m[5] as f32,
        (m[10] * INCH) as f32,
        m[6] as f32,
        m[7] as f32,
        m[8] as f32,
        (m[11] * INCH) as f32,
        0.0,
        0.0,
        0.0,
        1.0,
    ]))
}

type Ring = Vec<[f64; 3]>;

/// `_ring_raw(defn, loop)`: the loop's corners in inches, from its
/// connectivity.
fn ring_raw(defn: &Definition, lp: &[(i64, i32)]) -> Option<Ring> {
    let n = lp.len();
    let mut edges = Vec::with_capacity(n);
    for &(eid, _) in lp {
        edges.push(defn.edges.get(&eid)?);
    }
    let mut pts = Vec::with_capacity(n);
    for i in 0..n {
        let edge = edges[i];
        let (a, b) = (edge.v1_id, edge.v2_id);
        let j = if i + 1 < n { i + 1 } else { 0 };
        let mut vid = None;
        if lp[j].0 != lp[i].0 {
            let nxt = edges[j];
            let (na, nb) = (nxt.v1_id, nxt.v2_id);
            let a_shared = a == na || a == nb;
            let b_shared = b == na || b == nb;
            if a_shared != b_shared {
                vid = Some(if b_shared { b } else { a });
            }
        }
        let vid = match vid {
            Some(v) => v,
            None => {
                if lp[i].1 != 0 {
                    a
                } else {
                    b
                }
            }
        };
        let v = defn.vertices.get(&vid)?;
        pts.push([v.x, v.y, v.z]);
    }
    Some(pts)
}

/// `_defn_geom(defn)`: every resolved ring corner in local metres and inches,
/// and per face its slices.
struct Geom {
    pos_m: Vec<[f64; 3]>,
    raw: Vec<[f64; 3]>,
    /// `(face id, outer start, outer end, hole slices)`.
    meta: Vec<(i64, usize, usize, Vec<(usize, usize)>)>,
}

fn build_geom(defn: &Definition) -> Geom {
    let mut raw: Vec<[f64; 3]> = Vec::new();
    let mut meta = Vec::new();
    for (&fid, face) in defn.faces.iter() {
        if face.loops.is_empty() {
            continue;
        }
        let Some(outer) = ring_raw(defn, &face.loops[0]) else { continue };
        if outer.len() < 3 {
            continue;
        }
        let s0 = raw.len();
        raw.extend(outer);
        let s1 = raw.len();
        let mut hs = Vec::new();
        for lp in &face.loops[1..] {
            if let Some(h) = ring_raw(defn, lp) {
                if h.len() >= 3 {
                    hs.push((raw.len(), raw.len() + h.len()));
                    raw.extend(h);
                }
            }
        }
        meta.push((fid, s0, s1, hs));
    }
    let pos_m = raw.iter().map(|p| [p[0] * INCH, p[1] * INCH, p[2] * INCH]).collect();
    Geom { pos_m, raw, meta }
}

/// `_world_points(pos_m, xform)`: metres through the matrix, float32 as the
/// vertices will hold them. NumPy's `pos @ R.T + t` accumulates each row as
/// a fused chain (see `world::affine_point`).
fn world_points(pos_m: &[[f64; 3]], xform: &M4) -> Vec<V3> {
    if xform.is_identity() {
        return pos_m.iter().map(|p| V3(p[0] as f32, p[1] as f32, p[2] as f32)).collect();
    }
    let (rot, trans) = np_affine(xform);
    pos_m
        .iter()
        .map(|p| {
            let (x, y, z) = (p[0], p[1], p[2]);
            let row = |i: usize| z.mul_add(rot[i][2], y.mul_add(rot[i][1], x * rot[i][0])) + trans[i];
            V3(row(0) as f32, row(1) as f32, row(2) as f32)
        })
        .collect()
}

/// The soft/smooth/hidden edges' ends of a definition, in local metres.
fn build_soft(defn: &Definition) -> Vec<[f64; 3]> {
    let mut out = Vec::new();
    for (_, e) in defn.edges.iter() {
        if !(e.soft || e.smooth || e.hidden) {
            continue;
        }
        let (Some(va), Some(vb)) = (defn.vertices.get(&e.v1_id), defn.vertices.get(&e.v2_id)) else { continue };
        out.push([va.x * INCH, va.y * INCH, va.z * INCH]);
        out.push([vb.x * INCH, vb.y * INCH, vb.z * INCH]);
    }
    out
}

// ---- textures per face -----------------------------------------------------------------------

/// `_positioned_uvs(face, raw_ring, tex, matrix, projected)`.
fn positioned_uvs(face: &Face, raw: &[[f64; 3]], sw: f64, sh: f64, matrix: &[f64; 9], projected: bool) -> Option<Vec<(f64, f64)>> {
    let tw = sw / INCH;
    let th = sh / INCH;
    if tw <= 0.0 || th <= 0.0 {
        return None;
    }
    let m = [[matrix[0], matrix[1], matrix[2]], [matrix[3], matrix[4], matrix[5]], [matrix[6], matrix[7], matrix[8]]];
    let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    if det.abs() < 1e-15 {
        return None;
    }
    let inv = [
        [(m[1][1] * m[2][2] - m[1][2] * m[2][1]) / det, (m[0][2] * m[2][1] - m[0][1] * m[2][2]) / det, (m[0][1] * m[1][2] - m[0][2] * m[1][1]) / det],
        [(m[1][2] * m[2][0] - m[1][0] * m[2][2]) / det, (m[0][0] * m[2][2] - m[0][2] * m[2][0]) / det, (m[0][2] * m[1][0] - m[0][0] * m[1][2]) / det],
        [(m[1][0] * m[2][1] - m[1][1] * m[2][0]) / det, (m[0][1] * m[2][0] - m[0][0] * m[2][1]) / det, (m[0][0] * m[1][1] - m[0][1] * m[1][0]) / det],
    ];
    let basis = if projected { None } else { Some(projection_basis(face.normal.unwrap_or([0.0, 0.0, 1.0]))) };
    let mut uvs = Vec::with_capacity(raw.len());
    for &[x, y, z] in raw {
        let (x2, y2) = match &basis {
            None => (x, y),
            Some((xr, yr)) => (x * xr[0] + y * xr[1] + z * xr[2], x * yr[0] + y * yr[1] + z * yr[2]),
        };
        let u = x2 * inv[0][0] + y2 * inv[1][0] + inv[2][0];
        let v = x2 * inv[0][1] + y2 * inv[1][1] + inv[2][1];
        let q = x2 * inv[0][2] + y2 * inv[1][2] + inv[2][2];
        if q.abs() < 1e-12 {
            return None;
        }
        uvs.push((u / q / tw, v / q / th));
    }
    Some(uvs)
}

/// `_bake_uvs(entry, uv_matrix, projected)`: the entry's attrs with the
/// texture's per-face `uvw` (and `planar` for the default mapping).
fn bake_uvs(entry: &Entry, face: &Face, raw: &[[f64; 3]], outer: &[V3], uv_matrix: Option<&[f64; 9]>, projected: bool) -> Attrs {
    let Some((_path, sw, sh)) = &entry.tex else { return entry.base.clone() };
    let uvs = match uv_matrix {
        Some(m) => positioned_uvs(face, raw, *sw, *sh, m, projected),
        None => {
            let tw = sw / INCH;
            let th = sh / INCH;
            if tw > 0.0 && th > 0.0 {
                let (xr, yr) = projection_basis(face.normal.unwrap_or([0.0, 0.0, 1.0]));
                Some(raw.iter().map(|&[x, y, z]| ((x * xr[0] + y * xr[1] + z * xr[2]) / tw, (x * yr[0] + y * yr[1] + z * yr[2]) / th)).collect())
            } else {
                None
            }
        }
    };
    if let Some(uvs) = uvs {
        if let Some(uvw) = fit_uv_affine(outer, &uvs) {
            let mut out = Vec::with_capacity(entry.base.len());
            for (k, v) in &entry.base {
                if k == "texture" {
                    let mut tex = match v {
                        Value::Object(items) => items.clone(),
                        _ => Vec::new(),
                    };
                    tex.push(("uvw".into(), Value::Array(uvw.iter().map(|&x| fl(x)).collect())));
                    if uv_matrix.is_none() {
                        tex.push(("planar".into(), Value::Bool(true)));
                    }
                    out.push((k.clone(), Value::Object(tex)));
                } else {
                    out.push((k.clone(), v.clone()));
                }
            }
            return out;
        }
    }
    entry.base.clone()
}

// ---- the adapter -------------------------------------------------------------------------------

/// Sharing record for a definition placed more than once.
type ProtoKey = (i64, Option<i64>);
type Placement = (M4, Option<String>);

struct Adapter<'m> {
    /// The definitions, then the root.
    all: Vec<&'m Definition>,
    by_id: HashMap<i64, usize>,
    attr_map: HashMap<i64, Rc<Entry>>,
    geoms: Vec<Option<Rc<Geom>>>,
    softs: Vec<Option<Rc<Vec<[f64; 3]>>>>,
    projected: HashSet<(usize, i64)>,
}

/// Collection state for a `_collect` walk: the optional lists it fills.
struct Collect<'a> {
    proto_ids: &'a HashSet<i64>,
    proto_uses: Option<&'a mut OMap<ProtoKey, Vec<Placement>>>,
    layer_uses: Option<&'a mut Vec<(usize, M4, Option<i64>, String)>>,
    /// `Some` when image entities and face-me components are pulled out into
    /// groups of their own.
    image_uses: Option<&'a mut Vec<ImageUse>>,
}

/// `(definition, placement, inherited paint, layer)` of an image entity or a
/// face-me component to turn into a billboard group.
type ImageUse = (usize, M4, Option<i64>, Option<String>);

impl<'m> Adapter<'m> {
    fn geom(&mut self, di: usize) -> Rc<Geom> {
        if let Some(g) = &self.geoms[di] {
            return g.clone();
        }
        let g = Rc::new(build_geom(self.all[di]));
        self.geoms[di] = Some(g.clone());
        g
    }

    fn soft(&mut self, di: usize) -> Rc<Vec<[f64; 3]>> {
        if let Some(g) = &self.softs[di] {
            return g.clone();
        }
        let g = Rc::new(build_soft(self.all[di]));
        self.softs[di] = Some(g.clone());
        g
    }

    fn soft_segments(&mut self, di: usize, xform: &M4, out: &mut Vec<(V3, V3)>) {
        let arr = self.soft(di);
        if arr.is_empty() {
            return;
        }
        let wl = world_points(&arr, xform);
        for pair in wl.chunks(2) {
            out.push((pair[0], pair[1]));
        }
    }

    /// `_face_entry(...)`.
    #[allow(clippy::too_many_arguments)]
    fn face_entry(&self, di: usize, face: &Face, wl: &[V3], raw_l: &[[f64; 3]], s0: usize, s1: usize, holes_sl: &[(usize, usize)], inherited: Option<i64>, layer: Option<&str>) -> PFace {
        let is_root = di == self.all.len() - 1;
        let mut raw: Vec<[f64; 3]> = raw_l[s0..s1].to_vec();
        let mut outer: Vec<V3> = wl[s0..s1].to_vec();
        let mut uv_mat = face.uv_transform;
        let mut flipped = false;
        let get = |id: Option<i64>| id.and_then(|i| self.attr_map.get(&i)).cloned();
        let mut attrs: Option<Rc<Entry>> = get(face.material_id);
        if attrs.is_none() {
            let battrs = get(face.back_material_id);
            if battrs.is_some() {
                attrs = battrs;
                uv_mat = face.uv_transform_back;
                raw.reverse();
                outer.reverse();
                flipped = true;
            } else if inherited.is_some() {
                attrs = get(inherited);
            }
        }
        let holes: Vec<Vec<V3>> = holes_sl
            .iter()
            .map(|&(ha, hb)| {
                let mut h = wl[ha..hb].to_vec();
                if flipped {
                    h.reverse();
                }
                h
            })
            .collect();
        let front_src = attrs.clone();
        let uv_proj = (if flipped { face.uv_projected_back } else { face.uv_projected }) || self.projected.contains(&(di, face.id));
        let mut out: Option<Attrs> = attrs.as_ref().map(|e| bake_uvs(e, face, &raw, &outer, uv_mat.as_ref(), uv_proj));
        if !flipped {
            let mut back_src = get(face.back_material_id);
            if back_src.is_none() && inherited.is_some() {
                back_src = get(inherited);
            }
            if let Some(back) = back_src {
                let mut base: Attrs = out.clone().unwrap_or_default();
                let same = front_src.as_ref().is_some_and(|f| Rc::ptr_eq(f, &back));
                let value = if same { Value::Bool(true) } else { Value::Object(bake_uvs(&back, face, &raw, &outer, face.uv_transform_back.as_ref(), face.uv_projected_back)) };
                match base.iter_mut().find(|(k, _)| k == "back") {
                    Some(slot) => slot.1 = value,
                    None => base.push(("back".into(), value)),
                }
                out = Some(base);
            }
        }
        // the face's own tag, else the nearest tagged instance's
        let lay: Option<Value> = if is_root {
            match face.layer_id {
                Some(n) if n != 0 => Some(Value::int(n)),
                _ => layer.filter(|l| !l.is_empty()).map(s),
            }
        } else if !face.layer.is_empty() {
            Some(s(&face.layer))
        } else {
            layer.filter(|l| !l.is_empty()).map(s)
        };
        if let Some(lay) = lay {
            let mut a = out.unwrap_or_default();
            match a.iter_mut().find(|(k, _)| k == "layer") {
                Some(slot) => slot.1 = lay,
                None => a.push(("layer".into(), lay)),
            }
            out = Some(a);
        }
        PFace { outer, holes, attrs: out }
    }

    /// `_collect(...)`: the faces of `di` (through `xform`) and, recursively,
    /// of every definition its instances place.
    #[allow(clippy::too_many_arguments)]
    fn collect(&mut self, di: usize, xform: &M4, out: &mut Vec<PFace>, depth: usize, stack: &mut Vec<usize>, cx: &mut Collect, inherited: Option<i64>, edges_out: &mut Option<&mut Vec<(V3, V3)>>, layer: Option<String>, group_layer: Option<&str>) -> R<()> {
        if depth > MAX_DEPTH || stack.contains(&di) {
            return Ok(());
        }
        stack.push(di);
        let g = self.geom(di);
        if !g.meta.is_empty() {
            let wl = world_points(&g.pos_m, xform);
            let def = self.all[di];
            for (fid, s0, s1, hs) in &g.meta {
                let face = def.faces.get(fid).unwrap();
                let entry = self.face_entry(di, face, &wl, &g.raw, *s0, *s1, hs, inherited, layer.as_deref());
                out.push(entry);
            }
        }
        if let Some(e) = edges_out.as_deref_mut() {
            self.soft_segments(di, xform, e);
        }
        let def = self.all[di];
        for ins in &def.instances {
            let Some(rid) = ins.ref_idx else { continue };
            let Some(&ci) = self.by_id.get(&rid) else { continue };
            let child = self.all[ci];
            let placed = xform.mul(&instance_matrix(&ins.matrix)?);
            let child_inherited = ins.material_id.filter(|&m| m != 0).or(inherited);
            let child_layer: Option<String> = if !ins.layer.is_empty() { Some(ins.layer.clone()) } else { layer.clone() };
            let out_layer: Option<String> = child_layer.clone().filter(|l| !l.is_empty()).or_else(|| group_layer.map(str::to_string));
            let cid = child.id;
            if child.is_image {
                match cx.image_uses.as_deref_mut() {
                    // an Image entity pulled out so it can become its own group
                    Some(uses) => uses.push((ci, placed, child_inherited, out_layer)),
                    // inside a face-me component being flattened: part of it
                    None => {
                        let quads = self.image_quad_faces(ci, &placed, child_inherited);
                        out.extend(quads);
                    }
                }
                continue;
            }
            if child.always_faces_camera {
                if let Some(uses) = cx.image_uses.as_deref_mut() {
                    uses.push((ci, placed, child_inherited, out_layer));
                    continue;
                }
            }
            if cx.proto_uses.is_some() && cx.proto_ids.contains(&cid) {
                let key = (cid, child_inherited);
                let uses = cx.proto_uses.as_deref_mut().unwrap();
                if !uses.contains_key(&key) {
                    uses.insert(key, Vec::new());
                }
                uses.get_mut(&key).unwrap().push((placed, out_layer));
                continue;
            }
            if cx.layer_uses.is_some() && !ins.layer.is_empty() {
                cx.layer_uses.as_deref_mut().unwrap().push((ci, placed, child_inherited, ins.layer.clone()));
                continue;
            }
            self.collect(ci, &placed, out, depth + 1, stack, cx, child_inherited, edges_out, child_layer, group_layer)?;
        }
        stack.pop();
        Ok(())
    }

    /// `_image_quad_faces(child, placed, attr_map, inherited)`: the faces of an
    /// Image entity's quad, each showing the whole picture once (per-vertex UV
    /// is the vertex's place on the local quad, as an exact world to UV map).
    fn image_quad_faces(&mut self, di: usize, placed: &M4, inherited: Option<i64>) -> Vec<PFace> {
        let def = self.all[di];
        let rings: Vec<(Option<Ring>, &Face)> = def.faces.values().map(|f| (f.loops.first().and_then(|lp| ring_raw(def, lp)), f)).collect();
        let pts: Vec<[f64; 3]> = rings.iter().filter_map(|(r, _)| r.as_ref()).flatten().copied().collect();
        if pts.is_empty() {
            return Vec::new();
        }
        let (mut x0, mut x1, mut y0, mut y1) = (pts[0][0], pts[0][0], pts[0][1], pts[0][1]);
        for p in &pts {
            // Python's min and max keep the first of equal values
            x0 = if p[0] < x0 { p[0] } else { x0 };
            x1 = if p[0] > x1 { p[0] } else { x1 };
            y0 = if p[1] < y0 { p[1] } else { y0 };
            y1 = if p[1] > y1 { p[1] } else { y1 };
        }
        let wspan = if x1 - x0 != 0.0 { x1 - x0 } else { 1.0 };
        let hspan = if y1 - y0 != 0.0 { y1 - y0 } else { 1.0 };
        let mut faces = Vec::new();
        for (raw, face) in rings {
            let Some(raw) = raw.filter(|r| r.len() >= 3) else { continue };
            let outer: Vec<V3> = raw.iter().map(|&[x, y, z]| placed.map(V3((x * INCH) as f32, (y * INCH) as f32, (z * INCH) as f32))).collect();
            let mid = face.material_id.or(inherited);
            let mut attrs: Option<Attrs> = mid.and_then(|m| self.attr_map.get(&m)).map(|e| e.base.clone());
            if let Some(a) = attrs.as_mut() {
                if a.iter().any(|(k, _)| k == "texture") {
                    let uvs: Vec<(f64, f64)> = raw.iter().map(|&[x, y, _z]| ((x - x0) / wspan, (y - y0) / hspan)).collect();
                    if let Some(uvw) = fit_uv_affine(&outer, &uvs) {
                        for (k, v) in a.iter_mut() {
                            if k == "texture" {
                                if let Value::Object(items) = v {
                                    items.push(("uvw".into(), Value::Array(uvw.iter().map(|&x| fl(x)).collect())));
                                }
                            }
                        }
                    }
                }
            }
            faces.push(PFace { outer, holes: Vec::new(), attrs });
        }
        faces
    }

    fn subtree_polys(&self, di: usize, memo: &mut HashMap<usize, usize>, stack: &mut Vec<usize>) -> usize {
        if stack.contains(&di) {
            return 0;
        }
        if let Some(&n) = memo.get(&di) {
            return n;
        }
        stack.push(di);
        let mut n = self.all[di].faces.len();
        for ins in &self.all[di].instances {
            if let Some(&ci) = ins.ref_idx.and_then(|r| self.by_id.get(&r)) {
                n += self.subtree_polys(ci, memo, stack);
            }
        }
        stack.pop();
        memo.insert(di, n);
        n
    }

    fn census(&self, di: usize, uses: &mut OMap<i64, usize>, depth: usize, stack: &mut Vec<usize>) {
        if depth > MAX_DEPTH || stack.contains(&di) {
            return;
        }
        stack.push(di);
        for ins in &self.all[di].instances {
            let Some(&ci) = ins.ref_idx.and_then(|r| self.by_id.get(&r)) else { continue };
            let id = self.all[ci].id;
            let n = uses.get(&id).copied().unwrap_or(0);
            uses.insert(id, n + 1);
            self.census(ci, uses, depth + 1, stack);
        }
        stack.pop();
    }

    fn subtree_has_faceme(&self, di: usize, stack: &mut Vec<usize>) -> bool {
        let d = self.all[di];
        if d.is_image || d.always_faces_camera {
            return true;
        }
        if stack.contains(&di) {
            return false;
        }
        stack.push(di);
        let mut found = false;
        for ins in &d.instances {
            if let Some(&ci) = ins.ref_idx.and_then(|r| self.by_id.get(&r)) {
                if self.subtree_has_faceme(ci, stack) {
                    found = true;
                    break;
                }
            }
        }
        stack.pop();
        found
    }

    fn subtree_has_tagged(&self, di: usize, stack: &mut Vec<usize>) -> bool {
        if stack.contains(&di) {
            return false;
        }
        stack.push(di);
        let mut found = false;
        for ins in &self.all[di].instances {
            if !ins.layer.is_empty() {
                found = true;
                break;
            }
            if let Some(&ci) = ins.ref_idx.and_then(|r| self.by_id.get(&r)) {
                if self.subtree_has_tagged(ci, stack) {
                    found = true;
                    break;
                }
            }
        }
        stack.pop();
        found
    }

    fn shareable(&self, di: usize) -> bool {
        !(self.subtree_has_faceme(di, &mut Vec::new()) || self.subtree_has_tagged(di, &mut Vec::new()))
    }

    fn subtree_uses(&self, di: usize, wanted: &HashSet<i64>, stack: &mut Vec<usize>) -> bool {
        if stack.contains(&di) {
            return false;
        }
        stack.push(di);
        let mut found = false;
        for ins in &self.all[di].instances {
            let Some(cid) = ins.ref_idx else { continue };
            if wanted.contains(&cid) {
                found = true;
                break;
            }
            if let Some(&ci) = self.by_id.get(&cid) {
                if self.subtree_uses(ci, wanted, stack) {
                    found = true;
                    break;
                }
            }
        }
        stack.pop();
        found
    }

    /// `_mark_projected_faces(defn, attr_map)`.
    fn mark_projected(&mut self, di: usize) -> R<()> {
        let def = self.all[di];
        let mut groups: OMap<(i64, Vec<u64>), Vec<i64>> = OMap::new();
        for (&fid, f) in def.faces.iter() {
            let (Some(mid), Some(uvm)) = (f.material_id, f.uv_transform) else { continue };
            let Some(entry) = self.attr_map.get(&mid) else { continue };
            if entry.tex.is_none() {
                continue;
            }
            let key = (mid, uvm.iter().map(|&x| round_key(x, 1)).collect::<Vec<_>>());
            match groups.get_mut(&key) {
                Some(v) => v.push(fid),
                None => groups.insert(key, vec![fid]),
            }
        }
        let keys: Vec<(i64, Vec<u64>)> = groups.keys().cloned().collect();
        for key in keys {
            let faces = groups.get(&key).unwrap().clone();
            if faces.len() < 8 {
                continue;
            }
            let (_p, sw, sh) = self.attr_map[&key.0].tex.clone().unwrap();
            let sample = &faces[..faces.len().min(80)];
            let disc = |projected: bool| -> R<Option<f64>> {
                let mut uv_at: OMap<[u64; 3], Vec<(u64, u64)>> = OMap::new();
                for fid in sample {
                    let f = def.faces.get(fid).unwrap();
                    let Some(lp0) = f.loops.first() else { return unreadable("IndexError: list index out of range") };
                    let Some(raw) = ring_raw(def, lp0) else { return unreadable("TypeError: 'NoneType' object is not iterable") };
                    let Some(uvs) = positioned_uvs(f, &raw, sw, sh, f.uv_transform.as_ref().unwrap(), projected) else { return Ok(None) };
                    for (p, uv) in raw.iter().zip(uvs) {
                        let k = [round_key(p[0], 1), round_key(p[1], 1), round_key(p[2], 1)];
                        let v = (round_key(uv.0, 2), round_key(uv.1, 2));
                        match uv_at.get_mut(&k) {
                            Some(set) => {
                                if !set.contains(&v) {
                                    set.push(v)
                                }
                            }
                            None => uv_at.insert(k, vec![v]),
                        }
                    }
                }
                if uv_at.is_empty() {
                    return Ok(None);
                }
                let bad = uv_at.values().filter(|v| v.len() > 1).count();
                Ok(Some(bad as f64 / uv_at.len() as f64))
            };
            let local = disc(false)?;
            let plan = disc(true)?;
            if let (Some(l), Some(p)) = (local, plan) {
                if l > 0.3 && p < 0.1 {
                    for fid in faces {
                        self.projected.insert((di, fid));
                    }
                }
            }
        }
        Ok(())
    }
}

/// `round(x, digits)` as a hashable key (equal values give equal keys).
fn round_key(x: f64, digits: usize) -> u64 {
    if !x.is_finite() {
        return x.to_bits();
    }
    let r: f64 = format!("{x:.digits$}").parse().unwrap_or(x);
    (r + 0.0).to_bits()
}

/// `_merge_equal_protos(protos)`.
fn merge_equal_protos(mut protos: Vec<PProto>) -> Vec<PProto> {
    if protos.len() < 2 {
        return protos;
    }
    for _round in 0..8 {
        let mut buckets: OMap<(String, usize, usize, usize), Vec<usize>> = OMap::new();
        for (i, e) in protos.iter().enumerate() {
            let key = (e.name.clone(), e.faces.len(), e.soft_edges.len(), e.children.len());
            match buckets.get_mut(&key) {
                Some(v) => v.push(i),
                None => buckets.insert(key, vec![i]),
            }
        }
        let mut remap: HashMap<usize, usize> = HashMap::new();
        for bucket in buckets.values() {
            if bucket.len() < 2 {
                continue;
            }
            let mut seen: HashMap<String, usize> = HashMap::new();
            for &i in bucket {
                let e = &protos[i];
                let sig = proto_signature(e);
                let first = *seen.entry(sig).or_insert(i);
                if first != i {
                    remap.insert(i, first);
                }
            }
        }
        if remap.is_empty() {
            break;
        }
        let mut folded: Vec<(usize, usize)> = remap.iter().map(|(&a, &b)| (a, b)).collect();
        folded.sort();
        for (i, keep) in folded {
            if protos[i].instances.is_empty() {
                continue;
            }
            let src_instances = protos[i].instances.clone();
            let src_layers = protos[i].instance_layers.clone();
            let dst = &mut protos[keep];
            if src_layers.is_some() || dst.instance_layers.is_some() {
                let mut layers = dst.instance_layers.take().unwrap_or_else(|| vec![None; dst.instances.len()]);
                layers.extend(src_layers.unwrap_or_else(|| vec![None; src_instances.len()]));
                dst.instance_layers = Some(layers);
            }
            dst.instances.extend(src_instances);
        }
        let keep_ix: Vec<usize> = (0..protos.len()).filter(|i| !remap.contains_key(i)).collect();
        let new_of: HashMap<usize, usize> = keep_ix.iter().enumerate().map(|(n, &o)| (o, n)).collect();
        let mut out = Vec::with_capacity(keep_ix.len());
        for &i in &keep_ix {
            let mut e = protos[i].clone();
            for k in &mut e.children {
                k.proto = new_of[remap.get(&k.proto).unwrap_or(&k.proto)];
            }
            out.push(e);
        }
        protos = out;
    }
    protos
}

/// A text that is equal for equal values (floats by their bits, objects in
/// key order), standing for `repr(attrs)`.
fn value_text(v: &Value, out: &mut String) {
    use std::fmt::Write;
    match v {
        Value::Null => out.push('n'),
        Value::Bool(b) => out.push(if *b { 't' } else { 'f' }),
        Value::Int(i) => {
            let _ = write!(out, "i{i}");
        }
        Value::Float(x) => {
            let _ = write!(out, "d{:016x}", x.to_bits());
        }
        Value::Str(t) => {
            let _ = write!(out, "s{}:{t}", t.len());
        }
        Value::Array(a) => {
            out.push('[');
            for x in a {
                value_text(x, out);
                out.push(',');
            }
            out.push(']');
        }
        Value::Object(o) => {
            out.push('{');
            for (k, x) in o {
                let _ = write!(out, "{}:{k}=", k.len());
                value_text(x, out);
                out.push(',');
            }
            out.push('}');
        }
    }
}

fn proto_signature(e: &PProto) -> String {
    use std::fmt::Write;
    let mut out = String::new();
    let p = |out: &mut String, v: V3| {
        let _ = write!(out, "{:08x},{:08x},{:08x};", v.0.to_bits(), v.1.to_bits(), v.2.to_bits());
    };
    for f in &e.faces {
        out.push('F');
        for &v in &f.outer {
            p(&mut out, v);
        }
        for h in &f.holes {
            out.push('H');
            for &v in h {
                p(&mut out, v);
            }
        }
        out.push('A');
        match &f.attrs {
            Some(a) => value_text(&Value::Object(a.clone()), &mut out),
            None => out.push('N'),
        }
    }
    out.push('S');
    for (a, b) in &e.soft_edges {
        p(&mut out, *a);
        p(&mut out, *b);
    }
    out.push('C');
    for k in &e.children {
        let _ = write!(out, "{}:{:?}:", k.proto, k.layer);
        for x in k.xform.data() {
            let _ = write!(out, "{:08x},", x.to_bits());
        }
    }
    out
}

/// `file_layer_records(model)`.
fn file_layer_records(model: &Model) -> Vec<(String, bool)> {
    model.layers.iter().filter(|l| !(l.name.is_empty() || l.name == "Layer0" || l.name == "Untagged")).map(|l| (l.name.clone(), !l.hidden)).collect()
}

/// `_is_empty_model(model, skp_path, legacy_era)`: a 2021+ file with nothing
/// in it, told by its thumbnail being one flat colour.
fn is_empty_model(model: &Model, legacy_era: bool) -> bool {
    if legacy_era || !model.definitions.is_empty() {
        return false;
    }
    let r = &model.root;
    if !r.faces.is_empty() || !r.edges.is_empty() || !r.instances.is_empty() || !r.texts.is_empty() || !r.dimensions.is_empty() || !r.construction_lines.is_empty() || !r.construction_points.is_empty() {
        return false;
    }
    let Some(png) = &model.thumbnail else { return false };
    let Some((_w, _h, px)) = img::rgba8(png) else { return false };
    px.first().is_some_and(|first| px.iter().all(|p| p == first))
}

fn legacy_era_of(version: &str) -> bool {
    let v = version.trim_start_matches('{');
    let digits: String = v.chars().filter(|c| c.is_ascii_digit() || *c == '.').collect();
    let first = digits.split('.').next().unwrap_or("");
    !first.is_empty() && first.chars().all(|c| c.is_ascii_digit()) && first.parse::<i64>().map(|n| n < 21).unwrap_or(false)
}

/// `_adapt(model, name, skp_path)`.
pub fn adapt(model: &Model, name: &str, skp_path: &Path, cache_root: &Path) -> R<Payload> {
    let t0 = std::time::Instant::now();
    let trace = |what: &str| {
        if std::env::var_os("SKP_TRACE").is_some() {
            eprintln!("skp adapt: {what} at {:?}", t0.elapsed());
        }
    };
    let (attr_map, materials) = material_attrs(model, skp_path, cache_root)?;
    trace("materials");
    let mut all: Vec<&Definition> = model.definitions.iter().collect();
    all.push(&model.root);
    let n_defs = model.definitions.len();
    let mut by_id: HashMap<i64, usize> = HashMap::new();
    for (i, d) in model.definitions.iter().enumerate() {
        by_id.insert(d.id, i);
    }
    let mut ad = Adapter { all, by_id, attr_map, geoms: vec![None; n_defs + 1], softs: vec![None; n_defs + 1], projected: HashSet::new() };
    let root_ix = n_defs;
    let legacy_era = legacy_era_of(&model.version);
    if !legacy_era {
        ad.mark_projected(root_ix)?;
        for i in 0..n_defs {
            ad.mark_projected(i)?;
        }
    }

    // Shared-prototype census over the tree reachable from the root.
    let mut uses: OMap<i64, usize> = OMap::new();
    ad.census(root_ix, &mut uses, 0, &mut Vec::new());
    let mut memo = HashMap::new();
    let mut proto_ids: HashSet<i64> = HashSet::new();
    for (&did, &cnt) in uses.iter() {
        let Some(&di) = ad.by_id.get(&did) else { continue };
        if cnt < 2 || !ad.shareable(di) {
            continue;
        }
        let polys = ad.subtree_polys(di, &mut memo, &mut Vec::new());
        if polys >= INST_MIN_POLYS && polys * (cnt - 1) >= INST_MIN_SAVED {
            proto_ids.insert(did);
        }
    }
    // A definition placed once at the top level keeps its placement too.
    for ins in &ad.all[root_ix].instances {
        let Some(&ci) = ins.ref_idx.and_then(|r| ad.by_id.get(&r)) else { continue };
        let child = ad.all[ci];
        if uses.get(&child.id) != Some(&1) {
            continue;
        }
        if child.is_image || child.always_faces_camera {
            continue;
        }
        if !ad.shareable(ci) {
            continue;
        }
        if ad.subtree_uses(ci, &proto_ids, &mut Vec::new()) {
            continue;
        }
        proto_ids.insert(child.id);
    }

    trace("census and prototype choice");
    let mut groups: Vec<PGroup> = Vec::new();
    let mut proto_uses: OMap<ProtoKey, Vec<Placement>> = OMap::new();
    let mut layer_uses: Vec<(usize, M4, Option<i64>, String)> = Vec::new();
    let mut image_uses: Vec<ImageUse> = Vec::new();
    {
        // The root's own loose faces (no instance recursion).
        let ident = M4::identity();
        let mut loose: Vec<PFace> = Vec::new();
        let g = ad.geom(root_ix);
        if !g.meta.is_empty() {
            let wl = world_points(&g.pos_m, &ident);
            let root = ad.all[root_ix];
            for (fid, s0, s1, hs) in &g.meta {
                let face = root.faces.get(fid).unwrap();
                loose.push(ad.face_entry(root_ix, face, &wl, &g.raw, *s0, *s1, hs, None, None));
            }
        }
        if !loose.is_empty() {
            let mut loose_edges = Vec::new();
            ad.soft_segments(root_ix, &ident, &mut loose_edges);
            groups.push(PGroup { name: name.to_string(), faces: loose, soft_edges: Some(loose_edges), axes: None, layer: None, billboard: false });
        }
        // Each top-level instance: its own group, or a shared-prototype use.
        let root_def: &Definition = ad.all[root_ix];
        for ins in &root_def.instances {
            let Some(&ci) = ins.ref_idx.and_then(|r| ad.by_id.get(&r)) else { continue };
            let child = ad.all[ci];
            let placed = instance_matrix(&ins.matrix)?;
            // a top-level paint is taken as the file says (0 included)
            let inh = ins.material_id;
            let lay: Option<String> = (!ins.layer.is_empty()).then(|| ins.layer.clone());
            if child.is_image || child.always_faces_camera {
                image_uses.push((ci, placed, inh, lay));
                continue;
            }
            if proto_ids.contains(&child.id) {
                let key = (child.id, inh);
                if !proto_uses.contains_key(&key) {
                    proto_uses.insert(key, Vec::new());
                }
                proto_uses.get_mut(&key).unwrap().push((placed, lay));
                continue;
            }
            let mut sub = Vec::new();
            let mut sub_edges = Vec::new();
            {
                let mut cx = Collect { proto_ids: &proto_ids, proto_uses: Some(&mut proto_uses), layer_uses: Some(&mut layer_uses), image_uses: Some(&mut image_uses) };
                let mut eo = Some(&mut sub_edges);
                ad.collect(ci, &placed, &mut sub, 0, &mut Vec::new(), &mut cx, inh, &mut eo, None, lay.as_deref())?;
            }
            if !sub.is_empty() {
                groups.push(PGroup {
                    name: if child.name.is_empty() { name.to_string() } else { child.name.clone() },
                    faces: sub,
                    soft_edges: Some(sub_edges),
                    axes: Some(placed.data()),
                    layer: lay,
                    billboard: false,
                });
            }
        }
    }

    trace("top-level groups");
    // Tagged nested instances: their own groups carrying the layer. The list
    // grows while it is walked.
    let mut i = 0;
    while i < layer_uses.len() {
        let (ci, placed, inh, lay) = layer_uses[i].clone();
        i += 1;
        let mut sub = Vec::new();
        let mut sub_edges = Vec::new();
        {
            let mut cx = Collect { proto_ids: &proto_ids, proto_uses: Some(&mut proto_uses), layer_uses: Some(&mut layer_uses), image_uses: Some(&mut image_uses) };
            let mut eo = Some(&mut sub_edges);
            ad.collect(ci, &placed, &mut sub, 0, &mut Vec::new(), &mut cx, inh, &mut eo, None, Some(&lay))?;
        }
        if !sub.is_empty() {
            let child = ad.all[ci];
            groups.push(PGroup { name: if child.name.is_empty() { name.to_string() } else { child.name.clone() }, faces: sub, soft_edges: Some(sub_edges), axes: Some(placed.data()), layer: Some(lay), billboard: false });
        }
    }

    // Image entities and face-me components: their own groups. A cutout image
    // (real transparency) and every face-me component turn toward the camera;
    // an opaque photo stays a panel. An image shows its whole picture once.
    for (ci, placed, inh, lay) in std::mem::take(&mut image_uses) {
        let child = ad.all[ci];
        let (faces, billboard) = if child.is_image {
            let faces = ad.image_quad_faces(ci, &placed, inh);
            let tex = faces.iter().find_map(|f| match f.attrs.as_ref().and_then(|a| a.iter().find(|(k, _)| k == "texture")) {
                Some((_, Value::Object(t))) => t.iter().find(|(k, _)| k == "path").and_then(|(_, p)| if let Value::Str(p) = p { Some(p.clone()) } else { None }),
                _ => None,
            });
            let cutout = tex.is_some_and(|p| std::fs::read(&p).is_ok_and(|d| img::image_has_cutout(&d)));
            (faces, cutout)
        } else {
            let mut faces = Vec::new();
            let mut cx = Collect { proto_ids: &HashSet::new(), proto_uses: None, layer_uses: None, image_uses: None };
            ad.collect(ci, &placed, &mut faces, 0, &mut Vec::new(), &mut cx, inh, &mut None, None, None)?;
            (faces, true)
        };
        if faces.is_empty() {
            continue;
        }
        groups.push(PGroup { name: if child.name.is_empty() { name.to_string() } else { child.name.clone() }, faces, soft_edges: None, axes: None, layer: lay, billboard });
    }

    // Shared prototypes, built once in local coordinates per inherited paint.
    let mut protos: Vec<Option<PProto>> = Vec::new();
    let mut index_of: HashMap<ProtoKey, usize> = HashMap::new();
    fn build_proto(ad: &mut Adapter, proto_ids: &HashSet<i64>, protos: &mut Vec<Option<PProto>>, index_of: &mut HashMap<ProtoKey, usize>, did: i64, inh: Option<i64>, name: &str) -> R<Option<usize>> {
        let key = (did, inh);
        if let Some(&i) = index_of.get(&key) {
            return Ok(Some(i));
        }
        let Some(&di) = ad.by_id.get(&did) else { return Ok(None) };
        let idx = protos.len();
        index_of.insert(key, idx);
        protos.push(None);
        let mut local = Vec::new();
        let mut local_edges = Vec::new();
        let mut child_uses: OMap<ProtoKey, Vec<Placement>> = OMap::new();
        {
            let mut cx = Collect { proto_ids, proto_uses: Some(&mut child_uses), layer_uses: None, image_uses: None };
            let mut eo = Some(&mut local_edges);
            ad.collect(di, &M4::identity(), &mut local, 0, &mut Vec::new(), &mut cx, inh, &mut eo, None, None)?;
        }
        let mut children = Vec::new();
        let entries: Vec<(ProtoKey, Vec<Placement>)> = child_uses.iter().map(|(k, v)| (*k, v.clone())).collect();
        for ((cid, cinh), places) in entries {
            let Some(ci) = build_proto(ad, proto_ids, protos, index_of, cid, cinh, name)? else { continue };
            for (xf, ly) in places {
                children.push(PKid { proto: ci, xform: xf, layer: ly.filter(|l| !l.is_empty()) });
            }
        }
        let d = ad.all[di];
        protos[idx] = Some(PProto { name: if d.name.is_empty() { name.to_string() } else { d.name.clone() }, faces: local, soft_edges: local_edges, children, instances: Vec::new(), instance_layers: None });
        Ok(Some(idx))
    }
    let entries: Vec<(ProtoKey, Vec<Placement>)> = proto_uses.iter().map(|(k, v)| (*k, v.clone())).collect();
    for ((did, inh), places) in entries {
        if places.is_empty() {
            continue;
        }
        let Some(idx) = build_proto(&mut ad, &proto_ids, &mut protos, &mut index_of, did, inh, name)? else { continue };
        let p = protos[idx].as_mut().unwrap();
        p.instances = places.iter().map(|(xf, _)| *xf).collect();
        if places.iter().any(|(_, ly)| ly.as_ref().is_some_and(|l| !l.is_empty())) {
            p.instance_layers = Some(places.iter().map(|(_, ly)| ly.clone().filter(|l| !l.is_empty())).collect());
        }
    }
    let protos: Vec<PProto> = protos.into_iter().map(|p| p.expect("every reserved prototype slot is filled")).collect();
    trace("prototypes built");
    let protos = merge_equal_protos(protos);
    trace("prototypes merged");

    let mut payload = Payload { groups, protos, materials, ..Default::default() };
    if payload.groups.is_empty() && !payload.protos.iter().any(|e| !e.faces.is_empty() || !e.children.is_empty()) {
        if !is_empty_model(model, legacy_era) {
            return unreadable("the file yields no geometry");
        }
        payload.empty = true;
    }
    payload.layers = file_layer_records(model);
    for (n, pg) in model.pages.iter().enumerate() {
        let (Some(eye), Some(target)) = (pg.eye, pg.target) else { continue };
        let _ = n;
        let k = payload.scenes.len() + 1;
        payload.scenes.push(PScene {
            name: if pg.name.is_empty() { format!("Scene {k}") } else { pg.name.clone() },
            eye: [eye[0] * INCH, eye[1] * INCH, eye[2] * INCH],
            target: [target[0] * INCH, target[1] * INCH, target[2] * INCH],
            up: pg.up.unwrap_or([0.0, 0.0, 1.0]),
            fov: pg.fov,
            parallel: pg.parallel,
            ortho_height: pg.ortho_height * INCH,
            hidden_layers: pg.hidden_layers.clone(),
        });
    }
    for dm in &model.dimensions {
        let (Some(a), Some(b)) = (dm.a, dm.b) else { continue };
        payload.dimensions.push(PDim { a: [a[0] * INCH, a[1] * INCH, a[2] * INCH], b: [b[0] * INCH, b[1] * INCH, b[2] * INCH], offset: dm.offset * INCH, normal: dm.normal });
    }
    for tx in &model.root.texts {
        let body = tx.text.trim();
        let Some(pt) = tx.point else { continue };
        if body.is_empty() {
            continue;
        }
        payload.texts.push(PText {
            anchor: [pt[0] * INCH, pt[1] * INCH, pt[2] * INCH],
            label: tx.label_point.map(|l| [l[0] * INCH, l[1] * INCH, l[2] * INCH]),
            text: body.to_string(),
            hidden: tx.hidden,
        });
    }
    for st in &model.styles {
        if let Some(b) = st.back_color {
            payload.back_color = Some([b[0] as f64 / 255.0, b[1] as f64 / 255.0, b[2] as f64 / 255.0]);
            break;
        }
    }
    Ok(payload)
}

/// `skp_openskp.parse(path)` after `skp.parse_skp`'s checks: the payload, or
/// the reason the file counts as unreadable.
pub fn parse_skp(path: &Path, cache_root: &Path) -> R<Payload> {
    let data = std::fs::read(path).map_err(|e| SkpLoadError::Unreadable(e.to_string()))?;
    if !detect_skp(&data) {
        return unreadable("No pure SKP backend (format=unknown)");
    }
    let model = ingetrazo_skp::parse_bytes(&data).map_err(|e| SkpLoadError::Unreadable(e.to_string()))?;
    let stem = path.file_stem().map(|x| x.to_string_lossy().into_owned()).unwrap_or_default();
    adapt(&model, &stem, path, cache_root)
}

// ---- applying it ------------------------------------------------------------------------

fn finite(p: &V3) -> bool {
    p.0.is_finite() && p.1.is_finite() && p.2.is_finite()
}

/// `_build_mesh(faces, soft_edges)` with the file's own edge flags.
fn build_mesh(faces: &[PFace], soft_edges: &[(V3, V3)], h: Hypot) -> Mesh {
    let mut mesh = Mesh::new();
    mesh.hypot = h;
    let mut flat: Vec<V3> = Vec::new();
    let mut ring_sizes: Vec<usize> = Vec::new();
    let mut ring_counts: Vec<usize> = Vec::new();
    let mut attrs_list: Vec<Attrs> = Vec::new();
    for f in faces {
        let holes: Vec<&Vec<V3>> = if f.holes.len() > 1 {
            ingetrazo_model::topology::maximal_holes(&f.holes, h).into_iter().map(|i| &f.holes[i]).collect()
        } else {
            f.holes.iter().collect()
        };
        ring_counts.push(1 + holes.len());
        ring_sizes.push(f.outer.len());
        flat.extend_from_slice(&f.outer);
        for hole in holes {
            ring_sizes.push(hole.len());
            flat.extend_from_slice(hole);
        }
        attrs_list.push(f.attrs.clone().unwrap_or_default());
    }
    if !ring_counts.is_empty() {
        if flat.len() >= 1024 {
            let (vobjs, inverse) = mesh.bulk_weld(&flat);
            mesh.add_faces_welded(&vobjs, &inverse, &ring_sizes, &ring_counts, Some(&attrs_list), None);
        } else {
            let mut ci = 0;
            let mut ri = 0;
            for (fi, &nr) in ring_counts.iter().enumerate() {
                let mut rings: Vec<Vec<V3>> = Vec::new();
                for k in 0..nr {
                    let n = ring_sizes[ri + k];
                    rings.push(flat[ci..ci + n].to_vec());
                    ci += n;
                }
                ri += nr;
                // `add_face` raises on a corner that cannot be welded (not a
                // number); the import skips such a polygon after welding the
                // corners before it.
                if let Some((r, c)) = rings.iter().enumerate().find_map(|(r, ring)| ring.iter().position(|p| !finite(p)).map(|c| (r, c))) {
                    for ring in &rings[..r] {
                        for &p in ring {
                            mesh.vertex(p);
                        }
                    }
                    for &p in &rings[r][..c] {
                        mesh.vertex(p);
                    }
                    continue;
                }
                let face = mesh.add_face(&rings[0], &rings[1..]);
                for (k, v) in &attrs_list[fi] {
                    match mesh.face_mut(face).attrs.iter_mut().find(|(x, _)| x == k) {
                        Some(slot) => slot.1 = v.clone(),
                        None => mesh.face_mut(face).attrs.push((k.clone(), v.clone())),
                    }
                }
            }
        }
    }
    for &(a, b) in soft_edges {
        let (Some(va), Some(vb)) = (mesh.vertex_at(a), mesh.vertex_at(b)) else { continue };
        if va == vb {
            continue;
        }
        if let Some(e) = mesh.find_edge(va, vb) {
            mesh.edge_mut(e).soft = true;
        }
    }
    mesh
}

// ---- the fused build (images and face-me components) ---------------------------------------

/// `core._freeze`-like text of an attrs value for `formats.fuse._freeze_attr`:
/// equal for equal values, objects with their keys sorted.
fn frozen_text(v: &Value, out: &mut String) {
    use std::fmt::Write;
    match v {
        Value::Object(o) => {
            let mut items: Vec<&(String, Value)> = o.iter().collect();
            items.sort_by(|a, b| a.0.cmp(&b.0));
            out.push('{');
            for (k, x) in items {
                let _ = write!(out, "{}:{k}=", k.len());
                frozen_text(x, out);
                out.push(',');
            }
            out.push('}');
        }
        Value::Array(a) => {
            out.push('[');
            for x in a {
                frozen_text(x, out);
                out.push(',');
            }
            out.push(']');
        }
        Value::Float(x) => {
            // 1 and 1.0 are one key in Python
            if x.fract() == 0.0 && x.abs() < 9e15 {
                let _ = write!(out, "n{}", *x as i64);
            } else {
                let _ = write!(out, "d{:016x}", x.to_bits());
            }
        }
        Value::Int(i) => {
            let _ = write!(out, "n{i}");
        }
        other => value_text(other, out),
    }
}

/// `round(x, 4)` of a texture map entry, as a number.
fn round4(x: f64) -> f64 {
    if !x.is_finite() {
        return x;
    }
    format!("{x:.4}").parse::<f64>().map(|r| r + 0.0).unwrap_or(x)
}

/// What `fuse._attrs_sig(attrs)` and `_fuse_native` derive per loop: the
/// signature text, the fuzzy-UV group key and the rounded uvw, and the rank.
struct FuseSig {
    sig: String,
    group: Option<String>,
    uvw: Vec<f64>,
    rank: u8,
}

fn fuse_sig(attrs: &Option<Attrs>) -> FuseSig {
    let empty = Attrs::new();
    let a = attrs.as_ref().unwrap_or(&empty);
    let get = |k: &str| a.iter().rev().find(|(x, _)| x == k).map(|(_, v)| v);
    let color = get("color").filter(|c| !matches!(c, Value::Null));
    let tex = get("texture").filter(|t| ingetrazo_model::pyval::truthy(t));
    let (mut tsig, mut uvw, mut path) = (String::from("none"), Vec::new(), String::new());
    if let Some(t) = tex {
        let tg = |k: &str| t.get(k).cloned();
        let path_v = tg("path");
        let mut text = String::from("t:");
        for v in [path_v.clone(), tg("sw"), tg("sh"), Some(tg("rot").unwrap_or(Value::int(0)))] {
            match v {
                Some(v) => frozen_text(&v, &mut text),
                None => text.push('N'),
            }
            text.push('|');
        }
        if let Some(Value::Array(u)) = tg("uvw").filter(ingetrazo_model::pyval::truthy) {
            uvw = u.iter().filter_map(|x| if let Value::Float(f) = x { Some(round4(*f)) } else { None }).collect();
            for x in &uvw {
                text.push_str(&format!("{:016x},", x.to_bits()));
            }
        } else {
            text.push('N');
        }
        if let Some(Value::Str(p)) = path_v {
            path = p;
        }
        tsig = text;
    }
    let mut ctext = String::from("none");
    if let Some(c) = color {
        ctext.clear();
        frozen_text(c, &mut ctext);
    }
    let mut extras: Vec<&(String, Value)> = a.iter().filter(|(k, _)| k != "color" && k != "texture").collect();
    // dict order is insertion order; a later key replaced an earlier one in place
    extras.sort_by(|x, y| x.0.cmp(&y.0));
    let mut etext = String::new();
    for (k, v) in extras {
        etext.push_str(&format!("{}:{k}=", k.len()));
        frozen_text(v, &mut etext);
        etext.push(',');
    }
    let has_tex = tex.is_some();
    let group = (has_tex && !uvw.is_empty()).then(|| format!("{ctext}#{etext}#{path}"));
    FuseSig { sig: format!("{ctext}#{tsig}#{etext}"), group, uvw, rank: if has_tex { 2 } else { 1 } }
}

/// `fuse_coplanar_loops(raw)` through the engine, as `_fuse_native` hands it
/// the loops: each result is `(outer, holes, attrs, originals)`.
type Fused = (Vec<V3>, Vec<Vec<V3>>, Option<Attrs>, Vec<Vec<V3>>);

fn fuse_loops(loops: &[(&[V3], &Option<Attrs>)], h: Hypot) -> Vec<Fused> {
    use ingetrazo_geom::fuse::{fuse, Input, Region};
    let mut flat: Vec<V3> = Vec::new();
    let mut starts = vec![0usize];
    let (mut sigs, mut groups, mut ranks, mut uvws) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let mut sig_ids: HashMap<String, i64> = HashMap::new();
    let mut group_ids: HashMap<String, i64> = HashMap::new();
    for (pts, attrs) in loops {
        flat.extend_from_slice(pts);
        starts.push(flat.len());
        let s = fuse_sig(attrs);
        let n = sig_ids.len() as i64;
        sigs.push(*sig_ids.entry(s.sig).or_insert(n));
        match s.group {
            Some(g) => {
                let n = group_ids.len() as i64;
                groups.push(*group_ids.entry(g).or_insert(n));
            }
            None => groups.push(-1),
        }
        uvws.push(s.uvw);
        ranks.push(s.rank);
    }
    let inp = Input { pts: &flat, loop_start: &starts, sig: &sigs, tex_group: &groups, rank: &ranks, uvw: &uvws };
    let mut out = Vec::new();
    for region in fuse(&inp, 0.99999, false, h) {
        match region {
            Region::Traced(rings, lps) => {
                let rings: Vec<Vec<V3>> = rings.iter().map(|r| r.iter().map(|&k| flat[k]).collect()).collect();
                let mut it = rings.into_iter();
                let outer = it.next().unwrap_or_default();
                out.push((outer, it.collect(), loops[lps[0]].1.clone(), lps.iter().map(|&f| loops[f].0.to_vec()).collect()));
            }
            Region::Pass(lps) => {
                for f in lps {
                    out.push((loops[f].0.to_vec(), Vec::new(), loops[f].1.clone(), vec![loops[f].0.to_vec()]));
                }
            }
        }
    }
    out
}

fn update_attrs(face: &mut Vec<(String, Value)>, attrs: &Option<Attrs>) {
    if let Some(a) = attrs {
        for (k, v) in a {
            match face.iter_mut().find(|(x, _)| x == k) {
                Some(slot) => slot.1 = v.clone(),
                None => face.push((k.clone(), v.clone())),
            }
        }
    }
}

/// `_build_mesh(faces, None)`: the legacy clean-up of the DAE import. Faces
/// without holes are fused into coplanar same-paint regions (`_add_fused`),
/// faces with holes keep their rings, and shallow same-paint seams go soft.
fn build_mesh_fused(faces: &[PFace], h: Hypot) -> Mesh {
    let mut mesh = Mesh::new();
    mesh.hypot = h;
    let raw: Vec<(&[V3], &Option<Attrs>)> = faces.iter().filter(|f| f.holes.is_empty()).map(|f| (f.outer.as_slice(), &f.attrs)).collect();
    let add = |mesh: &mut Mesh, outer: &[V3], holes: &[Vec<V3>], attrs: &Option<Attrs>| -> bool {
        // `add_face` raises on a corner that is not a number
        if !outer.iter().all(finite) || !holes.iter().all(|r| r.iter().all(finite)) {
            return false;
        }
        let f = mesh.add_face(outer, holes);
        if attrs.as_ref().is_some_and(|a| !a.is_empty()) {
            update_attrs(&mut mesh.face_mut(f).attrs, attrs);
        }
        true
    };
    for (outer, holes, attrs, originals) in fuse_loops(&raw, h) {
        if !add(&mut mesh, &outer, &holes, &attrs) {
            // an irregular region keeps its pieces
            for pts in &originals {
                add(&mut mesh, pts, &[], &attrs);
            }
        }
    }
    for f in faces.iter().filter(|f| !f.holes.is_empty()) {
        add(&mut mesh, &f.outer, &f.holes, &f.attrs);
    }
    crate::glb::soften_smooth_edges(&mut mesh, h);
    mesh
}

fn push_mesh(scene: &mut Scene, mesh: Mesh) -> MeshId {
    scene.meshes.push(mesh);
    MeshId(scene.meshes.len() as u32 - 1)
}

/// `apply_payload(scene, payload)`: the backend name.
pub fn apply_payload(scene: &mut Scene, mut payload: Payload, h: Hypot) -> R<&'static str> {
    // The file's named materials join the scene registry.
    if !payload.materials.is_empty() {
        let mut registry = Registry::new();
        for (n, v) in &scene.materials {
            if let Ok(m) = SceneMaterial::from_dict(v) {
                let _ = registry.insert(Value::Str(n.clone()), m);
            }
        }
        let mut remap: HashMap<String, String> = HashMap::new();
        for raw in &payload.materials {
            let mat = SceneMaterial::from_dict(raw).map_err(|e| SkpLoadError::Unreadable(format!("{}: {}", e.kind, e.msg)))?;
            if !ingetrazo_model::pyval::truthy(&mat.name) {
                continue;
            }
            let before = registry.len();
            let name = match &mat.name {
                Value::Str(n) => n.clone(),
                _ => String::new(),
            };
            let final_name = register(&mut registry, mat).map_err(|e| SkpLoadError::Unreadable(format!("{}: {}", e.kind, e.msg)))?;
            let final_name = match final_name {
                Value::Str(n) => n,
                _ => String::new(),
            };
            if registry.len() > before {
                let (_, m) = registry.entries.last().unwrap();
                scene.materials.push((final_name.clone(), m.to_dict().map_err(|e| SkpLoadError::Unreadable(format!("{}: {}", e.kind, e.msg)))?));
            }
            if final_name != name {
                remap.insert(name, final_name);
            }
        }
        if !remap.is_empty() {
            let fix = |faces: &mut Vec<PFace>| {
                for f in faces {
                    if let Some(attrs) = &mut f.attrs {
                        for (k, v) in attrs.iter_mut() {
                            if k == "mat" {
                                if let Value::Str(n) = v {
                                    if let Some(to) = remap.get(n) {
                                        *n = to.clone();
                                    }
                                }
                            }
                        }
                    }
                }
            };
            for g in &mut payload.groups {
                fix(&mut g.faces);
            }
            for p in &mut payload.protos {
                fix(&mut p.faces);
            }
        }
    }

    // The file's layers (tags) join the scene's, keeping their visibility.
    {
        let mut known: HashSet<String> = scene
            .layers
            .iter()
            .filter_map(|l| match l.get("name") {
                Some(Value::Str(n)) => Some(n.clone()),
                _ => None,
            })
            .collect();
        for (name, visible) in &payload.layers {
            if !name.is_empty() && !known.contains(name) {
                let mut layer = SceneLayer::new(name);
                layer.visible = Value::Bool(*visible);
                scene.layers.push(layer.to_dict());
                known.insert(name.clone());
            }
        }
    }

    // The file's scenes become saved views.
    {
        let mut existing: HashSet<String> = scene.extra.saved_views.iter().map(|v| v.name.clone()).collect();
        for raw in &payload.scenes {
            if existing.contains(&raw.name) {
                continue;
            }
            let view = ingetrazo_model::saved_views::from_lookat(
                &raw.name,
                raw.eye,
                raw.target,
                raw.up,
                raw.fov,
                !raw.parallel,
                (raw.ortho_height != 0.0).then_some(raw.ortho_height),
                Some(raw.hidden_layers.clone()),
            )
            .map_err(|e| SkpLoadError::Unreadable(format!("{}: {}", e.kind, e.msg)))?;
            scene.extra.saved_views.push(view);
            existing.insert(raw.name.clone());
        }
    }

    // Linear dimensions: the .skp scalar offset becomes the in-plane
    // perpendicular times the distance.
    for raw in &payload.dimensions {
        let a = V3(raw.a[0] as f32, raw.a[1] as f32, raw.a[2] as f32);
        let b = V3(raw.b[0] as f32, raw.b[1] as f32, raw.b[2] as f32);
        let seg = b.sub(a);
        if (seg.length(h) as f64) < 1e-9 {
            continue;
        }
        let n = match raw.normal {
            Some(n) => V3(n[0] as f32, n[1] as f32, n[2] as f32),
            None => V3(0.0, 0.0, 1.0),
        };
        let mut perp = V3::cross(n, seg);
        if (perp.length(h) as f64) < 1e-9 {
            perp = V3::cross(V3(0.0, 0.0, 1.0), seg);
        }
        if (perp.length(h) as f64) < 1e-9 {
            perp = V3::cross(V3(1.0, 0.0, 0.0), seg);
        }
        let perp = perp.normalize(h);
        scene.dimensions.push(Dimension::new(a, b, perp.mul(raw.offset as f32), None, None, None));
    }

    // Leader texts.
    for raw in &payload.texts {
        if raw.hidden {
            continue;
        }
        let anchor = V3(raw.anchor[0] as f32, raw.anchor[1] as f32, raw.anchor[2] as f32);
        let label = match raw.label {
            Some(l) => V3(l[0] as f32, l[1] as f32, l[2] as f32),
            None => anchor,
        };
        scene.text_labels.push(TextLabel { anchor, offset: label.sub(anchor), text: raw.text.clone(), layer: None });
    }

    for gp in &payload.groups {
        let mesh = match &gp.soft_edges {
            Some(soft) => build_mesh(&gp.faces, soft, h),
            None => build_mesh_fused(&gp.faces, h),
        };
        if mesh.faces.is_empty() {
            continue;
        }
        let id = push_mesh(scene, mesh);
        let mut g = Group::new(id, gp.name.clone());
        if let Some(l) = gp.layer.as_ref().filter(|l| !l.is_empty()) {
            g.layer = Some(l.clone());
        }
        if let Some(axes) = &gp.axes {
            // the instance's own axes, column-major as data() wrote them
            let m = M4::from_data(*axes);
            if !m.is_identity() {
                g.axes = Some(m);
            }
        }
        if gp.billboard {
            // the real geometry turns toward the camera each frame
            g.billboard = Some(s("mesh"));
        }
        scene.groups.push(g);
    }

    // Shared components: one prototype mesh, a Group per placement.
    let proto_meshes: Vec<MeshId> = payload.protos.iter().map(|p| push_mesh(scene, build_mesh(&p.faces, &p.soft_edges, h))).collect();
    fn place(payload: &Payload, meshes: &[MeshId], idx: usize, xf: M4, layer: Option<&String>, depth: usize) -> Group {
        let pr = &payload.protos[idx];
        let mut g = Group::new(meshes[idx], pr.name.clone());
        g.xform = Some(xf);
        if let Some(l) = layer.filter(|l| !l.is_empty()) {
            g.layer = Some(l.clone());
        }
        if depth < 32 {
            let kids: Vec<Group> = pr.children.iter().map(|k| place(payload, meshes, k.proto, k.xform, k.layer.as_ref(), depth + 1)).collect();
            g.adopt(kids);
        }
        g
    }
    for (idx, pr) in payload.protos.iter().enumerate() {
        let empty = scene.mesh(proto_meshes[idx]).faces.is_empty();
        if empty && pr.children.is_empty() {
            continue;
        }
        for (i, xf) in pr.instances.iter().enumerate() {
            let layer = pr.instance_layers.as_ref().and_then(|l| l.get(i)).and_then(|l| l.as_ref());
            scene.groups.push(place(&payload, &proto_meshes, idx, *xf, layer, 0));
        }
    }
    if let Some(back) = payload.back_color {
        if scene.extra.back_face_color.is_none() {
            scene.extra.back_face_color = Some(Value::Array(back.iter().map(|&c| fl(c)).collect()));
        }
    }
    scene.bind_dimensions(MeshId(0));
    Ok("openskp")
}

/// `load_skp(scene, path)`: `parse_skp` then `apply_payload`.
pub fn load_skp(scene: &mut Scene, path: &Path, cache_root: &Path, h: Hypot) -> R<&'static str> {
    let payload = parse_skp(path, cache_root)?;
    apply_payload(scene, payload, h)
}
