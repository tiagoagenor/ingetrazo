// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! Wavefront OBJ with its `.mtl`: `formats/obj.py`.
//!
//! Export is `save_obj` (the per-face loop: the array version needs
//! Python's native engine). Python opens the files in text mode, so on Windows
//! every line ends in `\r\n`; [`ObjExport`] holds the text with `\n` and the
//! writers take `crlf` to say which the reference machine used.
//!
//! Import is [`parse_obj`] (everything `_load_obj_inner` reads) and
//! [`add_obj_faces`] (the single-piece path up to the edit engine). The rest
//! of the import (`run_stitch`, `orient_outward`, the coplanar fusing of big
//! or multi-part models) lives in the operations crate (M3); where Python
//! runs it, the caller gets a hook ([`Stitch`]) or an `Err` saying so.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use ingetrazo_formats::pyjson::Value;
use ingetrazo_geom::sweep::py_hypot3;
use ingetrazo_math::{Hypot, V3};
use ingetrazo_model::mesh::{key, FaceId, Key, Mesh};
use ingetrazo_model::pyalnum::is_alnum;
use ingetrazo_model::pyval;
use ingetrazo_model::scene::Scene;
use ingetrazo_model::texture::path_name;

use crate::pyutil;
use crate::tri::{export_triangles, face_normal};
use crate::uv::{affine_uv, fit_uv_affine, planar_uv};
use crate::world::{world_faces, WFace};

/// Cream painted on faces with no material colour.
pub const DEFAULT_COLOR: [f64; 3] = [0.96, 0.95, 0.925];

// ---- export ---------------------------------------------------------------------

/// What `save_obj` writes.
#[derive(Clone, Debug, PartialEq)]
pub struct ObjExport {
    pub obj: String,
    pub mtl: String,
    /// Texture images to copy next to the `.obj`: `(source path, file name)`,
    /// in material order. Python copies one only when it is a different file
    /// than the destination and is an image.
    pub textures: Vec<(String, String)>,
}

#[derive(Clone, PartialEq, Eq, Hash)]
enum MatKey {
    Tex(String),
    Color([u64; 3]),
}

struct Mat {
    color: [f64; 3],
    map: Option<String>,
    src: Option<String>,
    mat: Option<Value>,
}

/// A float as a dict key: `-0.0 == 0.0`.
fn fkey(x: f64) -> u64 {
    (x + 0.0).to_bits()
}

/// A rounded coordinate as `round(x, 6)` keys a dict.
fn round_key(x: f64) -> (bool, i64) {
    // A float32 times 1e6 is exact in double, so ties-to-even of the product
    // is `round(x, 6)`; anything wider goes through the decimal string.
    if x.abs() < 1e9 {
        (false, (x * 1e6).round_ties_even() as i64)
    } else {
        (true, fkey(pyutil::round6(x)) as i64)
    }
}

fn color_of(attrs: &WFace) -> [f64; 3] {
    match attrs.attr("color") {
        Some(Value::Array(c)) if !c.is_empty() => {
            let mut out = DEFAULT_COLOR;
            for (k, v) in c.iter().take(3).enumerate() {
                out[k] = pyval::float(v).unwrap_or(0.0);
            }
            out
        }
        _ => DEFAULT_COLOR,
    }
}

fn number_or_one(tex: &Value, key: &str) -> f64 {
    match tex.get(key) {
        Some(v) if pyval::truthy(v) => pyval::float(v).unwrap_or(1.0),
        _ => 1.0,
    }
}

/// `export_names`: a unique, export-safe name per material.
fn export_names(mats: &[&Mat]) -> Vec<String> {
    let mut used: HashSet<String> = HashSet::new();
    let mut out = Vec::new();
    for (i, info) in mats.iter().enumerate() {
        let raw = match &info.mat {
            Some(Value::Str(s)) => s.trim_matches(|c: char| pyutil::is_space(c)).to_string(),
            _ => String::new(),
        };
        let base = if raw.is_empty() {
            format!("mat{i}")
        } else {
            let cleaned: String = raw.chars().map(|c| if is_alnum(c) || matches!(c, '.' | '_' | '-') { c } else { '_' }).collect();
            let trimmed = cleaned.trim_matches(['.', '_', '-']);
            let mut b = if trimmed.is_empty() { format!("mat{i}") } else { trimmed.to_string() };
            if b.chars().next().is_some_and(|c| c.is_ascii_digit() || matches!(c, '\u{b2}' | '\u{b3}' | '\u{b9}')) {
                b = format!("m_{b}");
            }
            b
        };
        let (mut name, mut n) = (base.clone(), 2);
        while used.contains(&name) {
            name = format!("{base}_{n}");
            n += 1;
        }
        used.insert(name.clone());
        out.push(name);
    }
    out
}

type Corner = (usize, Option<usize>);

/// `save_obj(scene, path)`: the texts of the `.obj` and the `.mtl`
/// (`mtl_name` the `.mtl` file name its `mtllib` line names).
pub fn save_obj_text(scene: &Scene, mtl_name: &str, h: Hypot) -> ObjExport {
    let mut verts: Vec<V3> = Vec::new();
    let mut vindex: HashMap<[(bool, i64); 3], usize> = HashMap::new();
    let mut uvs: Vec<(f64, f64)> = Vec::new();
    let mut uvindex: HashMap<(u64, u64), usize> = HashMap::new();
    let mut mats: Vec<Mat> = Vec::new();
    let mut mat_index: HashMap<MatKey, usize> = HashMap::new();
    // Materials with at least one triangle, in order of the first one.
    let mut order: Vec<usize> = Vec::new();
    let mut groups: Vec<Vec<[Corner; 3]>> = Vec::new();
    let mut group_of: HashMap<usize, usize> = HashMap::new();

    for face in world_faces(scene) {
        let tex = face.attr("texture").filter(|t| t.get("path").is_some_and(pyval::truthy)).cloned();
        let (mi, uv_rule) = if let Some(tex) = &tex {
            let path = match tex.get("path") {
                Some(Value::Str(p)) => p.clone(),
                _ => String::new(),
            };
            let name = path_name(&path, cfg!(windows));
            let key = MatKey::Tex(name.clone());
            let mi = *mat_index.entry(key).or_insert_with(|| {
                mats.push(Mat { color: [1.0, 1.0, 1.0], map: Some(name.clone()), src: Some(path.clone()), mat: None });
                mats.len() - 1
            });
            let sw = number_or_one(tex, "sw");
            let sh = number_or_one(tex, "sh");
            let rot = tex.get("rot").map(|v| pyval::float(v).unwrap_or(0.0)).unwrap_or(0.0);
            let uvw = match tex.get("uvw") {
                Some(Value::Array(a)) if !a.is_empty() => {
                    let mut w = [0.0f64; 8];
                    for (k, v) in a.iter().take(8).enumerate() {
                        w[k] = pyval::float(v).unwrap_or(0.0);
                    }
                    Some(w)
                }
                _ => None,
            };
            (mi, Some((sw, sh, rot, uvw)))
        } else {
            let col = color_of(&face);
            let key = MatKey::Color([fkey(col[0]), fkey(col[1]), fkey(col[2])]);
            let mi = *mat_index.entry(key).or_insert_with(|| {
                mats.push(Mat { color: col, map: None, src: None, mat: None });
                mats.len() - 1
            });
            (mi, None)
        };
        if let Some(m) = face.attr("mat").filter(|m| pyval::truthy(m)) {
            if mats[mi].mat.is_none() {
                mats[mi].mat = Some(m.clone());
            }
        }
        let normal = face_normal(&face.outer, h);
        for tri in export_triangles(&face, h) {
            let uv = uv_rule.as_ref().map(|(sw, sh, rot, uvw)| match uvw {
                Some(w) => affine_uv(w, &tri),
                None => planar_uv(normal, &tri, *sw, *sh, *rot),
            });
            let mut corners = [(0usize, None); 3];
            for k in 0..3 {
                let p = tri[k];
                let vk = [round_key(p.0 as f64), round_key(p.1 as f64), round_key(p.2 as f64)];
                let vi = *vindex.entry(vk).or_insert_with(|| {
                    verts.push(p);
                    verts.len()
                });
                let ti = uv.as_ref().map(|uv| {
                    let (u, v) = uv[k];
                    let key = (fkey(pyutil::round6(u)), fkey(pyutil::round6(v)));
                    *uvindex.entry(key).or_insert_with(|| {
                        uvs.push((u, v));
                        uvs.len()
                    })
                });
                corners[k] = (vi, ti);
            }
            let gi = *group_of.entry(mi).or_insert_with(|| {
                order.push(mi);
                groups.push(Vec::new());
                groups.len() - 1
            });
            groups[gi].push(corners);
        }
    }

    let used: Vec<&Mat> = order.iter().map(|&i| &mats[i]).collect();
    let names = export_names(&used);
    let mut textures = Vec::new();
    let mut mtl = String::new();
    for (m, name) in used.iter().zip(&names) {
        if let (Some(map), Some(src)) = (&m.map, &m.src) {
            textures.push((src.clone(), map.clone()));
        }
        mtl.push_str(&format!("newmtl {name}\n"));
        mtl.push_str(&format!("Kd {:.4} {:.4} {:.4}\n", m.color[0], m.color[1], m.color[2]));
        if let Some(map) = &m.map {
            mtl.push_str(&format!("map_Kd {map}\n"));
        }
        mtl.push('\n');
    }
    let mut obj = String::from("# IngeTrazo OBJ export\n");
    obj.push_str(&format!("mtllib {mtl_name}\n"));
    for v in &verts {
        obj.push_str(&format!("v {:.6} {:.6} {:.6}\n", v.0 as f64, v.1 as f64, v.2 as f64));
    }
    for (u, v) in &uvs {
        obj.push_str(&format!("vt {u:.6} {v:.6}\n"));
    }
    for (gi, name) in names.iter().enumerate() {
        obj.push_str(&format!("usemtl {name}\n"));
        for tri in &groups[gi] {
            let toks: Vec<String> =
                tri.iter().map(|&(vi, ti)| match ti { Some(t) => format!("{vi}/{t}"), None => format!("{vi}") }).collect();
            obj.push_str(&format!("f {}\n", toks.join(" ")));
        }
    }
    ObjExport { obj, mtl, textures }
}

fn eol(text: &str, crlf: bool) -> Vec<u8> {
    if crlf {
        text.replace('\n', "\r\n").into_bytes()
    } else {
        text.as_bytes().to_vec()
    }
}

/// `save_obj(scene, path)`: writes `path`, the `.mtl` beside it and copies the
/// textures. A texture that cannot be copied is skipped, as in Python.
pub fn save_obj(scene: &Scene, path: &Path, h: Hypot, crlf: bool) -> std::io::Result<()> {
    let file = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    let mtl_file = pyutil::with_suffix(&file, ".mtl");
    let export = save_obj_text(scene, &mtl_file, h);
    let dir = path.parent().map(Path::to_path_buf).unwrap_or_default();
    for (src, name) in &export.textures {
        let (src, dst) = (PathBuf::from(src), dir.join(name));
        let same = match (std::fs::canonicalize(&src), std::fs::canonicalize(&dst)) {
            (Ok(a), Ok(b)) => a == b,
            _ => src == dst,
        };
        // The image check reads the file's own signature (Qt's TGA reader
        // is not here, so a TGA is not copied).
        if !same && ingetrazo_model::texture::is_image_file(&src, &|_| false) {
            let _ = std::fs::copy(&src, &dst);
        }
    }
    std::fs::write(path.with_file_name(&mtl_file), eol(&export.mtl, crlf))?;
    std::fs::write(path, eol(&export.obj, crlf))
}

// ---- import ---------------------------------------------------------------------

/// A `.mtl` entry: `Kd` and the resolved `map_Kd`.
#[derive(Clone, Debug, PartialEq)]
pub struct MtlMat {
    pub color: Option<[f64; 3]>,
    pub map: Option<String>,
}

/// `OBJ_UNITS`: metres per unit.
pub const OBJ_UNITS: [(&str, f64); 5] = [("m", 1.0), ("cm", 0.01), ("mm", 0.001), ("in", 0.0254), ("ft", 0.3048)];

/// `Y_UP_TO_Z_UP`: the quarter turn about X that stands a Y-up file up.
pub const Y_UP_TO_Z_UP: [f64; 9] = [1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0];

/// `_is_number`.
fn is_number(tok: &str) -> bool {
    pyutil::float(tok).is_some()
}

/// `_resolve_map(mtl, line)`.
fn resolve_map(mtl: &Path, line: &str) -> Option<String> {
    // `line.split(None, 1)[1]`: the rest after the tag, its leading
    // whitespace dropped.
    let line_t = line.trim_start_matches(|c: char| pyutil::is_space(c));
    let tag_end = line_t.find(|c: char| pyutil::is_space(c))?;
    let rest = line_t[tag_end..].trim_start_matches(|c: char| pyutil::is_space(c));
    if rest.is_empty() {
        return None;
    }
    let mut toks = pyutil::split(rest);
    while toks.first().is_some_and(|t| t.starts_with('-')) {
        toks.remove(0);
        while toks.first().is_some_and(|t| is_number(t)) {
            toks.remove(0);
        }
    }
    let joined = toks.join(" ");
    let value = joined.trim_matches(|c: char| pyutil::is_space(c)).trim_matches('"').to_string();
    if value.is_empty() {
        return None;
    }
    let here = mtl.parent().map(Path::to_path_buf).unwrap_or_default();
    let posix = value.replace('\\', "/");
    let name = posix.split('/').rfind(|p| !p.is_empty() && *p != ".").unwrap_or("").to_string();
    for cand in [here.join(&value), here.join(&name)] {
        if cand.is_file() {
            let s = cand.to_string_lossy().into_owned();
            return Some(if cfg!(windows) { s.replace('/', "\\") } else { s });
        }
    }
    None
}

/// `_parse_mtl(path)`: material name to its colour and image. `Err` where
/// Python raises (a `newmtl` without a name, a `Kd` that is not a number).
pub fn parse_mtl(path: &Path) -> Result<Vec<(String, MtlMat)>, String> {
    let mut mats: Vec<(String, MtlMat)> = Vec::new();
    let Ok(data) = std::fs::read(path) else { return Ok(mats) };
    let text = pyutil::decode_text(&data);
    let mut current: Option<usize> = None;
    for line in pyutil::splitlines(&text) {
        let parts = pyutil::split(line);
        if parts.is_empty() {
            continue;
        }
        if parts[0] == "newmtl" {
            let name = parts.get(1).ok_or("list index out of range")?.to_string();
            let entry = MtlMat { color: None, map: None };
            match mats.iter().position(|(n, _)| *n == name) {
                Some(i) => {
                    mats[i].1 = entry;
                    current = Some(i);
                }
                None => {
                    mats.push((name, entry));
                    current = Some(mats.len() - 1);
                }
            }
        } else if let Some(i) = current {
            if parts[0] == "Kd" && parts.len() >= 4 {
                let f = |s: &str| pyutil::float(s).ok_or_else(|| format!("could not convert string to float: {}", pyutil::repr(s)));
                mats[i].1.color = Some([f(parts[1])?, f(parts[2])?, f(parts[3])?]);
            } else if parts[0] == "map_Kd" && parts.len() >= 2 {
                mats[i].1.map = resolve_map(path, line);
            }
        }
    }
    Ok(mats)
}

/// `suggest_unit(path)` on the file's bytes: `"m"`, `"cm"` or `"mm"` from the
/// span of its `v` lines.
pub fn suggest_unit(data: &[u8]) -> &'static str {
    let mut lo = [f64::INFINITY; 3];
    let mut hi = [f64::NEG_INFINITY; 3];
    for line in data.split(|&b| b == b'\n') {
        if !line.starts_with(b"v ") {
            continue;
        }
        let text = String::from_utf8_lossy(line);
        let p = pyutil::split(&text);
        if p.len() < 4 {
            continue;
        }
        for k in 0..3 {
            let Some(v) = pyutil::float(p[k + 1]) else { return "m" };
            // Python's min/max: a NaN stays out of `lo`/`hi` only when first.
            if v < lo[k] {
                lo[k] = v;
            }
            if v > hi[k] {
                hi[k] = v;
            }
        }
    }
    let span = if hi[0] > lo[0] { (0..3).map(|k| hi[k] - lo[k]).fold(f64::NEG_INFINITY, f64::max) } else { 0.0 };
    if span > 20000.0 {
        "mm"
    } else if span > 200.0 {
        "cm"
    } else {
        "m"
    }
}

/// A polygon the file describes, before it becomes a face.
#[derive(Clone, Debug)]
pub struct PendingFace {
    pub corners: Vec<V3>,
    pub mat: Option<MtlMat>,
    pub loop_uv: Option<Vec<(f64, f64)>>,
    /// The `s` group the face was filed under (0 for none or `off`).
    pub smooth: i64,
    /// The file's `(o, g)` names: which part of the model it belongs to.
    pub piece: (String, String),
}

/// What `_load_obj_inner` reads from a file.
#[derive(Clone, Debug, Default)]
pub struct ObjModel {
    pub pending: Vec<PendingFace>,
}

/// Python's `int(text)` for the digits and sign OBJ indices use.
fn py_int(text: &str) -> Option<i64> {
    let t = text.trim_matches(|c: char| pyutil::is_space(c));
    let (neg, digits) = match t.strip_prefix('-') {
        Some(d) => (true, d),
        None => (false, t.strip_prefix('+').unwrap_or(t)),
    };
    if digits.is_empty() || digits.starts_with('_') || digits.ends_with('_') || digits.contains("__") {
        return None;
    }
    if !digits.chars().all(|c| c.is_ascii_digit() || c == '_') {
        return None;
    }
    let clean: String = digits.chars().filter(|&c| c != '_').collect();
    let n: i128 = clean.parse().ok()?;
    let n = if neg { -n } else { n };
    i64::try_from(n).ok()
}

/// `path.with_name(name)`.
fn with_name(path: &Path, name: &str) -> Result<PathBuf, String> {
    let bad = name.is_empty() || name == "." || name.contains('/') || (cfg!(windows) && name.contains('\\'));
    if bad {
        return Err(format!("Invalid name {}", pyutil::repr(name)));
    }
    Ok(path.with_file_name(name))
}

/// `_load_obj_inner`'s reading half: vertices, texture coordinates, faces with
/// their material, smoothing group and part. `scale` and `matrix` (row-major
/// 3x3, applied after `scale`) are as in Python.
pub fn parse_obj(path: &Path, scale: f64, matrix: Option<&[f64; 9]>) -> Result<ObjModel, String> {
    let data = std::fs::read(path).map_err(|e| e.to_string())?;
    parse_obj_text(&pyutil::decode_text(&data), path, scale, matrix)
}

/// [`parse_obj`] on the file's text; `path` locates the `.mtl` files.
pub fn parse_obj_text(text: &str, path: &Path, scale: f64, matrix: Option<&[f64; 9]>) -> Result<ObjModel, String> {
    let mut verts: Vec<V3> = Vec::new();
    let mut uvs: Vec<(f64, f64)> = Vec::new();
    let mut materials: Vec<(String, MtlMat)> = Vec::new();
    let mut current_mat: Option<MtlMat> = None;
    let mut current_smooth = 0i64;
    let (mut current_o, mut current_g) = (String::new(), String::new());
    let mut pending = Vec::new();
    let oob = || "list index out of range".to_string();
    let float = |s: &str| pyutil::float(s).ok_or_else(|| format!("could not convert string to float: {}", pyutil::repr(s)));

    for line in pyutil::splitlines(text) {
        let parts = pyutil::split(line);
        if parts.is_empty() {
            continue;
        }
        match parts[0] {
            "v" => {
                let mut x = float(parts.get(1).ok_or_else(oob)?)? * scale;
                let mut y = float(parts.get(2).ok_or_else(oob)?)? * scale;
                let mut z = float(parts.get(3).ok_or_else(oob)?)? * scale;
                if let Some(m) = matrix {
                    (x, y, z) = (m[0] * x + m[1] * y + m[2] * z, m[3] * x + m[4] * y + m[5] * z, m[6] * x + m[7] * y + m[8] * z);
                }
                verts.push(V3(x as f32, y as f32, z as f32));
            }
            "vt" => {
                let uv = (|| {
                    let u = pyutil::float(parts.get(1)?)?;
                    let v = if parts.len() > 2 { pyutil::float(parts[2])? } else { 0.0 };
                    Some((u, v))
                })();
                uvs.push(uv.unwrap_or((0.0, 0.0)));
            }
            "mtllib" => {
                materials = parse_mtl(&with_name(path, parts.get(1).ok_or_else(oob)?)?)?;
            }
            "usemtl" => {
                let name = parts.get(1).ok_or_else(oob)?;
                current_mat = materials.iter().find(|(n, _)| n == name).map(|(_, m)| m.clone());
            }
            "o" => {
                current_o = parts[1..].join(" ");
                current_g = String::new();
            }
            "g" => current_g = parts[1..].join(" "),
            "s" => {
                let v = if parts.len() > 1 { parts[1].to_lowercase() } else { "off".to_string() };
                current_smooth = if !v.is_empty() && v.chars().all(|c| c.is_ascii_digit()) { v.parse().unwrap_or(i64::MAX) } else { 0 };
            }
            "f" => {
                let mut idxs: Vec<i64> = Vec::new();
                let mut tidxs: Vec<i64> = Vec::new();
                let mut ok = true;
                for tok in &parts[1..] {
                    let bits: Vec<&str> = tok.split('/').collect();
                    let Some(raw) = py_int(bits[0]) else {
                        ok = false;
                        break;
                    };
                    idxs.push(if raw > 0 { raw - 1 } else { verts.len() as i64 + raw });
                    let t = bits.get(1).copied().unwrap_or("");
                    let ti = if t.is_empty() {
                        0
                    } else {
                        match py_int(t) {
                            Some(v) => v,
                            None => {
                                ok = false;
                                break;
                            }
                        }
                    };
                    tidxs.push(if t.is_empty() { -1 } else if ti > 0 { ti - 1 } else { uvs.len() as i64 + ti });
                }
                if !ok {
                    continue;
                }
                if idxs.len() >= 3 && idxs.iter().all(|&i| 0 <= i && (i as usize) < verts.len()) {
                    let loop_uv = if tidxs.iter().all(|&t| 0 <= t && (t as usize) < uvs.len()) {
                        Some(tidxs.iter().map(|&t| uvs[t as usize]).collect())
                    } else {
                        None
                    };
                    pending.push(PendingFace {
                        corners: idxs.iter().map(|&i| verts[i as usize]).collect(),
                        mat: current_mat.clone(),
                        loop_uv,
                        smooth: current_smooth,
                        piece: (current_o.clone(), current_g.clone()),
                    });
                }
            }
            _ => {}
        }
    }
    Ok(ObjModel { pending })
}

/// `SMOOTH_KEY`: where the import parks a face's smoothing group.
pub const SMOOTH_KEY: &str = "_s";

/// `_face_attrs(mat, loop, loop_uv, smooth)`: the attrs of one polygon, in
/// Python's insertion order.
pub fn face_attrs(f: &PendingFace) -> Vec<(String, Value)> {
    let mut base: Vec<(String, Value)> = Vec::new();
    if f.smooth != 0 {
        base.push((SMOOTH_KEY.to_string(), Value::int(f.smooth)));
    }
    let Some(mat) = &f.mat else { return base };
    if let Some(map) = &mat.map {
        let mut tex = vec![
            ("path".to_string(), Value::Str(map.clone())),
            ("sw".to_string(), Value::Float(1.0)),
            ("sh".to_string(), Value::Float(1.0)),
        ];
        if let Some(uv) = f.loop_uv.as_ref().filter(|u| !u.is_empty()) {
            if let Some(uvw) = fit_uv_affine(&f.corners, uv) {
                let glu = py_hypot3(uvw[0], uvw[1], uvw[2]);
                let glv = py_hypot3(uvw[4], uvw[5], uvw[6]);
                tex.push(("uvw".to_string(), Value::Array(uvw.iter().map(|&x| Value::Float(x)).collect())));
                pyval::set(&mut tex, "sw", Value::Float(if glu > 1e-9 { 1.0 / glu } else { 1.0 }));
                pyval::set(&mut tex, "sh", Value::Float(if glv > 1e-9 { 1.0 / glv } else { 1.0 }));
            }
        }
        base.push(("texture".to_string(), Value::Object(tex)));
        base.push(("back".to_string(), Value::Bool(true)));
        return base;
    }
    if let Some(color) = mat.color {
        let r4 = |x: f64| pyutil::round4(x);
        if color.map(r4) != DEFAULT_COLOR.map(r4) {
            base.push(("color".to_string(), Value::Array(color.iter().map(|&x| Value::Float(x)).collect())));
            base.push(("back".to_string(), Value::Bool(true)));
        }
    }
    base
}

/// The hook for what Python does to a small import after its faces are in:
/// `run_stitch(mesh, seed, new_faces, coplanar_merge=True)` and
/// `orient_outward(mesh)`, which belong to the operations crate.
pub type Stitch<'a> = &'a mut dyn FnMut(&mut Mesh, &HashSet<Key>, &HashSet<FaceId>);

/// How many polygons `_load_obj_inner` fuses as a reference group instead
/// of adding them to the loose mesh.
pub const MAX_FUSE_LOOPS: usize = 400;

/// `_load_obj_inner` for the path that needs no fuse pass (one part, at most
/// [`MAX_FUSE_LOOPS`] polygons): the faces go into the loose mesh, `stitch`
/// stands where Python runs `run_stitch` and `orient_outward`, then the
/// smoothing groups soften their edges and are forgotten.
///
/// `Err` for a model of several parts or more than [`MAX_FUSE_LOOPS`]
/// polygons: Python fuses those with `fuse_coplanar_loops`, not ported yet.
pub fn add_obj_faces(scene: &mut Scene, model: &ObjModel, h: Hypot, stitch: Option<Stitch>) -> Result<(), String> {
    let mut pieces: Vec<&(String, String)> = Vec::new();
    for p in &model.pending {
        if !pieces.contains(&&p.piece) {
            pieces.push(&p.piece);
        }
    }
    if pieces.len() > 1 {
        return Err("an OBJ of several parts is fused part by part (fuse_coplanar_loops), which is not ported yet".to_string());
    }
    if model.pending.len() > MAX_FUSE_LOOPS {
        return Err("an OBJ of more than 400 polygons is fused (fuse_coplanar_loops), which is not ported yet".to_string());
    }
    let mesh = &mut scene.meshes[0];
    let mut seed: HashSet<Key> = HashSet::new();
    let mut new_faces: HashSet<FaceId> = HashSet::new();
    for p in &model.pending {
        // Python's `add_face` raises on a NaN or infinite corner (its weld
        // key is an int), after welding the corners before it; the import
        // skips such a polygon ("skip a degenerate polygon").
        if let Some(bad) = p.corners.iter().position(|c| !(c.0.is_finite() && c.1.is_finite() && c.2.is_finite())) {
            for &c in &p.corners[..bad] {
                mesh.vertex(c);
            }
            continue;
        }
        let f = mesh.add_face(&p.corners, &[]);
        new_faces.insert(f);
        for (k, v) in face_attrs(p) {
            pyval::set(&mut mesh.face_mut(f).attrs, &k, v);
        }
        for &c in &p.corners {
            seed.insert(key(c));
        }
    }
    if let Some(stitch) = stitch {
        stitch(mesh, &seed, &new_faces);
    }
    soften_by_smoothing_group(mesh, h);
    drop_smoothing_groups(mesh);
    Ok(())
}

/// `_smooth_sig(attrs)`: colour and texture image only.
fn smooth_sig(attrs: &[(String, Value)]) -> (Option<Value>, Option<Value>) {
    let c = crate::world::attr(attrs, "color").filter(|c| !matches!(c, Value::Null)).cloned();
    let t = crate::world::attr(attrs, "texture").filter(|t| pyval::truthy(t)).and_then(|t| t.get("path").cloned());
    (c, t)
}

/// `soften_by_smoothing_group(mesh)`: the edges the file itself calls smooth.
pub fn soften_by_smoothing_group(mesh: &mut Mesh, h: Hypot) {
    const CREASE_COS: f64 = 0.17;
    let edges: Vec<_> = mesh.edges.clone();
    for e in edges {
        let ed = mesh.edge(e);
        if ed.soft || ed.faces.len() != 2 {
            continue;
        }
        let (f0, f1) = (ed.faces[0], ed.faces[1]);
        if f0 == f1 {
            continue;
        }
        let (a0, a1) = (&mesh.face(f0).attrs, &mesh.face(f1).attrs);
        if a0.is_empty() || a1.is_empty() {
            continue;
        }
        let g = crate::world::attr(a0, SMOOTH_KEY);
        if !g.is_some_and(pyval::truthy) || g != crate::world::attr(a1, SMOOTH_KEY) {
            continue;
        }
        if smooth_sig(a0) != smooth_sig(a1) {
            continue;
        }
        let normal = |f: FaceId| {
            let outer: Vec<V3> = mesh.face(f).outer.iter().map(|&v| mesh.position(v)).collect();
            face_normal(&outer, h)
        };
        let (n0, n1) = (normal(f0), normal(f1));
        if (n0.length(h) as f64) < 1e-9 || (n1.length(h) as f64) < 1e-9 || (n0.dot(n1) as f64).abs() > CREASE_COS {
            mesh.edge_mut(e).soft = true;
        }
    }
}

/// `drop_smoothing_groups(mesh)`.
pub fn drop_smoothing_groups(mesh: &mut Mesh) {
    for f in mesh.faces.clone() {
        mesh.face_mut(f).attrs.retain(|(k, _)| k != SMOOTH_KEY);
    }
}

/// `_tidy_blender_name(name)`.
pub fn tidy_blender_name(name: &str) -> String {
    let mut name = name.to_string();
    if let Some(i) = name.rfind('_') {
        let (head, tail) = (&name[..i], &name[i + 1..]);
        if !head.is_empty() && blender_mesh_part(tail) {
            name = head.to_string();
        }
    }
    // `re.sub(r"(\.\d{3})+$", "", name).strip() or name`
    let mut stripped = name.as_str();
    while stripped.len() >= 4 {
        let (rest, tail) = stripped.split_at(stripped.len() - 4);
        let b = tail.as_bytes();
        if b[0] == b'.' && b[1..].iter().all(u8::is_ascii_digit) {
            stripped = rest;
        } else {
            break;
        }
    }
    let t = stripped.trim_matches(|c: char| pyutil::is_space(c));
    if t.is_empty() { name } else { t.to_string() }
}

/// `re.fullmatch(r"[A-Z][A-Za-z]*(\.\d+)?", tail)`.
fn blender_mesh_part(tail: &str) -> bool {
    let b = tail.as_bytes();
    if b.is_empty() || !b[0].is_ascii_uppercase() {
        return false;
    }
    let mut i = 1;
    while i < b.len() && b[i].is_ascii_alphabetic() {
        i += 1;
    }
    if i == b.len() {
        return true;
    }
    b[i] == b'.' && i + 1 < b.len() && b[i + 1..].iter().all(u8::is_ascii_digit)
}

/// `_piece_name(key, number)` with the "Part {n}" label already translated
/// (`part` holds `{n}`).
pub fn piece_name(key: &(String, String), number: usize, part: &str) -> String {
    let (obj_name, group_name) = key;
    let pick = if group_name.is_empty() { obj_name } else { group_name };
    let name = tidy_blender_name(pick.trim_matches(|c: char| pyutil::is_space(c)));
    if name.is_empty() {
        return part.replace("{n}", &number.to_string());
    }
    if !name.chars().any(char::is_alphabetic) {
        return part.replace("{n}", &name);
    }
    name
}
