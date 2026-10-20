// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! glTF 2.0 / GLB export: `formats/gltf.py` `save_glb` over
//! `formats/meshexport.py`, byte for byte.
//!
//! The model keeps its structure (one node per group and component, nested
//! as in the model, copies of a component sharing one mesh, the loose
//! geometry a node of its own, face-me figures standing toward the Front
//! view as nodes with their feet and heading in `extras`). Positions and
//! normals are baked `(x, y, z) -> (x, z, -y)`; images ride in the one
//! binary buffer.
//!
//! Not ported: the camera path (`face_me=`, Render with Blender) and the
//! `only=` selection filter.

use std::collections::HashMap;
use std::path::Path;

use ingetrazo_formats::pyjson::{dumps, Options, Value};
use ingetrazo_math::{Hypot, M4, V3};
use ingetrazo_model::mesh::Mesh;
use ingetrazo_model::pyval;
use ingetrazo_model::scene::{Group, MeshId, Scene};
use ingetrazo_model::texture::{is_image_bytes, path_name};

use crate::figure::{facing_quad, feet, transformed_mesh, turned_toward};
use crate::finish;
use crate::img::image_has_cutout;
use crate::obj::{fkey, round_key, DEFAULT_COLOR};
use crate::pyutil;
use crate::tri::{export_triangles, face_normal};
use crate::uv::{affine_uv, planar_uv};
use crate::world::{billboard, face_visible, group_visible, mesh_faces, WFace};

const F32: i64 = 5126;
const U32: i64 = 5125;
const ARRAY_BUFFER: i64 = 34962;
const ELEMENT_ARRAY_BUFFER: i64 = 34963;

/// The words a file carries that the app translates (`tr(...)`): the names
/// of the loose geometry, of a nameless group and of a nameless figure.
#[derive(Clone, Debug)]
pub struct Labels {
    pub loose: String,
    pub group: String,
    pub figure: String,
}

impl Default for Labels {
    /// The untranslated texts, as `tr` returns them with no language loaded.
    fn default() -> Self {
        Labels { loose: "Loose geometry".into(), group: "Group".into(), figure: "Figure".into() }
    }
}

// ---- values ---------------------------------------------------------------------

fn s(text: &str) -> Value {
    Value::Str(text.to_string())
}

fn int(i: i64) -> Value {
    Value::int(i)
}

fn fl(x: f64) -> Value {
    Value::Float(x)
}

fn obj(items: Vec<(&str, Value)>) -> Value {
    Value::Object(items.into_iter().map(|(k, v)| (k.to_string(), v)).collect())
}

fn set(v: &mut Value, key: &str, value: Value) {
    if let Value::Object(items) = v {
        pyval::set(items, key, value);
    }
}

fn pop(v: &mut Value, key: &str) {
    if let Value::Object(items) = v {
        items.retain(|(k, _)| k != key);
    }
}

fn entry<'a>(v: &'a mut Value, key: &str) -> Option<&'a mut Value> {
    match v {
        Value::Object(items) => items.iter_mut().find(|(k, _)| k == key).map(|(_, v)| v),
        _ => None,
    }
}

// ---- collecting the triangles by material -------------------------------------

/// A material's key: `("color", rgb[, opacity])` or `("tex", name[, opacity])`.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
enum MKey {
    Color(Vec<u64>, Option<u64>),
    Tex(String, Option<u64>),
}

impl MKey {
    fn is_tex(&self) -> bool {
        matches!(self, MKey::Tex(..))
    }
}

struct MatInfo {
    color: Vec<Value>,
    map: Option<String>,
    src: Option<String>,
    opacity: Option<f64>,
    mat: Option<Value>,
    finish: String,
}

type Corner = (V3, Option<(f64, f64)>);

struct Tri {
    normal: V3,
    verts: [Corner; 3],
}

/// An insertion-ordered `dict[key, list of triangles]`.
#[derive(Default)]
struct Prims {
    keys: Vec<MKey>,
    tris: HashMap<MKey, Vec<Tri>>,
}

impl Prims {
    fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }
    fn push(&mut self, key: &MKey, tri: Tri) {
        if !self.tris.contains_key(key) {
            self.keys.push(key.clone());
        }
        self.tris.entry(key.clone()).or_default().push(tri);
    }
}

/// `materials`: the registry of every material seen, in order.
#[derive(Default)]
struct Registry {
    list: Vec<(MKey, MatInfo)>,
    index: HashMap<MKey, usize>,
}

fn number_or_one(tex: &Value, key: &str) -> f64 {
    match tex.get(key) {
        Some(v) if pyval::truthy(v) => pyval::float(v).unwrap_or(1.0),
        _ => 1.0,
    }
}

/// `_add_face(face, materials, prims)`.
fn add_face(face: &WFace, reg: &mut Registry, prims: &mut Prims, h: Hypot) -> Result<(), String> {
    let n = face_normal(&face.outer, h);
    let op = match face.attr("opacity") {
        None | Some(Value::Null) => None,
        Some(v) => {
            let x = pyval::float(v).map_err(|e| format!("{}: {}", e.kind, e.msg))?;
            if x >= 0.999 {
                None
            } else {
                Some(pyutil::round3(x))
            }
        }
    };
    let opk = op.map(fkey);
    let tex = face.attr("texture").filter(|t| t.get("path").is_some_and(pyval::truthy)).cloned();
    let tris = export_triangles(face, h);
    let (key, uv_rule) = if let Some(tex) = &tex {
        let path = match tex.get("path") {
            Some(Value::Str(p)) => p.clone(),
            _ => return Err("TypeError: texture path is not a string".to_string()),
        };
        let name = path_name(&path, cfg!(windows));
        let key = MKey::Tex(name.clone(), opk);
        if !reg.index.contains_key(&key) {
            reg.index.insert(key.clone(), reg.list.len());
            reg.list.push((
                key.clone(),
                MatInfo {
                    color: vec![fl(1.0), fl(1.0), fl(1.0)],
                    map: Some(name),
                    src: Some(path),
                    opacity: op,
                    mat: None,
                    finish: String::new(),
                },
            ));
        }
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
        (key, Some((sw, sh, rot, uvw)))
    } else {
        let color: Vec<Value> = match face.attr("color") {
            Some(Value::Array(c)) if !c.is_empty() => c.iter().take(3).cloned().collect(),
            _ => DEFAULT_COLOR.iter().map(|&x| fl(x)).collect(),
        };
        let mut bits = Vec::new();
        for c in &color {
            bits.push(fkey(pyval::float(c).map_err(|e| format!("{}: {}", e.kind, e.msg))?));
        }
        let key = MKey::Color(bits, opk);
        if !reg.index.contains_key(&key) {
            reg.index.insert(key.clone(), reg.list.len());
            reg.list.push((key.clone(), MatInfo { color, map: None, src: None, opacity: op, mat: None, finish: String::new() }));
        }
        (key, None)
    };
    if let Some(m) = face.attr("mat").filter(|m| pyval::truthy(m)) {
        let info = &mut reg.list[reg.index[&key]].1;
        if info.mat.is_none() {
            info.mat = Some(m.clone());
        }
    }
    for tri in tris {
        let uv = uv_rule.as_ref().map(|(sw, sh, rot, uvw)| match uvw {
            Some(w) => affine_uv(w, &tri),
            None => planar_uv(n, &tri, *sw, *sh, *rot),
        });
        let verts = [0, 1, 2].map(|k| (tri[k], uv.as_ref().map(|u| u[k])));
        prims.push(&key, Tri { normal: n, verts });
    }
    Ok(())
}

fn add_faces(faces: &[WFace], reg: &mut Registry, prims: &mut Prims, h: Hypot) -> Result<(), String> {
    for f in faces {
        add_face(f, reg, prims, h)?;
    }
    Ok(())
}

/// `_resolve_finishes(scene, materials)`.
fn resolve_finishes(scene: &Scene, reg: &mut Registry) {
    for (_, info) in &mut reg.list {
        let name = match &info.mat {
            Some(Value::Str(n)) => Some(n.as_str()),
            _ => None,
        };
        let chosen = name.and_then(|n| scene.materials.iter().find(|(k, _)| k == n)).and_then(|(_, m)| match m.get("finish") {
            Some(Value::Str(f)) => Some(f.as_str()),
            _ => None,
        });
        info.finish = finish::resolve(chosen, name, info.map.as_deref(), info.opacity);
    }
}

/// `export_names(materials)`: a unique, export-safe name per material.
fn export_names(reg: &Registry) -> Vec<String> {
    let mut used: std::collections::HashSet<String> = Default::default();
    let mut out = Vec::new();
    for (i, (_, info)) in reg.list.iter().enumerate() {
        let raw = match &info.mat {
            Some(Value::Str(t)) => t.trim_matches(pyutil::is_space).to_string(),
            _ => String::new(),
        };
        let base = if raw.is_empty() {
            format!("mat{i}")
        } else {
            let cleaned: String =
                raw.chars().map(|c| if ingetrazo_model::pyalnum::is_alnum(c) || matches!(c, '.' | '_' | '-') { c } else { '_' }).collect();
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

// ---- the model as a tree --------------------------------------------------------

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
enum MeshKey {
    Loose,
    Mesh(MeshId),
    Figure(usize),
}

struct Node {
    name: String,
    matrix: Option<M4>,
    mesh: Option<MeshKey>,
    children: Vec<Node>,
    extras: Option<Value>,
}

/// Where a face-me figure looks in a file with no camera: toward the Front
/// view.
const FILE_FACING: V3 = V3(0.0, -1.0, 0.0);

struct Tree<'a> {
    scene: &'a Scene,
    h: Hypot,
    labels: &'a Labels,
    reg: Registry,
    meshes: Vec<(MeshKey, Prims)>,
    figures: Vec<Node>,
    n_figures: usize,
}

impl Tree<'_> {
    fn part(&mut self, g: &Group, base: Option<M4>) -> Result<Option<Node>, String> {
        let m = g.xform;
        let world = match (base, m) {
            (None, m) => m,
            (Some(b), None) => Some(b),
            (Some(b), Some(m)) => Some(b.mul(&m)),
        };
        let name = if g.name.is_empty() { None } else { Some(g.name.clone()) };
        if billboard(g) {
            let scene = self.scene;
            let src = scene.mesh(g.mesh);
            let owned = world.map(|w| transformed_mesh(src, &w));
            let mesh: &Mesh = owned.as_ref().unwrap_or(src);
            let Some(feet) = feet(mesh) else { return Ok(None) };
            let of_mesh = matches!(&g.billboard, Some(Value::Str(k)) if k == "mesh");
            let turned = if of_mesh { turned_toward(mesh, FILE_FACING, self.h) } else { facing_quad(mesh, FILE_FACING, self.h) };
            let turned_ref: &Mesh = turned.as_ref().unwrap_or(mesh);
            let mut prims = Prims::default();
            add_faces(&mesh_faces(turned_ref), &mut self.reg, &mut prims, self.h)?;
            if !prims.is_empty() {
                let key = MeshKey::Figure(self.n_figures);
                self.n_figures += 1;
                self.meshes.push((key.clone(), prims));
                let yaw = (FILE_FACING.1 as f64).atan2(FILE_FACING.0 as f64);
                self.figures.push(Node {
                    name: name.unwrap_or_else(|| self.labels.figure.clone()),
                    matrix: None,
                    mesh: Some(key),
                    children: Vec::new(),
                    extras: Some(obj(vec![(
                        "ingetrazo_faceme",
                        Value::Array(vec![fl(feet.0 as f64), fl(feet.1 as f64), fl(feet.2 as f64), fl(yaw)]),
                    )])),
                });
            }
            return Ok(None);
        }
        let key = MeshKey::Mesh(g.mesh);
        let at = match self.meshes.iter().position(|(k, _)| *k == key) {
            Some(i) => i,
            None => {
                let mut prims = Prims::default();
                add_faces(&mesh_faces(self.scene.mesh(g.mesh)), &mut self.reg, &mut prims, self.h)?;
                self.meshes.push((key.clone(), prims));
                self.meshes.len() - 1
            }
        };
        let has_mesh = !self.meshes[at].1.is_empty();
        let mut node = Node {
            name: name.unwrap_or_else(|| self.labels.group.clone()),
            matrix: m,
            mesh: if has_mesh { Some(key) } else { None },
            children: Vec::new(),
            extras: None,
        };
        for child in &g.children {
            if let Some(sub) = self.part(child, world)? {
                node.children.push(sub);
            }
        }
        Ok(Some(node))
    }
}

/// `_file_hierarchy(scene)`: `(materials, meshes, roots)`.
fn file_hierarchy(scene: &Scene, h: Hypot, labels: &Labels) -> Result<(Registry, Vec<(MeshKey, Prims)>, Vec<Node>), String> {
    let mut t = Tree { scene, h, labels, reg: Registry::default(), meshes: Vec::new(), figures: Vec::new(), n_figures: 0 };
    let mut roots = Vec::new();
    let loose: Vec<WFace> = mesh_faces(scene.loose()).into_iter().filter(|f| face_visible(scene, f)).collect();
    if !loose.is_empty() {
        let mut prims = Prims::default();
        add_faces(&loose, &mut t.reg, &mut prims, h)?;
        t.meshes.push((MeshKey::Loose, prims));
        roots.push(Node { name: labels.loose.clone(), matrix: None, mesh: Some(MeshKey::Loose), children: Vec::new(), extras: None });
    }
    for g in &scene.groups {
        if !group_visible(scene, g) {
            continue;
        }
        if let Some(node) = t.part(g, None)? {
            roots.push(node);
        }
    }
    roots.append(&mut t.figures);
    resolve_finishes(scene, &mut t.reg);
    Ok((t.reg, t.meshes, roots))
}

/// `_mesh_name(key, tree)`: the name of the first node that draws it, looking
/// level by level.
fn mesh_name(key: &MeshKey, tree: &[Node]) -> String {
    let mut queue: std::collections::VecDeque<&Node> = tree.iter().collect();
    while let Some(part) = queue.pop_front() {
        if part.mesh.as_ref() == Some(key) {
            return part.name.clone();
        }
        queue.extend(part.children.iter());
    }
    "Mesh".to_string()
}

/// `_yup_matrix(m)`: a Z-up placement as the matching glTF node matrix.
fn yup_matrix(m: &M4) -> Value {
    let c = M4::from_rows([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]);
    let out = c.mul(m).mul(&c.transposed());
    Value::Array(out.data().iter().map(|&x| fl(x as f64)).collect())
}

/// `_mime(name)`.
fn mime(name: &str) -> &'static str {
    let ext = match name.rfind('.') {
        Some(i) if i > 0 => name[i..].to_lowercase(),
        _ => String::new(),
    };
    if ext == ".jpg" || ext == ".jpeg" {
        "image/jpeg"
    } else {
        "image/png"
    }
}

/// `str(Path(p))`: the text pathlib shows for a path.
fn path_str(p: &str) -> String {
    if cfg!(windows) {
        let norm = p.replace('/', "\\");
        let unc = norm.starts_with("\\\\") && !norm.starts_with("\\\\\\");
        let parts: Vec<&str> = norm.split('\\').filter(|x| !x.is_empty() && *x != ".").collect();
        let mut out = parts.join("\\");
        if norm.starts_with('\\') && !unc && !parts.is_empty() && !out.starts_with('\\') && !parts[0].ends_with(':') {
            out.insert(0, '\\');
        }
        if unc {
            out.insert_str(0, "\\\\");
        }
        if out.is_empty() {
            ".".to_string()
        } else {
            out
        }
    } else {
        let parts: Vec<&str> = p.split('/').filter(|x| !x.is_empty() && *x != ".").collect();
        let out = parts.join("/");
        let out = if p.starts_with("//") && !p.starts_with("///") {
            format!("//{out}")
        } else if p.starts_with('/') {
            format!("/{out}")
        } else {
            out
        };
        if out.is_empty() {
            ".".to_string()
        } else {
            out
        }
    }
}

// ---- the file ---------------------------------------------------------------------

/// The one binary buffer and what points into it.
#[derive(Default)]
struct Binary {
    buf: Vec<u8>,
    views: Vec<Value>,
    accessors: Vec<Value>,
}

impl Binary {
    fn align(&mut self) {
        while !self.buf.len().is_multiple_of(4) {
            self.buf.push(0);
        }
    }

    fn accessor(&mut self, data: &[u8], comp: i64, count: usize, kind: &str, target: i64, minmax: Option<(Vec<Value>, Vec<Value>)>) -> i64 {
        self.align();
        let mut view = obj(vec![("buffer", int(0)), ("byteOffset", int(self.buf.len() as i64)), ("byteLength", int(data.len() as i64))]);
        if target != 0 {
            set(&mut view, "target", int(target));
        }
        self.buf.extend_from_slice(data);
        self.views.push(view);
        let mut acc = obj(vec![
            ("bufferView", int(self.views.len() as i64 - 1)),
            ("componentType", int(comp)),
            ("count", int(count as i64)),
            ("type", s(kind)),
        ]);
        if let Some((mn, mx)) = minmax {
            set(&mut acc, "min", Value::Array(mn));
            set(&mut acc, "max", Value::Array(mx));
        }
        self.accessors.push(acc);
        self.accessors.len() as i64 - 1
    }
}

/// Python's `min(xs)` / `max(xs)` over floats: the first, replaced on `<`/`>`.
fn py_min(xs: &[f64]) -> f64 {
    xs.iter().skip(1).fold(xs[0], |a, &x| if x < a { x } else { a })
}

fn py_max(xs: &[f64]) -> f64 {
    xs.iter().skip(1).fold(xs[0], |a, &x| if x > a { x } else { a })
}

fn yup(v: V3) -> V3 {
    V3(v.0, v.2, -v.1)
}

/// `geolocation(scene)`: `(lat, lon, alt)` of the georef datum, if any.
fn geolocation(scene: &Scene) -> Option<(f64, f64, f64)> {
    let d = scene.extra.georef_datum.as_ref()?;
    let lat = pyval::float(d.get("lat")?).ok()?;
    let lon = pyval::float(d.get("lon")?).ok()?;
    let alt = d.get("alt").map(|a| pyval::float(a).unwrap_or(0.0)).unwrap_or(0.0);
    Some((lat, lon, alt))
}

/// `save_glb(scene, path)` into memory: the bytes of the `.glb`.
pub fn save_glb_bytes(scene: &Scene, h: Hypot, labels: &Labels) -> Result<Vec<u8>, String> {
    let (reg, meshes, tree) = file_hierarchy(scene, h, labels)?;
    let mut bin = Binary::default();

    // ---- materials (and the embedded images) ----
    let mut keys: Vec<MKey> = Vec::new();
    for (_, source) in &meshes {
        for k in &source.keys {
            if !keys.contains(k) {
                keys.push(k.clone());
            }
        }
    }
    let names = export_names(&reg);
    let mut gltf_materials: Vec<Value> = Vec::new();
    let mut images: Vec<Value> = Vec::new();
    let mut textures: Vec<Value> = Vec::new();
    let mut samplers: Vec<Value> = Vec::new();
    let mut mat_index: HashMap<MKey, usize> = HashMap::new();
    let mut tex_for_image: HashMap<String, usize> = HashMap::new();
    let mut used_ext: Vec<String> = Vec::new();
    for key in &keys {
        let at = reg.index[key];
        let info = &reg.list[at].1;
        let mut mat = obj(vec![
            ("name", s(&names[at])),
            ("doubleSided", Value::Bool(true)),
            ("pbrMetallicRoughness", obj(vec![("metallicFactor", fl(0.0)), ("roughnessFactor", fl(1.0))])),
        ]);
        if let Some(map) = &info.map {
            let src = info.src.as_deref().unwrap_or("");
            let skey = path_str(src);
            let mut tex_idx = tex_for_image.get(&skey).copied();
            if tex_idx.is_none() {
                let img = std::fs::read(Path::new(src)).ok().filter(|b| is_image_bytes(b));
                if let Some(img) = img {
                    bin.align();
                    let view = obj(vec![("buffer", int(0)), ("byteOffset", int(bin.buf.len() as i64)), ("byteLength", int(img.len() as i64))]);
                    bin.buf.extend_from_slice(&img);
                    bin.views.push(view);
                    images.push(obj(vec![("bufferView", int(bin.views.len() as i64 - 1)), ("mimeType", s(mime(map))), ("name", s(map))]));
                    if samplers.is_empty() {
                        samplers.push(obj(vec![("wrapS", int(10497)), ("wrapT", int(10497))]));
                    }
                    textures.push(obj(vec![("source", int(images.len() as i64 - 1)), ("sampler", int(0))]));
                    tex_for_image.insert(skey.clone(), textures.len() - 1);
                    tex_idx = Some(textures.len() - 1);
                }
            }
            if let Some(ti) = tex_idx {
                set(entry(&mut mat, "pbrMetallicRoughness").unwrap(), "baseColorTexture", obj(vec![("index", int(ti as i64))]));
                // A cut-out image stays cut out.
                if std::fs::read(Path::new(src)).map(|b| image_has_cutout(&b)).unwrap_or(false) {
                    set(&mut mat, "alphaMode", s("MASK"));
                    set(&mut mat, "alphaCutoff", fl(0.5));
                }
            } else {
                // The image is unreadable: white.
                set(entry(&mut mat, "pbrMetallicRoughness").unwrap(), "baseColorFactor", Value::Array(vec![int(1), int(1), int(1), int(1)]));
            }
        } else {
            if info.color.len() < 3 {
                return Err(format!("ValueError: not enough values to unpack (expected 3, got {})", info.color.len()));
            }
            let (r, g, b) = (info.color[0].clone(), info.color[1].clone(), info.color[2].clone());
            set(entry(&mut mat, "pbrMetallicRoughness").unwrap(), "baseColorFactor", Value::Array(vec![r, g, b, fl(1.0)]));
        }
        // The finish: plain glTF for every program, and the name in extras.
        if let Some((rough, metal, trans, ior)) = finish::pbr(&info.finish) {
            let pbr = entry(&mut mat, "pbrMetallicRoughness").unwrap();
            set(pbr, "roughnessFactor", fl(rough));
            set(pbr, "metallicFactor", fl(metal));
            set(&mut mat, "extras", obj(vec![("ingetrazo_finish", s(&info.finish))]));
            let mut ext: Vec<(&str, Value)> = Vec::new();
            if trans != 0.0 {
                ext.push(("KHR_materials_transmission", obj(vec![("transmissionFactor", fl(trans))])));
            }
            if ior != 1.5 {
                ext.push(("KHR_materials_ior", obj(vec![("ior", fl(ior))])));
            }
            if !ext.is_empty() {
                for (k, _) in &ext {
                    if !used_ext.iter().any(|u| u == k) {
                        used_ext.push(k.to_string());
                    }
                }
                set(&mut mat, "extensions", obj(ext));
            }
        }
        if let Some(op) = info.opacity {
            let pbr = entry(&mut mat, "pbrMetallicRoughness").unwrap();
            if entry(pbr, "baseColorFactor").is_none() {
                set(pbr, "baseColorFactor", Value::Array(vec![fl(1.0), fl(1.0), fl(1.0), fl(1.0)]));
            }
            if let Some(Value::Array(f)) = entry(pbr, "baseColorFactor") {
                f[3] = fl(op);
            }
            set(&mut mat, "alphaMode", s("BLEND"));
            pop(&mut mat, "alphaCutoff");
        }
        mat_index.insert(key.clone(), gltf_materials.len());
        gltf_materials.push(mat);
    }

    // ---- geometry: one primitive per material ----
    let primitives_of = |source: &Prims, bin: &mut Binary| -> Vec<Value> {
        let mut primitives = Vec::new();
        for key in &keys {
            let Some(tris) = source.tris.get(key) else { continue };
            let textured = key.is_tex();
            let mut positions: Vec<V3> = Vec::new();
            let mut normals: Vec<V3> = Vec::new();
            let mut uvs: Vec<(f64, f64)> = Vec::new();
            let mut indices: Vec<u32> = Vec::new();
            type VKey = [(bool, i64); 3];
            let mut vindex: HashMap<(VKey, [u64; 3], (u64, u64)), u32> = HashMap::new();
            for tri in tris {
                let nn = yup(tri.normal);
                for (pos, uv) in &tri.verts {
                    let p = yup(*pos);
                    let uvk = match (textured, uv) {
                        (true, Some((u, v))) => (fkey(pyutil::round6(*u)), fkey(pyutil::round6(*v))),
                        _ => (fkey(0.0), fkey(0.0)),
                    };
                    let vkey = (
                        [round_key(p.0 as f64), round_key(p.1 as f64), round_key(p.2 as f64)],
                        [fkey(pyutil::round4(nn.0 as f64)), fkey(pyutil::round4(nn.1 as f64)), fkey(pyutil::round4(nn.2 as f64))],
                        uvk,
                    );
                    let idx = *vindex.entry(vkey).or_insert_with(|| {
                        positions.push(p);
                        normals.push(nn);
                        if textured {
                            // glTF's texcoord origin is top-left: flip V.
                            let (u, v) = uv.unwrap_or((0.0, 0.0));
                            uvs.push((u, 1.0 - v));
                        }
                        positions.len() as u32 - 1
                    });
                    indices.push(idx);
                }
            }
            if positions.is_empty() {
                continue;
            }
            let mut pos_bytes = Vec::with_capacity(positions.len() * 12);
            for p in &positions {
                for c in [p.0, p.1, p.2] {
                    pos_bytes.extend_from_slice(&c.to_le_bytes());
                }
            }
            let col = |k: usize| -> Vec<f64> { positions.iter().map(|p| [p.0, p.1, p.2][k] as f64).collect() };
            let (xs, ys, zs) = (col(0), col(1), col(2));
            let pos_acc = bin.accessor(
                &pos_bytes,
                F32,
                positions.len(),
                "VEC3",
                ARRAY_BUFFER,
                Some((
                    vec![fl(py_min(&xs)), fl(py_min(&ys)), fl(py_min(&zs))],
                    vec![fl(py_max(&xs)), fl(py_max(&ys)), fl(py_max(&zs))],
                )),
            );
            let mut nrm_bytes = Vec::with_capacity(normals.len() * 12);
            for n in &normals {
                for c in [n.0, n.1, n.2] {
                    nrm_bytes.extend_from_slice(&c.to_le_bytes());
                }
            }
            let nrm_acc = bin.accessor(&nrm_bytes, F32, normals.len(), "VEC3", ARRAY_BUFFER, None);
            let mut attrs = vec![("POSITION", int(pos_acc)), ("NORMAL", int(nrm_acc))];
            if textured && !uvs.is_empty() {
                let mut uv_bytes = Vec::with_capacity(uvs.len() * 8);
                for (u, v) in &uvs {
                    uv_bytes.extend_from_slice(&(*u as f32).to_le_bytes());
                    uv_bytes.extend_from_slice(&(*v as f32).to_le_bytes());
                }
                attrs.push(("TEXCOORD_0", int(bin.accessor(&uv_bytes, F32, uvs.len(), "VEC2", ARRAY_BUFFER, None))));
            }
            let mut idx_bytes = Vec::with_capacity(indices.len() * 4);
            for i in &indices {
                idx_bytes.extend_from_slice(&i.to_le_bytes());
            }
            let idx_acc = bin.accessor(&idx_bytes, U32, indices.len(), "SCALAR", ELEMENT_ARRAY_BUFFER, None);
            primitives.push(obj(vec![
                ("attributes", obj(attrs)),
                ("indices", int(idx_acc)),
                ("material", int(mat_index[key] as i64)),
                ("mode", int(4)),
            ]));
        }
        primitives
    };

    // ---- assemble the glTF JSON ----
    let mut gl_meshes: Vec<Value> = Vec::new();
    let mut mesh_index: HashMap<MeshKey, usize> = HashMap::new();
    for (key, source) in &meshes {
        let prim_list = primitives_of(source, &mut bin);
        if !prim_list.is_empty() {
            mesh_index.insert(key.clone(), gl_meshes.len());
            gl_meshes.push(obj(vec![("name", s(&mesh_name(key, &tree))), ("primitives", Value::Array(prim_list))]));
        }
    }
    let mut extras = vec![("generator", s("IngeTrazo"))];
    if let Some((lat, lon, alt)) = geolocation(scene) {
        extras.push(("ingetrazo_geolocation", obj(vec![("lat", fl(lat)), ("lon", fl(lon)), ("alt", fl(alt))])));
    }
    let mut gltf = obj(vec![("asset", obj(vec![("version", s("2.0")), ("generator", s("IngeTrazo")), ("extras", obj(extras))]))]);

    fn emit(part: &Node, nodes: &mut Vec<Value>, mesh_index: &HashMap<MeshKey, usize>) -> i64 {
        let mut node = obj(vec![("name", s(&part.name))]);
        if let Some(mi) = part.mesh.as_ref().and_then(|k| mesh_index.get(k)) {
            set(&mut node, "mesh", int(*mi as i64));
        }
        if let Some(m) = part.matrix.as_ref().filter(|m| !m.is_identity()) {
            set(&mut node, "matrix", yup_matrix(m));
        }
        if let Some(e) = &part.extras {
            set(&mut node, "extras", e.clone());
        }
        nodes.push(node);
        let idx = nodes.len() - 1;
        let kids: Vec<Value> = part.children.iter().map(|c| int(emit(c, nodes, mesh_index))).collect();
        if !kids.is_empty() {
            set(&mut nodes[idx], "children", Value::Array(kids));
        }
        idx as i64
    }
    let mut nodes: Vec<Value> = Vec::new();
    let roots: Vec<Value> = tree.iter().map(|p| int(emit(p, &mut nodes, &mesh_index))).collect();
    if !gl_meshes.is_empty() {
        set(&mut gltf, "meshes", Value::Array(gl_meshes));
    }
    if !nodes.is_empty() {
        set(&mut gltf, "nodes", Value::Array(nodes));
    }
    set(&mut gltf, "scenes", Value::Array(vec![obj(vec![("nodes", Value::Array(roots))])]));
    set(&mut gltf, "scene", int(0));
    if !bin.accessors.is_empty() {
        set(&mut gltf, "accessors", Value::Array(std::mem::take(&mut bin.accessors)));
        set(&mut gltf, "bufferViews", Value::Array(std::mem::take(&mut bin.views)));
        set(&mut gltf, "buffers", Value::Array(vec![obj(vec![("byteLength", int(bin.buf.len() as i64))])]));
    }
    if !gltf_materials.is_empty() {
        set(&mut gltf, "materials", Value::Array(gltf_materials));
    }
    if !used_ext.is_empty() {
        used_ext.sort();
        set(&mut gltf, "extensionsUsed", Value::Array(used_ext.iter().map(|e| s(e)).collect()));
    }
    if !images.is_empty() {
        set(&mut gltf, "images", Value::Array(images));
        set(&mut gltf, "textures", Value::Array(textures));
        set(&mut gltf, "samplers", Value::Array(samplers));
    }

    let mut json = dumps(&gltf, &Options::compact()).map_err(|e| e.to_string())?.into_bytes();
    json.resize(json.len() + (4 - json.len() % 4) % 4, b' ');
    let mut bin_bytes = bin.buf;
    bin_bytes.resize(bin_bytes.len() + (4 - bin_bytes.len() % 4) % 4, 0);
    let total = 12 + 8 + json.len() + if bin_bytes.is_empty() { 0 } else { 8 + bin_bytes.len() };
    let mut out = Vec::with_capacity(total);
    out.extend_from_slice(&0x46546C67u32.to_le_bytes());
    out.extend_from_slice(&2u32.to_le_bytes());
    out.extend_from_slice(&(total as u32).to_le_bytes());
    out.extend_from_slice(&(json.len() as u32).to_le_bytes());
    out.extend_from_slice(&0x4E4F534Au32.to_le_bytes());
    out.extend_from_slice(&json);
    if !bin_bytes.is_empty() {
        out.extend_from_slice(&(bin_bytes.len() as u32).to_le_bytes());
        out.extend_from_slice(&0x004E4942u32.to_le_bytes());
        out.extend_from_slice(&bin_bytes);
    }
    Ok(out)
}

/// `save_glb(scene, path)`.
pub fn save_glb(scene: &Scene, path: &Path, h: Hypot, labels: &Labels) -> Result<(), String> {
    let bytes = save_glb_bytes(scene, h, labels)?;
    std::fs::write(path, bytes).map_err(|e| e.to_string())
}
