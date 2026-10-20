// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! Saving a [`Scene`] as an .igz, byte for byte as `formats/igz.py`
//! `save_scene` writes it: the same JSON (key order, optional blocks and
//! the conditions that write them), the same prototype table, the same
//! format 1 / 2 / 3 choice, textures packed content-addressed, geometry
//! blocks for big meshes, and the same ZIP.
//!
//! What the model does not hold yet stays the JSON the document brought
//! ([`Scene::extra`], the `Value` blocks of [`Scene`]); the typed models
//! `Layer` and `Material` normalise their blocks as their `to_dict` does.
//!
//! Known limits, each of them only for an input Python itself would take
//! another path for:
//! - a face with two or more holes in a format-3 geometry block: Python
//!   drops holes nested inside a bigger one (`topology._maximal_holes`,
//!   not ported); [`SaveOptions::maximal_holes`] is where that plugs in,
//!   and without it every hole is kept;
//! - lazy meshes (`core.lazy_mesh`, the experimental engine) have no
//!   counterpart: every mesh here is built.

use std::collections::HashMap;

use ingetrazo_formats::igz::{self, IgzError, CONTAINER_FORMAT, GEOMETRY_FORMAT, LARGE_DOCUMENT, PLAIN_FORMAT};
use ingetrazo_formats::igz_zip;
use ingetrazo_formats::pyjson::{self, Options, Value};
use ingetrazo_math::{M4, V3};

use ingetrazo_georef::geopath::GeoPath;
use ingetrazo_georef::points::GeoPoint;

use crate::guide::Guide;
use crate::image_plane::ImagePlane;
use crate::layers::Layer;
use crate::saved_views::SavedView;
use crate::section::SectionPlane;
use crate::textlabel::TextLabel;
use crate::materials::Material;
use crate::mesh::{Mesh, VertexId};
use crate::pyval::truthy;
use crate::scene::{Group, MeshId, Scene};
use crate::texture::{is_image_bytes, texture_file_name};

/// `igz._BULK_MIN`.
pub const BULK_MIN: usize = 1024;
const TEX_PREFIX: &str = "textures/";
const GEOM_PREFIX: &str = "geom/";
/// `igz._TEXTURE_WALK_SKIP`.
const TEXTURE_WALK_SKIP: [&str; 6] = ["vertices", "holes", "a", "b", "edges", "arrays"];

/// Reads a texture source: the file at a `"path"` (or an archive member
/// when a document is saved again from what it was opened as).
pub type TextureReader<'a> = &'a dyn Fn(&str) -> Option<Vec<u8>>;

/// `_maximal_holes`: of the holes of one face (as positions), the indices
/// to keep, in order.
pub type MaximalHoles<'a> = &'a dyn Fn(&[Vec<V3>]) -> Vec<usize>;

pub struct SaveOptions<'a> {
    /// `core.version.__version__`, written as `app_version`.
    pub app_version: &'a str,
    pub textures: TextureReader<'a>,
    /// `igz._LARGE_DOCUMENT`.
    pub large_document: usize,
    /// `igz._BULK_MIN`.
    pub bulk_min: usize,
    pub maximal_holes: Option<MaximalHoles<'a>>,
}

impl<'a> SaveOptions<'a> {
    pub fn new(app_version: &'a str, textures: TextureReader<'a>) -> SaveOptions<'a> {
        SaveOptions { app_version, textures, large_document: LARGE_DOCUMENT, bulk_min: BULK_MIN, maximal_holes: None }
    }
}

/// The result of a save: the file's bytes and what `save_scene` returns.
#[derive(Debug)]
pub struct Saved {
    pub bytes: Vec<u8>,
    /// `{"embedded", "missing"}`: images packed / unreadable.
    pub embedded: usize,
    pub missing: usize,
    /// The `igz_format` written.
    pub format: i64,
}

/// The document before it is serialized (for tests that compare parts).
pub struct Built {
    pub doc: Value,
    pub large: bool,
    /// Textures (and photo-mesh blobs): STORED members.
    pub blobs: Vec<(String, Vec<u8>)>,
    /// Geometry blocks: deflated members.
    pub geom: Vec<(String, Vec<u8>)>,
    pub missing: usize,
}

fn s(v: &str) -> Value {
    Value::Str(v.to_string())
}

fn float(x: f64) -> Value {
    Value::Float(x)
}

fn point(p: V3) -> Value {
    Value::Array(vec![float(f64::from(p.0)), float(f64::from(p.1)), float(f64::from(p.2))])
}

fn matrix(m: &M4) -> Value {
    Value::Array(m.data().iter().map(|&x| float(f64::from(x))).collect())
}

fn obj(entries: Vec<(&str, Value)>) -> Value {
    Value::Object(entries.into_iter().map(|(k, v)| (k.to_string(), v)).collect())
}

// ---- SHA-1 (the texture digest) ------------------------------------------------

/// `hashlib.sha1(data).hexdigest()`.
pub fn sha1_hex(data: &[u8]) -> String {
    let mut h: [u32; 5] = [0x6745_2301, 0xEFCD_AB89, 0x98BA_DCFE, 0x1032_5476, 0xC3D2_E1F0];
    let mut msg = data.to_vec();
    msg.push(0x80);
    while msg.len() % 64 != 56 {
        msg.push(0);
    }
    msg.extend_from_slice(&((data.len() as u64) * 8).to_be_bytes());
    for block in msg.as_chunks::<64>().0 {
        let mut w = [0u32; 80];
        for (i, c) in block.as_chunks::<4>().0.iter().enumerate() {
            w[i] = u32::from_be_bytes([c[0], c[1], c[2], c[3]]);
        }
        for i in 16..80 {
            w[i] = (w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16]).rotate_left(1);
        }
        let [mut a, mut b, mut c, mut d, mut e] = h;
        for (i, wi) in w.iter().enumerate() {
            let (f, k) = match i {
                0..=19 => ((b & c) | (!b & d), 0x5A82_7999u32),
                20..=39 => (b ^ c ^ d, 0x6ED9_EBA1),
                40..=59 => ((b & c) | (b & d) | (c & d), 0x8F1B_BCDC),
                _ => (b ^ c ^ d, 0xCA62_C1D6),
            };
            let t = a.rotate_left(5).wrapping_add(f).wrapping_add(e).wrapping_add(k).wrapping_add(*wi);
            e = d;
            d = c;
            c = b.rotate_left(30);
            b = a;
            a = t;
        }
        h = [h[0].wrapping_add(a), h[1].wrapping_add(b), h[2].wrapping_add(c), h[3].wrapping_add(d), h[4].wrapping_add(e)];
    }
    h.iter().map(|x| format!("{x:08x}")).collect()
}

// ---- meshes --------------------------------------------------------------------

/// `igz._face_attrs_json`: everything a face stores besides its loops.
pub fn face_attrs_json(attrs: &[(String, Value)]) -> Value {
    let get = |k: &str| attrs.iter().rev().find(|(n, _)| n == k).map(|(_, v)| v);
    let some = |k: &str| get(k).filter(|v| !matches!(v, Value::Null));
    let mut entry: Vec<(String, Value)> = Vec::new();
    if let Some(c) = some("color") {
        entry.push(("color".into(), c.clone()));
    }
    if let Some(t) = some("texture") {
        entry.push(("texture".into(), t.clone()));
    }
    if let Some(l) = some("layer") {
        entry.push(("layer".into(), l.clone()));
    }
    if let Some(i) = some("ifc") {
        entry.push(("ifc".into(), i.clone()));
    }
    if let Some(o) = some("opacity") {
        let x = match o {
            Value::Float(x) => *x,
            Value::Int(n) => n.parse().unwrap_or(0.0),
            Value::Bool(b) => f64::from(u8::from(*b)),
            _ => 0.0,
        };
        entry.push(("opacity".into(), float(x)));
    }
    if get("hidden").is_some_and(truthy) {
        entry.push(("hidden".into(), Value::Bool(true)));
    }
    if let Some(m) = get("mat").filter(|v| truthy(v)) {
        entry.push(("mat".into(), m.clone()));
    }
    match get("back") {
        Some(b @ Value::Object(_)) => entry.push(("back".into(), b.clone())),
        Some(Value::Bool(true)) => entry.push(("back".into(), Value::Bool(true))),
        _ => {}
    }
    Value::Object(entry)
}

fn ring_json(mesh: &Mesh, ring: &[VertexId]) -> Value {
    Value::Array(ring.iter().map(|&v| point(mesh.position(v))).collect())
}

fn mesh_lists_json(mesh: &Mesh) -> Value {
    let edges = mesh
        .edges
        .iter()
        .map(|&e| {
            let ed = mesh.edge(e);
            let mut entry = vec![("a".to_string(), point(mesh.position(ed.v0))), ("b".to_string(), point(mesh.position(ed.v1)))];
            if ed.soft {
                entry.push(("soft".into(), Value::Bool(true)));
            }
            if ed.hidden {
                entry.push(("hidden".into(), Value::Bool(true)));
            }
            if let Some(c) = ed.curve {
                entry.push(("curve".into(), Value::int(c)));
            }
            if let Some(l) = &ed.layer {
                entry.push(("layer".into(), s(l)));
            }
            Value::Object(entry)
        })
        .collect();
    let faces = mesh
        .faces
        .iter()
        .map(|&f| {
            let face = mesh.face(f);
            let mut entry = vec![("vertices".to_string(), ring_json(mesh, &face.outer))];
            if !face.holes.is_empty() {
                entry.push(("holes".into(), Value::Array(face.holes.iter().map(|h| ring_json(mesh, h)).collect())));
            }
            if let Value::Object(more) = face_attrs_json(&face.attrs) {
                entry.extend(more);
            }
            Value::Object(entry)
        })
        .collect();
    obj(vec![("edges", Value::Array(edges)), ("faces", Value::Array(faces))])
}

type Sink = Vec<(String, Vec<u8>)>;

/// `igz._mesh_json`. With a `sink` (a large document) a bulk-sized mesh goes
/// to a binary block and the JSON keeps only its reference.
fn mesh_json(mesh: &Mesh, sink: &mut Option<Sink>, opts: &SaveOptions) -> Result<Value, IgzError> {
    match sink {
        Some(sink) if bulk_sized(mesh.edges.len(), mesh.faces.len(), opts.bulk_min) => {
            Ok(obj(vec![("geom", pack_geometry(mesh, sink, opts)?)]))
        }
        _ => Ok(mesh_lists_json(mesh)),
    }
}

/// `igz._bulk_sized`.
fn bulk_sized(n_edges: usize, n_faces: usize, bulk_min: usize) -> bool {
    n_edges + n_faces * 4 >= bulk_min
}

/// One array of a geometry blob.
struct Arr {
    dtype: &'static str,
    shape: Vec<usize>,
    bytes: Vec<u8>,
}

fn arr_i32(shape: Vec<usize>, xs: &[i32]) -> Arr {
    Arr { dtype: "<i4", shape, bytes: xs.iter().flat_map(|x| x.to_le_bytes()).collect() }
}

/// `igz._pack_geometry`.
fn pack_geometry(mesh: &Mesh, sink: &mut Sink, opts: &SaveOptions) -> Result<Value, IgzError> {
    let mut corners: Vec<VertexId> = Vec::new();
    for &e in &mesh.edges {
        let ed = mesh.edge(e);
        corners.push(ed.v0);
        corners.push(ed.v1);
    }
    let mut ring_sizes: Vec<i32> = Vec::new();
    let mut ring_counts: Vec<i32> = Vec::new();
    let mut face_attr: Vec<i32> = Vec::new();
    let mut table: Vec<Value> = Vec::new();
    let mut table_index: HashMap<String, usize> = HashMap::new();
    let key_opts = Options::compact().sort_keys(true);
    for &f in &mesh.faces {
        let face = mesh.face(f);
        let mut holes: Vec<&Vec<VertexId>> = face.holes.iter().collect();
        if holes.len() > 1 {
            if let Some(maximal) = opts.maximal_holes {
                let pos: Vec<Vec<V3>> = holes.iter().map(|h| h.iter().map(|&v| mesh.position(v)).collect()).collect();
                let kept = maximal(&pos);
                holes = kept.into_iter().filter_map(|i| holes.get(i).copied()).collect();
            }
        }
        ring_counts.push(1 + holes.len() as i32);
        ring_sizes.push(face.outer.len() as i32);
        corners.extend(face.outer.iter().copied());
        for h in holes {
            ring_sizes.push(h.len() as i32);
            corners.extend(h.iter().copied());
        }
        let attrs = if face.attrs.is_empty() { None } else { Some(face_attrs_json(&face.attrs)) };
        match attrs {
            Some(a @ Value::Object(_)) if truthy(&a) => {
                let key = pyjson::dumps(&a, &key_opts).map_err(|_| IgzError::NonFinite { at: "geometry".into() })?;
                let k = *table_index.entry(key).or_insert_with(|| {
                    table.push(a);
                    table.len() - 1
                });
                face_attr.push(k as i32);
            }
            _ => face_attr.push(-1),
        }
    }

    // Number the vertex objects by first appearance.
    let mut rank: HashMap<VertexId, i32> = HashMap::new();
    let mut order: Vec<VertexId> = Vec::new();
    let mut index: Vec<i32> = Vec::with_capacity(corners.len());
    for &v in &corners {
        let next = order.len() as i32;
        let r = *rank.entry(v).or_insert_with(|| {
            order.push(v);
            next
        });
        index.push(r);
    }
    let mut pos_bytes: Vec<u8> = Vec::with_capacity(order.len() * 12);
    for &v in &order {
        let p = mesh.position(v);
        if !(p.0.is_finite() && p.1.is_finite() && p.2.is_finite()) {
            return Err(IgzError::NonFinite { at: "geometry".into() });
        }
        for c in [p.0, p.1, p.2] {
            pos_bytes.extend_from_slice(&c.to_le_bytes());
        }
    }
    let ne = mesh.edges.len();
    let mut arrays: Vec<(&str, Arr)> = vec![
        ("pos", Arr { dtype: "<f4", shape: vec![order.len(), 3], bytes: pos_bytes }),
        ("edges", arr_i32(vec![ne, 2], &index[..2 * ne])),
        ("rings", arr_i32(vec![index.len() - 2 * ne], &index[2 * ne..])),
        ("ring_sizes", arr_i32(vec![ring_sizes.len()], &ring_sizes)),
        ("ring_counts", arr_i32(vec![ring_counts.len()], &ring_counts)),
        ("face_attr", arr_i32(vec![face_attr.len()], &face_attr)),
    ];
    let mut flags = vec![0u8; ne];
    let mut curves: Option<Vec<i64>> = None;
    let mut layers: Option<Vec<i32>> = None;
    let mut layer_table: Vec<Value> = Vec::new();
    let mut layer_index: HashMap<String, usize> = HashMap::new();
    for (i, &e) in mesh.edges.iter().enumerate() {
        let ed = mesh.edge(e);
        if ed.soft {
            flags[i] |= 1;
        }
        if ed.hidden {
            flags[i] |= 2;
        }
        if let Some(c) = ed.curve {
            curves.get_or_insert_with(|| vec![-1; ne])[i] = c;
        }
        if let Some(l) = &ed.layer {
            let k = *layer_index.entry(l.clone()).or_insert_with(|| {
                layer_table.push(s(l));
                layer_table.len() - 1
            });
            layers.get_or_insert_with(|| vec![-1; ne])[i] = k as i32;
        }
    }
    arrays.push(("edge_flags", Arr { dtype: "|u1", shape: vec![ne], bytes: flags }));
    if let Some(c) = curves {
        arrays.push(("edge_curve", Arr { dtype: "<i8", shape: vec![ne], bytes: c.iter().flat_map(|x| x.to_le_bytes()).collect() }));
    }
    if let Some(l) = layers {
        arrays.push(("edge_layer", arr_i32(vec![ne], &l)));
    }
    Ok(write_geometry(arrays, table, layer_table, sink))
}

/// `igz._write_geometry`: one raw blob, each array at an 8-byte-aligned
/// offset; its JSON side.
fn write_geometry(arrays: Vec<(&str, Arr)>, table: Vec<Value>, layer_table: Vec<Value>, sink: &mut Sink) -> Value {
    let mut layout: Vec<(String, Value)> = Vec::new();
    let mut blob: Vec<u8> = Vec::new();
    for (name, arr) in arrays {
        let shape = Value::Array(arr.shape.iter().map(|&n| Value::int(n as i64)).collect());
        layout.push((name.to_string(), Value::Array(vec![s(arr.dtype), shape, Value::int(blob.len() as i64)])));
        blob.extend_from_slice(&arr.bytes);
        let pad = (8 - arr.bytes.len() % 8) % 8;
        blob.resize(blob.len() + pad, 0);
    }
    let member = format!("{GEOM_PREFIX}{}.bin", sink.len());
    sink.push((member.clone(), blob));
    let mut meta = vec![("member".to_string(), Value::Str(member)), ("arrays".to_string(), Value::Object(layout))];
    if !table.is_empty() {
        meta.push(("face_attrs".into(), Value::Array(table)));
    }
    if !layer_table.is_empty() {
        meta.push(("edge_layers".into(), Value::Array(layer_table)));
    }
    Value::Object(meta)
}

// ---- groups --------------------------------------------------------------------

struct Walk<'a> {
    scene: &'a Scene,
    opts: &'a SaveOptions<'a>,
    sink: Option<Sink>,
    proto_index: HashMap<MeshId, usize>,
    protos: Vec<Value>,
}

impl Walk<'_> {
    /// `_entry(g)`.
    fn entry(&mut self, g: &Group) -> Result<Value, IgzError> {
        let mut entry: Vec<(String, Value)> = Vec::new();
        let mesh_part = if let Some(xf) = &g.xform {
            let idx = match self.proto_index.get(&g.mesh) {
                Some(&i) => i,
                None => {
                    let i = self.protos.len();
                    self.proto_index.insert(g.mesh, i);
                    self.protos.push(Value::Null); // reserved
                    let scene = self.scene;
                    self.protos[i] = mesh_json(scene.mesh(g.mesh), &mut self.sink, self.opts)?;
                    i
                }
            };
            vec![("proto".to_string(), Value::int(idx as i64)), ("xform".to_string(), matrix(xf)), ("component".to_string(), Value::Bool(g.component))]
        } else {
            match mesh_json(self.scene.mesh(g.mesh), &mut self.sink, self.opts)? {
                Value::Object(items) => items,
                _ => Vec::new(),
            }
        };
        entry.extend(mesh_part);
        let mut put = |k: &str, v: Value| entry.push((k.to_string(), v));
        if !g.name.is_empty() {
            put("name", s(&g.name));
        }
        if let Some(a) = &g.axes {
            put("axes", matrix(a));
        }
        if let Some(l) = &g.layer {
            put("layer", s(l));
        }
        if let Some(i) = g.ifc.as_ref().filter(|v| truthy(v)) {
            put("ifc", i.clone());
        }
        if let Some(b) = g.billboard.as_ref().filter(|v| truthy(v)) {
            put("billboard", b.clone());
        }
        if let Some(m) = g.material.as_ref().filter(|v| truthy(v)) {
            put("material", m.clone());
        }
        if let Some(t) = g.text3d.as_ref().filter(|v| truthy(v)) {
            put("text3d", t.clone());
        }
        if let Some(e) = g.ext.as_ref().filter(|v| truthy(v)) {
            put("ext", e.clone());
        }
        if !g.uid.is_empty() {
            put("uid", s(&g.uid));
        }
        if g.hidden {
            put("hidden", Value::Bool(true));
        }
        if g.locked {
            put("locked", Value::Bool(true));
        }
        if let Some(x) = g.exploded.as_ref().filter(|v| truthy(v)) {
            put("exploded", x.clone());
        }
        if let Some(o) = g.explode_offset {
            put("explode_offset", Value::Array(o.iter().map(|&x| float(x)).collect()));
        }
        if !g.children.is_empty() {
            let mut kids = Vec::with_capacity(g.children.len());
            for c in &g.children {
                kids.push(self.entry(c)?);
            }
            entry.push(("children".to_string(), Value::Array(kids)));
        }
        Ok(Value::Object(entry))
    }
}

/// `igz._live_element_count`: faces + edges the document will hold; a
/// component prototype counts once however often it is placed.
pub fn live_element_count(scene: &Scene) -> usize {
    let count = |m: &Mesh| m.edges.len() + m.faces.len();
    let mut total = count(scene.loose());
    let mut seen: std::collections::HashSet<MeshId> = std::collections::HashSet::new();
    let mut stack: Vec<&Group> = scene.groups.iter().chain(scene.kept_components.iter()).collect();
    while let Some(g) = stack.pop() {
        if seen.insert(g.mesh) {
            total += count(scene.mesh(g.mesh));
        }
        stack.extend(g.children.iter());
    }
    total
}

// ---- textures ------------------------------------------------------------------

struct Packer<'a> {
    read: TextureReader<'a>,
    blobs: Vec<(String, Vec<u8>)>,
    resolved: HashMap<String, String>,
    missing: usize,
}

impl Packer<'_> {
    fn add_blob(&mut self, member: &str, data: Vec<u8>) {
        match self.blobs.iter_mut().find(|(n, _)| n == member) {
            Some(slot) => slot.1 = data,
            None => self.blobs.push((member.to_string(), data)),
        }
    }

    /// `_member_for`: the archive member for a source file ("" = unreadable).
    fn member_for(&mut self, src: &str) -> String {
        if let Some(m) = self.resolved.get(src) {
            return m.clone();
        }
        let data = (self.read)(src).filter(|d| is_image_bytes(d));
        let member = match data {
            None => {
                self.missing += 1;
                String::new()
            }
            Some(data) => {
                let digest = &sha1_hex(&data)[..16];
                let member = format!("{TEX_PREFIX}{digest}-{}", texture_file_name(src));
                self.add_blob(&member, data);
                member
            }
        };
        self.resolved.insert(src.to_string(), member.clone());
        member
    }

    /// A texture that already names its archive member (a scene opened from
    /// a document keeps the `"embed"` of blocks the model has not unpacked):
    /// the member's bytes ride along unchanged.
    fn keep_embedded(&mut self, member: &str) {
        if !self.blobs.iter().any(|(n, _)| n == member) {
            if let Some(data) = (self.read)(member) {
                self.add_blob(member, data);
            }
        }
    }

    fn texture(&mut self, tex: &mut Vec<(String, Value)>) {
        let text = |tex: &Vec<(String, Value)>, k: &str| match tex.iter().find(|(n, _)| n == k) {
            Some((_, Value::Str(x))) if !x.is_empty() => Some(x.clone()),
            _ => None,
        };
        if let Some(src) = text(tex, "path") {
            let member = self.member_for(&src);
            if !member.is_empty() {
                tex.retain(|(n, _)| n != "path");
                tex.retain(|(n, _)| n != "embed");
                tex.push(("embed".into(), Value::Str(member)));
            }
        } else if let Some(m) = text(tex, "embed") {
            self.keep_embedded(&m);
        }
        if let Some(base) = text(tex, "base") {
            let member = self.member_for(&base);
            if !member.is_empty() {
                tex.retain(|(n, _)| n != "base");
                tex.retain(|(n, _)| n != "base_embed");
                tex.push(("base_embed".into(), Value::Str(member)));
            }
        } else if let Some(m) = text(tex, "base_embed") {
            self.keep_embedded(&m);
        }
    }

    /// `_texture_entries` + `_pack_textures`' loop.
    fn walk(&mut self, node: &mut Value) {
        match node {
            Value::Object(items) => {
                for (k, v) in items.iter_mut() {
                    if k == "texture" {
                        if let Value::Object(tex) = v {
                            self.texture(tex);
                            continue;
                        }
                    }
                    if !TEXTURE_WALK_SKIP.contains(&k.as_str()) {
                        self.walk(v);
                    }
                }
            }
            Value::Array(items) => {
                if matches!(items.first(), Some(f) if !matches!(f, Value::Object(_) | Value::Array(_))) {
                    return; // a leaf value list (colour, offsets)
                }
                for v in items.iter_mut() {
                    self.walk(v);
                }
            }
            _ => {}
        }
    }
}

// ---- the document --------------------------------------------------------------


fn units_is_default(u: &Value) -> bool {
    let default = obj(vec![("length", s("m")), ("precision", Value::int(2))]);
    crate::pyval::eq(u, &default)
}

/// The document `save_scene` builds, before serializing it.
pub fn build_document(scene: &Scene, opts: &SaveOptions) -> Result<Built, IgzError> {
    let large = live_element_count(scene) > opts.large_document;
    let mut walk = Walk { scene, opts, sink: if large { Some(Vec::new()) } else { None }, proto_index: HashMap::new(), protos: Vec::new() };
    let mut payload: Vec<(String, Value)> = match mesh_json(scene.loose(), &mut walk.sink, opts)? {
        Value::Object(items) => items,
        _ => Vec::new(),
    };
    if !scene.groups.is_empty() || !scene.kept_components.is_empty() {
        let mut entries = Vec::with_capacity(scene.groups.len());
        for g in &scene.groups {
            entries.push(walk.entry(g)?);
        }
        payload.push(("groups".into(), Value::Array(entries)));
        if !scene.kept_components.is_empty() {
            let mut kept = Vec::with_capacity(scene.kept_components.len());
            for g in &scene.kept_components {
                kept.push(walk.entry(g)?);
            }
            payload.push(("kept_components".into(), Value::Array(kept)));
        }
        if !walk.protos.is_empty() {
            payload.push(("protos".into(), Value::Array(std::mem::take(&mut walk.protos))));
        }
    }
    let geom = walk.sink.take().unwrap_or_default();

    let layers: Vec<Value> = scene.layers.iter().map(|l| Layer::from_dict(l).map(|ly| ly.to_dict()).unwrap_or_else(|_| l.clone())).collect();
    let layer_flags = scene.layers.iter().any(|l| {
        let visible = l.get("visible").is_none_or(truthy);
        !visible || l.get("locked").is_some_and(truthy)
    });
    if scene.layers.len() > 1 || layer_flags {
        payload.push(("layers".into(), Value::Array(layers)));
    }
    let x = &scene.extra;
    if !x.saved_views.is_empty() {
        payload.push(("saved_views".into(), Value::Array(x.saved_views.iter().map(SavedView::to_dict).collect())));
    }
    if !x.compositions.is_empty() {
        payload.push(("compositions".into(), Value::Array(x.compositions.clone())));
    }
    if !scene.dimensions.is_empty() {
        payload.push(("dimensions".into(), Value::Array(scene.dimensions_igz())));
    }
    if !scene.materials.is_empty() {
        let mats = scene.materials.iter().map(|(_, m)| Material::from_dict(m).and_then(|m| m.to_dict()).unwrap_or_else(|_| m.clone())).collect();
        payload.push(("materials".into(), Value::Array(mats)));
    }
    if truthy(&x.dimension_style) {
        payload.push(("dimension_style".into(), x.dimension_style.clone()));
    }
    if let Some(cam) = &x.camera {
        if igz::non_finite_where(cam, "scene") == "scene" {
            payload.push(("camera".into(), cam.clone()));
        }
    }
    if let Some(u) = &x.units {
        if !units_is_default(u) {
            payload.push(("units".into(), u.clone()));
        }
    }
    if let Some(p @ Value::Object(items)) = &x.plugin_data {
        if !items.is_empty() {
            payload.push(("plugin_data".into(), p.clone()));
        }
    }
    if !x.custom_scales.is_empty() {
        payload.push(("custom_scales".into(), Value::Array(x.custom_scales.iter().map(|&n| float(n)).collect())));
    }
    if let Some(b) = x.back_face_color.as_ref().filter(|v| truthy(v)) {
        payload.push(("back_face_color".into(), b.clone()));
    }
    if !scene.section_planes.is_empty() {
        payload.push(("section_planes".into(), Value::Array(scene.section_planes.iter().map(SectionPlane::to_dict).collect())));
        let mut vis: Vec<(String, Value)> = Vec::new();
        if x.planes_hidden {
            vis.push(("planes_hidden".into(), Value::Bool(true)));
        }
        if x.cuts_hidden {
            vis.push(("cuts_hidden".into(), Value::Bool(true)));
        }
        if !vis.is_empty() {
            payload.push(("section_view".into(), Value::Object(vis)));
        }
    }
    let mut shown: Vec<(String, Value)> = Vec::new();
    if x.show_hidden_objects {
        shown.push(("objects".into(), Value::Bool(true)));
    }
    if x.show_hidden_geometry {
        shown.push(("geometry".into(), Value::Bool(true)));
    }
    if !shown.is_empty() {
        payload.push(("hidden_view".into(), Value::Object(shown)));
    }
    if let Some(d) = &x.display_style {
        payload.push(("display_style".into(), d.clone()));
    }
    if let Some(d) = &x.shadows {
        payload.push(("shadows".into(), d.clone()));
    }
    if let Some(d) = &x.georef_datum {
        payload.push(("georef".into(), obj(vec![("datum", d.clone())])));
    }
    if let Some(t) = &x.tile_layer {
        payload.push(("tile_layer".into(), t.clone()));
    }
    if !scene.geo_paths.is_empty() {
        payload.push(("geo_paths".into(), Value::Array(scene.geo_paths.iter().map(GeoPath::to_dict).collect())));
    }
    if !x.geo_points.is_empty() {
        payload.push(("geo_points".into(), Value::Array(x.geo_points.iter().map(GeoPoint::to_dict).collect())));
    }
    if !scene.text_labels.is_empty() {
        payload.push(("text_labels".into(), Value::Array(scene.text_labels.iter().map(TextLabel::to_dict).collect())));
    }
    if !scene.guides.is_empty() {
        payload.push(("guides".into(), Value::Array(scene.guides.iter().filter(|g| g.is_finite()).map(Guide::to_dict).collect())));
    }
    if !scene.image_planes.is_empty() {
        payload.push(("image_planes".into(), Value::Array(scene.image_planes.iter().map(ImagePlane::to_dict).collect())));
    }

    let mut payload = Value::Object(payload);
    let mut packer = Packer { read: opts.textures, blobs: Vec::new(), resolved: HashMap::new(), missing: 0 };
    packer.walk(&mut payload);
    let Packer { mut blobs, missing, .. } = packer;
    if let Some((meta, survey)) = &x.photo_mesh {
        if let Value::Object(items) = &mut payload {
            items.push(("photo_mesh".into(), meta.clone()));
        }
        for (n, d) in survey {
            match blobs.iter_mut().find(|(m, _)| m == n) {
                Some(slot) => slot.1 = d.clone(),
                None => blobs.push((n.clone(), d.clone())),
            }
        }
    }
    let format = if !geom.is_empty() {
        GEOMETRY_FORMAT
    } else if !blobs.is_empty() {
        CONTAINER_FORMAT
    } else {
        PLAIN_FORMAT
    };
    let doc = obj(vec![("igz_format", Value::int(format)), ("app_version", s(opts.app_version)), ("scene", payload)]);
    Ok(Built { doc, large, blobs, geom, missing })
}

/// `igz.save_scene`, to bytes: the file `path` would hold.
pub fn save_scene(scene: &Scene, opts: &SaveOptions) -> Result<Saved, IgzError> {
    let built = build_document(scene, opts)?;
    let text = igz::dumps_document(&built.doc, built.large)?;
    let format = built.doc.get("igz_format").and_then(Value::as_i64).unwrap_or(PLAIN_FORMAT);
    let bytes = if !built.blobs.is_empty() || !built.geom.is_empty() {
        igz_zip::write_container(&text, &built.blobs, &built.geom, built.large)?
    } else {
        text.into_bytes()
    };
    Ok(Saved { bytes, embedded: built.blobs.len(), missing: built.missing, format })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sha1_matches_the_published_vectors() {
        assert_eq!(sha1_hex(b"abc"), "a9993e364706816aba3e25717850c26c9cd0d89d");
        assert_eq!(sha1_hex(b""), "da39a3ee5e6b4b0d3255bfef95601890afd80709");
        let long = vec![b'a'; 1000];
        assert_eq!(sha1_hex(&long), "291e9a6c66994949b57ba5e650361e98fc36b1ba");
    }

    #[test]
    fn a_nan_coordinate_is_refused_not_written() {
        let mut s = Scene::new();
        let nan = V3(f32::NAN, 0.0, 0.0);
        s.dimensions = vec![crate::dimension::Dimension::new(nan, V3(1.0, 0.0, 0.0), V3(0.0, 1.0, 0.0), None, None, None)].into();
        let none = |_: &str| None;
        let err = save_scene(&s, &SaveOptions::new("x", &none)).unwrap_err();
        assert!(matches!(err, IgzError::NonFinite { .. }), "{err:?}");
    }
}
