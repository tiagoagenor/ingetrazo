// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! The scene (`core/scene.py`, `core/group.py`) and opening an .igz
//! document into it (`formats/igz.py` `_load_into_inner`).
//!
//! Meshes live in one arena; a group names its mesh by index, so the
//! instances of a component share one prototype exactly as Python's
//! groups share one `Mesh` object. Annotation and style blocks are kept as
//! the document's JSON until their typed models land (M2.4, M2.5).

use ingetrazo_formats::pyjson::Value;
use ingetrazo_math::{M4, V3};

use crate::mesh::Mesh;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct MeshId(pub u32);

#[derive(Clone, Debug)]
pub struct Group {
    pub mesh: MeshId,
    pub name: String,
    pub layer: Option<String>,
    pub ifc: Option<Value>,
    pub billboard: Option<Value>,
    /// Local → world for an instance; `None` for a classic group.
    pub xform: Option<M4>,
    pub children: Vec<Group>,
    pub text3d: Option<Value>,
    pub hidden: bool,
    pub uid: String,
    pub material: Option<Value>,
    pub axes: Option<M4>,
    pub component: bool,
    pub exploded: Option<Value>,
    pub explode_offset: Option<[f64; 3]>,
    pub ext: Option<Value>,
    pub locked: bool,
}

impl Group {
    pub fn new(mesh: MeshId, name: String) -> Group {
        Group {
            mesh,
            name,
            layer: None,
            ifc: None,
            billboard: None,
            xform: None,
            children: Vec::new(),
            text3d: None,
            hidden: false,
            uid: String::new(),
            material: None,
            axes: None,
            component: true,
            exploded: None,
            explode_offset: None,
            ext: None,
            locked: false,
        }
    }

    /// `is_component`: an instance (has an xform) flagged as a component.
    pub fn is_component(&self) -> bool {
        self.xform.is_some() && self.component
    }

    /// `adopt`: take nested placements; a group that owns placements is
    /// always an instance (identity when it had no matrix).
    pub fn adopt(&mut self, children: Vec<Group>) {
        self.children = children;
        if !self.children.is_empty() && self.xform.is_none() {
            self.xform = Some(M4::identity());
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct Scene {
    /// Every mesh: the loose one is `meshes[0]`.
    pub meshes: Vec<Mesh>,
    pub groups: Vec<Group>,
    pub kept_components: Vec<Group>,
    /// Document blocks kept as JSON for now (typed in M2.4/M2.5).
    pub layers: Vec<Value>,
    pub materials: Vec<(String, Value)>,
    pub dimensions: Vec<Value>,
    pub text_labels: Vec<Value>,
    pub guides: Vec<Value>,
    pub section_planes: Vec<Value>,
    pub image_planes: Vec<Value>,
    pub geo_paths: Vec<Value>,
    /// Blocks the .igz writer needs and the model does not edit yet, kept as
    /// the JSON the document brought (`igz_write::save_scene` writes them).
    pub extra: SceneExtra,
}

/// The rest of a document's blocks, as stored JSON (see [`Scene::extra`]).
/// `None` / empty = the block is absent from the file; Python's loader
/// resets the same fields to their defaults.
#[derive(Clone, Debug)]
pub struct SceneExtra {
    pub saved_views: Vec<Value>,
    pub compositions: Vec<Value>,
    pub geo_points: Vec<Value>,
    /// `scene.dimension_style` (always present in Python: a default dict).
    pub dimension_style: Value,
    pub camera: Option<Value>,
    /// `scene.units`; `None` = the metre default.
    pub units: Option<Value>,
    pub plugin_data: Option<Value>,
    pub custom_scales: Vec<f64>,
    pub back_face_color: Option<Value>,
    /// `not scene.show_section_planes` / `not scene.show_section_cuts`.
    pub planes_hidden: bool,
    pub cuts_hidden: bool,
    pub show_hidden_objects: bool,
    pub show_hidden_geometry: bool,
    /// `scene.display_style.to_dict()`, `None` while it is the default
    /// style (the writer leaves the default out).
    pub display_style: Option<Value>,
    /// `scene.shadows.to_dict()`, `None` while default.
    pub shadows: Option<Value>,
    /// `georef.datum` (`SceneDatum.to_dict()`).
    pub georef_datum: Option<Value>,
    pub tile_layer: Option<Value>,
    /// `photo_mesh`: its JSON block and the blobs `pack_mesh` made (stored
    /// members). Written, never read here.
    pub photo_mesh: Option<(Value, Vec<(String, Vec<u8>)>)>,
}

/// `Scene.dimension_style`'s default factory.
pub fn default_dimension_style() -> Value {
    let i = Value::int;
    Value::Object(vec![
        ("decimals".into(), i(2)),
        ("units".into(), s("m")),
        ("font_size".into(), i(9)),
        ("color".into(), Value::Array(vec![i(45), i(55), i(75)])),
        ("norma".into(), s("iso")),
        ("base_step_mm".into(), Value::Float(8.0)),
        ("ends".into(), s("arrow")),
    ])
}

impl Default for SceneExtra {
    fn default() -> Self {
        SceneExtra {
            saved_views: Vec::new(),
            compositions: Vec::new(),
            geo_points: Vec::new(),
            dimension_style: default_dimension_style(),
            camera: None,
            units: None,
            plugin_data: None,
            custom_scales: Vec::new(),
            back_face_color: None,
            planes_hidden: false,
            cuts_hidden: false,
            show_hidden_objects: false,
            show_hidden_geometry: false,
            display_style: None,
            shadows: None,
            georef_datum: None,
            tile_layer: None,
            photo_mesh: None,
        }
    }
}

pub const DEFAULT_LAYER: &str = "Layer 0";

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LoadError {
    Format(String),
    NewerFormat(i64),
}

fn num(v: &Value) -> Option<f64> {
    match v {
        Value::Float(x) => Some(*x),
        Value::Int(s) => s.parse().ok(),
        _ => None,
    }
}

fn point(v: &Value) -> Option<V3> {
    match v {
        Value::Array(c) if c.len() == 3 => Some(V3(num(&c[0])? as f32, num(&c[1])? as f32, num(&c[2])? as f32)),
        _ => None,
    }
}

fn truthy(v: Option<&Value>) -> bool {
    match v {
        None | Some(Value::Null) | Some(Value::Bool(false)) => false,
        Some(Value::Str(x)) => !x.is_empty(),
        Some(Value::Int(x)) => x != "0",
        Some(Value::Float(x)) => *x != 0.0,
        Some(Value::Array(a)) => !a.is_empty(),
        Some(Value::Object(o)) => !o.is_empty(),
        Some(Value::Bool(true)) => true,
    }
}

fn s(v: &str) -> Value {
    Value::Str(v.to_string())
}

/// `igz._face_attrs_from_json`, with an embedded texture's `"embed"` read
/// back as its `"path"` (`_unpack_textures`): here the archive member's
/// name, which the signature resolves to the member's bytes.
fn face_attrs(raw: &Value) -> Vec<(String, Value)> {
    let unembed = |t: &Value| -> Value {
        match t {
            Value::Object(items) => {
                let mut out: Vec<(String, Value)> = Vec::new();
                let has_path = items.iter().any(|(k, _)| k == "path");
                let has_base = items.iter().any(|(k, _)| k == "base");
                for (k, v) in items {
                    match k.as_str() {
                        "embed" if !has_path => out.push(("path".into(), v.clone())),
                        "base_embed" if !has_base => out.push(("base".into(), v.clone())),
                        "embed" | "base_embed" => {}
                        _ => out.push((k.clone(), v.clone())),
                    }
                }
                Value::Object(out)
            }
            other => other.clone(),
        }
    };
    let mut attrs: Vec<(String, Value)> = Vec::new();
    if let Some(c) = raw.get("color") {
        if !matches!(c, Value::Null) {
            attrs.push(("color".into(), c.clone()));
        }
    }
    if let Some(t) = raw.get("texture") {
        if !matches!(t, Value::Null) {
            attrs.push(("texture".into(), unembed(t)));
        }
    }
    if truthy(raw.get("layer")) {
        attrs.push(("layer".into(), raw.get("layer").cloned().unwrap_or(Value::Null)));
    }
    if truthy(raw.get("ifc")) {
        attrs.push(("ifc".into(), raw.get("ifc").cloned().unwrap_or(Value::Null)));
    }
    if let Some(o) = raw.get("opacity") {
        if let Some(x) = num(o) {
            attrs.push(("opacity".into(), Value::Float(x)));
        }
    }
    if truthy(raw.get("hidden")) {
        attrs.push(("hidden".into(), Value::Bool(true)));
    }
    if truthy(raw.get("mat")) {
        attrs.push(("mat".into(), raw.get("mat").cloned().unwrap_or(Value::Null)));
    }
    match raw.get("back") {
        Some(b @ Value::Object(_)) => attrs.push(("back".into(), unembed(b))),
        Some(Value::Bool(true)) => attrs.push(("back".into(), Value::Bool(true))),
        _ => {}
    }
    attrs
}

/// `igz._BULK_MIN`: a mesh at least this big (edges + 4 per face) loads in
/// one bulk pass, which numbers its vertices differently.
pub const BULK_MIN: usize = 1024;

/// `igz._bulk_sized`.
pub fn bulk_sized(n_edges: usize, n_faces: usize) -> bool {
    n_edges + n_faces * 4 >= BULK_MIN
}

fn finite3(p: &V3) -> bool {
    p.0.is_finite() && p.1.is_finite() && p.2.is_finite()
}

/// `igz._load_mesh_json`: the per-entity walk for a small mesh, the bulk
/// weld (`Mesh.bulk_weld` + `add_edges_welded` + `add_faces_welded`) for a
/// big one. Points that are not finite are dropped first, as
/// `_without_nonfinite` does.
pub fn load_mesh(mesh: &mut Mesh, payload: &Value) -> Result<(), LoadError> {
    if payload.get("geom").is_some() {
        return Err(LoadError::Format("format-3 geometry block (expand it first)".into()));
    }
    let n_edges = match payload.get("edges") {
        Some(Value::Array(a)) => a.len(),
        _ => 0,
    };
    let n_faces = match payload.get("faces") {
        Some(Value::Array(a)) => a.len(),
        _ => 0,
    };
    if bulk_sized(n_edges, n_faces) {
        return load_mesh_bulk(mesh, payload);
    }
    load_mesh_small(mesh, payload)
}

fn load_mesh_bulk(mesh: &mut Mesh, payload: &Value) -> Result<(), LoadError> {
    use crate::mesh::EdgeFlags;
    let ring = |v: Option<&Value>| -> Option<Vec<V3>> {
        match v {
            Some(Value::Array(ps)) => ps.iter().map(point).collect(),
            _ => None,
        }
    };
    let mut flat: Vec<V3> = Vec::new();
    let mut flags: Vec<EdgeFlags> = Vec::new();
    if let Some(Value::Array(edges)) = payload.get("edges") {
        for raw in edges {
            let (Some(a), Some(b)) = (raw.get("a").and_then(point), raw.get("b").and_then(point)) else {
                continue;
            };
            if !finite3(&a) || !finite3(&b) {
                continue;
            }
            flat.push(a);
            flat.push(b);
            flags.push(EdgeFlags {
                soft: truthy(raw.get("soft")),
                curve: raw.get("curve").and_then(Value::as_i64),
                layer: match raw.get("layer") {
                    Some(Value::Str(l)) => Some(l.clone()),
                    _ => None,
                },
                hidden: truthy(raw.get("hidden")),
            });
        }
    }
    let n_edge_pts = flat.len();
    let mut ring_sizes: Vec<usize> = Vec::new();
    let mut ring_counts: Vec<usize> = Vec::new();
    let mut attrs: Vec<Vec<(String, Value)>> = Vec::new();
    if let Some(Value::Array(faces)) = payload.get("faces") {
        for raw in faces {
            let Some(outer) = ring(raw.get("vertices")) else { continue };
            let holes: Vec<Vec<V3>> = match raw.get("holes") {
                Some(Value::Array(hs)) => hs.iter().filter_map(|h| ring(Some(h))).collect(),
                _ => Vec::new(),
            };
            if !outer.iter().all(finite3) || !holes.iter().all(|h| h.iter().all(finite3)) {
                continue;
            }
            // _load_mesh_json drops nested holes first, as add_face does.
            let holes: Vec<Vec<V3>> = if holes.len() > 1 {
                crate::topology::maximal_holes(&holes, mesh.hypot).into_iter().map(|i| holes[i].clone()).collect()
            } else {
                holes
            };
            ring_counts.push(1 + holes.len());
            ring_sizes.push(outer.len());
            flat.extend(outer);
            for h in holes {
                ring_sizes.push(h.len());
                flat.extend(h);
            }
            attrs.push(face_attrs(raw));
        }
    }
    if flat.is_empty() {
        return Ok(());
    }
    let (vobjs, inverse) = mesh.bulk_weld(&flat);
    let mut emap = None;
    if !flags.is_empty() {
        let ia: Vec<usize> = (0..n_edge_pts).step_by(2).map(|i| inverse[i]).collect();
        let ib: Vec<usize> = (1..n_edge_pts).step_by(2).map(|i| inverse[i]).collect();
        emap = Some(mesh.add_edges_welded(&vobjs, &ia, &ib, Some(&flags)));
    }
    if !ring_counts.is_empty() {
        mesh.add_faces_welded(&vobjs, &inverse[n_edge_pts..], &ring_sizes, &ring_counts, Some(&attrs), emap);
    }
    Ok(())
}

/// `igz._load_mesh_small`: the per-entity walk (edges, then faces).
fn load_mesh_small(mesh: &mut Mesh, payload: &Value) -> Result<(), LoadError> {
    if let Some(Value::Array(edges)) = payload.get("edges") {
        for raw in edges {
            let (Some(a), Some(b)) = (raw.get("a").and_then(point), raw.get("b").and_then(point)) else {
                continue;
            };
            if !(a.0.is_finite() && a.1.is_finite() && a.2.is_finite() && b.0.is_finite() && b.1.is_finite() && b.2.is_finite()) {
                continue;
            }
            let Ok(e) = mesh.add_edge(a, b) else { continue };
            let ed = mesh.edge_mut(e);
            if truthy(raw.get("soft")) {
                ed.soft = true;
            }
            if truthy(raw.get("hidden")) {
                ed.hidden = true;
            }
            if let Some(Value::Str(l)) = raw.get("layer") {
                if !l.is_empty() {
                    ed.layer = Some(l.clone());
                }
            }
            if let Some(c) = raw.get("curve").and_then(Value::as_i64) {
                ed.curve = Some(c);
            }
        }
    }
    if let Some(Value::Array(faces)) = payload.get("faces") {
        for raw in faces {
            let ring = |v: Option<&Value>| -> Option<Vec<V3>> {
                match v {
                    Some(Value::Array(ps)) => ps.iter().map(point).collect(),
                    _ => None,
                }
            };
            let Some(outer) = ring(raw.get("vertices")) else { continue };
            let holes: Vec<Vec<V3>> = match raw.get("holes") {
                Some(Value::Array(hs)) => hs.iter().filter_map(|h| ring(Some(h))).collect(),
                _ => Vec::new(),
            };
            let finite = |r: &Vec<V3>| r.iter().all(|p| p.0.is_finite() && p.1.is_finite() && p.2.is_finite());
            if !finite(&outer) || !holes.iter().all(finite) {
                continue;
            }
            let f = mesh.add_face(&outer, &holes);
            let attrs = face_attrs(raw);
            if !attrs.is_empty() {
                let fa = &mut mesh.face_mut(f).attrs;
                for (k, v) in attrs {
                    match fa.iter_mut().find(|(x, _)| *x == k) {
                        Some(slot) => slot.1 = v,
                        None => fa.push((k, v)),
                    }
                }
            }
        }
    }
    Ok(())
}

fn matrix16(v: Option<&Value>) -> Option<M4> {
    match v {
        Some(Value::Array(xs)) if xs.len() == 16 => {
            let mut d = [0.0f32; 16];
            for (i, x) in xs.iter().enumerate() {
                d[i] = num(x)? as f32;
            }
            // data() is column major; the 16-float constructor is General.
            Some(M4::from_data(d))
        }
        _ => None,
    }
}

impl Scene {
    pub fn new() -> Scene {
        Scene { meshes: vec![Mesh::new()], layers: vec![Value::Object(vec![("name".into(), s(DEFAULT_LAYER))])], ..Default::default() }
    }

    pub fn loose(&self) -> &Mesh {
        &self.meshes[0]
    }

    pub fn mesh(&self, id: MeshId) -> &Mesh {
        &self.meshes[id.0 as usize]
    }

    fn new_mesh(&mut self) -> MeshId {
        self.meshes.push(Mesh::new());
        MeshId(self.meshes.len() as u32 - 1)
    }

    /// Open a document (`{"igz_format", "scene"}`) as `load_into` does.
    /// Format-3 geometry blocks must be expanded into JSON meshes first.
    pub fn from_document(doc: &Value) -> Result<Scene, LoadError> {
        let fmt = doc.get("igz_format").and_then(Value::as_i64).unwrap_or(1);
        if fmt > 3 {
            return Err(LoadError::NewerFormat(fmt));
        }
        let empty = Value::Object(Vec::new());
        let payload = doc.get("scene").unwrap_or(&empty);
        let mut sc = Scene::new();
        load_mesh(&mut sc.meshes[0], payload)?;
        if let Some(Value::Array(ls)) = payload.get("layers") {
            if !ls.is_empty() {
                sc.layers = ls.clone();
                if !ls.iter().any(|l| matches!(l.get("name"), Some(Value::Str(n)) if n == DEFAULT_LAYER)) {
                    sc.layers.insert(0, Value::Object(vec![("name".into(), s(DEFAULT_LAYER))]));
                }
            }
        }
        if let Some(Value::Array(ms)) = payload.get("materials") {
            for m in ms {
                if let Some(Value::Str(name)) = m.get("name") {
                    if name.is_empty() {
                        continue;
                    }
                    match sc.materials.iter_mut().find(|(n, _)| n == name) {
                        Some(slot) => slot.1 = m.clone(),
                        None => sc.materials.push((name.clone(), m.clone())),
                    }
                }
            }
        }
        let mut protos: Vec<MeshId> = Vec::new();
        if let Some(Value::Array(ps)) = payload.get("protos") {
            for raw in ps {
                let id = sc.new_mesh();
                load_mesh(&mut sc.meshes[id.0 as usize], raw)?;
                protos.push(id);
            }
        }
        let mut groups = Vec::new();
        if let Some(Value::Array(gs)) = payload.get("groups") {
            for raw in gs {
                groups.push(sc.group_from(raw, &protos, 0)?);
            }
        }
        sc.groups = groups;
        let mut kept = Vec::new();
        if let Some(Value::Array(ks)) = payload.get("kept_components") {
            for raw in ks {
                if matches!(raw, Value::Object(_)) {
                    let g = sc.group_from(raw, &protos, 0)?;
                    if g.is_component() {
                        kept.push(g);
                    }
                }
            }
        }
        sc.kept_components = kept;
        let list = |k: &str| match payload.get(k) {
            Some(Value::Array(a)) => a.clone(),
            _ => Vec::new(),
        };
        sc.dimensions = list("dimensions");
        sc.text_labels = list("text_labels");
        sc.guides = list("guides");
        sc.section_planes = list("section_planes");
        sc.image_planes = list("image_planes");
        sc.geo_paths = list("geo_paths");
        sc.load_extra(payload);
        Ok(sc)
    }

    /// The blocks of `_load_into_inner` that only the writer reads yet.
    fn load_extra(&mut self, payload: &Value) {
        let list = |k: &str| match payload.get(k) {
            Some(Value::Array(a)) => a.clone(),
            _ => Vec::new(),
        };
        let object = |k: &str| match payload.get(k) {
            Some(v @ Value::Object(_)) => Some(v.clone()),
            _ => None,
        };
        let x = &mut self.extra;
        x.saved_views = list("saved_views");
        x.compositions = list("compositions");
        x.geo_points = list("geo_points");
        // scene.dimension_style.update({"ends": "tick", **style}): merged
        // over the default dict, keeping the default's key order.
        if let Value::Object(items) = &mut x.dimension_style {
            let mut merged: Vec<(String, Value)> = vec![("ends".into(), s("tick"))];
            if let Some(Value::Object(st)) = payload.get("dimension_style") {
                for (k, v) in st {
                    match merged.iter_mut().find(|(n, _)| n == k) {
                        Some(slot) => slot.1 = v.clone(),
                        None => merged.push((k.clone(), v.clone())),
                    }
                }
            }
            for (k, v) in merged {
                match items.iter_mut().find(|(n, _)| *n == k) {
                    Some(slot) => slot.1 = v,
                    None => items.push((k, v)),
                }
            }
        }
        x.camera = object("camera");
        x.units = object("units");
        x.plugin_data = object("plugin_data");
        if let Some(Value::Array(ns)) = payload.get("custom_scales") {
            x.custom_scales = ns.iter().filter_map(num).filter(|n| *n > 0.0).collect();
        }
        if let Some(Value::Array(c)) = payload.get("back_face_color") {
            if c.len() == 3 && c.iter().all(|v| num(v).is_some()) {
                x.back_face_color = Some(Value::Array(c.iter().map(|v| Value::Float(num(v).unwrap_or(0.0))).collect()));
            }
        }
        if let Some(v @ Value::Object(_)) = payload.get("section_view") {
            x.planes_hidden = truthy(v.get("planes_hidden"));
            x.cuts_hidden = truthy(v.get("cuts_hidden"));
        }
        if let Some(v @ Value::Object(_)) = payload.get("hidden_view") {
            x.show_hidden_objects = truthy(v.get("objects"));
            x.show_hidden_geometry = truthy(v.get("geometry"));
        }
        x.display_style = object("display_style");
        x.shadows = object("shadows");
        if let Some(g @ Value::Object(_)) = payload.get("georef") {
            if let Some(d @ Value::Object(_)) = g.get("datum") {
                x.georef_datum = Some(d.clone());
            }
        }
        x.tile_layer = payload.get("tile_layer").filter(|v| truthy(Some(v))).cloned();
    }

    /// `_group_from`.
    fn group_from(&mut self, raw: &Value, protos: &[MeshId], depth: usize) -> Result<Group, LoadError> {
        let name = match raw.get("name") {
            Some(Value::Str(n)) if !n.is_empty() => n.clone(),
            _ => String::new(), // Python numbers it "Group N"; documents always carry names
        };
        let xform = raw.get("xform");
        let mut g = if let (Some(x), Some(p)) = (xform.filter(|v| !matches!(v, Value::Null)), raw.get("proto")) {
            let idx = p.as_i64().ok_or_else(|| LoadError::Format("proto".into()))? as usize;
            let mesh = *protos.get(idx).ok_or_else(|| LoadError::Format(format!("proto {idx} missing")))?;
            let mut g = Group::new(mesh, name);
            g.xform = matrix16(Some(x));
            g
        } else {
            let id = self.new_mesh();
            load_mesh(&mut self.meshes[id.0 as usize], raw)?;
            Group::new(id, name)
        };
        g.axes = matrix16(raw.get("axes"));
        if truthy(raw.get("layer")) {
            if let Some(Value::Str(l)) = raw.get("layer") {
                g.layer = Some(l.clone());
            }
        }
        if truthy(raw.get("ifc")) {
            g.ifc = raw.get("ifc").cloned();
        }
        if truthy(raw.get("billboard")) {
            g.billboard = raw.get("billboard").cloned();
        }
        if let Some(t @ Value::Object(_)) = raw.get("text3d") {
            g.text3d = Some(t.clone());
        }
        if let Some(e @ Value::Object(_)) = raw.get("ext") {
            g.ext = Some(e.clone());
        }
        if let Some(Value::Str(u)) = raw.get("uid") {
            g.uid = u.clone();
        }
        g.hidden = truthy(raw.get("hidden"));
        g.locked = truthy(raw.get("locked"));
        if let Some(m @ Value::Object(_)) = raw.get("material") {
            g.material = Some(m.clone());
        }
        if let Some(e @ Value::Object(_)) = raw.get("exploded") {
            g.exploded = Some(e.clone());
        }
        if let Some(Value::Array(off)) = raw.get("explode_offset") {
            if off.len() == 3 {
                if let (Some(a), Some(b), Some(c)) = (num(&off[0]), num(&off[1]), num(&off[2])) {
                    g.explode_offset = Some([a, b, c]);
                }
            }
        }
        if depth < 32 {
            let mut kids = Vec::new();
            if let Some(Value::Array(cs)) = raw.get("children") {
                for c in cs {
                    kids.push(self.group_from(c, protos, depth + 1)?);
                }
            }
            g.adopt(kids);
        }
        if let Some(c) = raw.get("component") {
            g.component = truthy(Some(c));
        } else if let Some(Value::Array(cs)) = raw.get("children") {
            if cs.iter().any(|c| matches!(c, Value::Object(_)) && c.get("xform").is_none()) {
                g.component = false;
            }
        }
        Ok(g)
    }
}
