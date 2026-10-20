// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
// Port of openskp (MIT, Copyright (c) 2024-2026 Ahsan Mehmood): see NOTICE-openskp.md.
//! The typed model a parse yields: `openskp/model.py`.
//!
//! Python keeps ids in `dict`s, whose iteration order is insertion order and
//! whose duplicate assignment keeps the first position; [`OMap`] is that.
//! Lengths are in inches, as the file stores them.

use std::collections::HashMap;
use std::hash::Hash;
use std::rc::Rc;

/// An insertion-ordered map: a second assignment replaces the value and keeps
/// the first position, like a Python `dict`.
#[derive(Clone, Debug)]
pub struct OMap<K, V> {
    items: Vec<(K, V)>,
    index: HashMap<K, usize>,
}

impl<K, V> Default for OMap<K, V> {
    fn default() -> Self {
        OMap { items: Vec::new(), index: HashMap::new() }
    }
}

impl<K: Hash + Eq + Clone, V> OMap<K, V> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&mut self, k: K, v: V) {
        match self.index.get(&k) {
            Some(&i) => self.items[i].1 = v,
            None => {
                self.index.insert(k.clone(), self.items.len());
                self.items.push((k, v));
            }
        }
    }

    pub fn get(&self, k: &K) -> Option<&V> {
        self.index.get(k).map(|&i| &self.items[i].1)
    }

    pub fn get_mut(&mut self, k: &K) -> Option<&mut V> {
        match self.index.get(k) {
            Some(&i) => Some(&mut self.items[i].1),
            None => None,
        }
    }

    pub fn contains_key(&self, k: &K) -> bool {
        self.index.contains_key(k)
    }

    pub fn len(&self) -> usize {
        self.items.len()
    }

    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = (&K, &V)> {
        self.items.iter().map(|(k, v)| (k, v))
    }

    pub fn values(&self) -> impl Iterator<Item = &V> {
        self.items.iter().map(|(_, v)| v)
    }

    pub fn values_mut(&mut self) -> impl Iterator<Item = &mut V> {
        self.items.iter_mut().map(|(_, v)| v)
    }

    pub fn keys(&self) -> impl Iterator<Item = &K> {
        self.items.iter().map(|(k, _)| k)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Vertex {
    pub id: i64,
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

/// A directed edge. `layer` and `curve_id` are the raw ids of the file.
#[derive(Clone, Debug, PartialEq)]
pub struct Edge {
    pub id: i64,
    pub v1_id: i64,
    pub v2_id: i64,
    pub soft: bool,
    pub smooth: bool,
    pub hidden: bool,
    pub layer: Option<i64>,
    pub curve_id: Option<i64>,
}

/// One co-edge of a loop: the edge id and `1` (same direction) or `-1`.
pub type CoEdge = (i64, i32);

#[derive(Clone, Debug, PartialEq)]
pub struct Face {
    pub id: i64,
    pub loops: Vec<Vec<CoEdge>>,
    pub normal: Option<[f64; 3]>,
    pub material_id: Option<i64>,
    pub back_material_id: Option<i64>,
    pub uv_transform: Option<[f64; 9]>,
    pub uv_transform_back: Option<[f64; 9]>,
    pub uv_projected: bool,
    pub uv_projected_back: bool,
    pub hidden: bool,
    /// The face's own tag as the file stores it (an id), or `None`.
    pub layer_id: Option<i64>,
    /// The tag's name once [`Model::resolve_face_layers`] ran (it does at the
    /// end of every parse): empty for no tag or the default one.
    pub layer: String,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Layer {
    pub name: String,
    pub color: [i64; 3],
    pub hidden: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Style {
    pub name: String,
    pub front_color: Option<[i64; 3]>,
    pub back_color: Option<[i64; 3]>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Texture {
    pub filename: String,
    /// Tile size in inches; `0.0` when the file does not say.
    pub width: f64,
    pub height: f64,
    pub data: Option<Rc<Vec<u8>>>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Material {
    pub name: String,
    /// `(r, g, b, a)`, 0-255 (the XML may say more).
    pub color: [i64; 4],
    /// 1.0 opaque.
    pub transparency: f64,
    pub id: Option<i64>,
    pub texture: Option<Texture>,
    pub colorized: bool,
    pub colorize_type: i64,
}

/// A value of an attribute dictionary, typed as the file stores it.
#[derive(Clone, Debug, PartialEq)]
pub enum AttrValue {
    Null,
    Str(String),
    Float(f64),
    Int(i64),
    Vec3([f64; 3]),
    List(Vec<AttrValue>),
}

impl AttrValue {
    /// `_stringify_attr_value`.
    pub fn stringify(&self) -> String {
        match self {
            AttrValue::Null => String::new(),
            AttrValue::Str(s) => s.clone(),
            AttrValue::Float(x) => crate::pystr::float_repr(*x),
            AttrValue::Int(i) => i.to_string(),
            AttrValue::Vec3(v) => v.iter().map(|x| crate::pystr::float_repr(*x)).collect::<Vec<_>>().join(","),
            AttrValue::List(l) => l.iter().map(AttrValue::stringify).collect::<Vec<_>>().join(","),
        }
    }
}

pub type AttrDict = Vec<(String, Vec<(String, AttrValue)>)>;

#[derive(Clone, Debug, PartialEq)]
pub struct Instance {
    pub name: String,
    pub ref_idx: Option<i64>,
    pub guid: String,
    /// 13 numbers: a 3x3 row-major, the translation, a scale scalar. Empty
    /// when the file carries none.
    pub matrix: Vec<f64>,
    /// The tag's name (empty for none).
    pub layer: String,
    pub properties: Vec<(String, String)>,
    pub attribute_dictionaries: AttrDict,
    pub material_id: Option<i64>,
    pub hidden: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SectionPlane {
    pub plane: [f64; 4],
    pub name: String,
    pub label: String,
    pub hidden: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct TextEntity {
    pub text: String,
    pub hidden: bool,
    pub point: Option<[f64; 3]>,
    pub label_point: Option<[f64; 3]>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Dimension {
    pub text: String,
    pub hidden: bool,
    pub a: Option<[f64; 3]>,
    pub b: Option<[f64; 3]>,
    pub offset: f64,
    pub plane_x: Option<[f64; 3]>,
    pub normal: Option<[f64; 3]>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Page {
    pub name: String,
    pub eye: Option<[f64; 3]>,
    pub target: Option<[f64; 3]>,
    pub up: Option<[f64; 3]>,
    pub fov: f64,
    pub parallel: bool,
    pub ortho_height: f64,
    pub hidden_layers: Vec<String>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ConstructionLine {
    pub point: [f64; 3],
    pub direction: [f64; 3],
    pub start: Option<[f64; 3]>,
    pub end: Option<[f64; 3]>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ConstructionPoint {
    pub position: [f64; 3],
}

#[derive(Clone, Debug)]
pub struct Definition {
    pub id: i64,
    pub guid: String,
    pub name: String,
    pub vertices: OMap<i64, Vertex>,
    pub edges: OMap<i64, Edge>,
    pub faces: OMap<i64, Face>,
    pub instances: Vec<Instance>,
    pub section_planes: Vec<SectionPlane>,
    pub texts: Vec<TextEntity>,
    pub dimensions: Vec<Dimension>,
    pub construction_lines: Vec<ConstructionLine>,
    pub construction_points: Vec<ConstructionPoint>,
    pub always_faces_camera: bool,
    pub shadows_face_sun: bool,
    pub is_image: bool,
}

impl Default for Definition {
    fn default() -> Self {
        Definition {
            id: 0,
            guid: String::new(),
            name: String::new(),
            vertices: OMap::new(),
            edges: OMap::new(),
            faces: OMap::new(),
            instances: Vec::new(),
            section_planes: Vec::new(),
            texts: Vec::new(),
            dimensions: Vec::new(),
            construction_lines: Vec::new(),
            construction_points: Vec::new(),
            always_faces_camera: false,
            shadows_face_sun: false,
            is_image: false,
        }
    }
}

/// Everything a parse found.
#[derive(Clone, Debug, Default)]
pub struct Model {
    pub version: String,
    /// The named component / group definitions, in file order.
    pub definitions: Vec<Definition>,
    pub root: Definition,
    pub layers: Vec<Layer>,
    pub pages: Vec<Page>,
    pub dimensions: Vec<Dimension>,
    pub materials: Vec<Material>,
    /// TLV / slot material id to an index into `materials`, in file order.
    pub materials_by_id: Vec<(i64, usize)>,
    pub styles: Vec<Style>,
    pub units: Option<String>,
    /// The tag id to name table of the parse (`layer_id_to_name`).
    pub layer_id_to_name: HashMap<i64, String>,
    /// The zip's `meta/model_thumbnail.png`, VFF files only.
    pub thumbnail: Option<Rc<Vec<u8>>>,
}

impl Model {
    /// `openskp_compat.resolve_face_layers`: a face's tag id to its name; a
    /// face on the default tag, or on an id the table does not know, gets no
    /// layer. Returns how many changed.
    pub fn resolve_face_layers(&mut self) -> usize {
        const DEFAULT_TAGS: [&str; 3] = ["Layer0", "Layer 0", "Untagged"];
        let table = &self.layer_id_to_name;
        let mut changed = 0;
        for d in self.definitions.iter_mut() {
            for f in d.faces.values_mut() {
                if let Some(id) = f.layer_id {
                    let name = table.get(&id).map(String::as_str).unwrap_or("");
                    f.layer = if DEFAULT_TAGS.contains(&name) { String::new() } else { name.to_string() };
                    changed += 1;
                }
            }
        }
        changed
    }
}

/// `loose_edge_runs`' sibling the importer needs: a definition's version of
/// the `by id` lookup.
pub fn index_by_id(defs: &[Definition]) -> HashMap<i64, usize> {
    let mut by_id = HashMap::new();
    for (i, d) in defs.iter().enumerate() {
        by_id.insert(d.id, i);
    }
    by_id
}
