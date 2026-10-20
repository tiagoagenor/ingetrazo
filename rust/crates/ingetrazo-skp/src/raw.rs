// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
// Port of openskp (MIT, Copyright (c) 2024-2026 Ahsan Mehmood): see NOTICE-openskp.md.
//! What both container walkers produce (`_core.full_parse`'s dict) and the
//! step that turns it into the typed [`Model`] (`SkpFile.parse`).

use std::collections::HashMap;
use std::rc::Rc;

use crate::model::*;

#[derive(Clone, Debug)]
pub struct RawFace {
    pub loops: Vec<Vec<CoEdge>>,
    pub normal: [f64; 3],
    pub material_id: Option<i64>,
    pub back_material_id: Option<i64>,
    pub uv_transform: Option<[f64; 9]>,
    pub uv_transform_back: Option<[f64; 9]>,
    pub uv_projected: bool,
    pub uv_projected_back: bool,
    pub hidden: bool,
    pub layer: Option<i64>,
}

#[derive(Clone, Debug, Default)]
pub struct RawInst {
    pub name: Option<String>,
    pub ref_idx: Option<i64>,
    pub ref_guid: Option<String>,
    pub matrix: Vec<f64>,
    pub material_id: Option<i64>,
    pub hidden: bool,
    pub layer_id: Option<i64>,
    pub properties: Vec<(String, String)>,
    pub attribute_dictionaries: AttrDict,
}

/// `_GeometryBuilder` (VFF) / `_Builder` (legacy).
#[derive(Clone, Debug, Default)]
pub struct Builder {
    pub vertices: OMap<i64, [f64; 3]>,
    /// edge id to `(v1, v2)`; `None` for a missing reference.
    pub edges: OMap<i64, (Option<i64>, Option<i64>)>,
    pub edge_flags: HashMap<i64, u8>,
    pub edge_layers: HashMap<i64, i64>,
    pub edge_curves: HashMap<i64, i64>,
    pub faces: OMap<i64, RawFace>,
    pub instances: Vec<RawInst>,
    pub section_planes: Vec<SectionPlane>,
    pub texts: Vec<TextEntity>,
    pub dimensions: Vec<Dimension>,
    pub construction_lines: Vec<ConstructionLine>,
    pub construction_points: Vec<ConstructionPoint>,
}

#[derive(Clone, Debug, Default)]
pub struct RawDef {
    pub guid: String,
    pub name: Option<String>,
    pub always_faces_camera: bool,
    pub shadows_face_sun: bool,
    pub is_image: bool,
    pub builder: Builder,
}

#[derive(Clone, Debug)]
pub struct RawPage {
    pub name: String,
    pub eye: Option<[f64; 3]>,
    pub target: Option<[f64; 3]>,
    pub up: Option<[f64; 3]>,
    pub fov: f64,
    pub parallel: bool,
    pub ortho_height: f64,
    pub hidden_layer_ids: Vec<i64>,
}

#[derive(Clone, Debug)]
pub struct RawMaterial {
    pub name: String,
    pub color: [i64; 4],
    pub transparency: f64,
    pub colorized: bool,
    pub colorize_type: i64,
    pub texture: Option<Texture>,
}

/// `full_parse`'s result.
#[derive(Debug, Default)]
pub struct RawParse {
    pub version: String,
    pub layer_colors: OMap<String, [i64; 3]>,
    pub layer_hidden: HashMap<String, bool>,
    pub layer_id_to_name: HashMap<i64, String>,
    pub pages: Vec<RawPage>,
    pub dimensions: Vec<Dimension>,
    pub material_id_to_name: OMap<i64, String>,
    /// Materials by name (a later one of the same name replaces the earlier
    /// but keeps its place), as indices into `material_store`.
    pub materials: OMap<String, usize>,
    pub material_store: Vec<RawMaterial>,
    pub materials_by_folder: HashMap<String, usize>,
    /// Definitions by entity id, `None` standing for a definition without one.
    pub defs: OMap<Option<i64>, RawDef>,
    pub root: Builder,
    pub styles: Vec<Style>,
    pub units: Option<String>,
    pub thumbnail: Option<Rc<Vec<u8>>>,
}

fn convert_def(def_id: i64, d: &RawDef, layer_id_to_name: &HashMap<i64, String>) -> Definition {
    let b = &d.builder;
    let mut defn = Definition {
        id: def_id,
        guid: d.guid.clone(),
        name: d.name.clone().unwrap_or_default(),
        always_faces_camera: d.always_faces_camera,
        shadows_face_sun: d.shadows_face_sun,
        is_image: d.is_image,
        ..Default::default()
    };
    for (&v_id, p) in b.vertices.iter() {
        defn.vertices.insert(v_id, Vertex { id: v_id, x: p[0], y: p[1], z: p[2] });
    }
    for (&e_id, &(v1, v2)) in b.edges.iter() {
        let flags = b.edge_flags.get(&e_id).copied().unwrap_or(0);
        defn.edges.insert(
            e_id,
            Edge {
                id: e_id,
                v1_id: v1.unwrap_or(0),
                v2_id: v2.unwrap_or(0),
                soft: flags & 0x08 != 0,
                smooth: flags & 0x10 != 0,
                hidden: flags & 0x01 != 0,
                layer: b.edge_layers.get(&e_id).copied(),
                curve_id: b.edge_curves.get(&e_id).copied(),
            },
        );
    }
    for (&f_id, f) in b.faces.iter() {
        defn.faces.insert(
            f_id,
            Face {
                id: f_id,
                loops: f.loops.clone(),
                normal: Some(f.normal),
                material_id: f.material_id,
                back_material_id: f.back_material_id,
                uv_transform: f.uv_transform,
                uv_transform_back: f.uv_transform_back,
                uv_projected: f.uv_projected,
                uv_projected_back: f.uv_projected_back,
                hidden: f.hidden,
                layer_id: f.layer,
                layer: String::new(),
            },
        );
    }
    for inst in &b.instances {
        defn.instances.push(Instance {
            name: inst.name.clone().unwrap_or_default(),
            ref_idx: inst.ref_idx,
            guid: inst.ref_guid.clone().unwrap_or_default(),
            matrix: inst.matrix.clone(),
            material_id: inst.material_id,
            hidden: inst.hidden,
            layer: match inst.layer_id {
                Some(l) => layer_id_to_name.get(&l).cloned().unwrap_or_default(),
                None => String::new(),
            },
            properties: inst.properties.clone(),
            attribute_dictionaries: inst.attribute_dictionaries.clone(),
        });
    }
    defn.section_planes = b.section_planes.clone();
    defn.texts = b.texts.clone();
    defn.dimensions = b
        .dimensions
        .iter()
        .map(|d| Dimension { text: d.text.clone(), hidden: d.hidden, a: None, b: None, offset: 0.0, plane_x: None, normal: None })
        .collect();
    defn.construction_lines = b.construction_lines.clone();
    defn.construction_points = b.construction_points.clone();
    defn
}

/// `SkpFile.parse` after `full_parse`: the typed model, with face tags
/// resolved to names (`openskp_compat.resolve_face_layers`).
pub fn assemble(parsed: RawParse) -> Model {
    let mut model = Model { version: parsed.version.clone(), units: parsed.units.clone(), ..Default::default() };
    let l2n = &parsed.layer_id_to_name;
    for (key, d) in parsed.defs.iter() {
        let id = key.unwrap_or(0);
        model.definitions.push(convert_def(id, d, l2n));
    }
    model.root = convert_def(0, &RawDef { builder: parsed.root.clone(), ..Default::default() }, l2n);
    model.root.name = "ROOT_MODEL".into();
    model.root.guid = "ROOT".into();
    model.root.id = 0;
    for (name, &[r, g, b]) in parsed.layer_colors.iter() {
        model.layers.push(Layer { name: name.clone(), color: [r, g, b], hidden: parsed.layer_hidden.get(name).copied().unwrap_or(false) });
    }
    for pg in &parsed.pages {
        model.pages.push(Page {
            name: pg.name.clone(),
            eye: pg.eye,
            target: pg.target,
            up: pg.up,
            fov: pg.fov,
            parallel: pg.parallel,
            ortho_height: pg.ortho_height,
            hidden_layers: pg.hidden_layer_ids.iter().filter_map(|i| l2n.get(i).cloned()).collect(),
        });
    }
    model.dimensions = parsed.dimensions.clone();
    // Materials: one per name; the id join resolves a name, then a folder.
    let mut store_to_model: HashMap<usize, usize> = HashMap::new();
    for (_name, &si) in parsed.materials.iter() {
        let m = &parsed.material_store[si];
        store_to_model.insert(si, model.materials.len());
        model.materials.push(Material {
            name: m.name.clone(),
            color: m.color,
            transparency: m.transparency,
            id: None,
            texture: m.texture.clone(),
            colorized: m.colorized,
            colorize_type: m.colorize_type,
        });
    }
    for (&m_id, m_name) in parsed.material_id_to_name.iter() {
        let si = parsed.materials.get(m_name).copied().or_else(|| parsed.materials_by_folder.get(m_name).copied());
        let Some(mi) = si.and_then(|s| store_to_model.get(&s).copied()) else { continue };
        if model.materials[mi].id.is_none() {
            model.materials[mi].id = Some(m_id);
        }
        // a dict: a repeated id keeps its place
        match model.materials_by_id.iter_mut().find(|(i, _)| *i == m_id) {
            Some(slot) => slot.1 = mi,
            None => model.materials_by_id.push((m_id, mi)),
        }
    }
    model.styles = parsed.styles.clone();
    model.layer_id_to_name = parsed.layer_id_to_name.clone();
    model.thumbnail = parsed.thumbnail.clone();
    model.resolve_face_layers();
    model
}
