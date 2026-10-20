// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
// Port of openskp (MIT, Copyright (c) 2024-2026 Ahsan Mehmood): see NOTICE-openskp.md.
//! The VFF container of SketchUp 2021 and later: a zip holding `model.dat`
//! (TLV records), material and style XML, and `meta/meta.dat`
//! (`_core.full_parse`, the part after the legacy check).

use std::collections::HashMap;
use std::rc::Rc;

use crate::error::{Result, SkpError};
use crate::model::*;
use crate::pystr::{py_float, py_int, utf16le_ignore, utf8_replace};
use crate::raw::*;
use crate::tlv::{self, f64_at, find_all, find_child_tag, flat, flat_headers, u32_at, var_int, Node};
use crate::xml;
use crate::zip::Zip;

type R<T> = std::result::Result<T, String>;

const MAT_NS: &str = "http://sketchup.google.com/schemas/sketchup/1.0/material";
const STY_NS: &str = "http://sketchup.google.com/schemas/sketchup/1.0/style";
const TYP_NS: &str = "http://sketchup.google.com/schemas/1.0/types";

/// `_MAX_UNCOMPRESSED_ENTRY_BYTES` and the ratio guard of `_core`.
const MAX_ENTRY_BYTES: u64 = 16 * 1024 * 1024 * 1024;
const MAX_RATIO: u64 = 1000;
const RATIO_CHECK_FROM: u64 = 1024 * 1024;

fn validate_entry(zf: &Zip, name: &str) -> Result<()> {
    let info = zf.info(name).ok_or_else(|| SkpError::new(format!("KeyError: There is no item named '{name}' in the archive"), "zip_extract"))?;
    let declared = info.file_size;
    if declared == 0 {
        return Ok(());
    }
    if declared > MAX_ENTRY_BYTES {
        return Err(SkpError::new(
            format!("ZIP entry '{name}' declares {declared} bytes uncompressed, exceeding the {MAX_ENTRY_BYTES}-byte safety ceiling"),
            "zip_extract",
        ));
    }
    if declared >= RATIO_CHECK_FROM {
        let compressed = info.compress_size;
        if compressed == 0 || declared / compressed > MAX_RATIO || (declared as f64 / compressed as f64) > MAX_RATIO as f64 {
            return Err(SkpError::new(
                format!("ZIP entry '{name}' declares an implausible compression ratio ({declared} bytes from {compressed} bytes compressed) - likely a decompression bomb"),
                "zip_extract",
            ));
        }
    }
    Ok(())
}

fn read_entry(zf: &Zip, name: &str) -> Result<Vec<u8>> {
    zf.read(name).map_err(|e| SkpError::new(e, "zip_extract"))
}

// ---- geometry ---------------------------------------------------------------

fn hex_upper(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02X}")).collect()
}

/// `_extract_uv_transforms(dc05_payload)`: the front and back 3x3 mappings.
fn extract_uv_transforms(dc05: &[u8]) -> (Option<[f64; 9]>, Option<[f64; 9]>) {
    fn find<'a>(seq: &[(u16, &'a [u8])], tag: u16) -> Option<&'a [u8]> {
        seq.iter().find(|(t, _)| *t == tag).map(|(_, b)| *b)
    }
    let Some(dd05) = find(&flat(dc05), 0xDD05) else { return (None, None) };
    let Some(b136) = find(&flat(dd05), 0xB136) else { return (None, None) };
    let Some(b236) = find(&flat(b136), 0xB236) else { return (None, None) };
    let Some(t1027) = find(&flat(b236), 0x1027) else { return (None, None) };
    let sides = flat(t1027);
    let mut result = [None, None];
    for (k, side_tag) in [0x1127u16, 0x1227].iter().enumerate() {
        if let Some(side) = find(&sides, *side_tag) {
            if let Some(t1327) = find(&flat(side), 0x1327) {
                if let Some(t1527) = find(&flat(t1327), 0x1527) {
                    if t1527.len() == 72 {
                        let mut m = [0.0; 9];
                        for (i, x) in m.iter_mut().enumerate() {
                            *x = f64_at(t1527, i * 8).unwrap();
                        }
                        result[k] = Some(m);
                    }
                }
            }
        }
    }
    (result[0], result[1])
}

fn decode_vff_attr_value(a438: &Node) -> AttrValue {
    let Some(child) = a438.children.first() else { return AttrValue::Null };
    let p = child.payload;
    match child.tag {
        0xAD38 => AttrValue::Str(utf8_replace(p)),
        0xAF38 | 0xA938 if p.len() == 8 => AttrValue::Float(f64_at(p, 0).unwrap()),
        0xA738 if p.len() == 4 => AttrValue::Int(i32::from_le_bytes(p.try_into().unwrap()) as i64),
        0xB438 | 0xB538 if p.len() == 24 => AttrValue::Vec3([f64_at(p, 0).unwrap(), f64_at(p, 8).unwrap(), f64_at(p, 16).unwrap()]),
        0xAE38 => AttrValue::List(child.children.iter().map(decode_vff_attr_value).collect()),
        _ => AttrValue::Null,
    }
}

fn put<V>(list: &mut Vec<(String, V)>, key: String, v: V) {
    match list.iter_mut().find(|(k, _)| *k == key) {
        Some(slot) => slot.1 = v,
        None => list.push((key, v)),
    }
}

/// `extract_attribute_dictionaries(d007)`.
fn extract_attribute_dictionaries(d007: &Node) -> AttrDict {
    let Some(dc05) = d007.children.iter().find(|c| c.tag == 0xDC05) else { return Vec::new() };
    let elements = tlv::parse(dc05.payload, 0, dc05.payload.len(), tlv::is_prop_container);
    fn extract_entries(nodes: &[Node], entries: &mut Vec<(String, AttrValue)>) {
        let mut current: Option<String> = None;
        for n in nodes {
            if n.tag == 0xB636 {
                current = Some(utf8_replace(n.payload));
            } else if n.tag == 0xA438 && current.is_some() {
                put(entries, current.take().unwrap(), decode_vff_attr_value(n));
            } else {
                extract_entries(&n.children, entries);
            }
        }
    }
    fn walk(nodes: &[Node], dicts: &mut AttrDict) {
        for (i, n) in nodes.iter().enumerate() {
            if n.tag == 0xB436 {
                let name = utf8_replace(n.payload);
                let mut entries = Vec::new();
                if let Some(next) = nodes.get(i + 1) {
                    extract_entries(&next.children, &mut entries);
                }
                put(dicts, name, entries);
            } else {
                walk(&n.children, dicts);
            }
        }
    }
    let mut dicts = Vec::new();
    walk(&elements, &mut dicts);
    dicts
}

fn extract_geometry(elements: &[Node], b: &mut Builder) -> R<()> {
    for el in elements {
        match el.tag {
            0xC409 => {
                let v_id = tlv::extract_entity_id(el)?;
                let c509 = find_child_tag(&el.children, 0xC509);
                if let (Some(v_id), Some(c)) = (v_id, c509) {
                    if c.payload.len() >= 24 {
                        b.vertices.insert(v_id, [f64_at(c.payload, 0).unwrap(), f64_at(c.payload, 8).unwrap(), f64_at(c.payload, 16).unwrap()]);
                    }
                }
            }
            0xB80B => {
                if let Some(e_id) = tlv::extract_entity_id(el)? {
                    let rd = |n: Option<&Node>| -> R<Option<i64>> {
                        match n {
                            Some(n) => var_int(n.payload, 0, n.payload.len()).map(|v| Some(v as i64)).ok_or_else(|| "IndexError".to_string()),
                            None => Ok(None),
                        }
                    };
                    let v1 = rd(find_child_tag(&el.children, 0xB90B))?;
                    let v2 = rd(find_child_tag(&el.children, 0xBA0B))?;
                    b.edges.insert(e_id, (v1, v2));
                    if let Some(d007) = el.children.iter().find(|c| c.tag == 0xD007) {
                        if let Some(d307) = d007.children.iter().find(|c| c.tag == 0xD307) {
                            if !d307.payload.is_empty() {
                                b.edge_flags.insert(e_id, d307.payload[0]);
                            }
                        }
                        if let Some(d207) = d007.children.iter().find(|c| c.tag == 0xD207) {
                            if !d207.payload.is_empty() {
                                b.edge_layers.insert(e_id, var_int(d207.payload, 0, d207.payload.len()).unwrap() as i64);
                            }
                        }
                    }
                    if let Some(bb0b) = find_child_tag(&el.children, 0xBB0B) {
                        if !bb0b.payload.is_empty() {
                            let curve = var_int(bb0b.payload, 0, bb0b.payload.len()).unwrap();
                            if curve != 0 {
                                b.edge_curves.insert(e_id, curve as i64);
                            }
                        }
                    }
                }
            }
            0xAC0D => {
                if let Some(f_id) = tlv::extract_entity_id(el)? {
                    let mut normal = [0.0, 0.0, 1.0];
                    if let Some(ad0d) = find_child_tag(&el.children, 0xAD0D) {
                        if ad0d.payload.len() >= 24 {
                            normal = [f64_at(ad0d.payload, 0).unwrap(), f64_at(ad0d.payload, 8).unwrap(), f64_at(ad0d.payload, 16).unwrap()];
                        }
                    }
                    let mut loops: Vec<Vec<CoEdge>> = Vec::new();
                    if let Some(ae0d) = find_child_tag(&el.children, 0xAE0D) {
                        let mut loop_nodes = Vec::new();
                        find_all(&ae0d.children, 0x9411, &mut loop_nodes);
                        for ln in loop_nodes {
                            let mut co_edges = Vec::new();
                            let mut co_nodes = Vec::new();
                            find_all(&ln.children, 0xA00F, &mut co_nodes);
                            for cn in co_nodes {
                                let payload = cn.payload;
                                let (mut edge_id, mut orient): (Option<u64>, Option<u64>) = (None, None);
                                let mut sub_pos = 0usize;
                                while sub_pos + 6 < payload.len() {
                                    let sub_tag = u16::from_be_bytes([payload[sub_pos], payload[sub_pos + 1]]);
                                    let sub_size = u32_at(payload, sub_pos + 2).unwrap() as usize;
                                    if sub_pos + 6 + sub_size <= payload.len() {
                                        let val = var_int(payload, sub_pos + 6, sub_size).unwrap();
                                        if sub_tag == 0xA10F {
                                            edge_id = Some(val);
                                        } else if sub_tag == 0xA20F {
                                            orient = Some(val);
                                        }
                                    }
                                    sub_pos += 6 + sub_size;
                                }
                                if let (Some(e), Some(o)) = (edge_id, orient) {
                                    co_edges.push((e as i64, if o == 0 { 1 } else { -1 }));
                                }
                            }
                            if !co_edges.is_empty() {
                                loops.push(co_edges);
                            }
                        }
                    }
                    let (mut face_mat_id, mut uv_front, mut uv_back, mut face_hidden, mut face_layer_id) = (None, None, None, false, None);
                    if let Some(d007) = el.children.iter().find(|c| c.tag == 0xD007) {
                        if let Some(d107) = d007.children.iter().find(|c| c.tag == 0xD107) {
                            face_mat_id = Some(var_int(d107.payload, 0, d107.payload.len()).unwrap() as i64);
                        }
                        if let Some(d207) = d007.children.iter().find(|c| c.tag == 0xD207) {
                            if !d207.payload.is_empty() {
                                face_layer_id = Some(var_int(d207.payload, 0, d207.payload.len()).unwrap() as i64);
                            }
                        }
                        if let Some(dc05) = d007.children.iter().find(|c| c.tag == 0xDC05) {
                            let (f, bk) = extract_uv_transforms(dc05.payload);
                            uv_front = f;
                            uv_back = bk;
                        }
                        if let Some(d307) = d007.children.iter().find(|c| c.tag == 0xD307) {
                            if !d307.payload.is_empty() {
                                face_hidden = d307.payload[0] & 1 != 0;
                            }
                        }
                    }
                    let mut back_mat_id = None;
                    if let Some(af0d) = el.children.iter().find(|c| c.tag == 0xAF0D) {
                        if !af0d.payload.is_empty() {
                            back_mat_id = Some(var_int(af0d.payload, 0, af0d.payload.len()).unwrap() as i64);
                        }
                    }
                    b.faces.insert(
                        f_id,
                        RawFace {
                            loops,
                            normal,
                            material_id: face_mat_id,
                            back_material_id: back_mat_id,
                            uv_transform: uv_front,
                            uv_transform_back: uv_back,
                            uv_projected: false,
                            uv_projected_back: false,
                            hidden: face_hidden,
                            layer: face_layer_id,
                        },
                    );
                }
            }
            0x6419 => {
                let search: &[Node] = &el.children;
                let (mut guid, mut def_idx, mut name, mut matrix) = (None, None, None, Vec::new());
                if let Some(g) = find_child_tag(search, 0x6819) {
                    if g.payload.len() == 16 {
                        guid = Some(hex_upper(g.payload));
                    }
                }
                if let Some(d) = find_child_tag(search, 0x6719) {
                    def_idx = Some(var_int(d.payload, 0, d.payload.len()).ok_or("IndexError")? as i64);
                }
                if let Some(n) = find_child_tag(search, 0x6519) {
                    name = Some(utf8_replace(n.payload));
                }
                if let Some(m) = find_child_tag(search, 0x6619) {
                    if m.payload.len() >= 104 {
                        for i in 0..13 {
                            matrix.push(f64_at(m.payload, i * 8).unwrap());
                        }
                    }
                }
                let (mut inst_mat_id, mut inst_hidden, mut inst_layer_id, mut dicts) = (None, false, None, Vec::new());
                if let Some(d007) = el.children.iter().find(|c| c.tag == 0xD007) {
                    if let Some(d107) = d007.children.iter().find(|c| c.tag == 0xD107) {
                        inst_mat_id = Some(var_int(d107.payload, 0, d107.payload.len()).unwrap() as i64);
                    }
                    if let Some(d207) = d007.children.iter().find(|c| c.tag == 0xD207) {
                        if !d207.payload.is_empty() {
                            inst_layer_id = Some(var_int(d207.payload, 0, d207.payload.len()).unwrap() as i64);
                        }
                    }
                    dicts = extract_attribute_dictionaries(d007);
                    if let Some(d307) = d007.children.iter().find(|c| c.tag == 0xD307) {
                        if !d307.payload.is_empty() {
                            inst_hidden = d307.payload[0] & 1 != 0;
                        }
                    }
                }
                let properties = dicts
                    .iter()
                    .find(|(k, _)| k == "dynamic_attributes")
                    .map(|(_, e)| e.iter().map(|(k, v)| (k.clone(), v.stringify())).collect())
                    .unwrap_or_default();
                b.instances.push(RawInst {
                    name,
                    ref_idx: def_idx,
                    ref_guid: guid,
                    matrix,
                    material_id: inst_mat_id,
                    hidden: inst_hidden,
                    layer_id: inst_layer_id,
                    properties,
                    attribute_dictionaries: dicts,
                });
            }
            _ => {
                if !el.children.is_empty() {
                    extract_geometry(&el.children, b)?;
                }
            }
        }
    }
    Ok(())
}

// ---- the record walk ---------------------------------------------------------

fn id_of_dc05(payload: &[u8]) -> R<i64> {
    if payload.starts_with(&[0xDE, 0x05]) {
        let n = u32_at(payload, 2).ok_or("struct.error")? as usize;
        Ok(var_int(payload, 6, n).ok_or("IndexError")? as i64)
    } else {
        Ok(var_int(payload, 0, payload.len()).ok_or("IndexError")? as i64)
    }
}

fn collect_layers(nodes: &[Node], rp: &mut RawParse) -> R<()> {
    for el in nodes {
        if el.tag == 0x993A {
            for child in &el.children {
                if child.tag == 0x8C3C {
                    let dc05 = find_child_tag(&child.children, 0xDC05);
                    let name_node = find_child_tag(&child.children, 0x8D3C);
                    if let (Some(dc05), Some(name_node)) = (dc05, name_node) {
                        let l_id = id_of_dc05(dc05.payload)?;
                        let l_name = utf8_replace(name_node.payload);
                        rp.layer_id_to_name.insert(l_id, l_name.clone());
                        if let Some(h) = find_child_tag(&child.children, 0x8E3C) {
                            if !h.payload.is_empty() {
                                rp.layer_hidden.insert(l_name, h.payload[0] == 1);
                            }
                        }
                    }
                }
            }
        }
        collect_layers(&el.children, rp)?;
    }
    Ok(())
}

fn collect_material_ids(nodes: &[Node], rp: &mut RawParse) -> R<()> {
    for el in nodes {
        if el.tag == 0xC832 {
            let dc05 = find_child_tag(&el.children, 0xDC05);
            let name_node = find_child_tag(&el.children, 0xCC32);
            if let (Some(dc05), Some(name_node)) = (dc05, name_node) {
                let m_id = id_of_dc05(dc05.payload)?;
                rp.material_id_to_name.insert(m_id, utf8_replace(name_node.payload));
            }
        }
        collect_material_ids(&el.children, rp)?;
    }
    Ok(())
}

fn collect_defs(nodes: &[Node], rp: &mut RawParse) -> R<()> {
    for el in nodes {
        if el.tag == 0x7C15 {
            let (mut guid, mut name, mut faces_camera, mut shadows, mut is_image) = (String::new(), None, false, false, false);
            let mut guid_set = false;
            for child in &el.children {
                match child.tag {
                    0x7D15 if child.payload.len() == 16 => {
                        guid = hex_upper(child.payload);
                        guid_set = true;
                    }
                    0x7E15 => name = Some(utf8_replace(child.payload)),
                    0x581B => {
                        let pl = child.payload;
                        let mut pos = 0usize;
                        while pos + 6 <= pl.len() {
                            let sub_tag = u16::from_be_bytes([pl[pos], pl[pos + 1]]);
                            let sub_size = u32_at(pl, pos + 2).unwrap() as usize;
                            if pos + 6 + sub_size > pl.len() {
                                break;
                            }
                            if sub_tag == 0x5D1B && sub_size >= 1 {
                                faces_camera = var_int(pl, pos + 6, sub_size).unwrap() == 1;
                            } else if sub_tag == 0x5E1B && sub_size >= 1 {
                                shadows = var_int(pl, pos + 6, sub_size).unwrap() == 1;
                            }
                            pos += 6 + sub_size;
                        }
                    }
                    0x8315 if !child.payload.is_empty() => {
                        is_image = var_int(child.payload, 0, child.payload.len()).unwrap() == 2;
                    }
                    _ => {}
                }
            }
            let _ = guid_set;
            let ent_id = tlv::extract_entity_id(el)?;
            let mut builder = Builder::default();
            extract_geometry(&el.children, &mut builder)?;
            rp.defs.insert(ent_id, RawDef { guid, name, always_faces_camera: faces_camera, shadows_face_sun: shadows, is_image, builder });
        }
        collect_defs(&el.children, rp)?;
    }
    Ok(())
}

fn scan_vertex_positions(nodes: &[Node], id2pos: &mut HashMap<Vec<u8>, [f64; 3]>) -> R<()> {
    for el in nodes {
        if el.tag == 0xC409 {
            let dc05 = find_child_tag(&el.children, 0xDC05);
            let c509 = find_child_tag(&el.children, 0xC509);
            if let (Some(dc05), Some(c509)) = (dc05, c509) {
                if c509.payload.len() == 24 {
                    let p = dc05.payload;
                    let idb: &[u8] = if p.starts_with(&[0xDE, 0x05]) {
                        let idlen = u32_at(p, 2).ok_or("struct.error")? as usize;
                        p.get(6..).map(|r| &r[..idlen.min(r.len())]).unwrap_or(&[])
                    } else {
                        p
                    };
                    id2pos.insert(idb.to_vec(), [f64_at(c509.payload, 0).unwrap(), f64_at(c509.payload, 8).unwrap(), f64_at(c509.payload, 16).unwrap()]);
                }
            }
        }
        if !el.children.is_empty() {
            scan_vertex_positions(&el.children, id2pos)?;
        }
    }
    Ok(())
}

pub type Mat13 = [f64; 13];

/// `multiply_matrices(parent, child)` for two present matrices; `parent`
/// absent gives the child.
pub fn multiply_matrices(parent: Option<&Mat13>, child: &Mat13) -> Mat13 {
    let Some(parent) = parent else { return *child };
    let p_r0 = [parent[0], parent[1], parent[2], parent[9]];
    let p_r1 = [parent[3], parent[4], parent[5], parent[10]];
    let p_r2 = [parent[6], parent[7], parent[8], parent[11]];
    let c_c0 = [child[0], child[3], child[6], 0.0];
    let c_c1 = [child[1], child[4], child[7], 0.0];
    let c_c2 = [child[2], child[5], child[8], 0.0];
    let c_c3 = [child[9], child[10], child[11], 1.0];
    // Python's sum(): 0 + a + b + c + d, left to right
    let dot = |r: &[f64; 4], c: &[f64; 4]| (((0.0 + r[0] * c[0]) + r[1] * c[1]) + r[2] * c[2]) + r[3] * c[3];
    let mut out = [0.0; 13];
    out[0] = dot(&p_r0, &c_c0);
    out[1] = dot(&p_r0, &c_c1);
    out[2] = dot(&p_r0, &c_c2);
    out[3] = dot(&p_r1, &c_c0);
    out[4] = dot(&p_r1, &c_c1);
    out[5] = dot(&p_r1, &c_c2);
    out[6] = dot(&p_r2, &c_c0);
    out[7] = dot(&p_r2, &c_c1);
    out[8] = dot(&p_r2, &c_c2);
    out[9] = dot(&p_r0, &c_c3);
    out[10] = dot(&p_r1, &c_c3);
    out[11] = dot(&p_r2, &c_c3);
    out[12] = parent[12] * child[12];
    out
}

/// `transform_point(p, mat)`.
pub fn transform_point(p: [f64; 3], mat: &Mat13) -> [f64; 3] {
    let [x, y, z] = p;
    [
        mat[0] * x + mat[1] * y + mat[2] * z + mat[9],
        mat[3] * x + mat[4] * y + mat[5] * z + mat[10],
        mat[6] * x + mat[7] * y + mat[8] * z + mat[11],
    ]
}

fn scan_instance_transforms(nodes: &[Node], parent: Option<Mat13>, world: &mut HashMap<Vec<u8>, Mat13>) -> R<()> {
    for el in nodes {
        if el.tag == 0x6419 {
            let d007 = find_child_tag(&el.children, 0xD007);
            let dc05 = d007.and_then(|d| find_child_tag(&d.children, 0xDC05));
            let mut iid: Option<Vec<u8>> = None;
            if let Some(dc05) = dc05 {
                let p = dc05.payload;
                let idb: &[u8] = if p.starts_with(&[0xDE, 0x05]) {
                    let n = u32_at(p, 2).ok_or("struct.error")? as usize;
                    p.get(6..).map(|r| &r[..n.min(r.len())]).unwrap_or(&[])
                } else {
                    p
                };
                iid = Some(idb.to_vec());
            }
            let m = find_child_tag(&el.children, 0x6619);
            let mat: Option<Mat13> = match m {
                Some(m) if m.payload.len() == 104 => {
                    let mut a = [0.0; 13];
                    for (i, x) in a.iter_mut().enumerate() {
                        *x = f64_at(m.payload, i * 8).unwrap();
                    }
                    Some(a)
                }
                _ => None,
            };
            let here: Option<Mat13> = match &mat {
                Some(c) => Some(multiply_matrices(parent.as_ref(), c)),
                None => parent,
            };
            if let (Some(iid), Some(h)) = (iid.filter(|i| !i.is_empty()), here) {
                world.insert(iid, h);
            }
            scan_instance_transforms(&el.children, here, world)?;
        } else if !el.children.is_empty() {
            scan_instance_transforms(&el.children, parent, world)?;
        }
    }
    Ok(())
}

// ---- dimensions and pages (flat little-endian tag runs) ---------------------------

/// `_tlv_items(buf)`: `(tag, payload)` for a buffer that is entirely records
/// with a little-endian `u16` tag, or `None`.
fn tlv_items(buf: &[u8]) -> Option<Vec<(u16, &[u8])>> {
    let mut items = Vec::new();
    let mut off = 0usize;
    let n = buf.len();
    while off < n {
        if off + 6 > n {
            return None;
        }
        let tag = u16::from_le_bytes([buf[off], buf[off + 1]]);
        let ln = u32_at(buf, off + 2)? as usize;
        if tag == 0 || off + 6 + ln > n {
            return None;
        }
        items.push((tag, &buf[off + 6..off + 6 + ln]));
        off += 6 + ln;
    }
    Some(items)
}

fn tlv_find<'a>(items: &[(u16, &'a [u8])], tag: u16) -> Option<&'a [u8]> {
    items.iter().find(|(t, _)| *t == tag).map(|(_, p)| *p)
}

fn vec3(p: &[u8]) -> Option<[f64; 3]> {
    (p.len() == 24).then(|| [f64_at(p, 0).unwrap(), f64_at(p, 8).unwrap(), f64_at(p, 16).unwrap()])
}

fn find_subslice(h: &[u8], needle: &[u8], from: usize) -> Option<usize> {
    if from >= h.len() {
        return None;
    }
    h[from..].windows(needle.len()).position(|w| w == needle).map(|i| i + from)
}

/// `_parse_dimensions(model_dat, id2pos, inst_world)`.
fn parse_dimensions(model_dat: &[u8], id2pos: &HashMap<Vec<u8>, [f64; 3]>, inst_world: &HashMap<Vec<u8>, Mat13>) -> Vec<Dimension> {
    let mut dims = Vec::new();
    let mut i = 0usize;
    let n = model_dat.len();
    let point = |block: Option<&[u8]>| -> Option<[f64; 3]> {
        let items = tlv_items(block?).unwrap_or_default();
        let blk = tlv_find(&items, 0x5208)?;
        let sub = tlv_items(blk).unwrap_or_default();
        let typ = tlv_find(&sub, 0x5209).filter(|t| t.len() == 4).map(|t| u32::from_le_bytes(t.try_into().unwrap()));
        if typ == Some(1) {
            return tlv_find(&sub, 0x520A).and_then(vec3);
        }
        let reference = tlv_find(&sub, 0x520B);
        let f53fc = reference.filter(|r| !r.is_empty()).and_then(|r| tlv_find(&tlv_items(r).unwrap_or_default(), 0x53FC));
        let fi = match f53fc {
            Some(f) if !f.is_empty() => tlv_items(f).unwrap_or_default(),
            _ => Vec::new(),
        };
        let vid = tlv_find(&fi, 0x53FD).filter(|v| !v.is_empty())?;
        let iid = tlv_find(&fi, 0x53FE);
        let local = *id2pos.get(vid)?;
        if let Some(iid) = iid {
            if !iid.is_empty() && iid[0] > 0 && (iid[0] as usize) < iid.len() {
                if let Some(w) = inst_world.get(&iid[1..1 + iid[0] as usize]) {
                    return Some(transform_point(local, w));
                }
            }
        }
        Some(local)
    };
    while let Some(j) = find_subslice(model_dat, &[0xcc, 0x5b], i) {
        i = j + 1;
        if j + 6 > n {
            continue;
        }
        let ln = u32_at(model_dat, j + 2).unwrap() as usize;
        if ln < 40 || j + 6 + ln > n {
            continue;
        }
        let Some(body) = tlv_items(&model_dat[j + 6..j + 6 + ln]) else { continue };
        if !body.iter().any(|(t, _)| *t == 0x5BCD) || !body.iter().any(|(t, _)| *t == 0x5BCE) {
            continue;
        }
        let a = point(tlv_find(&body, 0x5BCD));
        let b = point(tlv_find(&body, 0x5BCE));
        let (Some(a), Some(b)) = (a, b) else { continue };
        dims.push(Dimension {
            text: String::new(),
            hidden: false,
            a: Some(a),
            b: Some(b),
            offset: tlv_find(&body, 0x5BD2).filter(|o| o.len() == 8).map(|o| f64_at(o, 0).unwrap()).unwrap_or(0.0),
            plane_x: tlv_find(&body, 0x5BCF).and_then(vec3),
            normal: tlv_find(&body, 0x5BD0).and_then(vec3),
        });
    }
    dims
}

fn find_page_node<'n, 'a>(nodes: &'n [Node<'a>]) -> Option<&'n Node<'a>> {
    for el in nodes {
        if el.tag == 0x0702 {
            return Some(el);
        }
        if let Some(r) = find_page_node(&el.children) {
            return Some(r);
        }
    }
    None
}

/// `_parse_pages(node)`.
fn parse_pages(payload: Option<&[u8]>) -> R<Vec<RawPage>> {
    let Some(payload) = payload else { return Ok(Vec::new()) };
    let mut pages = Vec::new();
    for (t60, p60) in tlv_items(payload).unwrap_or_default() {
        if t60 != 0x6D60 {
            continue;
        }
        for (t61, p61) in tlv_items(p60).unwrap_or_default() {
            if t61 != 0x6D61 {
                continue;
            }
            for (t48, p48) in tlv_items(p61).unwrap_or_default() {
                if t48 != 0x7148 {
                    continue;
                }
                let Some(items) = tlv_items(p48) else { continue };
                let mut page = RawPage { name: String::new(), eye: None, target: None, up: None, fov: 35.0, parallel: false, ortho_height: 0.0, hidden_layer_ids: Vec::new() };
                // _tlv_items(None) is a TypeError in Python
                let head = tlv_items(tlv_find(&items, 0x6F54).ok_or("TypeError: object of type 'NoneType' has no len()")?);
                let name = head.as_deref().and_then(|h| tlv_find(h, 0x6F55));
                if let Some(nm) = name.filter(|n| !n.is_empty()) {
                    page.name = utf8_replace(nm);
                }
                let cam_wrap = tlv_items(tlv_find(&items, 0x714A).ok_or("TypeError: object of type 'NoneType' has no len()")?);
                let cam = cam_wrap.as_deref().and_then(|c| tlv_find(c, 0x34BC)).and_then(tlv_items);
                if let Some(cam) = cam.filter(|c| !c.is_empty()) {
                    page.eye = tlv_find(&cam, 0x34BD).and_then(vec3);
                    page.target = tlv_find(&cam, 0x34BE).and_then(vec3);
                    page.up = tlv_find(&cam, 0x34BF).and_then(vec3);
                    if let Some(f) = tlv_find(&cam, 0x34C4).filter(|f| f.len() == 8) {
                        page.fov = f64_at(f, 0).unwrap();
                    }
                    if let Some(flag) = tlv_find(&cam, 0x34C2) {
                        page.parallel = !flag.is_empty() && flag[0] == 0;
                    }
                    if let Some(h) = tlv_find(&cam, 0x34C3).filter(|h| h.len() == 8) {
                        page.ortho_height = f64_at(h, 0).unwrap();
                    }
                }
                if let Some(hidden) = tlv_find(&items, 0x7150) {
                    let mut off = 0;
                    while off < hidden.len() {
                        let ln = hidden[off] as usize;
                        if ln == 0 || off + 1 + ln > hidden.len() {
                            break;
                        }
                        page.hidden_layer_ids.push(var_int(hidden, off + 1, ln).unwrap() as i64);
                        off += 1 + ln;
                    }
                }
                if page.eye.is_some() && page.target.is_some() {
                    pages.push(page);
                }
            }
        }
    }
    Ok(pages)
}

// ---- materials, styles, meta -------------------------------------------------

fn read_meta_units(meta: &[u8]) -> Option<String> {
    let mut pos = 0usize;
    while pos + 6 <= meta.len() {
        let tag = &meta[pos..pos + 2];
        let size = u32_at(meta, pos + 2)? as usize;
        if pos + 6 + size > meta.len() {
            break;
        }
        if tag == [0x64, 0x00] {
            return read_meta_units(&meta[pos + 6..pos + 6 + size]);
        }
        if tag == [0x6D, 0x00] {
            return Some(utf8_replace(&meta[pos + 6..pos + 6 + size]));
        }
        pos += 6 + size;
    }
    None
}

fn extract_texture(zf: &Zip, xml_name: &str, mat_elem: &xml::Elem) -> R<Option<Texture>> {
    let Some(tex_elem) = mat_elem.child(MAT_NS, "texture") else { return Ok(None) };
    let mut filename = tex_elem.get("textureFilename").unwrap_or("").to_string();
    let scale = |k: &str| -> R<f64> {
        match tex_elem.get(k) {
            None => Ok(0.0),
            Some("") => Ok(0.0),
            Some(s) => Ok(py_float(s).unwrap_or(0.0)),
        }
    };
    let x_scale = scale("xScale")?;
    let y_scale = scale("yScale")?;
    let folder = xml_name.rsplit_once('/').map(|(f, _)| f).unwrap_or(xml_name).to_string();
    let mut data: Option<Vec<u8>> = None;
    let candidate = if filename.is_empty() { None } else { Some(format!("{folder}/{filename}")) };
    let bad = |e: SkpError| e.to_string();
    if let Some(c) = candidate.as_ref().filter(|c| zf.contains(c)) {
        validate_entry(zf, c).map_err(bad)?;
        data = Some(zf.read(c)?);
    } else {
        for entry in zf.names() {
            if entry.starts_with(&format!("{folder}/")) && entry != xml_name && !entry.to_lowercase().ends_with(".xml") {
                validate_entry(zf, entry).map_err(bad)?;
                data = Some(zf.read(entry)?);
                if filename.is_empty() {
                    filename = entry.rsplit('/').next().unwrap_or("").to_string();
                }
                break;
            }
        }
    }
    if data.is_none() {
        let img_path = tex_elem
            .child(MAT_NS, "images")
            .and_then(|i| i.child(MAT_NS, "image"))
            .and_then(|i| i.get("path"))
            .unwrap_or("")
            .trim_start_matches(['.', '/'])
            .to_string();
        for cand in [img_path.clone(), format!("{folder}/{img_path}")] {
            if !cand.is_empty() && zf.contains(&cand) {
                validate_entry(zf, &cand).map_err(bad)?;
                data = Some(zf.read(&cand)?);
                if filename.is_empty() {
                    filename = cand.rsplit('/').next().unwrap_or("").to_string();
                }
                break;
            }
        }
    }
    Ok(Some(Texture { filename, width: x_scale, height: y_scale, data: data.map(Rc::new) }))
}

fn parse_material(zf: &Zip, name: &str, rp: &mut RawParse) -> R<()> {
    validate_entry(zf, name).map_err(|e| e.to_string())?;
    let xml_data = zf.read(name)?;
    let root = xml::parse(&xml_data)?;
    let Some(mat_elem) = root.descendant(MAT_NS, "material") else { return Ok(()) };
    let mat_name = mat_elem.get("name").unwrap_or("unknown").to_string();
    let channel = |k: &str| -> R<i64> {
        match mat_elem.get(k) {
            None => Ok(128),
            Some(s) => py_int(s).ok_or_else(|| format!("invalid literal for int(): {s:?}")),
        }
    };
    let (r, g, b) = (channel("colorRed")?, channel("colorGreen")?, channel("colorBlue")?);
    let trans = if mat_elem.get("useTrans") == Some("1") {
        let t = match mat_elem.get("trans") {
            None => 0.0,
            Some(s) => py_float(s).ok_or_else(|| format!("could not convert string to float: {s:?}"))?,
        };
        (1.0 - t).clamp(0.0, 1.0)
    } else {
        1.0
    };
    let folder_name = name.split('/').nth(1).unwrap_or("").to_string();
    let colorized = mat_elem.get("type") == Some("2");
    let colorize_type = match mat_elem.get("colorizeType") {
        None => 0,
        Some("") => 0,
        Some(s) => py_int(s).unwrap_or(0),
    };
    let texture = extract_texture(zf, name, mat_elem)?;
    let si = rp.material_store.len();
    rp.material_store.push(RawMaterial { name: mat_name.clone(), color: [r, g, b, 255], transparency: trans, colorized, colorize_type, texture });
    rp.materials.insert(mat_name.clone(), si);
    if !folder_name.is_empty() {
        rp.materials_by_folder.insert(folder_name, si);
    }
    if let Some(layer) = mat_name.strip_prefix("Layer_") {
        rp.layer_colors.insert(layer.to_string(), [r, g, b]);
        rp.layer_hidden.insert(layer.to_string(), false);
    }
    Ok(())
}

// ---- the whole file ----------------------------------------------------------

/// `full_parse` for a VFF file. `header` is the first 512 bytes of `data`.
pub fn full_parse(data: &[u8]) -> Result<RawParse> {
    let header = &data[..data.len().min(512)];
    let mut version = "unknown".to_string();
    if let Some(second) = find_subslice(header, &[0xFF, 0xFE, 0xFF], 4).filter(|&s| s > 0) {
        let text = utf16le_ignore(&header[second + 4..]);
        if let (Some(a), Some(b)) = (text.find('{'), text.find('}')) {
            if b > a {
                version = text[a..=b].to_string();
            }
        }
    }
    let mut pk_pos = find_subslice(header, b"PK\x03\x04", 0);
    if pk_pos.is_none() {
        let chunk = &data[..data.len().min(4096)];
        pk_pos = find_subslice(chunk, b"PK\x03\x04", 0);
    }
    let Some(pk_pos) = pk_pos else { return Err(SkpError::new("No ZIP container found", "zip_extract")) };
    let zip_bytes = &data[pk_pos..];
    let zf = Zip::open(zip_bytes).map_err(|e| SkpError::new(format!("BadZipFile: {e}"), "zip_extract"))?;

    let mut rp = RawParse { version, ..Default::default() };
    let names: Vec<String> = zf.names().map(str::to_string).collect();
    for name in &names {
        if name.ends_with("material.xml") && name.starts_with("materials/") {
            // anything wrong in one material file leaves that material out
            let _ = parse_material(&zf, name, &mut rp);
        }
    }
    if zf.contains("meta/model_thumbnail.png") {
        validate_entry(&zf, "meta/model_thumbnail.png")?;
        rp.thumbnail = Some(Rc::new(read_entry(&zf, "meta/model_thumbnail.png")?));
    }
    if zf.contains("meta/meta.dat") {
        rp.units = (|| {
            validate_entry(&zf, "meta/meta.dat").ok()?;
            read_meta_units(&zf.read("meta/meta.dat").ok()?)
        })();
    }
    for name in &names {
        if !(name.starts_with("styles/") && name.ends_with("style.xml")) {
            continue;
        }
        validate_entry(&zf, name)?;
        let bytes = read_entry(&zf, name)?;
        let Ok(sroot) = xml::parse(&bytes) else { continue };
        let Some(style_el) = sroot.child(STY_NS, "style") else { continue };
        let mut colors: HashMap<&str, [i64; 3]> = HashMap::new();
        for item in style_el.children_named(STY_NS, "item") {
            let iid = item.get("id").unwrap_or("");
            if iid != "4000" && iid != "4001" {
                continue;
            }
            let Some(var) = item.child(TYP_NS, "variant") else { continue };
            let Some(text) = var.text.as_deref().filter(|t| !t.is_empty()) else { continue };
            let Some(v) = py_int(text) else { continue };
            let v = (v as i128 & 0xFFFF_FFFF) as i64;
            colors.insert(if iid == "4000" { "4000" } else { "4001" }, [(v >> 16) & 255, (v >> 8) & 255, v & 255]);
        }
        rp.styles.push(Style { name: style_el.get("name").unwrap_or("").to_string(), front_color: colors.get("4000").copied(), back_color: colors.get("4001").copied() });
    }

    validate_entry(&zf, "model.dat")?;
    let model_dat = read_entry(&zf, "model.dat")?;
    drop(zf);

    let mut vertex_positions: HashMap<Vec<u8>, [f64; 3]> = HashMap::new();
    let mut instance_world: HashMap<Vec<u8>, Mat13> = HashMap::new();
    let mut page_payload: Option<Vec<u8>> = None;

    // iter_top_level_lazy
    let mut headers = flat_headers(&model_dat, 0, model_dat.len());
    if headers.len() == 1 && headers[0].0 == 0xF401 {
        let (_, off, size) = headers[0];
        headers = flat_headers(&model_dat, off + 6, off + 6 + size);
    }
    let mut expanded = Vec::new();
    for &(tag, offset, size) in &headers {
        if tag == 0xF901 {
            if let Some(defs) = unwrap_definitions_container(&model_dat, offset, size) {
                expanded.extend(defs);
                continue;
            }
        }
        expanded.push((tag, offset, size));
    }
    let total = expanded.len();
    for (index, &(tag_hex, offset, size)) in expanded.iter().enumerate() {
        let record_end = offset + 6 + size;
        let fail = |e: String| {
            let mut err = SkpError::new(format!("Failed while processing top-level record: {e}"), "tlv_walk");
            err.record_index = Some(index);
            err.total_records = Some(total);
            err.tag = Some(format!("{tag_hex:04X}"));
            err
        };
        let nodes = tlv::try_parse(&model_dat, offset, record_end, tlv::is_container).map_err(fail)?;
        let Some(el) = nodes.first() else { continue };
        let one = std::slice::from_ref(el);
        (|| -> R<()> {
            collect_layers(one, &mut rp)?;
            collect_material_ids(one, &mut rp)?;
            collect_defs(one, &mut rp)?;
            scan_vertex_positions(one, &mut vertex_positions)?;
            scan_instance_transforms(one, None, &mut instance_world)?;
            if page_payload.is_none() {
                if let Some(p) = find_page_node(one) {
                    page_payload = Some(p.payload.to_vec());
                }
            }
            if el.tag == 0xF601 {
                extract_geometry(&el.children, &mut rp.root)?;
            }
            Ok(())
        })()
        .map_err(fail)?;
    }

    rp.layer_id_to_name.entry(1).or_insert_with(|| "Layer0".to_string());
    if !rp.layer_colors.contains_key(&"Layer0".to_string()) {
        rp.layer_colors.insert("Layer0".into(), [136, 136, 136]);
    }
    rp.layer_hidden.entry("Layer0".into()).or_insert(false);
    rp.pages = parse_pages(page_payload.as_deref()).map_err(|e| SkpError::new(e, "pages"))?;
    rp.dimensions = parse_dimensions(&model_dat, &vertex_positions, &instance_world);
    Ok(rp)
}

/// `_unwrap_definitions_container`: the `7C15` headers behind `F901`, `7017`
/// and `7117`, or `None` when the wrapper does not have that shape.
fn unwrap_definitions_container(data: &[u8], offset: usize, size: usize) -> Option<Vec<(u16, usize, usize)>> {
    let mut level = flat_headers(data, offset + 6, offset + 6 + size);
    for expected in [0x7017u16, 0x7117] {
        let (_, c_off, c_size) = *level.iter().find(|h| h.0 == expected)?;
        level = flat_headers(data, c_off + 6, c_off + 6 + c_size);
    }
    let defs: Vec<_> = level.into_iter().filter(|h| h.0 == 0x7C15).collect();
    (!defs.is_empty()).then_some(defs)
}
