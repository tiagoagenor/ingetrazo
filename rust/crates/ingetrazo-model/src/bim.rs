// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! The data side of `core/bim.py`: the IFC tags that ride on faces and
//! groups, and the tables of classes, quantity-set names and budget units.
//!
//! Storage, as in Python: a loose face carries
//! `attrs["ifc"] = {"id": n, "class": ..., "name": ...}`; a group carries
//! `.ifc = {"class": ..., "name": ...}`. Geometry-dependent code
//! (`class_quantities`, `collect_objects`, the CSV) belongs with the mesh
//! operations, not here.

use ingetrazo_formats::pyjson::Value;

use crate::pyval::{self, as_int, get, int_value, s, truthy, PyResult};

/// Curated civil/architecture classes (IFC4 names).
pub const IFC_CLASSES: [&str; 16] = [
    "IfcWall",
    "IfcSlab",
    "IfcColumn",
    "IfcBeam",
    "IfcFooting",
    "IfcRoof",
    "IfcStair",
    "IfcRamp",
    "IfcDoor",
    "IfcWindow",
    "IfcRailing",
    "IfcCovering",
    "IfcPile",
    "IfcMember",
    "IfcSpace",
    "IfcReinforcingBar",
];

/// Density of reinforcing steel, kg/m3.
pub const STEEL_DENSITY: f64 = 7850.0;

/// Standard IFC4 quantity-set name per class (`BaseQuantities` otherwise).
pub const QSET_NAMES: [(&str, &str); 16] = [
    ("IfcWall", "Qto_WallBaseQuantities"),
    ("IfcSlab", "Qto_SlabBaseQuantities"),
    ("IfcColumn", "Qto_ColumnBaseQuantities"),
    ("IfcBeam", "Qto_BeamBaseQuantities"),
    ("IfcFooting", "Qto_FootingBaseQuantities"),
    ("IfcRoof", "Qto_RoofBaseQuantities"),
    ("IfcStair", "Qto_StairBaseQuantities"),
    ("IfcRamp", "Qto_RampBaseQuantities"),
    ("IfcDoor", "Qto_DoorBaseQuantities"),
    ("IfcWindow", "Qto_WindowBaseQuantities"),
    ("IfcRailing", "Qto_RailingBaseQuantities"),
    ("IfcCovering", "Qto_CoveringBaseQuantities"),
    ("IfcPile", "Qto_PileBaseQuantities"),
    ("IfcMember", "Qto_MemberBaseQuantities"),
    ("IfcSpace", "Qto_SpaceBaseQuantities"),
    ("IfcReinforcingBar", "Qto_ReinforcingElementBaseQuantities"),
];

/// Budget unit each class is measured in (`m2` otherwise).
pub const METRADO_UNIT: [(&str, &str); 16] = [
    ("IfcWall", "m2"),
    ("IfcSlab", "m2"),
    ("IfcRoof", "m2"),
    ("IfcStair", "m2"),
    ("IfcRamp", "m2"),
    ("IfcCovering", "m2"),
    ("IfcSpace", "m2"),
    ("IfcColumn", "m3"),
    ("IfcBeam", "m3"),
    ("IfcFooting", "m3"),
    ("IfcPile", "m"),
    ("IfcMember", "m"),
    ("IfcRailing", "m"),
    ("IfcDoor", "und"),
    ("IfcWindow", "und"),
    ("IfcReinforcingBar", "kg"),
];

/// `QSET_NAMES.get(class, "BaseQuantities")`.
pub fn qset_name(ifc_class: &str) -> &'static str {
    QSET_NAMES.iter().find(|(c, _)| *c == ifc_class).map_or("BaseQuantities", |(_, n)| n)
}

/// `METRADO_UNIT.get(class, "m2")`.
pub fn metrado_unit(ifc_class: &str) -> &'static str {
    METRADO_UNIT.iter().find(|(c, _)| *c == ifc_class).map_or("m2", |(_, u)| u)
}

/// `next_object_id(scene)`: a fresh BIM object id, the largest `id` among
/// the loose faces' tags plus one. `face_attrs` is every loose face's attrs
/// dict, in order.
pub fn next_object_id(face_attrs: &[Value]) -> PyResult<Value> {
    let mut top: i128 = 0;
    for attrs in face_attrs {
        if let Some(tag) = get(attrs, "ifc")? {
            if truthy(tag) {
                if let Some(id) = get(tag, "id")?.and_then(as_int) {
                    top = top.max(id);
                }
            }
        }
    }
    Ok(int_value(top + 1))
}

/// The tag `tag_faces` stores: `{"id", "class", "name"}`.
pub fn face_tag(obj_id: &Value, ifc_class: &Value, name: &Value) -> Value {
    Value::Object(vec![
        ("id".to_string(), obj_id.clone()),
        ("class".to_string(), ifc_class.clone()),
        ("name".to_string(), name.clone()),
    ])
}

/// `tag_faces`, for one face's attrs dict.
pub fn tag_face(attrs: &mut Vec<(String, Value)>, ifc_class: &Value, name: &Value, obj_id: &Value) {
    pyval::set(attrs, "ifc", face_tag(obj_id, ifc_class, name));
}

/// `untag_faces`, for one face's attrs dict.
pub fn untag_face(attrs: &mut Vec<(String, Value)>) {
    attrs.retain(|(k, _)| k != "ifc");
}

/// `tag_group`: the group's `.ifc` is `{"class", "name"}`.
pub fn group_tag(ifc_class: &Value, name: &Value) -> Value {
    Value::Object(vec![
        ("class".to_string(), ifc_class.clone()),
        ("name".to_string(), name.clone()),
    ])
}

/// The igz reader's `dict(raw["ifc"])` for a face or a group, taken only
/// when `raw.get("ifc")` is truthy; `None` otherwise. The same copy is what
/// the writer makes (`dict(ifc)`).
pub fn ifc_from_entry(raw: &Value) -> PyResult<Option<Vec<(String, Value)>>> {
    match get(raw, "ifc")? {
        Some(tag) if truthy(tag) => Ok(Some(pyval::dict(tag)?)),
        _ => Ok(None),
    }
}

/// A tag's class and name for the takeoff (`tag.get("class", "")`,
/// `tag.get("name", "")`).
pub fn tag_class_name(tag: &Value) -> PyResult<(Value, Value)> {
    Ok((pyval::get_or(tag, "class", s(""))?, pyval::get_or(tag, "name", s(""))?))
}
