// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! How a face's attrs dict (colour, texture, layer, IFC tag, opacity,
//! hidden, material name, back side) travel in the `.igz`: ports of
//! `_face_attrs_json` and `_face_attrs_from_json` in `formats/igz.py`.
//!
//! Both work on the loosely typed [`Value`] attrs Python keeps, with the same
//! `list()`, `dict()` and `float()` coercions, so a hand-edited or legacy
//! document is read and written the way Python does.

use ingetrazo_formats::pyjson::Value;

use crate::pyval::{self, get, truthy, PyResult};

fn present<'a>(attrs: &'a Value, key: &str) -> PyResult<Option<&'a Value>> {
    Ok(get(attrs, key)?.filter(|v| !matches!(v, Value::Null)))
}

/// `_face_attrs_json(face)`: everything a face stores besides its loops, as
/// the JSON entry the writer merges into the face (and format 3 keeps in its
/// shared attribute table). Keys come in the writer's order.
pub fn to_entry(attrs: &Value) -> PyResult<Vec<(String, Value)>> {
    let mut entry: Vec<(String, Value)> = Vec::new();
    if let Some(color) = present(attrs, "color")? {
        entry.push(("color".to_string(), Value::Array(pyval::tuple(color)?)));
    }
    if let Some(texture) = present(attrs, "texture")? {
        entry.push(("texture".to_string(), Value::Object(pyval::dict(texture)?)));
    }
    if let Some(layer) = present(attrs, "layer")? {
        entry.push(("layer".to_string(), layer.clone()));
    }
    if let Some(ifc) = present(attrs, "ifc")? {
        entry.push(("ifc".to_string(), Value::Object(pyval::dict(ifc)?)));
    }
    if let Some(opacity) = present(attrs, "opacity")? {
        entry.push(("opacity".to_string(), Value::Float(pyval::float(opacity)?)));
    }
    if get(attrs, "hidden")?.is_some_and(truthy) {
        entry.push(("hidden".to_string(), Value::Bool(true)));
    }
    if let Some(mat) = get(attrs, "mat")?.filter(|v| truthy(v)) {
        entry.push(("mat".to_string(), mat.clone()));
    }
    match get(attrs, "back")? {
        Some(Value::Object(back)) => {
            let mut copy = Vec::with_capacity(back.len());
            for (k, v) in back {
                let v = match v {
                    Value::Object(_) => Value::Object(pyval::dict(v)?),
                    other => other.clone(),
                };
                copy.push((k.clone(), v));
            }
            entry.push(("back".to_string(), Value::Object(copy)));
        }
        Some(Value::Bool(true)) => entry.push(("back".to_string(), Value::Bool(true))),
        _ => {}
    }
    Ok(entry)
}

/// `_face_attrs_from_json(raw)`: the attrs a stored face entry gives the
/// face, or `None` when it brings none.
pub fn from_entry(raw: &Value) -> PyResult<Option<Vec<(String, Value)>>> {
    let mut attrs: Vec<(String, Value)> = Vec::new();
    if let Some(color) = present(raw, "color")? {
        attrs.push(("color".to_string(), Value::Array(pyval::tuple(color)?)));
    }
    if let Some(texture) = present(raw, "texture")? {
        attrs.push(("texture".to_string(), Value::Object(pyval::dict(texture)?)));
    }
    if let Some(layer) = get(raw, "layer")?.filter(|v| truthy(v)) {
        attrs.push(("layer".to_string(), layer.clone()));
    }
    if let Some(ifc) = get(raw, "ifc")?.filter(|v| truthy(v)) {
        attrs.push(("ifc".to_string(), Value::Object(pyval::dict(ifc)?)));
    }
    if let Some(opacity) = present(raw, "opacity")? {
        attrs.push(("opacity".to_string(), Value::Float(pyval::float(opacity)?)));
    }
    if get(raw, "hidden")?.is_some_and(truthy) {
        attrs.push(("hidden".to_string(), Value::Bool(true)));
    }
    if let Some(mat) = get(raw, "mat")?.filter(|v| truthy(v)) {
        attrs.push(("mat".to_string(), mat.clone()));
    }
    match get(raw, "back")? {
        Some(back @ Value::Object(_)) => {
            attrs.push(("back".to_string(), Value::Object(pyval::dict(back)?)));
        }
        Some(Value::Bool(true)) => attrs.push(("back".to_string(), Value::Bool(true))),
        _ => {}
    }
    Ok(if attrs.is_empty() { None } else { Some(attrs) })
}
