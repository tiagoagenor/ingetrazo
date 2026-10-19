// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! Port of `core/layers.py`: the layer (tag) table of a document and its
//! `.igz` `"layers"` block.
//!
//! A layer never owns geometry; it is a label an entity carries (a face in
//! `attrs["layer"]`, an edge in its `layer` slot, a group in `.layer`).
//! `visible` and `locked` are kept as the JSON values the file brought,
//! because Python only ever asks them for their truth.

use ingetrazo_formats::pyjson::Value;

use crate::pyval::{self, get, get_or, s, truthy, PyResult};

/// The layer everything unlabelled lives on. A plain stored name, not a
/// translated one: layer names are document data.
pub const DEFAULT_LAYER: &str = "Layer 0";

/// Where an imported photogrammetric survey lands.
pub const SURVEY_LAYER: &str = "Survey";

/// A named tag with display state.
#[derive(Clone, Debug, PartialEq)]
pub struct Layer {
    pub name: Value,
    pub visible: Value,
    pub locked: Value,
}

impl Layer {
    /// `Layer(name)`: visible and unlocked.
    pub fn new(name: &str) -> Layer {
        Layer { name: s(name), visible: Value::Bool(true), locked: Value::Bool(false) }
    }

    /// `Layer.to_dict()`: `visible` / `locked` are written only when they
    /// differ from the default, as the literal `false` / `true`.
    pub fn to_dict(&self) -> Value {
        let mut entry = vec![("name".to_string(), self.name.clone())];
        if !truthy(&self.visible) {
            entry.push(("visible".to_string(), Value::Bool(false)));
        }
        if truthy(&self.locked) {
            entry.push(("locked".to_string(), Value::Bool(true)));
        }
        Value::Object(entry)
    }

    /// `Layer.from_dict(raw)`.
    pub fn from_dict(raw: &Value) -> PyResult<Layer> {
        let name = get_or(raw, "name", s(DEFAULT_LAYER))?;
        let visible = get_or(raw, "visible", Value::Bool(true))?;
        let locked = get_or(raw, "locked", Value::Bool(false))?;
        Ok(Layer { name, visible, locked })
    }
}

/// The layers a freshly reset scene holds: just the default one.
pub fn default_layers() -> Vec<Layer> {
    vec![Layer::new(DEFAULT_LAYER)]
}

/// What the igz loader does with `payload["layers"]`: nothing (the default
/// table) when it is absent or empty; otherwise every entry through
/// `Layer.from_dict`, with the default layer inserted first when the file
/// did not list it.
pub fn layers_from_payload(payload: &Value) -> PyResult<Vec<Layer>> {
    let mut layers = default_layers();
    if let Some(raw) = get(payload, "layers")? {
        if truthy(raw) {
            layers = Vec::new();
            for r in pyval::iter(raw)? {
                layers.push(Layer::from_dict(&r)?);
            }
            let default = s(DEFAULT_LAYER);
            if !layers.iter().any(|l| pyval::eq(&l.name, &default)) {
                layers.insert(0, Layer::new(DEFAULT_LAYER));
            }
        }
    }
    Ok(layers)
}

/// What the igz writer does: the table is written only when it holds more
/// than the default layer or a layer is hidden or locked.
pub fn layers_to_payload(layers: &[Layer]) -> Option<Value> {
    let worth = layers.len() > 1 || layers.iter().any(|l| !truthy(&l.visible) || truthy(&l.locked));
    worth.then(|| Value::Array(layers.iter().map(Layer::to_dict).collect()))
}

/// `layer_of(entity)` for a face, given its attrs dict: the layer name it
/// carries (the default when unlabelled).
pub fn layer_of_attrs(attrs: &Value) -> PyResult<Value> {
    match get(attrs, "layer")? {
        Some(v) if truthy(v) => Ok(v.clone()),
        _ => Ok(s(DEFAULT_LAYER)),
    }
}

/// `layer_of(entity)` for an entity with a `.layer` slot (edge, group).
pub fn layer_of_slot(layer: Option<&Value>) -> Value {
    match layer {
        Some(v) if truthy(v) => v.clone(),
        _ => s(DEFAULT_LAYER),
    }
}

/// `assign_layer(face, name)`: the default name clears the label.
pub fn assign_layer_attrs(attrs: &mut Vec<(String, Value)>, name: &Value) {
    if pyval::eq(name, &s(DEFAULT_LAYER)) || matches!(name, Value::Null) {
        attrs.retain(|(k, _)| k != "layer");
    } else {
        pyval::set(attrs, "layer", name.clone());
    }
}

/// `assign_layer(entity, name)` for a `.layer` slot: `None` clears it.
pub fn assign_layer_slot(name: &Value) -> Option<Value> {
    if pyval::eq(name, &s(DEFAULT_LAYER)) || matches!(name, Value::Null) {
        None
    } else {
        Some(name.clone())
    }
}
