// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! Port of `core/materials.py`: the named-material registry over the
//! per-face attrs, and the `.igz` `"materials"` block.
//!
//! A [`Material`] keeps what Python keeps, as loosely typed as Python keeps
//! it: `from_dict` copies `name`, `opacity` and `finish` as they come and
//! only applies `tuple()` / `dict()`, so a document with a string for a
//! colour loads (and fails, or not, at the same place Python does). The
//! exceptions are reported as [`PyError`] with the Python class name.

use ingetrazo_formats::pyjson::Value;

use crate::pyval::{self, err, get, s, truthy, PyError, PyResult};

/// A named paint recipe: what stamping this material puts on a face.
#[derive(Clone, Debug, PartialEq)]
pub struct Material {
    pub name: Value,
    /// RGB floats 0-1 (a tuple in Python), or `None` for a purely textured
    /// material.
    pub color: Option<Vec<Value>>,
    /// `{"path", "sw", "sh"}`, the `attrs["texture"]` dict.
    pub texture: Option<Vec<(String, Value)>>,
    /// 0-1 translucency, or `None` for opaque.
    pub opacity: Option<Value>,
    /// One of `core.finish.FINISHES`, or `None` = guessed from the name.
    pub finish: Option<Value>,
}

impl Material {
    /// `Material(name, color, texture, opacity)`.
    pub fn new(
        name: &str,
        color: Option<Vec<Value>>,
        texture: Option<Vec<(String, Value)>>,
        opacity: Option<Value>,
    ) -> Material {
        Material { name: s(name), color, texture, opacity, finish: None }
    }

    /// `Material.face_attrs()`: the attrs this material stamps on a face
    /// (its own name first).
    pub fn face_attrs(&self) -> PyResult<Vec<(String, Value)>> {
        let mut out = vec![("mat".to_string(), self.name.clone())];
        if let Some(c) = &self.color {
            out.push(("color".to_string(), Value::Array(c.clone())));
        }
        if let Some(t) = &self.texture {
            out.push(("texture".to_string(), Value::Object(t.clone())));
        }
        if let Some(o) = &self.opacity {
            out.push(("opacity".to_string(), Value::Float(pyval::float(o)?)));
        }
        Ok(out)
    }

    /// `Material.to_dict()`.
    pub fn to_dict(&self) -> PyResult<Value> {
        let mut entry = vec![("name".to_string(), self.name.clone())];
        if let Some(c) = &self.color {
            entry.push(("color".to_string(), Value::Array(c.clone())));
        }
        if let Some(t) = &self.texture {
            entry.push(("texture".to_string(), Value::Object(t.clone())));
        }
        if let Some(o) = &self.opacity {
            entry.push(("opacity".to_string(), Value::Float(pyval::float(o)?)));
        }
        if let Some(f) = &self.finish {
            entry.push(("finish".to_string(), f.clone()));
        }
        Ok(Value::Object(entry))
    }

    /// `Material.from_dict(raw)`.
    pub fn from_dict(raw: &Value) -> PyResult<Material> {
        let color = get(raw, "color")?;
        let name = pyval::get_or(raw, "name", s(""))?;
        let color = match color {
            Some(Value::Null) | None => None,
            Some(c) => Some(pyval::tuple(c)?),
        };
        let texture = match get(raw, "texture")? {
            Some(t) if truthy(t) => Some(pyval::dict(t)?),
            _ => None,
        };
        let opacity = match get(raw, "opacity")? {
            Some(Value::Null) | None => None,
            Some(o) => Some(o.clone()),
        };
        let finish = match get(raw, "finish")? {
            Some(f) if truthy(f) => Some(f.clone()),
            _ => None,
        };
        Ok(Material { name, color, texture, opacity, finish })
    }
}

/// `is_translucent(attrs)`. `has_cutout` answers `image_has_cutout(path)`,
/// which needs an image decoder and so lives outside the data model.
pub fn is_translucent(attrs: &Value, has_cutout: &dyn Fn(&Value) -> bool) -> PyResult<bool> {
    if !truthy(attrs) {
        return Ok(false);
    }
    let opacity = pyval::get_or(attrs, "opacity", Value::Float(1.0))?;
    if pyval::float(&opacity)? < 0.999 {
        return Ok(true);
    }
    if let Some(tex) = get(attrs, "texture")? {
        if truthy(tex) {
            if let Some(path) = get(tex, "path")? {
                if truthy(path) {
                    return Ok(has_cutout(path));
                }
            }
        }
    }
    Ok(false)
}

/// `back_is_default(attrs)`: whether the back of a face shows the style's
/// default back colour.
pub fn back_is_default(attrs: &Value, has_cutout: &dyn Fn(&Value) -> bool) -> PyResult<bool> {
    if truthy(attrs) {
        if let Some(b) = get(attrs, "back")? {
            if truthy(b) {
                return Ok(false);
            }
        }
    }
    Ok(!is_translucent(attrs, has_cutout)?)
}

/// The scene's material registry: a Python `dict` of name -> Material, in
/// insertion order, keys compared by `==` (so `1`, `1.0` and `True` are one
/// key) and unhashable names refused.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Registry {
    pub entries: Vec<(Value, Material)>,
}

impl Registry {
    pub fn new() -> Registry {
        Registry::default()
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// `name in materials`.
    pub fn contains(&self, name: &Value) -> PyResult<bool> {
        pyval::hashable(name)?;
        Ok(self.entries.iter().any(|(k, _)| pyval::eq(k, name)))
    }

    /// `materials[name]` (the caller checked `contains`).
    pub fn get(&self, name: &Value) -> Option<&Material> {
        self.entries.iter().find(|(k, _)| pyval::eq(k, name)).map(|(_, m)| m)
    }

    /// `materials[name] = mat`: an existing key keeps its place and its
    /// original spelling.
    pub fn insert(&mut self, name: Value, mat: Material) -> PyResult<()> {
        pyval::hashable(&name)?;
        match self.entries.iter_mut().find(|(k, _)| pyval::eq(k, &name)) {
            Some(slot) => slot.1 = mat,
            None => self.entries.push((name, mat)),
        }
        Ok(())
    }
}

fn same_recipe(a: &Material, b: &Material) -> bool {
    let color = match (&a.color, &b.color) {
        (None, None) => true,
        (Some(x), Some(y)) => pyval::seq_eq(x, y),
        _ => false,
    };
    let texture = match (&a.texture, &b.texture) {
        (None, None) => true,
        (Some(x), Some(y)) => {
            pyval::eq(&Value::Object(x.clone()), &Value::Object(y.clone()))
        }
        _ => false,
    };
    let opacity = match (&a.opacity, &b.opacity) {
        (None, None) => true,
        (Some(x), Some(y)) => pyval::eq(x, y),
        _ => false,
    };
    color && texture && opacity
}

/// `register(materials, mat)`: adds *mat*, deduplicating. Same name + same
/// recipe: the existing entry wins. Same name, other recipe: the new one
/// registers as `"name (2)"`, `"name (3)"`... Returns the final name.
pub fn register(materials: &mut Registry, mat: Material) -> PyResult<Value> {
    let base = if truthy(&mat.name) { mat.name.clone() } else { s("Material") };
    let mut name = base.clone();
    let mut n: u32 = 2;
    while materials.contains(&name)? {
        let other = materials.get(&name).expect("contained");
        if same_recipe(other, &mat) {
            return Ok(name);
        }
        name = Value::Str(format!("{} ({n})", pyval::str_of(&base)?));
        n += 1;
    }
    let mut mat = mat;
    if !pyval::eq(&name, &mat.name) {
        mat = Material {
            name: name.clone(),
            color: mat.color,
            texture: mat.texture,
            opacity: mat.opacity,
            finish: None,
        };
    }
    materials.insert(name.clone(), mat)?;
    Ok(name)
}

/// `has_own_material(attrs)`: painted itself, a colour or a texture of its
/// own.
pub fn has_own_material(attrs: &Value) -> PyResult<bool> {
    if !truthy(attrs) {
        return Ok(false);
    }
    let tex = get(attrs, "texture")?.is_some_and(|v| !matches!(v, Value::Null));
    if tex {
        return Ok(true);
    }
    Ok(get(attrs, "color")?.is_some_and(|v| !matches!(v, Value::Null)))
}

/// `key in material` followed by `material[key]`, for a container material
/// that should be a dict. Other types fail the way Python does.
fn material_item(material: &Value, key: &str) -> PyResult<Option<Value>> {
    match material {
        Value::Object(_) => Ok(material.get(key).cloned()),
        Value::Array(a) => {
            if a.iter().any(|v| pyval::eq(v, &s(key))) {
                err("TypeError", "list indices must be integers or slices, not str")
            } else {
                Ok(None)
            }
        }
        Value::Str(t) => {
            if t.contains(key) {
                err("TypeError", "string indices must be integers, not 'str'")
            } else {
                Ok(None)
            }
        }
        _ => err("TypeError", "argument of type is not iterable"),
    }
}

/// `effective_attrs(attrs, material)`: the attrs a face is DRAWN with inside
/// a container carrying `material`. The face's own material always wins;
/// faces on the default material take the container's, on both sides.
pub fn effective_attrs(attrs: &Value, material: &Value) -> PyResult<Value> {
    if !truthy(material) || has_own_material(attrs)? {
        return Ok(attrs.clone());
    }
    let mut merged = if truthy(attrs) { pyval::dict(attrs)? } else { Vec::new() };
    for key in ["color", "texture", "opacity", "mat"] {
        if let Some(v) = material_item(material, key)? {
            if !matches!(v, Value::Null) {
                pyval::set(&mut merged, key, v);
            }
        }
    }
    let has_back = merged.iter().find(|(k, _)| k == "back").is_some_and(|(_, v)| truthy(v));
    if !has_back {
        pyval::set(&mut merged, "back", Value::Bool(true));
    }
    Ok(Value::Object(merged))
}

/// `material_sig(material)`: the signature tuple (as an array) used in cache
/// keys, or `None` when there is no material.
pub fn material_sig(material: &Value) -> PyResult<Option<Value>> {
    if !truthy(material) {
        return Ok(None);
    }
    let tex = get(material, "texture")?;
    let color = match get(material, "color")? {
        Some(c) if !matches!(c, Value::Null) => Value::Array(pyval::tuple(c)?),
        _ => Value::Null,
    };
    let tex_sig = match tex {
        Some(t @ Value::Object(_)) => {
            let at = |k: &str| t.get(k).cloned().unwrap_or(Value::Null);
            Value::Array(vec![at("path"), at("sw"), at("sh"), at("rot")])
        }
        _ => Value::Null,
    };
    let at = |k: &str| material.get(k).cloned().unwrap_or(Value::Null);
    Ok(Some(Value::Array(vec![color, tex_sig, at("opacity"), at("mat")])))
}

/// What `formats/igz.py` does with the `"materials"` block when loading:
/// every entry through `Material.from_dict`, registered under its name when
/// the name is not empty.
pub fn registry_from_payload(payload: &Value) -> PyResult<Registry> {
    let mut reg = Registry::new();
    let raw = pyval::get_or(payload, "materials", Value::Array(Vec::new()))?;
    for item in pyval::iter(&raw)? {
        let mat = Material::from_dict(&item)?;
        if truthy(&mat.name) {
            reg.insert(mat.name.clone(), mat)?;
        }
    }
    Ok(reg)
}

/// What the igz writer does: `payload["materials"]` is the registry's
/// entries in order, and is left out when the registry is empty.
pub fn registry_to_payload(reg: &Registry) -> Result<Option<Value>, PyError> {
    if reg.is_empty() {
        return Ok(None);
    }
    let mut out = Vec::with_capacity(reg.len());
    for (_, m) in &reg.entries {
        out.push(m.to_dict()?);
    }
    Ok(Some(Value::Array(out)))
}
