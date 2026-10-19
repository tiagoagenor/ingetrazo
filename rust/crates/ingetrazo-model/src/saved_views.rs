// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! Saved views — "Scenes" (`core/saved_views.py`): a named snapshot of the
//! camera and of the visibility state, re-applied with one click. Pure
//! presentation data; it never owns geometry.
//!
//! Ported: the data, its `.igz` form and `from_lookat`, the importers'
//! entry. Not ported: `capture`, `recapture` and `apply`, which read and
//! write the live scene and camera, and `shown_state`/`georef_state` with
//! them; they come with the scene.
//!
//! The optional blocks (`style`, `section`, `georef`, `shadows`,
//! `hidden_shown`) are dicts the owning module knows the shape of; here they
//! are kept as the JSON objects they are, key order included.

use ingetrazo_formats::pyjson::Value;
use ingetrazo_math::pyfloat::{py_max2, py_min2};

use crate::pyconv::{dict_or_none, err, get, iter_list, object, py_float, truthy, PyErr, PyResult};

/// A Python number as the file holds it: a view loaded from a hand-edited
/// document may carry `0` where the app writes `0.0`, and it is written
/// back as it came.
#[derive(Clone, Debug, PartialEq)]
pub enum PyNum {
    /// The decimal digits of a Python int.
    Int(String),
    Float(f64),
}

impl PyNum {
    pub fn to_value(&self) -> Value {
        match self {
            PyNum::Int(s) => Value::Int(s.clone()),
            PyNum::Float(x) => Value::Float(*x),
        }
    }
}

/// `math.radians(x)`: `x * (pi / 180)`.
fn radians(x: f64) -> f64 {
    x * (std::f64::consts::PI / 180.0)
}

#[derive(Clone, Debug, PartialEq)]
pub struct SavedView {
    pub name: String,
    pub target: [PyNum; 3],
    pub distance: f64,
    pub yaw: f64,
    pub pitch: f64,
    pub fov_deg: f64,
    pub perspective: bool,
    /// Two-point perspective (verticals kept vertical).
    pub two_point: bool,
    /// Layer NAMES hidden in this view.
    pub hidden_layers: Vec<String>,
    /// Display-style snapshot (`Style.to_dict()`) or `None`.
    pub style: Option<Value>,
    /// Section state: the ACTIVE plane's uid (or `None` = no cut) and the
    /// two visibility toggles. `None` entirely = a view older than sections.
    pub section: Option<Value>,
    /// Geographic reference layers shown in this view.
    pub georef: Option<Value>,
    /// Shadow settings snapshot.
    pub shadows: Option<Value>,
    /// The objects (group uids) hidden when the scene was saved; `None` =
    /// hands-off, a list, even empty, is applied.
    pub hidden_objects: Option<Vec<Value>>,
    /// View ▸ Hidden Objects / Hidden Geometry as the scene was saved.
    pub hidden_shown: Option<Value>,
}

impl SavedView {
    /// `SavedView(name)`: every other argument at its default.
    pub fn named(name: &str) -> SavedView {
        SavedView {
            name: name.to_string(),
            target: [PyNum::Float(0.0), PyNum::Float(0.0), PyNum::Float(0.0)],
            distance: 20.0,
            yaw: -std::f64::consts::FRAC_PI_4,
            pitch: 0.5235987755982988,
            fov_deg: 45.0,
            perspective: true,
            two_point: false,
            hidden_layers: Vec::new(),
            style: None,
            section: None,
            georef: None,
            shadows: None,
            hidden_objects: None,
            hidden_shown: None,
        }
    }

    /// `SavedView(name, target, distance, yaw, pitch, fov_deg, perspective,
    /// two_point, hidden_layers, style, section, georef, shadows,
    /// hidden_objects, hidden_shown)`: an empty dict is no dict
    /// (`dict(x) if x else None`), an absent layer list is empty.
    pub fn new(
        name: &str,
        target: [PyNum; 3],
        distance: f64,
        yaw: f64,
        pitch: f64,
        fov_deg: f64,
        perspective: bool,
        two_point: bool,
        hidden_layers: Option<Vec<String>>,
        style: Option<Value>,
        section: Option<Value>,
        georef: Option<Value>,
        shadows: Option<Value>,
        hidden_objects: Option<Vec<Value>>,
        hidden_shown: Option<Value>,
    ) -> SavedView {
        let dict = |v: Option<Value>| v.filter(truthy);
        SavedView {
            name: name.to_string(),
            target,
            distance,
            yaw,
            pitch,
            fov_deg,
            perspective,
            two_point,
            hidden_layers: hidden_layers.unwrap_or_default(),
            style: dict(style),
            section: dict(section),
            georef: dict(georef),
            shadows: dict(shadows),
            hidden_objects,
            hidden_shown: dict(hidden_shown),
        }
    }

    pub fn to_dict(&self) -> Value {
        let mut items = vec![
            ("name", Value::Str(self.name.clone())),
            ("target", Value::Array(self.target.iter().map(PyNum::to_value).collect())),
            ("distance", Value::Float(self.distance)),
            ("yaw", Value::Float(self.yaw)),
            ("pitch", Value::Float(self.pitch)),
        ];
        if self.fov_deg != 45.0 {
            items.push(("fov", Value::Float(self.fov_deg)));
        }
        if !self.perspective {
            items.push(("parallel", Value::Bool(true)));
        }
        if self.two_point {
            items.push(("two_point", Value::Bool(true)));
        }
        if !self.hidden_layers.is_empty() {
            items.push(("hidden_layers", Value::Array(self.hidden_layers.iter().map(|s| Value::Str(s.clone())).collect())));
        }
        if let Some(style) = &self.style {
            items.push(("style", style.clone()));
        }
        if let Some(section) = &self.section {
            items.push(("section", section.clone()));
        }
        if let Some(georef) = &self.georef {
            items.push(("georef", georef.clone()));
        }
        if let Some(shadows) = &self.shadows {
            items.push(("shadows", shadows.clone()));
        }
        if let Some(objects) = &self.hidden_objects {
            items.push(("hidden_objects", Value::Array(objects.clone())));
        }
        if let Some(shown) = &self.hidden_shown {
            items.push(("hidden_shown", shown.clone()));
        }
        object(items)
    }

    /// `SavedView.from_dict`. Names and layer names that are not text, and
    /// target coordinates that are not numbers, are `NotModelled`.
    pub fn from_dict(raw: &Value) -> PyResult<SavedView> {
        let default_target = Value::Array(vec![Value::Float(0.0), Value::Float(0.0), Value::Float(0.0)]);
        let name = match get(raw, "name")? {
            None => "Scene".to_string(),
            Some(Value::Str(s)) => s.clone(),
            Some(_) => return err("NotModelled", "name is not text"),
        };
        // t = raw.get("target") or [0.0, 0.0, 0.0]; target=(t[0], t[1], t[2])
        let t = match get(raw, "target")? {
            Some(v) if truthy(v) => v,
            _ => &default_target,
        };
        let items = match t {
            Value::Array(a) => a,
            _ => return err("NotModelled", "target is not a list"),
        };
        if items.len() < 3 {
            return err("IndexError", "list index out of range");
        }
        let mut target = [PyNum::Float(0.0), PyNum::Float(0.0), PyNum::Float(0.0)];
        for (slot, v) in target.iter_mut().zip(items) {
            *slot = match v {
                Value::Int(s) => PyNum::Int(s.clone()),
                Value::Float(x) => PyNum::Float(*x),
                _ => return err("NotModelled", "target holds something that is not a number"),
            };
        }
        let num = |key: &str, default: f64| -> PyResult<f64> {
            match get(raw, key)? {
                None => Ok(default),
                Some(v) => py_float(v),
            }
        };
        let distance = num("distance", 20.0)?;
        let yaw = num("yaw", -std::f64::consts::FRAC_PI_4)?;
        let pitch = num("pitch", 0.5235987755982988)?;
        let fov_deg = num("fov", 45.0)?;
        let perspective = !get(raw, "parallel")?.is_some_and(truthy);
        let two_point = get(raw, "two_point")?.is_some_and(truthy);
        // list(hidden_layers or [])
        let hidden_layers = match get(raw, "hidden_layers")? {
            Some(v) if truthy(v) => match v {
                Value::Array(a) => a
                    .iter()
                    .map(|x| match x {
                        Value::Str(s) => Ok(s.clone()),
                        _ => err("NotModelled", "a layer name is not text"),
                    })
                    .collect::<PyResult<Vec<String>>>()?,
                _ => return err("NotModelled", "hidden_layers is not a list"),
            },
            _ => Vec::new(),
        };
        let style = dict_or_none(get(raw, "style")?)?;
        let section = dict_or_none(get(raw, "section")?)?;
        let georef = dict_or_none(get(raw, "georef")?)?;
        let shadows = dict_or_none(get(raw, "shadows")?)?;
        // list(hidden_objects) if hidden_objects is not None else None
        let hidden_objects = match get(raw, "hidden_objects")? {
            None | Some(Value::Null) => None,
            Some(Value::Array(a)) => Some(a.clone()),
            Some(_) => return Err(PyErr { kind: "NotModelled", msg: "hidden_objects is not a list".into() }),
        };
        let hidden_shown = dict_or_none(get(raw, "hidden_shown")?)?;
        Ok(SavedView {
            name,
            target,
            distance,
            yaw,
            pitch,
            fov_deg,
            perspective,
            two_point,
            hidden_layers,
            style,
            section,
            georef,
            shadows,
            hidden_objects,
            hidden_shown,
        })
    }
}

/// `from_lookat`: a [`SavedView`] from an eye/target/up camera (importer
/// entry). `eye`/`target`/`up` are (x, y, z) metres, Z-up. A parallel view
/// with an `ortho_height` is framed from it; a `fov_deg` whose half-angle
/// has a zero tangent is Python's `ZeroDivisionError`.
pub fn from_lookat(
    name: &str,
    eye: [f64; 3],
    target: [f64; 3],
    up: [f64; 3],
    fov_deg: f64,
    perspective: bool,
    ortho_height: Option<f64>,
    hidden_layers: Option<Vec<String>>,
) -> PyResult<SavedView> {
    let dx = eye[0] - target[0];
    let dy = eye[1] - target[1];
    let dz = eye[2] - target[2];
    let mut distance = (dx * dx + dy * dy + dz * dz).sqrt();
    if distance == 0.0 {
        distance = 1.0; // `or 1.0`
    }
    let mut pitch = py_max2(-1.0, py_min2(1.0, dz / distance)).asin();
    let limit = radians(89.0);
    let yaw = if pitch.abs() < limit - 1e-6 {
        dy.atan2(dx)
    } else if up[0] != 0.0 || up[1] != 0.0 {
        // Looking straight up/down: recover the screen rotation from the up
        // vector (at the pole the orbit camera's screen-up is -(cos yaw, sin yaw)).
        (-up[1]).atan2(-up[0])
    } else {
        radians(-90.0)
    };
    pitch = py_max2(-limit, py_min2(limit, pitch));
    if !perspective {
        if let Some(oh) = ortho_height.filter(|oh| *oh > 0.0) {
            let tan = (radians(fov_deg) / 2.0).tan();
            if tan == 0.0 {
                return err("ZeroDivisionError", "float division by zero");
            }
            distance = (oh / 2.0) / tan;
        }
    }
    let mut view = SavedView::named(name);
    view.target = [PyNum::Float(target[0]), PyNum::Float(target[1]), PyNum::Float(target[2])];
    view.distance = distance;
    view.yaw = yaw;
    view.pitch = pitch;
    view.fov_deg = fov_deg;
    view.perspective = perspective;
    view.hidden_layers = hidden_layers.unwrap_or_default();
    Ok(view)
}

/// Iterating `payload["saved_views"]` in `load_into`: every entry through
/// [`SavedView::from_dict`].
pub fn views_from_list(raw: &Value) -> PyResult<Vec<SavedView>> {
    iter_list(raw)?.iter().map(SavedView::from_dict).collect()
}
