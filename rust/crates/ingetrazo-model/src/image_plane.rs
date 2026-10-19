// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! Imported reference images (`core/image_plane.py`): a textured rectangle
//! you trace over. Reference, never geometry — it lives beside guides and
//! survey points, not in the mesh.
//!
//! Geometry is a corner plus two edge vectors: `u` along the width and `v`
//! along the height, the length of each being the real-world size in
//! metres. No image is decoded here (`image_aspect`, which reads a file's
//! header, stays with the image loader).

use ingetrazo_formats::pyjson::Value;
use ingetrazo_math::{Hypot, V3};

use crate::pyconv::{err, get, index, object, py_float, qvec3, text_or_empty, truthy, v3_value, PyResult};

/// Imported images land on this layer rather than the default one: you
/// import a plan to draw on top of it, and need to switch it off to look at
/// what you drew.
pub const IMAGE_LAYER: &str = "Images";

#[derive(Clone, Debug, PartialEq)]
pub struct ImagePlane {
    pub path: String,
    pub origin: V3,
    pub u: V3,
    pub v: V3,
    /// Pixel height / pixel width.
    pub aspect: f64,
    pub name: String,
    pub opacity: f64,
    pub layer: Option<String>,
    pub locked: bool,
}

impl ImagePlane {
    /// `ImagePlane(path, origin, u, v, aspect, name, opacity, layer,
    /// locked)`; an aspect that is not positive is 1.
    pub fn new(path: &str, origin: V3, u: V3, v: V3, aspect: f64, name: &str, opacity: f64, layer: Option<&str>, locked: bool) -> ImagePlane {
        ImagePlane {
            path: path.to_string(),
            origin,
            u,
            v,
            aspect: if aspect > 0.0 { aspect } else { 1.0 },
            name: name.to_string(),
            opacity,
            layer: layer.map(str::to_string),
            locked,
        }
    }

    /// The four corners, counter-clockwise seen from the front face:
    /// origin, +u, +u+v, +v.
    pub fn corners(&self) -> [V3; 4] {
        let (o, u, v) = (self.origin, self.u, self.v);
        [o, o.add(u), o.add(u).add(v), o.add(v)]
    }

    pub fn center(&self) -> V3 {
        self.origin.add(self.u.add(self.v).mul(0.5))
    }

    /// Real-world width in metres (the length of `u`).
    pub fn width(&self, h: Hypot) -> f64 {
        self.u.length(h) as f64
    }

    /// Real-world height in metres (the length of `v`).
    pub fn height(&self, h: Hypot) -> f64 {
        self.v.length(h) as f64
    }

    pub fn normal(&self, h: Hypot) -> V3 {
        let n = self.u.cross(self.v);
        if (n.length(h) as f64) > 1e-12 {
            n.normalized(h)
        } else {
            V3(0.0, 0.0, 1.0)
        }
    }

    /// `(point, normal)` — what the viewport feeds the work-plane inference.
    pub fn plane(&self, h: Hypot) -> (V3, V3) {
        (self.center(), self.normal(h))
    }

    /// The four border segments as `(a, b)` pairs, for the snap engine.
    pub fn border_edges(&self) -> [(V3, V3); 4] {
        let c = self.corners();
        [(c[0], c[1]), (c[1], c[2]), (c[2], c[3]), (c[3], c[0])]
    }

    pub fn contains_uv(&self, u_frac: f64, v_frac: f64) -> bool {
        -1e-9 <= u_frac && u_frac <= 1.0 + 1e-9 && -1e-9 <= v_frac && v_frac <= 1.0 + 1e-9
    }

    /// `point` expressed as `(u, v)` fractions of the rectangle. The
    /// quotients are Python floats: a float32 dot over a float32 squared
    /// length, divided in double.
    pub fn project(&self, point: V3) -> (f64, f64) {
        let d = point.sub(self.origin);
        let lu = self.u.length_squared();
        let lv = self.v.length_squared();
        (
            if (lu as f64) > 1e-18 { d.dot(self.u) as f64 / lu as f64 } else { 0.0 },
            if (lv as f64) > 1e-18 { d.dot(self.v) as f64 / lv as f64 } else { 0.0 },
        )
    }

    /// The `(u, v)` a rescale to `width` metres would produce, anchored at
    /// `origin` and keeping the picture's proportions by default.
    pub fn scaled(&self, width: f64, keep_aspect: bool, h: Hypot) -> (V3, V3) {
        let w = self.width(h);
        if w <= 1e-12 || width <= 0.0 {
            return (self.u, self.v);
        }
        let u = self.u.mul((width / w) as f32);
        if !keep_aspect {
            return (u, self.v);
        }
        let height = self.height(h);
        if height <= 1e-12 {
            return (u, self.v);
        }
        (u, self.v.mul(((width * self.aspect) / height) as f32))
    }

    pub fn to_dict(&self) -> Value {
        let mut items = vec![
            ("texture", object(vec![("path", Value::Str(self.path.clone()))])),
            ("origin", v3_value(self.origin)),
            ("u", v3_value(self.u)),
            ("v", v3_value(self.v)),
            ("aspect", Value::Float(self.aspect)),
        ];
        if !self.name.is_empty() {
            items.push(("name", Value::Str(self.name.clone())));
        }
        if self.opacity != 1.0 {
            items.push(("opacity", Value::Float(self.opacity)));
        }
        match self.layer.as_deref() {
            Some(l) if !l.is_empty() && l != IMAGE_LAYER => items.push(("layer", Value::Str(l.to_string()))),
            None => items.push(("layer", Value::Null)),
            _ => {}
        }
        if self.locked {
            items.push(("locked", Value::Bool(true)));
        }
        object(items)
    }

    /// `ImagePlane.from_dict`. A path or a layer that is not text is
    /// `NotModelled`.
    pub fn from_dict(raw: &Value) -> PyResult<ImagePlane> {
        let tex = get(raw, "texture")?;
        let path = match tex {
            Some(t @ Value::Object(_)) => t.get("path"),
            _ => get(raw, "path")?,
        };
        let path = match path {
            None => String::new(),
            Some(Value::Str(s)) => s.clone(),
            Some(_) => return err("NotModelled", "path is not text"),
        };
        let origin = qvec3(index(raw, "origin")?)?;
        let u = qvec3(index(raw, "u")?)?;
        let v = qvec3(index(raw, "v")?)?;
        let aspect = match get(raw, "aspect")? {
            None => 1.0,
            Some(x) => py_float(x)?,
        };
        let name = text_or_empty(get(raw, "name")?)?;
        let opacity = match get(raw, "opacity")? {
            None => 1.0,
            Some(x) => py_float(x)?,
        };
        let layer = match get(raw, "layer")? {
            None => Some(IMAGE_LAYER.to_string()),
            Some(Value::Null) => None,
            Some(Value::Str(s)) => Some(s.clone()),
            Some(_) => return err("NotModelled", "layer is not text"),
        };
        let locked = get(raw, "locked")?.is_some_and(truthy);
        Ok(ImagePlane::new(&path, origin, u, v, aspect, &name, opacity, layer.as_deref(), locked))
    }
}
