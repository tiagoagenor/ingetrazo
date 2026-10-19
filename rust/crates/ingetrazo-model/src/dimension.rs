// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! Linear dimension (`core/dimension.py`): a measured annotation between two
//! world points, aligned (the line parallel to a–b) or linear (parallel to
//! a world axis).
//!
//! Every line follows the Python, float32 where Qt is float32 and float64
//! where Python mixes a QVector3D result into its own arithmetic: a
//! `dotProduct` is a float32 widened to a Python float, so `t_a - dot(...)`
//! is a double subtraction and the `e * t_b` after it rounds the double back
//! to float32, exactly as PySide does.
//!
//! Anchors are mesh-bound in Python (`VertexAnchor` holds the vertex, its
//! mesh and the chain of group transforms, and `Dimension.a` reads through
//! them live). The scene does not exist in Rust yet, so an anchor here is
//! plain data — the vertex id, the mesh and group handles — and `a`/`b` are
//! the frozen positions, which is what Python reads when an endpoint has no
//! anchor. Rebinding (`bind`, `refresh_anchors`, the live position and
//! `resolve_vertex_anchor`) moves in with the scene.

use ingetrazo_formats::pyjson::Value;
use ingetrazo_math::pyfloat::{py_max2, py_min2};
use ingetrazo_math::{Hypot, V3};

use crate::mesh::VertexId;
use crate::pyconv::{err, get, index, object, qvec3, text_or_none, truthy, v3_value, PyResult};
use crate::units::{fmt_len, ModelUnits};

/// `math.tan(math.radians(20.0))` (`_ALIGNED_CONE_DEG`), as Python computed
/// it: the bits are embedded so the result does not depend on the C library
/// of the machine running the tests.
const TAN_ALIGNED_CONE: f64 = f64::from_bits(0x3fd74b49cf3902d4);
/// `math.sin(math.radians(15.0))` (`_LINEAR_MIN_DEG`).
const SIN_LINEAR_MIN: f64 = f64::from_bits(0x3fd0907dc1930690);
/// `math.cos(math.radians(15.0))`.
const COS_LINEAR_MIN: f64 = f64::from_bits(0x3feee8dd4748bf15);

/// A world axis a linear dimension can run along.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Axis {
    X,
    Y,
    Z,
}

impl Axis {
    /// `axis if axis in _AXIS_VECTORS else None`.
    pub fn from_name(name: &str) -> Option<Axis> {
        match name {
            "x" => Some(Axis::X),
            "y" => Some(Axis::Y),
            "z" => Some(Axis::Z),
            _ => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Axis::X => "x",
            Axis::Y => "y",
            Axis::Z => "z",
        }
    }

    /// `_AXIS_VECTORS[axis]`.
    pub fn vector(self) -> V3 {
        match self {
            Axis::X => V3(1.0, 0.0, 0.0),
            Axis::Y => V3(0.0, 1.0, 0.0),
            Axis::Z => V3(0.0, 0.0, 1.0),
        }
    }
}

/// A hold on one mesh vertex (`VertexAnchor`): plain handles until the
/// scene exists — `mesh` names the mesh that registers the vertex and
/// `chain` the groups whose transforms place it in the world (empty for
/// loose geometry), both by whatever id the scene gives them.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VertexAnchor {
    pub vertex: VertexId,
    pub mesh: u32,
    pub chain: Vec<u32>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Dimension {
    pub a: V3,
    pub b: V3,
    /// Displacement from the a–b segment to the dimension line.
    pub offset: V3,
    /// `None` = aligned; `Some` = linear along that world axis.
    pub axis: Option<Axis>,
    /// Layer (tag) the annotation lives on; `None` = default layer.
    pub layer: Option<String>,
    /// Custom text; `None` shows the measured value, and `<>` inside the
    /// text stands for it.
    pub text: Option<String>,
    pub anchor_a: Option<VertexAnchor>,
    pub anchor_b: Option<VertexAnchor>,
}

/// `(u, e)` for a linear dimension along `axis` (`_linear_frame`): `u` the
/// axis and `e` the extension-line direction. `None` for an aligned
/// dimension, a degenerate segment, or a segment that runs along the axis.
fn linear_frame(a: V3, b: V3, axis: Option<Axis>, h: Hypot) -> Option<(V3, V3)> {
    let u = axis?.vector();
    let ab = b.sub(a);
    let length = ab.length(h);
    if (length as f64) < 1e-9 {
        return None;
    }
    let d = ab.div(length);
    let e = d.sub(u.mul(d.dot(u)));
    if (e.length(h) as f64) < 1e-6 {
        return None;
    }
    Some((u, e.div(e.length(h))))
}

impl Dimension {
    /// `Dimension(a, b, offset, layer, text, axis)`.
    pub fn new(a: V3, b: V3, offset: V3, layer: Option<String>, text: Option<String>, axis: Option<&str>) -> Dimension {
        Dimension {
            a,
            b,
            offset,
            axis: axis.and_then(Axis::from_name),
            layer,
            text,
            anchor_a: None,
            anchor_b: None,
        }
    }

    /// Both endpoints held by vertices.
    pub fn anchored(&self) -> bool {
        self.anchor_a.is_some() && self.anchor_b.is_some()
    }

    /// What the annotation shows: the custom text with `<>` swapped for the
    /// formatted measurement, or the measurement itself.
    pub fn display_text(&self, measured: &str) -> String {
        match &self.text {
            Some(t) if !t.is_empty() => t.replace("<>", measured),
            _ => measured.to_string(),
        }
    }

    /// Measured length (metres): the segment's, or — linear — its extent
    /// along the axis.
    pub fn value(&self, h: Hypot) -> f64 {
        match linear_frame(self.a, self.b, self.axis, h) {
            Some((u, _)) => self.b.sub(self.a).dot(u).abs() as f64,
            None => self.b.sub(self.a).length(h) as f64,
        }
    }

    /// The measurement as the model's units read it (`fmt_len(value())`).
    pub fn label(&self, h: Hypot, units: &ModelUnits) -> String {
        fmt_len(self.value(h), units)
    }

    /// The offset with any component ALONG the measured segment removed.
    pub fn perpendicular_offset(&self, h: Hypot) -> V3 {
        if let Some((_, e)) = linear_frame(self.a, self.b, self.axis, h) {
            // Linear: only the pull along the extension direction counts.
            return e.mul(self.offset.dot(e));
        }
        let ab = self.b.sub(self.a);
        let length = ab.length(h);
        if (length as f64) < 1e-9 {
            return self.offset;
        }
        let dir = ab.div(length);
        let along = self.offset.dot(dir);
        self.offset.sub(dir.mul(along))
    }

    /// The dimension line's endpoints. Aligned: `a`/`b` shifted by the
    /// offset. Linear: each endpoint pushed along the extension direction
    /// until both sit on one line parallel to the axis.
    pub fn line_points(&self, h: Hypot) -> (V3, V3) {
        if let Some((_u, e)) = linear_frame(self.a, self.b, self.axis, h) {
            let t_a = self.offset.dot(e);
            // Python floats: a double subtraction, rounded to float32 by `e * t_b`.
            let t_b = ((t_a as f64) - (self.b.sub(self.a).dot(e) as f64)) as f32;
            return (self.a.add(e.mul(t_a)), self.b.add(e.mul(t_b)));
        }
        let off = self.perpendicular_offset(h);
        (self.a.add(off), self.b.add(off))
    }

    pub fn midpoint(&self, h: Hypot) -> V3 {
        let (ap, bp) = self.line_points(h);
        ap.add(bp).mul(0.5)
    }

    /// Offset placing the dimension line through `cursor` while staying
    /// parallel to `a`–`b`.
    pub fn offset_for_cursor(a: V3, b: V3, cursor: V3, h: Hypot) -> V3 {
        let ab = b.sub(a);
        let length = ab.length(h);
        if (length as f64) < 1e-9 {
            return cursor.sub(a);
        }
        let dir = ab.div(length);
        let to_cursor = cursor.sub(a);
        let along = to_cursor.dot(dir);
        to_cursor.sub(dir.mul(along))
    }

    /// `(offset, axis)` for the cursor's position — which dimension the
    /// drafter is asking for, and where its line goes.
    pub fn placement_for_cursor(a: V3, b: V3, cursor: V3, h: Hypot) -> (V3, Option<Axis>) {
        let ab = b.sub(a);
        let length = ab.length(h);
        if (length as f64) < 1e-9 {
            return (cursor.sub(a), None);
        }
        let d = ab.div(length);
        let aligned = Dimension::offset_for_cursor(a, b, cursor, h);
        let w = cursor.sub(a.add(b).mul(0.5));
        let along = (w.dot(d) as f64).abs();
        let perp = w.sub(d.mul(w.dot(d))).length(h) as f64;
        if along <= TAN_ALIGNED_CONE * perp {
            return (aligned, None);
        }
        let (lo, hi) = (SIN_LINEAR_MIN, COS_LINEAR_MIN);
        let mut best: Option<(f64, Axis, V3)> = None; // (overshoot, axis, offset)
        let mut beyond = 0;
        for axis in [Axis::X, Axis::Y, Axis::Z] {
            let Some((u, e)) = linear_frame(a, b, Some(axis), h) else { continue };
            let along = (d.dot(u) as f64).abs();
            if along < lo || along > hi {
                continue;
            }
            let t = cursor.sub(a).dot(e);
            let span = ab.dot(e);
            let (low, high) = (py_min2(0.0, span as f64), py_max2(0.0, span as f64));
            let t64 = t as f64;
            let over = if t64 < low {
                low - t64
            } else if t64 > high {
                t64 - high
            } else {
                0.0
            };
            if over <= 1e-9 {
                continue;
            }
            beyond += 1;
            if best.is_none_or(|b| over > b.0) {
                best = Some((over, axis, e.mul(t)));
            }
        }
        match best {
            Some((_, axis, offset)) if beyond <= 1 => (offset, Some(axis)),
            _ => (aligned, None),
        }
    }

    // ---- .igz (formats/igz.py) ------------------------------------------------

    /// One entry of `payload["dimensions"]` as `save_scene` writes it: `a`,
    /// `b`, `offset`, then `axis`, `layer` and `text` when set.
    pub fn to_igz_value(&self) -> Value {
        let mut items = vec![("a", v3_value(self.a)), ("b", v3_value(self.b)), ("offset", v3_value(self.offset))];
        if let Some(axis) = self.axis {
            items.push(("axis", Value::Str(axis.name().to_string())));
        }
        if let Some(layer) = self.layer.as_ref().filter(|l| !l.is_empty()) {
            items.push(("layer", Value::Str(layer.clone())));
        }
        if let Some(text) = self.text.as_ref().filter(|t| !t.is_empty()) {
            items.push(("text", Value::Str(text.clone())));
        }
        object(items)
    }

    /// One entry of `payload["dimensions"]` as `load_into` reads it. The
    /// `dim.bind(scene)` that follows in Python (anchoring each end to the
    /// vertex under it) waits for the scene.
    pub fn from_igz_value(raw: &Value) -> PyResult<Dimension> {
        let a = qvec3(index(raw, "a")?)?;
        let b = qvec3(index(raw, "b")?)?;
        let offset = qvec3(index(raw, "offset")?)?;
        let layer = text_or_none(get(raw, "layer")?)?;
        let text = text_or_none(get(raw, "text")?)?;
        // axis=raw.get("axis") or None, then `axis in _AXIS_VECTORS`: a list
        // or a dict is unhashable, anything else that is not a name is None.
        let axis = match get(raw, "axis")? {
            Some(Value::Str(s)) => Axis::from_name(s),
            Some(Value::Array(a)) if !a.is_empty() => return err("TypeError", "unhashable type: 'list'"),
            Some(Value::Object(o)) if !o.is_empty() => return err("TypeError", "unhashable type: 'dict'"),
            Some(v) if truthy(v) => None,
            _ => None,
        };
        Ok(Dimension { a, b, offset, axis, layer, text, anchor_a: None, anchor_b: None })
    }
}
