// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! Construction guides (`core/guide.py`, Tape Measure): an infinite dashed
//! line (`point` + `direction`) or, with no direction, a lone point. Pure
//! scaffolding — it never enters the mesh.

use ingetrazo_formats::pyjson::Value;
use ingetrazo_math::{Hypot, V3};

use crate::pyconv::{get, index, object, qvec3, truthy, v3_value, PyResult};

/// Half-length used when a conceptually infinite guide line needs segment
/// form (render clipping and snap both work on segments).
pub const GUIDE_HALF_LEN: f32 = 1.0e4;

#[derive(Clone, Debug, PartialEq)]
pub struct Guide {
    pub point: V3,
    /// Unit direction; `None` for a guide point.
    pub direction: Option<V3>,
    /// A guide point pulled from a vertex keeps the vertex it was measured
    /// from, to draw the dashed guide segment.
    pub origin: Option<V3>,
}

impl Guide {
    /// `Guide(point, direction, origin)`: a direction shorter than 1e-12
    /// is no direction, the rest is normalised, and only a guide point
    /// keeps an origin.
    pub fn new(point: V3, direction: Option<V3>, origin: Option<V3>, h: Hypot) -> Guide {
        let direction = match direction {
            Some(d) if (d.length(h) as f64) > 1e-12 => Some(d.normalized(h)),
            _ => None,
        };
        let origin = if direction.is_none() { origin } else { None };
        Guide { point, direction, origin }
    }

    pub fn is_line(&self) -> bool {
        self.direction.is_some()
    }

    /// Every coordinate is a number (#723: a guide at NaN cannot be drawn
    /// or snapped to, and the file refused to save over one).
    pub fn is_finite(&self) -> bool {
        [Some(self.point), self.direction, self.origin]
            .into_iter()
            .flatten()
            .all(|v| v.0.is_finite() && v.1.is_finite() && v.2.is_finite())
    }

    /// The guide as a long finite segment (for render clipping and snap).
    pub fn segment(&self) -> (V3, V3) {
        match self.direction {
            None => (self.point, self.point),
            Some(dir) => {
                let d = dir.mul(GUIDE_HALF_LEN);
                (self.point.sub(d), self.point.add(d))
            }
        }
    }

    /// Snap-engine duck typing: the segment's `.a`.
    pub fn a(&self) -> V3 {
        self.segment().0
    }

    /// … and its `.b`.
    pub fn b(&self) -> V3 {
        self.segment().1
    }

    pub fn to_dict(&self) -> Value {
        let mut items = vec![("point", v3_value(self.point))];
        if let Some(d) = self.direction {
            items.push(("direction", v3_value(d)));
        }
        if let Some(o) = self.origin {
            items.push(("origin", v3_value(o)));
        }
        object(items)
    }

    pub fn from_dict(data: &Value, h: Hypot) -> PyResult<Guide> {
        let d = get(data, "direction")?;
        let o = get(data, "origin")?;
        let point = qvec3(index(data, "point")?)?;
        let direction = match d {
            Some(v) if truthy(v) => Some(qvec3(v)?),
            _ => None,
        };
        let origin = match o {
            Some(v) if truthy(v) => Some(qvec3(v)?),
            _ => None,
        };
        Ok(Guide::new(point, direction, origin, h))
    }
}
