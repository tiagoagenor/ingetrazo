// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! `georef/geopath.py`: a traced terrain/road polyline. A georef entity apart
//! from the modelling engine (`Scene.geo_paths`): nodes in local scene
//! metres, open (road) or closed (lot), optionally filled with a terrain
//! surface.

use ingetrazo_formats::pyjson::Value;
use ingetrazo_math::pyfloat::py_sum;
use ingetrazo_math::{Hypot, V3};

use crate::geotiff::py_hypot2;
use crate::pyconv::{get, qvec3, text_or_empty, text_or_none, truthy, unpack, v3_value, PyResult};

/// A built terrain surface: 3D triangles in local metres.
pub type SurfaceTris = Vec<[V3; 3]>;

/// `GeoPath`.
#[derive(Clone, Debug, PartialEq)]
pub struct GeoPath {
    pub points: Vec<V3>,
    pub closed: bool,
    pub name: String,
    /// `None` (line only), `"flat"` or `"draped"`.
    pub surface: Option<String>,
    /// `_surface_tris`: the built triangles (not serialised).
    pub surface_tris: Option<SurfaceTris>,
}

impl GeoPath {
    pub fn new(points: &[V3], closed: bool, name: &str, surface: Option<&str>) -> GeoPath {
        GeoPath {
            points: points.to_vec(),
            closed,
            name: name.to_string(),
            surface: surface.map(str::to_string),
            surface_tris: None,
        }
    }

    /// `segments`: node pairs, including the closing edge if closed.
    pub fn segments(&self) -> Vec<(V3, V3)> {
        let pts = &self.points;
        let mut out: Vec<(V3, V3)> = pts.windows(2).map(|w| (w[0], w[1])).collect();
        if self.closed && pts.len() > 2 {
            out.push((pts[pts.len() - 1], pts[0]));
        }
        out
    }

    /// `length`: horizontal (XY) length along the path.
    pub fn length(&self) -> f64 {
        py_sum(
            self.segments().into_iter().map(|(a, b)| py_hypot2(b.0 as f64 - a.0 as f64, b.1 as f64 - a.1 as f64)),
            true,
        )
    }

    /// `perimeter`.
    pub fn perimeter(&self) -> f64 {
        self.length()
    }

    /// `area`: planimetric (XY) area of a closed polygon; 0 if open.
    pub fn area(&self) -> f64 {
        if !self.closed || self.points.len() < 3 {
            return 0.0;
        }
        let pts = &self.points;
        let n = pts.len();
        let s = py_sum(
            (0..n).map(|i| {
                let (p, q) = (pts[i], pts[(i + 1) % n]);
                p.0 as f64 * q.1 as f64 - q.0 as f64 * p.1 as f64
            }),
            true,
        );
        s.abs() * 0.5
    }

    /// `surface_area`: real 3D area of the built surface, or `None` without
    /// one. `h` is the platform's `std::hypot` flavour behind
    /// `QVector3D.length()`.
    pub fn surface_area(&self, h: Hypot) -> Option<f64> {
        let tris = self.surface_tris.as_ref()?;
        if tris.is_empty() {
            return None;
        }
        Some(py_sum(
            tris.iter().map(|[a, b, c]| b.sub(*a).cross(c.sub(*a)).length(h) as f64 * 0.5),
            true,
        ))
    }

    /// `profile_points`: the nodes the profile walks (closing the loop when
    /// closed).
    pub fn profile_points(&self) -> Vec<V3> {
        let mut pts = self.points.clone();
        if self.closed && pts.len() > 2 {
            pts.push(pts[0]);
        }
        pts
    }

    /// `to_dict`.
    pub fn to_dict(&self) -> Value {
        let mut entry = vec![("points".to_string(), Value::Array(self.points.iter().map(|p| v3_value(*p)).collect()))];
        if self.closed {
            entry.push(("closed".to_string(), Value::Bool(true)));
        }
        if !self.name.is_empty() {
            entry.push(("name".to_string(), Value::Str(self.name.clone())));
        }
        if let Some(s) = self.surface.as_ref().filter(|s| !s.is_empty()) {
            entry.push(("surface".to_string(), Value::Str(s.clone())));
        }
        Value::Object(entry)
    }

    /// `from_dict`.
    pub fn from_dict(data: &Value) -> PyResult<GeoPath> {
        let points = match get(data, "points")? {
            Some(v) => unpack(v)?.iter().map(qvec3).collect::<PyResult<Vec<V3>>>()?,
            None => Vec::new(),
        };
        let closed = get(data, "closed")?.map(truthy).unwrap_or(false);
        let name = text_or_empty(get(data, "name")?)?;
        let surface = text_or_none(get(data, "surface")?)?;
        Ok(GeoPath { points, closed, name, surface, surface_tris: None })
    }
}
