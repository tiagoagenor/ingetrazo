// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! A face's normal and triangles as the exporters take them:
//! `core.mesh.Face.normal` / `Face.triangulate` and
//! `formats.meshexport.export_triangles`.

use ingetrazo_geom::triangulate::triangulate;
use ingetrazo_math::{Hypot, V3};

use crate::world::WFace;

/// A triangle below this area (m²) has three collinear corners and is left
/// out of an export.
const MIN_TRI_AREA: f64 = 1e-9;

/// `Face.normal()`: Newell's sum in double, the length in float32, and a
/// plain division (not `normalized()`, which would return a null vector
/// below 1e-5).
pub fn face_normal(outer: &[V3], h: Hypot) -> V3 {
    let count = outer.len();
    if count < 3 {
        return V3(0.0, 0.0, 1.0);
    }
    let (mut nx, mut ny, mut nz) = (0.0f64, 0.0f64, 0.0f64);
    for i in 0..count {
        let c = outer[i];
        let n = outer[(i + 1) % count];
        let (x0, y0, z0) = (c.0 as f64, c.1 as f64, c.2 as f64);
        let (x1, y1, z1) = (n.0 as f64, n.1 as f64, n.2 as f64);
        nx += (y0 - y1) * (z0 + z1);
        ny += (z0 - z1) * (x0 + x1);
        nz += (x0 - x1) * (y0 + y1);
    }
    let n = V3(nx as f32, ny as f32, nz as f32);
    let length = n.length(h);
    if (length as f64) < 1e-9 {
        return V3(0.0, 0.0, 1.0);
    }
    n.div(length)
}

/// `Face.triangulate()`.
pub fn face_triangles(f: &WFace, h: Hypot) -> Vec<[V3; 3]> {
    if f.outer.len() < 3 {
        return Vec::new();
    }
    triangulate(&f.outer, &f.holes, Some(face_normal(&f.outer, h)), h)
}

/// `export_triangles(face)`: `Face.triangulate()` without the slivers.
pub fn export_triangles(f: &WFace, h: Hypot) -> Vec<[V3; 3]> {
    face_triangles(f, h)
        .into_iter()
        .filter(|t| {
            let (a, b, c) = (t[0], t[1], t[2]);
            let ab = [b.0 as f64 - a.0 as f64, b.1 as f64 - a.1 as f64, b.2 as f64 - a.2 as f64];
            let ac = [c.0 as f64 - a.0 as f64, c.1 as f64 - a.1 as f64, c.2 as f64 - a.2 as f64];
            let cx = ab[1] * ac[2] - ab[2] * ac[1];
            let cy = ab[2] * ac[0] - ab[0] * ac[2];
            let cz = ab[0] * ac[1] - ab[1] * ac[0];
            0.5 * (cx * cx + cy * cy + cz * cz).sqrt() >= MIN_TRI_AREA
        })
        .collect()
}
