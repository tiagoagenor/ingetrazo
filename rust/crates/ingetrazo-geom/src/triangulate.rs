// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! `core.triangulate.triangulate` on positions: a planar loop minus its
//! holes, as 3D triangles. A convex hole-free loop fans; anything else goes
//! through the earcut port (`facegeom::earcut`), node for node.

use ingetrazo_math::{Hypot, V3};

use crate::facegeom::{earcut, is_convex, plane_axes};

/// `core.triangulate._newell`: the Newell normal of a loop, each term in
/// double over the float32 coordinates and rounded to float32 as Python's
/// `QVector3D(...)` does; `(0, 0, 1)` when shorter than 1e-9.
pub fn newell(vs: &[V3], h: Hypot) -> V3 {
    let mut n = V3(0.0, 0.0, 0.0);
    let count = vs.len();
    for i in 0..count {
        let c = vs[i];
        let nx = vs[(i + 1) % count];
        let (cx, cy, cz) = (c.0 as f64, c.1 as f64, c.2 as f64);
        let (nxx, nxy, nxz) = (nx.0 as f64, nx.1 as f64, nx.2 as f64);
        n = n.add(V3(
            ((cy - nxy) * (cz + nxz)) as f32,
            ((cz - nxz) * (cx + nxx)) as f32,
            ((cx - nxx) * (cy + nxy)) as f32,
        ));
    }
    if (n.length(h) as f64) < 1e-9 {
        return V3(0.0, 0.0, 1.0);
    }
    n.normalized(h)
}

/// `core.triangulate.triangulate(outer, holes, normal)`.
pub fn triangulate(outer: &[V3], holes: &[Vec<V3>], normal: Option<V3>, h: Hypot) -> Vec<[V3; 3]> {
    if outer.len() < 3 {
        return Vec::new();
    }
    let holes: Vec<&Vec<V3>> = holes.iter().filter(|hl| hl.len() >= 3).collect();
    let normal = normal.unwrap_or_else(|| newell(outer, h));
    let (u, v) = plane_axes(normal, h);
    let origin = outer[0];
    let proj = |p: V3| {
        let rel = p.sub(origin);
        (rel.dot(u) as f64, rel.dot(v) as f64)
    };
    if holes.is_empty() {
        let poly: Vec<(f64, f64)> = outer.iter().map(|&p| proj(p)).collect();
        if is_convex(&poly) {
            return (1..outer.len() - 1).map(|i| [outer[0], outer[i], outer[i + 1]]).collect();
        }
    }
    let mut master: Vec<V3> = outer.to_vec();
    let mut pts: Vec<(f64, f64)> = outer.iter().map(|&p| proj(p)).collect();
    let mut hole_idx = Vec::new();
    for hl in holes {
        hole_idx.push(master.len());
        master.extend_from_slice(hl);
        pts.extend(hl.iter().map(|&p| proj(p)));
    }
    earcut(&pts, &hole_idx).into_iter().map(|[a, b, c]| [master[a], master[b], master[c]]).collect()
}
