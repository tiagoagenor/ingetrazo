// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! `georef/surface.py`: terrain-surface fill for a closed [`GeoPath`], built
//! from the DEM as plain triangle arrays.
//!
//! * flat: one least-squares slope plane through the polygon's terrain
//!   elevations;
//! * draped: the polygon subdivided and lifted to the DEM relief.
//!
//! Heights are ground-relative: every Z is offset by the terrain elevation at
//! the scene origin.

use std::collections::HashMap;

use ingetrazo_geom::facegeom::earcut;
use ingetrazo_math::pyfloat::py_max2;
use ingetrazo_math::V3;

use crate::datum::SceneDatum;
use crate::dem::ElevationSampler;
use crate::geopath::{GeoPath, SurfaceTris};
use crate::geotiff::py_hypot2;
use crate::pyconv::round3;

/// Subdivide draped triangles down to about this edge length.
const TARGET_EDGE_M: f64 = 12.0;
/// 4**5 = 1024 sub-triangles per earcut triangle at most.
const MAX_DEPTH: i32 = 5;

/// `ground_reference`: terrain elevation at the scene origin.
pub fn ground_reference(sampler: &dyn ElevationSampler, datum: &SceneDatum) -> Option<f64> {
    sampler.elevation_at(datum.lat, datum.lon)
}

/// `_fit_plane`: least-squares plane `z = a*x + b*y + c` through `(x, y, z)`
/// samples (normal equations, Cramer's rule).
pub fn fit_plane(pts: &[V3], zs: &[f64]) -> (f64, f64, f64) {
    let n = pts.len();
    let (mut sx, mut sy, mut sxx, mut syy, mut sxy) = (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64);
    let (mut sz, mut sxz, mut syz) = (0.0f64, 0.0f64, 0.0f64);
    for (p, &z) in pts.iter().zip(zs) {
        let (x, y) = (p.0 as f64, p.1 as f64);
        sx += x;
        sy += y;
        sxx += x * x;
        syy += y * y;
        sxy += x * y;
        sz += z;
        sxz += x * z;
        syz += y * z;
    }
    let a = [[sxx, sxy, sx], [sxy, syy, sy], [sx, sy, n as f64]];
    let b = [sxz, syz, sz];

    fn det3(m: &[[f64; 3]; 3]) -> f64 {
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
            - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    }

    let d = det3(&a);
    if d.abs() < 1e-12 {
        let avg = if n != 0 { sz / n as f64 } else { 0.0 };
        return (0.0, 0.0, avg);
    }
    let col = |i: usize| -> [[f64; 3]; 3] {
        let mut m = a;
        for r in 0..3 {
            m[r][i] = b[r];
        }
        m
    };
    (det3(&col(0)) / d, det3(&col(1)) / d, det3(&col(2)) / d)
}

fn ring_of(path: &GeoPath) -> Vec<(f64, f64)> {
    path.points.iter().map(|p| (p.0 as f64, p.1 as f64)).collect()
}

/// `build_flat_surface`: the best-fit single-slope plane over the polygon, or
/// `None` (fewer than three nodes, or the DEM not loaded yet).
pub fn build_flat_surface(path: &GeoPath, sampler: &dyn ElevationSampler, ground_ref: f64) -> Option<SurfaceTris> {
    let pts = &path.points;
    if pts.len() < 3 {
        return None;
    }
    let mut zs = Vec::with_capacity(pts.len());
    for p in pts {
        zs.push(sampler.elevation_at_local(V3(p.0, p.1, 0.0)));
    }
    if zs.iter().any(|z| z.is_none()) {
        return None;
    }
    let zs: Vec<f64> = zs.into_iter().map(|z| z.expect("checked")).collect();
    let (a, b, c) = fit_plane(pts, &zs);
    let ring = ring_of(path);
    let verts: Vec<V3> = pts
        .iter()
        .map(|p| V3(p.0, p.1, (a * p.0 as f64 + b * p.1 as f64 + c - ground_ref) as f32))
        .collect();
    Some(earcut(&ring, &[]).into_iter().map(|[i, j, k]| [verts[i], verts[j], verts[k]]).collect())
}

type P2 = (f64, f64);

/// `_subdivide`: uniform 4-way midpoint subdivision (matching edge splits
/// keep adjacent triangles watertight).
pub fn subdivide(tri: [P2; 3], depth: i32, out: &mut Vec<[P2; 3]>) {
    if depth <= 0 {
        out.push(tri);
        return;
    }
    let [(ax, ay), (bx, by), (cx, cy)] = tri;
    let ab = ((ax + bx) / 2.0, (ay + by) / 2.0);
    let bc = ((bx + cx) / 2.0, (by + cy) / 2.0);
    let ca = ((cx + ax) / 2.0, (cy + ay) / 2.0);
    subdivide([(ax, ay), ab, ca], depth - 1, out);
    subdivide([ab, (bx, by), bc], depth - 1, out);
    subdivide([ca, bc, (cx, cy)], depth - 1, out);
    subdivide([ab, bc, ca], depth - 1, out);
}

/// `build_draped_surface`: the polygon lifted to the DEM relief, or `None`
/// (fewer than three nodes, nothing to triangulate, or DEM not fully loaded).
pub fn build_draped_surface(path: &GeoPath, sampler: &dyn ElevationSampler, ground_ref: f64) -> Option<SurfaceTris> {
    if path.points.len() < 3 {
        return None;
    }
    let ring = ring_of(path);
    let base: Vec<[P2; 3]> = earcut(&ring, &[]).into_iter().map(|[i, j, k]| [ring[i], ring[j], ring[k]]).collect();
    if base.is_empty() {
        return None;
    }
    // One shared subdivision depth from the longest edge.
    let mut longest = 0.0f64;
    for [(x0, y0), (x1, y1), (x2, y2)] in &base {
        longest = py_max2(longest, py_hypot2(x1 - x0, y1 - y0));
        longest = py_max2(longest, py_hypot2(x2 - x1, y2 - y1));
        longest = py_max2(longest, py_hypot2(x0 - x2, y0 - y2));
    }
    let mut depth = 0i32;
    if longest > TARGET_EDGE_M {
        depth = MAX_DEPTH.min((longest / TARGET_EDGE_M).log2().ceil() as i32);
    }
    let mut tris2d = Vec::new();
    for tri in &base {
        subdivide(*tri, depth, &mut tris2d);
    }

    // Each unique 2D point (rounded key) is sampled once.
    let mut cache: HashMap<(u64, u64), V3> = HashMap::new();
    let key_bits = |v: f64| -> u64 {
        // -0.0 and 0.0 are one dict key in Python.
        (round3(v) + 0.0).to_bits()
    };
    let mut lift = |pt: P2| -> Option<V3> {
        let key = (key_bits(pt.0), key_bits(pt.1));
        if let Some(v) = cache.get(&key) {
            return Some(*v);
        }
        let e = sampler.elevation_at_local(V3(pt.0 as f32, pt.1 as f32, 0.0))?;
        let v = V3(pt.0 as f32, pt.1 as f32, (e - ground_ref) as f32);
        cache.insert(key, v);
        Some(v)
    };

    let mut out = Vec::with_capacity(tris2d.len());
    for [a, b, c] in tris2d {
        let (va, vb, vc) = (lift(a), lift(b), lift(c));
        match (va, vb, vc) {
            (Some(va), Some(vb), Some(vc)) => out.push([va, vb, vc]),
            _ => return None,
        }
    }
    Some(out)
}

/// `build_surface`: `path`'s surface triangles for its mode, or `None` if it
/// has none or is not ready.
pub fn build_surface(path: &GeoPath, sampler: &dyn ElevationSampler, datum: &SceneDatum) -> Option<SurfaceTris> {
    let mode = path.surface.as_deref().filter(|s| !s.is_empty())?;
    if path.points.len() < 3 {
        return None;
    }
    let ground = ground_reference(sampler, datum)?;
    match mode {
        "flat" => build_flat_surface(path, sampler, ground),
        "draped" => build_draped_surface(path, sampler, ground),
        _ => None,
    }
}
