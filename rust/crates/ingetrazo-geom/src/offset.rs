// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! `core.offset` line for line: the robust polygon offset (slide the edges,
//! mitre or bevel the corners, let the arrangement sort out the crossings,
//! keep the regions that sit `d` inside the original) and the open-chain
//! offset, plus the pieces of `core.topology` and `core.arrangement` they
//! lean on (`_offset_line_intersection`, `plane_basis`, `_point_in_polygon`).
//!
//! QVector3D is float32 (`ingetrazo_math::V3`, Qt's operation order); a
//! Python float is f64, and rounds to f32 only where it goes into a
//! QVector3D (`v * d`, `QVector3D(...)`) or a Qt method.
//!
//! The arrangement (M2/topology) and the triangulation (M1.3) are not ported
//! yet, so `offset_regions` and `interior_point` take them as closures: the
//! Python calls `planar_arrangement(segments, origin, normal)` and
//! `triangulate(loop, [], normal)`, and a closure returning `None` stands for
//! the exception the Python swallows. Everything around them is the port.

use ingetrazo_math::pyfloat::py_min2;
use ingetrazo_math::{Hypot, V3};

/// How far past the offset corner a mitre may reach before it is bevelled,
/// as a multiple of the distance.
pub const MITRE_LIMIT: f64 = 2.0;

/// The kept region must sit this fraction of `d` inside.
const KEEP: f64 = 0.75;

/// One face of the arrangement: the outer loop and its holes.
pub type Face = (Vec<V3>, Vec<Vec<V3>>);

/// `core.arrangement.plane_basis`: two orthonormal in-plane axes.
pub fn plane_basis(normal: V3, h: Hypot) -> (V3, V3) {
    let n = normal.normalized(h);
    let mut r = V3(1.0, 0.0, 0.0);
    if (r.dot(n) as f64).abs() > 0.9 {
        r = V3(0.0, 1.0, 0.0);
    }
    let u = r.sub(n.mul(r.dot(n))).normalized(h);
    let v = n.cross(u).normalized(h);
    (u, v)
}

/// `core.arrangement._point_in_polygon`: ray-cast even-odd test.
pub fn point_in_polygon(p: (f64, f64), poly: &[(f64, f64)]) -> bool {
    let (x, y) = p;
    let mut inside = false;
    let n = poly.len();
    let mut j = n.wrapping_sub(1);
    for i in 0..n {
        let (xi, yi) = poly[i];
        let (xj, yj) = poly[j];
        if ((yi > y) != (yj > y)) && (x < (xj - xi) * (y - yi) / (yj - yi + 1e-30) + xi) {
            inside = !inside;
        }
        j = i;
    }
    inside
}

/// `core.topology._offset_line_intersection`: the intersection of the
/// coplanar lines `p0 + s*u` and `p1 + t*v`; `p1` when they are (nearly)
/// parallel.
pub fn offset_line_intersection(p0: V3, u: V3, p1: V3, v: V3, n: V3) -> Option<V3> {
    let denom = u.cross(v).dot(n) as f64;
    if denom.abs() < 1e-9 {
        return Some(p1);
    }
    let s = p1.sub(p0).cross(v).dot(n) as f64 / denom;
    Some(p0.add(u.mul(s as f32)))
}

#[allow(clippy::manual_clamp)] // Python's clamp keeps NaN as NaN
pub fn dist_to_boundary(p: V3, lp: &[V3], h: Hypot) -> f64 {
    let mut best = f64::INFINITY;
    let count = lp.len();
    for i in 0..count {
        let a = lp[i];
        let ab = lp[(i + 1) % count].sub(a);
        let length2 = ab.length_squared() as f64;
        if length2 < 1e-18 {
            best = py_min2(best, p.sub(a).length(h) as f64);
            continue;
        }
        let mut t = p.sub(a).dot(ab) as f64 / length2;
        t = if t < 0.0 {
            0.0
        } else if t > 1.0 {
            1.0
        } else {
            t
        };
        best = py_min2(best, p.sub(a.add(ab.mul(t as f32))).length(h) as f64);
    }
    best
}

pub fn inside(p: V3, lp: &[V3], normal: V3, h: Hypot) -> bool {
    let (u, w) = plane_basis(normal, h);
    let origin = lp[0];
    let proj = |q: V3| -> (f64, f64) {
        let rel = q.sub(origin);
        (rel.dot(u) as f64, rel.dot(w) as f64)
    };
    let poly: Vec<(f64, f64)> = lp.iter().map(|&q| proj(q)).collect();
    point_in_polygon(proj(p), &poly)
}

/// A point genuinely inside `lp`: the centroid of its biggest triangle.
/// `triangulate(loop, normal)` is `core.triangulate.triangulate(loop, [],
/// normal)`; `None` is an exception.
pub fn interior_point<T>(lp: &[V3], normal: V3, mut triangulate: T) -> Option<V3>
where
    T: FnMut(&[V3], V3) -> Option<Vec<[V3; 3]>>,
{
    if lp.len() < 3 {
        return None;
    }
    let tris = triangulate(lp, normal)?;
    let mut best: Option<V3> = None;
    let mut best_area = 0.0f64;
    for [a, b, c] in tris {
        let area = b.sub(a).cross(c.sub(a)).length_squared() as f64;
        if area > best_area {
            best_area = area;
            best = Some(a.add(b).add(c).div(3.0f32));
        }
    }
    best
}

/// The slid edges, joined corner to corner (mitre, or bevel past the limit).
pub fn offset_segments(lp: &[V3], normal: V3, d: f64, h: Hypot) -> Vec<(V3, V3)> {
    let mut lines: Vec<(V3, V3, V3)> = Vec::new();
    let count = lp.len();
    for i in 0..count {
        let a = lp[i];
        let b = lp[(i + 1) % count];
        let edge = b.sub(a);
        if (edge.length(h) as f64) < 1e-9 {
            continue;
        }
        let direction = edge.normalized(h);
        let push = normal.cross(direction).normalized(h).mul(d as f32);
        lines.push((a.add(push), b.add(push), direction));
    }
    let total = lines.len();
    if total < 3 {
        return Vec::new();
    }
    // Two passes on purpose (see the Python): mitring while emitting would
    // leave the loop open by one corner.
    let reach = MITRE_LIMIT * d.abs() + 1e-9;
    let mut corners: Vec<Option<V3>> = Vec::with_capacity(total);
    for i in 0..total {
        let (_start, end, direction) = lines[i];
        let nxt = lines[(i + 1) % total];
        let corner = offset_line_intersection(end, direction, nxt.0, nxt.2, normal);
        corners.push(match corner {
            Some(c) if (c.sub(end).length(h) as f64) <= reach => Some(c),
            _ => None,
        });
    }
    let mut segments = Vec::new();
    for i in 0..total {
        let (start, end, _direction) = lines[i];
        let back = corners[(i + total - 1) % total];
        let here = corners[i];
        let a = back.unwrap_or(start);
        let b = here.unwrap_or(end);
        segments.push((a, b));
        if here.is_none() {
            // Bevel across the gap to the next line's own start.
            let nxt_start = lines[(i + 1) % total].0;
            segments.push((end, nxt_start));
        }
    }
    segments
}

/// Offset `lp` by `d` (positive inward) and return the regions that
/// survive. `arrange(segments, origin, normal)` is
/// `core.arrangement.planar_arrangement`; `None` is an exception.
pub fn offset_regions<A, T>(
    lp: &[V3],
    normal: V3,
    d: f64,
    h: Hypot,
    mut arrange: A,
    mut triangulate: T,
) -> Vec<Face>
where
    A: FnMut(&[(V3, V3)], V3, V3) -> Option<Vec<Face>>,
    T: FnMut(&[V3], V3) -> Option<Vec<[V3; 3]>>,
{
    if lp.len() < 3 || d.abs() < 1e-9 {
        return Vec::new();
    }
    let unit = normal.normalized(h);
    if (unit.length_squared() as f64) < 0.5 {
        return Vec::new();
    }
    let segments = offset_segments(lp, unit, d, h);
    if segments.is_empty() {
        return Vec::new();
    }
    let faces = match arrange(&segments, lp[0], unit) {
        Some(f) => f,
        None => return Vec::new(),
    };
    let mut kept = Vec::new();
    for (outer, holes) in faces {
        let point = match interior_point(&outer, unit, &mut triangulate) {
            Some(p) => p,
            None => continue,
        };
        // Signed depth: positive inside the ORIGINAL, negative outside.
        let mut depth = dist_to_boundary(point, lp, h);
        if !inside(point, lp, unit, h) {
            depth = -depth;
        }
        let drop = if d > 0.0 { depth < d * KEEP } else { depth < d };
        if drop {
            continue;
        }
        kept.push((outer, holes));
    }
    kept
}

/// Offset an OPEN polyline by `d` (the signed distance along
/// `cross(normal, direction)` of each segment). `None` when fewer than two
/// distinct points remain.
pub fn offset_chain(points: &[V3], normal: V3, d: f64, h: Hypot) -> Option<Vec<V3>> {
    let n = normal.normalized(h);
    let mut lines: Vec<(V3, V3, V3)> = Vec::new();
    for w in points.windows(2) {
        let (a, b) = (w[0], w[1]);
        let edge = b.sub(a);
        if (edge.length(h) as f64) < 1e-9 {
            continue;
        }
        let direction = edge.normalized(h);
        let push = n.cross(direction).normalized(h).mul(d as f32);
        lines.push((a.add(push), b.add(push), direction));
    }
    if lines.is_empty() {
        return None;
    }
    let reach = MITRE_LIMIT * d.abs() + 1e-9;
    let mut out = vec![lines[0].0];
    for i in 0..lines.len() - 1 {
        let (_start, end, direction) = lines[i];
        let nxt = lines[i + 1];
        let corner = offset_line_intersection(end, direction, nxt.0, nxt.2, n);
        match corner {
            Some(c) if (c.sub(end).length(h) as f64) <= reach => out.push(c),
            _ => {
                out.push(end); // bevel across the gap
                out.push(nxt.0);
            }
        }
    }
    out.push(lines[lines.len() - 1].1);
    if out.len() >= 2 {
        Some(out)
    } else {
        None
    }
}
