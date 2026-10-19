// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! `core.intersect`'s face crossing (Intersect Faces): `_Poly` built from
//! world points, `cross`, `_intervals`, `_overlap` and `_dedupe` — the
//! Python's arithmetic op for op. QVector3D work is f32 in Qt's order
//! (`facegeom`'s length/normalized/plane_axes, both fuzz branches of
//! normalized() included); Python float work is f64; a Python float
//! handed to a QVector3D operator is rounded to f32 first.

use std::collections::HashSet;

use crate::facegeom::{length, normalized, plane_axes, Hypot, V3};
use ingetrazo_math::pyfloat::{py_max2 as py_max, py_min2 as py_min};

const MIN_LEN: f64 = 1e-5;

#[inline]
fn add(a: V3, b: V3) -> V3 {
    V3(a.0 + b.0, a.1 + b.1, a.2 + b.2)
}

pub struct Poly {
    n: V3,
    d: f32,
    pub lo: [f32; 3],
    pub hi: [f32; 3],
    u: V3,
    v: V3,
    rings2d: Vec<Vec<(f64, f64)>>,
}

/// `world_polys` after the world mapping: ``rings`` the face's outer ring
/// then its holes, world points. None where the Python returns None.
pub fn poly(rings: &[&[V3]], h: Hypot) -> Option<Poly> {
    let outer = rings[0];
    let m = outer.len();
    if m < 3 {
        return None;
    }
    // _newell: Python floats, then QVector3D(x, y, z).
    let (mut x, mut y, mut z) = (0.0f64, 0.0f64, 0.0f64);
    for i in 0..m {
        let a = outer[i];
        let b = outer[(i + 1) % m];
        x += (a.1 as f64 - b.1 as f64) * (a.2 as f64 + b.2 as f64);
        y += (a.2 as f64 - b.2 as f64) * (a.0 as f64 + b.0 as f64);
        z += (a.0 as f64 - b.0 as f64) * (a.1 as f64 + b.1 as f64);
    }
    let nw = V3(x as f32, y as f32, z as f32);
    if (length(nw, h) as f64) < 1e-12 {
        return None;
    }
    let n = normalized(nw, h);
    let d = n.dot(outer[0]);
    let mut lo = [outer[0].0, outer[0].1, outer[0].2];
    let mut hi = lo;
    for p in &outer[1..] {
        // min()/max() over the list: a strictly smaller (larger) value
        // replaces the running one.
        for (k, c) in [p.0, p.1, p.2].into_iter().enumerate() {
            if c < lo[k] {
                lo[k] = c;
            }
            if c > hi[k] {
                hi[k] = c;
            }
        }
    }
    let (u, v) = plane_axes(n, h);
    let rings2d = rings
        .iter()
        .map(|r| r.iter().map(|&p| (p.dot(u) as f64, p.dot(v) as f64)).collect())
        .collect();
    Some(Poly { n, d, lo, hi, u, v, rings2d })
}

/// `_intervals(poly, origin, direction)`.
fn intervals(p: &Poly, origin: V3, dir: V3) -> Vec<(f64, f64)> {
    let ox = origin.dot(p.u) as f64;
    let oy = origin.dot(p.v) as f64;
    let dx = dir.dot(p.u) as f64;
    let dy = dir.dot(p.v) as f64;
    let mut ts: Vec<f64> = Vec::new();
    for ring in &p.rings2d {
        let n = ring.len();
        for i in 0..n {
            let (ax, ay) = ring[i];
            let (bx, by) = ring[(i + 1) % n];
            let sa = (ax - ox) * dy - (ay - oy) * dx;
            let sb = (bx - ox) * dy - (by - oy) * dx;
            if (sa >= 0.0) == (sb >= 0.0) {
                continue;
            }
            let k = sa / (sa - sb);
            let px = ax + (bx - ax) * k;
            let py = ay + (by - ay) * k;
            let den = dx * dx + dy * dy;
            ts.push(((px - ox) * dx + (py - oy) * dy) / den);
        }
    }
    // list.sort(): stable, and -0.0 == 0.0 keep their order.
    ts.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mut out = Vec::new();
    let mut i = 0;
    while i + 1 < ts.len() {
        out.push((ts[i], ts[i + 1]));
        i += 2;
    }
    out
}

/// `_overlap(xs, ys)`.
fn overlap(xs: &[(f64, f64)], ys: &[(f64, f64)]) -> Vec<(f64, f64)> {
    let (mut out, mut i, mut j) = (Vec::new(), 0, 0);
    while i < xs.len() && j < ys.len() {
        let lo = py_max(xs[i].0, ys[j].0);
        let hi = py_min(xs[i].1, ys[j].1);
        if hi > lo {
            out.push((lo, hi));
        }
        if xs[i].1 < ys[j].1 {
            i += 1;
        } else {
            j += 1;
        }
    }
    out
}

/// `cross(a, b)`: the segments where the two faces cross.
pub fn cross(a: &Poly, b: &Poly, h: Hypot, out: &mut Vec<(V3, V3)>) {
    let dir = a.n.cross(b.n);
    if (length(dir, h) as f64) < 1e-9 {
        return;
    }
    let dir = normalized(dir, h);
    let (n1, n2) = (a.n, b.n);
    let (d1, d2) = (a.d as f64, b.d as f64);
    let c = n1.dot(n2) as f64;
    let den = 1.0 - c * c;
    if den < 1e-12 {
        return;
    }
    let k1 = (d1 - d2 * c) / den;
    let k2 = (d2 - d1 * c) / den;
    let origin = add(n1.mul(k1 as f32), n2.mul(k2 as f32));
    for (t0, t1) in overlap(&intervals(a, origin, dir), &intervals(b, origin, dir)) {
        let p = add(origin, dir.mul(t0 as f32));
        let q = add(origin, dir.mul(t1 as f32));
        if (length(q.sub(p), h) as f64) > MIN_LEN {
            out.push((p, q));
        }
    }
}

/// `_dedupe(segs)`: the first segment of each rounded end pair.
pub fn dedupe(segs: Vec<(V3, V3)>) -> Vec<(V3, V3)> {
    let tol = 1e-6f64;
    let key = |p: V3| -> (i64, i64, i64) {
        (
            (p.0 as f64 / tol).round_ties_even() as i64,
            (p.1 as f64 / tol).round_ties_even() as i64,
            (p.2 as f64 / tol).round_ties_even() as i64,
        )
    };
    let mut seen = HashSet::new();
    let mut out = Vec::new();
    for (p, q) in segs {
        let (k1, k2) = (key(p), key(q));
        let k = if k1 <= k2 { (k1, k2) } else { (k2, k1) };
        if seen.insert(k) {
            out.push((p, q));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn two_squares_cross_on_a_line() {
        let h = Hypot::Msvc;
        let floor = [V3(-1.0, -1.0, 0.0), V3(1.0, -1.0, 0.0), V3(1.0, 1.0, 0.0), V3(-1.0, 1.0, 0.0)];
        let wall = [V3(0.0, -2.0, -1.0), V3(0.0, 2.0, -1.0), V3(0.0, 2.0, 1.0), V3(0.0, -2.0, 1.0)];
        let a = poly(&[&floor], h).unwrap();
        let b = poly(&[&wall], h).unwrap();
        let mut out = Vec::new();
        cross(&a, &b, h, &mut out);
        assert_eq!(out.len(), 1);
        let (p, q) = out[0];
        assert_eq!((p.0, p.2, q.0, q.2), (0.0, 0.0, 0.0, 0.0));
        assert_eq!((p.1.abs(), q.1.abs()), (1.0, 1.0));
        assert_eq!(dedupe(vec![out[0], (q, p)]).len(), 1);
    }
}
