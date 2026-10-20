// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! The cut of a section plane as geometry (`core/section.py`): the chords the
//! plane takes out of triangles ([`cut_chords`]), the filled areas they close
//! into ([`fill_groups`], [`fill_triangles`]) and the hatching of those areas
//! ([`hatch_segments`]).
//!
//! The Python runs NumPy over float64; every operation here is the same one in
//! the same order. Where NumPy hands a sum to OpenBLAS the order is that of
//! its x86-64 Haswell kernels (checked against NumPy 2.5.4, see
//! `scripts/golden/make_section_fixtures.py`):
//!
//! * `(N,3) @ (3,)` (`dgemv`): `fma(a2, b2, fma(a0, b0, a1*b1))` for N >= 2,
//!   the plain sum for N = 1 (NumPy routes a single row to `ddot`).
//! * `(N,2) @ (2,)` (`dgemv`): `fma(a0, b0, a1*b1)` for N >= 2, the plain
//!   sum for N = 1.
//! * `np.dot(x, y)` with `x` a column of an `(M,2)` array (stride 2): the
//!   strided `ddot` loop, two accumulators over blocks of four ([`ddot_strided`]).
//! * `np.dot` / `np.linalg.norm` of 3-vectors: `((0 + a0b0) + a1b1) + a2b2`.
//!
//! Every one of them starts from +0.0 (`ddot` and `dgemv` accumulate into a
//! zeroed result), so an exact negative zero comes out positive.
//!
//! `np.cos`/`np.sin` of the two hatch angles are embedded as the bits NumPy
//! produced, so they do not depend on the C library running the tests.

use ingetrazo_math::V3;

use crate::facegeom::earcut;
use crate::hlr_draw::section_loops;

/// One chord of a cut: the two ends, float32 as `cut_chords` returns them.
pub type Chord = [[f32; 3]; 2];

#[inline]
fn dot3(a: [f64; 3], b: [f64; 3]) -> f64 {
    ((0.0 + a[0] * b[0]) + a[1] * b[1]) + a[2] * b[2]
}

#[inline]
fn cross3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}

/// One row of `rows @ v` for float64 `(N,3) @ (3,)`.
#[inline]
fn gemv3(row: [f64; 3], v: [f64; 3], n_rows: usize) -> f64 {
    if n_rows == 1 {
        dot3(row, v)
    } else {
        // y starts at +0.0 and accumulates: a negative zero comes out positive.
        0.0 + row[2].mul_add(v[2], row[0].mul_add(v[0], row[1] * v[1]))
    }
}

/// One row of `rows @ v` for float64 `(N,2) @ (2,)`.
#[inline]
fn gemv2(row: [f64; 2], v: [f64; 2], n_rows: usize) -> f64 {
    if n_rows == 1 {
        (0.0 + row[0] * v[0]) + row[1] * v[1]
    } else {
        0.0 + row[0].mul_add(v[0], row[1] * v[1])
    }
}

/// `np.dot(x, y)` for two float64 vectors when one of them is strided
/// (OpenBLAS `ddot` with an increment other than 1): products in blocks of
/// four into two accumulators, the rest into the first.
pub fn ddot_strided(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len();
    let n1 = n & !3;
    let (mut t1, mut t2) = (0.0f64, 0.0f64);
    let mut i = 0;
    while i < n1 {
        let m1 = y[i] * x[i];
        let m2 = y[i + 1] * x[i + 1];
        let m3 = y[i + 2] * x[i + 2];
        let m4 = y[i + 3] * x[i + 3];
        t1 += m1 + m3;
        t2 += m2 + m4;
        i += 4;
    }
    while i < n {
        t1 += y[i] * x[i];
        i += 1;
    }
    t1 + t2
}

fn widen(p: [f32; 3]) -> [f64; 3] {
    [p[0] as f64, p[1] as f64, p[2] as f64]
}

/// `cut_chords`: the chords the plane `normal`/`point` cuts out of the
/// triangles `v0`, `v0+e1`, `v0+e2`. `None` when it misses them all.
/// `visible` leaves hidden triangles out.
pub fn cut_chords(
    v0: &[[f64; 3]],
    e1: &[[f64; 3]],
    e2: &[[f64; 3]],
    normal: V3,
    point: V3,
    visible: Option<&[bool]>,
) -> Option<Vec<Chord>> {
    let n_rows = v0.len();
    if n_rows == 0 {
        return None;
    }
    let nv = widen([normal.0, normal.1, normal.2]);
    let c = dot3(nv, widen([point.0, point.1, point.2]));
    let p0 = v0;
    let p1: Vec<[f64; 3]> = (0..n_rows).map(|i| [p0[i][0] + e1[i][0], p0[i][1] + e1[i][1], p0[i][2] + e1[i][2]]).collect();
    let p2: Vec<[f64; 3]> = (0..n_rows).map(|i| [p0[i][0] + e2[i][0], p0[i][1] + e2[i][1], p0[i][2] + e2[i][2]]).collect();
    let dist = |p: &[f64; 3]| gemv3(*p, nv, n_rows) - c;
    let mut out: Vec<Chord> = Vec::new();
    for i in 0..n_rows {
        let (d0, d1, d2) = (dist(&p0[i]), dist(&p1[i]), dist(&p2[i]));
        let edges = [(p0[i], p1[i], d0, d1), (p1[i], p2[i], d1, d2), (p2[i], p0[i], d2, d0)];
        let mut crossing: [Option<[f64; 3]>; 3] = [None; 3];
        let mut count = 0;
        for (k, (a, b, da, db)) in edges.iter().enumerate() {
            if da * db < 0.0 {
                let t = da / (da - db);
                crossing[k] = Some([a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t]);
                count += 1;
            }
        }
        if count != 2 || visible.is_some_and(|v| !v[i]) {
            continue;
        }
        let mut pts = crossing.iter().flatten();
        let (a, b) = (pts.next().expect("two"), pts.next().expect("two"));
        out.push([[a[0] as f32, a[1] as f32, a[2] as f32], [b[0] as f32, b[1] as f32, b[2] as f32]]);
    }
    if out.is_empty() {
        None
    } else {
        Some(out)
    }
}

/// `_plane_basis`: two unit vectors spanning the plane of `normal`.
pub fn plane_basis(normal: V3) -> ([f64; 3], [f64; 3]) {
    let mut n = widen([normal.0, normal.1, normal.2]);
    let norm = dot3(n, n).sqrt();
    let div = if norm == 0.0 { 1.0 } else { norm };
    for x in n.iter_mut() {
        *x /= div;
    }
    let helper = if n[0].abs() < 0.9 { [1.0, 0.0, 0.0] } else { [0.0, 1.0, 0.0] };
    let mut u = cross3(n, helper);
    let un = dot3(u, u).sqrt();
    for x in u.iter_mut() {
        *x /= un;
    }
    (u, cross3(n, u))
}

/// `_inside`: even-odd point-in-polygon for many points.
fn inside(points: &[[f64; 2]], poly: &[[f64; 2]]) -> Vec<bool> {
    let n = poly.len();
    points
        .iter()
        .map(|p| {
            let (x, y) = (p[0], p[1]);
            let mut count = 0usize;
            for j in 0..n {
                let (x1, y1) = (poly[j][0], poly[j][1]);
                let (x2, y2) = (poly[(j + 1) % n][0], poly[(j + 1) % n][1]);
                let straddle = (y1 > y) != (y2 > y);
                let dy = if y2 == y1 { 1.0 } else { y2 - y1 };
                let xint = x1 + (y - y1) * (x2 - x1) / dy;
                if straddle && x < xint {
                    count += 1;
                }
            }
            count % 2 == 1
        })
        .collect()
}

/// One filled area of a cut: an outer ring followed by its holes, in world
/// and in plane coordinates.
#[derive(Clone, Debug, PartialEq)]
pub struct FillArea {
    pub rings3: Vec<Vec<[f64; 3]>>,
    pub rings2: Vec<Vec<[f64; 2]>>,
}

/// `_fill_groups`: the filled areas of a cut and the plane basis `(u, v)`.
pub fn fill_groups(chords: &[Chord], normal: V3) -> (Vec<FillArea>, Option<([f64; 3], [f64; 3])>) {
    if chords.is_empty() {
        return (Vec::new(), None);
    }
    let segs: Vec<[[f64; 3]; 2]> = chords.iter().map(|c| [widen(c[0]), widen(c[1])]).collect();
    let rings = section_loops(&segs);
    if rings.is_empty() {
        return (Vec::new(), None);
    }
    let (u, v) = plane_basis(normal);
    let flat: Vec<Vec<[f64; 2]>> = rings
        .iter()
        .map(|r| {
            let n = r.len();
            r.iter().map(|p| [gemv3(*p, u, n), gemv3(*p, v, n)]).collect()
        })
        .collect();
    let area: Vec<f64> = flat
        .iter()
        .map(|f| {
            let n = f.len();
            let xs: Vec<f64> = f.iter().map(|p| p[0]).collect();
            let ys: Vec<f64> = f.iter().map(|p| p[1]).collect();
            let ys_roll: Vec<f64> = (0..n).map(|i| ys[(i + 1) % n]).collect();
            let xs_roll: Vec<f64> = (0..n).map(|i| xs[(i + 1) % n]).collect();
            (0.5 * (ddot_strided(&xs, &ys_roll) - ddot_strided(&ys, &xs_roll))).abs()
        })
        .collect();
    let lo: Vec<[f64; 2]> = flat.iter().map(|f| fold2(f, f64::min)).collect();
    let hi: Vec<[f64; 2]> = flat.iter().map(|f| fold2(f, f64::max)).collect();
    let count = rings.len();
    // Which rings hold ring i: a larger ring whose box holds i's box, then
    // i's vertices inside it.
    let mut holders: Vec<Vec<usize>> = vec![Vec::new(); count];
    for i in 0..count {
        for j in 0..count {
            let cand = area[j] > area[i] && lo[j][0] <= lo[i][0] && lo[j][1] <= lo[i][1] && hi[j][0] >= hi[i][0] && hi[j][1] >= hi[i][1];
            if !cand {
                continue;
            }
            let hits = inside(&flat[i], &flat[j]);
            let mean = hits.iter().filter(|&&b| b).count() as f64 / hits.len() as f64;
            if mean >= 0.9 {
                holders[i].push(j);
            }
        }
    }
    let depth: Vec<usize> = holders.iter().map(Vec::len).collect();
    // min(h, key=area): the first of the smallest.
    let parent: Vec<Option<usize>> = holders
        .iter()
        .map(|h| {
            let mut best: Option<usize> = None;
            for &j in h {
                if best.is_none_or(|b| area[j] < area[b]) {
                    best = Some(j);
                }
            }
            best
        })
        .collect();
    let mut holes_of: Vec<Vec<usize>> = vec![Vec::new(); count];
    for k in 0..count {
        if depth[k] % 2 == 1 {
            if let Some(p) = parent[k] {
                holes_of[p].push(k);
            }
        }
    }
    let mut groups = Vec::new();
    for i in 0..count {
        if depth[i] % 2 == 1 {
            continue; // a hole, cut out of its parent
        }
        let mut members = vec![i];
        members.extend(&holes_of[i]);
        groups.push(FillArea {
            rings3: members.iter().map(|&k| rings[k].clone()).collect(),
            rings2: members.iter().map(|&k| flat[k].clone()).collect(),
        });
    }
    (groups, Some((u, v)))
}

/// `f.min(axis=0)` / `f.max(axis=0)`: NumPy propagates NaN.
fn fold2(f: &[[f64; 2]], op: fn(f64, f64) -> f64) -> [f64; 2] {
    let mut acc = f[0];
    for p in &f[1..] {
        for k in 0..2 {
            acc[k] = if acc[k].is_nan() || p[k].is_nan() { f64::NAN } else { op(acc[k], p[k]) };
        }
    }
    acc
}

/// `fill_triangles`: the section fill as float32 triangles.
pub fn fill_triangles(chords: &[Chord], normal: V3) -> Vec<[[f32; 3]; 3]> {
    let (groups, _basis) = fill_groups(chords, normal);
    let mut out: Vec<[[f32; 3]; 3]> = Vec::new();
    for g in &groups {
        let mut pts2: Vec<(f64, f64)> = Vec::new();
        let mut starts: Vec<usize> = Vec::new();
        for (k, f) in g.rings2.iter().enumerate() {
            if k > 0 {
                starts.push(pts2.len());
            }
            pts2.extend(f.iter().map(|p| (p[0], p[1])));
        }
        let idx = earcut(&pts2, &starts);
        if idx.is_empty() {
            continue;
        }
        let all: Vec<[f64; 3]> = g.rings3.iter().flatten().copied().collect();
        for t in idx {
            let q = |i: usize| [all[i][0] as f32, all[i][1] as f32, all[i][2] as f32];
            out.push([q(t[0]), q(t[1]), q(t[2])]);
        }
    }
    out
}

/// `(cos, sin)` of 45 and 135 degrees as NumPy 2.5.4 returns them for
/// `np.radians(angle)`.
const HATCH_DIRS: [[f64; 2]; 2] = [
    [f64::from_bits(0x3fe6a09e667f3bcd), f64::from_bits(0x3fe6a09e667f3bcd)],
    [f64::from_bits(0xbfe6a09e667f3bcc), f64::from_bits(0x3fe6a09e667f3bcd)],
];

/// `hatch_segments`: world segments at 45 degrees in the plane (and at 135
/// too when `cross`), `spacing` apart, clipped to the filled areas.
pub fn hatch_segments(chords: &[Chord], normal: V3, spacing: f64, cross: bool) -> Vec<[[f32; 3]; 2]> {
    let (groups, basis) = fill_groups(chords, normal);
    let Some((u, v)) = basis else { return Vec::new() };
    if groups.is_empty() || spacing <= 0.0 {
        return Vec::new();
    }
    let n = cross3(u, v);
    // Every ring point lies in the plane: they share their offset along n.
    let f = dot3(groups[0].rings3[0][0], n);
    let off = [f * n[0], f * n[1], f * n[2]];
    let mut out: Vec<[[f32; 3]; 2]> = Vec::new();
    let dirs: &[[f64; 2]] = if cross { &HATCH_DIRS } else { &HATCH_DIRS[..1] };
    for d in dirs {
        let m = [-d[1], d[0]]; // across, line to line
        for g in &groups {
            // Lines at multiples of spacing from the plane's origin: neighbouring
            // areas hatch as one pattern.
            let total: usize = g.rings2.iter().map(Vec::len).sum();
            let mut smin = f64::INFINITY;
            let mut smax = f64::NEG_INFINITY;
            for p in g.rings2.iter().flatten() {
                let s = gemv2(*p, m, total);
                smin = smin.min(s);
                smax = smax.max(s);
            }
            let k0 = (smin / spacing).ceil() as i64;
            let k1 = (smax / spacing).floor() as i64;
            if k1 - k0 > 20000 {
                continue; // too fine to mean anything
            }
            for k in k0..=k1 {
                let s = k as f64 * spacing;
                let mut hits: Vec<f64> = Vec::new();
                for f in &g.rings2 {
                    let rows = f.len();
                    let sa_all: Vec<f64> = f.iter().map(|p| gemv2(*p, m, rows) - s).collect();
                    let mut sel: Vec<[f64; 2]> = Vec::new();
                    for i in 0..rows {
                        let j = (i + 1) % rows;
                        let (sa, sb) = (sa_all[i], sa_all[j]);
                        if (sa > 0.0) != (sb > 0.0) {
                            let tt = sa / (sa - sb);
                            let (a, b) = (f[i], f[j]);
                            sel.push([a[0] + (b[0] - a[0]) * tt, a[1] + (b[1] - a[1]) * tt]);
                        }
                    }
                    let cnt = sel.len();
                    hits.extend(sel.iter().map(|p| gemv2(*p, *d, cnt)));
                }
                hits.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let mut j = 0;
                while j + 1 < hits.len() {
                    let at = |h: f64| [d[0] * h + m[0] * s, d[1] * h + m[1] * s];
                    let (p0, p1) = (at(hits[j]), at(hits[j + 1]));
                    let world = |p: [f64; 2]| {
                        let c = |i: usize| ((p[0] * u[i] + p[1] * v[i]) + off[i]) as f32;
                        [c(0), c(1), c(2)]
                    };
                    out.push([world(p0), world(p1)]);
                    j += 2;
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn square(x0: f32, y0: f32, x1: f32, y1: f32) -> Vec<Chord> {
        let p = |x: f32, y: f32| [x, y, 0.0];
        vec![
            [p(x0, y0), p(x1, y0)],
            [p(x1, y0), p(x1, y1)],
            [p(x1, y1), p(x0, y1)],
            [p(x0, y1), p(x0, y0)],
        ]
    }

    #[test]
    fn a_square_fills_with_two_triangles() {
        let tris = fill_triangles(&square(0.0, 0.0, 2.0, 1.0), V3(0.0, 0.0, 1.0));
        assert_eq!(tris.len(), 2);
    }

    #[test]
    fn a_ring_inside_a_ring_is_a_hole() {
        let mut c = square(0.0, 0.0, 4.0, 4.0);
        c.extend(square(1.0, 1.0, 3.0, 3.0));
        let (groups, _) = fill_groups(&c, V3(0.0, 0.0, 1.0));
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].rings3.len(), 2);
        let h = hatch_segments(&c, V3(0.0, 0.0, 1.0), 0.5, false);
        assert!(!h.is_empty());
    }

    #[test]
    fn a_plane_through_a_triangle_leaves_one_chord() {
        let v0 = [[0.0, 0.0, -1.0]];
        let e1 = [[1.0, 0.0, 2.0]];
        let e2 = [[0.0, 1.0, 2.0]];
        let c = cut_chords(&v0, &e1, &e2, V3(0.0, 0.0, 1.0), V3(0.0, 0.0, 0.0), None).expect("cut");
        assert_eq!(c.len(), 1);
        assert!(cut_chords(&v0, &e1, &e2, V3(0.0, 0.0, 1.0), V3(0.0, 0.0, 9.0), None).is_none());
    }
}
