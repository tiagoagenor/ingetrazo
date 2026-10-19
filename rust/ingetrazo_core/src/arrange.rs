// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! `core.arrangement._build_graph`'s all-pairs split: every segment's
//! endpoints and the points where the others meet it, sorted along it —
//! `_intersections` and `_on_seg` line for line, in Python's float
//! arithmetic (f64, the same operations in the same order). The graph
//! (vertex keys, adjacency sets) stays in Python.

const EPS: f64 = 1e-7;
const ON_SEG: f64 = 1e-4;

type P = [f64; 2];

/// `_on_seg`: the clamped parameter of `p` on `a`–`b`, or None.
fn on_seg(p: P, a: P, b: P) -> Option<f64> {
    let abx = b[0] - a[0];
    let aby = b[1] - a[1];
    let l2 = abx * abx + aby * aby;
    if l2 < 1e-12 {
        return None;
    }
    let t = ((p[0] - a[0]) * abx + (p[1] - a[1]) * aby) / l2;
    let px = a[0] + t * abx;
    let py = a[1] + t * aby;
    let dx = p[0] - px;
    let dy = p[1] - py;
    if dx * dx + dy * dy > ON_SEG * ON_SEG {
        return None;
    }
    if t < -EPS || t > 1.0 + EPS {
        return None;
    }
    // min(1.0, max(0.0, t)): Python keeps the first argument on a tie.
    let m = if t > 0.0 { t } else { 0.0 };
    Some(if m < 1.0 { m } else { 1.0 })
}

/// `_intersections(a, b, c, d)`, appended to `out`.
fn intersections(a: P, b: P, c: P, d: P, out: &mut Vec<P>) {
    let rx = b[0] - a[0];
    let ry = b[1] - a[1];
    let sx = d[0] - c[0];
    let sy = d[1] - c[1];
    let rxs = rx * sy - ry * sx;
    let qpx = c[0] - a[0];
    let qpy = c[1] - a[1];
    if rxs.abs() > 1e-9 {
        let t = (qpx * sy - qpy * sx) / rxs;
        let u = (qpx * ry - qpy * rx) / rxs;
        if -EPS <= t && t <= 1.0 + EPS && -EPS <= u && u <= 1.0 + EPS {
            out.push([a[0] + t * rx, a[1] + t * ry]);
        }
    } else if (qpx * ry - qpy * rx).abs() < 1e-9 {
        for p in [c, d] {
            if on_seg(p, a, b).is_some() {
                out.push(p);
            }
        }
        for p in [a, b] {
            if on_seg(p, c, d).is_some() {
                out.push(p);
            }
        }
    }
}

/// ``(start (n+1,), points (m,2))``: segment i's split points are
/// ``points[start[i]..start[i+1]]`` — its two endpoints, then every hit
/// in j order, stably sorted by `_on_seg(p, a, b) or 0.0`.
pub fn splits(segs: &[[P; 2]]) -> (Vec<i64>, Vec<P>) {
    let n = segs.len();
    let mut start = Vec::with_capacity(n + 1);
    let mut pts: Vec<P> = Vec::new();
    let mut hits: Vec<P> = Vec::new();
    start.push(0i64);
    for i in 0..n {
        let [a, b] = segs[i];
        let mut row: Vec<P> = vec![a, b];
        for (j, s) in segs.iter().enumerate() {
            if i == j {
                continue;
            }
            hits.clear();
            intersections(a, b, s[0], s[1], &mut hits);
            for &p in &hits {
                if on_seg(p, a, b).is_some() {
                    row.push(p);
                }
            }
        }
        // list.sort is stable; the key never NaN for finite input (a NaN
        // key would compare false both ways, as in Python).
        let mut keyed: Vec<(f64, P)> =
            row.into_iter().map(|p| (on_seg(p, a, b).unwrap_or(0.0), p)).collect();
        keyed.sort_by(|x, y| x.0.partial_cmp(&y.0).unwrap_or(std::cmp::Ordering::Equal));
        pts.extend(keyed.into_iter().map(|(_, p)| p));
        start.push(pts.len() as i64);
    }
    (start, pts)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_cross_splits_both() {
        let s = [[[0.0, 0.0], [2.0, 0.0]], [[1.0, -1.0], [1.0, 1.0]]];
        let (start, pts) = splits(&s);
        assert_eq!(start, vec![0, 3, 6]);
        assert_eq!(&pts[0..3], &[[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]);
        assert_eq!(&pts[3..6], &[[1.0, -1.0], [1.0, 0.0], [1.0, 1.0]]);
    }

    #[test]
    fn collinear_overlap_adds_the_inner_ends() {
        let s = [[[0.0, 0.0], [3.0, 0.0]], [[1.0, 0.0], [2.0, 0.0]]];
        let (_, pts) = splits(&s);
        assert_eq!(&pts[0..4], &[[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]);
    }
}
