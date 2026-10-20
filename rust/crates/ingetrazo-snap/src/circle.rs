// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! The circle fit behind the Center and Arc-midpoint inferences
//! (`core/snap.py::fit_circle`, `Viewport._center_of_edge`,
//! `_mesh_arc_midpoints`).
//!
//! `fit_circle` is least squares in the points' own plane; Python gets the
//! plane from `np.linalg.eigh` and the circle from `np.linalg.lstsq`
//! (LAPACK). Here a Jacobi eigen-solver and the normal equations do the same
//! job to a few ulps of float64 — not the same bits as LAPACK. The result
//! goes through float32 (`QVector3D`), which hides those ulps except for a
//! coordinate within ~1e-9 relative of a float32 rounding boundary, so the
//! snaps built on it are in practice the same bits; this is the one place
//! the engine is not bit-for-bit by construction.

use std::collections::HashMap;

use ingetrazo_math::pyfloat::py_max2;
use ingetrazo_math::{Hypot, V3};
use ingetrazo_model::mesh::{EdgeId, Mesh, VertexId};

/// Eigen-decomposition of a symmetric 3x3 matrix (cyclic Jacobi): the
/// eigenvalues ascending and the matching unit eigenvectors as columns.
fn eigh3(a: [[f64; 3]; 3]) -> ([f64; 3], [[f64; 3]; 3]) {
    let mut a = a;
    let mut v = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    for _ in 0..64 {
        let off = a[0][1].abs() + a[0][2].abs() + a[1][2].abs();
        if off == 0.0 {
            break;
        }
        for (p, q) in [(0usize, 1usize), (0, 2), (1, 2)] {
            // LAPACK deflates an off-diagonal that is rounding next to its
            // diagonal (`|e| <= sqrt|d_p d_q| * eps`) and keeps the axes
            // exactly as they were; a rotation by 1e-9 rad would show up in
            // a centre that sits on an axis.
            if a[p][q] == 0.0 || a[p][q].abs() <= (a[p][p].abs() * a[q][q].abs()).sqrt() * 1.1102230246251565e-16 {
                continue;
            }
            let theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
            let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
            let t = if theta == 0.0 { 1.0 } else { t };
            let c = 1.0 / (t * t + 1.0).sqrt();
            let s = t * c;
            for k in 0..3 {
                let (akp, akq) = (a[k][p], a[k][q]);
                a[k][p] = c * akp - s * akq;
                a[k][q] = s * akp + c * akq;
            }
            for k in 0..3 {
                let (apk, aqk) = (a[p][k], a[q][k]);
                a[p][k] = c * apk - s * aqk;
                a[q][k] = s * apk + c * aqk;
            }
            for k in 0..3 {
                let (vkp, vkq) = (v[k][p], v[k][q]);
                v[k][p] = c * vkp - s * vkq;
                v[k][q] = s * vkp + c * vkq;
            }
        }
    }
    let mut idx = [0usize, 1, 2];
    idx.sort_by(|&i, &j| a[i][i].partial_cmp(&a[j][j]).unwrap_or(std::cmp::Ordering::Equal));
    let vals = [a[idx[0]][idx[0]], a[idx[1]][idx[1]], a[idx[2]][idx[2]]];
    let mut vecs = [[0.0; 3]; 3];
    for (col, &i) in idx.iter().enumerate() {
        for k in 0..3 {
            vecs[k][col] = v[k][i];
        }
    }
    (vals, vecs)
}

/// Solve a 3x3 system by Gaussian elimination with partial pivoting.
fn solve3(mut m: [[f64; 4]; 3]) -> Option<[f64; 3]> {
    for col in 0..3 {
        let mut piv = col;
        for r in col + 1..3 {
            if m[r][col].abs() > m[piv][col].abs() {
                piv = r;
            }
        }
        if m[piv][col] == 0.0 {
            return None;
        }
        m.swap(col, piv);
        for r in col + 1..3 {
            let f = m[r][col] / m[col][col];
            for c in col..4 {
                m[r][c] -= f * m[col][c];
            }
        }
    }
    let mut x = [0.0; 3];
    for r in (0..3).rev() {
        let mut s = m[r][3];
        for c in r + 1..3 {
            s -= m[r][c] * x[c];
        }
        x[r] = s / m[r][r];
    }
    Some(x)
}

/// `fit_circle(points, tol_rel=0.01)`: `(centre, radius)` of the circle
/// through coplanar `points`, or `None` when they do not sit on one.
pub fn fit_circle(points: &[V3]) -> Option<(V3, f64)> {
    let tol_rel = 0.01;
    let n = points.len();
    if n < 3 {
        return None;
    }
    let pts: Vec<[f64; 3]> = points.iter().map(|p| [p.0 as f64, p.1 as f64, p.2 as f64]).collect();
    let mut c0 = [0.0; 3];
    for p in &pts {
        for k in 0..3 {
            c0[k] += p[k];
        }
    }
    for c in &mut c0 {
        *c /= n as f64;
    }
    let q: Vec<[f64; 3]> = pts.iter().map(|p| [p[0] - c0[0], p[1] - c0[1], p[2] - c0[2]]).collect();
    let mut cov = [[0.0; 3]; 3];
    for r in &q {
        for i in 0..3 {
            for j in 0..3 {
                cov[i][j] += r[i] * r[j];
            }
        }
    }
    let (_vals, vecs) = eigh3(cov);
    let u = [vecs[0][2], vecs[1][2], vecs[2][2]];
    let w = [vecs[0][1], vecs[1][1], vecs[2][1]];
    let x: Vec<f64> = q.iter().map(|r| r[0] * u[0] + r[1] * u[1] + r[2] * u[2]).collect();
    let y: Vec<f64> = q.iter().map(|r| r[0] * w[0] + r[1] * w[1] + r[2] * w[2]).collect();
    // Normal equations of  [2x 2y 1] . [cx cy c] = x^2 + y^2.
    let mut ata = [[0.0; 4]; 3];
    for i in 0..n {
        let row = [2.0 * x[i], 2.0 * y[i], 1.0];
        let b = x[i] * x[i] + y[i] * y[i];
        for r in 0..3 {
            for c in 0..3 {
                ata[r][c] += row[r] * row[c];
            }
            ata[r][3] += row[r] * b;
        }
    }
    let sol = solve3(ata)?;
    let (mut cx, mut cy) = (sol[0], sol[1]);
    let r2 = sol[2] + cx * cx + cy * cy;
    if r2 <= 0.0 {
        return None;
    }
    let r = r2.sqrt();
    // The points are centred, so for a circle drawn around its own mean the
    // centre sits on it: an offset a trillionth of the radius is the solver's
    // rounding, not a measurement. (LAPACK's answer for such a symmetric ring
    // is the mean to the last bit, and the centre of a circle on an axis
    // would otherwise carry a 1e-17 that float32 keeps.)
    if cx.abs() < 1e-12 * r {
        cx = 0.0;
    }
    if cy.abs() < 1e-12 * r {
        cy = 0.0;
    }
    let mut dev = 0.0f64;
    for i in 0..n {
        let d = ((x[i] - cx).hypot(y[i] - cy) - r).abs();
        if d > dev {
            dev = d;
        }
    }
    if dev > py_max2(0.002, tol_rel * r) {
        return None;
    }
    let c = [c0[0] + u[0] * cx + w[0] * cy, c0[1] + u[1] * cx + w[1] * cy, c0[2] + u[2] * cx + w[2] * cy];
    Some((V3(c[0] as f32, c[1] as f32, c[2] as f32), r))
}

/// `Viewport._center_of_edge(edge, mesh)`: the centre of the circle or arc
/// the edge belongs to, `(centre, radius)`.
pub fn center_of_edge(mesh: &Mesh, edge: EdgeId) -> Option<(V3, f64)> {
    let mut seen: Vec<VertexId> = Vec::new();
    let mut pts: Vec<V3> = Vec::new();
    for e in mesh.curve_edges(edge) {
        let ed = mesh.edge(e);
        for v in [ed.v0, ed.v1] {
            if !seen.contains(&v) {
                seen.push(v);
                pts.push(mesh.position(v));
            } else {
                // a dict keyed by id keeps the first insertion position
            }
        }
    }
    if pts.len() >= 3 {
        fit_circle(&pts)
    } else {
        None
    }
}

/// `Viewport._mesh_arc_midpoints(mesh)`: the sweep midpoint of every open
/// curve chain of the mesh, in the mesh's own space.
pub fn mesh_arc_midpoints(mesh: &Mesh, h: Hypot) -> Vec<V3> {
    // curve id -> edges, in the order the ids first appear.
    let mut order: Vec<i64> = Vec::new();
    let mut by_curve: HashMap<i64, Vec<EdgeId>> = HashMap::new();
    for &e in &mesh.edges {
        if let Some(cid) = mesh.edge(e).curve {
            by_curve.entry(cid).or_insert_with(|| {
                order.push(cid);
                Vec::new()
            });
            by_curve.get_mut(&cid).unwrap().push(e);
        }
    }
    let mut mids = Vec::new();
    for cid in order {
        let edges = &by_curve[&cid];
        let mut verts: Vec<VertexId> = Vec::new();
        let mut degree: Vec<usize> = Vec::new();
        for &e in edges {
            let ed = mesh.edge(e);
            for v in [ed.v0, ed.v1] {
                match verts.iter().position(|&x| x == v) {
                    Some(i) => degree[i] += 1,
                    None => {
                        verts.push(v);
                        degree.push(1);
                    }
                }
            }
        }
        let ends = degree.iter().filter(|&&n| n == 1).count();
        if ends != 2 || verts.len() < 3 {
            continue; // a circle, or not a chain
        }
        let pos: Vec<V3> = verts.iter().map(|&v| mesh.position(v)).collect();
        let Some((c, r)) = fit_circle(&pos) else { continue };
        let mut m = V3(0.0, 0.0, 0.0);
        for &pt in &pos {
            m = m.add(pt.sub(c));
        }
        if (m.length(h) as f64) < 1e-9 {
            continue;
        }
        mids.push(c.add(m.normalized(h).mul(r as f32)));
    }
    mids
}

// ---- the curves on a face's boundary -----------------------------------------------

/// A boundary segment's key in `_boundary_runs`: the curve id, `soft`, or
/// the geometric signature.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RunKey {
    Id(i64),
    Soft,
    Geo,
}

/// `core.snap._boundary_runs(loop)`: runs of consecutive segments that
/// belong to one circle or arc, `(vertices, drawn)`.
pub fn boundary_runs(mesh: &Mesh, lp: &[VertexId], h: Hypot) -> Vec<(Vec<VertexId>, bool)> {
    let n = lp.len();
    if n < 3 {
        return Vec::new();
    }
    let segs: Vec<(Option<i64>, bool)> = (0..n)
        .map(|i| match mesh.find_edge(lp[i], lp[(i + 1) % n]) {
            Some(e) => (mesh.edge(e).curve, mesh.edge(e).soft),
            None => (None, false),
        })
        .collect();
    let turn = |i: usize| -> (Option<f64>, f64, f64) {
        let a = mesh.position(if i > 0 { lp[i - 1] } else { lp[n - 1] });
        let b = mesh.position(lp[i]);
        let c = mesh.position(lp[(i + 1) % n]);
        let (u, w) = (b.sub(a), c.sub(b));
        let (lu, lw) = (u.length(h) as f64, w.length(h) as f64);
        if lu < 1e-9 || lw < 1e-9 {
            return (None, lu, lw);
        }
        let cosang = py_max2(-1.0, ingetrazo_math::pyfloat::py_min2(1.0, u.dot(w) as f64 / (lu * lw)));
        (Some(cosang.acos() * (180.0 / std::f64::consts::PI)), lu, lw)
    };
    let mut keys: Vec<Option<RunKey>> = Vec::with_capacity(n);
    for i in 0..n {
        let (cid, soft) = segs[i];
        if let Some(c) = cid {
            keys.push(Some(RunKey::Id(c)));
        } else if soft {
            keys.push(Some(RunKey::Soft));
        } else {
            let (t, _lu, lw) = turn(i);
            let (tn, lu2, _lw2) = turn((i + 1) % n);
            let same = matches!((t, tn), (Some(t), Some(tn)) if 2.0 < t && t < 60.0
                && (t - tn).abs() < 1.5
                && (lu2 - lw).abs() < 0.05 * py_max2(lw, 1e-9));
            keys.push(if same { Some(RunKey::Geo) } else { None });
        }
    }
    if keys[0].is_some() && keys.iter().all(|k| *k == keys[0]) {
        return vec![(lp.to_vec(), matches!(keys[0], Some(RunKey::Id(_))))];
    }
    let mut start = 0;
    for j in 0..n {
        if keys[j] != keys[(j + n - 1) % n] {
            start = j;
            break;
        }
    }
    let order: Vec<usize> = (0..n).map(|j| (start + j) % n).collect();
    let mut runs: Vec<Vec<usize>> = Vec::new();
    let mut cur: Vec<usize> = Vec::new();
    let mut cur_key: Option<RunKey> = None;
    for &j in &order {
        let k = keys[j];
        if k.is_some() && k == cur_key {
            cur.push(j);
        } else {
            if cur_key.is_some() && cur.len() >= 2 {
                runs.push(std::mem::take(&mut cur));
            }
            cur = if k.is_some() { vec![j] } else { Vec::new() };
            cur_key = k;
        }
    }
    if cur_key.is_some() && cur.len() >= 2 {
        runs.push(cur);
    }
    runs.iter()
        .map(|r| {
            let mut verts: Vec<VertexId> = r.iter().map(|&j| lp[j]).collect();
            verts.push(lp[(r[r.len() - 1] + 1) % n]);
            (verts, matches!(keys[r[0]], Some(RunKey::Id(_))))
        })
        .collect()
}

/// `core.snap.curve_centers_of_face(face, xform)`: the centre of every
/// circle or arc on the face's boundary, `(centre, radius, (loop, run))`
/// in world space.
pub fn curve_centers_of_face(
    mesh: &Mesh,
    f: ingetrazo_model::FaceId,
    xform: Option<&ingetrazo_math::M4>,
    h: Hypot,
) -> Vec<(V3, f64, (usize, usize))> {
    let face = mesh.face(f);
    let mut out = Vec::new();
    let mut loops: Vec<&[VertexId]> = vec![face.outer.as_slice()];
    loops.extend(face.holes.iter().map(|hl| hl.as_slice()));
    for (li, lp) in loops.into_iter().enumerate() {
        for (ri, (verts, drawn)) in boundary_runs(mesh, lp, h).into_iter().enumerate() {
            if verts.len() < if drawn { 3 } else { 5 } {
                continue;
            }
            let pts: Vec<V3> = verts
                .iter()
                .map(|&v| {
                    let p = mesh.position(v);
                    match xform {
                        Some(m) => m.map(p),
                        None => p,
                    }
                })
                .collect();
            if let Some((c, r)) = fit_circle(&pts) {
                out.push((c, r, (li, ri)));
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ring(c: V3, r: f32, n: usize, arc: f32) -> Vec<V3> {
        (0..n).map(|i| {
            let t = arc * i as f32 / n as f32;
            V3(c.0 + r * t.cos(), c.1 + r * t.sin(), c.2)
        }).collect()
    }

    #[test]
    fn a_ring_gives_its_centre_and_radius() {
        let (c, r) = fit_circle(&ring(V3(3.0, 0.0, 0.0), 1.5, 24, std::f32::consts::TAU)).unwrap();
        assert!((c.0 - 3.0).abs() < 1e-6 && c.1.abs() < 1e-6 && c.2 == 0.0, "{c:?}");
        assert!((r - 1.5).abs() < 1e-6);
    }

    #[test]
    fn a_tilted_arc_is_fitted_in_its_own_plane() {
        let pts: Vec<V3> = ring(V3(0.0, 0.0, 0.0), 2.0, 12, 3.0).iter().map(|p| V3(p.0, p.1 * 0.6, p.1 * 0.8)).collect();
        let (c, r) = fit_circle(&pts).unwrap();
        assert!(c.0.abs() < 1e-5 && c.1.abs() < 1e-5 && c.2.abs() < 1e-5, "{c:?}");
        assert!((r - 2.0).abs() < 1e-4);
    }

    #[test]
    fn a_polyline_that_only_bends_is_not_a_circle() {
        let pts = [V3(0.0, 0.0, 0.0), V3(1.0, 0.0, 0.0), V3(2.0, 0.0, 0.0), V3(3.0, 1.0, 0.0), V3(3.0, 3.0, 0.0)];
        assert!(fit_circle(&pts).is_none());
        assert!(fit_circle(&pts[..2]).is_none());
    }

    #[test]
    fn the_eigen_solver_orders_ascending() {
        let (vals, vecs) = eigh3([[2.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 1.0]]);
        assert_eq!(vals, [1.0, 2.0, 5.0]);
        assert_eq!([vecs[2][0].abs(), vecs[0][1].abs(), vecs[1][2].abs()], [1.0, 1.0, 1.0]);
    }
}
