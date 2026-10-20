// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! Texture coordinates: `core/texture.py` `projection_basis`,
//! `projection_axes`, `planar_uv`, `fit_uv_affine` and `affine_uv`.

use ingetrazo_math::V3;

/// Below this sine of the tilt the .skp projection uses the world axes.
const VERTICAL_TOLERANCE: f64 = 1e-3;

type Axes = ([f64; 3], [f64; 3]);

/// `projection_basis((nx, ny, nz))`.
pub fn projection_basis(normal: [f64; 3]) -> Axes {
    let (mut nx, mut ny, mut nz) = (normal[0], normal[1], normal[2]);
    let ln = (nx * nx + ny * ny + nz * nz).sqrt();
    if ln > 1e-30 {
        nx /= ln;
        ny /= ln;
        nz /= ln;
    }
    let (mut xx, mut xy) = (-ny, nx);
    let lx = (xx * xx + xy * xy).sqrt();
    if lx < VERTICAL_TOLERANCE {
        return if nz > 0.0 {
            ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0])
        } else {
            ([-1.0, 0.0, 0.0], [0.0, 1.0, 0.0])
        };
    }
    xx /= lx;
    xy /= lx;
    ([xx, xy, 0.0], [-nz * xy, nz * xx, nx * xy - ny * xx])
}

/// `projection_axes(normal, rot)`: the basis as float32 vectors, turned in
/// the plane by `rot` degrees.
pub fn projection_axes(normal: V3, rot: f64) -> (V3, V3) {
    let (xr, yr) = projection_basis([normal.0 as f64, normal.1 as f64, normal.2 as f64]);
    let mut u = V3(xr[0] as f32, xr[1] as f32, xr[2] as f32);
    let mut v = V3(yr[0] as f32, yr[1] as f32, yr[2] as f32);
    if rot != 0.0 {
        let a = rot.to_radians();
        let (cos_a, sin_a) = (a.cos() as f32, a.sin() as f32);
        let nu = u.mul(cos_a).add(v.mul(sin_a));
        let nv = v.mul(cos_a).sub(u.mul(sin_a));
        u = nu;
        v = nv;
    }
    (u, v)
}

/// `planar_uv(normal, positions, sw, sh, rot)`.
pub fn planar_uv(normal: V3, positions: &[V3], sw: f64, sh: f64, rot: f64) -> Vec<(f64, f64)> {
    let (u_axis, v_axis) = projection_axes(normal, rot);
    let sw = if sw.abs() > 1e-9 { sw } else { 1.0 };
    let sh = if sh.abs() > 1e-9 { sh } else { 1.0 };
    positions.iter().map(|&p| (p.dot(u_axis) as f64 / sw, p.dot(v_axis) as f64 / sh)).collect()
}

/// `affine_uv(uvw, positions)`.
pub fn affine_uv(uvw: &[f64; 8], positions: &[V3]) -> Vec<(f64, f64)> {
    positions
        .iter()
        .map(|p| {
            let (x, y, z) = (p.0 as f64, p.1 as f64, p.2 as f64);
            (uvw[0] * x + uvw[1] * y + uvw[2] * z + uvw[3], uvw[4] * x + uvw[5] * y + uvw[6] * z + uvw[7])
        })
        .collect()
}

/// `fit_uv_affine(points, uvs)`: the world to UV map `[gu.xyz, u0, gv.xyz,
/// v0]` of a polygon with explicit UVs, or `None` when it is degenerate.
///
/// Python runs polygons of more than 13 corners through a NumPy Gram
/// matrix; the sums here are plain loops, which agree unless BLAS fuses a
/// multiply-add.
pub fn fit_uv_affine(points: &[V3], uvs: &[(f64, f64)]) -> Option<[f64; 8]> {
    let n = points.len();
    if n < 3 || uvs.len() < n {
        return None;
    }
    let pts: Vec<[f64; 3]> = points.iter().map(|p| [p.0 as f64, p.1 as f64, p.2 as f64]).collect();
    let [x0, y0, z0] = pts[0];
    let d: Vec<[f64; 3]> = pts[1..].iter().map(|p| [p[0] - x0, p[1] - y0, p[2] - z0]).collect();
    let m = n - 1;
    let (bi, bj);
    if m <= 12 {
        let mut best = 1e-24;
        let (mut ci, mut cj) = (usize::MAX, usize::MAX);
        for i in 0..m {
            let [ax, ay, az] = d[i];
            for j in (i + 1)..m {
                let [bx, by, bz] = d[j];
                let cx = ay * bz - az * by;
                let cy = az * bx - ax * bz;
                let cz = ax * by - ay * bx;
                let c2 = cx * cx + cy * cy + cz * cz;
                if c2 > best {
                    best = c2;
                    ci = i;
                    cj = j;
                }
            }
        }
        if ci == usize::MAX {
            return None;
        }
        bi = ci;
        bj = cj;
    } else {
        let dot = |a: [f64; 3], b: [f64; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
        let n2: Vec<f64> = d.iter().map(|&a| dot(a, a)).collect();
        let mut best = f64::NEG_INFINITY;
        let (mut ci, mut cj) = (0, 0);
        for i in 0..m {
            for j in 0..m {
                let g = dot(d[i], d[j]);
                let cl2 = n2[i] * n2[j] - g * g;
                if cl2 > best {
                    best = cl2;
                    ci = i;
                    cj = j;
                }
            }
        }
        if best <= 1e-24 {
            return None;
        }
        bi = ci;
        bj = cj;
    }
    let [e1x, e1y, e1z] = d[bi];
    let [e2x, e2y, e2z] = d[bj];
    let g11 = e1x * e1x + e1y * e1y + e1z * e1z;
    let g12 = e1x * e2x + e1y * e2y + e1z * e2z;
    let g22 = e2x * e2x + e2y * e2y + e2z * e2z;
    let det = g11 * g22 - g12 * g12;
    if det.abs() < 1e-18 {
        return None;
    }
    let (i, j) = (bi + 1, bj + 1);
    let mut out = [0.0; 8];
    for k in 0..2 {
        let at = |t: usize| if k == 0 { uvs[t].0 } else { uvs[t].1 };
        let d1 = at(i) - at(0);
        let d2 = at(j) - at(0);
        let a = (d1 * g22 - d2 * g12) / det;
        let b = (d2 * g11 - d1 * g12) / det;
        let gx = e1x * a + e2x * b;
        let gy = e1y * a + e2y * b;
        let gz = e1z * a + e2z * b;
        let c = at(0) - (gx * x0 + gy * y0 + gz * z0);
        out[4 * k..4 * k + 4].copy_from_slice(&[gx, gy, gz, c]);
    }
    Some(out)
}
