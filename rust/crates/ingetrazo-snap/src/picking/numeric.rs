// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! The NumPy arithmetic the pick index runs, operation for operation.
//!
//! The Python runs NumPy over float64 and Python floats over the f32 values
//! Qt hands out. Where NumPy hands a sum to OpenBLAS the order and the fused
//! multiply-add are those of its x86-64 kernels (Haswell-class AVX2 + FMA;
//! `rust/crates/ingetrazo-geom/src/hlr_draw.rs` has the notes, and
//! `tests/pick_parity.rs` checks them again here for 1..=70 rows):
//!
//! * `(N,3) @ (3,)` (`gemv`): `fma(a2, b2, fma(a0, b0, a1·b1))` for N ≥ 2;
//!   `np.dot` of two 3-vectors: `(a0b0 + a1b1) + a2b2`, no FMA.
//! * `(N,3) @ (3,4)` (`gemm`): `fma(a2, b2, fma(a1, b1, a0·b0))` for N ≥ 2;
//!   one row is the `gemv` order.
//! * `np.einsum("ij,ij->i")` on 3-vectors: `(a2b2 + a0b0) + a1b1`.
//! * `np.cross`: `a1 b2 − a2 b1` each product rounded on its own.

use ingetrazo_math::V3;

pub type P3 = [f64; 3];

#[inline]
pub fn p3(v: V3) -> P3 {
    [v.0 as f64, v.1 as f64, v.2 as f64]
}

/// `QVector3D(x, y, z)` of three Python floats.
#[inline]
pub fn v3(p: P3) -> V3 {
    V3(p[0] as f32, p[1] as f32, p[2] as f32)
}

/// `np.cross` of two 3-vectors.
#[inline]
pub fn cross3(a: P3, b: P3) -> P3 {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}

/// `np.dot` / `ddot` of two 3-vectors: left to right, no FMA.
#[inline]
pub fn dot3(a: P3, b: P3) -> f64 {
    (a[0] * b[0] + a[1] * b[1]) + a[2] * b[2]
}

/// `np.einsum("ij,ij->i", a, b)` on one pair of 3-vectors.
#[inline]
pub fn einsum3(a: P3, b: P3) -> f64 {
    (a[2] * b[2] + a[0] * b[0]) + a[1] * b[1]
}

/// One row of `rows @ v` for float64 `(N,3) @ (3,)`.
#[inline]
pub fn matvec3(row: P3, v: P3, n_rows: usize) -> f64 {
    if n_rows == 1 {
        dot3(row, v)
    } else {
        row[2].mul_add(v[2], row[0].mul_add(v[0], row[1] * v[1]))
    }
}

/// One element of `rows @ M.T` for float64 `(N,3) @ (3,K)`: row `p`, column
/// `m` of the right operand.
#[inline]
pub fn matmul3(p: P3, m: P3, n_rows: usize) -> f64 {
    if n_rows == 1 {
        p[2].mul_add(m[2], p[0].mul_add(m[0], p[1] * m[1]))
    } else {
        p[2].mul_add(m[2], p[1].mul_add(m[1], p[0] * m[0]))
    }
}

/// `pts @ rot.T` without the translation (`tv` of `_instance_chunk`).
pub fn rows_matmul(pts: &[P3], rot: &[[f64; 3]; 3]) -> Vec<P3> {
    let n = pts.len();
    pts.iter().map(|a| [matmul3(*a, rot[0], n), matmul3(*a, rot[1], n), matmul3(*a, rot[2], n)]).collect()
}

/// `np.linalg.det` of a 3x3: LAPACK's LU (left-looking, reciprocal scaling)
/// and `sign · exp(Σ log|uᵢᵢ|)`, which is how NumPy forms it.
///
/// Not bit-exact against NumPy for every matrix (about one in five differs
/// in the last bit); an identity or a pure scale, which is what the area
/// factor of a translated or scaled placement needs, is exact.
pub fn np_det3(l: &[[f64; 3]; 3]) -> f64 {
    let mut a = [[l[0][0], l[1][0], l[2][0]], [l[0][1], l[1][1], l[2][1]], [l[0][2], l[1][2], l[2][2]]]; // a[col][row]
    let mut ipiv = [0usize; 3];
    let mut sign = 1.0f64;
    for j in 0..3 {
        let mut b = a[j];
        for i in 0..j {
            let ip = ipiv[i];
            if ip != i {
                b.swap(i, ip);
            }
        }
        for i in 1..j {
            let mut s = 0.0;
            for k in 0..i {
                s += a[k][i] * b[k];
            }
            b[i] -= s;
        }
        for i in j..3 {
            let mut s = b[i];
            for k in 0..j {
                s -= b[k] * a[k][i];
            }
            b[i] = s;
        }
        let mut jp = j;
        for i in j..3 {
            if b[i].abs() > b[jp].abs() {
                jp = i;
            }
        }
        ipiv[j] = jp;
        a[j] = b;
        if b[jp] != 0.0 {
            if jp != j {
                for c in 0..=j {
                    a[c].swap(j, jp);
                }
                sign = -sign;
            }
            let t = 1.0 / a[j][j];
            for i in j + 1..3 {
                a[j][i] *= t;
            }
        }
    }
    let mut logdet = 0.0f64;
    for i in 0..3 {
        let d = a[i][i];
        if d < 0.0 {
            sign = -sign;
        }
        logdet += d.abs().ln();
    }
    sign * logdet.exp()
}

/// `math.hypot(x, y)` of CPython 3.12.
pub use ingetrazo_geom::sweep::py_hypot2;

/// `np.hypot`: the C library's.
#[inline]
pub fn np_hypot(a: f64, b: f64) -> f64 {
    a.hypot(b)
}

/// Python's `min`/`max` of two floats: the first wins a tie or a NaN.
#[inline]
pub fn py_min(a: f64, b: f64) -> f64 {
    if b < a {
        b
    } else {
        a
    }
}

#[inline]
pub fn py_max(a: f64, b: f64) -> f64 {
    if b > a {
        b
    } else {
        a
    }
}

/// `viewport._point_to_segment_distance_2d`.
pub fn point_to_segment_2d(p: (f64, f64), a: (f64, f64), b: (f64, f64)) -> f64 {
    let (ax, ay) = a;
    let (bx, by) = b;
    let (px, py) = p;
    let dx = bx - ax;
    let dy = by - ay;
    if dx == 0.0 && dy == 0.0 {
        return py_hypot2(px - ax, py - ay);
    }
    let t = ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy);
    let t = py_max(0.0, py_min(1.0, t));
    let qx = ax + t * dx;
    let qy = ay + t * dy;
    py_hypot2(px - qx, py - qy)
}

/// `viewport._ray_aabb`: the slab test with its tolerance for a ray
/// parallel to an axis. Plain floats.
pub fn ray_aabb(o: P3, d: P3, lo: P3, hi: P3, reach: f64) -> bool {
    let mut tmin = 0.0f64;
    let mut tmax = reach;
    for i in 0..3 {
        let di = d[i];
        if -1e-12 < di && di < 1e-12 {
            if o[i] < lo[i] - 1e-9 || o[i] > hi[i] + 1e-9 {
                return false;
            }
            continue;
        }
        let inv = 1.0 / di;
        let mut t1 = (lo[i] - o[i]) * inv;
        let mut t2 = (hi[i] - o[i]) * inv;
        if t1 > t2 {
            std::mem::swap(&mut t1, &mut t2);
        }
        if t1 > tmin {
            tmin = t1;
        }
        if t2 < tmax {
            tmax = t2;
        }
        if tmin > tmax {
            return false;
        }
    }
    true
}
