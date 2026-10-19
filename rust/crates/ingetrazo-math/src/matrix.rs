// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! `QMatrix4x4` (Qt 6.11, src/gui/math3d/qmatrix4x4.{h,cpp}), bit for bit.
//!
//! Qt keeps a type flag beside the 16 floats — identity, translation,
//! scale, a rotation about Z, any rotation, perspective — and takes a
//! shorter path of arithmetic when it can. The shorter path rounds
//! differently, so the flags are part of the value: a matrix built by
//! `rotate()` and the same 16 floats read back from a file (General) map a
//! point to results one ulp apart. Every operation here keeps the flags as
//! Qt does and follows Qt's expression order in float32; the determinant
//! and the general inverse are computed in double, as Qt does.

use crate::qt::{Hypot, V3};

pub const IDENTITY: u8 = 0x00;
pub const TRANSLATION: u8 = 0x01;
pub const SCALE: u8 = 0x02;
pub const ROTATION_2D: u8 = 0x04;
pub const ROTATION: u8 = 0x08;
pub const PERSPECTIVE: u8 = 0x10;
pub const GENERAL: u8 = 0x1f;

/// `m[column][row]`, as Qt stores it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct M4 {
    pub m: [[f32; 4]; 4],
    pub flags: u8,
}

impl Default for M4 {
    fn default() -> Self {
        M4::identity()
    }
}

type D4 = [[f64; 4]; 4];

fn to_doubles(m: &[[f32; 4]; 4]) -> D4 {
    let mut d = [[0.0f64; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            d[i][j] = m[i][j] as f64;
        }
    }
    d
}

fn det2(m: &D4, col0: usize, col1: usize, row0: usize, row1: usize) -> f64 {
    m[col0][row0] * m[col1][row1] - m[col0][row1] * m[col1][row0]
}

fn det3(m: &D4, col0: usize, col1: usize, col2: usize, row0: usize, row1: usize, row2: usize) -> f64 {
    m[col0][row0] * det2(m, col1, col2, row1, row2) - m[col1][row0] * det2(m, col0, col2, row1, row2)
        + m[col2][row0] * det2(m, col0, col1, row1, row2)
}

fn det4(m: &D4) -> f64 {
    let mut det = m[0][0] * det3(m, 1, 2, 3, 1, 2, 3);
    det -= m[1][0] * det3(m, 0, 2, 3, 1, 2, 3);
    det += m[2][0] * det3(m, 0, 1, 3, 1, 2, 3);
    det -= m[3][0] * det3(m, 0, 1, 2, 1, 2, 3);
    det
}

/// `qFuzzyCompare(double, double)`.
fn fuzzy_compare(p1: f64, p2: f64) -> bool {
    (p1 - p2).abs() * 1_000_000_000_000.0 <= p1.abs().min(p2.abs())
}

/// `qFuzzyIsNull(double)`.
fn fuzzy_is_null_d(d: f64) -> bool {
    d.abs() <= 0.000_000_000_001
}

/// `qFuzzyIsNull(float)`.
fn fuzzy_is_null_f(f: f32) -> bool {
    f.abs() <= 0.00001
}

/// `qDegreesToRadians(float)`.
fn radians(deg: f32) -> f32 {
    deg * (std::f64::consts::PI / 180.0) as f32
}

impl M4 {
    pub const fn identity() -> M4 {
        M4 {
            m: [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
            flags: IDENTITY,
        }
    }

    /// `QMatrix4x4(m11, m12, …, m44)`: the 16 values row by row. General.
    pub fn from_rows(v: [f32; 16]) -> M4 {
        let mut m = [[0.0f32; 4]; 4];
        for row in 0..4 {
            for col in 0..4 {
                m[col][row] = v[row * 4 + col];
            }
        }
        M4 { m, flags: GENERAL }
    }

    /// `data()`: the 16 floats column by column.
    pub fn data(&self) -> [f32; 16] {
        let mut out = [0.0f32; 16];
        for col in 0..4 {
            for row in 0..4 {
                out[col * 4 + row] = self.m[col][row];
            }
        }
        out
    }

    /// The matrix whose `data()` is `v`, flagged General — what a matrix
    /// read back from an .igz is.
    pub fn from_data(v: [f32; 16]) -> M4 {
        let mut m = [[0.0f32; 4]; 4];
        for col in 0..4 {
            for row in 0..4 {
                m[col][row] = v[col * 4 + row];
            }
        }
        M4 { m, flags: GENERAL }
    }

    pub fn column(&self, i: usize) -> [f32; 4] {
        self.m[i]
    }

    pub fn row(&self, i: usize) -> [f32; 4] {
        [self.m[0][i], self.m[1][i], self.m[2][i], self.m[3][i]]
    }

    pub fn set_column(&mut self, i: usize, v: [f32; 4]) {
        self.m[i] = v;
        self.flags = GENERAL;
    }

    pub fn set_row(&mut self, i: usize, v: [f32; 4]) {
        for (c, x) in v.into_iter().enumerate() {
            self.m[c][i] = x;
        }
        self.flags = GENERAL;
    }

    pub fn is_identity(&self) -> bool {
        if self.flags == IDENTITY {
            return true;
        }
        let m = &self.m;
        let want = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]];
        for c in 0..4 {
            for r in 0..4 {
                if m[c][r] != want[c][r] {
                    return false;
                }
            }
        }
        true
    }

    pub fn set_to_identity(&mut self) {
        *self = M4::identity();
    }

    // ---- products ------------------------------------------------------------

    /// `m1 * m2`.
    pub fn mul(&self, m2: &M4) -> M4 {
        let (a, b) = (&self.m, &m2.m);
        let flags = self.flags | m2.flags;
        let mut m = [[0.0f32; 4]; 4];
        if flags < ROTATION_2D {
            m[0][0] = a[0][0] * b[0][0];
            m[1][1] = a[1][1] * b[1][1];
            m[2][2] = a[2][2] * b[2][2];
            m[3][0] = a[3][0] + a[0][0] * b[3][0];
            m[3][1] = a[3][1] + a[1][1] * b[3][1];
            m[3][2] = a[3][2] + a[2][2] * b[3][2];
            m[3][3] = 1.0;
            return M4 { m, flags };
        }
        for c in 0..4 {
            for r in 0..4 {
                m[c][r] = a[0][r] * b[c][0] + a[1][r] * b[c][1] + a[2][r] * b[c][2] + a[3][r] * b[c][3];
            }
        }
        M4 { m, flags }
    }

    /// `*this *= other` — not the same arithmetic as `*this = *this * other`
    /// on the short path (it adds to the translation in place).
    pub fn mul_assign(&mut self, o: &M4) {
        let other = *o;
        self.flags |= other.flags;
        let m = &mut self.m;
        let b = &other.m;
        if self.flags < ROTATION_2D {
            m[3][0] += m[0][0] * b[3][0];
            m[3][1] += m[1][1] * b[3][1];
            m[3][2] += m[2][2] * b[3][2];
            m[0][0] *= b[0][0];
            m[1][1] *= b[1][1];
            m[2][2] *= b[2][2];
            return;
        }
        for r in 0..4 {
            let m0 = m[0][r] * b[0][0] + m[1][r] * b[0][1] + m[2][r] * b[0][2] + m[3][r] * b[0][3];
            let m1 = m[0][r] * b[1][0] + m[1][r] * b[1][1] + m[2][r] * b[1][2] + m[3][r] * b[1][3];
            let m2 = m[0][r] * b[2][0] + m[1][r] * b[2][1] + m[2][r] * b[2][2] + m[3][r] * b[2][3];
            m[3][r] = m[0][r] * b[3][0] + m[1][r] * b[3][1] + m[2][r] * b[3][2] + m[3][r] * b[3][3];
            m[0][r] = m0;
            m[1][r] = m1;
            m[2][r] = m2;
        }
    }

    // ---- mapping -------------------------------------------------------------

    /// `map(QVector3D)`: a point (w = 1).
    pub fn map(&self, p: V3) -> V3 {
        let m = &self.m;
        let (x, y, z) = (p.0, p.1, p.2);
        if self.flags == IDENTITY {
            p
        } else if self.flags < ROTATION_2D {
            V3(x * m[0][0] + m[3][0], y * m[1][1] + m[3][1], z * m[2][2] + m[3][2])
        } else if self.flags < ROTATION {
            V3(x * m[0][0] + y * m[1][0] + m[3][0], x * m[0][1] + y * m[1][1] + m[3][1], z * m[2][2] + m[3][2])
        } else {
            let xx = x * m[0][0] + y * m[1][0] + z * m[2][0] + m[3][0];
            let yy = x * m[0][1] + y * m[1][1] + z * m[2][1] + m[3][1];
            let zz = x * m[0][2] + y * m[1][2] + z * m[2][2] + m[3][2];
            let w = x * m[0][3] + y * m[1][3] + z * m[2][3] + m[3][3];
            if w == 1.0 {
                V3(xx, yy, zz)
            } else {
                V3(xx / w, yy / w, zz / w)
            }
        }
    }

    /// `mapVector(QVector3D)`: a direction (no translation).
    pub fn map_vector(&self, v: V3) -> V3 {
        let m = &self.m;
        if self.flags < SCALE {
            v
        } else if self.flags < ROTATION_2D {
            V3(v.0 * m[0][0], v.1 * m[1][1], v.2 * m[2][2])
        } else {
            V3(
                v.0 * m[0][0] + v.1 * m[1][0] + v.2 * m[2][0],
                v.0 * m[0][1] + v.1 * m[1][1] + v.2 * m[2][1],
                v.0 * m[0][2] + v.1 * m[1][2] + v.2 * m[2][2],
            )
        }
    }

    // ---- building ------------------------------------------------------------

    pub fn translate(&mut self, x: f32, y: f32, z: f32) {
        let f = self.flags;
        let m = &mut self.m;
        if f == IDENTITY {
            m[3][0] = x;
            m[3][1] = y;
            m[3][2] = z;
        } else if f == TRANSLATION {
            m[3][0] += x;
            m[3][1] += y;
            m[3][2] += z;
        } else if f == SCALE {
            m[3][0] = m[0][0] * x;
            m[3][1] = m[1][1] * y;
            m[3][2] = m[2][2] * z;
        } else if f == TRANSLATION | SCALE {
            m[3][0] += m[0][0] * x;
            m[3][1] += m[1][1] * y;
            m[3][2] += m[2][2] * z;
        } else if f < ROTATION {
            m[3][0] += m[0][0] * x + m[1][0] * y;
            m[3][1] += m[0][1] * x + m[1][1] * y;
            m[3][2] += m[2][2] * z;
        } else {
            m[3][0] += m[0][0] * x + m[1][0] * y + m[2][0] * z;
            m[3][1] += m[0][1] * x + m[1][1] * y + m[2][1] * z;
            m[3][2] += m[0][2] * x + m[1][2] * y + m[2][2] * z;
            m[3][3] += m[0][3] * x + m[1][3] * y + m[2][3] * z;
        }
        self.flags |= TRANSLATION;
    }

    /// `translate(x, y)`.
    pub fn translate2(&mut self, x: f32, y: f32) {
        let f = self.flags;
        let m = &mut self.m;
        if f == IDENTITY {
            m[3][0] = x;
            m[3][1] = y;
        } else if f == TRANSLATION {
            m[3][0] += x;
            m[3][1] += y;
        } else if f == SCALE {
            m[3][0] = m[0][0] * x;
            m[3][1] = m[1][1] * y;
        } else if f == TRANSLATION | SCALE {
            m[3][0] += m[0][0] * x;
            m[3][1] += m[1][1] * y;
        } else if f < ROTATION {
            m[3][0] += m[0][0] * x + m[1][0] * y;
            m[3][1] += m[0][1] * x + m[1][1] * y;
        } else {
            m[3][0] += m[0][0] * x + m[1][0] * y;
            m[3][1] += m[0][1] * x + m[1][1] * y;
            m[3][2] += m[0][2] * x + m[1][2] * y;
            m[3][3] += m[0][3] * x + m[1][3] * y;
        }
        self.flags |= TRANSLATION;
    }

    pub fn scale(&mut self, x: f32, y: f32, z: f32) {
        let f = self.flags;
        let m = &mut self.m;
        if f < SCALE {
            m[0][0] = x;
            m[1][1] = y;
            m[2][2] = z;
        } else if f < ROTATION_2D {
            m[0][0] *= x;
            m[1][1] *= y;
            m[2][2] *= z;
        } else if f < ROTATION {
            m[0][0] *= x;
            m[0][1] *= x;
            m[1][0] *= y;
            m[1][1] *= y;
            m[2][2] *= z;
        } else {
            for r in 0..4 {
                m[0][r] *= x;
            }
            for r in 0..4 {
                m[1][r] *= y;
            }
            for r in 0..4 {
                m[2][r] *= z;
            }
        }
        self.flags |= SCALE;
    }

    /// `scale(factor)`: the same factor on x, y and z.
    pub fn scale1(&mut self, factor: f32) {
        self.scale(factor, factor, factor);
    }

    /// `rotate(angle, x, y, z)`, the angle in degrees.
    pub fn rotate(&mut self, angle: f32, mut x: f32, mut y: f32, mut z: f32) {
        if angle == 0.0 {
            return;
        }
        let (c, mut s);
        if angle == 90.0 || angle == -270.0 {
            s = 1.0f32;
            c = 0.0f32;
        } else if angle == -90.0 || angle == 270.0 {
            s = -1.0;
            c = 0.0;
        } else if angle == 180.0 || angle == -180.0 {
            s = 0.0;
            c = -1.0;
        } else {
            let a = radians(angle);
            c = a.cos();
            s = a.sin();
        }
        let m = &mut self.m;
        if x == 0.0 {
            if y == 0.0 {
                if z != 0.0 {
                    if z < 0.0 {
                        s = -s;
                    }
                    for r in 0..4 {
                        let tmp = m[0][r];
                        m[0][r] = tmp * c + m[1][r] * s;
                        m[1][r] = m[1][r] * c - tmp * s;
                    }
                    self.flags |= ROTATION_2D;
                    return;
                }
            } else if z == 0.0 {
                if y < 0.0 {
                    s = -s;
                }
                for r in 0..4 {
                    let tmp = m[2][r];
                    m[2][r] = tmp * c + m[0][r] * s;
                    m[0][r] = m[0][r] * c - tmp * s;
                }
                self.flags |= ROTATION;
                return;
            }
        } else if y == 0.0 && z == 0.0 {
            if x < 0.0 {
                s = -s;
            }
            for r in 0..4 {
                let tmp = m[1][r];
                m[1][r] = tmp * c + m[2][r] * s;
                m[2][r] = m[2][r] * c - tmp * s;
            }
            self.flags |= ROTATION;
            return;
        }
        let mut len = (x as f64) * (x as f64) + (y as f64) * (y as f64) + (z as f64) * (z as f64);
        if !fuzzy_compare(len, 1.0) && !fuzzy_is_null_d(len) {
            len = len.sqrt();
            x = ((x as f64) / len) as f32;
            y = ((y as f64) / len) as f32;
            z = ((z as f64) / len) as f32;
        }
        let ic = 1.0f32 - c;
        let mut rot = M4 { m: [[0.0; 4]; 4], flags: ROTATION };
        rot.m[0][0] = x * x * ic + c;
        rot.m[1][0] = x * y * ic - z * s;
        rot.m[2][0] = x * z * ic + y * s;
        rot.m[0][1] = y * x * ic + z * s;
        rot.m[1][1] = y * y * ic + c;
        rot.m[2][1] = y * z * ic - x * s;
        rot.m[0][2] = x * z * ic - y * s;
        rot.m[1][2] = y * z * ic + x * s;
        rot.m[2][2] = z * z * ic + c;
        rot.m[3][3] = 1.0;
        self.mul_assign(&rot);
    }

    pub fn ortho(&mut self, left: f32, right: f32, bottom: f32, top: f32, near: f32, far: f32) {
        if left == right || bottom == top || near == far {
            return;
        }
        let width = right - left;
        let invheight = top - bottom;
        let clip = far - near;
        let mut o = M4 { m: [[0.0; 4]; 4], flags: TRANSLATION | SCALE };
        o.m[0][0] = 2.0 / width;
        o.m[3][0] = -(left + right) / width;
        o.m[1][1] = 2.0 / invheight;
        o.m[3][1] = -(top + bottom) / invheight;
        o.m[2][2] = -2.0 / clip;
        o.m[3][2] = -(near + far) / clip;
        o.m[3][3] = 1.0;
        self.mul_assign(&o);
    }

    pub fn perspective(&mut self, vertical_angle: f32, aspect: f32, near: f32, far: f32) {
        if near == far || aspect == 0.0 {
            return;
        }
        let rad = radians(vertical_angle / 2.0);
        let sine = rad.sin();
        if sine == 0.0 {
            return;
        }
        let cotan = rad.cos() / sine;
        let clip = far - near;
        let mut p = M4 { m: [[0.0; 4]; 4], flags: GENERAL };
        p.m[0][0] = cotan / aspect;
        p.m[1][1] = cotan;
        p.m[2][2] = -(near + far) / clip;
        p.m[3][2] = -(2.0 * near * far) / clip;
        p.m[2][3] = -1.0;
        self.mul_assign(&p);
    }

    /// `lookAt(eye, center, up)`; `h` is this Qt's `std::hypot` (lengths).
    pub fn look_at(&mut self, eye: V3, center: V3, up: V3, h: Hypot) {
        let forward = center.sub(eye);
        if fuzzy_is_null_f(forward.0) && fuzzy_is_null_f(forward.1) && fuzzy_is_null_f(forward.2) {
            return;
        }
        let forward = forward.normalize(h);
        let side = forward.cross(up).normalized(h);
        let up_v = side.cross(forward);
        let mut l = M4 { m: [[0.0; 4]; 4], flags: ROTATION };
        l.m[0][0] = side.0;
        l.m[1][0] = side.1;
        l.m[2][0] = side.2;
        l.m[0][1] = up_v.0;
        l.m[1][1] = up_v.1;
        l.m[2][1] = up_v.2;
        l.m[0][2] = -forward.0;
        l.m[1][2] = -forward.1;
        l.m[2][2] = -forward.2;
        l.m[3][3] = 1.0;
        self.mul_assign(&l);
        self.translate(-eye.0, -eye.1, -eye.2);
    }

    // ---- inverse and friends -------------------------------------------------

    pub fn transposed(&self) -> M4 {
        let mut m = [[0.0f32; 4]; 4];
        for r in 0..4 {
            for c in 0..4 {
                m[c][r] = self.m[r][c];
            }
        }
        M4 { m, flags: if self.flags & TRANSLATION != 0 { GENERAL } else { self.flags } }
    }

    pub fn determinant(&self) -> f64 {
        let f = self.flags;
        let m = &self.m;
        if f & !(TRANSLATION | ROTATION_2D | ROTATION) == IDENTITY {
            return 1.0;
        }
        if f < ROTATION_2D {
            return 1.0 * m[0][0] as f64 * m[1][1] as f64 * m[2][2] as f64;
        }
        let mm = to_doubles(m);
        if f < PERSPECTIVE {
            return det3(&mm, 0, 1, 2, 0, 1, 2);
        }
        det4(&mm)
    }

    fn orthonormal_inverse(&self) -> M4 {
        let m = &self.m;
        let mut r = [[0.0f32; 4]; 4];
        r[0][0] = m[0][0];
        r[1][0] = m[0][1];
        r[2][0] = m[0][2];
        r[0][1] = m[1][0];
        r[1][1] = m[1][1];
        r[2][1] = m[1][2];
        r[0][2] = m[2][0];
        r[1][2] = m[2][1];
        r[2][2] = m[2][2];
        r[3][0] = -(r[0][0] * m[3][0] + r[1][0] * m[3][1] + r[2][0] * m[3][2]);
        r[3][1] = -(r[0][1] * m[3][0] + r[1][1] * m[3][1] + r[2][1] * m[3][2]);
        r[3][2] = -(r[0][2] * m[3][0] + r[1][2] * m[3][1] + r[2][2] * m[3][2]);
        r[3][3] = 1.0;
        M4 { m: r, flags: self.flags }
    }

    /// `inverted(&invertible)`: the inverse, or the identity and `false`.
    pub fn inverted(&self) -> (M4, bool) {
        let f = self.flags;
        let m = &self.m;
        if f == IDENTITY {
            return (M4::identity(), true);
        }
        if f == TRANSLATION {
            let mut inv = M4::identity();
            inv.m[3][0] = -m[3][0];
            inv.m[3][1] = -m[3][1];
            inv.m[3][2] = -m[3][2];
            inv.flags = TRANSLATION;
            return (inv, true);
        }
        if f < ROTATION_2D {
            if m[0][0] == 0.0 || m[1][1] == 0.0 || m[2][2] == 0.0 {
                return (M4::identity(), false);
            }
            let mut inv = M4::identity();
            inv.m[0][0] = 1.0 / m[0][0];
            inv.m[1][1] = 1.0 / m[1][1];
            inv.m[2][2] = 1.0 / m[2][2];
            inv.m[3][0] = -m[3][0] * inv.m[0][0];
            inv.m[3][1] = -m[3][1] * inv.m[1][1];
            inv.m[3][2] = -m[3][2] * inv.m[2][2];
            inv.flags = f;
            return (inv, true);
        }
        if f & !(TRANSLATION | ROTATION_2D | ROTATION) == IDENTITY {
            return (self.orthonormal_inverse(), true);
        }
        let mm = to_doubles(m);
        if f < PERSPECTIVE {
            let mut det = det3(&mm, 0, 1, 2, 0, 1, 2);
            if det == 0.0 {
                return (M4::identity(), false);
            }
            det = 1.0 / det;
            let mut inv = [[0.0f32; 4]; 4];
            inv[0][0] = (det2(&mm, 1, 2, 1, 2) * det) as f32;
            inv[0][1] = (-det2(&mm, 0, 2, 1, 2) * det) as f32;
            inv[0][2] = (det2(&mm, 0, 1, 1, 2) * det) as f32;
            inv[1][0] = (-det2(&mm, 1, 2, 0, 2) * det) as f32;
            inv[1][1] = (det2(&mm, 0, 2, 0, 2) * det) as f32;
            inv[1][2] = (-det2(&mm, 0, 1, 0, 2) * det) as f32;
            inv[2][0] = (det2(&mm, 1, 2, 0, 1) * det) as f32;
            inv[2][1] = (-det2(&mm, 0, 2, 0, 1) * det) as f32;
            inv[2][2] = (det2(&mm, 0, 1, 0, 1) * det) as f32;
            inv[3][0] = -inv[0][0] * m[3][0] - inv[1][0] * m[3][1] - inv[2][0] * m[3][2];
            inv[3][1] = -inv[0][1] * m[3][0] - inv[1][1] * m[3][1] - inv[2][1] * m[3][2];
            inv[3][2] = -inv[0][2] * m[3][0] - inv[1][2] * m[3][1] - inv[2][2] * m[3][2];
            inv[3][3] = 1.0;
            return (M4 { m: inv, flags: f }, true);
        }
        let mut det = det4(&mm);
        if det == 0.0 {
            return (M4::identity(), false);
        }
        det = 1.0 / det;
        let d = |v: f64| (v * det) as f32;
        let mut inv = [[0.0f32; 4]; 4];
        inv[0][0] = d(det3(&mm, 1, 2, 3, 1, 2, 3));
        inv[0][1] = d(-det3(&mm, 0, 2, 3, 1, 2, 3));
        inv[0][2] = d(det3(&mm, 0, 1, 3, 1, 2, 3));
        inv[0][3] = d(-det3(&mm, 0, 1, 2, 1, 2, 3));
        inv[1][0] = d(-det3(&mm, 1, 2, 3, 0, 2, 3));
        inv[1][1] = d(det3(&mm, 0, 2, 3, 0, 2, 3));
        inv[1][2] = d(-det3(&mm, 0, 1, 3, 0, 2, 3));
        inv[1][3] = d(det3(&mm, 0, 1, 2, 0, 2, 3));
        inv[2][0] = d(det3(&mm, 1, 2, 3, 0, 1, 3));
        inv[2][1] = d(-det3(&mm, 0, 2, 3, 0, 1, 3));
        inv[2][2] = d(det3(&mm, 0, 1, 3, 0, 1, 3));
        inv[2][3] = d(-det3(&mm, 0, 1, 2, 0, 1, 3));
        inv[3][0] = d(-det3(&mm, 1, 2, 3, 0, 1, 2));
        inv[3][1] = d(det3(&mm, 0, 2, 3, 0, 1, 2));
        inv[3][2] = d(-det3(&mm, 0, 1, 3, 0, 1, 2));
        inv[3][3] = d(det3(&mm, 0, 1, 2, 0, 1, 2));
        (M4 { m: inv, flags: f }, true)
    }

    /// `optimize()`: recover the type flags from the values.
    pub fn optimize(&mut self) {
        let m = &self.m;
        self.flags = GENERAL;
        if m[0][3] != 0.0 || m[1][3] != 0.0 || m[2][3] != 0.0 || m[3][3] != 1.0 {
            return;
        }
        self.flags &= !PERSPECTIVE;
        if m[3][0] == 0.0 && m[3][1] == 0.0 && m[3][2] == 0.0 {
            self.flags &= !TRANSLATION;
        }
        if m[0][2] == 0.0 && m[1][2] == 0.0 && m[2][0] == 0.0 && m[2][1] == 0.0 {
            self.flags &= !ROTATION;
            if m[0][1] == 0.0 && m[1][0] == 0.0 {
                self.flags &= !ROTATION_2D;
                if m[0][0] == 1.0 && m[1][1] == 1.0 && m[2][2] == 1.0 {
                    self.flags &= !SCALE;
                }
            } else {
                let mm = to_doubles(m);
                let det = det2(&mm, 0, 1, 0, 1);
                let len_x = mm[0][0] * mm[0][0] + mm[0][1] * mm[0][1];
                let len_y = mm[1][0] * mm[1][0] + mm[1][1] * mm[1][1];
                let len_z = mm[2][2];
                if fuzzy_compare(det, 1.0) && fuzzy_compare(len_x, 1.0) && fuzzy_compare(len_y, 1.0) && fuzzy_compare(len_z, 1.0)
                {
                    self.flags &= !SCALE;
                }
            }
        } else {
            let mm = to_doubles(m);
            let det = det3(&mm, 0, 1, 2, 0, 1, 2);
            let len_x = mm[0][0] * mm[0][0] + mm[0][1] * mm[0][1] + mm[0][2] * mm[0][2];
            let len_y = mm[1][0] * mm[1][0] + mm[1][1] * mm[1][1] + mm[1][2] * mm[1][2];
            let len_z = mm[2][0] * mm[2][0] + mm[2][1] * mm[2][1] + mm[2][2] * mm[2][2];
            if fuzzy_compare(det, 1.0) && fuzzy_compare(len_x, 1.0) && fuzzy_compare(len_y, 1.0) && fuzzy_compare(len_z, 1.0) {
                self.flags &= !SCALE;
            }
        }
    }
}
