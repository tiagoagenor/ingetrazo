// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! The part of `QQuaternion` the orbit about a pivot uses
//! (`OrbitCamera.orbit_about`), in Qt's float32 arithmetic and operation
//! order: `fromAxisAndAngle`, the product, `rotatedVector`.

use ingetrazo_math::{Hypot, V3};

/// `QQuaternion`: scalar part first.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Quat {
    pub w: f32,
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

/// `qDegreesToRadians(float)`: `degrees * float(M_PI / 180)`.
fn degrees_to_radians(d: f32) -> f32 {
    d * (std::f64::consts::PI / 180.0) as f32
}

impl Quat {
    /// `QQuaternion::length()` as this Qt (MSVC build) compiles it: the
    /// four-argument `std::hypot` as a chain of `_Hypot3`-style steps over
    /// x, y, z, then w (read off `Qt6Gui.dll`, `QQuaternion::length`).
    pub fn length(self) -> f32 {
        fn step((m, s): (f32, f32), v: f32) -> (f32, f32) {
            if m.is_infinite() {
                return (m, 1.0);
            }
            if v.is_nan() {
                return (v, 1.0);
            }
            let x = v.abs();
            if m > 0.0 && !v.is_infinite() {
                if x > 0.0 {
                    if x > m {
                        let t = m / v;
                        (x, t * s * t + 1.0)
                    } else {
                        let t = v / m;
                        (m, t * t + s)
                    }
                } else {
                    (m, s)
                }
            } else {
                (x, 1.0)
            }
        }
        let mut acc = (self.x.abs(), 1.0f32);
        for v in [self.y, self.z, self.w] {
            acc = step(acc, v);
        }
        let (m, s) = acc;
        if !m.is_finite() {
            m
        } else if m <= 0.0 {
            0.0
        } else {
            s.sqrt() * m
        }
    }

    /// `QQuaternion::normalized()`: divides by the float length, always;
    /// the null quaternion when that is within `1e-5` of 0 (the shipped
    /// build has no "already unit" shortcut).
    pub fn normalized(self) -> Quat {
        let len = self.length();
        if len.abs() > 0.00001 {
            Quat { w: self.w / len, x: self.x / len, y: self.y / len, z: self.z / len }
        } else {
            Quat { w: 0.0, x: 0.0, y: 0.0, z: 0.0 }
        }
    }

    /// `QQuaternion::fromAxisAndAngle(QVector3D, float)`; the angle is in
    /// degrees.
    pub fn from_axis_and_angle(axis: V3, angle: f32, h: Hypot) -> Quat {
        let a = degrees_to_radians(angle / 2.0);
        let s = a.sin();
        let c = a.cos();
        let ax = axis.normalized(h);
        Quat { w: c, x: ax.0 * s, y: ax.1 * s, z: ax.2 * s }.normalized()
    }

    /// `q1 * q2`.
    pub fn mul(self, q2: Quat) -> Quat {
        let q1 = self;
        let ww = (q1.z + q1.x) * (q2.x + q2.y);
        let yy = (q1.w - q1.y) * (q2.w + q2.z);
        let zz = (q1.w + q1.y) * (q2.w - q2.z);
        let xx = ww + yy + zz;
        let qq = 0.5 * (xx + (q1.z - q1.x) * (q2.x - q2.y));
        let w = qq - ww + (q1.z - q1.y) * (q2.y - q2.z);
        let x = qq - xx + (q1.x + q1.w) * (q2.x + q2.w);
        let y = qq - yy + (q1.w - q1.x) * (q2.y + q2.z);
        let z = qq - zz + (q1.z + q1.y) * (q2.w - q2.x);
        Quat { w, x, y, z }
    }

    pub fn conjugated(self) -> Quat {
        Quat { w: self.w, x: -self.x, y: -self.y, z: -self.z }
    }

    /// `rotatedVector(v)`: `(q * QQuaternion(0, v) * q.conjugated()).vector()`.
    pub fn rotated_vector(self, v: V3) -> V3 {
        let r = self.mul(Quat { w: 0.0, x: v.0, y: v.1, z: v.2 }).mul(self.conjugated());
        V3(r.x, r.y, r.z)
    }
}
