// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! `core/camera.py` `OrbitCamera`: the orbital camera and the view and
//! projection matrices the Python viewport hands to OpenGL, in Qt's float32
//! arithmetic (through `ingetrazo_math::M4`).
//!
//! The matrices are OpenGL's (clip z in -1..1). [`Camera::wgpu_mvp`] folds in
//! the fixed remap to wgpu's 0..1 depth range; everything else (the view, the
//! frustum, the near/far rules, two-point perspective) is the Python code.

use ingetrazo_math::matrix::GENERAL;
use ingetrazo_math::{Hypot, M4, V3};

/// How close the camera may get to what it is looking at (`MIN_DISTANCE`).
pub const MIN_DISTANCE: f64 = 0.02;
/// Ceiling of the orbit distance (`MAX_DISTANCE`).
pub const MAX_DISTANCE: f64 = 10000.0;
/// How far behind the orbit target the view still reaches, in orbit
/// distances (`FAR_PER_DISTANCE`).
pub const FAR_PER_DISTANCE: f64 = 4.0;

/// Two-point perspective gives way to the ordinary one past this pitch.
const TWO_POINT_MIN_COS: f32 = 0.2;

#[derive(Clone, Copy, Debug)]
pub struct Camera {
    pub target: V3,
    pub distance: f64,
    pub yaw: f64,
    pub pitch: f64,
    pub up: V3,
    pub fov_deg: f64,
    pub aspect: f64,
    pub znear: f64,
    pub zfar: f64,
    pub perspective: bool,
    pub two_point: bool,
    /// The platform Qt's `std::hypot` formula (lengths inside `lookAt`).
    pub hypot: Hypot,
}

impl Default for Camera {
    /// `OrbitCamera()`: yaw -45 deg, pitch 30 deg, 20 m away, 45 deg fov.
    fn default() -> Camera {
        Camera {
            target: V3(0.0, 0.0, 0.0),
            distance: 20.0,
            yaw: (-45.0f64).to_radians(),
            pitch: 30.0f64.to_radians(),
            up: V3(0.0, 0.0, 1.0),
            fov_deg: 45.0,
            aspect: 1.0,
            znear: 0.1,
            zfar: 10000.0,
            perspective: true,
            two_point: false,
            hypot: Hypot::default(),
        }
    }
}

impl Camera {
    /// `eye()`: `target + QVector3D(d*cp*cy, d*cp*sy, d*sp)`.
    pub fn eye(&self) -> V3 {
        let cp = self.pitch.cos();
        let sp = self.pitch.sin();
        let cy = self.yaw.cos();
        let sy = self.yaw.sin();
        self.target.add(V3(
            (self.distance * cp * cy) as f32,
            (self.distance * cp * sy) as f32,
            (self.distance * sp) as f32,
        ))
    }

    /// `forward()`: unit view direction, eye to target.
    pub fn forward(&self) -> V3 {
        let d = self.target.sub(self.eye());
        if (d.length_squared() as f64) > 1e-18 {
            d.normalized(self.hypot)
        } else {
            V3(0.0, 1.0, 0.0)
        }
    }

    /// `up_vector()`: `up`, unless the sight line runs along it (the exact
    /// Top and Bottom views).
    pub fn up_vector(&self) -> V3 {
        let f = self.forward();
        if (f.dot(self.up).abs() as f64) < 0.9999 {
            return self.up;
        }
        let sign: f64 = if self.pitch > 0.0 { -1.0 } else { 1.0 };
        V3((sign * self.yaw.cos()) as f32, (sign * self.yaw.sin()) as f32, 0.0)
    }

    /// `set_aspect(w, h)`.
    pub fn set_aspect(&mut self, w: u32, h: u32) {
        self.aspect = w.max(1) as f64 / h.max(1) as f64;
    }

    /// `far_plane()`: `zfar`, growing with the distance when zoomed out.
    pub fn far_plane(&self) -> f64 {
        self.zfar.max(self.distance * FAR_PER_DISTANCE)
    }

    /// `_level_forward()`: the level sight line of a two-point perspective.
    fn level_forward(&self) -> Option<V3> {
        if !(self.two_point && self.perspective) {
            return None;
        }
        let f = self.forward();
        let h = V3(f.0, f.1, 0.0);
        let len = h.length(self.hypot);
        if len < TWO_POINT_MIN_COS {
            return None;
        }
        Some(h.div(len))
    }

    /// `view_matrix()`.
    pub fn view_matrix(&self) -> M4 {
        let mut m = M4::identity();
        if let Some(level) = self.level_forward() {
            let eye = self.eye();
            m.look_at(eye, eye.add(level), V3(0.0, 0.0, 1.0), self.hypot);
            return m;
        }
        m.look_at(self.eye(), self.target, self.up_vector(), self.hypot);
        m
    }

    /// `projection_matrix()`.
    pub fn projection_matrix(&self) -> M4 {
        let mut m = M4::identity();
        if self.perspective {
            let near = self.znear.min((self.distance * 0.02).max(1e-4));
            let far = self.far_plane();
            if let Some(level) = self.level_forward() {
                let half_h = near * (self.fov_deg.to_radians() / 2.0).tan();
                let half_w = half_h * self.aspect;
                let rel = self.target.sub(self.eye());
                let depth = rel.dot(level) as f64;
                let shift = rel.2 as f64 * near / depth;
                frustum(
                    &mut m,
                    -half_w as f32,
                    half_w as f32,
                    (-half_h + shift) as f32,
                    (half_h + shift) as f32,
                    near as f32,
                    far as f32,
                );
                return m;
            }
            m.perspective(self.fov_deg as f32, self.aspect as f32, near as f32, far as f32);
        } else {
            let half_h = self.distance * (self.fov_deg.to_radians() / 2.0).tan();
            let half_w = half_h * self.aspect;
            let far = self.far_plane();
            m.ortho(
                -half_w as f32,
                half_w as f32,
                -half_h as f32,
                half_h as f32,
                -far as f32,
                far as f32,
            );
        }
        m
    }

    /// Projection times view: the `mvp` the viewport uploads (OpenGL clip
    /// space, z in -1..1).
    pub fn mvp(&self) -> M4 {
        self.projection_matrix().mul(&self.view_matrix())
    }

    /// [`Camera::mvp`] remapped to wgpu's depth range (0..1), column major.
    pub fn wgpu_mvp(&self) -> [f32; 16] {
        gl_to_wgpu(&self.mvp())
    }

    /// `projection_mode()`: "parallel", "perspective" or "two_point".
    pub fn projection_mode(&self) -> &'static str {
        if !self.perspective {
            "parallel"
        } else if self.two_point {
            "two_point"
        } else {
            "perspective"
        }
    }

    /// `fit_to`: centre on the box and back up enough to frame it (the
    /// "zoom extents" the viewport starts with).
    pub fn fit_to(&mut self, lo: V3, hi: V3, margin: f64) {
        let c = V3((lo.0 + hi.0) * 0.5, (lo.1 + hi.1) * 0.5, (lo.2 + hi.2) * 0.5);
        let mut diag = hi.sub(lo).length(self.hypot) as f64;
        if diag < MIN_DISTANCE {
            diag = MIN_DISTANCE;
        }
        self.target = c;
        let fov = self.fov_deg.to_radians();
        self.distance = (diag * margin / (2.0 * (fov / 2.0).tan())).max(MIN_DISTANCE);
    }
}

/// `QMatrix4x4::frustum` applied to an identity matrix.
fn frustum(m: &mut M4, left: f32, right: f32, bottom: f32, top: f32, near: f32, far: f32) {
    if left == right || bottom == top || near == far {
        return;
    }
    let width = right - left;
    let invheight = top - bottom;
    let clip = far - near;
    let mut f = M4 { m: [[0.0; 4]; 4], flags: GENERAL };
    f.m[0][0] = 2.0 * near / width;
    f.m[2][0] = (left + right) / width;
    f.m[1][1] = 2.0 * near / invheight;
    f.m[2][1] = (top + bottom) / invheight;
    f.m[2][2] = -(near + far) / clip;
    f.m[3][2] = -(2.0 * near * far) / clip;
    f.m[2][3] = -1.0;
    m.mul_assign(&f);
}

/// Remap clip z from OpenGL's -1..1 to wgpu's 0..1: `z' = (z + w) / 2`.
pub fn gl_to_wgpu(m: &M4) -> [f32; 16] {
    let mut d = m.data();
    for col in 0..4 {
        let z = d[col * 4 + 2];
        let w = d[col * 4 + 3];
        d[col * 4 + 2] = 0.5 * (z + w);
    }
    d
}

/// Multiply two column-major 4x4 matrices (`a * b`).
pub fn mat_mul(a: &[f32; 16], b: &[f32; 16]) -> [f32; 16] {
    let mut r = [0.0f32; 16];
    for c in 0..4 {
        for row in 0..4 {
            let mut s = 0.0;
            for k in 0..4 {
                s += a[k * 4 + row] * b[c * 4 + k];
            }
            r[c * 4 + row] = s;
        }
    }
    r
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eye_matches_the_spherical_formula() {
        let c = Camera { target: V3(1.0, 2.0, 3.0), distance: 10.0, yaw: 0.0, pitch: 0.0, ..Default::default() };
        let e = c.eye();
        assert!((e.0 - 11.0).abs() < 1e-5 && (e.1 - 2.0).abs() < 1e-5 && (e.2 - 3.0).abs() < 1e-5);
    }

    #[test]
    fn target_projects_to_the_screen_centre() {
        let c = Camera { target: V3(4.0, -2.0, 1.0), ..Default::default() };
        let p = c.mvp().map(c.target);
        assert!(p.0.abs() < 1e-5 && p.1.abs() < 1e-5, "{p:?}");
        let two = Camera { two_point: true, target: V3(4.0, -2.0, 1.0), ..Default::default() };
        let q = two.mvp().map(two.target);
        assert!(q.0.abs() < 1e-4 && q.1.abs() < 1e-4, "{q:?}");
    }

    #[test]
    fn wgpu_depth_is_zero_at_near_and_one_at_far() {
        let c = Camera { target: V3(0.0, 0.0, 0.0), ..Default::default() };
        let m = c.wgpu_mvp();
        let near = c.znear.min((c.distance * 0.02).max(1e-4)) as f32;
        let f = c.forward();
        let eye = c.eye();
        let at = |d: f32| {
            let p = [eye.0 + f.0 * d, eye.1 + f.1 * d, eye.2 + f.2 * d, 1.0];
            let mut o = [0.0f32; 4];
            for r in 0..4 {
                for k in 0..4 {
                    o[r] += m[k * 4 + r] * p[k];
                }
            }
            o[2] / o[3]
        };
        assert!(at(near).abs() < 1e-3);
        assert!((at(c.far_plane() as f32) - 1.0).abs() < 1e-3);
    }
}
