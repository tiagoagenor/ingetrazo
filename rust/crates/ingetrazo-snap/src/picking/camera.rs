// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! The orbit camera (`core/camera.py`) and the pixel/world maps of the
//! viewport (`_pixel_to_ray`, `_world_to_pixel`, `_clip_segment_front`,
//! `_project_px`): what a pick needs to put a pixel in the scene and the
//! scene on the pixels.
//!
//! Two-point perspective (a view camera's frustum rise) is not ported; the
//! picks of a document in it are not covered.

use ingetrazo_geom::hlr_draw::OrbitCamera;
use ingetrazo_math::{Hypot, M4, V3};

use super::numeric::{matmul3, p3, py_max, py_min, P3};

/// `FAR_PER_DISTANCE`.
const FAR_PER_DISTANCE: f64 = 4.0;

/// An orbit camera with its projection settings.
#[derive(Clone, Copy, Debug)]
pub struct Camera {
    pub orbit: OrbitCamera,
    pub perspective: bool,
    pub fov_deg: f64,
    pub aspect: f64,
    pub znear: f64,
    pub zfar: f64,
}

impl Camera {
    /// `OrbitCamera()` of Python with the given pose.
    pub fn new(orbit: OrbitCamera) -> Camera {
        Camera { orbit, perspective: true, fov_deg: 45.0, aspect: 1.0, znear: 0.1, zfar: 10000.0 }
    }

    /// `set_aspect(w, h)`.
    pub fn set_aspect(&mut self, w: i32, h: i32) {
        self.aspect = w.max(1) as f64 / h.max(1) as f64;
    }

    pub fn eye(&self) -> V3 {
        self.orbit.eye()
    }

    /// `far_plane()`.
    pub fn far_plane(&self) -> f64 {
        py_max(self.zfar, self.orbit.distance * FAR_PER_DISTANCE)
    }

    /// `view_matrix()`.
    pub fn view_matrix(&self, h: Hypot) -> M4 {
        let mut m = M4::identity();
        m.look_at(self.orbit.eye(), self.orbit.target, self.orbit.up_vector(h), h);
        m
    }

    /// `projection_matrix()`.
    pub fn projection_matrix(&self) -> M4 {
        let mut m = M4::identity();
        if self.perspective {
            let near = py_min(self.znear, py_max(self.orbit.distance * 0.02, 1e-4));
            m.perspective(self.fov_deg as f32, self.aspect as f32, near as f32, self.far_plane() as f32);
        } else {
            let half_h = self.orbit.distance * (self.fov_deg * (std::f64::consts::PI / 180.0) / 2.0).tan();
            let half_w = half_h * self.aspect;
            let far = self.far_plane();
            m.ortho(-half_w as f32, half_w as f32, -half_h as f32, half_h as f32, -far as f32, far as f32);
        }
        m
    }
}

/// `QMatrix4x4 * QVector4D`: the four rows, left to right, in float32.
#[inline]
pub fn map4(m: &M4, v: [f32; 4]) -> [f32; 4] {
    let a = &m.m;
    [
        v[0] * a[0][0] + v[1] * a[1][0] + v[2] * a[2][0] + v[3] * a[3][0],
        v[0] * a[0][1] + v[1] * a[1][1] + v[2] * a[2][1] + v[3] * a[3][1],
        v[0] * a[0][2] + v[1] * a[1][2] + v[2] * a[2][2] + v[3] * a[3][2],
        v[0] * a[0][3] + v[1] * a[1][3] + v[2] * a[2][3] + v[3] * a[3][3],
    ]
}

/// A camera on a viewport of `w` x `h` pixels: its matrices, made once.
#[derive(Clone, Debug)]
pub struct View {
    pub w: i32,
    pub h: i32,
    pub cam: Camera,
    pub view: M4,
    pub proj: M4,
    pub mvp: M4,
    pub eye: V3,
    pub hyp: Hypot,
    /// `Viewport._np_mvp()`: `M[row][col]` in float64.
    mvp_np: [[f64; 4]; 4],
}

impl View {
    pub fn new(cam: Camera, w: i32, h: i32, hyp: Hypot) -> View {
        let view = cam.view_matrix(hyp);
        let proj = cam.projection_matrix();
        let mvp = proj.mul(&view);
        let mut np = [[0.0f64; 4]; 4];
        for r in 0..4 {
            for c in 0..4 {
                np[r][c] = mvp.m[c][r] as f64;
            }
        }
        View { w, h, cam, view, proj, mvp, eye: cam.eye(), hyp, mvp_np: np }
    }

    /// `_pixel_to_ray`: the camera ray (origin, unit direction) through a
    /// pixel, or `None` when it cannot be made.
    pub fn pixel_to_ray(&self, x: f64, y: f64) -> Option<(V3, V3)> {
        let w = self.w.max(1) as f64;
        let h = self.h.max(1) as f64;
        let ndc_x = 2.0 * x / w - 1.0;
        let ndc_y = 1.0 - 2.0 * y / h;
        let (inv, ok) = self.mvp.inverted();
        if !ok {
            return None;
        }
        let at = |depth: f64| inv.map(V3(ndc_x as f32, ndc_y as f32, depth as f32));
        let p_near = at(-1.0);
        let finite = |v: V3| v.0.is_finite() && v.1.is_finite() && v.2.is_finite();
        let mut direction = None;
        for depth in [1.0, 0.999, 0.9, 0.0] {
            let d = at(depth).sub(p_near);
            if finite(d) && (d.length(self.hyp) as f64) >= 1e-9 {
                direction = Some(d);
                break;
            }
        }
        let direction = direction?;
        if !finite(p_near) {
            return None;
        }
        Some((p_near, direction.normalized(self.hyp)))
    }

    /// `_world_to_pixel`: `None` behind the camera.
    pub fn world_to_pixel(&self, p: V3) -> Option<(f64, f64)> {
        let clip = map4(&self.mvp, [p.0, p.1, p.2, 1.0]);
        if (clip[3] as f64) <= 0.0 {
            return None;
        }
        let ndc_x = clip[0] as f64 / clip[3] as f64;
        let ndc_y = clip[1] as f64 / clip[3] as f64;
        let px = (ndc_x * 0.5 + 0.5) * self.w as f64;
        let py = (1.0 - (ndc_y * 0.5 + 0.5)) * self.h as f64;
        Some((px, py))
    }

    /// `_clip_segment_front`: the part of a segment in front of the camera.
    pub fn clip_segment_front(&self, a: V3, b: V3) -> Option<(V3, V3)> {
        let wa = map4(&self.mvp, [a.0, a.1, a.2, 1.0])[3] as f64;
        let wb = map4(&self.mvp, [b.0, b.1, b.2, 1.0])[3] as f64;
        let eps = 1e-3;
        if wa <= eps && wb <= eps {
            return None;
        }
        if wa <= eps || wb <= eps {
            let t = (eps - wa) / (wb - wa);
            let cut = a.add(b.sub(a).mul(t as f32));
            return Some(if wa <= eps { (cut, b) } else { (a, cut) });
        }
        Some((a, b))
    }

    /// What identifies this view for a cache of projections: the matrix and the size.
    pub fn key(&self) -> ([u64; 16], i32, i32) {
        let mut k = [0u64; 16];
        for r in 0..4 {
            for c in 0..4 {
                k[r * 4 + c] = self.mvp_np[r][c].to_bits();
            }
        }
        (k, self.w, self.h)
    }

    /// `_project_px`: world points to `(px, py, in_front)`.
    pub fn project_px(&self, pts: &[P3]) -> (Vec<f64>, Vec<f64>, Vec<bool>) {
        let n = pts.len();
        let m = &self.mvp_np;
        let mut px = Vec::with_capacity(n);
        let mut py = Vec::with_capacity(n);
        let mut ok = Vec::with_capacity(n);
        for p in pts {
            let clip = |k: usize| matmul3(*p, [m[k][0], m[k][1], m[k][2]], n) + m[k][3];
            let (cx, cy, w) = (clip(0), clip(1), clip(3));
            let good = w > 0.0;
            let safe = if good { w } else { 1.0 };
            px.push((cx / safe * 0.5 + 0.5) * self.w as f64);
            py.push((1.0 - (cy / safe * 0.5 + 0.5)) * self.h as f64);
            ok.push(good);
        }
        (px, py, ok)
    }

    /// The eye as float64.
    pub fn eye_p3(&self) -> P3 {
        p3(self.eye)
    }
}
