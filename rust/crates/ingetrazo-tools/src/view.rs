// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! The camera and the pixel/ray maths the tools read off the viewport
//! (`core/camera.py`, and `Viewport._pixel_to_ray`, `_world_to_pixel`,
//! `_ray_hit_plane`, `_project_to_lock_line` of `views/viewport.py`).
//!
//! No drawing: matrices are `ingetrazo-math` M4 (Qt's `QMatrix4x4`, float32),
//! pixels and distances are the Python floats (f64) the tools compute with.
//! Two-Point Perspective (`Camera.two_point`) is not carried: it is a
//! presentation mode no tool decision depends on, and `M4` has no
//! `frustum`.

use ingetrazo_math::{Hypot, M4, V3};

/// `core.camera.FAR_PER_DISTANCE`.
pub const FAR_PER_DISTANCE: f64 = 4.0;

/// The orbit camera (`core.camera.Camera`).
#[derive(Clone, Debug)]
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
}

impl Default for Camera {
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
        }
    }
}

impl Camera {
    /// `Camera.eye()`.
    pub fn eye(&self) -> V3 {
        let cp = self.pitch.cos();
        let sp = self.pitch.sin();
        let cy = self.yaw.cos();
        let sy = self.yaw.sin();
        self.target.add(V3((self.distance * cp * cy) as f32, (self.distance * cp * sy) as f32, (self.distance * sp) as f32))
    }

    /// `Camera.forward()`: unit view direction, eye -> target.
    pub fn forward(&self, h: Hypot) -> V3 {
        let d = self.target.sub(self.eye());
        if (d.length_squared() as f64) > 1e-18 {
            d.normalized(h)
        } else {
            V3(0.0, 1.0, 0.0)
        }
    }

    /// `Camera.up_vector()`: the up vector every projection must use.
    pub fn up_vector(&self, h: Hypot) -> V3 {
        let f = self.forward(h);
        if (f.dot(self.up) as f64).abs() < 0.9999 {
            return self.up;
        }
        let sign = if self.pitch > 0.0 { -1.0 } else { 1.0 };
        V3((sign * self.yaw.cos()) as f32, (sign * self.yaw.sin()) as f32, 0.0)
    }

    /// `Camera.view_matrix()`.
    pub fn view_matrix(&self, h: Hypot) -> M4 {
        let mut m = M4::identity();
        m.look_at(self.eye(), self.target, self.up_vector(h), h);
        m
    }

    /// `Camera.far_plane()`.
    pub fn far_plane(&self) -> f64 {
        ingetrazo_math::pyfloat::py_max2(self.zfar, self.distance * FAR_PER_DISTANCE)
    }

    /// `Camera.projection_matrix()`.
    pub fn projection_matrix(&self) -> M4 {
        let mut m = M4::identity();
        if self.perspective {
            let near = ingetrazo_math::pyfloat::py_min2(self.znear, ingetrazo_math::pyfloat::py_max2(self.distance * 0.02, 1e-4));
            m.perspective(self.fov_deg as f32, self.aspect as f32, near as f32, self.far_plane() as f32);
        } else {
            let half_h = self.distance * (self.fov_deg.to_radians() / 2.0).tan();
            let half_w = half_h * self.aspect;
            let far = self.far_plane();
            m.ortho((-half_w) as f32, half_w as f32, (-half_h) as f32, half_h as f32, (-far) as f32, far as f32);
        }
        m
    }
}

/// A camera and the size of the viewport it looks through.
#[derive(Clone, Debug)]
pub struct View {
    pub camera: Camera,
    pub width: u32,
    pub height: u32,
    pub hypot: Hypot,
}

fn finite(v: V3) -> bool {
    v.0.is_finite() && v.1.is_finite() && v.2.is_finite()
}

impl View {
    pub fn new(camera: Camera, width: u32, height: u32, hypot: Hypot) -> View {
        View { camera, width, height, hypot }
    }

    fn mvp(&self) -> M4 {
        self.camera.projection_matrix().mul(&self.camera.view_matrix(self.hypot))
    }

    /// `_pixel_to_ray`: the camera ray `(origin, unit direction)` through a
    /// pixel.
    pub fn pixel_to_ray(&self, x: f64, y: f64) -> Option<(V3, V3)> {
        let w = self.width.max(1) as f64;
        let h = self.height.max(1) as f64;
        let ndc_x = 2.0 * x / w - 1.0;
        let ndc_y = 1.0 - 2.0 * y / h;
        let (inv, ok) = self.mvp().inverted();
        if !ok {
            return None;
        }
        let p_near = inv.map(V3(ndc_x as f32, ndc_y as f32, -1.0));
        // The far plane gives the most exact direction; a point just short of
        // it serves when the depth term rounds to exactly -1 (#583).
        let mut direction: Option<V3> = None;
        for depth in [1.0f64, 0.999, 0.9, 0.0] {
            let d = inv.map(V3(ndc_x as f32, ndc_y as f32, depth as f32)).sub(p_near);
            if finite(d) && (d.length(self.hypot) as f64) >= 1e-9 {
                direction = Some(d);
                break;
            }
        }
        let direction = direction?;
        if !finite(p_near) {
            return None;
        }
        Some((p_near, direction.normalized(self.hypot)))
    }

    /// `_world_to_pixel`: world point -> screen pixel (`None` behind the
    /// camera).
    pub fn world_to_pixel(&self, world: V3) -> Option<(f64, f64)> {
        let m = self.mvp();
        // `QMatrix4x4 * QVector4D`: column-major, no flag shortcut.
        let (c0, c1, c2, c3) = (m.column(0), m.column(1), m.column(2), m.column(3));
        let (x, y, z, w) = (world.0, world.1, world.2, 1.0f32);
        let cx = x * c0[0] + y * c1[0] + z * c2[0] + w * c3[0];
        let cy = x * c0[1] + y * c1[1] + z * c2[1] + w * c3[1];
        let cw = x * c0[3] + y * c1[3] + z * c2[3] + w * c3[3];
        if (cw as f64) <= 0.0 {
            return None;
        }
        let ndc_x = cx as f64 / cw as f64;
        let ndc_y = cy as f64 / cw as f64;
        Some(((ndc_x * 0.5 + 0.5) * self.width as f64, (1.0 - (ndc_y * 0.5 + 0.5)) * self.height as f64))
    }

    /// `_ray_hit_plane`: where the cursor ray meets `(point, normal)`.
    pub fn ray_hit_plane(&self, cursor: (f64, f64), plane: (V3, V3)) -> Option<V3> {
        let (origin, direction) = self.pixel_to_ray(cursor.0, cursor.1)?;
        let (point, normal) = plane;
        let denom = normal.dot(direction) as f64;
        if denom.abs() < 1e-6 {
            return None;
        }
        let t = (normal.dot(point.sub(origin)) as f64) / denom;
        if t < 0.0 {
            return None;
        }
        Some(origin.add(direction.mul(t as f32)))
    }

    /// `_project_to_lock_line`: the closest point on the lock line
    /// (`start`, `lock_dir`) to the camera ray through the pixel.
    pub fn project_to_lock_line(&self, start: V3, lock_dir: V3, px: f64, py: f64) -> V3 {
        let Some((ray_origin, ray_dir)) = self.pixel_to_ray(px, py) else { return start };
        let d1 = lock_dir.normalized(self.hypot);
        let d2 = ray_dir;
        let r = start.sub(ray_origin);
        let b = d1.dot(d2) as f64;
        let d = d1.dot(r) as f64;
        let e = d2.dot(r) as f64;
        let denom = 1.0 - b * b;
        let t = if denom.abs() < 1e-6 { -d } else { (b * e - d) / denom };
        start.add(d1.mul(t as f32))
    }

    /// `Viewport._coplanar_eps`: how much deeper than the nearest hit a face
    /// may sit and still count as the same depth.
    pub fn coplanar_eps(&self, origin: V3, direction: V3, best_t: f64) -> f64 {
        let t_eye = self.camera.eye().sub(origin).dot(direction) as f64;
        ingetrazo_math::pyfloat::py_max2(1e-4, (best_t - t_eye).abs() * 1e-4)
    }
}
