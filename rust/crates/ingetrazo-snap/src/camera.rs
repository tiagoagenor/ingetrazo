// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! The camera and the projections the snap engine asks of the viewport:
//! `core/camera.py` (`OrbitCamera`'s eye and matrices) and the pixel
//! functions of `views/viewport.py` (`_world_to_pixel`, `_pixel_to_ray`,
//! `_project_to_lock_line`, `_ray_plane`, `_clip_segment_front`,
//! `_guide_snap_segment`).
//!
//! Qt's matrices are float32 with a type flag, so the camera builds its
//! matrices with the same `QMatrix4x4` calls Python makes (`lookAt`,
//! `perspective`, `ortho`) and multiplies them as `proj * view`. The Python
//! scalars (`yaw`, `distance`, `fov_deg`, the pixel arithmetic) are
//! float64.

use ingetrazo_math::pyfloat::{py_max2, py_min2};
use ingetrazo_math::{Hypot, M4, V3};

use crate::compute::radians;

/// `FAR_PER_DISTANCE`.
const FAR_PER_DISTANCE: f64 = 4.0;

/// `QMatrix4x4::map(QVector4D)`: the full product, no type-flag shortcut.
pub fn map_v4(m: &M4, v: [f32; 4]) -> [f32; 4] {
    let a = &m.m;
    let (x, y, z, w) = (v[0], v[1], v[2], v[3]);
    [
        x * a[0][0] + y * a[1][0] + z * a[2][0] + w * a[3][0],
        x * a[0][1] + y * a[1][1] + z * a[2][1] + w * a[3][1],
        x * a[0][2] + y * a[1][2] + z * a[2][2] + w * a[3][2],
        x * a[0][3] + y * a[1][3] + z * a[2][3] + w * a[3][3],
    ]
}

/// `OrbitCamera` (the two-point perspective mode is not ported: its frustum
/// shift is a view-camera feature no snap sample uses).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OrbitCamera {
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

impl Default for OrbitCamera {
    fn default() -> Self {
        OrbitCamera {
            target: V3(0.0, 0.0, 0.0),
            distance: 20.0,
            yaw: radians(-45.0),
            pitch: radians(30.0),
            up: V3(0.0, 0.0, 1.0),
            fov_deg: 45.0,
            aspect: 1.0,
            znear: 0.1,
            zfar: 10000.0,
            perspective: true,
        }
    }
}

impl OrbitCamera {
    /// `eye()`.
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

    /// `forward()`: the unit view direction, eye -> target.
    pub fn forward(&self, h: Hypot) -> V3 {
        let d = self.target.sub(self.eye());
        if (d.length_squared() as f64) > 1e-18 {
            d.normalized(h)
        } else {
            V3(0.0, 1.0, 0.0)
        }
    }

    /// `up_vector()`.
    pub fn up_vector(&self, h: Hypot) -> V3 {
        let f = self.forward(h);
        if (f.dot(self.up) as f64).abs() < 0.9999 {
            return self.up;
        }
        let sign = if self.pitch > 0.0 { -1.0 } else { 1.0 };
        V3((sign * self.yaw.cos()) as f32, (sign * self.yaw.sin()) as f32, 0.0)
    }

    /// `far_plane()`.
    pub fn far_plane(&self) -> f64 {
        py_max2(self.zfar, self.distance * FAR_PER_DISTANCE)
    }

    /// `view_matrix()`.
    pub fn view_matrix(&self, h: Hypot) -> M4 {
        let mut m = M4::identity();
        m.look_at(self.eye(), self.target, self.up_vector(h), h);
        m
    }

    /// `projection_matrix()`.
    pub fn projection_matrix(&self) -> M4 {
        let mut m = M4::identity();
        if self.perspective {
            let near = py_min2(self.znear, py_max2(self.distance * 0.02, 1e-4));
            m.perspective(self.fov_deg as f32, self.aspect as f32, near as f32, self.far_plane() as f32);
        } else {
            let half_h = self.distance * (radians(self.fov_deg) / 2.0).tan();
            let half_w = half_h * self.aspect;
            let far = self.far_plane();
            m.ortho(-half_w as f32, half_w as f32, -half_h as f32, half_h as f32, -far as f32, far as f32);
        }
        m
    }
}

/// The viewport's size and view: what every pixel question is asked
/// against.
#[derive(Clone, Copy, Debug)]
pub struct View {
    mvp: M4,
    inv: (M4, bool),
    pub cam: OrbitCamera,
    /// `self.width()` / `self.height()`: whole pixels.
    pub width: i32,
    pub height: i32,
    pub h: Hypot,
}

impl View {
    /// A view of `cam` in a `width` x `height` widget; `aspect` is the
    /// camera's own (the viewport sets it on resize).
    pub fn new(cam: OrbitCamera, width: i32, height: i32, h: Hypot) -> View {
        let mvp = cam.projection_matrix().mul(&cam.view_matrix(h));
        View { mvp, inv: mvp.inverted(), cam, width, height, h }
    }

    /// `projection_matrix() * view_matrix()`.
    pub fn mvp(&self) -> M4 {
        self.mvp
    }

    /// `_world_to_pixel`: `None` behind the camera.
    pub fn world_to_pixel(&self, w: V3) -> Option<(f64, f64)> {
        let clip = map_v4(&self.mvp(), [w.0, w.1, w.2, 1.0]);
        if (clip[3] as f64) <= 0.0 {
            return None;
        }
        let ndc_x = clip[0] as f64 / clip[3] as f64;
        let ndc_y = clip[1] as f64 / clip[3] as f64;
        let px = (ndc_x * 0.5 + 0.5) * self.width as f64;
        let py = (1.0 - (ndc_y * 0.5 + 0.5)) * self.height as f64;
        Some((px, py))
    }

    /// `_pixel_to_ray`: the camera ray `(origin, unit direction)` through a
    /// pixel.
    pub fn pixel_to_ray(&self, x: f64, y: f64) -> Option<(V3, V3)> {
        let w = self.width.max(1) as f64;
        let hh = self.height.max(1) as f64;
        let ndc_x = 2.0 * x / w - 1.0;
        let ndc_y = 1.0 - 2.0 * y / hh;
        let (inv, ok) = self.inv;
        if !ok {
            return None;
        }
        let q = |d: f64| V3(ndc_x as f32, ndc_y as f32, d as f32);
        let p_near = inv.map(q(-1.0));
        let finite = |v: V3| v.0.is_finite() && v.1.is_finite() && v.2.is_finite();
        let mut direction = None;
        for depth in [1.0, 0.999, 0.9, 0.0] {
            let d = inv.map(q(depth)).sub(p_near);
            if finite(d) && (d.length(self.h) as f64) >= 1e-9 {
                direction = Some(d);
                break;
            }
        }
        let direction = direction?;
        if !finite(p_near) {
            return None;
        }
        Some((p_near, direction.normalized(self.h)))
    }

    /// `_project_to_lock_line`: the closest point of the lock line to the
    /// camera ray through the pixel.
    pub fn project_to_lock_line(&self, start: V3, lock_dir: V3, px: f64, py: f64) -> V3 {
        let Some((ray_origin, ray_dir)) = self.pixel_to_ray(px, py) else {
            return start;
        };
        let d1 = lock_dir.normalized(self.h);
        let d2 = ray_dir;
        let r = start.sub(ray_origin);
        let b = d1.dot(d2) as f64;
        let d = d1.dot(r) as f64;
        let e = d2.dot(r) as f64;
        let denom = 1.0 - b * b;
        let t = if denom.abs() < 1e-6 { -d } else { (b * e - d) / denom };
        start.add(d1.mul(t as f32))
    }

    /// `_ray_plane`.
    pub fn ray_plane(&self, origin: V3, direction: V3, plane_point: V3, plane_normal: V3, allow_grazing: bool) -> Option<V3> {
        ray_plane(self.h, origin, direction, plane_point, plane_normal, allow_grazing)
    }

    /// `_clip_segment_front`: the part of a segment in front of the camera.
    pub fn clip_segment_front(&self, a: V3, b: V3) -> Option<(V3, V3)> {
        let mvp = self.mvp();
        let wa = map_v4(&mvp, [a.0, a.1, a.2, 1.0])[3] as f64;
        let wb = map_v4(&mvp, [b.0, b.1, b.2, 1.0])[3] as f64;
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

    /// `_guide_snap_segment`: the span of a guide line inside the view,
    /// computed in float64 from the guide's own point and direction.
    pub fn guide_snap_segment(&self, point: V3, direction: V3, margin: f64) -> Option<(V3, V3)> {
        let mvp = self.mvp();
        let c0 = map_v4(&mvp, [point.0, point.1, point.2, 1.0]);
        let c1 = map_v4(&mvp, [direction.0, direction.1, direction.2, 0.0]);
        let c0 = (c0[0] as f64, c0[1] as f64, c0[3] as f64);
        let c1 = (c1[0] as f64, c1[1] as f64, c1[3] as f64);
        let half = ingetrazo_model::guide::GUIDE_HALF_LEN as f64;
        let (mut lo, mut hi) = (-half, half);
        let planes = [
            (c0.2 - 1e-3, c1.2),
            (margin * c0.2 - c0.0, margin * c1.2 - c1.0),
            (margin * c0.2 + c0.0, margin * c1.2 + c1.0),
            (margin * c0.2 - c0.1, margin * c1.2 - c1.1),
            (margin * c0.2 + c0.1, margin * c1.2 + c1.1),
        ];
        for (f0, f1) in planes {
            if f1.abs() < 1e-15 {
                if f0 < 0.0 {
                    return None;
                }
                continue;
            }
            let s = -f0 / f1;
            if f1 > 0.0 {
                lo = py_max2(lo, s);
            } else {
                hi = py_min2(hi, s);
            }
            if lo >= hi {
                return None;
            }
        }
        let (px, py, pz) = (point.0 as f64, point.1 as f64, point.2 as f64);
        let (ux, uy, uz) = (direction.0 as f64, direction.1 as f64, direction.2 as f64);
        Some((
            V3((px + ux * lo) as f32, (py + uy * lo) as f32, (pz + uz * lo) as f32),
            V3((px + ux * hi) as f32, (py + uy * hi) as f32, (pz + uz * hi) as f32),
        ))
    }
}

/// `GRAZING_PLANE_DEG`.
pub const GRAZING_PLANE_DEG: f64 = 6.0;

/// `Viewport._ray_plane`.
pub fn ray_plane(h: Hypot, origin: V3, direction: V3, plane_point: V3, plane_normal: V3, allow_grazing: bool) -> Option<V3> {
    let length = plane_normal.length(h) as f64;
    if length < 1e-9 {
        return None;
    }
    let normal = plane_normal.div(length as f32);
    let denom = normal.dot(direction) as f64;
    if denom.abs() < 1e-12 {
        return None;
    }
    if !allow_grazing && denom.abs() < radians(GRAZING_PLANE_DEG).sin() {
        return None;
    }
    let t = normal.dot(plane_point.sub(origin)) as f64 / denom;
    if t < 0.0 {
        return None;
    }
    Some(origin.add(direction.mul(t as f32)))
}
