// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! The ray queries the snap engine makes of the pick index: occlusion
//! (`Viewport._is_occluded`) and the face under the cursor
//! (`_face_any_choice`, `_coplanar_eps`).
//!
//! Möller–Trumbore as `Viewport._ray_hits` writes it in NumPy: `p = cross(d,
//! e2)`, `det = e1 . p`, the hit tests `|det| > 1e-6`, `0 <= u <= 1`,
//! `v >= 0`, `u + v <= 1`, `t > 1e-6`. The box prefilter over the chunks
//! only skips work and is not reproduced; the 3-term sums run left to right
//! (NumPy's `einsum` may sum or fuse them differently in the last bit, which
//! `ingetrazo-geom`'s `occlusion` module bounds; a ray that close to a
//! triangle's edge is not one a fixture can tell apart).

use ingetrazo_math::{Hypot, V3};

use crate::camera::View;
use crate::world::World;

const DET_MIN: f64 = 1e-6;
const T_MIN: f64 = 1e-6;

#[inline]
fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}

/// `np.einsum("ij,ij->i", a, b)` of one row: NumPy's kernel sums the three
/// products as `(p0 + p2) + p1` (measured against NumPy 2.5.4, every row
/// count).
#[inline]
fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    (a[0] * b[0] + a[2] * b[2]) + a[1] * b[1]
}

/// `q @ d` for an `(n, 3)` array and a `(3,)` vector: OpenBLAS' `dgemv`
/// fuses the products as `fma(a2, b2, fma(a0, b0, a1 * b1))` for two or more
/// rows, and `ddot` for one adds them left to right (measured against NumPy
/// 2.5.4 on the machine of the fixtures).
#[inline]
fn matvec(a: [f64; 3], b: [f64; 3], single: bool) -> f64 {
    if single {
        (a[0] * b[0] + a[1] * b[1]) + a[2] * b[2]
    } else {
        a[2].mul_add(b[2], a[0].mul_add(b[0], a[1] * b[1]))
    }
}

/// One ray against one triangle: the hit parameter, if it is one.
#[inline]
fn hit(o: [f64; 3], d: [f64; 3], v0: [f64; 3], e1: [f64; 3], e2: [f64; 3], single: bool) -> Option<f64> {
    let p = cross(d, e2);
    let det = dot(e1, p);
    if !(det.abs() > DET_MIN) {
        return None;
    }
    let inv = 1.0 / det;
    let s = [o[0] - v0[0], o[1] - v0[1], o[2] - v0[2]];
    let u = dot(s, p) * inv;
    let q = cross(s, e1);
    let v = matvec(q, d, single) * inv;
    let t = dot(e2, q) * inv;
    if u >= 0.0 && u <= 1.0 && v >= 0.0 && u + v <= 1.0 && t > T_MIN {
        Some(t)
    } else {
        None
    }
}

fn arr(v: V3) -> [f64; 3] {
    [v.0 as f64, v.1 as f64, v.2 as f64]
}

impl World {
    /// The nearest hit of the ray against the triangles whose entity passes
    /// `mask` (`_ray_hits(..., reduce_global=True)`); infinity for none.
    pub fn nearest_hit(&self, o: [f64; 3], d: [f64; 3], mask: impl Fn(usize) -> bool) -> f64 {
        let mut best = f64::INFINITY;
        for t in &self.tris {
            if !mask(t.ent as usize) {
                continue;
            }
            if let Some(tt) = hit(o, d, t.v0, t.e1, t.e2, self.tris.len() == 1) {
                if tt < best {
                    best = tt;
                }
            }
        }
        best
    }

    /// Per entity, the nearest hit (`inf` for none): `_ray_hits` without
    /// `reduce_global`.
    pub fn entity_hits(&self, o: [f64; 3], d: [f64; 3], mask: impl Fn(usize) -> bool) -> Vec<f64> {
        let mut face_t = vec![f64::INFINITY; self.ents.len()];
        for t in &self.tris {
            let e = t.ent as usize;
            if !mask(e) {
                continue;
            }
            if let Some(tt) = hit(o, d, t.v0, t.e1, t.e2, self.tris.len() == 1) {
                if tt < face_t[e] {
                    face_t[e] = tt;
                }
            }
        }
        face_t
    }

    /// `Viewport._is_occluded(world)` with no section cut and a style that
    /// hides (not X-ray or wireframe).
    pub fn is_occluded(&self, eye: V3, world: V3, h: Hypot) -> bool {
        if self.tris.is_empty() {
            return false;
        }
        let delta = world.sub(eye);
        let dist = delta.length(h) as f64;
        if dist < 1e-9 {
            return false;
        }
        let d = delta.div(dist as f32);
        let nearest = self.nearest_hit(arr(eye), arr(d), |e| self.ents[e].vis);
        nearest < dist - 1e-3
    }

    /// `_coplanar_eps`: how much deeper than the nearest hit a face may sit
    /// and still be the one under the cursor.
    pub fn coplanar_eps(&self, view: &View, origin: V3, direction: V3, best_t: f64) -> f64 {
        let t_eye = view.cam.eye().sub(origin).dot(direction) as f64;
        let a = (best_t - t_eye).abs() * 1e-4;
        if a > 1e-4 {
            a
        } else {
            1e-4
        }
    }

    /// `_face_any_choice`: the entity of the front-most face under the
    /// cursor (the smallest of coplanar ones), if any.
    pub fn face_under(&self, view: &View, x: f64, y: f64) -> Option<usize> {
        if self.ents.is_empty() || self.tris.is_empty() {
            return None;
        }
        let (origin, direction) = view.pixel_to_ray(x, y)?;
        let face_t = self.entity_hits(arr(origin), arr(direction), |e| self.ents[e].sel);
        let mut best_t = f64::INFINITY;
        for &t in &face_t {
            if t < best_t {
                best_t = t;
            }
        }
        if !best_t.is_finite() {
            return None;
        }
        let eps = self.coplanar_eps(view, origin, direction, best_t);
        let mut chosen: Option<usize> = None;
        let mut n = 0;
        for (i, &t) in face_t.iter().enumerate() {
            if t <= best_t + eps {
                n += 1;
                // np.argmin over the candidates: the first of the smallest.
                match chosen {
                    None => chosen = Some(i),
                    Some(c) if self.ents[i].area < self.ents[c].area => chosen = Some(i),
                    _ => {}
                }
            }
        }
        let _ = n;
        chosen
    }
}
