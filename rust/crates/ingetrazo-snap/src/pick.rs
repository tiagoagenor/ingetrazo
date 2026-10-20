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
use crate::world::{Tri, World};

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

/// From this many triangles on, a world culls rays with a [`TriGrid`]: the
/// fixtures of the parity suites are far below it and keep the plain scan.
pub const GRID_MIN_TRIS: usize = 4096;
/// Triangles per box of the grid at most.
const GRID_CHUNK: usize = 96;

/// Runs of consecutive triangles with the box around each, so a ray tests
/// the triangles of the runs whose box it meets and no others. Only a filter:
/// a triangle a ray hits is in a box the ray meets (the boxes are padded for
/// rounding), and every triangle that is tested gets the same test as in the
/// plain scan, so the answers are the same.
#[derive(Clone, Debug, Default)]
pub struct TriGrid {
    runs: Vec<Run>,
}

#[derive(Clone, Copy, Debug)]
struct Run {
    lo: [f64; 3],
    hi: [f64; 3],
    start: u32,
    n: u32,
}

impl TriGrid {
    fn build(w: &World) -> TriGrid {
        let place_of = |ent: u32| match w.ents[ent as usize].face {
            crate::world::FaceRef::Placed { place, .. } => place,
            crate::world::FaceRef::Loose(_) => usize::MAX,
        };
        let mut runs = Vec::new();
        let mut i = 0usize;
        while i < w.tris.len() {
            let place = place_of(w.tris[i].ent);
            let mut j = i;
            let mut lo = [f64::INFINITY; 3];
            let mut hi = [f64::NEG_INFINITY; 3];
            while j < w.tris.len() && j - i < GRID_CHUNK && place_of(w.tris[j].ent) == place {
                let t = &w.tris[j];
                for k in 0..3 {
                    let (a, b, c) = (t.v0[k], t.v0[k] + t.e1[k], t.v0[k] + t.e2[k]);
                    lo[k] = lo[k].min(a).min(b).min(c);
                    hi[k] = hi[k].max(a).max(b).max(c);
                }
                j += 1;
            }
            for k in 0..3 {
                let pad = 1e-7 + 1e-9 * lo[k].abs().max(hi[k].abs());
                lo[k] -= pad;
                hi[k] += pad;
            }
            runs.push(Run { lo, hi, start: i as u32, n: (j - i) as u32 });
            i = j;
        }
        TriGrid { runs }
    }

    /// Whether the forward ray `o + t d` (`t >= 0`) meets the box of `r`.
    #[inline]
    fn meets(r: &Run, o: [f64; 3], d: [f64; 3]) -> bool {
        let mut tmin = 0.0f64;
        let mut tmax = f64::INFINITY;
        for k in 0..3 {
            if d[k].abs() < 1e-300 {
                if o[k] < r.lo[k] || o[k] > r.hi[k] {
                    return false;
                }
                continue;
            }
            let inv = 1.0 / d[k];
            let t1 = (r.lo[k] - o[k]) * inv;
            let t2 = (r.hi[k] - o[k]) * inv;
            let (a, b) = if t1 < t2 { (t1, t2) } else { (t2, t1) };
            if a > tmin {
                tmin = a;
            }
            if b < tmax {
                tmax = b;
            }
            if tmin > tmax {
                return false;
            }
        }
        true
    }
}

impl World {
    /// The grid of a big world, built on first use (`None` for a small one).
    fn grid(&self) -> Option<&TriGrid> {
        if self.tris.len() < GRID_MIN_TRIS {
            return None;
        }
        Some(self.grid.get_or_init(|| TriGrid::build(self)))
    }

    /// The triangles a ray has to test: all of them, or the runs of the grid
    /// whose box it meets. Calls `f` on each triangle to test.
    #[inline]
    fn for_candidates(&self, grid: Option<&TriGrid>, o: [f64; 3], d: [f64; 3], mut f: impl FnMut(&Tri)) {
        match grid {
            Some(g) => {
                for r in &g.runs {
                    if TriGrid::meets(r, o, d) {
                        self.tris[r.start as usize..(r.start + r.n) as usize].iter().for_each(&mut f);
                    }
                }
            }
            None => self.tris.iter().for_each(f),
        }
    }

    fn nearest_hit_via(&self, grid: Option<&TriGrid>, o: [f64; 3], d: [f64; 3], mask: impl Fn(usize) -> bool) -> f64 {
        let mut best = f64::INFINITY;
        let single = self.tris.len() == 1;
        self.for_candidates(grid, o, d, |t| {
            if !mask(t.ent as usize) {
                return;
            }
            if let Some(tt) = hit(o, d, t.v0, t.e1, t.e2, single) {
                if tt < best {
                    best = tt;
                }
            }
        });
        best
    }

    fn entity_hits_via(&self, grid: Option<&TriGrid>, o: [f64; 3], d: [f64; 3], mask: impl Fn(usize) -> bool) -> Vec<f64> {
        let mut face_t = vec![f64::INFINITY; self.ents.len()];
        let single = self.tris.len() == 1;
        self.for_candidates(grid, o, d, |t| {
            let e = t.ent as usize;
            if !mask(e) {
                return;
            }
            if let Some(tt) = hit(o, d, t.v0, t.e1, t.e2, single) {
                if tt < face_t[e] {
                    face_t[e] = tt;
                }
            }
        });
        face_t
    }

    /// The nearest hit of the ray against the triangles whose entity passes
    /// `mask` (`_ray_hits(..., reduce_global=True)`); infinity for none.
    pub fn nearest_hit(&self, o: [f64; 3], d: [f64; 3], mask: impl Fn(usize) -> bool) -> f64 {
        self.nearest_hit_via(self.grid(), o, d, mask)
    }

    /// [`World::nearest_hit`] over every triangle, grid or no grid (what the
    /// tests compare the grid with).
    pub fn nearest_hit_scan(&self, o: [f64; 3], d: [f64; 3], mask: impl Fn(usize) -> bool) -> f64 {
        self.nearest_hit_via(None, o, d, mask)
    }

    /// Per entity, the nearest hit (`inf` for none): `_ray_hits` without
    /// `reduce_global`.
    pub fn entity_hits(&self, o: [f64; 3], d: [f64; 3], mask: impl Fn(usize) -> bool) -> Vec<f64> {
        self.entity_hits_via(self.grid(), o, d, mask)
    }

    /// [`World::entity_hits`] over every triangle.
    pub fn entity_hits_scan(&self, o: [f64; 3], d: [f64; 3], mask: impl Fn(usize) -> bool) -> Vec<f64> {
        self.entity_hits_via(None, o, d, mask)
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
