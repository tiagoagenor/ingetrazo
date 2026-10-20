// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! The scene questions of the camera gestures, answered from the pick index
//! of `ingetrazo-snap` (`Viewport._orbit_pivot_at`, `_world_under_cursor`,
//! `_model_diag`). Like the Python viewport with no frame to read the depth
//! from, the answers come from rays against the triangles.

use ingetrazo_geom::hlr_draw::OrbitCamera as PickOrbit;
use ingetrazo_math::{Hypot, V3};
use ingetrazo_model::scene::{MeshId, Scene};
use ingetrazo_render::Camera;
use ingetrazo_snap::picking::numeric::p3;
use ingetrazo_snap::picking::ray::Hits;
use ingetrazo_snap::picking::{Camera as PickCamera, ChunkCache, FaceMode, PickIndex, Picker, State, View, PICK_THRESHOLD_PX};

use crate::nav::{pixel_to_ray, Probe};

/// The pick index of a scene, built when first needed and kept until the
/// scene changes ([`PickCache::invalidate`]).
#[derive(Default)]
pub struct PickCache {
    idx: Option<PickIndex>,
    chunks: ChunkCache,
    /// Bumped by [`PickCache::invalidate`]; `scene.version`.
    pub version: u64,
}

impl PickCache {
    /// The scene object was replaced (a document was opened): serials restart,
    /// so the prototypes go too.
    pub fn reset(&mut self) {
        self.chunks.clear();
        self.invalidate();
    }

    /// The scene changed: drop the index and start a new version.
    pub fn invalidate(&mut self) {
        self.idx = None;
        self.version += 1;
    }
}

/// A [`Probe`] over one scene.
pub struct ScenePick<'a> {
    pub scene: &'a Scene,
    pub cache: &'a mut PickCache,
    pub hypot: Hypot,
}

impl<'a> ScenePick<'a> {
    pub fn new(scene: &'a Scene, cache: &'a mut PickCache, hypot: Hypot) -> ScenePick<'a> {
        ScenePick { scene, cache, hypot }
    }

    /// Run `f` on a picker over the (cached) index.
    fn with_picker<R>(&mut self, f: impl FnOnce(&Picker) -> R) -> R {
        let scene = self.scene;
        let st = State::new(scene, MeshId(0), MeshId(0), None, false, FaceMode::of_scene(scene), self.hypot);
        let idx = match self.cache.idx.take() {
            Some(i) => i,
            None => PickIndex::build_cached(&st, &mut self.cache.chunks),
        };
        // The view is for the picks that need a camera; the gestures bring
        // their own (the renderer's), two-point included.
        let orbit = PickOrbit { target: V3(0.0, 0.0, 0.0), distance: 20.0, yaw: 0.0, pitch: 0.0, up: V3(0.0, 0.0, 1.0) };
        let view = View::new(PickCamera::new(orbit), 1, 1, self.hypot);
        let picker = Picker { st, idx, view, threshold: PICK_THRESHOLD_PX };
        let r = f(&picker);
        self.cache.idx = Some(picker.idx);
        r
    }

    /// `Scene.bounds()`: the box of everything visible.
    pub fn bounds(&mut self) -> Option<(V3, V3)> {
        self.with_picker(|p| p.scene_bounds())
    }
}

/// `views.viewport._ray_aabb_span`: `(t_in, t_out)` of the forward ray
/// through the box, or `None` if it misses.
pub fn ray_aabb_span(o: [f64; 3], d: [f64; 3], lo: [f64; 3], hi: [f64; 3]) -> Option<(f64, f64)> {
    let (mut tmin, mut tmax) = (0.0f64, f64::INFINITY);
    for i in 0..3 {
        let di = d[i];
        if -1e-12 < di && di < 1e-12 {
            if o[i] < lo[i] - 1e-9 || o[i] > hi[i] + 1e-9 {
                return None;
            }
            continue;
        }
        let mut t1 = (lo[i] - o[i]) / di;
        let mut t2 = (hi[i] - o[i]) / di;
        if t1 > t2 {
            std::mem::swap(&mut t1, &mut t2);
        }
        tmin = if t1 > tmin { t1 } else { tmin };
        tmax = if t2 < tmax { t2 } else { tmax };
        if tmin > tmax {
            return None;
        }
    }
    Some((tmin, tmax))
}

impl Probe for ScenePick<'_> {
    fn scene_version(&self) -> u64 {
        self.cache.version
    }

    fn pivot_at(&mut self, cam: &Camera, size: (i32, i32), x: f64, y: f64) -> V3 {
        let ray = pixel_to_ray(cam, size, x, y);
        let central = pixel_to_ray(cam, size, size.0 as f64 / 2.0, size.1 as f64 / 2.0);
        self.with_picker(|p| {
            if let Some((origin, direction)) = ray {
                if !p.idx.ents.is_empty() {
                    if let Some(face_t) = p.hover_face_t(origin, direction) {
                        let t = face_t.iter().copied().fold(f64::INFINITY, |a, b| if b < a { b } else { a });
                        if !face_t.is_empty() && t.is_finite() && t > 0.0 {
                            return origin.add(direction.mul(t as f32));
                        }
                    }
                }
            }
            if let Some((lo, hi)) = p.scene_bounds() {
                if let Some((o, d)) = central {
                    if let Some(span) = ray_aabb_span(p3(o), p3(d), p3(lo), p3(hi)) {
                        return o.add(d.mul(((span.0 + span.1) * 0.5) as f32));
                    }
                }
            }
            cam.target
        })
    }

    fn world_under_cursor(&mut self, cam: &Camera, size: (i32, i32), x: f64, y: f64) -> Option<V3> {
        let (origin, direction) = pixel_to_ray(cam, size, x, y)?;
        let best = self.with_picker(|p| {
            if p.idx.ents.is_empty() {
                return None;
            }
            match p.ray_hits(p3(origin), p3(direction), &p.idx.vis, true) {
                Some(Hits::Global(t)) if t.is_finite() => Some(t),
                _ => None,
            }
        });
        if let Some(t) = best {
            return Some(origin.add(direction.mul(t as f32)));
        }
        let h = cam.hypot;
        if (direction.2 as f64).abs() > 1e-6 {
            let t = -(origin.2 as f64) / direction.2 as f64;
            if t > 0.0 {
                return Some(origin.add(direction.mul(t as f32)));
            }
        }
        let view = cam.target.sub(cam.eye());
        if view.length(h) as f64 > 1e-9 {
            let view = view.normalized(h);
            let denom = direction.dot(view) as f64;
            if denom.abs() > 1e-6 {
                let t = cam.target.sub(origin).dot(view) as f64 / denom;
                if t > 0.0 {
                    return Some(origin.add(direction.mul(t as f32)));
                }
            }
        }
        None
    }

    fn model_diag(&mut self) -> f64 {
        let h = self.hypot;
        match self.bounds() {
            Some((lo, hi)) => hi.sub(lo).length(h) as f64,
            None => 0.0,
        }
    }
}
