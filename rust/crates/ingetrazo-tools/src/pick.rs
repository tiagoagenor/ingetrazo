// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! Face picking: what `Viewport.pick_face_any` answers a tool.
//!
//! The viewport owns the real picker (a cached index over the whole scene,
//! groups included). A tool only needs the answer, so it asks a [`Picker`];
//! tests and headless hosts use [`RayPicker`], the straightforward form of
//! the same rule: the front-most face under the cursor, and among faces at
//! the same depth (coplanar faces overlapping at the cursor) the smallest.

use ingetrazo_model::mesh::{FaceId, Mesh};
use ingetrazo_model::topology::PFace;
use ingetrazo_ops::pushpull::ray_triangle;

use crate::view::View;

/// A picked face. `closed_group` is the group half of `pick_face_any`: the
/// face lives inside a group or component that is not open, so the tool
/// refuses it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Pick {
    pub face: FaceId,
    pub closed_group: bool,
}

/// `viewport.pick_face_any(x, y)`.
pub trait Picker {
    fn pick_face_any(&self, view: Option<&View>, mesh: &Mesh, x: f64, y: f64) -> Option<Pick>;
}

/// A picker that always answers the same (the stub viewports of the tests).
pub struct FixedPick(pub Option<Pick>);

impl Picker for FixedPick {
    fn pick_face_any(&self, _view: Option<&View>, _mesh: &Mesh, _x: f64, _y: f64) -> Option<Pick> {
        self.0
    }
}

/// Ray casting against the faces of the active mesh.
pub struct RayPicker;

impl Picker for RayPicker {
    fn pick_face_any(&self, view: Option<&View>, mesh: &Mesh, x: f64, y: f64) -> Option<Pick> {
        let view = view?;
        let (origin, direction) = view.pixel_to_ray(x, y)?;
        let h = view.hypot;
        let mut hits: Vec<(FaceId, f64)> = Vec::new();
        for &f in &mesh.faces {
            let mut best: Option<f64> = None;
            for tri in PFace::of(mesh, f).triangulate(h) {
                if let Some(t) = ray_triangle(origin, direction, &tri) {
                    if best.is_none_or(|b| t < b) {
                        best = Some(t);
                    }
                }
            }
            if let Some(t) = best {
                hits.push((f, t));
            }
        }
        let best_t = hits.iter().map(|&(_, t)| t).fold(f64::INFINITY, f64::min);
        if !best_t.is_finite() {
            return None;
        }
        let eps = view.coplanar_eps(origin, direction, best_t);
        let cand: Vec<FaceId> = hits.iter().filter(|&&(_, t)| t <= best_t + eps).map(|&(f, _)| f).collect();
        if cand.len() == 1 {
            return Some(Pick { face: cand[0], closed_group: false });
        }
        // The smallest of the overlapping faces (`np.argmin`: the first).
        let mut pick = cand[0];
        let mut area = mesh.face_area(pick, h);
        for &f in &cand[1..] {
            let a = mesh.face_area(f, h);
            if a < area {
                area = a;
                pick = f;
            }
        }
        Some(Pick { face: pick, closed_group: false })
    }
}
