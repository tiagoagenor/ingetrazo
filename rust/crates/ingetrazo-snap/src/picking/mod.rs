// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! Picking: what the cursor picks (docs/rust-rewrite-plan.md, M3.12) —
//! face, edge, vertex, group or component and the placement it was hit
//! through, guide, dimension, text label, section plane, image plane — given
//! the scene, the camera and a pixel; and what a rubber-band box takes.
//!
//! This is `Viewport.pick_*`, the pick index they read (`_pick_index`) and
//! `tools/select.py`'s priority and box selection, operation for operation:
//! the tolerances in pixels, the tie-breaks (the smallest of the coplanar
//! faces, the nearest visible edge), the exclusion of hidden, locked and
//! layer-hidden things, the open group (what is outside it is in the index
//! for occlusion, and cannot be picked), x-ray and the active section cut.
//!
//! The arithmetic is Python's: float32 where Qt is float32, float64 where
//! NumPy or Python floats are, and NumPy's summation orders (see
//! [`numeric`]).
//!
//! Left out, and said so where it matters: face-me billboards, the lazy
//! placements of a model of over a million faces, the grid that cuts a big
//! chunk into cells, the experimental engine's index, and extension items.

pub mod camera;
pub mod edit;
pub mod index;
pub mod numeric;
pub mod picks;
pub mod ray;
pub mod select;
pub mod state;

pub use camera::{Camera, View};
pub use edit::enter_edit;
pub use index::{ChunkCache, EntRef, GScreenCache, PickIndex, Place};
pub use picks::{FaceHit, TextMetrics};
pub use select::Picked;
pub use state::{FaceMode, Path, State};

use ingetrazo_math::V3;
use ingetrazo_model::scene::Scene;
use ingetrazo_model::section::SectionPlane;
use ingetrazo_geom::facegeom::plane_axes;
use ingetrazo_ops::group::{affine_rows, iter_placements, np_affine};

use numeric::{p3, py_max, py_min, v3, P3};

/// `Viewport.pick_threshold_px`.
pub const PICK_THRESHOLD_PX: f64 = 8.0;

/// A scene on a camera: the index the picks read, and the picks.
pub struct Picker<'a> {
    pub st: State<'a>,
    pub idx: PickIndex,
    pub view: View,
    /// `pick_threshold_px`.
    pub threshold: f64,
}

impl<'a> Picker<'a> {
    pub fn new(st: State<'a>, view: View) -> Picker<'a> {
        let idx = PickIndex::build(&st);
        Picker { st, idx, view, threshold: PICK_THRESHOLD_PX }
    }

    /// The same scene seen from another camera.
    pub fn set_view(&mut self, view: View) {
        self.view = view;
    }

    /// `Scene.bounds()`: the box of everything visible, or `None`.
    pub fn scene_bounds(&self) -> Option<(V3, V3)> {
        let st = &self.st;
        let scene = st.scene;
        let mut lo = [f64::INFINITY; 3];
        let mut hi = [f64::NEG_INFINITY; 3];
        let mut seen = false;
        fn absorb(lo: &mut [f64; 3], hi: &mut [f64; 3], seen: &mut bool, p: V3) {
            *seen = true;
            let q = p3(p);
            for k in 0..3 {
                if q[k] < lo[k] {
                    lo[k] = q[k];
                }
                if q[k] > hi[k] {
                    hi[k] = q[k];
                }
            }
        }
        let root = scene.mesh(st.root_loose);
        for &eid in &root.edges {
            let e = root.edge(eid);
            if st.layer_visible(e.layer.as_deref()) {
                absorb(&mut lo, &mut hi, &mut seen, root.position(e.v0));
                absorb(&mut lo, &mut hi, &mut seen, root.position(e.v1));
            }
        }
        for &fid in &root.faces {
            let f = root.face(fid);
            if st.face_visible(&f.attrs) {
                for &v in &f.outer {
                    absorb(&mut lo, &mut hi, &mut seen, root.position(v));
                }
            }
        }
        let mut gl = lo;
        let mut gh = hi;
        for g in scene.groups.iter() {
            let bb = g.billboard.as_ref().map(ingetrazo_model::pyval::truthy).unwrap_or(false);
            if (g.hidden || !st.layer_visible(g.layer.as_deref())) || bb {
                continue;
            }
            for (pg, m) in iter_placements(g, None) {
                if !std::ptr::eq(pg, g) {
                    let pbb = pg.billboard.as_ref().map(ingetrazo_model::pyval::truthy).unwrap_or(false);
                    if pbb || pg.hidden || !st.layer_visible(pg.layer.as_deref()) {
                        continue;
                    }
                }
                let mesh = scene.mesh(pg.mesh);
                let mut arr: Vec<P3> = mesh.vertices.iter().map(|&v| p3(mesh.position(v))).collect();
                if arr.is_empty() {
                    continue;
                }
                if let Some(m) = m {
                    let (rot, trans) = np_affine(&m);
                    arr = affine_rows(&arr, &rot, &trans);
                }
                let mut l = [f64::INFINITY; 3];
                let mut u = [f64::NEG_INFINITY; 3];
                for p in &arr {
                    for k in 0..3 {
                        l[k] = l[k].min(p[k]);
                        u[k] = u[k].max(p[k]);
                    }
                }
                seen = true;
                for k in 0..3 {
                    gl[k] = py_min(gl[k], l[k]);
                    gh[k] = py_max(gh[k], u[k]);
                }
            }
        }
        if !seen {
            return None;
        }
        Some((v3(gl), v3(gh)))
    }

    /// `section_plane_frame`: the four world corners of the plane's drawn
    /// frame, the model's bounds projected onto the plane with a margin.
    pub fn section_plane_frame(&self, sp: &SectionPlane) -> [V3; 4] {
        let h = self.st.h;
        let (u, v) = plane_axes(sp.normal, h);
        let (cu, cv, half_u, half_v);
        match self.scene_bounds() {
            None => {
                half_u = 1.5;
                half_v = 1.5;
                cu = 0.0;
                cv = 0.0;
            }
            Some((lo, hi)) => {
                let mut us = Vec::new();
                let mut vs = Vec::new();
                for x in [lo.0, hi.0] {
                    for y in [lo.1, hi.1] {
                        for z in [lo.2, hi.2] {
                            let d = V3(x, y, z).sub(sp.point);
                            us.push(d.dot(u) as f64);
                            vs.push(d.dot(v) as f64);
                        }
                    }
                }
                let pmin = |a: &[f64]| a.iter().cloned().fold(a[0], py_min);
                let pmax = |a: &[f64]| a.iter().cloned().fold(a[0], py_max);
                cu = (pmin(&us) + pmax(&us)) * 0.5;
                cv = (pmin(&vs) + pmax(&vs)) * 0.5;
                half_u = (pmax(&us) - pmin(&us)) * 0.56 + 0.2;
                half_v = (pmax(&vs) - pmin(&vs)) * 0.56 + 0.2;
            }
        }
        let c = sp.point.add(u.mul(cu as f32)).add(v.mul(cv as f32));
        let (hu, hv) = (half_u as f32, half_v as f32);
        [
            c.sub(u.mul(hu)).sub(v.mul(hv)),
            c.add(u.mul(hu)).sub(v.mul(hv)),
            c.add(u.mul(hu)).add(v.mul(hv)),
            c.sub(u.mul(hu)).add(v.mul(hv)),
        ]
    }
}

/// Opens the scene for picking: the scene as it is, or after the groups
/// along `edit` have been opened (see [`enter_edit`]).
pub fn open_scene(scene: &Scene, edit: Option<&[usize]>, h: ingetrazo_math::Hypot) -> (Scene, ingetrazo_model::scene::MeshId) {
    match edit {
        None => (scene.clone(), ingetrazo_model::scene::MeshId(0)),
        Some(path) => enter_edit(scene, path, h),
    }
}
