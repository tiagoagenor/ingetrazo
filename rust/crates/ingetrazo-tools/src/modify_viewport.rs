// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! A headless [`Viewport`](crate::modify_base::Viewport): the camera and the
//! picks of `ingetrazo-snap` (held to the Python viewport by its own parity
//! tests), and a log of what a tool asked the viewport to show.
//!
//! The picks are made against the scene as it is when the tool asks - a live
//! preview (a drag that has moved the geometry) included - the way the
//! Python viewport re-reads its pick index when the scene version moves.

use std::cell::RefCell;

use ingetrazo_math::{Hypot, M4, V3};
use ingetrazo_model::mesh::{EdgeId, FaceId};
use ingetrazo_model::scene::{MeshId, Scene};
use ingetrazo_ops::command::Cx;
use ingetrazo_ops::group::Path;
use ingetrazo_snap::picking::{EntRef, FaceMode, PickIndex, Picker, State, TextMetrics, View};
use ingetrazo_snap::world::World;
use ingetrazo_snap::Axis;

use crate::modify_base::{EdgeAny, FaceHit, FaceRef, Gfx, Sel, Viewport};

/// Text metrics for a scene with no text labels: asking one is a bug.
pub struct NoText;

impl TextMetrics for NoText {
    fn height(&self) -> i32 {
        panic!("a text label was picked but no font metrics were given")
    }
    fn ascent(&self) -> i32 {
        panic!("a text label was picked but no font metrics were given")
    }
    fn descent(&self) -> i32 {
        panic!("a text label was picked but no font metrics were given")
    }
    fn advance(&self, _line: &str) -> i32 {
        panic!("a text label was picked but no font metrics were given")
    }
}

pub struct PickViewport {
    pub view: View,
    pub hypot: Hypot,
    /// `pick_threshold_px`.
    pub threshold: f64,
    pub metrics: Box<dyn TextMetrics>,
    /// `flash_status(text, msec)`, in order.
    pub flashes: Vec<(String, u32)>,
    /// `set_hover` and the group-preview hooks, in order.
    pub gfx: Vec<Gfx>,
    /// `ctrl_tapped()`: set by the host before it calls `on_key_release`.
    pub ctrl_alone: bool,
    /// Keep the pick index between calls (the window's viewport: building it
    /// is the cost of a pick), valid while [`PickViewport::hint`] and the
    /// meshes' serials stand. Off for the parity replays, which change the
    /// scene behind the viewport's back.
    pub use_cache: bool,
    /// What the host bumps when the model changes (`scene.version`).
    pub hint: u64,
    cache: RefCell<Option<(u64, u64, PickIndex)>>,
    /// The prototype chunks, kept across the rebuilds of the index.
    chunks: RefCell<ingetrazo_snap::picking::ChunkCache>,
}

impl PickViewport {
    pub fn new(view: View, hypot: Hypot) -> PickViewport {
        PickViewport {
            view,
            hypot,
            threshold: 8.0,
            metrics: Box::new(NoText),
            flashes: Vec::new(),
            gfx: Vec::new(),
            ctrl_alone: false,
            use_cache: false,
            hint: 0,
            cache: RefCell::new(None),
            chunks: RefCell::new(Default::default()),
        }
    }

    /// Run `f` on a picker over `scene` (its index kept when
    /// [`PickViewport::use_cache`]).
    pub fn with_picker<R>(&self, scene: &Scene, cx: &Cx, f: impl FnOnce(&Picker) -> R) -> R {
        let st = State::new(scene, cx.mesh, MeshId(0), None, false, FaceMode::of_scene(scene), self.hypot);
        let stamp = scene.meshes.iter().fold(scene.meshes.len() as u64, |a, m| a.wrapping_mul(1_000_003).wrapping_add(m.serial)) ^ (u64::from(cx.mesh.0) << 56);
        let idx = if self.use_cache {
            match self.cache.borrow_mut().take() {
                Some((hint, s, idx)) if hint == self.hint && s == stamp => idx,
                _ => PickIndex::build_cached(&st, &mut self.chunks.borrow_mut()),
            }
        } else {
            PickIndex::build(&st)
        };
        let p = Picker { st, idx, view: self.view.clone(), threshold: self.threshold };
        let r = f(&p);
        if self.use_cache {
            *self.cache.borrow_mut() = Some((self.hint, stamp, p.idx));
        }
        r
    }
}

impl Viewport for PickViewport {
    fn world_to_pixel(&self, p: V3) -> Option<(f64, f64)> {
        self.view.world_to_pixel(p)
    }

    fn pixel_to_ray(&self, x: f64, y: f64) -> Option<(V3, V3)> {
        self.view.pixel_to_ray(x, y)
    }

    fn eye(&self) -> V3 {
        self.view.eye
    }

    fn pick_threshold_px(&self) -> f64 {
        self.threshold
    }

    fn pick_face(&self, scene: &Scene, cx: &Cx, x: f64, y: f64) -> Option<FaceId> {
        let i = self.with_picker(scene, cx, |p| p.pick_face(x, y))?;
        Some(cx.mesh(scene).faces[i])
    }

    fn pick_face_any(&self, scene: &Scene, cx: &Cx, x: f64, y: f64) -> Option<FaceHit> {
        self.with_picker(scene, cx, |p| {
            let hit = p.pick_face_any(x, y)?;
            let face = match hit.face {
                EntRef::Loose(i) => FaceRef::Loose(cx.mesh(scene).faces[i]),
                EntRef::Placed { place, face } => FaceRef::Placed { placement: p.idx.places[place].path.clone(), index: face },
            };
            Some(FaceHit { face, owner: hit.owner, placement: hit.placement })
        })
    }

    fn pick_edge(&self, scene: &Scene, cx: &Cx, x: f64, y: f64) -> Option<EdgeId> {
        let i = self.with_picker(scene, cx, |p| p.pick_edge(x, y))?;
        Some(cx.mesh(scene).edges[i])
    }

    fn pick_visible_edge(&self, scene: &Scene, cx: &Cx, x: f64, y: f64) -> Option<EdgeId> {
        let i = self.with_picker(scene, cx, |p| p.pick_visible_edge(x, y))?;
        Some(cx.mesh(scene).edges[i])
    }

    fn pick_edge_any(&self, scene: &Scene, _cx: &Cx, x: f64, y: f64) -> Option<EdgeAny> {
        let world = World::build(scene, self.hypot);
        let view = ingetrazo_snap::View::new(self.snap_camera(), self.view.w, self.view.h, self.hypot);
        let hit = world.pick_edge_any(&view, self.threshold, x, y)?;
        Some(match hit {
            ingetrazo_snap::hover::HoverEdge::Loose(i) => EdgeAny::Loose(world.loose_edge_ids[i]),
            ingetrazo_snap::hover::HoverEdge::Group(i) => EdgeAny::Group { a: world.gedges[i].a, b: world.gedges[i].b },
        })
    }

    fn pick_group(&self, scene: &Scene, cx: &Cx, x: f64, y: f64) -> Option<Path> {
        self.with_picker(scene, cx, |p| p.pick_group(x, y))
    }

    fn pick_guide(&self, scene: &Scene, cx: &Cx, x: f64, y: f64) -> Option<usize> {
        self.with_picker(scene, cx, |p| p.pick_guide(x, y))
    }

    fn pick_dimension(&self, scene: &Scene, cx: &Cx, x: f64, y: f64) -> Option<usize> {
        self.with_picker(scene, cx, |p| p.pick_dimension(x, y))
    }

    fn pick_text_label(&self, scene: &Scene, cx: &Cx, x: f64, y: f64, rect_only: bool) -> Option<usize> {
        self.with_picker(scene, cx, |p| p.pick_text_label(x, y, rect_only, self.metrics.as_ref()))
    }

    fn pick_geopath(&self, scene: &Scene, cx: &Cx, x: f64, y: f64) -> Option<usize> {
        self.with_picker(scene, cx, |p| p.pick_geopath(x, y))
    }

    fn pick_section_plane(&self, scene: &Scene, cx: &Cx, x: f64, y: f64) -> Option<usize> {
        self.with_picker(scene, cx, |p| p.pick_section_plane(x, y))
    }

    fn pick_image_plane(&self, scene: &Scene, cx: &Cx, x: f64, y: f64) -> Option<usize> {
        self.with_picker(scene, cx, |p| p.pick_image_plane(x, y))
    }

    fn pick_axis(&self, scene: &Scene, cx: &Cx, x: f64, y: f64) -> Option<Axis> {
        self.with_picker(scene, cx, |p| p.pick_axis(x, y)).and_then(Axis::from_name)
    }

    fn flash_status(&mut self, text: &str, msec: u32) {
        self.flashes.push((text.to_string(), msec));
    }

    fn set_hover(&mut self, entity: Option<Sel>, placement: Option<Path>) {
        self.gfx.push(Gfx::Hover(entity, placement));
    }

    fn begin_groups_preview(&mut self, groups: &[Path], external: bool) {
        self.gfx.push(Gfx::Begin(groups.to_vec(), external));
    }

    fn set_groups_preview_offset(&mut self, delta: V3) {
        self.gfx.push(Gfx::Offset(delta));
    }

    fn set_groups_preview_matrix(&mut self, m: &M4) {
        self.gfx.push(Gfx::Matrix(*m));
    }

    fn end_groups_preview(&mut self) {
        self.gfx.push(Gfx::End);
    }

    fn ctrl_tapped(&self) -> bool {
        self.ctrl_alone
    }
}

impl PickViewport {
    /// The orbit camera of the snap engine's `View` (the same pose).
    fn snap_camera(&self) -> ingetrazo_snap::OrbitCamera {
        let o = self.view.cam.orbit;
        ingetrazo_snap::OrbitCamera {
            target: o.target,
            distance: o.distance,
            yaw: o.yaw,
            pitch: o.pitch,
            up: o.up,
            fov_deg: self.view.cam.fov_deg,
            aspect: self.view.cam.aspect,
            znear: self.view.cam.znear,
            zfar: self.view.cam.zfar,
            perspective: self.view.cam.perspective,
        }
    }
}
