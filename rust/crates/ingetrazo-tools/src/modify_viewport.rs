// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! A headless [`Viewport`](crate::modify_base::Viewport): the camera and the
//! picks of `ingetrazo-snap` (held to the Python viewport by its own parity
//! tests), and a log of what a tool asked the viewport to show.
//!
//! The picks are made against the scene as it is when the tool asks - a live
//! preview (a drag that has moved the geometry) included - the way the
//! Python viewport re-reads its pick index when the scene version moves.

use ingetrazo_math::{Hypot, M4, V3};
use ingetrazo_model::mesh::{EdgeId, FaceId};
use ingetrazo_model::scene::{MeshId, Scene};
use ingetrazo_ops::command::Cx;
use ingetrazo_ops::group::Path;
use ingetrazo_snap::picking::{EntRef, FaceMode, Picker, State, TextMetrics, View};
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
}

impl PickViewport {
    pub fn new(view: View, hypot: Hypot) -> PickViewport {
        PickViewport { view, hypot, threshold: 8.0, metrics: Box::new(NoText), flashes: Vec::new(), gfx: Vec::new(), ctrl_alone: false }
    }

    fn picker<'a>(&self, scene: &'a Scene, cx: &Cx) -> Picker<'a> {
        let st = State::new(scene, cx.mesh, MeshId(0), None, false, FaceMode::of_scene(scene), self.hypot);
        let mut p = Picker::new(st, self.view.clone());
        p.threshold = self.threshold;
        p
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
        let i = self.picker(scene, cx).pick_face(x, y)?;
        Some(cx.mesh(scene).faces[i])
    }

    fn pick_face_any(&self, scene: &Scene, cx: &Cx, x: f64, y: f64) -> Option<FaceHit> {
        let p = self.picker(scene, cx);
        let hit = p.pick_face_any(x, y)?;
        let face = match hit.face {
            EntRef::Loose(i) => FaceRef::Loose(cx.mesh(scene).faces[i]),
            EntRef::Placed { place, face } => FaceRef::Placed { placement: p.idx.places[place].path.clone(), index: face },
        };
        Some(FaceHit { face, owner: hit.owner, placement: hit.placement })
    }

    fn pick_edge(&self, scene: &Scene, cx: &Cx, x: f64, y: f64) -> Option<EdgeId> {
        let i = self.picker(scene, cx).pick_edge(x, y)?;
        Some(cx.mesh(scene).edges[i])
    }

    fn pick_visible_edge(&self, scene: &Scene, cx: &Cx, x: f64, y: f64) -> Option<EdgeId> {
        let i = self.picker(scene, cx).pick_visible_edge(x, y)?;
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
        self.picker(scene, cx).pick_group(x, y)
    }

    fn pick_guide(&self, scene: &Scene, cx: &Cx, x: f64, y: f64) -> Option<usize> {
        self.picker(scene, cx).pick_guide(x, y)
    }

    fn pick_dimension(&self, scene: &Scene, cx: &Cx, x: f64, y: f64) -> Option<usize> {
        self.picker(scene, cx).pick_dimension(x, y)
    }

    fn pick_text_label(&self, scene: &Scene, cx: &Cx, x: f64, y: f64, rect_only: bool) -> Option<usize> {
        self.picker(scene, cx).pick_text_label(x, y, rect_only, self.metrics.as_ref())
    }

    fn pick_geopath(&self, scene: &Scene, cx: &Cx, x: f64, y: f64) -> Option<usize> {
        self.picker(scene, cx).pick_geopath(x, y)
    }

    fn pick_section_plane(&self, scene: &Scene, cx: &Cx, x: f64, y: f64) -> Option<usize> {
        self.picker(scene, cx).pick_section_plane(x, y)
    }

    fn pick_image_plane(&self, scene: &Scene, cx: &Cx, x: f64, y: f64) -> Option<usize> {
        self.picker(scene, cx).pick_image_plane(x, y)
    }

    fn pick_axis(&self, scene: &Scene, cx: &Cx, x: f64, y: f64) -> Option<Axis> {
        self.picker(scene, cx).pick_axis(x, y).and_then(Axis::from_name)
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
