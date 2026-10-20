// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! What the cursor is over, as the inference engine acquires it
//! (`InferenceEngine.acquire`): the edge, the corner, the face and the
//! circle centre under the pointer, and the reference state that keeps them
//! (`encourage_point`, `clear_references`, `reset`).
//!
//! The picks are `Viewport.pick_edge`, `pick_edge_any`, `pick_vertex` and
//! `pick_face_placement` over the same projections the snap scene uses.
//! Candidates that tie in distance keep their index order (NumPy's
//! `argsort` is not stable, but a tie between two different positions is
//! not one a hand can make).
//!
//! The circle centre is taken from a hovered loose curve edge only; the
//! centres of a group's curve and of the curves on a face's boundary
//! (`curve_centers_of_face`) are not ported.

use ingetrazo_math::V3;

use crate::compute::closest_on_segment_2d;
use crate::engine::{project_px, segment_distance, InferenceState, Settings};
use crate::types::EdgeRef;
use crate::workplane::ToolState;
use crate::world::World;
use crate::View;

/// The edge under the cursor.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum HoverEdge {
    /// A loose edge (index into `world.loose_edges`).
    Loose(usize),
    /// A group's hard edge (index into `world.gedges`), as a world
    /// pseudo-edge.
    Group(usize),
}

impl HoverEdge {
    pub fn edge_ref(self, world: &World) -> EdgeRef {
        match self {
            HoverEdge::Loose(i) => EdgeRef { a: world.loose_edges[i].a, b: world.loose_edges[i].b },
            HoverEdge::Group(i) => EdgeRef { a: world.gedges[i].a, b: world.gedges[i].b },
        }
    }
}

fn f64v(p: V3) -> [f64; 3] {
    [p.0 as f64, p.1 as f64, p.2 as f64]
}

/// `Viewport.edge_point_under_cursor`.
fn edge_point_under_cursor(view: &View, a: V3, b: V3, x: f64, y: f64) -> Option<V3> {
    let pa = view.world_to_pixel(a)?;
    let pb = view.world_to_pixel(b)?;
    let (dx, dy) = (pb.0 - pa.0, pb.1 - pa.1);
    let l2 = dx * dx + dy * dy;
    if l2 < 1e-12 {
        return Some(a);
    }
    let t = ((x - pa.0) * dx + (y - pa.1) * dy) / l2;
    let t = ingetrazo_math::pyfloat::py_max2(0.0, ingetrazo_math::pyfloat::py_min2(1.0, t));
    Some(a.add(b.sub(a).mul(t as f32)))
}

impl World {
    /// `Viewport._pick_edge`: the loose edge nearest the cursor within the
    /// pick threshold; with `visible_only` the nearest of the first eight
    /// that is not hidden behind the model (soft edges do not count unless
    /// hidden geometry is shown).
    pub fn pick_edge(&self, view: &View, pick_px: f64, x: f64, y: f64, visible_only: bool) -> Option<usize> {
        if self.loose_edges.is_empty() {
            return None;
        }
        let pts: Vec<[f64; 3]> = self.loose_edges.iter().flat_map(|e| [f64v(e.a), f64v(e.b)]).collect();
        let proj = project_px(view, &pts);
        let mut d: Vec<f64> = Vec::with_capacity(self.loose_edges.len());
        let mut any = false;
        for (i, e) in self.loose_edges.iter().enumerate() {
            let sel = !e.hidden || self.show_hidden_geometry;
            let mut ok = proj[2 * i].2 && proj[2 * i + 1].2 && sel;
            if visible_only && !self.show_hidden_geometry {
                ok &= !e.soft;
            }
            any |= ok;
            d.push(if ok { segment_distance(proj[2 * i], proj[2 * i + 1], x, y) } else { f64::INFINITY });
        }
        if !any {
            return None;
        }
        if !visible_only {
            let mut i = 0;
            for k in 1..d.len() {
                if d[k] < d[i] {
                    i = k;
                }
            }
            return if d[i] < pick_px { Some(i) } else { None };
        }
        let mut order: Vec<usize> = (0..d.len()).collect();
        order.sort_by(|&a, &b| d[a].partial_cmp(&d[b]).unwrap_or(std::cmp::Ordering::Equal));
        for &i in order.iter().take(8) {
            if !(d[i] < pick_px) {
                return None;
            }
            let e = &self.loose_edges[i];
            let p = edge_point_under_cursor(view, e.a, e.b, x, y);
            if p.is_none() || !self.is_occluded(view.cam.eye(), p.unwrap(), view.h) {
                return Some(i);
            }
        }
        None
    }

    /// `Viewport.pick_edge_any`: the loose or group edge nearest the cursor.
    pub fn pick_edge_any(&self, view: &View, pick_px: f64, x: f64, y: f64) -> Option<HoverEdge> {
        let loose = self.pick_edge(view, pick_px, x, y, false);
        if self.gedges.is_empty() {
            return loose.map(HoverEdge::Loose);
        }
        let pts: Vec<[f64; 3]> = self.gedges.iter().flat_map(|g| [f64v(g.a), f64v(g.b)]).collect();
        let proj = project_px(view, &pts);
        if !proj.iter().any(|p| p.2) {
            return loose.map(HoverEdge::Loose);
        }
        let d: Vec<f64> = (0..self.gedges.len()).map(|i| segment_distance(proj[2 * i], proj[2 * i + 1], x, y)).collect();
        let mut i = 0;
        for k in 1..d.len() {
            if d[k] < d[i] {
                i = k;
            }
        }
        if d[i] >= pick_px {
            return loose.map(HoverEdge::Loose);
        }
        if let Some(li) = loose {
            let e = &self.loose_edges[li];
            if let (Some(pa), Some(pb)) = (view.world_to_pixel(e.a), view.world_to_pixel(e.b)) {
                let (dl, _t) = closest_on_segment_2d((x, y), pa, pb);
                if dl <= d[i] {
                    return Some(HoverEdge::Loose(li));
                }
            }
        }
        Some(HoverEdge::Group(i))
    }

    /// `Viewport.pick_vertex`: the corner nearest the cursor within the pick
    /// threshold that is not hidden behind the model.
    pub fn pick_vertex(&self, view: &View, pick_px: f64, x: f64, y: f64) -> Option<V3> {
        // [loose a's, loose b's, group a's, group b's]
        let mut cands: Vec<(f64, V3)> = Vec::new();
        let mut parts: Vec<(Vec<(f64, f64, bool)>, Vec<V3>)> = Vec::new();
        if !self.loose_edges.is_empty() {
            let pts: Vec<[f64; 3]> = self.loose_edges.iter().flat_map(|e| [f64v(e.a), f64v(e.b)]).collect();
            let proj = project_px(view, &pts);
            let a: Vec<(f64, f64, bool)> = (0..self.loose_edges.len()).map(|i| (proj[2 * i].0, proj[2 * i].1, proj[2 * i].2 && proj[2 * i + 1].2)).collect();
            let b: Vec<(f64, f64, bool)> = (0..self.loose_edges.len()).map(|i| (proj[2 * i + 1].0, proj[2 * i + 1].1, proj[2 * i].2 && proj[2 * i + 1].2)).collect();
            parts.push((a, self.loose_edges.iter().map(|e| e.a).collect()));
            parts.push((b, self.loose_edges.iter().map(|e| e.b).collect()));
        }
        if !self.gedges.is_empty() {
            let pts: Vec<[f64; 3]> = self.gedges.iter().flat_map(|g| [f64v(g.a), f64v(g.b)]).collect();
            let proj = project_px(view, &pts);
            let a: Vec<(f64, f64, bool)> = (0..self.gedges.len()).map(|i| (proj[2 * i].0, proj[2 * i].1, proj[2 * i].2 && proj[2 * i + 1].2)).collect();
            let b: Vec<(f64, f64, bool)> = (0..self.gedges.len()).map(|i| (proj[2 * i + 1].0, proj[2 * i + 1].1, proj[2 * i].2 && proj[2 * i + 1].2)).collect();
            parts.push((a, self.gedges.iter().map(|g| g.a).collect()));
            parts.push((b, self.gedges.iter().map(|g| g.b).collect()));
        }
        if parts.is_empty() {
            return None;
        }
        for (screen, verts) in &parts {
            for (s, v) in screen.iter().zip(verts) {
                let d = if s.2 { (s.0 - x).hypot(s.1 - y) } else { f64::INFINITY };
                if d < pick_px {
                    cands.push((d, *v));
                }
            }
        }
        cands.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        cands.into_iter().map(|c| c.1).find(|&v| !self.is_occluded(view.cam.eye(), v, view.h))
    }
}

impl World {
    /// `Viewport._center_of_group_edge`: the rim of a circle inside a
    /// top-level group, matched to the group's own edge through its matrix.
    fn center_of_group_edge(&self, i: usize, h: ingetrazo_math::Hypot) -> Option<(V3, f64)> {
        let g = &self.gedges[i];
        let place = &self.places[g.place];
        let mesh = &self.meshes[place.owner_mesh.0 as usize];
        let xf = place.owner_xform;
        if mesh.edges.is_empty() {
            return None;
        }
        let (la, lb) = match &xf {
            Some(m) => {
                let (inv, ok) = m.inverted();
                if !ok {
                    return None;
                }
                (inv.map(g.a), inv.map(g.b))
            }
            None => (g.a, g.b),
        };
        let edge = match (mesh.vertex_at(la), mesh.vertex_at(lb)) {
            (Some(va), Some(vb)) => mesh.find_edge(va, vb),
            _ => None,
        }?;
        mesh.edge(edge).curve?;
        let (c, r) = crate::circle::center_of_edge(mesh, edge)?;
        let _ = h;
        Some((match &xf {
            Some(m) => m.map(c),
            None => c,
        }, r))
    }

    /// `Viewport._center_of_face`: the centre of the circle or arc on the
    /// face's boundary (the one nearest the cursor when there are several).
    fn center_of_face(&self, view: &View, ent: usize, x: f64, y: f64, h: ingetrazo_math::Hypot) -> Option<(V3, f64)> {
        use crate::world::FaceRef;
        let (mesh_id, face, xf) = match self.ents[ent].face {
            FaceRef::Loose(f) => (ingetrazo_model::scene::MeshId(self.loose_id), f, None),
            FaceRef::Placed { mesh, face, place } => (mesh, face, self.places[place].xform),
        };
        let centers = crate::circle::curve_centers_of_face(&self.meshes[mesh_id.0 as usize], face, xf.as_ref(), h);
        if centers.is_empty() {
            return None;
        }
        if centers.len() == 1 {
            return Some((centers[0].0, centers[0].1));
        }
        let mut best: Option<(f64, V3, f64)> = None;
        for (c, r, _key) in &centers {
            let Some(pc) = view.world_to_pixel(*c) else { continue };
            let d = ingetrazo_geom::bevel_profiles::py_hypot2(pc.0 - x, pc.1 - y);
            if best.is_none() || d < best.unwrap().0 {
                best = Some((d, *c, *r));
            }
        }
        best.map(|b| (b.1, b.2))
    }

    /// `Viewport._update_center_ref`: the centre of the circle or arc under
    /// the cursor — the hovered edge's curve, else a curve on the face under
    /// it — as `(centre, radius)`.
    pub fn update_center_ref(&self, view: &View, hov: Option<HoverEdge>, x: f64, y: f64, h: ingetrazo_math::Hypot) -> Option<(V3, f32)> {
        let mut found: Option<(V3, f64)> = None;
        match hov {
            Some(HoverEdge::Loose(i)) => {
                if let Some(cid) = self.loose_edges[i].curve {
                    found = self.curve_centers.get(&cid).copied().flatten();
                }
            }
            Some(HoverEdge::Group(i)) => found = self.center_of_group_edge(i, h),
            None => {}
        }
        if found.is_none() {
            if let Some(ent) = self.face_under(view, x, y) {
                found = self.center_of_face(view, ent, x, y, h);
            }
        }
        found.map(|(c, r)| (c, r as f32))
    }
}

impl InferenceState {
    /// `encourage_point`: make `point` the newest encouraged point (two are
    /// kept, the older one drops), and the 'from point' reference.
    pub fn encourage_point(&mut self, point: V3, h: ingetrazo_math::Hypot) {
        self.encouraged.retain(|p| (p.sub(point).length(h) as f64) > 1e-6);
        self.encouraged.push(point);
        let n = self.encouraged.len();
        if n > 2 {
            self.encouraged.drain(..n - 2);
        }
        self.acquired_point = Some(point);
    }

    /// `clear_references`: drop cursor-acquired and encouraged references.
    pub fn clear_references(&mut self) {
        self.hover_edge = None;
        self.center_ref = None;
        self.acquired_edge = None;
        self.acquired_face_normal = None;
        self.acquired_point = None;
        self.encouraged.clear();
    }

    /// `reset`: forget every inference state tied to the document.
    pub fn reset(&mut self) {
        self.clear_references();
        self.axis_lock = None;
        self.reference_edge = None;
        self.reference_mode = None;
        self.shift_lock = None;
    }

    /// `acquire(px, py)`: refresh the hover state `snap` reads — the edge
    /// under the cursor, the circle centre it may encourage, and the soft
    /// references held while a segment is being drawn.
    #[allow(clippy::too_many_arguments)]
    pub fn acquire(&mut self, view: &View, world: &World, tool: &ToolState, settings: &Settings, x: f64, y: f64) {
        let pick = settings.pick_threshold_px;
        let hov = if tool.uses_snap || tool.hover_group_edges {
            world.pick_edge_any(view, pick, x, y)
        } else {
            world.pick_edge(view, pick, x, y, false).map(HoverEdge::Loose)
        };
        self.hover_edge = hov.map(|e| e.edge_ref(world));
        if tool.uses_snap {
            // Only the tools that snap can use a centre. Nothing under the
            // cursor keeps the last one (`_valid_center_ref`).
            self.hover_center = world.update_center_ref(view, hov, x, y, view.h);
            if let Some(c) = self.hover_center {
                self.center_ref = Some(c);
            }
        }
        let drawing = tool.start_point.is_some();
        if drawing && tool.uses_snap {
            if let Some(corner) = world.pick_vertex(view, pick, x, y) {
                self.acquired_point = Some(corner);
            }
        }
        if !drawing {
            self.acquired_edge = None;
            self.acquired_face_normal = None;
        } else {
            if let Some(he) = self.hover_edge {
                self.acquired_edge = Some(he);
            }
            if let Some(e) = world.face_under(view, x, y) {
                self.acquired_face_normal = Some(world.ents[e].plane_place.1);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ingetrazo_math::Hypot;

    #[test]
    fn two_encouraged_points_are_kept_and_the_newest_is_the_reference() {
        let mut s = InferenceState::default();
        let h = Hypot::Msvc;
        s.encourage_point(V3(1.0, 0.0, 0.0), h);
        s.encourage_point(V3(2.0, 0.0, 0.0), h);
        s.encourage_point(V3(3.0, 0.0, 0.0), h);
        assert_eq!(s.encouraged, vec![V3(2.0, 0.0, 0.0), V3(3.0, 0.0, 0.0)]);
        assert_eq!(s.acquired_point, Some(V3(3.0, 0.0, 0.0)));
        // Encouraging a point already held moves it to the end.
        s.encourage_point(V3(2.0, 0.0, 0.0), h);
        assert_eq!(s.encouraged, vec![V3(3.0, 0.0, 0.0), V3(2.0, 0.0, 0.0)]);
    }

    #[test]
    fn clearing_forgets_the_references_but_reset_forgets_the_locks_too() {
        let mut s = InferenceState { axis_lock: Some(crate::Axis::X), ..Default::default() };
        s.encourage_point(V3(1.0, 0.0, 0.0), Hypot::Msvc);
        s.clear_references();
        assert!(s.encouraged.is_empty() && s.acquired_point.is_none());
        assert_eq!(s.axis_lock, Some(crate::Axis::X));
        s.reset();
        assert_eq!(s.axis_lock, None);
    }
}
