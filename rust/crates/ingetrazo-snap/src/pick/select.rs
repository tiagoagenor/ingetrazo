// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! The Select tool's picking (`tools/select.py`): the priority between the
//! pick kinds for a click, and the rubber-band box in window and crossing
//! mode.

use std::collections::{HashMap, HashSet};

use ingetrazo_math::M4;
use ingetrazo_model::mesh::FaceId;
use ingetrazo_ops::group::{group_at, iter_placements};

use super::index::{base_chunk, instance_chunk, Chunk};
use super::numeric::P3;
use super::picks::TextMetrics;
use super::state::Path;
use super::Picker;

/// What a click or a box takes, by identity: loose faces and edges by their
/// index in the active mesh's lists, groups by path, annotations by their
/// index in the scene's lists.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Picked {
    Face(usize),
    Edge(usize),
    Group(Path),
    Dimension(usize),
    Label(usize),
    GeoPath(usize),
    Section(usize),
    Guide(usize),
    Image(usize),
}

type Rect = (f64, f64, f64, f64);

fn pt_in_rect(p: (f64, f64), r: Rect) -> bool {
    r.0 <= p.0 && p.0 <= r.2 && r.1 <= p.1 && p.1 <= r.3
}

/// `_seg_seg_2d`: whether two segments properly cross.
fn seg_seg_2d(p1: (f64, f64), p2: (f64, f64), p3_: (f64, f64), p4: (f64, f64)) -> bool {
    let ccw = |a: (f64, f64), b: (f64, f64), c: (f64, f64)| (c.1 - a.1) * (b.0 - a.0) - (b.1 - a.1) * (c.0 - a.0);
    let d1 = ccw(p3_, p4, p1);
    let d2 = ccw(p3_, p4, p2);
    let d3 = ccw(p1, p2, p3_);
    let d4 = ccw(p1, p2, p4);
    ((d1 > 0.0) != (d2 > 0.0)) && ((d3 > 0.0) != (d4 > 0.0))
}

/// `_seg_rect_overlap`: an endpoint inside, or an edge crossed.
fn seg_rect_overlap(a: (f64, f64), b: (f64, f64), r: Rect) -> bool {
    if pt_in_rect(a, r) || pt_in_rect(b, r) {
        return true;
    }
    let (x0, y0, x1, y1) = r;
    let c = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)];
    (0..4).any(|i| seg_seg_2d(a, b, c[i], c[(i + 1) % 4]))
}

/// `_seg_rect_mask` for one segment: window = both ends inside; crossing =
/// an end inside or the segment crossing one of the four borders.
fn seg_rect_mask(a: (f64, f64), b: (f64, f64), ok: bool, r: Rect, crossing: bool) -> bool {
    let (x0, y0, x1, y1) = r;
    let in_a = ok && a.0 >= x0 && a.0 <= x1 && a.1 >= y0 && a.1 <= y1;
    let in_b = ok && b.0 >= x0 && b.0 <= x1 && b.1 >= y0 && b.1 <= y1;
    if !crossing {
        return in_a && in_b;
    }
    if in_a || in_b {
        return true;
    }
    if !ok {
        return false;
    }
    for (px, py, qx, qy) in [(x0, y0, x1, y0), (x1, y0, x1, y1), (x1, y1, x0, y1), (x0, y1, x0, y0)] {
        let d1 = (qx - px) * (a.1 - py) - (qy - py) * (a.0 - px);
        let d2 = (qx - px) * (b.1 - py) - (qy - py) * (b.0 - px);
        let d3 = (b.0 - a.0) * (py - a.1) - (b.1 - a.1) * (px - a.0);
        let d4 = (b.0 - a.0) * (qy - a.1) - (b.1 - a.1) * (qx - a.0);
        if d1 * d2 < 0.0 && d3 * d4 < 0.0 {
            return true;
        }
    }
    false
}

impl Picker<'_> {
    /// `SelectTool._pick`: a group takes priority; then the edge under the
    /// cursor, a dimension, a text label, a georef path, a section plane, a
    /// guide, the front face, and last a reference image.
    pub fn select_pick(&self, x: f64, y: f64, fm: &dyn TextMetrics) -> Option<Picked> {
        if let Some(i) = self.pick_text_label(x, y, true, fm) {
            return Some(Picked::Label(i));
        }
        let group = self.pick_group(x, y);
        let edge = self.pick_visible_edge(x, y);
        if let (Some(_), Some(e)) = (&group, edge) {
            let a = super::numeric::v3(self.idx.edge_a[e]);
            let b = super::numeric::v3(self.idx.edge_b[e]);
            if let Some(p) = self.edge_point_under_cursor(a, b, x, y) {
                if !self.is_occluded(p) {
                    return Some(Picked::Edge(e));
                }
            }
        }
        if let Some(g) = group {
            return Some(Picked::Group(g));
        }
        if let Some(e) = edge {
            return Some(Picked::Edge(e));
        }
        if let Some(i) = self.pick_dimension(x, y) {
            return Some(Picked::Dimension(i));
        }
        if let Some(i) = self.pick_text_label(x, y, false, fm) {
            return Some(Picked::Label(i));
        }
        if let Some(i) = self.pick_geopath(x, y) {
            return Some(Picked::GeoPath(i));
        }
        if let Some(i) = self.pick_section_plane(x, y) {
            return Some(Picked::Section(i));
        }
        if let Some(i) = self.pick_guide(x, y) {
            return Some(Picked::Guide(i));
        }
        if let Some(f) = self.pick_face(x, y) {
            return Some(Picked::Face(f));
        }
        self.pick_image_plane(x, y).map(Picked::Image)
    }

    /// `SelectTool.on_box_select`: what the rubber band takes, in the order
    /// the tool hands it to the scene.
    pub fn box_select(&self, rect: Rect, crossing: bool) -> Vec<Picked> {
        let st = &self.st;
        let scene = st.scene;
        let mut picked = Vec::new();
        let view = &self.view;
        let idx = &self.idx;

        // The loose mesh: edges through the screen projection, faces from
        // their boundary edges.
        let mesh = scene.mesh(st.active);
        let n = idx.edge_a.len();
        if n > 0 {
            let (ax, ay, oka) = view.project_px(&idx.edge_a);
            let (bx, by, okb) = view.project_px(&idx.edge_b);
            let mask: Vec<bool> = (0..n).map(|i| seg_rect_mask((ax[i], ay[i]), (bx[i], by[i]), oka[i] && okb[i], rect, crossing)).collect();
            let face_ix: HashMap<FaceId, usize> = mesh.faces.iter().enumerate().map(|(i, &f)| (f, i)).collect();
            let face_sel = |f: FaceId| st.face_selectable(&mesh.face(f).attrs);
            for i in 0..n {
                if mask[i] {
                    let e = mesh.edge(mesh.edges[i]);
                    if st.layer_selectable(e.layer.as_deref()) && !e.hidden {
                        picked.push(Picked::Edge(i));
                    }
                }
            }
            if crossing {
                let mut seen = HashSet::new();
                for i in (0..n).filter(|&i| mask[i]) {
                    for &f in &mesh.edge(mesh.edges[i]).faces {
                        if seen.insert(f) && face_sel(f) {
                            picked.push(Picked::Face(face_ix[&f]));
                        }
                    }
                }
            } else {
                let mut face_edges: HashMap<FaceId, Vec<usize>> = HashMap::new();
                for (i, &eid) in mesh.edges.iter().enumerate() {
                    for &f in &mesh.edge(eid).faces {
                        face_edges.entry(f).or_default().push(i);
                    }
                }
                for (fi, &f) in mesh.faces.iter().enumerate() {
                    if let Some(es) = face_edges.get(&f) {
                        if !es.is_empty() && es.iter().all(|&e| mask[e]) && face_sel(f) {
                            picked.push(Picked::Face(fi));
                        }
                    }
                }
            }
        }

        // Groups and component instances.
        let ctx = st.ctx.as_deref();
        let cands: Vec<Path> = match ctx {
            None => (0..scene.groups.len()).map(|i| vec![i]).collect(),
            Some(c) => match group_at(&scene.groups, c) {
                Some(g) => (0..g.children.len()).map(|i| [c, &[i][..]].concat()).collect(),
                None => Vec::new(),
            },
        };
        for path in cands {
            let g = group_at(&scene.groups, &path).unwrap();
            if !self.group_object_selectable(&path) {
                continue;
            }
            let verdict = if g.billboard.as_ref().map(ingetrazo_model::pyval::truthy).unwrap_or(false) { None } else { self.box_group_fast(&path, rect, crossing) };
            let hit = match verdict {
                Some(v) => v,
                None => self.box_group_walk(&path, rect, crossing),
            };
            if hit {
                picked.push(Picked::Group(path));
            }
        }

        // Guides: an infinite line is never fully enclosed.
        for (i, g) in scene.guides.iter().enumerate() {
            if !st.layer_selectable(None) {
                continue;
            }
            if g.is_line() {
                if !crossing {
                    continue;
                }
                let (a, b) = g.segment();
                let Some((sa, sb)) = view.clip_segment_front(a, b) else { continue };
                if let (Some(pa), Some(pb)) = (view.world_to_pixel(sa), view.world_to_pixel(sb)) {
                    if seg_rect_overlap(pa, pb, rect) {
                        picked.push(Picked::Guide(i));
                    }
                }
            } else if let Some(p) = view.world_to_pixel(g.point) {
                if pt_in_rect(p, rect) {
                    picked.push(Picked::Guide(i));
                }
            }
        }
        for (i, dim) in scene.dimensions.iter().enumerate() {
            if !st.layer_selectable(dim.layer.as_deref()) {
                continue;
            }
            let (ap, bp) = dim.line_points(st.h);
            let (Some(pa), Some(pb)) = (view.world_to_pixel(ap), view.world_to_pixel(bp)) else { continue };
            if crossing {
                if seg_rect_overlap(pa, pb, rect) {
                    picked.push(Picked::Dimension(i));
                }
            } else if pt_in_rect(pa, rect) && pt_in_rect(pb, rect) {
                picked.push(Picked::Dimension(i));
            }
        }
        for (i, lab) in scene.text_labels.iter().enumerate() {
            if !st.layer_selectable(lab.layer.as_deref()) {
                continue;
            }
            let pa = view.world_to_pixel(lab.anchor);
            let Some(pp) = view.world_to_pixel(lab.position()) else { continue };
            if crossing {
                if pt_in_rect(pp, rect) || pa.is_some_and(|a| seg_rect_overlap(a, pp, rect)) {
                    picked.push(Picked::Label(i));
                }
            } else if pt_in_rect(pp, rect) && pa.is_none_or(|a| pt_in_rect(a, rect)) {
                picked.push(Picked::Label(i));
            }
        }
        picked
    }

    /// `entity_selectable(group)` of a real group of the tree.
    fn group_object_selectable(&self, path: &[usize]) -> bool {
        let g = group_at(&self.st.scene.groups, path).unwrap();
        if g.hidden && !self.st.show_hidden_objects() {
            return false;
        }
        if g.locked {
            return false;
        }
        self.st.layer_selectable(g.layer.as_deref())
    }

    /// The chunks of a group and of every placement below it
    /// (`_expand_placements(group)` then `_group_chunk`).
    fn group_chunks(&self, path: &[usize]) -> Vec<Chunk> {
        let scene = self.st.scene;
        let g = group_at(&scene.groups, path).unwrap();
        iter_placements(g, None)
            .into_iter()
            .map(|(pg, m)| {
                let base = base_chunk(scene.mesh(pg.mesh), self.st.h);
                match m {
                    None => base,
                    Some(m) => instance_chunk(&base, &m),
                }
            })
            .collect()
    }

    /// `_box_group_fast`: every corner of the chunks through the screen
    /// projection. `None` when the chunks hold nothing to project.
    fn box_group_fast(&self, path: &[usize], rect: Rect, crossing: bool) -> Option<bool> {
        let mut pts: Vec<P3> = Vec::new();
        let mut pairs: Vec<P3> = Vec::new();
        for c in self.group_chunks(path) {
            if !c.v0.is_empty() {
                pts.extend(c.v0.iter().cloned());
                pts.extend((0..c.v0.len()).map(|i| [c.v0[i][0] + c.e1[i][0], c.v0[i][1] + c.e1[i][1], c.v0[i][2] + c.e1[i][2]]));
                pts.extend((0..c.v0.len()).map(|i| [c.v0[i][0] + c.e2[i][0], c.v0[i][1] + c.e2[i][1], c.v0[i][2] + c.e2[i][2]]));
            }
            for e in &c.edges {
                pairs.push([e[0][0] as f64, e[0][1] as f64, e[0][2] as f64]);
                pairs.push([e[1][0] as f64, e[1][1] as f64, e[1][2] as f64]);
            }
        }
        let n_pairs = pairs.len();
        pts.extend(pairs);
        if pts.is_empty() {
            return None;
        }
        let (xs, ys, ok) = self.view.project_px(&pts);
        let (x0, y0, x1, y1) = rect;
        let inside: Vec<bool> = (0..pts.len()).map(|i| ok[i] && xs[i] >= x0 && xs[i] <= x1 && ys[i] >= y0 && ys[i] <= y1).collect();
        if crossing {
            if inside.iter().any(|&b| b) {
                return Some(true);
            }
            if n_pairs > 0 {
                let base = pts.len() - n_pairs;
                for k in (0..n_pairs).step_by(2) {
                    let (i, j) = (base + k, base + k + 1);
                    if seg_rect_mask((xs[i], ys[i]), (xs[j], ys[j]), ok[i] && ok[j], rect, true) {
                        return Some(true);
                    }
                }
            }
            return Some(false);
        }
        Some(ok.iter().all(|&b| b) && inside.iter().all(|&b| b))
    }

    /// The per-vertex walk `on_box_select` falls back to.
    fn box_group_walk(&self, path: &[usize], rect: Rect, crossing: bool) -> bool {
        let scene = self.st.scene;
        let g = group_at(&scene.groups, path).unwrap();
        let places: Vec<(ingetrazo_model::scene::MeshId, Option<M4>)> =
            iter_placements(g, None).into_iter().filter(|(pg, _)| !scene.mesh(pg.mesh).vertices.is_empty()).map(|(pg, m)| (pg.mesh, m)).collect();
        if places.is_empty() {
            return false;
        }
        let w2p = |mesh: &ingetrazo_model::mesh::Mesh, v, m: &Option<M4>| {
            let p = mesh.position(v);
            self.view.world_to_pixel(match m {
                Some(m) => m.map(p),
                None => p,
            })
        };
        if crossing {
            for (mid, m) in &places {
                let mesh = scene.mesh(*mid);
                for &v in &mesh.vertices {
                    if let Some(p) = w2p(mesh, v, m) {
                        if pt_in_rect(p, rect) {
                            return true;
                        }
                    }
                }
            }
            for (mid, m) in &places {
                let mesh = scene.mesh(*mid);
                for &e in &mesh.edges {
                    let ed = mesh.edge(e);
                    if let (Some(pa), Some(pb)) = (w2p(mesh, ed.v0, m), w2p(mesh, ed.v1, m)) {
                        if seg_rect_overlap(pa, pb, rect) {
                            return true;
                        }
                    }
                }
            }
            false
        } else {
            for (mid, m) in &places {
                let mesh = scene.mesh(*mid);
                for &v in &mesh.vertices {
                    match w2p(mesh, v, m) {
                        Some(p) if pt_in_rect(p, rect) => {}
                        _ => return false,
                    }
                }
            }
            true
        }
    }
}
