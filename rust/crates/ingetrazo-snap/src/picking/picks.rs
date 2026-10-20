// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! What each `pick_*` of the viewport answers for a pixel.

use ingetrazo_math::V3;
use ingetrazo_ops::group::group_at;

use super::index::{group_selectable, EntRef};
use super::numeric::{dot3, np_hypot, p3, point_to_segment_2d, py_hypot2, py_max, py_min, v3, matvec3};
use super::ray::Hits;
use super::state::Path;
use super::Picker;

/// The face a pick found and what it was found through.
#[derive(Clone, Debug, PartialEq)]
pub struct FaceHit {
    pub face: EntRef,
    /// The group a click on it selects; `None` for a loose face.
    pub owner: Option<Path>,
    /// The placement the face was hit through; `None` for a loose face.
    pub placement: Option<Path>,
}

/// The text the labels are laid out in: the metrics of the font that draws
/// them (Qt's `QFontMetrics` in Python).
pub trait TextMetrics {
    fn height(&self) -> i32;
    fn ascent(&self) -> i32;
    fn descent(&self) -> i32;
    /// `horizontalAdvance(line)`.
    fn advance(&self, line: &str) -> i32;
}

/// numpy `argsort` of floats with ties in index order.
pub fn argsort(d: &[f64]) -> Vec<usize> {
    let mut ix: Vec<usize> = (0..d.len()).collect();
    ix.sort_by(|&a, &b| d[a].partial_cmp(&d[b]).unwrap_or(std::cmp::Ordering::Equal));
    ix
}

/// `str.splitlines()`.
pub fn splitlines(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut chars = text.chars().peekable();
    let mut pending = false;
    while let Some(c) = chars.next() {
        match c {
            '\r' => {
                if chars.peek() == Some(&'\n') {
                    chars.next();
                }
                out.push(std::mem::take(&mut cur));
                pending = false;
            }
            '\n' | '\x0b' | '\x0c' | '\x1c' | '\x1d' | '\x1e' | '\u{85}' | '\u{2028}' | '\u{2029}' => {
                out.push(std::mem::take(&mut cur));
                pending = false;
            }
            _ => {
                cur.push(c);
                pending = true;
            }
        }
    }
    if pending {
        out.push(cur);
    }
    out
}

impl Picker<'_> {
    fn ray(&self, x: f64, y: f64) -> Option<(V3, V3)> {
        self.view.pixel_to_ray(x, y)
    }

    /// `pick_face`: the front-most loose face under the cursor; of several
    /// at the same depth, the smallest.
    pub fn pick_face(&self, x: f64, y: f64) -> Option<usize> {
        let (origin, direction) = self.ray(x, y)?;
        let idx = &self.idx;
        if idx.ents.is_empty() {
            return None;
        }
        let mut face_t = self.hover_face_t(origin, direction)?;
        for (i, t) in face_t.iter_mut().enumerate() {
            if !idx.loose[i] {
                *t = f64::INFINITY;
            }
        }
        let best_t = face_t.iter().cloned().fold(f64::INFINITY, py_min);
        if !best_t.is_finite() {
            return None;
        }
        let eps = self.coplanar_eps(origin, direction, best_t);
        let chosen = self.choose(&face_t, best_t + eps);
        match idx.ents[chosen] {
            EntRef::Loose(f) => Some(f),
            EntRef::Placed { .. } => None,
        }
    }

    /// The entity among those within `limit` with the smallest area.
    fn choose(&self, face_t: &[f64], limit: f64) -> usize {
        let cand: Vec<usize> = (0..face_t.len()).filter(|&i| face_t[i] <= limit).collect();
        if cand.len() == 1 {
            return cand[0];
        }
        let mut best = cand[0];
        for &c in &cand[1..] {
            if self.idx.area[c] < self.idx.area[best] {
                best = c;
            }
        }
        best
    }

    /// `pick_face_any` and `pick_face_placement`: the front-most face
    /// across the loose mesh and every group.
    pub fn pick_face_any(&self, x: f64, y: f64) -> Option<FaceHit> {
        let (origin, direction) = self.ray(x, y)?;
        let idx = &self.idx;
        if idx.ents.is_empty() {
            return None;
        }
        let face_t = self.hover_face_t(origin, direction)?;
        let best_t = face_t.iter().cloned().fold(f64::INFINITY, py_min);
        if !best_t.is_finite() {
            return None;
        }
        let eps = self.coplanar_eps(origin, direction, best_t);
        let chosen = self.choose(&face_t, best_t + eps);
        let e = idx.ents[chosen];
        Some(match e {
            EntRef::Loose(_) => FaceHit { face: e, owner: None, placement: None },
            EntRef::Placed { place, .. } => {
                let p = &idx.places[place];
                FaceHit { face: e, owner: Some(p.owner.clone()), placement: Some(p.path.clone()) }
            }
        })
    }

    /// `pick_group`: the group whose geometry the cursor hits (front-most
    /// face, or the nearest visible edge for a group that is only lines).
    pub fn pick_group(&self, x: f64, y: f64) -> Option<Path> {
        if let Some(ctx) = &self.st.ctx {
            let open = group_at(&self.st.scene.groups, ctx);
            if open.is_none_or(|g| g.children.is_empty()) {
                return None; // inside a plain group: its content
            }
        }
        let idx = &self.idx;
        if let Some((origin, direction)) = self.ray(x, y) {
            let mut best: Option<(f64, Path)> = None;
            let mut loose_t = f64::INFINITY;
            if !idx.ents.is_empty() {
                if let Some((face_t, touched)) = self.hover_face_t_touched(origin, direction) {
                    // Only the entities the ray hit count: the nearest loose
                    // one, and the nearest of the placed ones (np.argmin:
                    // the first of the smallest).
                    let mut nearest: Option<usize> = None;
                    for &i in &touched {
                        let t = face_t[i];
                        if idx.loose[i] {
                            loose_t = py_min(loose_t, t);
                        } else if nearest.is_none_or(|b| t < face_t[b] || (t == face_t[b] && i < b)) {
                            nearest = Some(i);
                        }
                    }
                    if let Some(i) = nearest.filter(|&i| face_t[i].is_finite()) {
                        if let EntRef::Placed { place, .. } = idx.ents[i] {
                            best = Some((face_t[i], idx.places[place].owner.clone()));
                        }
                    }
                }
            }
            if let Some((t, g)) = best {
                let eps = self.coplanar_eps(origin, direction, t);
                if loose_t.is_finite() && loose_t < t - eps {
                    return None;
                }
                return Some(g);
            }
        }
        // The edge fallback: the cursor over empty space, or a group of
        // lines only.
        let mut best_g = None;
        let best_d = self.threshold;
        if !idx.gedge_a.is_empty() {
            let gs = idx.gscreen.get(idx, &self.view);
            let (ax, ay, bx, by, oka, okb) = (&gs.ax, &gs.ay, &gs.bx, &gs.by, &gs.oka, &gs.okb);
            let n = ax.len();
            if (0..n).any(|i| oka[i] && okb[i]) {
                // The edges within the threshold, with where along each the
                // cursor's nearest point is. An edge whose screen box is
                // farther than the threshold is out without the arithmetic.
                let lim = best_d + 1e-6;
                let mut near: Vec<usize> = Vec::new();
                let mut dn: Vec<f64> = Vec::new();
                let mut tt: Vec<f64> = Vec::new();
                gs.for_each_near(x, y, best_d, |i| {
                    if !(oka[i] && okb[i] && idx.gedge_sel[i]) {
                        return;
                    }
                    if ax[i].min(bx[i]) - lim > x || ax[i].max(bx[i]) + lim < x || ay[i].min(by[i]) - lim > y || ay[i].max(by[i]) + lim < y {
                        return;
                    }
                    let dx = bx[i] - ax[i];
                    let dy = by[i] - ay[i];
                    let l2 = dx * dx + dy * dy;
                    let safe = if l2 > 1e-12 { l2 } else { 1.0 };
                    let t = py_max_np(((x - ax[i]) * dx + (y - ay[i]) * dy) / safe);
                    let d = np_hypot(ax[i] + t * dx - x, ay[i] + t * dy - y);
                    if d < best_d {
                        near.push(i);
                        dn.push(d);
                        tt.push(t);
                    }
                });
                for k in argsort(&dn) {
                    let i = near[k];
                    let a = v3(idx.gedge_a[i]);
                    let b = v3(idx.gedge_b[i]);
                    let p = a.add(b.sub(a).mul(tt[k] as f32));
                    if !self.is_occluded(p) {
                        best_g = Some(idx.gedge_groups[idx.gedge_gi[i]].clone());
                        break;
                    }
                }
            }
        }
        best_g
    }

    /// `edge_point_under_cursor`: the world point of the edge nearest the
    /// cursor, or `None`.
    pub fn edge_point_under_cursor(&self, a: V3, b: V3, x: f64, y: f64) -> Option<V3> {
        let pa = self.view.world_to_pixel(a)?;
        let pb = self.view.world_to_pixel(b)?;
        let dx = pb.0 - pa.0;
        let dy = pb.1 - pa.1;
        let l2 = dx * dx + dy * dy;
        if l2 < 1e-12 {
            return Some(a);
        }
        let t = ((x - pa.0) * dx + (y - pa.1) * dy) / l2;
        let t = py_max(0.0, py_min(1.0, t));
        Some(a.add(b.sub(a).mul(t as f32)))
    }

    /// `pick_edge`.
    pub fn pick_edge(&self, x: f64, y: f64) -> Option<usize> {
        self.pick_edge_core(x, y, false)
    }

    /// `pick_visible_edge`: an edge the user cannot see never wins.
    pub fn pick_visible_edge(&self, x: f64, y: f64) -> Option<usize> {
        self.pick_edge_core(x, y, true)
    }

    fn pick_edge_core(&self, x: f64, y: f64, visible_only: bool) -> Option<usize> {
        let idx = &self.idx;
        if idx.edge_a.is_empty() {
            return None;
        }
        let n = idx.edge_a.len();
        let (ax, ay, oka) = self.view.project_px(&idx.edge_a);
        let (bx, by, okb) = self.view.project_px(&idx.edge_b);
        let mut ok: Vec<bool> = (0..n).map(|i| oka[i] && okb[i] && idx.edge_sel[i]).collect();
        if visible_only && !self.st.show_hidden_geometry() {
            for i in 0..n {
                if idx.edge_soft[i] {
                    ok[i] = false;
                }
            }
        }
        if let Some(sp) = self.st.active_cut() {
            let nrm = p3(sp.normal);
            let c = dot3(nrm, p3(sp.point));
            for i in 0..n {
                let da = matvec3(idx.edge_a[i], nrm, n) - c;
                let db = matvec3(idx.edge_b[i], nrm, n) - c;
                if da > 1e-6 && db > 1e-6 {
                    ok[i] = false;
                }
            }
        }
        if !ok.iter().any(|&b| b) {
            return None;
        }
        let mut d = vec![f64::INFINITY; n];
        for i in 0..n {
            if !ok[i] {
                continue;
            }
            let dx = bx[i] - ax[i];
            let dy = by[i] - ay[i];
            let l2 = dx * dx + dy * dy;
            let safe = if l2 > 1e-12 { l2 } else { 1.0 };
            let t = py_max_np(((x - ax[i]) * dx + (y - ay[i]) * dy) / safe);
            d[i] = np_hypot(ax[i] + t * dx - x, ay[i] + t * dy - y);
        }
        if !visible_only {
            let mut i = 0;
            for k in 1..n {
                if d[k] < d[i] {
                    i = k;
                }
            }
            return (d[i] < self.threshold).then_some(idx.edge_ix[i]);
        }
        for i in argsort(&d).into_iter().take(8) {
            if !(d[i] < self.threshold) {
                return None;
            }
            let (a, b) = (v3(idx.edge_a[i]), v3(idx.edge_b[i]));
            match self.edge_point_under_cursor(a, b, x, y) {
                Some(p) if self.is_occluded(p) => continue,
                _ => return Some(idx.edge_ix[i]),
            }
        }
        None
    }

    /// `pick_vertex`: the scene corner nearest the cursor within the
    /// threshold; the nearest one in view wins.
    pub fn pick_vertex(&self, x: f64, y: f64) -> Option<V3> {
        let idx = &self.idx;
        let (mut xs, mut ys, mut oks): (Vec<f64>, Vec<f64>, Vec<bool>) = (Vec::new(), Vec::new(), Vec::new());
        let n = idx.edge_a.len();
        if n > 0 {
            let (ax, ay, oka) = self.view.project_px(&idx.edge_a);
            let (bx, by, okb) = self.view.project_px(&idx.edge_b);
            let ok: Vec<bool> = (0..n).map(|i| oka[i] && okb[i]).collect();
            xs.extend(&ax);
            xs.extend(&bx);
            ys.extend(&ay);
            ys.extend(&by);
            oks.extend(&ok);
            oks.extend(&ok);
        }
        let m = idx.gedge_a.len();
        if m > 0 {
            let gs = idx.gscreen.get(idx, &self.view);
            let ok: Vec<bool> = (0..m).map(|i| gs.oka[i] && gs.okb[i]).collect();
            xs.extend(&gs.ax);
            xs.extend(&gs.bx);
            ys.extend(&gs.ay);
            ys.extend(&gs.by);
            oks.extend(&ok);
            oks.extend(&ok);
        }
        if xs.is_empty() {
            return None;
        }
        let d: Vec<f64> = (0..xs.len()).map(|i| if oks[i] { np_hypot(xs[i] - x, ys[i] - y) } else { f64::INFINITY }).collect();
        let cand: Vec<usize> = (0..d.len()).filter(|&i| d[i] < self.threshold).collect();
        let vertex_of = |i: usize| -> V3 {
            if i < 2 * n {
                let k = i % n;
                let e = if i < n { idx.edge_a[k] } else { idx.edge_b[k] };
                v3(e)
            } else {
                let j = i - 2 * n;
                let arr = if j < m { &idx.gedge_a } else { &idx.gedge_b };
                v3(arr[j % m])
            }
        };
        let dc: Vec<f64> = cand.iter().map(|&i| d[i]).collect();
        let order: Vec<usize> = argsort(&dc).into_iter().map(|k| cand[k]).collect();
        if order.is_empty() {
            return None;
        }
        let first = vertex_of(order[0]);
        if !self.is_occluded(first) {
            return Some(first);
        }
        let (mut k, mut size) = (1usize, 8usize);
        while k < order.len() {
            let batch: Vec<V3> = order[k..(k + size).min(order.len())].iter().map(|&i| vertex_of(i)).collect();
            let hidden = self.occluded_many(&batch);
            for (v, h) in batch.iter().zip(hidden) {
                if !h {
                    return Some(*v);
                }
            }
            k += size;
            size *= 2;
        }
        None
    }

    /// `pick_dimension`: the dimension whose lines are closest.
    pub fn pick_dimension(&self, x: f64, y: f64) -> Option<usize> {
        let mut best = None;
        let mut best_d = self.threshold;
        for (i, dim) in self.st.scene.dimensions.iter().enumerate() {
            if !self.st.layer_selectable(dim.layer.as_deref()) {
                continue;
            }
            let (ap, bp) = dim.line_points(self.st.h);
            for (s, e) in [(dim.a, ap), (dim.b, bp), (ap, bp)] {
                let (Some(ps), Some(pe)) = (self.view.world_to_pixel(s), self.view.world_to_pixel(e)) else { continue };
                let d = point_to_segment_2d((x, y), ps, pe);
                if d < best_d {
                    best_d = d;
                    best = Some(i);
                }
            }
        }
        best
    }

    /// `pick_text_label`: the leader text under the cursor. A click on the
    /// drawn text block hits; failing that, the label position and its
    /// leader within twice the threshold. `rect_only` is the block alone.
    pub fn pick_text_label(&self, x: f64, y: f64, rect_only: bool, fm: &dyn TextMetrics) -> Option<usize> {
        let mut best = None;
        let mut best_d = self.threshold * 2.0;
        for (i, lab) in self.st.scene.text_labels.iter().enumerate() {
            if !self.st.layer_selectable(lab.layer.as_deref()) {
                continue;
            }
            let Some(pp) = self.view.world_to_pixel(lab.position()) else { continue };
            let pa = self.view.world_to_pixel(lab.anchor);
            let mut lines = splitlines(&lab.text);
            if lines.is_empty() {
                lines.push(String::new());
            }
            let block_w = lines.iter().map(|l| fm.advance(l)).max().unwrap_or(0) as f64;
            let x0 = match pa {
                Some(a) if pp.0 < a.0 => pp.0 - 6.0 - block_w,
                _ => pp.0 + 6.0,
            };
            for (k, line) in lines.iter().enumerate() {
                let base = pp.1 - 4.0 + (k as i32 * fm.height()) as f64;
                let adv = fm.advance(line) as f64;
                if x0 - 3.0 <= x
                    && x <= x0 + adv + 3.0
                    && base - fm.ascent() as f64 - 3.0 <= y
                    && y <= base + fm.descent() as f64 + 3.0
                {
                    return Some(i);
                }
            }
            if rect_only {
                continue;
            }
            let mut d = py_hypot2(pp.0 - x, pp.1 - y);
            if let Some(a) = pa {
                d = py_min(d, point_to_segment_2d((x, y), a, pp));
            }
            if d < best_d {
                best_d = d;
                best = Some(i);
            }
        }
        if rect_only {
            None
        } else {
            best
        }
    }

    /// `pick_geopath`: the georef path whose polyline is closest.
    pub fn pick_geopath(&self, x: f64, y: f64) -> Option<usize> {
        let mut best = None;
        let mut best_d = self.threshold;
        for (i, path) in self.st.scene.geo_paths.iter().enumerate() {
            for (a, b) in path.segments() {
                // `drape` is a no-op without a ground surface showing.
                let (Some(pa), Some(pb)) = (self.view.world_to_pixel(a), self.view.world_to_pixel(b)) else { continue };
                let d = point_to_segment_2d((x, y), pa, pb);
                if d < best_d {
                    best_d = d;
                    best = Some(i);
                }
            }
        }
        best
    }

    /// `pick_guide`: the construction guide nearest the cursor.
    pub fn pick_guide(&self, x: f64, y: f64) -> Option<usize> {
        let mut best = None;
        let mut best_d = self.threshold;
        for (i, g) in self.st.scene.guides.iter().enumerate() {
            let (a, b) = g.segment();
            let Some((sa, sb)) = self.view.clip_segment_front(a, b) else { continue };
            let (Some(pa), Some(pb)) = (self.view.world_to_pixel(sa), self.view.world_to_pixel(sb)) else { continue };
            let d = point_to_segment_2d((x, y), pa, pb);
            if d < best_d {
                best_d = d;
                best = Some(i);
            }
        }
        best
    }

    /// `pick_axis`: the model axis line under the cursor.
    pub fn pick_axis(&self, x: f64, y: f64) -> Option<&'static str> {
        let mut best = None;
        let mut best_d = self.threshold;
        let o = V3(0.0, 0.0, 0.0);
        for (name, axis) in [("x", V3(1.0, 0.0, 0.0)), ("y", V3(0.0, 1.0, 0.0)), ("z", V3(0.0, 0.0, 1.0))] {
            let d = axis.mul(ingetrazo_model::guide::GUIDE_HALF_LEN);
            let Some((sa, sb)) = self.view.clip_segment_front(o.sub(d), o.add(d)) else { continue };
            let (Some(pa), Some(pb)) = (self.view.world_to_pixel(sa), self.view.world_to_pixel(sb)) else { continue };
            let dist = point_to_segment_2d((x, y), pa, pb);
            if dist < best_d {
                best_d = dist;
                best = Some(name);
            }
        }
        best
    }

    /// `pick_section_plane`: the plane whose frame is nearest the cursor.
    pub fn pick_section_plane(&self, x: f64, y: f64) -> Option<usize> {
        if self.st.scene.section_planes.is_empty() || !self.st.show_section_planes() {
            return None;
        }
        let mut best = None;
        let mut best_d = self.threshold;
        for (i, sp) in self.st.scene.section_planes.iter().enumerate() {
            if sp.hidden {
                continue;
            }
            let corners = self.section_plane_frame(sp);
            for k in 0..4 {
                let Some((sa, sb)) = self.view.clip_segment_front(corners[k], corners[(k + 1) % 4]) else { continue };
                let (Some(a), Some(b)) = (self.view.world_to_pixel(sa), self.view.world_to_pixel(sb)) else { continue };
                let d = point_to_segment_2d((x, y), a, b);
                if d < best_d {
                    best_d = d;
                    best = Some(i);
                }
            }
        }
        best
    }

    /// `image_plane_at`: the topmost reference image under the cursor.
    pub fn image_plane_at(&self, x: f64, y: f64, selectable_only: bool) -> Option<usize> {
        let (origin, direction) = self.ray(x, y)?;
        let mut best = None;
        let mut best_t = f64::INFINITY;
        for (i, im) in self.st.scene.image_planes.iter().enumerate() {
            if !self.st.layer_visible(im.layer.as_deref()) {
                continue;
            }
            if selectable_only && (im.locked || !self.st.layer_selectable(im.layer.as_deref())) {
                continue;
            }
            if let Some(t) = self.plane_hit(im, origin, direction, best_t) {
                best = Some(i);
                best_t = t;
            }
        }
        best
    }

    fn plane_hit(&self, im: &ingetrazo_model::image_plane::ImagePlane, origin: V3, direction: V3, best_t: f64) -> Option<f64> {
        let n = im.normal(self.st.h);
        let denom = n.dot(direction) as f64;
        if denom.abs() < 1e-9 {
            return None;
        }
        let t = n.dot(im.center().sub(origin)) as f64 / denom;
        if t < 0.0 || t > best_t {
            return None;
        }
        let (fu, fv) = im.project(origin.add(direction.mul(t as f32)));
        im.contains_uv(fu, fv).then_some(t)
    }

    /// `pick_image_plane`: the selection entry point, locked images skipped.
    pub fn pick_image_plane(&self, x: f64, y: f64) -> Option<usize> {
        self.image_plane_at(x, y, true)
    }

    /// `pick_locked_image_plane`: a locked image under the cursor, for the
    /// context menu.
    pub fn pick_locked_image_plane(&self, x: f64, y: f64) -> Option<usize> {
        let (origin, direction) = self.ray(x, y)?;
        let mut best = None;
        let mut best_t = f64::INFINITY;
        for (i, im) in self.st.scene.image_planes.iter().enumerate() {
            if !im.locked {
                continue;
            }
            if !self.st.layer_visible(im.layer.as_deref()) || !self.st.layer_selectable(im.layer.as_deref()) {
                continue;
            }
            if let Some(t) = self.plane_hit(im, origin, direction, best_t) {
                best = Some(i);
                best_t = t;
            }
        }
        best
    }

    /// Whether a group placement may be picked (for callers that ask).
    pub fn placement_selectable(&self, place: usize) -> bool {
        group_selectable(&self.st, &self.idx.places[place])
    }

    /// `Hits` unwrapped for the per-entity case.
    #[allow(dead_code)]
    fn per_ent(h: Hits) -> Vec<f64> {
        match h {
            Hits::PerEnt(t) => t,
            Hits::Global(g) => vec![g],
        }
    }
}

/// `np.clip(x, 0.0, 1.0)`.
fn py_max_np(x: f64) -> f64 {
    if x.is_nan() {
        return x;
    }
    x.clamp(0.0, 1.0)
}
