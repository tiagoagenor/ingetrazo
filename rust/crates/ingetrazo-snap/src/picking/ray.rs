// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! The ray against the pick index (`Viewport._ray_hits`,
//! `_hover_face_t`, `_coplanar_eps`) and the occlusion of a point
//! (`_is_occluded`, `_occluded_many`).

use ingetrazo_math::V3;

use super::index::{PickIndex, Span};
use super::numeric::{cross3, dot3, einsum3, matmul3, matvec3, p3, ray_aabb, P3};
use super::state::FaceMode;
use super::Picker;

/// What `_ray_hits` answers.
pub enum Hits {
    /// The nearest `t` over everything (`reduce_global`).
    Global(f64),
    /// The nearest `t` of each entity (`inf` = none).
    PerEnt(Vec<f64>),
}

impl Picker<'_> {
    /// `_ray_hits`: Möller–Trumbore over the triangles of the chunks whose
    /// box the ray meets, one NumPy pass over their concatenation. `None`
    /// when the index holds no triangles.
    pub fn ray_hits(&self, o: P3, d: P3, mask: &[bool], reduce_global: bool) -> Option<Hits> {
        self.ray_hits_touched(o, d, mask, reduce_global, None)
    }

    /// [`Picker::ray_hits`] that also lists, when `touched` is given, the
    /// entities the ray hit (each once, in no order): the caller then reads
    /// those and not the whole per-entity array.
    pub fn ray_hits_touched(&self, o: P3, d: P3, mask: &[bool], reduce_global: bool, mut touched: Option<&mut Vec<usize>>) -> Option<Hits> {
        let idx = &self.idx;
        if !idx.has_tris() {
            return None;
        }
        let spans = self.spans_met(o, d);
        let mut best = f64::INFINITY;
        let mut face_t = if reduce_global { Vec::new() } else { vec![f64::INFINITY; idx.ents.len()] };
        let rows: Vec<usize> = spans.iter().flat_map(|s| s.start..s.start + s.n).collect();
        let n = rows.len();
        for &r in &rows {
            let (v0, e1, e2) = (idx.v0[r], idx.e1[r], idx.e2[r]);
            let p = cross3(d, e2);
            let det = einsum3(e1, p);
            let ok = det.abs() > 1e-6;
            let inv = if ok { 1.0 / det } else { 0.0 };
            let s = [o[0] - v0[0], o[1] - v0[1], o[2] - v0[2]];
            let u = einsum3(s, p) * inv;
            let q = cross3(s, e1);
            let v = matvec3(q, d, n) * inv;
            let t = einsum3(e2, q) * inv;
            let te = idx.tri_ent[r];
            let hit = ok && u >= 0.0 && u <= 1.0 && v >= 0.0 && u + v <= 1.0 && t > 1e-6 && mask[te];
            if hit {
                if reduce_global {
                    if t < best {
                        best = t;
                    }
                } else if t < face_t[te] {
                    if let Some(list) = touched.as_deref_mut() {
                        if face_t[te] == f64::INFINITY {
                            list.push(te);
                        }
                    }
                    face_t[te] = t;
                }
            }
        }
        Some(if reduce_global { Hits::Global(best) } else { Hits::PerEnt(face_t) })
    }

    /// The spans a ray meets: `_ray_aabb` on each chunk box, or, past 64
    /// spans, the vectorised slab test.
    fn spans_met(&self, o: P3, d: P3) -> Vec<Span> {
        let spans = &self.idx.spans;
        if spans.len() > 64 {
            let inf = f64::INFINITY;
            let inv: Vec<f64> = d.iter().map(|&x| 1.0 / if x.abs() > 1e-12 { x } else { 1e-12 }).collect();
            let mut met = Vec::new();
            for s in spans {
                let (lo, hi) = s.bbox.unwrap_or(([-inf; 3], [inf; 3]));
                let mut tmin = f64::NAN;
                let mut tmax = f64::NAN;
                for k in 0..3 {
                    let t1 = (lo[k] - o[k]) * inv[k];
                    let t2 = (hi[k] - o[k]) * inv[k];
                    let mn = if t1.is_nan() || t2.is_nan() { f64::NAN } else { t1.min(t2) };
                    let mx = if t1.is_nan() || t2.is_nan() { f64::NAN } else { t1.max(t2) };
                    if !mn.is_nan() && (tmin.is_nan() || mn > tmin) {
                        tmin = mn;
                    }
                    if !mx.is_nan() && (tmax.is_nan() || mx < tmax) {
                        tmax = mx;
                    }
                }
                let lo0 = if tmin.is_nan() { f64::NAN } else { tmin.max(0.0) };
                if tmax >= lo0 {
                    met.push(Span { bbox: None, start: s.start, n: s.n });
                }
            }
            // Spans with a box of None take every ray in the pass below.
            return met.into_iter().filter(|s| s.n > 0).collect();
        }
        spans
            .iter()
            .filter(|s| s.n > 0 && s.bbox.is_none_or(|(lo, hi)| ray_aabb(o, d, lo, hi, f64::INFINITY)))
            .copied()
            .collect()
    }

    /// `_hover_face_t`: the nearest hit of each selectable entity, with what
    /// lies beyond the active cut taken out.
    pub fn hover_face_t(&self, origin: V3, direction: V3) -> Option<Vec<f64>> {
        let (o, d) = (p3(origin), p3(direction));
        let mut face_t = match self.ray_hits(o, d, &self.idx.sel, false)? {
            Hits::PerEnt(t) => t,
            Hits::Global(_) => unreachable!(),
        };
        if let Some(sp) = self.st.active_cut() {
            let fin: Vec<usize> = (0..face_t.len()).filter(|&i| face_t[i].is_finite()).collect();
            if !fin.is_empty() {
                let n = p3(sp.normal);
                let c = dot3(n, p3(sp.point));
                for &i in &fin {
                    let t = face_t[i];
                    let pt = [o[0] + d[0] * t, o[1] + d[1] * t, o[2] + d[2] * t];
                    if matvec3(pt, n, fin.len()) - c > 1e-6 {
                        face_t[i] = f64::INFINITY;
                    }
                }
            }
        }
        Some(face_t)
    }

    /// [`Picker::hover_face_t`] with the list of the entities it hit (those
    /// whose time is finite), for the picks that read only those.
    pub fn hover_face_t_touched(&self, origin: V3, direction: V3) -> Option<(Vec<f64>, Vec<usize>)> {
        if self.st.active_cut().is_some() {
            let t = self.hover_face_t(origin, direction)?;
            let touched = (0..t.len()).filter(|&i| t[i].is_finite()).collect();
            return Some((t, touched));
        }
        let (o, d) = (p3(origin), p3(direction));
        let mut touched = Vec::new();
        match self.ray_hits_touched(o, d, &self.idx.sel, false, Some(&mut touched))? {
            Hits::PerEnt(t) => Some((t, touched)),
            Hits::Global(_) => unreachable!(),
        }
    }

    /// `_coplanar_eps`: how much deeper than the nearest hit a face may sit
    /// and still count as the same depth.
    pub fn coplanar_eps(&self, origin: V3, direction: V3, best_t: f64) -> f64 {
        let t_eye = self.view.eye.sub(origin).dot(direction) as f64;
        let e = (best_t - t_eye).abs() * 1e-4;
        if e > 1e-4 {
            e
        } else {
            1e-4
        }
    }

    /// `_is_occluded`: whether geometry sits between the eye and `world`.
    pub fn is_occluded(&self, world: V3) -> bool {
        let sp = self.st.active_cut();
        if let Some(sp) = sp {
            if (sp.side(world) as f64) > 1e-6 {
                return true;
            }
        }
        if matches!(self.st.face_mode, FaceMode::Xray | FaceMode::Wireframe) {
            return false;
        }
        if !self.idx.has_tris() {
            return false;
        }
        let origin = self.view.eye;
        let delta = world.sub(origin);
        let dist = delta.length(self.st.h);
        if (dist as f64) < 1e-9 {
            return false;
        }
        let dv = delta.div(dist);
        let (o, d) = (p3(origin), p3(dv));
        let dist = dist as f64;
        match sp {
            None => match self.ray_hits(o, d, &self.idx.vis, true) {
                Some(Hits::Global(nearest)) => nearest < dist - 1e-3,
                _ => false,
            },
            Some(sp) => {
                let face_t = match self.ray_hits(o, d, &self.idx.vis, false) {
                    Some(Hits::PerEnt(t)) => t,
                    _ => return false,
                };
                let hit: Vec<usize> = (0..face_t.len()).filter(|&i| face_t[i].is_finite() && face_t[i] < dist - 1e-3).collect();
                if hit.is_empty() {
                    return false;
                }
                let n = p3(sp.normal);
                let c = dot3(n, p3(sp.point));
                hit.iter().any(|&i| {
                    let t = face_t[i];
                    let pt = [o[0] + d[0] * t, o[1] + d[1] * t, o[2] + d[2] * t];
                    matvec3(pt, n, hit.len()) - c <= 1e-6
                })
            }
        }
    }

    /// `_occluded_many`: `is_occluded` for several points that cast from
    /// the one eye. With an active cut it asks point by point.
    pub fn occluded_many(&self, worlds: &[V3]) -> Vec<bool> {
        if !worlds.is_empty() && self.st.active_cut().is_some() {
            return worlds.iter().map(|&w| self.is_occluded(w)).collect();
        }
        let mut out = vec![false; worlds.len()];
        if worlds.is_empty() || matches!(self.st.face_mode, FaceMode::Xray | FaceMode::Wireframe) {
            return out;
        }
        let idx: &PickIndex = &self.idx;
        if !idx.has_tris() {
            return out;
        }
        let origin = self.view.eye;
        let mut rows = Vec::new();
        let mut dirs: Vec<P3> = Vec::new();
        let mut dists = Vec::new();
        for (i, &w) in worlds.iter().enumerate() {
            let delta = w.sub(origin);
            let dist = delta.length(self.st.h);
            if (dist as f64) < 1e-9 {
                continue;
            }
            rows.push(i);
            dirs.push(p3(delta.div(dist)));
            dists.push(dist as f64);
        }
        if rows.is_empty() {
            return out;
        }
        let o = p3(origin);
        let inf = f64::INFINITY;
        let spans = &idx.spans;
        let limit: Vec<f64> = dists.iter().map(|d| d - 1e-3).collect();
        let r = rows.len();
        let mut best = vec![inf; r];
        // The box prefilter, in both of its regimes.
        let mut meet = vec![vec![false; spans.len()]; r];
        for (si, s) in spans.iter().enumerate() {
            for ri in 0..r {
                let d = dirs[ri];
                let (lo, hi) = match s.bbox {
                    None => {
                        meet[ri][si] = true;
                        continue;
                    }
                    Some(b) => b,
                };
                let (tmin, tmax, ok);
                if spans.len() > 64 {
                    let mut mn = f64::NAN;
                    let mut mx = f64::NAN;
                    for k in 0..3 {
                        let inv = 1.0 / if d[k].abs() > 1e-12 { d[k] } else { 1e-12 };
                        let t1 = (lo[k] - o[k]) * inv;
                        let t2 = (hi[k] - o[k]) * inv;
                        let (a, b) = if t1.is_nan() || t2.is_nan() { (f64::NAN, f64::NAN) } else { (t1.min(t2), t1.max(t2)) };
                        if !a.is_nan() && (mn.is_nan() || a > mn) {
                            mn = a;
                        }
                        if !b.is_nan() && (mx.is_nan() || b < mx) {
                            mx = b;
                        }
                    }
                    tmin = if mn.is_nan() { f64::NAN } else { mn.max(0.0) };
                    tmax = mx;
                    ok = tmax >= tmin;
                } else {
                    let mut lo_t = f64::NEG_INFINITY;
                    let mut hi_t = inf;
                    let mut all = true;
                    for k in 0..3 {
                        let par = d[k].abs() < 1e-12;
                        let inv = 1.0 / if par { 1.0 } else { d[k] };
                        let t1 = (lo[k] - o[k]) * inv;
                        let t2 = (hi[k] - o[k]) * inv;
                        let (a, b) = if par { (f64::NEG_INFINITY, inf) } else { (t1.min(t2), t1.max(t2)) };
                        lo_t = lo_t.max(a);
                        hi_t = hi_t.min(b);
                        let inside = o[k] >= lo[k] - 1e-9 && o[k] <= hi[k] + 1e-9;
                        if par && !inside {
                            all = false;
                        }
                    }
                    tmin = lo_t.max(0.0);
                    tmax = hi_t;
                    ok = all && tmin <= tmax;
                }
                meet[ri][si] = ok && tmin < limit[ri];
            }
        }
        let _ = &mut best;
        for si in 0..spans.len() {
            let s = spans[si];
            if s.n == 0 {
                continue;
            }
            let live: Vec<usize> = (s.start..s.start + s.n).filter(|&t| idx.vis[idx.tri_ent[t]]).collect();
            if live.is_empty() {
                continue;
            }
            let n = live.len();
            for ri in 0..r {
                if !meet[ri][si] || best[ri] < limit[ri] {
                    continue;
                }
                let d = dirs[ri];
                let mut tm = inf;
                for &t_ix in &live {
                    let (v0, e1, e2) = (idx.v0[t_ix], idx.e1[t_ix], idx.e2[t_ix]);
                    let sv = [o[0] - v0[0], o[1] - v0[1], o[2] - v0[2]];
                    let q = cross3(sv, e1);
                    let uq = einsum3(e2, q);
                    let p = cross3(d, e2);
                    let det = einsum3(e1, p);
                    let ok = det.abs() > 1e-6;
                    let inv = if ok { 1.0 / det } else { 0.0 };
                    let u = einsum3(sv, p) * inv;
                    let v = if r == 1 && n >= 1 {
                        d[2].mul_add(q[2], d[0].mul_add(q[0], d[1] * q[1])) * inv
                    } else {
                        matmul3(d, q, 2) * inv
                    };
                    let t = uq * inv;
                    if ok && u >= 0.0 && u <= 1.0 && v >= 0.0 && u + v <= 1.0 && t > 1e-6 && t < tm {
                        tm = t;
                    }
                }
                if tm < best[ri] {
                    best[ri] = tm;
                }
            }
        }
        for (j, &i) in rows.iter().enumerate() {
            out[i] = best[j] < limit[j];
        }
        out
    }
}
