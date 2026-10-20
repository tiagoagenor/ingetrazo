// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! Section planes in the viewport: the clip plane, the cut line, the solid
//! and hatched fill (`core/section.py` `cut_chords`, `fill_triangles`,
//! `hatch_segments`; `Viewport._draw_section_fill` and
//! `_draw_section_cut_edges`), plus the guides the scene stores as JSON.

use std::sync::Arc;

use ingetrazo_formats::pyjson::Value;
use ingetrazo_geom::facegeom::earcut;
use ingetrazo_geom::hlr_draw::section_loops;
use ingetrazo_math::{Hypot, V3};
use ingetrazo_model::guide::Guide;
use ingetrazo_model::mesh::Mesh;
use ingetrazo_model::scene::Scene;
use ingetrazo_model::section::{effective_fill, SectionPlane};

use crate::bake::{face_tris, FaceVertex};
use crate::camera::Camera;
use crate::frame::{ColMode, Geo, PipeKey};
use crate::renderer::{Builder, Renderer, P};
use crate::style::{FaceMode, Style};

pub type Chord = [[f64; 3]; 2];

/// The active cut: its plane and the clip vector the shaders use (the kept
/// side is where `dot(world, clip) >= 0`).
#[derive(Clone, Debug)]
pub struct ActiveCut {
    pub plane: SectionPlane,
    pub clip: [f32; 4],
}

/// `scene.active_section()`, unless the cuts are switched off.
pub fn active_cut(scene: &Scene) -> Option<ActiveCut> {
    if scene.extra.cuts_hidden {
        return None;
    }
    for sp in scene.section_planes.iter() {
        if sp.active {
            let (n, o) = (sp.normal, sp.point);
            let clip = [-n.0, -n.1, -n.2, n.dot(o)];
            return Some(ActiveCut { plane: sp.clone(), clip });
        }
    }
    None
}

/// The scene's guides (`scene.guides`), the finite ones.
pub fn parse_guides(scene: &Scene, _h: Hypot) -> Vec<Guide> {
    scene.guides.iter().filter(|g| g.is_finite()).cloned().collect()
}

/// `cut_chords` over a mesh placed by `m`: the chords the plane takes out of
/// its visible triangles.
pub fn mesh_chords(mesh: &Mesh, m: &ingetrazo_math::M4, n: [f64; 3], c: f64, out: &mut Vec<Chord>) {
    for &fid in &mesh.faces {
        let face = mesh.face(fid);
        let hidden = face.attrs.iter().any(|(k, v)| k == "hidden" && ingetrazo_model::pyval::truthy(v));
        if hidden {
            continue;
        }
        for t in face_tris(mesh, fid) {
            let p: [[f64; 3]; 3] = [0, 1, 2].map(|i| {
                let q = m.map(t[i]);
                [q.0 as f64, q.1 as f64, q.2 as f64]
            });
            let d = p.map(|v| v[0] * n[0] + v[1] * n[1] + v[2] * n[2] - c);
            let mut hits: Vec<[f64; 3]> = Vec::with_capacity(2);
            for (a, b) in [(0usize, 1usize), (1, 2), (2, 0)] {
                if d[a] * d[b] < 0.0 {
                    let t = d[a] / (d[a] - d[b]);
                    hits.push([
                        p[a][0] + (p[b][0] - p[a][0]) * t,
                        p[a][1] + (p[b][1] - p[a][1]) * t,
                        p[a][2] + (p[b][2] - p[a][2]) * t,
                    ]);
                }
            }
            if hits.len() == 2 {
                out.push([hits[0], hits[1]]);
            }
        }
    }
}

fn plane_basis(n: [f64; 3]) -> ([f64; 3], [f64; 3]) {
    let l = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
    let n = if l > 0.0 { [n[0] / l, n[1] / l, n[2] / l] } else { n };
    let helper = if n[0].abs() < 0.9 { [1.0, 0.0, 0.0] } else { [0.0, 1.0, 0.0] };
    let cross = |a: [f64; 3], b: [f64; 3]| [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
    let mut u = cross(n, helper);
    let lu = (u[0] * u[0] + u[1] * u[1] + u[2] * u[2]).sqrt();
    u = [u[0] / lu, u[1] / lu, u[2] / lu];
    (u, cross(n, u))
}

fn inside(points: &[[f64; 2]], poly: &[[f64; 2]]) -> f64 {
    // Fraction of the points inside (even-odd).
    let mut count = 0usize;
    for p in points {
        let mut c = false;
        for i in 0..poly.len() {
            let (a, b) = (poly[i], poly[(i + 1) % poly.len()]);
            if (a[1] > p[1]) != (b[1] > p[1]) {
                let dy = if b[1] == a[1] { 1.0 } else { b[1] - a[1] };
                let xint = a[0] + (p[1] - a[1]) * (b[0] - a[0]) / dy;
                if p[0] < xint {
                    c = !c;
                }
            }
        }
        if c {
            count += 1;
        }
    }
    count as f64 / points.len().max(1) as f64
}

/// One filled area: an outer ring followed by its holes, in world and plane
/// coordinates.
pub struct FillGroup {
    pub rings3: Vec<Vec<[f64; 3]>>,
    pub rings2: Vec<Vec<[f64; 2]>>,
}

/// `_fill_groups`.
pub fn fill_groups(chords: &[Chord], normal: [f64; 3]) -> (Vec<FillGroup>, Option<([f64; 3], [f64; 3])>) {
    if chords.is_empty() {
        return (Vec::new(), None);
    }
    let rings = section_loops(chords);
    if rings.is_empty() {
        return (Vec::new(), None);
    }
    let (u, v) = plane_basis(normal);
    let flat: Vec<Vec<[f64; 2]>> = rings
        .iter()
        .map(|r| {
            r.iter()
                .map(|p| {
                    [p[0] * u[0] + p[1] * u[1] + p[2] * u[2], p[0] * v[0] + p[1] * v[1] + p[2] * v[2]]
                })
                .collect()
        })
        .collect();
    let area: Vec<f64> = flat
        .iter()
        .map(|f| {
            let n = f.len();
            let mut s = 0.0;
            for i in 0..n {
                let (a, b) = (f[i], f[(i + 1) % n]);
                s += a[0] * b[1] - a[1] * b[0];
            }
            (0.5 * s).abs()
        })
        .collect();
    let bounds: Vec<([f64; 2], [f64; 2])> = flat
        .iter()
        .map(|f| {
            let mut lo = [f64::INFINITY; 2];
            let mut hi = [f64::NEG_INFINITY; 2];
            for p in f {
                for k in 0..2 {
                    lo[k] = lo[k].min(p[k]);
                    hi[k] = hi[k].max(p[k]);
                }
            }
            (lo, hi)
        })
        .collect();
    let count = rings.len();
    let mut holders: Vec<Vec<usize>> = vec![Vec::new(); count];
    for i in 0..count {
        for j in 0..count {
            if area[j] > area[i]
                && bounds[j].0[0] <= bounds[i].0[0]
                && bounds[j].0[1] <= bounds[i].0[1]
                && bounds[j].1[0] >= bounds[i].1[0]
                && bounds[j].1[1] >= bounds[i].1[1]
                && inside(&flat[i], &flat[j]) >= 0.9
            {
                holders[i].push(j);
            }
        }
    }
    let depth: Vec<usize> = holders.iter().map(|h| h.len()).collect();
    let parent: Vec<Option<usize>> = holders
        .iter()
        .map(|h| h.iter().copied().min_by(|&a, &b| area[a].total_cmp(&area[b])))
        .collect();
    let mut holes_of: Vec<Vec<usize>> = vec![Vec::new(); count];
    for k in 0..count {
        if depth[k] % 2 == 1 {
            if let Some(p) = parent[k] {
                holes_of[p].push(k);
            }
        }
    }
    let mut groups = Vec::new();
    for i in 0..count {
        if depth[i] % 2 == 1 {
            continue;
        }
        let mut members = vec![i];
        members.extend(holes_of[i].iter().copied());
        groups.push(FillGroup {
            rings3: members.iter().map(|&k| rings[k].clone()).collect(),
            rings2: members.iter().map(|&k| flat[k].clone()).collect(),
        });
    }
    (groups, Some((u, v)))
}

/// `fill_triangles`: the cut as triangles (world coordinates).
pub fn fill_triangles(chords: &[Chord], normal: [f64; 3]) -> Vec<[[f64; 3]; 3]> {
    let (groups, _) = fill_groups(chords, normal);
    let mut out = Vec::new();
    for g in &groups {
        let mut pts2: Vec<(f64, f64)> = Vec::new();
        let mut pts3: Vec<[f64; 3]> = Vec::new();
        let mut starts: Vec<usize> = Vec::new();
        for (k, (f2, f3)) in g.rings2.iter().zip(&g.rings3).enumerate() {
            if k > 0 {
                starts.push(pts2.len());
            }
            pts2.extend(f2.iter().map(|p| (p[0], p[1])));
            pts3.extend(f3.iter().copied());
        }
        for t in earcut(&pts2, &starts) {
            out.push([pts3[t[0]], pts3[t[1]], pts3[t[2]]]);
        }
    }
    out
}

/// `hatch_segments`: lines at 45 degrees (and 135 when `cross`) clipped to
/// the filled areas, `spacing` apart.
pub fn hatch_segments(chords: &[Chord], normal: [f64; 3], spacing: f64, cross: bool) -> Vec<[[f64; 3]; 2]> {
    let (groups, basis) = fill_groups(chords, normal);
    let Some((u, v)) = basis else { return Vec::new() };
    if groups.is_empty() || spacing <= 0.0 {
        return Vec::new();
    }
    let n = [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]];
    let p0 = groups[0].rings3[0][0];
    let dn = p0[0] * n[0] + p0[1] * n[1] + p0[2] * n[2];
    let off = [dn * n[0], dn * n[1], dn * n[2]];
    let mut out = Vec::new();
    let angles: &[f64] = if cross { &[45.0, 135.0] } else { &[45.0] };
    for ang in angles {
        let t = ang.to_radians();
        let d = [t.cos(), t.sin()];
        let m = [-d[1], d[0]];
        for g in &groups {
            let (mut smin, mut smax) = (f64::INFINITY, f64::NEG_INFINITY);
            for f in &g.rings2 {
                for p in f {
                    let s = p[0] * m[0] + p[1] * m[1];
                    smin = smin.min(s);
                    smax = smax.max(s);
                }
            }
            let k0 = (smin / spacing).ceil() as i64;
            let k1 = (smax / spacing).floor() as i64;
            if k1 - k0 > 20000 {
                continue;
            }
            for k in k0..=k1 {
                let s = k as f64 * spacing;
                let mut hits: Vec<f64> = Vec::new();
                for f in &g.rings2 {
                    for i in 0..f.len() {
                        let (a, b) = (f[i], f[(i + 1) % f.len()]);
                        let sa = a[0] * m[0] + a[1] * m[1] - s;
                        let sb = b[0] * m[0] + b[1] * m[1] - s;
                        if (sa > 0.0) != (sb > 0.0) {
                            let tt = sa / (sa - sb);
                            let p = [a[0] + (b[0] - a[0]) * tt, a[1] + (b[1] - a[1]) * tt];
                            hits.push(p[0] * d[0] + p[1] * d[1]);
                        }
                    }
                }
                hits.sort_by(|a, b| a.total_cmp(b));
                let mut j = 0;
                while j + 1 < hits.len() {
                    let at = |h: f64| {
                        let q = [d[0] * h + m[0] * s, d[1] * h + m[1] * s];
                        [
                            q[0] * u[0] + q[1] * v[0] + off[0],
                            q[0] * u[1] + q[1] * v[1] + off[1],
                            q[0] * u[2] + q[1] * v[2] + off[2],
                        ]
                    };
                    out.push([at(hits[j]), at(hits[j + 1])]);
                    j += 2;
                }
            }
        }
    }
    out
}

/// Chords of the active cut against the whole visible model. Cached by what
/// they depend on (the meshes' serials, the placements, the plane): an orbit
/// repeats the same cut every frame.
pub fn scene_chords(r: &Renderer, scene: &Scene, cut: &ActiveCut) -> Arc<Vec<Chord>> {
    let n = [cut.plane.normal.0 as f64, cut.plane.normal.1 as f64, cut.plane.normal.2 as f64];
    let o = [cut.plane.point.0 as f64, cut.plane.point.1 as f64, cut.plane.point.2 as f64];
    let c = n[0] * o[0] + n[1] * o[1] + n[2] * o[2];
    let mut key: u64 = 0xcbf2_9ce4_8422_2325;
    let mut mix = |x: u64| {
        key = (key ^ x).wrapping_mul(0x0100_0000_01b3);
    };
    for v in n.iter().chain(o.iter()) {
        mix(v.to_bits());
    }
    for p in r.placements() {
        if p.ghost {
            continue;
        }
        let m = scene.mesh(p.mesh);
        mix(m.serial);
        let (a, b, f) = m.arena_sizes();
        mix((a + 31 * b + 997 * f) as u64);
        for x in p.matrix.map(|m| m.data()).unwrap_or([0.0; 16]) {
            mix(x.to_bits() as u64);
        }
    }
    let mut cache = r.chord_cache().lock().unwrap_or_else(|e| e.into_inner());
    if let Some((k, v)) = cache.as_ref() {
        if *k == key {
            return Arc::clone(v);
        }
    }
    let mut out = Vec::new();
    for p in r.placements() {
        if p.ghost {
            continue;
        }
        let m = p.matrix.unwrap_or_else(ingetrazo_math::M4::identity);
        mesh_chords(scene.mesh(p.mesh), &m, n, c, &mut out);
    }
    let out = Arc::new(out);
    *cache = Some((key, Arc::clone(&out)));
    out
}

fn tri_verts(tris: &[[[f64; 3]; 3]]) -> Vec<FaceVertex> {
    tris.iter()
        .flat_map(|t| t.iter())
        .map(|p| FaceVertex { pos: [p[0] as f32, p[1] as f32, p[2] as f32], uv: [0.0; 2], rgba: [255; 4], misc: 0 })
        .collect()
}

/// `_draw_section_fill`.
pub(crate) fn add_fill(b: &mut Builder, r: &Renderer, scene: &Scene, cut: &ActiveCut, style: &Style, mode: FaceMode) {
    if matches!(mode, FaceMode::Wireframe | FaceMode::Xray) {
        return;
    }
    let sc = style.section_fill_color;
    let Ok((pattern, color, spacing)) = effective_fill(
        cut.plane.fill.as_ref(),
        style.section_fill,
        [sc[0] as f64, sc[1] as f64, sc[2] as f64],
    ) else {
        return;
    };
    let pattern = match &pattern {
        Value::Str(s) => s.as_str(),
        _ => "solid",
    };
    if pattern == "none" || color.len() < 3 {
        return;
    }
    let chords = scene_chords(r, scene, cut);
    let chords: &[Chord] = &chords;
    if chords.is_empty() {
        return;
    }
    let normal = [cut.plane.normal.0 as f64, cut.plane.normal.1 as f64, cut.plane.normal.2 as f64];
    let tris = fill_triangles(chords, normal);
    if tris.is_empty() {
        return;
    }
    let fc = [color[0] as f32, color[1] as f32, color[2] as f32];
    let base = if pattern == "solid" { fc } else { fc.map(|c| c + (1.0 - c) * 0.78) };
    let range = b.dyn_faces(&tri_verts(&tris));
    let slot = b.slot(P {
        color: [base[0], base[1], base[2], 1.0],
        back: [base[0], base[1], base[2], 1.0],
        mode: ColMode::Flat,
        ..P::default()
    });
    // The fill lies IN the plane, which clips: drawn unclipped, a hair behind
    // so the thick cut line wins.
    b.cmd(PipeKey { bias: 2, ..PipeKey::face() }, slot, None, Geo::DynFaces, range, None);
    if pattern == "hatch" || pattern == "cross" {
        let lines = hatch_segments(chords, normal, spacing, pattern == "cross");
        if !lines.is_empty() {
            let verts: Vec<crate::bake::EdgeVertex> = lines
                .iter()
                .flat_map(|s| {
                    let a = [s[0][0] as f32, s[0][1] as f32, s[0][2] as f32];
                    let e = [s[1][0] as f32, s[1][1] as f32, s[1][2] as f32];
                    [crate::bake::EdgeVertex { pos: a, anchor: e }, crate::bake::EdgeVertex { pos: e, anchor: e }]
                })
                .collect();
            let rr = b.dyn_edges(&verts);
            let slot = b.slot(P { color: [fc[0], fc[1], fc[2], 1.0], ..P::default() });
            b.cmd(PipeKey::line(), slot, None, Geo::DynEdges, rr, None);
        }
    }
}

/// `_draw_section_cut_edges`: thick lines in the plane, about 2.5 px wide.
pub(crate) fn add_cut_edges(b: &mut Builder, r: &Renderer, scene: &Scene, cut: &ActiveCut, cam: &Camera, logical_height: f32, style: &Style) {
    let chords = scene_chords(r, scene, cut);
    let chords: &[Chord] = &chords;
    if chords.is_empty() {
        return;
    }
    let n = [cut.plane.normal.0 as f64, cut.plane.normal.1 as f64, cut.plane.normal.2 as f64];
    let eye = cam.eye();
    let eye = [eye.0 as f64, eye.1 as f64, eye.2 as f64];
    let h = (logical_height as f64).max(1.0);
    let fov = cam.fov_deg.to_radians();
    let mut tris: Vec<[[f64; 3]; 3]> = Vec::new();
    for ch in chords {
        let (a, bb) = (ch[0], ch[1]);
        let mut t = [bb[0] - a[0], bb[1] - a[1], bb[2] - a[2]];
        let ln = (t[0] * t[0] + t[1] * t[1] + t[2] * t[2]).sqrt();
        if ln <= 1e-9 {
            continue;
        }
        t = [t[0] / ln, t[1] / ln, t[2] / ln];
        let side = [n[1] * t[2] - n[2] * t[1], n[2] * t[0] - n[0] * t[2], n[0] * t[1] - n[1] * t[0]];
        let mid = [(a[0] + bb[0]) * 0.5, (a[1] + bb[1]) * 0.5, (a[2] + bb[2]) * 0.5];
        let dist = ((mid[0] - eye[0]).powi(2) + (mid[1] - eye[1]).powi(2) + (mid[2] - eye[2]).powi(2)).sqrt();
        let wpp = (dist * (fov * 0.5).tan() * 2.0 / h).max(1e-6);
        let w = wpp * 1.25;
        let lift = [n[0] * 0.6 * w, n[1] * 0.6 * w, n[2] * 0.6 * w];
        let off = [side[0] * w, side[1] * w, side[2] * w];
        let q = |p: [f64; 3], s: f64| [p[0] + s * off[0] + lift[0], p[1] + s * off[1] + lift[1], p[2] + s * off[2] + lift[2]];
        let (q0, q1, q2, q3) = (q(a, -1.0), q(bb, -1.0), q(bb, 1.0), q(a, 1.0));
        tris.push([q0, q1, q2]);
        tris.push([q0, q2, q3]);
    }
    if tris.is_empty() {
        return;
    }
    let ec = style.edge_color;
    let range = b.dyn_faces(&tri_verts(&tris));
    let slot = b.slot(P {
        color: [ec[0], ec[1], ec[2], 1.0],
        back: [ec[0], ec[1], ec[2], 1.0],
        mode: ColMode::Flat,
        ..P::default()
    });
    b.cmd(PipeKey { bias: 0, ..PipeKey::face() }, slot, None, Geo::DynFaces, range, None);
}

#[allow(dead_code)]
fn unused(_: V3) {}

#[cfg(test)]
mod tests {
    use super::*;

    fn square(z: f64, x0: f64, x1: f64, y0: f64, y1: f64) -> Vec<Chord> {
        vec![
            [[x0, y0, z], [x1, y0, z]],
            [[x1, y0, z], [x1, y1, z]],
            [[x1, y1, z], [x0, y1, z]],
            [[x0, y1, z], [x0, y0, z]],
        ]
    }

    #[test]
    fn a_ring_fills_with_two_triangles_and_a_hole_stays_open() {
        let solid = square(1.0, 0.0, 4.0, 0.0, 4.0);
        let t = fill_triangles(&solid, [0.0, 0.0, 1.0]);
        assert_eq!(t.len(), 2);
        let mut ring = solid.clone();
        ring.extend(square(1.0, 1.0, 3.0, 1.0, 3.0));
        let t = fill_triangles(&ring, [0.0, 0.0, 1.0]);
        let area: f64 = t
            .iter()
            .map(|t| {
                let u = [t[1][0] - t[0][0], t[1][1] - t[0][1]];
                let v = [t[2][0] - t[0][0], t[2][1] - t[0][1]];
                0.5 * (u[0] * v[1] - u[1] * v[0]).abs()
            })
            .sum();
        assert!((area - 12.0).abs() < 1e-9, "16 - 4 = 12, got {area}");
    }

    #[test]
    fn hatch_lines_stay_inside_the_ring() {
        let solid = square(0.0, 0.0, 2.0, 0.0, 2.0);
        let lines = hatch_segments(&solid, [0.0, 0.0, 1.0], 0.5, false);
        assert!(!lines.is_empty());
        for l in &lines {
            for p in l {
                assert!(p[0] > -1e-9 && p[0] < 2.0 + 1e-9 && p[1] > -1e-9 && p[1] < 2.0 + 1e-9, "{p:?}");
            }
        }
        let crossed = hatch_segments(&solid, [0.0, 0.0, 1.0], 0.5, true);
        assert!(crossed.len() > lines.len());
    }

    #[test]
    fn an_open_chain_is_not_filled() {
        let open = vec![[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], [[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]]];
        assert!(fill_triangles(&open, [0.0, 0.0, 1.0]).is_empty());
    }
}
