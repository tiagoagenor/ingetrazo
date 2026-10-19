// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! The geometry of `core/vertex_bevel.py`: bevelling the corners (vertices)
//! of a model instead of its edges, planned as a [`FilletPlan`] for the mesh
//! code to apply. Same arithmetic, same order as the Python; see
//! [`crate::fillet`] for the conventions (indices for identities, errors as
//! an enum, hash orders fixed).

use std::collections::HashMap;

use ingetrazo_math::pyfloat::{py_max2, py_sum};
use ingetrazo_math::{Hypot, V3};

use crate::fillet::{
    dotf, divf, exponent, facing, lenf, mulf, same, unit, zero, Corners, FilletError,
    FilletPlan, Measure, Rebuilt, TopoMesh, MIN_ANGLE_DEG, PLANE_TOL, PY_SUM_COMPENSATED,
    STRIP_KEYS,
};

const PI: f64 = std::f64::consts::PI;
const DEG_TO_RAD: f64 = PI / 180.0;

/// `_face_curve`: points from `origin + ua` to `origin + ub`.
fn face_curve(origin: V3, ua: V3, ub: V3, r: f64, segments: usize) -> Vec<V3> {
    let e = 2.0 / r;
    let mut pts = Vec::with_capacity(segments + 1);
    for k in 0..=segments {
        let (xa, xb) = if k == 0 {
            (0.0, 1.0)
        } else if k == segments {
            (1.0, 0.0)
        } else {
            let ang = PI / 2.0 * k as f64 / segments as f64;
            (ang.sin().powf(e), ang.cos().powf(e))
        };
        pts.push(origin.add(mulf(ua, 1.0 - xa)).add(mulf(ub, 1.0 - xb)));
    }
    pts
}

/// `_cap_point`: a point of the three-edge cap at barycentric `lambdas`.
fn cap_point(origin: V3, us: &[V3], lambdas: [f64; 3], r: f64) -> V3 {
    let s: Vec<f64> = lambdas.iter().map(|&lam| (lam * PI / 2.0).sin().powf(2.0)).collect();
    let total = py_sum(s.iter().copied(), PY_SUM_COMPENSATED);
    let mut out = origin;
    for (&u, &si) in us.iter().zip(s.iter()) {
        let x = py_max2(0.0, 1.0 - si / total).powf(1.0 / r);
        out = out.add(mulf(u, 1.0 - x));
    }
    out
}

type Key3 = (usize, usize, usize);

/// `plan_vertex_bevel`: bevel `vertices` by `size` along each of their
/// edges.
pub fn plan_vertex_bevel(
    mesh: &TopoMesh,
    vertices: &[usize],
    size: f64,
    segments: i64,
    profile: f64,
    h: Hypot,
) -> Result<FilletPlan, FilletError> {
    let mut verts: Vec<usize> = Vec::new();
    for &v in vertices {
        if !verts.contains(&v) {
            verts.push(v);
        }
    }
    if verts.is_empty() {
        return Err(FilletError::PickCorner);
    }
    if size <= 0.0 {
        return Err(FilletError::RadiusPositive);
    }
    if !(0.0 < profile && profile < 1.0) {
        return Err(FilletError::ProfileRange);
    }
    let segments = std::cmp::max(segments, 1) as usize;
    let r = exponent(profile);
    let mut plan = FilletPlan::new(size, segments, profile, Measure::Offset, Corners::Round, None);

    // The cut point on each (corner, edge) and the edge vector to it.
    let mut cut: HashMap<(usize, usize), V3> = HashMap::new();
    for &v in &verts {
        let vp = mesh.pos(v);
        if mesh.vertices[v].edges.is_empty() || mesh.vertex_faces(v).is_empty() {
            return Err(FilletError::OnlyCornerOfFace);
        }
        for &e in &mesh.vertices[v].edges {
            if mesh.edges[e].faces.is_empty() {
                return Err(FilletError::CornerEdgeNoFace(vp));
            }
            if mesh.edges[e].faces.len() > 2 {
                return Err(FilletError::TooManyFaces(vp));
            }
            let w = mesh.other(e, v);
            let need = size * if verts.contains(&w) { 2.0 } else { 1.0 };
            if need >= mesh.edge_length(e, h) - 1e-9 {
                return Err(FilletError::SizeTooLargeForEdge(vp));
            }
            let d = match unit(mesh.pos(w).sub(vp), h) {
                Some(d) => d,
                None => return Err(FilletError::ZeroLengthEdge),
            };
            cut.insert((v, e), mulf(d, size));
        }
    }

    // Each face's curve at each chosen corner, in the face's loop order.
    let mut curves: Vec<(Key3, Vec<V3>)> = Vec::new();
    let mut curve_idx: HashMap<Key3, usize> = HashMap::new();
    let mut touched: Vec<usize> = Vec::new();
    for &v in &verts {
        let vp = mesh.pos(v);
        for f in mesh.vertex_faces(v) {
            if !touched.contains(&f) {
                touched.push(f);
            }
            let face = &mesh.faces[f];
            let mut loops: Vec<&Vec<usize>> = vec![&face.loop_];
            loops.extend(face.holes.iter());
            for lp in loops {
                let m = lp.len();
                for (i, &x) in lp.iter().enumerate() {
                    if x != v {
                        continue;
                    }
                    let ep = mesh.find_edge(lp[(i + m - 1) % m], v);
                    let en = mesh.find_edge(v, lp[(i + 1) % m]);
                    let (ep, en) = match (ep, en) {
                        (Some(a), Some(b)) => (a, b),
                        _ => return Err(FilletError::UnreadableFaces),
                    };
                    let (ua, ub) = (cut[&(v, ep)], cut[&(v, en)]);
                    let cos_a = dotf(ua, ub) / (size * size);
                    if cos_a < -(MIN_ANGLE_DEG * DEG_TO_RAD).cos() {
                        return Err(FilletError::EdgesTooFlat(vp));
                    }
                    let k = (f, v, ep);
                    let pts = face_curve(vp, ua, ub, r, segments);
                    match curve_idx.get(&k) {
                        Some(&i) => curves[i].1 = pts,
                        None => {
                            curve_idx.insert(k, curves.len());
                            curves.push((k, pts));
                        }
                    }
                }
            }
        }
    }

    // Rebuild every face that loses a corner.
    let rebuilt_loop = |face: usize, lp: &[usize]| -> Vec<V3> {
        let mut out: Vec<V3> = Vec::new();
        let m = lp.len();
        for (i, &x) in lp.iter().enumerate() {
            if !verts.contains(&x) {
                out.push(mesh.pos(x));
                continue;
            }
            let ep = mesh.find_edge(lp[(i + m - 1) % m], x);
            // KeyError in Python when the edge is missing; checked above.
            if let Some(ep) = ep {
                if let Some(&ci) = curve_idx.get(&(face, x, ep)) {
                    out.extend(curves[ci].1.iter().copied());
                }
            }
        }
        let mut clean: Vec<V3> = Vec::new();
        for p in out {
            if clean.is_empty() || lenf(p.sub(clean[clean.len() - 1]), h) > 1e-7 {
                clean.push(p);
            }
        }
        if clean.len() > 1 && lenf(clean[0].sub(clean[clean.len() - 1]), h) <= 1e-7 {
            clean.pop();
        }
        clean
    };

    let mut attrs: Option<Vec<(String, String)>> = None;
    for &f in &touched {
        let face = &mesh.faces[f];
        let lp = rebuilt_loop(f, &face.loop_);
        let holes: Vec<Vec<V3>> = face.holes.iter().map(|hl| rebuilt_loop(f, hl)).collect();
        if lp.len() < 3 || holes.iter().any(|hl| hl.len() < 3) {
            return Err(FilletError::FaceCollapses);
        }
        plan.removed_faces.push(f);
        plan.rebuilt.push(Rebuilt { face: f, loop_: lp, holes });
        let keep: Vec<(String, String)> = STRIP_KEYS
            .iter()
            .filter_map(|k| face.paint.iter().find(|(kk, _)| kk == k).map(|(_, v)| (k.to_string(), v.clone())))
            .collect();
        attrs = Some(match attrs {
            None => keep,
            Some(a) => {
                if a == keep {
                    a
                } else {
                    Vec::new()
                }
            }
        });
    }
    plan.strip_attrs = Some(attrs.unwrap_or_default());

    // Close each solid corner.
    for &v in &verts {
        let vp = mesh.pos(v);
        let faces = mesh.vertex_faces(v);
        let closed = mesh.vertices[v].edges.iter().all(|&e| mesh.edges[e].faces.len() == 2);
        let face_curves: Vec<&Vec<V3>> = curves.iter().filter(|(k, _)| k.1 == v).map(|(_, p)| p).collect();
        if !closed {
            for pts in face_curves {
                if segments > 1 {
                    plan.curves.push(pts.clone());
                }
            }
            continue;
        }
        let vedges = &mesh.vertices[v].edges;
        if vedges.len() < 3 {
            return Err(FilletError::NothingToBevel(vp));
        }
        let mut out = zero();
        for &f in &faces {
            out = out.add(mesh.face_normal(f, h));
        }
        let tips: Vec<V3> = vedges.iter().map(|&e| vp.add(cut[&(v, e)])).collect();
        let mut centre = zero();
        for &p in &tips {
            centre = centre.add(p);
        }
        let centre = divf(centre, tips.len() as f64);
        let sign = if dotf(out, vp.sub(centre)) >= 0.0 { 1.0 } else { -1.0 };
        if vedges.len() == 3 {
            match three_edge_cap(mesh, v, &cut, &face_curves, r, segments, sign, h) {
                Some((cap, soft)) => {
                    plan.patches.extend(cap);
                    plan.soft.extend(soft);
                }
                None => return Err(FilletError::CouldNotClose(vp)),
            }
        } else {
            if segments > 1 {
                return Err(FilletError::MoreThanThree(vp));
            }
            match flat_polygon(&tips, out, h) {
                Some(poly) => plan.patches.push(facing(&poly, vp, sign)),
                None => return Err(FilletError::NotFlatEnough(vp)),
            }
        }
    }

    // Old edges go; the shortened ones keep their flags.
    let mut seen: Vec<usize> = Vec::new();
    for &v in &verts {
        for &e in &mesh.vertices[v].edges {
            if seen.contains(&e) {
                continue;
            }
            seen.push(e);
            plan.removed_edges.push(e);
            let (a, b) = (mesh.edges[e].v0, mesh.edges[e].v1);
            let pa = if verts.contains(&a) { mesh.pos(a).add(cut[&(a, e)]) } else { mesh.pos(a) };
            let pb = if verts.contains(&b) { mesh.pos(b).add(cut[&(b, e)]) } else { mesh.pos(b) };
            plan.flags.push((pa, pb, e));
        }
    }
    Ok(plan)
}

/// `_three_edge_cap`: (triangles, soft pairs), None when a side is missing.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn three_edge_cap(
    mesh: &TopoMesh,
    v: usize,
    cut: &HashMap<(usize, usize), V3>,
    face_curves: &[&Vec<V3>],
    r: f64,
    segments: usize,
    sign: f64,
    h: Hypot,
) -> Option<(Vec<Vec<V3>>, Vec<(V3, V3)>)> {
    let n = segments;
    let vp = mesh.pos(v);
    let edges = &mesh.vertices[v].edges;
    let us: Vec<V3> = edges.iter().map(|&e| cut[&(v, e)]).collect();
    let tips: Vec<V3> = us.iter().map(|&u| vp.add(u)).collect();
    let arc = |i: usize, j: usize| -> Option<Vec<V3>> {
        for pts in face_curves {
            if same(pts[0], tips[i], h) && same(pts[pts.len() - 1], tips[j], h) {
                return Some((*pts).clone());
            }
            if same(pts[pts.len() - 1], tips[i], h) && same(pts[0], tips[j], h) {
                return Some(pts.iter().rev().copied().collect());
            }
        }
        None
    };
    let (ab, bc, ac) = (arc(0, 1)?, arc(1, 2)?, arc(0, 2)?);
    let nf = n as f64;
    let mut grid: HashMap<(usize, usize), V3> = HashMap::new();
    for i in 0..=n {
        for j in 0..=(n - i) {
            let k = n - i - j;
            let p = if k == 0 {
                ab[j]
            } else if i == 0 {
                bc[k]
            } else if j == 0 {
                ac[k]
            } else {
                cap_point(vp, &us, [i as f64 / nf, j as f64 / nf, k as f64 / nf], r)
            };
            grid.insert((i, j), p);
        }
    }
    let mut tris = Vec::new();
    for i in 0..n {
        for j in 0..(n - i) {
            tris.push(facing(&[grid[&(i, j)], grid[&(i + 1, j)], grid[&(i, j + 1)]], vp, sign));
            if i + j + 1 < n {
                tris.push(facing(&[grid[&(i + 1, j)], grid[&(i + 1, j + 1)], grid[&(i, j + 1)]], vp, sign));
            }
        }
    }
    let mut soft = Vec::new();
    for i in 0..=n {
        for j in 0..=(n - i) {
            let p = grid[&(i, j)];
            for (di, dj) in [(1i64, 0i64), (0, 1), (-1, 1)] {
                let (ni, nj) = (i as i64 + di, j as i64 + dj);
                if ni < 0 || nj < 0 {
                    continue;
                }
                let q = match grid.get(&(ni as usize, nj as usize)) {
                    Some(&q) => q,
                    None => continue,
                };
                let (ni, nj) = (ni as usize, nj as usize);
                let ka = n as i64 - i as i64 - j as i64;
                let kb = n as i64 - ni as i64 - nj as i64;
                let along_side = (i == 0 && ni == 0) || (j == 0 && nj == 0) || (ka == 0 && kb == 0);
                if !along_side {
                    soft.push((p, q));
                }
            }
        }
    }
    Some((tris, soft))
}

/// `_flat_polygon`: `points` ordered around `normal`, None when not planar.
fn flat_polygon(points: &[V3], normal: V3, h: Hypot) -> Option<Vec<V3>> {
    let n = unit(normal, h)?;
    let mut c = zero();
    for &p in points {
        c = c.add(p);
    }
    let c = divf(c, points.len() as f64);
    let mut span = points.iter().fold(None, |m: Option<f64>, &p| {
        let l = lenf(p.sub(c), h);
        Some(match m {
            None => l,
            Some(m) => py_max2(m, l),
        })
    }).unwrap_or(0.0);
    if span == 0.0 {
        span = 1.0;
    }
    for &p in points {
        if dotf(p.sub(c), n).abs() > PLANE_TOL * 10.0 * py_max2(1.0, span) {
            return None;
        }
    }
    let reference = unit(points[0].sub(c), h)?;
    let side = n.cross(reference);
    let angle = |p: V3| {
        let d = p.sub(c);
        dotf(d, side).atan2(dotf(d, reference))
    };
    let mut keyed: Vec<(f64, V3)> = points.iter().map(|&p| (angle(p), p)).collect();
    keyed.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    Some(keyed.into_iter().map(|(_, p)| p).collect())
}

