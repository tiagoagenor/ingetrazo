// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! `core/arrangement.py`: the planar arrangement. Given segments in one
//! plane, rebuild the minimal bounded faces: project to the plane, split
//! every segment at its meetings with the others (`ingetrazo_geom::arrange`),
//! prune dangling chains, trace faces by the next-edge-clockwise rule, and
//! nest each hole in the smallest face that strictly contains it.
//!
//! Arithmetic as Python's: 2D coordinates are f64 (the f32 dot products of
//! `QVector3D` widened), `x ** 2` is the C runtime's `pow`, `math.hypot` is
//! CPython's own `vector_norm`, `sum()` is the 3.12 compensated sum.
//!
//! ORDER. Python keeps each vertex's neighbours in a `set` of integer
//! tuples. Its iteration order is a function of CPython's hash table, not
//! of the values, and is not reproduced: neighbours are kept in insertion
//! order. The set of faces and edges that come out does not depend on it;
//! the ORDER in which they are emitted (hence the order of `mesh.edges`,
//! `mesh.faces` and the arena ids a rebuild creates) can, when a vertex has
//! more than two neighbours. The L2 signature is order-free.

use std::collections::HashMap;
use std::hint::black_box;

use ingetrazo_geom::arrange::splits;
use ingetrazo_geom::sweep::py_hypot2;
use ingetrazo_math::pyfloat::{py_max2, py_min2, py_sum};
use ingetrazo_math::{Hypot, V3};

const TOL: f64 = 1e-4;

type P = [f64; 2];
type XY = (f64, f64);

/// `x ** 2` of the C runtime.
fn pow2(x: f64) -> f64 {
    x.powf(black_box(2.0))
}

/// `round(x / _TOL)`.
fn qkey(x: f64, y: f64) -> (i64, i64) {
    ((x / TOL).round_ties_even() as i64, (y / TOL).round_ties_even() as i64)
}

/// `_area_floor`.
fn area_floor(xy: &[XY]) -> f64 {
    let n = xy.len();
    let per = py_sum((0..n).map(|i| py_hypot2(xy[(i + 1) % n].0 - xy[i].0, xy[(i + 1) % n].1 - xy[i].1)), true);
    py_min2(TOL, TOL * per / 2.0)
}

/// `plane_basis`: two orthonormal in-plane axes for `normal`.
pub fn plane_basis(normal: V3, h: Hypot) -> (V3, V3) {
    let n = normal.normalized(h);
    let mut reference = V3(1.0, 0.0, 0.0);
    if (reference.dot(n) as f64).abs() > 0.9 {
        reference = V3(0.0, 1.0, 0.0);
    }
    let u = reference.sub(n.mul(reference.dot(n))).normalized(h);
    let v = n.cross(u).normalized(h);
    (u, v)
}

fn signed_area(l: &[XY]) -> f64 {
    let mut s = 0.0;
    let n = l.len();
    for i in 0..n {
        let (x0, y0) = l[i];
        let (x1, y1) = l[(i + 1) % n];
        s += x0 * y1 - x1 * y0;
    }
    s * 0.5
}

/// `_point_in_polygon`: ray-cast even-odd test.
pub fn point_in_polygon(p: XY, poly: &[XY]) -> bool {
    let (x, y) = p;
    let mut inside = false;
    let n = poly.len();
    let mut j = n.wrapping_sub(1);
    for i in 0..n {
        let (xi, yi) = poly[i];
        let (xj, yj) = poly[j];
        if ((yi > y) != (yj > y)) && (x < (xj - xi) * (y - yi) / (yj - yi + 1e-30) + xi) {
            inside = !inside;
        }
        j = i;
    }
    inside
}

/// The split edge graph: vertices in creation order, each with its
/// neighbours in insertion order.
struct Graph {
    coords: Vec<XY>,
    adj: Vec<Vec<usize>>,
}

/// `_build_graph`.
fn build_graph(segs: &[[XY; 2]]) -> Graph {
    let arr: Vec<[P; 2]> = segs.iter().map(|s| [[s[0].0, s[0].1], [s[1].0, s[1].1]]).collect();
    let (start, pts) = splits(&arr);
    let mut index: HashMap<(i64, i64), usize> = HashMap::new();
    let mut g = Graph { coords: Vec::new(), adj: Vec::new() };
    let mut vert = |p: P, g: &mut Graph| -> usize {
        let k = qkey(p[0], p[1]);
        *index.entry(k).or_insert_with(|| {
            g.coords.push((p[0], p[1]));
            g.adj.push(Vec::new());
            g.coords.len() - 1
        })
    };
    for i in 0..segs.len() {
        let row = &pts[start[i] as usize..start[i + 1] as usize];
        for k in 0..row.len().saturating_sub(1) {
            let ka = vert(row[k], &mut g);
            let kb = vert(row[k + 1], &mut g);
            if ka != kb {
                if !g.adj[ka].contains(&kb) {
                    g.adj[ka].push(kb);
                }
                if !g.adj[kb].contains(&ka) {
                    g.adj[kb].push(ka);
                }
            }
        }
    }
    g
}

/// `_prune_dangling`: drop degree-1 vertices until none is left. `alive`
/// marks the vertices still in the dict.
fn prune_dangling(adj: &mut [Vec<usize>], alive: &mut [bool]) {
    let mut changed = true;
    while changed {
        changed = false;
        for k in 0..adj.len() {
            if !alive[k] {
                continue;
            }
            if adj[k].len() == 1 {
                let other = adj[k][0];
                adj[other].retain(|&x| x != k);
                alive[k] = false;
                adj[k].clear();
                changed = true;
            } else if adj[k].is_empty() {
                alive[k] = false;
                changed = true;
            }
        }
    }
}

/// `_trace_faces`: every face as a vertex loop by the next-edge-clockwise
/// rule. Bounded faces come out CCW; the unbounded one CW.
fn trace_faces(coords: &[XY], adj: &[Vec<usize>], alive: &[bool]) -> Vec<Vec<usize>> {
    let mut order: Vec<Vec<usize>> = vec![Vec::new(); adj.len()];
    for k in 0..adj.len() {
        if !alive[k] {
            continue;
        }
        let (kx, ky) = coords[k];
        let mut ring = adj[k].clone();
        ring.sort_by(|&a, &b| {
            let ta = (coords[a].1 - ky).atan2(coords[a].0 - kx);
            let tb = (coords[b].1 - ky).atan2(coords[b].0 - kx);
            ta.partial_cmp(&tb).unwrap_or(std::cmp::Ordering::Equal)
        });
        order[k] = ring;
    }
    let next_he = |u: usize, v: usize| -> usize {
        let ring = &order[v];
        let idx = ring.iter().position(|&x| x == u).expect("u is a neighbour of v");
        ring[(idx + ring.len() - 1) % ring.len()]
    };
    let mut visited: std::collections::HashSet<(usize, usize)> = std::collections::HashSet::new();
    let mut loops = Vec::new();
    for u in 0..adj.len() {
        if !alive[u] {
            continue;
        }
        for &v in &adj[u] {
            if visited.contains(&(u, v)) {
                continue;
            }
            let mut lp = Vec::new();
            let (mut cu, mut cv) = (u, v);
            while !visited.contains(&(cu, cv)) {
                visited.insert((cu, cv));
                lp.push(cu);
                let nxt = next_he(cu, cv);
                cu = cv;
                cv = nxt;
                if lp.len() > 100000 {
                    break;
                }
            }
            if lp.len() >= 3 {
                loops.push(lp);
            }
        }
    }
    loops
}

/// `_interior_point`: a point safely inside the polygon.
fn interior_point(lp: &[XY]) -> XY {
    let n = lp.len() as f64;
    let cx = py_sum(lp.iter().map(|p| p.0), true) / n;
    let cy = py_sum(lp.iter().map(|p| p.1), true) / n;
    if point_in_polygon((cx, cy), lp) {
        return (cx, cy);
    }
    let (ax, ay) = lp[0];
    let (bx, by) = lp[1];
    let (mx, my) = ((ax + bx) / 2.0, (ay + by) / 2.0);
    let p = (mx * 0.999 + cx * 0.001, my * 0.999 + cy * 0.001);
    if point_in_polygon(p, lp) {
        return p;
    }
    scan_interior_point(lp, &[]).unwrap_or(p)
}

/// `scan_interior_point`: a point well inside `outer` and outside every
/// hole, or `None`.
pub fn scan_interior_point(outer: &[XY], holes: &[Vec<XY>]) -> Option<XY> {
    let mut loops: Vec<&[XY]> = vec![outer];
    loops.extend(holes.iter().map(|h| h.as_slice()));
    let mut ys: Vec<f64> = loops.iter().flat_map(|lp| lp.iter().map(|p| p.1)).collect();
    ys.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    ys.dedup();
    let mut best: Option<(f64, XY)> = None;
    for w in ys.windows(2) {
        let (y0, y1) = (w[0], w[1]);
        if y1 - y0 < 1e-9 {
            continue;
        }
        let y = (y0 + y1) / 2.0;
        let mut xs: Vec<f64> = Vec::new();
        for lp in &loops {
            let n = lp.len();
            for i in 0..n {
                let (ax, ay) = lp[i];
                let (bx, by) = lp[(i + 1) % n];
                if (ay > y) != (by > y) {
                    xs.push(ax + (y - ay) * (bx - ax) / (by - ay));
                }
            }
        }
        xs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let mut i = 0;
        while i + 1 < xs.len() {
            let (a, b) = (xs[i], xs[i + 1]);
            let w = py_min2(b - a, y1 - y0);
            if best.is_none() || w > best.unwrap().0 {
                best = Some((w, ((a + b) / 2.0, y)));
            }
            i += 2;
        }
    }
    best.map(|b| b.1)
}

/// A face in plane coordinates: its outer loop and its holes.
pub type FaceXY = (Vec<XY>, Vec<Vec<XY>>);

/// `_classify`: faces (CCW loops) with the CW loops nested in them as holes.
fn classify(loops: &[Vec<usize>], coords: &[XY]) -> Vec<FaceXY> {
    let mut faces: Vec<(Vec<XY>, f64)> = Vec::new();
    let mut holes: Vec<(Vec<XY>, f64)> = Vec::new();
    for lp in loops {
        let xy: Vec<XY> = lp.iter().map(|&k| coords[k]).collect();
        let area = signed_area(&xy);
        let floor = area_floor(&xy);
        if area > floor {
            faces.push((xy, area));
        } else if area < -floor {
            holes.push((xy, -area));
        }
    }
    if faces.is_empty() {
        return Vec::new();
    }
    if !holes.is_empty() {
        // `sort(reverse=True)` is stable: equal areas keep their order.
        holes.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        holes.remove(0);
    }
    faces.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    let mut face_holes: Vec<Vec<Vec<XY>>> = vec![Vec::new(); faces.len()];
    for (hxy, harea) in &holes {
        let hp = interior_point(hxy);
        for (fi, (fxy, farea)) in faces.iter().enumerate() {
            if *farea > *harea + area_floor(hxy) && point_in_polygon(hp, fxy) {
                face_holes[fi].push(hxy.clone());
                break;
            }
        }
    }
    faces.into_iter().zip(face_holes).map(|((f, _), hs)| (f, hs)).collect()
}

/// What `_arrange` hands back: the vertex coordinates and the full split
/// graph (before pruning), the faces in plane coordinates, and the map back
/// to 3D.
pub struct Arranged {
    coords: Vec<XY>,
    adj: Vec<Vec<usize>>,
    pub faces: Vec<FaceXY>,
    origin: V3,
    u: V3,
    v: V3,
}

impl Arranged {
    pub fn to3d(&self, xy: XY) -> V3 {
        self.origin.add(self.u.mul(xy.0 as f32)).add(self.v.mul(xy.1 as f32))
    }
}

/// `_arrange`.
pub fn arrange(segments: &[(V3, V3)], origin: V3, normal: V3, h: Hypot) -> Arranged {
    let (u, v) = plane_basis(normal, h);
    let to2d = |p: V3| -> XY {
        let d = p.sub(origin);
        (d.dot(u) as f64, d.dot(v) as f64)
    };
    let mut segs: Vec<[XY; 2]> = Vec::new();
    for &(a, b) in segments {
        let (pa, pb) = (to2d(a), to2d(b));
        if pow2(pa.0 - pb.0) + pow2(pa.1 - pb.1) > TOL * TOL {
            segs.push([pa, pb]);
        }
    }
    if segs.is_empty() {
        return Arranged { coords: Vec::new(), adj: Vec::new(), faces: Vec::new(), origin, u, v };
    }
    let g = build_graph(&segs);
    let mut adj = g.adj.clone();
    let mut alive = vec![true; adj.len()];
    prune_dangling(&mut adj, &mut alive);
    let loops = trace_faces(&g.coords, &adj, &alive);
    let faces = classify(&loops, &g.coords);
    Arranged { coords: g.coords, adj: g.adj, faces, origin, u, v }
}

/// `planar_rebuild`: the complete split edge set (so nothing is lost,
/// spurs included) and the faces as `(outer, holes)` in 3D.
pub fn planar_rebuild(segments: &[(V3, V3)], origin: V3, normal: V3, h: Hypot) -> (Vec<(V3, V3)>, Vec<(Vec<V3>, Vec<Vec<V3>>)>) {
    let a = arrange(segments, origin, normal, h);
    let mut edges = Vec::new();
    let mut seen: std::collections::HashSet<(usize, usize)> = std::collections::HashSet::new();
    for k in 0..a.adj.len() {
        for &m in &a.adj[k] {
            let key = if k <= m { (k, m) } else { (m, k) };
            if !seen.insert(key) {
                continue;
            }
            edges.push((a.to3d(a.coords[k]), a.to3d(a.coords[m])));
        }
    }
    let faces = a
        .faces
        .iter()
        .map(|(outer, holes)| {
            (outer.iter().map(|&p| a.to3d(p)).collect(), holes.iter().map(|hl| hl.iter().map(|&p| a.to3d(p)).collect()).collect())
        })
        .collect();
    (edges, faces)
}

/// `coplanar_plane`: `(origin, normal)` of the plane through `points` if
/// they are all coplanar, else `None`. Collinear or degenerate input gives
/// a horizontal plane.
pub fn coplanar_plane(points: &[V3], tol: f64, h: Hypot) -> Option<(V3, V3)> {
    if points.len() < 3 {
        return Some((points.first().copied().unwrap_or(V3(0.0, 0.0, 0.0)), V3(0.0, 0.0, 1.0)));
    }
    let o = points[0];
    let mut d1: Option<V3> = None;
    for &p in &points[1..] {
        if (p.sub(o).length(h) as f64) > 1e-6 {
            d1 = Some(p.sub(o).normalized(h));
            break;
        }
    }
    let Some(d1) = d1 else { return Some((o, V3(0.0, 0.0, 1.0))) };
    let mut n: Option<V3> = None;
    for &p in &points[1..] {
        let c = d1.cross(p.sub(o));
        if (c.length(h) as f64) > 1e-6 {
            n = Some(c.normalized(h));
            break;
        }
    }
    let Some(n) = n else { return Some((o, V3(0.0, 0.0, 1.0))) };
    for &p in points {
        if (n.dot(p.sub(o)) as f64).abs() > tol {
            return None;
        }
    }
    Some((o, n))
}

#[allow(dead_code)]
fn _unused(_: f64) -> f64 {
    py_max2(0.0, 0.0)
}
