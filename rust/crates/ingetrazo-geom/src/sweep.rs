// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! `core.sweep` (Follow Me): the geometry of a profile carried along a path,
//! from plain arrays, no mesh.
//!
//! What is here, op for op with the Python: [`order_path_edges`], the joint
//! planes and the sliding of every ring onto them ([`sweep_rings`]), the
//! lathe case ([`revolution_rings`] with its Sutherland-Hodgman half clip),
//! the wall quads and caps ([`sweep_preview_faces`], and [`sweep_plan`] — the
//! faces `sweep_profile` hands `Mesh.add_face`, in its order), the dragged
//! path ([`manual_path_start`], [`manual_path_extend`], `_route`) and
//! [`orient_closed_path`].
//!
//! What is not (M3, it needs the Rust mesh): consuming the profile face,
//! `Mesh.add_face` itself, `run_stitch`, softening/`orient_outward`. The plan
//! carries everything those steps are fed: the faces to add (walls, then caps),
//! the profile's boundary segments to clear, the `_key` seed set and the
//! dihedral test ([`curve_facet_soft`]).
//!
//! Float discipline: `QVector3D` is f32 (`V3`); a Python float multiplied into
//! one is rounded to f32 first; `math.*` and tuple arithmetic are f64.

use std::collections::{HashMap, VecDeque};

use ingetrazo_math::pyfloat::{py_max2, py_max3, py_min2, py_sum};
use ingetrazo_math::{Hypot, V3};

use crate::meshface::{centroid, key, normal};

/// `_CURVE_FACET_COS`: a seam whose dihedral cosine is above this reads as a
/// curve facet and is softened.
pub const CURVE_FACET_COS: f64 = 0.85;

/// The profile face: outer loop and holes as positions (`face.vertices`,
/// `face.holes`).
#[derive(Clone, Debug)]
pub struct Profile {
    pub outer: Vec<V3>,
    pub holes: Vec<Vec<V3>>,
}

/// Rings of a sweep: `rings[station][loop][corner]`, loop 0 the outer one.
pub type Ring = Vec<Vec<V3>>;

pub struct Rings {
    pub rings: Vec<Ring>,
    pub dirs: Vec<V3>,
    pub spans: usize,
}

/// CPython 3.12 `math.hypot(a, b)` (`vector_norm`: scaled, double-double
/// squares, one differential correction), which is not libm's hypot.
pub fn py_hypot2(a: f64, b: f64) -> f64 {
    let vec = [a.abs(), b.abs()];
    let found_nan = a.is_nan() || b.is_nan();
    let max = py_max2(vec[0], vec[1]);
    if max.is_infinite() {
        return max;
    }
    if found_nan {
        return f64::NAN;
    }
    vector_norm(&vec, max)
}

fn dl_mul(x: f64, y: f64) -> (f64, f64) {
    let z = x * y;
    (z, x.mul_add(y, -z))
}

fn dl_fast_sum(a: f64, b: f64) -> (f64, f64) {
    let hi = a + b;
    (hi, (a - hi) + b)
}

fn frexp_exp(x: f64) -> i32 {
    // The exponent e of x = m · 2^e with 0.5 <= |m| < 1 (x finite, nonzero).
    let bits = x.to_bits();
    let ex = ((bits >> 52) & 0x7ff) as i32;
    if ex == 0 {
        // subnormal
        let m = bits & 0x000f_ffff_ffff_ffff;
        let lz = m.leading_zeros() as i32 - 12;
        return -1022 - lz;
    }
    ex - 1022
}

fn vector_norm(vec: &[f64], max: f64) -> f64 {
    if max == 0.0 || vec.len() <= 1 {
        return max;
    }
    let max_e = frexp_exp(max);
    if max_e < -1023 {
        let dbl_min = f64::MIN_POSITIVE;
        let v: Vec<f64> = vec.iter().map(|x| x / dbl_min).collect();
        return dbl_min * vector_norm(&v, max / dbl_min);
    }
    let scale = f64::from_bits(((1023 - max_e) as u64) << 52); // ldexp(1.0, -max_e)
    let (mut csum, mut frac1, mut frac2) = (1.0f64, 0.0f64, 0.0f64);
    for &x in vec {
        let x = x * scale;
        let pr = dl_mul(x, x);
        let sm = dl_fast_sum(csum, pr.0);
        csum = sm.0;
        frac1 += pr.1;
        frac2 += sm.1;
    }
    let mut h = (csum - 1.0 + (frac1 + frac2)).sqrt();
    let pr = dl_mul(-h, h);
    let sm = dl_fast_sum(csum, pr.0);
    csum = sm.0;
    frac1 += pr.1;
    frac2 += sm.1;
    let x = csum - 1.0 + (frac1 + frac2);
    h += x / (2.0 * h);
    h / scale
}

fn f(x: f32) -> f64 {
    x as f64
}

/// `order_path_edges`: the selected edges (pairs of vertex ids; `pos` maps an
/// id to its position) chained into one polyline. `(points, closed)`, or
/// `None` for branching, disjoint pieces, or a chain that does not close.
pub fn order_path_edges(edges: &[(usize, usize)], pos: &[V3]) -> Option<(Vec<V3>, bool)> {
    if edges.is_empty() {
        return None;
    }
    // adj: a dict, insertion-ordered by first appearance of the vertex.
    let mut slot: HashMap<usize, usize> = HashMap::new();
    let mut order: Vec<usize> = Vec::new();
    let mut nbrs: Vec<Vec<usize>> = Vec::new();
    let mut push = |a: usize, b: usize| {
        let s = *slot.entry(a).or_insert_with(|| {
            order.push(a);
            nbrs.push(Vec::new());
            order.len() - 1
        });
        nbrs[s].push(b);
    };
    for &(v0, v1) in edges {
        push(v0, v1);
        push(v1, v0);
    }
    if nbrs.iter().any(|n| n.len() > 2) {
        return None;
    }
    let ends: Vec<usize> = (0..order.len()).filter(|&s| nbrs[s].len() == 1).map(|s| order[s]).collect();
    if !ends.is_empty() && ends.len() != 2 {
        return None;
    }
    let closed = ends.is_empty();
    let start = if closed { edges[0].0 } else { ends[0] };
    let mut pts = vec![pos[start]];
    let mut prev: Option<usize> = None;
    let mut cur = start;
    for _ in 0..edges.len() {
        let nxt = nbrs[slot[&cur]].iter().copied().find(|&v| Some(v) != prev)?;
        pts.push(pos[nxt]);
        prev = Some(cur);
        cur = nxt;
    }
    if closed {
        if cur != start {
            return None;
        }
        pts.pop();
    } else if pts.len() != edges.len() + 1 {
        return None;
    }
    Some((pts, closed))
}

/// `_project_ring`: each point slid parallel to `direction` onto the plane.
fn project_ring(points: &[V3], direction: V3, plane_pt: V3, plane_n: V3) -> Option<Vec<V3>> {
    let denom_dir = direction.dot(plane_n);
    if f(denom_dir).abs() < 1e-9 {
        return None;
    }
    let mut out = Vec::with_capacity(points.len());
    for &p in points {
        let t = f(plane_pt.sub(p).dot(plane_n)) / f(denom_dir);
        out.push(p.add(direction.mul(t as f32)));
    }
    Some(out)
}

/// `_wall`: a quad without its repeated corners, `None` under three.
pub fn wall(quad: &[V3], h: Hypot) -> Option<Vec<V3>> {
    let mut out: Vec<V3> = Vec::new();
    for &p in quad {
        if out.is_empty() || f(p.sub(*out.last().unwrap()).length(h)) >= 1e-9 {
            out.push(p);
        }
    }
    if out.len() > 1 && f(out[0].sub(*out.last().unwrap()).length(h)) < 1e-9 {
        out.pop();
    }
    if out.len() >= 3 {
        Some(out)
    } else {
        None
    }
}

type P3 = (f64, f64, f64);

/// `_clip_half`: the x >= 0 half of a closed loop, Sutherland-Hodgman.
fn clip_half(loop_: &[P3], keep_tol: f64) -> Option<Vec<P3>> {
    let mut out: Vec<P3> = Vec::new();
    let m = loop_.len();
    for i in 0..m {
        let (cur, nxt) = (loop_[i], loop_[(i + 1) % m]);
        let cin = cur.1 >= -keep_tol;
        let nin = nxt.1 >= -keep_tol;
        if cin {
            out.push(if cur.1 > keep_tol {
                (cur.0, py_max2(cur.1, 0.0), cur.2)
            } else {
                (cur.0, 0.0, 0.0)
            });
        }
        if cin != nin {
            let t = cur.1 / (cur.1 - nxt.1);
            out.push((cur.0 + (nxt.0 - cur.0) * t, 0.0, 0.0));
        }
    }
    let differs = |a: &P3, b: &P3| (a.0 - b.0).abs() > 1e-12 || (a.1 - b.1).abs() > 1e-12 || (a.2 - b.2).abs() > 1e-12;
    let mut clean: Vec<P3> = Vec::new();
    for q in out {
        if clean.is_empty() || differs(&q, clean.last().unwrap()) {
            clean.push(q);
        }
    }
    while clean.len() > 1 && !differs(&clean[0], clean.last().unwrap()) {
        clean.pop();
    }
    if clean.len() >= 3 {
        Some(clean)
    } else {
        None
    }
}

fn dot3(u: [f64; 3], v: [f64; 3]) -> f64 {
    u[0] * v[0] + u[1] * v[1] + u[2] * v[2]
}

fn cross3(u: [f64; 3], v: [f64; 3]) -> [f64; 3] {
    [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]]
}

/// `_revolution_rings`: the profile turned about the axis of a circular
/// closed path (a lathe), or `None`.
pub fn revolution_rings(face: &Profile, path: &[V3], h: Hypot) -> Option<Rings> {
    let n = path.len();
    if n < 3 {
        return None;
    }
    let pts: Vec<[f64; 3]> = path.iter().map(|p| [f(p.0), f(p.1), f(p.2)]).collect();
    let nf = n as f64;
    let c: [f64; 3] = [
        py_sum(pts.iter().map(|p| p[0]), true) / nf,
        py_sum(pts.iter().map(|p| p[1]), true) / nf,
        py_sum(pts.iter().map(|p| p[2]), true) / nf,
    ];
    let mut ax = [0.0f64; 3];
    for i in 0..n {
        let (a, b) = (pts[i], pts[(i + 1) % n]);
        ax[0] += (a[1] - b[1]) * (a[2] + b[2]);
        ax[1] += (a[2] - b[2]) * (a[0] + b[0]);
        ax[2] += (a[0] - b[0]) * (a[1] + b[1]);
    }
    let sq3 = |v: [f64; 3]| py_sum([v[0] * v[0], v[1] * v[1], v[2] * v[2]].into_iter(), true);
    let la = sq3(ax).sqrt();
    if la < 1e-12 {
        return None;
    }
    let ax = [ax[0] / la, ax[1] / la, ax[2] / la];
    let rel: Vec<[f64; 3]> = pts.iter().map(|p| [p[0] - c[0], p[1] - c[1], p[2] - c[2]]).collect();
    let radii: Vec<f64> = rel.iter().map(|r| sq3(*r).sqrt()).collect();
    let rmean = py_sum(radii.iter().copied(), true) / nf;
    if rmean < 1e-9 {
        return None;
    }
    let tol = py_max2(1e-4, 1e-3 * rmean);
    if rel.iter().any(|r| dot3(*r, ax).abs() > tol) {
        return None;
    }
    let rmax = radii.iter().copied().fold(radii[0], py_max2);
    let rmin = radii.iter().copied().fold(radii[0], py_min2);
    if rmax - rmin > tol + 1e-2 * rmean {
        return None;
    }
    let fnv = normal(&face.outer, h);
    let fnv = [f(fnv.0), f(fnv.1), f(fnv.2)];
    let lf = dot3(fnv, fnv).sqrt();
    if lf < 1e-12 {
        return None;
    }
    let fnv = [fnv[0] / lf, fnv[1] / lf, fnv[2] / lf];
    if dot3(fnv, ax).abs() > 1e-3 {
        return None;
    }
    let v0 = face.outer[0];
    if dot3([f(v0.0) - c[0], f(v0.1) - c[1], f(v0.2) - c[2]], fnv).abs() > tol {
        return None;
    }
    let mut radial = cross3(ax, fnv);
    let lr = dot3(radial, radial).sqrt();
    radial = [radial[0] / lr, radial[1] / lr, radial[2] / lr];
    let fc = centroid(&face.outer);
    if dot3([f(fc.0) - c[0], f(fc.1) - c[1], f(fc.2) - c[2]], radial) < 0.0 {
        radial = [-radial[0], -radial[1], -radial[2]];
    }
    let side = cross3(ax, radial);

    let local = |p: V3| -> P3 {
        let r = [f(p.0) - c[0], f(p.1) - c[1], f(p.2) - c[2]];
        (dot3(r, ax), dot3(r, radial), dot3(r, side))
    };
    // size = max(max(abs(q) for q in local(v)) for v in face.vertices) or 1.0
    let mut size: Option<f64> = None;
    for &v in &face.outer {
        let q = local(v);
        let m = py_max3(q.0.abs(), q.1.abs(), q.2.abs());
        size = Some(match size {
            None => m,
            Some(s) => py_max2(s, m),
        });
    }
    let size = match size {
        Some(s) if s != 0.0 => s,
        _ => 1.0,
    };
    let keep_tol = py_max2(1e-7, 1e-6 * size);
    let outer: Vec<P3> = face.outer.iter().map(|&v| local(v)).collect();
    let min_x = outer.iter().map(|q| q.1).fold(outer[0].1, py_min2);
    let outer: Vec<P3> = if min_x < -keep_tol {
        clip_half(&outer, keep_tol)?
    } else {
        outer
            .iter()
            .map(|q| if q.1.abs() > keep_tol { (q.0, q.1, 0.0) } else { (q.0, 0.0, 0.0) })
            .collect()
    };
    let mut loops: Vec<Vec<P3>> = vec![outer];
    for hole in &face.holes {
        let hl: Vec<P3> = hole.iter().map(|&v| local(v)).collect();
        let xmin = hl.iter().map(|q| q.1).fold(hl[0].1, py_min2);
        let xmax = hl.iter().map(|q| q.1).fold(hl[0].1, py_max2);
        if xmin >= -keep_tol {
            loops.push(hl);
        } else if xmax > keep_tol {
            return None;
        }
    }

    let mut rings: Vec<Ring> = Vec::with_capacity(n);
    for r in &rel {
        let mut ca = dot3(*r, radial);
        let mut sa = dot3(*r, side);
        let hy = py_hypot2(ca, sa);
        let norm = if hy != 0.0 { hy } else { 1.0 };
        ca /= norm;
        sa /= norm;
        let mut ring: Ring = Vec::with_capacity(loops.len());
        for lp in &loops {
            let mut pl = Vec::with_capacity(lp.len());
            for &(hh, x, y) in lp {
                let comp = |k: usize| -> f32 {
                    (c[k] + ax[k] * hh + (radial[k] * ca + side[k] * sa) * x + (side[k] * ca - radial[k] * sa) * y)
                        as f32
                };
                pl.push(V3(comp(0), comp(1), comp(2)));
            }
            ring.push(pl);
        }
        rings.push(ring);
    }
    let mut dirs = Vec::with_capacity(n);
    for i in 0..n {
        let d = path[(i + 1) % n].sub(path[i]);
        if f(d.length(h)) < 1e-9 {
            return None;
        }
        dirs.push(d.normalized(h));
    }
    Some(Rings { rings, dirs, spans: n })
}

/// `sweep_rings`: the profile's rings at every path station, or `None` on a
/// 180° reversal, a segment parallel to a joint plane or a zero-length span.
pub fn sweep_rings(face: &Profile, path: &[V3], closed: bool, h: Hypot) -> Option<Rings> {
    if closed {
        if let Some(rev) = revolution_rings(face, path, h) {
            return Some(rev);
        }
    }
    let n = path.len();
    let spans = if closed { n } else { n.wrapping_sub(1) };
    if n == 0 || spans < 1 {
        return None;
    }
    let mut dirs = Vec::with_capacity(spans);
    for i in 0..spans {
        let d = path[(i + 1) % n].sub(path[i]);
        if f(d.length(h)) < 1e-9 {
            return None;
        }
        dirs.push(d.normalized(h));
    }
    let joint_normal = |i: usize| -> Option<V3> {
        let (a, b);
        if closed {
            a = dirs[(i + spans - 1) % spans];
            b = dirs[i % spans];
        } else if i == 0 {
            return Some(dirs[0]);
        } else if i >= spans {
            return Some(*dirs.last().unwrap());
        } else {
            a = dirs[i - 1];
            b = dirs[i];
        }
        let s = a.add(b);
        if f(s.length(h)) < 1e-9 {
            return None;
        }
        Some(s.normalized(h))
    };
    let mut loops: Vec<Vec<V3>> = vec![face.outer.clone()];
    loops.extend(face.holes.iter().cloned());
    let mut rings: Vec<Ring> = Vec::with_capacity(n);
    for i in 0..n {
        let pn = joint_normal(i)?;
        let anchor = path[i % n];
        let mut ring: Ring = Vec::with_capacity(loops.len());
        if i == 0 {
            for lp in &loops {
                ring.push(project_ring(lp, dirs[0], anchor, pn)?);
            }
        } else {
            for prev_lp in rings.last().unwrap() {
                ring.push(project_ring(prev_lp, dirs[i - 1], anchor, pn)?);
            }
        }
        rings.push(ring);
    }
    Some(Rings { rings, dirs, spans })
}

/// One polygon the sweep adds: outer loop and holes.
#[derive(Clone, Debug, PartialEq)]
pub struct Poly {
    pub outer: Vec<V3>,
    pub holes: Vec<Vec<V3>>,
}

/// The wall quads of every span, in `sweep_profile`'s order.
fn walls(r: &Rings, closed: bool, h: Hypot) -> Vec<Vec<V3>> {
    let stations = r.rings.len();
    let mut out = Vec::new();
    for s in 0..r.spans {
        let r0 = &r.rings[s];
        let r1 = if closed { &r.rings[(s + 1) % stations] } else { &r.rings[s + 1] };
        for (lp0, lp1) in r0.iter().zip(r1.iter()) {
            let m = lp0.len();
            for j in 0..m {
                let (a, b) = (lp0[j], lp0[(j + 1) % m]);
                let (b2, a2) = (lp1[(j + 1) % m], lp1[j]);
                if let Some(q) = wall(&[a, b, b2, a2], h) {
                    out.push(q);
                }
            }
        }
    }
    out
}

/// `sweep_preview_faces`: the walls of every span and, on an open path, the
/// far cap.
pub fn sweep_preview_faces(face: &Profile, path: &[V3], closed: bool, h: Hypot) -> Vec<Poly> {
    let Some(r) = sweep_rings(face, path, closed, h) else {
        return Vec::new();
    };
    let mut out: Vec<Poly> = walls(&r, closed, h).into_iter().map(|q| Poly { outer: q, holes: vec![] }).collect();
    if !closed {
        let last = r.rings.last().unwrap();
        out.push(Poly { outer: last[0].clone(), holes: last[1..].to_vec() });
    }
    out
}

/// What `sweep_profile` does to the mesh, as data.
#[derive(Clone, Debug)]
pub struct SweepPlan {
    /// Faces for `Mesh.add_face`, in order: every wall, then (open path) the
    /// start cap (reversed loops) and the end cap.
    pub faces: Vec<Poly>,
    /// How many of `faces` are walls.
    pub wall_count: usize,
    /// The profile's boundary segments (outer loop, then holes), each as the
    /// two positions of `lp[i]`, `lp[(i+1) % n]`: after the walls are added,
    /// each that is an edge bounding no face is removed.
    pub profile_segments: Vec<(V3, V3)>,
    /// `{_key(p) for ring in rings for lp in ring for p in lp}` — the stitch
    /// seed; Python's set, here in first-seen order without repeats.
    pub seed: Vec<[f64; 3]>,
    pub dirs: Vec<V3>,
}

/// The geometry of `sweep_profile`, or `None` where it returns False.
pub fn sweep_plan(face: &Profile, path: &[V3], closed: bool, h: Hypot) -> Option<SweepPlan> {
    let r = sweep_rings(face, path, closed, h)?;
    let mut profile_segments = Vec::new();
    for lp in std::iter::once(&face.outer).chain(face.holes.iter()) {
        let n = lp.len();
        for i in 0..n {
            profile_segments.push((lp[i], lp[(i + 1) % n]));
        }
    }
    let quads = walls(&r, closed, h);
    let wall_count = quads.len();
    let mut faces: Vec<Poly> = quads.into_iter().map(|q| Poly { outer: q, holes: vec![] }).collect();
    if !closed {
        let first = &r.rings[0];
        faces.push(Poly {
            outer: first[0].iter().rev().copied().collect(),
            holes: first[1..].iter().map(|hl| hl.iter().rev().copied().collect()).collect(),
        });
        let last = r.rings.last().unwrap();
        faces.push(Poly { outer: last[0].clone(), holes: last[1..].to_vec() });
    }
    let mut seen: std::collections::HashSet<[u64; 3]> = std::collections::HashSet::new();
    let mut seed = Vec::new();
    for ring in &r.rings {
        for lp in ring {
            for &p in lp {
                let k = key(p);
                if seen.insert(k.map(f64::to_bits)) {
                    seed.push(k);
                }
            }
        }
    }
    Some(SweepPlan { faces, wall_count, profile_segments, seed, dirs: r.dirs })
}

/// The softening test of `sweep_profile`'s last loop: two unit face normals
/// meeting at a shallow dihedral (`_CURVE_FACET_COS < d < 0.99995`).
pub fn curve_facet_soft(n0: V3, n1: V3, h: Hypot) -> bool {
    let d = f(n0.normalized(h).dot(n1.normalized(h)));
    CURVE_FACET_COS < d && d < 0.99995
}

// ---- Manual (dragged) paths ------------------------------------------------

/// `manual_path_start`. The edge is `(a, b)` with ids and positions; returns
/// `(start point, vertex ids after it)`.
pub fn manual_path_start(
    face: &Profile,
    a: (usize, V3),
    b: (usize, V3),
    toward: Option<V3>,
    h: Hypot,
) -> (Option<V3>, Vec<usize>) {
    let c = centroid(&face.outer);
    let n = normal(&face.outer, h).normalized(h);
    let da = f(a.1.sub(c).dot(n));
    let db = f(b.1.sub(c).dot(n));
    if da * db < -1e-12 {
        let t = da / (da - db);
        let p = a.1.add(b.1.sub(a.1).mul(t as f32));
        let mut far = b.0;
        if let Some(tw) = toward {
            if f(a.1.sub(tw).length(h)) < f(b.1.sub(tw).length(h)) {
                far = a.0;
            }
        }
        return (Some(p), vec![far]);
    }
    let (near, far) = if da.abs() <= db.abs() { (a.0, b.0) } else { (b.0, a.0) };
    (None, vec![near, far])
}

/// The edges of a mesh as `manual_path_extend` reads them: endpoints of each
/// edge and, per vertex, its incident edges in the order Python iterates
/// `v.edges` (a set — id-ordered there; the order matters only when two
/// shortest routes tie).
pub struct PathGraph {
    pub edges: Vec<(usize, usize)>,
    pub vedges: Vec<Vec<usize>>,
}

/// `_route`: the shortest run of edges (breadth first) from `start` to either
/// target, never through vertices on the chain.
fn route(
    g: &PathGraph,
    start: usize,
    targets: (usize, usize),
    skip: &dyn Fn(usize) -> bool,
    blocked: &HashMap<usize, usize>,
    limit: usize,
) -> Option<Vec<usize>> {
    let mut prev: HashMap<usize, Option<usize>> = HashMap::new();
    prev.insert(start, None);
    let mut q: VecDeque<usize> = VecDeque::new();
    q.push_back(start);
    let mut hit: Option<usize> = None;
    let mut steps = 0usize;
    while !q.is_empty() && steps < limit && hit.is_none() {
        let v = q.pop_front().unwrap();
        steps += 1;
        for &e in &g.vedges[v] {
            if skip(e) {
                continue;
            }
            let (e0, e1) = g.edges[e];
            let w = if e0 == v { e1 } else { e0 };
            if prev.contains_key(&w) || (blocked.contains_key(&w) && w != start) {
                continue;
            }
            prev.insert(w, Some(v));
            if w == targets.0 || w == targets.1 {
                hit = Some(w);
                break;
            }
            q.push_back(w);
        }
    }
    let hit = hit?;
    let mut out = Vec::new();
    let mut v = Some(hit);
    while let Some(x) = v {
        if x == start {
            break;
        }
        out.push(x);
        v = prev[&x];
    }
    out.reverse();
    Some(out)
}

/// `manual_path_extend`: grow or shrink a dragged path (vertex ids) with the
/// edge `edge` under the cursor. True when the chain changed.
pub fn manual_path_extend(
    g: &PathGraph,
    chain: &mut Vec<usize>,
    edge: usize,
    skip: &dyn Fn(usize) -> bool,
    limit: usize,
) -> bool {
    if chain.is_empty() || skip(edge) {
        return false;
    }
    let last = *chain.last().unwrap();
    let (v0, v1) = g.edges[edge];
    // {id(v): i for i, v in enumerate(chain)} — a later index overwrites.
    let mut ids: HashMap<usize, usize> = HashMap::new();
    for (i, &v) in chain.iter().enumerate() {
        ids.insert(v, i);
    }
    if last == v0 || last == v1 {
        let other = if last == v0 { v1 } else { v0 };
        if chain.len() >= 2 && other == chain[chain.len() - 2] {
            chain.pop();
            return true;
        }
        if ids.contains_key(&other) {
            if other == chain[0] && chain.len() >= 3 {
                chain.push(other);
                return true;
            }
            return false;
        }
        chain.push(other);
        return true;
    }
    if ids.contains_key(&v0) && ids.contains_key(&v1) {
        let cut = ids[&v0].max(ids[&v1]) + 1;
        chain.truncate(cut);
        return true;
    }
    let Some(r) = route(g, last, (v0, v1), skip, &ids, limit) else {
        return false;
    };
    chain.extend(r.iter().copied());
    let reached = *r.last().unwrap();
    let other = if reached == v0 { v1 } else { v0 };
    if !ids.contains_key(&other) && other != reached {
        chain.push(other);
    }
    true
}

/// `orient_closed_path`: station 0 at the vertex nearest the profile, going
/// the way whose first segment stands closest to perpendicular to it.
pub fn orient_closed_path(pts: &[V3], face: &Profile, h: Hypot) -> Vec<V3> {
    let mut pts: Vec<V3> = pts.to_vec();
    if pts.len() < 3 {
        return pts;
    }
    let c = centroid(&face.outer);
    let mut k = 0usize;
    let mut best = pts[0].sub(c).length(h);
    for i in 1..pts.len() {
        let l = pts[i].sub(c).length(h);
        if l < best {
            best = l;
            k = i;
        }
    }
    pts.rotate_left(k);
    let n = normal(&face.outer, h).normalized(h);
    let fwd = pts[1].sub(pts[0]);
    let back = pts[pts.len() - 1].sub(pts[0]);
    if f(fwd.length(h)) > 1e-9 && f(back.length(h)) > 1e-9 {
        let af = f(fwd.normalized(h).dot(n)).abs();
        let ab = f(back.normalized(h).dot(n)).abs();
        if ab > af + 1e-9 {
            pts[1..].reverse();
        }
    }
    pts
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hypot_basic() {
        assert_eq!(py_hypot2(3.0, 4.0), 5.0);
        assert_eq!(py_hypot2(0.0, 0.0), 0.0);
        assert_eq!(py_hypot2(0.0, -2.0), 2.0);
    }
}
