// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! `core/cap_rebuild.py`: the deterministic plane rebuild of push/pull
//! ("path C").
//!
//! One recompute per touched plane: gather the mesh edges lying on the
//! plane, run the planar arrangement, classify each region by which side of
//! the plane holds material (parity ray casting, the primitive of
//! `orient_outward`), union each side's regions into the final face loops.
//!
//! Arithmetic as Python's: 2D coordinates are f64 (the f32 dot products of
//! `QVector3D` widened), 3D points are f32 (`origin + u * x + v * y`). The
//! jitter of the parity rays comes from `random.Random(54321)`
//! (`PyRandom`).
//!
//! ORDER. Python iterates sets of `Face` objects (address order) in
//! `seam_planes` and keeps a dict of planes in insertion order; faces here
//! go in ascending id (creation order). The plane a pass rebuilds first is
//! decided by `sorted(planes)` (the rounded key), not by that order; the
//! order only picks which face's centroid stands as the plane's origin.

use std::collections::{BTreeSet, HashMap, HashSet};
use std::rc::Rc;

use ingetrazo_geom::meshface;
use ingetrazo_geom::parity::Tris;
use ingetrazo_math::{Hypot, V3};
use ingetrazo_model::mesh::{EdgeId, FaceId, Mesh};
use ingetrazo_model::topology::PFace;

use crate::arrangement::{area_floor, arrange, interior_point, plane_basis, point_in_polygon, scan_interior_point, signed_area, XY};
use crate::orient::{face_components, ray_parity_outside_tris};
use crate::pyrandom::PyRandom;

const TOL: f64 = 1e-4;
/// A point this close to the plane (along its normal) counts as on it.
const ON_PLANE: f64 = 1e-4;
/// Normal offset of the material sample points.
const SAMPLE_OFF: f64 = 1e-3;

/// `(outer, holes, is_partition)` of a rebuilt face.
pub type Rebuilt = (Vec<V3>, Vec<Vec<V3>>, bool);

/// `(key, origin, normal)` of `plane_key`, the key as the rounded floats.
pub type PlaneKey = [f64; 4];

fn dotf(a: V3, b: V3) -> f64 {
    a.dot(b) as f64
}

fn lenf(a: V3, h: Hypot) -> f64 {
    a.length(h) as f64
}

/// `RebuildCache`: per-face memos shared across one push's plane rebuilds.
#[derive(Default)]
pub struct RebuildCache {
    /// Outlines of the strips a rebuild found inside the solid and dropped
    /// (issue #94), for `edges_along`.
    pub dropped: Vec<(V3, V3)>,
    tris: HashMap<FaceId, Rc<Vec<[V3; 3]>>>,
    norm: HashMap<FaceId, V3>,
    cent: HashMap<FaceId, V3>,
    /// Bumped by the caller whenever a rebuild actually changed the mesh.
    pub token: u64,
    packed: Option<(u64, (Option<Vec<FaceId>>, Option<Vec<usize>>), Rc<Tris>)>,
    comp: Option<HashMap<FaceId, usize>>,
    comp_token: Option<u64>,
    /// The operation's own faces: their components are the only solids a
    /// rebuild may touch.
    pub seed_faces: HashSet<FaceId>,
    scope: Option<(Vec<FaceId>, Vec<EdgeId>)>,
    scope_token: Option<u64>,
}

impl RebuildCache {
    pub fn new() -> RebuildCache {
        RebuildCache::default()
    }

    /// Face -> edge-connected component label, recomputed when the mesh
    /// changes.
    pub fn components(&mut self, m: &Mesh) -> &HashMap<FaceId, usize> {
        if self.comp.is_none() || self.comp_token != Some(self.token) {
            self.comp = Some(face_components(m));
            self.comp_token = Some(self.token);
        }
        self.comp.as_ref().unwrap()
    }

    /// `(faces, edges)` of the solids the operation is working in, or
    /// `None` when it has no seed to narrow by.
    pub fn scope(&mut self, m: &Mesh) -> Option<(Vec<FaceId>, Vec<EdgeId>)> {
        if self.seed_faces.is_empty() {
            return None;
        }
        if self.scope.is_some() && self.scope_token == Some(self.token) {
            return self.scope.clone();
        }
        let seeds = self.seed_faces.clone();
        let comp = self.components(m);
        let live: HashSet<usize> = seeds.iter().filter_map(|f| comp.get(f).copied()).collect();
        if live.is_empty() {
            self.scope = None;
            self.scope_token = Some(self.token);
            return None;
        }
        let faces: Vec<FaceId> = m.faces.iter().copied().filter(|f| comp.get(f).is_some_and(|c| live.contains(c))).collect();
        let fset: HashSet<FaceId> = faces.iter().copied().collect();
        // An edge belongs to the scope when a face of it does. A faceless
        // edge is plane debris a rebuild may still need to re-trace.
        let edges: Vec<EdgeId> = m.edges.iter().copied().filter(|&e| m.edge(e).faces.is_empty() || m.edge(e).faces.iter().any(|f| fset.contains(f))).collect();
        self.scope = Some((faces, edges));
        self.scope_token = Some(self.token);
        self.scope.clone()
    }

    pub fn triangles(&mut self, m: &Mesh, f: FaceId, h: Hypot) -> Rc<Vec<[V3; 3]>> {
        self.tris.entry(f).or_insert_with(|| Rc::new(PFace::of(m, f).triangulate(h))).clone()
    }

    /// `f.normal().normalized()`.
    pub fn normal(&mut self, m: &Mesh, f: FaceId, h: Hypot) -> V3 {
        *self.norm.entry(f).or_insert_with(|| m.face_normal(f, h).normalized(h))
    }

    pub fn centroid(&mut self, m: &Mesh, f: FaceId) -> V3 {
        *self.cent.entry(f).or_insert_with(|| meshface::centroid(&ingetrazo_model::topology::positions(m, &m.face(f).outer)))
    }

    /// The packed parity triangles of the counted faces (boundary, not
    /// shadowed, and — when `want` is given — in one of those components),
    /// memoised until the mesh changes.
    pub fn packed(&mut self, m: &Mesh, shadowed: &HashSet<FaceId>, want: Option<&HashSet<usize>>, h: Hypot) -> Rc<Tris> {
        let mut sh: Vec<FaceId> = shadowed.iter().copied().collect();
        sh.sort();
        let mut wk: Option<Vec<usize>> = want.map(|w| w.iter().copied().collect());
        if let Some(w) = wk.as_mut() {
            w.sort();
        }
        let key = (if sh.is_empty() { None } else { Some(sh) }, wk);
        if let Some((tok, k, val)) = &self.packed {
            if *tok == self.token && *k == key {
                return val.clone();
            }
        }
        let comp = if want.is_some() { Some(self.components(m).clone()) } else { None };
        let mut lists: Vec<Rc<Vec<[V3; 3]>>> = Vec::new();
        for &f in &m.faces {
            if m.face(f).interior || shadowed.contains(&f) {
                continue;
            }
            if let (Some(w), Some(c)) = (want, comp.as_ref()) {
                if !c.get(&f).is_some_and(|l| w.contains(l)) {
                    continue;
                }
            }
            lists.push(self.triangles(m, f, h));
        }
        let (mut a, mut e1, mut e2, mut idx) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for (li, tl) in lists.iter().enumerate() {
            for t in tl.iter() {
                let p = |v: V3| [v.0 as f64, v.1 as f64, v.2 as f64];
                let (ta, tb, tc) = (p(t[0]), p(t[1]), p(t[2]));
                a.push(ta);
                e1.push([tb[0] - ta[0], tb[1] - ta[1], tb[2] - ta[2]]);
                e2.push([tc[0] - ta[0], tc[1] - ta[1], tc[2] - ta[2]]);
                idx.push(li as i64);
            }
        }
        let val = Rc::new(Tris::new(a, e1, e2, idx));
        self.packed = Some((self.token, key, val.clone()));
        val
    }
}

/// `_key2`: the arrangement's integer cell of a 2D point.
fn key2(p: XY) -> (i64, i64) {
    ((p.0 / TOL).round_ties_even() as i64, (p.1 / TOL).round_ties_even() as i64)
}

fn on_plane(pos: V3, origin: V3, normal: V3) -> bool {
    dotf(pos.sub(origin), normal).abs() < ON_PLANE
}

/// `_region_test_point`: a point inside `outer` but outside every hole.
fn region_test_point(outer: &[XY], holes: &[Vec<XY>]) -> XY {
    let p = interior_point(outer);
    if point_in_polygon(p, outer) && !holes.iter().any(|h| point_in_polygon(p, h)) {
        return p;
    }
    let n = outer.len();
    for i in 0..n {
        let (ax, ay) = outer[i];
        let (bx, by) = outer[(i + 1) % n];
        let (mx, my) = ((ax + bx) / 2.0, (ay + by) / 2.0);
        let (dx, dy) = (bx - ax, by - ay);
        let ln = ingetrazo_geom::sweep::py_hypot2(dx, dy);
        if ln < TOL {
            continue;
        }
        // The interior of a CCW loop is to the left of a->b.
        let q = (mx - dy / ln * 1e-3, my + dx / ln * 1e-3);
        if point_in_polygon(q, outer) && !holes.iter().any(|h| point_in_polygon(q, h)) {
            return q;
        }
    }
    scan_interior_point(outer, holes).unwrap_or(p)
}

/// `_on_seg2`: whether 2D point `p` lies on segment `a`-`b`.
fn on_seg2(p: XY, a: XY, b: XY) -> bool {
    let tol = TOL / 2.0;
    let (abx, aby) = (b.0 - a.0, b.1 - a.1);
    let (apx, apy) = (p.0 - a.0, p.1 - a.1);
    if (abx * apy - aby * apx).abs() > tol * ingetrazo_math::pyfloat::py_max2(1.0, abx.abs() + aby.abs()) {
        return false;
    }
    let dot = apx * abx + apy * aby;
    -tol <= dot && dot <= abx * abx + aby * aby + tol
}

type Region = (Vec<XY>, Vec<Vec<XY>>);

/// `_union_outline`: union a set of 2D regions into `[(outer, [holes])]`,
/// dropping every edge interior to the union; `keep_segs` are creases.
fn union_outline(regions: &[Region], keep_segs: &[(XY, XY)]) -> Vec<Region> {
    type K = (i64, i64);
    let mut dir_count: Vec<((K, K), i64)> = Vec::new();
    let mut dir_index: HashMap<(K, K), usize> = HashMap::new();
    let mut coords: HashMap<K, XY> = HashMap::new();
    for (outer, holes) in regions {
        for lp in std::iter::once(outer).chain(holes.iter()) {
            let n = lp.len();
            for i in 0..n {
                let (a, b) = (lp[i], lp[(i + 1) % n]);
                let (ka, kb) = (key2(a), key2(b));
                coords.insert(ka, a);
                coords.insert(kb, b);
                match dir_index.get(&(ka, kb)) {
                    Some(&j) => dir_count[j].1 += 1,
                    None => {
                        dir_index.insert((ka, kb), dir_count.len());
                        dir_count.push(((ka, kb), 1));
                    }
                }
            }
        }
    }
    let is_crease = |ka: K, kb: K| -> bool {
        if keep_segs.is_empty() {
            return false;
        }
        let (pa, pb) = (coords[&ka], coords[&kb]);
        let mid = ((pa.0 + pb.0) / 2.0, (pa.1 + pb.1) / 2.0);
        keep_segs.iter().any(|&(a, b)| on_seg2(mid, a, b) && on_seg2(pa, a, b) && on_seg2(pb, a, b))
    };
    // Net direction per undirected edge: interior edges cancel, boundary
    // edges survive in the direction that keeps the solid on the left.
    let mut survivors: HashMap<(K, K), i64> = HashMap::new();
    for &((ka, kb), c) in &dir_count {
        if is_crease(ka, kb) {
            *survivors.entry((ka, kb)).or_insert(0) += c;
            continue;
        }
        let net = c - dir_index.get(&(kb, ka)).map_or(0, |&j| dir_count[j].1);
        if net > 0 {
            *survivors.entry((ka, kb)).or_insert(0) += net;
        }
    }
    // Trace the surviving directed edges into closed loops, choosing at each
    // junction the next edge clockwise from the arrival direction.
    let mut nbrs: HashMap<K, Vec<K>> = HashMap::new();
    let mut nbr_keys: Vec<K> = Vec::new();
    for &((ka, kb), _) in &dir_count {
        let c = survivors.get(&(ka, kb)).copied().unwrap_or(0);
        if c > 0 {
            for (x, y) in [(ka, kb), (kb, ka)] {
                let e = nbrs.entry(x).or_insert_with(|| {
                    nbr_keys.push(x);
                    Vec::new()
                });
                if !e.contains(&y) {
                    e.push(y);
                }
            }
        }
    }
    let mut order: HashMap<K, Vec<K>> = HashMap::new();
    for k in &nbr_keys {
        let (kx, ky) = coords[k];
        let mut ring = nbrs[k].clone();
        ring.sort_by(|a, b| {
            let ta = (coords[a].1 - ky).atan2(coords[a].0 - kx);
            let tb = (coords[b].1 - ky).atan2(coords[b].0 - kx);
            ta.partial_cmp(&tb).unwrap_or(std::cmp::Ordering::Equal)
        });
        order.insert(*k, ring);
    }
    let mut starts: Vec<(K, K)> = survivors.keys().copied().collect();
    starts.sort();
    let mut loops: Vec<Vec<XY>> = Vec::new();
    for start in starts {
        while survivors[&start] > 0 {
            let (u, v) = start;
            *survivors.get_mut(&start).unwrap() -= 1;
            let mut lp: Vec<K> = Vec::new();
            let (mut cu, mut cv) = (u, v);
            let mut ok = true;
            loop {
                lp.push(cu);
                let ring = &order[&cv];
                let idx = ring.iter().position(|&x| x == cu).expect("arrival in ring");
                let m = ring.len();
                let mut w = None;
                for s in 1..=m {
                    let cand = ring[(idx + m * 2 - s) % m];
                    if survivors.get(&(cv, cand)).copied().unwrap_or(0) > 0 || (cv, cand) == start {
                        w = Some(cand);
                        break;
                    }
                }
                let Some(w) = w else {
                    ok = false;
                    break;
                };
                if lp.len() > 100000 {
                    ok = false;
                    break;
                }
                cu = cv;
                cv = w;
                if (cu, cv) == start {
                    break;
                }
                *survivors.get_mut(&(cu, cv)).unwrap() -= 1;
            }
            if ok && lp.len() >= 3 {
                loops.push(lp.iter().map(|k| coords[k]).collect());
            }
        }
    }
    // CCW loops are outer faces, CW loops are holes; nest each hole in the
    // smallest outer that contains it.
    let mut outers: Vec<&Vec<XY>> = loops.iter().filter(|lp| signed_area(lp) > area_floor(lp)).collect();
    let holes: Vec<&Vec<XY>> = loops.iter().filter(|lp| signed_area(lp) < -area_floor(lp)).collect();
    outers.sort_by(|a, b| signed_area(a).partial_cmp(&signed_area(b)).unwrap_or(std::cmp::Ordering::Equal));
    let mut result: Vec<Region> = outers.iter().map(|o| ((*o).clone(), Vec::new())).collect();
    for hl in holes {
        let hp = interior_point(hl);
        let hkeys: BTreeSet<K> = hl.iter().map(|&p| key2(p)).collect();
        for (outer, hs) in result.iter_mut() {
            if outer.len() == hl.len() && outer.iter().map(|&p| key2(p)).collect::<BTreeSet<K>>() == hkeys {
                // The hole is this outer's own outline: it belongs to the
                // enclosing face.
                continue;
            }
            if point_in_polygon(hp, outer) {
                hs.push(hl.clone());
                break;
            }
        }
    }
    result
}

/// `_coplanar_on`.
fn coplanar_on(m: &Mesh, f: FaceId, origin: V3, normal: V3, cache: &mut RebuildCache, h: Hypot) -> bool {
    let fnr = cache.normal(m, f, h);
    if dotf(fnr, normal).abs() <= 0.999 {
        return false;
    }
    let fc = cache.centroid(m, f);
    on_plane(fc, origin, normal)
}

/// `rebuild_plane`: the solid boundary faces of `m` on the plane
/// `(origin, normal)`, `[(outer, holes, is_partition)]`, boundary loops
/// wound outward (toward the empty side); `None` when the plane carries too
/// little to face. Pure: it reads the mesh. `op` collects the push's rim
/// segments (and the rims of strips found inside the solid).
pub fn rebuild_plane(
    m: &Mesh,
    origin: V3,
    normal: V3,
    fresh: &HashSet<FaceId>,
    keep_mode: bool,
    removing: bool,
    op: &mut Vec<(V3, V3)>,
    cache: &mut RebuildCache,
    h: Hypot,
) -> Option<Vec<Rebuilt>> {
    let normal = normal.normalized(h);
    let (u, v) = plane_basis(normal, h);
    let (scope_faces, scope_edges) = match cache.scope(m) {
        None => (m.faces.clone(), m.edges.clone()),
        Some(s) => s,
    };
    let to2d = |p: V3| -> XY {
        let d = p.sub(origin);
        (dotf(d, u), dotf(d, v))
    };
    let to3d = |xy: XY| -> V3 { origin.add(u.mul(xy.0 as f32)).add(v.mul(xy.1 as f32)) };
    let epos = |e: EdgeId| (m.position(m.edge(e).v0), m.position(m.edge(e).v1));

    let segs: Vec<(V3, V3)> = scope_edges
        .iter()
        .map(|&e| epos(e))
        .filter(|&(a, b)| on_plane(a, origin, normal) && on_plane(b, origin, normal))
        .collect();
    if segs.len() < 3 {
        return None;
    }
    let arranged = arrange(&segs, origin, normal, h);
    let regions_3d: Vec<(Vec<V3>, Vec<Vec<V3>>)> =
        arranged.faces.iter().map(|(o, hs)| (o.iter().map(|&p| arranged.to3d(p)).collect(), hs.iter().map(|hl| hl.iter().map(|&p| arranged.to3d(p)).collect()).collect())).collect();
    if regions_3d.is_empty() {
        return None;
    }

    let live: HashSet<FaceId> = m.faces.iter().copied().collect();
    let fresh_set: HashSet<FaceId> = fresh.iter().copied().filter(|f| live.contains(f)).collect();
    let mut fresh_sorted: Vec<FaceId> = fresh_set.iter().copied().collect();
    fresh_sorted.sort();
    let mut shadowed: HashSet<FaceId> = HashSet::new();
    if keep_mode {
        for &f in &fresh_sorted {
            if m.face(f).interior {
                continue;
            }
            let fnr = cache.normal(m, f, h);
            let fc = cache.centroid(m, f);
            let (fu, fv) = plane_basis(fnr, h);
            let fto2d = |p: V3| -> XY {
                let d = p.sub(fc);
                (dotf(d, fu), dotf(d, fv))
            };
            for &g in &scope_faces {
                if g == f || fresh_set.contains(&g) {
                    continue;
                }
                let gn = cache.normal(m, g, h);
                if dotf(gn, fnr).abs() < 0.999 {
                    continue;
                }
                let gc = cache.centroid(m, g);
                if dotf(gc.sub(fc), fnr).abs() > ON_PLANE {
                    continue;
                }
                let pg = PFace::of(m, g);
                let outer: Vec<XY> = pg.vertices.iter().map(|&p| fto2d(p)).collect();
                if point_in_polygon((0.0, 0.0), &outer) && !pg.holes.iter().any(|hl| point_in_polygon((0.0, 0.0), &hl.iter().map(|&p| fto2d(p)).collect::<Vec<XY>>())) {
                    shadowed.insert(f);
                    break;
                }
            }
        }
    }
    // The parity set is restricted to the edge-connected component(s) this
    // plane belongs to.
    let mut want: HashSet<usize> = HashSet::new();
    {
        let comp = cache.components(m).clone();
        for &f in &scope_faces {
            if let Some(&c) = comp.get(&f) {
                if coplanar_on(m, f, origin, normal, cache, h) {
                    want.insert(c);
                }
            }
        }
        for f in &fresh_set {
            if let Some(&c) = comp.get(f) {
                want.insert(c);
            }
        }
    }
    let packed = cache.packed(m, &shadowed, if want.is_empty() { None } else { Some(&want) }, h);
    let mut rng = PyRandom::new(54321);

    type Poly = (Vec<XY>, Vec<Vec<XY>>);
    let proj_polys = |faces: &[FaceId], cache: &mut RebuildCache| -> Vec<(Poly, bool)> {
        let mut out = Vec::new();
        for &f in faces {
            if !coplanar_on(m, f, origin, normal, cache, h) {
                continue;
            }
            let pf = PFace::of(m, f);
            let plus = dotf(cache.normal(m, f, h), normal) > 0.0;
            out.push(((pf.vertices.iter().map(|&p| to2d(p)).collect(), pf.holes.iter().map(|hl| hl.iter().map(|&p| to2d(p)).collect()).collect()), plus));
        }
        out
    };
    let covers = |poly: &Poly, pt: XY| -> bool { point_in_polygon(pt, &poly.0) && !poly.1.iter().any(|hl| point_in_polygon(pt, hl)) };

    let fresh_boundary: Vec<FaceId> = fresh_sorted.iter().copied().filter(|&f| !m.face(f).interior).collect();
    let fresh_polys = if keep_mode { Vec::new() } else { proj_polys(&fresh_boundary, cache) };
    let fresh_cover_polys = proj_polys(&fresh_boundary, cache);
    let old_all: Vec<FaceId> = scope_faces.iter().copied().filter(|f| !fresh_set.contains(f) || m.face(*f).interior).collect();
    let old_polys = proj_polys(&old_all, cache);
    let old_boundary: Vec<FaceId> = scope_faces.iter().copied().filter(|f| !fresh_set.contains(f) && !m.face(*f).interior).collect();
    let old_boundary_polys = proj_polys(&old_boundary, cache);

    // Group boundary regions by which side holds the material.
    let mut solid_by_side: [Vec<Region>; 2] = [Vec::new(), Vec::new()]; // [True, False]
    let side = |plus: bool| if plus { 0 } else { 1 };
    let mut partitions: Vec<Region> = Vec::new();
    let mut sheets: Vec<Region> = Vec::new();
    for (outer, holes) in &regions_3d {
        let outer_xy: Vec<XY> = outer.iter().map(|&p| to2d(p)).collect();
        let holes_xy: Vec<Vec<XY>> = holes.iter().map(|hl| hl.iter().map(|&p| to2d(p)).collect()).collect();
        let ip_xy = region_test_point(&outer_xy, &holes_xy);
        let ip = to3d(ip_xy);
        let off = SAMPLE_OFF as f32;
        let mat_plus = ray_parity_outside_tris(ip.add(normal.mul(off)), normal, &packed, None, None, &mut rng, h) == Some(false);
        let mat_minus = ray_parity_outside_tris(ip.sub(normal.mul(off)), normal.neg(), &packed, None, None, &mut rng, h) == Some(false);
        if mat_plus != mat_minus {
            solid_by_side[side(mat_plus)].push((outer_xy, holes_xy));
            continue;
        }
        let mut decl: Vec<bool> = Vec::new();
        for (poly, p) in &fresh_polys {
            if covers(poly, ip_xy) && !decl.contains(p) {
                decl.push(*p);
            }
        }
        if mat_plus {
            // Material on both sides.
            if removing && decl.len() == 1 {
                solid_by_side[side(!decl[0])].push((outer_xy, holes_xy));
            } else if !removing
                && !keep_mode
                && old_boundary_polys.iter().any(|(poly, plus)| covers(poly, ip_xy) && fresh_polys.iter().any(|(fpoly, fplus)| covers(fpoly, ip_xy) && fplus != plus))
            {
                // The strip is inside the solid now, and so is its outline.
                let mut rim: Vec<(V3, V3)> = Vec::new();
                for lp in std::iter::once(&outer_xy).chain(holes_xy.iter()) {
                    for i in 0..lp.len() {
                        rim.push((to3d(lp[i]), to3d(lp[(i + 1) % lp.len()])));
                    }
                }
                op.extend(rim.iter().copied());
                cache.dropped.extend(rim);
            } else if (!removing || decl.is_empty())
                && (old_polys.iter().any(|(poly, _)| covers(poly, ip_xy)) || (keep_mode && fresh_cover_polys.iter().any(|(poly, _)| covers(poly, ip_xy))))
            {
                partitions.push((outer_xy, holes_xy));
            }
        } else if !removing && decl.len() == 1 {
            // An outward sweep envelops old boundary faces that still count
            // in parity: its own quad declares the new material side.
            solid_by_side[side(!decl[0])].push((outer_xy, holes_xy));
        } else if decl.is_empty() && old_polys.iter().any(|(poly, _)| covers(poly, ip_xy)) {
            sheets.push((outer_xy, holes_xy));
        }
        // otherwise: op debris (a flush-landed cap or collapsed fin)
    }

    // Creases: plane edges a non-coplanar face stands on, and (with `op`)
    // the face-bearing plane edges that lie on none of the op's rims.
    let mut op_rim_segs: Vec<(XY, XY)> = Vec::new();
    for &(a3, b3) in op.iter() {
        if on_plane(a3, origin, normal) && on_plane(b3, origin, normal) {
            op_rim_segs.push((to2d(a3), to2d(b3)));
        }
    }
    let on_op_rim = |e: EdgeId| -> bool {
        let (pa, pb) = epos(e);
        let (ea, eb) = (to2d(pa), to2d(pb));
        op_rim_segs.iter().any(|&(a, b)| on_seg2(ea, a, b) && on_seg2(eb, a, b))
    };
    let mut keep_segs: Vec<(XY, XY)> = Vec::new();
    for &e in &scope_edges {
        let (pa, pb) = epos(e);
        if !(on_plane(pa, origin, normal) && on_plane(pb, origin, normal)) {
            continue;
        }
        let faces = m.edge(e).faces.clone();
        let mut perp = false;
        for f in &faces {
            if dotf(cache.normal(m, *f, h), normal).abs() < 0.999 {
                perp = true;
                break;
            }
        }
        // A plane edge a non-coplanar face stands on is a crease; so is a
        // face-bearing plane edge lying on none of the op's rims (the user's
        // own subdivision).
        if perp || (!faces.is_empty() && !on_op_rim(e)) {
            keep_segs.push((to2d(pa), to2d(pb)));
        }
    }

    let lift = |region: &Region, reverse: bool| -> Rebuilt {
        let (outer, holes) = region;
        let mut o: Vec<V3> = outer.iter().map(|&p| to3d(p)).collect();
        let mut hs: Vec<Vec<V3>> = holes.iter().map(|hl| hl.iter().map(|&p| to3d(p)).collect()).collect();
        if reverse {
            o.reverse();
            for hl in hs.iter_mut() {
                hl.reverse();
            }
        }
        (o, hs, false)
    };
    let mut result: Vec<Rebuilt> = Vec::new();
    for (i, regions) in solid_by_side.iter().enumerate() {
        if regions.is_empty() {
            continue;
        }
        let mat_plus = i == 0;
        for region in union_outline(regions, &keep_segs) {
            // CCW in (u, v) gives a +normal face; flip when the material is
            // on the +side so the face points toward the empty side.
            result.push(lift(&region, mat_plus));
        }
    }
    if !partitions.is_empty() {
        for region in union_outline(&partitions, &keep_segs) {
            let mut r = lift(&region, false);
            r.2 = true;
            result.push(r);
        }
    }
    if !sheets.is_empty() {
        for region in union_outline(&sheets, &keep_segs) {
            result.push(lift(&region, false));
        }
    }
    Some(result)
}

/// `_canon_loop`: the cycle of weld keys rotated so the smallest leads.
fn canon_loop(lp: &[V3]) -> Vec<(i64, i64, i64)> {
    let r = |c: f32| ((c as f64) / 1e-4).round_ties_even() as i64;
    let keys: Vec<(i64, i64, i64)> = lp.iter().map(|p| (r(p.0), r(p.1), r(p.2))).collect();
    let mut i = 0;
    for j in 1..keys.len() {
        if keys[j] < keys[i] {
            i = j;
        }
    }
    let mut out = keys[i..].to_vec();
    out.extend_from_slice(&keys[..i]);
    out
}

type Canon = BTreeSet<(Vec<(i64, i64, i64)>, BTreeSet<Vec<(i64, i64, i64)>>, bool)>;

fn canon_faces(faces: &[Rebuilt]) -> Canon {
    faces.iter().map(|(o, hs, interior)| (canon_loop(o), hs.iter().map(|hl| canon_loop(hl)).collect(), *interior)).collect()
}

/// `apply_rebuild`: rebuild one plane of `m` in place; whether anything
/// changed (False when the plane already matches, so callers can iterate to
/// a fixpoint).
pub fn apply_rebuild(
    m: &mut Mesh,
    origin: V3,
    normal: V3,
    fresh: &HashSet<FaceId>,
    keep_mode: bool,
    removing: bool,
    prune: bool,
    op: &mut Vec<(V3, V3)>,
    cache: &mut RebuildCache,
    h: Hypot,
) -> bool {
    let normal = normal.normalized(h);
    let Some(rebuilt) = rebuild_plane(m, origin, normal, fresh, keep_mode, removing, op, cache, h) else {
        return false;
    };
    let scope_faces = match cache.scope(m) {
        None => m.faces.clone(),
        Some(s) => s.0,
    };
    let mut old: Vec<FaceId> = Vec::new();
    for f in scope_faces {
        if coplanar_on(m, f, origin, normal, cache, h) {
            old.push(f);
        }
    }
    let old_canon = canon_faces(
        &old.iter()
            .map(|&f| {
                let pf = PFace::of(m, f);
                (pf.vertices, pf.holes, m.face(f).interior)
            })
            .collect::<Vec<Rebuilt>>(),
    );
    if canon_faces(&rebuilt) == old_canon {
        return false; // plane already in its rebuilt form: stable
    }
    // From here the mesh changes, so the memoised parity set is stale.
    cache.token += 1;
    // Region inheritance: a new face takes the attrs of the old face whose
    // interior contains its interior point.
    let (u, v) = plane_basis(normal, h);
    let to2d = |p: V3| -> XY {
        let d = p.sub(origin);
        (dotf(d, u), dotf(d, v))
    };
    let donors: Vec<(Vec<XY>, Vec<Vec<XY>>, Vec<(String, ingetrazo_formats::pyjson::Value)>)> = old
        .iter()
        .filter(|&&f| !m.face(f).attrs.is_empty())
        .map(|&f| {
            let pf = PFace::of(m, f);
            (pf.vertices.iter().map(|&p| to2d(p)).collect(), pf.holes.iter().map(|hl| hl.iter().map(|&p| to2d(p)).collect()).collect(), m.face(f).attrs.clone())
        })
        .collect();
    for f in old {
        m.remove_face(f);
    }
    for (outer, holes, interior) in &rebuilt {
        let nf = m.add_face(outer, holes);
        m.face_mut(nf).interior = *interior;
        if !donors.is_empty() {
            let ip = region_test_point(&outer.iter().map(|&p| to2d(p)).collect::<Vec<XY>>(), &holes.iter().map(|hl| hl.iter().map(|&p| to2d(p)).collect()).collect::<Vec<Vec<XY>>>());
            for (o_xy, hs_xy, attrs) in &donors {
                if point_in_polygon(ip, o_xy) && !hs_xy.iter().any(|hl| point_in_polygon(ip, hl)) {
                    m.face_mut(nf).attrs = attrs.clone();
                    break;
                }
            }
        }
    }
    if prune {
        prune_plane_debris(m, origin, normal);
    }
    true
}

/// `prune_plane_debris`: drop edges that lie fully on the plane and face
/// nothing, and the vertices that leave behind with no incidence at all.
pub fn prune_plane_debris(m: &mut Mesh, origin: V3, normal: V3) {
    for e in m.edges.clone() {
        let (a, b) = (m.position(m.edge(e).v0), m.position(m.edge(e).v1));
        if m.edge(e).faces.is_empty() && on_plane(a, origin, normal) && on_plane(b, origin, normal) {
            m.remove_edge(e);
        }
    }
    // Only vertices no face loop references either.
    let mut referenced: HashSet<ingetrazo_model::VertexId> = HashSet::new();
    for &f in &m.faces {
        let face = m.face(f);
        referenced.extend(face.outer.iter().copied());
        for hl in &face.holes {
            referenced.extend(hl.iter().copied());
        }
    }
    let doomed: Vec<ingetrazo_model::VertexId> = m.vertices.iter().copied().filter(|&v| m.vertex_ref(v).edges.is_empty() && !referenced.contains(&v)).collect();
    m.drop_orphan_vertices(&doomed);
}

/// Python's `round(x, 3)`: the decimal expansion rounded half to even and
/// read back (`float.__round__` does the same).
fn round3(x: f64) -> f64 {
    let r: f64 = format!("{x:.3}").parse().unwrap_or(x);
    if r == 0.0 {
        0.0
    } else {
        r
    }
}

/// `plane_key`: `(key, origin, normal)` for the plane through `point` with
/// `normal`; the normal's largest-magnitude component is made positive.
pub fn plane_key(point: V3, normal: V3, h: Hypot) -> (PlaneKey, V3, V3) {
    let mut n = normal.normalized(h);
    let comps = [n.0, n.1, n.2];
    // `max(range(3), key=abs)` keeps the first of equals.
    let mut best = 0;
    for i in 1..3 {
        if comps[i].abs() > comps[best].abs() {
            best = i;
        }
    }
    if comps[best] < 0.0 {
        n = n.neg();
    }
    let d = dotf(n, point);
    ([round3(n.0 as f64), round3(n.1 as f64), round3(n.2 as f64), round3(d)], point, n)
}

/// A plane key as an orderable, hashable value.
fn pk_bits(k: &PlaneKey) -> [u64; 4] {
    [k[0].to_bits(), k[1].to_bits(), k[2].to_bits(), k[3].to_bits()]
}

/// `crack_planes`: a representative `(origin, normal)` for each distinct
/// plane carrying a crack (a boundary edge bordering one face), in order of
/// first appearance. `near` keeps only the cracks with an end inside it.
pub fn crack_planes(m: &Mesh, near: Option<[f64; 6]>, h: Hypot) -> Vec<(PlaneKey, V3, V3)> {
    let mut seen: HashSet<[u64; 4]> = HashSet::new();
    let mut out = Vec::new();
    let inside = |p: V3, nr: &[f64; 6]| -> bool {
        let (x, y, z) = (p.0 as f64, p.1 as f64, p.2 as f64);
        nr[0] <= x && x <= nr[3] && nr[1] <= y && y <= nr[4] && nr[2] <= z && z <= nr[5]
    };
    for &e in &m.edges {
        let ed = m.edge(e);
        if ed.faces.len() != 1 {
            continue;
        }
        if let Some(nr) = &near {
            if !(inside(m.position(ed.v0), nr) || inside(m.position(ed.v1), nr)) {
                continue;
            }
        }
        let f = ed.faces[0];
        let pf = PFace::of(m, f);
        let (key, origin, n) = plane_key(meshface::centroid(&pf.vertices), pf.normal(h), h);
        if seen.insert(pk_bits(&key)) {
            out.push((key, origin, n));
        }
    }
    out
}

/// `edges_along`: every edge of `m` lying along one of `segments`.
pub fn edges_along(m: &Mesh, segments: &[(V3, V3)], h: Hypot) -> Vec<EdgeId> {
    if segments.is_empty() {
        return Vec::new();
    }
    let on = |p: V3, a: V3, b: V3| -> bool {
        let ab = b.sub(a);
        let l2 = ab.length_squared() as f64;
        if l2 < 1e-18 {
            return lenf(p.sub(a), h) < ON_PLANE;
        }
        let t = dotf(p.sub(a), ab) / l2;
        if t < -1e-9 || t > 1.0 + 1e-9 {
            return false;
        }
        lenf(a.add(ab.mul(t as f32)).sub(p), h) < ON_PLANE
    };
    m.edges
        .iter()
        .copied()
        .filter(|&e| {
            let (pa, pb) = (m.position(m.edge(e).v0), m.position(m.edge(e).v1));
            segments.iter().any(|&(a, b)| on(pa, a, b) && on(pb, a, b))
        })
        .collect()
}

/// `seam_planes`: a representative `(origin, normal)` for each plane where
/// one of `new_faces` ended up coplanar-edge-adjacent to another face.
pub fn seam_planes(m: &Mesh, new_faces: &HashSet<FaceId>, h: Hypot) -> Vec<(PlaneKey, V3, V3)> {
    let live: HashSet<FaceId> = m.faces.iter().copied().collect();
    let mut sorted: Vec<FaceId> = new_faces.iter().copied().collect();
    sorted.sort();
    let mut seen: HashSet<[u64; 4]> = HashSet::new();
    let mut out = Vec::new();
    for f in sorted {
        if !live.contains(&f) {
            continue;
        }
        let nf = m.face_normal(f, h).normalized(h);
        let face = m.face(f);
        for lp in std::iter::once(&face.outer).chain(face.holes.iter()) {
            let n = lp.len();
            for i in 0..n {
                let Some(e) = m.find_edge(lp[i], lp[(i + 1) % n]) else { continue };
                for &g in &m.edge(e).faces {
                    if g == f {
                        continue;
                    }
                    if dotf(nf, m.face_normal(g, h).normalized(h)).abs() > 0.999 {
                        let (key, origin, cn) = plane_key(meshface::centroid(&ingetrazo_model::topology::positions(m, &face.outer)), nf, h);
                        if seen.insert(pk_bits(&key)) {
                            out.push((key, origin, cn));
                        }
                    }
                }
            }
        }
    }
    out
}

/// Sort planes by key, as `sorted(planes)` does.
pub fn key_cmp(a: &PlaneKey, b: &PlaneKey) -> std::cmp::Ordering {
    for i in 0..4 {
        match a[i].partial_cmp(&b[i]) {
            Some(std::cmp::Ordering::Equal) | None => continue,
            Some(o) => return o,
        }
    }
    std::cmp::Ordering::Equal
}

/// The key of `plane_key` as hashable bits, for callers that dedupe planes.
pub fn key_bits(k: &PlaneKey) -> [u64; 4] {
    pk_bits(k)
}

#[allow(dead_code)]
fn _unused(_: &Mesh) {}
