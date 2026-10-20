// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! `core/topology.py`: graph queries over the edge network (cycles that
//! close a face), face splitting and subtraction, polygon offset, the heal
//! passes, and the defect counters.
//!
//! A line-by-line port. Arithmetic keeps Python's types and order:
//! `QVector3D` values are float32 `V3` (Qt operator order, no FMA), Python
//! floats are `f64`, `sum()` is `py_sum` (3.12 Neumaier), `round()` is
//! `round_ties_even`, `x ** 2` and `x ** 0.5` are the C runtime's `pow`
//! (`powf` through `black_box`, so LLVM cannot fold them into `x * x` and
//! `sqrt`, which differ in the last bit now and then), and numpy's
//! `einsum("ij,ij->i")` over three columns is `(a0*b0 + a2*b2) + a1*b1`
//! (measured: its SIMD kernel pairs the even lanes first). Every function that
//! takes a length or a normal takes the `Hypot` of the Qt build the
//! fixtures come from.
//!
//! Python's functions take lists of `Edge`/`Face` objects; here they take
//! the mesh and arena ids, or, where only positions matter, plain
//! positions (`Seg` for an edge, `PFace` for a face). `Face` means
//! `core.mesh.Face` (Newell normal in double, divided by its length), the one
//! the mesh holds. Anything that returns "the edge" or "the face" answers
//! with its index in the list given, or its id.
//!
//! ORDER notes (Python iterates sets of objects in address order, which no
//! run repeats):
//!
//! - `fold_nonplanar_faces` walks `pieces`, a set of faces: here it is the
//!   insertion order. The result depends on it only when more than one
//!   merge is possible at once.
//! - `defects_near` walks each vertex's edge set: here by edge id. Only
//!   counts come out, and they do not depend on the order except at the
//!   float32 boundary of `_coplanar_overlap` (it is evaluated from
//!   the first face of each pair).
//! - `Mesh::find_edge` answers from a set too, but only differs for two
//!   edges on one vertex pair, which no operation here creates.

use std::borrow::Cow;
use std::collections::{HashMap, HashSet, VecDeque};
use std::hint::black_box;

use ingetrazo_geom::facegeom::plane_axes;
use ingetrazo_geom::meshface::{self as mf, ikey};
use ingetrazo_geom::triangulate::triangulate;
use ingetrazo_math::pyfloat::{py_max2, py_min2, py_sum};
use ingetrazo_math::{Hypot, V3};

use crate::mesh::{EdgeId, FaceId, Mesh, VertexId};

pub const KEY_DECIMALS: i32 = 4;
pub const PLANAR_TOLERANCE: f64 = 1e-3;
/// Two points closer than this weld together; also the gate for deciding a
/// crossing is real.
pub const SPLIT_TOLERANCE: f64 = 1e-4;
/// `core.mesh._STITCH_TOL`.
pub const STITCH_TOL: f64 = 1e-4;
/// Above this many faces the overlap heal is skipped entirely.
pub const HEAL_FACE_CAP: usize = 3000;
pub const DEFECT_KINDS: [&str; 6] =
    ["duplicate_face", "overlapping_face", "pinched_face", "crossed_face", "flat_face", "zero_edge"];

/// `core.topology._key` as three exact integers: two points have equal
/// `_key`s exactly when these are equal.
pub type Key = [i64; 3];
/// An edge as its two end positions (`edge.a`, `edge.b`).
pub type Seg = (V3, V3);

/// `core.topology._key(p)`.
pub fn key(p: V3) -> Key {
    ikey(p)
}

/// `x ** 2.0` of the C runtime.
fn pow2(x: f64) -> f64 {
    x.powf(black_box(2.0))
}

/// `x ** 0.5` of the C runtime.
fn pow_half(x: f64) -> f64 {
    x.powf(black_box(0.5))
}

/// numpy `einsum("ij,ij->i")` over a row of three.
fn einsum3(a: [f64; 3], b: [f64; 3]) -> f64 {
    (a[0] * b[0] + a[2] * b[2]) + a[1] * b[1]
}

fn lenf(v: V3, h: Hypot) -> f64 {
    v.length(h) as f64
}

fn dotf(a: V3, b: V3) -> f64 {
    a.dot(b) as f64
}

fn ekey(a: Key, b: Key) -> (Key, Key) {
    if a <= b {
        (a, b)
    } else {
        (b, a)
    }
}

fn keyset(ps: &[V3]) -> HashSet<Key> {
    ps.iter().map(|&p| key(p)).collect()
}

/// Positions of a vertex loop.
pub fn positions(m: &Mesh, ids: &[VertexId]) -> Vec<V3> {
    ids.iter().map(|&v| m.position(v)).collect()
}

/// `[(e.a, e.b) for e in edges]`.
pub fn edge_segs(m: &Mesh, edges: &[EdgeId]) -> Vec<Seg> {
    edges.iter().map(|&e| (m.position(m.edge(e).v0), m.position(m.edge(e).v1))).collect()
}

// ---- A face as positions ----------------------------------------------------

/// A face's loops as positions: what `Face.vertices` and `Face.holes` give.
#[derive(Clone, Debug, Default)]
pub struct PFace {
    pub vertices: Vec<V3>,
    pub holes: Vec<Vec<V3>>,
}

impl PFace {
    pub fn of(m: &Mesh, f: FaceId) -> PFace {
        let face = m.face(f);
        PFace { vertices: positions(m, &face.outer), holes: face.holes.iter().map(|h| positions(m, h)).collect() }
    }
    /// `Face.normal()`.
    pub fn normal(&self, h: Hypot) -> V3 {
        mf::normal(&self.vertices, h)
    }
    /// `Face.area()`.
    pub fn area(&self, h: Hypot) -> f64 {
        if self.vertices.len() < 3 {
            return 0.0;
        }
        0.5 * lenf(mf::newell(&self.vertices), h)
    }
    /// `Face.triangulate()`.
    pub fn triangulate(&self, h: Hypot) -> Vec<[V3; 3]> {
        if self.vertices.len() < 3 {
            return Vec::new();
        }
        triangulate(&self.vertices, &self.holes, Some(self.normal(h)), h)
    }
}

// ---- Mesh helpers (Mesh methods in Python) -------------------------------------

/// `Mesh.add_face` for a call from this module: nested holes are dropped
/// first (`_maximal_holes`) when there are two or more, as the Python
/// choke point does, so the vertices of a dropped hole are never created.
pub fn add_face_py(m: &mut Mesh, outer: &[V3], holes: &[Vec<V3>], h: Hypot) -> FaceId {
    if holes.len() > 1 {
        let keep = maximal_holes(holes, h);
        if keep.len() != holes.len() {
            let kept: Vec<Vec<V3>> = keep.iter().map(|&i| holes[i].clone()).collect();
            return m.add_face(outer, &kept);
        }
    }
    m.add_face(outer, holes)
}

/// `Mesh._link_edge` over the public API: the edge between two vertices,
/// made when missing. (A vertex that is not the weld registry's holder of
/// its cell would be swapped for the holder by `add_edge`: needs a public
/// `link_edge` on `Mesh`.)
pub fn link_edge(m: &mut Mesh, a: VertexId, b: VertexId) -> EdgeId {
    if let Some(e) = m.find_edge(a, b) {
        return e;
    }
    let (pa, pb) = (m.position(a), m.position(b));
    let e = m.add_edge(pa, pb).expect("link_edge: endpoints weld to one vertex");
    debug_assert!({
        let ed = m.edge(e);
        (ed.v0, ed.v1) == (a, b)
    });
    e
}

/// `Mesh.split_edge_at`: split `e` at the existing vertex `mid` on its
/// interior; returns the two halves.
pub fn split_edge_at(m: &mut Mesh, e: EdgeId, mid: VertexId) -> (EdgeId, EdgeId) {
    let (v0, v1) = (m.edge(e).v0, m.edge(e).v1);
    let incident = m.edge(e).faces.clone();
    let (soft, layer, curve) = {
        let ed = m.edge(e);
        (ed.soft, ed.layer.clone(), ed.curve)
    };
    m.remove_edge(e);
    let e0_new = m.find_edge(v0, mid).is_none();
    let e0 = link_edge(m, v0, mid);
    let e1_new = m.find_edge(mid, v1).is_none();
    let e1 = link_edge(m, mid, v1);
    for (sub, was_new) in [(e0, e0_new), (e1, e1_new)] {
        if was_new {
            let ed = m.edge_mut(sub);
            ed.soft = soft;
            ed.layer = layer.clone();
            ed.curve = curve;
        }
    }
    for f in incident {
        insert_into_face_loops(m, f, v0, v1, mid);
        for half in [e0, e1] {
            if !m.edge(half).faces.contains(&f) {
                m.edge_mut(half).faces.push(f);
            }
        }
    }
    (e0, e1)
}

/// `Mesh._insert_into_face_loops`: `mid` goes between `v0` and `v1` in the
/// first loop of the face (outer, then holes) that has that side.
fn insert_into_face_loops(m: &mut Mesh, f: FaceId, v0: VertexId, v1: VertexId, mid: VertexId) {
    let face = m.face_mut(f);
    for l in std::iter::once(&mut face.outer).chain(face.holes.iter_mut()) {
        let n = l.len();
        for i in 0..n {
            let j = (i + 1) % n;
            if (l[i] == v0 && l[j] == v1) || (l[i] == v1 && l[j] == v0) {
                l.insert(i + 1, mid);
                return;
            }
        }
    }
}

fn shared_edge_index(l: &[VertexId], v0: VertexId, v1: VertexId) -> Option<usize> {
    let n = l.len();
    (0..n).find(|&i| {
        let (a, b) = (l[i], l[(i + 1) % n]);
        (a == v0 && b == v1) || (a == v1 && b == v0)
    })
}

/// `core.mesh._splice_loops`.
fn splice_loops(loop_a: &[VertexId], loop_b: &[VertexId], v0: VertexId, v1: VertexId) -> Option<Vec<VertexId>> {
    let ia = shared_edge_index(loop_a, v0, v1)?;
    let ib = shared_edge_index(loop_b, v0, v1)?;
    let (na, nb) = (loop_a.len(), loop_b.len());
    let (u, w) = (loop_a[ia], loop_a[(ia + 1) % na]);
    if !(loop_b[ib] == w && loop_b[(ib + 1) % nb] == u) {
        return None;
    }
    let sa = (ia + 1) % na;
    let mut a_path: Vec<VertexId> = loop_a[sa..].to_vec();
    a_path.extend_from_slice(&loop_a[..sa]);
    let sb = (ib + 1) % nb;
    let mut b_path: Vec<VertexId> = loop_b[sb..].to_vec();
    b_path.extend_from_slice(&loop_b[..sb]);
    let mut merged: Vec<VertexId> = a_path[..a_path.len() - 1].to_vec();
    merged.extend_from_slice(&b_path[..b_path.len() - 1]);
    let distinct: HashSet<VertexId> = merged.iter().copied().collect();
    if distinct.len() != merged.len() {
        return None;
    }
    Some(merged)
}

/// `core.mesh._dominant_attrs`: the attrs of the largest face that has any.
fn dominant_attrs(m: &Mesh, faces: &[FaceId], h: Hypot) -> Vec<(String, ingetrazo_formats::pyjson::Value)> {
    let mut best = Vec::new();
    let mut best_area = -1.0f64;
    for &f in faces {
        let a = PFace::of(m, f).area(h);
        if !m.face(f).attrs.is_empty() && a > best_area {
            best = m.face(f).attrs.clone();
            best_area = a;
        }
    }
    best
}

/// `Mesh.dissolve_edge`: merge the two coplanar faces of `e` into one.
pub fn dissolve_edge(m: &mut Mesh, e: EdgeId, h: Hypot) -> Option<FaceId> {
    let fs = m.edge(e).faces.clone();
    if fs.len() != 2 {
        return None;
    }
    let (fa, fb) = (fs[0], fs[1]);
    if fa == fb {
        return None;
    }
    let dot = dotf(PFace::of(m, fa).normal(h), PFace::of(m, fb).normal(h));
    if dot.abs() < 0.999 {
        return None;
    }
    let mut loop_b = m.face(fb).outer.clone();
    if !(dot > 0.0) {
        loop_b.reverse();
    }
    let (v0, v1) = (m.edge(e).v0, m.edge(e).v1);
    let merged = splice_loops(&m.face(fa).outer, &loop_b, v0, v1)?;
    let loop_positions = positions(m, &merged);
    let hole_positions: Vec<Vec<V3>> =
        m.face(fa).holes.iter().chain(m.face(fb).holes.iter()).map(|hl| positions(m, hl)).collect();
    let attrs = dominant_attrs(m, &[fa, fb], h);
    m.remove_face(fa);
    m.remove_face(fb);
    m.remove_edge(e);
    let merged_face = add_face_py(m, &loop_positions, &hole_positions, h);
    m.face_mut(merged_face).attrs = attrs;
    Some(merged_face)
}

// ---- Position helpers ----------------------------------------------------------

/// `same_position`: two points coincide within the welding tolerance.
pub fn same_position(p: V3, q: V3) -> bool {
    key(p) == key(q)
}

/// `find_duplicate_edge`: the index of an edge whose endpoints coincide with
/// segment `a`-`b`, in either orientation.
pub fn find_duplicate_edge(edges: &[Seg], a: V3, b: V3) -> Option<usize> {
    let (ka, kb) = (key(a), key(b));
    if ka == kb {
        return None;
    }
    let target = ekey(ka, kb);
    for (i, &(p, q)) in edges.iter().enumerate() {
        let (ea, eb) = (key(p), key(q));
        if ea == eb {
            continue;
        }
        if ekey(ea, eb) == target {
            return Some(i);
        }
    }
    None
}

// ---- Cycles in the edge graph ---------------------------------------------------

/// One entry of an adjacency list: the neighbour's key and position.
pub type Nb = (Key, V3);

/// What `_smallest_cycle` needs of an adjacency: `in` and `get`.
pub trait Adjacency {
    /// The neighbours of `u`; `None` when `u` is absent or has none.
    fn get(&self, u: &Key) -> Option<Cow<'_, [Nb]>>;
}

/// `_cycle_adjacency`'s dict.
pub struct PlainAdj(pub HashMap<Key, Vec<Nb>>);

impl Adjacency for PlainAdj {
    fn get(&self, u: &Key) -> Option<Cow<'_, [Nb]>> {
        self.0.get(u).filter(|l| !l.is_empty()).map(|l| Cow::Borrowed(&l[..]))
    }
}

/// `_cycle_adjacency`: `adj[u] -> [(v_key, v_pos), ...]` of the edge graph,
/// in edge order, without an existing copy of segment `ka`-`kb`.
pub fn cycle_adjacency(edges: &[Seg], ka: Key, kb: Key) -> PlainAdj {
    let mut adj: HashMap<Key, Vec<Nb>> = HashMap::new();
    let target = ekey(ka, kb);
    for &(a, b) in edges {
        let (ea, eb) = (key(a), key(b));
        if ea == eb {
            continue;
        }
        if ekey(ea, eb) == target {
            continue;
        }
        adj.entry(ea).or_default().push((eb, b));
        adj.entry(eb).or_default().push((ea, a));
    }
    PlainAdj(adj)
}

/// `CycleGraph`: the adjacency of an edge list kept up to date edge by edge.
#[derive(Default)]
pub struct CycleGraph {
    adj: HashMap<Key, Vec<(u64, Key, V3)>>,
    entries: HashMap<EdgeId, (Key, u64, Key, u64)>,
    serial: u64,
}

impl CycleGraph {
    pub fn new() -> CycleGraph {
        CycleGraph::default()
    }

    pub fn from_edges(edges: &[(EdgeId, V3, V3)]) -> CycleGraph {
        let mut g = CycleGraph::new();
        for &(e, a, b) in edges {
            g.add(e, a, b);
        }
        g
    }

    /// `CycleGraph.add(edge)` with the edge's end positions.
    pub fn add(&mut self, edge: EdgeId, a: V3, b: V3) {
        let (ea, eb) = (key(a), key(b));
        if ea == eb {
            return;
        }
        let (xid, yid) = (self.serial, self.serial + 1);
        self.serial += 2;
        self.adj.entry(ea).or_default().push((xid, eb, b));
        self.adj.entry(eb).or_default().push((yid, ea, a));
        self.entries.insert(edge, (ea, xid, eb, yid));
    }

    /// `CycleGraph.remove(edge)`.
    pub fn remove(&mut self, edge: EdgeId) {
        let Some((ea, xid, eb, yid)) = self.entries.remove(&edge) else {
            return;
        };
        for (k, id) in [(ea, xid), (eb, yid)] {
            let lst = self.adj.get_mut(&k).expect("CycleGraph: node missing");
            if let Some(i) = lst.iter().position(|it| it.0 == id) {
                lst.remove(i);
            }
            if lst.is_empty() {
                self.adj.remove(&k);
            }
        }
    }

    /// The neighbours of a node, in edge order.
    pub fn neighbours(&self, k: &Key) -> Vec<Nb> {
        self.adj.get(k).map(|l| l.iter().map(|&(_, k, p)| (k, p)).collect()).unwrap_or_default()
    }
}

/// `_AdjWithout`: a graph's adjacency without segment `ka`-`kb`.
pub struct AdjWithout<'a> {
    graph: &'a CycleGraph,
    ka: Key,
    kb: Key,
}

impl<'a> AdjWithout<'a> {
    pub fn new(graph: &'a CycleGraph, ka: Key, kb: Key) -> AdjWithout<'a> {
        AdjWithout { graph, ka, kb }
    }
}

impl Adjacency for AdjWithout<'_> {
    fn get(&self, u: &Key) -> Option<Cow<'_, [Nb]>> {
        let lst = self.graph.adj.get(u)?;
        let out: Vec<Nb> = if *u == self.ka {
            lst.iter().filter(|e| e.1 != self.kb).map(|&(_, k, p)| (k, p)).collect()
        } else if *u == self.kb {
            lst.iter().filter(|e| e.1 != self.ka).map(|&(_, k, p)| (k, p)).collect()
        } else {
            lst.iter().map(|&(_, k, p)| (k, p)).collect()
        };
        if out.is_empty() {
            None
        } else {
            Some(Cow::Owned(out))
        }
    }
}

/// `_smallest_cycle`: breadth-first search from `b` to `a` over `adj`, never
/// entering a node of `skip`.
pub fn smallest_cycle(
    adj: &dyn Adjacency,
    b: V3,
    ka: Key,
    kb: Key,
    max_len: usize,
    skip: &HashSet<Key>,
) -> Option<Vec<V3>> {
    let la = adj.get(&ka)?;
    let lb = adj.get(&kb)?;
    if !skip.is_empty() {
        let a_ok = la.iter().any(|(v, _)| !skip.contains(v));
        let b_ok = lb.iter().any(|(v, _)| !skip.contains(v));
        if !(a_ok && b_ok) {
            return None;
        }
    }
    let mut parent: HashMap<Key, Option<Key>> = HashMap::new();
    let mut parent_pos: HashMap<Key, V3> = HashMap::new();
    parent.insert(kb, None);
    parent_pos.insert(kb, b);
    let mut q: VecDeque<Key> = VecDeque::new();
    q.push_back(kb);
    let mut found = false;
    while let Some(u) = q.pop_front() {
        if u == ka {
            found = true;
            break;
        }
        if let Some(nbs) = adj.get(&u) {
            for &(vk, vp) in nbs.iter() {
                if !parent.contains_key(&vk) && !skip.contains(&vk) {
                    parent.insert(vk, Some(u));
                    parent_pos.insert(vk, vp);
                    q.push_back(vk);
                }
            }
        }
    }
    if !found {
        return None;
    }
    let mut path = Vec::new();
    let mut cur = Some(ka);
    while let Some(c) = cur {
        path.push(parent_pos[&c]);
        cur = parent[&c];
    }
    if path.len() < 3 || path.len() > max_len {
        return None;
    }
    Some(path)
}

/// `find_smallest_cycle_through`: the smallest simple cycle in the edge graph
/// that contains segment `a`-`b` (which need not exist yet).
pub fn find_smallest_cycle_through(edges: &[Seg], a: V3, b: V3, max_len: usize) -> Option<Vec<V3>> {
    let (ka, kb) = (key(a), key(b));
    if ka == kb {
        return None;
    }
    smallest_cycle(&cycle_adjacency(edges, ka, kb), b, ka, kb, max_len, &HashSet::new())
}

fn same_cycle(c1: &[V3], c2: &[V3]) -> bool {
    keyset(c1) == keyset(c2)
}

fn cycles_over(adj: &dyn Adjacency, a: V3, b: V3, max_results: usize) -> Vec<Vec<V3>> {
    let (ka, kb) = (key(a), key(b));
    let Some(first) = smallest_cycle(adj, b, ka, kb, 32, &HashSet::new()) else {
        return Vec::new();
    };
    let mut cycles = vec![first];
    if max_results >= 2 {
        let mut interior = keyset(&cycles[0]);
        interior.remove(&ka);
        interior.remove(&kb);
        if !interior.is_empty() {
            if let Some(second) = smallest_cycle(adj, b, ka, kb, 32, &interior) {
                if !same_cycle(&second, &cycles[0]) {
                    cycles.push(second);
                }
            }
        }
    }
    cycles
}

/// `find_cycles_through(edges, a, b, max_results)`: up to `max_results`
/// distinct minimal cycles through segment `a`-`b`.
pub fn find_cycles_through(edges: &[Seg], a: V3, b: V3, max_results: usize) -> Vec<Vec<V3>> {
    let (ka, kb) = (key(a), key(b));
    if ka == kb {
        return Vec::new();
    }
    cycles_over(&cycle_adjacency(edges, ka, kb), a, b, max_results)
}

/// `find_cycles_through(..., graph=graph)`: the planner's live adjacency.
pub fn find_cycles_through_graph(graph: &CycleGraph, a: V3, b: V3, max_results: usize) -> Vec<Vec<V3>> {
    let (ka, kb) = (key(a), key(b));
    if ka == kb {
        return Vec::new();
    }
    cycles_over(&AdjWithout::new(graph, ka, kb), a, b, max_results)
}

// ---- Planarity and intersection ---------------------------------------------------

/// `is_planar`: whether `vertices` all lie on a common plane within
/// `tolerance`.
pub fn is_planar(vertices: &[V3], tolerance: f64, h: Hypot) -> bool {
    let n = vertices.len();
    if n < 3 {
        return false;
    }
    if n == 3 {
        let e1 = vertices[1].sub(vertices[0]);
        let e2 = vertices[2].sub(vertices[0]);
        return lenf(e1.cross(e2), h) > 1e-6;
    }
    let v0 = vertices[0];
    let mut plane_normal: Option<V3> = None;
    'outer: for i in 1..n - 1 {
        for j in i + 1..n {
            let cross = vertices[i].sub(v0).cross(vertices[j].sub(v0));
            if lenf(cross, h) > 1e-6 {
                plane_normal = Some(cross.normalized(h));
                break 'outer;
            }
        }
    }
    let Some(pn) = plane_normal else {
        return false;
    };
    for &v in vertices {
        if dotf(pn, v.sub(v0)).abs() > tolerance {
            return false;
        }
    }
    true
}

/// `segment_intersection`: where segment `p1-p2` meets segment `p3-p4` in 3D.
pub fn segment_intersection(p1: V3, p2: V3, p3: V3, p4: V3, tol: f64, h: Hypot) -> Option<V3> {
    let d1 = p2.sub(p1);
    let d2 = p4.sub(p3);
    let len1 = lenf(d1, h);
    let len2 = lenf(d2, h);
    if len1 < tol || len2 < tol {
        return None;
    }
    let components = |v: V3| [v.0 as f64, v.1 as f64, v.2 as f64];
    let d1c = components(d1);
    let d2c = components(d2);
    let u1 = [d1c[0] / len1, d1c[1] / len1, d1c[2] / len1];
    let u2 = [d2c[0] / len2, d2c[1] / len2, d2c[2] / len2];
    let dot = |l: [f64; 3], r: [f64; 3]| py_sum((0..3).map(|i| l[i] * r[i]), true);
    let a = dot(u1, u1);
    let b = dot(u1, u2);
    let c = dot(u2, u2);
    let w0 = p1.sub(p3);
    let w = components(w0);
    let d = dot(u1, w);
    let e = dot(u2, w);
    let denom = a * c - b * b;
    if denom < 1e-12 {
        return None;
    }
    let s = (b * e - c * d) / denom;
    let t = (a * e - b * d) / denom;
    let margin1 = tol / len1;
    let margin2 = tol / len2;
    if !(-margin1 <= s / len1 && s / len1 <= 1.0 + margin1) {
        return None;
    }
    if !(-margin2 <= t / len2 && t / len2 <= 1.0 + margin2) {
        return None;
    }
    let p1c = components(p1);
    let p3c = components(p3);
    let point_on_1: [f64; 3] = std::array::from_fn(|i| p1c[i] + u1[i] * s);
    let point_on_2: [f64; 3] = std::array::from_fn(|i| p3c[i] + u2[i] * t);
    let gap = pow_half(py_sum((0..3).map(|i| pow2(point_on_1[i] - point_on_2[i])), true));
    let distance_tol = py_max2(tol, 1.0e-7 * py_max2(len1, len2));
    if gap > distance_tol {
        return None;
    }
    let m: [f64; 3] = std::array::from_fn(|i| (point_on_1[i] + point_on_2[i]) * 0.5);
    Some(V3(m[0] as f32, m[1] as f32, m[2] as f32))
}

/// `_order_along`: deduplicate `points` and order them by their projection
/// along a->b, dropping any that coincide with an endpoint.
pub fn order_along(a: V3, b: V3, points: &[V3], h: Hypot) -> Vec<V3> {
    let d = b.sub(a);
    let mut uniq: Vec<V3> = Vec::new();
    for &p in points {
        if lenf(p.sub(a), h) <= SPLIT_TOLERANCE || lenf(p.sub(b), h) <= SPLIT_TOLERANCE {
            continue;
        }
        if !uniq.iter().any(|&q| lenf(p.sub(q), h) <= SPLIT_TOLERANCE) {
            uniq.push(p);
        }
    }
    let mut keyed: Vec<(f32, V3)> = uniq.iter().map(|&p| (p.sub(a).dot(d), p)).collect();
    keyed.sort_by(|x, y| x.0.partial_cmp(&y.0).unwrap_or(std::cmp::Ordering::Equal));
    keyed.into_iter().map(|(_, p)| p).collect()
}

/// `plan_edge_split`: the new edge broken at every interior crossing (ordered
/// from `a` to `b`), and, per crossed existing edge (by index), the interior
/// point where the new edge cuts it.
pub fn plan_edge_split(edges: &[Seg], a: V3, b: V3, h: Hypot) -> (Vec<Seg>, Vec<(usize, V3)>) {
    let mut new_cuts: Vec<V3> = Vec::new();
    let mut edge_cuts: Vec<(usize, V3)> = Vec::new();
    for (i, &(ea, eb)) in edges.iter().enumerate() {
        let Some(mut point) = segment_intersection(a, b, ea, eb, SPLIT_TOLERANCE, h) else {
            continue;
        };
        for anchor in [ea, eb, a, b] {
            if lenf(point.sub(anchor), h) <= SPLIT_TOLERANCE {
                point = anchor;
                break;
            }
        }
        if !(same_position(point, a) || same_position(point, b)) {
            new_cuts.push(point);
        }
        if !(same_position(point, ea) || same_position(point, eb)) {
            edge_cuts.push((i, point));
        }
    }
    let ordered = order_along(a, b, &new_cuts, h);
    let mut chain = vec![a];
    chain.extend(ordered);
    chain.push(b);
    let new_segments = (0..chain.len() - 1).map(|i| (chain[i], chain[i + 1])).collect();
    (new_segments, edge_cuts)
}

/// `face_exists`: a face with the same vertex set as `cycle` already exists.
pub fn face_exists(m: &Mesh, faces: &[FaceId], cycle: &[V3]) -> bool {
    let cycle_keys = keyset(cycle);
    faces.iter().any(|&f| keyset(&positions(m, &m.face(f).outer)) == cycle_keys)
}

// ---- Containment (face split / hole punching) ------------------------------------

type P2 = (f64, f64);

/// `_on_segment_2d`: 2D point `p` lies on segment `a`-`b` within `tol`.
pub fn on_segment_2d(p: P2, a: P2, b: P2, tol: f64) -> bool {
    let cross = (b.0 - a.0) * (p.1 - a.1) - (b.1 - a.1) * (p.0 - a.0);
    if cross.abs() > tol {
        return false;
    }
    let dot = (p.0 - a.0) * (b.0 - a.0) + (p.1 - a.1) * (b.1 - a.1);
    if dot < -tol {
        return false;
    }
    let sqlen = pow2(b.0 - a.0) + pow2(b.1 - a.1);
    dot <= sqlen + tol
}

/// `_strictly_inside_2d`: ray-cast point-in-polygon; boundary points are not
/// inside.
pub fn strictly_inside_2d(p: P2, poly: &[P2]) -> bool {
    let n = poly.len();
    let mut j = n.wrapping_sub(1);
    let mut inside = false;
    for i in 0..n {
        if on_segment_2d(p, poly[i], poly[j], 1e-7) {
            return false;
        }
        let (xi, yi) = poly[i];
        let (xj, yj) = poly[j];
        if (yi > p.1) != (yj > p.1) {
            let xint = xi + (xj - xi) * (p.1 - yi) / (yj - yi);
            if p.0 < xint {
                inside = !inside;
            }
        }
        j = i;
    }
    inside
}

/// `_point_inside_2d`: ray-cast point-in-polygon, boundary unspecified.
pub fn point_inside_2d(pt: P2, poly: &[P2]) -> bool {
    let (x, y) = pt;
    let mut inside = false;
    let n = poly.len();
    let mut j = n.wrapping_sub(1);
    for i in 0..n {
        let (xi, yi) = poly[i];
        let (xj, yj) = poly[j];
        if ((yi > y) != (yj > y)) && (x < (xj - xi) * (y - yi) / (yj - yi) + xi) {
            inside = !inside;
        }
        j = i;
    }
    inside
}

/// `proj(p)` of the face functions: `p` in the plane's (u, w) axes.
fn proj_fn(origin: V3, u: V3, w: V3) -> impl Fn(V3) -> P2 {
    move |p: V3| {
        let rel = p.sub(origin);
        (dotf(rel, u), dotf(rel, w))
    }
}

/// `loop_inside_face`: `loop_` lies entirely, strictly inside coplanar face
/// `mother`; with `outside_holes` a vertex in one of her holes is not hers.
pub fn loop_inside_face(mother: &PFace, loop_: &[V3], outside_holes: bool, h: Hypot) -> bool {
    if mother.vertices.len() < 3 || loop_.len() < 3 {
        return false;
    }
    let normal = mother.normal(h);
    let origin = mother.vertices[0];
    for &v in loop_ {
        if dotf(normal, v.sub(origin)).abs() > PLANAR_TOLERANCE {
            return false;
        }
    }
    let (u, w) = plane_axes(normal, h);
    let proj = proj_fn(origin, u, w);
    let poly2: Vec<P2> = mother.vertices.iter().map(|&p| proj(p)).collect();
    if !loop_.iter().all(|&v| strictly_inside_2d(proj(v), &poly2)) {
        return false;
    }
    if outside_holes {
        for hole in &mother.holes {
            let hole2: Vec<P2> = hole.iter().map(|&p| proj(p)).collect();
            if loop_.iter().any(|&v| strictly_inside_2d(proj(v), &hole2)) {
                return false;
            }
        }
    }
    true
}

/// `find_containing_face`: the smallest (by vertex count) face of `faces`
/// that strictly contains `loop_`, hole-aware.
pub fn find_containing_face(
    m: &Mesh,
    faces: &[FaceId],
    loop_: &[V3],
    exclude: Option<FaceId>,
    h: Hypot,
) -> Option<FaceId> {
    let mut best: Option<FaceId> = None;
    for &f in faces {
        if Some(f) == exclude {
            continue;
        }
        if loop_inside_face(&PFace::of(m, f), loop_, true, h)
            && (best.is_none() || m.face(f).outer.len() < m.face(best.unwrap()).outer.len())
        {
            best = Some(f);
        }
    }
    best
}

fn loop_edges(l: &[V3]) -> Vec<(Key, Key)> {
    let n = l.len();
    (0..n).map(|i| ekey(key(l[i]), key(l[(i + 1) % n]))).collect()
}

/// `orphaned_edges_at`: the edges incident to any of `vertices` that border
/// no face (indices into `edges`, in order).
pub fn orphaned_edges_at(m: &Mesh, edges: &[EdgeId], faces: &[FaceId], vertices: &[V3]) -> Vec<EdgeId> {
    let vkeys = keyset(vertices);
    let mut face_edges: HashSet<(Key, Key)> = HashSet::new();
    for &f in faces {
        let face = PFace::of(m, f);
        let loops: Vec<Vec<Key>> =
            std::iter::once(&face.vertices).chain(face.holes.iter()).map(|lp| lp.iter().map(|&p| key(p)).collect()).collect();
        if !loops.iter().any(|lk| lk.iter().any(|k| vkeys.contains(k))) {
            continue;
        }
        for lk in &loops {
            let n = lk.len();
            for i in 0..n {
                face_edges.insert(ekey(lk[i], lk[(i + 1) % n]));
            }
        }
    }
    let mut out = Vec::new();
    for &e in edges {
        let ed = m.edge(e);
        let (ka, kb) = (key(m.position(ed.v0)), key(m.position(ed.v1)));
        if (vkeys.contains(&ka) || vkeys.contains(&kb)) && !face_edges.contains(&ekey(ka, kb)) {
            out.push(e);
        }
    }
    out
}

/// `_faces_coplanar`.
pub fn faces_coplanar(n1: V3, n2: V3, h: Hypot) -> bool {
    dotf(n1.normalized(h), n2.normalized(h)).abs() > 0.999
}

/// `_unit_coplanar`: for normals already of unit length (the products are
/// Python floats, summed left to right).
pub fn unit_coplanar(n1: V3, n2: V3) -> bool {
    (n1.0 as f64 * n2.0 as f64 + n1.1 as f64 * n2.1 as f64 + n1.2 as f64 * n2.2 as f64).abs() > 0.999
}

fn face_all_edges(f: &PFace) -> HashSet<(Key, Key)> {
    let mut edges: HashSet<(Key, Key)> = loop_edges(&f.vertices).into_iter().collect();
    for hole in &f.holes {
        edges.extend(loop_edges(hole));
    }
    edges
}

/// `_point_on_seg_incl`: `pt` lies on segment `p`-`q`, endpoints included.
pub fn point_on_seg_incl(pt: V3, p: V3, q: V3, tol: f64, h: Hypot) -> bool {
    let pq = q.sub(p);
    let length = lenf(pq, h);
    if length < tol {
        return same_position(pt, p);
    }
    let t = dotf(pt.sub(p), pq) / (length * length);
    if t < -tol || t > 1.0 + tol {
        return false;
    }
    lenf(pt.sub(p.add(pq.mul(t as f32))), h) < tol
}

/// `_segment_on_face_boundary`: segment `a`-`b` lies on a boundary edge of
/// `face` (outer loop or a hole).
pub fn segment_on_face_boundary(a: V3, b: V3, face: &PFace, h: Hypot) -> bool {
    for lp in std::iter::once(&face.vertices).chain(face.holes.iter()) {
        let n = lp.len();
        for i in 0..n {
            let p = lp[i];
            let q = lp[(i + 1) % n];
            if point_on_seg_incl(a, p, q, SPLIT_TOLERANCE, h) && point_on_seg_incl(b, p, q, SPLIT_TOLERANCE, h) {
                return true;
            }
        }
    }
    false
}

/// `refine_loop_with_points`: insert any `points` that lie on an edge's
/// interior into `loop_`.
pub fn refine_loop_with_points(loop_: &[V3], points: &[V3], tol: f64, h: Hypot) -> Vec<V3> {
    let mut out: Vec<V3> = Vec::new();
    let n = loop_.len();
    let xyz: Vec<[f64; 3]> = points.iter().map(|p| [p.0 as f64, p.1 as f64, p.2 as f64]).collect();
    for i in 0..n {
        let a = loop_[i];
        let b = loop_[(i + 1) % n];
        out.push(a);
        let ab = b.sub(a);
        let length = lenf(ab, h);
        if length < tol {
            continue;
        }
        let (ax, ay, az) = (a.0 as f64, a.1 as f64, a.2 as f64);
        let (bx, by, bz) = (b.0 as f64, b.1 as f64, b.2 as f64);
        let mx = [ax.abs(), ay.abs(), az.abs(), bx.abs(), by.abs(), bz.abs()]
            .into_iter()
            .reduce(py_max2)
            .unwrap_or(0.0);
        let g = 2.0 * tol + 1e-6 * mx;
        let lo = [py_min2(ax, bx) - g, py_min2(ay, by) - g, py_min2(az, bz) - g];
        let hi = [py_max2(ax, bx) + g, py_max2(ay, by) + g, py_max2(az, bz) + g];
        let mut on: Vec<(f64, V3)> = Vec::new();
        for (j, c) in xyz.iter().enumerate() {
            if !((0..3).all(|k| c[k] >= lo[k] && c[k] <= hi[k])) {
                continue;
            }
            let p = points[j];
            if same_position(p, a) || same_position(p, b) {
                continue;
            }
            let t = dotf(p.sub(a), ab) / (length * length);
            if t <= tol / length || t >= 1.0 - tol / length {
                continue;
            }
            if lenf(p.sub(a.add(ab.mul(t as f32))), h) < tol {
                on.push((t, p));
            }
        }
        on.sort_by(|x, y| x.0.partial_cmp(&y.0).unwrap_or(std::cmp::Ordering::Equal));
        for (_, p) in on {
            out.push(p);
        }
    }
    out
}

/// How a push/pull side edge attaches to the model.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PushEdge {
    /// A face on the same plane shares this edge.
    Coplanar(FaceId),
    /// A non-coplanar face carries this edge on its boundary.
    Perp(FaceId),
    /// Nothing adjacent.
    Free,
}

/// `classify_push_edge`.
pub fn classify_push_edge(m: &Mesh, face: FaceId, a: V3, b: V3, faces: &[FaceId], h: Hypot) -> PushEdge {
    let fn_ = PFace::of(m, face).normal(h);
    let eset = ekey(key(a), key(b));
    for &g in faces {
        if g == face {
            continue;
        }
        let pg = PFace::of(m, g);
        if faces_coplanar(fn_, pg.normal(h), h) && (face_all_edges(&pg).contains(&eset) || segment_on_face_boundary(a, b, &pg, h)) {
            return PushEdge::Coplanar(g);
        }
    }
    for &s in faces {
        if s == face {
            continue;
        }
        let ps = PFace::of(m, s);
        if faces_coplanar(fn_, ps.normal(h), h) {
            continue;
        }
        if segment_on_face_boundary(a, b, &ps, h) {
            return PushEdge::Perp(s);
        }
    }
    PushEdge::Free
}

/// `face_is_bordered`: every boundary edge of `face` is also an edge of some
/// other face (its boundary or a hole).
pub fn face_is_bordered(m: &Mesh, face: FaceId, faces: &[FaceId]) -> bool {
    let base_edges = loop_edges(&positions(m, &m.face(face).outer));
    if base_edges.is_empty() {
        return false;
    }
    let mut others: HashSet<(Key, Key)> = HashSet::new();
    for &f in faces {
        if f == face {
            continue;
        }
        let pf = PFace::of(m, f);
        others.extend(loop_edges(&pf.vertices));
        for hole in &pf.holes {
            others.extend(loop_edges(hole));
        }
    }
    base_edges.iter().all(|e| others.contains(e))
}

// ---- Chord split (a new edge divides an existing face) -----------------------------

/// `_point_on_segment_3d`: `p` lies on the interior of segment `a`-`b`.
pub fn point_on_segment_3d(p: V3, a: V3, b: V3, tol: f64, h: Hypot) -> bool {
    let ab = b.sub(a);
    let length = lenf(ab, h);
    if length < tol {
        return false;
    }
    let t = dotf(p.sub(a), ab) / (length * length);
    if t < tol || t > 1.0 - tol {
        return false;
    }
    lenf(p.sub(a.add(ab.mul(t as f32))), h) < tol
}

/// Where a point sits on a face's boundary loop.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Loc {
    /// It is vertex `i`.
    Vertex(usize),
    /// It lies on edge `i -> i+1`.
    Edge(usize),
}

/// `_locate_on_loop`.
pub fn locate_on_loop(vertices: &[V3], p: V3, h: Hypot) -> Option<Loc> {
    let kp = key(p);
    for (i, &v) in vertices.iter().enumerate() {
        if key(v) == kp {
            return Some(Loc::Vertex(i));
        }
    }
    let n = vertices.len();
    for i in 0..n {
        if point_on_segment_3d(p, vertices[i], vertices[(i + 1) % n], SPLIT_TOLERANCE, h) {
            return Some(Loc::Edge(i));
        }
    }
    None
}

/// The loop with the on-edge endpoints inserted in order (the `aug` list of
/// `split_face_by_chord` and `split_face_by_chain`).
fn augment(vertices: &[V3], on_edge: &HashMap<usize, Vec<V3>>, h: Hypot) -> Vec<V3> {
    let mut aug: Vec<V3> = Vec::new();
    for (i, &v) in vertices.iter().enumerate() {
        aug.push(v);
        if let Some(pts) = on_edge.get(&i) {
            let mut keyed: Vec<(f32, V3)> = pts.iter().map(|&q| (q.sub(v).length(h), q)).collect();
            keyed.sort_by(|x, y| x.0.partial_cmp(&y.0).unwrap_or(std::cmp::Ordering::Equal));
            aug.extend(keyed.into_iter().map(|(_, q)| q));
        }
    }
    aug
}

/// `split_face_by_chord`: the two sub-loops segment `a`-`b` divides `face`
/// into, when it is a chord of it.
pub fn split_face_by_chord(face: &PFace, a: V3, b: V3, h: Hypot) -> Option<(Vec<V3>, Vec<V3>)> {
    if !face.holes.is_empty() || face.vertices.len() < 3 {
        return None;
    }
    let la = locate_on_loop(&face.vertices, a, h)?;
    let lb = locate_on_loop(&face.vertices, b, h)?;
    let mut on_edge: HashMap<usize, Vec<V3>> = HashMap::new();
    if let Loc::Edge(i) = la {
        on_edge.entry(i).or_default().push(a);
    }
    if let Loc::Edge(i) = lb {
        on_edge.entry(i).or_default().push(b);
    }
    let aug = augment(&face.vertices, &on_edge, h);
    let keys: Vec<Key> = aug.iter().map(|&v| key(v)).collect();
    let mut ia = keys.iter().position(|&k| k == key(a)).expect("chord end is in the loop");
    let mut ib = keys.iter().position(|&k| k == key(b)).expect("chord end is in the loop");
    if ia > ib {
        std::mem::swap(&mut ia, &mut ib);
    }
    let m = aug.len();
    if ib - ia <= 1 || (ia == 0 && ib == m - 1) {
        return None;
    }
    let normal = face.normal(h);
    let origin = face.vertices[0];
    let (u, w) = plane_axes(normal, h);
    let proj = proj_fn(origin, u, w);
    let mid = a.add(b).mul(0.5);
    let poly2: Vec<P2> = face.vertices.iter().map(|&v| proj(v)).collect();
    if !strictly_inside_2d(proj(mid), &poly2) {
        return None;
    }
    let loop_a: Vec<V3> = aug[ia..=ib].to_vec();
    let mut loop_b: Vec<V3> = aug[ib..].to_vec();
    loop_b.extend_from_slice(&aug[..=ia]);
    if loop_a.len() < 3 || loop_b.len() < 3 {
        return None;
    }
    Some((loop_a, loop_b))
}

/// `find_chord_split`: the first face (and its two halves) that segment
/// `a`-`b` chord-splits.
pub fn find_chord_split(m: &Mesh, faces: &[FaceId], a: V3, b: V3, h: Hypot) -> Option<(FaceId, Vec<V3>, Vec<V3>)> {
    for &f in faces {
        if let Some((x, y)) = split_face_by_chord(&PFace::of(m, f), a, b, h) {
            return Some((f, x, y));
        }
    }
    None
}

/// `_loop_with_point`.
fn loop_with_point(loop_: &[V3], ka: Key, kb: Key, point: V3) -> Option<Vec<V3>> {
    let kp = key(point);
    let n = loop_.len();
    for i in 0..n {
        let j = (i + 1) % n;
        let (ki, kj) = (key(loop_[i]), key(loop_[j]));
        if ekey(ki, kj) == ekey(ka, kb) {
            if kp == ki || kp == kj {
                return None;
            }
            let mut new = loop_.to_vec();
            new.insert(i + 1, point);
            return Some(new);
        }
    }
    None
}

/// `split_edge_in_faces`: propagate an edge split into the faces that share
/// that edge on their outer boundary; `(face, new_vertices)` for those that
/// changed.
pub fn split_edge_in_faces(
    m: &Mesh,
    faces: &[FaceId],
    edge_a: V3,
    edge_b: V3,
    point: V3,
    skip_endpoints: &[V3],
    h: Hypot,
) -> Vec<(FaceId, Vec<V3>)> {
    let (ka, kb) = (key(edge_a), key(edge_b));
    let mut out = Vec::new();
    for &f in faces {
        let verts = positions(m, &m.face(f).outer);
        if !skip_endpoints.is_empty() && skip_endpoints.iter().all(|&p| locate_on_loop(&verts, p, h).is_some()) {
            continue;
        }
        if let Some(nv) = loop_with_point(&verts, ka, kb, point) {
            out.push((f, nv));
        }
    }
    out
}

// ---- Loop subtraction (a face drawn against another's boundary) ----------------------

/// `find_subdividing_chain`: the interior chain `[P, ...interior..., Q]` of a
/// loop that shares a contiguous boundary arc with `face` and pushes one run
/// of vertices through its interior.
pub fn find_subdividing_chain(face: &PFace, loop_: &[V3], h: Hypot) -> Option<Vec<V3>> {
    if face.vertices.len() < 3 || loop_.len() < 3 {
        return None;
    }
    let normal = face.normal(h);
    let origin = face.vertices[0];
    let (u, w) = plane_axes(normal, h);
    let proj = proj_fn(origin, u, w);
    let poly2: Vec<P2> = face.vertices.iter().map(|&v| proj(v)).collect();
    // false: "bdry", true: "in"
    let mut labels: Vec<bool> = Vec::new();
    for &v in loop_ {
        if locate_on_loop(&face.vertices, v, h).is_some() {
            labels.push(false);
        } else if strictly_inside_2d(proj(v), &poly2) {
            labels.push(true);
        } else {
            return None;
        }
    }
    if !labels.contains(&true) || !labels.contains(&false) {
        return None;
    }
    let n = loop_.len();
    let start = labels.iter().position(|&l| !l).unwrap();
    let mut runs: Vec<Vec<usize>> = Vec::new();
    let mut cur: Vec<usize> = Vec::new();
    for k in 0..n {
        let idx = (start + k) % n;
        if labels[idx] {
            cur.push(idx);
        } else if !cur.is_empty() {
            runs.push(std::mem::take(&mut cur));
        }
    }
    if !cur.is_empty() {
        runs.push(cur);
    }
    if runs.len() != 1 {
        return None;
    }
    let run = &runs[0];
    let p = loop_[(run[0] + n - 1) % n];
    let q = loop_[(run[run.len() - 1] + 1) % n];
    if same_position(p, q) {
        return None;
    }
    let mut chain = vec![p];
    chain.extend(run.iter().map(|&i| loop_[i]));
    chain.push(q);
    Some(chain)
}

/// `_arc`: cyclic slice of `seq` from `i` to `j` inclusive, forward.
fn arc(seq: &[V3], i: usize, j: usize) -> Vec<V3> {
    if i <= j {
        seq[i..=j].to_vec()
    } else {
        let mut v = seq[i..].to_vec();
        v.extend_from_slice(&seq[..=j]);
        v
    }
}

/// `split_face_by_chain` (of a face given as its outer positions).
pub fn split_face_by_chain(vertices: &[V3], chain: &[V3], h: Hypot) -> Option<(Vec<V3>, Vec<V3>)> {
    let p = chain[0];
    let q = chain[chain.len() - 1];
    let middles: Vec<V3> = if chain.len() >= 2 { chain[1..chain.len() - 1].to_vec() } else { Vec::new() };
    let mut on_edge: HashMap<usize, Vec<V3>> = HashMap::new();
    for pt in [p, q] {
        let loc = locate_on_loop(vertices, pt, h)?;
        if let Loc::Edge(i) = loc {
            on_edge.entry(i).or_default().push(pt);
        }
    }
    let aug = augment(vertices, &on_edge, h);
    let keys: Vec<Key> = aug.iter().map(|&x| key(x)).collect();
    let ip = keys.iter().position(|&k| k == key(p)).expect("chain end is in the loop");
    let iq = keys.iter().position(|&k| k == key(q)).expect("chain end is in the loop");
    let mut region_pq = arc(&aug, ip, iq);
    region_pq.extend(middles.iter().rev().copied());
    let mut region_qp = arc(&aug, iq, ip);
    region_qp.extend(middles.iter().copied());
    if region_pq.len() < 3 || region_qp.len() < 3 {
        return None;
    }
    Some((region_pq, region_qp))
}

/// `subtract_loop_from_face`: the remainder of `face` after `loop_` is carved
/// out of it.
pub fn subtract_loop_from_face(face: &PFace, loop_: &[V3], h: Hypot) -> Option<Vec<V3>> {
    let chain = find_subdividing_chain(face, loop_, h)?;
    let (r1, r2) = split_face_by_chain(&face.vertices, &chain, h)?;
    let loop_keys = keyset(loop_);
    if keyset(&r1) == loop_keys {
        return Some(r2);
    }
    if keyset(&r2) == loop_keys {
        return Some(r1);
    }
    None
}

/// `carve_loop_by_chords`: the pieces of `face` left around `loop_` when every
/// corner of the loop sits on the face's boundary.
pub fn carve_loop_by_chords(face: &PFace, loop_: &[V3], h: Hypot) -> Option<Vec<Vec<V3>>> {
    if face.vertices.len() < 3 || loop_.len() < 3 {
        return None;
    }
    if loop_.iter().any(|&v| locate_on_loop(&face.vertices, v, h).is_none()) {
        return None;
    }
    let normal = face.normal(h);
    let origin = face.vertices[0];
    let (u, w) = plane_axes(normal, h);
    let proj = proj_fn(origin, u, w);
    let poly2: Vec<P2> = face.vertices.iter().map(|&v| proj(v)).collect();
    let mut chords: Vec<[V3; 2]> = Vec::new();
    let n = loop_.len();
    for i in 0..n {
        let (a, b) = (loop_[i], loop_[(i + 1) % n]);
        let mid = a.add(b).mul(0.5);
        if locate_on_loop(&face.vertices, mid, h).is_some() {
            continue;
        }
        if !strictly_inside_2d(proj(mid), &poly2) {
            return None;
        }
        chords.push([a, b]);
    }
    if chords.is_empty() {
        return None;
    }
    let mut pieces: Vec<Vec<V3>> = vec![face.vertices.clone()];
    for chord in &chords {
        let mut done = false;
        for idx in 0..pieces.len() {
            let mid = chord[0].add(chord[1]).mul(0.5);
            let p2: Vec<P2> = pieces[idx].iter().map(|&v| proj(v)).collect();
            if !strictly_inside_2d(proj(mid), &p2) {
                continue;
            }
            let (s0, s1) = split_face_by_chain(&pieces[idx], chord, h)?;
            pieces.splice(idx..idx + 1, [s0, s1]);
            done = true;
            break;
        }
        if !done {
            return None;
        }
    }
    let loop_keys = keyset(loop_);
    let rest: Vec<Vec<V3>> = pieces.iter().filter(|p| keyset(p) != loop_keys).cloned().collect();
    if rest.len() != pieces.len() - 1 {
        return None;
    }
    Some(rest)
}

// ---- Polygon offset ----------------------------------------------------------------

/// `_offset_line_intersection`.
fn offset_line_intersection(p0: V3, u: V3, p1: V3, v: V3, n: V3) -> V3 {
    let denom = dotf(u.cross(v), n);
    if denom.abs() < 1e-9 {
        return p1;
    }
    let s = dotf(p1.sub(p0).cross(v), n) / denom;
    p0.add(u.mul(s as f32))
}

/// `offset_loop`: offset a planar polygon by `d` in its plane (`d > 0`
/// inward); `None` if it degenerates.
pub fn offset_loop(loop_: &[V3], normal: V3, d: f64, h: Hypot) -> Option<Vec<V3>> {
    let count = loop_.len();
    if count < 3 {
        return None;
    }
    let n = normal.normalized(h);
    let mut offset_lines: Vec<(V3, V3)> = Vec::new();
    for i in 0..count {
        let a = loop_[i];
        let b = loop_[(i + 1) % count];
        let edge = b.sub(a);
        if lenf(edge, h) < SPLIT_TOLERANCE {
            return None;
        }
        let e = edge.normalized(h);
        let inward = n.cross(e).normalized(h);
        offset_lines.push((a.add(inward.mul(d as f32)), e));
    }
    let mut new_loop: Vec<V3> = Vec::new();
    for i in 0..count {
        let (p0, u) = offset_lines[(i + count - 1) % count];
        let (p1, v) = offset_lines[i];
        new_loop.push(offset_line_intersection(p0, u, p1, v, n));
    }
    for i in 0..count {
        let orig = loop_[(i + 1) % count].sub(loop_[i]);
        let new = new_loop[(i + 1) % count].sub(new_loop[i]);
        if lenf(new, h) < SPLIT_TOLERANCE {
            return None;
        }
        if dotf(orig.normalized(h), new.normalized(h)) <= 0.0 {
            return None;
        }
    }
    Some(new_loop)
}

/// `max_offset_distance`: the largest offset in the direction of `sign` that
/// `offset_loop` still accepts.
pub fn max_offset_distance(loop_: &[V3], normal: V3, sign: f64, limit: f64, h: Hypot) -> f64 {
    let step = limit.abs();
    if offset_loop(loop_, normal, sign * 1e-4, h).is_none() {
        return 0.0;
    }
    let (mut lo, mut hi) = (1e-4, step);
    for _ in 0..40 {
        let mid = (lo + hi) / 2.0;
        if offset_loop(loop_, normal, sign * mid, h).is_none() {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    lo
}

// ---- Heal overlapping coplanar faces -------------------------------------------------

/// `_interior_point`: the centroid of the loop's biggest triangle.
pub fn interior_point(loop_: &[V3], normal: V3, h: Hypot) -> Option<V3> {
    let tris = triangulate(loop_, &[], Some(normal), h);
    let mut best: Option<V3> = None;
    let mut best_area = 0.0f64;
    for [a, b, c] in tris {
        let area = b.sub(a).cross(c.sub(a)).length_squared() as f64;
        if area > best_area {
            best_area = area;
            best = Some(a.add(b).add(c).div(3.0));
        }
    }
    best
}

/// `_loop_inside_loop`: `inner` lies inside `outer` (coplanar, by centroid,
/// then by a point that really is inside).
pub fn loop_inside_loop(inner: &[V3], outer: &[V3], normal: V3, h: Hypot) -> bool {
    if inner.len() < 3 || outer.len() < 3 {
        return false;
    }
    let (u, w) = plane_axes(normal, h);
    let origin = outer[0];
    let proj = proj_fn(origin, u, w);
    let poly: Vec<P2> = outer.iter().map(|&p| proj(p)).collect();
    let c = inner.len() as f64;
    let cx = py_sum(inner.iter().map(|p| p.0 as f64), true) / c;
    let cy = py_sum(inner.iter().map(|p| p.1 as f64), true) / c;
    let cz = py_sum(inner.iter().map(|p| p.2 as f64), true) / c;
    if point_inside_2d(proj(V3(cx as f32, cy as f32, cz as f32)), &poly) {
        return true;
    }
    let Some(point) = interior_point(inner, normal, h) else {
        return false;
    };
    point_inside_2d(proj(point), &poly)
}

/// `_maximal_holes`: the indices of the holes kept (those not nested inside
/// a larger hole of the same face), in order.
pub fn maximal_holes(holes: &[Vec<V3>], h: Hypot) -> Vec<usize> {
    let area = |lp: &[V3]| -> f64 {
        let mut a = 0.0f64;
        let n = lp.len();
        for i in 0..n {
            a += lp[i].0 as f64 * lp[(i + 1) % n].1 as f64;
            a -= lp[(i + 1) % n].0 as f64 * lp[i].1 as f64;
        }
        a.abs() / 2.0
    };
    let areas: Vec<f64> = holes.iter().map(|hl| area(hl)).collect();
    let mut boxes: Vec<[f64; 6]> = Vec::new();
    for hl in holes {
        if hl.is_empty() {
            boxes.push([0.0; 6]);
            continue;
        }
        let fold = |f: fn(&V3) -> f64, pick: fn(f64, f64) -> f64| -> f64 {
            let mut it = hl.iter().map(f);
            let first = it.next().unwrap();
            it.fold(first, pick)
        };
        boxes.push([
            fold(|p| p.0 as f64, py_min2),
            fold(|p| p.1 as f64, py_min2),
            fold(|p| p.2 as f64, py_min2),
            fold(|p| p.0 as f64, py_max2),
            fold(|p| p.1 as f64, py_max2),
            fold(|p| p.2 as f64, py_max2),
        ]);
    }
    let span = boxes.iter().flat_map(|b| b.iter().map(|x| x.abs())).fold(f64::NEG_INFINITY, f64::max);
    let span = if boxes.is_empty() { 0.0 } else { span };
    let margin = 1e-4 * (1.0 + span);
    let mut keep: Vec<usize> = Vec::new();
    for (i, hl) in holes.iter().enumerate() {
        if hl.len() < 3 {
            keep.push(i);
            continue;
        }
        let nrm = hl[1].sub(hl[0]).cross(hl[2].sub(hl[0]));
        if lenf(nrm, h) < 1e-9 {
            keep.push(i);
            continue;
        }
        let nrm = nrm.normalized(h);
        let b = boxes[i];
        let nested = (0..holes.len()).any(|j| {
            let bj = &boxes[j];
            let cand = areas[j] > areas[i] + 1e-6
                && (bj[3] >= b[0] - margin)
                && (bj[0] <= b[3] + margin)
                && (bj[4] >= b[1] - margin)
                && (bj[1] <= b[4] + margin)
                && (bj[5] >= b[2] - margin)
                && (bj[2] <= b[5] + margin);
            cand && j != i && loop_inside_loop(hl, &holes[j], nrm, h)
        });
        if !nested {
            keep.push(i);
        }
    }
    keep
}

/// `_g_inside_a_hole`: `loop_` sits inside one of `face`'s holes.
pub fn g_inside_a_hole(face: &PFace, loop_: &[V3], h: Hypot) -> bool {
    let n = face.normal(h);
    face.holes.iter().filter(|hl| hl.len() >= 3).any(|hl| loop_inside_loop(loop_, hl, n, h))
}

/// `_point_in_face_solid`: inside the outer loop and outside every hole.
pub fn point_in_face_solid(face: &PFace, point: V3, h: Hypot) -> bool {
    let n = face.normal(h);
    let (u, w) = plane_axes(n, h);
    let origin = face.vertices[0];
    let proj = proj_fn(origin, u, w);
    let pt = proj(point);
    let poly: Vec<P2> = face.vertices.iter().map(|&v| proj(v)).collect();
    if !point_inside_2d(pt, &poly) {
        return false;
    }
    !face.holes.iter().any(|hl| {
        let p: Vec<P2> = hl.iter().map(|&v| proj(v)).collect();
        point_inside_2d(pt, &p)
    })
}

/// One face of `winding_footprint`: the pre-edit face, its normal, loops and
/// bounding box (`min x, y, z, max x, y, z`).
#[derive(Clone, Debug)]
pub struct FootprintEntry {
    pub face: FaceId,
    pub normal: V3,
    pub outer: Vec<V3>,
    pub holes: Vec<Vec<V3>>,
    pub bbox: [f64; 6],
}

/// `mesh_is_flat`: every face lies on one common plane.
pub fn mesh_is_flat(m: &Mesh, h: Hypot) -> bool {
    let faces = &m.faces;
    if faces.len() < 2 {
        return true;
    }
    let f0 = PFace::of(m, faces[0]);
    let n0 = f0.normal(h);
    let d0 = dotf(n0, f0.vertices[0]);
    for &f in &faces[1..] {
        let pf = PFace::of(m, f);
        if !faces_coplanar(n0, pf.normal(h), h) {
            return false;
        }
        if (dotf(n0, pf.vertices[0]) - d0).abs() > PLANAR_TOLERANCE {
            return false;
        }
    }
    true
}

/// `winding_footprint`: how the faces of a flat drawing face before an edit;
/// `None` where the orientation pass never runs.
pub fn winding_footprint(m: &Mesh, h: Hypot) -> Option<Vec<FootprintEntry>> {
    if m.faces.len() > HEAL_FACE_CAP || !mesh_is_flat(m, h) {
        return None;
    }
    let mut out = Vec::new();
    for &f in &m.faces {
        let pf = PFace::of(m, f);
        if pf.vertices.is_empty() {
            continue;
        }
        let ext = |g: fn(&V3) -> f64, pick: fn(f64, f64) -> f64| {
            let mut it = pf.vertices.iter().map(g);
            let first = it.next().unwrap();
            it.fold(first, pick)
        };
        let bbox = [
            ext(|p| p.0 as f64, py_min2),
            ext(|p| p.1 as f64, py_min2),
            ext(|p| p.2 as f64, py_min2),
            ext(|p| p.0 as f64, py_max2),
            ext(|p| p.1 as f64, py_max2),
            ext(|p| p.2 as f64, py_max2),
        ];
        out.push(FootprintEntry { face: f, normal: pf.normal(h), outer: pf.vertices.clone(), holes: pf.holes.clone(), bbox });
    }
    Some(out)
}

/// `_footprint_normal`: the normal of the pre-edit face that covered
/// `face`'s interior, or `None` where no face was.
pub fn footprint_normal(footprint: &[FootprintEntry], face: &PFace, h: Hypot) -> Option<V3> {
    let n = face.normal(h);
    let point = interior_point(&face.vertices, n, h)?;
    let t = PLANAR_TOLERANCE;
    let (px, py, pz) = (point.0 as f64, point.1 as f64, point.2 as f64);
    for e in footprint {
        let b = &e.bbox;
        if !(b[0] - t <= px && px <= b[3] + t && b[1] - t <= py && py <= b[4] + t && b[2] - t <= pz && pz <= b[5] + t) {
            continue;
        }
        if dotf(e.normal, n).abs() < 0.999 {
            continue;
        }
        let (u, w) = plane_axes(e.normal, h);
        let origin = e.outer[0];
        if dotf(point.sub(origin), e.normal).abs() > PLANAR_TOLERANCE {
            continue;
        }
        let proj = proj_fn(origin, u, w);
        let pt = proj(point);
        let poly: Vec<P2> = e.outer.iter().map(|&v| proj(v)).collect();
        if point_inside_2d(pt, &poly)
            && !e.holes.iter().any(|hl| {
                let p: Vec<P2> = hl.iter().map(|&v| proj(v)).collect();
                point_inside_2d(pt, &p)
            })
        {
            return Some(e.normal);
        }
    }
    None
}

/// `round(abs(x), 2)` of a float32, as an exact integer (the product has
/// at most 31 bits, so it is exact in double and its ties are real).
fn round2(x: f32) -> i64 {
    ((x.abs() as f64) * 100.0).round_ties_even() as i64
}

/// `orient_coplanar_faces`: flip faces whose winding came out reversed, so
/// coplanar faces face the same way. Returns the flipped faces (the old ids).
pub fn orient_coplanar_faces(m: &mut Mesh, footprint: Option<&[FootprintEntry]>, h: Hypot) -> Vec<FaceId> {
    let mut order: Vec<(i64, i64, i64, i64)> = Vec::new();
    let mut groups: HashMap<(i64, i64, i64, i64), Vec<FaceId>> = HashMap::new();
    for &f in &m.faces {
        let pf = PFace::of(m, f);
        let n = pf.normal(h);
        let axis = (round2(n.0), round2(n.1), round2(n.2));
        let c = mf::centroid(&pf.vertices);
        let dist = round2(n.dot(c));
        let k = (axis.0, axis.1, axis.2, dist);
        groups
            .entry(k)
            .or_insert_with(|| {
                order.push(k);
                Vec::new()
            })
            .push(f);
    }
    let owned: HashSet<FaceId> = footprint.map(|fp| fp.iter().map(|e| e.face).collect()).unwrap_or_default();
    let mut flipped: Vec<FaceId> = Vec::new();
    for k in &order {
        let fs = groups[k].clone();
        if fs.len() < 2 {
            continue;
        }
        let mut dominant = V3(0.0, 0.0, 0.0);
        for &f in &fs {
            let pf = PFace::of(m, f);
            dominant = dominant.add(pf.normal(h).mul(pf.area(h) as f32));
        }
        for &f in &fs {
            let pf = PFace::of(m, f);
            let mut target = dominant;
            if let Some(fp) = footprint {
                if owned.contains(&f) {
                    continue;
                }
                target = footprint_normal(fp, &pf, h).unwrap_or(dominant);
            }
            if pf.normal(h).dot(target) < 0.0 {
                let mut outer = pf.vertices.clone();
                outer.reverse();
                let holes = pf.holes.clone();
                let attrs = m.face(f).attrs.clone();
                m.remove_face(f);
                let nf = add_face_py(m, &outer, &holes, h);
                m.face_mut(nf).attrs = attrs;
                flipped.push(f);
            }
        }
    }
    flipped
}

/// `_collinear_overlapping`: edge `e1` lies collinearly over `e2` with
/// overlapping span.
pub fn collinear_overlapping(e1: Seg, e2: Seg, h: Hypot) -> bool {
    let (a, b) = e1;
    let (c, d) = e2;
    let d1 = b.sub(a);
    let d2 = d.sub(c);
    if lenf(d1, h) < 1e-9 || lenf(d2, h) < 1e-9 {
        return false;
    }
    let u = d2.normalized(h);
    if lenf(d1.normalized(h).cross(u), h) > 1e-4 {
        return false;
    }
    let rel = a.sub(c);
    if lenf(rel.sub(u.mul(rel.dot(u))), h) > PLANAR_TOLERANCE {
        return false;
    }
    let ta = dotf(a.sub(c), u);
    let tb = dotf(b.sub(c), u);
    let (lo1, hi1) = (py_min2(ta, tb), py_max2(ta, tb));
    py_min2(hi1, lenf(d2, h)) - py_max2(lo1, 0.0) > 1e-4
}

/// `sweep_tjunctions`: split every edge that runs past a vertex belonging to
/// another face, one batched pass per round. `seedkeys` (positions rounded
/// to the weld tolerance) bound the sweep to an operation's own box.
pub fn sweep_tjunctions(m: &mut Mesh, seedkeys: Option<&[[f64; 3]]>, h: Hypot) {
    let _ = h;
    let tol = STITCH_TOL;
    let tol2 = STITCH_TOL * STITCH_TOL;
    let seed: Option<([f64; 3], [f64; 3])> = match seedkeys {
        Some(sk) if !sk.is_empty() => {
            let mut lo = sk[0];
            let mut hi = sk[0];
            for k in sk {
                for a in 0..3 {
                    if k[a] < lo[a] {
                        lo[a] = k[a];
                    }
                    if k[a] > hi[a] {
                        hi[a] = k[a];
                    }
                }
            }
            Some((std::array::from_fn(|a| lo[a] - tol), std::array::from_fn(|a| hi[a] + tol)))
        }
        _ => None,
    };
    loop {
        let verts = m.vertices.clone();
        let edges = m.edges.clone();
        if verts.is_empty() || edges.is_empty() {
            break;
        }
        let vpos: Vec<[f64; 3]> = verts
            .iter()
            .map(|&v| {
                let p = m.position(v);
                [p.0 as f64, p.1 as f64, p.2 as f64]
            })
            .collect();
        let vslot: HashMap<VertexId, usize> = verts.iter().enumerate().map(|(i, &v)| (v, i)).collect();
        let ends: Vec<(usize, usize)> = edges.iter().map(|&e| (vslot[&m.edge(e).v0], vslot[&m.edge(e).v1])).collect();
        let va: Vec<[f64; 3]> = ends.iter().map(|e| vpos[e.0]).collect();
        let vb: Vec<[f64; 3]> = ends.iter().map(|e| vpos[e.1]).collect();
        let elo: Vec<[f64; 3]> = (0..edges.len()).map(|i| std::array::from_fn(|a| va[i][a].min(vb[i][a]))).collect();
        let ehi: Vec<[f64; 3]> = (0..edges.len()).map(|i| std::array::from_fn(|a| va[i][a].max(vb[i][a]))).collect();
        let seg_all: Vec<[f64; 3]> = (0..edges.len()).map(|i| std::array::from_fn(|a| vb[i][a] - va[i][a])).collect();
        let mut pick: Vec<usize> = Vec::new();
        for i in 0..edges.len() {
            let mut live = match &seed {
                None => true,
                Some((slo, shi)) => (0..3).all(|a| ehi[i][a] >= slo[a]) && (0..3).all(|a| elo[i][a] <= shi[a]),
            };
            live &= einsum3(seg_all[i], seg_all[i]) > tol2;
            if live {
                pick.push(i);
            }
        }
        if pick.is_empty() {
            break;
        }
        // Broad phase: the x-sorted vertices (stable), the slab of each edge.
        let mut order: Vec<usize> = (0..verts.len()).collect();
        order.sort_by(|&x, &y| vpos[x][0].partial_cmp(&vpos[y][0]).unwrap_or(std::cmp::Ordering::Equal));
        let xs: Vec<f64> = order.iter().map(|&i| vpos[i][0]).collect();
        let mut best: Vec<usize> = vec![verts.len(); edges.len()];
        for &ei in &pick {
            let lo = xs.partition_point(|&x| x < elo[ei][0] - tol);
            let hi = xs.partition_point(|&x| x <= ehi[ei][0] + tol);
            for &vidx in &order[lo..hi.max(lo)] {
                if vidx == ends[ei].0 || vidx == ends[ei].1 {
                    continue;
                }
                let mut keep = true;
                for ax in 1..=2 {
                    let coord = vpos[vidx][ax];
                    keep = keep && (coord >= elo[ei][ax] - tol) && (coord <= ehi[ei][ax] + tol);
                    if !keep {
                        break;
                    }
                }
                if !keep {
                    continue;
                }
                let seg = seg_all[ei];
                let rel: [f64; 3] = std::array::from_fn(|a| vpos[vidx][a] - va[ei][a]);
                let l2 = einsum3(seg, seg);
                let t = einsum3(rel, seg) / l2;
                let tol_t = tol / l2.sqrt();
                let mut ok = (t > tol_t) && (t < 1.0 - tol_t);
                let d: [f64; 3] = std::array::from_fn(|a| rel[a] - seg[a] * t);
                ok &= einsum3(d, d) < tol2;
                if ok && vidx < best[ei] {
                    best[ei] = vidx;
                }
            }
        }
        let hits: Vec<usize> = (0..edges.len()).filter(|&i| best[i] < verts.len()).collect();
        if hits.is_empty() {
            break;
        }
        for i in hits {
            split_edge_at(m, edges[i], verts[best[i]]);
        }
    }
}

/// `resolve_tjunctions`: split edges at T-junction vertices (the unscoped
/// sweep).
pub fn resolve_tjunctions(m: &mut Mesh, h: Hypot) {
    sweep_tjunctions(m, None, h);
}

/// `fold_nonplanar_faces` (*Autofold*): split every face a move has warped
/// out of its plane into planar pieces along fold edges. Returns the faces
/// folded (the originals, already removed).
pub fn fold_nonplanar_faces(m: &mut Mesh, tolerance: f64, h: Hypot) -> Vec<FaceId> {
    let mut folded = Vec::new();
    for face in m.faces.clone() {
        let pf = PFace::of(m, face);
        let mut pts = pf.vertices.clone();
        for hole in &pf.holes {
            pts.extend(hole.iter().copied());
        }
        if pts.len() <= 3 || is_planar(&pts, tolerance, h) {
            continue;
        }
        let tris = pf.triangulate(h);
        if tris.len() < 2 {
            continue;
        }
        let attrs = m.face(face).attrs.clone();
        m.remove_face(face);
        // A set in Python (address order); insertion order here.
        let mut pieces: Vec<FaceId> = Vec::new();
        for [a, b, c] in tris {
            let nf = add_face_py(m, &[a, b, c], &[], h);
            if !pieces.contains(&nf) {
                pieces.push(nf);
            }
        }
        for &piece in &pieces {
            m.face_mut(piece).attrs = attrs.clone();
        }
        let mut merged = true;
        while merged {
            merged = false;
            for f in pieces.clone() {
                if !m.faces.contains(&f) {
                    pieces.retain(|&x| x != f);
                    continue;
                }
                let lp = m.face(f).outer.clone();
                for i in 0..lp.len() {
                    let Some(e) = m.find_edge(lp[i], lp[(i + 1) % lp.len()]) else {
                        continue;
                    };
                    if m.edge(e).faces.len() != 2 {
                        continue;
                    }
                    let ef = m.edge(e).faces.clone();
                    let g = if ef[1] == f { ef[0] } else { ef[1] };
                    if g == f || !pieces.contains(&g) {
                        continue;
                    }
                    let nf = PFace::of(m, f).normal(h).normalized(h);
                    let ng = PFace::of(m, g).normal(h).normalized(h);
                    if dotf(nf, ng) > 0.999 {
                        if let Some(region) = dissolve_edge(m, e, h) {
                            m.face_mut(region).attrs = attrs.clone();
                            pieces.retain(|&x| x != f && x != g);
                            if !pieces.contains(&region) {
                                pieces.push(region);
                            }
                            merged = true;
                            break;
                        }
                    }
                }
                if merged {
                    break;
                }
            }
        }
        folded.push(face);
    }
    folded
}

/// `prune_collinear_orphan_edges`: remove edges that bound no face and lie
/// collinearly over another edge. Returns the removed edges.
pub fn prune_collinear_orphan_edges(m: &mut Mesh, h: Hypot) -> Vec<EdgeId> {
    let mut removed = Vec::new();
    let edges = m.edges.clone();
    let orphans: Vec<EdgeId> = edges.iter().copied().filter(|&e| m.edge(e).faces.is_empty()).collect();
    if orphans.is_empty() {
        return removed;
    }
    let segs = edge_segs(m, &edges);
    let boxes: Vec<([f64; 3], [f64; 3])> = segs
        .iter()
        .map(|&(a, b)| {
            let (a, b) = ([a.0 as f64, a.1 as f64, a.2 as f64], [b.0 as f64, b.1 as f64, b.2 as f64]);
            (
                std::array::from_fn(|k| a[k].min(b[k]) - PLANAR_TOLERANCE),
                std::array::from_fn(|k| a[k].max(b[k]) + PLANAR_TOLERANCE),
            )
        })
        .collect();
    let slot: HashMap<EdgeId, usize> = edges.iter().enumerate().map(|(i, &e)| (e, i)).collect();
    let mut gone: HashSet<EdgeId> = HashSet::new();
    for &e in &orphans {
        let i = slot[&e];
        let (lo_i, hi_i) = boxes[i];
        for j in 0..edges.len() {
            let (lo_j, hi_j) = boxes[j];
            if !((0..3).all(|k| hi_j[k] >= lo_i[k]) && (0..3).all(|k| lo_j[k] <= hi_i[k])) {
                continue;
            }
            let other = edges[j];
            if other == e || gone.contains(&other) {
                continue;
            }
            if collinear_overlapping(segs[i], segs[j], h) {
                m.remove_edge(e);
                removed.push(e);
                gone.insert(e);
                break;
            }
        }
    }
    removed
}

/// `heal_overlapping_faces`: clean up coplanar face overlaps. `partial` is
/// `None` for auto (on only for a flat drawing). `orient_outward` is
/// `core.orient.orient_outward`, run on a 3D model as the last step. Returns
/// the faces removed.
pub fn heal_overlapping_faces(
    m: &mut Mesh,
    coverage: f64,
    partial: Option<bool>,
    footprint: Option<&[FootprintEntry]>,
    h: Hypot,
    orient_outward: &mut dyn FnMut(&mut Mesh),
) -> Vec<FaceId> {
    if m.faces.len() > HEAL_FACE_CAP {
        return Vec::new();
    }
    let flat = mesh_is_flat(m, h);
    let partial = partial.unwrap_or(flat);

    prune_collinear_orphan_edges(m, h);
    resolve_tjunctions(m, h);
    if flat {
        orient_coplanar_faces(m, footprint, h);
    }

    // 1. Dedupe nested holes.
    for face in m.faces.clone() {
        if m.face(face).holes.len() < 2 {
            continue;
        }
        let pf = PFace::of(m, face);
        let keep = maximal_holes(&pf.holes, h);
        if keep.len() != pf.holes.len() {
            let kept: Vec<Vec<V3>> = keep.iter().map(|&i| pf.holes[i].clone()).collect();
            let attrs = m.face(face).attrs.clone();
            m.remove_face(face);
            let nf = add_face_py(m, &pf.vertices, &kept, h);
            m.face_mut(nf).attrs = attrs;
        }
    }

    // 2. Remove a redundant mother covered by faces that aren't in its holes.
    let mut removed: Vec<FaceId> = Vec::new();
    let snap: Vec<FaceId> = m.faces.clone();
    let pfs: Vec<PFace> = snap.iter().map(|&f| PFace::of(m, f)).collect();
    let nrm: Vec<[f64; 3]> = pfs
        .iter()
        .map(|p| {
            let n = p.normal(h);
            [n.0 as f64, n.1 as f64, n.2 as f64]
        })
        .collect();
    let area: Vec<f64> = pfs.iter().map(|p| p.area(h)).collect();
    let gemv = |i: usize, j: usize| -> f64 { (nrm[j][0] * nrm[i][0] + nrm[j][1] * nrm[i][1]) + nrm[j][2] * nrm[i][2] };
    let mut gone: HashSet<FaceId> = HashSet::new();
    for i in 0..snap.len() {
        let face = snap[i];
        let a_face = area[i];
        if a_face < SPLIT_TOLERANCE {
            continue;
        }
        let mut covered = 0.0f64;
        for j in 0..snap.len() {
            if !(gemv(i, j).abs() > 0.999 && area[j] < a_face) {
                continue;
            }
            let g = snap[j];
            if g == face || gone.contains(&g) {
                continue;
            }
            if loop_inside_face(&pfs[i], &pfs[j].vertices, false, h) && !g_inside_a_hole(&pfs[i], &pfs[j].vertices, h) {
                covered += area[j];
            }
        }
        if covered > coverage * a_face {
            m.remove_face(face);
            removed.push(face);
            gone.insert(face);
        }
    }

    // 3. (Flat plans) punch a matching hole in the bigger face.
    if partial {
        let snap3: Vec<FaceId> = m.faces.clone();
        let pfs3: Vec<PFace> = snap3.iter().map(|&f| PFace::of(m, f)).collect();
        let nrm3: Vec<[f64; 3]> = pfs3
            .iter()
            .map(|p| {
                let n = p.normal(h);
                [n.0 as f64, n.1 as f64, n.2 as f64]
            })
            .collect();
        let area3: Vec<f64> = pfs3.iter().map(|p| p.area(h)).collect();
        let gemv3 = |i: usize, j: usize| -> f64 { (nrm3[j][0] * nrm3[i][0] + nrm3[j][1] * nrm3[i][1]) + nrm3[j][2] * nrm3[i][2] };
        let mut punch: Vec<(FaceId, Vec<FaceId>)> = Vec::new();
        for i3 in 0..snap3.len() {
            let face = snap3[i3];
            if gone.contains(&face) {
                continue;
            }
            let tris = pfs3[i3].triangulate(h);
            if tris.is_empty() {
                continue;
            }
            let probes: Vec<V3> = tris.iter().map(|t| t[0].add(t[1]).add(t[2]).div(3.0)).collect();
            let a_face = area3[i3];
            for j3 in 0..snap3.len() {
                if !(gemv3(i3, j3).abs() > 0.999 && area3[j3] > a_face) {
                    continue;
                }
                let g = snap3[j3];
                if g == face || gone.contains(&g) {
                    continue;
                }
                if g_inside_a_hole(&pfs3[j3], &pfs3[i3].vertices, h) {
                    continue;
                }
                let inside = probes.iter().filter(|&&p| point_in_face_solid(&pfs3[j3], p, h)).count();
                if inside * 2 > probes.len() {
                    match punch.iter_mut().find(|(k, _)| *k == g) {
                        Some((_, v)) => v.push(face),
                        None => punch.push((g, vec![face])),
                    }
                    break;
                }
            }
        }
        for (g, smalls) in punch {
            let pg = PFace::of(m, g);
            let mut holes = pg.holes.clone();
            for s in &smalls {
                holes.push(positions(m, &m.face(*s).outer));
            }
            let attrs = m.face(g).attrs.clone();
            m.remove_face(g);
            let nf = add_face_py(m, &pg.vertices, &holes, h);
            m.face_mut(nf).attrs = attrs;
        }
    }

    // 4. (3D) Volumetric winding.
    if !flat {
        orient_outward(m);
    }
    removed
}

// ---- Defects ------------------------------------------------------------------------

/// `_loop_crosses_itself`: two non-adjacent sides of a closed loop cross or
/// touch, in the loop's own plane.
pub fn loop_crosses_itself(loop_: &[V3], normal: V3) -> bool {
    let count = loop_.len();
    if count < 4 {
        return false;
    }
    let n = [normal.0 as f64, normal.1 as f64, normal.2 as f64];
    let mut ax = 0;
    for i in 1..3 {
        if n[i].abs() > n[ax].abs() {
            ax = i;
        }
    }
    let rest: Vec<usize> = (0..3).filter(|&i| i != ax).collect();
    let (u, w) = (rest[0], rest[1]);
    let pts: Vec<P2> = loop_
        .iter()
        .map(|v| {
            let p = [v.0 as f64, v.1 as f64, v.2 as f64];
            (p[u], p[w])
        })
        .collect();
    let eps = 1e-9;
    let orient = |a: P2, b: P2, c: P2| -> i32 {
        let val = (b.0 - a.0) * (c.1 - a.1) - (b.1 - a.1) * (c.0 - a.0);
        if val.abs() < eps {
            0
        } else if val > 0.0 {
            1
        } else {
            -1
        }
    };
    let on_seg = |a: P2, b: P2, c: P2| -> bool {
        py_min2(a.0, b.0) - eps <= c.0
            && c.0 <= py_max2(a.0, b.0) + eps
            && py_min2(a.1, b.1) - eps <= c.1
            && c.1 <= py_max2(a.1, b.1) + eps
    };
    for i in 0..count {
        let (a, b) = (pts[i], pts[(i + 1) % count]);
        for j in i + 2..count {
            if i == 0 && j == count - 1 {
                continue;
            }
            let (c, d) = (pts[j], pts[(j + 1) % count]);
            let (o1, o2) = (orient(a, b, c), orient(a, b, d));
            let (o3, o4) = (orient(c, d, a), orient(c, d, b));
            if o1 != o2 && o3 != o4 && ![o1, o2, o3, o4].contains(&0) {
                return true;
            }
            if (o1 == 0 && on_seg(a, b, c)) || (o2 == 0 && on_seg(a, b, d)) || (o3 == 0 && on_seg(c, d, a)) || (o4 == 0 && on_seg(c, d, b))
            {
                return true;
            }
        }
    }
    false
}

/// `_coplanar_overlap`: two faces lie in one plane and cover some of the
/// same area.
pub fn coplanar_overlap(f: &PFace, g: &PFace, h: Hypot) -> bool {
    let nf = f.normal(h);
    let ng = g.normal(h);
    if dotf(nf, ng).abs() < 1.0 - 1e-6 {
        return false;
    }
    let origin = f.vertices[0];
    if g.vertices.iter().any(|&v| dotf(v.sub(origin), nf).abs() > SPLIT_TOLERANCE) {
        return false;
    }
    for (a, b) in [(f, g), (g, f)] {
        let inner = interior_point(&a.vertices, a.normal(h), h);
        if let Some(p) = inner {
            if point_in_face_solid(a, p, h) && point_in_face_solid(b, p, h) {
                return true;
            }
        }
    }
    let (u, w) = plane_axes(nf, h);
    let proj = proj_fn(origin, u, w);
    let sides = |face: &PFace| -> Vec<(P2, P2)> {
        let pts: Vec<P2> = face.vertices.iter().map(|&v| proj(v)).collect();
        (0..pts.len()).map(|i| (pts[i], pts[(i + 1) % pts.len()])).collect()
    };
    let orient = |a: P2, b: P2, c: P2| -> i32 {
        let val = (b.0 - a.0) * (c.1 - a.1) - (b.1 - a.1) * (c.0 - a.0);
        if val.abs() < 1e-9 {
            0
        } else if val > 0.0 {
            1
        } else {
            -1
        }
    };
    let sg = sides(g);
    for (a, b) in sides(f) {
        for &(c, d) in &sg {
            let (o1, o2) = (orient(a, b, c), orient(a, b, d));
            let (o3, o4) = (orient(c, d, a), orient(c, d, b));
            if ![o1, o2, o3, o4].contains(&0) && o1 != o2 && o3 != o4 {
                return true;
            }
        }
    }
    false
}

/// `defects_near`: count the topological defects among the edges and faces
/// that touch `vertices`, by kind (`DEFECT_KINDS` order). A vertex's edges
/// are walked by edge id.
pub fn defects_near(m: &Mesh, vertices: &[VertexId], area_eps: f64, h: Hypot) -> [usize; 6] {
    let mut counts = [0usize; 6];
    let mut edges: Vec<EdgeId> = Vec::new();
    let mut seen_e: HashSet<EdgeId> = HashSet::new();
    let mut faces: Vec<FaceId> = Vec::new();
    let mut seen_f: HashSet<FaceId> = HashSet::new();
    for &v in vertices {
        let mut ves = m.vertex_ref(v).edges.clone();
        ves.sort();
        for e in ves {
            if seen_e.insert(e) {
                edges.push(e);
            }
            for &f in &m.edge(e).faces {
                if seen_f.insert(f) {
                    faces.push(f);
                }
            }
        }
    }
    for &e in &edges {
        let ed = m.edge(e);
        if ed.v0 == ed.v1 || (m.position(ed.v0).sub(m.position(ed.v1)).length_squared() as f64) < 1e-18 {
            counts[5] += 1;
        }
        let distinct: HashSet<FaceId> = ed.faces.iter().copied().collect();
        if distinct.len() < ed.faces.len() {
            counts[2] += 1;
        }
    }
    let mut seen: HashSet<Vec<(VertexId, VertexId)>> = HashSet::new();
    for &f in &faces {
        let face = m.face(f);
        let pf = PFace::of(m, f);
        let distinct: HashSet<VertexId> = face.outer.iter().copied().collect();
        if distinct.len() < 3 || pf.area(h) < area_eps {
            counts[4] += 1;
        }
        for lp in std::iter::once(&face.outer).chain(face.holes.iter()) {
            let d: HashSet<VertexId> = lp.iter().copied().collect();
            if d.len() < lp.len() {
                counts[2] += 1;
            } else if lp.len() >= 4 && loop_crosses_itself(&positions(m, lp), pf.normal(h)) {
                counts[3] += 1;
            }
        }
        let n = face.outer.len();
        let mut sig: Vec<(VertexId, VertexId)> = (0..n)
            .map(|i| {
                let (a, b) = (face.outer[i], face.outer[(i + 1) % n]);
                if a <= b {
                    (a, b)
                } else {
                    (b, a)
                }
            })
            .collect();
        sig.sort();
        sig.dedup();
        if !seen.insert(sig) {
            counts[0] += 1;
        }
    }
    let flat: Vec<PFace> = faces
        .iter()
        .filter(|&&f| {
            let face = m.face(f);
            let d: HashSet<VertexId> = face.outer.iter().copied().collect();
            d.len() >= 3 && PFace::of(m, f).area(h) >= area_eps
        })
        .map(|&f| PFace::of(m, f))
        .collect();
    for i in 0..flat.len() {
        for j in i + 1..flat.len() {
            if coplanar_overlap(&flat[i], &flat[j], h) {
                counts[1] += 1;
            }
        }
    }
    counts
}

/// `new_defects`: the defect kinds `after` counts more of than `before`,
/// worst first.
pub fn new_defects(before: &[usize; 6], after: &[usize; 6]) -> Vec<&'static str> {
    (0..6).filter(|&i| after[i] > before[i]).map(|i| DEFECT_KINDS[i]).collect()
}
