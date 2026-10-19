// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! The editable mesh (`core/mesh.py`): a shared-vertex, non-manifold B-rep
//! kept in arenas.
//!
//! Python's mesh is objects that point at each other; undo puts the very
//! same objects back. Here every vertex, edge and face lives in an arena
//! for good and is named by its index: removing one takes it out of the
//! mesh's lists, never out of the arena, so an id held by a command (or by
//! a journal, later) can be relinked exactly as Python relinks the object.
//!
//! Order is part of the behaviour: `vertices`, `edges` and `faces` are the
//! Python lists, in the same order — `list.append` appends, `list.remove`
//! takes the first match out and shifts the rest — because the .igz bytes
//! and every exact golden follow them. A vertex's incident edges are a
//! Python `set` (iterated in address order, which no run repeats); here
//! they are kept in insertion order, and nothing exact may depend on it.

use std::collections::HashMap;

use ingetrazo_formats::pyjson::Value;
use ingetrazo_math::V3;

/// Weld resolution: positions within this many decimals are one vertex.
pub const KEY_DECIMALS: i32 = 4;
const KEY_SCALE: f64 = 10_000.0; // 10.0 ** KEY_DECIMALS

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct VertexId(pub u32);
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct EdgeId(pub u32);
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct FaceId(pub u32);

pub type Key = (i64, i64, i64);

/// `core.mesh._key`: the integer weld cell — Python's `round()` (half to
/// even) of each float32 coordinate, widened, times 10⁴.
pub fn key(p: V3) -> Key {
    let r = |c: f32| ((c as f64) * KEY_SCALE).round_ties_even() as i64;
    (r(p.0), r(p.1), r(p.2))
}

#[derive(Clone, Debug)]
pub struct Vertex {
    pub position: V3,
    /// Incident edges (a set in Python; insertion order here).
    pub edges: Vec<EdgeId>,
}

#[derive(Clone, Debug)]
pub struct Edge {
    pub v0: VertexId,
    pub v1: VertexId,
    /// Radial list of incident faces, in Python's list order.
    pub faces: Vec<FaceId>,
    pub soft: bool,
    pub hidden: bool,
    pub layer: Option<String>,
    pub curve: Option<i64>,
}

#[derive(Clone, Debug)]
pub struct Face {
    pub outer: Vec<VertexId>,
    pub holes: Vec<Vec<VertexId>>,
    pub interior: bool,
    /// `face.attrs`: an insertion-ordered dict of JSON-like values.
    pub attrs: Vec<(String, Value)>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DegenerateEdge;

/// Per-edge flags for `add_edges_welded`: `(soft, curve, layer, hidden)`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct EdgeFlags {
    pub soft: bool,
    pub curve: Option<i64>,
    pub layer: Option<String>,
    pub hidden: bool,
}

/// An identity-keeping snapshot (`Mesh.capture_state`).
#[derive(Clone, Debug)]
pub struct State {
    vertices: Vec<VertexId>,
    edges: Vec<EdgeId>,
    faces: Vec<FaceId>,
    registry: HashMap<Key, VertexId>,
    vpos: Vec<(VertexId, V3, Vec<EdgeId>)>,
    edata: Vec<(EdgeId, Vec<FaceId>, bool, bool, Option<String>, Option<i64>)>,
    fdata: Vec<(FaceId, Face)>,
}

#[derive(Clone, Debug, Default)]
pub struct Mesh {
    vs: Vec<Vertex>,
    es: Vec<Edge>,
    fs: Vec<Face>,
    /// `mesh.vertices`, `mesh.edges`, `mesh.faces`: Python's lists.
    pub vertices: Vec<VertexId>,
    pub edges: Vec<EdgeId>,
    pub faces: Vec<FaceId>,
    registry: HashMap<Key, VertexId>,
    /// Moved vertices waiting for a key another vertex still holds.
    displaced: HashMap<Key, Vec<VertexId>>,
    /// Bumped by every mutation (`_mut_serial`).
    pub serial: u64,
}

fn remove_first<T: PartialEq>(list: &mut Vec<T>, x: &T) -> bool {
    match list.iter().position(|y| y == x) {
        Some(i) => {
            list.remove(i);
            true
        }
        None => false,
    }
}

impl Mesh {
    pub fn new() -> Mesh {
        Mesh::default()
    }

    // ---- arenas --------------------------------------------------------------

    pub fn vertex_ref(&self, v: VertexId) -> &Vertex {
        &self.vs[v.0 as usize]
    }
    pub fn edge(&self, e: EdgeId) -> &Edge {
        &self.es[e.0 as usize]
    }
    pub fn edge_mut(&mut self, e: EdgeId) -> &mut Edge {
        &mut self.es[e.0 as usize]
    }
    pub fn face(&self, f: FaceId) -> &Face {
        &self.fs[f.0 as usize]
    }
    pub fn face_mut(&mut self, f: FaceId) -> &mut Face {
        &mut self.fs[f.0 as usize]
    }
    pub fn position(&self, v: VertexId) -> V3 {
        self.vs[v.0 as usize].position
    }
    /// How many vertices / edges / faces were ever created (arena sizes).
    pub fn arena_sizes(&self) -> (usize, usize, usize) {
        (self.vs.len(), self.es.len(), self.fs.len())
    }
    pub fn registry(&self) -> &HashMap<Key, VertexId> {
        &self.registry
    }
    pub fn displaced(&self) -> &HashMap<Key, Vec<VertexId>> {
        &self.displaced
    }

    fn touch(&mut self) {
        self.serial += 1;
    }

    // ---- vertices ------------------------------------------------------------

    /// The 26-cell rescue of `_lookup`: a vertex in a neighbouring cell
    /// within the real weld distance (float32 `lengthSquared` < 1e-8).
    fn probe_neighbours(&self, k: Key, p: V3) -> Option<VertexId> {
        let step = 10f64.powi(-KEY_DECIMALS);
        let tol2 = step * step;
        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    if dx == 0 && dy == 0 && dz == 0 {
                        continue;
                    }
                    if let Some(&n) = self.registry.get(&(k.0 + dx, k.1 + dy, k.2 + dz)) {
                        if (self.position(n).sub(p).length_squared() as f64) < tol2 {
                            return Some(n);
                        }
                    }
                }
            }
        }
        None
    }

    /// `vertex_at`: the existing vertex within weld tolerance, if any.
    pub fn vertex_at(&self, p: V3) -> Option<VertexId> {
        let k = key(p);
        match self.registry.get(&k) {
            Some(&v) => Some(v),
            None => self.probe_neighbours(k, p),
        }
    }

    /// `vertex`: get-or-create the shared vertex at `p`.
    pub fn vertex(&mut self, p: V3) -> VertexId {
        if let Some(v) = self.vertex_at(p) {
            return v;
        }
        let v = VertexId(self.vs.len() as u32);
        self.vs.push(Vertex { position: p, edges: Vec::new() });
        self.registry.insert(key(p), v);
        self.vertices.push(v);
        v
    }

    // ---- edges ---------------------------------------------------------------

    /// `find_edge`: the edge between two vertices, either orientation.
    pub fn find_edge(&self, a: VertexId, b: VertexId) -> Option<EdgeId> {
        let (small, target) = if self.vs[a.0 as usize].edges.len() <= self.vs[b.0 as usize].edges.len() { (a, b) } else { (b, a) };
        self.vs[small.0 as usize].edges.iter().copied().find(|&e| {
            let ed = self.edge(e);
            (if ed.v0 == small { ed.v1 } else { ed.v0 }) == target
        })
    }

    fn link_edge(&mut self, a: VertexId, b: VertexId) -> EdgeId {
        self.touch();
        if let Some(e) = self.find_edge(a, b) {
            return e;
        }
        let e = EdgeId(self.es.len() as u32);
        self.es.push(Edge { v0: a, v1: b, faces: Vec::new(), soft: false, hidden: false, layer: None, curve: None });
        self.vs[a.0 as usize].edges.push(e);
        if b != a {
            self.vs[b.0 as usize].edges.push(e);
        }
        self.edges.push(e);
        e
    }

    /// `add_edge`: a free edge between two positions (welded, deduplicated).
    pub fn add_edge(&mut self, a: V3, b: V3) -> Result<EdgeId, DegenerateEdge> {
        let v0 = self.vertex(a);
        let v1 = self.vertex(b);
        if v0 == v1 {
            return Err(DegenerateEdge);
        }
        Ok(self.link_edge(v0, v1))
    }

    /// `remove_edge`: detach and drop an edge (no cascade to faces).
    pub fn remove_edge(&mut self, e: EdgeId) {
        self.touch();
        let (v0, v1) = (self.edge(e).v0, self.edge(e).v1);
        remove_first(&mut self.vs[v0.0 as usize].edges, &e);
        remove_first(&mut self.vs[v1.0 as usize].edges, &e);
        remove_first(&mut self.edges, &e);
    }

    /// `relink_edge`: put a removed edge back (undo), same id.
    pub fn relink_edge(&mut self, e: EdgeId) {
        if self.edges.contains(&e) {
            return;
        }
        let (v0, v1) = (self.edge(e).v0, self.edge(e).v1);
        for v in [v0, v1] {
            let es = &mut self.vs[v.0 as usize].edges;
            if !es.contains(&e) {
                es.push(e);
            }
        }
        self.edges.push(e);
        for v in [v0, v1] {
            let k = key(self.position(v));
            self.registry.entry(k).or_insert(v);
        }
    }

    // ---- faces ---------------------------------------------------------------

    fn loops_of(&self, f: FaceId) -> Vec<Vec<VertexId>> {
        let face = self.face(f);
        let mut out = vec![face.outer.clone()];
        out.extend(face.holes.iter().cloned());
        out
    }

    /// `add_face`: weld the loops, create missing boundary edges and
    /// register the face on each (radial).
    ///
    /// Python drops holes nested in a bigger hole of the same face
    /// (`topology._maximal_holes`) when there are two or more; that lands
    /// with the topology port (M3) — until then `holes` must not nest.
    pub fn add_face(&mut self, outer: &[V3], holes: &[Vec<V3>]) -> FaceId {
        self.touch();
        let lp: Vec<VertexId> = outer.iter().map(|&p| self.vertex(p)).collect();
        let hs: Vec<Vec<VertexId>> = holes.iter().map(|h| h.iter().map(|&p| self.vertex(p)).collect()).collect();
        let f = FaceId(self.fs.len() as u32);
        self.fs.push(Face { outer: lp.clone(), holes: hs.clone(), interior: false, attrs: Vec::new() });
        for l in std::iter::once(&lp).chain(hs.iter()) {
            let n = l.len();
            for i in 0..n {
                let e = self.link_edge(l[i], l[(i + 1) % n]);
                self.es[e.0 as usize].faces.push(f);
            }
        }
        self.faces.push(f);
        f
    }

    /// `remove_face`: drop a face and detach it from its edges' radial
    /// lists; the edges stay.
    pub fn remove_face(&mut self, f: FaceId) {
        self.touch();
        for l in self.loops_of(f) {
            let n = l.len();
            for i in 0..n {
                if let Some(e) = self.find_edge(l[i], l[(i + 1) % n]) {
                    remove_first(&mut self.es[e.0 as usize].faces, &f);
                }
            }
        }
        remove_first(&mut self.faces, &f);
    }

    /// `relink_face`: put a removed face back (undo), recreating missing
    /// boundary edges.
    pub fn relink_face(&mut self, f: FaceId) {
        if self.faces.contains(&f) {
            return;
        }
        for l in self.loops_of(f) {
            let n = l.len();
            for i in 0..n {
                let e = self.link_edge(l[i], l[(i + 1) % n]);
                let fs = &mut self.es[e.0 as usize].faces;
                if !fs.contains(&f) {
                    fs.push(f);
                }
            }
        }
        self.faces.push(f);
    }

    /// `add_hole`: punch a hole loop into a face; returns the vertex loop.
    pub fn add_hole(&mut self, f: FaceId, loop_positions: &[V3]) -> Vec<VertexId> {
        let l: Vec<VertexId> = loop_positions.iter().map(|&p| self.vertex(p)).collect();
        self.fs[f.0 as usize].holes.push(l.clone());
        let n = l.len();
        for i in 0..n {
            let e = self.link_edge(l[i], l[(i + 1) % n]);
            let fs = &mut self.es[e.0 as usize].faces;
            if !fs.contains(&f) {
                fs.push(f);
            }
        }
        l
    }

    /// `remove_hole`.
    pub fn remove_hole(&mut self, f: FaceId, l: &[VertexId]) {
        let holes = &mut self.fs[f.0 as usize].holes;
        if let Some(i) = holes.iter().position(|h| h.as_slice() == l) {
            holes.remove(i);
        }
        let n = l.len();
        for i in 0..n {
            if let Some(e) = self.find_edge(l[i], l[(i + 1) % n]) {
                remove_first(&mut self.es[e.0 as usize].faces, &f);
            }
        }
    }

    // ---- orphans -------------------------------------------------------------

    /// `prune_orphan_vertices`: drop vertices no edge and no face uses.
    pub fn prune_orphan_vertices(&mut self) -> usize {
        let mut used = std::collections::HashSet::new();
        for &f in &self.faces {
            let face = &self.fs[f.0 as usize];
            used.extend(face.outer.iter().copied());
            for h in &face.holes {
                used.extend(h.iter().copied());
            }
        }
        let doomed: std::collections::HashSet<VertexId> =
            self.vertices.iter().copied().filter(|v| self.vs[v.0 as usize].edges.is_empty() && !used.contains(v)).collect();
        if doomed.is_empty() {
            return 0;
        }
        self.vertices.retain(|v| !doomed.contains(v));
        self.registry.retain(|_, v| !doomed.contains(v));
        self.touch();
        doomed.len()
    }

    /// `relink_vertices`: re-attach vertices a drop took out (redo).
    pub fn relink_vertices(&mut self, vs: &[VertexId]) {
        for &v in vs {
            if self.vertices.contains(&v) {
                continue;
            }
            self.vertices.push(v);
            let k = key(self.position(v));
            self.registry.entry(k).or_insert(v);
        }
        self.touch();
    }

    /// `drop_orphan_vertices`: drop those of `vs` that have no edge.
    pub fn drop_orphan_vertices(&mut self, vs: &[VertexId]) -> usize {
        let doomed: std::collections::HashSet<VertexId> =
            vs.iter().copied().filter(|v| self.vs[v.0 as usize].edges.is_empty() && self.vertices.contains(v)).collect();
        if doomed.is_empty() {
            return 0;
        }
        self.vertices.retain(|v| !doomed.contains(v));
        for &v in vs {
            if doomed.contains(&v) {
                let k = key(self.position(v));
                if self.registry.get(&k) == Some(&v) {
                    self.registry.remove(&k);
                }
            }
        }
        self.touch();
        doomed.len()
    }

    // ---- moving vertices -----------------------------------------------------

    /// `move_vertex`: `position += delta` (float32), re-keyed.
    pub fn move_vertex(&mut self, v: VertexId, delta: V3) {
        let p = self.position(v).add(delta);
        self.set_position(v, p);
    }

    /// `place_vertex`: put a vertex at `p` exactly.
    pub fn place_vertex(&mut self, v: VertexId, p: V3) {
        self.set_position(v, p);
    }

    fn set_position(&mut self, v: VertexId, p: V3) {
        self.touch();
        let old = key(self.position(v));
        self.vs[v.0 as usize].position = p;
        let new = key(p);
        if old == new {
            return;
        }
        if self.displaced.is_empty() && self.registry.get(&old) == Some(&v) && !self.registry.contains_key(&new) {
            self.registry.remove(&old);
            self.registry.insert(new, v);
            return;
        }
        self.rekey(v, old, new);
    }

    /// `_rekey`: file a moved vertex under its new key, queueing it when
    /// another vertex still holds that key.
    fn rekey(&mut self, v: VertexId, old: Key, new: Key) {
        if self.registry.get(&old) == Some(&v) {
            self.registry.remove(&old);
            if let Some(waiting) = self.displaced.remove(&old) {
                for w in waiting {
                    if w != v && !self.vs[w.0 as usize].edges.is_empty() && key(self.position(w)) == old {
                        self.registry.insert(old, w);
                        break;
                    }
                }
            }
        } else if let Some(waiting) = self.displaced.get_mut(&old) {
            waiting.retain(|&w| w != v);
            if waiting.is_empty() {
                self.displaced.remove(&old);
            }
        }
        match self.registry.get(&new) {
            None => {
                self.registry.insert(new, v);
            }
            Some(&holder) if holder != v => self.displaced.entry(new).or_default().push(v),
            Some(_) => {}
        }
    }

    // ---- splitting -----------------------------------------------------------

    /// `split_edge`: insert a shared vertex at `p` in `e` and in every
    /// incident face's loop; returns the two halves (or `e` twice when `p`
    /// welds to an endpoint).
    pub fn split_edge(&mut self, e: EdgeId, p: V3) -> (EdgeId, EdgeId) {
        let (v0, v1) = (self.edge(e).v0, self.edge(e).v1);
        let mid = self.vertex(p);
        if mid == v0 || mid == v1 {
            return (e, e);
        }
        let incident = self.edge(e).faces.clone();
        self.remove_edge(e);
        let e0 = self.link_edge(v0, mid);
        let e1 = self.link_edge(mid, v1);
        for f in incident {
            self.insert_into_face_loops(f, v0, v1, mid);
            for half in [e0, e1] {
                let fs = &mut self.es[half.0 as usize].faces;
                if !fs.contains(&f) {
                    fs.push(f);
                }
            }
        }
        (e0, e1)
    }

    fn insert_into_face_loops(&mut self, f: FaceId, v0: VertexId, v1: VertexId, mid: VertexId) {
        let face = &mut self.fs[f.0 as usize];
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

    // ---- bulk construction (imports, .igz load) --------------------------------

    /// `bulk_weld`: weld many points at once, the exact equivalent of
    /// `vertex()` on each in order but in Python's vectorized ORDER: cells
    /// with no neighbouring cell (in the batch or the registry) get their
    /// vertex first, in order of first appearance, at the cell's first
    /// point; then the points of the other ("candidate") cells replay the
    /// sequential lookup-probe-create in input order. Returns the distinct
    /// vertices touched (`vobjs`, in first-touch order) and, per point, its
    /// index in that list.
    pub fn bulk_weld(&mut self, pos: &[V3]) -> (Vec<VertexId>, Vec<usize>) {
        if pos.is_empty() {
            return (Vec::new(), Vec::new());
        }
        let keys: Vec<Key> = pos.iter().map(|&p| key(p)).collect();
        // Unique cells in creation (first appearance) order.
        let mut cell_of: HashMap<Key, usize> = HashMap::new();
        let mut first: Vec<usize> = Vec::new();
        let mut inverse: Vec<usize> = Vec::with_capacity(pos.len());
        for (i, k) in keys.iter().enumerate() {
            let c = *cell_of.entry(*k).or_insert_with(|| {
                first.push(i);
                first.len() - 1
            });
            inverse.push(c);
        }
        let nu = first.len();
        let mut candidate = vec![false; nu];
        for (c, &fi) in first.iter().enumerate() {
            let k = keys[fi];
            'n: for dx in -1..=1 {
                for dy in -1..=1 {
                    for dz in -1..=1 {
                        if (dx, dy, dz) != (0, 0, 0) && cell_of.contains_key(&(k.0 + dx, k.1 + dy, k.2 + dz)) {
                            candidate[c] = true;
                            break 'n;
                        }
                    }
                }
            }
        }
        if !self.registry.is_empty() {
            for (c, &fi) in first.iter().enumerate() {
                if candidate[c] {
                    continue;
                }
                let k = keys[fi];
                'r: for dx in -1..=1 {
                    for dy in -1..=1 {
                        for dz in -1..=1 {
                            if self.registry.contains_key(&(k.0 + dx, k.1 + dy, k.2 + dz)) {
                                candidate[c] = true;
                                break 'r;
                            }
                        }
                    }
                }
            }
        }
        let mut vobjs: Vec<VertexId> = Vec::new();
        let mut vindex: HashMap<VertexId, usize> = HashMap::new();
        let mut vid = |v: VertexId, vobjs: &mut Vec<VertexId>| -> usize {
            *vindex.entry(v).or_insert_with(|| {
                vobjs.push(v);
                vobjs.len() - 1
            })
        };
        let mut point_ids = vec![0usize; pos.len()];
        let mut cell_vid = vec![usize::MAX; nu];
        for c in 0..nu {
            if candidate[c] {
                continue;
            }
            let p = pos[first[c]];
            let v = VertexId(self.vs.len() as u32);
            self.vs.push(Vertex { position: p, edges: Vec::new() });
            self.registry.insert(keys[first[c]], v);
            self.vertices.push(v);
            cell_vid[c] = vid(v, &mut vobjs);
        }
        for i in 0..pos.len() {
            if !candidate[inverse[i]] {
                point_ids[i] = cell_vid[inverse[i]];
            }
        }
        for i in 0..pos.len() {
            if !candidate[inverse[i]] {
                continue;
            }
            let k = keys[i];
            let v = match self.registry.get(&k) {
                Some(&v) => v,
                None => match self.probe_neighbours(k, pos[i]) {
                    Some(v) => v,
                    None => {
                        let v = VertexId(self.vs.len() as u32);
                        self.vs.push(Vertex { position: pos[i], edges: Vec::new() });
                        self.registry.insert(k, v);
                        self.vertices.push(v);
                        v
                    }
                },
            };
            point_ids[i] = vid(v, &mut vobjs);
        }
        (vobjs, point_ids)
    }

    fn new_edge(&mut self, v0: VertexId, v1: VertexId) -> EdgeId {
        let e = EdgeId(self.es.len() as u32);
        self.es.push(Edge { v0, v1, faces: Vec::new(), soft: false, hidden: false, layer: None, curve: None });
        self.vs[v0.0 as usize].edges.push(e);
        if v1 != v0 {
            self.vs[v1.0 as usize].edges.push(e);
        }
        self.edges.push(e);
        e
    }

    /// `add_edges_welded`: the edges `(vobjs[ia[i]], vobjs[ib[i]])`, welded
    /// pairs skipped, duplicates within the call shared; `flags[i]` is
    /// `(soft, curve, layer, hidden)`. Returns the pair → edge map for
    /// `add_faces_welded`.
    pub fn add_edges_welded(
        &mut self,
        vobjs: &[VertexId],
        ia: &[usize],
        ib: &[usize],
        flags: Option<&[EdgeFlags]>,
    ) -> HashMap<(usize, usize), EdgeId> {
        self.touch();
        let mut emap: HashMap<(usize, usize), EdgeId> = HashMap::new();
        for i in 0..ia.len() {
            let (a, b) = (ia[i], ib[i]);
            let (v0, v1) = (vobjs[a], vobjs[b]);
            if v0 == v1 {
                continue;
            }
            let k = if a <= b { (a, b) } else { (b, a) };
            let e = match emap.get(&k) {
                Some(&e) => e,
                None => {
                    let e = self.new_edge(v0, v1);
                    emap.insert(k, e);
                    e
                }
            };
            if let Some(fl) = flags {
                let f = &fl[i];
                let ed = &mut self.es[e.0 as usize];
                if f.soft {
                    ed.soft = true;
                }
                if f.hidden {
                    ed.hidden = true;
                }
                if f.curve.is_some() {
                    ed.curve = f.curve;
                }
                if f.layer.is_some() {
                    ed.layer = f.layer.clone();
                }
            }
        }
        emap
    }

    /// `add_faces_welded`: faces from pre-welded corners (`inverse` indexes
    /// `vobjs`; rings concatenated face by face, outer first). Boundary
    /// edges are found or made in order of first segment occurrence; every
    /// segment registers its face on its edge in global segment order.
    pub fn add_faces_welded(
        &mut self,
        vobjs: &[VertexId],
        inverse: &[usize],
        ring_sizes: &[usize],
        face_ring_counts: &[usize],
        attrs: Option<&[Vec<(String, Value)>]>,
        edge_map: Option<HashMap<(usize, usize), EdgeId>>,
    ) -> Vec<FaceId> {
        self.touch();
        if face_ring_counts.is_empty() {
            return Vec::new();
        }
        let mut ring_starts = Vec::with_capacity(ring_sizes.len());
        let mut acc = 0usize;
        for &n in ring_sizes {
            ring_starts.push(acc);
            acc += n;
        }
        let total = acc;
        // Each corner to the next within its ring.
        let mut seg_b = vec![0usize; total];
        for (r, &st) in ring_starts.iter().enumerate() {
            let n = ring_sizes[r];
            for j in 0..n {
                seg_b[st + j] = inverse[st + (j + 1) % n];
            }
        }
        let mut emap = match edge_map {
            Some(m) => m,
            None => {
                let mut m = HashMap::new();
                let vid_of: HashMap<VertexId, usize> = vobjs.iter().enumerate().map(|(i, &v)| (v, i)).collect();
                for &e in &self.edges {
                    let ed = &self.es[e.0 as usize];
                    if let (Some(&i0), Some(&i1)) = (vid_of.get(&ed.v0), vid_of.get(&ed.v1)) {
                        m.entry(if i0 <= i1 { (i0, i1) } else { (i1, i0) }).or_insert(e);
                    }
                }
                m
            }
        };
        // Unique segments in order of first occurrence.
        let mut seg_edge: Vec<EdgeId> = Vec::with_capacity(total);
        let mut seen: HashMap<(usize, usize), EdgeId> = HashMap::new();
        for s in 0..total {
            let (a, b) = (inverse[s], seg_b[s]);
            let k = (a.min(b), a.max(b));
            let e = match seen.get(&k) {
                Some(&e) => e,
                None => {
                    let e = match emap.get(&k) {
                        Some(&e) => e,
                        None => {
                            let e = self.new_edge(vobjs[a], vobjs[b]);
                            emap.insert(k, e);
                            e
                        }
                    };
                    seen.insert(k, e);
                    e
                }
            };
            seg_edge.push(e);
        }
        let corners: Vec<VertexId> = inverse.iter().map(|&i| vobjs[i]).collect();
        let mut faces_out = Vec::with_capacity(face_ring_counts.len());
        let mut seg_face = Vec::with_capacity(total);
        let mut ri = 0usize;
        for (fi, &nr) in face_ring_counts.iter().enumerate() {
            let ring = |k: usize| corners[ring_starts[k]..ring_starts[k] + ring_sizes[k]].to_vec();
            let outer = ring(ri);
            let holes: Vec<Vec<VertexId>> = (1..nr).map(|k| ring(ri + k)).collect();
            let f = FaceId(self.fs.len() as u32);
            let mut face = Face { outer, holes, interior: false, attrs: Vec::new() };
            if let Some(a) = attrs {
                for (k, v) in &a[fi] {
                    match face.attrs.iter_mut().find(|(x, _)| x == k) {
                        Some(slot) => slot.1 = v.clone(),
                        None => face.attrs.push((k.clone(), v.clone())),
                    }
                }
            }
            self.fs.push(face);
            self.faces.push(f);
            faces_out.push(f);
            for k in 0..nr {
                seg_face.extend(std::iter::repeat_n(f, ring_sizes[ri + k]));
            }
            ri += nr;
        }
        for s in 0..total {
            self.es[seg_edge[s].0 as usize].faces.push(seg_face[s]);
        }
        faces_out
    }

    // ---- snapshots -----------------------------------------------------------

    /// `capture_state`.
    pub fn capture_state(&self) -> State {
        State {
            vertices: self.vertices.clone(),
            edges: self.edges.clone(),
            faces: self.faces.clone(),
            registry: self.registry.clone(),
            vpos: self.vertices.iter().map(|&v| (v, self.position(v), self.vs[v.0 as usize].edges.clone())).collect(),
            edata: self
                .edges
                .iter()
                .map(|&e| {
                    let ed = self.edge(e);
                    (e, ed.faces.clone(), ed.soft, ed.hidden, ed.layer.clone(), ed.curve)
                })
                .collect(),
            fdata: self.faces.iter().map(|&f| (f, self.face(f).clone())).collect(),
        }
    }

    /// `restore_state`: objects made since are dropped from the lists, the
    /// captured ones come back with their fields.
    pub fn restore_state(&mut self, s: &State) {
        self.touch();
        self.vertices = s.vertices.clone();
        self.edges = s.edges.clone();
        self.faces = s.faces.clone();
        self.registry = s.registry.clone();
        self.displaced.clear();
        for (v, p, es) in &s.vpos {
            let vx = &mut self.vs[v.0 as usize];
            vx.position = *p;
            vx.edges = es.clone();
        }
        for (e, fs, soft, hidden, layer, curve) in &s.edata {
            let ed = &mut self.es[e.0 as usize];
            ed.faces = fs.clone();
            ed.soft = *soft;
            ed.hidden = *hidden;
            ed.layer = layer.clone();
            ed.curve = *curve;
        }
        for (f, face) in &s.fdata {
            self.fs[f.0 as usize] = face.clone();
        }
    }

    /// `clear`.
    pub fn clear(&mut self) {
        self.touch();
        self.registry.clear();
        self.displaced.clear();
        self.vertices.clear();
        self.edges.clear();
        self.faces.clear();
    }
}
