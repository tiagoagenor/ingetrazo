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

use std::collections::{BTreeSet, HashMap, HashSet};

use ingetrazo_formats::pyjson::Value;
use ingetrazo_geom::meshface;
use ingetrazo_math::{Hypot, V3};

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

/// One open journal transaction: the lists, registry and displaced queue
/// as they were when it began, and each entity's value before its first
/// change in it (entities created inside it have none).
#[derive(Clone, Debug, Default)]
struct Txn {
    lists: (Vec<VertexId>, Vec<EdgeId>, Vec<FaceId>),
    registry: HashMap<Key, VertexId>,
    displaced: HashMap<Key, Vec<VertexId>>,
    arena: (usize, usize, usize),
    vold: HashMap<VertexId, Vertex>,
    eold: HashMap<EdgeId, Edge>,
    fold: HashMap<FaceId, Face>,
}

/// A committed change to a mesh (one command): enough to undo it and redo
/// it with the same ids, list order and weld state — Python's undo
/// "relinks the very objects", and so does this.
#[derive(Clone, Debug)]
pub struct Entry {
    lists: [(Vec<VertexId>, Vec<EdgeId>, Vec<FaceId>); 2],
    registry: [HashMap<Key, VertexId>; 2],
    displaced: [HashMap<Key, Vec<VertexId>>; 2],
    verts: Vec<(VertexId, Option<Vertex>, Vertex)>,
    edges: Vec<(EdgeId, Option<Edge>, Edge)>,
    faces: Vec<(FaceId, Option<Face>, Face)>,
}

impl Entry {
    /// How many entities the change touched (vertices, edges, faces).
    pub fn touched(&self) -> (usize, usize, usize) {
        (self.verts.len(), self.edges.len(), self.faces.len())
    }
}

#[derive(Clone, Debug, Default)]
pub struct Mesh {
    /// This Qt's `std::hypot` formula, for the lengths the mesh computes
    /// itself (the nested-hole test of `add_face`).
    pub hypot: ingetrazo_math::Hypot,
    txn: Option<Box<Txn>>,
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

    // ---- journal -------------------------------------------------------------

    fn vm(&mut self, v: VertexId) -> &mut Vertex {
        if let Some(t) = self.txn.as_deref_mut() {
            if (v.0 as usize) < t.arena.0 && !t.vold.contains_key(&v) {
                t.vold.insert(v, self.vs[v.0 as usize].clone());
            }
        }
        &mut self.vs[v.0 as usize]
    }

    fn em(&mut self, e: EdgeId) -> &mut Edge {
        if let Some(t) = self.txn.as_deref_mut() {
            if (e.0 as usize) < t.arena.1 && !t.eold.contains_key(&e) {
                t.eold.insert(e, self.es[e.0 as usize].clone());
            }
        }
        &mut self.es[e.0 as usize]
    }

    fn fm(&mut self, f: FaceId) -> &mut Face {
        if let Some(t) = self.txn.as_deref_mut() {
            if (f.0 as usize) < t.arena.2 && !t.fold.contains_key(&f) {
                t.fold.insert(f, self.fs[f.0 as usize].clone());
            }
        }
        &mut self.fs[f.0 as usize]
    }

    /// Start recording every change, for one undoable step.
    pub fn begin(&mut self) {
        assert!(self.txn.is_none(), "a journal transaction is already open");
        self.txn = Some(Box::new(Txn {
            lists: (self.vertices.clone(), self.edges.clone(), self.faces.clone()),
            registry: self.registry.clone(),
            displaced: self.displaced.clone(),
            arena: (self.vs.len(), self.es.len(), self.fs.len()),
            ..Txn::default()
        }));
    }

    /// Close the transaction: the entry that undoes and redoes it.
    pub fn commit(&mut self) -> Entry {
        let t = *self.txn.take().expect("no journal transaction open");
        let mut verts: Vec<(VertexId, Option<Vertex>, Vertex)> =
            t.vold.into_iter().map(|(v, old)| (v, Some(old), self.vs[v.0 as usize].clone())).collect();
        verts.extend((t.arena.0..self.vs.len()).map(|i| (VertexId(i as u32), None, self.vs[i].clone())));
        let mut edges: Vec<(EdgeId, Option<Edge>, Edge)> =
            t.eold.into_iter().map(|(e, old)| (e, Some(old), self.es[e.0 as usize].clone())).collect();
        edges.extend((t.arena.1..self.es.len()).map(|i| (EdgeId(i as u32), None, self.es[i].clone())));
        let mut faces: Vec<(FaceId, Option<Face>, Face)> =
            t.fold.into_iter().map(|(f, old)| (f, Some(old), self.fs[f.0 as usize].clone())).collect();
        faces.extend((t.arena.2..self.fs.len()).map(|i| (FaceId(i as u32), None, self.fs[i].clone())));
        Entry {
            lists: [t.lists, (self.vertices.clone(), self.edges.clone(), self.faces.clone())],
            registry: [t.registry, self.registry.clone()],
            displaced: [t.displaced, self.displaced.clone()],
            verts,
            edges,
            faces,
        }
    }

    /// Put the mesh back as it was before `entry` (its ids and order too).
    /// Entities the entry created stay in the arena, out of every list.
    pub fn undo(&mut self, entry: &Entry) {
        assert!(self.txn.is_none(), "undo inside an open transaction");
        self.apply(entry, 0);
    }

    /// Re-apply `entry` after an undo: the same ids, order and values.
    pub fn redo(&mut self, entry: &Entry) {
        assert!(self.txn.is_none(), "redo inside an open transaction");
        self.apply(entry, 1);
    }

    fn apply(&mut self, entry: &Entry, side: usize) {
        self.touch();
        let (v, e, f) = &entry.lists[side];
        self.vertices = v.clone();
        self.edges = e.clone();
        self.faces = f.clone();
        self.registry = entry.registry[side].clone();
        self.displaced = entry.displaced[side].clone();
        for (id, old, new) in &entry.verts {
            if let Some(x) = if side == 0 { old.as_ref() } else { Some(new) } {
                self.vs[id.0 as usize] = x.clone();
            }
        }
        for (id, old, new) in &entry.edges {
            if let Some(x) = if side == 0 { old.as_ref() } else { Some(new) } {
                self.es[id.0 as usize] = x.clone();
            }
        }
        for (id, old, new) in &entry.faces {
            if let Some(x) = if side == 0 { old.as_ref() } else { Some(new) } {
                self.fs[id.0 as usize] = x.clone();
            }
        }
    }

    // ---- arenas --------------------------------------------------------------

    pub fn vertex_ref(&self, v: VertexId) -> &Vertex {
        &self.vs[v.0 as usize]
    }
    pub fn edge(&self, e: EdgeId) -> &Edge {
        &self.es[e.0 as usize]
    }
    pub fn edge_mut(&mut self, e: EdgeId) -> &mut Edge {
        self.em(e)
    }
    pub fn face(&self, f: FaceId) -> &Face {
        &self.fs[f.0 as usize]
    }
    pub fn face_mut(&mut self, f: FaceId) -> &mut Face {
        self.fm(f)
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
        self.vm(a).edges.push(e);
        if b != a {
            self.vm(b).edges.push(e);
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
        remove_first(&mut self.vm(v0).edges, &e);
        remove_first(&mut self.vm(v1).edges, &e);
        remove_first(&mut self.edges, &e);
    }

    /// `relink_edge`: put a removed edge back (undo), same id.
    pub fn relink_edge(&mut self, e: EdgeId) {
        if self.edges.contains(&e) {
            return;
        }
        let (v0, v1) = (self.edge(e).v0, self.edge(e).v1);
        for v in [v0, v1] {
            let es = &mut self.vm(v).edges;
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
    /// With two or more holes, those nested in a bigger hole of the same
    /// face are dropped first (`topology._maximal_holes`), as in Python.
    pub fn add_face(&mut self, outer: &[V3], holes: &[Vec<V3>]) -> FaceId {
        self.touch();
        // Nested holes are dropped at this choke point, as Python does
        // (`topology._maximal_holes`) when there are two or more.
        let kept: Vec<Vec<V3>>;
        let holes = if holes.len() > 1 {
            kept = crate::topology::maximal_holes(holes, self.hypot).into_iter().map(|i| holes[i].clone()).collect();
            &kept[..]
        } else {
            holes
        };
        let lp: Vec<VertexId> = outer.iter().map(|&p| self.vertex(p)).collect();
        let hs: Vec<Vec<VertexId>> = holes.iter().map(|h| h.iter().map(|&p| self.vertex(p)).collect()).collect();
        let f = FaceId(self.fs.len() as u32);
        self.fs.push(Face { outer: lp.clone(), holes: hs.clone(), interior: false, attrs: Vec::new() });
        for l in std::iter::once(&lp).chain(hs.iter()) {
            let n = l.len();
            for i in 0..n {
                let e = self.link_edge(l[i], l[(i + 1) % n]);
                self.em(e).faces.push(f);
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
                    remove_first(&mut self.em(e).faces, &f);
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
                let fs = &mut self.em(e).faces;
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
        self.fm(f).holes.push(l.clone());
        let n = l.len();
        for i in 0..n {
            let e = self.link_edge(l[i], l[(i + 1) % n]);
            let fs = &mut self.em(e).faces;
            if !fs.contains(&f) {
                fs.push(f);
            }
        }
        l
    }

    /// `remove_hole`.
    pub fn remove_hole(&mut self, f: FaceId, l: &[VertexId]) {
        let holes = &mut self.fm(f).holes;
        if let Some(i) = holes.iter().position(|h| h.as_slice() == l) {
            holes.remove(i);
        }
        let n = l.len();
        for i in 0..n {
            if let Some(e) = self.find_edge(l[i], l[(i + 1) % n]) {
                remove_first(&mut self.em(e).faces, &f);
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
        self.vm(v).position = p;
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
                let fs = &mut self.em(half).faces;
                if !fs.contains(&f) {
                    fs.push(f);
                }
            }
        }
        (e0, e1)
    }

    fn insert_into_face_loops(&mut self, f: FaceId, v0: VertexId, v1: VertexId, mid: VertexId) {
        let face = self.fm(f);
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
        self.vm(v0).edges.push(e);
        if v1 != v0 {
            self.vm(v1).edges.push(e);
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
                let ed = self.em(e);
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
            self.em(seg_edge[s]).faces.push(seg_face[s]);
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
            let vx = self.vm(*v);
            vx.position = *p;
            vx.edges = es.clone();
        }
        for (e, fs, soft, hidden, layer, curve) in &s.edata {
            let ed = self.em(*e);
            ed.faces = fs.clone();
            ed.soft = *soft;
            ed.hidden = *hidden;
            ed.layer = layer.clone();
            ed.curve = *curve;
        }
        for (f, face) in &s.fdata {
            *self.fm(*f) = face.clone();
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

// ---- stitch primitives (core/mesh.py) -------------------------------------------
//
// Order notes. Python keeps `Vertex.edges` and the face sets of these methods
// in `set`s, iterated in address order that no run repeats. Everything below
// that iterates one uses ASCENDING ID (creation order: an edge or face keeps
// its arena index through remove/relink) — the order the parity fixtures
// force on Python (`core.mesh.set` replaced by a set that iterates in
// creation order). `Mesh::vertex_ref(v).edges` itself stays in insertion
// order; only `sorted_edges` is the Python-equivalent view.

/// Distance tolerance of the stitch pass (`_STITCH_TOL`).
pub const STITCH_TOL: f64 = 1e-4;

/// `core.mesh.PAINT_KEYS`.
pub const PAINT_KEYS: [&str; 4] = ["color", "mat", "texture", "opacity"];

/// A reversible stitch primitive's record (`rec` dicts of `core.mesh`),
/// consumed by [`Mesh::revert_stitch`].
#[derive(Clone, Debug)]
pub enum StitchRec {
    Split { edge: EdgeId, e0: EdgeId, e1: EdgeId, e0_new: bool, e1_new: bool, mid: VertexId, faces: Vec<FaceId> },
    Collapse { v: VertexId, e1: EdgeId, e2: EdgeId, n1: VertexId, n2: VertexId, new_edge: EdgeId, new_edge_new: bool, faces: Vec<FaceId> },
    Merge { edge: EdgeId, face_a: FaceId, face_b: FaceId, merged: FaceId },
    MergePair { face_a: FaceId, face_b: FaceId, edges: Vec<EdgeId>, merged: FaceId },
}

fn dict_set(d: &mut Vec<(String, Value)>, k: &str, v: Value) {
    match d.iter_mut().find(|(x, _)| x == k) {
        Some(slot) => slot.1 = v,
        None => d.push((k.to_string(), v)),
    }
}

/// `core.mesh.face_count` (a deferred mesh does not exist here).
pub fn face_count(m: &Mesh) -> usize {
    m.faces.len()
}

/// `core.mesh.vertex_count`.
pub fn vertex_count(m: &Mesh) -> usize {
    m.vertices.len()
}

/// `core.mesh.vertex_positions`: every position in `mesh.vertices` order.
pub fn vertex_positions(m: &Mesh) -> Vec<V3> {
    m.vertices.iter().map(|&v| m.position(v)).collect()
}

/// `core.mesh.face_attrs`: each face's attrs, in face order.
pub fn face_attrs(m: &Mesh) -> Vec<&Vec<(String, Value)>> {
    m.faces.iter().map(|&f| &m.face(f).attrs).collect()
}

/// `_shared_edge_index`: `i` where `loop[i]`-`loop[i+1]` is the edge `{v0, v1}`.
pub fn shared_edge_index(l: &[VertexId], v0: VertexId, v1: VertexId) -> Option<usize> {
    let n = l.len();
    for i in 0..n {
        let (a, b) = (l[i], l[(i + 1) % n]);
        if (a == v0 && b == v1) || (a == v1 && b == v0) {
            return Some(i);
        }
    }
    None
}

/// `_splice_loops`: two loops sharing the edge `{v0, v1}` merged into one,
/// the shared edge dropped; `None` unless the winding is consistent and the
/// result repeats no vertex.
pub fn splice_loops(loop_a: &[VertexId], loop_b: &[VertexId], v0: VertexId, v1: VertexId) -> Option<Vec<VertexId>> {
    let ia = shared_edge_index(loop_a, v0, v1)?;
    let ib = shared_edge_index(loop_b, v0, v1)?;
    let (na, nb) = (loop_a.len(), loop_b.len());
    let (u, w) = (loop_a[ia], loop_a[(ia + 1) % na]);
    if !(loop_b[ib] == w && loop_b[(ib + 1) % nb] == u) {
        return None;
    }
    let sa = (ia + 1) % na;
    let a_path: Vec<VertexId> = loop_a[sa..].iter().chain(loop_a[..sa].iter()).copied().collect();
    let sb = (ib + 1) % nb;
    let b_path: Vec<VertexId> = loop_b[sb..].iter().chain(loop_b[..sb].iter()).copied().collect();
    let mut merged: Vec<VertexId> = a_path[..a_path.len() - 1].to_vec();
    merged.extend_from_slice(&b_path[..b_path.len() - 1]);
    let distinct: HashSet<VertexId> = merged.iter().copied().collect();
    if distinct.len() != merged.len() {
        return None;
    }
    Some(merged)
}

fn pair_key(a: VertexId, b: VertexId) -> (VertexId, VertexId) {
    if a <= b {
        (a, b)
    } else {
        (b, a)
    }
}

/// `_trace_loops`: undirected boundary edges traced into closed vertex loops;
/// `None` if any vertex has valence other than 2.
///
/// Python starts each loop at `tuple(next(iter(edges)))` of a set of
/// frozensets — address order, in either direction. Here (and in the
/// fixture generator, which swaps the same rule into Python) a loop starts
/// at the first not yet traced outline edge, `u -> w` as it was listed.
pub fn trace_loops(outline: &[(VertexId, VertexId)]) -> Option<Vec<Vec<VertexId>>> {
    let mut adj: HashMap<VertexId, Vec<VertexId>> = HashMap::new();
    let mut edges: Vec<(VertexId, VertexId)> = Vec::new();
    let mut seen: HashSet<(VertexId, VertexId)> = HashSet::new();
    for &(u, w) in outline {
        if u == w {
            return None;
        }
        adj.entry(u).or_default().push(w);
        adj.entry(w).or_default().push(u);
        if seen.insert(pair_key(u, w)) {
            edges.push((u, w));
        }
    }
    if adj.values().any(|n| n.len() != 2) {
        return None;
    }
    let mut remaining = seen;
    let mut cursor = 0usize;
    let mut loops: Vec<Vec<VertexId>> = Vec::new();
    loop {
        while cursor < edges.len() && !remaining.contains(&pair_key(edges[cursor].0, edges[cursor].1)) {
            cursor += 1;
        }
        if cursor >= edges.len() {
            break;
        }
        let (a, b) = edges[cursor];
        remaining.remove(&pair_key(a, b));
        let mut l = vec![a, b];
        let (mut prev, mut cur) = (a, b);
        loop {
            let nbrs = &adj[&cur];
            let nxt = if nbrs[0] != prev { nbrs[0] } else { nbrs[1] };
            remaining.remove(&pair_key(cur, nxt));
            if nxt == l[0] {
                break;
            }
            l.push(nxt);
            prev = cur;
            cur = nxt;
        }
        loops.push(l);
    }
    Some(loops)
}

/// `_loop_area`: Newell-magnitude area of a vertex loop (orientation-free).
pub fn loop_area(m: &Mesh, l: &[VertexId], h: Hypot) -> f64 {
    let mut n = V3(0.0, 0.0, 0.0);
    let count = l.len();
    for i in 0..count {
        let (c, x) = (m.position(l[i]), m.position(l[(i + 1) % count]));
        let (cx, cy, cz) = (c.0 as f64, c.1 as f64, c.2 as f64);
        let (nx, ny, nz) = (x.0 as f64, x.1 as f64, x.2 as f64);
        n = n.add(V3(((cy - ny) * (cz + nz)) as f32, ((cz - nz) * (cx + nx)) as f32, ((cx - nx) * (cy + ny)) as f32));
    }
    0.5 * (n.length(h) as f64)
}

/// `_orient_union_loops`: put the traced union back on the source face's
/// front side (outer along `normal`, holes against it).
pub fn orient_union_loops(m: &Mesh, outer: &mut [VertexId], holes: &mut [Vec<VertexId>], normal: V3, h: Hypot) {
    if m.loop_normal(outer, h).dot(normal) < 0.0 {
        outer.reverse();
    }
    for hole in holes.iter_mut() {
        if m.loop_normal(hole, h).dot(normal) > 0.0 {
            hole.reverse();
        }
    }
}

/// `_dominant_attrs`: the attrs of the largest-area face that carries any.
pub fn dominant_attrs(m: &Mesh, faces: &[FaceId], h: Hypot) -> Vec<(String, Value)> {
    let mut best: Vec<(String, Value)> = Vec::new();
    let mut best_area = -1.0f64;
    for &f in faces {
        if !m.face(f).attrs.is_empty() && m.face_area(f, h) > best_area {
            best = m.face(f).attrs.clone();
            best_area = m.face_area(f, h);
        }
    }
    best
}

impl Mesh {
    // ---- helpers ---------------------------------------------------------------

    /// A vertex's incident edges in ascending id: Python's `set`, iterated in
    /// creation order (see the note above).
    pub fn sorted_edges(&self, v: VertexId) -> Vec<EdgeId> {
        let mut es = self.vs[v.0 as usize].edges.clone();
        es.sort();
        es
    }

    /// `Edge.other`.
    pub fn other(&self, e: EdgeId, v: VertexId) -> VertexId {
        let ed = self.edge(e);
        if v == ed.v0 {
            ed.v1
        } else {
            ed.v0
        }
    }

    fn loop_positions(&self, l: &[VertexId]) -> Vec<V3> {
        l.iter().map(|&v| self.position(v)).collect()
    }

    /// `Face(loop).normal()` of a loop of this mesh's vertices.
    pub fn loop_normal(&self, l: &[VertexId], h: Hypot) -> V3 {
        meshface::normal(&self.loop_positions(l), h)
    }

    /// `Face.normal()`.
    pub fn face_normal(&self, f: FaceId, h: Hypot) -> V3 {
        self.loop_normal(&self.face(f).outer, h)
    }

    /// `Face.area()`.
    pub fn face_area(&self, f: FaceId, h: Hypot) -> f64 {
        let outer = &self.face(f).outer;
        if outer.len() < 3 {
            return 0.0;
        }
        0.5 * (meshface::newell(&self.loop_positions(outer)).length(h) as f64)
    }

    // ---- curves ----------------------------------------------------------------

    /// `Mesh.next_curve_id`. `counter` is `core.mesh._CURVE_COUNTER`, which
    /// is process-wide in Python (and not undone with the mesh); here the
    /// caller owns it, starting at 1.
    pub fn next_curve_id(counter: &mut i64) -> i64 {
        let cid = *counter;
        *counter += 1;
        cid
    }

    /// `tag_curve`: every edge lying along the drawn path becomes one curve.
    pub fn tag_curve(&mut self, counter: &mut i64, loop_points: &[V3], closed: bool, h: Hypot) -> Option<i64> {
        self.touch();
        let pts = loop_points;
        if pts.len() < 2 {
            return None;
        }
        let cid = *counter;
        let mut pairs: Vec<(V3, V3)> = pts.windows(2).map(|w| (w[0], w[1])).collect();
        if closed {
            pairs.push((pts[pts.len() - 1], pts[0]));
        }
        let mut tagged = false;
        for e in self.edges.clone() {
            if self.edge(e).curve.is_some() {
                continue;
            }
            let (a, b) = (self.position(self.edge(e).v0), self.position(self.edge(e).v1));
            for &(pa, pb) in &pairs {
                if crate::stitch::point_on_seg_incl(a, pa, pb, h) && crate::stitch::point_on_seg_incl(b, pa, pb, h) {
                    self.em(e).curve = Some(cid);
                    tagged = true;
                    break;
                }
            }
        }
        if tagged {
            *counter += 1;
            Some(cid)
        } else {
            None
        }
    }

    /// `curve_edges`: all edges sharing `e`'s curve id (`[e]` if none).
    pub fn curve_edges(&self, e: EdgeId) -> Vec<EdgeId> {
        match self.edge(e).curve {
            None => vec![e],
            Some(cid) => self.edges.iter().copied().filter(|&x| self.edge(x).curve == Some(cid)).collect(),
        }
    }

    /// `resplit_curves`: curve ids re-partitioned into contiguous contours.
    pub fn resplit_curves(&mut self, counter: &mut i64) {
        self.touch();
        let mut order: Vec<i64> = Vec::new();
        let mut by_id: HashMap<i64, Vec<EdgeId>> = HashMap::new();
        for &e in &self.edges {
            if let Some(c) = self.es[e.0 as usize].curve {
                by_id
                    .entry(c)
                    .or_insert_with(|| {
                        order.push(c);
                        Vec::new()
                    })
                    .push(e);
            }
        }
        for cid in order {
            let edges = by_id[&cid].clone();
            let eset: HashSet<EdgeId> = edges.iter().copied().collect();
            let mut unvisited = eset.clone();
            let chains = {
                let is_break = |v: VertexId| -> bool {
                    let es = &self.vs[v.0 as usize].edges;
                    let cdeg = es.iter().filter(|e| eset.contains(e)).count();
                    cdeg != 2 || es.len() != cdeg
                };
                let mut chains: Vec<Vec<EdgeId>> = Vec::new();
                let seeds: Vec<EdgeId> = edges
                    .iter()
                    .copied()
                    .filter(|&e| is_break(self.es[e.0 as usize].v0) || is_break(self.es[e.0 as usize].v1))
                    .collect();
                for &seed in seeds.iter().chain(edges.iter()) {
                    if !unvisited.contains(&seed) {
                        continue;
                    }
                    let sed = &self.es[seed.0 as usize];
                    let start_v = if is_break(sed.v0) {
                        sed.v0
                    } else if is_break(sed.v1) {
                        sed.v1
                    } else {
                        sed.v0
                    };
                    let (mut chain, mut v, mut e) = (Vec::new(), start_v, seed);
                    while unvisited.contains(&e) {
                        unvisited.remove(&e);
                        chain.push(e);
                        v = self.other(e, v);
                        if is_break(v) {
                            break;
                        }
                        match self.sorted_edges(v).into_iter().find(|n| unvisited.contains(n)) {
                            None => break,
                            Some(n) => e = n,
                        }
                    }
                    chains.push(chain);
                }
                chains
            };
            if chains.len() <= 1 {
                continue;
            }
            for (i, chain) in chains.iter().enumerate() {
                let new_id = if i == 0 { cid } else { *counter };
                if i > 0 {
                    *counter += 1;
                }
                for &e in chain {
                    self.em(e).curve = Some(new_id);
                }
            }
        }
    }

    // ---- queries ---------------------------------------------------------------

    /// `surface_of`: the faces joined to `f` through soft edges. Python
    /// returns `list(set)` (address order); this is discovery order, `f`
    /// first — compare as a set.
    pub fn surface_of(&self, f: FaceId) -> Vec<FaceId> {
        let mut seen = vec![f];
        let mut stack = vec![f];
        while let Some(f) = stack.pop() {
            for l in self.loops_of(f) {
                let n = l.len();
                for i in 0..n {
                    let e = match self.find_edge(l[i], l[(i + 1) % n]) {
                        Some(e) if self.edge(e).soft => e,
                        _ => continue,
                    };
                    for &g in &self.edge(e).faces {
                        if g != f && !seen.contains(&g) {
                            seen.push(g);
                            stack.push(g);
                        }
                    }
                }
            }
        }
        seen
    }

    /// `edges_at`: a set in Python; ascending id here.
    pub fn edges_at(&self, v: VertexId) -> Vec<EdgeId> {
        self.sorted_edges(v)
    }

    /// `faces_at` (`Vertex.faces()`): a set in Python; ascending id here.
    pub fn faces_at(&self, v: VertexId) -> Vec<FaceId> {
        let mut out: Vec<FaceId> = self.vs[v.0 as usize].edges.iter().flat_map(|&e| self.es[e.0 as usize].faces.iter().copied()).collect();
        out.sort();
        out.dedup();
        out
    }

    // ---- stitch primitives -----------------------------------------------------

    /// `weld_coincident`: merge vertices sharing a position (same cell, then
    /// across a rounding boundary). Returns whether anything merged.
    pub fn weld_coincident(&mut self) -> bool {
        self.touch();
        let mut order: Vec<Key> = Vec::new();
        let mut groups: HashMap<Key, Vec<VertexId>> = HashMap::new();
        for &v in &self.vertices {
            let k = key(self.position(v));
            groups
                .entry(k)
                .or_insert_with(|| {
                    order.push(k);
                    Vec::new()
                })
                .push(v);
        }
        let mut changed = false;
        for k in order {
            let vs = &groups[&k];
            if vs.len() < 2 {
                continue;
            }
            let keep = vs[0];
            for &dup in &vs[1..] {
                self.weld_into(keep, dup);
            }
            changed = true;
        }
        let step = 10f64.powi(-KEY_DECIMALS);
        let tol2 = step * step;
        let mut merged = true;
        while merged {
            merged = false;
            for v in self.vertices.clone() {
                let k = key(self.position(v));
                if self.registry.get(&k) != Some(&v) {
                    continue;
                }
                'cells: for dx in -1..=1 {
                    for dy in -1..=1 {
                        for dz in -1..=1 {
                            let nk = (k.0 + dx, k.1 + dy, k.2 + dz);
                            if nk == k {
                                continue;
                            }
                            let w = match self.registry.get(&nk) {
                                Some(&w) => w,
                                None => continue,
                            };
                            if w == v || (self.position(w).sub(self.position(v)).length_squared() as f64) >= tol2 {
                                continue;
                            }
                            let (kp, dp) = if k < nk { (v, w) } else { (w, v) };
                            self.weld_into(kp, dp);
                            changed = true;
                            merged = true;
                            break 'cells;
                        }
                    }
                }
                if merged {
                    break;
                }
            }
        }
        changed
    }

    /// `dedupe_faces`: drop faces on an identical outer edge cycle; the one
    /// with more holes survives. Returns how many were removed.
    pub fn dedupe_faces(&mut self) -> usize {
        self.touch();
        let mut seen: HashMap<BTreeSet<(VertexId, VertexId)>, FaceId> = HashMap::new();
        let mut removed = 0;
        for f in self.faces.clone() {
            let outer = &self.face(f).outer;
            let n = outer.len();
            let sig: BTreeSet<(VertexId, VertexId)> = (0..n).map(|i| pair_key(outer[i], outer[(i + 1) % n])).collect();
            let other = match seen.get(&sig) {
                None => {
                    seen.insert(sig, f);
                    continue;
                }
                Some(&o) => o,
            };
            let (drop, keep) = if self.face(f).holes.len() > self.face(other).holes.len() { (other, f) } else { (f, other) };
            if !self.face(drop).attrs.is_empty() && self.face(keep).attrs.is_empty() {
                let a = self.face(drop).attrs.clone();
                self.fm(keep).attrs = a;
            }
            self.remove_face(drop);
            seen.insert(sig, keep);
            removed += 1;
        }
        removed
    }

    /// `_weld_into`: re-point everything on `dup` onto `keep`.
    pub fn weld_into(&mut self, keep: VertexId, dup: VertexId) {
        let dup_edges = self.sorted_edges(dup);
        // `affected` is a set in Python; the faces are independent of one
        // another, so the order they are visited in changes nothing.
        let mut affected: Vec<FaceId> = Vec::new();
        for &e in &dup_edges {
            for &f in &self.es[e.0 as usize].faces {
                if !affected.contains(&f) {
                    affected.push(f);
                }
            }
        }
        for e in dup_edges {
            let other = self.other(e, dup);
            let incident = self.es[e.0 as usize].faces.clone();
            self.remove_edge(e);
            if other == keep {
                continue;
            }
            let kept_edge = self.link_edge(keep, other);
            for f in incident {
                if !self.es[kept_edge.0 as usize].faces.contains(&f) {
                    self.em(kept_edge).faces.push(f);
                }
            }
        }
        for f in affected {
            let face = self.fm(f);
            for l in std::iter::once(&mut face.outer).chain(face.holes.iter_mut()) {
                for lv in l.iter_mut() {
                    if *lv == dup {
                        *lv = keep;
                    }
                }
                let mut i = 0;
                while l.len() > 1 && i < l.len() {
                    if l[i] == l[(i + 1) % l.len()] {
                        let at = if i + 1 < l.len() { i } else { 0 };
                        l.remove(at);
                    } else {
                        i += 1;
                    }
                }
            }
            let distinct: HashSet<VertexId> = self.fs[f.0 as usize].outer.iter().copied().collect();
            if distinct.len() < 3 {
                self.remove_face(f);
            }
        }
        remove_first(&mut self.vertices, &dup);
        let k = key(self.position(dup));
        if self.registry.get(&k) == Some(&dup) {
            self.registry.insert(k, keep);
        }
    }

    /// `interior_vertex_on`: a vertex strictly inside `edge` (a T-junction),
    /// in `mesh.vertices` order.
    pub fn interior_vertex_on(&self, edge: EdgeId, h: Hypot) -> Option<VertexId> {
        let (v0, v1) = (self.edge(edge).v0, self.edge(edge).v1);
        let a = self.position(v0);
        let b = self.position(v1);
        let ab = b.sub(a);
        let length = ab.length(h) as f64;
        if length < STITCH_TOL {
            return None;
        }
        for &v in &self.vertices {
            if v == v0 || v == v1 {
                continue;
            }
            let p = self.position(v);
            let t = (p.sub(a).dot(ab) as f64) / (length * length);
            if t <= STITCH_TOL / length || t >= 1.0 - STITCH_TOL / length {
                continue;
            }
            if (p.sub(a.add(ab.mul(t as f32))).length(h) as f64) < STITCH_TOL {
                return Some(v);
            }
        }
        None
    }

    /// `split_edge_at`: split `edge` at the existing vertex `mid`.
    pub fn split_edge_at(&mut self, edge: EdgeId, mid: VertexId) -> StitchRec {
        let (v0, v1) = (self.edge(edge).v0, self.edge(edge).v1);
        let incident = self.es[edge.0 as usize].faces.clone();
        self.remove_edge(edge);
        let e0_new = self.find_edge(v0, mid).is_none();
        let e0 = self.link_edge(v0, mid);
        let e1_new = self.find_edge(mid, v1).is_none();
        let e1 = self.link_edge(mid, v1);
        let (soft, layer, curve) = {
            let ed = self.edge(edge);
            (ed.soft, ed.layer.clone(), ed.curve)
        };
        for (sub, was_new) in [(e0, e0_new), (e1, e1_new)] {
            if was_new {
                let s = self.em(sub);
                s.soft = soft;
                s.layer = layer.clone();
                s.curve = curve;
            }
        }
        for &face in &incident {
            self.insert_into_face_loops(face, v0, v1, mid);
            for sub in [e0, e1] {
                if !self.es[sub.0 as usize].faces.contains(&face) {
                    self.em(sub).faces.push(face);
                }
            }
        }
        StitchRec::Split { edge, e0, e1, e0_new, e1_new, mid, faces: incident }
    }

    /// `collapsible_vertex`: a redundant valence-2 collinear vertex.
    pub fn collapsible_vertex(&self, v: VertexId, h: Hypot) -> bool {
        if self.vs[v.0 as usize].edges.len() != 2 {
            return false;
        }
        let es = self.sorted_edges(v);
        let (e1, e2) = (es[0], es[1]);
        let d1 = self.position(v).sub(self.position(self.other(e1, v)));
        let d2 = self.position(self.other(e2, v)).sub(self.position(v));
        if (d1.length(h) as f64) < STITCH_TOL || (d2.length(h) as f64) < STITCH_TOL {
            return false;
        }
        if (d1.normalized(h).dot(d2.normalized(h)) as f64) < 0.999 {
            return false;
        }
        let a: HashSet<FaceId> = self.edge(e1).faces.iter().copied().collect();
        let b: HashSet<FaceId> = self.edge(e2).faces.iter().copied().collect();
        a == b
    }

    /// `collapse_vertex`: remove the vertex `collapsible_vertex` approved,
    /// merging its two edges.
    pub fn collapse_vertex(&mut self, v: VertexId) -> StitchRec {
        let es = self.sorted_edges(v);
        let (e1, e2) = (es[0], es[1]);
        let (n1, n2) = (self.other(e1, v), self.other(e2, v));
        let mut faces: Vec<FaceId> = self.edge(e1).faces.clone();
        faces.sort();
        faces.dedup();
        self.remove_edge(e1);
        self.remove_edge(e2);
        let new_edge_existed = self.find_edge(n1, n2).is_some();
        let new_edge = self.link_edge(n1, n2);
        for &face in &faces {
            let f = self.fm(face);
            for l in std::iter::once(&mut f.outer).chain(f.holes.iter_mut()) {
                remove_first(l, &v);
            }
            if !self.es[new_edge.0 as usize].faces.contains(&face) {
                self.em(new_edge).faces.push(face);
            }
        }
        remove_first(&mut self.vertices, &v);
        let k = key(self.position(v));
        if self.registry.get(&k) == Some(&v) {
            self.registry.remove(&k);
        }
        StitchRec::Collapse { v, e1, e2, n1, n2, new_edge, new_edge_new: !new_edge_existed, faces }
    }

    /// `dissolve_edge_recorded`: merge the two faces of a coplanar edge,
    /// returning the reversible record.
    pub fn dissolve_edge_recorded(&mut self, edge: EdgeId, h: Hypot) -> Option<StitchRec> {
        if self.edge(edge).faces.len() != 2 {
            return None;
        }
        let (face_a, face_b) = (self.edge(edge).faces[0], self.edge(edge).faces[1]);
        if face_a == face_b {
            return None;
        }
        if (self.face_normal(face_a, h).dot(self.face_normal(face_b, h)) as f64) < 0.999 {
            return None;
        }
        let (v0, v1) = (self.edge(edge).v0, self.edge(edge).v1);
        let merged = splice_loops(&self.face(face_a).outer, &self.face(face_b).outer, v0, v1)?;
        let loop_positions = self.loop_positions(&merged);
        let hole_positions: Vec<Vec<V3>> =
            self.face(face_a).holes.iter().chain(self.face(face_b).holes.iter()).map(|hl| self.loop_positions(hl)).collect();
        self.remove_face(face_a);
        self.remove_face(face_b);
        self.remove_edge(edge);
        let new_face = self.add_face(&loop_positions, &hole_positions);
        let attrs = dominant_attrs(self, &[face_a, face_b], h);
        self.fm(new_face).attrs = attrs;
        Some(StitchRec::Merge { edge, face_a, face_b, merged: new_face })
    }

    /// `dissolve_coplanar_pair`: two coplanar faces merged into the boundary
    /// of their union, every shared edge removed.
    pub fn dissolve_coplanar_pair(&mut self, face_a: FaceId, face_b: FaceId, h: Hypot) -> Option<StitchRec> {
        if face_a == face_b {
            return None;
        }
        let n_a = self.face_normal(face_a, h);
        if (n_a.dot(self.face_normal(face_b, h)) as f64) < 0.999 {
            return None;
        }
        let mut loops: Vec<Vec<VertexId>> = self.loops_of(face_a);
        loops.extend(self.loops_of(face_b));
        // counts: insertion-ordered map of unordered vertex pairs.
        let mut counts: Vec<((VertexId, VertexId), usize)> = Vec::new();
        let mut index: HashMap<(VertexId, VertexId), usize> = HashMap::new();
        for l in &loops {
            let n = l.len();
            for i in 0..n {
                let k = pair_key(l[i], l[(i + 1) % n]);
                match index.get(&k) {
                    Some(&at) => counts[at].1 += 1,
                    None => {
                        index.insert(k, counts.len());
                        counts.push((k, 1));
                    }
                }
            }
        }
        let shared: Vec<(VertexId, VertexId)> = counts.iter().filter(|(_, c)| *c >= 2).map(|(k, _)| *k).collect();
        if shared.is_empty() {
            return None;
        }
        let mut outline: Vec<(VertexId, VertexId)> = Vec::new();
        for l in &loops {
            let n = l.len();
            for i in 0..n {
                let (u, w) = (l[i], l[(i + 1) % n]);
                if counts[index[&pair_key(u, w)]].1 == 1 {
                    outline.push((u, w));
                }
            }
        }
        let mut loops_out = trace_loops(&outline)?;
        if loops_out.is_empty() {
            // Python raises IndexError here (identical cycles); refuse instead.
            return None;
        }
        loops_out.sort_by(|a, b| loop_area(self, b, h).partial_cmp(&loop_area(self, a, h)).unwrap_or(std::cmp::Ordering::Equal));
        let mut outer = loops_out.remove(0);
        let mut holes = loops_out;
        orient_union_loops(self, &mut outer, &mut holes, n_a, h);
        let shared_edges: Vec<Option<EdgeId>> = shared.iter().map(|&(a, b)| self.find_edge(a, b)).collect();
        self.remove_face(face_a);
        self.remove_face(face_b);
        for e in shared_edges.iter().flatten() {
            self.remove_edge(*e);
        }
        let outer_pos = self.loop_positions(&outer);
        let hole_pos: Vec<Vec<V3>> = holes.iter().map(|hl| self.loop_positions(hl)).collect();
        let merged = self.add_face(&outer_pos, &hole_pos);
        let attrs = dominant_attrs(self, &[face_a, face_b], h);
        self.fm(merged).attrs = attrs;
        Some(StitchRec::MergePair { face_a, face_b, edges: shared_edges.into_iter().flatten().collect(), merged })
    }

    /// `dissolve_coplanar_region`: a set of mutually coplanar, edge-connected
    /// faces merged into one spanning the boundary of their union. `faces` is
    /// given in the order Python would list its set (the fixtures: ascending
    /// creation order).
    pub fn dissolve_coplanar_region(&mut self, faces: &[FaceId], h: Hypot) -> Option<FaceId> {
        if faces.len() < 2 {
            return None;
        }
        let n0 = self.face_normal(faces[0], h);
        if faces[1..].iter().any(|&f| (n0.dot(self.face_normal(f, h)) as f64).abs() < 0.999) {
            return None;
        }
        let mut counts: Vec<((VertexId, VertexId), usize)> = Vec::new();
        let mut index: HashMap<(VertexId, VertexId), usize> = HashMap::new();
        for &f in faces {
            for l in self.loops_of(f) {
                let n = l.len();
                for i in 0..n {
                    let k = pair_key(l[i], l[(i + 1) % n]);
                    match index.get(&k) {
                        Some(&at) => counts[at].1 += 1,
                        None => {
                            index.insert(k, counts.len());
                            counts.push((k, 1));
                        }
                    }
                }
            }
        }
        if !counts.iter().any(|(_, c)| *c >= 2) {
            return None;
        }
        let mut outline: Vec<(VertexId, VertexId)> = Vec::new();
        for &f in faces {
            for l in self.loops_of(f) {
                let n = l.len();
                for i in 0..n {
                    let (u, w) = (l[i], l[(i + 1) % n]);
                    if counts[index[&pair_key(u, w)]].1 == 1 {
                        outline.push((u, w));
                    }
                }
            }
        }
        let mut loops_out = trace_loops(&outline)?;
        if loops_out.is_empty() {
            if self.face(faces[0]).attrs.is_empty() {
                let a = dominant_attrs(self, &faces[1..], h);
                self.fm(faces[0]).attrs = a;
            }
            for &f in &faces[1..] {
                self.remove_face(f);
            }
            return Some(faces[0]);
        }
        loops_out.sort_by(|a, b| loop_area(self, b, h).partial_cmp(&loop_area(self, a, h)).unwrap_or(std::cmp::Ordering::Equal));
        let mut outer = loops_out.remove(0);
        let mut holes = loops_out;
        orient_union_loops(self, &mut outer, &mut holes, n0, h);
        let interior: Vec<Option<EdgeId>> = counts.iter().filter(|(_, c)| *c >= 2).map(|((a, b), _)| self.find_edge(*a, *b)).collect();
        for &f in faces {
            self.remove_face(f);
        }
        for e in interior.into_iter().flatten() {
            if self.edge(e).faces.is_empty() {
                self.remove_edge(e);
            }
        }
        let outer_pos = self.loop_positions(&outer);
        let hole_pos: Vec<Vec<V3>> = holes.iter().map(|hl| self.loop_positions(hl)).collect();
        let merged = self.add_face(&outer_pos, &hole_pos);
        let attrs = dominant_attrs(self, faces, h);
        self.fm(merged).attrs = attrs;
        Some(merged)
    }

    /// `revert_stitch`: undo one stitch primitive from its record.
    pub fn revert_stitch(&mut self, rec: &StitchRec) {
        match rec {
            StitchRec::Split { edge, e0, e1, e0_new, e1_new, mid, faces } => {
                for &face in faces {
                    let f = self.fm(face);
                    for l in std::iter::once(&mut f.outer).chain(f.holes.iter_mut()) {
                        remove_first(l, mid);
                    }
                }
                for (sub, was_new) in [(*e0, *e0_new), (*e1, *e1_new)] {
                    if was_new {
                        self.remove_edge(sub);
                    } else {
                        for face in faces {
                            if self.es[sub.0 as usize].faces.contains(face) {
                                remove_first(&mut self.em(sub).faces, face);
                            }
                        }
                    }
                }
                self.relink_edge(*edge);
                for &face in faces {
                    if !self.es[edge.0 as usize].faces.contains(&face) {
                        self.em(*edge).faces.push(face);
                    }
                }
            }
            StitchRec::Collapse { v, e1, e2, n1, n2, new_edge, new_edge_new, faces } => {
                if !self.vertices.contains(v) {
                    self.vertices.push(*v);
                }
                let k = key(self.position(*v));
                self.registry.entry(k).or_insert(*v);
                if *new_edge_new {
                    self.remove_edge(*new_edge);
                } else {
                    for face in faces {
                        if self.es[new_edge.0 as usize].faces.contains(face) {
                            remove_first(&mut self.em(*new_edge).faces, face);
                        }
                    }
                }
                self.relink_edge(*e1);
                self.relink_edge(*e2);
                for &face in faces {
                    self.insert_into_face_loops(face, *n1, *n2, *v);
                    for e in [*e1, *e2] {
                        if !self.es[e.0 as usize].faces.contains(&face) {
                            self.em(e).faces.push(face);
                        }
                    }
                }
            }
            StitchRec::Merge { edge, face_a, face_b, merged } => {
                self.remove_face(*merged);
                self.relink_edge(*edge);
                self.relink_face(*face_a);
                self.relink_face(*face_b);
            }
            StitchRec::MergePair { face_a, face_b, edges, merged } => {
                self.remove_face(*merged);
                for &e in edges {
                    self.relink_edge(e);
                }
                self.relink_face(*face_a);
                self.relink_face(*face_b);
            }
        }
    }

    /// `dissolve_edge`: merge the two coplanar faces on an edge into one
    /// (an opposite-wound neighbour is flipped for the splice).
    pub fn dissolve_edge(&mut self, edge: EdgeId, h: Hypot) -> Option<FaceId> {
        if self.edge(edge).faces.len() != 2 {
            return None;
        }
        let (face_a, face_b) = (self.edge(edge).faces[0], self.edge(edge).faces[1]);
        if face_a == face_b {
            return None;
        }
        let dot = self.face_normal(face_a, h).dot(self.face_normal(face_b, h)) as f64;
        if dot.abs() < 0.999 {
            return None;
        }
        let loop_b: Vec<VertexId> = if dot > 0.0 {
            self.face(face_b).outer.clone()
        } else {
            self.face(face_b).outer.iter().rev().copied().collect()
        };
        let (v0, v1) = (self.edge(edge).v0, self.edge(edge).v1);
        let merged = splice_loops(&self.face(face_a).outer, &loop_b, v0, v1)?;
        let loop_positions = self.loop_positions(&merged);
        let hole_positions: Vec<Vec<V3>> =
            self.face(face_a).holes.iter().chain(self.face(face_b).holes.iter()).map(|hl| self.loop_positions(hl)).collect();
        self.remove_face(face_a);
        self.remove_face(face_b);
        self.remove_edge(edge);
        let merged_face = self.add_face(&loop_positions, &hole_positions);
        let attrs = dominant_attrs(self, &[face_a, face_b], h);
        self.fm(merged_face).attrs = attrs;
        Some(merged_face)
    }

    /// `turn_face_over`: reverse a face's winding without moving paint from
    /// one side to the other (the back's own paint trades places).
    pub fn turn_face_over(&mut self, f: FaceId) {
        let face = self.fm(f);
        let back = match face.attrs.iter().find(|(k, _)| k == "back") {
            Some((_, Value::Object(b))) => Some(b.clone()),
            _ => None,
        };
        if let Some(back) = back {
            let mut front: Vec<(String, Value)> = Vec::new();
            for k in PAINT_KEYS {
                if let Some(i) = face.attrs.iter().position(|(x, _)| x == k) {
                    front.push(face.attrs.remove(i));
                }
            }
            if let Some(i) = face.attrs.iter().position(|(x, _)| x == "back") {
                face.attrs.remove(i);
            }
            for (k, v) in back {
                dict_set(&mut face.attrs, &k, v);
            }
            if !front.is_empty() {
                dict_set(&mut face.attrs, "back", Value::Object(front));
            }
        }
        face.outer.reverse();
        for hl in face.holes.iter_mut() {
            hl.reverse();
        }
    }
}
