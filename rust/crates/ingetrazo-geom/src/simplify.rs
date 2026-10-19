// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! `formats.fuse.simplify_mesh` (what `SimplifyMeshCommand` runs) on plain
//! arrays instead of the live `Mesh`.
//!
//! The decisions are the Python's, step for step:
//!
//! 1. faces with holes or `interior` are kept as they are; the rest go,
//!    with their attribute signature, to [`crate::fuse::fuse`]
//!    (`fuse_coplanar_loops`, `cos_tol = cos(max(radians(angle), 0.00045))`);
//! 2. the fused regions (and the faces of regions that did not trace) are
//!    re-added to a fresh welded mesh — the `Mesh` model below: vertices
//!    welded by `_key`, edges created in `add_face` order — then the holed
//!    faces;
//! 3. the flags of the old edges are stamped on the new edges with the same
//!    `frozenset((_key(a), _key(b)))`;
//! 4. `_collapse_all` removes redundant collinear valence-2 vertices, always
//!    the first collapsible one in vertex-list order (the rescan-free
//!    equivalent);
//! 5. `soften_smooth_edges` marks shallow same-material seams soft.
//!
//! The result is the new mesh as arrays ([`Simplified`]); the caller replaces
//! the live mesh with it (that is `mesh.restore_state(...)`; M3).
//!
//! Known gaps, all noted at their place:
//! - `Mesh.add_face` drops nested holes (`topology._maximal_holes`, which
//!   needs `core.triangulate`) when a face has more than one: pass
//!   `maximal_holes` (None keeps every hole);
//! - `add_face`'s `except Exception` fallback to the original loops is not
//!   reproduced (nothing in the arrays can raise);
//! - `Mesh._probe_neighbours` probes keys ±1.0 away (not ±1e-4: it adds the
//!   integer offset to the rounded float), whose vertices are a unit
//!   distance apart and can never be within the 1e-4 weld tolerance: it
//!   never matches, so the welding here is the exact-key lookup alone;
//! - `v.edges` is a Python set (id-ordered); here insertion order. It can
//!   only change which end a collapse-created edge calls `v0` and the order
//!   of that edge's face list.

use std::collections::HashMap;

use ingetrazo_math::{Hypot, V3};

use crate::fuse::{self, Region};
use crate::meshface::{ikey, normal};

/// Signature id of the faces without attributes (`_attrs_sig(None)`); the
/// caller gives the same id to any attrs whose signature equals it.
pub const NONE_SIG: i64 = -2;
/// Smoothing-signature id of the faces without attributes (`_smooth_sig` of
/// an empty or missing dict is `None`).
pub const NONE_SMOOTH: i64 = -2;

/// The per-attrs facts the pass reads. Python makes them from the attribute
/// dicts (`_attrs_sig`, `_smooth_sig`, texture groups: see
/// `formats.fuse._fuse_native`); ids are equal exactly when the Python
/// values compare equal.
#[derive(Clone, Debug)]
pub struct AttrInfo {
    pub sig: i64,
    /// Texture group (path, sw, sh, rot-less signature id) or -1.
    pub tex_group: i64,
    /// 2 textured, 1 otherwise.
    pub rank: u8,
    pub uvw: Vec<f64>,
    /// `_smooth_sig` id.
    pub smooth: i64,
}

/// `EDGE_FLAG_NAMES = ("soft", "curve", "layer", "hidden")`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct EdgeFlags {
    pub soft: bool,
    pub curve: Option<i64>,
    /// The layer name, interned by the caller.
    pub layer: Option<u32>,
    pub hidden: bool,
}

/// A face of the source mesh: positions, `interior`, and its attrs (None when
/// the dict is empty).
#[derive(Clone, Debug)]
pub struct SrcFace {
    pub outer: Vec<V3>,
    pub holes: Vec<Vec<V3>>,
    pub interior: bool,
    pub attrs: Option<usize>,
}

#[derive(Clone, Copy, Debug)]
pub struct SrcEdge {
    pub a: V3,
    pub b: V3,
    pub flags: EdgeFlags,
}

#[derive(Clone, Debug)]
pub struct OutEdge {
    pub v0: usize,
    pub v1: usize,
    /// Incident faces (radial list; a face repeats per segment).
    pub faces: Vec<usize>,
    pub flags: EdgeFlags,
}

#[derive(Clone, Debug)]
pub struct OutFace {
    pub outer: Vec<usize>,
    pub holes: Vec<Vec<usize>>,
    pub interior: bool,
    pub attrs: Option<usize>,
}

/// The mesh after the pass. `edges`, `faces` and `vertices` are in the
/// lists' order; ids index them.
#[derive(Clone, Debug, Default)]
pub struct Simplified {
    pub vertices: Vec<V3>,
    pub edges: Vec<OutEdge>,
    pub faces: Vec<OutFace>,
    /// `simplify_mesh`'s return value.
    pub removed: usize,
    /// True when the mesh had no faces: Python returns 0 and touches nothing.
    pub unchanged: bool,
}

#[derive(Debug, PartialEq, Eq)]
pub struct AngleError;

/// `Mesh.Vertex`/`Edge`/`Face` in arenas, with the operations the pass uses.
struct Soup {
    pos: Vec<V3>,
    vedges: Vec<Vec<usize>>,
    registry: HashMap<[i64; 3], usize>,
    evs: Vec<(usize, usize)>,
    efaces: Vec<Vec<usize>>,
    eflags: Vec<EdgeFlags>,
    /// `mesh.edges`, the list.
    edge_list: Vec<usize>,
    /// `mesh.vertices`, the list.
    vert_list: Vec<usize>,
    floop: Vec<Vec<usize>>,
    fholes: Vec<Vec<Vec<usize>>>,
    fint: Vec<bool>,
    fattrs: Vec<Option<usize>>,
}

impl Soup {
    fn new() -> Soup {
        Soup {
            pos: vec![],
            vedges: vec![],
            registry: HashMap::new(),
            evs: vec![],
            efaces: vec![],
            eflags: vec![],
            edge_list: vec![],
            vert_list: vec![],
            floop: vec![],
            fholes: vec![],
            fint: vec![],
            fattrs: vec![],
        }
    }

    fn vertex(&mut self, p: V3) -> usize {
        let k = ikey(p);
        if let Some(&v) = self.registry.get(&k) {
            return v;
        }
        let v = self.pos.len();
        self.pos.push(p);
        self.vedges.push(vec![]);
        self.registry.insert(k, v);
        self.vert_list.push(v);
        v
    }

    fn other(&self, e: usize, v: usize) -> usize {
        let (v0, v1) = self.evs[e];
        if v == v0 {
            v1
        } else {
            v0
        }
    }

    fn find_edge(&self, v0: usize, v1: usize) -> Option<usize> {
        let small = if self.vedges[v0].len() <= self.vedges[v1].len() { v0 } else { v1 };
        let target = if small == v0 { v1 } else { v0 };
        self.vedges[small].iter().copied().find(|&e| self.other(e, small) == target)
    }

    fn link_edge(&mut self, v0: usize, v1: usize) -> usize {
        if let Some(e) = self.find_edge(v0, v1) {
            return e;
        }
        let e = self.evs.len();
        self.evs.push((v0, v1));
        self.efaces.push(vec![]);
        self.eflags.push(EdgeFlags::default());
        // a set: adding the same edge twice (a self loop) is one entry
        self.vedges[v0].push(e);
        if v1 != v0 {
            self.vedges[v1].push(e);
        }
        self.edge_list.push(e);
        e
    }

    fn remove_edge(&mut self, e: usize) {
        let (v0, v1) = self.evs[e];
        self.vedges[v0].retain(|&x| x != e);
        self.vedges[v1].retain(|&x| x != e);
        if let Some(i) = self.edge_list.iter().position(|&x| x == e) {
            self.edge_list.remove(i);
        }
    }

    fn add_face(
        &mut self,
        outer: &[V3],
        holes: &[Vec<V3>],
        attrs: Option<usize>,
        interior: bool,
        maximal: Option<&dyn Fn(&[Vec<V3>]) -> Vec<usize>>,
    ) -> usize {
        let loop_: Vec<usize> = outer.iter().map(|&p| self.vertex(p)).collect();
        let kept: Vec<&Vec<V3>> = if holes.len() > 1 {
            match maximal {
                Some(f) => f(holes).into_iter().map(|i| &holes[i]).collect(),
                None => holes.iter().collect(),
            }
        } else {
            holes.iter().collect()
        };
        let hl: Vec<Vec<usize>> = kept.iter().map(|h| h.iter().map(|&p| self.vertex(p)).collect()).collect();
        let fid = self.floop.len();
        for lp in std::iter::once(&loop_).chain(hl.iter()) {
            let n = lp.len();
            for i in 0..n {
                let e = self.link_edge(lp[i], lp[(i + 1) % n]);
                self.efaces[e].push(fid);
            }
        }
        self.floop.push(loop_);
        self.fholes.push(hl);
        self.fint.push(interior);
        self.fattrs.push(attrs);
        fid
    }

    fn face_pts(&self, f: usize) -> Vec<V3> {
        self.floop[f].iter().map(|&v| self.pos[v]).collect()
    }

    /// `Mesh.collapsible_vertex`.
    fn collapsible(&self, v: usize, h: Hypot) -> bool {
        if self.vedges[v].len() != 2 {
            return false;
        }
        let (e1, e2) = (self.vedges[v][0], self.vedges[v][1]);
        let d1 = self.pos[v].sub(self.pos[self.other(e1, v)]);
        let d2 = self.pos[self.other(e2, v)].sub(self.pos[v]);
        if (d1.length(h) as f64) < 1e-4 || (d2.length(h) as f64) < 1e-4 {
            return false;
        }
        if (d1.normalized(h).dot(d2.normalized(h)) as f64) < 0.999 {
            return false;
        }
        let mut a = self.efaces[e1].clone();
        let mut b = self.efaces[e2].clone();
        a.sort_unstable();
        a.dedup();
        b.sort_unstable();
        b.dedup();
        a == b
    }

    /// `Mesh.collapse_vertex`.
    fn collapse(&mut self, v: usize) {
        let (e1, e2) = (self.vedges[v][0], self.vedges[v][1]);
        let (n1, n2) = (self.other(e1, v), self.other(e2, v));
        // list(set(e1.faces)): first-seen order here (id order in Python)
        let mut faces: Vec<usize> = vec![];
        for &f in &self.efaces[e1] {
            if !faces.contains(&f) {
                faces.push(f);
            }
        }
        self.remove_edge(e1);
        self.remove_edge(e2);
        let new_edge = self.link_edge(n1, n2);
        for &f in &faces {
            if let Some(i) = self.floop[f].iter().position(|&x| x == v) {
                self.floop[f].remove(i);
            }
            for hl in self.fholes[f].iter_mut() {
                if let Some(i) = hl.iter().position(|&x| x == v) {
                    hl.remove(i);
                }
            }
            if !self.efaces[new_edge].contains(&f) {
                self.efaces[new_edge].push(f);
            }
        }
        if let Some(i) = self.vert_list.iter().position(|&x| x == v) {
            self.vert_list.remove(i);
        }
        if self.registry.get(&ikey(self.pos[v])) == Some(&v) {
            self.registry.remove(&ikey(self.pos[v]));
        }
    }
}

/// `_collapse_all`.
fn collapse_all(m: &mut Soup, h: Hypot) {
    let order: Vec<usize> = m.vert_list.clone();
    let place: HashMap<usize, usize> = order.iter().enumerate().map(|(i, &v)| (v, i)).collect();
    let mut gone = vec![false; order.len()];
    let mut i = 0usize;
    while i < order.len() {
        let v = order[i];
        if gone[i] || !m.collapsible(v, h) {
            i += 1;
            continue;
        }
        let (e1, e2) = (m.vedges[v][0], m.vedges[v][1]);
        let (n1, n2) = (m.other(e1, v), m.other(e2, v));
        m.collapse(v);
        gone[i] = true;
        i = i.min(*place.get(&n1).unwrap_or(&i)).min(*place.get(&n2).unwrap_or(&i));
    }
}

/// `soften_smooth_edges(mesh, 0.85)`.
fn soften(m: &mut Soup, attrs: &[AttrInfo], h: Hypot) {
    let smooth = |a: Option<usize>| a.map_or(NONE_SMOOTH, |i| attrs[i].smooth);
    let mut normals: HashMap<usize, V3> = HashMap::new();
    for &e in &m.edge_list.clone() {
        if m.efaces[e].len() != 2 {
            continue;
        }
        let (f0, f1) = (m.efaces[e][0], m.efaces[e][1]);
        if f0 == f1 {
            continue;
        }
        if smooth(m.fattrs[f0]) != smooth(m.fattrs[f1]) {
            continue;
        }
        let n0 = *normals.entry(f0).or_insert_with(|| normal(&m.face_pts(f0), h));
        let n1 = *normals.entry(f1).or_insert_with(|| normal(&m.face_pts(f1), h));
        if (n0.dot(n1) as f64).abs() > 0.85 {
            m.eflags[e].soft = true;
        }
    }
}

/// `cos(max(radians(angle), 0.00045))`, or `AngleError` when Python raises
/// ValueError.
pub fn cos_tolerance(max_angle_degrees: f64) -> Result<f64, AngleError> {
    if !max_angle_degrees.is_finite() || !(0.0..=5.0).contains(&max_angle_degrees) {
        return Err(AngleError);
    }
    let tolerance = max_angle_degrees.to_radians().max(0.00045);
    Ok(tolerance.cos())
}

/// `simplify_mesh(mesh, max_angle_degrees, principal_planes)`.
///
/// `faces` and `edges` are the source mesh's lists in order; `attrs` the
/// table `SrcFace::attrs` indexes. `maximal_holes` is `_maximal_holes` for a
/// face with several holes (indexes of the holes kept; None keeps all).
pub fn simplify(
    faces: &[SrcFace],
    edges: &[SrcEdge],
    attrs: &[AttrInfo],
    max_angle_degrees: f64,
    principal_planes: bool,
    maximal_holes: Option<&dyn Fn(&[Vec<V3>]) -> Vec<usize>>,
    h: Hypot,
) -> Result<Simplified, AngleError> {
    let cos_tol = cos_tolerance(max_angle_degrees)?;
    if faces.is_empty() {
        return Ok(Simplified { unchanged: true, ..Default::default() });
    }
    let original_count = faces.len();
    // Faces with holes or interior stay; the rest are the loops fused.
    let mut loop_face: Vec<usize> = vec![]; // loops index -> faces index
    let mut holed: Vec<usize> = vec![];
    for (i, f) in faces.iter().enumerate() {
        if !f.holes.is_empty() || f.interior {
            holed.push(i);
        } else {
            loop_face.push(i);
        }
    }
    // the flat arrays `fuse` reads
    let mut pts: Vec<V3> = vec![];
    let mut starts = vec![0usize];
    let mut sig = vec![];
    let mut group = vec![];
    let mut rank = vec![];
    let mut uvw: Vec<Vec<f64>> = vec![];
    for &fi in &loop_face {
        pts.extend_from_slice(&faces[fi].outer);
        starts.push(pts.len());
        match faces[fi].attrs {
            Some(a) => {
                let ai = &attrs[a];
                sig.push(ai.sig);
                group.push(ai.tex_group);
                rank.push(ai.rank);
                uvw.push(ai.uvw.clone());
            }
            None => {
                sig.push(NONE_SIG);
                group.push(-1);
                rank.push(1);
                uvw.push(vec![]);
            }
        }
    }
    let inp = fuse::Input { pts: &pts, loop_start: &starts, sig: &sig, tex_group: &group, rank: &rank, uvw: &uvw };
    let regions = if loop_face.is_empty() { vec![] } else { fuse::fuse(&inp, cos_tol, principal_planes, h) };

    let mut s = Soup::new();
    let flat = |r: std::ops::Range<usize>| -> Vec<V3> { pts[r].to_vec() };
    for region in &regions {
        match region {
            Region::Pass(lps) => {
                for &lp in lps {
                    let outer = flat(starts[lp]..starts[lp + 1]);
                    s.add_face(&outer, &[], faces[loop_face[lp]].attrs, false, maximal_holes);
                }
            }
            Region::Traced(rings, lps) => {
                let mut rs: Vec<Vec<V3>> = rings.iter().map(|r| r.iter().map(|&i| pts[i]).collect()).collect();
                let outer = rs.remove(0);
                s.add_face(&outer, &rs, faces[loop_face[lps[0]]].attrs, false, maximal_holes);
            }
        }
    }
    for &fi in &holed {
        let f = &faces[fi];
        s.add_face(&f.outer, &f.holes, f.attrs, f.interior, maximal_holes);
    }

    // original_flags: the last edge with a given key pair wins
    let pair = |a: V3, b: V3| {
        let (ka, kb) = (ikey(a), ikey(b));
        if ka <= kb {
            (ka, kb)
        } else {
            (kb, ka)
        }
    };
    let mut original: HashMap<([i64; 3], [i64; 3]), EdgeFlags> = HashMap::new();
    for e in edges {
        original.insert(pair(e.a, e.b), e.flags);
    }
    for &e in &s.edge_list.clone() {
        let (v0, v1) = s.evs[e];
        if let Some(&fl) = original.get(&pair(s.pos[v0], s.pos[v1])) {
            s.eflags[e] = fl;
        }
    }

    collapse_all(&mut s, h);
    soften(&mut s, attrs, h);

    // the result, with dense ids in list order
    let vmap: HashMap<usize, usize> = s.vert_list.iter().enumerate().map(|(i, &v)| (v, i)).collect();
    let vertices: Vec<V3> = s.vert_list.iter().map(|&v| s.pos[v]).collect();
    let out_edges: Vec<OutEdge> = s
        .edge_list
        .iter()
        .map(|&e| OutEdge {
            v0: vmap[&s.evs[e].0],
            v1: vmap[&s.evs[e].1],
            faces: s.efaces[e].clone(),
            flags: s.eflags[e],
        })
        .collect();
    let out_faces: Vec<OutFace> = (0..s.floop.len())
        .map(|f| OutFace {
            outer: s.floop[f].iter().map(|v| vmap[v]).collect(),
            holes: s.fholes[f].iter().map(|hl| hl.iter().map(|v| vmap[v]).collect()).collect(),
            interior: s.fint[f],
            attrs: s.fattrs[f],
        })
        .collect();
    let removed = original_count.saturating_sub(out_faces.len());
    Ok(Simplified { vertices, edges: out_edges, faces: out_faces, removed, unchanged: false })
}
