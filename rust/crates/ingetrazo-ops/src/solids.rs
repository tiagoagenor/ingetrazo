// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! Solid booleans on groups and components (`core/solids.py`): Outer Shell,
//! Union, Subtract, Trim, Intersect and Split, with the same rules.
//!
//! A solid is a group or component instance with a closed volume: every
//! edge borders exactly two faces, no stray edges, no groups nested inside.
//! The arithmetic is Manifold (`manifold-sys`, the version of the Python
//! wheel). Each triangle carries the id of the source face it came from,
//! which is how materials survive; the result's triangles are merged back
//! into polygons with holes by [`ingetrazo_geom::fuse`] (coplanar, same
//! source face), collinear corners are dropped, and the seams of curved
//! surfaces stay soft.
//!
//! [`run`] returns data: the picks that leave and the new groups' meshes,
//! built through the `Mesh` API. [`apply`] and [`undo`] are the
//! `SolidOperationCommand` over the scene's top-level groups.

use std::borrow::Cow;
use std::collections::{HashMap, HashSet};
use std::fmt;

use ingetrazo_formats::pyjson::Value;
use ingetrazo_geom::fuse::{self, Region};
use ingetrazo_geom::meshface;
use ingetrazo_geom::triangulate::triangulate;
use ingetrazo_math::{Hypot, V3};
#[cfg(feature = "manifold")]
use ingetrazo_model::materials::effective_attrs;
use ingetrazo_model::mesh::{FaceId, Mesh};
use ingetrazo_model::pyval;
use ingetrazo_model::scene::{Group, MeshId, Scene};
use ingetrazo_model::topology::add_face_py;

use crate::world::world_mesh;

/// Operations, in the classic order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Op {
    OuterShell,
    Union,
    Subtract,
    Trim,
    Intersect,
    Split,
}

impl Op {
    pub const ALL: [Op; 6] = [Op::OuterShell, Op::Union, Op::Subtract, Op::Trim, Op::Intersect, Op::Split];

    /// The Python constant (`"outer_shell"`, `"union"`, ...).
    pub fn id(self) -> &'static str {
        match self {
            Op::OuterShell => "outer_shell",
            Op::Union => "union",
            Op::Subtract => "subtract",
            Op::Trim => "trim",
            Op::Intersect => "intersect",
            Op::Split => "split",
        }
    }

    pub fn from_id(id: &str) -> Option<Op> {
        Op::ALL.into_iter().find(|o| o.id() == id)
    }

    /// `RESULT_NAMES[op]`: the untranslated name given to the result groups.
    pub fn result_name(self) -> &'static str {
        match self {
            Op::OuterShell => "Outer shell",
            Op::Union => "Union",
            Op::Subtract => "Difference",
            Op::Trim => "Trimmed",
            Op::Intersect => "Intersection",
            Op::Split => "Split",
        }
    }
}

/// `SolidError`: an operation that cannot run. `Display` is the message
/// Python raises (its text is a key for the user's language).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SolidError {
    NotSolid,
    TwoSolids,
    Empty,
    NoOverlap,
    NotManifold,
    /// Built without the `manifold` feature.
    NoKernel,
}

impl fmt::Display for SolidError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            SolidError::NotSolid => "not a solid",
            SolidError::TwoSolids => "two solids",
            SolidError::Empty => "empty",
            SolidError::NoOverlap => "no overlap",
            SolidError::NotManifold => "not manifold",
            SolidError::NoKernel => "built without the manifold feature",
        })
    }
}

impl std::error::Error for SolidError {}

// ---- What is a solid -----------------------------------------------------------

/// `Face.triangulate()` of a face of `mesh`: the loop and holes through the
/// ear-clipper with the face's own normal.
fn face_triangles(mesh: &Mesh, f: FaceId, h: Hypot) -> Vec<[V3; 3]> {
    let face = mesh.face(f);
    if face.outer.len() < 3 {
        return Vec::new();
    }
    let outer: Vec<V3> = face.outer.iter().map(|&v| mesh.position(v)).collect();
    let holes: Vec<Vec<V3>> = face.holes.iter().map(|hl| hl.iter().map(|&v| mesh.position(v)).collect()).collect();
    triangulate(&outer, &holes, Some(meshface::normal(&outer, h)), h)
}

/// `core.bim.face_set_volume`: the volume the faces enclose, or `None` when
/// the set is not watertight by itself (every edge of the set borders
/// exactly two faces of the set). A signed tetrahedron sum over the set's
/// triangles.
fn face_set_volume(mesh: &Mesh, faces: &[FaceId], h: Hypot) -> Option<f64> {
    if faces.is_empty() {
        return None;
    }
    let mut counts: HashMap<(u32, u32), u32> = HashMap::new();
    for &f in faces {
        let face = mesh.face(f);
        for lp in std::iter::once(&face.outer).chain(face.holes.iter()) {
            let n = lp.len();
            for i in 0..n {
                let (a, b) = (lp[i].0, lp[(i + 1) % n].0);
                *counts.entry((a.min(b), a.max(b))).or_insert(0) += 1;
            }
        }
    }
    if counts.values().any(|&c| c != 2) {
        return None;
    }
    let mut total = 0.0f64;
    for &f in faces {
        for [t0, t1, t2] in face_triangles(mesh, f, h) {
            total += t0.dot(t1.cross(t2)) as f64 / 6.0;
        }
    }
    Some(total.abs())
}

/// `solid_report(mesh)`: `(is_solid, volume)`. Every edge between exactly
/// two faces, no stray edges, at least one face; the volume in m³ when solid.
pub fn solid_report(mesh: &Mesh, h: Hypot) -> (bool, Option<f64>) {
    if mesh.faces.is_empty() {
        return (false, None);
    }
    for &e in &mesh.edges {
        let n = mesh.edge(e).faces.iter().filter(|&&f| !mesh.face(f).interior).count();
        if n != 2 {
            return (false, None);
        }
    }
    let faces: Vec<FaceId> = mesh.faces.iter().copied().filter(|&f| !mesh.face(f).interior).collect();
    match face_set_volume(mesh, &faces, h) {
        Some(vol) if vol > 1e-12 => (true, Some(vol)),
        _ => (false, None),
    }
}

/// `solid_volume(group)`: the group's volume when it is a solid, else
/// `None`. Nested groups disqualify it.
pub fn solid_volume(scene: &Scene, group: &Group, h: Hypot) -> Option<f64> {
    if !group.children.is_empty() {
        return None;
    }
    let mesh = world_mesh(scene, group, h);
    let (ok, vol) = solid_report(&mesh, h);
    if ok {
        vol
    } else {
        None
    }
}

/// `is_solid(group)`.
pub fn is_solid(scene: &Scene, group: &Group, h: Hypot) -> bool {
    solid_volume(scene, group, h).is_some()
}

// ---- The result ------------------------------------------------------------------

/// A group the operation made: what `Group(mesh, name=...)` plus `layer`
/// carry. The mesh is the group's own (built through the `Mesh` API).
#[derive(Clone, Debug)]
pub struct SolidGroup {
    pub name: String,
    pub layer: Option<String>,
    pub mesh: Mesh,
}

/// What `run` returns: the picks to take out of the model (indexes into the
/// picks, in the order Python lists them) and the new groups, in the order
/// they should appear.
#[derive(Clone, Debug)]
pub struct SolidResult {
    pub removed: Vec<usize>,
    pub created: Vec<SolidGroup>,
}

// ---- Group <-> manifold -----------------------------------------------------------

/// Every source face, globally numbered across the inputs: what a result
/// triangle's face id points back to.
#[derive(Default)]
struct Sources {
    attrs: Vec<Value>,
    /// Pairs of source face ids joined by a soft edge (curved surfaces).
    soft_pairs: HashSet<(u32, u32)>,
}

/// One result triangle: its corners in float32 and the source face id.
struct Loop {
    pts: [V3; 3],
    src: i64,
}

/// `effective_material(group)`: the group's own paint (nested pieces cannot
/// occur in a solid).
#[cfg(feature = "manifold")]
fn effective_material(g: &Group) -> Option<&Value> {
    g.material.as_ref().filter(|m| pyval::truthy(m))
}

/// The bits of a float32 coordinate as Python's tuple-of-floats key sees it:
/// `-0.0` and `0.0` are one value.
#[cfg(feature = "manifold")]
fn coord_bits(x: f32) -> u32 {
    (x + 0.0).to_bits()
}

#[cfg(feature = "manifold")]
mod kernel {
    use super::*;
    use manifold_sys::{Manifold, MeshGl64, OpType, Status};

    /// `numpy`'s `ndarray.sum()` of a float64 vector: the first element plus
    /// the pairwise sum of the rest.
    fn np_sum(a: &[f64]) -> f64 {
        match a.split_first() {
            None => 0.0,
            Some((first, rest)) => first + pairwise(rest),
        }
    }

    fn pairwise(a: &[f64]) -> f64 {
        let n = a.len();
        if n < 8 {
            let mut res = 0.0;
            for x in a {
                res += x;
            }
            res
        } else if n <= 128 {
            let mut r = [a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]];
            let mut i = 8;
            while i + 8 <= n - (n % 8) {
                for j in 0..8 {
                    r[j] += a[i + j];
                }
                i += 8;
            }
            let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
            for x in &a[n - (n % 8)..] {
                res += x;
            }
            res
        } else {
            let mut n2 = n / 2;
            n2 -= n2 % 8;
            pairwise(&a[..n2]) + pairwise(&a[n2..])
        }
    }

    fn cross(b: [f64; 3], c: [f64; 3]) -> [f64; 3] {
        [b[1] * c[2] - b[2] * c[1], b[2] * c[0] - b[0] * c[2], b[0] * c[1] - b[1] * c[0]]
    }

    fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
        a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
    }

    fn build(v: &[[f64; 3]], t: &[[u64; 3]], f: &[u64]) -> Manifold {
        let flat_v: Vec<f64> = v.iter().flatten().copied().collect();
        let flat_t: Vec<u64> = t.iter().flatten().copied().collect();
        Manifold::from_mesh(&MeshGl64::new(&flat_v, 3, &flat_t, Some(f)))
    }

    /// `_to_manifold(group, sources)`.
    pub(super) fn to_manifold(scene: &Scene, group: &Group, sources: &mut Sources, h: Hypot) -> Result<Manifold, SolidError> {
        let mesh = world_mesh(scene, group, h);
        let paint = effective_material(group);
        let mut index: HashMap<[u32; 3], u64> = HashMap::new();
        let mut verts: Vec<[f64; 3]> = Vec::new();
        let mut tris: Vec<[u64; 3]> = Vec::new();
        let mut fids: Vec<u64> = Vec::new();
        let mut face_id: HashMap<FaceId, u32> = HashMap::new();
        let mut vid = |p: V3| -> u64 {
            let k = [coord_bits(p.0), coord_bits(p.1), coord_bits(p.2)];
            *index.entry(k).or_insert_with(|| {
                verts.push([p.0 as f64, p.1 as f64, p.2 as f64]);
                verts.len() as u64 - 1
            })
        };
        for &f in &mesh.faces {
            let face = mesh.face(f);
            if face.interior {
                continue;
            }
            let fid = sources.attrs.len() as u32;
            face_id.insert(f, fid);
            let own = Value::Object(face.attrs.clone());
            let attrs = match paint {
                Some(p) => effective_attrs(&own, p).unwrap_or(own),
                None => own,
            };
            sources.attrs.push(attrs);
            for [a, b, c] in face_triangles(&mesh, f, h) {
                tris.push([vid(a), vid(b), vid(c)]);
                fids.push(fid as u64);
            }
        }
        for &e in &mesh.edges {
            let ed = mesh.edge(e);
            if ed.soft {
                let fs: Vec<u32> = ed.faces.iter().filter_map(|f| face_id.get(f).copied()).collect();
                if fs.len() == 2 {
                    sources.soft_pairs.insert((fs[0].min(fs[1]), fs[0].max(fs[1])));
                }
            }
        }
        if tris.is_empty() {
            return Err(SolidError::Empty);
        }
        // Outward orientation: a negative signed volume means the faces point in.
        let vols: Vec<f64> = tris.iter().map(|t| dot(verts[t[0] as usize], cross(verts[t[1] as usize], verts[t[2] as usize]))).collect();
        if np_sum(&vols) < 0.0 {
            for t in &mut tris {
                t.swap(0, 2);
            }
        }
        let man = build(&verts, &tris, &fids);
        if man.status() == Status::NoError {
            return Ok(man);
        }
        // A face's triangulation can leave a corner of a neighbouring face
        // lying ON one of its triangle edges, which is not a closed triangle
        // surface. Only then, split those triangles at the corners and try
        // once more: what converts today converts exactly as before.
        let scale = verts.iter().flatten().fold(1.0f64, |m, x| m.max(x.abs()));
        let tol = 1e-9 * scale;
        let t_i: Vec<[i64; 3]> = tris.iter().map(|t| [t[0] as i64, t[1] as i64, t[2] as i64]).collect();
        let f_i: Vec<i64> = fids.iter().map(|&x| x as i64).collect();
        let out = ingetrazo_geom::conform::conform(&verts, &t_i, &f_i, tol);
        let t2: Vec<[u64; 3]> = out.tris.iter().map(|t| [t[0] as u64, t[1] as u64, t[2] as u64]).collect();
        let f2: Vec<u64> = out.fids.iter().map(|&x| x as u64).collect();
        let man = build(&out.verts, &t2, &f2);
        if man.status() != Status::NoError {
            return Err(SolidError::NotManifold);
        }
        Ok(man)
    }

    /// `_outer_shell_mask(v, t)`: keep the triangles of the shells that bound
    /// volume from outside; an inner void is a connected shell of negative
    /// signed volume.
    fn outer_shell_mask(v: &[[f64; 3]], t: &[[usize; 3]]) -> Vec<bool> {
        let mut parent: Vec<usize> = (0..v.len()).collect();
        fn find(parent: &mut [usize], mut x: usize) -> usize {
            while parent[x] != x {
                parent[x] = parent[parent[x]];
                x = parent[x];
            }
            x
        }
        for &[a, b, c] in t {
            let (ra, rb, rc) = (find(&mut parent, a), find(&mut parent, b), find(&mut parent, c));
            parent[rb] = ra;
            let r = find(&mut parent, rc);
            parent[r] = ra;
        }
        let roots: Vec<usize> = t.iter().map(|tri| find(&mut parent, tri[0])).collect();
        let vol: Vec<f64> = t.iter().map(|tri| dot(v[tri[0]], cross(v[tri[1]], v[tri[2]]))).collect();
        let mut by_root: HashMap<usize, Vec<f64>> = HashMap::new();
        for (r, x) in roots.iter().zip(&vol) {
            by_root.entry(*r).or_default().push(*x);
        }
        let negative: HashSet<usize> = by_root.into_iter().filter(|(_, xs)| np_sum(xs) < 0.0).map(|(r, _)| r).collect();
        roots.iter().map(|r| !negative.contains(r)).collect()
    }

    /// `_loops_from(man, sources, drop_voids)`.
    pub(super) fn loops_from(man: &Manifold, drop_voids: bool) -> Vec<Loop> {
        let mesh = man.to_mesh64();
        let np = mesh.num_prop();
        let props = mesh.vert_properties();
        let v: Vec<[f64; 3]> = props.chunks(np).map(|c| [c[0], c[1], c[2]]).collect();
        let t: Vec<[usize; 3]> = mesh.tri_verts().chunks(3).map(|c| [c[0] as usize, c[1] as usize, c[2] as usize]).collect();
        let fid = mesh.face_id();
        if t.is_empty() {
            return Vec::new();
        }
        let keep = if drop_voids { outer_shell_mask(&v, &t) } else { vec![true; t.len()] };
        let pt = |i: usize| V3(v[i][0] as f32, v[i][1] as f32, v[i][2] as f32);
        t.iter()
            .enumerate()
            .filter(|(k, _)| keep[*k])
            .map(|(k, tri)| Loop { pts: [pt(tri[0]), pt(tri[1]), pt(tri[2])], src: fid.get(k).copied().unwrap_or(0) as i64 })
            .collect()
    }

    pub(super) fn run(scene: &Scene, op: Op, groups: &[&Group], name: &dyn Fn(&str) -> String, h: Hypot) -> Result<SolidResult, SolidError> {
        if groups.len() < 2 {
            return Err(SolidError::TwoSolids);
        }
        for g in groups {
            if !is_solid(scene, g, h) {
                return Err(SolidError::NotSolid);
            }
        }
        let mut sources = Sources::default();
        let mut mans = Vec::with_capacity(groups.len());
        for g in groups {
            mans.push(to_manifold(scene, g, &mut sources, h)?);
        }
        let (first, second) = (groups[0], groups[1]);
        let label = name(op.result_name());
        let out = |man: &Manifold, layer: Option<String>, drop_voids: bool| -> Option<SolidGroup> {
            let loops = loops_from(man, drop_voids);
            if loops.is_empty() {
                return None;
            }
            Some(group_from_loops(&loops, &sources, label.clone(), layer, h))
        };
        let refs: Vec<&Manifold> = mans.iter().collect();
        let target_layer = if matches!(op, Op::Subtract | Op::Trim) { second.layer.clone() } else { first.layer.clone() };
        let all: Vec<usize> = (0..groups.len()).collect();
        match op {
            Op::Union | Op::OuterShell => {
                let man = Manifold::batch_boolean(&refs, OpType::Add);
                let res = out(&man, target_layer, op == Op::OuterShell).ok_or(SolidError::Empty)?;
                return Ok(SolidResult { removed: all, created: vec![res] });
            }
            Op::Intersect => {
                let man = Manifold::batch_boolean(&refs, OpType::Intersect);
                let res = out(&man, target_layer, false).ok_or(SolidError::NoOverlap)?;
                return Ok(SolidResult { removed: all, created: vec![res] });
            }
            _ => {}
        }
        let (a, b) = (&mans[0], &mans[1]);
        if a.intersection(b).is_empty() {
            return Err(SolidError::NoOverlap);
        }
        if matches!(op, Op::Subtract | Op::Trim) {
            let res = out(&b.difference(a), target_layer, false).ok_or(SolidError::Empty)?;
            let removed = if op == Op::Subtract { vec![0, 1] } else { vec![1] };
            return Ok(SolidResult { removed, created: vec![res] });
        }
        // SPLIT: B-A, A-B, A∩B: three groups.
        let made = [
            out(&b.difference(a), second.layer.clone(), false),
            out(&a.difference(b), first.layer.clone(), false),
            out(&a.intersection(b), first.layer.clone(), false),
        ];
        Ok(SolidResult { removed: vec![0, 1], created: made.into_iter().flatten().collect() })
    }
}

// ---- Rebuilding the result --------------------------------------------------------

/// A polygon on its way to becoming a face: the fused outline, its holes
/// and the source face its attrs come from. (Python also keeps the loops it
/// was fused from, to fall back on if `add_face` raised; it never does.)
struct Poly {
    outer: Vec<V3>,
    holes: Vec<Vec<V3>>,
    src: i64,
}

/// `_group_from_loops`: a new group with clean polygon faces: coplanar
/// triangles of one source face merged (`fuse`), collinear corners dropped,
/// and the seams between faces that came from a soft pair kept soft.
#[cfg_attr(not(feature = "manifold"), allow(dead_code))]
fn group_from_loops(loops: &[Loop], sources: &Sources, name: String, layer: Option<String>, h: Hypot) -> SolidGroup {
    // Triangles of one source face carry one signature (the source id is
    // part of it, so faces of different sources never merge, whatever
    // their colours); textured ones outrank bare ones among coincident
    // copies.
    let mut pts: Vec<V3> = Vec::with_capacity(loops.len() * 3);
    let mut starts = vec![0usize];
    let mut sig = Vec::with_capacity(loops.len());
    let mut rank = Vec::with_capacity(loops.len());
    for l in loops {
        pts.extend_from_slice(&l.pts);
        starts.push(pts.len());
        sig.push(l.src);
        let textured = usize::try_from(l.src)
            .ok()
            .and_then(|i| sources.attrs.get(i))
            .and_then(|a| a.get("texture"))
            .is_some_and(pyval::truthy);
        rank.push(if textured { 2u8 } else { 1u8 });
    }
    let group = vec![-1i64; loops.len()];
    let uvw: Vec<Vec<f64>> = vec![Vec::new(); loops.len()];
    let inp = fuse::Input { pts: &pts, loop_start: &starts, sig: &sig, tex_group: &group, rank: &rank, uvw: &uvw };
    let regions = fuse::fuse(&inp, 0.99999, false, h);
    let ring_pts = |lp: usize| pts[starts[lp]..starts[lp + 1]].to_vec();
    let mut polys: Vec<Poly> = Vec::new();
    for region in &regions {
        match region {
            Region::Pass(lps) => {
                for &lp in lps {
                    let r = ring_pts(lp);
                    polys.push(Poly { outer: r, holes: Vec::new(), src: loops[lp].src });
                }
            }
            Region::Traced(rings, lps) => {
                let mut rs: Vec<Vec<V3>> = rings.iter().map(|r| r.iter().map(|&i| pts[i]).collect()).collect();
                let outer = rs.remove(0);
                polys.push(Poly { outer, holes: rs, src: loops[lps[0]].src });
            }
        }
    }
    drop_collinear(&mut polys, 1e-9, h);
    let mut mesh = Mesh::new();
    let mut face_src: HashMap<FaceId, i64> = HashMap::new();
    for p in &polys {
        let clean: Vec<(String, Value)> = match usize::try_from(p.src).ok().and_then(|i| sources.attrs.get(i)) {
            Some(Value::Object(items)) => items.clone(),
            _ => Vec::new(),
        };
        let f = add_face_py(&mut mesh, &p.outer, &p.holes, h);
        if !clean.is_empty() {
            mesh.face_mut(f).attrs = clean;
        }
        face_src.insert(f, p.src);
    }
    for e in mesh.edges.clone() {
        let faces = &mesh.edge(e).faces;
        let soft = faces.len() == 2
            && match (u32::try_from(face_src[&faces[0]]), u32::try_from(face_src[&faces[1]])) {
                (Ok(a), Ok(b)) => sources.soft_pairs.contains(&(a.min(b), a.max(b))),
                _ => false,
            };
        if soft {
            mesh.edge_mut(e).soft = true;
        }
    }
    SolidGroup { name, layer, mesh }
}

/// `_drop_collinear`: a corner every ring passes straight through is not a
/// corner: the triangulation left them along edges, and the classic result
/// has none.
fn drop_collinear(polys: &mut [Poly], tol: f64, h: Hypot) {
    fn key(p: V3) -> [i64; 3] {
        let r = |x: f32| (x as f64 * 1e7).round_ties_even() as i64;
        [r(p.0), r(p.1), r(p.2)]
    }
    let mut straight: HashMap<[i64; 3], bool> = HashMap::new();
    for p in polys.iter() {
        for ring in std::iter::once(&p.outer).chain(p.holes.iter()) {
            let n = ring.len();
            for i in 0..n {
                let (p0, p1, p2) = (ring[(i + n - 1) % n], ring[i], ring[(i + 1) % n]);
                let (d1, d2) = (p1.sub(p0), p2.sub(p1));
                let cross_len = d1.cross(d2).length(h) as f64;
                let bound = tol * ((d1.length(h) as f64) * (d2.length(h) as f64)).max(1e-12) * 1e3;
                let ok = cross_len <= bound && d1.dot(d2) > 0.0;
                let e = straight.entry(key(p1)).or_insert(true);
                *e = *e && ok;
            }
        }
    }
    let drop: HashSet<[i64; 3]> = straight.into_iter().filter(|(_, v)| *v).map(|(k, _)| k).collect();
    if drop.is_empty() {
        return;
    }
    for p in polys.iter_mut() {
        for ring in std::iter::once(&mut p.outer).chain(p.holes.iter_mut()) {
            let dropped = ring.iter().filter(|&&q| drop.contains(&key(q))).count();
            if ring.len() - dropped >= 3 {
                ring.retain(|&q| !drop.contains(&key(q)));
            }
        }
    }
}

// ---- The operations ------------------------------------------------------------------

/// `run(op, groups)`: run `op` on `groups` (in pick order: for Subtract and
/// Trim the first is the cutter, the second the target). `name` translates
/// the result groups' name (`tr`).
#[cfg(feature = "manifold")]
pub fn run(scene: &Scene, op: Op, groups: &[&Group], name: &dyn Fn(&str) -> String, h: Hypot) -> Result<SolidResult, SolidError> {
    kernel::run(scene, op, groups, name, h)
}

#[cfg(not(feature = "manifold"))]
pub fn run(_scene: &Scene, _op: Op, _groups: &[&Group], _name: &dyn Fn(&str) -> String, _h: Hypot) -> Result<SolidResult, SolidError> {
    Err(SolidError::NoKernel)
}

/// A solid operation applied to a scene: what `undo` needs.
#[derive(Clone, Debug)]
pub struct Applied {
    /// `(index in scene.groups, group)` of every group that left, ascending.
    pub removed: Vec<(usize, Group)>,
    /// Where the results went in `scene.groups` and how many.
    pub at: usize,
    pub count: usize,
}

/// `SolidOperationCommand.do` on the scene's top-level groups: the inputs
/// leave, the results take the place of the first input that left. `picks`
/// are indexes into `scene.groups`; the results' meshes join the scene's
/// arena. Returns what [`undo`] needs.
pub fn apply(scene: &mut Scene, op: Op, picks: &[usize], name: &dyn Fn(&str) -> String, h: Hypot) -> Result<Applied, SolidError> {
    let result = {
        let groups: Vec<&Group> = picks.iter().map(|&i| &scene.groups[i]).collect();
        run(scene, op, &groups, name, h)?
    };
    let mut gone: Vec<usize> = result.removed.iter().map(|&k| picks[k]).collect();
    gone.sort_unstable();
    gone.dedup();
    let at = gone.first().copied().unwrap_or(scene.groups.len());
    let mut removed = Vec::with_capacity(gone.len());
    for &i in gone.iter().rev() {
        removed.push((i, scene.groups.remove(i)));
    }
    removed.reverse();
    let count = result.created.len();
    let mut made = Vec::with_capacity(count);
    for c in result.created {
        scene.meshes.push(c.mesh);
        let mut g = Group::new(MeshId(scene.meshes.len() as u32 - 1), c.name);
        g.layer = c.layer;
        made.push(g);
    }
    let tail = scene.groups.split_off(at.min(scene.groups.len()));
    scene.groups.extend(made);
    scene.groups.extend(tail);
    Ok(Applied { removed, at, count })
}

/// `SolidOperationCommand.undo`: the results go, the inputs come back where
/// they were.
pub fn undo(scene: &mut Scene, applied: Applied) {
    scene.groups.drain(applied.at..applied.at + applied.count);
    for (i, g) in applied.removed {
        let i = i.min(scene.groups.len());
        scene.groups.insert(i, g);
    }
}

/// `SolidOperationCommand` on the scene's top-level groups: one solid
/// operation as one undo step. The boolean runs on every `apply` (the same
/// inputs give the same result, so a redo replays the same placement).
pub struct SolidOperationCommand {
    op: Op,
    picks: Vec<usize>,
    applied: Option<Applied>,
}

impl SolidOperationCommand {
    pub fn new(op: Op, picks: Vec<usize>) -> Self {
        SolidOperationCommand { op, picks, applied: None }
    }
}

impl crate::command::Command for SolidOperationCommand {
    fn name(&self) -> &'static str {
        "solid_operation"
    }
    fn apply(&mut self, scene: &mut Scene, cx: &mut crate::command::Cx) -> Result<(), crate::command::Fail> {
        let applied = apply(scene, self.op, &self.picks, &|n| n.to_string(), cx.hypot).map_err(|e| crate::command::Fail::Failed(e.to_string()))?;
        self.applied = Some(applied);
        Ok(())
    }
    fn undo(&mut self, scene: &mut Scene, _cx: &mut crate::command::Cx) {
        if let Some(a) = self.applied.take() {
            undo(scene, a);
        }
    }
}

/// The world mesh of a group (for callers that show a solid's volume).
pub fn group_world_mesh<'a>(scene: &'a Scene, group: &Group, h: Hypot) -> Cow<'a, Mesh> {
    world_mesh(scene, group, h)
}
