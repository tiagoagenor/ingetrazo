// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! Edit > Intersect Faces, the part that decides WHICH edges to add:
//! `core.intersect.segments_for` (what takes part, in which mode, in which
//! order) and the length filter of `MainWindow._on_intersect_faces`.
//!
//! The kernels (`_Poly`, `cross`, `_intervals`, `_overlap`, `_dedupe`, the
//! box test) live in `ingetrazo-geom`; this module is the glue Python does
//! in `polys_of` / `_pairs` / `intersect` / `segments_for`:
//!
//! * the faces of the selection (loose faces and groups) become world
//!   polygons: a loose face as it stands, a group's faces through the world
//!   matrix of every placement `iter_placements` yields (depth first);
//! * a polygon `world_polys` rejects (fewer than 3 corners, a zero normal)
//!   is dropped before the pairing, so pair indexes count the kept ones;
//! * `_pairs` yields the box-touching pairs row-major (by `i`, then `j`);
//!   against another list the faces that cannot reach the other side's whole
//!   box are cut away first, twice, exactly as Python does;
//! * the crossings of every pair are concatenated in that order and deduped
//!   once at the end (the first of each rounded end pair stays).
//!
//! What happens to the segments afterwards is not Intersect's: the window
//! hands them to `core.edits.build_add_edges(scene, segs, detect_faces=True)`
//! and runs that one command (M3.1, `edits.rs`), so the history step, the
//! edges it adds, the faces it splits and its undo are the edits planner's.
//!
//! Python's `scene.selection` is a `set` of objects, so the order the
//! selected things come in is the interpreter's. Here the selection is an
//! ordered list; the fixtures pin Python to creation order (loose faces in
//! mesh order, then groups in scene order) and the callers pass it so.

use ingetrazo_geom::boxes::{touching, Box3};
use ingetrazo_geom::intersect::{cross, dedupe, poly, Poly};
use ingetrazo_math::{length, Hypot, M4, V3};
use ingetrazo_model::mesh::Mesh;
use ingetrazo_model::scene::{MeshId, Scene};
use ingetrazo_model::FaceId;

use crate::group::{group_at, iter_placements};

/// `WITH_MODEL`, `WITH_SELECTION`, `WITH_CONTEXT`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    /// The selection against everything it touches: the whole context and
    /// every group in it.
    Model,
    /// Only the selected things against each other.
    Selection,
    /// The selection against the rest of the context being edited (inside a
    /// group: its own geometry), without the groups.
    Context,
}

/// A selected thing that takes part: a face of the active mesh, or a group
/// (its index among the groups of the context being edited).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Entity {
    Face(FaceId),
    Group(usize),
}

/// The mesh welds points closer than this (`_WELD_LENGTH` of the window): a
/// shorter crossing is a point, and its two ends would weld into one vertex.
pub const WELD_LENGTH: f64 = 1e-4;

/// A crossing segment.
pub type Seg = (V3, V3);

/// Below this two boxes are apart (`_touching`'s `eps`).
const BOX_EPS: f64 = 1e-6;

/// The groups of the context being edited: `scene.groups`, or the children of
/// the group whose path is `edit` (`owner.children`).
fn context_groups<'a>(scene: &'a Scene, edit: Option<&[usize]>) -> &'a [ingetrazo_model::scene::Group] {
    match edit {
        Some(path) => group_at(&scene.groups, path).map(|g| g.children.as_slice()).unwrap_or(&[]),
        None => &scene.groups,
    }
}

/// `world_polys` for one face: its rings, each corner through `xform`.
fn face_poly(mesh: &Mesh, f: FaceId, xform: Option<&M4>, h: Hypot) -> Option<Poly> {
    let face = mesh.face(f);
    let w = |v| {
        let p = mesh.position(v);
        match xform {
            Some(m) => m.map(p),
            None => p,
        }
    };
    let rings: Vec<Vec<V3>> = std::iter::once(&face.outer).chain(face.holes.iter()).map(|r| r.iter().map(|&v| w(v)).collect()).collect();
    let refs: Vec<&[V3]> = rings.iter().map(|r| r.as_slice()).collect();
    poly(&refs, h)
}

/// `polys_of` for a mix of loose faces (of `mesh`) and groups.
fn polys_of(scene: &Scene, mesh: MeshId, faces: &[FaceId], groups: &[&ingetrazo_model::scene::Group], h: Hypot) -> Vec<Poly> {
    let mut out = Vec::new();
    // Order matters: Python walks the entities in the order given, so the
    // caller passes loose faces and groups as two lists only when it built
    // them that way (`rest`); a selection interleaves them: see `entities`.
    for &f in faces {
        if let Some(p) = face_poly(scene.mesh(mesh), f, None, h) {
            out.push(p);
        }
    }
    for g in groups {
        push_group(scene, g, h, &mut out);
    }
    out
}

fn push_group(scene: &Scene, g: &ingetrazo_model::scene::Group, h: Hypot, out: &mut Vec<Poly>) {
    for (pg, xf) in iter_placements(g, None) {
        let m = scene.mesh(pg.mesh);
        for &f in &m.faces {
            if let Some(p) = face_poly(m, f, xf.as_ref(), h) {
                out.push(p);
            }
        }
    }
}

/// `polys_of(entities)` in the order of `entities`.
fn polys_of_entities(scene: &Scene, mesh: MeshId, groups: &[ingetrazo_model::scene::Group], entities: &[Entity], h: Hypot) -> Vec<Poly> {
    let mut out = Vec::new();
    for e in entities {
        match *e {
            Entity::Face(f) => {
                if let Some(p) = face_poly(scene.mesh(mesh), f, None, h) {
                    out.push(p);
                }
            }
            Entity::Group(i) => {
                if let Some(g) = groups.get(i) {
                    push_group(scene, g, h, &mut out);
                }
            }
        }
    }
    out
}

fn boxes(polys: &[Poly]) -> Vec<Box3> {
    polys.iter().map(|p| ([p.lo[0] as f64, p.lo[1] as f64, p.lo[2] as f64], [p.hi[0] as f64, p.hi[1] as f64, p.hi[2] as f64])).collect()
}

/// `_pairs(lo_a, hi_a, lo_b, hi_b, same)`: the `(i, j)` pairs `intersect`
/// crosses, in its order.
fn pairs(a: &[Box3], b: &[Box3], same: bool) -> Vec<(usize, usize)> {
    if same {
        let (ii, jj) = touching(a, a, true, BOX_EPS);
        return ii.into_iter().zip(jj).map(|(i, j)| (i as usize, j as usize)).collect();
    }
    // Only the faces that reach the OTHER side's whole box can cross it.
    let extent = |bs: &[Box3]| -> ([f64; 3], [f64; 3]) {
        let mut lo = [f64::INFINITY; 3];
        let mut hi = [f64::NEG_INFINITY; 3];
        for (l, h) in bs {
            for k in 0..3 {
                lo[k] = lo[k].min(l[k]);
                hi[k] = hi[k].max(h[k]);
            }
        }
        (lo, hi)
    };
    let (b_lo, b_hi) = extent(b);
    let ia: Vec<usize> = (0..a.len()).filter(|&i| (0..3).all(|k| a[i].0[k] <= b_hi[k] + BOX_EPS && a[i].1[k] >= b_lo[k] - BOX_EPS)).collect();
    if ia.is_empty() {
        return Vec::new();
    }
    let a2: Vec<Box3> = ia.iter().map(|&i| a[i]).collect();
    let (a_lo, a_hi) = extent(&a2);
    let ib: Vec<usize> = (0..b.len()).filter(|&j| (0..3).all(|k| b[j].0[k] <= a_hi[k] + BOX_EPS && b[j].1[k] >= a_lo[k] - BOX_EPS)).collect();
    if ib.is_empty() {
        return Vec::new();
    }
    let b2: Vec<Box3> = ib.iter().map(|&j| b[j]).collect();
    let (ii, jj) = touching(&a2, &b2, false, BOX_EPS);
    ii.into_iter().zip(jj).map(|(i, j)| (ia[i as usize], ib[j as usize])).collect()
}

/// `intersect(polys_a, polys_b, same)`: every crossing between a face of `a`
/// and one of `b` (`same`: the two lists are one, each pair once), deduped.
pub fn intersect(a: &[Poly], b: &[Poly], same: bool, h: Hypot) -> Vec<Seg> {
    let mut out = Vec::new();
    if a.is_empty() || b.is_empty() {
        return out;
    }
    let ba = boxes(a);
    let bb = if same { ba.clone() } else { boxes(b) };
    for (i, j) in pairs(&ba, &bb, same) {
        let pb = if same { &a[j] } else { &b[j] };
        cross(&a[i], pb, h, &mut out);
    }
    dedupe(out)
}

/// `segments_for(scene, mode)`: the crossing segments Intersect Faces would
/// add. `active` is `scene.mesh` (the loose mesh, or the mesh of the group
/// being edited, whose path is `edit`); `selection` holds the selected faces
/// of that mesh and the selected groups among the context's.
pub fn segments_for(scene: &Scene, active: MeshId, edit: Option<&[usize]>, selection: &[Entity], mode: Mode, h: Hypot) -> Vec<Seg> {
    if selection.is_empty() {
        return Vec::new();
    }
    let kids = context_groups(scene, edit);
    let mine = polys_of_entities(scene, active, kids, selection, h);
    if mode == Mode::Selection {
        return intersect(&mine, &mine, true, h);
    }
    // `rest`: the faces of the active mesh not selected, then (Model) the
    // groups of the context not selected.
    let sel_faces: Vec<FaceId> = selection.iter().filter_map(|e| if let Entity::Face(f) = e { Some(*f) } else { None }).collect();
    let rest_faces: Vec<FaceId> = scene.mesh(active).faces.iter().copied().filter(|f| !sel_faces.contains(f)).collect();
    let rest_groups: Vec<&ingetrazo_model::scene::Group> = if mode == Mode::Model {
        let sel_groups: Vec<usize> = selection.iter().filter_map(|e| if let Entity::Group(i) = e { Some(*i) } else { None }).collect();
        kids.iter().enumerate().filter(|(i, _)| !sel_groups.contains(i)).map(|(_, g)| g).collect()
    } else {
        Vec::new()
    };
    let others = polys_of(scene, active, &rest_faces, &rest_groups, h);
    intersect(&mine, &others, false, h)
}

/// What `_on_intersect_faces` adds: the crossings of `segments_for` at least
/// `WELD_LENGTH` long (a shorter one is a point whose two ends weld into one
/// vertex, and that degenerate edge refused the whole step). Empty means
/// "No faces cross the selection". The edges themselves go through
/// `build_add_edges(scene, &segs, true)`.
pub fn intersect_faces(scene: &Scene, active: MeshId, edit: Option<&[usize]>, selection: &[Entity], mode: Mode, h: Hypot) -> Vec<Seg> {
    segments_for(scene, active, edit, selection, mode, h).into_iter().filter(|&(a, b)| length(b.sub(a), h) as f64 >= WELD_LENGTH).collect()
}
