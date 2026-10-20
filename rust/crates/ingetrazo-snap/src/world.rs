// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! The model as the snap engine sees it: what the viewport's pick index
//! (`Viewport._pick_index`, `_pick_block_part`, `_group_chunk`,
//! `_instance_chunk`) and `_snap_scene` gather from a scene.
//!
//! * the loose faces and the placed groups' faces as float64 triangles
//!   (`v0`, `e1`, `e2`) of one array, each triangle naming its entity (a
//!   face) and the entity its area, visibility, selectability and where it
//!   comes from;
//! * the loose edges (`scene.edges`) and the placed groups' hard edges, in
//!   world space;
//! * the placements (`Viewport._placements()`: a group followed by its
//!   nested placements, with composed matrices);
//! * the guides, the origin points of components and the arc midpoints.
//!
//! A component's triangles are its prototype's, mapped as NumPy maps them:
//! `pts @ L.T + t` in float64 through OpenBLAS (see `ingetrazo-ops`
//! `group.rs`: `fma(a2, b2, fma(a1, b1, a0*b0)) + t` for two or more rows,
//! the `dgemv` order for one), the direction vectors `e1`, `e2` without `t`;
//! its edges the same, then rounded to float32.
//!
//! Not modelled (no fixture reaches them): layers (every entity is on a
//! visible, unlocked layer), the active section cut, hidden-similar
//! components, reference-image borders, survey points, face-me figures and
//! the rest-of-model mode of an open group.

use ingetrazo_geom::meshface;
use ingetrazo_geom::triangulate::triangulate;
use ingetrazo_math::{Hypot, M4, V3};
use ingetrazo_model::guide::Guide;
use ingetrazo_model::mesh::{FaceId, Mesh};
use ingetrazo_model::pyconv::truthy;
use ingetrazo_model::scene::{Group, MeshId, Scene};
use ingetrazo_ops::group::np_affine;

use crate::types::{Context, SnapEdge};

/// A float64 triangle as the pick index holds it.
#[derive(Clone, Copy, Debug)]
pub struct Tri {
    pub v0: [f64; 3],
    pub e1: [f64; 3],
    pub e2: [f64; 3],
    /// The entity (face) it belongs to.
    pub ent: u32,
}

/// Where an entity's face lives.
#[derive(Clone, Copy, Debug)]
pub enum FaceRef {
    /// A face of the loose mesh.
    Loose(FaceId),
    /// A face of a placed group's mesh, through placement `place`.
    Placed { mesh: MeshId, face: FaceId, place: usize },
}

/// One pickable face (`idx.entities` and the `ent_*` arrays).
#[derive(Clone, Copy, Debug)]
pub struct Entity {
    /// `face_plane_world(face, placement.xform)`: the plane a tool reads when
    /// it asks for the placement (`_face_under(placement=True)`).
    pub plane_place: (V3, V3),
    /// `face_plane_world(face, owner.xform)`: the plane through the owner.
    pub plane_owner: (V3, V3),
    pub face: FaceRef,
    pub area: f64,
    /// `ent_vis`: occludes.
    pub vis: bool,
    /// `ent_sel`: can be picked.
    pub sel: bool,
    pub loose: bool,
}

/// A placement the index holds (`idx.ent_placements`).
#[derive(Clone, Debug)]
pub struct Place {
    /// The matrix that puts its mesh in the world (`None` for a classic
    /// top-level group whose mesh is already there).
    pub xform: Option<M4>,
    /// The top-level group's own matrix: what `pick_face_any` hands back as
    /// the face's group.
    pub owner_xform: Option<M4>,
    /// `owner.is_instance()`: the tip says "in component" rather than "in
    /// group".
    pub owner_is_instance: bool,
    pub mesh: MeshId,
    /// The top-level group's own mesh (`owner.mesh`).
    pub owner_mesh: MeshId,
    /// The placement is its own owner (a top-level group).
    pub top_level: bool,
}

/// A group hard edge in world space, as float32 (`gedge_a`, `gedge_b`).
#[derive(Clone, Copy, Debug)]
pub struct GEdge {
    pub a: V3,
    pub b: V3,
    /// The owning placement's `owner_is_instance`.
    pub context: Context,
    /// The placement it belongs to (`places`) and its edge in that mesh.
    pub place: usize,
    pub edge: ingetrazo_model::EdgeId,
}

/// The scene, flattened for the snap engine.
#[derive(Clone, Debug, Default)]
pub struct World {
    pub tris: Vec<Tri>,
    pub ents: Vec<Entity>,
    pub places: Vec<Place>,
    /// `scene.edges` of the loose mesh.
    pub loose_edges: Vec<SnapEdge>,
    pub gedges: Vec<GEdge>,
    /// The component origin points (`_component_origin_points`): one per
    /// placed group that has a matrix.
    pub origins: Vec<V3>,
    /// The middle of every open arc's sweep (`_arc_midpoints`), in world
    /// space.
    pub arc_mids: Vec<V3>,
    pub guides: Vec<Guide>,
    /// `scene.bounds()`.
    pub bounds: Option<(V3, V3)>,
    pub show_hidden_geometry: bool,
    /// Every mesh of the scene (the faces and edges the entities name).
    pub meshes: Vec<Mesh>,
    /// The mesh that is `scene.mesh`: the loose one, or the open group's.
    pub loose_id: u32,
    /// `loose_edges[i]`'s edge in that mesh.
    pub loose_edge_ids: Vec<ingetrazo_model::EdgeId>,
    /// The centre of each loose curve (by curve id): `_center_of_edge`.
    pub curve_centers: std::collections::HashMap<i64, Option<(V3, f64)>>,
}

impl World {
    pub fn has_triangles(&self) -> bool {
        !self.tris.is_empty()
    }
}

/// `np.array(pts) @ L.T (+ t)` over float64 rows, NumPy-through-BLAS order.
pub fn affine_rows(pts: &[[f64; 3]], rot: &[[f64; 3]; 3], trans: Option<&[f64; 3]>) -> Vec<[f64; 3]> {
    let single = pts.len() == 1;
    pts.iter()
        .map(|a| {
            let mut o = [0.0; 3];
            for r in 0..3 {
                let b = &rot[r];
                let s = if single {
                    a[2].mul_add(b[2], a[0].mul_add(b[0], a[1] * b[1]))
                } else {
                    a[2].mul_add(b[2], a[1].mul_add(b[1], a[0] * b[0]))
                };
                o[r] = match trans {
                    Some(t) => s + t[r],
                    None => s,
                };
            }
            o
        })
        .collect()
}

fn pos(mesh: &Mesh, v: ingetrazo_model::VertexId) -> V3 {
    mesh.position(v)
}

fn loop_positions(mesh: &Mesh, l: &[ingetrazo_model::VertexId]) -> Vec<V3> {
    l.iter().map(|&v| pos(mesh, v)).collect()
}

fn f64v(p: V3) -> [f64; 3] {
    [p.0 as f64, p.1 as f64, p.2 as f64]
}

fn face_hidden(mesh: &Mesh, f: FaceId) -> bool {
    mesh.face(f).attrs.iter().any(|(k, v)| k == "hidden" && truthy(v))
}

/// The triangles of a face, `(v0, e1, e2)` in float64 from float32 corners.
fn tri_rows(tris: &[[V3; 3]]) -> Vec<([f64; 3], [f64; 3], [f64; 3])> {
    tris.iter()
        .map(|t| {
            let (a, b, c) = (f64v(t[0]), f64v(t[1]), f64v(t[2]));
            (a, [b[0] - a[0], b[1] - a[1], b[2] - a[2]], [c[0] - a[0], c[1] - a[1], c[2] - a[2]])
        })
        .collect()
}

/// The groups of `scene.groups` expanded into placements, depth first, with
/// composed matrices (`Viewport._placements` / `_expand_placements`).
pub struct PlacementRef<'a> {
    pub group: &'a Group,
    /// `proxy.xform`: the group's own for a top-level one, composed for a
    /// nested one.
    pub xform: Option<M4>,
    /// The top-level group (`owner`).
    pub owner: &'a Group,
    /// Index of the top-level group in `scene.groups`.
    pub top: usize,
    pub hidden: bool,
}

pub fn placements(scene: &Scene) -> Vec<PlacementRef<'_>> {
    let mut out = Vec::new();
    for (gi, g) in scene.groups.iter().enumerate() {
        out.push(PlacementRef { group: g, xform: g.xform, owner: g, top: gi, hidden: g.hidden });
        fn walk<'a>(node: &'a Group, world: Option<M4>, top: &'a Group, gi: usize, hidden: bool, out: &mut Vec<PlacementRef<'a>>) {
            for child in &node.children {
                let m = match (world, child.xform) {
                    (None, x) => x,
                    (Some(w), Some(x)) => Some(w.mul(&x)),
                    (Some(w), None) => Some(w),
                };
                let hid = hidden || child.hidden;
                out.push(PlacementRef { group: child, xform: m, owner: top, top: gi, hidden: hid });
                if !child.children.is_empty() {
                    walk(child, m, top, gi, hid, out);
                }
            }
        }
        if !g.children.is_empty() {
            walk(g, g.xform, g, gi, g.hidden, &mut out);
        }
    }
    out
}

/// `face_plane_world(face, xform)`: a face's plane in world coordinates.
pub fn face_plane_world(mesh: &Mesh, f: FaceId, xform: Option<&M4>, h: Hypot) -> (V3, V3) {
    let outer = loop_positions(mesh, &mesh.face(f).outer);
    let Some(xf) = xform else {
        return (meshface::centroid(&outer), meshface::normal(&outer, h));
    };
    let pts: Vec<V3> = outer.iter().map(|&v| xf.map(v)).collect();
    let n = pts.len();
    let mut c = V3(0.0, 0.0, 0.0);
    for &p in &pts {
        c = c.add(p);
    }
    let c = c.div(n as f32);
    let (mut nx, mut ny, mut nz) = (0.0f64, 0.0f64, 0.0f64);
    for i in 0..n {
        let (a, b) = (pts[i], pts[(i + 1) % n]);
        let (ax, ay, az) = (a.0 as f64, a.1 as f64, a.2 as f64);
        let (bx, by, bz) = (b.0 as f64, b.1 as f64, b.2 as f64);
        nx += (ay - by) * (az + bz);
        ny += (az - bz) * (ax + bx);
        nz += (ax - bx) * (ay + by);
    }
    let normal = V3(nx as f32, ny as f32, nz as f32);
    if (normal.length(h) as f64) < 1e-12 {
        return (c, xf.map_vector(meshface::normal(&outer, h)).normalized(h));
    }
    (c, normal.normalized(h))
}

impl World {
    /// Gather `scene` as the viewport does with nothing open for editing.
    pub fn build(scene: &Scene, h: Hypot) -> World {
        World::build_ctx(scene, h, None)
    }

    /// Gather `scene` with the top-level group `gi` open for editing (one
    /// level): `Scene.begin_group_edit` points `scene.mesh` at the group's
    /// mesh — for a component instance a world-space copy of its
    /// definition, the instance left without a matrix — and the pick index
    /// leaves the group itself out, so everything the cursor meets of it is
    /// the loose part.
    pub fn build_editing(scene: &Scene, gi: usize, h: Hypot) -> World {
        let mut sc = scene.clone();
        let g = sc.groups[gi].clone();
        if g.children.is_empty() {
            if let Some(xf) = g.xform {
                let mut m = xf;
                let mut curve = 1i64;
                let world_copy = ingetrazo_ops::group::transformed_mesh(scene.mesh(g.mesh), &mut m, &mut curve);
                sc.meshes.push(world_copy);
                sc.groups[gi].mesh = MeshId(sc.meshes.len() as u32 - 1);
                sc.groups[gi].xform = None;
            }
        }
        World::build_ctx(&sc, h, Some(gi))
    }

    fn build_ctx(scene: &Scene, h: Hypot, editing: Option<usize>) -> World {
        let mut w = World { show_hidden_geometry: scene.extra.show_hidden_geometry, ..Default::default() };
        // What `scene.mesh` is: the loose mesh, or the open group's.
        let loose_id = match editing {
            Some(gi) => scene.groups[gi].mesh,
            None => MeshId(0),
        };
        let loose = scene.mesh(loose_id);
        w.meshes = scene.meshes.clone();
        w.loose_id = loose_id.0;

        // ---- loose faces: entities, triangles
        for &f in &loose.faces {
            let outer = loop_positions(loose, &loose.face(f).outer);
            let hidden = face_hidden(loose, f);
            let vis = !hidden;
            let sel = !hidden || scene.extra.show_hidden_geometry;
            let area = if outer.len() < 3 { 0.0 } else { 0.5 * (meshface::newell(&outer).length(h) as f64) };
            let i = w.ents.len() as u32;
            let plane = face_plane_world(loose, f, None, h);
            w.ents.push(Entity { plane_place: plane, plane_owner: plane, face: FaceRef::Loose(f), area, vis, sel, loose: true });
            if outer.len() >= 3 {
                let holes: Vec<Vec<V3>> = loose.face(f).holes.iter().map(|hl| loop_positions(loose, hl)).collect();
                let n = meshface::newell(&outer);
                let normal = if (n.length(h) as f64) > 1e-9 { n.normalized(h) } else { V3(0.0, 0.0, 1.0) };
                for (v0, e1, e2) in tri_rows(&triangulate(&outer, &holes, Some(normal), h)) {
                    w.tris.push(Tri { v0, e1, e2, ent: i });
                }
            }
        }

        // ---- loose edges
        for &e in &loose.edges {
            let ed = loose.edge(e);
            let mut se = SnapEdge::new(pos(loose, ed.v0), pos(loose, ed.v1));
            se.soft = ed.soft;
            se.hidden = ed.hidden;
            se.curve = ed.curve;
            if let Some(cid) = ed.curve {
                w.curve_centers.entry(cid).or_insert_with(|| crate::circle::center_of_edge(loose, e));
            }
            w.loose_edges.push(se);
            w.loose_edge_ids.push(e);
        }

        // ---- guides
        // The scene holds them typed (M3.9).
        for g in scene.guides.iter() {
            w.guides.push(g.clone());
        }

        // ---- placed groups
        let show_objs = scene.extra.show_hidden_objects;
        let all = placements(scene);
        let mut all_pts: Vec<(f64, f64, f64, f64, f64, f64)> = Vec::new();
        w.arc_mids = crate::circle::mesh_arc_midpoints(loose, h);
        for p in &all {
            let g = p.group;
            if g.billboard.as_ref().map(truthy).unwrap_or(false) {
                continue;
            }
            let mesh = scene.mesh(g.mesh);
            for m in crate::circle::mesh_arc_midpoints(mesh, h) {
                w.arc_mids.push(match &p.xform {
                    Some(x) => x.map(m),
                    None => m,
                });
            }
            // Origin points: every placed group with a matrix.
            if let Some(xf) = &p.xform {
                w.origins.push(xf.map(V3(0.0, 0.0, 0.0)));
            }
            // bounds (vectorised over the mesh's vertices)
            if !mesh.vertices.is_empty() {
                let mut lo = [f64::INFINITY; 3];
                let mut hi = [f64::NEG_INFINITY; 3];
                let pts: Vec<[f64; 3]> = mesh.vertices.iter().map(|&v| f64v(pos(mesh, v))).collect();
                let mapped = match &p.xform {
                    Some(m) => {
                        let (rot, trans) = np_affine(m);
                        affine_rows(&pts, &rot, Some(&trans))
                    }
                    None => pts,
                };
                for q in &mapped {
                    for k in 0..3 {
                        lo[k] = lo[k].min(q[k]);
                        hi[k] = hi[k].max(q[k]);
                    }
                }
                all_pts.push((lo[0], lo[1], lo[2], hi[0], hi[1], hi[2]));
            }
        }
        // The visible check of `placements()` is the group's `hidden`; ours
        // has no layers, so every placement counts.

        for p in &all {
            let g = p.group;
            if g.billboard.as_ref().map(truthy).unwrap_or(false) {
                continue;
            }
            // The open group (and what it owns) is the loose part, not a
            // placement of the index.
            if editing == Some(p.top) {
                continue;
            }
            let gvis = !p.hidden;
            let gsnap = !p.hidden || show_objs;
            let gsnap = gsnap && !g.locked && !p.owner.locked;
            // At the root every placement can be picked; inside a group only
            // its own children can, and none of the groups left is one.
            let gsel = gsnap && editing.is_none();
            if !(gvis || gsnap) {
                continue;
            }
            let mesh = scene.mesh(g.mesh);
            // faces the chunk keeps (hidden ones are out of it entirely)
            let faces: Vec<FaceId> = mesh.faces.iter().copied().filter(|&f| !face_hidden(mesh, f)).collect();
            let hard: Vec<_> = mesh
                .edges
                .iter()
                .copied()
                .filter(|&e| {
                    let ed = mesh.edge(e);
                    !ed.hidden && !ed.soft
                })
                .collect();
            if faces.is_empty() && hard.is_empty() {
                continue;
            }
            let place_ix = w.places.len();
            w.places.push(Place {
                xform: p.xform,
                owner_xform: p.owner.xform,
                owner_is_instance: p.owner.xform.is_some(),
                mesh: g.mesh,
                owner_mesh: p.owner.mesh,
                top_level: std::ptr::eq(p.group, p.owner),
            });
            // local triangles of this prototype
            let mut local: Vec<([f64; 3], [f64; 3], [f64; 3], u32)> = Vec::new();
            let base_ent = w.ents.len() as u32;
            for (k, &f) in faces.iter().enumerate() {
                let outer = loop_positions(mesh, &mesh.face(f).outer);
                let area;
                if outer.len() < 3 {
                    area = 0.0;
                } else {
                    let n_raw = meshface::newell(&outer);
                    let ln = n_raw.length(h);
                    let normal = if (ln as f64) > 1e-9 { n_raw.div(ln) } else { V3(0.0, 0.0, 1.0) };
                    area = 0.5 * (ln as f64);
                    let holes: Vec<Vec<V3>> = mesh.face(f).holes.iter().map(|hl| loop_positions(mesh, hl)).collect();
                    for (v0, e1, e2) in tri_rows(&triangulate(&outer, &holes, Some(normal), h)) {
                        local.push((v0, e1, e2, base_ent + k as u32));
                    }
                }
                w.ents.push(Entity {
                    plane_place: face_plane_world(mesh, f, p.xform.as_ref(), h),
                    plane_owner: face_plane_world(mesh, f, p.owner.xform.as_ref(), h),
                    face: FaceRef::Placed { mesh: g.mesh, face: f, place: place_ix },
                    area: if p.xform.is_some() { area * abs_det_pow(&p.xform) } else { area },
                    vis: gvis,
                    sel: gsel,
                    loose: false,
                });
            }
            // map them into the world
            match &p.xform {
                None => {
                    for (v0, e1, e2, ent) in local {
                        w.tris.push(Tri { v0, e1, e2, ent });
                    }
                }
                Some(m) => {
                    let (rot, trans) = np_affine(m);
                    let v0s: Vec<[f64; 3]> = local.iter().map(|t| t.0).collect();
                    let e1s: Vec<[f64; 3]> = local.iter().map(|t| t.1).collect();
                    let e2s: Vec<[f64; 3]> = local.iter().map(|t| t.2).collect();
                    let v0w = affine_rows(&v0s, &rot, Some(&trans));
                    let e1w = affine_rows(&e1s, &rot, None);
                    let e2w = affine_rows(&e2s, &rot, None);
                    for (i, t) in local.iter().enumerate() {
                        w.tris.push(Tri { v0: v0w[i], e1: e1w[i], e2: e2w[i], ent: t.3 });
                    }
                }
            }
            // hard edges
            if gsnap && !hard.is_empty() {
                let ctx = if p.owner.xform.is_some() { Context::Component } else { Context::Group };
                let pts: Vec<[f64; 3]> = hard
                    .iter()
                    .flat_map(|&e| {
                        let ed = mesh.edge(e);
                        [f64v(pos(mesh, ed.v0)), f64v(pos(mesh, ed.v1))]
                    })
                    .collect();
                let mapped: Vec<V3> = match &p.xform {
                    Some(m) => {
                        let (rot, trans) = np_affine(m);
                        affine_rows(&pts, &rot, Some(&trans)).iter().map(|q| V3(q[0] as f32, q[1] as f32, q[2] as f32)).collect()
                    }
                    None => pts.iter().map(|q| V3(q[0] as f32, q[1] as f32, q[2] as f32)).collect(),
                };
                for (k, pair) in mapped.chunks(2).enumerate() {
                    w.gedges.push(GEdge { a: pair[0], b: pair[1], context: ctx, place: place_ix, edge: hard[k] });
                }
            }
        }

        // ---- scene.bounds(): the REAL loose mesh and every placement
        w.bounds = scene_bounds(scene, h, &all_pts);
        w
    }
}

/// `abs(det) ** (2/3)`: how an instance chunk scales the prototype's areas.
fn abs_det_pow(m: &Option<M4>) -> f64 {
    let Some(m) = m else { return 1.0 };
    let (rot, _) = np_affine(m);
    // np.linalg.det of the 3x3 (LU); only its size matters here.
    let det = rot[0][0] * (rot[1][1] * rot[2][2] - rot[1][2] * rot[2][1])
        - rot[0][1] * (rot[1][0] * rot[2][2] - rot[1][2] * rot[2][0])
        + rot[0][2] * (rot[1][0] * rot[2][1] - rot[1][1] * rot[2][0]);
    det.abs().powf(2.0 / 3.0)
}

/// `Scene.bounds()` of the loose mesh and the placements.
fn scene_bounds(scene: &Scene, _h: Hypot, groups: &[(f64, f64, f64, f64, f64, f64)]) -> Option<(V3, V3)> {
    let loose = scene.loose();
    let mut lo = [f64::INFINITY; 3];
    let mut hi = [f64::NEG_INFINITY; 3];
    let mut seen = false;
    let mut absorb = |p: V3| {
        seen = true;
        let q = f64v(p);
        for k in 0..3 {
            if q[k] < lo[k] {
                lo[k] = q[k];
            }
            if q[k] > hi[k] {
                hi[k] = q[k];
            }
        }
    };
    for &e in &loose.edges {
        let ed = loose.edge(e);
        absorb(pos(loose, ed.v0));
        absorb(pos(loose, ed.v1));
    }
    for &f in &loose.faces {
        if face_hidden(loose, f) {
            continue;
        }
        for &v in &loose.face(f).outer {
            absorb(pos(loose, v));
        }
    }
    for g in groups {
        seen = true;
        let (l, hh) = ([g.0, g.1, g.2], [g.3, g.4, g.5]);
        for k in 0..3 {
            lo[k] = lo[k].min(l[k]);
            hi[k] = hi[k].max(hh[k]);
        }
    }
    if !seen {
        return None;
    }
    Some((V3(lo[0] as f32, lo[1] as f32, lo[2] as f32), V3(hi[0] as f32, hi[1] as f32, hi[2] as f32)))
}
