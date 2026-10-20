// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! What [`World::build_cached`](crate::world::World::build_cached) keeps
//! between builds (M4.9): the model is flattened again after every edit, and
//! on a big model that is the cost of an edit. Almost all of the flattening
//! is per mesh - triangulating faces, measuring them, the arcs, the hard
//! edges - and a mesh that was not touched has the same answers. They are
//! kept by mesh, valid while the mesh's [`MeshStamp`] stands, and shared by
//! every placement of the mesh (a component's copies triangulate once).
//!
//! The cache changes no result: a [`Proto`] holds exactly the values
//! `World::build` used to compute inline, in the same order and rounding.

use std::collections::HashMap;
use std::sync::Arc;

use ingetrazo_geom::meshface;
use ingetrazo_geom::triangulate::triangulate;
use ingetrazo_math::{Hypot, V3};
use ingetrazo_model::mesh::{FaceId, Mesh};
use ingetrazo_model::scene::{MeshId, Scene};

use crate::world::{f64v, face_hidden, face_plane_world, loop_positions, pos, tri_rows};

/// What a mesh's content is stamped with: a cached derivation of it is valid
/// while the stamp stands (the renderer's own test: every mutation bumps the
/// serial, and the arenas only grow).
pub type MeshStamp = (u64, (usize, usize, usize));

pub fn mesh_stamp(m: &Mesh) -> MeshStamp {
    (m.serial, m.arena_sizes())
}

/// A prototype mesh in its own coordinates: everything the world needs from
/// a placed group's mesh that does not depend on where it is placed.
pub struct Proto {
    /// The faces the chunk keeps (hidden ones are out of it entirely).
    pub faces: Vec<FaceId>,
    /// `0.5 * |newell|` of each kept face, 0 under three corners.
    pub area: Vec<f64>,
    /// `face_plane_world(face, None)` of each kept face.
    pub plane_local: Vec<(V3, V3)>,
    pub v0s: Vec<[f64; 3]>,
    pub e1s: Vec<[f64; 3]>,
    pub e2s: Vec<[f64; 3]>,
    /// The kept-face ordinal of each triangle.
    pub tri_k: Vec<u32>,
    pub hard: Vec<ingetrazo_model::EdgeId>,
    /// The ends of the hard edges, two per edge.
    pub hard_pts: Vec<[f64; 3]>,
    pub arc_mids: Vec<V3>,
    /// Every vertex position, for the bounds of a moved placement.
    pub verts: Vec<[f64; 3]>,
    /// The box of `verts` (what an unmoved placement contributes).
    pub verts_box: Option<([f64; 3], [f64; 3])>,
}

impl Proto {
    pub fn build(mesh: &Mesh, h: Hypot) -> Proto {
        let faces: Vec<FaceId> = mesh.faces.iter().copied().filter(|&f| !face_hidden(mesh, f)).collect();
        let mut p = Proto {
            faces: Vec::new(),
            area: Vec::new(),
            plane_local: Vec::new(),
            v0s: Vec::new(),
            e1s: Vec::new(),
            e2s: Vec::new(),
            tri_k: Vec::new(),
            hard: Vec::new(),
            hard_pts: Vec::new(),
            arc_mids: crate::circle::mesh_arc_midpoints(mesh, h),
            verts: mesh.vertices.iter().map(|&v| f64v(pos(mesh, v))).collect(),
            verts_box: None,
        };
        for (k, &f) in faces.iter().enumerate() {
            let outer = loop_positions(mesh, &mesh.face(f).outer);
            if outer.len() < 3 {
                p.area.push(0.0);
            } else {
                let n_raw = meshface::newell(&outer);
                let ln = n_raw.length(h);
                let normal = if (ln as f64) > 1e-9 { n_raw.div(ln) } else { V3(0.0, 0.0, 1.0) };
                p.area.push(0.5 * (ln as f64));
                let holes: Vec<Vec<V3>> = mesh.face(f).holes.iter().map(|hl| loop_positions(mesh, hl)).collect();
                for (v0, e1, e2) in tri_rows(&triangulate(&outer, &holes, Some(normal), h)) {
                    p.v0s.push(v0);
                    p.e1s.push(e1);
                    p.e2s.push(e2);
                    p.tri_k.push(k as u32);
                }
            }
            p.plane_local.push(face_plane_world(mesh, f, None, h));
        }
        p.faces = faces;
        p.hard = mesh
            .edges
            .iter()
            .copied()
            .filter(|&e| {
                let ed = mesh.edge(e);
                !ed.hidden && !ed.soft
            })
            .collect();
        for &e in &p.hard {
            let ed = mesh.edge(e);
            p.hard_pts.push(f64v(pos(mesh, ed.v0)));
            p.hard_pts.push(f64v(pos(mesh, ed.v1)));
        }
        if !p.verts.is_empty() {
            let mut lo = [f64::INFINITY; 3];
            let mut hi = [f64::NEG_INFINITY; 3];
            for q in &p.verts {
                for k in 0..3 {
                    lo[k] = lo[k].min(q[k]);
                    hi[k] = hi[k].max(q[k]);
                }
            }
            p.verts_box = Some((lo, hi));
        }
        p
    }
}

/// The prototypes by mesh and the mesh copies a world holds, each valid
/// while its mesh's [`MeshStamp`] stands: an edit rebuilds only the meshes it
/// touched. Drop it ([`WorldCache::clear`]) when the scene object is replaced
/// (serials restart from zero).
#[derive(Default)]
pub struct WorldCache {
    protos: HashMap<u32, (MeshStamp, Arc<Proto>)>,
    meshes: Vec<(MeshStamp, Arc<Mesh>)>,
}

impl WorldCache {
    pub fn clear(&mut self) {
        self.protos.clear();
        self.meshes.clear();
    }

    /// Make, in parallel, the prototypes of `ids` the cache does not hold
    /// fresh (a cold build triangulates every mesh of the model).
    pub fn prebuild(&mut self, scene: &Scene, ids: &[MeshId], h: Hypot) {
        use rayon::prelude::*;
        let mut todo: Vec<MeshId> = Vec::new();
        let mut seen = std::collections::HashSet::new();
        for &id in ids {
            let stamp = mesh_stamp(scene.mesh(id));
            let fresh = self.protos.get(&id.0).is_some_and(|(s, _)| *s == stamp);
            if !fresh && seen.insert(id.0) {
                todo.push(id);
            }
        }
        if todo.len() < 2 {
            return;
        }
        let built: Vec<(MeshId, MeshStamp, Proto)> = todo
            .par_iter()
            .map(|&id| {
                let m = scene.mesh(id);
                (id, mesh_stamp(m), Proto::build(m, h))
            })
            .collect();
        for (id, stamp, p) in built {
            self.protos.insert(id.0, (stamp, Arc::new(p)));
        }
    }

    pub fn proto(&mut self, scene: &Scene, id: MeshId, h: Hypot) -> Arc<Proto> {
        let mesh = scene.mesh(id);
        let stamp = mesh_stamp(mesh);
        if let Some((s, p)) = self.protos.get(&id.0) {
            if *s == stamp {
                return p.clone();
            }
        }
        let p = Arc::new(Proto::build(mesh, h));
        self.protos.insert(id.0, (stamp, p.clone()));
        p
    }

    /// The scene's meshes as shared copies; only the changed ones are cloned.
    pub fn mesh_copies(&mut self, scene: &Scene) -> Vec<Arc<Mesh>> {
        use rayon::prelude::*;
        self.meshes.truncate(scene.meshes.len());
        // The stale copies are cloned in parallel (a cold build clones all).
        let stale: Vec<usize> = (0..scene.meshes.len()).filter(|&i| self.meshes.get(i).is_none_or(|(s, _)| *s != mesh_stamp(&scene.meshes[i]))).collect();
        let fresh: Vec<(usize, MeshStamp, Arc<Mesh>)> = stale.par_iter().map(|&i| (i, mesh_stamp(&scene.meshes[i]), Arc::new(scene.meshes[i].clone()))).collect();
        for (i, stamp, a) in fresh {
            if i < self.meshes.len() {
                self.meshes[i] = (stamp, a);
            } else {
                // `stale` is in index order, so a new copy is always the next slot.
                self.meshes.push((stamp, a));
            }
        }
        self.meshes.iter().map(|(_, a)| a.clone()).collect()
    }
}
