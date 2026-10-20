// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! Dimension anchors against the scene (`core/dimension.py`: `VertexAnchor`,
//! `resolve_vertex_anchor`, `Dimension._live`, `bind`, `refresh_anchors`).
//!
//! An endpoint placed on a vertex holds it: it reads the vertex's position
//! through the chain of group transforms, and when the hold dies (the vertex
//! changed meshes, was erased or welded away) it looks again where the
//! endpoint last was, or freezes there.
//!
//! Python's chain holds the group objects. Groups here are owned by the tree,
//! so a [`ChainLink`] remembers the index among its siblings, the mesh and the
//! uid, and a link is found again by index first and by uid among its
//! siblings otherwise. A group that moved to another parent is a broken
//! chain (the anchor is dead and the endpoint resolves again) where Python's
//! object would still be alive in its new place.
//!
//! `active` is `scene.mesh`, the mesh being edited; the loose mesh is
//! `MeshId(0)`.

use ingetrazo_math::{M4, V3};

use crate::dimension::{ChainLink, VertexAnchor};
use crate::scene::{Group, MeshId, Scene};

/// Weld resolution of the mesh registry (0.1 mm): `resolve_vertex_anchor`'s `tol`.
pub const ANCHOR_TOL: f64 = 1e-4;

/// Which end of a dimension.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum End {
    A,
    B,
}

fn link_matches(g: &Group, link: &ChainLink) -> bool {
    g.mesh.0 == link.mesh && g.uid == link.uid
}

/// The group a link names among `siblings`.
fn find_link<'a>(siblings: &'a [Group], link: &ChainLink) -> Option<&'a Group> {
    if let Some(g) = siblings.get(link.index) {
        if link_matches(g, link) {
            return Some(g);
        }
    }
    if link.uid.is_empty() {
        return None;
    }
    siblings.iter().find(|g| link_matches(g, link))
}

impl VertexAnchor {
    /// The groups of the chain, top level down; `None` when one of them is no
    /// longer in the scene.
    fn groups<'a>(&self, scene: &'a Scene) -> Option<Vec<&'a Group>> {
        let mut out = Vec::with_capacity(self.chain.len());
        let mut level: &'a [Group] = &scene.groups;
        for link in &self.chain {
            let g = find_link(level, link)?;
            out.push(g);
            level = &g.children;
        }
        Some(out)
    }

    /// `matrix`: the composed placement transform, or `None` when every group
    /// on the chain keeps its geometry in world coordinates.
    fn matrix_of(groups: &[&Group]) -> Option<M4> {
        let mut m: Option<M4> = None;
        for g in groups {
            let Some(x) = g.xform else { continue };
            m = Some(match m {
                None => x,
                Some(acc) => acc.mul(&x),
            });
        }
        m
    }

    /// `alive`: the mesh registers this very vertex, something references it,
    /// and the mesh is still part of the model — the loose mesh, or a group
    /// the scene still lists.
    pub fn alive(&self, scene: &Scene) -> bool {
        let mesh = &scene.meshes[self.mesh as usize];
        let v = self.vertex;
        if v.0 as usize >= mesh.arena_sizes().0 {
            return false;
        }
        if mesh.vertex_at(mesh.position(v)) != Some(v) || mesh.vertex_ref(v).edges.is_empty() {
            return false;
        }
        if self.chain.is_empty() {
            return self.mesh == 0;
        }
        self.groups(scene).is_some()
    }

    /// `position`: the vertex's world position now — `None` once the hold is
    /// no longer alive, which is the anchor's cue to let go.
    pub fn position(&self, scene: &Scene) -> Option<V3> {
        if !self.alive(scene) {
            return None;
        }
        let groups = self.groups(scene)?;
        let p = scene.meshes[self.mesh as usize].position(self.vertex);
        Some(match VertexAnchor::matrix_of(&groups) {
            Some(m) => m.map(p),
            None => p,
        })
    }
}

/// `iter_placements` with each group's path and the links down to it.
fn walk(group: &Group, base: Option<M4>, chain: &mut Vec<ChainLink>, index: usize, out: &mut Vec<(Vec<ChainLink>, MeshId, Option<M4>)>) {
    let m = match (group.xform, base) {
        (m, None) => m,
        (None, Some(b)) => Some(b),
        (Some(x), Some(b)) => Some(b.mul(&x)),
    };
    chain.push(ChainLink { index, mesh: group.mesh.0, uid: group.uid.clone() });
    out.push((chain.clone(), group.mesh, m));
    for (i, c) in group.children.iter().enumerate() {
        walk(c, m, chain, i, out);
    }
    chain.pop();
}

/// `resolve_vertex_anchor`: the vertex sitting at `point` (world) as an
/// anchor, or `None`. The mesh being edited first, then every group
/// placement, the point taken back into each placement's local coordinates.
pub fn resolve_vertex_anchor(scene: &Scene, active: MeshId, point: V3) -> Option<VertexAnchor> {
    let h = scene.meshes[0].hypot;
    let hit = |mesh_id: MeshId, local: V3, chain: Vec<ChainLink>| -> Option<VertexAnchor> {
        let mesh = scene.mesh(mesh_id);
        let v = mesh.vertex_at(local)?;
        if (mesh.position(v).sub(local).length(h) as f64) > ANCHOR_TOL {
            return None;
        }
        if mesh.vertex_ref(v).edges.is_empty() {
            // An orphan: a vertex nothing references any more. A point held by
            // nothing holds nothing.
            return None;
        }
        Some(VertexAnchor { vertex: v, mesh: mesh_id.0, chain })
    };
    if let Some(found) = hit(active, point, Vec::new()) {
        return Some(found);
    }
    for (top, g) in scene.groups.iter().enumerate() {
        if g.billboard.as_ref().is_some_and(crate::pyconv::truthy) {
            continue;
        }
        let mut placements = Vec::new();
        walk(g, None, &mut Vec::new(), top, &mut placements);
        for (chain, mesh, m) in placements {
            let mut local = point;
            if let Some(m) = m {
                let (inv, ok) = m.inverted();
                if !ok {
                    continue;
                }
                local = inv.map(local);
            }
            if let Some(found) = hit(mesh, local, chain) {
                return Some(found);
            }
        }
    }
    None
}

impl Scene {
    /// `Dimension._live` without its side effects: where the end reads now
    /// and the anchor it ends up with.
    fn live_end(&self, anchor: &Option<VertexAnchor>, frozen: V3, active: MeshId) -> (V3, Option<VertexAnchor>) {
        let Some(a) = anchor else { return (frozen, None) };
        if let Some(p) = a.position(self) {
            return (p, anchor.clone());
        }
        // The hold is gone — but the geometry may just have changed hands:
        // look again where the endpoint last was.
        match resolve_vertex_anchor(self, active, frozen) {
            None => (frozen, None),
            Some(again) => match again.position(self) {
                Some(p) => (p, Some(again)),
                None => (frozen, None),
            },
        }
    }

    /// `Dimension.a` / `.b`: the end read through its anchor, as Python reads
    /// it (the frozen position follows, and a dead hold is re-taken).
    pub fn dimension_end(&mut self, i: usize, end: End, active: MeshId) -> V3 {
        let d = &self.dimensions[i];
        let (anchor, frozen) = match end {
            End::A => (d.anchor_a.clone(), d.a),
            End::B => (d.anchor_b.clone(), d.b),
        };
        if anchor.is_none() {
            return frozen;
        }
        let (p, again) = self.live_end(&anchor, frozen, active);
        let d = &mut self.dimensions[i];
        match end {
            End::A => {
                d.anchor_a = again;
                d.a = p;
            }
            End::B => {
                d.anchor_b = again;
                d.b = p;
            }
        }
        p
    }

    /// The ends a dimension is written with (`d.a`, `d.b`) without touching it.
    pub fn dimension_ends(&self, i: usize, active: MeshId) -> (V3, V3) {
        let d = &self.dimensions[i];
        (self.live_end(&d.anchor_a, d.a, active).0, self.live_end(&d.anchor_b, d.b, active).0)
    }

    /// `Dimension.bind`: each end takes the vertex under it, if there is one.
    pub fn bind_dimension(&mut self, i: usize, active: MeshId) {
        let (a, b) = (self.dimensions[i].a, self.dimensions[i].b);
        let (aa, ab) = (resolve_vertex_anchor(self, active, a), resolve_vertex_anchor(self, active, b));
        let d = &mut self.dimensions[i];
        d.anchor_a = aa;
        d.anchor_b = ab;
    }

    pub fn bind_dimensions(&mut self, active: MeshId) {
        for i in 0..self.dimensions.len() {
            self.bind_dimension(i, active);
        }
    }

    /// `Dimension.refresh_anchors`: re-take a hold that died from the vertex
    /// now standing at the endpoint. The history calls it after every command.
    pub fn refresh_dimension_anchors(&mut self, i: usize, active: MeshId) {
        for end in [End::A, End::B] {
            let d = &self.dimensions[i];
            let (anchor, frozen) = match end {
                End::A => (&d.anchor_a, d.a),
                End::B => (&d.anchor_b, d.b),
            };
            let Some(a) = anchor else { continue };
            if a.alive(self) {
                continue;
            }
            let again = resolve_vertex_anchor(self, active, frozen);
            let d = &mut self.dimensions[i];
            match end {
                End::A => d.anchor_a = again,
                End::B => d.anchor_b = again,
            }
        }
    }

    /// `History._rebind_dimensions`.
    pub fn rebind_dimensions(&mut self, active: MeshId) {
        for i in 0..self.dimensions.len() {
            self.refresh_dimension_anchors(i, active);
        }
    }

    /// What reading every dimension does (`d.a`, then `d.b`): the frozen
    /// ends follow the vertices and a dead hold is taken again. Python does
    /// it whenever anything reads a dimension — saving, drawing a frame —
    /// so a frozen end is only as old as the last read.
    pub fn read_dimensions(&mut self, active: MeshId) {
        for i in 0..self.dimensions.len() {
            self.dimension_end(i, End::A, active);
            self.dimension_end(i, End::B, active);
        }
    }

    /// `payload["dimensions"]` as `save_scene` writes it: each end read live.
    pub fn dimensions_igz(&self) -> Vec<ingetrazo_formats::pyjson::Value> {
        (0..self.dimensions.len())
            .map(|i| {
                let (a, b) = self.dimension_ends(i, MeshId(0));
                let mut d = self.dimensions[i].clone();
                d.a = a;
                d.b = b;
                d.to_igz_value()
            })
            .collect()
    }

    /// `Scene.active_section`: the index of the plane cutting the model.
    pub fn active_section(&self) -> Option<usize> {
        self.section_planes.iter().position(|sp| sp.active)
    }

    /// `Scene.set_active_section`: make the plane `id` the ONE active cut
    /// (`None` deactivates all).
    pub fn set_active_section(&mut self, id: Option<u32>) {
        let ids: Vec<u32> = self.section_planes.ids().to_vec();
        for (sp, pid) in self.section_planes.iter_mut().zip(ids) {
            sp.active = Some(pid) == id;
        }
    }
}
