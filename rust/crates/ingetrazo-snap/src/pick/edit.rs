// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! Opening a group for editing as the picks see it (`Scene.begin_group_edit`
//! of `core/scene.py`): inside a group the tools speak world coordinates, so
//! a component instance opens on a world copy of its definition, a container
//! pushes its matrix down into its children and its own mesh, and the loose
//! part of the scene becomes the open group's mesh.
//!
//! The scene is changed the way Python changes it; the picks then read the
//! changed scene like any other. Exploded-view offsets and the group's own
//! axes, which the picks never read, are left alone.

use ingetrazo_math::{Hypot, M4};
use ingetrazo_model::scene::{MeshId, Scene};
use ingetrazo_ops::group::{group_at, group_at_mut, transformed_mesh};

/// The scene after the groups along `path` have been opened one inside the
/// other (`begin_group_edit` of each prefix), and the mesh that is then the
/// loose part.
pub fn enter_edit(scene: &Scene, path: &[usize], _h: Hypot) -> (Scene, MeshId) {
    let mut sc = scene.clone();
    let mut active = MeshId(0);
    let mut curve: i64 = 1;
    for k in 1..=path.len() {
        let prefix = &path[..k];
        let (children, xform, mesh) = {
            let g = group_at(&sc.groups, prefix).expect("a group on the path");
            (!g.children.is_empty(), g.xform, g.mesh)
        };
        if children {
            // `_bake_container_xform`
            if let Some(x) = xform {
                if x.m != M4::identity().m {
                    {
                        let g = group_at_mut(&mut sc.groups, prefix).unwrap();
                        for c in &mut g.children {
                            c.xform = Some(x.mul(&c.xform.unwrap_or_else(M4::identity)));
                        }
                    }
                    if !sc.mesh(mesh).vertices.is_empty() {
                        let mut m = x;
                        let world = transformed_mesh(sc.mesh(mesh), &mut m, &mut curve);
                        sc.meshes.push(world);
                        let id = MeshId(sc.meshes.len() as u32 - 1);
                        group_at_mut(&mut sc.groups, prefix).unwrap().mesh = id;
                    }
                    group_at_mut(&mut sc.groups, prefix).unwrap().xform = Some(M4::identity());
                }
            }
        } else if let Some(x) = xform {
            // A component instance: a world copy of the definition.
            let mut m = x;
            let world = transformed_mesh(sc.mesh(mesh), &mut m, &mut curve);
            sc.meshes.push(world);
            let id = MeshId(sc.meshes.len() as u32 - 1);
            let g = group_at_mut(&mut sc.groups, prefix).unwrap();
            g.mesh = id;
            g.xform = None;
        }
        active = group_at(&sc.groups, prefix).unwrap().mesh;
    }
    (sc, active)
}
