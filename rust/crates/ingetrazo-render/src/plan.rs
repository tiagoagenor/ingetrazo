// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! The placements of a scene: what `Viewport._placements()` expands the group
//! tree into. One entry per placement, each drawing its own prototype mesh
//! through its own world matrix. The loose mesh is placement zero.

use ingetrazo_formats::pyjson::Value;
use ingetrazo_math::M4;
use ingetrazo_model::scene::{Group, MeshId, Scene};
use ingetrazo_ops::group::effective_material;

use crate::bake::LayerTable;

/// Path of a group: indexes into `scene.groups`, then into `children`.
/// Empty = the loose mesh.
pub type GroupPath = Vec<usize>;

#[derive(Clone, Debug)]
pub struct Placement {
    pub path: GroupPath,
    pub mesh: MeshId,
    /// Local to world; `None` = identity.
    pub matrix: Option<M4>,
    /// The container material dressing the default faces.
    pub paint: Option<Value>,
    /// A hidden object shown by View > Hidden Objects: drawn as a ghost.
    pub ghost: bool,
    /// Negative determinant: the winding flips.
    pub mirrored: bool,
    /// Outside the group open for editing: washed toward the background.
    pub faded: bool,
}

/// How the context of an open group is treated (`EDIT_REST_MODES`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum EditRest {
    Normal,
    #[default]
    Fade,
    Hide,
}

#[derive(Clone, Debug, Default)]
pub struct PlanOptions {
    pub show_hidden_objects: bool,
    /// The group open for editing.
    pub edit_path: Option<GroupPath>,
    pub edit_rest: EditRest,
}

/// Determinant sign of the upper-left 3x3.
pub fn is_mirrored(m: &M4) -> bool {
    let a = |c: usize, r: usize| m.m[c][r] as f64;
    let det = a(0, 0) * (a(1, 1) * a(2, 2) - a(2, 1) * a(1, 2))
        - a(1, 0) * (a(0, 1) * a(2, 2) - a(2, 1) * a(0, 2))
        + a(2, 0) * (a(0, 1) * a(1, 2) - a(1, 1) * a(0, 2));
    det < 0.0
}

pub fn collect_placements(scene: &Scene, layers: &LayerTable, opts: &PlanOptions) -> Vec<Placement> {
    let editing = opts.edit_path.is_some();
    let mut out = Vec::new();
    // The loose mesh: the context while a group is open.
    if !(editing && opts.edit_rest == EditRest::Hide) {
        out.push(Placement {
            path: Vec::new(),
            mesh: MeshId(0),
            matrix: None,
            paint: None,
            ghost: false,
            mirrored: false,
            faded: editing && opts.edit_rest == EditRest::Fade,
        });
    }
    for (i, g) in scene.groups.iter().enumerate() {
        let mut path = vec![i];
        walk(g, &mut path, None, false, layers, g, opts, &mut out);
    }
    out
}

#[allow(clippy::too_many_arguments)]
fn walk(
    g: &Group,
    path: &mut GroupPath,
    base: Option<M4>,
    parent_hidden: bool,
    layers: &LayerTable,
    top: &Group,
    opts: &PlanOptions,
    out: &mut Vec<Placement>,
) {
    let m = match (g.xform, base) {
        (m, None) => m,
        (None, Some(b)) => Some(b),
        (Some(x), Some(b)) => Some(b.mul(&x)),
    };
    let hidden = parent_hidden || g.hidden;
    // A hidden LAYER stays hidden, whatever the hidden-objects view says.
    if !layers.visible(g.layer.as_deref()) {
        return;
    }
    if hidden && !opts.show_hidden_objects {
        return;
    }
    // Face-me billboards are drawn by their own pass (not in M4.1).
    if g.billboard.is_some() {
        return;
    }
    let in_subject = opts.edit_path.as_ref().is_some_and(|e| path.starts_with(e));
    let skip_context = opts.edit_path.is_some() && !in_subject && opts.edit_rest == EditRest::Hide;
    if !skip_context {
        let paint = effective_material(Some(g), Some(top)).cloned();
        out.push(Placement {
            path: path.clone(),
            mesh: g.mesh,
            matrix: m,
            paint,
            ghost: hidden,
            mirrored: m.as_ref().is_some_and(is_mirrored),
            faded: opts.edit_path.is_some() && !in_subject && opts.edit_rest == EditRest::Fade,
        });
    }
    for (i, c) in g.children.iter().enumerate() {
        path.push(i);
        walk(c, path, m, hidden, layers, top, opts, out);
        path.pop();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loose_mesh_is_placement_zero_and_groups_follow() {
        let mut sc = Scene::new();
        sc.groups.push(Group::new(MeshId(0), "g".into()));
        let layers = LayerTable::default();
        let p = collect_placements(&sc, &layers, &PlanOptions::default());
        assert_eq!(p.len(), 2);
        assert!(p[0].path.is_empty() && p[1].path == vec![0]);
        assert!(!p[0].faded);
    }

    #[test]
    fn hidden_groups_are_skipped_or_ghosted() {
        let mut sc = Scene::new();
        let mut g = Group::new(MeshId(0), "g".into());
        g.hidden = true;
        sc.groups.push(g);
        let layers = LayerTable::default();
        assert_eq!(collect_placements(&sc, &layers, &PlanOptions::default()).len(), 1);
        let o = PlanOptions { show_hidden_objects: true, ..Default::default() };
        let p = collect_placements(&sc, &layers, &o);
        assert_eq!(p.len(), 2);
        assert!(p[1].ghost);
    }

    #[test]
    fn mirrored_matrices_are_detected() {
        let mut m = M4::identity();
        m.scale(-1.0, 1.0, 1.0);
        assert!(is_mirrored(&m));
        assert!(!is_mirrored(&M4::identity()));
    }

    #[test]
    fn editing_fades_the_context() {
        let mut sc = Scene::new();
        sc.groups.push(Group::new(MeshId(0), "a".into()));
        sc.groups.push(Group::new(MeshId(0), "b".into()));
        let layers = LayerTable::default();
        let o = PlanOptions { edit_path: Some(vec![1]), ..Default::default() };
        let p = collect_placements(&sc, &layers, &o);
        assert!(p[0].faded && p[1].faded && !p[2].faded);
    }
}
