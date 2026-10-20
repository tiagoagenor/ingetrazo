// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! The group-level commands that go with the transforms: Make Unique and
//! Group to Component (`core/history.py`), and the name helpers of
//! `core/kept_components.py` Make Unique uses.

use std::collections::HashSet;

use ingetrazo_math::M4;
use ingetrazo_model::scene::{Group, MeshId, Scene};

use crate::command::{Command, Cx, Fail};
use crate::group::{group_at, group_at_mut, make_unique, placement_paths, Path};

/// `names_in_use`: the names of every group and component in the model —
/// nested ones and the kept templates too — but the one at `skip`'s own.
pub fn names_in_use(scene: &Scene, skip: Option<&[usize]>) -> HashSet<String> {
    let mut out = HashSet::new();
    for p in placement_paths(&scene.groups) {
        if skip == Some(p.as_slice()) {
            continue;
        }
        if let Some(g) = group_at(&scene.groups, &p) {
            if !g.name.is_empty() {
                out.insert(g.name.clone());
            }
        }
    }
    for g in &scene.kept_components {
        if !g.name.is_empty() {
            out.insert(g.name.clone());
        }
    }
    out
}

/// `numbered_name`: `name` made free with a number: «Silla» → «Silla #2»,
/// and a name that already ends in one counts on from it — «Puerta #3» →
/// «Puerta #4» — skipping every name in `taken`.
pub fn numbered_name(name: &str, taken: &HashSet<String>) -> String {
    // `^(.*?)\s*#(\d+)$`: the last '#', digits after it to the end.
    let (base, mut n) = match name.rfind('#') {
        Some(i) if i + 1 < name.len() && name[i + 1..].chars().all(|c| c.is_ascii_digit()) && !name[..i].contains('\n') => {
            (name[..i].trim_end().to_string(), name[i + 1..].parse::<u64>().map(|v| v + 1).unwrap_or(2))
        }
        _ => (name.to_string(), 2),
    };
    while taken.contains(&format!("{base} #{n}")) {
        n += 1;
    }
    format!("{base} #{n}")
}

/// `MakeUniqueCommand`: bake a component instance into its OWN mesh so
/// edits stop touching the siblings' shared definition. With `rename` it
/// also gives the copy a name of its own when another object wears the same.
pub struct MakeUnique {
    pub path: Path,
    pub rename: bool,
    name: Option<String>,
    proto: Option<MeshId>,
    xform: Option<M4>,
    children: Vec<Group>,
    axes: Option<M4>,
    component: bool,
}

impl MakeUnique {
    pub fn new(path: &[usize], rename: bool) -> Self {
        MakeUnique { path: path.to_vec(), rename, name: None, proto: None, xform: None, children: Vec::new(), axes: None, component: true }
    }
}

impl Command for MakeUnique {
    fn name(&self) -> &'static str {
        "make_unique"
    }

    fn apply(&mut self, scene: &mut Scene, cx: &mut Cx) -> Result<(), Fail> {
        let g = group_at(&scene.groups, &self.path).ok_or_else(|| Fail::Failed("the group is no longer in the scene".into()))?;
        self.proto = Some(g.mesh);
        self.xform = g.xform;
        self.axes = g.axes;
        self.component = g.component;
        self.name = Some(g.name.clone());
        let new_name = if self.rename && !g.name.is_empty() {
            let taken = names_in_use(scene, Some(&self.path));
            if taken.contains(&g.name) {
                Some(numbered_name(&g.name, &taken))
            } else {
                None
            }
        } else {
            None
        };
        let Scene { meshes, groups, .. } = scene;
        let g = group_at_mut(groups, &self.path).ok_or_else(|| Fail::Failed("the group is no longer in the scene".into()))?;
        if let Some(n) = new_name {
            g.name = n;
        }
        self.children = make_unique(meshes, g, &mut cx.counters);
        Ok(())
    }

    fn undo(&mut self, scene: &mut Scene, _cx: &mut Cx) {
        let Some(g) = group_at_mut(&mut scene.groups, &self.path) else { return };
        if let Some(p) = self.proto {
            g.mesh = p;
        }
        g.xform = self.xform;
        g.children = std::mem::take(&mut self.children);
        g.axes = self.axes;
        g.component = self.component;
        if let Some(n) = &self.name {
            g.name = n.clone();
        }
    }
}

/// `GroupToComponentCommand`: convert a CLASSIC group into a component
/// instance in place — its mesh becomes the shared definition under an
/// identity transform.
pub struct GroupToComponent {
    pub path: Path,
    pub name: Option<String>,
    old_name: Option<String>,
    old_xform: Option<M4>,
    old_component: bool,
}

impl GroupToComponent {
    pub fn new(path: &[usize], name: Option<String>) -> Self {
        GroupToComponent { path: path.to_vec(), name, old_name: None, old_xform: None, old_component: true }
    }
}

impl Command for GroupToComponent {
    fn name(&self) -> &'static str {
        "group_to_component"
    }

    fn apply(&mut self, scene: &mut Scene, _cx: &mut Cx) -> Result<(), Fail> {
        let g = group_at_mut(&mut scene.groups, &self.path).ok_or_else(|| Fail::Failed("the group is no longer in the scene".into()))?;
        self.old_name = Some(g.name.clone());
        self.old_xform = g.xform;
        self.old_component = g.component;
        if g.xform.is_none() {
            g.xform = Some(M4::identity());
        }
        // A group of groups already carries a matrix: it only changes what
        // it IS — a component from now on (issue #90).
        g.component = true;
        if let Some(n) = &self.name {
            if !n.is_empty() {
                g.name = n.clone();
            }
        }
        Ok(())
    }

    fn undo(&mut self, scene: &mut Scene, _cx: &mut Cx) {
        let Some(g) = group_at_mut(&mut scene.groups, &self.path) else { return };
        g.xform = self.old_xform;
        g.component = self.old_component;
        if let Some(n) = &self.old_name {
            g.name = n.clone();
        }
    }
}
