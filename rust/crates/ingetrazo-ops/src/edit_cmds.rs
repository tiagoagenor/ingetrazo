// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! The attribute commands of `core/history.py` (M3.10): Hide, paint (face
//! colour, opacity, texture, back side, material tag), material finish and
//! restamp, Assign Layer, Purge Unused, Explode Curves and the extension
//! data command; with `core/purge.py`.
//!
//! None of them changes topology, so Python swaps attrs and flags directly
//! and so does this: every command captures what it overwrites on its first
//! `do` (or on every `do` where Python does) and writes it back on undo.
//!
//! Entities are named by where they live: a group by its path, a face or an
//! edge by its mesh and id.

use std::collections::HashSet;

use ingetrazo_formats::pyjson::Value;
use ingetrazo_model::layers::DEFAULT_LAYER;
use ingetrazo_model::materials::Material;
use ingetrazo_model::pyval::{self, truthy};
use ingetrazo_model::scene::{Group, MeshId, Scene};
use ingetrazo_model::{EdgeId, FaceId};

use crate::command::{Command, Cx, Fail};
use crate::group::{group_at, group_at_mut, iter_placements, Path};

fn gone() -> Fail {
    Fail::Failed("the entity is no longer in the scene".into())
}

/// A live object a command holds by reference.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Entity {
    Group(Path),
    Edge(MeshId, EdgeId),
    Face(MeshId, FaceId),
}

fn get_attr(attrs: &[(String, Value)], key: &str) -> Option<Value> {
    attrs.iter().find(|(k, _)| k == key).map(|(_, v)| v.clone())
}

fn pop_attr(attrs: &mut Vec<(String, Value)>, key: &str) {
    attrs.retain(|(k, _)| k != key);
}

/// Put `value` back under `key`, or remove the key when it was absent.
fn restore_attr(attrs: &mut Vec<(String, Value)>, key: &str, old: &Option<Value>) {
    match old {
        None => pop_attr(attrs, key),
        Some(v) => pyval::set(attrs, key, v.clone()),
    }
}

fn face_attrs_mut(scene: &mut Scene, f: (MeshId, FaceId)) -> &mut Vec<(String, Value)> {
    &mut scene.meshes[f.0 .0 as usize].face_mut(f.1).attrs
}

// ---- Hide ---------------------------------------------------------------------------------

fn is_hidden(scene: &Scene, e: &Entity) -> Result<bool, Fail> {
    Ok(match e {
        Entity::Face(m, f) => get_attr(&scene.mesh(*m).face(*f).attrs, "hidden").is_some_and(|v| truthy(&v)),
        Entity::Edge(m, e) => scene.mesh(*m).edge(*e).hidden,
        Entity::Group(p) => group_at(&scene.groups, p).ok_or_else(gone)?.hidden,
    })
}

fn set_hidden(scene: &mut Scene, e: &Entity, hidden: bool) -> Result<(), Fail> {
    match e {
        Entity::Face(m, f) => {
            let attrs = face_attrs_mut(scene, (*m, *f));
            if hidden {
                pyval::set(attrs, "hidden", Value::Bool(true));
            } else {
                pop_attr(attrs, "hidden");
            }
        }
        Entity::Edge(m, e) => scene.meshes[m.0 as usize].edge_mut(*e).hidden = hidden,
        Entity::Group(p) => group_at_mut(&mut scene.groups, p).ok_or_else(gone)?.hidden = hidden,
    }
    Ok(())
}

/// `HideCommand` (and `HideEdgesCommand`, its edges-only spelling).
pub struct Hide {
    entities: Vec<Entity>,
    hidden: bool,
    prev: Vec<bool>,
    name: &'static str,
}

impl Hide {
    pub fn new(entities: Vec<Entity>, hidden: bool) -> Self {
        Hide { entities, hidden, prev: Vec::new(), name: "hide" }
    }

    pub fn edges(edges: Vec<Entity>, hidden: bool) -> Self {
        Hide { entities: edges, hidden, prev: Vec::new(), name: "hide_edges" }
    }
}

impl Command for Hide {
    fn name(&self) -> &'static str {
        self.name
    }
    fn apply(&mut self, scene: &mut Scene, _cx: &mut Cx) -> Result<(), Fail> {
        self.prev = self.entities.iter().map(|e| is_hidden(scene, e)).collect::<Result<_, _>>()?;
        for e in &self.entities {
            set_hidden(scene, e, self.hidden)?;
        }
        Ok(())
    }
    fn undo(&mut self, scene: &mut Scene, _cx: &mut Cx) {
        for (e, prev) in self.entities.iter().zip(&self.prev) {
            let _ = set_hidden(scene, e, *prev);
        }
    }
}

// ---- face paint -------------------------------------------------------------------------------

type FaceRef = (MeshId, FaceId);

/// What the paint commands share: set one attr key on a list of faces (or
/// clear it with `None`), remember the old values on the first `do`.
struct AttrSwap {
    faces: Vec<FaceRef>,
    key: &'static str,
    old: Option<Vec<Option<Value>>>,
}

impl AttrSwap {
    fn new(faces: &[FaceRef], key: &'static str) -> Self {
        AttrSwap { faces: faces.to_vec(), key, old: None }
    }

    fn capture(&mut self, scene: &Scene) {
        if self.old.is_none() {
            self.old = Some(self.faces.iter().map(|&(m, f)| get_attr(&scene.mesh(m).face(f).attrs, self.key)).collect());
        }
    }

    fn put(&self, scene: &mut Scene, value: &Option<Value>) {
        for &f in &self.faces {
            let attrs = face_attrs_mut(scene, f);
            match value {
                None => pop_attr(attrs, self.key),
                Some(v) => pyval::set(attrs, self.key, v.clone()),
            }
        }
    }

    fn restore(&self, scene: &mut Scene, copy: impl Fn(&Value) -> Value) {
        for (&f, old) in self.faces.iter().zip(self.old.iter().flatten()) {
            let attrs = face_attrs_mut(scene, f);
            match old {
                None => pop_attr(attrs, self.key),
                Some(v) => pyval::set(attrs, self.key, copy(v)),
            }
        }
    }
}

/// `SetFaceColorCommand`: `attrs["color"]` (`None` clears it).
pub struct SetFaceColor {
    swap: AttrSwap,
    color: Option<Value>,
}

impl SetFaceColor {
    /// `color`: the colour's components (`list(color)`).
    pub fn new(faces: &[FaceRef], color: Option<Vec<Value>>) -> Self {
        SetFaceColor { swap: AttrSwap::new(faces, "color"), color: color.map(Value::Array) }
    }
}

impl Command for SetFaceColor {
    fn name(&self) -> &'static str {
        "set_face_color"
    }
    fn apply(&mut self, scene: &mut Scene, _cx: &mut Cx) -> Result<(), Fail> {
        self.swap.capture(scene);
        self.swap.put(scene, &self.color);
        Ok(())
    }
    fn undo(&mut self, scene: &mut Scene, _cx: &mut Cx) {
        self.swap.restore(scene, |v| match v {
            Value::Array(a) => Value::Array(a.clone()),
            other => other.clone(),
        });
    }
}

/// `SetFaceOpacityCommand`: `attrs["opacity"]` (`None` clears it).
pub struct SetFaceOpacity {
    swap: AttrSwap,
    opacity: Option<Value>,
}

impl SetFaceOpacity {
    pub fn new(faces: &[FaceRef], opacity: Option<f64>) -> Self {
        SetFaceOpacity { swap: AttrSwap::new(faces, "opacity"), opacity: opacity.map(Value::Float) }
    }
}

impl Command for SetFaceOpacity {
    fn name(&self) -> &'static str {
        "set_face_opacity"
    }
    fn apply(&mut self, scene: &mut Scene, _cx: &mut Cx) -> Result<(), Fail> {
        self.swap.capture(scene);
        self.swap.put(scene, &self.opacity);
        Ok(())
    }
    fn undo(&mut self, scene: &mut Scene, _cx: &mut Cx) {
        self.swap.restore(scene, Value::clone);
    }
}

/// `SetFaceTextureCommand`: `attrs["texture"]` (`None` clears it).
pub struct SetFaceTexture {
    swap: AttrSwap,
    tex: Option<Value>,
}

impl SetFaceTexture {
    /// `texture`: the `{"path", "sw", "sh"}` dict.
    pub fn new(faces: &[FaceRef], texture: Option<Value>) -> Self {
        SetFaceTexture { swap: AttrSwap::new(faces, "texture"), tex: texture }
    }
}

impl Command for SetFaceTexture {
    fn name(&self) -> &'static str {
        "set_face_texture"
    }
    fn apply(&mut self, scene: &mut Scene, _cx: &mut Cx) -> Result<(), Fail> {
        self.swap.capture(scene);
        self.swap.put(scene, &self.tex);
        Ok(())
    }
    fn undo(&mut self, scene: &mut Scene, _cx: &mut Cx) {
        self.swap.restore(scene, Value::clone);
    }
}

/// `SetFaceBackCommand`: the BACK side's material dict per face (`None`
/// puts the side back to the style's default).
pub struct SetFaceBack {
    swap: AttrSwap,
    backs: Vec<Option<Value>>,
}

impl SetFaceBack {
    /// One dict (or `None`) per face.
    pub fn new(faces: &[FaceRef], backs: Vec<Option<Value>>) -> Self {
        let backs = backs.into_iter().map(|b| b.filter(|v| matches!(v, Value::Object(_)))).collect();
        SetFaceBack { swap: AttrSwap::new(faces, "back"), backs }
    }

    /// One dict for all of them.
    pub fn same(faces: &[FaceRef], back: Option<Value>) -> Self {
        SetFaceBack::new(faces, vec![back; faces.len()])
    }
}

impl Command for SetFaceBack {
    fn name(&self) -> &'static str {
        "set_face_back"
    }
    fn apply(&mut self, scene: &mut Scene, _cx: &mut Cx) -> Result<(), Fail> {
        self.swap.capture(scene);
        for (&f, b) in self.swap.faces.iter().zip(&self.backs) {
            let attrs = face_attrs_mut(scene, f);
            match b {
                None => pop_attr(attrs, "back"),
                Some(v) => pyval::set(attrs, "back", v.clone()),
            }
        }
        Ok(())
    }
    fn undo(&mut self, scene: &mut Scene, _cx: &mut Cx) {
        // The old value may be `True` (a two-sided face) and comes back so.
        self.swap.restore(scene, Value::clone);
    }
}

/// `SetFaceMaterialTagCommand`: the registry identity `attrs["mat"]`; a
/// given `material` is registered on first use and removed on undo only if
/// this very command added it.
pub struct SetFaceMaterialTag {
    swap: AttrSwap,
    name: Option<String>,
    material: Option<Value>,
    registered: bool,
}

impl SetFaceMaterialTag {
    /// `material`: the registry entry (`Material.to_dict()`).
    pub fn new(faces: &[FaceRef], name: Option<String>, material: Option<Value>) -> Self {
        SetFaceMaterialTag { swap: AttrSwap::new(faces, "mat"), name, material, registered: false }
    }
}

impl Command for SetFaceMaterialTag {
    fn name(&self) -> &'static str {
        "set_face_material_tag"
    }
    fn apply(&mut self, scene: &mut Scene, _cx: &mut Cx) -> Result<(), Fail> {
        self.swap.capture(scene);
        if let (Some(m), Some(n)) = (&self.material, &self.name) {
            if !n.is_empty() && !scene.materials.iter().any(|(k, _)| k == n) {
                scene.materials.push((n.clone(), m.clone()));
                self.registered = true;
            }
        }
        let value = self.name.clone().map(Value::Str);
        self.swap.put(scene, &value);
        Ok(())
    }
    fn undo(&mut self, scene: &mut Scene, _cx: &mut Cx) {
        self.swap.restore(scene, Value::clone);
        if self.registered {
            if let Some(n) = &self.name {
                scene.materials.retain(|(k, _)| k != n);
            }
            self.registered = false;
        }
    }
}

// ---- materials ------------------------------------------------------------------------------------

/// `SetMaterialFinishCommand`: how a named material answers light in a
/// render (`None` = guess it from the name again). Only the registry entry
/// changes.
pub struct SetMaterialFinish {
    name: String,
    finish: Option<Value>,
    old: Option<Value>,
}

impl SetMaterialFinish {
    pub fn new(name: &str, finish: Option<Value>) -> Self {
        SetMaterialFinish { name: name.to_string(), finish: finish.filter(truthy), old: None }
    }

    fn put(scene: &mut Scene, name: &str, finish: &Option<Value>) -> Option<Option<Value>> {
        let (_, mat) = scene.materials.iter_mut().find(|(k, _)| k == name)?;
        let Value::Object(items) = mat else { return None };
        let old = items.iter().rev().find(|(k, _)| k == "finish").map(|(_, v)| v.clone()).filter(truthy);
        // `Material.to_dict` writes the finish last, and only when set.
        items.retain(|(k, _)| k != "finish");
        if let Some(f) = finish {
            items.push(("finish".into(), f.clone()));
        }
        Some(old)
    }
}

impl Command for SetMaterialFinish {
    fn name(&self) -> &'static str {
        "set_material_finish"
    }
    fn apply(&mut self, scene: &mut Scene, _cx: &mut Cx) -> Result<(), Fail> {
        // An unknown name is a quiet no-op, as in Python.
        if let Some(old) = SetMaterialFinish::put(scene, &self.name, &self.finish) {
            self.old = old;
        }
        Ok(())
    }
    fn undo(&mut self, scene: &mut Scene, _cx: &mut Cx) {
        let old = self.old.clone();
        SetMaterialFinish::put(scene, &self.name, &old);
    }
}

const RESTAMP_KEYS: [&str; 3] = ["color", "texture", "opacity"];

/// The placements a registry sweep reaches: every group of the model with
/// its nested ones, depth first (paths).
fn all_placements(groups: &[Group]) -> Vec<Path> {
    crate::group::placement_paths(groups)
}

/// `RestampMaterialCommand`: edit a registry material and RESTAMP every face
/// that wears it, and every container painted whole with it.
pub struct RestampMaterial {
    name: String,
    /// `Material.to_dict()` of the new recipe, and its `face_attrs()`.
    new: Value,
    stamp: Vec<(String, Value)>,
    old_material: Option<Option<Value>>,
    old_faces: Option<Vec<(FaceRef, [Option<Value>; 3])>>,
    old_groups: Option<Vec<(Path, Option<Value>)>>,
}

impl RestampMaterial {
    pub fn new(name: &str, new_material: &Material) -> Result<Self, String> {
        let stamp = new_material.face_attrs().map_err(|e| format!("{e:?}"))?;
        let new = new_material.to_dict().map_err(|e| format!("{e:?}"))?;
        Ok(RestampMaterial { name: name.to_string(), new, stamp, old_material: None, old_faces: None, old_groups: None })
    }

    fn target_faces(&self, scene: &Scene) -> Vec<FaceRef> {
        // Sibling instances share one prototype mesh: the identity dedupe.
        let mut seen: HashSet<MeshId> = HashSet::new();
        let mut meshes = vec![MeshId(0)];
        for p in all_placements(&scene.groups) {
            if let Some(g) = group_at(&scene.groups, &p) {
                if seen.insert(g.mesh) {
                    meshes.push(g.mesh);
                }
            }
        }
        let mut out = Vec::new();
        for id in meshes {
            let m = scene.mesh(id);
            for &f in &m.faces {
                if matches!(get_attr(&m.face(f).attrs, "mat"), Some(Value::Str(s)) if s == self.name) {
                    out.push((id, f));
                }
            }
        }
        out
    }

    fn painted_groups(&self, scene: &Scene) -> Vec<Path> {
        all_placements(&scene.groups)
            .into_iter()
            .filter(|p| {
                group_at(&scene.groups, p).is_some_and(|g| matches!(&g.material, Some(m @ Value::Object(_)) if matches!(m.get("mat"), Some(Value::Str(s)) if *s == self.name)))
            })
            .collect()
    }
}

impl Command for RestampMaterial {
    fn name(&self) -> &'static str {
        "restamp_material"
    }
    fn apply(&mut self, scene: &mut Scene, _cx: &mut Cx) -> Result<(), Fail> {
        if self.old_faces.is_none() {
            self.old_material = Some(scene.materials.iter().find(|(k, _)| *k == self.name).map(|(_, v)| v.clone()));
            let faces = self.target_faces(scene);
            self.old_faces = Some(
                faces
                    .into_iter()
                    .map(|f| {
                        let attrs = &scene.mesh(f.0).face(f.1).attrs;
                        (f, [get_attr(attrs, RESTAMP_KEYS[0]), get_attr(attrs, RESTAMP_KEYS[1]), get_attr(attrs, RESTAMP_KEYS[2])])
                    })
                    .collect(),
            );
            self.old_groups = Some(self.painted_groups(scene).into_iter().map(|p| {
                let m = group_at(&scene.groups, &p).and_then(|g| g.material.clone());
                (p, m)
            }).collect());
        }
        match scene.materials.iter_mut().find(|(k, _)| *k == self.name) {
            Some(slot) => slot.1 = self.new.clone(),
            None => scene.materials.push((self.name.clone(), self.new.clone())),
        }
        for (f, _) in self.old_faces.clone().unwrap_or_default() {
            let attrs = face_attrs_mut(scene, f);
            for k in RESTAMP_KEYS {
                match self.stamp.iter().find(|(sk, _)| sk == k) {
                    Some((_, v)) => pyval::set(attrs, k, v.clone()),
                    None => pop_attr(attrs, k),
                }
            }
        }
        for (p, _) in self.old_groups.clone().unwrap_or_default() {
            if let Some(g) = group_at_mut(&mut scene.groups, &p) {
                g.material = Some(Value::Object(self.stamp.clone()));
            }
        }
        Ok(())
    }

    fn undo(&mut self, scene: &mut Scene, _cx: &mut Cx) {
        match &self.old_material {
            Some(Some(v)) => match scene.materials.iter_mut().find(|(k, _)| *k == self.name) {
                Some(slot) => slot.1 = v.clone(),
                None => scene.materials.push((self.name.clone(), v.clone())),
            },
            _ => scene.materials.retain(|(k, _)| *k != self.name),
        }
        for (f, old) in self.old_faces.clone().unwrap_or_default() {
            let attrs = face_attrs_mut(scene, f);
            for (k, o) in RESTAMP_KEYS.iter().zip(old.iter()) {
                restore_attr(attrs, k, o);
            }
        }
        for (p, old) in self.old_groups.clone().unwrap_or_default() {
            if let Some(g) = group_at_mut(&mut scene.groups, &p) {
                g.material = old;
            }
        }
    }
}

// ---- Assign Layer ------------------------------------------------------------------------------------------

fn layer_of(scene: &Scene, e: &Entity) -> Result<String, Fail> {
    let named = |s: Option<&str>| match s {
        Some(l) if !l.is_empty() => l.to_string(),
        _ => DEFAULT_LAYER.to_string(),
    };
    Ok(match e {
        Entity::Face(m, f) => match get_attr(&scene.mesh(*m).face(*f).attrs, "layer") {
            Some(Value::Str(l)) => named(Some(&l)),
            Some(v) if truthy(&v) => format!("{v:?}"),
            _ => DEFAULT_LAYER.to_string(),
        },
        Entity::Edge(m, e) => named(scene.mesh(*m).edge(*e).layer.as_deref()),
        Entity::Group(p) => named(group_at(&scene.groups, p).ok_or_else(gone)?.layer.as_deref()),
    })
}

fn assign_layer(scene: &mut Scene, e: &Entity, name: &str) -> Result<(), Fail> {
    let value = if name == DEFAULT_LAYER { None } else { Some(name.to_string()) };
    match e {
        Entity::Face(m, f) => {
            let attrs = face_attrs_mut(scene, (*m, *f));
            match value {
                None => pop_attr(attrs, "layer"),
                Some(v) => pyval::set(attrs, "layer", Value::Str(v)),
            }
        }
        Entity::Edge(m, e) => scene.meshes[m.0 as usize].edge_mut(*e).layer = value,
        Entity::Group(p) => group_at_mut(&mut scene.groups, p).ok_or_else(gone)?.layer = value,
    }
    Ok(())
}

fn layer_named(scene: &Scene, name: &str) -> Option<usize> {
    scene.layers.iter().position(|l| matches!(l.get("name"), Some(Value::Str(n)) if n == name))
}

/// `AssignLayerCommand`: move entities onto a layer (the Tag field); a layer
/// that does not exist yet is created.
pub struct AssignLayer {
    entities: Vec<Entity>,
    layer: String,
    prev: Vec<String>,
    created: bool,
}

impl AssignLayer {
    pub fn new(entities: Vec<Entity>, layer: &str) -> Self {
        AssignLayer { entities, layer: layer.to_string(), prev: Vec::new(), created: false }
    }
}

impl Command for AssignLayer {
    fn name(&self) -> &'static str {
        "assign_layer"
    }
    fn apply(&mut self, scene: &mut Scene, _cx: &mut Cx) -> Result<(), Fail> {
        if layer_named(scene, &self.layer).is_none() {
            scene.layers.push(ingetrazo_model::layers::Layer::new(&self.layer).to_dict());
            self.created = true;
        }
        self.prev = self.entities.iter().map(|e| layer_of(scene, e)).collect::<Result<_, _>>()?;
        for e in &self.entities {
            assign_layer(scene, e, &self.layer)?;
        }
        Ok(())
    }
    fn undo(&mut self, scene: &mut Scene, _cx: &mut Cx) {
        for (e, prev) in self.entities.iter().zip(&self.prev) {
            let _ = assign_layer(scene, e, prev);
        }
        if self.created {
            if let Some(i) = layer_named(scene, &self.layer) {
                scene.layers.remove(i);
            }
        }
    }
}

// ---- Purge Unused (core/purge.py) ---------------------------------------------------------------------------------

/// The groups of the document: the model's with their nested placements, then
/// the kept templates' (`iter_groups(_document_groups)`).
fn document_groups(scene: &Scene) -> Vec<&Group> {
    let mut out: Vec<&Group> = Vec::new();
    for g in scene.groups.iter().chain(scene.kept_components.iter()) {
        for (pg, _m) in iter_placements(g, None) {
            out.push(pg);
        }
    }
    out
}

/// `iter_meshes`: the loose mesh plus every group's mesh, each once.
fn document_meshes(scene: &Scene) -> Vec<MeshId> {
    let mut seen: HashSet<MeshId> = HashSet::new();
    let mut out = Vec::new();
    for id in std::iter::once(MeshId(0)).chain(document_groups(scene).into_iter().map(|g| g.mesh)) {
        if seen.insert(id) {
            out.push(id);
        }
    }
    out
}

fn layer_value(v: Option<&Value>) -> String {
    match v {
        Some(Value::Str(s)) if !s.is_empty() => s.clone(),
        _ => DEFAULT_LAYER.to_string(),
    }
}

/// `used_layers`: every layer name something in the document carries.
pub fn used_layers(scene: &Scene) -> HashSet<String> {
    let mut used: HashSet<String> = HashSet::new();
    used.insert(DEFAULT_LAYER.to_string());
    for id in document_meshes(scene) {
        let m = scene.mesh(id);
        for &f in &m.faces {
            used.insert(layer_value(m.face(f).attrs.iter().find(|(k, _)| k == "layer").map(|(_, v)| v)));
        }
        for &e in &m.edges {
            used.insert(m.edge(e).layer.clone().filter(|l| !l.is_empty()).unwrap_or_else(|| DEFAULT_LAYER.to_string()));
        }
    }
    for g in document_groups(scene) {
        if let Some(l) = g.layer.as_ref().filter(|l| !l.is_empty()) {
            used.insert(l.clone());
        }
    }
    // `layer_of(ent)`: the entity's layer, or the default. (An image plane
    // read without a layer key already carries the Images layer.)
    let of = |l: &Option<String>| l.clone().filter(|l| !l.is_empty()).unwrap_or_else(|| DEFAULT_LAYER.to_string());
    for d in scene.dimensions.iter() {
        used.insert(of(&d.layer));
    }
    for t in scene.text_labels.iter() {
        used.insert(of(&t.layer));
    }
    for ip in scene.image_planes.iter() {
        used.insert(of(&ip.layer));
    }
    for _ in &scene.section_planes {
        used.insert(DEFAULT_LAYER.to_string());
    }
    used
}

/// `layers_held_by_scenes`: layer names a saved view switches off.
fn layers_held_by_scenes(scene: &Scene) -> HashSet<String> {
    let mut held = HashSet::new();
    for view in &scene.extra.saved_views {
        held.extend(view.hidden_layers.iter().cloned());
    }
    held
}

/// `unused_layers`: the registry entries a purge would remove.
fn unused_layers(scene: &Scene) -> Vec<(usize, Value)> {
    let mut keep = used_layers(scene);
    keep.extend(layers_held_by_scenes(scene));
    scene
        .layers
        .iter()
        .enumerate()
        .filter(|(_, l)| match l.get("name") {
            Some(Value::Str(n)) => n != DEFAULT_LAYER && !keep.contains(n),
            _ => false,
        })
        .map(|(i, l)| (i, l.clone()))
        .collect()
}

/// `used_materials`: every registry name a face or a painted container wears.
pub fn used_materials(scene: &Scene) -> HashSet<String> {
    let mut used = HashSet::new();
    for id in document_meshes(scene) {
        let m = scene.mesh(id);
        for &f in &m.faces {
            if let Some(Value::Str(n)) = get_attr(&m.face(f).attrs, "mat") {
                if !n.is_empty() {
                    used.insert(n);
                }
            }
        }
    }
    for g in document_groups(scene) {
        if let Some(Value::Str(n)) = g.material.as_ref().and_then(|m| m.get("mat")) {
            if !n.is_empty() {
                used.insert(n.clone());
            }
        }
    }
    used
}

fn unused_materials(scene: &Scene) -> Vec<(String, Value)> {
    let used = used_materials(scene);
    scene.materials.iter().filter(|(n, _)| !used.contains(n)).cloned().collect()
}

/// `PurgeUnusedCommand`: what goes is decided ONCE, on the first `do`; the
/// layers come back at their original indices and the materials return to
/// the registry with their recipe intact.
pub struct PurgeUnused {
    do_layers: bool,
    do_materials: bool,
    layers: Option<Vec<(usize, Value)>>,
    materials: Option<Vec<(String, Value)>>,
}

impl PurgeUnused {
    pub fn new(layers: bool, materials: bool) -> Self {
        PurgeUnused { do_layers: layers, do_materials: materials, layers: None, materials: None }
    }

    fn survey(&mut self, scene: &Scene) {
        self.layers = Some(if self.do_layers { unused_layers(scene) } else { Vec::new() });
        self.materials = Some(if self.do_materials { unused_materials(scene) } else { Vec::new() });
    }

    /// What a run would remove: `(layers, materials)`.
    pub fn counts(&mut self, scene: &Scene) -> (usize, usize) {
        if self.layers.is_none() {
            self.survey(scene);
        }
        (self.layers.as_ref().map_or(0, Vec::len), self.materials.as_ref().map_or(0, Vec::len))
    }
}

impl Command for PurgeUnused {
    fn name(&self) -> &'static str {
        "purge_unused"
    }
    fn apply(&mut self, scene: &mut Scene, _cx: &mut Cx) -> Result<(), Fail> {
        if self.layers.is_none() {
            self.survey(scene);
        }
        for (_, ly) in self.layers.iter().flatten() {
            if let Some(i) = scene.layers.iter().position(|l| l == ly) {
                scene.layers.remove(i);
            }
        }
        for (name, _) in self.materials.iter().flatten() {
            scene.materials.retain(|(k, _)| k != name);
        }
        Ok(())
    }
    fn undo(&mut self, scene: &mut Scene, _cx: &mut Cx) {
        let mut layers: Vec<(usize, Value)> = self.layers.clone().unwrap_or_default();
        layers.sort_by_key(|p| p.0);
        for (index, ly) in layers {
            if !scene.layers.contains(&ly) {
                let at = index.min(scene.layers.len());
                scene.layers.insert(at, ly);
            }
        }
        for (name, material) in self.materials.iter().flatten() {
            match scene.materials.iter_mut().find(|(k, _)| k == name) {
                Some(slot) => slot.1 = material.clone(),
                None => scene.materials.push((name.clone(), material.clone())),
            }
        }
    }
}

// ---- Explode Curves / extension data ------------------------------------------------------------------------------------

/// `ExplodeCurvesCommand`: the curves the given edges belong to become plain
/// segments; the geometry stays. Undo gives every edge its id back.
pub struct ExplodeCurves {
    edges: Vec<(MeshId, EdgeId)>,
    ids: Vec<(EdgeId, i64)>,
    pub curves: usize,
}

impl ExplodeCurves {
    pub fn new(edges: &[(MeshId, EdgeId)]) -> Self {
        ExplodeCurves { edges: edges.to_vec(), ids: Vec::new(), curves: 0 }
    }
}

impl Command for ExplodeCurves {
    fn name(&self) -> &'static str {
        "explode_curves"
    }
    fn apply(&mut self, scene: &mut Scene, cx: &mut Cx) -> Result<(), Fail> {
        let ids: HashSet<i64> = self.edges.iter().filter_map(|&(m, e)| scene.mesh(m).edge(e).curve).collect();
        self.curves = ids.len();
        let mesh = &mut scene.meshes[cx.mesh.0 as usize];
        self.ids = mesh.edges.iter().filter_map(|&e| mesh.edge(e).curve.filter(|c| ids.contains(c)).map(|c| (e, c))).collect();
        for (e, _) in self.ids.clone() {
            mesh.edge_mut(e).curve = None;
        }
        Ok(())
    }
    fn undo(&mut self, scene: &mut Scene, cx: &mut Cx) {
        let mesh = &mut scene.meshes[cx.mesh.0 as usize];
        for &(e, c) in &self.ids {
            mesh.edge_mut(e).curve = Some(c);
        }
    }
}

/// `SetPluginDataCommand`: one extension's document data
/// (`scene.plugin_data[key]`); `None` removes the key. Values are copied
/// through JSON both ways in Python; a [`Value`] is already a copy.
pub struct SetPluginData {
    key: String,
    value: Option<Value>,
    had: bool,
    before: Option<Value>,
}

impl SetPluginData {
    pub fn new(key: &str, value: Option<Value>) -> Self {
        SetPluginData { key: key.to_string(), value: value.filter(|v| !matches!(v, Value::Null)), had: false, before: None }
    }

    fn items(scene: &mut Scene) -> &mut Vec<(String, Value)> {
        if !matches!(scene.extra.plugin_data, Some(Value::Object(_))) {
            scene.extra.plugin_data = Some(Value::Object(Vec::new()));
        }
        match &mut scene.extra.plugin_data {
            Some(Value::Object(items)) => items,
            _ => unreachable!(),
        }
    }
}

impl Command for SetPluginData {
    fn name(&self) -> &'static str {
        "set_plugin_data"
    }
    fn apply(&mut self, scene: &mut Scene, _cx: &mut Cx) -> Result<(), Fail> {
        let items = SetPluginData::items(scene);
        let cur = items.iter().find(|(k, _)| *k == self.key).map(|(_, v)| v.clone());
        self.had = cur.is_some();
        self.before = cur;
        match &self.value {
            None => items.retain(|(k, _)| *k != self.key),
            Some(v) => pyval::set(items, &self.key, v.clone()),
        }
        Ok(())
    }
    fn undo(&mut self, scene: &mut Scene, _cx: &mut Cx) {
        let items = SetPluginData::items(scene);
        match (&self.before, self.had) {
            (Some(v), true) => pyval::set(items, &self.key, v.clone()),
            _ => items.retain(|(k, _)| *k != self.key),
        }
    }
}
