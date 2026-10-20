// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! The group commands the golden corpus does not drive (insert, split into
//! pieces, kept components, reshare, nested delete, group in context): each
//! one undoes to the signature it started from and redoes to the signature
//! it reached. Python's side of the same behaviour is `core/history.py`.

use ingetrazo_golden::model_sig::scene_signature;
use ingetrazo_golden::sig::{signature_hash, Files};
use ingetrazo_math::{M4, V3};
use ingetrazo_model::scene::{Group, MeshId, Scene};
use ingetrazo_ops::command::{Command, Cx, History};
use ingetrazo_ops::edit_cmds::{AssignLayer, Entity, PurgeUnused, SetPluginData};
use ingetrazo_ops::group::group_at;
use ingetrazo_ops::grouping::{
    AddKeptComponents, DeleteGroup, GroupInContext, GroupRef, InsertGroup, KeepComponent, MakeGroup, MakeNestedGroup, ReshareInstance, SplitIntoPieces,
};

fn sig(scene: &Scene) -> String {
    signature_hash(&scene_signature(scene, &Files { zip: None }))
}

fn quad(scene: &mut Scene, x0: f32, z: f32) -> MeshId {
    let mut m = ingetrazo_model::mesh::Mesh::new();
    m.add_face(&[V3(x0, 0.0, z), V3(x0 + 1.0, 0.0, z), V3(x0 + 1.0, 1.0, z), V3(x0, 1.0, z)], &[]);
    scene.meshes.push(m);
    MeshId(scene.meshes.len() as u32 - 1)
}

fn component(scene: &mut Scene, name: &str, x0: f32) -> Group {
    let mesh = quad(scene, x0, 0.0);
    let mut g = Group::new(mesh, name.to_string());
    let mut m = M4::identity();
    m.translate(x0, 0.5, 0.0);
    m.rotate(20.0, 0.0, 0.0, 1.0);
    g.xform = Some(m);
    g
}

/// Run `cmd`, then undo and redo it: the signature goes back and forth.
fn round_trip(scene: &mut Scene, cmd: Box<dyn Command>) {
    let (mut h, mut cx) = (History::new(), Cx::default());
    let before = sig(scene);
    h.execute(scene, &mut cx, cmd);
    assert!(h.last_error.is_none() && h.last_rejected.is_none(), "{:?} {:?}", h.last_error, h.last_rejected);
    let after = sig(scene);
    assert_ne!(before, after, "the command changed nothing");
    assert!(h.undo(scene, &mut cx));
    assert_eq!(sig(scene), before, "undo");
    assert!(h.redo(scene, &mut cx));
    assert_eq!(sig(scene), after, "redo");
}

#[test]
fn insert_group_goes_to_the_model_or_into_the_open_group() {
    let mut scene = Scene::new();
    let parent = component(&mut scene, "Padre", 3.0);
    scene.groups.push(parent);
    let new = component(&mut scene, "Nuevo", 9.0);
    round_trip(&mut scene, Box::new(InsertGroup::new(new.clone(), None)));
    // Into the open group: Python's redo expresses the group in the
    // container's space AGAIN (undo does not take the first one back), so
    // only the undo is a round trip.
    let (mut h, mut cx) = (History::new(), Cx::default());
    let before = sig(&scene);
    h.execute(&mut scene, &mut cx, Box::new(InsertGroup::new(new, Some(vec![0]))));
    assert_ne!(sig(&scene), before);
    assert!(h.undo(&mut scene, &mut cx));
    assert_eq!(sig(&scene), before);
}

#[test]
fn split_into_pieces_replaces_the_contents() {
    let mut scene = Scene::new();
    let g = component(&mut scene, "Mueble", 0.0);
    scene.groups.push(g);
    let a = component(&mut scene, "Pieza 1", 0.0);
    let b = component(&mut scene, "Pieza 2", 4.0);
    round_trip(&mut scene, Box::new(SplitIntoPieces::new(&[0], vec![a, b])));
}

#[test]
fn kept_components_come_and_go() {
    // Kept templates are not part of the L2 signature: count them.
    let mut scene = Scene::new();
    let g = component(&mut scene, "Silla", 0.0);
    scene.groups.push(g);
    let (mut h, mut cx) = (History::new(), Cx::default());
    h.execute(&mut scene, &mut cx, Box::new(KeepComponent::new(GroupRef::Scene(vec![0]), true)));
    assert_eq!(scene.kept_components.len(), 1);
    assert_eq!(scene.kept_components[0].mesh, scene.groups[0].mesh);
    // Keeping what is already kept is a no-op, and so is its undo.
    h.execute(&mut scene, &mut cx, Box::new(KeepComponent::new(GroupRef::Scene(vec![0]), true)));
    assert_eq!(scene.kept_components.len(), 1);
    assert!(h.undo(&mut scene, &mut cx));
    assert_eq!(scene.kept_components.len(), 1);
    // Turn it off, and bring it back by undo.
    h.execute(&mut scene, &mut cx, Box::new(KeepComponent::new(GroupRef::Kept(0), false)));
    assert!(scene.kept_components.is_empty());
    assert!(h.undo(&mut scene, &mut cx));
    assert_eq!(scene.kept_components.len(), 1);
    assert!(h.undo(&mut scene, &mut cx)); // the first keep
    assert!(scene.kept_components.is_empty());
    assert!(h.redo(&mut scene, &mut cx));
    assert_eq!(scene.kept_components.len(), 1);
    let t = component(&mut scene, "Plantilla", 7.0);
    h.execute(&mut scene, &mut cx, Box::new(AddKeptComponents::new(vec![t])));
    assert_eq!(scene.kept_components.len(), 2);
    assert!(h.undo(&mut scene, &mut cx));
    assert_eq!(scene.kept_components.len(), 1);
}

#[test]
fn deleting_a_nested_group_puts_it_back_in_its_place() {
    let mut scene = Scene::new();
    let mut parent = component(&mut scene, "Padre", 0.0);
    let kids = vec![component(&mut scene, "Uno", 2.0), component(&mut scene, "Dos", 4.0), component(&mut scene, "Tres", 6.0)];
    parent.adopt(kids);
    scene.groups.push(parent);
    let name = |s: &Scene, i: usize| group_at(&s.groups, &[0, i]).map(|g| g.name.clone());
    let (mut h, mut cx) = (History::new(), Cx::default());
    let before = sig(&scene);
    h.execute(&mut scene, &mut cx, Box::new(DeleteGroup::new(&[0, 1])));
    assert_eq!(name(&scene, 1).as_deref(), Some("Tres"));
    assert!(h.undo(&mut scene, &mut cx));
    assert_eq!(name(&scene, 1).as_deref(), Some("Dos"));
    assert_eq!(sig(&scene), before);
}

#[test]
fn reshare_writes_the_edit_into_the_definition() {
    let mut scene = Scene::new();
    let g = component(&mut scene, "Silla", 0.0);
    let (proto, xform) = (g.mesh, g.xform.unwrap_or_default());
    scene.groups.push(g);
    // The session's edited copy: the prototype moved up a metre in the world.
    let mut edited = ingetrazo_model::mesh::Mesh::new();
    let world: Vec<V3> = [V3(0.0, 0.0, 1.0), V3(1.0, 0.0, 1.0), V3(1.0, 1.0, 1.0), V3(0.0, 1.0, 1.0)].iter().map(|&p| xform.map(p)).collect();
    edited.add_face(&world, &[]);
    scene.meshes.push(edited);
    let edited_id = MeshId(scene.meshes.len() as u32 - 1);
    round_trip(&mut scene, Box::new(ReshareInstance::new(&[0], proto, xform, edited_id)));
}

#[test]
fn group_in_context_makes_the_group_inside_and_undoes_cleanly() {
    let mut scene = Scene::new();
    let mut ctx = component(&mut scene, "Abierto", 0.0);
    ctx.adopt(vec![component(&mut scene, "Uno", 1.0), component(&mut scene, "Dos", 5.0)]);
    ctx.component = false;
    scene.groups.push(ctx);
    let inner = MakeNestedGroup::new(&scene, &[], &[], &[0, 1], Some("Par".into()));
    round_trip(&mut scene, Box::new(GroupInContext::new(Box::new(inner), &[0])));
    // The two parts now sit inside one «Par» group.
    assert_eq!(group_at(&scene.groups, &[0]).map(|g| g.children.len()), Some(1));
    assert_eq!(group_at(&scene.groups, &[0, 0]).map(|g| g.children.len()), Some(2));
    // The same, from loose geometry (nothing grouped yet).
    let face = {
        scene.meshes[0].add_face(&[V3(10.0, 10.0, 0.0), V3(12.0, 10.0, 0.0), V3(12.0, 12.0, 0.0), V3(10.0, 12.0, 0.0)], &[]);
        scene.meshes[0].faces[0]
    };
    // Named: an automatic «Group N» takes the next number on every redo.
    let make = MakeGroup::new(&scene, &[(MeshId(0), face)], &[], false, Some("Fijo".into()));
    round_trip(&mut scene, Box::new(GroupInContext::new(Box::new(make), &[0])));
}

#[test]
fn layers_plugin_data_and_purge_round_trip() {
    let mut scene = Scene::new();
    let g = component(&mut scene, "Silla", 0.0);
    scene.groups.push(g);
    round_trip(&mut scene, Box::new(AssignLayer::new(vec![Entity::Group(vec![0])], "Muros")));
    // «Muros» was made by the command: nothing wears it after the undo, so
    // a purge finds it; with the group on it, it stays.
    let mut cx = Cx::default();
    let mut purge = PurgeUnused::new(true, true);
    assert_eq!(purge.counts(&scene), (0, 0));
    let mut h = History::new();
    h.execute(&mut scene, &mut cx, Box::new(AssignLayer::new(vec![], "Vacia")));
    let mut purge = PurgeUnused::new(true, true);
    assert_eq!(purge.counts(&scene), (1, 0));
    round_trip(&mut scene, Box::new(purge));
    // Extension data is not part of the L2 signature: look at it directly.
    let (mut h, mut cx) = (History::new(), Cx::default());
    h.execute(&mut scene, &mut cx, Box::new(SetPluginData::new("niveles", Some(ingetrazo_formats::pyjson::Value::Bool(true)))));
    assert!(matches!(&scene.extra.plugin_data, Some(v) if v.get("niveles").is_some()));
    assert!(h.undo(&mut scene, &mut cx));
    assert!(matches!(&scene.extra.plugin_data, Some(v) if v.get("niveles").is_none()));
}
