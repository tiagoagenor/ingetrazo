// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! Golden ops of the group, component, paint and purge commands (M3.10),
//! decoded into `ingetrazo-ops` commands.

use ingetrazo_formats::pyjson::Value;
use ingetrazo_model::materials::Material;
use ingetrazo_model::scene::{MeshId, Scene};
use ingetrazo_model::{EdgeId, FaceId};
use ingetrazo_ops::command::Command;
use ingetrazo_ops::edit_cmds::{
    AssignLayer, Entity, ExplodeCurves, Hide, PurgeUnused, RestampMaterial, SetFaceBack, SetFaceColor, SetFaceMaterialTag, SetFaceOpacity,
    SetFaceTexture, SetMaterialFinish, SetPluginData,
};
use ingetrazo_ops::group::Path;
use ingetrazo_ops::grouping::{
    DeleteGroup, ExplodeGroup, ExplodeView, GroupInContext, GroupRef, InContext, KeepComponent, Lock, MakeComponentOf, MakeGroup, MakeNestedGroup,
    MergeGroups, RenameGroup, ReparentGroups, ReplaceComponent, SetGroupMaterial,
};

use crate::decode::{arr, edge, f64_of, face, group_path, Res};

fn arg<'a>(args: &'a Value, name: &str) -> Res<&'a Value> {
    args.get(name).ok_or_else(|| format!("missing argument {name}"))
}

fn opt<'a>(args: &'a Value, name: &str) -> Option<&'a Value> {
    match args.get(name) {
        None | Some(Value::Null) => None,
        Some(v) => Some(v),
    }
}

fn flag(args: &Value, name: &str, default: bool) -> bool {
    match args.get(name) {
        Some(Value::Bool(b)) => *b,
        None | Some(Value::Null) => default,
        Some(Value::Int(s)) => s != "0",
        Some(_) => true,
    }
}

fn text(v: &Value) -> Res<String> {
    match v {
        Value::Str(s) => Ok(s.clone()),
        other => Err(format!("not text: {other:?}")),
    }
}

fn opt_text(args: &Value, name: &str) -> Res<Option<String>> {
    opt(args, name).map(text).transpose()
}

/// A JSON value with its `$t` tuples turned into lists (what a tuple is,
/// once Python writes it out).
fn plain(v: &Value) -> Res<Value> {
    match v {
        Value::Object(items) if items.len() == 1 && items[0].0 == "$t" => plain(&items[0].1),
        Value::Object(items) if items.iter().any(|(k, _)| k.starts_with('$')) => Err(format!("cannot decode {v:?}")),
        Value::Object(items) => Ok(Value::Object(items.iter().map(|(k, x)| plain(x).map(|x| (k.clone(), x))).collect::<Res<_>>()?)),
        Value::Array(a) => Ok(Value::Array(a.iter().map(plain).collect::<Res<_>>()?)),
        other => Ok(other.clone()),
    }
}

fn faces(scene: &Scene, v: &Value) -> Res<Vec<(MeshId, FaceId)>> {
    arr(v)?.iter().map(|f| face(scene, f)).collect()
}

fn edges(scene: &Scene, v: &Value) -> Res<Vec<(MeshId, EdgeId)>> {
    arr(v)?.iter().map(|e| edge(scene, e)).collect()
}

fn paths(scene: &Scene, v: &Value) -> Res<Vec<Path>> {
    arr(v)?.iter().map(|g| group_path(scene, g)).collect()
}

/// A group of `scene.groups` itself (a top-level one), as an index.
fn top_index(scene: &Scene, v: &Value, in_context: bool) -> Res<usize> {
    let p = group_path(scene, v)?;
    match p.as_slice() {
        [i] => Ok(*i),
        [.., i] if in_context => Ok(*i),
        _ => Err(format!("{p:?} is not a top-level group")),
    }
}

fn top_indices(scene: &Scene, v: &Value, in_context: bool) -> Res<Vec<usize>> {
    arr(v)?.iter().map(|g| top_index(scene, g, in_context)).collect()
}

fn entity(scene: &Scene, v: &Value) -> Res<Entity> {
    if let Ok(p) = group_path(scene, v) {
        return Ok(Entity::Group(p));
    }
    if let Ok((m, e)) = edge(scene, v) {
        return Ok(Entity::Edge(m, e));
    }
    face(scene, v).map(|(m, f)| Entity::Face(m, f)).map_err(|e| format!("not an entity ({e}): {v:?}"))
}

fn entities(scene: &Scene, v: &Value) -> Res<Vec<Entity>> {
    arr(v)?.iter().map(|e| entity(scene, e)).collect()
}

/// `{"$obj": "core.materials.Material", "data": {...}}`.
fn material(v: &Value) -> Res<Material> {
    match v {
        Value::Object(_) if matches!(v.get("$obj"), Some(Value::Str(c)) if c == "core.materials.Material") => {
            Material::from_dict(v.get("data").ok_or("Material without data")?).map_err(|e| format!("{e:?}"))
        }
        other => Err(format!("not a Material: {other:?}")),
    }
}

fn group_ref(scene: &Scene, v: &Value) -> Res<GroupRef> {
    if let Some(Value::Int(i)) = v.get("$kept") {
        let i: usize = i.parse().map_err(|_| "bad $kept index".to_string())?;
        if i >= scene.kept_components.len() {
            return Err(format!("$kept {i} is not in the scene"));
        }
        return Ok(GroupRef::Kept(i));
    }
    group_path(scene, v).map(GroupRef::Scene)
}

/// The inner command of a `group_in_context` step.
fn inner(scene: &Scene, v: &Value) -> Res<Box<dyn InContext>> {
    let c = v.get("$cmd").ok_or("group_in_context without a $cmd")?;
    let op = match c.get("op") {
        Some(Value::Str(s)) => s.as_str(),
        _ => return Err("inner command without op".into()),
    };
    let a = c.get("args").ok_or("inner command without args")?;
    match op {
        "make_group" => Ok(Box::new(make_group(scene, a, false)?)),
        "make_nested_group" => Ok(Box::new(MakeNestedGroup::new(
            scene,
            &faces(scene, arg(a, "faces")?)?,
            &edges(scene, arg(a, "edges")?)?,
            &top_indices(scene, arg(a, "groups")?, true)?,
            opt_text(a, "name")?,
        ))),
        "make_component_of" => Ok(Box::new(MakeComponentOf::new(
            scene,
            &faces(scene, arg(a, "faces")?)?,
            &edges(scene, arg(a, "edges")?)?,
            &top_indices(scene, arg(a, "groups")?, true)?,
            opt_text(a, "name")?,
        ))),
        "explode_group" => Ok(Box::new(ExplodeGroup::new(top_index(scene, arg(a, "group")?, true)?))),
        other => Err(format!("{other} cannot run in a group context")),
    }
}

fn make_group(scene: &Scene, a: &Value, _ctx: bool) -> Res<MakeGroup> {
    Ok(MakeGroup::new(scene, &faces(scene, arg(a, "faces")?)?, &edges(scene, arg(a, "edges")?)?, flag(a, "component", false), opt_text(a, "name")?))
}

/// The command of `op`, or `None` when this module does not know the op.
pub fn build(op: &str, args: &Value, scene: &Scene) -> Option<Res<Box<dyn Command>>> {
    fn b<C: Command + 'static>(c: C) -> Res<Box<dyn Command>> {
        Ok(Box::new(c))
    }
    let a = args;
    let r: Res<Box<dyn Command>> = match op {
        "make_group" => make_group(scene, a, false).and_then(b),
        "make_nested_group" => (|| {
            b(MakeNestedGroup::new(
                scene,
                &faces(scene, arg(a, "faces")?)?,
                &edges(scene, arg(a, "edges")?)?,
                &top_indices(scene, arg(a, "groups")?, false)?,
                opt_text(a, "name")?,
            ))
        })(),
        "make_component_of" => (|| {
            b(MakeComponentOf::new(
                scene,
                &faces(scene, arg(a, "faces")?)?,
                &edges(scene, arg(a, "edges")?)?,
                &top_indices(scene, arg(a, "groups")?, false)?,
                opt_text(a, "name")?,
            ))
        })(),
        "group_in_context" => (|| b(GroupInContext::new(inner(scene, arg(a, "inner")?)?, &group_path(scene, arg(a, "context")?)?)))(),
        "merge_groups" => (|| b(MergeGroups::new(&top_indices(scene, arg(a, "groups")?, false)?)))(),
        "explode_group" => (|| b(ExplodeGroup::new(top_index(scene, arg(a, "group")?, false)?)))(),
        "explode_view" => (|| {
            let mode = opt_text(a, "mode")?.unwrap_or_else(|| "outward".into());
            b(ExplodeView::new(&group_path(scene, arg(a, "container")?)?, f64_of(arg(a, "factor")?)?, &mode))
        })(),
        "reparent_groups" => (|| {
            let target = opt(a, "target").map(|t| group_path(scene, t)).transpose()?;
            b(ReparentGroups::new(&paths(scene, arg(a, "groups")?)?, target))
        })(),
        "delete_group" => (|| b(DeleteGroup::new(&group_path(scene, arg(a, "group")?)?)))(),
        "rename_group" => (|| b(RenameGroup::new(&group_path(scene, arg(a, "group")?)?, text(arg(a, "name")?)?)))(),
        "lock" => (|| b(Lock::new(&paths(scene, arg(a, "groups")?)?, flag(a, "locked", true))))(),
        "set_group_material" => (|| {
            let m = opt(a, "material").map(plain).transpose()?;
            b(SetGroupMaterial::new(&group_path(scene, arg(a, "group")?)?, m))
        })(),
        "replace_component" => (|| {
            let source = group_path(scene, arg(a, "source")?)?;
            b(ReplaceComponent::new(scene, &paths(scene, arg(a, "targets")?)?, &source)?)
        })(),
        "keep_component" => (|| b(KeepComponent::new(group_ref(scene, arg(a, "instance")?)?, flag(a, "keep", true))))(),
        "hide" => (|| b(Hide::new(entities(scene, arg(a, "entities")?)?, flag(a, "hidden", true))))(),
        "hide_edges" => (|| {
            let es = edges(scene, arg(a, "edges")?)?.into_iter().map(|(m, e)| Entity::Edge(m, e)).collect();
            b(Hide::edges(es, flag(a, "hidden", true)))
        })(),
        "assign_layer" => (|| b(AssignLayer::new(entities(scene, arg(a, "entities")?)?, &text(arg(a, "layer")?)?)))(),
        "set_face_color" => (|| {
            let color = match opt(a, "color") {
                None => None,
                Some(c) => match plain(c)? {
                    Value::Array(items) => Some(items),
                    other => return Err(format!("not a colour: {other:?}")),
                },
            };
            b(SetFaceColor::new(&faces(scene, arg(a, "faces")?)?, color))
        })(),
        "set_face_opacity" => (|| {
            let o = opt(a, "opacity").map(f64_of).transpose()?;
            b(SetFaceOpacity::new(&faces(scene, arg(a, "faces")?)?, o))
        })(),
        "set_face_texture" => (|| {
            let t = opt(a, "texture").map(plain).transpose()?;
            b(SetFaceTexture::new(&faces(scene, arg(a, "faces")?)?, t))
        })(),
        "set_face_back" => (|| {
            let fs = faces(scene, arg(a, "faces")?)?;
            match opt(a, "backs") {
                None => b(SetFaceBack::same(&fs, None)),
                Some(d @ Value::Object(_)) => b(SetFaceBack::same(&fs, Some(plain(d)?))),
                Some(l) => {
                    let backs = arr(l)?.iter().map(|x| if matches!(x, Value::Null) { Ok(None) } else { plain(x).map(Some) }).collect::<Res<Vec<_>>>()?;
                    b(SetFaceBack::new(&fs, backs))
                }
            }
        })(),
        "set_face_material_tag" => (|| {
            let reg = opt(a, "material").map(|m| material(m).and_then(|m| m.to_dict().map_err(|e| format!("{e:?}")))).transpose()?;
            b(SetFaceMaterialTag::new(&faces(scene, arg(a, "faces")?)?, opt_text(a, "name")?, reg))
        })(),
        "set_material_finish" => (|| b(SetMaterialFinish::new(&text(arg(a, "name")?)?, opt(a, "finish").cloned())))(),
        "restamp_material" => (|| {
            let m = material(arg(a, "new_material")?)?;
            b(RestampMaterial::new(&text(arg(a, "name")?)?, &m)?)
        })(),
        "purge_unused" => b(PurgeUnused::new(flag(a, "layers", true), flag(a, "materials", true))),
        "explode_curves" => (|| b(ExplodeCurves::new(&edges(scene, arg(a, "edges")?)?)))(),
        "set_plugin_data" => (|| b(SetPluginData::new(&text(arg(a, "key")?)?, opt(a, "value").map(plain).transpose()?)))(),
        _ => return None,
    };
    Some(r)
}
