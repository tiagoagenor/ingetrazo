// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! Golden ops of the transform commands (M3.5), decoded into
//! `ingetrazo-ops` commands: the constructor arguments by parameter name,
//! as `from_op` binds them.

use ingetrazo_formats::pyjson::Value;
use ingetrazo_math::V3;
use ingetrazo_model::scene::Scene;
use ingetrazo_model::FaceId;
use ingetrazo_ops::command::Command;
use ingetrazo_ops::group_cmds::{GroupToComponent, MakeUnique};
use ingetrazo_ops::transform::{
    ChangeAxes, Factor, FlipGroups, FlipVertices, MoveGroup, MoveVertices, RotateGroup, RotateVertices, ScaleGroup, ScaleVertices,
};

use crate::decode::{f64_of, face, group_path, mat4, tuple, vec3, vec3s, Res};

fn arg<'a>(args: &'a Value, name: &str) -> Res<&'a Value> {
    args.get(name).ok_or_else(|| format!("missing argument {name}"))
}

fn opt<'a>(args: &'a Value, name: &str) -> Option<&'a Value> {
    match args.get(name) {
        None | Some(Value::Null) => None,
        Some(v) => Some(v),
    }
}

/// A scale factor: a number, or a tuple of three.
fn factor(v: &Value) -> Res<Factor> {
    match v {
        Value::Object(_) => {
            let t = tuple(v)?;
            if t.len() != 3 {
                return Err("a factor tuple needs 3 numbers".into());
            }
            Ok(Factor::Axes([f64_of(&t[0])?, f64_of(&t[1])?, f64_of(&t[2])?]))
        }
        Value::Array(a) if a.len() == 3 => Ok(Factor::Axes([f64_of(&a[0])?, f64_of(&a[1])?, f64_of(&a[2])?])),
        other => Ok(Factor::Uniform(f64_of(other)?)),
    }
}

/// `axes`: `None`, or a tuple of three unit vectors.
fn axes(args: &Value) -> Res<Option<[V3; 3]>> {
    let Some(v) = opt(args, "axes") else { return Ok(None) };
    let t = tuple(v)?;
    if t.len() != 3 {
        return Err("axes need 3 vectors".into());
    }
    Ok(Some([vec3(&t[0])?, vec3(&t[1])?, vec3(&t[2])?]))
}

/// The command of `op`, or `None` when this module does not know the op.
pub fn build(op: &str, args: &Value, scene: &Scene) -> Option<Res<Box<dyn Command>>> {
    let r: Res<Box<dyn Command>> = match op {
        "move_vertices" => (|| Ok(Box::new(MoveVertices::new(&vec3s(arg(args, "positions")?)?, vec3(arg(args, "delta")?)?)) as Box<dyn Command>))(),
        "rotate_vertices" => (|| {
            Ok(Box::new(RotateVertices::new(
                &vec3s(arg(args, "positions")?)?,
                vec3(arg(args, "center")?)?,
                vec3(arg(args, "axis")?)?,
                f64_of(arg(args, "degrees")?)?,
            )) as Box<dyn Command>)
        })(),
        "scale_vertices" => (|| {
            Ok(Box::new(ScaleVertices::new(&vec3s(arg(args, "positions")?)?, vec3(arg(args, "center")?)?, factor(arg(args, "factor")?)?, axes(args)?))
                as Box<dyn Command>)
        })(),
        "flip_vertices" => (|| {
            let mut faces: Vec<FaceId> = Vec::new();
            if let Some(Value::Array(fs)) = opt(args, "faces") {
                for f in fs {
                    faces.push(face(scene, f)?.1);
                }
            }
            Ok(Box::new(FlipVertices::new(&vec3s(arg(args, "positions")?)?, vec3(arg(args, "centre")?)?, vec3(arg(args, "axis")?)?, &faces)) as Box<dyn Command>)
        })(),
        "move_group" => (|| Ok(Box::new(MoveGroup::new(&group_path(scene, arg(args, "group")?)?, vec3(arg(args, "delta")?)?)) as Box<dyn Command>))(),
        "rotate_group" => (|| {
            Ok(Box::new(RotateGroup::new(
                &group_path(scene, arg(args, "group")?)?,
                vec3(arg(args, "center")?)?,
                vec3(arg(args, "axis")?)?,
                f64_of(arg(args, "degrees")?)?,
            )) as Box<dyn Command>)
        })(),
        "scale_group" => (|| {
            Ok(Box::new(ScaleGroup::new(&group_path(scene, arg(args, "group")?)?, vec3(arg(args, "center")?)?, factor(arg(args, "factor")?)?, axes(args)?))
                as Box<dyn Command>)
        })(),
        "flip_groups" => (|| {
            let paths = match arg(args, "groups")? {
                Value::Array(gs) => gs.iter().map(|g| group_path(scene, g)).collect::<Res<Vec<_>>>()?,
                other => return Err(format!("groups is not a list: {other:?}")),
            };
            Ok(Box::new(FlipGroups::new(&paths, vec3(arg(args, "centre")?)?, vec3(arg(args, "axis")?)?)) as Box<dyn Command>)
        })(),
        "change_axes" => (|| Ok(Box::new(ChangeAxes::new(&group_path(scene, arg(args, "group")?)?, mat4(arg(args, "frame")?)?)) as Box<dyn Command>))(),
        "make_unique" => (|| {
            let rename = matches!(opt(args, "rename"), Some(Value::Bool(true)));
            Ok(Box::new(MakeUnique::new(&group_path(scene, arg(args, "group")?)?, rename)) as Box<dyn Command>)
        })(),
        "group_to_component" => (|| {
            let name = match opt(args, "name") {
                Some(Value::Str(n)) => Some(n.clone()),
                _ => None,
            };
            Ok(Box::new(GroupToComponent::new(&group_path(scene, arg(args, "group")?)?, name)) as Box<dyn Command>)
        })(),
        _ => return None,
    };
    Some(r)
}
