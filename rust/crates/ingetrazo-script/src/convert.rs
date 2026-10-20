// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! Rhai values to model values and back: numbers, points, handles and the
//! `attrs` maps a script hands to a face.

use ingetrazo_formats::pyjson::Value;
use ingetrazo_math::V3;
use ingetrazo_model::scene::{MeshId, Scene};
use ingetrazo_model::{EdgeId, FaceId};
use ingetrazo_ops::edit_cmds::Entity;
use rhai::{Array, Dynamic, EvalAltResult, ImmutableString, Map, Position};

pub type Res<T> = Result<T, Box<EvalAltResult>>;

/// A script-visible failure.
pub fn fail<T>(msg: impl Into<String>) -> Res<T> {
    Err(Box::new(EvalAltResult::ErrorRuntime(Dynamic::from(msg.into()), Position::NONE)))
}

/// A face by where it lives: its mesh and its id. It goes stale when the
/// face is erased; every function that takes one checks.
#[derive(Clone, Debug, PartialEq)]
pub struct FaceH {
    pub mesh: MeshId,
    pub id: FaceId,
}

/// An edge, as [`FaceH`].
#[derive(Clone, Debug, PartialEq)]
pub struct EdgeH {
    pub mesh: MeshId,
    pub id: EdgeId,
}

/// A group or component instance by its path in the group tree
/// (`[2, 0]` = the first child of the third top-level group).
#[derive(Clone, Debug, PartialEq)]
pub struct GroupH {
    pub path: Vec<usize>,
}

pub fn num(d: &Dynamic) -> Res<f64> {
    if let Ok(f) = d.as_float() {
        return Ok(f);
    }
    if let Ok(i) = d.as_int() {
        return Ok(i as f64);
    }
    fail(format!("expected a number, got {}", d.type_name()))
}

/// A point: a `Vec3`, or a three-number array.
pub fn point(d: &Dynamic) -> Res<V3> {
    if let Some(v) = d.clone().try_cast::<V3>() {
        return Ok(v);
    }
    if let Some(a) = d.clone().try_cast::<Array>() {
        if a.len() == 3 {
            return Ok(V3(num(&a[0])? as f32, num(&a[1])? as f32, num(&a[2])? as f32));
        }
    }
    fail(format!("expected a point (vec3 or [x, y, z]), got {}", d.type_name()))
}

pub fn points(a: &Array) -> Res<Vec<V3>> {
    a.iter().map(point).collect()
}

/// One entity handle.
pub fn entity(d: &Dynamic) -> Res<Entity> {
    if let Some(f) = d.clone().try_cast::<FaceH>() {
        return Ok(Entity::Face(f.mesh, f.id));
    }
    if let Some(e) = d.clone().try_cast::<EdgeH>() {
        return Ok(Entity::Edge(e.mesh, e.id));
    }
    if let Some(g) = d.clone().try_cast::<GroupH>() {
        return Ok(Entity::Group(g.path));
    }
    fail(format!("expected a face, edge or group handle, got {}", d.type_name()))
}

/// A handle, or an array of them.
pub fn entities(d: &Dynamic) -> Res<Vec<Entity>> {
    match d.clone().try_cast::<Array>() {
        Some(a) => a.iter().map(entity).collect(),
        None => Ok(vec![entity(d)?]),
    }
}

/// Is the entity still in the scene?
pub fn alive(scene: &Scene, e: &Entity) -> bool {
    match e {
        Entity::Face(m, f) => scene.meshes.get(m.0 as usize).is_some_and(|mesh| mesh.faces.contains(f)),
        Entity::Edge(m, e) => scene.meshes.get(m.0 as usize).is_some_and(|mesh| mesh.edges.contains(e)),
        Entity::Group(p) => ingetrazo_ops::group::group_at(&scene.groups, p).is_some(),
    }
}

/// Entities that must all be alive.
pub fn live_entities(scene: &Scene, d: &Dynamic) -> Res<Vec<Entity>> {
    let list = entities(d)?;
    if list.iter().any(|e| !alive(scene, e)) {
        return fail("stale handle: the entity is no longer in the model");
    }
    Ok(list)
}

/// A script value as the JSON-like value a face attribute holds.
pub fn to_value(d: &Dynamic) -> Res<Value> {
    if d.is_unit() {
        return Ok(Value::Null);
    }
    if let Ok(b) = d.as_bool() {
        return Ok(Value::Bool(b));
    }
    if let Ok(i) = d.as_int() {
        return Ok(Value::int(i));
    }
    if let Ok(f) = d.as_float() {
        return Ok(Value::Float(f));
    }
    if let Some(s) = d.clone().try_cast::<ImmutableString>() {
        return Ok(Value::Str(s.to_string()));
    }
    if let Some(a) = d.clone().try_cast::<Array>() {
        return Ok(Value::Array(a.iter().map(to_value).collect::<Res<_>>()?));
    }
    if let Some(m) = d.clone().try_cast::<Map>() {
        let mut items = Vec::new();
        for (k, v) in m.iter() {
            items.push((k.to_string(), to_value(v)?));
        }
        return Ok(Value::Object(items));
    }
    fail(format!("cannot store a {} in a face", d.type_name()))
}

/// A stored value as a script value.
pub fn from_value(v: &Value) -> Dynamic {
    match v {
        Value::Null => Dynamic::UNIT,
        Value::Bool(b) => Dynamic::from(*b),
        Value::Int(s) => s.parse::<i64>().map(Dynamic::from).unwrap_or_else(|_| Dynamic::from(s.clone())),
        Value::Float(f) => Dynamic::from(*f),
        Value::Str(s) => Dynamic::from(s.clone()),
        Value::Array(a) => Dynamic::from_array(a.iter().map(from_value).collect()),
        Value::Object(o) => {
            let mut m = Map::new();
            for (k, x) in o {
                m.insert(k.as_str().into(), from_value(x));
            }
            Dynamic::from_map(m)
        }
    }
}

/// `attrs` of a script map (insertion order is the map's key order).
pub fn attrs_of(m: &Map) -> Res<Vec<(String, Value)>> {
    let mut out = Vec::new();
    for (k, v) in m.iter() {
        out.push((k.to_string(), to_value(v)?));
    }
    Ok(out)
}
