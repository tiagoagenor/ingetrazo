// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! Argument values and selectors (golden/SCHEMA.md §4) decoded against a
//! Rust `Scene`: `$v`, `$m`, `$t`, and the selectors of live objects
//! (`$group`, `$face`, `$edge`, `$vertex`, `$mesh`).

use ingetrazo_formats::pyjson::Value;
use ingetrazo_math::{M4, V3};
use ingetrazo_model::scene::{MeshId, Scene};
use ingetrazo_model::{EdgeId, FaceId, VertexId};
use ingetrazo_ops::group::{group_at, num, Path};

pub type Res<T> = Result<T, String>;

/// The payload of a one-key `$tag` object.
fn tagged<'a>(v: &'a Value, tag: &str) -> Option<&'a Value> {
    match v {
        Value::Object(items) if items.len() == 1 && items[0].0 == tag => Some(&items[0].1),
        _ => None,
    }
}

pub fn f64_of(v: &Value) -> Res<f64> {
    num(v).ok_or_else(|| format!("not a number: {v:?}"))
}

pub fn arr(v: &Value) -> Res<&[Value]> {
    match v {
        Value::Array(a) => Ok(a),
        _ => Err(format!("not a list: {v:?}")),
    }
}

fn triple(v: &Value) -> Res<[f64; 3]> {
    let a = arr(v)?;
    if a.len() != 3 {
        return Err(format!("not a point: {v:?}"));
    }
    Ok([f64_of(&a[0])?, f64_of(&a[1])?, f64_of(&a[2])?])
}

/// `{"$v": [x, y, z]}`: a QVector3D (float32).
pub fn vec3(v: &Value) -> Res<V3> {
    let p = tagged(v, "$v").ok_or_else(|| format!("not a $v: {v:?}"))?;
    let t = triple(p)?;
    Ok(V3(t[0] as f32, t[1] as f32, t[2] as f32))
}

pub fn vec3s(v: &Value) -> Res<Vec<V3>> {
    arr(v)?.iter().map(vec3).collect()
}

/// `{"$m": [16 numbers]}`: `QMatrix4x4(*x).transposed()`, i.e. the matrix
/// whose `data()` is the list — General, as a matrix read from a file is.
pub fn mat4(v: &Value) -> Res<M4> {
    let p = tagged(v, "$m").ok_or_else(|| format!("not a $m: {v:?}"))?;
    let a = arr(p)?;
    if a.len() != 16 {
        return Err("a $m needs 16 numbers".into());
    }
    let mut d = [0.0f32; 16];
    for (i, x) in a.iter().enumerate() {
        d[i] = f64_of(x)? as f32;
    }
    Ok(M4::from_data(d))
}

/// `{"$t": [...]}`: the items of a tuple.
pub fn tuple(v: &Value) -> Res<&[Value]> {
    match tagged(v, "$t") {
        Some(p) => arr(p),
        None => Err(format!("not a $t: {v:?}")),
    }
}

/// `{"$group": [i, j, …]}`: an index path from `scene.groups` through `children`.
pub fn group_path(scene: &Scene, v: &Value) -> Res<Path> {
    let p = tagged(v, "$group").ok_or_else(|| format!("not a $group: {v:?}"))?;
    let path: Path = arr(p)?.iter().map(|x| f64_of(x).map(|n| n as usize)).collect::<Res<_>>()?;
    if group_at(&scene.groups, &path).is_none() {
        return Err(format!("$group {path:?} is not in the scene"));
    }
    Ok(path)
}

/// The mesh named by a selector's `in`: `null` is the loose mesh, a list the
/// mesh of the group at that path.
fn mesh_at(scene: &Scene, v: Option<&Value>) -> Res<MeshId> {
    match v {
        None | Some(Value::Null) => Ok(MeshId(0)),
        Some(p) => {
            let path: Path = arr(p)?.iter().map(|x| f64_of(x).map(|n| n as usize)).collect::<Res<_>>()?;
            group_at(&scene.groups, &path).map(|g| g.mesh).ok_or_else(|| format!("no group at {path:?}"))
        }
    }
}

/// Python float equality on float32 values (`-0.0 == 0.0`).
fn same(p: V3, t: [f64; 3]) -> bool {
    (p.0 as f64) == t[0] && (p.1 as f64) == t[1] && (p.2 as f64) == t[2]
}

fn points(v: &Value) -> Res<Vec<[f64; 3]>> {
    arr(v)?.iter().map(triple).collect()
}

/// `{"$face": {"loop": [...], "in": path|null}}`: the face of that mesh
/// whose outer loop has exactly that vertex SET. Returns the mesh too.
pub fn face(scene: &Scene, v: &Value) -> Res<(MeshId, FaceId)> {
    let p = tagged(v, "$face").ok_or_else(|| format!("not a $face: {v:?}"))?;
    let id = mesh_at(scene, p.get("in"))?;
    let want = points(p.get("loop").ok_or("$face without loop")?)?;
    let mesh = scene.mesh(id);
    let hits: Vec<FaceId> = mesh
        .faces
        .iter()
        .copied()
        .filter(|&f| {
            let have: Vec<V3> = mesh.face(f).outer.iter().map(|&x| mesh.position(x)).collect();
            have.iter().all(|&q| want.iter().any(|&t| same(q, t))) && want.iter().all(|&t| have.iter().any(|&q| same(q, t)))
        })
        .collect();
    match hits.as_slice() {
        [f] => Ok((id, *f)),
        _ => Err(format!("$face matched {} faces", hits.len())),
    }
}

/// `{"$edge": {"ends": [a, b], "in": …}}`.
pub fn edge(scene: &Scene, v: &Value) -> Res<(MeshId, EdgeId)> {
    let p = tagged(v, "$edge").ok_or_else(|| format!("not an $edge: {v:?}"))?;
    let id = mesh_at(scene, p.get("in"))?;
    let want = points(p.get("ends").ok_or("$edge without ends")?)?;
    let mesh = scene.mesh(id);
    let hits: Vec<EdgeId> = mesh
        .edges
        .iter()
        .copied()
        .filter(|&e| {
            let ed = mesh.edge(e);
            let (a, b) = (mesh.position(ed.v0), mesh.position(ed.v1));
            want.len() == 2 && ((same(a, want[0]) && same(b, want[1])) || (same(a, want[1]) && same(b, want[0])))
        })
        .collect();
    match hits.as_slice() {
        [e] => Ok((id, *e)),
        _ => Err(format!("$edge matched {} edges", hits.len())),
    }
}

/// `{"$vertex": {"at": [x,y,z], "in": …}}`.
pub fn vertex(scene: &Scene, v: &Value) -> Res<(MeshId, VertexId)> {
    let p = tagged(v, "$vertex").ok_or_else(|| format!("not a $vertex: {v:?}"))?;
    let id = mesh_at(scene, p.get("in"))?;
    let at = triple(p.get("at").ok_or("$vertex without at")?)?;
    let mesh = scene.mesh(id);
    let hits: Vec<VertexId> = mesh.vertices.iter().copied().filter(|&x| same(mesh.position(x), at)).collect();
    match hits.as_slice() {
        [x] => Ok((id, *x)),
        _ => Err(format!("$vertex matched {} vertices", hits.len())),
    }
}
