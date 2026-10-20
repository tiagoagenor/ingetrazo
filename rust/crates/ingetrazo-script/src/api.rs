// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! The functions a script can call. Everything here either reads the scene
//! or builds an `ingetrazo-ops` command and executes it through the
//! history ([`crate::exec`]); nothing writes the model directly.
//!
//! # Reference
//!
//! Numbers are Rhai integers or floats; `Vec3` is a point or a vector (its
//! components are `f32`, as the model's). Anywhere a point is expected a
//! three-number array `[x, y, z]` works too. "Entities" means one handle or
//! an array of handles (`Face`, `Edge`, `Group`).
//!
//! ## Points
//! `vec3(x, y, z)`, `v.x v.y v.z` (readable and writable), `a + b`, `a - b`,
//! `-a`, `a * k`, `k * a`, `a / k`, `a == b`, `dot(a, b)`, `cross(a, b)`,
//! `v.length()`, `v.normalized()`, `distance(a, b)`.
//!
//! ## Reading the model (the active mesh)
//! * `faces()`, `edges()` -> arrays of handles; `vertices()` -> array of `Vec3`.
//! * `face_count()`, `edge_count()`, `vertex_count()`.
//! * Face: `f.points()`, `f.normal()`, `f.area()`, `f.layer`, `f.color`
//!   (array or `()`), `f.opacity`, `f.material` (name or `()`), `f.hidden`,
//!   `f.attr("key")`, `f.alive`.
//! * Edge: `e.a`, `e.b`, `e.length()`, `e.soft`, `e.hidden`, `e.layer`, `e.alive`.
//! * `groups()` -> the top-level groups and components; Group: `g.name`,
//!   `g.path`, `g.layer`, `g.hidden`, `g.locked`, `g.is_component`,
//!   `g.children()`, `g.faces()`, `g.edges()`, `g.alive`; `find_group("name")`
//!   (depth-first, `()` if none).
//! * `layers()`, `materials()`, `material("name")` (a map or `()`),
//!   `dimensions()`, `text_labels()`, `guides()`, `section_planes()` -> arrays
//!   of names / maps.
//! * `selection()` -> the handles the host set as the UI selection.
//!
//! ## Changing the model (each call one command; the run is one undo step)
//! * Drawing, through the edits planner (it welds, splits and detects faces
//!   as the tools do): `add_edge(a, b)`, `add_polyline(points)`,
//!   `add_polygon(points)` (closed; the face inside is detected).
//! * Explicit faces: `add_face(points)`, `add_face(points, attrs_map)` where
//!   `attrs_map` is stamped on the face (`#{color: [r, g, b], layer: "Muros"}`,
//!   colours are floats 0-1, the Paint idiom). Returns the face handle.
//! * `erase(entities)`, `flip_faces(faces)`.
//! * `translate(entities, delta)`, `rotate(entities, center, axis, degrees)`,
//!   `scale(entities, center, factor)` (a number, or a `Vec3` per axis).
//!   Faces and edges move their vertices; groups move as a whole.
//! * `make_group(entities)`, `make_group(entities, name)`,
//!   `make_component(entities, name)` -> the new group handle (an entity list
//!   of groups nests them); `rename_group(g, name)`, `delete_group(g)`,
//!   `explode_group(g)`, `lock_group(g, bool)`.
//! * `hide(entities)`, `unhide(entities)`, `assign_layer(entities, "name")`,
//!   `set_color(faces, r, g, b)`, `clear_color(faces)`, `set_opacity(faces, a)`,
//!   `set_material(faces, "name")` (a material that exists in the document).
//! * Annotations: `add_dimension(a, b, offset)`, `add_text_label(anchor, text)`.
//!
//! `print(x)` / `debug(x)` go to the run's output. A failing call (a stale
//! handle, a command that declines) stops the script with its message and
//! the run is rolled back.

use ingetrazo_formats::pyjson::Value;
use ingetrazo_math::V3;
use ingetrazo_model::dimension::Dimension;
use ingetrazo_model::mesh::Mesh;
use ingetrazo_model::scene::{MeshId, Scene};
use ingetrazo_model::textlabel::TextLabel;
use ingetrazo_ops::command::Command;
use ingetrazo_ops::draw::{AddFaceCommand, EraseSelectionCommand, FlipFacesCommand};
use ingetrazo_ops::edit_cmds::{AssignLayer, Entity, Hide, SetFaceColor, SetFaceMaterialTag, SetFaceOpacity};
use ingetrazo_ops::edits::build_add_edges;
use ingetrazo_ops::group::{group_at, Path};
use ingetrazo_ops::annotate::{AddDimension, AddTextLabel};
use ingetrazo_ops::grouping::{DeleteGroup, ExplodeGroup, Lock, MakeComponentOf, MakeGroup, MakeNestedGroup, RenameGroup};
use ingetrazo_ops::transform::{Factor, MoveGroup, MoveVertices, RotateGroup, RotateVertices, ScaleGroup, ScaleVertices};
use rhai::{Array, Dynamic, Engine, ImmutableString, Map};

use crate::convert::*;
use crate::{exec, exec_all, Shared};

fn get_attr<'a>(attrs: &'a [(String, Value)], key: &str) -> Option<&'a Value> {
    attrs.iter().find(|(k, _)| k == key).map(|(_, v)| v)
}

fn v3d(p: V3) -> Dynamic {
    Dynamic::from(p)
}

fn loop_points(m: &Mesh, ring: &[ingetrazo_model::VertexId]) -> Vec<V3> {
    ring.iter().map(|&v| m.position(v)).collect()
}

fn active(st: &crate::State) -> &Mesh {
    st.cx.mesh(&st.scene)
}

fn face_handles(mesh: MeshId, m: &Mesh) -> Array {
    m.faces.iter().map(|&id| Dynamic::from(FaceH { mesh, id })).collect()
}

fn edge_handles(mesh: MeshId, m: &Mesh) -> Array {
    m.edges.iter().map(|&id| Dynamic::from(EdgeH { mesh, id })).collect()
}

/// The vertex positions an entity list moves, and the group paths in it.
/// Faces and edges must belong to the active mesh.
fn gather(st: &crate::State, es: &[Entity]) -> Res<(Vec<V3>, Vec<Path>)> {
    let mut pts = Vec::new();
    let mut paths = Vec::new();
    for e in es {
        match e {
            Entity::Face(mid, f) => {
                if *mid != st.cx.mesh {
                    return fail("the face is not in the active mesh");
                }
                let m = st.scene.mesh(*mid);
                let face = m.face(*f);
                pts.extend(loop_points(m, &face.outer));
                for h in &face.holes {
                    pts.extend(loop_points(m, h));
                }
            }
            Entity::Edge(mid, ed) => {
                if *mid != st.cx.mesh {
                    return fail("the edge is not in the active mesh");
                }
                let m = st.scene.mesh(*mid);
                let edge = m.edge(*ed);
                pts.push(m.position(edge.v0));
                pts.push(m.position(edge.v1));
            }
            Entity::Group(p) => paths.push(p.clone()),
        }
    }
    Ok((pts, paths))
}

fn live(sh: &Shared, d: &Dynamic) -> Res<Vec<Entity>> {
    live_entities(&sh.borrow().scene, d)
}

fn faces_only(es: &[Entity]) -> Res<Vec<(MeshId, ingetrazo_model::FaceId)>> {
    es.iter()
        .map(|e| match e {
            Entity::Face(m, f) => Ok((*m, *f)),
            _ => fail("expected faces"),
        })
        .collect()
}

fn group_path(sh: &Shared, d: &Dynamic) -> Res<Path> {
    let g = d.clone().try_cast::<GroupH>().ok_or_else(|| fail::<()>("expected a group handle").unwrap_err())?;
    if group_at(&sh.borrow().scene.groups, &g.path).is_none() {
        return fail("stale handle: the group is no longer in the model");
    }
    Ok(g.path)
}

fn group_value<R>(sh: &Shared, g: &GroupH, f: impl FnOnce(&ingetrazo_model::scene::Group, &Scene) -> R) -> Res<R> {
    let st = sh.borrow();
    match group_at(&st.scene.groups, &g.path) {
        Some(grp) => Ok(f(grp, &st.scene)),
        None => fail("stale handle: the group is no longer in the model"),
    }
}

fn layer_name(attrs: &[(String, Value)]) -> String {
    match get_attr(attrs, "layer") {
        Some(Value::Str(s)) if !s.is_empty() => s.clone(),
        _ => ingetrazo_model::layers::DEFAULT_LAYER.to_string(),
    }
}

/// Register the whole API on `engine`, bound to the document in `sh`.
pub(crate) fn register(engine: &mut Engine, sh: &Shared) {
    register_points(engine, sh);
    register_handles(engine);
    register_reads(engine, sh);
    register_draw(engine, sh);
    register_edit(engine, sh);
    register_groups(engine, sh);
    register_annotations(engine, sh);
}

fn register_points(engine: &mut Engine, sh: &Shared) {
    engine.register_type_with_name::<V3>("Vec3");
    engine.register_fn("vec3", |x: Dynamic, y: Dynamic, z: Dynamic| -> Res<V3> { Ok(V3(num(&x)? as f32, num(&y)? as f32, num(&z)? as f32)) });
    engine.register_get_set("x", |v: &mut V3| v.0 as f64, |v: &mut V3, x: f64| v.0 = x as f32);
    engine.register_get_set("y", |v: &mut V3| v.1 as f64, |v: &mut V3, y: f64| v.1 = y as f32);
    engine.register_get_set("z", |v: &mut V3| v.2 as f64, |v: &mut V3, z: f64| v.2 = z as f32);
    engine.register_fn("+", |a: V3, b: V3| a.add(b));
    engine.register_fn("-", |a: V3, b: V3| a.sub(b));
    engine.register_fn("-", |a: V3| a.neg());
    engine.register_fn("*", |a: V3, k: f64| a.mul(k as f32));
    engine.register_fn("*", |k: f64, a: V3| a.mul(k as f32));
    engine.register_fn("*", |a: V3, k: i64| a.mul(k as f32));
    engine.register_fn("*", |k: i64, a: V3| a.mul(k as f32));
    engine.register_fn("/", |a: V3, k: f64| a.div(k as f32));
    engine.register_fn("==", |a: V3, b: V3| a == b);
    engine.register_fn("!=", |a: V3, b: V3| a != b);
    engine.register_fn("dot", |a: V3, b: V3| a.dot(b) as f64);
    engine.register_fn("cross", |a: V3, b: V3| a.cross(b));
    let s = sh.clone();
    engine.register_fn("length", move |a: V3| a.length(s.borrow().cx.hypot) as f64);
    let s = sh.clone();
    engine.register_fn("normalized", move |a: V3| a.normalized(s.borrow().cx.hypot));
    let s = sh.clone();
    engine.register_fn("distance", move |a: V3, b: V3| b.sub(a).length(s.borrow().cx.hypot) as f64);
    engine.register_fn("to_string", |a: &mut V3| format!("vec3({}, {}, {})", a.0, a.1, a.2));
    engine.register_fn("to_debug", |a: &mut V3| format!("vec3({}, {}, {})", a.0, a.1, a.2));
}

fn register_handles(engine: &mut Engine) {
    engine.register_type_with_name::<FaceH>("Face");
    engine.register_type_with_name::<EdgeH>("Edge");
    engine.register_type_with_name::<GroupH>("Group");
    engine.register_fn("==", |a: FaceH, b: FaceH| a == b);
    engine.register_fn("==", |a: EdgeH, b: EdgeH| a == b);
    engine.register_fn("==", |a: GroupH, b: GroupH| a == b);
    engine.register_fn("to_string", |g: &mut FaceH| format!("Face({}:{})", g.mesh.0, g.id.0));
    engine.register_fn("to_string", |g: &mut EdgeH| format!("Edge({}:{})", g.mesh.0, g.id.0));
    engine.register_fn("to_string", |g: &mut GroupH| format!("Group{:?}", g.path));
}

fn register_reads(engine: &mut Engine, sh: &Shared) {
    let s = sh.clone();
    engine.register_fn("faces", move || -> Array {
        let st = s.borrow();
        face_handles(st.cx.mesh, active(&st))
    });
    let s = sh.clone();
    engine.register_fn("edges", move || -> Array {
        let st = s.borrow();
        edge_handles(st.cx.mesh, active(&st))
    });
    let s = sh.clone();
    engine.register_fn("vertices", move || -> Array {
        let st = s.borrow();
        let m = active(&st);
        m.vertices.iter().map(|&v| v3d(m.position(v))).collect()
    });
    let s = sh.clone();
    engine.register_fn("face_count", move || active(&s.borrow()).faces.len() as i64);
    let s = sh.clone();
    engine.register_fn("edge_count", move || active(&s.borrow()).edges.len() as i64);
    let s = sh.clone();
    engine.register_fn("vertex_count", move || active(&s.borrow()).vertices.len() as i64);

    // ---- faces
    let s = sh.clone();
    engine.register_fn("points", move |f: FaceH| -> Res<Array> {
        let st = s.borrow();
        let e = Entity::Face(f.mesh, f.id);
        if !alive(&st.scene, &e) {
            return fail("stale handle: the face is no longer in the model");
        }
        let m = st.scene.mesh(f.mesh);
        Ok(loop_points(m, &m.face(f.id).outer).into_iter().map(v3d).collect())
    });
    let s = sh.clone();
    engine.register_fn("normal", move |f: FaceH| -> Res<V3> {
        let st = s.borrow();
        if !alive(&st.scene, &Entity::Face(f.mesh, f.id)) {
            return fail("stale handle: the face is no longer in the model");
        }
        Ok(st.scene.mesh(f.mesh).face_normal(f.id, st.cx.hypot))
    });
    let s = sh.clone();
    engine.register_fn("area", move |f: FaceH| -> Res<f64> {
        let st = s.borrow();
        if !alive(&st.scene, &Entity::Face(f.mesh, f.id)) {
            return fail("stale handle: the face is no longer in the model");
        }
        Ok(st.scene.mesh(f.mesh).face_area(f.id, st.cx.hypot))
    });
    let s = sh.clone();
    engine.register_fn("attr", move |f: FaceH, key: ImmutableString| -> Res<Dynamic> {
        let st = s.borrow();
        if !alive(&st.scene, &Entity::Face(f.mesh, f.id)) {
            return fail("stale handle: the face is no longer in the model");
        }
        Ok(get_attr(&st.scene.mesh(f.mesh).face(f.id).attrs, &key).map_or(Dynamic::UNIT, from_value))
    });
    let face_prop = |engine: &mut Engine, name: &'static str, sh: &Shared, key: &'static str| {
        let s = sh.clone();
        engine.register_get(name, move |f: &mut FaceH| -> Res<Dynamic> {
            let st = s.borrow();
            if !alive(&st.scene, &Entity::Face(f.mesh, f.id)) {
                return fail("stale handle: the face is no longer in the model");
            }
            Ok(get_attr(&st.scene.mesh(f.mesh).face(f.id).attrs, key).map_or(Dynamic::UNIT, from_value))
        });
    };
    face_prop(engine, "color", sh, "color");
    face_prop(engine, "opacity", sh, "opacity");
    face_prop(engine, "material", sh, "mat");
    let s = sh.clone();
    engine.register_get("layer", move |f: &mut FaceH| -> Res<String> {
        let st = s.borrow();
        if !alive(&st.scene, &Entity::Face(f.mesh, f.id)) {
            return fail("stale handle: the face is no longer in the model");
        }
        Ok(layer_name(&st.scene.mesh(f.mesh).face(f.id).attrs))
    });
    let s = sh.clone();
    engine.register_get("hidden", move |f: &mut FaceH| -> Res<bool> {
        let st = s.borrow();
        if !alive(&st.scene, &Entity::Face(f.mesh, f.id)) {
            return fail("stale handle: the face is no longer in the model");
        }
        Ok(get_attr(&st.scene.mesh(f.mesh).face(f.id).attrs, "hidden").is_some_and(ingetrazo_model::pyval::truthy))
    });
    let s = sh.clone();
    engine.register_get("alive", move |f: &mut FaceH| alive(&s.borrow().scene, &Entity::Face(f.mesh, f.id)));

    // ---- edges
    let end = |engine: &mut Engine, name: &'static str, sh: &Shared, first: bool| {
        let s = sh.clone();
        engine.register_get(name, move |e: &mut EdgeH| -> Res<V3> {
            let st = s.borrow();
            if !alive(&st.scene, &Entity::Edge(e.mesh, e.id)) {
                return fail("stale handle: the edge is no longer in the model");
            }
            let m = st.scene.mesh(e.mesh);
            let ed = m.edge(e.id);
            Ok(m.position(if first { ed.v0 } else { ed.v1 }))
        });
    };
    end(engine, "a", sh, true);
    end(engine, "b", sh, false);
    let s = sh.clone();
    engine.register_fn("length", move |e: EdgeH| -> Res<f64> {
        let st = s.borrow();
        if !alive(&st.scene, &Entity::Edge(e.mesh, e.id)) {
            return fail("stale handle: the edge is no longer in the model");
        }
        let m = st.scene.mesh(e.mesh);
        let ed = m.edge(e.id);
        Ok(m.position(ed.v1).sub(m.position(ed.v0)).length(st.cx.hypot) as f64)
    });
    let s = sh.clone();
    engine.register_get("soft", move |e: &mut EdgeH| -> Res<bool> {
        let st = s.borrow();
        if !alive(&st.scene, &Entity::Edge(e.mesh, e.id)) {
            return fail("stale handle: the edge is no longer in the model");
        }
        Ok(st.scene.mesh(e.mesh).edge(e.id).soft)
    });
    let s = sh.clone();
    engine.register_get("hidden", move |e: &mut EdgeH| -> Res<bool> {
        let st = s.borrow();
        if !alive(&st.scene, &Entity::Edge(e.mesh, e.id)) {
            return fail("stale handle: the edge is no longer in the model");
        }
        Ok(st.scene.mesh(e.mesh).edge(e.id).hidden)
    });
    let s = sh.clone();
    engine.register_get("layer", move |e: &mut EdgeH| -> Res<String> {
        let st = s.borrow();
        if !alive(&st.scene, &Entity::Edge(e.mesh, e.id)) {
            return fail("stale handle: the edge is no longer in the model");
        }
        Ok(st.scene.mesh(e.mesh).edge(e.id).layer.clone().filter(|l| !l.is_empty()).unwrap_or_else(|| ingetrazo_model::layers::DEFAULT_LAYER.to_string()))
    });
    let s = sh.clone();
    engine.register_get("alive", move |e: &mut EdgeH| alive(&s.borrow().scene, &Entity::Edge(e.mesh, e.id)));

    // ---- groups
    let s = sh.clone();
    engine.register_fn("groups", move || -> Array { (0..s.borrow().scene.groups.len()).map(|i| Dynamic::from(GroupH { path: vec![i] })).collect() });
    let s = sh.clone();
    engine.register_fn("find_group", move |name: ImmutableString| -> Dynamic {
        fn walk(list: &[ingetrazo_model::scene::Group], base: &mut Vec<usize>, name: &str) -> Option<Vec<usize>> {
            for (i, g) in list.iter().enumerate() {
                base.push(i);
                if g.name == name {
                    return Some(base.clone());
                }
                if let Some(p) = walk(&g.children, base, name) {
                    return Some(p);
                }
                base.pop();
            }
            None
        }
        match walk(&s.borrow().scene.groups, &mut Vec::new(), &name) {
            Some(path) => Dynamic::from(GroupH { path }),
            None => Dynamic::UNIT,
        }
    });
    let s = sh.clone();
    engine.register_get("name", move |g: &mut GroupH| group_value(&s, g, |grp, _| grp.name.clone()));
    engine.register_get("path", |g: &mut GroupH| -> Array { g.path.iter().map(|&i| Dynamic::from(i as i64)).collect() });
    let s = sh.clone();
    engine.register_get("layer", move |g: &mut GroupH| group_value(&s, g, |grp, _| grp.layer.clone().filter(|l| !l.is_empty()).unwrap_or_else(|| ingetrazo_model::layers::DEFAULT_LAYER.to_string())));
    let s = sh.clone();
    engine.register_get("hidden", move |g: &mut GroupH| group_value(&s, g, |grp, _| grp.hidden));
    let s = sh.clone();
    engine.register_get("locked", move |g: &mut GroupH| group_value(&s, g, |grp, _| grp.locked));
    let s = sh.clone();
    engine.register_get("is_component", move |g: &mut GroupH| group_value(&s, g, |grp, _| grp.is_component()));
    let s = sh.clone();
    engine.register_get("alive", move |g: &mut GroupH| group_at(&s.borrow().scene.groups, &g.path).is_some());
    let s = sh.clone();
    engine.register_fn("children", move |g: GroupH| -> Res<Array> {
        group_value(&s, &g, |grp, _| {
            (0..grp.children.len())
                .map(|i| {
                    let mut p = g.path.clone();
                    p.push(i);
                    Dynamic::from(GroupH { path: p })
                })
                .collect()
        })
    });
    let s = sh.clone();
    engine.register_fn("faces", move |g: GroupH| -> Res<Array> { group_value(&s, &g, |grp, scene| face_handles(grp.mesh, scene.mesh(grp.mesh))) });
    let s = sh.clone();
    engine.register_fn("edges", move |g: GroupH| -> Res<Array> { group_value(&s, &g, |grp, scene| edge_handles(grp.mesh, scene.mesh(grp.mesh))) });

    // ---- document
    let s = sh.clone();
    engine.register_fn("layers", move || -> Array {
        s.borrow().scene.layers.iter().filter_map(|l| l.get("name")).map(from_value).collect()
    });
    let s = sh.clone();
    engine.register_fn("materials", move || -> Array { s.borrow().scene.materials.iter().map(|(n, _)| Dynamic::from(n.clone())).collect() });
    let s = sh.clone();
    engine.register_fn("material", move |name: ImmutableString| -> Dynamic {
        s.borrow().scene.materials.iter().find(|(n, _)| n.as_str() == name.as_str()).map_or(Dynamic::UNIT, |(_, v)| from_value(v))
    });
    let s = sh.clone();
    engine.register_fn("dimensions", move || -> Array {
        let st = s.borrow();
        st.scene
            .dimensions
            .iter()
            .map(|d| {
                let mut m = Map::new();
                m.insert("a".into(), v3d(d.a));
                m.insert("b".into(), v3d(d.b));
                m.insert("offset".into(), v3d(d.offset));
                m.insert("length".into(), Dynamic::from(d.value(st.cx.hypot)));
                m.insert("text".into(), d.text.clone().map_or(Dynamic::UNIT, Dynamic::from));
                Dynamic::from_map(m)
            })
            .collect()
    });
    let s = sh.clone();
    engine.register_fn("text_labels", move || -> Array {
        s.borrow()
            .scene
            .text_labels
            .iter()
            .map(|t| {
                let mut m = Map::new();
                m.insert("anchor".into(), v3d(t.anchor));
                m.insert("offset".into(), v3d(t.offset));
                m.insert("text".into(), Dynamic::from(t.text.clone()));
                m.insert("layer".into(), t.layer.clone().map_or(Dynamic::UNIT, Dynamic::from));
                Dynamic::from_map(m)
            })
            .collect()
    });
    let s = sh.clone();
    engine.register_fn("guides", move || -> Array {
        s.borrow()
            .scene
            .guides
            .iter()
            .map(|g| {
                let mut m = Map::new();
                m.insert("point".into(), v3d(g.point));
                m.insert("direction".into(), g.direction.map_or(Dynamic::UNIT, v3d));
                Dynamic::from_map(m)
            })
            .collect()
    });
    let s = sh.clone();
    engine.register_fn("section_planes", move || -> Array {
        s.borrow()
            .scene
            .section_planes
            .iter()
            .map(|p| {
                let mut m = Map::new();
                m.insert("name".into(), Dynamic::from(p.name.clone()));
                m.insert("symbol".into(), Dynamic::from(p.symbol.clone()));
                Dynamic::from_map(m)
            })
            .collect()
    });
    let s = sh.clone();
    engine.register_fn("selection", move || -> Array {
        s.borrow()
            .selection
            .iter()
            .map(|e| match e {
                Entity::Face(mesh, id) => Dynamic::from(FaceH { mesh: *mesh, id: *id }),
                Entity::Edge(mesh, id) => Dynamic::from(EdgeH { mesh: *mesh, id: *id }),
                Entity::Group(p) => Dynamic::from(GroupH { path: p.clone() }),
            })
            .collect()
    });
}

fn register_draw(engine: &mut Engine, sh: &Shared) {
    let add_segments = |sh: &Shared, segs: Vec<(V3, V3)>| -> Res<()> {
        if segs.is_empty() {
            return Ok(());
        }
        let cmd = {
            let st = sh.borrow();
            build_add_edges(&st.scene, &st.cx, &segs, true, Vec::new(), None, false)
        };
        exec(sh, cmd)
    };
    let s = sh.clone();
    engine.register_fn("add_edge", move |a: Dynamic, b: Dynamic| -> Res<()> {
        let (a, b) = (point(&a)?, point(&b)?);
        if a == b {
            return fail("add_edge: the two ends are the same point");
        }
        add_segments(&s, vec![(a, b)])
    });
    let s = sh.clone();
    engine.register_fn("add_polyline", move |pts: Array| -> Res<()> {
        let p = points(&pts)?;
        add_segments(&s, p.windows(2).filter(|w| w[0] != w[1]).map(|w| (w[0], w[1])).collect())
    });
    let s = sh.clone();
    engine.register_fn("add_polygon", move |pts: Array| -> Res<()> {
        let p = points(&pts)?;
        if p.len() < 3 {
            return fail("add_polygon: needs at least three points");
        }
        let segs = (0..p.len()).map(|i| (p[i], p[(i + 1) % p.len()])).filter(|(a, b)| a != b).collect();
        add_segments(&s, segs)
    });

    let add_face = |sh: &Shared, pts: Array, attrs: Option<Vec<(String, Value)>>| -> Res<FaceH> {
        let p = points(&pts)?;
        if p.len() < 3 {
            return fail("add_face: needs at least three points");
        }
        let cmd = AddFaceCommand::new(p, true, None, attrs);
        let made = cmd.face.clone();
        exec(sh, Box::new(cmd))?;
        let mesh = sh.borrow().cx.mesh;
        match made.get() {
            Some(id) => Ok(FaceH { mesh, id }),
            None => fail("add_face: no face was made"),
        }
    };
    let s = sh.clone();
    engine.register_fn("add_face", move |pts: Array| add_face(&s, pts, None));
    let s = sh.clone();
    engine.register_fn("add_face", move |pts: Array, attrs: Map| {
        let a = attrs_of(&attrs)?;
        add_face(&s, pts, Some(a))
    });

    let s = sh.clone();
    engine.register_fn("erase", move |what: Dynamic| -> Res<()> {
        let es = live(&s, &what)?;
        let (ends, loops, groups) = {
            let st = s.borrow();
            let mut ends = Vec::new();
            let mut loops = Vec::new();
            let mut groups = Vec::new();
            for e in &es {
                match e {
                    Entity::Edge(mid, ed) => {
                        let m = st.scene.mesh(*mid);
                        ends.push((m.position(m.edge(*ed).v0), m.position(m.edge(*ed).v1)));
                    }
                    Entity::Face(mid, f) => {
                        let m = st.scene.mesh(*mid);
                        loops.push(loop_points(m, &m.face(*f).outer));
                    }
                    Entity::Group(p) => groups.push(p.clone()),
                }
            }
            (ends, loops, groups)
        };
        let mut cmds: Vec<Box<dyn Command>> = Vec::new();
        if !ends.is_empty() || !loops.is_empty() {
            cmds.push(Box::new(EraseSelectionCommand::new(ends, loops)));
        }
        // Deepest and last first, so a path stays valid while the rest go.
        let mut groups = groups;
        groups.sort();
        for p in groups.into_iter().rev() {
            cmds.push(Box::new(DeleteGroup::new(&p)));
        }
        exec_all(&s, cmds)
    });
    let s = sh.clone();
    engine.register_fn("flip_faces", move |what: Dynamic| -> Res<()> {
        let es = live(&s, &what)?;
        exec(&s, Box::new(FlipFacesCommand { faces: faces_only(&es)? }))
    });
}

fn register_edit(engine: &mut Engine, sh: &Shared) {
    let s = sh.clone();
    engine.register_fn("translate", move |what: Dynamic, delta: Dynamic| -> Res<()> {
        let delta = point(&delta)?;
        let es = live(&s, &what)?;
        let (pts, paths) = gather(&s.borrow(), &es)?;
        let mut cmds: Vec<Box<dyn Command>> = Vec::new();
        if !pts.is_empty() {
            cmds.push(Box::new(MoveVertices::new(&pts, delta)));
        }
        for p in &paths {
            cmds.push(Box::new(MoveGroup::new(p, delta)));
        }
        exec_all(&s, cmds)
    });
    let s = sh.clone();
    engine.register_fn("rotate", move |what: Dynamic, center: Dynamic, axis: Dynamic, degrees: Dynamic| -> Res<()> {
        let (center, axis, deg) = (point(&center)?, point(&axis)?, num(&degrees)?);
        if axis.is_null() {
            return fail("rotate: the axis is the zero vector");
        }
        let es = live(&s, &what)?;
        let (pts, paths) = gather(&s.borrow(), &es)?;
        let mut cmds: Vec<Box<dyn Command>> = Vec::new();
        if !pts.is_empty() {
            cmds.push(Box::new(RotateVertices::new(&pts, center, axis, deg)));
        }
        for p in &paths {
            cmds.push(Box::new(RotateGroup::new(p, center, axis, deg)));
        }
        exec_all(&s, cmds)
    });
    let scale = |sh: &Shared, what: Dynamic, center: Dynamic, factor: Factor| -> Res<()> {
        let center = point(&center)?;
        let es = live(sh, &what)?;
        let (pts, paths) = gather(&sh.borrow(), &es)?;
        let mut cmds: Vec<Box<dyn Command>> = Vec::new();
        if !pts.is_empty() {
            cmds.push(Box::new(ScaleVertices::new(&pts, center, factor, None)));
        }
        for p in &paths {
            cmds.push(Box::new(ScaleGroup::new(p, center, factor, None)));
        }
        exec_all(sh, cmds)
    };
    let s = sh.clone();
    engine.register_fn("scale", move |what: Dynamic, center: Dynamic, k: Dynamic| -> Res<()> {
        if let Some(v) = k.clone().try_cast::<V3>() {
            return scale(&s, what, center, Factor::Axes([v.0 as f64, v.1 as f64, v.2 as f64]));
        }
        let k = num(&k)?;
        if k == 0.0 {
            return fail("scale: the factor is zero");
        }
        scale(&s, what, center, Factor::Uniform(k))
    });

    let s = sh.clone();
    engine.register_fn("hide", move |what: Dynamic| -> Res<()> {
        let es = live(&s, &what)?;
        exec(&s, Box::new(Hide::new(es, true)))
    });
    let s = sh.clone();
    engine.register_fn("unhide", move |what: Dynamic| -> Res<()> {
        let es = live(&s, &what)?;
        exec(&s, Box::new(Hide::new(es, false)))
    });
    let s = sh.clone();
    engine.register_fn("assign_layer", move |what: Dynamic, layer: ImmutableString| -> Res<()> {
        if layer.is_empty() {
            return fail("assign_layer: the layer needs a name");
        }
        let es = live(&s, &what)?;
        exec(&s, Box::new(AssignLayer::new(es, &layer)))
    });
    let s = sh.clone();
    engine.register_fn("set_color", move |what: Dynamic, r: Dynamic, g: Dynamic, b: Dynamic| -> Res<()> {
        let es = live(&s, &what)?;
        let faces = faces_only(&es)?;
        let comp = |d: &Dynamic| -> Res<Value> {
            let x = num(d)?;
            if !(0.0..=1.0).contains(&x) {
                return fail("set_color: components are 0 to 1");
            }
            Ok(Value::Float(x))
        };
        exec(&s, Box::new(SetFaceColor::new(&faces, Some(vec![comp(&r)?, comp(&g)?, comp(&b)?]))))
    });
    let s = sh.clone();
    engine.register_fn("clear_color", move |what: Dynamic| -> Res<()> {
        let es = live(&s, &what)?;
        exec(&s, Box::new(SetFaceColor::new(&faces_only(&es)?, None)))
    });
    let s = sh.clone();
    engine.register_fn("set_opacity", move |what: Dynamic, a: Dynamic| -> Res<()> {
        let a = num(&a)?;
        if !(0.0..=1.0).contains(&a) {
            return fail("set_opacity: the opacity is 0 to 1");
        }
        let es = live(&s, &what)?;
        exec(&s, Box::new(SetFaceOpacity::new(&faces_only(&es)?, Some(a))))
    });
    let s = sh.clone();
    engine.register_fn("set_material", move |what: Dynamic, name: ImmutableString| -> Res<()> {
        let es = live(&s, &what)?;
        let faces = faces_only(&es)?;
        let mat = s.borrow().scene.materials.iter().find(|(n, _)| n.as_str() == name.as_str()).map(|(_, v)| v.clone());
        if mat.is_none() {
            return fail(format!("set_material: no material named {name}"));
        }
        exec(&s, Box::new(SetFaceMaterialTag::new(&faces, Some(name.to_string()), mat)))
    });
}

fn register_groups(engine: &mut Engine, sh: &Shared) {
    /// Group the entities; `component` makes an instance of a definition.
    fn make(sh: &Shared, what: &Dynamic, name: Option<String>, component: bool) -> Res<GroupH> {
        let es = live(sh, what)?;
        let mut faces = Vec::new();
        let mut edges = Vec::new();
        let mut groups = Vec::new();
        for e in &es {
            match e {
                Entity::Face(m, f) => faces.push((*m, *f)),
                Entity::Edge(m, ed) => edges.push((*m, *ed)),
                Entity::Group(p) if p.len() == 1 => groups.push(p[0]),
                Entity::Group(_) => return fail("make_group: only top-level groups can be nested"),
            }
        }
        if faces.is_empty() && edges.is_empty() && groups.is_empty() {
            return fail("make_group: nothing to group");
        }
        let cmd: Box<dyn Command> = {
            let st = sh.borrow();
            if groups.is_empty() {
                Box::new(MakeGroup::new(&st.scene, &faces, &edges, component, name))
            } else if component {
                Box::new(MakeComponentOf::new(&st.scene, &faces, &edges, &groups, name))
            } else {
                Box::new(MakeNestedGroup::new(&st.scene, &faces, &edges, &groups, name))
            }
        };
        exec(sh, cmd)?;
        let n = sh.borrow().scene.groups.len();
        Ok(GroupH { path: vec![n.saturating_sub(1)] })
    }
    let s = sh.clone();
    engine.register_fn("make_group", move |what: Dynamic| make(&s, &what, None, false));
    let s = sh.clone();
    engine.register_fn("make_group", move |what: Dynamic, name: ImmutableString| make(&s, &what, Some(name.to_string()), false));
    let s = sh.clone();
    engine.register_fn("make_component", move |what: Dynamic, name: ImmutableString| make(&s, &what, Some(name.to_string()), true));

    let s = sh.clone();
    engine.register_fn("rename_group", move |g: Dynamic, name: ImmutableString| -> Res<()> {
        let p = group_path(&s, &g)?;
        exec(&s, Box::new(RenameGroup::new(&p, name.to_string())))
    });
    let s = sh.clone();
    engine.register_fn("delete_group", move |g: Dynamic| -> Res<()> {
        let p = group_path(&s, &g)?;
        exec(&s, Box::new(DeleteGroup::new(&p)))
    });
    let s = sh.clone();
    engine.register_fn("explode_group", move |g: Dynamic| -> Res<()> {
        let p = group_path(&s, &g)?;
        if p.len() != 1 {
            return fail("explode_group: only top-level groups");
        }
        exec(&s, Box::new(ExplodeGroup::new(p[0])))
    });
    let s = sh.clone();
    engine.register_fn("lock_group", move |g: Dynamic, locked: bool| -> Res<()> {
        let p = group_path(&s, &g)?;
        exec(&s, Box::new(Lock::new(&[p], locked)))
    });
}

fn register_annotations(engine: &mut Engine, sh: &Shared) {
    let s = sh.clone();
    engine.register_fn("add_dimension", move |a: Dynamic, b: Dynamic, offset: Dynamic| -> Res<()> {
        let (a, b, offset) = (point(&a)?, point(&b)?, point(&offset)?);
        if a == b {
            return fail("add_dimension: the two ends are the same point");
        }
        exec(&s, Box::new(AddDimension::new(Dimension::new(a, b, offset, None, None, None))))
    });
    let s = sh.clone();
    engine.register_fn("add_text_label", move |anchor: Dynamic, text: ImmutableString| -> Res<()> {
        let anchor = point(&anchor)?;
        exec(&s, Box::new(AddTextLabel::new(TextLabel { anchor, offset: V3(0.0, 0.0, 0.0), text: text.to_string(), layer: None })))
    });
}
