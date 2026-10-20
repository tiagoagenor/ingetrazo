// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! Golden ops of the annotation commands (M3.9): dimensions, leader texts,
//! guides, section planes, image planes, georef data, saved scenes and
//! Straighten Model, decoded into `ingetrazo-ops` commands. The constructor
//! arguments by parameter name; the objects a command adds arrive as
//! `{"$obj": class, "data": ...}` and are rebuilt with their `from_dict`.

use ingetrazo_formats::pyjson::Value;
use ingetrazo_georef::datum::SceneDatum;
use ingetrazo_georef::geopath::GeoPath;
use ingetrazo_georef::points::GeoPoint;
use ingetrazo_math::V3;
use ingetrazo_model::dimension::Dimension;
use ingetrazo_model::guide::Guide;
use ingetrazo_model::image_plane::ImagePlane;
use ingetrazo_model::scene::Scene;
use ingetrazo_model::section::SectionPlane;
use ingetrazo_model::textlabel::TextLabel;
use ingetrazo_ops::annotate::{
    AddDimension, AddGuide, AddTextLabel, ChangeGuide, DeleteDimensions, DeleteGuides, DeleteScenes, DeleteTextLabels, EditDimensionText, EditTextLabel,
    MoveDimensions, MoveMode, MoveTextLabels,
};
use ingetrazo_ops::command::{Command, Fail};
use ingetrazo_ops::reference::{
    AddGeoPath, AddGeoPoints, AddImagePlane, DeleteGeoPaths, DeleteGeoPoints, DeleteImagePlanes, MoveGeoPathNode, MoveImagePlanes, RotateImagePlanes,
    ScaleImagePlanes, SetGeoPathSurface, SetImagePlaneOpacity, ToggleGeoPathClosed, TransformImagePlane,
};
use ingetrazo_ops::section_cmds::{
    DeleteSectionPlanes, MoveSectionPlanes, PlaceSectionPlane, RenameSectionPlane, ReverseSectionPlane, RotateSectionPlanes, SetActiveSection,
    SetSectionFill, SetSectionMode,
};
use ingetrazo_ops::straighten::StraightenModel;
use ingetrazo_ops::transform::Factor;

use crate::decode::{arr, f64_of, group_path, tagged, tuple, vec3, Res};

fn arg<'a>(args: &'a Value, name: &str) -> Res<&'a Value> {
    args.get(name).ok_or_else(|| format!("missing argument {name}"))
}

fn opt<'a>(args: &'a Value, name: &str) -> Option<&'a Value> {
    match args.get(name) {
        None | Some(Value::Null) => None,
        Some(v) => Some(v),
    }
}

fn fail(e: Fail) -> String {
    format!("{e:?}")
}

fn text(v: &Value) -> Res<String> {
    match v {
        Value::Str(s) => Ok(s.clone()),
        other => Err(format!("not text: {other:?}")),
    }
}

fn flag(v: &Value) -> bool {
    match v {
        Value::Bool(b) => *b,
        Value::Null => false,
        Value::Int(s) => s != "0",
        Value::Float(x) => *x != 0.0,
        Value::Str(s) => !s.is_empty(),
        Value::Array(a) => !a.is_empty(),
        Value::Object(o) => !o.is_empty(),
    }
}

/// `[x, y, z]` as a `to_dict` payload writes it.
fn point(v: &Value) -> Res<V3> {
    let a = arr(v)?;
    if a.len() != 3 {
        return Err(format!("not a point: {v:?}"));
    }
    Ok(V3(f64_of(&a[0])? as f32, f64_of(&a[1])? as f32, f64_of(&a[2])? as f32))
}

fn optional_text(v: Option<&Value>) -> Res<Option<String>> {
    match v {
        None | Some(Value::Null) => Ok(None),
        Some(x) => text(x).map(Some),
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

fn axes(args: &Value) -> Res<Option<[V3; 3]>> {
    let Some(v) = opt(args, "axes") else { return Ok(None) };
    let t = tuple(v)?;
    if t.len() != 3 {
        return Err("axes need 3 vectors".into());
    }
    Ok(Some([vec3(&t[0])?, vec3(&t[1])?, vec3(&t[2])?]))
}

/// `{"$ref": [list, i]}`: the identity of item `i` of the scene list `want`.
pub fn ref_id(scene: &Scene, v: &Value, want: &str) -> Res<u32> {
    let p = tagged(v, "$ref").ok_or_else(|| format!("not a $ref: {v:?}"))?;
    let (name, i) = match arr(p)? {
        [Value::Str(n), i] => (n.as_str(), f64_of(i)? as usize),
        _ => return Err(format!("bad $ref: {v:?}")),
    };
    if name != want {
        return Err(format!("$ref to {name}, wanted {want}"));
    }
    let id = match name {
        "dimensions" => scene.dimensions.id_at(i),
        "text_labels" => scene.text_labels.id_at(i),
        "guides" => scene.guides.id_at(i),
        "section_planes" => scene.section_planes.id_at(i),
        "image_planes" => scene.image_planes.id_at(i),
        "geo_paths" => scene.geo_paths.id_at(i),
        "geo_points" => scene.extra.geo_points.id_at(i),
        "saved_views" => scene.extra.saved_views.id_at(i),
        other => return Err(format!("$ref to {other}: not an annotation list")),
    };
    id.ok_or_else(|| format!("$ref {name}[{i}] is not in the scene"))
}

/// A list of `$ref`s into the scene list `want`.
pub fn ref_ids(scene: &Scene, v: &Value, want: &str) -> Res<Vec<u32>> {
    arr(v)?.iter().map(|x| ref_id(scene, x, want)).collect()
}

/// `{"$obj": class, "data": {...}}`: the class name and its data.
fn object(v: &Value) -> Res<(&str, &Value)> {
    match (v.get("$obj"), v.get("data")) {
        (Some(Value::Str(c)), Some(d)) => Ok((c.as_str(), d)),
        _ => Err(format!("not an $obj: {v:?}")),
    }
}

fn class_data<'a>(v: &'a Value, want: &str) -> Res<&'a Value> {
    let (class, data) = object(v)?;
    if class != want {
        return Err(format!("{class}, wanted {want}"));
    }
    Ok(data)
}

/// `{"$dict": [[k, v], ...]}`: the pairs.
fn dict_pairs(v: &Value) -> Res<Vec<(&Value, &Value)>> {
    let p = tagged(v, "$dict").ok_or_else(|| format!("not a $dict: {v:?}"))?;
    arr(p)?
        .iter()
        .map(|kv| match arr(kv)? {
            [k, x] => Ok((k, x)),
            _ => Err("a $dict entry is not a pair".to_string()),
        })
        .collect()
}

/// A value with its `$t` tuples read as lists (a colour tuple is a list in
/// the document).
fn untuple(v: &Value) -> Value {
    match v {
        Value::Array(a) => Value::Array(a.iter().map(untuple).collect()),
        Value::Object(items) => match tagged(v, "$t") {
            Some(Value::Array(a)) => Value::Array(a.iter().map(untuple).collect()),
            _ => Value::Object(items.iter().map(|(k, x)| (k.clone(), untuple(x))).collect()),
        },
        other => other.clone(),
    }
}

/// `Dimension(a, b, offset, layer=, text=, axis=)` from the recorder's data.
fn dimension(v: &Value) -> Res<Dimension> {
    let d = class_data(v, "Dimension")?;
    let get = |k: &str| d.get(k).ok_or_else(|| format!("Dimension without {k}"));
    Ok(Dimension::new(
        point(get("a")?)?,
        point(get("b")?)?,
        point(get("offset")?)?,
        optional_text(d.get("layer"))?,
        optional_text(d.get("text"))?,
        optional_text(d.get("axis"))?.as_deref(),
    ))
}

fn py(e: ingetrazo_model::pyconv::PyErr) -> String {
    format!("{}: {}", e.kind, e.msg)
}

fn py_georef(e: ingetrazo_georef::pyconv::PyErr) -> String {
    format!("{}: {}", e.kind, e.msg)
}

fn boxed<C: Command + 'static>(c: C) -> Res<Box<dyn Command>> {
    Ok(Box::new(c))
}

/// The command of `op`, or `None` when this module does not know the op.
pub fn build(op: &str, args: &Value, scene: &Scene) -> Option<Res<Box<dyn Command>>> {
    let h = scene.meshes[0].hypot;
    let r: Res<Box<dyn Command>> = match op {
        // ---- dimensions
        "add_dimension" => arg(args, "dimension").and_then(dimension).and_then(|d| boxed(AddDimension::new(d))),
        "delete_dimensions" => arg(args, "dimensions").and_then(|v| ref_ids(scene, v, "dimensions")).and_then(|ids| boxed(DeleteDimensions::new(ids))),
        "move_dimensions" => (|| {
            let (mut modes, mut free) = (Vec::new(), Vec::new());
            for (k, m) in dict_pairs(arg(args, "modes")?)? {
                let mode = if text(m)? == "line" { MoveMode::Line } else { MoveMode::Rigid };
                if k.get("$obj").is_some() {
                    free.push((dimension(k)?, mode));
                } else {
                    modes.push((ref_id(scene, k, "dimensions")?, mode));
                }
            }
            boxed(MoveDimensions::with_free(modes, free, vec3(arg(args, "delta")?)?))
        })(),
        "edit_dimension_text" => (|| {
            let id = ref_id(scene, arg(args, "dim")?, "dimensions")?;
            boxed(EditDimensionText::new(scene, id, optional_text(opt(args, "text"))?).map_err(fail)?)
        })(),
        // ---- leader texts
        "add_text_label" => (|| {
            let label = TextLabel::from_dict(class_data(arg(args, "label")?, "core.textlabel.TextLabel")?).map_err(py)?;
            boxed(AddTextLabel::new(label))
        })(),
        "move_text_labels" => (|| boxed(MoveTextLabels::new(ref_ids(scene, arg(args, "labels")?, "text_labels")?, vec3(arg(args, "delta")?)?)))(),
        "edit_text_label" => (|| {
            let id = ref_id(scene, arg(args, "label")?, "text_labels")?;
            boxed(EditTextLabel::new(scene, id, &text(arg(args, "text")?)?).map_err(fail)?)
        })(),
        "delete_text_labels" => (|| boxed(DeleteTextLabels::new(ref_ids(scene, arg(args, "labels")?, "text_labels")?)))(),
        // ---- guides
        "add_guide" => (|| {
            let guide = Guide::from_dict(class_data(arg(args, "guide")?, "core.guide.Guide")?, h).map_err(py)?;
            boxed(AddGuide::new(guide))
        })(),
        "change_guide" => (|| {
            let id = ref_id(scene, arg(args, "guide")?, "guides")?;
            boxed(ChangeGuide::new(scene, id, vec3(arg(args, "direction")?)?))
        })(),
        "delete_guides" => (|| boxed(DeleteGuides::new(ref_ids(scene, arg(args, "guides")?, "guides")?)))(),
        // ---- section planes
        "place_section_plane" => (|| {
            let mut uids = scene.section_uids;
            let plane = SectionPlane::from_dict(class_data(arg(args, "plane")?, "core.section.SectionPlane")?, &mut uids, h).map_err(py)?;
            boxed(PlaceSectionPlane::new(plane))
        })(),
        "delete_section_planes" => (|| boxed(DeleteSectionPlanes::new(ref_ids(scene, arg(args, "planes")?, "section_planes")?)))(),
        "reverse_section_plane" => (|| boxed(ReverseSectionPlane::new(ref_id(scene, arg(args, "plane")?, "section_planes")?)))(),
        "set_active_section" => (|| {
            let plane = match opt(args, "plane") {
                Some(p) => Some(ref_id(scene, p, "section_planes")?),
                None => None,
            };
            boxed(SetActiveSection::new(plane))
        })(),
        "set_section_mode" => (|| {
            let plane = ref_id(scene, arg(args, "plane")?, "section_planes")?;
            boxed(SetSectionMode::new(plane, flag(arg(args, "cut")?), flag(arg(args, "frame")?)))
        })(),
        "set_section_fill" => (|| {
            let plane = ref_id(scene, arg(args, "plane")?, "section_planes")?;
            boxed(SetSectionFill::new(plane, opt(args, "fill").map(untuple)))
        })(),
        "rename_section_plane" => (|| {
            let plane = ref_id(scene, arg(args, "plane")?, "section_planes")?;
            boxed(RenameSectionPlane::new(plane, &text(arg(args, "name")?)?, &text(arg(args, "symbol")?)?))
        })(),
        "move_section_planes" => (|| boxed(MoveSectionPlanes::new(ref_ids(scene, arg(args, "planes")?, "section_planes")?, vec3(arg(args, "delta")?)?)))(),
        "rotate_section_planes" => (|| {
            boxed(RotateSectionPlanes::new(
                ref_ids(scene, arg(args, "planes")?, "section_planes")?,
                vec3(arg(args, "centre")?)?,
                vec3(arg(args, "axis")?)?,
                f64_of(arg(args, "deg")?)?,
            ))
        })(),
        // ---- image planes
        "add_image_plane" => (|| {
            let image = ImagePlane::from_dict(class_data(arg(args, "image")?, "core.image_plane.ImagePlane")?).map_err(py)?;
            boxed(AddImagePlane::new(image))
        })(),
        "delete_image_planes" => (|| boxed(DeleteImagePlanes::new(ref_ids(scene, arg(args, "images")?, "image_planes")?)))(),
        "transform_image_plane" => (|| {
            let id = ref_id(scene, arg(args, "image")?, "image_planes")?;
            let v = |n: &str| -> Res<Option<V3>> { opt(args, n).map(vec3).transpose() };
            boxed(TransformImagePlane::new(id, v("origin")?, v("u")?, v("v")?))
        })(),
        "set_image_plane_opacity" => (|| {
            let id = ref_id(scene, arg(args, "image")?, "image_planes")?;
            boxed(SetImagePlaneOpacity::new(scene, id, f64_of(arg(args, "opacity")?)?).map_err(fail)?)
        })(),
        "move_image_planes" => (|| boxed(MoveImagePlanes::new(ref_ids(scene, arg(args, "images")?, "image_planes")?, vec3(arg(args, "delta")?)?)))(),
        "rotate_image_planes" => (|| {
            boxed(RotateImagePlanes::new(
                ref_ids(scene, arg(args, "images")?, "image_planes")?,
                vec3(arg(args, "centre")?)?,
                vec3(arg(args, "axis")?)?,
                f64_of(arg(args, "deg")?)?,
            ))
        })(),
        "scale_image_planes" => (|| {
            boxed(ScaleImagePlanes::new(
                ref_ids(scene, arg(args, "images")?, "image_planes")?,
                vec3(arg(args, "anchor")?)?,
                factor(arg(args, "factor")?)?,
                axes(args)?,
            ))
        })(),
        // ---- georef
        "add_geo_path" => (|| {
            let path = GeoPath::from_dict(class_data(arg(args, "path")?, "georef.geopath.GeoPath")?).map_err(py_georef)?;
            boxed(AddGeoPath::new(path))
        })(),
        "add_geo_points" => (|| {
            let mut points = Vec::new();
            for p in arr(arg(args, "points")?)? {
                points.push(GeoPoint::from_dict(class_data(p, "georef.points.GeoPoint")?).map_err(py_georef)?);
            }
            let datum = match opt(args, "datum") {
                Some(d) => Some(SceneDatum::from_dict(class_data(d, "georef.datum.SceneDatum")?)?.to_dict()),
                None => None,
            };
            boxed(AddGeoPoints::new(points, datum))
        })(),
        "delete_geo_points" => (|| boxed(DeleteGeoPoints::new(ref_ids(scene, arg(args, "points")?, "geo_points")?)))(),
        "delete_geo_paths" => (|| boxed(DeleteGeoPaths::new(ref_ids(scene, arg(args, "paths")?, "geo_paths")?)))(),
        "toggle_geo_path_closed" => (|| boxed(ToggleGeoPathClosed::new(ref_ids(scene, arg(args, "paths")?, "geo_paths")?)))(),
        "set_geo_path_surface" => (|| {
            let mode = optional_text(opt(args, "mode"))?;
            boxed(SetGeoPathSurface::new(ref_ids(scene, arg(args, "paths")?, "geo_paths")?, mode))
        })(),
        "move_geo_path_node" => (|| {
            let id = ref_id(scene, arg(args, "path")?, "geo_paths")?;
            boxed(MoveGeoPathNode::new(id, f64_of(arg(args, "index")?)? as i64, vec3(arg(args, "new_point")?)?))
        })(),
        // ---- saved scenes
        "delete_scenes" => (|| boxed(DeleteScenes::new(ref_ids(scene, arg(args, "views")?, "saved_views")?)))(),
        // ---- the whole scene
        "straighten_model" => (|| boxed(StraightenModel::new(&group_path(scene, arg(args, "group")?)?)))(),
        _ => return None,
    };
    Some(r)
}
