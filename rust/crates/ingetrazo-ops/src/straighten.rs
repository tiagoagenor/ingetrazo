// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! `StraightenModelCommand` (`core/history.py`): put a model that was turned
//! and dragged onto its site back on its own axes, and turn the MAP under it
//! instead (a north angle).
//!
//! The inverse of the group's placement is applied to everything in the
//! scene — every group, the loose mesh, the annotations, guides, section
//! planes, image planes, geo data, saved views and the world anchors of the
//! sheets — so nothing moves relative to anything else, and the datum is
//! re-anchored at the group's origin with the north angle grown by its turn.
//!
//! Compositions stay the JSON the document brought, so their items are edited
//! as JSON (`a_world`, `b_world`, `cam_target`, `cam_yaw`); the terrain is
//! display-only and not modelled yet.

use ingetrazo_formats::pyjson::Value;
use ingetrazo_georef::datum::SceneDatum;
use ingetrazo_georef::geotiff::py_hypot2;
use ingetrazo_math::{M4, V3};
use ingetrazo_model::anchors::End;
use ingetrazo_model::mesh::State;
use ingetrazo_model::saved_views::PyNum;
use ingetrazo_model::scene::{MeshId, Scene};

use crate::command::{Command, Cx, Fail};
use crate::group::{group_at, remap_uvws, Path};

fn fail<T>(why: &str) -> Result<T, Fail> {
    Err(Fail::Failed(format!("ValueError: {why}")))
}

/// `placement_rotation_deg`: the turn about Z (degrees, counter-clockwise
/// seen from above) of a rigid placement; a placement that tilts or scales
/// has no north angle.
pub fn placement_rotation_deg(xform: &M4) -> Result<f64, Fail> {
    let (c0, c1, c2) = (xform.column(0), xform.column(1), xform.column(2));
    let f = |x: f32| x as f64;
    if f(c2[0]).abs() > 1e-6 || f(c2[1]).abs() > 1e-6 || (f(c2[2]) - 1.0).abs() > 1e-6 || f(c0[2]).abs() > 1e-6 || f(c1[2]).abs() > 1e-6 {
        return fail("the model is tilted (its Z axis is not vertical)");
    }
    let sx = py_hypot2(f(c0[0]), f(c0[1]));
    let sy = py_hypot2(f(c1[0]), f(c1[1]));
    if (sx - 1.0).abs() > 1e-4 || (sy - 1.0).abs() > 1e-4 {
        return fail("the model is scaled");
    }
    Ok(f(c0[1]).atan2(f(c0[0])).to_degrees())
}

pub struct StraightenModel {
    /// The placed model: a top-level group.
    path: Path,
    degrees: f64,
    placement: Option<M4>,
    old_datum: Option<Value>,
    new_datum: Option<Value>,
    /// `_snaps`: each classic mesh before and after.
    snaps: Vec<(MeshId, State, State)>,
}

impl StraightenModel {
    pub fn new(path: &[usize]) -> Self {
        StraightenModel { path: path.to_vec(), degrees: 0.0, placement: None, old_datum: None, new_datum: None, snaps: Vec::new() }
    }

    /// `_transform_mesh`: move the vertices the first time, snapshot both
    /// states, and replay the snapshots after that so undo/redo are exact.
    fn transform_mesh(&mut self, scene: &mut Scene, id: MeshId, m: &mut M4, redo: bool) {
        let mesh = &mut scene.meshes[id.0 as usize];
        if mesh.vertices.is_empty() {
            return;
        }
        if let Some(s) = self.snaps.iter().find(|s| s.0 == id) {
            mesh.restore_state(if redo { &s.2 } else { &s.1 });
            return;
        }
        let before = mesh.capture_state();
        for v in mesh.vertices.clone() {
            let p = mesh.position(v);
            mesh.move_vertex(v, m.map(p).sub(p));
        }
        remap_uvws(mesh, m);
        let after = mesh.capture_state();
        self.snaps.push((id, before, after));
    }

    /// `_apply`: move everything by the rigid `m` (`turn_deg` is its turn
    /// about Z, for the cameras).
    fn rigid(&mut self, scene: &mut Scene, cx: &Cx, m: &mut M4, turn_deg: f64, redo: bool) {
        for i in 0..scene.groups.len() {
            if let Some(x) = scene.groups[i].xform {
                scene.groups[i].xform = Some(m.mul(&x)); // instances compose
            } else {
                let mesh = scene.groups[i].mesh;
                self.transform_mesh(scene, mesh, m, redo);
            }
        }
        self.transform_mesh(scene, cx.mesh, m, redo);
        for i in 0..scene.dimensions.len() {
            let a = scene.dimension_end(i, End::A, cx.mesh);
            let b = scene.dimension_end(i, End::B, cx.mesh);
            let d = &mut scene.dimensions[i];
            let offset = m.map_vector(d.offset);
            d.a = m.map(a);
            d.b = m.map(b);
            d.offset = offset;
        }
        for t in scene.text_labels.iter_mut() {
            let (anchor, offset) = (m.map(t.anchor), m.map_vector(t.offset));
            t.anchor = anchor;
            t.offset = offset;
        }
        for g in scene.guides.iter_mut() {
            g.point = m.map(g.point);
            if let Some(d) = g.direction {
                g.direction = Some(m.map_vector(d));
            }
        }
        for sp in scene.section_planes.iter_mut() {
            let (point, normal) = (m.map(sp.point), m.map_vector(sp.normal));
            sp.point = point;
            sp.normal = normal;
        }
        for im in scene.image_planes.iter_mut() {
            let (o, u, v) = (m.map(im.origin), m.map_vector(im.u), m.map_vector(im.v));
            im.origin = o;
            im.u = u;
            im.v = v;
        }
        for gp in scene.geo_paths.iter_mut() {
            gp.points = gp.points.iter().map(|&q| m.map(q)).collect();
            gp.surface_tris = None;
        }
        for gpt in scene.extra.geo_points.iter_mut() {
            gpt.position = m.map(gpt.position);
        }
        let turn = turn_deg * (std::f64::consts::PI / 180.0);
        for sv in scene.extra.saved_views.iter_mut() {
            let num = |n: &PyNum| match n {
                PyNum::Int(s) => s.parse::<f64>().unwrap_or(0.0),
                PyNum::Float(x) => *x,
            };
            let q = m.map(V3(num(&sv.target[0]) as f32, num(&sv.target[1]) as f32, num(&sv.target[2]) as f32));
            sv.target = [PyNum::Float(q.0 as f64), PyNum::Float(q.1 as f64), PyNum::Float(q.2 as f64)];
            sv.yaw += turn;
        }
        for comp in scene.extra.compositions.iter_mut() {
            move_composition(comp, &*m, turn);
        }
    }
}

/// The attributes of `all_items()` that hold world points, in the item lists
/// of a composition's record.
const ITEM_LISTS: [&str; 14] = [
    "frames", "texts", "images", "scalebars", "nortes", "leyendas", "shapes", "cotas", "cotas_ang", "cotas_rad", "etiquetas", "perfiles", "niveles",
    "llamadas",
];

fn truthy_list(v: &Value) -> Option<[f64; 3]> {
    let Value::Array(a) = v else { return None };
    if a.is_empty() {
        return None;
    }
    let n = |i: usize| match a.get(i) {
        Some(Value::Float(x)) => *x,
        Some(Value::Int(s)) => s.parse().unwrap_or(0.0),
        _ => 0.0,
    };
    Some([n(0), n(1), n(2)])
}

fn move_item(item: &mut Value, m: &M4, turn: f64) {
    let Value::Object(fields) = item else { return };
    let pt = |c: [f64; 3]| {
        let q = m.map(V3(c[0] as f32, c[1] as f32, c[2] as f32));
        Value::Array(vec![Value::Float(q.0 as f64), Value::Float(q.1 as f64), Value::Float(q.2 as f64)])
    };
    for (k, v) in fields.iter_mut() {
        match k.as_str() {
            "a_world" | "b_world" => {
                if let Some(c) = truthy_list(v) {
                    *v = pt(c);
                }
            }
            _ => {}
        }
    }
    let target = fields.iter().position(|(k, v)| k == "cam_target" && truthy_list(v).is_some());
    if let Some(i) = target {
        let c = truthy_list(&fields[i].1).expect("checked");
        fields[i].1 = pt(c);
        if let Some((_, yaw)) = fields.iter_mut().find(|(k, v)| k == "cam_yaw" && !matches!(v, Value::Null)) {
            let y = match yaw {
                Value::Float(x) => *x,
                Value::Int(s) => s.parse().unwrap_or(0.0),
                _ => 0.0,
            };
            *yaw = Value::Float(y + turn);
        }
    }
}

fn move_composition(comp: &mut Value, m: &M4, turn: f64) {
    let Value::Object(fields) = comp else { return };
    for (k, v) in fields.iter_mut() {
        if ITEM_LISTS.contains(&k.as_str()) {
            if let Value::Array(items) = v {
                for it in items {
                    move_item(it, m, turn);
                }
            }
        } else if k == "cajetin" {
            move_item(v, m, turn);
        }
    }
}

impl Command for StraightenModel {
    fn name(&self) -> &'static str {
        "straighten_model"
    }

    fn apply(&mut self, scene: &mut Scene, cx: &mut Cx) -> Result<(), Fail> {
        let xform = match group_at(&scene.groups, &self.path).filter(|_| self.path.len() == 1) {
            Some(g) => g.xform,
            None => None,
        };
        let Some(xform) = xform else { return fail("pick a placed component at the top level") };
        let Some(datum) = scene.extra.georef_datum.clone() else { return fail("the scene has no location yet") };
        if self.placement.is_none() {
            self.degrees = placement_rotation_deg(&xform)?;
            self.placement = Some(xform);
            let sd = SceneDatum::from_dict(&datum).map_err(Fail::Failed)?;
            let t = xform.column(3);
            let (lat, lon, _) = sd.local_to_geodetic(V3(t[0], t[1], 0.0));
            self.old_datum = Some(datum);
            self.new_datum = Some(SceneDatum::new(lat, lon, sd.alt, sd.north + self.degrees).to_dict());
        }
        let placement = self.placement.expect("set above");
        let (mut w, ok) = placement.inverted();
        if !ok {
            return fail("the placement is not invertible");
        }
        self.rigid(scene, cx, &mut w, -self.degrees, true);
        scene.groups[self.path[0]].xform = Some(M4::identity()); // exactly on its axes, no float dust
        scene.extra.georef_datum = self.new_datum.clone();
        Ok(())
    }

    fn undo(&mut self, scene: &mut Scene, cx: &mut Cx) {
        let Some(mut placement) = self.placement else { return };
        let degrees = self.degrees;
        self.rigid(scene, cx, &mut placement, degrees, false);
        if let Some(g) = scene.groups.get_mut(self.path[0]) {
            g.xform = self.placement;
        }
        scene.extra.georef_datum = self.old_datum.clone();
    }
}
