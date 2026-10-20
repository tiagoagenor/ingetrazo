// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! The commands of `core/history.py` for reference data: imported images
//! (`ImagePlane`) and georef paths and survey points. Items are named by the
//! identity their list gave them (see [`crate::annotate`]).

use ingetrazo_formats::pyjson::Value;
use ingetrazo_georef::geopath::GeoPath;
use ingetrazo_georef::points::GeoPoint;
use ingetrazo_math::pyfloat::{py_max2, py_min2};
use ingetrazo_math::V3;
use ingetrazo_model::image_plane::ImagePlane;
use ingetrazo_model::layers::Layer;
use ingetrazo_model::scene::Scene;

use crate::annotate::{Held, Removal};
use crate::command::{Command, Cx, Fail};
use crate::transform::{invert_scale_factor, rotation_matrix, scale_matrix, Factor};

fn missing(what: &str) -> Fail {
    Fail::Failed(format!("{what} is not in the scene"))
}

/// `scene.layer(name)`: the index of the layer called `name`.
fn layer_index(scene: &Scene, name: &str) -> Option<usize> {
    scene.layers.iter().position(|l| matches!(l.get("name"), Some(Value::Str(n)) if n == name))
}

// ---- image planes -----------------------------------------------------------------

/// `AddImagePlaneCommand`: the image lands on its own layer, which the
/// command makes when it is missing and takes back on undo.
pub struct AddImagePlane {
    held: Held<ImagePlane>,
    made_layer: bool,
}

impl AddImagePlane {
    pub fn new(image: ImagePlane) -> Self {
        AddImagePlane { held: Held::new(image), made_layer: false }
    }
}

impl Command for AddImagePlane {
    fn name(&self) -> &'static str {
        "add_image_plane"
    }
    fn apply(&mut self, scene: &mut Scene, _cx: &mut Cx) -> Result<(), Fail> {
        let name = self.held.item.as_ref().and_then(|i| i.layer.clone()).filter(|n| !n.is_empty());
        self.made_layer = false;
        if let Some(name) = name {
            if layer_index(scene, &name).is_none() {
                scene.layers.push(Layer::new(&name).to_dict());
                self.made_layer = true;
            }
        }
        self.held.put(&mut scene.image_planes).map(|_| ())
    }
    fn undo(&mut self, scene: &mut Scene, _cx: &mut Cx) {
        self.held.take(&mut scene.image_planes);
        if self.made_layer {
            let name = self.held.item.as_ref().and_then(|i| i.layer.clone()).filter(|n| !n.is_empty());
            if let Some(i) = name.and_then(|n| layer_index(scene, &n)) {
                scene.layers.remove(i);
            }
            self.made_layer = false;
        }
    }
}

/// `DeleteImagePlanesCommand`.
pub struct DeleteImagePlanes {
    ids: Vec<u32>,
    removal: Removal<ImagePlane>,
}

impl DeleteImagePlanes {
    pub fn new(ids: Vec<u32>) -> Self {
        DeleteImagePlanes { ids, removal: Removal::default() }
    }
}

impl Command for DeleteImagePlanes {
    fn name(&self) -> &'static str {
        "delete_image_planes"
    }
    fn apply(&mut self, scene: &mut Scene, _cx: &mut Cx) -> Result<(), Fail> {
        self.removal = Removal::remove(&mut scene.image_planes, &self.ids);
        Ok(())
    }
    fn undo(&mut self, scene: &mut Scene, _cx: &mut Cx) {
        self.removal.put_back(&mut scene.image_planes);
    }
}

/// `TransformImagePlaneCommand`: set an image's corner and edge vectors.
pub struct TransformImagePlane {
    id: u32,
    new: (Option<V3>, Option<V3>, Option<V3>),
    old: Option<(V3, V3, V3)>,
}

impl TransformImagePlane {
    pub fn new(id: u32, origin: Option<V3>, u: Option<V3>, v: Option<V3>) -> Self {
        TransformImagePlane { id, new: (origin, u, v), old: None }
    }
}

impl Command for TransformImagePlane {
    fn name(&self) -> &'static str {
        "transform_image_plane"
    }
    fn apply(&mut self, scene: &mut Scene, _cx: &mut Cx) -> Result<(), Fail> {
        let im = scene.image_planes.get_id_mut(self.id).ok_or_else(|| missing("the image"))?;
        self.old = Some((im.origin, im.u, im.v));
        let (origin, u, v) = self.new;
        if let Some(o) = origin {
            im.origin = o;
        }
        if let Some(u) = u {
            im.u = u;
        }
        if let Some(v) = v {
            im.v = v;
        }
        Ok(())
    }
    fn undo(&mut self, scene: &mut Scene, _cx: &mut Cx) {
        let (Some((o, u, v)), Some(im)) = (self.old, scene.image_planes.get_id_mut(self.id)) else { return };
        im.origin = o;
        im.u = u;
        im.v = v;
    }
}

/// `SetImagePlaneOpacityCommand`.
pub struct SetImagePlaneOpacity {
    id: u32,
    opacity: f64,
    old: f64,
}

impl SetImagePlaneOpacity {
    /// `max(0.0, min(1.0, float(opacity)))`; the old value is read now.
    pub fn new(scene: &Scene, id: u32, opacity: f64) -> Result<Self, Fail> {
        let old = scene.image_planes.get_id(id).ok_or_else(|| missing("the image"))?.opacity;
        Ok(SetImagePlaneOpacity { id, opacity: py_max2(0.0, py_min2(1.0, opacity)), old })
    }
}

impl Command for SetImagePlaneOpacity {
    fn name(&self) -> &'static str {
        "set_image_plane_opacity"
    }
    fn apply(&mut self, scene: &mut Scene, _cx: &mut Cx) -> Result<(), Fail> {
        scene.image_planes.get_id_mut(self.id).ok_or_else(|| missing("the image"))?.opacity = self.opacity;
        Ok(())
    }
    fn undo(&mut self, scene: &mut Scene, _cx: &mut Cx) {
        if let Some(im) = scene.image_planes.get_id_mut(self.id) {
            im.opacity = self.old;
        }
    }
}

/// `MoveImagePlanesCommand`: only the origin corner moves.
pub struct MoveImagePlanes {
    ids: Vec<u32>,
    delta: V3,
}

impl MoveImagePlanes {
    pub fn new(ids: Vec<u32>, delta: V3) -> Self {
        MoveImagePlanes { ids, delta }
    }

    fn shift(&self, scene: &mut Scene, delta: V3) {
        for &id in &self.ids {
            if let Some(im) = scene.image_planes.get_id_mut(id) {
                im.origin = im.origin.add(delta);
            }
        }
    }
}

impl Command for MoveImagePlanes {
    fn name(&self) -> &'static str {
        "move_image_planes"
    }
    fn apply(&mut self, scene: &mut Scene, _cx: &mut Cx) -> Result<(), Fail> {
        self.shift(scene, self.delta);
        Ok(())
    }
    fn undo(&mut self, scene: &mut Scene, _cx: &mut Cx) {
        self.shift(scene, self.delta.neg());
    }
}

/// `RotateImagePlanesCommand`: the origin orbits and both edge vectors turn.
pub struct RotateImagePlanes {
    ids: Vec<u32>,
    centre: V3,
    axis: V3,
    deg: f64,
}

impl RotateImagePlanes {
    pub fn new(ids: Vec<u32>, centre: V3, axis: V3, deg: f64) -> Self {
        RotateImagePlanes { ids, centre, axis, deg }
    }

    fn turn(&self, scene: &mut Scene, deg: f64) {
        let h = scene.meshes[0].hypot;
        let m = rotation_matrix(self.centre, self.axis, deg, h);
        for &id in &self.ids {
            if let Some(im) = scene.image_planes.get_id_mut(id) {
                im.origin = m.map(im.origin);
                im.u = m.map_vector(im.u);
                im.v = m.map_vector(im.v);
            }
        }
    }
}

impl Command for RotateImagePlanes {
    fn name(&self) -> &'static str {
        "rotate_image_planes"
    }
    fn apply(&mut self, scene: &mut Scene, _cx: &mut Cx) -> Result<(), Fail> {
        self.turn(scene, self.deg);
        Ok(())
    }
    fn undo(&mut self, scene: &mut Scene, _cx: &mut Cx) {
        self.turn(scene, -self.deg);
    }
}

/// `ScaleImagePlanesCommand`: scale about `anchor`; both edge vectors take
/// the same factor, so the picture is never distorted (unless a per-axis
/// factor is asked for).
pub struct ScaleImagePlanes {
    ids: Vec<u32>,
    anchor: V3,
    factor: Factor,
    axes: Option<[V3; 3]>,
}

impl ScaleImagePlanes {
    pub fn new(ids: Vec<u32>, anchor: V3, factor: Factor, axes: Option<[V3; 3]>) -> Self {
        ScaleImagePlanes { ids, anchor, factor, axes }
    }

    fn scale(&self, scene: &mut Scene, factor: Factor) {
        let m = scale_matrix(self.anchor, factor, self.axes);
        for &id in &self.ids {
            if let Some(im) = scene.image_planes.get_id_mut(id) {
                im.origin = m.map(im.origin);
                im.u = m.map_vector(im.u);
                im.v = m.map_vector(im.v);
            }
        }
    }
}

impl Command for ScaleImagePlanes {
    fn name(&self) -> &'static str {
        "scale_image_planes"
    }
    fn apply(&mut self, scene: &mut Scene, _cx: &mut Cx) -> Result<(), Fail> {
        self.scale(scene, self.factor);
        Ok(())
    }
    fn undo(&mut self, scene: &mut Scene, _cx: &mut Cx) {
        self.scale(scene, invert_scale_factor(self.factor));
    }
}

// ---- georef paths and points --------------------------------------------------------

/// `AddGeoPathCommand`.
pub struct AddGeoPath {
    held: Held<GeoPath>,
}

impl AddGeoPath {
    pub fn new(path: GeoPath) -> Self {
        AddGeoPath { held: Held::new(path) }
    }
}

impl Command for AddGeoPath {
    fn name(&self) -> &'static str {
        "add_geo_path"
    }
    fn apply(&mut self, scene: &mut Scene, _cx: &mut Cx) -> Result<(), Fail> {
        self.held.put(&mut scene.geo_paths).map(|_| ())
    }
    fn undo(&mut self, scene: &mut Scene, _cx: &mut Cx) {
        self.held.take(&mut scene.geo_paths);
    }
}

/// `AddGeoPointsCommand`: a batch of survey points, and the datum the import
/// anchored the scene with, when it did.
pub struct AddGeoPoints {
    points: Vec<GeoPoint>,
    ids: Vec<u32>,
    datum: Option<Value>,
    prev_datum: Option<Value>,
}

impl AddGeoPoints {
    /// `datum` is `SceneDatum.to_dict()`.
    pub fn new(points: Vec<GeoPoint>, datum: Option<Value>) -> Self {
        AddGeoPoints { points, ids: Vec::new(), datum, prev_datum: None }
    }
}

impl Command for AddGeoPoints {
    fn name(&self) -> &'static str {
        "add_geo_points"
    }
    fn apply(&mut self, scene: &mut Scene, _cx: &mut Cx) -> Result<(), Fail> {
        if let Some(d) = &self.datum {
            self.prev_datum = scene.extra.georef_datum.clone();
            scene.extra.georef_datum = Some(d.clone());
        }
        let points = std::mem::take(&mut self.points);
        if self.ids.is_empty() {
            self.ids = scene.extra.geo_points.extend_items(points);
        } else {
            // A redo: the points come back under the identities they had.
            for (id, p) in self.ids.clone().into_iter().zip(points) {
                let end = scene.extra.geo_points.len();
                scene.extra.geo_points.insert_at(end, id, p);
            }
        }
        Ok(())
    }
    fn undo(&mut self, scene: &mut Scene, _cx: &mut Cx) {
        for &id in &self.ids {
            if let Some((_, p)) = scene.extra.geo_points.remove_id(id) {
                self.points.push(p);
            }
        }
        if self.datum.is_some() {
            scene.extra.georef_datum = self.prev_datum.clone();
        }
    }
}

/// `DeleteGeoPointsCommand`.
pub struct DeleteGeoPoints {
    ids: Vec<u32>,
    removal: Removal<GeoPoint>,
}

impl DeleteGeoPoints {
    pub fn new(ids: Vec<u32>) -> Self {
        DeleteGeoPoints { ids, removal: Removal::default() }
    }
}

impl Command for DeleteGeoPoints {
    fn name(&self) -> &'static str {
        "delete_geo_points"
    }
    fn apply(&mut self, scene: &mut Scene, _cx: &mut Cx) -> Result<(), Fail> {
        self.removal = Removal::remove(&mut scene.extra.geo_points, &self.ids);
        Ok(())
    }
    fn undo(&mut self, scene: &mut Scene, _cx: &mut Cx) {
        self.removal.put_back(&mut scene.extra.geo_points);
    }
}

/// `DeleteGeoPathsCommand`.
pub struct DeleteGeoPaths {
    ids: Vec<u32>,
    removal: Removal<GeoPath>,
}

impl DeleteGeoPaths {
    pub fn new(ids: Vec<u32>) -> Self {
        DeleteGeoPaths { ids, removal: Removal::default() }
    }
}

impl Command for DeleteGeoPaths {
    fn name(&self) -> &'static str {
        "delete_geo_paths"
    }
    fn apply(&mut self, scene: &mut Scene, _cx: &mut Cx) -> Result<(), Fail> {
        self.removal = Removal::remove(&mut scene.geo_paths, &self.ids);
        Ok(())
    }
    fn undo(&mut self, scene: &mut Scene, _cx: &mut Cx) {
        self.removal.put_back(&mut scene.geo_paths);
    }
}

/// `ToggleGeoPathClosedCommand`: flip open and closed; do and undo are one.
pub struct ToggleGeoPathClosed {
    ids: Vec<u32>,
}

impl ToggleGeoPathClosed {
    pub fn new(ids: Vec<u32>) -> Self {
        ToggleGeoPathClosed { ids }
    }

    fn flip(&self, scene: &mut Scene) {
        for &id in &self.ids {
            if let Some(p) = scene.geo_paths.get_id_mut(id) {
                p.closed = !p.closed;
            }
        }
    }
}

impl Command for ToggleGeoPathClosed {
    fn name(&self) -> &'static str {
        "toggle_geo_path_closed"
    }
    fn apply(&mut self, scene: &mut Scene, _cx: &mut Cx) -> Result<(), Fail> {
        self.flip(scene);
        Ok(())
    }
    fn undo(&mut self, scene: &mut Scene, _cx: &mut Cx) {
        self.flip(scene);
    }
}

/// `SetGeoPathSurfaceCommand`: the terrain-surface mode (`None`, `"flat"`,
/// `"draped"`); a surface implies a closed polygon, so this also closes the
/// path, and undo restores both. The built triangles depend on the DEM and
/// are cleared here.
pub struct SetGeoPathSurface {
    mode: Option<String>,
    ids: Vec<u32>,
    prev: Vec<(u32, Option<String>, bool)>,
}

impl SetGeoPathSurface {
    pub fn new(ids: Vec<u32>, mode: Option<String>) -> Self {
        SetGeoPathSurface { mode, ids, prev: Vec::new() }
    }
}

impl Command for SetGeoPathSurface {
    fn name(&self) -> &'static str {
        "set_geo_path_surface"
    }
    fn apply(&mut self, scene: &mut Scene, _cx: &mut Cx) -> Result<(), Fail> {
        self.prev = self.ids.iter().filter_map(|&id| scene.geo_paths.get_id(id).map(|p| (id, p.surface.clone(), p.closed))).collect();
        for &id in &self.ids {
            if let Some(p) = scene.geo_paths.get_id_mut(id) {
                p.surface = self.mode.clone();
                if self.mode.as_ref().is_some_and(|m| !m.is_empty()) {
                    p.closed = true;
                }
                p.surface_tris = None;
            }
        }
        Ok(())
    }
    fn undo(&mut self, scene: &mut Scene, _cx: &mut Cx) {
        for (id, surface, closed) in self.prev.clone() {
            if let Some(p) = scene.geo_paths.get_id_mut(id) {
                p.surface = surface;
                p.closed = closed;
                p.surface_tris = None;
            }
        }
    }
}

/// `MoveGeoPathNodeCommand`: move one node of a path.
pub struct MoveGeoPathNode {
    id: u32,
    index: i64,
    new: V3,
    old: Option<V3>,
}

impl MoveGeoPathNode {
    pub fn new(id: u32, index: i64, new_point: V3) -> Self {
        MoveGeoPathNode { id, index, new: new_point, old: None }
    }

    /// Python's list index: negative counts from the end.
    fn at(&self, len: usize) -> Option<usize> {
        let i = if self.index < 0 { self.index + len as i64 } else { self.index };
        (0..len as i64).contains(&i).then_some(i as usize)
    }
}

impl Command for MoveGeoPathNode {
    fn name(&self) -> &'static str {
        "move_geo_path_node"
    }
    fn apply(&mut self, scene: &mut Scene, _cx: &mut Cx) -> Result<(), Fail> {
        let p = scene.geo_paths.get_id_mut(self.id).ok_or_else(|| missing("the path"))?;
        let i = self.at(p.points.len()).ok_or_else(|| Fail::Failed("IndexError: list index out of range".into()))?;
        self.old = Some(p.points[i]);
        p.points[i] = self.new;
        Ok(())
    }
    fn undo(&mut self, scene: &mut Scene, _cx: &mut Cx) {
        let (Some(old), Some(p)) = (self.old, scene.geo_paths.get_id_mut(self.id)) else { return };
        if let Some(i) = self.at(p.points.len()) {
            p.points[i] = old;
        }
    }
}
