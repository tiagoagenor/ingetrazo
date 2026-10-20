// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! What the scene says about what may be seen and touched
//! (`Scene.entity_visible`, `entity_selectable`, `_layer_state`,
//! `active_section`) and the display switches the picks read.

use ingetrazo_formats::pyjson::Value;
use ingetrazo_math::Hypot;
use ingetrazo_model::layers::DEFAULT_LAYER;
use ingetrazo_model::pyval::truthy;
use ingetrazo_model::scene::{MeshId, Scene};
use ingetrazo_model::section::SectionPlane;

/// Where a group is in the scene tree: indexes from `scene.groups` through
/// `children`.
pub type Path = Vec<usize>;

/// The face mode of the style the frame draws with; only the two that show
/// everything matter to a pick.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FaceMode {
    Solid,
    Xray,
    Wireframe,
}

impl FaceMode {
    /// `style.face_mode`.
    pub fn from_name(name: &str) -> FaceMode {
        match name {
            "xray" => FaceMode::Xray,
            "wireframe" => FaceMode::Wireframe,
            _ => FaceMode::Solid,
        }
    }

    /// The face mode of the scene's display style (`"textures"` when it is
    /// the default style the file does not carry).
    pub fn of_scene(scene: &Scene) -> FaceMode {
        let name = scene.extra.display_style.as_ref().and_then(|v| v.get("face_mode")).and_then(|v| match v {
            Value::Str(s) => Some(s.as_str()),
            _ => None,
        });
        FaceMode::from_name(name.unwrap_or("textures"))
    }
}

/// A scene seen by the picks: the mesh that is the loose part now, the
/// group open for editing, and the switches.
#[derive(Clone)]
pub struct State<'a> {
    pub scene: &'a Scene,
    /// `scene.mesh`: the loose part (the open group's mesh inside a group).
    pub active: MeshId,
    /// `scene.loose_mesh`: the model's own loose mesh.
    pub root_loose: MeshId,
    /// `scene.edit_group`, by path.
    pub ctx: Option<Path>,
    /// `Viewport._rest_is_hidden()`.
    pub rest_hidden: bool,
    pub face_mode: FaceMode,
    pub h: Hypot,
    layers: Vec<(String, bool, bool)>,
}

impl<'a> State<'a> {
    pub fn new(scene: &'a Scene, active: MeshId, root_loose: MeshId, ctx: Option<Path>, rest_hidden: bool, face_mode: FaceMode, h: Hypot) -> State<'a> {
        let layers = scene
            .layers
            .iter()
            .map(|raw| {
                let name = match raw.get("name") {
                    Some(Value::Str(s)) => s.clone(),
                    _ => DEFAULT_LAYER.to_string(),
                };
                let visible = raw.get("visible").map(truthy).unwrap_or(true);
                let locked = raw.get("locked").map(truthy).unwrap_or(false);
                (name, visible, locked)
            })
            .collect();
        State { scene, active, root_loose, ctx, rest_hidden, face_mode, h, layers }
    }

    pub fn show_hidden_objects(&self) -> bool {
        self.scene.extra.show_hidden_objects
    }

    pub fn show_hidden_geometry(&self) -> bool {
        self.scene.extra.show_hidden_geometry
    }

    pub fn show_section_planes(&self) -> bool {
        !self.scene.extra.planes_hidden
    }

    /// `(visible, locked)` of a layer by name (`None` / empty: the default
    /// layer); unknown names read as visible and unlocked.
    pub fn layer_state(&self, name: Option<&str>) -> (bool, bool) {
        let name = match name {
            Some(n) if !n.is_empty() => n,
            _ => DEFAULT_LAYER,
        };
        match self.layers.iter().find(|(n, _, _)| n == name) {
            Some((_, v, l)) => (*v, *l),
            None => (true, false),
        }
    }

    /// `entity_visible` of something that is not hidden by itself.
    pub fn layer_visible(&self, layer: Option<&str>) -> bool {
        self.layer_state(layer).0
    }

    /// `entity_selectable` of something with only a layer to answer to.
    pub fn layer_selectable(&self, layer: Option<&str>) -> bool {
        let (v, l) = self.layer_state(layer);
        v && !l
    }

    /// `attrs.get("hidden")` of a face.
    pub fn face_hidden(attrs: &[(String, Value)]) -> bool {
        attrs.iter().any(|(k, v)| k == "hidden" && truthy(v))
    }

    /// `layer_of(face)`: the name in `attrs`, if there is one.
    pub fn face_layer(attrs: &[(String, Value)]) -> Option<&str> {
        attrs.iter().find(|(k, _)| k == "layer").and_then(|(_, v)| match v {
            Value::Str(s) if !s.is_empty() => Some(s.as_str()),
            _ => None,
        })
    }

    pub fn face_visible(&self, attrs: &[(String, Value)]) -> bool {
        !Self::face_hidden(attrs) && self.layer_visible(Self::face_layer(attrs))
    }

    pub fn face_selectable(&self, attrs: &[(String, Value)]) -> bool {
        if Self::face_hidden(attrs) && !self.show_hidden_geometry() {
            return false;
        }
        self.layer_selectable(Self::face_layer(attrs))
    }

    /// `_active_cut`: the section plane actually cutting (cuts shown and
    /// one of them active).
    pub fn active_cut(&self) -> Option<&'a SectionPlane> {
        if self.scene.extra.cuts_hidden {
            return None;
        }
        self.scene.section_planes.iter().find(|sp| sp.active)
    }
}
