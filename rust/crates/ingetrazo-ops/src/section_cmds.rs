// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! The section-plane commands of `core/history.py`: place, delete, reverse,
//! activate, mode, fill, rename, move and rotate. The planes are named by
//! the identity their list gave them (see [`crate::annotate`]).

use ingetrazo_formats::pyjson::Value;
use ingetrazo_math::V3;
use ingetrazo_model::scene::Scene;
use ingetrazo_model::section::SectionPlane;

use crate::annotate::{Held, Removal};
use crate::command::{Command, Cx, Fail};
use crate::transform::rotation_matrix;

fn missing() -> Fail {
    Fail::Failed("the section plane is not in the scene".into())
}

/// `prev if prev in scene.section_planes else None`.
fn still_there(scene: &Scene, id: Option<u32>) -> Option<u32> {
    id.filter(|&i| scene.section_planes.contains_id(i))
}

/// `PlaceSectionPlaneCommand`: the new plane becomes the ACTIVE cut; undo
/// restores the previous one.
pub struct PlaceSectionPlane {
    held: Held<SectionPlane>,
    prev_active: Option<u32>,
}

impl PlaceSectionPlane {
    pub fn new(plane: SectionPlane) -> Self {
        PlaceSectionPlane { held: Held::new(plane), prev_active: None }
    }
}

impl Command for PlaceSectionPlane {
    fn name(&self) -> &'static str {
        "place_section_plane"
    }
    fn apply(&mut self, scene: &mut Scene, _cx: &mut Cx) -> Result<(), Fail> {
        self.prev_active = scene.active_section().and_then(|i| scene.section_planes.id_at(i));
        // The uid counter stays ahead of the plane's own uid.
        if let Some(p) = self.held.item.as_ref() {
            scene.section_uids.take(Some(p.uid));
        }
        self.held.put(&mut scene.section_planes)?;
        scene.set_active_section(self.held.id);
        Ok(())
    }
    fn undo(&mut self, scene: &mut Scene, _cx: &mut Cx) {
        self.held.take(&mut scene.section_planes);
        let prev = still_there(scene, self.prev_active);
        scene.set_active_section(prev);
    }
}

/// `DeleteSectionPlanesCommand`.
pub struct DeleteSectionPlanes {
    ids: Vec<u32>,
    removal: Removal<SectionPlane>,
}

impl DeleteSectionPlanes {
    pub fn new(ids: Vec<u32>) -> Self {
        DeleteSectionPlanes { ids, removal: Removal::default() }
    }
}

impl Command for DeleteSectionPlanes {
    fn name(&self) -> &'static str {
        "delete_section_planes"
    }
    fn apply(&mut self, scene: &mut Scene, _cx: &mut Cx) -> Result<(), Fail> {
        self.removal = Removal::remove(&mut scene.section_planes, &self.ids);
        Ok(())
    }
    fn undo(&mut self, scene: &mut Scene, _cx: &mut Cx) {
        self.removal.put_back(&mut scene.section_planes);
    }
}

/// `ReverseSectionPlaneCommand`: flip which side the plane hides.
pub struct ReverseSectionPlane {
    id: u32,
}

impl ReverseSectionPlane {
    pub fn new(id: u32) -> Self {
        ReverseSectionPlane { id }
    }

    fn flip(&self, scene: &mut Scene) -> Result<(), Fail> {
        scene.section_planes.get_id_mut(self.id).ok_or_else(missing)?.flip();
        Ok(())
    }
}

impl Command for ReverseSectionPlane {
    fn name(&self) -> &'static str {
        "reverse_section_plane"
    }
    fn apply(&mut self, scene: &mut Scene, _cx: &mut Cx) -> Result<(), Fail> {
        self.flip(scene)
    }
    fn undo(&mut self, scene: &mut Scene, _cx: &mut Cx) {
        let _ = self.flip(scene);
    }
}

/// `SetActiveSectionCommand`: make a plane the active cut (`None` deactivates).
pub struct SetActiveSection {
    plane: Option<u32>,
    prev: Option<u32>,
}

impl SetActiveSection {
    pub fn new(plane: Option<u32>) -> Self {
        SetActiveSection { plane, prev: None }
    }
}

impl Command for SetActiveSection {
    fn name(&self) -> &'static str {
        "set_active_section"
    }
    fn apply(&mut self, scene: &mut Scene, _cx: &mut Cx) -> Result<(), Fail> {
        self.prev = scene.active_section().and_then(|i| scene.section_planes.id_at(i));
        scene.set_active_section(self.plane);
        Ok(())
    }
    fn undo(&mut self, scene: &mut Scene, _cx: &mut Cx) {
        let prev = still_there(scene, self.prev);
        scene.set_active_section(prev);
    }
}

/// `SetSectionModeCommand`: whether the plane CUTS and whether its FRAME shows.
pub struct SetSectionMode {
    plane: u32,
    cut: bool,
    frame: bool,
    prev: Option<(Option<u32>, bool)>,
}

impl SetSectionMode {
    pub fn new(plane: u32, cut: bool, frame: bool) -> Self {
        SetSectionMode { plane, cut, frame, prev: None }
    }
}

impl Command for SetSectionMode {
    fn name(&self) -> &'static str {
        "set_section_mode"
    }
    fn apply(&mut self, scene: &mut Scene, _cx: &mut Cx) -> Result<(), Fail> {
        let active = scene.active_section().and_then(|i| scene.section_planes.id_at(i));
        let sp = scene.section_planes.get_id(self.plane).ok_or_else(missing)?;
        self.prev = Some((active, sp.hidden));
        let is_active = sp.active;
        if self.cut {
            scene.set_active_section(Some(self.plane));
        } else if is_active {
            scene.set_active_section(None);
        }
        scene.section_planes.get_id_mut(self.plane).ok_or_else(missing)?.hidden = !self.frame;
        Ok(())
    }
    fn undo(&mut self, scene: &mut Scene, _cx: &mut Cx) {
        let Some((active, hidden)) = self.prev else { return };
        let active = still_there(scene, active);
        scene.set_active_section(active);
        if let Some(sp) = scene.section_planes.get_id_mut(self.plane) {
            sp.hidden = hidden;
        }
    }
}

/// `SetSectionFillCommand`: the plane's own fill over the style's; `None`
/// goes back to the style's.
pub struct SetSectionFill {
    plane: u32,
    fill: Option<Value>,
    prev: Option<Value>,
}

impl SetSectionFill {
    /// `dict(fill) if fill else None`.
    pub fn new(plane: u32, fill: Option<Value>) -> Self {
        let fill = fill.filter(|f| matches!(f, Value::Object(o) if !o.is_empty()));
        SetSectionFill { plane, fill, prev: None }
    }
}

impl Command for SetSectionFill {
    fn name(&self) -> &'static str {
        "set_section_fill"
    }
    fn apply(&mut self, scene: &mut Scene, _cx: &mut Cx) -> Result<(), Fail> {
        let sp = scene.section_planes.get_id_mut(self.plane).ok_or_else(missing)?;
        self.prev = sp.fill.clone().filter(|f| matches!(f, Value::Object(o) if !o.is_empty()));
        sp.fill = self.fill.clone();
        Ok(())
    }
    fn undo(&mut self, scene: &mut Scene, _cx: &mut Cx) {
        if let Some(sp) = scene.section_planes.get_id_mut(self.plane) {
            sp.fill = self.prev.clone();
        }
    }
}

/// `RenameSectionPlaneCommand`: the name and the symbol.
pub struct RenameSectionPlane {
    plane: u32,
    name: String,
    symbol: String,
    prev: Option<(String, String)>,
}

impl RenameSectionPlane {
    pub fn new(plane: u32, name: &str, symbol: &str) -> Self {
        RenameSectionPlane { plane, name: name.to_string(), symbol: symbol.to_string(), prev: None }
    }
}

impl Command for RenameSectionPlane {
    fn name(&self) -> &'static str {
        "rename_section_plane"
    }
    fn apply(&mut self, scene: &mut Scene, _cx: &mut Cx) -> Result<(), Fail> {
        let sp = scene.section_planes.get_id_mut(self.plane).ok_or_else(missing)?;
        self.prev = Some((sp.name.clone(), sp.symbol.clone()));
        sp.name = self.name.clone();
        sp.symbol = self.symbol.clone();
        Ok(())
    }
    fn undo(&mut self, scene: &mut Scene, _cx: &mut Cx) {
        if let (Some(sp), Some((name, symbol))) = (scene.section_planes.get_id_mut(self.plane), self.prev.clone()) {
            sp.name = name;
            sp.symbol = symbol;
        }
    }
}

/// `MoveSectionPlanesCommand`.
pub struct MoveSectionPlanes {
    ids: Vec<u32>,
    delta: V3,
}

impl MoveSectionPlanes {
    pub fn new(ids: Vec<u32>, delta: V3) -> Self {
        MoveSectionPlanes { ids, delta }
    }
}

impl Command for MoveSectionPlanes {
    fn name(&self) -> &'static str {
        "move_section_planes"
    }
    fn apply(&mut self, scene: &mut Scene, _cx: &mut Cx) -> Result<(), Fail> {
        for &id in &self.ids {
            if let Some(p) = scene.section_planes.get_id_mut(id) {
                p.point = p.point.add(self.delta);
            }
        }
        Ok(())
    }
    fn undo(&mut self, scene: &mut Scene, _cx: &mut Cx) {
        for &id in &self.ids {
            if let Some(p) = scene.section_planes.get_id_mut(id) {
                p.point = p.point.sub(self.delta);
            }
        }
    }
}

/// `RotateSectionPlanesCommand`: the origin orbits and the normal turns.
pub struct RotateSectionPlanes {
    ids: Vec<u32>,
    centre: V3,
    axis: V3,
    deg: f64,
}

impl RotateSectionPlanes {
    pub fn new(ids: Vec<u32>, centre: V3, axis: V3, deg: f64) -> Self {
        RotateSectionPlanes { ids, centre, axis, deg }
    }

    fn turn(&self, scene: &mut Scene, deg: f64) {
        let h = scene.meshes[0].hypot;
        let m = rotation_matrix(self.centre, self.axis, deg, h);
        for &id in &self.ids {
            if let Some(p) = scene.section_planes.get_id_mut(id) {
                p.point = m.map(p.point);
                let n = m.map_vector(p.normal);
                if (n.length(h) as f64) > 1e-12 {
                    p.normal = n.normalized(h);
                }
            }
        }
    }
}

impl Command for RotateSectionPlanes {
    fn name(&self) -> &'static str {
        "rotate_section_planes"
    }
    fn apply(&mut self, scene: &mut Scene, _cx: &mut Cx) -> Result<(), Fail> {
        self.turn(scene, self.deg);
        Ok(())
    }
    fn undo(&mut self, scene: &mut Scene, _cx: &mut Cx) {
        self.turn(scene, -self.deg);
    }
}
