// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! The annotation commands of `core/history.py` that edit the lists of a
//! scene: dimensions, leader texts, guides, saved scenes — plus the pieces
//! the commands of [`crate::section_cmds`] and [`crate::reference`] share.
//!
//! Python's commands keep the annotation objects themselves and compare
//! them by identity (`in`, `list.remove`, `list.index`: none of these classes
//! defines `__eq__`). Here an item is named by the identity its [`IdList`]
//! gave it, and the object a command adds travels with the command, in and
//! out of the scene as the command is done and undone, so a redo puts the
//! very object back with whatever it had become.
//!
//! `scene.selection` and `scene.version` have no counterpart in the model
//! yet: the `selection.discard/add/clear` calls and the version bumps of the
//! Python commands are left out. Neither is part of a document or of an L2
//! signature.

use ingetrazo_math::V3;
use ingetrazo_model::anchors::End;
use ingetrazo_model::dimension::Dimension;
use ingetrazo_model::guide::Guide;
use ingetrazo_model::idlist::IdList;
use ingetrazo_model::saved_views::SavedView;
use ingetrazo_model::scene::Scene;
use ingetrazo_model::textlabel::TextLabel;

use crate::command::{Command, Cx, Fail};

/// An item a command added: it waits in the command while the scene does not
/// hold it.
pub struct Held<T> {
    pub item: Option<T>,
    pub id: Option<u32>,
}

impl<T> Held<T> {
    pub fn new(item: T) -> Held<T> {
        Held { item: Some(item), id: None }
    }

    /// `list.append(item)`: the index it landed at. The item keeps the
    /// identity it had the first time, so the commands that refer to it
    /// still find it after an undo and a redo.
    pub fn put(&mut self, list: &mut IdList<T>) -> Result<usize, Fail> {
        let item = self.item.take().ok_or_else(|| Fail::Failed("the item is not available".into()))?;
        match self.id {
            Some(id) => list.insert_at(list.len(), id, item),
            None => self.id = Some(list.push(item)),
        }
        Ok(list.len() - 1)
    }

    /// `if item in list: list.remove(item)`.
    pub fn take(&mut self, list: &mut IdList<T>) {
        if let Some(id) = self.id {
            if let Some((_, item)) = list.remove_id(id) {
                self.item = Some(item);
            }
        }
    }
}

/// What a delete command remembers: where each item stood (`_restore`) and
/// the items themselves.
pub struct Removal<T> {
    restore: Vec<(usize, u32)>,
    taken: Vec<(u32, T)>,
}

impl<T> Default for Removal<T> {
    fn default() -> Self {
        Removal { restore: Vec::new(), taken: Vec::new() }
    }
}

impl<T: Clone> Removal<T> {
    /// `_restore = [(list.index(d), d) for d in ids if d in list]`, then
    /// `list.remove(d)` for each that is still in.
    pub fn remove(list: &mut IdList<T>, ids: &[u32]) -> Removal<T> {
        let restore: Vec<(usize, u32)> = ids.iter().filter_map(|&d| list.index_of(d).map(|i| (i, d))).collect();
        let mut taken = Vec::new();
        for &d in ids {
            if let Some((_, item)) = list.remove_id(d) {
                taken.push((d, item));
            }
        }
        Removal { restore, taken }
    }

    /// `for i, d in sorted(_restore): list.insert(i, d)`.
    pub fn put_back(&self, list: &mut IdList<T>) {
        let mut order = self.restore.clone();
        order.sort_by_key(|t| t.0);
        for (i, d) in order {
            if let Some((_, item)) = self.taken.iter().find(|(id, _)| *id == d) {
                list.insert_at(i, d, item.clone());
            }
        }
    }
}

fn missing(what: &str) -> Fail {
    Fail::Failed(format!("{what} is not in the scene"))
}

// ---- dimensions -------------------------------------------------------------------

/// `AddDimensionCommand`: appended and bound to the vertices under its ends.
pub struct AddDimension {
    held: Held<Dimension>,
}

impl AddDimension {
    pub fn new(dimension: Dimension) -> Self {
        AddDimension { held: Held::new(dimension) }
    }
}

impl Command for AddDimension {
    fn name(&self) -> &'static str {
        "add_dimension"
    }
    fn apply(&mut self, scene: &mut Scene, cx: &mut Cx) -> Result<(), Fail> {
        let at = self.held.put(&mut scene.dimensions)?;
        scene.bind_dimension(at, cx.mesh);
        Ok(())
    }
    fn undo(&mut self, scene: &mut Scene, _cx: &mut Cx) {
        self.held.take(&mut scene.dimensions);
    }
}

/// `DeleteDimensionsCommand`.
pub struct DeleteDimensions {
    ids: Vec<u32>,
    removal: Removal<Dimension>,
}

impl DeleteDimensions {
    pub fn new(ids: Vec<u32>) -> Self {
        DeleteDimensions { ids, removal: Removal::default() }
    }
}

impl Command for DeleteDimensions {
    fn name(&self) -> &'static str {
        "delete_dimensions"
    }
    fn apply(&mut self, scene: &mut Scene, _cx: &mut Cx) -> Result<(), Fail> {
        self.removal = Removal::remove(&mut scene.dimensions, &self.ids);
        Ok(())
    }
    fn undo(&mut self, scene: &mut Scene, _cx: &mut Cx) {
        self.removal.put_back(&mut scene.dimensions);
    }
}

/// How `MoveDimensionsCommand` moves one dimension.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MoveMode {
    /// An anchored dimension slides its line.
    Line,
    /// A free end moves rigidly.
    Rigid,
}

/// `MoveDimensionsCommand`: `modes` is the tool's dict, in its order. A key
/// is a dimension of the scene, or one that is not in it (the command works
/// on the object, wherever it lives).
pub struct MoveDimensions {
    modes: Vec<(u32, MoveMode)>,
    free: Vec<(Dimension, MoveMode)>,
    delta: V3,
}

impl MoveDimensions {
    pub fn new(modes: Vec<(u32, MoveMode)>, delta: V3) -> Self {
        MoveDimensions { modes, free: Vec::new(), delta }
    }

    /// The same, with dimensions that sit outside the scene.
    pub fn with_free(modes: Vec<(u32, MoveMode)>, free: Vec<(Dimension, MoveMode)>, delta: V3) -> Self {
        MoveDimensions { modes, free, delta }
    }

    /// `shift(dim, mode, step)` for a dimension nobody anchors: its ends read
    /// as they are.
    fn shift_free(dim: &mut Dimension, mode: MoveMode, step: V3, h: ingetrazo_math::Hypot) {
        match mode {
            MoveMode::Line => {
                let step = MoveDimensions::line_step(dim.b.sub(dim.a), step, h);
                dim.offset = dim.offset.add(step);
            }
            MoveMode::Rigid => {
                if dim.anchor_a.is_none() {
                    dim.a = dim.a.add(step);
                }
                if dim.anchor_b.is_none() {
                    dim.b = dim.b.add(step);
                }
            }
        }
    }

    /// The step with its component along the measured segment removed.
    fn line_step(ab: V3, step: V3, h: ingetrazo_math::Hypot) -> V3 {
        let length = ab.length(h);
        if (length as f64) > 1e-9 {
            let d = ab.div(length);
            return step.sub(d.mul(step.dot(d)));
        }
        step
    }

    /// `shift(dim, mode, step)`.
    fn shift(scene: &mut Scene, cx: &Cx, id: u32, mode: MoveMode, step: V3) {
        let Some(i) = scene.dimensions.index_of(id) else { return };
        let h = scene.meshes[0].hypot;
        match mode {
            MoveMode::Line => {
                let b = scene.dimension_end(i, End::B, cx.mesh);
                let a = scene.dimension_end(i, End::A, cx.mesh);
                let step = MoveDimensions::line_step(b.sub(a), step, h);
                let dim = &mut scene.dimensions[i];
                dim.offset = dim.offset.add(step);
            }
            MoveMode::Rigid => MoveDimensions::shift_free(&mut scene.dimensions[i], mode, step, h),
        }
    }
}

impl Command for MoveDimensions {
    fn name(&self) -> &'static str {
        "move_dimensions"
    }
    fn apply(&mut self, scene: &mut Scene, cx: &mut Cx) -> Result<(), Fail> {
        for &(id, mode) in &self.modes {
            MoveDimensions::shift(scene, cx, id, mode, self.delta);
        }
        let h = scene.meshes[0].hypot;
        for (dim, mode) in &mut self.free {
            MoveDimensions::shift_free(dim, *mode, self.delta, h);
        }
        Ok(())
    }
    fn undo(&mut self, scene: &mut Scene, cx: &mut Cx) {
        for &(id, mode) in &self.modes {
            MoveDimensions::shift(scene, cx, id, mode, self.delta.neg());
        }
        let h = scene.meshes[0].hypot;
        for (dim, mode) in &mut self.free {
            MoveDimensions::shift_free(dim, *mode, self.delta.neg(), h);
        }
    }
}

/// `EditDimensionTextCommand`: `None` goes back to the measured value.
pub struct EditDimensionText {
    id: u32,
    new: Option<String>,
    old: Option<String>,
}

impl EditDimensionText {
    /// The old text is read when the command is made, as Python does.
    pub fn new(scene: &Scene, id: u32, text: Option<String>) -> Result<Self, Fail> {
        let old = scene.dimensions.get_id(id).ok_or_else(|| missing("the dimension"))?.text.clone();
        Ok(EditDimensionText { id, new: text.filter(|t| !t.is_empty()), old })
    }
}

impl Command for EditDimensionText {
    fn name(&self) -> &'static str {
        "edit_dimension_text"
    }
    fn apply(&mut self, scene: &mut Scene, _cx: &mut Cx) -> Result<(), Fail> {
        scene.dimensions.get_id_mut(self.id).ok_or_else(|| missing("the dimension"))?.text = self.new.clone();
        Ok(())
    }
    fn undo(&mut self, scene: &mut Scene, _cx: &mut Cx) {
        if let Some(d) = scene.dimensions.get_id_mut(self.id) {
            d.text = self.old.clone();
        }
    }
}

// ---- leader texts -----------------------------------------------------------------

/// `AddTextLabelCommand`.
pub struct AddTextLabel {
    held: Held<TextLabel>,
}

impl AddTextLabel {
    pub fn new(label: TextLabel) -> Self {
        AddTextLabel { held: Held::new(label) }
    }
}

impl Command for AddTextLabel {
    fn name(&self) -> &'static str {
        "add_text_label"
    }
    fn apply(&mut self, scene: &mut Scene, _cx: &mut Cx) -> Result<(), Fail> {
        self.held.put(&mut scene.text_labels).map(|_| ())
    }
    fn undo(&mut self, scene: &mut Scene, _cx: &mut Cx) {
        self.held.take(&mut scene.text_labels);
    }
}

/// `MoveTextLabelsCommand`: the label end moves, the anchor stays.
pub struct MoveTextLabels {
    ids: Vec<u32>,
    delta: V3,
}

impl MoveTextLabels {
    pub fn new(ids: Vec<u32>, delta: V3) -> Self {
        MoveTextLabels { ids, delta }
    }
}

impl Command for MoveTextLabels {
    fn name(&self) -> &'static str {
        "move_text_labels"
    }
    fn apply(&mut self, scene: &mut Scene, _cx: &mut Cx) -> Result<(), Fail> {
        for &id in &self.ids {
            if let Some(t) = scene.text_labels.get_id_mut(id) {
                t.offset = t.offset.add(self.delta);
            }
        }
        Ok(())
    }
    fn undo(&mut self, scene: &mut Scene, _cx: &mut Cx) {
        for &id in &self.ids {
            if let Some(t) = scene.text_labels.get_id_mut(id) {
                t.offset = t.offset.sub(self.delta);
            }
        }
    }
}

/// `EditTextLabelCommand`.
pub struct EditTextLabel {
    id: u32,
    new: String,
    old: String,
}

impl EditTextLabel {
    pub fn new(scene: &Scene, id: u32, text: &str) -> Result<Self, Fail> {
        let old = scene.text_labels.get_id(id).ok_or_else(|| missing("the label"))?.text.clone();
        Ok(EditTextLabel { id, new: text.to_string(), old })
    }
}

impl Command for EditTextLabel {
    fn name(&self) -> &'static str {
        "edit_text_label"
    }
    fn apply(&mut self, scene: &mut Scene, _cx: &mut Cx) -> Result<(), Fail> {
        scene.text_labels.get_id_mut(self.id).ok_or_else(|| missing("the label"))?.text = self.new.clone();
        Ok(())
    }
    fn undo(&mut self, scene: &mut Scene, _cx: &mut Cx) {
        if let Some(t) = scene.text_labels.get_id_mut(self.id) {
            t.text = self.old.clone();
        }
    }
}

/// `DeleteTextLabelsCommand`.
pub struct DeleteTextLabels {
    ids: Vec<u32>,
    removal: Removal<TextLabel>,
}

impl DeleteTextLabels {
    pub fn new(ids: Vec<u32>) -> Self {
        DeleteTextLabels { ids, removal: Removal::default() }
    }
}

impl Command for DeleteTextLabels {
    fn name(&self) -> &'static str {
        "delete_text_labels"
    }
    fn apply(&mut self, scene: &mut Scene, _cx: &mut Cx) -> Result<(), Fail> {
        self.removal = Removal::remove(&mut scene.text_labels, &self.ids);
        Ok(())
    }
    fn undo(&mut self, scene: &mut Scene, _cx: &mut Cx) {
        self.removal.put_back(&mut scene.text_labels);
    }
}

// ---- guides -----------------------------------------------------------------------

/// `AddGuideCommand`.
pub struct AddGuide {
    held: Held<Guide>,
}

impl AddGuide {
    pub fn new(guide: Guide) -> Self {
        AddGuide { held: Held::new(guide) }
    }
}

impl Command for AddGuide {
    fn name(&self) -> &'static str {
        "add_guide"
    }
    fn apply(&mut self, scene: &mut Scene, _cx: &mut Cx) -> Result<(), Fail> {
        self.held.put(&mut scene.guides).map(|_| ())
    }
    fn undo(&mut self, scene: &mut Scene, _cx: &mut Cx) {
        self.held.take(&mut scene.guides);
    }
}

/// `ChangeGuideCommand`: re-aim a guide line.
pub struct ChangeGuide {
    id: u32,
    new: V3,
    old: Option<V3>,
}

impl ChangeGuide {
    /// `QVector3D(direction).normalized()`.
    pub fn new(scene: &Scene, id: u32, direction: V3) -> Self {
        ChangeGuide { id, new: direction.normalized(scene.meshes[0].hypot), old: None }
    }
}

impl Command for ChangeGuide {
    fn name(&self) -> &'static str {
        "change_guide"
    }
    fn apply(&mut self, scene: &mut Scene, _cx: &mut Cx) -> Result<(), Fail> {
        let g = scene.guides.get_id_mut(self.id).ok_or_else(|| missing("the guide"))?;
        self.old = g.direction;
        g.direction = Some(self.new);
        Ok(())
    }
    fn undo(&mut self, scene: &mut Scene, _cx: &mut Cx) {
        if let Some(g) = scene.guides.get_id_mut(self.id) {
            g.direction = self.old;
        }
    }
}

/// `DeleteGuidesCommand`.
pub struct DeleteGuides {
    ids: Vec<u32>,
    removal: Removal<Guide>,
}

impl DeleteGuides {
    pub fn new(ids: Vec<u32>) -> Self {
        DeleteGuides { ids, removal: Removal::default() }
    }
}

impl Command for DeleteGuides {
    fn name(&self) -> &'static str {
        "delete_guides"
    }
    fn apply(&mut self, scene: &mut Scene, _cx: &mut Cx) -> Result<(), Fail> {
        self.removal = Removal::remove(&mut scene.guides, &self.ids);
        Ok(())
    }
    fn undo(&mut self, scene: &mut Scene, _cx: &mut Cx) {
        self.removal.put_back(&mut scene.guides);
    }
}

// ---- saved scenes -----------------------------------------------------------------

/// `DeleteScenesCommand`: delete saved scenes, several at once; undo puts
/// each back in its place.
pub struct DeleteScenes {
    ids: Vec<u32>,
    removal: Removal<SavedView>,
}

impl DeleteScenes {
    pub fn new(ids: Vec<u32>) -> Self {
        DeleteScenes { ids, removal: Removal::default() }
    }
}

impl Command for DeleteScenes {
    fn name(&self) -> &'static str {
        "delete_scenes"
    }
    fn apply(&mut self, scene: &mut Scene, _cx: &mut Cx) -> Result<(), Fail> {
        self.removal = Removal::remove(&mut scene.extra.saved_views, &self.ids);
        Ok(())
    }
    fn undo(&mut self, scene: &mut Scene, _cx: &mut Cx) {
        self.removal.put_back(&mut scene.extra.saved_views);
    }
}
