// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! The annotation blocks of an .igz document (`formats/igz.py`): how
//! `save_scene` writes them and `load_into` reads them back — saved views,
//! dimensions, section planes (and the `section_view` switches), leader
//! texts, guides and image planes.
//!
//! The blocks are interleaved with others in the file (layers, materials,
//! the mesh…); [`Annotations::to_igz_entries`] gives only these, in the
//! order `save_scene` writes them, for the document writer to place.

use ingetrazo_formats::pyjson::Value;
use ingetrazo_math::Hypot;

use crate::dimension::Dimension;
use crate::guide::Guide;
use crate::image_plane::ImagePlane;
use crate::pyconv::{get, iter_list, object, py_float, truthy, PyResult};
use crate::saved_views::{views_from_list, SavedView};
use crate::section::{SectionPlane, UidCounter};
use crate::textlabel::TextLabel;

#[derive(Clone, Debug, PartialEq)]
pub struct Annotations {
    pub saved_views: Vec<SavedView>,
    pub dimensions: Vec<Dimension>,
    pub section_planes: Vec<SectionPlane>,
    /// `Scene.show_section_planes` / `show_section_cuts`.
    pub show_section_planes: bool,
    pub show_section_cuts: bool,
    pub text_labels: Vec<TextLabel>,
    pub guides: Vec<Guide>,
    pub image_planes: Vec<ImagePlane>,
    /// `Scene.load_repairs`: guides left out for holding a coordinate that
    /// is not a number (#185).
    pub load_repairs: usize,
}

impl Default for Annotations {
    fn default() -> Self {
        Annotations {
            saved_views: Vec::new(),
            dimensions: Vec::new(),
            section_planes: Vec::new(),
            show_section_planes: true,
            show_section_cuts: true,
            text_labels: Vec::new(),
            guides: Vec::new(),
            image_planes: Vec::new(),
            load_repairs: 0,
        }
    }
}

/// `_finite_points`: every coordinate of every list reads as a finite
/// float; anything that does not is not finite.
fn finite_points(points: &[Option<&Value>]) -> bool {
    points.iter().all(|p| {
        let items: Vec<Value> = match p {
            None => Vec::new(),
            Some(Value::Array(a)) => a.clone(),
            Some(Value::Str(s)) => s.chars().map(|c| Value::Str(c.to_string())).collect(),
            Some(Value::Object(o)) => o.iter().map(|(k, _)| Value::Str(k.clone())).collect(),
            Some(_) => return false,
        };
        items.iter().all(|c| py_float(c).is_ok_and(f64::is_finite))
    })
}

impl Annotations {
    /// The blocks `save_scene` writes for these annotations, as
    /// `(key, value)` in file order; a block that would be empty is left
    /// out, `section_view` appears only when a switch is off, and a guide
    /// holding a NaN is left out (#723).
    pub fn to_igz_entries(&self) -> Vec<(&'static str, Value)> {
        let mut out = Vec::new();
        if !self.saved_views.is_empty() {
            out.push(("saved_views", Value::Array(self.saved_views.iter().map(SavedView::to_dict).collect())));
        }
        if !self.dimensions.is_empty() {
            out.push(("dimensions", Value::Array(self.dimensions.iter().map(Dimension::to_igz_value).collect())));
        }
        if !self.section_planes.is_empty() {
            out.push(("section_planes", Value::Array(self.section_planes.iter().map(SectionPlane::to_dict).collect())));
            let mut vis = Vec::new();
            if !self.show_section_planes {
                vis.push(("planes_hidden", Value::Bool(true)));
            }
            if !self.show_section_cuts {
                vis.push(("cuts_hidden", Value::Bool(true)));
            }
            if !vis.is_empty() {
                out.push(("section_view", object(vis)));
            }
        }
        if !self.text_labels.is_empty() {
            out.push(("text_labels", Value::Array(self.text_labels.iter().map(TextLabel::to_dict).collect())));
        }
        if !self.guides.is_empty() {
            out.push(("guides", Value::Array(self.guides.iter().filter(|g| g.is_finite()).map(Guide::to_dict).collect())));
        }
        if !self.image_planes.is_empty() {
            out.push(("image_planes", Value::Array(self.image_planes.iter().map(ImagePlane::to_dict).collect())));
        }
        out
    }

    /// What `load_into` makes of the annotation blocks of `payload` (the
    /// document's `"scene"` object). `uids` is the section planes' uid
    /// counter; the blocks are read in `load_into`'s order, so the first
    /// error is the one Python raises.
    pub fn from_igz_payload(payload: &Value, uids: &mut UidCounter, h: Hypot) -> PyResult<Annotations> {
        let mut a = Annotations::default();
        let empty = Value::Array(Vec::new());
        let block = |key: &str| -> PyResult<&Value> { Ok(get(payload, key)?.unwrap_or(&empty)) };

        a.saved_views = views_from_list(block("saved_views")?)?;
        for raw in iter_list(block("dimensions")?)? {
            a.dimensions.push(Dimension::from_igz_value(raw)?);
        }
        if let Some(Value::Array(raw_planes)) = get(payload, "section_planes")? {
            for raw in raw_planes {
                a.section_planes.push(SectionPlane::from_dict(raw, uids, h)?);
            }
            // One active cut max: keep the FIRST marked active.
            let mut seen_active = false;
            for sp in &mut a.section_planes {
                if sp.active && seen_active {
                    sp.active = false;
                }
                seen_active = seen_active || sp.active;
            }
        }
        if let Some(vis @ Value::Object(_)) = get(payload, "section_view")? {
            a.show_section_planes = !vis.get("planes_hidden").is_some_and(truthy);
            a.show_section_cuts = !vis.get("cuts_hidden").is_some_and(truthy);
        }
        for raw in iter_list(block("text_labels")?)? {
            a.text_labels.push(TextLabel::from_dict(raw)?);
        }
        for raw in iter_list(block("guides")?)? {
            // A guide at (nan, nan, nan) stalled every tool (#185): left out.
            let direction = get(raw, "direction")?.filter(|v| truthy(v));
            let origin = get(raw, "origin")?.filter(|v| truthy(v));
            if !finite_points(&[get(raw, "point")?, direction, origin]) {
                a.load_repairs += 1;
                continue;
            }
            a.guides.push(Guide::from_dict(raw, h)?);
        }
        for raw in iter_list(block("image_planes")?)? {
            a.image_planes.push(ImagePlane::from_dict(raw)?);
        }
        Ok(a)
    }
}
