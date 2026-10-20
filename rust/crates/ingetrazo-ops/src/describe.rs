// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! A command written down as JSON: its class and constructor arguments,
//! positions as float32 bits. This is what `scripts/golden/make_edits_fixtures.py`
//! records for the commands Python's planner builds, so a Rust plan can be
//! compared with it exactly.

use ingetrazo_formats::pyjson::Value;
use ingetrazo_math::V3;

use crate::command::Command;
use crate::draw::{
    AddEdgeCommand, AddFaceCommand, AutoTagDrawnFacesCommand, DeleteEdgesCommand, DeleteFaceCommand, EraseSelectionCommand,
    FlipFacesCommand, SnapshotCompound, TagCurveCommand,
};
use crate::rebuild::{RebuildPlanarFacesCommand, RebuildPlaneFacesCommand};

fn s(x: &str) -> Value {
    Value::Str(x.to_string())
}

/// A position as its three float32 bit patterns, lower-case hex.
pub fn bits(p: V3) -> Value {
    Value::Array(vec![
        Value::Str(format!("{:08x}", p.0.to_bits())),
        Value::Str(format!("{:08x}", p.1.to_bits())),
        Value::Str(format!("{:08x}", p.2.to_bits())),
    ])
}

fn pts(l: &[V3]) -> Value {
    Value::Array(l.iter().map(|&p| bits(p)).collect())
}

fn opt_bool(b: Option<bool>) -> Value {
    match b {
        Some(x) => Value::Bool(x),
        None => Value::Null,
    }
}

/// The command as a JSON list `[class, args...]`, or `None` for a command
/// this does not describe.
pub fn describe(c: &dyn Command) -> Option<Value> {
    let any = c.as_any();
    let list = |items: Vec<Value>| Some(Value::Array(items));
    if let Some(x) = any.downcast_ref::<AddEdgeCommand>() {
        return list(vec![s("AddEdge"), bits(x.a), bits(x.b), opt_bool(x.soft), x.curve.map_or(Value::Null, Value::int)]);
    }
    if let Some(x) = any.downcast_ref::<AddFaceCommand>() {
        let holes = match &x.preset_holes {
            Some(h) => Value::Array(h.iter().map(|l| pts(l)).collect()),
            None => Value::Null,
        };
        let attrs = match &x.attrs {
            Some(a) => Value::Object(a.clone()),
            None => Value::Null,
        };
        return list(vec![s("AddFace"), pts(&x.vertices), Value::Bool(x.auto), holes, attrs]);
    }
    if let Some(x) = any.downcast_ref::<DeleteEdgesCommand>() {
        let ends = x.endpoints.iter().map(|&(a, b)| Value::Array(vec![bits(a), bits(b)])).collect();
        return list(vec![s("DeleteEdges"), Value::Array(ends), Value::Bool(x.cascade_faces)]);
    }
    if let Some(x) = any.downcast_ref::<DeleteFaceCommand>() {
        return list(vec![s("DeleteFace"), pts(&x.loop_)]);
    }
    if any.is::<RebuildPlanarFacesCommand>() {
        return list(vec![s("RebuildPlanar")]);
    }
    if let Some(x) = any.downcast_ref::<RebuildPlaneFacesCommand>() {
        return list(vec![s("RebuildPlane"), bits(x.origin), bits(x.normal)]);
    }
    if let Some(x) = any.downcast_ref::<TagCurveCommand>() {
        return list(vec![s("TagCurve"), pts(&x.pts), Value::Bool(x.closed)]);
    }
    if let Some(x) = any.downcast_ref::<AutoTagDrawnFacesCommand>() {
        return list(vec![s("AutoTag"), Value::Object(x.tag.clone()), Value::int(x.face_commands.len() as i64)]);
    }
    if let Some(x) = any.downcast_ref::<EraseSelectionCommand>() {
        let ends = x.edge_endpoints.iter().map(|&(a, b)| Value::Array(vec![bits(a), bits(b)])).collect();
        let loops = x.face_loops.iter().map(|l| pts(l)).collect();
        return list(vec![s("Erase"), Value::Array(ends), Value::Array(loops)]);
    }
    if let Some(x) = any.downcast_ref::<FlipFacesCommand>() {
        return list(vec![s("Flip"), Value::int(x.faces.len() as i64)]);
    }
    if let Some(x) = any.downcast_ref::<SnapshotCompound>() {
        let inner: Option<Vec<Value>> = x.inner.iter().map(|c| describe(c.as_ref())).collect();
        return list(vec![s("Snapshot"), Value::Array(inner?)]);
    }
    None
}
