// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! Running the steps of a script (golden/SCHEMA.md §3) against the Rust
//! engine: the start document opened into a `Scene`, each Command step built
//! from its `args`, run through `History` with the step's counters set, and
//! the L2 signature of the scene compared with the step's `sig`.
//!
//! A step whose op no module here builds ends the replay as `Unsupported`;
//! everything before it ran and matched. New ops add a `build` function in a
//! module of their own (see `ops_transform`) and one line in [`build_command`].

use ingetrazo_formats::igz::{self, Zip};
use ingetrazo_formats::pyjson::Value;
use ingetrazo_model::scene::Scene;
use ingetrazo_ops::command::{Command, Cx, History};

use crate::decode::Res;
use crate::sig::{signature_hash, Files};
use crate::{model_sig, ops_groups, ops_transform, Outcome, Script};

/// The Command `op` builds from `args` against `scene`: `None` when no
/// module of the engine knows the op yet.
pub fn build_command(op: &str, args: &Value, scene: &Scene) -> Option<Res<Box<dyn Command>>> {
    ops_transform::build(op, args, scene).or_else(|| ops_groups::build(op, args, scene))
}

fn text<'a>(step: &'a Value, key: &str) -> &'a str {
    match step.get(key) {
        Some(Value::Str(s)) => s,
        _ => "?",
    }
}

fn counter(step: &Value, key: &str) -> Option<i64> {
    step.get("counters").and_then(|c| c.get(key)).and_then(Value::as_i64)
}

pub fn run_steps(script: &Script, steps: &[Value]) -> Outcome {
    let doc = match igz::read_document(&script.start) {
        Ok(d) => d,
        Err(e) => return Outcome::Broken(format!("start.igz: {e}")),
    };
    let zip = if igz::is_container(&script.start) {
        match Zip::open(&script.start) {
            Ok(z) => Some(z),
            Err(e) => return Outcome::Broken(format!("start.igz: {e}")),
        }
    } else {
        None
    };
    let files = Files { zip: zip.as_ref() };
    let mut scene = match Scene::from_document(&doc) {
        Ok(s) => s,
        Err(e) => return Outcome::Broken(format!("start.igz: {e:?}")),
    };
    let start_hash = match script.doc.get("start_sig") {
        Some(Value::Str(h)) => h.clone(),
        _ => return Outcome::Broken("no start_sig".into()),
    };
    if signature_hash(&model_sig::scene_signature(&scene, &files)) != start_hash {
        return Outcome::Start(vec!["(the model opens the start differently)".into()]);
    }
    let mut history = History::new();
    let mut cx = Cx::default();
    for (i, step) in steps.iter().enumerate() {
        let op = text(step, "op").to_string();
        let diverges = |why: String| Outcome::Diverges { at: i, op: op.clone(), why };
        match op.as_str() {
            "undo" | "redo" => {
                let got = if op == "undo" { history.undo(&mut scene, &mut cx) } else { history.redo(&mut scene, &mut cx) };
                let want = matches!(step.get("ok"), Some(Value::Bool(true)));
                if got != want {
                    return diverges(format!("{op} returned {got}, Python {want}"));
                }
            }
            _ => {
                let empty = Value::Object(Vec::new());
                let args = step.get("args").unwrap_or(&empty);
                let cmd = match build_command(&op, args, &scene) {
                    None => return Outcome::Unsupported { at: i, op: op.clone() },
                    Some(Err(e)) => return diverges(e),
                    Some(Ok(c)) => c,
                };
                if let Some(g) = counter(step, "group") {
                    cx.counters.group = g;
                }
                if let Some(c) = counter(step, "curve") {
                    cx.counters.curve = c;
                }
                history.execute(&mut scene, &mut cx, cmd);
                if history.last_rejected.is_some() != step.get("rejected").is_some() {
                    return diverges(format!("rejected: {:?}", history.last_rejected));
                }
                if let Some(e) = &history.last_error {
                    return diverges(e.clone());
                }
            }
        }
        let want = text(step, "sig");
        let got = signature_hash(&model_sig::scene_signature(&scene, &files));
        if got != want {
            return diverges(format!("signature differs after {op}"));
        }
    }
    // The runner also undoes everything: the start must come back exactly.
    while history.undo(&mut scene, &mut cx) {}
    let end_hash = signature_hash(&model_sig::scene_signature(&scene, &files));
    if end_hash != *script.undo_all.as_ref().unwrap_or(&start_hash) {
        return Outcome::Diverges { at: steps.len(), op: "undo_all".into(), why: "undoing every step does not restore the start".into() };
    }
    Outcome::Ok
}
