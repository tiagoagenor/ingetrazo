// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! The golden corpus, Rust side (golden/SCHEMA.md): read the op scripts
//! Python harvested and replay them as far as the Rust engine goes,
//! reporting the first point where it parts from Python.
//!
//! The start document is read and its L2 signature compared, part by part,
//! with the one Python recorded; the steps then run through `replay.rs` (the
//! ops `ingetrazo-ops` has: drawing, erasing, the plane rebuilds) with the
//! signature compared after each one, and stop at the first op the Rust
//! engine does not run yet. Each milestone that lands an op moves that
//! point further on.

pub mod model_sig;
pub mod replay;
pub mod sig;

use std::path::{Path, PathBuf};

use ingetrazo_formats::igz::{self, Zip};
use ingetrazo_formats::pyjson::{self, Value};

/// One script of the corpus.
pub struct Script {
    pub id: String,
    pub doc: Value,
    pub start: Vec<u8>,
    pub start_sig: Option<Value>,
}

/// Where a replay stopped.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Outcome {
    /// Every step ran and matched.
    Ok,
    /// The start document's signature differs; the parts that differ.
    Start(Vec<String>),
    /// Step `at` is an op the Rust engine does not run yet.
    Unsupported { at: usize, op: String },
    /// The script could not be read.
    Broken(String),
    /// Step `at` ran and parted from Python: `what` says how.
    Diverged { at: usize, op: String, what: String },
}

pub fn ids(root: &Path) -> Vec<String> {
    let mut out: Vec<String> = std::fs::read_dir(root.join("ops"))
        .map(|d| {
            d.filter_map(|e| e.ok())
                .filter_map(|e| e.path().file_stem().map(|s| s.to_string_lossy().into_owned()))
                .collect()
        })
        .unwrap_or_default();
    out.sort();
    out
}

pub fn load(root: &Path, id: &str) -> Result<Script, String> {
    let text = std::fs::read_to_string(root.join("ops").join(format!("{id}.json"))).map_err(|e| e.to_string())?;
    let doc = pyjson::loads(&text).map_err(|e| e.to_string())?;
    if doc.get("schema").and_then(Value::as_i64) != Some(1) {
        return Err("not schema 1".into());
    }
    let file = match doc.get("start").and_then(|s| s.get("igz")) {
        Some(Value::Str(f)) => f.clone(),
        _ => return Err("start is not an .igz of the repository".into()),
    };
    // Paths are relative to the repository, which holds golden/.
    let repo: PathBuf = root.join("..");
    let start = std::fs::read(repo.join(&file)).map_err(|e| format!("{file}: {e}"))?;
    let sig_file = repo.join(file.trim_end_matches(".igz").to_string() + ".sig.json");
    let start_sig = std::fs::read_to_string(sig_file).ok().and_then(|t| pyjson::loads(&t).ok());
    Ok(Script { id: id.to_string(), doc, start, start_sig })
}

/// The L2 signature of an .igz file's document.
pub fn file_signature(raw: &[u8]) -> Result<Value, String> {
    let doc = igz::read_document(raw).map_err(|e| e.to_string())?;
    let zip = if igz::is_container(raw) { Some(Zip::open(raw).map_err(|e| e.to_string())?) } else { None };
    sig::scene_signature(&doc, &sig::Files { zip: zip.as_ref() })
}

/// Replay `script` as far as the Rust engine goes.
pub fn replay(script: &Script) -> Outcome {
    let got = match file_signature(&script.start) {
        Ok(s) => s,
        Err(e) => return Outcome::Broken(format!("start.igz: {e}")),
    };
    let want_hash = match script.doc.get("start_sig") {
        Some(Value::Str(h)) => h.clone(),
        _ => return Outcome::Broken("no start_sig".into()),
    };
    if sig::signature_hash(&got) != want_hash {
        let parts = match (&got, &script.start_sig) {
            (Value::Object(g), Some(w)) => g
                .iter()
                .filter(|(k, v)| w.get(k).map(sig::canon_dumps) != Some(sig::canon_dumps(v)))
                .map(|(k, _)| k.clone())
                .collect(),
            _ => vec!["(no start.sig.json to compare parts)".to_string()],
        };
        return Outcome::Start(parts);
    }
    match script.doc.get("steps") {
        Some(Value::Array(_)) => {
            // A panic in the engine is a divergence of that script, not the
            // end of the run.
            match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| replay::run_steps(script))) {
                Ok(o) => o,
                Err(p) => {
                    let msg = p
                        .downcast_ref::<String>()
                        .cloned()
                        .or_else(|| p.downcast_ref::<&str>().map(|s| s.to_string()))
                        .unwrap_or_default();
                    Outcome::Diverged { at: 0, op: "?".into(), what: format!("panic: {msg}") }
                }
            }
        }
        _ => Outcome::Broken("no steps".into()),
    }
}

/// The L2 signature of an .igz opened into the Rust model (`Scene`),
/// rather than computed from the document.
pub fn model_signature(raw: &[u8]) -> Result<Value, String> {
    let doc = igz::read_document(raw).map_err(|e| e.to_string())?;
    let zip = if igz::is_container(raw) { Some(Zip::open(raw).map_err(|e| e.to_string())?) } else { None };
    let scene = ingetrazo_model::scene::Scene::from_document(&doc).map_err(|e| format!("{e:?}"))?;
    Ok(model_sig::scene_signature(&scene, &sig::Files { zip: zip.as_ref() }))
}
