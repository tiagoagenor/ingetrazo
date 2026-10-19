// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! Every start scene of the golden corpus opened into the Rust model
//! (`Scene::from_document`) gives the L2 signature Python's live scene had.

use std::path::Path;

use ingetrazo_formats::pyjson::Value;
use ingetrazo_golden::{ids, load, model_signature, sig};

#[test]
fn rust_model_opens_every_start_scene_as_python_had_it() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../golden");
    let mut bad = Vec::new();
    let mut seen = std::collections::HashSet::new();
    let all = ids(&root);
    for id in &all {
        let s = load(&root, id).unwrap();
        let file = match s.doc.get("start").and_then(|x| x.get("igz")) {
            Some(Value::Str(f)) => f.clone(),
            _ => continue,
        };
        if !seen.insert(file.clone()) {
            continue;
        }
        let want = match s.doc.get("start_sig") {
            Some(Value::Str(h)) => h.clone(),
            _ => continue,
        };
        match model_signature(&s.start) {
            Ok(got) => {
                if sig::signature_hash(&got) != want {
                    let parts: Vec<String> = match (&got, &s.start_sig) {
                        (Value::Object(g), Some(w)) => g
                            .iter()
                            .filter(|(k, v)| w.get(k).map(sig::canon_dumps) != Some(sig::canon_dumps(v)))
                            .map(|(k, _)| k.clone())
                            .collect(),
                        _ => vec![],
                    };
                    bad.push(format!("{file}: {}", parts.join(",")));
                }
            }
            Err(e) => bad.push(format!("{file}: {e}")),
        }
    }
    assert!(bad.is_empty(), "{} of {} start scenes differ:\n{}", bad.len(), seen.len(), bad.iter().take(15).cloned().collect::<Vec<_>>().join("\n"));
}
