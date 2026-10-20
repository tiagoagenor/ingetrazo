// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! The golden corpus replays in Rust as far as the engine goes.
//!
//! * Every script's start document reads back to the L2 signature Python
//!   recorded; one whose start differs, or that cannot be read, is a failure.
//! * A script that stops at an op the engine lacks is fine (that is the work
//!   still ahead).
//! * `golden/replay_ok.txt` lists the scripts that replay to the end; each
//!   must keep doing so (CI blocks any change that turns a passing golden
//!   red). Regenerate it, after a milestone lands more, with
//!   `ingetrazo-cli golden replay --list-ok golden/replay_ok.txt`.

use std::collections::HashSet;
use std::path::Path;

use ingetrazo_golden::{ids, load, replay, Outcome};

#[test]
fn every_golden_script_starts_where_python_started() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../golden");
    let all = ids(&root);
    let pinned: HashSet<String> =
        std::fs::read_to_string(root.join("replay_ok.txt")).unwrap_or_default().lines().map(str::to_string).collect();
    let mut bad = Vec::new();
    let mut ok = 0;
    for id in &all {
        match load(&root, id).map(|s| replay(&s)) {
            Ok(Outcome::Ok) => ok += 1,
            Ok(Outcome::Unsupported { at, op }) => {
                if pinned.contains(id) {
                    bad.push(format!("{id}: used to replay to the end, now stops at step {at} ({op})"));
                }
            }
            Ok(Outcome::Diverges { at, op, why }) => {
                if pinned.contains(id) {
                    bad.push(format!("{id}: used to replay to the end, now diverges at step {at} ({op}): {why}"));
                }
            }
            Ok(other) => bad.push(format!("{id}: {other:?}")),
            Err(e) => bad.push(format!("{id}: {e}")),
        }
    }
    assert!(bad.is_empty(), "{} of {} scripts:\n{}", bad.len(), all.len(), bad.join("\n"));
    assert!(ok >= pinned.len(), "{ok} scripts replay to the end, {} are pinned", pinned.len());
}

/// The scripts whose every step the engine runs (M3.1 drawing, M3.5
/// transforms, M3.9 annotations, M3.10 groups/paint/purge; undo and redo)
/// replay to the end with Python's signatures; this number only goes up.
#[test]
fn scripts_of_the_ported_ops_replay_to_the_end() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../golden");
    let done = ids(&root).iter().filter(|id| matches!(load(&root, id).map(|s| replay(&s)), Ok(Outcome::Ok))).count();
    assert!(done >= 628, "only {done} scripts replay to the end");
}

/// The hand-written annotation scripts (scripts/golden/make_annotation_scripts.py)
/// carry the whole `document.json` Python wrote after every step, so they pin
/// what the L2 signature leaves out: where dimension ends read once the
/// geometry moved, every annotation's numbers, the datum and the sheets' world
/// anchors after Straighten Model.
#[test]
fn the_annotation_scripts_replay_with_python_documents() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../golden");
    let mut seen = 0;
    for id in ids(&root).iter().filter(|id| id.starts_with("hand-annotations-")) {
        let outcome = replay(&load(&root, id).expect("script"));
        assert_eq!(outcome, Outcome::Ok, "{id}");
        seen += 1;
    }
    assert!(seen >= 7, "only {seen} annotation scripts");
}
