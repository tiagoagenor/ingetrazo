// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! Every script of the golden corpus replays in Rust at least to its first
//! step: the start document reads back to the L2 signature Python recorded.
//! A script that stops at an op the engine lacks is fine (that is the work
//! still ahead); one whose start differs, or that cannot be read, is not.

use std::path::Path;

use ingetrazo_golden::{ids, load, replay, Outcome};

#[test]
fn every_golden_script_starts_where_python_started() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../golden");
    let all = ids(&root);
    let mut bad = Vec::new();
    for id in &all {
        match load(&root, id).map(|s| replay(&s)) {
            Ok(Outcome::Ok) | Ok(Outcome::Unsupported { .. }) => {}
            Ok(other) => bad.push(format!("{id}: {other:?}")),
            Err(e) => bad.push(format!("{id}: {e}")),
        }
    }
    assert!(bad.is_empty(), "{} of {} scripts:\n{}", bad.len(), all.len(), bad.join("\n"));
}

/// The scripts whose every step the engine runs (M3.5 the transform commands,
/// M3.10 groups, components, paint and purge; undo and redo) replay to the end
/// with Python's signatures; this
/// number only goes up as ops land.
#[test]
fn scripts_of_the_ported_ops_replay_to_the_end() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../golden");
    let done = ids(&root).iter().filter(|id| matches!(load(&root, id).map(|s| replay(&s)), Ok(Outcome::Ok))).count();
    assert!(done >= 143, "only {done} scripts replay to the end");
}
