// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! Move against the Python app (scripts/golden/make_modify_tool_fixtures.py,
//! scripts/golden/modify_move.py): the REAL `tools.move.MoveTool` driven by
//! real pointer and key events through a real camera - grabs of faces, edges,
//! corners and groups, the live preview, drops, typed distances and 3D
//! deltas, Ctrl copies and their `3x` `/3` `5x2m` arrays, Esc, undo and redo
//! mid-tool, welds - and the Rust tool must leave the whole state the same
//! after every event (see `tests/common/mod.rs`).

mod common;

use ingetrazo_tools::move_tool::MoveTool;

#[test]
fn move_programs_match_python() {
    let stats = common::run_cases("move", &|| Box::new(MoveTool::new()));
    eprintln!("move: {stats:?}");
    assert!(stats.programs >= 15);
}
