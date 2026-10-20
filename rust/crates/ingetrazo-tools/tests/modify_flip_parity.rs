// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! Flip against the Python app (scripts/golden/make_modify_tool_fixtures.py,
//! scripts/golden/modify_flip.py): the REAL `tools.flip.FlipTool` driven by
//! real pointer and key events through a real camera - the plane under the
//! cursor, the arrow-key picks, Ctrl-tap copies of faces, edges (curves,
//! soft) and groups, flips of loose geometry and groups, undo and redo - and
//! the Rust tool must leave the whole state the same after every event (see
//! `tests/common/mod.rs`).

mod common;

use ingetrazo_tools::flip::FlipTool;

#[test]
fn flip_programs_match_python() {
    let stats = common::run_cases("flip", &|| Box::new(FlipTool::new()));
    eprintln!("flip: {stats:?}");
    assert!(stats.programs >= 15);
}
