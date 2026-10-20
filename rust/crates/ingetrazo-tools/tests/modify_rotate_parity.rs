// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! Rotate against the Python app (scripts/golden/make_modify_tool_fixtures.py,
//! scripts/golden/modify_rotate.py): the REAL `tools.rotate.RotateTool` driven
//! by real pointer and key events through a real camera - the protractor disc
//! inferred from faces, arrow locks, Shift freeze, the click-drag axis with its
//! live tilt, the live swing with ticks and free angles, typed angles and
//! slopes, the hot retype, Ctrl copies and their polar arrays, loose faces,
//! edges, curves, classic groups, instances and containers, Esc, undo and redo
//! mid-tool - and the Rust tool must leave the whole state the same after every
//! event (see `tests/common/mod.rs`).

mod common;

use ingetrazo_tools::rotate::RotateTool;

#[test]
fn rotate_programs_match_python() {
    let stats = common::run_cases("rotate", &|| Box::new(RotateTool::new()));
    eprintln!("rotate: {stats:?}");
    assert!(stats.programs >= 15);
}
