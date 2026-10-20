// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! Scale against the Python app (scripts/golden/make_modify_tool_fixtures.py,
//! scripts/golden/modify_scale.py): the REAL `tools.scale.ScaleTool` driven by
//! real pointer and key events through a real camera - the grip box on loose
//! geometry, groups and instances (on their own axes), corner/edge/face
//! grips, click-move-click and drag-release, About Center and Scale Uniformly
//! taps, typed factors and absolute sizes, the hot retype, snapping a held
//! grip, click-to-select, Esc, undo and redo - and the Rust tool must leave
//! the whole state the same after every event (see `tests/common/mod.rs`).

mod common;

use ingetrazo_tools::scale::ScaleTool;

#[test]
fn scale_programs_match_python() {
    let stats = common::run_cases("scale", &|| Box::new(ScaleTool::new()));
    eprintln!("scale: {stats:?}");
    assert!(stats.programs >= 15);
}
