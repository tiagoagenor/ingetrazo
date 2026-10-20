// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! Protractor against the Python app (scripts/golden/make_modify_tool_fixtures.py,
//! scripts/golden/modify_protractor.py): the REAL `tools.protractor.ProtractorTool`
//! driven by real pointer and key events through a real camera - plane inference
//! on faces and on the ground, arrow-key locks, Shift freeze, click-drag axis with
//! the live tilt, the 15 degree ticks and the free 0.1 degree, the exact-snap rule,
//! guides, typed angles and slopes, the hot retype, measure-only mode, Esc, undo
//! and redo - and the Rust tool must leave the whole state the same after every
//! event (see `tests/common/mod.rs`).

mod common;

use ingetrazo_tools::protractor::ProtractorTool;

#[test]
fn protractor_programs_match_python() {
    let stats = common::run_cases("protractor", &|| Box::new(ProtractorTool::new()));
    eprintln!("protractor: {stats:?}");
    assert!(stats.programs >= 15);
}
