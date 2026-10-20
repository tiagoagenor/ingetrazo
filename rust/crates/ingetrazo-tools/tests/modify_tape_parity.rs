// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! The Tape Measure against the Python app (scripts/golden/make_modify_tool_fixtures.py,
//! scripts/golden/modify_tape.py): the REAL `tools.tape.TapeMeasureTool` driven by
//! real pointer and key events - measures, guide lines pulled off loose, group,
//! guide and axis sources, guide points from named points, typed distances, the
//! three Ctrl modes, Esc, undo and redo, the document's units - and the Rust
//! tool must leave the whole state the same after every event.

mod common;

use ingetrazo_model::units::ModelUnits;
use ingetrazo_tools::tape::TapeMeasureTool;

#[test]
fn tape_programs_match_python() {
    let stats = common::run_cases_with("tape", &|_case, rig| {
        rig.units = ModelUnits::of(rig.scene.extra.units.as_ref());
        Box::new(TapeMeasureTool::new())
    });
    eprintln!("tape: {stats:?}");
    assert!(stats.programs >= 15);
}
