// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! Offset against the Python app (scripts/golden/make_modify_tool_fixtures.py,
//! scripts/golden/modify_offset.py): the REAL `tools.offset.OffsetTool` driven by
//! real pointer and key events through a real camera - faces, edges, rings,
//! curves, typed distances, the hot retype, Edges mode - and the Rust tool must
//! leave the whole state the same after every event (see `tests/common/mod.rs`).

mod common;

use ingetrazo_tools::offset::OffsetTool;

#[test]
fn offset_programs_match_python() {
    let stats = common::run_cases("offset", &|| Box::new(OffsetTool::new()));
    eprintln!("offset: {stats:?}");
    assert!(stats.programs >= 15);
}
