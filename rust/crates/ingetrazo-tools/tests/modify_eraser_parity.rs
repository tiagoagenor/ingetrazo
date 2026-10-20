// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! The Eraser against the Python app (scripts/golden/make_modify_tool_fixtures.py,
//! scripts/golden/modify_eraser.py): the REAL `tools.eraser.EraserTool` driven by
//! real pointer events - clicks and strokes over edges, curves, groups, guides,
//! dimensions, leader texts and paths, Shift hide strokes, Esc, undo and redo -
//! and the Rust tool must leave the whole state the same after every event.

mod common;

use std::collections::BTreeMap;

use ingetrazo_formats::pyjson::Value;
use ingetrazo_snap::picking::TextMetrics;
use ingetrazo_tools::eraser::EraserTool;

/// The text metrics Qt gave the Python viewport for the scene's labels.
struct Metrics {
    height: i32,
    ascent: i32,
    descent: i32,
    adv: BTreeMap<String, i32>,
}

impl TextMetrics for Metrics {
    fn height(&self) -> i32 {
        self.height
    }
    fn ascent(&self) -> i32 {
        self.ascent
    }
    fn descent(&self) -> i32 {
        self.descent
    }
    fn advance(&self, line: &str) -> i32 {
        *self.adv.get(line).unwrap_or_else(|| panic!("no advance recorded for {line:?}"))
    }
}

fn metrics(v: &Value) -> Metrics {
    let int = |k: &str| v.get(k).unwrap().as_i64().unwrap() as i32;
    let adv = match v.get("adv").unwrap() {
        Value::Object(items) => items.iter().map(|(k, v)| (k.clone(), v.as_i64().unwrap() as i32)).collect(),
        other => panic!("{other:?}"),
    };
    Metrics { height: int("height"), ascent: int("ascent"), descent: int("descent"), adv }
}

#[test]
fn eraser_programs_match_python() {
    let stats = common::run_cases_with("eraser", &|case, rig| {
        rig.vp.metrics = Box::new(metrics(case.get("setup").unwrap().get("metrics").unwrap()));
        Box::new(EraserTool::new())
    });
    eprintln!("eraser: {stats:?}");
    assert!(stats.programs >= 15);
}
