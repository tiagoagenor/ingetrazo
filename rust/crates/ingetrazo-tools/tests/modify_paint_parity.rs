// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! Paint against the Python app (scripts/golden/make_modify_tool_fixtures.py,
//! scripts/golden/modify_paint.py): the REAL `tools.paint.PaintTool` driven by
//! real pointer and key events - colour, texture (positioned, rotated,
//! continuous over a curved surface), selections, front and back sides, the
//! default material, groups painted as a whole, the Alt / armed eyedropper and
//! the recent row - each program starting from its own class-level paint state
//! (recorded in the case's `setup`), and the Rust tool must leave the whole
//! state the same after every event.

mod common;

use ingetrazo_formats::pyjson::Value;
use ingetrazo_math::V3;
use ingetrazo_tools::paint::{PaintMaterial, PaintState, PaintTool};

fn state_of(init: &Value) -> PaintState {
    let mut st = PaintState::default();
    let nn = |k: &str| init.get(k).filter(|v| !matches!(v, Value::Null)).cloned();
    if let Some(Value::Array(c)) = init.get("color") {
        st.color = c.clone();
    }
    st.texture = nn("texture");
    st.material = nn("material").map(|d| PaintMaterial { name: common::text(d.get("name").unwrap()).to_string(), dict: d });
    st.opacity = nn("opacity");
    st.sample_armed = common::is_true(init.get("armed"));
    st.texture_plane = nn("plane").map(|p| {
        let n = common::arr(p.get("n").unwrap());
        (V3(common::num(&n[0]) as f32, common::num(&n[1]) as f32, common::num(&n[2]) as f32), common::num(p.get("d").unwrap()))
    });
    st.is_default = common::is_true(init.get("default"));
    if let Some(Value::Array(r)) = init.get("recent") {
        st.recent = r.clone();
    }
    st
}

#[test]
fn paint_programs_match_python() {
    let stats = common::run_cases_with("paint", &|case, _rig| {
        let init = case.get("setup").unwrap().get("init").unwrap();
        Box::new(PaintTool::with_state(state_of(init)))
    });
    eprintln!("paint: {stats:?}");
    assert!(stats.programs >= 15);
}
