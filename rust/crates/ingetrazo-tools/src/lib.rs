// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! IngeTrazo's tools (docs/rust-rewrite-plan.md, section 2): tool state
//! machines with no UI code. A tool takes the camera ray of the pointer,
//! modifier keys and typed text; it answers with preview geometry and the
//! `Command` to execute, and is tested headless.

// The tools' `probe` fills a `Probe::default()` field by field, the way the
// Python tools answer one method at a time.
#![allow(clippy::field_reassign_with_default)]

pub mod pick;
pub mod pushpull;
pub mod view;

// The MODIFY and MEASURE tools (M3.13): what they share, then one per tool.
pub mod modify_base;
pub mod modify_viewport;

pub mod eraser;
pub mod flip;
pub mod move_tool;
pub mod offset;
pub mod paint;
pub mod protractor;
pub mod rotate;
pub mod scale;
pub mod tape;
