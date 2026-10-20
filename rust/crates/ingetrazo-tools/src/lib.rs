// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! IngeTrazo's tools (docs/rust-rewrite-plan.md, section 2): tool state
//! machines with no UI code. A tool takes the camera ray of the pointer,
//! modifier keys and typed text; it answers with preview geometry and the
//! `Command` to execute, and is tested headless.

pub mod pick;
pub mod pushpull;
pub mod view;
