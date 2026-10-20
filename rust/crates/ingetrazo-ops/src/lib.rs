// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! IngeTrazo's operations (docs/rust-rewrite-plan.md, M3): commands and
//! undo, the edits planner, the drawing commands. Push/pull, transforms and
//! solids join here.

pub mod command;
pub mod draw;
pub mod edits;
pub mod orient;
pub mod rebuild;

pub use command::{CmdResult, Command, CommandError, Env, History};
