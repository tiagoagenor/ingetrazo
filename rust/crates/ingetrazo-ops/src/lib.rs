// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! IngeTrazo's operations (docs/rust-rewrite-plan.md, M3): the `Command`s
//! of `core/history.py`, the group functions of `core/group.py` they use,
//! and the solid booleans (`core/solids.py`) through Manifold.

pub mod annotate;
pub mod arrangement;
pub mod command;
pub mod describe;
pub mod draw;
pub mod followme;
pub mod edit_cmds;
pub mod edits;
pub mod group;
pub mod group_cmds;
pub mod grouping;
pub mod intersect;
pub mod orient;
pub mod pyrandom;
pub mod pyset;
pub mod rebuild;
pub mod reference;
pub mod section_cmds;
pub mod solids;
pub mod straighten;
pub mod transform;
pub mod world;
