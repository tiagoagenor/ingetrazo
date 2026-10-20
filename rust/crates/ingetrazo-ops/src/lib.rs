// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! IngeTrazo's operations (docs/rust-rewrite-plan.md, M3): the `Command`s
//! of `core/history.py`, the group functions of `core/group.py` they use,
//! and the solid booleans (`core/solids.py`) through Manifold.

pub mod annotate;
pub mod command;
pub mod group;
pub mod group_cmds;
pub mod reference;
pub mod section_cmds;
pub mod solids;
pub mod straighten;
pub mod transform;
pub mod world;
