// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! IngeTrazo's tools (docs/rust-rewrite-plan.md, section 2): tool state
//! machines with no UI code. A tool takes the camera ray of the pointer,
//! modifier keys and typed text; it answers with preview geometry and the
//! `Command` to execute, and is tested headless.
//!
//! * [`pushpull`] and its [`view`] / [`pick`] (M3.4).
//! * The drawing tools (M3.13): [`line`], [`rectangle`], ... on the shared
//!   [`base`] (`tools/base.py`), driven by a [`host`] that plays the viewport
//!   (pointer, snap, keys, value box) and reads typed values with [`vcb`].

pub mod arc;
pub mod base;
pub mod circle;
pub mod freehand;
pub mod host;
pub mod line;
pub mod pick;
pub mod pushpull;
pub mod rectangle;
pub mod rotated_rectangle;
pub mod units;
pub mod vcb;
pub mod view;
