// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! IngeTrazo's Rust application (docs/rust-rewrite-plan.md, M4.3 and M4.4):
//! the egui window, the viewport drawn by `ingetrazo-render`, and the camera
//! navigation, a port of what `views/viewport.py` does with the mouse.
//!
//! * [`nav`]: orbit, pan, zoom, zoom extents and the gesture state machine,
//!   bit for bit the Python camera (`tests/nav_parity.rs`).
//! * [`probe`]: what the scene tells a gesture (the point under the cursor).
//! * [`quat`]: `QQuaternion`, for the orbit about a pivot.
//! * [`viewport`]: the renderer in an egui paint callback.
//! * [`document`]: the open file.
//! * [`app`]: the window.

pub mod app;
pub mod document;
pub mod nav;
pub mod probe;
pub mod quat;
pub mod viewport;
