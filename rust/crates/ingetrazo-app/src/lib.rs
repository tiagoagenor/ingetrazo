// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! IngeTrazo's Rust application (docs/rust-rewrite-plan.md, M4.3 to M4.8):
//! the egui window, the viewport drawn by `ingetrazo-render`, the camera
//! navigation, the tools and the file.
//!
//! * [`nav`]: orbit, pan, zoom, zoom extents and the gesture state machine,
//!   bit for bit the Python camera (`tests/nav_parity.rs`).
//! * [`probe`]: what the scene tells a gesture (the point under the cursor).
//! * [`quat`]: `QQuaternion`, for the orbit about a pivot.
//! * [`viewport`]: the renderer in an egui paint callback.
//! * [`document`]: the open file, and writing it back.
//! * [`toolbox`], [`adapters`]: the tools of the window, behind the one host
//!   of `ingetrazo-tools`.
//! * [`session`]: the document, the camera and the tools with no window, fed
//!   by [`session::Input`] events; [`input`] turns egui's events into them.
//! * [`app`]: the window.

pub mod adapters;
pub mod app;
pub mod bench;
pub mod document;
pub mod input;
pub mod nav;
pub mod paths;
pub mod probe;
pub mod quat;
pub mod session;
pub mod toolbox;
pub mod viewport;
