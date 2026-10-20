// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! IngeTrazo's mesh interchange formats (docs/rust-rewrite-plan.md, M5.1 and
//! M5.2): OBJ with its MTL, and STL, ported from `formats/obj.py` and
//! `formats/stl.py` byte for byte.
//!
//! This is its own crate because the exporters walk the scene
//! (`ingetrazo-model`), and `ingetrazo-formats` must not depend on it.
//! Export reads the Rust `Scene`; import builds into it.

pub mod obj;
pub mod pyutil;
pub mod stl;
pub mod tri;
pub mod uv;
pub mod world;
