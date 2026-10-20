// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! IngeTrazo mesh interchange formats (docs/rust-rewrite-plan.md, M5.1, M5.2 and
//! M5.4): OBJ with its MTL, STL and COLLADA, ported from `formats/obj.py`,
//! `formats/stl.py` and `formats/dae.py` byte for byte.
//!
//! This is its own crate because the exporters walk the scene
//! (`ingetrazo-model`), and `ingetrazo-formats` must not depend on it.
//! Export reads the Rust `Scene`; import builds into it.

pub mod dae;
pub mod dae_in;
pub mod obj;
pub mod pyutil;
pub mod stl;
pub mod tri;
pub mod uv;
pub mod world;
pub mod xml;
