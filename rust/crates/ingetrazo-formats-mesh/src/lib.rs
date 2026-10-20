// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! IngeTrazo's mesh interchange formats (docs/rust-rewrite-plan.md, M5.1 to
//! M5.3): OBJ with its MTL, STL, and glTF/GLB, ported from `formats/obj.py`,
//! `formats/stl.py`, `formats/gltf.py` and `formats/glb.py` byte for byte.
//!
//! This is its own crate because the exporters walk the scene
//! (`ingetrazo-model`), and `ingetrazo-formats` must not depend on it.
//! Export reads the Rust `Scene`; import builds into it.

pub mod figure;
pub mod finish;
pub mod glb;
pub mod gltf;
pub mod img;
pub mod obj;
pub mod pyutil;
pub mod stl;
pub mod tri;
pub mod uv;
pub mod world;
