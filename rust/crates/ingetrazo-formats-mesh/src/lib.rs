// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! IngeTrazo's mesh interchange formats (docs/rust-rewrite-plan.md, M5.1 to
//! M5.5): OBJ with its MTL, STL, COLLADA, glTF/GLB and DXF, ported from
//! `formats/obj.py`, `formats/stl.py`, `formats/dae.py`, `formats/gltf.py`,
//! `formats/glb.py`, `formats/dxf_in.py` and `formats/dxf_out.py` byte for byte;
//! M5.6 adds the DWG bridge (`formats/dwg_bridge.py`, [`dwg`]).
//!
//! This is its own crate because the exporters walk the scene
//! (`ingetrazo-model`), and `ingetrazo-formats` must not depend on it.
//! Export reads the Rust `Scene`; import builds into it.

pub mod dae;
pub mod dae_in;
pub mod dwg;
pub mod dxf;
pub mod figure;
pub mod finish;
pub mod fusing;
pub mod glb;
pub mod gltf;
pub mod img;
pub mod obj;
pub mod pyutil;
pub mod stl;
pub mod tri;
pub mod uv;
pub mod world;
pub mod xml;
