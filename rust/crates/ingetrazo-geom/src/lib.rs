// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! IngeTrazo's geometry kernels, with no Python in them.
//!
//! Every kernel here mirrors a Python function line by line — the same
//! arithmetic in the same order — so both give the same drawing. The only
//! thing a kernel may add is an index that finds the SAME candidates faster.

pub mod arrange;
pub mod bevel_profiles;
pub mod boxes;
pub mod conform;
pub mod facegeom;
pub mod fillet;
pub mod fuse;
pub mod gltape;
pub mod hlr;
pub mod hlr_draw;
pub mod instances;
pub mod intersect;
pub mod meshface;
pub mod occlusion;
pub mod offset;
pub mod parity;
pub mod plan;
pub mod section;
pub mod simplify;
pub mod sweep;
pub mod tiff;
pub mod triangulate;
pub mod vertex_bevel;
pub mod volume;
pub mod weld;
