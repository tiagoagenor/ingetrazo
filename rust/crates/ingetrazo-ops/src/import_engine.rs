// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! The edit engine's part in the mesh imports (M5): what `formats/obj.py`,
//! `formats/stl.py`, `formats/dae.py` and `formats/dxf_in.py` run after they
//! have added their faces.
//!
//! The import code lives in `ingetrazo-formats-mesh`, which this crate
//! depends on (the operations open and save files), so it cannot call
//! `run_stitch` or `orient_outward` itself: it takes hooks
//! ([`ingetrazo_formats_mesh::obj::Stitch`] and
//! [`ingetrazo_formats_mesh::stl::ImportEngine`]) and this module is what
//! the application hands them. Moving `orient_outward` down to the model
//! crate was the other way round; it needs the ray index of
//! `ingetrazo-geom`, the 2D arrangement helpers and the Python `random`
//! twin, all of this crate, so the hook stays on this side of the arrow.

use std::collections::HashSet;

use ingetrazo_formats_mesh::stl::ImportEngine;
use ingetrazo_math::Hypot;
use ingetrazo_model::mesh::{FaceId, Key, Mesh};
use ingetrazo_model::stitch::run_stitch;

use crate::orient::{orient_outward, orient_outward_with};

/// `run_stitch(mesh, seed, new_faces, coplanar_merge=True)`.
fn stitch(mesh: &mut Mesh, seed: &HashSet<Key>, new_faces: &mut HashSet<FaceId>, h: Hypot) {
    let seed: HashSet<[i64; 3]> = seed.iter().map(|k| [k.0, k.1, k.2]).collect();
    run_stitch(mesh, &seed, Some(new_faces), true, true, h);
}

/// The hook of the OBJ and COLLADA imports: `run_stitch(..., coplanar_merge=
/// True)` and then `orient_outward(mesh)`.
pub fn stitch_and_orient(h: Hypot) -> impl FnMut(&mut Mesh, &HashSet<Key>, &HashSet<FaceId>) {
    move |mesh, seed, new_faces| {
        stitch(mesh, seed, &mut new_faces.clone(), h);
        orient_outward(mesh, h);
    }
}

/// The hook of the DXF import: `run_stitch(..., coplanar_merge=True)` alone.
pub fn stitch_only(h: Hypot) -> impl FnMut(&mut Mesh, &HashSet<Key>, &HashSet<FaceId>) {
    move |mesh, seed, new_faces| stitch(mesh, seed, &mut new_faces.clone(), h)
}

/// The engine the STL import asks for.
#[derive(Clone, Copy, Debug)]
pub struct Engine {
    pub hypot: Hypot,
}

impl ImportEngine for Engine {
    fn run_stitch(&mut self, mesh: &mut Mesh, seed: &HashSet<Key>, new_faces: &mut HashSet<FaceId>) {
        stitch(mesh, seed, new_faces, self.hypot);
    }

    fn orient_outward(&mut self, mesh: &mut Mesh, only: Option<&[FaceId]>) {
        orient_outward_with(mesh, self.hypot, 12345, only);
    }
}
