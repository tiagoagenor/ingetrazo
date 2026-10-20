// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! `core.orient.orient_outward`, as far as it is ported.
//!
//! What is here is the flat case: a drawing whose faces all lie on one
//! plane has no volume, no partitions and nothing to flip, so every face
//! gets `interior = False` and none is turned. A mesh with volume needs the
//! seeded ray-parity pass (`random.Random(12345)`, packed triangles) that is
//! not ported yet; [`orient_outward`] says so by returning `false`, and the
//! commands that run it (`heal_overlapping_faces` on a 3D mesh) report
//! `CommandError::Unsupported`.

use ingetrazo_geom::meshface;
use ingetrazo_math::{Hypot, V3};
use ingetrazo_model::mesh::Mesh;

/// `_all_coplanar`: every face on one plane (a 2D drawing).
pub fn all_coplanar(m: &Mesh, h: Hypot) -> bool {
    let pos = |f| m.face(f).outer.iter().map(|&v| m.position(v)).collect::<Vec<V3>>();
    let first = pos(m.faces[0]);
    let n = meshface::normal(&first, h);
    let o = meshface::centroid(&first);
    for &f in &m.faces {
        let lp = pos(f);
        let fnormal = meshface::normal(&lp, h);
        if (fnormal.dot(n) as f64).abs() < 0.999 {
            return false;
        }
        if (meshface::centroid(&lp).sub(o).dot(n) as f64).abs() > 1e-4 {
            return false;
        }
    }
    true
}

/// `orient_outward(mesh)` for the cases ported. Returns whether it could
/// run; `false` means the mesh has volume and the pass is not ported.
pub fn orient_outward(m: &mut Mesh, h: Hypot) -> bool {
    if m.faces.is_empty() {
        return true;
    }
    if all_coplanar(m, h) {
        for f in m.faces.clone() {
            if m.face(f).interior {
                m.face_mut(f).interior = false;
            }
        }
        return true;
    }
    false
}
