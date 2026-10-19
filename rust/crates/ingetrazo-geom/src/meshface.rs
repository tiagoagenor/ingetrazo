// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! What `core.mesh.Face` computes from its loop's positions (`normal`,
//! `centroid`) and `core.topology._key`, from plain arrays.
//!
//! `Face._newell` adds in DOUBLE precision and rounds once into a
//! `QVector3D`; the normal is that vector divided (not `normalized()`) by
//! its Qt length. `_key` is `round(x, 4)` of the float32 coordinate, which
//! for a float32 is exact as `round_ties_even(x · 1e4) / 1e4` (the product
//! has at most 38 significant bits, so it is exact in f64 and its ties are
//! real ties; the quotient is the correctly rounded decimal, as
//! `float.__round__` gives).

use ingetrazo_math::pyfloat::py_sum;
use ingetrazo_math::{Hypot, V3};

/// `Face._newell()` of a loop of positions.
pub fn newell(loop_: &[V3]) -> V3 {
    let count = loop_.len();
    let (mut nx, mut ny, mut nz) = (0.0f64, 0.0f64, 0.0f64);
    for i in 0..count {
        let c = loop_[i];
        let n = loop_[(i + 1) % count];
        let (x0, y0, z0) = (c.0 as f64, c.1 as f64, c.2 as f64);
        let (x1, y1, z1) = (n.0 as f64, n.1 as f64, n.2 as f64);
        nx += (y0 - y1) * (z0 + z1);
        ny += (z0 - z1) * (x0 + x1);
        nz += (x0 - x1) * (y0 + y1);
    }
    V3(nx as f32, ny as f32, nz as f32)
}

/// `Face.normal()`: (0, 0, 1) for fewer than three corners or a Newell
/// vector shorter than 1e-9, else the Newell vector divided by its length.
pub fn normal(loop_: &[V3], h: Hypot) -> V3 {
    if loop_.len() < 3 {
        return V3(0.0, 0.0, 1.0);
    }
    let n = newell(loop_);
    let length = n.length(h);
    if (length as f64) < 1e-9 {
        return V3(0.0, 0.0, 1.0);
    }
    n.div(length)
}

/// `Face.centroid()`: the mean of the corners, summed as Python floats
/// (compensated sum, 3.12+) and rounded to float32 once.
pub fn centroid(loop_: &[V3]) -> V3 {
    let count = loop_.len();
    if count == 0 {
        return V3(0.0, 0.0, 0.0);
    }
    let c = count as f64;
    let cx = py_sum(loop_.iter().map(|p| p.0 as f64), true) / c;
    let cy = py_sum(loop_.iter().map(|p| p.1 as f64), true) / c;
    let cz = py_sum(loop_.iter().map(|p| p.2 as f64), true) / c;
    V3(cx as f32, cy as f32, cz as f32)
}

/// `round(x, 4)` of a float32 coordinate, as the float `_key` holds (−0
/// folded into +0: tuples of floats compare equal either way).
pub fn round4(x: f32) -> f64 {
    let k = (x as f64 * 1e4).round_ties_even();
    let v = k / 1e4;
    if v == 0.0 {
        0.0
    } else {
        v
    }
}

/// `core.topology._key(p)`.
pub fn key(p: V3) -> [f64; 3] {
    [round4(p.0), round4(p.1), round4(p.2)]
}

/// The same key as three exact integers (`round_ties_even(x · 1e4)`): two
/// points have equal `_key`s exactly when these are equal.
pub fn ikey(p: V3) -> [i64; 3] {
    let r = |x: f32| (x as f64 * 1e4).round_ties_even() as i64;
    [r(p.0), r(p.1), r(p.2)]
}
