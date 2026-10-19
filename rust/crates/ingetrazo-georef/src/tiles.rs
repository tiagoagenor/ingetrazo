// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! The slippy-map math of `georef/tiles.py` that the DEM sampler uses.

use ingetrazo_math::pyfloat::{py_max2, py_min2};

/// `2 ** zoom` as the float it becomes in `... * n` (exact for any power of
/// two, negative exponents included, as Python's int/float `**`).
fn two_pow(zoom: i64) -> f64 {
    2.0f64.powi(zoom as i32)
}

/// `deg2num`: geodetic degrees to fractional tile coordinates at `zoom`.
pub fn deg2num(lat_deg: f64, lon_deg: f64, zoom: i64) -> (f64, f64) {
    let lat_deg = py_max2(-85.05112878, py_min2(85.05112878, lat_deg));
    let lat_rad = lat_deg.to_radians();
    let n = two_pow(zoom);
    let xtile = (lon_deg + 180.0) / 360.0 * n;
    let ytile = (1.0 - lat_rad.tan().asinh() / std::f64::consts::PI) / 2.0 * n;
    (xtile, ytile)
}

/// `tiles_covering`: every tile `(x, y)` overlapping the geodetic bbox.
pub fn tiles_covering(
    lat_s: f64,
    lon_w: f64,
    lat_n: f64,
    lon_e: f64,
    zoom: i64,
) -> Vec<(i64, i64)> {
    let (x0, y0) = deg2num(lat_n, lon_w, zoom);
    let (x1, y1) = deg2num(lat_s, lon_e, zoom);
    let n: i64 = 1i64 << zoom;
    let xa = py_min2(x0, x1).floor() as i64;
    let xb = py_max2(x0, x1).floor() as i64;
    let ya = py_min2(y0, y1).floor() as i64;
    let yb = py_max2(y0, y1).floor() as i64;
    let mut out = Vec::new();
    for tx in xa..=xb {
        for ty in ya..=yb {
            if 0 <= tx && tx < n && 0 <= ty && ty < n {
                out.push((tx, ty));
            }
        }
    }
    out
}
