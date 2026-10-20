// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! `georef/terrain.py`: DEM relief as a regular grid of local-metre vertices
//! with UVs into a tile mosaic. Display-only, never part of `Scene.mesh`.
//!
//! [`build_terrain`] is the grid geometry as plain arrays; [`mosaic_plan`] is
//! the compositing math of `build_mosaic` (where each tile lands in the
//! mosaic): the pixels themselves are the app's `QImage`/GPU work.

use ingetrazo_math::pyfloat::{py_max2, py_min2};
use ingetrazo_math::V3;

use crate::datum::SceneDatum;
use crate::dem::ElevationSampler;
use crate::tiles::{deg2num, tiles_covering};

pub const TILE_PX: i64 = 256;

/// `TerrainObject`.
#[derive(Clone, Debug, PartialEq)]
pub struct TerrainObject {
    /// Local metres, row 0 = north edge.
    pub vertices: Vec<V3>,
    /// `(u, v)` into the mosaic.
    pub uvs: Vec<(f64, f64)>,
    pub triangles: Vec<(usize, usize, usize)>,
    /// `(tx0, ty0, ntx, nty, zoom)`.
    pub tile_range: (i64, i64, i64, i64, i64),
    /// Grid columns (east).
    pub nx: i64,
    /// Grid rows (north).
    pub ny: i64,
    /// `(minx, miny, maxx, maxy)` local metres.
    pub bbox: (f64, f64, f64, f64),
    pub visible: bool,
}

impl TerrainObject {
    /// `height_at`: bilinearly interpolated terrain Z at local `(x, y)`, or
    /// `None` outside the captured area.
    pub fn height_at(&self, x: f64, y: f64) -> Option<f64> {
        let (nx, ny) = (self.nx, self.ny);
        let (minx, miny, maxx, maxy) = self.bbox;
        if nx < 2 || ny < 2 || maxx <= minx || maxy <= miny {
            return None;
        }
        if !(minx <= x && x <= maxx && miny <= y && y <= maxy) {
            return None;
        }
        let fi = (x - minx) / (maxx - minx) * (nx - 1) as f64;
        let fj = (maxy - y) / (maxy - miny) * (ny - 1) as f64;
        let i0 = 0i64.max((nx - 2).min(fi as i64));
        let j0 = 0i64.max((ny - 2).min(fj as i64));
        let tx = fi - i0 as f64;
        let ty = fj - j0 as f64;
        let z = |j: i64, i: i64| self.vertices[(j * nx + i) as usize].2 as f64;
        let z00 = z(j0, i0);
        let z10 = z(j0, i0 + 1);
        let z01 = z(j0 + 1, i0);
        let z11 = z(j0 + 1, i0 + 1);
        let top = z00 * (1.0 - tx) + z10 * tx;
        let bot = z01 * (1.0 - tx) + z11 * tx;
        Some(top * (1.0 - ty) + bot * ty)
    }

    /// `bounds`: `(min, max)` corners, or `None` for an empty grid.
    pub fn bounds(&self) -> Option<(V3, V3)> {
        let first = self.vertices.first()?;
        let mut lo = (first.0 as f64, first.1 as f64, first.2 as f64);
        let mut hi = lo;
        for v in &self.vertices[1..] {
            lo = (
                py_min2(lo.0, v.0 as f64),
                py_min2(lo.1, v.1 as f64),
                py_min2(lo.2, v.2 as f64),
            );
            hi = (
                py_max2(hi.0, v.0 as f64),
                py_max2(hi.1, v.1 as f64),
                py_max2(hi.2, v.2 as f64),
            );
        }
        Some((V3(lo.0 as f32, lo.1 as f32, lo.2 as f32), V3(hi.0 as f32, hi.1 as f32, hi.2 as f32)))
    }
}

/// `_bbox_ll`: geodetic bbox `(lat_s, lon_w, lat_n, lon_e)` of a local-metre
/// `(minx, miny, maxx, maxy)` rectangle.
pub fn bbox_ll(datum: &SceneDatum, bbox: (f64, f64, f64, f64)) -> (f64, f64, f64, f64) {
    let (minx, miny, maxx, maxy) = bbox;
    let mut lats = Vec::with_capacity(4);
    let mut lons = Vec::with_capacity(4);
    for (lx, ly) in [(minx, miny), (maxx, miny), (maxx, maxy), (minx, maxy)] {
        let (la, lo, _) = datum.local_to_geodetic(V3(lx as f32, ly as f32, 0.0));
        lats.push(la);
        lons.push(lo);
    }
    let fmin = |v: &[f64]| v.iter().copied().fold(v[0], py_min2);
    let fmax = |v: &[f64]| v.iter().copied().fold(v[0], py_max2);
    (fmin(&lats), fmin(&lons), fmax(&lats), fmax(&lons))
}

/// Python's `round(x)` to int: half to even.
fn round_int(x: f64) -> i64 {
    x.round_ties_even() as i64
}

/// `build_terrain` with Python's defaults `target_cell_m=40.0`,
/// `max_grid=180`, `zoom=15`: the DEM-lifted grid over the local-metre
/// `bbox`, or `None` if the DEM is not fully loaded (or the box is empty).
///
/// A non-positive or non-finite `target_cell_m` is a Python exception
/// (`ZeroDivisionError`, `OverflowError`); here it returns `None`.
pub fn build_terrain(
    datum: &SceneDatum,
    sampler: &dyn ElevationSampler,
    ground_ref: f64,
    bbox: (f64, f64, f64, f64),
    target_cell_m: f64,
    max_grid: i64,
    zoom: i64,
) -> Option<TerrainObject> {
    let (minx, miny, maxx, maxy) = bbox;
    let (w, h) = (maxx - minx, maxy - miny);
    if w <= 0.0 || h <= 0.0 {
        return None;
    }
    if !(target_cell_m > 0.0) || !(w / target_cell_m).is_finite() || !(h / target_cell_m).is_finite() {
        return None;
    }
    let nx = 2i64.max(max_grid.min(round_int(w / target_cell_m) + 1));
    let ny = 2i64.max(max_grid.min(round_int(h / target_cell_m) + 1));

    let (lat_s, lon_w, lat_n, lon_e) = bbox_ll(datum, bbox);
    let tiles = tiles_covering(lat_s, lon_w, lat_n, lon_e, zoom);
    if tiles.is_empty() {
        return None;
    }
    let tx0 = tiles.iter().map(|t| t.0).min().expect("non-empty");
    let ty0 = tiles.iter().map(|t| t.1).min().expect("non-empty");
    let ntx = tiles.iter().map(|t| t.0).max().expect("non-empty") - tx0 + 1;
    let nty = tiles.iter().map(|t| t.1).max().expect("non-empty") - ty0 + 1;

    let mut verts = Vec::with_capacity((nx * ny) as usize);
    let mut uvs = Vec::with_capacity((nx * ny) as usize);
    for j in 0..ny {
        let y = maxy - h * j as f64 / (ny - 1) as f64;
        for i in 0..nx {
            let x = minx + w * i as f64 / (nx - 1) as f64;
            let p = V3(x as f32, y as f32, 0.0);
            let e = sampler.elevation_at_local(p)?;
            verts.push(V3(x as f32, y as f32, (e - ground_ref) as f32));
            let (lat, lon, _) = datum.local_to_geodetic(p);
            let (xf, yf) = deg2num(lat, lon, zoom);
            uvs.push(((xf - tx0 as f64) / ntx as f64, (yf - ty0 as f64) / nty as f64));
        }
    }

    let mut tris = Vec::with_capacity(((nx - 1) * (ny - 1) * 2) as usize);
    for j in 0..ny - 1 {
        for i in 0..nx - 1 {
            let a = (j * nx + i) as usize;
            let b = a + 1;
            let c = a + nx as usize;
            let d = c + 1;
            tris.push((a, c, b));
            tris.push((b, c, d));
        }
    }
    Some(TerrainObject {
        vertices: verts,
        uvs,
        triangles: tris,
        tile_range: (tx0, ty0, ntx, nty, zoom),
        nx,
        ny,
        bbox,
        visible: true,
    })
}

/// Where each tile lands in the mosaic `build_mosaic` paints.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MosaicPlan {
    pub width: i64,
    pub height: i64,
    /// The `QImage.fill` colour (`0xFF8A8A8A`): neutral grey so a tile still
    /// loading does not read black.
    pub fill: u32,
    /// `(x, y, zoom, dest_x, dest_y)` in paint order.
    pub tiles: Vec<(i64, i64, i64, i64, i64)>,
}

/// `build_mosaic` without the painting: the size of the `Format_RGB32`
/// mosaic and the destination of every available tile of the terrain's zoom
/// that lies in its tile range (in the order given). `None` when no tile
/// qualifies.
pub fn mosaic_plan(terrain: &TerrainObject, available: &[(i64, i64, i64)]) -> Option<MosaicPlan> {
    let (tx0, ty0, ntx, nty, zoom) = terrain.tile_range;
    let mut tiles = Vec::new();
    for &(x, y, z) in available {
        if z != zoom {
            continue;
        }
        if tx0 <= x && x < tx0 + ntx && ty0 <= y && y < ty0 + nty {
            tiles.push((x, y, z, (x - tx0) * TILE_PX, (y - ty0) * TILE_PX));
        }
    }
    if tiles.is_empty() {
        return None;
    }
    Some(MosaicPlan { width: ntx * TILE_PX, height: nty * TILE_PX, fill: 0xFF8A8A8A, tiles })
}
