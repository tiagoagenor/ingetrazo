// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! `georef/dem.py`: RGB-encoded DEM tiles, heightfields and elevation lookup.
//!
//! Heightfields are float32 arrays like NumPy's, and the arithmetic stays in
//! float32 where NumPy 2 keeps it there (`float32 * Python float` is float32).

use std::collections::HashMap;

use ingetrazo_math::V3;

use crate::datum::SceneDatum;
use crate::tiles::deg2num;

/// DEM tiles, like map tiles, are 256x256.
pub const TILE_PX: i64 = 256;

/// `DEMSource`: an `{z}/{x}/{y}` URL template plus its RGB encoding.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DemSource {
    pub id: String,
    pub name: String,
    pub url_template: String,
    /// `"terrarium"` or `"mapbox"`.
    pub encoding: String,
    pub max_zoom: i64,
    pub attribution: String,
}

impl DemSource {
    pub fn url(&self, x: i64, y: i64, z: i64) -> String {
        self.url_template
            .replace("{z}", &z.to_string())
            .replace("{x}", &x.to_string())
            .replace("{y}", &y.to_string())
    }
}

/// `AWS_TERRAIN`.
pub fn aws_terrain() -> DemSource {
    DemSource {
        id: "aws_terrarium".to_string(),
        name: "AWS Terrain Tiles (SRTM/NED)".to_string(),
        url_template: "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"
            .to_string(),
        encoding: "terrarium".to_string(),
        max_zoom: 15,
        attribution: "Terrain Tiles hosted on AWS by Mapzen/Nextzen \u{2014} sources include \
                      SRTM, NED, Copernicus. https://registry.opendata.aws/terrain-tiles/"
            .to_string(),
    }
}

/// A `(H, W)` float32 heightfield, row-major.
#[derive(Clone, Debug, PartialEq)]
pub struct Field {
    pub h: usize,
    pub w: usize,
    pub data: Vec<f32>,
}

impl Field {
    pub fn new(h: usize, w: usize, data: Vec<f32>) -> Field {
        assert_eq!(data.len(), h * w);
        Field { h, w, data }
    }

    #[inline]
    fn at(&self, y: usize, x: usize) -> f32 {
        self.data[y * self.w + x]
    }
}

/// `decode_elevation`: tightly packed RGBA8888 (`w * h * 4` bytes, the alpha
/// ignored) to a float32 metre heightfield.
///
/// - `terrarium`: `h = R*256 + G + B/256 - 32768`
/// - `mapbox`:    `h = -10000 + (R*65536 + G*256 + B) * 0.1`
pub fn decode_elevation(rgba: &[u8], w: usize, h: usize, encoding: &str) -> Field {
    assert!(rgba.len() >= w * h * 4);
    let mut data = Vec::with_capacity(w * h);
    for i in 0..w * h {
        let r = rgba[i * 4] as f32;
        let g = rgba[i * 4 + 1] as f32;
        let b = rgba[i * 4 + 2] as f32;
        let v = if encoding == "mapbox" {
            -10000.0f32 + (r * 65536.0f32 + g * 256.0f32 + b) * (0.1f64 as f32)
        } else {
            (r * 256.0f32 + g + b / 256.0f32) - 32768.0f32
        };
        data.push(v);
    }
    Field { h, w, data }
}

/// `bilinear`: sample `field` at fractional pixel `(px, py)`, clamped to the
/// array. The four reads and the blend are float32 (NumPy scalars), the
/// result widened to a Python float.
pub fn bilinear(field: &Field, px: f64, py: f64) -> f64 {
    let (h, w) = (field.h as i64, field.w as i64);
    let fx_floor = px.floor();
    let fy_floor = py.floor();
    let mut x0 = fx_floor as i64;
    let mut y0 = fy_floor as i64;
    let fx = px - fx_floor;
    let fy = py - fy_floor;
    x0 = x0.max(0).min(w - 1);
    y0 = y0.max(0).min(h - 1);
    let x1 = (x0 + 1).min(w - 1);
    let y1 = (y0 + 1).min(h - 1);
    let (x0, y0, x1, y1) = (x0 as usize, y0 as usize, x1 as usize, y1 as usize);
    let (wx0, wx1) = ((1.0 - fx) as f32, fx as f32);
    let (wy0, wy1) = ((1.0 - fy) as f32, fy as f32);
    let top = field.at(y0, x0) * wx0 + field.at(y0, x1) * wx1;
    let bot = field.at(y1, x0) * wx0 + field.at(y1, x1) * wx1;
    (top * wy0 + bot * wy1) as f64
}

/// `pixel_in_tile`: `(tile_x, tile_y, px, py)` of `(lat, lon)` at `zoom`.
pub fn pixel_in_tile(lat: f64, lon: f64, zoom: i64) -> (i64, i64, f64, f64) {
    let (xf, yf) = deg2num(lat, lon, zoom);
    let tx = xf.floor() as i64;
    let ty = yf.floor() as i64;
    (tx, ty, (xf - tx as f64) * TILE_PX as f64, (yf - ty as f64) * TILE_PX as f64)
}

/// `DEMSampler` without Qt signals or the network: the lookup over the
/// heightfields decoded so far. A tile that is not loaded reads as `None`;
/// [`DemSampler::missing_tiles`] says which to fetch.
#[derive(Clone, Debug)]
pub struct DemSampler {
    pub datum: SceneDatum,
    pub source: DemSource,
    pub zoom: i64,
    fields: HashMap<(i64, i64, i64), Field>,
}

impl DemSampler {
    pub fn new(datum: SceneDatum, source: DemSource, zoom: i64) -> DemSampler {
        DemSampler { datum, source, zoom, fields: HashMap::new() }
    }

    /// Stores a decoded tile (Python's `_fields[(x, y, zoom)] = ...`).
    pub fn insert_field(&mut self, tx: i64, ty: i64, field: Field) {
        self.fields.insert((tx, ty, self.zoom), field);
    }

    /// The tiles covering the bbox that are not loaded yet (`ensure_area`
    /// would request exactly these).
    pub fn missing_tiles(&self, lat_s: f64, lon_w: f64, lat_n: f64, lon_e: f64) -> Vec<(i64, i64)> {
        crate::tiles::tiles_covering(lat_s, lon_w, lat_n, lon_e, self.zoom)
            .into_iter()
            .filter(|&(tx, ty)| !self.fields.contains_key(&(tx, ty, self.zoom)))
            .collect()
    }

    /// `elevation_at`: metres at `(lat, lon)`, or `None` if not loaded.
    pub fn elevation_at(&self, lat: f64, lon: f64) -> Option<f64> {
        let (tx, ty, px, py) = pixel_in_tile(lat, lon, self.zoom);
        let field = self.fields.get(&(tx, ty, self.zoom))?;
        Some(bilinear(field, px, py))
    }

    /// `elevation_at_local`: elevation at a local-metre scene point.
    pub fn elevation_at_local(&self, point: V3) -> Option<f64> {
        let (lat, lon, _) = self.datum.local_to_geodetic(point);
        self.elevation_at(lat, lon)
    }

    pub fn has_tile(&self, lat: f64, lon: f64) -> bool {
        let (tx, ty, _, _) = pixel_in_tile(lat, lon, self.zoom);
        self.fields.contains_key(&(tx, ty, self.zoom))
    }
}

/// What terrain, surfaces and profiles ask of an elevation source: Python's
/// duck-typed `sampler` (`datum`, `elevation_at`, `elevation_at_local`,
/// `ensure_area`). [`DemSampler`] is the real one; the app wires a fetching
/// sampler through the same trait.
pub trait ElevationSampler {
    fn datum(&self) -> &SceneDatum;

    /// Metres at `(lat, lon)`, or `None` if the tile is not loaded.
    fn elevation_at(&self, lat: f64, lon: f64) -> Option<f64>;

    /// Elevation at a local-metre scene point (via the datum).
    fn elevation_at_local(&self, point: V3) -> Option<f64> {
        let (lat, lon, _) = self.datum().local_to_geodetic(point);
        self.elevation_at(lat, lon)
    }

    /// Request every tile covering the geodetic bbox (idempotent). The
    /// default asks for nothing; an app sampler queues the downloads.
    fn ensure_area(&self, _lat_s: f64, _lon_w: f64, _lat_n: f64, _lon_e: f64) {}
}

impl ElevationSampler for DemSampler {
    fn datum(&self) -> &SceneDatum {
        &self.datum
    }
    fn elevation_at(&self, lat: f64, lon: f64) -> Option<f64> {
        DemSampler::elevation_at(self, lat, lon)
    }
}
