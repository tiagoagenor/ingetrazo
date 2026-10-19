// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! `georef/datum.py`: the scene datum, UTM (WGS84 Transverse Mercator).
//!
//! Every expression keeps Python's operand order and associativity; Python's
//! `**` is `powf` (C `pow`), never `powi`.

use ingetrazo_formats::pyjson::Value;
use ingetrazo_math::V3;

// WGS84 ellipsoid + UTM constants.
const A: f64 = 6378137.0;
const F: f64 = 1.0 / 298.257223563;
const E2: f64 = F * (2.0 - F);
const EP2: f64 = E2 / (1.0 - E2);
const K0: f64 = 0.9996;
const FALSE_EASTING: f64 = 500000.0;
const FALSE_NORTHING: f64 = 10000000.0;

#[inline]
fn pw(x: f64, n: f64) -> f64 {
    x.powf(n)
}

/// Python's float `%`: the result takes the divisor's sign.
pub fn py_mod(a: f64, b: f64) -> f64 {
    let m = a % b;
    if m != 0.0 {
        if (b < 0.0) != (m < 0.0) {
            m + b
        } else {
            m
        }
    } else {
        0.0f64.copysign(b)
    }
}

/// `zone_for_lon`: `int((lon + 180) / 6) + 1` (`int` truncates toward zero).
pub fn zone_for_lon(lon: f64) -> i64 {
    (((lon + 180.0) / 6.0).trunc() as i64) + 1
}

/// `_lon0`: central meridian of `zone`, in radians.
pub fn lon0(zone: i64) -> f64 {
    (((zone - 1) * 6 - 180 + 3) as f64).to_radians()
}

/// `utm_forward`: `(easting, northing)` of `lat`/`lon` (degrees) in `zone`.
pub fn utm_forward(lat: f64, lon: f64, zone: i64) -> (f64, f64) {
    let latr = lat.to_radians();
    let lonr = lon.to_radians();
    let n = A / (1.0 - E2 * pw(latr.sin(), 2.0)).sqrt();
    let t = pw(latr.tan(), 2.0);
    let c = EP2 * pw(latr.cos(), 2.0);
    let a = latr.cos() * (lonr - lon0(zone));
    let m = A
        * ((1.0 - E2 / 4.0 - 3.0 * pw(E2, 2.0) / 64.0 - 5.0 * pw(E2, 3.0) / 256.0) * latr
            - (3.0 * E2 / 8.0 + 3.0 * pw(E2, 2.0) / 32.0 + 45.0 * pw(E2, 3.0) / 1024.0)
                * (2.0 * latr).sin()
            + (15.0 * pw(E2, 2.0) / 256.0 + 45.0 * pw(E2, 3.0) / 1024.0) * (4.0 * latr).sin()
            - (35.0 * pw(E2, 3.0) / 3072.0) * (6.0 * latr).sin());
    let easting = K0
        * n
        * (a + (1.0 - t + c) * pw(a, 3.0) / 6.0
            + (5.0 - 18.0 * t + pw(t, 2.0) + 72.0 * c - 58.0 * EP2) * pw(a, 5.0) / 120.0)
        + FALSE_EASTING;
    let mut northing = K0
        * (m + n
            * latr.tan()
            * (pw(a, 2.0) / 2.0
                + (5.0 - t + 9.0 * c + 4.0 * pw(c, 2.0)) * pw(a, 4.0) / 24.0
                + (61.0 - 58.0 * t + pw(t, 2.0) + 600.0 * c - 330.0 * EP2) * pw(a, 6.0)
                    / 720.0));
    if lat < 0.0 {
        northing += FALSE_NORTHING;
    }
    (easting, northing)
}

/// `utm_inverse`: UTM to geodetic `(lat, lon)` in degrees.
pub fn utm_inverse(easting: f64, northing: f64, zone: i64, northern: bool) -> (f64, f64) {
    let x = easting - FALSE_EASTING;
    let y = if northern { northing } else { northing - FALSE_NORTHING };
    let m = y / K0;
    let mu = m / (A * (1.0 - E2 / 4.0 - 3.0 * pw(E2, 2.0) / 64.0 - 5.0 * pw(E2, 3.0) / 256.0));
    let e1 = (1.0 - (1.0 - E2).sqrt()) / (1.0 + (1.0 - E2).sqrt());
    let phi1 = mu
        + (3.0 * e1 / 2.0 - 27.0 * pw(e1, 3.0) / 32.0) * (2.0 * mu).sin()
        + (21.0 * pw(e1, 2.0) / 16.0 - 55.0 * pw(e1, 4.0) / 32.0) * (4.0 * mu).sin()
        + (151.0 * pw(e1, 3.0) / 96.0) * (6.0 * mu).sin()
        + (1097.0 * pw(e1, 4.0) / 512.0) * (8.0 * mu).sin();
    let c1 = EP2 * pw(phi1.cos(), 2.0);
    let t1 = pw(phi1.tan(), 2.0);
    let n1 = A / (1.0 - E2 * pw(phi1.sin(), 2.0)).sqrt();
    let r1 = A * (1.0 - E2) / pw(1.0 - E2 * pw(phi1.sin(), 2.0), 1.5);
    let d = x / (n1 * K0);
    let lat = phi1
        - (n1 * phi1.tan() / r1)
            * (pw(d, 2.0) / 2.0
                - (5.0 + 3.0 * t1 + 10.0 * c1 - 4.0 * pw(c1, 2.0) - 9.0 * EP2) * pw(d, 4.0)
                    / 24.0
                + (61.0 + 90.0 * t1 + 298.0 * c1 + 45.0 * pw(t1, 2.0)
                    - 252.0 * EP2
                    - 3.0 * pw(c1, 2.0))
                    * pw(d, 6.0)
                    / 720.0);
    let lon = lon0(zone)
        + (d - (1.0 + 2.0 * t1 + c1) * pw(d, 3.0) / 6.0
            + (5.0 - 2.0 * c1 + 28.0 * t1 - 3.0 * pw(c1, 2.0) + 8.0 * EP2 + 24.0 * pw(t1, 2.0))
                * pw(d, 5.0)
                / 120.0)
            / phi1.cos();
    (lat.to_degrees(), lon.to_degrees())
}

/// `_wrap_deg`: fold an angle into (-180, 180].
pub fn wrap_deg(deg: f64) -> f64 {
    let d = py_mod(deg + 180.0, 360.0) - 180.0;
    if d == -180.0 {
        180.0
    } else {
        d
    }
}

/// `SceneDatum`: anchors the scene at a geographic point. `lat`, `lon`,
/// `alt` and `north` are the truth; the rest is derived in [`SceneDatum::new`]
/// (Python's `__post_init__`).
#[derive(Clone, Debug)]
pub struct SceneDatum {
    pub lat: f64,
    pub lon: f64,
    pub alt: f64,
    /// Degrees clockwise from the model's +Y axis, wrapped into (-180, 180].
    pub north: f64,
    pub zone: i64,
    pub northern: bool,
    east0: f64,
    north0: f64,
    cos_n: f64,
    sin_n: f64,
}

impl SceneDatum {
    pub fn new(lat: f64, lon: f64, alt: f64, north: f64) -> SceneDatum {
        let north = wrap_deg(north);
        let zone = zone_for_lon(lon);
        let northern = lat >= 0.0;
        let (east0, north0) = utm_forward(lat, lon, zone);
        let b = north.to_radians();
        SceneDatum {
            lat,
            lon,
            alt,
            north,
            zone,
            northern,
            east0,
            north0,
            cos_n: b.cos(),
            sin_n: b.sin(),
        }
    }

    /// UTM coordinates of the anchor.
    pub fn anchor_utm(&self) -> (f64, f64) {
        (self.east0, self.north0)
    }

    /// `(cos, sin)` of the north angle.
    pub fn cos_sin_north(&self) -> (f64, f64) {
        (self.cos_n, self.sin_n)
    }

    pub fn grid_to_local_xy(&self, east: f64, north: f64) -> (f64, f64) {
        let (c, s) = (self.cos_n, self.sin_n);
        (east * c + north * s, north * c - east * s)
    }

    pub fn local_to_grid_xy(&self, x: f64, y: f64) -> (f64, f64) {
        let (c, s) = (self.cos_n, self.sin_n);
        (x * c - y * s, x * s + y * c)
    }

    /// `north_vector`: a QVector3D, so float32.
    pub fn north_vector(&self) -> V3 {
        V3(self.sin_n as f32, self.cos_n as f32, 0.0)
    }

    pub fn hemisphere(&self) -> &'static str {
        if self.northern {
            "N"
        } else {
            "S"
        }
    }

    /// `geodetic_to_local`; `alt = None` means the scene's reference plane.
    pub fn geodetic_to_local(&self, lat: f64, lon: f64, alt: Option<f64>) -> V3 {
        let (east, north) = utm_forward(lat, lon, self.zone);
        let z = match alt {
            None => 0.0,
            Some(a) => a - self.alt,
        };
        let (x, y) = self.grid_to_local_xy(east - self.east0, north - self.north0);
        V3(x as f32, y as f32, z as f32)
    }

    pub fn local_to_geodetic(&self, p: V3) -> (f64, f64, f64) {
        let (de, dn) = self.local_to_grid_xy(p.0 as f64, p.1 as f64);
        let (lat, lon) = utm_inverse(de + self.east0, dn + self.north0, self.zone, self.northern);
        (lat, lon, p.2 as f64 + self.alt)
    }

    pub fn utm_to_local(&self, east: f64, north: f64, alt: f64) -> V3 {
        let (x, y) = self.grid_to_local_xy(east - self.east0, north - self.north0);
        V3(x as f32, y as f32, (alt - self.alt) as f32)
    }

    pub fn local_to_utm(&self, p: V3) -> (f64, f64, f64) {
        let (de, dn) = self.local_to_grid_xy(p.0 as f64, p.1 as f64);
        (de + self.east0, dn + self.north0, p.2 as f64 + self.alt)
    }

    /// `to_dict` as a Python-json value: `north` only when non-zero.
    pub fn to_dict(&self) -> Value {
        let mut d = vec![
            ("lat".to_string(), Value::Float(self.lat)),
            ("lon".to_string(), Value::Float(self.lon)),
            ("alt".to_string(), Value::Float(self.alt)),
        ];
        if self.north != 0.0 {
            d.push(("north".to_string(), Value::Float(self.north)));
        }
        Value::Object(d)
    }

    /// `from_dict`: `lat` and `lon` are required (a missing key is Python's
    /// `KeyError`), `alt` and `north` default to 0. Values go through
    /// Python's `float()`: ints, floats and bools, and numeric strings.
    pub fn from_dict(data: &Value) -> Result<SceneDatum, String> {
        let get = |k: &str| data.get(k).ok_or_else(|| format!("KeyError: '{k}'"));
        let lat = to_float(get("lat")?)?;
        let lon = to_float(get("lon")?)?;
        let alt = match data.get("alt") {
            Some(v) => to_float(v)?,
            None => 0.0,
        };
        let north = match data.get("north") {
            Some(v) => to_float(v)?,
            None => 0.0,
        };
        Ok(SceneDatum::new(lat, lon, alt, north))
    }
}

/// Python's `float(x)` for a json value.
fn to_float(v: &Value) -> Result<f64, String> {
    match v {
        Value::Float(x) => Ok(*x),
        Value::Int(s) => s.parse::<f64>().map_err(|e| e.to_string()),
        Value::Bool(b) => Ok(if *b { 1.0 } else { 0.0 }),
        Value::Str(s) => s
            .trim()
            .parse::<f64>()
            .map_err(|_| format!("ValueError: could not convert string to float: '{s}'")),
        _ => Err("TypeError: float() argument".to_string()),
    }
}

