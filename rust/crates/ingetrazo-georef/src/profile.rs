// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! `georef/profile.py`: the longitudinal terrain profile. A polyline and an
//! [`ElevationSampler`] give station / elevation pairs, total length and
//! per-segment slopes; samples on tiles not loaded yet are `None` and
//! `complete` is `false`.
//!
//! `selected_geopath` (picking the path out of a `Scene`'s selection) is the
//! scene's: it needs `Scene.selection` and `Scene.geo_paths`.

use ingetrazo_math::pyfloat::{py_max2, py_min2};
use ingetrazo_math::V3;

use crate::dem::ElevationSampler;
use crate::geotiff::py_hypot2;
use crate::pyconv::fmt_f3;

/// `ProfileSample`.
#[derive(Clone, Debug, PartialEq)]
pub struct ProfileSample {
    /// Horizontal distance from the start (m).
    pub station: f64,
    pub x: f64,
    pub y: f64,
    /// Terrain elevation (m), `None` if not loaded.
    pub elevation: Option<f64>,
}

/// `Profile`.
#[derive(Clone, Debug, PartialEq)]
pub struct Profile {
    pub samples: Vec<ProfileSample>,
    pub length: f64,
    /// Every sample has an elevation.
    pub complete: bool,
}

impl Profile {
    pub fn elevations(&self) -> Vec<f64> {
        self.samples.iter().filter_map(|s| s.elevation).collect()
    }

    pub fn min_elevation(&self) -> Option<f64> {
        let es = self.elevations();
        let first = *es.first()?;
        Some(es.into_iter().fold(first, py_min2))
    }

    pub fn max_elevation(&self) -> Option<f64> {
        let es = self.elevations();
        let first = *es.first()?;
        Some(es.into_iter().fold(first, py_max2))
    }

    /// `total_gain`: cumulative uphill gain (m).
    pub fn total_gain(&self) -> f64 {
        let mut gain = 0.0f64;
        let mut prev: Option<f64> = None;
        for s in &self.samples {
            let Some(e) = s.elevation else { continue };
            if let Some(p) = prev {
                if e > p {
                    gain += e - p;
                }
            }
            prev = Some(e);
        }
        gain
    }

    /// `slopes`: `(station, slope_percent)` between consecutive samples that
    /// both have an elevation.
    pub fn slopes(&self) -> Vec<(f64, f64)> {
        let mut out = Vec::new();
        for w in self.samples.windows(2) {
            let (a, b) = (&w[0], &w[1]);
            let (Some(ea), Some(eb)) = (a.elevation, b.elevation) else { continue };
            let ds = b.station - a.station;
            if ds > 1e-9 {
                out.push((a.station, (eb - ea) / ds * 100.0));
            }
        }
        out
    }
}

/// `polyline_length`: horizontal (XY) length.
pub fn polyline_length(pts: &[V3]) -> f64 {
    let mut total = 0.0f64;
    for w in pts.windows(2) {
        total += py_hypot2(w[1].0 as f64 - w[0].0 as f64, w[1].1 as f64 - w[0].1 as f64);
    }
    total
}

/// `point_at_station`: the `(x, y)` at horizontal chainage `s` along `pts`.
pub fn point_at_station(pts: &[V3], s: f64) -> (f64, f64) {
    if pts.is_empty() {
        return (0.0, 0.0);
    }
    if s <= 0.0 {
        return (pts[0].0 as f64, pts[0].1 as f64);
    }
    let mut acc = 0.0f64;
    for w in pts.windows(2) {
        let (a, b) = (w[0], w[1]);
        let (ax, ay, bx, by) = (a.0 as f64, a.1 as f64, b.0 as f64, b.1 as f64);
        let seg = py_hypot2(bx - ax, by - ay);
        if acc + seg >= s {
            let t = if seg > 1e-12 { (s - acc) / seg } else { 0.0 };
            return (ax + (bx - ax) * t, ay + (by - ay) * t);
        }
        acc += seg;
    }
    let last = pts[pts.len() - 1];
    (last.0 as f64, last.1 as f64)
}

/// `sample_profile`: terrain elevation along `polyline`. `spacing` is the
/// chainage step in metres; `None` is length/256, at least 1 m.
///
/// A zero `spacing` is Python's `ZeroDivisionError` (`Err`).
pub fn sample_profile(
    polyline: &[V3],
    sampler: &dyn ElevationSampler,
    spacing: Option<f64>,
) -> Result<Profile, String> {
    let pts = polyline;
    if pts.len() < 2 {
        return Ok(Profile { samples: Vec::new(), length: 0.0, complete: true });
    }
    let length = polyline_length(pts);
    if length <= 1e-9 {
        return Ok(Profile { samples: Vec::new(), length: 0.0, complete: true });
    }
    let spacing = match spacing {
        Some(s) => s,
        None => py_max2(length / 256.0, 1.0),
    };
    if spacing == 0.0 {
        return Err("ZeroDivisionError: float division by zero".to_string());
    }
    let n = ((length / spacing).ceil() as i64).max(1);

    // Make sure the DEM covering the trace is (being) fetched.
    let mut lats = Vec::with_capacity(pts.len());
    let mut lons = Vec::with_capacity(pts.len());
    for p in pts {
        let (la, lo, _) = sampler.datum().local_to_geodetic(*p);
        lats.push(la);
        lons.push(lo);
    }
    let fmin = |v: &[f64]| v.iter().copied().fold(v[0], py_min2);
    let fmax = |v: &[f64]| v.iter().copied().fold(v[0], py_max2);
    sampler.ensure_area(fmin(&lats), fmin(&lons), fmax(&lats), fmax(&lons));

    let mut samples = Vec::with_capacity(n as usize + 1);
    let mut complete = true;
    for i in 0..=n {
        let s = py_min2(length, i as f64 * spacing);
        let (x, y) = point_at_station(pts, s);
        let elev = sampler.elevation_at_local(V3(x as f32, y as f32, 0.0));
        if elev.is_none() {
            complete = false;
        }
        samples.push(ProfileSample { station: s, x, y, elevation: elev });
    }
    Ok(Profile { samples, length, complete })
}

/// `VERTICAL_NOTES[vertical]`, `"local"` for an unknown key.
pub fn vertical_note(vertical: &str) -> &'static str {
    match vertical {
        "odm" => {
            "photogrammetric survey (ODM); WGS84 ellipsoidal heights unless \
             the processing used ground control points"
        }
        "dem" => "global DEM (SRTM-derived, orthometric)",
        _ => "local reference: heights are relative, not absolute",
    }
}

/// `profile_to_csv`: the profile as CSV text (station, x, y, elevation), a
/// leading comment naming the vertical reference.
pub fn profile_to_csv(profile: &Profile, vertical: &str, datum_alt: Option<f64>) -> String {
    let mut lines = vec![format!("# elevation reference: {}", vertical_note(vertical))];
    if let Some(alt) = datum_alt {
        lines.push(format!("# scene Z=0 is at {} m in that reference", fmt_f3(alt)));
    }
    lines.push("station_m,x_m,y_m,elevation_m".to_string());
    for s in &profile.samples {
        let elev = match s.elevation {
            None => String::new(),
            Some(e) => fmt_f3(e),
        };
        lines.push(format!("{},{},{},{}", fmt_f3(s.station), fmt_f3(s.x), fmt_f3(s.y), elev));
    }
    lines.join("\n") + "\n"
}
