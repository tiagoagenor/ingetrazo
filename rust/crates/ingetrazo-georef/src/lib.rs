// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! IngeTrazo's georeferencing, ported line by line from `georef/datum.py`,
//! `georef/dem.py` and `georef/geotiff.py` (plus the slippy-map math of
//! `georef/tiles.py` that the DEM sampler needs).
//!
//! * [`datum`]: UTM forward/inverse and [`datum::SceneDatum`].
//! * [`tiles`]: `deg2num` and `tiles_covering`.
//! * [`dem`]: terrarium/mapbox decode, bilinear sampling, [`dem::DemSampler`].
//! * [`geotiff`]: TIFF/BigTIFF header, GeoKeys, strip/tile decoding to RGBA.
//!
//! # Exactness
//!
//! Every function keeps Python's floating-point operation order. Python's
//! `x ** n` on floats is C `pow`, so it is `f64::powf` here (never `powi`);
//! `math.radians`/`degrees` are the same `x * (pi/180)` products as
//! `f64::to_radians`/`to_degrees`; Python's float `%` is [`datum::py_mod`].
//! `QVector3D` values are f32 ([`ingetrazo_math::V3`]); a Python float going
//! into one is rounded to nearest, as PySide does with `as f32`. NumPy 2
//! (NEP 50) keeps `float32 scalar * Python float` in float32, which is how
//! [`dem::bilinear`] computes.
//!
//! The only dependence outside this code is the C math library: `sin`, `cos`,
//! `tan`, `asinh`, `sqrt` and `pow` are the platform's libm in both Python
//! and Rust, so the fixtures recorded by
//! `scripts/golden/make_georef_fixtures.py` are compared bit for bit. A
//! different libm (glibc vs the Windows UCRT vs macOS) may differ in the last
//! place for `tan`/`asinh`/`pow`; that would be a property of the platform
//! that recorded the fixtures, not of this port.
//!
//! # Not ported
//!
//! * Qt-only parts: `DEMSampler`'s `QObject` signal and tile fetching (the
//!   sampler here is the pure lookup over decoded fields; the network
//!   belongs to the tiles milestone) and `decode_elevation`'s
//!   `QImage.convertToFormat` (the caller hands over RGBA8888 bytes).
//! * JPEG-in-TIFF pixel decoding, which Python delegates to Qt's JPEG
//!   plugin; the header is read, `read_rgba` returns an error for it.
//! * The process-pool shortcut of `read_rgba` (POSIX only, same output).
//! * Python exceptions that are not `GeoTiffError` (a `struct.error` on a
//!   truncated header, say) are `GeoTiffError`s here.

pub mod datum;
pub mod dem;
pub mod geotiff;
pub mod tiles;
