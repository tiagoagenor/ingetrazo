// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! `georef/tiles.py`: slippy-map math, tile sources and their URL templates,
//! the display-only [`TileLayer`] and the on-disk LRU [`TileCache`].
//!
//! Not here (the app's): the decoded images a `TileLayer` stashes in
//! `images` (Qt `QImage`s, GL textures) and `default_cache_dir` (Qt's
//! `QStandardPaths`).

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use ingetrazo_formats::pyjson::Value;
use ingetrazo_math::pyfloat::{py_max2, py_min2};
use ingetrazo_math::V3;

use crate::datum::SceneDatum;
use crate::pyconv::{err, get, py_float, py_int, truthy, PyResult};

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

/// `num2deg`: fractional tile coordinates to geodetic `(lat, lon)`. For
/// integer tile indices this is the tile's north-west corner.
pub fn num2deg(xtile: f64, ytile: f64, zoom: i64) -> (f64, f64) {
    let n = two_pow(zoom);
    let lon_deg = xtile / n * 360.0 - 180.0;
    let lat_rad = (std::f64::consts::PI * (1.0 - 2.0 * ytile / n)).sinh().atan();
    (lat_rad.to_degrees(), lon_deg)
}

/// `tile_bbox`: `(lat_south, lon_west, lat_north, lon_east)` of a tile.
pub fn tile_bbox(x: i64, y: i64, zoom: i64) -> (f64, f64, f64, f64) {
    let (lat_n, lon_w) = num2deg(x as f64, y as f64, zoom);
    let (lat_s, lon_e) = num2deg((x + 1) as f64, (y + 1) as f64, zoom);
    (lat_s, lon_w, lat_n, lon_e)
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

// ---- Tile sources -------------------------------------------------------------

/// `TileSource`: a slippy-map provider, an `{z}/{x}/{y}` URL template plus
/// metadata. `{s}` rotates through `subdomains`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TileSource {
    pub id: String,
    pub name: String,
    pub url_template: String,
    pub max_zoom: i64,
    pub attribution: String,
    pub subdomains: Vec<String>,
    pub tile_size: i64,
}

impl TileSource {
    /// The dataclass with its defaults (`max_zoom=19`, `tile_size=256`, no
    /// attribution, no subdomains).
    pub fn new(id: &str, name: &str, url_template: &str) -> TileSource {
        TileSource {
            id: id.to_string(),
            name: name.to_string(),
            url_template: url_template.to_string(),
            max_zoom: 19,
            attribution: String::new(),
            subdomains: Vec::new(),
            tile_size: 256,
        }
    }

    /// `url`: `{s}`, `{z}`, `{x}`, `{y}` replaced in that order.
    pub fn url(&self, x: i64, y: i64, z: i64) -> String {
        let s: &str = if self.subdomains.is_empty() {
            ""
        } else {
            &self.subdomains[(x + y).rem_euclid(self.subdomains.len() as i64) as usize]
        };
        self.url_template
            .replace("{s}", s)
            .replace("{z}", &z.to_string())
            .replace("{x}", &x.to_string())
            .replace("{y}", &y.to_string())
    }
}

/// `PRESETS`, in Python's dict order. All licensed for this use; never add
/// Google here (invariant #5).
pub fn presets() -> Vec<TileSource> {
    vec![
        TileSource {
            id: "esri_imagery".to_string(),
            name: "Esri World Imagery (satellite)".to_string(),
            url_template: "https://server.arcgisonline.com/ArcGIS/rest/services/\
                           World_Imagery/MapServer/tile/{z}/{y}/{x}"
                .to_string(),
            max_zoom: 19,
            attribution: "Esri, Maxar, Earthstar Geographics, and the GIS community".to_string(),
            subdomains: Vec::new(),
            tile_size: 256,
        },
        TileSource {
            id: "s2cloudless".to_string(),
            name: "Sentinel-2 cloudless (EOX)".to_string(),
            url_template: "https://tiles.maps.eox.at/wmts/1.0.0/\
                           s2cloudless-2020_3857/default/GoogleMapsCompatible/\
                           {z}/{y}/{x}.jpg"
                .to_string(),
            max_zoom: 17,
            attribution: "Sentinel-2 cloudless \u{2014} https://s2maps.eu by EOX IT Services \
                          GmbH (Contains modified Copernicus Sentinel data)"
                .to_string(),
            subdomains: Vec::new(),
            tile_size: 256,
        },
        TileSource {
            id: "osm".to_string(),
            name: "OpenStreetMap (streets)".to_string(),
            url_template: "https://tile.openstreetmap.org/{z}/{x}/{y}.png".to_string(),
            max_zoom: 19,
            attribution: "\u{a9} OpenStreetMap contributors".to_string(),
            subdomains: Vec::new(),
            tile_size: 256,
        },
    ]
}

/// `PRESETS.get(id)`.
pub fn preset(id: &str) -> Option<TileSource> {
    presets().into_iter().find(|p| p.id == id)
}

pub const DEFAULT_SOURCE_ID: &str = "esri_imagery";

/// `source_slug`: filesystem-safe id fragment of a user-given name
/// (`re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")`, else `"fuente"`).
pub fn source_slug(name: &str) -> String {
    let lower = name.to_lowercase();
    let mut out = String::new();
    let mut in_run = false;
    for c in lower.chars() {
        if c.is_ascii_lowercase() || c.is_ascii_digit() {
            out.push(c);
            in_run = false;
        } else if !in_run {
            out.push('-');
            in_run = true;
        }
    }
    let slug = out.trim_matches('-');
    if slug.is_empty() {
        "fuente".to_string()
    } else {
        slug.to_string()
    }
}

/// `custom_source`: a user-pasted XYZ source. A non-empty `name` makes it a
/// saved source with its own id (and so its own cache directory).
pub fn custom_source(url_template: &str, max_zoom: i64, name: Option<&str>) -> TileSource {
    let name = name.filter(|n| !n.is_empty());
    TileSource {
        id: match name {
            Some(n) => format!("custom-{}", source_slug(n)),
            None => "custom".to_string(),
        },
        name: name.unwrap_or("XYZ personalizado").to_string(),
        url_template: url_template.to_string(),
        max_zoom,
        attribution: "Fuente personalizada (definida por el usuario)".to_string(),
        subdomains: Vec::new(),
        tile_size: 256,
    }
}

// ---- Tile layer ---------------------------------------------------------------

/// `TileLayer`: the base-map layer, a chosen source shown as flat Z=0 quads.
/// Display-only (never part of `Scene.mesh`). The decoded images are the
/// app's (Python's `images` dict).
#[derive(Clone, Debug, PartialEq)]
pub struct TileLayer {
    pub source: TileSource,
    pub zoom: i64,
    pub radius_m: f64,
    pub visible: bool,
    /// 0..1.
    pub opacity: f64,
    /// Capture rectangles `(cx, cy, hw, hh)` in local metres.
    pub patches: Vec<(f64, f64, f64, f64)>,
}

impl TileLayer {
    /// `TileLayer(source, zoom, radius_m)`; Python's defaults are 16 and
    /// 1200.0.
    pub fn new(source: TileSource, zoom: i64, radius_m: f64) -> TileLayer {
        TileLayer {
            source,
            zoom,
            radius_m,
            visible: true,
            opacity: 1.0,
            patches: vec![(0.0, 0.0, radius_m, radius_m)],
        }
    }

    /// `to_dict`: the CAPTURE, not the imagery.
    pub fn to_dict(&self) -> Value {
        Value::Object(vec![
            ("source".to_string(), Value::Str(self.source.id.clone())),
            ("zoom".to_string(), Value::int(self.zoom)),
            ("radius_m".to_string(), Value::Float(self.radius_m)),
            ("visible".to_string(), Value::Bool(self.visible)),
            ("opacity".to_string(), Value::Float(self.opacity)),
            (
                "patches".to_string(),
                Value::Array(
                    self.patches
                        .iter()
                        .map(|p| {
                            Value::Array(vec![
                                Value::Float(p.0),
                                Value::Float(p.1),
                                Value::Float(p.2),
                                Value::Float(p.3),
                            ])
                        })
                        .collect(),
                ),
            ),
            ("url_template".to_string(), Value::Str(self.source.url_template.clone())),
            ("name".to_string(), Value::Str(self.source.name.clone())),
            ("attribution".to_string(), Value::Str(self.source.attribution.clone())),
        ])
    }

    /// `from_dict`: presets are looked up by id, anything else is rebuilt
    /// from the saved template. Patches of a length other than four are
    /// `NotModelled` (Python accepts them and fails later, unpacking).
    pub fn from_dict(raw: &Value) -> PyResult<TileLayer> {
        let src_val = get(raw, "source")?;
        let preset_found = match src_val {
            None => None,
            Some(Value::Str(s)) => preset(s),
            Some(Value::Array(_)) | Some(Value::Object(_)) => {
                return err("TypeError", "unhashable type");
            }
            Some(_) => None,
        };
        let source = match preset_found {
            Some(p) => p,
            None => {
                let id = match src_val {
                    None => "custom".to_string(),
                    Some(Value::Str(s)) => s.clone(),
                    Some(_) => return err("NotModelled", "a source id that is not text"),
                };
                let text = |key: &str, default: &str| -> PyResult<String> {
                    crate::pyconv::text_default(get(raw, key)?, default)
                };
                TileSource {
                    name: text("name", &id)?,
                    url_template: text("url_template", "")?,
                    attribution: text("attribution", "")?,
                    id,
                    ..TileSource::new("", "", "")
                }
            }
        };
        let zoom = match get(raw, "zoom")? {
            Some(v) => py_int(v)?,
            None => 16,
        };
        let radius = match get(raw, "radius_m")? {
            Some(v) => py_float(v)?,
            None => 1200.0,
        };
        let mut layer = TileLayer::new(source, zoom, radius);
        if let Some(patches) = get(raw, "patches")? {
            if truthy(patches) {
                let items = crate::pyconv::unpack(patches)?;
                let mut out = Vec::with_capacity(items.len());
                for p in &items {
                    let cells = crate::pyconv::unpack(p)?;
                    let mut vals = Vec::with_capacity(cells.len());
                    for c in &cells {
                        vals.push(py_float(c)?);
                    }
                    if vals.len() != 4 {
                        return err("NotModelled", "a capture patch that is not (cx, cy, hw, hh)");
                    }
                    out.push((vals[0], vals[1], vals[2], vals[3]));
                }
                layer.patches = out;
            }
        }
        layer.visible = match get(raw, "visible")? {
            Some(v) => truthy(v),
            None => true,
        };
        let op = match get(raw, "opacity")? {
            Some(v) => py_float(v)?,
            None => 1.0,
        };
        layer.opacity = py_max2(0.0, py_min2(1.0, op));
        Ok(layer)
    }

    /// `set_rectangle`: one rectangle `width_m` (E-W) by `length_m` (N-S)
    /// centred at local `(cx, cy)`.
    pub fn set_rectangle(&mut self, width_m: f64, length_m: f64, cx: f64, cy: f64) {
        self.patches = vec![(cx, cy, width_m / 2.0, length_m / 2.0)];
    }

    /// `add_patch`.
    pub fn add_patch(&mut self, cx: f64, cy: f64, hw: f64, hh: f64) {
        self.patches.push((cx, cy, hw, hh));
    }

    fn patch_tiles(&self, datum: &SceneDatum, cx: f64, cy: f64, hw: f64, hh: f64) -> Vec<(i64, i64)> {
        let mut lats = Vec::with_capacity(4);
        let mut lons = Vec::with_capacity(4);
        for (lx, ly) in [(cx - hw, cy - hh), (cx + hw, cy - hh), (cx + hw, cy + hh), (cx - hw, cy + hh)] {
            let (la, lo, _) = datum.local_to_geodetic(V3(lx as f32, ly as f32, 0.0));
            lats.push(la);
            lons.push(lo);
        }
        let fmin = |v: &[f64]| v.iter().copied().fold(v[0], py_min2);
        let fmax = |v: &[f64]| v.iter().copied().fold(v[0], py_max2);
        tiles_covering(fmin(&lats), fmin(&lons), fmax(&lats), fmax(&lons), self.zoom)
    }

    /// `flat_tiles`: the union of the tiles covering every capture patch
    /// (deduped, sorted).
    pub fn flat_tiles(&self, datum: &SceneDatum) -> Vec<(i64, i64)> {
        let mut seen = BTreeSet::new();
        for &(cx, cy, hw, hh) in &self.patches {
            seen.extend(self.patch_tiles(datum, cx, cy, hw, hh));
        }
        seen.into_iter().collect()
    }

    /// `cap_detail`: lower `zoom` until the capture fits `max_tiles`
    /// (Python defaults 500 and 8). Returns `true` if the zoom was reduced.
    pub fn cap_detail(&mut self, datum: &SceneDatum, max_tiles: usize, min_zoom: i64) -> bool {
        let mut reduced = false;
        while self.flat_tiles(datum).len() > max_tiles && self.zoom > min_zoom {
            self.zoom -= 1;
            reduced = true;
        }
        reduced
    }

    /// `visible_tiles`: tiles covering a `+-radius_m` square around the datum.
    pub fn visible_tiles(&self, datum: &SceneDatum) -> Vec<(i64, i64)> {
        self.patch_tiles(datum, 0.0, 0.0, self.radius_m, self.radius_m)
    }

    /// `quad_local`: tile `(x, y)` as two Z=0 triangles in local metres,
    /// six `(position, (u, v))` vertices.
    pub fn quad_local(&self, datum: &SceneDatum, x: i64, y: i64) -> Vec<(V3, (f64, f64))> {
        let (lat_s, lon_w, lat_n, lon_e) = tile_bbox(x, y, self.zoom);
        let nw = datum.geodetic_to_local(lat_n, lon_w, None);
        let ne = datum.geodetic_to_local(lat_n, lon_e, None);
        let se = datum.geodetic_to_local(lat_s, lon_e, None);
        let sw = datum.geodetic_to_local(lat_s, lon_w, None);
        vec![
            (nw, (0.0, 0.0)),
            (ne, (1.0, 0.0)),
            (se, (1.0, 1.0)),
            (nw, (0.0, 0.0)),
            (se, (1.0, 1.0)),
            (sw, (0.0, 1.0)),
        ]
    }
}

// ---- Disk cache (LRU) ---------------------------------------------------------

/// One cached file as eviction sees it.
#[derive(Clone, Debug, PartialEq)]
pub struct CacheFile {
    pub path: PathBuf,
    /// `st_mtime`, seconds.
    pub mtime: f64,
    pub size: u64,
}

/// The files `_evict_if_needed` deletes, in deletion order: oldest `mtime`
/// first (a stable sort, as Python's `sorted`) until the total fits
/// `max_bytes`. Pure; [`TileCache`] applies it.
pub fn eviction_plan(files: &[CacheFile], max_bytes: u64) -> Vec<usize> {
    let mut total: i128 = files.iter().map(|f| f.size as i128).sum();
    let mut out = Vec::new();
    if total <= max_bytes as i128 {
        return out;
    }
    let mut order: Vec<usize> = (0..files.len()).collect();
    order.sort_by(|&a, &b| files[a].mtime.partial_cmp(&files[b].mtime).unwrap_or(std::cmp::Ordering::Equal));
    for i in order {
        if total <= max_bytes as i128 {
            break;
        }
        total -= files[i].size as i128;
        out.push(i);
    }
    out
}

/// `TileCache`: bytes in, bytes out, `root/<source_id>/<z>/<x>/<y>`, LRU by
/// budget (default 256 MiB).
#[derive(Clone, Debug)]
pub struct TileCache {
    pub root: PathBuf,
    pub max_bytes: u64,
}

fn mtime_secs(md: &std::fs::Metadata) -> f64 {
    match md.modified() {
        Ok(t) => match t.duration_since(std::time::UNIX_EPOCH) {
            Ok(d) => d.as_secs_f64(),
            Err(e) => -e.duration().as_secs_f64(),
        },
        Err(_) => 0.0,
    }
}

fn set_mtime_secs(p: &Path, secs: f64) {
    // Best effort, like Python's `os.utime` in a try/except OSError.
    let t = if secs >= 0.0 {
        std::time::UNIX_EPOCH + std::time::Duration::from_secs_f64(secs)
    } else {
        std::time::UNIX_EPOCH - std::time::Duration::from_secs_f64(-secs)
    };
    if let Ok(f) = std::fs::OpenOptions::new().write(true).open(p) {
        let _ = f.set_modified(t);
    }
}

fn walk_files(dir: &Path, out: &mut Vec<PathBuf>) {
    if let Ok(rd) = std::fs::read_dir(dir) {
        for e in rd.flatten() {
            let p = e.path();
            if p.is_dir() {
                walk_files(&p, out);
            } else if p.is_file() {
                out.push(p);
            }
        }
    }
}

impl TileCache {
    pub const DEFAULT_MAX_BYTES: u64 = 256 * 1024 * 1024;

    /// `TileCache(root, max_bytes)`: creates `root`.
    pub fn new(root: impl Into<PathBuf>, max_bytes: u64) -> TileCache {
        let root = root.into();
        let _ = std::fs::create_dir_all(&root);
        TileCache { root, max_bytes }
    }

    pub fn path_for(&self, source_id: &str, x: i64, y: i64, z: i64) -> PathBuf {
        self.root.join(source_id).join(z.to_string()).join(x.to_string()).join(y.to_string())
    }

    /// `get`: the cached bytes (and a bump of the file's mtime so the LRU
    /// treats it as fresh), or `None`.
    pub fn get(&self, source_id: &str, x: i64, y: i64, z: i64) -> Option<Vec<u8>> {
        let p = self.path_for(source_id, x, y, z);
        if p.exists() {
            let data = std::fs::read(&p).ok()?;
            if let Ok(md) = std::fs::metadata(&p) {
                set_mtime_secs(&p, mtime_secs(&md) + 1.0);
            }
            return Some(data);
        }
        None
    }

    fn files(&self) -> Vec<PathBuf> {
        let mut v = Vec::new();
        walk_files(&self.root, &mut v);
        v
    }

    /// `put`: store the tile (empty data is ignored), stamp it newer than
    /// every other tile, evict down to the budget.
    pub fn put(&self, source_id: &str, x: i64, y: i64, z: i64, data: &[u8]) {
        if data.is_empty() {
            return;
        }
        let p = self.path_for(source_id, x, y, z);
        if let Some(parent) = p.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        if std::fs::write(&p, data).is_err() {
            return;
        }
        let files = self.files();
        // `_stamp_newest`: an mtime strictly newer than every other tile.
        if let Ok(md) = std::fs::metadata(&p) {
            let mine = mtime_secs(&md);
            let mut newest = mine;
            let mut first = true;
            for f in files.iter().filter(|f| **f != p) {
                if let Ok(m) = std::fs::metadata(f) {
                    let t = mtime_secs(&m);
                    if first || t > newest {
                        newest = t;
                    }
                    first = false;
                }
            }
            if first {
                newest = mine;
            }
            if mine <= newest {
                set_mtime_secs(&p, newest + 1.0);
            }
        }
        self.evict_if_needed(&files);
    }

    fn evict_if_needed(&self, files: &[PathBuf]) {
        let infos: Vec<CacheFile> = files
            .iter()
            .filter_map(|f| {
                std::fs::metadata(f).ok().map(|m| CacheFile { path: f.clone(), mtime: mtime_secs(&m), size: m.len() })
            })
            .collect();
        for i in eviction_plan(&infos, self.max_bytes) {
            let _ = std::fs::remove_file(&infos[i].path);
        }
    }
}
