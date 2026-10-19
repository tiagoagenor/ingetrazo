// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! The crate against what the Python functions return
//! (scripts/golden/make_georef_fixtures.py): every float is compared as bits.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use ingetrazo_formats::pyjson::{dumps, loads, Options, Value};
use ingetrazo_georef::datum::{self, SceneDatum};
use ingetrazo_georef::dem::{self, aws_terrain, DemSampler, Field};
use ingetrazo_georef::geotiff::{self, py_hypot2};
use ingetrazo_georef::tiles;
use ingetrazo_math::V3;

fn data(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data").join(name)
}

fn records(name: &str) -> Vec<Value> {
    let text = std::fs::read_to_string(data(name)).unwrap();
    text.lines().map(|l| loads(l).unwrap()).collect()
}

fn get<'a>(v: &'a Value, k: &str) -> &'a Value {
    v.get(k).unwrap_or_else(|| panic!("missing key {k}"))
}
fn s<'a>(v: &'a Value, k: &str) -> &'a str {
    match get(v, k) {
        Value::Str(x) => x,
        o => panic!("{k}: {o:?}"),
    }
}
fn int(v: &Value, k: &str) -> i64 {
    get(v, k).as_i64().unwrap_or_else(|| panic!("{k} not an int"))
}
fn boolean(v: &Value, k: &str) -> bool {
    match get(v, k) {
        Value::Bool(b) => *b,
        o => panic!("{k}: {o:?}"),
    }
}
fn arr<'a>(v: &'a Value, k: &str) -> &'a [Value] {
    match get(v, k) {
        Value::Array(a) => a,
        o => panic!("{k}: {o:?}"),
    }
}
fn is_null(v: &Value, k: &str) -> bool {
    matches!(get(v, k), Value::Null)
}
fn hd(h: &str) -> f64 {
    f64::from_bits(u64::from_str_radix(h, 16).unwrap())
}
fn hf(h: &str) -> f32 {
    f32::from_bits(u32::from_str_radix(h, 16).unwrap())
}
fn d(v: &Value, k: &str) -> f64 {
    hd(s(v, k))
}
fn vd(v: &Value) -> f64 {
    match v {
        Value::Str(x) => hd(x),
        o => panic!("{o:?}"),
    }
}
fn v3(v: &Value) -> V3 {
    match v {
        Value::Array(a) => match (&a[0], &a[1], &a[2]) {
            (Value::Str(x), Value::Str(y), Value::Str(z)) => V3(hf(x), hf(y), hf(z)),
            o => panic!("{o:?}"),
        },
        o => panic!("{o:?}"),
    }
}
fn f32s(h: &str) -> Vec<f32> {
    (0..h.len() / 8).map(|i| hf(&h[i * 8..i * 8 + 8])).collect()
}
fn unhex(h: &str) -> Vec<u8> {
    (0..h.len() / 2).map(|i| u8::from_str_radix(&h[i * 2..i * 2 + 2], 16).unwrap()).collect()
}

/// Collects mismatches per function so one run shows which ones differ.
#[derive(Default)]
struct Tally {
    checked: BTreeMap<String, usize>,
    failed: BTreeMap<String, Vec<String>>,
}

impl Tally {
    fn ok(&mut self, f: &str) {
        *self.checked.entry(f.to_string()).or_default() += 1;
    }
    fn fail(&mut self, f: &str, msg: String) {
        self.ok(f);
        self.failed.entry(f.to_string()).or_default().push(msg);
    }
    fn f64(&mut self, f: &str, ctx: &str, got: f64, want: f64) {
        if got.to_bits() == want.to_bits() || (got.is_nan() && want.is_nan()) {
            self.ok(f);
        } else {
            self.fail(f, format!("{ctx}: got {got:e} ({:016x}) want {want:e} ({:016x})", got.to_bits(), want.to_bits()));
        }
    }
    fn f32(&mut self, f: &str, ctx: &str, got: f32, want: f32) {
        if got.to_bits() == want.to_bits() || (got.is_nan() && want.is_nan()) {
            self.ok(f);
        } else {
            self.fail(f, format!("{ctx}: got {got:e} want {want:e}"));
        }
    }
    fn eq<T: PartialEq + std::fmt::Debug>(&mut self, f: &str, ctx: &str, got: T, want: T) {
        if got == want {
            self.ok(f);
        } else {
            self.fail(f, format!("{ctx}: got {got:?} want {want:?}"));
        }
    }
    fn finish(self) {
        let mut bad = false;
        for (f, n) in &self.checked {
            let fails = self.failed.get(f).map_or(0, |v| v.len());
            eprintln!("{f}: {n} checked, {fails} differ");
        }
        for (f, v) in &self.failed {
            bad = true;
            eprintln!("--- {f}: {} mismatches, first:", v.len());
            for m in v.iter().take(5) {
                eprintln!("    {m}");
            }
        }
        assert!(!bad, "some values differ from Python");
    }
}

fn v3_vs(t: &mut Tally, f: &str, ctx: &str, got: V3, want: V3) {
    t.f32(f, ctx, got.0, want.0);
    t.f32(f, ctx, got.1, want.1);
    t.f32(f, ctx, got.2, want.2);
}

fn ds(t: &mut Tally, f: &str, ctx: &str, got: &[f64], want: &[Value]) {
    assert_eq!(got.len(), want.len());
    for (g, w) in got.iter().zip(want) {
        t.f64(f, ctx, *g, vd(w));
    }
}

#[test]
fn datum_matches_python() {
    let mut t = Tally::default();
    for r in records("datum.jsonl") {
        match s(&r, "fn") {
            "zone_for_lon" => {
                let lon = d(&r, "lon");
                t.eq("zone_for_lon", &format!("{lon}"), datum::zone_for_lon(lon), int(&r, "out"));
            }
            "lon0" => {
                let z = int(&r, "zone");
                t.f64("lon0", &format!("{z}"), datum::lon0(z), d(&r, "out"));
            }
            "utm_forward" => {
                let (lat, lon, zone) = (d(&r, "lat"), d(&r, "lon"), int(&r, "zone"));
                let (e, n) = datum::utm_forward(lat, lon, zone);
                let ctx = format!("{lat} {lon} {zone}");
                t.f64("utm_forward.easting", &ctx, e, d(&r, "e"));
                t.f64("utm_forward.northing", &ctx, n, d(&r, "n"));
            }
            "utm_inverse" => {
                let (e, n, zone) = (d(&r, "e"), d(&r, "n"), int(&r, "zone"));
                let (lat, lon) = datum::utm_inverse(e, n, zone, boolean(&r, "northern"));
                let ctx = format!("{e} {n} {zone}");
                t.f64("utm_inverse.lat", &ctx, lat, d(&r, "lat"));
                t.f64("utm_inverse.lon", &ctx, lon, d(&r, "lon"));
            }
            "wrap_deg" => {
                let x = d(&r, "x");
                t.f64("wrap_deg", &format!("{x}"), datum::wrap_deg(x), d(&r, "out"));
            }
            "hypot" => {
                let (x, y) = (d(&r, "x"), d(&r, "y"));
                t.f64("py_hypot2", &format!("{x} {y}"), py_hypot2(x, y), d(&r, "out"));
            }
            "datum" => {
                let sd = SceneDatum::new(d(&r, "lat"), d(&r, "lon"), d(&r, "alt"), d(&r, "north"));
                let o = get(&r, "out");
                let ctx = format!("datum {} {}", sd.lat, sd.lon);
                t.f64("datum.lat", &ctx, sd.lat, d(o, "lat"));
                t.f64("datum.lon", &ctx, sd.lon, d(o, "lon"));
                t.f64("datum.alt", &ctx, sd.alt, d(o, "alt"));
                t.f64("datum.north", &ctx, sd.north, d(o, "north"));
                t.eq("datum.zone", &ctx, sd.zone, int(o, "zone"));
                t.eq("datum.northern", &ctx, sd.northern, boolean(o, "northern"));
                t.eq("datum.hemisphere", &ctx, sd.hemisphere(), s(o, "hemisphere"));
                let (e0, n0) = sd.anchor_utm();
                t.f64("datum.east0", &ctx, e0, d(o, "east0"));
                t.f64("datum.north0", &ctx, n0, d(o, "north0"));
                let (c, sn) = sd.cos_sin_north();
                t.f64("datum.cos_n", &ctx, c, d(o, "cos_n"));
                t.f64("datum.sin_n", &ctx, sn, d(o, "sin_n"));
                v3_vs(&mut t, "datum.north_vector", &ctx, sd.north_vector(), v3(get(o, "north_vector")));
                let dumped = dumps(&sd.to_dict(), &Options::default()).unwrap();
                t.eq("datum.to_dict", &ctx, dumped, s(o, "to_dict").to_string());
                for p in arr(&r, "points") {
                    let op = s(p, "op");
                    let name = format!("datum.{op}");
                    let out = arr(p, "out");
                    match op {
                        "geodetic_to_local" => {
                            let alt = if is_null(p, "alt") { None } else { Some(d(p, "alt")) };
                            let got = sd.geodetic_to_local(d(p, "lat"), d(p, "lon"), alt);
                            v3_vs(&mut t, &name, &ctx, got, v3(get(p, "out")));
                        }
                        "local_to_geodetic" => {
                            let (a, b, c) = sd.local_to_geodetic(v3(get(p, "p")));
                            ds(&mut t, &name, &ctx, &[a, b, c], out);
                        }
                        "local_to_utm" => {
                            let (a, b, c) = sd.local_to_utm(v3(get(p, "p")));
                            ds(&mut t, &name, &ctx, &[a, b, c], out);
                        }
                        "utm_to_local" => {
                            let alt = if is_null(p, "alt") { 0.0 } else { d(p, "alt") };
                            let got = sd.utm_to_local(d(p, "e"), d(p, "n"), alt);
                            v3_vs(&mut t, &name, &ctx, got, v3(get(p, "out")));
                        }
                        "grid_to_local_xy" => {
                            let (a, b) = sd.grid_to_local_xy(d(p, "a"), d(p, "b"));
                            ds(&mut t, &name, &ctx, &[a, b], out);
                        }
                        "local_to_grid_xy" => {
                            let (a, b) = sd.local_to_grid_xy(d(p, "a"), d(p, "b"));
                            ds(&mut t, &name, &ctx, &[a, b], out);
                        }
                        o => panic!("op {o}"),
                    }
                }
            }
            "from_dict" => {
                let input = loads(s(&r, "data")).unwrap();
                let got = SceneDatum::from_dict(&input);
                match (r.get("error"), got) {
                    (Some(_), Err(_)) => t.ok("from_dict"),
                    (Some(e), Ok(_)) => t.fail("from_dict", format!("{input:?}: Python raised {e:?}")),
                    (None, Err(e)) => t.fail("from_dict", format!("{input:?}: {e}")),
                    (None, Ok(sd)) => {
                        let o = get(&r, "out");
                        let ctx = format!("{input:?}");
                        t.f64("from_dict.lat", &ctx, sd.lat, d(o, "lat"));
                        t.f64("from_dict.lon", &ctx, sd.lon, d(o, "lon"));
                        t.f64("from_dict.alt", &ctx, sd.alt, d(o, "alt"));
                        t.f64("from_dict.north", &ctx, sd.north, d(o, "north"));
                        t.eq("from_dict.zone", &ctx, sd.zone, int(o, "zone"));
                        let dumped = dumps(&sd.to_dict(), &Options::default()).unwrap();
                        t.eq("from_dict.to_dict", &ctx, dumped, s(o, "to_dict").to_string());
                    }
                }
            }
            o => panic!("fn {o}"),
        }
    }
    t.finish();
}

#[test]
fn dem_matches_python() {
    let mut t = Tally::default();
    for r in records("dem.jsonl") {
        match s(&r, "fn") {
            "bilinear" => {
                let (h, w) = (int(&r, "h") as usize, int(&r, "w") as usize);
                let field = Field::new(h, w, f32s(s(&r, "field")));
                let (px, py) = (d(&r, "px"), d(&r, "py"));
                t.f64("bilinear", &format!("{h}x{w} {px} {py}"), dem::bilinear(&field, px, py), d(&r, "out"));
            }
            "decode" => {
                let (h, w) = (int(&r, "h") as usize, int(&r, "w") as usize);
                let got = dem::decode_elevation(&unhex(s(&r, "rgba")), w, h, s(&r, "encoding"));
                let want = f32s(s(&r, "out"));
                for (i, (g, wv)) in got.data.iter().zip(&want).enumerate() {
                    t.f32("decode_elevation", &format!("{} px {i}", s(&r, "encoding")), *g, *wv);
                }
            }
            "deg2num" => {
                let (lat, lon, z) = (d(&r, "lat"), d(&r, "lon"), int(&r, "zoom"));
                let (x, y) = tiles::deg2num(lat, lon, z);
                let ctx = format!("{lat} {lon} {z}");
                t.f64("deg2num.x", &ctx, x, d(&r, "x"));
                t.f64("deg2num.y", &ctx, y, d(&r, "y"));
            }
            "pixel_in_tile" => {
                let (lat, lon, z) = (d(&r, "lat"), d(&r, "lon"), int(&r, "zoom"));
                let (tx, ty, px, py) = dem::pixel_in_tile(lat, lon, z);
                let ctx = format!("{lat} {lon} {z}");
                t.eq("pixel_in_tile.tile", &ctx, (tx, ty), (int(&r, "tx"), int(&r, "ty")));
                t.f64("pixel_in_tile.px", &ctx, px, d(&r, "px"));
                t.f64("pixel_in_tile.py", &ctx, py, d(&r, "py"));
            }
            "tiles_covering" => {
                let got = tiles::tiles_covering(d(&r, "s"), d(&r, "w"), d(&r, "n"), d(&r, "e"), int(&r, "zoom"));
                let want: Vec<(i64, i64)> = arr(&r, "out")
                    .iter()
                    .map(|p| match p {
                        Value::Array(a) => (a[0].as_i64().unwrap(), a[1].as_i64().unwrap()),
                        o => panic!("{o:?}"),
                    })
                    .collect();
                t.eq("tiles_covering", &format!("{:?}", r.get("s")), got, want);
            }
            "sampler" => {
                let dat = SceneDatum::new(d(&r, "lat"), d(&r, "lon"), d(&r, "alt"), d(&r, "north"));
                let zoom = int(&r, "zoom");
                let mut sm = DemSampler::new(dat, aws_terrain(), zoom);
                for tile in arr(&r, "tiles") {
                    sm.insert_field(int(tile, "tx"), int(tile, "ty"), Field::new(256, 256, f32s(s(tile, "field"))));
                }
                for q in arr(&r, "queries") {
                    let op = s(q, "op");
                    let name = format!("sampler.{op}");
                    match op {
                        "has_tile" => {
                            t.eq(&name, "", sm.has_tile(d(q, "lat"), d(q, "lon")), boolean(q, "out"))
                        }
                        _ => {
                            let got = if op == "elevation_at" {
                                sm.elevation_at(d(q, "lat"), d(q, "lon"))
                            } else {
                                sm.elevation_at_local(v3(get(q, "p")))
                            };
                            match (got, is_null(q, "out")) {
                                (None, true) => t.ok(&name),
                                (Some(g), false) => t.f64(&name, "", g, d(q, "out")),
                                (g, _) => t.fail(&name, format!("got {g:?}, python null={}", is_null(q, "out"))),
                            }
                        }
                    }
                }
            }
            o => panic!("fn {o}"),
        }
    }
    t.finish();
}

fn ints(v: &Value) -> Vec<i64> {
    match v {
        Value::Array(a) => a.iter().map(|x| x.as_i64().unwrap()).collect(),
        o => panic!("{o:?}"),
    }
}

fn opt_i(v: &Value) -> Option<i64> {
    match v {
        Value::Null => None,
        o => o.as_i64(),
    }
}

fn compare_info(t: &mut Tally, ctx: &str, info: &geotiff::GeoTiffInfo, want: &Value) {
    t.eq("info.width", ctx, info.width, int(want, "width"));
    t.eq("info.height", ctx, info.height, int(want, "height"));
    t.eq("info.samples", ctx, info.samples, int(want, "samples"));
    t.eq("info.bits", ctx, info.bits.clone(), ints(get(want, "bits")));
    t.eq("info.compression", ctx, info.compression, int(want, "compression"));
    t.eq("info.photometric", ctx, info.photometric, int(want, "photometric"));
    t.eq("info.planar", ctx, info.planar, int(want, "planar"));
    t.eq("info.tiled", ctx, info.tiled, boolean(want, "tiled"));
    t.eq("info.tile_w", ctx, info.tile_w, int(want, "tile_w"));
    t.eq("info.tile_h", ctx, info.tile_h, int(want, "tile_h"));
    t.eq("info.offsets", ctx, info.offsets.clone(), ints(get(want, "offsets")));
    t.eq("info.bytecounts", ctx, info.bytecounts.clone(), ints(get(want, "bytecounts")));
    t.eq("info.predictor", ctx, info.predictor, int(want, "predictor"));
    let jt: String = info.jpeg_tables.iter().map(|b| format!("{b:02x}")).collect();
    t.eq("info.jpeg_tables", ctx, jt, s(want, "jpeg_tables").to_string());
    t.eq("info.extra_samples", ctx, info.extra_samples.clone(), ints(get(want, "extra_samples")));
    t.eq("info.sample_format", ctx, info.sample_format.clone(), ints(get(want, "sample_format")));
    t.eq("info.big", ctx, info.big, boolean(want, "big"));
    t.eq(
        "info.byte_order",
        ctx,
        if info.little_endian { "<" } else { ">" },
        s(want, "byte_order"),
    );
    match (info.transform, get(want, "transform")) {
        (None, Value::Null) => t.ok("info.transform"),
        (Some(g), w @ Value::Array(_)) => ds(t, "info.transform", ctx, &g, match w {
            Value::Array(a) => a,
            _ => unreachable!(),
        }),
        (g, w) => t.fail("info.transform", format!("{ctx}: {g:?} vs {w:?}")),
    }
    t.eq("info.epsg", ctx, info.epsg, opt_i(get(want, "epsg")));
    t.eq("info.zone", ctx, info.zone, opt_i(get(want, "zone")));
    let northern = match get(want, "northern") {
        Value::Null => None,
        Value::Bool(b) => Some(*b),
        o => panic!("{o:?}"),
    };
    t.eq("info.northern", ctx, info.northern, northern);
    t.eq("info.geographic", ctx, info.geographic, boolean(want, "geographic"));
    t.eq("info.citation", ctx, info.citation.clone(), s(want, "citation").to_string());
    let nodata = match get(want, "nodata") {
        Value::Null => None,
        Value::Str(x) => Some(x.clone()),
        o => panic!("{o:?}"),
    };
    t.eq("info.nodata", ctx, info.nodata.clone(), nodata);
    t.eq("info.georeferenced", ctx, info.georeferenced(), boolean(want, "georeferenced"));
}

#[test]
fn geotiff_matches_python() {
    let mut t = Tally::default();
    for r in records("geotiff.jsonl") {
        if s(&r, "fn") == "unsupported_reason" {
            let base = geotiff::read_info(data("tif/std_c8_p1_t0.tif")).unwrap();
            let mut info = base.clone();
            if let Value::Object(set) = get(&r, "set") {
                for (k, v) in set {
                    let list = || -> Vec<i64> { ints(v) };
                    let n = || v.as_i64().unwrap();
                    match k.as_str() {
                        "compression" => info.compression = n(),
                        "planar" => info.planar = n(),
                        "bits" => info.bits = list(),
                        "sample_format" => info.sample_format = list(),
                        "photometric" => info.photometric = n(),
                        "samples" => info.samples = n(),
                        o => panic!("mutation {o}"),
                    }
                }
            }
            let want = match get(&r, "out") {
                Value::Null => None,
                Value::Str(x) => Some(x.clone()),
                o => panic!("{o:?}"),
            };
            t.eq("unsupported_reason", &format!("{:?}", get(&r, "set")), geotiff::unsupported_reason(&info), want);
            continue;
        }
        let file = s(&r, "file");
        let name = Path::new(file).file_name().unwrap().to_string_lossy().into_owned();
        let got = geotiff::read_info(data(file));
        if let Some(e) = r.get("info_error") {
            match got {
                Err(g) => {
                    if s(e, "cls") == "GeoTiffError" {
                        t.eq("info_error.msg", file, g.0, s(e, "msg").to_string());
                    } else {
                        t.ok("info_error.other");
                    }
                }
                Ok(_) => t.fail("info_error", format!("{file}: Python raised {e:?}")),
            }
            continue;
        }
        let info = match got {
            Ok(i) => i,
            Err(e) => {
                t.fail("read_info", format!("{file}: {e}"));
                continue;
            }
        };
        compare_info(&mut t, file, &info, get(&r, "info"));
        t.eq("describe", file, geotiff::describe(&info), s(&r, "describe").to_string());
        let want_reason = match get(&r, "unsupported") {
            Value::Null => None,
            Value::Str(x) => Some(x.clone()),
            o => panic!("{o:?}"),
        };
        t.eq("unsupported_reason.file", file, geotiff::unsupported_reason(&info), want_reason);
        if r.get("ground_size").is_some() {
            let (w, h) = info.ground_size();
            ds(&mut t, "ground_size", file, &[w, h], arr(&r, "ground_size"));
            for p in arr(&r, "pixel_to_map") {
                let (x, y) = info.pixel_to_map(d(p, "col"), d(p, "row"));
                ds(&mut t, "pixel_to_map", file, &[x, y], arr(p, "out"));
            }
        }
        if r.get("corners").is_some() {
            let got = info.corners_geodetic();
            let want = arr(&r, "corners");
            assert_eq!(got.len(), want.len());
            for (g, w) in got.iter().zip(want) {
                let w = match w {
                    Value::Array(a) => a,
                    o => panic!("{o:?}"),
                };
                ds(&mut t, "corners_geodetic", file, &[g.0, g.1], w);
            }
        }
        if info.compression != geotiff::COMPRESSION_JPEG {
            let cancel = geotiff::read_rgba(&info, 8192, Some(&mut |_| false));
            match (get(&r, "cancel"), cancel) {
                (Value::Null, Ok(_)) => t.ok("cancel"),
                (Value::Null, Err(e)) => t.fail("cancel", format!("{file}: unexpected {e}")),
                (w, Err(e)) => {
                    if s(w, "cls") == "GeoTiffError" {
                        t.eq("cancel", file, e.0, s(w, "msg").to_string());
                    } else {
                        t.ok("cancel");
                    }
                }
                (w, Ok(_)) => t.fail("cancel", format!("{file}: Python raised {w:?}")),
            }
        }
        for run in arr(&r, "rgba") {
            let max_px = int(run, "max_px");
            let mut seen = Vec::new();
            let got = {
                let mut cb = |fr: f64| {
                    seen.push(fr);
                    true
                };
                geotiff::read_rgba(&info, max_px, Some(&mut cb))
            };
            let ctx = format!("{name} max_px={max_px}");
            if let Some(e) = run.get("error") {
                match got {
                    Err(g) => {
                        if s(e, "cls") == "GeoTiffError" {
                            t.eq("read_rgba.error", &ctx, g.0, s(e, "msg").to_string());
                        } else {
                            t.ok("read_rgba.error_other");
                        }
                    }
                    Ok(_) => t.fail("read_rgba.error", format!("{ctx}: Python raised {e:?}")),
                }
                continue;
            }
            match got {
                Err(e) => t.fail("read_rgba", format!("{ctx}: {e}")),
                Ok((pic, factor)) => {
                    t.eq("read_rgba.factor", &ctx, factor, int(run, "factor"));
                    t.eq("read_rgba.shape", &ctx, (pic.h as i64, pic.w as i64), (int(run, "h"), int(run, "w")));
                    t.eq("read_rgba.pixels", &ctx, pic.data == unhex(s(run, "hex")), true);
                    ds(&mut t, "read_rgba.progress", &ctx, &seen, arr(run, "progress"));
                }
            }
        }
    }
    t.finish();
}
