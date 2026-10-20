// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! The second half of the crate against what the Python functions return
//! (scripts/golden/make_georef2_fixtures.py): tiles, the tile fetcher's
//! planning, terrain, surfaces, geopaths, survey points, geocoding, profiles
//! and KML/GeoJSON import. Every float is compared as bits.

use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};

use ingetrazo_formats::pyjson::{dumps, loads, Options, Value};
use ingetrazo_georef::datum::SceneDatum;
use ingetrazo_georef::dem::{aws_terrain, DemSampler, ElevationSampler, Field};
use ingetrazo_georef::geocode::{self, Place};
use ingetrazo_georef::geoimport::{self, GeoFeature};
use ingetrazo_georef::geopath::GeoPath;
use ingetrazo_georef::points::{self, GeoPoint, PointRow};
use ingetrazo_georef::profile::{self, Profile, ProfileSample};
use ingetrazo_georef::surface;
use ingetrazo_georef::terrain::{self, TerrainObject};
use ingetrazo_georef::tile_fetcher::{self, RequestOutcome, TileKey, TilePlanner};
use ingetrazo_georef::tiles::{self, CacheFile, TileCache, TileLayer, TileSource};
use ingetrazo_math::{Hypot, V3};

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
fn has(v: &Value, k: &str) -> bool {
    v.get(k).is_some()
}
fn s<'a>(v: &'a Value, k: &str) -> &'a str {
    match get(v, k) {
        Value::Str(x) => x,
        o => panic!("{k}: {o:?}"),
    }
}
fn opt_s<'a>(v: &'a Value, k: &str) -> Option<&'a str> {
    match get(v, k) {
        Value::Str(x) => Some(x),
        Value::Null => None,
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
fn items(v: &Value) -> &[Value] {
    match v {
        Value::Array(a) => a,
        o => panic!("{o:?}"),
    }
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
fn od(v: &Value) -> Option<f64> {
    match v {
        Value::Null => None,
        o => Some(vd(o)),
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
fn v3s(v: &Value) -> Vec<V3> {
    items(v).iter().map(v3).collect()
}
fn unhex(h: &str) -> Vec<u8> {
    (0..h.len() / 2).map(|i| u8::from_str_radix(&h[i * 2..i * 2 + 2], 16).unwrap()).collect()
}
fn dumps_default(v: &Value) -> String {
    dumps(v, &Options::default()).unwrap()
}

fn fnv64(bytes: &[u8]) -> String {
    let mut h: u64 = 0xCBF29CE484222325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001B3);
    }
    format!("{h:016x}")
}

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
    fn v3(&mut self, f: &str, ctx: &str, got: V3, want: V3) {
        self.f32(f, ctx, got.0, want.0);
        self.f32(f, ctx, got.1, want.1);
        self.f32(f, ctx, got.2, want.2);
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

fn datum_of(v: &Value) -> SceneDatum {
    SceneDatum::new(d(v, "lat"), d(v, "lon"), d(v, "alt"), d(v, "north"))
}

/// `field_of(seed)` of the fixture script (dyadic rationals, exact in f32).
fn field_of(seed: i64) -> Field {
    let mut out = Vec::with_capacity(256 * 256);
    for y in 0..256i64 {
        for x in 0..256i64 {
            let v = (((x * 7 + y * 13 + seed * 101) % 509) as f64) * 0.0625
                + (((x * y + seed) % 31) as f64) * 0.125
                + ((x + y) as f64) * 0.25
                + (seed as f64) * 10.0;
            out.push(v as f32);
        }
    }
    Field::new(256, 256, out)
}

fn sampler_of(datum: SceneDatum, zoom: i64, specs: &[Value]) -> DemSampler {
    let mut sm = DemSampler::new(datum, aws_terrain(), zoom);
    for t in specs {
        sm.insert_field(int(t, "tx"), int(t, "ty"), field_of(int(t, "seed")));
    }
    sm
}

fn src_of(v: &Value) -> TileSource {
    TileSource {
        id: s(v, "id").to_string(),
        name: s(v, "name").to_string(),
        url_template: s(v, "url_template").to_string(),
        max_zoom: int(v, "max_zoom"),
        attribution: s(v, "attribution").to_string(),
        subdomains: arr(v, "subdomains")
            .iter()
            .map(|x| match x {
                Value::Str(t) => t.clone(),
                o => panic!("{o:?}"),
            })
            .collect(),
        tile_size: int(v, "tile_size"),
    }
}

fn layer_of(v: &Value) -> TileLayer {
    TileLayer {
        source: src_of(get(v, "source")),
        zoom: int(v, "zoom"),
        radius_m: d(v, "radius_m"),
        visible: boolean(v, "visible"),
        opacity: d(v, "opacity"),
        patches: arr(v, "patches")
            .iter()
            .map(|p| {
                let a = items(p);
                (vd(&a[0]), vd(&a[1]), vd(&a[2]), vd(&a[3]))
            })
            .collect(),
    }
}

fn tile_list(rec: &Value) -> (i64, String, Option<Vec<(i64, i64)>>) {
    let list = rec.get("list").map(|l| items(l).iter().map(|p| (items(p)[0].as_i64().unwrap(), items(p)[1].as_i64().unwrap())).collect());
    (int(rec, "n"), s(rec, "hash").to_string(), list)
}

fn check_tiles(t: &mut Tally, f: &str, ctx: &str, got: &[(i64, i64)], want: &Value) {
    let (n, hash, list) = tile_list(want);
    t.eq(&format!("{f}.n"), ctx, got.len() as i64, n);
    let mut b = Vec::new();
    for &(x, y) in got {
        b.extend_from_slice(&x.to_be_bytes());
        b.extend_from_slice(&y.to_be_bytes());
    }
    t.eq(&format!("{f}.hash"), ctx, fnv64(&b), hash);
    if let Some(l) = list {
        t.eq(&format!("{f}.list"), ctx, got.to_vec(), l);
    }
}

fn scratch(tag: &str) -> PathBuf {
    use std::sync::atomic::{AtomicUsize, Ordering};
    static N: AtomicUsize = AtomicUsize::new(0);
    let p = std::env::temp_dir().join(format!("ingetrazo-georef-{tag}-{}-{}", std::process::id(), N.fetch_add(1, Ordering::SeqCst)));
    let _ = std::fs::remove_dir_all(&p);
    std::fs::create_dir_all(&p).unwrap();
    p
}

fn rel_files(root: &Path) -> Vec<String> {
    fn walk(dir: &Path, root: &Path, out: &mut Vec<String>) {
        for e in std::fs::read_dir(dir).unwrap().flatten() {
            let p = e.path();
            if p.is_dir() {
                walk(&p, root, out);
            } else {
                out.push(p.strip_prefix(root).unwrap().to_string_lossy().replace('\\', "/"));
            }
        }
    }
    let mut v = Vec::new();
    walk(root, root, &mut v);
    v.sort();
    v
}

// ---------------------------------------------------------------------------
// tiles
// ---------------------------------------------------------------------------

/// Python's exception class (or `NotModelled` for what these types do not
/// hold) against the fixture's record of a `from_dict`-like call.
fn expect_err(t: &mut Tally, f: &str, ctx: &str, rec: &Value, got: Result<(), ingetrazo_georef::pyconv::PyErr>) {
    match (rec.get("error"), got) {
        (Some(Value::Str(want)), Err(e)) => t.eq(f, ctx, e.kind, want.as_str()),
        (Some(w), Ok(())) => t.fail(f, format!("{ctx}: Python raised {w:?}, got Ok")),
        (None, Err(e)) if e.kind == "NotModelled" && has(rec, "nm") => t.ok(&format!("{f}.notmodelled")),
        (None, Err(e)) => t.fail(f, format!("{ctx}: Python accepted, got {e:?}")),
        (None, Ok(())) => t.ok(f),
        (w, g) => panic!("{w:?} {g:?}"),
    }
}

#[test]
fn tiles_match_python() {
    let mut t = Tally::default();
    for r in records("tiles.jsonl") {
        match s(&r, "fn") {
            "num2deg" => {
                let (x, y, z) = (d(&r, "x"), d(&r, "y"), int(&r, "zoom"));
                let (lat, lon) = tiles::num2deg(x, y, z);
                let ctx = format!("{x} {y} {z}");
                t.f64("num2deg.lat", &ctx, lat, d(&r, "lat"));
                t.f64("num2deg.lon", &ctx, lon, d(&r, "lon"));
            }
            "tile_bbox" => {
                let (x, y, z) = (int(&r, "x"), int(&r, "y"), int(&r, "zoom"));
                let b = tiles::tile_bbox(x, y, z);
                let out = arr(&r, "out");
                let ctx = format!("{x} {y} {z}");
                t.f64("tile_bbox", &ctx, b.0, vd(&out[0]));
                t.f64("tile_bbox", &ctx, b.1, vd(&out[1]));
                t.f64("tile_bbox", &ctx, b.2, vd(&out[2]));
                t.f64("tile_bbox", &ctx, b.3, vd(&out[3]));
            }
            "tiles_covering" => {
                let got = tiles::tiles_covering(d(&r, "s"), d(&r, "w"), d(&r, "n"), d(&r, "e"), int(&r, "zoom"));
                check_tiles(&mut t, "tiles_covering", &format!("{:?}", r.get("s")), &got, get(&r, "out"));
            }
            "preset" => {
                let want = src_of(get(&r, "src"));
                let got = tiles::preset(&want.id).expect("preset");
                let id = want.id.clone();
                t.eq("preset", &id, got, want);
            }
            "url" => {
                let src = src_of(get(&r, "src"));
                let (x, y, z) = (int(&r, "x"), int(&r, "y"), int(&r, "z"));
                t.eq("url", &format!("{x} {y} {z}"), src.url(x, y, z), s(&r, "out").to_string());
            }
            "slug" => t.eq("source_slug", s(&r, "name"), tiles::source_slug(s(&r, "name")), s(&r, "out").to_string()),
            "custom_source" => {
                let got = tiles::custom_source("https://e/{z}/{x}/{y}", int(&r, "max_zoom"), opt_s(&r, "name"));
                t.eq("custom_source", &format!("{:?}", r.get("name")), got, src_of(get(&r, "out")));
            }
            "layer" => {
                let dat = datum_of(get(&r, "datum"));
                let layer = layer_of(get(&r, "layer"));
                let ctx = format!("{} z{}", layer.source.id, layer.zoom);
                t.eq("layer.to_dict", &ctx, dumps_default(&layer.to_dict()), s(&r, "to_dict").to_string());
                check_tiles(&mut t, "layer.flat_tiles", &ctx, &layer.flat_tiles(&dat), get(&r, "flat_tiles"));
                check_tiles(&mut t, "layer.visible_tiles", &ctx, &layer.visible_tiles(&dat), get(&r, "visible_tiles"));
                for q in arr(&r, "quads") {
                    let got = layer.quad_local(&dat, int(q, "x"), int(q, "y"));
                    let want = arr(q, "out");
                    t.eq("layer.quad.len", &ctx, got.len(), want.len());
                    for (g, w) in got.iter().zip(want) {
                        let w = items(w);
                        t.v3("layer.quad.pos", &ctx, g.0, v3(&w[0]));
                        t.f64("layer.quad.u", &ctx, (g.1).0, vd(&items(&w[1])[0]));
                        t.f64("layer.quad.v", &ctx, (g.1).1, vd(&items(&w[1])[1]));
                    }
                }
                for c in arr(&r, "caps") {
                    let mut lc = layer.clone();
                    let reduced = lc.cap_detail(&dat, int(c, "max_tiles") as usize, int(c, "min_zoom"));
                    t.eq("layer.cap.reduced", &ctx, reduced, boolean(c, "reduced"));
                    t.eq("layer.cap.zoom", &ctx, lc.zoom, int(c, "zoom"));
                }
            }
            "layer_from_dict" => {
                let input = loads(s(&r, "data")).unwrap();
                let ctx = s(&r, "data").to_string();
                match TileLayer::from_dict(&input) {
                    Ok(layer) if !has(&r, "nm") => {
                        if has(&r, "error") {
                            t.fail("layer_from_dict", format!("{ctx}: Python raised"));
                            continue;
                        }
                        t.eq(
                            "layer_from_dict.layer",
                            &ctx,
                            dumps_default(&layer.to_dict()),
                            dumps_default(&layer_of(get(&r, "out")).to_dict()),
                        );
                        t.eq("layer_from_dict.source", &ctx, layer.source.clone(), layer_of(get(&r, "out")).source);
                        t.eq("layer_from_dict.to_dict", &ctx, dumps_default(&layer.to_dict()), s(&r, "to_dict").to_string());
                    }
                    Ok(_) => t.fail("layer_from_dict", format!("{ctx}: expected NotModelled")),
                    Err(e) => expect_err(&mut t, "layer_from_dict", &ctx, &r, Err(e)),
                }
            }
            "cache_path" => {
                let c = TileCache::new(scratch("path"), TileCache::DEFAULT_MAX_BYTES);
                let rel = c
                    .path_for(s(&r, "sid"), int(&r, "x"), int(&r, "y"), int(&r, "z"))
                    .strip_prefix(&c.root)
                    .unwrap()
                    .to_string_lossy()
                    .replace('\\', "/");
                t.eq("cache_path", s(&r, "sid"), rel, s(&r, "out").to_string());
                let _ = std::fs::remove_dir_all(&c.root);
            }
            "cache_run" => {
                let root = scratch("run");
                let c = TileCache::new(&root, int(&r, "budget") as u64);
                let mut got_results: Vec<Option<i64>> = Vec::new();
                for op in arr(&r, "ops") {
                    let o = items(op);
                    let (sid, x, y, z) = (
                        match &o[1] {
                            Value::Str(x) => x.clone(),
                            o => panic!("{o:?}"),
                        },
                        o[2].as_i64().unwrap(),
                        o[3].as_i64().unwrap(),
                        o[4].as_i64().unwrap(),
                    );
                    match &o[0] {
                        Value::Str(k) if k == "put" => {
                            c.put(&sid, x, y, z, &vec![b'x'; o[5].as_i64().unwrap() as usize]);
                            got_results.push(None);
                        }
                        _ => got_results.push(c.get(&sid, x, y, z).map(|b| b.len() as i64)),
                    }
                }
                let want: Vec<Option<i64>> = arr(&r, "results").iter().map(|v| v.as_i64()).collect();
                t.eq("cache_run.results", &format!("{:?}", r.get("ops")), got_results, want);
                let want_files: Vec<String> = arr(&r, "survivors")
                    .iter()
                    .map(|v| match v {
                        Value::Str(x) => x.clone(),
                        o => panic!("{o:?}"),
                    })
                    .collect();
                t.eq("cache_run.survivors", &format!("{:?}", r.get("ops")), rel_files(&root), want_files);
                let _ = std::fs::remove_dir_all(&root);
            }
            "evict" => {
                let files: Vec<CacheFile> = arr(&r, "files")
                    .iter()
                    .map(|f| {
                        let f = items(f);
                        let mt = match &f[1] {
                            Value::Int(x) => x.parse::<f64>().unwrap(),
                            Value::Float(x) => *x,
                            o => panic!("{o:?}"),
                        };
                        CacheFile {
                            path: match &f[0] {
                                Value::Str(x) => PathBuf::from(x),
                                o => panic!("{o:?}"),
                            },
                            mtime: mt,
                            size: f[2].as_i64().unwrap() as u64,
                        }
                    })
                    .collect();
                let plan = tiles::eviction_plan(&files, int(&r, "budget") as u64);
                let mut surv: Vec<String> = files
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| !plan.contains(i))
                    .map(|(_, f)| f.path.to_string_lossy().replace('\\', "/"))
                    .collect();
                surv.sort();
                let want: Vec<String> = arr(&r, "survivors")
                    .iter()
                    .map(|v| match v {
                        Value::Str(x) => x.clone(),
                        o => panic!("{o:?}"),
                    })
                    .collect();
                t.eq("evict", &format!("{:?}", r.get("files")), surv, want);
            }
            o => panic!("fn {o}"),
        }
    }
    t.finish();
}

// ---------------------------------------------------------------------------
// tile_fetcher
// ---------------------------------------------------------------------------

fn key_of(v: &Value) -> TileKey {
    let a = items(v);
    (
        match &a[0] {
            Value::Str(x) => x.clone(),
            o => panic!("{o:?}"),
        },
        a[1].as_i64().unwrap(),
        a[2].as_i64().unwrap(),
        a[3].as_i64().unwrap(),
    )
}

#[test]
fn tile_fetcher_matches_python() {
    let mut t = Tally::default();
    let mut sources: HashMap<String, TileSource> = HashMap::new();
    for p in tiles::presets() {
        sources.insert(p.id.clone(), p);
    }
    for r in records("tile_fetcher.jsonl") {
        match s(&r, "fn") {
            "scenario" => {
                let mut cache: HashMap<TileKey, bool> = HashMap::new();
                for i in arr(&r, "init") {
                    let a = items(i);
                    let good = matches!(&a[4], Value::Str(k) if k == "good");
                    cache.insert(key_of(i), good);
                }
                let mut planner = TilePlanner::new(tile_fetcher::user_agent(tile_fetcher::APP_VERSION));
                for (n, op) in arr(&r, "ops").iter().enumerate() {
                    let ctx = format!("op {n}: {:?}", op.get("op"));
                    let started_want: Vec<(TileKey, String)> = op
                        .get("started")
                        .map(items)
                        .unwrap_or(&[])
                        .iter()
                        .map(|st| {
                            let a = items(st);
                            let url = match &a[4] {
                                Value::Str(x) => x.clone(),
                                o => panic!("{o:?}"),
                            };
                            (key_of(st), url)
                        })
                        .collect();
                    match s(op, "op") {
                        "request" => {
                            let src = sources[s(op, "src")].clone();
                            let (x, y, z) = (int(op, "x"), int(op, "y"), int(op, "z"));
                            let key = (src.id.clone(), x, y, z);
                            let q0 = planner.queue_len() as i64;
                            let outcome = planner.request(&src, x, y, z, || cache.get(&key).copied().unwrap_or(false));
                            let mut started: Vec<(TileKey, String)> = Vec::new();
                            let is_hit = matches!(outcome, RequestOutcome::CacheHit);
                            t.eq("fetch.image", &ctx, is_hit, boolean(op, "image"));
                            if let RequestOutcome::Start(req) = outcome {
                                assert_eq!(req.user_agent, tile_fetcher::user_agent(tile_fetcher::APP_VERSION));
                                started.push((req.key, req.url));
                            }
                            t.eq("fetch.started", &ctx, started, started_want);
                            t.eq("fetch.queued", &ctx, planner.queue_len() as i64 - q0, int(op, "queued"));
                        }
                        "finish" => {
                            let key = key_of(get(op, "key"));
                            let starts = planner.finished(&key);
                            let started: Vec<(TileKey, String)> = starts.into_iter().map(|q| (q.key, q.url)).collect();
                            t.eq("fetch.finish.started", &ctx, started, started_want);
                            let (err, data, decodes): (Option<&str>, &[u8], bool) = match s(op, "kind") {
                                "ok" => (None, b"png", true),
                                "garbage" => (None, b"junk", false),
                                "empty" => (None, b"", false),
                                _ => (Some("Host not found"), b"", false),
                            };
                            let want_failed: Vec<String> = arr(op, "failed")
                                .iter()
                                .map(|v| match v {
                                    Value::Str(x) => x.clone(),
                                    o => panic!("{o:?}"),
                                })
                                .collect();
                            match tile_fetcher::classify_reply(err, data, decodes) {
                                tile_fetcher::TileOutcome::Ready => {
                                    cache.insert(key.clone(), true);
                                    t.eq("fetch.finish.ready", &ctx, 1, int(op, "ready"));
                                    t.eq("fetch.finish.failed", &ctx, Vec::<String>::new(), want_failed);
                                }
                                tile_fetcher::TileOutcome::Failed(reason) => {
                                    t.eq("fetch.finish.ready", &ctx, 0, int(op, "ready"));
                                    t.eq("fetch.finish.failed", &ctx, vec![reason], want_failed);
                                }
                            }
                        }
                        "cancel" => {
                            planner.cancel_all();
                        }
                        o => panic!("op {o}"),
                    }
                    let want_in: Vec<TileKey> = arr(op, "inflight").iter().map(key_of).collect();
                    let want_q: Vec<TileKey> = arr(op, "queue").iter().map(key_of).collect();
                    t.eq("fetch.inflight", &ctx, planner.inflight_keys(), want_in);
                    t.eq("fetch.queue", &ctx, planner.queue_keys(), want_q);
                }
            }
            "max_zoom" => t.eq("max_zoom", s(&r, "src"), sources[s(&r, "src")].max_zoom, int(&r, "max_zoom")),
            "user_agent" => t.eq(
                "user_agent",
                "",
                tile_fetcher::user_agent(tile_fetcher::APP_VERSION),
                s(&r, "out").to_string(),
            ),
            o => panic!("fn {o}"),
        }
    }
    t.finish();
}

// ---------------------------------------------------------------------------
// terrain
// ---------------------------------------------------------------------------

fn hash_terrain(tr: &TerrainObject) -> String {
    let mut b = Vec::new();
    for v in &tr.vertices {
        b.extend_from_slice(&v.0.to_be_bytes());
        b.extend_from_slice(&v.1.to_be_bytes());
        b.extend_from_slice(&v.2.to_be_bytes());
    }
    for (u, v) in &tr.uvs {
        b.extend_from_slice(&u.to_be_bytes());
        b.extend_from_slice(&v.to_be_bytes());
    }
    for &(a, c, e) in &tr.triangles {
        b.extend_from_slice(&(a as u32).to_be_bytes());
        b.extend_from_slice(&(c as u32).to_be_bytes());
        b.extend_from_slice(&(e as u32).to_be_bytes());
    }
    fnv64(&b)
}

#[test]
fn terrain_matches_python() {
    let mut t = Tally::default();
    for r in records("terrain.jsonl") {
        let dat = datum_of(get(&r, "datum"));
        let bb = arr(&r, "bbox");
        let bbox = (vd(&bb[0]), vd(&bb[1]), vd(&bb[2]), vd(&bb[3]));
        let sampler = sampler_of(dat.clone(), int(&r, "sampler_zoom"), arr(&r, "tiles"));
        let ll = terrain::bbox_ll(&dat, bbox);
        let want_ll = arr(&r, "ll");
        let ctx = format!("{bbox:?} {}", int(&r, "zoom"));
        t.f64("bbox_ll", &ctx, ll.0, vd(&want_ll[0]));
        t.f64("bbox_ll", &ctx, ll.1, vd(&want_ll[1]));
        t.f64("bbox_ll", &ctx, ll.2, vd(&want_ll[2]));
        t.f64("bbox_ll", &ctx, ll.3, vd(&want_ll[3]));
        let got = terrain::build_terrain(&dat, &sampler, d(&r, "ground"), bbox, d(&r, "cell"), int(&r, "max_grid"), int(&r, "zoom"));
        let out = get(&r, "out");
        let Some(tr) = got else {
            t.eq("terrain.none", &ctx, true, matches!(out, Value::Null));
            continue;
        };
        if matches!(out, Value::Null) {
            t.fail("terrain.none", format!("{ctx}: Python returned None"));
            continue;
        }
        t.eq("terrain.nx", &ctx, tr.nx, int(out, "nx"));
        t.eq("terrain.ny", &ctx, tr.ny, int(out, "ny"));
        let tr_range: Vec<i64> = items(get(out, "tile_range")).iter().map(|v| v.as_i64().unwrap()).collect();
        t.eq("terrain.tile_range", &ctx, vec![tr.tile_range.0, tr.tile_range.1, tr.tile_range.2, tr.tile_range.3, tr.tile_range.4], tr_range);
        t.eq("terrain.nverts", &ctx, tr.vertices.len() as i64, int(out, "nverts"));
        t.eq("terrain.ntris", &ctx, tr.triangles.len() as i64, int(out, "ntris"));
        t.eq("terrain.hash", &ctx, hash_terrain(&tr), s(out, "hash").to_string());
        if has(out, "vertices") {
            for (g, w) in tr.vertices.iter().zip(arr(out, "vertices")) {
                t.v3("terrain.vertex", &ctx, *g, v3(w));
            }
            for (g, w) in tr.uvs.iter().zip(arr(out, "uvs")) {
                t.f64("terrain.uv", &ctx, g.0, vd(&items(w)[0]));
                t.f64("terrain.uv", &ctx, g.1, vd(&items(w)[1]));
            }
            let want_tris: Vec<(usize, usize, usize)> = arr(out, "triangles")
                .iter()
                .map(|x| {
                    let a = items(x);
                    (a[0].as_i64().unwrap() as usize, a[1].as_i64().unwrap() as usize, a[2].as_i64().unwrap() as usize)
                })
                .collect();
            t.eq("terrain.triangles", &ctx, tr.triangles.clone(), want_tris);
        }
        for q in arr(&r, "height_at") {
            let q = items(q);
            let (x, y) = (vd(&q[0]), vd(&q[1]));
            match (tr.height_at(x, y), od(&q[2])) {
                (Some(g), Some(w)) => t.f64("terrain.height_at", &ctx, g, w),
                (None, None) => t.ok("terrain.height_at"),
                (g, w) => t.fail("terrain.height_at", format!("{ctx} {x} {y}: got {g:?} want {w:?}")),
            }
        }
        let (lo, hi) = tr.bounds().expect("bounds");
        let bounds = arr(&r, "bounds");
        t.v3("terrain.bounds", &ctx, lo, v3(&bounds[0]));
        t.v3("terrain.bounds", &ctx, hi, v3(&bounds[1]));
        let mo = get(&r, "mosaic");
        let keys: Vec<(i64, i64, i64)> = arr(mo, "keys")
            .iter()
            .map(|k| {
                let a = items(k);
                (a[0].as_i64().unwrap(), a[1].as_i64().unwrap(), a[2].as_i64().unwrap())
            })
            .collect();
        let plan = terrain::mosaic_plan(&tr, &keys);
        let want = get(mo, "out");
        match (plan, want) {
            (None, Value::Null) => t.ok("mosaic.none"),
            (Some(p), w @ Value::Object(_)) => {
                t.eq("mosaic.w", &ctx, p.width, int(w, "w"));
                t.eq("mosaic.h", &ctx, p.height, int(w, "h"));
                let mut got: Vec<(i64, i64, i64)> = p.tiles.iter().map(|x| (x.0, x.1, x.2)).collect();
                let mut want_p: Vec<(i64, i64, i64)> = arr(w, "placed")
                    .iter()
                    .map(|k| {
                        let a = items(k);
                        (a[0].as_i64().unwrap(), a[1].as_i64().unwrap(), a[2].as_i64().unwrap())
                    })
                    .collect();
                got.sort();
                want_p.sort();
                t.eq("mosaic.placed", &ctx, got, want_p);
                for x in &p.tiles {
                    t.eq("mosaic.dest", &ctx, (x.3, x.4), ((x.0 - tr.tile_range.0) * 256, (x.1 - tr.tile_range.1) * 256));
                }
            }
            (g, w) => t.fail("mosaic", format!("{ctx}: got {g:?} want {w:?}")),
        }
    }
    t.finish();
}

// ---------------------------------------------------------------------------
// surface + geopath
// ---------------------------------------------------------------------------

fn hash_tris(tris: &[[V3; 3]]) -> String {
    let mut b = Vec::new();
    for t in tris {
        for v in t {
            b.extend_from_slice(&v.0.to_be_bytes());
            b.extend_from_slice(&v.1.to_be_bytes());
            b.extend_from_slice(&v.2.to_be_bytes());
        }
    }
    fnv64(&b)
}

fn check_tris(t: &mut Tally, f: &str, ctx: &str, got: Option<Vec<[V3; 3]>>, want: &Value) {
    match (got, want) {
        (None, Value::Null) => t.ok(f),
        (Some(g), w @ Value::Object(_)) => {
            t.eq(&format!("{f}.n"), ctx, g.len() as i64, int(w, "n"));
            t.eq(&format!("{f}.hash"), ctx, hash_tris(&g), s(w, "hash").to_string());
            if has(w, "tris") {
                for (a, b) in g.iter().zip(arr(w, "tris")) {
                    for (v, wv) in a.iter().zip(items(b)) {
                        t.v3(&format!("{f}.tri"), ctx, *v, v3(wv));
                    }
                }
            }
        }
        (g, w) => t.fail(f, format!("{ctx}: got {} want {}", g.is_some(), !matches!(w, Value::Null))),
    }
}

#[test]
fn surface_matches_python() {
    let mut t = Tally::default();
    for r in records("surface.jsonl") {
        match s(&r, "fn") {
            "fit_plane" => {
                let pts = v3s(get(&r, "pts"));
                let zs: Vec<f64> = arr(&r, "zs").iter().map(vd).collect();
                let (a, b, c) = surface::fit_plane(&pts, &zs);
                let out = arr(&r, "out");
                let ctx = format!("{} pts", pts.len());
                t.f64("fit_plane", &ctx, a, vd(&out[0]));
                t.f64("fit_plane", &ctx, b, vd(&out[1]));
                t.f64("fit_plane", &ctx, c, vd(&out[2]));
            }
            "subdivide" => {
                let tri: Vec<(f64, f64)> = arr(&r, "tri").iter().map(|p| (vd(&items(p)[0]), vd(&items(p)[1]))).collect();
                let mut out = Vec::new();
                surface::subdivide([tri[0], tri[1], tri[2]], int(&r, "depth") as i32, &mut out);
                let want = arr(&r, "out");
                t.eq("subdivide.n", "", out.len(), want.len());
                for (g, w) in out.iter().zip(want) {
                    for (p, wp) in g.iter().zip(items(w)) {
                        t.f64("subdivide", "", p.0, vd(&items(wp)[0]));
                        t.f64("subdivide", "", p.1, vd(&items(wp)[1]));
                    }
                }
            }
            "surface" => {
                let dat = datum_of(get(&r, "datum"));
                let sampler = sampler_of(dat.clone(), int(&r, "sampler_zoom"), arr(&r, "tiles"));
                let pts: Vec<V3> = arr(&r, "poly").iter().map(|p| V3(hf(match &items(p)[0] { Value::Str(x) => x, o => panic!("{o:?}") }), hf(match &items(p)[1] { Value::Str(x) => x, o => panic!("{o:?}") }), 0.0)).collect();
                let mode = opt_s(&r, "mode");
                let path = GeoPath::new(&pts, true, "", mode);
                let ctx = format!("{} pts {:?}", pts.len(), mode);
                let ground = surface::ground_reference(&sampler, &dat);
                t.eq("ground_reference", &ctx, ground.map(f64::to_bits), od(get(&r, "ground")).map(f64::to_bits));
                check_tris(&mut t, "build_surface", &ctx, surface::build_surface(&path, &sampler, &dat), get(&r, "build_surface"));
                if let Some(g) = ground {
                    check_tris(&mut t, "flat", &ctx, surface::build_flat_surface(&path, &sampler, g), get(&r, "flat"));
                    check_tris(&mut t, "draped", &ctx, surface::build_draped_surface(&path, &sampler, g), get(&r, "draped"));
                }
            }
            o => panic!("fn {o}"),
        }
    }
    t.finish();
}

#[test]
fn geopath_matches_python() {
    let mut t = Tally::default();
    for r in records("geopath.jsonl") {
        match s(&r, "fn") {
            "path" => {
                let pts = v3s(get(&r, "pts"));
                let name = s(&r, "name");
                let mut gp = GeoPath::new(&pts, boolean(&r, "closed"), name, opt_s(&r, "surface"));
                let ctx = format!("{} pts", pts.len());
                let segs = gp.segments();
                let want = arr(&r, "segments");
                t.eq("path.segments.n", &ctx, segs.len(), want.len());
                for ((a, b), w) in segs.iter().zip(want) {
                    t.v3("path.segments", &ctx, *a, v3(&items(w)[0]));
                    t.v3("path.segments", &ctx, *b, v3(&items(w)[1]));
                }
                t.f64("path.length", &ctx, gp.length(), d(&r, "length"));
                t.f64("path.perimeter", &ctx, gp.perimeter(), d(&r, "perimeter"));
                t.f64("path.area", &ctx, gp.area(), d(&r, "area"));
                let pp = gp.profile_points();
                let wpp = arr(&r, "profile_points");
                t.eq("path.profile_points.n", &ctx, pp.len(), wpp.len());
                for (g, w) in pp.iter().zip(wpp) {
                    t.v3("path.profile_points", &ctx, *g, v3(w));
                }
                t.eq("path.to_dict", &ctx, dumps_default(&gp.to_dict()), s(&r, "to_dict").to_string());
                gp.surface_tris = match get(&r, "tris") {
                    Value::Null => None,
                    tv => Some(items(tv).iter().map(|tri| {
                        let a = items(tri);
                        [v3(&a[0]), v3(&a[1]), v3(&a[2])]
                    }).collect()),
                };
                let sa = gp.surface_area(Hypot::Msvc);
                t.eq("path.surface_area", &ctx, sa.map(f64::to_bits), od(get(&r, "surface_area")).map(f64::to_bits));
                let rt = GeoPath::from_dict(&loads(s(&r, "to_dict")).unwrap()).unwrap();
                t.eq("path.roundtrip", &ctx, dumps_default(&rt.to_dict()), s(&r, "roundtrip").to_string());
            }
            "path_from_dict" => {
                let input = loads(s(&r, "data")).unwrap();
                let ctx = s(&r, "data").to_string();
                match GeoPath::from_dict(&input) {
                    Ok(gp) if !has(&r, "nm") && !has(&r, "error") => {
                        t.eq("path_from_dict.to_dict", &ctx, dumps_default(&gp.to_dict()), s(&r, "to_dict").to_string());
                        t.eq("path_from_dict.closed", &ctx, gp.closed, boolean(&r, "closed"));
                        let want = arr(&r, "points");
                        t.eq("path_from_dict.n", &ctx, gp.points.len(), want.len());
                        for (g, w) in gp.points.iter().zip(want) {
                            t.v3("path_from_dict.points", &ctx, *g, v3(w));
                        }
                    }
                    Ok(_) => t.fail("path_from_dict", format!("{ctx}: Ok but Python raised or NotModelled expected")),
                    Err(e) => expect_err(&mut t, "path_from_dict", &ctx, &r, Err(e)),
                }
            }
            o => panic!("fn {o}"),
        }
    }
    t.finish();
}

// ---------------------------------------------------------------------------
// points
// ---------------------------------------------------------------------------

fn row_of(v: &Value) -> PointRow {
    PointRow {
        name: s(v, "name").to_string(),
        north: d(v, "north"),
        east: d(v, "east"),
        z: d(v, "z"),
        desc: s(v, "desc").to_string(),
    }
}

#[test]
fn points_match_python() {
    let mut t = Tally::default();
    for r in records("points.jsonl") {
        match s(&r, "fn") {
            "csv" => {
                let text = s(&r, "text");
                let got = points::parse_points_csv(text);
                match (got, r.get("rows")) {
                    (Ok(rows), Some(w)) => {
                        let w = items(w);
                        t.eq("csv.n", text, rows.len(), w.len());
                        for (g, wr) in rows.iter().zip(w) {
                            let want = row_of(wr);
                            t.eq("csv.name", text, g.name.clone(), want.name);
                            t.eq("csv.desc", text, g.desc.clone(), want.desc);
                            t.f64("csv.north", text, g.north, want.north);
                            t.f64("csv.east", text, g.east, want.east);
                            t.f64("csv.z", text, g.z, want.z);
                        }
                    }
                    (Err(_), None) => t.ok("csv.error"),
                    (g, w) => t.fail("csv", format!("{text:?}: got {g:?} want {w:?}")),
                }
            }
            "points_from_rows" => {
                let dat = datum_of(get(&r, "datum"));
                let rows: Vec<PointRow> = arr(&r, "rows").iter().map(row_of).collect();
                let got = points::points_from_rows(&rows, &dat);
                for (g, w) in got.iter().zip(arr(&r, "out")) {
                    t.v3("points_from_rows", "", g.position, v3(get(w, "p")));
                    t.eq("points_from_rows.name", "", g.name.clone(), s(w, "name").to_string());
                    t.eq("points_from_rows.desc", "", g.desc.clone(), s(w, "desc").to_string());
                    t.eq("points_from_rows.to_dict", "", dumps_default(&g.to_dict()), s(w, "to_dict").to_string());
                }
            }
            "datum_for_rows" => {
                let row = get(&r, "row");
                let rows = [PointRow { name: "A".into(), north: d(row, "north"), east: d(row, "east"), z: d(row, "z"), desc: String::new() }];
                let dat = points::datum_for_rows(&rows, int(&r, "zone"), boolean(&r, "northern"));
                let o = get(&r, "out");
                let ctx = format!("zone {}", int(&r, "zone"));
                t.f64("datum_for_rows.lat", &ctx, dat.lat, d(o, "lat"));
                t.f64("datum_for_rows.lon", &ctx, dat.lon, d(o, "lon"));
                t.f64("datum_for_rows.alt", &ctx, dat.alt, d(o, "alt"));
                t.f64("datum_for_rows.north", &ctx, dat.north, d(o, "north"));
                t.eq("datum_for_rows.zone", &ctx, dat.zone, int(o, "zone"));
                t.eq("datum_for_rows.northern", &ctx, dat.northern, boolean(o, "northern"));
                let (e0, n0) = dat.anchor_utm();
                t.f64("datum_for_rows.east0", &ctx, e0, d(o, "east0"));
                t.f64("datum_for_rows.north0", &ctx, n0, d(o, "north0"));
            }
            "point_from_dict" => {
                let input = loads(s(&r, "data")).unwrap();
                let ctx = s(&r, "data").to_string();
                match GeoPoint::from_dict(&input) {
                    Ok(p) if !has(&r, "nm") && !has(&r, "error") => {
                        t.eq("point_from_dict.to_dict", &ctx, dumps_default(&p.to_dict()), s(&r, "to_dict").to_string());
                        t.v3("point_from_dict.p", &ctx, p.position, v3(get(&r, "p")));
                    }
                    Ok(_) => t.fail("point_from_dict", format!("{ctx}: Ok but Python raised or NotModelled expected")),
                    Err(e) => expect_err(&mut t, "point_from_dict", &ctx, &r, Err(e)),
                }
            }
            "point" => {
                let p = GeoPoint::new(v3(get(&r, "p")), s(&r, "name"), s(&r, "desc"));
                t.eq("point.to_dict", "", dumps_default(&p.to_dict()), s(&r, "to_dict").to_string());
            }
            o => panic!("fn {o}"),
        }
    }
    t.finish();
}

// ---------------------------------------------------------------------------
// geocode
// ---------------------------------------------------------------------------

#[test]
fn geocode_matches_python() {
    let mut t = Tally::default();
    for r in records("geocode.jsonl") {
        match s(&r, "fn") {
            "url" => t.eq("geocode_url", s(&r, "q"), geocode::geocode_url(s(&r, "q"), int(&r, "limit")), s(&r, "out").to_string()),
            "url_default" => t.eq("geocode_url.default", s(&r, "q"), geocode::geocode_url(s(&r, "q"), 8), s(&r, "out").to_string()),
            "plan_search" => {
                let q = s(&r, "q");
                let calls: Vec<String> = arr(&r, "calls")
                    .iter()
                    .map(|v| match v {
                        Value::Str(x) => x.clone(),
                        o => panic!("{o:?}"),
                    })
                    .collect();
                match geocode::plan_search(q) {
                    geocode::SearchPlan::Empty => {
                        t.eq("plan_search.empty", q, (calls.len() as i64, int(&r, "empty")), (0, 1));
                    }
                    geocode::SearchPlan::Get(url) => t.eq("plan_search.get", q, (vec![url], int(&r, "empty")), (calls, 0)),
                }
            }
            "places" => {
                let data = unhex(s(&r, "data"));
                let ctx = s(&r, "data").to_string();
                match (geocode::parse_places(&data), r.get("out"), r.get("error")) {
                    (Ok(got), Some(w), None) => {
                        let w = items(w);
                        t.eq("places.n", &ctx, got.len(), w.len());
                        for (g, wp) in got.iter().zip(w) {
                            let Place { name, lat, lon } = g;
                            t.eq("places.name", &ctx, name.clone(), s(wp, "name").to_string());
                            t.f64("places.lat", &ctx, *lat, d(wp, "lat"));
                            t.f64("places.lon", &ctx, *lon, d(wp, "lon"));
                        }
                    }
                    (Err(e), None, Some(Value::Str(k))) => t.eq("places.error", &ctx, e.kind, k.as_str()),
                    (g, w, e) => t.fail("places", format!("{ctx}: got {g:?} want {w:?} / {e:?}")),
                }
            }
            "ip" => {
                let data = unhex(s(&r, "data"));
                let ctx = s(&r, "data").to_string();
                match (geocode::parse_ip(&data), r.get("out"), r.get("error")) {
                    (Ok(None), Some(Value::Null), None) => t.ok("ip.none"),
                    (Ok(Some((la, lo, label))), Some(Value::Array(w)), None) => {
                        t.f64("ip.lat", &ctx, la, vd(&w[0]));
                        t.f64("ip.lon", &ctx, lo, vd(&w[1]));
                        t.eq("ip.label", &ctx, label, match &w[2] { Value::Str(x) => x.clone(), o => panic!("{o:?}") });
                    }
                    (Err(e), None, Some(Value::Str(k))) => t.eq("ip.error", &ctx, e.kind, k.as_str()),
                    (g, w, e) => t.fail("ip", format!("{ctx}: got {g:?} want {w:?} / {e:?}")),
                }
            }
            "consts" => {
                t.eq("consts.nominatim", "", geocode::NOMINATIM, s(&r, "nominatim"));
                t.eq("consts.ipapi", "", geocode::IPAPI, s(&r, "ipapi"));
                t.eq("consts.ua", "", tile_fetcher::user_agent(tile_fetcher::APP_VERSION), s(&r, "ua").to_string());
            }
            o => panic!("fn {o}"),
        }
    }
    t.finish();
}

// ---------------------------------------------------------------------------
// profile
// ---------------------------------------------------------------------------

fn compare_profile(t: &mut Tally, ctx: &str, p: &Profile, want: &Value) {
    let ws = arr(want, "samples");
    t.eq("profile.samples.n", ctx, p.samples.len(), ws.len());
    t.f64("profile.length", ctx, p.length, d(want, "length"));
    t.eq("profile.complete", ctx, p.complete, boolean(want, "complete"));
    for (g, w) in p.samples.iter().zip(ws) {
        let w = items(w);
        t.f64("profile.station", ctx, g.station, vd(&w[0]));
        t.f64("profile.x", ctx, g.x, vd(&w[1]));
        t.f64("profile.y", ctx, g.y, vd(&w[2]));
        match (g.elevation, od(&w[3])) {
            (Some(a), Some(b)) => t.f64("profile.elevation", ctx, a, b),
            (None, None) => t.ok("profile.elevation"),
            (a, b) => t.fail("profile.elevation", format!("{ctx}: {a:?} vs {b:?}")),
        }
    }
    t.eq("profile.min", ctx, p.min_elevation().map(f64::to_bits), od(get(want, "min")).map(f64::to_bits));
    t.eq("profile.max", ctx, p.max_elevation().map(f64::to_bits), od(get(want, "max")).map(f64::to_bits));
    t.f64("profile.gain", ctx, p.total_gain(), d(want, "gain"));
    let sl = p.slopes();
    let wsl = arr(want, "slopes");
    t.eq("profile.slopes.n", ctx, sl.len(), wsl.len());
    for (g, w) in sl.iter().zip(wsl) {
        t.f64("profile.slope.station", ctx, g.0, vd(&items(w)[0]));
        t.f64("profile.slope.pct", ctx, g.1, vd(&items(w)[1]));
    }
    let el = p.elevations();
    let wel = arr(want, "elevations");
    t.eq("profile.elevations.n", ctx, el.len(), wel.len());
    for (g, w) in el.iter().zip(wel) {
        t.f64("profile.elevations", ctx, *g, vd(w));
    }
    if let Some(Value::Array(csvs)) = want.get("csv") {
        for c in csvs {
            let got = profile::profile_to_csv(p, s(c, "vertical"), od(get(c, "alt")));
            t.eq("profile.csv", &format!("{ctx} {} {:?}", s(c, "vertical"), c.get("alt")), got, s(c, "out").to_string());
        }
    }
}

#[test]
fn profile_matches_python() {
    let mut t = Tally::default();
    for r in records("profile.jsonl") {
        match s(&r, "fn") {
            "helpers" => {
                let pts = v3s(get(&r, "pts"));
                t.f64("polyline_length", "", profile::polyline_length(&pts), d(&r, "length"));
                for q in arr(&r, "at") {
                    let q = items(q);
                    let (x, y) = profile::point_at_station(&pts, vd(&q[0]));
                    t.f64("point_at_station.x", "", x, vd(&q[1]));
                    t.f64("point_at_station.y", "", y, vd(&q[2]));
                }
            }
            "sample" => {
                let dat = datum_of(get(&r, "datum"));
                let pts = v3s(get(&r, "pts"));
                let sampler = sampler_of(dat, int(&r, "sampler_zoom"), arr(&r, "tiles"));
                let spacing = od(get(&r, "spacing"));
                let ctx = format!("{} pts spacing {:?}", pts.len(), spacing);
                match (profile::sample_profile(&pts, &sampler, spacing), r.get("out"), r.get("error")) {
                    (Ok(p), Some(w), None) => compare_profile(&mut t, &ctx, &p, w),
                    (Err(_), None, Some(Value::Str(k))) => t.eq("sample.error", &ctx, k.as_str(), "ZeroDivisionError"),
                    (g, w, e) => t.fail("sample", format!("{ctx}: got {g:?} want {w:?} / {e:?}")),
                }
            }
            "manual" => {
                let samples: Vec<ProfileSample> = arr(&r, "samples")
                    .iter()
                    .map(|w| {
                        let w = items(w);
                        ProfileSample { station: vd(&w[0]), x: vd(&w[1]), y: vd(&w[2]), elevation: od(&w[3]) }
                    })
                    .collect();
                let p = Profile { samples, length: d(&r, "length"), complete: boolean(&r, "complete") };
                compare_profile(&mut t, "manual", &p, &r);
            }
            "format" => {
                let vals: Vec<f64> = arr(&r, "vals").iter().map(vd).collect();
                let samples: Vec<ProfileSample> =
                    vals.iter().map(|&v| ProfileSample { station: v, x: v, y: v, elevation: Some(v) }).collect();
                let p = Profile { samples, length: 0.0, complete: true };
                let got = profile::profile_to_csv(&p, "local", None);
                let want = s(&r, "out");
                // Compare line by line so a failure names the value.
                for (i, (g, w)) in got.lines().zip(want.lines()).enumerate() {
                    let v = if i >= 2 { vals.get(i - 2).copied() } else { None };
                    t.eq("format", &format!("line {i}: {v:?}"), g.to_string(), w.to_string());
                }
                t.eq("format.lines", "", got.lines().count(), want.lines().count());
            }
            "note" => t.eq("vertical_note", s(&r, "vertical"), profile::vertical_note(s(&r, "vertical")), s(&r, "out")),
            o => panic!("fn {o}"),
        }
    }
    t.finish();
}

// ---------------------------------------------------------------------------
// geoimport
// ---------------------------------------------------------------------------

fn compare_feats(t: &mut Tally, f: &str, ctx: &str, got: &[GeoFeature], want: &Value) {
    let w = items(want);
    t.eq(&format!("{f}.n"), ctx, got.len(), w.len());
    for (g, wf) in got.iter().zip(w) {
        t.eq(&format!("{f}.closed"), ctx, g.closed, boolean(wf, "closed"));
        t.eq(&format!("{f}.name"), ctx, g.name.clone(), s(wf, "name").to_string());
        let wp = arr(wf, "points");
        t.eq(&format!("{f}.points.n"), ctx, g.points.len(), wp.len());
        for (p, w) in g.points.iter().zip(wp) {
            t.f64(&format!("{f}.lat"), ctx, p.0, vd(&items(w)[0]));
            t.f64(&format!("{f}.lon"), ctx, p.1, vd(&items(w)[1]));
        }
    }
}

fn short(text: &str) -> String {
    text.chars().take(60).collect()
}

#[test]
fn geoimport_matches_python() {
    let mut t = Tally::default();
    for r in records("geoimport.jsonl") {
        match s(&r, "fn") {
            "kml" => {
                let text = s(&r, "text");
                let got = geoimport::parse_kml(text);
                match r.get("out") {
                    Some(w) => compare_feats(&mut t, "kml", &short(text), &got, w),
                    None => t.fail("kml", format!("Python raised {:?}", r.get("error"))),
                }
            }
            "geojson" => {
                let text = s(&r, "text");
                let ctx = short(text);
                match (geoimport::parse_geojson(text), r.get("out")) {
                    (Ok(got), Some(w)) => compare_feats(&mut t, "geojson", &ctx, &got, w),
                    (Err(e), None) if has(&r, "error") => t.eq("geojson.error", &ctx, e.kind, s(&r, "error")),
                    (Err(e), None) if has(&r, "nm") && e.kind == "NotModelled" => t.ok("geojson.notmodelled"),
                    (Ok(_), None) if has(&r, "nm") => t.ok("geojson.notmodelled"),
                    (g, w) => t.fail("geojson", format!("{ctx}: got {g:?} want {w:?} / {:?}", r.get("error"))),
                }
            }
            "load" | "load_bytes" => {
                let text = if s(&r, "fn") == "load" {
                    s(&r, "text").to_string()
                } else {
                    String::from_utf8_lossy(&unhex(s(&r, "data"))).into_owned()
                };
                let ctx = format!("{} {}", s(&r, "ext"), short(&text));
                match (geoimport::load_features_text(s(&r, "ext"), &text), r.get("out")) {
                    (Ok(got), Some(w)) => compare_feats(&mut t, "load", &ctx, &got, w),
                    (Err(e), None) => t.eq("load.error", &ctx, e.kind, s(&r, "error")),
                    (g, w) => t.fail("load", format!("{ctx}: got {g:?} want {w:?}")),
                }
            }
            "kmz" => {
                let data = unhex(s(&r, "data"));
                let ctx = format!("{} bytes", data.len());
                match (geoimport::parse_kmz_bytes(&data), r.get("out")) {
                    (Ok(got), Some(w)) => compare_feats(&mut t, "kmz", &ctx, &got, w),
                    (Err(_), None) => t.eq("kmz.error", &ctx, s(&r, "error"), "BadZipFile"),
                    (g, w) => t.fail("kmz", format!("{ctx}: got {g:?} want {w:?} / {:?}", r.get("error"))),
                }
            }
            o => panic!("fn {o}"),
        }
    }
    t.finish();
}

// ---------------------------------------------------------------------------
// the elevation trait is object safe and DemSampler implements it
// ---------------------------------------------------------------------------

#[test]
fn elevation_sampler_is_a_trait_object() {
    let dat = SceneDatum::new(-12.0464, -77.0428, 154.0, 0.0);
    let sm = DemSampler::new(dat, aws_terrain(), 13);
    let dynamic: &dyn ElevationSampler = &sm;
    assert_eq!(dynamic.elevation_at(-12.0464, -77.0428), None);
    assert_eq!(dynamic.elevation_at_local(V3(0.0, 0.0, 0.0)), None);
}
