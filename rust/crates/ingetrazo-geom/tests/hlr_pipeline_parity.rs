// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! `hlr_draw` against `core/hlr.py` (scripts/golden/make_hlr_fixtures.py): the
//! same segments, world endpoints, line classes and section rings, bit for
//! bit, on boxes, stacks, prisms with soft edges, an L-shape, coplanar
//! overlaps, silhouettes, sections with and without the cap, hidden lines and
//! perspective (two-point included).

use std::collections::HashMap;
use std::path::Path;

use ingetrazo_formats::pyjson::{loads, Value};
use ingetrazo_geom::hlr_draw::*;
use ingetrazo_math::{Hypot, M4, V3};

fn arr(v: &Value) -> &Vec<Value> {
    match v {
        Value::Array(a) => a,
        _ => panic!("{v:?}"),
    }
}

fn s(v: &Value) -> &str {
    match v {
        Value::Str(h) => h,
        _ => panic!("{v:?}"),
    }
}

fn get<'a>(v: &'a Value, k: &str) -> &'a Value {
    v.get(k).unwrap_or_else(|| panic!("missing {k}"))
}

fn f32h(v: &Value) -> f32 {
    f32::from_bits(u32::from_str_radix(s(v), 16).unwrap())
}

fn f64h(v: &Value) -> f64 {
    f64::from_bits(u64::from_str_radix(s(v), 16).unwrap())
}

/// A string of 16-digit hex numbers.
fn nums(v: &Value) -> Vec<f64> {
    let h = s(v);
    (0..h.len() / 16).map(|i| f64::from_bits(u64::from_str_radix(&h[i * 16..i * 16 + 16], 16).unwrap())).collect()
}

fn v3(v: &Value) -> V3 {
    let a = arr(v);
    V3(f32h(&a[0]), f32h(&a[1]), f32h(&a[2]))
}

fn p3(c: &[f64]) -> [f64; 3] {
    [c[0], c[1], c[2]]
}

fn geometry(g: &Value) -> Geometry {
    let tris = nums(get(g, "tris")).chunks(9).map(|c| [p3(&c[0..3]), p3(&c[3..6]), p3(&c[6..9])]).collect();
    let hard = nums(get(g, "hard")).chunks(6).map(|c| [p3(&c[0..3]), p3(&c[3..6])]).collect();
    let open: Vec<i64> = arr(get(g, "soft_open")).iter().map(|x| x.as_i64().unwrap()).collect();
    let soft = nums(get(g, "soft"))
        .chunks(12)
        .zip(open)
        .map(|(c, o)| SoftEdge { p0: p3(&c[0..3]), p1: p3(&c[3..6]), na: p3(&c[6..9]), nb: if o == 1 { None } else { Some(p3(&c[9..12])) } })
        .collect();
    Geometry { tris, hard, soft }
}

fn camera(c: &Value) -> OrbitCamera {
    OrbitCamera {
        target: v3(get(c, "target")),
        distance: f64h(get(c, "distance")),
        yaw: f64h(get(c, "yaw")),
        pitch: f64h(get(c, "pitch")),
        up: v3(get(c, "up")),
    }
}

fn section(v: &Value) -> Option<SectionPlane> {
    match v {
        Value::Null => None,
        _ => Some(SectionPlane { normal: v3(get(v, "normal")), point: v3(get(v, "point")) }),
    }
}

fn bits(x: &[f64]) -> Vec<u64> {
    x.iter().map(|v| v.to_bits()).collect()
}

fn first_diff(got: &[f64], want: &[f64]) -> String {
    if got.len() != want.len() {
        return format!("length {} vs {}", got.len(), want.len());
    }
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        if g.to_bits() != w.to_bits() {
            return format!("[{i}] {g:e} vs {w:e}");
        }
    }
    String::new()
}

fn flat4(v: &[[f64; 4]]) -> Vec<f64> {
    v.iter().flatten().copied().collect()
}

fn flat_w(v: &[Seg]) -> Vec<f64> {
    v.iter().flat_map(|s| s.iter().flatten()).copied().collect()
}

fn check(fails: &mut Vec<String>, name: &str, what: &str, got: &[f64], want: &[f64]) {
    if bits(got) != bits(want) {
        fails.push(format!("{name}: {what} {}", first_diff(got, want)));
    }
}

#[test]
fn hlr_pipeline_matches_python() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/hlr_pipeline.jsonl");
    let text = std::fs::read_to_string(path).unwrap();
    let h = Hypot::Msvc;
    let mut geoms: HashMap<String, Geometry> = HashMap::new();
    let mut fails: Vec<String> = Vec::new();
    let (mut n, mut nseg) = (0usize, 0usize);
    for line in text.lines() {
        let c = loads(line).unwrap();
        let kind = s(get(&c, "kind"));
        if kind == "geom" {
            geoms.insert(s(get(&c, "id")).to_string(), geometry(get(&c, "geom")));
            continue;
        }
        let name = s(get(&c, "name")).to_string();
        let geom = &geoms[s(get(&c, "geom"))];
        let out = get(&c, "out");
        n += 1;
        match kind {
            "drawing" | "view" => {
                let cam = camera(get(&c, "cam"));
                let sec = section(get(&c, "section"));
                let vectorised = s(get(&c, "mode")) == "arrays";
                let d = if kind == "drawing" {
                    let o = get(&c, "opts");
                    let flag = |k: &str| matches!(get(o, k), Value::Bool(true));
                    let opts = HlrOptions { profiles: flag("profiles"), fills: flag("fills"), caps: flag("caps"), hidden: flag("hidden") };
                    let b = camera_basis(&cam, h);
                    let want_b = nums(get(&c, "basis"));
                    let got_b: Vec<f64> = [b.eye, b.right, b.up, b.fwd].iter().flatten().copied().collect();
                    check(&mut fails, &name, "basis", &got_b, &want_b);
                    hlr_drawing(geom, vectorised, sec.as_ref(), &cam, &opts, h)
                } else {
                    hlr_view(geom, vectorised, sec.as_ref(), &cam, h)
                };
                nseg += d.segs.len();
                check(&mut fails, &name, "segs", &flat4(&d.segs), &nums(get(out, "segs")));
                check(&mut fails, &name, "world", &flat_w(&d.world), &nums(get(out, "world")));
                if kind == "drawing" {
                    let want_k: Vec<i8> = arr(get(out, "kinds")).iter().map(|x| x.as_i64().unwrap() as i8).collect();
                    if d.kinds != want_k {
                        fails.push(format!("{name}: kinds differ"));
                    }
                    let want_l = arr(get(out, "loops"));
                    if d.loops.len() != want_l.len() {
                        fails.push(format!("{name}: {} loops vs {}", d.loops.len(), want_l.len()));
                    } else {
                        for (i, (l, w)) in d.loops.iter().zip(want_l).enumerate() {
                            let got: Vec<f64> = l.iter().flatten().copied().collect();
                            check(&mut fails, &name, &format!("loop {i}"), &got, &nums(w));
                        }
                    }
                }
            }
            "persp" => {
                let cam = camera(get(&c, "cam"));
                let sec = section(get(&c, "section"));
                let mat = |k: &str| {
                    let t = s(get(&c, k));
                    let d: Vec<f32> = (0..16).map(|i| f32::from_bits(u32::from_str_radix(&t[i * 8..i * 8 + 8], 16).unwrap())).collect();
                    M4::from_data(d.try_into().unwrap())
                };
                let pv = PerspectiveView { view: mat("view"), projection: mat("proj"), fov_deg: f64h(get(&c, "fov")), aspect: f64h(get(&c, "aspect")) };
                let segs = hlr_perspective(geom, sec.as_ref(), &cam, &pv, h);
                nseg += segs.len();
                check(&mut fails, &name, "segs", &flat4(&segs), &nums(get(out, "segs")));
            }
            "clip" => {
                let sp = section(get(&c, "section")).unwrap();
                let split = matches!(get(&c, "split"), Value::Bool(true));
                let r = clip_to_section(geom, &sp, split);
                let tris: Vec<f64> = r.tris.iter().flat_map(|t| t.iter().flatten()).copied().collect();
                check(&mut fails, &name, "tris", &tris, &nums(get(out, "tris")));
                check(&mut fails, &name, "hard", &flat_w(&r.hard), &nums(get(out, "hard")));
                check(&mut fails, &name, "cuts", &flat_w(&r.cuts), &nums(get(out, "cuts")));
                let soft: Vec<f64> = r
                    .soft
                    .iter()
                    .flat_map(|e| [e.p0, e.p1, e.na, e.nb.unwrap_or([0.0; 3])].into_iter().flatten())
                    .collect();
                check(&mut fails, &name, "soft", &soft, &nums(get(out, "soft")));
                let open: Vec<i64> = r.soft.iter().map(|e| e.nb.is_none() as i64).collect();
                let want_open: Vec<i64> = arr(get(out, "soft_open")).iter().map(|x| x.as_i64().unwrap()).collect();
                if open != want_open {
                    fails.push(format!("{name}: soft_open differs"));
                }
            }
            other => panic!("unknown kind {other}"),
        }
    }
    assert!(
        fails.is_empty(),
        "{} problems in {n} cases ({nseg} segments):\n{}",
        fails.len(),
        fails.iter().take(25).cloned().collect::<Vec<_>>().join("\n")
    );
    assert!(n > 300 && nseg > 5000, "fixtures look truncated: {n} cases, {nseg} segments");
}
