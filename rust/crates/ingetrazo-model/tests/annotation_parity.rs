// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! The annotation classes against the Python ones
//! (scripts/golden/make_annotation_fixtures.py): dimensions, leader texts,
//! guides, section planes, saved views and image planes, with the length
//! text they use and the .igz blocks that carry them. Float32 vectors are
//! compared as bits, float64 as bits, `to_dict` as the very text
//! `json.dumps` wrote.

use std::path::Path;

use ingetrazo_formats::pyjson::{dumps, loads, Options, Value};
use ingetrazo_math::{Hypot, V3};
use ingetrazo_model::annotations::Annotations;
use ingetrazo_model::dimension::Dimension;
use ingetrazo_model::guide::Guide;
use ingetrazo_model::image_plane::ImagePlane;
use ingetrazo_model::pyconv::PyErr;
use ingetrazo_model::saved_views::{from_lookat, PyNum, SavedView};
use ingetrazo_model::section::{effective_fill, next_symbol, SectionPlane, UidCounter};
use ingetrazo_model::textlabel::TextLabel;
use ingetrazo_model::units::{fmt_len, format_length, ModelUnits};

/// The fixtures come from Windows: MSVC's hypot.
const H: Hypot = Hypot::Msvc;

fn rows(name: &str) -> Vec<Value> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/annotations").join(name);
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let v: Vec<Value> = text.lines().map(|l| loads(l).unwrap()).collect();
    assert!(v.len() > 1);
    v.into_iter().skip(1).collect() // the first line is the meta
}

fn g<'a>(v: &'a Value, k: &str) -> &'a Value {
    v.get(k).unwrap_or_else(|| panic!("no {k} in {v:?}"))
}

fn s(v: &Value) -> &str {
    match v {
        Value::Str(x) => x,
        _ => panic!("{v:?}"),
    }
}

fn opt_s(v: &Value) -> Option<String> {
    match v {
        Value::Null => None,
        Value::Str(x) => Some(x.clone()),
        _ => panic!("{v:?}"),
    }
}

fn arr(v: &Value) -> &Vec<Value> {
    match v {
        Value::Array(a) => a,
        _ => panic!("{v:?}"),
    }
}

fn b(v: &Value) -> bool {
    match v {
        Value::Bool(x) => *x,
        _ => panic!("{v:?}"),
    }
}

fn n(v: &Value) -> i64 {
    v.as_i64().unwrap_or_else(|| panic!("{v:?}"))
}

fn h32(v: &Value) -> u32 {
    u32::from_str_radix(s(v), 16).unwrap()
}

fn h64(v: &Value) -> u64 {
    u64::from_str_radix(s(v), 16).unwrap()
}

fn f32v(v: &Value) -> f32 {
    f32::from_bits(h32(v))
}

fn f64v(v: &Value) -> f64 {
    f64::from_bits(h64(v))
}

fn v3(v: &Value) -> V3 {
    let a = arr(v);
    V3(f32v(&a[0]), f32v(&a[1]), f32v(&a[2]))
}

fn bits(p: V3) -> [u32; 3] {
    [p.0.to_bits(), p.1.to_bits(), p.2.to_bits()]
}

fn want(v: &Value) -> [u32; 3] {
    let a = arr(v);
    [h32(&a[0]), h32(&a[1]), h32(&a[2])]
}

fn j(v: &Value) -> String {
    dumps(v, &Options::default()).unwrap()
}

fn opt_axis(v: Option<ingetrazo_model::dimension::Axis>) -> Option<String> {
    v.map(|a| a.name().to_string())
}

fn want_axis(v: &Value) -> Option<String> {
    opt_s(v)
}

fn err_kind(e: &PyErr) -> &'static str {
    e.kind
}

#[test]
fn length_text_matches_core_units() {
    let mut formatted = 0;
    let mut fmt = 0;
    for r in rows("units.jsonl") {
        if let Some(m) = r.get("m") {
            let u = opt_s(g(&r, "u"));
            let got = format_length(f64v(m), u.as_deref().unwrap_or(""), n(g(&r, "d")));
            assert_eq!(got, s(g(&r, "out")), "{r:?}");
            formatted += 1;
        } else {
            let units = ModelUnits::of(Some(g(&r, "settings")));
            assert_eq!(fmt_len(f64v(g(&r, "fmt")), &units), s(g(&r, "out")), "{r:?}");
            fmt += 1;
        }
    }
    assert!(formatted > 2000 && fmt > 200);
}

#[test]
fn dimensions_match_core_dimension() {
    let mut linear = 0;
    let all = rows("dimension.jsonl");
    for r in &all {
        let (a, bb, off) = (v3(g(r, "a")), v3(g(r, "b")), v3(g(r, "off")));
        let axis = opt_s(g(r, "axis"));
        let d = Dimension::new(a, bb, off, opt_s(g(r, "layer")), opt_s(g(r, "text")), axis.as_deref());
        assert_eq!(opt_axis(d.axis), want_axis(g(r, "o_axis")), "{r:?}");
        assert_eq!(d.value(H).to_bits(), h64(g(r, "value")), "value {r:?}");
        let units = ModelUnits::of(Some(g(r, "settings")));
        let label = d.label(H, &units);
        assert_eq!(label, s(g(r, "label")), "{r:?}");
        assert_eq!(bits(d.perpendicular_offset(H)), want(g(r, "perp")), "perp {r:?}");
        let (p, q) = d.line_points(H);
        let lp = arr(g(r, "line"));
        assert_eq!((bits(p), bits(q)), (want(&lp[0]), want(&lp[1])), "line {r:?}");
        assert_eq!(bits(d.midpoint(H)), want(g(r, "mid")), "mid {r:?}");
        let disp = arr(g(r, "disp"));
        assert_eq!(d.display_text(&label), s(&disp[0]), "{r:?}");
        assert_eq!(d.display_text("12.5 m"), s(&disp[1]), "{r:?}");
        assert_eq!(d.display_text(""), s(&disp[2]), "{r:?}");
        let cursor = v3(g(r, "cursor"));
        assert_eq!(bits(Dimension::offset_for_cursor(a, bb, cursor, H)), want(g(r, "ofc")), "ofc {r:?}");
        let (po, pa) = Dimension::placement_for_cursor(a, bb, cursor, H);
        let pfc = arr(g(r, "pfc"));
        assert_eq!((bits(po), opt_axis(pa)), (want(&pfc[0]), opt_s(&pfc[1])), "pfc {r:?}");
        linear += pa.is_some() as usize;
        assert_eq!(j(&d.to_igz_value()), s(g(r, "igz")), "igz {r:?}");
    }
    assert!(all.len() >= 400 && linear > 20, "{} rows, {linear} linear placements", all.len());
}

#[test]
fn dimensions_read_like_load_into() {
    let (mut ok, mut bad) = (0, 0);
    for r in rows("dimension_load.jsonl") {
        let got = Dimension::from_igz_value(g(&r, "raw"));
        match (r.get("out"), r.get("err")) {
            (Some(out), _) => {
                assert_eq!(j(&got.unwrap_or_else(|e| panic!("{e:?} for {r:?}")).to_igz_value()), s(out), "{r:?}");
                ok += 1;
            }
            (_, Some(e)) => {
                assert_eq!(err_kind(&got.expect_err(&format!("{r:?}"))), s(e), "{r:?}");
                bad += 1;
            }
            _ => panic!("{r:?}"),
        }
    }
    assert!(ok > 200 && bad >= 5);
}

#[test]
fn text_labels_match_core_textlabel() {
    let (mut made, mut read, mut bad) = (0, 0, 0);
    for r in rows("textlabel.jsonl") {
        if let Some(raw) = r.get("raw") {
            let got = TextLabel::from_dict(raw);
            match (r.get("out"), r.get("err")) {
                (Some(out), _) => {
                    let t = got.unwrap_or_else(|e| panic!("{e:?} for {r:?}"));
                    assert_eq!(j(&t.to_dict()), s(g(out, "dict")), "{r:?}");
                    assert_eq!(bits(t.position()), want(g(out, "pos")), "{r:?}");
                    read += 1;
                }
                (_, Some(e)) => {
                    assert_eq!(err_kind(&got.expect_err(&format!("{r:?}"))), s(e), "{r:?}");
                    bad += 1;
                }
                _ => panic!("{r:?}"),
            }
        } else {
            let t = TextLabel { anchor: v3(g(&r, "anchor")), offset: v3(g(&r, "offset")), text: s(g(&r, "text")).to_string(), layer: opt_s(g(&r, "layer")) };
            assert_eq!(bits(t.position()), want(g(&r, "pos")), "{r:?}");
            assert_eq!(j(&t.to_dict()), s(g(&r, "dict")), "{r:?}");
            made += 1;
        }
    }
    assert!(made > 200 && read > 50 && bad >= 2);
}

#[test]
fn guides_match_core_guide() {
    let (mut made, mut read, mut bad, mut nonfinite) = (0, 0, 0, 0);
    for r in rows("guide.jsonl") {
        if let Some(raw) = r.get("raw") {
            let got = Guide::from_dict(raw, H);
            match (r.get("out"), r.get("err")) {
                (Some(out), _) => {
                    let gd = got.unwrap_or_else(|e| panic!("{e:?} for {r:?}"));
                    assert_eq!(j(&gd.to_dict()), s(g(out, "dict")), "{r:?}");
                    assert_eq!(gd.is_finite(), b(g(out, "finite")), "{r:?}");
                    read += 1;
                }
                (_, Some(e)) => {
                    assert_eq!(err_kind(&got.expect_err(&format!("{r:?}"))), s(e), "{r:?}");
                    bad += 1;
                }
                _ => panic!("{r:?}"),
            }
        } else {
            let opt = |v: &Value| if matches!(v, Value::Null) { None } else { Some(v3(v)) };
            let gd = Guide::new(v3(g(&r, "point")), opt(g(&r, "dir")), opt(g(&r, "origin")), H);
            let o = |v: Option<V3>| v.map(bits);
            assert_eq!(o(gd.direction), opt(g(&r, "o_dir")).map(bits), "dir {r:?}");
            assert_eq!(o(gd.origin), opt(g(&r, "o_origin")).map(bits), "origin {r:?}");
            assert_eq!(gd.is_line(), b(g(&r, "is_line")), "{r:?}");
            assert_eq!(gd.is_finite(), b(g(&r, "finite")), "{r:?}");
            nonfinite += !gd.is_finite() as usize;
            let (sa, sb) = gd.segment();
            let seg = arr(g(&r, "seg"));
            assert_eq!((bits(sa), bits(sb)), (want(&seg[0]), want(&seg[1])), "seg {r:?}");
            assert_eq!((bits(gd.a()), bits(gd.b())), (want(g(&r, "a")), want(g(&r, "b"))), "{r:?}");
            assert_eq!(j(&gd.to_dict()), s(g(&r, "dict")), "{r:?}");
            made += 1;
        }
    }
    assert!(made > 250 && read > 50 && bad >= 3 && nonfinite > 5);
}

fn same_err(got: &Result<SectionPlane, PyErr>, r: &Value) {
    let e = got.as_ref().expect_err(&format!("{r:?}"));
    assert_eq!(e.kind, s(g(r, "err")), "{r:?}");
}

#[test]
fn section_planes_match_core_section() {
    let mut uids = UidCounter::default();
    let (mut made, mut read, mut bad) = (0, 0, 0);
    for r in rows("section.jsonl") {
        let got: Result<SectionPlane, PyErr> = if s(g(&r, "op")) == "new" {
            let uid = match g(&r, "uid") {
                Value::Null => None,
                v => Some(n(v)),
            };
            let mut sp = SectionPlane::new(
                v3(g(&r, "point")),
                v3(g(&r, "normal")),
                s(g(&r, "name")),
                s(g(&r, "symbol")),
                b(g(&r, "active")),
                uid,
                b(g(&r, "hidden")),
                &mut uids,
                H,
            );
            let fill = g(&r, "fill");
            sp.fill = if ingetrazo_model::pyconv::truthy(fill) { Some(fill.clone()) } else { None };
            made += 1;
            Ok(sp)
        } else {
            SectionPlane::from_dict(g(&r, "raw"), &mut uids, H)
        };
        assert_eq!(uids.last, n(g(&r, "next_uid")), "counter after {r:?}");
        let Some(res) = r.get("res") else {
            same_err(&got, &r);
            bad += 1;
            continue;
        };
        let mut sp = got.unwrap_or_else(|e| panic!("{e:?} for {r:?}"));
        if s(g(&r, "op")) == "from" {
            read += 1;
        }
        assert_eq!(bits(sp.normal), want(g(res, "normal")), "{r:?}");
        assert_eq!(sp.uid, n(g(res, "uid")), "{r:?}");
        assert_eq!(j(&sp.to_dict()), s(g(res, "dict")), "{r:?}");
        assert_eq!(sp.fill.as_ref().map(j), opt_s(g(res, "fill")), "{r:?}");
        for (p, sd) in arr(g(res, "pts")).iter().zip(arr(g(res, "side"))) {
            assert_eq!(sp.side(v3(p)).to_bits(), h32(sd), "side {r:?}");
        }
        for e in arr(g(res, "eff")) {
            let e = arr(e);
            let col = arr(&e[1]);
            let style_color = [num(&col[0]), num(&col[1]), num(&col[2])];
            let (pat, color, spacing) = effective_fill(sp.fill.as_ref(), b(&e[0]), style_color).unwrap();
            assert_eq!(pat, e[2], "{r:?}");
            let want_col: Vec<u64> = arr(&e[3]).iter().map(h64).collect();
            assert_eq!(color.iter().map(|c| c.to_bits()).collect::<Vec<_>>(), want_col, "{r:?}");
            assert_eq!(spacing.to_bits(), h64(&e[4]), "{r:?}");
        }
        sp.flip();
        assert_eq!(j(&sp.to_dict()), s(g(res, "flipped")), "{r:?}");
        assert_eq!(sp.side(v3(&arr(g(res, "pts"))[0])).to_bits(), h32(g(res, "flip_side")), "{r:?}");
    }
    assert!(made > 100 && read > 100 && bad >= 3);
}

fn num(v: &Value) -> f64 {
    match v {
        Value::Float(x) => *x,
        _ => panic!("{v:?}"),
    }
}

#[test]
fn section_symbols_and_fills_match() {
    for r in rows("section_symbols.jsonl") {
        let syms: Vec<Option<String>> = arr(g(&r, "syms")).iter().map(opt_s).collect();
        let refs: Vec<Option<&str>> = syms.iter().map(|x| x.as_deref()).collect();
        assert_eq!(next_symbol(&refs), s(g(&r, "out")), "{r:?}");
    }
    let (mut ok, mut bad) = (0, 0);
    for r in rows("section_fill.jsonl") {
        let col = arr(g(&r, "col"));
        let got = effective_fill(Some(g(&r, "fill")), b(g(&r, "sf")), [num(&col[0]), num(&col[1]), num(&col[2])]);
        match (r.get("out"), r.get("err")) {
            (Some(out), _) => {
                let out = arr(out);
                let (pat, color, spacing) = got.unwrap_or_else(|e| panic!("{e:?} for {r:?}"));
                assert_eq!(pat, out[0], "{r:?}");
                let want_col: Vec<u64> = arr(&out[1]).iter().map(h64).collect();
                assert_eq!(color.iter().map(|c| c.to_bits()).collect::<Vec<_>>(), want_col, "{r:?}");
                assert_eq!(spacing.to_bits(), h64(&out[2]), "{r:?}");
                ok += 1;
            }
            (_, Some(e)) => {
                assert_eq!(got.expect_err(&format!("{r:?}")).kind, s(e), "{r:?}");
                bad += 1;
            }
            _ => panic!("{r:?}"),
        }
    }
    assert!(ok > 50 && bad >= 8);
}

fn pynum(v: &Value) -> PyNum {
    match v {
        Value::Int(i) => PyNum::Int(i.clone()),
        Value::Float(x) => PyNum::Float(*x),
        _ => panic!("{v:?}"),
    }
}

fn opt_dict(v: &Value) -> Option<Value> {
    match v {
        Value::Null => None,
        d => Some(d.clone()),
    }
}

fn float(v: &Value) -> f64 {
    match v {
        Value::Float(x) => *x,
        _ => panic!("{v:?}"),
    }
}

#[test]
fn saved_views_match_core_saved_views() {
    for r in rows("saved_view.jsonl") {
        let a = g(&r, "args");
        let t = arr(g(a, "target"));
        let hidden_layers = match g(a, "hidden_layers") {
            Value::Null => None,
            l => Some(arr(l).iter().map(|x| s(x).to_string()).collect::<Vec<_>>()),
        };
        let hidden_objects = match g(a, "hidden_objects") {
            Value::Null => None,
            l => Some(arr(l).clone()),
        };
        let v = SavedView::new(
            s(g(a, "name")),
            [pynum(&t[0]), pynum(&t[1]), pynum(&t[2])],
            float(g(a, "distance")),
            float(g(a, "yaw")),
            float(g(a, "pitch")),
            float(g(a, "fov_deg")),
            b(g(a, "perspective")),
            b(g(a, "two_point")),
            hidden_layers,
            opt_dict(g(a, "style")),
            opt_dict(g(a, "section")),
            opt_dict(g(a, "georef")),
            opt_dict(g(a, "shadows")),
            hidden_objects,
            opt_dict(g(a, "hidden_shown")),
        );
        assert_eq!(j(&v.to_dict()), s(g(&r, "dict")), "{r:?}");
    }
    let (mut ok, mut bad) = (0, 0);
    for r in rows("saved_view_load.jsonl") {
        let got = SavedView::from_dict(g(&r, "raw"));
        match (r.get("out"), r.get("err")) {
            (Some(out), _) => {
                assert_eq!(j(&got.unwrap_or_else(|e| panic!("{e:?} for {r:?}")).to_dict()), s(out), "{r:?}");
                ok += 1;
            }
            (_, Some(e)) => {
                assert_eq!(got.expect_err(&format!("{r:?}")).kind, s(e), "{r:?}");
                bad += 1;
            }
            _ => panic!("{r:?}"),
        }
    }
    assert!(ok > 100 && bad >= 10);
}

#[test]
fn from_lookat_matches_core_saved_views() {
    let (mut ok, mut bad) = (0, 0);
    let tri = |v: &Value| {
        let a = arr(v);
        [float(&a[0]), float(&a[1]), float(&a[2])]
    };
    for r in rows("saved_view_lookat.jsonl") {
        let oh = match g(&r, "oh") {
            Value::Null => None,
            v => Some(float(v)),
        };
        let hl = match g(&r, "hl") {
            Value::Null => None,
            l => Some(arr(l).iter().map(|x| s(x).to_string()).collect::<Vec<_>>()),
        };
        let got = from_lookat("Importada", tri(g(&r, "eye")), tri(g(&r, "target")), tri(g(&r, "up")), float(g(&r, "fov")), b(g(&r, "persp")), oh, hl);
        match (r.get("out"), r.get("err")) {
            (Some(out), _) => {
                assert_eq!(j(&got.unwrap_or_else(|e| panic!("{e:?} for {r:?}")).to_dict()), s(out), "{r:?}");
                ok += 1;
            }
            (_, Some(e)) => {
                assert_eq!(got.expect_err(&format!("{r:?}")).kind, s(e), "{r:?}");
                bad += 1;
            }
            _ => panic!("{r:?}"),
        }
    }
    assert!(ok > 200 && bad >= 3);
}

#[test]
fn image_planes_match_core_image_plane() {
    let (mut made, mut read, mut bad) = (0, 0, 0);
    for r in rows("image_plane.jsonl") {
        if let Some(raw) = r.get("raw") {
            let got = ImagePlane::from_dict(raw);
            match (r.get("out"), r.get("err")) {
                (Some(out), _) => {
                    let ip = got.unwrap_or_else(|e| panic!("{e:?} for {r:?}"));
                    assert_eq!(j(&ip.to_dict()), s(g(out, "dict")), "{r:?}");
                    assert_eq!(ip.layer, opt_s(g(out, "layer")), "{r:?}");
                    assert_eq!(ip.aspect.to_bits(), h64(g(out, "aspect")), "{r:?}");
                    read += 1;
                }
                (_, Some(e)) => {
                    assert_eq!(got.expect_err(&format!("{r:?}")).kind, s(e), "{r:?}");
                    bad += 1;
                }
                _ => panic!("{r:?}"),
            }
            continue;
        }
        let layer = opt_s(g(&r, "layer"));
        let ip = ImagePlane::new(
            s(g(&r, "path")),
            v3(g(&r, "origin")),
            v3(g(&r, "u")),
            v3(g(&r, "v")),
            f64v(g(&r, "aspect")),
            s(g(&r, "name")),
            f64v(g(&r, "opacity")),
            layer.as_deref(),
            b(g(&r, "locked")),
        );
        assert_eq!(ip.aspect.to_bits(), h64(g(&r, "o_aspect")), "{r:?}");
        let corners = ip.corners();
        for (c, w) in corners.iter().zip(arr(g(&r, "corners"))) {
            assert_eq!(bits(*c), want(w), "corner {r:?}");
        }
        assert_eq!(bits(ip.center()), want(g(&r, "center")), "{r:?}");
        assert_eq!(ip.width(H).to_bits(), h64(g(&r, "width")), "{r:?}");
        assert_eq!(ip.height(H).to_bits(), h64(g(&r, "height")), "{r:?}");
        assert_eq!(bits(ip.normal(H)), want(g(&r, "normal")), "{r:?}");
        let (pp, pn) = ip.plane(H);
        let plane = arr(g(&r, "plane"));
        assert_eq!((bits(pp), bits(pn)), (want(&plane[0]), want(&plane[1])), "{r:?}");
        for ((p, q), w) in ip.border_edges().iter().zip(arr(g(&r, "border"))) {
            let w = arr(w);
            assert_eq!((bits(*p), bits(*q)), (want(&w[0]), want(&w[1])), "border {r:?}");
        }
        for w in arr(g(&r, "uv_in")) {
            let w = arr(w);
            assert_eq!(ip.contains_uv(f64v(&w[0]), f64v(&w[1])), b(&w[2]), "{r:?}");
        }
        let (pu, pv) = ip.project(v3(g(&r, "pt")));
        let prj = arr(g(&r, "project"));
        assert_eq!((pu.to_bits(), pv.to_bits()), (h64(&prj[0]), h64(&prj[1])), "project {r:?}");
        for sc in arr(g(&r, "scaled")) {
            let sc = arr(sc);
            let (su, sv) = ip.scaled(f64v(&sc[0]), b(&sc[1]), H);
            assert_eq!((bits(su), bits(sv)), (want(&sc[2]), want(&sc[3])), "scaled {r:?}");
        }
        assert_eq!(j(&ip.to_dict()), s(g(&r, "dict")), "{r:?}");
        made += 1;
    }
    assert!(made > 150 && read > 50 && bad >= 3);
}

#[test]
fn whole_payloads_read_and_write_like_the_igz_loader() {
    let (mut ok, mut bad, mut unsavable, mut repaired) = (0, 0, 0, 0);
    for r in rows("scene.jsonl") {
        let mut uids = UidCounter::default();
        let got = Annotations::from_igz_payload(g(&r, "sections"), &mut uids, H);
        if let Some(e) = r.get("err") {
            let e = s(e);
            if let Some(kind) = e.strip_prefix("save:") {
                // Read fine; the writer refuses a NaN or an infinity (allow_nan=False).
                assert_eq!(kind, "NonFiniteDocumentError");
                let a = got.unwrap_or_else(|e| panic!("{e:?} for {r:?}"));
                let strict = Options::default().allow_nan(false);
                assert!(a.to_igz_entries().iter().any(|(_, v)| dumps(v, &strict).is_err()), "{r:?}");
                unsavable += 1;
            } else {
                assert_eq!(got.expect_err(&format!("{r:?}")).kind, e, "{r:?}");
                bad += 1;
            }
            continue;
        }
        let a = got.unwrap_or_else(|e| panic!("{e:?} for {r:?}"));
        let out: Vec<(String, String)> = a.to_igz_entries().iter().map(|(k, v)| (k.to_string(), j(v))).collect();
        let want: Vec<(String, String)> = arr(g(&r, "out")).iter().map(|p| (s(&arr(p)[0]).to_string(), s(&arr(p)[1]).to_string())).collect();
        assert_eq!(out, want, "{r:?}");
        assert_eq!(a.load_repairs as i64, n(g(&r, "repairs")), "{r:?}");
        assert_eq!(a.show_section_planes, b(g(&r, "planes_shown")), "{r:?}");
        assert_eq!(a.show_section_cuts, b(g(&r, "cuts_shown")), "{r:?}");
        assert_eq!(uids.last, n(g(&r, "next_uid")), "{r:?}");
        repaired += (a.load_repairs > 0) as usize;
        ok += 1;
    }
    assert!(ok > 50 && bad >= 3 && unsavable >= 1 && repaired >= 3, "{ok} {bad} {unsavable} {repaired}");
}
