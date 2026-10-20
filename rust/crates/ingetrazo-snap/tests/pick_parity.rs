// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! Picking against the Python app (scripts/golden/make_pick_fixtures.py):
//! for 11 scenes (loose boxes over a coplanar floor, component instances
//! rotated, scaled and mirrored, nested placements, hidden and locked
//! groups and layers, lines-only groups, dimensions, text labels, guides,
//! section planes, image planes, georef paths, an active cut, x-ray,
//! wireframe, hidden things shown) seen from several cameras, with groups
//! opened for editing at several depths, the real `Viewport` answered for
//! hundreds of pixels each: `pick_face`, `pick_face_any`,
//! `pick_face_placement`, `pick_group`, `pick_edge`, `pick_visible_edge`,
//! `pick_vertex`, `pick_dimension`, `pick_text_label`, `pick_guide`,
//! `pick_section_plane`, `pick_image_plane`, `pick_locked_image_plane`,
//! `pick_axis`, `pick_geopath` and the Select tool's priority — and for
//! rubber-band boxes the Select tool's window and crossing selection. Every
//! answer must be the very same entity (or f32 bits).
//!
//! The camera the fixture records is checked first: the view and projection
//! matrices (f32 bits) and the eye must come out of `Camera` as Python made
//! them.
//!
//! The fixtures come from a Windows machine: `length()` is MSVC's `hypot`.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use ingetrazo_formats::igz;
use ingetrazo_formats::pyjson::{loads, Value};
use ingetrazo_geom::hlr_draw::OrbitCamera;
use ingetrazo_math::{Hypot, V3};
use ingetrazo_model::scene::{MeshId, Scene};
use ingetrazo_snap::picking::{open_scene, Camera, EntRef, FaceMode, Picked, Picker, State, TextMetrics, View};

const H: Hypot = Hypot::Msvc;
const W: i32 = 1000;
const HH: i32 = 600;

fn data() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/pick")
}

fn open(name: &str) -> Scene {
    let raw = std::fs::read(data().join(format!("{name}.igz.json"))).unwrap();
    let doc = igz::read_document(&raw).unwrap();
    Scene::from_document(&doc).unwrap_or_else(|e| panic!("{name}: {e:?}"))
}

fn arr(v: &Value) -> &[Value] {
    match v {
        Value::Array(a) => a,
        other => panic!("not an array: {other:?}"),
    }
}

fn text(v: &Value) -> &str {
    match v {
        Value::Str(s) => s,
        other => panic!("not a string: {other:?}"),
    }
}

fn int(v: &Value) -> i64 {
    v.as_i64().unwrap_or_else(|| panic!("not an int: {v:?}"))
}

fn f64_of(v: &Value) -> f64 {
    f64::from_bits(u64::from_str_radix(text(v), 16).unwrap())
}

fn f32_of(v: &Value) -> f32 {
    f32::from_bits(u32::from_str_radix(text(v), 16).unwrap())
}

fn v3_of(v: &Value) -> V3 {
    let a = arr(v);
    V3(f32_of(&a[0]), f32_of(&a[1]), f32_of(&a[2]))
}

fn bits(x: f32) -> Value {
    Value::Str(format!("{:08x}", x.to_bits()))
}

fn vbits(p: V3) -> Value {
    Value::Array(vec![bits(p.0), bits(p.1), bits(p.2)])
}

fn path_value(p: &[usize]) -> Value {
    Value::Array(p.iter().map(|&i| Value::int(i as i64)).collect())
}

fn idx_value(i: usize) -> Value {
    Value::int(i as i64)
}

fn s(x: &str) -> Value {
    Value::Str(x.to_string())
}

struct Metrics {
    height: i32,
    ascent: i32,
    descent: i32,
    adv: BTreeMap<String, i32>,
}

impl TextMetrics for Metrics {
    fn height(&self) -> i32 {
        self.height
    }
    fn ascent(&self) -> i32 {
        self.ascent
    }
    fn descent(&self) -> i32 {
        self.descent
    }
    fn advance(&self, line: &str) -> i32 {
        *self.adv.get(line).unwrap_or_else(|| panic!("no advance recorded for {line:?}"))
    }
}

fn metrics(v: &Value) -> Metrics {
    let adv = match v.get("adv").unwrap() {
        Value::Object(items) => items.iter().map(|(k, v)| (k.clone(), int(v) as i32)).collect(),
        other => panic!("{other:?}"),
    };
    Metrics { height: int(v.get("height").unwrap()) as i32, ascent: int(v.get("ascent").unwrap()) as i32, descent: int(v.get("descent").unwrap()) as i32, adv }
}

fn camera(c: &Value) -> Camera {
    let g = |k: &str| c.get(k).unwrap();
    let orbit = OrbitCamera { target: v3_of(g("target")), distance: f64_of(g("distance")), yaw: f64_of(g("yaw")), pitch: f64_of(g("pitch")), up: v3_of(g("up")) };
    let mut cam = Camera::new(orbit);
    cam.perspective = matches!(g("perspective"), Value::Bool(true));
    cam.fov_deg = f64_of(g("fov"));
    cam.aspect = f64_of(g("aspect"));
    cam.znear = f64_of(g("znear"));
    cam.zfar = f64_of(g("zfar"));
    cam
}

fn m4_bits(m: &ingetrazo_math::M4) -> Value {
    Value::Array(m.data().iter().map(|&x| bits(x)).collect())
}

fn face_enc(picker: &Picker, face: EntRef) -> Value {
    match face {
        EntRef::Loose(i) => Value::Array(vec![s("l"), idx_value(i)]),
        EntRef::Placed { place, face } => Value::Array(vec![s("g"), path_value(&picker.idx.places[place].path), idx_value(face)]),
    }
}

fn picked_enc(p: &Picked) -> Value {
    let pair = |k: &str, i: usize| Value::Array(vec![s(k), idx_value(i)]);
    match p {
        Picked::Face(i) => pair("face", *i),
        Picked::Edge(i) => pair("edge", *i),
        Picked::Group(g) => Value::Array(vec![s("group"), path_value(g)]),
        Picked::Dimension(i) => pair("dim", *i),
        Picked::Label(i) => pair("label", *i),
        Picked::GeoPath(i) => pair("geo", *i),
        Picked::Section(i) => pair("section", *i),
        Picked::Guide(i) => pair("guide", *i),
        Picked::Image(i) => pair("image", *i),
    }
}

/// The battery the generator asked Python, in Rust.
fn answers(picker: &Picker, fm: &Metrics, x: f64, y: f64) -> BTreeMap<String, Value> {
    let mut out: BTreeMap<String, Value> = BTreeMap::new();
    let mut put = |k: &str, v: Option<Value>| {
        if let Some(v) = v {
            out.insert(k.to_string(), v);
        }
    };
    put("f", picker.pick_face(x, y).map(|i| Value::Array(vec![s("l"), idx_value(i)])));
    if let Some(h) = picker.pick_face_any(x, y) {
        put("fa", Some(Value::Array(vec![
            face_enc(picker, h.face),
            h.owner.as_deref().map_or(Value::Null, path_value),
            h.placement.as_deref().map_or(Value::Null, path_value),
        ])));
    }
    put("g", picker.pick_group(x, y).map(|p| path_value(&p)));
    put("e", picker.pick_edge(x, y).map(idx_value));
    put("ve", picker.pick_visible_edge(x, y).map(idx_value));
    put("v", picker.pick_vertex(x, y).map(vbits));
    put("d", picker.pick_dimension(x, y).map(idx_value));
    put("t", picker.pick_text_label(x, y, false, fm).map(idx_value));
    put("tr", picker.pick_text_label(x, y, true, fm).map(idx_value));
    put("gd", picker.pick_guide(x, y).map(idx_value));
    put("sp", picker.pick_section_plane(x, y).map(idx_value));
    put("im", picker.pick_image_plane(x, y).map(idx_value));
    put("il", picker.pick_locked_image_plane(x, y).map(idx_value));
    put("ax", picker.pick_axis(x, y).map(s));
    put("gp", picker.pick_geopath(x, y).map(idx_value));
    put("s", picker.select_pick(x, y, fm).map(|p| picked_enc(&p)));
    out
}

fn expected_map(v: &Value) -> BTreeMap<String, Value> {
    match v {
        Value::Object(items) => items.iter().map(|(k, v)| (k.clone(), v.clone())).collect(),
        other => panic!("{other:?}"),
    }
}

#[test]
fn picks_match_python() {
    let raw = std::fs::read(data().join("cases.json")).unwrap();
    let views = loads(std::str::from_utf8(&raw).unwrap()).unwrap();
    let mut scenes: BTreeMap<String, Scene> = BTreeMap::new();
    let mut n_px = 0usize;
    let mut n_rect = 0usize;
    let mut bad: Vec<String> = Vec::new();
    let mut n_amb = 0usize;
    let mut by_key: BTreeMap<String, usize> = BTreeMap::new();
    for (vi, view) in arr(&views).iter().enumerate() {
        let name = text(view.get("scene").unwrap()).to_string();
        let tag = format!("view {vi} ({name}, edit {:?}, rest {})", view.get("edit").unwrap(), text(view.get("rest").unwrap()));
        let base = scenes.entry(name.clone()).or_insert_with(|| open(&name));
        let edit: Option<Vec<usize>> = match view.get("edit").unwrap() {
            Value::Null => None,
            v => Some(arr(v).iter().map(|i| int(i) as usize).collect()),
        };
        let (sc, active) = open_scene(base, edit.as_deref(), H);
        let rest_hidden = edit.is_some() && text(view.get("rest").unwrap()) == "hide";
        let face_mode = FaceMode::from_name(text(view.get("face_mode").unwrap()));

        let crec = view.get("cam").unwrap();
        let cam = camera(crec);
        let v = View::new(cam, W, HH, H);
        for (key, got) in [("view", m4_bits(&v.view)), ("proj", m4_bits(&v.proj)), ("eye", vbits(v.eye))] {
            if &got != crec.get(key).unwrap() {
                bad.push(format!("{tag}: camera {key}\n  rust   {got:?}\n  python {:?}", crec.get(key).unwrap()));
            }
        }
        let fm = metrics(view.get("metrics").unwrap());
        let st = State::new(&sc, active, MeshId(0), edit.clone(), rest_hidden, face_mode, H);
        let picker = Picker::new(st, v);

        for rec in arr(view.get("px").unwrap()) {
            let r = arr(rec);
            let (x, y) = (f64_of(&r[0]), f64_of(&r[1]));
            let mut want = expected_map(&r[2]);
            let mut got = answers(&picker, &fm, x, y);
            n_px += 1;
            // The answers NumPy's unstable argsort chose between equals.
            if let Some(Value::Array(amb)) = want.remove("amb") {
                for k in amb.iter().map(text) {
                    want.remove(k);
                    got.remove(k);
                    n_amb += 1;
                }
            }
            if got != want {
                let diff: Vec<String> = want
                    .keys()
                    .chain(got.keys())
                    .collect::<std::collections::BTreeSet<_>>()
                    .into_iter()
                    .filter(|k| got.get(*k) != want.get(*k))
                    .map(|k| {
                        *by_key.entry(k.clone()).or_insert(0usize) += 1;
                        format!("{k}: rust {:?} python {:?}", got.get(k), want.get(k))
                    })
                    .collect();
                bad.push(format!("{tag} px ({x}, {y}): {}", diff.join("; ")));
            }
        }
        for rec in arr(view.get("rects").unwrap()) {
            let r = arr(rec);
            let rect: Vec<f64> = arr(&r[0]).iter().map(f64_of).collect();
            let crossing = matches!(r[1], Value::Bool(true));
            let got: Vec<Value> = picker.box_select((rect[0], rect[1], rect[2], rect[3]), crossing).iter().map(picked_enc).collect();
            let want = arr(&r[2]).to_vec();
            n_rect += 1;
            if got != want {
                bad.push(format!("{tag} rect {rect:?} crossing {crossing}: rust {got:?} python {want:?}"));
            }
        }
    }
    eprintln!("{n_px} pixels and {n_rect} boxes over {} views, {} differ ({n_amb} answers left to numpy's unstable argsort); by key {by_key:?}", arr(&views).len(), bad.len());
    for b in bad.iter().take(25) {
        eprintln!("{b}");
    }
    assert!(bad.is_empty(), "{} answers differ", bad.len());
}
