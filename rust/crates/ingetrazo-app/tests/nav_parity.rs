// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! Camera navigation against the Python viewport
//! (scripts/golden/make_nav_fixtures.py): the real `Viewport` received
//! synthetic mouse and wheel events on four `examples/*.igz` models —
//! middle-drag orbits (about the pivot under the cursor), Shift+middle-drag
//! pans, wheel notches zoom towards the cursor (over a burst, with the
//! cached pin), the left button in the Orbit, Pan and Zoom modes, zoom
//! extents, the standard views, perspective, parallel and two-point
//! projections, the pitch clamp and both distance limits, with and without
//! the inverted orbit and wheel preferences — and after each event the
//! camera is compared bit for bit.
//!
//! Two replays of every sequence:
//! * `maths`: the camera calls the viewport made, with the arguments it made
//!   them (the pivot, the focus, the depth), through `nav::orbit_about` and
//!   friends only;
//! * `gestures`: the events through `Navigator` and the scene probe, which
//!   find their own pivots and foci (ray against the pick index).
//!
//! The fixtures come from a Windows machine: `length()` is MSVC's `hypot`.

use std::path::{Path, PathBuf};

use ingetrazo_app::nav::{self, Button, NavMode, NavSettings, Navigator, Probe};
use ingetrazo_app::probe::{PickCache, ScenePick};
use ingetrazo_formats::pyjson::{loads, Value};
use ingetrazo_math::{Hypot, V3};
use ingetrazo_model::scene::Scene;
use ingetrazo_render::document::open_igz;
use ingetrazo_render::Camera;

const H: Hypot = Hypot::Msvc;

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../..")
}

fn open(name: &str) -> Scene {
    let raw = std::fs::read(root().join("examples").join(format!("{name}.igz"))).unwrap();
    let (mut scene, _) = open_igz(&raw).unwrap_or_else(|e| panic!("{name}: {e}"));
    for m in &mut scene.meshes {
        m.hypot = H;
    }
    scene
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

fn num(v: &Value) -> f64 {
    match v {
        Value::Float(f) => *f,
        Value::Int(_) => v.as_i64().unwrap() as f64,
        other => panic!("not a number: {other:?}"),
    }
}

fn truthy(v: &Value) -> bool {
    matches!(v, Value::Bool(true))
}

fn f64_of(v: &Value) -> f64 {
    f64::from_bits(u64::from_str_radix(text(v), 16).unwrap())
}

fn v3_of(v: &Value) -> V3 {
    let a = arr(v);
    let f = |i: usize| f32::from_bits(u32::from_str_radix(text(&a[i]), 16).unwrap());
    V3(f(0), f(1), f(2))
}

fn get<'a>(v: &'a Value, k: &str) -> &'a Value {
    v.get(k).unwrap_or_else(|| panic!("missing {k}"))
}

/// A recorded camera call argument.
fn arg_f64(a: &Value) -> f64 {
    if let Some(i) = a.get("int") {
        num(i)
    } else {
        f64_of(get(a, "f64"))
    }
}

fn arg_opt_f64(a: &Value) -> Option<f64> {
    if a.get("none").is_some() {
        None
    } else {
        Some(arg_f64(a))
    }
}

/// The camera, as a list of its bit patterns (and a readable form).
fn cam_bits(c: &Camera) -> (Vec<u64>, String) {
    let bits = vec![
        c.target.0.to_bits() as u64,
        c.target.1.to_bits() as u64,
        c.target.2.to_bits() as u64,
        c.distance.to_bits(),
        c.yaw.to_bits(),
        c.pitch.to_bits(),
        c.fov_deg.to_bits(),
        c.aspect.to_bits(),
        c.perspective as u64,
        c.two_point as u64,
    ];
    let shown = format!(
        "target {:?} distance {} yaw {} pitch {} persp {} two_point {}",
        c.target, c.distance, c.yaw, c.pitch, c.perspective, c.two_point
    );
    (bits, shown)
}

fn recorded(v: &Value) -> (Vec<u64>, String) {
    let t = v3_of(get(v, "target"));
    let bits = vec![
        t.0.to_bits() as u64,
        t.1.to_bits() as u64,
        t.2.to_bits() as u64,
        f64_of(get(v, "distance")).to_bits(),
        f64_of(get(v, "yaw")).to_bits(),
        f64_of(get(v, "pitch")).to_bits(),
        f64_of(get(v, "fov")).to_bits(),
        f64_of(get(v, "aspect")).to_bits(),
        truthy(get(v, "perspective")) as u64,
        truthy(get(v, "two_point")) as u64,
    ];
    let shown = format!(
        "target {t:?} distance {} yaw {} pitch {} persp {} two_point {}",
        f64_of(get(v, "distance")),
        f64_of(get(v, "yaw")),
        f64_of(get(v, "pitch")),
        truthy(get(v, "perspective")),
        truthy(get(v, "two_point")),
    );
    (bits, shown)
}

fn load_cases() -> Vec<Value> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/nav/cases.json");
    let raw = std::fs::read(path).unwrap();
    match loads(std::str::from_utf8(&raw).unwrap()).unwrap() {
        Value::Array(a) => a,
        other => panic!("{other:?}"),
    }
}

fn start_camera(case: &Value) -> (Camera, (i32, i32)) {
    let (w, h) = (get(case, "w").as_i64().unwrap() as i32, get(case, "h").as_i64().unwrap() as i32);
    let mut cam = Camera { hypot: H, ..Camera::default() };
    cam.set_aspect(w as u32, h as u32);
    (cam, (w, h))
}

/// `MainWindow._on_zoom_extents` (without face-me figures and images).
fn zoom_extents(cam: &mut Camera, probe: &mut ScenePick) {
    if let Some((lo, hi)) = probe.bounds() {
        nav::fit_box(cam, lo, hi, nav::FIT_MARGIN);
    }
}

/// The events that are no gesture: true when handled.
fn simple_event(ev: &Value, cam: &mut Camera, nav_state: &mut Navigator, probe: &mut ScenePick) -> bool {
    match text(get(ev, "type")) {
        "zoom_extents" => zoom_extents(cam, probe),
        "toggle_projection" => nav::toggle_projection(cam),
        "toggle_two_point" => nav::toggle_two_point(cam),
        "set_view" => nav::set_view(cam, text(get(ev, "name"))),
        "nav_mode" => match get(ev, "mode") {
            Value::Null => nav_state.set_nav_mode(None),
            m => nav_state.set_nav_mode(NavMode::from_name(text(m))),
        },
        _ => return false,
    }
    true
}

fn button(ev: &Value) -> Button {
    match text(get(ev, "button")) {
        "left" => Button::Left,
        "middle" => Button::Middle,
        other => panic!("{other}"),
    }
}

/// Replay with the camera calls Python made, with Python's arguments.
fn replay_maths(case: &Value) -> Vec<String> {
    let name = text(get(case, "scene"));
    let scene = open(name);
    let mut cache = PickCache::default();
    let mut probe = ScenePick::new(&scene, &mut cache, H);
    let (mut cam, _) = start_camera(case);
    let mut nav_state = Navigator::default();
    let mut bad = Vec::new();
    for (i, e) in arr(get(case, "events")).iter().enumerate() {
        let ev = get(e, "ev");
        if !simple_event(ev, &mut cam, &mut nav_state, &mut probe) {
            for call in arr(get(e, "calls")) {
                let a = get(call, "args");
                match text(get(call, "method")) {
                    "orbit" => nav::orbit(&mut cam, arg_f64(get(a, "dx")), arg_f64(get(a, "dy")), arg_f64(get(a, "h")) as i32),
                    "orbit_about" => nav::orbit_about(&mut cam, v3_of(get(get(a, "pivot"), "v3")), arg_f64(get(a, "dx")), arg_f64(get(a, "dy")), arg_f64(get(a, "h")) as i32),
                    "pan" => nav::pan(&mut cam, arg_f64(get(a, "dx")), arg_f64(get(a, "dy")), arg_f64(get(a, "h")) as i32, arg_opt_f64(get(a, "depth"))),
                    "zoom" => nav::zoom(&mut cam, arg_f64(get(a, "steps"))),
                    "zoom_to" => nav::zoom_to(&mut cam, arg_f64(get(a, "steps")), v3_of(get(get(a, "focus"), "v3")), arg_f64(get(a, "min_step"))),
                    other => panic!("{other}"),
                }
            }
        }
        let (got, shown) = cam_bits(&cam);
        let (want, want_shown) = recorded(get(e, "camera_after"));
        if got != want {
            bad.push(format!("{name} event {i} {ev:?}\n  rust   {shown}\n  python {want_shown}"));
            if bad.len() > 3 {
                break;
            }
            // Carry on from Python's camera, so one slip is one report.
            let t = v3_of(get(get(e, "camera_after"), "target"));
            cam.target = t;
            cam.distance = f64_of(get(get(e, "camera_after"), "distance"));
            cam.yaw = f64_of(get(get(e, "camera_after"), "yaw"));
            cam.pitch = f64_of(get(get(e, "camera_after"), "pitch"));
        }
    }
    bad
}

/// Records what the probe answers, to compare with what Python passed on.
struct Logging<'p, 'a> {
    inner: &'p mut ScenePick<'a>,
    pivots: Vec<V3>,
    foci: Vec<Option<V3>>,
}

impl Probe for Logging<'_, '_> {
    fn scene_version(&self) -> u64 {
        self.inner.scene_version()
    }
    fn pivot_at(&mut self, cam: &Camera, size: (i32, i32), x: f64, y: f64) -> V3 {
        let p = self.inner.pivot_at(cam, size, x, y);
        self.pivots.push(p);
        p
    }
    fn world_under_cursor(&mut self, cam: &Camera, size: (i32, i32), x: f64, y: f64) -> Option<V3> {
        let f = self.inner.world_under_cursor(cam, size, x, y);
        self.foci.push(f);
        f
    }
    fn model_diag(&mut self) -> f64 {
        self.inner.model_diag()
    }
}

/// Replay the events through the gesture state machine and the scene probe.
fn replay_gestures(case: &Value) -> Vec<String> {
    let name = text(get(case, "scene"));
    let scene = open(name);
    let mut cache = PickCache::default();
    let mut scene_probe = ScenePick::new(&scene, &mut cache, H);
    let (mut cam, size) = start_camera(case);
    let mut nav_state =
        Navigator::new(NavSettings { invert_orbit_y: truthy(get(case, "invert_orbit_y")), invert_wheel: truthy(get(case, "invert_wheel")) });
    let mut bad = Vec::new();
    for (i, e) in arr(get(case, "events")).iter().enumerate() {
        let ev = get(e, "ev");
        let mut probe = Logging { inner: &mut scene_probe, pivots: Vec::new(), foci: Vec::new() };
        let handled = match text(get(ev, "type")) {
            "press" => {
                nav_state.press(&cam, &mut probe, size, button(ev), num(get(ev, "x")), num(get(ev, "y")), truthy(get(ev, "shift")));
                true
            }
            "move" => {
                nav_state.drag(&mut cam, size, num(get(ev, "x")), num(get(ev, "y")), truthy(get(ev, "shift")), true);
                true
            }
            "release" => {
                nav_state.release(button(ev));
                true
            }
            "wheel" => {
                nav_state.wheel(&mut cam, &mut probe, size, num(get(ev, "x")), num(get(ev, "y")), num(get(ev, "delta")));
                true
            }
            _ => false,
        };
        let (pivots, foci) = (probe.pivots, probe.foci);
        if !handled {
            simple_event(ev, &mut cam, &mut nav_state, &mut scene_probe);
        }
        let (got, shown) = cam_bits(&cam);
        let (want, want_shown) = recorded(get(e, "camera_after"));
        if got != want {
            let mut note = String::new();
            for call in arr(get(e, "calls")) {
                let a = get(call, "args");
                if let Some(p) = a.get("pivot") {
                    note += &format!("\n  python pivot {:?}, rust {:?}", v3_of(get(p, "v3")), pivots);
                }
                if let Some(f) = a.get("focus") {
                    note += &format!("\n  python focus {:?}, rust {:?}", v3_of(get(f, "v3")), foci);
                }
            }
            bad.push(format!("{name} event {i} {ev:?}\n  rust   {shown}\n  python {want_shown}{note}"));
            if bad.len() > 3 {
                break;
            }
            let t = v3_of(get(get(e, "camera_after"), "target"));
            cam.target = t;
            cam.distance = f64_of(get(get(e, "camera_after"), "distance"));
            cam.yaw = f64_of(get(get(e, "camera_after"), "yaw"));
            cam.pitch = f64_of(get(get(e, "camera_after"), "pitch"));
        }
    }
    bad
}

#[test]
#[cfg_attr(not(windows), ignore = "fixtures recorded with the MSVC CRT's libm (sin/cos/atan2/pow differ by an ulp elsewhere)")]
fn navigation_maths_match_python() {
    let cases = load_cases();
    let mut bad = Vec::new();
    let mut events = 0;
    for case in &cases {
        events += arr(get(case, "events")).len();
        bad.extend(replay_maths(case));
    }
    assert!(bad.is_empty(), "{} of {events} events differ:\n{}", bad.len(), bad.join("\n"));
    eprintln!("{} cases, {events} events: every camera equal to Python's, bit for bit", cases.len());
}

#[test]
#[cfg_attr(not(windows), ignore = "fixtures recorded with the MSVC CRT's libm (sin/cos/atan2/pow differ by an ulp elsewhere)")]
fn gestures_match_python() {
    let cases = load_cases();
    let mut bad = Vec::new();
    let mut events = 0;
    for case in &cases {
        events += arr(get(case, "events")).len();
        bad.extend(replay_gestures(case));
    }
    assert!(bad.is_empty(), "{} of {events} events differ:\n{}", bad.len(), bad.join("\n"));
    eprintln!("{} cases, {events} events through gestures and probe: every camera equal to Python's", cases.len());
}
