// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! The drawing tools against the Python app
//! (scripts/golden/make_draw_tool_fixtures.py).
//!
//! Each program plays pointer, key and value-box events on a scene through a
//! real camera: the REAL `Viewport` ran them headless with the real tool,
//! the real snap engine and the real history. The Rust host replays the very
//! same events (the pixels as the doubles Python used) and must agree after
//! EVERY event on
//!
//! * the whole ordered state of the active mesh and of every group's
//!   (vertices as float32 bits, edges with flags, curve ids and faces, faces
//!   with loops, holes, interior flag and attrs);
//! * the tool's own state, the preview lines, the floating label and the
//!   measurements box text, the value box's buffer and caption;
//! * the cursor's snap (kind, point, axis, colour), the arrow lock, the Down
//!   reference, the Alt mode, the Shift lock, the encouraged points;
//! * the messages flashed and the undo / redo depths;
//!
//! and after undoing everything.
//!
//! The fixtures come from a Windows machine: `length()` is MSVC's `hypot`.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use ingetrazo_formats::igz;
use ingetrazo_formats::pyjson::{loads, Value};
use ingetrazo_golden::sig::{canon_dumps, sha, Files};
use ingetrazo_math::{Hypot, V3};
use ingetrazo_model::mesh::Mesh;
use ingetrazo_model::scene::{MeshId, Scene};
use ingetrazo_ops::command::Cx;
use ingetrazo_snap::{OrbitCamera, View};
use ingetrazo_tools::base::*;
use ingetrazo_tools::host::Host;

const H: Hypot = Hypot::Msvc;

fn data() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/draw")
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

fn num(v: &Value) -> f64 {
    match v {
        Value::Float(f) => *f,
        Value::Int(i) => i.parse().unwrap(),
        other => panic!("not a number: {other:?}"),
    }
}

fn hex32(x: f32) -> String {
    format!("{:08x}", x.to_bits())
}

fn hex64(x: f64) -> String {
    format!("{:016x}", x.to_bits())
}

/// `pb`: a point as 24 hex digits.
fn pb(p: V3) -> Value {
    Value::Str(format!("{}{}{}", hex32(p.0), hex32(p.1), hex32(p.2)))
}

fn dbits(x: f64) -> Value {
    Value::Str(format!("d{}", hex64(x)))
}

fn from_hex32(s: &str) -> f32 {
    f32::from_bits(u32::from_str_radix(s, 16).unwrap())
}

fn from_pb(s: &str) -> V3 {
    V3(from_hex32(&s[0..8]), from_hex32(&s[8..16]), from_hex32(&s[16..24]))
}

fn from_dbits(v: &Value) -> f64 {
    let s = text(v);
    assert!(s.starts_with('d'));
    f64::from_bits(u64::from_str_radix(&s[1..], 16).unwrap())
}

// ---- state ------------------------------------------------------------------------------------

fn bits32(x: f32) -> String {
    hex32(x)
}

/// `mesh_state` of the script.
fn mesh_state(m: &Mesh) -> Value {
    let vi: HashMap<_, usize> = m.vertices.iter().enumerate().map(|(i, &v)| (v, i)).collect();
    let fi: HashMap<_, usize> = m.faces.iter().enumerate().map(|(i, &f)| (f, i)).collect();
    let ix = |v: &ingetrazo_model::VertexId| Value::int(vi.get(v).map(|&i| i as i64).unwrap_or(-1));
    let int = |i: usize| Value::int(i as i64);
    let s = |x: String| Value::Str(x);
    let files = Files { zip: None };
    let v = m
        .vertices
        .iter()
        .map(|&v| {
            let p = m.position(v);
            s(format!("{}{}{}", bits32(p.0), bits32(p.1), bits32(p.2)))
        })
        .collect();
    let e = m
        .edges
        .iter()
        .map(|&e| {
            let ed = m.edge(e);
            Value::Array(vec![
                ix(&ed.v0),
                ix(&ed.v1),
                int(ed.soft as usize),
                int(ed.hidden as usize),
                ed.layer.clone().map(Value::Str).unwrap_or(Value::Null),
                ed.curve.map(Value::int).unwrap_or(Value::Null),
                Value::Array(ed.faces.iter().map(|f| int(fi[f])).collect()),
            ])
        })
        .collect();
    let f = m
        .faces
        .iter()
        .map(|&f| {
            let fc = m.face(f);
            let attrs = if fc.attrs.is_empty() { String::new() } else { sha(&canon_dumps(&files.canonical(&Value::Object(fc.attrs.clone()))), 16) };
            Value::Array(vec![
                Value::Array(fc.outer.iter().map(ix).collect()),
                Value::Array(fc.holes.iter().map(|h| Value::Array(h.iter().map(ix).collect())).collect()),
                int(fc.interior as usize),
                s(attrs),
            ])
        })
        .collect();
    Value::Object(vec![("v".into(), Value::Array(v)), ("e".into(), Value::Array(e)), ("f".into(), Value::Array(f))])
}

fn scene_state(scene: &Scene, active: MeshId) -> Value {
    Value::Object(vec![
        ("m".into(), mesh_state(scene.mesh(active))),
        ("g".into(), Value::Array(scene.groups.iter().map(|g| mesh_state(scene.mesh(g.mesh))).collect())),
        ("edit".into(), Value::Bool(false)),
    ])
}

fn first_difference(path: &str, a: &Value, b: &Value) -> Option<String> {
    match (a, b) {
        (Value::Array(x), Value::Array(y)) => {
            for (i, (p, q)) in x.iter().zip(y).enumerate() {
                if let Some(d) = first_difference(&format!("{path}[{i}]"), p, q) {
                    return Some(d);
                }
            }
            (x.len() != y.len()).then(|| format!("{path}: {} entries, Python {}", x.len(), y.len()))
        }
        (Value::Object(x), Value::Object(y)) => {
            for (k, p) in x {
                match y.iter().find(|(kk, _)| kk == k) {
                    Some((_, q)) => {
                        if let Some(d) = first_difference(&format!("{path}.{k}"), p, q) {
                            return Some(d);
                        }
                    }
                    None => return Some(format!("{path}.{k} missing in Python")),
                }
            }
            for (k, _) in y {
                if !x.iter().any(|(kk, _)| kk == k) {
                    return Some(format!("{path}.{k} missing in Rust"));
                }
            }
            None
        }
        _ => (a != b).then(|| format!("{path}: {a:?} vs Python {b:?}")),
    }
}

/// A [`Field`] as the script's `enc` writes the Python value.
fn field_value(f: &Field) -> Value {
    match f {
        Field::Null => Value::Null,
        Field::Bool(b) => Value::Bool(*b),
        Field::Int(i) => Value::int(*i),
        Field::Float(x) => dbits(*x),
        Field::Str(s) => Value::Str(s.clone()),
        Field::P(p) => pb(*p),
        Field::Plane(a, b) => Value::Array(vec![pb(*a), pb(*b)]),
        Field::Color(c) => Value::Array(c.iter().map(|x| dbits(*x)).collect()),
        Field::List(l) => Value::Array(l.iter().map(field_value).collect()),
    }
}

fn opt_str(s: Option<&str>) -> Value {
    s.map(|s| Value::Str(s.to_string())).unwrap_or(Value::Null)
}

fn view_of(c: &Value) -> View {
    let g = |k: &str| c.get(k).unwrap();
    let cam = OrbitCamera {
        target: from_pb(text(g("target"))),
        distance: from_dbits(g("distance")),
        yaw: from_dbits(g("yaw")),
        pitch: from_dbits(g("pitch")),
        up: from_pb(text(g("up"))),
        fov_deg: from_dbits(g("fov")),
        aspect: from_dbits(g("aspect")),
        znear: from_dbits(g("znear")),
        zfar: from_dbits(g("zfar")),
        perspective: g("persp") == &Value::Bool(true),
    };
    View::new(cam, g("w").as_i64().unwrap() as i32, g("h").as_i64().unwrap() as i32, H)
}

/// Play the events of one program; returns the number of events.
fn play(label: &str, case: &Value, states: &[Value], make: &dyn Fn() -> Box<dyn DrawTool>, tally: &mut Tally) {
    let scene = open(text(case.get("scene").unwrap()));
    let mut cx = Cx { hypot: H, mesh: MeshId(0), ..Cx::default() };
    cx.counters.curve = 5000;
    let mut host = Host::new(scene, cx, view_of(case.get("cam").unwrap()));
    host.set_active_tool(make());
    let pre = &states[case.get("pre").unwrap().as_i64().unwrap() as usize];
    if let Some(d) = first_difference("pre", &scene_state(&host.core.scene, host.core.cx.mesh), pre) {
        panic!("{label}: reopened scene differs: {d}");
    }
    for (k, ev) in arr(case.get("events").unwrap()).iter().enumerate() {
        let kind = text(ev.get("e").unwrap());
        let elabel = format!("{label}, event {k} ({kind})");
        host.core.flashes.clear();
        let mods = ev.get("m").map(|m| m.as_i64().unwrap() as u32).unwrap_or(0);
        let mut ret: Option<bool> = None;
        match kind {
            "hover" => host.hover(num(ev.get("x").unwrap()), num(ev.get("y").unwrap()), mods),
            "press" => host.press(num(ev.get("x").unwrap()), num(ev.get("y").unwrap()), mods),
            "double" => host.double_click(num(ev.get("x").unwrap()), num(ev.get("y").unwrap()), mods),
            "release" => host.release(),
            "keydown" => {
                host.key_press(ev.get("k").unwrap().as_i64().unwrap() as i32, mods, "");
            }
            "keyup" => host.key_release(ev.get("k").unwrap().as_i64().unwrap() as i32),
            "type" => {
                for ch in text(ev.get("t").unwrap()).chars() {
                    let code = ch.to_ascii_uppercase() as i32;
                    host.key_press(code, mods, &ch.to_string());
                }
            }
            "undo" => ret = Some(host.undo()),
            "redo" => ret = Some(host.redo()),
            "dwell" => host.dwell(),
            "reselect" => host.reselect_tool(),
            other => panic!("{elabel}: unknown event {other}"),
        }
        if let Some(r) = ret {
            assert_eq!(Value::Bool(r), *ev.get("ret").unwrap(), "{elabel}: return");
        }
        compare(&elabel, &host, ev, states);
        tally.events += 1;
        if kind == "press" || kind == "double" {
            tally.clicks += 1;
        }
    }
    // Undone: every step taken back.
    while host.core.history.undo_len() > 0 {
        host.core.history_undo();
    }
    if let Some(u) = case.get("undone").map(arr).and_then(|u| u.last()) {
        let want = &states[u.as_i64().unwrap() as usize];
        if let Some(d) = first_difference("undone", &scene_state(&host.core.scene, host.core.cx.mesh), want) {
            panic!("{label}: after undoing everything: {d}");
        }
    }
    tally.programs += 1;
}

fn compare(elabel: &str, host: &Host, ev: &Value, states: &[Value]) {
    let core = &host.core;
    let want = &states[ev.get("st").unwrap().as_i64().unwrap() as usize];
    if let Some(d) = first_difference("state", &scene_state(&core.scene, core.cx.mesh), want) {
        panic!("{elabel}: {d}");
    }
    // The plane the cursor was read on.
    let lwp = field_value(&Field::plane(core.last_work_plane));
    if let Some(d) = first_difference("last work plane", &lwp, ev.get("lwp").unwrap()) {
        panic!("{elabel}: {d}");
    }
    let tool = host.tool();
    // The cursor's snap.
    let snap = match &core.last_snap {
        None => Value::Null,
        Some(s) => Value::Array(vec![
            Value::Str(s.kind.as_str().into()),
            pb(s.point),
            opt_str(s.axis.map(|a| a.name())),
            Value::Array(s.color.iter().map(|c| dbits(*c)).collect()),
        ]),
    };
    if let Some(d) = first_difference("snap", &snap, ev.get("snap").unwrap()) {
        panic!("{elabel}: {d}");
    }
    // The preview.
    let band = Value::Array(tool.rubber_band_lines(core).iter().map(|(a, b)| Value::Array(vec![pb(*a), pb(*b)])).collect());
    if let Some(d) = first_difference("band", &band, ev.get("band").unwrap()) {
        panic!("{elabel}: {d}");
    }
    let label = match tool.value_label(core) {
        None => Value::Null,
        Some((t, p)) => Value::Array(vec![Value::Str(t), pb(p)]),
    };
    if let Some(d) = first_difference("label", &label, ev.get("label").unwrap()) {
        panic!("{elabel}: {d}");
    }
    assert_eq!(Value::Str(host.measurement_text()), *ev.get("meas").unwrap(), "{elabel}: measurements box");
    assert_eq!(Value::Str(core.value_buffer.clone()), *ev.get("vcb").unwrap(), "{elabel}: value buffer");
    let cap = tool.vcb_label();
    assert_eq!(if cap.is_empty() { Value::Null } else { Value::Str(cap) }, *ev.get("cap").unwrap(), "{elabel}: caption");
    let flash = Value::Array(core.flashes.iter().map(|s| Value::Str(s.clone())).collect());
    if let Some(d) = first_difference("flash", &flash, ev.get("flash").unwrap()) {
        panic!("{elabel}: {d}");
    }
    assert_eq!(opt_str(core.inference.axis_lock.map(|a| a.name())), *ev.get("lock").unwrap(), "{elabel}: arrow lock");
    let inf = &core.inference;
    let rm = opt_str(inf.reference_mode.map(|m| match m {
        ingetrazo_snap::RefMode::Parallel => "parallel",
        ingetrazo_snap::RefMode::Perpendicular => "perpendicular",
    }));
    let lm = Value::Str(
        match inf.linear_mode {
            None | Some(ingetrazo_snap::LinearMode::All) => "all",
            Some(ingetrazo_snap::LinearMode::Off) => "off",
            Some(ingetrazo_snap::LinearMode::ParallelPerp) => "parallel_perp",
        }
        .into(),
    );
    let sl = match inf.shift_lock {
        None => Value::Null,
        Some((d, c)) => Value::Array(vec![pb(d), Value::Array(c.iter().map(|x| dbits(*x)).collect())]),
    };
    let got_inf = Value::Object(vec![
        ("rm".into(), rm),
        ("lm".into(), lm),
        ("sl".into(), sl),
        ("ap".into(), inf.acquired_point.map(pb).unwrap_or(Value::Null)),
        ("enc".into(), Value::Array(inf.encouraged.iter().map(|p| pb(*p)).collect())),
    ]);
    if let Some(d) = first_difference("inference", &got_inf, ev.get("inf").unwrap()) {
        panic!("{elabel}: {d}");
    }
    // The tool's own state.
    let tf = tool.fields();
    let got_tf = Value::Object(tf.iter().map(|(k, f)| (k.to_string(), field_value(f))).collect());
    if let Some(d) = first_difference("tool", &got_tf, ev.get("tf").unwrap()) {
        panic!("{elabel}: {d}");
    }
    assert_eq!(core.history.undo_len() as i64, ev.get("depth").unwrap().as_i64().unwrap(), "{elabel}: undo depth");
    assert_eq!(core.history.redo_len() as i64, ev.get("redo").unwrap().as_i64().unwrap(), "{elabel}: redo depth");
    assert_eq!(opt_str(core.history.last_error.as_deref()), *ev.get("err").unwrap(), "{elabel}: history error");
}

#[derive(Default)]
struct Tally {
    programs: usize,
    events: usize,
    clicks: usize,
}

fn run(key: &str, make: &dyn Fn() -> Box<dyn DrawTool>) {
    let raw = std::fs::read_to_string(data().join(format!("cases_{key}.json"))).unwrap();
    let file = loads(&raw).unwrap();
    let states = arr(file.get("states").unwrap());
    let mut tally = Tally::default();
    for (n, case) in arr(file.get("cases").unwrap()).iter().enumerate() {
        let label = format!("{key} program {n} on {}", text(case.get("scene").unwrap()));
        play(&label, case, states, make, &mut tally);
    }
    eprintln!("{key}: {} programs, {} events, {} clicks", tally.programs, tally.events, tally.clicks);
    assert!(tally.programs >= 20 || key == "line" && tally.programs >= 10, "{key}: too few programs");
}

#[test]
fn line_programs_match_python() {
    run("line", &|| Box::new(ingetrazo_tools::line::LineTool::new()));
}

#[test]
fn rectangle_programs_match_python() {
    run("rectangle", &|| Box::new(ingetrazo_tools::rectangle::RectangleTool::new()));
}

#[test]
#[cfg_attr(not(windows), ignore = "fixtures recorded with the MSVC CRT's libm (sin/cos/atan2/pow differ by an ulp elsewhere)")]
fn rotated_rectangle_programs_match_python() {
    run("rotated_rectangle", &|| Box::new(ingetrazo_tools::rotated_rectangle::RotatedRectangleTool::new()));
}

#[test]
fn circle_programs_match_python() {
    run("circle", &|| Box::new(ingetrazo_tools::circle::RadialTool::circle()));
}

#[test]
fn polygon_programs_match_python() {
    run("polygon", &|| Box::new(ingetrazo_tools::circle::RadialTool::polygon()));
}

#[test]
#[cfg_attr(not(windows), ignore = "fixtures recorded with the MSVC CRT's libm (sin/cos/atan2/pow differ by an ulp elsewhere)")]
fn arc_programs_match_python() {
    run("arc", &|| Box::new(ingetrazo_tools::arc::ArcTool::new()));
}

#[test]
fn three_point_arc_programs_match_python() {
    run("arc3", &|| Box::new(ingetrazo_tools::arc::ThreePointArcTool::new()));
}

#[test]
fn center_arc_programs_match_python() {
    run("center_arc", &|| Box::new(ingetrazo_tools::arc::CenterArcTool::center_arc()));
}

#[test]
fn pie_programs_match_python() {
    run("pie", &|| Box::new(ingetrazo_tools::arc::CenterArcTool::pie()));
}

#[test]
fn freehand_programs_match_python() {
    run("freehand", &|| Box::new(ingetrazo_tools::freehand::FreehandTool::new()));
}
