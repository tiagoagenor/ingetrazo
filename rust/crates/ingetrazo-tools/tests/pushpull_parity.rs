// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! Push/Pull against the Python app (scripts/golden/make_pushpull_fixtures.py).
//!
//! Each program drives the REAL `tools.pushpull.PushPullTool` on a stub
//! viewport: boxes from rectangles, recesses, through-holes, bumps, pushes
//! to zero, copy pushes (Ctrl), faces with holes, curved faces, slanted
//! faces, pulls against a wall, flat sheets, faces inside an open group, the
//! repeat of the last distance, the hot retype, undo and redo. The Rust tool
//! must leave the whole ordered state of the active mesh (and of every
//! group's) the very same after every step, after undoing everything and
//! after redoing everything - vertices as float32 bits, edges with flags,
//! curve ids and faces, faces with loops, holes, interior flag and attrs -
//! with the status it flashed and the last distance it kept.
//!
//! The fixtures come from a Windows machine: `length()` is MSVC's `hypot`.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use ingetrazo_formats::igz;
use ingetrazo_formats::pyjson::{loads, Value};
use ingetrazo_golden::sig::{canon_dumps, sha, Files};
use ingetrazo_math::Hypot;
use ingetrazo_model::mesh::Mesh;
use ingetrazo_model::scene::{MeshId, Scene};
use ingetrazo_model::units::ModelUnits;
use ingetrazo_ops::command::{Cx, History};
use ingetrazo_ops::group::{push_mesh, transformed_mesh};
use ingetrazo_tools::pick::{FixedPick, Pick, Picker, RayPicker};
use ingetrazo_tools::view::{Camera, View};
use ingetrazo_tools::pushpull::{Env, HoverEdge, Pointer, PushPullTool};

const H: Hypot = Hypot::Msvc;

fn data() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/pushpull")
}

fn open(name: &str) -> Scene {
    let raw = std::fs::read(data().join(format!("{name}.igz.json"))).unwrap();
    let doc = igz::read_document(&raw).unwrap();
    Scene::from_document(&doc).unwrap_or_else(|e| panic!("{name}: {e:?}"))
}

fn bits32(x: f32) -> String {
    format!("{:08x}", x.to_bits())
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

/// `mesh_state` of the script: the whole ordered state of a mesh.
fn mesh_state(m: &Mesh) -> Value {
    let vi: HashMap<_, usize> = m.vertices.iter().enumerate().map(|(i, &v)| (v, i)).collect();
    let fi: HashMap<_, usize> = m.faces.iter().enumerate().map(|(i, &f)| (f, i)).collect();
    let int = |i: usize| Value::int(i as i64);
    let s = |x: String| Value::Str(x);
    let files = Files { zip: None };
    let v = m
        .vertices
        .iter()
        .map(|&v| {
            let p = m.position(v);
            Value::Array(vec![s(bits32(p.0)), s(bits32(p.1)), s(bits32(p.2))])
        })
        .collect();
    let e = m
        .edges
        .iter()
        .map(|&e| {
            let ed = m.edge(e);
            Value::Array(vec![
                int(vi[&ed.v0]),
                int(vi[&ed.v1]),
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
                Value::Array(fc.outer.iter().map(|v| int(vi[v])).collect()),
                Value::Array(fc.holes.iter().map(|h| Value::Array(h.iter().map(|v| int(vi[v])).collect())).collect()),
                int(fc.interior as usize),
                s(attrs),
            ])
        })
        .collect();
    Value::Object(vec![("v".into(), Value::Array(v)), ("e".into(), Value::Array(e)), ("f".into(), Value::Array(f))])
}

/// `snap` of the script: the active mesh, every group's mesh, whether a
/// group is open.
fn snap(scene: &Scene, active: MeshId, edit: bool) -> Value {
    Value::Object(vec![
        ("m".into(), mesh_state(scene.mesh(active))),
        ("g".into(), Value::Array(scene.groups.iter().map(|g| mesh_state(scene.mesh(g.mesh))).collect())),
        ("edit".into(), Value::Bool(edit)),
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
            None
        }
        _ => (a != b).then(|| format!("{path}: {a:?} vs Python {b:?}")),
    }
}

/// A host that types distances: no camera, the picker answers one face.
struct Host {
    scene: Scene,
    cx: Cx,
    history: History,
    tool: PushPullTool,
    units: ModelUnits,
    edit: bool,
}

impl Host {
    fn step_env<'a>(&'a mut self, picker: &'a dyn Picker) -> (Env<'a>, &'a mut PushPullTool) {
        let env = Env {
            scene: &mut self.scene,
            cx: &mut self.cx,
            history: &mut self.history,
            view: None,
            picker,
            hover_edge: None,
            snap_threshold_px: 9.0,
            hidden_reference: None,
            units: &self.units,
        };
        (env, &mut self.tool)
    }
}

fn ptr(copy: bool) -> Pointer {
    Pointer { x: 0.0, y: 0.0, ctrl: copy }
}

fn num(v: &Value) -> f64 {
    match v {
        Value::Float(f) => *f,
        Value::Int(i) => i.parse().unwrap(),
        other => panic!("not a number: {other:?}"),
    }
}

#[test]
fn pushpull_programs_match_python() {
    let cases = loads(&std::fs::read_to_string(data().join("cases.json")).unwrap()).unwrap();
    let (mut programs, mut steps_run, mut pushes, mut undone, mut flashed) = (0, 0, 0, 0, 0);
    for (n, case) in arr(&cases).iter().enumerate() {
        let name = text(case.get("scene").unwrap());
        let label = format!("program {n} on {name}");
        let scene = open(name);
        let mut host = Host { scene, cx: Cx { hypot: H, mesh: MeshId(0), ..Cx::default() }, history: History::new(), tool: PushPullTool::new(), units: ModelUnits::default(), edit: false };
        host.cx.counters.curve = 5000;
        if let Some(d) = first_difference("pre", &snap(&host.scene, host.cx.mesh, host.edit), case.get("pre").unwrap()) {
            panic!("{label}: reopened scene differs: {d}");
        }
        for (k, step) in arr(case.get("steps").unwrap()).iter().enumerate() {
            let kind = text(step.get("kind").unwrap());
            let slabel = format!("{label}, step {k} ({kind})");
            host.tool.flashes.clear();
            let face_of = |h: &Host| h.scene.mesh(h.cx.mesh).faces[step.get("face").unwrap().as_i64().unwrap() as usize];
            let copy = step.get("copy").map(|c| c == &Value::Bool(true)).unwrap_or(false);
            match kind {
                "value" => {
                    let face = face_of(&host);
                    let d = num(step.get("d").unwrap());
                    let pick = FixedPick(None);
                    let (mut env, tool) = host.step_env(&pick);
                    tool.hovered_face = Some(face);
                    tool.hover_closed_group = false;
                    tool.on_click(&mut env, ptr(copy));
                    assert_eq!(tool.dragging, step.get("dragging") == Some(&Value::Bool(true)), "{slabel}: dragging");
                    let ret = tool.on_value(&mut env, d);
                    assert_eq!(ret, step.get("ret") == Some(&Value::Bool(true)), "{slabel}: on_value");
                    pushes += 1;
                }
                "click2" => {
                    let face = face_of(&host);
                    let d = num(step.get("d").unwrap());
                    let pick = FixedPick(None);
                    let (mut env, tool) = host.step_env(&pick);
                    tool.hovered_face = Some(face);
                    tool.hover_closed_group = false;
                    tool.on_click(&mut env, ptr(copy));
                    tool.extrusion = d;
                    tool.on_click(&mut env, ptr(copy));
                    pushes += 1;
                }
                "double" => {
                    let face = face_of(&host);
                    let pick = FixedPick(Some(Pick { face, closed_group: false }));
                    let (mut env, tool) = host.step_env(&pick);
                    tool.on_double_click(&mut env, ptr(copy));
                    pushes += 1;
                }
                "op" => {
                    let face = face_of(&host);
                    let d = num(step.get("d").unwrap());
                    let pick = FixedPick(None);
                    let (mut env, tool) = host.step_env(&pick);
                    let ret = tool.push_direct(&mut env, face, d, copy);
                    assert_eq!(ret, step.get("ret") == Some(&Value::Bool(true)), "{slabel}: push_direct");
                    pushes += 1;
                }
                "closed" => {
                    // A face inside a group that is not open: not shaded, refused.
                    let face = face_of(&host);
                    let pick = FixedPick(Some(Pick { face, closed_group: true }));
                    let (mut env, tool) = host.step_env(&pick);
                    tool.on_hover(&mut env, ptr(false));
                    assert_eq!(tool.hover.is_none(), step.get("hover0") == Some(&Value::Null), "{slabel}: shaded");
                    tool.on_click(&mut env, ptr(false));
                    let saved = tool.last_distance;
                    tool.last_distance = Some(1.0);
                    tool.on_double_click(&mut env, ptr(false));
                    tool.last_distance = saved;
                }
                "tiny" => {
                    let face = face_of(&host);
                    let pick = FixedPick(None);
                    let (mut env, tool) = host.step_env(&pick);
                    tool.hovered_face = Some(face);
                    tool.hover_closed_group = false;
                    tool.on_click(&mut env, ptr(false));
                    tool.on_click(&mut env, ptr(false)); // nothing dragged: the hint
                    tool.on_click(&mut env, ptr(false)); // said once per drag
                    tool.on_release(&mut env, false); // no press far enough away
                    let ret = tool.on_value(&mut env, 0.0);
                    assert_eq!(ret, step.get("ret") == Some(&Value::Bool(true)), "{slabel}: on_value(0)");
                }
                "undo_mid" => {
                    let face = face_of(&host);
                    let pick = FixedPick(None);
                    let (mut env, tool) = host.step_env(&pick);
                    tool.hovered_face = Some(face);
                    tool.hover_closed_group = false;
                    tool.on_click(&mut env, ptr(false));
                    assert_eq!(tool.on_undo(&mut env), step.get("ret") == Some(&Value::Bool(true)), "{slabel}: on_undo");
                    assert_eq!(tool.on_redo(&mut env), step.get("ret2") == Some(&Value::Bool(true)), "{slabel}: on_redo");
                }
                "esc" => {
                    let face = face_of(&host);
                    let d = num(step.get("d").unwrap());
                    let pick = FixedPick(None);
                    let (mut env, tool) = host.step_env(&pick);
                    tool.hovered_face = Some(face);
                    tool.hover_closed_group = false;
                    tool.on_click(&mut env, ptr(copy));
                    tool.extrusion = d;
                    assert_eq!(tool.dragging, step.get("drag_mid") == Some(&Value::Bool(true)), "{slabel}: dragging");
                    tool.on_cancel(&mut env);
                }
                "activate" => {
                    // Activated with exactly one face selected: the tool arms on it.
                    let face = face_of(&host);
                    let d = num(step.get("d").unwrap());
                    let pick = FixedPick(None);
                    let (mut env, tool) = host.step_env(&pick);
                    tool.on_activate(&mut env, Some(face));
                    assert_eq!(tool.hover == Some(face), step.get("armed") == Some(&Value::Bool(true)), "{slabel}: armed");
                    tool.on_hover(&mut env, ptr(false)); // the pick (nothing) does not steal it
                    assert_eq!(tool.hover == Some(face), step.get("still_armed") == Some(&Value::Bool(true)), "{slabel}: still armed");
                    tool.on_click(&mut env, ptr(copy));
                    assert_eq!(tool.dragging, step.get("armed_drag") == Some(&Value::Bool(true)), "{slabel}: armed drag");
                    let ret = tool.on_value(&mut env, d);
                    assert_eq!(ret, step.get("ret") == Some(&Value::Bool(true)), "{slabel}: on_value");
                }
                "ctrl_tap" => {
                    assert_eq!(host.tool.on_key_ctrl_press(), step.get("pressed") == Some(&Value::Bool(true)), "{slabel}: press");
                    assert_eq!(host.tool.on_key_ctrl_release(true), step.get("ret") == Some(&Value::Bool(true)), "{slabel}: release");
                }
                "retype" => {
                    let d = num(step.get("d").unwrap());
                    let pick = FixedPick(None);
                    let (mut env, tool) = host.step_env(&pick);
                    let ret = tool.on_value(&mut env, d);
                    assert_eq!(ret, step.get("ret") == Some(&Value::Bool(true)), "{slabel}: retype");
                }
                "undo" => {
                    let ret = host.history.undo(&mut host.scene, &mut host.cx);
                    assert_eq!(ret, step.get("ret") == Some(&Value::Bool(true)), "{slabel}: undo");
                }
                "redo" => {
                    let ret = host.history.redo(&mut host.scene, &mut host.cx);
                    assert_eq!(ret, step.get("ret") == Some(&Value::Bool(true)), "{slabel}: redo");
                }
                "edit" => {
                    let g = step.get("group").unwrap().as_i64().unwrap() as usize;
                    if let Some(mut x) = host.scene.groups[g].xform {
                        // A component instance: the session edits a world-space
                        // COPY of the shared definition (begin_group_edit).
                        let proto = host.scene.groups[g].mesh;
                        let copy = transformed_mesh(host.scene.mesh(proto), &mut x, &mut host.cx.counters.curve);
                        let id = push_mesh(&mut host.scene.meshes, copy);
                        host.scene.groups[g].mesh = id;
                        host.scene.groups[g].xform = None;
                    }
                    host.cx.mesh = host.scene.groups[g].mesh;
                    host.edit = true;
                }
                "end_edit" => {
                    host.cx.mesh = MeshId(0);
                    host.edit = false;
                }
                other => panic!("{slabel}: kind {other}"),
            }
            if let Some(want) = step.get("drag_end") {
                assert_eq!(host.tool.dragging, want == &Value::Bool(true), "{slabel}: still dragging");
            }
            if host.tool.dragging {
                let pick = FixedPick(None);
                let (mut env, tool) = host.step_env(&pick);
                tool.on_cancel(&mut env);
            }
            assert_eq!(host.tool.keep_mode(), step.get("keep_mode") == Some(&Value::Bool(true)), "{slabel}: keep mode");
            assert_eq!(host.tool.hover.and_then(|f| face_index(&host, f)), step.get("hover").unwrap().as_i64(), "{slabel}: hover");
            if let Some(d) = first_difference("state", &snap(&host.scene, host.cx.mesh, host.edit), step.get("state").unwrap()) {
                panic!("{slabel}: state differs: {d}");
            }
            let want_status = match step.get("status").unwrap() {
                Value::Null => None,
                Value::Str(s) => Some(s.as_str()),
                other => panic!("status {other:?}"),
            };
            assert_eq!(host.tool.last_flash(), want_status, "{slabel}: status");
            flashed += want_status.is_some() as usize;
            let want_last = match step.get("last_distance").unwrap() {
                Value::Null => None,
                v => Some(num(v)),
            };
            assert_eq!(host.tool.last_distance.map(f64::to_bits), want_last.map(f64::to_bits), "{slabel}: last_distance");
            assert_eq!(host.history.undo_len() as i64, step.get("undo_depth").unwrap().as_i64().unwrap(), "{slabel}: undo depth");
            steps_run += 1;
        }
        for (k, want) in arr(case.get("undone").unwrap()).iter().enumerate() {
            assert!(host.history.undo(&mut host.scene, &mut host.cx), "{label}: undo {k}");
            // An open group stays open while its commands are undone; the
            // script records the same instant.
            if let Some(d) = first_difference("undone", &snap(&host.scene, host.cx.mesh, host.edit), want) {
                panic!("{label}: after undo {k} differs: {d}");
            }
            undone += 1;
        }
        for (k, want) in arr(case.get("redone").unwrap()).iter().enumerate() {
            assert!(host.history.redo(&mut host.scene, &mut host.cx), "{label}: redo {k}");
            if let Some(d) = first_difference("redone", &snap(&host.scene, host.cx.mesh, host.edit), want) {
                panic!("{label}: after redo {k} differs: {d}");
            }
        }
        programs += 1;
    }
    eprintln!("{programs} programs, {steps_run} steps ({pushes} pushes, {flashed} flashed), {undone} undos");
    assert!(programs >= 90 && steps_run >= 200, "{programs} programs, {steps_run} steps");
}

// ---- the pointer, through a real camera ------------------------------------------

fn pbits_v3(p: ingetrazo_math::V3) -> Value {
    Value::Array(vec![Value::Str(bits32(p.0)), Value::Str(bits32(p.1)), Value::Str(bits32(p.2))])
}

fn v3_from_bits(v: &Value) -> ingetrazo_math::V3 {
    let a = arr(v);
    let f = |s: &Value| f32::from_bits(u32::from_str_radix(text(s), 16).unwrap());
    ingetrazo_math::V3(f(&a[0]), f(&a[1]), f(&a[2]))
}

fn opt_str(v: &Value) -> Option<&str> {
    match v {
        Value::Null => None,
        Value::Str(s) => Some(s),
        other => panic!("not a string or null: {other:?}"),
    }
}

fn face_index(host: &Host, f: ingetrazo_model::mesh::FaceId) -> Option<i64> {
    host.scene.mesh(host.cx.mesh).faces.iter().position(|&x| x == f).map(|i| i as i64)
}

#[test]
fn pushpull_pointer_matches_python() {
    let cases = loads(&std::fs::read_to_string(data().join("pointer.json")).unwrap()).unwrap();
    let (mut events, mut clicks, mut inferred) = (0, 0, 0);
    for (n, case) in arr(&cases).iter().enumerate() {
        let name = text(case.get("scene").unwrap());
        let label = format!("pointer program {n} on {name}");
        let spec = case.get("cam").unwrap();
        let g = |k: &str| num(spec.get(k).unwrap());
        let t = arr(spec.get("target").unwrap());
        let camera = Camera {
            target: ingetrazo_math::V3(num(&t[0]) as f32, num(&t[1]) as f32, num(&t[2]) as f32),
            distance: g("distance"),
            yaw: g("yaw").to_radians(),
            pitch: g("pitch").to_radians(),
            aspect: g("w") / g("h"),
            perspective: spec.get("perspective").is_none_or(|v| v == &Value::Bool(true)),
            ..Camera::default()
        };
        let view = View::new(camera, g("w") as u32, g("h") as u32, H);
        let scene = open(name);
        let mut host = Host { scene, cx: Cx { hypot: H, mesh: MeshId(0), ..Cx::default() }, history: History::new(), tool: PushPullTool::new(), units: ModelUnits::default(), edit: false };
        host.cx.counters.curve = 5000;
        if let Some(d) = first_difference("pre", &snap(&host.scene, host.cx.mesh, host.edit), case.get("pre").unwrap()) {
            panic!("{label}: reopened scene differs: {d}");
        }
        for (k, ev) in arr(case.get("events").unwrap()).iter().enumerate() {
            let kind = text(ev.get("e").unwrap());
            let elabel = format!("{label}, event {k} ({kind})");
            host.tool.flashes.clear();
            let p = Pointer { x: ev.get("x").map_or(0.0, num), y: ev.get("y").map_or(0.0, num), ctrl: ev.get("ctrl") == Some(&Value::Bool(true)) };
            let hover_edge = ev.get("edge").map(|e| HoverEdge { a: v3_from_bits(&arr(e)[0]), b: v3_from_bits(&arr(e)[1]), edge: None });
            let picker = RayPicker;
            match kind {
                "undo" => {
                    host.history.undo(&mut host.scene, &mut host.cx);
                }
                "redo" => {
                    host.history.redo(&mut host.scene, &mut host.cx);
                }
                _ => {
                    let mut env = Env {
                        scene: &mut host.scene,
                        cx: &mut host.cx,
                        history: &mut host.history,
                        view: Some(&view),
                        picker: &picker,
                        hover_edge,
                        snap_threshold_px: 9.0,
                        hidden_reference: None,
                        units: &host.units,
                    };
                    let tool = &mut host.tool;
                    match kind {
                        "hover" => tool.on_hover(&mut env, p),
                        "click" => {
                            tool.on_click(&mut env, p);
                            clicks += 1;
                        }
                        "double" => tool.on_double_click(&mut env, p),
                        "release" => tool.on_release(&mut env, false),
                        "value" => {
                            let ret = tool.on_value(&mut env, num(ev.get("d").unwrap()));
                            assert_eq!(ret, ev.get("ret") == Some(&Value::Bool(true)), "{elabel}: on_value");
                        }
                        "cancel" => tool.on_cancel(&mut env),
                        other => panic!("{elabel}: event {other}"),
                    }
                }
            }
            let tool = &host.tool;
            let want_ext = num(ev.get("ext").unwrap());
            assert_eq!(tool.extrusion.to_bits(), want_ext.to_bits(), "{elabel}: extrusion {} vs Python {want_ext}", tool.extrusion);
            // The inference the drag engaged (Python keeps it set while idle too).
            let want_kind = opt_str(ev.get("ik").unwrap());
            if tool.dragging {
                assert_eq!(tool.inference_marker().map(|m| m.1), want_kind, "{elabel}: inference kind");
                if let Some((p, _)) = tool.inference_marker() {
                    assert_eq!(pbits_v3(p), *ev.get("ip").unwrap(), "{elabel}: inference point");
                }
            }
            inferred += (tool.dragging && want_kind.is_some()) as usize;
            assert_eq!(tool.hover.and_then(|f| face_index(&host, f)), ev.get("hover").unwrap().as_i64(), "{elabel}: hover");
            let mut supp: Vec<i64> = tool.suppressed.iter().filter_map(|&f| face_index(&host, f)).collect();
            supp.sort();
            let want_supp: Vec<i64> = match ev.get("supp").unwrap() {
                Value::Null => Vec::new(),
                v => arr(v).iter().map(|i| i.as_i64().unwrap()).collect(),
            };
            assert_eq!(supp, want_supp, "{elabel}: suppressed faces");
            // The overlay sweep and its wireframe.
            let files = Files { zip: None };
            let light: Vec<Value> = tool
                .preview_faces()
                .iter()
                .map(|f| {
                    let attrs = match &f.attrs {
                        Some(a) if !a.is_empty() => sha(&canon_dumps(&files.canonical(&Value::Object(a.clone()))), 16),
                        _ => String::new(),
                    };
                    Value::Array(vec![
                        Value::Array(f.vertices.iter().map(|&p| pbits_v3(p)).collect()),
                        Value::Array(f.holes.iter().map(|hl| Value::Array(hl.iter().map(|&p| pbits_v3(p)).collect())).collect()),
                        Value::Str(attrs),
                    ])
                })
                .collect();
            assert_eq!(Value::Array(light), *ev.get("light").unwrap(), "{elabel}: overlay faces");
            let band = Value::Array(tool.rubber_band_lines(H).iter().map(|&(a, b)| Value::Array(vec![pbits_v3(a), pbits_v3(b)])).collect());
            assert_eq!(band, *ev.get("band").unwrap(), "{elabel}: rubber band");
            let label_v = match tool.value_label(&host.units) {
                None => Value::Null,
                Some((t, mid)) => Value::Array(vec![Value::Str(t), pbits_v3(mid)]),
            };
            assert_eq!(label_v, *ev.get("label").unwrap(), "{elabel}: value label");
            assert_eq!(tool.dragging, ev.get("drag") == Some(&Value::Bool(true)), "{elabel}: dragging");
            assert_eq!(tool.last_flash(), opt_str(ev.get("status").unwrap()), "{elabel}: status");
            let want_last = match ev.get("last_distance").unwrap() {
                Value::Null => None,
                v => Some(num(v)),
            };
            assert_eq!(tool.last_distance.map(f64::to_bits), want_last.map(f64::to_bits), "{elabel}: last_distance");
            assert_eq!(host.history.undo_len() as i64, ev.get("depth").unwrap().as_i64().unwrap(), "{elabel}: undo depth");
            if let Some(d) = first_difference("state", &snap(&host.scene, host.cx.mesh, host.edit), ev.get("state").unwrap()) {
                panic!("{elabel}: state differs: {d}");
            }
            events += 1;
        }
    }
    eprintln!("{events} pointer events, {clicks} clicks, {inferred} dragging with an inference");
    assert!(events >= 150, "{events} events");
}
