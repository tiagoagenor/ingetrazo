// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! The replay shared by the parity tests of the MODIFY and MEASURE tools
//! (`modify_*_parity.rs`; fixtures from `scripts/golden/make_modify_tool_fixtures.py`).
//!
//! Python drove the REAL tool through the REAL `Viewport` (real mouse and key
//! events through a real camera, the VCB typed for real) and recorded, call by
//! call, what the tool was handed (`calls`) and, after every event, the whole
//! state. Here the same calls are made on the Rust tool, with the picks and
//! the camera of `ingetrazo-snap` standing in for the viewport's, and the
//! state after every event must be the very same: the ordered meshes (vertices
//! as float32 bits, edges with flags/curve/faces, faces with loops, holes,
//! interior flag and attrs hash), every group's transform, the guides, the
//! selection, the undo depth, the tool's probe (rubber band, guide preview,
//! label, status clause...), the statuses it flashed and the viewport hooks it
//! drove.
//!
//! The fixtures come from a Windows machine: `length()` is MSVC's `hypot`.

#![allow(dead_code)]

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use ingetrazo_formats::igz;
use ingetrazo_formats::pyjson::{loads, Value};
use ingetrazo_geom::hlr_draw::OrbitCamera;
use ingetrazo_golden::sig::{canon_dumps, sha, Files};
use ingetrazo_math::{Hypot, M4, V3};
use ingetrazo_model::mesh::Mesh;
use ingetrazo_model::scene::{Group, MeshId, Scene};
use ingetrazo_model::units::ModelUnits;
use ingetrazo_ops::command::{Cx, History};
use ingetrazo_snap::picking::{Camera, View};
use ingetrazo_snap::{Axis, Context, SnapKind};
use ingetrazo_tools::modify_base::{Env, Gfx, ModifyTool, Mods, Probe, Sel, Selection, SnapInfo, ToolCx, Vcb};
use ingetrazo_tools::modify_viewport::PickViewport;

pub const H: Hypot = Hypot::Msvc;

pub fn data(tool: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/modify").join(tool)
}

pub fn open(tool: &str, name: &str) -> Scene {
    let raw = std::fs::read(data(tool).join(format!("{name}.igz.json"))).unwrap();
    let doc = igz::read_document(&raw).unwrap();
    Scene::from_document(&doc).unwrap_or_else(|e| panic!("{name}: {e:?}"))
}

// ---- JSON helpers -------------------------------------------------------------------------

pub fn bits32(x: f32) -> String {
    format!("{:08x}", x.to_bits())
}

pub fn unbits(s: &Value) -> f32 {
    let t = text(s);
    f32::from_bits(u32::from_str_radix(t, 16).unwrap())
}

pub fn v3_bits(p: V3) -> Value {
    Value::Array(vec![s(bits32(p.0)), s(bits32(p.1)), s(bits32(p.2))])
}

pub fn v3_of(v: &Value) -> V3 {
    let a = arr(v);
    V3(unbits(&a[0]), unbits(&a[1]), unbits(&a[2]))
}

pub fn m4_bits(m: &M4) -> Value {
    Value::Array(m.data().iter().map(|&x| s(bits32(x))).collect())
}

pub fn s(x: impl Into<String>) -> Value {
    Value::Str(x.into())
}

pub fn int(i: i64) -> Value {
    Value::int(i)
}

pub fn flt(f: f64) -> Value {
    Value::Float(f)
}

pub fn arr(v: &Value) -> &[Value] {
    match v {
        Value::Array(a) => a,
        other => panic!("not an array: {other:?}"),
    }
}

pub fn text(v: &Value) -> &str {
    match v {
        Value::Str(t) => t,
        other => panic!("not a string: {other:?}"),
    }
}

pub fn num(v: &Value) -> f64 {
    match v {
        Value::Float(f) => *f,
        Value::Int(i) => i.parse().unwrap(),
        other => panic!("not a number: {other:?}"),
    }
}

pub fn is_true(v: Option<&Value>) -> bool {
    v == Some(&Value::Bool(true))
}

pub fn obj(items: Vec<(&str, Value)>) -> Value {
    Value::Object(items.into_iter().map(|(k, v)| (k.to_string(), v)).collect())
}

fn opt(v: Option<Value>) -> Value {
    v.unwrap_or(Value::Null)
}

// ---- the state --------------------------------------------------------------------------------

/// `mesh_state` of the harness.
pub fn mesh_state(m: &Mesh) -> Value {
    let vi: HashMap<_, usize> = m.vertices.iter().enumerate().map(|(i, &v)| (v, i)).collect();
    let fi: HashMap<_, usize> = m.faces.iter().enumerate().map(|(i, &f)| (f, i)).collect();
    let files = Files { zip: None };
    let v = m.vertices.iter().map(|&v| v3_bits(m.position(v))).collect();
    let e = m
        .edges
        .iter()
        .map(|&e| {
            let ed = m.edge(e);
            Value::Array(vec![
                int(vi[&ed.v0] as i64),
                int(vi[&ed.v1] as i64),
                int(ed.soft as i64),
                int(ed.hidden as i64),
                ed.layer.clone().map(Value::Str).unwrap_or(Value::Null),
                ed.curve.map(Value::int).unwrap_or(Value::Null),
                Value::Array(ed.faces.iter().map(|f| int(fi[f] as i64)).collect()),
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
                Value::Array(fc.outer.iter().map(|v| int(vi[v] as i64)).collect()),
                Value::Array(fc.holes.iter().map(|h| Value::Array(h.iter().map(|v| int(vi[v] as i64)).collect())).collect()),
                int(fc.interior as i64),
                s(attrs),
            ])
        })
        .collect();
    obj(vec![("v", Value::Array(v)), ("e", Value::Array(e)), ("f", Value::Array(f))])
}

fn group_state(scene: &Scene, g: &Group) -> Value {
    obj(vec![
        ("name", s(g.name.clone())),
        ("uid", s(g.uid.clone())),
        ("hidden", int(g.hidden as i64)),
        ("locked", int(g.locked as i64)),
        ("xform", g.xform.as_ref().map(m4_bits).unwrap_or(Value::Null)),
        ("axes", g.axes.as_ref().map(m4_bits).unwrap_or(Value::Null)),
        ("mesh", mesh_state(scene.mesh(g.mesh))),
        ("kids", Value::Array(g.children.iter().map(|c| group_state(scene, c)).collect())),
    ])
}

/// How a selected entity is named in the fixtures.
pub fn sel_ref(scene: &Scene, cx: &Cx, e: &Sel) -> Value {
    let m = cx.mesh(scene);
    let list = |k: &str, i: usize| Value::Array(vec![s(k), int(i as i64)]);
    match e {
        Sel::Face(f) => list("face", m.faces.iter().position(|x| x == f).expect("selected face is gone")),
        Sel::Edge(x) => list("edge", m.edges.iter().position(|y| y == x).expect("selected edge is gone")),
        Sel::Group(p) => Value::Array(vec![s("group"), Value::Array(p.iter().map(|&i| int(i as i64)).collect())]),
        Sel::Dimension(i) => list("dim", *i),
        Sel::Label(i) => list("label", *i),
        Sel::Guide(i) => list("guide", *i),
        Sel::Section(i) => list("section", *i),
        Sel::Image(i) => list("image", *i),
        Sel::GeoPath(i) => list("geo", *i),
    }
}

pub fn sel_of(scene: &Scene, cx: &Cx, r: &Value) -> Sel {
    let a = arr(r);
    let kind = text(&a[0]);
    let idx = |v: &Value| v.as_i64().unwrap() as usize;
    match kind {
        "face" => Sel::Face(cx.mesh(scene).faces[idx(&a[1])]),
        "edge" => Sel::Edge(cx.mesh(scene).edges[idx(&a[1])]),
        "group" => Sel::Group(arr(&a[1]).iter().map(idx).collect()),
        "dim" => Sel::Dimension(idx(&a[1])),
        "label" => Sel::Label(idx(&a[1])),
        "guide" => Sel::Guide(idx(&a[1])),
        "section" => Sel::Section(idx(&a[1])),
        "image" => Sel::Image(idx(&a[1])),
        "geo" => Sel::GeoPath(idx(&a[1])),
        other => panic!("selection kind {other}"),
    }
}

pub fn scene_state(scene: &Scene, cx: &Cx, sel: &Selection) -> Value {
    obj(vec![
        ("m", mesh_state(cx.mesh(scene))),
        ("g", Value::Array(scene.groups.iter().map(|g| group_state(scene, g)).collect())),
        (
            "guides",
            Value::Array(
                scene
                    .guides
                    .iter()
                    .map(|g| Value::Array(vec![v3_bits(g.point), g.direction.map(v3_bits).unwrap_or(Value::Null), g.origin.map(v3_bits).unwrap_or(Value::Null)]))
                    .collect(),
            ),
        ),
        ("sel", Value::Array(sel.iter().map(|e| sel_ref(scene, cx, e)).collect())),
        ("edit", Value::Bool(cx.mesh != MeshId(0))),
    ])
}

fn path_value(p: &[usize]) -> Value {
    Value::Array(p.iter().map(|&i| int(i as i64)).collect())
}

fn segs(v: &[(V3, V3)]) -> Value {
    Value::Array(v.iter().map(|&(a, b)| Value::Array(vec![v3_bits(a), v3_bits(b)])).collect())
}

/// A [`Probe`] in the shape of the harness's `generic_probe`.
pub fn probe_value(p: &Probe) -> Value {
    let mut out: Vec<(&str, Value)> = Vec::new();
    if let Some(rb) = &p.rb {
        out.push(("rb", segs(rb)));
    }
    if let Some(gl) = &p.gl {
        out.push(("gl", segs(gl)));
    }
    if let Some(vl) = &p.vl {
        out.push(("vl", match vl {
            None => Value::Null,
            Some((t, at)) => Value::Array(vec![s(t.clone()), v3_bits(*at)]),
        }));
    }
    if let Some(sc) = &p.sc {
        out.push(("sc", s(sc.clone())));
    }
    if let Some(b) = p.plus {
        out.push(("plus", Value::Bool(b)));
    }
    out.push(("wf", match p.wf {
        None => Value::Null,
        Some(c) => Value::Array(c.iter().map(|&x| flt(x)).collect()),
    }));
    if let Some(c) = &p.cap {
        out.push(("cap", s(c.clone())));
    }
    if let Some(b) = p.unitless {
        out.push(("unitless", Value::Bool(b)));
    }
    out.push(("uses_snap", Value::Bool(p.uses_snap)));
    if let Some(ex) = &p.excl {
        out.push(("excl", match ex {
            None => Value::Null,
            Some((edges, groups)) => {
                let mut e: Vec<usize> = edges.clone();
                e.sort_unstable();
                // json.dumps of a path, sorted as text
                let mut g: Vec<String> = groups.iter().map(|p| format!("[{}]", p.iter().map(|i| i.to_string()).collect::<Vec<_>>().join(", "))).collect();
                g.sort();
                Value::Array(vec![Value::Array(e.iter().map(|&i| int(i as i64)).collect()), Value::Array(g.into_iter().map(Value::Str).collect())])
            }
        }));
    }
    if let Some(b) = p.ap {
        out.push(("ap", Value::Bool(b)));
    }
    out.push(("start", p.start.map(v3_bits).unwrap_or(Value::Null)));
    out.push(("hover", p.hover.map(v3_bits).unwrap_or(Value::Null)));
    if let Some(x) = &p.x {
        out.push(("x", x.clone()));
    }
    obj(out)
}

// ---- the first difference ------------------------------------------------------------------------------

pub fn first_difference(path: &str, rust: &Value, py: &Value) -> Option<String> {
    match (rust, py) {
        (Value::Array(x), Value::Array(y)) => {
            for (i, (p, q)) in x.iter().zip(y).enumerate() {
                if let Some(d) = first_difference(&format!("{path}[{i}]"), p, q) {
                    return Some(d);
                }
            }
            (x.len() != y.len()).then(|| format!("{path}: Rust {} entries, Python {}", x.len(), y.len()))
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
        _ => (rust != py).then(|| format!("{path}: Rust {rust:?} vs Python {py:?}")),
    }
}

// ---- the rig ---------------------------------------------------------------------------------------------

pub struct Rig {
    pub scene: Scene,
    pub cx: Cx,
    pub history: History,
    pub sel: Selection,
    pub vp: PickViewport,
    pub units: ModelUnits,
    pub version: u64,
}

pub fn camera_of(spec: &Value) -> (View, i32, i32) {
    let g = |k: &str| spec.get(k).unwrap();
    let w = spec.get("w").map(|v| v.as_i64().unwrap() as i32).unwrap_or(1000);
    let h = spec.get("h").map(|v| v.as_i64().unwrap() as i32).unwrap_or(600);
    let t = arr(g("target"));
    let orbit = OrbitCamera {
        target: V3(num(&t[0]) as f32, num(&t[1]) as f32, num(&t[2]) as f32),
        distance: num(g("distance")),
        yaw: num(g("yaw")).to_radians(),
        pitch: num(g("pitch")).to_radians(),
        up: V3(0.0, 0.0, 1.0),
    };
    let mut cam = Camera::new(orbit);
    cam.perspective = spec.get("perspective").map(|v| v == &Value::Bool(true)).unwrap_or(true);
    if let Some(f) = spec.get("fov") {
        cam.fov_deg = num(f);
    }
    cam.set_aspect(w, h);
    (View::new(cam, w, h, H), w, h)
}

impl Rig {
    pub fn new(tool: &str, case: &Value) -> Rig {
        let name = text(case.get("scene").unwrap());
        let scene = open(tool, name);
        let (view, _, _) = camera_of(case.get("cam").unwrap());
        let mut cx = Cx { hypot: H, mesh: MeshId(0), ..Cx::default() };
        cx.counters.curve = 5000;
        cx.counters.uid = 1;
        Rig { scene, cx, history: History::new(), sel: Selection::default(), vp: PickViewport::new(view, H), units: ModelUnits::default(), version: 0 }
    }

    pub fn env(&mut self) -> Env<'_> {
        Env { scene: &mut self.scene, cx: &mut self.cx, history: &mut self.history, sel: &mut self.sel, vp: &mut self.vp, units: &self.units, version: &mut self.version }
    }

    pub fn state(&self) -> Value {
        scene_state(&self.scene, &self.cx, &self.sel)
    }
}

// ---- decoding the calls ----------------------------------------------------------------------------------------

pub fn snap_kind(k: &str) -> SnapKind {
    match k {
        "axis" => SnapKind::Axis,
        "reference" => SnapKind::Reference,
        "close" => SnapKind::Close,
        "endpoint" => SnapKind::Endpoint,
        "intersection" => SnapKind::Intersection,
        "midpoint" => SnapKind::Midpoint,
        "origin" => SnapKind::Origin,
        "on_edge" => SnapKind::OnEdge,
        "on_line" => SnapKind::OnLine,
        "aligned" => SnapKind::Aligned,
        "axis_inference" => SnapKind::AxisInference,
        "on_face" => SnapKind::OnFace,
        "none" => SnapKind::None,
        "center" => SnapKind::Center,
        "component_origin" => SnapKind::ComponentOrigin,
        "arc_midpoint" => SnapKind::ArcMidpoint,
        "from_point" => SnapKind::FromPoint,
        "extension" => SnapKind::Extension,
        "through_point" => SnapKind::ThroughPoint,
        "perp_face" => SnapKind::PerpFace,
        "on_axis" => SnapKind::OnAxis,
        other => panic!("snap kind {other:?}"),
    }
}

pub fn tool_cx(v: &Value) -> ToolCx {
    let sc = arr(v.get("screen").unwrap());
    let snap = match v.get("snap").unwrap() {
        Value::Null => None,
        sn => Some(SnapInfo {
            kind: snap_kind(text(sn.get("kind").unwrap())),
            point: v3_of(sn.get("point").unwrap()),
            axis: match sn.get("axis").unwrap() {
                Value::Null => None,
                a => Axis::from_name(text(a)),
            },
            context: match sn.get("context").unwrap() {
                Value::Null => None,
                c => match text(c) {
                    "group" => Some(Context::Group),
                    "component" => Some(Context::Component),
                    other => panic!("snap context {other:?}"),
                },
            },
        }),
    };
    ToolCx { world: v3_of(v.get("world").unwrap()), screen: (num(&sc[0]), num(&sc[1])), mods: Mods(v.get("mods").unwrap().as_i64().unwrap() as u32), snap }
}

pub fn vcb_of(v: &Value) -> Vcb {
    if let Some(n) = v.get("num") {
        return Vcb::Num(num(n));
    }
    if let Some(t) = v.get("tuple") {
        return Vcb::Tuple(arr(t).iter().map(|i| num(i.get("num").expect("a tuple of numbers"))).collect());
    }
    if let Some(a) = v.get("abs_len") {
        return Vcb::AbsLen(Box::new(vcb_of(a)));
    }
    panic!("value {v:?}")
}

fn gfx_value(g: &Gfx, scene: &Scene, cx: &Cx) -> Value {
    match g {
        Gfx::Hover(e, p) => Value::Array(vec![
            s("hover"),
            e.as_ref().map(|e| sel_ref(scene, cx, e)).unwrap_or(Value::Null),
            p.as_ref().map(|p| Value::Array(vec![s("group"), path_value(p)])).unwrap_or(Value::Null),
        ]),
        Gfx::Begin(gs, ext) => Value::Array(vec![s("begin"), Value::Array(gs.iter().map(|p| Value::Array(vec![s("group"), path_value(p)])).collect()), Value::Bool(*ext)]),
        Gfx::Offset(d) => Value::Array(vec![s("offset"), v3_bits(*d)]),
        Gfx::Matrix(m) => Value::Array(vec![s("matrix"), m4_bits(m)]),
        Gfx::End => Value::Array(vec![s("end")]),
    }
}

/// Make one recorded call on the tool; returns what it answered, when it
/// answers.
pub fn replay_call(rig: &mut Rig, tool: &mut dyn ModifyTool, call: &Value, label: &str) -> Option<bool> {
    let m = text(call.get("m").unwrap());
    match m {
        "on_activate" => tool.on_activate(&mut rig.env()),
        "on_deactivate" => tool.on_deactivate(&mut rig.env()),
        "on_click" => tool.on_click(&mut rig.env(), &tool_cx(call.get("ctx").unwrap())),
        "on_hover" => tool.on_hover(&mut rig.env(), &tool_cx(call.get("ctx").unwrap())),
        "on_double_click" => tool.on_double_click(&mut rig.env(), &tool_cx(call.get("ctx").unwrap())),
        "on_triple_click" => tool.on_triple_click(&mut rig.env(), &tool_cx(call.get("ctx").unwrap())),
        "on_cancel" => tool.on_cancel(&mut rig.env()),
        "close_array_window" => tool.close_array_window(),
        "on_release" => tool.on_release(&mut rig.env()),
        "on_key" => {
            let key = call.get("key").unwrap().as_i64().unwrap() as i32;
            let mods = Mods(call.get("mods").unwrap().as_i64().unwrap() as u32);
            return Some(tool.on_key(&mut rig.env(), key, mods));
        }
        "on_key_release" => {
            let key = call.get("key").unwrap().as_i64().unwrap() as i32;
            rig.vp.ctrl_alone = is_true(call.get("tapped"));
            return Some(tool.on_key_release(&mut rig.env(), key));
        }
        "on_value" => return Some(tool.on_value(&mut rig.env(), &vcb_of(call.get("value").unwrap()))),
        "on_array_value" => {
            let count = call.get("count").unwrap().as_i64().unwrap();
            let mode = text(call.get("mode").unwrap()).chars().next().unwrap();
            let step = match call.get("step").unwrap() {
                Value::Null => None,
                v => Some(num(v)),
            };
            return Some(tool.on_array_value(&mut rig.env(), count, mode, step));
        }
        "on_box_select" => {
            let r = arr(call.get("rect").unwrap());
            tool.on_box_select(&mut rig.env(), [num(&r[0]), num(&r[1]), num(&r[2]), num(&r[3])], is_true(call.get("crossing")));
        }
        "vp_flash" => rig.vp.flashes.push((text(call.get("text").unwrap()).to_string(), call.get("msec").unwrap().as_i64().unwrap() as u32)),
        "vp_clear_selection" => {
            if !rig.sel.is_empty() {
                rig.sel.clear();
                rig.version += 1;
            }
        }
        "vp_select" => {
            let ents: Vec<Sel> = arr(call.get("ents").unwrap()).iter().map(|r| sel_of(&rig.scene, &rig.cx, r)).collect();
            rig.sel.select(&ents, text(call.get("mode").unwrap()));
            rig.version += 1;
        }
        other => panic!("{label}: unknown call {other}"),
    }
    None
}

/// Statistics of a run.
#[derive(Default, Debug)]
pub struct Stats {
    pub programs: usize,
    pub events: usize,
    pub calls: usize,
    pub commands: usize,
    pub flashes: usize,
}

/// Run every case of `tool` through `make()` and compare each event.
pub fn run_cases(tool_dir: &str, make: &dyn Fn() -> Box<dyn ModifyTool>) -> Stats {
    run_cases_with(tool_dir, &|_, _| make())
}

/// `run_cases` for a tool that needs the program's own extras (the `setup`
/// record of the module: text metrics, the initial class-level state...):
/// `make` gets the case and the rig, to set up the rig's viewport and build
/// the tool.
pub fn run_cases_with(tool_dir: &str, make: &dyn Fn(&Value, &mut Rig) -> Box<dyn ModifyTool>) -> Stats {
    let raw = std::fs::read_to_string(data(tool_dir).join("cases.json")).unwrap();
    let cases = loads(&raw).unwrap();
    let mut stats = Stats::default();
    for case in arr(&cases) {
        let n = case.get("n").unwrap().as_i64().unwrap();
        let label0 = format!("program {n} on {}", text(case.get("scene").unwrap()));
        let mut rig = Rig::new(tool_dir, case);
        let mut tool = make(case, &mut rig);
        if let Some(d) = first_difference("pre", &rig.state(), case.get("pre").unwrap()) {
            panic!("{label0}: reopened scene differs: {d}");
        }
        // the tool picked up
        let act = case.get("activate").unwrap();
        for call in arr(act.get("calls").unwrap()) {
            replay_call(&mut rig, tool.as_mut(), call, &label0);
        }
        let mut got = rig_observation(&mut rig, tool.as_ref());
        compare_event(&label0, "activate", &mut rig, &got, act);
        for (k, ev) in arr(case.get("events").unwrap()).iter().enumerate() {
            let kind = text(ev.get("e").unwrap());
            let label = format!("{label0}, event {k} ({kind})");
            rig.vp.flashes.clear();
            rig.vp.gfx.clear();
            match kind {
                "select" => {
                    let ents: Vec<Sel> = arr(ev.get("refs").unwrap()).iter().map(|r| sel_of(&rig.scene, &rig.cx, r)).collect();
                    rig.sel.select(&ents, text(ev.get("mode").unwrap()));
                    rig.version += 1;
                }
                "undo" => {
                    rig.env().undo();
                }
                "redo" => {
                    rig.env().redo();
                }
                "camera" => {
                    let (view, _, _) = camera_of(ev.get("cam").unwrap());
                    rig.vp.view = view;
                    rig.version += 1;
                }
                _ => {}
            }
            for call in arr(ev.get("calls").unwrap()) {
                let ret = replay_call(&mut rig, tool.as_mut(), call, &label);
                if let (Some(r), Some(want)) = (ret, call.get("ret")) {
                    assert_eq!(r, want == &Value::Bool(true), "{label}: {} answered {r}", text(call.get("m").unwrap()));
                }
                stats.calls += 1;
            }
            got = rig_observation(&mut rig, tool.as_ref());
            compare_event(&label, kind, &mut rig, &got, ev);
            stats.events += 1;
            stats.flashes += rig.vp.flashes.len();
            // depth
            let depth = rig.history.undo_len() as i64;
            assert_eq!(depth, ev.get("depth").unwrap().as_i64().unwrap(), "{label}: undo depth");
            assert_eq!(rig.history.redo_len() as i64, ev.get("redo").unwrap().as_i64().unwrap(), "{label}: redo depth");
        }
        stats.programs += 1;
    }
    stats
}

/// The observable state after an event: scene state and probe.
fn rig_observation(rig: &mut Rig, tool: &dyn ModifyTool) -> Value {
    let env = Env { scene: &mut rig.scene, cx: &mut rig.cx, history: &mut rig.history, sel: &mut rig.sel, vp: &mut rig.vp, units: &rig.units, version: &mut rig.version };
    let probe = probe_value(&tool.probe(&env));
    obj(vec![("state", scene_state_of(&env)), ("probe", probe)])
}

fn scene_state_of(env: &Env) -> Value {
    scene_state(env.scene, env.cx, env.sel)
}

fn compare_event(label: &str, kind: &str, rig: &mut Rig, got: &Value, want: &Value) {
    let _ = kind;
    // the statuses it flashed, the hooks it drove
    let flashes = Value::Array(rig.vp.flashes.iter().map(|(t, ms)| Value::Array(vec![s(t.clone()), int(*ms as i64)])).collect());
    if let Some(w) = want.get("flash") {
        if let Some(d) = first_difference("flash", &flashes, w) {
            panic!("{label}: {d}");
        }
    }
    let gfx = Value::Array(rig.vp.gfx.iter().map(|g| gfx_value(g, &rig.scene, &rig.cx)).collect());
    if let Some(w) = want.get("gfx") {
        if let Some(d) = first_difference("gfx", &gfx, w) {
            panic!("{label}: {d}");
        }
    }
    if let Some(d) = first_difference("state", got.get("state").unwrap(), want.get("state").unwrap()) {
        if std::env::var("MODIFY_DEBUG").is_ok() {
            eprintln!("RUST  : {}", canon_dumps(got.get("state").unwrap().get("m").unwrap()));
            eprintln!("PYTHON: {}", canon_dumps(want.get("state").unwrap().get("m").unwrap()));
        }
        panic!("{label}: {d}");
    }
    if let Some(d) = first_difference("probe", got.get("probe").unwrap(), want.get("probe").unwrap()) {
        panic!("{label}: {d}");
    }
    let rej = rig.history.last_rejected.clone();
    let want_rej = match want.get("rejected") {
        None | Some(Value::Null) => None,
        Some(v) => Some(text(v).to_string()),
    };
    if want.get("rejected").is_some() {
        assert_eq!(rej, want_rej, "{label}: history.last_rejected");
    }
}
