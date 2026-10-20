// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! The Rust snap engine against the real Python one
//! (scripts/golden/make_snap_fixtures.py): thousands of cursor samples over
//! varied scenes, cameras and tool states, each with what `Viewport.
//! inference.snap` answered and the intermediate values it worked from.
//!
//! The pieces are held apart so a mismatch says where it began:
//!
//! * the camera: matrices and the pixel functions;
//! * the occluder, against every ray question Python asked;
//! * the snap scene (edges), the cursor's world point, the work-plane
//!   normal and the face under the cursor, each against Python's;
//! * `compute_snap` alone, from Python's own arguments and the answers it
//!   got to the occlusion questions;
//! * the whole engine, from the scene and the tool state to the answer;
//! * the ScreenTip, English and Spanish.
//!
//! The fixtures come from a Windows machine: `length()` is MSVC's `hypot`.

use std::cell::RefCell;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

use ingetrazo_formats::igz;
use ingetrazo_formats::pyjson::{loads, Value};
use ingetrazo_i18n::Catalog;
use ingetrazo_math::{Hypot, V3};
use ingetrazo_model::scene::Scene;
use ingetrazo_snap::camera::{OrbitCamera, View};
use ingetrazo_snap::compute::{compute_snap, LinearMode, RefMode};
use ingetrazo_snap::engine::{self, snap_params, snap_scene, tooltip, InferenceState, Settings, SnapInput};
use ingetrazo_snap::types::*;
use ingetrazo_snap::workplane::{PlaneCx, ToolState};
use ingetrazo_snap::World;

const H: Hypot = Hypot::Msvc;

fn data() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data")
}

// ---- decoding ------------------------------------------------------------------

fn arr(v: &Value) -> &Vec<Value> {
    match v {
        Value::Array(a) => a,
        _ => panic!("not an array: {v:?}"),
    }
}

fn i(v: &Value) -> i64 {
    v.as_i64().unwrap_or_else(|| panic!("not an int: {v:?}"))
}

fn num(v: &Value) -> f64 {
    match v {
        Value::Float(f) => *f,
        Value::Int(s) => s.parse().unwrap(),
        _ => panic!("not a number: {v:?}"),
    }
}

fn s(v: &Value) -> &str {
    match v {
        Value::Str(x) => x,
        _ => panic!("not a string: {v:?}"),
    }
}

fn b(v: &Value) -> bool {
    match v {
        Value::Bool(x) => *x,
        _ => panic!("not a bool: {v:?}"),
    }
}

fn opt(v: Option<&Value>) -> Option<&Value> {
    match v {
        Some(Value::Null) | None => None,
        x => x,
    }
}

fn f32b(v: &Value) -> f32 {
    f32::from_bits(i(v) as u32)
}

fn p3(v: &Value) -> V3 {
    let a = arr(v);
    V3(f32b(&a[0]), f32b(&a[1]), f32b(&a[2]))
}

fn op3(v: Option<&Value>) -> Option<V3> {
    opt(v).map(p3)
}

fn bits3(p: V3) -> [u32; 3] {
    [p.0.to_bits(), p.1.to_bits(), p.2.to_bits()]
}

fn rgb(v: &Value) -> Rgb {
    let a = arr(v);
    [num(&a[0]), num(&a[1]), num(&a[2])]
}

fn axis_of(v: Option<&Value>) -> Option<Axis> {
    opt(v).and_then(|x| Axis::from_name(s(x)))
}

// ---- the fixtures ----------------------------------------------------------------

struct Fixtures {
    header: Value,
    samples: Vec<Value>,
    scenes: HashMap<String, World>,
}

fn load() -> Fixtures {
    let header = loads(&std::fs::read_to_string(data().join("header.json")).unwrap()).unwrap();
    let text = std::fs::read_to_string(data().join("samples.jsonl")).unwrap();
    let samples: Vec<Value> = text.lines().map(|l| loads(l).unwrap()).collect();
    let mut scenes = HashMap::new();
    if let Some(Value::Object(items)) = header.get("scenes") {
        for (name, meta) in items {
            let raw = std::fs::read(data().join("scenes").join(format!("{name}.igz.json"))).unwrap();
            let doc = igz::read_document(&raw).unwrap();
            let scene = Scene::from_document(&doc).unwrap_or_else(|e| panic!("{name}: {e:?}"));
            let world = match opt(meta.get("edit")) {
                Some(path) => World::build_editing(&scene, i(&arr(path)[0]) as usize, H),
                None => World::build(&scene, H),
            };
            scenes.insert(name.clone(), world);
        }
    }
    Fixtures { header, samples, scenes }
}

fn view_of(c: &Value) -> View {
    let cam = OrbitCamera {
        target: p3(c.get("target").unwrap()),
        distance: num(c.get("dist").unwrap()),
        yaw: num(c.get("yaw").unwrap()),
        pitch: num(c.get("pitch").unwrap()),
        up: p3(c.get("up").unwrap()),
        fov_deg: num(c.get("fov").unwrap()),
        aspect: num(c.get("aspect").unwrap()),
        znear: num(c.get("znear").unwrap()),
        zfar: num(c.get("zfar").unwrap()),
        perspective: b(c.get("persp").unwrap()),
    };
    View::new(cam, i(c.get("w").unwrap()) as i32, i(c.get("h").unwrap()) as i32, H)
}

/// The view of a sample: its camera, in the window size it was asked in
/// (Qt's layout resized the offscreen window now and then; `wh` says so).
fn view_s(cams: &[Value], smp: &Value) -> View {
    let c = &cams[i(smp.get("c").unwrap()) as usize];
    let v = view_of(c);
    match smp.get("wh") {
        Some(wh) => View::new(v.cam, i(&arr(wh)[0]) as i32, i(&arr(wh)[1]) as i32, H),
        None => v,
    }
}

fn axes_of(v: &Value) -> Axes {
    Axes {
        x: p3(v.get("x").unwrap()),
        y: p3(v.get("y").unwrap()),
        z: p3(v.get("z").unwrap()),
        origin: p3(v.get("o").unwrap()),
        world: b(v.get("w").unwrap()),
    }
}

fn settings_of(h: &Value) -> Settings {
    let t = h.get("thresholds").unwrap();
    Settings {
        snap_threshold_px: num(t.get("snap").unwrap()),
        edge_snap_threshold_px: num(t.get("edge").unwrap()),
        inference_angle_deg: num(t.get("angle").unwrap()),
        pick_threshold_px: num(t.get("pick").unwrap()),
        center_min_radius_px: num(t.get("center_min").unwrap()),
        center_hint_px: num(t.get("center_hint").unwrap()),
    }
}

fn plane_of(v: Option<&Value>) -> Option<(V3, V3)> {
    opt(v).map(|x| {
        let a = arr(x);
        (p3(&a[0]), p3(&a[1]))
    })
}

fn tool_of(st: &Value) -> ToolState {
    ToolState {
        uses_snap: b(st.get("us").unwrap()),
        is_plane_lock: b(st.get("plc").unwrap()),
        prefers_vertical_drag: b(st.get("pv").unwrap()),
        radial_arm: b(st.get("ra").unwrap()),
        magnetic_axis_deg: opt(st.get("mag")).map(num),
        screen_axis_px: opt(st.get("scr")).map(num),
        magnet_off_plane: b(st.get("mop").unwrap()),
        axis_source: b(st.get("axs").unwrap()),
        axis_source_empty_only: b(st.get("axe").unwrap()),
        hover_group_edges: b(st.get("hge").unwrap()),
        start_point: op3(st.get("sp")),
        chain_first_point: op3(st.get("cf")),
        chain_first_is_start: b(st.get("cfs").unwrap()),
        work_plane: plane_of(st.get("wp")),
        plane_lock: axis_of(st.get("pl")),
        plane_ref: op3(st.get("pr")),
        own_plane: plane_of(st.get("own")),
    }
}

fn edge_ref(v: Option<&Value>) -> Option<EdgeRef> {
    opt(v).map(|x| {
        let a = arr(x);
        EdgeRef { a: p3(&a[0]), b: p3(&a[1]) }
    })
}

fn state_of(st: &Value) -> InferenceState {
    InferenceState {
        axis_lock: axis_of(st.get("lock")),
        reference_edge: edge_ref(st.get("re")),
        reference_mode: opt(st.get("rm")).map(|m| if s(m) == "perpendicular" { RefMode::Perpendicular } else { RefMode::Parallel }),
        shift_lock: opt(st.get("sl")).map(|x| {
            let a = arr(x);
            (p3(&a[0]), rgb(&a[1]))
        }),
        acquired_edge: edge_ref(st.get("ae")),
        acquired_point: op3(st.get("ap")),
        acquired_face_normal: op3(st.get("afn")),
        encouraged: arr(st.get("enc").unwrap()).iter().map(p3).collect(),
        linear_mode: Some(match s(st.get("lm").unwrap()) {
            "off" => LinearMode::Off,
            "parallel_perp" => LinearMode::ParallelPerp,
            _ => LinearMode::All,
        }),
        hover_edge: None,
        hover_center: None,
        center_ref: opt(st.get("ctr")).map(|x| {
            let a = arr(x);
            (p3(&a[0]), f32b(&a[1]))
        }),
    }
}

fn pool_edges(h: &Value, ix: i64) -> Vec<SnapEdge> {
    arr(&arr(h.get("pool").unwrap())[ix as usize])
        .iter()
        .map(|row| {
            let r = arr(row);
            let a = V3(f32b(&r[0]), f32b(&r[1]), f32b(&r[2]));
            let bb = V3(f32b(&r[3]), f32b(&r[4]), f32b(&r[5]));
            let flags = i(&r[6]);
            let mut e = SnapEdge::new(a, bb);
            e.soft = flags & 1 != 0;
            e.hidden = flags & 2 != 0;
            e.figure = flags & 4 != 0;
            e.center = flags & 8 != 0;
            e.component_origin = flags & 16 != 0;
            e.arc_midpoint = flags & 32 != 0;
            e.guide = flags & 64 != 0;
            e.context = if flags & 128 != 0 {
                Some(Context::Group)
            } else if flags & 256 != 0 {
                Some(Context::Component)
            } else {
                None
            };
            e.curve = if flags & 512 != 0 { Some(0) } else { None };
            e
        })
        .collect()
}

// ---- comparing answers ---------------------------------------------------------------

fn same_rgb(a: Rgb, b: Rgb) -> bool {
    a.iter().zip(b.iter()).all(|(x, y)| x.to_bits() == y.to_bits())
}

fn show(r: &SnapResult) -> String {
    format!(
        "{} {:?} axis={:?} guide={:?} gc={:?} guides={:?} ctx={:?}",
        r.kind.as_str(),
        bits3(r.point).map(f32::from_bits),
        r.axis.map(|a| a.name()),
        r.guide.map(|(a, bb)| (bits3(a).map(f32::from_bits), bits3(bb).map(f32::from_bits))),
        r.guide_color,
        r.guides.as_ref().map(|g| g.len()),
        r.context
    )
}

fn show_want(w: &Value) -> String {
    format!(
        "{} {:?} axis={:?} guide={:?} gc={:?} gs={:?} cx={:?}",
        s(w.get("k").unwrap()),
        p3(w.get("p").unwrap()).0.to_string() + "," + &p3(w.get("p").unwrap()).1.to_string() + "," + &p3(w.get("p").unwrap()).2.to_string(),
        opt(w.get("ax")).map(s),
        opt(w.get("g")).map(|g| (p3(&arr(g)[0]), p3(&arr(g)[1]))),
        opt(w.get("gc")),
        opt(w.get("gs")).map(|g| arr(g).len()),
        opt(w.get("cx")).map(s)
    )
}

/// `Err(why)` when `got` differs from the fixture's result row `want`.
fn check(got: &SnapResult, want: &Value) -> Result<(), String> {
    let bad = |what: &str| Err(format!("{what}: got {}  want {}", show(got), show_want(want)));
    if got.kind.as_str() != s(want.get("k").unwrap()) {
        return bad("kind");
    }
    if bits3(got.point) != bits3(p3(want.get("p").unwrap())) {
        return bad("point");
    }
    if !same_rgb(got.color, rgb(want.get("c").unwrap())) {
        return bad("color");
    }
    if got.axis != axis_of(want.get("ax")) {
        return bad("axis");
    }
    match (got.guide, opt(want.get("g"))) {
        (None, None) => {}
        (Some((a, bb)), Some(g)) => {
            let g = arr(g);
            if bits3(a) != bits3(p3(&g[0])) || bits3(bb) != bits3(p3(&g[1])) {
                return bad("guide");
            }
        }
        _ => return bad("guide presence"),
    }
    match (got.guide_color, opt(want.get("gc"))) {
        (None, None) => {}
        (Some(a), Some(g)) => {
            if !same_rgb(a, rgb(g)) {
                return bad("guide colour");
            }
        }
        _ => return bad("guide colour presence"),
    }
    match (&got.guides, opt(want.get("gs"))) {
        (None, None) => {}
        (Some(gs), Some(w)) => {
            let w = arr(w);
            if gs.len() != w.len() {
                return bad("guides len");
            }
            for (g, ww) in gs.iter().zip(w) {
                let ww = arr(ww);
                if bits3(g.0) != bits3(p3(&ww[0])) || bits3(g.1) != bits3(p3(&ww[1])) || !same_rgb(g.2, rgb(&ww[2])) {
                    return bad("guides");
                }
            }
        }
        _ => return bad("guides presence"),
    }
    let want_ctx = opt(want.get("cx")).map(|c| if s(c) == "component" { Context::Component } else { Context::Group });
    if got.context != want_ctx {
        return bad("context");
    }
    Ok(())
}

/// A tally of mismatches by category: the first few kept to read.
#[derive(Default)]
struct Tally {
    checked: usize,
    bad: usize,
    first: Vec<String>,
}

impl Tally {
    fn ok(&mut self) {
        self.checked += 1;
    }
    fn fail(&mut self, msg: String) {
        self.checked += 1;
        self.bad += 1;
        if self.first.len() < 12 {
            self.first.push(msg);
        }
    }
    fn report(&self, name: &str) -> String {
        if self.bad == 0 {
            return String::new();
        }
        format!("{name}: {} of {} differ\n  {}\n", self.bad, self.checked, self.first.join("\n  "))
    }
}

// ---- tests -------------------------------------------------------------------------

#[test]
fn camera_matrices_and_pixel_functions() {
    let f = load();
    let mut t = Tally::default();
    let want_bits = |v: &Value| -> Vec<u32> { arr(v).iter().map(|x| i(x) as u32).collect() };
    for (k, c) in arr(f.header.get("cams").unwrap()).iter().enumerate() {
        let v = view_of(c);
        let view: Vec<u32> = v.cam.view_matrix(H).data().iter().map(|x| x.to_bits()).collect();
        let proj: Vec<u32> = v.cam.projection_matrix().data().iter().map(|x| x.to_bits()).collect();
        let eye = bits3(v.cam.eye());
        if view != want_bits(c.get("view").unwrap()) {
            t.fail(format!("cam {k}: view matrix"));
        } else if proj != want_bits(c.get("proj").unwrap()) {
            t.fail(format!("cam {k}: projection matrix"));
        } else if eye.to_vec() != want_bits(c.get("eye").unwrap()) {
            t.fail(format!("cam {k}: eye"));
        } else {
            t.ok();
        }
    }
    for case in arr(f.header.get("proj").unwrap()) {
        let cams = arr(f.header.get("cams").unwrap());
        let v = view_of(&cams[i(case.get("cam").unwrap()) as usize]);
        for r in arr(case.get("ray").unwrap()) {
            let px = arr(r.get("px").unwrap());
            let got = v.pixel_to_ray(num(&px[0]), num(&px[1]));
            let want = (p3(r.get("o").unwrap()), p3(r.get("d").unwrap()));
            match got {
                Some((o, d)) if bits3(o) == bits3(want.0) && bits3(d) == bits3(want.1) => t.ok(),
                other => t.fail(format!("ray {px:?}: got {other:?} want {want:?}")),
            }
        }
        for w in arr(case.get("w2p").unwrap()) {
            let got = v.world_to_pixel(p3(w.get("p").unwrap()));
            let want = opt(w.get("r")).map(|r| (num(&arr(r)[0]), num(&arr(r)[1])));
            let same = match (got, want) {
                (None, None) => true,
                (Some(a), Some(bb)) => a.0.to_bits() == bb.0.to_bits() && a.1.to_bits() == bb.1.to_bits(),
                _ => false,
            };
            if same {
                t.ok()
            } else {
                t.fail(format!("w2p: got {got:?} want {want:?}"))
            }
        }
        for l in arr(case.get("lock").unwrap()) {
            let px = arr(l.get("px").unwrap());
            let got = v.project_to_lock_line(p3(l.get("s").unwrap()), p3(l.get("d").unwrap()), num(&px[0]), num(&px[1]));
            if bits3(got) == bits3(p3(l.get("r").unwrap())) {
                t.ok()
            } else {
                t.fail(format!("lock line: got {got:?} want {:?}", p3(l.get("r").unwrap())))
            }
        }
        for pl in arr(case.get("plane").unwrap()) {
            let got = v.ray_plane(p3(pl.get("o").unwrap()), p3(pl.get("d").unwrap()), p3(pl.get("p").unwrap()), p3(pl.get("n").unwrap()), b(pl.get("ag").unwrap()));
            let want = op3(pl.get("r"));
            if got.map(bits3) == want.map(bits3) {
                t.ok()
            } else {
                t.fail(format!("ray_plane: got {got:?} want {want:?}"))
            }
        }
        for c in arr(case.get("clip").unwrap()) {
            let got = v.clip_segment_front(p3(c.get("a").unwrap()), p3(c.get("b").unwrap()));
            let want = opt(c.get("r")).map(|r| (p3(&arr(r)[0]), p3(&arr(r)[1])));
            if got.map(|(a, bb)| (bits3(a), bits3(bb))) == want.map(|(a, bb)| (bits3(a), bits3(bb))) {
                t.ok()
            } else {
                t.fail(format!("clip: got {got:?} want {want:?}"))
            }
        }
        for g in arr(case.get("guide").unwrap()) {
            let got = v.guide_snap_segment(p3(g.get("p").unwrap()), p3(g.get("d").unwrap()), 1.5);
            let want = opt(g.get("r")).map(|r| (p3(&arr(r)[0]), p3(&arr(r)[1])));
            if got.map(|(a, bb)| (bits3(a), bits3(bb))) == want.map(|(a, bb)| (bits3(a), bits3(bb))) {
                t.ok()
            } else {
                t.fail(format!("guide span: got {got:?} want {want:?}"))
            }
        }
    }
    let msg = t.report("camera");
    assert!(msg.is_empty(), "{msg}");
    assert!(t.checked > 200, "{}", t.checked);
}

#[test]
fn the_occluder_answers_every_ray_python_asked() {
    let f = load();
    let cams = arr(f.header.get("cams").unwrap());
    let mut t = Tally::default();
    for smp in &f.samples {
        let name = s(smp.get("s").unwrap());
        let Some(occ) = smp.get("occ") else { continue };
        let view = view_s(cams, smp);
        let world = &f.scenes[name];
        for q in arr(occ) {
            let q = arr(q);
            let p = V3(f32b(&q[0]), f32b(&q[1]), f32b(&q[2]));
            let got = world.is_occluded(view.cam.eye(), p, H);
            if got == (i(&q[3]) == 1) {
                t.ok();
            } else {
                t.fail(format!("{name}: occlusion of {p:?}: got {got}"));
            }
        }
    }
    let msg = t.report("occluder");
    assert!(msg.is_empty(), "{msg}");
    assert!(t.checked > 1000, "{}", t.checked);
}

#[test]
fn the_cursor_point_the_plane_and_the_face_under_it() {
    let f = load();
    let cams = arr(f.header.get("cams").unwrap());
    let states = arr(f.header.get("states").unwrap());
    let settings = settings_of(&f.header);
    let mut tw = Tally::default();
    let mut tp = Tally::default();
    let mut tf = Tally::default();
    for smp in &f.samples {
        let name = s(smp.get("s").unwrap());
        let view = view_s(cams, smp);
        let world = &f.scenes[name];
        let st = &states[i(smp.get("t").unwrap()) as usize];
        let tool = tool_of(st);
        let state = state_of(st);
        let axes = axes_of(f.header.get("scenes").unwrap().get(name).unwrap().get("axes").unwrap());
        let px = arr(smp.get("px").unwrap());
        let (x, y) = (num(&px[0]), num(&px[1]));
        let pc = PlaneCx { view: &view, world, axes: &axes, tool: &tool, axis_locked: state.axis_lock.is_some(), inference_angle_deg: settings.inference_angle_deg };
        let (wr, plane) = pc.world_from_pixel(x, y);
        let want_wr = op3(smp.get("wr"));
        if wr.map(bits3) == want_wr.map(bits3) {
            tw.ok();
        } else {
            tw.fail(format!("{name} t{} ({x},{y}): world {wr:?} want {want_wr:?}", i(smp.get("t").unwrap())));
        }
        if want_wr.is_some() {
            let want_n = op3(smp.get("wpn"));
            if plane.map(|p| bits3(p.1)) == want_n.map(bits3) {
                tp.ok();
            } else {
                tp.fail(format!("{name} t{} ({x},{y}): normal {:?} want {want_n:?}", i(smp.get("t").unwrap()), plane.map(|p| p.1)));
            }
            if tool.uses_snap {
                let fu = world.face_under(&view, x, y).is_some();
                let want_fu = smp.get("fu").is_some();
                if fu == want_fu {
                    tf.ok();
                } else {
                    tf.fail(format!("{name} t{} ({x},{y}): face under {fu} want {want_fu}", i(smp.get("t").unwrap())));
                }
            }
        }
    }
    let msg = format!("{}{}{}", tw.report("world point"), tp.report("plane normal"), tf.report("face under"));
    assert!(msg.is_empty(), "{msg}");
}

#[test]
fn the_snap_scene_is_pythons() {
    let f = load();
    let cams = arr(f.header.get("cams").unwrap());
    let states = arr(f.header.get("states").unwrap());
    let settings = settings_of(&f.header);
    let mut t = Tally::default();
    for smp in &f.samples {
        let name = s(smp.get("s").unwrap());
        let Some(ed) = smp.get("ed") else { continue };
        let view = view_s(cams, smp);
        let world = &f.scenes[name];
        let state = state_of(&states[i(smp.get("t").unwrap()) as usize]);
        let px = arr(smp.get("px").unwrap());
        let got = snap_scene(&view, world, &state, &settings, num(&px[0]), num(&px[1]));
        let want = pool_edges(&f.header, i(ed));
        let got: Vec<SnapEdge> = got.into_iter().map(|mut e| { e.curve = e.curve.map(|_| 0); e }).collect();
        if got == want {
            t.ok();
        } else {
            let extra: Vec<&SnapEdge> = got.iter().filter(|e| !want.contains(e)).collect();
            let missing: Vec<&SnapEdge> = want.iter().filter(|e| !got.contains(e)).collect();
            let short = |e: &SnapEdge| format!("{:?}->{:?} g={} c={} o={} a={}", e.a, e.b, e.guide, e.center, e.component_origin, e.arc_midpoint);
            let dists: Vec<String> = engine::gedge_distances(&view, world, num(&px[0]), num(&px[1])).iter().map(|d| format!("{d:.2}")).collect();
            t.fail(format!(
                "{name} t{} ({},{}): {} edges want {}; extra {:?}; missing {:?}; gedge dists {:?}",
                i(smp.get("t").unwrap()),
                num(&px[0]),
                num(&px[1]),
                got.len(),
                want.len(),
                extra.iter().map(|e| short(e)).collect::<Vec<_>>(),
                missing.iter().map(|e| short(e)).collect::<Vec<_>>(),
                dists
            ));
        }
    }
    let msg = t.report("snap scene");
    assert!(msg.is_empty(), "{msg}");
}

/// Python's answers to the occlusion questions, replayed.
struct Replay<'a> {
    view: &'a View,
    px: f64,
    py: f64,
    occ: HashMap<[u32; 3], bool>,
    missing: RefCell<Vec<[f32; 3]>>,
}

impl SnapEnv for Replay<'_> {
    fn world_to_pixel(&self, w: V3) -> Option<(f64, f64)> {
        self.view.world_to_pixel(w)
    }
    fn project_onto_line(&self, start: V3, dir: V3) -> V3 {
        self.view.project_to_lock_line(start, dir, self.px, self.py)
    }
    fn is_occluded(&self, w: V3) -> bool {
        match self.occ.get(&bits3(w)) {
            Some(&r) => r,
            None => {
                self.missing.borrow_mut().push([w.0, w.1, w.2]);
                false
            }
        }
    }
}

#[test]
fn compute_snap_from_pythons_own_arguments() {
    let f = load();
    let cams = arr(f.header.get("cams").unwrap());
    let states = arr(f.header.get("states").unwrap());
    let settings = settings_of(&f.header);
    let mut t = Tally::default();
    let mut args_t = Tally::default();
    for smp in &f.samples {
        let (Some(rc), Some(ed)) = (smp.get("rc"), smp.get("ed")) else { continue };
        let name = s(smp.get("s").unwrap());
        let view = view_s(cams, smp);
        let st = &states[i(smp.get("t").unwrap()) as usize];
        let tool = tool_of(st);
        let state = state_of(st);
        let axes = axes_of(f.header.get("scenes").unwrap().get(name).unwrap().get("axes").unwrap());
        let px = arr(smp.get("px").unwrap());
        let (x, y) = (num(&px[0]), num(&px[1]));
        let edges = pool_edges(&f.header, i(ed));
        let world_raw = p3(smp.get("wr").unwrap());
        let wpn = op3(smp.get("wpn"));
        let fu = smp.get("fu").is_some();
        let world = World::default();
        let inp = SnapInput { view: &view, world: &world, axes: &axes, tool: &tool, state: &state, settings: &settings, shift_held: smp.get("sh").is_some() };
        let p = snap_params(&inp, &edges, world_raw, wpn, fu, x, y);
        // The arguments Python recorded for every twentieth sample.
        if let Some(a) = smp.get("a") {
            let mut diffs = Vec::new();
            let eq = |c: bool, what: &str, d: &mut Vec<String>| {
                if !c {
                    d.push(what.to_string())
                }
            };
            eq(bits3(p.candidate_world) == bits3(p3(a.get("cw").unwrap())), "cw", &mut diffs);
            eq(p.chain_first_point.map(bits3) == op3(a.get("cf")).map(bits3), "cf", &mut diffs);
            eq(p.chain_first_point.is_none() || p.chain_first_is_start == b(a.get("cfs").unwrap()), "cfs", &mut diffs);
            eq(p.start_point.map(bits3) == op3(a.get("sp")).map(bits3), "sp", &mut diffs);
            eq(p.axis_lock == axis_of(a.get("al")), "al", &mut diffs);
            eq(p.shift_held == b(a.get("sh").unwrap()), "sh", &mut diffs);
            eq(p.inference_angle_deg == num(a.get("ia").unwrap()), "ia", &mut diffs);
            eq(p.screen_axis_px == opt(a.get("sa")).map(num), "sa", &mut diffs);
            eq(p.face_under_cursor == b(a.get("fu").unwrap()), "fu", &mut diffs);
            eq(p.edge_threshold_px == opt(a.get("et")).map(num), "et", &mut diffs);
            eq(p.magnetic_axis_deg == opt(a.get("ma")).map(num), "ma", &mut diffs);
            eq(p.acquired_point.map(bits3) == op3(a.get("ap")).map(bits3), "ap", &mut diffs);
            eq(p.acquired_face_normal.map(bits3) == op3(a.get("afn")).map(bits3), "afn", &mut diffs);
            eq(p.acquired_points.len() == arr(a.get("aps").unwrap()).len(), "aps", &mut diffs);
            eq(p.shift_lock_dir.map(bits3) == op3(a.get("sld")).map(bits3), "sld", &mut diffs);
            eq(p.work_plane_normal.map(bits3) == op3(a.get("wpn")).map(bits3), "wpn", &mut diffs);
            eq(p.radial_arm == b(a.get("ra").unwrap()), "ra", &mut diffs);
            eq(p.axis_in_plane == b(a.get("aip").unwrap()), "aip", &mut diffs);
            eq(
                p.reference_edge.map(|e| (bits3(e.a), bits3(e.b))) == opt(a.get("re")).map(|r| (bits3(p3(&arr(r)[0])), bits3(p3(&arr(r)[1])))),
                "re",
                &mut diffs,
            );
            eq(
                p.reference_mode.map(|m| m == RefMode::Perpendicular) == opt(a.get("rm")).map(|m| s(m) == "perpendicular"),
                "rm",
                &mut diffs,
            );
            if diffs.is_empty() {
                args_t.ok();
            } else {
                args_t.fail(format!("{name} t{}: arguments differ: {diffs:?}", i(smp.get("t").unwrap())));
            }
        }
        let mut occ = HashMap::new();
        if let Some(o) = smp.get("occ") {
            for q in arr(o) {
                let q = arr(q);
                occ.insert([i(&q[0]) as u32, i(&q[1]) as u32, i(&q[2]) as u32], i(&q[3]) == 1);
            }
        }
        let env = Replay { view: &view, px: x, py: y, occ, missing: RefCell::new(Vec::new()) };
        let cx = Cx { h: H, axes };
        let got = compute_snap(&env, cx, &p);
        match check(&got, rc) {
            Ok(()) if env.missing.borrow().is_empty() => t.ok(),
            Ok(()) => t.fail(format!("{name}: asked a ray Python did not: {:?}", env.missing.borrow().first())),
            Err(why) => t.fail(format!("{name} t{} ({x},{y}) {}: {why}", i(smp.get("t").unwrap()), states[i(smp.get("t").unwrap()) as usize].get("tool").map(s).unwrap_or(""))),
        }
    }
    let msg = format!("{}{}", args_t.report("recorded arguments"), t.report("compute_snap"));
    assert!(msg.is_empty(), "{msg}");
    assert!(t.checked > 5000, "{}", t.checked);
}

#[test]
fn the_whole_engine() {
    let f = load();
    let cams = arr(f.header.get("cams").unwrap());
    let states = arr(f.header.get("states").unwrap());
    let settings = settings_of(&f.header);
    let mut t = Tally::default();
    for smp in &f.samples {
        let name = s(smp.get("s").unwrap());
        let view = view_s(cams, smp);
        let world = &f.scenes[name];
        let st = &states[i(smp.get("t").unwrap()) as usize];
        let tool = tool_of(st);
        let state = state_of(st);
        let axes = axes_of(f.header.get("scenes").unwrap().get(name).unwrap().get("axes").unwrap());
        let px = arr(smp.get("px").unwrap());
        let (x, y) = (num(&px[0]), num(&px[1]));
        let inp = SnapInput { view: &view, world, axes: &axes, tool: &tool, state: &state, settings: &settings, shift_held: smp.get("sh").is_some() };
        let (got, _trace) = engine::snap(&inp, x, y);
        let want = opt(smp.get("r"));
        match (got, want) {
            (None, None) => t.ok(),
            (Some(g), Some(w)) => match check(&g, w) {
                Ok(()) => t.ok(),
                Err(why) => t.fail(format!("{name} t{} ({x},{y}) {}: {why}", i(smp.get("t").unwrap()), s(st.get("tool").unwrap()))),
            },
            (g, w) => t.fail(format!("{name} t{} ({x},{y}): got {:?} want {:?}", i(smp.get("t").unwrap()), g.map(|g| show(&g)), w.map(show_want))),
        }
    }
    let msg = t.report("engine");
    assert!(msg.is_empty(), "{msg}");
    assert!(t.checked > 5000, "{}", t.checked);
}

#[test]
fn the_screen_tip_in_english_and_spanish() {
    let f = load();
    let cams = arr(f.header.get("cams").unwrap());
    let _ = cams;
    let en = Catalog::empty("en");
    let es = Catalog::load("es", None);
    let mut t = Tally::default();
    for smp in &f.samples {
        let Some(r) = opt(smp.get("r")) else { continue };
        let kind = s(r.get("k").unwrap());
        let got_key = [
            SnapKind::Endpoint,
            SnapKind::Midpoint,
            SnapKind::ArcMidpoint,
            SnapKind::ComponentOrigin,
            SnapKind::OnLine,
            SnapKind::OnAxis,
            SnapKind::OnEdge,
            SnapKind::OnFace,
            SnapKind::Origin,
            SnapKind::Extension,
            SnapKind::Intersection,
            SnapKind::FromPoint,
            SnapKind::Aligned,
            SnapKind::ThroughPoint,
            SnapKind::PerpFace,
            SnapKind::Center,
            SnapKind::Close,
            SnapKind::Axis,
            SnapKind::Reference,
            SnapKind::AxisInference,
            SnapKind::None,
        ]
        .into_iter()
        .find(|k| k.as_str() == kind)
        .unwrap();
        let mut res = SnapResult::new(p3(r.get("p").unwrap()), got_key, [0.0; 3]);
        res.axis = axis_of(r.get("ax"));
        res.context = opt(r.get("cx")).map(|c| if s(c) == "component" { Context::Component } else { Context::Group });
        let want = opt(smp.get("lb"));
        let got_en = tooltip(&res, &|k| en.tr(k).to_string());
        let got_es = tooltip(&res, &|k| es.tr(k).to_string());
        match want {
            None => {
                if got_en.is_none() {
                    t.ok()
                } else {
                    t.fail(format!("{kind}: tip {got_en:?} want none"))
                }
            }
            Some(w) => {
                let w = arr(w);
                if got_en.as_deref() == Some(s(&w[0])) && got_es.as_deref() == Some(s(&w[1])) {
                    t.ok()
                } else {
                    t.fail(format!("{kind}: tip {got_en:?}/{got_es:?} want {:?}", (s(&w[0]), s(&w[1]))))
                }
            }
        }
    }
    let msg = t.report("tooltip");
    assert!(msg.is_empty(), "{msg}");
    assert!(t.checked > 1000);
}

fn edge_bits(e: Option<EdgeRef>) -> Option<([u32; 3], [u32; 3])> {
    e.map(|e| (bits3(e.a), bits3(e.b)))
}

fn want_edge(v: Option<&Value>) -> Option<([u32; 3], [u32; 3])> {
    opt(v).map(|x| (bits3(p3(&arr(x)[0])), bits3(p3(&arr(x)[1]))))
}

#[test]
fn what_the_hover_acquires() {
    let f = load();
    let cams = arr(f.header.get("cams").unwrap());
    let states = arr(f.header.get("states").unwrap());
    let settings = settings_of(&f.header);
    let text = std::fs::read_to_string(data().join("acquire.jsonl")).unwrap();
    let mut t = Tally::default();
    for line in text.lines() {
        let smp = loads(line).unwrap();
        let name = s(smp.get("s").unwrap());
        let view = {
            let c = &cams[i(smp.get("c").unwrap()) as usize];
            view_of(c)
        };
        let world = &f.scenes[name];
        let st = &states[i(smp.get("t").unwrap()) as usize];
        let tool = tool_of(st);
        let mut state = state_of(st);
        for pxy in arr(smp.get("px").unwrap()) {
            let a = arr(pxy);
            state.acquire(&view, world, &tool, &settings, num(&a[0]), num(&a[1]));
        }
        let mut bad = Vec::new();
        if edge_bits(state.hover_edge) != want_edge(smp.get("he")) {
            bad.push(format!("hover edge {:?} want {:?}", edge_bits(state.hover_edge).map(|e| (f32::from_bits(e.0[0]), f32::from_bits(e.0[1]))), want_edge(smp.get("he")).map(|e| (f32::from_bits(e.0[0]), f32::from_bits(e.0[1])))));
        }
        if edge_bits(state.acquired_edge) != want_edge(smp.get("ae")) {
            bad.push("acquired edge".to_string());
        }
        if state.acquired_point.map(bits3) != op3(smp.get("ap")).map(bits3) {
            bad.push(format!("acquired point {:?} want {:?}", state.acquired_point, op3(smp.get("ap"))));
        }
        if state.acquired_face_normal.map(bits3) != op3(smp.get("afn")).map(bits3) {
            bad.push("acquired face normal".to_string());
        }
        let want_c = |v: Option<&Value>| opt(v).map(|x| (bits3(p3(&arr(x)[0])), i(&arr(x)[1]) as u32));
        if state.hover_center.map(|c| (bits3(c.0), c.1.to_bits())) != want_c(smp.get("hc")) {
            bad.push(format!("hover centre {:?} want {:?}", state.hover_center, want_c(smp.get("hc"))));
        }
        if state.center_ref.map(|c| (bits3(c.0), c.1.to_bits())) != want_c(smp.get("ctr")) {
            bad.push(format!("kept centre {:?} want {:?}", state.center_ref, want_c(smp.get("ctr"))));
        }
        if bad.is_empty() {
            t.ok();
        } else {
            t.fail(format!("{name} t{} {} {:?}: {}", i(smp.get("t").unwrap()), s(st.get("tool").unwrap()), smp.get("px").map(|p| arr(p).len()), bad.join("; ")));
        }
    }
    let msg = t.report("acquire");
    assert!(msg.is_empty(), "{msg}");
    assert!(t.checked > 500, "{}", t.checked);
}
