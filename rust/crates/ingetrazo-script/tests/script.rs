// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! The Rhai engine against the model (M8.1): scripts that draw, transform,
//! group, paint and annotate leave the scene in the L2 signature the same
//! Commands give when executed directly, are undone in one step, and are
//! held by the limits and the sandbox.

use std::time::Duration;

use ingetrazo_golden::{model_sig, sig};
use ingetrazo_math::V3;
use ingetrazo_model::scene::{MeshId, Scene};
use ingetrazo_model::dimension::Dimension;
use ingetrazo_model::textlabel::TextLabel;
use ingetrazo_ops::annotate::{AddDimension, AddTextLabel};
use ingetrazo_ops::command::{Command, Cx, History};
use ingetrazo_ops::draw::AddFaceCommand;
use ingetrazo_ops::edit_cmds::{AssignLayer, Entity, SetFaceColor};
use ingetrazo_ops::edits::{build_add_edge, build_add_edges};
use ingetrazo_ops::grouping::{MakeGroup, RenameGroup};
use ingetrazo_ops::transform::{Factor, MoveGroup, MoveVertices, RotateVertices, ScaleVertices};
use ingetrazo_formats::pyjson::Value;
use ingetrazo_script::{Limits, ScriptEngine};

struct Doc {
    scene: Scene,
    cx: Cx,
    hist: History,
}

impl Doc {
    fn new() -> Doc {
        Doc { scene: Scene::new(), cx: Cx::default(), hist: History::new() }
    }

    fn sig(&self) -> String {
        sig::signature_hash(&model_sig::scene_signature(&self.scene, &sig::Files { zip: None }))
    }

    fn run(&mut self, source: &str) -> Result<ingetrazo_script::Report, ingetrazo_script::ScriptError> {
        ScriptEngine::default().run(&mut self.scene, &mut self.cx, &mut self.hist, source)
    }

    fn direct(&mut self, cmd: Box<dyn Command>) {
        self.hist.execute(&mut self.scene, &mut self.cx, cmd);
        assert!(self.hist.last_error.is_none() && self.hist.last_rejected.is_none(), "{:?} {:?}", self.hist.last_error, self.hist.last_rejected);
    }

    fn undo(&mut self) -> bool {
        self.hist.undo(&mut self.scene, &mut self.cx)
    }
}

fn v(x: f32, y: f32, z: f32) -> V3 {
    V3(x, y, z)
}

const SQUARE: &str = "[vec3(0,0,0), vec3(4,0,0), vec3(4,3,0), vec3(0,3,0)]";

fn square() -> Vec<V3> {
    vec![v(0.0, 0.0, 0.0), v(4.0, 0.0, 0.0), v(4.0, 3.0, 0.0), v(0.0, 3.0, 0.0)]
}

fn draw_square_directly(d: &mut Doc) {
    let p = square();
    let segs: Vec<_> = (0..4).map(|i| (p[i], p[(i + 1) % 4])).collect();
    let cmd = build_add_edges(&d.scene, &d.cx, &segs, true, Vec::new(), None, false);
    d.direct(cmd);
}

#[test]
fn drawing_matches_the_planner_and_undoes_in_one_step() {
    let mut s = Doc::new();
    let empty = s.sig();
    let r = s.run(&format!("add_polygon({SQUARE}); add_edge(vec3(0,0,0), vec3(4,3,0)); face_count()")).unwrap();
    assert_eq!(r.value, "2", "the diagonal splits the square into two faces");
    assert_eq!(s.hist.undo_len(), 1, "a run is one undo step");

    let mut d = Doc::new();
    draw_square_directly(&mut d);
    let cmd = build_add_edge(&d.scene, &d.cx, v(0.0, 0.0, 0.0), v(4.0, 3.0, 0.0), true, None);
    d.direct(cmd);
    assert_eq!(s.sig(), d.sig());

    assert!(s.undo());
    assert_eq!(s.sig(), empty);
    assert!(s.hist.redo(&mut s.scene, &mut s.cx));
    assert_eq!(s.sig(), d.sig());
}

#[test]
fn explicit_faces_paint_and_layers() {
    let mut s = Doc::new();
    let empty = s.sig();
    s.run(&format!(
        "let f = add_face({SQUARE}, #{{color: [0.62, 0.6, 0.58]}});
         set_color(f, 0.8, 0.1, 0.1);
         assign_layer(f, \"Estructura\");
         set_opacity(f, 0.5);"
    ))
    .unwrap();
    assert_eq!(s.hist.undo_len(), 1);

    let mut d = Doc::new();
    let attrs = vec![("color".to_string(), Value::Array(vec![Value::Float(0.62), Value::Float(0.6), Value::Float(0.58)]))];
    let cmd = AddFaceCommand::new(square(), true, None, Some(attrs));
    let face = cmd.face.clone();
    d.direct(Box::new(cmd));
    let f = (MeshId(0), face.get().unwrap());
    d.direct(Box::new(SetFaceColor::new(&[f], Some(vec![Value::Float(0.8), Value::Float(0.1), Value::Float(0.1)]))));
    d.direct(Box::new(AssignLayer::new(vec![Entity::Face(f.0, f.1)], "Estructura")));
    d.direct(Box::new(ingetrazo_ops::edit_cmds::SetFaceOpacity::new(&[f], Some(0.5))));
    assert_eq!(s.sig(), d.sig());

    assert!(s.undo());
    assert_eq!(s.sig(), empty);
}

#[test]
fn move_rotate_and_scale_match_the_vertex_commands() {
    let mut s = Doc::new();
    s.run(&format!(
        "add_polygon({SQUARE});
         translate(faces(), vec3(1, 2, 3));
         rotate(faces(), vec3(0, 0, 0), vec3(0, 0, 1), 90);
         scale(faces(), vec3(0, 0, 0), 2.0);"
    ))
    .unwrap();
    assert_eq!(s.hist.undo_len(), 1);

    let mut d = Doc::new();
    draw_square_directly(&mut d);
    let ring = |d: &Doc| -> Vec<V3> {
        let m = d.scene.loose();
        let f = m.faces[0];
        m.face(f).outer.iter().map(|&x| m.position(x)).collect()
    };
    let ps = ring(&d);
    d.direct(Box::new(MoveVertices::new(&ps, v(1.0, 2.0, 3.0))));
    let ps = ring(&d);
    d.direct(Box::new(RotateVertices::new(&ps, v(0.0, 0.0, 0.0), v(0.0, 0.0, 1.0), 90.0)));
    let ps = ring(&d);
    d.direct(Box::new(ScaleVertices::new(&ps, v(0.0, 0.0, 0.0), Factor::Uniform(2.0), None)));
    assert_eq!(s.sig(), d.sig());
}

#[test]
fn grouping_matches_the_group_commands() {
    let mut s = Doc::new();
    let empty = s.sig();
    let r = s
        .run(&format!(
            "add_polygon({SQUARE});
             let g = make_group(faces(), \"Mesa\");
             translate(g, vec3(5, 0, 0));
             rename_group(g, \"Mesa 2\");
             [g.name, groups().len(), g.faces().len()]"
        ))
        .unwrap();
    assert_eq!(r.value, "[\"Mesa 2\", 1, 1]");
    assert_eq!(s.hist.undo_len(), 1);

    let mut d = Doc::new();
    draw_square_directly(&mut d);
    let f = d.scene.loose().faces[0];
    d.direct(Box::new(MakeGroup::new(&d.scene, &[(MeshId(0), f)], &[], false, Some("Mesa".into()))));
    d.direct(Box::new(MoveGroup::new(&[0], v(5.0, 0.0, 0.0))));
    d.direct(Box::new(RenameGroup::new(&[0], "Mesa 2".into())));
    assert_eq!(s.sig(), d.sig());

    assert!(s.undo());
    assert_eq!(s.sig(), empty);
}

#[test]
fn annotations_match_their_commands() {
    let mut s = Doc::new();
    s.run("add_dimension(vec3(0,0,0), vec3(4,0,0), vec3(0,1,0)); add_text_label([1, 1, 0], \"Sala\");").unwrap();
    let mut d = Doc::new();
    d.direct(Box::new(AddDimension::new(Dimension::new(v(0.0, 0.0, 0.0), v(4.0, 0.0, 0.0), v(0.0, 1.0, 0.0), None, None, None))));
    d.direct(Box::new(AddTextLabel::new(TextLabel { anchor: v(1.0, 1.0, 0.0), offset: v(0.0, 0.0, 0.0), text: "Sala".into(), layer: None })));
    assert_eq!(s.sig(), d.sig());
    let r = s.run("[dimensions().len(), dimensions()[0].length, text_labels()[0].text]").unwrap();
    assert_eq!(r.value, "[1, 4.0, \"Sala\"]");
    assert_eq!(s.hist.undo_len(), 1, "a read-only run adds no undo step");
}

#[test]
fn reading_the_model() {
    let mut s = Doc::new();
    s.run(&format!("add_polygon({SQUARE});")).unwrap();
    let r = s
        .run(
            "let f = faces()[0];
             [faces().len(), edges().len(), vertices().len(), f.area(), f.normal().z.abs(), f.points().len(), f.layer, f.alive,
              edges()[0].length() > 0.0, layers()]",
        )
        .unwrap();
    assert_eq!(r.value, "[1, 4, 4, 12.0, 1.0, 4, \"Layer 0\", true, true, [\"Layer 0\"]]");
}

#[test]
fn a_failed_run_is_rolled_back_whole() {
    let mut s = Doc::new();
    s.run(&format!("add_polygon({SQUARE});")).unwrap();
    s.undo();
    let before = s.sig();
    assert_eq!(s.hist.redo_len(), 1);
    let e = s.run(&format!("add_polygon({SQUARE}); translate(faces(), vec3(1,0,0)); throw \"boom\";")).unwrap_err();
    assert_eq!(e.message, "boom");
    assert_eq!(s.sig(), before, "no half-built geometry");
    assert_eq!(s.hist.undo_len(), 0);
    assert_eq!(s.hist.redo_len(), 1, "the redo stack survives a failed run");
}

#[test]
fn a_read_only_run_keeps_redo() {
    let mut s = Doc::new();
    s.run(&format!("add_polygon({SQUARE});")).unwrap();
    s.undo();
    s.run("print(face_count()); face_count()").unwrap();
    assert_eq!(s.hist.redo_len(), 1);
    let r = s.run("print(\"hola\"); 1 + 1").unwrap();
    assert_eq!((r.output.as_str(), r.value.as_str(), r.commands), ("hola\n", "2", 0));
}

#[test]
fn stale_handles_stop_the_script() {
    let mut s = Doc::new();
    let e = s
        .run(&format!("add_polygon({SQUARE}); let f = faces()[0]; erase(f); f.area()"))
        .unwrap_err();
    assert!(e.message.contains("stale handle"), "{e}");
    assert_eq!(s.hist.undo_len(), 0);
    assert_eq!(s.scene.loose().faces.len(), 0);
}

#[test]
fn erase_and_flip() {
    let mut s = Doc::new();
    s.run(&format!("add_polygon({SQUARE}); let n = faces()[0].normal().z; flip_faces(faces()); if faces()[0].normal().z == n {{ throw \"same\" }}")).unwrap();
    s.run("erase(faces()); face_count()").map(|r| assert_eq!(r.value, "0")).unwrap();
}

#[test]
fn selection_is_what_the_host_set() {
    let mut s = Doc::new();
    s.run(&format!("add_polygon({SQUARE});")).unwrap();
    let f = s.scene.loose().faces[0];
    let mut eng = ScriptEngine::default();
    eng.set_selection(vec![Entity::Face(MeshId(0), f)]);
    let r = eng.run(&mut s.scene, &mut s.cx, &mut s.hist, "set_color(selection(), 1.0, 0.0, 0.0); faces()[0].color").unwrap();
    assert_eq!(r.value, "[1.0, 0.0, 0.0]");
}

// ---- limits and sandbox ----------------------------------------------------

#[test]
fn an_infinite_loop_stops() {
    let mut s = Doc::new();
    let before = s.sig();
    let e = ScriptEngine::new(Limits { max_operations: 10_000, ..Limits::default() })
        .run(&mut s.scene, &mut s.cx, &mut s.hist, &format!("add_polygon({SQUARE}); loop {{ }}"))
        .unwrap_err();
    assert!(e.message.to_lowercase().contains("operations"), "{e}");
    assert_eq!(s.sig(), before, "and the run is rolled back");
}

#[test]
fn the_wall_clock_stops_a_slow_script() {
    let mut s = Doc::new();
    let mut eng = ScriptEngine::new(Limits { max_operations: u64::MAX, timeout: Some(Duration::from_millis(50)), ..Limits::default() });
    let e = eng.run(&mut s.scene, &mut s.cx, &mut s.hist, "let i = 0; loop { i += 1; }").unwrap_err();
    assert!(e.message.contains("out of time"), "{e}");
}

#[test]
fn runaway_recursion_stops() {
    let mut s = Doc::new();
    let e = s.run("fn f(n) { f(n + 1) } f(0)").unwrap_err();
    assert!(e.message.to_lowercase().contains("stack") || e.message.to_lowercase().contains("call"), "{e}");
}

#[test]
fn sizes_and_command_counts_are_capped() {
    let mut s = Doc::new();
    let mut eng = ScriptEngine::new(Limits { max_array_size: 100, max_commands: 3, ..Limits::default() });
    let e = eng.run(&mut s.scene, &mut s.cx, &mut s.hist, "let a = []; for i in 0..1000 { a.push(i); }").unwrap_err();
    assert!(e.message.to_lowercase().contains("array"), "{e}");
    let e = eng
        .run(&mut s.scene, &mut s.cx, &mut s.hist, "for i in 0..10 { add_edge(vec3(i, 0, 0), vec3(i, 1, 0)); }")
        .unwrap_err();
    assert!(e.message.contains("too many commands"), "{e}");
    assert_eq!(s.scene.loose().edges.len(), 0);
}

#[test]
fn there_is_no_filesystem_network_or_eval() {
    let mut s = Doc::new();
    for code in [
        "import \"std/fs\" as fs; 1",
        "eval(\"1 + 1\")",
        "open(\"C:/secrets.txt\")",
        "read_file(\"/etc/passwd\")",
        "File::open(\"x\")",
        "http_get(\"http://example.com\")",
        "system(\"calc\")",
    ] {
        assert!(s.run(code).is_err(), "{code} must not run");
    }
    assert_eq!(s.hist.undo_len(), 0);
}

#[test]
fn syntax_errors_name_the_line() {
    let mut s = Doc::new();
    let e = s.run("let a = 1;\nlet b = ;").unwrap_err();
    assert_eq!(e.line, Some(2), "{e}");
}

// ---- the showcase ----------------------------------------------------------

#[test]
fn the_pavilion_showcase_builds_in_one_undoable_step() {
    let mut s = Doc::new();
    let empty = s.sig();
    let r = s.run(include_str!("../examples/pavilion.rhai")).unwrap();
    assert!(r.output.contains("Pabellon listo"), "{}", r.output);
    assert_eq!(s.hist.undo_len(), 1);
    // slab, 4 columns, 2 walls, roof: 8 groups; the glazed front stays loose.
    assert_eq!(s.scene.groups.len(), 8);
    assert_eq!(s.scene.loose().faces.len(), 1);
    assert!(s.undo());
    assert_eq!(s.sig(), empty);
    assert!(s.hist.redo(&mut s.scene, &mut s.cx));
    assert_eq!(s.scene.groups.len(), 8);
}
