// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! The script console (M8.2) headless: one undo step per run, rollback with
//! line and column, a session that remembers, history, completion and
//! cancellation.

use std::sync::atomic::Ordering;
use std::time::Duration;

use ingetrazo_model::scene::Scene;
use ingetrazo_ops::command::{Cx, History};
use ingetrazo_script::console::{CompletionKind, Console, Role};
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
    fn faces(&self) -> usize {
        self.cx.mesh(&self.scene).faces.len()
    }
}

fn submit(c: &mut Console, d: &mut Doc, text: &str) -> ingetrazo_script::console::Submission {
    c.submit(&mut d.scene, &mut d.cx, &mut d.hist, text)
}

const SQUARE: &str = "add_face([vec3(0,0,0), vec3(4,0,0), vec3(4,3,0), vec3(0,3,0)])";

#[test]
fn a_run_is_one_undo_step_and_undo_restores() {
    let mut c = Console::default();
    let mut d = Doc::new();
    let s = submit(&mut c, &mut d, &format!("{SQUARE};\n{SQUARE};\nprint(\"two\");"));
    assert!(s.ok, "{:?}", s.error);
    assert!(s.changed);
    assert_eq!(s.output, "two\n");
    assert_eq!(d.hist.undo_len(), 1);
    assert!(d.faces() >= 1);
    assert!(d.hist.undo(&mut d.scene, &mut d.cx));
    assert_eq!(d.faces(), 0);
}

#[test]
fn echo_output_and_value_lines_carry_roles() {
    let mut c = Console::default();
    let mut d = Doc::new();
    let s = submit(&mut c, &mut d, "print(\"hi\");\n1 + 2");
    let roles: Vec<(Role, &str)> = s.lines.iter().map(|l| (l.role, l.text.as_str())).collect();
    assert_eq!(roles, vec![(Role::Echo, ">>> print(\"hi\");"), (Role::Echo, "... 1 + 2"), (Role::Output, "hi"), (Role::Value, "3")]);
    // The transcript has the welcome plus these.
    assert!(c.transcript().len() >= 4 + 3);
    c.clear_transcript();
    assert_eq!(c.transcript().len(), 3);
}

#[test]
fn inspecting_leaves_no_undo_entry_and_keeps_redo() {
    let mut c = Console::default();
    let mut d = Doc::new();
    submit(&mut c, &mut d, SQUARE);
    d.hist.undo(&mut d.scene, &mut d.cx);
    assert_eq!(d.hist.redo_len(), 1);
    let s = submit(&mut c, &mut d, "face_count()");
    assert!(s.ok && !s.changed);
    assert_eq!(d.hist.undo_len(), 0);
    assert_eq!(d.hist.redo_len(), 1, "an inspection must not eat the redo stack");
}

#[test]
fn an_error_rolls_back_whole_and_reports_line_and_column() {
    let mut c = Console::default();
    let mut d = Doc::new();
    let s = submit(&mut c, &mut d, &format!("{SQUARE};\nlet x = 1;\nfoo_bar(x);"));
    assert!(!s.ok && !s.cancelled);
    let e = s.error.expect("error");
    assert_eq!(e.line, Some(3));
    assert!(e.column.is_some());
    assert_eq!(d.faces(), 0, "half-built geometry must not survive");
    assert_eq!(d.hist.undo_len(), 0);
    let text = s.lines.iter().find(|l| l.role == Role::Error).unwrap().text.clone();
    assert!(text.starts_with("line 3, column "), "{text}");
    assert!(text.contains("3 | foo_bar(x);"), "{text}");
    assert!(text.contains('^'), "{text}");
    assert!(s.lines.iter().any(|l| l.role == Role::Muted && l.text.contains("rolled back")));
}

#[test]
fn syntax_errors_have_positions_too() {
    let mut c = Console::default();
    let mut d = Doc::new();
    let s = submit(&mut c, &mut d, "let a = 1;\nlet = ;");
    let e = s.error.expect("error");
    assert_eq!(e.line, Some(2));
    assert!(e.column.is_some());
}

#[test]
fn print_before_an_error_is_kept() {
    let mut c = Console::default();
    let mut d = Doc::new();
    let s = submit(&mut c, &mut d, "print(\"before\"); nope();");
    assert!(!s.ok);
    assert!(s.lines.iter().any(|l| l.role == Role::Output && l.text == "before"));
}

#[test]
fn variables_and_functions_persist_between_submissions() {
    let mut c = Console::default();
    let mut d = Doc::new();
    assert!(submit(&mut c, &mut d, "let w = 4.0; fn twice(x) { x * 2.0 }").ok);
    let s = submit(&mut c, &mut d, "twice(w)");
    assert!(s.ok, "{:?}", s.error);
    assert_eq!(s.lines.last().unwrap().text, "8.0");
    let names: Vec<String> = c.completions("w").into_iter().filter(|x| x.kind == CompletionKind::Variable).map(|x| x.name).collect();
    assert_eq!(names, vec!["w"]);
    // A failed run leaves the session as it was.
    assert!(!submit(&mut c, &mut d, "let w = 99.0; boom();").ok);
    assert_eq!(submit(&mut c, &mut d, "w").lines.last().unwrap().text, "4.0");
    c.reset_session();
    assert!(!submit(&mut c, &mut d, "w").ok);
}

#[test]
fn history_walks_back_and_forth_keeping_the_draft() {
    let mut c = Console::default();
    let mut d = Doc::new();
    assert_eq!(c.history_prev("x"), None);
    submit(&mut c, &mut d, "1");
    submit(&mut c, &mut d, "2");
    submit(&mut c, &mut d, "2"); // a repeat is not stored twice
    assert_eq!(c.history(), ["1", "2"]);
    assert_eq!(c.history_prev("half typed"), Some("2"));
    assert_eq!(c.history_prev("ignored"), Some("1"));
    assert_eq!(c.history_prev(""), None);
    assert_eq!(c.history_next(), Some("2"));
    assert_eq!(c.history_next(), Some("half typed"));
    assert_eq!(c.history_next(), None);
}

#[test]
fn multi_line_input_is_recognised() {
    assert!(!Console::is_complete("for i in 0..3 {"));
    assert!(!Console::is_complete("for i in 0..3 {\n  print(i);"));
    assert!(Console::is_complete("for i in 0..3 {\n  print(i);\n}"));
    let mut c = Console::default();
    let mut d = Doc::new();
    let s = submit(&mut c, &mut d, "let n = 0;\nfor i in 0..3 {\n  n += i;\n}\nn");
    assert_eq!(s.lines.last().unwrap().text, "3");
}

#[test]
fn completions_come_from_the_registered_api() {
    let mut c = Console::default();
    let f = c.completions("add_p");
    let names: Vec<&str> = f.iter().map(|x| x.name.as_str()).collect();
    assert_eq!(names, ["add_polygon", "add_polyline"]);
    assert_eq!(f[0].kind, CompletionKind::Function);
    assert!(f[0].signature.starts_with("add_polygon(array)"), "{}", f[0].signature);
    let m = c.completions("are");
    assert_eq!((m[0].name.as_str(), m[0].kind), ("area", CompletionKind::Method));
    assert!(c.completions("col").iter().any(|x| x.name == "color" && x.kind == CompletionKind::Property));
    assert!(c.completions("wh").iter().any(|x| x.name == "while" && x.kind == CompletionKind::Keyword));
    assert!(c.completions("zzz").is_empty());
}

#[test]
fn every_registered_function_is_in_the_reference() {
    let mut e = ScriptEngine::default();
    let reference = ingetrazo_script::api::REFERENCE;
    let mut missing = Vec::new();
    for sig in e.signatures() {
        let name = sig.split('(').next().unwrap();
        let name = name.trim_start_matches("get$").trim_start_matches("set$");
        if !name.chars().next().is_some_and(|c| c.is_alphabetic()) || matches!(name, "to_string" | "to_debug") {
            continue; // operators and the printing glue
        }
        if !reference.contains(name) {
            missing.push(name.to_string());
        }
    }
    missing.sort();
    missing.dedup();
    assert!(missing.is_empty(), "documented nowhere in api_reference.md: {missing:?}");
}

#[test]
fn the_cancel_token_stops_a_long_script_and_rolls_back() {
    let limits = Limits { max_operations: 0, timeout: None, ..Limits::default() };
    let mut c = Console::new(limits);
    let mut d = Doc::new();
    let token = c.cancel_token();
    let t = std::thread::spawn(move || {
        std::thread::sleep(Duration::from_millis(100));
        token.store(true, Ordering::Relaxed);
    });
    let s = submit(&mut c, &mut d, &format!("{SQUARE}; loop {{ }}"));
    t.join().unwrap();
    assert!(!s.ok);
    assert!(s.cancelled);
    assert_eq!(d.faces(), 0);
    // The token is cleared for the next run.
    assert!(submit(&mut c, &mut d, "1 + 1").ok);
}

#[test]
fn run_script_labels_the_file_and_is_not_in_the_history() {
    let mut c = Console::default();
    let mut d = Doc::new();
    let s = c.run_script(&mut d.scene, &mut d.cx, &mut d.hist, "pavilion.rhai", "print(\"ran\");");
    assert!(s.ok);
    assert_eq!(s.lines[0].role, Role::Run);
    assert!(c.history().is_empty());
}

#[test]
fn the_sandbox_has_no_import_or_eval() {
    let mut c = Console::default();
    let mut d = Doc::new();
    assert!(!submit(&mut c, &mut d, "import \"x\" as y;").ok);
    assert!(!submit(&mut c, &mut d, "eval(\"1\")").ok);
}
