// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! The application end to end, with no window (M4.5 to M4.8): egui events -
//! the ones a frame of the real window would carry - go through the same
//! translator the window uses, into the session, into the tools of
//! `ingetrazo-tools`; then the model is read back, undone, redone, saved
//! and opened again.

use eframe::egui::{Event, Key, Modifiers, PointerButton, Pos2, Rect};
use ingetrazo_app::input::Translator;
use ingetrazo_app::nav;
use ingetrazo_app::session::Session;
use ingetrazo_app::toolbox::ToolId;
use ingetrazo_formats::igz;
use ingetrazo_math::V3;

/// A window with nobody's hand on it yet: a `Session` seen from above, and
/// the translator that feeds it.
struct Sim {
    s: Session,
    t: Translator,
    rect: Rect,
    time: f64,
    mods: Modifiers,
}

impl Sim {
    fn new() -> Sim {
        let mut s = Session::new();
        s.set_size(1000, 700);
        nav::set_view(&mut s.cam, "top");
        s.cam.distance = 14.0;
        s.cam.target = V3(2.0, 1.0, 0.0);
        s.sync();
        Sim { s, t: Translator::new(), rect: Rect::from_min_size(Pos2::ZERO, eframe::egui::vec2(1000.0, 700.0)), time: 0.0, mods: Modifiers::NONE }
    }

    /// One frame carrying `events`.
    fn frame(&mut self, events: Vec<Event>) {
        self.time += 1.0;
        for input in self.t.translate(&events, self.mods, self.time, self.rect, true) {
            self.s.handle(&input);
        }
    }

    fn px(&mut self, p: V3) -> Pos2 {
        self.s.sync();
        let (x, y) = self.s.to_pixel(p).expect("the point is in front of the camera");
        Pos2::new(x as f32, y as f32)
    }

    fn hover(&mut self, p: V3) {
        let pos = self.px(p);
        self.frame(vec![Event::PointerMoved(pos)]);
    }

    fn click(&mut self, p: V3) {
        let pos = self.px(p);
        let mods = self.mods;
        let b = |pressed| Event::PointerButton { pos, button: PointerButton::Primary, pressed, modifiers: mods };
        self.frame(vec![Event::PointerMoved(pos), b(true)]);
        self.frame(vec![b(false)]);
    }

    /// Drag the left button from one point to another (the Select box).
    fn drag(&mut self, a: V3, b: V3) {
        let (pa, pb) = (self.px(a), self.px(b));
        let mods = self.mods;
        let ev = |pos, pressed| Event::PointerButton { pos, button: PointerButton::Primary, pressed, modifiers: mods };
        self.frame(vec![Event::PointerMoved(pa), ev(pa, true)]);
        self.frame(vec![Event::PointerMoved(Pos2::new((pa.x + pb.x) / 2.0, (pa.y + pb.y) / 2.0))]);
        self.frame(vec![Event::PointerMoved(pb)]);
        self.frame(vec![ev(pb, false)]);
    }

    fn type_text(&mut self, text: &str) {
        for ch in text.chars() {
            self.frame(vec![Event::Text(ch.to_string())]);
        }
    }

    fn key(&mut self, key: Key) {
        let mods = self.mods;
        let ev = |pressed| Event::Key { key, physical_key: None, pressed, repeat: false, modifiers: mods };
        self.frame(vec![ev(true), ev(false)]);
    }

    /// Ctrl+Z, as the window runs it.
    fn undo(&mut self) -> bool {
        self.s.undo()
    }

    fn redo(&mut self) -> bool {
        self.s.redo()
    }

    /// The corners of everything the loose mesh holds.
    fn bounds(&self) -> ([f32; 3], [f32; 3]) {
        let m = self.s.scene().mesh(ingetrazo_model::scene::MeshId(0));
        let mut lo = [f32::INFINITY; 3];
        let mut hi = [f32::NEG_INFINITY; 3];
        for &v in &m.vertices {
            let p = m.position(v);
            for (k, c) in [p.0, p.1, p.2].into_iter().enumerate() {
                lo[k] = lo[k].min(c);
                hi[k] = hi[k].max(c);
            }
        }
        (lo, hi)
    }

    fn faces(&self) -> usize {
        self.s.scene().mesh(ingetrazo_model::scene::MeshId(0)).faces.len()
    }
}

fn near(a: f32, b: f32) -> bool {
    (a - b).abs() < 1e-4
}

/// The Rust document text of the model, as a save would write it.
fn text_of(s: &mut Session, path: &std::path::Path) -> String {
    s.save_to(path).expect("the model saves");
    igz::document_text(&std::fs::read(path).unwrap()).unwrap()
}

#[test]
fn a_box_is_drawn_pushed_moved_undone_redone_saved_and_opened() {
    let dir = std::env::temp_dir().join(format!("ingetrazo-session-flow-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let mut sim = Sim::new();
    assert!(!sim.s.dirty());

    // A rectangle: the first corner on the origin, the size typed.
    sim.s.handle(&ingetrazo_app::session::Input::Text { text: "r".into(), mods: 0 });
    assert_eq!(sim.s.tool, ToolId::Rectangle, "R picks the Rectangle");
    sim.click(V3(0.0, 0.0, 0.0));
    sim.type_text("2;1");
    assert_eq!(sim.s.host.core.value_buffer, "2;1", "the value box takes the digits");
    sim.key(Key::Enter);
    assert_eq!(sim.faces(), 1, "the rectangle is a face");
    let (lo, hi) = sim.bounds();
    assert!(near(hi[0] - lo[0], 2.0) && near(hi[1] - lo[1], 1.0), "{lo:?} {hi:?}");
    assert!(sim.s.dirty(), "drawing dirties the document");
    assert!(sim.s.title().ends_with(" *"), "{}", sim.s.title());

    // Push/Pull the face a metre (1000 mm typed).
    sim.s.handle(&ingetrazo_app::session::Input::Text { text: "p".into(), mods: 0 });
    assert_eq!(sim.s.tool, ToolId::PushPull);
    sim.hover(V3(1.0, 0.5, 0.0));
    sim.click(V3(1.0, 0.5, 0.0));
    sim.type_text("1000mm");
    sim.key(Key::Enter);
    assert_eq!(sim.faces(), 6, "the face became a box");
    let (lo, hi) = sim.bounds();
    assert!(near(hi[2] - lo[2], 1.0), "a metre tall: {lo:?} {hi:?}");

    // Select everything with a window, then Move it 3 m along X.
    sim.s.handle(&ingetrazo_app::session::Input::Text { text: " ".into(), mods: 0 });
    assert_eq!(sim.s.tool, ToolId::Select, "Space picks Select");
    sim.drag(V3(-1.0, 3.0, 0.0), V3(5.0, -2.0, 0.0));
    assert!(!sim.s.host.core.selection.is_empty(), "the box selected the model");
    sim.s.handle(&ingetrazo_app::session::Input::Text { text: "m".into(), mods: 0 });
    assert_eq!(sim.s.tool, ToolId::Move);
    let (before_lo, _) = sim.bounds();
    sim.click(V3(0.0, 0.0, 0.0));
    sim.hover(V3(3.0, 0.0, 0.0));
    sim.type_text("3");
    sim.key(Key::Enter);
    let (lo, hi) = sim.bounds();
    assert!(near(lo[0] - before_lo[0], 3.0), "moved 3 m along X: {before_lo:?} -> {lo:?}");
    assert!(near(hi[0] - lo[0], 2.0) && near(hi[1] - lo[1], 1.0) && near(hi[2] - lo[2], 1.0), "still the same box: {lo:?} {hi:?}");
    assert!(near(lo[1], 0.0) && near(lo[2], 0.0), "along X only: {lo:?}");
    let after_move = (lo, hi);

    // The model as it stands, written.
    let first = dir.join("box.igz");
    let doc_a = text_of(&mut sim.s, &first);
    assert!(!sim.s.dirty(), "a save cleans the document");
    assert!(!sim.s.title().ends_with('*'));

    // Undo everything: nothing is left and the document is as new.
    sim.s.set_tool(ToolId::Select);
    let mut steps = 0;
    while sim.undo() {
        steps += 1;
        assert!(steps < 10, "more undo steps than commands");
    }
    assert_eq!(steps, 3, "rectangle, push, move");
    assert_eq!(sim.faces(), 0);
    assert!(sim.s.host.core.selection.is_empty(), "what is gone leaves the selection");
    assert!(sim.s.dirty(), "undone past the save: dirty again");
    // And redo everything: the very same model.
    while sim.redo() {}
    assert!(!sim.s.dirty(), "redone back to the save: clean");
    assert_eq!(sim.bounds(), after_move);
    let second = dir.join("box-redone.igz");
    assert_eq!(text_of(&mut sim.s, &second), doc_a, "redo restores the very same document");

    // Open it in another session: the same model, and it writes the same file.
    let mut again = Session::new();
    again.set_size(1000, 700);
    again.open(&first).expect("the saved file opens");
    assert!(!again.dirty());
    assert_eq!(again.title(), "IngeTrazo \u{2014} box.igz");
    again.cam = sim.s.cam;
    let third = dir.join("box-again.igz");
    assert_eq!(text_of(&mut again, &third), doc_a, "save, open, save is the identity");

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn undo_in_the_middle_of_a_tool_steps_the_tool_back_first() {
    let mut sim = Sim::new();
    sim.s.handle(&ingetrazo_app::session::Input::Text { text: "l".into(), mods: 0 });
    sim.click(V3(0.0, 0.0, 0.0));
    sim.click(V3(2.0, 0.0, 0.0));
    sim.click(V3(2.0, 2.0, 0.0));
    let mesh = |sim: &Sim| sim.s.scene().mesh(ingetrazo_model::scene::MeshId(0)).edges.len();
    assert_eq!(mesh(&sim), 2, "two segments of the chain");
    // Ctrl+Z steps the chain back one vertex; the line tool is still busy.
    assert!(sim.undo());
    assert_eq!(mesh(&sim), 1);
    assert!(sim.s.host.tool().busy(), "the chain goes on from the previous vertex");
    // Esc ends the chain; undo then takes the last segment.
    sim.key(Key::Escape);
    assert!(!sim.s.host.tool().busy());
    assert!(sim.undo());
    assert_eq!(mesh(&sim), 0);
    assert!(!sim.undo(), "nothing left");
    assert!(!sim.s.dirty());
}

#[test]
fn shortcuts_pick_the_tools_and_digits_are_the_value_box_not_a_shortcut() {
    let mut sim = Sim::new();
    for (ch, tool) in [('l', ToolId::Line), ('r', ToolId::Rectangle), ('c', ToolId::Circle), ('a', ToolId::Arc), ('p', ToolId::PushPull), ('m', ToolId::Move), ('q', ToolId::Rotate), ('s', ToolId::Scale), ('f', ToolId::Offset), ('e', ToolId::Eraser), ('b', ToolId::Paint), ('t', ToolId::Tape), (' ', ToolId::Select)] {
        sim.type_text(&ch.to_string());
        assert_eq!(sim.s.tool, tool, "{ch:?}");
    }
    sim.type_text("O");
    assert_eq!(sim.s.tool, ToolId::CenterArc, "Shift+O");
    sim.type_text("H");
    assert_eq!(sim.s.tool, ToolId::Protractor, "Shift+H");

    // After the first click of a rectangle, "m" typed after a digit is the
    // unit of the value, not the Move tool.
    sim.type_text("r");
    sim.click(V3(0.0, 0.0, 0.0));
    sim.type_text("2m");
    assert_eq!(sim.s.tool, ToolId::Rectangle);
    assert_eq!(sim.s.host.core.value_buffer, "2m");
    // Esc clears the typed text first, then cancels the rectangle.
    sim.key(Key::Escape);
    assert_eq!(sim.s.host.core.value_buffer, "");
    assert!(sim.s.host.tool().busy());
    sim.key(Key::Escape);
    assert!(!sim.s.host.tool().busy());
}

#[test]
fn the_status_bar_says_what_to_do_and_the_value_box_what_it_holds() {
    let mut sim = Sim::new();
    sim.type_text("r");
    assert!(sim.s.status_line().starts_with("Rectangle - Click the first corner"), "{}", sim.s.status_line());
    let (caption, value, typing) = sim.s.vcb().expect("the Rectangle has a value box");
    assert_eq!((caption.as_str(), value.as_str(), typing), ("Dimensions", "", false));
    sim.click(V3(0.0, 0.0, 0.0));
    assert!(sim.s.status_line().contains("opposite corner"), "{}", sim.s.status_line());
    sim.hover(V3(3.0, 2.0, 0.0));
    let (_, live, _) = sim.s.vcb().unwrap();
    assert!(!live.is_empty(), "the live size shows while the cursor moves");
    sim.type_text("4");
    let (_, typed, typing) = sim.s.vcb().unwrap();
    assert_eq!((typed.as_str(), typing), ("4", true));
    // Select has no value box (Esc clears the typed text and ends the rectangle first).
    sim.key(Key::Escape);
    sim.key(Key::Escape);
    sim.type_text(" ");
    assert!(sim.s.vcb().is_none());
}

#[test]
fn the_example_opens_saves_and_reopens_to_the_same_file() {
    let dir = std::env::temp_dir().join(format!("ingetrazo-session-example-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let example = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../examples/arco-yanque.igz");
    let mut s = Session::new();
    s.set_size(800, 600);
    s.open(&example).expect("the example opens");
    assert!(!s.dirty() && s.path.is_some());
    assert!(!s.can_undo(), "a document just opened has no history");
    let a = dir.join("a.igz");
    let text_a = text_of(&mut s, &a);
    let mut t = Session::new();
    t.set_size(800, 600);
    t.open(&a).unwrap();
    t.cam = s.cam;
    let b = dir.join("b.igz");
    assert_eq!(text_of(&mut t, &b), text_a, "open, save, open, save: the same document");
    let _ = std::fs::remove_dir_all(&dir);
}

/// A 2 x 1 x 1 box standing on the origin, drawn and pushed through the UI.
fn box_scene(sim: &mut Sim) {
    sim.type_text("r");
    sim.click(V3(0.0, 0.0, 0.0));
    sim.type_text("2;1");
    sim.key(Key::Enter);
    sim.type_text("p");
    sim.hover(V3(1.0, 0.5, 0.0));
    sim.click(V3(1.0, 0.5, 0.0));
    sim.type_text("1");
    sim.key(Key::Enter);
    sim.type_text(" ");
    assert_eq!(sim.faces(), 6);
}

#[test]
fn every_tool_takes_the_pointer_the_keys_and_shows_its_preview() {
    let mut sim = Sim::new();
    box_scene(&mut sim);
    sim.s.cam.pitch = 35.0f64.to_radians();
    sim.s.cam.yaw = (-60.0f64).to_radians();
    let spots = [V3(0.0, 0.0, 0.0), V3(2.0, 0.0, 1.0), V3(1.0, 0.5, 1.0), V3(2.0, 1.0, 0.5), V3(4.0, 2.0, 0.0), V3(-1.0, 1.0, 0.0)];
    for id in ingetrazo_app::toolbox::all() {
        sim.s.set_tool(id);
        assert_eq!(sim.s.tool, id);
        for p in spots {
            sim.hover(p);
            sim.click(p);
            sim.hover(V3(p.0 + 0.5, p.1 + 0.25, p.2));
            let _ = (sim.s.overlay(), sim.s.status_line(), sim.s.vcb(), sim.s.render_cues());
        }
        sim.type_text("12");
        sim.key(Key::Enter);
        sim.key(Key::Escape);
        sim.key(Key::Escape);
        sim.type_text("0");
        sim.key(Key::Escape);
        // Whatever the tool did, the model still undoes to nothing and redoes.
        let before = sim.faces();
        let mut n = 0;
        while sim.undo() {
            n += 1;
            assert!(n < 400, "{id:?}: undo never ends");
        }
        while sim.redo() {}
        assert_eq!(sim.faces(), before, "{id:?}: redo after undo-all");
    }
}

#[test]
fn a_circle_takes_its_sides_and_radius_from_the_value_box() {
    let mut sim = Sim::new();
    sim.type_text("c");
    sim.type_text("8s");
    sim.key(Key::Enter);
    sim.click(V3(0.0, 0.0, 0.0));
    sim.type_text("1");
    sim.key(Key::Enter);
    let m = sim.s.scene().mesh(ingetrazo_model::scene::MeshId(0));
    assert_eq!(m.faces.len(), 1, "an 8-sided disc");
    assert_eq!(m.edges.len(), 8);
    let (lo, hi) = sim.bounds();
    assert!(hi[0] - lo[0] > 1.8 && hi[0] - lo[0] < 2.0 + 1e-4, "a radius of 1 m: {lo:?} {hi:?}");
}

#[test]
fn push_pull_in_the_middle_of_a_drag_is_cancelled_by_undo_and_commits_on_the_second_click() {
    let mut sim = Sim::new();
    sim.type_text("r");
    sim.click(V3(0.0, 0.0, 0.0));
    sim.type_text("2;1");
    sim.key(Key::Enter);
    sim.s.cam.pitch = 40.0f64.to_radians();
    sim.s.cam.yaw = (-60.0f64).to_radians();
    sim.type_text("p");
    sim.hover(V3(1.0, 0.5, 0.0));
    sim.click(V3(1.0, 0.5, 0.0));
    assert!(sim.s.host.tool().busy(), "the drag has begun");
    sim.hover(V3(1.0, 0.5, 1.0));
    assert!(!sim.s.overlay().rubber.is_empty(), "the sweep is previewed");
    assert!(sim.s.overlay().label.is_some(), "with its distance");
    // Ctrl+Z abandons the drag and leaves the rectangle.
    assert!(sim.undo());
    assert!(!sim.s.host.tool().busy());
    assert_eq!(sim.faces(), 1);
    // Drag again and click to commit: the second click is the commit.
    sim.hover(V3(1.0, 0.5, 0.0));
    sim.click(V3(1.0, 0.5, 0.0));
    sim.hover(V3(1.0, 0.5, 1.0));
    sim.click(V3(1.0, 0.5, 1.0));
    assert_eq!(sim.faces(), 6, "the second click committed the push");
}

#[test]
fn the_eraser_and_paint_and_select_change_the_model_through_the_pointer() {
    let mut sim = Sim::new();
    box_scene(&mut sim);
    sim.s.cam.pitch = 40.0f64.to_radians();
    sim.s.cam.yaw = (-60.0f64).to_radians();
    // Select picks a face; a click on empty space clears it again.
    sim.hover(V3(1.0, 0.5, 1.0));
    sim.click(V3(1.0, 0.5, 1.0));
    assert!(!sim.s.host.core.selection.is_empty(), "the face under the cursor is selected");
    sim.click(V3(6.0, 6.0, 0.0));
    assert!(sim.s.host.core.selection.is_empty(), "a click on nothing clears the selection");
    // Paint colours the face: one command.
    sim.type_text("b");
    let depth = sim.s.host.core.history.undo_len();
    sim.hover(V3(1.0, 0.5, 1.0));
    sim.click(V3(1.0, 0.5, 1.0));
    assert_eq!(sim.s.host.core.history.undo_len(), depth + 1, "painting is one undo step");
    // The eraser takes an edge away.
    sim.type_text("e");
    let edges = sim.s.scene().mesh(ingetrazo_model::scene::MeshId(0)).edges.len();
    sim.hover(V3(2.0, 0.5, 0.0));
    sim.click(V3(2.0, 0.5, 0.0));
    assert!(sim.s.scene().mesh(ingetrazo_model::scene::MeshId(0)).edges.len() < edges, "an edge was erased");
}

/// What the Python app wrote (`formats/igz.py`, in every format and with
/// textures): the application's open and its save write the same bytes.
#[test]
fn documents_python_wrote_are_written_back_byte_for_byte() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../golden/starts");
    let mut checked = 0;
    for entry in std::fs::read_dir(&dir).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().is_none_or(|e| e != "igz") {
            continue;
        }
        let raw = std::fs::read(&path).unwrap();
        let (doc, _textures) = ingetrazo_app::document::open_path(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        let mut scene = (*doc.scene).clone();
        let written = ingetrazo_app::document::write_bytes(&mut scene, &doc.raw).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        let name = path.file_name().unwrap().to_string_lossy().into_owned();
        // The files were written by the app version of the day.
        let version = igz::document_text(&raw).unwrap();
        if !version.contains(&format!("\"app_version\":\"{}\"", ingetrazo_app::document::APP_VERSION)) && !version.contains(&format!("\"app_version\": \"{}\"", ingetrazo_app::document::APP_VERSION)) {
            eprintln!("skipped {name}: written by another app version");
            continue;
        }
        assert!(written.bytes == raw, "{name}: the save is not what Python wrote ({} vs {} bytes)", written.bytes.len(), raw.len());
        checked += 1;
    }
    assert!(checked >= 100, "only {checked} documents were of this app version");
}
