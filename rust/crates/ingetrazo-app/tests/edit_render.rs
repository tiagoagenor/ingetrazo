// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! What the user edits reaches the picture: a box drawn through the session
//! is drawn by the viewport's renderer, and gone from it after undo, back
//! after redo. Skipped when the machine has no wgpu adapter
//! (INGETRAZO_RENDER_FALLBACK=1 asks for the software one).


use ingetrazo_app::nav;
use ingetrazo_app::session::{Btn, Input, Session};
use ingetrazo_app::viewport::GpuView;
use ingetrazo_math::V3;
use ingetrazo_render::offscreen::Offscreen;
use ingetrazo_tools::base::KEY_RETURN;

fn text(s: &mut Session, t: &str) {
    for c in t.chars() {
        s.handle(&Input::Text { text: c.to_string(), mods: 0 });
    }
}

fn click(s: &mut Session, p: V3) {
    s.sync();
    let (x, y) = s.to_pixel(p).unwrap();
    s.handle(&Input::Move { x, y, mods: 0 });
    s.handle(&Input::Press { button: Btn::Left, x, y, mods: 0 });
    s.handle(&Input::Release { button: Btn::Left, x, y, mods: 0 });
}

fn triangles(dev: &Offscreen, view: &mut GpuView, s: &mut Session) -> u32 {
    s.take_scene_changed();
    let (w, h) = (320u32, 200u32);
    let mut encoder = dev.device.create_command_encoder(&Default::default());
    view.prepare(&dev.device, &mut encoder, s.scene(), &s.cam, (w, h));
    view.stats.triangles as u32
}

#[test]
fn an_edit_is_drawn_and_its_undo_and_redo_are_too() {
    let Some(dev) = Offscreen::new() else {
        eprintln!("SKIP: no wgpu adapter on this machine");
        return;
    };
    let mut s = Session::new();
    s.set_size(320, 200);
    nav::set_view(&mut s.cam, "iso");
    s.cam.distance = 12.0;
    let mut view = GpuView::new(&dev.device, &dev.queue, eframe::egui_wgpu::wgpu::TextureFormat::Rgba8Unorm);
    assert_eq!(triangles(&dev, &mut view, &mut s), 0, "an empty model draws no triangles");

    text(&mut s, "r");
    click(&mut s, V3(0.0, 0.0, 0.0));
    text(&mut s, "2;1");
    s.handle(&Input::Key { key: KEY_RETURN, mods: 0, pressed: true });
    let flat = triangles(&dev, &mut view, &mut s);
    assert!(flat > 0, "the rectangle is drawn");

    text(&mut s, "p");
    click(&mut s, V3(1.0, 0.5, 0.0));
    text(&mut s, "1");
    s.handle(&Input::Key { key: KEY_RETURN, mods: 0, pressed: true });
    let boxed = triangles(&dev, &mut view, &mut s);
    assert!(boxed > flat, "the pushed box has more faces: {boxed} vs {flat}");

    assert!(s.undo());
    assert_eq!(triangles(&dev, &mut view, &mut s), flat, "undo shows the rectangle again");
    assert!(s.undo());
    assert_eq!(triangles(&dev, &mut view, &mut s), 0, "and then nothing");
    while s.redo() {}
    assert_eq!(triangles(&dev, &mut view, &mut s), boxed, "redo draws the box again");
}
