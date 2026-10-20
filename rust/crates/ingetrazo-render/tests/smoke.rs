// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
#![allow(clippy::chunks_exact_to_as_chunks)]
mod common;

use common::*;
use ingetrazo_render::{Camera, NoTextures};

#[test]
fn a_cube_renders_and_saves_a_png() {
    let Some(dev) = device() else { return };
    let mut sc = cube_scene();
    paint_all(&mut sc.meshes[0], [0.8, 0.5, 0.3]);
    let mut r = new_renderer(&dev);
    r.sync(&sc, &mut NoTextures, &Default::default());
    let cam = view_of(&sc, &mut r);
    for name in ["Default", "Shaded", "Hidden line", "Monochrome", "Wireframe", "X-ray"] {
        let img = render(&dev, &mut r, &sc, &cam, &style(name), 400, 300);
        img.write_png(&out_dir().join(format!("cube-{}.png", name.replace(' ', "_")))).unwrap();
    }
    let _ = Camera::default();
}
