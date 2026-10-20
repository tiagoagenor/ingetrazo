// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! The Rust renderer against pictures the Python viewport (OpenGL, MSAA 4x)
//! drew from the same scenes (`scripts/golden/make_render_fixtures.py`).
//!
//! The corpus rule for images (golden/tolerances.toml `render_pixels`) is
//! "every channel within 2, for 99.5 % of the pixels". Each case is compared
//! twice:
//!
//! * **faces only** (no edges, profiles, axes or guides): triangles are
//!   rasterised the same way by every GPU, so the corpus rule applies as
//!   written and is asserted;
//! * **full** (with the line work): a 1 px GL line and a 1 px wgpu line do
//!   not land on the same samples (NVIDIA's GL snaps lines to the pixel
//!   corner grid, wgpu draws exact rectangles), so a few percent of the
//!   pixels differ along every edge. The full comparison asserts a floor
//!   measured on the reference machine and prints the numbers.
//!
//! The differences are saved next to the renders (`target/render-out`).
#![allow(clippy::chunks_exact_to_as_chunks)]
mod common;

use std::path::{Path, PathBuf};

use common::*;
use ingetrazo_formats::pyjson::{loads, Value};
use ingetrazo_math::V3;
use ingetrazo_render::document::open_igz;
use ingetrazo_render::offscreen::Image;
use ingetrazo_render::{Camera, Style};

/// The `render_pixels` rule of golden/tolerances.toml.
const CHANNEL: u8 = 2;
const FRACTION_EQUAL: f64 = 0.995;
/// What the line work may cost: a floor on the full comparison. Measured on
/// an RTX 3060 (98.4-99.9 % on Vulkan, 97.2-99.0 % on DX12 for the
/// synthetic scenes) and set under the DX12 numbers, whose lines differ a
/// little more from the OpenGL ones.
const FULL_FLOOR: f64 = 0.96;

/// The floor of the faces-only comparison: the corpus rule, except where the
/// scene has a reason (each documented):
///
/// * `translucent`: blended faces stack their silhouettes, each layer
///   repeating the MSAA coverage difference of its outline;
/// * `arco_yanque`: a real model of thousands of small faces has several
///   times the silhouette length of the synthetic scenes;
/// * `arco_yanque_shaded`: the Python viewport has a state leak in the lean
///   instanced pass of the Shaded style: after a textured run the
///   back-face colour uniform keeps that texture's average colour, so the
///   rear of an unpainted textured face in an instanced component shows it
///   instead of the back tint. The Rust renderer draws the tint the code
///   documents, and the arch's spandrel wall (seen from behind) differs.
fn faces_floor(id: &str) -> f64 {
    match id {
        "translucent" => 0.994,
        // Texture filtering is not specified to the bit: the Microsoft
        // software adapter (WARP) lands at 98.45 %, the RTX at 99.68 %.
        "textured" => 0.98,
        // The horizon of the backdrop is a 1 px line: 99.76 % on Vulkan, 99.10 %
        // on D3D12, where lines are rasterised another way.
        "blocks_default_sky" => 0.99,
        "arco_yanque" => 0.99,
        "arco_yanque_shaded" => 0.975,
        _ => FRACTION_EQUAL,
    }
}

/// The floor of the comparison with the line work (see the module docs).
fn full_floor(id: &str) -> f64 {
    match id {
        // A real model: lines everywhere (see `faces_floor`).
        "arco_yanque" => 0.93,
        "arco_yanque_shaded" => 0.92,
        _ => FULL_FLOOR,
    }
}

fn data() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data")
}

fn num(v: &Value) -> f64 {
    match v {
        Value::Float(f) => *f,
        Value::Int(s) => s.parse().unwrap(),
        _ => panic!("number expected"),
    }
}

fn camera_of(v: &Value, w: u32, h: u32) -> Camera {
    let t = match v.get("target") {
        Some(Value::Array(a)) => V3(num(&a[0]) as f32, num(&a[1]) as f32, num(&a[2]) as f32),
        _ => panic!("target"),
    };
    let flag = |k: &str| matches!(v.get(k), Some(Value::Bool(true)));
    let mut c = Camera {
        target: t,
        distance: num(v.get("distance").unwrap()),
        yaw: num(v.get("yaw").unwrap()),
        pitch: num(v.get("pitch").unwrap()),
        perspective: flag("perspective"),
        two_point: flag("two_point"),
        ..Default::default()
    };
    c.set_aspect(w, h);
    c
}

/// A visible diff: grey where equal, red-scaled where not.
fn diff_image(a: &Image, b: &Image) -> Image {
    let mut rgba = Vec::with_capacity(a.rgba.len());
    for (p, q) in a.rgba.chunks_exact(4).zip(b.rgba.chunks_exact(4)) {
        let d = (0..3).map(|k| p[k].abs_diff(q[k])).max().unwrap();
        if d <= CHANNEL {
            rgba.extend_from_slice(&[235, 235, 235, 255]);
        } else {
            rgba.extend_from_slice(&[255, 255u8.saturating_sub(d.saturating_mul(3)), 0, 255]);
        }
    }
    Image { width: a.width, height: a.height, rgba }
}

struct Outcome {
    id: String,
    full: f64,
    full_max: u8,
    faces: Option<(f64, u8)>,
}

fn run_case(dev: &ingetrazo_render::offscreen::Offscreen, case: &Value, w: u32, h: u32, view_h: f32) -> Outcome {
    let text = |k: &str| match case.get(k) {
        Some(Value::Str(s)) => s.clone(),
        _ => panic!("{k}"),
    };
    let (id, style_name) = (text("id"), text("style"));
    let faces_only = matches!(case.get("faces_only"), Some(Value::Bool(true)));
    // A case may name a real document of the repository instead of its own.
    let igz_path = match case.get("igz") {
        Some(Value::Str(rel)) => Path::new(env!("CARGO_MANIFEST_DIR")).join("../../..").join(rel),
        _ => data().join(format!("{id}.igz")),
    };
    let raw = std::fs::read(&igz_path).unwrap_or_else(|e| panic!("{igz_path:?}: {e}"));
    let (scene, mut tex) = open_igz(&raw).unwrap_or_else(|e| panic!("{id}: {e}"));
    let cam = camera_of(case.get("camera").unwrap(), w, h);
    let style = Style::builtin(&style_name).unwrap();
    let mut r = new_renderer(dev);

    let mut go = |name: &str, opts: &ingetrazo_render::RenderOptions| -> (f64, u8) {
        let (img, _) = dev.render_image(&mut r, &scene, &mut tex, &cam, opts, w, h);
        let reference = Image::read_png(&data().join(format!("{id}{name}.png"))).unwrap();
        let cmp = img.compare(&reference, CHANNEL);
        img.write_png(&out_dir().join(format!("parity-{id}{name}-rust.png"))).unwrap();
        diff_image(&img, &reference).write_png(&out_dir().join(format!("parity-{id}{name}-diff.png"))).unwrap();
        if std::env::var("INGETRAZO_PARITY_LIST").is_ok_and(|v| v == format!("{id}{name}")) {
            let mut n = 0;
            for y in 0..h {
                for x in 0..w {
                    let (a, b) = (img.pixel(x, y), reference.pixel(x, y));
                    let d = (0..3).map(|k| a[k].abs_diff(b[k])).max().unwrap();
                    if d > CHANNEL && n < 60 {
                        println!("  ({x},{y}) rust={:?} ref={:?}", &a[..3], &b[..3]);
                        n += 1;
                    }
                }
            }
        }
        (cmp.fraction_equal, cmp.max_diff)
    };

    let full_opts = ingetrazo_render::RenderOptions { style: Some(style.clone()), logical_height: Some(view_h), ..Default::default() };
    let (full, full_max) = go("", &full_opts);
    let faces = faces_only.then(|| {
        let lineless = Style { edges: false, profiles: false, back_edges: false, ..style.clone() };
        let opts = ingetrazo_render::RenderOptions { style: Some(lineless), overlays: false, logical_height: Some(view_h), ..Default::default() };
        go(".faces", &opts)
    });
    Outcome { id, full, full_max, faces }
}

/// The camera matrices are the Python ones to the last bit (the Rust
/// `Camera` is a port of `OrbitCamera`, Qt's float32 arithmetic included).
#[test]
fn camera_matrices_are_bit_exact() {
    let meta = loads(&std::fs::read_to_string(data().join("render_cases.json")).unwrap()).unwrap();
    let (w, h) = match meta.get("size") {
        Some(Value::Array(a)) => (num(&a[0]) as u32, num(&a[1]) as u32),
        _ => panic!("size"),
    };
    let Some(Value::Array(cases)) = meta.get("cases") else { panic!("cases") };
    for case in cases {
        let cam = camera_of(case.get("camera").unwrap(), w, h);
        let id = format!("{:?}", case.get("id"));
        let floats = |k: &str| -> Vec<f32> {
            match case.get(k) {
                Some(Value::Array(a)) => a.iter().map(|v| num(v) as f32).collect(),
                _ => panic!("{k}"),
            }
        };
        let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&cam.view_matrix().data()), bits(&floats("view")), "{id} view");
        assert_eq!(bits(&cam.projection_matrix().data()), bits(&floats("projection")), "{id} projection");
        assert_eq!(bits(&cam.mvp().data()), bits(&floats("mvp")), "{id} mvp");
    }
}

#[test]
fn rust_frames_against_the_python_viewport() {
    let Some(dev) = device() else { return };
    let meta = loads(&std::fs::read_to_string(data().join("render_cases.json")).unwrap()).unwrap();
    let (w, h) = match meta.get("size") {
        Some(Value::Array(a)) => (num(&a[0]) as u32, num(&a[1]) as u32),
        _ => panic!("size"),
    };
    // The widget height the section cut lines were sized by.
    let view_h = meta.get("view_height").map(num).unwrap_or(h as f64) as f32;
    let Some(Value::Array(cases)) = meta.get("cases") else { panic!("cases") };
    let outcomes: Vec<Outcome> = cases.iter().map(|c| run_case(&dev, c, w, h, view_h)).collect();
    println!("adapter: {} ({:?})", dev.info.name, dev.info.backend);
    println!("{:<26} {:>16} {:>16}", "case", "faces (+-2, max)", "full (+-2, max)");
    let mut failures = Vec::new();
    for o in &outcomes {
        let faces = o
            .faces
            .map(|(f, m)| format!("{:>8.3}% {:>4}", f * 100.0, m))
            .unwrap_or_else(|| format!("{:>14}", "-"));
        println!("{:<26} {:>16} {:>8.3}% {:>4}", o.id, faces, o.full * 100.0, o.full_max);
        if let Some((f, _)) = o.faces {
            if f < faces_floor(&o.id) {
                failures.push(format!("{} faces: {:.3}% < {:.1}%", o.id, f * 100.0, faces_floor(&o.id) * 100.0));
            }
        }
        if o.full < full_floor(&o.id) {
            failures.push(format!("{} full: {:.3}% < {:.1}%", o.id, o.full * 100.0, full_floor(&o.id) * 100.0));
        }
    }
    assert!(failures.is_empty(), "parity failures:\n{}", failures.join("\n"));
}
