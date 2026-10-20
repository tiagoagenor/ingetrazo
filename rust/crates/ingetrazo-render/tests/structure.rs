// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! Structural invariants of the renderer: what is baked and when, the shape
//! of a frame per style, depth order, transparency, instancing, selection,
//! the edit-context fade, section clipping. The command list
//! (`Renderer::plan`) is data, so most of these need no GPU; the ones that
//! read pixels skip on a machine with no adapter.
#![allow(clippy::chunks_exact_to_as_chunks)]
mod common;

use common::*;
use ingetrazo_formats::pyjson::Value;
use ingetrazo_math::{M4, V3};
use ingetrazo_model::scene::{Group, MeshId, Scene};
use ingetrazo_render::frame::{ColMode, Geo};
use ingetrazo_render::offscreen::Image;
use ingetrazo_render::{Camera, EditRest, NoTextures, RenderOptions, RendererConfig, Selection, Style};

fn opts_with(style: &str) -> RenderOptions {
    RenderOptions { style: Some(Style::builtin(style).unwrap()), overlays: false, ..Default::default() }
}

fn px(img: &Image, x: u32, y: u32) -> [u8; 3] {
    let p = img.pixel(x, y);
    [p[0], p[1], p[2]]
}

fn near(a: [u8; 3], b: [u8; 3], tol: u8) -> bool {
    (0..3).all(|k| a[k].abs_diff(b[k]) <= tol)
}

/// A scene with a unit cube prototype (mesh 1) and `n` instances of it.
fn instanced(n: usize) -> Scene {
    let mut sc = Scene::new();
    let mut proto = ingetrazo_model::mesh::Mesh::new();
    cube_into(&mut proto, V3(0.0, 0.0, 0.0), 1.0);
    sc.meshes.push(proto);
    for i in 0..n {
        sc.groups.push(instance(MeshId(1), translation(i as f32 * 2.0, 0.0, 0.0), &format!("c{i}")));
    }
    sc
}

#[test]
fn a_component_is_baked_once_however_many_placements() {
    let Some(dev) = device() else { return };
    let mut r = new_renderer(&dev);
    let draws = |n: usize, r: &mut ingetrazo_render::Renderer| {
        let sc = instanced(n);
        let cam = Camera::default();
        let plan = r.plan(&sc, &mut NoTextures, &cam, (64, 64), &opts_with("Hidden line"));
        let face_draws = plan.cmds().filter(|c| matches!(c.geo, Geo::Faces(_))).count();
        (plan.triangle_count(), face_draws, r.prototype_stats())
    };
    let (t5, d5, stats5) = draws(5, &mut r);
    r.invalidate();
    let (t50, d50, stats50) = draws(50, &mut r);
    assert_eq!(t5, 12 * 5);
    assert_eq!(t50, 12 * 50, "triangles submitted count every instance");
    assert_eq!(d5, d50, "the number of draw calls does not grow with the placements");
    // The loose mesh (empty) and the one prototype: two bakes, 12 triangles
    // and 12 edges in the prototype whatever the count.
    assert_eq!(stats5, stats50);
    assert_eq!(stats50[1], (12, 12, 0));
}

#[test]
fn only_what_changed_is_rebaked() {
    let Some(dev) = device() else { return };
    let mut r = new_renderer(&dev);
    let mut sc = instanced(3);
    cube_into(&mut sc.meshes[0], V3(0.0, 5.0, 0.0), 1.0);
    let o = RenderOptions::default();
    let first = r.sync(&sc, &mut NoTextures, &o);
    assert_eq!(first.rebuilt, 2, "loose mesh and prototype");
    let again = r.sync(&sc, &mut NoTextures, &o);
    assert_eq!(again.rebuilt, 0, "nothing moved, nothing rebaked");
    assert_eq!(again.uploaded_bytes, 4 * 80, "only the instance table is rewritten");
    // Edit the loose mesh: only it is rebaked.
    cube_into(&mut sc.meshes[0], V3(0.0, 9.0, 0.0), 1.0);
    assert_eq!(r.sync(&sc, &mut NoTextures, &o).rebuilt, 1);
    assert_eq!(r.prototype_stats()[0].0, 24);
    // Edit the prototype: every instance follows, with one rebake.
    cube_into(&mut sc.meshes[1], V3(3.0, 3.0, 0.0), 1.0);
    assert_eq!(r.sync(&sc, &mut NoTextures, &o).rebuilt, 1);
    assert_eq!(r.prototype_stats()[1].0, 24);
    // Moving a placement rebakes nothing.
    sc.groups[0].xform = Some(translation(0.0, 0.0, 5.0));
    assert_eq!(r.sync(&sc, &mut NoTextures, &o).rebuilt, 0);
}

#[test]
fn each_style_draws_what_it_promises() {
    let Some(dev) = device() else { return };
    let mut r = new_renderer(&dev);
    let mut sc = cube_scene();
    paint_all(&mut sc.meshes[0], [0.8, 0.5, 0.3]);
    let cam = Camera::default();
    let mut plan_of = |name: &str| r.plan(&sc, &mut NoTextures, &cam, (64, 64), &opts_with(name));
    let modes = |p: &ingetrazo_render::frame::FramePlan| -> Vec<(ColMode, bool, bool)> {
        p.cmds()
            .filter(|c| matches!(c.geo, Geo::Faces(_)))
            .map(|c| {
                let v0 = p.params[c.slot as usize].v0;
                let m = match v0[1] as u32 {
                    0 => ColMode::Vertex,
                    1 => ColMode::Texture,
                    2 => ColMode::Flat,
                    _ => ColMode::FlatTint,
                };
                (m, c.pipe.blend, c.pipe.depth_write)
            })
            .collect()
    };
    // Wireframe: edges only.
    let wf = plan_of("Wireframe");
    assert!(!wf.cmds().any(|c| matches!(c.geo, Geo::Faces(_))));
    assert!(wf.cmds().any(|c| matches!(c.geo, Geo::Hard(_))));
    // Hidden line: flat, opaque, depth written; the front colour is white.
    let hl = plan_of("Hidden line");
    assert_eq!(modes(&hl), vec![(ColMode::Flat, false, true)]);
    let slot = hl.cmds().find(|c| matches!(c.geo, Geo::Faces(_))).unwrap().slot;
    assert_eq!(hl.params[slot as usize].color, [1.0, 1.0, 1.0, 1.0]);
    // Monochrome: flat as well, with the back tint apart from the front.
    let mono = plan_of("Monochrome");
    assert_eq!(modes(&mono), vec![(ColMode::Flat, false, true)]);
    let slot = mono.cmds().find(|c| matches!(c.geo, Geo::Faces(_))).unwrap().slot;
    assert_ne!(mono.params[slot as usize].color, mono.params[slot as usize].back_color);
    // Textures / Shaded: vertex colours, opaque.
    assert_eq!(modes(&plan_of("Default")), vec![(ColMode::Vertex, false, true)]);
    assert_eq!(modes(&plan_of("Shaded")), vec![(ColMode::Vertex, false, true)]);
    // X-ray: blended, no depth writes, and its edges in two depth passes.
    let xr = plan_of("X-ray");
    assert_eq!(modes(&xr)[0], (ColMode::Vertex, true, false));
    assert_eq!(xr.passes.len(), 2, "the depth is cleared between the edges and the selection cues");
    let hidden_edges = xr.cmds().filter(|c| matches!(c.geo, Geo::Hard(_))).count();
    assert_eq!(hidden_edges, 2, "faded behind the faces, then in front");
    // The sky is part of Default and absent from the Architectural preset.
    assert!(plan_of("Default").cmds().any(|c| matches!(c.geo, Geo::SkyTris)));
    assert!(!plan_of("Architectural").cmds().any(|c| matches!(c.geo, Geo::SkyTris)));
}

#[test]
fn the_nearer_box_wins_whatever_the_draw_order() {
    let Some(dev) = device() else { return };
    // Looking along -x from +x, in parallel projection: the box at x = 3 is
    // in front of the one at x = 0.
    let build = |near_first: bool| {
        let mut sc = Scene::new();
        let mut a = ingetrazo_model::mesh::Mesh::new();
        let mut b = ingetrazo_model::mesh::Mesh::new();
        cube_into(&mut a, V3(0.0, -0.5, -0.5), 1.0);
        cube_into(&mut b, V3(3.0, -0.5, -0.5), 1.0);
        paint_all(&mut a, [1.0, 0.0, 0.0]);
        paint_all(&mut b, [0.0, 0.0, 1.0]);
        let (ga, gb) = (Group::new(MeshId(1), "far".into()), Group::new(MeshId(2), "near".into()));
        sc.meshes.push(a);
        sc.meshes.push(b);
        if near_first {
            sc.groups.extend([gb, ga]);
        } else {
            sc.groups.extend([ga, gb]);
        }
        sc
    };
    let cam = Camera { target: V3(2.0, 0.0, 0.0), distance: 10.0, yaw: 0.0, pitch: 0.0, perspective: false, ..Default::default() };
    for order in [true, false] {
        let sc = build(order);
        let mut r = new_renderer(&dev);
        let img = render(&dev, &mut r, &sc, &cam, &opts_with("Shaded"), 128, 128);
        let c = px(&img, 64, 64);
        assert!(c[2] > 150 && c[0] < 60, "blue box in front, got {c:?} (near drawn first: {order})");
    }
}

#[test]
fn translucent_faces_blend_the_same_in_any_group_order() {
    let Some(dev) = device() else { return };
    let build = |swap: bool| {
        let mut sc = Scene::new();
        let mut a = ingetrazo_model::mesh::Mesh::new();
        let mut b = ingetrazo_model::mesh::Mesh::new();
        cube_into(&mut a, V3(0.0, -0.25, -0.25), 0.5);
        cube_into(&mut b, V3(2.0, -0.25, -0.25), 0.5);
        for m in [&mut a, &mut b] {
            let ids: Vec<_> = m.faces.clone();
            for f in ids {
                m.face_mut(f).attrs.push(("opacity".into(), Value::Float(0.5)));
            }
        }
        paint_all(&mut a, [1.0, 0.0, 0.0]);
        paint_all(&mut b, [0.0, 0.0, 1.0]);
        sc.meshes.push(a);
        sc.meshes.push(b);
        let (ga, gb) = (Group::new(MeshId(1), "far".into()), Group::new(MeshId(2), "near".into()));
        if swap {
            sc.groups.extend([gb, ga]);
        } else {
            sc.groups.extend([ga, gb]);
        }
        sc
    };
    let cam = Camera { target: V3(1.0, 0.0, 0.0), distance: 10.0, yaw: 0.0, pitch: 0.0, perspective: false, ..Default::default() };
    let mut imgs = Vec::new();
    for swap in [false, true] {
        let sc = build(swap);
        let mut r = new_renderer(&dev);
        imgs.push(render(&dev, &mut r, &sc, &cam, &opts_with("Shaded"), 128, 128));
    }
    let (a, b) = (px(&imgs[0], 64, 64), px(&imgs[1], 64, 64));
    assert!(near(a, b, 1), "{a:?} vs {b:?}");
    // The nearer (x = 2) blue layer is on top: more blue than red.
    assert!(a[2] > a[0], "{a:?}");
}

#[test]
fn a_mirrored_instance_looks_like_the_mirrored_geometry() {
    let Some(dev) = device() else { return };
    // Instance of a painted cube through scale(-1, 1, 1) ...
    let mut inst = Scene::new();
    let mut proto = ingetrazo_model::mesh::Mesh::new();
    cube_into(&mut proto, V3(1.0, 0.0, 0.0), 1.0);
    paint_all(&mut proto, [0.9, 0.5, 0.2]);
    inst.meshes.push(proto);
    let mut m = M4::identity();
    m.scale(-1.0, 1.0, 1.0);
    inst.groups.push(instance(MeshId(1), m, "mirrored"));
    // ... against the same cube built already mirrored (x in -2..-1).
    let mut plain = Scene::new();
    cube_into(&mut plain.meshes[0], V3(-2.0, 0.0, 0.0), 1.0);
    paint_all(&mut plain.meshes[0], [0.9, 0.5, 0.2]);
    let cam = Camera { target: V3(-1.5, 0.5, 0.5), distance: 8.0, yaw: -0.7, pitch: 0.5, ..Default::default() };
    let mut r1 = new_renderer(&dev);
    let mut r2 = new_renderer(&dev);
    let a = render(&dev, &mut r1, &inst, &cam, &opts_with("Shaded"), 128, 128);
    let b = render(&dev, &mut r2, &plain, &cam, &opts_with("Shaded"), 128, 128);
    let cmp = a.compare(&b, 2);
    assert!(cmp.fraction_equal > 0.995, "mirrored instance: {cmp:?}");
    // And no back tint shows through the faces (the faces face outward).
    let tint = [(0.62f32 * 255.0) as u8, (0.70f32 * 255.0) as u8, (0.78f32 * 255.0) as u8];
    for y in 0..128 {
        for x in 0..128 {
            assert!(!near(px(&a, x, y), tint, 3) || near(px(&b, x, y), tint, 3), "back tint at {x},{y}");
        }
    }
}

#[test]
fn back_faces_show_the_tint_and_two_sided_paint_does_not() {
    let Some(dev) = device() else { return };
    let build = |two_sided: bool| {
        let mut sc = Scene::new();
        let f = sc.meshes[0].add_face(&[V3(-1.0, 0.0, -1.0), V3(1.0, 0.0, -1.0), V3(1.0, 0.0, 1.0), V3(-1.0, 0.0, 1.0)], &[]);
        let mut attrs = vec![("color".to_string(), Value::Array(vec![Value::Float(1.0), Value::Float(1.0), Value::Float(0.0)]))];
        if two_sided {
            attrs.push(("back".to_string(), Value::Bool(true)));
        }
        sc.meshes[0].face_mut(f).attrs = attrs;
        sc
    };
    // The face normal is -y (winding seen from -y is CCW): look from +y at its back.
    let cam = Camera { target: V3(0.0, 0.0, 0.0), distance: 6.0, yaw: std::f64::consts::FRAC_PI_2, pitch: 0.0, perspective: false, ..Default::default() };
    let front_cam = Camera { yaw: -std::f64::consts::FRAC_PI_2, ..cam };
    let mut r = new_renderer(&dev);
    let one = build(false);
    let back = render(&dev, &mut r, &one, &cam, &opts_with("Shaded"), 64, 64);
    let front = render(&dev, &mut r, &one, &front_cam, &opts_with("Shaded"), 64, 64);
    let tint = px(&back, 32, 32);
    assert!(near(tint, [158, 178, 199], 3), "the default back tint, got {tint:?}");
    let f = px(&front, 32, 32);
    assert!(f[0] > 150 && f[2] < 60, "the painted side is yellow-ish: {f:?}");
    let two = build(true);
    let both = render(&dev, &mut r, &two, &cam, &opts_with("Shaded"), 64, 64);
    let c = px(&both, 32, 32);
    assert!(c[0] > 150 && c[2] < 60, "two-sided paint shows the paint behind: {c:?}");
    // Monochrome tints every back; Hidden line none.
    let mono = render(&dev, &mut r, &two, &cam, &opts_with("Monochrome"), 64, 64);
    assert!(near(px(&mono, 32, 32), [255, 255, 255], 3) || near(px(&mono, 32, 32), tint, 3));
}

#[test]
fn hidden_layers_and_hidden_faces_leave_the_frame() {
    let Some(dev) = device() else { return };
    let mut r = new_renderer(&dev);
    let mut sc = cube_scene();
    let cam = Camera::default();
    let tris = |r: &mut ingetrazo_render::Renderer, sc: &Scene| r.plan(sc, &mut NoTextures, &cam, (32, 32), &opts_with("Shaded")).triangle_count();
    assert_eq!(tris(&mut r, &sc), 12);
    // One face hidden.
    let f = sc.meshes[0].faces[0];
    sc.meshes[0].face_mut(f).attrs.push(("hidden".into(), Value::Bool(true)));
    assert_eq!(tris(&mut r, &sc), 10);
    // The hidden face comes back as a ghost with View > Hidden Geometry.
    sc.extra.show_hidden_geometry = true;
    let plan = r.plan(&sc, &mut NoTextures, &cam, (32, 32), &opts_with("Shaded"));
    assert_eq!(plan.triangle_count(), 12, "10 solid + 2 ghost");
    sc.extra.show_hidden_geometry = false;
    // A layer switched off takes its faces with it.
    let f1 = sc.meshes[0].faces[1];
    sc.meshes[0].face_mut(f1).attrs.push(("layer".into(), Value::Str("Walls".into())));
    sc.layers.push(Value::Object(vec![
        ("name".into(), Value::Str("Walls".into())),
        ("visible".into(), Value::Bool(false)),
    ]));
    assert_eq!(tris(&mut r, &sc), 8);
    // Hidden groups are skipped; shown as ghosts with Hidden Objects.
    let mut g = Group::new(MeshId(0), "h".into());
    g.hidden = true;
    sc.groups.push(g);
    let before = tris(&mut r, &sc);
    sc.extra.show_hidden_objects = true;
    let after = tris(&mut r, &sc);
    assert!(after > before, "the hidden object appears as a ghost ({before} -> {after})");
}

#[test]
fn selection_and_hover_paint_the_orange_and_the_blue() {
    let Some(dev) = device() else { return };
    let sc = cube_scene();
    let mut r = new_renderer(&dev);
    r.sync(&sc, &mut NoTextures, &Default::default());
    let cam = view_of(&sc, &mut r);
    let plain = render(&dev, &mut r, &sc, &cam, &opts_with("Hidden line"), 160, 120);
    let mut sel = opts_with("Hidden line");
    let edge = sc.meshes[0].edges[0];
    sel.selection = Selection { edges: vec![(vec![], edge)], faces: vec![(vec![], sc.meshes[0].faces[1])], groups: vec![] };
    let marked = render(&dev, &mut r, &sc, &cam, &sel, 160, 120);
    let orange = |p: [u8; 3]| p[0] > 200 && p[1] > 90 && p[1] < 150 && p[2] < 90;
    let count = |img: &Image| (0..120).flat_map(|y| (0..160).map(move |x| (x, y))).filter(|&(x, y)| orange(px(img, x, y))).count();
    assert_eq!(count(&plain), 0);
    assert!(count(&marked) > 10, "the selected edge and the dotted face are orange");
    // The same selection never reaches an export.
    let mut exp = sel.clone();
    exp.export = Some(ingetrazo_render::ExportLines { edge_px: 1, profile_px: 1 });
    let exported = render(&dev, &mut r, &sc, &cam, &exp, 160, 120);
    assert_eq!(count(&exported), 0);
    // Hover: a light-blue edge.
    let mut hov = opts_with("X-ray");
    hov.hover.edge = Some((vec![], edge));
    let hovered = render(&dev, &mut r, &sc, &cam, &hov, 160, 120);
    let blue = |p: [u8; 3]| p[2] > 200 && p[2] as i32 > p[0] as i32 + 40 && p[2] as i32 > p[1] as i32 + 25;
    assert!((0..120).flat_map(|y| (0..160).map(move |x| (x, y))).any(|(x, y)| blue(px(&hovered, x, y))));
}

#[test]
fn the_context_of_an_open_group_is_washed_toward_grey() {
    let Some(dev) = device() else { return };
    let mut sc = Scene::new();
    let mut a = ingetrazo_model::mesh::Mesh::new();
    let mut b = ingetrazo_model::mesh::Mesh::new();
    cube_into(&mut a, V3(-3.0, -0.5, -0.5), 1.0);
    cube_into(&mut b, V3(2.0, -0.5, -0.5), 1.0);
    paint_all(&mut a, [1.0, 0.0, 0.0]);
    paint_all(&mut b, [1.0, 0.0, 0.0]);
    sc.meshes.push(a);
    sc.meshes.push(b);
    sc.groups.push(Group::new(MeshId(1), "context".into()));
    sc.groups.push(Group::new(MeshId(2), "subject".into()));
    let cam = Camera { target: V3(0.0, 0.0, 0.0), distance: 12.0, yaw: -std::f64::consts::FRAC_PI_2, pitch: 0.2, perspective: false, ..Default::default() };
    let mut r = new_renderer(&dev);
    let mut o = opts_with("Shaded");
    o.edit_path = Some(vec![1]);
    let img = render(&dev, &mut r, &sc, &cam, &o, 200, 100);
    // Screen x grows with world x for this camera: the subject is on the right.
    let ctx = px(&img, 100 - 3 * 100 / 8 - 6, 50);
    let sub = px(&img, 100 + 2 * 100 / 8 + 12, 50);
    let _ = (ctx, sub);
    let mut counts = [0usize; 2];
    for y in 0..100 {
        for x in 0..200 {
            let p = px(&img, x, y);
            if p[0] > 150 && p[1] < 60 {
                counts[1] += 1; // saturated red: the subject
            } else if p[0] > 150 && p[1] > 100 && p[1] < 160 && p[2] > 90 && p[2] < 160 {
                counts[0] += 1; // red washed 75 % toward grey: the context
            }
        }
    }
    assert!(counts[0] > 30 && counts[1] > 30, "faded context and full-colour subject: {counts:?}");
    // Hiding the rest removes the context altogether.
    o.edit_rest = EditRest::Hide;
    let hidden = r.plan(&sc, &mut NoTextures, &cam, (200, 100), &o);
    assert_eq!(hidden.triangle_count(), 12);
}

#[test]
fn a_section_plane_clips_the_model_and_caps_it() {
    let Some(dev) = device() else { return };
    let mut sc = cube_scene();
    paint_all(&mut sc.meshes[0], [0.8, 0.8, 0.2]);
    // Keep y <= 1 (the normal side, +y, is hidden).
    sc.section_planes.push(Value::Object(vec![
        ("point".into(), Value::Array(vec![Value::Float(0.0), Value::Float(1.0), Value::Float(0.0)])),
        ("normal".into(), Value::Array(vec![Value::Float(0.0), Value::Float(1.0), Value::Float(0.0)])),
        ("active".into(), Value::Bool(true)),
    ]));
    // Looking along +y from y = -10 (yaw = -90 deg), parallel: the cut cap
    // is not facing us; look from +y instead to see the cut face.
    let cam = Camera { target: V3(1.0, 1.0, 1.0), distance: 10.0, yaw: std::f64::consts::FRAC_PI_2, pitch: 0.0, perspective: false, ..Default::default() };
    let mut r = new_renderer(&dev);
    let img = render(&dev, &mut r, &sc, &cam, &opts_with("Shaded"), 128, 128);
    // The section fill colour (solid, the style's grey) is what we see at the centre.
    let c = px(&img, 64, 64);
    let fill = Style::default().section_fill_color.map(|v| (v * 255.0).round() as u8);
    assert!(near(c, fill, 3), "the cut area is filled: {c:?} vs {fill:?}");
    // Without the plane we see the cube's yellow back-side face (y = 2).
    sc.section_planes.clear();
    let whole = render(&dev, &mut r, &sc, &cam, &opts_with("Shaded"), 128, 128);
    assert!(px(&whole, 64, 64)[2] < 100);
    // With the cuts switched off the plane does not clip.
    sc.section_planes.push(Value::Object(vec![
        ("point".into(), Value::Array(vec![Value::Float(0.0), Value::Float(1.0), Value::Float(0.0)])),
        ("normal".into(), Value::Array(vec![Value::Float(0.0), Value::Float(1.0), Value::Float(0.0)])),
        ("active".into(), Value::Bool(true)),
    ]));
    sc.extra.cuts_hidden = true;
    let off = render(&dev, &mut r, &sc, &cam, &opts_with("Shaded"), 128, 128);
    assert!(near(px(&off, 64, 64), px(&whole, 64, 64), 2));
}

#[test]
fn overlay_lines_float_or_hide_behind_the_model() {
    let Some(dev) = device() else { return };
    let mut sc = cube_scene();
    paint_all(&mut sc.meshes[0], [0.9, 0.9, 0.9]);
    // A segment through the cube's middle, seen from +x in parallel: it is
    // inside the cube, so a depth-tested line hides and a floating one does not.
    let cam = Camera { target: V3(1.0, 1.0, 1.0), distance: 10.0, yaw: 0.0, pitch: 0.0, perspective: false, ..Default::default() };
    let line = |depth_test: bool| {
        let mut o = opts_with("Shaded");
        o.overlay_lines.push(ingetrazo_render::OverlayLine {
            a: V3(1.0, 0.0, 1.0),
            b: V3(1.0, 2.0, 1.0),
            color: [1.0, 0.0, 0.0],
            depth_test,
        });
        o
    };
    let mut r = new_renderer(&dev);
    let floating = render(&dev, &mut r, &sc, &cam, &line(false), 128, 128);
    let hidden = render(&dev, &mut r, &sc, &cam, &line(true), 128, 128);
    let red = |p: [u8; 3]| p[0] as i32 > p[1] as i32 + 50 && p[0] as i32 > p[2] as i32 + 50;
    let count = |img: &Image| (0..128).flat_map(|y| (0..128).map(move |x| (x, y))).filter(|&(x, y)| red(px(img, x, y))).count();
    assert!(count(&floating) > 20, "the rubber band floats on top");
    assert_eq!(count(&hidden), 0, "a depth-tested line hides inside the cube");
}

#[test]
fn billboards_face_the_camera_whatever_the_yaw() {
    let Some(dev) = device() else { return };
    // A flat figure in the xz plane, face-me: seen from any side it keeps its width.
    let mut sc = Scene::new();
    let mut m = ingetrazo_model::mesh::Mesh::new();
    let f = m.add_face(&[V3(-1.0, 0.0, 0.0), V3(1.0, 0.0, 0.0), V3(1.0, 0.0, 2.0), V3(-1.0, 0.0, 2.0)], &[]);
    m.face_mut(f).attrs = vec![("color".into(), Value::Array(vec![Value::Float(0.0), Value::Float(0.8), Value::Float(0.0)]))];
    sc.meshes.push(m);
    let mut g = Group::new(MeshId(1), "figure".into());
    g.billboard = Some(Value::Str("mesh".into()));
    sc.groups.push(g);
    let green = |p: [u8; 3]| p[1] > 150 && p[0] < 60 && p[2] < 60;
    let mut widths = Vec::new();
    for yaw in [-1.4f64, -0.3, 0.8, 2.2] {
        let cam = Camera { target: V3(0.0, 0.0, 1.0), distance: 8.0, yaw, pitch: 0.0, perspective: false, ..Default::default() };
        let mut r = new_renderer(&dev);
        let img = render(&dev, &mut r, &sc, &cam, &opts_with("Shaded"), 160, 80);
        widths.push((0..160).filter(|&x| green(px(&img, x, 40))).count());
    }
    let (lo, hi) = (*widths.iter().min().unwrap(), *widths.iter().max().unwrap());
    assert!(lo > 20 && hi - lo <= 2, "constant width from every side: {widths:?}");
    // Not in the plan sheets: a plan style leaves the figure out.
    let mut r = new_renderer(&dev);
    let mut o = opts_with("Shaded");
    o.billboards = false;
    let none = render(&dev, &mut r, &sc, &Camera::default(), &o, 64, 64);
    assert!(!(0..64).flat_map(|y| (0..64).map(move |x| (x, y))).any(|(x, y)| green(px(&none, x, y))));
}

#[test]
fn frames_are_deterministic_and_survive_a_resize() {
    let Some(dev) = device() else { return };
    let mut sc = cube_scene();
    paint_all(&mut sc.meshes[0], [0.5, 0.7, 0.9]);
    let mut r = new_renderer(&dev);
    r.sync(&sc, &mut NoTextures, &Default::default());
    let cam = view_of(&sc, &mut r);
    let a = render(&dev, &mut r, &sc, &cam, &Default::default(), 200, 120);
    let b = render(&dev, &mut r, &sc, &cam, &Default::default(), 200, 120);
    assert_eq!(a.rgba, b.rgba);
    let small = render(&dev, &mut r, &sc, &cam, &Default::default(), 80, 50);
    assert_eq!((small.width, small.height), (80, 50));
    let again = render(&dev, &mut r, &sc, &cam, &Default::default(), 200, 120);
    assert_eq!(a.rgba, again.rgba);
    // Without MSAA the faces are the same and only the line pixels differ.
    let mut single = dev.renderer(RendererConfig { samples: 1, ..Default::default() });
    let s = render(&dev, &mut single, &sc, &cam, &opts_with("Shaded"), 200, 120);
    let m = render(&dev, &mut r, &sc, &cam, &opts_with("Shaded"), 200, 120);
    assert!(s.compare(&m, 2).fraction_equal > 0.97);
}
