// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! The ID buffer against the CPU picker (M4.2).
//!
//! The M3.12 picker is held bit-exact to Python by `ingetrazo-snap`'s
//! `pick_parity.rs` over 14 fixture scenes and 94 views; this test renders
//! the very same scenes from the very same cameras (opened groups
//! included) into the ID layers and compares `pick_face` with
//! `pick_face_any` and the owner of the face with `pick_group`, pixel by
//! pixel. A ray through an exact point and a rasterised pixel cannot agree
//! at triangle borders, so pixels within 1 px of a change in the face layer
//! are skipped; the rest must agree at 99 % or better. The agreement
//! numbers are printed (`cargo test -p ingetrazo-render --test id_pick --
//! --nocapture`).
//!
//! Edges and vertices are compared too, and reported, with their own
//! (lower) bar: the CPU picker measures distance to the exact segment, the
//! GPU to the pixel centre.
//!
//! The tests skip when the machine has no wgpu adapter.
#![allow(clippy::chunks_exact_to_as_chunks)]
mod common;

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use common::*;
use ingetrazo_formats::igz;
use ingetrazo_formats::pyjson::{loads, Value};
use ingetrazo_geom::hlr_draw::OrbitCamera;
use ingetrazo_math::{Hypot, M4, V3};
use ingetrazo_model::scene::{Group, MeshId, Scene};
use ingetrazo_render::pick::{IdPicker, Layer, Layers, RectMode};
use ingetrazo_render::{Camera, RenderOptions, Style};
use ingetrazo_snap::picking::{open_scene, Camera as SnapCamera, EntRef, FaceMode, Picker, State, View};

const H: Hypot = Hypot::Msvc;
const W: i32 = 1000;
const HH: i32 = 600;

fn data() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../ingetrazo-snap/tests/data/pick")
}

fn open(name: &str) -> Scene {
    let raw = std::fs::read(data().join(format!("{name}.igz.json"))).unwrap();
    let doc = igz::read_document(&raw).unwrap();
    Scene::from_document(&doc).unwrap_or_else(|e| panic!("{name}: {e:?}"))
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

fn int(v: &Value) -> i64 {
    v.as_i64().unwrap_or_else(|| panic!("not an int: {v:?}"))
}

fn f64_of(v: &Value) -> f64 {
    f64::from_bits(u64::from_str_radix(text(v), 16).unwrap())
}

fn f32_of(v: &Value) -> f32 {
    f32::from_bits(u32::from_str_radix(text(v), 16).unwrap())
}

fn v3_of(v: &Value) -> V3 {
    let a = arr(v);
    V3(f32_of(&a[0]), f32_of(&a[1]), f32_of(&a[2]))
}

/// The fixture's camera as both pickers need it.
fn cameras(c: &Value) -> (SnapCamera, Camera) {
    let g = |k: &str| c.get(k).unwrap();
    let orbit = OrbitCamera {
        target: v3_of(g("target")),
        distance: f64_of(g("distance")),
        yaw: f64_of(g("yaw")),
        pitch: f64_of(g("pitch")),
        up: v3_of(g("up")),
    };
    let mut cpu = SnapCamera::new(orbit);
    cpu.perspective = matches!(g("perspective"), Value::Bool(true));
    cpu.fov_deg = f64_of(g("fov"));
    cpu.aspect = f64_of(g("aspect"));
    cpu.znear = f64_of(g("znear"));
    cpu.zfar = f64_of(g("zfar"));
    let gpu = Camera {
        target: orbit.target,
        distance: orbit.distance,
        yaw: orbit.yaw,
        pitch: orbit.pitch,
        up: orbit.up,
        fov_deg: cpu.fov_deg,
        aspect: cpu.aspect,
        znear: cpu.znear,
        zfar: cpu.zfar,
        perspective: cpu.perspective,
        two_point: false,
        hypot: H,
    };
    (cpu, gpu)
}

/// An entity named the way both pickers can say it: the loose part (the
/// open group's mesh while one is open) or a placement, and the index of
/// the face in the mesh's `faces`.
#[derive(Clone, Debug, PartialEq, Eq)]
enum Ent {
    Loose(usize),
    Placed(Vec<usize>, usize),
}

#[derive(Default)]
struct Tally {
    total: usize,
    agree: usize,
    both_none: usize,
    skipped_border: usize,
    kinds: BTreeMap<String, usize>,
    examples: Vec<String>,
}

impl Tally {
    fn miss(&mut self, kind: &str, example: String) {
        let n = self.kinds.entry(kind.to_string()).or_default();
        *n += 1;
        if *n <= 3 {
            self.examples.push(example);
        }
    }

    fn rate(&self) -> f64 {
        if self.total == 0 {
            1.0
        } else {
            self.agree as f64 / self.total as f64
        }
    }

    fn report(&self, what: &str) {
        eprintln!(
            "{what}: {}/{} agree ({:.3} %), {} empty on both sides, {} skipped near an ID border; differences {:?}",
            self.agree,
            self.total,
            self.rate() * 100.0,
            self.both_none,
            self.skipped_border,
            self.kinds
        );
        for e in &self.examples {
            eprintln!("    {e}");
        }
    }
}

#[test]
fn face_and_group_picks_agree_with_the_cpu_picker() {
    let Some(dev) = device() else { return };
    let raw = std::fs::read(data().join("cases.json")).unwrap();
    let views = loads(std::str::from_utf8(&raw).unwrap()).unwrap();
    let mut scenes: BTreeMap<String, Scene> = BTreeMap::new();
    let mut picker = IdPicker::new(&dev.device, &dev.queue);
    // The same without occlusion, to tell what an edge miss is.
    let mut open_picker = IdPicker::new(&dev.device, &dev.queue);
    open_picker.options.visible_edges_only = false;
    let mut faces = Tally::default();
    let mut groups = Tally::default();
    let mut edges = Tally::default();
    let mut n_views = 0usize;
    let mut per_scene: BTreeMap<String, (usize, usize)> = BTreeMap::new();

    for (vi, view) in arr(&views).iter().enumerate() {
        let name = text(view.get("scene").unwrap()).to_string();
        let base = scenes.entry(name.clone()).or_insert_with(|| open(&name));
        let edit: Option<Vec<usize>> = match view.get("edit").unwrap() {
            Value::Null => None,
            v => Some(arr(v).iter().map(|i| int(i) as usize).collect()),
        };
        let (sc, active) = open_scene(base, edit.as_deref(), H);
        let rest = text(view.get("rest").unwrap()).to_string();
        let rest_hidden = edit.is_some() && rest == "hide";
        let face_mode_name = text(view.get("face_mode").unwrap());
        let (cpu_cam, gpu_cam) = cameras(view.get("cam").unwrap());

        // The CPU side, as in pick_parity.rs.
        let cpu_view = View::new(cpu_cam, W, HH, H);
        let st = State::new(&sc, active, MeshId(0), edit.clone(), rest_hidden, FaceMode::from_name(face_mode_name), H);
        let cpu = Picker::new(st, cpu_view);

        // The GPU side.
        let style = match face_mode_name {
            "xray" => Style::builtin("X-ray"),
            "wireframe" => Style::builtin("Wireframe"),
            _ => Style::builtin("Default"),
        };
        let opts = RenderOptions {
            style,
            edit_path: edit.clone(),
            edit_rest: if rest_hidden { ingetrazo_render::EditRest::Hide } else { ingetrazo_render::EditRest::Fade },
            ..Default::default()
        };
        picker.invalidate();
        assert!(picker.prepare(&sc, &gpu_cam, (W as u32, HH as u32), &opts), "first prepare renders");
        open_picker.invalidate();
        open_picker.prepare(&sc, &gpu_cam, (W as u32, HH as u32), &opts);
        let face_img = picker.read_layer(Layer::Faces, 0, 0, W as u32, HH as u32).expect("face layer");
        n_views += 1;

        // How the CPU sees an edge from a pixel: its distance in pixels and
        // whether it is hidden.
        let edge_info = |i: usize, x: f64, y: f64| -> (f64, bool) {
            let v = |p: [f64; 3]| V3(p[0] as f32, p[1] as f32, p[2] as f32);
            let (a, b) = (v(cpu.idx.edge_a[i]), v(cpu.idx.edge_b[i]));
            let (Some(pa), Some(pb)) = (cpu.view.world_to_pixel(a), cpu.view.world_to_pixel(b)) else {
                return (f64::NAN, false);
            };
            let (dx, dy) = (pb.0 - pa.0, pb.1 - pa.1);
            let l2 = dx * dx + dy * dy;
            let t = if l2 > 1e-12 { (((x - pa.0) * dx + (y - pa.1) * dy) / l2).clamp(0.0, 1.0) } else { 0.0 };
            let d = ((pa.0 + t * dx - x).powi(2) + (pa.1 + t * dy - y).powi(2)).sqrt();
            let hidden = cpu.edge_point_under_cursor(a, b, x, y).is_some_and(|p| cpu.is_occluded(p));
            (d, hidden)
        };
        let ent_of_gpu = |placement: &Vec<usize>, mesh: MeshId, face: ingetrazo_model::FaceId| -> Ent {
            let ix = sc.mesh(mesh).faces.iter().position(|&f| f == face).expect("face in mesh");
            let loose = match &edit {
                None => placement.is_empty(),
                Some(c) => placement == c,
            };
            if loose {
                Ent::Loose(ix)
            } else {
                Ent::Placed(placement.clone(), ix)
            }
        };
        let ent_of_cpu = |e: EntRef| -> Ent {
            match e {
                EntRef::Loose(i) => Ent::Loose(i),
                EntRef::Placed { place, face } => Ent::Placed(cpu.idx.places[place].path.clone(), face),
            }
        };

        for rec in arr(view.get("px").unwrap()) {
            let r = arr(rec);
            let (x, y) = (f64_of(&r[0]), f64_of(&r[1]));
            let (ix, iy) = (x.floor() as i64, y.floor() as i64);
            // The GPU answers for the pixel; ask the CPU about its centre.
            let (x, y) = (ix as f64 + 0.5, iy as f64 + 0.5);
            if ix < 0 || iy < 0 || ix >= W as i64 || iy >= HH as i64 {
                continue;
            }
            // A change in the ID buffer within 1 px: rasterisation decides.
            let centre = face_img.at(ix as u32, iy as u32);
            let mut border = false;
            for dy in -1i64..=1 {
                for dx in -1i64..=1 {
                    let (nx, ny) = (ix + dx, iy + dy);
                    if nx >= 0 && ny >= 0 && nx < W as i64 && ny < HH as i64 && face_img.at(nx as u32, ny as u32) != centre {
                        border = true;
                    }
                }
            }
            let tag = format!("view {vi} ({name}, edit {edit:?}, {face_mode_name}) px ({x:.2}, {y:.2})");
            let hits = picker.pick_layers(x, y);
            if border {
                faces.skipped_border += 1;
                groups.skipped_border += 1;
            } else {
                let want = cpu.pick_face_any(x, y);
                let got = hits.face.clone();
                let e = per_scene.entry(name.clone()).or_default();
                match (&want, &got) {
                    (None, None) => {
                        faces.both_none += 1;
                    }
                    _ => {
                        faces.total += 1;
                        e.0 += 1;
                        let w = want.as_ref().map(|h| ent_of_cpu(h.face));
                        let g = got.as_ref().map(|h| ent_of_gpu(&h.placement, h.mesh, h.face));
                        if w == g {
                            faces.agree += 1;
                            e.1 += 1;
                        } else {
                            let kind = match (&w, &g) {
                                (Some(_), None) => "cpu-only",
                                (None, Some(_)) => "gpu-only",
                                (Some(Ent::Loose(_)), Some(Ent::Loose(_))) => "other loose face",
                                (Some(Ent::Placed(a, _)), Some(Ent::Placed(b, _))) if a == b => "other face, same placement",
                                _ => "other placement",
                            };
                            faces.miss(kind, format!("{tag}: gpu {g:?} cpu {w:?}"));
                        }
                    }
                }
                // Groups: the owner of the nearest group face (or, without
                // a face, of the group edge in the band).
                let wg = cpu.pick_group(x, y);
                let gg = picker.pick_group(x, y);
                if wg.is_none() && gg.is_none() {
                    groups.both_none += 1;
                } else {
                    groups.total += 1;
                    if wg == gg {
                        groups.agree += 1;
                    } else {
                        let kind = match (&wg, &gg) {
                            (Some(_), None) => "cpu-only",
                            (None, Some(_)) => "gpu-only",
                            _ => "other group",
                        };
                        groups.miss(&format!("{kind} [{name}]"), format!("{tag}: gpu {gg:?} cpu {wg:?}"));
                    }
                }
            }
            // Edges: the nearest visible loose edge within 8 px.
            let wantv = cpu.pick_visible_edge(x, y);
            let gv = hits.edge.as_ref();
            edges.total += 1;
            let loose_edge_ix = |h: &ingetrazo_render::pick::EdgeHit| {
                let loose = match &edit {
                    None => h.placement.is_empty(),
                    Some(c) => &h.placement == c,
                };
                loose.then(|| sc.mesh(h.mesh).edges.iter().position(|&e| e == h.edge).expect("edge in mesh"))
            };
            match (wantv, gv) {
                (None, None) => {
                    edges.total -= 1;
                    edges.both_none += 1;
                }
                (Some(w), Some(g)) => match loose_edge_ix(g) {
                    Some(i) if i == w => edges.agree += 1,
                    Some(i) => {
                        let (dg, og) = edge_info(i, x, y);
                        let (dw, ow) = edge_info(w, x, y);
                        let kind = if og {
                            "other edge: the GPU one is occluded for the CPU"
                        } else if ow {
                            "other edge: the CPU one is occluded here"
                        } else if (dg - dw).abs() < 0.75 {
                            "other edge: a tie within 0.75 px"
                        } else {
                            "other edge: farther"
                        };
                        edges.miss(kind, format!("{tag}: gpu {i} ({dg:.2} px, occluded {og}) cpu {w} ({dw:.2} px, occluded {ow})"));
                    }
                    None => {
                        // A group's edge is nearer: the CPU only looks at loose ones.
                        edges.total -= 1;
                        edges.skipped_border += 1;
                    }
                },
                (Some(w), None) => {
                    let open = open_picker.pick_layers(x, y).edge;
                    let kind = if open.is_some() { "cpu-only (occluded here)" } else { "cpu-only (no band here)" };
                    edges.miss(kind, format!("{tag}: cpu {w}"));
                }
                (None, Some(g)) => {
                    if loose_edge_ix(g).is_some() {
                        edges.miss("gpu-only", format!("{tag}: gpu {g:?}"));
                    } else {
                        edges.total -= 1;
                        edges.skipped_border += 1;
                    }
                }
            }
        }
    }
    eprintln!("{n_views} views, {} px records", faces.total + faces.both_none + faces.skipped_border);
    faces.report("faces ");
    groups.report("groups");
    edges.report("edges ");
    for (s, (n, ok)) in &per_scene {
        eprintln!("    {s}: {ok}/{n} face picks");
    }
    assert!(faces.rate() >= 0.99, "face agreement {:.3} %", faces.rate() * 100.0);
    assert!(groups.rate() >= 0.99, "group agreement {:.3} %", groups.rate() * 100.0);
    assert!(edges.rate() >= 0.90, "edge agreement {:.3} %", edges.rate() * 100.0);
}

// ---- the ID pass on small scenes --------------------------------------------------------------

fn pick_scene() -> Scene {
    // Two cubes side by side along x.
    let mut sc = Scene::new();
    cube_into(&mut sc.meshes[0], V3(0.0, 0.0, 0.0), 2.0);
    cube_into(&mut sc.meshes[0], V3(4.0, 0.0, 0.0), 2.0);
    sc
}

fn front_camera(sc: &Scene) -> Camera {
    let mut cam = Camera { yaw: -std::f64::consts::FRAC_PI_2, pitch: 0.3, ..Camera::default() };
    let _ = sc;
    cam.fit_to(V3(0.0, 0.0, 0.0), V3(6.0, 2.0, 2.0), 1.4);
    cam
}

#[test]
fn the_id_pass_is_lazy() {
    let Some(dev) = device() else { return };
    let mut sc = pick_scene();
    let cam = front_camera(&sc);
    let opts = RenderOptions::default();
    let mut p = IdPicker::new(&dev.device, &dev.queue);
    assert!(p.prepare(&sc, &cam, (640, 480), &opts));
    assert!(!p.prepare(&sc, &cam, (640, 480), &opts), "nothing changed");
    assert!(!p.prepare(&sc, &cam, (640, 480), &opts));
    assert_eq!(p.stats().renders, 1);
    assert_eq!(p.stats().reused, 2);
    // The camera moved.
    let mut cam2 = cam;
    cam2.yaw += 0.1;
    assert!(p.prepare(&sc, &cam2, (640, 480), &opts));
    // The viewport was resized.
    assert!(p.prepare(&sc, &cam2, (800, 480), &opts));
    // The scene was edited.
    let baked = p.stats().baked_meshes;
    sc.meshes[0].add_face(&[V3(8.0, 0.0, 0.0), V3(9.0, 0.0, 0.0), V3(9.0, 1.0, 0.0)], &[]);
    assert!(p.prepare(&sc, &cam2, (800, 480), &opts));
    assert_eq!(p.stats().baked_meshes, baked + 1);
    // A group appeared: the mesh is not baked again.
    sc.groups.push(Group::new(MeshId(0), "g".into()));
    assert!(p.prepare(&sc, &cam2, (800, 480), &opts));
    assert_eq!(p.stats().baked_meshes, baked + 1);
    // The group moved.
    sc.groups[0].xform = Some(translation(0.0, 5.0, 0.0));
    assert!(p.prepare(&sc, &cam2, (800, 480), &opts));
    assert!(!p.prepare(&sc, &cam2, (800, 480), &opts));
    // A tolerance changed.
    p.options.edge_tolerance_px = 4.0;
    assert!(p.prepare(&sc, &cam2, (800, 480), &opts));
}

#[test]
fn faces_edges_and_vertices_are_found_under_the_cursor() {
    let Some(dev) = device() else { return };
    let sc = pick_scene();
    let cam = front_camera(&sc);
    let (w, h) = (800u32, 600u32);
    let mut p = IdPicker::new(&dev.device, &dev.queue);
    p.prepare(&sc, &cam, (w, h), &RenderOptions::default());
    // Project a world point to pixels the way the pass does.
    let mut c = cam;
    c.set_aspect(w, h);
    let to_px = |q: V3| {
        let m = c.mvp();
        let v = [q.0, q.1, q.2, 1.0];
        let row = |r: usize| (0..4).map(|k| m.m[k][r] * v[k]).sum::<f32>() as f64;
        let (cx, cy, cw) = (row(0), row(1), row(3));
        ((cx / cw * 0.5 + 0.5) * w as f64, (1.0 - (cy / cw * 0.5 + 0.5)) * h as f64)
    };
    // The front (y = 0) face of the first cube, at its centre.
    let (fx, fy) = to_px(V3(1.0, 0.0, 1.0));
    let hit = p.pick_layers(fx, fy);
    let face = hit.face.expect("a face at the cube's centre");
    assert!(face.placement.is_empty() && face.owner.is_none());
    let mesh = &sc.meshes[0];
    let f = mesh.face(face.face);
    assert!(f.outer.iter().all(|&v| mesh.position(v).1 == 0.0), "a y = 0 face");
    assert!(f.outer.iter().all(|&v| mesh.position(v).0 <= 2.0), "of the first cube");
    assert!(hit.edge.is_none() && hit.vertex.is_none(), "far from every edge");
    // Beside the bottom-front edge of the second cube.
    let (ex, ey) = to_px(V3(5.0, 0.0, 0.0));
    let hit = p.pick_layers(ex, ey + 3.0);
    let e = hit.edge.expect("the edge within the band");
    let (a, b) = (mesh.position(mesh.edge(e.edge).v0), mesh.position(mesh.edge(e.edge).v1));
    assert!(a.1 == 0.0 && b.1 == 0.0 && a.2 == 0.0 && b.2 == 0.0 && a.0 >= 4.0, "{a:?} {b:?}");
    // The hidden edge at the back of the first cube: only `pick_edge` finds it.
    let (bx, by) = to_px(V3(1.0, 2.0, 0.0));
    assert!(p.pick_layers(bx, by).edge.is_none(), "hidden behind the cube");
    let mut all = IdPicker::new(&dev.device, &dev.queue);
    all.options.visible_edges_only = false;
    all.prepare(&sc, &cam, (w, h), &RenderOptions::default());
    let e = all.pick_layers(bx, by).edge.expect("every edge counts without the occlusion");
    let (a, b) = (mesh.position(mesh.edge(e.edge).v0), mesh.position(mesh.edge(e.edge).v1));
    assert!(a.1 == 2.0 && b.1 == 2.0 && a.2 == 0.0 && b.2 == 0.0 && a.0 <= 2.0, "{a:?} {b:?}");
    // A corner.
    let (vx, vy) = to_px(V3(6.0, 0.0, 0.0));
    let hit = p.pick_layers(vx + 2.0, vy - 2.0);
    let v = hit.vertex.expect("the corner within its disc");
    assert_eq!(mesh.position(v.vertex), V3(6.0, 0.0, 0.0));
    assert!(matches!(p.pick_pixel(vx + 2.0, vy - 2.0), Some(ingetrazo_render::pick::Hit::Vertex(_))));
    // The sky.
    assert_eq!(p.pick_pixel(2.0, 2.0), None);
    // Outside the viewport.
    assert_eq!(p.pick_pixel(-5.0, 10.0), None);
    assert_eq!(p.pick_pixel(10_000.0, 10.0), None);
}

#[test]
fn rubber_bands_take_the_visible_picture() {
    let Some(dev) = device() else { return };
    let sc = pick_scene();
    let cam = front_camera(&sc);
    let (w, h) = (800u32, 600u32);
    let mut p = IdPicker::new(&dev.device, &dev.queue);
    p.prepare(&sc, &cam, (w, h), &RenderOptions::default());
    let everything = (0.0, 0.0, w as f64, h as f64);
    let crossing = p.pick_rect(everything, RectMode::Crossing);
    let window = p.pick_rect(everything, RectMode::Window);
    let faces = |hits: &[ingetrazo_render::pick::Hit]| {
        hits.iter().filter(|h| matches!(h, ingetrazo_render::pick::Hit::Face(_))).count()
    };
    // Seen from the front and above: the front, top and (for the left cube) right faces.
    assert!(faces(&crossing) >= 4, "{} faces", faces(&crossing));
    assert!(crossing.len() > faces(&crossing), "edges too");
    // The cubes stand wholly in the picture: the window takes the same visible faces.
    assert_eq!(faces(&window), faces(&crossing));
    // A band over the left half cuts the second cube.
    let mut c = cam;
    c.set_aspect(w, h);
    let m = c.mvp();
    let v = [5.0f32, 0.0, 1.0, 1.0];
    let row = |r: usize| (0..4).map(|k| m.m[k][r] * v[k]).sum::<f32>() as f64;
    let mid_second = (row(0) / row(3) * 0.5 + 0.5) * w as f64;
    let left = (0.0, 0.0, mid_second, h as f64);
    let cr = p.pick_rect(left, RectMode::Crossing);
    let wi = p.pick_rect(left, RectMode::Window);
    assert!(!cr.is_empty() && !wi.is_empty());
    assert!(wi.len() < cr.len(), "window {} < crossing {}", wi.len(), cr.len());
    for hit in &wi {
        assert!(cr.contains(hit));
    }
    // An empty band.
    assert!(p.pick_rect((0.0, 0.0, 3.0, 3.0), RectMode::Crossing).is_empty());
    // Faces only.
    let only = p.pick_rect_layers(everything, RectMode::Crossing, Layers::FACES);
    assert_eq!(only.len(), faces(&crossing));
}

#[test]
fn what_is_hidden_or_locked_or_cut_is_not_picked() {
    let Some(dev) = device() else { return };
    // A cube in the loose mesh, and a component of one in front of it.
    let mut sc = Scene::new();
    cube_into(&mut sc.meshes[0], V3(0.0, 4.0, 0.0), 2.0);
    let mut proto = ingetrazo_model::mesh::Mesh::new();
    cube_into(&mut proto, V3(0.0, 0.0, 0.0), 2.0);
    sc.meshes.push(proto);
    sc.groups.push(instance(MeshId(1), M4::identity(), "front"));
    let mut cam = Camera { yaw: -std::f64::consts::FRAC_PI_2, pitch: 0.0, ..Camera::default() };
    cam.fit_to(V3(0.0, 0.0, 0.0), V3(2.0, 6.0, 2.0), 1.6);
    let (w, h) = (400u32, 300u32);
    let opts = RenderOptions::default();
    let centre = (w as f64 / 2.0, h as f64 / 2.0);
    let mut p = IdPicker::new(&dev.device, &dev.queue);

    p.prepare(&sc, &cam, (w, h), &opts);
    let f = p.pick_face(centre.0, centre.1).expect("the component's face");
    assert_eq!(f.placement, vec![0]);
    assert_eq!(f.owner, Some(vec![0]));
    assert_eq!(p.pick_group(centre.0, centre.1), Some(vec![0]));
    assert_eq!(p.pick_layers(centre.0, centre.1).group, Some(vec![0]));

    // Hidden: the face behind shows through.
    sc.groups[0].hidden = true;
    p.prepare(&sc, &cam, (w, h), &opts);
    let f = p.pick_face(centre.0, centre.1).expect("the loose cube");
    assert!(f.placement.is_empty());
    sc.groups[0].hidden = false;

    // Locked: the same, the group cannot be touched.
    sc.groups[0].locked = true;
    p.prepare(&sc, &cam, (w, h), &opts);
    let f = p.pick_face(centre.0, centre.1).expect("the loose cube");
    assert!(f.placement.is_empty());
    sc.groups[0].locked = false;

    // A section cut between the two: the front cube is cut away.
    sc.section_planes.push(ingetrazo_model::section::SectionPlane {
        point: V3(0.0, 3.0, 0.0),
        normal: V3(0.0, -1.0, 0.0),
        name: "cut".into(),
        symbol: String::new(),
        active: true,
        uid: 1,
        hidden: false,
        fill: None,
    });
    p.prepare(&sc, &cam, (w, h), &opts);
    let f = p.pick_face(centre.0, centre.1);
    assert!(f.as_ref().is_some_and(|f| f.placement.is_empty()), "{f:?}");
    sc.section_planes[0].active = false;

    // An open group: the loose cube is outside it and cannot be picked.
    let edit = RenderOptions { edit_path: Some(vec![0]), ..Default::default() };
    p.prepare(&sc, &cam, (w, h), &edit);
    let f = p.pick_face(centre.0, centre.1).expect("the open group's face");
    assert_eq!(f.placement, vec![0]);
    assert_eq!(f.owner, None, "the open group's own faces are its content");
    assert_eq!(p.pick_face(w as f64 - 20.0, 20.0), None);
}
