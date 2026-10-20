// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! The speed-ups of M4.9 answer exactly what the plain code answers: the
//! caches that keep the snap world and the pick index between edits, and the
//! boxes that cull a ray, over a real model (examples/pileta-fuente-yanque:
//! tens of thousands of faces, groups and component copies).

use std::path::Path;

use ingetrazo_app::session::{Btn, Input, Session};
use ingetrazo_math::{Hypot, V3};
use ingetrazo_model::scene::{MeshId, Scene};
use ingetrazo_snap::picking::{ChunkCache, FaceMode, PickIndex, State};
use ingetrazo_snap::world_cache::WorldCache;
use ingetrazo_snap::World;

fn pileta() -> Session {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../examples/pileta-fuente-yanque.igz");
    let mut s = Session::new();
    s.set_size(800, 500);
    s.open(&path).expect("the example opens");
    s.zoom_extents();
    s
}

fn same_world(a: &World, b: &World, what: &str) {
    // `Debug` prints every number as it is: equal text is equal bits.
    assert_eq!(format!("{:?}", a.tris), format!("{:?}", b.tris), "{what}: triangles");
    assert_eq!(format!("{:?}", a.ents), format!("{:?}", b.ents), "{what}: entities");
    assert_eq!(format!("{:?}", a.gedges), format!("{:?}", b.gedges), "{what}: group edges");
    assert_eq!(format!("{:?}", a.places), format!("{:?}", b.places), "{what}: placements");
    assert_eq!(format!("{:?}", a.origins), format!("{:?}", b.origins), "{what}: origins");
    assert_eq!(format!("{:?}", a.arc_mids), format!("{:?}", b.arc_mids), "{what}: arc midpoints");
    assert_eq!(format!("{:?}", a.bounds), format!("{:?}", b.bounds), "{what}: bounds");
    assert_eq!(format!("{:?}", a.loose_edges), format!("{:?}", b.loose_edges), "{what}: loose edges");
}

fn same_index(a: &PickIndex, b: &PickIndex, what: &str) {
    assert_eq!(format!("{:?}", a.v0), format!("{:?}", b.v0), "{what}: v0");
    assert_eq!(format!("{:?}", a.e1), format!("{:?}", b.e1), "{what}: e1");
    assert_eq!(format!("{:?}", a.e2), format!("{:?}", b.e2), "{what}: e2");
    assert_eq!(a.tri_ent, b.tri_ent, "{what}: tri_ent");
    assert_eq!(format!("{:?}", a.ents), format!("{:?}", b.ents), "{what}: ents");
    assert_eq!(format!("{:?}", a.area), format!("{:?}", b.area), "{what}: areas");
    assert_eq!((&a.vis, &a.sel, &a.loose, &a.place_of), (&b.vis, &b.sel, &b.loose, &b.place_of), "{what}: masks");
    assert_eq!(format!("{:?}", a.spans), format!("{:?}", b.spans), "{what}: spans");
    assert_eq!(format!("{:?}", a.gedge_a), format!("{:?}", b.gedge_a), "{what}: group edges a");
    assert_eq!(format!("{:?}", a.gedge_b), format!("{:?}", b.gedge_b), "{what}: group edges b");
    assert_eq!((&a.gedge_gi, &a.gedge_sel), (&b.gedge_gi, &b.gedge_sel), "{what}: group edge ids");
    assert_eq!(format!("{:?}", a.edge_a), format!("{:?}", b.edge_a), "{what}: loose edges");
    assert_eq!(format!("{:?}", a.places), format!("{:?}", b.places), "{what}: placements");
}

fn draw_a_line(s: &mut Session) {
    let click = |s: &mut Session, p: V3| {
        s.sync();
        let (x, y) = s.to_pixel(p).unwrap();
        s.handle(&Input::Move { x, y, mods: 0 });
        s.handle(&Input::Press { button: Btn::Left, x, y, mods: 0 });
        s.handle(&Input::Release { button: Btn::Left, x, y, mods: 0 });
    };
    s.handle(&Input::Text { text: "l".into(), mods: 0 });
    click(s, V3(1.0, 1.0, 0.0));
    click(s, V3(3.0, 2.0, 0.0));
}

#[test]
fn the_cached_world_is_the_plain_world_before_and_after_an_edit() {
    let mut s = pileta();
    let h = Hypot::default();
    let mut cache = WorldCache::default();
    let plain = World::build(s.scene(), h);
    assert!(plain.tris.len() > 4096, "the example must be big enough to exercise the grid: {}", plain.tris.len());
    same_world(&World::build_cached(s.scene(), h, &mut cache), &plain, "cold");
    same_world(&World::build_cached(s.scene(), h, &mut cache), &plain, "warm");

    // An edit changes one mesh: the rest comes from the cache.
    draw_a_line(&mut s);
    assert!(s.host.core.history.undo_len() > 0, "the line was drawn");
    let plain = World::build(s.scene(), h);
    same_world(&World::build_cached(s.scene(), h, &mut cache), &plain, "after the edit");
    // ... and the host's own cached world (what the tools read) too.
    same_world(s.host.core.world(), &plain, "the host's world");

    assert!(s.undo());
    let plain = World::build(s.scene(), h);
    same_world(&World::build_cached(s.scene(), h, &mut cache), &plain, "after undo");
}

#[test]
fn the_cached_pick_index_is_the_plain_index_before_and_after_an_edit() {
    let mut s = pileta();
    let mut cache = ChunkCache::default();
    fn state(scene: &Scene) -> State<'_> {
        State::new(scene, MeshId(0), MeshId(0), None, false, FaceMode::of_scene(scene), Hypot::default())
    }
    let plain = PickIndex::build(&state(s.scene()));
    assert!(plain.v0.len() > 4096);
    same_index(&PickIndex::build_cached(&state(s.scene()), &mut cache), &plain, "cold");
    same_index(&PickIndex::build_cached(&state(s.scene()), &mut cache), &plain, "warm");
    draw_a_line(&mut s);
    let plain = PickIndex::build(&state(s.scene()));
    same_index(&PickIndex::build_cached(&state(s.scene()), &mut cache), &plain, "after the edit");
}

#[test]
fn the_ray_grid_answers_what_the_plain_scan_answers() {
    let mut s = pileta();
    s.sync();
    let view = s.host.core.view;
    let world = World::build(s.scene(), Hypot::default());
    assert!(world.tris.len() > 4096);
    let (w, h) = (s.size.0 as f64, s.size.1 as f64);
    let mut rays = 0;
    let mut hits = 0;
    // A lattice of pixels, in and a bit beyond the viewport, and the rays
    // of a few other cameras.
    for iy in 0..40 {
        for ix in 0..64 {
            let (x, y) = (ix as f64 * w / 63.0 * 1.1 - w * 0.05, iy as f64 * h / 39.0 * 1.1 - h * 0.05);
            let Some((o, d)) = view.pixel_to_ray(x, y) else { continue };
            let (o, d) = ([o.0 as f64, o.1 as f64, o.2 as f64], [d.0 as f64, d.1 as f64, d.2 as f64]);
            for mask in [&(|e: usize| world.ents[e].sel) as &dyn Fn(usize) -> bool, &|e: usize| world.ents[e].vis] {
                let (g, p) = (world.nearest_hit(o, d, mask), world.nearest_hit_scan(o, d, mask));
                assert_eq!(g.to_bits(), p.to_bits(), "nearest hit at ({x},{y})");
                let (g, p) = (world.entity_hits(o, d, mask), world.entity_hits_scan(o, d, mask));
                assert_eq!(g.len(), p.len());
                assert!(g.iter().zip(&p).all(|(a, b)| a.to_bits() == b.to_bits()), "per-entity hits at ({x},{y})");
                rays += 1;
                if p.iter().any(|t| t.is_finite()) {
                    hits += 1;
                }
            }
        }
    }
    assert!(rays > 4000 && hits > 100, "{rays} rays, {hits} of them hit the model");
}

#[test]
fn the_screen_grid_reaches_every_edge_the_cursor_is_near() {
    let mut s = pileta();
    s.sync();
    let world = World::build(s.scene(), Hypot::default());
    assert!(world.gedges.len() > 5000, "the example has {} group edges", world.gedges.len());
    let (w, h) = (s.size.0 as f64, s.size.1 as f64);
    let (mut checked, mut near_total) = (0, 0);
    // Cameras: the framed one, and orbited ones (edges behind the camera, huge edges).
    for yaw_step in 0..4 {
        let mut cam = s.cam;
        cam.yaw += yaw_step as f64 * 0.9;
        cam.pitch = (cam.pitch + yaw_step as f64 * 0.2).min(1.2);
        s.cam = cam;
        s.sync();
        let view = s.host.core.view;
        for iy in 0..25 {
            for ix in 0..40 {
                let (x, y) = (ix as f64 * w / 39.0, iy as f64 * h / 24.0);
                for lim in [8.0, 48.0] {
                    let (covered, near) = world.near_edges_are_covered(&view, x, y, lim);
                    assert!(covered, "an edge within {lim} px of ({x}, {y}) was not reached (yaw step {yaw_step})");
                    checked += 1;
                    near_total += near;
                }
            }
        }
    }
    assert!(checked > 7000 && near_total > 1000, "{checked} queries found {near_total} near edges");
}

#[test]
fn axis_parallel_rays_and_rays_from_inside_are_not_lost() {
    let mut s = pileta();
    s.sync();
    let world = World::build(s.scene(), Hypot::default());
    let (lo, hi) = world.bounds.expect("the example has geometry");
    let mid = [(lo.0 as f64 + hi.0 as f64) / 2.0, (lo.1 as f64 + hi.1 as f64) / 2.0, (lo.2 as f64 + hi.2 as f64) / 2.0];
    for dir in [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.6, 0.0, 0.8], [0.0, 0.6, -0.8]] {
        for k in 0..30 {
            let f = k as f64 / 29.0;
            let o = [lo.0 as f64 + (hi.0 - lo.0) as f64 * f, lo.1 as f64 + (hi.1 - lo.1) as f64 * (1.0 - f), mid[2]];
            let all = |_: usize| true;
            assert_eq!(world.nearest_hit(o, dir, all).to_bits(), world.nearest_hit_scan(o, dir, all).to_bits(), "{dir:?} from {o:?}");
        }
    }
}
