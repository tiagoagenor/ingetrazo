// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! The arena Mesh against `core.mesh.Mesh` (scripts/golden/make_mesh_fixtures.py):
//! the same primitives in the same order leave the same lists, in the same
//! order, the same links, the same weld registry and displaced queue.

use std::path::Path;

use ingetrazo_formats::pyjson::{loads, Value};
use ingetrazo_math::V3;
use ingetrazo_model::mesh::{EdgeId, FaceId, Mesh, State, VertexId};

fn arr(v: &Value) -> &Vec<Value> {
    match v {
        Value::Array(a) => a,
        _ => panic!("{v:?}"),
    }
}

fn f(v: &Value) -> f32 {
    match v {
        Value::Str(h) => f32::from_bits(u32::from_str_radix(h, 16).unwrap()),
        _ => panic!("{v:?}"),
    }
}

fn p(v: &Value) -> V3 {
    let a = arr(v);
    V3(f(&a[0]), f(&a[1]), f(&a[2]))
}

fn n(v: &Value) -> i64 {
    v.as_i64().unwrap_or_else(|| panic!("{v:?}"))
}

fn hx(x: f32) -> String {
    format!("{:08x}", x.to_bits())
}

/// The state in the fixture's JSON shape, compact.
fn state(m: &Mesh) -> String {
    let ids = |xs: &[u32]| xs.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(",");
    let mut out = String::from("{\"v\":[");
    out += &m
        .vertices
        .iter()
        .map(|&v| {
            let q = m.position(v);
            format!("[{},\"{}\",\"{}\",\"{}\"]", v.0, hx(q.0), hx(q.1), hx(q.2))
        })
        .collect::<Vec<_>>()
        .join(",");
    out += "],\"e\":[";
    out += &m
        .edges
        .iter()
        .map(|&e| {
            let ed = m.edge(e);
            let mut a: Vec<u32> = m.vertex_ref(ed.v0).edges.iter().map(|x| x.0).collect();
            let mut b: Vec<u32> = m.vertex_ref(ed.v1).edges.iter().map(|x| x.0).collect();
            a.sort();
            b.sort();
            let fs: Vec<u32> = ed.faces.iter().map(|x| x.0).collect();
            format!("[{},{},{},[{}],[{}],[{}]]", e.0, ed.v0.0, ed.v1.0, ids(&fs), ids(&a), ids(&b))
        })
        .collect::<Vec<_>>()
        .join(",");
    out += "],\"f\":[";
    out += &m
        .faces
        .iter()
        .map(|&fid| {
            let fc = m.face(fid);
            let o: Vec<u32> = fc.outer.iter().map(|x| x.0).collect();
            let hs: Vec<String> = fc.holes.iter().map(|h| format!("[{}]", ids(&h.iter().map(|x| x.0).collect::<Vec<_>>()))).collect();
            format!("[{},[{}],[{}]]", fid.0, ids(&o), hs.join(","))
        })
        .collect::<Vec<_>>()
        .join(",");
    let mut reg: Vec<(i64, i64, i64, u32)> = m.registry().iter().map(|(k, v)| (k.0, k.1, k.2, v.0)).collect();
    reg.sort();
    out += "],\"reg\":[";
    out += &reg.iter().map(|r| format!("[{},{},{},{}]", r.0, r.1, r.2, r.3)).collect::<Vec<_>>().join(",");
    let mut disp: Vec<(i64, i64, i64, Vec<u32>)> =
        m.displaced().iter().map(|(k, v)| (k.0, k.1, k.2, v.iter().map(|x| x.0).collect())).collect();
    disp.sort();
    out += "],\"disp\":[";
    out += &disp.iter().map(|d| format!("[{},{},{},[{}]]", d.0, d.1, d.2, ids(&d.3))).collect::<Vec<_>>().join(",");
    out += "]}";
    out
}

fn canon(v: &Value) -> String {
    ingetrazo_formats::pyjson::dumps(v, &ingetrazo_formats::pyjson::Options::compact()).unwrap()
}

#[test]
fn mesh_primitives_match_core_mesh() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/mesh_ops.jsonl");
    let text = std::fs::read_to_string(path).unwrap();
    let mut fails = Vec::new();
    let mut steps = 0usize;
    'seq: for (si, line) in text.lines().enumerate() {
        let seq = loads(line).unwrap();
        let mut m = Mesh::new();
        let mut snaps: Vec<State> = Vec::new();
        for (i, step) in arr(&seq).iter().enumerate() {
            let op = arr(step.get("op").unwrap());
            let name = match &op[0] {
                Value::Str(s) => s.as_str(),
                _ => unreachable!(),
            };
            match name {
                "add_face" => {
                    let outer: Vec<V3> = arr(&op[1]).iter().map(p).collect();
                    let holes: Vec<Vec<V3>> = arr(&op[2]).iter().map(|h| arr(h).iter().map(p).collect()).collect();
                    m.add_face(&outer, &holes);
                }
                "add_edge" => {
                    let r = m.add_edge(p(&op[1]), p(&op[2]));
                    if r.is_err() != (op.len() > 3) {
                        fails.push(format!("seq {si} step {i}: add_edge degenerate mismatch"));
                        continue 'seq;
                    }
                }
                "remove_face" => m.remove_face(FaceId(n(&op[1]) as u32)),
                "remove_edge" => m.remove_edge(EdgeId(n(&op[1]) as u32)),
                "relink_face" => m.relink_face(FaceId(n(&op[1]) as u32)),
                "relink_edge" => m.relink_edge(EdgeId(n(&op[1]) as u32)),
                "split_edge" => {
                    m.split_edge(EdgeId(n(&op[1]) as u32), p(&op[2]));
                }
                "move_vertex" => m.move_vertex(VertexId(n(&op[1]) as u32), p(&op[2])),
                "place_vertex" => m.place_vertex(VertexId(n(&op[1]) as u32), p(&op[2])),
                "add_hole" => {
                    let h: Vec<V3> = arr(&op[2]).iter().map(p).collect();
                    m.add_hole(FaceId(n(&op[1]) as u32), &h);
                }
                "remove_hole" => {
                    let f = FaceId(n(&op[1]) as u32);
                    let l = m.face(f).holes[n(&op[2]) as usize].clone();
                    m.remove_hole(f, &l);
                }
                "prune_orphan_vertices" => {
                    let got = m.prune_orphan_vertices();
                    if got as i64 != n(&op[1]) {
                        fails.push(format!("seq {si} step {i}: pruned {got} vs {}", n(&op[1])));
                        continue 'seq;
                    }
                }
                "drop_orphan_vertices" => {
                    let vs: Vec<VertexId> = arr(&op[1]).iter().map(|x| VertexId(n(x) as u32)).collect();
                    let got = m.drop_orphan_vertices(&vs);
                    if got as i64 != n(&op[2]) {
                        fails.push(format!("seq {si} step {i}: dropped {got} vs {}", n(&op[2])));
                        continue 'seq;
                    }
                }
                "capture" => snaps.push(m.capture_state()),
                "restore" => {
                    let s = snaps[n(&op[1]) as usize].clone();
                    m.restore_state(&s);
                }
                other => panic!("op {other}"),
            }
            steps += 1;
            let want = canon(step.get("state").unwrap());
            let got = state(&m);
            if got != want {
                let at = got.bytes().zip(want.bytes()).take_while(|(a, b)| a == b).count();
                let lo = at.saturating_sub(60);
                fails.push(format!(
                    "seq {si} step {i} after {}:\n  rust:   {}\n  python: {}",
                    canon(&step.get("op").unwrap().clone()).chars().take(120).collect::<String>(),
                    &got[lo..(at + 60).min(got.len())],
                    &want[lo..(at + 60).min(want.len())]
                ));
                continue 'seq;
            }
        }
    }
    assert!(fails.is_empty(), "{} sequences diverge ({steps} steps matched):\n{}", fails.len(), fails.iter().take(8).cloned().collect::<Vec<_>>().join("\n"));
    assert!(steps > 1000, "only {steps} steps");
}

#[test]
fn bulk_load_matches_load_mesh_json() {
    use ingetrazo_model::scene::load_mesh;
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/mesh_bulk.jsonl");
    let text = std::fs::read_to_string(path).unwrap();
    let mut fails = Vec::new();
    let mut n = 0;
    for (ci, line) in text.lines().enumerate() {
        let case = loads(line).unwrap();
        let mut m = Mesh::new();
        for q in arr(case.get("pre").unwrap()) {
            let outer: Vec<V3> = arr(q).iter().map(p).collect();
            m.add_face(&outer, &[]);
        }
        load_mesh(&mut m, case.get("payload").unwrap()).unwrap();
        let want = canon(case.get("state").unwrap());
        let got = state(&m);
        if got != want {
            let at = got.bytes().zip(want.bytes()).take_while(|(a, b)| a == b).count();
            let lo = at.saturating_sub(80);
            fails.push(format!(
                "case {ci}:\n  rust:   {}\n  python: {}",
                &got[lo..(at + 80).min(got.len())],
                &want[lo..(at + 80).min(want.len())]
            ));
        }
        n += 1;
    }
    assert!(fails.is_empty(), "{} of {n} bulk loads differ:\n{}", fails.len(), fails.iter().take(4).cloned().collect::<Vec<_>>().join("\n"));
}

/// Apply one fixture op (no checks of its return values).
fn run_op(m: &mut Mesh, op: &[Value], snaps: &mut Vec<State>) {
    let name = match &op[0] {
        Value::Str(s) => s.as_str(),
        _ => unreachable!(),
    };
    match name {
        "add_face" => {
            let outer: Vec<V3> = arr(&op[1]).iter().map(p).collect();
            let holes: Vec<Vec<V3>> = arr(&op[2]).iter().map(|h| arr(h).iter().map(p).collect()).collect();
            m.add_face(&outer, &holes);
        }
        "add_edge" => {
            let _ = m.add_edge(p(&op[1]), p(&op[2]));
        }
        "remove_face" => m.remove_face(FaceId(n(&op[1]) as u32)),
        "remove_edge" => m.remove_edge(EdgeId(n(&op[1]) as u32)),
        "relink_face" => m.relink_face(FaceId(n(&op[1]) as u32)),
        "relink_edge" => m.relink_edge(EdgeId(n(&op[1]) as u32)),
        "split_edge" => {
            m.split_edge(EdgeId(n(&op[1]) as u32), p(&op[2]));
        }
        "move_vertex" => m.move_vertex(VertexId(n(&op[1]) as u32), p(&op[2])),
        "place_vertex" => m.place_vertex(VertexId(n(&op[1]) as u32), p(&op[2])),
        "add_hole" => {
            let h: Vec<V3> = arr(&op[2]).iter().map(p).collect();
            m.add_hole(FaceId(n(&op[1]) as u32), &h);
        }
        "remove_hole" => {
            let f = FaceId(n(&op[1]) as u32);
            let l = m.face(f).holes[n(&op[2]) as usize].clone();
            m.remove_hole(f, &l);
        }
        "prune_orphan_vertices" => {
            m.prune_orphan_vertices();
        }
        "drop_orphan_vertices" => {
            let vs: Vec<VertexId> = arr(&op[1]).iter().map(|x| VertexId(n(x) as u32)).collect();
            m.drop_orphan_vertices(&vs);
        }
        "capture" => snaps.push(m.capture_state()),
        "restore" => {
            let s = snaps[n(&op[1]) as usize].clone();
            m.restore_state(&s);
        }
        other => panic!("op {other}"),
    }
}

/// The journal: every step a transaction; undoing them all walks back
/// through Python's recorded states exactly, and redoing walks forward.
#[test]
fn journal_undo_redo_walks_the_recorded_states() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/mesh_ops.jsonl");
    let text = std::fs::read_to_string(path).unwrap();
    let mut fails = Vec::new();
    let mut checked = 0usize;
    for (si, line) in text.lines().enumerate() {
        let seq = loads(line).unwrap();
        let steps = arr(&seq);
        let mut m = Mesh::new();
        let mut snaps = Vec::new();
        let mut entries = Vec::new();
        let mut states = vec![state(&m)];
        for step in steps {
            m.begin();
            run_op(&mut m, arr(step.get("op").unwrap()), &mut snaps);
            entries.push(m.commit());
            states.push(canon(step.get("state").unwrap()));
        }
        if state(&m) != *states.last().unwrap() {
            fails.push(format!("seq {si}: forward state differs"));
            continue;
        }
        let mut ok = true;
        for i in (0..entries.len()).rev() {
            m.undo(&entries[i]);
            checked += 1;
            if state(&m) != states[i] {
                fails.push(format!("seq {si}: undo of step {i} does not give the state before it"));
                ok = false;
                break;
            }
        }
        if !ok {
            continue;
        }
        for (i, e) in entries.iter().enumerate() {
            m.redo(e);
            checked += 1;
            if state(&m) != states[i + 1] {
                fails.push(format!("seq {si}: redo of step {i} does not give the state after it"));
                break;
            }
        }
    }
    assert!(fails.is_empty(), "{} sequences fail ({checked} undo/redo steps checked):\n{}", fails.len(), fails.iter().take(8).cloned().collect::<Vec<_>>().join("\n"));
    assert!(checked > 5000, "only {checked} steps");
}

/// `add_face` drops a hole nested in a bigger hole of the same face (the
/// `_maximal_holes` choke point), and keeps holes side by side.
#[test]
fn add_face_drops_nested_holes() {
    let sq = |x: f32, y: f32, s: f32| vec![V3(x, y, 0.0), V3(x + s, y, 0.0), V3(x + s, y + s, 0.0), V3(x, y + s, 0.0)];
    let mut m = Mesh::new();
    let f = m.add_face(&sq(0.0, 0.0, 10.0), &[sq(1.0, 1.0, 6.0), sq(2.0, 2.0, 2.0), sq(8.0, 8.0, 1.0)]);
    assert_eq!(m.face(f).holes.len(), 2, "the hole inside the first hole is dropped");
}
