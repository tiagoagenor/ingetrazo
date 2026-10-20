// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! The stitch primitives and the stitch pass against `core.mesh` /
//! `core.history` (scripts/golden/make_stitch_fixtures.py): the same ops
//! leave the same lists in the same order, links, flags, attrs, weld
//! registry, displaced queue and curve counter after every step; and
//! wrapped in `begin`/`commit`, undo then redo walks the same states.

use std::collections::HashSet;
use std::path::Path;

use ingetrazo_formats::pyjson::{dumps, loads, Options, Value};
use ingetrazo_math::{Hypot, V3};
use ingetrazo_model::mesh::{self, EdgeId, Entry, FaceId, Mesh, StitchRec, VertexId};
use ingetrazo_model::stitch::{self, CoplanarMergeCommand, PruneOrphanEdgesCommand, StitchSolidCommand};

const H: Hypot = Hypot::Msvc;

fn arr(v: &Value) -> &Vec<Value> {
    match v {
        Value::Array(a) => a,
        _ => panic!("{v:?}"),
    }
}

fn s(v: &Value) -> &str {
    match v {
        Value::Str(x) => x,
        _ => panic!("{v:?}"),
    }
}

fn f(v: &Value) -> f32 {
    f32::from_bits(u32::from_str_radix(s(v), 16).unwrap())
}

fn p(v: &Value) -> V3 {
    let a = arr(v);
    V3(f(&a[0]), f(&a[1]), f(&a[2]))
}

fn n(v: &Value) -> i64 {
    v.as_i64().unwrap_or_else(|| panic!("{v:?}"))
}

fn hx(x: f32) -> Value {
    Value::Str(format!("{:08x}", x.to_bits()))
}

fn ids<T: Copy>(xs: &[T], id: impl Fn(T) -> u32) -> Value {
    Value::Array(xs.iter().map(|&x| Value::int(id(x) as i64)).collect())
}

fn canon(v: &Value) -> String {
    dumps(v, &Options::compact()).unwrap()
}

fn bits(q: V3) -> Value {
    Value::Array(vec![hx(q.0), hx(q.1), hx(q.2)])
}

/// The state in the fixture's JSON shape.
fn state(m: &Mesh) -> Value {
    let int = |x: i64| Value::int(x);
    let v = m
        .vertices
        .iter()
        .map(|&v| {
            let q = m.position(v);
            Value::Array(vec![int(v.0 as i64), hx(q.0), hx(q.1), hx(q.2)])
        })
        .collect();
    let e = m
        .edges
        .iter()
        .map(|&e| {
            let ed = m.edge(e);
            Value::Array(vec![
                int(e.0 as i64),
                int(ed.v0.0 as i64),
                int(ed.v1.0 as i64),
                ids(&ed.faces, |x| x.0),
                ids(&m.sorted_edges(ed.v0), |x| x.0),
                ids(&m.sorted_edges(ed.v1), |x| x.0),
                Value::Bool(ed.soft),
                Value::Bool(ed.hidden),
                ed.layer.clone().map_or(Value::Null, Value::Str),
                ed.curve.map_or(Value::Null, Value::int),
            ])
        })
        .collect();
    let fa = m
        .faces
        .iter()
        .map(|&fid| {
            let fc = m.face(fid);
            Value::Array(vec![
                int(fid.0 as i64),
                ids(&fc.outer, |x| x.0),
                Value::Array(fc.holes.iter().map(|h| ids(h, |x| x.0)).collect()),
                Value::Bool(fc.interior),
                Value::Object(fc.attrs.clone()),
            ])
        })
        .collect();
    let mut reg: Vec<(i64, i64, i64, u32)> = m.registry().iter().map(|(k, v)| (k.0, k.1, k.2, v.0)).collect();
    reg.sort();
    let mut disp: Vec<(i64, i64, i64, Vec<u32>)> = m.displaced().iter().map(|(k, v)| (k.0, k.1, k.2, v.iter().map(|x| x.0).collect())).collect();
    disp.sort();
    Value::Object(vec![
        ("v".into(), Value::Array(v)),
        ("e".into(), Value::Array(e)),
        ("f".into(), Value::Array(fa)),
        ("reg".into(), Value::Array(reg.iter().map(|r| Value::Array(vec![int(r.0), int(r.1), int(r.2), int(r.3 as i64)])).collect())),
        (
            "disp".into(),
            Value::Array(
                disp.iter()
                    .map(|d| Value::Array(vec![int(d.0), int(d.1), int(d.2), Value::Array(d.3.iter().map(|&x| int(x as i64)).collect())]))
                    .collect(),
            ),
        ),
    ])
}

enum Cmd {
    Stitch(StitchSolidCommand),
    Merge(CoplanarMergeCommand),
    Prune(PruneOrphanEdgesCommand),
}

struct Ctx {
    recs: Vec<StitchRec>,
    cmds: Vec<Cmd>,
    counter: i64,
}

fn vid(v: &Value) -> VertexId {
    VertexId(n(v) as u32)
}
fn eid(v: &Value) -> EdgeId {
    EdgeId(n(v) as u32)
}
fn fid(v: &Value) -> FaceId {
    FaceId(n(v) as u32)
}

fn sorted_ids<T: Copy + Ord>(mut xs: Vec<T>, id: impl Fn(T) -> u32) -> Value {
    xs.sort();
    ids(&xs, id)
}

fn opt_id(x: Option<u32>) -> Value {
    x.map_or(Value::Null, |i| Value::int(i as i64))
}

/// One fixture op; the value it returns, in the fixture's `ret` shape.
fn exec(m: &mut Mesh, c: &mut Ctx, op: &[Value]) -> Value {
    let name = s(&op[0]);
    match name {
        "add_face" => {
            let outer: Vec<V3> = arr(&op[1]).iter().map(p).collect();
            let holes: Vec<Vec<V3>> = arr(&op[2]).iter().map(|h| arr(h).iter().map(p).collect()).collect();
            m.add_face(&outer, &holes);
        }
        "add_edge" => {
            if m.add_edge(p(&op[1]), p(&op[2])).is_err() {
                return Value::Str("degenerate".into());
            }
        }
        "remove_face" => m.remove_face(fid(&op[1])),
        "remove_edge" => m.remove_edge(eid(&op[1])),
        "split_edge" => {
            m.split_edge(eid(&op[1]), p(&op[2]));
        }
        "move_vertex" => m.move_vertex(vid(&op[1]), p(&op[2])),
        "place_vertex" => m.place_vertex(vid(&op[1]), p(&op[2])),
        "set_attrs" => match &op[2] {
            Value::Object(kv) => m.face_mut(fid(&op[1])).attrs = kv.clone(),
            other => panic!("{other:?}"),
        },
        "set_edge" => {
            let ed = m.edge_mut(eid(&op[1]));
            ed.soft = op[2] == Value::Bool(true);
            ed.hidden = op[3] == Value::Bool(true);
            ed.layer = if let Value::Str(l) = &op[4] { Some(l.clone()) } else { None };
            ed.curve = op[5].as_i64();
        }
        "turn_face_over" => m.turn_face_over(fid(&op[1])),
        "tag_curve" => {
            let pts: Vec<V3> = arr(&op[1]).iter().map(p).collect();
            let r = m.tag_curve(&mut c.counter, &pts, op[2] == Value::Bool(true), H);
            return r.map_or(Value::Null, Value::int);
        }
        "resplit_curves" => m.resplit_curves(&mut c.counter),
        "curve_edges" => return ids(&m.curve_edges(eid(&op[1])), |x| x.0),
        "surface_of" => return sorted_ids(m.surface_of(fid(&op[1])), |x| x.0),
        "edges_at" => return sorted_ids(m.edges_at(vid(&op[1])), |x| x.0),
        "faces_at" => return sorted_ids(m.faces_at(vid(&op[1])), |x| x.0),
        "weld_coincident" => return Value::Bool(m.weld_coincident()),
        "dedupe_faces" => return Value::int(m.dedupe_faces() as i64),
        "weld_into" => m.weld_into(vid(&op[1]), vid(&op[2])),
        "interior_vertex_on" => return opt_id(m.interior_vertex_on(eid(&op[1]), H).map(|v| v.0)),
        "split_edge_at" => {
            let r = m.split_edge_at(eid(&op[1]), vid(&op[2]));
            c.recs.push(r);
        }
        "collapsible_vertex" => return Value::Bool(m.collapsible_vertex(vid(&op[1]), H)),
        "collapse_vertex" => {
            let r = m.collapse_vertex(vid(&op[1]));
            c.recs.push(r);
        }
        "dissolve_edge_recorded" => {
            let r = m.dissolve_edge_recorded(eid(&op[1]), H);
            let ok = r.is_some();
            c.recs.extend(r);
            return Value::Bool(ok);
        }
        "dissolve_coplanar_pair" => {
            let r = m.dissolve_coplanar_pair(fid(&op[1]), fid(&op[2]), H);
            let ok = r.is_some();
            c.recs.extend(r);
            return Value::Bool(ok);
        }
        "dissolve_coplanar_region" => {
            let fs: Vec<FaceId> = arr(&op[1]).iter().map(fid).collect();
            return opt_id(m.dissolve_coplanar_region(&fs, H).map(|x| x.0));
        }
        "dissolve_edge" => return opt_id(m.dissolve_edge(eid(&op[1]), H).map(|x| x.0)),
        "revert_stitch" => {
            let i = n(&op[1]) as usize;
            let rec = c.recs[i].clone();
            m.revert_stitch(&rec);
            if i + 1 == c.recs.len() {
                c.recs.pop();
            }
        }
        "run_stitch" => {
            let pts: Vec<V3> = arr(&op[1]).iter().map(p).collect();
            let seed = stitch::seed_keys(&pts);
            let mut nf: Option<HashSet<FaceId>> = if let Value::Array(a) = &op[2] { Some(a.iter().map(fid).collect()) } else { None };
            stitch::run_stitch(m, &seed, nf.as_mut(), op[3] == Value::Bool(true), op[4] == Value::Bool(true), H);
            if let Some(nf) = nf {
                return sorted_ids(nf.into_iter().collect(), |x| x.0);
            }
        }
        "cmd" => {
            let pts: Vec<V3> = arr(&op[2]).iter().map(p).collect();
            let mut cmd = match s(&op[1]) {
                "stitch_solid" => Cmd::Stitch(StitchSolidCommand::new(&pts)),
                "coplanar_merge" => Cmd::Merge(CoplanarMergeCommand::new(&pts)),
                "prune_orphan_edges" => Cmd::Prune(PruneOrphanEdgesCommand::new(&pts)),
                other => panic!("{other}"),
            };
            match &mut cmd {
                Cmd::Stitch(x) => x.apply(m, H),
                Cmd::Merge(x) => x.apply(m, H),
                Cmd::Prune(x) => x.apply(m),
            }
            c.cmds.push(cmd);
        }
        "cmd_undo" => match &mut c.cmds[n(&op[1]) as usize] {
            Cmd::Stitch(x) => x.undo(m),
            Cmd::Merge(x) => x.undo(m),
            Cmd::Prune(x) => x.undo(m),
        },
        "counts" => {
            return Value::Object(vec![
                ("fc".into(), Value::int(mesh::face_count(m) as i64)),
                ("vc".into(), Value::int(mesh::vertex_count(m) as i64)),
                ("vp".into(), Value::Array(mesh::vertex_positions(m).into_iter().map(bits).collect())),
                ("fa".into(), Value::Array(mesh::face_attrs(m).into_iter().map(|a| Value::Object(a.clone())).collect())),
            ]);
        }
        other => panic!("op {other}"),
    }
    Value::Null
}

fn load() -> Vec<Value> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/stitch/stitch_ops.jsonl");
    std::fs::read_to_string(path).unwrap().lines().map(|l| loads(l).unwrap()).collect()
}

fn show(a: &str, b: &str) -> String {
    let at = a.bytes().zip(b.bytes()).take_while(|(x, y)| x == y).count();
    let lo = at.saturating_sub(70);
    format!("\n  rust:   {}\n  python: {}", &a[lo..(at + 70).min(a.len())], &b[lo..(at + 70).min(b.len())])
}

#[test]
fn stitch_matches_core_mesh() {
    let mut fails = Vec::new();
    let mut steps = 0usize;
    'seq: for (si, seq) in load().iter().enumerate() {
        let mut m = Mesh::new();
        let mut c = Ctx { recs: Vec::new(), cmds: Vec::new(), counter: 1 };
        for (i, step) in arr(seq).iter().enumerate() {
            let op = arr(step.get("op").unwrap());
            let ret = exec(&mut m, &mut c, op);
            steps += 1;
            let want_ret = step.get("ret").cloned().unwrap_or(Value::Null);
            if canon(&ret) != canon(&want_ret) {
                fails.push(format!("seq {si} step {i} {}: ret{}", s(&op[0]), show(&canon(&ret), &canon(&want_ret))));
                continue 'seq;
            }
            if c.counter != step.get("ctr").unwrap().as_i64().unwrap() {
                fails.push(format!("seq {si} step {i} {}: curve counter {} vs {:?}", s(&op[0]), c.counter, step.get("ctr")));
                continue 'seq;
            }
            if let Some(want) = step.get("state") {
                let (got, want) = (canon(&state(&m)), canon(want));
                if got != want {
                    fails.push(format!("seq {si} step {i} {}:{}", s(&op[0]), show(&got, &want)));
                    continue 'seq;
                }
            }
        }
    }
    assert!(fails.is_empty(), "{} sequences diverge ({steps} steps matched):\n{}", fails.len(), fails.iter().take(6).cloned().collect::<Vec<_>>().join("\n"));
    assert!(steps > 4000, "only {steps} steps");
}

/// Every step a journal transaction: undoing them all walks back through the
/// states, redoing walks forward again (ids, list order and weld state).
#[test]
fn stitch_journal_undo_redo_walks_the_recorded_states() {
    let mut fails = Vec::new();
    let mut checked = 0usize;
    for (si, seq) in load().iter().enumerate() {
        let mut m = Mesh::new();
        let mut c = Ctx { recs: Vec::new(), cmds: Vec::new(), counter: 1 };
        let mut entries: Vec<Entry> = Vec::new();
        let mut states = vec![canon(&state(&m))];
        let mut ok = true;
        for (i, step) in arr(seq).iter().enumerate() {
            m.begin();
            exec(&mut m, &mut c, arr(step.get("op").unwrap()));
            entries.push(m.commit());
            let got = canon(&state(&m));
            if let Some(want) = step.get("state") {
                if got != canon(want) {
                    fails.push(format!("seq {si} step {i}: forward state differs{}", show(&got, &canon(want))));
                    ok = false;
                    break;
                }
            }
            states.push(got);
        }
        if !ok {
            continue;
        }
        for i in (0..entries.len()).rev() {
            m.undo(&entries[i]);
            checked += 1;
            let got = canon(&state(&m));
            if got != states[i] {
                fails.push(format!("seq {si}: undo of step {i} ({}) is not the state before it{}", s(&arr(arr(seq)[i].get("op").unwrap())[0]), show(&got, &states[i])));
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
            let got = canon(&state(&m));
            if got != states[i + 1] {
                fails.push(format!("seq {si}: redo of step {i} is not the state after it{}", show(&got, &states[i + 1])));
                break;
            }
        }
    }
    assert!(fails.is_empty(), "{} sequences fail ({checked} undo/redo steps checked):\n{}", fails.len(), fails.iter().take(6).cloned().collect::<Vec<_>>().join("\n"));
    assert!(checked > 8000, "only {checked} steps");
}
