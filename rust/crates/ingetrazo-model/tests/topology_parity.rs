// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! `topology` against `core/topology.py` (scripts/golden/make_topology_fixtures.py):
//! random scenes, and on each a sequence of calls to the real Python
//! functions, recorded with their arguments and results bit for bit. A
//! mutating call also records the whole mesh state after it.

use std::path::Path;

use ingetrazo_formats::pyjson::{dumps, loads, Options, Value};
use ingetrazo_math::{Hypot, V3};
use ingetrazo_model::mesh::{EdgeId, FaceId, Mesh, VertexId};
use ingetrazo_model::topology::{self as t, CycleGraph, FootprintEntry, Loc, PFace, PushEdge};

const H: Hypot = Hypot::Msvc;

fn arr(v: &Value) -> &Vec<Value> {
    match v {
        Value::Array(a) => a,
        _ => panic!("{v:?}"),
    }
}

fn is_null(v: &Value) -> bool {
    matches!(v, Value::Null)
}

fn hex_f32(v: &Value) -> f32 {
    match v {
        Value::Str(h) => f32::from_bits(u32::from_str_radix(h, 16).unwrap()),
        _ => panic!("{v:?}"),
    }
}

fn hex_f64(v: &Value) -> f64 {
    match v {
        Value::Str(h) => f64::from_bits(u64::from_str_radix(h, 16).unwrap()),
        _ => panic!("{v:?}"),
    }
}

fn pt(v: &Value) -> V3 {
    let a = arr(v);
    V3(hex_f32(&a[0]), hex_f32(&a[1]), hex_f32(&a[2]))
}

fn pts(v: &Value) -> Vec<V3> {
    arr(v).iter().map(pt).collect()
}

fn p2(v: &Value) -> (f64, f64) {
    let a = arr(v);
    (hex_f64(&a[0]), hex_f64(&a[1]))
}

fn p2l(v: &Value) -> Vec<(f64, f64)> {
    arr(v).iter().map(p2).collect()
}

fn n(v: &Value) -> usize {
    v.as_i64().unwrap_or_else(|| panic!("{v:?}")) as usize
}

fn ids(v: &Value) -> Vec<usize> {
    arr(v).iter().map(n).collect()
}

fn fid(v: &Value) -> FaceId {
    FaceId(n(v) as u32)
}

fn eids(v: &Value) -> Vec<EdgeId> {
    ids(v).into_iter().map(|i| EdgeId(i as u32)).collect()
}

fn fids(v: &Value) -> Vec<FaceId> {
    ids(v).into_iter().map(|i| FaceId(i as u32)).collect()
}

// ---- Encoders (the fixture's shapes) ----

fn ep(p: V3) -> Value {
    Value::Array([p.0, p.1, p.2].iter().map(|x| Value::Str(format!("{:08x}", x.to_bits()))).collect())
}

fn epl(ps: &[V3]) -> Value {
    Value::Array(ps.iter().map(|&p| ep(p)).collect())
}

fn ef(x: f64) -> Value {
    Value::Str(format!("{:016x}", x.to_bits()))
}

fn ei(i: usize) -> Value {
    Value::int(i as i64)
}

fn eb(b: bool) -> Value {
    Value::Bool(b)
}

fn eopt<T>(o: Option<T>, f: impl Fn(T) -> Value) -> Value {
    match o {
        Some(x) => f(x),
        None => Value::Null,
    }
}

fn canon(v: &Value) -> String {
    dumps(v, &Options::compact()).unwrap()
}

fn efaces(fs: &[FaceId]) -> Value {
    Value::Array(fs.iter().map(|f| ei(f.0 as usize)).collect())
}

fn state(m: &Mesh) -> Value {
    let v = Value::Array(
        m.vertices
            .iter()
            .map(|&v| {
                let q = m.position(v);
                Value::Array(vec![ei(v.0 as usize), Value::Str(format!("{:08x}", q.0.to_bits())), Value::Str(format!("{:08x}", q.1.to_bits())), Value::Str(format!("{:08x}", q.2.to_bits()))])
            })
            .collect(),
    );
    let sorted_edges = |vx: VertexId| {
        let mut a: Vec<u32> = m.vertex_ref(vx).edges.iter().map(|x| x.0).collect();
        a.sort();
        Value::Array(a.into_iter().map(|x| ei(x as usize)).collect())
    };
    let e = Value::Array(
        m.edges
            .iter()
            .map(|&e| {
                let ed = m.edge(e);
                Value::Array(vec![
                    ei(e.0 as usize),
                    ei(ed.v0.0 as usize),
                    ei(ed.v1.0 as usize),
                    efaces(&ed.faces),
                    sorted_edges(ed.v0),
                    sorted_edges(ed.v1),
                    Value::Bool(ed.soft),
                    ed.layer.clone().map(Value::Str).unwrap_or(Value::Null),
                    ed.curve.map(Value::int).unwrap_or(Value::Null),
                ])
            })
            .collect(),
    );
    let vids = |l: &[VertexId]| Value::Array(l.iter().map(|x| ei(x.0 as usize)).collect());
    let f = Value::Array(
        m.faces
            .iter()
            .map(|&fid| {
                let fc = m.face(fid);
                Value::Array(vec![
                    ei(fid.0 as usize),
                    vids(&fc.outer),
                    Value::Array(fc.holes.iter().map(|h| vids(h)).collect()),
                    Value::Object(fc.attrs.clone()),
                ])
            })
            .collect(),
    );
    let mut reg: Vec<(i64, i64, i64, u32)> = m.registry().iter().map(|(k, v)| (k.0, k.1, k.2, v.0)).collect();
    reg.sort();
    let reg = Value::Array(
        reg.iter().map(|r| Value::Array(vec![Value::int(r.0), Value::int(r.1), Value::int(r.2), ei(r.3 as usize)])).collect(),
    );
    Value::Object(vec![("v".into(), v), ("e".into(), e), ("f".into(), f), ("reg".into(), reg)])
}

fn efootprint(fp: &Option<Vec<FootprintEntry>>) -> Value {
    match fp {
        None => Value::Null,
        Some(list) => Value::Array(
            list.iter()
                .map(|e| {
                    Value::Array(vec![
                        ei(e.face.0 as usize),
                        ep(e.normal),
                        epl(&e.outer),
                        Value::Array(e.holes.iter().map(|h| epl(h)).collect()),
                        Value::Array(e.bbox.iter().map(|&x| ef(x)).collect()),
                    ])
                })
                .collect(),
        ),
    }
}

fn eloops(ls: &[Vec<V3>]) -> Value {
    Value::Array(ls.iter().map(|l| epl(l)).collect())
}

fn echunk(r: Option<(Vec<V3>, Vec<V3>)>) -> Value {
    eopt(r, |(a, b)| Value::Array(vec![epl(&a), epl(&b)]))
}

fn fp_arg<'a>(fps: &'a [Option<Vec<FootprintEntry>>], v: &Value) -> Option<&'a [FootprintEntry]> {
    if is_null(v) {
        return None;
    }
    let a = arr(v);
    fps[n(&a[1])].as_deref()
}

/// One recorded call, run on the Rust side; the result in the fixture's shape.
fn run_call(m: &mut Mesh, fps: &mut Vec<Option<Vec<FootprintEntry>>>, c: &[Value]) -> Value {
    let name = match &c[0] {
        Value::Str(s) => s.as_str(),
        _ => unreachable!(),
    };
    let opt = |i: usize| c.get(i);
    match name {
        "same_position" => eb(t::same_position(pt(&c[1]), pt(&c[2]))),
        "is_planar" => eb(t::is_planar(&pts(&c[1]), opt(2).map(hex_f64).unwrap_or(1e-3), H)),
        "segment_intersection" => {
            let tol = opt(5).map(hex_f64).unwrap_or(1e-4);
            eopt(t::segment_intersection(pt(&c[1]), pt(&c[2]), pt(&c[3]), pt(&c[4]), tol, H), ep)
        }
        "_order_along" => epl(&t::order_along(pt(&c[1]), pt(&c[2]), &pts(&c[3]), H)),
        "_strictly_inside_2d" => eb(t::strictly_inside_2d(p2(&c[1]), &p2l(&c[2]))),
        "_point_inside_2d" => eb(t::point_inside_2d(p2(&c[1]), &p2l(&c[2]))),
        "_on_segment_2d" => {
            // not public: through strictly_inside_2d's boundary rule is not the same; use the public wrapper
            eb(t::on_segment_2d(p2(&c[1]), p2(&c[2]), p2(&c[3]), 1e-7))
        }
        "find_duplicate_edge" => {
            let es = eids(&c[1]);
            let segs = t::edge_segs(m, &es);
            eopt(t::find_duplicate_edge(&segs, pt(&c[2]), pt(&c[3])), |i| ei(es[i].0 as usize))
        }
        "find_smallest_cycle_through" => {
            let segs = t::edge_segs(m, &eids(&c[1]));
            let max_len = opt(4).map(n).unwrap_or(32);
            eopt(t::find_smallest_cycle_through(&segs, pt(&c[2]), pt(&c[3]), max_len), |v| epl(&v))
        }
        "find_cycles_through" => {
            let segs = t::edge_segs(m, &eids(&c[1]));
            let mr = opt(4).map(n).unwrap_or(2);
            eloops(&t::find_cycles_through(&segs, pt(&c[2]), pt(&c[3]), mr))
        }
        "plan_edge_split" => {
            let es = eids(&c[1]);
            let segs = t::edge_segs(m, &es);
            let (news, cuts) = t::plan_edge_split(&segs, pt(&c[2]), pt(&c[3]), H);
            Value::Array(vec![
                Value::Array(news.iter().map(|&(a, b)| Value::Array(vec![ep(a), ep(b)])).collect()),
                Value::Array(cuts.iter().map(|&(i, p)| Value::Array(vec![ei(es[i].0 as usize), ep(p)])).collect()),
            ])
        }
        "cycle_graph" => {
            let mut g = CycleGraph::new();
            let seg = |m: &Mesh, e: EdgeId| (m.position(m.edge(e).v0), m.position(m.edge(e).v1));
            for op in arr(&c[1]) {
                let op = arr(op);
                match &op[0] {
                    Value::Str(s) if s == "init" => {
                        for e in eids(&op[1]) {
                            let (a, b) = seg(m, e);
                            g.add(e, a, b);
                        }
                    }
                    Value::Str(s) if s == "add" => {
                        let e = EdgeId(n(&op[1]) as u32);
                        let (a, b) = seg(m, e);
                        g.add(e, a, b);
                    }
                    Value::Str(s) if s == "remove" => g.remove(EdgeId(n(&op[1]) as u32)),
                    other => panic!("{other:?}"),
                }
            }
            eloops(&t::find_cycles_through_graph(&g, pt(&c[2]), pt(&c[3]), n(&c[4])))
        }
        "face_exists" => eb(t::face_exists(m, &fids(&c[1]), &pts(&c[2]))),
        "loop_inside_face" => {
            let oh = opt(3).map(|v| matches!(v, Value::Bool(true))).unwrap_or(false);
            eb(t::loop_inside_face(&PFace::of(m, fid(&c[1])), &pts(&c[2]), oh, H))
        }
        "find_containing_face" => {
            let ex = opt(3).map(fid);
            eopt(t::find_containing_face(m, &fids(&c[1]), &pts(&c[2]), ex, H), |f| ei(f.0 as usize))
        }
        "_g_inside_a_hole" => eb(t::g_inside_a_hole(&PFace::of(m, fid(&c[1])), &pts(&c[2]), H)),
        "_point_in_face_solid" => eb(t::point_in_face_solid(&PFace::of(m, fid(&c[1])), pt(&c[2]), H)),
        "orphaned_edges_at" => {
            let r = t::orphaned_edges_at(m, &eids(&c[1]), &fids(&c[2]), &pts(&c[3]));
            Value::Array(r.iter().map(|e| ei(e.0 as usize)).collect())
        }
        "classify_push_edge" => {
            let r = t::classify_push_edge(m, fid(&c[1]), pt(&c[2]), pt(&c[3]), &fids(&c[4]), H);
            match r {
                PushEdge::Coplanar(g) => Value::Array(vec![Value::Str("coplanar".into()), ei(g.0 as usize)]),
                PushEdge::Perp(g) => Value::Array(vec![Value::Str("perp".into()), ei(g.0 as usize)]),
                PushEdge::Free => Value::Array(vec![Value::Str("free".into()), Value::Null]),
            }
        }
        "face_is_bordered" => eb(t::face_is_bordered(m, fid(&c[1]), &fids(&c[2]))),
        "_segment_on_face_boundary" => eb(t::segment_on_face_boundary(pt(&c[1]), pt(&c[2]), &PFace::of(m, fid(&c[3])), H)),
        "_faces_coplanar" => eb(t::faces_coplanar(pt(&c[1]), pt(&c[2]), H)),
        "_unit_coplanar" => eb(t::unit_coplanar(pt(&c[1]), pt(&c[2]))),
        "_point_on_seg_incl" => eb(t::point_on_seg_incl(pt(&c[1]), pt(&c[2]), pt(&c[3]), 1e-4, H)),
        "_point_on_segment_3d" => eb(t::point_on_segment_3d(pt(&c[1]), pt(&c[2]), pt(&c[3]), 1e-4, H)),
        "refine_loop_with_points" => {
            let tol = opt(3).map(hex_f64).unwrap_or(1e-4);
            epl(&t::refine_loop_with_points(&pts(&c[1]), &pts(&c[2]), tol, H))
        }
        "split_face_by_chord" => echunk(t::split_face_by_chord(&PFace::of(m, fid(&c[1])), pt(&c[2]), pt(&c[3]), H)),
        "find_chord_split" => eopt(t::find_chord_split(m, &fids(&c[1]), pt(&c[2]), pt(&c[3]), H), |(f, a, b)| {
            Value::Array(vec![ei(f.0 as usize), epl(&a), epl(&b)])
        }),
        "split_edge_in_faces" => {
            let r = t::split_edge_in_faces(m, &fids(&c[1]), pt(&c[2]), pt(&c[3]), pt(&c[4]), &pts(&c[5]), H);
            Value::Array(r.iter().map(|(f, l)| Value::Array(vec![ei(f.0 as usize), epl(l)])).collect())
        }
        "_locate_on_loop" => eopt(t::locate_on_loop(&pts(&c[1]), pt(&c[2]), H), |l| match l {
            Loc::Vertex(i) => Value::Array(vec![Value::Str("vertex".into()), ei(i)]),
            Loc::Edge(i) => Value::Array(vec![Value::Str("edge".into()), ei(i)]),
        }),
        "find_subdividing_chain" => eopt(t::find_subdividing_chain(&PFace::of(m, fid(&c[1])), &pts(&c[2]), H), |v| epl(&v)),
        "subtract_loop_from_face" => eopt(t::subtract_loop_from_face(&PFace::of(m, fid(&c[1])), &pts(&c[2]), H), |v| epl(&v)),
        "carve_loop_by_chords" => eopt(t::carve_loop_by_chords(&PFace::of(m, fid(&c[1])), &pts(&c[2]), H), |v| eloops(&v)),
        "split_face_by_chain" => echunk(t::split_face_by_chain(&PFace::of(m, fid(&c[1])).vertices, &pts(&c[2]), H)),
        "offset_loop" => eopt(t::offset_loop(&pts(&c[1]), pt(&c[2]), hex_f64(&c[3]), H), |v| epl(&v)),
        "max_offset_distance" => {
            let sign = opt(3).map(hex_f64).unwrap_or(1.0);
            ef(t::max_offset_distance(&pts(&c[1]), pt(&c[2]), sign, 10.0, H))
        }
        "maximal_holes" => {
            let holes: Vec<Vec<V3>> = arr(&c[1]).iter().map(pts).collect();
            Value::Array(t::maximal_holes(&holes, H).into_iter().map(ei).collect())
        }
        "_loop_inside_loop" => eb(t::loop_inside_loop(&pts(&c[1]), &pts(&c[2]), pt(&c[3]), H)),
        "_interior_point" => eopt(t::interior_point(&pts(&c[1]), pt(&c[2]), H), ep),
        "_loop_crosses_itself" => eb(t::loop_crosses_itself(&pts(&c[1]), pt(&c[2]))),
        "_collinear_overlapping" => {
            let segs = t::edge_segs(m, &[EdgeId(n(&c[1]) as u32), EdgeId(n(&c[2]) as u32)]);
            eb(t::collinear_overlapping(segs[0], segs[1], H))
        }
        "_collinear_overlapping_pts" => eb(t::collinear_overlapping((pt(&c[1]), pt(&c[2])), (pt(&c[3]), pt(&c[4])), H)),
        "defects_near" => {
            let vs: Vec<VertexId> = ids(&c[1]).into_iter().map(|i| VertexId(i as u32)).collect();
            let counts = t::defects_near(m, &vs, 1e-10, H);
            Value::Array(
                t::DEFECT_KINDS.iter().zip(counts).map(|(k, n)| Value::Array(vec![Value::Str((*k).into()), ei(n)])).collect(),
            )
        }
        "new_defects" => {
            let arr6 = |v: &Value| -> [usize; 6] {
                let a = ids(v);
                std::array::from_fn(|i| a[i])
            };
            Value::Array(t::new_defects(&arr6(&c[1]), &arr6(&c[2])).into_iter().map(|k| Value::Str(k.into())).collect())
        }
        "_coplanar_overlap" => eb(t::coplanar_overlap(&PFace::of(m, fid(&c[1])), &PFace::of(m, fid(&c[2])), H)),
        "_mesh_is_flat" => eb(t::mesh_is_flat(m, H)),
        "winding_footprint" => {
            let fp = t::winding_footprint(m, H);
            let v = efootprint(&fp);
            fps.push(fp);
            v
        }
        "_footprint_normal" => {
            let fp = fp_arg(fps, &c[1]).expect("footprint");
            eopt(t::footprint_normal(fp, &PFace::of(m, fid(&c[2])), H), ep)
        }
        "orient_coplanar_faces" => {
            let fp = fp_arg(fps, &c[2]).map(|s| s.to_vec());
            efaces(&t::orient_coplanar_faces(m, fp.as_deref(), H))
        }
        "sweep_tjunctions" => {
            let seeds: Option<Vec<[f64; 3]>> = opt(2).filter(|v| !is_null(v)).map(|v| {
                arr(v)
                    .iter()
                    .map(|k| {
                        let a = arr(k);
                        [hex_f64(&a[0]), hex_f64(&a[1]), hex_f64(&a[2])]
                    })
                    .collect()
            });
            t::sweep_tjunctions(m, seeds.as_deref(), H);
            Value::Null
        }
        "resolve_tjunctions" => {
            t::resolve_tjunctions(m, H);
            Value::Null
        }
        "fold_nonplanar_faces" => {
            let tol = opt(2).map(hex_f64).unwrap_or(1e-3);
            efaces(&t::fold_nonplanar_faces(m, tol, H))
        }
        "prune_collinear_orphan_edges" => {
            let r = t::prune_collinear_orphan_edges(m, H);
            Value::Array(r.iter().map(|e| ei(e.0 as usize)).collect())
        }
        "heal_overlapping_faces" => {
            let coverage = hex_f64(&c[2]);
            let partial = match &c[3] {
                Value::Bool(b) => Some(*b),
                _ => None,
            };
            let fp = fp_arg(fps, &c[4]).map(|s| s.to_vec());
            efaces(&t::heal_overlapping_faces(m, coverage, partial, fp.as_deref(), H, &mut |_m: &mut Mesh| {}))
        }
        other => panic!("call {other}"),
    }
}

fn build(m: &mut Mesh, ops: &[Value]) {
    for op in ops {
        let op = arr(op);
        let name = match &op[0] {
            Value::Str(s) => s.as_str(),
            _ => unreachable!(),
        };
        match name {
            "add_face" => {
                let outer = pts(&op[1]);
                let holes: Vec<Vec<V3>> = arr(&op[2]).iter().map(pts).collect();
                t::add_face_py(m, &outer, &holes, H);
            }
            "add_edge" => {
                let _ = m.add_edge(pt(&op[1]), pt(&op[2]));
            }
            "attrs" => {
                if let Value::Object(items) = &op[2] {
                    m.face_mut(fid(&op[1])).attrs = items.clone();
                }
            }
            "remove_face" => m.remove_face(fid(&op[1])),
            other => panic!("op {other}"),
        }
    }
}

#[test]
fn topology_matches_core_topology() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/topology/topology_cases.jsonl");
    let text = std::fs::read_to_string(path).unwrap();
    let mut fails: Vec<String> = Vec::new();
    let (mut calls, mut mutators) = (0usize, 0usize);
    for (si, line) in text.lines().enumerate() {
        let scene = loads(line).unwrap();
        let mut m = Mesh::new();
        build(&mut m, arr(scene.get("build").unwrap()));
        let mut fps: Vec<Option<Vec<FootprintEntry>>> = Vec::new();
        for (ci, rec) in arr(scene.get("calls").unwrap()).iter().enumerate() {
            let c = arr(rec.get("c").unwrap());
            let mutating = rec.get("s").is_some();
            let before = if mutating { Some(canon(&state(&m))) } else { None };
            if mutating {
                m.begin();
            }
            let result = run_call(&mut m, &mut fps, c);
            let entry = if mutating { Some(m.commit()) } else { None };
            let got = canon(&result);
            let want = canon(rec.get("r").unwrap());
            calls += 1;
            if got != want {
                let at = got.bytes().zip(want.bytes()).take_while(|(a, b)| a == b).count();
                let lo = at.saturating_sub(70);
                let name = match &c[0] {
                    Value::Str(s) => s.clone(),
                    _ => String::new(),
                };
                fails.push(format!(
                    "scene {si} call {ci} {name}:\n  rust:   {}\n  python: {}",
                    &got[lo..(at + 70).min(got.len())],
                    &want[lo..(at + 70).min(want.len())]
                ));
                break;
            }
            if let Some(s) = rec.get("s") {
                mutators += 1;
                let got = canon(&state(&m));
                let want = canon(s);
                // Every change went through the journal: undo gives the state
                // before the call, redo the state after it.
                let entry = entry.unwrap();
                m.undo(&entry);
                if canon(&state(&m)) != before.unwrap() {
                    fails.push(format!("scene {si} call {ci} {}: undo does not restore the mesh", canon(&c[0])));
                    break;
                }
                m.redo(&entry);
                if got != want {
                    let at = got.bytes().zip(want.bytes()).take_while(|(a, b)| a == b).count();
                    let lo = at.saturating_sub(70);
                    fails.push(format!(
                        "scene {si} call {ci} {} state:\n  rust:   {}\n  python: {}",
                        canon(&c[0]),
                        &got[lo..(at + 70).min(got.len())],
                        &want[lo..(at + 70).min(want.len())]
                    ));
                    break;
                }
            }
        }
    }
    assert!(fails.is_empty(), "{} scenes diverge ({calls} calls, {mutators} mutator states matched):\n{}", fails.len(), fails.iter().take(10).cloned().collect::<Vec<_>>().join("\n"));
    assert!(calls > 1000, "only {calls} calls");
}
