// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! The edits planner and the drawing commands against `core.edits` /
//! `core.history` (scripts/golden/make_edits_fixtures.py): random drawing
//! sessions (lines, rectangles, circles, polylines, erases, reversals, undo
//! and redo) on grid coordinates. At every step the Rust planner must build
//! the command list Python's did, and running it must land on the L2
//! signature Python's scene had.
//!
//! A scene with volume asks `heal_overlapping_faces` for `orient_outward`,
//! which is not ported: such a scenario is checked up to that step.

use std::path::Path;

use ingetrazo_formats::pyjson::{loads, Value};
use ingetrazo_golden::{model_sig, sig};
use ingetrazo_math::{Hypot, V3};
use ingetrazo_model::scene::Scene;
use ingetrazo_ops::describe::describe;
use ingetrazo_ops::draw::{AddFaceCommand, EraseSelectionCommand, FlipFacesCommand, TagCurveCommand};
use ingetrazo_ops::edits::{build_add_edge, build_add_edges};
use ingetrazo_ops::rebuild::RebuildPlanarFacesCommand;
use ingetrazo_ops::{Command, CommandError, Env, History};

fn arr(v: &Value) -> &Vec<Value> {
    match v {
        Value::Array(a) => a,
        _ => panic!("{v:?}"),
    }
}

fn pt(v: &Value) -> V3 {
    let a = arr(v);
    let f = |i: usize| match &a[i] {
        Value::Str(h) => f32::from_bits(u32::from_str_radix(h, 16).unwrap()),
        o => panic!("{o:?}"),
    };
    V3(f(0), f(1), f(2))
}

fn pts(v: &Value) -> Vec<V3> {
    arr(v).iter().map(pt).collect()
}

fn segs(v: &Value) -> Vec<(V3, V3)> {
    arr(v).iter().map(|s| (pt(&arr(s)[0]), pt(&arr(s)[1]))).collect()
}

fn extra(v: &Value) -> Vec<Box<dyn Command>> {
    let mut out: Vec<Box<dyn Command>> = Vec::new();
    for x in arr(v) {
        let a = arr(x);
        let name = match &a[0] {
            Value::Str(n) => n.as_str(),
            _ => panic!(),
        };
        match name {
            "AddFace" => out.push(Box::new(AddFaceCommand::new(pts(&a[1]), true, None, None))),
            "TagCurve" => out.push(Box::new(TagCurveCommand { pts: pts(&a[1]), closed: matches!(a[2], Value::Bool(true)) })),
            "RebuildPlanar" => out.push(Box::new(RebuildPlanarFacesCommand::new())),
            other => panic!("{other}"),
        }
    }
    out
}

fn canon(v: &Value) -> String {
    sig::canon_dumps(v)
}

#[test]
fn rust_planner_builds_the_commands_python_built() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/edits_plans.jsonl");
    let text = std::fs::read_to_string(path).unwrap();
    let (mut scenarios, mut full, mut checked, mut bad) = (0, 0, 0, Vec::new());
    for (n, line) in text.lines().enumerate() {
        let sc = loads(line).unwrap();
        scenarios += 1;
        let mut scene = Scene::new();
        scene.meshes[0].hypot = Hypot::Msvc;
        let mut env = Env { hypot: Hypot::Msvc, ..Env::default() };
        if let Some(t @ Value::Object(_)) = sc.get("ifc") {
            env.active_ifc = Some(t.clone());
        }
        let mut hist = History::new();
        let mut complete = true;
        for (i, step) in arr(sc.get("steps").unwrap()).iter().enumerate() {
            let kind = match step.get("kind") {
                Some(Value::Str(k)) => k.as_str(),
                _ => panic!(),
            };
            let want_sig = match step.get("sig") {
                Some(Value::Str(h)) => h.clone(),
                _ => panic!(),
            };
            if kind == "undo" || kind == "redo" {
                let ok = if kind == "undo" { hist.undo(&mut scene, &mut env) } else { hist.redo(&mut scene, &mut env) };
                assert!(ok);
            } else {
                env.curve = step.get("ctr").and_then(Value::as_i64).unwrap();
                let eye = match step.get("eye") {
                    Some(e @ Value::Array(_)) => Some(pt(e)),
                    _ => None,
                };
                let cmd: Box<dyn Command> = match kind {
                    "line" => build_add_edge(&scene, &env, pt(step.get("a").unwrap()), pt(step.get("b").unwrap()), true, eye),
                    "poly" | "rect" | "circle" => {
                        let detect = matches!(step.get("detect"), Some(Value::Bool(true)));
                        let rebuild = matches!(step.get("plane_rebuild"), Some(Value::Bool(true)));
                        let ex = step.get("extra").map(extra).unwrap_or_default();
                        build_add_edges(&scene, &env, &segs(step.get("segs").unwrap()), detect, ex, None, rebuild)
                    }
                    "erase" => {
                        let ends = segs(step.get("edges").unwrap());
                        Box::new(EraseSelectionCommand::new(ends, Vec::new()))
                    }
                    "erase_face" => {
                        let loops = arr(step.get("faces").unwrap()).iter().map(pts).collect();
                        Box::new(EraseSelectionCommand::new(Vec::new(), loops))
                    }
                    "flip" => {
                        let m = env.mesh(&scene);
                        let mut faces = Vec::new();
                        for l in arr(step.get("faces").unwrap()) {
                            let want = pts(l);
                            let mut w: Vec<[u32; 3]> = want.iter().map(|p| [p.0.to_bits(), p.1.to_bits(), p.2.to_bits()]).collect();
                            w.sort();
                            w.dedup();
                            let hit = m.faces.iter().copied().find(|&f| {
                                let mut g: Vec<[u32; 3]> =
                                    m.face(f).outer.iter().map(|&v| m.position(v)).map(|p| [p.0.to_bits(), p.1.to_bits(), p.2.to_bits()]).collect();
                                g.sort();
                                g.dedup();
                                g == w
                            });
                            faces.push((env.active, hit.expect("flip target")));
                        }
                        Box::new(FlipFacesCommand { faces })
                    }
                    other => panic!("{other}"),
                };
                if let (Some(plan), Some(got)) = (step.get("plan"), describe(cmd.as_ref())) {
                    if canon(plan) != canon(&got) {
                        bad.push(format!("scenario {n} step {i} ({kind}): the plan differs\n  want {}\n  got  {}", canon(plan), canon(&got)));
                        complete = false;
                        break;
                    }
                }
                match hist.execute(cmd, &mut scene, &mut env) {
                    Ok(()) => {}
                    Err(CommandError::Unsupported(_)) => {
                        complete = false;
                        break;
                    }
                    Err(e) => {
                        bad.push(format!("scenario {n} step {i} ({kind}): {e:?}"));
                        complete = false;
                        break;
                    }
                }
            }
            let got = sig::signature_hash(&model_sig::scene_signature(&scene, &sig::Files { zip: None }));
            if got != want_sig {
                bad.push(format!("scenario {n} step {i} ({kind}): the scene differs after it"));
                complete = false;
                break;
            }
            checked += 1;
        }
        if complete {
            full += 1;
        }
    }
    eprintln!("{scenarios} scenarios, {full} to the end, {checked} steps checked");
    assert!(bad.is_empty(), "{} problems:\n{}", bad.len(), bad.iter().take(8).cloned().collect::<Vec<_>>().join("\n"));
    assert!(full * 2 > scenarios, "only {full} of {scenarios} scenarios ran to the end");
}
