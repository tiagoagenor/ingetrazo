// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! Golden ops of the drawing commands (M3.1), decoded into `ingetrazo-ops`
//! commands: `add_edge`, `add_face`, `delete_*`, `erase_selection`,
//! `flip_faces`, `snapshot_compound`, `compound`, the plane rebuilds, face
//! paint, and the rest of `draw.rs`.
//!
//! A command NESTED in another (`{"$cmd": op}`) is decoded when it RUNS
//! (`Deferred`), against the scene as it is then, as the Python recorder's
//! `from_op` does: a selector inside a compound sees what the commands
//! before it did. `$ran` refers to a sibling that already ran in the same
//! step; the registry of those is per thread and per step ([`new_step`]).

use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::rc::Rc;

use ingetrazo_formats::pyjson::Value;
use ingetrazo_math::{Hypot, V3};
use ingetrazo_model::mesh::{FaceId, Mesh};
use ingetrazo_model::scene::{MeshId, Scene};
use ingetrazo_model::stitch::{CoplanarMergeCommand, PruneOrphanEdgesCommand, StitchSolidCommand};
use ingetrazo_ops::command::{Command, Cx, Fail};
use ingetrazo_ops::draw::*;
use ingetrazo_ops::rebuild::{RebuildPlanarFacesCommand, RebuildPlaneFacesCommand};

/// The prefix of the failure a nested op nobody builds yet leaves in
/// `History::last_error`; the runner reports it as `Unsupported`.
pub const UNSUPPORTED: &str = "UNSUPPORTED ";


/// Why a step's ops could not be decoded.
#[derive(Debug)]
pub enum DecodeError {
    /// An op the Rust engine does not have (yet).
    Unsupported(String),
    /// The script does not say what the schema promises.
    Bad(String),
}

type D<T> = Result<T, DecodeError>;

fn bad<T>(msg: impl Into<String>) -> D<T> {
    Err(DecodeError::Bad(msg.into()))
}

/// Commands that ran in the current step and carry a `rid` (`$ran`).
#[derive(Default)]
pub struct Registry {
    faces: HashMap<i64, Rc<Cell<Option<FaceId>>>>,
}

type Reg = Rc<RefCell<Registry>>;

// ---- values ------------------------------------------------------------------

fn num(v: &Value) -> Option<f64> {
    match v {
        Value::Int(s) => s.parse::<f64>().ok(),
        Value::Float(x) => Some(*x),
        _ => None,
    }
}

fn arr(v: &Value) -> D<&Vec<Value>> {
    match v {
        Value::Array(a) => Ok(a),
        _ => bad(format!("expected a list, got {v:?}")),
    }
}

fn point(v: &Value) -> D<V3> {
    // `{"$v": [x, y, z]}` or the bare `[x, y, z]` of a selector.
    let inner = v.get("$v").unwrap_or(v);
    let a = arr(inner)?;
    if a.len() != 3 {
        return bad("a point has three numbers");
    }
    let c = |i: usize| num(&a[i]).map(|x| x as f32).ok_or_else(|| DecodeError::Bad("a coordinate".into()));
    Ok(V3(c(0)?, c(1)?, c(2)?))
}

fn points(v: &Value) -> D<Vec<V3>> {
    arr(v)?.iter().map(point).collect()
}

fn arg<'a>(args: &'a Value, name: &str) -> Option<&'a Value> {
    args.get(name).filter(|v| !matches!(v, Value::Null))
}

/// A plain value with the schema's wrappers undone: a tuple (`$t`) is a
/// list to everything that reads it (`json.dumps` writes it as one).
fn plain(v: &Value) -> Value {
    match v {
        Value::Object(o) if o.len() == 1 && o[0].0 == "$t" => plain(&o[0].1),
        Value::Object(o) => Value::Object(o.iter().map(|(k, x)| (k.clone(), plain(x))).collect()),
        Value::Array(a) => Value::Array(a.iter().map(plain).collect()),
        o => o.clone(),
    }
}

fn dict_of(v: &Value) -> D<Vec<(String, Value)>> {
    match plain(v) {
        Value::Object(o) => Ok(o),
        _ => bad("expected a dict"),
    }
}

fn group_mesh(scene: &Scene, path: &Value) -> D<MeshId> {
    match path {
        Value::Null => Ok(MeshId(0)),
        Value::Array(p) => {
            let mut list = &scene.groups;
            let mut found = None;
            for i in p {
                let i = i.as_i64().ok_or_else(|| DecodeError::Bad("group path".into()))? as usize;
                let g = list.get(i).ok_or_else(|| DecodeError::Bad("group path out of range".into()))?;
                found = Some(g);
                list = &g.children;
            }
            found.map(|g| g.mesh).ok_or_else(|| DecodeError::Bad("empty group path".into()))
        }
        _ => bad("group path"),
    }
}

fn bits(p: V3) -> [u32; 3] {
    [p.0.to_bits(), p.1.to_bits(), p.2.to_bits()]
}

/// `{"$face": {"loop": [...], "in": path}}`: the loop, and the face of that
/// mesh with exactly these corners.
fn face_sel(scene: &Scene, v: &Value) -> D<(Vec<V3>, MeshId, Option<FaceId>)> {
    // `core.geometry.Face` passed by value: its positions, no mesh object.
    if let Some(Value::Str(cls)) = v.get("$obj") {
        if cls == "geometry.Face" {
            let data = v.get("data").ok_or_else(|| DecodeError::Bad("face data".into()))?;
            let lp = points(data.get("vertices").ok_or_else(|| DecodeError::Bad("vertices".into()))?)?;
            return Ok((lp, MeshId(0), None));
        }
    }
    let body = v.get("$face").ok_or_else(|| DecodeError::Bad("a $face selector".into()))?;
    let lp = points(body.get("loop").ok_or_else(|| DecodeError::Bad("loop".into()))?)?;
    let mesh = group_mesh(scene, body.get("in").unwrap_or(&Value::Null))?;
    let m = &scene.meshes[mesh.0 as usize];
    let mut want: Vec<[u32; 3]> = lp.iter().map(|&p| bits(p)).collect();
    want.sort();
    want.dedup();
    let hit = m.faces.iter().copied().find(|&f| {
        let mut got: Vec<[u32; 3]> = m.face(f).outer.iter().map(|&v| bits(m.position(v))).collect();
        got.sort();
        got.dedup();
        got == want
    });
    Ok((lp, mesh, hit))
}

/// `{"$edge": {"ends": [a, b], "in": path}}`.
fn edge_sel(scene: &Scene, v: &Value) -> D<(V3, V3, MeshId, Option<ingetrazo_model::EdgeId>)> {
    // `core.geometry.Edge` passed by value: two positions, no mesh object.
    if let Some(Value::Str(cls)) = v.get("$obj") {
        if cls == "geometry.Edge" {
            let data = v.get("data").ok_or_else(|| DecodeError::Bad("edge data".into()))?;
            let a = point(data.get("a").ok_or_else(|| DecodeError::Bad("a".into()))?)?;
            let b = point(data.get("b").ok_or_else(|| DecodeError::Bad("b".into()))?)?;
            return Ok((a, b, MeshId(0), None));
        }
    }
    let body = v.get("$edge").ok_or_else(|| DecodeError::Bad("an $edge selector".into()))?;
    let ends = points(body.get("ends").ok_or_else(|| DecodeError::Bad("ends".into()))?)?;
    if ends.len() != 2 {
        return bad("an edge has two ends");
    }
    let mesh = group_mesh(scene, body.get("in").unwrap_or(&Value::Null))?;
    let m: &Mesh = &scene.meshes[mesh.0 as usize];
    let (a, b) = (bits(ends[0]), bits(ends[1]));
    let hit = m.edges.iter().copied().find(|&e| {
        let ed = m.edge(e);
        let (p, q) = (bits(m.position(ed.v0)), bits(m.position(ed.v1)));
        (p == a && q == b) || (p == b && q == a)
    });
    Ok((ends[0], ends[1], mesh, hit))
}

// ---- commands ----------------------------------------------------------------

/// A command decoded when it first runs.
struct Deferred {
    op: Value,
    reg: Reg,
    built: Option<Box<dyn Command>>,
    name: &'static str,
}

impl Deferred {
    fn boxed(op: Value, reg: &Reg) -> Box<dyn Command> {
        Box::new(Deferred { op, reg: reg.clone(), built: None, name: "Deferred" })
    }
}

impl Command for Deferred {
    fn apply(&mut self, scene: &mut Scene, cx: &mut Cx) -> Result<(), Fail> {
        if self.built.is_none() {
            match decode(&self.op, scene, &self.reg) {
                Ok(c) => {
                    self.name = c.name();
                    self.built = Some(c);
                }
                Err(DecodeError::Unsupported(op)) => return Err(Fail::Failed(format!("{UNSUPPORTED}{op}"))),
                Err(DecodeError::Bad(why)) => return Err(Fail::Failed(format!("decode: {why}"))),
            }
        }
        match self.built.as_mut() {
            Some(c) => c.apply(scene, cx),
            None => Ok(()),
        }
    }
    fn undo(&mut self, scene: &mut Scene, cx: &mut Cx) {
        if let Some(c) = self.built.as_mut() {
            c.undo(scene, cx);
        }
    }
    fn name(&self) -> &'static str {
        self.name
    }
}

thread_local! {
    static REG: Reg = Rc::new(RefCell::new(Registry::default()));
}

/// A new step begins: no command has run in it yet.
pub fn new_step() {
    REG.with(|r| r.borrow_mut().faces.clear());
}

/// The command of a top-level step `op`, or `None` when this module does not
/// know the op (the shape `run::build_command` chains).
pub fn build(op: &str, args: &Value, scene: &Scene) -> Option<Result<Box<dyn Command>, String>> {
    let items = vec![("op".to_string(), Value::Str(op.to_string())), ("args".to_string(), args.clone())];
    let reg = REG.with(|r| r.clone());
    match decode(&Value::Object(items), scene, &reg) {
        Ok(c) => Some(Ok(c)),
        Err(DecodeError::Unsupported(_)) => None,
        Err(DecodeError::Bad(why)) => Some(Err(why)),
    }
}

fn nested(v: Option<&Value>, reg: &Reg) -> D<Vec<Box<dyn Command>>> {
    let Some(v) = v else { return Ok(Vec::new()) };
    let mut out = Vec::new();
    for item in arr(v)? {
        let op = item.get("$cmd").ok_or_else(|| DecodeError::Bad("a $cmd".into()))?;
        out.push(Deferred::boxed(op.clone(), reg));
    }
    Ok(out)
}

fn endpoints(scene: &Scene, v: Option<&Value>) -> D<Vec<(V3, V3)>> {
    let Some(v) = v else { return Ok(Vec::new()) };
    arr(v)?.iter().map(|e| edge_sel(scene, e).map(|(a, b, _, _)| (a, b))).collect()
}

/// Decode one op (`{"op", "args", "rid"?}`) against the scene as it is now.
pub fn decode(op: &Value, scene: &Scene, reg: &Reg) -> D<Box<dyn Command>> {
    let name = match op.get("op") {
        Some(Value::Str(n)) => n.as_str(),
        _ => return bad("an op without a name"),
    };
    let empty = Value::Object(Vec::new());
    let args = op.get("args").unwrap_or(&empty);
    let rid = op.get("rid").and_then(Value::as_i64);
    Ok(match name {
        "add_edge" => {
            let a = point(args.get("a").ok_or_else(|| DecodeError::Bad("a".into()))?)?;
            let b = point(args.get("b").ok_or_else(|| DecodeError::Bad("b".into()))?)?;
            let soft = match args.get("soft") {
                Some(Value::Bool(s)) => Some(*s),
                _ => None,
            };
            let curve = args.get("curve").and_then(Value::as_i64);
            Box::new(AddEdgeCommand::new(a, b, soft, curve))
        }
        "planned_when_run" => {
            let mut segments = Vec::new();
            for s in arr(args.get("segments").ok_or_else(|| DecodeError::Bad("segments".into()))?)? {
                let pair = arr(s.get("$t").unwrap_or(s))?;
                if pair.len() != 2 {
                    return bad("a segment has two ends");
                }
                segments.push((point(&pair[0])?, point(&pair[1])?));
            }
            Box::new(ingetrazo_ops::edits::PlannedWhenRun::new(segments))
        }
        "simplify_mesh" => {
            let angle = arg(args, "max_angle_degrees").and_then(num).unwrap_or(0.0);
            Box::new(ingetrazo_ops::edits::SimplifyMeshCommand::new(angle))
        }
        "add_face" => {
            let vertices = points(args.get("vertices").ok_or_else(|| DecodeError::Bad("vertices".into()))?)?;
            let auto = !matches!(args.get("auto"), Some(Value::Bool(false)));
            let holes = match arg(args, "holes") {
                Some(h) => Some(arr(h)?.iter().map(points).collect::<D<Vec<_>>>()?),
                None => None,
            };
            let attrs = match arg(args, "attrs") {
                Some(a) => Some(dict_of(a)?),
                None => None,
            };
            let cmd = AddFaceCommand::new(vertices, auto, holes, attrs);
            if let Some(r) = rid {
                reg.borrow_mut().faces.insert(r, cmd.face.clone());
            }
            Box::new(cmd)
        }
        "delete_edges" => {
            let cascade = !matches!(args.get("cascade_faces"), Some(Value::Bool(false)));
            Box::new(DeleteEdgesCommand::new(endpoints(scene, args.get("edges"))?, cascade))
        }
        "delete_face" => {
            let (lp, _, _) = face_sel(scene, args.get("face").ok_or_else(|| DecodeError::Bad("face".into()))?)?;
            Box::new(DeleteFaceCommand::new(lp))
        }
        "erase_selection" => {
            let edges = endpoints(scene, args.get("edges"))?;
            let mut faces = Vec::new();
            if let Some(fs) = args.get("faces") {
                for f in arr(fs)? {
                    faces.push(face_sel(scene, f)?.0);
                }
            }
            Box::new(EraseSelectionCommand::new(edges, faces))
        }
        "flip_faces" => {
            let mut faces = Vec::new();
            if let Some(fs) = args.get("faces") {
                for f in arr(fs)? {
                    let (_, mesh, hit) = face_sel(scene, f)?;
                    if let Some(h) = hit {
                        faces.push((mesh, h));
                    }
                }
            }
            Box::new(FlipFacesCommand { faces })
        }
        "tag_curve" => {
            let pts = points(args.get("loop_points").ok_or_else(|| DecodeError::Bad("loop_points".into()))?)?;
            let closed = !matches!(args.get("closed"), Some(Value::Bool(false)));
            Box::new(TagCurveCommand { pts, closed })
        }
        "snapshot_compound" => Box::new(SnapshotCompound::new(nested(args.get("inner"), reg)?)),
        "compound" => Box::new(CompoundCommand { commands: nested(args.get("commands"), reg)? }),
        "mesh_snapshot" => {
            let seed = match args.get("stitch_seed") {
                Some(s) => points(s)?,
                None => Vec::new(),
            };
            Box::new(MeshSnapshotCommand::new(nested(args.get("inner"), reg)?, seed))
        }
        "auto_tag_drawn_faces" => {
            let mut face_commands = Vec::new();
            for r in arr(args.get("face_commands").ok_or_else(|| DecodeError::Bad("face_commands".into()))?)? {
                let n = r.get("$ran").and_then(Value::as_i64).ok_or_else(|| DecodeError::Bad("$ran".into()))?;
                let h = reg.borrow().faces.get(&n).cloned();
                face_commands.push(h.ok_or_else(|| DecodeError::Bad(format!("$ran {n} did not run")))?);
            }
            let tag = dict_of(args.get("tag").ok_or_else(|| DecodeError::Bad("tag".into()))?)?;
            Box::new(AutoTagDrawnFacesCommand { face_commands, tag })
        }
        "prune_on_undo" => Box::new(PruneOnUndo::default()),
        "stitch_solid" => Box::new(StitchSolidCommand::new(&points(args.get("seed_positions").ok_or_else(|| DecodeError::Bad("seed".into()))?)?)),
        "coplanar_merge" => Box::new(CoplanarMergeCommand::new(&points(args.get("seed_positions").ok_or_else(|| DecodeError::Bad("seed".into()))?)?)),
        "prune_orphan_edges" => Box::new(PruneOrphanEdgesCommand::new(&points(args.get("vertices").ok_or_else(|| DecodeError::Bad("vertices".into()))?)?)),
        "rebuild_planar_faces" => Box::new(RebuildPlanarFacesCommand::new()),
        "rebuild_plane_faces" => {
            let o = point(args.get("origin").ok_or_else(|| DecodeError::Bad("origin".into()))?)?;
            let n = point(args.get("normal").ok_or_else(|| DecodeError::Bad("normal".into()))?)?;
            Box::new(RebuildPlaneFacesCommand::from_args(o, n, Hypot::Msvc))
        }
        "set_face_color" | "set_face_opacity" | "set_face_material_tag" | "set_face_texture" | "set_face_back" => {
            let mut faces = Vec::new();
            if let Some(fs) = args.get("faces") {
                for f in arr(fs)? {
                    let (_, mesh, hit) = face_sel(scene, f)?;
                    if let Some(h) = hit {
                        faces.push((mesh, h));
                    }
                }
            }
            let nul = |k: &str| arg(args, k).map(plain);
            match name {
                "set_face_color" => Box::new(SetFaceAttrCommand::color(faces, nul("color"))),
                "set_face_texture" => Box::new(SetFaceAttrCommand::texture(faces, nul("texture"))),
                "set_face_back" => {
                    let n = faces.len();
                    let backs: Vec<Option<Value>> = match arg(args, "backs").map(plain) {
                        None => vec![None; n],
                        Some(Value::Array(items)) => items.into_iter().map(|b| if matches!(b, Value::Object(_)) { Some(b) } else { None }).collect(),
                        Some(b @ Value::Object(_)) => vec![Some(b); n],
                        Some(_) => vec![None; n],
                    };
                    Box::new(SetFaceBackCommand::new(faces, backs))
                }
                "set_face_opacity" => Box::new(SetFaceAttrCommand::opacity(faces, nul("opacity").map(|v| match v {
                    Value::Int(s) => Value::Float(s.parse().unwrap_or(0.0)),
                    o => o,
                }))),
                _ => {
                    // `material` is a Material passed by value (`$obj`): its to_dict().
                    let name = nul("name");
                    let material = match (arg(args, "material"), &name) {
                        (Some(m), Some(Value::Str(n))) => match m.get("data") {
                            Some(d) => Some((n.clone(), plain(d))),
                            None => return bad("a material without data"),
                        },
                        _ => None,
                    };
                    Box::new(SetFaceAttrCommand::material_tag(faces, name, material))
                }
            }
        }
        "hide_edges" => {
            let mut edges = Vec::new();
            if let Some(es) = args.get("edges") {
                for e in arr(es)? {
                    let (_, _, mesh, hit) = edge_sel(scene, e)?;
                    if let Some(h) = hit {
                        edges.push((mesh, h));
                    }
                }
            }
            let hidden = !matches!(args.get("hidden"), Some(Value::Bool(false)));
            Box::new(HideEdgesCommand::new(edges, hidden))
        }
        "explode_curves" => {
            let mut ids = Vec::new();
            if let Some(es) = args.get("edges") {
                for e in arr(es)? {
                    let (_, _, mesh, hit) = edge_sel(scene, e)?;
                    if let Some(h) = hit {
                        if let Some(c) = scene.meshes[mesh.0 as usize].edge(h).curve {
                            if !ids.contains(&c) {
                                ids.push(c);
                            }
                        }
                    }
                }
            }
            Box::new(ExplodeCurvesCommand::new(ids))
        }
        other => match crate::run::build_others(other, args, scene) {
            Some(Ok(c)) => c,
            Some(Err(e)) => return bad(e),
            None => return Err(DecodeError::Unsupported(other.to_string())),
        },
    })
}

