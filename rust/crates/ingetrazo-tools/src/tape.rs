// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! The Tape Measure tool (`tools/tape.py`): measure distances and place
//! construction guides.
//!
//! Two behaviours, decided by what the first click lands on: from an edge's
//! body (loose, or a group's), a guide line's body or a model axis, dragging
//! pulls out an infinite guide LINE parallel to it at the dragged (or typed)
//! offset; from free space, or a spot on an edge, the second click just
//! measures (the distance is flashed); from a named point (endpoint, origin,
//! intersection...) the second click, or a typed distance, leaves a guide
//! POINT with its dashed segment back to the start. Ctrl taps cycle three
//! modes: guide lines, guide points, measure only. Esc cancels.
//!
//! Where it differs from Python (reported):
//!
//! * The model axes are the world's (`core.axes.AXES` follows the open
//!   group's own axes while one is open for editing: not covered).
//! * `axis_source`, `magnetic_axis_deg` and `screen_axis_px` are read by the
//!   viewport/snap host, not by the tool: they are constants here.

use ingetrazo_formats::pyjson::Value;
use ingetrazo_math::V3;
use ingetrazo_model::guide::Guide;
use ingetrazo_model::units::{fmt_len, format_length, localized, ModelUnits};
use ingetrazo_ops::annotate::AddGuide;
use ingetrazo_snap::{Axis, SnapKind};

use crate::modify_base::{keys, EdgeAny, Env, ModifyTool, Mods, Probe, ToolCx, Vcb};

/// The viewport shows "On axis" while this tool hovers a model axis before
/// its first click.
pub const AXIS_SOURCE: bool = true;
/// The Line tool's axis magnet, both halves.
pub const MAGNETIC_AXIS_DEG: f64 = 3.0;
pub const SCREEN_AXIS_PX: f64 = 9.0;

/// `fmt_len_fine`: `fmt_len` at least to the millimetre.
pub fn fmt_len_fine(metres: f64, units: &ModelUnits) -> String {
    let fine = match units.length.as_str() {
        "m" | "ft" => 3,
        "cm" => 1,
        "mm" => 0,
        _ => 2, // in, ft-in, in-frac, ft-in-frac
    };
    localized(&format_length(metres, &units.length, units.precision.max(fine)), &units.decimal)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mode {
    Line,
    Point,
    Measure,
}

impl Mode {
    const ALL: [Mode; 3] = [Mode::Line, Mode::Point, Mode::Measure];

    fn flash(self) -> &'static str {
        match self {
            Mode::Line => "Guides: guide lines",
            Mode::Point => "Guides: guide points",
            Mode::Measure => "Guides: off \u{2014} measure only",
        }
    }

    fn name(self) -> &'static str {
        match self {
            Mode::Line => "line",
            Mode::Point => "point",
            Mode::Measure => "measure",
        }
    }
}

/// The source of a guide line: anything with an `.a` / `.b` span.
#[derive(Clone, Copy, Debug)]
struct Source {
    a: V3,
    b: V3,
    kind: &'static str,
}

pub struct TapeMeasureTool {
    pub start_point: Option<V3>,
    pub hover_point: Option<V3>,
    edge: Option<Source>,
    from_point: bool,
    measured: Option<f64>,
    mode: Mode,
}

impl Default for TapeMeasureTool {
    fn default() -> Self {
        TapeMeasureTool::new()
    }
}

/// `_POINT_KINDS`: a first click on one of these is a real point.
fn is_point_kind(k: SnapKind) -> bool {
    matches!(k, SnapKind::Endpoint | SnapKind::Origin | SnapKind::Intersection | SnapKind::ComponentOrigin | SnapKind::Center)
}

fn axis_vector(a: Axis) -> V3 {
    match a {
        Axis::X => V3(1.0, 0.0, 0.0),
        Axis::Y => V3(0.0, 1.0, 0.0),
        Axis::Z => V3(0.0, 0.0, 1.0),
    }
}

fn bits(x: f32) -> Value {
    Value::Str(format!("{:08x}", x.to_bits()))
}

impl TapeMeasureTool {
    pub fn new() -> TapeMeasureTool {
        TapeMeasureTool { start_point: None, hover_point: None, edge: None, from_point: false, measured: None, mode: Mode::Line }
    }

    /// `cursor_plus`: this one will leave a guide.
    pub fn cursor_plus(&self) -> bool {
        self.mode != Mode::Measure
    }

    fn reset(&mut self) {
        self.start_point = None;
        self.edge = None;
        self.from_point = false;
    }

    fn toggle_mode(&mut self, env: &mut Env) {
        let i = Mode::ALL.iter().position(|&m| m == self.mode).expect("a mode");
        self.mode = Mode::ALL[(i + 1) % Mode::ALL.len()];
        env.flash(self.mode.flash());
    }

    fn edge_dir(&self, env: &Env) -> V3 {
        let e = self.edge.expect("a source edge");
        e.b.sub(e.a).normalized(env.cx.hypot)
    }

    /// `_guide_offset`: the cursor's offset perpendicular to the source edge.
    fn guide_offset(&self, env: &Env, world: V3) -> Option<V3> {
        let start = self.start_point?;
        self.edge?;
        let d = self.edge_dir(env);
        let delta = world.sub(start);
        Some(delta.sub(d.mul(delta.dot(d))))
    }

    fn place_guide_point(&self, env: &mut Env, at: V3, origin: Option<V3>) {
        let guide = Guide::new(at, None, origin, env.cx.hypot);
        if !guide.is_finite() {
            return; // a degenerate pick: leave nothing (#723)
        }
        env.execute(Box::new(AddGuide::new(guide)));
        let start = self.start_point.expect("a start");
        let d = at.sub(start).length(env.cx.hypot) as f64;
        let text = fmt_len_fine(d, env.units);
        env.flash_for(if origin.is_some() { format!("Guide point at {text}, with its segment") } else { format!("Guide point at {text}") }, 3000);
    }

    fn place_guide(&self, env: &mut Env, offset: V3) {
        let start = self.start_point.expect("a start");
        let guide = Guide::new(start.add(offset), Some(self.edge_dir(env)), None, env.cx.hypot);
        if !guide.is_finite() {
            return; // a degenerate pick: leave nothing (#723)
        }
        env.execute(Box::new(AddGuide::new(guide)));
        let d = offset.length(env.cx.hypot) as f64;
        env.flash_for(format!("Guide at {}", fmt_len_fine(d, env.units)), 3000);
    }
}

impl ModifyTool for TapeMeasureTool {
    fn on_activate(&mut self, _env: &mut Env) {
        // Picking the tool up starts in guide mode, as users expect.
        self.mode = Mode::Line;
        self.reset();
    }

    fn on_deactivate(&mut self, _env: &mut Env) {
        self.reset();
        self.hover_point = None;
    }

    fn on_key(&mut self, _env: &mut Env, key: i32, _mods: Mods) -> bool {
        // Ctrl toggles guide creation, on the RELEASE of a Ctrl pressed alone.
        key == keys::CONTROL
    }

    fn on_key_release(&mut self, env: &mut Env, key: i32) -> bool {
        if key != keys::CONTROL {
            return false;
        }
        if !env.vp.ctrl_tapped() {
            return false; // Ctrl was part of a shortcut
        }
        self.toggle_mode(env);
        true
    }

    fn on_click(&mut self, env: &mut Env, cx: &ToolCx) {
        let h = env.cx.hypot;
        let (sx, sy) = cx.screen;
        let kind = cx.snap_kind();
        let Some(start) = self.start_point else {
            self.start_point = Some(cx.world);
            // pick_edge_any: a loose edge, or a group's as a world pseudo-edge.
            let mut edge: Option<Source> = match env.vp.pick_edge_any(env.scene, env.cx, sx, sy) {
                Some(EdgeAny::Loose(e)) => {
                    let m = env.cx.mesh(env.scene);
                    let ed = m.edge(e);
                    Some(Source { a: m.position(ed.v0), b: m.position(ed.v1), kind: "loose" })
                }
                Some(EdgeAny::Group { a, b }) => Some(Source { a, b, kind: "group" }),
                None => None,
            };
            if edge.is_none() {
                // A guide LINE is a source too.
                if let Some(i) = env.vp.pick_guide(env.scene, env.cx, sx, sy) {
                    let g = &env.scene.guides[i];
                    if g.is_line() {
                        let (a, b) = g.segment();
                        edge = Some(Source { a, b, kind: "guide" });
                    }
                }
            }
            self.from_point = is_point_kind(kind);
            if edge.is_none() && kind == SnapKind::OnAxis {
                if let Some(snap) = cx.snap.as_ref() {
                    if let Some(axis) = snap.axis {
                        // The cue already found the axis and put the point ON it.
                        let g = Guide::new(V3(0.0, 0.0, 0.0), Some(axis_vector(axis)), None, h);
                        let (a, b) = g.segment();
                        edge = Some(Source { a, b, kind: "axis" });
                        self.start_point = Some(snap.point);
                    }
                }
            }
            if edge.is_none() && !self.from_point {
                // The model's own axes are sources too.
                if let Some(axis) = env.vp.pick_axis(env.scene, env.cx, sx, sy) {
                    let dir = axis_vector(axis);
                    let g = Guide::new(V3(0.0, 0.0, 0.0), Some(dir), None, h);
                    let (a, b) = g.segment();
                    edge = Some(Source { a, b, kind: "axis" });
                    self.start_point = Some(dir.mul(cx.world.dot(dir)));
                }
            }
            let excluded = matches!(kind, SnapKind::Endpoint | SnapKind::Midpoint | SnapKind::Close | SnapKind::Origin | SnapKind::Intersection);
            self.edge = if self.mode == Mode::Line && edge.is_some() && !excluded { edge } else { None };
            return;
        };
        if self.mode == Mode::Point {
            // A guide POINT at the measured spot: no source edge needed.
            self.place_guide_point(env, cx.world, None);
        } else if self.edge.is_some() {
            if let Some(offset) = self.guide_offset(env, cx.world) {
                self.place_guide(env, offset);
            }
        } else if self.mode == Mode::Line && self.from_point && !is_point_kind(kind) && (cx.world.sub(start).length(h) as f64) > 1e-9 {
            // From a named point to a free spot: a guide point there, with its segment back.
            self.place_guide_point(env, cx.world, Some(start));
        } else {
            let dist = cx.world.sub(start).length(h) as f64;
            self.measured = Some(dist);
            env.flash_for(format!("Distance: {}", fmt_len_fine(dist, env.units)), 4000);
        }
        self.reset();
    }

    fn on_hover(&mut self, _env: &mut Env, cx: &ToolCx) {
        self.hover_point = Some(cx.world);
    }

    /// Typing a distance places the guide, line or point, exactly there.
    fn on_value(&mut self, env: &mut Env, value: &Vcb) -> bool {
        let h = env.cx.hypot;
        let v = match value {
            Vcb::Num(v) => *v,
            // a tuple (and a tagged length, which is one to this tool)
            Vcb::Tuple(_) | Vcb::AbsLen(_) => return false,
        };
        let (Some(start), Some(hover)) = (self.start_point, self.hover_point) else { return false };
        if self.mode == Mode::Point {
            let d = hover.sub(start);
            if (d.length(h) as f64) < 1e-9 {
                return false;
            }
            self.place_guide_point(env, start.add(d.normalized(h).mul(v as f32)), None);
            self.reset();
            return true;
        }
        if self.edge.is_none() {
            if self.mode == Mode::Line && self.from_point {
                let d = hover.sub(start);
                if (d.length(h) as f64) < 1e-9 {
                    return false;
                }
                self.place_guide_point(env, start.add(d.normalized(h).mul(v as f32)), Some(start));
                self.reset();
                return true;
            }
            return false;
        }
        let Some(offset) = self.guide_offset(env, hover) else { return false };
        if (offset.length(h) as f64) < 1e-9 {
            return false;
        }
        self.place_guide(env, offset.normalized(h).mul(v as f32));
        self.reset();
        true
    }

    fn on_cancel(&mut self, _env: &mut Env) {
        self.reset();
    }

    fn probe(&self, env: &Env) -> Probe {
        let h = env.cx.hypot;
        let mut p = Probe::default();
        let (start, hover) = (self.start_point, self.hover_point);
        p.rb = Some(match (start, hover) {
            (Some(s), Some(hv)) => {
                let offset = if self.edge.is_some() { self.guide_offset(env, hv) } else { None };
                match offset {
                    Some(o) => vec![(s, s.add(o))],
                    None => vec![(s, hv)],
                }
            }
            _ => Vec::new(),
        });
        p.gl = Some(match (self.edge, start, hover) {
            (Some(_), Some(s), Some(hv)) => match self.guide_offset(env, hv) {
                Some(o) => vec![Guide::new(s.add(o), Some(self.edge_dir(env)), None, h).segment()],
                None => Vec::new(),
            },
            _ => Vec::new(),
        });
        p.vl = Some(match (start, hover) {
            (Some(s), Some(hv)) => {
                let d = if self.edge.is_some() {
                    match self.guide_offset(env, hv) {
                        Some(o) => o.length(h) as f64,
                        None => 0.0,
                    }
                } else {
                    hv.sub(s).length(h) as f64
                };
                let mid = s.add(hv).mul(0.5);
                Some((fmt_len(d, env.units), mid))
            }
            _ => None,
        });
        let names = [("line", "guide lines"), ("point", "guide points"), ("measure", "measure")];
        let parts: Vec<String> = names.iter().map(|&(k, v)| if k == self.mode.name() { format!("[{v}]") } else { v.to_string() }).collect();
        p.sc = Some(format!("Ctrl = {}", parts.join(" / ")));
        p.plus = Some(self.cursor_plus());
        p.start = start;
        p.hover = hover;
        p.x = Some(Value::Object(vec![
            ("mode".into(), Value::Str(self.mode.name().into())),
            ("from_point".into(), Value::Bool(self.from_point)),
            ("edge".into(), self.edge.map(|e| Value::Str(e.kind.into())).unwrap_or(Value::Null)),
            ("edge_a".into(), self.edge.map(|e| Value::Array(vec![bits(e.a.0), bits(e.a.1), bits(e.a.2)])).unwrap_or(Value::Null)),
            ("measured".into(), self.measured.map(Value::Float).unwrap_or(Value::Null)),
            ("n_guides".into(), Value::int(env.scene.guides.len() as i64)),
        ]));
        p
    }
}
