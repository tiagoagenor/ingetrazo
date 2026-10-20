// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! The snap engine's vocabulary (`core/snap.py`): colours, kinds, the
//! result, the edge stand-ins it reads and the drawing axes
//! (`core/axes.py`).

use ingetrazo_math::{Hypot, V3};

/// An RGB triple of Python floats.
pub type Rgb = [f64; 3];

pub const COLOR_ENDPOINT: Rgb = [0.16, 0.62, 0.36];
pub const COLOR_MIDPOINT: Rgb = [0.20, 0.66, 0.74];
pub const COLOR_ON_EDGE: Rgb = [0.86, 0.22, 0.27];
pub const COLOR_ON_FACE: Rgb = [0.42, 0.46, 0.92];
pub const COLOR_ORIGIN: Rgb = [0.95, 0.45, 0.16];
pub const COLOR_CLOSE: Rgb = [0.20, 0.40, 0.78];
pub const COLOR_AXIS_X: Rgb = [0.86, 0.22, 0.27];
pub const COLOR_AXIS_Y: Rgb = [0.16, 0.62, 0.36];
pub const COLOR_AXIS_Z: Rgb = [0.20, 0.40, 0.78];
pub const COLOR_REFERENCE: Rgb = [0.85, 0.30, 0.80];
pub const COLOR_EXTENSION: Rgb = [0.55, 0.55, 0.58];
pub const COLOR_NONE: Rgb = [0.0, 0.0, 0.0];

/// A drawing axis. Python keys the axes by `"x"`, `"y"`, `"z"` and walks
/// them in that order.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Axis {
    X,
    Y,
    Z,
}

impl Axis {
    pub const ALL: [Axis; 3] = [Axis::X, Axis::Y, Axis::Z];

    pub fn name(self) -> &'static str {
        match self {
            Axis::X => "x",
            Axis::Y => "y",
            Axis::Z => "z",
        }
    }

    pub fn from_name(s: &str) -> Option<Axis> {
        match s {
            "x" => Some(Axis::X),
            "y" => Some(Axis::Y),
            "z" => Some(Axis::Z),
            _ => None,
        }
    }

    /// `AXIS_COLORS[name]`.
    pub fn color(self) -> Rgb {
        match self {
            Axis::X => COLOR_AXIS_X,
            Axis::Y => COLOR_AXIS_Y,
            Axis::Z => COLOR_AXIS_Z,
        }
    }

    fn idx(self) -> usize {
        self as usize
    }
}

/// `core.axes`: the drawing axes, the world's at the top level and the open
/// group's own inside it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Axes {
    pub x: V3,
    pub y: V3,
    pub z: V3,
    pub origin: V3,
    /// `is_world()`: no frame is synced.
    pub world: bool,
}

impl Default for Axes {
    fn default() -> Self {
        Axes::world()
    }
}

impl Axes {
    pub fn world() -> Axes {
        Axes {
            x: V3(1.0, 0.0, 0.0),
            y: V3(0.0, 1.0, 0.0),
            z: V3(0.0, 0.0, 1.0),
            origin: V3(0.0, 0.0, 0.0),
            world: true,
        }
    }

    /// `AXES[name]`.
    pub fn axis(&self, a: Axis) -> V3 {
        [self.x, self.y, self.z][a.idx()]
    }

    /// `to_local(v)`: a world direction in drawing-axes components.
    pub fn to_local(&self, v: V3) -> V3 {
        V3(v.dot(self.x), v.dot(self.y), v.dot(self.z))
    }
}

/// `SnapResult.context`: the tip adds "in group" / "in component".
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Context {
    Group,
    Component,
}

/// The kinds `compute_snap` and the engine around it answer with (the
/// strings of `SnapResult.kind`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SnapKind {
    Axis,
    Reference,
    Close,
    Endpoint,
    Intersection,
    Midpoint,
    Origin,
    OnEdge,
    OnLine,
    Aligned,
    AxisInference,
    OnFace,
    None,
    Center,
    ComponentOrigin,
    ArcMidpoint,
    FromPoint,
    Extension,
    ThroughPoint,
    PerpFace,
    OnAxis,
}

impl SnapKind {
    pub fn as_str(self) -> &'static str {
        match self {
            SnapKind::Axis => "axis",
            SnapKind::Reference => "reference",
            SnapKind::Close => "close",
            SnapKind::Endpoint => "endpoint",
            SnapKind::Intersection => "intersection",
            SnapKind::Midpoint => "midpoint",
            SnapKind::Origin => "origin",
            SnapKind::OnEdge => "on_edge",
            SnapKind::OnLine => "on_line",
            SnapKind::Aligned => "aligned",
            SnapKind::AxisInference => "axis_inference",
            SnapKind::OnFace => "on_face",
            SnapKind::None => "none",
            SnapKind::Center => "center",
            SnapKind::ComponentOrigin => "component_origin",
            SnapKind::ArcMidpoint => "arc_midpoint",
            SnapKind::FromPoint => "from_point",
            SnapKind::Extension => "extension",
            SnapKind::ThroughPoint => "through_point",
            SnapKind::PerpFace => "perp_face",
            SnapKind::OnAxis => "on_axis",
        }
    }

    /// `POINT_SNAP_KINDS`: the kinds that name a real point of the model.
    pub fn is_point(self) -> bool {
        matches!(
            self,
            SnapKind::Endpoint
                | SnapKind::Midpoint
                | SnapKind::Intersection
                | SnapKind::OnEdge
                | SnapKind::OnLine
                | SnapKind::Close
                | SnapKind::Origin
                | SnapKind::Center
                | SnapKind::ComponentOrigin
                | SnapKind::ArcMidpoint
        )
    }
}

/// `SnapResult`.
#[derive(Clone, Debug, PartialEq)]
pub struct SnapResult {
    pub point: V3,
    pub kind: SnapKind,
    pub color: Rgb,
    pub axis: Option<Axis>,
    /// The dashed guide to draw.
    pub guide: Option<(V3, V3)>,
    pub guide_color: Option<Rgb>,
    /// Extra dashed guides, `(a, b, rgb)`.
    pub guides: Option<Vec<(V3, V3, Rgb)>>,
    pub context: Option<Context>,
    /// An extension's own words for the ScreenTip, already translated.
    pub label: Option<String>,
}

impl SnapResult {
    pub fn new(point: V3, kind: SnapKind, color: Rgb) -> SnapResult {
        SnapResult { point, kind, color, axis: None, guide: None, guide_color: None, guides: None, context: None, label: None }
    }
}

/// The edge stand-in the engine reads (`Edge` and the viewport's
/// `_SnapEdge`): world endpoints and the flags `compute_snap` asks of them.
#[derive(Clone, Debug, PartialEq)]
pub struct SnapEdge {
    pub a: V3,
    pub b: V3,
    pub soft: bool,
    pub hidden: bool,
    pub curve: Option<i64>,
    /// A face-me figure's feet.
    pub figure: bool,
    /// The degenerate pseudo-edge that carries a circle's centre.
    pub center: bool,
    pub component_origin: bool,
    pub arc_midpoint: bool,
    /// A construction guide line: 'on line'.
    pub guide: bool,
    pub context: Option<Context>,
}

impl SnapEdge {
    pub fn new(a: V3, b: V3) -> SnapEdge {
        SnapEdge {
            a,
            b,
            soft: false,
            hidden: false,
            curve: None,
            figure: false,
            center: false,
            component_origin: false,
            arc_midpoint: false,
            guide: false,
            context: None,
        }
    }
}

/// The reference edge of the Down-arrow lock and the acquired edge: only
/// their endpoints are read.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EdgeRef {
    pub a: V3,
    pub b: V3,
}

/// What `compute_snap` asks of the viewport: the callbacks of its
/// signature, already bound to the cursor.
pub trait SnapEnv {
    /// `world_to_pixel`: `None` behind the camera.
    fn world_to_pixel(&self, w: V3) -> Option<(f64, f64)>;
    /// `project_onto_line(start, dir)`: the closest point of the lock line
    /// to the cursor ray.
    fn project_onto_line(&self, start: V3, dir: V3) -> V3;
    /// `is_occluded(world)`.
    fn is_occluded(&self, w: V3) -> bool;
}

/// Shared numeric context: the `std::hypot` formula of this Qt and the
/// drawing axes.
#[derive(Clone, Copy, Debug)]
pub struct Cx {
    pub h: Hypot,
    pub axes: Axes,
}

impl Cx {
    /// `v.length()`, widened to a Python float.
    #[inline]
    pub fn len(&self, v: V3) -> f64 {
        v.length(self.h) as f64
    }

    /// `QVector3D.dotProduct(a, b)` as a Python float.
    #[inline]
    pub fn dot(&self, a: V3, b: V3) -> f64 {
        a.dot(b) as f64
    }

    #[inline]
    pub fn norm(&self, v: V3) -> V3 {
        v.normalized(self.h)
    }
}
