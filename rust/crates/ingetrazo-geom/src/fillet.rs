// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! The geometry of `core/fillet.py`: planning the rounding (fillet / bevel)
//! of the edges of a solid, line by line with the same arithmetic.
//!
//! What the Python `plan_fillet` reads off `core.mesh` objects (edges, faces,
//! vertices compared with `is`) is a plain [`TopoMesh`] here, with indices
//! for identities. What it returns is a [`FilletPlan`] of plain data —
//! positions, loops, indices of the faces and edges to remove — for the
//! mesh code to apply (`apply_fillet` stays with the Rust mesh, M2/M3).
//! Messages are a [`FilletError`]; its [`message`](FilletError::message) is
//! the English source text `tr()` gets.
//!
//! Orders the Python leaves to hash order are fixed here: the faces of a
//! vertex come in ascending index (`TopoMesh::vertex_faces`) and its edges
//! in the order of `TopoVertex::edges`, which the caller sets.
//!
//! QVector3D is float32 (`V3`); a Python float multiplied into one is
//! rounded to f32 first; Python float arithmetic and `math.*` are f64.

use std::collections::HashMap;
use std::hash::Hash;

use ingetrazo_math::pyfloat::{py_max2, py_min2, py_sum};
use ingetrazo_math::{Hypot, V3};

use crate::facegeom::plane_axes;

/// CPython 3.12's `sum()` of floats is compensated (Neumaier).
pub(crate) const PY_SUM_COMPENSATED: bool = true;

/// Faces meeting at less than this angle (or more than 180° minus it) are
/// too flat or too sharp to round.
pub const MIN_ANGLE_DEG: f64 = 2.0;
/// Rebuilt loops may leave their face plane by this much (relative).
pub const PLANE_TOL: f64 = 1e-4;
/// The profile that is the circular arc — the plain fillet.
pub const CIRCULAR: f64 = 0.5;
/// Two computed points closer than this are the SAME point.
const SAME: f64 = 1e-5;
/// New edges between faces meeting at less than this read as one surface.
pub const SMOOTH_DEG: f64 = 30.0;
/// The `Face.attrs` keys a strip inherits (in the order fillet reads them).
pub const STRIP_KEYS: [&str; 4] = ["color", "texture", "mat", "opacity"];

const PI: f64 = std::f64::consts::PI;
const DEG_TO_RAD: f64 = PI / 180.0; // CPython's math.radians factor

// ---- inputs ---------------------------------------------------------------

/// A vertex: its position and its incident edges (the order the caller
/// chooses; Python iterates a set).
#[derive(Clone, Debug)]
pub struct TopoVertex {
    pub position: V3,
    pub edges: Vec<usize>,
}

/// An edge between two vertices with its radial list of faces, in order.
#[derive(Clone, Debug)]
pub struct TopoEdge {
    pub v0: usize,
    pub v1: usize,
    pub faces: Vec<usize>,
}

/// A face: outer loop and hole loops of vertex indices, and the paint
/// attributes (`STRIP_KEYS` present in `Face.attrs`, values as opaque text).
#[derive(Clone, Debug, Default)]
pub struct TopoFace {
    pub loop_: Vec<usize>,
    pub holes: Vec<Vec<usize>>,
    pub paint: Vec<(String, String)>,
}

/// The slice of `core.mesh.Mesh` the planners read.
#[derive(Clone, Debug, Default)]
pub struct TopoMesh {
    pub vertices: Vec<TopoVertex>,
    pub edges: Vec<TopoEdge>,
    pub faces: Vec<TopoFace>,
}

impl TopoMesh {
    pub fn pos(&self, v: usize) -> V3 {
        self.vertices[v].position
    }

    /// `Vertex.faces()` as a list, ascending by face index.
    pub fn vertex_faces(&self, v: usize) -> Vec<usize> {
        let mut out: Vec<usize> = Vec::new();
        for &e in &self.vertices[v].edges {
            for &f in &self.edges[e].faces {
                if !out.contains(&f) {
                    out.push(f);
                }
            }
        }
        out.sort_unstable();
        out
    }

    /// `Edge.other`.
    pub fn other(&self, e: usize, v: usize) -> usize {
        let ed = &self.edges[e];
        if v == ed.v0 {
            ed.v1
        } else {
            ed.v0
        }
    }

    /// `Edge.length()`.
    pub fn edge_length(&self, e: usize, h: Hypot) -> f64 {
        let ed = &self.edges[e];
        self.pos(ed.v1).sub(self.pos(ed.v0)).length(h) as f64
    }

    /// `Mesh.find_edge`.
    pub fn find_edge(&self, a: usize, b: usize) -> Option<usize> {
        let (small, target) = if self.vertices[a].edges.len() <= self.vertices[b].edges.len() {
            (a, b)
        } else {
            (b, a)
        };
        self.vertices[small].edges.iter().copied().find(|&e| self.other(e, small) == target)
    }

    /// `Face.vertices`.
    pub fn face_vertices(&self, f: usize) -> Vec<V3> {
        self.faces[f].loop_.iter().map(|&v| self.pos(v)).collect()
    }

    /// `Face.normal()`: Newell in f64, divided by hand.
    pub fn face_normal(&self, f: usize, h: Hypot) -> V3 {
        let lp = &self.faces[f].loop_;
        let count = lp.len();
        if count < 3 {
            return V3(0.0, 0.0, 1.0);
        }
        let (mut nx, mut ny, mut nz) = (0.0f64, 0.0f64, 0.0f64);
        for i in 0..count {
            let c = self.pos(lp[i]);
            let n = self.pos(lp[(i + 1) % count]);
            let (x0, y0, z0) = (c.0 as f64, c.1 as f64, c.2 as f64);
            let (x1, y1, z1) = (n.0 as f64, n.1 as f64, n.2 as f64);
            nx += (y0 - y1) * (z0 + z1);
            ny += (z0 - z1) * (x0 + x1);
            nz += (x0 - x1) * (y0 + y1);
        }
        let n = V3(nx as f32, ny as f32, nz as f32);
        let length = n.length(h);
        if (length as f64) < 1e-9 {
            return V3(0.0, 0.0, 1.0);
        }
        n.div(length)
    }
}

/// How the size of a fillet is read (`MEASURES`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Measure {
    Radius,
    Offset,
    Width,
    Depth,
}

impl Measure {
    /// Anything unknown reads as the radius, like the Python.
    pub fn parse(s: &str) -> Measure {
        match s {
            "offset" => Measure::Offset,
            "width" => Measure::Width,
            "depth" => Measure::Depth,
            _ => Measure::Radius,
        }
    }
}

/// How a box corner is closed (`CORNERS`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Corners {
    Round,
    Cut,
}

impl Corners {
    pub fn parse(s: &str) -> Corners {
        if s == "cut" {
            Corners::Cut
        } else {
            Corners::Round
        }
    }
}

/// The section's shape when it is not a superellipse: a preset or a drawn
/// outline of `(x, y, smooth)` control points.
#[derive(Clone, Debug, PartialEq)]
pub enum Shape {
    Preset(Preset),
    Custom(Vec<(f64, f64, bool)>),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Preset {
    Steps,
    Supports,
    Cornice,
    Crown,
}

impl Shape {
    /// `shape in bevel_profiles.PRESETS`, else None.
    pub fn parse(s: &str) -> Option<Shape> {
        match s {
            "steps" => Some(Shape::Preset(Preset::Steps)),
            "supports" => Some(Shape::Preset(Preset::Supports)),
            "cornice" => Some(Shape::Preset(Preset::Cornice)),
            "crown" => Some(Shape::Preset(Preset::Crown)),
            _ => None,
        }
    }
}

// ---- outputs --------------------------------------------------------------

/// A face rebuilt on new points: the original face (for its attrs and
/// `interior`), the new outer loop and hole loops.
#[derive(Clone, Debug)]
pub struct Rebuilt {
    pub face: usize,
    pub loop_: Vec<V3>,
    pub holes: Vec<Vec<V3>>,
}

/// `FilletPlan`, as plain data.
#[derive(Clone, Debug)]
pub struct FilletPlan {
    pub radius: f64,
    pub segments: usize,
    pub profile: f64,
    pub measure: Measure,
    pub corners: Corners,
    pub shape: Option<Shape>,
    pub removed_faces: Vec<usize>,
    pub removed_edges: Vec<usize>,
    pub rebuilt: Vec<Rebuilt>,
    pub strips: Vec<Vec<V3>>,
    pub patches: Vec<Vec<V3>>,
    pub soft: Vec<(V3, V3)>,
    pub curves: Vec<Vec<V3>>,
    /// `face_attrs["strip"]`: None when the plan sets none.
    pub strip_attrs: Option<Vec<(String, String)>>,
    /// (p, q, edge index): the flags of that old edge go to the edge p–q.
    pub flags: Vec<(V3, V3, usize)>,
}

impl FilletPlan {
    pub fn new(radius: f64, segments: usize, profile: f64, measure: Measure, corners: Corners, shape: Option<Shape>) -> FilletPlan {
        FilletPlan {
            radius,
            segments,
            profile,
            measure,
            corners,
            shape,
            removed_faces: Vec::new(),
            removed_edges: Vec::new(),
            rebuilt: Vec::new(),
            strips: Vec::new(),
            patches: Vec::new(),
            soft: Vec::new(),
            curves: Vec::new(),
            strip_attrs: None,
            flags: Vec::new(),
        }
    }

    /// `new_faces`: strips then patches.
    pub fn new_faces(&self) -> Vec<&Vec<V3>> {
        self.strips.iter().chain(self.patches.iter()).collect()
    }
}

/// Why a plan cannot be made: the `tr()` messages of fillet and vertex
/// bevel. `p` is the position the message names.
#[derive(Clone, Debug, PartialEq)]
pub enum FilletError {
    PickEdge,
    RadiusPositive,
    ProfileRange,
    DrawnProfile,
    EdgeNeedsTwoFaces,
    ZeroLengthEdge,
    UnreadableFaces,
    TooFlatOrSharp,
    TooManyFaces(V3),
    EndFaceOblique(V3),
    RadiusTooLargeForFace(V3),
    ConflictingCuts(V3),
    EdgesFoldBack(V3),
    EdgesTooSharp(V3),
    EdgesShareNoFace(V3),
    CornerNotSimple(V3),
    CornerTooSkewed(V3),
    RadiusTooLargeForEdge(V3),
    MoreThanThree(V3),
    FaceCollapses,
    BendsFace(V3),
    InternalEdgeEnd,
    CouldNotClose(V3),
    // vertex bevel
    PickCorner,
    OnlyCornerOfFace,
    CornerEdgeNoFace(V3),
    SizeTooLargeForEdge(V3),
    EdgesTooFlat(V3),
    NothingToBevel(V3),
    NotFlatEnough(V3),
}

/// `_fmt`: `(x, y, z)` with two decimals, as Python's f-string prints it.
pub fn fmt_pos(p: V3) -> String {
    format!("({:.2}, {:.2}, {:.2})", p.0 as f64, p.1 as f64, p.2 as f64)
}

impl FilletError {
    /// The message with its `{p}` filled in (English, the `tr()` source).
    pub fn message(&self) -> String {
        use FilletError::*;
        let (text, p): (&str, Option<V3>) = match self {
            PickEdge => ("Pick an edge to round.", None),
            RadiusPositive => ("The radius must be positive.", None),
            ProfileRange => ("The profile must be between 0 and 1.", None),
            DrawnProfile => ("The drawn profile must run from one face to the other.", None),
            EdgeNeedsTwoFaces => ("Only an edge between exactly two faces can be rounded.", None),
            ZeroLengthEdge => ("A zero-length edge cannot be rounded.", None),
            UnreadableFaces => ("Could not read the faces around the edge.", None),
            TooFlatOrSharp => ("The faces meet too flat or too sharp to round.", None),
            TooManyFaces(p) => ("The corner at {p} has too many faces to round.", Some(*p)),
            EndFaceOblique(p) => ("The end face at {p} is too oblique to the edge.", Some(*p)),
            RadiusTooLargeForFace(p) => ("The radius is too large for the face at {p}.", Some(*p)),
            ConflictingCuts(p) => ("Conflicting cuts at {p}.", Some(*p)),
            EdgesFoldBack(p) => ("The two edges at {p} fold back on each other.", Some(*p)),
            EdgesTooSharp(p) => ("The edges at {p} meet too sharply to round.", Some(*p)),
            EdgesShareNoFace(p) => ("The edges at {p} share no face.", Some(*p)),
            CornerNotSimple(p) => ("The corner at {p} is not a simple corner.", Some(*p)),
            CornerTooSkewed(p) => ("The corner at {p} is too skewed to round.", Some(*p)),
            RadiusTooLargeForEdge(p) => ("The radius is too large for the edge at {p}.", Some(*p)),
            MoreThanThree(p) => ("More than three rounded edges meet at {p}.", Some(*p)),
            FaceCollapses => ("A face would collapse — the radius is too large.", None),
            BendsFace(p) => ("Rounding would bend the face at {p}.", Some(*p)),
            InternalEdgeEnd => ("Internal error: an edge end was not resolved.", None),
            CouldNotClose(p) => ("Could not close the corner at {p}.", Some(*p)),
            PickCorner => ("Pick a corner to bevel.", None),
            OnlyCornerOfFace => ("Only a corner of a face can be bevelled.", None),
            CornerEdgeNoFace(p) => ("The corner at {p} has an edge with no face.", Some(*p)),
            SizeTooLargeForEdge(p) => ("The size is too large for the edge at {p}.", Some(*p)),
            EdgesTooFlat(p) => ("The edges at {p} are too flat to bevel.", Some(*p)),
            NothingToBevel(p) => ("Nothing to bevel at {p}.", Some(*p)),
            NotFlatEnough(p) => ("The corner at {p} is not flat enough to cut.", Some(*p)),
        };
        match p {
            Some(p) => text.replace("{p}", &fmt_pos(p)),
            None => text.to_string(),
        }
    }
}

// ---- small helpers ----------------------------------------------------------

#[inline]
pub(crate) fn zero() -> V3 {
    V3(0.0, 0.0, 0.0)
}
#[inline]
pub(crate) fn lenf(v: V3, h: Hypot) -> f64 {
    v.length(h) as f64
}
#[inline]
pub(crate) fn dotf(a: V3, b: V3) -> f64 {
    a.dot(b) as f64
}
/// `QVector3D * float`: the float is rounded to f32 first.
#[inline]
pub(crate) fn mulf(v: V3, s: f64) -> V3 {
    v.mul(s as f32)
}
#[inline]
pub(crate) fn divf(v: V3, s: f64) -> V3 {
    v.div(s as f32)
}
#[inline]
fn from64(x: f64, y: f64, z: f64) -> V3 {
    V3(x as f32, y as f32, z as f32)
}

/// Python's `round(x)` of a float: ties to even.
pub(crate) fn py_round(x: f64) -> i64 {
    x.round_ties_even() as i64
}

type Key = (i64, i64, i64);

pub(crate) fn same(a: V3, b: V3, h: Hypot) -> bool {
    lenf(a.sub(b), h) < SAME
}

fn key(p: V3) -> Key {
    (py_round(p.0 as f64 * 1e5), py_round(p.1 as f64 * 1e5), py_round(p.2 as f64 * 1e5))
}

/// `frozenset((a, b))` of two keys, as an ordered pair; None when equal.
fn pair_key(a: Key, b: Key) -> (Key, Key, bool) {
    if a <= b {
        (a, b, a == b)
    } else {
        (b, a, false)
    }
}

pub(crate) fn unit(v: V3, h: Hypot) -> Option<V3> {
    let ln = v.length(h);
    if (ln as f64) < 1e-12 {
        None
    } else {
        Some(v.div(ln))
    }
}

/// A dict that keeps insertion order (a replaced key keeps its place).
struct OMap<K, V> {
    idx: HashMap<K, usize>,
    items: Vec<(K, V)>,
}

impl<K: Hash + Eq + Clone, V> OMap<K, V> {
    fn new() -> Self {
        OMap { idx: HashMap::new(), items: Vec::new() }
    }
    fn get(&self, k: &K) -> Option<&V> {
        self.idx.get(k).map(|&i| &self.items[i].1)
    }
    fn get_mut(&mut self, k: &K) -> Option<&mut V> {
        match self.idx.get(k) {
            Some(&i) => Some(&mut self.items[i].1),
            None => None,
        }
    }
    fn insert(&mut self, k: K, v: V) {
        match self.idx.get(&k) {
            Some(&i) => self.items[i].1 = v,
            None => {
                self.idx.insert(k.clone(), self.items.len());
                self.items.push((k, v));
            }
        }
    }
}

// ---- bevel_profiles (TODO(M1.4 merge): use crate::bevel_profiles) -----------

/// `math.hypot(x, y)` as CPython 3.12 computes it (`vector_norm`).
pub fn py_hypot2(x: f64, y: f64) -> f64 {
    fn split(x: f64) -> (f64, f64) {
        let t = x * 134217729.0;
        let hi = t - (t - x);
        (hi, x - hi)
    }
    fn dl_mul(x: f64, y: f64) -> (f64, f64) {
        let (xh, xl) = split(x);
        let (yh, yl) = split(y);
        let p = xh * yh;
        let q = xh * yl + xl * yh;
        let z = p + q;
        let zz = p - z + q + xl * yl;
        (z, zz)
    }
    fn fast_sum(a: f64, b: f64) -> (f64, f64) {
        let x = a + b;
        (x, (a - x) + b)
    }
    let v = [x.abs(), y.abs()];
    let max = py_max2(v[0], v[1]);
    if max.is_infinite() {
        return max;
    }
    if x.is_nan() || y.is_nan() {
        return f64::NAN;
    }
    if max == 0.0 {
        return max;
    }
    let mut vec = v;
    let mut mx = max;
    let mut extra = 1.0;
    if mx < f64::MIN_POSITIVE {
        // subnormal: scale up into the normal range (CPython divides by DBL_MIN)
        vec = [vec[0] / f64::MIN_POSITIVE, vec[1] / f64::MIN_POSITIVE];
        mx /= f64::MIN_POSITIVE;
        extra = f64::MIN_POSITIVE;
    }
    // frexp: mx = m * 2^e with 0.5 <= m < 1
    let bits = mx.to_bits();
    let e = ((bits >> 52) & 0x7ff) as i32 - 1022;
    let scale = f64::from_bits(((1023 - e) as u64) << 52);
    let mut csum = 1.0f64;
    let mut frac1 = 0.0f64;
    let mut frac2 = 0.0f64;
    for &c in &vec {
        let xs = c * scale;
        let (phi, plo) = dl_mul(xs, xs);
        let (shi, slo) = fast_sum(csum, phi);
        csum = shi;
        frac1 += plo;
        frac2 += slo;
    }
    let mut hh = (csum - 1.0 + (frac1 + frac2)).sqrt();
    let (phi, plo) = dl_mul(-hh, hh);
    let (shi, slo) = fast_sum(csum, phi);
    csum = shi;
    frac1 += plo;
    frac2 += slo;
    let xx = csum - 1.0 + (frac1 + frac2);
    hh += xx / (2.0 * hh);
    extra * (hh / scale)
}

const CORNICE: [(f64, f64, bool); 13] = [
    (1.0, 0.0, false), (1.0, 0.125, false), (0.92, 0.16, true),
    (0.875, 0.25, false), (0.8, 0.25, false), (0.733, 0.433, true),
    (0.582, 0.522, true), (0.4, 0.6, true), (0.289, 0.727, true),
    (0.25, 0.925, false), (0.175, 0.925, false), (0.175, 1.0, false),
    (0.0, 1.0, false),
];
const CROWN: [(f64, f64, bool); 11] = [
    (1.0, 0.0, false), (1.0, 0.25, false), (0.75, 0.25, false),
    (0.75, 0.325, false), (0.925, 0.4, true), (0.975, 0.5, true),
    (0.94, 0.65, true), (0.85, 0.75, true), (0.75, 0.875, true),
    (0.7, 1.0, false), (0.0, 1.0, false),
];

fn profile_steps(segments: i64) -> Vec<(f64, f64)> {
    let k = std::cmp::max(1, (segments + 1) / 2);
    let mut pts = vec![(1.0, 0.0)];
    for i in 0..k {
        let x = 1.0 - (i + 1) as f64 / k as f64;
        pts.push((x, i as f64 / k as f64));
        pts.push((x, (i + 1) as f64 / k as f64));
    }
    let last = pts.len() - 1;
    pts[last] = (0.0, 1.0);
    pts
}

fn profile_supports(segments: i64) -> Vec<(f64, f64)> {
    let arc = std::cmp::max(1, segments - 2);
    let mut pts = vec![(1.0, 0.0)];
    for k in 0..=arc {
        let a = PI / 2.0 * k as f64 / arc as f64;
        pts.push((0.5 + 0.5 * a.cos(), 0.5 + 0.5 * a.sin()));
    }
    pts.push((0.0, 1.0));
    pts
}

fn catmull(p0: (f64, f64), p1: (f64, f64), p2: (f64, f64), p3: (f64, f64), t: f64) -> (f64, f64) {
    let t2 = t * t;
    let t3 = t * t * t;
    let f = |a: f64, b: f64, c: f64, d: f64| {
        0.5 * (2.0 * b + (-a + c) * t + (2.0 * a - 5.0 * b + 4.0 * c - d) * t2
            + (-a + 3.0 * b - 3.0 * c + d) * t3)
    };
    (f(p0.0, p1.0, p2.0, p3.0), f(p0.1, p1.1, p2.1, p3.1))
}

fn profile_from_controls(controls: &[(f64, f64, bool)], segments: i64) -> Vec<(f64, f64)> {
    let pts: Vec<(f64, f64)> = controls.iter().map(|&(x, y, _)| (x, y)).collect();
    let smooth: Vec<bool> = controls.iter().map(|&(_, _, s)| s).collect();
    let pieces = pts.len() - 1;
    let curved: Vec<usize> = (0..pieces).filter(|&i| smooth[i] || smooth[i + 1]).collect();
    let spare = std::cmp::max(0, segments - (pieces - curved.len()) as i64);
    let length = |i: usize| {
        let ((x0, y0), (x1, y1)) = (pts[i], pts[i + 1]);
        py_hypot2(x1 - x0, y1 - y0)
    };
    let mut total = py_sum(curved.iter().map(|&i| length(i)), PY_SUM_COMPENSATED);
    if total == 0.0 {
        total = 1.0;
    }
    let mut out = vec![pts[0]];
    for i in 0..pieces {
        if !curved.contains(&i) {
            out.push(pts[i + 1]);
            continue;
        }
        let k = std::cmp::max(1, py_round(spare as f64 * length(i) / total));
        let p0 = if i > 0 && smooth[i] { pts[i - 1] } else { pts[i] };
        let p3 = if i + 2 < pts.len() && smooth[i + 1] { pts[i + 2] } else { pts[i + 1] };
        for j in 1..=k {
            out.push(catmull(p0, pts[i], pts[i + 1], p3, j as f64 / k as f64));
        }
    }
    out
}

/// `bevel_profiles.valid`.
pub fn profile_valid(controls: &[(f64, f64, bool)]) -> bool {
    if controls.len() < 2 {
        return false;
    }
    let first = controls[0];
    let last = controls[controls.len() - 1];
    if (first.0 - 1.0).abs() > 1e-9 || first.1.abs() > 1e-9 {
        return false;
    }
    if last.0.abs() > 1e-9 || (last.1 - 1.0).abs() > 1e-9 {
        return false;
    }
    controls
        .iter()
        .all(|&(x, y, _)| [x, y].iter().all(|&c| -1e-9 <= c && c <= 1.0 + 1e-9))
}

/// `bevel_profiles.sample`.
pub fn profile_sample(shape: &Shape, segments: i64) -> Vec<(f64, f64)> {
    let segments = std::cmp::max(segments, 1);
    match shape {
        Shape::Custom(c) => profile_from_controls(c, segments),
        Shape::Preset(Preset::Steps) => profile_steps(segments),
        Shape::Preset(Preset::Supports) => profile_supports(segments),
        Shape::Preset(Preset::Cornice) => profile_from_controls(&CORNICE, segments),
        Shape::Preset(Preset::Crown) => profile_from_controls(&CROWN, segments),
    }
}

/// `bevel_profiles.mirrored`.
pub fn profile_mirrored(points: &[(f64, f64)]) -> Vec<(f64, f64)> {
    points.iter().rev().map(|&(x, y)| (y, x)).collect()
}

fn is_asymmetric(shape: &Shape) -> bool {
    match shape {
        Shape::Custom(_) => true,
        Shape::Preset(p) => matches!(p, Preset::Cornice | Preset::Crown),
    }
}

// ---- the section of one edge -------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Kind {
    Cap,
    Open,
    Chain,
    Corner,
}

struct End {
    vid: usize,
    kind: Kind,
    sec: Vec<V3>,
}

struct EdgeRound {
    edge: usize,
    f1: usize,
    f2: usize,
    d: V3,
    w1: V3,
    w2: V3,
    t: f64,
    c_off: V3,
    offsets: Vec<V3>,
    sign: f64,
    radius: f64,
    ends: Vec<End>,
}

impl EdgeRound {
    fn end(&self, vid: usize) -> Option<&End> {
        self.ends.iter().find(|e| e.vid == vid)
    }
    fn set_end(&mut self, vid: usize, kind: Kind, sec: Vec<V3>) {
        if let Some(e) = self.ends.iter_mut().find(|e| e.vid == vid) {
            e.kind = kind;
            e.sec = sec;
        } else {
            self.ends.push(End { vid, kind, sec });
        }
    }
}

/// `_plane_of`.
fn plane_of(mesh: &TopoMesh, f: usize, h: Hypot) -> (V3, V3) {
    let n = mesh.face_normal(f, h);
    let p = mesh.pos(mesh.faces[f].loop_[0]);
    (p, if (n.length_squared() as f64) > 1e-18 { n.normalized(h) } else { n })
}

/// `_inward`: unit vector in the face's plane, perpendicular to the edge
/// v0–v1, pointing into the face.
fn inward(mesh: &TopoMesh, f: usize, v0: usize, v1: usize, h: Hypot) -> Option<V3> {
    let face = &mesh.faces[f];
    let mut d: Option<V3> = None;
    let n = face.loop_.len();
    for i in 0..n {
        let (a, b) = (face.loop_[i], face.loop_[(i + 1) % n]);
        if a == v0 && b == v1 {
            d = Some(mesh.pos(v1).sub(mesh.pos(v0)));
            break;
        }
        if a == v1 && b == v0 {
            d = Some(mesh.pos(v0).sub(mesh.pos(v1)));
            break;
        }
    }
    if d.is_none() {
        for hole in &face.holes {
            let m = hole.len();
            for i in 0..m {
                let (a, b) = (hole[i], hole[(i + 1) % m]);
                if a == v0 && b == v1 {
                    d = Some(mesh.pos(v1).sub(mesh.pos(v0)));
                } else if a == v1 && b == v0 {
                    d = Some(mesh.pos(v0).sub(mesh.pos(v1)));
                } else {
                    continue;
                }
                break;
            }
            if d.is_some() {
                break;
            }
        }
    }
    let d = d?;
    let normal = mesh.face_normal(f, h);
    unit(normal.cross(d), h)
}

/// `_section_offsets`: (t, centre offset, points).
fn section_offsets(w1: V3, w2: V3, radius: f64, segments: usize, h: Hypot) -> Option<(f64, V3, Vec<V3>)> {
    let cos_a = py_max2(-1.0, py_min2(1.0, dotf(w1, w2)));
    let alpha = cos_a.acos();
    if alpha < MIN_ANGLE_DEG * DEG_TO_RAD || alpha > PI - MIN_ANGLE_DEG * DEG_TO_RAD {
        return None;
    }
    let t = radius / (alpha / 2.0).tan();
    let c_off = mulf(w1.add(w2), radius / alpha.sin());
    let t1 = mulf(w1, t);
    let t2 = mulf(w2, t);
    let a = unit(t1.sub(c_off), h);
    let rb = t2.sub(c_off);
    let a = a?;
    let b = unit(rb.sub(mulf(a, dotf(rb, a))), h)?;
    // `_unit(rb)` is evaluated even when it is None only to fail later; the
    // Python would raise on None there, which cannot happen once b exists.
    let urb = unit(rb, h)?;
    let phi = py_max2(-1.0, py_min2(1.0, dotf(a, urb))).acos();
    let mut pts = Vec::with_capacity(segments + 1);
    for k in 0..=segments {
        let ang = phi * k as f64 / segments as f64;
        pts.push(c_off.add(mulf(mulf(a, ang.cos()).add(mulf(b, ang.sin())), radius)));
    }
    Some((t, c_off, pts))
}

/// `_tangent_distance`.
fn tangent_distance(size: f64, alpha: f64, measure: Measure) -> f64 {
    let half = alpha / 2.0;
    match measure {
        Measure::Offset => size,
        Measure::Width => size / (2.0 * half.sin()),
        Measure::Depth => size / half.cos(),
        Measure::Radius => size / half.tan(),
    }
}

/// `_exponent`: the superellipse exponent of a profile.
pub fn exponent(profile: f64) -> f64 {
    4.0f64.ln() / (1.0 / profile).ln()
}

/// `_superellipse_offsets`: (centre, points).
fn superellipse_offsets(w1: V3, w2: V3, t: f64, profile: f64, segments: usize) -> (V3, Vec<V3>) {
    let t1 = mulf(w1, t);
    let t2 = mulf(w2, t);
    let centre = t1.add(t2);
    let (a, b) = (t1.sub(centre), t2.sub(centre));
    let e = 2.0 / exponent(profile);
    let mut pts = Vec::with_capacity(segments + 1);
    for k in 0..=segments {
        let (x, y) = if k == 0 {
            (1.0, 0.0)
        } else if k == segments {
            (0.0, 1.0)
        } else {
            let ang = PI / 2.0 * k as f64 / segments as f64;
            (ang.cos().powf(e), ang.sin().powf(e))
        };
        pts.push(centre.add(mulf(a, x)).add(mulf(b, y)));
    }
    (centre, pts)
}

/// `_outline_offsets`.
fn outline_offsets(w1: V3, w2: V3, t: f64, points: &[(f64, f64)]) -> (V3, Vec<V3>) {
    let t1 = mulf(w1, t);
    let t2 = mulf(w2, t);
    let centre = t1.add(t2);
    let (a, b) = (t1.sub(centre), t2.sub(centre));
    (centre, points.iter().map(|&(x, y)| centre.add(mulf(a, x)).add(mulf(b, y))).collect())
}

/// `_cut`: the strip's generators through `P + offset` cut by a plane.
fn cut(p: V3, offsets: &[V3], d: V3, plane: (V3, V3)) -> Option<Vec<V3>> {
    let (q, m) = plane;
    let denom = dotf(d, m);
    if denom.abs() < 0.05 {
        return None;
    }
    let mut out = Vec::with_capacity(offsets.len());
    for &off in offsets {
        let base = p.add(off);
        let s = dotf(q.sub(base), m) / denom;
        out.push(base.add(mulf(d, s)));
    }
    Some(out)
}

/// `_line_intersection_2d`.
fn line_intersection_2d(p: V3, u: V3, q: V3, v: V3, normal: V3) -> Option<V3> {
    let w = u.cross(v);
    let den = dotf(w, normal);
    if den.abs() < 1e-12 {
        return None;
    }
    let s = dotf(q.sub(p).cross(v), normal) / den;
    Some(p.add(mulf(u, s)))
}

fn point_inside_2d(pt: (f64, f64), poly: &[(f64, f64)]) -> bool {
    let (x, y) = pt;
    let mut inside = false;
    let n = poly.len();
    let mut j = n.wrapping_sub(1);
    for i in 0..n {
        let (xi, yi) = poly[i];
        let (xj, yj) = poly[j];
        if ((yi > y) != (yj > y)) && (x < (xj - xi) * (y - yi) / (yj - yi) + xi) {
            inside = !inside;
        }
        j = i;
    }
    inside
}

/// `core.topology._point_in_face_solid`.
fn point_in_face_solid(mesh: &TopoMesh, f: usize, point: V3, h: Hypot) -> bool {
    let n = mesh.face_normal(f, h);
    let (u, w) = plane_axes(n, h);
    let origin = mesh.pos(mesh.faces[f].loop_[0]);
    let proj = |p: V3| {
        let rel = p.sub(origin);
        (dotf(rel, u), dotf(rel, w))
    };
    let pt = proj(point);
    let outer: Vec<(f64, f64)> = mesh.face_vertices(f).into_iter().map(proj).collect();
    if !point_inside_2d(pt, &outer) {
        return false;
    }
    !mesh.faces[f].holes.iter().any(|hl| {
        let poly: Vec<(f64, f64)> = hl.iter().map(|&v| proj(mesh.pos(v))).collect();
        point_inside_2d(pt, &poly)
    })
}

// ---- plan_fillet ---------------------------------------------------------------

#[derive(PartialEq, Eq, Clone, Copy)]
enum Repl {
    Ok,
    Conflict,
    Outside,
}

#[allow(clippy::too_many_arguments)]
fn set_replacement(
    mesh: &TopoMesh,
    replace: &mut HashMap<(usize, usize), Vec<V3>>,
    face: usize,
    v: usize,
    pts: &[V3],
    r: Option<&EdgeRound>,
    h: Hypot,
) -> Repl {
    if let Some(r) = r {
        if pts.len() == 1 && (face == r.f1 || face == r.f2) {
            let w = if face == r.f1 { r.w1 } else { r.w2 };
            let ed = &mesh.edges[r.edge];
            let mid = mesh.pos(ed.v0).add(mesh.pos(ed.v1)).mul(0.5);
            let p = pts[0];
            let q = p.add(mulf(mid.sub(p).sub(mulf(w, r.t)), 1e-3));
            if !point_in_face_solid(mesh, face, q, h) {
                return Repl::Outside;
            }
        }
    }
    let k = (face, v);
    let cur = match replace.get(&k) {
        None => {
            replace.insert(k, pts.to_vec());
            return Repl::Ok;
        }
        Some(c) => c.clone(),
    };
    if pts.len() == 1 && cur.iter().any(|&p| same(p, pts[0], h)) {
        return Repl::Ok;
    }
    if cur.len() == 1 && pts.iter().any(|&p| same(p, cur[0], h)) {
        replace.insert(k, pts.to_vec());
        return Repl::Ok;
    }
    if cur.len() == pts.len() && cur.iter().zip(pts.iter()).all(|(&a, &b)| same(a, b, h)) {
        return Repl::Ok;
    }
    Repl::Conflict
}

/// `_shared_attrs`: the new faces take the material when both neighbours
/// share it.
fn shared_attrs(mesh: &TopoMesh, f1: usize, f2: usize) -> Vec<(String, String)> {
    let get = |f: usize, k: &str| mesh.faces[f].paint.iter().find(|(kk, _)| kk == k).map(|(_, v)| v.clone());
    if STRIP_KEYS.iter().all(|k| get(f1, k) == get(f2, k)) {
        STRIP_KEYS
            .iter()
            .filter_map(|k| get(f1, k).map(|v| (k.to_string(), v)))
            .collect()
    } else {
        Vec::new()
    }
}

/// `_newell` of fillet.py: f32 accumulation of f64 products.
fn newell(pts: &[V3]) -> V3 {
    let mut n = zero();
    let m = pts.len();
    for i in 0..m {
        let (a, b) = (pts[i], pts[(i + 1) % m]);
        let (ay, az, ax) = (a.1 as f64, a.2 as f64, a.0 as f64);
        let (by, bz, bx) = (b.1 as f64, b.2 as f64, b.0 as f64);
        n = n.add(from64((ay - by) * (az + bz), (az - bz) * (ax + bx), (ax - bx) * (ay + by)));
    }
    n
}

fn mean_point(loop_: &[V3]) -> V3 {
    let mut c = zero();
    for &p in loop_ {
        c = c.add(p);
    }
    c.div(loop_.len() as f32)
}

/// Whether `_oriented` reverses the loop.
fn oriented_flips(loop_: &[V3], centre: V3, sign: f64) -> bool {
    let c = mean_point(loop_);
    let n = newell(loop_);
    dotf(n, c.sub(centre)) * sign < 0.0
}

fn oriented(loop_: &[V3], centre: V3, sign: f64) -> Vec<V3> {
    let mut v = loop_.to_vec();
    if oriented_flips(loop_, centre, sign) {
        v.reverse();
    }
    v
}

/// `_facing`: wound so its normal points toward `target` (times `sign`).
pub(crate) fn facing(loop_: &[V3], target: V3, sign: f64) -> Vec<V3> {
    let c = mean_point(loop_);
    let mut v = loop_.to_vec();
    if dotf(newell(loop_), target.sub(c)) * sign < 0.0 {
        v.reverse();
    }
    v
}

fn lerp(p: V3, q: V3, s: f64) -> V3 {
    p.add(mulf(q.sub(p), s))
}

/// `plan_fillet`: plan the rounding of `edges` to the size `radius` read as
/// `measure`, the section shaped by `profile` (or the drawn `shape`) and box
/// corners closed as `corners` says.
#[allow(clippy::too_many_arguments)]
pub fn plan_fillet(
    mesh: &TopoMesh,
    edges: &[usize],
    radius: f64,
    segments: i64,
    profile: f64,
    measure: Measure,
    corners: Corners,
    shape: Option<&Shape>,
    h: Hypot,
) -> Result<FilletPlan, FilletError> {
    // dict.fromkeys(edges)
    let mut uniq: Vec<usize> = Vec::new();
    for &e in edges {
        if !uniq.contains(&e) {
            uniq.push(e);
        }
    }
    let edges = uniq;
    if edges.is_empty() {
        return Err(FilletError::PickEdge);
    }
    if radius <= 0.0 {
        return Err(FilletError::RadiusPositive);
    }
    if !(0.0 < profile && profile < 1.0) {
        return Err(FilletError::ProfileRange);
    }
    if let Some(Shape::Custom(c)) = shape {
        if !profile_valid(c) {
            return Err(FilletError::DrawnProfile);
        }
    }
    let outline: Option<Vec<(f64, f64)>> = shape.map(|s| profile_sample(s, segments));
    let circular = (profile - CIRCULAR).abs() < 1e-9;
    let segments = std::cmp::max(segments, 1) as usize;
    let mut rounds: Vec<EdgeRound> = Vec::new();
    for &e in &edges {
        let faces = &mesh.edges[e].faces;
        if faces.len() != 2 {
            return Err(FilletError::EdgeNeedsTwoFaces);
        }
        let (f1, f2) = (faces[0], faces[1]);
        let (ev0, ev1) = (mesh.edges[e].v0, mesh.edges[e].v1);
        let d = match unit(mesh.pos(ev1).sub(mesh.pos(ev0)), h) {
            Some(d) => d,
            None => return Err(FilletError::ZeroLengthEdge),
        };
        let w1 = inward(mesh, f1, ev0, ev1, h);
        let w2 = inward(mesh, f2, ev0, ev1, h);
        let (w1, w2) = match (w1, w2) {
            (Some(a), Some(b)) => (a, b),
            _ => return Err(FilletError::UnreadableFaces),
        };
        let alpha = py_max2(-1.0, py_min2(1.0, dotf(w1, w2))).acos();
        if alpha < MIN_ANGLE_DEG * DEG_TO_RAD || alpha > PI - MIN_ANGLE_DEG * DEG_TO_RAD {
            return Err(FilletError::TooFlatOrSharp);
        }
        let t = tangent_distance(radius, alpha, measure);
        let arc_radius = t * (alpha / 2.0).tan();
        let sec = match section_offsets(w1, w2, arc_radius, segments, h) {
            Some(s) => s,
            None => return Err(FilletError::TooFlatOrSharp),
        };
        let (t, mut c_off, mut offsets) = sec;
        let n1 = mesh.face_normal(f1, h);
        let sign = if dotf(n1, mulf(w1, t).sub(c_off)) >= 0.0 { 1.0 } else { -1.0 };
        if let (Some(sh), Some(ol)) = (shape, outline.as_ref()) {
            let mut pts = ol.clone();
            if is_asymmetric(sh)
                && (mesh.face_normal(f2, h).normalized(h).2 as f64)
                    > (mesh.face_normal(f1, h).normalized(h).2 as f64) + 1e-6
            {
                pts = profile_mirrored(ol);
            }
            let (c, o) = outline_offsets(w1, w2, t, &pts);
            c_off = c;
            offsets = o;
        } else if !circular {
            let (c, o) = superellipse_offsets(w1, w2, t, profile, segments);
            c_off = c;
            offsets = o;
        }
        rounds.push(EdgeRound { edge: e, f1, f2, d, w1, w2, t, c_off, offsets, sign, radius: arc_radius, ends: Vec::new() });
    }

    // Which rounded edges meet at each vertex.
    let mut at_vertex: Vec<(usize, Vec<usize>)> = Vec::new();
    let mut vpos: HashMap<usize, usize> = HashMap::new();
    for (ri, r) in rounds.iter().enumerate() {
        for v in [mesh.edges[r.edge].v0, mesh.edges[r.edge].v1] {
            let i = *vpos.entry(v).or_insert_with(|| {
                at_vertex.push((v, Vec::new()));
                at_vertex.len() - 1
            });
            at_vertex[i].1.push(ri);
        }
    }

    let mut replace: HashMap<(usize, usize), Vec<V3>> = HashMap::new();
    let mut plan = FilletPlan::new(radius, segments, profile, measure, corners, shape.cloned());
    let mut patch_corners: Vec<(Vec<usize>, Vec<Vec<V3>>, usize)> = Vec::new();

    for (vid, group) in &at_vertex {
        let vid = *vid;
        let vp = mesh.pos(vid);
        let faces_here = mesh.vertex_faces(vid);
        if group.len() == 1 {
            let ri = group[0];
            let (f1, f2, d) = (rounds[ri].f1, rounds[ri].f2, rounds[ri].d);
            let caps: Vec<usize> = faces_here.iter().copied().filter(|&g| g != f1 && g != f2).collect();
            if caps.len() > 1 {
                return Err(FilletError::TooManyFaces(vp));
            }
            let (plane, kind) = if let Some(&c) = caps.first() {
                (plane_of(mesh, c, h), Kind::Cap)
            } else {
                ((vp, d), Kind::Open)
            };
            let sec = match cut(vp, &rounds[ri].offsets, d, plane) {
                Some(s) => s,
                None => return Err(FilletError::EndFaceOblique(vp)),
            };
            rounds[ri].set_end(vid, kind, sec.clone());
            let r = &rounds[ri];
            let mut outcome = vec![
                set_replacement(mesh, &mut replace, f1, vid, &sec[..1], Some(r), h),
                set_replacement(mesh, &mut replace, f2, vid, &sec[sec.len() - 1..], Some(r), h),
            ];
            if let Some(&c) = caps.first() {
                outcome.push(set_replacement(mesh, &mut replace, c, vid, &sec, Some(r), h));
            }
            if outcome.contains(&Repl::Outside) {
                return Err(FilletError::RadiusTooLargeForFace(vp));
            }
            if !outcome.iter().all(|&o| o == Repl::Ok) {
                return Err(FilletError::ConflictingCuts(vp));
            }
        } else if group.len() == 2 {
            let (ra_i, rb_i) = (group[0], group[1]);
            let (a_f1, a_f2, b_f1, b_f2) = (rounds[ra_i].f1, rounds[ra_i].f2, rounds[rb_i].f1, rounds[rb_i].f2);
            if faces_here.iter().any(|g| ![a_f1, a_f2, b_f1, b_f2].contains(g)) {
                return Err(FilletError::TooManyFaces(vp));
            }
            let a_other = mesh.pos(mesh.other(rounds[ra_i].edge, vid));
            let b_other = mesh.pos(mesh.other(rounds[rb_i].edge, vid));
            let u = unit(a_other.sub(vp), h);
            let u2 = unit(b_other.sub(vp), h);
            // `_unit(u - u2)` raises in Python when either is None; neither
            // can be, the edges having length.
            let (u, u2) = match (u, u2) {
                (Some(a), Some(b)) => (a, b),
                _ => return Err(FilletError::EdgesFoldBack(vp)),
            };
            let m = match unit(u.sub(u2), h) {
                Some(m) => m,
                None => return Err(FilletError::EdgesFoldBack(vp)),
            };
            let plane = (vp, m);
            let sec = match cut(vp, &rounds[ra_i].offsets, rounds[ra_i].d, plane) {
                Some(s) => s,
                None => return Err(FilletError::EdgesTooSharp(vp)),
            };
            rounds[ra_i].set_end(vid, Kind::Chain, sec.clone());
            let sec_b: Vec<V3> = if b_f1 == a_f1 || b_f2 == a_f2 {
                sec.clone()
            } else if b_f1 == a_f2 || b_f2 == a_f1 {
                sec.iter().rev().copied().collect()
            } else {
                return Err(FilletError::EdgesShareNoFace(vp));
            };
            rounds[rb_i].set_end(vid, Kind::Chain, sec_b.clone());
            let (ra, rb) = (&rounds[ra_i], &rounds[rb_i]);
            let outcome = [
                set_replacement(mesh, &mut replace, ra.f1, vid, &sec[..1], Some(ra), h),
                set_replacement(mesh, &mut replace, ra.f2, vid, &sec[sec.len() - 1..], Some(ra), h),
                set_replacement(mesh, &mut replace, rb.f1, vid, &sec_b[..1], Some(rb), h),
                set_replacement(mesh, &mut replace, rb.f2, vid, &sec_b[sec_b.len() - 1..], Some(rb), h),
            ];
            if outcome.contains(&Repl::Outside) {
                return Err(FilletError::RadiusTooLargeForFace(vp));
            }
            if !outcome.iter().all(|&o| o == Repl::Ok) {
                return Err(FilletError::ConflictingCuts(vp));
            }
        } else if group.len() == 3 {
            let mut allowed: Vec<usize> = Vec::new();
            for &ri in group {
                allowed.push(rounds[ri].f1);
                allowed.push(rounds[ri].f2);
            }
            if faces_here.iter().any(|g| !allowed.contains(g)) {
                return Err(FilletError::TooManyFaces(vp));
            }
            let mut sections: Vec<Vec<V3>> = Vec::new();
            for &ri in group {
                let mut cut_pts: Vec<V3> = Vec::new();
                for (f, w) in [(rounds[ri].f1, rounds[ri].w1), (rounds[ri].f2, rounds[ri].w2)] {
                    let other: Vec<usize> = group
                        .iter()
                        .copied()
                        .filter(|&q| q != ri && (rounds[q].f1 == f || rounds[q].f2 == f))
                        .collect();
                    if other.len() != 1 {
                        return Err(FilletError::CornerNotSimple(vp));
                    }
                    let q = &rounds[other[0]];
                    let wq = if q.f1 == f { q.w1 } else { q.w2 };
                    let normal = mesh.face_normal(f, h).normalized(h);
                    let r = &rounds[ri];
                    let hit = line_intersection_2d(vp.add(mulf(w, r.t)), r.d, vp.add(mulf(wq, q.t)), q.d, normal);
                    match hit {
                        Some(p) => cut_pts.push(p),
                        None => return Err(FilletError::CornerNotSimple(vp)),
                    }
                }
                let r = &rounds[ri];
                let s1 = dotf(cut_pts[0].sub(vp), r.d);
                let s2 = dotf(cut_pts[1].sub(vp), r.d);
                if (s1 - s2).abs() > 1e-3 * py_max2(1.0, r.t) {
                    return Err(FilletError::CornerTooSkewed(vp));
                }
                if s1.abs() > mesh.edge_length(r.edge, h) * 0.5 + 1e-9 {
                    return Err(FilletError::RadiusTooLargeForEdge(vp));
                }
                let p = vp.add(mulf(r.d, s1));
                let sec: Vec<V3> = r.offsets.iter().map(|&off| p.add(off)).collect();
                rounds[ri].set_end(vid, Kind::Corner, sec.clone());
                sections.push(sec.clone());
                let r = &rounds[ri];
                let outcome = [
                    set_replacement(mesh, &mut replace, r.f1, vid, &sec[..1], Some(r), h),
                    set_replacement(mesh, &mut replace, r.f2, vid, &sec[sec.len() - 1..], Some(r), h),
                ];
                if outcome.contains(&Repl::Outside) {
                    return Err(FilletError::RadiusTooLargeForFace(vp));
                }
                if !outcome.iter().all(|&o| o == Repl::Ok) {
                    return Err(FilletError::ConflictingCuts(vp));
                }
            }
            patch_corners.push((group.clone(), sections, vid));
        } else {
            return Err(FilletError::MoreThanThree(vp));
        }
    }

    // Rebuild every face that loses a vertex.
    let mut touched: Vec<usize> = Vec::new();
    for (vid, _) in &at_vertex {
        for f in mesh.vertex_faces(*vid) {
            if !touched.contains(&f) {
                touched.push(f);
            }
        }
    }
    let replaced_vertices: Vec<usize> = at_vertex.iter().map(|(v, _)| *v).collect();

    let rebuilt_loop = |face: usize, lp: &[usize]| -> Option<Vec<V3>> {
        let mut out: Vec<V3> = Vec::new();
        let n = lp.len();
        for (i, &x) in lp.iter().enumerate() {
            if !replaced_vertices.contains(&x) {
                out.push(mesh.pos(x));
                continue;
            }
            let mut seq = replace.get(&(face, x))?.clone();
            if seq.len() > 1 {
                let prev = mesh.pos(lp[(i + n - 1) % n]);
                let nxt = mesh.pos(lp[(i + 1) % n]);
                let straight = lenf(prev.sub(seq[0]), h) + lenf(nxt.sub(seq[seq.len() - 1]), h);
                let flipped = lenf(prev.sub(seq[seq.len() - 1]), h) + lenf(nxt.sub(seq[0]), h);
                if flipped < straight {
                    seq.reverse();
                }
            }
            out.extend(seq);
        }
        let mut clean: Vec<V3> = Vec::new();
        for p in out {
            if clean.is_empty() || lenf(p.sub(clean[clean.len() - 1]), h) > 1e-7 {
                clean.push(p);
            }
        }
        if clean.len() > 1 && lenf(clean[0].sub(clean[clean.len() - 1]), h) <= 1e-7 {
            clean.pop();
        }
        Some(clean)
    };

    for &f in &touched {
        let (p0, normal) = plane_of(mesh, f, h);
        let lp = match rebuilt_loop(f, &mesh.faces[f].loop_) {
            Some(l) if l.len() >= 3 => l,
            _ => return Err(FilletError::FaceCollapses),
        };
        let mut holes = Vec::new();
        for hole in &mesh.faces[f].holes {
            match rebuilt_loop(f, hole) {
                Some(hl) if hl.len() >= 3 => holes.push(hl),
                _ => return Err(FilletError::FaceCollapses),
            }
        }
        let mut span = lp.iter().fold(None, |m: Option<f64>, &p| {
            let l = lenf(p.sub(p0), h);
            Some(match m {
                None => l,
                Some(m) => py_max2(m, l),
            })
        }).unwrap_or(0.0);
        if span == 0.0 {
            span = 1.0;
        }
        for &p in &lp {
            if dotf(p.sub(p0), normal).abs() > PLANE_TOL * py_max2(1.0, span) {
                return Err(FilletError::BendsFace(p));
            }
        }
        plan.removed_faces.push(f);
        plan.rebuilt.push(Rebuilt { face: f, loop_: lp, holes });
    }

    // Strips.
    for ri in 0..rounds.len() {
        let r = &rounds[ri];
        let (ev0, ev1) = (mesh.edges[r.edge].v0, mesh.edges[r.edge].v1);
        let (a, b) = match (r.end(ev0), r.end(ev1)) {
            (Some(a), Some(b)) => (&a.sec, &b.sec),
            _ => return Err(FilletError::InternalEdgeEnd),
        };
        let along = dotf(b[0].sub(a[0]), r.d);
        if along <= 1e-6 {
            return Err(FilletError::RadiusTooLargeForEdge(mesh.pos(ev0)));
        }
        let n = a.len() - 1;
        let c_mid = a[0].add(b[0]).mul(0.5).sub(mulf(r.w1, r.t)).add(r.c_off);
        let chord = [a[0], a[n], b[n], b[0]];
        let flip = oriented_flips(&chord, c_mid, r.sign);
        for k in 0..n {
            let quad = vec![a[k], a[k + 1], b[k + 1], b[k]];
            plan.strips.push(if flip { quad.into_iter().rev().collect() } else { quad });
        }
        if shape.is_none() {
            for k in 1..n {
                plan.soft.push((a[k], b[k]));
            }
            plan.soft.push((a[0], b[0]));
            plan.soft.push((a[n], b[n]));
        }
        for e in &r.ends {
            let sec = &e.sec;
            if shape.is_some() {
                if matches!(e.kind, Kind::Cap | Kind::Open) {
                    plan.curves.push(sec.clone());
                }
            } else if e.kind == Kind::Chain || (e.kind == Kind::Corner && corners == Corners::Round) {
                for k in 0..n {
                    plan.soft.push((sec[k], sec[k + 1]));
                }
            } else {
                plan.curves.push(sec.clone());
            }
        }
        if plan.strip_attrs.is_none() {
            plan.strip_attrs = Some(shared_attrs(mesh, r.f1, r.f2));
        }
        plan.removed_edges.push(r.edge);
    }

    // Corner patches.
    for (group, sections, vid) in &patch_corners {
        let grp: Vec<&EdgeRound> = group.iter().map(|&i| &rounds[i]).collect();
        let vp = mesh.pos(*vid);
        if corners == Corners::Cut {
            let faces = corner_cut(mesh, &grp, sections, *vid, h);
            if let Some(f) = faces {
                if shape.is_none() || closes(&f, sections, None) {
                    plan.patches.extend(f);
                    continue;
                }
            }
            if shape.is_none() {
                return Err(FilletError::CouldNotClose(vp));
            }
        }
        if let (Some(sh), Some(ol)) = (shape, outline.as_ref()) {
            let mut faces = if matches!(sh, Shape::Preset(Preset::Steps)) {
                corner_steps(&grp, sections, ol, h)
            } else {
                corner_product(&grp, sections, ol, &plan.strips, h)
            };
            if let Some(f) = &faces {
                if !closes(f, sections, None) {
                    let (split, splits) = split_on_sides(f, sections, h);
                    if closes(&split, sections, Some(&splits)) {
                        faces = Some(split);
                        split_strips(&mut plan, &splits);
                    } else {
                        faces = None;
                    }
                }
            }
            if faces.is_none() {
                let (f, _soft) = corner_patch(&grp, sections, 0.0, segments, None, true, h);
                faces = f;
                if let Some(f) = faces.take() {
                    faces = Some(wound_like(f, &plan.strips));
                }
                if let Some(f) = &faces {
                    if !closes(f, sections, None) {
                        faces = None;
                    }
                }
            }
            if faces.is_none() {
                faces = corner_cut(mesh, &grp, sections, *vid, h);
            }
            match faces {
                Some(f) => plan.patches.extend(f),
                None => return Err(FilletError::CouldNotClose(vp)),
            }
            continue;
        }
        let corner_radius =
            py_sum(grp.iter().map(|r| r.radius), PY_SUM_COMPENSATED) / grp.len() as f64;
        let (tris, soft) = corner_patch(
            &grp,
            sections,
            corner_radius,
            segments,
            if circular { None } else { Some(profile) },
            false,
            h,
        );
        match tris {
            None => return Err(FilletError::CouldNotClose(sections[0][0])),
            Some(t) => {
                plan.patches.extend(t);
                plan.soft.extend(soft.unwrap_or_default());
            }
        }
    }

    if shape.is_some() {
        plan.soft = smooth_seams(&plan, h);
    }

    // Old edges that lose an endpoint go too; a free edge stays.
    let mut seen: Vec<usize> = plan.removed_edges.clone();
    for (vid, _) in &at_vertex {
        for &e in &mesh.vertices[*vid].edges {
            if seen.contains(&e) || mesh.edges[e].faces.is_empty() {
                continue;
            }
            seen.push(e);
            plan.removed_edges.push(e);
        }
    }
    Ok(plan)
}

/// `_smooth_seams`: the edges of a drawn outline's faces to soften.
fn smooth_seams(plan: &FilletPlan, h: Hypot) -> Vec<(V3, V3)> {
    let mut normals: OMap<(Key, Key), Vec<(V3, V3, V3)>> = OMap::new();
    let mut loops: Vec<&Vec<V3>> = plan.rebuilt.iter().map(|r| &r.loop_).collect();
    loops.extend(plan.strips.iter());
    loops.extend(plan.patches.iter());
    for lp in loops {
        let n = match unit(newell(lp), h) {
            Some(n) => n,
            None => continue,
        };
        let m = lp.len();
        for i in 0..m {
            let (a, b) = (lp[i], lp[(i + 1) % m]);
            let (k0, k1, _) = pair_key(key(a), key(b));
            let k = (k0, k1);
            match normals.get_mut(&k) {
                Some(v) => v.push((n, a, b)),
                None => normals.insert(k, vec![(n, a, b)]),
            }
        }
    }
    let cos_max = (SMOOTH_DEG * DEG_TO_RAD).cos();
    let mut soft = Vec::new();
    for (_k, entries) in &normals.items {
        if entries.len() != 2 {
            continue;
        }
        let (n1, p, q) = entries[0];
        let (n2, _p, _q) = entries[1];
        if dotf(n1, n2) > cos_max {
            soft.push((p, q));
        }
    }
    soft
}

fn meet(sa: &[V3], sb: &[V3], h: Hypot) -> Option<V3> {
    for i in [0, sa.len() - 1] {
        for j in [0, sb.len() - 1] {
            if same(sa[i], sb[j], h) {
                return Some(sa[i]);
            }
        }
    }
    None
}

/// The corner's centre: the mean of the sections' own centres.
fn corner_centre(group: &[&EdgeRound], sections: &[Vec<V3>]) -> V3 {
    let mut centre = zero();
    for (r, sec) in group.iter().zip(sections.iter()) {
        centre = centre.add(sec[0].sub(mulf(r.w1, r.t)).add(r.c_off));
    }
    centre.div(3.0)
}

/// `_corner_steps`: a box corner where three staircases meet.
fn corner_steps(group: &[&EdgeRound], sections: &[Vec<V3>], outline: &[(f64, f64)], h: Hypot) -> Option<Vec<Vec<V3>>> {
    let k = (outline.len() as i64 - 1).div_euclid(2);
    if k < 1 {
        return None;
    }
    let (s1, s2, s3) = (&sections[0], &sections[1], &sections[2]);
    let c_x = meet(s2, s3, h)?;
    let c_y = meet(s3, s1, h)?;
    let c_z = meet(s1, s2, h)?;
    let centre = corner_centre(group, sections);
    let (a_v, b_v, c_v) = (c_x.sub(centre), c_y.sub(centre), c_z.sub(centre));
    let kf = k as f64;
    let at = |a: f64, b: f64, c: f64| centre.add(mulf(a_v, a / kf)).add(mulf(b_v, b / kf)).add(mulf(c_v, c / kf));
    // Some(true/false), None past a section.
    let kept = |a: i64, b: i64, c: i64| -> Option<bool> {
        if a.min(b).min(c) < 0 {
            return None;
        }
        if a.max(b).max(c) >= k {
            return Some(false);
        }
        Some(a + b < k - 1 && a + c < k - 1 && b + c < k - 1)
    };
    let mut faces = Vec::new();
    for a in 0..k {
        for b in 0..k {
            for c in 0..k {
                if kept(a, b, c) != Some(true) {
                    continue;
                }
                for axis in 0..3usize {
                    let mut n = [a, b, c];
                    n[axis] += 1;
                    if kept(n[0], n[1], n[2]) != Some(false) {
                        continue;
                    }
                    let lo = n;
                    let others: Vec<usize> = (0..3).filter(|&i| i != axis).collect();
                    let (u, w) = (others[0], others[1]);
                    let mut quad = Vec::new();
                    for (du, dw) in [(0, 0), (1, 0), (1, 1), (0, 1)] {
                        let mut q = lo;
                        q[u] += du;
                        q[w] += dw;
                        quad.push(at(q[0] as f64, q[1] as f64, q[2] as f64));
                    }
                    let mid = at(n[0] as f64 + 0.5, n[1] as f64 + 0.5, n[2] as f64 + 0.5);
                    faces.push(facing(&quad, mid, group[0].sign));
                }
            }
        }
    }
    Some(faces)
}

fn matches_pts(sec: &[V3], pts: &[V3], h: Hypot) -> bool {
    if sec.len() != pts.len() {
        return false;
    }
    sec.iter().zip(pts.iter()).all(|(&a, &b)| same(a, b, h))
        || sec.iter().zip(pts.iter().rev()).all(|(&a, &b)| same(a, b, h))
}

/// `_corner_product`: the spherical product of the outline with itself.
fn corner_product(group: &[&EdgeRound], sections: &[Vec<V3>], outline: &[(f64, f64)], strips: &[Vec<V3>], h: Hypot) -> Option<Vec<Vec<V3>>> {
    let (s1, s2, s3) = (&sections[0], &sections[1], &sections[2]);
    let corner_pts = [meet(s2, s3, h), meet(s3, s1, h), meet(s1, s2, h)];
    let corner_pts: Vec<V3> = corner_pts.iter().copied().collect::<Option<Vec<V3>>>()?;
    let centre = corner_centre(group, sections);
    let mirrored = profile_mirrored(outline);
    for read in [outline, &mirrored[..]] {
        let m = read.len() - 1;
        for pole in 0..3usize {
            let c_z = corner_pts[pole];
            let sides: Vec<usize> = (0..3).filter(|&i| i != pole).collect();
            for swap in [false, true] {
                let (ix, iy) = if swap { (sides[1], sides[0]) } else { (sides[0], sides[1]) };
                let (c_x, c_y) = (corner_pts[ix], corner_pts[iy]);
                let (a_v, b_v, c_v) = (c_x.sub(centre), c_y.sub(centre), c_z.sub(centre));
                let at = |u: usize, v: usize| {
                    let (xu, yu) = read[u];
                    let (xv, yv) = read[v];
                    centre.add(mulf(a_v, xu * xv)).add(mulf(b_v, xu * yv)).add(mulf(c_v, yu))
                };
                let row0: Vec<V3> = (0..=m).map(|v| at(0, v)).collect();
                let col0: Vec<V3> = (0..=m).map(|u| at(u, 0)).collect();
                let colm: Vec<V3> = (0..=m).map(|u| at(u, m)).collect();
                if !(matches_pts(&sections[pole], &row0, h)
                    && matches_pts(&sections[iy], &col0, h)
                    && matches_pts(&sections[ix], &colm, h))
                {
                    continue;
                }
                let grid: Vec<Vec<V3>> = (0..=m).map(|u| (0..=m).map(|v| at(u, v)).collect()).collect();
                let mut faces = Vec::new();
                for u in 0..m {
                    for v in 0..m {
                        let quad = [grid[u][v], grid[u + 1][v], grid[u + 1][v + 1], grid[u][v + 1]];
                        faces.extend(cells(&quad, h));
                    }
                }
                return Some(wound_like(faces, strips));
            }
        }
    }
    None
}

/// `_split_on_sides`: (faces, splits in insertion order).
type Splits = Vec<((Key, Key), (V3, V3, Vec<V3>))>;

fn split_on_sides(faces: &[Vec<V3>], sections: &[Vec<V3>], h: Hypot) -> (Vec<Vec<V3>>, Splits) {
    let mut seen: HashMap<Key, ()> = HashMap::new();
    let mut pts: Vec<V3> = Vec::new();
    for f in faces {
        for &q in f {
            if seen.insert(key(q), ()).is_none() {
                pts.push(q);
            }
        }
    }
    let inner = |a: V3, b: V3| -> Vec<V3> {
        let d = b.sub(a);
        let ln2 = d.length_squared() as f64;
        if ln2 < 1e-18 {
            return Vec::new();
        }
        let mut hits: Vec<(f64, V3)> = Vec::new();
        for &q in &pts {
            let t_ = dotf(q.sub(a), d) / ln2;
            if 1e-6 < t_ && t_ < 1.0 - 1e-6 && lenf(a.add(mulf(d, t_)).sub(q), h) < 1e-7 {
                hits.push((t_, q));
            }
        }
        hits.sort_by(|x, y| x.0.partial_cmp(&y.0).unwrap_or(std::cmp::Ordering::Equal));
        hits.into_iter().map(|(_, q)| q).collect()
    };
    let mut out = Vec::new();
    for f in faces {
        let mut g = Vec::new();
        let m = f.len();
        for i in 0..m {
            g.push(f[i]);
            g.extend(inner(f[i], f[(i + 1) % m]));
        }
        out.push(g);
    }
    let mut splits: OMap<(Key, Key), (V3, V3, Vec<V3>)> = OMap::new();
    for sec in sections {
        for i in 0..sec.len() - 1 {
            let mids = inner(sec[i], sec[i + 1]);
            if !mids.is_empty() {
                splits.insert((key(sec[i]), key(sec[i + 1])), (sec[i], sec[i + 1], mids));
            }
        }
    }
    (out, splits.items)
}

/// `_split_strips`.
fn split_strips(plan: &mut FilletPlan, splits: &Splits) {
    if splits.is_empty() {
        return;
    }
    let mut table: OMap<(Key, Key), Vec<V3>> = OMap::new();
    for ((a_k, b_k), (_a, _b, mids)) in splits {
        table.insert((*a_k, *b_k), mids.clone());
        table.insert((*b_k, *a_k), mids.iter().rev().copied().collect());
    }
    let mut new = Vec::new();
    for lp in &plan.strips {
        let mut g = Vec::new();
        let m = lp.len();
        for i in 0..m {
            g.push(lp[i]);
            if let Some(mids) = table.get(&(key(lp[i]), key(lp[(i + 1) % m]))) {
                g.extend(mids.iter().copied());
            }
        }
        new.push(g);
    }
    plan.strips = new;
}

/// `_closes`: whether a corner's faces close it exactly.
fn closes(faces: &[Vec<V3>], sections: &[Vec<V3>], splits: Option<&Splits>) -> bool {
    let mut count: HashMap<(Key, Key), i64> = HashMap::new();
    for f in faces {
        let m = f.len();
        for i in 0..m {
            let (k0, k1, eq) = pair_key(key(f[i]), key(f[(i + 1) % m]));
            if eq {
                return false;
            }
            *count.entry((k0, k1)).or_insert(0) += 1;
        }
    }
    let table: HashMap<(Key, Key), &Vec<V3>> = match splits {
        Some(s) => s.iter().map(|(k, v)| (*k, &v.2)).collect(),
        None => HashMap::new(),
    };
    let mut uses: HashMap<(Key, Key), i64> = HashMap::new();
    for sec in sections {
        for i in 0..sec.len() - 1 {
            let mut run = vec![sec[i]];
            if let Some(m) = table.get(&(key(sec[i]), key(sec[i + 1]))) {
                run.extend(m.iter().copied());
            }
            run.push(sec[i + 1]);
            for w in run.windows(2) {
                let (k0, k1, eq) = pair_key(key(w[0]), key(w[1]));
                if !eq {
                    *uses.entry((k0, k1)).or_insert(0) += 1;
                }
            }
        }
    }
    let rim: std::collections::HashSet<(Key, Key)> = uses.iter().filter(|(_, &n)| n == 1).map(|(k, _)| *k).collect();
    let open_sides: std::collections::HashSet<(Key, Key)> = count.iter().filter(|(_, &n)| n == 1).map(|(k, _)| *k).collect();
    count.values().all(|&n| n <= 2)
        && open_sides == rim
        && !uses.iter().any(|(k, &n)| n > 1 && count.contains_key(k))
}

/// `_cells`: one grid cell as faces.
fn cells(quad: &[V3; 4], h: Hypot) -> Vec<Vec<V3>> {
    let mut pts: Vec<V3> = Vec::new();
    for &p in quad {
        if !pts.iter().any(|&q| same(p, q, h)) {
            pts.push(p);
        }
    }
    if pts.len() < 3 {
        return Vec::new();
    }
    if pts.len() == 3 {
        return if lenf(newell(&pts), h) > 1e-12 { vec![pts] } else { Vec::new() };
    }
    let n = unit(newell(&pts), h);
    if let Some(n) = n {
        if pts.iter().all(|&p| dotf(p.sub(pts[0]), n).abs() < 1e-7) {
            return vec![pts];
        }
    }
    let mut out = Vec::new();
    for tri in [vec![pts[0], pts[1], pts[2]], vec![pts[0], pts[2], pts[3]]] {
        if lenf(newell(&tri), h) > 1e-12 {
            out.push(tri);
        }
    }
    out
}

/// `_wound_like`.
fn wound_like(faces: Vec<Vec<V3>>, strips: &[Vec<V3>]) -> Vec<Vec<V3>> {
    let mut strip_sides: HashMap<(Key, Key), ()> = HashMap::new();
    for lp in strips {
        let m = lp.len();
        for i in 0..m {
            strip_sides.insert((key(lp[i]), key(lp[(i + 1) % m])), ());
        }
    }
    for f in &faces {
        let m = f.len();
        for i in 0..m {
            let (a, b) = (key(f[i]), key(f[(i + 1) % m]));
            if strip_sides.contains_key(&(a, b)) {
                return faces.iter().map(|x| x.iter().rev().copied().collect()).collect();
            }
            if strip_sides.contains_key(&(b, a)) {
                return faces;
            }
        }
    }
    faces
}

/// `_corner_cut`: the flat close of a box corner.
fn corner_cut(mesh: &TopoMesh, group: &[&EdgeRound], sections: &[Vec<V3>], vid: usize, h: Hypot) -> Option<Vec<Vec<V3>>> {
    let sign = group[0].sign;
    let tip = mesh.pos(vid);
    let mut out = zero();
    for f in mesh.vertex_faces(vid) {
        out = out.add(mesh.face_normal(f, h));
    }
    let mut corners: Vec<(V3, Vec<usize>)> = Vec::new();
    for (ri, sec) in sections.iter().enumerate() {
        for p in [sec[0], sec[sec.len() - 1]] {
            if let Some(c) = corners.iter_mut().find(|c| same(c.0, p, h)) {
                c.1.push(ri);
            } else {
                corners.push((p, vec![ri]));
            }
        }
    }
    if corners.len() != 3 || corners.iter().any(|(_, rs)| rs.len() != 2) {
        return None;
    }
    let mut bottoms = Vec::new();
    for (p, rs) in &corners {
        let (ra, rb) = (group[rs[0]], group[rs[1]]);
        let mut down = unit(ra.d.cross(rb.d), h)?;
        if dotf(down, out) > 0.0 {
            down = down.neg();
        }
        let depth = (lenf(ra.c_off, h) + lenf(rb.c_off, h)) / 2.0 / 2.0f64.sqrt();
        bottoms.push(p.add(mulf(down, depth)));
    }
    let bottom_at = |q: V3| -> Option<V3> {
        for ((p, _), b) in corners.iter().zip(bottoms.iter()) {
            if same(*p, q, h) {
                return Some(*b);
            }
        }
        None
    };
    let centre = !(same(bottoms[0], bottoms[1], h) && same(bottoms[0], bottoms[2], h));
    let mut faces = Vec::new();
    for sec in sections {
        let mut lp: Vec<V3> = sec.clone();
        if centre {
            lp.push(bottom_at(sec[sec.len() - 1])?);
            lp.push(bottom_at(sec[0])?);
        } else {
            lp.push(bottoms[0]);
        }
        faces.push(facing(&lp, tip, sign));
    }
    if centre {
        faces.push(facing(&bottoms, tip, sign));
    }
    Some(faces)
}

type Soft = Vec<(V3, V3)>;

/// `_corner_patch`: triangles filling the corner between three strip ends;
/// the second is the soft pairs.
#[allow(clippy::too_many_arguments)]
fn corner_patch(
    group: &[&EdgeRound],
    sections: &[Vec<V3>],
    radius: f64,
    segments: usize,
    profile: Option<f64>,
    drawn: bool,
    h: Hypot,
) -> (Option<Vec<Vec<V3>>>, Option<Soft>) {
    let n = if drawn { sections[0].len() - 1 } else { segments };
    if sections.iter().any(|s| s.len() != n + 1) {
        return (None, None);
    }
    let (s1, s2, s3) = (&sections[0], &sections[1], &sections[2]);
    let meet5 = |sa: &[V3], sb: &[V3]| -> Option<V3> {
        for i in [0, sa.len() - 1] {
            for j in [0, sb.len() - 1] {
                if lenf(sa[i].sub(sb[j]), h) < 1e-5 {
                    return Some(sa[i]);
                }
            }
        }
        None
    };
    let (c12, c23, c31) = match (meet5(s1, s2), meet5(s2, s3), meet5(s3, s1)) {
        (Some(a), Some(b), Some(c)) => (a, b, c),
        _ => return (None, None),
    };
    let centre = corner_centre(group, sections);
    let (ea, eb, ec) = match (unit(c12.sub(centre), h), unit(c23.sub(centre), h), unit(c31.sub(centre), h)) {
        (Some(a), Some(b), Some(c)) => (a, b, c),
        _ => return (None, None),
    };
    let oriented_arc = |sec: &[V3], start: V3, end: V3| -> Option<Vec<V3>> {
        let last = sec.len() - 1;
        if lenf(sec[0].sub(start), h) < 1e-5 && lenf(sec[last].sub(end), h) < 1e-5 {
            return Some(sec.to_vec());
        }
        if lenf(sec[last].sub(start), h) < 1e-5 && lenf(sec[0].sub(end), h) < 1e-5 {
            return Some(sec.iter().rev().copied().collect());
        }
        None
    };
    let mut arcs: [Option<Vec<V3>>; 3] = [None, None, None]; // ab, bc, ac
    for s in [s1, s2, s3] {
        for (idx, (p, q)) in [(c12, c23), (c23, c31), (c12, c31)].into_iter().enumerate() {
            if let Some(a) = oriented_arc(s, p, q) {
                if arcs[idx].is_none() {
                    arcs[idx] = Some(a);
                }
            }
        }
    }
    let (ab, bc, ac) = match (&arcs[0], &arcs[1], &arcs[2]) {
        (Some(a), Some(b), Some(c)) => (a, b, c),
        _ => return (None, None),
    };
    let nf = n as f64;
    let f = |i: usize| (i as f64 * PI / (2.0 * nf)).sin();
    let e = match profile {
        Some(p) => 2.0 / exponent(p),
        None => 0.0,
    };
    let mut grid: HashMap<(usize, usize), V3> = HashMap::new();
    for i in 0..=n {
        for j in 0..=(n - i) {
            let k = n - i - j;
            let p = if k == 0 {
                ab[j]
            } else if i == 0 {
                bc[k]
            } else if j == 0 {
                ac[k]
            } else if drawn {
                let r0 = lerp(ac[n - i], ab[n - i], j as f64 / (n - i) as f64);
                let r1 = lerp(bc[n - j], ab[j], i as f64 / (n - j) as f64);
                let r2 = lerp(bc[k], ac[k], i as f64 / (n - k) as f64);
                divf(r0.add(r1).add(r2), 3.0)
            } else if profile.is_some() {
                let (wa, wb, wc) = (f(i), f(j), f(k));
                let ln = (wa * wa + wb * wb + wc * wc).sqrt();
                centre
                    .add(mulf(c12.sub(centre), (wa / ln).powf(e)))
                    .add(mulf(c23.sub(centre), (wb / ln).powf(e)))
                    .add(mulf(c31.sub(centre), (wc / ln).powf(e)))
            } else {
                let dirn = match unit(mulf(ea, f(i)).add(mulf(eb, f(j))).add(mulf(ec, f(k))), h) {
                    Some(d) => d,
                    None => return (None, None),
                };
                centre.add(mulf(dirn, radius))
            };
            grid.insert((i, j), p);
        }
    }
    let mut tris = Vec::new();
    let sign = group[0].sign;
    for i in 0..n {
        for j in 0..(n - i) {
            tris.push(oriented(&[grid[&(i, j)], grid[&(i + 1, j)], grid[&(i, j + 1)]], centre, sign));
            if i + j + 1 < n {
                tris.push(oriented(&[grid[&(i + 1, j)], grid[&(i + 1, j + 1)], grid[&(i, j + 1)]], centre, sign));
            }
        }
    }
    let mut soft = Vec::new();
    for i in 0..=n {
        for j in 0..=(n - i) {
            let p = grid[&(i, j)];
            for (di, dj) in [(1i64, 0i64), (0, 1), (-1, 1)] {
                let (ni, nj) = (i as i64 + di, j as i64 + dj);
                if ni < 0 || nj < 0 {
                    continue;
                }
                if let Some(&q) = grid.get(&(ni as usize, nj as usize)) {
                    soft.push((p, q));
                }
            }
        }
    }
    (Some(tris), Some(soft))
}
