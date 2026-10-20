// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! The rest of `core/hlr.py` (M1.8): everything around the visibility
//! kernels of [`crate::hlr`] that is geometry. The camera basis, the clip to
//! a section plane, the section cap (the poché as an occluder), the edge
//! pipeline of `hlr_drawing` (silhouette rule, spans, hidden parts, the
//! Profiles probes), `_merge_collinear`, `_drop_hidden_under_visible`, the
//! perspective path (`hlr_perspective`: clip to the view pyramid, projective
//! map to a parallel view) and `hlr_view`.
//!
//! Plain arrays in, plain arrays out. `collect_geometry` / `collect_circles`
//! read the Scene and stay with the model; they hand this module the
//! collected triangles and edges.
//!
//! The Python runs NumPy over float64 and Python floats over the f32 values
//! Qt hands out, so every operation here is the f64 one in the same order.
//! Where NumPy hands a sum to BLAS the order and the fused multiply-add are
//! those of OpenBLAS 0.3.34's x86-64 kernels (checked against NumPy 2.5 on
//! this machine, see `scripts/golden/make_hlr_fixtures.py`):
//!
//! * `np.dot(a, b)` / `np.linalg.norm` on 3-vectors: `(a0b0 + a1b1) + a2b2`,
//!   no FMA (the `ddot` tail loop).
//! * `(N,3) @ (3,)` (`gemv`): `fma(a2, b2, fma(a0, b0, a1·b1))` for N ≥ 2, the
//!   plain sum for N = 1 (NumPy routes a single row to `dot`).
//! * `(N,3) @ (3,4)` (`gemm`): `fma(a2, b2, fma(a1, b1, a0·b0))` for N ≥ 2,
//!   the plain sum for N = 1 and for a lone vector.
//! * `np.einsum("ij,ij->i")` on 3-vectors: `(a2b2 + a0b0) + a1b1`.
//!
//! Those orders belong to the CPU dispatch of the OpenBLAS inside NumPy; the
//! fixtures were made on a Haswell-class kernel (AVX2 + FMA).

use std::collections::HashMap;

use ingetrazo_math::pyfloat::{py_max2, py_max3, py_min2};
use ingetrazo_math::{Hypot, M4, V3};

use crate::facegeom::plane_axes;
use crate::hlr::{interval_from_linear, solve3, Occluders, T_EPS};
use crate::section;
use crate::sweep::py_hypot2;

type P3 = [f64; 3];
/// One triangle, three corners.
pub type Tri = [P3; 3];
/// One edge, two endpoints.
pub type Seg = [P3; 2];

/// Line classes of a drawing, per segment (`HlrDrawing.kinds`).
pub const KIND_EDGE: i8 = 0;
pub const KIND_PROFILE: i8 = 1;
pub const KIND_CUT: i8 = 2;
pub const KIND_HIDDEN: i8 = 3;

// ── Camera ──────────────────────────────────────────────────────────────────

/// The part of `core.camera.OrbitCamera` the hidden-line pass reads.
#[derive(Clone, Copy, Debug)]
pub struct OrbitCamera {
    pub target: V3,
    pub distance: f64,
    pub yaw: f64,
    pub pitch: f64,
    pub up: V3,
}

impl OrbitCamera {
    /// `OrbitCamera.eye()`: `target + QVector3D(d·cp·cy, d·cp·sy, d·sp)`.
    pub fn eye(&self) -> V3 {
        let cp = self.pitch.cos();
        let sp = self.pitch.sin();
        let cy = self.yaw.cos();
        let sy = self.yaw.sin();
        self.target.add(V3(
            (self.distance * cp * cy) as f32,
            (self.distance * cp * sy) as f32,
            (self.distance * sp) as f32,
        ))
    }

    /// `OrbitCamera.forward()`.
    pub fn forward(&self, h: Hypot) -> V3 {
        let d = self.target.sub(self.eye());
        if (d.length_squared() as f64) > 1e-18 {
            d.normalized(h)
        } else {
            V3(0.0, 1.0, 0.0)
        }
    }

    /// `OrbitCamera.up_vector()`.
    pub fn up_vector(&self, h: Hypot) -> V3 {
        let f = self.forward(h);
        if (f.dot(self.up).abs() as f64) < 0.9999 {
            return self.up;
        }
        let sign: f64 = if self.pitch > 0.0 { -1.0 } else { 1.0 };
        V3((sign * self.yaw.cos()) as f32, (sign * self.yaw.sin()) as f32, 0.0)
    }
}

fn v3f(v: V3) -> P3 {
    [v.0 as f64, v.1 as f64, v.2 as f64]
}

/// `np.dot` / `ddot` of two 3-vectors: left to right, no FMA.
#[inline]
fn dot3(a: P3, b: P3) -> f64 {
    (a[0] * b[0] + a[1] * b[1]) + a[2] * b[2]
}

/// `np.cross` of two 3-vectors.
#[inline]
fn cross3(a: P3, b: P3) -> P3 {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}

/// `rows @ v` for float64 `(N,3) @ (3,)` (OpenBLAS `dgemv`; one row goes to
/// `ddot`).
#[inline]
fn matvec3(row: P3, v: P3, n_rows: usize) -> f64 {
    if n_rows == 1 {
        dot3(row, v)
    } else {
        row[2].mul_add(v[2], row[0].mul_add(v[0], row[1] * v[1]))
    }
}

/// One row of `(N,3) @ M[:, :3].T` (`dgemm`; a single row or vector is
/// plain).
#[inline]
fn matmul_row(p: P3, m: P3, n_rows: usize) -> f64 {
    if n_rows == 1 {
        dot3(p, m)
    } else {
        p[2].mul_add(m[2], p[1].mul_add(m[1], p[0] * m[0]))
    }
}

/// `np.einsum("ij,ij->i")` on one pair of 3-vectors.
#[inline]
fn einsum3(a: P3, b: P3) -> f64 {
    (a[2] * b[2] + a[0] * b[0]) + a[1] * b[1]
}

/// Camera basis: eye, right, up, forward as f64 rows, from an OrbitCamera
/// (the derivation of its lookAt).
#[derive(Clone, Copy, Debug)]
pub struct Basis {
    pub eye: P3,
    pub right: P3,
    pub up: P3,
    pub fwd: P3,
}

/// `core.hlr.camera_basis`.
pub fn camera_basis(camera: &OrbitCamera, h: Hypot) -> Basis {
    let e = v3f(camera.eye());
    let t = v3f(camera.target);
    let up = v3f(camera.up_vector(h));
    let mut f = [t[0] - e[0], t[1] - e[1], t[2] - e[2]];
    let n = py_max2(dot3(f, f).sqrt(), 1e-12);
    f = [f[0] / n, f[1] / n, f[2] / n];
    let mut r = cross3(f, up);
    let n = py_max2(dot3(r, r).sqrt(), 1e-12);
    r = [r[0] / n, r[1] / n, r[2] / n];
    let u = cross3(r, f);
    Basis { eye: e, right: r, up: u, fwd: f }
}

/// `_to_cam`: world points → camera space (x right, y up, z depth away).
fn to_cam(points: &[P3], b: &Basis) -> Vec<P3> {
    let n = points.len();
    points
        .iter()
        .map(|p| {
            let d = [p[0] - b.eye[0], p[1] - b.eye[1], p[2] - b.eye[2]];
            [matvec3(d, b.right, n), matvec3(d, b.up, n), matvec3(d, b.fwd, n)]
        })
        .collect()
}

// ── Section clip ────────────────────────────────────────────────────────────

/// `core.section.SectionPlane`: the hidden side is where `side(p) > 0`.
#[derive(Clone, Copy, Debug)]
pub struct SectionPlane {
    pub normal: V3,
    pub point: V3,
}

/// A soft edge with the normals of its faces; `nb` is `None` for an open
/// boundary.
#[derive(Clone, Copy, Debug)]
pub struct SoftEdge {
    pub p0: P3,
    pub p1: P3,
    pub na: P3,
    pub nb: Option<P3>,
}

/// What `collect_geometry` returns (world space).
#[derive(Clone, Debug, Default)]
pub struct Geometry {
    pub tris: Vec<Tri>,
    pub hard: Vec<Seg>,
    pub soft: Vec<SoftEdge>,
}

/// What `clip_to_section` returns; `cuts` is empty unless `split_cuts`.
#[derive(Clone, Debug, Default)]
pub struct Clipped {
    pub tris: Vec<Tri>,
    pub hard: Vec<Seg>,
    pub soft: Vec<SoftEdge>,
    pub cuts: Vec<Seg>,
}

/// `core.hlr.clip_to_section`.
pub fn clip_to_section(geom: &Geometry, plane: &SectionPlane, split_cuts: bool) -> Clipped {
    let nx = plane.normal.0 as f64;
    let ny = plane.normal.1 as f64;
    let nz = plane.normal.2 as f64;
    let c = nx * plane.point.0 as f64 + ny * plane.point.1 as f64 + nz * plane.point.2 as f64;
    let eps = 1e-9;

    let dist = |p: P3| p[0] * nx + p[1] * ny + p[2] * nz - c;
    let lerp = |a: P3, b: P3, da: f64, db: f64| -> P3 {
        let t = da / (da - db);
        [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t]
    };

    let mut out_tris: Vec<Tri> = Vec::new();
    let mut cut_edges: Vec<Seg> = Vec::new();
    for tri in &geom.tris {
        let d = [dist(tri[0]), dist(tri[1]), dist(tri[2])];
        let hidden = [d[0] > eps, d[1] > eps, d[2] > eps];
        let nh = hidden.iter().filter(|&&x| x).count();
        if nh == 0 {
            out_tris.push(*tri);
            continue;
        }
        if nh == 3 {
            continue;
        }
        // Sutherland–Hodgman against the plane (keep d <= 0), remembering
        // the crossing chord.
        let mut poly: Vec<P3> = Vec::new();
        let mut crossings: Vec<P3> = Vec::new();
        for i in 0..3 {
            let (a, b) = (tri[i], tri[(i + 1) % 3]);
            let (da, db) = (d[i], d[(i + 1) % 3]);
            if da <= eps {
                poly.push(a);
            }
            if (da > eps) != (db > eps) {
                let x = lerp(a, b, da, db);
                poly.push(x);
                crossings.push(x);
            }
        }
        if crossings.len() == 2 {
            cut_edges.push([crossings[0], crossings[1]]);
        }
        for i in 1..poly.len().saturating_sub(1) {
            out_tris.push([poly[0], poly[i], poly[i + 1]]);
        }
    }

    let clip_seg = |p0: P3, p1: P3| -> Option<Seg> {
        let (d0, d1) = (dist(p0), dist(p1));
        if d0 > eps && d1 > eps {
            return None;
        }
        if d0 <= eps && d1 <= eps {
            return Some([p0, p1]);
        }
        let x = lerp(p0, p1, d0, d1);
        Some(if d0 <= eps { [p0, x] } else { [x, p1] })
    };

    let mut out_hard: Vec<Seg> = geom.hard.iter().filter_map(|s| clip_seg(s[0], s[1])).collect();
    if !split_cuts {
        out_hard.extend(cut_edges.iter().copied());
    }
    let out_soft: Vec<SoftEdge> = geom
        .soft
        .iter()
        .filter_map(|s| clip_seg(s.p0, s.p1).map(|seg| SoftEdge { p0: seg[0], p1: seg[1], na: s.na, nb: s.nb }))
        .collect();
    Clipped {
        tris: out_tris,
        hard: out_hard,
        soft: out_soft,
        cuts: if split_cuts { cut_edges } else { Vec::new() },
    }
}

/// `core.hlr.section_loops`: the chords chained into closed rings, in world
/// space. The Python's own walk (no native engine), tolerance as it
/// computes it.
pub fn section_loops(cut_edges: &[Seg]) -> Vec<Vec<P3>> {
    if cut_edges.is_empty() {
        return Vec::new();
    }
    let mut mn = [f64::INFINITY; 3];
    let mut mx = [f64::NEG_INFINITY; 3];
    for e in cut_edges {
        for p in e {
            for k in 0..3 {
                if p[k] < mn[k] {
                    mn[k] = p[k];
                }
                if p[k] > mx[k] {
                    mx[k] = p[k];
                }
            }
        }
    }
    let span = (0..3).map(|k| mx[k] - mn[k]).fold(f64::NEG_INFINITY, f64::max);
    let tol = py_max2(span * 1e-7, 1e-9);
    let (start, pts) = section::loops(cut_edges, tol);
    (0..start.len() - 1).map(|r| pts[start[r] as usize..start[r + 1] as usize].to_vec()).collect()
}

// ── Spans arithmetic ────────────────────────────────────────────────────────

/// `subtract_spans`: `spans` minus `hidden` (both sorted, disjoint).
pub fn subtract_spans(spans: &[(f64, f64)], hidden: &[(f64, f64)]) -> Vec<(f64, f64)> {
    if hidden.is_empty() {
        return spans.to_vec();
    }
    let mut out = Vec::new();
    for &(lo, hi) in spans {
        let mut cur = lo;
        for &(c0, c1) in hidden {
            if c1 <= cur || c0 >= hi {
                continue;
            }
            if c0 - cur > T_EPS {
                out.push((cur, py_min2(c0, hi)));
            }
            cur = py_max2(cur, c1);
            if cur >= hi {
                break;
            }
        }
        if hi - cur > T_EPS {
            out.push((cur, hi));
        }
    }
    out
}

// ── The section cap ─────────────────────────────────────────────────────────

/// `cap_depth_affine`: `(alpha, beta, gamma)` of the cut plane's camera
/// depth, or `None` when it is seen edge-on.
pub fn cap_depth_affine(plane: &SectionPlane, b: &Basis, h: Hypot) -> Option<P3> {
    let (u, v) = plane_axes(plane.normal, h);
    let p = plane.point;
    let (px, py, pz) = (p.0 as f64, p.1 as f64, p.2 as f64);
    let pts = [
        [px, py, pz],
        [px + u.0 as f64, py + u.1 as f64, pz + u.2 as f64],
        [px + v.0 as f64, py + v.1 as f64, pz + v.2 as f64],
    ];
    let c = to_cam(&pts, b);
    let area2 = (c[1][0] - c[0][0]) * (c[2][1] - c[0][1]) - (c[1][1] - c[0][1]) * (c[2][0] - c[0][0]);
    if area2.abs() < 1e-9 {
        return None;
    }
    let m = [[c[0][0], c[0][1], 1.0], [c[1][0], c[1][1], 1.0], [c[2][0], c[2][1], 1.0]];
    solve3(m, [c[0][2], c[1][2], c[2][2]])
}

/// `cap_ring_edges`: every ring as endpoint pairs `P→Q`, all rings stacked.
pub fn cap_ring_edges(loops: &[Vec<[f64; 2]>]) -> (Vec<[f64; 2]>, Vec<[f64; 2]>) {
    let mut p = Vec::new();
    let mut q = Vec::new();
    for r in loops {
        for i in 0..r.len() {
            p.push(r[i]);
            q.push(r[(i + 1) % r.len()]);
        }
    }
    (p, q)
}

/// `_inside_cap`: even-odd point-in-region over the stacked ring edges.
fn inside_cap(px: f64, py: f64, p: &[[f64; 2]], q: &[[f64; 2]]) -> bool {
    let mut any = false;
    let mut count = 0usize;
    for k in 0..p.len() {
        let (y0, y1) = (p[k][1], q[k][1]);
        if (y0 > py) != (y1 > py) {
            any = true;
            let (xa, xb) = (p[k][0], q[k][0]);
            let xi = xa + (xb - xa) * (py - y0) / (y1 - y0);
            if xi > px {
                count += 1;
            }
        }
    }
    any && (count & 1) == 1
}

/// `cap_hidden_spans`: the t-intervals of one edge the cap hides.
pub fn cap_hidden_spans(
    a2: [f64; 2],
    b2: [f64; 2],
    az: f64,
    bz: f64,
    p: &[[f64; 2]],
    q: &[[f64; 2]],
    abg: P3,
    eps: f64,
) -> Vec<(f64, f64)> {
    let d2 = [b2[0] - a2[0], b2[1] - a2[1]];
    let behind = interval_from_linear(
        az - (abg[0] * a2[0] + abg[1] * a2[1] + abg[2]) - eps,
        (bz - az) - (abg[0] * d2[0] + abg[1] * d2[1]),
    );
    let Some(behind) = behind else { return Vec::new() };
    let lo = py_max2(0.0, behind.0);
    let hi = py_min2(1.0, behind.1);
    if hi - lo <= T_EPS {
        return Vec::new();
    }
    // Where the edge crosses the outline the parity flips: cut it there and
    // ask each piece once, at its midpoint.
    let mut cuts: Vec<f64> = Vec::new();
    for k in 0..p.len() {
        let e = [q[k][0] - p[k][0], q[k][1] - p[k][1]];
        let w = [p[k][0] - a2[0], p[k][1] - a2[1]];
        let denom = d2[0] * e[1] - d2[1] * e[0];
        if denom.abs() > 1e-15 {
            let t = (w[0] * e[1] - w[1] * e[0]) / denom;
            let s = (w[0] * d2[1] - w[1] * d2[0]) / denom;
            if s >= 0.0 && s <= 1.0 && t > lo && t < hi {
                cuts.push(t);
            }
        }
    }
    cuts.sort_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal));
    let mut bounds = Vec::with_capacity(cuts.len() + 2);
    bounds.push(lo);
    bounds.extend(cuts);
    bounds.push(hi);
    let mut out: Vec<(f64, f64)> = Vec::new();
    for i in 0..bounds.len() - 1 {
        let (t0, t1) = (bounds[i], bounds[i + 1]);
        if t1 - t0 <= T_EPS {
            continue;
        }
        let tm = (t0 + t1) * 0.5;
        if inside_cap(a2[0] + tm * d2[0], a2[1] + tm * d2[1], p, q) {
            if let Some(last) = out.last_mut() {
                if t0 - last.1 <= T_EPS {
                    last.1 = t1;
                    continue;
                }
            }
            out.push((t0, t1));
        }
    }
    out
}

// ── The drawing ─────────────────────────────────────────────────────────────

/// What `hlr_drawing` returns: segments in the camera plane, their world
/// endpoints, a class per segment and the closed section-cut rings.
#[derive(Clone, Debug, Default)]
pub struct HlrDrawing {
    pub segs: Vec<[f64; 4]>,
    pub world: Vec<Seg>,
    pub kinds: Vec<i8>,
    pub loops: Vec<Vec<[f64; 2]>>,
}

/// `hlr_drawing`'s keyword arguments.
#[derive(Clone, Copy, Debug)]
pub struct HlrOptions {
    pub profiles: bool,
    pub fills: bool,
    pub caps: bool,
    pub hidden: bool,
}

impl Default for HlrOptions {
    fn default() -> Self {
        HlrOptions { profiles: true, fills: true, caps: true, hidden: false }
    }
}

fn or_one(x: f64) -> f64 {
    // Python `x or 1.0`
    if x == 0.0 {
        1.0
    } else {
        x
    }
}

fn lerp_pt(a: P3, b: P3, t: f64) -> P3 {
    [a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]), a[2] + t * (b[2] - a[2])]
}

/// `hlr_drawing`. `vectorised` says the geometry came as the arrays
/// `Viewport.hlr_geometry()` returns (`geometry=` given): the silhouette
/// rule then runs as one `soft_n @ fwd`, whose rounding differs from the
/// per-edge `np.dot` of `collect_geometry`'s lists (a section plane turns
/// the arrays back into lists, as in Python).
pub fn hlr_drawing(
    geom: &Geometry,
    vectorised: bool,
    section: Option<&SectionPlane>,
    camera: &OrbitCamera,
    opts: &HlrOptions,
    h: Hypot,
) -> HlrDrawing {
    let mut tris = geom.tris.clone();
    let mut hard = geom.hard.clone();
    let mut soft = geom.soft.clone();
    let mut vec_rule = vectorised;
    let mut cut_edges: Vec<Seg> = Vec::new();
    if let Some(sp) = section {
        let c = clip_to_section(geom, sp, true);
        tris = c.tris;
        hard = c.hard;
        soft = c.soft;
        cut_edges = c.cuts;
        vec_rule = false;
    }
    let fwd0 = camera_basis(camera, h).fwd;

    let n_hard = hard.len();
    if vec_rule {
        if !soft.is_empty() {
            let n = soft.len();
            let nan = [f64::NAN; 3];
            for s in &soft {
                let fa = matvec3(s.na, fwd0, n);
                let fb = matvec3(s.nb.unwrap_or(nan), fwd0, n);
                if fb.is_nan() || ((fa < 0.0) != (fb < 0.0)) {
                    hard.push([s.p0, s.p1]);
                }
            }
        }
    } else {
        for s in &soft {
            let fa = dot3(s.na, fwd0);
            match s.nb {
                None => hard.push([s.p0, s.p1]), // open boundary: always a profile
                Some(nb) => {
                    let fb = dot3(nb, fwd0);
                    if (fa < 0.0) != (fb < 0.0) {
                        hard.push([s.p0, s.p1]);
                    }
                }
            }
        }
    }
    if hard.is_empty() && cut_edges.is_empty() {
        return HlrDrawing::default();
    }
    // Row layout of E: [hard edges | soft silhouettes | cut chords].
    let n_sil = hard.len() - n_hard;
    let mut e_all: Vec<Seg> = hard;
    e_all.extend(cut_edges.iter().copied());

    let basis = camera_basis(camera, h);
    let starts: Vec<P3> = e_all.iter().map(|e| e[0]).collect();
    let ends: Vec<P3> = e_all.iter().map(|e| e[1]).collect();
    let a_cam = to_cam(&starts, &basis);
    let b_cam = to_cam(&ends, &basis);

    let have_tris = !tris.is_empty();
    let mut tc: Vec<Tri> = Vec::new();
    let (mut eps, mut delta, mut ptol) = (0.0f64, 0.0f64, 0.0f64);
    if have_tris {
        let flat: Vec<P3> = tris.iter().flat_map(|t| t.iter().copied()).collect();
        let cam = to_cam(&flat, &basis);
        tc = cam.chunks(3).map(|c| [c[0], c[1], c[2]]).collect();
        let (mut zmin, mut zmax) = (f64::INFINITY, f64::NEG_INFINITY);
        let (mut xmin, mut xmax) = (f64::INFINITY, f64::NEG_INFINITY);
        let (mut ymin, mut ymax) = (f64::INFINITY, f64::NEG_INFINITY);
        for t in &tc {
            for v in t {
                zmin = zmin.min(v[2]);
                zmax = zmax.max(v[2]);
                xmin = xmin.min(v[0]);
                xmax = xmax.max(v[0]);
                ymin = ymin.min(v[1]);
                ymax = ymax.max(v[1]);
            }
        }
        // depth-based epsilon: relative to the scene's depth span
        let span = or_one(zmax - zmin);
        eps = py_max2(span * 1e-4, 1e-9);
        // the sideways probe of the profile test: a hair, in drawing units
        let ext = or_one(f64::max(xmax - xmin, ymax - ymin));
        delta = py_max2(ext * 1e-4, 1e-9);
        // a surface counts as adjacent up to slope 10 (≈ 84°)
        ptol = eps + delta * 10.0;
    }

    let (mut xmin, mut xmax) = (f64::INFINITY, f64::NEG_INFINITY);
    let (mut ymin, mut ymax) = (f64::INFINITY, f64::NEG_INFINITY);
    for v in a_cam.iter().chain(b_cam.iter()) {
        xmin = xmin.min(v[0]);
        xmax = xmax.max(v[0]);
        ymin = ymin.min(v[1]);
        ymax = ymax.max(v[1]);
    }
    let ext_e = or_one(f64::max(xmax - xmin, ymax - ymin));
    let zero_len = ext_e * 1e-9; // a vertical seen end-on: a point

    // The section cap: the rings the composer fills also COVER.
    let mut loops: Vec<Vec<[f64; 2]>> = Vec::new();
    let mut cap: Option<(Vec<[f64; 2]>, Vec<[f64; 2]>, P3, f64, ([f64; 2], [f64; 2]))> = None;
    if !cut_edges.is_empty() {
        let sp = section.expect("cut edges come from a section plane");
        for ring in section_loops(&cut_edges) {
            loops.push(to_cam(&ring, &basis).iter().map(|p| [p[0], p[1]]).collect());
        }
        let abg = if !loops.is_empty() && opts.caps { cap_depth_affine(sp, &basis, h) } else { None };
        if let Some(abg) = abg {
            let (cp, cq) = cap_ring_edges(&loops);
            let (mut zlo, mut zhi) = (f64::INFINITY, f64::NEG_INFINITY);
            for v in a_cam.iter().chain(b_cam.iter()) {
                zlo = zlo.min(v[2]);
                zhi = zhi.max(v[2]);
            }
            let cap_eps = py_max3((zhi - zlo) * 1e-4, eps, 1e-9);
            let (mut bmin, mut bmax) = ([f64::INFINITY; 2], [f64::NEG_INFINITY; 2]);
            for p in &cp {
                for k in 0..2 {
                    bmin[k] = bmin[k].min(p[k]);
                    bmax[k] = bmax[k].max(p[k]);
                }
            }
            cap = Some((cp, cq, abg, cap_eps, (bmin, bmax)));
        }
    }

    // The two costly questions, answered for every row at once.
    let occ = if have_tris { Some(Occluders::new(tc.clone())) } else { None };
    let (n_off, n_spans) = match &occ {
        Some(o) => {
            let edges: Vec<(P3, P3)> = a_cam.iter().zip(b_cam.iter()).map(|(a, b)| (*a, *b)).collect();
            o.visible_spans_all(&edges, eps)
        }
        None => (Vec::new(), Vec::new()),
    };

    let mut out: Vec<[f64; 4]> = Vec::new();
    let mut out_w: Vec<Seg> = Vec::new();
    let mut out_k: Vec<i8> = Vec::new();
    let mut probes: Vec<(usize, [f64; 2], [f64; 2], f64)> = Vec::new();
    for i in 0..a_cam.len() {
        let a2 = [a_cam[i][0], a_cam[i][1]];
        let b2 = [b_cam[i][0], b_cam[i][1]];
        let (az, bz) = (a_cam[i][2], b_cam[i][2]);
        let is_cut = i >= n_hard + n_sil;
        let base_kind = if is_cut {
            KIND_CUT
        } else if i >= n_hard {
            KIND_PROFILE
        } else {
            KIND_EDGE
        };
        let mut spans: Vec<(f64, f64)> = if occ.is_some() {
            n_spans[n_off[i] as usize..n_off[i + 1] as usize].to_vec()
        } else {
            vec![(0.0, 1.0)]
        };
        // The hidden part: the rest of the edge, before the section cap
        // takes what lies inside the material.
        let mut hid = if opts.hidden && i < n_hard { subtract_spans(&[(0.0, 1.0)], &spans) } else { Vec::new() };
        if let Some((cp, cq, abg, cap_eps, (bmin, bmax))) = &cap {
            if !spans.is_empty() || !hid.is_empty() {
                let elo = [py_min2(a2[0], b2[0]), py_min2(a2[1], b2[1])];
                let ehi = [py_max2(a2[0], b2[0]), py_max2(a2[1], b2[1])];
                if !(ehi[0] < bmin[0] || elo[0] > bmax[0] || ehi[1] < bmin[1] || elo[1] > bmax[1]) {
                    let inside = cap_hidden_spans(a2, b2, az, bz, cp, cq, *abg, *cap_eps);
                    spans = subtract_spans(&spans, &inside);
                    hid = subtract_spans(&hid, &inside);
                }
            }
        }
        let (w0, w1) = (e_all[i][0], e_all[i][1]);
        for &(t0, t1) in &hid {
            if (t1 - t0) < 1e-9 {
                continue;
            }
            let p = [a2[0] + t0 * (b2[0] - a2[0]), a2[1] + t0 * (b2[1] - a2[1])];
            let q = [a2[0] + t1 * (b2[0] - a2[0]), a2[1] + t1 * (b2[1] - a2[1])];
            if py_hypot2(q[0] - p[0], q[1] - p[1]) < zero_len {
                continue;
            }
            out.push([p[0], p[1], q[0], q[1]]);
            out_k.push(KIND_HIDDEN);
            out_w.push([lerp_pt(w0, w1, t0), lerp_pt(w0, w1, t1)]);
        }
        for &(t0, t1) in &spans {
            if (t1 - t0) < 1e-9 {
                continue;
            }
            let p = [a2[0] + t0 * (b2[0] - a2[0]), a2[1] + t0 * (b2[1] - a2[1])];
            let q = [a2[0] + t1 * (b2[0] - a2[0]), a2[1] + t1 * (b2[1] - a2[1])];
            if py_hypot2(q[0] - p[0], q[1] - p[1]) < zero_len {
                continue;
            }
            let kind = base_kind;
            if opts.profiles && kind == KIND_EDGE && have_tris {
                // Profile test: probe a hair to each side of the visible
                // span's midpoint; background on either side = outline.
                let d = [q[0] - p[0], q[1] - p[1]];
                let ln = py_hypot2(d[0], d[1]);
                if ln > 1e-12 {
                    let nrm = [(-d[1] / ln) * delta, (d[0] / ln) * delta];
                    let mid = [(p[0] + q[0]) * 0.5, (p[1] + q[1]) * 0.5];
                    let tm = (t0 + t1) * 0.5;
                    let zm = az + tm * (bz - az);
                    probes.push((out.len(), mid, nrm, zm));
                }
            }
            out.push([p[0], p[1], q[0], q[1]]);
            out_k.push(kind);
            out_w.push([lerp_pt(w0, w1, t0), lerp_pt(w0, w1, t1)]);
        }
    }
    if !probes.is_empty() {
        let o = occ.as_ref().expect("probes need triangles");
        let flat: Vec<([f64; 2], [f64; 2], f64)> = probes.iter().map(|p| (p.1, p.2, p.3)).collect();
        let flags = o.profile_flags(&flat, delta, ptol);
        for (pr, outline) in probes.iter().zip(flags) {
            if outline {
                out_k[pr.0] = KIND_PROFILE;
            }
        }
    }
    let (mut segs, mut world, mut kinds) = (out, out_w, out_k);
    if segs.len() > 1 {
        let r = merge_collinear(&segs, &world, &kinds, ext_e * 1e-9);
        segs = r.0;
        world = r.1;
        kinds = r.2;
    }
    if opts.hidden && !segs.is_empty() {
        let r = drop_hidden_under_visible(&segs, &world, &kinds, py_max2(ext_e * 1e-6, 1e-9));
        segs = r.0;
        world = r.1;
        kinds = r.2;
    }
    HlrDrawing { segs, world, kinds, loops: if opts.fills { loops } else { Vec::new() } }
}

/// `hlr_view`: the segments of `hlr_drawing(profiles=False, fills=False)`.
pub fn hlr_view(
    geom: &Geometry,
    vectorised: bool,
    section: Option<&SectionPlane>,
    camera: &OrbitCamera,
    h: Hypot,
) -> HlrDrawing {
    let opts = HlrOptions { profiles: false, fills: false, caps: true, hidden: false };
    hlr_drawing(geom, vectorised, section, camera, &opts, h)
}

// ── Collinear merge ─────────────────────────────────────────────────────────

fn floor_i(x: f64) -> i64 {
    x.floor() as i64
}

type Out = (Vec<[f64; 4]>, Vec<Seg>, Vec<i8>);

/// `_merge_collinear`: runs of segments of one class that meet end to end and
/// run collinear in 3D become single segments.
fn merge_collinear(segs: &[[f64; 4]], world: &[Seg], kinds: &[i8], tol: f64) -> Out {
    let n = segs.len();
    if n < 2 {
        return (segs.to_vec(), world.to_vec(), kinds.to_vec());
    }
    let inv = 1.0 / tol;
    let mut cells: HashMap<(i64, i64), Vec<(usize, usize)>> = HashMap::new();
    for i in 0..n {
        for end in 0..2 {
            let (x, y) = (segs[i][2 * end], segs[i][2 * end + 1]);
            cells.entry((floor_i(x * inv), floor_i(y * inv))).or_default().push((i, end));
        }
    }
    let mut used = vec![false; n];
    let (mut out_s, mut out_w, mut out_k): Out = (Vec::new(), Vec::new(), Vec::new());

    // From segment i's endpoint `end`, walk over collinear neighbours;
    // returns the far endpoint reached (2D, world).
    let follow = |used: &mut Vec<bool>, i: usize, end: usize, wdir: P3| -> ([f64; 2], P3) {
        let mut p2 = [segs[i][2 * end], segs[i][2 * end + 1]];
        let mut pw = world[i][end];
        loop {
            let mut nxt: Option<(usize, usize)> = None;
            let (cx, cy) = (floor_i(p2[0] * inv), floor_i(p2[1] * inv));
            'search: for dx in -1..=1i64 {
                for dy in -1..=1i64 {
                    let Some(list) = cells.get(&(cx + dx, cy + dy)) else { continue };
                    for &(j, jend) in list {
                        if j == i || used[j] || kinds[j] != kinds[i] {
                            continue;
                        }
                        let q2 = [segs[j][2 * jend], segs[j][2 * jend + 1]];
                        if (q2[0] - p2[0]).abs() > tol || (q2[1] - p2[1]).abs() > tol {
                            continue;
                        }
                        let wj = world[j][jend];
                        let m = f64::max(
                            f64::max((wj[0] - pw[0]).abs(), (wj[1] - pw[1]).abs()),
                            (wj[2] - pw[2]).abs(),
                        );
                        if m > tol {
                            continue;
                        }
                        let wo = world[j][1 - jend];
                        let other = [wo[0] - wj[0], wo[1] - wj[1], wo[2] - wj[2]];
                        let ln = dot3(other, other).sqrt();
                        if ln < 1e-15 {
                            continue;
                        }
                        let cr = cross3(wdir, [other[0] / ln, other[1] / ln, other[2] / ln]);
                        if dot3(cr, cr) > 1e-14 || dot3(wdir, other) <= 0.0 {
                            continue;
                        }
                        nxt = Some((j, jend));
                        break 'search;
                    }
                }
            }
            let Some((j, jend)) = nxt else { return (p2, pw) };
            used[j] = true;
            p2 = [segs[j][2 * (1 - jend)], segs[j][2 * (1 - jend) + 1]];
            pw = world[j][1 - jend];
        }
    };

    for i in 0..n {
        if used[i] {
            continue;
        }
        used[i] = true;
        let d = [world[i][1][0] - world[i][0][0], world[i][1][1] - world[i][0][1], world[i][1][2] - world[i][0][2]];
        let ln = dot3(d, d).sqrt();
        if ln < 1e-15 {
            out_s.push(segs[i]);
            out_w.push(world[i]);
            out_k.push(kinds[i]);
            continue;
        }
        let wdir = [d[0] / ln, d[1] / ln, d[2] / ln];
        let (b2, bw) = follow(&mut used, i, 1, wdir);
        let (a2, aw) = follow(&mut used, i, 0, [-wdir[0], -wdir[1], -wdir[2]]);
        out_s.push([a2[0], a2[1], b2[0], b2[1]]);
        out_w.push([aw, bw]);
        out_k.push(kinds[i]);
    }
    (out_s, out_w, out_k)
}

// ── Hidden under visible ────────────────────────────────────────────────────

type LineKey = (i64, i64);

/// `_drop_hidden_under_visible`: a hidden line that falls on a visible one is
/// not drawn.
fn drop_hidden_under_visible(segs: &[[f64; 4]], world: &[Seg], kinds: &[i8], tol: f64) -> Out {
    if !kinds.contains(&KIND_HIDDEN) {
        return (segs.to_vec(), world.to_vec(), kinds.to_vec());
    }
    let atol = 1e-6; // radians
    let inv_d = 1.0 / py_max2(tol, 1e-12);

    let key_and_span = |row: [f64; 4]| -> Option<(LineKey, (f64, f64))> {
        let [x0, y0, x1, y1] = row;
        let (dx, dy) = (x1 - x0, y1 - y0);
        let ln = py_hypot2(dx, dy);
        if ln < 1e-15 {
            return None;
        }
        let (mut ux, mut uy) = (dx / ln, dy / ln);
        if ux < -1e-12 || (ux.abs() <= 1e-12 && uy < 0.0) {
            ux = -ux; // one direction per line
            uy = -uy;
        }
        let ang = uy.atan2(ux);
        let off = ux * y0 - uy * x0; // signed distance of the line
        let t0 = ux * x0 + uy * y0;
        let t1 = ux * x1 + uy * y1;
        Some((
            ((ang / atol).round_ties_even() as i64, (off * inv_d).round_ties_even() as i64),
            (py_min2(t0, t1), py_max2(t0, t1)),
        ))
    };

    let mut lines: HashMap<LineKey, Vec<(f64, f64)>> = HashMap::new();
    for i in 0..segs.len() {
        if kinds[i] != KIND_HIDDEN {
            if let Some((k, s)) = key_and_span(segs[i]) {
                lines.entry(k).or_default().push(s);
            }
        }
    }
    if lines.is_empty() {
        return (segs.to_vec(), world.to_vec(), kinds.to_vec());
    }
    let (mut keep_s, mut keep_w, mut keep_k): Out = (Vec::new(), Vec::new(), Vec::new());
    for i in 0..segs.len() {
        if kinds[i] != KIND_HIDDEN {
            keep_s.push(segs[i]);
            keep_w.push(world[i]);
            keep_k.push(kinds[i]);
            continue;
        }
        let Some(((ka, ko), (lo, hi))) = key_and_span(segs[i]) else { continue };
        // Hidden pieces already kept cover the line too: a box's front and
        // back edges fall on the same line on paper, once is enough.
        let mut cover: Vec<(f64, f64)> = Vec::new();
        for da in -1..=1i64 {
            for dof in -1..=1i64 {
                if let Some(l) = lines.get(&(ka + da, ko + dof)) {
                    cover.extend(l.iter().copied());
                }
            }
        }
        if cover.is_empty() {
            keep_s.push(segs[i]);
            keep_w.push(world[i]);
            keep_k.push(kinds[i]);
            lines.entry((ka, ko)).or_default().push((lo, hi));
            continue;
        }
        let span = hi - lo;
        if span < 1e-15 {
            continue;
        }
        let mut cov: Vec<(f64, f64)> = cover
            .iter()
            .filter(|(c0, c1)| *c1 > lo && *c0 < hi)
            .map(|(c0, c1)| (py_max2(0.0, (c0 - lo) / span), py_min2(1.0, (c1 - lo) / span)))
            .collect();
        cov.sort_by(|p, q| {
            p.0.partial_cmp(&q.0).unwrap_or(std::cmp::Ordering::Equal).then(p.1.partial_cmp(&q.1).unwrap_or(std::cmp::Ordering::Equal))
        });
        let mut merged: Vec<(f64, f64)> = Vec::new();
        for (c0, c1) in cov {
            if let Some(last) = merged.last_mut() {
                if c0 <= last.1 {
                    last.1 = py_max2(last.1, c1);
                    continue;
                }
            }
            merged.push((c0, c1));
        }
        // The segment may run either way along its line.
        let [x0, y0, x1, y1] = segs[i];
        let hyp = py_hypot2(x1 - x0, y1 - y0);
        let ux = (x1 - x0) / hyp;
        let uy = (y1 - y0) / hyp;
        let forward = ux > 1e-12 || (ux.abs() <= 1e-12 && uy > 0.0);
        let (w0, w1) = (world[i][0], world[i][1]);
        for (t0, t1) in subtract_spans(&[(0.0, 1.0)], &merged) {
            let (a, b) = if forward { (t0, t1) } else { (1.0 - t1, 1.0 - t0) };
            keep_s.push([x0 + a * (x1 - x0), y0 + a * (y1 - y0), x0 + b * (x1 - x0), y0 + b * (y1 - y0)]);
            keep_w.push([lerp_pt(w0, w1, a), lerp_pt(w0, w1, b)]);
            keep_k.push(KIND_HIDDEN);
            lines.entry((ka, ko)).or_default().push((lo + t0 * span, lo + t1 * span));
        }
    }
    (keep_s, keep_w, keep_k)
}

// ── Perspective ─────────────────────────────────────────────────────────────

/// The view pyramid in clip space, rows `a` with the kept side a·p ≥ 0:
/// left, right, bottom, top, near.
const PYRAMID: [[f64; 4]; 5] = [
    [1.0, 0.0, 0.0, 1.0],
    [-1.0, 0.0, 0.0, 1.0],
    [0.0, 1.0, 0.0, 1.0],
    [0.0, -1.0, 0.0, 1.0],
    [0.0, 0.0, 1.0, 1.0],
];

type P4 = [f64; 4];

#[inline]
fn dot4(a: P4, p: P4) -> f64 {
    ((a[0] * p[0] + a[1] * p[1]) + a[2] * p[2]) + a[3] * p[3]
}

/// `_clip_polygon`: Sutherland–Hodgman against the view pyramid.
fn clip_polygon(mut poly: Vec<P4>) -> Vec<P4> {
    for a in &PYRAMID {
        let mut out: Vec<P4> = Vec::new();
        let n = poly.len();
        for i in 0..n {
            let (p, q) = (poly[i], poly[(i + 1) % n]);
            let (dp, dq) = (dot4(*a, p), dot4(*a, q));
            if dp >= 0.0 {
                out.push(p);
            }
            if (dp >= 0.0) != (dq >= 0.0) {
                let t = dp / (dp - dq);
                out.push([
                    p[0] + (q[0] - p[0]) * t,
                    p[1] + (q[1] - p[1]) * t,
                    p[2] + (q[2] - p[2]) * t,
                    p[3] + (q[3] - p[3]) * t,
                ]);
            }
        }
        poly = out;
        if poly.len() < 3 {
            return Vec::new();
        }
    }
    poly
}

/// `_clip_triangles`: clip-space triangles → the parts inside the pyramid.
fn clip_triangles(c: &[[P4; 3]]) -> Vec<[P4; 3]> {
    let mut parts: Vec<[P4; 3]> = Vec::new();
    let mut fans: Vec<[P4; 3]> = Vec::new();
    for tri in c {
        let d: Vec<[f64; 5]> = tri
            .iter()
            .map(|v| {
                let mut r = [0.0; 5];
                for (k, a) in PYRAMID.iter().enumerate() {
                    r[k] = dot4(*a, *v);
                }
                r
            })
            .collect();
        let inside = d.iter().all(|r| r.iter().all(|&x| x >= 0.0));
        let outside = (0..5).any(|k| d.iter().all(|r| r[k] < 0.0));
        if inside {
            parts.push(*tri);
        } else if !outside {
            let poly = clip_polygon(tri.to_vec());
            for i in 1..poly.len().saturating_sub(1) {
                fans.push([poly[0], poly[i], poly[i + 1]]);
            }
        }
    }
    parts.extend(fans);
    parts
}

/// numpy `np.maximum` / `np.minimum` propagate NaN.
fn np_max(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else if a >= b {
        a
    } else {
        b
    }
}

fn np_min(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else if a <= b {
        a
    } else {
        b
    }
}

/// `_clip_segments`: clip-space segments → the parts inside the pyramid.
fn clip_segments(a: &[P4], b: &[P4]) -> (Vec<P4>, Vec<P4>) {
    let n = a.len();
    let mut t0 = vec![0.0f64; n];
    let mut t1 = vec![1.0f64; n];
    let mut keep = vec![true; n];
    for pl in &PYRAMID {
        for i in 0..n {
            let (da, db) = (dot4(a[i], *pl), dot4(b[i], *pl));
            if da < 0.0 && db < 0.0 {
                keep[i] = false;
            }
            let t = da / (da - db);
            if da < 0.0 {
                t0[i] = np_max(t0[i], t);
            }
            if db < 0.0 {
                t1[i] = np_min(t1[i], t);
            }
        }
    }
    let mut oa = Vec::new();
    let mut ob = Vec::new();
    for i in 0..n {
        if !(keep[i] && t0[i] < t1[i]) {
            continue;
        }
        let d = [b[i][0] - a[i][0], b[i][1] - a[i][1], b[i][2] - a[i][2], b[i][3] - a[i][3]];
        oa.push([a[i][0] + d[0] * t0[i], a[i][1] + d[1] * t0[i], a[i][2] + d[2] * t0[i], a[i][3] + d[3] * t0[i]]);
        ob.push([a[i][0] + d[0] * t1[i], a[i][1] + d[1] * t1[i], a[i][2] + d[2] * t1[i], a[i][3] + d[3] * t1[i]]);
    }
    (oa, ob)
}

/// The perspective of `hlr_perspective`: the camera's matrices (the view
/// matrix may be the two-point one) and the lens.
#[derive(Clone, Copy, Debug)]
pub struct PerspectiveView {
    pub view: M4,
    pub projection: M4,
    pub fov_deg: f64,
    pub aspect: f64,
}

/// `hlr_perspective`: the visible segments in the camera's perspective, as
/// `(x0, y0, x1, y1)` in metres on the picture plane through the target.
pub fn hlr_perspective(
    geom: &Geometry,
    section: Option<&SectionPlane>,
    camera: &OrbitCamera,
    pv: &PerspectiveView,
    h: Hypot,
) -> Vec<[f64; 4]> {
    let (tris, hard, soft, cut) = match section {
        Some(sp) => {
            let c = clip_to_section(geom, sp, true);
            (c.tris, c.hard, c.soft, c.cuts)
        }
        None => (geom.tris.clone(), geom.hard.clone(), geom.soft.clone(), Vec::new()),
    };
    let mut lines: Vec<Seg> = hard;
    lines.extend(cut.iter().copied());
    if !soft.is_empty() {
        // A soft edge is a profile where its two faces turn opposite ways to
        // the EYE — in perspective the sight line differs per edge.
        let e = v3f(camera.eye());
        let nan = [f64::NAN; 3];
        for s in &soft {
            let v = [
                (s.p0[0] + s.p1[0]) / 2.0 - e[0],
                (s.p0[1] + s.p1[1]) / 2.0 - e[1],
                (s.p0[2] + s.p1[2]) / 2.0 - e[2],
            ];
            let fa = einsum3(s.na, v);
            let fb = einsum3(s.nb.unwrap_or(nan), v);
            if fb.is_nan() || ((fa < 0.0) != (fb < 0.0)) {
                lines.push([s.p0, s.p1]);
            }
        }
    }

    let m4 = pv.projection.mul(&pv.view);
    let data = m4.data();
    // M[r][c] = data[c*4 + r]
    let mm = |r: usize, c: usize| data[c * 4 + r] as f64;
    let to_clip = |p: &[P3]| -> Vec<P4> {
        let n = p.len();
        p.iter()
            .map(|q| {
                let mut o = [0.0f64; 4];
                for r in 0..4 {
                    o[r] = matmul_row(*q, [mm(r, 0), mm(r, 1), mm(r, 2)], n) + mm(r, 3);
                }
                o
            })
            .collect()
    };

    let t = v3f(camera.target);
    let w_t = to_clip(&[t])[0][3];
    if w_t <= 1e-9 {
        return Vec::new();
    }
    let half_h = w_t * (pv.fov_deg * (std::f64::consts::PI / 180.0) / 2.0).tan();
    let scale = [half_h * pv.aspect, half_h, w_t];
    let to_parallel = |c: P4| -> P3 {
        let w = c[3];
        [(c[0] / w) * scale[0], (c[1] / w) * scale[1], (1.0 / w) * scale[2]]
    };

    let flat: Vec<P3> = tris.iter().flat_map(|t| t.iter().copied()).collect();
    let clip_flat = to_clip(&flat);
    let ctris: Vec<[P4; 3]> = clip_flat.chunks(3).map(|c| [c[0], c[1], c[2]]).collect();
    let tclip = clip_triangles(&ctris);
    let starts: Vec<P3> = lines.iter().map(|e| e[0]).collect();
    let ends: Vec<P3> = lines.iter().map(|e| e[1]).collect();
    let (ca, cb) = clip_segments(&to_clip(&starts), &to_clip(&ends));
    if ca.is_empty() {
        return Vec::new();
    }
    let tp: Vec<Tri> = tclip.iter().map(|t| [to_parallel(t[0]), to_parallel(t[1]), to_parallel(t[2])]).collect();
    let lp: Vec<Seg> = ca.iter().zip(cb.iter()).map(|(a, b)| [to_parallel(*a), to_parallel(*b)]).collect();

    // A parallel camera looking down −Z onto that space: x right, y up, and
    // the depth growing away from it.
    let t_max = if tp.is_empty() {
        0.0
    } else {
        tp.iter().flat_map(|t| t.iter()).map(|v| v[2]).fold(f64::NEG_INFINITY, f64::max)
    };
    let l_max = lp.iter().flat_map(|s| s.iter()).map(|v| v[2]).fold(f64::NEG_INFINITY, f64::max);
    let top = OrbitCamera {
        target: V3(0.0, 0.0, 0.0),
        distance: py_max2(t_max, l_max) + 1.0,
        yaw: -std::f64::consts::PI / 2.0,
        pitch: std::f64::consts::PI / 2.0,
        up: V3(0.0, 0.0, 1.0),
    };
    let g = Geometry { tris: tp, hard: lp, soft: Vec::new() };
    let opts = HlrOptions { profiles: false, fills: false, caps: false, hidden: false };
    hlr_drawing(&g, true, None, &top, &opts, h).segs
}
