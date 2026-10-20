// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! The slice of ezdxf (1.4.5, with its C extensions) that `formats/dxf_in.py`
//! leans on, op for op in f64: `Vec3`, `OCS`, `Matrix44`, `ConstructionEllipse`,
//! the cubic Bezier arc/ellipse construction, `Bezier4P.flattening`, `Path`
//! and the 2D polyline (bulge) builder. Every formula keeps ezdxf's operand
//! order so the f64 results match; trigonometry and `hypot` are the platform
//! libm's, as in the Python extension modules.
//!
//! ezdxf raises (`ValueError`, `ZeroDivisionError`, `RecursionError`) where
//! these return `Err(Raised)`; `_entity_segments` swallows every one of them.

use std::f64::consts::{PI, TAU};

/// What ezdxf raises (`ValueError`, `ZeroDivisionError`, ...) where the
/// importer catches it, or lets it end a block reference.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Raised;

pub type R<T> = Result<T, Raised>;

/// `ezdxf.math.Vec3` (x, y, z as f64).
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct P {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

pub const NULLVEC: P = P { x: 0.0, y: 0.0, z: 0.0 };
pub const X_AXIS: P = P { x: 1.0, y: 0.0, z: 0.0 };
pub const Y_AXIS: P = P { x: 0.0, y: 1.0, z: 0.0 };
pub const Z_AXIS: P = P { x: 0.0, y: 0.0, z: 1.0 };

/// `isclose(a, b, rel_tol, abs_tol)` of `acc/vector.pyx` (= `math.isclose`).
pub fn isclose(a: f64, b: f64, rel_tol: f64, abs_tol: f64) -> bool {
    let diff = (b - a).abs();
    diff <= (rel_tol * b).abs() || diff <= (rel_tol * a).abs() || diff <= abs_tol
}

/// `REL_TOL` / `ABS_TOL` of `constants.h`.
pub const REL_TOL: f64 = 1e-9;
pub const ABS_TOL: f64 = 1e-12;

/// Python's `a % b` for floats.
pub fn py_mod(a: f64, b: f64) -> f64 {
    let r = a % b;
    if r == 0.0 {
        0.0f64.copysign(b)
    } else if (r < 0.0) != (b < 0.0) {
        r + b
    } else {
        r
    }
}

/// `math.degrees`.
pub fn degrees(x: f64) -> f64 {
    x * (180.0 / PI)
}

/// `math.radians`.
pub fn radians(x: f64) -> f64 {
    x * (PI / 180.0)
}

impl P {
    pub const fn new(x: f64, y: f64, z: f64) -> P {
        P { x, y, z }
    }
    pub fn add(self, o: P) -> P {
        P::new(self.x + o.x, self.y + o.y, self.z + o.z)
    }
    pub fn sub(self, o: P) -> P {
        P::new(self.x - o.x, self.y - o.y, self.z - o.z)
    }
    pub fn mul(self, f: f64) -> P {
        P::new(self.x * f, self.y * f, self.z * f)
    }
    pub fn cross(self, o: P) -> P {
        P::new(self.y * o.z - self.z * o.y, self.z * o.x - self.x * o.z, self.x * o.y - self.y * o.x)
    }
    pub fn magnitude(self) -> f64 {
        (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }
    /// `normalize(length)`; a null vector divides by zero (Cython raises).
    pub fn normalize(self, length: f64) -> R<P> {
        let m = self.magnitude();
        if m == 0.0 {
            return Err(Raised);
        }
        let factor = length / m;
        Ok(P::new(self.x * factor, self.y * factor, self.z * factor))
    }
    pub fn isclose(self, o: P, rel_tol: f64, abs_tol: f64) -> bool {
        isclose(self.x, o.x, rel_tol, abs_tol) && isclose(self.y, o.y, rel_tol, abs_tol) && isclose(self.z, o.z, rel_tol, abs_tol)
    }
    /// `isclose` with the class defaults.
    pub fn close(self, o: P) -> bool {
        self.isclose(o, REL_TOL, ABS_TOL)
    }
    pub fn dist(self, o: P) -> f64 {
        let (dx, dy, dz) = (self.x - o.x, self.y - o.y, self.z - o.z);
        (dx * dx + dy * dy + dz * dz).sqrt()
    }
    pub fn lerp(self, o: P, f: f64) -> P {
        P::new(self.x + (o.x - self.x) * f, self.y + (o.y - self.y) * f, self.z + (o.z - self.z) * f)
    }
}

// ---- Matrix44 ----------------------------------------------------------------

/// `ezdxf.math.Matrix44`: 16 doubles, row vectors (translation in row 3).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Mat {
    pub m: [f64; 16],
}

impl Mat {
    pub fn identity() -> Mat {
        let mut m = [0.0; 16];
        m[0] = 1.0;
        m[5] = 1.0;
        m[10] = 1.0;
        m[15] = 1.0;
        Mat { m }
    }

    /// `Matrix44.ucs(ux, uy, uz)` with the origin at zero.
    pub fn ucs(ux: P, uy: P, uz: P) -> Mat {
        let mut r = Mat::identity();
        r.m[0] = ux.x;
        r.m[1] = ux.y;
        r.m[2] = ux.z;
        r.m[4] = uy.x;
        r.m[5] = uy.y;
        r.m[6] = uy.z;
        r.m[8] = uz.x;
        r.m[9] = uz.y;
        r.m[10] = uz.z;
        r
    }

    /// `Matrix44.axis_rotate(axis, angle)`.
    pub fn axis_rotate(axis: P, angle: f64) -> R<Mat> {
        let mut mat = Mat::identity();
        let (cos_a, sin_a) = (angle.cos(), angle.sin());
        let one_m_cos = 1.0 - cos_a;
        let a = axis.normalize(1.0)?;
        let (x, y, z) = (a.x, a.y, a.z);
        let m = &mut mat.m;
        m[0] = x * x * one_m_cos + cos_a;
        m[1] = y * x * one_m_cos + z * sin_a;
        m[2] = x * z * one_m_cos - y * sin_a;
        m[4] = x * y * one_m_cos - z * sin_a;
        m[5] = y * y * one_m_cos + cos_a;
        m[6] = y * z * one_m_cos + x * sin_a;
        m[8] = x * z * one_m_cos + y * sin_a;
        m[9] = y * z * one_m_cos - x * sin_a;
        m[10] = z * z * one_m_cos + cos_a;
        Ok(mat)
    }

    /// `self * other` (`__imul__`), terms summed left to right.
    pub fn mul(&self, other: &Mat) -> Mat {
        let (a, b) = (&self.m, &other.m);
        let mut r = [0.0; 16];
        for row in 0..4 {
            for col in 0..4 {
                r[row * 4 + col] = a[row * 4] * b[col] + a[row * 4 + 1] * b[4 + col] + a[row * 4 + 2] * b[8 + col] + a[row * 4 + 3] * b[12 + col];
            }
        }
        Mat { m: r }
    }

    /// `transform(vector)`: a point.
    pub fn transform(&self, v: P) -> P {
        let m = &self.m;
        P::new(
            v.x * m[0] + v.y * m[4] + v.z * m[8] + m[12],
            v.x * m[1] + v.y * m[5] + v.z * m[9] + m[13],
            v.x * m[2] + v.y * m[6] + v.z * m[10] + m[14],
        )
    }

    /// `transform_direction(vector)` (= `ocs_to_wcs`).
    pub fn direction(&self, v: P) -> P {
        let m = &self.m;
        P::new(v.x * m[0] + v.y * m[4] + v.z * m[8], v.x * m[1] + v.y * m[5] + v.z * m[9], v.x * m[2] + v.y * m[6] + v.z * m[10])
    }
}

// ---- OCS ---------------------------------------------------------------------

/// `ezdxf.math.OCS`.
#[derive(Clone, Copy, Debug)]
pub struct Ocs {
    pub transform: bool,
    pub matrix: Mat,
}

impl Ocs {
    pub fn new(extrusion: P) -> R<Ocs> {
        let az = extrusion.normalize(1.0)?;
        let transform = !az.close(Z_AXIS);
        let mut matrix = Mat::identity();
        if transform {
            let ax = if az.x.abs() < 1.0 / 64.0 && az.y.abs() < 1.0 / 64.0 { Y_AXIS.cross(az) } else { Z_AXIS.cross(az) };
            let ax = ax.normalize(1.0)?;
            let ay = az.cross(ax).normalize(1.0)?;
            matrix = Mat::ucs(ax, ay, az);
        }
        Ok(Ocs { transform, matrix })
    }

    pub fn uz(&self) -> P {
        if self.transform {
            P::new(self.matrix.m[8], self.matrix.m[9], self.matrix.m[10])
        } else {
            Z_AXIS
        }
    }

    pub fn ux(&self) -> P {
        if self.transform {
            P::new(self.matrix.m[0], self.matrix.m[1], self.matrix.m[2])
        } else {
            X_AXIS
        }
    }

    pub fn uy(&self) -> P {
        if self.transform {
            P::new(self.matrix.m[4], self.matrix.m[5], self.matrix.m[6])
        } else {
            Y_AXIS
        }
    }

    pub fn to_wcs(&self, p: P) -> P {
        if self.transform {
            self.matrix.direction(p)
        } else {
            p
        }
    }

    /// `ocs_from_wcs` (`ucs_direction_from_wcs`).
    pub fn from_wcs(&self, p: P) -> P {
        if self.transform {
            let m = &self.matrix.m;
            P::new(p.x * m[0] + p.y * m[1] + p.z * m[2], p.x * m[4] + p.y * m[5] + p.z * m[6], p.x * m[8] + p.y * m[9] + p.z * m[10])
        } else {
            p
        }
    }
}

/// `math.degrees` / `math.radians` constants of `transformtools`.
const DEG: f64 = 180.0 / PI;
const RADIANS: f64 = PI / 180.0;

/// `ezdxf.math.OCSTransform`: an entity's OCS carried through a matrix.
#[derive(Clone, Copy, Debug)]
pub struct OcsTransform {
    pub old: Ocs,
    pub new: Ocs,
    pub m: Mat,
    pub uniform: bool,
}

impl OcsTransform {
    /// `OCSTransform(extrusion, m)`.
    pub fn new(extrusion: P, m: &Mat) -> R<OcsTransform> {
        // transform_extrusion
        let ocs = Ocs::new(extrusion)?;
        let x_axis = m.direction(ocs.to_wcs(X_AXIS));
        let y_axis = m.direction(ocs.to_wcs(Y_AXIS));
        let msq = |v: P| v.x * v.x + v.y * v.y + v.z * v.z;
        let (a, b) = (msq(x_axis), msq(y_axis));
        // math.isclose(a, b, abs_tol=1e-9)
        let uniform = isclose(a, b, 1e-9, 1e-9);
        let new_extrusion = x_axis.cross(y_axis).normalize(1.0)?;
        Ok(OcsTransform { old: ocs, new: Ocs::new(new_extrusion)?, m: *m, uniform })
    }

    /// `OCSTransform.from_ocs(old, new, m)`.
    pub fn from_ocs(old: Ocs, new: Ocs, m: &Mat) -> OcsTransform {
        OcsTransform { old, new, m: *m, uniform: true }
    }

    pub fn new_extrusion(&self) -> P {
        self.new.uz()
    }

    /// `transform_length(length)` with the default reflection.
    pub fn transform_length(&self, length: P) -> f64 {
        self.m.direction(self.old.to_wcs(length)).magnitude() * 1.0
    }

    pub fn transform_vertex(&self, v: P) -> P {
        self.new.from_wcs(self.m.transform(self.old.to_wcs(v)))
    }

    pub fn transform_direction(&self, d: P) -> P {
        self.new.from_wcs(self.m.direction(self.old.to_wcs(d)))
    }

    pub fn transform_angle(&self, angle: f64) -> f64 {
        let d = self.transform_direction(P::new(angle.cos() * 1.0, angle.sin() * 1.0, 0.0));
        d.y.atan2(d.x)
    }

    pub fn transform_deg_angle(&self, angle: f64) -> f64 {
        self.transform_angle(angle * RADIANS) * DEG
    }

    pub fn transform_ccw_arc_angles(&self, start: f64, end: f64) -> (f64, f64) {
        let mut old_span = arc_angle_span_rad(start, end);
        let new_start = self.transform_angle(start);
        let new_end = self.transform_angle(end);
        let new_span = if isclose(old_span, PI, 1e-9, 0.0) {
            old_span = 1.0;
            let check = self.transform_angle(start + old_span);
            arc_angle_span_rad(new_start, check)
        } else if isclose(old_span, TAU, 1e-9, 0.0) {
            return (new_start, new_start + TAU);
        } else {
            arc_angle_span_rad(new_start, new_end)
        };
        if isclose(old_span, new_span, 1e-8, 0.0) {
            (new_start, new_end)
        } else {
            (new_end, new_start)
        }
    }

    pub fn transform_ccw_arc_angles_deg(&self, start: f64, end: f64) -> (f64, f64) {
        let (s, e) = self.transform_ccw_arc_angles(start * RADIANS, end * RADIANS);
        (s * DEG, e * DEG)
    }
}

/// `arc_angle_span_deg` (the C version).
pub fn arc_angle_span_deg(start: f64, end: f64) -> f64 {
    const DEG_ABS_TOL: f64 = 1e-13;
    if isclose(start, end, REL_TOL, DEG_ABS_TOL) {
        return 0.0;
    }
    let start = py_mod(start, 360.0);
    if isclose(start, py_mod(end, 360.0), REL_TOL, DEG_ABS_TOL) {
        return 360.0;
    }
    let mut end = end;
    if !isclose(end, 360.0, REL_TOL, DEG_ABS_TOL) {
        end = py_mod(end, 360.0);
    }
    if end < start {
        end += 360.0;
    }
    end - start
}

// ---- Bezier curves and ellipses -----------------------------------------------

pub type Curve = [P; 4];

/// `Bezier4P.reverse()` over a list, then the list reversed
/// (`reverse_bezier_curves`).
pub fn reverse_curves(curves: &[Curve]) -> Vec<Curve> {
    let mut out: Vec<Curve> = curves.iter().map(|c| [c[3], c[2], c[1], c[0]]).collect();
    out.reverse();
    out
}

/// `FastCubicCurve`: the curve relative to its first control point.
struct Fast {
    offset: P,
    p1: P,
    p2: P,
    p3: P,
}

impl Fast {
    fn new(c: &Curve) -> Fast {
        let o = c[0];
        Fast { offset: o, p1: c[1].sub(o), p2: c[2].sub(o), p3: c[3].sub(o) }
    }

    fn point(&self, t: f64) -> P {
        let t2 = t * t;
        let one_minus_t = 1.0 - t;
        let b = 3.0 * one_minus_t * one_minus_t * t;
        let c = 3.0 * one_minus_t * t2;
        let d = t2 * t;
        let mut r = NULLVEC;
        r.x += self.p1.x * b;
        r.y += self.p1.y * b;
        r.z += self.p1.z * b;
        r.x += self.p2.x * c;
        r.y += self.p2.y * c;
        r.z += self.p2.z * c;
        r.x += self.p3.x * d;
        r.y += self.p3.y * d;
        r.z += self.p3.z * d;
        r.x += self.offset.x;
        r.y += self.offset.y;
        r.z += self.offset.z;
        r
    }
}

const RECURSION_LIMIT: i64 = 1000;

struct Flat<'a> {
    curve: &'a Fast,
    distance: f64,
    points: Vec<P>,
    level: i64,
    error: bool,
}

impl Flat<'_> {
    fn flatten(&mut self, start: P, end: P, start_t: f64, end_t: f64) {
        if self.level > RECURSION_LIMIT {
            self.error = true;
            return;
        }
        self.level += 1;
        let mid_t = (start_t + end_t) * 0.5;
        let mid = self.curve.point(mid_t);
        let d = mid.dist(start.lerp(end, 0.5));
        if d < self.distance {
            self.points.push(end);
        } else {
            self.flatten(start, mid, start_t, mid_t);
            self.flatten(mid, end, mid_t, end_t);
        }
        self.level -= 1;
    }
}

/// `Bezier4P(curve).flattening(distance, segments)`.
pub fn flatten_bezier4(curve: &Curve, distance: f64, segments: i64) -> R<Vec<P>> {
    // NaN or infinite control points never satisfy the flatness test and
    // ezdxf's recursion would run for ever; refuse them as it ends up doing.
    if curve.iter().any(|p| !(p.x.is_finite() && p.y.is_finite() && p.z.is_finite())) {
        return Err(Raised);
    }
    let fast = Fast::new(curve);
    let dt = 1.0 / segments as f64;
    let mut t0 = 0.0;
    let mut f = Flat { curve: &fast, distance, points: vec![curve[0]], level: 0, error: false };
    let mut start = curve[0];
    while t0 < 1.0 {
        let t1 = t0 + dt;
        let (end, t1) = if isclose(t1, 1.0, REL_TOL, ABS_TOL) { (curve[3], 1.0) } else { (fast.point(t1), t1) };
        f.level = 0;
        f.error = false;
        f.flatten(start, end, t0, t1);
        if f.error {
            return Err(Raised);
        }
        t0 = t1;
        start = end;
    }
    Ok(f.points)
}

/// `ezdxf.math.arc_angle_span_rad` (the C version).
pub fn arc_angle_span_rad(start: f64, end: f64) -> f64 {
    const RAD_ABS_TOL: f64 = 1e-15;
    if isclose(start, end, REL_TOL, RAD_ABS_TOL) {
        return 0.0;
    }
    let start = py_mod(start, TAU);
    if isclose(start, py_mod(end, TAU), REL_TOL, RAD_ABS_TOL) {
        return TAU;
    }
    let mut end = end;
    if !isclose(end, TAU, REL_TOL, RAD_ABS_TOL) {
        end = py_mod(end, TAU);
    }
    if end < start {
        end += TAU;
    }
    end - start
}

fn normalize_rad_angle(a: f64) -> f64 {
    let r = a % TAU;
    if r < 0.0 {
        r + TAU
    } else {
        r
    }
}

/// `cubic_bezier_arc_parameters(start, end, segments)`: unit-circle curves.
fn arc_parameters(start_angle: f64, end_angle: f64, segments: i64) -> R<Vec<Curve>> {
    if segments < 1 {
        return Err(Raised);
    }
    let delta = end_angle - start_angle;
    if !(delta > 0.0) {
        return Err(Raised);
    }
    let mut arc_count = (delta / PI * 2.0).ceil() as i64;
    if segments > arc_count {
        arc_count = segments;
    }
    let segment_angle = delta / arc_count as f64;
    let tangent_length = (4.0 / 3.0) * (segment_angle / 4.0).tan();
    let mut angle = start_angle;
    let from_angle = |a: f64| P::new(a.cos() * 1.0, a.sin() * 1.0, 0.0);
    let mut end_point = from_angle(angle);
    let mut out = Vec::new();
    for _ in 0..arc_count {
        let start_point = end_point;
        angle += segment_angle;
        end_point = from_angle(angle);
        let cp1 = P::new(start_point.x - start_point.y * tangent_length, start_point.y + start_point.x * tangent_length, 0.0);
        let cp2 = P::new(end_point.x + end_point.y * tangent_length, end_point.y - end_point.x * tangent_length, 0.0);
        out.push([start_point, cp1, cp2, end_point]);
    }
    Ok(out)
}

/// `ezdxf.math.ConstructionEllipse`.
#[derive(Clone, Copy, Debug)]
pub struct Ellipse {
    pub center: P,
    pub major_axis: P,
    pub minor_axis: P,
    pub extrusion: P,
    pub ratio: f64,
    pub start_param: f64,
    pub end_param: f64,
}

/// `ellipse.minor_axis(major_axis, extrusion, ratio)`.
fn minor_axis(major_axis: P, extrusion: P, ratio: f64) -> R<P> {
    extrusion.cross(major_axis).normalize(major_axis.magnitude() * ratio)
}

/// `ellipse.vertex(param, major, minor, center, ratio)`.
fn vertex(param: f64, major_axis: P, minor_axis: P, center: P, ratio: f64) -> R<P> {
    let x_axis = major_axis.normalize(1.0)?;
    let y_axis = minor_axis.normalize(1.0)?;
    let radius_x = major_axis.magnitude();
    let radius_y = radius_x * ratio;
    let x = x_axis.mul(param.cos() * radius_x);
    let y = y_axis.mul(param.sin() * radius_y);
    Ok(center.add(x).add(y))
}

/// `math.isclose(a, b, rel_tol=1e-9, abs_tol=abs)`.
fn py_isclose(a: f64, b: f64, abs_tol: f64) -> bool {
    isclose(a, b, 1e-9, abs_tol)
}

/// `rytz_axis_construction(d1, d2)`: (major axis, minor axis, ratio).
fn rytz_axis_construction(d1: P, d2: P) -> R<(P, P, f64)> {
    let q = d1;
    let p1 = if py_isclose(d1.z, 0.0, 1e-9) && py_isclose(d2.z, 0.0, 1e-9) {
        // Vec3.orthogonal(ccw=False)
        P::new(d2.y, -d2.x, d2.z)
    } else {
        let extrusion = d1.cross(d2);
        extrusion.cross(d2).normalize(d2.magnitude())?
    };
    let d = p1.lerp(q, 0.5);
    let radius = d.magnitude();
    let radius_vector = q.sub(p1).normalize(radius)?;
    let a = d.sub(radius_vector);
    let b = d.add(radius_vector);
    if a.close(NULLVEC) || b.close(NULLVEC) {
        return Err(Raised);
    }
    let major_len = a.sub(q).magnitude();
    let minor_len = b.sub(q).magnitude();
    if py_isclose(major_len, 0.0, 0.0) || py_isclose(minor_len, 0.0, 0.0) {
        return Err(Raised);
    }
    let ratio = minor_len / major_len;
    Ok((b.normalize(major_len)?, a.normalize(minor_len)?, ratio))
}

fn mid_param(start: f64, end: f64) -> f64 {
    let end = if end < start { end + TAU } else { end };
    (start + end) / 2.0
}

impl Ellipse {
    /// `ConstructionEllipse(center, major_axis, extrusion, ratio, start, end)`.
    pub fn new(center: P, major_axis: P, extrusion: P, ratio: f64, start_param: f64, end_param: f64) -> R<Ellipse> {
        if major_axis.close(NULLVEC) {
            return Err(Raised);
        }
        let minor_axis = minor_axis(major_axis, extrusion, ratio)?;
        Ok(Ellipse { center, major_axis, minor_axis, extrusion, ratio, start_param, end_param })
    }

    /// `ConstructionEllipse.from_arc(center, radius, extrusion, start, end)`
    /// (angles in degrees, counter-clockwise).
    pub fn from_arc(center: P, radius: f64, extrusion: P, start_angle: f64, end_angle: f64) -> R<Ellipse> {
        let radius = radius.abs();
        if NULLVEC.close(extrusion) {
            return Err(Raised);
        }
        let ocs = Ocs::new(extrusion)?;
        let center = ocs.to_wcs(center);
        let major = ocs.to_wcs(P::new(radius, 0.0, 0.0));
        Ellipse::new(center, major, extrusion, 1.0, radians(start_angle), radians(end_angle))
    }

    pub fn param_span(&self) -> f64 {
        arc_angle_span_rad(self.start_param, self.end_param)
    }

    /// `vertex(start_param, ...)`: the start point.
    pub fn start_point(&self) -> R<P> {
        vertex(self.start_param, self.major_axis, self.minor_axis, self.center, self.ratio)
    }

    /// `ConstructionEllipse.transform(m)`, returning the moved ellipse.
    pub fn transform(&self, m: &Mat) -> R<Ellipse> {
        let new_center = m.transform(self.center);
        let (old_start_param, old_end_param) = (self.start_param, self.end_param);
        let (mut start_param, mut end_param) = (old_start_param, old_end_param);
        let old_minor_axis = minor_axis(self.major_axis, self.extrusion, self.ratio)?;
        let mut new_major_axis = m.direction(self.major_axis);
        let mut new_minor_axis = m.direction(old_minor_axis);
        let dot_product = {
            let (a, b) = (new_major_axis.normalize(1.0)?, new_minor_axis.normalize(1.0)?);
            a.x * b.x + a.y * b.y + a.z * b.z
        };
        let new_extrusion;
        let mut new_ratio;
        let adjust_params;
        if dot_product.abs() > 1e-6 {
            let (maj, min, ratio) = rytz_axis_construction(new_major_axis, new_minor_axis)?;
            new_major_axis = maj;
            new_minor_axis = min;
            new_ratio = ratio;
            new_extrusion = new_major_axis.cross(new_minor_axis).normalize(1.0)?;
            adjust_params = true;
        } else {
            let mag = new_major_axis.magnitude();
            if mag == 0.0 {
                return Err(Raised);
            }
            new_ratio = new_minor_axis.magnitude() / mag;
            new_extrusion = new_major_axis.cross(new_minor_axis).normalize(1.0)?;
            new_minor_axis = minor_axis(new_major_axis, new_extrusion, new_ratio)?;
            adjust_params = false;
        }
        if adjust_params && !py_isclose(start_param, end_param, 1e-9) {
            let x_axis = new_major_axis.normalize(1.0)?;
            let y_axis = new_minor_axis.normalize(1.0)?;
            let old_param_span = py_mod(end_param - start_param, TAU);
            let dotp = |a: P, b: P| a.x * b.x + a.y * b.y + a.z * b.z;
            let param = |vec: P| -> R<f64> {
                if new_ratio == 0.0 {
                    return Err(Raised);
                }
                let dy = dotp(y_axis, vec) / new_ratio;
                let dx = dotp(x_axis, vec);
                Ok(py_mod(dy.atan2(dx), TAU))
            };
            let start_point = m.transform(vertex(start_param, self.major_axis, old_minor_axis, self.center, self.ratio)?);
            let end_point = m.transform(vertex(end_param, self.major_axis, old_minor_axis, self.center, self.ratio)?);
            start_param = param(start_point.sub(new_center))?;
            end_param = param(end_point.sub(new_center))?;
            if !py_isclose(old_param_span, PI, 1e-9) {
                let new_param_span = py_mod(end_param - start_param, TAU);
                if !py_isclose(old_param_span, new_param_span, 1e-9) {
                    std::mem::swap(&mut start_param, &mut end_param);
                }
            } else {
                let old_chk = m.transform(vertex(mid_param(old_start_param, old_end_param), self.major_axis, old_minor_axis, self.center, self.ratio)?);
                let new_chk = vertex(mid_param(start_param, end_param), new_major_axis, new_minor_axis, new_center, new_ratio)?;
                if !old_chk.isclose(new_chk, REL_TOL, 1e-9) {
                    std::mem::swap(&mut start_param, &mut end_param);
                }
            }
        }
        if new_ratio > 1.0 {
            new_major_axis = minor_axis(new_major_axis, new_extrusion, new_ratio)?;
            new_ratio = 1.0 / new_ratio;
            new_minor_axis = minor_axis(new_major_axis, new_extrusion, new_ratio)?;
            if !(py_isclose(start_param, 0.0, 0.0) && py_isclose(end_param, TAU, 0.0)) {
                start_param -= PI / 2.0;
                end_param -= PI / 2.0;
            }
        }
        start_param = py_mod(start_param, TAU);
        end_param = py_mod(end_param, TAU);
        if py_isclose(start_param, end_param, 0.0) {
            start_param = 0.0;
            end_param = TAU;
        }
        Ok(Ellipse {
            center: new_center,
            major_axis: new_major_axis,
            minor_axis: new_minor_axis,
            extrusion: new_extrusion,
            ratio: new_ratio,
            start_param,
            end_param,
        })
    }
}

/// `cubic_bezier_from_ellipse(ellipse, segments)`.
pub fn bezier_from_ellipse(e: &Ellipse, segments: i64) -> R<Vec<Curve>> {
    let param_span = e.param_span();
    if param_span.abs() < 1e-9 {
        return Ok(Vec::new());
    }
    let start_angle = normalize_rad_angle(e.start_param);
    let mut end_angle = start_angle + param_span;
    while start_angle > end_angle {
        end_angle += TAU;
    }
    let mut out = Vec::new();
    for cp in arc_parameters(start_angle, end_angle, segments)? {
        let mut c = [NULLVEC; 4];
        for i in 0..4 {
            let a = e.major_axis.mul(cp[i].x);
            let b = e.minor_axis.mul(cp[i].y);
            c[i] = P::new(e.center.x + a.x + b.x, e.center.y + a.y + b.y, e.center.z + a.z + b.z);
        }
        out.push(c);
    }
    Ok(out)
}

// ---- Path --------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
pub enum Cmd {
    Line(P),
    /// end, ctrl1, ctrl2
    Curve4(P, P, P),
}

/// `ezdxf.path.Path`, lines and cubic curves only.
#[derive(Clone, Debug)]
pub struct Path {
    pub start: P,
    pub cmds: Vec<Cmd>,
}

impl Path {
    pub fn new(start: P) -> Path {
        Path { start, cmds: Vec::new() }
    }

    /// `path.end`: the last vertex.
    pub fn end(&self) -> P {
        match self.cmds.last() {
            None => self.start,
            Some(Cmd::Line(p)) => *p,
            Some(Cmd::Curve4(e, _, _)) => *e,
        }
    }

    pub fn line_to(&mut self, p: P) {
        self.cmds.push(Cmd::Line(p));
    }

    pub fn curve4_to(&mut self, end: P, c1: P, c2: P) {
        self.cmds.push(Cmd::Curve4(end, c1, c2));
    }

    /// `Path.is_closed`: more than one vertex and the first is close to the last.
    pub fn is_closed(&self) -> bool {
        !self.cmds.is_empty() && self.start.close(self.end())
    }

    /// `Path.close()`.
    pub fn close(&mut self) {
        if !self.is_closed() {
            let s = self.start;
            self.line_to(s);
        }
    }

    /// `path.to_wcs(ocs, elevation)`: every vertex (control points too)
    /// with z replaced by the elevation, then OCS to WCS.
    pub fn to_wcs(&mut self, ocs: &Ocs, elevation: f64) {
        let f = |v: P| ocs.to_wcs(P::new(v.x, v.y, elevation));
        self.start = f(self.start);
        for c in &mut self.cmds {
            *c = match *c {
                Cmd::Line(p) => Cmd::Line(f(p)),
                Cmd::Curve4(e, a, b) => Cmd::Curve4(f(e), f(a), f(b)),
            };
        }
    }

    /// `Path.flattening(distance, segments)`: nothing for a path without
    /// commands.
    pub fn flattening(&self, distance: f64, segments: i64) -> R<Vec<P>> {
        if self.cmds.is_empty() {
            return Ok(Vec::new());
        }
        let mut out = vec![self.start];
        let mut start = self.start;
        for c in &self.cmds {
            match *c {
                Cmd::Line(p) => {
                    out.push(p);
                    start = p;
                }
                Cmd::Curve4(end, c1, c2) => {
                    if distance == 0.0 {
                        return Err(Raised);
                    }
                    let pts = flatten_bezier4(&[start, c1, c2, end], distance, segments)?;
                    out.extend(pts.into_iter().skip(1));
                    start = end;
                }
            }
        }
        Ok(out)
    }
}

/// `tools.add_bezier4p(path, curves)`.
pub fn add_bezier4p(path: &mut Path, curves: Vec<Curve>) {
    if curves.is_empty() {
        return;
    }
    let last_end = curves[curves.len() - 1][3];
    let curves = if path.end().close(last_end) { reverse_curves(&curves) } else { curves };
    for c in curves {
        let (start, ctrl1, ctrl2, end) = (c[0], c[1], c[2], c[3]);
        if !start.close(path.end()) {
            path.line_to(start);
        }
        if start.isclose(ctrl1, 1e-15, 0.0) && end.isclose(ctrl2, 1e-15, 0.0) {
            path.line_to(end);
        } else {
            path.curve4_to(end, ctrl1, ctrl2);
        }
    }
}

/// `tools.add_ellipse(path, ellipse, segments=1, reset=True)`.
pub fn add_ellipse(path: &mut Path, e: &Ellipse, segments: i64) -> R<()> {
    if e.param_span().abs() < 1e-9 {
        return Ok(());
    }
    if path.cmds.is_empty() {
        path.start = e.start_point()?;
    }
    add_bezier4p(path, bezier_from_ellipse(e, segments)?);
    Ok(())
}

// ---- 2D polylines --------------------------------------------------------------

const IS_CLOSE_TOL: f64 = 1e-10;

/// `bulge_to_arc(p1, p2, bulge)` on the xy parts: (center, start angle,
/// end angle, radius).
pub fn bulge_to_arc(p1: P, p2: P, bulge: f64) -> ((f64, f64), f64, f64, f64) {
    let angle = |a: (f64, f64), b: (f64, f64)| (b.1 - a.1).atan2(b.0 - a.0);
    let (s, e) = ((p1.x, p1.y), (p2.x, p2.y));
    let dist = (s.0 - e.0).hypot(s.1 - e.1);
    let r = dist * (1.0 + (bulge * bulge)) / 4.0 / bulge;
    let a = angle(s, e) + (PI / 2.0 - bulge.atan() * 2.0);
    let c = (s.0 + a.cos() * r, s.1 + a.sin() * r);
    if bulge < 0.0 {
        (c, angle(c, e), angle(c, s), r.abs())
    } else {
        (c, angle(c, s), angle(c, e), r.abs())
    }
}

/// `tools.add_2d_polyline(path, points, close, ocs, elevation)` for an empty
/// path; `points` are (x, y, bulge).
pub fn add_2d_polyline(path: &mut Path, points: &[(f64, f64, f64)], close: bool, ocs: &Ocs, elevation: f64) -> R<()> {
    let bulge_to = |path: &mut Path, p1: P, p2: P, bulge: f64| -> R<()> {
        if p1.isclose(p2, IS_CLOSE_TOL, 0.0) {
            return Ok(());
        }
        let (center, start_angle, end_angle, radius) = bulge_to_arc(p1, p2, bulge);
        let start_angle = py_mod(start_angle, TAU);
        let mut end_angle = py_mod(end_angle, TAU);
        if start_angle > end_angle {
            end_angle += TAU;
        }
        // np.linspace(start, end, 2) is [start, end] exactly.
        let e = Ellipse::from_arc(P::new(center.0, center.1, 0.0), radius, Z_AXIS, degrees(start_angle), degrees(end_angle))?;
        let mut curves = bezier_from_ellipse(&e, 1)?;
        if curves.is_empty() {
            return Err(Raised);
        }
        if curves[0][0].isclose(p2, IS_CLOSE_TOL, 0.0) {
            curves = reverse_curves(&curves);
        }
        add_bezier4p(path, curves);
        Ok(())
    };
    let mut prev: Option<P> = None;
    let mut prev_bulge = 0.0;
    for &(x, y, b) in points {
        let bulge = if b.abs() < 1e-6 { 0.0 } else { b };
        let point = P::new(x, y, 0.0);
        let Some(pp) = prev else {
            path.start = point;
            prev = Some(point);
            prev_bulge = bulge;
            continue;
        };
        if prev_bulge != 0.0 {
            bulge_to(path, pp, point, prev_bulge)?;
        } else {
            path.line_to(point);
        }
        prev = Some(point);
        prev_bulge = bulge;
    }
    if close && !path.start.isclose(path.end(), IS_CLOSE_TOL, 0.0) {
        if prev_bulge != 0.0 {
            let (e, s) = (path.end(), path.start);
            bulge_to(path, e, s, prev_bulge)?;
        } else {
            let s = path.start;
            path.line_to(s);
        }
    }
    if ocs.transform || elevation != 0.0 {
        path.to_wcs(ocs, elevation);
    }
    Ok(())
}

/// `from_vertices(vertices, close)`: a polyline through 3D points,
/// consecutive duplicates dropped.
pub fn from_vertices(vertices: &[P], close: bool) -> Path {
    if vertices.len() < 2 {
        return Path::new(NULLVEC);
    }
    let mut path = Path::new(vertices[0]);
    for &v in &vertices[1..] {
        if !path.end().close(v) {
            path.line_to(v);
        }
    }
    if close {
        path.close();
    }
    path
}
