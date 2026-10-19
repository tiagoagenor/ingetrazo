// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! `core.bevel_profiles` line for line: the shaped bevel presets (steps,
//! supports, cornice, crown) and hand-drawn outlines, sampled into points of
//! the unit square. Python floats are f64 and the maths is `math.*`, so the
//! same operations run here in the same order.
//!
//! Not ported: `controls`/`valid` receiving something that is not a list of
//! `(number, number, truthy)` triples (Python's `TypeError` path); the typed
//! API makes it unrepresentable.

use ingetrazo_math::pyfloat::py_sum;

/// A control point `(x, y, smooth)`.
pub type Control = (f64, f64, bool);

/// What `shape` can be in Python: a preset name, or a list of controls.
#[derive(Clone, Copy, Debug)]
pub enum Shape<'a> {
    Name(&'a str),
    Controls(&'a [Control]),
}

/// `sample` was given a name that is no preset (Python's `ValueError`).
#[derive(Debug, PartialEq, Eq)]
pub struct UnknownShape;

/// Presets offered beside the superellipse, in menu order.
pub const PRESETS: [&str; 4] = ["steps", "supports", "cornice", "crown"];

/// Presets that read differently from each side (`ASYMMETRIC`).
pub fn is_asymmetric(name: &str) -> bool {
    name == "cornice" || name == "crown"
}

const CORNICE: [Control; 13] = [
    (1.0, 0.0, false), (1.0, 0.125, false), (0.92, 0.16, true),
    (0.875, 0.25, false), (0.8, 0.25, false), (0.733, 0.433, true),
    (0.582, 0.522, true), (0.4, 0.6, true), (0.289, 0.727, true),
    (0.25, 0.925, false), (0.175, 0.925, false), (0.175, 1.0, false),
    (0.0, 1.0, false),
];
const CROWN: [Control; 11] = [
    (1.0, 0.0, false), (1.0, 0.25, false), (0.75, 0.25, false),
    (0.75, 0.325, false), (0.925, 0.4, true), (0.975, 0.5, true),
    (0.94, 0.65, true), (0.85, 0.75, true), (0.75, 0.875, true),
    (0.7, 1.0, false), (0.0, 1.0, false),
];

/// `max(1, (int(segments) + 1) // 2)`.
pub fn steps_count(segments: i64) -> i64 {
    std::cmp::max(1, (segments + 1).div_euclid(2))
}

fn steps(segments: i64) -> Vec<(f64, f64)> {
    let k = steps_count(segments);
    let kf = k as f64;
    let mut pts = vec![(1.0, 0.0)];
    for i in 0..k {
        let x = 1.0 - (i + 1) as f64 / kf;
        pts.push((x, i as f64 / kf));
        pts.push((x, (i + 1) as f64 / kf));
    }
    let last = pts.len() - 1;
    pts[last] = (0.0, 1.0);
    pts
}

fn supports(segments: i64) -> Vec<(f64, f64)> {
    let arc = std::cmp::max(1, segments - 2);
    let mut pts = vec![(1.0, 0.0)];
    for k in 0..=arc {
        let a = std::f64::consts::PI / 2.0 * k as f64 / arc as f64;
        pts.push((0.5 + 0.5 * a.cos(), 0.5 + 0.5 * a.sin()));
    }
    pts.push((0.0, 1.0));
    pts
}

fn catmull(p0: (f64, f64), p1: (f64, f64), p2: (f64, f64), p3: (f64, f64), t: f64) -> (f64, f64) {
    let t2 = t * t;
    let t3 = t * t * t;
    let one = |a: f64, b: f64, c: f64, d: f64| -> f64 {
        0.5 * (2.0 * b + (-a + c) * t + (2.0 * a - 5.0 * b + 4.0 * c - d) * t2
            + (-a + 3.0 * b - 3.0 * c + d) * t3)
    };
    (one(p0.0, p1.0, p2.0, p3.0), one(p0.1, p1.1, p2.1, p3.1))
}

/// CPython 3.12's `math.hypot(x, y)`: its own `vector_norm`, not the C
/// library's `hypot`, so a 1-ulp difference cannot tip a `round()`.
pub fn py_hypot2(x: f64, y: f64) -> f64 {
    let (ax, ay) = (x.abs(), y.abs());
    if ax.is_infinite() || ay.is_infinite() {
        return f64::INFINITY;
    }
    if x.is_nan() || y.is_nan() {
        return f64::NAN;
    }
    let max = if ay > ax { ay } else { ax };
    let mut vec = [ax, ay];
    vector_norm(&mut vec, max)
}

/// The binary exponent `e` of C's `frexp` (x = m * 2^e, 0.5 <= m < 1).
fn frexp_exp(x: f64) -> i32 {
    let e = ((x.to_bits() >> 52) & 0x7ff) as i32;
    if e == 0 {
        // Subnormal: scale it into the normal range first.
        return frexp_exp(x * (2.0f64).powi(54)) - 54;
    }
    e - 1022
}

/// `ldexp(1.0, e)` for -1022 <= e <= 1023.
fn ldexp1(e: i32) -> f64 {
    f64::from_bits(((e + 1023) as u64) << 52)
}

fn vector_norm(vec: &mut [f64; 2], max: f64) -> f64 {
    if max == 0.0 {
        return max;
    }
    let max_e = frexp_exp(max);
    if max_e < -1023 {
        for v in vec.iter_mut() {
            *v /= f64::MIN_POSITIVE;
        }
        return f64::MIN_POSITIVE * vector_norm(vec, max / f64::MIN_POSITIVE);
    }
    let scale = ldexp1(-max_e);
    let mut csum = 1.0f64;
    let mut frac1 = 0.0f64;
    let mut frac2 = 0.0f64;
    // CPython's dl_mul calls C's fma(): one correctly rounded fused op, the
    // exact error term of the product (not a contraction of Rust code).
    let dl_mul = |x: f64, y: f64| -> (f64, f64) {
        let z = x * y;
        (z, x.mul_add(y, -z))
    };
    let dl_fast_sum = |a: f64, b: f64| -> (f64, f64) {
        let hi = a + b;
        (hi, (a - hi) + b)
    };
    for v in vec.iter() {
        let x = *v * scale;
        let pr = dl_mul(x, x);
        let sm = dl_fast_sum(csum, pr.0);
        csum = sm.0;
        frac1 += pr.1;
        frac2 += sm.1;
    }
    let mut h = (csum - 1.0 + (frac1 + frac2)).sqrt();
    let pr = dl_mul(-h, h);
    let sm = dl_fast_sum(csum, pr.0);
    csum = sm.0;
    frac1 += pr.1;
    frac2 += sm.1;
    let x = csum - 1.0 + (frac1 + frac2);
    h += x / (2.0 * h);
    h / scale
}

fn from_controls(controls: &[Control], segments: i64) -> Vec<(f64, f64)> {
    let pts: Vec<(f64, f64)> = controls.iter().map(|&(x, y, _)| (x, y)).collect();
    let smooth: Vec<bool> = controls.iter().map(|&(_, _, s)| s).collect();
    let npieces = pts.len() - 1;
    let curved: Vec<usize> = (0..npieces).filter(|&i| smooth[i] || smooth[i + 1]).collect();
    let spare = std::cmp::max(0, segments - (npieces - curved.len()) as i64);

    let length = |i: usize| -> f64 {
        let ((x0, y0), (x1, y1)) = (pts[i], pts[i + 1]);
        py_hypot2(x1 - x0, y1 - y0)
    };
    let mut total = py_sum(curved.iter().map(|&i| length(i)), true);
    if total == 0.0 {
        total = 1.0;
    }
    let mut out = vec![pts[0]];
    for i in 0..npieces {
        if !curved.contains(&i) {
            out.push(pts[i + 1]);
            continue;
        }
        let k = std::cmp::max(1, (spare as f64 * length(i) / total).round_ties_even() as i64);
        // A sharp end is its own neighbour, so the curve leaves a corner
        // straight toward the next point.
        let p0 = if i > 0 && smooth[i] { pts[i - 1] } else { pts[i] };
        let p3 = if i + 2 < pts.len() && smooth[i + 1] { pts[i + 2] } else { pts[i + 1] };
        for j in 1..=k {
            out.push(catmull(p0, pts[i], pts[i + 1], p3, j as f64 / k as f64));
        }
    }
    out
}

/// Whether `controls` is a drawable outline: two points or more, from
/// (1, 0) to (0, 1), all inside the unit square.
pub fn valid(controls: &[Control]) -> bool {
    let pts = controls;
    if pts.len() < 2 {
        return false;
    }
    if (pts[0].0 - 1.0).abs() > 1e-9 || pts[0].1.abs() > 1e-9 {
        return false;
    }
    let l = pts.len() - 1;
    if pts[l].0.abs() > 1e-9 || (pts[l].1 - 1.0).abs() > 1e-9 {
        return false;
    }
    pts.iter()
        .all(|&(x, y, _)| [x, y].iter().all(|&c| -1e-9 <= c && c <= 1.0 + 1e-9))
}

fn quarter_round() -> Vec<Control> {
    let mut pts = vec![(1.0, 0.0, false)];
    for k in 1..=3 {
        let a = std::f64::consts::PI / 8.0 * k as f64;
        pts.push((a.cos(), a.sin(), true));
    }
    pts.push((0.0, 1.0, false));
    pts
}

/// `controls(shape, segments)`: `shape` as editable control points.
pub fn controls(shape: Shape, segments: i64) -> Vec<Control> {
    let name = match shape {
        Shape::Controls(c) if valid(c) => return c.to_vec(),
        Shape::Controls(_) => return quarter_round(),
        Shape::Name(name) => name,
    };
    match name {
        "cornice" => CORNICE.to_vec(),
        "crown" => CROWN.to_vec(),
        "steps" => steps(segments).into_iter().map(|(x, y)| (x, y, false)).collect(),
        "supports" => {
            let mut pts = vec![(1.0, 0.0, false), (1.0, 0.5, false)];
            for k in 1..=3 {
                let a = std::f64::consts::PI / 8.0 * k as f64;
                pts.push((0.5 + 0.5 * a.cos(), 0.5 + 0.5 * a.sin(), true));
            }
            pts.push((0.5, 1.0, false));
            pts.push((0.0, 1.0, false));
            pts
        }
        _ => quarter_round(),
    }
}

/// `sample(shape, segments)`: the points of `shape` from (1, 0) to (0, 1).
pub fn sample(shape: Shape, segments: i64) -> Result<Vec<(f64, f64)>, UnknownShape> {
    let segments = std::cmp::max(segments, 1);
    match shape {
        Shape::Controls(_) => Ok(from_controls(&controls(shape, segments), segments)),
        Shape::Name("steps") => Ok(steps(segments)),
        Shape::Name("supports") => Ok(supports(segments)),
        Shape::Name("cornice") => Ok(from_controls(&CORNICE, segments)),
        Shape::Name("crown") => Ok(from_controls(&CROWN, segments)),
        Shape::Name(_) => Err(UnknownShape),
    }
}

/// The same outline read from the other face.
pub fn mirrored(points: &[(f64, f64)]) -> Vec<(f64, f64)> {
    points.iter().rev().map(|&(x, y)| (y, x)).collect()
}
