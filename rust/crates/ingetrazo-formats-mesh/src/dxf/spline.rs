// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! SPLINE to a `Path`: `ezdxf.path.make_path(spline)`, for the case CAD
//! writes almost always: control points, cubic, non-rational, clamped, which
//! ezdxf decomposes exactly into cubic Beziers (`BSpline.bezier_decomposition`).
//!
//! Not ported (they return `Err`, so the spline imports as nothing where
//! ezdxf approximates it): rational splines, other degrees, unclamped
//! (periodic) knot vectors and splines that only carry fit points
//! (`cubic_bezier_approximation`, `fit_points_to_cad_cv`).

use super::entities::SplineData;
use super::geom::{add_bezier4p, isclose, Curve, Path, Raised, NULLVEC, P, R};

/// `round_knots(knots, tolerance)`.
fn round_knots(knots: &[f64], tolerance: f64) -> Vec<f64> {
    if !(tolerance > 0.0) {
        return knots.to_vec();
    }
    let ndigits = -(tolerance.log10().trunc() as i64);
    if ndigits <= 0 {
        return knots.to_vec();
    }
    knots
        .iter()
        .map(|&k| {
            if ndigits > 300 || !k.is_finite() {
                k
            } else {
                format!("{:.*}", ndigits as usize, k).parse::<f64>().unwrap_or(k)
            }
        })
        .collect()
}

/// `open_uniform_knot_vector(count, order, normalize=True)`.
fn open_uniform_knots(count: usize, order: usize) -> Vec<f64> {
    let k = count - order;
    let max_value = (count - order + 1) as f64;
    let mut knots = vec![0.0; order];
    knots.extend((0..k).map(|v| (1.0 + v as f64) / max_value));
    knots.extend(vec![1.0; order]);
    knots
}

/// `BSpline.bezier_decomposition()`.
fn bezier_decomposition(cps: &[P], knots: &[f64], p: usize) -> R<Vec<[P; 4]>> {
    let n = cps.len() - 1;
    let mut alphas = vec![0.0f64; knots.len()];
    let m = n + p + 1;
    let mut a = p;
    let mut b = p + 1;
    let mut bezier_points: Vec<P> = cps[0..p + 1].to_vec();
    let mut out = Vec::new();
    while b < m {
        let mut next = vec![NULLVEC; p + 1];
        let i = b;
        while b < m && isclose(knots[b + 1], knots[b], 1e-9, 0.0) {
            b += 1;
        }
        let mult = b - i + 1;
        if mult < p {
            let numer = knots[b] - knots[a];
            let mut j = p;
            while j > mult {
                let d = knots[a + j] - knots[a];
                if d == 0.0 {
                    return Err(Raised);
                }
                alphas[j - mult - 1] = numer / d;
                j -= 1;
            }
            let r = p - mult;
            for j in 1..=r {
                let save = r - j;
                let s = mult + j;
                let mut k = p;
                while k >= s {
                    let alpha = alphas[k - s];
                    bezier_points[k] = bezier_points[k].mul(alpha).add(bezier_points[k - 1].mul(1.0 - alpha));
                    k -= 1;
                }
                if b < m {
                    next[save] = bezier_points[p];
                }
            }
        }
        if bezier_points.len() != 4 {
            return Err(Raised);
        }
        out.push([bezier_points[0], bezier_points[1], bezier_points[2], bezier_points[3]]);
        if b < m {
            for i in (p - mult)..=p {
                next[i] = cps[b - p + i];
            }
            a = b;
            b += 1;
            bezier_points = next;
        }
    }
    Ok(out)
}

pub fn make_path(s: &SplineData) -> R<Path> {
    if s.control_points.is_empty() {
        return Err(Raised);
    }
    let count = s.control_points.len();
    if s.degree < 1 {
        return Err(Raised);
    }
    let order = s.degree as usize + 1;
    if order > count {
        return Err(Raised);
    }
    let mut knots = if s.knots.is_empty() { open_uniform_knots(count, order) } else { round_knots(&s.knots, s.knot_tolerance) };
    if knots.len() != count + order {
        return Err(Raised);
    }
    if !s.knots.is_empty() && knots[0] != 0.0 {
        let min_val = knots[0];
        let max_val = knots[knots.len() - 1] - min_val;
        if max_val == 0.0 {
            return Err(Raised);
        }
        knots = knots.iter().map(|v| (v - min_val) / max_val).collect();
    }
    let clamped = knots[..order].iter().all(|&k| k == knots[0]) && knots[knots.len() - order..].iter().all(|&k| k == knots[knots.len() - 1]);
    if !(s.degree == 3 && s.weights.is_empty() && clamped) {
        return Err(Raised);
    }
    let mut path = Path::new(s.control_points[0]);
    let curves: Vec<Curve> = bezier_decomposition(&s.control_points, &knots, 3)?;
    add_bezier4p(&mut path, curves);
    Ok(path)
}
