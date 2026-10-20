// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! SPLINE to a `Path`: `ezdxf.path.make_path(spline)` for a SPLINE that
//! carries control points. A clamped, non-rational cubic is decomposed
//! exactly into cubic Beziers (`BSpline.bezier_decomposition`); every other
//! one is approximated through its chord-length parameters, the curve's
//! points at the subdivided parameters and the cubic Bezier interpolation
//! through them (`BSpline.cubic_bezier_approximation(level=4)`), with the
//! Cython `Basis` / `Evaluator` arithmetic and CPython's compensated `sum`.
//!
//! Not ported (they return `Err`, so the spline imports as nothing where
//! ezdxf draws it): splines that only carry fit points
//! (`fit_points_to_cad_cv` solves its system through the numerics of
//! NumPy).

use ingetrazo_math::pyfloat::py_sum;

use super::entities::SplineData;
use super::geom::{add_bezier4p, isclose, Curve, Path, Raised, NULLVEC, P, R, ABS_TOL, REL_TOL};

/// `acc/constants.h`: AutoCAD limits the order to 12.
const MAX_SPLINE_ORDER: usize = 12;

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

/// `ezdxf.acc.bspline.Basis`.
struct Basis {
    knots: Vec<f64>,
    order: usize,
    count: usize,
    weights: Vec<f64>,
    max_t: f64,
}

impl Basis {
    fn new(knots: Vec<f64>, order: usize, count: usize, weights: Vec<f64>) -> R<Basis> {
        if order < 2 || order >= MAX_SPLINE_ORDER || count < 2 {
            return Err(Raised);
        }
        if !weights.is_empty() && weights.len() != count {
            return Err(Raised);
        }
        if knots.len() != order + count {
            return Err(Raised);
        }
        let max_t = knots[knots.len() - 1];
        Ok(Basis { knots, order, count, weights, max_t })
    }

    /// `find_span(u)`: Cython's C `int`, so -1 is possible.
    fn find_span(&self, u: f64) -> i64 {
        let knots = &self.knots;
        let count = self.count;
        let p = self.order - 1;
        if u >= knots[count] {
            return count as i64 - 1;
        }
        if knots[p] == 0.0 {
            // bisect_right(knots, u, p, count) - 1
            let (mut lo, mut hi) = (p, count);
            while lo < hi {
                let mid = (lo + hi) / 2;
                if u < knots[mid] {
                    hi = mid;
                } else {
                    lo = mid + 1;
                }
            }
            lo as i64 - 1
        } else {
            let mut span = 0usize;
            while knots[span] <= u && span < count {
                span += 1;
            }
            span as i64 - 1
        }
    }

    /// `basis_funcs(span, u)`.
    fn basis_funcs(&self, span: i64, u: f64) -> R<Vec<f64>> {
        let order = self.order;
        let knots = &self.knots;
        let mut n = [0.0f64; MAX_SPLINE_ORDER];
        let mut left = [0.0f64; MAX_SPLINE_ORDER];
        let mut right = [0.0f64; MAX_SPLINE_ORDER];
        n[0] = 1.0;
        for j in 1..order {
            let i1 = (span + 1 - j as i64).max(0) as usize;
            left[j] = u - knots[i1];
            right[j] = knots[(span + j as i64) as usize] - u;
            let mut saved = 0.0;
            for r in 0..j {
                let temp_r = right[r + 1];
                let temp_l = left[j - r];
                let denominator = temp_r + temp_l;
                if denominator == 0.0 {
                    // Cython's float division raises ZeroDivisionError.
                    return Err(Raised);
                }
                let temp = n[r] / denominator;
                n[r] = saved + temp_r * temp;
                saved = temp_l * temp;
            }
            n[j] = saved;
        }
        let result: Vec<f64> = n[..order].to_vec();
        Ok(if self.weights.is_empty() { result } else { self.span_weighting(result, span) })
    }

    /// `span_weighting(nbasis, span)`, with Python's slice of the weights.
    fn span_weighting(&self, nbasis: Vec<f64>, span: i64) -> Vec<f64> {
        let len = self.weights.len() as i64;
        let norm = |i: i64| -> usize {
            let i = if i < 0 { i + len } else { i };
            i.clamp(0, len) as usize
        };
        let start = norm(span - self.order as i64 + 1);
        let stop = norm(span + 1);
        let window = if start < stop { &self.weights[start..stop] } else { &self.weights[0..0] };
        let products: Vec<f64> = nbasis.iter().zip(window).map(|(nb, w)| nb * w).collect();
        if products.len() != nbasis.len() {
            return nbasis;
        }
        let s = py_sum(products.iter().copied(), true);
        if s != 0.0 {
            products.iter().map(|p| p / s).collect()
        } else {
            nbasis
        }
    }

    /// `Evaluator.point(u)`; the control points are indexed like a Python
    /// tuple (a negative index wraps round).
    fn point(&self, control_points: &[P], u: f64) -> R<P> {
        let u = if isclose(u, self.max_t, REL_TOL, ABS_TOL) { self.max_t } else { u };
        let p = self.order - 1;
        let span = self.find_span(u);
        let n = self.basis_funcs(span, u)?;
        let len = control_points.len() as i64;
        let mut sum = NULLVEC;
        for i in 0..=p {
            let mut idx = span - p as i64 + i as i64;
            if idx < 0 {
                idx += len;
            }
            if idx < 0 || idx >= len {
                return Err(Raised);
            }
            let c = control_points[idx as usize];
            sum.x += c.x * n[i];
            sum.y += c.y * n[i];
            sum.z += c.z * n[i];
        }
        Ok(sum)
    }
}

/// `BSpline.bezier_decomposition()`.
fn bezier_decomposition(cps: &[P], knots: &[f64], p: usize) -> R<Vec<Curve>> {
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

/// `_solve_tridiagonal_matrix(a, b, c, r)`; a zero pivot is Python's
/// `ZeroDivisionError`.
fn solve_tridiagonal(a: &[f64], b: &[f64], c: &[f64], r: &[f64]) -> R<Vec<f64>> {
    let n = a.len();
    let mut u = vec![0.0; n];
    let mut gam = vec![0.0; n];
    let mut bet = b[0];
    if bet == 0.0 {
        return Err(Raised);
    }
    u[0] = r[0] / bet;
    for j in 1..n {
        gam[j] = c[j - 1] / bet;
        bet = b[j] - a[j] * gam[j];
        if bet == 0.0 {
            return Err(Raised);
        }
        u[j] = (r[j] - a[j] * u[j - 1]) / bet;
    }
    for j in (0..n.saturating_sub(1)).rev() {
        u[j] -= gam[j + 1] * u[j + 1];
    }
    Ok(u)
}

/// `cubic_bezier_interpolation(points)`.
fn cubic_bezier_interpolation(pnts: &[P]) -> R<Vec<Curve>> {
    if pnts.len() < 3 {
        return Ok(Vec::new());
    }
    let num = pnts.len() - 1;
    let mut b = vec![4.0; num];
    let mut a = vec![1.0; num];
    let c = vec![1.0; num];
    b[0] = 2.0;
    b[num - 1] = 7.0;
    a[num - 1] = 2.0;
    let mut rhs: Vec<P> = vec![pnts[0].add(pnts[1].mul(2.0))];
    for i in 1..num.saturating_sub(1) {
        rhs.push(pnts[i].mul(2.0).add(pnts[i + 1]).mul(2.0));
    }
    // 2.0 * (2.0 * p[i] + p[i + 1]): v3_mul by 2.0 after the sum.
    rhs.push(pnts[num - 1].mul(8.0).add(pnts[num]));
    let col = |f: &dyn Fn(&P) -> f64| -> R<Vec<f64>> { solve_tridiagonal(&a, &b, &c, &rhs.iter().map(f).collect::<Vec<f64>>()) };
    let xs = col(&|p| p.x)?;
    let ys = col(&|p| p.y)?;
    let zs = col(&|p| p.z)?;
    let cp1: Vec<P> = (0..num).map(|i| P::new(xs[i], ys[i], zs[i])).collect();
    let mut cp2: Vec<P> = pnts[1..].iter().zip(&cp1[1..]).map(|(p, cp)| p.mul(2.0).sub(*cp)).collect();
    cp2.push(cp1[num - 1].add(pnts[num]).mul(1.0 / 2.0));
    Ok((0..num).map(|i| [pnts[i], cp1[i], cp2[i], pnts[i + 1]]).collect())
}

pub fn make_path(s: &SplineData) -> R<Path> {
    // Only splines with control points: the fit-point form is not ported.
    if s.control_points.is_empty() {
        return Err(Raised);
    }
    let cps = &s.control_points;
    let count = cps.len();
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
    let basis = Basis::new(knots.clone(), order, count, s.weights.clone())?;
    let mut path = Path::new(basis.point(cps, 0.0)?);
    let curves: Vec<Curve> = if s.degree == 3 && s.weights.is_empty() && clamped {
        bezier_decomposition(cps, &knots, 3)?
    } else {
        // approximation_params(level=4)
        let distances: Vec<f64> = cps.windows(2).map(|w| w[0].dist(w[1])).collect();
        let total = py_sum(distances.iter().copied(), true);
        let mut params: Vec<f64> = Vec::new();
        if total.abs() > 1e-12 {
            params.push(0.0);
            let mut acc = 0.0;
            for d in &distances[..distances.len() - 1] {
                acc += d;
                params.push(acc / total);
            }
            params.push(1.0);
        }
        if !params.is_empty() {
            if basis.max_t != 1.0 {
                params = params.iter().map(|p| p * basis.max_t).collect();
            }
            for _ in 0..3 {
                let mut next = Vec::with_capacity(params.len() * 2);
                for i in 0..params.len() - 1 {
                    next.push(params[i]);
                    next.push((params[i] + params[i + 1]) / 2.0);
                }
                next.push(params[params.len() - 1]);
                params = next;
            }
        }
        let pts: Vec<P> = params.iter().map(|&u| basis.point(cps, u)).collect::<R<Vec<P>>>()?;
        cubic_bezier_interpolation(&pts)?
    };
    add_bezier4p(&mut path, curves);
    Ok(path)
}
