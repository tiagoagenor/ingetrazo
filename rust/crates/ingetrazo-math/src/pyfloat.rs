// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! Python's float built-ins, so a port gives the Python app's numbers.

/// CPython's `sum()` of floats (3.12+): Neumaier's compensated sum, the
/// compensation added at the end when it is finite and not zero.
pub fn py_sum<I: Iterator<Item = f64>>(xs: I, compensated: bool) -> f64 {
    let mut f = 0.0f64;
    let mut c = 0.0f64;
    for x in xs {
        if !compensated {
            f += x;                 // CPython before 3.12
            continue;
        }
        let t = f + x;
        if f.abs() >= x.abs() {
            c += (f - t) + x;
        } else {
            c += (x - t) + f;
        }
        f = t;
    }
    if c != 0.0 && c.is_finite() {
        f += c;
    }
    f
}

// Python's min/max keep the FIRST of equal values and compare with `<`/`>`.
#[inline]
pub fn py_min2(a: f64, b: f64) -> f64 {
    if b < a { b } else { a }
}
#[inline]
pub fn py_max2(a: f64, b: f64) -> f64 {
    if b > a { b } else { a }
}
#[inline]
pub fn py_min3(a: f64, b: f64, c: f64) -> f64 {
    py_min2(py_min2(a, b), c)
}
#[inline]
pub fn py_max3(a: f64, b: f64, c: f64) -> f64 {
    py_max2(py_max2(a, b), c)
}
