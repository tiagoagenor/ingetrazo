// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! `PyRandom` against CPython 3.12's `random.Random(seed)`
//! (scripts/golden/make_pyset_fixtures.py): `random()` for 700 draws (past
//! one regeneration of the state) and `uniform()` after them, as hex floats.

use std::path::Path;

use ingetrazo_formats::pyjson::{loads, Value};
use ingetrazo_ops::pyrandom::PyRandom;

/// Parse Python's `float.hex()`: `0x1.8p-3`, `-0x0.0p+0`.
fn from_hex(s: &str) -> f64 {
    let (neg, s) = match s.strip_prefix('-') {
        Some(r) => (true, r),
        None => (false, s),
    };
    let s = s.strip_prefix("0x").unwrap();
    let (mant, exp) = s.split_once('p').unwrap();
    let (int, frac) = mant.split_once('.').unwrap_or((mant, ""));
    let mut m = u64::from_str_radix(int, 16).unwrap() as f64;
    let mut scale = 1.0 / 16.0;
    for c in frac.chars() {
        m += c.to_digit(16).unwrap() as f64 * scale;
        scale /= 16.0;
    }
    let v = m * 2f64.powi(exp.parse::<i32>().unwrap());
    if neg {
        -v
    } else {
        v
    }
}

#[test]
fn random_draws_are_pythons() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/pyrandom.json");
    let cases = loads(&std::fs::read_to_string(path).unwrap()).unwrap();
    let Value::Array(cases) = cases else { panic!() };
    for c in &cases {
        let seed = c.get("seed").and_then(Value::as_i64).unwrap() as u64;
        let mut r = PyRandom::new(seed);
        let Some(Value::Array(rs)) = c.get("random") else { panic!() };
        for (i, want) in rs.iter().enumerate() {
            let Value::Str(h) = want else { panic!() };
            assert_eq!(r.random().to_bits(), from_hex(h).to_bits(), "seed {seed} draw {i}");
        }
        let Some(Value::Array(us)) = c.get("uniform") else { panic!() };
        for (i, want) in us.iter().enumerate() {
            let Value::Str(h) = want else { panic!() };
            assert_eq!(r.uniform(-3e-4, 3e-4).to_bits(), from_hex(h).to_bits(), "seed {seed} uniform {i}");
        }
    }
}
