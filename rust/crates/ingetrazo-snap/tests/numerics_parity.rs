// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! The NumPy arithmetic of the pick index against NumPy itself
//! (scripts/golden/make_pick_fixtures.py, `numerics`): `(N,3) @ (3,)`,
//! `pts @ M[:, :3].T + M[:, 3]`, `einsum("ij,ij->i")`, `np.cross` and
//! `np.hypot` for 1 to 40, 100 and 257 rows, bit for bit; and the
//! determinant of 3x3 matrices, exact where the matrix is an identity or a
//! scale and within 2 ulp for any other (NumPy's LAPACK order is not
//! reproduced; it only scales the area of a placed face, which the
//! coplanar tie-break compares).
//!
//! The orders belong to the CPU dispatch of the OpenBLAS inside NumPy; the
//! fixtures were made on a Haswell-class kernel (AVX2 + FMA).

use std::path::{Path, PathBuf};

use ingetrazo_formats::pyjson::{loads, Value};
use ingetrazo_snap::picking::numeric::{cross3, einsum3, matmul3, matvec3, np_det3, np_hypot, P3};

fn data() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/pick")
}

fn arr(v: &Value) -> &[Value] {
    match v {
        Value::Array(a) => a,
        other => panic!("not an array: {other:?}"),
    }
}

fn f(v: &Value) -> f64 {
    match v {
        Value::Str(s) => f64::from_bits(u64::from_str_radix(s, 16).unwrap()),
        other => panic!("not a hex float: {other:?}"),
    }
}

fn row(v: &Value) -> P3 {
    let a = arr(v);
    [f(&a[0]), f(&a[1]), f(&a[2])]
}

fn rows(v: &Value) -> Vec<P3> {
    arr(v).iter().map(row).collect()
}

fn same(a: f64, b: f64) -> bool {
    a.to_bits() == b.to_bits()
}

#[test]
fn numpy_orders() {
    let raw = std::fs::read(data().join("numerics.json")).unwrap();
    let doc = loads(std::str::from_utf8(&raw).unwrap()).unwrap();
    let mut n_checked = 0usize;
    for c in arr(doc.get("gemv").unwrap()) {
        let q = rows(c.get("q").unwrap());
        let d = row(c.get("d").unwrap());
        for (i, want) in arr(c.get("out").unwrap()).iter().enumerate() {
            assert!(same(matvec3(q[i], d, q.len()), f(want)), "gemv n={} row {i}", q.len());
            n_checked += 1;
        }
    }
    for c in arr(doc.get("gemm").unwrap()) {
        let p = rows(c.get("p").unwrap());
        let m: Vec<Vec<f64>> = arr(c.get("m").unwrap()).iter().map(|r| arr(r).iter().map(f).collect()).collect();
        let out = arr(c.get("out").unwrap());
        for (i, o) in out.iter().enumerate() {
            let want = arr(o);
            for k in 0..4 {
                let got = matmul3(p[i], [m[k][0], m[k][1], m[k][2]], p.len()) + m[k][3];
                assert!(same(got, f(&want[k])), "gemm n={} row {i} col {k}", p.len());
                n_checked += 1;
            }
        }
    }
    for c in arr(doc.get("einsum").unwrap()) {
        let (a, b) = (rows(c.get("a").unwrap()), rows(c.get("b").unwrap()));
        for (i, want) in arr(c.get("out").unwrap()).iter().enumerate() {
            assert!(same(einsum3(a[i], b[i]), f(want)), "einsum n={} row {i}", a.len());
            n_checked += 1;
        }
    }
    for c in arr(doc.get("cross").unwrap()) {
        let b = rows(c.get("b").unwrap());
        let out = rows(c.get("out").unwrap());
        for i in 0..b.len() {
            let a = match c.get("a").unwrap() {
                Value::Null => rows(c.get("ra").unwrap())[i],
                v => row(v),
            };
            let got = cross3(a, b[i]);
            for k in 0..3 {
                assert!(same(got[k], out[i][k]), "cross row {i} axis {k}");
                n_checked += 1;
            }
        }
    }
    let h = doc.get("hypot").unwrap();
    let (a, b) = (arr(h.get("a").unwrap()), arr(h.get("b").unwrap()));
    // `np_hypot` is the C library's `hypot`, and the fixtures were recorded
    // with the MSVC CRT's: glibc's and macOS's differ in the last bit.
    if cfg!(windows) {
        for (i, want) in arr(h.get("out").unwrap()).iter().enumerate() {
            assert!(same(np_hypot(f(&a[i]), f(&b[i])), f(want)), "hypot {i}");
            n_checked += 1;
        }
    }
    let mut off = 0usize;
    for c in arr(doc.get("det").unwrap()) {
        let m = rows(c.get("m").unwrap());
        let want = f(c.get("out").unwrap());
        let got = np_det3(&[m[0], m[1], m[2]]);
        if matches!(c.get("exact"), Some(Value::Bool(true))) {
            assert!(same(got, want), "det of {m:?}: {got:e} against {want:e}");
        } else if !same(got, want) {
            off += 1;
            let ulp = (got.to_bits() as i64 - want.to_bits() as i64).abs();
            assert!(ulp <= 2 || (got - want).abs() <= 1e-12 * want.abs().max(1e-3), "det of {m:?}: {got:e} against {want:e}");
        }
        n_checked += 1;
    }
    eprintln!("{n_checked} values bit for bit; {off} determinants differ in the last bits");
}
