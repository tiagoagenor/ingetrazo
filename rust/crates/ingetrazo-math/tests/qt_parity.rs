// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! QMatrix4x4 and QVector3D against what Qt 6.11 computed
//! (scripts/golden/make_qt_math_fixtures.py): every float and every type
//! flag, exactly.

use std::path::Path;

use ingetrazo_formats::pyjson::{loads, Value};
use ingetrazo_math::matrix::M4;
use ingetrazo_math::{Hypot, V3};

/// The fixtures were written on Windows: MSVC's hypot.
const H: Hypot = Hypot::Msvc;

fn lines(name: &str) -> Vec<Value> {
    let p = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data").join(name);
    let text = std::fs::read_to_string(p).unwrap();
    text.lines().skip(1).map(|l| loads(l).unwrap()).collect()
}

fn f(v: &Value) -> f32 {
    match v {
        Value::Str(h) => f32::from_bits(u32::from_str_radix(h, 16).unwrap()),
        _ => panic!("{v:?}"),
    }
}

fn hx(x: f32) -> String {
    format!("{:08x}", x.to_bits())
}

fn arr(v: &Value) -> &Vec<Value> {
    match v {
        Value::Array(a) => a,
        _ => panic!("{v:?}"),
    }
}

fn v3(v: &Value) -> V3 {
    let a = arr(v);
    V3(f(&a[0]), f(&a[1]), f(&a[2]))
}

fn vb(v: V3) -> [String; 3] {
    [hx(v.0), hx(v.1), hx(v.2)]
}

fn vbits(v: &Value) -> [String; 3] {
    let a = arr(v);
    [s(&a[0]), s(&a[1]), s(&a[2])]
}

fn s(v: &Value) -> String {
    match v {
        Value::Str(x) => x.clone(),
        _ => panic!("{v:?}"),
    }
}

fn build(c: &Value) -> M4 {
    let mut m = match c.get("start") {
        Some(Value::Str(x)) if x == "general16" => {
            let vals = arr(c.get("values").unwrap());
            let mut a = [0.0f32; 16];
            for (i, x) in vals.iter().enumerate() {
                a[i] = f(x);
            }
            M4::from_rows(a)
        }
        _ => M4::identity(),
    };
    for op in arr(c.get("ops").unwrap()) {
        let op = arr(op);
        let name = s(&op[0]);
        let a: Vec<f32> = if name == "mul" || name == "mul_assign" { vec![] } else { op[1..].iter().map(f).collect() };
        match name.as_str() {
            "translate" => m.translate(a[0], a[1], a[2]),
            "translate2" => m.translate2(a[0], a[1]),
            "scale" => m.scale(a[0], a[1], a[2]),
            "scale1" => m.scale1(a[0]),
            "rotate" => m.rotate(a[0], a[1], a[2], a[3]),
            "mul" => m = m.mul(&build(&op[1])),
            "mul_assign" => m.mul_assign(&build(&op[1])),
            "inverted" => m = m.inverted().0,
            "transposed" => m = m.transposed(),
            "optimize" => m.optimize(),
            "look_at" => m.look_at(V3(a[0], a[1], a[2]), V3(a[3], a[4], a[5]), V3(a[6], a[7], a[8]), H),
            "ortho" => m.ortho(a[0], a[1], a[2], a[3], a[4], a[5]),
            "perspective" => m.perspective(a[0], a[1], a[2], a[3]),
            "general16" => {
                let mut v = [0.0f32; 16];
                v.copy_from_slice(&a);
                m = M4::from_rows(v);
            }
            other => panic!("op {other}"),
        }
    }
    m
}

fn mbits(m: &M4) -> Vec<String> {
    m.data().iter().map(|x| hx(*x)).collect()
}

#[test]
fn qmatrix4x4_matches_qt() {
    let rows = lines("qmatrix4x4.jsonl");
    let mut fails = Vec::new();
    for (i, r) in rows.iter().enumerate() {
        let m = build(r.get("chain").unwrap());
        let mut why = Vec::new();
        let want: Vec<String> = arr(r.get("m").unwrap()).iter().map(s).collect();
        if mbits(&m) != want {
            why.push(format!("m {:?} vs {:?}", mbits(&m), want));
        }
        if Some(m.flags as i64) != r.get("flags").and_then(Value::as_i64) {
            why.push(format!("flags {} vs {:?}", m.flags, r.get("flags")));
        }
        if Some(&Value::Bool(m.is_identity())) != r.get("identity") {
            why.push("isIdentity".into());
        }
        for pair in arr(r.get("map").unwrap()) {
            let pair = arr(pair);
            if vb(m.map(v3(&pair[0]))) != vbits(&pair[1]) {
                why.push(format!("map {:?}", vbits(&pair[0])));
            }
        }
        for pair in arr(r.get("map_vector").unwrap()) {
            let pair = arr(pair);
            if vb(m.map_vector(v3(&pair[0]))) != vbits(&pair[1]) {
                why.push(format!("mapVector {:?}", vbits(&pair[0])));
            }
        }
        let inv = r.get("inverted").unwrap();
        let (im, ok) = m.inverted();
        let want: Vec<String> = arr(inv.get("m").unwrap()).iter().map(s).collect();
        if mbits(&im) != want || Some(im.flags as i64) != inv.get("flags").and_then(Value::as_i64) || Some(&Value::Bool(ok)) != inv.get("ok") {
            why.push(format!("inverted {:?} f{} vs {:?}", mbits(&im), im.flags, want));
        }
        if format!("{:016x}", m.determinant().to_bits()) != s(r.get("det").unwrap()) {
            why.push(format!("det {:e}", m.determinant()));
        }
        if !why.is_empty() {
            fails.push(format!("#{i} {}: {}", ingetrazo_formats::pyjson::dumps(r.get("chain").unwrap(), &Default::default()).unwrap(), why.join("; ")));
        }
    }
    assert!(fails.is_empty(), "{} of {} chains differ:\n{}", fails.len(), rows.len(), fails.iter().take(12).cloned().collect::<Vec<_>>().join("\n"));
}

#[test]
fn qvector3d_matches_qt() {
    let rows = lines("qvector3d.jsonl");
    let mut fails = Vec::new();
    for (i, r) in rows.iter().enumerate() {
        let g = |k: &str| r.get(k).unwrap();
        let (a, b, c, k) = (v3(g("a")), v3(g("b")), v3(g("c")), f(g("s")));
        let mut why = Vec::new();
        let mut chk_v = |name: &str, got: V3| {
            if vb(got) != vbits(g(name)) {
                why.push(name.to_string());
            }
        };
        chk_v("add", a.add(b));
        chk_v("sub", a.sub(b));
        chk_v("neg", a.neg());
        chk_v("mul_s", a.mul(k));
        chk_v("s_mul", a.mul(k));
        chk_v("mul_v", a.mul_v(b));
        chk_v("cross", a.cross(b));
        chk_v("normalized", a.normalized(H));
        chk_v("normal2", V3::normal2(a, b, H));
        chk_v("normal3", V3::normal3(a, b, c, H));
        if !matches!(g("div_s"), Value::Null) {
            chk_v("div_s", a.div(k));
        }
        let mut chk_f = |name: &str, got: f32| {
            if hx(got) != s(g(name)) {
                why.push(name.to_string());
            }
        };
        chk_f("dot", a.dot(b));
        chk_f("len", a.length(H));
        chk_f("len2", a.length_squared());
        chk_f("dist_point", a.distance_to_point(b, H));
        chk_f("dist_plane", a.distance_to_plane(b, c));
        chk_f("dist_plane3", a.distance_to_plane3(a.add(b), b, c, H));
        chk_f("dist_line", a.distance_to_line(b, c, H));
        if !why.is_empty() {
            fails.push(format!("#{i}: {}", why.join(", ")));
        }
    }
    assert!(fails.is_empty(), "{} of {} rows differ:\n{}", fails.len(), rows.len(), fails.iter().take(20).cloned().collect::<Vec<_>>().join("\n"));
}
