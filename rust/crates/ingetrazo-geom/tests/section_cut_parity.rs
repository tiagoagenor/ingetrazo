// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! `section_cut` against `core/section.py` (scripts/golden/make_section_fixtures.py):
//! the chords a plane cuts out of triangles, the fill triangles and the hatch
//! segments, bit for bit.

use std::path::Path;

use ingetrazo_formats::pyjson::{loads, Value};
use ingetrazo_geom::section_cut::{cut_chords, fill_triangles, hatch_segments, Chord};
use ingetrazo_math::V3;

fn s(v: &Value) -> &str {
    match v {
        Value::Str(h) => h,
        _ => panic!("{v:?}"),
    }
}

fn get<'a>(v: &'a Value, k: &str) -> &'a Value {
    v.get(k).unwrap_or_else(|| panic!("missing {k}"))
}

fn f32s(hex: &str) -> Vec<f32> {
    (0..hex.len() / 8).map(|i| f32::from_bits(u32::from_str_radix(&hex[i * 8..i * 8 + 8], 16).unwrap())).collect()
}

fn triples(hex: &str) -> Vec<[f32; 3]> {
    f32s(hex).chunks(3).map(|c| [c[0], c[1], c[2]]).collect()
}

fn wide(v: &[[f32; 3]]) -> Vec<[f64; 3]> {
    v.iter().map(|p| [p[0] as f64, p[1] as f64, p[2] as f64]).collect()
}

fn v3(v: &Value) -> V3 {
    let Value::Array(a) = v else { panic!("{v:?}") };
    let f = |x: &Value| f32::from_bits(u32::from_str_radix(s(x), 16).unwrap());
    V3(f(&a[0]), f(&a[1]), f(&a[2]))
}

fn bits(v: &[[f32; 3]]) -> Vec<u32> {
    v.iter().flatten().map(|x| x.to_bits()).collect()
}

/// Compare two lists of points; on a difference say where, not everything.
fn same(what: &str, got: &[[f32; 3]], want: &[[f32; 3]]) -> Result<(), String> {
    if bits(got) == bits(want) {
        return Ok(());
    }
    let at = (0..got.len().min(want.len())).find(|&i| bits(&got[i..=i]) != bits(&want[i..=i]));
    Err(format!("{what}: got {} points, Python {}; first difference at {at:?}: {:?} vs {:?}", got.len(), want.len(), at.map(|i| got[i]), at.map(|i| want[i])))
}

fn chords_of(hex: &str) -> Vec<Chord> {
    let p = triples(hex);
    p.chunks(2).map(|c| [c[0], c[1]]).collect()
}

#[test]
fn the_section_cut_matches_core_section() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/section_cut.jsonl");
    let text = std::fs::read_to_string(path).unwrap();
    let (mut cuts, mut fills, mut hatches) = (0, 0, 0);
    let mut bad: Vec<String> = Vec::new();
    for line in text.lines() {
        let c = loads(line).unwrap();
        let name = s(get(&c, "name")).to_string();
        match s(get(&c, "kind")) {
            "cut" => {
                let (v0, e1, e2) = (triples(s(get(&c, "v0"))), triples(s(get(&c, "e1"))), triples(s(get(&c, "e2"))));
                let visible: Option<Vec<bool>> = match get(&c, "visible") {
                    Value::Str(m) => Some(m.chars().map(|ch| ch == '1').collect()),
                    _ => None,
                };
                let got = cut_chords(&wide(&v0), &wide(&e1), &wide(&e2), v3(get(&c, "normal")), v3(get(&c, "point")), visible.as_deref());
                match get(&c, "out") {
                    Value::Str(hex) => {
                        let want = triples(hex);
                        let got = got.unwrap_or_else(|| panic!("{name}: no chords, Python cut {}", want.len() / 2));
                        let flat: Vec<[f32; 3]> = got.iter().flat_map(|c| [c[0], c[1]]).collect();
                        if let Err(e) = same(&format!("{name}: chords"), &flat, &want) { bad.push(e); }
                    }
                    _ => assert!(got.is_none(), "{name}: chords where Python cut none"),
                }
                cuts += 1;
            }
            "fill" => {
                let chords = chords_of(s(get(&c, "chords")));
                let n = v3(get(&c, "normal"));
                let tris = fill_triangles(&chords, n);
                let flat: Vec<[f32; 3]> = tris.iter().flatten().copied().collect();
                if let Err(e) = same(&format!("{name}: fill"), &flat, &triples(s(get(&c, "tris")))) { bad.push(e); }
                fills += 1;
                let Value::Array(hs) = get(&c, "hatches") else { panic!() };
                for h in hs {
                    let spacing = f64::from_bits(u64::from_str_radix(s(get(h, "spacing")), 16).unwrap());
                    let cross = matches!(get(h, "cross"), Value::Bool(true));
                    let got = hatch_segments(&chords, n, spacing, cross);
                    let flat: Vec<[f32; 3]> = got.iter().flatten().copied().collect();
                    if let Err(e) = same(&format!("{name}: hatch {spacing} cross={cross}"), &flat, &triples(s(get(h, "out")))) { bad.push(e); }
                    hatches += 1;
                }
            }
            other => panic!("{other}"),
        }
    }
    assert!(bad.is_empty(), "{} differ:
{}", bad.len(), bad.iter().take(12).cloned().collect::<Vec<_>>().join("
"));
    assert!(cuts > 100 && fills > 100 && hatches > 300, "{cuts} cuts, {fills} fills, {hatches} hatches");
}
