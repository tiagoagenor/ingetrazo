// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! DXF against `formats/dxf_out.py` and `formats/dxf_in.py`
//! (scripts/golden/make_dxf_fixtures.py): export is compared byte for byte
//! with what the real writer produced, import by the L2 signature of the
//! scene Python built from the same file (with `run_stitch`, the edit
//! engine, a no-op on both sides) and by the stats Python returned.
//!
//! The fixtures come from a Windows machine: DXF lines end in `\r\n`
//! (Python's text mode) and `length()` is MSVC's `hypot`.

use std::path::{Path, PathBuf};

use ingetrazo_formats::pyjson::{loads, Value};
use ingetrazo_formats_mesh::dxf::export::{ascii_bytes, fmt_g9, save_dxf_layers, save_dxf_layers_text, LayerGroup};
use ingetrazo_formats_mesh::dxf::import::{load_dxf_bytes, read_unit_scale, suggest_unit_scale, DxfStats};
use ingetrazo_formats_mesh::dxf::read::read_dxf;
use ingetrazo_golden::model_sig::scene_signature;
use ingetrazo_golden::sig::{canon_dumps, Files};
use ingetrazo_math::Hypot;
use ingetrazo_model::scene::Scene;

const H: Hypot = Hypot::Msvc;

fn data() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/dxf")
}

fn bits(s: &str) -> f64 {
    f64::from_bits(u64::from_str_radix(s, 16).unwrap())
}

fn arr(v: &Value) -> &[Value] {
    match v {
        Value::Array(a) => a,
        other => panic!("{other:?}"),
    }
}

fn text(v: Option<&Value>) -> Option<&str> {
    match v {
        Some(Value::Str(s)) => Some(s),
        _ => None,
    }
}

fn num(v: &Value) -> f64 {
    match v {
        Value::Float(x) => *x,
        Value::Int(i) => i.parse().unwrap(),
        other => panic!("{other:?}"),
    }
}

fn read_json(path: &Path) -> Value {
    loads(&std::fs::read_to_string(path).unwrap()).unwrap()
}

// ---- export -------------------------------------------------------------------

fn groups_of(case: &Value) -> Vec<LayerGroup> {
    arr(case.get("groups").unwrap())
        .iter()
        .map(|g| LayerGroup {
            layer: text(g.get("layer")).unwrap().to_string(),
            segments: arr(g.get("segs").unwrap())
                .iter()
                .map(|s| {
                    let s = arr(s);
                    [bits(text(s.first()).unwrap()), bits(text(s.get(1)).unwrap()), bits(text(s.get(2)).unwrap()), bits(text(s.get(3)).unwrap())]
                })
                .collect(),
            linetype: text(g.get("ltype")).map(str::to_string),
        })
        .collect()
}

#[test]
fn view_dxf_matches_python_byte_for_byte() {
    let dir = data().join("export");
    let cases = read_json(&dir.join("cases.json"));
    let mut n = 0;
    for case in arr(&cases) {
        let name = text(case.get("name")).unwrap();
        let groups = groups_of(case);
        let (txt, lines) = save_dxf_layers_text(&groups);
        let got = ascii_bytes(&txt, true);
        let want = std::fs::read(dir.join(format!("{name}.dxf"))).unwrap();
        assert_eq!(lines as i64, case.get("lines").unwrap().as_i64().unwrap(), "{name}: line count");
        assert_eq!(got.len(), want.len(), "{name}: length");
        if let Some(i) = got.iter().zip(&want).position(|(a, b)| a != b) {
            let lo = i.saturating_sub(60);
            panic!("{name}: first byte differing at {i}: got {:?} want {:?}", String::from_utf8_lossy(&got[lo..(i + 40).min(got.len())]), String::from_utf8_lossy(&want[lo..(i + 40).min(want.len())]));
        }
        n += 1;
    }
    assert_eq!(n, 10);
}

#[test]
fn save_dxf_layers_writes_the_file() {
    let tmp = std::env::temp_dir().join(format!("ingetrazo-dxf-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp).unwrap();
    let dir = data().join("export");
    let cases = read_json(&dir.join("cases.json"));
    for case in arr(&cases) {
        let name = text(case.get("name")).unwrap();
        let out = tmp.join(format!("{name}.dxf"));
        let n = save_dxf_layers(&out, &groups_of(case), true).unwrap();
        assert_eq!(n as i64, case.get("lines").unwrap().as_i64().unwrap());
        assert_eq!(std::fs::read(&out).unwrap(), std::fs::read(dir.join(format!("{name}.dxf"))).unwrap(), "{name}");
    }
    let _ = std::fs::remove_dir_all(&tmp);
}

#[test]
fn nine_significant_digits_match_python() {
    let text = std::fs::read_to_string(data().join("export/g9.txt")).unwrap();
    let mut n = 0;
    for line in text.lines() {
        let (b, want) = line.split_once(' ').unwrap();
        let x = bits(b);
        assert_eq!(fmt_g9(x), want, "format(x, '.9g') of {x:e}");
        n += 1;
    }
    assert!(n >= 4000, "{n}");
}

// ---- import -------------------------------------------------------------------

fn stem(file: &str) -> &str {
    file.rsplit_once('.').map(|(s, _)| s).unwrap_or(file)
}

fn stats_text(s: &DxfStats) -> String {
    let comps: std::collections::BTreeMap<_, _> = s.components.iter().cloned().collect();
    format!(
        "entities={} edges={} faces={} groups={} components={:?} skipped={:?} offset={:?}",
        s.entities, s.edges, s.faces, s.groups, comps, s.skipped, s.offset.map(|o| o.map(f64::to_bits))
    )
}

fn want_stats(v: &Value) -> String {
    let ints = |k: &str| v.get(k).unwrap().as_i64().unwrap();
    let dict = |k: &str| -> std::collections::BTreeMap<String, usize> {
        match v.get(k).unwrap() {
            Value::Object(items) => items.iter().map(|(k, v)| (k.clone(), v.as_i64().unwrap() as usize)).collect(),
            other => panic!("{other:?}"),
        }
    };
    let offset = match v.get("offset").unwrap() {
        Value::Null => None,
        Value::Array(a) => Some([num(&a[0]).to_bits(), num(&a[1]).to_bits(), num(&a[2]).to_bits()]),
        other => panic!("{other:?}"),
    };
    format!(
        "entities={} edges={} faces={} groups={} components={:?} skipped={:?} offset={:?}",
        ints("entities"),
        ints("edges"),
        ints("faces"),
        ints("groups"),
        dict("components"),
        dict("skipped"),
        offset
    )
}

fn sig_of(scene: &Scene) -> String {
    canon_dumps(&scene_signature(scene, &Files { zip: None }))
}

fn scale_unit(v: &Value) -> (Option<f64>, i64) {
    let a = arr(v);
    (if matches!(a[0], Value::Null) { None } else { Some(num(&a[0])) }, a[1].as_i64().unwrap())
}


#[derive(Default, Debug)]
struct Tally {
    compared: usize,
    errors: usize,
    opened: usize,
    pending: usize,
    failures: Vec<String>,
}

/// Runs every case of `dir/cases.json`; mismatches are collected, not thrown.
fn run_import_cases(dir: &Path) -> Tally {
    let mut t = Tally::default();
    for case in arr(&read_json(&dir.join("cases.json"))) {
        let file = text(case.get("file")).unwrap().to_string();
        let scale = num(case.get("scale").unwrap());
        let label = format!("{file} (scale {scale})");
        let bytes = std::fs::read(dir.join(&file)).unwrap();
        let doc = read_dxf(&bytes);
        if case.get("open_error").is_some() {
            if doc.is_err() {
                t.errors += 1;
            } else {
                t.failures.push(format!("{label}: Python could not open it"));
            }
            continue;
        }
        let doc = match doc {
            Ok(d) => d,
            Err(e) => {
                t.failures.push(format!("{label}: {e}"));
                continue;
            }
        };
        if read_unit_scale(&doc) != scale_unit(case.get("header").unwrap()) {
            t.failures.push(format!("{label}: header unit {:?}", read_unit_scale(&doc)));
        }
        if suggest_unit_scale(&doc) != scale_unit(case.get("suggest").unwrap()) {
            t.failures.push(format!("{label}: suggested unit {:?}", suggest_unit_scale(&doc)));
        }
        t.opened += 1;
        if case.get("sig").is_none() && case.get("error").is_none() {
            continue;
        }
        let mut scene = Scene::new();
        let name = text(case.get("name"));
        let result = load_dxf_bytes(&mut scene, &bytes, stem(&file), scale, name, H, None);
        if case.get("pending").is_some() {
            // What the port does not do yet: Python imports it, this may
            // import less or nothing; stats and signature stay unchecked.
            t.pending += 1;
            continue;
        }
        if let Some(want) = text(case.get("error")) {
            match result {
                Err(got) if got == want.split_once(": ").unwrap().1 => t.errors += 1,
                other => t.failures.push(format!("{label}: Python raised {want}, got {other:?}")),
            }
            continue;
        }
        let stats = match result {
            Ok(s) => s,
            Err(e) => {
                t.failures.push(format!("{label}: {e}"));
                continue;
            }
        };
        let (got_stats, want_stats) = (stats_text(&stats), want_stats(case.get("stats").unwrap()));
        if got_stats != want_stats {
            t.failures.push(format!("{label}: stats\n   got {got_stats}\n  want {want_stats}"));
            continue;
        }
        let want_text = std::fs::read_to_string(dir.join(text(case.get("sig")).unwrap())).unwrap();
        let want = canon_dumps(&loads(&want_text).unwrap());
        let got = sig_of(&scene);
        if got != want {
            let at = got.bytes().zip(want.bytes()).position(|(a, b)| a != b).unwrap_or(0);
            let lo = at.saturating_sub(80);
            t.failures.push(format!("{label}: signature differs near byte {at}\n   got: {}\n  want: {}", &got[lo..(at + 80).min(got.len())], &want[lo..(at + 80).min(want.len())]));
            continue;
        }
        t.compared += 1;
    }
    t
}

#[test]
fn imports_give_the_signature_python_had() {
    let t = run_import_cases(&data().join("import"));
    assert!(t.failures.is_empty(), "{}", t.failures.join("\n"));
    assert_eq!(t.compared, 249, "signatures compared");
    assert_eq!(t.errors, 18, "errors compared");
    assert_eq!(t.opened, 261, "files opened");
    assert_eq!(t.pending, 3, "cases waiting for a port");
}

/// A directory of random documents and Python's signatures for them (the
/// scratch fuzzer writes it); `INGETRAZO_DXF_FUZZ_DIR` names it.
#[test]
fn random_documents_give_the_signature_python_had() {
    let Some(dir) = std::env::var_os("INGETRAZO_DXF_FUZZ_DIR") else { return };
    let t = run_import_cases(Path::new(&dir));
    eprintln!("{} compared, {} errors, {} pending, {} failures", t.compared, t.errors, t.pending, t.failures.len());
    assert!(t.failures.is_empty(), "{} failures\n{}", t.failures.len(), t.failures.join("\n"));
}
