// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! OBJ and STL against `formats/obj.py` and `formats/stl.py`
//! (scripts/golden/make_mesh_formats_fixtures.py): export is compared byte
//! for byte with what the real exporters wrote, import by the L2 signature
//! of the scene Python had after reading the same file.
//!
//! The fixtures come from a Windows machine: OBJ and MTL lines end in
//! `\r\n` (Python's text mode) and `length()` is MSVC's `hypot`.

use std::path::{Path, PathBuf};

use ingetrazo_formats::igz;
use ingetrazo_formats::pyjson::{loads, Value};
use ingetrazo_formats_mesh::obj::{self, add_obj_faces, parse_obj, save_obj, save_obj_text};
use ingetrazo_formats_mesh::pyutil;
use ingetrazo_formats_mesh::stl::{load_stl, save_stl_bytes, Simplify};
use ingetrazo_golden::model_sig::scene_signature;
use ingetrazo_golden::sig::{canon_dumps, Files};
use ingetrazo_math::Hypot;
use ingetrazo_model::scene::Scene;

const H: Hypot = Hypot::Msvc;
const SCENES: [&str; 7] = ["faces", "textured", "components", "bulk", "weld", "classic", "empty"];

fn data() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data")
}

fn crlf(text: &str) -> Vec<u8> {
    text.replace('\n', "\r\n").into_bytes()
}

/// The scene Python exported, opened from its .igz JSON; texture paths
/// (kept relative in the file) pointed at the fixture folder.
fn scene_of(name: &str) -> Scene {
    let dir = data().join("export");
    let raw = std::fs::read(dir.join(format!("{name}.igz.json"))).unwrap();
    let doc = igz::read_document(&raw).unwrap();
    let mut scene = Scene::from_document(&doc).unwrap_or_else(|e| panic!("{name}: {e:?}"));
    for mesh in &mut scene.meshes {
        for f in mesh.faces.clone() {
            for (k, v) in &mut mesh.face_mut(f).attrs {
                if k == "texture" {
                    if let Value::Object(items) = v {
                        for (tk, tv) in items {
                            if tk == "path" {
                                if let Value::Str(p) = tv {
                                    *p = dir.join(&*p).to_string_lossy().into_owned();
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    scene
}

#[test]
fn obj_and_mtl_text_match_python_byte_for_byte() {
    for name in SCENES {
        let scene = scene_of(name);
        let export = save_obj_text(&scene, &format!("{name}.mtl"), H);
        let dir = data().join("export");
        let want_obj = std::fs::read(dir.join(format!("{name}.obj"))).unwrap();
        let want_mtl = std::fs::read(dir.join(format!("{name}.mtl"))).unwrap();
        assert!(String::from_utf8(crlf(&export.obj)).unwrap() == String::from_utf8(want_obj).unwrap(), "{name}.obj differs");
        assert!(String::from_utf8(crlf(&export.mtl)).unwrap() == String::from_utf8(want_mtl).unwrap(), "{name}.mtl differs");
    }
}

#[test]
fn save_obj_writes_the_files_and_copies_the_textures() {
    let tmp = std::env::temp_dir().join(format!("ingetrazo-obj-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp).unwrap();
    for name in ["textured", "components"] {
        let scene = scene_of(name);
        let out = tmp.join(format!("{name}.obj"));
        save_obj(&scene, &out, H, true).unwrap();
        let dir = data().join("export");
        for ext in ["obj", "mtl"] {
            assert_eq!(std::fs::read(tmp.join(format!("{name}.{ext}"))).unwrap(), std::fs::read(dir.join(format!("{name}.{ext}"))).unwrap(), "{name}.{ext}");
        }
        for (_, file) in save_obj_text(&scene, "x.mtl", H).textures {
            assert_eq!(std::fs::read(tmp.join(&file)).unwrap(), std::fs::read(dir.join(&file)).unwrap(), "copy of {file}");
        }
    }
    let _ = std::fs::remove_dir_all(&tmp);
}

#[test]
fn stl_bytes_match_python_in_every_unit() {
    for name in SCENES {
        let scene = scene_of(name);
        for unit in ["m", "mm", "in"] {
            let got = save_stl_bytes(&scene, unit, H).unwrap();
            let want = std::fs::read(data().join("export").join(format!("{name}.{unit}.stl"))).unwrap();
            assert_eq!(got.len(), want.len(), "{name}.{unit}: length");
            if let Some(i) = got.iter().zip(&want).position(|(a, b)| a != b) {
                panic!("{name}.{unit}: first byte differing at {i} (triangle {})", i.saturating_sub(84) / 50);
            }
        }
    }
    assert!(save_stl_bytes(&scene_of("empty"), "furlong", H).is_err());
}

fn bits(s: &str) -> f64 {
    f64::from_bits(u64::from_str_radix(s, 16).unwrap())
}

#[test]
fn float_formatting_and_rounding_match_python() {
    let text = std::fs::read_to_string(data().join("numbers.txt")).unwrap();
    let mut n = 0;
    for line in text.lines() {
        let p: Vec<&str> = line.split(' ').collect();
        let x = bits(p[0]);
        assert_eq!(format!("{x:.6}"), p[1], "{x:e} as .6f");
        assert_eq!(format!("{x:.4}"), p[2], "{x:e} as .4f");
        assert_eq!(pyutil::round6(x).to_bits(), bits(p[3]).to_bits(), "round({x:e}, 6)");
        assert_eq!(pyutil::round4(x).to_bits(), bits(p[4]).to_bits(), "round({x:e}, 4)");
        n += 1;
    }
    assert!(n > 15000, "{n}");
}

// ---- import -------------------------------------------------------------------

fn sig_of(scene: &Scene) -> String {
    canon_dumps(&scene_signature(scene, &Files { zip: None }))
}

fn stem(file: &str) -> String {
    match file.rfind('.') {
        Some(i) if i > 0 => file[..i].to_string(),
        _ => file.to_string(),
    }
}

fn message(error: &str) -> &str {
    error.split_once(": ").map(|(_, m)| m).unwrap_or(error)
}

fn scale_of(case: &Value) -> f64 {
    match case.get("scale") {
        Some(Value::Float(x)) => *x,
        Some(Value::Int(i)) => i.parse().unwrap(),
        Some(Value::Str(s)) => pyutil::float(s).unwrap(),
        other => panic!("{other:?}"),
    }
}

fn str_of(v: Option<&Value>) -> Option<&str> {
    match v {
        Some(Value::Str(s)) => Some(s),
        _ => None,
    }
}

fn read_cases() -> Vec<Value> {
    let text = std::fs::read_to_string(data().join("import/cases.json")).unwrap();
    match loads(&text).unwrap() {
        Value::Array(a) => a,
        v => panic!("{v:?}"),
    }
}

fn matrix_of(case: &Value) -> Option<[f64; 9]> {
    match case.get("matrix") {
        Some(Value::Str(s)) if s == "y_up" => Some(obj::Y_UP_TO_Z_UP),
        Some(Value::Array(a)) => {
            let mut m = [0.0; 9];
            for (k, v) in a.iter().enumerate() {
                m[k] = match v {
                    Value::Float(x) => *x,
                    Value::Int(i) => i.parse().unwrap(),
                    other => panic!("{other:?}"),
                };
            }
            Some(m)
        }
        _ => None,
    }
}

#[test]
fn imports_give_the_signature_python_had() {
    let dir = data().join("import");
    let mut compared = 0;
    let mut skipped = Vec::new();
    let mut errors = 0;
    for case in read_cases() {
        let file = str_of(case.get("file")).unwrap().to_string();
        let kind = str_of(case.get("kind")).unwrap();
        let scale = scale_of(&case);
        let path = dir.join(&file);
        let label = format!("{file} (scale {scale})");
        let mut scene = Scene::new();
        if let Some(unit) = str_of(case.get("unit")) {
            assert_eq!(obj::suggest_unit(&std::fs::read(&path).unwrap()), unit, "{label}: suggested unit");
        }
        let result: Result<(), String> = if kind == "stl" {
            let simplify = Simplify::parse(str_of(case.get("simplify")).unwrap_or("none")).unwrap();
            let data = std::fs::read(&path).unwrap();
            load_stl(&mut scene, &stem(&file), &data, scale, simplify).map(|_| ())
        } else {
            parse_obj(&path, scale, matrix_of(&case).as_ref()).and_then(|m| add_obj_faces(&mut scene, &m, H, None))
        };
        if let Some(want) = str_of(case.get("error")) {
            let got = result.expect_err(&format!("{label}: Python raised {want}"));
            assert_eq!(got, message(want), "{label}: error message");
            errors += 1;
            continue;
        }
        let want_text = std::fs::read_to_string(dir.join(str_of(case.get("sig")).unwrap())).unwrap();
        let want = canon_dumps(&loads(&want_text).unwrap());
        match result {
            Err(e) if e.contains("not ported") => {
                // What Python fuses with code that lives in the edit engine.
                assert!(case.get("fused").is_some() || case.get("simplify").is_some(), "{label}: {e}");
                skipped.push(file);
            }
            Err(e) => panic!("{label}: {e}"),
            Ok(()) => {
                assert!(case.get("fused").is_none(), "{label}: Python fused it");
                assert_eq!(sig_of(&scene), want, "{label}: signature");
                compared += 1;
            }
        }
    }
    assert_eq!(compared, 41, "signatures compared");
    assert_eq!(errors, 25, "errors compared");
    skipped.sort();
    assert_eq!(skipped, ["bigobj.obj", "cube.bin.stl", "parts.obj"]);
}

#[test]
fn part_names_match_python() {
    let text = std::fs::read_to_string(data().join("import/pieces.json")).unwrap();
    let Value::Array(rows) = loads(&text).unwrap() else { panic!() };
    assert_eq!(rows.len(), 19);
    for row in rows {
        let Value::Array(r) = row else { panic!() };
        let s = |i: usize| str_of(r.get(i)).unwrap().to_string();
        let n = r[2].as_i64().unwrap() as usize;
        let key = (s(0), s(1));
        assert_eq!(obj::piece_name(&key, n, "Part {n}"), s(3), "piece name of {key:?} #{n}");
        let tidy_of = if key.0.is_empty() { &key.1 } else { &key.0 };
        assert_eq!(obj::tidy_blender_name(tidy_of), s(4), "tidy name of {tidy_of:?}");
    }
}
