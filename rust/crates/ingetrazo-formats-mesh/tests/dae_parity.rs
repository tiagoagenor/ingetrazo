// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! COLLADA against `formats/dae.py` (scripts/golden/make_dae_fixtures.py):
//! export is compared byte for byte with what the real `save_dae` wrote (the
//! `<created>` / `<modified>` stamps masked), import by the L2 signature of
//! the scene Python had after reading the same file.
//!
//! The fixtures come from a Windows machine: the file is in text mode, so
//! every line ends in `\r\n`, and `length()` is MSVC's `hypot`. Python's
//! `run_stitch` and `orient_outward` (M3) were replaced by no-ops, so the
//! import is compared up to that point.

use std::path::{Path, PathBuf};

use ingetrazo_formats::igz;
use ingetrazo_formats::pyjson::{loads, Value};
use ingetrazo_formats_mesh::dae::{save_dae, save_dae_text};
use ingetrazo_formats_mesh::dae_in::{load_dae, py_str_hash_seed0};
use ingetrazo_golden::model_sig::scene_signature;
use ingetrazo_golden::sig::{canon_dumps, Files};
use ingetrazo_math::Hypot;
use ingetrazo_model::scene::Scene;

const H: Hypot = Hypot::Msvc;
const MASK: &str = "@@NOW@@";

fn data() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data")
}

fn str_of(v: Option<&Value>) -> Option<&str> {
    match v {
        Some(Value::Str(s)) => Some(s),
        _ => None,
    }
}

fn flag(case: &Value, key: &str) -> bool {
    matches!(case.get(key), Some(Value::Bool(true)))
}

fn read_array(path: &Path) -> Vec<Value> {
    match loads(&std::fs::read_to_string(path).unwrap()).unwrap() {
        Value::Array(a) => a,
        v => panic!("{v:?}"),
    }
}

fn message(error: &str) -> &str {
    error.split_once(": ").map(|(_, m)| m).unwrap_or(error)
}

/// The scene Python exported, opened from its .igz JSON; texture paths (kept
/// relative to `tests/data/export` in the file) pointed there.
fn scene_from(igz_path: &Path) -> Scene {
    let dir = data().join("export");
    let raw = std::fs::read(igz_path).unwrap();
    let doc = igz::read_document(&raw).unwrap();
    let mut scene = Scene::from_document(&doc).unwrap_or_else(|e| panic!("{igz_path:?}: {e:?}"));
    for mesh in &mut scene.meshes {
        for f in mesh.faces.clone() {
            for (k, v) in &mut mesh.face_mut(f).attrs {
                if k == "texture" {
                    if let Value::Object(items) = v {
                        for (tk, tv) in items {
                            if let (true, Value::Str(p)) = (tk == "path", tv) {
                                *p = dir.join(&*p).to_string_lossy().into_owned();
                            }
                        }
                    }
                }
            }
        }
    }
    scene
}

fn exported_scene(entry: &Value) -> Scene {
    let name = str_of(entry.get("scene")).unwrap();
    if flag(entry, "igz") {
        scene_from(&data().join(format!("dae/export/{name}.igz.json")))
    } else {
        scene_from(&data().join(format!("export/{name}.igz.json")))
    }
}

/// Python's text mode on Windows.
fn crlf(text: &str) -> Vec<u8> {
    text.replace('\n', "\r\n").into_bytes()
}

/// `text` with the two stamps replaced by [`MASK`]; asserts they look like
/// `isoformat()` of a UTC time without microseconds.
fn mask_stamps(text: &str) -> String {
    let mut out = text.to_string();
    for tag in ["created", "modified"] {
        let open = format!("<{tag}>");
        let at = out.find(&open).unwrap() + open.len();
        let end = at + out[at..].find(&format!("</{tag}>")).unwrap();
        let stamp = out[at..end].to_string();
        let b = stamp.as_bytes();
        let digit = |i: usize| b[i].is_ascii_digit();
        assert!(
            stamp.len() == 25
                && [0, 1, 2, 3, 5, 6, 8, 9, 11, 12, 14, 15, 17, 18].iter().all(|&i| digit(i))
                && &stamp[4..5] == "-"
                && &stamp[10..11] == "T"
                && &stamp[19..] == "+00:00",
            "stamp {stamp:?}"
        );
        out.replace_range(at..end, MASK);
    }
    out
}

#[test]
fn dae_text_matches_python_byte_for_byte() {
    let mut compared = 0;
    let mut errors = 0;
    for entry in read_array(&data().join("dae/export/index.json")) {
        let scene = exported_scene(&entry);
        let project = str_of(entry.get("project")).unwrap_or("IngeTrazo model");
        if let Some(want) = str_of(entry.get("error")) {
            let got = save_dae_text(&scene, project, MASK, H).expect_err("Python raised");
            assert_eq!(got, message(want));
            errors += 1;
            continue;
        }
        let file = str_of(entry.get("file")).unwrap();
        let export = save_dae_text(&scene, project, MASK, H).unwrap();
        let want = std::fs::read(data().join("dae/export").join(file)).unwrap();
        let want = mask_stamps(&String::from_utf8(want).unwrap().replace("\r\n", "\n"));
        let got = String::from_utf8(crlf(&export.xml)).unwrap().replace("\r\n", "\n");
        if got != want {
            let (a, b): (Vec<&str>, Vec<&str>) = (got.lines().collect(), want.lines().collect());
            let at = a.iter().zip(&b).position(|(x, y)| x != y).unwrap_or(a.len().min(b.len()));
            let cut = |s: &str| s.chars().take(300).collect::<String>();
            panic!("{file} differs at line {at} ({} vs {} lines)\n got: {}\nwant: {}", a.len(), b.len(), cut(a.get(at).unwrap_or(&"")), cut(b.get(at).unwrap_or(&"")));
        }
        // The raw bytes carry \r\n and no final newline.
        let raw = std::fs::read(data().join("dae/export").join(file)).unwrap();
        assert!(raw.windows(2).any(|w| w == b"\r\n"), "{file}: CRLF");
        assert!(!raw.ends_with(b"\n"), "{file}: no final newline");
        compared += 1;
    }
    assert_eq!((compared, errors), (10, 1));
}

#[test]
fn save_dae_writes_the_file_and_copies_the_textures() {
    let tmp = std::env::temp_dir().join(format!("ingetrazo-dae-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp).unwrap();
    let index = read_array(&data().join("dae/export/index.json"));
    for entry in index.iter().filter(|e| matches!(str_of(e.get("scene")), Some("textured" | "geo"))) {
        let name = str_of(entry.get("scene")).unwrap();
        let scene = exported_scene(entry);
        let out = tmp.join(format!("{name}.dae"));
        save_dae(&scene, &out, "IngeTrazo model", MASK, H, true).unwrap();
        let got = std::fs::read(&out).unwrap();
        let want = std::fs::read(data().join(format!("dae/export/{name}.dae"))).unwrap();
        let (got, want) = (String::from_utf8(got).unwrap(), String::from_utf8(want).unwrap());
        assert_eq!(got.replace(MASK, "T"), mask_stamps(&want).replace(MASK, "T"), "{name}.dae");
        for (_, file) in save_dae_text(&scene, "x", MASK, H).unwrap().textures {
            assert_eq!(std::fs::read(tmp.join(&file)).unwrap(), std::fs::read(data().join("dae/export").join(&file)).unwrap(), "copy of {file}");
        }
    }
    let _ = std::fs::remove_dir_all(&tmp);
}

fn sig_of(scene: &Scene) -> String {
    canon_dumps(&scene_signature(scene, &Files { zip: None }))
}

fn float_of(v: Option<&Value>) -> f64 {
    match v {
        Some(Value::Float(x)) => *x,
        Some(Value::Int(i)) => i.parse().unwrap(),
        other => panic!("{other:?}"),
    }
}

#[test]
fn imports_give_the_signature_python_had() {
    let dir = data().join("dae/import");
    let (mut compared, mut errors, mut xml_errors) = (0, 0, 0);
    let mut skipped = Vec::new();
    for case in read_array(&dir.join("cases.json")) {
        let file = str_of(case.get("file")).unwrap().to_string();
        let mut scene = Scene::new();
        let result = load_dae(&mut scene, &dir.join(&file), H, None);
        if let Some(want) = str_of(case.get("error")) {
            let got = result.expect_err(&format!("{file}: Python raised {want}"));
            if flag(&case, "xml_error") {
                // The text of an XML syntax error is expat's.
                xml_errors += 1;
            } else {
                assert_eq!(got, message(want), "{file}: error message");
                errors += 1;
            }
            continue;
        }
        let want_text = std::fs::read_to_string(dir.join(str_of(case.get("sig")).unwrap())).unwrap();
        let want = canon_dumps(&loads(&want_text).unwrap());
        match result {
            Err(e) if e.starts_with("not ported") => {
                assert!(flag(&case, "fused") || flag(&case, "qimage"), "{file}: {e}");
                skipped.push(file);
            }
            Err(e) => panic!("{file}: {e}"),
            Ok(()) => {
                assert!(!flag(&case, "fused") && !flag(&case, "qimage"), "{file}: Python fused it or needed Qt images");
                assert_eq!(sig_of(&scene), want, "{file}: signature");
                match case.get("georef") {
                    Some(g) => {
                        let d = scene.extra.georef_datum.as_ref().unwrap_or_else(|| panic!("{file}: georef"));
                        for k in ["lat", "lon", "alt"] {
                            assert_eq!(float_of(d.get(k)).to_bits(), float_of(g.get(k)).to_bits(), "{file}: georef {k}");
                        }
                    }
                    None => assert!(scene.extra.georef_datum.is_none(), "{file}: georef"),
                }
                compared += 1;
            }
        }
    }
    skipped.sort();
    assert_eq!(skipped, ["../export/bulk.dae", "../export/weld.dae", "big.dae", "limit401.dae", "panel.dae", "sprite-child.dae", "sprite.dae", "texture-badindex.dae", "texture-degenerate.dae", "texture-nouv.dae"]);
    assert_eq!((compared, errors, xml_errors), (48, 12, 3));
}

#[test]
fn the_stitch_hook_gets_the_new_faces_and_their_corners() {
    let path = data().join("dae/import/polygons.dae");
    let mut scene = Scene::new();
    let mut calls = 0;
    let mut seen = (0, 0);
    let mut hook = |_: &mut ingetrazo_model::mesh::Mesh, seed: &std::collections::HashSet<ingetrazo_model::mesh::Key>, faces: &std::collections::HashSet<ingetrazo_model::mesh::FaceId>| {
        calls += 1;
        seen = (seed.len(), faces.len());
    };
    load_dae(&mut scene, &path, H, Some(&mut hook)).unwrap();
    assert_eq!(calls, 1);
    // 9 distinct corners; the quad, the triangle and the pentagon.
    assert_eq!(seen, (9, 3));
    // A model Python fuses never reaches the hook.
    let mut called = false;
    let mut hook = |_: &mut ingetrazo_model::mesh::Mesh, _: &std::collections::HashSet<ingetrazo_model::mesh::Key>, _: &std::collections::HashSet<ingetrazo_model::mesh::FaceId>| called = true;
    assert!(load_dae(&mut Scene::new(), &data().join("dae/import/big.dae"), H, Some(&mut hook)).is_err());
    assert!(!called);
}

#[test]
fn python_string_hash_without_salt() {
    // `hash(s)` under PYTHONHASHSEED=0, from CPython 3.12.
    let text = std::fs::read_to_string(data().join("dae/import/hashes.txt")).unwrap();
    let mut n = 0;
    for line in text.lines() {
        let (h, s) = line.split_once(' ').unwrap();
        assert_eq!(py_str_hash_seed0(s), h.parse::<i64>().unwrap(), "hash of {s:?}");
        n += 1;
    }
    assert!(n >= 8);
}
