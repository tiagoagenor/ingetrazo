// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! glTF/GLB against `formats/gltf.py` and `formats/glb.py`
//! (scripts/golden/make_gltf_fixtures.py): export is compared byte for byte
//! with what the real exporter wrote, import by the L2 signature of the
//! scene Python had after reading the same file (or the message it raised).
//!
//! The fixtures come from a Windows machine: `length()` is MSVC's `hypot`
//! and NumPy is the OpenBLAS build of the main virtual environment.

use std::path::{Path, PathBuf};

use ingetrazo_formats::igz;
use ingetrazo_formats::pyjson::{loads, Value};
use ingetrazo_formats_mesh::glb::{det3, load_glb, solve3};
use ingetrazo_formats_mesh::gltf::{save_glb_bytes, Labels};
use ingetrazo_golden::model_sig::scene_signature;
use ingetrazo_golden::sig::{canon_dumps, Files};
use ingetrazo_math::Hypot;
use ingetrazo_model::scene::Scene;

const H: Hypot = Hypot::Msvc;
/// The scenes of the OBJ/STL fixtures (their .igz lives in `export/`) and the
/// three of this format.
const OLD: [&str; 7] = ["faces", "textured", "components", "bulk", "weld", "classic", "empty"];
const NEW: [&str; 3] = ["figures", "hierarchy", "finishes"];

fn data() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data")
}

/// The scene Python exported, opened from its .igz JSON; texture paths (kept
/// relative in the file) pointed at the fixture folder.
fn scene_of(name: &str) -> Scene {
    let dir = if OLD.contains(&name) { data().join("export") } else { data().join("gltf") };
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
                                    if !p.is_empty() {
                                        *p = dir.join(&*p).to_string_lossy().into_owned();
                                    }
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

/// The JSON chunk of a GLB as text.
fn json_chunk(glb: &[u8]) -> String {
    let len = u32::from_le_bytes([glb[12], glb[13], glb[14], glb[15]]) as usize;
    String::from_utf8_lossy(&glb[20..20 + len]).into_owned()
}

#[test]
fn glb_bytes_match_python() {
    for name in OLD.iter().chain(NEW.iter()) {
        let scene = scene_of(name);
        let got = save_glb_bytes(&scene, H, &Labels::default()).unwrap_or_else(|e| panic!("{name}: {e}"));
        let want = std::fs::read(data().join("gltf").join(format!("{name}.glb"))).unwrap();
        if got != want {
            let (gj, wj) = (json_chunk(&got), json_chunk(&want));
            if gj != wj {
                let at = gj.bytes().zip(wj.bytes()).position(|(a, b)| a != b).unwrap_or(gj.len().min(wj.len()));
                let lo = at.saturating_sub(120);
                panic!(
                    "{name}.glb: JSON differs at {at}\n got: {}\nwant: {}",
                    &gj[lo..(at + 160).min(gj.len())],
                    &wj[lo..(at + 160).min(wj.len())]
                );
            }
            let at = got.iter().zip(&want).position(|(a, b)| a != b).unwrap_or(got.len().min(want.len()));
            panic!("{name}.glb: binary differs at byte {at} (lengths {} and {})", got.len(), want.len());
        }
    }
}

#[test]
fn an_empty_scene_still_writes_a_file() {
    let got = save_glb_bytes(&Scene::new(), H, &Labels::default()).unwrap();
    assert_eq!(got, std::fs::read(data().join("gltf/empty.glb")).unwrap());
}

// ---- import ------------------------------------------------------------------------

fn sig_of(scene: &Scene) -> String {
    canon_dumps(&scene_signature(scene, &Files { zip: None }))
}

fn str_of(v: Option<&Value>) -> Option<&str> {
    match v {
        Some(Value::Str(s)) => Some(s),
        _ => None,
    }
}

/// The messages Python (not NumPy or `json`) wrote itself, plus the NumPy one
/// every empty batch raises: compared word for word. Every other error is
/// only required to be one.
const OWN_MESSAGES: [&str; 5] = [
    "ValueError: No triangle geometry found in the glTF file",
    "ValueError: sparse accessors are not supported",
    "ValueError: GLB without a JSON chunk",
    "ValueError: Draco-compressed GLB \u{2014} re-export/download uncompressed",
    "ValueError: zero-size array to reduction operation minimum which has no identity",
];

#[test]
fn imports_give_the_signature_python_had() {
    let dir = data().join("gltf/import");
    let cache = std::env::temp_dir().join(format!("ingetrazo-glb-test-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&cache);
    let text = std::fs::read_to_string(dir.join("cases.json")).unwrap();
    let Value::Array(cases) = loads(&text).unwrap() else { panic!() };
    let (mut compared, mut errors) = (0, 0);
    for case in &cases {
        let file = str_of(case.get("file")).unwrap();
        let mut scene = Scene::new();
        let result = load_glb(&mut scene, &dir.join(file), &cache, H);
        if let Some(want) = str_of(case.get("error")) {
            let got = result.expect_err(&format!("{file}: Python raised {want}"));
            if OWN_MESSAGES.contains(&want) {
                let got = got.strip_prefix("ValueError: ").unwrap_or(&got);
                assert_eq!(want.split_once(": ").unwrap().1, got, "{file}: error message");
            }
            errors += 1;
            continue;
        }
        result.unwrap_or_else(|e| panic!("{file}: {e}"));
        let want = std::fs::read_to_string(dir.join(str_of(case.get("sig")).unwrap())).unwrap();
        let want = canon_dumps(&loads(&want).unwrap());
        let got = sig_of(&scene);
        if got != want {
            let at = got.bytes().zip(want.bytes()).position(|(a, b)| a != b).unwrap_or(got.len().min(want.len()));
            let lo = at.saturating_sub(150);
            panic!(
                "{file}: signature differs at {at}\n got: {}\nwant: {}",
                &got[lo..(at + 200).min(got.len())],
                &want[lo..(at + 200).min(want.len())]
            );
        }
        compared += 1;
    }
    let _ = std::fs::remove_dir_all(&cache);
    assert_eq!(compared, 62, "signatures compared");
    assert_eq!(errors, 24, "errors compared");
}

fn bits(s: &str) -> f64 {
    f64::from_bits(u64::from_str_radix(s, 16).unwrap())
}

#[test]
fn the_three_by_three_solve_is_numpys_to_the_last_bit() {
    let text = std::fs::read_to_string(data().join("gltf/solve.txt")).unwrap();
    let mut n = 0;
    for line in text.lines() {
        let v: Vec<f64> = line.split(' ').map(bits).collect();
        let a = [[v[0], v[1], v[2]], [v[3], v[4], v[5]], [v[6], v[7], v[8]]];
        let x = solve3(&a, [v[9], v[10], v[11]]);
        for k in 0..3 {
            assert_eq!(x[k].to_bits(), v[12 + k].to_bits(), "system {n}, unknown {k}: {a:?}");
        }
        let det = det3(&a);
        let want = v[15];
        assert!((det - want).abs() <= 1e-9 * want.abs().max(1e-300), "system {n}: det {det} vs {want}");
        n += 1;
    }
    assert!(n > 1400, "{n}");
}
