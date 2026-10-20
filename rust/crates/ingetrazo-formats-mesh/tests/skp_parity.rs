// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! The SketchUp import against `formats/skp.py` (scripts/golden/
//! make_skp_fixtures.py): each synthetic `.skp` of `ingetrazo-skp/tests/data`
//! is imported here and the L2 signature of the scene (plus the things the
//! signature only counts: saved views, dimensions, leader texts, the back
//! colour) must be the one Python had after `parse_skp` + `apply_payload`.
//!
//! A file Python cannot read is refused here; one with a re-tinted texture in
//! a format other than PNG or JPEG is refused as `NotPorted`.
//!
//! The fixtures come from a Windows machine: `length()` is MSVC's `hypot`.
//! `real_cases.json` lists models of that machine (not in the repository):
//! those that are there are imported too and compared by hash.

use std::path::{Path, PathBuf};

use ingetrazo_formats::pyjson::{loads, Value};
use ingetrazo_formats_mesh::skp_in::{self, SkpLoadError};
use ingetrazo_golden::model_sig::scene_signature;
use ingetrazo_golden::sig::{canon_dumps, f32hex, Files};
use ingetrazo_math::Hypot;
use ingetrazo_model::scene::{MeshId, Scene};
use ingetrazo_skp::dump::{model_dump, sha256_hex};

const H: Hypot = Hypot::Msvc;

fn data() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../ingetrazo-skp/tests/data")
}

/// A texture cache of its own for each import (the tests run in threads).
fn cache() -> PathBuf {
    static N: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let n = N.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    let d = std::env::temp_dir().join(format!("ingetrazo-skp-parity-{}-{n}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// Every PNG of the texture cache becomes `px:<w>x<h>:<sha256 of its RGBA
/// pixels>`, as `pixelize_textures` of the fixture script does: a re-tinted
/// texture is written by another PNG encoder than Qt's, so the signature can
/// only be compared through the pixels.
fn pixelize(dir: &Path) {
    for entry in std::fs::read_dir(dir).unwrap() {
        let p = entry.unwrap().path();
        if p.is_dir() {
            pixelize(&p);
            continue;
        }
        let data = std::fs::read(&p).unwrap();
        if let Some((w, h, px)) = ingetrazo_formats_mesh::img::rgba8(&data) {
            let raw: Vec<u8> = px.iter().flatten().copied().collect();
            std::fs::write(&p, format!("px:{w}x{h}:{}", sha256_hex(&raw))).unwrap();
        }
    }
}

fn cases(file: &str) -> Vec<Value> {
    match loads(&std::fs::read_to_string(data().join(file)).unwrap()).unwrap() {
        Value::Array(a) => a,
        v => panic!("{v:?}"),
    }
}

fn str_of<'a>(v: &'a Value, k: &str) -> Option<&'a str> {
    match v.get(k) {
        Some(Value::Str(s)) => Some(s),
        _ => None,
    }
}

fn hex3(v: ingetrazo_math::V3) -> Value {
    Value::Array(vec![Value::Str(f32hex(v.0 as f64)), Value::Str(f32hex(v.1 as f64)), Value::Str(f32hex(v.2 as f64))])
}

fn fb(x: f64) -> Value {
    Value::Str(format!("{:016x}", x.to_bits()))
}

/// What `extras_of(scene)` of the fixture script records.
fn extras_of(scene: &Scene) -> Value {
    let views: Vec<Value> = scene.extra.saved_views.iter().map(|v| v.to_dict()).collect();
    let dims: Vec<Value> = (0..scene.dimensions.len())
        .map(|i| {
            let (a, b) = scene.dimension_ends(i, MeshId(0));
            Value::Array(vec![hex3(a), hex3(b), hex3(scene.dimensions[i].offset)])
        })
        .collect();
    let texts: Vec<Value> = scene.text_labels.iter().map(|t| Value::Array(vec![hex3(t.anchor), hex3(t.offset), Value::Str(t.text.clone())])).collect();
    let back = match &scene.extra.back_face_color {
        Some(Value::Array(c)) => Value::Array(c.iter().map(|x| fb(if let Value::Float(f) = x { *f } else { panic!("{x:?}") })).collect()),
        _ => Value::Null,
    };
    Value::Object(vec![
        ("saved_views".into(), Value::Array(views)),
        ("dimensions".into(), Value::Array(dims)),
        ("text_labels".into(), Value::Array(texts)),
        ("back_face_color".into(), back),
        ("groups".into(), Value::int(scene.groups.len() as i64)),
    ])
}

fn sig_text(scene: &Scene) -> String {
    canon_dumps(&scene_signature(scene, &Files { zip: None }))
}

fn import(path: &Path) -> Result<Scene, SkpLoadError> {
    let mut scene = Scene::new();
    let dir = cache();
    skp_in::load_skp(&mut scene, path, &dir, H)?;
    pixelize(&dir);
    Ok(scene)
}

#[test]
#[cfg_attr(not(windows), ignore = "fixtures recorded with the MSVC CRT's libm (sin/cos/atan2/pow differ by an ulp elsewhere)")]
fn imports_give_the_signature_python_had() {
    let (mut compared, mut refused, mut unreadable, mut jpeg_tints) = (0, 0, 0, 0);
    let mut sequences = 0;
    for case in cases("cases.json") {
        let name = str_of(&case, "name").unwrap().to_string();
        if str_of(&case, "kind") == Some("sequence") {
            // files imported one after the other into one scene
            let Some(Value::Array(files)) = case.get("files") else { panic!("{name}") };
            let dir = cache();
            let mut scene = Scene::new();
            for f in files {
                let Value::Str(f) = f else { panic!("{f:?}") };
                skp_in::load_skp(&mut scene, &data().join(format!("{f}.skp")), &dir, H).unwrap_or_else(|e| panic!("{name}: {f}: {e}"));
            }
            pixelize(&dir);
            let want = canon_dumps(&loads(&std::fs::read_to_string(data().join(format!("{name}.sig.json"))).unwrap()).unwrap());
            assert_eq!(sig_text(&scene), want, "{name}: signature");
            assert_eq!(canon_dumps(&extras_of(&scene)), canon_dumps(case.get("extras").unwrap()), "{name}: extras");
            sequences += 1;
            continue;
        }
        let file = str_of(&case, "file").unwrap().to_string();
        let outcome = str_of(case.get("import").unwrap(), "outcome").unwrap().to_string();
        let result = import(&data().join(&file));
        match outcome.as_str() {
            "unreadable" => {
                match result {
                    Err(SkpLoadError::Unreadable(_)) => {}
                    other => panic!("{file}: Python could not read it, got {:?}", other.map(|_| ())),
                }
                unreadable += 1;
            }
            "not_ported" => {
                match result {
                    Err(SkpLoadError::NotPorted(_)) => {}
                    other => panic!("{file}: expected NotPorted, got {:?}", other.map(|_| ())),
                }
                refused += 1;
            }
            "ok" => {
                let scene = result.unwrap_or_else(|e| panic!("{file}: {e}"));
                if matches!(case.get("import").and_then(|i| i.get("tint_from_jpeg")), Some(Value::Bool(true))) {
                    // Qt and zune-jpeg decode a JPEG a level apart
                    jpeg_tints += 1;
                    continue;
                }
                let want = canon_dumps(&loads(&std::fs::read_to_string(data().join(format!("{name}.sig.json"))).unwrap()).unwrap());
                let got = sig_text(&scene);
                if got != want {
                    let at = got.bytes().zip(want.bytes()).position(|(a, b)| a != b).unwrap_or(got.len().min(want.len()));
                    let cut = |s: &str| s[at.saturating_sub(150)..(at + 150).min(s.len())].to_string();
                    panic!("{file}: the signature differs at byte {at} ({} vs {} bytes)\n got: {}\nwant: {}", got.len(), want.len(), cut(&got), cut(&want));
                }
                let extras = canon_dumps(&extras_of(&scene));
                let want = canon_dumps(case.get("extras").unwrap());
                assert_eq!(extras, want, "{file}: extras");
                compared += 1;
            }
            o => panic!("{o}"),
        }
    }
    assert_eq!((compared, refused, unreadable, jpeg_tints, sequences), (31, 0, 11, 0, 3));
}

#[test]
fn the_import_keeps_what_the_file_declares() {
    // A few facts the signatures hold, said plainly.
    let scene = import(&data().join("vff-basic.skp")).unwrap();
    // 9 placements of the 64-face tile: one prototype mesh for the 7 left as
    // they are and one for the 2 painted red as a whole
    let shared: Vec<_> = scene.groups.iter().filter(|g| g.name == "Tile").collect();
    assert_eq!(shared.len(), 9);
    assert!(shared.iter().all(|g| g.xform.is_some()));
    let mut meshes: Vec<_> = shared.iter().map(|g| g.mesh).collect();
    meshes.sort();
    meshes.dedup();
    assert_eq!(meshes.len(), 2);
    assert_eq!(shared.iter().filter(|g| g.mesh == shared[0].mesh).count(), 7);
    assert_eq!(scene.mesh(shared[0].mesh).faces.len(), 64);
    assert_eq!(shared.iter().filter(|g| g.layer.as_deref() == Some("Roof")).count(), 3);
    // the tags of the file join the scene's list, with their visibility
    let layers: Vec<String> = scene.layers.iter().map(canon_dumps).collect();
    assert!(layers.iter().any(|l| l.contains("Roof") && l.contains("false")));
    // the materials of the file are in the registry, named
    assert!(scene.materials.iter().any(|(n, _)| n == "Glass"));
    // the style's back colour
    assert!(scene.extra.back_face_color.is_some());
    // scenes of the .skp become saved views
    let scene = import(&data().join("vff-pages.skp")).unwrap();
    assert_eq!(scene.extra.saved_views.len(), 3);
    assert_eq!(scene.extra.saved_views[1].name, "Scene 2");
    assert_eq!(scene.dimensions.len(), 3);
}

#[test]
fn an_empty_model_is_a_blank_page_and_a_busy_thumbnail_is_not() {
    let mut scene = Scene::new();
    let payload = skp_in::parse_skp(&data().join("vff-empty.skp"), &cache()).unwrap();
    assert!(payload.empty);
    skp_in::apply_payload(&mut scene, payload, H).unwrap();
    assert!(scene.groups.is_empty());
    assert!(matches!(skp_in::parse_skp(&data().join("vff-empty-busy.skp"), &cache()), Err(SkpLoadError::Unreadable(_))));
}

#[test]
fn a_second_import_of_the_same_file_merges_materials_and_layers() {
    // Python: `register` dedups, layers already present are left alone.
    let mut scene = Scene::new();
    skp_in::load_skp(&mut scene, &data().join("legacy-basic.skp"), &cache(), H).unwrap();
    let (m, l) = (scene.materials.len(), scene.layers.len());
    skp_in::load_skp(&mut scene, &data().join("legacy-basic.skp"), &cache(), H).unwrap();
    assert_eq!((scene.materials.len(), scene.layers.len()), (m, l));
    // the model came in twice
    assert!(scene.groups.len() >= 2);
}

#[test]
fn unreadable_input_never_panics() {
    let c = cache();
    for name in ["err-not-skp", "err-empty", "err-truncated-zip", "err-corrupt-zip", "err-legacy-garbage"] {
        assert!(matches!(skp_in::parse_skp(&data().join(format!("{name}.skp")), &c), Err(SkpLoadError::Unreadable(_))), "{name}");
    }
    assert!(matches!(skp_in::parse_skp(&data().join("missing.skp"), &c), Err(SkpLoadError::Unreadable(_))));
    assert!(skp_in::detect_skp(b"PK\x03\x04rest"));
    assert!(skp_in::detect_skp(&std::fs::read(data().join("legacy-basic.skp")).unwrap()));
    assert!(!skp_in::detect_skp(b"hello"));
}

#[test]
fn damaged_models_never_panic_the_importer() {
    // A few bytes of a good file changed at random: when the parse still
    // succeeds the adapter must answer (a payload or an error), not panic.
    let mut state = 11u64;
    let mut next = move || {
        state = state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    };
    let c = cache();
    let (mut adapted, mut refused) = (0, 0);
    for name in ["legacy-components", "legacy-textures", "vff-basic", "vff-pages", "vff-textures", "vff-odd", "vff-drape"] {
        let good = std::fs::read(data().join(format!("{name}.skp"))).unwrap();
        for _ in 0..100 {
            let mut bad = good.clone();
            for _ in 0..1 + next() % 6 {
                let at = (next() as usize) % bad.len();
                bad[at] = next() as u8;
            }
            if let Ok(model) = ingetrazo_skp::parse_bytes(&bad) {
                match skp_in::adapt(&model, "fuzz", &data().join(format!("{name}.skp")), &c) {
                    Ok(_) => adapted += 1,
                    Err(_) => refused += 1,
                }
            }
        }
    }
    assert!(adapted > 0, "{adapted} adapted, {refused} refused");
}

/// Models of the author's machine, by hash (`real_cases.json`). Skipped one
/// by one when the file is not there. Minutes long (the signature of a scene
/// with thousands of components is slow to compute), so it only runs on
/// request: `cargo test --release -p ingetrazo-formats-mesh --test skp_parity
/// -- --ignored`.
#[test]
#[ignore = "models of the author's machine; run with --ignored"]
fn local_models_import_like_python() {
    let list = data().join("real_cases.json");
    if !list.exists() {
        return;
    }
    let (mut checked, mut skipped, mut approximate) = (0, 0, 0);
    for case in cases("real_cases.json") {
        let path = PathBuf::from(str_of(&case, "path").unwrap());
        if !path.exists() {
            skipped += 1;
            continue;
        }
        let name = str_of(&case, "name").unwrap();
        let outcome = str_of(case.get("import").unwrap(), "outcome").unwrap();
        if let Some(_err) = str_of(&case, "parse_error") {
            assert!(ingetrazo_skp::parse_file(&path).is_err(), "{name}: Python failed to parse");
            checked += 1;
            continue;
        }
        let model = ingetrazo_skp::parse_file(&path).unwrap_or_else(|e| panic!("{name} {}: {e}", path.display()));
        let want = str_of(&case, "model_sha").unwrap();
        assert_eq!(sha256_hex(model_dump(&model).text().as_bytes()), want, "{name}: the parse");
        let result = import(&path);
        match outcome {
            "ok" => {
                let scene = result.unwrap_or_else(|e| panic!("{name}: {e}"));
                if matches!(case.get("import").and_then(|i| i.get("tint_from_jpeg")), Some(Value::Bool(true))) {
                    // Qt and zune-jpeg decode a JPEG a level apart: the
                    // re-tinted pixels are not the same
                    approximate += 1;
                    continue;
                }
                assert_eq!(sha256_hex(sig_text(&scene).as_bytes()), str_of(&case, "sig_sha").unwrap(), "{name}: signature");
                assert_eq!(sha256_hex(canon_dumps(&extras_of(&scene)).as_bytes()), str_of(&case, "extras_sha").unwrap(), "{name}: extras");
            }
            "not_ported" => assert!(matches!(result, Err(SkpLoadError::NotPorted(_))), "{name}"),
            _ => assert!(matches!(result, Err(SkpLoadError::Unreadable(_))), "{name}"),
        }
        checked += 1;
    }
    eprintln!("local models: {checked} compared ({approximate} with a JPEG tint, imported but not compared), {skipped} not on this machine");
}
