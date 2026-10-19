// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! Rust reads every .igz format as Python does (M2.7): each fixture of
//! `ingetrazo-formats/tests/data/igz` (written by
//! `scripts/golden/make_igz_format_fixtures.py`) has the L2 signature Python
//! computed after reopening it, and the formats of one scene agree.

use std::path::{Path, PathBuf};

use ingetrazo_formats::igz::{self, Zip};
use ingetrazo_formats::igz_geom::expand_document;
use ingetrazo_formats::pyjson::{self, Value};
use ingetrazo_golden::{file_signature, sig::canon_dumps};

fn dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../ingetrazo-formats/tests/data/igz")
}

fn fixtures() -> Vec<PathBuf> {
    let mut v: Vec<PathBuf> = std::fs::read_dir(dir())
        .expect("fixture dir")
        .map(|e| e.expect("entry").path())
        .filter(|p| p.extension().is_some_and(|x| x == "igz"))
        .collect();
    v.sort();
    v
}

#[test]
fn every_fixture_has_the_signature_python_computed() {
    let all = fixtures();
    assert!(all.len() >= 8, "fixtures missing: {}", all.len());
    for path in all {
        let raw = std::fs::read(&path).expect("read");
        let got = file_signature(&raw).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        let want_text = std::fs::read_to_string(path.with_extension("sig.json")).expect("sig.json");
        let want = pyjson::loads(&want_text).expect("sig json");
        assert_eq!(canon_dumps(&got), canon_dumps(&want), "{}", path.display());
    }
}

#[test]
fn the_formats_of_one_scene_agree() {
    let mut by_scene: Vec<(String, Vec<(String, String)>)> = Vec::new();
    for path in fixtures() {
        let stem = path.file_name().unwrap().to_string_lossy().into_owned();
        let (scene, fmt) = stem.trim_end_matches(".igz").rsplit_once('.').expect("name.fN.igz");
        let sig = canon_dumps(&file_signature(&std::fs::read(&path).unwrap()).unwrap());
        match by_scene.iter_mut().find(|(s, _)| s == scene) {
            Some((_, v)) => v.push((fmt.to_string(), sig)),
            None => by_scene.push((scene.to_string(), vec![(fmt.to_string(), sig)])),
        }
    }
    for (scene, variants) in by_scene {
        assert!(variants.len() >= 2, "{scene}: one format only");
        for (fmt, sig) in &variants[1..] {
            assert_eq!(*sig, variants[0].1, "{scene}: {fmt} differs from {}", variants[0].0);
        }
    }
}

#[test]
fn format_three_documents_expand_to_plain_meshes() {
    let mut seen = 0;
    for path in fixtures().into_iter().filter(|p| p.to_string_lossy().contains(".f3.")) {
        let raw = std::fs::read(&path).unwrap();
        let doc = igz::read_document(&raw).unwrap();
        assert_eq!(doc.get("igz_format").and_then(Value::as_i64), Some(3));
        let zip = Zip::open(&raw).unwrap();
        let plain = expand_document(&doc, Some(&zip)).unwrap();
        let text = canon_dumps(&plain);
        assert!(!text.contains("\"geom\""), "{}: a geometry block is left", path.display());
        seen += 1;
    }
    assert!(seen >= 4);
}
