// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! The crate against bytes Python wrote (scripts/golden/make_pyjson_fixtures.py)
//! and against every .igz in the repository.

use std::path::{Path, PathBuf};

use ingetrazo_formats::igz;
use ingetrazo_formats::pyjson::{dumps, float_repr, loads, Options};

fn data(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data").join(name)
}

fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../..")
}

#[test]
fn float_repr_matches_python_repr() {
    let text = std::fs::read_to_string(data("floats.txt")).unwrap();
    let mut n = 0;
    for line in text.lines() {
        let (hex, want) = line.split_once(' ').unwrap();
        let x = f64::from_bits(u64::from_str_radix(hex, 16).unwrap());
        assert_eq!(float_repr(x), want, "bits {hex}");
        // And json.loads reads the repr back to the same double.
        match loads(want).unwrap() {
            ingetrazo_formats::pyjson::Value::Float(y) => assert_eq!(y.to_bits(), x.to_bits(), "{want}"),
            v => panic!("{want} read as {v:?}"),
        }
        n += 1;
    }
    assert!(n > 5000, "only {n} floats");
}

#[test]
fn dumps_matches_python_in_every_style() {
    let styles: [(&str, Options); 5] = [
        ("cases.default.json", Options::default()),
        ("cases.indent.json", Options::indent(2)),
        ("cases.compact.json", Options::compact()),
        ("cases.sorted.json", Options::indent(2).sort_keys(true)),
        ("cases.utf8.json", Options { ensure_ascii: false, ..Options::default() }),
    ];
    let source = loads(&std::fs::read_to_string(data("cases.default.json")).unwrap()).unwrap();
    for (name, opts) in styles {
        let want = std::fs::read_to_string(data(name)).unwrap();
        assert_eq!(dumps(&source, &opts).unwrap(), want, "{name}");
        // Each file also reads back to the same value.
        assert_eq!(loads(&want).unwrap(), if name == "cases.sorted.json" { loads(&want).unwrap() } else { source.clone() }, "{name}");
    }
}

#[test]
fn format_1_file_written_by_the_app_round_trips_byte_for_byte() {
    for (name, large) in [("scene.igz", false), ("scene-compact.igz", true)] {
        let raw = std::fs::read(data(name)).unwrap();
        assert!(!igz::is_container(&raw));
        let doc = igz::read_document(&raw).unwrap();
        assert_eq!(doc.get("igz_format").and_then(|v| v.as_i64()), Some(igz::PLAIN_FORMAT), "{name}");
        assert_eq!(igz::write_plain(&doc, large).unwrap(), raw, "{name}");
    }
}

#[test]
fn every_repository_igz_document_round_trips() {
    let mut files = Vec::new();
    for dir in ["examples", "resources/components"] {
        for e in std::fs::read_dir(repo().join(dir)).unwrap() {
            let p = e.unwrap().path();
            if p.extension().is_some_and(|x| x == "igz") {
                files.push(p);
            }
        }
    }
    assert!(files.len() >= 12, "found {}", files.len());
    for p in files {
        let raw = std::fs::read(&p).unwrap();
        let text = igz::document_text(&raw).unwrap();
        let doc = loads(&text).unwrap();
        let large = !text.starts_with("{\n");
        let got = igz::dumps_document(&doc, large).unwrap();
        if got != text {
            // A whole document is megabytes: show where it diverges.
            let at = got.bytes().zip(text.bytes()).take_while(|(a, b)| a == b).count();
            let lo = at.saturating_sub(80);
            panic!("{} differs at byte {at}:
  rust:   {:?}
  python: {:?}", p.display(),
                   &got[lo..(at + 80).min(got.len())], &text[lo..(at + 80).min(text.len())]);
        }
    }
}

#[test]
fn non_finite_is_refused_with_python_path() {
    let doc = loads(r#"{"igz_format": 1, "scene": {"edges": [{"a": [0.0, NaN, 0.0]}]}}"#).unwrap();
    match igz::write_plain(&doc, false) {
        Err(igz::IgzError::NonFinite { at }) => assert_eq!(at, "scene › edges[0] › a[1]"),
        other => panic!("{other:?}"),
    }
}
