// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! M2.9: every `.igz` of the repository, opened and saved again, is what the
//! Python app writes for the same open and save
//! (scripts/golden/make_roundtrip_fixtures.py -> data/roundtrip_corpus.json).
//!
//! Compared per file: the inner `document.json` byte for byte (its hash, and
//! its text when the fixture keeps it), the container's members (names,
//! order, sizes, bytes) and, where Python's output is deterministic, the
//! whole file. The one pinned nondeterminism: a document with no group uids
//! (the shipped examples) gets random ones from `core.group.new_uid` at every
//! open, so for those files the uids are blanked on both sides and the
//! whole-file hash is not compared.

use std::path::{Path, PathBuf};

use ingetrazo_formats::igz::{self, Zip};
use ingetrazo_formats::igz_geom::expand_document;
use ingetrazo_formats::pyjson::{self, Value};
use ingetrazo_model::igz_write::{save_scene, SaveOptions};
use ingetrazo_model::scene::Scene;
use sha2::{Digest, Sha256};

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../..")
}

fn sha(b: &[u8]) -> String {
    Sha256::digest(b).iter().map(|x| format!("{x:02x}")).collect()
}

/// What git's autocrlf may have done to a plain JSON file of the working tree.
fn lf(raw: &[u8]) -> Vec<u8> {
    if igz::is_container(raw) {
        raw.to_vec()
    } else {
        String::from_utf8_lossy(raw).replace("\r\n", "\n").into_bytes()
    }
}

/// `"uid": "<16 hex>"` -> `"uid":"~"`, as the generator blanks it.
fn blank_uids(doc: &[u8]) -> Vec<u8> {
    let key = b"\"uid\":";
    let mut out = Vec::with_capacity(doc.len());
    let mut i = 0;
    while i < doc.len() {
        if doc[i..].starts_with(key) {
            let mut j = i + key.len();
            while j < doc.len() && doc[j] == b' ' {
                j += 1;
            }
            let hex = doc.get(j + 1..j + 17);
            if doc.get(j) == Some(&b'"')
                && hex.is_some_and(|h| h.iter().all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(c)))
                && doc.get(j + 17) == Some(&b'"')
            {
                out.extend_from_slice(b"\"uid\":\"~\"");
                i = j + 18;
                continue;
            }
        }
        out.push(doc[i]);
        i += 1;
    }
    out
}

/// Open `raw` as the Rust app does and save it again.
fn resave(raw: &[u8], version: &str) -> Vec<u8> {
    let doc = igz::read_document(raw).expect("document");
    let zip = igz::is_container(raw).then(|| Zip::open(raw).expect("zip"));
    let expanded = expand_document(&doc, zip.as_ref()).expect("expand");
    let scene = Scene::from_document(&expanded).expect("scene");
    let read = |src: &str| zip.as_ref().and_then(|z| z.read(src).ok());
    let opts = SaveOptions::new(version, &read);
    save_scene(&scene, &opts).expect("save").bytes
}

fn str_of<'a>(v: &'a Value, key: &str) -> &'a str {
    match v.get(key) {
        Some(Value::Str(s)) => s,
        _ => "",
    }
}

fn compare(name: &str, want: &Value, got: &[u8]) -> Result<(), String> {
    let random_uids = matches!(want.get("random_uids"), Some(Value::Bool(true)));
    let blank = |n: &str, d: Vec<u8>| if random_uids && n == "document.json" { blank_uids(&d) } else { d };
    let container = igz::is_container(got);
    let mut parts: Vec<(String, Vec<u8>)> = Vec::new();
    if container {
        let z = Zip::open(got).map_err(|e| format!("{name}: {e:?}"))?;
        for e in z.entries() {
            parts.push((e.name.clone(), blank(&e.name, z.read(&e.name).map_err(|e| format!("{name}: {e:?}"))?)));
        }
    } else {
        parts.push(("document.json".into(), blank("document.json", got.to_vec())));
    }
    match (want.get("entries"), container) {
        (Some(Value::Array(es)), true) => {
            let wn: Vec<&str> = es.iter().map(|e| if let Value::Array(a) = e { if let Some(Value::Str(s)) = a.first() { s.as_str() } else { "" } } else { "" }).collect();
            let gn: Vec<&str> = parts.iter().map(|(n, _)| n.as_str()).collect();
            if wn != gn {
                return Err(format!("{name}: members {gn:?} vs python {wn:?}"));
            }
            for (e, (n, d)) in es.iter().zip(&parts) {
                let Value::Array(a) = e else { unreachable!() };
                let size = a.get(1).and_then(Value::as_i64).unwrap_or(-1);
                let hash = match a.get(2) {
                    Some(Value::Str(s)) => s.as_str(),
                    _ => "",
                };
                if size != d.len() as i64 || hash != sha(d) {
                    return Err(format!("{name}: member {n} differs ({} vs python {size} bytes)", d.len()));
                }
            }
        }
        (Some(Value::Null) | None, false) => {}
        (_, c) => return Err(format!("{name}: container {c} but python's is not")),
    }
    let doc = &parts.iter().find(|(n, _)| n == "document.json").ok_or_else(|| format!("{name}: no document"))?.1;
    if let Some(Value::Str(text)) = want.get("document") {
        if text.as_bytes() != doc.as_slice() {
            let at = text.bytes().zip(doc.iter()).position(|(a, b)| a != *b).unwrap_or(text.len().min(doc.len()));
            return Err(format!("{name}: document differs at byte {at}\n  rust:   {}\n  python: {}", String::from_utf8_lossy(&doc[at.saturating_sub(40)..(at + 40).min(doc.len())]), &text[at.saturating_sub(40)..(at + 40).min(text.len())]));
        }
    }
    if str_of(want, "document_sha256") != sha(doc) {
        return Err(format!("{name}: document.json differs from python's ({} bytes, python {})", doc.len(), want.get("document_size").and_then(Value::as_i64).unwrap_or(-1)));
    }
    if !random_uids && str_of(want, "sha256") != sha(got) {
        return Err(format!("{name}: file bytes differ ({} vs python {})", got.len(), want.get("size").and_then(Value::as_i64).unwrap_or(-1)));
    }
    Ok(())
}

#[test]
fn every_igz_of_the_repository_round_trips_like_python() {
    let text = std::fs::read_to_string(root().join("rust/crates/ingetrazo-model/tests/data/roundtrip_corpus.json")).unwrap();
    let corpus = pyjson::loads(&text).unwrap();
    let version = str_of(&corpus, "app_version").to_string();
    let Some(Value::Object(files)) = corpus.get("files") else { panic!("no files") };
    assert!(files.len() >= 250, "{} files", files.len());
    let mut bad = Vec::new();
    for (rel, want) in files {
        let raw = lf(&std::fs::read(root().join(rel)).unwrap_or_else(|e| panic!("{rel}: {e}")));
        let got = resave(&raw, &version);
        // ROUNDTRIP_DUMP=<dir>: keep what Rust wrote, to diff a failure by hand.
        if let Some(dir) = std::env::var_os("ROUNDTRIP_DUMP") {
            std::fs::write(Path::new(&dir).join(rel.replace('/', "__")), &got).unwrap();
        }
        if let Err(e) = compare(rel, want, &got) {
            bad.push(e);
        }
    }
    assert!(bad.is_empty(), "{} of {} differ from python's:\n{}", bad.len(), files.len(), bad.join("\n"));
}

/// The corpus lists every `.igz` there is: a new file must be added to the
/// fixture (rerun the generator), not slip past the test.
#[test]
fn the_corpus_covers_every_igz() {
    fn walk(dir: &Path, out: &mut Vec<String>, base: &Path) {
        let Ok(rd) = std::fs::read_dir(dir) else { return };
        for e in rd.flatten() {
            let p = e.path();
            let n = e.file_name().to_string_lossy().into_owned();
            if p.is_dir() {
                // `benchmarks/models` holds the generated city models, ignored by git.
                let generated = n == "models" && dir.ends_with("benchmarks");
                if !generated && !matches!(n.as_str(), ".venv" | "target" | ".git" | ".claude" | "node_modules" | "vendor") {
                    walk(&p, out, base);
                }
            } else if n.ends_with(".igz") {
                out.push(p.strip_prefix(base).unwrap().to_string_lossy().replace('\\', "/"));
            }
        }
    }
    let base = root().canonicalize().unwrap();
    let mut found = Vec::new();
    walk(&base, &mut found, &base);
    found.sort();
    let text = std::fs::read_to_string(base.join("rust/crates/ingetrazo-model/tests/data/roundtrip_corpus.json")).unwrap();
    let corpus = pyjson::loads(&text).unwrap();
    let Some(Value::Object(files)) = corpus.get("files") else { panic!("no files") };
    let mut listed: Vec<&str> = files.iter().map(|(k, _)| k.as_str()).collect();
    listed.sort();
    assert_eq!(found.iter().map(String::as_str).collect::<Vec<_>>(), listed);
}
