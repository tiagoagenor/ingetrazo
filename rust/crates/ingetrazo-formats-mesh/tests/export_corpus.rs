// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! M5.8: every exporter against the real one over every `.igz` of the
//! repository (examples, shipped components, golden start scenes, the
//! fixtures of the other crates). Python opened each document and exported
//! it as the File menu does (scripts/golden/make_export_corpus.py ->
//! data/export_corpus.json): OBJ (+ MTL and the textures it copies), STL in
//! each unit of the dialog, COLLADA (its two time stamps masked), GLB, whole
//! and "only the selection", and the DXF writer fed with the top-view outline
//! of every face. The Rust side opens the same document and
//! must write the very bytes: the fixture holds a size and a sha256 per file.
//!
//! The fixtures come from a Windows machine: text files end in `\r\n`
//! (Python's text mode) and `length()` is MSVC's `hypot`.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use ingetrazo_formats::igz::{self, Zip};
use ingetrazo_formats::igz_geom::expand_document;
use ingetrazo_formats::pyjson::{loads, Value};
use ingetrazo_formats_mesh::dae::save_dae;
use ingetrazo_formats_mesh::dxf::export::{save_dxf_layers, LayerGroup};
use ingetrazo_formats_mesh::gltf::{save_glb_bytes_only, Labels, Only};
use ingetrazo_formats_mesh::obj::save_obj;
use ingetrazo_formats_mesh::stl::save_stl_bytes;
use ingetrazo_formats_mesh::world::world_faces;
use ingetrazo_math::Hypot;
use ingetrazo_model::igz_write::sha1_hex;
use ingetrazo_model::scene::Scene;
use ingetrazo_model::texture::texture_file_name;
use sha2::{Digest, Sha256};

const H: Hypot = Hypot::Msvc;
const UNITS: [&str; 5] = ["mm", "cm", "m", "in", "ft"];
const STAMP: &str = "T";

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

/// `_unpack_textures`: each embedded image goes to a file named as the
/// app's texture cache names it (`<sha1[:16]>-<safe name>`) and the entry
/// points at it.
fn unpack_textures(scene: &mut Scene, zip: &Zip, dir: &Path) {
    fn walk(v: &mut Value, zip: &Zip, dir: &Path) {
        match v {
            Value::Object(items) => {
                for (k, x) in items.iter_mut() {
                    match x {
                        Value::Str(member) if (k == "path" || k == "base") && member.starts_with("textures/") => {
                            if let Ok(data) = zip.read(member) {
                                let name = member.rsplit('/').next().unwrap_or(member);
                                let file = dir.join(format!("{}-{}", &sha1_hex(&data)[..16], texture_file_name(name)));
                                if !file.exists() {
                                    std::fs::write(&file, &data).unwrap();
                                }
                                *member = file.to_string_lossy().into_owned();
                            }
                        }
                        other => walk(other, zip, dir),
                    }
                }
            }
            Value::Array(a) => a.iter_mut().for_each(|x| walk(x, zip, dir)),
            _ => {}
        }
    }
    for mesh in &mut scene.meshes {
        for f in mesh.faces.clone() {
            for (_, v) in &mut mesh.face_mut(f).attrs {
                walk(v, zip, dir);
            }
        }
    }
    for (_, m) in &mut scene.materials {
        walk(m, zip, dir);
    }
}

fn open(rel: &str, cache: &Path) -> Scene {
    let raw = lf(&std::fs::read(root().join(rel)).unwrap());
    let doc = igz::read_document(&raw).unwrap_or_else(|e| panic!("{rel}: {e:?}"));
    let zip = igz::is_container(&raw).then(|| Zip::open(&raw).unwrap());
    let expanded = expand_document(&doc, zip.as_ref()).unwrap_or_else(|e| panic!("{rel}: {e:?}"));
    let mut scene = Scene::from_document(&expanded).unwrap_or_else(|e| panic!("{rel}: {e:?}"));
    if let Some(z) = &zip {
        unpack_textures(&mut scene, z, cache);
    }
    scene
}

/// `{file name: [size, sha256]}` of every file in `dir`, as the fixture has it.
fn files_of(dir: &Path, mask_stamps: bool) -> BTreeMap<String, (usize, String)> {
    let mut out = BTreeMap::new();
    for e in std::fs::read_dir(dir).unwrap().flatten() {
        let mut data = std::fs::read(e.path()).unwrap();
        let name = e.file_name().to_string_lossy().into_owned();
        if mask_stamps && name.ends_with(".dae") {
            data = mask(&data);
        }
        out.insert(name, (data.len(), sha(&data)));
    }
    out
}

/// `<created>…</created>` and `<modified>…</modified>` -> the same with `T`.
fn mask(data: &[u8]) -> Vec<u8> {
    let mut text = String::from_utf8(data.to_vec()).unwrap();
    for tag in ["created", "modified"] {
        let (open, close) = (format!("<{tag}>"), format!("</{tag}>"));
        let at = text.find(&open).unwrap() + open.len();
        let end = at + text[at..].find(&close).unwrap();
        text.replace_range(at..end, "T");
    }
    text.into_bytes()
}

type Written = Result<BTreeMap<String, (usize, String)>, String>;

fn want_of(v: &Value) -> Written {
    if let Some(Value::Str(e)) = v.get("error") {
        return Err(e.clone());
    }
    let Value::Object(items) = v else { panic!("{v:?}") };
    Ok(items
        .iter()
        .map(|(name, pair)| {
            let Value::Array(p) = pair else { panic!("{pair:?}") };
            let Value::Str(h) = &p[1] else { panic!("{p:?}") };
            (name.clone(), (p[0].as_i64().unwrap() as usize, h.clone()))
        })
        .collect())
}

fn single(name: &str, bytes: &[u8]) -> Written {
    Ok(BTreeMap::from([(name.to_string(), (bytes.len(), sha(bytes)))]))
}

/// What Python selects for "only the selection": the even-numbered loose
/// faces and top-level groups.
fn selection(scene: &Scene) -> Only {
    Only { faces: scene.loose().faces.iter().step_by(2).copied().collect(), groups: (0..scene.groups.len()).step_by(2).collect() }
}

/// `make_export_corpus.dxf_groups`: the outline segments of every face per
/// layer, and the hole rings on a dashed layer.
fn dxf_groups(scene: &Scene) -> Vec<LayerGroup> {
    fn ring(pts: &[ingetrazo_math::V3]) -> Vec<[f64; 4]> {
        let n = pts.len();
        (0..n).map(|i| [pts[i].0 as f64, pts[i].1 as f64, pts[(i + 1) % n].0 as f64, pts[(i + 1) % n].1 as f64]).collect()
    }
    let mut layers: Vec<(String, Vec<[f64; 4]>)> = Vec::new();
    let mut holes: Vec<[f64; 4]> = Vec::new();
    for face in world_faces(scene) {
        let name = match face.attr("layer") {
            Some(Value::Str(s)) if !s.is_empty() => s.clone(),
            _ => "VISTA".to_string(),
        };
        let at = match layers.iter().position(|(n, _)| *n == name) {
            Some(i) => i,
            None => {
                layers.push((name, Vec::new()));
                layers.len() - 1
            }
        };
        layers[at].1.extend(ring(&face.outer));
        for h in &face.holes {
            holes.extend(ring(h));
        }
    }
    let mut groups: Vec<LayerGroup> = layers.into_iter().map(|(n, s)| LayerGroup::new(&n, s)).collect();
    groups.push(LayerGroup::dashed("agujeros", holes));
    groups
}

fn describe(a: &Written) -> String {
    match a {
        Err(e) => format!("raised {e}"),
        Ok(m) => format!("{} files, {:?}", m.len(), m.iter().map(|(k, v)| format!("{k}:{}", v.0)).collect::<Vec<_>>()),
    }
}

/// One document: every output Python made for it, compared. Returns how many
/// matched and a line for each that did not.
fn check_document(rel: &str, outputs: &Value, tmp: &Path, cache: &Path, doc_no: usize) -> (usize, Vec<String>) {
    let scene = open(rel, cache);
    let Value::Object(outputs) = outputs else { panic!() };
    let (mut compared, mut bad) = (0usize, Vec::<String>::new());
    let mut scratch = 0;
    for (what, want) in outputs {
        let want = want_of(want);
        let mut fresh = || {
            scratch += 1;
            let d = tmp.join(format!("d{doc_no}-w{scratch}"));
            std::fs::create_dir_all(&d).unwrap();
            d
        };
        let got: Written = match what.as_str() {
            "obj" => {
                let d = fresh();
                save_obj(&scene, &d.join("model.obj"), H, true).map(|_| files_of(&d, false)).map_err(|e| e.to_string())
            }
            "dae" => {
                let d = fresh();
                save_dae(&scene, &d.join("model.dae"), "IngeTrazo model", STAMP, H, true).map(|_| files_of(&d, true))
            }
            "dxf" => {
                let d = fresh();
                save_dxf_layers(&d.join("model.dxf"), &dxf_groups(&scene), true).map(|_| files_of(&d, false)).map_err(|e| e.to_string())
            }
            "glb" => save_glb_bytes_only(&scene, H, &Labels::default(), None).and_then(|b| single("model.glb", &b)),
            "glb.only" => save_glb_bytes_only(&scene, H, &Labels::default(), Some(&selection(&scene))).and_then(|b| single("model.glb", &b)),
            s if s.starts_with("stl.") => {
                let unit = UNITS.iter().find(|u| s[4..] == ***u).unwrap_or_else(|| panic!("unit {s}"));
                save_stl_bytes(&scene, unit, H).and_then(|b| single("model.stl", &b))
            }
            other => panic!("{rel}: unknown output {other}"),
        };
        // A Python exception is compared by its class: the messages of an OS
        // error and Rust's differ.
        let same = match (&want, &got) {
            (Err(w), Err(g)) => w.split(':').next() == g.split(':').next() || w == g,
            (a, b) => a == b,
        };
        if same {
            compared += 1;
        } else {
            bad.push(format!("{rel} {what}: python {} / rust {}", describe(&want), describe(&got)));
        }
    }
    (compared, bad)
}

#[test]
fn every_exporter_matches_python_over_the_corpus() {
    let text = std::fs::read_to_string(root().join("rust/crates/ingetrazo-formats-mesh/tests/data/export_corpus.json")).unwrap();
    let corpus = loads(&text).unwrap();
    let Some(Value::Object(files)) = corpus.get("files") else { panic!("no files") };
    assert!(files.len() >= 250, "{} documents", files.len());
    let tmp = std::env::temp_dir().join(format!("ingetrazo-export-corpus-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&tmp);
    let cache = tmp.join("cache");
    std::fs::create_dir_all(&cache).unwrap();
    // The documents are independent: spread them over the cores.
    let workers = std::thread::available_parallelism().map_or(1, |n| n.get()).min(8);
    let (mut compared, mut bad) = (0usize, Vec::<String>::new());
    std::thread::scope(|scope| {
        let handles: Vec<_> = (0..workers)
            .map(|w| {
                let (tmp, cache) = (&tmp, &cache);
                scope.spawn(move || {
                    let (mut ok, mut bad) = (0usize, Vec::<String>::new());
                    for (i, (rel, outputs)) in files.iter().enumerate().filter(|(i, _)| i % workers == w) {
                        let (n, b) = check_document(rel, outputs, tmp, cache, i);
                        ok += n;
                        bad.extend(b);
                    }
                    (ok, bad)
                })
            })
            .collect();
        for h in handles {
            let (n, b) = h.join().unwrap();
            compared += n;
            bad.extend(b);
        }
    });
    let _ = std::fs::remove_dir_all(&tmp);
    bad.sort();
    assert!(bad.is_empty(), "{} of {} exports differ:\n{}", bad.len(), compared + bad.len(), bad.join("\n"));
    assert!(compared > 2500, "{compared} exports compared");
}
