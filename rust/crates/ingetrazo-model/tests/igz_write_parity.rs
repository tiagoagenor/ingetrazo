// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! Saving a scene as an .igz against `formats/igz.py` `save_scene`
//! (scripts/golden/make_igz_write_fixtures.py): a Rust round trip of every
//! .igz Python wrote — the golden start scenes, the shipped examples and
//! components, and the fixtures built for this test in all three formats —
//! writes the very bytes it read; the container's `document.json` and every
//! member are compared first so a failure says where it differs.

use std::path::{Path, PathBuf};

use ingetrazo_formats::igz::{self, Zip};
use ingetrazo_formats::igz_geom::expand_document;
use ingetrazo_formats::pyjson::Value;
use ingetrazo_math::{M4, V3};
use ingetrazo_model::mesh::Mesh;
use ingetrazo_model::igz_write::{save_scene, SaveOptions};
use ingetrazo_model::scene::{Group, MeshId, Scene};

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../..")
}

/// Open `raw` as the Rust app does and save it again.
fn resave(raw: &[u8], tweak: impl Fn(&mut SaveOptions)) -> Vec<u8> {
    let doc = igz::read_document(raw).expect("document");
    let zip = igz::is_container(raw).then(|| Zip::open(raw).expect("zip"));
    let expanded = expand_document(&doc, zip.as_ref()).expect("expand");
    let scene = Scene::from_document(&expanded).expect("scene");
    let version = match doc.get("app_version") {
        Some(Value::Str(v)) => v.clone(),
        _ => String::new(),
    };
    let read = |src: &str| zip.as_ref().and_then(|z| z.read(src).ok());
    let mut opts = SaveOptions::new(&version, &read);
    tweak(&mut opts);
    save_scene(&scene, &opts).expect("save").bytes
}

/// Compare two files; for containers also each member, so the report names
/// the first one that differs.
fn compare(name: &str, expected: &[u8], got: &[u8]) -> Result<(), String> {
    if igz::is_container(expected) != igz::is_container(got) {
        return Err(format!("{name}: container vs plain"));
    }
    if igz::is_container(expected) {
        let (a, b) = (Zip::open(expected).unwrap(), Zip::open(got).unwrap());
        let names = |z: &Zip| z.entries().iter().map(|e| e.name.clone()).collect::<Vec<_>>();
        if names(&a) != names(&b) {
            return Err(format!("{name}: members {:?} vs {:?}", names(&a), names(&b)));
        }
        for n in names(&a) {
            if a.read(&n).unwrap() != b.read(&n).unwrap() {
                return Err(format!("{name}: member {n} differs"));
            }
        }
    }
    if expected != got {
        return Err(format!("{name}: bytes differ ({} vs {})", expected.len(), got.len()));
    }
    Ok(())
}

fn igz_files(dir: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    if let Ok(rd) = std::fs::read_dir(dir) {
        for e in rd.flatten() {
            let p = e.path();
            if p.is_dir() {
                out.extend(igz_files(&p));
            } else if p.extension().is_some_and(|x| x == "igz") {
                out.push(p);
            }
        }
    }
    out.sort();
    out
}

/// What git's autocrlf may have done to a plain JSON file of the working tree.
fn lf(raw: &[u8]) -> Vec<u8> {
    if igz::is_container(raw) {
        raw.to_vec()
    } else {
        String::from_utf8_lossy(raw).replace("\r\n", "\n").into_bytes()
    }
}

/// A file zipfile wrote on Linux says `create_system` 3 in its central
/// directory; the Windows reference build says 0. Nothing else differs.
fn as_windows(raw: &[u8]) -> Vec<u8> {
    let mut raw = raw.to_vec();
    if let Some(eocd) = raw.windows(4).rposition(|w| w == b"PK") {
        let count = u16::from_le_bytes([raw[eocd + 10], raw[eocd + 11]]) as usize;
        let mut at = u32::from_le_bytes([raw[eocd + 16], raw[eocd + 17], raw[eocd + 18], raw[eocd + 19]]) as usize;
        for _ in 0..count {
            raw[at + 5] = 0;
            let w = |o: usize| u16::from_le_bytes([raw[at + o], raw[at + o + 1]]) as usize;
            at += 46 + w(28) + w(30) + w(32);
        }
    }
    raw
}

/// `legacy`: the files come from an older build (no format 3, written on
/// Linux): saved with no size limit and compared as a Windows build wrote them.
fn round_trip_dir(dir: &Path, min: usize, legacy: bool) {
    let files = igz_files(dir);
    assert!(files.len() >= min, "{}: {} files", dir.display(), files.len());
    let mut bad = Vec::new();
    for f in &files {
        let raw = lf(&std::fs::read(f).unwrap());
        let (want, got) = if legacy {
            let got = resave(&raw, |o| o.large_document = usize::MAX);
            (if igz::is_container(&raw) { as_windows(&raw) } else { raw }, got)
        } else {
            let got = resave(&raw, |_| {});
            (raw, got)
        };
        let name = f.display().to_string();
        // Before format 3 a large document was always a container, even with
        // nothing but document.json in it: its JSON must be what is written.
        let only_doc = legacy && igz::is_container(&want) && !igz::is_container(&got) && Zip::open(&want).unwrap().entries().len() == 1;
        let result = if only_doc {
            // ... and compact, as a large document is.
            let doc = Zip::open(&want).unwrap().read(igz::DOC_ENTRY).unwrap();
            let v = ingetrazo_formats::pyjson::loads(std::str::from_utf8(&got).unwrap()).unwrap();
            let compact = ingetrazo_formats::pyjson::dumps(&v, &ingetrazo_formats::pyjson::Options::compact().allow_nan(false)).unwrap();
            if doc == compact.as_bytes() { Ok(()) } else { Err(format!("{name}: document.json differs")) }
        } else {
            compare(&name, &want, &got)
        };
        if let Err(e) = result {
            bad.push(e);
        }
    }
    assert!(bad.is_empty(), "{} of {} differ:\n{}", bad.len(), files.len(), bad.join("\n"));
}

#[test]
fn golden_starts_round_trip_byte_identical() {
    round_trip_dir(&root().join("golden/starts"), 200, false);
}

#[test]
fn shipped_documents_round_trip_byte_identical() {
    round_trip_dir(&root().join("resources/components"), 8, true);
}

/// The thresholds a fixture was written under, from its name
/// (`<scene>.<f1|f2|f3|fast|mix>.igz`; see make_igz_write_fixtures.py).
fn thresholds(path: &Path) -> Option<(usize, usize)> {
    let name = path.file_name().unwrap().to_string_lossy().into_owned();
    match name.rsplit('.').nth(1).unwrap() {
        "f1" | "f2" => None,
        "f3" => Some((0, 1)),
        "fast" => Some((0, 1_000_000_000)),
        "mix" => Some((0, 60)),
        other => panic!("{name}: unknown variant {other}"),
    }
}

/// Not part of the committed suite (the files are megabytes): point
/// `IGZ_WRITE_EXTRA` at a directory of .igz files Python saved with the
/// default thresholds (big documents, bulk-loaded meshes) to round-trip them.
#[test]
fn extra_directory_round_trips_byte_identical() {
    let Some(dir) = std::env::var_os("IGZ_WRITE_EXTRA") else { return };
    round_trip_dir(Path::new(&dir), 1, false);
}

// ---- scenes built in Rust, against the files Python saved for the same ----

fn v(x: f64, y: f64, z: f64) -> V3 {
    V3(x as f32, y as f32, z as f32)
}

fn xform(dx: f32, dy: f32, dz: f32, angle: f32) -> M4 {
    let mut m = M4::identity();
    m.translate(dx, dy, dz);
    if angle != 0.0 {
        m.rotate(angle, 0.0, 0.0, 1.0);
    }
    m
}

/// `make_igz_format_fixtures.scene_curves` (its 0.7071 is the fixture's, not sqrt(1/2)).
#[allow(clippy::approx_constant)]
fn scene_curves() -> Scene {
    let mut s = Scene::new();
    let m = &mut s.meshes[0];
    let pts: Vec<V3> = (0..6).map(|i| v(i as f64, (i % 3) as f64 * 0.5, 0.0)).collect();
    for (i, w) in pts.windows(2).enumerate() {
        let e = m.add_edge(w[0], w[1]).unwrap();
        m.edge_mut(e).curve = Some(4);
        if i == 2 {
            m.edge_mut(e).soft = true;
        }
    }
    let ring: Vec<V3> = (0..4).map(|k| v(10.0 + 2.0 * 0.7071 * k as f64, 0.0, 0.0)).collect();
    for w in ring.windows(2) {
        let e = m.add_edge(w[0], w[1]).unwrap();
        m.edge_mut(e).curve = Some(9);
        m.edge_mut(e).layer = Some("Guias".into());
    }
    let h = m.add_edge(v(0.0, 5.0, 0.0), v(3.0, 5.0, 1.5)).unwrap();
    m.edge_mut(h).hidden = true;
    m.edge_mut(h).layer = Some("Ocultas".into());
    let s1 = m.add_edge(v(0.0, 6.0, 0.0), v(3.0, 6.0, 0.0)).unwrap();
    m.edge_mut(s1).soft = true;
    m.edge_mut(s1).hidden = true;
    m.add_edge(v(0.0, 7.0, 0.0), v(3.0, 7.25, 0.1)).unwrap();
    s
}

fn num(x: f64) -> Value {
    Value::Float(x)
}

/// `make_igz_format_fixtures.scene_groups`, uids as `fix_uids` numbers them.
fn scene_groups() -> Scene {
    let mut s = Scene::new();
    s.meshes[0].add_face(&[v(-5.0, -5.0, 0.0), v(-4.0, -5.0, 0.0), v(-4.0, -4.0, 0.0)], &[]);
    let mut proto = Mesh::new();
    let f = proto.add_face(&[v(0.0, 0.0, 0.0), v(1.0, 0.0, 0.0), v(1.0, 1.0, 0.0), v(0.0, 1.0, 0.0)], &[vec![v(0.25, 0.25, 0.0), v(0.5, 0.25, 0.0), v(0.5, 0.5, 0.0)]]);
    let e = proto.add_edge(v(0.0, 0.0, 0.0), v(0.0, 0.0, 1.0)).unwrap();
    proto.edge_mut(e).curve = Some(3);
    proto.face_mut(f).attrs.push(("color".into(), Value::Array(vec![num(0.2), num(0.6), num(0.2)])));
    s.meshes.push(proto);
    let proto_id = MeshId(1);
    for (k, dx) in [0.0f32, 5.0, 10.0].into_iter().enumerate() {
        let mut g = Group::new(proto_id, format!("Silla {k}"));
        g.xform = Some(xform(dx, 20.0, 0.0, 15.0 * k as f32));
        if k == 2 {
            g.layer = Some("Mobiliario".into());
            g.hidden = true;
        }
        g.uid = format!("{:016x}", k + 1);
        s.groups.push(g);
    }
    let mut classic_mesh = Mesh::new();
    classic_mesh.add_face(&[v(0.0, 30.0, 0.0), v(2.0, 30.0, 0.0), v(2.0, 31.0, 0.0)], &[]);
    let e = classic_mesh.add_edge(v(0.0, 30.0, 0.0), v(0.0, 30.0, 1.0)).unwrap();
    classic_mesh.edge_mut(e).layer = Some("Patas".into());
    s.meshes.push(classic_mesh);
    let mut classic = Group::new(MeshId(2), "Mesa".into());
    classic.locked = true;
    classic.uid = format!("{:016x}", 4);
    let mut leaf = Mesh::new();
    leaf.add_face(&[v(0.0, 0.0, 0.0), v(0.5, 0.0, 0.0), v(0.5, 0.5, 0.2)], &[]);
    s.meshes.push(leaf);
    let mut parent_mesh = Mesh::new();
    parent_mesh.add_face(&[v(0.0, 40.0, 0.0), v(3.0, 40.0, 0.0), v(3.0, 43.0, 0.0), v(0.0, 43.0, 0.0)], &[]);
    s.meshes.push(parent_mesh);
    let mut parent = Group::new(MeshId(4), "Conjunto".into());
    parent.uid = format!("{:016x}", 5);
    let mut kids = Vec::new();
    for k in 0..2 {
        let mut child = Group::new(MeshId(3), format!("Hoja {k}"));
        child.xform = Some(xform(k as f32, 41.0, 0.0, 0.0));
        child.uid = format!("{:016x}", 6 + k);
        kids.push(child);
    }
    let mut inner_mesh = Mesh::new();
    inner_mesh.add_face(&[v(0.0, 50.0, 0.0), v(1.0, 50.0, 0.0), v(1.0, 51.0, 0.0)], &[]);
    s.meshes.push(inner_mesh);
    let mut inner = Group::new(MeshId(5), "Interno".into());
    inner.uid = format!("{:016x}", 8);
    kids.push(inner);
    parent.adopt(kids);
    parent.component = false;
    s.groups.push(classic);
    s.groups.push(parent);
    s
}

fn native_matches(name: &str, scene: &Scene) {
    let (large, bulk) = thresholds(Path::new(name)).unwrap_or((20_000, 1024));
    let expected = std::fs::read(root().join("rust/crates/ingetrazo-model/tests/data/igz_write").join(name)).unwrap();
    let none = |_: &str| None;
    let mut opts = SaveOptions::new("0.5.8", &none);
    opts.large_document = large;
    opts.bulk_min = bulk;
    let got = save_scene(scene, &opts).unwrap().bytes;
    compare(name, &expected, &got).unwrap();
}

#[test]
fn scenes_built_in_rust_write_what_python_wrote() {
    let curves = scene_curves();
    let groups = scene_groups();
    for variant in ["f1", "f3", "fast", "mix"] {
        native_matches(&format!("curves.{variant}.igz"), &curves);
        native_matches(&format!("groups.{variant}.igz"), &groups);
    }
}

fn fixtures() -> Vec<PathBuf> {
    igz_files(&root().join("rust/crates/ingetrazo-model/tests/data/igz_write"))
}

#[test]
fn written_fixtures_round_trip_byte_identical() {
    let files = fixtures();
    assert!(files.len() >= 25, "{} fixtures", files.len());
    let mut bad = Vec::new();
    for f in &files {
        let raw = std::fs::read(f).unwrap();
        let limits = thresholds(f);
        let got = resave(&raw, |o| {
            if let Some((large, bulk)) = limits {
                o.large_document = large;
                o.bulk_min = bulk;
            }
        });
        if let Err(e) = compare(&f.file_name().unwrap().to_string_lossy(), &raw, &got) {
            bad.push(e);
        }
    }
    assert!(bad.is_empty(), "{} of {} differ:\n{}", bad.len(), files.len(), bad.join("\n"));
}

#[test]
fn fixtures_cover_every_format_and_block() {
    // The fixtures must exercise what the writer claims to: all three
    // formats, a compact document with no binary block, every optional
    // block of save_scene, textures stored, geometry blocks deflated.
    let mut formats = std::collections::BTreeSet::new();
    let mut blocks = std::collections::BTreeSet::new();
    for f in fixtures() {
        let raw = std::fs::read(&f).unwrap();
        let doc = igz::read_document(&raw).unwrap();
        formats.insert((doc.get("igz_format").and_then(Value::as_i64).unwrap(), igz::is_container(&raw)));
        if let Some(Value::Object(items)) = doc.get("scene") {
            blocks.extend(items.iter().map(|(k, _)| k.clone()));
        }
    }
    assert!(formats.contains(&(1, false)) && formats.contains(&(2, true)) && formats.contains(&(3, true)), "{formats:?}");
    for want in [
        "edges",
        "faces",
        "groups",
        "kept_components",
        "protos",
        "layers",
        "saved_views",
        "compositions",
        "dimensions",
        "materials",
        "dimension_style",
        "camera",
        "units",
        "plugin_data",
        "custom_scales",
        "back_face_color",
        "section_planes",
        "section_view",
        "hidden_view",
        "display_style",
        "shadows",
        "georef",
        "tile_layer",
        "geo_paths",
        "geo_points",
        "text_labels",
        "guides",
        "image_planes",
    ] {
        assert!(blocks.contains(want), "no fixture has {want}: {blocks:?}");
    }
}
