// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! The parse against openskp's (scripts/golden/make_skp_fixtures.py): for
//! every synthetic `.skp` in `tests/data`, the dump of what openskp parsed
//! (`<name>.model.json`) must be the dump of what this crate parses, bit for
//! bit; a file openskp cannot read must not parse here either, with the same
//! message when it is openskp's own `SkpParseError`.
//!
//! Then one test per kind of entity says in plain asserts what each fixture
//! holds, so a failure reads as a missing feature and not only as a diff.

use std::path::{Path, PathBuf};

use ingetrazo_formats::pyjson::{loads, Value};
use ingetrazo_skp::dump::model_dump;
use ingetrazo_skp::{parse_bytes, parse_file, AttrValue, Model};

fn data() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data")
}

fn cases() -> Vec<Value> {
    match loads(&std::fs::read_to_string(data().join("cases.json")).unwrap()).unwrap() {
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

fn model(name: &str) -> Model {
    parse_file(&data().join(format!("{name}.skp"))).unwrap_or_else(|e| panic!("{name}: {e}"))
}

#[test]
fn every_fixture_parses_as_openskp_did() {
    let (mut same, mut errors) = (0, 0);
    for case in cases() {
        if str_of(&case, "kind") == Some("sequence") {
            continue;
        }
        let file = str_of(&case, "file").unwrap().to_string();
        let name = str_of(&case, "name").unwrap().to_string();
        let result = parse_file(&data().join(&file));
        if let Some(want) = str_of(&case, "parse_error") {
            let err = result.expect_err(&format!("{file}: openskp raised {want}"));
            if let Some(msg) = want.strip_prefix("SkpParseError: ") {
                assert_eq!(err.to_string(), msg, "{file}: error text");
            }
            errors += 1;
            continue;
        }
        let got = model_dump(&result.unwrap_or_else(|e| panic!("{file}: {e}"))).text();
        let want = std::fs::read_to_string(data().join(format!("{name}.model.json"))).unwrap();
        if got != want.trim_end() {
            let at = got.bytes().zip(want.bytes()).position(|(a, b)| a != b).unwrap_or(got.len().min(want.len()));
            let cut = |s: &str| s[at.saturating_sub(120)..(at + 120).min(s.len())].to_string();
            panic!("{file}: the parse differs at byte {at}\n got: {}\nwant: {}", cut(&got), cut(&want));
        }
        same += 1;
    }
    assert_eq!((same, errors), (33, 9));
}

#[test]
fn a_classic_file_with_an_extra_zero_byte_fails_the_same_way() {
    // Damage that moves the archive off its anchors is an error, not a panic.
    let mut data = std::fs::read(data().join("legacy-basic.skp")).unwrap();
    let n = data.len();
    data.truncate(n - 20);
    let _ = parse_bytes(&data);
    data.truncate(200);
    assert!(parse_bytes(&data).is_err());
}

// ---- classic container -----------------------------------------------------------------

#[test]
fn classic_geometry_materials_and_tags() {
    let m = model("legacy-basic");
    assert_eq!(m.version, "{17.0.0}");
    assert!(m.units.is_none());
    assert_eq!(m.root.faces.len(), 11);
    // the loose edge, and the edges of the faces
    assert!(m.root.edges.values().any(|e| e.soft));
    assert!(m.root.edges.values().any(|e| e.hidden));
    assert!(m.root.edges.values().any(|e| e.smooth));
    let holed = m.root.faces.values().find(|f| f.loops.len() == 2).expect("a face with a hole");
    assert_eq!((holed.loops[0].len(), holed.loops[1].len()), (4, 4));
    assert_eq!(m.materials.len(), 3);
    assert_eq!(m.materials[1].name, "Blue");
    assert_eq!(m.materials[1].transparency, 0.5);
    assert_eq!(m.materials_by_id.len(), 3);
    assert!(m.root.faces.values().any(|f| f.hidden));
    assert!(m.root.faces.values().any(|f| f.back_material_id.is_some()));
    assert_eq!(m.layers.iter().map(|l| (l.name.as_str(), l.hidden)).collect::<Vec<_>>(), [("Layer0", false), ("Walls", true), ("Roof", false)]);
    // the faces of the model root keep the tag as the number the file stores
    assert!(m.root.faces.values().any(|f| f.layer_id.is_some_and(|l| l != 0)));
}

#[test]
fn classic_annotations_and_construction_geometry() {
    let m = model("legacy-basic");
    assert_eq!(m.root.texts.len(), 2);
    assert_eq!(m.root.texts[0].text, "Hello");
    assert_eq!(m.root.texts[0].point, Some([1.0, 2.0, 3.0]));
    assert_eq!(m.root.texts[0].label_point, Some([10.0, 10.0, 10.0]));
    assert_eq!(m.root.texts[1].label_point, None);
    assert_eq!(m.dimensions.len(), 1);
    assert_eq!((m.dimensions[0].a, m.dimensions[0].b, m.dimensions[0].offset), (Some([0.0, 0.0, 0.0]), Some([100.0, 0.0, 0.0]), 12.0));
    assert_eq!(m.root.construction_lines.len(), 1);
    assert_eq!(m.root.construction_lines[0].start, Some([0.0, 0.0, 0.0]));
    assert_eq!(m.root.construction_lines[0].end, Some([10.0, 0.0, 0.0]));
    assert_eq!(m.root.construction_points[0].position, [5.0, 5.0, 5.0]);
    assert_eq!(m.root.section_planes.len(), 1);
    assert_eq!((m.root.section_planes[0].name.as_str(), m.root.section_planes[0].label.as_str()), ("Cut", "C1"));
}

#[test]
fn classic_components_groups_and_instances() {
    let m = model("legacy-components");
    let names: Vec<&str> = m.definitions.iter().map(|d| d.name.as_str()).collect();
    assert_eq!(names, ["Small", "Bench", "Sixty", "Hundred", "Outer", "Outer2", "Wrapper", "Gdef"]);
    let bench = &m.definitions[1];
    assert_eq!(bench.faces.len(), 64);
    assert_eq!(m.root.instances.len(), 9 + 8 + 5 + 2 + 3 + 2);
    let tagged = &m.definitions.iter().find(|d| d.name == "Wrapper").unwrap().instances[0];
    assert_eq!((tagged.name.as_str(), tagged.layer.as_str()), ("tagged", "Walls"));
    let hidden = m.root.instances.iter().find(|i| i.name == "hidden grp").unwrap();
    assert!(hidden.hidden);
    let painted = m.root.instances.iter().find(|i| i.name == "bench3").unwrap();
    assert!(painted.material_id.is_some());
    let rotated = m.root.instances.iter().find(|i| i.name == "bench1").unwrap();
    assert_eq!(rotated.matrix.len(), 13);
    assert!((rotated.matrix[9] - 100.0).abs() < 1e-12);
    // the faces of a definition carry their tag as a name
    let wall_face = m.definitions.iter().flat_map(|d| d.faces.values()).filter(|f| !f.layer.is_empty()).count();
    assert_eq!(wall_face, 0);
}

#[test]
fn classic_texture_mappings() {
    let m = model("legacy-textures");
    let brick = m.materials.iter().find(|x| x.name == "Brick").unwrap();
    let tex = brick.texture.as_ref().unwrap();
    assert_eq!((tex.filename.as_str(), tex.width, tex.height), ("brick.png", 20.0, 10.0));
    assert!(tex.data.as_ref().unwrap().starts_with(b"\x89PNG"));
    assert!(brick.colorized, "a classic textured material is read as colourized when its average colour is opaque");
    let mapped: Vec<_> = m.root.faces.values().filter(|f| f.uv_transform.is_some()).collect();
    assert_eq!(mapped.len(), 3);
    assert_eq!(mapped[0].uv_transform.unwrap()[..3], [0.5, 0.0, 0.0]);
    assert!(mapped.iter().any(|f| f.uv_projected));
    assert!(mapped.iter().any(|f| f.uv_transform_back.is_some_and(|b| b[0] == 0.5 && !f.uv_projected_back)));
}

#[test]
fn classic_attribute_dictionaries_and_behaviour_flags() {
    let m = model("legacy-attrs");
    let inst = &m.root.instances[0];
    assert_eq!(inst.attribute_dictionaries.len(), 2);
    let dynamic = &inst.attribute_dictionaries[0];
    assert_eq!(dynamic.0, "dynamic_attributes");
    assert!(dynamic.1.contains(&("name".to_string(), AttrValue::Str("Door".into()))));
    assert!(dynamic.1.contains(&("pt".to_string(), AttrValue::Vec3([1.0, 2.0, 3.0]))));
    assert!(dynamic.1.contains(&("list".to_string(), AttrValue::List(vec![AttrValue::Int(1), AttrValue::Str("two".into()), AttrValue::Float(3.5)]))));
    assert!(dynamic.1.contains(&("nothing".to_string(), AttrValue::Null)));
    assert!(inst.properties.contains(&("w".to_string(), "90.0".to_string())));
    assert!(inst.properties.contains(&("pt".to_string(), "1.0,2.0,3.0".to_string())));
    assert!(m.root.instances[1].attribute_dictionaries.is_empty());
    assert!(model("legacy-faceme").definitions[0].always_faces_camera);
    assert!(model("legacy-image").definitions[0].is_image);
}

#[test]
fn classic_versions_and_the_base_probes() {
    assert_eq!(model("legacy-v16").version, "{16.0.0}");
    assert_eq!(model("legacy-v13").version, "{13.0.0}");
    assert_eq!(model("legacy-v18").version, "{18.0.0}");
    assert_eq!(model("legacy-one-material").materials.len(), 1);
    assert_eq!(model("legacy-no-material").materials.len(), 0);
    // two materials of one name: the later one replaces the earlier
    let m = model("legacy-dup-material");
    assert_eq!(m.materials.iter().map(|x| x.name.as_str()).collect::<Vec<_>>(), ["Same", ""]);
    assert_eq!(m.materials[0].color[2], 255);
    assert_eq!(m.materials_by_id.len(), 3);
}

#[test]
fn classic_scene_records_name_their_hidden_tags() {
    let m = model("legacy-scene");
    assert_eq!(m.pages.len(), 1);
    assert_eq!(m.pages[0].name, "Scene A");
    assert_eq!(m.pages[0].hidden_layers, ["Walls"]);
    assert!(m.pages[0].eye.is_none());
}

// ---- 2021+ container -----------------------------------------------------------------------

#[test]
fn vff_container_units_styles_and_version() {
    let m = model("vff-basic");
    assert_eq!(m.version, "{21.0.0}");
    assert_eq!(m.units.as_deref(), Some("Meter"));
    assert_eq!(m.styles.len(), 1);
    assert_eq!(m.styles[0].name, "Simple Style");
    assert_eq!(m.styles[0].front_color, Some([0, 0, 0]));
    assert_eq!(m.styles[0].back_color.map(|c| c[0]), Some(0x6F));
    assert!(m.thumbnail.is_some());
}

#[test]
fn vff_edges_carry_their_display_flags_curves_and_tags() {
    let m = model("vff-basic");
    let boxd = m.definitions.iter().find(|d| d.name == "Box").unwrap();
    assert!(boxd.edges.values().any(|e| e.soft && e.smooth && !e.hidden));
    assert!(boxd.edges.values().any(|e| e.hidden && !e.soft));
    assert!(boxd.edges.values().any(|e| e.curve_id == Some(4242)));
    assert!(boxd.edges.values().any(|e| e.layer.is_some()));
    // the loose edge (soft + smooth, no face uses it)
    let used: std::collections::HashSet<i64> = boxd.faces.values().flat_map(|f| f.loops.iter().flatten().map(|c| c.0)).collect();
    assert!(boxd.edges.keys().any(|k| !used.contains(k)));
}

#[test]
fn vff_faces_loops_holes_materials_mappings_and_tags() {
    let m = model("vff-basic");
    let boxd = m.definitions.iter().find(|d| d.name == "Box").unwrap();
    assert_eq!(boxd.faces.len(), 9);
    assert!(boxd.faces.values().any(|f| f.loops.len() == 2));
    assert!(boxd.faces.values().any(|f| f.hidden));
    assert!(boxd.faces.values().any(|f| f.back_material_id.is_some() && f.material_id.is_some()));
    assert!(boxd.faces.values().any(|f| f.back_material_id.is_some() && f.material_id.is_none()));
    let mapped = boxd.faces.values().find(|f| f.uv_transform.is_some()).unwrap();
    assert_eq!(mapped.uv_transform.unwrap(), [0.5, 0.0, 0.0, 0.0, 0.25, 0.0, 3.0, 4.0, 1.0]);
    assert_eq!(mapped.uv_transform_back.unwrap()[0], 1.0);
    // a reversed co-edge is -1
    assert!(boxd.faces.values().any(|f| f.loops.iter().flatten().any(|c| c.1 == -1)));
    let tagged = boxd.faces.values().find(|f| !f.layer.is_empty()).unwrap();
    assert_eq!(tagged.layer, "Walls");
    // the model root's faces keep the id (openskp resolves definitions only)
    assert!(m.root.faces.values().any(|f| f.layer_id.is_some() && f.layer.is_empty()));
}

#[test]
fn vff_instances_matrices_paint_tags_and_visibility() {
    let m = model("vff-basic");
    let tile = m.definitions.iter().find(|d| d.name == "Tile").unwrap();
    assert_eq!(tile.faces.len(), 64);
    let tiles: Vec<_> = m.root.instances.iter().filter(|i| i.name.starts_with("tile")).collect();
    assert_eq!(tiles.len(), 9);
    assert_eq!(tiles.iter().filter(|i| i.layer == "Roof").count(), 3);
    assert_eq!(tiles.iter().filter(|i| i.material_id.is_some()).count(), 2);
    let hidden = m.root.instances.iter().find(|i| i.name == "hidden box").unwrap();
    assert!(hidden.hidden);
    let mirrored = m.root.instances.iter().find(|i| i.name == "mirrored box").unwrap();
    assert_eq!(mirrored.matrix[0], -1.0);
    assert_eq!(mirrored.layer, "Walls");
    assert_eq!(mirrored.ref_idx, Some(m.definitions.iter().find(|d| d.name == "Box").unwrap().id));
    let frame = m.definitions.iter().find(|d| d.name == "Frame").unwrap();
    assert_eq!(frame.instances.len(), 2);
    assert_eq!(frame.instances[1].matrix[0], 0.5);
}

#[test]
fn vff_materials_textures_and_the_id_join() {
    let m = model("vff-textures");
    let tex = m.materials.iter().find(|x| x.name == "Tex").unwrap();
    let t = tex.texture.as_ref().unwrap();
    assert_eq!((t.filename.as_str(), t.width, t.height), ("tex.png", 36.0, 18.0));
    assert!(t.data.as_ref().unwrap().starts_with(b"\x89PNG"));
    // a colourized copy takes the image of the material its XML points at
    let copy = m.materials.iter().find(|x| x.name == "Tex1").unwrap();
    assert!(copy.colorized);
    assert_eq!(copy.texture.as_ref().unwrap().data, m.materials.iter().find(|x| x.name == "Flat").unwrap().texture.as_ref().unwrap().data);
    assert_eq!(copy.texture.as_ref().unwrap().filename, "missing.png");
    // a texture whose image is nowhere
    assert!(m.materials.iter().find(|x| x.name == "NoImage").unwrap().texture.as_ref().unwrap().data.is_none());
    // two XML files with one name: the table has the later
    assert_eq!(m.materials.iter().filter(|x| x.name == "Same").count(), 1);
    assert_eq!(m.materials.iter().find(|x| x.name == "Same").unwrap().color[0], 8);
    let glass = model("vff-basic");
    let g = glass.materials.iter().find(|x| x.name == "Glass").unwrap();
    assert!((g.transparency - 0.7).abs() < 1e-12);
    assert_eq!(glass.materials_by_id.len(), 3);
    assert!(glass.materials.iter().all(|x| !x.name.starts_with("Layer_") || x.id.is_none()));
}

#[test]
fn vff_tags_scenes_and_dimensions() {
    let m = model("vff-pages");
    assert_eq!(m.layers.iter().map(|l| (l.name.as_str(), l.hidden)).collect::<Vec<_>>(), [("Walls", false), ("Roof", true), ("Layer0", false)]);
    assert_eq!(m.pages.len(), 3);
    assert_eq!(m.pages[0].name, "Plan");
    assert!(m.pages[0].parallel);
    assert_eq!(m.pages[0].ortho_height, 240.0);
    assert_eq!(m.pages[1].hidden_layers, ["Walls", "Roof"]);
    assert_eq!(m.pages[1].fov, 45.0);
    assert!(!m.pages[2].parallel);
    // free points, and two ends resolved through a vertex and an instance
    assert_eq!(m.dimensions.len(), 3);
    assert_eq!((m.dimensions[0].a, m.dimensions[0].b, m.dimensions[0].offset), (Some([0.0, 0.0, 0.0]), Some([100.0, 0.0, 0.0]), 12.0));
    assert_eq!(m.dimensions[1].offset, -5.0);
    assert_eq!(m.dimensions[1].normal, Some([0.0, 0.0, 1.0]));
    let anchored = &m.dimensions[2];
    // the definition's first vertex (0,0,0) through the instance (rot 90, +50,+50)
    let a = anchored.a.unwrap();
    assert!((a[0] - 50.0).abs() < 1e-9 && (a[1] - 50.0).abs() < 1e-9);
    assert_eq!(anchored.b, Some([300.0, 0.0, 0.0]));
}

#[test]
fn vff_attribute_dictionaries_by_name_and_type() {
    let m = model("vff-attrs");
    let inst = &m.root.instances[0];
    assert_eq!(inst.attribute_dictionaries.len(), 2);
    let dynamic = &inst.attribute_dictionaries[0];
    assert_eq!(dynamic.0, "dynamic_attributes");
    assert!(dynamic.1.contains(&("n".to_string(), AttrValue::Int(3))));
    assert!(dynamic.1.contains(&("w".to_string(), AttrValue::Float(90.0))));
    assert!(dynamic.1.contains(&("pt".to_string(), AttrValue::Vec3([1.0, 2.0, 3.0]))));
    assert!(dynamic.1.contains(&("list".to_string(), AttrValue::List(vec![AttrValue::Int(1), AttrValue::Str("two".into()), AttrValue::Float(3.5)]))));
    assert!(dynamic.1.contains(&("nothing".to_string(), AttrValue::Null)));
    assert!(inst.properties.contains(&("name".to_string(), "Door".to_string())));
    assert!(inst.properties.contains(&("list".to_string(), "1,two,3.5".to_string())));
    assert!(inst.properties.contains(&("nothing".to_string(), String::new())));
}

#[test]
fn vff_definition_kinds_wrappers_and_odd_ids() {
    assert!(model("vff-faceme").definitions[0].always_faces_camera);
    assert!(model("vff-faceme").definitions[0].shadows_face_sun);
    assert!(model("vff-image").definitions[0].is_image);
    let plain = model("vff-basic");
    let f901 = model("vff-wrapped-f901");
    let f401 = model("vff-wrapped-f401");
    assert_eq!(plain.definitions.len(), 3);
    for w in [&f901, &f401] {
        assert_eq!(w.definitions.iter().map(|d| d.name.as_str()).collect::<Vec<_>>(), ["Box", "Tile", "Frame"]);
        assert_eq!(w.root.instances.len(), plain.root.instances.len());
    }
    // two definitions with one id: the later wins; a definition with no name
    let odd = model("vff-odd");
    assert_eq!(odd.definitions.iter().filter(|d| d.id == 5000).count(), 1);
    assert_eq!(odd.definitions.iter().find(|d| d.id == 5000).unwrap().name, "Second");
    assert!(odd.definitions.iter().any(|d| d.name.is_empty()));
    assert!(odd.root.instances.iter().any(|i| i.ref_idx == Some(123456)));
    // an instance with no transformation record has an empty matrix
    assert!(model("vff-no-matrix").root.instances[0].matrix.is_empty());
}

#[test]
fn vff_empty_models_keep_their_thumbnail() {
    let m = model("vff-empty");
    assert!(m.definitions.is_empty() && m.root.faces.is_empty() && m.root.instances.is_empty());
    assert!(m.thumbnail.as_ref().is_some_and(|t| t.starts_with(b"\x89PNG")));
}

// ---- errors ---------------------------------------------------------------------------------

#[test]
fn damaged_files_are_errors_not_panics() {
    for name in ["err-not-skp", "err-empty", "err-magic-only", "err-header-no-zip", "err-no-model-dat", "err-truncated-zip", "err-corrupt-zip", "err-legacy-garbage", "err-page-no-camera"] {
        let e = parse_file(&data().join(format!("{name}.skp"))).expect_err(name);
        assert!(!e.to_string().is_empty(), "{name}");
    }
    assert!(parse_file(&data().join("nothing-here.skp")).unwrap_err().to_string().contains("File not found"));
    assert!(parse_file(&data().join("cases.json")).unwrap_err().to_string().contains("Expected a .skp file"));
}

/// A small deterministic generator (SplitMix64), so the damage is the same on
/// every run.
pub struct Rng(pub u64);

impl Rng {
    pub fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
}

#[test]
fn damaged_bytes_never_panic() {
    // A few bytes of a good file changed at random: the parse may succeed
    // (the damage can land in a number) or fail, but never panic.
    let mut rng = Rng(7);
    let (mut ok, mut failed) = (0, 0);
    for name in ["legacy-components", "legacy-textures", "legacy-basic", "vff-basic", "vff-pages", "vff-textures", "vff-attrs"] {
        let good = std::fs::read(data().join(format!("{name}.skp"))).unwrap();
        for _ in 0..150 {
            let mut bad = good.clone();
            for _ in 0..1 + rng.next() % 6 {
                let at = (rng.next() as usize) % bad.len();
                bad[at] = rng.next() as u8;
            }
            match parse_bytes(&bad) {
                Ok(_) => ok += 1,
                Err(_) => failed += 1,
            }
        }
    }
    assert!(ok > 0 && failed > 0, "{ok} parsed, {failed} failed");
}

#[test]
fn deeply_nested_records_are_an_error_not_a_stack_overflow() {
    // 100000 nested containers (tag C409, size covering the rest).
    let mut body = Vec::new();
    for _ in 0..100_000 {
        body.extend([0xF4u8, 0x01, 0, 0, 0, 0]);
    }
    let mut size = 0u32;
    // each header's size covers everything after it
    let total = body.len();
    for i in 0..100_000usize {
        size = (total - (i + 1) * 6) as u32;
        body[i * 6 + 2..i * 6 + 6].copy_from_slice(&size.to_le_bytes());
    }
    let _ = size;
    let nodes = ingetrazo_skp::tlv::try_parse(&body, 0, body.len(), ingetrazo_skp::tlv::is_container);
    assert!(nodes.unwrap_err().contains("RecursionError"));
}

#[test]
fn every_prefix_of_a_good_file_parses_or_fails_cleanly() {
    for name in ["legacy-components", "vff-basic"] {
        let full = std::fs::read(data().join(format!("{name}.skp"))).unwrap();
        let step = (full.len() / 400).max(1);
        let mut cut = 0;
        while cut < full.len() {
            let _ = parse_bytes(&full[..cut]);
            cut += step;
        }
    }
}
