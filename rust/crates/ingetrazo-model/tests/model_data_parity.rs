// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! The document data classes against the Python ones
//! (scripts/golden/make_model_data_fixtures.py): for every input the
//! `json.dumps` text Python produced, or the exception class it raised, must
//! come out of the Rust port byte for byte.

use std::path::{Path, PathBuf};

use ingetrazo_formats::pyjson::{dumps, loads, Options, Value};
use ingetrazo_model::pyval::{PyError, PyResult};
use ingetrazo_model::{bim, face_attrs, layers, materials, texture};

fn text(v: &Value) -> String {
    dumps(v, &Options::default()).unwrap()
}

fn data(name: &str) -> Vec<Value> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/model_data").join(name);
    let body = std::fs::read_to_string(&path).unwrap();
    body.lines().map(|l| loads(l).unwrap_or_else(|e| panic!("{name}: {e}"))).collect()
}

fn s(v: &Value, key: &str) -> String {
    match v.get(key) {
        Some(Value::Str(t)) => t.clone(),
        other => panic!("{key}: {other:?}"),
    }
}

fn load(v: &Value, key: &str) -> Value {
    loads(&s(v, key)).unwrap_or_else(|e| panic!("input {}: {e}", s(v, key)))
}

#[derive(Default)]
struct Checker {
    checked: usize,
    failures: Vec<String>,
}

impl Checker {
    /// `expected` is `{"ok": text}` or `{"err": class}`.
    fn eq(&mut self, what: &str, input: &str, expected: &Value, got: PyResult<Value>) {
        self.checked += 1;
        let ok = match (expected.get("ok"), expected.get("err"), &got) {
            (Some(Value::Str(want)), _, Ok(v)) => {
                let have = text(v);
                if &have == want {
                    true
                } else {
                    self.failures.push(format!("{what}: {input}\n  want {want}\n  have {have}"));
                    return;
                }
            }
            (_, Some(Value::Str(kind)), Err(PyError { kind: k, .. })) => kind == k,
            _ => false,
        };
        if !ok {
            let want = expected.get("ok").or(expected.get("err")).map(text);
            self.failures.push(format!("{what}: {input}\n  want {want:?}\n  have {got:?}"));
        }
    }

    fn done(self) {
        let n = self.failures.len();
        assert!(
            self.failures.is_empty(),
            "{n} of {} checks differ; first ones:\n{}",
            self.checked,
            self.failures.iter().take(8).cloned().collect::<Vec<_>>().join("\n")
        );
        assert!(self.checked > 0);
    }
}

fn obj(items: Vec<(&str, Value)>) -> Value {
    Value::Object(items.into_iter().map(|(k, v)| (k.to_string(), v)).collect())
}

fn kv(items: Vec<(String, Value)>) -> Value {
    Value::Object(items)
}

fn opt(v: Option<Value>) -> Value {
    v.unwrap_or(Value::Null)
}

#[test]
fn materials() {
    let mut c = Checker::default();
    for case in data("materials.jsonl") {
        let input = s(&case, "in");
        let raw = load(&case, "in");
        let m = materials::Material::from_dict(&raw);
        match case.get("from").unwrap() {
            Value::Object(o) if o[0].0 == "err" => {
                let Value::Str(kind) = &o[0].1 else { panic!() };
                c.checked += 1;
                match &m {
                    Err(e) if e.kind == kind => {}
                    other => c.failures.push(format!("from_dict {input}: want {kind}, have {other:?}")),
                }
                continue;
            }
            _ => {}
        }
        let m = match m {
            Ok(m) => m,
            Err(e) => {
                c.failures.push(format!("from_dict {input}: unexpected {e:?}"));
                continue;
            }
        };
        c.eq("to_dict", &input, case.get("to_dict").unwrap(), m.to_dict());
        c.eq("face_attrs", &input, case.get("face_attrs").unwrap(), m.face_attrs().map(kv));
        if let Some(round) = case.get("round") {
            let again = loads(&text(&m.to_dict().unwrap())).unwrap();
            c.eq("round", &input, round, materials::Material::from_dict(&again).and_then(|x| x.to_dict()));
        }
    }
    c.done();
}

#[test]
fn registry_and_stamping() {
    let mut c = Checker::default();
    for case in data("registry.jsonl") {
        let input = case.get("in").map(text).unwrap_or_else(|| s(&case, "attrs"));
        match s(&case, "op").as_str() {
            "materials_block" => {
                let payload = load(&case, "in");
                let got = materials::registry_from_payload(&payload)
                    .and_then(|reg| materials::registry_to_payload(&reg))
                    .map(opt);
                c.eq("materials_block", &input, case.get("out").unwrap(), got);
            }
            "register" => {
                let Value::Array(items) = load(&case, "in") else { panic!() };
                let got = (|| -> PyResult<Value> {
                    let mut reg = materials::Registry::new();
                    let mut names = Vec::new();
                    for raw in &items {
                        let m = materials::Material::from_dict(raw)?;
                        names.push(materials::register(&mut reg, m)?);
                    }
                    let list = materials::registry_to_payload(&reg)?;
                    Ok(obj(vec![("names", Value::Array(names)), ("registry", opt(list))]))
                })();
                c.eq("register", &input, case.get("out").unwrap(), got);
            }
            "stamp" => {
                let attrs = load(&case, "attrs");
                let mat = load(&case, "material");
                c.eq(
                    "has_own",
                    &input,
                    case.get("has_own").unwrap(),
                    materials::has_own_material(&attrs).map(Value::Bool),
                );
                c.eq(
                    "effective",
                    &input,
                    case.get("effective").unwrap(),
                    materials::effective_attrs(&attrs, &mat),
                );
                c.eq("sig", &input, case.get("sig").unwrap(), materials::material_sig(&mat).map(opt));
                let nocut = |_: &Value| false;
                c.eq(
                    "translucent",
                    &input,
                    case.get("translucent").unwrap(),
                    materials::is_translucent(&attrs, &nocut).map(Value::Bool),
                );
                c.eq(
                    "back_default",
                    &input,
                    case.get("back_default").unwrap(),
                    materials::back_is_default(&attrs, &nocut).map(Value::Bool),
                );
            }
            other => panic!("{other}"),
        }
    }
    c.done();
}

#[test]
fn layers() {
    let mut c = Checker::default();
    for case in data("layers.jsonl") {
        let input = case.get("in").map(text).unwrap_or_default();
        match s(&case, "op").as_str() {
            "layer" => {
                let raw = load(&case, "in");
                let ly = layers::Layer::from_dict(&raw);
                if let Some(Value::Str(kind)) = case.get("from").unwrap().get("err") {
                    c.checked += 1;
                    match &ly {
                        Err(e) if e.kind == kind => {}
                        other => c.failures.push(format!("layer {input}: want {kind}, have {other:?}")),
                    }
                    continue;
                }
                let ly = ly.unwrap_or_else(|e| panic!("{input}: {e:?}"));
                c.eq("layer.to_dict", &input, case.get("to_dict").unwrap(), Ok(ly.to_dict()));
                let again = loads(&text(&ly.to_dict())).unwrap();
                c.eq(
                    "layer.round",
                    &input,
                    case.get("round").unwrap(),
                    layers::Layer::from_dict(&again).map(|x| x.to_dict()),
                );
            }
            "layers_block" => {
                let payload = load(&case, "in");
                let got = layers::layers_from_payload(&payload).map(|ls| {
                    obj(vec![
                        ("loaded", Value::Array(ls.iter().map(layers::Layer::to_dict).collect())),
                        ("written", opt(layers::layers_to_payload(&ls))),
                    ])
                });
                c.eq("layers_block", &input, case.get("out").unwrap(), got);
            }
            "layer_ops" => {
                let attrs = load(&case, "attrs");
                let name = load(&case, "name");
                let input = format!("{} {}", s(&case, "attrs"), s(&case, "name"));
                c.eq("of_face", &input, case.get("of_face").unwrap(), layers::layer_of_attrs(&attrs));
                c.eq(
                    "of_slot",
                    &input,
                    case.get("of_slot").unwrap(),
                    Ok(layers::layer_of_slot(attrs.get("layer"))),
                );
                c.eq("of_none", &input, case.get("of_none").unwrap(), Ok(layers::layer_of_slot(None)));
                let Value::Object(mut items) = attrs.clone() else { panic!() };
                layers::assign_layer_attrs(&mut items, &name);
                c.eq("assign_face", &input, case.get("assign_face").unwrap(), Ok(kv(items)));
                c.eq(
                    "assign_slot",
                    &input,
                    case.get("assign_slot").unwrap(),
                    Ok(opt(layers::assign_layer_slot(&name))),
                );
            }
            other => panic!("{other}"),
        }
    }
    c.done();
}

#[test]
fn bim() {
    let mut c = Checker::default();
    for case in data("bim.jsonl") {
        match s(&case, "op").as_str() {
            "bim_consts" => {
                let classes = Value::Array(bim::IFC_CLASSES.iter().map(|t| Value::Str(t.to_string())).collect());
                let pairs = |t: &[(&str, &str)]| {
                    kv(t.iter().map(|(k, v)| (k.to_string(), Value::Str(v.to_string()))).collect())
                };
                assert_eq!(text(&classes), s(&case, "classes"));
                assert_eq!(text(&pairs(&bim::QSET_NAMES)), s(&case, "qsets"));
                assert_eq!(text(&pairs(&bim::METRADO_UNIT)), s(&case, "units"));
                assert_eq!(text(&Value::Float(bim::STEEL_DENSITY)), s(&case, "steel"));
                c.checked += 1;
            }
            "next_object_id" => {
                let Value::Array(faces) = load(&case, "in") else { panic!() };
                c.eq("next_object_id", &s(&case, "in"), case.get("out").unwrap(), bim::next_object_id(&faces));
            }
            "tag" => {
                let (cls, name, id) = (load(&case, "class"), load(&case, "name"), load(&case, "id"));
                let mut f = vec![
                    ("color".to_string(), loads("[1, 2, 3]").unwrap()),
                    ("ifc".to_string(), Value::Str("old".into())),
                ];
                let mut g: Vec<(String, Value)> = Vec::new();
                bim::tag_face(&mut f, &cls, &name, &id);
                bim::tag_face(&mut g, &cls, &name, &id);
                let tagged = text(&Value::Array(vec![kv(f.clone()), kv(g)]));
                bim::untag_face(&mut f);
                let got = obj(vec![
                    ("tagged", Value::Str(tagged)),
                    ("untagged", kv(f)),
                    ("group", bim::group_tag(&cls, &name)),
                ]);
                c.eq("tag", &s(&case, "id"), case.get("out").unwrap(), Ok(got));
            }
            "ifc_entry" => {
                let raw = load(&case, "in");
                let got = bim::ifc_from_entry(&raw).map(|o| opt(o.map(kv)));
                c.eq("ifc_entry", &s(&case, "in"), case.get("out").unwrap(), got);
            }
            other => panic!("{other}"),
        }
    }
    c.done();
}

fn hex(h: &str) -> Vec<u8> {
    (0..h.len() / 2).map(|i| u8::from_str_radix(&h[2 * i..2 * i + 2], 16).unwrap()).collect()
}

fn scratch() -> PathBuf {
    let dir = std::env::temp_dir().join(format!("ingetrazo-modeldata-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

#[test]
fn texture_data() {
    let mut c = Checker::default();
    let dir = scratch();
    let mut n = 0;
    for case in data("texture.jsonl") {
        match s(&case, "op").as_str() {
            "tex_meta" => {}
            "tex_file_name" => {
                let Value::Str(name) = load(&case, "in") else { panic!() };
                let got = Ok(Value::Str(texture::texture_file_name_for(&name, true)));
                c.eq("tex_file_name", &s(&case, "in"), case.get("out").unwrap(), got);
            }
            "tex_is_image_bytes" => {
                let got = Ok(Value::Bool(texture::is_image_bytes(&hex(&s(&case, "in")))));
                c.eq("is_image_bytes", &s(&case, "in"), case.get("out").unwrap(), got);
            }
            "tex_is_image_file" => {
                n += 1;
                let path = match case.get("head") {
                    Some(Value::Str(h)) => {
                        let p = dir.join(format!("f{n}{}", s(&case, "suffix")));
                        std::fs::write(&p, hex(h)).unwrap();
                        p
                    }
                    _ => dir.join("missing.png"),
                };
                let got = Ok(Value::Bool(texture::is_image_file(&path, &|_| false)));
                c.eq("is_image_file", &s(&case, "suffix"), case.get("out").unwrap(), got);
            }
            "tex_dict" => {
                let d = load(&case, "in");
                let input = s(&case, "in");
                let from = texture::Texture::from_dict(&d)
                    .map(|t| obj(vec![("as_dict", t.as_dict()), ("path", t.path.clone())]));
                c.eq("Texture.from_dict", &input, case.get("from").unwrap(), from);
                c.eq("flattened", &input, case.get("flattened").unwrap(), texture::flattened_texture(&d));
                c.eq("untinted", &input, case.get("untinted").unwrap(), texture::untinted_texture(&d));
            }
            other => panic!("{other}"),
        }
    }
    let _ = std::fs::remove_dir_all(&dir);
    c.done();
}

#[test]
fn face_attrs_data() {
    let mut c = Checker::default();
    for case in data("face_attrs.jsonl") {
        let input = s(&case, "in");
        let raw = load(&case, "in");
        let read = |v: &Value| face_attrs::from_entry(v).map(|o| opt(o.map(kv)));
        let write = |v: &Value| face_attrs::to_entry(v).map(kv);
        c.eq("write", &input, case.get("write").unwrap(), write(&raw));
        c.eq("read", &input, case.get("read").unwrap(), read(&raw));
        if let Some(w) = case.get("write_read") {
            let written = loads(&text(&write(&raw).unwrap())).unwrap();
            c.eq("write_read", &input, w, read(&written));
        }
        if let Some(w) = case.get("read_write") {
            let attrs = face_attrs::from_entry(&raw).unwrap().map(kv).unwrap_or(Value::Object(vec![]));
            c.eq("read_write", &input, w, write(&attrs));
        }
    }
    c.done();
}
