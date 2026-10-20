// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! The translation layer against what `core.i18n` returns
//! (scripts/golden/make_i18n_fixtures.py): every key of every language with
//! several sets of field values (numbers, unicode text, the wrong type, a
//! missing field), errors compared by Python exception class, plus
//! `source_of` and the file-order key list of each catalog.

use std::path::{Path, PathBuf};

use ingetrazo_formats::pyjson::{loads, Value};
use ingetrazo_i18n::{Arg, Catalog};

fn data(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data").join(name)
}

fn get<'a>(v: &'a Value, k: &str) -> &'a Value {
    v.get(k).unwrap_or_else(|| panic!("missing key {k}"))
}

fn arr(v: &Value) -> &[Value] {
    match v {
        Value::Array(a) => a,
        _ => panic!("not an array"),
    }
}

fn string(v: &Value) -> &str {
    match v {
        Value::Str(s) => s,
        _ => panic!("not a string"),
    }
}

fn fnv(s: &str) -> String {
    let mut h: u32 = 0x811C_9DC5;
    for b in s.bytes() {
        h = (h ^ u32::from(b)).wrapping_mul(0x0100_0193);
    }
    format!("{h:08x}")
}

fn decode_args(v: &Value) -> Vec<(String, Arg)> {
    let Value::Object(items) = v else { panic!("args not an object") };
    items
        .iter()
        .map(|(name, a)| {
            let arg = if let Some(i) = a.get("i") {
                Arg::Int(i.as_i64().unwrap())
            } else if let Some(f) = a.get("f") {
                let bits = u64::from_str_radix(string(f), 16).unwrap();
                Arg::Float(f64::from_bits(bits))
            } else {
                Arg::Str(string(get(a, "s")).to_string())
            };
            (name.clone(), arg)
        })
        .collect()
}

#[test]
fn matches_python() {
    let text = std::fs::read_to_string(data("tr.jsonl")).unwrap();
    let mut lines = text.lines().map(|l| loads(l).unwrap());
    let head = lines.next().unwrap();
    let langs: Vec<String> = arr(get(&head, "langs")).iter().map(|v| string(v).to_string()).collect();
    let keys: Vec<String> = arr(get(&head, "keys")).iter().map(|v| string(v).to_string()).collect();
    let catalogs: Vec<Catalog> = langs.iter().map(|l| Catalog::load(l, None)).collect();

    let (mut checked, mut errors) = (0usize, 0usize);
    for rec in lines {
        if let Some(k) = rec.get("k") {
            let key = &keys[k.as_i64().unwrap() as usize];
            let argsets: Vec<Vec<(String, Arg)>> = arr(get(&rec, "a")).iter().map(decode_args).collect();
            for (li, lang) in langs.iter().enumerate() {
                let outs = arr(get(get(&rec, "o"), lang));
                assert_eq!(outs.len(), argsets.len());
                for (vi, want) in outs.iter().enumerate() {
                    let args: Vec<(&str, Arg)> =
                        argsets[vi].iter().map(|(n, a)| (n.as_str(), a.clone())).collect();
                    let got = catalogs[li].tr_args(key, &args);
                    let ctx = format!("{lang} variant {vi} key {key:?}");
                    match (&got, want.get("o"), want.get("e")) {
                        (Ok(g), Some(Value::Str(w)), _) => assert_eq!(g, w, "{ctx}"),
                        (Err(e), _, Some(Value::Str(w))) => {
                            assert_eq!(e.python_class(), w, "{ctx}");
                            errors += 1;
                        }
                        _ => panic!("{ctx}: rust {got:?} python {want:?}"),
                    }
                    checked += 1;
                }
            }
        } else {
            let lang = string(get(&rec, "lang"));
            let li = langs.iter().position(|l| l == lang).unwrap();
            let cat = &catalogs[li];
            assert_eq!(cat.language(), string(get(&rec, "current")));
            for row in arr(get(&rec, "plain")) {
                let r = arr(row);
                let key = &keys[r[0].as_i64().unwrap() as usize];
                assert_eq!(fnv(cat.tr(key)), string(&r[1]), "{lang} plain {key:?}");
                let got = cat.tr_args(key, &[("unrelated_zz", Arg::Int(1))]).unwrap();
                assert_eq!(fnv(&got), string(&r[2]), "{lang} plain+kw {key:?}");
                checked += 2;
            }
            let cat_keys = arr(get(&rec, "cat_keys"));
            assert_eq!(cat.len(), cat_keys.len(), "{lang} catalog size");
            let so = arr(get(&rec, "source_of"));
            for (i, (k, v)) in cat.entries().enumerate() {
                assert_eq!(k, keys[cat_keys[i].as_i64().unwrap() as usize], "{lang} order");
                let want = so[i].as_i64().unwrap();
                assert!(want >= 0);
                assert_eq!(cat.source_of(v), keys[want as usize], "{lang} source_of {v:?}");
            }
            // Text that is not a translation comes back as it is.
            assert_eq!(cat.source_of("\u{0}not a translation"), "\u{0}not a translation");
        }
    }
    assert!(checked > 50_000, "only {checked} comparisons");
    assert!(errors > 100, "only {errors} error cases");
    eprintln!("{checked} comparisons, {errors} Python errors matched");
}
