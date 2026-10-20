// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! The checker: every key of every embedded language file parses as a Python
//! format string and uses only the fields (and specs) the English source
//! has, and only the constructs this port implements.

use std::collections::BTreeMap;

use ingetrazo_formats::pyjson::{loads, Value};
use ingetrazo_i18n::format::fields;
use ingetrazo_i18n::{available_languages, embedded_json, Catalog, Arg, LANGUAGE_NAMES};

/// Every spec the shipped strings use; anything else needs a port first.
const SUPPORTED_SPECS: [&str; 8] = ["", "+.2f", "02d", ".0f", ".3f", ",.0f", ".1f", ".2f"];

#[test]
fn language_files() {
    let langs = available_languages();
    let names: Vec<&str> = LANGUAGE_NAMES.iter().map(|(c, _)| *c).collect();
    let mut sorted = names.clone();
    sorted.sort();
    assert_eq!(langs, sorted);
    for lang in &langs {
        // Every value is a string (Python would str() anything else).
        let Value::Object(items) = loads(embedded_json(lang).unwrap()).unwrap() else {
            panic!("{lang}: not an object")
        };
        for (k, v) in &items {
            assert!(matches!(v, Value::Str(_)), "{lang}: non-string value for {k:?}");
        }
    }
}

#[test]
fn every_key_formats_with_the_source_fields() {
    let mut problems: BTreeMap<String, Vec<String>> = BTreeMap::new();
    let mut counts = Vec::new();
    for lang in available_languages() {
        let cat = Catalog::load(&lang, None);
        counts.push((lang.clone(), cat.len()));
        for (key, tr) in cat.entries() {
            if key == "_comment" {
                continue;
            }
            let mut note = |m: String| problems.entry(lang.clone()).or_default().push(format!("{key:?}: {m}"));
            let src = match fields(key) {
                Ok(f) => f,
                Err(e) => {
                    // Literal braces in a key are fine while it is never
                    // formatted; the translation must then have none either.
                    if tr.contains(['{', '}']) {
                        note(format!("key does not parse ({e}) but the translation has braces"));
                    }
                    continue;
                }
            };
            let dst = match fields(tr) {
                Ok(f) => f,
                Err(e) => {
                    note(format!("translation does not parse: {e}"));
                    continue;
                }
            };
            for (name, spec) in src.iter().chain(dst.iter()) {
                if !SUPPORTED_SPECS.contains(&spec.as_str()) {
                    note(format!("unsupported spec {spec:?} of {{{name}}}"));
                }
            }
            for (name, spec) in &dst {
                match src.iter().find(|(n, _)| n == name) {
                    None => note(format!("translation uses {{{name}}}, absent from the source")),
                    Some((_, s)) if s != spec => {
                        note(format!("{{{name}}} has spec {spec:?}, the source has {s:?}"))
                    }
                    _ => {}
                }
            }
            // And it really formats with the source's field values.
            let args: Vec<(&str, Arg)> = src
                .iter()
                .map(|(n, s)| {
                    (n.as_str(), if s.ends_with('d') { Arg::Int(3) } else if s.ends_with('f') { Arg::Float(1.5) } else { Arg::Str("x".into()) })
                })
                .collect();
            if let Err(e) = cat.tr_args(key, &args) {
                note(format!("does not format: {e}"));
            }
        }
    }
    for (lang, n) in &counts {
        eprintln!("{lang}: {n} keys");
    }
    for (lang, list) in &problems {
        for p in list {
            eprintln!("PROBLEM {lang} {p}");
        }
    }
    assert!(problems.is_empty(), "{} problems", problems.values().map(Vec::len).sum::<usize>());
}
