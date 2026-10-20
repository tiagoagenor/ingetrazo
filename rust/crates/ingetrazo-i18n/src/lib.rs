// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! UI translation, a port of `core/i18n.py`.
//!
//! English is the source language: the text passed to [`tr`] is the English
//! string itself, so anything untranslated falls back to readable English.
//! The other languages are the existing flat `{english: translation}` maps in
//! `i18n/<lang>.json`, embedded at compile time (the app ships no loose
//! files). Python reads `i18n/` from the install folder only; the Rust port
//! additionally lets [`set_override_dir`] point at a folder whose
//! `<lang>.json` wins over the embedded one.
//!
//! `tr(text)` returns the translation as is; `tr_args(text, args)` runs
//! Python's `str.format` on it ([`format`]) and, when the translation does not
//! format (missing field, bad spec), formats the English source instead,
//! exactly like Python. If that fails too, the error is returned where Python
//! would have raised.

pub mod format;

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use ingetrazo_formats::pyjson::{float_repr, loads, Value};

pub use format::{format, Arg, FormatError};

/// How each language names itself in the Language menu and Preferences.
pub const LANGUAGE_NAMES: [(&str, &str); 10] = [
    ("en", "English"),
    ("es", "Español"),
    ("pt-BR", "Português (Brasil)"),
    ("zh-CN", "简体中文"),
    ("it", "Italiano"),
    ("id", "Bahasa Indonesia"),
    ("fr", "Français"),
    ("de", "Deutsch"),
    ("pl", "Polski"),
    ("ru", "Русский"),
];

/// The language files compiled into the binary, `(code, json)`, sorted by code.
const EMBEDDED: [(&str, &str); 10] = [
    ("de", include_str!("../../../../i18n/de.json")),
    ("en", include_str!("../../../../i18n/en.json")),
    ("es", include_str!("../../../../i18n/es.json")),
    ("fr", include_str!("../../../../i18n/fr.json")),
    ("id", include_str!("../../../../i18n/id.json")),
    ("it", include_str!("../../../../i18n/it.json")),
    ("pl", include_str!("../../../../i18n/pl.json")),
    ("pt-BR", include_str!("../../../../i18n/pt-BR.json")),
    ("ru", include_str!("../../../../i18n/ru.json")),
    ("zh-CN", include_str!("../../../../i18n/zh-CN.json")),
];

/// The embedded JSON text of one language (`None` for a code without a file).
pub fn embedded_json(lang: &str) -> Option<&'static str> {
    EMBEDDED.iter().find(|(c, _)| *c == lang).map(|(_, j)| *j)
}

/// The language the app starts in when nothing is saved, from the OS locale
/// (`main.py::_init_language`): Spanish for any Spanish locale, Brazilian
/// Portuguese for any Portuguese one (until a pt-PT catalogue exists),
/// English otherwise. `locale` is a BCP 47 / POSIX name such as `es-MX`,
/// `pt_PT.UTF-8` or `en-US`.
pub fn language_for_locale(locale: &str) -> &'static str {
    let primary = locale
        .split(['-', '_', '.', '@'])
        .next()
        .unwrap_or("")
        .to_ascii_lowercase();
    match primary.as_str() {
        "es" => "es",
        "pt" => "pt-BR",
        _ => "en",
    }
}

/// One language's `{english: translation}` map in file order (a repeated key
/// keeps its first position and its last value, like a Python dict).
#[derive(Clone, Debug, Default)]
pub struct Catalog {
    lang: String,
    entries: Vec<(String, String)>,
    index: HashMap<String, usize>,
}

/// Python's `str(value)` for a JSON value (`{str(k): str(v)}` in `set_language`).
fn py_str(v: &Value) -> String {
    match v {
        Value::Str(s) => s.clone(),
        Value::Null => "None".into(),
        Value::Bool(true) => "True".into(),
        Value::Bool(false) => "False".into(),
        Value::Int(s) => s.clone(),
        Value::Float(f) => float_repr(*f),
        // Never in the shipped files (a test pins it).
        Value::Array(_) | Value::Object(_) => String::new(),
    }
}

impl Catalog {
    /// The empty catalog: `tr` returns the source string unchanged.
    pub fn empty(lang: &str) -> Self {
        Catalog { lang: lang.to_string(), ..Default::default() }
    }

    /// Parses a language file; a broken file gives an empty catalog.
    pub fn from_json(lang: &str, json: &str) -> Self {
        let mut c = Catalog::empty(lang);
        if let Ok(Value::Object(items)) = loads(json) {
            for (k, v) in &items {
                c.insert(k.clone(), py_str(v));
            }
        }
        c
    }

    fn insert(&mut self, k: String, v: String) {
        match self.index.get(&k) {
            Some(&i) => self.entries[i].1 = v,
            None => {
                self.index.insert(k.clone(), self.entries.len());
                self.entries.push((k, v));
            }
        }
    }

    /// `set_language`: English or a language without a file is empty; the
    /// override folder, if any, is tried before the embedded file.
    pub fn load(lang: &str, override_dir: Option<&Path>) -> Self {
        let lang = if lang.is_empty() { "en" } else { lang };
        if lang == "en" {
            return Catalog::empty(lang);
        }
        if let Some(dir) = override_dir {
            let path = dir.join(format!("{lang}.json"));
            if path.exists() {
                return match std::fs::read(&path).ok().and_then(|b| String::from_utf8(b).ok()) {
                    Some(text) => Catalog::from_json(lang, &text),
                    None => Catalog::empty(lang),
                };
            }
        }
        match embedded_json(lang) {
            Some(json) => Catalog::from_json(lang, json),
            None => Catalog::empty(lang),
        }
    }

    pub fn language(&self) -> &str {
        &self.lang
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// `(english, translation)` in file order.
    pub fn entries(&self) -> impl Iterator<Item = (&str, &str)> {
        self.entries.iter().map(|(k, v)| (k.as_str(), v.as_str()))
    }

    pub fn get(&self, key: &str) -> Option<&str> {
        self.index.get(key).map(|&i| self.entries[i].1.as_str())
    }

    /// `tr(text)` without keyword arguments.
    pub fn tr<'a>(&'a self, text: &'a str) -> &'a str {
        self.get(text).unwrap_or(text)
    }

    /// `tr(text, **args)`; with no `args` it is [`Catalog::tr`] (Python formats
    /// only when keyword arguments are given, so literal braces stay safe).
    pub fn tr_args(&self, text: &str, args: &[(&str, Arg)]) -> Result<String, FormatError> {
        let out = self.tr(text);
        if args.is_empty() {
            return Ok(out.to_string());
        }
        match format(out, args) {
            Err(FormatError::Key(_) | FormatError::Index | FormatError::Value(_)) => {
                format(text, args)
            }
            other => other,
        }
    }

    /// The English source of a string `tr` produced; text that is not a
    /// translation comes back as it is. When two keys share a translation the
    /// later one wins.
    pub fn source_of<'a>(&'a self, text: &'a str) -> &'a str {
        self.entries
            .iter()
            .rev()
            .find(|(_, v)| v == text)
            .map(|(k, _)| k.as_str())
            .unwrap_or(text)
    }
}

/// Language codes with a file (embedded, plus any in the override folder).
pub fn available_languages() -> Vec<String> {
    let mut langs: Vec<String> = EMBEDDED.iter().map(|(c, _)| (*c).to_string()).collect();
    if let Some(dir) = override_dir() {
        if let Ok(rd) = std::fs::read_dir(dir) {
            for e in rd.flatten() {
                let p = e.path();
                if p.extension().and_then(|x| x.to_str()) == Some("json") {
                    if let Some(stem) = p.file_stem().and_then(|s| s.to_str()) {
                        langs.push(stem.to_string());
                    }
                }
            }
        }
    }
    langs.sort();
    langs.dedup();
    langs
}

struct State {
    catalog: Catalog,
    override_dir: Option<PathBuf>,
}

static STATE: Mutex<Option<State>> = Mutex::new(None);

fn with_state<R>(f: impl FnOnce(&mut State) -> R) -> R {
    let mut guard = STATE.lock().unwrap_or_else(|e| e.into_inner());
    let st = guard.get_or_insert_with(|| State {
        catalog: Catalog::empty("en"),
        override_dir: None,
    });
    f(st)
}

/// A folder whose `<lang>.json` files win over the embedded ones; applies to
/// the next [`set_language`].
pub fn set_override_dir(dir: Option<PathBuf>) {
    with_state(|s| s.override_dir = dir);
}

fn override_dir() -> Option<PathBuf> {
    with_state(|s| s.override_dir.clone())
}

/// Activates `lang` for the following [`tr`] calls (`core.i18n.set_language`).
pub fn set_language(lang: &str) {
    with_state(|s| s.catalog = Catalog::load(lang, s.override_dir.as_deref()));
}

pub fn current_language() -> String {
    with_state(|s| s.catalog.language().to_string())
}

/// `tr(text)` in the active language.
pub fn tr(text: &str) -> String {
    with_state(|s| s.catalog.tr(text).to_string())
}

/// `tr(text, **args)` in the active language.
pub fn tr_args(text: &str, args: &[(&str, Arg)]) -> Result<String, FormatError> {
    with_state(|s| s.catalog.tr_args(text, args))
}

/// `source_of(text)` in the active language.
pub fn source_of(text: &str) -> String {
    with_state(|s| s.catalog.source_of(text).to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn locale_choice() {
        assert_eq!(language_for_locale("es-MX"), "es");
        assert_eq!(language_for_locale("pt_PT.UTF-8"), "pt-BR");
        assert_eq!(language_for_locale("en-US"), "en");
        assert_eq!(language_for_locale("de-DE"), "en");
    }
}
