// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! The sheets of a document, as `Composicion.from_dict` then `to_dict` leave
//! them (core/composition.py). The model keeps a sheet as a JSON value, but
//! opening a document is not a copy: every record is rebuilt from its
//! dataclass, so a sheet saved by an older version gains the keys added
//! since (at their defaults), one from a newer version loses the keys this
//! version does not know, and a frame that still carries the old fixed scale
//! label turns it into a text block. Saving the document as it was read
//! would write something else than the Python app does.
//!
//! The field tables are generated (scripts/golden/make_composition_schema.py).

use ingetrazo_formats::pyjson::{self, Value};

use crate::composition_schema as sch;
use crate::composition_schema::Schema;
use crate::pyconv::{py_float, truthy, PyErr, PyResult};

/// `PT_TO_MM`.
const PT_TO_MM: f64 = 25.4 / 72.0;

/// `Composicion.from_dict(raw).to_dict()`; `None` when Python's `from_dict`
/// raises (the sheet is skipped with a warning).
pub fn normalize_sheet(raw: &Value) -> Option<Value> {
    match rebuild(raw) {
        Ok(v) => Some(v),
        // A value of a type these records do not model (a number for a
        // colour name): keep the sheet as it was rather than guess.
        Err(PyErr { kind: "NotModelled", .. }) => Some(raw.clone()),
        Err(_) => None,
    }
}

fn s(t: &str) -> Value {
    Value::Str(t.to_string())
}

fn default_of(json: &str) -> Value {
    pyjson::loads(json).expect("generated default")
}

/// `kind(**raw)` then `asdict`: the declared fields in order, from `raw`
/// where it has them and at their defaults where it does not.
fn build(schema: Schema, raw: &Value) -> Value {
    Value::Object(schema.iter().map(|(name, default)| ((*name).to_string(), raw.get(name).cloned().unwrap_or_else(|| default_of(default)))).collect())
}

/// `_items`: the records of a list; whatever is not a record is skipped.
fn items(schema: Schema, raw: Option<&Value>) -> Vec<Value> {
    match raw {
        Some(Value::Array(a)) => a.iter().filter(|r| matches!(r, Value::Object(_))).map(|r| build(schema, r)).collect(),
        _ => Vec::new(),
    }
}

fn set(obj: &mut Value, key: &str, v: Value) {
    if let Value::Object(items) = obj {
        match items.iter_mut().find(|(k, _)| k == key) {
            Some(slot) => slot.1 = v,
            None => items.push((key.to_string(), v)),
        }
    }
}

/// `str(x)` of the values a record may hold in a text field.
fn py_str(v: &Value) -> PyResult<String> {
    match v {
        Value::Str(t) => Ok(t.clone()),
        Value::Int(d) => Ok(d.clone()),
        Value::Bool(b) => Ok(if *b { "True" } else { "False" }.into()),
        Value::Null => Ok("None".into()),
        _ => Err(PyErr { kind: "NotModelled", msg: "str() of a list, dict or float".into() }),
    }
}

fn num(v: &Value) -> PyResult<f64> {
    py_float(v)
}

fn floats(v: Option<&Value>) -> PyResult<Vec<Value>> {
    match v {
        Some(Value::Array(a)) => a.iter().map(|x| Ok(Value::Float(num(x)?))).collect(),
        Some(x) if !truthy(x) => Ok(Vec::new()),
        None => Ok(Vec::new()),
        Some(_) => Err(PyErr { kind: "NotModelled", msg: "guides that are not a list".into() }),
    }
}

fn rebuild(raw: &Value) -> PyResult<Value> {
    if !matches!(raw, Value::Object(_)) {
        return Err(PyErr { kind: "AttributeError", msg: "sheet is not a record".into() });
    }
    let get = |k: &str, default: Value| raw.get(k).cloned().unwrap_or(default);
    let mut out: Vec<(String, Value)> = vec![
        ("name".into(), get("name", s("Lámina 1"))),
        ("paper".into(), get("paper", s("A4"))),
        ("landscape".into(), Value::Bool(raw.get("landscape").map(truthy).unwrap_or(true))),
        ("margin_mm".into(), Value::Float(match raw.get("margin_mm") {
            Some(m) => num(m)?,
            None => 10.0,
        })),
    ];
    let mut frames = items(sch::MARCOVISTA, raw.get("frames"));
    let mut texts = items(sch::TEXTOITEM, raw.get("texts"));
    let images = items(sch::IMAGENITEM, raw.get("images"));
    let scalebars = items(sch::BARRAESCALA, raw.get("scalebars"));
    let lists: [(&str, Schema); 10] = [
        ("nortes", sch::FLECHANORTE),
        ("leyendas", sch::LEYENDA),
        ("shapes", sch::FORMAITEM),
        ("cotas", sch::COTAITEM),
        ("cotas_ang", sch::COTAANGULARITEM),
        ("cotas_rad", sch::COTARADIALITEM),
        ("etiquetas", sch::ETIQUETAITEM),
        ("perfiles", sch::PERFILTERRENO),
        ("niveles", sch::NIVELITEM),
        ("llamadas", sch::LLAMADAITEM),
    ];
    let lists: Vec<(&str, Vec<Value>)> = lists.iter().map(|(k, sc)| (*k, items(sc, raw.get(k)))).collect();
    let cajetin = match raw.get("cajetin") {
        Some(c @ Value::Object(_)) => Some(cajetin(c)),
        _ => None,
    };
    let mut guides: Option<Value> = None;
    if let Some(g @ Value::Object(_)) = raw.get("guides") {
        let v = floats(g.get("v"))?;
        let h = floats(g.get("h"))?;
        if !v.is_empty() || !h.is_empty() {
            guides = Some(Value::Object(vec![("v".into(), Value::Array(v)), ("h".into(), Value::Array(h))]));
        }
    }
    let border = match raw.get("border") {
        Some(b @ Value::Object(_)) if b.get("on").is_some_and(truthy) => {
            let f = |k: &str, d: f64| -> PyResult<Value> { Ok(Value::Float(b.get(k).map(num).transpose()?.unwrap_or(d))) };
            let t = |k: &str, d: &str| -> PyResult<Value> { Ok(Value::Str(b.get(k).map(py_str).transpose()?.unwrap_or_else(|| d.to_string()))) };
            Some(Value::Object(vec![
                ("on".into(), Value::Bool(true)),
                ("mm".into(), f("mm", 0.5)?),
                ("color".into(), t("color", "#1e242c")?),
                ("radius_mm".into(), f("radius_mm", 0.0)?),
                ("style".into(), t("style", "single")?),
            ]))
        }
        _ => None,
    };
    migrate_fixed_scale_labels(&mut frames, &mut texts, &[&images, &scalebars], &lists, cajetin.as_ref())?;

    out.push(("frames".into(), Value::Array(frames)));
    if let Some(b) = border {
        out.push(("border".into(), b));
    }
    for (key, list) in [("texts", texts), ("images", images), ("scalebars", scalebars)] {
        if !list.is_empty() {
            out.push((key.into(), Value::Array(list)));
        }
    }
    for (key, list) in lists {
        if !list.is_empty() {
            out.push((key.into(), Value::Array(list)));
        }
    }
    if let Some(c) = cajetin {
        out.push(("cajetin".into(), c));
    }
    if let Some(g) = guides {
        out.push(("guides".into(), g));
    }
    Ok(Value::Object(out))
}

/// `Cajetin(**raw)` and its `__post_init__`: the rows of an old document's
/// fixed fields become `campos`.
fn cajetin(raw: &Value) -> Value {
    let mut c = build(sch::CAJETIN, raw);
    if !c.get("campos").is_some_and(truthy) {
        let row = |label: &str, attr: &str| Value::Array(vec![s(label), c.get(attr).cloned().unwrap_or(Value::Null)]);
        let rows = vec![row("PROYECTO", "proyecto"), row("AUTOR", "autor"), row("FECHA", "fecha"), row("ESCALA", "escala"), row("LÁMINA", "lamina")];
        set(&mut c, "campos", Value::Array(rows));
    }
    c
}

/// `_migrate_fixed_scale_labels`: a frame with the old `show_scale` flag
/// gets a text block bound to it, in the same place and size.
fn migrate_fixed_scale_labels(frames: &mut [Value], texts: &mut Vec<Value>, others: &[&Vec<Value>], lists: &[(&str, Vec<Value>)], cajetin: Option<&Value>) -> PyResult<()> {
    if !frames.iter().any(|f| f.get("show_scale").is_some_and(truthy)) {
        return Ok(());
    }
    let mut zs: Vec<f64> = Vec::new();
    let all = frames.iter().chain(texts.iter()).chain(others.iter().flat_map(|l| l.iter())).chain(lists.iter().flat_map(|(_, l)| l.iter())).chain(cajetin);
    for it in all {
        zs.push(match it.get("z") {
            Some(z) => num(z)?,
            None => 0.0,
        });
    }
    let mut z = zs.iter().cloned().fold(f64::NEG_INFINITY, f64::max) + 1.0;
    for f in frames.iter_mut() {
        if !f.get("show_scale").is_some_and(truthy) {
            continue;
        }
        set(f, "show_scale", Value::Bool(false));
        if !f.get("uid").is_some_and(truthy) {
            set(f, "uid", Value::Str(format!("{}{}", crate::scene::fresh_uid(), crate::scene::fresh_uid())));
        }
        let text_of = |k: &str, d: &str| -> PyResult<String> {
            match f.get(k) {
                Some(v) if truthy(v) => py_str(v),
                _ => Ok(d.to_string()),
            }
        };
        let template = text_of("scale_text", "ESC. 1:{n}")?;
        let text = template.replace("1:{n}", "{escala}").replace("{n}", "{escala}");
        let size_mm = match f.get("scale_mm") {
            Some(v) if truthy(v) => num(v)?,
            _ => 3.0,
        }
        .max(1.5);
        let pos = text_of("scale_pos", "under-right")?;
        let (w, h) = (40.0, size_mm * 1.4);
        let n = |k: &str| num(f.get(k).unwrap_or(&Value::Null));
        let (fx, fy, fw, fh) = (n("x_mm")?, n("y_mm")?, n("w_mm")?, n("h_mm")?);
        let (x, y, align, bg) = match pos.as_str() {
            "under-left" => (fx, fy + fh + 1.0, "left", ""),
            "inside-bl" => (fx + 1.0, fy + fh - h - 1.0, "left", "#ffffff"),
            "inside-br" => (fx + fw - w - 1.0, fy + fh - h - 1.0, "right", "#ffffff"),
            _ => (fx + fw - w, fy + fh + 1.0, "right", ""),
        };
        let mut t = build(sch::TEXTOITEM, &Value::Object(Vec::new()));
        for (k, v) in [
            ("x_mm", Value::Float(x)),
            ("y_mm", Value::Float(y)),
            ("w_mm", Value::Float(w)),
            ("text", Value::Str(text)),
            ("size_pt", Value::Float(size_mm / PT_TO_MM)),
            ("bold", Value::Bool(true)),
            ("align", s(align)),
            ("bg_color", s(bg)),
            ("bg_opacity", Value::Float(if bg.is_empty() { 1.0 } else { 0.86 })),
            ("frame_uid", f.get("uid").cloned().unwrap_or(Value::Null)),
            ("follow", Value::Bool(true)),
            ("z", Value::Float(z)),
        ] {
            set(&mut t, k, v);
        }
        texts.push(t);
        z += 1.0;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn norm(json: &str) -> String {
        let v = normalize_sheet(&pyjson::loads(json).unwrap()).unwrap();
        pyjson::dumps(&v, &pyjson::Options::compact().allow_nan(false)).unwrap()
    }

    #[test]
    fn an_empty_sheet_gets_its_defaults() {
        assert_eq!(norm("{}"), r#"{"name":"Lámina 1","paper":"A4","landscape":true,"margin_mm":10.0,"frames":[]}"#.replace("Lámina", "L\\u00e1mina"));
    }

    #[test]
    fn an_old_item_gains_the_new_keys_and_a_foreign_key_is_dropped() {
        let out = norm(r#"{"texts":[{"text":"x","from_the_future":1}]}"#);
        assert!(out.contains(r#""text":"x","size_pt":14.0"#), "{out}");
        assert!(out.contains(r#""list_name":"","hidden":false}"#), "{out}");
        assert!(!out.contains("from_the_future"));
    }

    #[test]
    fn the_title_block_rows_come_from_the_fixed_fields() {
        let out = norm(r#"{"cajetin":{"proyecto":"P"}}"#);
        assert!(out.contains(r#""campos":[["PROYECTO","P"],["AUTOR",""]"#), "{out}");
    }

    #[test]
    fn a_sheet_that_is_not_a_record_is_skipped() {
        assert!(normalize_sheet(&Value::Null).is_none());
        assert!(normalize_sheet(&pyjson::loads(r#"{"margin_mm":"wide"}"#).unwrap()).is_none());
    }
}
