// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! The document blocks that Python rebuilds as objects on opening and writes
//! back from them: the display style (`Style.from_dict` / `to_dict`), the
//! shadow settings (`ShadowSettings`), the georeferencing datum and the map
//! capture. A document saved by an older version lacks the keys added since;
//! writing it as read would not be what the Python app writes. A block equal
//! to the default is not written at all (`if d != Style().to_dict()`).

use ingetrazo_formats::pyjson::Value;
use ingetrazo_georef::datum::SceneDatum;
use ingetrazo_georef::tiles::TileLayer;

use crate::pyconv::{py_float, py_int, truthy};

/// `core.style.FACE_MODES`.
const FACE_MODES: [&str; 6] = ["textures", "shaded", "hidden_line", "monochrome", "wireframe", "xray"];

fn f(x: f64) -> Value {
    Value::Float(x)
}

fn rgb(c: [f64; 3]) -> Value {
    Value::Array(c.iter().map(|x| f(*x)).collect())
}

fn obj(items: Vec<(&str, Value)>) -> Value {
    Value::Object(items.into_iter().map(|(k, v)| (k.to_string(), v)).collect())
}

/// Python's `==` on two JSON values: `1 == 1.0`.
fn py_eq(a: &Value, b: &Value) -> bool {
    let n = |v: &Value| match v {
        Value::Int(d) => d.parse::<f64>().ok(),
        Value::Float(x) => Some(*x),
        Value::Bool(b) => Some(*b as u8 as f64),
        _ => None,
    };
    match (a, b) {
        (Value::Array(x), Value::Array(y)) => x.len() == y.len() && x.iter().zip(y).all(|(p, q)| py_eq(p, q)),
        (Value::Object(x), Value::Object(y)) => x.len() == y.len() && x.iter().all(|(k, v)| b.get(k).is_some_and(|w| py_eq(v, w))),
        _ => match (n(a), n(b)) {
            (Some(p), Some(q)) => p == q,
            _ => a == b,
        },
    }
}

/// `tuple(raw.get(key, default))` written back with `list(...)`.
fn tuple_of(raw: &Value, key: &str, default: [f64; 3]) -> Option<Value> {
    match raw.get(key) {
        None => Some(rgb(default)),
        Some(Value::Array(a)) => Some(Value::Array(a.clone())),
        Some(_) => None,
    }
}

/// `_opt_rgb`: the first three components as floats, else `None`.
fn opt_rgb(v: Option<&Value>) -> Value {
    match v {
        Some(Value::Array(a)) => {
            let nums: Option<Vec<f64>> = a.iter().map(|x| py_float(x).ok()).collect();
            match nums {
                Some(n) if n.len() >= 3 => Value::Array(n[..3].iter().map(|x| f(*x)).collect()),
                _ => Value::Null,
            }
        }
        _ => Value::Null,
    }
}

fn style_dict(raw: &Value) -> Option<Value> {
    let flag = |k: &str, d: bool| raw.get(k).map(truthy).unwrap_or(d);
    let mode = match raw.get("face_mode") {
        Some(Value::Str(m)) if FACE_MODES.contains(&m.as_str()) => m.clone(),
        None => "textures".to_string(),
        _ => "textures".to_string(),
    };
    Some(obj(vec![
        ("name", raw.get("name").cloned().unwrap_or_else(|| Value::Str("Default".into()))),
        ("face_mode", Value::Str(mode)),
        ("edges", Value::Bool(flag("edges", true))),
        ("profiles", Value::Bool(flag("profiles", true))),
        ("back_edges", Value::Bool(flag("back_edges", false))),
        ("edge_color", tuple_of(raw, "edge_color", [0.13, 0.17, 0.23])?),
        ("front_color", tuple_of(raw, "front_color", [1.0, 1.0, 1.0])?),
        ("back_color", opt_rgb(raw.get("back_color"))),
        ("background", tuple_of(raw, "background", [0.90, 0.91, 0.92])?),
        ("sky", Value::Bool(flag("sky", true))),
        ("sky_color", tuple_of(raw, "sky_color", [0.925, 0.935, 0.945])?),
        ("ground_color", tuple_of(raw, "ground_color", [0.815, 0.820, 0.815])?),
        ("section_fill", Value::Bool(flag("section_fill", true))),
        ("section_fill_color", tuple_of(raw, "section_fill_color", [0.35, 0.37, 0.41])?),
    ]))
}

/// `Style.from_dict(raw).to_dict()`, or `None` when that is the default
/// style (the block is then left out of the file).
pub fn display_style(raw: &Value) -> Option<Value> {
    // Something these keys cannot hold (a colour that is not a list): keep it.
    let Some(d) = style_dict(raw) else { return Some(raw.clone()) };
    let default = style_dict(&Value::Object(Vec::new()))?;
    if py_eq(&d, &default) {
        None
    } else {
        Some(d)
    }
}

/// `ShadowSettings.from_dict(raw).to_dict()`, or `None` for the default.
pub fn shadows(raw: &Value) -> Option<Value> {
    let int = |key: &str, lo: i64, hi: i64, default: i64| -> i64 {
        match raw.get(key) {
            None => default,
            Some(v) => match py_int(v) {
                Ok(n) if (lo..=hi).contains(&n) => n,
                _ => default,
            },
        }
    };
    let dark = match raw.get("darkness") {
        None => 0.55,
        Some(v) => py_float(v).unwrap_or(0.55),
    };
    // `max(0.0, min(1.0, dark))` with Python's argument-order rules for NaN.
    let min = if dark < 1.0 { dark } else { 1.0 };
    let dark = if min > 0.0 { min } else { 0.0 };
    let mut items = vec![
        ("enabled", Value::Bool(raw.get("enabled").map(truthy).unwrap_or(false))),
        ("month", Value::int(int("month", 1, 12, 3))),
        ("day", Value::int(int("day", 1, 31, 21))),
        ("hour", Value::int(int("hour", 0, 23, 12))),
        ("minute", Value::int(int("minute", 0, 59, 0))),
        ("darkness", f(dark)),
    ];
    match raw.get("utc_offset") {
        None | Some(Value::Null) => {}
        Some(v) => match py_int(v) {
            Ok(n) => items.push(("utc_offset", Value::int(n))),
            Err(_) => return Some(raw.clone()),
        },
    }
    let d = obj(items);
    let default = obj(vec![
        ("enabled", Value::Bool(false)),
        ("month", Value::int(3)),
        ("day", Value::int(21)),
        ("hour", Value::int(12)),
        ("minute", Value::int(0)),
        ("darkness", f(0.55)),
    ]);
    if py_eq(&d, &default) {
        None
    } else {
        Some(d)
    }
}

/// `SceneDatum.from_dict(raw).to_dict()`.
pub fn datum(raw: &Value) -> Value {
    match SceneDatum::from_dict(raw) {
        Ok(d) => d.to_dict(),
        Err(_) => raw.clone(),
    }
}

/// `TileLayer.from_dict(raw).to_dict()`; a block that does not rebuild costs
/// the map, not the file (`None`).
pub fn tile_layer(raw: &Value) -> Option<Value> {
    if !truthy(raw) {
        return None;
    }
    match TileLayer::from_dict(raw) {
        Ok(t) => Some(t.to_dict()),
        Err(e) if e.kind == "NotModelled" => Some(raw.clone()),
        Err(_) => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ingetrazo_formats::pyjson;

    fn v(t: &str) -> Value {
        pyjson::loads(t).unwrap()
    }

    #[test]
    fn a_style_missing_back_edges_gains_it() {
        let d = display_style(&v(r#"{"name":"Mine","face_mode":"shaded"}"#)).unwrap();
        assert!(d.get("back_edges").is_some());
        assert_eq!(d.get("back_color"), Some(&Value::Null));
    }

    #[test]
    fn the_default_style_and_shadows_are_not_written() {
        assert!(display_style(&v("{}")).is_none());
        assert!(shadows(&v(r#"{"enabled":false,"month":3}"#)).is_none());
        assert!(shadows(&v(r#"{"enabled":true}"#)).is_some());
    }
}
