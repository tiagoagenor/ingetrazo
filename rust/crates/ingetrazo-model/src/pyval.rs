// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! The little bit of Python's data model the document classes lean on.
//!
//! `Material.from_dict`, `Layer.from_dict` and the igz readers take whatever
//! JSON a file holds and apply `tuple()`, `dict()`, `float()`, `bool()` and
//! `==` to it. To accept and refuse the same documents as Python, those
//! builtins are reproduced here over [`Value`], together with the class of
//! the exception Python would raise ([`PyError::kind`]).

use ingetrazo_formats::pyjson::{float_repr, Value};

/// A Python exception: its class name and a message (the message is for
/// humans; parity is on the class).
#[derive(Clone, Debug, PartialEq)]
pub struct PyError {
    pub kind: &'static str,
    pub msg: String,
}

pub type PyResult<T> = Result<T, PyError>;

pub fn err<T>(kind: &'static str, msg: impl Into<String>) -> PyResult<T> {
    Err(PyError { kind, msg: msg.into() })
}

pub fn s(text: &str) -> Value {
    Value::Str(text.to_string())
}

/// `bool(value)`.
pub fn truthy(v: &Value) -> bool {
    match v {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Int(d) => d != "0" && d != "-0",
        Value::Float(x) => *x != 0.0,
        Value::Str(t) => !t.is_empty(),
        Value::Array(a) => !a.is_empty(),
        Value::Object(o) => !o.is_empty(),
    }
}

fn type_name(v: &Value) -> &'static str {
    match v {
        Value::Null => "NoneType",
        Value::Bool(_) => "bool",
        Value::Int(_) => "int",
        Value::Float(_) => "float",
        Value::Str(_) => "str",
        Value::Array(_) => "list",
        Value::Object(_) => "dict",
    }
}

/// `value.get(key)` for something that must be a dict: `Ok(None)` when the
/// key is missing, `AttributeError` when `value` is not a dict.
pub fn get<'a>(v: &'a Value, key: &str) -> PyResult<Option<&'a Value>> {
    match v {
        Value::Object(_) => Ok(v.get(key)),
        _ => err("AttributeError", format!("'{}' object has no attribute 'get'", type_name(v))),
    }
}

/// `value.get(key, default)`.
pub fn get_or(v: &Value, key: &str, default: Value) -> PyResult<Value> {
    Ok(get(v, key)?.cloned().unwrap_or(default))
}

/// Sets `key` on an ordered dict: an existing key keeps its place.
pub fn set(items: &mut Vec<(String, Value)>, key: &str, value: Value) {
    match items.iter_mut().find(|(k, _)| k == key) {
        Some(slot) => slot.1 = value,
        None => items.push((key.to_string(), value)),
    }
}

/// `iter(value)`: the items a `for` loop or `tuple()` sees.
pub fn iter(v: &Value) -> PyResult<Vec<Value>> {
    match v {
        Value::Array(a) => Ok(a.clone()),
        Value::Str(t) => Ok(t.chars().map(|c| Value::Str(c.to_string())).collect()),
        Value::Object(o) => Ok(o.iter().map(|(k, _)| Value::Str(k.clone())).collect()),
        _ => err("TypeError", format!("'{}' object is not iterable", type_name(v))),
    }
}

/// A dict key as Python's `json.dumps` writes it (`int` -> `"1"`,
/// `True` -> `"true"`, `None` -> `"null"`); containers are unhashable.
fn key_text(v: &Value) -> PyResult<String> {
    Ok(match v {
        Value::Str(t) => t.clone(),
        Value::Int(d) => d.clone(),
        Value::Float(x) if x.is_finite() => float_repr(*x),
        Value::Float(x) if x.is_nan() => "NaN".to_string(),
        Value::Float(x) => if *x > 0.0 { "Infinity" } else { "-Infinity" }.to_string(),
        Value::Bool(b) => if *b { "true" } else { "false" }.to_string(),
        Value::Null => "null".to_string(),
        _ => return err("TypeError", format!("unhashable type: '{}'", type_name(v))),
    })
}

/// `dict(value)`: a dict is copied; a sequence is read as `[key, value]`
/// pairs (anything of length two: a list, a two-character string, a dict of
/// two keys). Non-string keys are stored as the text `json.dumps` writes.
pub fn dict(v: &Value) -> PyResult<Vec<(String, Value)>> {
    if let Value::Object(o) = v {
        return Ok(o.clone());
    }
    let items = match v {
        Value::Array(_) | Value::Str(_) => iter(v)?,
        _ => return err("TypeError", format!("'{}' object is not iterable", type_name(v))),
    };
    // Identity of each key as Python hashes it (`1 == 1.0 == True`), apart
    // from the text it is written with: `{1: a, "1": b}` has two keys that
    // both dump as "1".
    let mut keys: Vec<Value> = Vec::new();
    let mut out: Vec<(String, Value)> = Vec::new();
    for (i, item) in items.iter().enumerate() {
        let pair = match item {
            Value::Array(_) | Value::Str(_) | Value::Object(_) => iter(item)?,
            _ => {
                return err(
                    "TypeError",
                    format!("cannot convert dictionary update sequence element #{i} to a sequence"),
                )
            }
        };
        if pair.len() != 2 {
            return err(
                "ValueError",
                format!(
                    "dictionary update sequence element #{i} has length {}; 2 is required",
                    pair.len()
                ),
            );
        }
        let text = key_text(&pair[0])?;
        match keys.iter().position(|k| eq(k, &pair[0])) {
            Some(at) => out[at].1 = pair[1].clone(),
            None => {
                keys.push(pair[0].clone());
                out.push((text, pair[1].clone()));
            }
        }
    }
    Ok(out)
}

/// `tuple(value)` / `list(value)`.
pub fn tuple(v: &Value) -> PyResult<Vec<Value>> {
    iter(v)
}

/// `float(text)` as CPython parses it (ASCII digits; surrounding whitespace
/// and single underscores between digits allowed).
fn float_from_str(text: &str) -> PyResult<f64> {
    let bad = || err("ValueError", format!("could not convert string to float: {text:?}"));
    let t = text.trim();
    if t.is_empty() {
        return bad();
    }
    let chars: Vec<char> = t.chars().collect();
    let mut clean = String::with_capacity(t.len());
    for (i, &c) in chars.iter().enumerate() {
        if c == '_' {
            let before = i > 0 && chars[i - 1].is_ascii_digit();
            let after = i + 1 < chars.len() && chars[i + 1].is_ascii_digit();
            if !(before && after) {
                return bad();
            }
        } else {
            clean.push(c);
        }
    }
    // Rust also reads `inf`, `infinity` and `nan` in any case, as Python.
    if !clean.is_ascii() {
        return bad();
    }
    match clean.parse::<f64>() {
        Ok(x) => Ok(x),
        Err(_) => bad(),
    }
}

/// `float(value)`.
pub fn float(v: &Value) -> PyResult<f64> {
    match v {
        Value::Bool(b) => Ok(if *b { 1.0 } else { 0.0 }),
        Value::Float(x) => Ok(*x),
        Value::Int(d) => {
            let x: f64 = d.parse().map_err(|_| PyError { kind: "ValueError", msg: d.clone() })?;
            if x.is_infinite() {
                return err("OverflowError", "int too large to convert to float");
            }
            Ok(x)
        }
        Value::Str(t) => float_from_str(t),
        _ => err(
            "TypeError",
            format!("float() argument must be a string or a real number, not '{}'", type_name(v)),
        ),
    }
}

/// A number as exact data for comparison: Python compares `int` and `float`
/// by value, not by converting one to the other.
enum Num {
    I(i128),
    F(f64),
}

fn num(v: &Value) -> Option<Num> {
    match v {
        Value::Bool(b) => Some(Num::I(*b as i128)),
        Value::Int(d) => match d.parse::<i128>() {
            Ok(i) => Some(Num::I(i)),
            Err(_) => Some(Num::F(d.parse::<f64>().unwrap_or(f64::NAN))),
        },
        Value::Float(x) => Some(Num::F(*x)),
        _ => None,
    }
}

fn num_eq(a: &Num, b: &Num) -> bool {
    match (a, b) {
        (Num::I(x), Num::I(y)) => x == y,
        (Num::F(x), Num::F(y)) => x == y,
        (Num::I(i), Num::F(f)) | (Num::F(f), Num::I(i)) => {
            f.is_finite() && f.fract() == 0.0 && f.abs() < 1.0e38 && (*f as i128) == *i
        }
    }
}

/// `a == b`: numbers by value (`1 == 1.0 == True`), dicts without regard to
/// order, sequences item by item.
pub fn eq(a: &Value, b: &Value) -> bool {
    if let (Some(x), Some(y)) = (num(a), num(b)) {
        return num_eq(&x, &y);
    }
    match (a, b) {
        (Value::Null, Value::Null) => true,
        (Value::Str(x), Value::Str(y)) => x == y,
        (Value::Array(x), Value::Array(y)) => seq_eq(x, y),
        (Value::Object(x), Value::Object(y)) => {
            x.len() == y.len()
                && x.iter().all(|(k, v)| y.iter().find(|(k2, _)| k2 == k).is_some_and(|(_, w)| eq(v, w)))
        }
        _ => false,
    }
}

/// `a == b` on two tuples/lists given as item slices.
pub fn seq_eq(a: &[Value], b: &[Value]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(p, q)| eq(p, q))
}

/// Whether `hash(value)` would succeed (a dict key).
pub fn hashable(v: &Value) -> PyResult<()> {
    match v {
        Value::Array(_) | Value::Object(_) => {
            err("TypeError", format!("unhashable type: '{}'", type_name(v)))
        }
        _ => Ok(()),
    }
}

/// `str(value)` for scalars (what an f-string prints).
pub fn str_of(v: &Value) -> PyResult<String> {
    match v {
        Value::Str(t) => Ok(t.clone()),
        Value::Null => Ok("None".to_string()),
        Value::Bool(b) => Ok(if *b { "True" } else { "False" }.to_string()),
        Value::Int(d) => Ok(d.clone()),
        Value::Float(x) if x.is_finite() => Ok(float_repr(*x)),
        Value::Float(x) if x.is_nan() => Ok("nan".to_string()),
        Value::Float(x) => Ok(if *x > 0.0 { "inf" } else { "-inf" }.to_string()),
        _ => err("TypeError", "unsupported"),
    }
}

/// `isinstance(value, int)` (a bool is an int) as a number, or `None`.
pub fn as_int(v: &Value) -> Option<i128> {
    match v {
        Value::Bool(b) => Some(*b as i128),
        Value::Int(d) => d.parse().ok(),
        _ => None,
    }
}

/// An `int` as a [`Value`].
pub fn int_value(i: i128) -> Value {
    Value::Int(i.to_string())
}
