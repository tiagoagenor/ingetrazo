// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! What Python does with a JSON value on its way into a `from_dict`: the
//! truthiness test, `float(x)`, `int(x)`, `QVector3D(*x)`. The annotation
//! classes (dimension, text label, guide, section plane, saved view, image
//! plane) read their files through these, so a value the Python app accepts
//! is read the same here, and one it refuses is refused with the same
//! exception name.

use ingetrazo_formats::pyjson::Value;
use ingetrazo_math::V3;

/// A Python exception: its class name (`KeyError`, `TypeError`…) and a
/// message. `NotModelled` marks a value Python would keep as an arbitrary
/// object that these data classes do not hold (a number where a name is
/// expected, for instance).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PyErr {
    pub kind: &'static str,
    pub msg: String,
}

pub type PyResult<T> = Result<T, PyErr>;

pub fn err<T>(kind: &'static str, msg: impl Into<String>) -> PyResult<T> {
    Err(PyErr { kind, msg: msg.into() })
}

/// Python's `bool(x)` of a JSON value.
pub fn truthy(v: &Value) -> bool {
    match v {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Int(s) => s != "0",
        Value::Float(x) => *x != 0.0,
        Value::Str(s) => !s.is_empty(),
        Value::Array(a) => !a.is_empty(),
        Value::Object(o) => !o.is_empty(),
    }
}

/// `raw.get(key)` of a dict; `raw` anything else is an `AttributeError`.
pub fn get<'a>(raw: &'a Value, key: &str) -> PyResult<Option<&'a Value>> {
    match raw {
        Value::Object(_) => Ok(raw.get(key)),
        _ => err("AttributeError", "object has no attribute 'get'"),
    }
}

/// `raw[key]` of a dict: `KeyError` when absent.
pub fn index<'a>(raw: &'a Value, key: &str) -> PyResult<&'a Value> {
    match raw {
        Value::Object(_) => raw.get(key).ok_or_else(|| PyErr { kind: "KeyError", msg: key.to_string() }),
        _ => err("TypeError", "object is not subscriptable"),
    }
}

/// `float(x)` of an int, a float or a bool (`QVector3D`'s own conversion);
/// an int beyond double range is an `OverflowError`.
pub fn number(v: &Value) -> PyResult<f64> {
    match v {
        Value::Float(x) => Ok(*x),
        Value::Bool(b) => Ok(if *b { 1.0 } else { 0.0 }),
        Value::Int(s) => match s.parse::<f64>() {
            Ok(x) if x.is_finite() => Ok(x),
            _ => err("OverflowError", "int too large to convert to float"),
        },
        _ => err("TypeError", "a number is required"),
    }
}

/// `float(x)`: numbers as above, and text such as `"1.5"`.
pub fn py_float(v: &Value) -> PyResult<f64> {
    match v {
        Value::Str(s) => s
            .trim()
            .parse::<f64>()
            .or_else(|_| err("ValueError", format!("could not convert string to float: {s:?}"))),
        Value::Null | Value::Array(_) | Value::Object(_) => err("TypeError", "float() argument must be a string or a real number"),
        _ => number(v),
    }
}

/// `int(x)` for the uids a file may carry.
pub fn py_int(v: &Value) -> PyResult<i64> {
    match v {
        Value::Bool(b) => Ok(*b as i64),
        Value::Int(s) => s.parse::<i64>().or_else(|_| err("OverflowError", "uid beyond 64 bits")),
        Value::Float(x) => {
            if x.is_nan() {
                err("ValueError", "cannot convert float NaN to integer")
            } else if x.is_infinite() || x.abs() >= 9.2e18 {
                err("OverflowError", "cannot convert float to integer")
            } else {
                Ok(x.trunc() as i64)
            }
        }
        Value::Str(s) => s.trim().parse::<i64>().or_else(|_| err("ValueError", format!("invalid literal for int(): {s:?}"))),
        _ => err("TypeError", "int() argument must be a string or a real number"),
    }
}

/// The items `*x` unpacks: a list's elements, a string's characters, a
/// dict's keys. Anything else is a `TypeError`.
fn unpack(v: &Value) -> PyResult<Vec<Value>> {
    match v {
        Value::Array(a) => Ok(a.clone()),
        Value::Str(s) => Ok(s.chars().map(|c| Value::Str(c.to_string())).collect()),
        Value::Object(o) => Ok(o.iter().map(|(k, _)| Value::Str(k.clone())).collect()),
        _ => err("TypeError", "argument after * must be an iterable"),
    }
}

/// `QVector3D(*x)`: three numbers, or none at all (the null vector). PySide
/// rounds each double to float32, as the C++ cast does.
pub fn qvec3(v: &Value) -> PyResult<V3> {
    let items = unpack(v)?;
    match items.len() {
        0 => Ok(V3(0.0, 0.0, 0.0)),
        3 => Ok(V3(number(&items[0])? as f32, number(&items[1])? as f32, number(&items[2])? as f32)),
        _ => err("TypeError", "QVector3D takes 0 or 3 arguments"),
    }
}

/// `list(x)` of the vector's three coordinates, as `to_dict` writes them:
/// Python floats holding the float32 values.
pub fn v3_value(v: V3) -> Value {
    Value::Array(vec![Value::Float(v.0 as f64), Value::Float(v.1 as f64), Value::Float(v.2 as f64)])
}

/// A `str` field: a JSON string, or any falsy value standing for `""`
/// (written the same, since a falsy name is never written).
pub fn text_or_empty(v: Option<&Value>) -> PyResult<String> {
    match v {
        None => Ok(String::new()),
        Some(Value::Str(s)) => Ok(s.clone()),
        Some(x) if !truthy(x) => Ok(String::new()),
        Some(_) => err("NotModelled", "a text field holds something that is not text"),
    }
}

/// `x or None` of a text field: a non-empty string, else `None`.
pub fn text_or_none(v: Option<&Value>) -> PyResult<Option<String>> {
    match v {
        None => Ok(None),
        Some(Value::Str(s)) if !s.is_empty() => Ok(Some(s.clone())),
        Some(x) if !truthy(x) => Ok(None),
        Some(_) => err("NotModelled", "a text field holds something that is not text"),
    }
}

/// `{"k": v, …}` in the order given.
pub fn object(items: Vec<(&str, Value)>) -> Value {
    Value::Object(items.into_iter().map(|(k, v)| (k.to_string(), v)).collect())
}

/// `dict(x) if x else None`: a non-empty dict is copied, a falsy value is
/// `None`.
pub fn dict_or_none(v: Option<&Value>) -> PyResult<Option<Value>> {
    match v {
        None => Ok(None),
        Some(x) if !truthy(x) => Ok(None),
        Some(Value::Object(o)) => Ok(Some(Value::Object(o.clone()))),
        Some(_) => err("TypeError", "cannot convert to dict"),
    }
}

/// Iterating a payload block that should be a list: a list's items, an
/// empty dict's none, anything else a `TypeError`.
pub fn iter_list(v: &Value) -> PyResult<&[Value]> {
    match v {
        Value::Array(a) => Ok(a),
        Value::Object(o) if o.is_empty() => Ok(&[]),
        _ => err("TypeError", "object is not iterable"),
    }
}
