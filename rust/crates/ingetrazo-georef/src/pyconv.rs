// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! What Python does with a JSON value on its way into a `from_dict` or a
//! parser: truthiness, `float(x)`, `int(x)`, `str(x)`, `QVector3D(*x)`, plus
//! the string helpers whose Unicode rules differ from Rust's (`str.strip`,
//! `str.split`, `str.splitlines`, `float("1_0.5")`, `"%.3f"`).
//!
//! The georef data classes read their files through these, so a value the
//! Python app accepts is read the same here, and one it refuses is refused
//! with the same exception name. (A copy of the helpers of
//! `ingetrazo-model`'s `pyconv`: the model will depend on this crate for
//! `Scene.geo_paths`, so this crate cannot depend on the model.)

use ingetrazo_formats::pyjson::{float_repr, Value};
use ingetrazo_math::V3;

/// A Python exception: its class name and a message. `NotModelled` marks a
/// value Python would keep as an arbitrary object that these data classes do
/// not hold (a number where a name is expected, for instance).
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
        Value::Str(s) => float_str(s).ok_or_else(|| PyErr {
            kind: "ValueError",
            msg: format!("could not convert string to float: {s:?}"),
        }),
        Value::Null | Value::Array(_) | Value::Object(_) => {
            err("TypeError", "float() argument must be a string or a real number")
        }
        _ => number(v),
    }
}

/// `int(x)` as the data classes use it (`TileLayer.zoom`).
pub fn py_int(v: &Value) -> PyResult<i64> {
    match v {
        Value::Bool(b) => Ok(*b as i64),
        Value::Int(s) => s.parse::<i64>().or_else(|_| err("OverflowError", "int beyond 64 bits")),
        Value::Float(x) => {
            if x.is_nan() {
                err("ValueError", "cannot convert float NaN to integer")
            } else if x.is_infinite() || x.abs() >= 9.2e18 {
                err("OverflowError", "cannot convert float to integer")
            } else {
                Ok(x.trunc() as i64)
            }
        }
        Value::Str(s) => {
            let t = py_strip(s);
            let bad = t.is_empty() || t.contains("__") || t.starts_with('_') || t.ends_with('_');
            let t2: String = t.chars().filter(|&c| c != '_').collect();
            if bad {
                return err("ValueError", format!("invalid literal for int(): {s:?}"));
            }
            t2.parse::<i64>().or_else(|_| err("ValueError", format!("invalid literal for int(): {s:?}")))
        }
        _ => err("TypeError", "int() argument must be a string or a real number"),
    }
}

/// The items `*x` unpacks or `for i in x` walks: a list's elements, a
/// string's characters, a dict's keys. Anything else is a `TypeError`.
pub fn unpack(v: &Value) -> PyResult<Vec<Value>> {
    match v {
        Value::Array(a) => Ok(a.clone()),
        Value::Str(s) => Ok(s.chars().map(|c| Value::Str(c.to_string())).collect()),
        Value::Object(o) => Ok(o.iter().map(|(k, _)| Value::Str(k.clone())).collect()),
        _ => err("TypeError", "object is not iterable"),
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

/// A `str` field: a JSON string, or any falsy value standing for `""`.
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

/// A text field read with `raw.get(key, default)` that must be a string.
pub fn text_default(v: Option<&Value>, default: &str) -> PyResult<String> {
    match v {
        None => Ok(default.to_string()),
        Some(Value::Str(s)) => Ok(s.clone()),
        Some(_) => err("NotModelled", "a text field holds something that is not text"),
    }
}

/// `{"k": v, ...}` in the order given.
pub fn object(items: Vec<(&str, Value)>) -> Value {
    Value::Object(items.into_iter().map(|(k, v)| (k.to_string(), v)).collect())
}

// ---- str helpers ------------------------------------------------------------

/// `str.isspace` of one character: Unicode White_Space plus the ASCII
/// separators U+001C..U+001F that Python also counts.
pub fn is_py_space(c: char) -> bool {
    c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c)
}

/// `str.strip()`.
pub fn py_strip(s: &str) -> &str {
    s.trim_matches(is_py_space)
}

/// `str.split()` (no argument): runs of whitespace separate, no empty items.
pub fn py_split_ws(s: &str) -> Vec<&str> {
    s.split(is_py_space).filter(|t| !t.is_empty()).collect()
}

/// `str.splitlines()`.
pub fn py_splitlines(s: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let mut start = 0;
    let mut it = s.char_indices().peekable();
    while let Some((i, c)) = it.next() {
        let is_break = matches!(
            c,
            '\n' | '\r'
                | '\u{0b}'
                | '\u{0c}'
                | '\u{1c}'
                | '\u{1d}'
                | '\u{1e}'
                | '\u{85}'
                | '\u{2028}'
                | '\u{2029}'
        );
        if is_break {
            out.push(&s[start..i]);
            let mut end = i + c.len_utf8();
            if c == '\r' {
                if let Some(&(j, '\n')) = it.peek() {
                    it.next();
                    end = j + 1;
                }
            }
            start = end;
        }
    }
    if start < s.len() {
        out.push(&s[start..]);
    }
    out
}

/// First code points of the Unicode decimal-digit (Nd) runs of ten, as
/// Unicode 15.0 (the data of the Python this port follows) defines them.
const ND_ZEROS: [u32; 68] = [
    0x660, 0x6f0, 0x7c0, 0x966, 0x9e6, 0xa66, 0xae6, 0xb66, 0xbe6, 0xc66, 0xce6, 0xd66, 0xde6, 0xe50, 0xed0,
    0xf20, 0x1040, 0x1090, 0x17e0, 0x1810, 0x1946, 0x19d0, 0x1a80, 0x1a90, 0x1b50, 0x1bb0, 0x1c40, 0x1c50,
    0xa620, 0xa8d0, 0xa900, 0xa9d0, 0xa9f0, 0xaa50, 0xabf0, 0xff10, 0x104a0, 0x10d30, 0x11066, 0x110f0,
    0x11136, 0x111d0, 0x112f0, 0x11450, 0x114d0, 0x11650, 0x116c0, 0x11730, 0x118e0, 0x11950, 0x11c50,
    0x11d50, 0x11da0, 0x11f50, 0x16a60, 0x16ac0, 0x16b50, 0x1d7ce, 0x1d7d8, 0x1d7e2, 0x1d7ec, 0x1d7f6,
    0x1e140, 0x1e2f0, 0x1e4f0, 0x1e950, 0x1fbf0, 0x30,
];

/// The decimal value of a Unicode digit (`unicodedata.decimal`), or `None`.
fn decimal_digit(c: char) -> Option<char> {
    if c.is_ascii_digit() {
        return Some(c);
    }
    let cp = c as u32;
    for &z in &ND_ZEROS {
        if cp >= z && cp < z + 10 {
            return char::from_digit(cp - z, 10);
        }
    }
    None
}

/// Python's `float(s)` of a string: surrounding whitespace, a sign, digits
/// (any Unicode decimal digit) with single underscores between digits,
/// `inf`/`infinity`/`nan` in any case.
pub fn float_str(s: &str) -> Option<f64> {
    let t = py_strip(s);
    if t.is_empty() {
        return None;
    }
    // Unicode digits become ASCII ones first.
    let chars: Vec<char> = t.chars().map(|c| decimal_digit(c).unwrap_or(c)).collect();
    let mut cleaned = String::with_capacity(t.len());
    for (i, &c) in chars.iter().enumerate() {
        if c == '_' {
            let before = i > 0 && chars[i - 1].is_ascii_digit();
            let after = i + 1 < chars.len() && chars[i + 1].is_ascii_digit();
            if !(before && after) {
                return None;
            }
        } else {
            cleaned.push(c);
        }
    }
    if !cleaned.is_ascii() {
        return None;
    }
    let body = cleaned.trim_start_matches(['+', '-']);
    if cleaned.len() - body.len() > 1 {
        return None;
    }
    let lower = body.to_ascii_lowercase();
    if lower.starts_with(|c: char| c.is_ascii_alphabetic())
        && !(lower == "inf" || lower == "infinity" || lower == "nan")
    {
        return None;
    }
    cleaned.parse::<f64>().ok()
}

/// `str(x)` of a JSON value (containers as Python's repr).
pub fn py_str(v: &Value) -> String {
    match v {
        Value::Str(s) => s.clone(),
        _ => py_repr(v),
    }
}

fn str_repr(s: &str) -> String {
    let quote = if s.contains('\'') && !s.contains('"') { '"' } else { '\'' };
    let mut out = String::new();
    out.push(quote);
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c == quote => {
                out.push('\\');
                out.push(c);
            }
            c if (c as u32) < 0x20 || c as u32 == 0x7f => out.push_str(&format!("\\x{:02x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push(quote);
    out
}

/// `repr(x)` of a JSON value.
pub fn py_repr(v: &Value) -> String {
    match v {
        Value::Null => "None".to_string(),
        Value::Bool(true) => "True".to_string(),
        Value::Bool(false) => "False".to_string(),
        Value::Int(s) => s.clone(),
        Value::Float(x) => {
            if x.is_nan() {
                "nan".to_string()
            } else if x.is_infinite() {
                if *x > 0.0 {
                    "inf".to_string()
                } else {
                    "-inf".to_string()
                }
            } else {
                float_repr(*x)
            }
        }
        Value::Str(s) => str_repr(s),
        Value::Array(a) => format!("[{}]", a.iter().map(py_repr).collect::<Vec<_>>().join(", ")),
        Value::Object(o) => format!(
            "{{{}}}",
            o.iter().map(|(k, v)| format!("{}: {}", str_repr(k), py_repr(v))).collect::<Vec<_>>().join(", ")
        ),
    }
}

/// `f"{x:.3f}"`: three decimals, exactly rounded (half to even on the exact
/// binary value), `nan`/`inf` spelled as Python does.
pub fn fmt_f3(x: f64) -> String {
    if x.is_nan() {
        "nan".to_string()
    } else if x.is_infinite() {
        if x > 0.0 {
            "inf".to_string()
        } else {
            "-inf".to_string()
        }
    } else {
        format!("{x:.3}")
    }
}

/// Python's `round(x, 3)` of a float (correctly rounded decimal, half to
/// even on the exact value, then back to the nearest double).
pub fn round3(x: f64) -> f64 {
    if !x.is_finite() {
        return x;
    }
    format!("{x:.3}").parse::<f64>().unwrap_or(x)
}
