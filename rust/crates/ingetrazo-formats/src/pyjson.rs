// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! Python's `json` module, byte for byte: what `json.loads` reads and what
//! `json.dumps` writes, so an .igz saved here is the file the Python app
//! would have saved.
//!
//! The parts that matter for identical bytes:
//! - floats are written as `repr(float)`: the shortest digits that read
//!   back to the same double, fixed notation for exponents -4..15 and
//!   `1e-05` / `1e+16` outside it, `.0` on a whole number;
//! - `ensure_ascii`: anything outside space..`~` becomes `\uXXXX` in lower
//!   case hex, astral characters as a surrogate pair;
//! - objects keep their key order (a Python dict's insertion order); a
//!   duplicate key read twice keeps its first place and its last value;
//! - integers keep every digit (Python's are unbounded).

use std::collections::HashMap;
use std::fmt;

#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    Null,
    Bool(bool),
    /// A Python int, as its decimal digits (`-0` reads as `0`).
    Int(String),
    Float(f64),
    Str(String),
    Array(Vec<Value>),
    Object(Vec<(String, Value)>),
}

impl Value {
    pub fn int(i: i64) -> Value {
        Value::Int(i.to_string())
    }

    /// The value under `key` of an object (the last one, as Python's dict).
    pub fn get(&self, key: &str) -> Option<&Value> {
        match self {
            Value::Object(items) => items.iter().rev().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }

    pub fn as_i64(&self) -> Option<i64> {
        match self {
            Value::Int(s) => s.parse().ok(),
            _ => None,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct JsonError {
    pub msg: String,
    /// Byte offset in the text (reading), 0 when writing.
    pub pos: usize,
}

impl fmt::Display for JsonError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} (at byte {})", self.msg, self.pos)
    }
}

impl std::error::Error for JsonError {}

fn err<T>(msg: &str, pos: usize) -> Result<T, JsonError> {
    Err(JsonError { msg: msg.to_string(), pos })
}

// ---- writing ---------------------------------------------------------------

/// `json.dumps`'s keyword arguments.
#[derive(Clone, Copy, Debug)]
pub struct Options {
    pub indent: Option<usize>,
    pub item_sep: &'static str,
    pub key_sep: &'static str,
    pub sort_keys: bool,
    pub ensure_ascii: bool,
    pub allow_nan: bool,
}

impl Default for Options {
    /// `json.dumps(obj)`.
    fn default() -> Self {
        Options { indent: None, item_sep: ", ", key_sep: ": ", sort_keys: false, ensure_ascii: true, allow_nan: true }
    }
}

impl Options {
    /// `json.dumps(obj, indent=n)`: with an indent Python drops the space
    /// after the comma.
    pub fn indent(n: usize) -> Self {
        Options { indent: Some(n), item_sep: ",", ..Options::default() }
    }

    /// `json.dumps(obj, separators=(",", ":"))`.
    pub fn compact() -> Self {
        Options { item_sep: ",", key_sep: ":", ..Options::default() }
    }

    pub fn allow_nan(self, allow: bool) -> Self {
        Options { allow_nan: allow, ..self }
    }

    pub fn sort_keys(self, sort: bool) -> Self {
        Options { sort_keys: sort, ..self }
    }
}

/// `json.dumps(value, **opts)`.
pub fn dumps(value: &Value, opts: &Options) -> Result<String, JsonError> {
    let mut out = String::new();
    encode(value, opts, 0, &mut out)?;
    Ok(out)
}

fn newline_indent(opts: &Options, level: usize, out: &mut String) {
    if let Some(n) = opts.indent {
        out.push('\n');
        for _ in 0..n * level {
            out.push(' ');
        }
    }
}

fn encode(value: &Value, opts: &Options, level: usize, out: &mut String) -> Result<(), JsonError> {
    match value {
        Value::Null => out.push_str("null"),
        Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Value::Int(s) => out.push_str(s),
        Value::Float(x) => out.push_str(&float_json(*x, opts.allow_nan)?),
        Value::Str(s) => encode_str(s, opts.ensure_ascii, out),
        Value::Array(items) => {
            if items.is_empty() {
                out.push_str("[]");
                return Ok(());
            }
            out.push('[');
            for (i, v) in items.iter().enumerate() {
                if i > 0 {
                    out.push_str(opts.item_sep);
                }
                newline_indent(opts, level + 1, out);
                encode(v, opts, level + 1, out)?;
            }
            newline_indent(opts, level, out);
            out.push(']');
        }
        Value::Object(items) => {
            if items.is_empty() {
                out.push_str("{}");
                return Ok(());
            }
            let mut order: Vec<&(String, Value)> = items.iter().collect();
            if opts.sort_keys {
                order.sort_by(|a, b| a.0.cmp(&b.0)); // stable, as Python's sorted()
            }
            out.push('{');
            for (i, (k, v)) in order.into_iter().enumerate() {
                if i > 0 {
                    out.push_str(opts.item_sep);
                }
                newline_indent(opts, level + 1, out);
                encode_str(k, opts.ensure_ascii, out);
                out.push_str(opts.key_sep);
                encode(v, opts, level + 1, out)?;
            }
            newline_indent(opts, level, out);
            out.push('}');
        }
    }
    Ok(())
}

fn float_json(x: f64, allow_nan: bool) -> Result<String, JsonError> {
    if x.is_finite() {
        return Ok(float_repr(x));
    }
    if !allow_nan {
        return err("Out of range float values are not JSON compliant", 0);
    }
    Ok(if x.is_nan() {
        "NaN"
    } else if x > 0.0 {
        "Infinity"
    } else {
        "-Infinity"
    }
    .to_string())
}

/// Python's `repr(x)` for a finite float.
pub fn float_repr(x: f64) -> String {
    if x == 0.0 {
        return if x.is_sign_negative() { "-0.0" } else { "0.0" }.to_string();
    }
    // Rust's `{:e}` gives the shortest digit COUNT that round-trips. Which
    // digits: Python (dtoa mode 0) takes, among the shortest, the one
    // nearest the exact value, ties to even — 802.39385986328125 (a float32)
    // is "802.3938598632812" there, while `{:e}` rounds the tie up. Exact
    // formatting at that many digits rounds half to even; it is the answer
    // unless, at a power of two, that decimal falls outside the rounding
    // interval — then the shortest one `{:e}` found is.
    let shortest = format!("{x:e}");
    let n = shortest.split_once('e').map_or(0, |(m, _)| m.bytes().filter(u8::is_ascii_digit).count());
    let nearest = format!("{:.*e}", n.saturating_sub(1), x);
    let s = if nearest.parse::<f64>() == Ok(x) { nearest } else { shortest };
    let (mant, exp) = s.split_once('e').expect("{:e} has an exponent");
    let exp: i32 = exp.parse().expect("{:e} exponent is an integer");
    let (neg, mant) = match mant.strip_prefix('-') {
        Some(m) => (true, m),
        None => (false, mant),
    };
    let digits: String = mant.chars().filter(|&c| c != '.').collect();
    let n = digits.len() as i32;
    let decpt = exp + 1; // digits[..decpt] are the integer part
    let mut out = String::with_capacity(digits.len() + 8);
    if neg {
        out.push('-');
    }
    if -4 < decpt && decpt <= 16 {
        if decpt <= 0 {
            out.push_str("0.");
            out.extend(std::iter::repeat_n('0', (-decpt) as usize));
            out.push_str(&digits);
        } else if decpt >= n {
            out.push_str(&digits);
            out.extend(std::iter::repeat_n('0', (decpt - n) as usize));
            out.push_str(".0");
        } else {
            out.push_str(&digits[..decpt as usize]);
            out.push('.');
            out.push_str(&digits[decpt as usize..]);
        }
    } else {
        out.push_str(&digits[..1]);
        if n > 1 {
            out.push('.');
            out.push_str(&digits[1..]);
        }
        out.push('e');
        out.push(if exp < 0 { '-' } else { '+' });
        let e = exp.unsigned_abs();
        if e < 10 {
            out.push('0');
        }
        out.push_str(&e.to_string());
    }
    out
}

fn encode_str(s: &str, ensure_ascii: bool, out: &mut String) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            ' '..='~' => out.push(c),
            // Without ensure_ascii Python escapes only the controls.
            _ if !ensure_ascii && c >= ' ' => out.push(c),
            _ => {
                let mut buf = [0u16; 2];
                for unit in c.encode_utf16(&mut buf) {
                    out.push_str(&format!("\\u{unit:04x}"));
                }
            }
        }
    }
    out.push('"');
}

// ---- reading ---------------------------------------------------------------

/// `json.loads(text)` (strict: no control characters inside strings).
/// `NaN`, `Infinity` and `-Infinity` read as floats, as in Python.
pub fn loads(text: &str) -> Result<Value, JsonError> {
    let mut p = Parser { s: text.as_bytes(), i: 0 };
    p.ws();
    let v = p.value()?;
    p.ws();
    if p.i != p.s.len() {
        return err("Extra data", p.i);
    }
    Ok(v)
}

struct Parser<'a> {
    s: &'a [u8],
    i: usize,
}

impl Parser<'_> {
    fn ws(&mut self) {
        while let Some(b' ' | b'\t' | b'\n' | b'\r') = self.s.get(self.i) {
            self.i += 1;
        }
    }

    fn eat(&mut self, word: &str) -> bool {
        if self.s[self.i..].starts_with(word.as_bytes()) {
            self.i += word.len();
            true
        } else {
            false
        }
    }

    fn value(&mut self) -> Result<Value, JsonError> {
        match self.s.get(self.i) {
            None => err("Expecting value", self.i),
            Some(b'{') => self.object(),
            Some(b'[') => self.array(),
            Some(b'"') => Ok(Value::Str(self.string()?)),
            Some(b'n') if self.eat("null") => Ok(Value::Null),
            Some(b't') if self.eat("true") => Ok(Value::Bool(true)),
            Some(b'f') if self.eat("false") => Ok(Value::Bool(false)),
            Some(b'N') if self.eat("NaN") => Ok(Value::Float(f64::NAN)),
            Some(b'I') if self.eat("Infinity") => Ok(Value::Float(f64::INFINITY)),
            Some(b'-') if self.eat("-Infinity") => Ok(Value::Float(f64::NEG_INFINITY)),
            Some(b'-' | b'0'..=b'9') => self.number(),
            Some(_) => err("Expecting value", self.i),
        }
    }

    fn digits(&mut self) -> usize {
        let start = self.i;
        while let Some(b'0'..=b'9') = self.s.get(self.i) {
            self.i += 1;
        }
        self.i - start
    }

    /// Python's NUMBER_RE: `(-?(?:0|[1-9]\d*))(\.\d+)?([eE][-+]?\d+)?`, where
    /// an optional part that does not match whole is simply not taken.
    fn number(&mut self) -> Result<Value, JsonError> {
        let start = self.i;
        if self.s[self.i] == b'-' {
            self.i += 1;
        }
        match self.s.get(self.i) {
            Some(b'0') => self.i += 1,
            Some(b'1'..=b'9') => {
                self.digits();
            }
            _ => return err("Expecting value", start),
        }
        let int_end = self.i;
        let mut is_float = false;
        if self.s.get(self.i) == Some(&b'.') {
            let save = self.i;
            self.i += 1;
            if self.digits() == 0 {
                self.i = save;
            } else {
                is_float = true;
            }
        }
        if let Some(b'e' | b'E') = self.s.get(self.i) {
            let save = self.i;
            self.i += 1;
            if let Some(b'+' | b'-') = self.s.get(self.i) {
                self.i += 1;
            }
            if self.digits() == 0 {
                self.i = save;
            } else {
                is_float = true;
            }
        }
        let text = std::str::from_utf8(&self.s[start..self.i]).expect("ASCII digits");
        if is_float {
            return Ok(Value::Float(text.parse().expect("a JSON number parses as f64")));
        }
        debug_assert_eq!(int_end, self.i);
        Ok(Value::Int(if text == "-0" { "0".to_string() } else { text.to_string() }))
    }

    fn hex4(&mut self) -> Result<u32, JsonError> {
        let h = self.s.get(self.i..self.i + 4).and_then(|h| std::str::from_utf8(h).ok());
        match h.and_then(|h| u32::from_str_radix(h, 16).ok().filter(|_| h.bytes().all(|b| b.is_ascii_hexdigit()))) {
            Some(u) => {
                self.i += 4;
                Ok(u)
            }
            None => err("Invalid \\uXXXX escape", self.i - 2),
        }
    }

    fn string(&mut self) -> Result<String, JsonError> {
        let start = self.i;
        self.i += 1; // the opening quote
        let mut out = String::new();
        loop {
            // The run up to the next quote, backslash or control character.
            let run = self.i;
            while let Some(&b) = self.s.get(self.i) {
                if b == b'"' || b == b'\\' || b < 0x20 {
                    break;
                }
                self.i += 1;
            }
            out.push_str(std::str::from_utf8(&self.s[run..self.i]).map_err(|_| JsonError {
                msg: "Invalid UTF-8".into(),
                pos: run,
            })?);
            match self.s.get(self.i) {
                None => return err("Unterminated string starting at", start),
                Some(b'"') => {
                    self.i += 1;
                    return Ok(out);
                }
                Some(b'\\') => {
                    self.i += 1;
                    let c = match self.s.get(self.i) {
                        Some(b'"') => '"',
                        Some(b'\\') => '\\',
                        Some(b'/') => '/',
                        Some(b'b') => '\u{8}',
                        Some(b'f') => '\u{c}',
                        Some(b'n') => '\n',
                        Some(b'r') => '\r',
                        Some(b't') => '\t',
                        Some(b'u') => {
                            self.i += 1;
                            let u = self.hex4()?;
                            let code = if (0xd800..0xdc00).contains(&u)
                                && self.s.get(self.i..self.i + 2) == Some(b"\\u")
                            {
                                let save = self.i;
                                self.i += 2;
                                let lo = self.hex4()?;
                                if (0xdc00..0xe000).contains(&lo) {
                                    0x10000 + ((u - 0xd800) << 10) + (lo - 0xdc00)
                                } else {
                                    self.i = save;
                                    u
                                }
                            } else {
                                u
                            };
                            // Python keeps a lone surrogate; a Rust string
                            // cannot hold one.
                            match char::from_u32(code) {
                                Some(c) => {
                                    out.push(c);
                                    continue;
                                }
                                None => return err("Lone surrogate in a string (not supported)", self.i - 6),
                            }
                        }
                        _ => return err("Invalid \\escape", self.i - 1),
                    };
                    out.push(c);
                    self.i += 1;
                }
                Some(_) => return err("Invalid control character at", self.i),
            }
        }
    }

    fn array(&mut self) -> Result<Value, JsonError> {
        self.i += 1;
        let mut items = Vec::new();
        self.ws();
        if self.s.get(self.i) == Some(&b']') {
            self.i += 1;
            return Ok(Value::Array(items));
        }
        loop {
            self.ws();
            items.push(self.value()?);
            self.ws();
            match self.s.get(self.i) {
                Some(b',') => self.i += 1,
                Some(b']') => {
                    self.i += 1;
                    return Ok(Value::Array(items));
                }
                _ => return err("Expecting ',' delimiter", self.i),
            }
        }
    }

    fn object(&mut self) -> Result<Value, JsonError> {
        self.i += 1;
        let mut items: Vec<(String, Value)> = Vec::new();
        let mut index: HashMap<String, usize> = HashMap::new();
        self.ws();
        if self.s.get(self.i) == Some(&b'}') {
            self.i += 1;
            return Ok(Value::Object(items));
        }
        loop {
            self.ws();
            if self.s.get(self.i) != Some(&b'"') {
                return err("Expecting property name enclosed in double quotes", self.i);
            }
            let key = self.string()?;
            self.ws();
            if self.s.get(self.i) != Some(&b':') {
                return err("Expecting ':' delimiter", self.i);
            }
            self.i += 1;
            self.ws();
            let v = self.value()?;
            // dict[key] = v: a repeated key keeps its first place.
            match index.get(&key) {
                Some(&at) => items[at].1 = v,
                None => {
                    index.insert(key.clone(), items.len());
                    items.push((key, v));
                }
            }
            self.ws();
            match self.s.get(self.i) {
                Some(b',') => self.i += 1,
                Some(b'}') => {
                    self.i += 1;
                    return Ok(Value::Object(items));
                }
                _ => return err("Expecting ',' delimiter", self.i),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[allow(clippy::excessive_precision)] // the exact value is the point
    fn repr_matches_python() {
        // Values and their repr() taken from CPython 3.12.
        let cases: &[(f64, &str)] = &[
            (1.0, "1.0"),
            (-0.0, "-0.0"),
            (0.1, "0.1"),
            (1e-4, "0.0001"),
            (1e-5, "1e-05"),
            (1.5e-5, "1.5e-05"),
            (123456789.0, "123456789.0"),
            (1e15, "1000000000000000.0"),
            (1e16, "1e+16"),
            (1.2345e16, "1.2345e+16"),
            (9999999999999998.0, "9999999999999998.0"),
            (1e22, "1e+22"),
            (5e-324, "5e-324"),
            (1.7976931348623157e308, "1.7976931348623157e+308"),
            (0.30000000000000004, "0.30000000000000004"),
            (-2.5, "-2.5"),
            (100.0, "100.0"),
            // A tie between two shortest candidates: Python takes the even one.
            (802.39385986328125, "802.3938598632812"),
            (0.000123, "0.000123"),
        ];
        for &(x, want) in cases {
            assert_eq!(float_repr(x), want, "{x:e}");
        }
    }

    #[test]
    fn dumps_like_python() {
        let v = loads(r#"{"a": [1, 2.5, {"b": []}], "c": {}, "d": "é😀\u001f\n", "e": null}"#).unwrap();
        assert_eq!(
            dumps(&v, &Options::default()).unwrap(),
            r#"{"a": [1, 2.5, {"b": []}], "c": {}, "d": "\u00e9\ud83d\ude00\u001f\n", "e": null}"#
        );
        assert_eq!(
            dumps(&v, &Options::compact()).unwrap(),
            r#"{"a":[1,2.5,{"b":[]}],"c":{},"d":"\u00e9\ud83d\ude00\u001f\n","e":null}"#
        );
        assert_eq!(
            dumps(&v, &Options::indent(2)).unwrap(),
            "{\n  \"a\": [\n    1,\n    2.5,\n    {\n      \"b\": []\n    }\n  ],\n  \"c\": {},\n  \"d\": \"\\u00e9\\ud83d\\ude00\\u001f\\n\",\n  \"e\": null\n}"
        );
    }

    #[test]
    fn duplicate_keys_keep_first_place_last_value() {
        let v = loads(r#"{"x": 1, "y": 2, "x": 3}"#).unwrap();
        assert_eq!(dumps(&v, &Options::compact()).unwrap(), r#"{"x":3,"y":2}"#);
    }

    #[test]
    fn numbers_read_as_python() {
        let v = loads("[-0, 12345678901234567890123, 1E2, 1.0e-3, -0.0]").unwrap();
        assert_eq!(dumps(&v, &Options::compact()).unwrap(), "[0,12345678901234567890123,100.0,0.001,-0.0]");
        assert!(loads("01").is_err());
        assert!(loads("1.").is_err());
    }

    #[test]
    fn nan_is_refused_unless_allowed() {
        let v = loads("[NaN, Infinity, -Infinity]").unwrap();
        assert!(dumps(&v, &Options::compact().allow_nan(false)).is_err());
        assert_eq!(dumps(&v, &Options::compact()).unwrap(), "[NaN,Infinity,-Infinity]");
    }

    #[test]
    fn strict_strings() {
        assert!(loads("\"a\u{1}b\"").is_err());
        assert!(loads(r#""\ud800""#).is_err());
        assert_eq!(loads(r#""\/""#).unwrap(), Value::Str("/".into()));
    }
}
