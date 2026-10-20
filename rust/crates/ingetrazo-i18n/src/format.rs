// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! Python's `str.format(**kwargs)` for the field kinds the translations use:
//! `{name}`, `{name:spec}` and the `{{` / `}}` escapes, with the standard
//! format-spec mini-language for `d`, `f`/`F`, `s` and the empty type.
//!
//! Where Python raises, this returns the same class of error
//! ([`FormatError::Key`], [`Index`](FormatError::Index),
//! [`Value`](FormatError::Value)). Constructs no string in `i18n/*.json` or
//! any `tr()` call uses (attribute/index access, `!r` conversions, nested
//! replacement fields, `e`/`g`/`x`/... types) are [`FormatError::Unsupported`]
//! instead of being silently approximated; the checker test pins that none
//! occurs.

use ingetrazo_formats::pyjson::float_repr;

/// A keyword argument value, as Python would hold it.
#[derive(Clone, Debug, PartialEq)]
pub enum Arg {
    Int(i64),
    Float(f64),
    Str(String),
}

impl From<i64> for Arg {
    fn from(v: i64) -> Self {
        Arg::Int(v)
    }
}
impl From<i32> for Arg {
    fn from(v: i32) -> Self {
        Arg::Int(v.into())
    }
}
impl From<u32> for Arg {
    fn from(v: u32) -> Self {
        Arg::Int(v.into())
    }
}
impl From<usize> for Arg {
    fn from(v: usize) -> Self {
        Arg::Int(v as i64)
    }
}
impl From<f64> for Arg {
    fn from(v: f64) -> Self {
        Arg::Float(v)
    }
}
impl From<f32> for Arg {
    fn from(v: f32) -> Self {
        Arg::Float(v.into())
    }
}
impl From<&str> for Arg {
    fn from(v: &str) -> Self {
        Arg::Str(v.to_string())
    }
}
impl From<String> for Arg {
    fn from(v: String) -> Self {
        Arg::Str(v)
    }
}

/// What `str.format` would have raised.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FormatError {
    /// `KeyError`: a field name with no keyword argument.
    Key(String),
    /// `IndexError`: a positional field (`{}`, `{0}`) with no positional args.
    Index,
    /// `ValueError`: bad braces or a spec the value's type does not accept.
    Value(String),
    /// Valid Python this port does not implement.
    Unsupported(String),
}

impl FormatError {
    /// The Python exception class name.
    pub fn python_class(&self) -> &'static str {
        match self {
            FormatError::Key(_) => "KeyError",
            FormatError::Index => "IndexError",
            FormatError::Value(_) => "ValueError",
            FormatError::Unsupported(_) => "NotImplementedError",
        }
    }
}

impl std::fmt::Display for FormatError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FormatError::Key(k) => write!(f, "KeyError: {k:?}"),
            FormatError::Index => write!(f, "IndexError: replacement index out of range"),
            FormatError::Value(m) => write!(f, "ValueError: {m}"),
            FormatError::Unsupported(m) => write!(f, "unsupported format construct: {m}"),
        }
    }
}

impl std::error::Error for FormatError {}

fn value_err<T>(m: &str) -> Result<T, FormatError> {
    Err(FormatError::Value(m.to_string()))
}
fn unsupported<T>(m: &str) -> Result<T, FormatError> {
    Err(FormatError::Unsupported(m.to_string()))
}

/// `template.format(**args)`.
pub fn format(template: &str, args: &[(&str, Arg)]) -> Result<String, FormatError> {
    let mut out = String::with_capacity(template.len());
    walk(template, &mut out, &mut |field, out| render_field(field, args, out))?;
    Ok(out)
}

/// The `(name, spec)` of every replacement field, in order, after the same
/// syntax checks `format` makes (so a template that `format` rejects for its
/// braces is rejected here too). Used by the checker test.
pub fn fields(template: &str) -> Result<Vec<(String, String)>, FormatError> {
    let mut found = Vec::new();
    let mut sink = String::new();
    walk(template, &mut sink, &mut |field, _| {
        let (name, rest) = match field.find([':', '!']) {
            Some(p) => (&field[..p], &field[p..]),
            None => (field, ""),
        };
        found.push((name.to_string(), rest.strip_prefix(':').unwrap_or(rest).to_string()));
        Ok(())
    })?;
    Ok(found)
}

/// Splits a template into literal text (pushed to `out`, escapes resolved)
/// and replacement fields (handed to `on_field`), failing like Python's
/// parser on stray or unterminated braces.
fn walk(
    template: &str,
    out: &mut String,
    on_field: &mut dyn FnMut(&str, &mut String) -> Result<(), FormatError>,
) -> Result<(), FormatError> {
    let chars: Vec<char> = template.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        if c == '{' {
            if chars.get(i + 1) == Some(&'{') {
                out.push('{');
                i += 2;
                continue;
            }
            if i + 1 >= chars.len() {
                return value_err("Single '{' encountered in format string");
            }
            // A replacement field: up to its matching '}', nested braces
            // (only legal in the spec) counted.
            let start = i + 1;
            let mut depth = 1;
            let mut j = start;
            while j < chars.len() {
                match chars[j] {
                    '{' => depth += 1,
                    '}' => {
                        depth -= 1;
                        if depth == 0 {
                            break;
                        }
                    }
                    _ => {}
                }
                j += 1;
            }
            if depth != 0 {
                return value_err("expected '}' before end of string");
            }
            let field: String = chars[start..j].iter().collect();
            on_field(&field, out)?;
            i = j + 1;
        } else if c == '}' {
            if chars.get(i + 1) == Some(&'}') {
                out.push('}');
                i += 2;
            } else {
                return value_err("Single '}' encountered in format string");
            }
        } else {
            out.push(c);
            i += 1;
        }
    }
    Ok(())
}

fn render_field(field: &str, args: &[(&str, Arg)], out: &mut String) -> Result<(), FormatError> {
    let (name, rest) = match field.find([':', '!']) {
        Some(p) => (&field[..p], &field[p..]),
        None => (field, ""),
    };
    if rest.starts_with('!') {
        return unsupported("conversion (!r, !s, !a)");
    }
    let spec = rest.strip_prefix(':').unwrap_or("");
    if spec.contains('{') || spec.contains('}') {
        return unsupported("nested replacement field in a format spec");
    }
    if name.contains(['.', '[']) {
        return unsupported("attribute or index access in a field name");
    }
    if name.is_empty() || name.chars().all(|c| c.is_ascii_digit()) {
        return Err(FormatError::Index);
    }
    // Python looks the name up in kwargs; the last duplicate wins like dict.
    let arg = args
        .iter()
        .rev()
        .find(|(k, _)| *k == name)
        .map(|(_, v)| v)
        .ok_or_else(|| FormatError::Key(name.to_string()))?;
    out.push_str(&format_value(arg, spec)?);
    Ok(())
}

#[derive(Default)]
struct Spec {
    fill: Option<char>,
    align: Option<char>,
    sign: Option<char>,
    zero: bool,
    width: usize,
    group: Option<char>,
    prec: Option<usize>,
    ty: Option<char>,
}

fn parse_spec(spec: &str) -> Result<Spec, FormatError> {
    let c: Vec<char> = spec.chars().collect();
    let mut s = Spec::default();
    let mut p = 0;
    let is_align = |ch: char| matches!(ch, '<' | '>' | '=' | '^');
    if c.len() >= 2 && is_align(c[1]) {
        s.fill = Some(c[0]);
        s.align = Some(c[1]);
        p = 2;
    } else if !c.is_empty() && is_align(c[0]) {
        s.align = Some(c[0]);
        p = 1;
    }
    if p < c.len() && matches!(c[p], '+' | '-' | ' ') {
        s.sign = Some(c[p]);
        p += 1;
    }
    if p < c.len() && matches!(c[p], 'z' | '#') {
        return unsupported("the 'z' and '#' spec options");
    }
    if s.fill.is_none() && p < c.len() && c[p] == '0' {
        s.zero = true;
        p += 1;
    }
    let digits = |p: &mut usize| -> Option<usize> {
        let st = *p;
        while *p < c.len() && c[*p].is_ascii_digit() {
            *p += 1;
        }
        if *p == st {
            None
        } else {
            c[st..*p].iter().collect::<String>().parse().ok()
        }
    };
    if let Some(w) = digits(&mut p) {
        s.width = w;
    }
    if p < c.len() && matches!(c[p], ',' | '_') {
        s.group = Some(c[p]);
        p += 1;
    }
    if p < c.len() && c[p] == '.' {
        p += 1;
        match digits(&mut p) {
            Some(n) => s.prec = Some(n),
            None => return value_err("Format specifier missing precision"),
        }
    }
    if p < c.len() {
        s.ty = Some(c[p]);
        p += 1;
    }
    if p < c.len() {
        return value_err("Invalid format specifier");
    }
    Ok(s)
}

/// Pads `body` (a sign already split off for numbers) to the spec's width.
fn pad(sign: &str, body: &str, s: &Spec, numeric: bool) -> String {
    let len = sign.chars().count() + body.chars().count();
    if len >= s.width {
        return format!("{sign}{body}");
    }
    let (mut fill, mut align) = (s.fill.unwrap_or(' '), s.align);
    if s.zero {
        fill = '0';
        if align.is_none() && numeric {
            align = Some('=');
        }
    }
    let align = align.unwrap_or(if numeric { '>' } else { '<' });
    let total = s.width - len;
    let f = |n: usize| fill.to_string().repeat(n);
    match align {
        '<' => format!("{sign}{body}{}", f(total)),
        '^' => format!("{}{sign}{body}{}", f(total / 2), f(total - total / 2)),
        '=' => format!("{sign}{}{body}", f(total)),
        _ => format!("{}{sign}{body}", f(total)),
    }
}

fn group_digits(int_digits: &str, sep: char) -> String {
    let n = int_digits.len();
    let mut out = String::with_capacity(n + n / 3);
    for (i, ch) in int_digits.chars().enumerate() {
        if i > 0 && (n - i).is_multiple_of(3) {
            out.push(sep);
        }
        out.push(ch);
    }
    out
}

fn sign_str(negative: bool, s: &Spec) -> &'static str {
    if negative {
        "-"
    } else {
        match s.sign {
            Some('+') => "+",
            Some(' ') => " ",
            _ => "",
        }
    }
}

fn format_value(arg: &Arg, spec: &str) -> Result<String, FormatError> {
    if spec.is_empty() {
        return Ok(match arg {
            Arg::Int(i) => i.to_string(),
            Arg::Float(f) => float_repr(*f),
            Arg::Str(s) => s.clone(),
        });
    }
    let s = parse_spec(spec)?;
    match arg {
        Arg::Str(v) => format_str(v, &s),
        Arg::Int(i) => match s.ty {
            None | Some('d') => format_int(*i, &s),
            Some('f') | Some('F') => format_float(*i as f64, &s),
            Some('s') => value_err("Unknown format code 's' for object of type 'int'"),
            Some(_) => unsupported("integer presentation type other than d, f, F"),
        },
        Arg::Float(f) => match s.ty {
            Some('f') | Some('F') => format_float(*f, &s),
            None => unsupported("float formatting without a presentation type"),
            Some(c) if c.is_ascii_alphabetic() && !matches!(c, 'e' | 'E' | 'g' | 'G' | 'n') => {
                Err(FormatError::Value(format!(
                    "Unknown format code '{c}' for object of type 'float'"
                )))
            }
            Some(_) => unsupported("float presentation type other than f, F"),
        },
    }
}

fn format_str(v: &str, s: &Spec) -> Result<String, FormatError> {
    match s.ty {
        None | Some('s') => {}
        Some(c) => {
            return Err(FormatError::Value(format!(
                "Unknown format code '{c}' for object of type 'str'"
            )))
        }
    }
    if s.sign.is_some() {
        return value_err("Sign not allowed in string format specifier");
    }
    if s.align == Some('=') {
        return value_err("'=' alignment not allowed in string format specifier");
    }
    if s.group.is_some() {
        return value_err("Cannot specify ',' with 's'.");
    }
    let body: String = match s.prec {
        Some(n) => v.chars().take(n).collect(),
        None => v.to_string(),
    };
    Ok(pad("", &body, s, false))
}

fn format_int(i: i64, s: &Spec) -> Result<String, FormatError> {
    if s.prec.is_some() {
        return value_err("Precision not allowed in integer format specifier");
    }
    let sign = sign_str(i < 0, s);
    let mut digits = i.unsigned_abs().to_string();
    if let Some(sep) = s.group {
        if s.zero || s.align == Some('=') {
            return unsupported("digit grouping combined with zero padding");
        }
        digits = group_digits(&digits, sep);
    }
    Ok(pad(sign, &digits, s, true))
}

fn format_float(f: f64, s: &Spec) -> Result<String, FormatError> {
    let upper = s.ty == Some('F');
    let sign = sign_str(f.is_sign_negative() && !f.is_nan(), s);
    if !f.is_finite() {
        if s.zero {
            return unsupported("zero padding of nan/inf");
        }
        let word = match (f.is_nan(), upper) {
            (true, false) => "nan",
            (true, true) => "NAN",
            (false, false) => "inf",
            (false, true) => "INF",
        };
        return Ok(pad(sign, word, s, true));
    }
    let mut body = format!("{:.*}", s.prec.unwrap_or(6), f.abs());
    if let Some(sep) = s.group {
        if s.zero || s.align == Some('=') {
            return unsupported("digit grouping combined with zero padding");
        }
        let (int_part, frac) = match body.find('.') {
            Some(p) => (body[..p].to_string(), body[p..].to_string()),
            None => (body.clone(), String::new()),
        };
        body = format!("{}{frac}", group_digits(&int_part, sep));
    }
    Ok(pad(sign, &body, s, true))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f(t: &str, a: &[(&str, Arg)]) -> Result<String, FormatError> {
        format(t, a)
    }

    #[test]
    fn basics() {
        assert_eq!(f("a {{x}} {n:02d}", &[("n", 5.into())]).unwrap(), "a {x} 05");
        assert_eq!(f("{x:+.2f}|{x:,.0f}", &[("x", 1234.5.into())]).unwrap(), "+1234.50|1,234");
        assert_eq!(f("{a}", &[]), Err(FormatError::Key("a".into())));
        assert_eq!(f("{}", &[]), Err(FormatError::Index));
        assert!(matches!(f("{", &[]), Err(FormatError::Value(_))));
        assert!(matches!(f("}", &[]), Err(FormatError::Value(_))));
        assert!(matches!(f("{n:d}", &[("n", 1.5.into())]), Err(FormatError::Value(_))));
        assert_eq!(f("{n:.0f}", &[("n", (-0.4).into())]).unwrap(), "-0");
        assert_eq!(f("{n:.0f}", &[("n", 2.5.into())]).unwrap(), "2");
    }
}
