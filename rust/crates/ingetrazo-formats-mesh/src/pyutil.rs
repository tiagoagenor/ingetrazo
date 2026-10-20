// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! Python's text built-ins the two formats lean on: `str.split()`,
//! `str.splitlines()`, `float(text)`, `round(x, 6)`, and the text decoding of
//! `formats.obj.read_obj_text`.

use ingetrazo_formats::pyjson::Value;
use ingetrazo_model::pyval;

/// Whether `str.split()` treats `c` as a separator (`str.isspace()`).
pub fn is_space(c: char) -> bool {
    c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c)
}

/// `text.split()`.
pub fn split(text: &str) -> Vec<&str> {
    text.split(is_space).filter(|p| !p.is_empty()).collect()
}

/// `round(x, 4)`.
pub fn round4(x: f64) -> f64 {
    if !x.is_finite() {
        return x;
    }
    format!("{x:.4}").parse().unwrap_or(x)
}

/// `round(x, 3)`.
pub fn round3(x: f64) -> f64 {
    if !x.is_finite() {
        return x;
    }
    format!("{x:.3}").parse().unwrap_or(x)
}

/// `text.splitlines()`: `\n`, `\r`, `\r\n`, `\v`, `\f`, FS/GS/RS, NEL, LS and
/// PS end a line; a final terminator does not start an empty one.
pub fn splitlines(text: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let mut start = 0;
    let mut chars = text.char_indices().peekable();
    while let Some((i, c)) = chars.next() {
        let is_break = matches!(c, '\n' | '\r' | '\u{b}' | '\u{c}' | '\u{1c}' | '\u{1d}' | '\u{1e}' | '\u{85}' | '\u{2028}' | '\u{2029}');
        if !is_break {
            continue;
        }
        out.push(&text[start..i]);
        let mut end = i + c.len_utf8();
        if c == '\r' {
            if let Some(&(_, '\n')) = chars.peek() {
                chars.next();
                end += 1;
            }
        }
        start = end;
    }
    if start < text.len() {
        out.push(&text[start..]);
    }
    out
}

/// `float(text)`; `None` where Python raises `ValueError`.
pub fn float(text: &str) -> Option<f64> {
    pyval::float(&Value::Str(text.to_string())).ok()
}

/// `repr(text)` for the messages that quote user text: single quotes unless
/// the text holds one and no double quote.
pub fn repr(text: &str) -> String {
    let quote = if text.contains('\'') && !text.contains('"') { '"' } else { '\'' };
    let mut out = String::new();
    out.push(quote);
    for c in text.chars() {
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

/// `round(x, 6)` of a float: the double nearest to `x` rounded to six
/// decimals, ties to even on the exact binary value (Python's dtoa mode 3).
pub fn round6(x: f64) -> f64 {
    if !x.is_finite() {
        return x;
    }
    format!("{x:.6}").parse().unwrap_or(x)
}

/// `Path(name).with_suffix(".mtl").name` for a file name.
pub fn with_suffix(name: &str, suffix: &str) -> String {
    match name.rfind('.') {
        Some(i) if i > 0 && i + 1 < name.len() => format!("{}{}", &name[..i], suffix),
        _ => format!("{name}{suffix}"),
    }
}

/// `read_obj_text`: UTF-8 (BOM allowed), else cp1252 with `errors="replace"`.
pub fn decode_text(data: &[u8]) -> String {
    match std::str::from_utf8(data) {
        Ok(t) => t.strip_prefix('\u{feff}').unwrap_or(t).to_string(),
        Err(_) => data.iter().map(|&b| cp1252(b)).collect(),
    }
}

fn cp1252(b: u8) -> char {
    const HIGH: [u16; 32] = [
        0x20ac, 0xfffd, 0x201a, 0x0192, 0x201e, 0x2026, 0x2020, 0x2021, 0x02c6, 0x2030, 0x0160, 0x2039, 0x0152, 0xfffd, 0x017d, 0xfffd,
        0xfffd, 0x2018, 0x2019, 0x201c, 0x201d, 0x2022, 0x2013, 0x2014, 0x02dc, 0x2122, 0x0161, 0x203a, 0x0153, 0xfffd, 0x017e, 0x0178,
    ];
    match b {
        0x80..=0x9f => char::from_u32(HIGH[(b - 0x80) as usize] as u32).unwrap_or('\u{fffd}'),
        _ => b as char,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splits_like_python() {
        assert_eq!(split("  a\tb \u{a0}c\u{1f}d "), vec!["a", "b", "c", "d"]);
        assert_eq!(splitlines("a\r\nb\rc\n\nd"), vec!["a", "b", "c", "", "d"]);
        assert_eq!(splitlines("a\n"), vec!["a"]);
        assert!(splitlines("").is_empty());
    }

    #[test]
    fn rounds_ties_to_even() {
        assert_eq!(round6(0.0078125), 0.007812);
        assert_eq!(round6(0.0234375), 0.023438);
        assert_eq!(round6(-0.0000004), 0.0);
    }

    #[test]
    fn suffix_replacement() {
        assert_eq!(with_suffix("a.b.obj", ".mtl"), "a.b.mtl");
        assert_eq!(with_suffix("model", ".mtl"), "model.mtl");
        assert_eq!(with_suffix(".obj", ".mtl"), ".obj.mtl");
    }
}
