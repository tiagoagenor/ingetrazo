// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! The value box (VCB): what a typed buffer means.
//!
//! `Viewport._parse_value_buffer`, `_merge_mixed_numbers` and the length
//! reading of `core.units.parse` (`parse_length_field`), for a document in
//! metres - the unit every bare number is read in here (`bare_number_scale`
//! is 1). The metric fields (`2`, `1,5`, `30cm`, `1500mm`, `1m20`), the
//! imperial marks (`2"`, `5'`, `5'3"`, `3/4"`) and fractions are read;
//! a unit word of another kind (an area, an angle) is refused, as Python
//! refuses it. Not read: the Spanish and English long names beyond the ones
//! in [`length_unit`], and the `5-3 1/2` feet-inches hyphen form of an
//! imperial document.

use ingetrazo_math::V3;

const IN_M: f64 = 0.0254;
const FT_M: f64 = 0.3048;
const MAX_TYPED: f64 = 1e9;

/// What the Enter key hands the tool.
#[derive(Clone, Debug, PartialEq)]
pub enum VcbValue {
    /// One length (or count, or angle).
    Num(f64),
    /// Two fields: a rectangle's `width;height`, a rotated one's `width;angle`.
    Pair(f64, f64),
    /// Three fields: a line's `dx;dy;dz`.
    Triple(f64, f64, f64),
    /// `24s`.
    Segments(i64),
    /// `2r`.
    Radius(f64),
    /// A rise:run slope, an array (`3x`), a profile: the tools here do not
    /// take them.
    Other,
}

/// The tools' view of a typed value.
impl VcbValue {
    pub fn as_pair(&self) -> Option<(f64, f64)> {
        match self {
            VcbValue::Pair(a, b) => Some((*a, *b)),
            _ => None,
        }
    }
}

pub fn point_of(v: V3) -> (f32, f32, f32) {
    (v.0, v.1, v.2)
}

/// `core.units.plain_marks` for the characters a keyboard types.
pub fn plain_marks(text: &str) -> String {
    let mut out = String::new();
    for ch in text.chars() {
        let frac = match ch {
            '\u{bd}' => Some("1/2"),
            '\u{bc}' => Some("1/4"),
            '\u{be}' => Some("3/4"),
            '\u{215b}' => Some("1/8"),
            '\u{215c}' => Some("3/8"),
            '\u{215d}' => Some("5/8"),
            '\u{215e}' => Some("7/8"),
            _ => None,
        };
        if let Some(f) = frac {
            if out.chars().last().is_some_and(|c| c.is_ascii_digit()) {
                out.push(' ');
            }
            out.push_str(f);
            continue;
        }
        out.push(match ch {
            '\u{2033}' | '\u{201d}' | '\u{201c}' => '"',
            '\u{2032}' | '\u{2019}' | '\u{2018}' => '\'',
            c => c,
        });
    }
    out
}

/// `VULGAR_FRACTIONS` holds this character.
pub fn is_vulgar_fraction(ch: char) -> bool {
    matches!(
        ch,
        '\u{bd}' | '\u{2153}' | '\u{2154}' | '\u{bc}' | '\u{be}' | '\u{2155}' | '\u{2156}' | '\u{2157}' | '\u{2158}' | '\u{2159}' | '\u{215a}' | '\u{2150}' | '\u{215b}' | '\u{215c}' | '\u{215d}' | '\u{215e}' | '\u{2151}' | '\u{2152}'
    )
}

// ---- numbers --------------------------------------------------------------------------------

/// `core.units._number`: `2`, `2.5`, `.5`, `3/4`, `1-1/2`, `1 1/2`.
fn number(text: &str) -> Option<f64> {
    let t = text.trim();
    let b = t.as_bytes();
    let mut d0 = 0;
    while d0 < b.len() && b[d0].is_ascii_digit() {
        d0 += 1;
    }
    let mut v;
    let mut whole: Option<i64> = None;
    let mut j = d0;
    while d0 > 0 && j < b.len() && (b[j] == b' ' || b[j] == b'-') {
        j += 1;
    }
    if d0 > 0 && j > d0 {
        // `(\d+)[ -]+` then the number: only a fraction may follow it.
        let (fv, is_fraction) = fraction_or_decimal(&t[j..])?;
        if !is_fraction {
            return None; // `1-2` is two numbers, not one
        }
        whole = t[..d0].parse::<i64>().ok();
        v = fv;
    } else {
        v = fraction_or_decimal(t)?.0;
    }
    if let Some(w) = whole {
        v += w as f64;
    }
    if v.is_finite() && v.abs() <= MAX_TYPED {
        Some(v)
    } else {
        None
    }
}

/// `(\d+)/(\d+)` (true) or `\d+(\.\d*)?|\.\d+` (false), whole string.
fn fraction_or_decimal(s: &str) -> Option<(f64, bool)> {
    if let Some((n, d)) = s.split_once('/') {
        if !n.is_empty() && !d.is_empty() && n.bytes().all(|c| c.is_ascii_digit()) && d.bytes().all(|c| c.is_ascii_digit()) {
            let (n, d) = (n.parse::<i64>().ok()?, d.parse::<i64>().ok()?);
            if d == 0 {
                return None;
            }
            return Some((n as f64 / d as f64, true));
        }
        return None;
    }
    let b = s.as_bytes();
    if b.is_empty() {
        return None;
    }
    let mut j = 0;
    let int_digits = {
        while j < b.len() && b[j].is_ascii_digit() {
            j += 1;
        }
        j
    };
    if int_digits > 0 {
        if j < b.len() && b[j] == b'.' {
            j += 1;
            while j < b.len() && b[j].is_ascii_digit() {
                j += 1;
            }
        }
    } else {
        if b[0] != b'.' {
            return None;
        }
        j = 1;
        let start = j;
        while j < b.len() && b[j].is_ascii_digit() {
            j += 1;
        }
        if j == start {
            return None;
        }
    }
    if j != b.len() {
        return None;
    }
    // Python's float() reads "3." as 3.0.
    s.trim_end_matches('.').parse::<f64>().ok().map(|v| (v, false))
}

// ---- units ------------------------------------------------------------------------------------

/// The length rows of `_ALIASES`: word -> metres per unit.
fn length_unit(word: &str) -> Option<f64> {
    Some(match word {
        "mm" | "millimetre" | "millimetres" | "millimeter" | "millimeters" => 1e-3,
        "cm" | "centimetre" | "centimetres" | "centimeter" | "centimeters" => 1e-2,
        "m" | "metre" | "metres" | "meter" | "meters" | "metro" | "metros" => 1.0,
        "km" | "kilometre" | "kilometres" | "kilometer" | "kilometers" => 1e3,
        "\"" | "in" | "inch" | "inches" | "pulgada" | "pulgadas" => IN_M,
        "'" | "ft" | "foot" | "feet" | "pie" | "pies" => FT_M,
        "yd" | "yard" | "yards" | "yarda" | "yardas" => 0.9144,
        "mi" | "mile" | "miles" | "milla" | "millas" => 1609.344,
        "us-ft" | "usft" | "survey-ft" => 1200.0 / 3937.0,
        _ => return None,
    })
}

/// A word that names a unit of another kind (area, volume, angle): refused.
fn other_kind(word: &str) -> bool {
    matches!(
        word,
        "mm2" | "cm2" | "m2" | "ha" | "in2" | "ft2" | "yd2" | "acre" | "acres" | "ac" | "cm3" | "l" | "m3" | "in3" | "ft3" | "yd3" | "gal" | "\u{b0}" | "deg" | "degree" | "degrees" | "rad" | "radian" | "radians" | "grad" | "gon"
    )
}

/// `core.units.parse(field, "length")` in a metre document: the value in
/// metres, or `None`.
pub fn parse_length_field(field: &str) -> Option<f64> {
    let raw = plain_marks(field);
    let mut t: String = raw.trim().to_lowercase().replace(',', ".");
    // `re.sub(r"\s+", " ", ...)`
    let mut collapsed = String::new();
    let mut prev_space = false;
    for ch in t.chars() {
        if ch.is_whitespace() {
            if !prev_space {
                collapsed.push(' ');
            }
            prev_space = true;
        } else {
            collapsed.push(ch);
            prev_space = false;
        }
    }
    t = collapsed;
    if t.is_empty() {
        return None;
    }
    let mut sign = 1.0;
    if t.starts_with('+') || t.starts_with('-') {
        if t.starts_with('-') {
            sign = -1.0;
        }
        t = t[1..].trim_start().to_string();
    }
    if t.is_empty() {
        return None;
    }
    if let Some(v) = feet_inches(&t) {
        return v.map(|x| sign * x);
    }
    // The terms: `qty [word]` repeated.
    let chars: Vec<char> = t.chars().collect();
    let mut pos = 0usize;
    let mut terms: Vec<(String, Option<String>)> = Vec::new();
    while pos < chars.len() {
        let (qty_end, qty) = match_qty(&chars, pos)?;
        let mut p = qty_end;
        while p < chars.len() && chars[p] == ' ' {
            p += 1;
        }
        let (word, wend) = match_word(&chars, p);
        let mut q = wend;
        while q < chars.len() && chars[q] == ' ' {
            q += 1;
        }
        // `_TERM.match(t, pos)` always consumes the qty; an empty match fails.
        if q == pos {
            return None;
        }
        terms.push((qty, word));
        pos = q;
    }
    let mut total = 0.0f64;
    let mut previous: Option<&'static str> = None;
    let n_terms = terms.len();
    for (i, (num_txt, word)) in terms.iter().enumerate() {
        let num = number(num_txt)?;
        match word {
            None => {
                if n_terms == 1 {
                    return Some(sign * (num * 1.0));
                }
                let down = match previous {
                    Some("m") => Some(1e-2),
                    Some("ft") => Some(IN_M),
                    _ => None,
                };
                if i != n_terms - 1 {
                    return None;
                }
                let down = down?;
                total += num * down;
                break;
            }
            Some(w) => {
                if let Some(f) = length_unit(w) {
                    total += num * f;
                    previous = match w.as_str() {
                        "m" | "metre" | "metres" | "meter" | "meters" | "metro" | "metros" => Some("m"),
                        "'" | "ft" | "foot" | "feet" | "pie" | "pies" => Some("ft"),
                        _ => Some("other"),
                    };
                } else {
                    let _ = other_kind(w);
                    return None;
                }
            }
        }
    }
    Some(sign * total)
}

/// `_QTY` at `pos`: `\d+[ -]+\d+/\d+ | \d+/\d+ | \d+(\.\d*)? | \.\d+`.
fn match_qty(c: &[char], pos: usize) -> Option<(usize, String)> {
    let dig = |from: usize| {
        let mut j = from;
        while j < c.len() && c[j].is_ascii_digit() {
            j += 1;
        }
        j
    };
    let d0 = dig(pos);
    if d0 > pos {
        // `\d+[ -]+\d+/\d+`
        let mut j = d0;
        while j < c.len() && (c[j] == ' ' || c[j] == '-') {
            j += 1;
        }
        if j > d0 {
            let d1 = dig(j);
            if d1 > j && d1 < c.len() && c[d1] == '/' {
                let d2 = dig(d1 + 1);
                if d2 > d1 + 1 {
                    return Some((d2, c[pos..d2].iter().collect()));
                }
            }
        }
        // `\d+/\d+`
        if d0 < c.len() && c[d0] == '/' {
            let d2 = dig(d0 + 1);
            if d2 > d0 + 1 {
                return Some((d2, c[pos..d2].iter().collect()));
            }
        }
        // `\d+(\.\d*)?`
        let mut j = d0;
        if j < c.len() && c[j] == '.' {
            j += 1;
            j = dig(j);
        }
        return Some((j, c[pos..j].iter().collect()));
    }
    if pos < c.len() && c[pos] == '.' {
        let d = dig(pos + 1);
        if d > pos + 1 {
            return Some((d, c[pos..d].iter().collect()));
        }
    }
    None
}

/// `_WORD` at `pos`: letters (with an optional `^2`/`2`/`3`) or one mark.
fn match_word(c: &[char], pos: usize) -> (Option<String>, usize) {
    if pos >= c.len() {
        return (None, pos);
    }
    if matches!(c[pos], '\u{b0}' | '\'' | '"') {
        return (Some(c[pos].to_string()), pos + 1);
    }
    let mut j = pos;
    while j < c.len() && c[j].is_alphabetic() {
        j += 1;
    }
    if j == pos {
        return (None, pos);
    }
    // `(?:\^?[23](?![\d./]))?`
    let mut k = j;
    if k < c.len() && c[k] == '^' {
        k += 1;
    }
    if k < c.len() && (c[k] == '2' || c[k] == '3') {
        let after = k + 1;
        if after >= c.len() || !(c[after].is_ascii_digit() || c[after] == '.' || c[after] == '/') {
            return (Some(c[pos..after].iter().collect()), after);
        }
    }
    (Some(c[pos..j].iter().collect()), j)
}

/// `_FEET_INCHES.fullmatch(t)`: `Some(Some(m))` when it is a feet-inches
/// length, `Some(None)` when it matched but the numbers were refused, `None`
/// when it is not that form.
fn feet_inches(t: &str) -> Option<Option<f64>> {
    let c: Vec<char> = t.chars().collect();
    let (qend, ft_txt) = match_qty(&c, 0)?;
    let mut p = qend;
    while p < c.len() && c[p] == ' ' {
        p += 1;
    }
    let rest: String = c[p..].iter().collect();
    let mut matched = false;
    for w in ["feet", "foot", "pies", "pie", "ft", "'"] {
        if rest.starts_with(w) {
            p += w.chars().count();
            matched = true;
            break;
        }
    }
    if !matched {
        return None;
    }
    while p < c.len() && c[p] == ' ' {
        p += 1;
    }
    if p < c.len() && c[p] == '-' {
        p += 1;
        while p < c.len() && c[p] == ' ' {
            p += 1;
        }
    }
    let mut inch: Option<String> = None;
    if p < c.len() {
        let (iend, itxt) = match_qty(&c, p)?;
        let mut q = iend;
        while q < c.len() && c[q] == ' ' {
            q += 1;
        }
        let rest: String = c[q..].iter().collect();
        for w in ["pulgadas", "pulgada", "inches", "inch", "in", "\""] {
            if rest.starts_with(w) {
                q += w.chars().count();
                break;
            }
        }
        if q != c.len() {
            return None;
        }
        inch = Some(itxt);
    }
    let ft = number(&ft_txt);
    let inch_v = match &inch {
        Some(s) => number(s),
        None => Some(0.0),
    };
    Some(match (ft, inch_v) {
        (Some(f), Some(i)) => Some(f * FT_M + i * IN_M),
        _ => None,
    })
}

// ---- the buffer -------------------------------------------------------------------------------

/// `_merge_mixed_numbers`: `1 1/2"` is one mixed number; `5' 3"` one length.
fn merge_mixed_numbers(fields: Vec<String>) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let mut i = 0;
    while i < fields.len() {
        let f = &fields[i];
        let nxt = fields.get(i + 1);
        if let Some(n) = nxt {
            if is_whole_or_feet(f) && is_fraction_field(&n.to_lowercase()) {
                let joint = if f.ends_with('\'') || f.ends_with('-') { "" } else { "-" };
                out.push(format!("{f}{joint}{n}"));
                i += 2;
                continue;
            }
        }
        out.push(f.clone());
        i += 1;
    }
    let mut joined: Vec<String> = Vec::new();
    for f in out {
        let low = f.to_lowercase();
        if joined.last().is_some_and(|l| is_feet_only(l)) && is_inches_field(&low) {
            let last = joined.last_mut().unwrap();
            last.push_str(&f);
        } else {
            joined.push(f);
        }
    }
    joined
}

/// `-?\d+(?:'-?\d*)?`
fn is_whole_or_feet(s: &str) -> bool {
    let b = s.as_bytes();
    let mut i = 0;
    if i < b.len() && b[i] == b'-' {
        i += 1;
    }
    let d = i;
    while i < b.len() && b[i].is_ascii_digit() {
        i += 1;
    }
    if i == d {
        return false;
    }
    if i == b.len() {
        return true;
    }
    if b[i] != b'\'' {
        return false;
    }
    i += 1;
    if i < b.len() && b[i] == b'-' {
        i += 1;
    }
    while i < b.len() && b[i].is_ascii_digit() {
        i += 1;
    }
    i == b.len()
}

/// `\d+/\d+(?:mm|cm|m|in|ft|"|')?`
fn is_fraction_field(s: &str) -> bool {
    let b = s.as_bytes();
    let mut i = 0;
    while i < b.len() && b[i].is_ascii_digit() {
        i += 1;
    }
    if i == 0 || i >= b.len() || b[i] != b'/' {
        return false;
    }
    let j = i + 1;
    let mut k = j;
    while k < b.len() && b[k].is_ascii_digit() {
        k += 1;
    }
    if k == j {
        return false;
    }
    matches!(&s[k..], "" | "mm" | "cm" | "m" | "in" | "ft" | "\"" | "'")
}

/// `-?[\d./-]+'`
fn is_feet_only(s: &str) -> bool {
    let Some(body) = s.strip_suffix('\'') else { return false };
    let b = body.as_bytes();
    let mut i = 0;
    if i < b.len() && b[i] == b'-' {
        i += 1;
    }
    let rest = &body[i..];
    !rest.is_empty() && rest.bytes().all(|c| c.is_ascii_digit() || c == b'.' || c == b'/' || c == b'-')
}

/// `[\d./-]+(?:"|in)`
fn is_inches_field(s: &str) -> bool {
    let body = if let Some(b) = s.strip_suffix('"') {
        b
    } else if let Some(b) = s.strip_suffix("in") {
        b
    } else {
        return false;
    };
    !body.is_empty() && body.bytes().all(|c| c.is_ascii_digit() || c == b'.' || c == b'/' || c == b'-')
}

/// `Viewport._parse_value_buffer(buffer, comma_lists)`; `None` is a parse
/// error (Python's `None`).
pub fn parse_value_buffer(buffer: &str, comma_lists: bool) -> Option<VcbValue> {
    let buffer = plain_marks(buffer);
    // `re.sub(r",\s+", ";", buffer.strip())`
    let mut stripped_in = String::new();
    {
        let t = buffer.trim();
        let c: Vec<char> = t.chars().collect();
        let mut i = 0;
        while i < c.len() {
            if c[i] == ',' && i + 1 < c.len() && c[i + 1].is_whitespace() {
                stripped_in.push(';');
                i += 1;
                while i < c.len() && c[i].is_whitespace() {
                    i += 1;
                }
            } else {
                stripped_in.push(c[i]);
                i += 1;
            }
        }
    }
    let mut buffer = stripped_in;
    if comma_lists && !buffer.contains(';') && !buffer.trim().chars().any(|c| c.is_whitespace()) {
        buffer = buffer.replace(',', " ");
    }
    let normalized = buffer.replace(',', ".").replace(';', " ");
    let stripped = normalized.trim().to_string();
    let low = stripped.to_lowercase();
    // Arrays: `3x`, `3*`, `*3`, `5x10m`, `/3`, `3/`.
    if is_array_form(&low) {
        return Some(VcbValue::Other);
    }
    if low.ends_with('r') && !stripped.contains(':') {
        let body = &low[..low.len() - 1];
        return match fraction_or_decimal(body) {
            Some((v, false)) if !body.contains('/') => Some(VcbValue::Radius(v)),
            _ => None,
        };
    }
    if low.ends_with('p') && !stripped.contains(':') {
        return match fraction_or_decimal(&low[..low.len() - 1]) {
            Some((_, false)) => Some(VcbValue::Other),
            _ => None,
        };
    }
    if low.ends_with('s') && !stripped.contains(':') {
        let body = &low[..low.len() - 1];
        if !body.is_empty() && body.bytes().all(|c| c.is_ascii_digit()) {
            return body.parse::<i64>().ok().map(VcbValue::Segments);
        }
        return None;
    }
    if normalized.contains(':') {
        return ratio_form(&normalized).map(|_| VcbValue::Other);
    }
    let mut nums: Vec<f64> = Vec::new();
    for group in buffer.replace(',', ".").split(';') {
        let words: Vec<String> = group.split_whitespace().map(|s| s.to_string()).collect();
        for p in merge_mixed_numbers(words) {
            nums.push(parse_length_field(&p)?);
        }
    }
    match nums.len() {
        1 => Some(VcbValue::Num(nums[0])),
        2 => Some(VcbValue::Pair(nums[0], nums[1])),
        3 => Some(VcbValue::Triple(nums[0], nums[1], nums[2])),
        _ => None,
    }
}

/// `(\d+)\s*[x*]\s*(.+)`-style array forms (the tools here take no array).
fn is_array_form(s: &str) -> bool {
    let b: Vec<char> = s.chars().collect();
    let digits = |from: usize| {
        let mut j = from;
        while j < b.len() && b[j].is_ascii_digit() {
            j += 1;
        }
        j
    };
    let skip_ws = |mut j: usize| {
        while j < b.len() && b[j].is_whitespace() {
            j += 1;
        }
        j
    };
    // `(\d+)\s*[x*]` ... (alone, or followed by a step)
    let d = digits(0);
    if d > 0 {
        let j = skip_ws(d);
        if j < b.len() && (b[j] == 'x' || b[j] == '*') {
            return true;
        }
        // `(\d+)\s*/`
        if j < b.len() && b[j] == '/' && j + 1 == b.len() {
            return true;
        }
    }
    // `[x*]\s*(\d+)`
    if !b.is_empty() && (b[0] == 'x' || b[0] == '*') {
        let j = skip_ws(1);
        let e = digits(j);
        if e > j && e == b.len() {
            return true;
        }
    }
    // `/\s*(\d+)`
    if !b.is_empty() && b[0] == '/' {
        let j = skip_ws(1);
        let e = digits(j);
        if e > j && e == b.len() {
            return true;
        }
    }
    false
}

/// `\s*(-?(?:\d+\.?\d*|\.\d+)):(\d+\.?\d*|\.\d+)\s*`
fn ratio_form(s: &str) -> Option<f64> {
    let t = s.trim();
    let (a, b) = t.split_once(':')?;
    let rise = fraction_or_decimal(a.strip_prefix('-').unwrap_or(a)).filter(|v| !a.contains('/') && !v.1)?.0;
    let rise = if a.starts_with('-') { -rise } else { rise };
    let run = fraction_or_decimal(b).filter(|v| !v.1)?.0;
    if run <= 0.0 || !rise.is_finite() || !run.is_finite() {
        return None;
    }
    Some(rise.atan2(run).to_degrees())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plain_numbers_and_units() {
        assert_eq!(parse_length_field("2"), Some(2.0));
        assert_eq!(parse_length_field("2,5"), Some(2.5));
        assert_eq!(parse_length_field("30cm"), Some(30.0 * 1e-2));
        assert_eq!(parse_length_field("1500 mm"), Some(1500.0 * 1e-3));
        assert_eq!(parse_length_field("-4"), Some(-4.0));
        assert_eq!(parse_length_field("1m20"), Some(1.0 + 20.0 * 1e-2));
        assert_eq!(parse_length_field("abc"), None);
        assert_eq!(parse_length_field("5 m2"), None);
    }

    #[test]
    fn fields_and_forms() {
        assert_eq!(parse_value_buffer("3;4", false), Some(VcbValue::Pair(3.0, 4.0)));
        assert_eq!(parse_value_buffer("200,100", true), Some(VcbValue::Pair(200.0, 100.0)));
        assert_eq!(parse_value_buffer("2,5", false), Some(VcbValue::Num(2.5)));
        assert_eq!(parse_value_buffer("24s", false), Some(VcbValue::Segments(24)));
        assert_eq!(parse_value_buffer("2r", false), Some(VcbValue::Radius(2.0)));
        assert_eq!(parse_value_buffer("1;2;3", false), Some(VcbValue::Triple(1.0, 2.0, 3.0)));
        assert_eq!(parse_value_buffer("1:2", false), Some(VcbValue::Other));
    }

    #[test]
    fn imperial_marks() {
        let v = parse_length_field("5'3\"").unwrap();
        assert!((v - (5.0 * FT_M + 3.0 * IN_M)).abs() < 1e-12);
        assert_eq!(parse_length_field("3/4\""), Some(0.75 * IN_M));
    }
}
