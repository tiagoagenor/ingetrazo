// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
// Port of openskp (MIT, Copyright (c) 2024-2026 Ahsan Mehmood): see NOTICE-openskp.md.
//! The few Python string behaviours the reader leans on: `int()` and
//! `float()` of attribute text, `repr(float)`, and the two `utf-16-le`
//! decodes (`errors="ignore"` for the version string, `"replace"` for names).

/// Python whitespace as `str.strip()` removes it.
fn is_space(c: char) -> bool {
    c.is_whitespace() || matches!(c, '\u{1c}'..='\u{1f}')
}

/// Underscores allowed only between digits, as Python's numeric literals do.
fn strip_underscores(s: &str) -> Option<String> {
    if !s.contains('_') {
        return Some(s.to_string());
    }
    let b: Vec<char> = s.chars().collect();
    let mut out = String::new();
    for (i, &c) in b.iter().enumerate() {
        if c == '_' {
            let prev = i.checked_sub(1).and_then(|j| b.get(j)).is_some_and(|p| p.is_ascii_digit());
            let next = b.get(i + 1).is_some_and(|n| n.is_ascii_digit());
            if !(prev && next) {
                return None;
            }
        } else {
            out.push(c);
        }
    }
    Some(out)
}

/// `int(text)`.
pub fn py_int(text: &str) -> Option<i64> {
    let t = text.trim_matches(is_space);
    let (neg, digits) = match t.strip_prefix('-') {
        Some(d) => (true, d),
        None => (false, t.strip_prefix('+').unwrap_or(t)),
    };
    if digits.is_empty() || !digits.chars().all(|c| c.is_ascii_digit() || c == '_') {
        return None;
    }
    let clean = strip_underscores(digits)?;
    let n: i128 = clean.parse().unwrap_or(i128::MAX);
    let n = if neg { -n } else { n };
    Some(n.clamp(i64::MIN as i128, i64::MAX as i128) as i64)
}

/// `float(text)`.
pub fn py_float(text: &str) -> Option<f64> {
    let t = text.trim_matches(is_space);
    if t.is_empty() {
        return None;
    }
    let lower = t.to_ascii_lowercase();
    let body = lower.trim_start_matches(['+', '-']);
    if matches!(body, "inf" | "infinity" | "nan") {
        if lower.len() - body.len() > 1 {
            return None;
        }
        let neg = lower.starts_with('-');
        return Some(match body {
            "nan" => f64::NAN,
            _ => {
                if neg {
                    f64::NEG_INFINITY
                } else {
                    f64::INFINITY
                }
            }
        });
    }
    let clean = strip_underscores(&lower)?;
    if !clean.chars().all(|c| c.is_ascii_digit() || matches!(c, '+' | '-' | '.' | 'e')) {
        return None;
    }
    clean.parse::<f64>().ok()
}

/// `repr(float)`.
pub fn float_repr(x: f64) -> String {
    if x.is_nan() {
        return "nan".into();
    }
    if x.is_infinite() {
        return if x < 0.0 { "-inf".into() } else { "inf".into() };
    }
    if x == 0.0 {
        return if x.is_sign_negative() { "-0.0".into() } else { "0.0".into() };
    }
    let sci = format!("{:e}", x);
    let (mant, exp) = sci.split_once('e').unwrap_or((&sci, "0"));
    let exp: i32 = exp.parse().unwrap_or(0);
    let (sign, mant) = match mant.strip_prefix('-') {
        Some(m) => ("-", m),
        None => ("", mant),
    };
    let digits: String = mant.chars().filter(|c| *c != '.').collect();
    if (-4..16).contains(&exp) {
        let s = if exp >= 0 {
            let int_len = exp as usize + 1;
            if digits.len() <= int_len {
                format!("{}{}.0", digits, "0".repeat(int_len - digits.len()))
            } else {
                format!("{}.{}", &digits[..int_len], &digits[int_len..])
            }
        } else {
            format!("0.{}{}", "0".repeat((-exp - 1) as usize), digits)
        };
        format!("{sign}{s}")
    } else {
        let m = if digits.len() == 1 { digits.clone() } else { format!("{}.{}", &digits[..1], &digits[1..]) };
        format!("{sign}{m}e{}{:02}", if exp < 0 { '-' } else { '+' }, exp.abs())
    }
}

/// `bytes.decode("utf-16-le", errors="ignore")`: unpaired surrogates and a
/// trailing odd byte are dropped.
pub fn utf16le_ignore(b: &[u8]) -> String {
    let units: Vec<u16> = b.as_chunks::<2>().0.iter().map(|c| u16::from_le_bytes(*c)).collect();
    let mut out = String::new();
    let mut i = 0;
    while i < units.len() {
        let u = units[i];
        if (0xD800..0xDC00).contains(&u) {
            if let Some(&lo) = units.get(i + 1) {
                if (0xDC00..0xE000).contains(&lo) {
                    let c = 0x10000 + (((u as u32) - 0xD800) << 10) + ((lo as u32) - 0xDC00);
                    out.push(char::from_u32(c).unwrap_or('\u{fffd}'));
                    i += 2;
                    continue;
                }
            }
            i += 1;
        } else if (0xDC00..0xE000).contains(&u) {
            i += 1;
        } else {
            out.push(char::from_u32(u as u32).unwrap_or('\u{fffd}'));
            i += 1;
        }
    }
    out
}

/// `bytes.decode("utf-16-le", errors="replace")`.
pub fn utf16le_replace(b: &[u8]) -> String {
    let units: Vec<u16> = b.as_chunks::<2>().0.iter().map(|c| u16::from_le_bytes(*c)).collect();
    let mut s = String::from_utf16_lossy(&units);
    if b.len() % 2 == 1 {
        s.push('\u{fffd}');
    }
    s
}

/// `bytes.decode("utf-8", errors="replace")`.
pub fn utf8_replace(b: &[u8]) -> String {
    String::from_utf8_lossy(b).into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reprs_like_python() {
        assert_eq!(float_repr(1.0), "1.0");
        assert_eq!(float_repr(0.1), "0.1");
        assert_eq!(float_repr(-2.5), "-2.5");
        assert_eq!(float_repr(1e16), "1e+16");
        assert_eq!(float_repr(123456789012345680.0), "1.2345678901234568e+17");
        assert_eq!(float_repr(0.0001), "0.0001");
        assert_eq!(float_repr(0.00001), "1e-05");
        assert_eq!(float_repr(1234.5), "1234.5");
        assert_eq!(float_repr(100.0), "100.0");
    }

    #[test]
    fn parses_like_python() {
        assert_eq!(py_int(" 12 "), Some(12));
        assert_eq!(py_int("1_0"), Some(10));
        assert_eq!(py_int("1__0"), None);
        assert_eq!(py_int("1.5"), None);
        assert_eq!(py_float("1e3"), Some(1000.0));
        assert_eq!(py_float("0.3"), Some(0.3));
        assert!(py_float("x").is_none());
        assert!(py_float("nan").unwrap().is_nan());
    }

    #[test]
    fn utf16_decodes() {
        assert_eq!(utf16le_ignore(&[b'{', 0, b'1', 0, 0x00, 0xD8, b'}', 0]), "{1}");
        assert_eq!(utf16le_replace(&[b'a', 0, 0x00, 0xD8]), "a\u{fffd}");
    }
}
