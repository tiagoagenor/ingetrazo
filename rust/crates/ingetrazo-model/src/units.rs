// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! How a length reads (`core/units.py`): `format_length`, `fmt_len` and the
//! document settings they read. Only what the annotations need — a
//! dimension's text — is ported; the rest of the units table (areas,
//! volumes, angles, parsing) belongs to the tools that use it.
//!
//! `_fixed` is `Decimal(f"{x:.12g}")` rounded half up, which is done here on
//! the twelve significant digits `{:.11e}` gives (the same correctly rounded
//! digits); a value of 10²⁸ or more, where Python's decimal context would
//! raise, is not reproduced.

use ingetrazo_formats::pyjson::Value;

use crate::pyconv::py_int;

const IN_M: f64 = 0.0254;
const FT_M: f64 = 0.3048;

/// The length units the model may be set to (`_BARE_SCALE`'s keys).
const MODEL_LENGTHS: [&str; 8] = ["m", "cm", "mm", "in", "ft", "ft-in", "in-frac", "ft-in-frac"];

/// The units a document reads with: `core.units.model_units_of`, for the
/// three settings a length text uses.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModelUnits {
    pub length: String,
    pub precision: i64,
    /// `"."` or `","`.
    pub decimal: String,
}

impl Default for ModelUnits {
    /// `DEFAULT_MODEL_UNITS`: metres, two decimals, a point.
    fn default() -> Self {
        ModelUnits { length: "m".into(), precision: 2, decimal: ".".into() }
    }
}

impl ModelUnits {
    /// `model_units_of(scene)` for a scene's `units` (or a payload's):
    /// anything invalid is the default.
    pub fn of(units: Option<&Value>) -> ModelUnits {
        let Some(u @ Value::Object(_)) = units else { return ModelUnits::default() };
        let Some(Value::Str(length)) = u.get("length") else { return ModelUnits::default() };
        if !MODEL_LENGTHS.contains(&length.as_str()) {
            return ModelUnits::default();
        }
        // _precision(): int(value) clamped to 0..6, 2 when it is not a number.
        let precision = match u.get("precision") {
            Some(v) => match py_int(v) {
                Ok(n) => n.clamp(0, 6),
                Err(_) => 2,
            },
            None => 2,
        };
        let decimal = match u.get("decimal") {
            Some(Value::Str(d)) if d == "," => ",".to_string(),
            _ => ".".to_string(),
        };
        ModelUnits { length: length.clone(), precision, decimal }
    }
}

/// `core.units.fmt_len` with the document's settings given.
pub fn fmt_len(metres: f64, units: &ModelUnits) -> String {
    localized(&format_length(metres, &units.length, units.precision), &units.decimal)
}

/// `localized`: the document's decimal separator.
pub fn localized(text: &str, decimal: &str) -> String {
    if decimal == "," {
        text.replace('.', ",")
    } else {
        text.to_string()
    }
}

/// `1.2345` to `n` decimals "the way a person rounds it": half up on the
/// number as written to twelve significant digits, and never `-0.00`.
pub fn fixed(value: f64, n: usize) -> String {
    if !value.is_finite() {
        return if value.is_nan() { "nan".into() } else if value > 0.0 { "inf".into() } else { "-inf".into() };
    }
    let neg = value.is_sign_negative();
    let s = format!("{:.11e}", value.abs());
    let (mant, exp) = s.split_once('e').expect("exponent form");
    let exp: i32 = exp.parse().expect("exponent");
    let digits: Vec<u8> = mant.bytes().filter(|b| *b != b'.').collect(); // 12 digits
    let point = exp + 1; // digits before the decimal point
    let (mut int, mut frac): (Vec<u8>, Vec<u8>) = if point <= 0 {
        let mut f = vec![b'0'; (-point) as usize];
        f.extend_from_slice(&digits);
        (vec![b'0'], f)
    } else if point >= digits.len() as i32 {
        let mut i = digits.clone();
        i.extend(std::iter::repeat_n(b'0', point as usize - digits.len()));
        (i, Vec::new())
    } else {
        (digits[..point as usize].to_vec(), digits[point as usize..].to_vec())
    };
    if frac.len() > n {
        let up = frac[n] >= b'5'; // ROUND_HALF_UP: the discarded part is 0.5 or more
        frac.truncate(n);
        if up {
            let int_len = int.len();
            let mut all = int;
            all.extend_from_slice(&frac);
            let mut i = all.len();
            loop {
                if i == 0 {
                    all.insert(0, b'1');
                    break;
                }
                i -= 1;
                if all[i] == b'9' {
                    all[i] = b'0';
                } else {
                    all[i] += 1;
                    break;
                }
            }
            let grown = all.len() - (int_len + n);
            int = all[..int_len + grown].to_vec();
            frac = all[int_len + grown..].to_vec();
        }
    } else {
        frac.resize(n, b'0');
    }
    let zero = int.iter().chain(frac.iter()).all(|d| *d == b'0');
    let mut out = String::new();
    if neg && !zero {
        out.push('-');
    }
    out.push_str(std::str::from_utf8(&int).unwrap());
    if n > 0 {
        out.push('.');
        out.push_str(std::str::from_utf8(&frac).unwrap());
    }
    out
}

fn gcd(a: i128, b: i128) -> i128 {
    if b == 0 {
        a
    } else {
        gcd(b, a % b)
    }
}

/// `6.5` → `6 1/2`; `0.75` → `3/4`; `6.0` → `6`.
fn frac_inches(inches: f64, decimals: i64) -> String {
    let denom: i128 = match decimals.clamp(0, 4) {
        0 => 1,
        1 => 4,
        2 => 16,
        3 => 32,
        _ => 64,
    };
    let num = (inches * denom as f64).round_ties_even() as i128; // Python's round()
    let g = gcd(num, denom);
    let (p, q) = (num / g, denom / g);
    let (whole, rest) = (p / q, p % q);
    if rest == 0 {
        whole.to_string()
    } else if whole == 0 {
        format!("{rest}/{q}")
    } else {
        format!("{whole} {rest}/{q}")
    }
}

/// `x // y` of two floats, CPython's `float_floor_div`.
fn floor_div(vx: f64, wx: f64) -> f64 {
    let m = vx % wx;
    let mut div = (vx - m) / wx;
    if m != 0.0 && (wx < 0.0) != (m < 0.0) {
        div -= 1.0; // CPython also moves the remainder, which is not used here
    }
    if div != 0.0 {
        let mut fd = div.floor();
        if div - fd > 0.5 {
            fd += 1.0;
        }
        fd
    } else {
        0.0f64.copysign(vx / wx)
    }
}

/// `format_length`: a length in metres as text in the named unit. An empty
/// or unknown unit reads as metres.
pub fn format_length(metres: f64, units: &str, decimals: i64) -> String {
    let n = decimals.clamp(0, 6) as usize;
    let u = if units.is_empty() { "m" } else { units };
    if !metres.is_finite() {
        return "\u{2014}".to_string(); // never «nan m», never a crash
    }
    match u {
        "in" => return format!("{}\"", fixed(metres / IN_M, n)),
        "ft" => return format!("{}'", fixed(metres / FT_M, n)),
        "ft-in" | "ft-in-frac" | "in-frac" => {
            let sign = if metres < 0.0 { "-" } else { "" };
            let total_in = metres.abs() / IN_M;
            if u == "in-frac" {
                let txt = frac_inches(total_in, n as i64);
                return format!("{}{}\"", if txt != "0" { sign } else { "" }, txt);
            }
            let mut feet = floor_div(total_in, 12.0);
            let mut inches = total_in - feet * 12.0;
            if u == "ft-in" {
                // round(inches, n) >= 12: Python rounds the exact value, as {:.n} does
                let rounded: f64 = format!("{:.*}", n, inches).parse().unwrap();
                if rounded >= 12.0 {
                    feet += 1.0;
                    inches = 0.0;
                }
                let txt = format!("{:.*}", n, inches);
                let zero = feet == 0.0 && txt.parse::<f64>().unwrap() == 0.0;
                return format!("{}{}'{}\"", if zero { "" } else { sign }, feet, txt);
            }
            let mut txt = frac_inches(inches, n as i64);
            if txt == "12" {
                feet += 1.0;
                txt = "0".to_string();
            }
            let zero = feet == 0.0 && txt == "0";
            return format!("{}{}'{}\"", if zero { "" } else { sign }, feet, txt);
        }
        _ => {}
    }
    let row = match u {
        "mm" => Some((1e-3, "mm")),
        "cm" => Some((1e-2, "cm")),
        "m" => Some((1.0, "m")),
        "km" => Some((1e3, "km")),
        "yd" => Some((0.9144, "yd")),
        "mi" => Some((1609.344, "mi")),
        "us-ft" => Some((1200.0 / 3937.0, "ft (US)")),
        _ => None,
    };
    match row {
        Some((factor, symbol)) => format!("{} {}", fixed(metres / factor, n), symbol),
        None => format!("{} m", fixed(metres, n)),
    }
}

