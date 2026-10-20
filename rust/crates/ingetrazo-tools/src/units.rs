// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! The readouts the drawing tools put on the value box (`core.units`:
//! `fmt_num`, `fmt_pair`, `fmt_angle_deg`), for a document in degrees to one
//! decimal - the default - and any length unit `ingetrazo-model` formats.

use ingetrazo_model::units::{fmt_len, fixed, localized, ModelUnits};

/// `fmt_num`: the number alone, for `a x b` pairs (metric); the imperial
/// forms keep their marks because the mark IS the unit.
pub fn fmt_num(metres: f64, units: &ModelUnits) -> String {
    match units.length.as_str() {
        "m" | "cm" | "mm" => {
            let factor = match units.length.as_str() {
                "m" => 1.0,
                "cm" => 100.0,
                _ => 1000.0,
            };
            localized(&fixed(metres * factor, units.precision.clamp(0, 6) as usize), &units.decimal)
        }
        _ => fmt_len(metres, units),
    }
}

/// `fmt_pair`: `3.00 x 2.00 m` (metric: one unit at the end) or
/// `3'0" x 2'0"` (imperial: each number carries its mark).
pub fn fmt_pair(a: f64, b: f64, units: &ModelUnits) -> String {
    match units.length.as_str() {
        "m" | "cm" | "mm" => format!("{} \u{d7} {}", fmt_num(a, units), fmt_len(b, units)),
        _ => format!("{} \u{d7} {}", fmt_len(a, units), fmt_len(b, units)),
    }
}

/// `fmt_angle_deg(degrees, signed)` in a document that reads degrees to one
/// decimal.
pub fn fmt_angle_deg(degrees: f64, signed: bool, units: &ModelUnits) -> String {
    let radians = degrees * (std::f64::consts::PI / 180.0);
    let factor = std::f64::consts::PI / 180.0;
    let txt = format!("{}\u{b0}", fixed(radians / factor, 1));
    let txt = if signed && !txt.starts_with('-') && !txt.starts_with('\u{2014}') { format!("+{txt}") } else { txt };
    localized(&txt, &units.decimal)
}
