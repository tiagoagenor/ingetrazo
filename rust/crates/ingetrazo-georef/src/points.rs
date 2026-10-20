// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! `georef/points.py`: surveyed GPS / total-station control points
//! (`Scene.geo_points`) and the survey-CSV (`P,N,E,Z[,D]`) reader.

use ingetrazo_formats::pyjson::Value;
use ingetrazo_math::V3;

use crate::datum::{utm_inverse, SceneDatum};
use crate::pyconv::{
    float_str, get, py_split_ws, py_splitlines, py_strip, qvec3, text_or_empty, v3_value, PyResult,
};

/// `GeoPoint`: one surveyed control point, in local scene metres.
#[derive(Clone, Debug, PartialEq)]
pub struct GeoPoint {
    pub position: V3,
    pub name: String,
    pub desc: String,
}

impl GeoPoint {
    pub fn new(position: V3, name: &str, desc: &str) -> GeoPoint {
        GeoPoint { position, name: name.to_string(), desc: desc.to_string() }
    }

    /// `to_dict`: `name` and `desc` only when non-empty.
    pub fn to_dict(&self) -> Value {
        let mut entry = vec![("position".to_string(), v3_value(self.position))];
        if !self.name.is_empty() {
            entry.push(("name".to_string(), Value::Str(self.name.clone())));
        }
        if !self.desc.is_empty() {
            entry.push(("desc".to_string(), Value::Str(self.desc.clone())));
        }
        Value::Object(entry)
    }

    /// `from_dict`: `position` defaults to the origin.
    pub fn from_dict(data: &Value) -> PyResult<GeoPoint> {
        let position = match get(data, "position")? {
            Some(v) => qvec3(v)?,
            None => V3(0.0, 0.0, 0.0),
        };
        Ok(GeoPoint {
            position,
            name: text_or_empty(get(data, "name")?)?,
            desc: text_or_empty(get(data, "desc")?)?,
        })
    }
}

/// One row of a survey CSV.
#[derive(Clone, Debug, PartialEq)]
pub struct PointRow {
    pub name: String,
    pub north: f64,
    pub east: f64,
    pub z: f64,
    pub desc: String,
}

/// `_num`: a coordinate cell, tolerating the comma decimal separator.
fn num(cell: &str) -> Option<f64> {
    float_str(&py_strip(cell).replace(',', "."))
}

/// `next(csv.reader(io.StringIO(line), delimiter=...))` for one line with no
/// line break in it: the default `excel` dialect (`"` quotes, doubled to
/// escape; a lone closing quote followed by text keeps the text).
pub fn csv_fields(line: &str, delimiter: char) -> Vec<String> {
    #[derive(PartialEq)]
    enum St {
        StartField,
        InField,
        InQuoted,
        QuoteInQuoted,
    }
    let mut fields = Vec::new();
    let mut cur = String::new();
    let mut st = St::StartField;
    for c in line.chars() {
        match st {
            St::StartField => {
                if c == '"' {
                    st = St::InQuoted;
                } else if c == delimiter {
                    fields.push(std::mem::take(&mut cur));
                } else {
                    cur.push(c);
                    st = St::InField;
                }
            }
            St::InField => {
                if c == delimiter {
                    fields.push(std::mem::take(&mut cur));
                    st = St::StartField;
                } else {
                    cur.push(c);
                }
            }
            St::InQuoted => {
                if c == '"' {
                    st = St::QuoteInQuoted;
                } else {
                    cur.push(c);
                }
            }
            St::QuoteInQuoted => {
                if c == '"' {
                    cur.push(c);
                    st = St::InQuoted;
                } else if c == delimiter {
                    fields.push(std::mem::take(&mut cur));
                    st = St::StartField;
                } else {
                    cur.push(c);
                    st = St::InField;
                }
            }
        }
    }
    // End of line (or of data inside an open quote): the last field is saved.
    fields.push(cur);
    fields
}

/// `parse_points_csv`: rows of `{name, north, east, z, desc}` from
/// `P,N,E,Z[,D]` text with `,`, `;`, tab or whitespace delimiters; header,
/// blank and `#` lines are skipped. `Err` (Python's `ValueError`) when no
/// point parses.
pub fn parse_points_csv(text: &str) -> Result<Vec<PointRow>, String> {
    let mut rows = Vec::new();
    for raw in py_splitlines(text) {
        let line = py_strip(raw);
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let cells: Vec<String> = if line.contains(';') {
            csv_fields(line, ';')
        } else if line.contains(',') {
            csv_fields(line, ',')
        } else {
            py_split_ws(line).into_iter().map(str::to_string).collect()
        };
        let cells: Vec<String> = cells.iter().map(|c| py_strip(c).to_string()).collect();
        if cells.len() < 4 {
            continue;
        }
        let (Some(north), Some(east), Some(z)) = (num(&cells[1]), num(&cells[2]), num(&cells[3])) else {
            continue; // header (or stray text) row
        };
        let desc = if cells.len() > 4 { py_strip(&cells[4..].join(",")).to_string() } else { String::new() };
        rows.push(PointRow { name: cells[0].clone(), north, east, z, desc });
    }
    if rows.is_empty() {
        return Err("No survey points found (expected P,N,E,Z[,desc])".to_string());
    }
    Ok(rows)
}

/// `points_from_rows`: UTM rows to [`GeoPoint`]s in the datum's local frame.
pub fn points_from_rows(rows: &[PointRow], datum: &SceneDatum) -> Vec<GeoPoint> {
    rows.iter().map(|r| GeoPoint::new(datum.utm_to_local(r.east, r.north, r.z), &r.name, &r.desc)).collect()
}

/// `datum_for_rows`: a [`SceneDatum`] anchored at the first point (used when
/// the scene has none yet), its altitude the point's own elevation, honouring
/// the survey's UTM zone even when the anchor longitude derives another.
///
/// Panics on an empty `rows` (Python's `IndexError`); `parse_points_csv`
/// never returns an empty list.
pub fn datum_for_rows(rows: &[PointRow], zone: i64, northern: bool) -> SceneDatum {
    let first = &rows[0];
    let (lat, lon) = utm_inverse(first.east, first.north, zone, northern);
    let mut datum = SceneDatum::new(lat, lon, first.z, 0.0);
    if datum.zone != zone {
        // The "freeze the zone" doctrine: offsets stay pure in the survey's
        // frame.
        datum.freeze_zone(zone, northern);
    }
    datum
}
