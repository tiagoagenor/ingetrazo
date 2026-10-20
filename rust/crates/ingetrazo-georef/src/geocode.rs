// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! `georef/geocode.py`: location lookup for the project locator. Pure URL
//! builders and response parsers (Nominatim place search, ipapi.co "locate
//! me"); the GET itself is the app's [`HttpGet`] (`search` and `locate` here
//! are the `Geocoder` / `IpLocator` flows over it, with no Qt signals).

use ingetrazo_formats::pyjson::{loads, Value};

use crate::pyconv::{get, py_float, py_str, py_strip, truthy, PyResult};
use crate::tile_fetcher::{user_agent, HttpGet, APP_VERSION};

pub const NOMINATIM: &str = "https://nominatim.openstreetmap.org/search";
pub const IPAPI: &str = "https://ipapi.co/json/";

/// `Place`: a geocoding candidate.
#[derive(Clone, Debug, PartialEq)]
pub struct Place {
    pub name: String,
    pub lat: f64,
    pub lon: f64,
}

/// `urllib.parse.quote(s)`: UTF-8 percent-encoding keeping letters, digits,
/// `_.-~` and `/`.
pub fn quote(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for &b in s.as_bytes() {
        if b.is_ascii_alphanumeric() || matches!(b, b'_' | b'.' | b'-' | b'~' | b'/') {
            out.push(b as char);
        } else {
            out.push_str(&format!("%{b:02X}"));
        }
    }
    out
}

/// `geocode_url` (Python's default `limit` is 8).
pub fn geocode_url(query: &str, limit: i64) -> String {
    format!("{NOMINATIM}?q={}&format=jsonv2&limit={}", quote(query), limit)
}

fn decode_utf16(b: &[u8], big: bool) -> Option<String> {
    if !b.len().is_multiple_of(2) {
        return None;
    }
    let units: Vec<u16> = b
        .chunks(2)
        .map(|c| if big { u16::from_be_bytes([c[0], c[1]]) } else { u16::from_le_bytes([c[0], c[1]]) })
        .collect();
    String::from_utf16(&units).ok()
}

fn decode_utf32(b: &[u8], big: bool) -> Option<String> {
    if !b.len().is_multiple_of(4) {
        return None;
    }
    let mut s = String::new();
    for c in b.chunks(4) {
        let v = if big { u32::from_be_bytes([c[0], c[1], c[2], c[3]]) } else { u32::from_le_bytes([c[0], c[1], c[2], c[3]]) };
        s.push(char::from_u32(v)?);
    }
    Some(s)
}

/// `json.loads(bytes(data))`: `json.detect_encoding` picks UTF-8/16/32 from a
/// BOM or the NUL pattern; text that does not decode, or does not parse, is
/// `None` (Python's `ValueError`).
fn loads_bytes(b: &[u8]) -> Option<Value> {
    let text = if b.starts_with(&[0xff, 0xfe, 0x00, 0x00]) {
        decode_utf32(&b[4..], false)
    } else if b.starts_with(&[0x00, 0x00, 0xfe, 0xff]) {
        decode_utf32(&b[4..], true)
    } else if b.starts_with(&[0xff, 0xfe]) {
        decode_utf16(&b[2..], false)
    } else if b.starts_with(&[0xfe, 0xff]) {
        decode_utf16(&b[2..], true)
    } else if b.starts_with(&[0xef, 0xbb, 0xbf]) {
        String::from_utf8(b[3..].to_vec()).ok()
    } else if b.len() >= 4 && b[0] == 0 {
        if b[1] != 0 { decode_utf16(b, true) } else { decode_utf32(b, true) }
    } else if b.len() >= 4 && b[1] == 0 {
        if b[2] != 0 || b[3] != 0 { decode_utf16(b, false) } else { decode_utf32(b, false) }
    } else if b.len() == 2 && b[0] == 0 {
        decode_utf16(b, true)
    } else if b.len() == 2 && b[1] == 0 {
        decode_utf16(b, false)
    } else {
        String::from_utf8(b.to_vec()).ok()
    }?;
    if text.starts_with('\u{feff}') {
        return None; // "Unexpected UTF-8 BOM"
    }
    loads(&text).ok()
}

/// `parse_places`: a Nominatim jsonv2 response into candidates, best effort:
/// undecodable input or a non-list is `[]`, an entry without a usable
/// `lat`/`lon` is skipped. `Err` for what Python lets escape: an entry that
/// is not an object (`AttributeError`) or an int too large for a float
/// (`OverflowError`).
///
/// A `display_name` that is not text becomes its `str()`.
pub fn parse_places(data: &[u8]) -> PyResult<Vec<Place>> {
    let Some(Value::Array(arr)) = loads_bytes(data) else { return Ok(Vec::new()) };
    let mut out = Vec::new();
    for r in &arr {
        let name = match get(r, "display_name")? {
            None => String::new(),
            Some(v) => py_str(v),
        };
        let Some(lat_v) = r.get("lat") else { continue };
        let lat = match py_float(lat_v) {
            Ok(x) => x,
            Err(e) if e.kind == "OverflowError" => return Err(e),
            Err(_) => continue,
        };
        let Some(lon_v) = r.get("lon") else { continue };
        let lon = match py_float(lon_v) {
            Ok(x) => x,
            Err(e) if e.kind == "OverflowError" => return Err(e),
            Err(_) => continue,
        };
        out.push(Place { name, lat, lon });
    }
    Ok(out)
}

/// `parse_ip`: an ipapi.co response into `(lat, lon, label)`, or `None`.
/// `Err` for a response that is not an object (`AttributeError`) or an int
/// too large for a float.
pub fn parse_ip(data: &[u8]) -> PyResult<Option<(f64, f64, String)>> {
    let Some(d) = loads_bytes(data) else { return Ok(None) };
    let lat = get(&d, "latitude")?;
    let lon = d.get("longitude");
    let (lat, lon) = match (lat, lon) {
        (Some(a), Some(b)) if !matches!(a, Value::Null) && !matches!(b, Value::Null) => (a, b),
        _ => return Ok(None),
    };
    let parts: Vec<String> = ["city", "region", "country_name"]
        .iter()
        .filter_map(|k| d.get(k))
        .filter(|v| truthy(v))
        .map(py_str)
        .collect();
    let label = parts.join(", ");
    let conv = |v: &Value| match py_float(v) {
        Ok(x) => Ok(Some(x)),
        Err(e) if e.kind == "OverflowError" => Err(e),
        Err(_) => Ok(None),
    };
    let (Some(la), Some(lo)) = (conv(lat)?, conv(lon)?) else { return Ok(None) };
    Ok(Some((la, lo, label)))
}

/// `Geocoder.search`'s first step: what to do with a typed query.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SearchPlan {
    /// Blank query: `resultsReady([])` at once, no request.
    Empty,
    /// GET this URL.
    Get(String),
}

/// `Geocoder.search` planning: `(query or "").strip()`, then the URL for the
/// stripped query with `limit=8`.
pub fn plan_search(query: &str) -> SearchPlan {
    let q = py_strip(query);
    if q.is_empty() {
        SearchPlan::Empty
    } else {
        SearchPlan::Get(geocode_url(q, 8))
    }
}

/// The whole `Geocoder.search` flow over an HTTP client.
pub fn search(http: &dyn HttpGet, query: &str) -> Result<Vec<Place>, String> {
    match plan_search(query) {
        SearchPlan::Empty => Ok(Vec::new()),
        SearchPlan::Get(url) => {
            let body = http.get(&url, &user_agent(APP_VERSION))?;
            parse_places(&body).map_err(|e| format!("{}: {}", e.kind, e.msg))
        }
    }
}

/// The whole `IpLocator.locate` flow: `Ok(Err(..))` is the
/// `failed("no location in response")` case.
pub fn locate(http: &dyn HttpGet) -> Result<(f64, f64, String), String> {
    let body = http.get(IPAPI, &user_agent(APP_VERSION))?;
    match parse_ip(&body) {
        Ok(Some(r)) => Ok(r),
        Ok(None) => Err("no location in response".to_string()),
        Err(e) => Err(format!("{}: {}", e.kind, e.msg)),
    }
}
