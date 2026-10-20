// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! A reader for ASCII DXF: the tags of the HEADER, BLOCKS and ENTITIES
//! sections, grouped into entities the way ezdxf hands them to
//! `formats/dxf_in.py` (POLYLINE owns its VERTEX records, an INSERT its
//! ATTRIBs). Lenient on structure, like ezdxf's `recover` fall-back: a
//! missing ENDSEC or ENDBLK does not stop the read, comments are skipped.
//! Binary DXF is not read.

use std::collections::HashMap;

use super::geom::{P, NULLVEC};
use crate::pyutil;

/// One group code and its value, as text.
#[derive(Clone, Debug)]
pub struct Tag {
    pub code: i32,
    pub value: String,
}

/// An entity: its type name, its tags (after the `0` tag) and, for a
/// POLYLINE, its VERTEX records.
#[derive(Clone, Debug, Default)]
pub struct RawEntity {
    pub kind: String,
    pub tags: Vec<Tag>,
    pub subs: Vec<RawEntity>,
}

#[derive(Clone, Debug)]
pub struct Block {
    pub name: String,
    pub base_point: P,
    pub entities: Vec<RawEntity>,
}

#[derive(Clone, Debug, Default)]
pub struct Doc {
    /// `$ACADVER`, `AC1009` (R12) when absent.
    pub acadver: String,
    /// First value of every header variable.
    pub header: HashMap<String, String>,
    /// Whether the file had a HEADER section: ezdxf fills in its default
    /// header (with `$INSUNITS` = 6) when it has none.
    pub has_header: bool,
    pub modelspace: Vec<RawEntity>,
    pub blocks: Vec<Block>,
}

impl Doc {
    /// `doc.blocks.get(name)`: block names compare without case.
    pub fn block(&self, name: &str) -> Option<&Block> {
        let key = name.to_lowercase();
        self.blocks.iter().find(|b| b.name.to_lowercase() == key)
    }

    /// `int(doc.header.get("$INSUNITS", 0) or 0)`.
    pub fn insunits(&self) -> i64 {
        if !self.has_header {
            return 6;
        }
        self.header.get("$INSUNITS").and_then(|v| v.trim().parse::<i64>().ok()).unwrap_or(0)
    }

    /// `dxfversion == DXF12`.
    pub fn r12(&self) -> bool {
        self.acadver == "AC1009"
    }
}

impl RawEntity {
    pub fn tag(&self, code: i32) -> Option<&str> {
        self.tags.iter().rev().find(|t| t.code == code).map(|t| t.value.as_str())
    }

    pub fn has(&self, code: i32) -> bool {
        self.tags.iter().any(|t| t.code == code)
    }

    pub fn f(&self, code: i32) -> Option<f64> {
        self.tag(code).and_then(pyutil::float)
    }

    pub fn f_or(&self, code: i32, default: f64) -> f64 {
        self.f(code).unwrap_or(default)
    }

    pub fn i(&self, code: i32) -> Option<i64> {
        self.tag(code).and_then(|v| v.trim().parse::<i64>().ok())
    }

    pub fn i_or(&self, code: i32, default: i64) -> i64 {
        self.i(code).unwrap_or(default)
    }

    /// A point attribute: x at `code`, y at `code + 10`, z at `code + 20`;
    /// `None` when none of the three is present.
    pub fn pt(&self, code: i32) -> Option<P> {
        if !self.has(code) && !self.has(code + 10) && !self.has(code + 20) {
            return None;
        }
        Some(P::new(self.f_or(code, 0.0), self.f_or(code + 10, 0.0), self.f_or(code + 20, 0.0)))
    }

    /// A point attribute with ezdxf's default (the null vector).
    pub fn pt_or_null(&self, code: i32) -> P {
        self.pt(code).unwrap_or(NULLVEC)
    }

    /// `getattr(e.dxf, "layer", "0") or "0"`.
    pub fn layer(&self) -> String {
        match self.tag(8) {
            Some(l) if !l.is_empty() => l.to_string(),
            _ => "0".to_string(),
        }
    }
}

fn find(hay: &[u8], needle: &[u8]) -> Option<usize> {
    hay.windows(needle.len()).position(|w| w == needle)
}

/// The value of a header variable, read from the raw bytes (the file's
/// encoding depends on two of them).
fn raw_header_value(data: &[u8], name: &str) -> Option<String> {
    let at = find(data, name.as_bytes())?;
    let tail = &data[at + name.len()..];
    let text: String = tail.iter().take(200).map(|&b| b as char).collect();
    let mut lines = text.split(['\n', '\r']).filter(|l| !l.is_empty());
    lines.next()?; // the group code
    Some(lines.next()?.trim().to_string())
}

fn decode(data: &[u8]) -> String {
    let ver = raw_header_value(data, "$ACADVER").unwrap_or_else(|| "AC1009".to_string());
    if ver.as_str() >= "AC1021" {
        String::from_utf8_lossy(data).into_owned()
    } else {
        // ANSI_1252 is the codepage of everything in the corpus; other
        // pages read as cp1252 too.
        data.iter().map(|&b| pyutil::cp1252(b)).collect()
    }
}

/// Reads the tags: pairs of lines (universal newlines); comments skipped.
fn tags_of(text: &str) -> Result<Vec<Tag>, String> {
    let text = text.replace("\r\n", "\n").replace('\r', "\n");
    let mut lines = text.split('\n');
    let mut tags = Vec::new();
    let mut line = 1;
    while let Some(code) = lines.next() {
        if code.is_empty() {
            // The empty string at the end of the file (readline gives EOF).
            if lines.clone().next().is_none() {
                break;
            }
        }
        let group: i32 = code.trim().parse().map_err(|_| format!("Invalid group code \"{code}\" at line {line}."))?;
        let Some(value) = lines.next() else { break };
        line += 2;
        if group == 0 && value == "EOF" {
            tags.push(Tag { code: 0, value: "EOF".to_string() });
            break;
        }
        if group != 999 {
            tags.push(Tag { code: group, value: value.to_string() });
        }
    }
    Ok(tags)
}

#[derive(PartialEq)]
enum Num {
    Float,
    Int,
}

/// `TYPE_TABLE` of `ezdxf.lldxf.types`: the numeric group codes.
fn numeric(code: i32) -> Option<Num> {
    match code {
        10..=59 | 110..=149 | 210..=239 | 460..=469 | 1010..=1059 => Some(Num::Float),
        60..=79 | 90..=99 | 160..=179 | 270..=299 | 370..=389 | 400..=409 | 420..=429 | 440..=459 | 1060..=1071 => Some(Num::Int),
        _ => None,
    }
}

/// `_search_float`: the first `[+-]?\d+(:?\.\d*)?(:?[eE][+-]?\d+)?` of the
/// text, as `float()` reads it (the odd `:?` of the pattern included).
fn search_float(s: &str) -> Option<f64> {
    let b: Vec<char> = s.chars().collect();
    let digit = |i: usize| b.get(i).is_some_and(|c| c.is_ascii_digit());
    for start in 0..b.len() {
        let mut j = start;
        if matches!(b[j], '+' | '-') {
            j += 1;
        }
        if !digit(j) {
            continue;
        }
        while digit(j) {
            j += 1;
        }
        let mut m = j;
        let colon = usize::from(b.get(m) == Some(&':'));
        if b.get(m + colon) == Some(&'.') {
            m += colon + 1;
            while digit(m) {
                m += 1;
            }
        }
        let mut e = m;
        let colon = usize::from(b.get(e) == Some(&':'));
        if matches!(b.get(e + colon), Some('e' | 'E')) {
            let mut k = e + colon + 1;
            if matches!(b.get(k), Some('+' | '-')) {
                k += 1;
            }
            if digit(k) {
                while digit(k) {
                    k += 1;
                }
                e = k;
            }
        }
        let text: String = b[start..e].iter().collect();
        return pyutil::float(&text);
    }
    pyutil::float(s)
}

/// `_search_int`: the first `[+-]?\d+`.
fn search_int(s: &str) -> Option<i64> {
    let b: Vec<char> = s.chars().collect();
    for start in 0..b.len() {
        let mut j = start;
        if matches!(b[j], '+' | '-') {
            j += 1;
        }
        let from = j;
        while b.get(j).is_some_and(|c| c.is_ascii_digit()) {
            j += 1;
        }
        if j > from {
            let text: String = b[start..j].iter().collect();
            return text.parse().ok();
        }
    }
    s.trim().parse().ok()
}

/// `decode_dxf_unicode`: `\U+xxxx` to the character.
fn decode_dxf_unicode(s: &str) -> String {
    let b: Vec<char> = s.chars().collect();
    let mut out = String::new();
    let mut i = 0;
    while i < b.len() {
        if b[i] == '\\' && b.get(i + 1) == Some(&'U') && b.get(i + 2) == Some(&'+') && i + 7 <= b.len() {
            let hex: String = b[i + 3..i + 7].iter().collect();
            if let Some(c) = u32::from_str_radix(&hex, 16).ok().and_then(char::from_u32) {
                out.push(c);
                i += 7;
                continue;
            }
        }
        out.push(b[i]);
        i += 1;
    }
    out
}

/// ezdxf reads a file in two goes: strictly, and, when a number does not
/// parse, with its `recover` tag compiler, which also upper-cases structure
/// names and decodes `\U+xxxx` in strings. This is that second reading.
fn recover(tags: Vec<Tag>) -> Result<Vec<Tag>, String> {
    let mut out = Vec::with_capacity(tags.len());
    for t in tags {
        let value = if t.code == 0 {
            t.value.trim().to_uppercase()
        } else {
            match numeric(t.code) {
                Some(Num::Float) => match pyutil::float(&t.value) {
                    Some(_) => t.value,
                    None => {
                        let stripped: String = t.value.chars().filter(|c| !c.is_whitespace()).collect();
                        let v = search_float(&stripped).ok_or_else(|| format!("Invalid tag (code={}, value=\"{}\").", t.code, t.value))?;
                        format!("{v:?}")
                    }
                },
                Some(Num::Int) => match t.value.trim().parse::<i64>() {
                    Ok(_) => t.value,
                    Err(_) => {
                        let v = search_int(&t.value).ok_or_else(|| format!("Invalid tag (code={}, value=\"{}\").", t.code, t.value))?;
                        v.to_string()
                    }
                },
                None if (310..=319).contains(&t.code) || t.code == 1004 => t.value,
                None => decode_dxf_unicode(&t.value),
            }
        };
        out.push(Tag { code: t.code, value });
    }
    Ok(out)
}

/// Whether the strict reading would stop at a number.
fn has_invalid_number(tags: &[Tag]) -> bool {
    tags.iter().any(|t| match numeric(t.code) {
        Some(Num::Float) => pyutil::float(&t.value).is_none(),
        Some(Num::Int) => t.value.trim().parse::<i64>().is_err(),
        None => false,
    })
}

/// Splits a section's tags into entities and links VERTEX / ATTRIB records
/// to their POLYLINE / INSERT, as ezdxf's `entity_linker` does: the records
/// after a POLYLINE (or an INSERT with `attribs_follow`) must be its
/// VERTEXes (ATTRIBs) up to a SEQEND, anything else is a structure error;
/// a stray VERTEX, ATTRIB or SEQEND stays an entity of its own.
fn entities_of(tags: &[Tag]) -> Result<Vec<RawEntity>, String> {
    let mut records: Vec<RawEntity> = Vec::new();
    for t in tags {
        if t.code == 0 {
            records.push(RawEntity { kind: t.value.trim().to_string(), tags: Vec::new(), subs: Vec::new() });
        } else if let Some(r) = records.last_mut() {
            r.tags.push(t.clone());
        }
    }
    let mut out: Vec<RawEntity> = Vec::new();
    let mut main: Option<usize> = None;
    let mut expected = "";
    for r in records {
        if let Some(mi) = main {
            if r.kind == "SEQEND" {
                main = None;
            } else if r.kind == expected {
                out[mi].subs.push(r);
            } else {
                return Err(format!("Expected DXF entity {} or SEQEND", r.kind));
            }
            continue;
        }
        let links = match r.kind.as_str() {
            "POLYLINE" => Some("VERTEX"),
            "INSERT" if r.i(66).unwrap_or(0) != 0 => Some("ATTRIB"),
            _ => None,
        };
        out.push(r);
        if let Some(e) = links {
            main = Some(out.len() - 1);
            expected = e;
        }
    }
    Ok(out)
}

/// `ezdxf.readfile` for ASCII DXF.
pub fn read_dxf(data: &[u8]) -> Result<Doc, String> {
    if data.starts_with(b"AutoCAD Binary DXF") {
        return Err("binary DXF is not supported".to_string());
    }
    let mut tags = tags_of(&decode(data))?;
    if has_invalid_number(&tags) {
        tags = recover(tags)?;
    }
    if !(tags.first().is_some_and(|t| t.code == 0 && t.value.trim() == "SECTION")) {
        return Err("is not a DXF file".to_string());
    }
    let mut doc = Doc { acadver: raw_header_value(data, "$ACADVER").unwrap_or_else(|| "AC1009".to_string()), ..Default::default() };
    let mut i = 0;
    while i < tags.len() {
        if !(tags[i].code == 0 && tags[i].value.trim() == "SECTION") {
            i += 1;
            continue;
        }
        let name = if i + 1 < tags.len() && tags[i + 1].code == 2 { tags[i + 1].value.trim().to_string() } else { String::new() };
        let start = (i + 2).min(tags.len());
        // A section ends at ENDSEC, EOF or the next SECTION; one that runs
        // off the end of the file is lost.
        let mut end = start;
        while end < tags.len() && !(tags[end].code == 0 && matches!(tags[end].value.trim(), "ENDSEC" | "EOF" | "SECTION")) {
            end += 1;
        }
        if end >= tags.len() {
            break;
        }
        let body = &tags[start..end];
        match name.as_str() {
            "HEADER" => {
                doc.has_header = true;
                let mut k = 0;
                while k < body.len() {
                    if body[k].code == 9 {
                        let var = body[k].value.trim().to_string();
                        if k + 1 < body.len() && body[k + 1].code != 9 {
                            doc.header.entry(var).or_insert_with(|| body[k + 1].value.clone());
                        }
                    }
                    k += 1;
                }
            }
            "BLOCKS" => {
                let mut current: Option<(Block, Vec<Tag>)> = None;
                let mut k = 0;
                while k < body.len() {
                    // Collect one record at a time: BLOCK opens, ENDBLK closes.
                    let mut e = k + 1;
                    while e < body.len() && body[e].code != 0 {
                        e += 1;
                    }
                    let kind = body[k].value.trim();
                    if body[k].code == 0 && kind == "BLOCK" {
                        // A block that never reached its ENDBLK is lost.
                        current.take();
                        let rec = RawEntity { kind: "BLOCK".to_string(), tags: body[k + 1..e].to_vec(), subs: Vec::new() };
                        let name = rec.tag(2).or_else(|| rec.tag(3)).unwrap_or("").to_string();
                        current = Some((Block { name, base_point: rec.pt_or_null(10), entities: Vec::new() }, Vec::new()));
                    } else if body[k].code == 0 && kind == "ENDBLK" {
                        if let Some((b, t)) = current.take() {
                            doc.blocks.push(Block { entities: entities_of(&t)?, ..b });
                        }
                    } else if let Some((_, t)) = current.as_mut() {
                        t.extend_from_slice(&body[k..e]);
                    }
                    k = e;
                }
            }
            "ENTITIES" => {
                doc.modelspace = entities_of(body)?.into_iter().filter(|e| e.i(67).unwrap_or(0) != 1).collect();
            }
            _ => {}
        }
        i = end;
    }
    Ok(doc)
}
