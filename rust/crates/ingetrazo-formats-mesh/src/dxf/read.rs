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
    let tags = tags_of(&decode(data))?;
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
