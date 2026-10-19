// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! The .igz document (`formats/igz.py`): its JSON, in either shape.
//!
//! Format 1 is the bare JSON document; formats 2 and 3 are a ZIP holding
//! `document.json` beside textures and geometry blocks. This module reads
//! the document of any of them and writes format 1 exactly as the Python
//! app does; the ZIP writer (byte-identical deflate) comes with M2.

use std::fmt;
use std::io::Read;

use crate::pyjson::{self, JsonError, Options, Value};

/// Bare JSON document (no images to carry).
pub const PLAIN_FORMAT: i64 = 1;
/// ZIP container: document.json + textures/.
pub const CONTAINER_FORMAT: i64 = 2;
/// + big meshes as binary blocks under geom/.
pub const GEOMETRY_FORMAT: i64 = 3;
/// The newest this build reads.
pub const CURRENT_FORMAT: i64 = GEOMETRY_FORMAT;
pub const DOC_ENTRY: &str = "document.json";
const ZIP_MAGIC: &[u8] = b"PK\x03\x04";
/// Above this many faces + edges a document is written compact, not
/// indented (`igz._LARGE_DOCUMENT`).
pub const LARGE_DOCUMENT: usize = 20_000;

#[derive(Debug)]
pub enum IgzError {
    /// Not a ZIP we can read, or no document.json in it.
    Container(String),
    Json(JsonError),
    /// A NaN or an infinity in the payload: Python refuses to write it
    /// (`NonFiniteDocumentError`); `where` is the path it reports.
    NonFinite { at: String },
}

impl fmt::Display for IgzError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            IgzError::Container(m) => write!(f, "not an IngeTrazo document: {m}"),
            IgzError::Json(e) => write!(f, "document.json: {e}"),
            IgzError::NonFinite { at } => write!(f, "a point with invalid coordinates at {at}"),
        }
    }
}

impl std::error::Error for IgzError {}

impl From<JsonError> for IgzError {
    fn from(e: JsonError) -> Self {
        IgzError::Json(e)
    }
}

pub fn is_container(raw: &[u8]) -> bool {
    raw.starts_with(ZIP_MAGIC)
}

/// The document's JSON text, from a file of any format.
pub fn document_text(raw: &[u8]) -> Result<String, IgzError> {
    let bytes = if is_container(raw) {
        Zip::open(raw)?.read(DOC_ENTRY)?
    } else {
        raw.to_vec()
    };
    String::from_utf8(bytes).map_err(|_| IgzError::Container("document.json is not UTF-8".into()))
}

/// The document of a file of any format, as `json.loads` reads it.
pub fn read_document(raw: &[u8]) -> Result<Value, IgzError> {
    Ok(pyjson::loads(&document_text(raw)?)?)
}

/// `json.dumps` of a whole document as `save_scene` writes it: compact for
/// a large document, indented by two otherwise; NaN refused.
pub fn dumps_document(data: &Value, large: bool) -> Result<String, IgzError> {
    let opts = if large { Options::compact() } else { Options::indent(2) }.allow_nan(false);
    pyjson::dumps(data, &opts).map_err(|_| IgzError::NonFinite {
        at: non_finite_where(data.get("scene").unwrap_or(data), "scene"),
    })
}

/// The bytes of a format-1 file holding `data`.
pub fn write_plain(data: &Value, large: bool) -> Result<Vec<u8>, IgzError> {
    Ok(dumps_document(data, large)?.into_bytes())
}

/// `igz._non_finite_where`: the first NaN or infinity, depth first.
pub fn non_finite_where(node: &Value, path: &str) -> String {
    let mut stack: Vec<(&Value, String)> = vec![(node, path.to_string())];
    while let Some((cur, here)) = stack.pop() {
        match cur {
            Value::Float(x) if !x.is_finite() => return here,
            Value::Object(items) => {
                stack.extend(items.iter().rev().map(|(k, v)| (v, format!("{here} › {k}"))));
            }
            Value::Array(items) => {
                stack.extend(items.iter().enumerate().rev().map(|(i, v)| (v, format!("{here}[{i}]"))));
            }
            _ => {}
        }
    }
    path.to_string()
}

// ---- a minimal ZIP reader ----------------------------------------------------

/// The members of a ZIP as Python's `zipfile` writes them: no encryption,
/// stored or deflated, no ZIP64 (an .igz member over 4 GB would need it).
pub struct Zip<'a> {
    raw: &'a [u8],
    entries: Vec<Entry>,
}

#[derive(Clone, Debug)]
pub struct Entry {
    pub name: String,
    pub method: u16,
    pub compressed: usize,
    pub size: usize,
    pub crc32: u32,
    header: usize,
}

fn u16_at(b: &[u8], at: usize) -> Option<u16> {
    Some(u16::from_le_bytes(b.get(at..at + 2)?.try_into().ok()?))
}

fn u32_at(b: &[u8], at: usize) -> Option<u32> {
    Some(u32::from_le_bytes(b.get(at..at + 4)?.try_into().ok()?))
}

fn bad(m: &str) -> IgzError {
    IgzError::Container(m.to_string())
}

impl<'a> Zip<'a> {
    pub fn open(raw: &'a [u8]) -> Result<Zip<'a>, IgzError> {
        // The end-of-central-directory record: the last PK\5\6 within the
        // final 64 KB + 22 bytes (an archive comment may follow it).
        let lo = raw.len().saturating_sub(22 + 0xffff);
        let eocd = (lo..raw.len().saturating_sub(21))
            .rev()
            .find(|&i| raw[i..].starts_with(b"PK\x05\x06"))
            .ok_or_else(|| bad("no end of central directory"))?;
        let count = u16_at(raw, eocd + 10).ok_or_else(|| bad("truncated"))? as usize;
        let mut at = u32_at(raw, eocd + 16).ok_or_else(|| bad("truncated"))? as usize;
        let mut entries = Vec::with_capacity(count);
        for _ in 0..count {
            if u32_at(raw, at) != Some(0x0201_4b50) {
                return Err(bad("bad central directory"));
            }
            let field = |off: usize| u32_at(raw, at + off).ok_or_else(|| bad("truncated"));
            let method = u16_at(raw, at + 10).ok_or_else(|| bad("truncated"))?;
            let crc32 = field(16)?;
            let compressed = field(20)? as usize;
            let size = field(24)? as usize;
            let n = u16_at(raw, at + 28).ok_or_else(|| bad("truncated"))? as usize;
            let extra = u16_at(raw, at + 30).ok_or_else(|| bad("truncated"))? as usize;
            let comment = u16_at(raw, at + 32).ok_or_else(|| bad("truncated"))? as usize;
            let header = field(42)? as usize;
            let name = raw.get(at + 46..at + 46 + n).ok_or_else(|| bad("truncated"))?;
            entries.push(Entry {
                name: String::from_utf8_lossy(name).into_owned(),
                method,
                compressed,
                size,
                crc32,
                header,
            });
            at += 46 + n + extra + comment;
        }
        Ok(Zip { raw, entries })
    }

    pub fn entries(&self) -> &[Entry] {
        &self.entries
    }

    /// The uncompressed bytes of member `name`, CRC checked.
    pub fn read(&self, name: &str) -> Result<Vec<u8>, IgzError> {
        let e = self
            .entries
            .iter()
            .find(|e| e.name == name)
            .ok_or_else(|| IgzError::Container(format!("no {name}")))?;
        let h = e.header;
        if u32_at(self.raw, h) != Some(0x0403_4b50) {
            return Err(bad("bad local header"));
        }
        let n = u16_at(self.raw, h + 26).ok_or_else(|| bad("truncated"))? as usize;
        let extra = u16_at(self.raw, h + 28).ok_or_else(|| bad("truncated"))? as usize;
        let start = h + 30 + n + extra;
        let data = self.raw.get(start..start + e.compressed).ok_or_else(|| bad("truncated member"))?;
        let out = match e.method {
            0 => data.to_vec(),
            8 => {
                let mut out = Vec::with_capacity(e.size);
                flate2::read::DeflateDecoder::new(data)
                    .read_to_end(&mut out)
                    .map_err(|err| IgzError::Container(format!("{name}: {err}")))?;
                out
            }
            m => return Err(IgzError::Container(format!("{name}: compression method {m}"))),
        };
        let mut crc = flate2::Crc::new();
        crc.update(&out);
        if out.len() != e.size || crc.sum() != e.crc32 {
            return Err(IgzError::Container(format!("{name}: bad CRC or size")));
        }
        Ok(out)
    }
}
