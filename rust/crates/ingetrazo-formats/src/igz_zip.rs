// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! The .igz container written byte for byte as `igz._write_container`
//! (Python's `zipfile` with fixed 1980-01-01 stamps) writes it.
//!
//! The uncompressed members are identical on every platform. The deflated
//! bytes are those of zlib: this crate links madler zlib (`libz-sys`, via
//! flate2's `zlib` feature) at the same level and memLevel as CPython's
//! `zlib.compressobj`, so they equal what a CPython built on the same zlib
//! version writes (the Windows reference build). A CPython on a zlib-ng
//! distribution deflates differently, and so does its output here.

use crate::igz::{IgzError, DOC_ENTRY};

/// One member of a container: deflated at `level` (raw deflate, zlib's
/// default memLevel 8 and strategy, as `zipfile` asks of
/// `zlib.compressobj`) or STORED when `level` is `None`.
pub struct ZipItem<'a> {
    pub name: &'a str,
    pub data: &'a [u8],
    pub level: Option<u32>,
}

/// Python's `compresslevel=None`: `Z_DEFAULT_COMPRESSION`, which zlib maps
/// to 6.
pub const DEFAULT_LEVEL: u32 = 6;

/// 1980-01-01 00:00:00 (`_ZIP_EPOCH`) as MS-DOS time and date.
const DOS_TIME: u16 = 0;
const DOS_DATE: u16 = 33;
/// `0o600 << 16`: the external attributes `zipfile` gives a member written
/// from a `ZipInfo` on Windows.
const EXTERNAL_ATTR: u32 = 0o600 << 16;
/// Members of 2 GB and more make `zipfile` write ZIP64 records, which this
/// writer does not.
const ZIP64_LIMIT: usize = 0x7fff_ffff;

fn fail(msg: String) -> IgzError {
    IgzError::Container(msg)
}

/// Raw-deflate `data` at `level` with the zlib this crate links.
pub fn deflate(data: &[u8], level: u32) -> Result<Vec<u8>, IgzError> {
    use std::io::Write;
    let mut enc = flate2::write::DeflateEncoder::new(Vec::new(), flate2::Compression::new(level));
    enc.write_all(data).map_err(|e| fail(format!("deflate: {e}")))?;
    enc.finish().map_err(|e| fail(format!("deflate: {e}")))
}

/// The bytes `zipfile.ZipFile(BytesIO(), "w")` produces for `writestr` of
/// each item in turn with a fixed-date `ZipInfo`: local header (CRC and
/// sizes filled in), data, then the central directory and its end record,
/// with the same version fields, flags and attributes. No extra fields, no
/// comments.
pub fn write_zip(items: &[ZipItem]) -> Result<Vec<u8>, IgzError> {
    struct Done {
        name: Vec<u8>,
        flags: u16,
        method: u16,
        crc: u32,
        csize: u32,
        size: u32,
        offset: u32,
    }
    if items.len() > 0xffff {
        return Err(fail("too many members for a ZIP without ZIP64".into()));
    }
    let mut out: Vec<u8> = Vec::new();
    let mut done: Vec<Done> = Vec::with_capacity(items.len());
    for it in items {
        if it.data.len() >= ZIP64_LIMIT || it.name.len() > 0xffff {
            return Err(fail(format!("{}: too big for a ZIP without ZIP64", it.name)));
        }
        // zipfile's _encodeFilenameFlags: ASCII as is; anything else UTF-8
        // with the "language encoding" flag.
        let flags: u16 = if it.name.is_ascii() { 0 } else { 0x800 };
        let mut crc = flate2::Crc::new();
        crc.update(it.data);
        let packed;
        let (method, body): (u16, &[u8]) = match it.level {
            Some(level) => {
                packed = deflate(it.data, level)?;
                (8, &packed)
            }
            None => (0, it.data),
        };
        if out.len() + body.len() >= ZIP64_LIMIT {
            return Err(fail("archive too big for a ZIP without ZIP64".into()));
        }
        let rec = Done {
            name: it.name.as_bytes().to_vec(),
            flags,
            method,
            crc: crc.sum(),
            csize: body.len() as u32,
            size: it.data.len() as u32,
            offset: out.len() as u32,
        };
        out.extend_from_slice(b"PK\x03\x04");
        out.extend_from_slice(&20u16.to_le_bytes()); // version needed
        out.extend_from_slice(&rec.flags.to_le_bytes());
        out.extend_from_slice(&rec.method.to_le_bytes());
        out.extend_from_slice(&DOS_TIME.to_le_bytes());
        out.extend_from_slice(&DOS_DATE.to_le_bytes());
        out.extend_from_slice(&rec.crc.to_le_bytes());
        out.extend_from_slice(&rec.csize.to_le_bytes());
        out.extend_from_slice(&rec.size.to_le_bytes());
        out.extend_from_slice(&(rec.name.len() as u16).to_le_bytes());
        out.extend_from_slice(&0u16.to_le_bytes()); // extra length
        out.extend_from_slice(&rec.name);
        out.extend_from_slice(body);
        done.push(rec);
    }
    let dir_start = out.len();
    for rec in &done {
        out.extend_from_slice(b"PK\x01\x02");
        // create_version, create_system (0 = Windows), extract_version, reserved
        out.extend_from_slice(&[20, 0, 20, 0]);
        out.extend_from_slice(&rec.flags.to_le_bytes());
        out.extend_from_slice(&rec.method.to_le_bytes());
        out.extend_from_slice(&DOS_TIME.to_le_bytes());
        out.extend_from_slice(&DOS_DATE.to_le_bytes());
        out.extend_from_slice(&rec.crc.to_le_bytes());
        out.extend_from_slice(&rec.csize.to_le_bytes());
        out.extend_from_slice(&rec.size.to_le_bytes());
        out.extend_from_slice(&(rec.name.len() as u16).to_le_bytes());
        out.extend_from_slice(&0u16.to_le_bytes()); // extra length
        out.extend_from_slice(&0u16.to_le_bytes()); // comment length
        out.extend_from_slice(&0u16.to_le_bytes()); // disk number
        out.extend_from_slice(&0u16.to_le_bytes()); // internal attributes
        out.extend_from_slice(&EXTERNAL_ATTR.to_le_bytes());
        out.extend_from_slice(&rec.offset.to_le_bytes());
        out.extend_from_slice(&rec.name);
    }
    let dir_size = out.len() - dir_start;
    out.extend_from_slice(b"PK\x05\x06");
    out.extend_from_slice(&0u16.to_le_bytes());
    out.extend_from_slice(&0u16.to_le_bytes());
    out.extend_from_slice(&(done.len() as u16).to_le_bytes());
    out.extend_from_slice(&(done.len() as u16).to_le_bytes());
    out.extend_from_slice(&(dir_size as u32).to_le_bytes());
    out.extend_from_slice(&(dir_start as u32).to_le_bytes());
    out.extend_from_slice(&0u16.to_le_bytes());
    Ok(out)
}

/// `igz._write_container`: `document.json` deflated, then the `deflated`
/// members (geometry blocks) deflated, sorted by name, then the `blobs`
/// (textures) STORED, sorted by name. `fast` deflates at level 1.
pub fn write_container(doc: &str, blobs: &[(String, Vec<u8>)], deflated: &[(String, Vec<u8>)], fast: bool) -> Result<Vec<u8>, IgzError> {
    let level = if fast { 1 } else { DEFAULT_LEVEL };
    let mut geo: Vec<&(String, Vec<u8>)> = deflated.iter().collect();
    geo.sort_by(|a, b| a.0.cmp(&b.0));
    let mut tex: Vec<&(String, Vec<u8>)> = blobs.iter().collect();
    tex.sort_by(|a, b| a.0.cmp(&b.0));
    let mut items = vec![ZipItem { name: DOC_ENTRY, data: doc.as_bytes(), level: Some(level) }];
    items.extend(geo.iter().map(|(n, d)| ZipItem { name: n, data: d, level: Some(level) }));
    items.extend(tex.iter().map(|(n, d)| ZipItem { name: n, data: d, level: None }));
    write_zip(&items)
}
