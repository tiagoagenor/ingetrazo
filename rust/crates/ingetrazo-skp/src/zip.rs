// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
// Port of openskp (MIT, Copyright (c) 2024-2026 Ahsan Mehmood): see NOTICE-openskp.md.
//! The slice of Python's `zipfile` the reader uses: the central directory in
//! file order, and entries stored or deflated, checked against their CRC.

use std::collections::HashMap;
use std::io::Read;

#[derive(Clone, Debug)]
pub struct Entry {
    pub name: String,
    pub method: u16,
    pub crc: u32,
    pub compress_size: u64,
    pub file_size: u64,
    header_offset: u64,
}

pub struct Zip<'a> {
    data: &'a [u8],
    concat: i64,
    pub entries: Vec<Entry>,
    by_name: HashMap<String, usize>,
}

fn u16_at(d: &[u8], at: usize) -> Option<u16> {
    d.get(at..at + 2).map(|b| u16::from_le_bytes([b[0], b[1]]))
}
fn u32_at(d: &[u8], at: usize) -> Option<u32> {
    d.get(at..at + 4).map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
}
fn u64_at(d: &[u8], at: usize) -> Option<u64> {
    d.get(at..at + 8).map(|b| u64::from_le_bytes(b.try_into().unwrap()))
}

const EOCD: &[u8; 4] = b"PK\x05\x06";
const BAD: &str = "File is not a zip file";

impl<'a> Zip<'a> {
    /// `zipfile.ZipFile(io.BytesIO(data))`.
    pub fn open(data: &'a [u8]) -> Result<Zip<'a>, String> {
        let n = data.len();
        // _EndRecData: the record with no comment, else the last one in the
        // final 64 KiB.
        let eocd_at = if n >= 22 && &data[n - 22..n - 18] == EOCD && data[n - 2..] == [0, 0] {
            n - 22
        } else {
            let start = n.saturating_sub(65535 + 22);
            let tail = &data[start..];
            match tail.windows(4).rposition(|w| w == EOCD) {
                Some(i) if tail.len() - i >= 22 => start + i,
                _ => return Err(BAD.into()),
            }
        };
        let e = &data[eocd_at..];
        let mut size_cd = u32_at(e, 12).ok_or(BAD)? as u64;
        let mut offset_cd = u32_at(e, 16).ok_or(BAD)? as u64;
        let mut total = u16_at(e, 10).ok_or(BAD)? as u64;
        let mut location = eocd_at as i64;
        let mut concat_adj = 0i64;
        // zip64 locator right before the record
        if eocd_at >= 20 && &data[eocd_at - 20..eocd_at - 16] == b"PK\x06\x07" {
            let loc = &data[eocd_at - 20..eocd_at];
            let rec_off = u64_at(loc, 8).ok_or(BAD)?;
            let rec_at = rec_off as usize;
            if data.get(rec_at..rec_at + 4) == Some(b"PK\x06\x06") {
                let r = &data[rec_at..];
                total = u64_at(r, 32).ok_or(BAD)?;
                size_cd = u64_at(r, 40).ok_or(BAD)?;
                offset_cd = u64_at(r, 48).ok_or(BAD)?;
                location = rec_at as i64;
                concat_adj = 0;
            }
        }
        let concat = location - size_cd as i64 - offset_cd as i64 - concat_adj;
        let cd_start = offset_cd as i64 + concat;
        if cd_start < 0 || cd_start as usize > n {
            return Err(BAD.into());
        }
        let mut pos = cd_start as usize;
        let mut entries = Vec::new();
        let mut by_name = HashMap::new();
        let end = pos + size_cd as usize;
        while pos < end.min(n) {
            let h = data.get(pos..pos + 46).ok_or("Truncated central directory")?;
            if &h[..4] != b"PK\x01\x02" {
                return Err("Bad magic number for central directory".into());
            }
            let flags = u16_at(h, 8).unwrap();
            let method = u16_at(h, 10).unwrap();
            let crc = u32_at(h, 16).unwrap();
            let mut comp = u32_at(h, 20).unwrap() as u64;
            let mut size = u32_at(h, 24).unwrap() as u64;
            let fnlen = u16_at(h, 28).unwrap() as usize;
            let exlen = u16_at(h, 30).unwrap() as usize;
            let cmlen = u16_at(h, 32).unwrap() as usize;
            let mut off = u32_at(h, 42).unwrap() as u64;
            let name_b = data.get(pos + 46..pos + 46 + fnlen).ok_or("Truncated central directory")?;
            let extra = data.get(pos + 46 + fnlen..pos + 46 + fnlen + exlen).ok_or("Truncated central directory")?;
            // zip64 extra: the 32-bit fields that read 0xFFFFFFFF, in order.
            let mut ex = 0;
            while ex + 4 <= extra.len() {
                let tp = u16_at(extra, ex).unwrap();
                let ln = u16_at(extra, ex + 2).unwrap() as usize;
                if tp == 1 {
                    let mut at = ex + 4;
                    let end_ex = (ex + 4 + ln).min(extra.len());
                    for field in [&mut size, &mut comp, &mut off] {
                        if *field == 0xFFFF_FFFF && at + 8 <= end_ex {
                            *field = u64_at(extra, at).unwrap();
                            at += 8;
                        }
                    }
                }
                ex += 4 + ln;
            }
            let name = if flags & 0x800 != 0 { String::from_utf8_lossy(name_b).into_owned() } else { name_b.iter().map(|&b| b as char).collect() };
            by_name.insert(name.clone(), entries.len());
            entries.push(Entry { name, method, crc, compress_size: comp, file_size: size, header_offset: off });
            pos += 46 + fnlen + exlen + cmlen;
        }
        let _ = total;
        Ok(Zip { data, concat, entries, by_name })
    }

    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.entries.iter().map(|e| e.name.as_str())
    }

    pub fn contains(&self, name: &str) -> bool {
        self.by_name.contains_key(name)
    }

    pub fn info(&self, name: &str) -> Option<&Entry> {
        self.by_name.get(name).map(|&i| &self.entries[i])
    }

    /// `zf.read(name)`.
    pub fn read(&self, name: &str) -> Result<Vec<u8>, String> {
        let e = self.info(name).ok_or_else(|| format!("There is no item named {name:?} in the archive"))?;
        let at = e.header_offset as i64 + self.concat;
        if at < 0 {
            return Err("Bad magic number for file header".into());
        }
        let at = at as usize;
        let h = self.data.get(at..at + 30).ok_or("Truncated file header")?;
        if &h[..4] != b"PK\x03\x04" {
            return Err("Bad magic number for file header".into());
        }
        let fnlen = u16_at(h, 26).unwrap() as usize;
        let exlen = u16_at(h, 28).unwrap() as usize;
        let flags = u16_at(h, 6).unwrap();
        let local_name = self.data.get(at + 30..at + 30 + fnlen).ok_or("Truncated file header")?;
        let local = if flags & 0x800 != 0 { String::from_utf8_lossy(local_name).into_owned() } else { local_name.iter().map(|&b| b as char).collect() };
        if local != e.name {
            return Err(format!("File name in directory {:?} and header {:?} differ.", e.name, local));
        }
        let start = at + 30 + fnlen + exlen;
        let raw = self.data.get(start..start.saturating_add(e.compress_size as usize)).ok_or("Truncated file data")?;
        let out = match e.method {
            0 => raw.to_vec(),
            8 => {
                let mut out = Vec::with_capacity((e.file_size as usize).min(1 << 30));
                flate2::read::DeflateDecoder::new(raw).read_to_end(&mut out).map_err(|x| format!("Error -3 while decompressing data: {x}"))?;
                out
            }
            m => return Err(format!("That compression method is not supported ({m})")),
        };
        if crc32fast::hash(&out) != e.crc {
            return Err("Bad CRC-32 for file".into());
        }
        Ok(out)
    }
}
