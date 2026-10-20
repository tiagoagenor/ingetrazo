// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
// Port of openskp (MIT, Copyright (c) 2024-2026 Ahsan Mehmood): see NOTICE-openskp.md.
//! The SketchUp `.skp` reader (docs/rust-rewrite-plan.md, M5.7): a port of
//! the openskp parser IngeTrazo pinned, with the two fixes it patched in at
//! run time (`formats/openskp_compat.py`).
//!
//! Two containers are read, as openskp reads them:
//!
//! * **VFF** (SketchUp 2021 and later): a zip with `model.dat` (tag-length-value
//!   records: definitions, vertices, edges, faces and loops, instances,
//!   layers, materials, scenes, dimensions), the material and style XML, and
//!   `meta/meta.dat` ([`vff`]).
//! * **Classic MFC** (2013-2020): one uncompressed `CArchive` object stream
//!   ([`legacy`]).
//!
//! Both end in the same [`Model`]. Writing a `.skp` is not ported (the app
//! no longer exports one).

// A parse error is rare and carries where it happened; boxing it would only
// add noise to every `?`.
#![allow(clippy::result_large_err)]

pub mod dump;
pub mod error;
pub mod legacy;
pub mod model;
pub mod pystr;
pub mod raw;
pub mod tlv;
pub mod vff;
pub mod xml;
pub mod zip;

pub use error::{Result, SkpError};
pub use model::*;

/// `_core.full_parse` over the bytes of a file, then `SkpFile.parse`'s typed
/// model (face tags resolved to names).
pub fn parse_bytes(data: &[u8]) -> Result<Model> {
    let header = &data[..data.len().min(512)];
    if !header.starts_with(&[0xFF, 0xFE, 0xFF, 0x0E]) {
        return Err(SkpError::new("Not a valid SketchUp file (bad header magic)", "header"));
    }
    let raw = if legacy::is_legacy(header) { legacy::full_parse_legacy(data)? } else { vff::full_parse(data)? };
    Ok(raw::assemble(raw))
}

/// `SkpFile.open(path).parse()`.
pub fn parse_file(path: &std::path::Path) -> Result<Model> {
    if !path.exists() {
        return Err(SkpError::new(format!("File not found: {}", path.display()), "open"));
    }
    if path.extension().map(|e| e.to_string_lossy().to_lowercase()) != Some("skp".to_string()) {
        return Err(SkpError::new(format!("Expected a .skp file, got: {}", path.extension().map(|e| format!(".{}", e.to_string_lossy())).unwrap_or_default()), "open"));
    }
    let data = std::fs::read(path).map_err(|e| SkpError::new(e.to_string(), "open"))?;
    parse_bytes(&data)
}
