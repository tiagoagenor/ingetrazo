// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! DXF import and export (docs/rust-rewrite-plan.md, M5.5): `formats/dxf_in.py`
//! and `formats/dxf_out.py`.
//!
//! The Python importer parses with ezdxf and flattens curves through its
//! `path` module; this crate carries its own ASCII reader ([`read`]) and the
//! few ezdxf algorithms the importer uses ([`geom`]), op for op, instead of
//! the `dxf` crate: only a port of ezdxf's arithmetic reproduces the
//! flattened points to the bit. The exporter is a plain text writer with no
//! library behind it ([`export`]).

pub mod entities;
pub mod export;
pub mod geom;
pub mod import;
pub mod read;
pub mod spline;
