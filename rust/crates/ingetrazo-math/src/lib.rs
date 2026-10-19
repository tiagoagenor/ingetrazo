// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! IngeTrazo's numbers: Qt's float32 vector arithmetic and Python's float
//! built-ins, reproduced exactly so the Rust app draws what the Python one
//! did. No dependencies.

pub mod pyfloat;
pub mod qt;

pub use qt::{length, normalized, Hypot, V3};
