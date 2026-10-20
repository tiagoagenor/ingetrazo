// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
// Port of openskp (MIT, Copyright (c) 2024-2026 Ahsan Mehmood): see NOTICE-openskp.md.
//! `openskp/errors.py`: a parse failure with where it happened.

use std::fmt;

#[derive(Clone, Debug, Default, PartialEq)]
pub struct SkpError {
    pub message: String,
    pub stage: Option<&'static str>,
    pub record_index: Option<usize>,
    pub total_records: Option<usize>,
    pub tag: Option<String>,
    pub offset: Option<usize>,
    pub definition_id: Option<i64>,
}

impl SkpError {
    pub fn new(message: impl Into<String>, stage: &'static str) -> SkpError {
        SkpError { message: message.into(), stage: Some(stage), ..Default::default() }
    }
}

impl fmt::Display for SkpError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.message)?;
        if let Some(s) = self.stage {
            write!(f, " | stage={s}")?;
        }
        if let (Some(i), Some(n)) = (self.record_index, self.total_records) {
            write!(f, " | record={i}/{n}")?;
        }
        if let Some(t) = &self.tag {
            write!(f, " | tag={t}")?;
        }
        if let Some(o) = self.offset {
            write!(f, " | offset=0x{o:X}")?;
        }
        if let Some(d) = self.definition_id {
            write!(f, " | definition_id={d}")?;
        }
        Ok(())
    }
}

impl std::error::Error for SkpError {}

pub type Result<T> = std::result::Result<T, SkpError>;
