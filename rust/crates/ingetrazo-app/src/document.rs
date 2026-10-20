// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! The open document: its scene, where it came from, and its textures.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use ingetrazo_model::scene::Scene;
use ingetrazo_render::document::open_igz;

use crate::viewport::Textures;

/// What the application holds open.
pub struct Document {
    pub scene: Arc<Scene>,
    /// `None` for a new, unsaved model.
    pub path: Option<PathBuf>,
}

impl Document {
    /// A new, empty model.
    pub fn empty() -> Document {
        Document { scene: Arc::new(Scene::new()), path: None }
    }

    /// The window title: the file name, or "Untitled".
    pub fn title(&self) -> String {
        let name = self.path.as_deref().and_then(Path::file_name).map(|n| n.to_string_lossy().into_owned());
        format!("{} - IngeTrazo", name.as_deref().unwrap_or("Untitled"))
    }
}

/// Read an .igz: the document and the textures embedded in it.
pub fn open_path(path: &Path) -> Result<(Document, Textures), String> {
    let raw = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let (scene, textures) = open_igz(&raw).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok((Document { scene: Arc::new(scene), path: Some(path.to_path_buf()) }, Box::new(textures)))
}
