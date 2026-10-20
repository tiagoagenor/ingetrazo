// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! The open document: its scene, where it came from, and its textures; and
//! writing it back (`MainWindow._do_save` -> `igz.save_scene`).

use std::path::{Path, PathBuf};
use std::sync::Arc;

use ingetrazo_formats::igz::{self, Zip};
use ingetrazo_formats::pyjson::Value;
use ingetrazo_model::igz_write::{save_scene, SaveOptions};
use ingetrazo_model::scene::{MeshId, Scene};
use ingetrazo_render::document::open_igz;

use crate::paths::{resource_path, InstallTextures};
use crate::viewport::Textures;

/// `core.version.__version__`, written into every document as `app_version`.
pub const APP_VERSION: &str = "0.5.8";

/// What the application holds open.
pub struct Document {
    pub scene: Arc<Scene>,
    /// `None` for a new, unsaved model.
    pub path: Option<PathBuf>,
    /// The file as it was read: the textures embedded in it are archive
    /// members that a save packs again.
    pub raw: Arc<Vec<u8>>,
}

impl Document {
    /// A new, empty model.
    pub fn empty() -> Document {
        Document { scene: Arc::new(Scene::new()), path: None, raw: Arc::new(Vec::new()) }
    }

    /// The window title: `IngeTrazo - name`, with the ` *` of unsaved work.
    pub fn title(&self) -> String {
        window_title(self.path.as_deref(), false)
    }
}

/// `MainWindow._update_title`.
pub fn window_title(path: Option<&Path>, dirty: bool) -> String {
    let name = path.and_then(Path::file_name).map(|n| n.to_string_lossy().into_owned());
    let marker = if dirty { " *" } else { "" };
    format!("IngeTrazo \u{2014} {}{marker}", name.as_deref().unwrap_or("Untitled"))
}

/// Read an .igz: the document and the textures embedded in it.
pub fn open_path(path: &Path) -> Result<(Document, Textures), String> {
    let raw = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let (scene, textures) = open_igz(&raw).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok((Document { scene: Arc::new(scene), path: Some(path.to_path_buf()), raw: Arc::new(raw) }, Box::new(InstallTextures(textures))))
}

/// What a save wrote.
#[derive(Debug)]
pub struct Written {
    pub bytes: Vec<u8>,
    /// Textures packed into the file / images that could not be read.
    pub embedded: usize,
    pub missing: usize,
}

/// `MainWindow._camera_dict`: the pose the document keeps.
pub fn camera_block(cam: &ingetrazo_render::Camera) -> Option<Value> {
    let finite = cam.target.0.is_finite() && cam.target.1.is_finite() && cam.target.2.is_finite() && cam.distance.is_finite() && cam.yaw.is_finite() && cam.pitch.is_finite();
    if !finite {
        return None;
    }
    let f = |x: f64| Value::Float(x);
    Some(Value::Object(vec![
        ("target".into(), Value::Array(vec![f(cam.target.0 as f64), f(cam.target.1 as f64), f(cam.target.2 as f64)])),
        ("distance".into(), f(cam.distance)),
        ("yaw".into(), f(cam.yaw)),
        ("pitch".into(), f(cam.pitch)),
        ("fov_deg".into(), f(cam.fov_deg)),
        ("perspective".into(), Value::Bool(cam.perspective)),
        ("two_point".into(), Value::Bool(cam.two_point)),
    ]))
}

/// `igz.save_scene` to bytes: the file the document is. `raw` is the file it
/// was opened from (its archive holds the embedded textures); a texture that
/// is a path on disk is read from there.
pub fn write_bytes(scene: &mut Scene, raw: &[u8]) -> Result<Written, String> {
    // Saving reads every dimension (`read_dimensions`).
    scene.read_dimensions(MeshId(0));
    let zip = if igz::is_container(raw) { Zip::open(raw).ok() } else { None };
    let reader = |name: &str| zip.as_ref().and_then(|z| z.read(name).ok()).or_else(|| std::fs::read(name).ok()).or_else(|| std::fs::read(resource_path(name)).ok());
    let opts = SaveOptions::new(APP_VERSION, &reader);
    let saved = save_scene(scene, &opts).map_err(|e| format!("{e:?}"))?;
    Ok(Written { bytes: saved.bytes, embedded: saved.embedded, missing: saved.missing })
}

/// Write `bytes` to `path` through a temporary file next to it, so a save
/// that dies halfway leaves the previous file whole; the previous version is
/// kept as `name.igz.bak` first (Preferences > General > backup).
pub fn write_file(path: &Path, bytes: &[u8]) -> Result<(), String> {
    if path.is_file() {
        let mut bak = path.as_os_str().to_owned();
        bak.push(".bak");
        // A failed backup does not stop the save (Python says so in the status bar).
        let _ = std::fs::copy(path, PathBuf::from(bak));
    }
    let mut tmp = path.as_os_str().to_owned();
    tmp.push(".tmp");
    let tmp = PathBuf::from(tmp);
    std::fs::write(&tmp, bytes).map_err(|e| format!("{}: {e}", path.display()))?;
    std::fs::rename(&tmp, path).map_err(|e| {
        let _ = std::fs::remove_file(&tmp);
        format!("{}: {e}", path.display())
    })
}
