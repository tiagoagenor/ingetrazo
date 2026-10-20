// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! What a caller hands the renderer besides the scene and the camera.

use ingetrazo_math::V3;
use ingetrazo_model::{EdgeId, FaceId};

use crate::plan::{EditRest, GroupPath};
use crate::style::{Rgb, Style};

/// Things to highlight. Elements are named by the placement that owns their
/// mesh (empty path = the loose mesh) and their id inside that mesh.
#[derive(Clone, Debug, Default)]
pub struct Selection {
    pub edges: Vec<(GroupPath, EdgeId)>,
    pub faces: Vec<(GroupPath, FaceId)>,
    /// Whole groups: drawn as the box that wraps them.
    pub groups: Vec<GroupPath>,
}

#[derive(Clone, Debug, Default)]
pub struct Hover {
    pub edge: Option<(GroupPath, EdgeId)>,
    pub face: Option<(GroupPath, FaceId)>,
}

/// A segment drawn on top of the scene (tool previews, rubber bands).
#[derive(Clone, Copy, Debug)]
pub struct OverlayLine {
    pub a: V3,
    pub b: V3,
    pub color: Rgb,
    /// Depth tested like the model, or floating on top.
    pub depth_test: bool,
}

/// Line pen widths in pixels for exports (`_export_edge_px`,
/// `_export_profile_px`): a GL line is one pixel whatever the DPI, so the
/// pass is redrawn at sub-pixel offsets to reach the width.
#[derive(Clone, Copy, Debug)]
pub struct ExportLines {
    pub edge_px: u32,
    pub profile_px: u32,
}

#[derive(Clone, Debug)]
pub struct RenderOptions {
    /// The style this frame draws with; `None` = the scene's own
    /// `display_style`.
    pub style: Option<Style>,
    /// Axes and guides: off on plan sheets and styled composer frames.
    pub overlays: bool,
    /// The sky and ground backdrop (also needs `style.sky`).
    pub backdrop: bool,
    /// Face-me billboards (2D people, trees): off on plan sheets, where a
    /// coloured cutout person would give the raster away.
    pub billboards: bool,
    /// `Some` while rendering an export: no selection cues, thicker lines.
    pub export: Option<ExportLines>,
    pub selection: Selection,
    pub hover: Hover,
    /// The drawing axes' origin.
    pub axes_origin: V3,
    /// Indexes into `scene.guides` drawn as selected.
    pub guides_selected: Vec<usize>,
    /// The group open for editing and how the context is treated.
    pub edit_path: Option<GroupPath>,
    pub edit_rest: EditRest,
    pub overlay_lines: Vec<OverlayLine>,
    /// Height in logical pixels of the viewport the picture is a rendering
    /// of: the thick section-cut lines are 2.5 logical pixels wide, so an
    /// export larger than the widget gets proportionally thicker lines.
    /// `None` = the target's own height.
    pub logical_height: Option<f32>,
}

impl Default for RenderOptions {
    fn default() -> RenderOptions {
        RenderOptions {
            style: None,
            overlays: true,
            backdrop: true,
            billboards: true,
            export: None,
            selection: Selection::default(),
            hover: Hover::default(),
            axes_origin: V3(0.0, 0.0, 0.0),
            guides_selected: Vec::new(),
            edit_path: None,
            edit_rest: EditRest::Fade,
            overlay_lines: Vec::new(),
            logical_height: None,
        }
    }
}

/// What a frame did, for the status bar and the benchmarks.
#[derive(Clone, Copy, Debug, Default)]
pub struct FrameStats {
    pub draw_calls: usize,
    pub triangles: u64,
    pub instances: usize,
    pub prototypes: usize,
    /// Prototypes (re)baked and uploaded by this frame's sync.
    pub rebuilt: usize,
    pub uploaded_bytes: u64,
}
