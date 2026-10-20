// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! IngeTrazo's wgpu renderer (docs/rust-rewrite-plan.md, M4.1): a port of
//! what `views/viewport.py` draws with OpenGL.
//!
//! * [`camera`]: `OrbitCamera` and its matrices.
//! * [`style`]: display styles (`core/style.py`).
//! * [`bake`]: meshes to vertex data (faces, edges, texture coordinates).
//! * [`plan`]: the placements of a scene (component instancing).
//! * [`renderer`]: GPU resources and the frame; the frame is first a
//!   [`frame::FramePlan`] (testable data), then encoded.
//! * [`offscreen`]: headless rendering to an image (tests, PNG export).
//! * [`blit`]: copying the result into a host's pass (egui paint callbacks).
//!
//! The crate does not depend on any UI toolkit.

// Pixel loops read better as `chunks_exact(4)` than as `as_chunks::<4>()`.
#![allow(clippy::chunks_exact_to_as_chunks)]

pub mod bake;
pub mod billboard;
pub mod blit;
pub mod camera;
pub mod document;
pub mod frame;
pub mod offscreen;
pub mod options;
pub mod overlay;
pub mod plan;
pub mod renderer;
pub mod section;
pub mod style;
pub mod textures;

pub use camera::Camera;
pub use options::{ExportLines, FrameStats, Hover, OverlayLine, RenderOptions, Selection};
pub use plan::{EditRest, GroupPath};
pub use renderer::{Renderer, RendererConfig};
pub use style::{FaceMode, Style};
pub use textures::{NoTextures, TextureImage, TextureSource};
