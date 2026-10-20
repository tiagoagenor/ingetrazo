// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! IngeTrazo's snap and inference engine (M3.11), with no Qt and no
//! drawing: given the scene, a camera and a cursor pixel (plus the tool's
//! anchor points and axis locks) it answers what the cursor snaps to —
//! endpoint, midpoint, on-edge, on-face, centre, intersection, the axis
//! inferences, guide lines — the inferred 3D point and the tooltip.
//!
//! * [`compute`]: `core/snap.py::compute_snap` and its helpers, the rules.
//! * [`camera`]: `OrbitCamera` and the projections `Viewport` offers them.
//! * [`world`]: the model flattened the way the viewport's pick index and
//!   `_snap_scene` gather it.
//! * [`pick`]: occlusion and the face under the cursor.
//! * [`workplane`]: the drawing plane and the cursor's world point.
//! * [`engine`]: `core/inference.py::InferenceEngine.snap`, the snap scene,
//!   the axis-source cue and the tip.
//! * [`hover`]: what the pointer is over (`InferenceEngine.acquire`) and the
//!   encouraged points.
//! * [`circle`]: the circle fit behind the Center and Arc-midpoint
//!   inferences.
//!
//! Every piece is held to the Python by `tests/snap_parity.rs` (fixtures from
//! `scripts/golden/make_snap_fixtures.py`, the real `Viewport` run headless).
//!
//! Not ported, because nothing in the fixtures reaches it: an active section
//! cut (`_active_cut`), layers, face-me figures, reference-image borders,
//! survey points, the selection box's corners, the screen grid of
//! `_gedge_dist` (past 20 000 group edges), the lazy placements of huge
//! models, the extension snap providers and the camera's two-point
//! perspective. The circle fit is not LAPACK (see [`circle`]).

pub mod camera;
pub mod circle;
pub mod compute;
pub mod engine;
pub mod hover;
pub mod pick;
pub mod types;
pub mod workplane;
pub mod world;

pub use camera::{OrbitCamera, View};
pub use compute::{compute_snap, LinearMode, RefMode, SnapParams};
pub use engine::{snap, InferenceState, Settings, SnapInput, SnapTrace};
pub use types::*;
pub use workplane::ToolState;
pub use world::World;
