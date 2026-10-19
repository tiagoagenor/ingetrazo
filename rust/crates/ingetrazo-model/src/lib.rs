// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! IngeTrazo's model (docs/rust-rewrite-plan.md, M2): the editable mesh
//! first; the scene, groups and components, annotations and the journal
//! undo follow.

pub mod mesh;

pub use mesh::{Edge, EdgeId, Face, FaceId, Mesh, Vertex, VertexId};
