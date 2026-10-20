// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! The frame as data: the list of draw commands a frame is made of, built on
//! the CPU before anything is encoded. Tests read the list to check what a
//! style draws, in which order and with which state; the renderer just walks
//! it.

use bytemuck::{Pod, Zeroable};

use crate::bake::{EdgeVertex, FaceVertex};
use crate::overlay::SkyVertex;

/// The per-draw uniform block (see `shader.wgsl`).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct ParamsData {
    pub mvp: [f32; 16],
    pub clip: [f32; 4],
    pub color: [f32; 4],
    pub back_color: [f32; 4],
    pub fade_color: [f32; 4],
    pub v0: [f32; 4],
    pub v1: [f32; 4],
    pub v2: [f32; 4],
}

/// How the fragment shader colours faces.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ColMode {
    /// The vertex colour (the shaded material colour).
    Vertex = 0,
    /// The texture times the vertex shade.
    Texture = 1,
    /// One flat colour in front, another behind.
    Flat = 2,
    /// Flat colour with the back tint on unpainted backs (Shaded textures).
    FlatTint = 3,
}

/// Screen-space patterns (`u_stipple`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stipple {
    None = 0,
    /// Hidden faces: a 1 px grid every 4 px.
    Grid = 1,
    /// Hidden edges: dotted.
    Dotted = 2,
    /// Selected faces: 2x2 px dots every 6 px.
    Dots = 3,
    /// Back Edges: dashes along the line.
    Dashed = 4,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PipeKind {
    Face,
    Line,
    Soft,
    SkyTri,
    SkyLine,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DepthCmp {
    LessEqual,
    Greater,
    Always,
}

/// Everything a pipeline is created from.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PipeKey {
    pub kind: PipeKind,
    /// Alpha blending (SRC_ALPHA, ONE_MINUS_SRC_ALPHA).
    pub blend: bool,
    pub depth_write: bool,
    pub cmp: DepthCmp,
    /// Polygon offset (1, 1): faces sit a hair behind their edges.
    pub bias: u8,
    pub color_write: bool,
}

impl PipeKey {
    pub fn face() -> PipeKey {
        PipeKey { kind: PipeKind::Face, blend: false, depth_write: true, cmp: DepthCmp::LessEqual, bias: 1, color_write: true }
    }
    pub fn line() -> PipeKey {
        PipeKey { kind: PipeKind::Line, blend: false, depth_write: true, cmp: DepthCmp::LessEqual, bias: 0, color_write: true }
    }
}

/// The vertex data a draw reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Geo {
    Faces(usize),
    Hard(usize),
    Soft(usize),
    GhostEdges(usize),
    DynFaces,
    DynEdges,
    SkyTris,
    SkyLine,
}

#[derive(Clone, Debug, PartialEq)]
pub struct DrawCmd {
    pub pipe: PipeKey,
    /// Index into [`FramePlan::params`].
    pub slot: u32,
    /// The texture bound (index into the renderer's texture table).
    pub tex: Option<usize>,
    pub geo: Geo,
    pub verts: (u32, u32),
    /// Instances: the proto's slice of the instance buffers, or `None` for
    /// the single identity instance.
    pub insts: Option<(u32, u32)>,
}

/// One render pass. The first clears colour and depth; every later one keeps
/// the colour and clears the depth again (X-ray's selection cues, which show
/// through the faces).
#[derive(Clone, Debug, Default)]
pub struct PassPlan {
    pub cmds: Vec<DrawCmd>,
}

#[derive(Clone, Debug, Default)]
pub struct FramePlan {
    pub clear: [f32; 3],
    pub params: Vec<ParamsData>,
    pub passes: Vec<PassPlan>,
    pub dyn_faces: Vec<FaceVertex>,
    pub dyn_edges: Vec<EdgeVertex>,
    pub sky_tris: Vec<SkyVertex>,
    pub sky_line: Vec<SkyVertex>,
}

impl FramePlan {
    pub fn draw_count(&self) -> usize {
        self.passes.iter().map(|p| p.cmds.len()).sum()
    }

    /// Triangles submitted by face draws (instances counted).
    pub fn triangle_count(&self) -> u64 {
        self.passes
            .iter()
            .flat_map(|p| p.cmds.iter())
            .filter(|c| matches!(c.geo, Geo::Faces(_) | Geo::DynFaces) && c.pipe.color_write)
            .map(|c| (c.verts.1 - c.verts.0) as u64 / 3 * c.insts.map_or(1, |i| i.1 as u64))
            .sum()
    }

    /// The commands of every pass, in draw order.
    pub fn cmds(&self) -> impl Iterator<Item = &DrawCmd> {
        self.passes.iter().flat_map(|p| p.cmds.iter())
    }
}
