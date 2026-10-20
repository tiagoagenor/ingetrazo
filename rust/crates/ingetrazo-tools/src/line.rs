// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! The Line tool (`tools/line.py`): click points to draw edges; closing a
//! loop makes a face.
//!
//! The first click sets the start of a fresh chain; each next click commits a
//! segment and chains into the next; snapping to the chain's first point
//! (kind `close`) ends the polygon; Esc cancels the chain; Ctrl+Z steps the
//! chain back one vertex; the value box takes a length along the rubber band
//! (negative = the other way) or a `dx;dy;dz` delta.

use ingetrazo_math::V3;
use ingetrazo_ops::edits::build_add_edge;
use ingetrazo_snap::{SnapKind, ToolState};

use crate::base::*;
use crate::host::Core;
use crate::vcb::VcbValue;

pub struct LineTool {
    pub start_point: Option<V3>,
    pub hover_point: Option<V3>,
    pub chain_first_point: Option<V3>,
    pub chain_vertices: Vec<V3>,
    pub work_plane: Option<(V3, V3)>,
}

impl Default for LineTool {
    fn default() -> Self {
        LineTool::new()
    }
}

impl LineTool {
    pub fn new() -> LineTool {
        LineTool { start_point: None, hover_point: None, chain_first_point: None, chain_vertices: Vec::new(), work_plane: None }
    }

    fn reset(&mut self) {
        self.start_point = None;
        self.chain_first_point = None;
        self.chain_vertices.clear();
        self.work_plane = None;
    }

    /// `_commit_edge`: one new edge as one command, welded, split and faced.
    fn commit_edge(&self, core: &mut Core, start: V3, end: V3) {
        let eye = Some(core.eye());
        let cmd = build_add_edge(&core.scene, &core.cx, start, end, true, eye);
        core.exec(cmd);
    }
}

impl DrawTool for LineTool {
    fn name(&self) -> &'static str {
        "Line"
    }

    fn snap_state(&self, _core: &Core) -> ToolState {
        ToolState {
            uses_snap: true,
            is_plane_lock: false,
            magnetic_axis_deg: Some(MAGNET_DEG),
            screen_axis_px: Some(MAGNET_PX),
            magnet_off_plane: true,
            axis_source: true,
            axis_source_empty_only: true,
            start_point: self.start_point,
            chain_first_point: self.chain_first_point,
            // `chain_first_point is start_point`: the same object while the
            // chain has its first vertex only.
            chain_first_is_start: self.chain_first_point.is_some() && self.start_point.is_some() && self.chain_vertices.len() == 1,
            work_plane: self.work_plane,
            ..ToolState::default()
        }
    }

    fn start_point(&self) -> Option<V3> {
        self.start_point
    }
    fn work_plane(&self) -> Option<(V3, V3)> {
        self.work_plane
    }
    fn set_work_plane(&mut self, plane: Option<(V3, V3)>) {
        self.work_plane = plane;
    }
    fn vcb_label(&self) -> String {
        "Length".into()
    }

    fn on_activate(&mut self, _core: &mut Core) {
        self.reset();
    }

    fn on_deactivate(&mut self, _core: &mut Core) {
        self.reset();
        self.hover_point = None;
    }

    fn on_click(&mut self, core: &mut Core, ctx: &Ctx) {
        let clicked = ctx.world;
        let Some(start) = self.start_point else {
            self.start_point = Some(clicked);
            self.chain_first_point = Some(clicked);
            self.chain_vertices = vec![clicked];
            return;
        };
        if (clicked.sub(start).length(core.h()) as f64) < 1e-9 {
            // The start itself: nothing to draw (issue #34).
            return;
        }
        self.commit_edge(core, start, clicked);
        if ctx.snap.kind == SnapKind::Close {
            self.reset();
        } else {
            self.chain_vertices.push(clicked);
            self.start_point = Some(clicked);
        }
    }

    fn on_hover(&mut self, _core: &mut Core, ctx: &Ctx) {
        self.hover_point = Some(ctx.world);
    }

    fn on_cancel(&mut self, _core: &mut Core) {
        self.reset();
    }

    /// Ctrl+Z in the middle of a chain: the segment just drawn is undone and
    /// the chain steps back to its previous vertex (#175).
    fn on_undo(&mut self, core: &mut Core) -> bool {
        if self.start_point.is_none() {
            return false;
        }
        if self.chain_vertices.len() > 1 {
            if !core.history_undo() {
                self.reset();
                return true;
            }
            self.chain_vertices.pop();
            self.start_point = self.chain_vertices.last().copied();
            if self.chain_vertices.len() == 1 {
                self.chain_first_point = self.start_point;
            }
        } else {
            self.reset();
        }
        true
    }

    fn on_value(&mut self, core: &mut Core, value: &VcbValue) -> bool {
        let Some(start) = self.start_point else { return false };
        let h = core.h();
        let new_endpoint = match value {
            VcbValue::Triple(dx, dy, dz) => V3(((start.0 as f64) + dx) as f32, ((start.1 as f64) + dy) as f32, ((start.2 as f64) + dz) as f32),
            VcbValue::Pair(..) => return false, // a rectangle's W x H, not a line delta
            VcbValue::Num(value) => {
                // A negative length runs AWAY from the cursor (issue #58).
                let Some(hover) = self.hover_point else { return false };
                if value.abs() < 1e-9 {
                    return false;
                }
                let delta = hover.sub(start);
                if (delta.length(h) as f64) < 1e-9 {
                    return false;
                }
                let direction = delta.normalized(h);
                start.add(direction.mul(*value as f32))
            }
            _ => return false,
        };
        self.commit_edge(core, start, new_endpoint);
        self.chain_vertices.push(new_endpoint);
        self.start_point = Some(new_endpoint);
        self.hover_point = Some(new_endpoint);
        true
    }

    fn rubber_band_lines(&self, _core: &Core) -> Vec<(V3, V3)> {
        match (self.start_point, self.hover_point) {
            (Some(s), Some(h)) => vec![(s, h)],
            _ => Vec::new(),
        }
    }

    fn fields(&self) -> Vec<(&'static str, Field)> {
        vec![
            ("start_point", Field::point(self.start_point)),
            ("hover_point", Field::point(self.hover_point)),
            ("chain_first_point", Field::point(self.chain_first_point)),
            ("chain_vertices", Field::List(self.chain_vertices.iter().map(|p| Field::P(*p)).collect())),
            ("work_plane", Field::plane(self.work_plane)),
        ]
    }
}
