// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! The Freehand tool (`tools/freehand.py`): press and DRAG to sketch. The
//! stroke samples the cursor, gets simplified (Ramer-Douglas-Peucker in
//! screen space, so zoom level = detail level), and lands as ONE curve
//! entity. A stroke that ends back at its start closes, and a closed flat
//! stroke becomes a face through the shared curve pipeline.

use ingetrazo_math::V3;
use ingetrazo_snap::ToolState;

use crate::arc::commit_arc;
use crate::base::*;
use crate::host::Core;

/// The cursor must travel this far to add a sample.
const MIN_SAMPLE_PX: f64 = 3.0;
/// RDP tolerance, in screen pixels.
const SIMPLIFY_PX: f64 = 1.6;
/// Release near the start: the stroke closes.
const CLOSE_PX: f64 = 10.0;

/// Mark the screen-space RDP survivors between `idx0` and `idx1`.
fn rdp(idx0: usize, idx1: usize, spts: &[(f64, f64)], keep: &mut [bool]) {
    let (ax, ay) = spts[idx0];
    let (bx, by) = spts[idx1];
    let (dx, dy) = (bx - ax, by - ay);
    let l2 = dx * dx + dy * dy;
    let mut worst: Option<usize> = None;
    let mut worst_d = SIMPLIFY_PX;
    for i in idx0 + 1..idx1 {
        let (px, py) = spts[i];
        let d = if l2 < 1e-12 {
            ((px - ax).powi(2) + (py - ay).powi(2)).powf(0.5)
        } else {
            let t = ingetrazo_math::pyfloat::py_max2(0.0, ingetrazo_math::pyfloat::py_min2(1.0, ((px - ax) * dx + (py - ay) * dy) / l2));
            ((px - (ax + t * dx)).powi(2) + (py - (ay + t * dy)).powi(2)).powf(0.5)
        };
        if d > worst_d {
            worst = Some(i);
            worst_d = d;
        }
    }
    if let Some(w) = worst {
        keep[w] = true;
        rdp(idx0, w, spts, keep);
        rdp(w, idx1, spts, keep);
    }
}

pub struct FreehandTool {
    pub start_point: Option<V3>,
    work_plane: Option<(V3, V3)>,
    world: Vec<V3>,
    screen: Vec<(f64, f64)>,
    stroke: bool,
}

impl Default for FreehandTool {
    fn default() -> Self {
        FreehandTool::new()
    }
}

impl FreehandTool {
    pub fn new() -> FreehandTool {
        FreehandTool { start_point: None, work_plane: None, world: Vec::new(), screen: Vec::new(), stroke: false }
    }

    fn reset(&mut self) {
        self.stroke = false;
        self.start_point = None;
        self.work_plane = None;
        self.world.clear();
        self.screen.clear();
    }
}

impl DrawTool for FreehandTool {
    fn name(&self) -> &'static str {
        "Freehand"
    }

    /// The stroke follows the hand, not the magnets.
    fn uses_snap(&self) -> bool {
        false
    }

    fn snap_state(&self, _core: &Core) -> ToolState {
        ToolState { uses_snap: false, is_plane_lock: false, start_point: self.start_point, work_plane: self.work_plane, ..ToolState::default() }
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
    fn busy(&self) -> bool {
        self.start_point.is_some() || self.stroke
    }
    fn vcb_label(&self) -> String {
        // `vcb_label = None`: the box is hidden.
        String::new()
    }

    fn on_activate(&mut self, _core: &mut Core) {
        self.reset();
    }

    fn on_deactivate(&mut self, _core: &mut Core) {
        self.reset();
    }

    fn on_click(&mut self, _core: &mut Core, ctx: &Ctx) {
        self.stroke = true;
        self.start_point = Some(ctx.world);
        self.world = vec![ctx.world];
        self.screen = vec![ctx.screen];
    }

    fn on_hover(&mut self, _core: &mut Core, ctx: &Ctx) {
        if !self.stroke {
            return;
        }
        let (sx, sy) = ctx.screen;
        let (lx, ly) = *self.screen.last().expect("a sample");
        if (sx - lx).powi(2) + (sy - ly).powi(2) < MIN_SAMPLE_PX.powi(2) {
            return;
        }
        self.world.push(ctx.world);
        self.screen.push((sx, sy));
    }

    fn on_release(&mut self, core: &mut Core) {
        if !self.stroke {
            return;
        }
        let world = std::mem::take(&mut self.world);
        let screen = std::mem::take(&mut self.screen);
        self.reset();
        if world.len() < 3 {
            return;
        }
        // Close the loop when the hand came back to the start.
        let (sx, sy) = screen[0];
        let (ex, ey) = screen[screen.len() - 1];
        let closed = (ex - sx).powi(2) + (ey - sy).powi(2) <= CLOSE_PX.powi(2);
        let mut keep = vec![false; world.len()];
        keep[0] = true;
        let last = keep.len() - 1;
        keep[last] = true;
        rdp(0, world.len() - 1, &screen, &mut keep);
        let mut pts: Vec<V3> = world.iter().zip(&keep).filter(|(_, k)| **k).map(|(p, _)| *p).collect();
        if closed && pts.len() >= 3 {
            let first = pts[0];
            let n = pts.len();
            pts[n - 1] = first; // weld the seam exactly
        }
        if pts.len() >= 2 {
            commit_arc(core, &pts, None, None);
        }
    }

    fn on_cancel(&mut self, _core: &mut Core) {
        self.reset();
    }

    fn rubber_band_lines(&self, _core: &Core) -> Vec<(V3, V3)> {
        if self.world.len() < 2 {
            return Vec::new();
        }
        (0..self.world.len() - 1).map(|i| (self.world[i], self.world[i + 1])).collect()
    }

    fn fields(&self) -> Vec<(&'static str, Field)> {
        vec![
            ("start_point", Field::point(self.start_point)),
            ("_world", if self.world.is_empty() { Field::List(Vec::new()) } else { Field::List(self.world.iter().map(|p| Field::P(*p)).collect()) }),
            ("_screen", Field::List(self.screen.iter().map(|(a, b)| Field::List(vec![Field::Float(*a), Field::Float(*b)])).collect())),
            ("_stroke", Field::Bool(self.stroke)),
        ]
    }
}
