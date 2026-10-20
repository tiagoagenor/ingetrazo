// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! A grid over the screen for the questions "which of these segments can be
//! within `lim` pixels of the cursor" (M4.9). A hover asks it of every group
//! hard edge of the model (hundreds of thousands of them) several times: with
//! the segments binned by the cells their screen box covers, only the few
//! under the cursor are looked at.
//!
//! It is a filter and nothing else: the candidates come back in index order,
//! and every one of them gets the same arithmetic as before, so a segment
//! within the limit is found exactly as in the full scan.

/// Pixels per cell.
const CELL: f64 = 32.0;
/// How far past the viewport segments are kept (a query must stay inside).
const MARGIN: f64 = 160.0;

/// Segments binned by the cells their box covers.
#[derive(Debug, Default)]
pub struct ScreenGrid {
    x0: f64,
    y0: f64,
    nx: usize,
    ny: usize,
    cells: Vec<Vec<u32>>,
}

impl ScreenGrid {
    /// A grid for a `w` x `h` viewport over `n` segments. `bbox(i)` is the
    /// screen box `(x0, y0, x1, y1)` of segment `i`, `None` for a segment
    /// that can never be near anything (behind the camera, not a number).
    pub fn build(n: usize, w: f64, h: f64, bbox: impl Fn(usize) -> Option<[f64; 4]>) -> ScreenGrid {
        let (x0, y0) = (-MARGIN, -MARGIN);
        let nx = (((w + 2.0 * MARGIN) / CELL).ceil() as usize).max(1);
        let ny = (((h + 2.0 * MARGIN) / CELL).ceil() as usize).max(1);
        let mut cells: Vec<Vec<u32>> = vec![Vec::new(); nx * ny];
        for i in 0..n {
            let Some([bx0, by0, bx1, by1]) = bbox(i) else { continue };
            if !(bx0.is_finite() && by0.is_finite() && bx1.is_finite() && by1.is_finite()) {
                continue;
            }
            // Outside the kept region altogether: no query reaches it.
            let (gx1, gy1) = (x0 + nx as f64 * CELL, y0 + ny as f64 * CELL);
            if bx1 < x0 || bx0 >= gx1 || by1 < y0 || by0 >= gy1 {
                continue;
            }
            let cx0 = (((bx0 - x0) / CELL).floor().max(0.0)) as usize;
            let cy0 = (((by0 - y0) / CELL).floor().max(0.0)) as usize;
            let cx1 = ((((bx1 - x0) / CELL).floor()) as usize).min(nx - 1);
            let cy1 = ((((by1 - y0) / CELL).floor()) as usize).min(ny - 1);
            for cy in cy0..=cy1 {
                for cx in cx0..=cx1 {
                    cells[cy * nx + cx].push(i as u32);
                }
            }
        }
        ScreenGrid { x0, y0, nx, ny, cells }
    }

    /// The segments whose box may come within `lim` of `(x, y)`, in index
    /// order and each once; `None` when the square around the cursor leaves
    /// the grid (the caller then looks at every segment).
    pub fn candidates(&self, x: f64, y: f64, lim: f64) -> Option<Vec<u32>> {
        let (qx0, qy0, qx1, qy1) = (x - lim, y - lim, x + lim, y + lim);
        let (gx1, gy1) = (self.x0 + self.nx as f64 * CELL, self.y0 + self.ny as f64 * CELL);
        if !(qx0 >= self.x0 && qy0 >= self.y0 && qx1 < gx1 && qy1 < gy1) {
            return None;
        }
        let cx0 = ((qx0 - self.x0) / CELL).floor() as usize;
        let cy0 = ((qy0 - self.y0) / CELL).floor() as usize;
        let cx1 = (((qx1 - self.x0) / CELL).floor() as usize).min(self.nx - 1);
        let cy1 = (((qy1 - self.y0) / CELL).floor() as usize).min(self.ny - 1);
        let mut out: Vec<u32> = Vec::new();
        for cy in cy0..=cy1 {
            for cx in cx0..=cx1 {
                out.extend_from_slice(&self.cells[cy * self.nx + cx]);
            }
        }
        out.sort_unstable();
        out.dedup();
        Some(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn candidates_are_a_superset_of_the_segments_within_the_limit_in_index_order() {
        // A scatter of segments, deterministic.
        let mut s = 12345u64;
        let mut rnd = |m: f64| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (s >> 33) as f64 / (1u64 << 31) as f64 * m
        };
        let segs: Vec<[f64; 4]> = (0..3000)
            .map(|_| {
                let (x, y) = (rnd(900.0) - 50.0, rnd(700.0) - 50.0);
                [x, y, x + rnd(200.0) - 100.0, y + rnd(200.0) - 100.0]
            })
            .collect();
        let boxed = |i: usize| {
            let g = segs[i];
            Some([g[0].min(g[2]), g[1].min(g[3]), g[0].max(g[2]), g[1].max(g[3])])
        };
        let grid = ScreenGrid::build(segs.len(), 800.0, 600.0, boxed);
        for _ in 0..200 {
            let (x, y, lim) = (rnd(800.0), rnd(600.0), 8.0 + rnd(40.0));
            let got = grid.candidates(x, y, lim).expect("inside the viewport");
            assert!(got.windows(2).all(|w| w[0] < w[1]), "sorted and unique");
            for i in 0..segs.len() {
                let b = boxed(i).unwrap();
                let near = b[0] - lim <= x && x <= b[2] + lim && b[1] - lim <= y && y <= b[3] + lim;
                if near {
                    assert!(got.contains(&(i as u32)), "segment {i} near ({x}, {y}) was dropped");
                }
            }
        }
        // A query that leaves the grid says so.
        assert!(grid.candidates(-1000.0, 10.0, 8.0).is_none());
    }
}
