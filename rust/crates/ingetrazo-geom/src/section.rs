// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! `core.hlr.section_loops`: a section cut's plane∩triangle chords
//! chained into closed rings. The same node merging (a 3×3×3 neighbourhood
//! of `floor(p / tol)` cells, first match in dx→dy→dz order, nodes in
//! creation order) and the same walk (chords taken in index order); the
//! hash map only looks cells up.

use std::collections::HashMap;

type Cell = (i64, i64, i64);

/// ``(ring_start (R+1,), points (P,3))``: ring r is
/// ``points[ring_start[r]..ring_start[r+1]]``, in the order Python returns
/// them. ``tol`` as the Python computed it.
pub fn loops(chords_in: &[[[f64; 3]; 2]], tol: f64) -> (Vec<i64>, Vec<[f64; 3]>) {
    let inv = 1.0 / tol;
    let mut cells: HashMap<Cell, Vec<usize>> = HashMap::new();
    let mut nodes: Vec<[f64; 3]> = Vec::new();
    let mut node_of = |pt: [f64; 3], cells: &mut HashMap<Cell, Vec<usize>>| -> usize {
        let c = (
            (pt[0] * inv).floor() as i64,
            (pt[1] * inv).floor() as i64,
            (pt[2] * inv).floor() as i64,
        );
        for dx in -1..=1i64 {
            for dy in -1..=1i64 {
                for dz in -1..=1i64 {
                    if let Some(list) = cells.get(&(c.0 + dx, c.1 + dy, c.2 + dz)) {
                        for &i in list {
                            let q = nodes[i];
                            if (q[0] - pt[0]).abs() <= tol
                                && (q[1] - pt[1]).abs() <= tol
                                && (q[2] - pt[2]).abs() <= tol
                            {
                                return i;
                            }
                        }
                    }
                }
            }
        }
        nodes.push(pt);
        cells.entry(c).or_default().push(nodes.len() - 1);
        nodes.len() - 1
    };
    let mut adj: HashMap<usize, Vec<usize>> = HashMap::new();
    let mut chords: Vec<(usize, usize)> = Vec::new();
    for e in chords_in {
        let a = node_of(e[0], &mut cells);
        let b = node_of(e[1], &mut cells);
        if a == b {
            continue;
        }
        let idx = chords.len();
        chords.push((a, b));
        adj.entry(a).or_default().push(idx);
        adj.entry(b).or_default().push(idx);
    }
    let mut used = vec![false; chords.len()];
    let mut start_out = vec![0i64];
    let mut pts: Vec<[f64; 3]> = Vec::new();
    let empty: Vec<usize> = Vec::new();
    for start in 0..chords.len() {
        if used[start] {
            continue;
        }
        let (a0, b0) = chords[start];
        used[start] = true;
        let mut ring = vec![a0, b0];
        let (mut cur, mut prev) = (b0, start);
        let mut closed = false;
        loop {
            let mut nxt = None;
            for &c in adj.get(&cur).unwrap_or(&empty) {
                if c != prev && !used[c] {
                    nxt = Some(c);
                    break;
                }
            }
            let Some(n) = nxt else { break };
            used[n] = true;
            let (x, y) = chords[n];
            cur = if x == cur { y } else { x };
            prev = n;
            if cur == a0 {
                closed = true;
                break;
            }
            ring.push(cur);
        }
        if closed && ring.len() >= 3 {
            pts.extend(ring.iter().map(|&i| nodes[i]));
            start_out.push(pts.len() as i64);
        }
    }
    (start_out, pts)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_square_cut_closes_one_ring() {
        let p = |x: f64, y: f64| [x, y, 0.0];
        let c = [[p(0.0, 0.0), p(1.0, 0.0)], [p(1.0, 1.0), p(1.0, 0.0)],
                 [p(1.0, 1.0), p(0.0, 1.0)], [p(0.0, 1.0), p(0.0, 0.0)],
                 [p(5.0, 5.0), p(6.0, 5.0)]];               // an open chain
        let (start, pts) = loops(&c, 1e-7);
        assert_eq!(start, vec![0, 4]);
        assert_eq!(pts, vec![p(0.0, 0.0), p(1.0, 0.0), p(1.0, 1.0), p(0.0, 1.0)]);
    }
}
