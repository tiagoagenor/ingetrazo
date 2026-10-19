// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! `core.lazy_mesh.plan` after the weld — what `Mesh.add_edges_welded` and
//! `Mesh.add_faces_welded` make on the welded vertices — with hash maps in
//! place of the NumPy unique/sort passes. The order is the NumPy one: every
//! "unique" keeps first-occurrence order, `edge.faces` lists each edge's
//! faces in global ring-segment order. The maps only look up; no map
//! iteration order reaches a result.

use std::collections::HashMap;

/// The topology of a planned mesh. `curve`/`layer` are -1 for None.
pub struct Topology {
    pub edges: Vec<[i64; 2]>,
    pub soft: Vec<bool>,
    pub hidden: Vec<bool>,
    pub curve: Vec<i64>,
    pub layer: Vec<i64>,
    pub corners: Vec<i64>,
    pub ef_start: Vec<i64>,
    pub ef_face: Vec<i64>,
}

#[inline]
fn key(a: i64, b: i64) -> (i64, i64) {
    if a <= b { (a, b) } else { (b, a) }
}

/// ``inverse``: point → vertex (the weld). ``edges``: point index pairs.
/// ``flags``/``curves``/``layers``: per input edge, or None — flags None
/// ignores the other two, as the Python does. ``rings``: every ring corner
/// (point index); ``sizes``: corners per ring (all > 0); ``counts``: rings
/// per face (all > 0). The caller checks the two "> 0".
pub fn topology(
    inverse: &[i64],
    edges: &[[i64; 2]],
    flags: Option<&[i64]>,
    curves: Option<&[i64]>,
    layers: Option<&[i64]>,
    rings: &[i64],
    sizes: &[i64],
    counts: &[i64],
) -> Topology {
    // add_edges_welded: skip welded-shut pairs, first occurrence wins the
    // orientation, flags accumulate (soft/hidden any, curve/layer last).
    let mut index: HashMap<(i64, i64), usize> = HashMap::with_capacity(edges.len());
    let mut out_edges: Vec<[i64; 2]> = Vec::new();
    let mut soft: Vec<bool> = Vec::new();
    let mut hidden: Vec<bool> = Vec::new();
    let mut curve: Vec<i64> = Vec::new();
    let mut layer: Vec<i64> = Vec::new();
    for (j, e) in edges.iter().enumerate() {
        let a = inverse[e[0] as usize];
        let b = inverse[e[1] as usize];
        if a == b {
            continue;
        }
        let k = *index.entry(key(a, b)).or_insert_with(|| {
            out_edges.push([a, b]);
            soft.push(false);
            hidden.push(false);
            curve.push(-1);
            layer.push(-1);
            out_edges.len() - 1
        });
        if let Some(f) = flags {
            let f = f[j];
            soft[k] |= f & 1 != 0;
            hidden[k] |= f & 2 != 0;
            if let Some(c) = curves {
                if c[j] >= 0 {
                    curve[k] = c[j];
                }
            }
            if let Some(l) = layers {
                if l[j] >= 0 {
                    layer[k] = l[j];
                }
            }
        }
    }
    let ne1 = out_edges.len();
    let corners: Vec<i64> = rings.iter().map(|&p| inverse[p as usize]).collect();
    if counts.is_empty() {
        return Topology {
            edges: out_edges,
            soft,
            hidden,
            curve,
            layer,
            corners,
            ef_start: vec![0; ne1 + 1],
            ef_face: Vec::new(),
        };
    }
    // add_faces_welded: every ring segment; the boundary edges it lacks
    // are appended in first-occurrence order and orientation.
    let mut seg_edge: Vec<usize> = Vec::with_capacity(corners.len());
    let mut seg_face: Vec<i64> = Vec::with_capacity(corners.len());
    let mut ring = 0usize;
    let mut start = 0usize;
    for (face, &nr) in counts.iter().enumerate() {
        for _ in 0..nr {
            let n = sizes[ring] as usize;
            for i in 0..n {
                let sa = corners[start + i];
                let sb = corners[start + (i + 1) % n];
                let e = *index.entry(key(sa, sb)).or_insert_with(|| {
                    out_edges.push([sa, sb]);
                    soft.push(false);
                    hidden.push(false);
                    curve.push(-1);
                    layer.push(-1);
                    out_edges.len() - 1
                });
                seg_edge.push(e);
                seg_face.push(face as i64);
            }
            start += n;
            ring += 1;
        }
    }
    // edge.faces: per edge, its segments' faces in global segment order
    // (a stable counting sort by edge).
    let ne = out_edges.len();
    let mut ef_start = vec![0i64; ne + 1];
    for &e in &seg_edge {
        ef_start[e + 1] += 1;
    }
    for i in 0..ne {
        ef_start[i + 1] += ef_start[i];
    }
    let mut fill: Vec<i64> = ef_start[..ne].to_vec();
    let mut ef_face = vec![0i64; seg_edge.len()];
    for (s, &e) in seg_edge.iter().enumerate() {
        ef_face[fill[e] as usize] = seg_face[s];
        fill[e] += 1;
    }
    Topology { edges: out_edges, soft, hidden, curve, layer, corners, ef_start, ef_face }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn square_with_a_shared_edge() {
        // Two unit squares sharing the edge 1-2; points already welded.
        let inverse: Vec<i64> = (0..6).collect();
        let rings = [0, 1, 2, 3, 1, 4, 5, 2];
        let t = topology(&inverse, &[[1, 2]], Some(&[1]), None, None, &rings, &[4, 4], &[1, 1]);
        assert_eq!(t.edges[0], [1, 2]);
        assert!(t.soft[0] && !t.hidden[0]);
        assert_eq!(t.edges.len(), 7);
        // The shared edge lists both faces, in segment order.
        assert_eq!(&t.ef_face[t.ef_start[0] as usize..t.ef_start[1] as usize], &[0, 1]);
        assert_eq!(*t.ef_start.last().unwrap(), 8);
    }

    #[test]
    fn welded_shut_edges_are_skipped_and_last_curve_wins() {
        let inverse = [0, 0, 1];
        let t = topology(&inverse, &[[0, 1], [1, 2], [2, 0]], Some(&[0, 2, 0]),
                         Some(&[-1, 5, 7]), None, &[], &[], &[]);
        assert_eq!(t.edges, vec![[0, 1]]);
        assert_eq!(t.curve, vec![7]);
        assert!(t.hidden[0]);
        assert_eq!(t.ef_start, vec![0, 0]);
    }
}
