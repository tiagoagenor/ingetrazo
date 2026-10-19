// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! `formats.fuse.fuse_coplanar_loops` after the attribute signatures: the
//! coplanar same-material regions of a soup of polygons and their traced
//! outlines. Op for op with the Python: `_newell` (f64 terms rounded to
//! f32, summed in f32), `n / ln` in f32, Qt's length and dot products,
//! `_key`'s `round(x, 4)` as `round_ties_even(x · 1e4)` (exact for a
//! float32 x), dicts walked in insertion order, `_trace_region` as A5 left
//! it, the stable reverse sort by area. Maps only look up.

use std::collections::HashMap;

use crate::facegeom::{length, Hypot, V3};

/// `_key(p)`: the three rounded coordinates, as exact f64 (−0 folded).
type Key = [u64; 3];

#[inline]
fn key(p: V3) -> Key {
    let r = |x: f32| {
        let v = (x as f64 * 1e4).round_ties_even();
        (if v == 0.0 { 0.0 } else { v }).to_bits()
    };
    [r(p.0), r(p.1), r(p.2)]
}

#[inline]
fn key_lt_eq(a: &Key, b: &Key) -> bool {
    // Python compares the rounded floats; compare the f64 values.
    let fa = a.map(f64::from_bits);
    let fb = b.map(f64::from_bits);
    for k in 0..3 {
        if fa[k] < fb[k] {
            return true;
        }
        if fa[k] > fb[k] {
            return false;
        }
    }
    true
}

fn newell(pts: &[V3]) -> V3 {
    let m = pts.len();
    let mut n = V3(0.0, 0.0, 0.0);
    for i in 0..m {
        let a = pts[i];
        let b = pts[(i + 1) % m];
        let t = V3(
            ((a.1 as f64 - b.1 as f64) * (a.2 as f64 + b.2 as f64)) as f32,
            ((a.2 as f64 - b.2 as f64) * (a.0 as f64 + b.0 as f64)) as f32,
            ((a.0 as f64 - b.0 as f64) * (a.1 as f64 + b.1 as f64)) as f32,
        );
        n = V3(n.0 + t.0, n.1 + t.1, n.2 + t.2);
    }
    n
}

/// Per input loop: its points (global indices into the flat point list),
/// signature id, texture group (-1: not a fuzzy-UV candidate) and uvw.
pub struct Input<'a> {
    pub pts: &'a [V3],
    pub loop_start: &'a [usize],
    pub sig: &'a [i64],
    pub tex_group: &'a [i64],
    pub rank: &'a [u8],
    pub uvw: &'a [Vec<f64>],
}

struct Face {
    lp: usize,
    n: V3,
}

pub enum Region {
    /// The faces (input loop indices) passed through unchanged.
    Pass(Vec<usize>),
    /// Traced: rings of point indices (outer first), the region's faces.
    Traced(Vec<Vec<usize>>, Vec<usize>),
}

fn compatible(inp: &Input, a: usize, b: usize) -> bool {
    if inp.sig[a] == inp.sig[b] {
        return true;
    }
    let (ga, gb) = (inp.tex_group[a], inp.tex_group[b]);
    if ga < 0 || ga != gb {
        return false;
    }
    let (ua, ub) = (&inp.uvw[a], &inp.uvw[b]);
    !ua.is_empty()
        && !ub.is_empty()
        && ua.iter().zip(ub).all(|(&x, &y)| (x - y).abs() <= 0.02 + 0.01 * x.abs().max(y.abs()))
}

fn ring(inp: &Input, lp: usize) -> std::ops::Range<usize> {
    inp.loop_start[lp]..inp.loop_start[lp + 1]
}

pub fn fuse(inp: &Input, cos_tol: f64, principal: bool, h: Hypot) -> Vec<Region> {
    let nloops = inp.loop_start.len() - 1;
    let keys: Vec<Key> = inp.pts.iter().map(|&p| key(p)).collect();
    // Faces, the coincident triangle copies folded (the best-ranked kept).
    let mut faces: Vec<Face> = Vec::new();
    let mut seen_tris: HashMap<Vec<Key>, usize> = HashMap::new();
    for lp in 0..nloops {
        let r = ring(inp, lp);
        if r.len() < 3 {
            continue;
        }
        let pts = &inp.pts[r.clone()];
        let nw = newell(pts);
        let ln = length(nw, h) as f64;
        if ln < 1e-10 {
            continue;
        }
        let l32 = ln as f32;
        let n = V3(nw.0 / l32, nw.1 / l32, nw.2 / l32);
        let entry = Face { lp, n };
        if r.len() == 3 {
            let mut tkey: Vec<Key> = keys[r.clone()].to_vec();
            tkey.sort();
            tkey.dedup();
            match seen_tris.get(&tkey) {
                None => {
                    seen_tris.insert(tkey, faces.len());
                    faces.push(entry);
                }
                Some(&prev) => {
                    if inp.rank[lp] > inp.rank[faces[prev].lp] {
                        faces[prev] = entry;
                    }
                }
            }
            continue;
        }
        faces.push(entry);
    }
    let nf = faces.len();
    let mut parent: Vec<usize> = (0..nf).collect();
    fn find(parent: &mut [usize], mut i: usize) -> usize {
        while parent[i] != i {
            parent[i] = parent[parent[i]];
            i = parent[i];
        }
        i
    }
    // edge_map: undirected edges in first-insertion order.
    let mut edge_slot: HashMap<(Key, Key), usize> = HashMap::new();
    let mut edge_faces: Vec<Vec<usize>> = Vec::new();
    for (i, f) in faces.iter().enumerate() {
        let r = ring(inp, f.lp);
        let m = r.len();
        for k in 0..m {
            let ka = keys[r.start + k];
            let kb = keys[r.start + (k + 1) % m];
            if ka == kb {
                continue;
            }
            let e = if key_lt_eq(&ka, &kb) { (ka, kb) } else { (kb, ka) };
            let s = *edge_slot.entry(e).or_insert_with(|| {
                edge_faces.push(Vec::new());
                edge_faces.len() - 1
            });
            edge_faces[s].push(i);
        }
    }
    let on_principal = |n: V3| -> bool {
        (n.0.abs().max(n.1.abs()).max(n.2.abs()) as f64) >= 1.0 - 1e-7
    };
    for idxs in &edge_faces {
        if idxs.len() != 2 {
            continue;
        }
        let (i, j) = (idxs[0], idxs[1]);
        if principal && !(on_principal(faces[i].n) && on_principal(faces[j].n)) {
            continue;
        }
        let (li, lj) = (faces[i].lp, faces[j].lp);
        if !compatible(inp, li, lj) {
            continue;
        }
        if (faces[i].n.dot(faces[j].n).abs() as f64) < cos_tol {
            continue;
        }
        let ri = find(&mut parent, i);
        let rj = find(&mut parent, j);
        if ri == rj {
            continue;
        }
        if (faces[ri].n.dot(faces[rj].n).abs() as f64) < cos_tol {
            continue;
        }
        if !compatible(inp, faces[ri].lp, faces[rj].lp) {
            continue;
        }
        parent[rj] = ri;
    }
    // regions in first-appearance order of their roots.
    let mut region_slot: HashMap<usize, usize> = HashMap::new();
    let mut regions: Vec<Vec<usize>> = Vec::new();
    for i in 0..nf {
        let r = find(&mut parent, i);
        let s = *region_slot.entry(r).or_insert_with(|| {
            regions.push(Vec::new());
            regions.len() - 1
        });
        regions[s].push(i);
    }
    regions
        .into_iter()
        .map(|reg| {
            let lps: Vec<usize> = reg.iter().map(|&i| faces[i].lp).collect();
            if lps.len() == 1 {
                return Region::Pass(lps);
            }
            match trace(inp, &keys, &reg.iter().map(|&i| (faces[i].lp, faces[i].n)).collect::<Vec<_>>(), h) {
                Some(rings) => Region::Traced(rings, lps),
                None => Region::Pass(lps),
            }
        })
        .collect()
}

/// `_trace_region` for a region of two or more faces: rings of point
/// indices (outer first, oriented), or None.
fn trace(inp: &Input, keys: &[Key], region: &[(usize, V3)], h: Hypot) -> Option<Vec<Vec<usize>>> {
    let mut count: Vec<usize> = Vec::new();
    let mut first: Vec<(Key, Key)> = Vec::new();
    let mut slot: HashMap<(Key, Key), usize> = HashMap::new();
    let mut pos: HashMap<Key, usize> = HashMap::new();
    for &(lp, _n) in region {
        let r = ring(inp, lp);
        let m = r.len();
        for k in 0..m {
            let (ia, ib) = (r.start + k, r.start + (k + 1) % m);
            let (ka, kb) = (keys[ia], keys[ib]);
            if ka == kb {
                continue;
            }
            let e = if key_lt_eq(&ka, &kb) { (ka, kb) } else { (kb, ka) };
            let s = *slot.entry(e).or_insert_with(|| {
                count.push(0);
                first.push((ka, kb));
                count.len() - 1
            });
            count[s] += 1;
            pos.entry(ka).or_insert(ia);
            pos.entry(kb).or_insert(ib);
        }
    }
    let boundary: Vec<usize> = (0..count.len()).filter(|&s| count[s] == 1).collect();
    if boundary.is_empty() || count.iter().any(|&c| c > 2) {
        return None;
    }
    let mut adj: HashMap<Key, Vec<Key>> = HashMap::new();
    for &s in &boundary {
        let (a, b) = first[s];
        adj.entry(a).or_default().push(b);
        adj.entry(b).or_default().push(a);
    }
    if adj.values().any(|v| v.len() != 2) {
        return None;
    }
    let mut unvisited = vec![false; count.len()];
    for &s in &boundary {
        unvisited[s] = true;
    }
    let undirected = |a: Key, b: Key| if key_lt_eq(&a, &b) { (a, b) } else { (b, a) };
    let mut traced: Vec<Vec<usize>> = Vec::new();
    for &s0 in &boundary {
        if !unvisited[s0] {
            continue;
        }
        let (start, mut cur) = first[s0];
        unvisited[s0] = false;
        let mut walk = vec![start, cur];
        loop {
            let nb = &adj[&cur];
            let (a, b) = (nb[0], nb[1]);
            let nxt = if a == walk[walk.len() - 2] { b } else { a };
            let e = slot[&undirected(cur, nxt)];
            if !unvisited[e] {
                break;
            }
            unvisited[e] = false;
            if nxt == start {
                break;
            }
            walk.push(nxt);
            cur = nxt;
        }
        if *walk.last().unwrap() == start {
            walk.pop();
        }
        if walk.len() < 3 {
            return None;
        }
        traced.push(walk.iter().map(|k| pos[k]).collect());
    }
    // Largest first (a stable sort, descending), the outer against the
    // region's first normal, holes the other way.
    let n_ref = region[0].1;
    let mut areas: Vec<(f32, V3, Vec<usize>)> = traced
        .into_iter()
        .map(|lp| {
            let pts: Vec<V3> = lp.iter().map(|&i| inp.pts[i]).collect();
            let nw = newell(&pts);
            (length(nw, h), nw, lp)
        })
        .collect();
    areas.sort_by(|x, y| y.0.partial_cmp(&x.0).unwrap_or(std::cmp::Ordering::Equal));
    let mut out = Vec::with_capacity(areas.len());
    for (k, (_l, nw, mut lp)) in areas.into_iter().enumerate() {
        let d = nw.dot(n_ref);
        if (k == 0 && d < 0.0) || (k > 0 && d > 0.0) {
            lp.reverse();
        }
        out.push(lp);
    }
    Some(out)
}
