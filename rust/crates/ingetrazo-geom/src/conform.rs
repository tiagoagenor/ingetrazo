// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! `core.solids._conform_t_junctions`: every triangle with a mesh vertex
//! inside one of its edges is fanned from its centroid through those
//! vertices. The Python's scalar f64 arithmetic op for op; the vertices
//! near an edge come from an index sorted by x instead of a scan of all of
//! them (the box test itself is the same comparisons, and the hits are
//! sorted by (s, i), so the order they are found in does not matter).

pub struct Out {
    pub verts: Vec<[f64; 3]>,
    pub tris: Vec<[u32; 3]>,
    pub fids: Vec<u32>,
}

/// ``v`` finite vertices, ``t`` triangles, ``f`` their face ids, ``tol``
/// as the Python computed it.
pub fn conform(v: &[[f64; 3]], t: &[[i64; 3]], f: &[i64], tol: f64) -> Out {
    let tol2 = tol * tol;
    let mut by_x: Vec<usize> = (0..v.len()).collect();
    by_x.sort_by(|&i, &j| v[i][0].total_cmp(&v[j][0]).then(i.cmp(&j)));
    let xs: Vec<f64> = by_x.iter().map(|&i| v[i][0]).collect();
    let mut verts: Vec<[f64; 3]> = v.to_vec();
    let mut tris = Vec::with_capacity(t.len());
    let mut fids = Vec::with_capacity(t.len());
    let mut hits: Vec<(f64, usize)> = Vec::new();
    for (tri, &fid) in t.iter().zip(f) {
        let mut ring: Vec<usize> = Vec::with_capacity(3);
        let mut split = false;
        for k in 0..3 {
            let a = tri[k] as usize;
            let b = tri[(k + 1) % 3] as usize;
            ring.push(a);
            let [ax, ay, az] = v[a];
            let [bx, by, bz] = v[b];
            let (dx, dy, dz) = (bx - ax, by - ay, bz - az);
            let length2 = dx * dx + dy * dy + dz * dz;
            if length2 <= tol2 {
                continue;
            }
            // np.minimum(pa, pb) - tol, np.maximum(pa, pb) + tol.
            let lo = [ax.min(bx) - tol, ay.min(by) - tol, az.min(bz) - tol];
            let hi = [ax.max(bx) + tol, ay.max(by) + tol, az.max(bz) + tol];
            hits.clear();
            let from = xs.partition_point(|&x| x < lo[0]);
            for (&x, &i) in xs[from..].iter().zip(&by_x[from..]) {
                if x > hi[0] {
                    break;
                }
                let p = v[i];
                if !(p[1] >= lo[1] && p[1] <= hi[1] && p[2] >= lo[2] && p[2] <= hi[2]) {
                    continue;
                }
                if i == a || i == b {
                    continue;
                }
                let [px, py, pz] = p;
                let s = ((px - ax) * dx + (py - ay) * dy + (pz - az) * dz) / length2;
                if s <= 1e-9 || s >= 1.0 - 1e-9 {
                    continue;
                }
                let ox = px - (ax + s * dx);
                let oy = py - (ay + s * dy);
                let oz = pz - (az + s * dz);
                if ox * ox + oy * oy + oz * oz <= tol2 {
                    hits.push((s, i));
                }
            }
            if !hits.is_empty() {
                split = true;
                hits.sort_by(|x, y| x.0.total_cmp(&y.0).then(x.1.cmp(&y.1)));
                ring.extend(hits.iter().map(|&(_, i)| i));
            }
        }
        if !split {
            tris.push([tri[0] as u32, tri[1] as u32, tri[2] as u32]);
            fids.push(fid as u32);
            continue;
        }
        let c = verts.len();
        let [p0, p1, p2] = [v[tri[0] as usize], v[tri[1] as usize], v[tri[2] as usize]];
        verts.push([
            (p0[0] + p1[0] + p2[0]) / 3.0,
            (p0[1] + p1[1] + p2[1]) / 3.0,
            (p0[2] + p1[2] + p2[2]) / 3.0,
        ]);
        let n = ring.len();
        for k in 0..n {
            tris.push([c as u32, ring[k] as u32, ring[(k + 1) % n] as u32]);
            fids.push(fid as u32);
        }
    }
    Out { verts, tris, fids }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_t_junction_fans_the_triangle() {
        let v = [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [1.0, 0.0, 0.0]];
        let o = conform(&v, &[[0, 1, 2]], &[7], 1e-9);
        assert_eq!(o.verts.len(), 5);
        assert_eq!(o.verts[4], [2.0 / 3.0, 2.0 / 3.0, 0.0]);
        assert_eq!(o.tris, vec![[4, 0, 3], [4, 3, 1], [4, 1, 2], [4, 2, 0]]);
        assert_eq!(o.fids, vec![7; 4]);
    }
}
