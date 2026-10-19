// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! `core.sweep` and `formats.fuse.simplify_mesh` against what the Python
//! computed (scripts/golden/make_sweep_fixtures.py): every float, bit for bit.
//! Large outputs travel as an FNV-1a digest over their float32 words.

use std::path::Path;

use ingetrazo_formats::pyjson::{loads, Value};
use ingetrazo_geom::simplify::{self, AttrInfo, EdgeFlags, SrcEdge, SrcFace};
use ingetrazo_geom::sweep::{self, PathGraph, Poly, Profile, Ring};
use ingetrazo_math::{Hypot, V3};

/// The fixtures were written on Windows: MSVC's hypot.
const H: Hypot = Hypot::Msvc;

fn rows(name: &str) -> Vec<Value> {
    let p = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data").join(name);
    let text = std::fs::read_to_string(p).unwrap();
    text.lines().skip(1).map(|l| loads(l).unwrap()).collect()
}

fn arr(v: &Value) -> &Vec<Value> {
    match v {
        Value::Array(a) => a,
        _ => panic!("{v:?}"),
    }
}

fn s(v: &Value) -> &str {
    match v {
        Value::Str(x) => x,
        _ => panic!("{v:?}"),
    }
}

fn b(v: &Value) -> bool {
    match v {
        Value::Bool(x) => *x,
        _ => panic!("{v:?}"),
    }
}

fn u(v: &Value) -> usize {
    v.as_i64().unwrap() as usize
}

fn f32h(v: &Value) -> f32 {
    f32::from_bits(u32::from_str_radix(s(v), 16).unwrap())
}

fn f64h(v: &Value) -> f64 {
    f64::from_bits(u64::from_str_radix(s(v), 16).unwrap())
}

fn p3(v: &Value) -> V3 {
    let a = arr(v);
    V3(f32h(&a[0]), f32h(&a[1]), f32h(&a[2]))
}

fn pts(v: &Value) -> Vec<V3> {
    arr(v).iter().map(p3).collect()
}

fn opt(v: &Value) -> Option<&Value> {
    match v {
        Value::Null => None,
        x => Some(x),
    }
}

fn bits(p: V3) -> [u32; 3] {
    [p.0.to_bits(), p.1.to_bits(), p.2.to_bits()]
}

struct Digest(u64);

impl Digest {
    fn new() -> Digest {
        Digest(0xCBF29CE484222325)
    }
    fn word(&mut self, w: u32) {
        for shift in [0, 8, 16, 24] {
            self.0 ^= ((w >> shift) & 0xFF) as u64;
            self.0 = self.0.wrapping_mul(0x100000001B3);
        }
    }
    fn pt(&mut self, p: V3) {
        self.word(p.0.to_bits());
        self.word(p.1.to_bits());
        self.word(p.2.to_bits());
    }
    fn polys(&mut self, polys: &[Poly]) {
        self.word(polys.len() as u32);
        for q in polys {
            self.word(q.outer.len() as u32);
            q.outer.iter().for_each(|&p| self.pt(p));
            self.word(q.holes.len() as u32);
            for h in &q.holes {
                self.word(h.len() as u32);
                h.iter().for_each(|&p| self.pt(p));
            }
        }
    }
    fn hex(&self) -> String {
        format!("{:016x}", self.0)
    }
}

fn profile(v: &Value) -> Profile {
    Profile { outer: pts(v.get("o").unwrap()), holes: arr(v.get("h").unwrap()).iter().map(pts).collect() }
}

fn rings_digest(r: &[Ring]) -> String {
    let mut d = Digest::new();
    d.word(r.len() as u32);
    for ring in r {
        d.word(ring.len() as u32);
        for lp in ring {
            d.word(lp.len() as u32);
            lp.iter().for_each(|&p| d.pt(p));
        }
    }
    d.hex()
}

fn same(a: &[V3], b: &[V3], what: &str, i: usize) {
    assert_eq!(a.iter().map(|&p| bits(p)).collect::<Vec<_>>(), b.iter().map(|&p| bits(p)).collect::<Vec<_>>(), "{what} case {i}");
}

#[test]
fn sweep_rings_previews_and_plans() {
    let mut revolution = 0;
    let mut n = 0;
    for (i, c) in rows("sweep_rings.jsonl").iter().enumerate() {
        n += 1;
        let face = profile(c.get("face").unwrap());
        let path = pts(c.get("path").unwrap());
        let closed = b(c.get("closed").unwrap());
        let got = sweep::sweep_rings(&face, &path, closed, H);
        match opt(c.get("rings").unwrap()) {
            None => assert!(got.is_none(), "rings case {i}"),
            Some(r) => {
                let g = got.as_ref().unwrap_or_else(|| panic!("rings case {i}: None"));
                assert_eq!(rings_digest(&g.rings), s(r.get("digest").unwrap()), "rings digest case {i}");
                assert_eq!(g.rings.len(), u(r.get("n").unwrap()));
                assert_eq!(g.spans, u(r.get("s").unwrap()), "spans case {i}");
                same(&g.dirs, &pts(r.get("d").unwrap()), "dirs", i);
                if let Some(full) = r.get("r") {
                    for (si, ring) in arr(full).iter().enumerate() {
                        for (li, lp) in arr(ring).iter().enumerate() {
                            same(&g.rings[si][li], &pts(lp), "ring", i);
                        }
                    }
                }
                if closed && g.rings.len() >= 3 && g.spans == g.rings.len() {
                    // the lathe: every station is the profile turned
                    if sweep::revolution_rings(&face, &path, H).is_some() {
                        revolution += 1;
                    }
                }
            }
        }
        let prev = sweep::sweep_preview_faces(&face, &path, closed, H);
        let mut d = Digest::new();
        d.polys(&prev);
        assert_eq!(d.hex(), s(c.get("preview").unwrap()), "preview case {i}");
        assert_eq!(prev.len(), u(c.get("preview_n").unwrap()));
        let plan = sweep::sweep_plan(&face, &path, closed, H);
        match opt(c.get("plan").unwrap()) {
            None => assert!(plan.is_none(), "plan case {i}"),
            Some(e) => {
                let p = plan.unwrap_or_else(|| panic!("plan case {i}: None"));
                let mut d = Digest::new();
                d.polys(&p.faces);
                assert_eq!(d.hex(), s(e.get("digest").unwrap()), "plan digest case {i}");
                assert_eq!(p.faces.len(), u(e.get("n").unwrap()));
                assert_eq!(p.wall_count, u(e.get("walls").unwrap()), "walls case {i}");
                let mut sg = Digest::new();
                for (a, b) in &p.profile_segments {
                    sg.pt(*a);
                    sg.pt(*b);
                }
                assert_eq!(sg.hex(), s(e.get("segs").unwrap()), "segments case {i}");
                // the seed set: sorted as Python sorted the tuples, as >d words
                let mut seed = p.seed.clone();
                seed.sort_by(|a, b| a.partial_cmp(b).unwrap());
                assert_eq!(seed.len(), u(e.get("seed_n").unwrap()), "seed size case {i}");
                let mut sd = Digest::new();
                for k in &seed {
                    for c in k {
                        let w = c.to_bits();
                        sd.word((w >> 32) as u32);
                        sd.word(w as u32);
                    }
                }
                assert_eq!(sd.hex(), s(e.get("seed").unwrap()), "seed case {i}");
            }
        }
    }
    assert!(n >= 500, "fixtures present");
    assert!(revolution >= 20, "lathe cases exercised: {revolution}");
}

#[test]
fn hypot_orient_path_facets_and_dragged_paths() {
    let mut counts = [0usize; 6];
    for (i, c) in rows("sweep_misc.jsonl").iter().enumerate() {
        match s(c.get("k").unwrap()) {
            "hypot" => {
                counts[0] += 1;
                let r = sweep::py_hypot2(f64h(c.get("a").unwrap()), f64h(c.get("b").unwrap()));
                assert_eq!(r.to_bits(), f64h(c.get("r").unwrap()).to_bits(), "hypot case {i}");
            }
            "order" => {
                counts[1] += 1;
                let pos = pts(c.get("pos").unwrap());
                let edges: Vec<(usize, usize)> = arr(c.get("edges").unwrap()).iter().map(|e| (u(&arr(e)[0]), u(&arr(e)[1]))).collect();
                let got = sweep::order_path_edges(&edges, &pos);
                match opt(c.get("res").unwrap()) {
                    None => assert!(got.is_none(), "order case {i}"),
                    Some(r) => {
                        let (p, closed) = got.unwrap_or_else(|| panic!("order case {i}: None"));
                        assert_eq!(closed, b(r.get("c").unwrap()), "closed case {i}");
                        same(&p, &pts(r.get("p").unwrap()), "order", i);
                    }
                }
            }
            "orient" => {
                counts[2] += 1;
                let face = profile(c.get("face").unwrap());
                let got = sweep::orient_closed_path(&pts(c.get("pts").unwrap()), &face, H);
                same(&got, &pts(c.get("res").unwrap()), "orient", i);
            }
            "mstart" => {
                counts[3] += 1;
                let face = profile(c.get("face").unwrap());
                let a = arr(c.get("a").unwrap());
                let bb = arr(c.get("b").unwrap());
                let toward = opt(c.get("toward").unwrap()).map(p3);
                let (pt, chain) = sweep::manual_path_start(&face, (u(&a[0]), p3(&a[1])), (u(&bb[0]), p3(&bb[1])), toward, H);
                let res = arr(c.get("res").unwrap());
                assert_eq!(pt.map(bits), opt(&res[0]).map(|v| bits(p3(v))), "mstart point case {i}");
                assert_eq!(chain, arr(&res[1]).iter().map(u).collect::<Vec<_>>(), "mstart chain case {i}");
            }
            "mext" => {
                counts[4] += 1;
                let npos = arr(c.get("pos").unwrap()).len();
                let edges: Vec<(usize, usize)> = arr(c.get("edges").unwrap()).iter().map(|e| (u(&arr(e)[0]), u(&arr(e)[1]))).collect();
                let mut vedges = vec![Vec::new(); npos];
                for (k, &(v0, v1)) in edges.iter().enumerate() {
                    vedges[v0].push(k);
                    if v1 != v0 {
                        vedges[v1].push(k);
                    }
                }
                let g = PathGraph { edges, vedges };
                let skip_set: Vec<usize> = arr(c.get("skip").unwrap()).iter().map(u).collect();
                let skip = |e: usize| skip_set.contains(&e);
                let limit = u(c.get("limit").unwrap());
                let mut chain = vec![u(c.get("start").unwrap())];
                for (si, st) in arr(c.get("steps").unwrap()).iter().enumerate() {
                    let st = arr(st);
                    let changed = sweep::manual_path_extend(&g, &mut chain, u(&st[0]), &skip, limit);
                    assert_eq!(changed, b(&st[1]), "mext changed case {i} step {si}");
                    assert_eq!(chain, arr(&st[2]).iter().map(u).collect::<Vec<_>>(), "mext chain case {i} step {si}");
                }
            }
            "facet" => {
                counts[5] += 1;
                let soft = sweep::curve_facet_soft(p3(c.get("n0").unwrap()), p3(c.get("n1").unwrap()), H);
                assert_eq!(soft, b(c.get("soft").unwrap()), "facet case {i}");
            }
            k => panic!("unknown kind {k}"),
        }
    }
    assert!(counts.iter().all(|&n| n >= 100), "{counts:?}");
}

fn flags(v: &Value) -> EdgeFlags {
    let a = arr(v);
    EdgeFlags {
        soft: b(&a[0]),
        curve: opt(&a[1]).map(|x| x.as_i64().unwrap()),
        layer: opt(&a[2]).map(|x| x.as_i64().unwrap() as u32),
        hidden: b(&a[3]),
    }
}

#[test]
fn simplify_matches_python() {
    let mut merged = 0;
    let mut simplified = 0;
    let mut errors = 0;
    let mut collapsed = 0;
    let mut soft = 0;
    for (i, c) in rows("simplify.jsonl").iter().enumerate() {
        match s(c.get("k").unwrap()) {
            "cos" => {
                let a = f64h(c.get("a").unwrap());
                match opt(c.get("r").unwrap()) {
                    None => assert!(simplify::cos_tolerance(a).is_err(), "cos case {i}"),
                    Some(r) => {
                        assert_eq!(simplify::cos_tolerance(a).unwrap().to_bits(), f64h(r).to_bits(), "cos case {i}: {a}")
                    }
                }
            }
            "simplify" => {
                let pos = pts(c.get("pos").unwrap());
                let attrs: Vec<AttrInfo> = arr(c.get("attrs").unwrap())
                    .iter()
                    .map(|a| AttrInfo {
                        sig: a.get("sig").unwrap().as_i64().unwrap(),
                        tex_group: a.get("grp").unwrap().as_i64().unwrap(),
                        rank: a.get("rank").unwrap().as_i64().unwrap() as u8,
                        uvw: arr(a.get("uvw").unwrap()).iter().map(f64h).collect(),
                        smooth: a.get("smooth").unwrap().as_i64().unwrap(),
                    })
                    .collect();
                let faces: Vec<SrcFace> = arr(c.get("faces").unwrap())
                    .iter()
                    .map(|f| {
                        let f = arr(f);
                        let lp = |v: &Value| -> Vec<V3> { arr(v).iter().map(|x| pos[u(x)]).collect() };
                        SrcFace {
                            outer: lp(&f[0]),
                            holes: arr(&f[1]).iter().map(lp).collect(),
                            interior: b(&f[2]),
                            attrs: opt(&f[3]).map(u),
                        }
                    })
                    .collect();
                let edges: Vec<SrcEdge> = arr(c.get("edges").unwrap())
                    .iter()
                    .map(|e| {
                        let e = arr(e);
                        SrcEdge { a: pos[u(&e[0])], b: pos[u(&e[1])], flags: flags(&e[2]) }
                    })
                    .collect();
                let angle = f64h(c.get("angle").unwrap());
                let principal = b(c.get("principal").unwrap());
                let got = simplify::simplify(&faces, &edges, &attrs, angle, principal, None, H);
                let res = c.get("res").unwrap();
                if matches!(res, Value::Str(x) if x == "error") {
                    errors += 1;
                    assert!(got.is_err(), "simplify case {i}: should be an error");
                    continue;
                }
                let g = got.unwrap_or_else(|_| panic!("simplify case {i}: error"));
                simplified += 1;
                assert_eq!(g.removed, u(c.get("removed").unwrap()), "removed case {i}");
                if g.removed > 0 {
                    merged += 1;
                }
                collapsed += (g.vertices.len() < pos.len()) as usize;
                soft += g.edges.iter().filter(|e| e.flags.soft && !edges.iter().any(|o| o.flags.soft)).count().min(1);
                let want_v: Vec<[u32; 3]> = arr(res.get("v").unwrap()).iter().map(|x| bits(pos[u(x)])).collect();
                let got_v: Vec<[u32; 3]> = g.vertices.iter().map(|&p| bits(p)).collect();
                assert_eq!(got_v, want_v, "vertices case {i}");
                let we = arr(res.get("e").unwrap());
                assert_eq!(g.edges.len(), we.len(), "edge count case {i}");
                for (k, (ge, w)) in g.edges.iter().zip(we).enumerate() {
                    let w = arr(w);
                    let (a, bb) = (u(&w[0]), u(&w[1]));
                    // a collapse-created edge's orientation follows Python's set order
                    assert_eq!((ge.v0.min(ge.v1), ge.v0.max(ge.v1)), (a.min(bb), a.max(bb)), "edge {k} ends case {i}");
                    let mut fs = ge.faces.clone();
                    fs.sort_unstable();
                    assert_eq!(fs, arr(&w[2]).iter().map(u).collect::<Vec<_>>(), "edge {k} faces case {i}");
                    assert_eq!(ge.flags, flags(&w[3]), "edge {k} flags case {i}");
                }
                let wf = arr(res.get("f").unwrap());
                assert_eq!(g.faces.len(), wf.len(), "face count case {i}");
                for (k, (gf, w)) in g.faces.iter().zip(wf).enumerate() {
                    let w = arr(w);
                    assert_eq!(gf.outer, arr(&w[0]).iter().map(u).collect::<Vec<_>>(), "face {k} loop case {i}");
                    let wh: Vec<Vec<usize>> = arr(&w[1]).iter().map(|h| arr(h).iter().map(u).collect()).collect();
                    assert_eq!(gf.holes, wh, "face {k} holes case {i}");
                    assert_eq!(gf.interior, b(&w[2]), "face {k} interior case {i}");
                    assert_eq!(gf.attrs, opt(&w[3]).map(u), "face {k} attrs case {i}");
                }
            }
            k => panic!("unknown kind {k}"),
        }
    }
    assert!(simplified >= 50 && merged >= 30 && errors == 5 && collapsed >= 20 && soft >= 10, "{simplified} {merged} {errors} {collapsed} {soft}");
}
