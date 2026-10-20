// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! The pick index (`Viewport._pick_index`): the triangles of every face the
//! picks can see, with per-face masks (visible, selectable) and areas, the
//! loose edges, and the hard edges of every group placement.
//!
//! Python bakes each placement into world space ("a chunk": its mesh's
//! triangles as `v0, e1, e2` rows in float64 and its hard edges in float32,
//! moved by `pts @ L.T + t` through BLAS for an instance) and joins the
//! chunks after the loose part. This module makes the same arrays, in the
//! same order, with the same rounding.
//!
//! Not reproduced: the lazy placements of a model over a million faces, the
//! grid that cuts a chunk of more than 8192 triangles into cells (a ray
//! then meets cells, not chunks), and face-me billboards.

use ingetrazo_geom::meshface;
use ingetrazo_geom::triangulate::triangulate;
use ingetrazo_math::{Hypot, M4, V3};
use ingetrazo_model::mesh::Mesh;
use ingetrazo_model::pyval::truthy;
use ingetrazo_model::scene::{Group, MeshId};
use ingetrazo_ops::group::{affine_rows, np_affine};

use super::numeric::{np_det3, p3, rows_matmul, P3};
use super::state::{Path, State};

/// A placement of a group: the group itself, or a nested placement below
/// it (`Viewport._placements()`, whose proxies carry the composed world
/// matrix and the group a click on them selects).
#[derive(Clone, Debug)]
pub struct Place {
    /// The node of the tree this placement is.
    pub path: Path,
    pub mesh: MeshId,
    /// The world matrix; `None` for an untransformed top-level group.
    pub xform: Option<M4>,
    pub hidden: bool,
    pub layer: Option<String>,
    pub billboard: bool,
    /// The group a click on it selects (`owner`, or itself).
    pub owner: Path,
    /// The group's own lock (a proxy has none).
    pub self_locked: bool,
    /// Its owner's lock.
    pub owner_locked: bool,
    /// `g is ctx` (only a real group can be).
    pub is_ctx: bool,
    /// `g.context is ctx`, ctx being open.
    pub in_context: bool,
}

/// `Viewport._placements()` for the scene: each top-level group followed by
/// the nested placements below it, depth first.
pub fn placements(st: &State) -> Vec<Place> {
    let ctx = st.ctx.as_deref();
    let mut out = Vec::new();
    for (i, g) in st.scene.groups.iter().enumerate() {
        let path = vec![i];
        out.push(Place {
            path: path.clone(),
            mesh: g.mesh,
            xform: g.xform,
            hidden: g.hidden,
            layer: g.layer.clone().filter(|l| !l.is_empty()),
            billboard: billboard(g),
            owner: path.clone(),
            self_locked: g.locked,
            owner_locked: false,
            is_ctx: ctx == Some(path.as_slice()),
            in_context: false,
        });
        if g.children.is_empty() {
            continue;
        }
        let (vis, locked) = st.layer_state(g.layer.as_deref());
        let forced = if !vis || locked { g.layer.clone().filter(|l| !l.is_empty()) } else { None };
        let mut w = Walk { st, ctx, forced, top: path.clone(), out: &mut out };
        w.walk(g, &path, g.layer.as_deref(), g.xform, None, g.hidden);
    }
    out
}

fn billboard(g: &Group) -> bool {
    g.billboard.as_ref().map(truthy).unwrap_or(false)
}

struct Walk<'a, 'b, 's> {
    st: &'a State<'s>,
    ctx: Option<&'a [usize]>,
    forced: Option<String>,
    top: Path,
    out: &'b mut Vec<Place>,
}

impl Walk<'_, '_, '_> {
    /// `walk(node, world, dentro, hidden)` of `_expand_placements`.
    fn walk(&mut self, node: &Group, node_path: &Path, node_layer: Option<&str>, world: Option<M4>, dentro: Option<Path>, hidden: bool) {
        let node_is_ctx = self.ctx == Some(node_path.as_slice());
        for (ci, child) in node.children.iter().enumerate() {
            let m = match world {
                None => child.xform,
                Some(w) => match child.xform {
                    Some(x) => Some(w.mul(&x)),
                    None => Some(w),
                },
            };
            let mut cpath = node_path.clone();
            cpath.push(ci);
            let layer = self
                .forced
                .clone()
                .or_else(|| child.layer.clone().filter(|l| !l.is_empty()))
                .or_else(|| node_layer.map(str::to_string).filter(|l| !l.is_empty()));
            let hid = hidden || child.hidden;
            let child_is_ctx = self.ctx == Some(cpath.as_slice());
            let (owner, in_context) = if child_is_ctx || (self.ctx.is_some() && node_is_ctx) {
                (cpath.clone(), true)
            } else if let Some(d) = &dentro {
                (d.clone(), self.ctx.is_some())
            } else {
                (self.top.clone(), false)
            };
            let owner_locked = group_flag_locked(self.st, &owner);
            self.out.push(Place {
                path: cpath.clone(),
                mesh: child.mesh,
                xform: m,
                hidden: hid,
                layer: layer.clone(),
                billboard: billboard(child),
                owner,
                self_locked: false,
                owner_locked,
                is_ctx: false,
                in_context,
            });
            if !child.children.is_empty() {
                let next = if self.ctx.is_some() && node_is_ctx { Some(cpath.clone()) } else { dentro.clone() };
                self.walk(child, &cpath, child.layer.as_deref(), m, next, hid);
            }
        }
    }
}

fn group_flag_locked(st: &State, path: &[usize]) -> bool {
    ingetrazo_ops::group::group_at(&st.scene.groups, path).map(|g| g.locked).unwrap_or(false)
}

/// What a placement's world arrays are made of.
#[derive(Clone, Debug, Default)]
pub struct Chunk {
    /// Index in `mesh.faces` of each kept face (hidden faces are not in).
    pub faces: Vec<usize>,
    pub areas: Vec<f64>,
    pub v0: Vec<P3>,
    pub e1: Vec<P3>,
    pub e2: Vec<P3>,
    /// The kept-face ordinal of each triangle.
    pub tri_ent: Vec<usize>,
    /// The hard edges (soft and hidden ones are not in), float32.
    pub edges: Vec<[[f32; 3]; 2]>,
    pub bbox: Option<(P3, P3)>,
}

/// `_group_chunk` of a prototype mesh in its own coordinates.
pub fn base_chunk(mesh: &Mesh, h: Hypot) -> Chunk {
    let mut c = Chunk::default();
    let mut tris: Vec<[P3; 3]> = Vec::new();
    for (fi, &fid) in mesh.faces.iter().enumerate() {
        let face = mesh.face(fid);
        if State::face_hidden(&face.attrs) {
            continue;
        }
        let i = c.faces.len();
        c.faces.push(fi);
        let outer: Vec<V3> = face.outer.iter().map(|&v| mesh.position(v)).collect();
        let holes: Vec<Vec<V3>> = face.holes.iter().map(|hl| hl.iter().map(|&v| mesh.position(v)).collect()).collect();
        if outer.len() < 3 {
            c.areas.push(0.0);
            continue;
        }
        let n_raw = meshface::newell(&outer);
        let ln = n_raw.length(h);
        let normal = if (ln as f64) > 1e-9 { n_raw.div(ln) } else { V3(0.0, 0.0, 1.0) };
        c.areas.push(0.5 * ln as f64);
        for t in triangulate(&outer, &holes, Some(normal), h) {
            tris.push([p3(t[0]), p3(t[1]), p3(t[2])]);
            c.tri_ent.push(i);
        }
    }
    for t in &tris {
        c.v0.push(t[0]);
        c.e1.push([t[1][0] - t[0][0], t[1][1] - t[0][1], t[1][2] - t[0][2]]);
        c.e2.push([t[2][0] - t[0][0], t[2][1] - t[0][1], t[2][2] - t[0][2]]);
    }
    for &eid in &mesh.edges {
        let e = mesh.edge(eid);
        if e.hidden || e.soft {
            continue;
        }
        let a = mesh.position(e.v0);
        let b = mesh.position(e.v1);
        c.edges.push([[a.0, a.1, a.2], [b.0, b.1, b.2]]);
    }
    // The world box (frustum culling): triangle corners cover every face,
    // hard-edge endpoints cover edge-only content.
    let mut lo = [f64::INFINITY; 3];
    let mut hi = [f64::NEG_INFINITY; 3];
    let mut any = false;
    for i in 0..c.v0.len() {
        let a = c.v0[i];
        let b = [a[0] + c.e1[i][0], a[1] + c.e1[i][1], a[2] + c.e1[i][2]];
        let d = [a[0] + c.e2[i][0], a[1] + c.e2[i][1], a[2] + c.e2[i][2]];
        for p in [a, b, d] {
            any = true;
            for k in 0..3 {
                lo[k] = lo[k].min(p[k]);
                hi[k] = hi[k].max(p[k]);
            }
        }
    }
    for e in &c.edges {
        for p in e {
            any = true;
            for k in 0..3 {
                lo[k] = lo[k].min(p[k] as f64);
                hi[k] = hi[k].max(p[k] as f64);
            }
        }
    }
    c.bbox = any.then_some((lo, hi));
    c
}

/// `_instance_chunk`: the prototype's chunk moved to the world by `m`.
pub fn instance_chunk(base: &Chunk, m: &M4) -> Chunk {
    let (rot, trans) = np_affine(m);
    let det = np_det3(&rot);
    let scale = det.abs().powf(2.0 / 3.0);
    let mut c = Chunk {
        faces: base.faces.clone(),
        areas: base.areas.iter().map(|a| a * scale).collect(),
        tri_ent: base.tri_ent.clone(),
        ..Chunk::default()
    };
    if !base.v0.is_empty() {
        c.v0 = affine_rows(&base.v0, &rot, &trans);
        c.e1 = rows_matmul(&base.e1, &rot);
        c.e2 = rows_matmul(&base.e2, &rot);
    }
    if !base.edges.is_empty() {
        let flat: Vec<P3> = base.edges.iter().flat_map(|e| [[e[0][0] as f64, e[0][1] as f64, e[0][2] as f64], [e[1][0] as f64, e[1][1] as f64, e[1][2] as f64]]).collect();
        let moved = affine_rows(&flat, &rot, &trans);
        c.edges = moved.chunks(2).map(|p| [[p[0][0] as f32, p[0][1] as f32, p[0][2] as f32], [p[1][0] as f32, p[1][1] as f32, p[1][2] as f32]]).collect();
    }
    if let Some((lo, hi)) = base.bbox {
        let mut corners = Vec::with_capacity(8);
        for x in [lo[0], hi[0]] {
            for y in [lo[1], hi[1]] {
                for z in [lo[2], hi[2]] {
                    corners.push([x, y, z]);
                }
            }
        }
        let tc = affine_rows(&corners, &rot, &trans);
        let mut l = [f64::INFINITY; 3];
        let mut u = [f64::NEG_INFINITY; 3];
        for p in &tc {
            for k in 0..3 {
                l[k] = l[k].min(p[k]);
                u[k] = u[k].max(p[k]);
            }
        }
        c.bbox = Some((l, u));
    }
    c
}

/// One entity of the index.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EntRef {
    /// A face of the loose mesh, by its index in `mesh.faces`.
    Loose(usize),
    /// A face of a placement's mesh: the placement (index in
    /// `PickIndex::places`) and the face's index in that mesh's `faces`.
    Placed { place: usize, face: usize },
}

/// A run of triangles with the box around them.
#[derive(Clone, Copy, Debug)]
pub struct Span {
    pub bbox: Option<(P3, P3)>,
    pub start: usize,
    pub n: usize,
}

#[derive(Default)]
pub struct PickIndex {
    pub ents: Vec<EntRef>,
    pub area: Vec<f64>,
    pub sel: Vec<bool>,
    pub vis: Vec<bool>,
    pub loose: Vec<bool>,
    /// Per entity: index into `places`, -1 for a loose face.
    pub place_of: Vec<i32>,
    /// The placements that made chunks, in index order.
    pub places: Vec<Place>,
    pub v0: Vec<P3>,
    pub e1: Vec<P3>,
    pub e2: Vec<P3>,
    pub tri_ent: Vec<usize>,
    pub spans: Vec<Span>,
    /// The loose mesh's edges: index in `mesh.edges`, ends, masks.
    pub edge_ix: Vec<usize>,
    pub edge_a: Vec<P3>,
    pub edge_b: Vec<P3>,
    pub edge_sel: Vec<bool>,
    pub edge_soft: Vec<bool>,
    /// The hard edges of the placements.
    pub gedge_a: Vec<P3>,
    pub gedge_b: Vec<P3>,
    pub gedge_gi: Vec<usize>,
    pub gedge_sel: Vec<bool>,
    /// The owner (what a click selects) of each group with edges here.
    pub gedge_groups: Vec<Path>,
    /// `_context_placements()`: the placements the context lets you touch.
    pub context: Vec<Place>,
    /// The hard edges of the placements on the screen of the last view
    /// asked (a hover asks more than once with the same camera).
    pub gscreen: GScreenCache,
}

impl PickIndex {
    pub fn has_tris(&self) -> bool {
        !self.v0.is_empty()
    }

    pub fn build(st: &State) -> PickIndex {
        PickIndex::build_cached(st, &mut ChunkCache::default())
    }

    /// [`PickIndex::build`] over a cache of the prototype chunks: a mesh the
    /// scene did not change since the last build is not triangulated again.
    pub fn build_cached(st: &State, cache: &mut ChunkCache) -> PickIndex {
        let h = st.h;
        let scene = st.scene;
        let mut idx = PickIndex::default();
        let mesh = scene.mesh(st.active);

        // The loose part.
        let mut loose_tris = 0usize;
        for (i, &fid) in mesh.faces.iter().enumerate() {
            let face = mesh.face(fid);
            let outer: Vec<V3> = face.outer.iter().map(|&v| mesh.position(v)).collect();
            idx.ents.push(EntRef::Loose(i));
            idx.vis.push(st.face_visible(&face.attrs));
            idx.sel.push(st.face_selectable(&face.attrs));
            idx.loose.push(true);
            idx.place_of.push(-1);
            if outer.len() < 3 {
                idx.area.push(0.0);
                continue;
            }
            let n = meshface::newell(&outer);
            idx.area.push(0.5 * n.length(h) as f64);
            let normal = if (n.length(h) as f64) > 1e-9 { n.normalized(h) } else { V3(0.0, 0.0, 1.0) };
            let holes: Vec<Vec<V3>> = face.holes.iter().map(|hl| hl.iter().map(|&v| mesh.position(v)).collect()).collect();
            for t in triangulate(&outer, &holes, Some(normal), h) {
                let (a, b, c) = (p3(t[0]), p3(t[1]), p3(t[2]));
                idx.v0.push(a);
                idx.e1.push([b[0] - a[0], b[1] - a[1], b[2] - a[2]]);
                idx.e2.push([c[0] - a[0], c[1] - a[1], c[2] - a[2]]);
                idx.tri_ent.push(i);
                loose_tris += 1;
            }
        }
        if loose_tris > 0 {
            idx.spans.push(Span { bbox: None, start: 0, n: loose_tris });
        }

        // Which placements the context lets you touch, and which enter.
        let all = placements(st);
        let ctx = st.ctx.as_deref();
        let context: Vec<Place> = match ctx {
            None => all.clone(),
            Some(_) => all.iter().filter(|g| g.in_context && ctx != Some(g.owner.as_slice())).cloned().collect(),
        };
        let candidates: Vec<&Place> = if ctx.is_none() || st.rest_hidden {
            context.iter().collect()
        } else {
            all.iter().filter(|g| !g.is_ctx && ctx != Some(g.owner.as_slice())).collect()
        };
        // With nothing open every placement is in the context (it is `all`).
        let context_paths: Option<std::collections::HashSet<&Path>> = ctx.map(|_| context.iter().map(|c| &c.path).collect());
        let tocable = |g: &Place| context_paths.as_ref().is_none_or(|s| s.contains(&g.path));

        // The placements that enter, in index order, with their flags.
        let mut entering: Vec<(&Place, bool, bool, bool)> = Vec::new();
        for g in candidates {
            if g.billboard {
                continue;
            }
            let gvis = !g.hidden && st.layer_visible(g.layer.as_deref());
            let gsnap = group_selectable(st, g);
            let gsel = gsnap && tocable(g);
            if !(gvis || gsnap) {
                continue;
            }
            entering.push((g, gvis, gsnap, gsel));
        }
        // The prototypes the cache does not hold (or holds stale) are made
        // first, all at once; then each placement's moved chunk is made in
        // parallel, a batch at a time, and appended in index order.
        let wanted: Vec<MeshId> = entering.iter().map(|e| e.0.mesh).collect();
        cache.prebuild(scene, &wanted, h);
        struct Entry<'a> {
            g: &'a Place,
            gvis: bool,
            gsnap: bool,
            gsel: bool,
            base: std::sync::Arc<Chunk>,
        }
        let mut entries: Vec<Entry> = Vec::with_capacity(entering.len());
        for (g, gvis, gsnap, gsel) in entering {
            let base = cache.base(scene, g.mesh, h);
            if base.faces.is_empty() && base.edges.is_empty() {
                continue;
            }
            entries.push(Entry { g, gvis, gsnap, gsel, base });
        }
        // Room for everything first: the arrays are big.
        let (nf, nt, ne) = entries.iter().fold((0, 0, 0), |a, e| (a.0 + e.base.faces.len(), a.1 + e.base.v0.len(), a.2 + e.base.edges.len()));
        idx.ents.reserve(nf);
        idx.area.reserve(nf);
        idx.vis.reserve(nf);
        idx.sel.reserve(nf);
        idx.loose.reserve(nf);
        idx.place_of.reserve(nf);
        idx.v0.reserve(nt);
        idx.e1.reserve(nt);
        idx.e2.reserve(nt);
        idx.tri_ent.reserve(nt);
        idx.gedge_a.reserve(ne);
        idx.gedge_b.reserve(ne);
        idx.gedge_gi.reserve(ne);
        idx.gedge_sel.reserve(ne);
        for batch in entries.chunks(PLACEMENT_BATCH) {
            use rayon::prelude::*;
            let moved: Vec<Option<Chunk>> = batch.par_iter().map(|e| e.g.xform.as_ref().map(|m| instance_chunk(&e.base, m))).collect();
            for (e, mv) in batch.iter().zip(moved) {
            let g: &Place = e.g;
            let (gvis, gsnap, gsel) = (e.gvis, e.gsnap, e.gsel);
            let chunk: &Chunk = mv.as_ref().unwrap_or(&e.base);
            let pi = idx.places.len();
            idx.places.push(g.clone());
            let off = idx.ents.len();
            for (k, &fi) in chunk.faces.iter().enumerate() {
                idx.ents.push(EntRef::Placed { place: pi, face: fi });
                idx.area.push(chunk.areas[k]);
                idx.vis.push(gvis);
                idx.sel.push(gsel);
                idx.loose.push(false);
                idx.place_of.push(pi as i32);
            }
            if !chunk.v0.is_empty() {
                let start = idx.v0.len();
                idx.v0.extend_from_slice(&chunk.v0);
                idx.e1.extend_from_slice(&chunk.e1);
                idx.e2.extend_from_slice(&chunk.e2);
                idx.tri_ent.extend(chunk.tri_ent.iter().map(|t| t + off));
                idx.spans.push(Span { bbox: chunk.bbox, start, n: chunk.v0.len() });
            }
            if gsnap && !chunk.edges.is_empty() {
                let gi = idx.gedge_groups.len();
                for e in &chunk.edges {
                    idx.gedge_a.push([e[0][0] as f64, e[0][1] as f64, e[0][2] as f64]);
                    idx.gedge_b.push([e[1][0] as f64, e[1][1] as f64, e[1][2] as f64]);
                    idx.gedge_gi.push(gi);
                    idx.gedge_sel.push(gsel);
                }
                idx.gedge_groups.push(g.owner.clone());
            }
            }
        }

        // The loose edges.
        for (i, &eid) in mesh.edges.iter().enumerate() {
            let e = mesh.edge(eid);
            let a = p3(mesh.position(e.v0));
            let b = p3(mesh.position(e.v1));
            idx.edge_ix.push(i);
            idx.edge_a.push(a);
            idx.edge_b.push(b);
            idx.edge_soft.push(e.soft);
            idx.edge_sel.push(st.layer_selectable(e.layer.as_deref()) && (!e.hidden || st.show_hidden_geometry()));
        }
        idx.context = context;
        idx
    }
}

/// How many placements are moved to the world at once (in parallel) before
/// they are appended to the index.
const PLACEMENT_BATCH: usize = 256;

/// Both ends of every placement hard edge on the screen.
pub struct GScreen {
    pub ax: Vec<f64>,
    pub ay: Vec<f64>,
    pub oka: Vec<bool>,
    pub bx: Vec<f64>,
    pub by: Vec<f64>,
    pub okb: Vec<bool>,
    grid: crate::screen_grid::ScreenGrid,
}

impl GScreen {
    /// Call `f` with the index of every edge that may come within `lim`
    /// pixels of `(x, y)`, in index order (every edge when the pixel is off
    /// the grid): the others are farther than `lim`.
    pub fn for_each_near(&self, x: f64, y: f64, lim: f64, mut f: impl FnMut(usize)) {
        match self.grid.candidates(x, y, lim + 1e-6) {
            Some(c) => c.into_iter().for_each(|i| f(i as usize)),
            None => (0..self.ax.len()).for_each(f),
        }
    }
}

type GScreenKey = ([u64; 16], i32, i32);

/// The last [`GScreen`] and the view it is for.
#[derive(Default)]
pub struct GScreenCache(std::sync::Mutex<Option<(GScreenKey, std::sync::Arc<GScreen>)>>);

impl GScreenCache {
    /// The projection of the hard edges of `idx` in `view`, kept while the
    /// view stands.
    pub fn get(&self, idx: &PickIndex, view: &super::camera::View) -> std::sync::Arc<GScreen> {
        let key = view.key();
        let mut slot = self.0.lock().unwrap_or_else(|e| e.into_inner());
        if let Some((k, g)) = slot.as_ref() {
            if *k == key {
                return g.clone();
            }
        }
        let (ax, ay, oka) = view.project_px(&idx.gedge_a);
        let (bx, by, okb) = view.project_px(&idx.gedge_b);
        let grid = crate::screen_grid::ScreenGrid::build(ax.len(), view.w as f64, view.h as f64, |i| {
            (oka[i] && okb[i]).then(|| [ax[i].min(bx[i]), ay[i].min(by[i]), ax[i].max(bx[i]), ay[i].max(by[i])])
        });
        let g = std::sync::Arc::new(GScreen { ax, ay, oka, bx, by, okb, grid });
        *slot = Some((key, g.clone()));
        g
    }
}

/// The prototype chunks of a model by mesh, each valid while its mesh's
/// serial and arena sizes stand (every mutation bumps the serial): an edit
/// rebuilds only the meshes it touched. Drop it with [`ChunkCache::clear`]
/// when the scene object is replaced (serials restart from zero).
#[derive(Default)]
pub struct ChunkCache {
    chunks: std::collections::HashMap<u32, ((u64, (usize, usize, usize)), std::sync::Arc<Chunk>)>,
}

impl ChunkCache {
    pub fn clear(&mut self) {
        self.chunks.clear();
    }

    /// Make, in parallel, the chunks of `ids` the cache does not hold fresh.
    fn prebuild(&mut self, scene: &ingetrazo_model::scene::Scene, ids: &[MeshId], h: Hypot) {
        use rayon::prelude::*;
        let mut todo: Vec<MeshId> = Vec::new();
        let mut seen = std::collections::HashSet::new();
        for &id in ids {
            let m = scene.mesh(id);
            let stamp = (m.serial, m.arena_sizes());
            let fresh = self.chunks.get(&id.0).is_some_and(|(s, _)| *s == stamp);
            if !fresh && seen.insert(id.0) {
                todo.push(id);
            }
        }
        if todo.len() < 2 {
            return;
        }
        let built: Vec<(MeshId, (u64, (usize, usize, usize)), Chunk)> = todo
            .par_iter()
            .map(|&id| {
                let m = scene.mesh(id);
                (id, (m.serial, m.arena_sizes()), base_chunk(m, h))
            })
            .collect();
        for (id, stamp, c) in built {
            self.chunks.insert(id.0, (stamp, std::sync::Arc::new(c)));
        }
    }

    fn base(&mut self, scene: &ingetrazo_model::scene::Scene, id: MeshId, h: Hypot) -> std::sync::Arc<Chunk> {
        let mesh = scene.mesh(id);
        let stamp = (mesh.serial, mesh.arena_sizes());
        if let Some((s, c)) = self.chunks.get(&id.0) {
            if *s == stamp {
                return c.clone();
            }
        }
        let c = std::sync::Arc::new(base_chunk(mesh, h));
        self.chunks.insert(id.0, (stamp, c.clone()));
        c
    }
}

/// `Scene.entity_selectable(placement)`.
pub fn group_selectable(st: &State, g: &Place) -> bool {
    if g.hidden && !st.show_hidden_objects() {
        return false;
    }
    if g.self_locked || g.owner_locked {
        return false;
    }
    st.layer_selectable(g.layer.as_deref())
}
