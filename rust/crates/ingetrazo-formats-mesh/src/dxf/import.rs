// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! DXF import: `formats/dxf_in.py`. Model space is sorted into one bucket
//! per CAD layer, block definitions into prototype buckets, then every
//! bucket becomes a welded mesh and every INSERT an instance placement.
//!
//! Where Python runs `run_stitch(mesh, seed, new_faces, coplanar_merge=True)`
//! (a layer of at most 400 faces) this calls the [`Stitch`] hook, which is
//! the edit engine (M3). A bigger layer is fused by `fuse_coplanar_loops`
//! (`ingetrazo_geom::fuse`) exactly as Python does.

use std::collections::{BTreeMap, HashSet};

use ingetrazo_geom::fuse::{self, Region};
use ingetrazo_math::{Hypot, M4, V3};
use ingetrazo_model::layers::Layer;
use ingetrazo_model::mesh::{key, EdgeFlags, FaceId, Key, Mesh};
use ingetrazo_model::scene::{Group, MeshId, Scene};

use super::entities::{contents_matrix, entity_face, entity_segments, explode, Ent, Geom};
use super::geom::P;
use super::read::{read_dxf, Doc};
use crate::obj::{Stitch, MAX_FUSE_LOOPS};

/// Metres per drawing unit by `$INSUNITS` code.
pub fn insunits_to_m(code: i64) -> Option<f64> {
    Some(match code {
        1 => 0.0254,
        2 => 0.3048,
        3 => 1609.344,
        4 => 0.001,
        5 => 0.01,
        6 => 1.0,
        7 => 1000.0,
        10 => 0.9144,
        _ => return None,
    })
}

/// Coordinates beyond this (metres, after unit scale) trigger recentring.
pub const RECENTER_BEYOND_M: f64 = 100_000.0;

/// Maximum sagitta in metres.
const SAGITTA_M: f64 = 0.01;
const SANE_MEDIAN_M: (f64, f64) = (0.05, 80.0);
const MAX_INSERT_DEPTH: usize = 8;

const NO_GEOMETRY: &str = "no importable geometry (lines, polylines, arcs, circles, ellipses, splines or faces) in the model space";

/// `read_unit_scale(doc)`: `(metres_per_unit, code)`.
pub fn read_unit_scale(doc: &Doc) -> (Option<f64>, i64) {
    let code = doc.insunits();
    (insunits_to_m(code), code)
}

/// `suggest_unit_scale(doc)`: the header's unit unless the drawing's own
/// median entity size rules it out.
pub fn suggest_unit_scale(doc: &Doc) -> (Option<f64>, i64) {
    let (header_scale, code) = read_unit_scale(doc);
    let mut sizes: Vec<f64> = Vec::new();
    for raw in &doc.modelspace {
        if sizes.len() >= 4000 {
            break;
        }
        match Ent::parse(raw, doc).geom {
            Geom::Line(a, b) => sizes.push(b.sub(a).magnitude()),
            Geom::Circle { radius, .. } | Geom::Arc { radius, .. } => sizes.push(2.0 * radius),
            Geom::LwPoly { pts, .. } => {
                for k in 0..pts.len().saturating_sub(1) {
                    let dx = pts[k + 1].0 - pts[k].0;
                    let dy = pts[k + 1].1 - pts[k].1;
                    sizes.push((dx * dx + dy * dy).powf(0.5));
                }
            }
            _ => {}
        }
    }
    sizes.retain(|&s| s > 1e-9);
    if sizes.is_empty() {
        return (header_scale, code);
    }
    sizes.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median = sizes[sizes.len() / 2];
    let sane = |scale: f64| SANE_MEDIAN_M.0 <= median * scale && median * scale <= SANE_MEDIAN_M.1;
    if let Some(h) = header_scale {
        if sane(h) {
            return (Some(h), code);
        }
    }
    for candidate in [1.0, 0.01, 0.001, 0.0254, 0.3048] {
        if sane(candidate) {
            return (Some(candidate), code);
        }
    }
    (header_scale, code)
}

#[derive(Default)]
struct Bucket {
    /// (point run, curved)
    runs: Vec<(Vec<P>, bool)>,
    faces: Vec<Vec<P>>,
}

#[derive(Default)]
struct Counts {
    entities: usize,
    skipped: BTreeMap<String, usize>,
}

/// `_iter_flat(space, depth)` over already-typed entities: block references
/// explode in place (a child on layer "0" inherits its reference's layer).
fn iter_flat(doc: &Doc, space: Vec<Ent>, depth: usize, out: &mut Vec<(Ent, String)>) {
    for e in space {
        let layer = e.layer.clone();
        match &e.geom {
            Geom::Insert { .. } if depth < MAX_INSERT_DEPTH => {
                let Ok((name, m)) = contents_matrix(doc, &e.geom) else { continue };
                let Some(block) = doc.block(&name) else { continue };
                // A broken block loses only itself.
                let Ok(children) = explode(doc, block, &m) else { continue };
                let mut sub_out = Vec::new();
                iter_flat(doc, children, depth + 1, &mut sub_out);
                for (child, sub) in sub_out {
                    let l = if sub == "0" { layer.clone() } else { sub };
                    out.push((child, l));
                }
            }
            Geom::Dropped => {}
            _ => out.push((e, layer)),
        }
    }
}

/// `_bucket_one(e, layer, ...)`.
fn bucket_one(e: &Ent, layer: &str, sagitta: f64, counts: &mut Counts, count_entity: bool, per_layer: &mut BTreeMap<String, Bucket>) {
    if let Some(loop_) = entity_face(&e.geom) {
        if count_entity {
            counts.entities += 1;
        }
        per_layer.entry(layer.to_string()).or_default().faces.push(loop_);
        return;
    }
    let Some((runs, curved)) = entity_segments(&e.geom, sagitta) else {
        if !matches!(e.geom, Geom::Silent | Geom::Dropped) {
            *counts.skipped.entry(e.kind.clone()).or_insert(0) += 1;
        }
        return;
    };
    if runs.is_empty() {
        return;
    }
    if count_entity {
        counts.entities += 1;
    }
    per_layer.entry(layer.to_string()).or_default().runs.extend(runs.into_iter().map(|r| (r, curved)));
}

fn v3(p: P, scale: f64, offset: [f64; 3]) -> V3 {
    V3((p.x * scale - offset[0]) as f32, (p.y * scale - offset[1]) as f32, (p.z * scale - offset[2]) as f32)
}

/// `_build_mesh(bucket, scale, offset)`.
fn build_mesh(bucket: &Bucket, scale: f64, offset: [f64; 3], curve_ids: &mut i64, h: Hypot, stitch: &mut Option<Stitch>) -> Mesh {
    let mut mesh = Mesh::new();
    let mut pts: Vec<V3> = Vec::new();
    let mut ia: Vec<usize> = Vec::new();
    let mut ib: Vec<usize> = Vec::new();
    let mut flags: Vec<EdgeFlags> = Vec::new();
    for (run, curved) in &bucket.runs {
        let curve_id = if *curved && run.len() > 2 {
            *curve_ids += 1;
            Some(*curve_ids)
        } else {
            None
        };
        let base = pts.len();
        for &p in run {
            pts.push(v3(p, scale, offset));
        }
        for k in 0..run.len() - 1 {
            ia.push(base + k);
            ib.push(base + k + 1);
            flags.push(EdgeFlags { soft: false, curve: curve_id, layer: None, hidden: false });
        }
    }
    if !pts.is_empty() {
        let (vobjs, ids) = mesh.bulk_weld(&pts);
        let a: Vec<usize> = ia.iter().map(|&i| ids[i]).collect();
        let b: Vec<usize> = ib.iter().map(|&i| ids[i]).collect();
        mesh.add_edges_welded(&vobjs, &a, &b, Some(&flags));
    }
    if bucket.faces.len() > MAX_FUSE_LOOPS {
        // A big layer of 3DFACEs: one near-linear fuse pass instead of
        // run_stitch's coplanar merge.
        let mut flat: Vec<V3> = Vec::new();
        let mut starts = vec![0usize];
        for loop_ in &bucket.faces {
            flat.extend(loop_.iter().map(|&p| v3(p, scale, offset)));
            starts.push(flat.len());
        }
        let n = bucket.faces.len();
        let sig = vec![0i64; n];
        let group = vec![-1i64; n];
        let rank = vec![1u8; n];
        let uvw: Vec<Vec<f64>> = vec![Vec::new(); n];
        let inp = fuse::Input { pts: &flat, loop_start: &starts, sig: &sig, tex_group: &group, rank: &rank, uvw: &uvw };
        for region in fuse::fuse(&inp, 0.99999, false, h) {
            match region {
                Region::Pass(lps) => {
                    for lp in lps {
                        mesh.add_face(&flat[starts[lp]..starts[lp + 1]], &[]);
                    }
                }
                Region::Traced(rings, _lps) => {
                    let pos = |ring: &Vec<usize>| ring.iter().map(|&i| flat[i]).collect::<Vec<V3>>();
                    let outer = pos(&rings[0]);
                    let holes: Vec<Vec<V3>> = rings[1..].iter().map(pos).collect();
                    mesh.add_face(&outer, &holes);
                }
            }
        }
        return mesh;
    }
    if !bucket.faces.is_empty() {
        let mut added: HashSet<FaceId> = HashSet::new();
        let mut seed: HashSet<Key> = HashSet::new();
        for loop_ in &bucket.faces {
            let corners: Vec<V3> = loop_.iter().map(|&p| v3(p, scale, offset)).collect();
            // Python's add_face raises on a NaN or infinite corner (its weld
            // key is an int) after welding the corners before it; the import
            // skips such a polygon.
            if let Some(bad) = corners.iter().position(|c| !(c.0.is_finite() && c.1.is_finite() && c.2.is_finite())) {
                for &c in &corners[..bad] {
                    mesh.vertex(c);
                }
                continue;
            }
            let f = mesh.add_face(&corners, &[]);
            added.insert(f);
            for &c in &corners {
                seed.insert(key(c));
            }
        }
        if !added.is_empty() {
            if let Some(stitch) = stitch.as_mut() {
                stitch(&mut mesh, &seed, &added);
            }
        }
    }
    mesh
}

/// `_fold_xform(m44, scale, offset)`: ezdxf's row-vector matrix as the
/// QMatrix4x4 of a prototype in metres.
fn fold_xform(m: &super::geom::Mat, scale: f64, offset: [f64; 3]) -> M4 {
    let r = |i: usize, j: usize| m.m[i * 4 + j];
    let f = |x: f64| x as f32;
    M4::from_rows([
        f(r(0, 0)), f(r(1, 0)), f(r(2, 0)), f(r(3, 0) * scale - offset[0]),
        f(r(0, 1)), f(r(1, 1)), f(r(2, 1)), f(r(3, 1) * scale - offset[1]),
        f(r(0, 2)), f(r(1, 2)), f(r(2, 2)), f(r(3, 2) * scale - offset[2]),
        0.0, 0.0, 0.0, 1.0,
    ])
}

/// What `load_dxf` returns.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct DxfStats {
    pub entities: usize,
    pub edges: usize,
    pub faces: usize,
    pub groups: usize,
    /// Block name and how many times it was placed, in first-use order.
    pub components: Vec<(String, usize)>,
    pub skipped: BTreeMap<String, usize>,
    pub offset: Option<[f64; 3]>,
}

/// `load_dxf(scene, path, scale=..., doc=..., name=...)` on file contents.
/// `stem` is the file name without its extension, the layer-0 group's name
/// unless `name` is given. `stitch` stands where Python runs `run_stitch`.
pub fn load_dxf_bytes(scene: &mut Scene, data: &[u8], stem: &str, scale: f64, name: Option<&str>, h: Hypot, stitch: Option<Stitch>) -> Result<DxfStats, String> {
    let doc = read_dxf(data)?;
    load_dxf_doc(scene, &doc, stem, scale, name, h, stitch)
}

/// `load_dxf` on an opened document.
pub fn load_dxf_doc(scene: &mut Scene, doc: &Doc, stem: &str, scale: f64, name: Option<&str>, h: Hypot, mut stitch: Option<Stitch>) -> Result<DxfStats, String> {
    let sagitta = (SAGITTA_M / scale).max(1e-6);

    // ---- Pass 1: model space into buckets and component placements ----
    let mut per_layer: BTreeMap<String, Bucket> = BTreeMap::new();
    let mut counts = Counts::default();
    let mut placements: Vec<(String, super::geom::Mat, String)> = Vec::new();
    for raw in &doc.modelspace {
        let e = Ent::parse(raw, doc);
        match &e.geom {
            Geom::Insert { .. } => {
                // A reference whose matrix fails loses only itself.
                if let Ok((name, m)) = contents_matrix(doc, &e.geom) {
                    placements.push((name, m, e.layer.clone()));
                    counts.entities += 1;
                }
            }
            Geom::Dropped => {}
            _ => bucket_one(&e, &e.layer, sagitta, &mut counts, true, &mut per_layer),
        }
    }

    // ---- Block prototypes: one bucket per used definition ----
    let mut proto_names: Vec<String> = Vec::new();
    let mut proto_buckets: Vec<Bucket> = Vec::new();
    for (block_name, _m, _layer) in &placements {
        if proto_names.contains(block_name) {
            continue;
        }
        let Some(block) = doc.block(block_name) else { continue };
        let space: Vec<Ent> = block.entities.iter().map(|r| Ent::parse(r, doc)).collect();
        let mut flat = Vec::new();
        iter_flat(doc, space, 0, &mut flat);
        let mut one: BTreeMap<String, Bucket> = BTreeMap::new();
        let mut scratch = Counts { entities: 0, skipped: std::mem::take(&mut counts.skipped) };
        for (child, _sub) in &flat {
            bucket_one(child, "0", sagitta, &mut scratch, false, &mut one);
        }
        counts.skipped = scratch.skipped;
        proto_names.push(block_name.clone());
        proto_buckets.push(one.remove("0").unwrap_or_default());
    }

    if per_layer.is_empty() && !proto_buckets.iter().any(|b| !b.runs.is_empty() || !b.faces.is_empty()) {
        return Err(NO_GEOMETRY.to_string());
    }

    // ---- Measure for the UTM recentring (placements count too) ----
    let mut lo = [f64::INFINITY; 3];
    let mut hi = [f64::NEG_INFINITY; 3];
    let mut absorb = |x: f64, y: f64, z: f64| {
        for (i, c) in [x, y, z].into_iter().enumerate() {
            if c < lo[i] {
                lo[i] = c;
            }
            if c > hi[i] {
                hi[i] = c;
            }
        }
    };
    for bucket in per_layer.values() {
        for (run, _c) in &bucket.runs {
            for p in run {
                absorb(p.x * scale, p.y * scale, p.z * scale);
            }
        }
        for loop_ in &bucket.faces {
            for p in loop_ {
                absorb(p.x * scale, p.y * scale, p.z * scale);
            }
        }
    }
    for (_n, m, _l) in &placements {
        absorb(m.m[12] * scale, m.m[13] * scale, m.m[14] * scale);
    }
    let mut offset = [0.0f64; 3];
    let far = lo.iter().chain(hi.iter()).map(|c| c.abs()).fold(f64::NEG_INFINITY, |a, b| if b > a { b } else { a });
    if lo[0] < hi[0] && far > RECENTER_BEYOND_M {
        offset = [lo[0].trunc(), lo[1].trunc(), lo[2].trunc()];
    }

    // ---- Build: one tagged group per layer, one proto per block ----
    let mut groups_made = 0usize;
    let mut edges_made = 0usize;
    let mut faces_made = 0usize;
    let mut curve_ids = 0i64;

    fn ensure_tag(scene: &mut Scene, layer: &str) {
        if layer == "0" {
            return;
        }
        let known = scene.layers.iter().any(|l| l.get("name").and_then(|n| if let ingetrazo_formats::pyjson::Value::Str(s) = n { Some(s.as_str()) } else { None }) == Some(layer));
        if !known {
            scene.layers.push(Layer::new(layer).to_dict());
        }
    }

    for (layer, bucket) in &per_layer {
        let mesh = build_mesh(bucket, scale, offset, &mut curve_ids, h, &mut stitch);
        if mesh.edges.is_empty() && mesh.faces.is_empty() {
            continue;
        }
        edges_made += mesh.edges.len();
        faces_made += mesh.faces.len();
        scene.meshes.push(mesh);
        let id = MeshId(scene.meshes.len() as u32 - 1);
        let group_name = if layer != "0" { layer.clone() } else { name.map(str::to_string).filter(|n| !n.is_empty()).unwrap_or_else(|| stem.to_string()) };
        let mut group = Group::new(id, group_name);
        if layer != "0" {
            group.layer = Some(layer.clone());
            ensure_tag(scene, layer);
        }
        scene.groups.push(group);
        groups_made += 1;
    }

    let mut protos: Vec<(String, MeshId)> = Vec::new();
    for (block_name, bucket) in proto_names.iter().zip(&proto_buckets) {
        let mesh = build_mesh(bucket, scale, [0.0; 3], &mut curve_ids, h, &mut stitch);
        if !mesh.edges.is_empty() || !mesh.faces.is_empty() {
            scene.meshes.push(mesh);
            protos.push((block_name.clone(), MeshId(scene.meshes.len() as u32 - 1)));
        }
    }
    let mut components: Vec<(String, usize)> = Vec::new();
    for (block_name, m, layer) in &placements {
        let Some((_, id)) = protos.iter().find(|(n, _)| n == block_name) else { continue };
        let mut inst = Group::new(*id, block_name.clone());
        inst.xform = Some(fold_xform(m, scale, offset));
        if layer != "0" {
            inst.layer = Some(layer.clone());
            ensure_tag(scene, layer);
        }
        scene.groups.push(inst);
        groups_made += 1;
        let n = match components.iter_mut().find(|(c, _)| c == block_name) {
            Some(entry) => {
                entry.1 += 1;
                entry.1
            }
            None => {
                components.push((block_name.clone(), 1));
                1
            }
        };
        if n == 1 {
            edges_made += scene.meshes[id.0 as usize].edges.len();
        }
    }

    if groups_made == 0 {
        return Err(NO_GEOMETRY.to_string());
    }
    Ok(DxfStats {
        entities: counts.entities,
        edges: edges_made,
        faces: faces_made,
        groups: groups_made,
        components,
        skipped: counts.skipped,
        offset: if offset != [0.0; 3] { Some(offset) } else { None },
    })
}
