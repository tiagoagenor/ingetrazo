// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! Face-me billboards (2D people and trees): `Viewport._billboard_quad`,
//! `_faceme_base` and `_draw_faceme_mesh`. A billboard is a quad (a textured
//! cutout) or a flat silhouette mesh that turns about the vertical axis
//! through its feet to face the camera, every frame.

use ingetrazo_formats::pyjson::Value;
use ingetrazo_geom::facegeom::face_geometry;
use ingetrazo_math::{M4, V3};
use ingetrazo_model::mesh::Mesh;
use ingetrazo_model::pyval::truthy;
use ingetrazo_model::scene::{Group, MeshId, Scene};

use crate::bake::{face_uv_axes, FaceVertex, LayerTable};
use crate::plan::GroupPath;
use crate::style::DEFAULT_FACE_COLOR;

#[derive(Clone, Debug)]
pub struct Billboard {
    pub path: GroupPath,
    pub mesh: MeshId,
    pub matrix: Option<M4>,
    /// `billboard == "mesh"`: the silhouette geometry turns, not a quad.
    pub silhouette: bool,
}

/// The visible billboards of a scene, with their composed matrices.
pub fn collect(scene: &Scene, layers: &LayerTable) -> Vec<Billboard> {
    fn walk(g: &Group, path: &mut GroupPath, base: Option<M4>, layers: &LayerTable, out: &mut Vec<Billboard>) {
        let m = match (g.xform, base) {
            (m, None) => m,
            (None, Some(b)) => Some(b),
            (Some(x), Some(b)) => Some(b.mul(&x)),
        };
        if g.hidden || !layers.visible(g.layer.as_deref()) {
            return;
        }
        if let Some(b) = &g.billboard {
            if truthy(b) {
                out.push(Billboard {
                    path: path.clone(),
                    mesh: g.mesh,
                    matrix: m,
                    silhouette: matches!(b, Value::Str(s) if s == "mesh"),
                });
            }
        }
        for (i, c) in g.children.iter().enumerate() {
            path.push(i);
            walk(c, path, m, layers, out);
            path.pop();
        }
    }
    let mut out = Vec::new();
    for (i, g) in scene.groups.iter().enumerate() {
        walk(g, &mut vec![i], None, layers, &mut out);
    }
    out
}

/// A billboard's triangles for this frame, in world coordinates.
#[derive(Default)]
pub struct BillboardGeom {
    /// Per texture path.
    pub textured: Vec<(String, Vec<FaceVertex>)>,
    /// Per solid colour.
    pub colored: Vec<([f32; 3], Vec<FaceVertex>)>,
}

fn num(v: &Value) -> Option<f64> {
    match v {
        Value::Float(f) => Some(*f),
        Value::Int(s) => s.parse().ok(),
        _ => None,
    }
}

fn world_positions(mesh: &Mesh, m: Option<&M4>) -> Vec<V3> {
    mesh.vertices.iter().map(|&v| m.map_or(mesh.position(v), |m| m.map(mesh.position(v)))).collect()
}

fn texture_path(attrs: &[(String, Value)]) -> Option<(&Value, &str)> {
    match attrs.iter().find(|(k, _)| k == "texture") {
        Some((_, t @ Value::Object(_))) => match t.get("path") {
            Some(Value::Str(p)) if !p.is_empty() => Some((t, p.as_str())),
            _ => None,
        },
        _ => None,
    }
}

/// `_faceme_dir`: the direction a sprite at `anchor` turns toward.
fn facing(anchor: V3, eye: V3, target: V3, perspective: bool) -> V3 {
    if perspective {
        eye.sub(anchor)
    } else {
        eye.sub(target)
    }
}

/// Build the geometry of one billboard for the camera `eye`/`target`.
pub fn build(scene: &Scene, b: &Billboard, eye: V3, target: V3, perspective: bool) -> Option<BillboardGeom> {
    let mesh = scene.mesh(b.mesh);
    let m = b.matrix.as_ref();
    let h = mesh.hypot;
    let wpos = world_positions(mesh, m);
    if wpos.is_empty() {
        return None;
    }
    let (mut lo, mut hi) = ([f32::MAX; 3], [f32::MIN; 3]);
    for p in &wpos {
        for (k, c) in [p.0, p.1, p.2].into_iter().enumerate() {
            lo[k] = lo[k].min(c);
            hi[k] = hi[k].max(c);
        }
    }
    // The anchor is the figure's feet.
    let anchor = V3((lo[0] + hi[0]) / 2.0, (lo[1] + hi[1]) / 2.0, lo[2]);
    let mut out = BillboardGeom::default();
    if !b.silhouette {
        // The simple quad of the first textured face.
        let first = mesh.faces.iter().find_map(|&f| texture_path(&mesh.face(f).attrs).map(|(_, p)| p.to_string()))?;
        let w = ((hi[0] - lo[0]) as f64).hypot((hi[1] - lo[1]) as f64) as f32;
        let ht = hi[2] - lo[2];
        if (w as f64) < 1e-9 || (ht as f64) < 1e-9 {
            return None;
        }
        let d = facing(anchor, eye, target, perspective);
        let mut d = V3(d.0, d.1, 0.0);
        if d.length(h) < 1e-6 {
            d = V3(1.0, 0.0, 0.0);
        }
        let d = d.normalized(h);
        let r = V3(-d.1, d.0, 0.0);
        let up = V3(0.0, 0.0, 1.0);
        let c0 = anchor.sub(r.mul(w / 2.0));
        let c1 = anchor.add(r.mul(w / 2.0));
        let corners = [c0, c1, c1.add(up.mul(ht)), c0.add(up.mul(ht))];
        let uvs = [[0.0f32, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        let verts: Vec<FaceVertex> = [0usize, 1, 2, 0, 2, 3]
            .iter()
            .map(|&i| FaceVertex {
                pos: [corners[i].0, corners[i].1, corners[i].2],
                uv: uvs[i],
                rgba: [255; 4],
                misc: crate::bake::pack_misc(0, 1.0, 32),
            })
            .collect();
        out.textured.push((first, verts));
        return Some(out);
    }

    // The silhouette mesh: faces in world space, turned about the anchor.
    let mut tri_sets: Vec<(Option<(String, Value)>, [f32; 3], Vec<[V3; 3]>, V3)> = Vec::new();
    let mut n0: Option<(f64, V3)> = None;
    for &fid in &mesh.faces {
        let face = mesh.face(fid);
        if face.outer.len() < 3 {
            continue;
        }
        let mut pos: Vec<V3> = Vec::new();
        let mut rings_idx: Vec<Vec<u32>> = Vec::new();
        for ring in std::iter::once(&face.outer).chain(face.holes.iter()) {
            let start = pos.len() as u32;
            for &v in ring {
                let p = mesh.position(v);
                pos.push(m.map_or(p, |m| m.map(p)));
            }
            rings_idx.push((start..pos.len() as u32).collect());
        }
        let rings: Vec<&[u32]> = rings_idx.iter().map(|r| r.as_slice()).collect();
        let g = face_geometry(&pos, &rings, h, true);
        let area = 0.5 * g.ln as f64;
        if n0.is_none_or(|(a, _)| area > a) {
            n0 = Some((area, g.normal));
        }
        let tris: &[[u32; 3]] = g.tris_unit.as_deref().unwrap_or(&g.tris);
        let t3: Vec<[V3; 3]> = tris.iter().map(|t| [pos[t[0] as usize], pos[t[1] as usize], pos[t[2] as usize]]).collect();
        match texture_path(&face.attrs) {
            Some((tex, path)) => tri_sets.push((Some((path.to_string(), tex.clone())), [0.0; 3], t3, g.normal)),
            None => {
                let col = match face.attrs.iter().find(|(k, _)| k == "color") {
                    Some((_, Value::Array(a))) if a.len() >= 3 => {
                        [num(&a[0]).unwrap_or(0.0) as f32, num(&a[1]).unwrap_or(0.0) as f32, num(&a[2]).unwrap_or(0.0) as f32]
                    }
                    _ => DEFAULT_FACE_COLOR,
                };
                tri_sets.push((None, col, t3, g.normal));
            }
        }
    }
    let (_, n) = n0?;
    let mut nh = [n.0 as f64, n.1 as f64];
    let ln = nh[0].hypot(nh[1]);
    if ln <= 1e-9 {
        return None;
    }
    nh = [nh[0] / ln, nh[1] / ln];
    let fd = facing(anchor, eye, target, perspective);
    let mut d = [fd.0 as f64, fd.1 as f64];
    let l = d[0].hypot(d[1]);
    d = if l < 1e-9 { nh } else { [d[0] / l, d[1] / l] };
    let cos = nh[0] * d[0] + nh[1] * d[1];
    let sin = nh[0] * d[1] - nh[1] * d[0];
    let rot = |p: V3| {
        let (x, y) = (p.0 as f64 - anchor.0 as f64, p.1 as f64 - anchor.1 as f64);
        V3((anchor.0 as f64 + cos * x - sin * y) as f32, (anchor.1 as f64 + sin * x + cos * y) as f32, p.2)
    };
    for (tex, col, tris, normal) in tri_sets {
        match tex {
            Some((path, tdict)) => {
                let (gu, cu, gv, cv) = face_uv_axes(&tdict, normal);
                let verts: Vec<FaceVertex> = tris
                    .iter()
                    .flat_map(|t| t.iter())
                    .map(|&p| {
                        let q = rot(p);
                        FaceVertex {
                            pos: [q.0, q.1, q.2],
                            uv: [(gu.dot(p) as f64 + cu) as f32, (gv.dot(p) as f64 + cv) as f32],
                            rgba: [255; 4],
                            misc: crate::bake::pack_misc(0, 1.0, 32),
                        }
                    })
                    .collect();
                match out.textured.iter_mut().find(|(p, _)| *p == path) {
                    Some(e) => e.1.extend(verts),
                    None => out.textured.push((path, verts)),
                }
            }
            None => {
                let verts: Vec<FaceVertex> = tris
                    .iter()
                    .flat_map(|t| t.iter())
                    .map(|&p| {
                        let q = rot(p);
                        FaceVertex { pos: [q.0, q.1, q.2], uv: [0.0; 2], rgba: [255; 4], misc: 0 }
                    })
                    .collect();
                match out.colored.iter_mut().find(|(c, _)| *c == col) {
                    Some(e) => e.1.extend(verts),
                    None => out.colored.push((col, verts)),
                }
            }
        }
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn quad_scene(billboard: Value) -> Scene {
        let mut sc = Scene::new();
        let mut mesh = Mesh::new();
        let f = mesh.add_face(&[V3(-1.0, 0.0, 0.0), V3(1.0, 0.0, 0.0), V3(1.0, 0.0, 2.0), V3(-1.0, 0.0, 2.0)], &[]);
        mesh.face_mut(f).attrs = vec![(
            "texture".into(),
            Value::Object(vec![("path".into(), Value::Str("t.png".into()))]),
        )];
        sc.meshes.push(mesh);
        let mut g = Group::new(MeshId(1), "figure".into());
        g.billboard = Some(billboard);
        sc.groups.push(g);
        sc
    }

    #[test]
    fn a_quad_billboard_turns_to_face_the_eye() {
        let sc = quad_scene(Value::Bool(true));
        let layers = LayerTable::default();
        let bbs = collect(&sc, &layers);
        assert_eq!(bbs.len(), 1);
        // The eye on +x: the quad lies in the yz plane, still 2 wide and 2 tall.
        let g = build(&sc, &bbs[0], V3(10.0, 0.0, 1.0), V3(0.0, 0.0, 1.0), true).unwrap();
        assert_eq!(g.textured.len(), 1);
        let v = &g.textured[0].1;
        assert_eq!(v.len(), 6);
        assert!(v.iter().all(|p| p.pos[0].abs() < 1e-5), "in the plane x = 0");
        let ys: Vec<f32> = v.iter().map(|p| p.pos[1]).collect();
        assert!(ys.iter().cloned().fold(f32::MIN, f32::max) > 0.99);
        let zs: Vec<f32> = v.iter().map(|p| p.pos[2]).collect();
        assert!((zs.iter().cloned().fold(f32::MIN, f32::max) - 2.0).abs() < 1e-5);
    }

    #[test]
    fn a_silhouette_billboard_keeps_its_feet_and_turns() {
        let sc = quad_scene(Value::Str("mesh".into()));
        let layers = LayerTable::default();
        let bbs = collect(&sc, &layers);
        assert!(bbs[0].silhouette);
        let g = build(&sc, &bbs[0], V3(10.0, 0.0, 1.0), V3(0.0, 0.0, 1.0), true).unwrap();
        let v = &g.textured[0].1;
        // The plane (normal -y) turned a quarter turn about the feet at the origin.
        assert!(v.iter().all(|p| p.pos[1].abs() > 0.99 || p.pos[1].abs() < 1e-5));
        assert!(v.iter().all(|p| p.pos[0].abs() < 1e-5), "{v:?}");
        assert!(v.iter().any(|p| p.pos[2] == 0.0) && v.iter().any(|p| (p.pos[2] - 2.0).abs() < 1e-6));
    }

    #[test]
    fn hidden_billboards_are_not_collected() {
        let mut sc = quad_scene(Value::Bool(true));
        sc.groups[0].hidden = true;
        assert!(collect(&sc, &LayerTable::default()).is_empty());
    }
}
