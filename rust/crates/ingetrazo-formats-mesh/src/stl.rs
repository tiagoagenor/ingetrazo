// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! STL, binary and ASCII: `formats/stl.py`.
//!
//! Export is the binary writer (`save_stl`, the per-triangle loop: the array
//! version needs Python's native engine, which is off in the reference). Import
//! is `parse_stl` + `add_stl_mesh` up to the edit engine: a small import
//! goes on through `run_stitch` and `orient_outward` in Python, which live in
//! the operations crate (M3), so [`StlMesh::stitch_pending`] says when that
//! step is still to run. Everything before it, and the large-import path
//! (which skips it), is exact.

use ingetrazo_geom::sweep::py_hypot3;
use ingetrazo_math::{Hypot, V3};
use ingetrazo_model::mesh::Mesh;
use ingetrazo_model::scene::{Group, MeshId, Scene};

use crate::pyutil;
use crate::tri::export_triangles;
use crate::world::world_faces;

/// `STL_UNITS`: metres per unit.
pub const STL_UNITS: [(&str, f64); 5] = [("m", 1.0), ("cm", 0.01), ("mm", 0.001), ("in", 0.0254), ("ft", 0.3048)];

const BINARY_HEADER_SIZE: usize = 84;
const BINARY_TRIANGLE_SIZE: usize = 50;
const MAX_FLOAT32: f64 = 3.4028234663852886e38;
/// `formats.dae._MAX_FUSE_LOOPS`: above it an import stays one reference group.
pub const MAX_FUSE_LOOPS: usize = 400;

fn unit_metres(unit: &str) -> Option<f64> {
    STL_UNITS.iter().find(|(u, _)| *u == unit).map(|&(_, m)| m)
}

/// `_scaled_point`: `value * scale` per coordinate, which must be finite and
/// fit a float32.
fn scaled_point(values: [f64; 3], scale: f64) -> Result<[f64; 3], String> {
    let p = [values[0] * scale, values[1] * scale, values[2] * scale];
    if p.iter().all(|v| v.is_finite() && v.abs() <= MAX_FLOAT32) {
        Ok(p)
    } else {
        Err("STL contains a coordinate outside the supported range".to_string())
    }
}

/// `iter_triangles(scene)`: every world-space triangle of the scene.
pub fn iter_triangles(scene: &Scene, h: Hypot) -> Vec<[V3; 3]> {
    world_faces(scene).iter().flat_map(|f| export_triangles(f, h)).collect()
}

/// `_normal(a, b, c)`.
fn triangle_normal(a: V3, b: V3, c: V3, h: Hypot) -> V3 {
    let n = b.sub(a).cross(c.sub(a));
    let length = n.length(h);
    if (length as f64) > 1e-12 {
        n.div(length)
    } else {
        V3(0.0, 0.0, 0.0)
    }
}

/// `save_stl(scene, path, unit)`: the file's bytes. `Err` carries the
/// exception message Python raises (`KeyError` for an unknown unit).
pub fn save_stl_bytes(scene: &Scene, unit: &str, h: Hypot) -> Result<Vec<u8>, String> {
    let scale = 1.0 / unit_metres(unit).ok_or_else(|| format!("KeyError: {}", pyutil::repr(unit)))?;
    let tris = iter_triangles(scene, h);
    let mut out = Vec::with_capacity(BINARY_HEADER_SIZE + tris.len() * BINARY_TRIANGLE_SIZE);
    let mut header = b"IngeTrazo STL export".to_vec();
    header.resize(80, 0);
    out.extend_from_slice(&header);
    out.extend_from_slice(&(tris.len() as u32).to_le_bytes());
    for [a, b, c] in tris {
        let n = triangle_normal(a, b, c, h);
        for x in [n.0, n.1, n.2] {
            out.extend_from_slice(&x.to_le_bytes());
        }
        for p in [a, b, c] {
            for x in scaled_point([p.0 as f64, p.1 as f64, p.2 as f64], scale)? {
                out.extend_from_slice(&(x as f32).to_le_bytes());
            }
        }
        out.extend_from_slice(&0u16.to_le_bytes());
    }
    Ok(out)
}

// ---- import -------------------------------------------------------------------

/// How an import merges flat triangles (`simplify_mode`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Simplify {
    None,
    Principal,
    All,
}

impl Simplify {
    pub fn parse(mode: &str) -> Result<Simplify, String> {
        match mode {
            "none" => Ok(Simplify::None),
            "principal" => Ok(Simplify::Principal),
            "all" => Ok(Simplify::All),
            _ => Err("STL simplification mode must be none, principal, or all".to_string()),
        }
    }
}

/// A parsed STL, ready for [`add_stl_mesh`].
pub struct StlMesh {
    pub mesh: Mesh,
    /// Python now runs `run_stitch(coplanar_merge=True)` and
    /// `orient_outward` on it (an import of up to [`MAX_FUSE_LOOPS`]
    /// triangles); the mesh here is the one just before that.
    pub stitch_pending: bool,
}

fn cross_of(a: [f64; 3], b: [f64; 3], c: [f64; 3]) -> [f64; 3] {
    let ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    let ac = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
    [ab[1] * ac[2] - ab[2] * ac[1], ab[2] * ac[0] - ab[0] * ac[2], ab[0] * ac[1] - ab[1] * ac[0]]
}

/// `_binary_triangles`: the facets kept, as doubles.
fn binary_triangles(raw: &[u8], count: usize, scale: f64) -> Result<Vec<[[f64; 3]; 3]>, String> {
    let whole = raw.len() / BINARY_TRIANGLE_SIZE;
    let n = whole.min(count);
    let f32_at = |rec: usize, k: usize| {
        let at = rec * BINARY_TRIANGLE_SIZE + 12 + 4 * k;
        f32::from_le_bytes([raw[at], raw[at + 1], raw[at + 2], raw[at + 3]]) as f64 * scale
    };
    let mut tris = Vec::with_capacity(n);
    let mut first_bad = n;
    for r in 0..n {
        let v: Vec<f64> = (0..9).map(|k| f32_at(r, k)).collect();
        if first_bad == n && !v.iter().all(|x| x.is_finite() && x.abs() <= MAX_FLOAT32) {
            first_bad = r;
        }
        tris.push([[v[0], v[1], v[2]], [v[3], v[4], v[5]], [v[6], v[7], v[8]]]);
    }
    if whole < count || first_bad < n {
        return Err(if first_bad >= n {
            "STL binary facet data is truncated".to_string()
        } else {
            "STL contains a coordinate outside the supported range".to_string()
        });
    }
    Ok(tris
        .into_iter()
        .filter(|t| {
            let c = cross_of(t[0], t[1], t[2]);
            let length = ((c[0] * c[0] + c[1] * c[1]) + c[2] * c[2]).sqrt();
            if (length - 1e-12).abs() <= 1e-15 {
                py_hypot3(c[0], c[1], c[2]) > 1e-12
            } else {
                length > 1e-12
            }
        })
        .collect())
}

/// `_non_degenerate`.
fn non_degenerate(t: &[[f64; 3]; 3]) -> bool {
    let c = cross_of(t[0], t[1], t[2]);
    py_hypot3(c[0], c[1], c[2]) > 1e-12
}

/// The lines of a text file opened in Python's text mode: `\n`, `\r\n` and a
/// lone `\r` all end a line.
fn text_lines(text: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let b = text.as_bytes();
    let mut start = 0;
    let mut i = 0;
    while i < b.len() {
        match b[i] {
            b'\n' => {
                out.push(&text[start..i]);
                start = i + 1;
            }
            b'\r' => {
                out.push(&text[start..i]);
                if b.get(i + 1) == Some(&b'\n') {
                    i += 1;
                }
                start = i + 1;
            }
            _ => {}
        }
        i += 1;
    }
    if start < b.len() {
        out.push(&text[start..]);
    }
    out
}

#[derive(PartialEq)]
enum State {
    Outside,
    Outer,
    Vertices,
    EndLoop,
    EndFacet,
}

/// `_ascii_stl`: the triangles of an ASCII file (or its first error).
fn ascii_triangles(data: &[u8], scale: f64) -> Result<Vec<[[f64; 3]; 3]>, String> {
    let lossy = String::from_utf8_lossy(data);
    let text: &str = lossy.strip_prefix('\u{feff}').unwrap_or(&lossy);
    let mut state = State::Outside;
    let mut vertices: Vec<[f64; 3]> = Vec::new();
    let mut out = Vec::new();
    for (idx, line) in text_lines(text).into_iter().enumerate() {
        let line_number = idx + 1;
        let parts = pyutil::split(line);
        if parts.is_empty() || parts[0].starts_with('#') {
            continue;
        }
        let keyword = parts[0].to_lowercase();
        match state {
            State::Outside => {
                if keyword == "facet" {
                    let bad = || format!("Invalid STL facet normal on line {line_number}");
                    if parts.len() != 5 || parts[1].to_lowercase() != "normal" {
                        return Err(bad());
                    }
                    let mut normal = [0.0f64; 3];
                    for (k, v) in parts[2..].iter().enumerate() {
                        normal[k] = pyutil::float(v).ok_or_else(bad)?;
                    }
                    if !normal.iter().all(|x| x.is_finite()) {
                        return Err(format!("Non-finite STL facet normal on line {line_number}"));
                    }
                    vertices.clear();
                    state = State::Outer;
                }
            }
            State::Outer => {
                if parts.len() != 2 || keyword != "outer" || parts[1].to_lowercase() != "loop" {
                    return Err(format!("Expected 'outer loop' on STL line {line_number}"));
                }
                state = State::Vertices;
            }
            State::Vertices => {
                if keyword != "vertex" || parts.len() != 4 {
                    return Err(format!("Expected an STL vertex on line {line_number}"));
                }
                let mut v = [0.0f64; 3];
                for (k, tok) in parts[1..].iter().enumerate() {
                    v[k] = pyutil::float(tok).ok_or_else(|| {
                        format!(
                            "Invalid STL vertex on line {line_number}: could not convert string to float: {}",
                            pyutil::repr(tok)
                        )
                    })?;
                }
                let p = scaled_point(v, scale).map_err(|e| format!("Invalid STL vertex on line {line_number}: {e}"))?;
                vertices.push(p);
                if vertices.len() == 3 {
                    state = State::EndLoop;
                }
            }
            State::EndLoop => {
                if keyword != "endloop" || parts.len() != 1 {
                    return Err(format!("Expected 'endloop' on STL line {line_number}"));
                }
                state = State::EndFacet;
            }
            State::EndFacet => {
                if keyword != "endfacet" || parts.len() != 1 {
                    return Err(format!("Expected 'endfacet' on STL line {line_number}"));
                }
                out.push([vertices[0], vertices[1], vertices[2]]);
                state = State::Outside;
            }
        }
    }
    if state != State::Outside {
        return Err("STL ASCII facet is incomplete".to_string());
    }
    Ok(out)
}

/// `_parse_stl_inner` on the file's bytes.
pub fn parse_stl(data: &[u8], scale: f64, simplify: Simplify) -> Result<StlMesh, String> {
    if !scale.is_finite() || scale <= 0.0 {
        return Err("STL unit scale must be a positive finite number".to_string());
    }
    let size = data.len();
    let mut binary = false;
    let mut count = 0usize;
    if size >= BINARY_HEADER_SIZE {
        count = u32::from_le_bytes([data[80], data[81], data[82], data[83]]) as usize;
        let expected = BINARY_HEADER_SIZE + count * BINARY_TRIANGLE_SIZE;
        binary = size >= expected && (size == expected || !data[..5].eq_ignore_ascii_case(b"solid"));
    }
    let mut target = Mesh::new();
    let add = |target: &mut Mesh, t: &[[f64; 3]; 3]| {
        let ring: Vec<V3> = t.iter().map(|p| V3(p[0] as f32, p[1] as f32, p[2] as f32)).collect();
        target.add_face(&ring, &[]);
    };
    if binary {
        let raw = &data[BINARY_HEADER_SIZE..BINARY_HEADER_SIZE + count * BINARY_TRIANGLE_SIZE];
        let kept = binary_triangles(raw, count, scale)?;
        if kept.is_empty() {
            return Err("STL file contains no usable triangles".to_string());
        }
        for t in &kept {
            add(&mut target, t);
        }
        if simplify == Simplify::None && kept.len() > MAX_FUSE_LOOPS {
            return Ok(StlMesh { mesh: target, stitch_pending: false });
        }
    } else {
        for t in ascii_triangles(data, scale)?.iter().filter(|t| non_degenerate(t)) {
            add(&mut target, t);
        }
    }
    if target.faces.is_empty() {
        return Err("STL file contains no usable triangles".to_string());
    }
    if simplify != Simplify::None {
        return Err("STL simplification (principal, all) needs the fuse pass, which is not ported yet".to_string());
    }
    if target.faces.len() > MAX_FUSE_LOOPS {
        return Ok(StlMesh { mesh: target, stitch_pending: false });
    }
    Ok(StlMesh { mesh: target, stitch_pending: true })
}

/// `add_stl_mesh(scene, path, target)`: a large mesh becomes a reference
/// group named after the file, a small one is added to the loose mesh.
pub fn add_stl_mesh(scene: &mut Scene, stem: &str, target: Mesh) {
    if target.faces.len() > MAX_FUSE_LOOPS {
        scene.meshes.push(target);
        let id = MeshId(scene.meshes.len() as u32 - 1);
        scene.groups.push(Group::new(id, stem.to_string()));
    } else {
        let faces: Vec<_> = target
            .faces
            .iter()
            .map(|&f| {
                let face = target.face(f);
                (
                    face.outer.iter().map(|&v| target.position(v)).collect::<Vec<_>>(),
                    face.holes.iter().map(|h| h.iter().map(|&v| target.position(v)).collect::<Vec<_>>()).collect::<Vec<_>>(),
                )
            })
            .collect();
        for (outer, holes) in faces {
            scene.meshes[0].add_face(&outer, &holes);
        }
    }
}

/// `load_stl(scene, path, scale=…)` on the file's bytes; `stem` is the file
/// name without its suffix. Returns whether the stitch step is still to run.
pub fn load_stl(scene: &mut Scene, stem: &str, data: &[u8], scale: f64, simplify: Simplify) -> Result<bool, String> {
    let parsed = parse_stl(data, scale, simplify)?;
    let pending = parsed.stitch_pending;
    add_stl_mesh(scene, stem, parsed.mesh);
    Ok(pending)
}
