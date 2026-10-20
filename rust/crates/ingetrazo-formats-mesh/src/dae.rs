// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! COLLADA (`.dae`): `formats/dae.py`, with the geometry collection of
//! `formats/meshexport.py` that export shares with glTF.
//!
//! Export is [`save_dae_text`]: the scene's world triangles grouped by
//! material, one shared `<geometry>`, the text Python joins with `\n` (on
//! Windows its text mode writes `\r\n`: [`save_dae`] takes `crlf`). The
//! `<created>` and `<modified>` stamps are a parameter (Python writes the
//! current UTC time).
//!
//! Import is [`load_dae`]: the XML read by [`crate::xml`], the node tree
//! walked with baked transforms, materials and textures resolved the way
//! `_Dae` does, and the polygons added to the loose mesh. What Python then
//! does with `run_stitch` and `orient_outward` (the operations crate, M3) is
//! the `stitch` hook; what it does with `fuse_coplanar_loops` (models of more
//! than [`MAX_FUSE_LOOPS`] polygons, which become groups, and face-me
//! sprites) is an `Err` starting with `not ported`.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use ingetrazo_formats::pyjson::Value;
use ingetrazo_math::{Hypot, V3};
use ingetrazo_model::pyval;
use ingetrazo_model::scene::Scene;
use ingetrazo_model::texture::path_name;

use crate::obj::{export_names, fkey, round_key, tex_rule, Mat, DEFAULT_COLOR};
use crate::pyutil;
use crate::tri::{export_triangles, face_normal};
use crate::uv::{affine_uv, planar_uv};
use crate::world::world_faces_meshexport;

/// `_MAX_FUSE_LOOPS`.
pub const MAX_FUSE_LOOPS: usize = 400;

// ---- export ---------------------------------------------------------------------

/// `round(x, digits)` of a float: the double nearest the decimal rounding of
/// its exact value (ties to even), as Python\'s `round` gives it.
fn round_digits(x: f64, digits: usize) -> f64 {
    if !x.is_finite() {
        return x;
    }
    format!("{x:.digits$}").parse().unwrap_or(x)
}

/// What `save_dae` writes.
#[derive(Clone, Debug, PartialEq)]
pub struct DaeExport {
    /// The document, its lines joined by `\n` (no final newline, as Python\'s
    /// `"\n".join(out)`).
    pub xml: String,
    /// Texture images to copy next to the `.dae`: `(source path, file name)`.
    pub textures: Vec<(String, String)>,
}

#[derive(Clone, PartialEq, Eq, Hash)]
enum MatKey {
    /// `("tex", name[, opacity])`.
    Tex(String, Option<u64>),
    /// `("color", rgb[, opacity])`.
    Color(Vec<u64>, Option<u64>),
}

struct DMat {
    inner: Mat,
    /// The colour as the faces carried it (up to three numbers).
    color: Vec<f64>,
}

type Corner = (V3, Option<(f64, f64)>);
type Tri = (V3, [Corner; 3]);

fn opacity_of(v: Option<&Value>) -> Result<Option<f64>, String> {
    match v {
        None | Some(Value::Null) => Ok(None),
        Some(v) => {
            let x = pyval::float(v).map_err(|e| format!("{e:?}"))?;
            Ok(if x >= 0.999 { None } else { Some(round_digits(x, 3)) })
        }
    }
}

/// `_xml_escape`.
fn xml_escape(s: &str) -> String {
    s.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;").replace('"', "&quot;")
}

/// `meshexport.geolocation(scene)`.
fn geolocation(scene: &Scene) -> Option<(f64, f64, f64)> {
    let g = scene.extra.georef_datum.as_ref()?;
    let lat = pyval::float(g.get("lat")?).ok()?;
    let lon = pyval::float(g.get("lon")?).ok()?;
    let alt = g.get("alt").and_then(|a| pyval::float(a).ok()).unwrap_or(0.0);
    Some((lat, lon, alt))
}

/// `save_dae(scene, path, project_name)`: the text of the `.dae`. `now` is
/// the `datetime.now(timezone.utc).replace(microsecond=0).isoformat()` stamp
/// (`2026-01-31T12:00:00+00:00`).
pub fn save_dae_text(scene: &Scene, project_name: &str, now: &str, h: Hypot) -> Result<DaeExport, String> {
    // ---- collect_geometry
    let mut mats: Vec<DMat> = Vec::new();
    let mut mat_index: HashMap<MatKey, usize> = HashMap::new();
    // Triangles per material, materials in order of their first triangle.
    let mut prim_order: Vec<usize> = Vec::new();
    let mut prims: HashMap<usize, Vec<Tri>> = HashMap::new();

    for face in world_faces_meshexport(scene) {
        let n = face_normal(&face.outer, h);
        let op = opacity_of(face.attr("opacity"))?.map(|x| (x + 0.0).to_bits());
        let tex = face.attr("texture").filter(|t| t.get("path").is_some_and(pyval::truthy)).cloned();
        let (mi, rule) = if let Some(tex) = &tex {
            let path = match tex.get("path") {
                Some(Value::Str(p)) => p.clone(),
                _ => String::new(),
            };
            let name = path_name(&path, cfg!(windows));
            let mi = *mat_index.entry(MatKey::Tex(name.clone(), op)).or_insert_with(|| {
                mats.push(DMat {
                    inner: Mat { color: [1.0, 1.0, 1.0], map: Some(name.clone()), src: Some(path.clone()), mat: None },
                    color: vec![1.0, 1.0, 1.0],
                });
                mats.len() - 1
            });
            (mi, Some(tex_rule(tex)))
        } else {
            let col: Vec<f64> = match face.attr("color") {
                Some(Value::Array(c)) if !c.is_empty() => {
                    c.iter().take(3).map(|v| pyval::float(v).map_err(|e| format!("{e:?}"))).collect::<Result<_, _>>()?
                }
                Some(c) if pyval::truthy(c) => return Err("unsupported colour attribute".to_string()),
                _ => DEFAULT_COLOR.to_vec(),
            };
            let key = MatKey::Color(col.iter().map(|&x| fkey(x)).collect(), op);
            let mi = *mat_index.entry(key).or_insert_with(|| {
                let mut rgb = [0.0f64; 3];
                for (k, v) in col.iter().enumerate() {
                    rgb[k] = *v;
                }
                mats.push(DMat { inner: Mat { color: rgb, map: None, src: None, mat: None }, color: col.clone() });
                mats.len() - 1
            });
            (mi, None)
        };
        if let Some(m) = face.attr("mat").filter(|m| pyval::truthy(m)) {
            if mats[mi].inner.mat.is_none() {
                mats[mi].inner.mat = Some(m.clone());
            }
        }
        for tri in export_triangles(&face, h) {
            let uv = rule.as_ref().map(|(sw, sh, rot, uvw)| match uvw {
                Some(w) => affine_uv(w, &tri),
                None => planar_uv(n, &tri, *sw, *sh, *rot),
            });
            let corners = [0, 1, 2].map(|k| (tri[k], uv.as_ref().map(|uv| uv[k])));
            prims
                .entry(mi)
                .or_insert_with(|| {
                    prim_order.push(mi);
                    Vec::new()
                })
                .push((n, corners));
        }
    }

    // ---- export_names over every material seen, then save_dae
    let all: Vec<&Mat> = mats.iter().map(|m| &m.inner).collect();
    let names = export_names(&all);
    let keys = &prim_order;

    let mut positions: Vec<V3> = Vec::new();
    let mut normals: Vec<V3> = Vec::new();
    let mut uvs: Vec<(f64, f64)> = Vec::new();
    let mut pidx: HashMap<[(bool, i64); 3], usize> = HashMap::new();
    let mut nidx: HashMap<[u64; 3], usize> = HashMap::new();
    let mut uidx: HashMap<(u64, u64), usize> = HashMap::new();
    let mut groups: Vec<(usize, bool, Vec<String>)> = Vec::new();
    for &mi in keys {
        let textured = mats[mi].inner.map.is_some();
        let mut toks: Vec<String> = Vec::new();
        for (normal, verts) in &prims[&mi] {
            let nkey = [normal.0, normal.1, normal.2].map(|x| fkey(round_digits(x as f64, 5)));
            let ni = *nidx.entry(nkey).or_insert_with(|| {
                normals.push(*normal);
                normals.len() - 1
            });
            for (pos, uv) in verts {
                let pkey = [round_key(pos.0 as f64), round_key(pos.1 as f64), round_key(pos.2 as f64)];
                let pi = *pidx.entry(pkey).or_insert_with(|| {
                    positions.push(*pos);
                    positions.len() - 1
                });
                toks.push(pi.to_string());
                toks.push(ni.to_string());
                if textured {
                    let (u, v) = uv.unwrap_or((0.0, 0.0));
                    let ukey = (fkey(pyutil::round6(u)), fkey(pyutil::round6(v)));
                    let ui = *uidx.entry(ukey).or_insert_with(|| {
                        uvs.push((u, v));
                        uvs.len() - 1
                    });
                    toks.push(ui.to_string());
                }
            }
        }
        groups.push((mi, textured, toks));
    }

    let mut textures = Vec::new();
    for &mi in keys {
        if let (Some(map), Some(src)) = (&mats[mi].inner.map, &mats[mi].inner.src) {
            if !map.is_empty() {
                textures.push((src.clone(), map.clone()));
            }
        }
    }

    let mut out: Vec<String> = Vec::new();
    out.push("<?xml version=\"1.0\" encoding=\"utf-8\"?>".to_string());
    out.push("<COLLADA xmlns=\"http://www.collada.org/2005/11/COLLADASchema\" version=\"1.4.1\">".to_string());
    out.push("<asset>".to_string());
    out.push("<contributor><authoring_tool>IngeTrazo</authoring_tool></contributor>".to_string());
    out.push(format!("<created>{now}</created><modified>{now}</modified>"));
    if let Some((lat, lon, alt)) = geolocation(scene) {
        out.push("<coverage><geographic_location>".to_string());
        out.push(format!("<longitude>{lon:.9}</longitude>"));
        out.push(format!("<latitude>{lat:.9}</latitude>"));
        out.push(format!("<altitude mode=\"absolute\">{alt:.3}</altitude>"));
        out.push("</geographic_location></coverage>".to_string());
    }
    out.push("<unit name=\"meter\" meter=\"1\"/>".to_string());
    out.push("<up_axis>Z_UP</up_axis>".to_string());
    out.push("</asset>".to_string());

    let images: Vec<String> = keys
        .iter()
        .enumerate()
        .filter_map(|(i, &mi)| mats[mi].inner.map.as_ref().filter(|m| !m.is_empty()).map(|m| (i, m)))
        .map(|(i, m)| format!("<image id=\"img{i}\" name=\"img{i}\"><init_from>{}</init_from></image>", xml_escape(m)))
        .collect();
    if !images.is_empty() {
        out.push("<library_images>".to_string());
        out.extend(images);
        out.push("</library_images>".to_string());
    }

    out.push("<library_effects>".to_string());
    for (i, &mi) in keys.iter().enumerate() {
        let info = &mats[mi];
        out.push(format!("<effect id=\"eff{i}\"><profile_COMMON>"));
        if info.inner.map.as_ref().is_some_and(|m| !m.is_empty()) {
            out.push(format!("<newparam sid=\"surf{i}\"><surface type=\"2D\"><init_from>img{i}</init_from></surface></newparam>"));
            out.push(format!("<newparam sid=\"samp{i}\"><sampler2D><source>surf{i}</source></sampler2D></newparam>"));
            out.push(format!(
                "<technique sid=\"common\"><lambert><diffuse><texture texture=\"samp{i}\" texcoord=\"UVSET0\"/></diffuse></lambert></technique>"
            ));
        } else {
            if info.color.len() != 3 {
                return Err(if info.color.len() < 3 {
                    format!("not enough values to unpack (expected 3, got {})", info.color.len())
                } else {
                    "too many values to unpack (expected 3)".to_string()
                });
            }
            let (r, g, b) = (info.color[0], info.color[1], info.color[2]);
            out.push(format!("<technique sid=\"common\"><lambert><diffuse><color>{r:.4} {g:.4} {b:.4} 1</color></diffuse></lambert></technique>"));
        }
        out.push("</profile_COMMON></effect>".to_string());
    }
    out.push("</library_effects>".to_string());

    out.push("<library_materials>".to_string());
    for (i, &mi) in keys.iter().enumerate() {
        out.push(format!("<material id=\"mat{i}\" name=\"{}\"><instance_effect url=\"#eff{i}\"/></material>", names[mi]));
    }
    out.push("</library_materials>".to_string());

    out.push("<library_geometries>".to_string());
    out.push("<geometry id=\"geom0\" name=\"IngeTrazo\"><mesh>".to_string());
    let pos_floats: Vec<String> = positions.iter().flat_map(|p| [p.0, p.1, p.2]).map(|c| format!("{:.6}", c as f64)).collect();
    out.push(format!(
        "<source id=\"geom0-pos\"><float_array id=\"geom0-pos-array\" count=\"{}\">{}</float_array><technique_common><accessor source=\"#geom0-pos-array\" count=\"{}\" stride=\"3\"><param name=\"X\" type=\"float\"/><param name=\"Y\" type=\"float\"/><param name=\"Z\" type=\"float\"/></accessor></technique_common></source>",
        positions.len() * 3,
        pos_floats.join(" "),
        positions.len()
    ));
    let nrm_floats: Vec<String> = normals.iter().flat_map(|p| [p.0, p.1, p.2]).map(|c| format!("{:.5}", c as f64)).collect();
    out.push(format!(
        "<source id=\"geom0-nrm\"><float_array id=\"geom0-nrm-array\" count=\"{}\">{}</float_array><technique_common><accessor source=\"#geom0-nrm-array\" count=\"{}\" stride=\"3\"><param name=\"X\" type=\"float\"/><param name=\"Y\" type=\"float\"/><param name=\"Z\" type=\"float\"/></accessor></technique_common></source>",
        normals.len() * 3,
        nrm_floats.join(" "),
        normals.len()
    ));
    if !uvs.is_empty() {
        let uv_floats: Vec<String> = uvs.iter().flat_map(|&(s, t)| [s, t]).map(|c| format!("{c:.6}")).collect();
        out.push(format!(
            "<source id=\"geom0-uv\"><float_array id=\"geom0-uv-array\" count=\"{}\">{}</float_array><technique_common><accessor source=\"#geom0-uv-array\" count=\"{}\" stride=\"2\"><param name=\"S\" type=\"float\"/><param name=\"T\" type=\"float\"/></accessor></technique_common></source>",
            uvs.len() * 2,
            uv_floats.join(" "),
            uvs.len()
        ));
    }
    out.push("<vertices id=\"geom0-vertices\"><input semantic=\"POSITION\" source=\"#geom0-pos\"/></vertices>".to_string());
    for (i, (_mi, textured, toks)) in groups.iter().enumerate() {
        let stride = if *textured { 3 } else { 2 };
        out.push(format!("<triangles material=\"sym{i}\" count=\"{}\">", toks.len() / (3 * stride)));
        out.push("<input semantic=\"VERTEX\" source=\"#geom0-vertices\" offset=\"0\"/>".to_string());
        out.push("<input semantic=\"NORMAL\" source=\"#geom0-nrm\" offset=\"1\"/>".to_string());
        if *textured {
            out.push("<input semantic=\"TEXCOORD\" source=\"#geom0-uv\" offset=\"2\" set=\"0\"/>".to_string());
        }
        out.push(format!("<p>{}</p>", toks.join(" ")));
        out.push("</triangles>".to_string());
    }
    out.push("</mesh></geometry>".to_string());
    out.push("</library_geometries>".to_string());

    out.push("<library_visual_scenes>".to_string());
    out.push(format!("<visual_scene id=\"scene\"><node id=\"model\" name=\"{}\">", xml_escape(project_name)));
    out.push("<instance_geometry url=\"#geom0\"><bind_material><technique_common>".to_string());
    for (i, (_mi, textured, _p)) in groups.iter().enumerate() {
        out.push(format!("<instance_material symbol=\"sym{i}\" target=\"#mat{i}\">"));
        if *textured {
            out.push("<bind_vertex_input semantic=\"UVSET0\" input_semantic=\"TEXCOORD\" input_set=\"0\"/>".to_string());
        }
        out.push("</instance_material>".to_string());
    }
    out.push("</technique_common></bind_material></instance_geometry>".to_string());
    out.push("</node></visual_scene></library_visual_scenes>".to_string());
    out.push("<scene><instance_visual_scene url=\"#scene\"/></scene>".to_string());
    out.push("</COLLADA>".to_string());
    Ok(DaeExport { xml: out.join("\n"), textures })
}

/// `save_dae(scene, path, project_name)`: writes `path` and copies the
/// textures beside it. `crlf` is what Python\'s text mode does on Windows.
pub fn save_dae(scene: &Scene, path: &Path, project_name: &str, now: &str, h: Hypot, crlf: bool) -> Result<(), String> {
    let export = save_dae_text(scene, project_name, now, h)?;
    let dir = path.parent().map(Path::to_path_buf).unwrap_or_default();
    for (src, name) in &export.textures {
        let (src, dst) = (PathBuf::from(src), dir.join(name));
        let same = match (std::fs::canonicalize(&src), std::fs::canonicalize(&dst)) {
            (Ok(a), Ok(b)) => a == b,
            _ => src == dst,
        };
        if !same && ingetrazo_model::texture::is_image_file(&src, &|_| false) {
            let _ = std::fs::copy(&src, &dst);
        }
    }
    let bytes = if crlf { export.xml.replace('\n', "\r\n").into_bytes() } else { export.xml.into_bytes() };
    std::fs::write(path, bytes).map_err(|e| e.to_string())
}
