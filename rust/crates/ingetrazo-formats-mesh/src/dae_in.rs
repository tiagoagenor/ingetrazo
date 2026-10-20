// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! COLLADA import: `load_dae` and its helpers in `formats/dae.py`.
//!
//! The document is read by [`crate::xml`]; the node tree is walked with the
//! transforms baked in (`matrix` / `translate` / `rotate` / `scale` composed
//! as Qt composes them, in float32), `instance_node` recurses into the
//! library, and each `instance_geometry` yields its polygons with the colour
//! or texture its bound material resolves to. A model of up to
//! [`MAX_FUSE_LOOPS`] polygons goes into the loose mesh, then `stitch` stands
//! where Python runs `run_stitch` and `orient_outward`.
//!
//! Models of more than [`MAX_FUSE_LOOPS`] polygons are reference imports: one
//! group per assembly of the node tree (the greedy splitter of `_bucketize`),
//! a shared prototype mesh per reused component (`instances`), billboard
//! groups for the face-me sprites, every mesh through `fuse_coplanar_loops`.
//! The average colour of an image is `QImage.scaled(1, 1, ...)`
//! ([`crate::img::smooth_average`], PNG only: any other format is an `Err`
//! starting with `not ported`); it is worked out only when a polygon falls
//! back to the material colour.
//!
//! A material whose image is missing gets the tint Python derives from
//! `hash(ref)`; that hash is salted per process, so the tint is what a run
//! with `PYTHONHASHSEED=0` gives ([`py_str_hash_seed0`]).

use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::rc::Rc;

use ingetrazo_formats::pyjson::Value;
use ingetrazo_geom::sweep::py_hypot3;
use ingetrazo_math::{Hypot, M4, V3};
use ingetrazo_model::mesh::{key, FaceId, Key, Mesh};
use ingetrazo_model::pyval;
use ingetrazo_model::scene::{Group, MeshId, Scene};

use crate::fusing::{add_fused, fuse_coplanar_loops, group_name, new_group, push_mesh, soften_smooth_edges, RawLoop};
use crate::obj::Stitch;
use crate::pyutil;
use crate::uv::fit_uv_affine;
use crate::xml::{self, local, Doc};

pub use crate::dae::MAX_FUSE_LOOPS;

fn ns(tag: &str) -> String {
    format!("{{http://www.collada.org/2005/11/COLLADASchema}}{tag}")
}

/// `_MAX_DEPTH` of the node walk (`depth > 32`).
const MAX_DEPTH: usize = 32;

// ---- Python scalars -------------------------------------------------------------

/// `int(text)`, saturating where Python\'s integer would not fit an `i64`.
fn py_int(text: &str) -> Result<i64, String> {
    let bad = || format!("invalid literal for int() with base 10: {}", pyutil::repr(text));
    let t = text.trim_matches(|c: char| pyutil::is_space(c));
    let (neg, digits) = match t.strip_prefix('-') {
        Some(d) => (true, d),
        None => (false, t.strip_prefix('+').unwrap_or(t)),
    };
    if digits.is_empty() || digits.starts_with('_') || digits.ends_with('_') || digits.contains("__") {
        return Err(bad());
    }
    if !digits.chars().all(|c| c.is_ascii_digit() || c == '_') {
        return Err(bad());
    }
    let clean: String = digits.chars().filter(|&c| c != '_').collect();
    let n: i128 = clean.parse().unwrap_or(i128::MAX);
    let n = if neg { -n } else { n };
    Ok(n.clamp(i64::MIN as i128, i64::MAX as i128) as i64)
}

fn py_float(text: &str) -> Result<f64, String> {
    pyutil::float(text).ok_or_else(|| format!("could not convert string to float: {}", pyutil::repr(text)))
}

/// `_floats(text)`.
fn floats(text: Option<&str>) -> Result<Vec<f64>, String> {
    pyutil::split(text.unwrap_or("")).into_iter().map(py_float).collect()
}

/// `_ints(text)`.
fn ints(text: Option<&str>) -> Result<Vec<i64>, String> {
    pyutil::split(text.unwrap_or("")).into_iter().map(py_int).collect()
}

/// `idx[i]` with Python\'s negative indices.
fn at(idx: &[i64], i: i64) -> Result<i64, String> {
    let n = idx.len() as i64;
    let j = if i < 0 { i + n } else { i };
    if (0..n).contains(&j) {
        Ok(idx[j as usize])
    } else {
        Err("list index out of range".to_string())
    }
}

/// `urllib.parse.unquote`: percent escapes decoded as UTF-8 (replacing what is
/// not), one run of ASCII at a time.
fn unquote(s: &str) -> String {
    if !s.contains('%') {
        return s.to_string();
    }
    let mut out = String::new();
    let mut run: Vec<u8> = Vec::new();
    let flush = |run: &mut Vec<u8>, out: &mut String| {
        if !run.is_empty() {
            out.push_str(&String::from_utf8_lossy(run));
            run.clear();
        }
    };
    let b = s.as_bytes();
    let mut i = 0;
    while i < b.len() {
        let c = b[i];
        if c >= 0x80 {
            flush(&mut run, &mut out);
            let ch = s[i..].chars().next().unwrap_or('\u{fffd}');
            out.push(ch);
            i += ch.len_utf8();
            continue;
        }
        if c == b'%' && i + 2 < b.len() && b[i + 1].is_ascii_hexdigit() && b[i + 2].is_ascii_hexdigit() {
            run.push(u8::from_str_radix(&s[i + 1..i + 3], 16).unwrap_or(b'%'));
            i += 3;
        } else {
            run.push(c);
            i += 1;
        }
    }
    flush(&mut run, &mut out);
    out
}

/// `str(Path(...))` of a joined path: the separators of the platform, no `.`
/// parts, no doubled separators.
fn path_str(p: &Path) -> String {
    let s = p.to_string_lossy().into_owned();
    if !cfg!(windows) {
        let rooted = s.starts_with('/');
        let parts: Vec<&str> = s.split('/').filter(|x| !x.is_empty() && *x != ".").collect();
        return format!("{}{}", if rooted { "/" } else { "" }, parts.join("/"));
    }
    let s = s.replace('/', "\\");
    let (head, rest) = if let Some(r) = s.strip_prefix("\\\\") {
        ("\\\\", r.to_string())
    } else {
        ("", s.clone())
    };
    let mut parts: Vec<&str> = rest.split('\\').filter(|x| !x.is_empty() && *x != ".").collect();
    let mut prefix = head.to_string();
    if head.is_empty() {
        if rest.starts_with('\\') {
            prefix = "\\".to_string();
        } else if parts.first().is_some_and(|d| d.len() == 2 && d.ends_with(':')) {
            let drive = parts.remove(0);
            prefix = if rest[drive.len()..].starts_with('\\') { format!("{drive}\\") } else { drive.to_string() };
        }
    }
    format!("{prefix}{}", parts.join("\\"))
}

/// `hash(text)` of a `str` under `PYTHONHASHSEED=0`: SipHash-1-3 with a zero
/// key over the string\'s storage (one, two or four bytes per character,
/// whichever the widest character needs).
pub fn py_str_hash_seed0(text: &str) -> i64 {
    let chars: Vec<u32> = text.chars().map(|c| c as u32).collect();
    if chars.is_empty() {
        return 0;
    }
    let widest = chars.iter().copied().max().unwrap_or(0);
    let mut data: Vec<u8> = Vec::new();
    for &c in &chars {
        if widest <= 0xff {
            data.push(c as u8);
        } else if widest <= 0xffff {
            data.extend((c as u16).to_le_bytes());
        } else {
            data.extend(c.to_le_bytes());
        }
    }
    let mut v = [0x736f6d6570736575u64, 0x646f72616e646f6d, 0x6c7967656e657261, 0x7465646279746573];
    let round = |v: &mut [u64; 4]| {
        v[0] = v[0].wrapping_add(v[1]);
        v[2] = v[2].wrapping_add(v[3]);
        v[1] = v[1].rotate_left(13);
        v[3] = v[3].rotate_left(16);
        v[1] ^= v[0];
        v[3] ^= v[2];
        v[0] = v[0].rotate_left(32);
        v[2] = v[2].wrapping_add(v[1]);
        v[0] = v[0].wrapping_add(v[3]);
        v[1] = v[1].rotate_left(17);
        v[3] = v[3].rotate_left(21);
        v[1] ^= v[2];
        v[3] ^= v[0];
        v[2] = v[2].rotate_left(32);
    };
    let mut b = (data.len() as u64) << 56;
    let (words, tail) = data.as_chunks::<8>();
    for c in words {
        let m = u64::from_le_bytes(*c);
        v[3] ^= m;
        round(&mut v);
        v[0] ^= m;
    }
    for (k, &byte) in tail.iter().enumerate() {
        b |= (byte as u64) << (8 * k);
    }
    v[3] ^= b;
    round(&mut v);
    v[0] ^= b;
    v[2] ^= 0xff;
    for _ in 0..3 {
        round(&mut v);
    }
    let h = (v[0] ^ v[1] ^ v[2] ^ v[3]) as i64;
    if h == -1 {
        -2
    } else {
        h
    }
}

// ---- the document ---------------------------------------------------------------

/// A colour a material resolves to.
#[derive(Clone, Debug)]
enum ColorV {
    Rgb(Vec<f64>),
    /// The average of an image that exists on disk, worked out when a polygon
    /// needs it (Python works it out for every textured material, but only
    /// a polygon without a texture of its own looks at the result): the
    /// file, and the `init_from` text the tint is derived from if Qt cannot
    /// read it.
    ImageAverage { file: PathBuf, reference: String },
}

/// A texture: the image path and the UVs of a polygon\'s corners.
type Tex = (String, Vec<(f64, f64)>);

/// One polygon with its material: the `(points, color, tex)` triples of Python.
#[derive(Clone, Debug)]
struct Loop {
    pts: Vec<V3>,
    color: Option<ColorV>,
    tex: Option<Tex>,
}

#[derive(Clone, Debug)]
struct MatInfo {
    color: Option<ColorV>,
    path: Option<String>,
}

struct Dae<'a> {
    doc: &'a Doc,
    base_dir: Option<PathBuf>,
    by_id: HashMap<String, usize>,
    scale: f64,
    up: String,
    src_cache: RefCell<HashMap<usize, Rc<(Vec<f64>, i64)>>>,
    /// `inst_counts`: how many times each library node is instanced.
    inst_counts: HashMap<String, usize>,
    /// `_polys_cache`: the first count asked for a node stays (Python keys
    /// it by the node alone, whatever depth later callers ask at).
    polys_cache: RefCell<HashMap<usize, i64>>,
    /// `_img_alpha`: whether an image has a cut-out, by path.
    img_alpha: RefCell<HashMap<String, bool>>,
}

impl<'a> Dae<'a> {
    fn new(doc: &'a Doc, base_dir: Option<PathBuf>) -> Result<Dae<'a>, String> {
        let by_id = xml::ids(doc);
        let mut scale = 1.0;
        let mut up = "Z_UP".to_string();
        if let Some(asset) = doc.find(Doc::ROOT, &ns("asset")) {
            if let Some(unit) = doc.find(asset, &ns("unit")) {
                if let Some(m) = doc.get(unit, "meter").filter(|m| !m.is_empty()) {
                    scale = py_float(m)?;
                }
            }
            if let Some(u) = doc.find(asset, &ns("up_axis")) {
                if let Some(t) = doc.el(u).text.as_deref() {
                    up = t.trim_matches(|c: char| pyutil::is_space(c)).to_string();
                }
            }
        }
        let mut inst_counts: HashMap<String, usize> = HashMap::new();
        for el in doc.iter_tag(Doc::ROOT, &ns("instance_node")) {
            *inst_counts.entry(doc.get(el, "url").unwrap_or("").to_string()).or_insert(0) += 1;
        }
        Ok(Dae {
            doc,
            base_dir,
            by_id,
            scale,
            up,
            src_cache: RefCell::new(HashMap::new()),
            inst_counts,
            polys_cache: RefCell::new(HashMap::new()),
            img_alpha: RefCell::new(HashMap::new()),
        })
    }

    /// `polys_of(node_el, depth)`.
    fn polys_of(&self, node: usize, depth: usize) -> i64 {
        if let Some(&n) = self.polys_cache.borrow().get(&node) {
            return n;
        }
        let n = self.subtree_count(node, depth);
        self.polys_cache.borrow_mut().insert(node, n);
        n
    }

    /// `zup_matrix()`: `to_zup` as a matrix, to conjugate instance transforms.
    fn zup_matrix(&self) -> M4 {
        let s = self.scale as f32;
        match self.up.as_str() {
            "Y_UP" => M4::from_rows([s, 0.0, 0.0, 0.0, 0.0, 0.0, -s, 0.0, 0.0, s, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
            "X_UP" => M4::from_rows([0.0, s, 0.0, 0.0, s, 0.0, 0.0, 0.0, 0.0, 0.0, s, 0.0, 0.0, 0.0, 0.0, 1.0]),
            _ => {
                let mut m = M4::identity();
                m.scale1(s);
                m
            }
        }
    }

    /// `_image_has_cutout(dae, path)`.
    fn image_has_cutout(&self, path: &str) -> bool {
        if let Some(&hit) = self.img_alpha.borrow().get(path) {
            return hit;
        }
        let ok = std::fs::read(path).is_ok_and(|data| crate::img::image_has_cutout(&data));
        self.img_alpha.borrow_mut().insert(path.to_string(), ok);
        ok
    }

    /// `ref(url)`.
    fn reference(&self, url: Option<&str>) -> Option<usize> {
        self.by_id.get(url.unwrap_or("").trim_start_matches('#')).copied()
    }

    /// `to_zup(p)`.
    fn to_zup(&self, p: V3) -> V3 {
        let s = self.scale;
        let (x, y, z) = (p.0 as f64, p.1 as f64, p.2 as f64);
        match self.up.as_str() {
            "Y_UP" => V3((x * s) as f32, (-z * s) as f32, (y * s) as f32),
            "X_UP" => V3((y * s) as f32, (x * s) as f32, (z * s) as f32),
            _ => V3((x * s) as f32, (y * s) as f32, (z * s) as f32),
        }
    }

    fn tag_is(&self, id: usize, name: &str) -> bool {
        local(&self.doc.el(id).tag) == name
    }

    /// `_source_floats(dae, source_el)`.
    fn source_floats(&self, src: usize) -> Result<Rc<(Vec<f64>, i64)>, String> {
        if let Some(hit) = self.src_cache.borrow().get(&src) {
            return Ok(hit.clone());
        }
        let doc = self.doc;
        let arr = doc.find(src, &ns("float_array"));
        let data = floats(arr.and_then(|a| doc.el(a).text.as_deref()))?;
        let mut stride = 3;
        if let Some(tc) = doc.find(src, &ns("technique_common")) {
            if let Some(acc) = doc.find(tc, &ns("accessor")) {
                if let Some(s) = doc.get(acc, "stride").filter(|s| !s.is_empty()) {
                    stride = py_int(s)?;
                }
            }
        }
        let out = Rc::new((data, stride));
        self.src_cache.borrow_mut().insert(src, out.clone());
        Ok(out)
    }

    /// `_positions(dae, mesh_el)`.
    fn positions(&self, mesh_el: usize) -> Result<Vec<V3>, String> {
        let doc = self.doc;
        let Some(verts) = doc.find(mesh_el, &ns("vertices")) else { return Ok(Vec::new()) };
        for inp in doc.findall(verts, &ns("input")) {
            if doc.get(inp, "semantic") == Some("POSITION") {
                let Some(src) = self.reference(doc.get(inp, "source")) else { return Ok(Vec::new()) };
                let sf = self.source_floats(src)?;
                let (data, stride) = (&sf.0, sf.1);
                let stop = data.len() as i64 - 2;
                if stride == 0 {
                    return Err("range() arg 3 must not be zero".to_string());
                }
                if stride < 0 {
                    return if 0 > stop { Err("list index out of range".to_string()) } else { Ok(Vec::new()) };
                }
                let mut out = Vec::new();
                let mut i = 0i64;
                while i < stop {
                    let k = i as usize;
                    out.push(V3(data[k] as f32, data[k + 1] as f32, data[k + 2] as f32));
                    i += stride;
                }
                return Ok(out);
            }
        }
        Ok(Vec::new())
    }

    /// `_prim_inputs(prim_el)`: `(vertex_offset, texcoord_offset,
    /// texcoord_source, stride)`.
    fn prim_inputs(&self, prim: usize) -> Result<(i64, Option<i64>, Option<String>, i64), String> {
        let doc = self.doc;
        let (mut v_off, mut uv_off, mut uv_src, mut max_off) = (0i64, None, None, 0i64);
        for inp in doc.findall(prim, &ns("input")) {
            let off = py_int(doc.get(inp, "offset").unwrap_or("0"))?;
            max_off = max_off.max(off);
            match doc.get(inp, "semantic") {
                Some("VERTEX") => v_off = off,
                Some("TEXCOORD") if uv_off.is_none() => {
                    uv_off = Some(off);
                    uv_src = doc.get(inp, "source").map(str::to_string);
                }
                _ => {}
            }
        }
        Ok((v_off, uv_off, uv_src, max_off + 1))
    }

    /// `_prim_loops(prim_el, positions, uv_count)`: vertex index loops with
    /// their UV index loops.
    #[allow(clippy::type_complexity)]
    fn prim_loops(&self, prim: usize, n_positions: usize, uv_count: usize) -> Result<Vec<(Vec<i64>, Option<Vec<i64>>)>, String> {
        let doc = self.doc;
        let kind = local(&doc.el(prim).tag).to_string();
        let (v_off, uv_off, _src, stride) = self.prim_inputs(prim)?;
        let take_uv = uv_off.is_some() && uv_count > 0;
        let mut loops: Vec<(Vec<i64>, Option<Vec<i64>>)> = Vec::new();
        let emit = |loops: &mut Vec<(Vec<i64>, Option<Vec<i64>>)>, idx: &[i64], start: i64, nverts: i64| -> Result<(), String> {
            let mut vl = Vec::new();
            for j in 0..nverts.max(0) {
                vl.push(at(idx, start + j * stride + v_off)?);
            }
            let ul = if take_uv {
                let mut ul = Vec::new();
                for j in 0..nverts.max(0) {
                    ul.push(at(idx, start + j * stride + uv_off.unwrap_or(0))?);
                }
                Some(ul)
            } else {
                None
            };
            loops.push((vl, ul));
            Ok(())
        };
        let p_text = |p: Option<usize>| p.and_then(|p| doc.el(p).text.as_deref().map(str::to_string));
        match kind.as_str() {
            "triangles" => {
                let idx = ints(p_text(doc.find(prim, &ns("p"))).as_deref())?;
                let verts_per = 3 * stride;
                let mut k = 0i64;
                while k < idx.len() as i64 - verts_per + 1 {
                    emit(&mut loops, &idx, k, 3)?;
                    k += verts_per;
                }
            }
            "polylist" => {
                let counts = ints(p_text(doc.find(prim, &ns("vcount"))).as_deref())?;
                let idx = ints(p_text(doc.find(prim, &ns("p"))).as_deref())?;
                let mut pos = 0i64;
                for c in counts {
                    if pos + c * stride <= idx.len() as i64 {
                        emit(&mut loops, &idx, pos, c)?;
                    }
                    pos += c * stride;
                }
            }
            "polygons" => {
                for p in doc.findall(prim, &ns("p")) {
                    let idx = ints(doc.el(p).text.as_deref())?;
                    emit(&mut loops, &idx, 0, idx.len() as i64 / stride)?;
                }
            }
            _ => {}
        }
        let mut out = Vec::new();
        for (vl, ul) in loops {
            if vl.len() < 3 || !vl.iter().all(|&i| 0 <= i && (i as usize) < n_positions) {
                continue;
            }
            let ul = ul.filter(|u| u.iter().all(|&i| 0 <= i && (i as usize) < uv_count));
            out.push((vl, ul));
        }
        Ok(out)
    }

    fn init_ref(&self, image: usize) -> String {
        match self.doc.find(image, &ns("init_from")) {
            Some(i) => self.doc.el(i).text.clone().unwrap_or_default().trim_matches(|c: char| pyutil::is_space(c)).to_string(),
            None => String::new(),
        }
    }

    fn candidate(&self, reference: &str) -> Option<PathBuf> {
        self.base_dir.as_ref().map(|b| b.join(unquote(reference)))
    }

    /// `_image_color(dae, image_el)`.
    fn image_color(&self, image: usize) -> Option<ColorV> {
        let reference = self.init_ref(image);
        if reference.is_empty() {
            return None;
        }
        if let Some(file) = self.candidate(&reference).filter(|c| c.is_file()) {
            return Some(ColorV::ImageAverage { file, reference });
        }
        Some(ColorV::Rgb(tint(&reference)))
    }

    /// `_image_file(dae, image_el)`.
    fn image_file(&self, image: usize) -> Option<String> {
        let reference = self.init_ref(image);
        if reference.is_empty() {
            return None;
        }
        let c = self.candidate(&reference)?;
        c.is_file().then(|| path_str(&c))
    }

    /// `_effect_diffuse(dae, material_el)`: the literal colour, or the
    /// diffuse texture\'s `<image>`.
    fn effect_diffuse(&self, material: usize) -> Result<(Option<Vec<f64>>, Option<usize>), String> {
        let doc = self.doc;
        let ie = doc.find(material, &ns("instance_effect"));
        let effect = ie.and_then(|ie| self.reference(doc.get(ie, "url")));
        let Some(effect) = effect else { return Ok((None, None)) };
        for shader in ["lambert", "phong", "blinn", "constant"] {
            for el in doc.iter_tag(effect, &ns(shader)) {
                let Some(diffuse) = doc.find(el, &ns("diffuse")) else { continue };
                if let Some(col) = doc.find(diffuse, &ns("color")) {
                    let vals = floats(doc.el(col).text.as_deref())?;
                    if vals.len() >= 3 {
                        return Ok((Some(vals[..3].to_vec()), None));
                    }
                }
                if let Some(tex) = doc.find(diffuse, &ns("texture")) {
                    // sampler -> surface -> image (some exporters put the
                    // image id directly in texture/@texture)
                    let sid = doc.get(tex, "texture").unwrap_or("").to_string();
                    let mut target = self.by_id.get(&sid).copied();
                    if target.is_none_or(|t| !self.tag_is(t, "image")) {
                        for np in doc.iter_tag(effect, &ns("newparam")) {
                            if doc.get(np, "sid") != Some(sid.as_str()) {
                                continue;
                            }
                            let s2d = doc.find(np, &ns("sampler2D"));
                            let src = s2d.and_then(|s| doc.find(s, &ns("source")));
                            let surf_sid = match src {
                                Some(s) => doc.el(s).text.clone().unwrap_or_default().trim_matches(|c: char| pyutil::is_space(c)).to_string(),
                                None => String::new(),
                            };
                            for np2 in doc.iter_tag(effect, &ns("newparam")) {
                                if doc.get(np2, "sid") != Some(surf_sid.as_str()) {
                                    continue;
                                }
                                let surf = doc.find(np2, &ns("surface"));
                                let init = surf.and_then(|s| doc.find(s, &ns("init_from")));
                                if let Some(i) = init {
                                    if let Some(t) = doc.el(i).text.as_deref().filter(|t| !t.is_empty()) {
                                        target = self.by_id.get(t.trim_matches(|c: char| pyutil::is_space(c))).copied();
                                    }
                                }
                            }
                            break;
                        }
                    }
                    if let Some(t) = target.filter(|&t| self.tag_is(t, "image")) {
                        return Ok((None, Some(t)));
                    }
                }
            }
        }
        Ok((None, None))
    }

    /// `_material_map(dae, inst_geom_el)`: symbol to colour and image.
    fn material_map(&self, inst_geom: usize) -> Result<HashMap<Option<String>, MatInfo>, String> {
        let doc = self.doc;
        let mut out: HashMap<Option<String>, MatInfo> = HashMap::new();
        for im in doc.iter_tag(inst_geom, &ns("instance_material")) {
            let Some(mat) = self.reference(doc.get(im, "target")) else { continue };
            let (color, img) = self.effect_diffuse(mat)?;
            let path = img.and_then(|i| self.image_file(i));
            let mut color = color.map(ColorV::Rgb);
            if color.is_none() {
                if let Some(i) = img {
                    color = self.image_color(i);
                }
            }
            if color.is_some() || path.is_some() {
                out.insert(doc.get(im, "symbol").map(str::to_string), MatInfo { color, path });
            }
        }
        Ok(out)
    }

    /// `_node_matrix(node_el)`.
    fn node_matrix(&self, node: usize) -> Result<M4, String> {
        let doc = self.doc;
        let mut m = M4::identity();
        for &el in &doc.el(node).children {
            let text = doc.el(el).text.as_deref();
            match local(&doc.el(el).tag) {
                "matrix" => {
                    let vals = floats(text)?;
                    if vals.len() == 16 {
                        let mut v = [0.0f32; 16];
                        for (k, x) in vals.iter().enumerate() {
                            v[k] = *x as f32;
                        }
                        m = m.mul(&M4::from_rows(v));
                    }
                }
                "translate" => {
                    let v = floats(text)?;
                    if v.len() >= 3 {
                        m.translate(v[0] as f32, v[1] as f32, v[2] as f32);
                    }
                }
                "rotate" => {
                    let v = floats(text)?;
                    if v.len() >= 4 {
                        m.rotate(v[3] as f32, v[0] as f32, v[1] as f32, v[2] as f32);
                    }
                }
                "scale" => {
                    let v = floats(text)?;
                    if v.len() >= 3 {
                        m.scale(v[0] as f32, v[1] as f32, v[2] as f32);
                    }
                }
                _ => {}
            }
        }
        Ok(m)
    }

    /// `_prim_uvs(dae, prim_el)`.
    fn prim_uvs(&self, prim: usize) -> Result<Vec<(f64, f64)>, String> {
        let (_v, uv_off, uv_src, _s) = self.prim_inputs(prim)?;
        let Some(uv_src) = uv_src.filter(|s| uv_off.is_some() && !s.is_empty()) else { return Ok(Vec::new()) };
        let Some(src) = self.reference(Some(&uv_src)) else { return Ok(Vec::new()) };
        let sf = self.source_floats(src)?;
        let (data, stride) = (&sf.0, sf.1.max(2));
        let mut out = Vec::new();
        let mut i = 0i64;
        while i < data.len() as i64 - stride + 1 {
            out.push((data[i as usize], data[i as usize + 1]));
            i += stride;
        }
        Ok(out)
    }

    /// `_collect_direct(dae, node_el, m, out)`.
    fn collect_direct(&self, node: usize, m: &M4, out: &mut Vec<Loop>) -> Result<(), String> {
        let doc = self.doc;
        for &el in &doc.el(node).children {
            if !self.tag_is(el, "instance_geometry") {
                continue;
            }
            let Some(geom) = self.reference(doc.get(el, "url")) else { continue };
            let mats = self.material_map(el)?;
            let Some(mesh_el) = doc.find(geom, &ns("mesh")) else { continue };
            let positions = self.positions(mesh_el)?;
            let world: Vec<V3> = positions.iter().map(|&p| m.map(p)).collect();
            for &prim in &doc.el(mesh_el).children {
                if !["triangles", "polylist", "polygons"].contains(&local(&doc.el(prim).tag)) {
                    continue;
                }
                let mat = mats.get(&doc.get(prim, "material").map(str::to_string));
                let color = mat.and_then(|m| m.color.clone());
                let path = mat.and_then(|m| m.path.clone());
                let uvpts = if path.is_some() { self.prim_uvs(prim)? } else { Vec::new() };
                for (lp, ul) in self.prim_loops(prim, world.len(), uvpts.len())? {
                    let tex = match (&ul, &path) {
                        (Some(ul), Some(p)) => Some((p.clone(), ul.iter().map(|&i| uvpts[i as usize]).collect())),
                        _ => None,
                    };
                    out.push(Loop { pts: lp.iter().map(|&i| world[i as usize]).collect(), color: color.clone(), tex });
                }
            }
        }
        Ok(())
    }

    /// `_looks_faceme(dae, loops)`.
    fn looks_faceme(&self, loops: &[Loop], h: Hypot) -> Result<bool, String> {
        if !(1..=MAX_FUSE_LOOPS).contains(&loops.len()) {
            return Ok(false);
        }
        let mut path: Option<&str> = None;
        let mut n_ref: Option<V3> = None;
        let mut corners: HashSet<[i64; 3]> = HashSet::new();
        for l in loops {
            let Some((p, _)) = &l.tex else { return Ok(false) };
            match path {
                None => path = Some(p),
                Some(q) if q != p => return Ok(false),
                _ => {}
            }
            let zpts: Vec<V3> = l.pts.iter().map(|&p| self.to_zup(p)).collect();
            let n = newell(&zpts);
            let ln = n.length(h);
            if (ln as f64) < 1e-12 {
                continue;
            }
            let n = n.div(ln);
            if (n.2 as f64).abs() > 0.08 {
                return Ok(false);
            }
            match n_ref {
                None => n_ref = Some(n),
                Some(r) => {
                    if (n.dot(r) as f64).abs() < 0.99 {
                        return Ok(false);
                    }
                }
            }
            for p in &zpts {
                corners.insert(corner_key(*p));
            }
        }
        if n_ref.is_none() {
            return Ok(false);
        }
        Ok(corners.len() > 4 || path.is_some_and(|p| self.image_has_cutout(p)))
    }

    /// `_node_label(node_el)`.
    fn node_label(&self, node: usize) -> String {
        let doc = self.doc;
        [doc.get(node, "name"), doc.get(node, "id")].into_iter().flatten().find(|s| !s.is_empty()).unwrap_or("node").to_string()
    }

    /// `_collect(dae, node_el, xform, out, depth, faceme, instances)`: walk a
    /// node tree, baking transforms. With `faceme` given, a child component
    /// whose geometry is a face-me silhouette is pulled out into it instead
    /// of flattening into the parent; with `instances` given, a reused
    /// component worth sharing is pulled out as an [`Instance`] instead of
    /// baking a copy.
    fn collect(&self, node: usize, xform: &M4, out: &mut Vec<Loop>, depth: usize, h: Hypot, mut sinks: Sinks) -> Result<(), String> {
        if depth > MAX_DEPTH {
            return Ok(());
        }
        let doc = self.doc;
        let m = xform.mul(&self.node_matrix(node)?);
        self.collect_direct(node, &m, out)?;
        for &el in &doc.el(node).children {
            let (target, d2) = match local(&doc.el(el).tag) {
                "node" => (Some(el), depth),
                "instance_node" => (self.reference(doc.get(el, "url")), depth + 1),
                _ => (None, depth),
            };
            let Some(target) = target else { continue };
            if let (true, Some(instances)) = (local(&doc.el(el).tag) == "instance_node", sinks.instances.as_deref_mut()) {
                let count = self.inst_counts.get(doc.get(el, "url").unwrap_or("")).copied().unwrap_or(1);
                let polys = self.polys_of(target, d2);
                if count >= 2 && polys >= INST_MIN_POLYS && polys * (count as i64 - 1) >= INST_MIN_SAVED {
                    let name = doc.get(el, "name").filter(|n| !n.is_empty()).map_or_else(|| self.node_label(target), str::to_string);
                    instances.push(Instance { name, node: target, xform: m, depth: d2 });
                    continue;
                }
            }
            if sinks.faceme.is_some() {
                let mut sub = Vec::new();
                self.collect(target, &m, &mut sub, d2, h, Sinks { faceme: sinks.faceme.as_deref_mut(), instances: sinks.instances.as_deref_mut() })?;
                if !sub.is_empty() && self.looks_faceme(&sub, h)? {
                    let name = doc.get(el, "name").filter(|n| !n.is_empty()).map_or_else(|| self.node_label(target), str::to_string);
                    if let Some(faceme) = sinks.faceme.as_deref_mut() {
                        faceme.push((name, sub));
                    }
                } else {
                    out.extend(sub);
                }
            } else {
                self.collect(target, &m, out, d2, h, Sinks { faceme: None, instances: sinks.instances.as_deref_mut() })?;
            }
        }
        Ok(())
    }

    /// `_prim_count(prim)`.
    fn prim_count(&self, prim: usize) -> i64 {
        match local(&self.doc.el(prim).tag) {
            "triangles" | "polylist" => py_int(self.doc.get(prim, "count").unwrap_or("0")).unwrap_or(0),
            "polygons" => self.doc.findall(prim, &ns("p")).len() as i64,
            _ => 0,
        }
    }

    /// `_direct_count`.
    fn direct_count(&self, node: usize) -> i64 {
        let doc = self.doc;
        let mut n = 0;
        for &el in &doc.el(node).children {
            if !self.tag_is(el, "instance_geometry") {
                continue;
            }
            let geom = self.reference(doc.get(el, "url"));
            if let Some(mesh_el) = geom.and_then(|g| doc.find(g, &ns("mesh"))) {
                n += doc.el(mesh_el).children.iter().map(|&p| self.prim_count(p)).sum::<i64>();
            }
        }
        n
    }

    /// `_subtree_count`.
    fn subtree_count(&self, node: usize, depth: usize) -> i64 {
        if depth > MAX_DEPTH {
            return 0;
        }
        let doc = self.doc;
        let mut n = self.direct_count(node);
        for &el in &doc.el(node).children {
            match local(&doc.el(el).tag) {
                "node" => n += self.subtree_count(el, depth),
                "instance_node" => {
                    if let Some(t) = self.reference(doc.get(el, "url")) {
                        n += self.subtree_count(t, depth + 1);
                    }
                }
                _ => {}
            }
        }
        n
    }
}

/// `fuse._newell(pts)`: float32 terms from double products, summed in float32.
fn newell(pts: &[V3]) -> V3 {
    let m = pts.len();
    let mut n = V3(0.0, 0.0, 0.0);
    for i in 0..m {
        let (a, b) = (pts[i], pts[(i + 1) % m]);
        let t = V3(
            ((a.1 as f64 - b.1 as f64) * (a.2 as f64 + b.2 as f64)) as f32,
            ((a.2 as f64 - b.2 as f64) * (a.0 as f64 + b.0 as f64)) as f32,
            ((a.0 as f64 - b.0 as f64) * (a.1 as f64 + b.1 as f64)) as f32,
        );
        n = V3(n.0 + t.0, n.1 + t.1, n.2 + t.2);
    }
    n
}

/// `fuse._key(p)`: `round(x, 4)` of each coordinate (an exact integer cell for
/// a float32).
fn corner_key(p: V3) -> [i64; 3] {
    let r = |x: f32| (x as f64 * 1e4).round_ties_even() as i64;
    [r(p.0), r(p.1), r(p.2)]
}

// ---- attributes and loading -----------------------------------------------------

/// The light tint a missing image gets, from `hash(ref)`.
fn tint(reference: &str) -> Vec<f64> {
    let h = py_str_hash_seed0(reference);
    [0, 4, 8].iter().map(|&s| 0.78 + (((h >> s) & 15) as f64) / 100.0).collect()
}

/// What Qt's loaders do with a file: read it, or (the bytes are not an image
/// it knows) answer with a null image.
fn looks_like_an_image(data: &[u8]) -> bool {
    data.starts_with(b"\x89PNG") || data.starts_with(&[0xff, 0xd8, 0xff]) || data.starts_with(b"GIF8") || data.starts_with(b"BM") || data.starts_with(b"II*\0") || data.starts_with(b"MM\0*")
        || (data.len() > 12 && &data[..4] == b"RIFF" && &data[8..12] == b"WEBP")
}

/// `_image_color` for an image that exists: the average of
/// `QImage(path).scaled(1, 1, IgnoreAspectRatio, SmoothTransformation)`, the
/// tint of `reference` if Qt cannot read the file.
fn average_or_tint(file: &Path, reference: &str) -> Result<Vec<f64>, String> {
    let data = std::fs::read(file).map_err(|e| e.to_string())?;
    if !looks_like_an_image(&data) {
        return Ok(tint(reference));
    }
    match crate::img::smooth_average(&data) {
        Ok(c) => Ok(c.to_vec()),
        Err(e) if e.starts_with("not ported: unreadable PNG") => Ok(tint(reference)),
        Err(e) => Err(e),
    }
}

/// `_face_attrs(zpts, color, tex)`, in Python\'s key order.
fn face_attrs(zpts: &[V3], color: &Option<ColorV>, tex: &Option<Tex>) -> Result<Vec<(String, Value)>, String> {
    let rgb = |c: &[f64]| -> Vec<(String, Value)> {
        vec![
            ("color".to_string(), Value::Array(c.iter().take(3).map(|&x| Value::Float(x)).collect())),
            ("back".to_string(), Value::Bool(true)),
        ]
    };
    if let Some((path, uvs)) = tex {
        if let Some(m) = fit_uv_affine(zpts, uvs) {
            let glu = py_hypot3(m[0], m[1], m[2]);
            let glv = py_hypot3(m[4], m[5], m[6]);
            let texture = vec![
                ("path".to_string(), Value::Str(path.clone())),
                ("uvw".to_string(), Value::Array(m.iter().map(|&x| Value::Float(x)).collect())),
                ("sw".to_string(), Value::Float(if glu > 1e-9 { 1.0 / glu } else { 1.0 })),
                ("sh".to_string(), Value::Float(if glv > 1e-9 { 1.0 / glv } else { 1.0 })),
            ];
            return Ok(vec![("texture".to_string(), Value::Object(texture)), ("back".to_string(), Value::Bool(true))]);
        }
    }
    match color {
        Some(ColorV::Rgb(c)) => Ok(rgb(c)),
        Some(ColorV::ImageAverage { file, reference }) => Ok(rgb(&average_or_tint(file, reference)?)),
        None => Ok(Vec::new()),
    }
}

/// `_adopt_geolocation(scene, root)`.
fn adopt_geolocation(scene: &mut Scene, doc: &Doc) {
    if scene.extra.georef_datum.is_some() {
        return;
    }
    let Some(asset) = doc.find(Doc::ROOT, &ns("asset")) else { return };
    let Some(cov) = doc.find(asset, &ns("coverage")) else { return };
    let Some(geo) = doc.find(cov, &ns("geographic_location")) else { return };
    let (Some(lat_el), Some(lon_el)) = (doc.find(geo, &ns("latitude")), doc.find(geo, &ns("longitude"))) else { return };
    let (Some(lat_t), Some(lon_t)) = (doc.el(lat_el).text.as_deref(), doc.el(lon_el).text.as_deref()) else { return };
    let alt_t = doc.find(geo, &ns("altitude")).and_then(|a| doc.el(a).text.as_deref());
    let (Some(lat), Some(lon)) = (pyutil::float(lat_t), pyutil::float(lon_t)) else { return };
    let alt = match alt_t {
        Some(t) => match pyutil::float(t) {
            Some(a) => a,
            None => return,
        },
        None => 0.0,
    };
    scene.extra.georef_datum = Some(Value::Object(vec![
        ("lat".to_string(), Value::Float(lat)),
        ("lon".to_string(), Value::Float(lon)),
        ("alt".to_string(), Value::Float(alt)),
    ]));
}

/// `_INST_MIN_SAVED` and `_INST_MIN_POLYS`: a reused component becomes a
/// shared instance only when sharing saves that many polygons and the
/// component is that big.
const INST_MIN_SAVED: i64 = 400;
const INST_MIN_POLYS: i64 = 60;
/// `_MAX_GROUPS` and `_SPLIT_MIN`: how far the greedy splitter goes.
const MAX_GROUPS: usize = 250;
const SPLIT_MIN: i64 = 4000;

/// A reused component pulled out of a node walk: `(name, target_node,
/// pre_xform, depth)`.
struct Instance {
    name: String,
    node: usize,
    xform: M4,
    depth: usize,
}

/// A face-me sprite: `(name, loops)`.
type Sprite = (String, Vec<Loop>);

/// What a node walk may set aside: the `faceme` and `instances` arguments of
/// `_collect`.
#[derive(Default)]
struct Sinks<'a> {
    faceme: Option<&'a mut Vec<Sprite>>,
    instances: Option<&'a mut Vec<Instance>>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    Subtree,
    Direct,
}

/// A heap entry of `_bucketize`: `(-tris, tie, kind, node, xform, depth,
/// name, via_inst)`.
struct Entry {
    neg: i64,
    tie: u64,
    kind: Kind,
    node: usize,
    xform: M4,
    depth: usize,
    name: String,
    via: bool,
}

impl Dae<'_> {
    /// `_bucketize(dae, top_nodes, faceme, instances)`: split the visual
    /// scene into `(name, loops)` buckets along the node hierarchy, at most
    /// [`MAX_GROUPS`] of them, the largest assemblies split first. Face-me
    /// silhouettes found anywhere land in `faceme`, reused components in
    /// `instances`.
    fn bucketize(&self, top_nodes: &[usize], h: Hypot, faceme: &mut Vec<Sprite>, instances: &mut Vec<Instance>) -> Result<Vec<Sprite>, String> {
        let doc = self.doc;
        let mut tie = 0u64;
        let mut next_tie = || {
            tie += 1;
            tie - 1
        };
        // Python's heapq: the smallest `(neg, tie)` first; `tie` is unique.
        let mut heap: Vec<Entry> = Vec::new();
        let pop = |heap: &mut Vec<Entry>| -> Option<Entry> {
            let i = (0..heap.len()).min_by_key(|&i| (heap[i].neg, heap[i].tie))?;
            Some(heap.swap_remove(i))
        };
        for &nd in top_nodes {
            let n = self.subtree_count(nd, 0);
            if n > 0 {
                heap.push(Entry { neg: -n, tie: next_tie(), kind: Kind::Subtree, node: nd, xform: M4::identity(), depth: 0, name: self.node_label(nd), via: false });
            }
        }
        let mut finals: Vec<Entry> = Vec::new();
        while !heap.is_empty() && finals.len() + heap.len() < MAX_GROUPS {
            let Some(entry) = pop(&mut heap) else { break };
            if -entry.neg <= SPLIT_MIN {
                heap.push(entry);
                break; // the largest left is small: all are done
            }
            let kids: Vec<usize> = doc.el(entry.node).children.iter().copied().filter(|&e| self.tag_is(e, "node")).collect();
            let insts: Vec<usize> = doc.el(entry.node).children.iter().copied().filter(|&e| self.tag_is(e, "instance_node")).collect();
            if entry.kind == Kind::Direct || entry.depth > MAX_DEPTH || (kids.is_empty() && insts.is_empty()) {
                finals.push(entry); // big but unsplittable: keep as it is
                continue;
            }
            let m = entry.xform.mul(&self.node_matrix(entry.node)?);
            let dg = self.direct_count(entry.node);
            if dg > 0 {
                heap.push(Entry { neg: -dg, tie: next_tie(), kind: Kind::Direct, node: entry.node, xform: m, depth: entry.depth, name: entry.name.clone(), via: false });
            }
            for el in kids {
                let n = self.subtree_count(el, entry.depth);
                if n > 0 {
                    heap.push(Entry { neg: -n, tie: next_tie(), kind: Kind::Subtree, node: el, xform: m, depth: entry.depth, name: self.node_label(el), via: false });
                }
            }
            for el in insts {
                let Some(target) = self.reference(doc.get(el, "url")) else { continue };
                let n = self.subtree_count(target, entry.depth + 1);
                if n > 0 {
                    let name = doc.get(el, "name").filter(|n| !n.is_empty()).map_or_else(|| self.node_label(target), str::to_string);
                    heap.push(Entry { neg: -n, tie: next_tie(), kind: Kind::Subtree, node: target, xform: m, depth: entry.depth + 1, name, via: true });
                }
            }
        }
        finals.extend(heap);
        finals.sort_by_key(|e| (e.neg, e.tie));
        let mut bucket_inst: HashMap<usize, usize> = HashMap::new();
        for e in &finals {
            if e.via && e.kind == Kind::Subtree {
                *bucket_inst.entry(e.node).or_insert(0) += 1;
            }
        }
        let mut buckets: Vec<Sprite> = Vec::new();
        for e in finals {
            if e.via && e.kind == Kind::Subtree && bucket_inst.get(&e.node).copied().unwrap_or(0) >= 2 {
                instances.push(Instance { name: e.name, node: e.node, xform: e.xform, depth: e.depth });
                continue;
            }
            let mut loops = Vec::new();
            if e.kind == Kind::Direct {
                self.collect_direct(e.node, &e.xform, &mut loops)?;
            } else {
                self.collect(e.node, &e.xform, &mut loops, e.depth, h, Sinks { faceme: Some(&mut *faceme), instances: Some(&mut *instances) })?;
                if !loops.is_empty() && self.looks_faceme(&loops, h)? {
                    faceme.push((e.name, loops)); // the bucket itself is a sprite
                    continue;
                }
            }
            if !loops.is_empty() {
                buckets.push((e.name, loops));
            }
        }
        Ok(buckets)
    }

    /// `build_mesh(loops)` of the reference import: the polygons in Z-up
    /// metres, fused, their smooth seams softened.
    fn build_mesh(&self, loops: &[Loop], h: Hypot) -> Result<Mesh, String> {
        let mut raw: Vec<RawLoop> = Vec::with_capacity(loops.len());
        for l in loops {
            let zpts: Vec<V3> = l.pts.iter().map(|&p| self.to_zup(p)).collect();
            let attrs = face_attrs(&zpts, &l.color, &l.tex)?;
            raw.push((zpts, attrs));
        }
        let mut target = Mesh::new();
        target.hypot = h;
        add_fused(&mut target, &fuse_coplanar_loops(&raw, 0.99999, false, h));
        soften_smooth_edges(&mut target, h);
        Ok(target)
    }
}

/// `load_dae(scene, path)` on the file's bytes; `path` is where it lives
/// (the textures are looked up beside it). `stitch` stands where Python runs
/// `run_stitch(..., coplanar_merge=True)` and `orient_outward`;
/// `group_counter` is the process-wide `Group N` counter.
pub fn load_dae_bytes(scene: &mut Scene, data: &[u8], path: &Path, h: Hypot, stitch: Option<Stitch>, group_counter: &mut i64) -> Result<(), String> {
    let doc = xml::parse(data)?;
    let dae = Dae::new(&doc, Some(path.parent().map(Path::to_path_buf).unwrap_or_default()))?;
    adopt_geolocation(scene, &doc);
    let mut top_nodes: Vec<usize> = Vec::new();
    for vs in doc.iter_tag(Doc::ROOT, &ns("visual_scene")) {
        top_nodes.extend(doc.findall(vs, &ns("node")));
    }
    let total: i64 = top_nodes.iter().map(|&n| dae.subtree_count(n, 0)).sum();
    if total > MAX_FUSE_LOOPS as i64 {
        return load_reference(scene, &dae, &top_nodes, h, group_counter);
    }

    let mut pending: Vec<Loop> = Vec::new();
    let mut faceme: Vec<Sprite> = Vec::new();
    for &node in &top_nodes {
        let mut sub = Vec::new();
        dae.collect(node, &M4::identity(), &mut sub, 0, h, Sinks { faceme: Some(&mut faceme), instances: None })?;
        if !sub.is_empty() && dae.looks_faceme(&sub, h)? {
            faceme.push((dae.node_label(node), sub)); // the top node is a sprite
        } else {
            pending.extend(sub);
        }
    }
    // Face-me sprites become billboard groups even in a small import.
    for (name, loops) in &faceme {
        let target = dae.build_mesh(loops, h)?;
        if !target.faces.is_empty() {
            let mut g = new_group(scene, target, name, group_counter);
            g.billboard = Some(Value::Str("mesh".to_string()));
            scene.groups.push(g);
        }
    }
    if pending.is_empty() && faceme.is_empty() {
        // No visual scene (bare geometry library): import it un-instanced.
        for geom in doc.iter_tag(Doc::ROOT, &ns("geometry")) {
            let Some(mesh_el) = doc.find(geom, &ns("mesh")) else { continue };
            let positions = dae.positions(mesh_el)?;
            for &prim in &doc.el(mesh_el).children {
                if ["triangles", "polylist", "polygons"].contains(&local(&doc.el(prim).tag)) {
                    for (lp, _ul) in dae.prim_loops(prim, positions.len(), 0)? {
                        pending.push(Loop { pts: lp.iter().map(|&i| positions[i as usize]).collect(), color: None, tex: None });
                    }
                }
            }
        }
    }
    if pending.is_empty() {
        if !faceme.is_empty() {
            return Ok(());
        }
        return Err("No geometry found in the COLLADA file".to_string());
    }
    if pending.len() > MAX_FUSE_LOOPS {
        // A bare geometry library too big for the editable pipeline: one
        // reference group named after the file.
        let target = dae.build_mesh(&pending, h)?;
        let stem = path.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
        let group = new_group(scene, target, &stem, group_counter);
        scene.groups.push(group);
        return Ok(());
    }

    let mesh = &mut scene.meshes[0];
    let mut seed: HashSet<Key> = HashSet::new();
    let mut new_faces: HashSet<FaceId> = HashSet::new();
    for l in &pending {
        let pts: Vec<V3> = l.pts.iter().map(|&p| dae.to_zup(p)).collect();
        // Python's `add_face` raises on a NaN or infinite corner (its weld
        // key is an int), after welding the corners before it; the import
        // skips such a polygon.
        if let Some(bad) = pts.iter().position(|c| !(c.0.is_finite() && c.1.is_finite() && c.2.is_finite())) {
            for &c in &pts[..bad] {
                mesh.vertex(c);
            }
            continue;
        }
        let f = mesh.add_face(&pts, &[]);
        new_faces.insert(f);
        for (k, v) in face_attrs(&pts, &l.color, &l.tex)? {
            pyval::set(&mut mesh.face_mut(f).attrs, &k, v);
        }
        for &p in &pts {
            seed.insert(key(p));
        }
    }
    if let Some(stitch) = stitch {
        stitch(mesh, &seed, &new_faces);
    }
    Ok(())
}

/// The reference import of a model of more than [`MAX_FUSE_LOOPS`]
/// polygons: one group per assembly of the node tree, a shared prototype
/// per reused component, billboard groups for the face-me sprites.
fn load_reference(scene: &mut Scene, dae: &Dae, top_nodes: &[usize], h: Hypot, counter: &mut i64) -> Result<(), String> {
    let mut faceme: Vec<Sprite> = Vec::new();
    let mut instances: Vec<Instance> = Vec::new();
    let buckets = dae.bucketize(top_nodes, h, &mut faceme, &mut instances)?;
    if buckets.is_empty() && faceme.is_empty() && instances.is_empty() {
        return Err("No geometry found in the COLLADA file".to_string());
    }
    let zm = dae.zup_matrix();
    let (zm_inv, _) = zm.inverted();
    for inst in &instances {
        dae.polys_of(inst.node, inst.depth);
    }
    for (name, loops) in &buckets {
        let target = dae.build_mesh(loops, h)?;
        if !target.faces.is_empty() {
            let g = new_group(scene, target, name, counter);
            scene.groups.push(g);
        }
    }
    // Components: ONE prototype mesh in Z-up LOCAL coordinates, shared by
    // every copy; each copy is a group with only a local to world matrix. A
    // component carrying a face-me sprite inside falls back to per-copy
    // geometry (the sprite must be extracted at its world spot).
    let mut protos: HashMap<usize, Option<MeshId>> = HashMap::new();
    for inst in &instances {
        if let std::collections::hash_map::Entry::Vacant(slot) = protos.entry(inst.node) {
            let mut probe: Vec<Sprite> = Vec::new();
            let mut local_loops: Vec<Loop> = Vec::new();
            dae.collect(inst.node, &M4::identity(), &mut local_loops, inst.depth, h, Sinks { faceme: Some(&mut probe), instances: None })?;
            let proto = if probe.is_empty() {
                let mesh = dae.build_mesh(&local_loops, h)?;
                if mesh.faces.is_empty() { None } else { Some(push_mesh(scene, mesh)) }
            } else {
                None // sprites inside: per-copy fallback
            };
            slot.insert(proto);
        }
        if let Some(Some(id)) = protos.get(&inst.node) {
            let mut g = Group::new(*id, group_name(counter, &inst.name));
            g.xform = Some(zm.mul(&inst.xform).mul(&zm_inv));
            scene.groups.push(g);
            continue;
        }
        // Fallback: bake this copy in world coordinates (sprites and all).
        let mut loops = Vec::new();
        dae.collect(inst.node, &inst.xform, &mut loops, inst.depth, h, Sinks { faceme: Some(&mut faceme), instances: None })?;
        if !loops.is_empty() && dae.looks_faceme(&loops, h)? {
            faceme.push((inst.name.clone(), loops));
            continue;
        }
        let target = dae.build_mesh(&loops, h)?;
        if !target.faces.is_empty() {
            let g = new_group(scene, target, &inst.name, counter);
            scene.groups.push(g);
        }
    }
    for (name, loops) in &faceme {
        let target = dae.build_mesh(loops, h)?;
        if !target.faces.is_empty() {
            let mut g = new_group(scene, target, name, counter);
            g.billboard = Some(Value::Str("mesh".to_string())); // geometry turns toward the camera
            scene.groups.push(g);
        }
    }
    Ok(())
}

/// `load_dae(scene, path)`.
pub fn load_dae(scene: &mut Scene, path: &Path, h: Hypot, stitch: Option<Stitch>, group_counter: &mut i64) -> Result<(), String> {
    let data = std::fs::read(path).map_err(|e| e.to_string())?;
    load_dae_bytes(scene, &data, path, h, stitch, group_counter)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unquote_like_urllib() {
        assert_eq!(unquote("a%20b"), "a b");
        assert_eq!(unquote("100%"), "100%");
        assert_eq!(unquote("%zz%41"), "%zzA");
        assert_eq!(unquote("r%C3%BAstico"), "r\u{fa}stico");
        assert_eq!(unquote("%FF"), "\u{fffd}");
        assert_eq!(unquote("%4"), "%4");
    }

    #[test]
    fn python_ints() {
        assert_eq!(py_int(" 12 "), Ok(12));
        assert_eq!(py_int("-1_0"), Ok(-10));
        assert!(py_int("1.0").is_err());
        assert!(py_int("").is_err());
        assert_eq!(py_int("99999999999999999999999"), Ok(i64::MAX));
    }
}
