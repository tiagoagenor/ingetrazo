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
//! Not ported yet, each an `Err` starting with `not ported`:
//! - models of more than [`MAX_FUSE_LOOPS`] polygons (`fuse_coplanar_loops`
//!   builds one group per assembly) and face-me sprites (billboard groups);
//! - the average colour of an image (`QImage.scaled(1, 1, ...)`): it is only
//!   read when a polygon falls back to the material colour, and then only for
//!   a material whose image file exists next to the `.dae`.
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
use ingetrazo_model::mesh::{key, FaceId, Key};
use ingetrazo_model::pyval;
use ingetrazo_model::scene::Scene;

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
    /// The average of an image that exists on disk (not ported).
    ImageAverage,
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
        Ok(Dae { doc, base_dir, by_id, scale, up, src_cache: RefCell::new(HashMap::new()) })
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
        if self.candidate(&reference).is_some_and(|c| c.is_file()) {
            return Some(ColorV::ImageAverage);
        }
        let h = py_str_hash_seed0(&reference);
        Some(ColorV::Rgb([0, 4, 8].iter().map(|&s| 0.78 + (((h >> s) & 15) as f64) / 100.0).collect()))
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
        if corners.len() > 4 {
            return Ok(true);
        }
        Err("not ported: a rectangular face-me candidate needs the transparency of its image".to_string())
    }

    /// `_collect(dae, node_el, xform, out, depth, faceme)` with `faceme`
    /// given: a child component that reads as a face-me sprite is where
    /// Python builds a billboard group (not ported).
    fn collect(&self, node: usize, xform: &M4, out: &mut Vec<Loop>, depth: usize, h: Hypot) -> Result<(), String> {
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
            let mut sub = Vec::new();
            self.collect(target, &m, &mut sub, d2, h)?;
            if !sub.is_empty() && self.looks_faceme(&sub, h)? {
                return Err("not ported: face-me sprites become billboard groups (fuse_coplanar_loops)".to_string());
            }
            out.extend(sub);
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

/// `_face_attrs(zpts, color, tex)`, in Python\'s key order.
fn face_attrs(zpts: &[V3], color: &Option<ColorV>, tex: &Option<Tex>) -> Result<Vec<(String, Value)>, String> {
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
        Some(ColorV::Rgb(c)) => Ok(vec![
            ("color".to_string(), Value::Array(c.iter().take(3).map(|&x| Value::Float(x)).collect())),
            ("back".to_string(), Value::Bool(true)),
        ]),
        Some(ColorV::ImageAverage) => Err("not ported: the average colour of an image (QImage.scaled)".to_string()),
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

/// `load_dae(scene, path)` on the file\'s bytes; `path` is where it lives
/// (the textures are looked up beside it). `stitch` stands where Python runs
/// `run_stitch(..., coplanar_merge=True)` and `orient_outward`.
pub fn load_dae_bytes(scene: &mut Scene, data: &[u8], path: &Path, h: Hypot, stitch: Option<Stitch>) -> Result<(), String> {
    let doc = xml::parse(data)?;
    let dae = Dae::new(&doc, Some(path.parent().map(Path::to_path_buf).unwrap_or_default()))?;
    adopt_geolocation(scene, &doc);
    let mut top_nodes: Vec<usize> = Vec::new();
    for vs in doc.iter_tag(Doc::ROOT, &ns("visual_scene")) {
        top_nodes.extend(doc.findall(vs, &ns("node")));
    }
    let total: i64 = top_nodes.iter().map(|&n| dae.subtree_count(n, 0)).sum();
    if total > MAX_FUSE_LOOPS as i64 {
        return Err("not ported: a model of more than 400 polygons becomes groups (fuse_coplanar_loops)".to_string());
    }

    let mut pending: Vec<Loop> = Vec::new();
    for &node in &top_nodes {
        let mut sub = Vec::new();
        dae.collect(node, &M4::identity(), &mut sub, 0, h)?;
        if !sub.is_empty() && dae.looks_faceme(&sub, h)? {
            return Err("not ported: face-me sprites become billboard groups (fuse_coplanar_loops)".to_string());
        }
        pending.extend(sub);
    }
    if pending.is_empty() {
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
        return Err("No geometry found in the COLLADA file".to_string());
    }
    if pending.len() > MAX_FUSE_LOOPS {
        return Err("not ported: a model of more than 400 polygons becomes one group (fuse_coplanar_loops)".to_string());
    }

    let mesh = &mut scene.meshes[0];
    let mut seed: HashSet<Key> = HashSet::new();
    let mut new_faces: HashSet<FaceId> = HashSet::new();
    for l in &pending {
        let pts: Vec<V3> = l.pts.iter().map(|&p| dae.to_zup(p)).collect();
        // Python\'s `add_face` raises on a NaN or infinite corner (its weld
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

/// `load_dae(scene, path)`.
pub fn load_dae(scene: &mut Scene, path: &Path, h: Hypot, stitch: Option<Stitch>) -> Result<(), String> {
    let data = std::fs::read(path).map_err(|e| e.to_string())?;
    load_dae_bytes(scene, &data, path, h, stitch)
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
