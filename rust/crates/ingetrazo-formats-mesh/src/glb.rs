// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! glTF 2.0 / GLB import: `formats/glb.py`, built the way Python builds it
//! without the Rust engine (`INGETRAZO_RUST` unset): triangles bulk-welded in
//! world Z-up metres, textures extracted to the texture cache and mapped
//! through per-triangle world to UV affine maps, facet seams softened.
//!
//! The arithmetic is NumPy's, replayed in the order OpenBLAS and NumPy's own
//! loops run it (found by comparing against NumPy; see the notes on
//! [`matmul_row`], [`einsum3`] and [`solve3`]): a texture's `uvw` is part of
//! the signature, and a 1-ulp difference changes its hash.
//!
//! Errors come back as `Err(text)`; for the ones Python raises itself the
//! text is its message (`ValueError`), for the ones NumPy, `json` or a
//! missing key raise it is only a description.

use std::path::{Path, PathBuf};

use ingetrazo_formats::pyjson::{loads, Value};
use ingetrazo_math::{Hypot, V3};
use ingetrazo_model::mesh::Mesh;
use ingetrazo_model::pyval;
use ingetrazo_model::scene::{Group, Scene};

use crate::img::has_alpha_channel;
use crate::tri::{face_area, face_normal};
use crate::world::attr;

/// Above this many edges the softening pass is skipped.
pub const SOFTEN_EDGE_CAP: usize = 500_000;

// ---- JSON helpers --------------------------------------------------------------

type R<T> = Result<T, String>;

fn jint(v: &Value) -> R<i64> {
    match v {
        Value::Int(_) => v.as_i64().ok_or_else(|| "Python int too large to convert".to_string()),
        Value::Bool(b) => Ok(*b as i64),
        _ => Err("TypeError: an integer is required".to_string()),
    }
}

fn jnum(v: &Value) -> R<f64> {
    match v {
        Value::Int(_) | Value::Float(_) | Value::Bool(_) => pyval::float(v).map_err(|e| e.msg),
        _ => Err("TypeError: a number is required".to_string()),
    }
}

/// `js.get(key)` where `js` must be a dict.
fn get<'a>(v: &'a Value, key: &str) -> R<Option<&'a Value>> {
    match v {
        Value::Object(_) => Ok(v.get(key)),
        _ => Err("AttributeError: object has no attribute 'get'".to_string()),
    }
}

/// `js.get(key, [])` as a list.
fn list<'a>(v: &'a Value, key: &str) -> R<&'a [Value]> {
    match get(v, key)? {
        None => Ok(&[]),
        Some(Value::Array(a)) => Ok(a),
        Some(_) => Err("TypeError: not a list".to_string()),
    }
}

/// `items[i]`, Python's negative indices included.
fn at(items: &[Value], i: i64) -> R<&Value> {
    let n = items.len() as i64;
    let j = if i < 0 { i + n } else { i };
    if j < 0 || j >= n {
        return Err("IndexError: list index out of range".to_string());
    }
    Ok(&items[j as usize])
}

/// `d[key]`.
fn req<'a>(v: &'a Value, key: &str) -> R<&'a Value> {
    match v {
        Value::Object(_) => v.get(key).ok_or_else(|| format!("KeyError: '{key}'")),
        _ => Err("TypeError: indices must be integers or slices, not str".to_string()),
    }
}

fn has(v: &Value, key: &str) -> bool {
    matches!(v, Value::Object(_)) && v.get(key).is_some()
}

// ---- the container -------------------------------------------------------------

fn base64_decode(text: &str) -> R<Vec<u8>> {
    let mut vals = Vec::new();
    let mut pad = 0;
    for c in text.bytes() {
        let v = match c {
            b'A'..=b'Z' => c - b'A',
            b'a'..=b'z' => c - b'a' + 26,
            b'0'..=b'9' => c - b'0' + 52,
            b'+' => 62,
            b'/' => 63,
            b'=' => {
                pad += 1;
                continue;
            }
            _ => continue,
        };
        if pad > 0 {
            break;
        }
        vals.push(v);
    }
    let rem = vals.len() % 4;
    if rem == 1 {
        return Err("binascii.Error: Invalid base64-encoded string".to_string());
    }
    if rem != 0 && pad < 4 - rem {
        return Err("binascii.Error: Incorrect padding".to_string());
    }
    let mut out = Vec::with_capacity(vals.len() * 3 / 4);
    for chunk in vals.chunks(4) {
        let mut acc = 0u32;
        for (k, &v) in chunk.iter().enumerate() {
            acc |= (v as u32) << (18 - 6 * k);
        }
        let bytes = acc.to_be_bytes();
        out.extend_from_slice(&bytes[1..1 + chunk.len() * 6 / 8]);
    }
    Ok(out)
}

/// `json.loads(bytes)`.
fn json_of(bytes: &[u8]) -> R<Value> {
    let text = std::str::from_utf8(bytes).map_err(|e| format!("UnicodeDecodeError: {e}"))?;
    let text = text.strip_prefix('\u{feff}').unwrap_or(text);
    loads(text).map_err(|e| format!("JSONDecodeError: {e}"))
}

/// `parse_container(path)`: the glTF JSON and its buffers.
pub fn parse_container(path: &Path) -> R<(Value, Vec<Vec<u8>>)> {
    let data = std::fs::read(path).map_err(|e| format!("OSError: {e}"))?;
    let (js, default) = if data.starts_with(b"glTF") {
        if data.len() < 12 {
            return Err("struct.error: unpack requires a buffer of 12 bytes".to_string());
        }
        let mut off = 12usize;
        let mut js = None;
        let mut bin_chunk: Vec<u8> = Vec::new();
        while off + 8 <= data.len() {
            let clen = u32::from_le_bytes([data[off], data[off + 1], data[off + 2], data[off + 3]]) as usize;
            let ctype = u32::from_le_bytes([data[off + 4], data[off + 5], data[off + 6], data[off + 7]]);
            let start = (off + 8).min(data.len());
            let end = (off + 8 + clen).min(data.len());
            let chunk = &data[start..end];
            if ctype == 0x4E4F534A {
                js = Some(json_of(chunk)?);
            } else if ctype == 0x004E4942 {
                bin_chunk = chunk.to_vec();
            }
            off += 8 + clen + (4 - clen % 4) % 4;
        }
        (js.ok_or_else(|| "GLB without a JSON chunk".to_string())?, bin_chunk)
    } else {
        (json_of(&data)?, Vec::new())
    };
    let mut buffers = Vec::new();
    for buf in list(&js, "buffers")? {
        match get(buf, "uri")? {
            None | Some(Value::Null) => buffers.push(default.clone()),
            Some(Value::Str(uri)) if uri.starts_with("data:") => {
                let payload = uri.split_once(',').ok_or("IndexError: list index out of range")?.1;
                buffers.push(base64_decode(payload)?);
            }
            Some(Value::Str(uri)) => {
                let file = path.parent().unwrap_or(Path::new("")).join(uri);
                buffers.push(std::fs::read(file).map_err(|e| format!("OSError: {e}"))?);
            }
            Some(_) => return Err("AttributeError: uri".to_string()),
        }
    }
    Ok((js, buffers))
}

// ---- accessors -------------------------------------------------------------------

/// An accessor as NumPy returns it: `(count, ncomp)` row-major.
pub enum Cols {
    Int(Vec<i64>),
    Float(Vec<f64>),
}

pub struct Accessor {
    pub count: usize,
    pub n: usize,
    pub data: Cols,
}

impl Accessor {
    fn float(&self, i: usize) -> f64 {
        match &self.data {
            Cols::Int(v) => v[i] as f64,
            Cols::Float(v) => v[i],
        }
    }

    /// Row `r`, column `c`, as float64.
    pub fn at(&self, r: usize, c: usize) -> f64 {
        self.float(r * self.n + c)
    }

    /// `.astype(np.int64)` of the flattened data (a float truncates).
    fn ints(&self) -> Vec<i64> {
        match &self.data {
            Cols::Int(v) => v.clone(),
            Cols::Float(v) => v.iter().map(|&x| x as i64).collect(),
        }
    }
}

/// `(struct code, size)` of a glTF componentType.
fn component(code: i64) -> R<(char, usize)> {
    Ok(match code {
        5120 => ('b', 1),
        5121 => ('B', 1),
        5122 => ('h', 2),
        5123 => ('H', 2),
        5125 => ('I', 4),
        5126 => ('f', 4),
        _ => return Err(format!("KeyError: {code}")),
    })
}

fn ncomp(kind: &str) -> R<usize> {
    Ok(match kind {
        "SCALAR" => 1,
        "VEC2" => 2,
        "VEC3" => 3,
        "VEC4" => 4,
        "MAT4" => 16,
        _ => return Err(format!("KeyError: '{kind}'")),
    })
}

/// `_accessor(js, buffers, idx)`.
pub fn read_accessor(js: &Value, buffers: &[Vec<u8>], idx: i64) -> R<Accessor> {
    let acc = at(list(js, "accessors")?, idx)?;
    if has(acc, "sparse") {
        return Err("sparse accessors are not supported".to_string());
    }
    let (fmt, size) = component(jint(req(acc, "componentType")?)?)?;
    let n = match req(acc, "type")? {
        Value::Str(k) => ncomp(k)?,
        _ => return Err("TypeError: unhashable type".to_string()),
    };
    let count = jint(req(acc, "count")?)?;
    let bv = at(list(js, "bufferViews")?, jint(req(acc, "bufferView")?)?)?;
    let buf_index = match get(bv, "buffer")? {
        Some(v) => jint(v)?,
        None => 0,
    };
    let buf = at_buf(buffers, buf_index)?;
    let off_bv = match get(bv, "byteOffset")? {
        Some(v) => jint(v)?,
        None => 0,
    };
    let off_acc = match get(acc, "byteOffset")? {
        Some(v) => jint(v)?,
        None => 0,
    };
    let start = off_bv + off_acc;
    let stride = match get(bv, "byteStride")? {
        Some(v) if pyval::truthy(v) => jint(v)?,
        _ => (size * n) as i64,
    };
    if count < 0 || start < 0 || start > buf.len() as i64 {
        return Err("ValueError: offset must be non-negative and no greater than buffer length".to_string());
    }
    let (count, start, stride, row) = (count as usize, start as usize, stride as usize, size * n);
    if count > 0 {
        let needed = if stride == row { count * row } else { stride * count - (stride - row) };
        if start + needed > buf.len() {
            return Err("ValueError: buffer is smaller than requested size".to_string());
        }
    }
    let mut raw_vals: Vec<f64> = Vec::with_capacity(count * n);
    let mut int_vals: Vec<i64> = Vec::with_capacity(count * n);
    let as_int = acc_is_int(fmt, n);
    for r in 0..count {
        for c in 0..n {
            let at = start + r * stride + c * size;
            let b = &buf[at..at + size];
            let (f, i) = match fmt {
                'b' => (b[0] as i8 as f64, b[0] as i8 as i64),
                'B' => (b[0] as f64, b[0] as i64),
                'h' => {
                    let v = i16::from_le_bytes([b[0], b[1]]);
                    (v as f64, v as i64)
                }
                'H' => {
                    let v = u16::from_le_bytes([b[0], b[1]]);
                    (v as f64, v as i64)
                }
                'I' => {
                    let v = u32::from_le_bytes([b[0], b[1], b[2], b[3]]);
                    (v as f64, v as i64)
                }
                _ => {
                    let v = f32::from_le_bytes([b[0], b[1], b[2], b[3]]);
                    (v as f64, 0)
                }
            };
            raw_vals.push(f);
            int_vals.push(i);
        }
    }
    if as_int {
        return Ok(Accessor { count, n, data: Cols::Int(int_vals) });
    }
    if matches!(get(acc, "normalized")?, Some(v) if pyval::truthy(v)) {
        let peak = match fmt {
            'b' => Some(127.0),
            'B' => Some(255.0),
            'h' => Some(32767.0),
            'H' => Some(65535.0),
            _ => None,
        };
        if let Some(p) = peak {
            for v in &mut raw_vals {
                *v /= p;
            }
        }
    }
    Ok(Accessor { count, n, data: Cols::Float(raw_vals) })
}

/// `acc["componentType"] == 5125 or fmt in ("H", "B") and n == 1`.
fn acc_is_int(fmt: char, n: usize) -> bool {
    fmt == 'I' || ((fmt == 'H' || fmt == 'B') && n == 1)
}

fn at_buf(buffers: &[Vec<u8>], i: i64) -> R<&Vec<u8>> {
    let n = buffers.len() as i64;
    let j = if i < 0 { i + n } else { i };
    if j < 0 || j >= n {
        return Err("IndexError: list index out of range".to_string());
    }
    Ok(&buffers[j as usize])
}

// ---- node transforms ---------------------------------------------------------------

type M44 = [[f64; 4]; 4];

fn eye() -> M44 {
    let mut m = [[0.0; 4]; 4];
    for (i, row) in m.iter_mut().enumerate() {
        row[i] = 1.0;
    }
    m
}

/// `a @ b` for 4 x 4 float64: OpenBLAS runs each element as the first
/// product, then the other three each fused into the running sum.
fn matmul4(a: &M44, b: &M44) -> M44 {
    let mut c = [[0.0; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            let mut acc = a[i][0] * b[0][j];
            for k in 1..4 {
                acc = a[i][k].mul_add(b[k][j], acc);
            }
            c[i][j] = acc;
        }
    }
    c
}

fn floats(v: &Value, want: usize) -> R<Vec<f64>> {
    match v {
        Value::Array(a) if a.len() == want => a.iter().map(jnum).collect(),
        _ => Err("ValueError: wrong number of values".to_string()),
    }
}

/// `local(nd)`: a node's own matrix.
fn local(nd: &Value) -> R<M44> {
    if let Some(m) = get(nd, "matrix")? {
        let v = floats(m, 16)?;
        // reshape(4, 4).T: glTF stores column-major.
        let mut out = [[0.0; 4]; 4];
        for r in 0..4 {
            for c in 0..4 {
                out[r][c] = v[c * 4 + r];
            }
        }
        return Ok(out);
    }
    let mut m = eye();
    if let Some(s) = get(nd, "scale")? {
        let mut v = match s {
            Value::Array(a) => a.iter().map(jnum).collect::<R<Vec<f64>>>()?,
            _ => return Err("TypeError: scale".to_string()),
        };
        v.push(1.0);
        if v.len() != 4 {
            return Err("ValueError: operands could not be broadcast together".to_string());
        }
        let mut d = [[0.0; 4]; 4];
        for i in 0..4 {
            d[i][i] = v[i];
        }
        m = matmul4(&m, &d);
    }
    if let Some(r) = get(nd, "rotation")? {
        let q = floats(r, 4)?;
        let (x, y, z, w) = (q[0], q[1], q[2], q[3]);
        let r3 = [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ];
        let mut rm = eye();
        for i in 0..3 {
            for j in 0..3 {
                rm[i][j] = r3[i][j];
            }
        }
        m = matmul4(&rm, &m);
    }
    if let Some(t) = get(nd, "translation")? {
        let v = floats(t, 3)?;
        let mut tm = eye();
        for i in 0..3 {
            tm[i][3] = v[i];
        }
        m = matmul4(&tm, &m);
    }
    Ok(m)
}

/// `_node_transforms(js)`: `(node index, world matrix)` for every node
/// reached from the scenes' roots, depth first.
fn node_transforms(js: &Value) -> R<Vec<(usize, M44)>> {
    let nodes = list(js, "nodes")?;
    let mut roots: Vec<i64> = Vec::new();
    let scenes = list(js, "scenes")?;
    if scenes.is_empty() {
        // `or [{}]`: no scene, no roots.
    }
    for sc in scenes {
        for n in list(sc, "nodes")? {
            roots.push(jint(n)?);
        }
    }
    if roots.is_empty() {
        roots = (0..nodes.len() as i64).collect();
    }
    let mut out = Vec::new();
    fn walk(nodes: &[Value], i: i64, parent: &M44, depth: usize, out: &mut Vec<(usize, M44)>) -> R<()> {
        if depth > 900 {
            return Err("RecursionError: maximum recursion depth exceeded".to_string());
        }
        let nd = at(nodes, i)?;
        let world = matmul4(parent, &local(nd)?);
        let n = nodes.len() as i64;
        out.push((if i < 0 { (i + n) as usize } else { i as usize }, world));
        for c in list(nd, "children")? {
            walk(nodes, jint(c)?, &world, depth + 1, out)?;
        }
        Ok(())
    }
    for r in roots {
        walk(nodes, r, &eye(), 0, &mut out)?;
    }
    Ok(out)
}

// ---- images and materials ---------------------------------------------------------

/// `hashlib.sha1(data).hexdigest()`.
pub fn sha1_hex(data: &[u8]) -> String {
    let mut h: [u32; 5] = [0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476, 0xC3D2E1F0];
    let mut msg = data.to_vec();
    msg.push(0x80);
    while msg.len() % 64 != 56 {
        msg.push(0);
    }
    msg.extend_from_slice(&((data.len() as u64) * 8).to_be_bytes());
    for block in msg.chunks(64) {
        let mut w = [0u32; 80];
        for i in 0..16 {
            w[i] = u32::from_be_bytes([block[4 * i], block[4 * i + 1], block[4 * i + 2], block[4 * i + 3]]);
        }
        for i in 16..80 {
            w[i] = (w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16]).rotate_left(1);
        }
        let [mut a, mut b, mut c, mut d, mut e] = h;
        for (i, wi) in w.iter().enumerate() {
            let (f, k) = match i {
                0..=19 => ((b & c) | (!b & d), 0x5A827999),
                20..=39 => (b ^ c ^ d, 0x6ED9EBA1),
                40..=59 => ((b & c) | (b & d) | (c & d), 0x8F1BBCDC),
                _ => (b ^ c ^ d, 0xCA62C1D6u32),
            };
            let t = a.rotate_left(5).wrapping_add(f).wrapping_add(e).wrapping_add(k).wrapping_add(*wi);
            e = d;
            d = c;
            c = b.rotate_left(30);
            b = a;
            a = t;
        }
        h = [h[0].wrapping_add(a), h[1].wrapping_add(b), h[2].wrapping_add(c), h[3].wrapping_add(d), h[4].wrapping_add(e)];
    }
    h.iter().map(|x| format!("{x:08x}")).collect()
}

/// `texture_cache_root()`: where the app keeps extracted textures.
pub fn texture_cache_root() -> PathBuf {
    if let Some(over) = std::env::var_os("INGETRAZO_TEXTURE_CACHE").filter(|v| !v.is_empty()) {
        return PathBuf::from(over);
    }
    let base = if cfg!(windows) {
        std::env::var_os("LOCALAPPDATA").map(PathBuf::from)
    } else if cfg!(target_os = "macos") {
        std::env::var_os("HOME").map(|h| PathBuf::from(h).join("Library/Application Support"))
    } else {
        std::env::var_os("XDG_DATA_HOME")
            .filter(|v| !v.is_empty())
            .map(PathBuf::from)
            .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".local/share")))
    };
    base.unwrap_or_else(|| PathBuf::from(".")).join("IngeTrazo").join("textures")
}

/// `Path(path).stem`.
fn stem_of(path: &Path) -> String {
    let name = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    match name.rfind('.') {
        Some(i) if i > 0 => name[..i].to_string(),
        _ => name,
    }
}

/// `str(path)` as Python shows a path: the text given, in the platform's form.
fn path_text(path: &Path) -> String {
    let t = path.to_string_lossy().into_owned();
    if cfg!(windows) {
        t.replace('/', "\\")
    } else {
        t
    }
}

/// `_extract_images(js, buffers, path)`: image index to the file written to
/// the texture cache (or `None`).
fn extract_images(js: &Value, buffers: &[Vec<u8>], path: &Path, cache_root: &Path) -> R<Vec<Option<String>>> {
    let stem: String = stem_of(path).chars().take(40).collect();
    let meta = std::fs::metadata(path).map_err(|e| format!("OSError: {e}"))?;
    let mtime_ns = meta
        .modified()
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let tag = &sha1_hex(format!("{}|{}|{}", path_text(path), meta.len(), mtime_ns).as_bytes())[..10];
    let out_dir = cache_root.join("glb").join(format!("{stem}-{tag}"));
    let mut paths = Vec::new();
    for (i, img) in list(js, "images")?.iter().enumerate() {
        let one = || -> R<Option<String>> {
            let blob: Vec<u8> = if has(img, "bufferView") {
                let bv = at(list(js, "bufferViews")?, jint(req(img, "bufferView")?)?)?;
                let buf = at_buf(buffers, match get(bv, "buffer")? {
                    Some(v) => jint(v)?,
                    None => 0,
                })?;
                let start = match get(bv, "byteOffset")? {
                    Some(v) => jint(v)?,
                    None => 0,
                };
                let len = jint(req(bv, "byteLength")?)?;
                let n = buf.len() as i64;
                let lo = if start < 0 { (start + n).max(0) } else { start.min(n) };
                let end = start + len;
                let hi = if end < 0 { (end + n).max(0) } else { end.min(n) };
                if hi > lo {
                    buf[lo as usize..hi as usize].to_vec()
                } else {
                    Vec::new()
                }
            } else {
                match get(img, "uri")? {
                    Some(Value::Str(uri)) if uri.starts_with("data:") => {
                        base64_decode(uri.split_once(',').ok_or("IndexError")?.1)?
                    }
                    Some(Value::Str(uri)) if !uri.is_empty() => {
                        std::fs::read(path.parent().unwrap_or(Path::new("")).join(uri)).map_err(|e| e.to_string())?
                    }
                    Some(Value::Str(_)) | None => return Ok(None),
                    Some(v) if !pyval::truthy(v) => return Ok(None),
                    Some(_) => return Err("AttributeError".to_string()),
                }
            };
            let ext = if blob.starts_with(b"\x89PNG") { ".png" } else { ".jpg" };
            std::fs::create_dir_all(&out_dir).map_err(|e| e.to_string())?;
            let fp = out_dir.join(format!("img_{i}{ext}"));
            if !fp.exists() {
                std::fs::write(&fp, &blob).map_err(|e| e.to_string())?;
            }
            Ok(Some(path_text(&fp)))
        };
        // A broken image loses its texture only.
        paths.push(one().unwrap_or(None));
    }
    Ok(paths)
}

/// What a primitive's material asks for.
struct Recipe {
    tex_path: Option<String>,
    uvt: (f64, f64, f64, f64),
    rgba: Option<Vec<f64>>,
    uv_set: i64,
    opacity: Option<f64>,
}

fn rgba_of(v: Option<&Value>) -> R<Option<Vec<f64>>> {
    match v {
        None | Some(Value::Null) => Ok(None),
        Some(Value::Array(a)) => Ok(Some(a.iter().map(jnum).collect::<R<Vec<f64>>>()?)),
        Some(_) => Err("TypeError: rgba".to_string()),
    }
}

/// `_material_recipe(js, mat_idx, image_paths)`.
fn material_recipe(js: &Value, mat_idx: Option<i64>, image_paths: &[Option<String>]) -> R<Recipe> {
    let mats = list(js, "materials")?;
    let none = Recipe { tex_path: None, uvt: (0.0, 0.0, 1.0, 1.0), rgba: None, uv_set: 0, opacity: None };
    let Some(mi) = mat_idx else { return Ok(none) };
    if mi >= mats.len() as i64 {
        return Ok(none);
    }
    let mat = at(mats, mi)?;
    let sg = match get(mat, "extensions")? {
        Some(e) => get(e, "KHR_materials_pbrSpecularGlossiness")?,
        None => None,
    };
    let (tex_info, rgba_v): (Option<&Value>, Option<&Value>) = match sg {
        Some(sg) => (get(sg, "diffuseTexture")?, get(sg, "diffuseFactor")?),
        None => match get(mat, "pbrMetallicRoughness")? {
            Some(pbr) => (get(pbr, "baseColorTexture")?, get(pbr, "baseColorFactor")?),
            None => (None, None),
        },
    };
    let rgba = rgba_of(rgba_v)?;
    let mut tex_path = None;
    let mut uvt = (0.0, 0.0, 1.0, 1.0);
    let mut uv_set = 0;
    if let Some(info) = tex_info.filter(|v| !matches!(v, Value::Null)) {
        let tex = at(list(js, "textures")?, jint(req(info, "index")?)?)?;
        if let Some(src) = get(tex, "source")?.filter(|v| !matches!(v, Value::Null)) {
            let src = jint(src)?;
            if src < image_paths.len() as i64 {
                tex_path = at_opt(image_paths, src)?;
            }
        }
        if let Some(v) = get(info, "texCoord")? {
            uv_set = jint(v)?;
        }
        let tt = match get(info, "extensions")? {
            Some(e) => get(e, "KHR_texture_transform")?,
            None => None,
        };
        if let Some(tt) = tt.filter(|t| pyval::truthy(t)) {
            let pair = |key: &str, default: [f64; 2]| -> R<[f64; 2]> {
                match get(tt, key)? {
                    None => Ok(default),
                    Some(Value::Array(a)) if a.len() == 2 => Ok([jnum(&a[0])?, jnum(&a[1])?]),
                    Some(_) => Err("ValueError: cannot unpack".to_string()),
                }
            };
            let [ox, oy] = pair("offset", [0.0, 0.0])?;
            let [sx, sy] = pair("scale", [1.0, 1.0])?;
            uvt = (ox, oy, sx, sy);
        }
    }
    let mut opacity = None;
    if matches!(get(mat, "alphaMode")?, Some(Value::Str(m)) if m == "BLEND") {
        let a = match &rgba {
            Some(c) if !c.is_empty() && c.len() > 3 => c[3],
            _ => 1.0,
        };
        if a < 0.999 {
            opacity = Some(a);
        } else if tex_path.is_none() {
            opacity = Some(0.999);
        }
    }
    Ok(Recipe { tex_path, uvt, rgba, uv_set, opacity })
}

fn at_opt(items: &[Option<String>], i: i64) -> R<Option<String>> {
    let n = items.len() as i64;
    let j = if i < 0 { i + n } else { i };
    if j < 0 || j >= n {
        return Err("IndexError".to_string());
    }
    Ok(items[j as usize].clone())
}

// ---- triangles ---------------------------------------------------------------------

type Tri3 = [[f64; 3]; 3];

/// One primitive's triangles in world Z-up metres, with its UVs and recipe.
pub struct Batch {
    tris: Vec<Tri3>,
    uvs: Option<Vec<[[f64; 2]; 3]>>,
    tex_path: Option<String>,
    rgba: Option<Vec<f64>>,
    opacity: Option<f64>,
}

/// `pts @ rot.T + trans` and `@ zup.T` for one row, as NumPy computes a
/// float64 matmul: the first product, then each next one fused into the
/// running sum; the translation is added afterwards, on its own.
fn matmul_row(p: [f64; 3], rows: [[f64; 3]; 3]) -> [f64; 3] {
    let mut out = [0.0; 3];
    for j in 0..3 {
        out[j] = p[2].mul_add(rows[j][2], p[1].mul_add(rows[j][1], p[0] * rows[j][0]));
    }
    out
}

/// `read_triangles(path)`: the file as world-space triangle batches.
pub fn read_triangles(path: &Path, cache_root: &Path) -> R<(Vec<Batch>, Value)> {
    let (js, buffers) = parse_container(path)?;
    for ext in list(&js, "extensionsRequired")? {
        let Value::Str(e) = ext else { return Err("AttributeError: lower".to_string()) };
        if e.to_lowercase().contains("draco") {
            return Err("Draco-compressed GLB — re-export/download uncompressed".to_string());
        }
    }
    let image_paths = extract_images(&js, &buffers, path, cache_root)?;
    // `world_pts @ zup.T`: output column j is the dot of the point with row j.
    let zup: [[f64; 3]; 3] = [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]];
    let mut batches = Vec::new();
    let meshes = list(&js, "meshes")?;
    let nodes = list(&js, "nodes")?;
    for (node_idx, world) in node_transforms(&js)? {
        let nd = at(nodes, node_idx as i64)?;
        let Some(mi) = get(nd, "mesh")?.filter(|v| !matches!(v, Value::Null)) else { continue };
        let mi = jint(mi)?;
        if mi >= meshes.len() as i64 {
            continue;
        }
        for prim in list(at(meshes, mi)?, "primitives")? {
            let mode = match get(prim, "mode")? {
                Some(v) => jint(v)?,
                None => 4,
            };
            if mode != 4 {
                continue;
            }
            let attrs = match get(prim, "attributes")? {
                Some(a) => a,
                None => &Value::Object(Vec::new()),
            };
            let Some(pos_idx) = get(attrs, "POSITION")? else { continue };
            let pos = read_accessor(&js, &buffers, jint(pos_idx)?)?;
            if pos.n < 3 {
                return Err("ValueError: shapes are not aligned".to_string());
            }
            let idx: Vec<i64> = match get(prim, "indices")? {
                Some(i) => read_accessor(&js, &buffers, jint(i)?)?.ints(),
                None => (0..pos.count as i64).collect(),
            };
            let whole = idx.len() / 3 * 3;
            let idx = &idx[..whole];
            let material = get(prim, "material")?.filter(|v| !matches!(v, Value::Null));
            let recipe = material_recipe(&js, material.map(jint).transpose()?, &image_paths)?;
            let rot_rows = [
                [world[0][0], world[0][1], world[0][2]],
                [world[1][0], world[1][1], world[1][2]],
                [world[2][0], world[2][1], world[2][2]],
            ];
            let trans = [world[0][3], world[1][3], world[2][3]];
            let world_pts: Vec<[f64; 3]> = (0..pos.count)
                .map(|r| {
                    let m = matmul_row([pos.at(r, 0), pos.at(r, 1), pos.at(r, 2)], rot_rows);
                    matmul_row([m[0] + trans[0], m[1] + trans[1], m[2] + trans[2]], zup)
                })
                .collect();
            let pick = |i: i64| -> R<usize> {
                let n = world_pts.len() as i64;
                let j = if i < 0 { i + n } else { i };
                if j < 0 || j >= n {
                    return Err(format!("IndexError: index {i} is out of bounds for axis 0 with size {n}"));
                }
                Ok(j as usize)
            };
            let mut tris = Vec::with_capacity(whole / 3);
            let mut picked = Vec::with_capacity(whole);
            for i in idx {
                picked.push(pick(*i)?);
            }
            for t in picked.chunks(3) {
                tris.push([world_pts[t[0]], world_pts[t[1]], world_pts[t[2]]]);
            }
            let mut tex_path = recipe.tex_path.clone();
            let mut uvs = None;
            let uv_key = format!("TEXCOORD_{}", recipe.uv_set);
            if tex_path.is_some() {
                if let Some(uv_idx) = get(attrs, &uv_key)? {
                    let uv = read_accessor(&js, &buffers, jint(uv_idx)?)?;
                    if uv.n < 2 {
                        return Err("IndexError: too many indices for array".to_string());
                    }
                    let (ox, oy, sx, sy) = recipe.uvt;
                    let mapped: Vec<[f64; 2]> = (0..uv.count).map(|r| [uv.at(r, 0) * sx + ox, uv.at(r, 1) * sy + oy]).collect();
                    let mut per = Vec::with_capacity(tris.len());
                    for t in picked.chunks(3) {
                        let mut row = [[0.0; 2]; 3];
                        for k in 0..3 {
                            let j = t[k];
                            // The UV array is indexed with the same indices.
                            if j >= mapped.len() {
                                return Err("IndexError: index is out of bounds for axis 0".to_string());
                            }
                            row[k] = mapped[j];
                        }
                        per.push(row);
                    }
                    uvs = Some(per);
                } else {
                    tex_path = None; // a texture without UVs
                }
            }
            batches.push(Batch { tris, uvs, tex_path, rgba: recipe.rgba, opacity: recipe.opacity });
        }
    }
    Ok((batches, js))
}

// ---- the texture maps ------------------------------------------------------------------

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}

/// `np.einsum("ij,ij->i", a, b)` for three terms: NumPy's SIMD loop adds the
/// first and third product, then the second.
fn einsum3(a: [f64; 3], b: [f64; 3]) -> f64 {
    (a[0] * b[0] + a[2] * b[2]) + a[1] * b[1]
}

/// OpenBLAS's LU of one 3 x 3 matrix (rows `a`): left-looking `getf2` with
/// partial pivoting (the first of equal pivots), the column below the pivot
/// scaled by the pivot's reciprocal. Plain multiplies and subtractions, no
/// fused multiply-add: found by trying the variants against NumPy
/// (tests/data/gltf/solve.txt).
fn lu3(a: &[[f64; 3]; 3]) -> ([[f64; 3]; 3], [usize; 3]) {
    let mut a = *a;
    let mut piv = [0usize; 3];
    for j in 0..3 {
        for i in 0..j {
            let ip = piv[i];
            if ip != i {
                let t = a[i][j];
                a[i][j] = a[ip][j];
                a[ip][j] = t;
            }
        }
        for i in 1..j {
            let mut s = 0.0;
            for k in 0..i {
                s += a[i][k] * a[k][j];
            }
            a[i][j] -= s;
        }
        if j > 0 {
            for i in j..3 {
                let mut s = 0.0;
                for k in 0..j {
                    s += a[i][k] * a[k][j];
                }
                a[i][j] -= s;
            }
        }
        let mut jp = j;
        for i in j + 1..3 {
            if a[i][j].abs() > a[jp][j].abs() {
                jp = i;
            }
        }
        piv[j] = jp;
        if a[jp][j] != 0.0 {
            if jp != j {
                for c in 0..=j {
                    let t = a[j][c];
                    a[j][c] = a[jp][c];
                    a[jp][c] = t;
                }
            }
            let r = 1.0 / a[j][j];
            for i in j + 1..3 {
                a[i][j] *= r;
            }
        }
    }
    (a, piv)
}

/// `np.linalg.solve` of one 3 x 3 system: [`lu3`], then `L` forward by
/// columns and `U` backward dividing by the diagonal.
pub fn solve3(a: &[[f64; 3]; 3], b: [f64; 3]) -> [f64; 3] {
    let (lu, piv) = lu3(a);
    let mut x = b;
    for i in 0..3 {
        x.swap(i, piv[i]);
    }
    for j in 0..3 {
        for i in j + 1..3 {
            x[i] -= x[j] * lu[i][j];
        }
    }
    for j in (0..3).rev() {
        x[j] /= lu[j][j];
        for i in 0..j {
            x[i] -= x[j] * lu[i][j];
        }
    }
    x
}

/// `np.linalg.det` of a 3 x 3: sign times `exp(sum(log|u_ii|))` of the LU.
pub fn det3(a: &[[f64; 3]; 3]) -> f64 {
    let (lu, piv) = lu3(a);
    let mut sign = 1.0;
    let mut logdet = 0.0;
    for i in 0..3 {
        if piv[i] != i {
            sign = -sign;
        }
        let d = lu[i][i];
        if d < 0.0 {
            sign = -sign;
        }
        logdet += d.abs().ln();
    }
    if lu.iter().enumerate().any(|(i, r)| r[i] == 0.0) {
        return 0.0;
    }
    sign * logdet.exp()
}

/// `_triangle_uvws(tris, uvs)`: per triangle the world to UV affine
/// `[gu, cu, gv, cv]` and whether it could be solved.
fn triangle_uvws(tris: &[Tri3], uvs: &[[[f64; 2]; 3]]) -> Vec<([f64; 8], bool)> {
    tris.iter()
        .zip(uvs)
        .map(|(t, uv)| {
            let (p0, p1, p2) = (t[0], t[1], t[2]);
            let e1 = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]];
            let e2 = [p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]];
            let n = cross(e1, e2);
            let a = [e1, e2, n];
            let det = det3(&a);
            let ok = det.abs() > 1e-18;
            let (mut gu, mut gv) = ([0.0; 3], [0.0; 3]);
            if ok {
                gu = solve3(&a, [uv[1][0] - uv[0][0], uv[2][0] - uv[0][0], 0.0]);
                gv = solve3(&a, [uv[1][1] - uv[0][1], uv[2][1] - uv[0][1], 0.0]);
            }
            let cu = uv[0][0] - einsum3(gu, p0);
            let cv = uv[0][1] - einsum3(gv, p0);
            ([gu[0], gu[1], gu[2], cu, gv[0], gv[1], gv[2], cv], ok)
        })
        .collect()
}

/// `np.linalg.norm(v, axis=1)` of one row: `sqrt((a² + b²) + c²)`.
fn norm3(v: [f64; 3]) -> f64 {
    ((v[0] * v[0] + v[1] * v[1]) + v[2] * v[2]).sqrt()
}

// ---- the mesh ----------------------------------------------------------------------------

type Attrs = Vec<(String, Value)>;

fn fl(x: f64) -> Value {
    Value::Float(x)
}

/// `build_mesh(batches)` without the Rust engine.
pub fn build_mesh(batches: &[Batch], h: Hypot) -> R<Mesh> {
    if batches.is_empty() {
        return Err("No triangle geometry found in the glTF file".to_string());
    }
    let mut flat: Vec<V3> = Vec::new();
    let mut attrs_list: Vec<Attrs> = Vec::new();
    for b in batches {
        flat.extend(b.tris.iter().flat_map(|t| t.iter().map(|p| V3(p[0] as f32, p[1] as f32, p[2] as f32))));
        match &b.tex_path {
            None => {
                let color: Option<Vec<f64>> = b.rgba.as_ref().filter(|c| !c.is_empty()).map(|c| c.iter().take(3).copied().collect());
                let mut base: Attrs = Vec::new();
                if let Some(c) = &color {
                    if *c != [1.0, 1.0, 1.0] {
                        base.push(("color".into(), Value::Array(c.iter().map(|&x| fl(x)).collect())));
                        base.push(("back".into(), Value::Bool(true)));
                    }
                }
                if let Some(o) = b.opacity {
                    base.push(("opacity".into(), fl(o)));
                }
                attrs_list.extend(std::iter::repeat_n(base, b.tris.len()));
            }
            Some(path) => {
                let uvs = b.uvs.as_ref().expect("a textured batch has UVs");
                let solved = triangle_uvws(&b.tris, uvs);
                for (uvw, ok) in solved {
                    if !ok {
                        attrs_list.push(Vec::new());
                        continue;
                    }
                    let glu = norm3([uvw[0], uvw[1], uvw[2]]);
                    let glv = norm3([uvw[4], uvw[5], uvw[6]]);
                    let sw = if glu > 1e-9 { 1.0 / glu.max(1e-9) } else { 1.0 };
                    let sh = if glv > 1e-9 { 1.0 / glv.max(1e-9) } else { 1.0 };
                    let tex = vec![
                        ("path".to_string(), Value::Str(path.clone())),
                        ("uvw".to_string(), Value::Array(uvw.iter().map(|&x| fl(x)).collect())),
                        ("sw".to_string(), fl(sw)),
                        ("sh".to_string(), fl(sh)),
                    ];
                    let mut a: Attrs = vec![("texture".into(), Value::Object(tex)), ("back".into(), Value::Bool(true))];
                    if let Some(o) = b.opacity {
                        a.push(("opacity".into(), fl(o)));
                    }
                    attrs_list.push(a);
                }
            }
        }
    }
    let mut mesh = Mesh::new();
    let (vobjs, inverse) = mesh.bulk_weld(&flat);
    let n_faces = flat.len() / 3;
    mesh.add_faces_welded(&vobjs, &inverse, &vec![3; n_faces], &vec![1; n_faces], Some(attrs_list.as_slice()), None);
    if mesh.edges.len() <= SOFTEN_EDGE_CAP {
        soften_smooth_edges(&mut mesh, h);
    }
    hide_cutout_card_edges(&mut mesh, h);
    Ok(mesh)
}

/// `_smooth_sig(attrs)`: colour and texture image only; `None` for no attrs.
fn smooth_sig(attrs: &[(String, Value)]) -> Option<(Option<Value>, Option<Value>)> {
    if attrs.is_empty() {
        return None;
    }
    let c = attr(attrs, "color").filter(|c| !matches!(c, Value::Null)).cloned();
    let t = attr(attrs, "texture").filter(|t| pyval::truthy(t)).and_then(|t| t.get("path").cloned());
    Some((c, t))
}

/// `formats.fuse.soften_smooth_edges(mesh)`: edges between two same-material
/// faces meeting at a shallow dihedral go soft.
pub fn soften_smooth_edges(mesh: &mut Mesh, h: Hypot) {
    let mut normals: std::collections::HashMap<u32, V3> = Default::default();
    for e in mesh.edges.clone() {
        let ed = mesh.edge(e);
        if ed.faces.len() != 2 {
            continue;
        }
        let (f0, f1) = (ed.faces[0], ed.faces[1]);
        if f0 == f1 {
            continue;
        }
        if smooth_sig(&mesh.face(f0).attrs) != smooth_sig(&mesh.face(f1).attrs) {
            continue;
        }
        let mut normal_of = |f: ingetrazo_model::mesh::FaceId| -> V3 {
            *normals.entry(f.0).or_insert_with(|| {
                let outer: Vec<V3> = mesh.face(f).outer.iter().map(|&v| mesh.position(v)).collect();
                face_normal(&outer, h)
            })
        };
        let (n0, n1) = (normal_of(f0), normal_of(f1));
        if (n0.dot(n1) as f64).abs() > 0.85 {
            mesh.edge_mut(e).soft = true;
        }
    }
}

/// `_hide_cutout_card_edges(mesh)`: the visible outline of a leaf card is its
/// texture, not its quad. A texture is foliage-like when it carries alpha or
/// its faces are confetti (mean area under 0.02 m2); edges whose faces are
/// all foliage-like go hidden.
pub fn hide_cutout_card_edges(mesh: &mut Mesh, h: Hypot) {
    let mut stats: Vec<(String, usize, f64)> = Vec::new();
    for &f in &mesh.faces {
        let face = mesh.face(f);
        let path = match attr(&face.attrs, "texture") {
            Some(t) => match t.get("path") {
                Some(Value::Str(p)) if !p.is_empty() => Some(p.clone()),
                _ => None,
            },
            None => None,
        };
        if let Some(p) = path {
            let outer: Vec<V3> = face.outer.iter().map(|&v| mesh.position(v)).collect();
            let area = face_area(&outer, h);
            match stats.iter_mut().find(|(k, _, _)| *k == p) {
                Some(s) => {
                    s.1 += 1;
                    s.2 += area;
                }
                None => stats.push((p, 1, area)),
            }
        }
    }
    let foliage: Vec<(String, bool)> = stats
        .iter()
        .map(|(p, n, area)| {
            let alpha = std::fs::read(p).map(|b| has_alpha_channel(&b)).unwrap_or(false);
            (p.clone(), alpha || (*n > 200 && area / (*n as f64) < 0.02))
        })
        .collect();
    if !foliage.iter().any(|(_, f)| *f) {
        return;
    }
    let is_foliage = |mesh: &Mesh, f: ingetrazo_model::mesh::FaceId| -> bool {
        match attr(&mesh.face(f).attrs, "texture").and_then(|t| t.get("path")) {
            Some(Value::Str(p)) => foliage.iter().any(|(k, v)| k == p && *v),
            _ => false,
        }
    };
    for e in mesh.edges.clone() {
        let faces = mesh.edge(e).faces.clone();
        if !faces.is_empty() && faces.iter().all(|&f| is_foliage(mesh, f)) {
            mesh.edge_mut(e).hidden = true;
        }
    }
}

// ---- loading ------------------------------------------------------------------------------

/// `_guess_scale(batches)`: the first of metres / inches / centimetres /
/// millimetres that puts the largest dimension in 0.2 to 40.
pub fn guess_scale(batches: &[Batch]) -> R<f64> {
    let (mut lo, mut hi) = ([f64::INFINITY; 3], [f64::NEG_INFINITY; 3]);
    // np.minimum / np.maximum propagate NaN.
    let min = |a: f64, b: f64| if a.is_nan() || b.is_nan() { f64::NAN } else if b < a { b } else { a };
    let max = |a: f64, b: f64| if a.is_nan() || b.is_nan() { f64::NAN } else if b > a { b } else { a };
    for b in batches {
        if b.tris.is_empty() {
            return Err("ValueError: zero-size array to reduction operation minimum which has no identity".to_string());
        }
        for t in &b.tris {
            for p in t {
                for k in 0..3 {
                    lo[k] = min(lo[k], p[k]);
                    hi[k] = max(hi[k], p[k]);
                }
            }
        }
    }
    let mut dim = f64::NEG_INFINITY;
    for k in 0..3 {
        let d = hi[k] - lo[k];
        dim = if d.is_nan() || dim.is_nan() { f64::NAN } else if d > dim { d } else { dim };
    }
    if dim <= 0.0 {
        return Ok(1.0);
    }
    for s in [1.0, 0.0254, 0.01, 0.001] {
        if 0.2 <= dim * s && dim * s <= 40.0 {
            return Ok(s);
        }
    }
    Ok(1.0)
}

/// `load_glb(scene, path)`: the file added to `scene` as a reference group
/// named after the file (or the embedded Sketchfab title). Extracted textures
/// go under `cache_root` (see [`texture_cache_root`]).
pub fn load_glb(scene: &mut Scene, path: &Path, cache_root: &Path, h: Hypot) -> R<()> {
    let (mut batches, js) = read_triangles(path, cache_root)?;
    // Our own files are in metres, as glTF says: guessing would shrink a
    // model wider than 40 m to inches.
    let ours = matches!(
        get(&js, "asset")?.map(|a| get(a, "generator")).transpose()?.flatten(),
        Some(Value::Str(g)) if g == "IngeTrazo"
    );
    let s = if ours { 1.0 } else { guess_scale(&batches)? };
    if s != 1.0 {
        for b in &mut batches {
            for t in &mut b.tris {
                for p in t.iter_mut() {
                    for c in p.iter_mut() {
                        *c *= s;
                    }
                }
            }
        }
    }
    let mesh = build_mesh(&batches, h)?;
    let title = match get(&js, "asset")? {
        Some(asset) => match get(asset, "extras")? {
            Some(e) if pyval::truthy(e) => get(e, "title")?.cloned(),
            _ => None,
        },
        None => None,
    };
    let name_src = match title {
        Some(Value::Str(t)) if !t.is_empty() => t,
        Some(v) if pyval::truthy(&v) => return Err("TypeError: title".to_string()),
        _ => stem_of(path),
    };
    let name: String = name_src.chars().take(60).collect();
    scene.meshes.push(mesh);
    let id = ingetrazo_model::scene::MeshId(scene.meshes.len() as u32 - 1);
    scene.groups.push(Group::new(id, name));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sha1_matches_the_known_digests() {
        assert_eq!(sha1_hex(b""), "da39a3ee5e6b4b0d3255bfef95601890afd80709");
        assert_eq!(sha1_hex(b"abc"), "a9993e364706816aba3e25717850c26c9cd0d89d");
        assert_eq!(sha1_hex(&[b'a'; 1000]), "291e9a6c66994949b57ba5e650361e98fc36b1ba");
    }

    #[test]
    fn base64_like_python() {
        assert_eq!(base64_decode("QUJD").unwrap(), b"ABC");
        assert_eq!(base64_decode("QUI=").unwrap(), b"AB");
        assert_eq!(base64_decode("QQ==").unwrap(), b"A");
        assert_eq!(base64_decode("QU JD\n").unwrap(), b"ABC");
        assert!(base64_decode("QQ").is_err());
    }
}
