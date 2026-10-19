// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! `georef/geotiff.py`: GeoTIFF orthomosaics as georeferenced images.
//!
//! Reads classic and BigTIFF headers (first IFD), the GeoTIFF placement
//! (affine transform + CRS from the GeoKey directory) and decodes 8/16-bit
//! grey / RGB / RGBA rasters, tiled or stripped, contiguous or planar,
//! uncompressed / Deflate / PackBits / LZW, reduced by an integer factor
//! while streaming. JPEG-in-TIFF headers are read but their pixels are not
//! decoded (Python hands those to Qt).

use std::collections::HashMap;
use std::fmt;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use flate2::read::ZlibDecoder;
use ingetrazo_formats::pyjson::float_repr;
use ingetrazo_geom::tiff::{lzw_decode, unpackbits};

use crate::datum::utm_inverse;

// TIFF tags
const IMAGE_WIDTH: u16 = 256;
const IMAGE_LENGTH: u16 = 257;
const BITS_PER_SAMPLE: u16 = 258;
const COMPRESSION: u16 = 259;
const PHOTOMETRIC: u16 = 262;
const STRIP_OFFSETS: u16 = 273;
const SAMPLES_PER_PIXEL: u16 = 277;
const ROWS_PER_STRIP: u16 = 278;
const STRIP_BYTE_COUNTS: u16 = 279;
const PLANAR_CONFIG: u16 = 284;
const PREDICTOR: u16 = 317;
const TILE_WIDTH: u16 = 322;
const TILE_LENGTH: u16 = 323;
const TILE_OFFSETS: u16 = 324;
const TILE_BYTE_COUNTS: u16 = 325;
const EXTRA_SAMPLES: u16 = 338;
const SAMPLE_FORMAT: u16 = 339;
const JPEG_TABLES: u16 = 347;
const MODEL_PIXEL_SCALE: u16 = 33550;
const MODEL_TIEPOINT: u16 = 33922;
const MODEL_TRANSFORMATION: u16 = 34264;
const GEO_KEY_DIRECTORY: u16 = 34735;
const GEO_ASCII_PARAMS: u16 = 34737;
const GDAL_NODATA: u16 = 42113;

// GeoKeys
const GT_MODEL_TYPE: i64 = 1024;
const GT_RASTER_TYPE: i64 = 1025;
const GT_CITATION: i64 = 1026;
const GEOGRAPHIC_TYPE: i64 = 2048;
const PROJECTED_CS_TYPE: i64 = 3072;
const PCS_CITATION: i64 = 3073;

pub const COMPRESSION_NONE: i64 = 1;
pub const COMPRESSION_LZW: i64 = 5;
pub const COMPRESSION_JPEG: i64 = 7;
pub const COMPRESSION_DEFLATE: i64 = 8;
pub const COMPRESSION_PACKBITS: i64 = 32773;
pub const COMPRESSION_DEFLATE_OLD: i64 = 32946;

/// `GeoTiffError`: the file is not a TIFF this reader can place or decode;
/// the message says which part.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GeoTiffError(pub String);

impl fmt::Display for GeoTiffError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for GeoTiffError {}

fn err<T>(msg: impl Into<String>) -> Result<T, GeoTiffError> {
    Err(GeoTiffError(msg.into()))
}

/// Header of a GeoTIFF: the raster layout plus its map placement.
#[derive(Clone, Debug)]
pub struct GeoTiffInfo {
    pub path: PathBuf,
    pub width: i64,
    pub height: i64,
    pub samples: i64,
    pub bits: Vec<i64>,
    pub compression: i64,
    pub photometric: i64,
    pub planar: i64,
    pub tiled: bool,
    pub tile_w: i64,
    pub tile_h: i64,
    pub offsets: Vec<i64>,
    pub bytecounts: Vec<i64>,
    pub predictor: i64,
    pub jpeg_tables: Vec<u8>,
    pub extra_samples: Vec<i64>,
    pub sample_format: Vec<i64>,
    pub big: bool,
    /// `true` for `II` (little endian), `false` for `MM`.
    pub little_endian: bool,
    /// Pixel to map affine `(a, b, c, d, e, f)` of the pixel's top-left corner.
    pub transform: Option<[f64; 6]>,
    pub epsg: Option<i64>,
    pub zone: Option<i64>,
    pub northern: Option<bool>,
    pub geographic: bool,
    pub citation: String,
    pub nodata: Option<String>,
}

impl GeoTiffInfo {
    fn file_name(&self) -> String {
        self.path
            .file_name()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default()
    }

    pub fn georeferenced(&self) -> bool {
        self.transform.is_some() && (self.geographic || self.zone.is_some())
    }

    pub fn pixel_to_map(&self, col: f64, row: f64) -> (f64, f64) {
        let [a, b, c, d, e, f] = self.transform.expect("pixel_to_map without a transform");
        (a * col + b * row + c, d * col + e * row + f)
    }

    /// `(lat, lon)` of the corners: bottom-left, bottom-right, top-right,
    /// top-left.
    pub fn corners_geodetic(&self) -> Vec<(f64, f64)> {
        let (w, h) = (self.width as f64, self.height as f64);
        let px = [(0.0, h), (w, h), (w, 0.0), (0.0, 0.0)];
        let mut out = Vec::new();
        for (col, row) in px {
            let (x, y) = self.pixel_to_map(col, row);
            if self.geographic {
                out.push((y, x));
            } else {
                out.push(utm_inverse(
                    x,
                    y,
                    self.zone.expect("corners_geodetic without a zone"),
                    self.northern.unwrap_or(false),
                ));
            }
        }
        out
    }

    /// Approximate metres covered `(width, height)`.
    pub fn ground_size(&self) -> (f64, f64) {
        let [a, b, _c, d, e, f] = self.transform.expect("ground_size without a transform");
        let mut w = py_hypot2(a, d) * self.width as f64;
        let mut h = py_hypot2(b, e) * self.height as f64;
        if self.geographic {
            let lat = (f + e * self.height as f64 / 2.0).to_radians();
            w *= 111_320.0 * lat.cos();
            h *= 110_540.0;
        }
        (w, h)
    }
}

// ---- math.hypot ------------------------------------------------------------

/// `2.0 ** e` for any exponent a double can hold (subnormals included).
fn ldexp1(e: i32) -> f64 {
    if e >= -1022 {
        f64::from_bits(((e + 1023) as u64) << 52)
    } else {
        f64::from_bits(1u64 << (e + 1074))
    }
}

/// C `frexp` exponent: `x = m * 2**e` with `0.5 <= |m| < 1` (x finite, non-zero).
fn frexp_exp(x: f64) -> i32 {
    let bits = x.abs().to_bits();
    let be = ((bits >> 52) & 0x7ff) as i32;
    if be == 0 {
        // subnormal: normalise through 2**54
        frexp_exp(x * f64::from_bits(((54 + 1023) as u64) << 52)) - 54
    } else {
        be - 1022
    }
}

/// CPython's `dl_mul`: the error-free product `(hi, lo)`.
fn dl_mul(x: f64, y: f64) -> (f64, f64) {
    let z = x * y;
    (z, x.mul_add(y, -z))
}

/// CPython's `dl_fast_sum`.
fn dl_fast_sum(a: f64, b: f64) -> (f64, f64) {
    let x = a + b;
    (x, (a - x) + b)
}

/// CPython 3.12 `vector_norm` (the core of `math.hypot`).
fn vector_norm(vec: &mut [f64], max: f64) -> f64 {
    let n = vec.len();
    if max.is_infinite() {
        return max;
    }
    if max == 0.0 || n <= 1 {
        return max;
    }
    let max_e = frexp_exp(max);
    if max_e < -1023 {
        let dbl_min = f64::MIN_POSITIVE;
        for v in vec.iter_mut() {
            *v /= dbl_min;
        }
        let r = vector_norm(vec, max / dbl_min);
        return dbl_min * r;
    }
    let scale = ldexp1(-max_e);
    let mut csum = 1.0f64;
    let mut frac1 = 0.0f64;
    let mut frac2 = 0.0f64;
    for v in vec.iter() {
        let x = *v * scale;
        let (phi, plo) = dl_mul(x, x);
        let (shi, slo) = dl_fast_sum(csum, phi);
        csum = shi;
        frac1 += plo;
        frac2 += slo;
    }
    let mut h = (csum - 1.0 + (frac1 + frac2)).sqrt();
    let (phi, plo) = dl_mul(-h, h);
    let (shi, slo) = dl_fast_sum(csum, phi);
    csum = shi;
    frac1 += plo;
    frac2 += slo;
    let x = csum - 1.0 + (frac1 + frac2);
    h += x / (2.0 * h);
    h / scale
}

/// Python 3.12's `math.hypot(x, y)`. Not libm's `hypot`: CPython has its own
/// scaled algorithm, so `f64::hypot` can differ in the last place.
pub fn py_hypot2(x: f64, y: f64) -> f64 {
    let (ax, ay) = (x.abs(), y.abs());
    if ax.is_nan() || ay.is_nan() {
        // CPython: an infinity wins over NaN, otherwise NaN.
        return if ax.is_infinite() || ay.is_infinite() { f64::INFINITY } else { f64::NAN };
    }
    let max = if ay > ax { ay } else { ax };
    let mut v = [ax, ay];
    vector_norm(&mut v, max)
}

// ---- Header ----------------------------------------------------------------

/// A decoded IFD value: ASCII as text (latin-1), UNDEFINED as bytes,
/// integer types as ints, FLOAT/DOUBLE/RATIONAL as floats.
#[derive(Clone, Debug)]
enum Tag {
    Ascii(String),
    Bytes(Vec<u8>),
    Ints(Vec<i128>),
    Floats(Vec<f64>),
}

impl Tag {
    /// Python `int(v)` of every element (floats truncate).
    fn ints(&self) -> Option<Vec<i64>> {
        match self {
            Tag::Ints(v) => Some(v.iter().map(|&x| x as i64).collect()),
            Tag::Floats(v) => Some(v.iter().map(|&x| x as i64).collect()),
            Tag::Bytes(v) => Some(v.iter().map(|&x| x as i64).collect()),
            Tag::Ascii(_) => None,
        }
    }

    /// Python `float(v)` of every element.
    fn floats(&self) -> Option<Vec<f64>> {
        match self {
            Tag::Ints(v) => Some(v.iter().map(|&x| x as f64).collect()),
            Tag::Floats(v) => Some(v.clone()),
            Tag::Bytes(v) => Some(v.iter().map(|&x| x as f64).collect()),
            Tag::Ascii(_) => None,
        }
    }
}

type Tags = HashMap<u16, Tag>;

struct Reader {
    f: File,
    le: bool,
}

impl Reader {
    fn u16(&self, b: &[u8]) -> u16 {
        let a = [b[0], b[1]];
        if self.le {
            u16::from_le_bytes(a)
        } else {
            u16::from_be_bytes(a)
        }
    }
    fn u32(&self, b: &[u8]) -> u32 {
        let a = [b[0], b[1], b[2], b[3]];
        if self.le {
            u32::from_le_bytes(a)
        } else {
            u32::from_be_bytes(a)
        }
    }
    fn u64(&self, b: &[u8]) -> u64 {
        let mut a = [0u8; 8];
        a.copy_from_slice(&b[..8]);
        if self.le {
            u64::from_le_bytes(a)
        } else {
            u64::from_be_bytes(a)
        }
    }

    /// Python's `f.read(n)` after `seek`: whatever is there, up to `n`.
    fn read_at(&mut self, offset: u64, n: u64) -> Vec<u8> {
        if self.f.seek(SeekFrom::Start(offset)).is_err() {
            return Vec::new();
        }
        let mut out = Vec::new();
        let _ = (&mut self.f).take(n).read_to_end(&mut out);
        out
    }
}

fn type_size(typ: u16) -> Option<u64> {
    Some(match typ {
        1 | 2 | 6 | 7 => 1,
        3 | 8 => 2,
        4 | 9 | 11 => 4,
        5 | 10 | 12 | 16 | 17 | 18 => 8,
        _ => return None,
    })
}

/// `read_info`: parse the first IFD of a TIFF/BigTIFF and its GeoTIFF
/// placement. Anything that is not a readable TIFF is a [`GeoTiffError`].
pub fn read_info(path: impl AsRef<Path>) -> Result<GeoTiffInfo, GeoTiffError> {
    let path = path.as_ref().to_path_buf();
    let name = path
        .file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default();
    let mut file = File::open(&path).map_err(|e| GeoTiffError(format!("{name}: {e}")))?;
    let mut head = Vec::new();
    (&mut file).take(16).read_to_end(&mut head).map_err(|e| GeoTiffError(e.to_string()))?;
    if head.len() < 8 || !(head.starts_with(b"II") || head.starts_with(b"MM")) {
        return err(format!("{name}: not a TIFF file"));
    }
    let le = head.starts_with(b"II");
    let mut rd = Reader { f: file, le };
    let magic = rd.u16(&head[2..4]);
    let (big, ifd) = if magic == 42 {
        (false, rd.u32(&head[4..8]) as u64)
    } else if magic == 43 {
        if head.len() < 16 {
            return err(format!("{name}: truncated BigTIFF header"));
        }
        (true, rd.u64(&head[8..16]))
    } else {
        return err(format!("{name}: not a TIFF file"));
    };
    let tags = read_ifd(&mut rd, ifd, big, &name)?;

    let width = tags.get(&IMAGE_WIDTH).and_then(|t| t.ints());
    let height = tags.get(&IMAGE_LENGTH).and_then(|t| t.ints());
    let (width, height) = match (width, height) {
        (Some(w), Some(h)) if !w.is_empty() && !h.is_empty() => (w[0], h[0]),
        _ => return err(format!("{name}: the TIFF has no image size")),
    };
    let ints_or = |tag: u16, default: &[i64]| -> Vec<i64> {
        tags.get(&tag).and_then(|t| t.ints()).unwrap_or_else(|| default.to_vec())
    };
    let first = |tag: u16, default: i64| -> i64 {
        ints_or(tag, &[default]).first().copied().unwrap_or(default)
    };
    let samples = first(SAMPLES_PER_PIXEL, 1);
    let mut bits = ints_or(BITS_PER_SAMPLE, &[1]);
    if (bits.len() as i64) < samples {
        let last = match bits.last() {
            Some(&b) => b,
            None => return err(format!("{name}: empty BitsPerSample")),
        };
        while (bits.len() as i64) < samples {
            bits.push(last);
        }
    }
    let compression = first(COMPRESSION, 1);
    let photometric = first(PHOTOMETRIC, if samples >= 3 { 2 } else { 1 });
    let planar = first(PLANAR_CONFIG, 1);
    let tiled = tags.contains_key(&TILE_OFFSETS);
    let (tile_w, tile_h, offsets, counts);
    if tiled {
        tile_w = match tags.get(&TILE_WIDTH).and_then(|t| t.ints()).and_then(|v| v.first().copied())
        {
            Some(v) => v,
            None => return err(format!("{name}: the TIFF has no tile width")),
        };
        tile_h =
            match tags.get(&TILE_LENGTH).and_then(|t| t.ints()).and_then(|v| v.first().copied()) {
                Some(v) => v,
                None => return err(format!("{name}: the TIFF has no tile length")),
            };
        offsets = ints_or(TILE_OFFSETS, &[]);
        counts = ints_or(TILE_BYTE_COUNTS, &[]);
    } else {
        if !tags.contains_key(&STRIP_OFFSETS) {
            return err(format!("{name}: the TIFF has no image data"));
        }
        tile_w = width;
        let rps = first(ROWS_PER_STRIP, height);
        tile_h = if rps != 0 { rps.min(height) } else { height };
        offsets = ints_or(STRIP_OFFSETS, &[]);
        counts = ints_or(STRIP_BYTE_COUNTS, &[]);
    }
    if counts.is_empty() {
        return err(format!("{name}: the TIFF has no byte counts"));
    }
    let jpeg_tables = if compression == COMPRESSION_JPEG {
        match tags.get(&JPEG_TABLES) {
            Some(Tag::Bytes(b)) => b.clone(),
            Some(Tag::Ints(v)) => v.iter().map(|&x| x as u8).collect(),
            _ => Vec::new(),
        }
    } else {
        Vec::new()
    };
    let nodata = tags.get(&GDAL_NODATA).map(tag_str);
    let mut info = GeoTiffInfo {
        path,
        width,
        height,
        samples,
        bits,
        compression,
        photometric,
        planar,
        tiled,
        tile_w,
        tile_h,
        offsets,
        bytecounts: counts,
        predictor: first(PREDICTOR, 1),
        jpeg_tables,
        extra_samples: ints_or(EXTRA_SAMPLES, &[]),
        sample_format: ints_or(SAMPLE_FORMAT, &[]),
        big,
        little_endian: le,
        transform: None,
        epsg: None,
        zone: None,
        northern: None,
        geographic: false,
        citation: String::new(),
        nodata,
    };
    read_placement(&mut info, &tags)?;
    Ok(info)
}

/// Python's `str(value)` for a tag value stored under GDAL_NODATA.
fn tag_str(t: &Tag) -> String {
    fn tuple<T>(v: &[T], f: impl Fn(&T) -> String) -> String {
        let items: Vec<String> = v.iter().map(f).collect();
        match items.len() {
            0 => "()".to_string(),
            1 => format!("({},)", items[0]),
            _ => format!("({})", items.join(", ")),
        }
    }
    match t {
        Tag::Ascii(s) => s.clone(),
        Tag::Ints(v) => tuple(v, |x| x.to_string()),
        Tag::Floats(v) => tuple(v, |x| float_repr(*x)),
        Tag::Bytes(b) => {
            // bytes.__repr__: single quotes unless the data holds ' and no ".
            let quote = if b.contains(&b'\'') && !b.contains(&b'"') { '"' } else { '\'' };
            let mut s = format!("b{quote}");
            for &c in b {
                match c {
                    b'\\' => s.push_str("\\\\"),
                    b'\'' if quote == '\'' => s.push_str("\\'"),
                    b'\t' => s.push_str("\\t"),
                    b'\n' => s.push_str("\\n"),
                    b'\r' => s.push_str("\\r"),
                    0x20..=0x7e => s.push(c as char),
                    _ => s.push_str(&format!("\\x{c:02x}")),
                }
            }
            s.push(quote);
            s
        }
    }
}

/// `_read_ifd`: the tag to value map of one IFD.
fn read_ifd(rd: &mut Reader, offset: u64, big: bool, name: &str) -> Result<Tags, GeoTiffError> {
    let trunc = || GeoTiffError(format!("{name}: truncated IFD"));
    let count_len = if big { 8 } else { 2 };
    let b = rd.read_at(offset, count_len);
    if (b.len() as u64) < count_len {
        return Err(trunc());
    }
    let n = if big { rd.u64(&b) } else { rd.u16(&b) as u64 };
    let esize: u64 = if big { 20 } else { 12 };
    let raw = rd.read_at(offset + count_len, n.saturating_mul(esize));
    let inline: u64 = if big { 8 } else { 4 };
    let mut tags: Tags = HashMap::new();
    // Python decodes every entry header first (a short entry is a
    // struct.error), then reads values in order.
    let mut entries = Vec::new();
    for i in 0..n as usize {
        let e = raw.get(i * esize as usize..(i + 1) * esize as usize).ok_or_else(trunc)?;
        let (tag, typ, cnt, val) = if big {
            (rd.u16(&e[0..2]), rd.u16(&e[2..4]), rd.u64(&e[4..12]), e[12..20].to_vec())
        } else {
            (rd.u16(&e[0..2]), rd.u16(&e[2..4]), rd.u32(&e[4..8]) as u64, e[8..12].to_vec())
        };
        entries.push((tag, typ, cnt, val));
    }
    for (tag, typ, cnt, val) in entries {
        let Some(size) = type_size(typ) else { continue };
        let total = size.saturating_mul(cnt);
        let data: Vec<u8> = if total <= inline {
            val[..total as usize].to_vec()
        } else {
            let off = if big { rd.u64(&val) } else { rd.u32(&val) as u64 };
            let d = rd.read_at(off, total);
            if (d.len() as u64) < total {
                continue;
            }
            d
        };
        let le = rd.le;
        let value = match typ {
            2 => {
                let s: String = data.iter().map(|&c| c as char).collect();
                Tag::Ascii(s.trim_end_matches('\0').to_string())
            }
            7 => Tag::Bytes(data),
            5 | 10 => {
                let word = |i: usize| -> (i128, i128) {
                    let a = &data[i * 8..i * 8 + 4];
                    let b = &data[i * 8 + 4..i * 8 + 8];
                    let rd32 = |x: &[u8]| -> i128 {
                        let arr = [x[0], x[1], x[2], x[3]];
                        if typ == 5 {
                            (if le { u32::from_le_bytes(arr) } else { u32::from_be_bytes(arr) })
                                as i128
                        } else {
                            (if le { i32::from_le_bytes(arr) } else { i32::from_be_bytes(arr) })
                                as i128
                        }
                    };
                    (rd32(a), rd32(b))
                };
                Tag::Floats(
                    (0..cnt as usize)
                        .map(|i| {
                            let (num, den) = word(i);
                            if den != 0 {
                                // Python's int / int is the correctly rounded
                                // quotient; both fit f64 exactly.
                                num as f64 / den as f64
                            } else {
                                0.0
                            }
                        })
                        .collect(),
                )
            }
            11 | 12 => Tag::Floats(
                (0..cnt as usize)
                    .map(|i| {
                        if typ == 11 {
                            let a = [data[i * 4], data[i * 4 + 1], data[i * 4 + 2], data[i * 4 + 3]];
                            (if le { f32::from_le_bytes(a) } else { f32::from_be_bytes(a) }) as f64
                        } else {
                            let mut a = [0u8; 8];
                            a.copy_from_slice(&data[i * 8..i * 8 + 8]);
                            if le {
                                f64::from_le_bytes(a)
                            } else {
                                f64::from_be_bytes(a)
                            }
                        }
                    })
                    .collect(),
            ),
            _ => {
                let sz = size as usize;
                Tag::Ints(
                    (0..cnt as usize)
                        .map(|i| {
                            let c = &data[i * sz..(i + 1) * sz];
                            let mut buf = [0u8; 8];
                            if le {
                                buf[..sz].copy_from_slice(c);
                            } else {
                                buf[8 - sz..].copy_from_slice(c);
                            }
                            let u = if le { u64::from_le_bytes(buf) } else { u64::from_be_bytes(buf) };
                            match typ {
                                1 | 3 | 4 | 16 | 18 => u as i128,
                                6 => u as u8 as i8 as i128,
                                8 => u as u16 as i16 as i128,
                                9 => u as u32 as i32 as i128,
                                _ => u as i64 as i128, // 17
                            }
                        })
                        .collect(),
                )
            }
        };
        tags.insert(tag, value);
    }
    Ok(tags)
}

/// Python `str.isspace`-like `\s` for latin-1 characters.
fn is_py_space(c: char) -> bool {
    matches!(c, ' ' | '\t' | '\n' | '\r' | '\x0b' | '\x0c' | '\x1c'..='\x1f' | '\u{85}' | '\u{a0}')
}

/// `_UTM_RE.search(text)`: `UTM\s*zone\s*(\d{1,2})\s*([NS])`, ignoring case.
fn find_utm_zone(text: &str) -> Option<(i64, bool)> {
    let ch: Vec<char> = text.chars().collect();
    let lower = |c: char| c.to_ascii_lowercase();
    for start in 0..ch.len() {
        let mut i = start;
        let lit = |i: &mut usize, word: &str| -> bool {
            for w in word.chars() {
                if *i < ch.len() && lower(ch[*i]) == w {
                    *i += 1;
                } else {
                    return false;
                }
            }
            true
        };
        if !lit(&mut i, "utm") {
            continue;
        }
        while i < ch.len() && is_py_space(ch[i]) {
            i += 1;
        }
        if !lit(&mut i, "zone") {
            continue;
        }
        while i < ch.len() && is_py_space(ch[i]) {
            i += 1;
        }
        // \d{1,2}: greedy, backtracking from two digits to one.
        let d0 = i;
        let mut nd = 0;
        while nd < 2 && d0 + nd < ch.len() && ch[d0 + nd].is_ascii_digit() {
            nd += 1;
        }
        while nd >= 1 {
            let mut j = d0 + nd;
            // \s* then [NS]; \s* may backtrack but the next char must be
            // N/S which is not whitespace, so greedy is enough.
            while j < ch.len() && is_py_space(ch[j]) {
                j += 1;
            }
            if j < ch.len() && matches!(lower(ch[j]), 'n' | 's') {
                let zone: i64 = ch[d0..d0 + nd].iter().collect::<String>().parse().unwrap();
                return Some((zone, lower(ch[j]) == 'n'));
            }
            nd -= 1;
        }
    }
    None
}

/// `_read_placement`: the affine transform and the CRS from the GeoTIFF tags.
fn read_placement(info: &mut GeoTiffInfo, tags: &Tags) -> Result<(), GeoTiffError> {
    let scale = tags.get(&MODEL_PIXEL_SCALE).and_then(|t| t.floats());
    let tie = tags.get(&MODEL_TIEPOINT).and_then(|t| t.floats());
    let matrix = tags.get(&MODEL_TRANSFORMATION).and_then(|t| t.floats());
    let mut transform: Option<[f64; 6]> = None;
    match (&matrix, &scale, &tie) {
        (Some(m), _, _) if m.len() >= 16 => {
            transform = Some([m[0], m[1], m[3], m[4], m[5], m[7]]);
        }
        (_, Some(s), Some(t)) if t.len() >= 6 => {
            if s.len() < 2 {
                return err("ModelPixelScale has fewer than two values");
            }
            let (sx, sy) = (s[0], s[1]);
            let (i, j, x, y) = (t[0], t[1], t[3], t[4]);
            transform = Some([sx, 0.0, x - i * sx, 0.0, -sy, y + j * sy]);
        }
        _ => {}
    }
    let keys = tags.get(&GEO_KEY_DIRECTORY).and_then(|t| t.ints());
    let ascii_params: String = match tags.get(&GEO_ASCII_PARAMS) {
        Some(Tag::Ascii(s)) => s.clone(),
        _ => String::new(),
    };
    let ascii_chars: Vec<char> = ascii_params.chars().collect();
    let mut model_type: Option<i64> = None;
    let mut raster_type: i64 = 1;
    let mut epsg: Option<i64> = None;
    let mut geographic_type: Option<i64> = None;
    let mut citation = String::new();
    if let Some(keys) = keys.filter(|k| k.len() >= 4) {
        let n = keys[3];
        for k in 0..n.max(0) {
            let base = (4 + 4 * k) as usize;
            if base + 3 >= keys.len() {
                break;
            }
            let (key, loc, count, value) = (keys[base], keys[base + 1], keys[base + 2], keys[base + 3]);
            if key == GT_MODEL_TYPE {
                model_type = Some(value);
            } else if key == GT_RASTER_TYPE {
                raster_type = value;
            } else if key == PROJECTED_CS_TYPE {
                epsg = Some(value);
            } else if key == GEOGRAPHIC_TYPE {
                geographic_type = Some(value);
            } else if (key == GT_CITATION || key == PCS_CITATION) && loc == GEO_ASCII_PARAMS as i64 {
                let start = (value.max(0) as usize).min(ascii_chars.len());
                let end = ((value + count).max(0) as usize).min(ascii_chars.len()).max(start);
                let piece: String = ascii_chars[start..end].iter().collect();
                citation.push_str(piece.trim_end_matches('|'));
                citation.push('|');
            }
        }
    }
    info.citation = citation.trim_matches('|').to_string();
    info.epsg = match epsg {
        Some(e) if e != 0 && e != 32767 => Some(e),
        _ => None,
    };
    if let (Some(t), true) = (transform.as_mut(), raster_type == 2) {
        // PixelIsPoint: the tie point names the pixel centre.
        let [a, b, c, d, e, f] = *t;
        *t = [a, b, c - 0.5 * a - 0.5 * b, d, e, f - 0.5 * d - 0.5 * e];
    }
    info.transform = transform;
    let mut zone: Option<i64> = None;
    let mut northern: Option<bool> = None;
    match epsg {
        Some(e) if e != 0 && (32601..=32660).contains(&e) => {
            zone = Some(e - 32600);
            northern = Some(true);
        }
        Some(e) if e != 0 && (32701..=32760).contains(&e) => {
            zone = Some(e - 32700);
            northern = Some(false);
        }
        _ => {
            if let Some((z, n)) = find_utm_zone(&ascii_params) {
                zone = Some(z);
                northern = Some(n);
            }
        }
    }
    if zone.is_some() {
        info.zone = zone;
        info.northern = northern;
    } else if model_type == Some(2)
        || (epsg.is_none()
            && geographic_type.is_some()
            && transform.is_some_and(|t| t[2].abs() <= 180.0 && t[5].abs() <= 90.0))
    {
        info.geographic = true;
    }
    Ok(())
}

// ---- Pixels ----------------------------------------------------------------

/// `unsupported_reason`: why this reader cannot decode the raster, or `None`.
pub fn unsupported_reason(info: &GeoTiffInfo) -> Option<String> {
    if ![
        COMPRESSION_NONE,
        COMPRESSION_LZW,
        COMPRESSION_JPEG,
        COMPRESSION_DEFLATE,
        COMPRESSION_PACKBITS,
        COMPRESSION_DEFLATE_OLD,
    ]
    .contains(&info.compression)
    {
        return Some(format!("compression scheme {}", info.compression));
    }
    let ns = info.samples.max(0) as usize;
    if info.bits.iter().take(ns).any(|b| *b != 8 && *b != 16) {
        return Some(format!("{}-bit samples", info.bits[0]));
    }
    if !info.sample_format.is_empty()
        && info.sample_format.iter().take(ns).any(|sf| *sf != 1 && *sf != 4)
    {
        return Some("floating-point samples (an elevation model, not a picture)".to_string());
    }
    if ![0, 1, 2, 6].contains(&info.photometric) {
        return Some(format!("photometric interpretation {}", info.photometric));
    }
    if ![1, 2, 3, 4].contains(&info.samples) && info.photometric != 2 {
        return Some(format!("{} samples per pixel", info.samples));
    }
    if info.compression == COMPRESSION_JPEG && (info.bits[0] != 8 || info.planar != 1) {
        return Some("JPEG-compressed samples in this layout".to_string());
    }
    None
}

/// An `(h, w, 4)` uint8 RGBA picture, row-major.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Rgba {
    pub h: usize,
    pub w: usize,
    pub data: Vec<u8>,
}

/// `read_rgba`: decode the raster to RGBA reduced by an integer factor so
/// its longer side fits `max_px`; returns `(picture, factor)`. `progress`
/// gets the fraction done per band and may return `false` to abort.
pub fn read_rgba(
    info: &GeoTiffInfo,
    max_px: i64,
    mut progress: Option<&mut dyn FnMut(f64) -> bool>,
) -> Result<(Rgba, i64), GeoTiffError> {
    let name = info.file_name();
    if let Some(reason) = unsupported_reason(info) {
        return err(format!("{name}: {reason} is not supported"));
    }
    if info.compression == COMPRESSION_JPEG {
        return err(format!(
            "{name}: JPEG-in-TIFF pixels are decoded by Qt in the Python app; not ported"
        ));
    }
    if info.width <= 0 || info.height <= 0 || info.tile_w <= 0 || info.tile_h <= 0 {
        return err(format!("{name}: empty raster or tile size"));
    }
    let mut factor: i64 = 1;
    while (info.width.max(info.height) as f64) / (factor as f64) > max_px as f64 {
        factor *= 2;
        if factor > 1 << 40 {
            return err(format!("{name}: cannot reduce to {max_px} px"));
        }
    }
    if info.tiled && factor > 1 {
        while info.tile_w % factor != 0 || info.tile_h % factor != 0 {
            factor *= 2;
            if factor > 1 << 40 {
                return err(format!("{name}: tile size does not reduce"));
            }
        }
    }
    let out_w = ceil_div(info.width, factor) as usize;
    let out_h = ceil_div(info.height, factor) as usize;
    let mut out = Rgba { h: out_h, w: out_w, data: vec![0u8; out_h * out_w * 4] };
    let mut reducer = RowReducer::new(factor as usize);
    let bands = ceil_div(info.height, info.tile_h);
    let mut f = File::open(&info.path).map_err(|e| GeoTiffError(format!("{name}: {e}")))?;
    for band in 0..bands {
        let rows = decode_band(&mut f, info, band)?;
        reducer.push(&mut out, rows)?;
        if let Some(cb) = progress.as_mut() {
            if !cb((band + 1) as f64 / bands as f64) {
                return err("cancelled");
            }
        }
    }
    reducer.finish(&mut out);
    Ok((out, factor))
}

fn ceil_div(a: i64, b: i64) -> i64 {
    -((-a).div_euclid(b))
}

/// Full-resolution rows of one band as RGBA.
struct Band {
    rows: usize,
    width: usize,
    data: Vec<u8>,
}

/// `_decode_band`: rows `[band * tile_h, ...)` as an `(rows, W, 4)` array.
fn decode_band(f: &mut File, info: &GeoTiffInfo, band: i64) -> Result<Band, GeoTiffError> {
    let name = info.file_name();
    let y0 = band * info.tile_h;
    let rows = info.tile_h.min(info.height - y0) as usize;
    let width = info.width as usize;
    let samples = info.samples as usize;
    let planes = if info.planar == 2 { samples } else { 1 };
    let per_plane = if info.planar == 2 { 1 } else { samples };
    let (tiles_x, per_plane_tiles) = if info.tiled {
        let tx = ceil_div(info.width, info.tile_w);
        let ty = ceil_div(info.height, info.tile_h);
        (tx, tx * ty)
    } else {
        (1, ceil_div(info.height, info.tile_h))
    };
    // numpy.empty in Python; zeros here for blocks the file does not hold.
    let mut raw = vec![0u8; rows * width * samples];
    for plane in 0..planes as i64 {
        for tx in 0..tiles_x {
            let index = plane * per_plane_tiles + band * tiles_x + tx;
            if index as usize >= info.offsets.len() {
                continue;
            }
            let Some(&count) = info.bytecounts.get(index as usize) else {
                return err(format!("{name}: byte count {index} missing"));
            };
            f.seek(SeekFrom::Start(info.offsets[index as usize].max(0) as u64))
                .map_err(|e| GeoTiffError(format!("{name}: {e}")))?;
            let mut data = Vec::new();
            (&mut *f)
                .take(count.max(0) as u64)
                .read_to_end(&mut data)
                .map_err(|e| GeoTiffError(format!("{name}: {e}")))?;
            let x0 = (tx * info.tile_w) as usize;
            let cols = (info.tile_w.min(info.width - tx * info.tile_w)) as usize;
            let block = decode_block(&data, info, per_plane)?;
            let tw = info.tile_w as usize;
            for r in 0..rows {
                for c in 0..cols {
                    for s in 0..per_plane {
                        let v = block[(r * tw + c) * per_plane + s];
                        if info.planar == 2 {
                            raw[(r * width + x0 + c) * samples + plane as usize] = v;
                        } else {
                            raw[(r * width + x0 + c) * samples + s] = v;
                        }
                    }
                }
            }
        }
    }
    let data = to_rgba(&raw, rows, width, samples, info)?;
    Ok(Band { rows, width, data })
}

/// `_decode_block`: one tile or strip to `(tile_h, tile_w, samples)` uint8
/// (16-bit data is taken down to its high byte).
fn decode_block(data: &[u8], info: &GeoTiffInfo, samples: usize) -> Result<Vec<u8>, GeoTiffError> {
    let name = info.file_name();
    let (th, tw) = (info.tile_h as usize, info.tile_w as usize);
    let need = th
        .checked_mul(tw)
        .and_then(|x| x.checked_mul(samples))
        .ok_or_else(|| GeoTiffError(format!("{name}: tile too large")))?;
    let owned: Vec<u8>;
    let data: &[u8] = match info.compression {
        COMPRESSION_DEFLATE | COMPRESSION_DEFLATE_OLD => {
            let mut out = Vec::new();
            ZlibDecoder::new(data)
                .read_to_end(&mut out)
                .map_err(|e| GeoTiffError(format!("{name}: zlib: {e}")))?;
            owned = out;
            &owned
        }
        COMPRESSION_PACKBITS => {
            owned = unpackbits(data);
            &owned
        }
        COMPRESSION_LZW => {
            owned = lzw_decode(data);
            &owned
        }
        _ => data,
    };
    if info.bits[0] == 16 {
        if data.len() & 1 != 0 {
            return err(format!("{name}: buffer size must be a multiple of element size"));
        }
        let mut arr: Vec<u16> = (0..data.len() / 2)
            .map(|i| {
                let a = [data[2 * i], data[2 * i + 1]];
                if info.little_endian {
                    u16::from_le_bytes(a)
                } else {
                    u16::from_be_bytes(a)
                }
            })
            .collect();
        arr.resize(need, 0);
        if info.predictor == 2 {
            cumsum_rows(&mut arr, tw, samples, |a, b| a.wrapping_add(b));
        }
        return Ok(arr.iter().map(|&v| (v >> 8) as u8).collect());
    }
    let mut arr = data[..data.len().min(need)].to_vec();
    arr.resize(need, 0);
    if info.predictor == 2 {
        cumsum_rows(&mut arr, tw, samples, |a: u8, b: u8| a.wrapping_add(b));
    }
    Ok(arr)
}

/// `np.cumsum(arr, axis=1)` on a `(th, tw, samples)` array.
fn cumsum_rows<T: Copy>(arr: &mut [T], tw: usize, samples: usize, add: impl Fn(T, T) -> T) {
    let row_len = tw * samples;
    if row_len == 0 {
        return;
    }
    for row in arr.chunks_mut(row_len) {
        for c in 1..tw {
            for s in 0..samples {
                row[c * samples + s] = add(row[(c - 1) * samples + s], row[c * samples + s]);
            }
        }
    }
}

/// `_to_rgba`: samples to RGBA.
fn to_rgba(
    raw: &[u8],
    rows: usize,
    width: usize,
    samples: usize,
    info: &GeoTiffInfo,
) -> Result<Vec<u8>, GeoTiffError> {
    let mut rgba = vec![0u8; rows * width * 4];
    let n = rows * width;
    if samples == 1 || (samples == 2 && (info.photometric == 0 || info.photometric == 1)) {
        for i in 0..n {
            let mut grey = raw[i * samples];
            if info.photometric == 0 {
                grey = 255 - grey;
            }
            rgba[i * 4] = grey;
            rgba[i * 4 + 1] = grey;
            rgba[i * 4 + 2] = grey;
            rgba[i * 4 + 3] = if samples == 2 { raw[i * samples + 1] } else { 255 };
        }
        return Ok(rgba);
    }
    if samples < 3 {
        return err(format!("{}: {} samples cannot make RGB", info.file_name(), samples));
    }
    for i in 0..n {
        rgba[i * 4] = raw[i * samples];
        rgba[i * 4 + 1] = raw[i * samples + 1];
        rgba[i * 4 + 2] = raw[i * samples + 2];
        rgba[i * 4 + 3] = if samples >= 4 { raw[i * samples + 3] } else { 255 };
    }
    Ok(rgba)
}

/// `_RowReducer`: streams full-resolution row bands into a picture reduced
/// by an integer factor (box average), carrying leftover rows between bands.
struct RowReducer {
    k: usize,
    row: usize,
    carry: Option<Band>,
}

impl RowReducer {
    fn new(k: usize) -> RowReducer {
        RowReducer { k, row: 0, carry: None }
    }

    fn push(&mut self, out: &mut Rgba, mut rows: Band) -> Result<(), GeoTiffError> {
        let k = self.k;
        if k == 1 {
            let n = rows.rows;
            let w = rows.width.min(out.w);
            if self.row + n > out.h {
                return err("band does not fit the output");
            }
            for r in 0..n {
                let dst = ((self.row + r) * out.w) * 4;
                let src = r * rows.width * 4;
                out.data[dst..dst + w * 4].copy_from_slice(&rows.data[src..src + w * 4]);
            }
            self.row += n;
            return Ok(());
        }
        if let Some(c) = self.carry.take() {
            let mut data = c.data;
            data.extend_from_slice(&rows.data);
            rows = Band { rows: c.rows + rows.rows, width: rows.width, data };
        }
        let full = (rows.rows / k) * k;
        if full > 0 {
            let head = Band {
                rows: full,
                width: rows.width,
                data: rows.data[..full * rows.width * 4].to_vec(),
            };
            self.emit(out, &head)?;
        }
        if rows.rows > full {
            self.carry = Some(Band {
                rows: rows.rows - full,
                width: rows.width,
                data: rows.data[full * rows.width * 4..].to_vec(),
            });
        }
        Ok(())
    }

    fn finish(&mut self, out: &mut Rgba) {
        if let Some(c) = self.carry.take() {
            if c.rows > 0 {
                let _ = self.emit(out, &c);
            }
        }
    }

    /// `_emit`: box mean of `k x k` blocks, edge-padded to a multiple of `k`.
    fn emit(&mut self, out: &mut Rgba, rows: &Band) -> Result<(), GeoTiffError> {
        let k = self.k;
        let (h, w) = (rows.rows, rows.width);
        let ph = h.div_ceil(k) * k;
        let pw = w.div_ceil(k) * k;
        let (nh, nw) = (ph / k, pw / k);
        if self.row + nh > out.h {
            return err("band does not fit the output");
        }
        let cols = nw.min(out.w);
        for by in 0..nh {
            for bx in 0..cols {
                for ch in 0..4 {
                    let mut sum: u64 = 0;
                    for dy in 0..k {
                        let y = (by * k + dy).min(h - 1);
                        for dx in 0..k {
                            let x = (bx * k + dx).min(w - 1);
                            sum += rows.data[(y * w + x) * 4 + ch] as u64;
                        }
                    }
                    out.data[((self.row + by) * out.w + bx) * 4 + ch] = (sum / (k * k) as u64) as u8;
                }
            }
        }
        self.row += nh;
        Ok(())
    }
}

/// `describe`: one line for the status bar: size, scheme and placement.
pub fn describe(info: &GeoTiffInfo) -> String {
    let comp = match info.compression {
        1 => "raw".to_string(),
        5 => "LZW".to_string(),
        7 => "JPEG".to_string(),
        8 | 32946 => "Deflate".to_string(),
        32773 => "PackBits".to_string(),
        c => c.to_string(),
    };
    let where_ = match info.zone {
        Some(z) if z != 0 => {
            format!("UTM {}{}", z, if info.northern == Some(true) { 'N' } else { 'S' })
        }
        _ => {
            if info.geographic {
                "lat/lon".to_string()
            } else {
                "no CRS".to_string()
            }
        }
    };
    format!(
        "{}\u{d7}{} px, {}\u{d7}{} bit, {}, {}, {}",
        info.width,
        info.height,
        info.samples,
        info.bits[0],
        comp,
        if info.tiled { "tiled" } else { "strips" },
        where_
    )
}
