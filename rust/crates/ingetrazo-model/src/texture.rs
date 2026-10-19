// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! The pure-data part of `core/texture.py`: the texture dict, the file name a
//! texture gets inside a document or the cache, and the "is this really an
//! image" test. Image decoding (cut-out detection, tinting, the TGA probe)
//! and the UV maths are not here.

use std::io::Read;
use std::path::Path;

use ingetrazo_formats::pyjson::Value;

use crate::pyalnum::is_alnum;
use crate::pyval::{self, err, get, set, truthy, PyResult};

/// Characters kept of an image name inside the cache.
pub const NAME_LIMIT: usize = 64;

/// `Texture`: an image file and the real-world size of one tile (metres).
#[derive(Clone, Debug, PartialEq)]
pub struct Texture {
    pub path: Value,
    pub sw: f64,
    pub sh: f64,
}

impl Texture {
    /// `Texture.as_dict()`.
    pub fn as_dict(&self) -> Value {
        Value::Object(vec![
            ("path".to_string(), self.path.clone()),
            ("sw".to_string(), Value::Float(self.sw)),
            ("sh".to_string(), Value::Float(self.sh)),
        ])
    }

    /// `Texture.from_dict(d)`: `d["path"]` is required; `sw` and `sh`
    /// default to 1.0 and go through `float()`.
    pub fn from_dict(d: &Value) -> PyResult<Texture> {
        let path = match d {
            Value::Object(_) => match d.get("path") {
                Some(p) => p.clone(),
                None => return err("KeyError", "'path'"),
            },
            _ => return err("TypeError", "object is not subscriptable"),
        };
        let sw = pyval::float(&pyval::get_or(d, "sw", Value::Float(1.0))?)?;
        let sh = pyval::float(&pyval::get_or(d, "sh", Value::Float(1.0))?)?;
        Ok(Texture { path, sw, sh })
    }
}

/// `Path(name).name` as pathlib computes it: `windows` selects
/// `PureWindowsPath` (both separators, drives and UNC prefixes) or
/// `PurePosixPath`.
fn path_name(name: &str, windows: bool) -> String {
    let rest: String;
    if windows {
        let norm: Vec<char> = name.replace('/', "\\").chars().collect();
        let mut at = 0;
        if norm.len() >= 2 && norm[0] == '\\' && norm[1] == '\\' {
            // UNC: `\\server\share` is the drive; nothing after it, no name.
            let find = |from: usize| norm.iter().skip(from).position(|&c| c == '\\').map(|i| i + from);
            let start = {
                let head: String = norm.iter().take(8).collect::<String>().to_uppercase();
                if head == "\\\\?\\UNC\\" {
                    8
                } else {
                    2
                }
            };
            match find(start).and_then(|i| find(i + 1)) {
                Some(i2) => at = i2,
                None => return String::new(),
            }
        } else if norm.len() >= 2 && norm[1] == ':' {
            at = 2;
        }
        rest = norm[at..].iter().collect();
        return last_part(&rest, '\\');
    }
    last_part(name, '/')
}

fn last_part(path: &str, sep: char) -> String {
    path.split(sep).rfind(|p| !p.is_empty() && *p != ".").unwrap_or("").to_string()
}

/// `Path(name).suffix`.
fn suffix(name: &str, windows: bool) -> String {
    let base = path_name(name, windows);
    match base.rfind('.') {
        Some(i) if i > 0 && i + 1 < base.len() => base[i..].to_string(),
        _ => String::new(),
    }
}

fn trim_dots(text: &str) -> &str {
    text.trim_matches([' ', '.'])
}

fn strip_digest_prefix(text: &str) -> &str {
    let mut rest = text;
    loop {
        let b = rest.as_bytes();
        let hex16 = b.len() >= 17
            && b[..16].iter().all(|c| matches!(c, b'0'..=b'9' | b'a'..=b'f'))
            && b[16] == b'-';
        if !hex16 {
            return rest;
        }
        rest = &rest[17..];
    }
}

fn clip(text: &str, chars: usize) -> &str {
    match text.char_indices().nth(chars) {
        Some((i, _)) => &text[..i],
        None => text,
    }
}

/// `texture_file_name(name)` for the platform this build runs on.
pub fn texture_file_name(name: &str) -> String {
    texture_file_name_for(name, cfg!(windows))
}

/// `texture_file_name(name)`: a filesystem-safe base name: characters
/// outside the safe set dropped, any content-hash prefix stripped, clipped to
/// [`NAME_LIMIT`]. `windows` picks the path rules `pathlib` used when the
/// name was split into directory and file.
pub fn texture_file_name_for(name: &str, windows: bool) -> String {
    let base = path_name(name, windows);
    let kept: String = base.chars().filter(|&c| is_alnum(c) || matches!(c, ' ' | '.' | '_' | '-')).collect();
    let mut safe = trim_dots(&kept).to_string();
    safe = trim_dots(strip_digest_prefix(&safe)).to_string();
    if safe.chars().count() > NAME_LIMIT {
        let clipped = match safe.rfind('.') {
            Some(dot) => {
                let (stem, ext) = (&safe[..dot], &safe[dot + 1..]);
                let n = ext.chars().count();
                if n > 0 && n <= 5 && ext.chars().all(is_alnum) {
                    let head = clip(stem, NAME_LIMIT - n - 1);
                    Some(format!("{}.{}", head.trim_end_matches([' ', '.']), ext))
                } else {
                    None
                }
            }
            None => None,
        };
        safe = clipped.unwrap_or_else(|| clip(&safe, NAME_LIMIT).trim_end_matches([' ', '.']).to_string());
    }
    if safe.is_empty() {
        "texture.png".to_string()
    } else {
        safe
    }
}

const IMAGE_MAGIC: [&[u8]; 7] = [
    b"\x89PNG\r\n\x1a\n",
    b"\xff\xd8\xff",
    b"GIF87a",
    b"GIF89a",
    b"BM",
    b"II*\x00",
    b"MM\x00*",
];

/// `is_image_bytes(data)`: whether `data` starts like a PNG, JPEG, GIF, BMP,
/// TIFF or WebP file.
pub fn is_image_bytes(data: &[u8]) -> bool {
    let head = &data[..data.len().min(16)];
    if IMAGE_MAGIC.iter().any(|m| head.starts_with(m)) {
        return true;
    }
    head.len() >= 12 && &head[..4] == b"RIFF" && &head[8..12] == b"WEBP"
}

/// `is_image_file(path)`: whether `path` is a readable image. TGA has no
/// signature: it is accepted by extension plus `tga_readable`, the decoder's
/// answer (Qt's `QImageReader.canRead` in Python), which this crate does not
/// have.
pub fn is_image_file(path: &Path, tga_readable: &dyn Fn(&Path) -> bool) -> bool {
    let mut head = Vec::new();
    let read = std::fs::File::open(path).and_then(|f| f.take(16).read_to_end(&mut head));
    if read.is_err() {
        return false;
    }
    if is_image_bytes(&head) {
        return true;
    }
    let name = path.to_string_lossy();
    suffix(&name, cfg!(windows)).to_lowercase() == ".tga" && tga_readable(path)
}

/// `flattened_texture(tex)` with no normal: `tex` without its per-face map
/// (`uvw`). (With a normal, a fitted map is converted to a planar look, which
/// needs the vector maths and lives with the geometry.)
pub fn flattened_texture(tex: &Value) -> PyResult<Value> {
    match tex {
        Value::Object(items) => {
            Ok(Value::Object(items.iter().filter(|(k, _)| k != "uvw").cloned().collect()))
        }
        _ => err("AttributeError", "object has no attribute 'items'"),
    }
}

/// `untinted_texture(tex)`: the entry with its colour removed: back to the
/// original image.
pub fn untinted_texture(tex: &Value) -> PyResult<Value> {
    let base = get(tex, "base")?;
    let mut entry: Vec<(String, Value)> = match tex {
        Value::Object(items) => items
            .iter()
            .filter(|(k, _)| !matches!(k.as_str(), "base" | "tint" | "tint_mode"))
            .cloned()
            .collect(),
        _ => unreachable!("get() already refused a non-dict"),
    };
    if let Some(b) = base.filter(|b| truthy(b)) {
        set(&mut entry, "path", b.clone());
    }
    Ok(Value::Object(entry))
}
