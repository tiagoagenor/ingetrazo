// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! A canonical dump of a parsed [`Model`], the Rust half of
//! `scripts/golden/skp_dump.py`: every float as the hex of its big-endian
//! double, everything else as itself, keys sorted, no spaces, ASCII only.
//! Two parses are equal when the dumps are equal text. It exists so the
//! parity tests can compare with openskp's parse bit for bit (and a real
//! model by the hash of its dump).

use std::collections::BTreeMap;
use std::fmt::Write;

use crate::model::*;

/// A JSON value with ordered keys.
#[derive(Clone, Debug)]
pub enum J {
    Null,
    Bool(bool),
    Int(i64),
    Str(String),
    Arr(Vec<J>),
    Obj(BTreeMap<String, J>),
}

impl J {
    /// `json.dumps(v, sort_keys=True, separators=(",", ":"))`.
    pub fn text(&self) -> String {
        let mut out = String::new();
        self.write(&mut out);
        out
    }

    fn write(&self, out: &mut String) {
        match self {
            J::Null => out.push_str("null"),
            J::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
            J::Int(i) => {
                let _ = write!(out, "{i}");
            }
            J::Str(s) => write_str(s, out),
            J::Arr(a) => {
                out.push('[');
                for (i, x) in a.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    x.write(out);
                }
                out.push(']');
            }
            J::Obj(o) => {
                out.push('{');
                for (i, (k, v)) in o.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    write_str(k, out);
                    out.push(':');
                    v.write(out);
                }
                out.push('}');
            }
        }
    }
}

fn write_str(s: &str, out: &mut String) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 || (c as u32) > 0x7f => {
                let mut buf = [0u16; 2];
                for u in c.encode_utf16(&mut buf) {
                    let _ = write!(out, "\\u{u:04x}");
                }
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

fn fb(x: f64) -> J {
    J::Str(format!("{:016x}", x.to_bits()))
}

fn fbs(v: &[f64]) -> J {
    J::Arr(v.iter().map(|&x| fb(x)).collect())
}

fn ofbs(v: &Option<[f64; 3]>) -> J {
    match v {
        Some(p) => fbs(p),
        None => J::Null,
    }
}

fn s(x: &str) -> J {
    J::Str(x.to_string())
}

fn i(x: i64) -> J {
    J::Int(x)
}

fn oi(x: Option<i64>) -> J {
    x.map(J::Int).unwrap_or(J::Null)
}

fn arr(items: Vec<J>) -> J {
    J::Arr(items)
}

fn val(v: &AttrValue) -> J {
    let one = |k: &str, x: J| J::Obj(BTreeMap::from([(k.to_string(), x)]));
    match v {
        AttrValue::Null => J::Null,
        AttrValue::Int(n) => one("i", J::Int(*n)),
        AttrValue::Float(x) => one("f", fb(*x)),
        AttrValue::Str(t) => one("s", s(t)),
        AttrValue::Vec3(p) => one("p", fbs(p)),
        AttrValue::List(l) => one("l", arr(l.iter().map(val).collect())),
    }
}

fn dump_def(d: &Definition, is_root: bool) -> J {
    let mut o = BTreeMap::new();
    o.insert("id".to_string(), i(d.id));
    o.insert("guid".into(), s(&d.guid));
    o.insert("name".into(), s(&d.name));
    o.insert("afc".into(), J::Bool(d.always_faces_camera));
    o.insert("sfs".into(), J::Bool(d.shadows_face_sun));
    o.insert("img".into(), J::Bool(d.is_image));
    o.insert("v".into(), arr(d.vertices.iter().map(|(k, v)| arr(vec![i(*k), fb(v.x), fb(v.y), fb(v.z)])).collect()));
    o.insert(
        "e".into(),
        arr(d
            .edges
            .iter()
            .map(|(k, e)| arr(vec![i(*k), i(e.id), i(e.v1_id), i(e.v2_id), J::Bool(e.soft), J::Bool(e.smooth), J::Bool(e.hidden), oi(e.layer), oi(e.curve_id)]))
            .collect()),
    );
    o.insert(
        "f".into(),
        arr(d
            .faces
            .iter()
            .map(|(k, f)| {
                let layer = match f.layer_id {
                    None => J::Null,
                    Some(n) if is_root => arr(vec![s("i"), i(n)]),
                    Some(_) => arr(vec![s("s"), s(&f.layer)]),
                };
                arr(vec![
                    i(*k),
                    i(f.id),
                    arr(f.loops.iter().map(|lp| arr(lp.iter().map(|&(e, sg)| arr(vec![i(e), i(sg as i64)])).collect())).collect()),
                    ofbs(&f.normal),
                    oi(f.material_id),
                    oi(f.back_material_id),
                    f.uv_transform.as_ref().map(|m| fbs(m)).unwrap_or(J::Null),
                    f.uv_transform_back.as_ref().map(|m| fbs(m)).unwrap_or(J::Null),
                    J::Bool(f.uv_projected),
                    J::Bool(f.uv_projected_back),
                    J::Bool(f.hidden),
                    layer,
                ])
            })
            .collect()),
    );
    o.insert(
        "i".into(),
        arr(d
            .instances
            .iter()
            .map(|n| {
                arr(vec![
                    s(&n.name),
                    oi(n.ref_idx),
                    s(&n.guid),
                    fbs(&n.matrix),
                    s(&n.layer),
                    oi(n.material_id),
                    J::Bool(n.hidden),
                    arr(n.properties.iter().map(|(k, v)| arr(vec![s(k), s(v)])).collect()),
                    arr(n.attribute_dictionaries.iter().map(|(name, e)| arr(vec![s(name), arr(e.iter().map(|(k, v)| arr(vec![s(k), val(v)])).collect())])).collect()),
                ])
            })
            .collect()),
    );
    o.insert("sp".into(), arr(d.section_planes.iter().map(|p| arr(vec![fbs(&p.plane), s(&p.name), s(&p.label), J::Bool(p.hidden)])).collect()));
    o.insert("t".into(), arr(d.texts.iter().map(|t| arr(vec![s(&t.text), J::Bool(t.hidden), ofbs(&t.point), ofbs(&t.label_point)])).collect()));
    o.insert("d".into(), arr(d.dimensions.iter().map(dump_dim).collect()));
    o.insert("cl".into(), arr(d.construction_lines.iter().map(|c| arr(vec![fbs(&c.point), fbs(&c.direction), ofbs(&c.start), ofbs(&c.end)])).collect()));
    o.insert("cp".into(), arr(d.construction_points.iter().map(|c| arr(vec![fbs(&c.position)])).collect()));
    J::Obj(o)
}

fn dump_dim(x: &Dimension) -> J {
    arr(vec![s(&x.text), J::Bool(x.hidden), ofbs(&x.a), ofbs(&x.b), fb(x.offset), ofbs(&x.plane_x), ofbs(&x.normal)])
}

/// The dump of a model (see the module doc).
pub fn model_dump(m: &Model) -> J {
    let mut o = BTreeMap::new();
    o.insert("version".to_string(), s(&m.version));
    o.insert("units".into(), m.units.as_deref().map(s).unwrap_or(J::Null));
    o.insert("defs".into(), arr(m.definitions.iter().map(|d| dump_def(d, false)).collect()));
    o.insert("root".into(), dump_def(&m.root, true));
    o.insert("layers".into(), arr(m.layers.iter().map(|l| arr(vec![s(&l.name), i(l.color[0]), i(l.color[1]), i(l.color[2]), J::Bool(l.hidden)])).collect()));
    o.insert(
        "pages".into(),
        arr(m
            .pages
            .iter()
            .map(|p| arr(vec![s(&p.name), ofbs(&p.eye), ofbs(&p.target), ofbs(&p.up), fb(p.fov), J::Bool(p.parallel), fb(p.ortho_height), arr(p.hidden_layers.iter().map(|x| s(x)).collect())]))
            .collect()),
    );
    o.insert("dims".into(), arr(m.dimensions.iter().map(dump_dim).collect()));
    o.insert(
        "mats".into(),
        arr(m
            .materials
            .iter()
            .map(|x| {
                let tex = match &x.texture {
                    None => J::Null,
                    Some(t) => arr(vec![s(&t.filename), fb(t.width), fb(t.height), t.data.as_ref().map(|d| s(&sha256_hex(d))).unwrap_or(J::Null)]),
                };
                arr(vec![s(&x.name), arr(x.color.iter().map(|&c| i(c)).collect()), fb(x.transparency), oi(x.id), J::Bool(x.colorized), i(x.colorize_type), tex])
            })
            .collect()),
    );
    o.insert("by_id".into(), arr(m.materials_by_id.iter().map(|&(k, ix)| arr(vec![i(k), i(ix as i64)])).collect()));
    o.insert(
        "styles".into(),
        arr(m
            .styles
            .iter()
            .map(|st| {
                let c = |c: &Option<[i64; 3]>| c.map(|c| arr(c.iter().map(|&x| i(x)).collect())).unwrap_or(J::Null);
                arr(vec![s(&st.name), c(&st.front_color), c(&st.back_color)])
            })
            .collect()),
    );
    J::Obj(o)
}

/// SHA-256 of `data`, as lower-case hex (the dump hashes texture images).
pub fn sha256_hex(data: &[u8]) -> String {
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1,
        0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138,
        0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116, 0x1e376c08, 0x2748774c,
        0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
    ];
    let mut h: [u32; 8] = [0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19];
    let mut msg = data.to_vec();
    msg.push(0x80);
    while msg.len() % 64 != 56 {
        msg.push(0);
    }
    msg.extend(((data.len() as u64) * 8).to_be_bytes());
    for chunk in msg.chunks(64) {
        let mut w = [0u32; 64];
        for t in 0..16 {
            w[t] = u32::from_be_bytes([chunk[4 * t], chunk[4 * t + 1], chunk[4 * t + 2], chunk[4 * t + 3]]);
        }
        for t in 16..64 {
            let s0 = w[t - 15].rotate_right(7) ^ w[t - 15].rotate_right(18) ^ (w[t - 15] >> 3);
            let s1 = w[t - 2].rotate_right(17) ^ w[t - 2].rotate_right(19) ^ (w[t - 2] >> 10);
            w[t] = w[t - 16].wrapping_add(s0).wrapping_add(w[t - 7]).wrapping_add(s1);
        }
        let mut v = h;
        for t in 0..64 {
            let s1 = v[4].rotate_right(6) ^ v[4].rotate_right(11) ^ v[4].rotate_right(25);
            let ch = (v[4] & v[5]) ^ (!v[4] & v[6]);
            let t1 = v[7].wrapping_add(s1).wrapping_add(ch).wrapping_add(K[t]).wrapping_add(w[t]);
            let s0 = v[0].rotate_right(2) ^ v[0].rotate_right(13) ^ v[0].rotate_right(22);
            let maj = (v[0] & v[1]) ^ (v[0] & v[2]) ^ (v[1] & v[2]);
            let t2 = s0.wrapping_add(maj);
            v = [t1.wrapping_add(t2), v[0], v[1], v[2], v[3].wrapping_add(t1), v[4], v[5], v[6]];
        }
        for k in 0..8 {
            h[k] = h[k].wrapping_add(v[k]);
        }
    }
    h.iter().map(|x| format!("{x:08x}")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sha256_known_vectors() {
        assert_eq!(sha256_hex(b""), "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
        assert_eq!(sha256_hex(b"abc"), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    }

    #[test]
    fn json_text_is_python_canonical() {
        let j = J::Obj(BTreeMap::from([("b".to_string(), J::Int(1)), ("a".to_string(), J::Arr(vec![J::Str("\u{f1}\"x\u{1F600}".into()), J::Null, J::Bool(true)]))]));
        assert_eq!(j.text(), "{\"a\":[\"\\u00f1\\\"x\\ud83d\\ude00\",null,true],\"b\":1}");
    }
}
