// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! The DXF writer of `formats/dxf_out.py`: an ASCII R12 file with LINE
//! entities, byte for byte. Python opens the file in text mode with
//! `encoding="ascii", errors="replace"`, so on Windows every line ends in
//! `\r\n` and a character outside ASCII becomes `?`; [`save_dxf_layers_text`]
//! returns the text with `\n` and [`save_dxf_layers`] takes `crlf`.

use std::path::Path;

/// One `(layer, segments[, linetype])` group: segments are `(x0, y0, x1, y1)`.
#[derive(Clone, Debug)]
pub struct LayerGroup {
    pub layer: String,
    pub segments: Vec<[f64; 4]>,
    /// `"DASHED"` for hidden lines; empty or `None` is `CONTINUOUS`.
    pub linetype: Option<String>,
}

impl LayerGroup {
    pub fn new(layer: &str, segments: Vec<[f64; 4]>) -> LayerGroup {
        LayerGroup { layer: layer.to_string(), segments, linetype: None }
    }

    pub fn dashed(layer: &str, segments: Vec<[f64; 4]>) -> LayerGroup {
        LayerGroup { layer: layer.to_string(), segments, linetype: Some("DASHED".to_string()) }
    }
}

/// `_layer_name(name)`: `(name or "VISTA").strip().upper().replace(" ", "_") or "VISTA"`.
pub fn layer_name(name: &str) -> String {
    let base = if name.is_empty() { "VISTA" } else { name };
    let up = base.trim().to_uppercase().replace(' ', "_");
    if up.is_empty() {
        "VISTA".to_string()
    } else {
        up
    }
}

/// `format(x, ".9g")`.
pub fn fmt_g9(x: f64) -> String {
    if x.is_nan() {
        return "nan".to_string();
    }
    if x.is_infinite() {
        return if x < 0.0 { "-inf" } else { "inf" }.to_string();
    }
    if x == 0.0 {
        return if x.is_sign_negative() { "-0" } else { "0" }.to_string();
    }
    // Nine significant digits, correctly rounded, as an exponent form.
    let sci = format!("{:.8e}", x);
    let (mant, exp) = sci.split_once('e').unwrap();
    let exp: i32 = exp.parse().unwrap();
    let (sign, mant) = match mant.strip_prefix('-') {
        Some(m) => ("-", m),
        None => ("", mant),
    };
    let digits: String = mant.chars().filter(|c| *c != '.').collect();
    let mut out = String::from(sign);
    if exp < -4 || exp >= 9 {
        let d = digits.trim_end_matches('0');
        let d = if d.is_empty() { "0" } else { d };
        out.push_str(&d[..1]);
        if d.len() > 1 {
            out.push('.');
            out.push_str(&d[1..]);
        }
        out.push_str(&format!("e{}{:02}", if exp < 0 { '-' } else { '+' }, exp.abs()));
    } else if exp >= 0 {
        let int_len = exp as usize + 1;
        let (int, frac) = digits.split_at(int_len);
        out.push_str(int);
        let frac = frac.trim_end_matches('0');
        if !frac.is_empty() {
            out.push('.');
            out.push_str(frac);
        }
    } else {
        let zeros = (-exp - 1) as usize;
        let d = digits.trim_end_matches('0');
        out.push_str("0.");
        out.push_str(&"0".repeat(zeros));
        out.push_str(d);
    }
    out
}

/// `save_dxf_layers(path, groups)` as text with `\n` line ends and the
/// number of LINE entities written.
pub fn save_dxf_layers_text(groups: &[LayerGroup]) -> (String, usize) {
    let mut names: Vec<(String, String)> = Vec::new();
    for g in groups {
        let name = layer_name(&g.layer);
        let ltype = match &g.linetype {
            Some(l) if !l.is_empty() => l.clone(),
            _ => "CONTINUOUS".to_string(),
        };
        if !names.iter().any(|(n, _)| *n == name) {
            names.push((name, ltype));
        }
    }
    let dashed = names.iter().any(|(_, l)| l == "DASHED");
    let mut w = String::new();
    w.push_str("0\nSECTION\n2\nTABLES\n");
    if dashed {
        w.push_str(
            "0\nTABLE\n2\nLTYPE\n70\n1\n\
             0\nLTYPE\n2\nDASHED\n70\n0\n3\n__ __ __ __\n72\n65\n\
             73\n2\n40\n0.75\n49\n0.5\n49\n-0.25\n0\nENDTAB\n",
        );
    }
    w.push_str(&format!("0\nTABLE\n2\nLAYER\n70\n{}\n", names.len()));
    for (name, ltype) in &names {
        w.push_str(&format!("0\nLAYER\n2\n{name}\n70\n0\n62\n7\n6\n{ltype}\n"));
    }
    w.push_str("0\nENDTAB\n0\nENDSEC\n");
    w.push_str("0\nSECTION\n2\nENTITIES\n");
    let mut n = 0;
    for g in groups {
        let name = layer_name(&g.layer);
        for s in &g.segments {
            w.push_str(&format!(
                "0\nLINE\n8\n{name}\n10\n{}\n20\n{}\n30\n0\n11\n{}\n21\n{}\n31\n0\n",
                fmt_g9(s[0]),
                fmt_g9(s[1]),
                fmt_g9(s[2]),
                fmt_g9(s[3])
            ));
            n += 1;
        }
    }
    w.push_str("0\nENDSEC\n0\nEOF\n");
    (w, n)
}

/// The text as `open(path, "w", encoding="ascii", errors="replace")` writes
/// it: characters outside ASCII become `?`, lines end in `\r\n` when `crlf`.
pub fn ascii_bytes(text: &str, crlf: bool) -> Vec<u8> {
    let mut out = Vec::with_capacity(text.len());
    for c in text.chars() {
        if c == '\n' && crlf {
            out.extend_from_slice(b"\r\n");
        } else if c.is_ascii() {
            out.push(c as u8);
        } else {
            out.push(b'?');
        }
    }
    out
}

/// `save_dxf_layers(path, groups)`: writes the file; returns the line count.
pub fn save_dxf_layers(path: &Path, groups: &[LayerGroup], crlf: bool) -> std::io::Result<usize> {
    let (text, n) = save_dxf_layers_text(groups);
    std::fs::write(path, ascii_bytes(&text, crlf))?;
    Ok(n)
}

/// `save_dxf_lines(path, segments, layer)`.
pub fn save_dxf_lines(path: &Path, segments: Vec<[f64; 4]>, layer: &str, crlf: bool) -> std::io::Result<usize> {
    save_dxf_layers(path, &[LayerGroup::new(layer, segments)], crlf)
}
