// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! Re-tinting a texture: `core.texture.colorize_image`, with `rgb_to_hls` and
//! `hls_to_rgb`.
//!
//! The .skp format lets a textured material carry a colour and re-tints its
//! image toward it. Mode 0 ("shift") moves every pixel by the difference
//! between the image average and the colour, mode 1 ("tint") replaces hue and
//! saturation and keeps each pixel's lightness. Alpha is kept.
//!
//! The arithmetic is NumPy's, element by element in `f64`, so a PNG gives
//! the same pixels. What differs from Python is the file: it is written by
//! the `png` crate, not Qt's encoder (the bytes differ, the pixels do not);
//! and a JPEG source is decoded by `zune-jpeg`, which may differ from
//! libjpeg by a level or two.

use crate::img::{self, Decoded};

/// Python's `a % b` for floats (`np.remainder`).
fn py_mod(a: f64, b: f64) -> f64 {
    // `fmod(a, 1.0)` is `a - trunc(a)`, exactly, and a library call is slow
    // on every pixel
    if b == 1.0 && a.abs() < 4.5e15 {
        let m = a - a.trunc();
        return if m < 0.0 {
            m + 1.0
        } else if m == 0.0 {
            0.0
        } else {
            m
        };
    }
    let mut m = a % b;
    if m != 0.0 {
        if (b < 0.0) != (m < 0.0) {
            m += b;
        }
    } else {
        m = 0.0f64.copysign(b);
    }
    m
}

/// `rgb_to_hls` on one colour in 0..1: `(hue, lum, sat)`.
pub fn rgb_to_hls(r: f64, g: f64, b: f64) -> (f64, f64, f64) {
    let maxc = r.max(g).max(b);
    let minc = r.min(g).min(b);
    let lum = (maxc + minc) / 2.0;
    let delta = maxc - minc;
    let nz = delta > 1e-12;
    let denom = if lum <= 0.5 { maxc + minc } else { 2.0 - maxc - minc };
    let sat = if nz && denom > 1e-12 { delta / denom } else { 0.0 };
    let safe = if nz { delta } else { 1.0 };
    let rc = (maxc - r) / safe;
    let gc = (maxc - g) / safe;
    let bc = (maxc - b) / safe;
    let hue = if maxc == r {
        bc - gc
    } else if maxc == g {
        2.0 + rc - bc
    } else {
        4.0 + gc - rc
    };
    let hue = if nz { py_mod(hue / 6.0, 1.0) } else { 0.0 };
    (hue, lum, sat)
}

/// `hls_to_rgb` on one colour: `[r, g, b]` in 0..1.
pub fn hls_to_rgb(hue: f64, lum: f64, sat: f64) -> [f64; 3] {
    let m2 = if lum <= 0.5 { lum * (1.0 + sat) } else { lum + sat - lum * sat };
    let m1 = 2.0 * lum - m2;
    let channel = |h: f64| -> f64 {
        let h = py_mod(h, 1.0);
        if h < 1.0 / 6.0 {
            m1 + (m2 - m1) * h * 6.0
        } else if h < 0.5 {
            m2
        } else if h < 2.0 / 3.0 {
            m1 + (m2 - m1) * (2.0 / 3.0 - h) * 6.0
        } else {
            m1
        }
    };
    [channel(hue + 1.0 / 3.0), channel(hue), channel(hue - 1.0 / 3.0)]
}

/// `colorize_image(data, target_rgb, ctype)`: PNG bytes, or `data` unchanged
/// when the image cannot be read.
pub fn colorize_image(data: &[u8], target: &[i64; 4], ctype: i64) -> Vec<u8> {
    let Decoded::Image(w, h, px) = img::decode_rgba8(data) else { return data.to_vec() };
    if w == 0 || h == 0 {
        return data.to_vec();
    }
    // The mean of the visible pixels (all of them when none is visible),
    // accumulated row by row as NumPy reduces axis 0.
    let mut sum = [0.0f64; 3];
    let mut count = 0usize;
    let any_visible = px.iter().any(|p| p[3] > 0);
    for p in &px {
        if any_visible && p[3] == 0 {
            continue;
        }
        for k in 0..3 {
            sum[k] += p[k] as f64 / 255.0;
        }
        count += 1;
    }
    let avg = [sum[0] / count as f64, sum[1] / count as f64, sum[2] / count as f64];
    let (h0, l0, s0) = rgb_to_hls(avg[0], avg[1], avg[2]);
    let (ht, lt, st) = rgb_to_hls(target[0] as f64 / 255.0, target[1] as f64 / 255.0, target[2] as f64 / 255.0);
    let mut out = Vec::with_capacity(w * h * 4);
    for p in &px {
        let (hue, lum, sat) = rgb_to_hls(p[0] as f64 / 255.0, p[1] as f64 / 255.0, p[2] as f64 / 255.0);
        let (hue, sat) = if ctype == 1 {
            (ht, st)
        } else {
            (py_mod(hue + (ht - h0), 1.0), (sat + (st - s0)).clamp(0.0, 1.0))
        };
        let lum = (lum + (lt - l0)).clamp(0.0, 1.0);
        let rgb = hls_to_rgb(hue, lum, sat);
        for c in rgb {
            out.push((c * 255.0 + 0.5).clamp(0.0, 255.0) as u8);
        }
        out.push(p[3]);
    }
    let mut png_bytes = Vec::new();
    let mut enc = png::Encoder::new(&mut png_bytes, w as u32, h as u32);
    enc.set_color(png::ColorType::Rgba);
    enc.set_depth(png::BitDepth::Eight);
    enc.set_compression(png::Compression::Fast);
    let ok = enc.write_header().and_then(|mut wr| wr.write_image_data(&out)).is_ok();
    if ok {
        png_bytes
    } else {
        data.to_vec()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hls_round_trips_like_colorsys() {
        // colorsys.rgb_to_hls(0.2, 0.4, 0.6) -> (0.5833333333333334, 0.4, 0.5)
        let (h, l, s) = rgb_to_hls(0.2, 0.4, 0.6);
        assert!((h - 0.5833333333333334).abs() < 1e-15 && (l - 0.4).abs() < 1e-15 && (s - 0.5).abs() < 1e-15);
        let rgb = hls_to_rgb(h, l, s);
        assert!((rgb[0] - 0.2).abs() < 1e-12 && (rgb[1] - 0.4).abs() < 1e-12 && (rgb[2] - 0.6).abs() < 1e-12);
    }

    #[test]
    fn python_modulo_takes_the_sign_of_the_divisor() {
        assert_eq!(py_mod(-0.25, 1.0), 0.75);
        assert_eq!(py_mod(1.5, 1.0), 0.5);
        assert!(py_mod(-0.0, 1.0).is_sign_positive());
    }
}
