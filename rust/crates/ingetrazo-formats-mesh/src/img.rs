// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! What the glTF exporter and importer ask of an image, answered without Qt:
//! `QImage.hasAlphaChannel()` and `core.texture.image_has_cutout`.
//!
//! PNG is decoded exactly (the `png` crate, palettes and `tRNS` expanded, as
//! Qt does). JPEG never has alpha. For GIF and WebP only the header flag is
//! read, and BMP, TIFF and TGA are taken as opaque: a see-through image in
//! one of those formats is not recognised (JPEG and PNG cover the textures
//! the app writes and the ones Sketchfab ships).

use std::io::Cursor;

/// The decoded alpha plane of a PNG, 8 bits per pixel, with its size; `None`
/// when the file is not a PNG, is damaged, or has no alpha at all.
struct Alpha {
    width: usize,
    height: usize,
    plane: Vec<u8>,
}

enum Probe {
    /// Not a PNG, or one Qt could not read.
    NotPng,
    /// A readable PNG without alpha.
    Opaque,
    Alpha(Alpha),
}

/// `qt_div_257`: the 16-bit channel as 8 bits, as `QRgba64::alpha8` does.
fn div_257(x: u32) -> u8 {
    ((x + 128 - ((x + 128) >> 8)) >> 8) as u8
}

fn probe_png(data: &[u8]) -> Probe {
    if !data.starts_with(b"\x89PNG\r\n\x1a\n") {
        return Probe::NotPng;
    }
    let mut decoder = png::Decoder::new(Cursor::new(data));
    decoder.set_transformations(png::Transformations::EXPAND);
    let Ok(mut reader) = decoder.read_info() else { return Probe::NotPng };
    // A palette's transparency is an alpha channel only if some entry is not
    // opaque (Qt: `has_alpha_clut`); a colour key always makes one.
    let info = reader.info();
    if info.color_type == png::ColorType::Indexed && !info.trns.as_ref().is_some_and(|t| t.iter().any(|&a| a < 255)) {
        return Probe::Opaque;
    }
    let Some(size) = reader.output_buffer_size() else { return Probe::NotPng };
    let mut buf = vec![0u8; size];
    let Ok(frame) = reader.next_frame(&mut buf) else { return Probe::NotPng };
    let (channels, alpha_at) = match frame.color_type {
        png::ColorType::Rgba => (4, 3),
        png::ColorType::GrayscaleAlpha => (2, 1),
        _ => return Probe::Opaque,
    };
    let bytes = if frame.bit_depth == png::BitDepth::Sixteen { 2 } else { 1 };
    let (w, h) = (frame.width as usize, frame.height as usize);
    let mut plane = Vec::with_capacity(w * h);
    for px in 0..w * h {
        let at = (px * channels + alpha_at) * bytes;
        plane.push(if bytes == 2 { div_257(u32::from(buf[at]) << 8 | u32::from(buf[at + 1])) } else { buf[at] });
    }
    Probe::Alpha(Alpha { width: w, height: h, plane })
}

/// `not img.isNull() and img.hasAlphaChannel()`.
pub fn has_alpha_channel(data: &[u8]) -> bool {
    match probe_png(data) {
        Probe::Alpha(_) => true,
        Probe::Opaque | Probe::NotPng if data.starts_with(b"\x89PNG") => false,
        _ => {
            if data.starts_with(b"GIF8") && data.len() > 13 {
                // The first Graphic Control Extension: bit 0 of its flags is
                // "has a transparent colour".
                return data.windows(4).position(|w| w == [0x21, 0xf9, 0x04]).and_then(|i| data.get(i + 3)).is_some_and(|f| f & 1 == 1);
            }
            if data.len() > 30 && &data[..4] == b"RIFF" && &data[8..12] == b"WEBP" && &data[12..16] == b"VP8X" {
                return data[20] & 0x10 != 0;
            }
            false
        }
    }
}

/// `source index of destination pixel d` of `QImage.scaled(32, 32,
/// IgnoreAspectRatio, FastTransformation)`: the pixel under the centre of
/// the destination one, ties (an exact boundary) going to the lower pixel.
fn nearest(d: usize, src: usize) -> usize {
    ((2 * d + 1) * src).div_ceil(64).saturating_sub(1)
}

/// `core.texture.image_has_cutout`: some pixel of the image scaled to 32 x 32
/// is see-through (alpha under 32). An unreadable image reads as opaque.
pub fn image_has_cutout(data: &[u8]) -> bool {
    let Probe::Alpha(a) = probe_png(data) else { return false };
    if a.width == 0 || a.height == 0 {
        return false;
    }
    for y in 0..32 {
        let sy = nearest(y, a.height).min(a.height - 1);
        for x in 0..32 {
            let sx = nearest(x, a.width).min(a.width - 1);
            if a.plane[sy * a.width + sx] < 32 {
                return true;
            }
        }
    }
    false
}

// ---- the average colour: QImage.scaled(1, 1, IgnoreAspectRatio, SmoothTransformation) ----
//
// Qt 6's `qSmoothScaleImage` (imlib2's area sampling), determined against
// PySide6 6.11 (scripts/golden/make_qimage_fixtures.py holds the cases):
//
// * the weights of `s` source pixels into one are 14-bit fractions of 1:
//   `Cp = ceil(2^14 / s)` for the first and the middle pixels, the rest for
//   the last (`s = 1` gives the single weight 2^14);
// * a row is summed unshifted, `(row >> 4) * weight_y` is summed over the
//   rows, and the result is `>> 24` (so the sums keep 4 bits less than
//   exact: that is what makes a value just above an integer fall short);
// * an image with alpha is premultiplied first (`qPremultiply`) and the four
//   channels are scaled; the result is un-premultiplied by `convertToFormat`
//   (exact rounding, the ties of which Qt's SIMD code settles one way or the
//   other: [`TIES_UP`] lists the 74 `(alpha, channel)` pairs that go up) or,
//   for a palette with alpha, by `QImage.pixelColor` (`qUnpremultiply`).
//
// What is not reproduced: the premultiplied channels of a pixel can exceed
// its alpha only after the scaling (independent truncation); Qt then
// saturates in ways that are not modelled, and a handful of pixels (under
// 0.2 % of random images with very low alpha) come out a few levels off.
// 16-bit PNGs, JPEGs and the rest of Qt's loaders are not read.

/// `(alpha, premultiplied channel)` pairs whose exact `channel * 255 / alpha`
/// is a tie and that `QImage::convertToFormat(ARGB32_Premultiplied -> ARGB32)`
/// rounds up; every other tie goes down.
const TIES_UP: [(u8, u8); 74] = [
    (46, 23), (50, 5), (50, 25), (50, 45), (54, 27), (58, 29), (62, 31), (66, 33), (90, 9), (90, 21), (90, 33), (90, 45), (90, 57), (90, 69), (90, 81), (92, 46), (94, 47), (98, 49), (100, 10),
    (100, 50), (100, 90), (108, 54), (110, 11), (110, 55), (110, 99), (116, 58), (118, 59), (124, 62), (132, 66), (150, 95), (154, 77), (158, 79), (178, 89), (180, 18), (180, 42), (180, 66),
    (180, 90), (180, 114), (180, 138), (180, 162), (184, 92), (186, 93), (188, 94), (194, 97), (196, 98), (198, 99), (200, 20), (200, 100), (200, 180), (214, 107), (216, 108), (220, 22), (220, 110),
    (220, 198), (222, 111), (230, 23), (230, 115), (230, 207), (232, 116), (236, 118), (238, 7), (238, 35), (238, 63), (238, 91), (238, 147), (238, 175), (238, 203), (248, 124), (250, 25),
    (250, 75), (250, 125), (250, 175), (250, 225), (254, 127),
];

/// The weights of `s` pixels scaled into one (`qimageCalcApoints` and the
/// loops of `qt_qimageScaleAARGBA_down_xy`).
fn smooth_weights(s: usize) -> Vec<u64> {
    let cp = (1u64 << 14).div_ceil(s as u64);
    let mut ws = vec![cp];
    let mut j = (1u64 << 14).saturating_sub(cp);
    while j > cp {
        ws.push(cp);
        j -= cp;
    }
    if j > 0 {
        ws.push(j);
    }
    ws
}

/// One channel of the image scaled to a pixel: `plane` is `w * h` values.
fn smooth_channel(plane: &[u8], w: usize, h: usize) -> u8 {
    let (wx, wy) = (smooth_weights(w), smooth_weights(h));
    let mut tot = 0u64;
    for (y, &weight_y) in wy.iter().enumerate().take(h) {
        let mut rx = 0u64;
        for (x, &weight_x) in wx.iter().enumerate().take(w) {
            rx += u64::from(plane[y * w + x]) * weight_x;
        }
        tot += (rx >> 4) * weight_y;
    }
    (tot >> 24).min(255) as u8
}

/// The scaling of a gray or palette image of at most 2 x 2 pixels (not 1 x 1).
fn tiny_channel(plane: &[u8], w: usize, h: usize) -> u8 {
    let weights = |s: usize| -> Vec<u32> { if s == 2 { vec![129, 127] } else { vec![256] } };
    let (wx, wy) = (weights(w), weights(h));
    let mut tot = 0u32;
    for (x, &weight_x) in wx.iter().enumerate() {
        let column: u32 = wy.iter().enumerate().map(|(y, &weight_y)| u32::from(plane[y * w + x]) * weight_y).sum();
        tot += (column >> 8) * weight_x;
    }
    (tot >> 8).min(255) as u8
}

/// `qPremultiply` of one channel.
fn premultiply(c: u8, a: u8) -> u8 {
    let t = u32::from(c) * u32::from(a);
    ((t + (t >> 8) + 0x80) >> 8) as u8
}

/// `convertToFormat(ARGB32_Premultiplied -> ARGB32)` of one channel.
fn unpremultiply_convert(a: u8, p: u8) -> u8 {
    if a == 0 {
        return 0;
    }
    if a == 255 {
        return p;
    }
    if p > a {
        return 255;
    }
    let (num, den) = (u32::from(p) * 255, u32::from(a));
    let (q, r) = (num / den, num % den);
    if 2 * r > den || (2 * r == den && TIES_UP.contains(&(a, p))) {
        (q + 1) as u8
    } else {
        q as u8
    }
}

/// `qUnpremultiply` of one channel (what `QImage::pixelColor` does to a
/// premultiplied image).
fn unpremultiply_pixel(a: u8, p: u8) -> u8 {
    if a == 255 || a == 0 {
        return p;
    }
    let inv = 0x00ff_00ffu32 / u32::from(a);
    ((u32::from(p) * inv + 0x8000) >> 16).min(255) as u8
}

/// A decoded 8-bit image: what Qt's loader makes of a PNG, as the planes
/// the scaler reads.
enum Raster {
    /// Opaque: R, G, B planes; `rgb` when Qt loads it as RGB32 (a PNG of colour
    /// type RGB), else Grayscale8, Mono or Indexed8.
    Opaque { planes: [Vec<u8>; 3], rgb: bool },
    /// `Format_ARGB32`: R, G, B and alpha planes, not premultiplied.
    Argb([Vec<u8>; 4]),
    /// A palette with alpha (`Format_Indexed8`): the same planes, but Qt
    /// scales it premultiplied and `pixelColor` un-premultiplies.
    IndexedAlpha([Vec<u8>; 4]),
}

fn decode_raster(data: &[u8]) -> Result<(usize, usize, Raster), String> {
    if !data.starts_with(b"\x89PNG\r\n\x1a\n") {
        return Err("not ported: the average colour of an image that is not a PNG (Qt's JPEG and other decoders)".to_string());
    }
    let mut decoder = png::Decoder::new(Cursor::new(data));
    decoder.set_transformations(png::Transformations::EXPAND);
    let mut reader = decoder.read_info().map_err(|e| format!("not ported: unreadable PNG ({e})"))?;
    let info = reader.info();
    let palette = info.color_type == png::ColorType::Indexed;
    let has_trns = info.trns.is_some();
    let trns_alpha = info.trns.as_ref().is_some_and(|t| t.iter().any(|&a| a < 255));
    let size = reader.output_buffer_size().ok_or_else(|| "not ported: PNG too large".to_string())?;
    let mut buf = vec![0u8; size];
    let frame = reader.next_frame(&mut buf).map_err(|e| format!("not ported: unreadable PNG ({e})"))?;
    if frame.bit_depth == png::BitDepth::Sixteen {
        return Err("not ported: the average colour of a 16-bit PNG".to_string());
    }
    let (w, h) = (frame.width as usize, frame.height as usize);
    let n = w * h;
    let (channels, kind) = match frame.color_type {
        png::ColorType::Rgb => (3, 0),
        png::ColorType::Rgba => (4, 1),
        png::ColorType::Grayscale => (1, 0),
        png::ColorType::GrayscaleAlpha => (2, 1),
        png::ColorType::Indexed => return Err("not ported: unexpanded palette".to_string()),
    };
    let px = |i: usize, c: usize| buf[i * channels + c];
    let planes = |alpha: bool| -> Vec<Vec<u8>> {
        let mut p = vec![Vec::with_capacity(n); if alpha { 4 } else { 3 }];
        for i in 0..n {
            let (r, g, b) = if channels >= 3 { (px(i, 0), px(i, 1), px(i, 2)) } else { (px(i, 0), px(i, 0), px(i, 0)) };
            p[0].push(r);
            p[1].push(g);
            p[2].push(b);
            if alpha {
                p[3].push(if channels == 4 { px(i, 3) } else if channels == 2 { px(i, 1) } else { 255 });
            }
        }
        p
    };
    let raster = if palette && trns_alpha {
        let p = planes(true);
        Raster::IndexedAlpha(p.try_into().map_err(|_| "plane count".to_string())?)
    } else if kind == 1 || (!palette && has_trns) {
        // RGBA, gray with alpha, and a colour key (which EXPAND turned into
        // an alpha channel) are ARGB32 to Qt.
        let p = planes(true);
        Raster::Argb(p.try_into().map_err(|_| "plane count".to_string())?)
    } else {
        let p = planes(false);
        Raster::Opaque { planes: p.try_into().map_err(|_| "plane count".to_string())?, rgb: frame.color_type == png::ColorType::Rgb && !palette }
    };
    Ok((w, h, raster))
}

/// The premultiplied red, green and blue and the alpha of the image scaled
/// to a pixel: `(alpha, [r, g, b])`.
fn scale_premultiplied(p: &[Vec<u8>; 4], w: usize, h: usize) -> (u8, [u8; 3]) {
    let pm = |c: usize| -> Vec<u8> { p[c].iter().zip(&p[3]).map(|(&v, &a)| premultiply(v, a)).collect() };
    let a = smooth_channel(&p[3], w, h);
    (a, [0, 1, 2].map(|c| smooth_channel(&pm(c), w, h)))
}

/// `QImage(path).scaled(1, 1, IgnoreAspectRatio, SmoothTransformation)
/// .pixelColor(0, 0)` as 8-bit `[r, g, b]`, for a PNG.
pub fn smooth_average_rgb(data: &[u8]) -> Result<[u8; 3], String> {
    let (w, h, raster) = decode_raster(data)?;
    if w == 0 || h == 0 {
        return Err("not ported: an empty image".to_string());
    }
    Ok(match raster {
        // A pixel scaled to a pixel is the pixel itself.
        Raster::Opaque { planes, .. } if w == 1 && h == 1 => [planes[0][0], planes[1][0], planes[2][0]],
        // Grayscale8 and Indexed8 of two pixels or four have a path of their
        // own: 8-bit weights (129 and 127 of 256), columns first, each pass
        // floored.
        Raster::Opaque { planes, rgb: false } if w <= 2 && h <= 2 => planes.map(|p| tiny_channel(&p, w, h)),
        Raster::Opaque { planes, .. } => planes.map(|p| smooth_channel(&p, w, h)),
        // A pixel scaled to a pixel is the pixel itself.
        Raster::Argb(p) | Raster::IndexedAlpha(p) if w == 1 && h == 1 => [p[0][0], p[1][0], p[2][0]],
        Raster::Argb(p) => {
            let (a, pm) = scale_premultiplied(&p, w, h);
            pm.map(|v| unpremultiply_convert(a, v))
        }
        Raster::IndexedAlpha(p) => {
            let (a, pm) = scale_premultiplied(&p, w, h);
            pm.map(|v| unpremultiply_pixel(a, v))
        }
    })
}

/// [`smooth_average_rgb`] as `[c.redF(), c.greenF(), c.blueF()]`.
pub fn smooth_average(data: &[u8]) -> Result<[f64; 3], String> {
    // `QColor::redF()` is a float32: the 16-bit channel over 65535.
    smooth_average_rgb(data).map(|c| c.map(|v| f64::from(f32::from(u16::from(v) * 257) / 65535.0)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nearest_matches_qt_fast_scaling() {
        // Qt: width 64 samples 0, 2, 4...; width 128 samples 1, 5, 9...
        assert_eq!((0..4).map(|d| nearest(d, 64)).collect::<Vec<_>>(), [0, 2, 4, 6]);
        assert_eq!((0..4).map(|d| nearest(d, 128)).collect::<Vec<_>>(), [1, 5, 9, 13]);
        assert_eq!((0..3).map(|d| nearest(d, 192)).collect::<Vec<_>>(), [2, 8, 14]);
        assert_eq!(nearest(31, 512), 503);
        assert_eq!((0..4).map(|d| nearest(d, 4)).collect::<Vec<_>>(), [0, 0, 0, 0]);
    }
}
