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

/// The source pixel of destination pixel `d` of `n` when `QImage.scaled`
/// shrinks `src` pixels to `n` with `FastTransformation`: the pixel under
/// the centre of the destination one (see [`nearest`], which is `n = 32`).
pub fn nearest_n(d: usize, n: usize, src: usize) -> usize {
    ((2 * d + 1) * src).div_ceil(2 * n).saturating_sub(1)
}

/// What `QImage.fromData(data).convertToFormat(Format_ARGB32)` gives.
pub enum Decoded {
    /// Width, height and straight (not premultiplied) RGBA pixels.
    Image(usize, usize, Vec<[u8; 4]>),
    /// A PNG or JPEG that cannot be read: `QImage.isNull()`.
    Null,
    /// Another format (Qt reads GIF, BMP, WebP...): not decoded here.
    Unknown,
}

/// Decode a PNG or a JPEG. A JPEG is decoded with `zune-jpeg`, whose IDCT and
/// colour conversion may differ from libjpeg's by a level or two: fine for
/// the averages it is asked for.
pub fn decode_rgba8(data: &[u8]) -> Decoded {
    if data.starts_with(b"\x89PNG\r\n\x1a\n") {
        return match rgba8(data) {
            Some((w, h, px)) => Decoded::Image(w, h, px),
            None => Decoded::Null,
        };
    }
    if data.starts_with(&[0xFF, 0xD8, 0xFF]) {
        let mut decoder = zune_jpeg::JpegDecoder::new(zune_jpeg::zune_core::bytestream::ZCursor::new(data));
        let Ok(pixels) = decoder.decode() else { return Decoded::Null };
        let Some(info) = decoder.info() else { return Decoded::Null };
        let (w, h) = (info.width as usize, info.height as usize);
        if w == 0 || h == 0 || pixels.len() % (w * h) != 0 {
            return Decoded::Null;
        }
        let channels = pixels.len() / (w * h);
        let px = (0..w * h)
            .map(|i| match channels {
                1 => [pixels[i], pixels[i], pixels[i], 255],
                2 => [pixels[2 * i], pixels[2 * i], pixels[2 * i], pixels[2 * i + 1]],
                3 => [pixels[3 * i], pixels[3 * i + 1], pixels[3 * i + 2], 255],
                _ => [pixels[channels * i], pixels[channels * i + 1], pixels[channels * i + 2], 255],
            })
            .collect();
        return Decoded::Image(w, h, px);
    }
    Decoded::Unknown
}

/// A PNG as `QImage.convertToFormat(Format_ARGB32)` reads it: size and
/// straight (not premultiplied) RGBA pixels. `None` when the data is not a
/// readable PNG.
pub fn rgba8(data: &[u8]) -> Option<(usize, usize, Vec<[u8; 4]>)> {
    if !data.starts_with(b"\x89PNG\r\n\x1a\n") {
        return None;
    }
    let mut decoder = png::Decoder::new(Cursor::new(data));
    decoder.set_transformations(png::Transformations::EXPAND);
    let mut reader = decoder.read_info().ok()?;
    let mut buf = vec![0u8; reader.output_buffer_size()?];
    let frame = reader.next_frame(&mut buf).ok()?;
    let (w, h) = (frame.width as usize, frame.height as usize);
    let channels = match frame.color_type {
        png::ColorType::Rgba => 4,
        png::ColorType::Rgb => 3,
        png::ColorType::GrayscaleAlpha => 2,
        png::ColorType::Grayscale => 1,
        png::ColorType::Indexed => return None,
    };
    let bytes = if frame.bit_depth == png::BitDepth::Sixteen { 2 } else { 1 };
    let sample = |px: usize, c: usize| -> u8 {
        let at = (px * channels + c) * bytes;
        if bytes == 2 {
            div_257(u32::from(buf[at]) << 8 | u32::from(buf[at + 1]))
        } else {
            buf[at]
        }
    };
    let mut out = Vec::with_capacity(w * h);
    for px in 0..w * h {
        out.push(match channels {
            4 => [sample(px, 0), sample(px, 1), sample(px, 2), sample(px, 3)],
            3 => [sample(px, 0), sample(px, 1), sample(px, 2), 255],
            2 => [sample(px, 0), sample(px, 0), sample(px, 0), sample(px, 1)],
            _ => [sample(px, 0), sample(px, 0), sample(px, 0), 255],
        });
    }
    Some((w, h, out))
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
