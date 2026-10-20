// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! Texture images for the renderer. The renderer never reads files: the
//! application hands images over through a [`TextureSource`] (documents keep
//! them as archive members, the library as files on disk), the way
//! `Viewport._get_texture` reads `QImage(path)`.

use std::collections::HashMap;

/// A decoded image, RGBA8, rows top to bottom.
#[derive(Clone, Debug)]
pub struct TextureImage {
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<u8>,
}

impl TextureImage {
    pub fn new(width: u32, height: u32, rgba: Vec<u8>) -> TextureImage {
        assert_eq!(rgba.len(), (width * height * 4) as usize);
        TextureImage { width, height, rgba }
    }

    fn px(&self, x: u32, y: u32) -> [u8; 4] {
        let i = ((y * self.width + x) * 4) as usize;
        [self.rgba[i], self.rgba[i + 1], self.rgba[i + 2], self.rgba[i + 3]]
    }

    /// Nearest-neighbour sample on an n x n grid (the `scaled(32, 32,
    /// FastTransformation)` of `image_has_cutout`).
    fn grid(&self, n: u32) -> impl Iterator<Item = [u8; 4]> + '_ {
        (0..n).flat_map(move |j| {
            (0..n).map(move |i| {
                let x = ((i as f32 + 0.5) * self.width as f32 / n as f32) as u32;
                let y = ((j as f32 + 0.5) * self.height as f32 / n as f32) as u32;
                self.px(x.min(self.width - 1), y.min(self.height - 1))
            })
        })
    }
}

/// Where the renderer asks for images, by the path a face's texture names.
pub trait TextureSource {
    fn load(&mut self, path: &str) -> Option<TextureImage>;
}

/// A source with no images: textured faces draw nothing in the texture
/// styles and in their average grey in Shaded.
pub struct NoTextures;

impl TextureSource for NoTextures {
    fn load(&mut self, _path: &str) -> Option<TextureImage> {
        None
    }
}

impl TextureSource for HashMap<String, TextureImage> {
    fn load(&mut self, path: &str) -> Option<TextureImage> {
        self.get(path).cloned()
    }
}

/// What the renderer keeps per texture path.
pub struct TexInfo {
    pub image: Option<TextureImage>,
    /// `image_has_cutout`: some pixels see-through (alpha < 32 on a 32x32
    /// grid) - the material counts as translucent.
    pub has_cutout: bool,
    /// `_get_texture`'s `_cutout`: under 60 % of a 64x64 grid is opaque -
    /// foliage; sampled without mips and cut hard at 0.5.
    pub hard_cutout: bool,
    /// `_texture_avg_color`: the colour of the Shaded style.
    pub average: [f32; 3],
    pub uploaded: bool,
}

impl TexInfo {
    pub fn from_image(image: Option<TextureImage>) -> TexInfo {
        match image {
            None => TexInfo {
                image: None,
                has_cutout: false,
                hard_cutout: false,
                average: [0.78, 0.78, 0.78],
                uploaded: false,
            },
            Some(img) => {
                let has_alpha = img.rgba.chunks_exact(4).any(|p| p[3] != 255);
                let has_cutout = has_alpha && img.grid(32).any(|p| p[3] < 32);
                let hard_cutout = has_alpha && {
                    let n = 64u32;
                    let opaque = img.grid(n).filter(|p| p[3] > 128).count();
                    (opaque as f32 / (n * n) as f32) < 0.6
                };
                let average = average_color(&img);
                TexInfo { image: Some(img), has_cutout, hard_cutout, average, uploaded: false }
            }
        }
    }
}

/// The mean of a 4x4 box-reduced image, as `scaled(4, 4, Smooth)` averages.
fn average_color(img: &TextureImage) -> [f32; 3] {
    let n = 4u32;
    let (mut r, mut g, mut b, mut cnt) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for j in 0..n {
        for i in 0..n {
            let x0 = i * img.width / n;
            let x1 = (((i + 1) * img.width) / n).max(x0 + 1).min(img.width);
            let y0 = j * img.height / n;
            let y1 = (((j + 1) * img.height) / n).max(y0 + 1).min(img.height);
            let (mut sr, mut sg, mut sb, mut c) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
            for y in y0..y1 {
                for x in x0..x1 {
                    let p = img.px(x.min(img.width - 1), y.min(img.height - 1));
                    sr += p[0] as f64;
                    sg += p[1] as f64;
                    sb += p[2] as f64;
                    c += 1.0;
                }
            }
            if c > 0.0 {
                r += (sr / c).round();
                g += (sg / c).round();
                b += (sb / c).round();
                cnt += 1.0;
            }
        }
    }
    if cnt == 0.0 {
        return [0.78, 0.78, 0.78];
    }
    [(r / (255.0 * cnt)) as f32, (g / (255.0 * cnt)) as f32, (b / (255.0 * cnt)) as f32]
}

/// The mip chain of an image, rows flipped so row 0 is the picture's bottom
/// (`QOpenGLTexture(img.mirrored())`: OBJ/.skp V runs bottom-up). Level 0
/// first; each level a 2x2 box average.
pub fn mip_chain(img: &TextureImage) -> Vec<(u32, u32, Vec<u8>)> {
    let (w, h) = (img.width, img.height);
    let mut base = vec![0u8; img.rgba.len()];
    let row = (w * 4) as usize;
    for y in 0..h as usize {
        let src = (h as usize - 1 - y) * row;
        base[y * row..(y + 1) * row].copy_from_slice(&img.rgba[src..src + row]);
    }
    let mut out = vec![(w, h, base)];
    while out.last().map(|l| l.0 > 1 || l.1 > 1).unwrap_or(false) {
        let (pw, ph, prev) = out.last().unwrap();
        let (nw, nh) = ((*pw / 2).max(1), (*ph / 2).max(1));
        let mut next = vec![0u8; (nw * nh * 4) as usize];
        for y in 0..nh {
            for x in 0..nw {
                for c in 0..4usize {
                    let mut s = 0u32;
                    for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                        let sx = (x * 2 + dx).min(pw - 1);
                        let sy = (y * 2 + dy).min(ph - 1);
                        s += prev[((sy * pw + sx) * 4) as usize + c] as u32;
                    }
                    next[((y * nw + x) * 4) as usize + c] = ((s + 2) / 4) as u8;
                }
            }
        }
        out.push((nw, nh, next));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mip_chain_ends_at_one_pixel_and_flips_rows() {
        // 2x2: top row red, bottom row blue.
        let img = TextureImage::new(
            2,
            2,
            vec![255, 0, 0, 255, 255, 0, 0, 255, 0, 0, 255, 255, 0, 0, 255, 255],
        );
        let chain = mip_chain(&img);
        assert_eq!(chain.len(), 2);
        assert_eq!(&chain[0].2[..4], &[0, 0, 255, 255], "row 0 is the picture's bottom");
        assert_eq!(chain[1].2.len(), 4);
    }

    #[test]
    fn cutout_detection_follows_the_alpha() {
        let opaque = TexInfo::from_image(Some(TextureImage::new(2, 2, vec![255; 16])));
        assert!(!opaque.has_cutout && !opaque.hard_cutout);
        let mut px = vec![255u8; 16 * 16 * 4];
        for p in px.chunks_exact_mut(4).take(200) {
            p[3] = 0;
        }
        let leaf = TexInfo::from_image(Some(TextureImage::new(16, 16, px)));
        assert!(leaf.has_cutout);
        assert!((opaque.average[0] - 1.0).abs() < 1e-6);
    }
}
