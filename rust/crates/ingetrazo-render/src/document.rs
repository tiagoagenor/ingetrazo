// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! Opening an .igz for the renderer: the scene, plus a [`TextureSource`] that
//! reads the images embedded in the container.

use ingetrazo_formats::igz::{self, Zip};
use ingetrazo_formats::igz_geom::expand_document;
use ingetrazo_model::scene::Scene;

use crate::textures::{TextureImage, TextureSource};

/// Textures embedded in a document: archive members by name.
pub struct IgzTextures {
    raw: Vec<u8>,
}

impl IgzTextures {
    pub fn new(raw: Vec<u8>) -> IgzTextures {
        IgzTextures { raw }
    }
}

impl TextureSource for IgzTextures {
    fn load(&mut self, path: &str) -> Option<TextureImage> {
        if !igz::is_container(&self.raw) {
            return None;
        }
        let zip = Zip::open(&self.raw).ok()?;
        let bytes = zip.read(path).ok()?;
        decode_image(&bytes)
    }
}

/// Decode a PNG or a JPEG (by its first bytes) into RGBA8: the formats the
/// documents and the materials library carry.
pub fn decode_image(bytes: &[u8]) -> Option<TextureImage> {
    if bytes.starts_with(&[0x89, b'P', b'N', b'G']) {
        decode_png(bytes)
    } else if bytes.starts_with(&[0xff, 0xd8]) {
        decode_jpeg(bytes)
    } else {
        None
    }
}

/// Decode a JPEG into RGBA8.
pub fn decode_jpeg(bytes: &[u8]) -> Option<TextureImage> {
    use zune_core::colorspace::ColorSpace;
    use zune_core::options::DecoderOptions;
    let opts = DecoderOptions::default().jpeg_set_out_colorspace(ColorSpace::RGBA);
    let mut d = zune_jpeg::JpegDecoder::new_with_options(std::io::Cursor::new(bytes), opts);
    let rgba = d.decode().ok()?;
    let info = d.info()?;
    let (w, h) = (info.width as u32, info.height as u32);
    (rgba.len() == (w * h * 4) as usize).then(|| TextureImage::new(w, h, rgba))
}

/// Decode a PNG into RGBA8.
pub fn decode_png(bytes: &[u8]) -> Option<TextureImage> {
    let mut reader = png::Decoder::new(std::io::Cursor::new(bytes)).read_info().ok()?;
    let mut buf = vec![0; reader.output_buffer_size()?];
    let info = reader.next_frame(&mut buf).ok()?;
    buf.truncate(info.buffer_size());
    let (w, h) = (info.width, info.height);
    let rgba: Vec<u8> = match (info.color_type, info.bit_depth) {
        (png::ColorType::Rgba, png::BitDepth::Eight) => buf,
        (png::ColorType::Rgb, png::BitDepth::Eight) => buf.chunks_exact(3).flat_map(|p| [p[0], p[1], p[2], 255]).collect(),
        (png::ColorType::Grayscale, png::BitDepth::Eight) => buf.iter().flat_map(|&g| [g, g, g, 255]).collect(),
        (png::ColorType::GrayscaleAlpha, png::BitDepth::Eight) => {
            buf.chunks_exact(2).flat_map(|p| [p[0], p[0], p[0], p[1]]).collect()
        }
        _ => return None,
    };
    Some(TextureImage::new(w, h, rgba))
}

/// Open an .igz file: the scene and its embedded textures.
pub fn open_igz(raw: &[u8]) -> Result<(Scene, IgzTextures), String> {
    let doc = igz::read_document(raw).map_err(|e| e.to_string())?;
    let zip = if igz::is_container(raw) { Some(Zip::open(raw).map_err(|e| e.to_string())?) } else { None };
    let doc = expand_document(&doc, zip.as_ref()).map_err(|e| e.to_string())?;
    let scene = Scene::from_document(&doc).map_err(|e| format!("{e:?}"))?;
    Ok((scene, IgzTextures::new(raw.to_vec())))
}
