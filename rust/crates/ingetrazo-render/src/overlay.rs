// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! Geometry the viewport builds every frame, outside the model: the sky and
//! ground backdrop, the drawing axes, the guides and the selection cues.
//! Pure CPU code (ports of `_draw_sky`, `_axes_vertices`, `_guide_vertices`).

use bytemuck::{Pod, Zeroable};
use ingetrazo_math::{M4, V3};
use ingetrazo_model::guide::Guide;

use crate::bake::EdgeVertex;
use crate::camera::Camera;
use crate::style::{Rgb, Style};

/// A backdrop vertex: NDC position and colour.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct SkyVertex {
    pub pos: [f32; 2],
    pub color: [f32; 3],
}

const SKY_GRAD_SPAN: f32 = 1.4;
const GROUND_GRAD_SPAN: f32 = 0.9;

/// `_horizon_ndc_y`: NDC y of the horizon (the ground plane at infinity). May
/// exceed +-1 when the horizon is off screen.
pub fn horizon_ndc_y(camera: &Camera, mvp: &M4) -> f32 {
    let eye = camera.eye();
    let fwd = camera.target.sub(eye);
    let dh = V3(fwd.0, fwd.1, 0.0);
    if dh.length(camera.hypot) < 1e-5 {
        return if fwd.2 < 0.0 { 2.0 } else { -2.0 };
    }
    let dh = dh.normalized(camera.hypot);
    let far = eye.add(dh.mul(1.0e6));
    let d = mvp.data();
    let y = d[1] * far.0 + d[5] * far.1 + d[9] * far.2 + d[13];
    let w = d[3] * far.0 + d[7] * far.1 + d[11] * far.2 + d[15];
    if w.abs() < 1e-9 {
        return if fwd.2 < 0.0 { 2.0 } else { -2.0 };
    }
    y / w
}

fn mix(a: Rgb, b: Rgb, t: f32) -> Rgb {
    [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t]
}

/// `_sky_gradient`: `(sky_bottom, sky_top, ground_top, ground_bottom, line)`.
pub fn sky_gradient(sky: Rgb, ground: Rgb, hy: f32) -> (Rgb, Rgb, Rgb, Rgb, Rgb) {
    let white = [1.0, 1.0, 1.0];
    let sky_h = mix(sky, white, 0.55);
    let gnd_h = mix(ground, white, 0.30);
    let sky_at = |y: f32| mix(sky_h, sky, ((y - hy) / SKY_GRAD_SPAN).clamp(0.0, 1.0));
    let gnd_at = |y: f32| mix(gnd_h, ground, ((hy - y) / GROUND_GRAD_SPAN).clamp(0.0, 1.0));
    let y0 = hy.clamp(-1.0, 1.0);
    let line = mix(mix(sky_h, gnd_h, 0.5), [0.0; 3], 0.12);
    (sky_at(y0), sky_at(1.0), gnd_at(y0), gnd_at(-1.0), line)
}

/// The backdrop: gradient triangles and the horizon line.
pub struct Sky {
    pub tris: Vec<SkyVertex>,
    pub line: Vec<SkyVertex>,
}

/// `_draw_sky`.
pub fn sky(style: &Style, camera: &Camera, mvp: &M4) -> Sky {
    let hy_raw = horizon_ndc_y(camera, mvp);
    let hy = hy_raw.clamp(-1.0, 1.0);
    let (c_sky0, c_sky1, c_gnd0, c_gnd1, c_line) = sky_gradient(style.sky_color, style.ground_color, hy_raw);
    let mut tris = Vec::new();
    let mut quad = |y0: f32, y1: f32, c0: Rgb, c1: Rgb| {
        let v = |x: f32, y: f32, c: Rgb| SkyVertex { pos: [x, y], color: c };
        tris.extend([
            v(-1.0, y0, c0),
            v(1.0, y0, c0),
            v(1.0, y1, c1),
            v(-1.0, y0, c0),
            v(1.0, y1, c1),
            v(-1.0, y1, c1),
        ]);
    };
    if hy < 1.0 {
        quad(hy, 1.0, c_sky0, c_sky1);
    }
    if hy > -1.0 {
        quad(-1.0, hy, c_gnd1, c_gnd0);
    }
    let mut line = Vec::new();
    if -1.0 < hy && hy < 1.0 {
        line.push(SkyVertex { pos: [-1.0, hy], color: c_line });
        line.push(SkyVertex { pos: [1.0, hy], color: c_line });
    }
    Sky { tris, line }
}

fn seg(a: [f32; 3], b: [f32; 3]) -> [EdgeVertex; 2] {
    [EdgeVertex { pos: a, anchor: b }, EdgeVertex { pos: b, anchor: b }]
}

/// The axis colours of `paintGL`.
pub const AXIS_COLORS: [Rgb; 3] = [[0.86, 0.22, 0.27], [0.16, 0.62, 0.36], [0.20, 0.40, 0.78]];

/// `_axes_vertices`: per axis a long solid positive line and evenly spaced
/// dashes on the negative side. `origin` is the drawing axes' origin.
pub fn axes(spacing: f32, origin: V3) -> [Vec<EdgeVertex>; 3] {
    let spacing = spacing.max(1e-4);
    let dash = spacing * 0.18;
    let pos_len = 1.0e5f32;
    let dirs = [[1.0f32, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    let mut out: [Vec<EdgeVertex>; 3] = Default::default();
    for (k, d) in dirs.iter().enumerate() {
        let o = [origin.0, origin.1, origin.2];
        let at = |t: f32| [o[0] + d[0] * t, o[1] + d[1] * t, o[2] + d[2] * t];
        out[k].extend(seg(o, at(pos_len)));
        for i in 0..520 {
            let t0 = i as f32 * spacing;
            out[k].extend(seg(at(-t0), at(-(t0 + dash))));
        }
    }
    out
}

/// `_guide_vertices`: guides as world-space dashes. Returns one run of
/// segments per guide with its selection flag.
pub fn guides(
    guides: &[Guide],
    spacing: f32,
    selected: &[usize],
    centre: V3,
    reach: f32,
) -> Vec<(Vec<EdgeVertex>, bool)> {
    let spacing = spacing.max(1e-6);
    let reach = if reach > 0.0 { reach } else { spacing * 500.0 };
    let dash = spacing * 0.5;
    let n = ((2.0 * reach / spacing) as i64).clamp(1, 4000);
    let mut out = Vec::new();
    for (gi, g) in guides.iter().enumerate() {
        let sel = selected.contains(&gi);
        let mut v = Vec::new();
        match g.direction {
            None => {
                let (Some(o), p) = (g.origin, g.point) else { continue };
                let d = p.sub(o);
                let length = (d.0 * d.0 + d.1 * d.1 + d.2 * d.2).sqrt();
                if length < 1e-9 {
                    continue;
                }
                let u = V3(d.0 / length, d.1 / length, d.2 / length);
                let mut t = 0.0f32;
                while t < length {
                    let a = o.add(u.mul(t));
                    let b = o.add(u.mul((t + dash).min(length)));
                    v.extend(seg([a.0, a.1, a.2], [b.0, b.1, b.2]));
                    t += spacing;
                }
            }
            Some(d) => {
                let t0 = centre.sub(g.point).dot(d) - reach;
                for k in 0..n {
                    let s = t0 + k as f32 * spacing;
                    let a = g.point.add(d.mul(s));
                    let b = g.point.add(d.mul(s + dash));
                    v.extend(seg([a.0, a.1, a.2], [b.0, b.1, b.2]));
                }
            }
        }
        if !v.is_empty() {
            out.push((v, sel));
        }
    }
    out
}

/// The twelve edges of a box given its eight corners in
/// `oriented_box_corners` order (neighbours differ in one bit).
pub fn box_edges(corners: &[V3; 8]) -> Vec<EdgeVertex> {
    let mut v = Vec::new();
    for i in 0..8usize {
        for bit in [1usize, 2, 4] {
            let j = i | bit;
            if j != i {
                let (a, b) = (corners[i], corners[j]);
                v.extend(seg([a.0, a.1, a.2], [b.0, b.1, b.2]));
            }
        }
    }
    v
}

/// The eight corners of an axis-aligned box, in that bit order.
pub fn aabb_corners(lo: [f32; 3], hi: [f32; 3]) -> [V3; 8] {
    let mut c = [V3(0.0, 0.0, 0.0); 8];
    for (i, p) in c.iter_mut().enumerate() {
        *p = V3(
            if i & 1 == 0 { lo[0] } else { hi[0] },
            if i & 2 == 0 { lo[1] } else { hi[1] },
            if i & 4 == 0 { lo[2] } else { hi[2] },
        );
    }
    c
}

/// `_shifted_mvp`: the matrix nudged by a clip-space offset (sub-pixel line
/// thickening for exports), `T(dx, dy, 0) * mvp`.
pub fn shifted_mvp(mvp: &[f32; 16], dx: f32, dy: f32) -> [f32; 16] {
    let mut m = *mvp;
    for col in 0..4 {
        let w = m[col * 4 + 3];
        m[col * 4] += dx * w;
        m[col * 4 + 1] += dy * w;
    }
    m
}

/// `_line_jitter`: NDC offsets that thicken a 1 px line to about `px` pixels
/// by redrawing it shifted. Exports only.
pub fn line_jitter(px: u32, w: u32, h: u32) -> Vec<(f32, f32)> {
    if px <= 1 {
        return vec![(0.0, 0.0)];
    }
    let px = px.min(8);
    let r = (px - 1) as f32 / 2.0;
    let offs: Vec<f32> = (0..px).map(|i| -r + i as f32).collect();
    let mut out = Vec::new();
    for &dx in &offs {
        for &dy in &offs {
            out.push((dx * 2.0 / w.max(1) as f32, dy * 2.0 / h.max(1) as f32));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn horizon_is_on_screen_for_a_level_camera() {
        let c = Camera { pitch: 0.0, ..Default::default() };
        let y = horizon_ndc_y(&c, &c.mvp());
        assert!(y.abs() < 1e-3, "{y}");
        let down = Camera { pitch: 80f64.to_radians(), ..Default::default() };
        assert!(horizon_ndc_y(&down, &down.mvp()) > 1.0);
    }

    #[test]
    fn sky_covers_the_whole_screen() {
        let c = Camera { pitch: 10f64.to_radians(), ..Default::default() };
        let s = sky(&Style::default(), &c, &c.mvp());
        assert!(s.tris.len() == 12 && s.line.len() == 2);
        let top = Camera { pitch: 89f64.to_radians(), ..Default::default() };
        let s = sky(&Style::default(), &top, &top.mvp());
        assert_eq!(s.tris.len(), 6, "looking down: all ground");
    }

    #[test]
    fn axes_have_a_solid_positive_and_dashed_negative_half() {
        let a = axes(0.5, V3(0.0, 0.0, 0.0));
        assert_eq!(a[0].len(), 2 + 520 * 2);
        assert_eq!(a[2][0].pos, [0.0, 0.0, 0.0]);
        assert_eq!(a[2][1].pos, [0.0, 0.0, 1.0e5]);
    }

    #[test]
    fn guides_dash_around_the_camera_target() {
        let g = Guide::new(V3(0.0, 0.0, 0.0), Some(V3(1.0, 0.0, 0.0)), None, Default::default());
        let r = guides(&[g], 0.5, &[0], V3(10.0, 0.0, 0.0), 5.0);
        assert_eq!(r.len(), 1);
        assert!(r[0].1);
        let xs: Vec<f32> = r[0].0.iter().map(|v| v.pos[0]).collect();
        assert!(xs.iter().all(|&x| (4.9..=15.6).contains(&x)), "{xs:?}");
    }

    #[test]
    fn jitter_offsets_form_a_pixel_disc() {
        assert_eq!(line_jitter(1, 100, 100), vec![(0.0, 0.0)]);
        assert_eq!(line_jitter(2, 100, 100).len(), 4);
        assert_eq!(line_jitter(3, 100, 100).len(), 9);
    }
}
