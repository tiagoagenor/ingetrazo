// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! Camera navigation (docs/rust-rewrite-plan.md, M4.4): the maths of
//! `OrbitCamera.orbit / orbit_about / pan / zoom / zoom_to / fit_box` and the
//! gesture state machine of `Viewport.mousePressEvent / mouseMoveEvent /
//! mouseReleaseEvent / wheelEvent`, operating on the renderer's
//! [`Camera`] (which already is the port of `OrbitCamera`'s pose and
//! matrices).
//!
//! Qt's float32 vectors stay float32 (`V3`), Python's floats stay f64, in
//! the same operation order, so the camera after a gesture is the Python
//! camera bit for bit (`tests/nav_parity.rs`).
//!
//! What the camera turns about or zooms towards comes from the scene
//! (`Viewport._orbit_pivot_at`, `_world_under_cursor`): the [`Probe`] trait,
//! implemented over the pick index in [`crate::probe`].

use ingetrazo_math::pyfloat::{py_max2, py_max3, py_min2};
use ingetrazo_math::{M4, V3};
use ingetrazo_render::camera::{MAX_DISTANCE, MIN_DISTANCE};
use ingetrazo_render::Camera;
use ingetrazo_snap::camera::map_v4;

use crate::quat::Quat;

/// The wheel zoom factor per notch (`OrbitCamera.zoom`).
pub const ZOOM_BASE: f64 = 0.9;
/// `Viewport.mouseMoveEvent`: the Zoom nav mode, steps per dragged pixel.
pub const ZOOM_DRAG_STEPS: f64 = 0.035;
/// How far from the cached pin the cursor may stray and still reuse it.
const FOCUS_REUSE_PX: f64 = 24.0;

/// `math.radians(89.0)`: the pitch limit.
fn pitch_limit() -> f64 {
    89.0f64.to_radians()
}

fn finite3(v: V3) -> bool {
    v.0.is_finite() && v.1.is_finite() && v.2.is_finite()
}

/// `OrbitCamera.is_finite`: every number of the pose is a real number.
pub fn is_finite(cam: &Camera) -> bool {
    finite3(cam.target) && cam.distance.is_finite() && cam.yaw.is_finite() && cam.pitch.is_finite()
}

/// `OrbitCamera.orbit`: turn the model under the cursor, Blender style.
/// `viewport_h` is the height of the viewport in pixels; callers that
/// invert the vertical axis (Preferences) negate `dy`.
pub fn orbit(cam: &mut Camera, dx: f64, dy: f64, viewport_h: i32) {
    let scale = std::f64::consts::PI / viewport_h.max(1) as f64;
    cam.yaw -= dx * scale;
    cam.pitch = py_max2(py_min2(cam.pitch + dy * scale, pitch_limit()), (-89.0f64).to_radians());
}

/// `OrbitCamera.orbit_about`: [`orbit`] around `pivot` instead of the target;
/// the whole camera (eye and target) turns rigidly about it.
pub fn orbit_about(cam: &mut Camera, pivot: V3, dx: f64, dy: f64, viewport_h: i32) {
    if !finite3(pivot) {
        orbit(cam, dx, dy, viewport_h);
        return;
    }
    let (old_yaw, old_pitch) = (cam.yaw, cam.pitch);
    let old_target = cam.target;
    orbit(cam, dx, dy, viewport_h);
    let d_yaw = cam.yaw - old_yaw;
    let d_pitch = cam.pitch - old_pitch;
    if d_yaw.abs() < 1e-12 && d_pitch.abs() < 1e-12 {
        return;
    }
    let h = cam.hypot;
    let z = V3(0.0, 0.0, 1.0);
    let rot_yaw = Quat::from_axis_and_angle(z, d_yaw.to_degrees() as f32, h);
    let cp = old_pitch.cos();
    let offset = V3((cp * cam.yaw.cos()) as f32, (cp * cam.yaw.sin()) as f32, old_pitch.sin() as f32);
    let axis = offset.cross(z);
    let rot = if (axis.length(h) as f64) < 1e-9 {
        rot_yaw
    } else {
        Quat::from_axis_and_angle(axis.normalized(h), d_pitch.to_degrees() as f32, h).mul(rot_yaw)
    };
    cam.target = pivot.add(rot.rotated_vector(cam.target.sub(pivot)));
    if !is_finite(cam) {
        cam.yaw = old_yaw;
        cam.pitch = old_pitch;
        cam.target = old_target;
    }
}

/// `OrbitCamera.pan`: slide the view by a drag of `(dx, dy)` pixels. `depth`
/// is how far in front of the eye the grabbed point lies (perspective only).
pub fn pan(cam: &mut Camera, dx: f64, dy: f64, viewport_h: i32, depth: Option<f64>) {
    let h = cam.hypot;
    let (cp, sp) = (cam.pitch.cos(), cam.pitch.sin());
    let (cy, sy) = (cam.yaw.cos(), cam.yaw.sin());
    let forward = V3((-cp * cy) as f32, (-cp * sy) as f32, (-sp) as f32);
    let right = forward.cross(cam.up_vector()).normalized(h);
    let screen_up = right.cross(forward).normalized(h);
    let mut span = cam.distance;
    if let Some(d) = depth {
        if cam.perspective && d.is_finite() && d > 1e-6 {
            span = d;
        }
    }
    let world_per_pixel = 2.0 * span * (cam.fov_deg.to_radians() / 2.0).tan() / viewport_h.max(1) as f64;
    cam.target = cam.target.sub(right.mul((dx * world_per_pixel) as f32));
    cam.target = cam.target.add(screen_up.mul((dy * world_per_pixel) as f32));
}

/// `OrbitCamera.zoom`: `steps` wheel notches, towards the target.
pub fn zoom(cam: &mut Camera, steps: f64) {
    let factor = ZOOM_BASE.powf(steps);
    cam.distance = py_max2(MIN_DISTANCE, py_min2(cam.distance * factor, MAX_DISTANCE));
}

/// `OrbitCamera.zoom_to`: zoom keeping the world point `focus` (under the
/// cursor) fixed on screen. `min_step` is the least a zoom OUT retreats.
pub fn zoom_to(cam: &mut Camera, steps: f64, focus: V3, min_step: f64) {
    let h = cam.hypot;
    let eye_before = cam.eye();
    if steps < 0.0 && eye_before.sub(focus).length(h) as f64 >= MAX_DISTANCE {
        zoom(cam, steps);
        return;
    }
    let mut factor = ZOOM_BASE.powf(steps);
    let new_distance = py_max2(MIN_DISTANCE, py_min2(cam.distance * factor, MAX_DISTANCE));
    if steps < 0.0 && cam.distance > 0.0 {
        factor = py_min2(factor, new_distance / cam.distance);
    }
    let before = (cam.target, cam.distance);
    cam.target = focus.add(cam.target.sub(focus).mul(factor as f32));
    cam.distance = new_distance;
    if steps < 0.0 && min_step > 0.0 && new_distance < MAX_DISTANCE {
        let moved = cam.eye().sub(eye_before).length(h) as f64;
        if moved < min_step {
            let back = cam.eye().sub(cam.target);
            if back.length(h) as f64 > 1e-9 {
                cam.target = cam.target.add(back.normalized(h).mul((min_step - moved) as f32));
            }
        }
    }
    if !is_finite(cam) {
        cam.target = before.0;
        cam.distance = before.1;
    }
}

/// `OrbitCamera.toggle_projection`.
pub fn toggle_projection(cam: &mut Camera) {
    cam.perspective = !cam.perspective;
    cam.two_point = false;
}

/// `OrbitCamera.toggle_two_point`: turning it on also turns a parallel view
/// back into a perspective.
pub fn toggle_two_point(cam: &mut Camera) {
    cam.two_point = !cam.two_point;
    if cam.two_point {
        cam.perspective = true;
    }
}

/// `OrbitCamera.set_projection`: "parallel", "perspective" or "two_point"
/// (anything else is ignored).
pub fn set_projection(cam: &mut Camera, mode: &str) {
    if !matches!(mode, "parallel" | "perspective" | "two_point") {
        return;
    }
    cam.perspective = mode != "parallel";
    cam.two_point = mode == "two_point";
}

/// `OrbitCamera.set_view`: the standard architectural views (Z up); an
/// unknown name is ignored.
pub fn set_view(cam: &mut Camera, name: &str) {
    let (yaw, pitch) = match name {
        "top" => ((-90.0f64).to_radians(), 90.0f64.to_radians()),
        "bottom" => ((-90.0f64).to_radians(), (-90.0f64).to_radians()),
        "front" => ((-90.0f64).to_radians(), 0.0),
        "back" => (90.0f64.to_radians(), 0.0),
        "right" => (0.0, 0.0),
        "left" => (180.0f64.to_radians(), 0.0),
        "iso" => ((-45.0f64).to_radians(), 30.0f64.to_radians()),
        _ => return,
    };
    cam.yaw = yaw;
    cam.pitch = pitch;
}

/// The default margin of `OrbitCamera.fit_box`.
pub const FIT_MARGIN: f64 = 1.1;

/// `OrbitCamera.fit_box`: Zoom Extents. Frames the box as the camera looks
/// at it now, so the model fills the window in this orientation and for
/// this window shape.
pub fn fit_box(cam: &mut Camera, lo: V3, hi: V3, margin: f64) {
    let h = cam.hypot;
    let f = cam.forward();
    let r = f.cross(cam.up_vector());
    if (r.length_squared() as f64) < 1e-18 {
        cam.fit_to(lo, hi, 1.3);
        return;
    }
    let r = r.normalized(h);
    let u = r.cross(f);
    let corners: Vec<V3> = [lo.0, hi.0]
        .iter()
        .flat_map(|&x| [lo.1, hi.1].into_iter().flat_map(move |y| [lo.2, hi.2].into_iter().map(move |z| V3(x, y, z))))
        .collect();
    let center = lo.add(hi).mul(0.5);
    let pr: Vec<f64> = corners.iter().map(|c| c.sub(center).dot(r) as f64).collect();
    let pu: Vec<f64> = corners.iter().map(|c| c.sub(center).dot(u) as f64).collect();
    let (pr_max, pr_min) = (max_of(&pr), min_of(&pr));
    let (pu_max, pu_min) = (max_of(&pu), min_of(&pu));
    let target = center
        .add(r.mul(((pr_max + pr_min) * 0.5) as f32))
        .add(u.mul(((pu_max + pu_min) * 0.5) as f32));
    let half_w = (pr_max - pr_min) * 0.5;
    let half_h = (pu_max - pu_min) * 0.5;
    let t = (cam.fov_deg.to_radians() / 2.0).tan();
    let aspect = if cam.aspect > 1e-6 { cam.aspect } else { 1.0 };
    let half = py_max3(half_h, half_w / aspect, MIN_DISTANCE) * margin;
    let mut distance = half / t;
    if cam.perspective {
        let toward = corners.iter().map(|c| -(c.sub(target).dot(f)) as f64).fold(f64::NEG_INFINITY, py_max2);
        distance += py_max2(toward, 0.0);
    }
    cam.target = target;
    cam.distance = py_min2(py_max2(distance, MIN_DISTANCE), MAX_DISTANCE);
    if cam.perspective {
        settle_perspective_fit(cam, &corners, margin);
    }
}

fn max_of(xs: &[f64]) -> f64 {
    xs.iter().copied().fold(f64::NEG_INFINITY, py_max2)
}

fn min_of(xs: &[f64]) -> f64 {
    xs.iter().copied().fold(f64::INFINITY, py_min2)
}

/// `OrbitCamera._settle_perspective_fit`: measure the real projection, slide
/// the target so the box is centred on screen, then walk the distance in
/// until its wider half-span sits at `1 / margin` of the window.
fn settle_perspective_fit(cam: &mut Camera, corners: &[V3], margin: f64) {
    let h = cam.hypot;
    let goal = 1.0 / margin;
    let t = (cam.fov_deg.to_radians() / 2.0).tan();
    for _ in 0..8 {
        let mvp: M4 = cam.projection_matrix().mul(&cam.view_matrix());
        let mut xs = Vec::with_capacity(corners.len());
        let mut ys = Vec::with_capacity(corners.len());
        for c in corners {
            let p = map_v4(&mvp, [c.0, c.1, c.2, 1.0]);
            if (p[3] as f64) <= 1e-9 {
                return;
            }
            xs.push(p[0] as f64 / p[3] as f64);
            ys.push(p[1] as f64 / p[3] as f64);
        }
        let cx = (max_of(&xs) + min_of(&xs)) * 0.5;
        let cy = (max_of(&ys) + min_of(&ys)) * 0.5;
        let reach = py_max2((max_of(&xs) - min_of(&xs)) * 0.5, (max_of(&ys) - min_of(&ys)) * 0.5);
        if reach <= 0.0 {
            return;
        }
        if (reach - goal).abs() < 0.005 && cx.abs() < 0.005 && cy.abs() < 0.005 {
            return;
        }
        let f = cam.forward();
        let r = f.cross(cam.up_vector()).normalized(h);
        let u = r.cross(f);
        let half_h = cam.distance * t;
        cam.target = cam.target.add(r.mul((cx * half_h * cam.aspect) as f32)).add(u.mul((cy * half_h) as f32));
        let nearest = corners.iter().map(|c| -(c.sub(cam.target).dot(f)) as f64).fold(f64::NEG_INFINITY, py_max2);
        let gap = cam.distance - nearest;
        cam.distance = py_min2(py_max2(nearest + gap * reach / goal, MIN_DISTANCE), MAX_DISTANCE);
    }
}

// ---- what the scene tells the camera -----------------------------------

/// The scene questions a gesture asks (`Viewport._orbit_pivot_at`,
/// `_world_under_cursor`, `_model_diag`, `scene.version`). The sizes are
/// the viewport's, in whole pixels.
pub trait Probe {
    /// `scene.version`: the zoom pin lives while it holds.
    fn scene_version(&self) -> u64;
    /// `_orbit_pivot_at(x, y)`: the model point under the cursor, else the
    /// middle of what the view's central ray crosses of the model's box,
    /// else the camera target.
    fn pivot_at(&mut self, cam: &Camera, size: (i32, i32), x: f64, y: f64) -> V3;
    /// `_world_under_cursor(x, y)`: the geometry hit, else the ground, else
    /// the focal plane.
    fn world_under_cursor(&mut self, cam: &Camera, size: (i32, i32), x: f64, y: f64) -> Option<V3>;
    /// `_model_diag()`: the model's size, for the zoom step.
    fn model_diag(&mut self) -> f64;
}

/// `Viewport._depth_of`: how far `point` lies in front of the eye.
pub fn depth_of(cam: &Camera, point: V3) -> f64 {
    point.sub(cam.eye()).dot(cam.forward()) as f64
}

/// `Viewport._pixel_to_ray`: the camera ray `(origin, unit direction)`
/// through a pixel.
pub fn pixel_to_ray(cam: &Camera, size: (i32, i32), x: f64, y: f64) -> Option<(V3, V3)> {
    let w = size.0.max(1) as f64;
    let h = size.1.max(1) as f64;
    let ndc_x = 2.0 * x / w - 1.0;
    let ndc_y = 1.0 - 2.0 * y / h;
    let (inv, ok) = cam.mvp().inverted();
    if !ok {
        return None;
    }
    let at = |depth: f64| inv.map(V3(ndc_x as f32, ndc_y as f32, depth as f32));
    let p_near = at(-1.0);
    let mut direction = None;
    for depth in [1.0, 0.999, 0.9, 0.0] {
        let d = at(depth).sub(p_near);
        if finite3(d) && (d.length(cam.hypot) as f64) >= 1e-9 {
            direction = Some(d);
            break;
        }
    }
    let direction = direction?;
    if !finite3(p_near) {
        return None;
    }
    Some((p_near, direction.normalized(cam.hypot)))
}

/// `Viewport._world_to_pixel`: `None` behind the camera.
pub fn world_to_pixel(cam: &Camera, size: (i32, i32), world: V3) -> Option<(f64, f64)> {
    let clip = map_v4(&cam.mvp(), [world.0, world.1, world.2, 1.0]);
    if (clip[3] as f64) <= 0.0 {
        return None;
    }
    let ndc_x = clip[0] as f64 / clip[3] as f64;
    let ndc_y = clip[1] as f64 / clip[3] as f64;
    Some(((ndc_x * 0.5 + 0.5) * size.0 as f64, (1.0 - (ndc_y * 0.5 + 0.5)) * size.1 as f64))
}

// ---- gestures ------------------------------------------------------------

/// A camera navigation mode (`Viewport.nav_mode`): while one is set, the
/// left button drives the camera. (Zoom Window, a rubber band, is not in
/// this milestone.)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NavMode {
    Orbit,
    Pan,
    Zoom,
}

impl NavMode {
    pub fn from_name(name: &str) -> Option<NavMode> {
        match name {
            "orbit" => Some(NavMode::Orbit),
            "pan" => Some(NavMode::Pan),
            "zoom" => Some(NavMode::Zoom),
            _ => None,
        }
    }
}

/// The mouse buttons that navigate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Button {
    Left,
    Middle,
}

/// The preferences that change how the gestures read.
#[derive(Clone, Copy, Debug, Default)]
pub struct NavSettings {
    /// Preferences > orbit with the vertical axis reversed.
    pub invert_orbit_y: bool,
    /// Preferences > General > invert the wheel.
    pub invert_wheel: bool,
}

/// `Viewport._zoom_focus`: the world point the last wheel notch pinned.
#[derive(Clone, Debug)]
struct ZoomFocus {
    pose: [f64; 6],
    x: f64,
    y: f64,
    version: u64,
    focus: Option<V3>,
}

/// `qRound` on the pointer position (`QPointF.toPoint()`): half away from
/// zero.
fn q_round(v: f64) -> i32 {
    if v >= 0.0 {
        (v + 0.5) as i32
    } else {
        (v - 0.5) as i32
    }
}

/// Python's `round(x, ndigits)`: correctly rounded decimal.
fn py_round(x: f64, digits: usize) -> f64 {
    if !x.is_finite() {
        return x;
    }
    format!("{x:.digits$}").parse().unwrap_or(x)
}

/// The state of the viewport's camera gestures.
#[derive(Debug, Default)]
pub struct Navigator {
    pub settings: NavSettings,
    pub nav_mode: Option<NavMode>,
    last_pos: Option<(i32, i32)>,
    pan_mode: bool,
    middle_drag_active: bool,
    orbit_pivot: Option<V3>,
    pan_depth: Option<f64>,
    zoom_focus: Option<ZoomFocus>,
}

impl Navigator {
    pub fn new(settings: NavSettings) -> Navigator {
        Navigator { settings, ..Default::default() }
    }

    /// Whether a camera drag is in progress.
    pub fn dragging(&self) -> bool {
        self.last_pos.is_some()
    }

    /// `set_nav_mode`: a drag in progress ends here.
    pub fn set_nav_mode(&mut self, mode: Option<NavMode>) {
        self.nav_mode = mode;
        self.end_drag();
    }

    /// `_end_camera_drag`.
    pub fn end_drag(&mut self) {
        self.last_pos = None;
        self.orbit_pivot = None;
        self.pan_mode = false;
        self.middle_drag_active = false;
    }

    /// `mousePressEvent` for the camera: whether the press began a camera
    /// drag (the middle button always; the left one in a nav mode).
    pub fn press(&mut self, cam: &Camera, probe: &mut dyn Probe, size: (i32, i32), button: Button, x: f64, y: f64, shift: bool) -> bool {
        match button {
            Button::Middle => {
                self.last_pos = Some((q_round(x), q_round(y)));
                self.pan_mode = shift;
                self.middle_drag_active = true;
                self.orbit_pivot = Some(probe.pivot_at(cam, size, x, y));
                self.pan_depth = self.orbit_pivot.map(|p| depth_of(cam, p));
                true
            }
            Button::Left => {
                let Some(mode) = self.nav_mode else { return false };
                self.last_pos = Some((q_round(x), q_round(y)));
                self.pan_mode = mode == NavMode::Pan || shift;
                self.orbit_pivot = if mode == NavMode::Orbit { Some(probe.pivot_at(cam, size, x, y)) } else { None };
                self.pan_depth = if self.pan_mode { Some(depth_of(cam, probe.pivot_at(cam, size, x, y))) } else { None };
                true
            }
        }
    }

    /// `mouseMoveEvent` for the camera. `any_button` is whether a button is
    /// held (a camera drag whose release was lost ends here). Returns
    /// whether the camera moved.
    pub fn drag(&mut self, cam: &mut Camera, size: (i32, i32), x: f64, y: f64, shift: bool, any_button: bool) -> bool {
        if self.last_pos.is_some() && !any_button {
            self.end_drag();
        }
        if self.middle_drag_active && self.nav_mode.is_none() {
            // `_set_middle_drag_pan_mode`: Shift, held or not, switches the
            // drag between orbit and pan.
            self.pan_mode = shift;
        }
        let Some(last) = self.last_pos else { return false };
        let p = (q_round(x), q_round(y));
        let (dx, dy) = ((p.0 - last.0) as f64, (p.1 - last.1) as f64);
        self.last_pos = Some(p);
        let height = size.1;
        if self.nav_mode == Some(NavMode::Zoom) {
            zoom(cam, -dy * ZOOM_DRAG_STEPS);
        } else if self.pan_mode {
            pan(cam, dx, dy, height, self.pan_depth);
        } else {
            let ody = if self.settings.invert_orbit_y { -dy } else { dy };
            match self.orbit_pivot {
                Some(pivot) => orbit_about(cam, pivot, dx, ody, height),
                None => orbit(cam, dx, ody, height),
            }
        }
        true
    }

    /// `mouseReleaseEvent` for the camera.
    pub fn release(&mut self, button: Button) {
        match button {
            Button::Middle => self.end_drag(),
            Button::Left => {
                if self.nav_mode.is_some() {
                    self.last_pos = None;
                    self.orbit_pivot = None;
                    self.pan_mode = false;
                }
            }
        }
    }

    /// `wheelEvent`: `delta` is `angleDelta().y()` (120 per notch). Zooms
    /// towards the point under the cursor, which stays pinned while the
    /// camera and the scene stay as they are.
    pub fn wheel(&mut self, cam: &mut Camera, probe: &mut dyn Probe, size: (i32, i32), x: f64, y: f64, delta: f64) {
        let mut steps = delta / 120.0;
        if self.settings.invert_wheel {
            steps = -steps;
        }
        let version = probe.scene_version();
        let pose = |c: &Camera| {
            [
                py_round(c.yaw, 9),
                py_round(c.pitch, 9),
                py_round(c.distance, 6),
                py_round(c.target.0 as f64, 6),
                py_round(c.target.1 as f64, 6),
                py_round(c.target.2 as f64, 6),
            ]
        };
        let now = pose(cam);
        let focus = match &self.zoom_focus {
            Some(c) if c.pose == now && (x - c.x).abs() < FOCUS_REUSE_PX && (y - c.y).abs() < FOCUS_REUSE_PX && version == c.version => c.focus,
            cached => {
                // The pose changed, but the cursor usually still rests on
                // the same world point: project the pin instead of picking.
                let kept = cached.as_ref().and_then(|c| c.focus.filter(|_| c.version == version)).filter(|&f| {
                    world_to_pixel(cam, size, f).is_some_and(|p| (p.0 - x).abs() < FOCUS_REUSE_PX && (p.1 - y).abs() < FOCUS_REUSE_PX)
                });
                kept.or_else(|| probe.world_under_cursor(cam, size, x, y))
            }
        };
        match focus {
            Some(f) => {
                let diag = probe.model_diag();
                zoom_to(cam, steps, f, py_max2(0.05, 0.01 * diag));
            }
            None => zoom(cam, steps),
        }
        self.zoom_focus = Some(ZoomFocus { pose: pose(cam), x, y, version, focus });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn orbit_clamps_the_pitch() {
        let mut c = Camera::default();
        orbit(&mut c, 0.0, 1e6, 600);
        assert_eq!(c.pitch, pitch_limit());
        orbit(&mut c, 0.0, -1e6, 600);
        assert_eq!(c.pitch, (-89.0f64).to_radians());
    }

    #[test]
    fn zoom_stays_between_the_limits() {
        let mut c = Camera::default();
        zoom(&mut c, 1e4);
        assert_eq!(c.distance, MIN_DISTANCE);
        zoom(&mut c, -1e4);
        assert_eq!(c.distance, MAX_DISTANCE);
    }

    #[test]
    fn python_rounding() {
        assert_eq!(py_round(0.1234567891234, 9), 0.123456789);
        assert_eq!(py_round(-2.5e-7, 6), -0.0);
        assert_eq!(q_round(2.5), 3);
        assert_eq!(q_round(-2.5), -3);
    }
}
