// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! `core/style.py`: display styles. A style bundles how the model DRAWS (the
//! face mode, edge look and background); it is read from the document's
//! `display_style` block as the Python `Style.from_dict` does.

use ingetrazo_formats::pyjson::Value;

pub type Rgb = [f32; 3];

/// Face Styles (`FACE_MODES`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FaceMode {
    /// Shaded with textures: the default working look.
    Textures,
    /// Colours only: textured faces draw in their texture's average colour.
    Shaded,
    /// Flat white faces plus edges: the printing/plan style.
    HiddenLine,
    /// Flat default front/back colours, no materials: the reversed-face
    /// checker.
    Monochrome,
    /// Edges only, nothing occludes.
    Wireframe,
    /// Everything translucent, edges always visible.
    Xray,
}

impl FaceMode {
    pub fn from_name(name: &str) -> FaceMode {
        match name {
            "shaded" => FaceMode::Shaded,
            "hidden_line" => FaceMode::HiddenLine,
            "monochrome" => FaceMode::Monochrome,
            "wireframe" => FaceMode::Wireframe,
            "xray" => FaceMode::Xray,
            _ => FaceMode::Textures,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            FaceMode::Textures => "textures",
            FaceMode::Shaded => "shaded",
            FaceMode::HiddenLine => "hidden_line",
            FaceMode::Monochrome => "monochrome",
            FaceMode::Wireframe => "wireframe",
            FaceMode::Xray => "xray",
        }
    }
}

/// The classic default back-face blue-grey (`DEFAULT_BACK_COLOR`).
pub const DEFAULT_BACK_COLOR: Rgb = [0.62, 0.70, 0.78];

/// `Viewport.DEFAULT_FACE_COLOR`.
pub const DEFAULT_FACE_COLOR: Rgb = [0.96, 0.95, 0.925];

/// X-ray: how opaque each translucent face layer is (`XRAY_FACE_OPACITY`).
pub const XRAY_FACE_OPACITY: f32 = 0.6;
/// X-ray: how far hidden edges are washed toward the background.
pub const XRAY_HIDDEN_EDGE_FADE: f32 = 0.6;
/// How far the context is mixed toward the fade colour while a group is
/// open for editing (`EDIT_REST_FADE`).
pub const EDIT_REST_FADE: f32 = 0.75;

#[derive(Clone, Debug, PartialEq)]
pub struct Style {
    pub name: String,
    pub face_mode: FaceMode,
    pub edges: bool,
    /// Silhouette/profile edge pass.
    pub profiles: bool,
    /// Back Edges (K): the edges hidden behind faces, drawn dashed.
    pub back_edges: bool,
    pub edge_color: Rgb,
    /// Hidden line / monochrome faces.
    pub front_color: Rgb,
    /// `None` = automatic (the document's adopted tint, else the default).
    pub back_color: Option<Rgb>,
    pub background: Rgb,
    pub sky: bool,
    pub sky_color: Rgb,
    pub ground_color: Rgb,
    pub section_fill: bool,
    pub section_fill_color: Rgb,
}

impl Default for Style {
    fn default() -> Style {
        Style {
            name: "Default".into(),
            face_mode: FaceMode::Textures,
            edges: true,
            profiles: true,
            back_edges: false,
            edge_color: [0.13, 0.17, 0.23],
            front_color: [1.0, 1.0, 1.0],
            back_color: None,
            background: [0.90, 0.91, 0.92],
            sky: true,
            sky_color: [0.925, 0.935, 0.945],
            ground_color: [0.815, 0.820, 0.815],
            section_fill: true,
            section_fill_color: [0.35, 0.37, 0.41],
        }
    }
}

impl Style {
    /// `BUILTIN_STYLES`, by name.
    pub fn builtin(name: &str) -> Option<Style> {
        let white = [1.0, 1.0, 1.0];
        let d = Style::default();
        Some(match name {
            "Default" => d,
            "Architectural" => Style { name: name.into(), background: white, sky: false, ..d },
            "Shaded" => Style { name: name.into(), face_mode: FaceMode::Shaded, ..d },
            "Hidden line" => Style {
                name: name.into(),
                face_mode: FaceMode::HiddenLine,
                background: white,
                sky: false,
                ..d
            },
            "Monochrome" => Style {
                name: name.into(),
                face_mode: FaceMode::Monochrome,
                background: white,
                sky: false,
                ..d
            },
            "Wireframe" => Style {
                name: name.into(),
                face_mode: FaceMode::Wireframe,
                background: white,
                sky: false,
                ..d
            },
            "X-ray" => Style { name: name.into(), face_mode: FaceMode::Xray, ..d },
            _ => return None,
        })
    }

    /// The composer's `plano_style` overrides: "tecnico" is Hidden line,
    /// "lineas" is Wireframe, both on white without a sky.
    pub fn plano(kind: &str) -> Option<Style> {
        let mode = match kind {
            "tecnico" => FaceMode::HiddenLine,
            "lineas" => FaceMode::Wireframe,
            _ => return None,
        };
        Some(Style {
            name: kind.into(),
            face_mode: mode,
            sky: false,
            background: [1.0, 1.0, 1.0],
            ..Style::default()
        })
    }

    /// `Style.from_dict`: a missing or malformed key keeps the default.
    pub fn from_value(raw: &Value) -> Style {
        let d = Style::default();
        let flag = |k: &str, dv: bool| match raw.get(k) {
            Some(Value::Bool(b)) => *b,
            Some(Value::Int(s)) => s != "0",
            Some(Value::Float(f)) => *f != 0.0,
            Some(Value::Null) | None => dv,
            _ => dv,
        };
        let rgb = |k: &str, dv: Rgb| rgb_of(raw.get(k)).unwrap_or(dv);
        Style {
            name: match raw.get("name") {
                Some(Value::Str(s)) => s.clone(),
                _ => d.name,
            },
            face_mode: match raw.get("face_mode") {
                Some(Value::Str(s)) => FaceMode::from_name(s),
                _ => FaceMode::Textures,
            },
            edges: flag("edges", d.edges),
            profiles: flag("profiles", d.profiles),
            back_edges: flag("back_edges", d.back_edges),
            edge_color: rgb("edge_color", d.edge_color),
            front_color: rgb("front_color", d.front_color),
            back_color: rgb_of(raw.get("back_color")),
            background: rgb("background", d.background),
            sky: flag("sky", d.sky),
            sky_color: rgb("sky_color", d.sky_color),
            ground_color: rgb("ground_color", d.ground_color),
            section_fill: flag("section_fill", d.section_fill),
            section_fill_color: rgb("section_fill_color", d.section_fill_color),
        }
    }

    /// `effective_back_color`: the style's own tint, else the document's
    /// adopted one, else the classic default.
    pub fn effective_back_color(&self, scene_back: Option<Rgb>) -> Rgb {
        self.back_color.or(scene_back).unwrap_or(DEFAULT_BACK_COLOR)
    }
}

/// An RGB triple from a JSON array of at least three numbers.
pub fn rgb_of(v: Option<&Value>) -> Option<Rgb> {
    match v {
        Some(Value::Array(a)) if a.len() >= 3 => {
            let n = |x: &Value| match x {
                Value::Float(f) => Some(*f as f32),
                Value::Int(s) => s.parse::<f32>().ok(),
                _ => None,
            };
            Some([n(&a[0])?, n(&a[1])?, n(&a[2])?])
        }
        _ => None,
    }
}

/// `edit_fade_color(background)`: what the rest of the model fades TOWARDS
/// while a group is edited.
pub fn edit_fade_color(background: Rgb) -> Rgb {
    let f = |c: f32| (0.5 * c as f64 + 0.5 * 0.55) as f32;
    [f(background[0]), f(background[1]), f(background[2])]
}

#[cfg(test)]
mod tests {
    use super::*;
    use ingetrazo_formats::pyjson::loads;

    #[test]
    fn style_round_trips_the_python_dict() {
        let v = loads(
            r#"{"name":"X","face_mode":"xray","edges":false,"edge_color":[1,0,0],
                "back_color":[0.1,0.2,0.3],"sky":false}"#,
        )
        .unwrap();
        let s = Style::from_value(&v);
        assert_eq!(s.face_mode, FaceMode::Xray);
        assert!(!s.edges && !s.sky && s.profiles);
        assert_eq!(s.edge_color, [1.0, 0.0, 0.0]);
        assert_eq!(s.back_color, Some([0.1, 0.2, 0.3]));
    }

    #[test]
    fn unknown_mode_falls_back_to_textures() {
        let v = loads(r#"{"face_mode":"nope"}"#).unwrap();
        assert_eq!(Style::from_value(&v).face_mode, FaceMode::Textures);
    }
}
