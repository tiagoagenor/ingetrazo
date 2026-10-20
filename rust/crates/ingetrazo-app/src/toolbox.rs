// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
//! The tools the window offers (M4.5, M4.6): their names, shortcuts and
//! toolbar groups as `views/main_window.py` lays them out, the constructors
//! that put a headless tool behind the host, and the one-line hint of the
//! status bar (`views/status_hints.py`) for each step.

use ingetrazo_tools::arc::{ArcTool, CenterArcTool, ThreePointArcTool};
use ingetrazo_tools::base::{DrawTool, Field};
use ingetrazo_tools::circle::RadialTool;
use ingetrazo_tools::eraser::EraserTool;
use ingetrazo_tools::flip::FlipTool;
use ingetrazo_tools::freehand::FreehandTool;
use ingetrazo_tools::line::LineTool;
use ingetrazo_tools::move_tool::MoveTool;
use ingetrazo_tools::offset::OffsetTool;
use ingetrazo_tools::paint::PaintTool;
use ingetrazo_tools::protractor::ProtractorTool;
use ingetrazo_tools::rectangle::RectangleTool;
use ingetrazo_tools::rotate::RotateTool;
use ingetrazo_tools::rotated_rectangle::RotatedRectangleTool;
use ingetrazo_tools::scale::ScaleTool;
use ingetrazo_tools::tape::TapeMeasureTool;

use crate::adapters::{ModifyAdapter, ModifyKind, PushPullAdapter, SelectAdapter};

/// A tool of the window.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ToolId {
    Select,
    Eraser,
    Paint,
    Line,
    Freehand,
    Rectangle,
    RotatedRect,
    Circle,
    Polygon,
    Arc,
    Arc3,
    CenterArc,
    Pie,
    PushPull,
    Move,
    Rotate,
    Scale,
    Flip,
    Offset,
    Tape,
    Protractor,
}

/// A toolbar of the window (`MainWindow._build_toolbar`'s layout).
pub struct ToolGroup {
    pub title: &'static str,
    pub tools: &'static [ToolId],
}

/// The toolbars, in the Python window's order (the tools not ported yet are
/// not listed).
pub const GROUPS: &[ToolGroup] = &[
    ToolGroup { title: "Main", tools: &[ToolId::Select, ToolId::Eraser, ToolId::Paint] },
    ToolGroup {
        title: "Draw",
        tools: &[
            ToolId::Line,
            ToolId::Freehand,
            ToolId::Rectangle,
            ToolId::RotatedRect,
            ToolId::Circle,
            ToolId::Polygon,
            ToolId::Arc,
            ToolId::Arc3,
            ToolId::CenterArc,
            ToolId::Pie,
        ],
    },
    ToolGroup { title: "Modify", tools: &[ToolId::PushPull, ToolId::Move, ToolId::Rotate, ToolId::Scale, ToolId::Flip, ToolId::Offset] },
    ToolGroup { title: "Annotate", tools: &[ToolId::Tape, ToolId::Protractor] },
];

/// Every tool, in toolbar order.
pub fn all() -> impl Iterator<Item = ToolId> {
    GROUPS.iter().flat_map(|g| g.tools.iter().copied())
}

/// What a tool shows its user: the key the Python window files it under,
/// the name on its button and the shortcut. A shortcut is the character a
/// keyboard types for it, so `Shift+O` is `'O'`.
impl ToolId {
    pub fn key(self) -> &'static str {
        match self {
            ToolId::Select => "select",
            ToolId::Eraser => "eraser",
            ToolId::Paint => "paint",
            ToolId::Line => "line",
            ToolId::Freehand => "freehand",
            ToolId::Rectangle => "rectangle",
            ToolId::RotatedRect => "rotated_rect",
            ToolId::Circle => "circle",
            ToolId::Polygon => "polygon",
            ToolId::Arc => "arc",
            ToolId::Arc3 => "arc3",
            ToolId::CenterArc => "center_arc",
            ToolId::Pie => "pie",
            ToolId::PushPull => "pushpull",
            ToolId::Move => "move",
            ToolId::Rotate => "rotate",
            ToolId::Scale => "scale",
            ToolId::Flip => "flip",
            ToolId::Offset => "offset",
            ToolId::Tape => "tape",
            ToolId::Protractor => "protractor",
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            ToolId::Select => "Select",
            ToolId::Eraser => "Eraser",
            ToolId::Paint => "Paint",
            ToolId::Line => "Line",
            ToolId::Freehand => "Freehand",
            ToolId::Rectangle => "Rectangle",
            ToolId::RotatedRect => "Rotated Rect",
            ToolId::Circle => "Circle",
            ToolId::Polygon => "Polygon",
            ToolId::Arc => "Arc",
            ToolId::Arc3 => "3-Point Arc",
            ToolId::CenterArc => "Center Arc",
            ToolId::Pie => "Pie",
            ToolId::PushPull => "Push / Pull",
            ToolId::Move => "Move",
            ToolId::Rotate => "Rotate",
            ToolId::Scale => "Scale",
            ToolId::Flip => "Flip",
            ToolId::Offset => "Offset",
            ToolId::Tape => "Tape Measure",
            ToolId::Protractor => "Protractor",
        }
    }

    /// The character that picks the tool (`Tool.shortcut`); `' '` is Select.
    pub fn shortcut(self) -> Option<char> {
        match self {
            ToolId::Select => Some(' '),
            ToolId::Eraser => Some('e'),
            ToolId::Paint => Some('b'),
            ToolId::Line => Some('l'),
            ToolId::Rectangle => Some('r'),
            ToolId::Circle => Some('c'),
            ToolId::Arc => Some('a'),
            ToolId::Arc3 => Some('j'),
            ToolId::CenterArc => Some('O'),
            ToolId::PushPull => Some('p'),
            ToolId::Move => Some('m'),
            ToolId::Rotate => Some('q'),
            ToolId::Scale => Some('s'),
            ToolId::Offset => Some('f'),
            ToolId::Tape => Some('t'),
            ToolId::Protractor => Some('H'),
            ToolId::Freehand | ToolId::RotatedRect | ToolId::Polygon | ToolId::Pie | ToolId::Flip => None,
        }
    }

    /// The shortcut as the menus write it (`Shift+O`, `L`, `Space`).
    pub fn shortcut_label(self) -> Option<String> {
        let c = self.shortcut()?;
        Some(match c {
            ' ' => "Space".to_string(),
            c if c.is_ascii_uppercase() => format!("Shift+{c}"),
            c => c.to_ascii_uppercase().to_string(),
        })
    }

    /// The tool a typed character picks.
    pub fn from_shortcut(c: char) -> Option<ToolId> {
        all().find(|t| t.shortcut() == Some(c))
    }

    /// A text for the button where there is no icon yet.
    pub fn glyph(self) -> &'static str {
        match self {
            ToolId::Select => "Sel",
            ToolId::Eraser => "Era",
            ToolId::Paint => "Pnt",
            ToolId::Line => "Lin",
            ToolId::Freehand => "Fre",
            ToolId::Rectangle => "Rec",
            ToolId::RotatedRect => "RoR",
            ToolId::Circle => "Cir",
            ToolId::Polygon => "Pol",
            ToolId::Arc => "Arc",
            ToolId::Arc3 => "Ar3",
            ToolId::CenterArc => "CAr",
            ToolId::Pie => "Pie",
            ToolId::PushPull => "P/P",
            ToolId::Move => "Mov",
            ToolId::Rotate => "Rot",
            ToolId::Scale => "Sca",
            ToolId::Flip => "Flp",
            ToolId::Offset => "Off",
            ToolId::Tape => "Tap",
            ToolId::Protractor => "Pro",
        }
    }
}

/// A fresh tool, ready for `Host::set_active_tool`.
pub fn make(id: ToolId) -> Box<dyn DrawTool> {
    match id {
        ToolId::Select => Box::new(SelectAdapter::new()),
        ToolId::Eraser => Box::new(ModifyAdapter::new(ModifyKind::Eraser, Box::new(EraserTool::new()))),
        ToolId::Paint => Box::new(ModifyAdapter::new(ModifyKind::Paint, Box::new(PaintTool::new()))),
        ToolId::Line => Box::new(LineTool::new()),
        ToolId::Freehand => Box::new(FreehandTool::new()),
        ToolId::Rectangle => Box::new(RectangleTool::new()),
        ToolId::RotatedRect => Box::new(RotatedRectangleTool::new()),
        ToolId::Circle => Box::new(RadialTool::circle()),
        ToolId::Polygon => Box::new(RadialTool::polygon()),
        ToolId::Arc => Box::new(ArcTool::new()),
        ToolId::Arc3 => Box::new(ThreePointArcTool::new()),
        ToolId::CenterArc => Box::new(CenterArcTool::center_arc()),
        ToolId::Pie => Box::new(CenterArcTool::pie()),
        ToolId::PushPull => Box::new(PushPullAdapter::new()),
        ToolId::Move => Box::new(ModifyAdapter::new(ModifyKind::Move, Box::new(MoveTool::new()))),
        ToolId::Rotate => Box::new(ModifyAdapter::new(ModifyKind::Rotate, Box::new(RotateTool::new()))),
        ToolId::Scale => Box::new(ModifyAdapter::new(ModifyKind::Scale, Box::new(ScaleTool::new()))),
        ToolId::Flip => Box::new(ModifyAdapter::new(ModifyKind::Flip, Box::new(FlipTool::new()))),
        ToolId::Offset => Box::new(ModifyAdapter::new(ModifyKind::Offset, Box::new(OffsetTool::new()))),
        ToolId::Tape => Box::new(ModifyAdapter::new(ModifyKind::Tape, Box::new(TapeMeasureTool::new()))),
        ToolId::Protractor => Box::new(ModifyAdapter::new(ModifyKind::Protractor, Box::new(ProtractorTool::new()))),
    }
}

// ---- the status hint -----------------------------------------------------------------------

/// `status_hints.phase_of`: which step the tool is in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Phase {
    Idle,
    Started,
    /// The arc's second and third steps.
    End,
    Bulge,
}

/// `phase_of(key, tool)`.
pub fn phase_of(id: ToolId, tool: &dyn DrawTool) -> Phase {
    let fields = tool.fields();
    let field = |name: &str| fields.iter().find(|(n, _)| *n == name).map(|(_, f)| f);
    if id == ToolId::Arc {
        if field("end_point").is_some_and(|f| !matches!(f, Field::Null)) {
            return Phase::Bulge;
        }
        return if tool.start_point().is_some() { Phase::End } else { Phase::Idle };
    }
    if tool.start_point().is_some() || tool.busy() {
        return Phase::Started;
    }
    if matches!(field("dragging"), Some(Field::Bool(true))) {
        return Phase::Started;
    }
    Phase::Idle
}

/// The hint text of a tool and its step (`HINTS`); `idle` before the first
/// click, `started` after it.
pub fn hint_text(id: ToolId, phase: Phase) -> &'static str {
    let two = |idle: &'static str, started: &'static str| if phase == Phase::Idle { idle } else { started };
    match id {
        ToolId::Select => "Shift = add/remove, Ctrl = add. Double-click a face or an edge to take what touches it.",
        ToolId::Eraser => "Drag over edges to erase them. Shift = hide instead.",
        ToolId::Paint => "Click a face to paint it. Alt = sample the material under the cursor.",
        ToolId::Line => two("Click the start point. Arrows lock an axis, Shift locks the inference.", "Click the end point, or type the length and Enter."),
        ToolId::Freehand => "Press and drag to draw a freehand line.",
        ToolId::Rectangle => two("Click the first corner. Arrows pick the plane; Ctrl = centre.", "Click the opposite corner, or type width;height and Enter."),
        ToolId::RotatedRect => two("Click the first corner of the base edge.", "Edge end, then width (or type them). Shift holds the direction."),
        ToolId::Circle | ToolId::Polygon => two(
            "Click the centre. Arrows pick the plane, Down = perpendicular to an edge. Type the sides first.",
            "Click the radius, or type it and Enter.",
        ),
        ToolId::Arc => match phase {
            Phase::Idle => "Click the start point - on an edge, a tangent arc. Type the segments first.",
            Phase::Started | Phase::End => "End point. Magenta = same distance; double-click rounds.",
            Phase::Bulge => "Click the bulge, or type it. 'Nr' = radius, 'Ns' = segments. Alt keeps the corner.",
        },
        ToolId::Arc3 => two("Click the start point. Type the segments first.", "Click a point the arc passes through, then the end."),
        ToolId::CenterArc => two("Click the centre. Type the segments first.", "Click the start of the arc, then its end; or type the angle."),
        ToolId::Pie => two("Click the centre. Type the segments first.", "Click the start of the wedge, then its end; or type the angle."),
        ToolId::PushPull => two("Click a face and move.", "Move, or type the distance and Enter. Double-click repeats the last."),
        ToolId::Move => two(
            "Click what to move (or select it first). Red + on a group = rotate it.",
            "Click the destination, or type the distance and Enter.",
        ),
        ToolId::Rotate => two(
            "Click the centre of rotation - on a face, the protractor takes its plane.",
            "Click the start of the angle, then the end; or type the degrees.",
        ),
        ToolId::Scale => two(
            "Select something, then drag a grip: corners uniform, edges along an axis.",
            "Drag, or type a factor (2) or a size (2m). Ctrl = centre, Shift = uniform.",
        ),
        ToolId::Flip => "Click the plane to mirror the selection across.",
        ToolId::Offset => two("Click a face, or connected edges, to offset.", "Move to set the offset, or type it and Enter."),
        ToolId::Tape => two("Click a point or an edge to measure. Arrows lock an axis.", "Click the second point, or type the distance and Enter."),
        ToolId::Protractor => two("Click the vertex of the angle.", "Click the start of the angle, then the end; or type the degrees."),
    }
}

/// `NAV_HINTS`.
pub fn nav_hint(mode: crate::nav::NavMode) -> &'static str {
    match mode {
        crate::nav::NavMode::Orbit => "Drag to orbit. Shift = pan. Wheel = zoom. The middle button orbits from any tool.",
        crate::nav::NavMode::Pan => "Drag to pan. Wheel = zoom.",
        crate::nav::NavMode::Zoom => "Drag up to zoom in, down to zoom out. Wheel zooms at the cursor.",
    }
}

/// `hint_for`: the one line the status bar shows, led by the tool's name
/// (`MainWindow._update_status_hint`); the tool's own clause follows while
/// it is idle.
pub fn status_line(id: ToolId, tool: &dyn DrawTool, core: &ingetrazo_tools::host::Core, nav: Option<crate::nav::NavMode>) -> String {
    if let Some(mode) = nav {
        return format!("{:?} - {}", mode, nav_hint(mode));
    }
    let phase = phase_of(id, tool);
    let mut text = hint_text(id, phase).to_string();
    if phase == Phase::Idle {
        let clause = tool.overlay(core).clause;
        if !clause.is_empty() {
            text = format!("{text} | {clause}");
        }
    } else if tool.uses_snap() {
        let mode = match core.inference.linear_mode.unwrap_or(ingetrazo_snap::LinearMode::All) {
            ingetrazo_snap::LinearMode::All => "all",
            ingetrazo_snap::LinearMode::Off => "none",
            ingetrazo_snap::LinearMode::ParallelPerp => "parallel / perpendicular",
        };
        text = format!("{text} | Alt = inferences ({mode})");
    }
    format!("{} - {text}", id.name())
}
