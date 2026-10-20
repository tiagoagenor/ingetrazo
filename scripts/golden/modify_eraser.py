# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
"""The Eraser tool's scenes and programs for ``make_modify_tool_fixtures.py``
(``tools/eraser.py``, docs/rust-rewrite-plan.md M3.13).

Two pins, both reported in the Rust tool's docs:

* ``marked`` is a Python ``set`` of objects (address order): the tool under
  test keeps it in insertion order;
* the red box a marked GROUP draws on the stroke (``_group_box_segments``,
  the viewport's ``_group_obb``) is not ported: the tool under test draws none.
"""
from __future__ import annotations

import math
import random

from _modify_harness import (ALT, CTRL, SHIFT, V, above, at, box, box_faces, click, dbl, deactivate, esc,  # noqa: F401
                             group, hover, key, keyup, press, redo, release, select, tap, typed, undo, xf,
                             Mesh, OrderedSelection, Scene, activate, camera, pbits, px)
from core.dimension import Dimension
from core.guide import Guide
from core.layers import Layer
from core.mesh import Edge
from core.textlabel import TextLabel
from georef.geopath import GeoPath
from PySide6.QtGui import QFont, QFontMetrics
from tools.eraser import EraserTool

NAME = "eraser"


class PinnedEraser(EraserTool):
    """The Eraser with an insertion-ordered ``marked`` and no group box."""

    @property
    def marked(self):
        return self._marked

    @marked.setter
    def marked(self, value):
        self._marked = OrderedSelection(value)

    def _group_box_segments(self, group) -> list:
        return []


make_tool = PinnedEraser

CAM = dict(target=(1, 0, 0.8), distance=20.0, yaw=-50.0, pitch=30.0, w=1000, h=600)
CAM_ORTHO = dict(target=(1, 0, 0.8), distance=20.0, yaw=-50.0, pitch=30.0, w=1000, h=600, perspective=False)
CAM_LOW = dict(target=(1, 0, 1.0), distance=22.0, yaw=-30.0, pitch=8.0, w=1000, h=600)
CAM_TOP = dict(target=(1, 0, 0.0), distance=22.0, yaw=-90.0, pitch=89.0, w=1000, h=600)


def circle(m: Mesh, cx, cy, cz, r, n, face=False):
    pts = [V(cx + r * math.cos(2 * math.pi * k / n), cy + r * math.sin(2 * math.pi * k / n), cz) for k in range(n)]
    if face:
        m.add_face(pts)
    else:
        for k in range(n):
            m.add_edge(pts[k], pts[(k + 1) % n])
    m.tag_curve(pts, True)
    return pts


def cmid(cx, cy, cz, r, k, n):
    """The middle of the k-th chord of the circle drawn above."""
    a = 2 * math.pi * k / n
    b = 2 * math.pi * (k + 1) / n
    return (cx + r * (math.cos(a) + math.cos(b)) / 2, cy + r * (math.sin(a) + math.sin(b)) / 2, cz)


def scn_tools() -> Scene:
    s = Scene()
    m = s.mesh
    m.add_face([V(0, 0, 0), V(2, 0, 0), V(2, 4, 0), V(0, 4, 0)])
    m.add_face([V(2, 0, 0), V(4, 0, 0), V(4, 4, 0), V(2, 4, 0)])
    box_faces(m, 6, 0, 0, 3, 3, 3)
    m.add_edge(V(-3, 0, 0), V(-3, 2, 1))
    m.add_edge(V(-3, 3, 0), V(-1, 3, 0))
    circle(m, -4, -3, 0, 1.5, 12)
    circle(m, 2, -4, 0, 1.2, 12, face=True)
    soft = m.add_edge(V(5, 5, 0), V(6, 6, 0))
    soft.soft = True
    hid = m.add_edge(V(7, 5, 0), V(8, 6, 0))
    hid.hidden = True
    return s


def scn_annot() -> Scene:
    s = Scene()
    s.layers.append(Layer("Hid", visible=False))
    s.layers.append(Layer("Lock", locked=True))
    box_faces(s.mesh, 0, 0, 0, 3, 2, 2)
    s.mesh.add_face([V(4, 0, 0), V(7, 0, 0), V(7, 3, 0), V(4, 3, 0)])
    s.dimensions.append(Dimension(V(0, 0, 2), V(3, 0, 2), V(0, 0, 1.2)))
    s.dimensions.append(Dimension(V(4, 0, 0), V(7, 3, 0), V(0, -1, 0), axis="x"))
    s.dimensions.append(Dimension(V(0, 2, 0), V(0, 2, 2), V(-1, 0, 0), layer="Hid"))
    s.dimensions.append(Dimension(V(3, 2, 0), V(3, 2, 2), V(1, 0, 0), layer="Lock"))
    s.dimensions.append(Dimension(V(-3, -2, 1), V(-3, 2, 1), V(0, 0, 0.8), axis="y"))
    s.text_labels.append(TextLabel(V(1, 1, 2), V(1.5, 0.5, 1.5), "Roof"))
    s.text_labels.append(TextLabel(V(5, 1, 0), V(-3, -1, 1.0), "Left of the anchor\nsecond line"))
    s.text_labels.append(TextLabel(V(-2, 0, 0), V(0.5, 0.5, 2), "H", layer="Hid"))
    s.text_labels.append(TextLabel(V(6, 2, 0), V(0.3, 0.8, 0.5), "Locked", layer="Lock"))
    s.guides.append(Guide(V(1, 1, 0), V(1, 0.5, 0)))
    s.guides.append(Guide(V(2, -2, 0), V(0, 0, 1)))
    s.guides.append(Guide(V(6, -2, 1)))
    s.guides.append(Guide(V(-2, 3, 0.5), None, V(-2, 3, 0)))
    s.geo_paths.append(GeoPath([V(-5, -3, 0), V(-2, -3.5, 0), V(0, -4, 0)], False, "road"))
    s.geo_paths.append(GeoPath([V(8, -2, 0), V(10, -2, 0), V(10, 1, 0), V(8, 1, 0)], True, "lot"))
    s.groups.append(group(box(8, 4, 0, 1, 1, 1), "g"))
    return s


def scn_groups() -> Scene:
    s = Scene()
    proto = box(0, 0, 0, 1, 1, 1, (0.2, 0.6, 0.9))
    g0 = group(box(-5, 0, 0, 2, 2, 1.5), "classic")
    g1 = group(proto, "inst_a", xf(6, 0, 0))
    g2 = group(proto, "inst_b", xf(9, 2, 0, 30))
    k1 = group(proto, "kid1", xf(1, 0, 0, 20))
    k2 = group(proto, "kid2", xf(0, 2, 0))
    ctr = group(Mesh(), "container", xf(6, 6, 0, 10), children=[k1, k2])
    g4 = group(box(-5, 5, 0, 1, 1, 1), "classic2")
    s.groups.extend([g0, g1, g2, ctr, g4])
    s.mesh.add_face([V(0, 0, 0), V(4, 0, 0), V(4, 4, 0), V(0, 4, 0)])
    s.mesh.add_edge(V(0, 6, 0), V(3, 6, 0))
    return s


def scn_occl() -> Scene:
    """A tall box in front of a wall: edges behind a face are not visible."""
    s = Scene()
    box_faces(s.mesh, 0, 0, 0, 4, 4, 4)
    box_faces(s.mesh, 6, 0, 0, 3, 3, 2)
    s.mesh.add_edge(V(2, 6, 0), V(6, 6, 0))
    return s


SCENES = {"tools": scn_tools, "annot": scn_annot, "groups": scn_groups, "occl": scn_occl}

PROGRAMS: list = []


def prog(scene, cam, *events):
    PROGRAMS.append((scene, cam, list(events)))


def stroke(pts, **mods):
    """A press on the first point, drags through the rest, a release on the last."""
    evs = [hover(pts[0]), press(pts[0], **mods)]
    for p in pts[1:]:
        evs.append(hover(p, drag=True, **mods))
    evs.append(release(pts[-1], **mods))
    return evs


# the plate and its divider
DIV = (2, 2, 0)
PLATE_L = (0, 2, 0)
PLATE_B = (3, 0, 0)
# the box
BOX_TOP_F = (7.5, 0, 3)
BOX_TOP_R = (9, 1.5, 3)
BOX_VERT = (6, 0, 1.5)
BOX_BOT_F = (7.5, 0, 0)
L1 = (-3, 1, 0.5)
L2 = (-2, 3, 0)
C1 = [cmid(-4, -3, 0, 1.5, k, 12) for k in range(12)]
C2 = [cmid(2, -4, 0, 1.2, k, 12) for k in range(12)]
SOFT = (5.5, 5.5, 0)
HID = (7.5, 5.5, 0)

# clicks: a divider (merges), an outer edge (takes the face), a loose edge, a curve, a soft and a hidden edge
prog("tools", CAM, hover(DIV), click(DIV), undo(), redo())
prog("tools", CAM, hover(PLATE_L), click(PLATE_L), undo())
prog("tools", CAM, click(L1), click(L2), undo(), undo(), redo())
prog("tools", CAM, hover(C1[3]), click(C1[3]), undo())
prog("tools", CAM, click(C2[5]), redo(), undo())
prog("tools", CAM, hover(SOFT), click(SOFT), hover(HID), click(HID))
prog("tools", CAM, click(BOX_TOP_F), click(BOX_VERT), undo())
prog("tools", CAM, click(px(30, 30)), hover(px(31, 30)), release(px(31, 30)))
# strokes across several edges: one undo step
prog("tools", CAM, *stroke([L1, (-3, 1.5, 0.75), L2]), undo(), redo())
prog("tools", CAM, *stroke([BOX_TOP_F, (8.2, 0, 3), BOX_TOP_R, BOX_VERT, BOX_BOT_F]), undo())
prog("tools", CAM, *stroke([DIV, (3, 3, 0), (3, 4, 0), PLATE_L, (1, 0, 0)]), undo(), redo())
prog("tools", CAM, *stroke([C1[0], C1[1], C1[2], C2[4], L1]), undo())
# Esc cancels the stroke
prog("tools", CAM, hover(L1), press(L1), hover((-2, 3, 0), drag=True), esc(), release((-2, 3, 0)), hover(L1))
prog("tools", CAM, hover(DIV), press(DIV), esc(), hover(DIV), click(L1))
# Shift at press: hide the stroke, not erase
prog("tools", CAM, *stroke([L1, L2], shift=True), undo(), redo(), undo())
prog("tools", CAM, *stroke([BOX_TOP_F, BOX_TOP_R, BOX_VERT], shift=True), undo())
prog("tools", CAM, hover(C1[2]), press(C1[2], shift=True), hover(C1[5], drag=True), release(C1[5]), undo())
prog("tools", CAM, hover(DIV), click(DIV, shift=True), click(DIV, shift=True), undo())
prog("tools", CAM, *stroke([L1, L2]), *stroke([L1, L2], shift=True))
# the modifier latches at the press: Shift later does nothing
prog("tools", CAM, hover(L1), press(L1), hover(L2, drag=True, shift=True), release(L2, shift=True))
prog("tools", CAM, hover(L1), press(L1, shift=True), hover(L2, drag=True), release(L2))
# a selection survives or loses what the erase takes
prog("tools", CAM, select(("edge", 19)), hover(L1), click(L1), undo())
prog("tools", CAM, select(("face", 0), ("edge", 19)), click(PLATE_L), undo(), redo())
prog("tools", CAM, select(("edge", 19), ("face", 1)), click(PLATE_B), undo())
prog("tools", CAM, select(("edge", 19)), click(L1, shift=True), undo())
# cameras
prog("tools", CAM_TOP, *stroke([PLATE_L, (1, 0, 0), PLATE_B, DIV]), undo())
prog("tools", CAM_ORTHO, *stroke([L1, L2]), undo())
prog("tools", CAM_LOW, *stroke([BOX_TOP_F, BOX_VERT]), undo())
# occlusion: the far edges of a box are not erased through its faces
prog("occl", CAM, hover((4, 4, 2)), click((4, 4, 2)), hover((1, 4, 4)), click((1, 4, 4)), hover((2, 4, 0)), click((2, 4, 0)))
prog("occl", CAM, *stroke([(0, 0, 2), (2, 0, 4), (4, 0, 2), (2, 4, 4), (0, 4, 2)]), undo())
prog("occl", CAM, *stroke([(4, 3, 0), (4, 1, 0), (3, 0, 0), (1, 0, 0), (0, 1, 0)]), undo())
prog("occl", CAM, hover((3, 0, 4)), click((3, 0, 4)), hover((7.5, 0, 2)), click((7.5, 0, 2)), click((4, 6, 0)))
prog("occl", CAM_LOW, *stroke([(2, 0, 4), (4, 1, 4), (4, 3, 4), (2, 4, 4)]))
prog("occl", CAM, *stroke([(2, 0, 4), (4, 1, 4), (4, 3, 4)], shift=True), undo())
# annotations: guides, dimensions, texts, paths
G1 = (1.5, 1.25, 0)
G2 = (2, -2, 2)
G3 = (6, -2, 1)
D1 = (1.5, 0, 3.2)
D2 = (5.5, -1.2, 0)
LAB1 = (1.5, 0.5, 1.5)
LAB2 = (2.0, 0.0, 1.0)
GEO1 = (-3.5, -3.25, 0)
GEO2 = (10, -0.5, 0)
for p in (G1, G2, G3, D1, D2, LAB1, LAB2, GEO1, GEO2):
    prog("annot", CAM, hover(p), click(p), undo(), redo())
prog("annot", CAM, *stroke([G1, G2, G3]), undo(), redo())
prog("annot", CAM, *stroke([D1, D2, LAB1, GEO1]), undo(), redo())
prog("annot", CAM, *stroke([LAB1, (1.5, 1, 1.7), G1, GEO2]), undo())
prog("annot", CAM, *stroke([G1, D1, GEO1, (0, 0, 1)]), undo())
prog("annot", CAM, *stroke([G1, D1, GEO1, (0, 0, 1)], shift=True), undo())
prog("annot", CAM, *stroke([(8.5, 4.5, 1), G1, GEO1]), undo(), redo())
prog("annot", CAM, hover(px(500, 300)), hover(px(480, 290)), click(px(480, 290)))
prog("annot", CAM, select(("guide", 0), ("guide", 1), ("dim", 0), ("dim", 1)), click(G1))
prog("annot", CAM, select(("guide", 1), ("guide", 2), ("geo", 1), ("label", 1)), click(G1), click(GEO1), click(LAB2))
prog("annot", CAM, select(("dim", 4), ("dim", 0)), click(D1), click(D2))
prog("annot", CAM, select(("guide", 2), ("label", 0)), *stroke([G1, LAB1]))
# groups erased whole; later groups' paths shift
GA = (-4, 1, 0.75)
GB = (6.5, 0.5, 0.5)
GC = (9, 2, 0.5)
GK = (6.5, 7, 0.5)
GD = (-4.5, 5.5, 0.5)
prog("groups", CAM, hover(GA), click(GA), undo(), redo())
prog("groups", CAM, hover(GB), click(GB), undo())
prog("groups", CAM, click(GC), undo())
prog("groups", CAM, click(GK), undo())
prog("groups", CAM, click(GD), undo())
prog("groups", CAM, *stroke([GA, GB, GC]), undo(), redo())
prog("groups", CAM, *stroke([GA, (1, 2, 0), GB, GD]), undo())
prog("groups", CAM, select(("group", [2])), click(GB), click(GA))
prog("groups", CAM, select(("group", [1]), ("group", [4])), click(GA))
prog("groups", CAM, select(("group", [1]), ("group", [4]), ("face", 0)), *stroke([GA, GB]))
prog("groups", CAM, *stroke([GB, GC], shift=True), undo(), redo(), undo())
prog("groups", CAM, *stroke([GA, (1, 2, 0), (2, 6, 0)], shift=True), undo())
prog("groups", CAM, *stroke([GA, (1, 2, 0), (2, 6, 0)]), undo())
prog("groups", CAM, click(GA, shift=True), click(GA, shift=True), undo(), undo())
prog("groups", CAM_ORTHO, *stroke([GA, GB, GC, GD]), undo())
prog("groups", CAM_TOP, *stroke([GA, GB, GC, GD, (1, 6, 0)]), undo())


# ---- seeded random programs --------------------------------------------------------------------

def random_program(seed: int, scene: str, cam: dict, n: int = 24):
    rnd = random.Random(seed)
    pts = {
        "tools": [DIV, PLATE_L, PLATE_B, BOX_TOP_F, BOX_TOP_R, BOX_VERT, BOX_BOT_F, L1, L2, C1[0], C1[3], C1[7], C2[2],
                  C2[6], SOFT, HID, (3, 3, 0), (1, 1, 0), (8, 1, 3)],
        "annot": [G1, G2, G3, D1, D2, LAB1, LAB2, GEO1, GEO2, (1.5, 0, 0), (5, 1.5, 0), (8.5, 4.5, 1), (0, 0, 1)],
        "groups": [GA, GB, GC, GK, GD, (2, 2, 0), (1, 6, 0), (4, 3, 0), (-1, 0, 0)],
        "occl": [(4, 4, 2), (1, 4, 4), (2, 4, 0), (2, 0, 4), (4, 0, 2), (7.5, 0, 2), (4, 6, 0), (2, 2, 4), (6, 0, 1)],
    }[scene]
    refs = {"tools": [("face", 0), ("edge", 19), ("face", 1)], "annot": [("face", 0), ("guide", 0)],
            "groups": [("face", 0)], "occl": [("face", 0), ("edge", 0)]}[scene]
    evs = []
    held = False
    for i in range(n):
        r = rnd.random()
        p = rnd.choice(pts)
        q = (p[0] + rnd.choice((0, 0, 0.2, -0.3)), p[1] + rnd.choice((0, 0, 0.3)), p[2])
        sh = rnd.random() < 0.25
        if i < 2 and r < 0.4:
            evs.append(select(rnd.choice(refs)))
        elif held:
            if r < 0.6:
                evs.append(hover(q, drag=True))
            elif r < 0.9:
                evs.append(release(q))
                held = False
            else:
                evs.append(esc())
                held = False
        elif r < 0.25:
            evs.append(hover(q))
        elif r < 0.45:
            evs.append(click(q, shift=sh))
        elif r < 0.7:
            evs.append(press(q, shift=sh))
            held = True
        elif r < 0.78:
            evs.append(undo())
        elif r < 0.84:
            evs.append(redo())
        elif r < 0.9:
            evs.append(esc())
        else:
            evs.append(hover(q))
    if held:
        evs.append(release(q))
    return (scene, cam, evs)


for _seed in range(300, 340):
    _sc = ("tools", "annot", "groups", "occl")[_seed % 4]
    PROGRAMS.append(random_program(_seed, _sc, (CAM, CAM_ORTHO, CAM_LOW, CAM_TOP)[(_seed // 4) % 4]))


# ---- per-program extras, probe -------------------------------------------------------------------

def setup(s, n):
    """The text metrics the pick of a label measures (Qt's, on the viewport)."""
    sc = s.vp.scene
    style = getattr(sc, "dimension_style", {})
    font = QFont()
    font.setPointSize(int(style.get("font_size", 9)))
    font.setBold(True)
    fm = QFontMetrics(font, s.vp)
    adv = {}
    for lab in sc.text_labels:
        for ln in (lab.text.splitlines() or [""]):
            adv[ln] = fm.horizontalAdvance(ln)
    return {"metrics": {"height": fm.height(), "ascent": fm.ascent(), "descent": fm.descent(), "adv": adv}}


def extra_probe(tool, vp, ids):
    from core.dimension import Dimension as D
    from core.group import Group
    m = list(tool.marked)
    return {
        "stroke": bool(tool._stroke),
        "hide": bool(tool._hide),
        "n_marked": len(m),
        "n_edges": sum(isinstance(x, Edge) for x in m),
        "n_groups": sum(isinstance(x, Group) for x in m),
        "n_guides": sum(isinstance(x, Guide) for x in m),
        "n_dims": sum(isinstance(x, D) for x in m),
        "n_geo": sum(isinstance(x, GeoPath) for x in m),
        "n_labels": sum(isinstance(x, TextLabel) for x in m),
        "viewport": tool._viewport is not None,
    }
