# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
"""The Tape Measure tool's scenes and programs for
``make_modify_tool_fixtures.py`` (``tools/tape.py``, docs/rust-rewrite-plan.md
M3.13)."""
from __future__ import annotations

import math
import random

from _modify_harness import (ALT, CTRL, SHIFT, V, above, at, box, box_faces, click, dbl, deactivate, esc,  # noqa: F401
                             group, hover, key, keyup, press, redo, release, select, tap, typed, undo, xf,
                             Mesh, Scene, activate, camera, pbits, px)
from core.group import Group  # noqa: F401
from core.guide import Guide
from core.mesh import Edge
from tools.tape import TapeMeasureTool

NAME = "tape"
make_tool = TapeMeasureTool

CAM = dict(target=(2, 1, 1), distance=20.0, yaw=-50.0, pitch=30.0, w=1000, h=600)
CAM_ORTHO = dict(target=(2, 1, 1), distance=20.0, yaw=-50.0, pitch=30.0, w=1000, h=600, perspective=False)
CAM_LOW = dict(target=(2, 1, 1), distance=22.0, yaw=-25.0, pitch=10.0, w=1000, h=600)
CAM_TOP = dict(target=(2, 1, 0), distance=22.0, yaw=-90.0, pitch=89.0, w=1000, h=600)


def circle(m: Mesh, cx, cy, cz, r, n):
    pts = [V(cx + r * math.cos(2 * math.pi * k / n), cy + r * math.sin(2 * math.pi * k / n), cz) for k in range(n)]
    for k in range(n):
        m.add_edge(pts[k], pts[(k + 1) % n])
    m.tag_curve(pts, True)


def scn_edges() -> Scene:
    s = Scene()
    m = s.mesh
    m.add_edge(V(-3, 2, 0), V(3, 2, 0))
    m.add_edge(V(4, -2, 0), V(7, 2, 0))
    m.add_edge(V(-5, -3, 0), V(-5, -3, 3))
    box_faces(m, 8, -3, 0, 2, 2, 2)
    m.add_face([V(0, 5, 0), V(4, 5, 0), V(4, 8, 0), V(0, 8, 0)])
    circle(m, -3, 6, 0, 1.5, 16)
    return s


def scn_guides() -> Scene:
    s = Scene()
    m = s.mesh
    m.add_edge(V(-3, 2, 0), V(3, 2, 0))
    box_faces(m, 8, -3, 0, 2, 2, 2)
    s.guides.append(Guide(V(1, 4, 0), V(1, 0, 0)))
    s.guides.append(Guide(V(-2, -2, 0), V(0, 1, 0)))
    s.guides.append(Guide(V(3, -3, 0), V(1, 1, 0)))
    s.guides.append(Guide(V(5, 5, 0)))
    s.guides.append(Guide(V(-4, 5, 0), None, V(-4, 5, 2)))
    return s


def scn_groups() -> Scene:
    s = Scene()
    proto = box(0, 0, 0, 1, 1, 1)
    g0 = group(box(-5, 0, 0, 2, 2, 1.5), "classic")
    g1 = group(proto, "inst_a", xf(6, 0, 0))
    g2 = group(proto, "inst_b", xf(9, 2, 0, 30))
    s.groups.extend([g0, g1, g2])
    s.mesh.add_edge(V(0, 4, 0), V(4, 4, 0))
    s.mesh.add_edge(V(0, 4, 0), V(0, 4, 3))
    return s


def scn_axes() -> Scene:
    """Nothing near the origin: the model axes are the only sources."""
    s = Scene()
    s.mesh.add_face([V(6, 6, 0), V(7, 6, 0), V(7, 7, 0), V(6, 7, 0)])
    return s


def scn_units() -> Scene:
    s = Scene()
    s.units = {"length": "cm", "precision": 1, "decimal": ","}
    s.mesh.add_edge(V(-3, 2, 0), V(3, 2, 0))
    s.mesh.add_edge(V(4, -2, 0), V(7, 2, 0))
    return s


SCENES = {"edges": scn_edges, "guides": scn_guides, "groups": scn_groups, "axes": scn_axes, "units": scn_units}

PROGRAMS: list = []


def prog(scene, cam, *events):
    PROGRAMS.append((scene, cam, list(events)))


E1 = (0.9, 2, 0)          # body of the long edge
E1_END = (3, 2, 0)
E1_END2 = (-3, 2, 0)
E2 = (5.5, 0, 0)
E2_END = (4, -2, 0)
E3 = (-5, -3, 1.5)
E3_TOP = (-5, -3, 3)
BOXV = (8, -3, 1)          # a vertical box edge
BOX_TOP = (9, -3, 2)
BOX_CORNER = (10, -1, 2)
FREE = (2, -1, 0)
FREE2 = (3, -2, 0)
FREE3 = (-1, -2, 0)
FREE4 = (3.5, 5, 0)
PLATE_EDGE = (2, 5, 0)
PLATE_C = (2, 6.5, 0)
CIRC = (-3 + 1.5, 6, 0)
CIRC_SEG = (-3 + 1.5 * (math.cos(0) + math.cos(2 * math.pi / 16)) / 2, 6 + 1.5 * (math.sin(2 * math.pi / 16)) / 2, 0)

# measuring in free space
prog("edges", CAM, hover(FREE), click(FREE), hover(FREE2), hover((3, -2.5, 0)), click((3, -2.5, 0)), undo())
prog("edges", CAM, hover(FREE), click(FREE), hover((2, -3, 0)), click((2, -3, 0)), hover(FREE3), click(FREE3), hover(FREE2), click(FREE2))
# a guide line pulled from an edge body, by click and by typed distance
prog("edges", CAM, hover(E1), click(E1), hover((0.9, 3, 0)), hover((0.9, 3.7, 0)), click((0.9, 3.7, 0)), undo(), redo())
prog("edges", CAM, hover(E1), click(E1), hover((0.9, 0.5, 0)), click((0.9, 0.5, 0)))
prog("edges", CAM, hover(E1), click(E1), hover((0.9, 3, 0)), typed("1.5"), undo(), redo())
prog("edges", CAM, hover(E2), click(E2), hover((4.5, 1, 0)), typed("2"), hover(E2), click(E2), hover((6.5, -1, 0)), click((6.5, -1, 0)))
prog("edges", CAM, hover(E3), click(E3), hover((-4, -3, 1.5)), hover((-3.2, -3, 1.5)), click((-3.2, -3, 1.5)))
prog("edges", CAM, hover(E3), click(E3), hover((-5, -2, 1.5)), typed("0.8"), typed("1"))
prog("edges", CAM, hover(E1), click(E1), typed("1"), hover((0.9, 3, 0)), typed("-1"), typed("0"), typed("2;1"))
# a typed value with nothing started, a tuple, a unit
prog("edges", CAM, typed("1"), hover(E1), click(E1), hover((0.9, 3, 0)), typed("2m"), undo())
prog("edges", CAM, hover(E1), click(E1), hover((0.9, 3, 0)), typed("30cm"), typed("1" + "0" * 400))
prog("edges", CAM, hover(E1), click(E1), hover((0.9, 3, 0)), typed("1" + "0" * 400))
# a box edge, a plate edge, a circle, a vertical edge
prog("edges", CAM, hover(BOXV), click(BOXV), hover((9, -4, 1)), click((9, -4, 1)))
prog("edges", CAM, hover(BOX_TOP), click(BOX_TOP), hover((9, -3, 4)), typed("1.2"))
prog("edges", CAM, hover(PLATE_EDGE), click(PLATE_EDGE), hover((2, 3, 0)), click((2, 3, 0)))
prog("edges", CAM, hover(CIRC_SEG), click(CIRC_SEG), hover((-1, 4, 0)), click((-1, 4, 0)))
prog("edges", CAM, hover(PLATE_C), click(PLATE_C), hover((2, 7.5, 0)), click((2, 7.5, 0)))
# from a named point: a guide point with its segment, typed or clicked
prog("edges", CAM, hover(E1_END), click(E1_END), hover(FREE4), click(FREE4), undo(), redo())
prog("edges", CAM, hover(E1_END), click(E1_END), hover(FREE4), typed("1.25"), undo())
prog("edges", CAM, hover(E2_END), click(E2_END), hover((4, -4, 0)), hover((4, -5, 0)), click((4, -5, 0)))
prog("edges", CAM, hover(E3_TOP), click(E3_TOP), hover((-5, -3, 5)), typed("1"))
prog("edges", CAM, hover(BOX_CORNER), click(BOX_CORNER), hover((10, -1, 4)), click((10, -1, 4)))
prog("edges", CAM, hover(E1_END), click(E1_END), hover(E1_END2), click(E1_END2))
prog("edges", CAM, hover(E1_END), click(E1_END), hover(E1_END), typed("1"), click(E1_END))
prog("edges", CAM, hover(E1_END), click(E1_END), hover(E2_END), click(E2_END))
# the three Ctrl modes: guide lines, guide points, measure only
prog("edges", CAM, tap("ctrl"), hover(E1), click(E1), hover((0.9, 3, 0)), click((0.9, 3, 0)), undo(), redo())
prog("edges", CAM, tap("ctrl"), hover(E1), click(E1), hover((0.9, 3, 0)), typed("2"), tap("ctrl"), hover(E1), click(E1),
     hover((0.9, 3, 0)), click((0.9, 3, 0)))
prog("edges", CAM, tap("ctrl"), tap("ctrl"), hover(E1), click(E1), hover((0.9, 3, 0)), typed("1"), click((0.9, 3, 0)))
prog("edges", CAM, tap("ctrl"), tap("ctrl"), tap("ctrl"), hover(E1), click(E1), hover((0.9, 3, 0)), click((0.9, 3, 0)))
prog("edges", CAM, hover(E1_END), click(E1_END), tap("ctrl"), hover(FREE4), click(FREE4), hover(FREE), click(FREE), tap("ctrl"),
     hover(FREE2), click(FREE2))
prog("edges", CAM, hover(E1), click(E1), tap("ctrl"), hover((0.9, 3, 0)), typed("1.75"), undo())
prog("edges", CAM, hover(E1), click(E1), tap("ctrl"), tap("ctrl"), hover((0.9, 3, 0)), click((0.9, 3, 0)))
prog("edges", CAM, tap("ctrl"), tap("ctrl"), hover(E1_END), click(E1_END), hover(FREE4), click(FREE4))
prog("edges", CAM, tap("ctrl"), hover(FREE), click(FREE), hover(FREE2), typed("2.5"), hover(FREE3), click(FREE3))
prog("edges", CAM, tap("ctrl"), hover(FREE), click(FREE), hover(FREE), typed("2.5"), click(FREE))
# Ctrl used as part of a chord is not a tap; other keys; the arrow-key locks
prog("edges", CAM, key("ctrl", ctrl=True), key("right"), keyup("ctrl"), hover(E1), click(E1), hover((0.9, 3, 0)), click((0.9, 3, 0)))
prog("edges", CAM, key("ctrl", ctrl=True), keyup("ctrl"), key("ctrl", ctrl=True), key("up"), keyup("ctrl"), hover(E1), click(E1))
prog("edges", CAM, hover(FREE), click(FREE), key("right"), hover(FREE2), click(FREE2), key("left"), key("up"))
prog("edges", CAM, hover(FREE), click(FREE), key("shift"), hover(FREE2), keyup("shift"), click(FREE2))
# Esc, deactivate, camera changes mid-measure
prog("edges", CAM, hover(E1), click(E1), hover((0.9, 3, 0)), esc(), hover((0.9, 3, 0)), click((0.9, 3, 0)))
prog("edges", CAM, hover(E1), click(E1), hover((0.9, 3, 0)), typed("1"), esc(), esc())
prog("edges", CAM, tap("ctrl"), hover(E1), click(E1), deactivate(), activate(), hover((0.9, 3, 0)), click((0.9, 3, 0)))
prog("edges", CAM, tap("ctrl"), deactivate(), activate(), hover(E1), click(E1), hover((0.9, 3, 0)), click((0.9, 3, 0)))
prog("edges", CAM, hover(E1), click(E1), hover((0.9, 3, 0)), camera(dict(CAM, yaw=-20.0)), hover((0.9, 3.4, 0)), click((0.9, 3.4, 0)))
# cameras
prog("edges", CAM_TOP, hover(E1), click(E1), hover((0.9, 3, 0)), click((0.9, 3, 0)), hover(E2), click(E2), hover((6, -1, 0)), typed("0.7"))
prog("edges", CAM_ORTHO, hover(E1), click(E1), hover((0.9, 3, 0)), click((0.9, 3, 0)), hover(FREE), click(FREE), hover(FREE2), click(FREE2))
prog("edges", CAM_LOW, hover(E1), click(E1), hover((0.9, 3, 0)), typed("1.1"), hover(FREE), click(FREE), hover(FREE4), click(FREE4))
# existing guides: a guide line is a source too; a guide point is not
G1 = (-1, 4, 0)
G2 = (-2, 0, 0)
G3 = (4, -2, 0)
GP = (5, 5, 0)
GPO = (-4, 5, 0)
prog("guides", CAM, hover(G1), click(G1), hover((-1, 6, 0)), click((-1, 6, 0)), undo(), undo(), redo())
prog("guides", CAM, hover(G2), click(G2), hover((-1, 0, 0)), typed("1.5"))
prog("guides", CAM, hover(G3), click(G3), hover((4, 0, 0)), hover((5, -1, 0)), click((5, -1, 0)))
prog("guides", CAM, hover(GP), click(GP), hover((6, 6, 0)), click((6, 6, 0)))
prog("guides", CAM, hover(GPO), click(GPO), hover((-3, 7, 0)), click((-3, 7, 0)))
prog("guides", CAM, hover(E1), click(E1), hover((0.9, 3, 0)), click((0.9, 3, 0)), hover(G1), click(G1), hover((-1, 5, 0)), click((-1, 5, 0)))
prog("guides", CAM, tap("ctrl"), hover(G1), click(G1), hover((-1, 6, 0)), click((-1, 6, 0)))
prog("guides", CAM, tap("ctrl"), tap("ctrl"), hover(G1), click(G1), hover((-1, 6, 0)), click((-1, 6, 0)))
prog("guides", CAM, hover(G1), click(G1), hover((-1, 4.0, 0)), typed("1"), typed("-1"))
# groups: a group's edge is a source too (pick_edge_any)
GE1 = (-4, 0, 0.75)
GE2 = (6.5, 0, 0)
GE3 = (9.4, 1.6, 0.5)
LE1 = (2, 4, 0)
LE2 = (0, 4, 1.5)
prog("groups", CAM, hover(GE1), click(GE1), hover((-4, -2, 0.75)), click((-4, -2, 0.75)), undo(), redo())
prog("groups", CAM, hover(GE2), click(GE2), hover((6.5, -2, 0)), typed("1.5"))
prog("groups", CAM, hover(GE3), click(GE3), hover((9.4, 3, 0.5)), hover((9.4, 4, 0.5)), click((9.4, 4, 0.5)))
prog("groups", CAM, hover(LE1), click(LE1), hover((2, 6, 0)), click((2, 6, 0)), hover(LE2), click(LE2), hover((2, 4, 1.5)), click((2, 4, 1.5)))
prog("groups", CAM, tap("ctrl"), hover(GE1), click(GE1), hover((-4, -2, 0.75)), click((-4, -2, 0.75)))
prog("groups", CAM, hover((-5, 0, 0)), click((-5, 0, 0)), hover((-6, -1, 0)), click((-6, -1, 0)))
# the model axes are sources; nothing is drawn there
AX = (3, 0, 0)
AY = (0, 3, 0)
AZ = (0, 0, 3)
prog("axes", CAM, hover(AX), click(AX), hover((3, 2, 0)), click((3, 2, 0)), undo())
prog("axes", CAM, hover(AY), click(AY), hover((2, 3, 0)), typed("1.5"))
prog("axes", CAM, hover(AZ), click(AZ), hover((2, 0, 3)), click((2, 0, 3)))
prog("axes", CAM, hover((0, 0, 0)), click((0, 0, 0)), hover((3, 3, 0)), click((3, 3, 0)))
prog("axes", CAM, hover((0, 0, 0)), click((0, 0, 0)), hover((3, 3, 0)), typed("2"))
prog("axes", CAM, tap("ctrl"), hover(AX), click(AX), hover((3, 2, 0)), click((3, 2, 0)))
prog("axes", CAM, tap("ctrl"), tap("ctrl"), hover(AX), click(AX), hover((3, 2, 0)), click((3, 2, 0)))
prog("axes", CAM_TOP, hover(AX), click(AX), hover((3, 2, 0)), click((3, 2, 0)), hover(AY), click(AY), hover((2, 3, 0)), click((2, 3, 0)))
prog("axes", CAM_ORTHO, hover((4, 0.01, 0)), click((4, 0.01, 0)), hover((4, 3, 0)), click((4, 3, 0)))
prog("axes", CAM_LOW, hover(AX), click(AX), hover((3, 2, 0)), click((3, 2, 0)))
prog("axes", CAM, hover((2, 0, 0)), click((2, 0, 0)), hover(FREE), click(FREE), hover(AX), click(AX), hover((3, 1, 0)), click((3, 1, 0)))
# the document's units (cm, one decimal, a comma)
prog("units", CAM, hover(E1), click(E1), hover((0.9, 3, 0)), click((0.9, 3, 0)), hover(FREE), click(FREE), hover(FREE2), click(FREE2))
prog("units", CAM, hover(E1_END), click(E1_END), hover(FREE4), click(FREE4), hover(E1_END2), typed("2"))
prog("units", CAM, hover(E1), click(E1), hover((0.9, 3, 0)), typed("37.5"))


# ---- seeded random programs --------------------------------------------------------------------

POINTS = {
    "edges": [E1, E1_END, E1_END2, E2, E2_END, E3, E3_TOP, BOXV, BOX_TOP, BOX_CORNER, FREE, FREE2, FREE3, FREE4,
              PLATE_EDGE, PLATE_C, CIRC_SEG, (0.9, 3, 0), (-1, 4, 0), AX],
    "guides": [E1, E1_END, G1, G2, G3, GP, GPO, (-1, 6, 0), (4, 0, 0), FREE, FREE4, BOXV],
    "groups": [GE1, GE2, GE3, LE1, LE2, (-4, -2, 0.75), (2, 6, 0), (9.4, 4, 0.5), FREE],
    "axes": [AX, AY, AZ, (0, 0, 0), (3, 2, 0), (2, 3, 0), (3, 3, 0), (4, 0.01, 0), (2, 0, 3)],
    "units": [E1, E1_END, E1_END2, E2, E2_END, FREE, FREE2, FREE4, (0.9, 3, 0)],
}


def random_program(seed: int, scene: str, cam: dict, n: int = 26):
    rnd = random.Random(seed)
    pts = POINTS[scene]
    evs = []
    for _ in range(n):
        r = rnd.random()
        p = rnd.choice(pts)
        q = (p[0] + rnd.choice((0, 0, 0.4, -0.5)), p[1] + rnd.choice((0, 0, 0.6, -0.3)), p[2])
        if r < 0.30:
            evs.append(hover(q))
        elif r < 0.60:
            evs.append(hover(p))
            evs.append(click(p))
        elif r < 0.67:
            evs.append(tap("ctrl"))
        elif r < 0.78:
            evs.append(typed(rnd.choice(["1", "2.5", "-1", "0.3", "0", "1m", "20cm", "3;4", "1.75", "12"])))
        elif r < 0.84:
            evs.append(esc())
        elif r < 0.89:
            evs.append(undo())
        elif r < 0.92:
            evs.append(redo())
        elif r < 0.96:
            evs.append(key(rnd.choice(["right", "left", "up"])))
        elif r < 0.98:
            evs.append(deactivate())
            evs.append(activate())
        else:
            evs.append(click(q))
    return (scene, cam, evs)


for _seed in range(700, 750):
    _sc = ("edges", "guides", "groups", "axes", "units")[_seed % 5]
    PROGRAMS.append(random_program(_seed, _sc, (CAM, CAM_ORTHO, CAM_LOW, CAM_TOP)[(_seed // 5) % 4]))


# ---- probe --------------------------------------------------------------------------------------------

def extra_probe(tool, vp, ids):
    e = tool._edge
    if e is None:
        kind = None
    elif isinstance(e, Guide):
        kind = "guide" if any(e is g for g in vp.scene.guides) else "axis"
    elif isinstance(e, Edge):
        kind = "loose"
    else:
        kind = "group"
    return {
        "mode": tool._mode,
        "from_point": bool(tool._from_point),
        "edge": kind,
        "edge_a": None if e is None else pbits(e.a),
        "measured": tool._measured,
        "n_guides": len(vp.scene.guides),
    }
