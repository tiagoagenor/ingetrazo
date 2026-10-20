# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
"""The Scale tool's scenes and programs for ``make_modify_tool_fixtures.py``
(``tools/scale.py``, docs/rust-rewrite-plan.md M3.13)."""
from __future__ import annotations

import math
import random
import struct

from _modify_harness import (ALT, CTRL, SHIFT, V, above, at, box, box_faces, click, dbl, deactivate, esc,  # noqa: F401
                             group, hover, key, keyup, press, redo, release, select, tap, typed, undo, xf,
                             Mesh, Scene, activate, camera, pbits, px)
from core.vecmath import QMatrix4x4, QVector3D
from tools.scale import ScaleTool

NAME = "scale"
make_tool = ScaleTool

CAM = dict(target=(2, 2, 0.8), distance=14.0, yaw=-50.0, pitch=30.0, w=1000, h=600)
CAM_ORTHO = dict(target=(2, 2, 0.8), distance=14.0, yaw=-50.0, pitch=30.0, w=1000, h=600, perspective=False)
CAM_LOW = dict(target=(2, 2, 1.0), distance=16.0, yaw=-30.0, pitch=8.0, w=1000, h=600)
CAM_TOP = dict(target=(2, 2, 0.0), distance=16.0, yaw=-90.0, pitch=89.0, w=1000, h=600)
CAM_FAR = dict(target=(3, 4, 0.8), distance=26.0, yaw=-40.0, pitch=35.0, w=1000, h=600)


def scn_cube() -> Scene:
    s = Scene()
    box_faces(s.mesh, 0, 0, 0, 3, 3, 3)                       # faces 0..5, edges 0..11
    s.mesh.add_face([V(6, 0, 0), V(8, 0, 0), V(8, 2, 0), V(6, 2, 0)])    # face 6, edges 12..15
    s.mesh.add_edge(V(10, 0, 0), V(10, 2, 1))                           # edge 16
    return s


def scn_plate() -> Scene:
    s = Scene()
    m = s.mesh
    m.add_face([V(0, 0, 0), V(4, 0, 0), V(4, 4, 0), V(0, 4, 0)])
    m.add_face([V(6, 0, 0), V(9, 0, 0), V(9, 2, 0), V(6, 2, 0)])
    m.add_edge(V(-3, 0, 0), V(-3, 2, 1))                               # edge 8
    return s


def scn_wall() -> Scene:
    """Flat in y (a wall), and a line along x (one active axis)."""
    s = Scene()
    m = s.mesh
    m.add_face([V(0, 0, 0), V(4, 0, 0), V(4, 0, 3), V(0, 0, 3)])
    m.add_face([V(4, 0, 0), V(7, 0, 0), V(7, 0, 2), V(4, 0, 2)])
    m.add_edge(V(0, 5, 0), V(4, 5, 0))                                  # edge 7
    m.add_edge(V(6, 5, 0), V(6, 5, 2))                                  # edge 8
    return s


def scn_mixed() -> Scene:
    """A face with a hole, a circle (a curve), a soft edge, a textured face."""
    s = Scene()
    m = s.mesh
    m.add_face([V(0, 0, 0), V(6, 0, 0), V(6, 6, 0), V(0, 6, 0)],
               [[V(2, 2, 0), V(2, 4, 0), V(4, 4, 0), V(4, 2, 0)]])     # face 0
    pts = [V(10 + 1.5 * math.cos(2 * math.pi * i / 12), 3 + 1.5 * math.sin(2 * math.pi * i / 12), 0.0)
           for i in range(12)]
    f = m.add_face(pts)                                                 # face 1
    m.tag_curve(pts)
    f.attrs["color"] = [0.9, 0.3, 0.2]
    q = m.add_face([V(0, 8, 0), V(3, 8, 0), V(3, 8, 2), V(0, 8, 2)])    # face 2
    q.attrs["texture"] = {"name": "t", "uvw": [0.5, 0.0, 0.0, 0.1, 0.0, 0.0, 0.5, 0.2]}
    for e in m.edges:
        if abs(e.v0.position.x() - 3.0) < 1e-6 and abs(e.v1.position.x() - 3.0) < 1e-6 and e.v0.position.y() == 8.0:
            e.soft = True
    m.add_edge(V(-3, 1, 0), V(-3, 5, 2))
    return s


def scn_groups() -> Scene:
    s = Scene()
    proto = box(0, 0, 0, 1, 1, 1, (0.2, 0.6, 0.9))
    g0 = group(box(-5, 0, 0, 2, 2, 1.5), "classic")
    g1 = group(proto, "inst_a", xf(6, 0, 0))
    g2 = group(proto, "inst_b", xf(9, 2, 0, 30))
    g3 = group(box(0, 8, 0, 2, 1, 1), "axed")
    g3.axes = xf(0, 8, 0, 25)
    s.groups.extend([g0, g1, g2, g3])
    s.mesh.add_face([V(0, 0, 0), V(4, 0, 0), V(4, 4, 0), V(0, 4, 0)])
    return s


def scn_container() -> Scene:
    s = Scene()
    p = box(0, 0, 0, 1, 1, 1)
    k1 = group(p, "kid1", xf(1, 0, 0, 20))
    k2 = group(p, "kid2", xf(0, 2, 0))
    ctr = group(Mesh(), "container", xf(6, 1, 0, 10), children=[k1, k2])
    cls = group(box(0, 6, 0, 2, 2, 2), "plain")
    s.groups.extend([ctr, cls])
    s.mesh.add_face([V(0, 0, 0), V(3, 0, 0), V(3, 3, 0), V(0, 3, 0)])
    return s


SCENES = {"cube": scn_cube, "plate": scn_plate, "wall": scn_wall, "mixed": scn_mixed, "groups": scn_groups,
          "container": scn_container}

PROGRAMS: list = []


def prog(scene, cam, *events):
    PROGRAMS.append((scene, cam, list(events)))


def G(lo, hi, tx, ty, tz):
    return tuple(lo[i] + (hi[i] - lo[i]) * t for i, t in enumerate((tx, ty, tz)))


CUBE = [("face", i) for i in range(6)]
CLO, CHI = (0, 0, 0), (3, 3, 3)


def cg(tx, ty, tz):
    return G(CLO, CHI, tx, ty, tz)


# ---- the cube: 26 grips ------------------------------------------------------------------------
# corner, click-move-click, undo/redo
prog("cube", CAM, select(*CUBE), hover(cg(1, 1, 1)), hover(at(cg(1, 1, 1), 2, 1)), click(cg(1, 1, 1)),
     hover((4, 4, 4)), hover((5, 5, 5)), click((5, 5, 5)), undo(), redo(), undo())
# a face grip (one axis)
prog("cube", CAM, select(*CUBE), hover(cg(1, .5, .5)), click(cg(1, .5, .5)), hover((4.5, 1.5, 1.5)),
     hover((6, 1.5, 1.5)), click((6, 1.5, 1.5)))
prog("cube", CAM, select(*CUBE), hover(cg(.5, .5, 1)), click(cg(.5, .5, 1)), hover((1.5, 1.5, 4)), click((1.5, 1.5, 5)))
# an edge grip (two axes)
prog("cube", CAM, select(*CUBE), hover(cg(1, 1, .5)), click(cg(1, 1, .5)), hover((4, 4, 1.5)), hover((5, 4.5, 1.5)),
     click((5, 4.5, 1.5)), undo())
prog("cube", CAM, select(*CUBE), hover(cg(.5, 0, 1)), click(cg(.5, 0, 1)), hover((1.5, -1, 4)), click((1.5, -1, 4)))
# drag and release
prog("cube", CAM, select(*CUBE), hover(cg(1, 1, 1)), press(cg(1, 1, 1)), hover((3.5, 3.5, 3.5), drag=True),
     hover((5, 5, 5), drag=True), release((5, 5, 5)))
prog("cube", CAM, select(*CUBE), hover(cg(1, .5, .5)), press(cg(1, .5, .5)), hover((5, 1.5, 1.5), drag=True),
     release((5, 1.5, 1.5)), undo(), redo())
prog("cube", CAM, select(*CUBE), hover(cg(1, 1, 1)), press(cg(1, 1, 1)), hover(at(cg(1, 1, 1), 2, 0), drag=True),
     release(at(cg(1, 1, 1), 2, 0)), hover((4, 4, 4)), click((4, 4, 4)))
# About Center (a Ctrl tap) before and after the grab; Scale Uniformly (a Shift tap)
prog("cube", CAM, select(*CUBE), tap("ctrl"), hover(cg(1, 1, 1)), click(cg(1, 1, 1)), hover((4, 4, 4)), click((4, 4, 4)))
prog("cube", CAM, select(*CUBE), hover(cg(1, 1, 1)), click(cg(1, 1, 1)), hover((4, 4, 4)), tap("ctrl"), hover((4.5, 4.5, 4.5)),
     tap("ctrl"), hover((5, 5, 5)), click((5, 5, 5)))
prog("cube", CAM, select(*CUBE), tap("shift"), hover(cg(1, 1, .5)), click(cg(1, 1, .5)), hover((4, 4, 1.5)),
     hover((5, 5, 1.5)), tap("shift"), hover((5, 4, 1.5)), click((5, 4, 1.5)))
prog("cube", CAM, select(*CUBE), tap("shift"), hover(cg(1, .5, .5)), click(cg(1, .5, .5)), hover((5, 1.5, 1.5)),
     click((5, 1.5, 1.5)))
prog("cube", CAM, select(*CUBE), tap("ctrl"), tap("shift"), hover(cg(.5, 1, .5)), click(cg(.5, 1, .5)), hover((1.5, 5, 1.5)),
     click((1.5, 5, 1.5)))
# past the anchor (a mirror) and a factor too small
prog("cube", CAM, select(*CUBE), hover(cg(1, .5, .5)), click(cg(1, .5, .5)), hover((1, 1.5, 1.5)), hover((-1, 1.5, 1.5)),
     click((-2, 1.5, 1.5)), undo())
prog("cube", CAM, select(*CUBE), hover(cg(1, .5, .5)), click(cg(1, .5, .5)), hover((1, 1.5, 1.5)),
     hover((0.0, 1.5, 1.5)), hover((0.0002, 1.5, 1.5)), click((0.0, 1.5, 1.5)))
prog("cube", CAM, select(*CUBE), hover(cg(1, 1, 1)), click(cg(1, 1, 1)), hover((-1, -1, -1)), click((-1, -1, -1)))
# typed factors
prog("cube", CAM, select(*CUBE), hover(cg(1, 1, 1)), click(cg(1, 1, 1)), hover((4, 4, 4)), typed("2"), undo(), redo())
prog("cube", CAM, select(*CUBE), hover(cg(1, 1, 1)), click(cg(1, 1, 1)), typed("1;2;3"), typed("2;3;4"))
prog("cube", CAM, select(*CUBE), hover(cg(1, 1, .5)), click(cg(1, 1, .5)), typed("2;3"), typed("1;2;3"), undo())
prog("cube", CAM, select(*CUBE), hover(cg(1, .5, .5)), click(cg(1, .5, .5)), typed("2;3"), typed("3"))
prog("cube", CAM, select(*CUBE), hover(cg(1, .5, .5)), click(cg(1, .5, .5)), typed("6m"))
prog("cube", CAM, select(*CUBE), hover(cg(1, 1, 1)), click(cg(1, 1, 1)), typed("1.5m"), typed("2m"))
prog("cube", CAM, select(*CUBE), hover(cg(1, 1, .5)), click(cg(1, 1, .5)), typed("4m;5m"))
prog("cube", CAM, select(*CUBE), hover(cg(1, 1, 1)), click(cg(1, 1, 1)), typed("-1"))
prog("cube", CAM, select(*CUBE), hover(cg(1, 1, 1)), click(cg(1, 1, 1)), typed("0"), typed("0.00001"), typed("1"), typed("abc"))
prog("cube", CAM, select(*CUBE), hover(cg(1, .5, .5)), click(cg(1, .5, .5)), hover((5, 1.5, 1.5)), typed("-2"))
# the hot retype after a commit
prog("cube", CAM, select(*CUBE), hover(cg(1, 1, 1)), click(cg(1, 1, 1)), typed("2"), typed("3"), typed("0.5"), undo())
prog("cube", CAM, select(*CUBE), hover(cg(1, 1, .5)), click(cg(1, 1, .5)), typed("2;3"), typed("1;4"), typed("5m;2m"), typed("2"))
prog("cube", CAM, select(*CUBE), hover(cg(1, .5, .5)), click(cg(1, .5, .5)), typed("2"), typed("4m"), click(px(5, 5)), typed("3"))
prog("cube", CAM, select(*CUBE), hover(cg(1, 1, 1)), click(cg(1, 1, 1)), typed("2"), undo(), typed("3"))
prog("cube", CAM, select(*CUBE), hover(cg(1, 1, 1)), click(cg(1, 1, 1)), typed("2"), tap("ctrl"), typed("3"))
# Esc mid-way, cancels the preview; Esc again clears the selection
prog("cube", CAM, select(*CUBE), hover(cg(1, 1, 1)), click(cg(1, 1, 1)), hover((5, 5, 5)), esc(), hover((5, 5, 5)),
     esc(), hover(cg(1, 1, 1)))
prog("cube", CAM, select(*CUBE), hover(cg(1, 1, 1)), click(cg(1, 1, 1)), hover((5, 5, 5)), tap("ctrl"), esc(), hover(cg(1, 1, 1)))
# a grip snapped to named points of other objects
prog("cube", CAM, select(*CUBE), hover(cg(1, .5, .5)), click(cg(1, .5, .5)), hover((6, 0, 0)), hover((8, 2, 0)),
     hover((7, 0, 0)), click((8, 0, 0)), undo())
prog("cube", CAM, select(*CUBE), hover(cg(1, 1, 1)), click(cg(1, 1, 1)), hover((6, 2, 0)), hover((8, 2, 0)), click((8, 2, 0)))
prog("cube", CAM, select(*CUBE), hover(cg(1, 1, .5)), click(cg(1, 1, .5)), hover((8, 2, 0)), hover((10, 2, 1)), click((10, 2, 1)))
prog("cube", CAM, select(*CUBE), hover(cg(.5, .5, 1)), click(cg(.5, .5, 1)), hover((10, 2, 1)), hover((10, 0, 0)),
     click((10, 2, 1)))
prog("cube", CAM, select(*CUBE), hover(cg(1, .5, .5)), click(cg(1, .5, .5)), hover((7, 1, 0)), hover((9, 1, 0.5)),
     hover((10, 1, 0.5)), click((10, 1, 0.5)))
# the tool put down mid-scale and picked up again; a camera change; no grips at all
prog("cube", CAM, select(*CUBE), hover(cg(1, 1, 1)), click(cg(1, 1, 1)), hover((5, 5, 5)), deactivate(), activate(),
     hover(cg(1, 1, 1)))
prog("cube", CAM, select(*CUBE), hover(cg(1, 1, 1)), click(cg(1, 1, 1)), hover((4, 4, 4)), camera(dict(CAM, yaw=-20.0)),
     hover((5, 5, 5)), click((5, 5, 5)))
prog("cube", CAM, select(*CUBE), hover(cg(1, 1, 1)), camera(dict(CAM, yaw=-20.0)), hover(cg(1, 1, 1)), click(cg(1, 1, 1)))
# click-to-select with nothing selected
prog("cube", CAM, hover((1.5, 1.5, 3)), click((1.5, 1.5, 3)), hover((3, 0, 3)), click((3, 0, 3)), hover((4, 0, 3)),
     click((4, 0, 3)))
prog("cube", CAM, click((7, 1, 0)), hover((8, 2, 0)), click((8, 2, 0)), hover((9, 0, 0)), click((9, 0, 0)), esc())
prog("cube", CAM, click(px(5, 5)), hover(px(6, 6)), esc(), typed("2"))
prog("cube", CAM, hover((10, 1, 0.5)), click((10, 1, 0.5)), hover((10, 2, 1)), click((10, 2, 1)), hover((10, 0, 0)),
     click((10, 0, 0)), undo())
prog("cube", CAM, select(("face", 6)), hover((7, 1, 0)), click((7, 1, 0)), hover((9, 3, 0)), click((9, 3, 0)), undo(), redo())
prog("cube", CAM, select(("edge", 16)), hover((10, 1, .5)), click((10, 1, .5)), hover((11, 1, .5)), click((11, 1, .5)))
prog("cube", CAM, select(("face", 6), ("edge", 16)), hover((7, 1, 0)), click((7, 1, 0)), hover((12, 3, 0)), click((12, 3, 0)),
     undo())
prog("cube", CAM, select(("face", 1), ("face", 5)), hover((1.5, 1.5, 3)), click((1.5, 1.5, 3)), tap("ctrl"),
     hover((3, 3, 4)), click((3, 3, 4)))
prog("cube", CAM_ORTHO, select(*CUBE), hover(cg(1, 1, 1)), click(cg(1, 1, 1)), hover((4, 4, 4)), click((4, 4, 4)), undo())
prog("cube", CAM_LOW, select(*CUBE), hover(cg(1, .5, .5)), click(cg(1, .5, .5)), hover((5, 1.5, 1.5)), click((5, 1.5, 1.5)))
prog("cube", CAM_TOP, select(*CUBE), hover(cg(1, 1, .5)), click(cg(1, 1, .5)), hover((5, 5, 1.5)), click((5, 5, 1.5)))
prog("cube", CAM_FAR, select(*CUBE), hover(cg(0, 0, 0)), press(cg(0, 0, 0)), hover((-2, -2, -2), drag=True), release((-2, -2, -2)))

# ---- a flat selection: 8 grips -----------------------------------------------------------------------
PLO, PHI = (0, 0, 0), (4, 4, 0)


def pg(tx, ty):
    return G(PLO, PHI, tx, ty, 0)


prog("plate", CAM, select(("face", 0)), hover(pg(1, 1)), click(pg(1, 1)), hover((5, 5, 0)), hover((6, 6, 0)), click((6, 6, 0)),
     undo(), redo())
prog("plate", CAM, select(("face", 0)), hover(pg(1, .5)), click(pg(1, .5)), hover((6, 2, 0)), click((6, 2, 0)))
prog("plate", CAM, select(("face", 0)), hover(pg(.5, 0)), click(pg(.5, 0)), hover((2, -2, 0)), click((2, -2, 0)))
prog("plate", CAM, select(("face", 0)), hover(pg(0, 0)), press(pg(0, 0)), hover((-1, -1, 0), drag=True), release((-1, -1, 0)))
prog("plate", CAM_TOP, select(("face", 0)), hover(pg(1, 1)), click(pg(1, 1)), hover((6, 6, 0)), click((6, 6, 0)))
prog("plate", CAM_TOP, select(("face", 0)), tap("ctrl"), hover(pg(1, .5)), click(pg(1, .5)), hover((6, 2, 0)), click((6, 2, 0)))
prog("plate", CAM_TOP, select(("face", 0)), hover(pg(1, 1)), click(pg(1, 1)), typed("2;3"), typed("1;2;3"))
prog("plate", CAM, select(("face", 0)), hover(pg(1, 1)), click(pg(1, 1)), typed("8m"), typed("2"))
prog("plate", CAM, select(("face", 0)), tap("shift"), hover(pg(1, .5)), click(pg(1, .5)), hover((6, 2, 0)), click((6, 2, 0)))
prog("plate", CAM, select(("face", 0), ("face", 1)), hover((0, 0, 0)), click((0, 0, 0)), hover((10, 5, 0)), click((10, 5, 0)),
     undo())
prog("plate", CAM, select(("edge", 8)), hover((-3, 2, 1)), click((-3, 2, 1)), hover((-3, 4, 2)), click((-3, 4, 2)))
prog("plate", CAM, select(("edge", 8)), hover((-3, 0, 0)), click((-3, 0, 0)), hover((-3, -2, 0)), click((-3, -2, 0)))
prog("plate", CAM, select(("face", 0), ("edge", 8)), hover((-3, 0, 0)), click((-3, 0, 0)), hover((-5, 0, 0)), click((-5, 0, 0)))
prog("plate", CAM, hover((2, 2, 0)), click((2, 2, 0)), hover(pg(1, 1)), click(pg(1, 1)), hover((5, 5, 0)), click((5, 5, 0)))
prog("plate", CAM, hover((2, 0, 0)), click((2, 0, 0)), hover((2, 0.1, 0)), esc(), esc())

# ---- walls and lines (flat in y, a single axis) -----------------------------------------------------------
prog("wall", CAM, select(("face", 0), ("face", 1)), hover((0, 0, 0)), click((0, 0, 0)), hover((10, 0, 4)), click((10, 0, 4)),
     undo())
prog("wall", CAM, select(("face", 0)), hover((2, 0, 3)), click((2, 0, 3)), hover((2, 0, 5)), click((2, 0, 5)))
prog("wall", CAM, select(("face", 0)), hover((4, 0, 1.5)), click((4, 0, 1.5)), hover((6, 0, 1.5)), typed("2"))
prog("wall", CAM, select(("edge", 8)), hover((4, 5, 0)), click((4, 5, 0)), hover((6, 5, 0)), click((6, 5, 0)), undo())
prog("wall", CAM, select(("edge", 8)), tap("ctrl"), hover((0, 5, 0)), click((0, 5, 0)), hover((-2, 5, 0)), click((-2, 5, 0)))
prog("wall", CAM, select(("edge", 9)), hover((6, 5, 2)), click((6, 5, 2)), hover((6, 5, 4)), typed("3"), typed("2"))
prog("wall", CAM, select(("edge", 8)), hover((4, 5, 0)), click((4, 5, 0)), typed("2m"), typed("3"))
prog("wall", CAM_TOP, select(("edge", 8), ("edge", 9)), hover((6, 5, 2)), click((6, 5, 2)), hover((8, 5, 2)), click((8, 5, 2)))

# ---- faces with holes, curves, soft edges, a texture ----------------------------------------------------
prog("mixed", CAM, select(("face", 0)), hover((6, 6, 0)), click((6, 6, 0)), hover((9, 9, 0)), click((9, 9, 0)), undo(), redo())
prog("mixed", CAM, select(("face", 1)), hover((11.5, 4.5, 0)), click((11.5, 4.5, 0)), hover((13, 6, 0)), click((13, 6, 0)))
prog("mixed", CAM, select(("face", 2)), hover((3, 8, 2)), click((3, 8, 2)), hover((5, 8, 4)), click((5, 8, 4)), undo())
prog("mixed", CAM, select(("face", 2)), hover((1.5, 8, 2)), click((1.5, 8, 2)), hover((1.5, 8, 4)), click((1.5, 8, 4)))
prog("mixed", CAM, select(("face", 0), ("face", 1), ("face", 2)), hover((0, 0, 0)), click((0, 0, 0)), hover((-1, -1, 0)),
     click((-1, -1, 0)), undo(), redo())
prog("mixed", CAM, select(("face", 0), ("face", 1), ("face", 2)), tap("ctrl"), hover((0, 0, 0)), click((0, 0, 0)),
     typed("0.5"), typed("2"))
prog("mixed", CAM, hover((1, 3, 0)), click((1, 3, 0)), hover((6, 3, 0)), click((6, 3, 0)), hover((7, 7, 0)), click((7, 7, 0)))

# ---- groups: classic, instances, own axes ---------------------------------------------------------------
GC = ((-5, 0, 0), (-3, 2, 1.5))
prog("groups", CAM, select(("group", [0])), hover(G(*GC, 1, 1, 1)), click(G(*GC, 1, 1, 1)), hover((-2, 3, 2.5)),
     click((-2, 3, 2.5)), undo(), redo())
prog("groups", CAM, select(("group", [0])), hover(G(*GC, 1, .5, .5)), click(G(*GC, 1, .5, .5)), hover((-1, 1, .75)),
     click((-1, 1, .75)))
prog("groups", CAM, select(("group", [0])), hover(G(*GC, 1, 1, .5)), click(G(*GC, 1, 1, .5)), typed("2;3"), typed("1;1"))
prog("groups", CAM, select(("group", [0])), tap("ctrl"), hover(G(*GC, 1, 1, 1)), press(G(*GC, 1, 1, 1)),
     hover((-2, 3, 2.5), drag=True), release((-2, 3, 2.5)))
prog("groups", CAM, select(("group", [1])), hover((7, 1, 1)), click((7, 1, 1)), hover((8, 2, 2)), click((8, 2, 2)), undo())
prog("groups", CAM, select(("group", [1])), hover((7, .5, .5)), click((7, .5, .5)), hover((9, .5, .5)), click((9, .5, .5)))
prog("groups", CAM, select(("group", [1])), hover((7, 1, 1)), click((7, 1, 1)), typed("2"), typed("3"), undo(), redo())
prog("groups", CAM, select(("group", [1])), hover((7, 1, 1)), click((7, 1, 1)), typed("2m"))
prog("groups", CAM, select(("group", [1]), ("group", [0])), hover((-5, 0, 0)), click((-5, 0, 0)), hover((-8, -3, -1)),
     click((-8, -3, -1)), undo())
# the instance rotated 30 degrees: the box on its OWN axes
Bm = xf(9, 2, 0, 30)


def bgrip(tx, ty, tz):
    p = Bm.map(QVector3D(*G((0, 0, 0), (1, 1, 1), tx, ty, tz)))
    return (p.x(), p.y(), p.z())


prog("groups", CAM, select(("group", [2])), hover(bgrip(1, 1, 1)), click(bgrip(1, 1, 1)), hover(bgrip(2, 2, 2)),
     hover(bgrip(3, 3, 3)), click(bgrip(3, 3, 3)), undo(), redo())
prog("groups", CAM, select(("group", [2])), hover(bgrip(1, .5, .5)), click(bgrip(1, .5, .5)), hover(bgrip(2.5, .5, .5)),
     click(bgrip(2.5, .5, .5)))
prog("groups", CAM, select(("group", [2])), hover(bgrip(1, 1, .5)), click(bgrip(1, 1, .5)), hover(bgrip(2, 2, .5)),
     click(bgrip(2, 2, .5)))
prog("groups", CAM, select(("group", [2])), hover(bgrip(1, 1, .5)), click(bgrip(1, 1, .5)), typed("2;3"), typed("1;2"))
prog("groups", CAM, select(("group", [2])), tap("ctrl"), hover(bgrip(.5, 1, .5)), press(bgrip(.5, 1, .5)),
     hover(bgrip(.5, 2, .5), drag=True), release(bgrip(.5, 2, .5)))
prog("groups", CAM, select(("group", [2])), hover(bgrip(1, 1, 1)), click(bgrip(1, 1, 1)), typed("2"), typed("3"),
     undo())
# the classic group with its own axes (the box is measured in those axes)
class GroupBox:
    """The box the tool builds on one selected group, computed the way the
    tool does, so a program can aim at its grips."""

    def __init__(self, builder, path):
        from core.group import frame_axes, group_frame, iter_placements
        sc = builder()
        g = sc.groups[path[0]]
        for i in path[1:]:
            g = g.children[i]
        frame = group_frame(g)
        if frame is None:
            self.o = QVector3D(0, 0, 0)
            self.ax = (QVector3D(1, 0, 0), QVector3D(0, 1, 0), QVector3D(0, 0, 1))
        else:
            o, x, y, z = frame_axes(frame)
            self.o, self.ax = o, (x, y, z)
        lo = [float("inf")] * 3
        hi = [float("-inf")] * 3
        for pg, m in iter_placements(g):
            for v in pg.mesh.vertices:
                p = m.map(v.position) if m is not None else v.position
                d = p - self.o
                loc = (QVector3D.dotProduct(d, self.ax[0]), QVector3D.dotProduct(d, self.ax[1]),
                       QVector3D.dotProduct(d, self.ax[2]))
                for i in range(3):
                    lo[i] = min(lo[i], loc[i])
                    hi[i] = max(hi[i], loc[i])
        self.lo, self.hi = tuple(lo), tuple(hi)

    def g(self, tx, ty, tz):
        c = G(self.lo, self.hi, tx, ty, tz)
        p = self.o + self.ax[0] * c[0] + self.ax[1] * c[1] + self.ax[2] * c[2]
        return (p.x(), p.y(), p.z())

    def points(self):
        ctr = tuple((self.lo[i] + self.hi[i]) / 2 for i in range(3))
        out = []
        for tx in (0, .5, 1):
            for ty in (0, .5, 1):
                for tz in (0, .5, 1):
                    c = G(self.lo, self.hi, tx, ty, tz)
                    for k in (0, -.4, .4, 1.7):
                        q = tuple(ctr[i] + (c[i] - ctr[i]) * (1 + k) for i in range(3))
                        p = self.o + self.ax[0] * q[0] + self.ax[1] * q[1] + self.ax[2] * q[2]
                        out.append((p.x(), p.y(), p.z()))
        return out


AX = GroupBox(scn_groups, [3])
agrip = AX.g
prog("groups", CAM, select(("group", [3])), hover(agrip(1, 1, 1)), click(agrip(1, 1, 1)), hover(agrip(2, 2, 2)),
     click(agrip(2, 2, 2)), undo(), redo())
prog("groups", CAM, select(("group", [3])), hover(agrip(1, .5, .5)), click(agrip(1, .5, .5)), typed("2"), typed("3"))
prog("groups", CAM, select(("group", [3])), hover(agrip(1, 1, .5)), click(agrip(1, 1, .5)), hover(agrip(1.5, 1.5, .5)),
     click(agrip(1.5, 1.5, .5)))
# a mixed selection: a group and loose faces (the world frame)
prog("groups", CAM, select(("group", [1]), ("face", 0)), hover((0, 0, 0)), click((0, 0, 0)), hover((-3, -3, 0)),
     click((-3, -3, 0)), undo())
prog("groups", CAM, select(("group", [0]), ("group", [2]), ("group", [3])), hover((-5, 0, 0)), click((-5, 0, 0)),
     hover((-7, -2, 0)), click((-7, -2, 0)), undo())
# click-to-select a group, then scale it
prog("groups", CAM, click((-4, 1, .75)), hover((-3, 2, 1.5)), click((-3, 2, 1.5)), hover((-2, 3, 2.5)), click((-2, 3, 2.5)))
prog("groups", CAM, click((6.5, .5, .5)), hover((7, 1, 1)), click((7, 1, 1)), hover((8, 2, 2)), click((8, 2, 2)))
prog("groups", CAM, click((2, 2, 0)), hover((4, 4, 0)), click((4, 4, 0)), hover((6, 6, 0)), click((6, 6, 0)))

# ---- nested containers ---------------------------------------------------------------------------------------
CT = GroupBox(scn_container, [0])
CP = GroupBox(scn_container, [1])
prog("container", CAM, select(("group", [0])), hover(CT.g(1, 1, 1)), click(CT.g(1, 1, 1)), hover(CT.g(2, 2, 2)),
     click(CT.g(2, 2, 2)), undo(), redo())
prog("container", CAM, select(("group", [0])), hover(CT.g(1, .5, .5)), click(CT.g(1, .5, .5)), hover(CT.g(2, .5, .5)),
     click(CT.g(2, .5, .5)))
prog("container", CAM, select(("group", [0])), hover(CT.g(1, 1, .5)), click(CT.g(1, 1, .5)), typed("2;3"), typed("1;2"))
prog("container", CAM, select(("group", [1])), hover(CP.g(1, 1, 1)), click(CP.g(1, 1, 1)), hover(CP.g(2, 2, 2)),
     click(CP.g(2, 2, 2)))
prog("container", CAM, select(("group", [0]), ("group", [1])), hover((0, 6, 0)), click((0, 6, 0)), hover((-2, 4, -1)),
     click((-2, 4, -1)))
prog("container", CAM, select(("group", [0])), tap("ctrl"), hover(CT.g(1, 1, 1)), click(CT.g(1, 1, 1)), typed("2"), typed("3"))
prog("container", CAM, click((6.5, 1.5, .5)), hover((7, 2, 1)), click((7, 2, 1)))


# ---- seeded random programs ---------------------------------------------------------------------------------

def box_points(lo, hi, m=None):
    """The grips of a box (lo, hi) and points past/short of each, in the world."""
    out = []
    ctr = tuple((lo[i] + hi[i]) / 2 for i in range(3))
    for tx in (0, .5, 1):
        for ty in (0, .5, 1):
            for tz in (0, .5, 1):
                g = G(lo, hi, tx, ty, tz)
                out.append(g)
                for k in (-.4, .4, 1.7, 2.5):
                    out.append(tuple(ctr[i] + (g[i] - ctr[i]) * (1 + k) for i in range(3)))
    if m is not None:
        out = [(lambda p: (p.x(), p.y(), p.z()))(m.map(QVector3D(*q))) for q in out]
    return out


EXTRA = [(6, 0, 0), (8, 2, 0), (10, 2, 1), (4, 4, 0), (2, 2, 0), (0, 0, 0), (12, 3, 0)]

# scene -> list of (refs, point list)
RND = {
    "cube": [(CUBE, box_points(CLO, CHI)), ([("face", 6)], box_points((6, 0, 0), (8, 2, 0))),
             ([("face", 1), ("face", 5)], box_points((0, 0, 0), (3, 3, 3)))],
    "plate": [([("face", 0)], box_points((0, 0, 0), (4, 4, 0))), ([("face", 0), ("face", 1)], box_points((0, 0, 0), (9, 4, 0))),
              ([("edge", 8)], box_points((-3, 0, 0), (-3, 2, 1)))],
    "wall": [([("face", 0), ("face", 1)], box_points((0, 0, 0), (7, 0, 3))), ([("edge", 8)], box_points((0, 5, 0), (4, 5, 0)))],
    "mixed": [([("face", 0)], box_points((0, 0, 0), (6, 6, 0))), ([("face", 1)], box_points((8.5, 1.5, 0), (11.5, 4.5, 0))),
              ([("face", 0), ("face", 1), ("face", 2)], box_points((0, 0, 0), (11.5, 8, 2)))],
    "groups": [([("group", [0])], box_points(*GC)), ([("group", [1])], box_points((6, 0, 0), (7, 1, 1))),
               ([("group", [2])], box_points((0, 0, 0), (1, 1, 1), Bm)),
               ([("group", [3])], AX.points()),
               ([("group", [0]), ("face", 0)], box_points((-5, 0, 0), (4, 4, 1.5)))],
    "container": [([("group", [0])], CT.points()), ([("group", [1])], CP.points()),
                  ([("group", [0]), ("group", [1])], box_points((0, 0, 0), (9, 8, 3)))],
}

CAMS = (CAM, CAM_ORTHO, CAM_LOW, CAM_TOP, CAM_FAR)


def random_program(seed: int, scene: str, cam: dict, n: int = 26):
    rnd = random.Random(seed)
    choices = RND[scene]
    refs, pts = rnd.choice(choices)
    evs = []
    if rnd.random() < 0.9:
        evs.append(select(*refs))

    def pt():
        p = rnd.choice(pts) if rnd.random() < 0.85 else rnd.choice(EXTRA)
        return (p[0] + rnd.choice((0, 0, 0.05)), p[1], p[2])
    while len(evs) < n:
        r = rnd.random()
        if r < 0.24:
            evs.append(hover(pt()))
        elif r < 0.44:
            evs.append(click(pt()))
        elif r < 0.51:
            a, b = pt(), pt()
            evs += [hover(a), press(a), hover(b, drag=True), hover(pt(), drag=True), release(b)]
        elif r < 0.57:
            evs.append(tap("ctrl"))
        elif r < 0.62:
            evs.append(tap("shift"))
        elif r < 0.73:
            evs.append(typed(rnd.choice(["2", "0.5", "-1", "1;2;3", "2;3", "3m", "1;1", "0", "x", "4m;2m", "1.5"])))
        elif r < 0.77:
            evs.append(esc())
        elif r < 0.83:
            evs.append(undo())
        elif r < 0.86:
            evs.append(redo())
        elif r < 0.91:
            evs.append(select(*rnd.choice(choices)[0]))
        elif r < 0.93:
            evs.append(camera(rnd.choice(CAMS)))
        else:
            evs.append(hover(pt(), shift=rnd.random() < 0.3, ctrl=rnd.random() < 0.2))
    return (scene, cam, evs)


for _seed in range(300, 340):
    _sc = ("cube", "plate", "wall", "mixed", "groups", "container")[_seed % 6]
    PROGRAMS.append(random_program(_seed, _sc, CAMS[_seed % 5]))


# ---- the probe -------------------------------------------------------------------------------------------------

def _fnv(words):
    h = 0xcbf29ce484222325
    for w in words:
        for b in struct.pack("<I", w):
            h ^= b
            h = (h * 0x100000001b3) & 0xFFFFFFFFFFFFFFFF
    return h >> 4


def _bits_of(p):
    return [struct.unpack("<I", struct.pack("<f", c))[0] for c in (p.x(), p.y(), p.z())]


def extra_probe(tool, vp, ids):
    grips = tool._grips
    st = tool.scale_box_state()
    state = None
    if st is not None:
        words, gwords = [], []
        for a, b in st["segments"]:
            words += _bits_of(a) + _bits_of(b)
        for p, _role in st["grips"]:
            gwords += _bits_of(p)
        state = {"n_seg": len(st["segments"]), "seg_sum": _fnv(words), "grip_sum": _fnv(gwords),
                 "roles": [r for _p, r in st["grips"]]}

    def idx(g):
        if g is None:
            return None
        return next(i for i, x in enumerate(grips) if x is g)
    return {
        "lo": None if tool._lo is None else pbits(tool._lo),
        "hi": None if tool._hi is None else pbits(tool._hi),
        "axes": None if tool._axes is None else [pbits(a) for a in tool._axes],
        "orig": None if tool._orig is None else pbits(tool._orig),
        "n_grips": len(grips),
        "box_fresh": tool._box_version == vp.scene.version,
        "n_groups": len(tool._groups),
        "n_pos": len(tool._positions),
        "n_verts": len(tool._verts),
        "grip": idx(tool._grip),
        "hover_grip": idx(tool._hover_grip),
        "anchor": None if tool._anchor is None else pbits(tool._anchor),
        "factors": [float(f) for f in tool._factors],
        "grabbed_px": None if tool._grabbed_px is None else [float(c) for c in tool._grabbed_px],
        "moved": bool(tool._moved),
        "about_center": bool(tool.about_center),
        "uniform": bool(tool.uniform),
        "last": tool._last is not None,
        "orig_loaded": getattr(tool, "_orig_v", None) is not None,
        "ctrl_toggled": bool(getattr(tool, "_ctrl_toggled", False)),
        "state": state,
    }
