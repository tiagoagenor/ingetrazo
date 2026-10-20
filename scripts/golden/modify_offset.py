# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
"""The Offset tool's scenes and programs for ``make_modify_tool_fixtures.py``
(``tools/offset.py``, docs/rust-rewrite-plan.md M3.13)."""
from __future__ import annotations

import math
import random

from _modify_harness import (V, above, at, box, box_faces, click, dbl, deactivate, esc,  # noqa: F401
                             group, hover, key, keyup, press, redo, release, select, tap, typed, undo, xf,
                             Mesh, Scene, activate, camera, pbits, px)
from tools.offset import OffsetTool

NAME = "offset"
make_tool = OffsetTool

CAM = dict(target=(5, 5, 0), distance=24.0, yaw=-50.0, pitch=40.0, w=1000, h=600)
CAM_TOP = dict(target=(5, 5, 0), distance=22.0, yaw=-90.0, pitch=89.0, w=1000, h=600)
CAM_ORTHO = dict(target=(5, 5, 0), distance=24.0, yaw=-50.0, pitch=40.0, w=1000, h=600, perspective=False)
CAM_LOW = dict(target=(5, 5, 0), distance=26.0, yaw=-30.0, pitch=20.0, w=1000, h=600)
CAM_BOX = dict(target=(4, 3, 1.5), distance=18.0, yaw=-50.0, pitch=30.0, w=1000, h=600)


def _poly(m, pts, **kw):
    return m.add_face([V(*p) for p in pts], **kw)


def scn_slab() -> Scene:
    s = Scene()
    m = s.mesh
    _poly(m, [(0, 0), (4, 0), (4, 3), (0, 3)])                                    # face 0
    _poly(m, [(6, 0), (8, 0), (8, 2), (6, 2)])                                    # face 1
    _poly(m, [(0, 6), (4, 6), (4, 7.5), (1.5, 7.5), (1.5, 10), (0, 10)])          # face 2 (L)
    _poly(m, [(6, 5), (10, 5), (10, 5.3), (6, 5.3)])                              # face 3 (thin strip)
    f = _poly(m, [(6, 7), (9, 7), (9, 9), (6, 9)])                                # face 4 with attrs
    f.attrs["color"] = [0.8, 0.2, 0.2]
    return s


def scn_box() -> Scene:
    s = Scene()
    m = s.mesh
    box_faces(m, 0, 0, 0, 3, 3, 3)                                                # faces 0..5
    _poly(m, [(0, 8, 0), (4, 8, 0), (4, 8, 3), (0, 8, 3)])                        # 6: wall
    _poly(m, [(6, 0, 0), (9, 0, 0), (9, 2, 1.5), (6, 2, 1.5)])                    # 7: tilted
    _poly(m, [(6, 5, 0), (9, 5, 0), (9, 7, 0), (6, 7, 0)])                        # 8: floor
    return s


def scn_ring() -> Scene:
    s = Scene()
    m = s.mesh
    _poly(m, [(0, 0), (6, 0), (6, 6), (0, 6)], hole_loops=[[V(2, 2), V(4, 2), V(4, 4), V(2, 4)]])        # 0
    _poly(m, [(8, 0), (12, 0), (12, 6), (8, 6)], hole_loops=[[V(8.5, 2), V(10, 2), V(10, 4), V(8.5, 4)]])  # 1
    _poly(m, [(0, 8), (5, 8), (5, 11), (0, 11)], hole_loops=[[V(1, 9), V(2, 9), V(2, 10), V(1, 10)],
                                                            [V(3, 9), V(4, 9), V(4, 10), V(3, 10)]])      # 2
    return s


def scn_curves() -> Scene:
    s = Scene()
    m = s.mesh
    cx, cy, r = 3.0, 3.0, 2.0
    circ = [V(cx + r * math.cos(2 * math.pi * i / 24), cy + r * math.sin(2 * math.pi * i / 24), 0) for i in range(24)]
    m.add_face(circ)
    m.tag_curve(circ, closed=True)
    ax, ay, ar = 9.0, 2.0, 2.0
    arc = [V(ax + ar * math.cos(math.pi * i / 8), ay + ar * math.sin(math.pi * i / 8), 0) for i in range(9)]
    m.add_face(arc)
    m.tag_curve(arc, closed=False)
    _poly(m, [(0, 8), (4, 8), (4, 10), (0, 10)])
    return s


def scn_dumb() -> Scene:
    s = Scene()
    _poly(s.mesh, [(0, 0), (3, 0), (3, 1.3), (5, 1.3), (5, 0), (8, 0), (8, 3), (5, 3), (5, 1.7), (3, 1.7), (3, 3),
                   (0, 3)])
    _poly(s.mesh, [(0, 6), (6, 6), (6, 8), (0, 8)])
    return s


def scn_edges() -> Scene:
    s = Scene()
    m = s.mesh

    def line(*pts):
        for a, b in zip(pts, pts[1:]):
            m.add_edge(V(*a), V(*b))
    line((0, 0, 0), (3, 0, 0), (3, 2, 0), (5, 2, 0), (5, 0.5, 0))                  # edges 0-3: open polyline
    line((7, 0, 0), (10, 0, 0), (10, 3, 0), (7, 3, 0), (7, 0, 0))                  # 4-7: closed, no face
    line((0, 6, 0), (3, 6, 0))                                                     # 8: lone line
    line((6, 6, 0), (8, 6, 0), (10, 6, 0))                                         # 9, 10
    line((8, 6, 0), (8, 8, 0))                                                     # 11: T junction at (8,6)
    line((0, 9, 0), (2, 9, 0), (4, 9, 0))                                          # 12, 13: collinear run
    line((12, 0, 0), (12, 3, 0), (12, 3, 2), (12, 0, 2))                           # 14-16: vertical polyline
    return s


def scn_groups() -> Scene:
    s = Scene()
    _poly(s.mesh, [(0, 0), (4, 0), (4, 3), (0, 3)])
    s.groups.append(group(box(6, 0, 0, 2, 2, 1), "classic"))
    s.groups.append(group(box(0, 0, 0, 1, 1, 1), "inst", xf(6, 5, 0)))
    return s


SCENES = {"slab": scn_slab, "box": scn_box, "ring": scn_ring, "curves": scn_curves, "dumb": scn_dumb,
          "edges": scn_edges, "groups": scn_groups}

PROGRAMS: list = []


def prog(scene, cam, *events):
    PROGRAMS.append((scene, cam, list(events)))


# ---- faces: drag and click ------------------------------------------------------------------
# inward drag on the 4x3 rectangle, undo/redo
prog("slab", CAM, hover((2, 1.5, 0)), click((2, 0.2, 0)), hover((2, 0.4, 0)), hover((2, 0.7, 0)), click((2, 0.7, 0)),
     undo(), redo())
# outward drag
prog("slab", CAM, hover((2, 1.5, 0)), click((2, 0.2, 0)), hover((2, -0.3, 0)), hover((2, -0.8, 0)), click((2, -0.8, 0)),
     undo())
# crossing flips inward/outward, then commit
prog("slab", CAM, hover((2, 1.5, 0)), click((2, 0.2, 0)), hover((2, 0.5, 0)), hover((2, -0.5, 0)), hover((2, 0.3, 0)),
     click((2, 0.3, 0)))
# a click with no movement is ignored, then a real one
prog("slab", CAM, hover((2, 1.5, 0)), click((2, 0.2, 0)), click((2, 0.2, 0)), hover((2, 0.9, 0)), click((2, 0.9, 0)))
# reference edge: the nearest one to the click (right side)
prog("slab", CAM, hover((2, 1.5, 0)), click((3.8, 1.5, 0)), hover((3.5, 1.5, 0)), hover((3.2, 1.5, 0)),
     click((3.2, 1.5, 0)))
prog("slab", CAM, hover((2, 1.5, 0)), click((0.2, 2.8, 0)), hover((0.7, 2.3, 0)), click((0.7, 2.3, 0)), undo(), redo())
# typed distances
prog("slab", CAM, hover((2, 1.5, 0)), click((2, 0.2, 0)), typed("0.4"), undo(), redo())
prog("slab", CAM, hover((2, 1.5, 0)), click((2, 0.2, 0)), hover((2, -0.3, 0)), typed("0.5"))
prog("slab", CAM, hover((2, 1.5, 0)), click((2, 0.2, 0)), hover((2, 0.6, 0)), typed("0"), typed("1;2"), typed("20cm"))
prog("slab", CAM, hover((2, 1.5, 0)), click((2, 0.2, 0)), typed("0"), typed("-0.3"), typed("2"))
# the hot retype after the commit: replaced, refused (too far), retyped again, zero
prog("slab", CAM, hover((2, 1.5, 0)), click((2, 0.2, 0)), typed("0.3"), typed("0.5"), typed("0.2"), typed("5"),
     typed("0.6"), typed("0"), undo())
prog("slab", CAM, hover((2, 1.5, 0)), click((2, 0.2, 0)), hover((2, -0.2, 0)), typed("0.3"), typed("0.6"), typed("9"),
     typed("0.4"))
# the retype window closes: hover/pick, other command
prog("slab", CAM, hover((2, 1.5, 0)), click((2, 0.2, 0)), typed("0.3"), undo(), typed("0.4"), redo(), typed("0.2"))
prog("slab", CAM, hover((2, 1.5, 0)), click((2, 0.2, 0)), typed("0.3"), hover((7, 1, 0)), click((7, 0.2, 0)),
     typed("0.2"), typed("0.4"))
prog("slab", CAM, hover((2, 1.5, 0)), click((2, 0.2, 0)), typed("0.3"), esc(), typed("0.4"))
# refusals: a thin strip, a small square, the limit message
prog("slab", CAM, hover((8, 5.15, 0)), click((8, 5.02, 0)), hover((8, 5.2, 0)), click((8, 5.2, 0)))
prog("slab", CAM, hover((8, 5.15, 0)), click((8, 5.02, 0)), typed("0.2"), typed("0.1"))
prog("slab", CAM, hover((8, 5.15, 0)), click((8, 5.02, 0)), hover((8, 4.6, 0)), typed("0.25"))
prog("slab", CAM, hover((7, 1, 0)), click((6.1, 1, 0)), typed("1.5"), typed("0.4"))
prog("slab", CAM, hover((7, 1, 0)), click((6.1, 1, 0)), hover((5.0, 1, 0)), typed("1.5"))
prog("slab", CAM, hover((7, 1, 0)), click((6.1, 1, 0)), typed("1"), typed("0.3"))
# concave L: inward, outward, corner bevels
prog("slab", CAM, hover((0.5, 9, 0)), click((0.1, 9, 0)), hover((0.3, 9, 0)), hover((0.5, 9, 0)), click((0.5, 9, 0)))
prog("slab", CAM, hover((0.5, 9, 0)), click((0.1, 9, 0)), typed("0.3"), undo(), redo())
prog("slab", CAM, hover((0.5, 9, 0)), click((1.4, 9, 0)), hover((2.0, 9, 0)), typed("0.4"))
prog("slab", CAM, hover((0.5, 9, 0)), click((3, 6.2, 0)), hover((3, 5.5, 0)), click((3, 5.5, 0)), undo())
prog("slab", CAM, hover((2, 6.8, 0)), click((2, 7.3, 0)), typed("0.5"), typed("0.2"))
# face attrs kept
prog("slab", CAM, hover((7.5, 8, 0)), click((7.5, 7.1, 0)), hover((7.5, 7.5, 0)), click((7.5, 7.5, 0)), undo(), redo())
prog("slab", CAM, hover((7.5, 8, 0)), click((7.5, 7.1, 0)), hover((7.5, 6.5, 0)), click((7.5, 6.5, 0)))
# Esc, deactivate/activate mid-drag, hover without face, double click
prog("slab", CAM, hover((2, 1.5, 0)), click((2, 0.2, 0)), hover((2, 0.6, 0)), esc(), hover((2, 0.6, 0)), click((2, 0.6, 0)))
prog("slab", CAM, hover((2, 1.5, 0)), click((2, 0.2, 0)), hover((2, 0.6, 0)), deactivate(), activate(), hover((2, 0.6, 0)),
     click((2, 0.6, 0)))
prog("slab", CAM, hover((2, 1.5, 0)), click((2, 0.2, 0)), typed("0.3"), deactivate(), activate(), typed("0.2"))
prog("slab", CAM, hover(px(5, 5)), click(px(5, 5)), hover((20, 20, 0)), click((20, 20, 0)), typed("0.3"), esc())
prog("slab", CAM, hover((2, 1.5, 0)), dbl((2, 0.2, 0)), hover((2, 0.6, 0)), dbl((2, 0.6, 0)))
prog("slab", CAM, hover((2, 1.5, 0)), click((2, 0.2, 0)), hover((2, 0.6, 0)), camera(dict(CAM, yaw=-20.0)),
     hover((2, 0.7, 0)), click((2, 0.7, 0)))
# cameras
prog("slab", CAM_TOP, hover((2, 1.5, 0)), click((2, 0.2, 0)), hover((2, 0.5, 0)), hover((2, 0.9, 0)), click((2, 0.9, 0)))
prog("slab", CAM_ORTHO, hover((2, 1.5, 0)), click((2, 0.2, 0)), hover((2, 0.5, 0)), click((2, 0.5, 0)), undo())
prog("slab", CAM_LOW, hover((2, 1.5, 0)), click((2, 0.2, 0)), hover((2, 0.8, 0)), hover((2, -0.5, 0)), click((2, -0.5, 0)))
# ---- boxes, walls, tilted faces -------------------------------------------------------------
prog("box", CAM_BOX, hover((1.5, 1.5, 3)), click((1.5, 0.2, 3)), hover((1.5, 0.5, 3)), hover((1.5, 0.8, 3)),
     click((1.5, 0.8, 3)), undo(), redo())
prog("box", CAM_BOX, hover((1.5, 1.5, 3)), click((1.5, 0.2, 3)), typed("0.3"), typed("0.2"))
prog("box", CAM_BOX, hover((1.5, 1.5, 3)), click((1.5, 0.2, 3)), hover((1.5, -0.4, 3)), click((1.5, -0.4, 3)))
prog("box", CAM_BOX, hover((3, 1.5, 1.5)), click((3, 0.2, 1.5)), hover((3, 0.5, 1.5)), click((3, 0.5, 1.5)))
prog("box", CAM_BOX, hover((2, 8, 1.5)), click((2, 8, 0.2)), hover((2, 8, 0.6)), hover((2, 8, 0.9)), click((2, 8, 0.9)),
     undo())
prog("box", CAM_BOX, hover((2, 8, 1.5)), click((0.2, 8, 1.5)), typed("0.5"), typed("0.25"))
prog("box", CAM_BOX, hover((2, 8, 1.5)), click((2, 8, 2.8)), hover((2, 8, 3.5)), click((2, 8, 3.5)))
prog("box", CAM_BOX, hover((7.5, 1, 0.75)), click((7.5, 0.1, 0.05)), hover((7.5, 0.5, 0.25)), hover((7.5, 0.8, 0.4)),
     click((7.5, 0.8, 0.4)), undo())
prog("box", CAM_BOX, hover((7.5, 1, 0.75)), click((7.5, 0.1, 0.05)), typed("0.3"), typed("0.15"))
prog("box", CAM_BOX, hover((7.5, 6, 0)), click((7.5, 5.1, 0)), hover((7.5, 5.5, 0)), click((7.5, 5.5, 0)))
prog("box", CAM_BOX, hover((7.5, 6, 0)), click((7.5, 5.1, 0)), typed("0.5"), typed("2"))
# ---- rings (holes) ----------------------------------------------------------------------------
prog("ring", CAM, hover((1, 3, 0)), click((0.2, 3, 0)), hover((0.5, 3, 0)), hover((0.9, 3, 0)), click((0.9, 3, 0)),
     undo(), redo())
prog("ring", CAM, hover((1, 3, 0)), click((0.2, 3, 0)), typed("0.4"), typed("0.8"), typed("1.5"))
prog("ring", CAM, hover((1, 3, 0)), click((0.2, 3, 0)), typed("1.2"))
prog("ring", CAM, hover((1, 3, 0)), click((0.2, 3, 0)), hover((-0.4, 3, 0)), click((-0.4, 3, 0)))
prog("ring", CAM, hover((1, 3, 0)), click((0.2, 3, 0)), hover((-0.4, 3, 0)), typed("0.5"))
# the offset runs into the hole (flash)
prog("ring", CAM, hover((9, 1, 0)), click((11.8, 3, 0)), hover((11.2, 3, 0)), click((11.2, 3, 0)))
prog("ring", CAM, hover((9, 1, 0)), click((8.1, 3, 0)), typed("0.5"))
prog("ring", CAM, hover((9, 1, 0)), click((8.1, 3, 0)), typed("0.3"), typed("0.6"), typed("0.2"))
prog("ring", CAM, hover((9, 1, 0)), click((8.1, 3, 0)), typed("1.5"))
prog("ring", CAM, hover((9, 1, 0)), click((9, 0.2, 0)), typed("0.8"), typed("0.4"))
prog("ring", CAM, hover((9, 1, 0)), click((9, 0.2, 0)), typed("1.2"))
prog("ring", CAM, hover((9, 1, 0)), click((9, 0.2, 0)), hover((9, -0.4, 0)), click((9, -0.4, 0)), undo(), redo())
# two holes: regions with holes placed; a hole left in the strip
prog("ring", CAM, hover((0.4, 9.5, 0)), click((0.1, 9.5, 0)), typed("0.3"), undo(), redo())
prog("ring", CAM, hover((0.4, 9.5, 0)), click((0.1, 9.5, 0)), typed("0.8"))
prog("ring", CAM, hover((0.4, 9.5, 0)), click((2.5, 8.1, 0)), typed("0.7"))
# ---- curves --------------------------------------------------------------------------------------
C_CIRCLE = (3 + 2 * math.cos(0.1), 3 + 2 * math.sin(0.1), 0)
prog("curves", CAM, hover((3, 3, 0)), click((3.5, 3.5, 0)), hover((3.6, 3.6, 0)), hover((3.3, 3.3, 0)),
     click((3.3, 3.3, 0)), undo(), redo())
prog("curves", CAM, hover((3, 3, 0)), click((4.8, 3, 0)), typed("0.4"), typed("0.3"))
prog("curves", CAM, hover((3, 3, 0)), click((4.8, 3, 0)), typed("2.5"), typed("1"))
prog("curves", CAM, hover((3, 3, 0)), click((4.8, 3, 0)), hover((5.5, 3, 0)), click((5.5, 3, 0)), undo())
prog("curves", CAM, hover((9, 1, 0)), click((9, 1, 0)), hover((9, 1.3, 0)), click((9, 1.3, 0)))
prog("curves", CAM, hover((9, 1, 0)), click((9, 0.1, 0)), typed("0.3"), undo(), redo())
prog("curves", CAM, hover((9, 1, 0)), click((9, 3.9, 0)), typed("0.4"), typed("0.2"))
prog("curves", CAM, hover((9, 1, 0)), click((10.9, 2.3, 0)), hover((10.5, 2.2, 0)), click((10.5, 2.2, 0)))
prog("curves", CAM, hover((9, 1, 0)), click((9, 0.1, 0)), hover((9, -0.5, 0)), click((9, -0.5, 0)))
prog("curves", CAM, hover((2, 9, 0)), click((2, 8.2, 0)), typed("0.3"), hover((3, 3, 0)), click((4.8, 3, 0)), typed("0.3"))
# ---- regions splitting: the dumbbell -----------------------------------------------------------------
prog("dumb", CAM, hover((1, 1, 0)), click((1, 0.1, 0)), typed("0.3"), undo(), redo())
prog("dumb", CAM, hover((1, 1, 0)), click((1, 0.1, 0)), typed("0.15"))
prog("dumb", CAM, hover((1, 1, 0)), click((1, 0.1, 0)), hover((1, 0.3, 0)), hover((1, 0.5, 0)), click((1, 0.5, 0)))
prog("dumb", CAM, hover((1, 1, 0)), click((1, 0.1, 0)), typed("0.2"), typed("0.4"), typed("0.9"))
prog("dumb", CAM, hover((1, 1, 0)), click((1, 0.1, 0)), hover((1, -0.5, 0)), click((1, -0.5, 0)), undo())
prog("dumb", CAM, hover((1, 1, 0)), click((1, 0.1, 0)), typed("0.8"), typed("1.1"))
prog("dumb", CAM, hover((1, 1, 0)), click((4, 1.4, 0)), typed("0.25"))
prog("dumb", CAM, hover((3, 7, 0)), click((3, 6.1, 0)), typed("0.9"), typed("1.1"))
# ---- Edges mode ------------------------------------------------------------------------------------
# an open polyline taken by a click on one of its edges
prog("edges", CAM, hover((1.5, 0, 0)), click((1.5, 0, 0)), hover((1.5, 0.3, 0)), hover((1.5, 0.6, 0)), click((1.5, 0.6, 0)),
     undo(), redo())
prog("edges", CAM, hover((1.5, 0, 0)), click((1.5, 0, 0)), typed("0.4"), undo(), redo())
prog("edges", CAM, hover((1.5, 0, 0)), click((1.5, 0, 0)), hover((1.5, -0.4, 0)), click((1.5, -0.4, 0)))
prog("edges", CAM, hover((1.5, 0, 0)), click((1.5, 0, 0)), typed("0.3"), typed("0.5"), typed("2"), typed("0.1"))
prog("edges", CAM, hover((3, 1, 0)), click((3, 1, 0)), hover((2.5, 1, 0)), click((2.5, 1, 0)))
prog("edges", CAM, hover((4, 2, 0)), click((4, 2, 0)), typed("0.5"), hover((5, 1, 0)), click((5, 1.2, 0)), typed("0.2"))
# a closed loop with no face
prog("edges", CAM, hover((8.5, 0, 0)), click((8.5, 0, 0)), hover((8.5, 0.5, 0)), click((8.5, 0.5, 0)), undo(), redo())
prog("edges", CAM, hover((8.5, 0, 0)), click((8.5, 0, 0)), typed("0.5"), typed("0.7"))
prog("edges", CAM, hover((8.5, 0, 0)), click((8.5, 0, 0)), hover((8.5, -0.6, 0)), click((8.5, -0.6, 0)))
prog("edges", CAM, hover((8.5, 0, 0)), click((8.5, 0, 0)), typed("2"), typed("0.3"))
prog("edges", CAM, hover((8.5, 0, 0)), click((8.5, 0, 0)), typed("1.6"))
# one lone line, a junction, a straight run (flash texts)
prog("edges", CAM, hover((1.5, 6, 0)), click((1.5, 6, 0)), hover((1.5, 6.5, 0)))
prog("edges", CAM, hover((7, 6, 0)), click((7, 6, 0)), hover((8, 7, 0)), click((8, 7, 0)))
prog("edges", CAM, hover((8, 7, 0)), click((8, 7, 0)))
prog("edges", CAM, hover((1, 9, 0)), click((1, 9, 0)), hover((3, 9, 0)), click((3, 9, 0)))
# a vertical polyline
prog("edges", CAM, hover((12, 1.5, 0)), click((12, 1.5, 0)), hover((12.4, 1.5, 0)), hover((12.8, 1.5, 0)),
     click((12.8, 1.5, 0)), undo())
prog("edges", CAM, hover((12, 1.5, 0)), click((12, 1.5, 0)), typed("0.4"), typed("0.2"))
prog("edges", CAM, hover((12, 3, 1)), click((12, 3, 1)), hover((11.5, 3, 1)), click((11.5, 3, 1)))
# armed from the selection: edges selected BEFORE the tool is picked up
prog("edges", CAM, select(("edge", 0), ("edge", 1), ("edge", 2), ("edge", 3)), deactivate(), activate(),
     hover((1.5, 0, 0)), click((1.5, 0.1, 0)), hover((1.5, 0.6, 0)), click((1.5, 0.6, 0)), undo(), redo())
prog("edges", CAM, select(("edge", 0), ("edge", 1), ("edge", 2), ("edge", 3)), deactivate(), activate(),
     click((1.5, 0.1, 0)), typed("0.4"), typed("0.2"))
prog("edges", CAM, select(("edge", 4), ("edge", 5), ("edge", 6), ("edge", 7)), deactivate(), activate(),
     hover((8.5, 0.5, 0)), click((8.5, 0.5, 0)), hover((8.5, 1.0, 0)), esc(), hover((8.5, 0.5, 0)), click((8.5, 0.5, 0)))
prog("edges", CAM, select(("edge", 0), ("edge", 1), ("edge", 2), ("edge", 3)), deactivate(), activate(),
     click((1.5, 0.1, 0)), hover((1.5, 0.5, 0)), deactivate(), activate(), hover((1.5, 0.5, 0)), click((1.5, 0.5, 0)))
# an unarmed selection: one edge, mixed with a face, two separate runs, a junction, collinear
prog("edges", CAM, select(("edge", 0)), deactivate(), activate(), hover((1.5, 0, 0)), click((1.5, 0, 0)))
prog("edges", CAM, select(("edge", 0), ("edge", 1), ("edge", 5)), deactivate(), activate(), hover((1.5, 0, 0)),
     click((1.5, 0, 0)))
prog("edges", CAM, select(("edge", 9), ("edge", 10), ("edge", 11)), deactivate(), activate(), click((9, 6, 0)))
prog("edges", CAM, select(("edge", 12), ("edge", 13)), deactivate(), activate(), click((1, 9, 0)))
prog("edges", CAM, select(("edge", 14), ("edge", 15), ("edge", 16)), deactivate(), activate(), click((12, 1.5, 0)),
     hover((12.5, 1.5, 0)), click((12.5, 1.5, 0)))
prog("edges", CAM, select(("edge", 4), ("edge", 5)), deactivate(), activate(), hover((8.5, 0, 0)), click((8.5, 0, 0)),
     hover((8.5, 0.3, 0)), click((8.5, 0.3, 0)))
prog("edges", CAM, select(("edge", 0), ("edge", 1), ("edge", 2), ("edge", 3)), deactivate(), activate(),
     select(("edge", 4)), hover((8.5, 0, 0)), click((8.5, 0, 0)))
# ---- a group on the side: its faces are not picked by pick_face -----------------------------------
prog("groups", CAM, hover((1, 1, 0)), click((0.1, 1, 0)), hover((0.5, 1, 0)), click((0.5, 1, 0)))
prog("groups", CAM, hover((7, 1, 1)), click((7, 1, 1)), hover((7, 1, 1)), click((6.5, 0.5, 1)), typed("0.2"))
prog("groups", CAM, hover((6.5, 5.5, 1)), click((6.5, 5.5, 1)), typed("0.1"), hover((1, 1, 0)), click((1, 0.1, 0)),
     typed("0.3"), undo())


# ---- seeded random programs ---------------------------------------------------------------------------

POINTS = {
    "slab": [(2, 1.5, 0), (2, 0.2, 0), (3.8, 1.5, 0), (2, 0.8, 0), (2, -0.5, 0), (7, 1, 0), (6.1, 1, 0), (7, 1, 0),
             (0.5, 9, 0), (0.1, 9, 0), (3, 6.2, 0), (8, 5.15, 0), (8, 5.02, 0), (7.5, 8, 0), (7.5, 7.1, 0), (7.5, 6.5, 0),
             (2, 7.3, 0), (20, 20, 0)],
    "box": [(1.5, 1.5, 3), (1.5, 0.2, 3), (1.5, 0.8, 3), (3, 1.5, 1.5), (3, 0.5, 1.5), (2, 8, 1.5), (2, 8, 0.2),
            (2, 8, 0.9), (7.5, 1, 0.75), (7.5, 0.1, 0.05), (7.5, 0.8, 0.4), (7.5, 6, 0), (7.5, 5.1, 0), (7.5, 5.5, 0)],
    "ring": [(1, 3, 0), (0.2, 3, 0), (0.9, 3, 0), (-0.4, 3, 0), (9, 1, 0), (8.1, 3, 0), (11.8, 3, 0), (11.2, 3, 0),
             (9, 0.2, 0), (9, -0.4, 0), (0.4, 9.5, 0), (0.1, 9.5, 0), (2.5, 8.1, 0)],
    "curves": [(3, 3, 0), (3.5, 3.5, 0), (4.8, 3, 0), (3.3, 3.3, 0), (5.5, 3, 0), (9, 1, 0), (9, 0.1, 0), (9, 1.3, 0),
               (9, 3.9, 0), (10.9, 2.3, 0), (10.5, 2.2, 0), (9, -0.5, 0), (2, 9, 0), (2, 8.2, 0)],
    "dumb": [(1, 1, 0), (1, 0.1, 0), (1, 0.5, 0), (1, -0.5, 0), (4, 1.4, 0), (4, 1, 0), (3, 7, 0), (3, 6.1, 0), (3, 6.6, 0)],
    "edges": [(1.5, 0, 0), (1.5, 0.6, 0), (1.5, -0.4, 0), (3, 1, 0), (2.5, 1, 0), (4, 2, 0), (5, 1.2, 0), (8.5, 0, 0),
              (8.5, 0.5, 0), (8.5, -0.6, 0), (1.5, 6, 0), (7, 6, 0), (8, 7, 0), (1, 9, 0), (12, 1.5, 0), (12.8, 1.5, 0),
              (12, 3, 1)],
    "groups": [(1, 1, 0), (0.1, 1, 0), (0.5, 1, 0), (7, 1, 1), (6.5, 5.5, 1), (6.5, 0.5, 1)],
}
REFS = {
    "edges": [("edge", i) for i in range(17)],
    "slab": [("face", 0), ("face", 1)],
}


def random_program(seed: int, scene: str, cam: dict, n: int = 24):
    rnd = random.Random(seed)
    pts = POINTS[scene]
    evs = []
    for _ in range(n):
        r = rnd.random()
        p = rnd.choice(pts)
        q = (p[0] + rnd.choice((0, 0, 0.2, -0.3, 0.5)), p[1] + rnd.choice((0, 0, 0.2, -0.3, 0.5)), p[2])
        if r < 0.30:
            evs.append(hover(q))
        elif r < 0.55:
            evs.append(click(q))
        elif r < 0.72:
            evs.append(typed(rnd.choice(["0.1", "0.25", "0.4", "0.6", "1", "2", "0", "-0.3", "1;2", "5", "0.05"])))
        elif r < 0.78:
            evs.append(esc())
        elif r < 0.84:
            evs.append(undo())
        elif r < 0.88:
            evs.append(redo())
        elif r < 0.93 and scene in REFS:
            k = rnd.randint(1, min(4, len(REFS[scene])))
            evs.append(select(*rnd.sample(REFS[scene], k)))
            if rnd.random() < 0.8:
                evs.append(deactivate())
                evs.append(activate())
        elif r < 0.96:
            evs.append(deactivate())
            evs.append(activate())
        else:
            evs.append(hover(q))
    return (scene, cam, evs)


_SCENES = ("slab", "box", "ring", "curves", "dumb", "edges", "slab", "edges")
for _seed in range(200, 248):
    _sc = _SCENES[_seed % len(_SCENES)]
    _cam = CAM_BOX if _sc == "box" else (CAM, CAM_ORTHO, CAM_LOW, CAM_TOP)[_seed % 4]
    PROGRAMS.append(random_program(_seed, _sc, _cam))


def extra_probe(tool, vp, ids):
    last = tool._last
    return {
        "hovered_face": tool.hovered_face,
        "hovered_edge": tool.hovered_edge,
        "base_face": tool.base_face,
        "distance": float(tool.distance),
        "dragging": bool(tool.dragging),
        "n_loop": len(tool._loop),
        "normal": tool._normal,
        "ref_point": tool._ref_point,
        "ref_inward": tool._ref_inward,
        "chain": bool(tool._chain),
        "closed": bool(tool._closed),
        "armed": bool(tool._armed),
        "last": last is not None,
        "last_chain": None if last is None else bool(last["chain"]),
        "last_sign": None if last is None else float(last["sign"]),
    }
