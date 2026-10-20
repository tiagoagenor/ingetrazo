# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
"""The Flip tool's scenes and programs for ``make_modify_tool_fixtures.py``
(``tools/flip.py``, docs/rust-rewrite-plan.md M3.13)."""
from __future__ import annotations

import math
import random

from _modify_harness import (ALT, CTRL, SHIFT, V, above, at, box, box_faces, click, dbl, deactivate, esc,  # noqa: F401
                             group, hover, key, keyup, press, redo, release, select, tap, typed, undo, xf,
                             Mesh, Scene, activate, camera, pbits, px)
from tools.flip import FlipTool

NAME = "flip"
make_tool = FlipTool

CAM = dict(target=(2, 2, 0.8), distance=14.0, yaw=-50.0, pitch=30.0, w=1000, h=600)
CAM_ORTHO = dict(target=(2, 2, 0.8), distance=14.0, yaw=-50.0, pitch=30.0, w=1000, h=600, perspective=False)
CAM_LOW = dict(target=(2, 2, 1.0), distance=16.0, yaw=-30.0, pitch=8.0, w=1000, h=600)
CAM_TOP = dict(target=(2, 2, 0.0), distance=16.0, yaw=-90.0, pitch=89.0, w=1000, h=600)
CAM_FAR = dict(target=(5, 4, 0.8), distance=30.0, yaw=-40.0, pitch=35.0, w=1000, h=600)
CAMS = (CAM, CAM_ORTHO, CAM_LOW, CAM_TOP, CAM_FAR)


def scn_cube() -> Scene:
    s = Scene()
    box_faces(s.mesh, 0, 0, 0, 3, 3, 3)
    s.mesh.add_face([V(6, 0, 0), V(8, 0, 0), V(8, 2, 0), V(6, 2, 0)])    # face 6
    s.mesh.add_edge(V(10, 0, 0), V(10, 2, 1))                           # edge 16
    return s


def scn_mixed() -> Scene:
    s = Scene()
    m = s.mesh
    m.add_face([V(0, 0, 0), V(6, 0, 0), V(6, 6, 0), V(0, 6, 0)],
               [[V(2, 2, 0), V(2, 4, 0), V(4, 4, 0), V(4, 2, 0)]])     # face 0, edges 0..7
    pts = [V(10 + 1.5 * math.cos(2 * math.pi * i / 12), 3 + 1.5 * math.sin(2 * math.pi * i / 12), 0.0)
           for i in range(12)]
    f = m.add_face(pts)                                                 # face 1, edges 8..19
    m.tag_curve(pts)
    f.attrs["color"] = [0.9, 0.3, 0.2]
    q = m.add_face([V(0, 8, 0), V(3, 8, 0), V(3, 8, 2), V(0, 8, 2)])    # face 2, edges 20..23
    q.attrs["texture"] = {"name": "t", "uvw": [0.5, 0.0, 0.0, 0.1, 0.0, 0.0, 0.5, 0.2]}
    for e in m.edges:
        if abs(e.v0.position.x() - 3.0) < 1e-6 and abs(e.v1.position.x() - 3.0) < 1e-6 and e.v0.position.y() == 8.0:
            e.soft = True
    m.add_edge(V(-3, 1, 0), V(-3, 5, 2))                                # edge 24
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


SCENES = {"cube": scn_cube, "mixed": scn_mixed, "groups": scn_groups, "container": scn_container}

PROGRAMS: list = []


def prog(scene, cam, *events):
    PROGRAMS.append((scene, cam, list(events)))


def plane_points(lo, hi):
    """Points on (and just off) the three planes of a selection's box."""
    c = [(lo[i] + hi[i]) / 2 for i in range(3)]
    h = [max((hi[i] - lo[i]) / 2, 0.3) * 1.15 for i in range(3)]
    out = []
    for a in (-.8, -.3, 0, .3, .8):
        for b in (-.8, -.3, 0.2, .6):
            out.append((c[0], c[1] + a * h[1], c[2] + b * h[2]))
            out.append((c[0] + a * h[0], c[1], c[2] + b * h[2]))
            out.append((c[0] + a * h[0], c[1] + b * h[1], c[2]))
    out += [tuple(c), (c[0] + 2 * h[0], c[1], c[2]), (c[0], c[1] - 3 * h[1], c[2] + 3 * h[2])]
    return out


CUBE = [("face", i) for i in range(6)]
CP = plane_points((0, 0, 0), (3, 3, 3))
# the three planes of a box, by hover and click
prog("cube", CAM, select(*CUBE), hover((1.5, 1.5, 1.5)), hover((1.5, 0.5, 2.5)), click((1.5, 0.5, 2.5)), undo(), redo())
prog("cube", CAM, select(*CUBE), hover((0.5, 1.5, 2.5)), click((0.5, 1.5, 2.5)), hover((2.5, 1.5, 0.5)), click((2.5, 1.5, 0.5)))
prog("cube", CAM, select(*CUBE), hover((0.5, 0.5, 1.5)), click((0.5, 0.5, 1.5)), undo())
prog("cube", CAM, select(*CUBE), hover((1.5, 3.2, 1.5)), click((1.5, 3.2, 1.5)), hover((3.2, 1.5, 1.5)), click((3.2, 1.5, 1.5)))
# arrow keys pick a plane (toggling), anywhere on screen
prog("cube", CAM, select(*CUBE), key("right"), hover((1.5, 1.5, 1.5)), click(px(20, 20)), key("right"), click(px(20, 20)))
prog("cube", CAM, select(*CUBE), key("left"), hover((1.5, 1.5, 1.5)), click((1.5, 1.5, 1.5)), key("up"), click((1.5, 1.5, 1.5)))
prog("cube", CAM, select(*CUBE), key("up"), key("left"), key("left"), hover((0.5, 1.5, 2.5)), click((0.5, 1.5, 2.5)), esc(),
     click((0.5, 1.5, 2.5)))
# Ctrl TAP toggles copy mode; ONE flip uses it
prog("cube", CAM, select(*CUBE), tap("ctrl"), hover((1.5, 0.5, 2.5)), click((1.5, 0.5, 2.5)), hover((1.5, 0.5, 2.5)),
     click((1.5, 0.5, 2.5)), undo(), undo(), redo())
prog("cube", CAM, select(*CUBE), tap("ctrl"), tap("ctrl"), hover((1.5, 0.5, 2.5)), click((1.5, 0.5, 2.5)))
prog("cube", CAM, select(*CUBE), hover((0.5, 1.5, 2.5)), tap("ctrl"), click((0.5, 1.5, 2.5)), tap("ctrl"), key("up"),
     click((0.5, 1.5, 2.5)))
prog("cube", CAM, select(*CUBE), key("ctrl"), hover((0.5, 1.5, 2.5)), keyup("ctrl"), click((0.5, 1.5, 2.5)))
prog("cube", CAM, select(*CUBE), tap("shift"), tap("alt"), key("enter"), hover((1.5, 1.5, 1.5)), click((1.5, 1.5, 1.5)))
# faces, edges and a mix, loose; with copies
prog("cube", CAM, select(("face", 6)), hover((7, 1, 0)), hover((7, 1, 0.1)), click((7, 1, 0.1)), undo(), redo())
prog("cube", CAM, select(("face", 6)), tap("ctrl"), hover((7, 1, 0.1)), click((7, 1, 0.1)), undo())
prog("cube", CAM, select(("face", 6)), key("right"), tap("ctrl"), click(px(5, 5)), key("up"), click(px(5, 5)))
prog("cube", CAM, select(("edge", 16)), hover((10, 1, .5)), click((10, 1, .5)))
prog("cube", CAM, select(("edge", 16)), tap("ctrl"), key("left"), click(px(5, 5)), undo())
prog("cube", CAM, select(("edge", 16), ("face", 6)), tap("ctrl"), key("right"), click(px(5, 5)), key("up"), click(px(5, 5)))
prog("cube", CAM, select(("edge", 16), ("face", 6)), key("up"), click(px(5, 5)), undo(), redo())
prog("cube", CAM, select(("face", 1), ("face", 5)), key("up"), click(px(5, 5)))
prog("cube", CAM, select(("face", 0), ("face", 2), ("face", 4)), tap("ctrl"), key("right"), click(px(5, 5)))
# nothing selected: the flash on activate, on click, the planes of nothing
prog("cube", CAM, hover((1.5, 1.5, 1.5)), click((1.5, 1.5, 1.5)), key("up"), click((1.5, 1.5, 1.5)), tap("ctrl"),
     click((1.5, 1.5, 1.5)))
prog("cube", CAM, hover((1.5, 1.5, 3)), select(*CUBE), hover((1.5, 1.5, 3)), click((1.5, 1.5, 3)), deactivate(), activate(),
     select(), hover((1.5, 1.5, 1.5)))
prog("cube", CAM, select(*CUBE), tap("ctrl"), deactivate(), activate(), hover((1.5, 0.5, 2.5)), click((1.5, 0.5, 2.5)))
prog("cube", CAM, select(*CUBE), key("up"), deactivate(), activate(), click(px(5, 5)), hover((1.5, 1.5, 1.5)))
# cameras
prog("cube", CAM_ORTHO, select(*CUBE), hover((1.5, 0.5, 2.5)), click((1.5, 0.5, 2.5)), undo())
prog("cube", CAM_LOW, select(*CUBE), hover((0.5, 1.5, 2.5)), click((0.5, 1.5, 2.5)), tap("ctrl"), click((0.5, 1.5, 2.5)))
prog("cube", CAM_TOP, select(*CUBE), hover((1.5, 0.5, 1.5)), hover((1.5, 1.5, 1.5)), click((1.5, 1.5, 1.5)), key("left"),
     click((1.5, 1.5, 1.5)))
prog("cube", CAM_FAR, select(*CUBE), hover((1.5, 1.5, 1.5)), click((1.5, 1.5, 1.5)), camera(CAM), hover((1.5, 1.5, 1.5)),
     click((1.5, 1.5, 1.5)))
# holes, a curve, a soft edge, attrs and a texture
prog("mixed", CAM, select(("face", 0)), hover((3, 1, 0)), key("right"), click(px(1, 1)), undo(), redo())
prog("mixed", CAM, select(("face", 0)), tap("ctrl"), key("up"), click(px(1, 1)), undo(), redo(), undo())
prog("mixed", CAM, select(("face", 1)), tap("ctrl"), key("right"), click(px(1, 1)), key("up"), click(px(1, 1)))
prog("mixed", CAM, select(("face", 2)), tap("ctrl"), key("left"), click(px(1, 1)), tap("ctrl"), key("left"), key("right"),
     click(px(1, 1)))
prog("mixed", CAM, select(("face", 2)), key("right"), click(px(1, 1)), undo())
prog("mixed", CAM, select(("face", 0), ("face", 1), ("face", 2)), tap("ctrl"), key("up"), click(px(1, 1)), undo())
prog("mixed", CAM, select(*[("edge", i) for i in range(8, 20)]), tap("ctrl"), key("right"), click(px(1, 1)), key("up"),
     click(px(1, 1)), undo(), redo())
prog("mixed", CAM, select(*[("edge", i) for i in range(8, 20)]), key("left"), click(px(1, 1)), undo())
prog("mixed", CAM, select(*[("edge", i) for i in range(20, 24)], ("edge", 24)), tap("ctrl"), key("up"), click(px(1, 1)))
prog("mixed", CAM, select(("edge", 24)), tap("ctrl"), key("left"), click(px(1, 1)), key("right"), click(px(1, 1)))
prog("mixed", CAM_TOP, select(("face", 0), ("face", 1)), hover((6, 3, 0)), hover((3, 3, 0)), click((3, 3, 0)),
     tap("ctrl"), click((3, 3, 0)))
# groups: classic, instances, rotated, own axes
prog("groups", CAM, select(("group", [0])), key("right"), click(px(1, 1)), undo(), redo())
prog("groups", CAM, select(("group", [0])), tap("ctrl"), key("up"), click(px(1, 1)), undo(), redo())
prog("groups", CAM, select(("group", [1])), key("left"), click(px(1, 1)), undo(), redo())
prog("groups", CAM, select(("group", [1])), tap("ctrl"), key("left"), click(px(1, 1)), undo(), redo())
prog("groups", CAM, select(("group", [2])), key("right"), click(px(1, 1)), key("right"), key("up"), click(px(1, 1)))
prog("groups", CAM, select(("group", [2])), tap("ctrl"), key("right"), click(px(1, 1)), undo())
prog("groups", CAM, select(("group", [3])), key("up"), click(px(1, 1)), undo(), redo())
prog("groups", CAM, select(("group", [3])), tap("ctrl"), key("left"), click(px(1, 1)), undo())
prog("groups", CAM, select(("group", [0]), ("group", [1]), ("group", [2])), tap("ctrl"), key("right"), click(px(1, 1)),
     undo(), redo())
prog("groups", CAM, select(("group", [0]), ("group", [1]), ("face", 0)), tap("ctrl"), key("up"), click(px(1, 1)), undo())
prog("groups", CAM, select(("group", [0]), ("group", [1]), ("face", 0)), key("left"), click(px(1, 1)), undo(), redo())
prog("groups", CAM, select(("group", [2])), hover((9, 2, 0.5)), hover((9.5, 2.5, 0.8)), click((9.5, 2.5, 0.8)))
prog("groups", CAM, select(("group", [1])), hover((6.5, 0.5, 0.5)), hover((6.5, 0.2, 0.7)), click((6.5, 0.2, 0.7)),
     tap("ctrl"), hover((6.5, 0.5, 0.7)), click((6.5, 0.5, 0.7)))
# nested containers
prog("container", CAM, select(("group", [0])), key("right"), click(px(1, 1)), undo(), redo())
prog("container", CAM, select(("group", [0])), tap("ctrl"), key("up"), click(px(1, 1)), undo())
prog("container", CAM, select(("group", [0]), ("group", [1])), tap("ctrl"), key("left"), click(px(1, 1)), undo(), redo())
prog("container", CAM, select(("group", [1])), key("up"), click(px(1, 1)), key("up"), key("right"), click(px(1, 1)))
prog("container", CAM, select(("group", [0]), ("face", 0)), tap("ctrl"), key("right"), click(px(1, 1)), undo())


# ---- seeded random programs --------------------------------------------------------------------------------

def bounds_of(scene_builder, refs):
    """The box the tool measures for a selection (read off the real tool)."""
    return None


RND = {
    "cube": [(CUBE, plane_points((0, 0, 0), (3, 3, 3))), ([("face", 6)], plane_points((6, 0, 0), (8, 2, 0))),
             ([("edge", 16)], plane_points((10, 0, 0), (10, 2, 1))),
             ([("face", 6), ("edge", 16), ("face", 1)], plane_points((0, 0, 0), (10, 3, 3)))],
    "mixed": [([("face", 0)], plane_points((0, 0, 0), (6, 6, 0))), ([("face", 1)], plane_points((8.5, 1.5, 0), (11.5, 4.5, 0))),
              ([("face", 2)], plane_points((0, 8, 0), (3, 8, 2))),
              ([("face", 0), ("face", 1), ("face", 2)], plane_points((0, 0, 0), (11.5, 8, 2))),
              ([("edge", i) for i in range(8, 20)], plane_points((8.5, 1.5, 0), (11.5, 4.5, 0)))],
    "groups": [([("group", [0])], plane_points((-5, 0, 0), (-3, 2, 1.5))), ([("group", [1])], plane_points((6, 0, 0), (7, 1, 1))),
               ([("group", [2])], plane_points((8, 1, 0), (10.5, 3.2, 1))),
               ([("group", [3])], plane_points((0, 8, 0), (2, 9, 1))),
               ([("group", [0]), ("group", [1]), ("face", 0)], plane_points((-5, 0, 0), (7, 4, 1.5)))],
    "container": [([("group", [0])], plane_points((6, 1, 0), (9, 4, 1))), ([("group", [1])], plane_points((0, 6, 0), (2, 8, 2))),
                  ([("group", [0]), ("group", [1]), ("face", 0)], plane_points((0, 0, 0), (9, 8, 2)))],
}
EXTRA = [(1.5, 1.5, 1.5), (7, 1, 0), (10, 1, .5), (3, 3, 0), (-4, 1, .75), (6.5, .5, .5)]


def random_program(seed: int, scene: str, cam: dict, n: int = 24):
    rnd = random.Random(seed)
    choices = RND[scene]
    refs, pts = rnd.choice(choices)
    evs = []
    if rnd.random() < 0.9:
        evs.append(select(*refs))

    def pt():
        p = rnd.choice(pts) if rnd.random() < 0.85 else rnd.choice(EXTRA)
        return (p[0], p[1], p[2])
    while len(evs) < n:
        r = rnd.random()
        if r < 0.25:
            evs.append(hover(pt()))
        elif r < 0.45:
            evs.append(click(pt()))
        elif r < 0.55:
            evs.append(tap("ctrl"))
        elif r < 0.70:
            evs.append(key(rnd.choice(["right", "left", "up", "down", "right", "left"])))
        elif r < 0.75:
            evs.append(esc())
        elif r < 0.81:
            evs.append(undo())
        elif r < 0.85:
            evs.append(redo())
        elif r < 0.90:
            evs.append(select(*rnd.choice(choices)[0]))
        elif r < 0.92:
            evs.append(camera(rnd.choice(CAMS)))
        elif r < 0.94:
            evs.append(rnd.choice([deactivate(), activate()]))
        else:
            evs.append(hover(pt(), ctrl=rnd.random() < 0.3))
    return (scene, cam, evs)


for _seed in range(500, 548):
    _sc = ("cube", "mixed", "groups", "container")[_seed % 4]
    PROGRAMS.append(random_program(_seed, _sc, CAMS[(_seed // 4) % 5]))


def extra_probe(tool, vp, ids):
    return {
        "hover_axis": tool._hover_axis,
        "lock_axis": tool._lock_axis,
        "copy": bool(tool._copy),
        "ctrl_toggled": bool(tool._ctrl_toggled),
    }
