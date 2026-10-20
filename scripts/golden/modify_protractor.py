# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
"""The Protractor tool's scenes and programs for ``make_modify_tool_fixtures.py``
(``tools/protractor.py``, docs/rust-rewrite-plan.md M3.13). The scenes and the
shared protractor probe are reused by ``modify_rotate.py``."""
from __future__ import annotations

import random
import struct

from _modify_harness import (ALT, CTRL, SHIFT, V, above, at, box, box_faces, click, dbl, deactivate, esc,  # noqa: F401
                             group, hover, key, keyup, press, redo, release, select, tap, typed, undo, xf,
                             Mesh, Scene, activate, camera, pbits, px)
from tools.protractor import ProtractorTool

NAME = "protractor"
make_tool = ProtractorTool

CAM = dict(target=(2, 2, 0.8), distance=14.0, yaw=-50.0, pitch=30.0, w=1000, h=600)
CAM_ORTHO = dict(target=(2, 2, 0.8), distance=14.0, yaw=-50.0, pitch=30.0, w=1000, h=600, perspective=False)
CAM_LOW = dict(target=(2, 2, 1.0), distance=16.0, yaw=-30.0, pitch=8.0, w=1000, h=600)
CAM_TOP = dict(target=(2, 2, 0.0), distance=16.0, yaw=-90.0, pitch=89.0, w=1000, h=600)


def d64(x):
    return None if x is None else struct.pack(">d", float(x)).hex()


def bits_or_none(p):
    return None if p is None else pbits(p)


def base_probe(tool) -> dict:
    """ProtractorBase's own state (shared with Rotate)."""
    wp = tool.work_plane
    return {
        "ref": bits_or_none(tool.ref_point),
        "work_plane": None if wp is None else [pbits(wp[0]), pbits(wp[1])],
        "axis_pick": tool._axis_pick,
        "shift_normal": bits_or_none(tool._shift_normal),
        "custom_axis": bits_or_none(tool._custom_axis),
        "drag_armed": bool(tool._axis_drag_armed),
        "drag_live": bits_or_none(tool._axis_drag_live),
        "disc_r": d64(tool._disc_r),
        "snap_ticks": bool(tool._snap_ticks),
        "exact_snap": bool(tool._exact_snap),
        "ctrl_toggled": bool(getattr(tool, "_ctrl_toggled", False)),
    }


# ---- scenes (shared with Rotate) ---------------------------------------------------------------

def scn_flat() -> Scene:
    s = Scene()
    m = s.mesh
    m.add_face([V(0, 0, 0), V(4, 0, 0), V(4, 4, 0), V(0, 4, 0)])
    m.add_face([V(6, 0, 0), V(9, 0, 0), V(9, 2, 0), V(6, 2, 0)])
    m.add_edge(V(-3, 0, 0), V(-3, 2, 1))
    return s


def scn_solids() -> Scene:
    s = Scene()
    m = s.mesh
    box_faces(m, 0, 0, 0, 3, 3, 3)
    m.add_face([V(5, 0, 0), V(8, 0, 0), V(8, 3, 1.5), V(5, 3, 1.5)])
    m.add_edge(V(-2, 0, 0), V(-2, 3, 0))
    m.add_edge(V(-2, 3, 0), V(0, 5, 0))
    return s


def scn_groups() -> Scene:
    s = Scene()
    proto = box(0, 0, 0, 1, 1, 1, (0.2, 0.6, 0.9))
    g0 = group(box(-5, 0, 0, 2, 2, 1.5), "classic")
    g1 = group(proto, "inst_a", xf(6, 0, 0))
    g2 = group(proto, "inst_b", xf(9, 2, 0, 30))
    s.groups.extend([g0, g1, g2])
    s.mesh.add_face([V(0, 0, 0), V(4, 0, 0), V(4, 4, 0), V(0, 4, 0)])
    return s


def scn_container() -> Scene:
    s = Scene()
    p = box(0, 0, 0, 1, 1, 1)
    k1 = group(p, "kid1", xf(1, 0, 0, 20))
    k2 = group(p, "kid2", xf(0, 2, 0))
    ctr = group(Mesh(), "container", xf(6, 1, 0, 10), children=[k1, k2])
    s.groups.append(ctr)
    s.mesh.add_face([V(0, 0, 0), V(3, 0, 0), V(3, 3, 0), V(0, 3, 0)])
    return s


def scn_curves() -> Scene:
    """A face with a hole, a circle tagged as a curve, a soft edge, a lone edge."""
    import math

    import core.mesh as cm
    cm._CURVE_COUNTER = 1       # the scene must not depend on what was built before
    s = Scene()
    m = s.mesh
    m.add_face([V(0, 0, 0), V(5, 0, 0), V(5, 5, 0), V(0, 5, 0)])
    m.add_face([V(1.5, 1.5, 0), V(3.5, 1.5, 0), V(3.5, 3.5, 0), V(1.5, 3.5, 0)])
    pts = [V(7 + math.cos(2 * math.pi * k / 12), 1 + math.sin(2 * math.pi * k / 12), 0) for k in range(12)]
    for i in range(12):
        m.add_edge(pts[i], pts[(i + 1) % 12])
    m.tag_curve(pts, True)
    e = m.add_edge(V(6, 4, 0), V(8, 4, 1))
    e.soft = True
    m.add_edge(V(-3, 0, 0), V(-3, 2, 1))
    return s


SCENES = {"flat": scn_flat, "solids": scn_solids, "groups": scn_groups, "container": scn_container,
          "curves": scn_curves}

PROGRAMS: list = []


def prog(scene, cam, *events):
    PROGRAMS.append((scene, cam, list(events)))


C0 = (2, 2, 0)
REF = at(C0, 70, 0)
NEAR_A = at(C0, 42, -45)
NEAR_B = at(C0, 42, 45)
FAR_A = at(C0, 150, -90)
FAR_B = at(C0, 160, 70)
FAR_C = at(C0, -140, 20)

# the guide: centre, base arm, sweep, click (near the disc: 15-degree ticks)
prog("flat", CAM, hover(C0), click(C0), hover(REF), click(REF), hover(NEAR_A), hover(at(C0, 45, -52)),
     click(NEAR_A), undo(), redo())
# far from the disc: free 0.1-degree angle
prog("flat", CAM, hover(C0), click(C0), hover(REF), click(REF), hover(FAR_A), hover(at(C0, 151, -91)), click(FAR_A))
prog("flat", CAM, hover(C0), click(C0), hover(REF), click(REF), hover(FAR_B), click(FAR_B), undo())
prog("flat", CAM, hover(C0), click(C0), hover(REF), click(REF), hover(FAR_C), click(FAR_C), redo())
# typed angle (the sign follows the sweep)
prog("flat", CAM, hover(C0), click(C0), hover(REF), click(REF), hover(NEAR_A), typed("45"))
prog("flat", CAM, hover(C0), click(C0), hover(REF), click(REF), hover(NEAR_B), typed("30"))
prog("flat", CAM, hover(C0), click(C0), hover(REF), click(REF), hover(FAR_B), typed("-12.5"))
prog("flat", CAM, hover(C0), click(C0), hover(REF), click(REF), typed("20"))
prog("flat", CAM, hover(C0), click(C0), hover(REF), click(REF), hover(NEAR_A), typed("3:12"))
prog("flat", CAM, hover(C0), click(C0), hover(REF), click(REF), hover(NEAR_B), typed("1:6"))
# the hot retype after the commit
prog("flat", CAM, hover(C0), click(C0), hover(REF), click(REF), hover(NEAR_A), click(NEAR_A), typed("60"),
     typed("-20"), typed("3:12"), undo(), typed("15"))
prog("flat", CAM, hover(C0), click(C0), hover(REF), click(REF), hover(NEAR_B), typed("45"), typed("10"), typed("-10"),
     click(C0), typed("5"))
prog("flat", CAM, typed("30"), hover(C0), typed("30"))
# Ctrl: measure-only mode (its flash, status clause, cursor plus)
prog("flat", CAM, tap("ctrl"), hover(C0), click(C0), hover(REF), click(REF), hover(NEAR_A), click(NEAR_A),
     tap("ctrl"), hover(C0))
prog("flat", CAM, hover(C0), click(C0), hover(REF), click(REF), tap("ctrl"), hover(FAR_A), click(FAR_A),
     typed("10"), hover(C0))
prog("flat", CAM, hover(C0), click(C0), hover(REF), click(REF), tap("ctrl"), hover(NEAR_B), typed("33"))
prog("flat", CAM, key("ctrl"), hover(C0), keyup("ctrl"), hover(C0), tap("ctrl"), tap("ctrl"), key("ctrl"),
     keyup("ctrl"))
# arrow-key plane locks (Right=red, Left=green, Up=blue, the same arrow releases)
prog("flat", CAM, key("right"), hover(C0), hover((3, 3, 0)), click((3, 3, 0)), hover(at((3, 3, 0), 60, 0)),
     click(at((3, 3, 0), 60, 0)), hover(at((3, 3, 0), 40, -60)), click(at((3, 3, 0), 40, -60)))
prog("flat", CAM, key("left"), hover(C0), click(C0), hover(at(C0, 0, -70)), click(at(C0, 0, -70)),
     hover(at(C0, 30, -90)), click(at(C0, 30, -90)))
prog("flat", CAM, key("up"), hover(C0), key("up"), hover((3, 3, 0)), key("right"), key("right"), hover((1, 1, 0)),
     esc(), hover((1, 2, 0)))
prog("flat", CAM, key("up"), key("left"), key("up"), hover(C0), click(C0), key("right"), hover(at(C0, 60, 0)))
# plane inference on faces (the disc is aligned to the face under the cursor, coloured by axis)
prog("solids", CAM, hover((1.5, 1.5, 3)), hover((1.6, 1.5, 3)), click((1.5, 1.5, 3)), hover(at((1.5, 1.5, 3), 60, 0)),
     click(at((1.5, 1.5, 3), 60, 0)), hover(at((1.5, 1.5, 3), 45, -50)), click(at((1.5, 1.5, 3), 45, -50)))
prog("solids", CAM, hover((3, 1.5, 1.5)), click((3, 1.5, 1.5)), hover((3, 2.2, 1.5)), click((3, 2.2, 1.5)),
     hover((3, 2.2, 2.2)), click((3, 2.2, 2.2)))
prog("solids", CAM, hover((1.5, 0, 1.5)), click((1.5, 0, 1.5)), hover((2.2, 0, 1.5)), click((2.2, 0, 1.5)),
     hover((2.2, 0, 2.4)), typed("25"))
prog("solids", CAM, hover((6.5, 1.5, 0.75)), click((6.5, 1.5, 0.75)), hover((7.5, 1.5, 0.75)),
     click((7.5, 1.5, 0.75)), hover((7.5, 2, 1.1)), click((7.5, 2, 1.1)))
prog("solids", CAM, hover((6.5, 1.5, 0.75)), hover((1.5, 1.5, 3)), hover((-1, 4, 0)), hover((6.5, 1.5, 0.75)),
     hover((1.5, 0, 1.5)))
# Shift freezes the plane
prog("solids", CAM, hover((1.5, 1.5, 3)), hover((3, 1.5, 1.5), shift=True), hover((6.5, 1.5, 0.75), shift=True),
     hover((6.5, 1.5, 0.75)), hover((3, 1.5, 1.5), shift=True), click((3, 1.5, 1.5), shift=True),
     hover((3, 2, 2)), click((3, 2, 2)))
prog("solids", CAM, hover((-1, 4, 0), shift=True), hover((1.5, 1.5, 3), shift=True), hover((1.5, 1.5, 3)),
     hover((1.5, 1.5, 3), shift=True), key("right"), hover((1.5, 1.5, 3), shift=True))
# click-drag: a custom rotation axis with the live tilt preview
prog("flat", CAM, hover(C0), press(C0), hover(at(C0, 5, 3), drag=True), release(at(C0, 5, 3)), hover(REF),
     click(REF), hover(NEAR_A), click(NEAR_A))
prog("flat", CAM, hover(C0), press(C0), hover(at(C0, 30, 0), drag=True), hover(at(C0, 80, -40), drag=True),
     hover(at(C0, 100, -120), drag=True), release(at(C0, 100, -120)), hover(at(C0, 110, -100)),
     click(at(C0, 110, -100)), hover(at(C0, 60, -90)), click(at(C0, 60, -90)), hover(at(C0, 150, -30)),
     click(at(C0, 150, -30)))
prog("flat", CAM, hover(C0), press(C0), hover(at(C0, 0, -90), drag=True), release(at(C0, 0, -90)), hover(at(C0, 0, -60)),
     esc(), hover(at(C0, 10, 10)))
prog("solids", CAM, hover((1.5, 1.5, 3)), press((1.5, 1.5, 3)), hover((2.5, 1.5, 4), drag=True),
     release((2.5, 1.5, 4)), hover((2.5, 2, 4)), click((2.5, 2, 4)), hover((2.5, 2.5, 4.5)), click((2.5, 2.5, 4.5)))
prog("flat", CAM, hover(C0), press(C0), release(C0), hover(REF), click(REF), hover(NEAR_B), click(NEAR_B))
# the drag threshold sits at 8 px: just under it is a click, just over it tilts
for _dx, _dy in ((8.6, 0), (7.4, 0), (0, 8.6), (0, -7.4), (6.1, 6.1)):
    prog("flat", CAM, hover(C0), press(C0), hover(at(C0, _dx, _dy), drag=True), release(at(C0, _dx, _dy)),
         hover(REF), click(REF), hover(NEAR_A), click(NEAR_A))
# the exact-snap rule: aiming at an endpoint lands the arm on it
prog("flat", CAM, hover((0, 0, 0)), click((0, 0, 0)), hover((4, 0, 0)), click((4, 0, 0)), hover((4, 4, 0)),
     click((4, 4, 0)))
prog("flat", CAM, hover((0, 0, 0)), click((0, 0, 0)), hover((4, 0, 0)), click((4, 0, 0)), hover((4.0, 2.7, 0)),
     click((4.0, 2.7, 0)), hover((0, 4, 0)), hover((9, 2, 0)), click((9, 2, 0)))
prog("solids", CAM, hover((0, 0, 3)), click((0, 0, 3)), hover((3, 0, 3)), click((3, 0, 3)), hover((3, 3, 3)),
     click((3, 3, 3)))
# Esc at every stage
prog("flat", CAM, hover(C0), click(C0), esc(), hover(REF), click(REF), esc(), esc(), hover(NEAR_A))
prog("flat", CAM, hover(C0), click(C0), hover(REF), click(REF), hover(NEAR_A), esc(), hover(NEAR_B), click(NEAR_B))
# the tool re-picked
prog("flat", CAM, key("up"), hover(C0), click(C0), hover(REF), deactivate(), activate(), hover(C0), key("up"),
     activate(), hover(C0))
prog("flat", CAM, tap("ctrl"), hover(C0), activate(), hover(C0), tap("ctrl"))
# a degenerate reference (the same point): ignored; a sweep onto the centre
prog("flat", CAM, hover(C0), click(C0), click(C0), hover(REF), click(REF), hover(C0), click(C0), hover(NEAR_A),
     click(NEAR_A))
# undo mid-tool
prog("flat", CAM, hover(C0), click(C0), hover(REF), click(REF), hover(NEAR_A), click(NEAR_A), hover(C0), click(C0),
     undo(), hover(REF), click(REF), undo(), redo(), hover(FAR_A), click(FAR_A), undo(), undo(), redo())
# groups: a face of an instance (local normal), a classic group, a container
prog("groups", CAM, hover((6.5, 0.5, 1)), click((6.5, 0.5, 1)), hover((7.2, 0.5, 1)), click((7.2, 0.5, 1)),
     hover((7.2, 0.8, 1)), click((7.2, 0.8, 1)))
prog("groups", CAM, hover((9, 2, 0.5)), hover((9.3, 2.1, 1)), hover((9.5, 2.5, 0.5)), click((9.5, 2.5, 0.5)),
     hover((10, 2.5, 0.5)), typed("40"))
prog("groups", CAM, hover((-4, 1, 1.5)), click((-4, 1, 1.5)), hover((-3, 1, 1.5)), click((-3, 1, 1.5)),
     hover((-3, 1.6, 1.5)), click((-3, 1.6, 1.5)))
prog("container", CAM, hover((6.5, 1.5, 1)), hover((1.5, 1.5, 0)), hover((6.8, 3.4, 1)), click((6.8, 3.4, 1)),
     hover((7.4, 3.4, 1)), click((7.4, 3.4, 1)), hover((7.4, 3.8, 1)), click((7.4, 3.8, 1)))
prog("curves", CAM, hover((0.5, 0.5, 0)), click((0.5, 0.5, 0)), hover((1.5, 0.5, 0)), click((1.5, 0.5, 0)),
     hover((1.5, 1.5, 0)), click((1.5, 1.5, 0)))
prog("curves", CAM, hover((7, 1, 0)), click((7, 1, 0)), hover((8, 1, 0)), click((8, 1, 0)), hover((7.5, 1.9, 0)),
     click((7.5, 1.9, 0)))
# the cameras
prog("flat", CAM_TOP, hover(C0), click(C0), hover((3, 2, 0)), click((3, 2, 0)), hover((3.2, 3, 0)),
     click((3.2, 3, 0)))
prog("flat", CAM_ORTHO, hover(C0), click(C0), hover(REF), click(REF), hover(FAR_A), click(FAR_A))
prog("flat", CAM_LOW, hover(C0), click(C0), hover((3, 2, 0)), click((3, 2, 0)), hover((3.6, 3.5, 0)),
     click((3.6, 3.5, 0)), typed("20"))
prog("solids", CAM_LOW, hover((1.5, 0, 1.5)), key("right"), hover((1.5, 0, 1.5)), click((1.5, 0, 1.5)),
     hover((2.5, 0, 1.5)), click((2.5, 0, 1.5)), hover((2.5, 0.5, 1.5)), click((2.5, 0.5, 1.5)))
prog("flat", CAM, hover(C0), click(C0), hover(REF), camera(dict(CAM, yaw=-20.0)), hover(NEAR_A), click(NEAR_A))
prog("flat", CAM, hover(px(5, 5)), click(px(5, 5)), hover(px(300, 300)), click(px(300, 300)), hover(px(310, 200)),
     click(px(310, 200)))


# ---- seeded random programs -----------------------------------------------------------------

def random_program(seed: int, scene: str, cam: dict, n: int = 24):
    rnd = random.Random(seed)
    pts = {
        "flat": [(2, 2, 0), (4, 4, 0), (0, 0, 0), (3, 2, 0), (1, 3, 0), (7.5, 1, 0), (5, 5, 0), (-3, 1, 0.5)],
        "solids": [(1.5, 1.5, 3), (3, 1.5, 1.5), (1.5, 0, 1.5), (6.5, 1.5, 0.75), (-1, 4, 0), (3, 3, 3), (2, 2, 0)],
        "groups": [(6.5, 0.5, 1), (9, 2, 0.5), (-4, 1, 1.5), (2, 2, 0), (4, 4, 0), (3, 1, 0)],
        "container": [(6.5, 1.5, 1), (1.5, 1.5, 0), (6.8, 3.4, 1), (2, 2, 0), (8, 3, 0)],
        "curves": [(0.5, 0.5, 0), (7, 1, 0), (2.5, 2.5, 0), (7.5, 1.9, 0), (6, 4, 0), (4, 4, 0)],
    }[scene]
    evs = []
    for _ in range(n):
        r = rnd.random()
        p = rnd.choice(pts)
        q = (p[0] + rnd.choice((0, 0, 0.3, -0.4, 1.0)), p[1] + rnd.choice((0, 0, 0.5, -0.5)), p[2])
        w = at(q, rnd.choice((0, 0, 30, 70, 130)), rnd.choice((0, -40, 40, 90)))
        if r < 0.28:
            evs.append(hover(w))
        elif r < 0.52:
            evs.append(click(w))
        elif r < 0.58:
            evs.append(tap("ctrl"))
        elif r < 0.68:
            evs.append(typed(rnd.choice(["45", "3:12", "-30", "12.5", "90", "1:6", "0", "7"])))
        elif r < 0.73:
            evs.append(esc())
        elif r < 0.78:
            evs.append(undo())
        elif r < 0.81:
            evs.append(redo())
        elif r < 0.88:
            evs.append(key(rnd.choice(["right", "left", "up"])))
        elif r < 0.92:
            evs.append(press(w))
        elif r < 0.95:
            evs.append(release(w))
        else:
            evs.append(hover(w, shift=rnd.random() < 0.6))
    return (scene, cam, evs)


for _seed in range(300, 340):
    _sc = ("flat", "solids", "groups", "container", "curves")[_seed % 5]
    PROGRAMS.append(random_program(_seed, _sc, (CAM, CAM_ORTHO, CAM_LOW, CAM_TOP)[_seed % 4]))


def extra_probe(tool, vp, ids):
    last = tool._last
    x = base_probe(tool)
    x.update({
        "guides": bool(tool._guides),
        "last": last is not None,
        "last_sign": None if last is None else d64(last["sign"]),
        "last_a0": None if last is None else d64(last["a0"]),
        "last_u": None if last is None else pbits(last["u"]),
        "last_v": None if last is None else pbits(last["v"]),
        "last_live": last is not None and any(g is last["guide"] for g in vp.scene.guides),
    })
    return x
