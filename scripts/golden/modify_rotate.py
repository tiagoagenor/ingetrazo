# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
"""The Rotate tool's scenes and programs for ``make_modify_tool_fixtures.py``
(``tools/rotate.py``, docs/rust-rewrite-plan.md M3.13). Section planes,
reference images and Move's rotation grips are not ported: no program
selects or hovers them."""
from __future__ import annotations

import random

from _modify_harness import (ALT, CTRL, SHIFT, V, above, at, box, box_faces, click, dbl, deactivate, esc,  # noqa: F401
                             group, hover, key, keyup, press, redo, release, select, tap, typed, undo, xf,
                             Mesh, Scene, activate, camera, pbits, px)
from modify_protractor import (CAM, CAM_LOW, CAM_ORTHO, CAM_TOP, SCENES as _PSCENES, base_probe, d64)  # noqa: F401
from tools.rotate import RotateTool

NAME = "rotate"


def make_tool():
    return RotateTool()


SCENES = dict(_PSCENES)

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


def turn(c, ref, sweep, *extra):
    """centre, reference arm, sweep, third click."""
    return [hover(c), click(c), hover(ref), click(ref), hover(sweep), click(sweep), *extra]


# a face turned about its centre: near the disc (ticks) and far (free angle)
prog("flat", CAM, select(("face", 0)), *turn(C0, REF, NEAR_A), undo(), redo())
prog("flat", CAM, select(("face", 0)), hover(C0), click(C0), hover(REF), click(REF), hover(at(C0, 45, -52)),
     hover(NEAR_A), hover(FAR_A), hover(at(C0, 151, -91)), click(FAR_A), undo())
prog("flat", CAM, select(("face", 1)), *turn((7.5, 1, 0), at((7.5, 1, 0), 80, 0), at((7.5, 1, 0), 130, 60)), undo())
# nothing selected: the click grabs the loose face / edge under the cursor
prog("flat", CAM, hover((7.5, 1, 0)), click((7.5, 1, 0)), hover(at((7.5, 1, 0), 80, 0)),
     click(at((7.5, 1, 0), 80, 0)), hover(at((7.5, 1, 0), 40, -50)), click(at((7.5, 1, 0), 40, -50)))
prog("flat", CAM, hover((-3, 1, 0.5)), click((-3, 1, 0.5)), hover(at((-3, 1, 0.5), 70, 0)),
     click(at((-3, 1, 0.5), 70, 0)), hover(at((-3, 1, 0.5), 60, -60)), click(at((-3, 1, 0.5), 60, -60)))
prog("flat", CAM, select(("edge", 8)), key("right"), hover((-3, 1, 0.5)), click((-3, 1, 0.5)), hover((-3, 1.5, 0.5)),
     click((-3, 1.5, 0.5)), hover((-3, 1.5, 1.2)), click((-3, 1.5, 1.2)))
# nothing to grab
prog("flat", CAM, hover(px(40, 40)), click(px(40, 40)), hover(C0), click(C0), hover(REF), click(REF))
# typed angle: the sign follows the sweep; a slope arrives as degrees
prog("flat", CAM, select(("face", 0)), hover(C0), click(C0), hover(REF), click(REF), hover(NEAR_A), typed("30"))
prog("flat", CAM, select(("face", 0)), hover(C0), click(C0), hover(REF), click(REF), hover(NEAR_B), typed("30"), undo())
prog("flat", CAM, select(("face", 0)), hover(C0), click(C0), hover(REF), click(REF), hover(FAR_B), typed("3:12"))
prog("flat", CAM, select(("face", 0)), hover(C0), click(C0), hover(REF), click(REF), typed("45"))
prog("flat", CAM, select(("face", 0)), hover(C0), click(C0), hover(REF), click(REF), hover(NEAR_A), typed("0"))
# the hot retype after the commit
prog("flat", CAM, select(("face", 0)), *turn(C0, REF, NEAR_A), typed("60"), typed("-20"), typed("1:6"), typed("0"),
     undo(), typed("15"))
prog("flat", CAM, select(("face", 0)), *turn(C0, REF, NEAR_B), typed("10"), redo(), typed("5"), hover(C0),
     click(C0), typed("5"))
prog("flat", CAM, select(("face", 0), ("face", 1), ("edge", 8)), *turn(C0, REF, FAR_A), typed("25"), typed("-25"))
# Ctrl: copy mode (the + cursor, the flashes, the wireframe preview), Ctrl again to cancel it
prog("flat", CAM, select(("face", 0)), tap("ctrl"), hover(C0), click(C0), hover(REF), click(REF), hover(NEAR_A),
     hover(FAR_A), click(FAR_A), undo(), redo())
prog("flat", CAM, select(("face", 0)), hover(C0), click(C0), hover(REF), click(REF), hover(NEAR_A), tap("ctrl"),
     hover(FAR_A), tap("ctrl"), hover(NEAR_B), click(NEAR_B))
prog("flat", CAM, select(("face", 0)), hover(C0), click(C0), hover(REF), click(REF), tap("ctrl"), hover(NEAR_B),
     typed("40"), typed("3x"), typed("/3"), typed("2x"), undo(), redo())
prog("flat", CAM, select(("face", 0), ("edge", 8)), tap("ctrl"), *turn(C0, REF, NEAR_A), typed("4x"), typed("/2"),
     typed("x3"), typed("5"), typed("2x"))
prog("flat", CAM, select(("face", 0)), tap("ctrl"), *turn(C0, REF, NEAR_A), esc(), typed("2x"), typed("20"))
prog("flat", CAM, select(("face", 1)), tap("ctrl"), *turn((7.5, 1, 0), at((7.5, 1, 0), 80, 0),
     at((7.5, 1, 0), 40, -50)), typed("3x"), click((7.5, 1, 0)), typed("2x"))
prog("flat", CAM, select(("face", 0)), tap("ctrl"), hover(C0), tap("ctrl"), hover(C0), click(C0), hover(REF),
     click(REF), hover(NEAR_A), click(NEAR_A))
prog("flat", CAM, select(("face", 0)), key("ctrl"), hover(C0), keyup("ctrl"), hover(C0), key("ctrl"), keyup("ctrl"),
     hover(C0), click(C0))
prog("flat", CAM, select(("edge", 8)), tap("ctrl"), hover((-3, 1, 0.5)), click((-3, 1, 0.5)),
     hover(at((-3, 1, 0.5), 60, 0)), click(at((-3, 1, 0.5), 60, 0)), hover(at((-3, 1, 0.5), 40, -50)),
     click(at((-3, 1, 0.5), 40, -50)), typed("2x"))
# arrow-key plane locks, Shift freeze, the plane inferred from faces
prog("flat", CAM, select(("face", 0)), key("right"), hover(C0), click(C0), hover(at(C0, 0, -70)),
     click(at(C0, 0, -70)), hover(at(C0, 30, -90)), click(at(C0, 30, -90)))
prog("flat", CAM, select(("face", 0)), key("left"), hover(C0), key("left"), hover(C0), key("up"), hover(C0),
     click(C0), hover(REF), click(REF), hover(NEAR_A), esc(), hover(C0))
prog("flat", CAM, select(("face", 0)), hover(C0), hover((7.5, 1, 0), shift=True), hover(C0, shift=True),
     click(C0, shift=True), hover(REF), click(REF), hover(NEAR_A), click(NEAR_A))
prog("solids", CAM, select(("face", 1)), hover((1.5, 1.5, 3)), click((1.5, 1.5, 3)), hover(at((1.5, 1.5, 3), 60, 0)),
     click(at((1.5, 1.5, 3), 60, 0)), hover(at((1.5, 1.5, 3), 40, -50)), click(at((1.5, 1.5, 3), 40, -50)), undo())
prog("solids", CAM, select(("face", 5)), hover((3, 1.5, 1.5)), click((3, 1.5, 1.5)), hover((3, 2.2, 1.5)),
     click((3, 2.2, 1.5)), hover((3, 2.2, 2.2)), click((3, 2.2, 2.2)))
prog("solids", CAM, select(("face", 6)), hover((6.5, 1.5, 0.75)), click((6.5, 1.5, 0.75)), hover((7.5, 1.5, 0.75)),
     click((7.5, 1.5, 0.75)), hover((7.5, 2, 1.1)), click((7.5, 2, 1.1)), typed("20"))
prog("solids", CAM, select(("face", 1)), hover((1.5, 1.5, 3)), hover((3, 1.5, 1.5), shift=True), hover((1.5, 0, 1.5)),
     hover((3, 1.5, 1.5), shift=True), click((3, 1.5, 1.5), shift=True), hover((3, 2.2, 1.5)), click((3, 2.2, 1.5)))
prog("solids", CAM, select(("edge", 8)), hover((-2, 1.5, 0)), click((-2, 1.5, 0)), hover((-2, 2.5, 0)),
     click((-2, 2.5, 0)), hover((-1, 2.5, 0)), click((-1, 2.5, 0)))
# click-drag from the centre: the custom axis, its live tilt preview
prog("flat", CAM, select(("face", 0)), hover(C0), press(C0), hover(at(C0, 5, 3), drag=True), release(at(C0, 5, 3)),
     hover(REF), click(REF), hover(NEAR_A), click(NEAR_A))
prog("flat", CAM, select(("face", 0)), hover(C0), press(C0), hover(at(C0, 30, 0), drag=True),
     hover(at(C0, 80, -40), drag=True), hover(at(C0, 100, -120), drag=True), release(at(C0, 100, -120)),
     hover(at(C0, 110, -100)), click(at(C0, 110, -100)), hover(at(C0, 60, -90)), click(at(C0, 60, -90)),
     hover(at(C0, 150, -30)), click(at(C0, 150, -30)), undo())
prog("flat", CAM, select(("face", 0), ("edge", 8)), hover(C0), press(C0), hover(at(C0, 0, -90), drag=True),
     release(at(C0, 0, -90)), hover(at(C0, 0, -60)), esc(), hover(at(C0, 10, 10)))
prog("solids", CAM, select(("face", 1), ("face", 0)), hover((1.5, 1.5, 3)), press((1.5, 1.5, 3)),
     hover((2.5, 1.5, 4), drag=True), release((2.5, 1.5, 4)), hover((2.5, 2, 4)), click((2.5, 2, 4)),
     hover((2.5, 2.5, 4.5)), click((2.5, 2.5, 4.5)))
# the drag threshold sits at 8 px: just under it is a click, just over it tilts
for _dx, _dy in ((8.6, 0), (7.4, 0), (0, 8.6), (0, -7.4), (6.1, 6.1)):
    prog("flat", CAM, select(("face", 0)), hover(C0), press(C0), hover(at(C0, _dx, _dy), drag=True),
         release(at(C0, _dx, _dy)), hover(REF), click(REF), hover(NEAR_A), click(NEAR_A))
# the exact-snap rule: aiming at corners lands the arm on them
prog("flat", CAM, select(("face", 1)), hover((6, 0, 0)), click((6, 0, 0)), hover((9, 0, 0)), click((9, 0, 0)),
     hover((9, 2, 0)), click((9, 2, 0)))
prog("flat", CAM, select(("face", 0)), hover((0, 0, 0)), click((0, 0, 0)), hover((4, 0, 0)), click((4, 0, 0)),
     hover((4.0, 2.7, 0)), click((4.0, 2.7, 0)))
prog("flat", CAM, select(("face", 0)), hover((0, 0, 0)), click((0, 0, 0)), hover((4, 0, 0)), click((4, 0, 0)),
     hover((4, 4, 0)), click((4, 4, 0)), undo(), hover((0, 4, 0)))
# Esc at every stage, undo mid-tool, the tool re-picked
prog("flat", CAM, select(("face", 0)), hover(C0), click(C0), esc(), hover(REF), click(REF), esc(), esc(), hover(NEAR_A))
prog("flat", CAM, select(("face", 0)), hover(C0), click(C0), hover(REF), click(REF), hover(NEAR_A), hover(FAR_A), esc(),
     hover(NEAR_B), click(NEAR_B))
prog("flat", CAM, select(("face", 0)), hover(C0), click(C0), hover(REF), click(REF), hover(FAR_A), deactivate(),
     activate(), hover(C0), click(C0), hover(REF), click(REF), hover(NEAR_B), tap("ctrl"), deactivate(), activate(),
     hover(NEAR_B))
prog("flat", CAM, select(("face", 0)), hover(C0), click(C0), hover(REF), click(REF), hover(FAR_A), undo(), hover(FAR_B),
     redo(), click(FAR_B), undo(), undo(), redo(), redo())
prog("flat", CAM, select(("face", 0)), key("up"), hover(C0), click(C0), hover(REF), deactivate(), activate(),
     hover(C0), key("up"))
# classic group, instances, containers
G_CLASSIC = (-4, 1, 0.75)
G_A = (6.5, 0.5, 0.5)
G_B = (9, 2, 0.5)
prog("groups", CAM, hover(G_CLASSIC), click(G_CLASSIC), hover(at(G_CLASSIC, 80, 0)), click(at(G_CLASSIC, 80, 0)),
     hover(at(G_CLASSIC, 40, -50)), hover(at(G_CLASSIC, 150, -90)), click(at(G_CLASSIC, 150, -90)), undo(), redo())
prog("groups", CAM, hover(G_A), click(G_A), hover(at(G_A, 80, 0)), click(at(G_A, 80, 0)), hover(at(G_A, 40, -50)),
     click(at(G_A, 40, -50)), undo(), redo())
prog("groups", CAM, hover(G_B), click(G_B), hover(at(G_B, 80, 0)), click(at(G_B, 80, 0)), hover(at(G_B, 40, 50)),
     typed("37"), typed("-10"))
prog("groups", CAM, select(("group", [1]), ("group", [2])), hover(G_A), click(G_A), hover(at(G_A, 80, 0)),
     click(at(G_A, 80, 0)), hover(at(G_A, 40, -50)), click(at(G_A, 40, -50)), undo())
prog("groups", CAM, select(("group", [0]), ("group", [1]), ("face", 0)), hover(C0), click(C0), hover(REF),
     click(REF), hover(NEAR_A), click(NEAR_A), undo(), redo())
prog("groups", CAM, hover(G_A), click(G_A), hover(at(G_A, 80, 0)), click(at(G_A, 80, 0)), hover(at(G_A, 40, -50)),
     esc(), hover(at(G_A, 40, 50)), click(at(G_A, 40, 50)))
prog("groups", CAM, hover(G_A), click(G_A), hover(at(G_A, 80, 0)), click(at(G_A, 80, 0)), hover(at(G_A, 40, -50)),
     deactivate(), activate(), hover(G_A))
prog("groups", CAM, hover(G_CLASSIC), click(G_CLASSIC), tap("ctrl"), hover(at(G_CLASSIC, 80, 0)),
     click(at(G_CLASSIC, 80, 0)), hover(at(G_CLASSIC, 40, -50)), click(at(G_CLASSIC, 40, -50)), typed("2x"), undo(),
     redo(), typed("/3"))
prog("groups", CAM, hover(G_A), click(G_A), tap("ctrl"), hover(at(G_A, 80, 0)), click(at(G_A, 80, 0)),
     hover(at(G_A, 40, -50)), click(at(G_A, 40, -50)), typed("3x"), typed("/2"), typed("25"))
prog("groups", CAM, select(("group", [1]), ("face", 0)), tap("ctrl"), hover(C0), click(C0), hover(REF), click(REF),
     hover(NEAR_A), hover(FAR_A), click(FAR_A), typed("2x"))
prog("groups", CAM, hover(G_B), click(G_B), tap("ctrl"), hover(at(G_B, 80, 0)), click(at(G_B, 80, 0)), typed("45"),
     typed("2x"))
prog("container", CAM, hover((6.5, 1.5, 0.5)), click((6.5, 1.5, 0.5)), hover(at((6.5, 1.5, 0.5), 80, 0)),
     click(at((6.5, 1.5, 0.5), 80, 0)), hover(at((6.5, 1.5, 0.5), 40, -50)), click(at((6.5, 1.5, 0.5), 40, -50)),
     undo(), redo())
prog("container", CAM, select(("group", [0])), tap("ctrl"), hover((6.5, 1.5, 0.5)), click((6.5, 1.5, 0.5)),
     hover(at((6.5, 1.5, 0.5), 80, 0)), click(at((6.5, 1.5, 0.5), 80, 0)), hover(at((6.5, 1.5, 0.5), 40, -50)),
     click(at((6.5, 1.5, 0.5), 40, -50)), typed("2x"))
prog("container", CAM, hover((1.5, 1.5, 0)), click((1.5, 1.5, 0)), hover(at((1.5, 1.5, 0), 70, 0)),
     click(at((1.5, 1.5, 0), 70, 0)), hover(at((1.5, 1.5, 0), 40, 50)), typed("20"))
# curves, soft edges, a face with a hole
CURVE = [("edge", i) for i in range(8, 20)]
prog("curves", CAM, select(*CURVE), hover((7, 1, 0)), click((7, 1, 0)), hover(at((7, 1, 0), 80, 0)),
     click(at((7, 1, 0), 80, 0)), hover(at((7, 1, 0), 40, -50)), click(at((7, 1, 0), 40, -50)), undo(), redo())
prog("curves", CAM, select(*CURVE), tap("ctrl"), hover((7, 1, 0)), click((7, 1, 0)), hover(at((7, 1, 0), 80, 0)),
     click(at((7, 1, 0), 80, 0)), hover(at((7, 1, 0), 40, -50)), click(at((7, 1, 0), 40, -50)), typed("3x"),
     typed("/4"), undo())
prog("curves", CAM, select(("face", 0), ("edge", 20), ("edge", 21)), tap("ctrl"), hover((2.5, 2.5, 0)),
     click((2.5, 2.5, 0)), hover(at((2.5, 2.5, 0), 80, 0)), click(at((2.5, 2.5, 0), 80, 0)),
     hover(at((2.5, 2.5, 0), 40, -50)), click(at((2.5, 2.5, 0), 40, -50)), typed("2x"))
prog("curves", CAM, select(("face", 0)), hover((2.5, 2.5, 0)), click((2.5, 2.5, 0)), hover(at((2.5, 2.5, 0), 80, 0)),
     click(at((2.5, 2.5, 0), 80, 0)), hover(at((2.5, 2.5, 0), 40, -50)), click(at((2.5, 2.5, 0), 40, -50)))
prog("curves", CAM, hover((7, 0, 0)), click((7, 0, 0)), hover((8, 0, 0)), click((8, 0, 0)), hover((7.5, 0.8, 0)),
     click((7.5, 0.8, 0)))
prog("curves", CAM, select(("edge", 20)), hover((7, 4, 0.5)), click((7, 4, 0.5)), hover(at((7, 4, 0.5), 80, 0)),
     click(at((7, 4, 0.5), 80, 0)), hover(at((7, 4, 0.5), 40, -50)), click(at((7, 4, 0.5), 40, -50)))
# cameras
prog("flat", CAM_TOP, select(("face", 0)), *turn(C0, (3, 2, 0), (3.2, 3, 0)))
prog("flat", CAM_ORTHO, select(("face", 0)), *turn(C0, REF, FAR_A), undo())
prog("flat", CAM_LOW, select(("face", 0)), hover(C0), click(C0), hover((3, 2, 0)), click((3, 2, 0)),
     hover((3.6, 3.5, 0)), click((3.6, 3.5, 0)), typed("20"))
prog("flat", CAM, select(("face", 0)), hover(C0), click(C0), hover(REF), camera(dict(CAM, yaw=-20.0)), hover(NEAR_A),
     click(NEAR_A))


# ---- seeded random programs ------------------------------------------------------------------

def random_program(seed: int, scene: str, cam: dict, n: int = 28):
    rnd = random.Random(seed)
    pts = {
        "flat": [(2, 2, 0), (4, 4, 0), (0, 0, 0), (3, 2, 0), (1, 3, 0), (7.5, 1, 0), (6, 0, 0), (-3, 1, 0.5)],
        "solids": [(1.5, 1.5, 3), (3, 1.5, 1.5), (1.5, 0, 1.5), (6.5, 1.5, 0.75), (-2, 1.5, 0), (3, 3, 3), (2, 2, 0)],
        "groups": [(6.5, 0.5, 0.5), (9, 2, 0.5), (-4, 1, 0.75), (2, 2, 0), (4, 4, 0), (3, 1, 0)],
        "container": [(6.5, 1.5, 0.5), (1.5, 1.5, 0), (6.8, 3.4, 0.5), (2, 2, 0), (8, 3, 0)],
        "curves": [(0.5, 0.5, 0), (7, 1, 0), (2.5, 2.5, 0), (7.5, 1.9, 0), (6, 4, 0), (4, 4, 0)],
    }[scene]
    refs = {
        "flat": [("face", 0), ("face", 1), ("edge", 8)],
        "solids": [("face", 0), ("face", 1), ("face", 6), ("edge", 13)],
        "groups": [("group", [0]), ("group", [1]), ("group", [2]), ("face", 0)],
        "container": [("group", [0]), ("face", 0)],
        "curves": [("face", 0), ("edge", 9), ("edge", 20), ("edge", 21)],
    }[scene]
    evs = [select(rnd.choice(refs))] if rnd.random() < 0.7 else []
    for _ in range(n):
        r = rnd.random()
        p = rnd.choice(pts)
        q = (p[0] + rnd.choice((0, 0, 0.3, -0.4)), p[1] + rnd.choice((0, 0, 0.5, -0.5)), p[2])
        w = at(q, rnd.choice((0, 0, 30, 70, 130)), rnd.choice((0, -40, 40, 90)))
        if r < 0.27:
            evs.append(hover(w))
        elif r < 0.52:
            evs.append(click(w))
        elif r < 0.60:
            evs.append(tap("ctrl"))
        elif r < 0.70:
            evs.append(typed(rnd.choice(["45", "3:12", "-30", "12.5", "90", "3x", "/3", "2x", "7", "0"])))
        elif r < 0.75:
            evs.append(esc())
        elif r < 0.80:
            evs.append(undo())
        elif r < 0.83:
            evs.append(redo())
        elif r < 0.86:
            evs.append(select(rnd.choice(refs), mode=rnd.choice(["replace", "add"])))
        elif r < 0.90:
            evs.append(key(rnd.choice(["right", "left", "up"])))
        elif r < 0.93:
            evs.append(press(w))
        elif r < 0.95:
            evs.append(release(w))
        elif r < 0.97:
            evs.append(rnd.choice([deactivate(), activate()]))
        else:
            evs.append(hover(w, shift=rnd.random() < 0.6))
    return (scene, cam, evs)


for _seed in range(400, 450):
    _sc = ("flat", "solids", "groups", "container", "curves")[_seed % 5]
    PROGRAMS.append(random_program(_seed, _sc, (CAM, CAM_ORTHO, CAM_LOW, CAM_TOP)[_seed % 4]))


def extra_probe(tool, vp, ids):
    last = tool._last
    stack = vp.history.undo_stack
    x = base_probe(tool)
    x.update({
        "copy": bool(tool._copy),
        "n_groups": len(tool._groups),
        "n_pos": len(tool._positions),
        "n_verts": len(tool._verts),
        "sel_faces": len(tool._sel_faces),
        "sel_edges": len(tool._sel_edges),
        "base_segments": len(tool._base_segments),
        "preview_deg": d64(tool._preview_deg),
        "has_orig": tool._orig is not None,
        "vp_preview": bool(getattr(tool, "_vp_preview", False)),
        "last": last is not None,
        "last_copy": None if last is None else bool(last["copy"]),
        "last_deg": None if last is None else d64(last["deg"]),
        "last_sign": None if last is None else d64(last["sign"]),
        "last_top": last is not None and bool(stack) and stack[-1] is last["cmd"],
    })
    return x
