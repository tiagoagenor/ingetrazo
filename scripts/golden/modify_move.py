# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
"""The Move tool's scenes and programs for ``make_modify_tool_fixtures.py``
(``tools/move.py``, docs/rust-rewrite-plan.md M3.13)."""
from __future__ import annotations

import random

from _modify_harness import (ALT, CTRL, SHIFT, V, above, at, box, box_faces, click, dbl, deactivate, esc,  # noqa: F401
                             group, hover, key, keyup, press, redo, release, select, tap, typed, undo, xf,
                             Mesh, Scene, activate, camera, pbits, px)
from tools.move import MoveTool

NAME = "move"
make_tool = MoveTool

CAM = dict(target=(2, 2, 0.8), distance=14.0, yaw=-50.0, pitch=30.0, w=1000, h=600)
CAM_ORTHO = dict(target=(2, 2, 0.8), distance=14.0, yaw=-50.0, pitch=30.0, w=1000, h=600, perspective=False)
CAM_LOW = dict(target=(2, 2, 1.0), distance=16.0, yaw=-30.0, pitch=8.0, w=1000, h=600)
CAM_TOP = dict(target=(2, 2, 0.0), distance=16.0, yaw=-90.0, pitch=89.0, w=1000, h=600)


def scn_plate() -> Scene:
    s = Scene()
    m = s.mesh
    m.add_face([V(0, 0, 0), V(4, 0, 0), V(4, 4, 0), V(0, 4, 0)])
    m.add_face([V(6, 0, 0), V(9, 0, 0), V(9, 2, 0), V(6, 2, 0)])
    m.add_edge(V(-3, 0, 0), V(-3, 2, 1))
    return s


def scn_cube() -> Scene:
    s = Scene()
    box_faces(s.mesh, 0, 0, 0, 3, 3, 3)
    s.mesh.add_face([V(6, 0, 0), V(8, 0, 0), V(8, 2, 0), V(6, 2, 0)])
    return s


def scn_ridge() -> Scene:
    """A box 4x4x2 whose top is two faces meeting at y=2 (a ridge to lift)."""
    s = Scene()
    m = s.mesh
    for quad in ([V(0, 0, 0), V(0, 4, 0), V(4, 4, 0), V(4, 0, 0)],
                 [V(0, 0, 0), V(4, 0, 0), V(4, 0, 2), V(0, 0, 2)],
                 [V(0, 4, 0), V(0, 4, 2), V(4, 4, 2), V(4, 4, 0)],
                 [V(0, 0, 0), V(0, 0, 2), V(0, 4, 2), V(0, 4, 0)],
                 [V(4, 0, 0), V(4, 4, 0), V(4, 4, 2), V(4, 0, 2)],
                 [V(0, 0, 2), V(4, 0, 2), V(4, 2, 2), V(0, 2, 2)],
                 [V(0, 2, 2), V(4, 2, 2), V(4, 4, 2), V(0, 4, 2)]):
        m.add_face(quad)
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


def scn_tight() -> Scene:
    """Squares that touch: a move can weld or be refused."""
    s = Scene()
    m = s.mesh
    m.add_face([V(0, 0, 0), V(2, 0, 0), V(2, 2, 0), V(0, 2, 0)])
    m.add_face([V(2, 0, 0), V(4, 0, 0), V(4, 2, 0), V(2, 2, 0)])
    m.add_face([V(0, 3, 0), V(2, 3, 0), V(2, 5, 0), V(0, 5, 0)])
    m.add_edge(V(5, 0, 0), V(7, 0, 0))
    m.add_edge(V(5, 0, 0), V(5, 2, 0))
    return s


SCENES = {"plate": scn_plate, "cube": scn_cube, "ridge": scn_ridge, "groups": scn_groups,
          "container": scn_container, "tight": scn_tight}

# ---- programs -----------------------------------------------------------------------------

PROGRAMS: list = []


def prog(scene, cam, *events):
    PROGRAMS.append((scene, cam, list(events)))


C0 = (2, 2, 0)           # centre of the plate
# a face grabbed by a click, moved, dropped
prog("plate", CAM, hover(C0), click(C0), hover((3, 2, 0)), hover((4, 3, 0)), hover((5, 3, 0)),
     click((5, 3, 0)), undo(), redo())
# the rubber band, a label, the axis magnet: moving 'along x'
prog("plate", CAM, hover(C0), click(C0), hover(at((5, 2, 0), 0, 3)), click(at((5, 2, 0), 0, 3)))
# an edge grabbed on its body
prog("plate", CAM, hover((2, 0, 0)), click((2, 0, 0)), hover((2, -1, 0)), hover((2, -2, 0)), click((2, -2, 0)))
# a corner grabbed on the Endpoint snap: only that vertex moves, the face deforms
prog("plate", CAM, hover((4, 4, 0)), click((4, 4, 0)), hover((5, 5, 0)), hover((5, 6, 0)), click((5, 6, 0)),
     undo())
# the loose edge
prog("plate", CAM, hover((-3, 1, 0.5)), click((-3, 1, 0.5)), hover((-3, 1, 2.5)), click((-3, 1, 2.5)))
# typed distance along the current direction
prog("plate", CAM, hover(C0), click(C0), hover((3, 2, 0)), typed("2"), undo(), redo())
prog("plate", CAM, hover(C0), click(C0), hover((2, 3, 0)), typed("-1.5"))
prog("plate", CAM, hover(C0), click(C0), hover((2, 3, 0)), typed("1;2;3"))
prog("plate", CAM, hover(C0), click(C0), hover((2, 3, 0)), typed("1;2"), typed("0.5m"), typed("abc"))
prog("plate", CAM, hover(C0), click(C0), typed("2"), hover((3, 3, 0)), typed("30cm"))
# Esc mid-move and with a selection
prog("plate", CAM, hover(C0), click(C0), hover((3, 3, 0)), hover((4, 3, 0)), esc(), hover((4, 3, 0)))
prog("plate", CAM, select(("face", 0)), hover((6.5, 0.5, 0)), click((6.5, 0.5, 0)), hover((7.5, 1.5, 0)), esc(), esc())
# a selection is moved from any click
prog("plate", CAM, select(("face", 1)), hover(C0), click(C0), hover(at(C0, 60, 20)), click(at(C0, 60, 20)), undo())
prog("plate", CAM, select(("face", 0), ("face", 1), ("edge", 8)), hover(C0), click(C0), hover((3, 3, 0)),
     click((3, 3, 0)), undo(), redo(), undo())
prog("plate", CAM, select(("edge", 0)), hover((2, 0, 0)), click((2, 0, 0)), hover((2, 1, 0)), typed("0.7"))
# copy: Ctrl tapped before the grab, after it, and its array windows
prog("plate", CAM, hover(C0), click(C0), tap("ctrl"), hover((3, 3, 0)), hover((5, 3, 0)), click((5, 3, 0)))
prog("plate", CAM, hover(C0), click(C0), tap("ctrl"), hover((5, 2, 0)), click((5, 2, 0)), typed("3x"),
     typed("/3"), typed("2x"), undo(), redo())
prog("plate", CAM, hover(C0), click(C0), tap("ctrl"), hover((5, 2, 0)), click((5, 2, 0)), typed("x4"),
     typed("5*"), esc(), typed("3x"))
prog("plate", CAM, hover(C0), click(C0), tap("ctrl"), hover((5, 2, 0)), typed("3x2m"), typed("2x"))
prog("plate", CAM, hover(C0), click(C0), tap("ctrl"), hover((5, 2, 0)), typed("2"), typed("4x"), typed("/2"))
prog("plate", CAM, hover(C0), click(C0), hover((3, 2, 0)), tap("ctrl"), hover((4, 2, 0)), tap("ctrl"),
     hover((4.5, 2, 0)), click((4.5, 2, 0)))
prog("plate", CAM, select(("edge", 8)), hover((-3, 1, 0.5)), tap("ctrl"), click((-3, 1, 0.5)), hover((-1, 1, 0.5)),
     click((-1, 1, 0.5)), typed("2x"))
prog("plate", CAM, select(("face", 0), ("edge", 8)), hover(C0), click(C0), tap("ctrl"), hover((4, 4, 0)),
     click((4, 4, 0)), typed("2x"), undo())
# a key that is not Ctrl, the arrow-key axis locks (the viewport's), Shift
prog("plate", CAM, hover(C0), click(C0), key("right"), hover((3, 3, 0)), hover((4, 3.5, 0)), click((4, 3.5, 0)))
prog("plate", CAM, hover(C0), click(C0), key("up"), hover((3, 3, 0)), hover((3.5, 3, 2)), key("up"), hover((4, 4, 0)),
     click((4, 4, 0)))
prog("plate", CAM, hover(C0), click(C0), key("shift"), hover((3, 2.4, 0)), keyup("shift"), hover((4, 2.6, 0)),
     click((4, 2.6, 0)))
prog("plate", CAM, hover(C0), click(C0), key("left"), hover((3, 3, 0)), esc(), esc(), hover((4, 4, 0)))
# a cube: a face moves and drags its neighbours' corners along
prog("cube", CAM, hover((1.5, 1.5, 3)), click((1.5, 1.5, 3)), hover((1.5, 1.5, 3.5)), hover((1.5, 1.5, 4.5)),
     click((1.5, 1.5, 4.5)), undo(), redo())
prog("cube", CAM, hover((3, 1.5, 1.5)), click((3, 1.5, 1.5)), hover((4, 1.5, 1.5)), hover((5, 1.5, 1.5)),
     click((5, 1.5, 1.5)))
prog("cube", CAM, hover((3, 3, 3)), click((3, 3, 3)), hover((3, 3, 4)), typed("1.25"))
prog("cube", CAM, hover((3, 0, 3)), click((3, 0, 3)), hover((3, 0, 3.5)), hover((3, 1, 3.5)), click((3, 1, 3.5)))
prog("cube", CAM, select(("face", 1)), hover((1.5, 1.5, 3)), click((1.5, 1.5, 3)), hover((1.5, 1.5, 4)), click((1.5, 1.5, 4)),
     undo())
prog("cube", CAM, select(("face", 0), ("face", 1)), hover((1.5, 1.5, 3)), click((1.5, 1.5, 3)), tap("ctrl"),
     hover((1.5, 1.5, 6)), click((1.5, 1.5, 6)), typed("2x"))
# the ridge: an edge lifted straight up, in the three cameras
prog("ridge", CAM, hover((2, 2, 2)), click((2, 2, 2)), hover((2, 2, 2.5)), hover((2, 2, 3)), click((2, 2, 3)), undo())
prog("ridge", CAM, hover((2, 2, 2)), click((2, 2, 2)), hover((2, 2, 3)), typed("1.5"))
prog("ridge", CAM_LOW, hover((2, 0, 1)), click((2, 0, 1)), hover((2, 0, 1.6)), hover((2, 0, 2.4)), click((2, 0, 2.4)))
prog("ridge", CAM_ORTHO, hover((2, 2, 2)), click((2, 2, 2)), hover((2, 2, 3)), click((2, 2, 3)))
prog("ridge", CAM, select(("edge", 11)), hover((2, 2, 2)), click((2, 2, 2)), hover((2, 3, 2)), hover((2, 3.5, 2.5)),
     click((2, 3.5, 2.5)))
# cameras
prog("plate", CAM_TOP, hover(C0), click(C0), hover((3, 3, 0)), hover((3.2, 3.1, 0)), click((3.2, 3.1, 0)))
prog("plate", CAM_ORTHO, hover(C0), click(C0), hover((3, 3, 0)), click((3, 3, 0)), undo())
prog("plate", CAM_LOW, hover(C0), click(C0), hover((3, 2, 0)), hover((5, 2, 0)), click((5, 2, 0)))
# welds: a corner onto another corner, an edge onto an edge
prog("tight", CAM, hover((0, 2, 0)), click((0, 2, 0)), hover((0.5, 3, 0)), hover((0, 3, 0)), click((0, 3, 0)))
prog("tight", CAM, hover((2, 5, 0)), click((2, 5, 0)), hover((2, 3, 0)), click((2, 3, 0)), undo())
prog("tight", CAM, hover((0, 5, 0)), click((0, 5, 0)), hover((1, 4, 0)), click((1, 4, 0)))
prog("tight", CAM, hover((4, 2, 0)), click((4, 2, 0)), hover((5, 2, 0)), click((5, 2, 0)))
prog("tight", CAM, hover((1, 4, 0)), click((1, 4, 0)), hover((1, 2.0003, 0)), hover((1, 2, 0)), click((1, 2, 0)))
prog("tight", CAM, hover((6, 0, 0)), click((6, 0, 0)), hover((6, 1, 0)), hover((5, 1, 0)), click((5, 1, 0)))
prog("tight", CAM, select(("edge", 0)), hover((1, 0, 0)), click((1, 0, 0)), hover((1, 0.0004, 0)), typed("0.0004"))
# groups: a classic group, instances (xform), a container
G_CLASSIC = (-4, 1, 0.75)
G_A = (6.5, 0.5, 0.5)
G_B = (9, 2, 0.5)
prog("groups", CAM, hover(G_CLASSIC), click(G_CLASSIC), hover((-3, 1, 0.75)), hover((-2, 2, 0.75)), click((-2, 2, 0.75)),
     undo(), redo())
prog("groups", CAM, hover(G_A), click(G_A), hover((7.5, 0.5, 0.5)), hover((8, 1.5, 0.5)), click((8, 1.5, 0.5)), undo())
prog("groups", CAM, hover(G_A), click(G_A), hover((7.5, 0.5, 0.5)), typed("2"))
prog("groups", CAM, select(("group", [1])), hover(C0), click(C0), hover((3, 3, 0)), click((3, 3, 0)))
prog("groups", CAM, select(("group", [0]), ("group", [2]), ("face", 0)), hover(C0), click(C0), hover((3, 3, 0)),
     click((3, 3, 0)), undo())
prog("groups", CAM, hover(G_CLASSIC), click(G_CLASSIC), tap("ctrl"), hover((-2, 2, 0.75)), click((-2, 2, 0.75)),
     typed("2x"), undo(), redo())
prog("groups", CAM, hover(G_A), click(G_A), tap("ctrl"), hover((8, 0.5, 0.5)), click((8, 0.5, 0.5)), typed("3x"),
     typed("/2"))
prog("groups", CAM, hover(G_A), click(G_A), tap("ctrl"), hover((8, 0.5, 0.5)), click((8, 0.5, 0.5)), esc(), typed("2x"))
prog("groups", CAM, hover(G_A), click(G_A), hover((8, 0.5, 0.5)), esc(), hover((8, 0.5, 0.5)))
prog("groups", CAM, select(("group", [1])), tap("ctrl"), hover(C0), click(C0), hover((3, 3, 0)), click((3, 3, 0)))
prog("groups", CAM, hover(G_B), click(G_B), tap("ctrl"), hover((9, 4, 0.5)), typed("2"), undo())
prog("container", CAM, hover((7.17, 1.86, 1.0)), click((7.17, 1.86, 1.0)), hover((7.17, 3.5, 1.0)),
     click((7.17, 3.5, 1.0)), undo(), redo())
prog("container", CAM, hover((1.5, 1.5, 0)), click((1.5, 1.5, 0)), hover((2.5, 1.5, 0)), click((2.5, 1.5, 0)))
prog("container", CAM, select(("group", [0])), hover((1.5, 1.5, 0)), click((1.5, 1.5, 0)), tap("ctrl"),
     hover((3, 1.5, 0)), click((3, 1.5, 0)), typed("2x"))
# the tool put down mid-move, picked up again, retyped and cancelled
prog("plate", CAM, hover(C0), click(C0), hover((3, 3, 0)), deactivate(), activate(), hover((3, 3, 0)))
prog("plate", CAM, hover(C0), click(C0), tap("ctrl"), hover((3, 3, 0)), deactivate(), activate(), hover(C0))
prog("plate", CAM, hover((20, 20, 0)), click((20, 20, 0)), hover((21, 20, 0)))
prog("plate", CAM, click(px(5, 5)), hover(px(6, 6)), typed("1"), esc())
prog("plate", CAM, hover(C0), dbl(C0), hover((3, 3, 0)), dbl((3, 3, 0)))
prog("plate", CAM, hover(C0), click(C0), hover((3, 3, 0)), camera(dict(CAM, yaw=-20.0)), hover((3.2, 3.1, 0)),
     click((3.2, 3.1, 0)))


# ---- seeded random programs -----------------------------------------------------------------------

def random_program(seed: int, scene: str, cam: dict, n: int = 22):
    rnd = random.Random(seed)
    pts = {
        "plate": [(2, 2, 0), (4, 4, 0), (0, 0, 0), (2, 0, 0), (7.5, 1, 0), (6, 0, 0), (-3, 1, 0.5), (3, 3, 0), (1, 3, 0),
                  (5, 5, 0), (0, 4, 0)],
        "cube": [(1.5, 1.5, 3), (3, 1.5, 1.5), (3, 3, 3), (3, 0, 3), (0, 3, 1.5), (7, 1, 0), (1.5, 0, 1.5), (2, 2, 4)],
        "ridge": [(2, 2, 2), (2, 1, 2), (2, 3, 2), (0, 0, 0), (4, 4, 2), (2, 0, 1), (2, 2, 3.5)],
        "groups": [(-4, 1, 0.75), (6.5, 0.5, 0.5), (9, 2, 0.5), (2, 2, 0), (4, 4, 0), (-2, 2, 0.75), (8, 3, 0.5)],
        "tight": [(0, 2, 0), (2, 5, 0), (4, 2, 0), (1, 1, 0), (3, 1, 0), (6, 0, 0), (5, 1, 0), (2, 2, 0), (1, 4, 0)],
        "container": [(7.17, 1.86, 1.0), (2, 2, 0), (1.5, 1.5, 0), (8, 3, 0), (7.17, 3.5, 1.0)],
    }[scene]
    refs = {
        "plate": [("face", 0), ("face", 1), ("edge", 8)],
        "cube": [("face", 0), ("face", 1), ("face", 5)],
        "ridge": [("edge", 11), ("face", 5), ("face", 6)],
        "groups": [("group", [0]), ("group", [1]), ("face", 0)],
        "tight": [("face", 0), ("face", 1), ("edge", 0)],
        "container": [("group", [0]), ("face", 0)],
    }[scene]
    evs = []
    for _ in range(n):
        r = rnd.random()
        p = rnd.choice(pts)
        jitter = (rnd.choice((0, 0, 0.3, -0.4, 1.0)), rnd.choice((0, 0, 0.5, -0.5)), 0)
        q = (p[0] + jitter[0], p[1] + jitter[1], p[2])
        if r < 0.30:
            evs.append(hover(q))
        elif r < 0.55:
            evs.append(click(q))
        elif r < 0.62:
            evs.append(tap("ctrl"))
        elif r < 0.70:
            evs.append(typed(rnd.choice(["1", "2.5", "-1", "0.3", "3x", "/2", "2;1;0", "1m", "4x", "x3"])))
        elif r < 0.76:
            evs.append(esc())
        elif r < 0.82:
            evs.append(undo())
        elif r < 0.86:
            evs.append(redo())
        elif r < 0.91:
            evs.append(select(rnd.choice(refs)))
        elif r < 0.94:
            evs.append(key(rnd.choice(["right", "left", "up"])))
        else:
            evs.append(hover(q, shift=rnd.random() < 0.5))
    return (scene, cam, evs)


for _seed in range(100, 150):
    _sc = ("plate", "cube", "ridge", "groups", "tight", "container")[_seed % 6]
    PROGRAMS.append(random_program(_seed, _sc, (CAM, CAM_ORTHO, CAM_LOW)[_seed % 3]))


def extra_probe(tool, vp, ids):
    """Move's own state (the rotation grips are not ported: a program that
    brings one under the cursor is refused by the generator)."""
    return {
        "copy": bool(tool._copy),
        "grab": None if tool.grab is None else pbits(tool.grab),
        "n_pos": len(tool._positions),
        "n_verts": len(tool._verts),
        "n_groups": len(tool._groups),
        "vp_preview": bool(getattr(tool, "_vp_preview", False)),
        "preview_delta": pbits(tool._preview_delta),
        "sel_faces": len(tool._sel_faces),
        "sel_edges": len(tool._sel_edges),
        "base_segments": len(tool._base_segments),
        "last": tool._last is not None,
        "hot_grip": tool._hot_grip,
        "grip_rot": tool._grip_rot is not None,
        "moving_faces": len(tool.moving_face_ids()),
        "ctrl_toggled": bool(tool._ctrl_toggled),
        # a held group that an undo took out of the scene: not representable
        "stale": any(id(g) not in ids.gpath for g in tool._groups),
    }


def is_valid(rec) -> bool:
    """Programs that bring a rotation grip under the cursor are not ported."""
    for ev in rec["events"]:
        x = ev["probe"].get("x") or {}
        if x.get("hot_grip") is not None or x.get("grip_rot"):
            return False
    return True
