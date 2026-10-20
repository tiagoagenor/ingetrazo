# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
"""The Paint tool's scenes and programs for ``make_modify_tool_fixtures.py``
(``tools/paint.py``, docs/rust-rewrite-plan.md M3.13).

``PaintTool``'s current paint is CLASS-level state: every program starts from
the state its camera dict names under ``"init"`` (``setup`` applies it to the
class and records it for the Rust test). The tool under test is the real
``PaintTool`` plus an ``on_key_release`` observer that answers ``False``: the
viewport keeps ``_alt_down`` for the key it releases itself, and this makes
the release visible to the replay (it changes no behaviour).

Hovering a face INSIDE a group is kept out of the programs: the viewport is
asked to highlight a face of a group's own mesh, which a ``Sel`` cannot name.
"""
from __future__ import annotations

import math
import random

from _modify_harness import (ALT, CTRL, SHIFT, V, above, at, box, box_faces, click, dbl, deactivate, esc,  # noqa: F401
                             group, hover, key, keyup, press, redo, release, select, tap, typed, undo, xf,
                             Mesh, Scene, activate, camera, pbits, px)
from core.materials import Material
from core.vecmath import QVector3D
from tools.paint import PaintTool

NAME = "paint"


class ObservedPaint(PaintTool):
    def on_key_release(self, viewport, key) -> bool:
        return False


make_tool = ObservedPaint

BASE = dict(target=(4, 1, 0.8), distance=26.0, w=1000, h=600)
CAM = dict(BASE, yaw=-50.0, pitch=30.0)
CAM_ORTHO = dict(BASE, yaw=-50.0, pitch=30.0, perspective=False)
CAM_UNDER = dict(BASE, yaw=-40.0, pitch=-35.0)
CAM_TOP = dict(BASE, yaw=-90.0, pitch=89.0)
CAM_LOW = dict(BASE, yaw=-20.0, pitch=10.0)
GCAM = dict(target=(2, 2, 0.8), distance=26.0, yaw=-50.0, pitch=30.0, w=1000, h=600)
GCAM_UNDER = dict(target=(2, 2, 0.8), distance=26.0, yaw=-30.0, pitch=-30.0, w=1000, h=600)
GCAM_TOP = dict(target=(2, 2, 0.8), distance=26.0, yaw=-90.0, pitch=89.0, w=1000, h=600)

BRICK = {"path": "brick.png", "sw": 0.5, "sh": 0.5}
TILE_UVW = {"path": "tile.png", "sw": 1.0, "sh": 1.0, "uvw": [0.5, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0]}


def scn_plates() -> Scene:
    s = Scene()
    s.materials["Brick"] = Material("Brick", (0.7, 0.3, 0.2), dict(BRICK), None)
    s.materials["Glass"] = Material("Glass", (0.5, 0.5, 0.9), None, 0.5)
    m = s.mesh
    m.add_face([V(0, 0, 0), V(4, 0, 0), V(4, 4, 0), V(0, 4, 0)])                       # 0 plain
    b = m.add_face([V(6, 0, 0), V(9, 0, 0), V(9, 2, 0), V(6, 2, 0)])                   # 1 coloured
    b.attrs["color"] = [0.1, 0.7, 0.2]
    c = m.add_face([V(0, -4, 0), V(4, -4, 0), V(4, -4, 2), V(0, -4, 2)])               # 2 wall, textured, tagged
    c.attrs["texture"] = dict(BRICK)
    c.attrs["mat"] = "Brick"
    m.add_face([V(-7, 0, 0), V(-3, 0, 0), V(-3, 4, 0), V(-7, 4, 0)],                  # 3 with a hole
               [[V(-6, 1, 0), V(-4, 1, 0), V(-4, 3, 0), V(-6, 3, 0)]])
    e = m.add_face([V(10, 0, 0), V(13, 0, 0), V(13, 3, 0), V(10, 3, 0)])               # 4 positioned texture
    e.attrs["texture"] = dict(TILE_UVW)
    f = m.add_face([V(13, 0, 0), V(16, 0, 0), V(16, 3, 0), V(13, 3, 0)])               # 5 same plane
    f.attrs["texture"] = dict(TILE_UVW)
    g = m.add_face([V(6, 4, 0), V(9, 4, 0), V(9, 6, 0), V(6, 6, 0)])                   # 6 translucent
    g.attrs["color"] = [0.5, 0.5, 0.9]
    g.attrs["opacity"] = 0.5
    g.attrs["mat"] = "Glass"
    h = m.add_face([V(0, 6, 0), V(4, 6, 0), V(4, 8, 0), V(0, 8, 0)])                   # 7 back painted
    h.attrs["back"] = {"color": [0.9, 0.1, 0.1]}
    i = m.add_face([V(-7, 6, 0), V(-3, 6, 0), V(-3, 8, 0), V(-7, 8, 0)])               # 8 two-sided
    i.attrs["back"] = True
    i.attrs["color"] = [0.3, 0.3, 0.3]
    return s


def scn_curved() -> Scene:
    s = Scene()
    m = s.mesh
    box_faces(m, 0, 0, 0, 3, 3, 3)
    m.add_face([V(-5, 0, 0), V(-2, 0, 0), V(-2, 3, 0), V(-5, 3, 0)])
    n, r, cx, cy, hgt = 12, 1.5, 8.0, 1.5, 3.0
    ring = [(cx + r * math.cos(2 * math.pi * k / n), cy + r * math.sin(2 * math.pi * k / n)) for k in range(n)]
    for k in range(n):
        (x0, y0), (x1, y1) = ring[k], ring[(k + 1) % n]
        m.add_face([V(x0, y0, 0), V(x1, y1, 0), V(x1, y1, hgt), V(x0, y0, hgt)])
    m.add_face([V(x, y, hgt) for x, y in ring])
    for k in range(n):
        x0, y0 = ring[k]
        a, b = V(x0, y0, 0), V(x0, y0, hgt)
        e = m.find_edge(m.vertex_at(a), m.vertex_at(b))
        e.soft = True
    return s


def scn_groups() -> Scene:
    s = Scene()
    proto = box(0, 0, 0, 1, 1, 1, (0.2, 0.6, 0.9))
    own = box(0, 0, 0, 1, 1, 1)
    own.faces[1].attrs["color"] = [0.9, 0.2, 0.2]
    g0 = group(box(-5, 0, 0, 2, 2, 1.5), "classic")
    g1 = group(proto, "inst_a", xf(6, 0, 0))
    g2 = group(proto, "inst_b", xf(9, 2, 0, 30))
    k1 = group(proto, "kid1", xf(1, 0, 0, 20))
    k2 = group(proto, "kid2", xf(0, 2, 0))
    ctr = group(Mesh(), "container", xf(6, 6, 0, 10), children=[k1, k2])
    g4 = group(box(-5, 5, 0, 1, 1, 1), "painted")
    g4.material = {"color": [0.2, 0.2, 0.9], "mat": "Blue"}
    g5 = group(own, "own_face", xf(0, 6, 0))
    s.materials["Blue"] = Material("Blue", (0.2, 0.2, 0.9), None, None)
    s.groups.extend([g0, g1, g2, ctr, g4, g5])
    s.mesh.add_face([V(0, 0, 0), V(4, 0, 0), V(4, 4, 0), V(0, 4, 0)])
    s.mesh.add_face([V(0, -4, 0), V(4, -4, 0), V(4, -4, 2), V(0, -4, 2)])
    return s


SCENES = {"plates": scn_plates, "curved": scn_curved, "groups": scn_groups}

# ---- initial paint states ---------------------------------------------------------------------

BRICK_MAT = {"name": "Brick", "color": [0.7, 0.3, 0.2], "texture": dict(BRICK)}
GLASS_MAT = {"name": "Glass", "color": [0.5, 0.5, 0.9], "opacity": 0.5}
NEWMAT = {"name": "Fresh", "color": [0.4, 0.9, 0.4]}
PLANE_Z = {"n": [0.0, 0.0, 1.0], "d": 0.0}

I_DEFAULT = {}
I_COL = {"color": [0.1, 0.6, 0.9]}
I_TEX = {"texture": {"path": "wood.png", "sw": 0.6, "sh": 0.4}}
I_TEX_ROT = {"texture": {"path": "wood.png", "sw": 0.6, "sh": 0.4, "rot": 30.0}}
I_UVW = {"texture": dict(TILE_UVW), "plane": PLANE_Z}
I_UVW_NOPLANE = {"texture": dict(TILE_UVW)}
I_UVW_WALL = {"texture": {"path": "tile.png", "sw": 1.0, "sh": 1.0,
                          "uvw": [0.0, 0.0, 0.8, 0.0, 0.5, 0.0, 0.0, 0.0]},
              "plane": {"n": [0.0, -1.0, 0.0], "d": 4.0}}
I_MAT = {"material": BRICK_MAT, "color": [0.7, 0.3, 0.2]}
I_MAT_TEX = {"material": BRICK_MAT, "texture": dict(BRICK)}
I_NEW = {"material": NEWMAT, "color": [0.4, 0.9, 0.4]}
I_GLASS = {"material": GLASS_MAT, "color": [0.5, 0.5, 0.9], "opacity": 0.5}
I_OPA = {"opacity": 0.4}
I_NODEF = {"default": True}
I_ARMED = {"armed": True, "color": [0.3, 0.3, 0.3]}
I_RECENT = {"color": [0.2, 0.3, 0.4], "recent": [{"color": [0.9, 0.9, 0.1]}, {"color": [0.2, 0.3, 0.4]},
                                                   {"texture": dict(BRICK), "mat": "Brick"}]}
INITS = [I_DEFAULT, I_COL, I_TEX, I_TEX_ROT, I_UVW, I_UVW_NOPLANE, I_UVW_WALL, I_MAT, I_MAT_TEX, I_NEW, I_GLASS,
         I_OPA, I_NODEF, I_ARMED, I_RECENT]


def cam(c, init):
    return dict(c, init=init)


PROGRAMS: list = []


def prog(scene, c, init, *events):
    PROGRAMS.append((scene, cam(c, init), list(events)))


# face points
A, B, C, D, E, F, G, H, I2 = ((2, 2, 0), (7.5, 1, 0), (2, -4, 1), (-6.5, 0.5, 0), (11.5, 1.5, 0), (14.5, 1.5, 0),
                              (7.5, 5, 0), (2, 7, 0), (-5, 7, 0))
SEL = lambda *ix: select(*[("face", i) for i in ix])  # noqa: E731

# a plain colour click and its undo/redo
prog("plates", CAM, I_DEFAULT, hover(A), click(A), undo(), redo())
prog("plates", CAM, I_COL, click(A), click(B), click(A), undo(), undo())
prog("plates", CAM, I_COL, hover(A), hover(B), hover(px(500, 20)), hover(C), hover(D), click(px(500, 20)), click(D))
prog("plates", CAM, I_COL, click(G), click(H), click(I2), click(E))
# textures
prog("plates", CAM, I_TEX, click(A), click(B), undo(), redo())
prog("plates", CAM, I_TEX_ROT, click(C), click(A))
prog("plates", CAM, I_UVW, click(E), click(F), click(A), click(C))
prog("plates", CAM, I_UVW_NOPLANE, click(A), click(C), click(F))
prog("plates", CAM, I_UVW_WALL, click(C), click(A), click(E))
# a selection is painted as a whole when the face is part of it
prog("plates", CAM, I_COL, SEL(0, 1), click(A), undo(), redo())
prog("plates", CAM, I_TEX, SEL(0, 1, 6), click(G), click(E))
prog("plates", CAM, I_UVW, SEL(4, 5, 0), click(F), click(C))
prog("plates", CAM, I_MAT, SEL(0, 3), click(D), click(B))
# material identity and opacity
prog("plates", CAM, I_MAT, click(A), click(B), undo(), undo(), redo())
prog("plates", CAM, I_MAT_TEX, click(A), click(H), undo())
prog("plates", CAM, I_NEW, click(A), undo(), click(B), redo())
prog("plates", CAM, I_GLASS, click(A), click(C), click(E))
prog("plates", CAM, I_OPA, click(G), click(A))
# the default material removes the paint
prog("plates", CAM, I_NODEF, click(B), click(C), click(G), click(A), undo(), redo())
prog("plates", CAM, I_NODEF, SEL(1, 2, 6), click(B), undo())
# the back side: seen from underneath
prog("plates", CAM_UNDER, I_COL, click(A), click(B), click(D), undo())
prog("plates", CAM_UNDER, I_TEX, click(A), click(H))
prog("plates", CAM_UNDER, I_UVW, click(E), click(A), click(C))
prog("plates", CAM_UNDER, I_UVW_NOPLANE, click(F), click(A))
prog("plates", CAM_UNDER, I_GLASS, click(A), click(G), click(H))
prog("plates", CAM_UNDER, I_MAT, SEL(0, 1), click(A), click(I2), undo(), redo())
prog("plates", CAM_UNDER, I_NODEF, click(H), click(I2), click(A), click(B))
prog("plates", CAM_UNDER, I_OPA, click(G), click(H))
prog("plates", CAM_LOW, I_COL, click(C), click(A), click(B))
prog("plates", CAM_TOP, I_COL, click(A), click(E), hover(A), click(C))
prog("plates", CAM_ORTHO, I_TEX, click(A), click(C), click(E))
# the eyedropper: Alt held at the click, then paint what was sampled
prog("plates", CAM, I_DEFAULT, click(B, alt=True), click(A), click(H))
prog("plates", CAM, I_DEFAULT, click(C, alt=True), click(A), click(B), undo())
prog("plates", CAM, I_DEFAULT, click(E, alt=True), click(A), click(C), click(F))
prog("plates", CAM, I_DEFAULT, click(G, alt=True), click(A), click(H))
prog("plates", CAM, I_COL, click(A, alt=True), click(B), click(A))
prog("plates", CAM, I_COL, click(I2, alt=True), click(A))
prog("plates", CAM, I_COL, click(D, alt=True), click(B), click(px(500, 20), alt=True))
prog("plates", CAM_UNDER, I_DEFAULT, click(H, alt=True), click(A), click(B))
prog("plates", CAM_UNDER, I_DEFAULT, click(I2, alt=True), click(A))
prog("plates", CAM_UNDER, I_DEFAULT, click(B, alt=True), click(H), click(C, alt=True), click(A))
prog("plates", CAM_UNDER, I_DEFAULT, click(A, alt=True), click(B))
# Alt as a held key, its release, the armed toolbar eyedropper
prog("plates", CAM, I_COL, key("alt", alt=True), hover(B, alt=True), click(B, alt=True), keyup("alt"), hover(A), click(A))
prog("plates", CAM, I_COL, key("alt", alt=True), click(B), click(C), keyup("alt"), click(A))
prog("plates", CAM, I_COL, tap("alt"), click(B), click(A))
prog("plates", CAM, I_ARMED, click(B), click(A), click(A))
prog("plates", CAM, I_ARMED, hover(A), click(px(500, 20)), click(C), click(H))
prog("plates", CAM, I_ARMED, key("alt", alt=True), keyup("alt"), click(E), click(F), click(B))
# the recent row
prog("plates", CAM, I_RECENT, click(A), click(B, alt=True), click(A), click(C, alt=True), click(B), click(G, alt=True), click(A))
prog("plates", CAM, I_COL, *[ev for p in (B, C, E, G, B, C, E, G, B) for ev in (click(p, alt=True), click(A))])
# hover and the tool put down
prog("plates", CAM, I_COL, hover(A), deactivate(), activate(), hover(B), deactivate())
prog("plates", CAM, I_COL, hover(A), esc(), click(A), esc())
prog("plates", CAM, I_COL, SEL(0), esc(), click(A), select(("face", 1)), click(A))
# a curved surface is painted whole
CUBE_TOP = (1.5, 1.5, 3)
CYL_SIDE = (8 + 1.5 * math.cos(math.radians(15)), 1.5 + 1.5 * math.sin(math.radians(15)), 1.5)
CYL_SIDE2 = (8 + 1.5 * math.cos(math.radians(255)), 1.5 + 1.5 * math.sin(math.radians(255)), 1.5)
CYL_LID = (8, 1.5, 3)
CYL_SIDE3 = (8 + 1.5 * math.cos(math.radians(-75)), 1.5 + 1.5 * math.sin(math.radians(-75)), 1.5)
prog("curved", CAM, I_COL, click(CYL_SIDE), click(CUBE_TOP), click(CYL_LID), undo(), redo())
prog("curved", CAM, I_TEX, click(CYL_SIDE3), click(CUBE_TOP), undo())
prog("curved", CAM, I_TEX_ROT, click(CYL_SIDE3), click(CYL_LID))
prog("curved", CAM, I_UVW, click(CYL_SIDE3), click(CUBE_TOP), click(CYL_LID))
prog("curved", CAM, I_UVW_NOPLANE, click(CYL_SIDE3), click(CYL_LID))
prog("curved", CAM, I_MAT_TEX, click(CYL_SIDE3), undo(), redo())
prog("curved", CAM, I_GLASS, click(CYL_SIDE3), click(CUBE_TOP))
prog("curved", CAM, I_NODEF, click(CYL_SIDE3), click(CUBE_TOP))
prog("curved", CAM, I_TEX, SEL(15, 17), click(CYL_SIDE3), click(CYL_SIDE))
prog("curved", CAM, I_COL, SEL(15, 17), click(CYL_SIDE3), click(CYL_SIDE))
prog("curved", CAM_UNDER, I_COL, click(CYL_SIDE2), click(CUBE_TOP))
prog("curved", CAM_TOP, I_TEX, click(CYL_LID), click(CUBE_TOP))
prog("curved", CAM, I_DEFAULT, click(CYL_SIDE3, alt=True), click(CUBE_TOP), click(CYL_SIDE))
prog("curved", CAM_ORTHO, I_TEX, hover(CYL_SIDE3), click(CYL_SIDE3), hover(CUBE_TOP))
# groups: the container is painted as a whole
GA = (-4, 1, 0.75)
GB = (6.5, 0.5, 0.5)
GC = (9, 2, 0.5)
GK = (6.5, 7, 0.5)
GP = (-4.5, 5.5, 0.5)
GO = (0.5, 6.5, 0.5)
GOWN = (0.5, 6.5, 1)
GL = (2, 2, 0)
GW = (2, -4, 1)
prog("groups", GCAM, I_COL, click(GA), click(GB), click(GC), undo(), redo())
prog("groups", GCAM, I_COL, click(GK), click(GP), click(GL), click(GW))
prog("groups", GCAM, I_MAT, click(GA), click(GB), undo(), redo())
prog("groups", GCAM, I_TEX, click(GA), click(GC))
prog("groups", GCAM, I_UVW, click(GA), click(GB), click(GL))
prog("groups", GCAM, I_NODEF, click(GP), click(GA), click(GK))
prog("groups", GCAM, I_GLASS, click(GB), click(GK))
prog("groups", GCAM, I_COL, SEL(0), click(GA), click(GL))
prog("groups", GCAM, I_DEFAULT, click(GP, alt=True), click(GA), click(GB))
prog("groups", GCAM, I_DEFAULT, click(GOWN, alt=True), click(GA), click(GL))
prog("groups", GCAM, I_DEFAULT, click(GA, alt=True), click(GL), click(GB))
prog("groups", GCAM, I_DEFAULT, click(GK, alt=True), click(GL), click(GA))
prog("groups", GCAM, I_DEFAULT, click(GC, alt=True), click(GL), click(GA))
prog("groups", GCAM_UNDER, I_COL, click(GA), click(GB), click(GL))
prog("groups", GCAM_UNDER, I_DEFAULT, click(GB, alt=True), click(GL), click(GP, alt=True), click(GL))
prog("groups", GCAM_TOP, I_COL, click(GA), click(GB), click(GC), click(GK))
prog("groups", GCAM, I_ARMED, click(GP), click(GL), click(GB))
prog("groups", GCAM, I_RECENT, click(GA), click(GB), click(GL, alt=True), click(GK))
prog("groups", GCAM, I_COL, hover(GL), click(GL), hover(px(900, 580)), click(px(900, 580)))


# ---- seeded random programs --------------------------------------------------------------------------

def random_program(seed: int, scene: str, c: dict, n: int = 20):
    rnd = random.Random(seed)
    pts = {
        "plates": [A, B, C, D, E, F, G, H, I2, (1, 1, 0), (3, 3, 0), (8, 1.5, 0), (0.5, 7.5, 0), (px(30, 30))],
        "curved": [CUBE_TOP, CYL_SIDE, CYL_SIDE2, CYL_SIDE3, CYL_LID, (-3.5, 1.5, 0), (3, 1.5, 1.5)],
        "groups": [GA, GB, GC, GK, GP, GO, GOWN, GL, GW, (3, 3, 0)],
    }[scene]
    loose = {"plates": pts, "curved": pts, "groups": [GL, GW, (3, 3, 0), (1, 1, 0)]}[scene]
    nface = {"plates": 9, "curved": 20, "groups": 2}[scene]
    init = rnd.choice(INITS)
    evs = []
    for _ in range(n):
        r = rnd.random()
        p = rnd.choice(pts)
        if r < 0.22:
            evs.append(hover(rnd.choice(loose)))
        elif r < 0.55:
            evs.append(click(p))
        elif r < 0.68:
            evs.append(click(p, alt=True))
        elif r < 0.73:
            evs.append(key("alt", alt=True))
        elif r < 0.78:
            evs.append(keyup("alt"))
        elif r < 0.84:
            evs.append(undo())
        elif r < 0.88:
            evs.append(redo())
        elif r < 0.93:
            evs.append(select(*[("face", rnd.randrange(nface)) for _ in range(rnd.choice((1, 2, 3)))]))
        elif r < 0.96:
            evs.append(esc())
        else:
            evs.append(hover(rnd.choice(loose), alt=True))
    return (scene, cam(c, init), evs)


for _seed in range(500, 545):
    _sc = ("plates", "curved", "groups")[_seed % 3]
    _cams = (CAM, CAM_UNDER, CAM_ORTHO, CAM_TOP, CAM_LOW) if _sc != "groups" else (GCAM, GCAM_UNDER, GCAM_TOP)
    PROGRAMS.append(random_program(_seed, _sc, _cams[(_seed // 3) % len(_cams)]))


# ---- per-program state, probe -----------------------------------------------------------------------------

def _material(d):
    return None if d is None else Material(d["name"], tuple(d["color"]) if d.get("color") else None,
                                           dict(d["texture"]) if d.get("texture") else None, d.get("opacity"))


def setup(s, n):
    """Put the class-level paint into the state this program names."""
    init = s.cam_spec.get("init", {})
    PaintTool.current_color = tuple(init.get("color", (0.80, 0.45, 0.30)))
    PaintTool.current_texture = None if init.get("texture") is None else dict(init["texture"])
    PaintTool.current_material = _material(init.get("material"))
    PaintTool.current_opacity = init.get("opacity")
    PaintTool.sample_armed = bool(init.get("armed", False))
    pl = init.get("plane")
    PaintTool.current_texture_plane = None if pl is None else (QVector3D(*pl["n"]), pl["d"])
    PaintTool.current_is_default = bool(init.get("default", False))
    if "recent" in ObservedPaint.__dict__:      # ``cls.recent = ...`` assigns on the subclass
        del ObservedPaint.recent
    PaintTool.recent = [dict(e) for e in init.get("recent", [])]
    return {"init": {
        "color": list(PaintTool.current_color),
        "texture": PaintTool.current_texture,
        "material": None if PaintTool.current_material is None else PaintTool.current_material.to_dict(),
        "opacity": PaintTool.current_opacity,
        "armed": PaintTool.sample_armed,
        "plane": pl,
        "default": PaintTool.current_is_default,
        "recent": PaintTool.recent,
    }}


def extra_probe(tool, vp, ids):
    pl = PaintTool.current_texture_plane
    return {
        "color": list(PaintTool.current_color),
        "texture": PaintTool.current_texture,
        "material": None if PaintTool.current_material is None else PaintTool.current_material.name,
        "opacity": PaintTool.current_opacity,
        "armed": PaintTool.sample_armed,
        "plane": None if pl is None else [pbits(pl[0]), pl[1]],
        "default": PaintTool.current_is_default,
        "recent": tool.recent,
        "alt_down": bool(vp._alt_down),
        "mats": list(vp.scene.materials.keys()),
    }


def is_valid(rec) -> bool:
    """A hover over a group's face cannot be named (see the module docs); and
    a mouse move the tool is never told about (the cursor ray missed the
    work plane) still re-reads Alt in the viewport, which the replay cannot see."""
    alt = False
    for ev in rec["events"]:
        for g in ev["gfx"]:
            if g[0] == "hover" and g[1] is not None and g[1][0] == "unknown":
                return False
        now = ev["probe"]["x"]["alt_down"]
        if ev["e"] == "hover" and not ev["calls"] and now != alt:
            return False
        alt = now
    return True
