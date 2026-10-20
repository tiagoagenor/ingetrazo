# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
"""Fixtures that hold ``ingetrazo-snap``' picking to the Python app
(docs/rust-rewrite-plan.md, M3.12): what the cursor picks, given a scene, a
camera and a pixel; and what a rubber-band box takes.

Each scene is saved as a bare-JSON .igz (``<scene>.igz.json``) and reopened in
a fresh viewport, so Python and Rust start from the very same bits. For every
(scene, open group, camera) "view" the REAL ``Viewport`` (offscreen Qt, no GL
context needed) answers, for many pixels, the whole battery: ``pick_face``,
``pick_face_any``, ``pick_face_placement``, ``pick_group``, ``pick_edge``,
``pick_visible_edge``, ``pick_vertex``, ``pick_dimension``,
``pick_text_label`` (whole and ``rect_only``), ``pick_guide``,
``pick_section_plane``, ``pick_image_plane``, ``pick_locked_image_plane``,
``pick_axis``, ``pick_geopath`` and the Select tool's own priority
(``SelectTool._pick``); and for many rectangles ``SelectTool.on_box_select``
in window and crossing mode.

Entities are recorded by stable identity, never by value: a loose face or edge
by its index in the active mesh's list (creation order), a group by its path in
the scene tree (indexes through ``children``), a face of a group by the path of
the placement it was hit through plus its index in that placement's mesh, an
annotation by its index in its scene list, a vertex by its f32 bits.

``cases.json`` holds the views; each carries the camera (f64 bits), the
matrices Python built from it (f32 bits), the open group, the text metrics
Qt measured on this machine, and the answers.

Python's own order is not repeatable in one place, pinned here: the selection
is a ``set`` and ``Scene.select`` only stores it, so the box select is
recorded from the ordered list ``on_box_select`` hands to ``scene.select``.

Committed; rerun only to widen (PYTHONPATH at the repository root):

    python scripts/golden/make_pick_fixtures.py

Deleted with the Python app at M10.
"""
from __future__ import annotations

import json
import math
import os
import random
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.pop("INGETRAZO_RUST", None)

from PySide6.QtGui import QFont, QFontMetrics, QMatrix4x4, QVector3D as V  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

_APP = QApplication.instance() or QApplication([])

from core.dimension import Dimension  # noqa: E402
from core.group import Group  # noqa: E402
from core.guide import Guide  # noqa: E402
from core.image_plane import ImagePlane  # noqa: E402
from core.layers import Layer  # noqa: E402
from core.mesh import Mesh  # noqa: E402
from core.scene import Scene  # noqa: E402
from core.section import SectionPlane  # noqa: E402
from core.textlabel import TextLabel  # noqa: E402
from formats import igz  # noqa: E402
from georef.geopath import GeoPath  # noqa: E402
from tools.select import SelectTool  # noqa: E402
from views.viewport import Viewport, _active_cut  # noqa: E402

OUT = ROOT / "rust" / "crates" / "ingetrazo-snap" / "tests" / "data" / "pick"

W, H = 1000, 600


def h32(x: float) -> str:
    return struct.pack(">f", x).hex()


def h64(x: float) -> str:
    return struct.pack(">d", x).hex()


def vbits(p) -> list:
    return [h32(p.x()), h32(p.y()), h32(p.z())]


def mbits(m: QMatrix4x4) -> list:
    return [h32(x) for x in m.data()]


# ---- scenes ------------------------------------------------------------------------

def box_faces(m: Mesh, x, y, z, sx, sy, sz, color=None):
    x1, y1, z1 = x + sx, y + sy, z + sz
    sides = [
        [V(x, y, z), V(x, y1, z), V(x1, y1, z), V(x1, y, z)],
        [V(x, y, z1), V(x1, y, z1), V(x1, y1, z1), V(x, y1, z1)],
        [V(x, y, z), V(x1, y, z), V(x1, y, z1), V(x, y, z1)],
        [V(x, y1, z), V(x, y1, z1), V(x1, y1, z1), V(x1, y1, z)],
        [V(x, y, z), V(x, y, z1), V(x, y1, z1), V(x, y1, z)],
        [V(x1, y, z), V(x1, y1, z), V(x1, y1, z1), V(x1, y, z1)],
    ]
    out = []
    for pts in sides:
        f = m.add_face(pts)
        if color:
            f.attrs["color"] = list(color)
        out.append(f)
    return out


def box(x, y, z, sx, sy, sz, color=None) -> Mesh:
    m = Mesh()
    box_faces(m, x, y, z, sx, sy, sz, color)
    return m


def xf(dx=0.0, dy=0.0, dz=0.0, angle=0.0, axis=(0, 0, 1), scale=None):
    m = QMatrix4x4()
    m.translate(dx, dy, dz)
    if angle:
        m.rotate(angle, *axis)
    if scale:
        m.scale(*scale)
    return m


def group(mesh, name, m=None, children=None, layer=None, hidden=False,
          locked=False) -> Group:
    g = Group(mesh, name=name)
    if m is not None:
        g.xform = m
    if children:
        g.adopt(children)
    g.layer = layer
    g.hidden = hidden
    g.locked = locked
    return g


def slab_with_hole(m: Mesh, x, y, z, s=3.0, hole=1.0):
    h0, h1 = (s - hole) / 2, (s + hole) / 2
    return m.add_face([V(x, y, z), V(x + s, y, z), V(x + s, y + s, z), V(x, y + s, z)],
                      [[V(x + h0, y + h0, z), V(x + h0, y + h1, z),
                        V(x + h1, y + h1, z), V(x + h1, y + h0, z)]])


def scene_loose() -> Scene:
    s = Scene()
    s.layers.append(Layer("Hid", visible=False))
    s.layers.append(Layer("Lock", locked=True))
    m = s.mesh
    box_faces(m, 0, 0, 0, 2, 2, 2, (0.9, 0.2, 0.2))
    box_faces(m, 3.5, 0.5, 0, 1.5, 1.5, 1.5)
    # a floor coplanar with the boxes' bottoms, and a patch drawn on it
    m.add_face([V(-4, -4, 0), V(9, -4, 0), V(9, 5, 0), V(-4, 5, 0)])
    m.add_face([V(5.5, 2.5, 0), V(7.5, 2.5, 0), V(7.5, 3.8, 0), V(5.5, 3.8, 0)])
    # L shaped (concave) wall
    m.add_face([V(-3, 4, 0), V(-3, 4, 3), V(0, 4, 3), V(0, 4, 1.5),
                V(-1.5, 4, 1.5), V(-1.5, 4, 0)])
    slab_with_hole(m, -8, -2, 0.5)
    # a prism with soft side edges
    ring = [V(math.cos(i * math.pi / 3) * 0.8 + 6.5, math.sin(i * math.pi / 3) * 0.8 - 2.5, 0)
            for i in range(6)]
    top = [V(p.x(), p.y(), 1.2) for p in ring]
    for i in range(6):
        j = (i + 1) % 6
        m.add_face([ring[i], ring[j], top[j], top[i]])
    m.add_face(top)
    for e in m.edges:
        a, b = e.a, e.b
        if abs(a.x() - b.x()) < 1e-6 and abs(a.y() - b.y()) < 1e-6 and a.x() > 5.5 \
                and a.y() < -1.5:
            e.soft = True
    # loose lines
    m.add_edge(V(5, 5, 0), V(7, 6, 2))
    m.add_edge(V(-5, -3, 1), V(-6.5, -3.5, 2.5))
    hid = m.add_edge(V(-5, 3, 0.2), V(-6, 3.5, 0.2))
    hid.hidden = True
    lay = m.add_edge(V(2.5, -3, 0.1), V(4.5, -3.2, 0.1))
    lay.layer = "Hid"
    # hidden face, hidden-layer face, locked-layer face
    f = m.add_face([V(0, -3, 2.5), V(1.2, -3, 2.5), V(1.2, -3, 3.7), V(0, -3, 3.7)])
    f.attrs["hidden"] = True
    f = m.add_face([V(1.5, -3, 2.5), V(2.7, -3, 2.5), V(2.7, -3, 3.7), V(1.5, -3, 3.7)])
    f.attrs["layer"] = "Hid"
    f = m.add_face([V(3.0, -3, 2.5), V(4.2, -3, 2.5), V(4.2, -3, 3.7), V(3.0, -3, 3.7)])
    f.attrs["layer"] = "Lock"
    # two nearly coplanar faces (a 0.5 mm step) and two exactly coincident
    m.add_face([V(-7, 6, 0), V(-4, 6, 0), V(-4, 8, 0), V(-7, 8, 0)])
    m.add_face([V(-6.5, 6.5, 0.0005), V(-4.5, 6.5, 0.0005), V(-4.5, 7.5, 0.0005),
                V(-6.5, 7.5, 0.0005)])
    return s


def proto_pair():
    p = box(0, 0, 0, 1, 1, 1, (0.2, 0.6, 0.9))
    q = Mesh()
    slab_with_hole(q, 0, 0, 0, s=1.0, hole=0.4)
    q.add_edge(V(0, 0, 0.5), V(1, 1, 0.6))
    return p, q


def scene_groups() -> Scene:
    s = Scene()
    s.layers.append(Layer("Hid", visible=False))
    s.layers.append(Layer("Lock", locked=True))
    s.layers.append(Layer("Vis"))
    p, q = proto_pair()
    g0 = group(box(-2, -2, 0, 2, 2, 1.5, (0.9, 0.7, 0.2)), "classic")
    g1 = group(p, "inst", xf(4, 0, 0))
    g2 = group(p, "inst_rot", xf(6, 2, 0, 30), )
    g3 = group(p, "inst_scaled", xf(8, 0, 0, 0, scale=(1.5, 1.0, 2.0)))
    g4 = group(p, "inst_mirror", xf(-4, 0, 0, 0, scale=(-1.0, 1.0, 1.0)))
    # a container with nested placements
    k1 = group(p, "kid1", xf(1, 0, 0, 20))
    k2 = group(q, "kid2", xf(0, 2, 0, 0, scale=(1.0, 1.0, 1.0)))
    gk = group(p, "grandkid", xf(0.2, 0.2, 1.1, 45))
    k1.adopt([gk])
    ctr = group(Mesh(), "container", xf(0, 8, 0, 15), children=[k1, k2])
    g5 = group(box(10, 4, 0, 1, 1, 1), "hidden", hidden=True)
    g6 = group(box(10, 6, 0, 1, 1, 1), "locked", locked=True)
    g7 = group(box(10, 8, 0, 1, 1, 1), "hidlayer", layer="Hid")
    g8 = group(box(12, 4, 0, 1, 1, 1), "locklayer", layer="Lock")
    lm = Mesh()
    lm.add_edge(V(-8, 0, 0), V(-6, 0, 2))
    lm.add_edge(V(-6, 0, 2), V(-6, 2, 2))
    g9 = group(lm, "lines")
    sm = box(12, 8, 0, 1.5, 1.5, 1.5)
    sm.faces[0].attrs["hidden"] = True
    sm.add_edge(V(12, 8, 1.5), V(13.5, 9.5, 1.5)).soft = True
    g10 = group(sm, "partial")
    g11 = group(p, "twin_a", xf(4, 4, 0))
    g12 = group(p, "twin_b", xf(4, 4, 0))
    s.groups.extend([g0, g1, g2, g3, g4, ctr, g5, g6, g7, g8, g9, g10, g11, g12])
    # containers whose state their children inherit: a locked one, a hidden
    # one, one on a hidden tag and one on a locked tag (the children carry
    # no tag of their own)
    for k, (name, kw) in enumerate((("lockedctr", {"locked": True}),
                                    ("hiddenctr", {"hidden": True}),
                                    ("hidtagctr", {"layer": "Hid"}),
                                    ("locktagctr", {"layer": "Lock"}))):
        kids = [group(p, f"{name}_a", xf(0.0, 0.0, 0.0, 10)),
                group(q, f"{name}_b", xf(1.5, 0.5, 0.0, 0)),
                # a tag of its own, shown: the container's tag still wins
                group(p, f"{name}_c", xf(0.0, 1.5, 0.0, 0), layer="Vis")]
        s.groups.append(group(Mesh(), name, xf(-6 + 3.5 * k, -7, 0, 0), children=kids, **kw))
    # loose geometry among them: a box standing before a component, a floor
    box_faces(s.mesh, 5, -3, 0, 1, 1, 2)
    s.mesh.add_face([V(-14, -14, -0.5), V(16, -14, -0.5), V(16, 14, -0.5), V(-14, 14, -0.5)])
    s.mesh.add_face([V(3, 3.5, 0), V(5.5, 3.5, 0), V(5.5, 6, 0), V(3, 6, 0)])
    s.mesh.add_edge(V(-3, 6, 0), V(0, 6, 3))
    return s


def scene_annot() -> Scene:
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
    s.section_planes.append(SectionPlane(V(1.5, 1, 1), V(0, 1, 0), name="A", symbol="A"))
    s.section_planes.append(SectionPlane(V(5, 1, 1), V(1, 0, 0), name="B", symbol="B"))
    p3 = SectionPlane(V(-1, 1, 1), V(0, 0, 1), name="C", symbol="C", hidden=True)
    s.section_planes.append(p3)
    s.image_planes.append(ImagePlane("a.png", V(-4, -4, 0.01), V(4, 0, 0), V(0, 3, 0),
                                     0.75, "img"))
    s.image_planes.append(ImagePlane("b.png", V(-6, 1, 0.5), V(0, 0, 3), V(2, 0, 0), 1.5,
                                     "wall", locked=True))
    s.image_planes.append(ImagePlane("c.png", V(2, 4, 0.02), V(3, 0, 0), V(0, 2, 0), 0.7,
                                     "hidden", layer="Hid"))
    s.image_planes.append(ImagePlane("d.png", V(7, 4, 0.02), V(2, 0, 0), V(0, 2, 0), 1.0,
                                     "lockl", layer="Lock"))
    s.geo_paths.append(GeoPath([V(-5, -3, 0), V(-2, -3.5, 0), V(0, -4, 0)], False, "road"))
    s.geo_paths.append(GeoPath([V(8, -2, 0), V(10, -2, 0), V(10, 1, 0), V(8, 1, 0)], True, "lot"))
    g = group(box(8, 4, 0, 1, 1, 1), "g")
    s.groups.append(g)
    return s


def scene_cut() -> Scene:
    s = Scene()
    box_faces(s.mesh, 0, 0, 0, 3, 3, 3)
    box_faces(s.mesh, 4, 0, 0, 2, 2, 1)
    s.mesh.add_edge(V(-2, -1, 0), V(7, 4, 3))
    p, _q = proto_pair()
    s.groups.append(group(p, "cut_inst", xf(1, 1.5, 0.5, 20)))
    s.groups.append(group(box(1, 5, 0, 2, 2, 2), "cut_classic"))
    s.section_planes.append(SectionPlane(V(1.5, 1.5, 1.5), V(1, 0, 0), name="A", symbol="A",
                                         active=True))
    s.section_planes.append(SectionPlane(V(1.5, 1.5, 1.5), V(0, 0, 1), name="B", symbol="B"))
    return s


def scene_cut_off() -> Scene:
    s = scene_cut()
    s.show_section_cuts = False
    return s


def _styled(face_mode):
    def build():
        s = scene_cut()
        from core.style import Style
        s.display_style = Style(name=face_mode, face_mode=face_mode)
        return s
    return build


def scene_hiddenview() -> Scene:
    s = Scene()
    m = s.mesh
    box_faces(m, 0, 0, 0, 2, 2, 2)
    f = m.add_face([V(3, 0, 0), V(5, 0, 0), V(5, 0, 2), V(3, 0, 2)])
    f.attrs["hidden"] = True
    e = m.add_edge(V(-1, 3, 0), V(3, 4, 0))
    e.hidden = True
    s.groups.append(group(box(6, 0, 0, 1, 1, 1), "ghost", hidden=True))
    s.groups.append(group(box(6, 3, 0, 1, 1, 1), "show"))
    sm = box(-3, -4, 0, 1, 1, 1)
    sm.faces[1].attrs["hidden"] = True
    s.groups.append(group(sm, "gface"))
    s.show_hidden_objects = True
    s.show_hidden_geometry = True
    return s


def scene_hiddenview_off() -> Scene:
    s = scene_hiddenview()
    s.show_hidden_objects = False
    s.show_hidden_geometry = False
    return s


def scene_empty() -> Scene:
    return Scene()


def scene_many() -> Scene:
    """80 placements of one box: past 64 chunk spans the ray's box test is
    the vectorised slab."""
    s = Scene()
    p = box(0, 0, 0, 0.6, 0.6, 0.6, (0.4, 0.7, 0.4))
    for i in range(10):
        for j in range(8):
            s.groups.append(group(p, f"m{i}_{j}", xf(i * 1.0, j * 1.0, 0, 7.0 * (i + j))))
    s.mesh.add_face([V(-2, -2, -0.1), V(12, -2, -0.1), V(12, 10, -0.1), V(-2, 10, -0.1)])
    return s


def scene_tiny() -> Scene:
    """One triangle and one edge: a ray against a single triangle, a
    projection of a single edge."""
    s = Scene()
    s.mesh.add_face([V(0, 0, 0), V(3, 0, 0), V(0, 2, 0)])
    s.mesh.add_edge(V(4, 3, 0), V(6, 3.5, 1))
    return s


def scene_tiny_inst() -> Scene:
    """The same inside groups: an instance of a one-triangle prototype, one
    of a one-edge prototype."""
    s = Scene()
    t = Mesh()
    t.add_face([V(0, 0, 0), V(1, 0, 0), V(0, 1, 0)])
    e = Mesh()
    e.add_edge(V(0, 0, 0), V(1, 1, 1))
    s.groups.append(group(t, "tri", xf(1, 1, 0, 25)))
    s.groups.append(group(e, "edge", xf(3, 1, 0, 40)))
    return s


SCENES = {
    "loose": scene_loose,
    "groups": scene_groups,
    "annot": scene_annot,
    "cut": scene_cut,
    "cut_off": scene_cut_off,
    "xray": _styled("xray"),
    "wire": _styled("wireframe"),
    "hiddenline": _styled("hidden_line"),
    "hiddenview": scene_hiddenview,
    "hiddenview_off": scene_hiddenview_off,
    "empty": scene_empty,
    "many": scene_many,
    "tiny": scene_tiny,
    "tiny_inst": scene_tiny_inst,
}


def save_scene_json(scene, dest: Path) -> None:
    saved = (igz._LARGE_DOCUMENT, igz._pack_textures)
    igz._LARGE_DOCUMENT = 10 ** 9
    igz._pack_textures = lambda payload: ({}, 0)
    try:
        igz.save_scene(scene, dest)
    finally:
        igz._LARGE_DOCUMENT, igz._pack_textures = saved
    raw = dest.read_bytes()
    assert not raw.startswith(b"PK"), dest
    doc = json.loads(raw.decode("utf-8"))
    dest.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n",
                    encoding="utf-8", newline="\n")


# ---- viewport, identity, answers ---------------------------------------------------

class Ids:
    """Stable identities of everything a pick can return."""

    def __init__(self, vp: Viewport) -> None:
        sc = vp.scene
        self.sc = sc
        self.face = {id(f): i for i, f in enumerate(sc.mesh.faces)}
        self.edge = {id(e): i for i, e in enumerate(sc.mesh.edges)}
        self.gpath: dict = {}

        def walk(g, path):
            self.gpath[id(g)] = path
            for i, c in enumerate(g.children):
                walk(c, path + [i])
        for i, g in enumerate(sc.groups):
            walk(g, [i])
        # the ctx group, when it is nested, is reached through its ancestors
        vp._placements()
        cache = getattr(vp, "_placement_proxies", None) or {}
        real = {}

        def collect(g):
            real[id(g)] = g
            for c in g.children:
                collect(c)
        for g in sc.groups:
            collect(g)
        for k, p in cache.items():
            if k in self.gpath:
                self.gpath[id(p)] = self.gpath[k]
        self.dim = {id(d): i for i, d in enumerate(sc.dimensions)}
        self.lab = {id(d): i for i, d in enumerate(sc.text_labels)}
        self.guide = {id(d): i for i, d in enumerate(sc.guides)}
        self.sec = {id(d): i for i, d in enumerate(sc.section_planes)}
        self.img = {id(d): i for i, d in enumerate(sc.image_planes)}
        self.geo = {id(d): i for i, d in enumerate(sc.geo_paths)}

    def path(self, g):
        return None if g is None else self.gpath[id(g)]

    def face_enc(self, face, placement):
        if face is None:
            return None
        if id(face) in self.face:
            return ["l", self.face[id(face)]]
        faces = placement.mesh.faces
        for i, f in enumerate(faces):
            if f is face:
                return ["g", self.path(placement), i]
        raise AssertionError("face of no placement")

    def enc(self, ent):
        """The Select tool's answer, by kind."""
        if ent is None:
            return None
        if id(ent) in self.face:
            return ["face", self.face[id(ent)]]
        if id(ent) in self.edge:
            return ["edge", self.edge[id(ent)]]
        if id(ent) in self.gpath:
            return ["group", self.gpath[id(ent)]]
        for kind, table in (("dim", self.dim), ("label", self.lab), ("guide", self.guide),
                            ("section", self.sec), ("image", self.img), ("geo", self.geo)):
            if id(ent) in table:
                return [kind, table[id(ent)]]
        raise AssertionError(f"unknown entity {ent!r}")


def load_viewport(name: str) -> Viewport:
    vp = Viewport(None)
    vp.resize(W, H)
    vp.camera.set_aspect(W, H)
    vp.flash_status = lambda *a, **k: None
    igz.load_into(vp.scene, OUT / f"{name}.igz.json")
    vp.scene.version += 1
    return vp


def set_camera(vp: Viewport, cam: dict) -> None:
    c = vp.camera
    c.yaw = cam["yaw"]
    c.pitch = cam["pitch"]
    c.distance = cam["distance"]
    c.target = V(*cam["target"])
    c.perspective = cam["perspective"]
    c.fov_deg = cam["fov"]
    c.set_aspect(W, H)
    vp.scene.version += 1


def cam_rec(vp: Viewport, cam: dict) -> dict:
    c = vp.camera
    return {
        "yaw": h64(c.yaw), "pitch": h64(c.pitch), "distance": h64(c.distance),
        "target": vbits(c.target), "up": vbits(c.up), "perspective": c.perspective,
        "fov": h64(c.fov_deg), "aspect": h64(c.aspect), "znear": h64(c.znear),
        "zfar": h64(c.zfar),
        "view": mbits(c.view_matrix()), "proj": mbits(c.projection_matrix()),
        "eye": vbits(c.eye()),
    }


def metrics_rec(vp: Viewport) -> dict:
    sc = vp.scene
    style = getattr(sc, "dimension_style", {})
    font = QFont()
    font.setPointSize(int(style.get("font_size", 9)))
    font.setBold(True)
    fm = QFontMetrics(font, vp)
    adv = {}
    for lab in sc.text_labels:
        for ln in (lab.text.splitlines() or [""]):
            adv[ln] = fm.horizontalAdvance(ln)
    return {"height": fm.height(), "ascent": fm.ascent(), "descent": fm.descent(), "adv": adv}


def _edge_d(vp: Viewport, x: float, y: float):
    """The distances ``Viewport._pick_edge`` sorts (visible-only form)."""
    import numpy as np
    idx = vp._pick_index(near="skip")
    if idx.edge_a is None:
        return idx, None
    ax, ay, oka = vp._project_px(idx.edge_a)
    bx, by, okb = vp._project_px(idx.edge_b)
    ok = oka & okb & idx.edge_sel
    if idx.edge_soft is not None and not vp.scene.show_hidden_geometry:
        ok &= ~idx.edge_soft
    sp = _active_cut(vp.scene)
    if sp is not None:
        n = np.array([sp.normal.x(), sp.normal.y(), sp.normal.z()])
        c = float(n @ [sp.point.x(), sp.point.y(), sp.point.z()])
        da = idx.edge_a @ n - c
        db = idx.edge_b @ n - c
        ok &= ~((da > 1e-6) & (db > 1e-6))
    dx, dy = bx - ax, by - ay
    l2 = dx * dx + dy * dy
    safe = np.where(l2 > 1e-12, l2, 1.0)
    t = np.clip(((x - ax) * dx + (y - ay) * dy) / safe, 0.0, 1.0)
    d = np.hypot(ax + t * dx - x, ay + t * dy - y)
    return idx, np.where(ok, d, np.inf)


def tie_visible_edge(vp: Viewport, x: float, y: float, ret) -> bool:
    """Whether another edge at exactly the returned one's distance could
    have been returned instead: NumPy's argsort (AVX2 on this machine) is
    not stable, so it is the machine that chose."""
    import numpy as np
    if ret is None:
        return False
    idx, d = _edge_d(vp, x, y)
    if d is None:
        return False
    i = next(k for k, e in enumerate(idx.edges) if e is ret)
    for j in np.flatnonzero(d == d[i]):
        if int(j) == i or not d[j] < vp.pick_threshold_px:
            continue
        p = vp.edge_point_under_cursor(idx.edges[int(j)], x, y)
        if p is None or not vp._is_occluded(p):
            return True
    return False


def tie_vertex(vp: Viewport, x: float, y: float, ret) -> bool:
    """The same for ``pick_vertex``: another corner, elsewhere in space, at
    the very distance of the returned one and in view."""
    import numpy as np
    if ret is None:
        return False
    idx = vp._pick_index(near=("px", x, y))
    xs, ys, oks, pos = [], [], [], []
    n = len(idx.edges) if idx.edge_a is not None else 0
    if n:
        ax, ay, oka = vp._project_px(idx.edge_a)
        bx, by, okb = vp._project_px(idx.edge_b)
        ok = oka & okb
        xs += [ax, bx]; ys += [ay, by]; oks += [ok, ok]
        pos += [idx.edge_a, idx.edge_b]
    gp = vp._gedge_screen()
    if gp is not None:
        ax, ay, bx, by, ok = gp
        xs += [ax, bx]; ys += [ay, by]; oks += [ok, ok]
        pos += [idx.gedge_a, idx.gedge_b]
    if not xs:
        return False
    px, py, ok = np.concatenate(xs), np.concatenate(ys), np.concatenate(oks)
    pts = np.concatenate(pos).astype(np.float32)
    d = np.where(ok, np.hypot(px - x, py - y), np.inf)
    want = np.float32([ret.x(), ret.y(), ret.z()])
    same = (pts == want).all(axis=1)
    if not same.any():
        return False
    d0 = d[same].min()
    for j in np.flatnonzero((d == d0) & ~same):
        w = V(*[float(c) for c in pts[j]])
        if not vp._is_occluded(w):
            return True
    return False


def tie_group_edge(vp: Viewport, x: float, y: float) -> bool:
    """``pick_group``'s edge fallback: two unoccluded edges of different
    groups at the same distance."""
    import numpy as np
    idx = vp._pick_index(near=("px", x, y))
    proj = vp._gedge_screen()
    if proj is None:
        return False
    ax, ay, bx, by, ok = proj
    dx, dy = bx - ax, by - ay
    l2 = dx * dx + dy * dy
    safe = np.where(l2 > 1e-12, l2, 1.0)
    t = np.clip(((x - ax) * dx + (y - ay) * dy) / safe, 0.0, 1.0)
    d = np.hypot(ax + t * dx - x, ay + t * dy - y)
    sel = getattr(idx, "gedge_sel", None)
    if sel is not None:
        ok = ok & sel
    d = np.where(ok, d, np.inf)
    near = np.where(d < vp.pick_threshold_px)[0]
    seen: dict = {}
    for i in near:
        a = V(*idx.gedge_a[i])
        p = a + (V(*idx.gedge_b[i]) - a) * float(t[i])
        if vp._is_occluded(p):
            continue
        key = float(d[i])
        owner = id(idx.gedge_groups[int(idx.gedge_gi[i])])
        if key in seen and seen[key] != owner:
            return True
        seen.setdefault(key, owner)
    return False


def answers(vp: Viewport, ids: Ids, x: float, y: float, tool: SelectTool) -> dict:
    out: dict = {}
    amb: list = []

    def put(k, v):
        if v is not None:
            out[k] = v
    put("f", ids.face_enc(vp.pick_face(x, y), None))
    face, owner = vp.pick_face_any(x, y)
    pf, pl = vp.pick_face_placement(x, y)
    assert pf is face
    if face is not None:
        put("fa", [ids.face_enc(face, pl), ids.path(owner), ids.path(pl)])
    g = vp.pick_group(x, y)
    put("g", ids.path(g))
    if g is not None and tie_group_edge(vp, x, y):
        amb.append("g")
    e = vp.pick_edge(x, y)
    put("e", None if e is None else ids.edge[id(e)])
    e = vp.pick_visible_edge(x, y)
    put("ve", None if e is None else ids.edge[id(e)])
    if tie_visible_edge(vp, x, y, e):
        amb.append("ve")
    v = vp.pick_vertex(x, y)
    put("v", None if v is None else vbits(v))
    if tie_vertex(vp, x, y, v):
        amb.append("v")
    d = vp.pick_dimension(x, y)
    put("d", None if d is None else ids.dim[id(d)])
    t = vp.pick_text_label(x, y)
    put("t", None if t is None else ids.lab[id(t)])
    t = vp.pick_text_label(x, y, rect_only=True)
    put("tr", None if t is None else ids.lab[id(t)])
    g = vp.pick_guide(x, y)
    put("gd", None if g is None else ids.guide[id(g)])
    s = vp.pick_section_plane(x, y)
    put("sp", None if s is None else ids.sec[id(s)])
    i = vp.pick_image_plane(x, y)
    put("im", None if i is None else ids.img[id(i)])
    i = vp.pick_locked_image_plane(x, y)
    put("il", None if i is None else ids.img[id(i)])
    put("ax", vp.pick_axis(x, y))
    p = vp.pick_geopath(x, y)
    put("gp", None if p is None else ids.geo[id(p)])
    sel = ids.enc(tool._pick(vp, x, y))
    put("s", sel)
    if sel is not None and ((sel[0] == "edge" and "ve" in amb)
                            or (sel[0] == "group" and "g" in amb)):
        amb.append("s")
    if amb:
        out["amb"] = amb
    return out


def rect_answer(vp: Viewport, ids: Ids, tool: SelectTool, rect, crossing) -> list:
    sc = vp.scene
    got: list = []
    orig = sc.select
    sc.select = lambda picked, additive=False, mode=None: got.append(list(picked))
    try:
        tool.on_box_select(vp, rect, crossing, additive=False, mode="replace")
    finally:
        sc.select = orig
    assert len(got) == 1
    return [ids.enc(e) for e in got[0]]


# ---- pixels ------------------------------------------------------------------------

def interesting_points(vp: Viewport) -> list:
    """World points whose pixels make the hard cases: every corner, edge
    middles and face centres of the loose mesh and of the placements."""
    sc = vp.scene
    pts = []
    for v in sc.mesh.vertices:
        pts.append(v.position)
    for e in sc.mesh.edges:
        pts.append((e.a + e.b) * 0.5)
    for f in sc.mesh.faces:
        pts.append(f.centroid())
    for g in vp._placements():
        if getattr(g, "billboard", False):
            continue
        m = g.xform
        for v in g.mesh.vertices:
            pts.append(m.map(v.position) if m is not None else v.position)
        for f in g.mesh.faces:
            c = f.centroid()
            pts.append(m.map(c) if m is not None else c)
    for d in sc.dimensions:
        ap, bp = d.line_points()
        pts += [d.a, d.b, ap, bp, (ap + bp) * 0.5]
    for lab in sc.text_labels:
        pts += [lab.anchor, lab.position()]
    for g in sc.guides:
        pts.append(g.point)
    for p in sc.geo_paths:
        pts += list(p.points)
    for im in sc.image_planes:
        pts += list(im.corners()) + [im.center()]
    for sp in sc.section_planes:
        pts += list(vp.section_plane_frame(sp))
    return pts


def pixel_set(vp: Viewport, rnd: random.Random, grid: int) -> list:
    px = []
    # a regular grid, offset so no pixel sits on a symmetric axis
    for i in range(grid):
        for j in range(max(2, grid * 3 // 5)):
            px.append((W * (i + 0.37) / grid, H * (j + 0.41) / max(2, grid * 3 // 5)))
    pts = interesting_points(vp)
    rnd.shuffle(pts)
    for p in pts[:70]:
        q = vp._world_to_pixel(p)
        if q is None:
            continue
        px.append(q)                                   # exactly on it
        dx, dy = rnd.uniform(-9, 9), rnd.uniform(-9, 9)
        px.append((q[0] + dx, q[1] + dy))              # around the threshold
        px.append((round(q[0]), round(q[1])))          # whole pixels, as a mouse gives
    px += [(-5.0, 10.0), (W + 7.0, H / 2), (W / 2, -3.0), (0.0, 0.0), (float(W), float(H)),
           (W / 2, H / 2), (500.0, 300.0), (499.5, 300.5)]
    return px


def rect_set(vp: Viewport, rnd: random.Random, n: int) -> list:
    rects = []
    pts = [vp._world_to_pixel(p) for p in interesting_points(vp)]
    pts = [p for p in pts if p is not None]
    for _ in range(n):
        if pts and rnd.random() < 0.6:
            a, b = rnd.choice(pts), rnd.choice(pts)
            x0, x1 = sorted((a[0] + rnd.uniform(-12, 12), b[0] + rnd.uniform(-12, 12)))
            y0, y1 = sorted((a[1] + rnd.uniform(-12, 12), b[1] + rnd.uniform(-12, 12)))
        else:
            x0, x1 = sorted((rnd.uniform(-50, W + 50), rnd.uniform(-50, W + 50)))
            y0, y1 = sorted((rnd.uniform(-50, H + 50), rnd.uniform(-50, H + 50)))
        rects.append((x0, y0, x1, y1))
    rects += [(0.0, 0.0, float(W), float(H)), (-10.0, -10.0, W + 10.0, H + 10.0),
              (400.0, 200.0, 600.0, 400.0)]
    return rects


# ---- views -------------------------------------------------------------------------

def cams(kind: str) -> list:
    r = math.radians
    base = {"fov": 45.0, "perspective": True}
    out = [
        dict(base, yaw=r(-45.0), pitch=r(30.0), distance=22.0, target=(3.0, 1.0, 0.5)),
        dict(base, yaw=-math.pi / 2, pitch=math.pi / 2, distance=26.0, target=(3.0, 1.0, 0.0),
             perspective=False),                                    # top, parallel
        dict(base, yaw=r(-90.0), pitch=r(5.0), distance=18.0, target=(3.0, 2.0, 1.0)),
        dict(base, yaw=r(120.0), pitch=r(-25.0), distance=15.0, target=(2.0, 1.0, 1.0),
             fov=60.0),                                             # from below
        dict(base, yaw=r(20.0), pitch=r(40.0), distance=14.0, target=(5.0, 3.0, 0.0),
             perspective=False),
        dict(base, yaw=-math.pi / 2, pitch=-math.pi / 2, distance=20.0,
             target=(2.0, 1.0, 0.0)),                               # bottom
        dict(base, yaw=r(-135.0), pitch=r(55.0), distance=7.0, target=(1.0, 1.0, 1.0)),
    ]
    return out


def views_for(name: str) -> list:
    """(edit path or None, rest mode) combinations of a scene."""
    if name == "groups":
        return [(None, "fade"), ([0], "fade"), ([1], "fade"), ([5], "fade"),
                ([5], "hide"), ([5, 0], "fade"), ([4], "fade"), ([9], "fade")]
    if name in ("loose", "annot", "cut", "hiddenview"):
        return [(None, "fade")]
    return [(None, "fade")]


def find_group(sc, path):
    g = sc.groups[path[0]]
    for i in path[1:]:
        g = g.children[i]
    return g


def make_view(name: str, edit, rest: str, cam: dict, rnd: random.Random,
              grid: int, nrect: int) -> dict:
    vp = load_viewport(name)
    sc = vp.scene
    if edit is not None:
        # level by level, as the user opens nested groups
        for k in range(1, len(edit) + 1):
            sc.begin_group_edit(find_group(sc, edit[:k]))
    vp._edit_rest_mode = rest
    set_camera(vp, cam)
    ids = Ids(vp)
    tool = SelectTool()
    rec = {
        "scene": name, "edit": edit, "rest": rest, "cam": cam_rec(vp, cam),
        "metrics": metrics_rec(vp),
        "face_mode": vp._effective_style().face_mode,
    }
    pix = pixel_set(vp, rnd, grid)
    rec["px"] = [[h64(x), h64(y), answers(vp, ids, x, y, tool)] for x, y in pix]
    rects = rect_set(vp, rnd, nrect)
    rec["rects"] = []
    for r in rects:
        for crossing in (False, True):
            rec["rects"].append([[h64(v) for v in r], crossing,
                                 rect_answer(vp, ids, tool, r, crossing)])
    vp.deleteLater()
    return rec


def numerics() -> dict:
    """The NumPy arithmetic the index runs, recorded for 1..40, 100 and 257
    rows: ``(N,3) @ (3,)``, ``pts @ M[:, :3].T + M[:, 3]``, ``einsum``,
    ``np.cross``, ``np.hypot`` and ``np.linalg.det`` of 3x3 matrices."""
    import numpy as np
    rng = np.random.default_rng(42)
    out: dict = {"gemv": [], "gemm": [], "einsum": [], "cross": [], "hypot": [], "det": []}

    def rows(a):
        return [[h64(float(v)) for v in r] for r in a]
    for n in list(range(1, 41)) + [100, 257]:
        scale = float(rng.choice([1.0, 100.0, 1e-3]))
        q = rng.normal(size=(n, 3)) * scale
        d = rng.normal(size=3)
        out["gemv"].append({"q": rows(q), "d": rows([d])[0], "out": [h64(float(v)) for v in q @ d]})
        m = rng.normal(size=(4, 4))
        out["gemm"].append({"p": rows(q), "m": rows(m), "out": rows(q @ m[:, :3].T + m[:, 3])})
        e1, e2 = rng.normal(size=(n, 3)), rng.normal(size=(n, 3)) * scale
        out["einsum"].append({"a": rows(e1), "b": rows(e2),
                              "out": [h64(float(v)) for v in np.einsum("ij,ij->i", e1, e2)]})
        out["cross"].append({"a": rows([d])[0], "b": rows(e2), "out": rows(np.cross(d, e2))})
        # the other cross, row by row
        out["cross"].append({"a": None, "ra": rows(e1), "b": rows(e2), "out": rows(np.cross(e1, e2))})
    ha, hb = rng.normal(size=400) * 300.0, rng.normal(size=400) * 300.0
    out["hypot"] = {"a": [h64(float(v)) for v in ha], "b": [h64(float(v)) for v in hb],
                    "out": [h64(float(v)) for v in np.hypot(ha, hb)]}
    mats = [np.eye(3), np.diag([1.5, 1.0, 2.0]), np.diag([-1.0, 1.0, 1.0]), np.zeros((3, 3))]
    for ang in (15.0, 30.0, 45.0, 90.0, 120.0):
        m = QMatrix4x4()
        m.rotate(ang, 0, 0, 1)
        d = m.data()
        mats.append(np.array([[d[0], d[4], d[8]], [d[1], d[5], d[9]], [d[2], d[6], d[10]]],
                             dtype=np.float64))
    for _ in range(60):
        mats.append(rng.normal(size=(3, 3)))
    out["det"] = [{"m": rows(m), "out": h64(float(np.linalg.det(m))),
                   "exact": bool(i < 4)} for i, m in enumerate(mats)]
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "numerics.json").write_text(
        json.dumps(numerics(), sort_keys=True, separators=(",", ":")), encoding="utf-8",
        newline="\n")
    rnd = random.Random(20260920)
    for name, build in SCENES.items():
        save_scene_json(build(), OUT / f"{name}.igz.json")
    views = []
    for name in SCENES:
        grid = 10 if name in ("empty",) else 12
        for edit, rest in views_for(name):
            cam_list = cams(name)
            if edit is not None:
                cam_list = cam_list[:4]
            if name in ("xray", "wire", "hiddenline", "cut_off", "hiddenview_off",
                        "many", "tiny", "tiny_inst"):
                cam_list = cam_list[:3]
            for cam in cam_list:
                views.append(make_view(name, edit, rest, cam, rnd, grid, 14))
                print(name, edit, rest, len(views), flush=True)
    (OUT / "cases.json").write_text(
        json.dumps(views, sort_keys=True, separators=(",", ":")), encoding="utf-8",
        newline="\n")
    print(len(views), "views")


if __name__ == "__main__":
    main()
