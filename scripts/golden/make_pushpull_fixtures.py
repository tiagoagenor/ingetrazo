# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
"""Fixtures that hold ``ingetrazo-ops``' push/pull operation and
``ingetrazo-tools``' Push/Pull tool to ``tools/pushpull.py``
(docs/rust-rewrite-plan.md, M3.4).

Each *program* is a sequence of tool steps on a scene: a box from a
rectangle, a recess in an existing solid, a face pulled out of a box side,
a through-hole, a push to zero, copy pushes (Ctrl), a pull against a wall,
faces with holes, curved (soft) faces, faces inside an open group, the
repeat of the last distance, tilted faces, a hot retype, the inward limit
and the refusal guard. Scenes are built with the same helpers the tests use
(``tests/test_fuzz_engine``: ``_draw_rect`` and ``_push``), saved as
bare-JSON .igz (``<scene>.igz.json``) and reopened in a fresh scene, so
Python and Rust start from the very same bits.

Every step drives the REAL ``tools.pushpull.PushPullTool`` the way the tests
do, on a stub viewport (offscreen Qt, no window):

* ``value``  : hover the face, click (starts the drag), type a distance (the
               VCB) - Ctrl held on the click for a copy push;
* ``click2`` : click, set the extrusion as a hover would, click again;
* ``double`` : double-click on the face (repeats ``last_distance``);
* ``op``     : the ``_push`` helper of the fuzz bench - the drag state set up
               by hand, the inward limit computed, the extrusion clamped;
* ``retype`` : type a distance on the idle tool (the hot retype);
* ``undo`` / ``redo`` / ``edit`` / ``end_edit``.

``cases.json`` holds per program the ordered state of the active mesh
(vertices as float32 bits, edges with flags/curve/faces, faces with loops,
holes, interior flag and attrs hash) before, after every step (with the
status message the tool flashed, ``last_distance`` and the undo depth),
after undoing everything and after redoing everything.

Python's own order is not repeatable in a few places, pinned here (the Rust
side takes the same order, see ``rust/crates/ingetrazo-ops/src/pushpull.rs``):

* ``core.mesh.set`` is swapped for a set that iterates its mesh objects in
  creation order (``Vertex.edges``, ``Vertex.faces()``), and so are
  ``core.history._coplanar_component`` and ``core.mesh._trace_loops`` (as in
  ``make_stitch_fixtures.py``);
* ``cap_rebuild.seam_planes`` walks the operation's faces (a ``set`` of
  objects, address order) in creation order - it decides which face's
  centroid stands as a plane's origin.

Committed; rerun only to widen (PYTHONPATH at the repository root):

    python scripts/golden/make_pushpull_fixtures.py

Deleted with the Python app at M10.
"""
from __future__ import annotations

import json
import math
import os
import random
import shutil
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.pop("INGETRAZO_RUST", None)

from PySide6.QtCore import QPointF, Qt  # noqa: E402
from PySide6.QtGui import QGuiApplication, QMatrix4x4  # noqa: E402

import core.cap_rebuild as cr  # noqa: E402
import core.history as ch  # noqa: E402
import core.mesh as cm  # noqa: E402
import tools.pushpull as tp  # noqa: E402
from core.edits import build_add_edge, build_add_edges  # noqa: E402
from core.group import Group  # noqa: E402
from core.history import (AddFaceCommand, History, MoveVerticesCommand,  # noqa: E402
                          TagCurveCommand)
from core.mesh import Mesh  # noqa: E402
from core.orient import orient_outward  # noqa: E402
from core.scene import Scene  # noqa: E402
from core.vecmath import QVector3D  # noqa: E402
from formats import igz  # noqa: E402
from tests.golden.signature import _dumps, canonical, sha  # noqa: E402
from tests.test_fuzz_engine import _draw_rect  # noqa: E402
from tools.base import ToolContext  # noqa: E402

OUT = ROOT / "rust" / "crates" / "ingetrazo-tools" / "tests" / "data" / "pushpull"

_APP = QGuiApplication.instance() or QGuiApplication([])


def V(x, y, z=0.0):
    return QVector3D(float(x), float(y), float(z))

# ---- pinning Python's order ---------------------------------------------------------

ORD: dict = {}                       # id(object) -> creation ordinal
KEEP: list = []                      # keeps them alive (ids stay unique)


def _reg(o) -> None:
    if id(o) not in ORD:
        ORD[id(o)] = len(KEEP)
        KEEP.append(o)


class DetSet(set):
    """A set that iterates mesh objects in creation order."""

    def __iter__(self):
        return iter(sorted(set.__iter__(self), key=lambda o: ORD.get(id(o), 1 << 60)))


cm.set = DetSet


def _wrap(name: str) -> None:
    orig = getattr(Mesh, name)

    def wrapper(self, *a, **k):
        o = orig(self, *a, **k)
        _reg(o)
        return o
    setattr(Mesh, name, wrapper)


_wrap("vertex")
_wrap("_link_edge")
_wrap("add_face")

_orig_component = ch._coplanar_component
ch._coplanar_component = lambda *a, **k: DetSet(_orig_component(*a, **k))


def _det_trace_loops(outline):
    adj: dict = {}
    edges: dict = {}
    for u, w in outline:
        if u is w:
            return None
        adj.setdefault(u, []).append(w)
        adj.setdefault(w, []).append(u)
        edges.setdefault(frozenset((u, w)), (u, w))
    for nbrs in adj.values():
        if len(nbrs) != 2:
            return None
    loops = []
    while edges:
        a, b = next(iter(edges.values()))
        edges.pop(frozenset((a, b)), None)
        loop = [a, b]
        prev, cur = a, b
        while True:
            nbrs = adj[cur]
            nxt = nbrs[0] if nbrs[0] is not prev else nbrs[1]
            edges.pop(frozenset((cur, nxt)), None)
            if nxt is loop[0]:
                break
            loop.append(nxt)
            prev, cur = cur, nxt
        loops.append(loop)
    return loops


cm._trace_loops = _det_trace_loops

_orig_seam_planes = cr.seam_planes


def _seam_planes_in_creation_order(mesh, new_faces):
    ordered = sorted(new_faces, key=lambda o: ORD.get(id(o), 1 << 60))
    return _orig_seam_planes(mesh, ordered)


tp.seam_planes = _seam_planes_in_creation_order


def _pin_resolve_stage() -> None:
    """``PushPullTool._resolve_stage`` walks ``{f for v in movers for f in
    v.faces()}``, a set of Face objects (address order); that order decides
    which two normals ``_slide_rate`` keeps when a mover drags three. Run it
    in creation order."""
    import inspect
    import textwrap
    src = textwrap.dedent(inspect.getsource(tp.PushPullTool._resolve_stage))
    old = "for f in {f for v in movers for f in v.faces()}:"
    assert old in src
    src = src.replace(old, "for f in sorted({f for v in movers for f in v.faces()},\n                   key=lambda o: ORD.get(id(o), 1 << 60)):")
    ns = dict(tp.__dict__)
    ns["ORD"] = ORD
    exec(src, ns)
    tp.PushPullTool._resolve_stage = ns["_resolve_stage"]


_pin_resolve_stage()


def register_all(mesh) -> None:
    """Number the objects a reopened scene brought, in list order."""
    for v in mesh.vertices:
        _reg(v)
    for e in mesh.edges:
        _reg(e)
    for f in mesh.faces:
        _reg(f)


# ---- encoding ----------------------------------------------------------------------

def h32(x: float) -> str:
    return struct.pack(">f", x).hex()


def pbits(p) -> list:
    return [h32(p.x()), h32(p.y()), h32(p.z())]


def mesh_state(mesh) -> dict:
    vi = {id(v): i for i, v in enumerate(mesh.vertices)}
    fi = {id(f): i for i, f in enumerate(mesh.faces)}
    return {
        "v": [pbits(v.position) for v in mesh.vertices],
        "e": [[vi[id(e.v0)], vi[id(e.v1)], 1 if e.soft else 0, 1 if e.hidden else 0,
               e.layer, e.curve, [fi[id(f)] for f in e.faces]] for e in mesh.edges],
        "f": [[[vi[id(v)] for v in f.loop],
               [[vi[id(v)] for v in h] for h in f.hole_loops],
               1 if getattr(f, "interior", False) else 0,
               sha(_dumps(canonical(f.attrs)), 16) if f.attrs else ""]
              for f in mesh.faces],
    }


def snap(scene) -> dict:
    """The active mesh, every group's mesh and whether a group is open."""
    return {"m": mesh_state(scene.mesh), "g": [mesh_state(g.mesh) for g in scene.groups],
            "edit": scene.edit_group is not None}


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


def reopen(path: Path) -> Scene:
    s = Scene()
    igz.load_into(s, path)
    return s


# ---- building scenes (the tests' own helpers) -------------------------------------------

def op_push(scene, hist, face, dist, group=None, keep_base=False) -> bool:
    """tests/test_fuzz_engine._push: the drag state set up by hand."""
    class _Stub:
        def __init__(self, scene, history):
            self.scene, self.history = scene, history

        def update(self): pass
        def set_hover(self, e): pass
        def set_suppressed_faces(self, f): pass
        def flash_status(self, text, msec=2500): pass

    tool = tp.PushPullTool()
    tool.base_face = face
    tool.dragging = True
    tool._group = group
    target = tool._target_scene(scene)
    tool._anchor = face.centroid()
    tool._normal = face.normal()
    tool._attached, tool._prism_cap = tool._classify_base(target)
    tool._cap_positions = tool._cap_loop_positions(face)
    tool._keep_base = keep_base
    tool._compute_inward_limit(target)
    tool.extrusion = float(dist)
    tool._clamp_extrusion()
    if abs(tool.extrusion) < 1e-6:
        tool._reset()
        return False
    tool._commit(_Stub(scene, hist))
    return True


def up(face, dist):
    return dist if face.normal().z() > 0 else -dist


def rect(scene, hist, x0, y0, z0, w, h, plane="z"):
    if plane == "z":
        c = [V(x0, y0, z0), V(x0 + w, y0, z0), V(x0 + w, y0 + h, z0), V(x0, y0 + h, z0)]
    elif plane == "x":      # a wall at x = z0 (y, z span)
        c = [V(z0, x0, y0), V(z0, x0 + w, y0), V(z0, x0 + w, y0 + h), V(z0, x0, y0 + h)]
    else:                   # a wall at y = z0 (x, z span)
        c = [V(x0, z0, y0), V(x0 + w, z0, y0), V(x0 + w, z0, y0 + h), V(x0, z0, y0 + h)]
    _draw_rect(scene, hist, c, [])
    return c


def face_where(scene, pred, nth=0):
    hits = [f for f in scene.mesh.faces if pred(f)]
    return hits[nth]


def on_z(z):
    return lambda f: all(abs(v.z() - z) < 1e-6 for v in f.vertices)


def on_x(x):
    return lambda f: all(abs(v.x() - x) < 1e-6 for v in f.vertices)


def on_y(y):
    return lambda f: all(abs(v.y() - y) < 1e-6 for v in f.vertices)


def area_near(a):
    return lambda f: abs(f.area() - a) < 1e-5


def both(*ps):
    return lambda f: all(p(f) for p in ps)


def cube(size=4.0, height=3.0, x0=0.0, y0=0.0):
    scene = Scene()
    hist = History(scene)
    rect(scene, hist, x0, y0, 0.0, size, size)
    f = scene.mesh.faces[0]
    op_push(scene, hist, f, up(f, height))
    return scene, hist


def scn_cube():
    return cube()[0]


def scn_cube_top_rect():
    scene, hist = cube()
    rect(scene, hist, 1.0, 1.0, 3.0, 1.0, 1.0)
    return scene


def scn_cube_top_rect_big():
    scene, hist = cube(size=6.0, height=3.0)
    rect(scene, hist, 1.0, 1.0, 3.0, 2.5, 2.0)
    rect(scene, hist, 4.0, 4.0, 3.0, 1.0, 1.0)
    return scene


def scn_cube_side_rect():
    scene, hist = cube()
    rect(scene, hist, 1.0, 1.0, 4.0, 1.5, 1.0, plane="x")
    return scene


def scn_cube_side_rect_front():
    scene, hist = cube()
    rect(scene, hist, 1.0, 0.5, 0.0, 1.5, 1.5, plane="y")
    return scene


def scn_cube_corner_rect():
    scene, hist = cube()
    rect(scene, hist, 3.0, 3.0, 3.0, 1.0, 1.0)      # a corner of the top
    return scene


def scn_cube_two_floors():
    """A 4x4 box 3 high with a 4x4x3 box on half of its top."""
    scene, hist = cube()
    rect(scene, hist, 0.0, 0.0, 3.0, 2.0, 4.0)
    sub = face_where(scene, both(on_z(3.0), area_near(8.0)))
    op_push(scene, hist, sub, up(sub, 2.0))
    return scene


def scn_ring():
    scene = Scene()
    outer = [V(0, 0, 0), V(4, 0, 0), V(4, 4, 0), V(0, 4, 0)]
    hole = [V(1, 1, 0), V(1, 3, 0), V(3, 3, 0), V(3, 1, 0)]
    scene.mesh.add_face(outer, [hole])
    return scene


def scn_ring_on_box():
    scene, hist = cube(size=6.0)
    rect(scene, hist, 1.0, 1.0, 3.0, 4.0, 4.0)
    rect(scene, hist, 2.0, 2.0, 3.0, 2.0, 2.0)
    return scene


def circle_pts(r, n, cx, cy, z):
    return [V(cx + r * math.cos(2 * math.pi * k / n), cy + r * math.sin(2 * math.pi * k / n), z)
            for k in range(n)]


def draw_circle(scene, hist, pts):
    n = len(pts)
    segs = [(pts[i], pts[(i + 1) % n]) for i in range(n)]
    hist.execute(build_add_edges(scene, segs, detect_faces=False, extra=[
        AddFaceCommand(list(pts)), TagCurveCommand(list(pts), closed=True)]))


def scn_disc():
    scene = Scene()
    hist = History(scene)
    draw_circle(scene, hist, circle_pts(1.5, 24, 0.0, 0.0, 0.0))
    return scene


def scn_disc_on_box():
    scene, hist = cube(size=6.0)
    draw_circle(scene, hist, circle_pts(1.5, 24, 3.0, 3.0, 3.0))
    return scene


def scn_hexagon():
    scene = Scene()
    hist = History(scene)
    draw_circle(scene, hist, circle_pts(1.5, 6, 0.0, 0.0, 0.0))
    return scene


def scn_gable():
    scene = Scene()
    hist = History(scene)
    _draw_rect(scene, hist, [V(0, 0), V(4, 0), V(4, 4), V(0, 4)], [])
    f = scene.mesh.faces[0]
    op_push(scene, hist, f, up(f, 3.0))
    hist.execute(build_add_edge(scene, V(2, 0, 3), V(2, 4, 3)))
    hist.execute(MoveVerticesCommand([V(2, 0, 3), V(2, 4, 3)], V(0, 0, 1)))
    return scene


def scn_wedge():
    scene = Scene()
    hist = History(scene)
    _draw_rect(scene, hist, [V(0, 0), V(4, 0), V(4, 4), V(0, 4)], [])
    f = scene.mesh.faces[0]
    op_push(scene, hist, f, up(f, 3.0))
    hist.execute(MoveVerticesCommand([V(0, 0, 3), V(0, 4, 3)], V(0, 0, -2)))
    return scene


def scn_tilted_sheet():
    """A rectangle tilted 30 degrees about x, free in space."""
    scene = Scene()
    m = QMatrix4x4()
    m.rotate(30.0, 1.0, 0.0, 0.0)
    pts = [m.map(V(x, y, 0.0)) for x, y in ((0, 0), (3, 0), (3, 2), (0, 2))]
    scene.mesh.add_face(pts)
    return scene


def scn_step_square():
    scene, hist = cube()
    hist.execute(build_add_edges(scene, [(V(0, 2, 3), V(4, 2, 3))]))
    front = [f for f in scene.mesh.faces if on_z(3.0)(f) and max(v.y() for v in f.vertices) <= 2 + 1e-9][0]
    op_push(scene, hist, front, -1.0)
    hist.execute(build_add_edges(scene, [(V(1, 1, 2), V(2, 1, 2)), (V(2, 1, 2), V(2, 2, 2)),
                                         (V(1, 1, 2), V(1, 2, 2))]))
    return scene


def scn_plan():
    scene = Scene()
    hist = History(scene)
    _draw_rect(scene, hist, [V(0, 0), V(3, 0), V(3, 4), V(0, 4)], [])
    _draw_rect(scene, hist, [V(3, 0), V(6, 0), V(6, 4), V(3, 4)], [])
    return scene


def scn_sheet_beside_box():
    scene, hist = cube(size=3.0, height=2.0)
    rect(scene, hist, 6.0, 0.0, 0.0, 2.0, 2.0)
    rect(scene, hist, 6.0, 3.0, 0.0, 2.0, 2.0)
    return scene


def scn_open_cube_and_rect():
    scene, hist = cube(size=1.0, height=1.0)
    top = face_where(scene, on_z(1.0))
    hist.execute(ch.EraseSelectionCommand([], faces=[top]))
    rect(scene, hist, 3.0, 0.0, 0.0, 1.0, 1.0)
    return scene


def scn_triangle_prism():
    scene = Scene()
    hist = History(scene)
    scene.mesh.add_face([V(-0.2, -2.8), V(2.7, -7.2), V(4.1, -2.7)])
    f = scene.mesh.faces[0]
    op_push(scene, hist, f, up(f, 3.0))
    return scene


def scn_group_cube():
    scene = Scene()
    g = Group()
    hist = History(scene)
    scene.groups.append(g)
    scene.begin_group_edit(g)
    rect(scene, hist, 0.0, 0.0, 0.0, 4.0, 4.0)
    f = scene.mesh.faces[0]
    op_push(scene, hist, f, up(f, 3.0))
    rect(scene, hist, 1.0, 1.0, 3.0, 1.0, 1.0)
    scene.end_group_edit()
    # a loose box beside it, to tell the meshes apart
    rect(scene, hist, 10.0, 0.0, 0.0, 2.0, 2.0)
    return scene


def scn_stairs():
    """Three steps: 4 wide, each 1 deeper and 1 higher."""
    scene = Scene()
    hist = History(scene)
    rect(scene, hist, 0.0, 0.0, 0.0, 4.0, 3.0)
    f = scene.mesh.faces[0]
    op_push(scene, hist, f, up(f, 1.0))
    rect(scene, hist, 0.0, 0.0, 1.0, 4.0, 2.0)
    g = face_where(scene, both(on_z(1.0), area_near(8.0)))
    op_push(scene, hist, g, up(g, 1.0))
    rect(scene, hist, 0.0, 0.0, 2.0, 4.0, 1.0)
    h = face_where(scene, both(on_z(2.0), area_near(4.0)))
    op_push(scene, hist, h, up(h, 1.0))
    return scene


TEXTURE = {"path": "tile.png", "sw": 0.5, "sh": 0.5, "rot": 0,
           "uvw": [1.0, 0.0, 0.0, 0.25, 0.0, 1.0, 0.0, 0.5, 0.0, 0.0, 1.0, 0.0]}
IFC = {"class": "IfcSlab", "name": "S1"}


def scn_painted_rect():
    """A textured, tagged rectangle: the material extrudes with the shape."""
    scene = Scene()
    hist = History(scene)
    rect(scene, hist, 0.0, 0.0, 0.0, 4.0, 3.0)
    f = scene.mesh.faces[0]
    f.attrs.update({"color": [200, 30, 30, 255], "texture": dict(TEXTURE), "ifc": dict(IFC)})
    return scene


def scn_painted_box():
    """A box whose top is painted and tagged, with a painted inner rectangle."""
    scene, hist = cube(size=6.0)
    top = face_where(scene, on_z(3.0))
    top.attrs.update({"color": [10, 120, 200, 255], "ifc": dict(IFC)})
    rect(scene, hist, 1.0, 1.0, 3.0, 2.0, 2.0)
    inner = face_where(scene, both(on_z(3.0), area_near(4.0)))
    inner.attrs.update({"color": [250, 200, 0, 255], "mat": "oak", "texture": dict(TEXTURE), "opacity": 0.8})
    side = face_where(scene, on_x(6.0))
    side.attrs.update({"color": [90, 90, 90, 255]})
    return scene


def scn_painted_disc():
    scene = Scene()
    hist = History(scene)
    draw_circle(scene, hist, circle_pts(1.5, 16, 0.0, 0.0, 0.0))
    scene.mesh.faces[0].attrs.update({"color": [30, 200, 30, 255], "texture": dict(TEXTURE)})
    return scene


def scn_arc_rect():
    """A rectangle with two rounded corners: two open curves of three
    segments, the rest straight."""
    scene = Scene()
    hist = History(scene)
    pts = [V(0, 0), V(3, 0), V(3.8, 0.2), V(4.2, 0.8), V(4.0, 1.0), V(4, 3), V(0.5, 3), V(0.15, 2.85),
           V(0, 2.5)]
    n = len(pts)
    segs = [(pts[i], pts[(i + 1) % n]) for i in range(n)]
    hist.execute(build_add_edges(scene, segs, detect_faces=False, extra=[
        AddFaceCommand(list(pts)),
        TagCurveCommand([pts[1], pts[2], pts[3], pts[4]], closed=False),
        TagCurveCommand([pts[6], pts[7], pts[8]], closed=False)]))
    return scene


def scn_arc_rect_on_box():
    scene, hist = cube(size=6.0)
    z = 3.0
    pts = [V(1, 1, z), V(3, 1, z), V(3.6, 1.2, z), V(3.9, 1.8, z), V(4.0, 2.0, z), V(4, 3.5, z), V(1, 3.5, z)]
    n = len(pts)
    segs = [(pts[i], pts[(i + 1) % n]) for i in range(n)]
    hist.execute(build_add_edges(scene, segs, detect_faces=False, extra=[
        AddFaceCommand(list(pts)),
        TagCurveCommand([pts[1], pts[2], pts[3], pts[4]], closed=False)]))
    return scene


def scn_two_holes():
    scene = Scene()
    outer = [V(0, 0, 0), V(6, 0, 0), V(6, 4, 0), V(0, 4, 0)]
    h1 = [V(1, 1, 0), V(1, 3, 0), V(2.5, 3, 0), V(2.5, 1, 0)]
    h2 = [V(3.5, 1, 0), V(3.5, 3, 0), V(5, 3, 0), V(5, 1, 0)]
    scene.mesh.add_face(outer, [h1, h2])
    return scene


def scn_instance_cube():
    """A cube definition placed twice (identity and shifted 3 along x)."""
    scene = Scene()
    proto = Mesh()
    pts = [V(0, 0, 0), V(1, 0, 0), V(1, 1, 0), V(0, 1, 0), V(0, 0, 1), V(1, 0, 1), V(1, 1, 1), V(0, 1, 1)]
    quads = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
    for q in quads:
        proto.add_face([pts[i] for i in q])
    a, b = Group(proto, name="Pilar"), Group(proto, name="Pilar")
    xb = QMatrix4x4()
    xb.translate(3.0, 0.0, 0.0)
    a.xform, b.xform = QMatrix4x4(), xb
    scene.groups += [a, b]
    return scene


SCENES = {
    "painted_rect": scn_painted_rect,
    "painted_box": scn_painted_box,
    "painted_disc": scn_painted_disc,
    "arc_rect": scn_arc_rect,
    "arc_rect_on_box": scn_arc_rect_on_box,
    "two_holes": scn_two_holes,
    "instance_cube": scn_instance_cube,
    "cube": scn_cube,
    "cube_top_rect": scn_cube_top_rect,
    "cube_top_rect_big": scn_cube_top_rect_big,
    "cube_side_rect": scn_cube_side_rect,
    "cube_side_rect_front": scn_cube_side_rect_front,
    "cube_corner_rect": scn_cube_corner_rect,
    "cube_two_floors": scn_cube_two_floors,
    "ring": scn_ring,
    "ring_on_box": scn_ring_on_box,
    "disc": scn_disc,
    "disc_on_box": scn_disc_on_box,
    "hexagon": scn_hexagon,
    "gable": scn_gable,
    "wedge": scn_wedge,
    "tilted_sheet": scn_tilted_sheet,
    "step_square": scn_step_square,
    "plan": scn_plan,
    "sheet_beside_box": scn_sheet_beside_box,
    "open_cube_and_rect": scn_open_cube_and_rect,
    "triangle_prism": scn_triangle_prism,
    "group_cube": scn_group_cube,
    "stairs": scn_stairs,
}

# ---- the programs ---------------------------------------------------------------------
# A step: (kind, face selector or None, distance, copy flag). The selector is
# a predicate over the live scene's faces, ``(pred, nth)`` picks the nth hit.

def S(kind, sel=None, d=None, copy=False, nth=0, group=None):
    return {"kind": kind, "sel": sel, "d": d, "copy": copy, "nth": nth, "group": group}


def top(z=3.0):
    return on_z(z)


PROGRAMS = []


def prog(scene, *steps):
    PROGRAMS.append((scene, list(steps)))


# a box from a rectangle: pull, push back, collapse
prog("cube", S("value", on_z(3.0), 2.0), S("value", on_z(5.0), -1.0), S("value", on_z(4.0), 1.0),
     S("undo"), S("undo"), S("redo"), S("redo"))
prog("cube", S("value", on_z(3.0), -1.0), S("value", on_z(2.0), -2.0))      # shrink; collapse flat
prog("cube", S("value", on_z(3.0), -3.0))                                      # push to zero
prog("cube", S("value", on_z(3.0), -10.0))                                     # beyond: the inward limit
prog("cube", S("value", on_z(3.0), 1e-5), S("value", on_z(3.0), 0.0001), S("value", on_z(3.0), 0.0))
prog("cube", S("click2", on_z(3.0), 1.5), S("click2", on_z(4.5), -0.5))
prog("cube", S("click2", on_z(3.0), 50.0))
prog("cube", S("op", on_z(3.0), 2.0), S("op", on_z(5.0), -4.0), S("op", on_z(1.0), 1.0))
prog("cube", S("value", on_z(3.0), 2.0, copy=True), S("value", on_z(5.0), 1.0, copy=True),
     S("undo"), S("undo"), S("redo"))
prog("cube", S("value", on_z(3.0), -1.0, copy=True), S("value", on_z(2.0), -1.0, copy=True))
prog("cube", S("value", on_z(3.0), 2.0, copy=True), S("value", on_z(0.0), -1.0, copy=True))
prog("cube", S("value", on_z(0.0), 1.0), S("value", on_z(-1.0), -2.0))          # the bottom, both ways
prog("cube", S("value", on_x(4.0), 1.0), S("value", on_x(5.0), -2.0))          # a side wall out and in
prog("cube", S("value", on_x(4.0), -1.0), S("value", on_x(3.0), -3.0))
prog("cube", S("value", on_x(4.0), -5.0))
prog("cube", S("value", on_y(0.0), 2.0, copy=True), S("value", on_x(0.0), 0.5))
prog("cube", S("op", on_x(4.0), -2.0, copy=True), S("op", on_y(4.0), 1.0, copy=True))
prog("cube", S("value", on_z(3.0), 2.0), S("retype", None, 3.0), S("retype", None, -1.0),
     S("retype", None, 0.5), S("undo"))
prog("cube", S("value", on_z(3.0), 2.0), S("double", on_z(5.0)), S("double", on_x(4.0)),
     S("double", on_z(0.0), copy=True))
prog("cube", S("value", on_z(3.0), 1.0, copy=True), S("retype", None, 2.0))
prog("cube", S("double", on_z(3.0)), S("value", on_z(3.0), 0.7), S("double", on_x(4.0)))

# a recess, a through-hole, a bump on an existing solid
prog("cube_top_rect", S("value", both(on_z(3.0), area_near(1.0)), -1.0),
     S("value", both(on_z(2.0), area_near(1.0)), -1.0),
     S("value", both(on_z(1.0), area_near(1.0)), 3.0))
prog("cube_top_rect", S("value", both(on_z(3.0), area_near(1.0)), -3.0))
prog("cube_top_rect", S("value", both(on_z(3.0), area_near(1.0)), -4.0))
prog("cube_top_rect", S("value", both(on_z(3.0), area_near(1.0)), -10.0))
prog("cube_top_rect", S("value", both(on_z(3.0), area_near(1.0)), 2.0), S("value", both(on_z(5.0), area_near(1.0)), -2.0))
prog("cube_top_rect", S("value", both(on_z(3.0), area_near(1.0)), 1.5), S("value", both(on_z(4.5), area_near(1.0)), -4.5))
prog("cube_top_rect", S("value", both(on_z(3.0), area_near(1.0)), -2.0, copy=True),
     S("value", both(on_z(1.0), area_near(1.0)), -1.0, copy=True))
prog("cube_top_rect", S("value", both(on_z(3.0), area_near(1.0)), -3.0, copy=True))
prog("cube_top_rect", S("value", both(on_z(3.0), area_near(1.0)), 1.0, copy=True),
     S("value", both(on_z(4.0), area_near(1.0)), 1.0, copy=True))
prog("cube_top_rect", S("value", both(on_z(3.0), area_near(1.0)), -0.5), S("double", both(on_z(2.5), area_near(1.0))))
prog("cube_top_rect", S("op", both(on_z(3.0), area_near(1.0)), -1.5), S("op", both(on_z(1.5), area_near(1.0)), -1.5),
     S("op", both(on_z(0.0), area_near(1.0)), 0.5))
prog("cube_top_rect", S("value", both(on_z(3.0), lambda f: bool(f.holes)), 1.0), S("value", both(on_z(3.0), area_near(1.0)), 1.0))
prog("cube_top_rect_big", S("value", both(on_z(3.0), area_near(5.0)), -3.0),
     S("value", both(on_z(3.0), area_near(1.0)), -1.0))
prog("cube_top_rect_big", S("value", both(on_z(3.0), area_near(5.0)), -1.0), S("value", both(on_z(2.0), area_near(5.0)), -2.0))
prog("cube_top_rect_big", S("value", both(on_z(3.0), area_near(5.0)), 1.0), S("value", both(on_z(3.0), area_near(1.0)), 2.0))
prog("cube_top_rect_big", S("value", both(on_z(3.0), area_near(5.0)), 2.0, copy=True),
     S("value", both(on_z(3.0), area_near(1.0)), -2.0, copy=True))
prog("cube_side_rect", S("value", both(on_x(4.0), area_near(1.5)), 1.0), S("value", both(on_x(5.0), area_near(1.5)), -2.0))
prog("cube_side_rect", S("value", both(on_x(4.0), area_near(1.5)), -1.0), S("value", both(on_x(3.0), area_near(1.5)), -3.0))
prog("cube_side_rect", S("value", both(on_x(4.0), area_near(1.5)), -4.0))
prog("cube_side_rect", S("value", both(on_x(4.0), area_near(1.5)), -4.0, copy=True))
prog("cube_side_rect", S("click2", both(on_x(4.0), area_near(1.5)), 2.0), S("click2", both(on_x(6.0), area_near(1.5)), -6.0))
prog("cube_side_rect_front", S("value", both(on_y(0.0), area_near(2.25)), -1.0), S("value", both(on_y(1.0), area_near(2.25)), -3.0))
prog("cube_side_rect_front", S("value", both(on_y(0.0), area_near(2.25)), 1.0), S("value", both(on_y(-1.0), area_near(2.25)), -1.0))
prog("cube_corner_rect", S("value", both(on_z(3.0), area_near(1.0)), -1.0), S("value", both(on_z(2.0), area_near(1.0)), -2.5))
prog("cube_corner_rect", S("value", both(on_z(3.0), area_near(1.0)), 1.0), S("value", both(on_z(4.0), area_near(1.0)), -1.0))
prog("cube_corner_rect", S("value", both(on_z(3.0), area_near(1.0)), -3.0))
prog("cube_corner_rect", S("value", both(on_z(3.0), area_near(1.0)), -3.0, copy=True), S("undo"), S("redo"))
prog("cube_two_floors", S("value", both(on_z(5.0), area_near(8.0)), -3.0), S("value", both(on_z(3.0), area_near(16.0)), 1.0))
prog("cube_two_floors", S("value", both(on_z(5.0), area_near(8.0)), 1.0), S("value", both(on_z(6.0), area_near(8.0)), -4.0))
prog("cube_two_floors", S("value", both(on_z(3.0), area_near(8.0), lambda f: f.centroid().x() > 2.5), 1.0),
     S("value", both(on_z(4.0), area_near(8.0), lambda f: f.centroid().x() > 2.5), -4.0))
prog("cube_two_floors", S("value", both(on_x(2.0), area_near(8.0)), 1.0), S("value", both(on_x(3.0), area_near(8.0)), -2.0))
prog("cube_two_floors", S("value", both(on_x(2.0), area_near(8.0)), -1.0), S("value", both(on_x(1.0), area_near(8.0)), -4.0))
prog("cube_two_floors", S("value", both(on_z(5.0), area_near(8.0)), 1.0, copy=True), S("value", both(on_z(3.0), area_near(8.0), lambda f: f.centroid().x() > 2.5), 1.0, copy=True))

# faces with holes
prog("ring", S("value", lambda f: bool(f.holes), 2.0), S("value", on_z(2.0), -1.0))
prog("ring", S("value", lambda f: bool(f.holes), 2.0, copy=True))
prog("ring", S("value", lambda f: bool(f.holes), -2.0), S("undo"), S("redo"))
prog("ring", S("value", lambda f: bool(f.holes), 2.0), S("value", both(on_z(2.0), lambda f: bool(f.holes)), 1.0),
     S("value", both(on_z(3.0), lambda f: bool(f.holes)), -4.0))
prog("ring", S("value", lambda f: bool(f.holes), 1.0, copy=True), S("value", on_z(1.0), 1.0))
prog("ring_on_box", S("value", both(on_z(3.0), lambda f: bool(f.holes)), -1.0),
     S("value", both(on_z(3.0), lambda f: bool(f.holes)), -1.0))
prog("ring_on_box", S("value", both(on_z(3.0), lambda f: bool(f.holes)), 1.0),
     S("value", both(on_z(3.0), area_near(4.0)), 2.0))
prog("ring_on_box", S("value", both(on_z(3.0), lambda f: bool(f.holes)), -3.0),
     S("value", both(on_z(3.0), area_near(4.0)), -3.0))
prog("ring_on_box", S("value", both(on_z(3.0), area_near(4.0)), -1.0),
     S("value", both(on_z(3.0), lambda f: bool(f.holes)), -1.0))
prog("ring_on_box", S("value", both(on_z(3.0), lambda f: bool(f.holes)), 1.0, copy=True))

# curved (soft) faces
prog("disc", S("value", None, 2.0), S("value", on_z(2.0), -1.0), S("value", on_z(1.0), 3.0),
     S("undo"), S("undo"), S("redo"))
prog("disc", S("value", None, -2.0, copy=True), S("value", on_z(-2.0), 1.0))
prog("disc", S("value", None, 2.0), S("value", on_z(2.0), -2.0))
prog("hexagon", S("value", None, 2.0), S("value", on_z(2.0), -1.0))
prog("disc_on_box", S("value", both(on_z(3.0), lambda f: len(f.vertices) == 24), -1.0),
     S("value", both(on_z(2.0), lambda f: len(f.vertices) == 24), -1.0))
prog("disc_on_box", S("value", both(on_z(3.0), lambda f: len(f.vertices) == 24), -3.0))
prog("disc_on_box", S("value", both(on_z(3.0), lambda f: len(f.vertices) == 24), 2.0),
     S("value", both(on_z(5.0), lambda f: len(f.vertices) == 24), -2.0))
prog("disc_on_box", S("value", both(on_z(3.0), lambda f: len(f.vertices) == 24), 2.0, copy=True))

# tilted faces
prog("gable", S("value", lambda f: f.normal().z() > 0.5 and f.normal().x() > 0.1, 0.3),
     S("value", lambda f: f.normal().z() > 0.5 and f.normal().x() > 0.1, -0.3))
prog("gable", S("value", lambda f: len(f.vertices) == 5 and f.normal().y() < -0.5, 1.0))
prog("gable", S("value", lambda f: len(f.vertices) == 5 and f.normal().y() < -0.5, -1.0),
     S("value", lambda f: len(f.vertices) == 5 and f.normal().y() > 0.5, 0.5))
prog("gable", S("value", lambda f: f.normal().z() > 0.5 and f.normal().x() < -0.1, -0.5))
prog("gable", S("value", lambda f: f.normal().z() > 0.5 and f.normal().x() > 0.1, 0.4, copy=True))
prog("wedge", S("value", lambda f: abs(f.normal().z()) > 0.3 and f.centroid().z() > 1.0, 0.7),
     S("value", lambda f: abs(f.normal().x()) > 0.9 and f.centroid().x() > 3.0, 1.1))
prog("wedge", S("value", lambda f: abs(f.normal().z()) > 0.3 and f.centroid().z() > 1.0, -0.4))
prog("wedge", S("value", lambda f: abs(f.normal().y()) > 0.9 and f.centroid().y() < 1.0, 0.6),
     S("value", lambda f: abs(f.normal().z()) > 0.3 and f.centroid().z() > 1.0, 0.5))
prog("wedge", S("value", lambda f: abs(f.normal().x()) > 0.9 and f.centroid().x() > 3.0, -0.4),
     S("value", lambda f: abs(f.normal().y()) > 0.9 and f.centroid().y() > 3.0, 0.5))
prog("tilted_sheet", S("value", None, 1.0), S("value", None, 0.5))
prog("tilted_sheet", S("value", None, -1.0, copy=True), S("undo"))
prog("triangle_prism", S("value", on_z(3.0), 1.0), S("value", on_z(4.0), -2.0))
prog("triangle_prism", S("value", lambda f: abs(f.normal().z()) < 0.1 and len(f.vertices) == 4, -0.5),
     S("value", lambda f: abs(f.normal().z()) < 0.1 and len(f.vertices) == 4, 0.5, nth=1))
prog("triangle_prism", S("value", on_z(0.0), 1.0), S("value", on_z(-1.0), -4.0))

# against a wall, steps, stairs
for dist in (0.5, 1.0, 1.5, -0.5):
    prog("step_square", S("value", both(on_z(2.0), lambda f: len(f.vertices) == 4 and abs(f.area() - 1.0) < 1e-6), dist))
prog("step_square", S("value", both(on_z(2.0), lambda f: len(f.vertices) == 4 and abs(f.area() - 1.0) < 1e-6), 0.5),
     S("undo"), S("value", both(on_z(2.0), lambda f: len(f.vertices) == 4 and abs(f.area() - 1.0) < 1e-6), 1.0))
prog("step_square", S("value", both(on_z(2.0), lambda f: len(f.vertices) == 4 and abs(f.area() - 1.0) < 1e-6), 1.0, copy=True))
def tread(z):
    return lambda f: abs(f.centroid().z() - z) < 1e-6 and f.normal().z() > 0.9


def riser(y):
    return lambda f: abs(f.centroid().y() - y) < 1e-6 and f.normal().y() > 0.9


prog("stairs", S("value", tread(3.0), 1.0), S("value", tread(4.0), -2.0))
prog("stairs", S("value", tread(3.0), -3.0))
prog("stairs", S("value", tread(2.0), 1.0), S("value", tread(3.0), 0.5))
prog("stairs", S("value", tread(1.0), -1.0), S("value", tread(2.0), -0.5))
prog("stairs", S("value", riser(3.0), -1.0), S("value", riser(2.0), 2.0))
prog("stairs", S("value", riser(1.0), 1.0), S("value", tread(3.0), 1.0, copy=True))
prog("stairs", S("value", tread(2.0), 2.0, copy=True), S("value", tread(1.0), -0.5, copy=True))
#prog("stairs", S("value", both(on_z(1.0), area_near(4.0)), 2.0), S("value", both(on_z(2.0), area_near(8.0)), 1.0))
#prog("stairs", S("value", both(on_z(2.0), area_near(4.0)), -2.0))
#prog("stairs", S("value", both(on_y(0.0), area_near(4.0)), -1.0), S("value", both(on_y(0.0), area_near(4.0)), 1.0))

# flat drawings and sheets
prog("plan", S("value", lambda f: f.centroid().x() < 3.0, 2.7), S("value", lambda f: f.centroid().x() > 3.0, 2.7))
prog("plan", S("value", lambda f: f.centroid().x() < 3.0, 2.7, copy=True))
prog("plan", S("value", lambda f: f.centroid().x() < 3.0, -2.7), S("value", lambda f: f.centroid().x() > 3.0, 1.5))
prog("plan", S("value", lambda f: f.centroid().x() < 3.0, 1.0), S("value", lambda f: f.centroid().x() < 3.0 and f.centroid().z() > 0.5, 1.0))
prog("sheet_beside_box", S("value", lambda f: on_z(0.0)(f) and f.centroid().x() > 5.0 and f.centroid().y() < 2.5, 1.5),
     S("value", lambda f: on_z(0.0)(f) and f.centroid().x() > 5.0 and f.centroid().y() > 2.5, 1.0),
     S("value", on_z(2.0), 1.0))
prog("sheet_beside_box", S("value", lambda f: on_z(0.0)(f) and f.centroid().x() > 5.0, 1.5, copy=True))
prog("open_cube_and_rect", S("value", lambda f: on_z(0.0)(f) and f.centroid().x() > 2.0, 1.0),
     S("value", lambda f: on_z(0.0)(f) and f.centroid().x() < 2.0, 0.5))

# inside an open group: the group's mesh is the active one
prog("group_cube", S("edit", None, None, group=0), S("value", both(on_z(3.0), area_near(1.0)), -1.0),
     S("value", both(on_z(3.0), lambda f: bool(f.holes)), 1.0), S("undo"), S("end_edit"))
prog("group_cube", S("edit", None, None, group=0), S("value", both(on_z(3.0), area_near(1.0)), -3.0),
     S("end_edit"), S("value", lambda f: f.centroid().x() > 9.0, 2.0))
prog("group_cube", S("edit", None, None, group=0), S("value", on_z(3.0), 2.0, copy=True), S("value", on_x(4.0), 1.0),
     S("undo"), S("redo"), S("end_edit"))


# paint, BIM tags and curves ride along with the extrusion
prog("painted_rect", S("value", None, 2.0), S("value", on_z(2.0), -1.0), S("undo"), S("redo"))
prog("painted_rect", S("value", None, 2.0, copy=True))
prog("painted_rect", S("value", None, -2.0), S("value", on_z(-2.0), 1.0))
prog("painted_rect", S("value", None, 1.0), S("double", on_x(4.0)), S("double", on_y(0.0), copy=True))
prog("painted_box", S("value", both(on_z(3.0), area_near(4.0)), -1.0), S("value", both(on_z(2.0), area_near(4.0)), -1.0))
prog("painted_box", S("value", both(on_z(3.0), area_near(4.0)), 1.0), S("value", both(on_z(4.0), area_near(4.0)), -1.0))
prog("painted_box", S("value", both(on_z(3.0), area_near(4.0)), -3.0))
prog("painted_box", S("value", both(on_z(3.0), area_near(4.0)), -1.0, copy=True), S("value", on_x(6.0), 1.0))
prog("painted_box", S("value", on_x(6.0), 1.0), S("value", on_x(7.0), -2.0))
prog("painted_box", S("value", lambda f: on_z(3.0)(f) and bool(f.holes), 1.0))
prog("painted_disc", S("value", None, 2.0), S("value", on_z(2.0), -2.0))
prog("painted_disc", S("value", None, -1.5, copy=True))
prog("arc_rect", S("value", None, 2.0), S("value", on_z(2.0), -1.0))
prog("arc_rect", S("value", None, 1.5, copy=True))
prog("arc_rect", S("value", None, -1.0), S("undo"), S("value", None, 3.0))
prog("arc_rect_on_box", S("value", both(on_z(3.0), lambda f: len(f.vertices) == 7), -1.0))
prog("arc_rect_on_box", S("value", both(on_z(3.0), lambda f: len(f.vertices) == 7), -3.0))
prog("arc_rect_on_box", S("value", both(on_z(3.0), lambda f: len(f.vertices) == 7), 1.5))
prog("arc_rect_on_box", S("value", both(on_z(3.0), lambda f: len(f.vertices) == 7), -1.5, copy=True))
prog("two_holes", S("value", lambda f: len(f.holes) == 2, 2.0), S("value", lambda f: len(f.holes) == 2, 1.0))
prog("two_holes", S("value", lambda f: len(f.holes) == 2, 2.0, copy=True))
prog("two_holes", S("value", lambda f: len(f.holes) == 2, -2.0), S("undo"), S("redo"))
# a face of a component instance, pushed in its world copy
prog("instance_cube", S("edit", None, None, group=1), S("value", both(on_z(1.0)), 2.0),
     S("value", on_z(3.0), -1.0), S("undo"))
prog("instance_cube", S("edit", None, None, group=1), S("value", on_x(4.0), 1.0, copy=True))

# the tool's small behaviours: refusals, hints, the Ctrl tap, preselection, Esc, undo mid-drag
prog("cube", S("closed", on_z(3.0)), S("value", on_z(3.0), 1.0))
prog("cube", S("ctrl_tap"), S("value", on_z(3.0), 2.0), S("value", on_z(5.0), -1.0))
prog("cube", S("ctrl_tap"), S("ctrl_tap"), S("value", on_z(3.0), 2.0))
prog("cube", S("ctrl_tap"), S("esc", on_z(3.0), 1.0), S("value", on_z(3.0), 1.0))
prog("cube", S("tiny", on_z(3.0)), S("value", on_z(3.0), 1.0))
prog("cube", S("undo_mid", on_z(3.0)))
prog("cube", S("esc", on_z(3.0), 1.0), S("esc", on_z(3.0), 1.0, copy=True), S("value", on_z(3.0), 1.0))
prog("cube_top_rect", S("activate", both(on_z(3.0), area_near(1.0)), -1.0))
prog("cube", S("activate", on_z(3.0), 2.0, copy=True), S("activate", on_x(4.0), -1.0))
prog("cube", S("value", on_z(3.0), 1.0), S("double", on_z(4.0)), S("closed", on_z(5.0)), S("double", on_z(5.0)))

# random pushes on random faces of the scenes above (seeded)
RANDOM_SCENES = ["cube", "cube_top_rect", "cube_top_rect_big", "ring_on_box", "disc_on_box", "gable",
                 "wedge", "plan", "triangle_prism", "cube_two_floors", "cube_corner_rect",
                 "step_square", "sheet_beside_box", "cube_side_rect", "painted_box", "arc_rect_on_box",
                 "two_holes", "hexagon"]
for _i, _name in enumerate(RANDOM_SCENES):
    for _k in range(4):
        prog(_name, *[S("rand", d=1000 * (_i + 1) + 10 * _k + j) for j in range(5)])


# ---- the stub viewport and the runner ------------------------------------------------------------

class StubViewport:
    """tests/test_pushpull_ux._StubViewport (no camera: the drag is typed)."""

    last_status = None

    def __init__(self, scene, pick=None) -> None:
        self.scene = scene
        self.history = History(scene)
        self._pick = pick
        self.last_hover = None
        self.suppressed = None

    def update(self): pass

    def flash_status(self, text, msec=2500):
        self.last_status = text

    def set_hover(self, entity):
        self.last_hover = entity

    def set_suppressed_faces(self, faces):
        self.suppressed = set(faces)

    def pick_face(self, x, y):
        return self._pick

    _pick_group = None

    def pick_face_any(self, x, y):
        return self._pick, self._pick_group


def ctx(vp, copy=False):
    return ToolContext(viewport=vp, world=V(0, 0, 0), screen=QPointF(0, 0),
                       modifiers=Qt.ControlModifier if copy else Qt.NoModifier, snap=None)


def run_program(path: Path, steps: list) -> dict:
    ORD.clear()
    KEEP.clear()
    scene = reopen(path)
    register_all(scene.mesh)
    for g in scene.groups:
        register_all(g.mesh)
    cm._CURVE_COUNTER = 5000
    tp.PushPullTool.last_distance = None
    vp = StubViewport(scene)
    tool = tp.PushPullTool()
    rec = {"pre": snap(scene), "steps": []}
    depth_marks = []
    for st in steps:
        kind = st["kind"]
        vp.last_status = None
        forced = None
        if kind == "rand":
            rnd = random.Random(st["d"])
            faces = scene.mesh.faces
            forced = faces[rnd.randrange(len(faces))]
            st = dict(st, kind="value", copy=rnd.random() < 0.2,
                      d=rnd.choice([0.5, 1.0, 1.5, 2.0, 3.0, 0.25, -0.5, -1.0, -2.0, -3.0, -0.25, 4.0, -6.0]))
            kind = "value"
        out = {"kind": kind}
        if kind in ("value", "click2", "double", "op", "closed", "tiny", "undo_mid", "esc", "activate"):
            pred = st["sel"]
            if forced is not None:
                face = forced
            elif pred is None:
                face = scene.mesh.faces[0]
            else:
                hits = [f for f in scene.mesh.faces if pred(f)]
                if len(hits) <= st["nth"]:
                    print("NO FACE", path.name, len(rec["steps"]), st["d"], file=sys.stderr)
                    for i, f in enumerate(scene.mesh.faces):
                        c = f.centroid()
                        print("  ", i, len(f.vertices), len(f.holes), round(f.area(), 3),
                              (round(c.x(), 3), round(c.y(), 3), round(c.z(), 3)), file=sys.stderr)
                    raise LookupError(path.name)
                face = hits[st["nth"]]
            out["face"] = scene.mesh.faces.index(face)
            out["d"] = st["d"]
            out["copy"] = st["copy"]
            if kind == "value":
                tool.hovered_face = face
                tool._hover_group = None
                tool.on_click(ctx(vp, st["copy"]))
                out["dragging"] = tool.dragging
                ret = tool.on_value(vp, st["d"])
                out["ret"] = ret
            elif kind == "click2":
                tool.hovered_face = face
                tool._hover_group = None
                tool.on_click(ctx(vp, st["copy"]))
                tool.extrusion = st["d"]
                tool.on_click(ctx(vp, st["copy"]))
            elif kind == "double":
                vp._pick = face
                tool.on_double_click(ctx(vp, st["copy"]))
            elif kind == "op":
                out["ret"] = op_push(scene, vp.history, face, st["d"], keep_base=st["copy"])
            elif kind == "closed":
                # a face inside a group that is not open: shaded not, refused
                vp._pick, vp._pick_group = face, object()
                tool.on_hover(ctx(vp))
                out["hover0"] = scene.mesh.faces.index(vp.last_hover) if vp.last_hover is not None else None
                tool.on_click(ctx(vp))
                saved = tp.PushPullTool.last_distance
                tp.PushPullTool.last_distance = 1.0
                tool.on_double_click(ctx(vp))
                tp.PushPullTool.last_distance = saved
                vp._pick, vp._pick_group = None, None
            elif kind == "tiny":
                tool.hovered_face = face
                tool._hover_group = None
                tool.on_click(ctx(vp))
                tool.on_click(ctx(vp))          # nothing dragged: the hint
                tool.on_click(ctx(vp))          # said once per drag
                tool.on_release(vp)             # no press far enough away
                out["ret"] = tool.on_value(vp, 0.0)
            elif kind == "undo_mid":
                tool.hovered_face = face
                tool._hover_group = None
                tool.on_click(ctx(vp))
                out["ret"] = tool.on_undo(vp)
                out["ret2"] = tool.on_redo(vp)
            elif kind == "esc":
                tool.hovered_face = face
                tool._hover_group = None
                tool.on_click(ctx(vp, st["copy"]))
                tool.extrusion = st["d"]
                out["drag_mid"] = tool.dragging
                tool.on_cancel(vp)
            elif kind == "activate":
                # activated with exactly one face selected: the tool arms on it
                scene.selection = {face}
                tool.on_activate(vp)
                out["armed"] = vp.last_hover is face
                tool.on_hover(ctx(vp))          # the pick (nothing) does not steal it
                out["still_armed"] = vp.last_hover is face
                tool.on_click(ctx(vp, st["copy"]))
                out["armed_drag"] = tool.dragging
                out["ret"] = tool.on_value(vp, st["d"])
            out["drag_end"] = tool.dragging
            if tool.dragging:
                tool.on_cancel(vp)
        elif kind == "retype":
            out["d"] = st["d"]
            out["ret"] = tool.on_value(vp, st["d"])
            if tool.dragging:
                tool.on_cancel(vp)
        elif kind == "ctrl_tap":
            out["pressed"] = tool.on_key(vp, Qt.Key_Control, Qt.ControlModifier)
            out["ret"] = tool.on_key_release(vp, Qt.Key_Control)
        elif kind == "undo":
            out["ret"] = vp.history.undo()
        elif kind == "redo":
            out["ret"] = vp.history.redo()
        elif kind == "edit":
            scene.begin_group_edit(scene.groups[st["group"]])
            out["group"] = st["group"]
            tool.on_cancel(vp) if tool.dragging else None
        elif kind == "end_edit":
            scene.end_group_edit()
        else:
            raise ValueError(kind)
        out["state"] = snap(scene)
        out["status"] = vp.last_status
        out["keep_mode"] = tool._keep_mode
        out["hover"] = scene.mesh.faces.index(vp.last_hover) if vp.last_hover in scene.mesh.faces else None
        out["last_distance"] = tp.PushPullTool.last_distance
        out["undo_depth"] = len(vp.history.undo_stack)
        out["last_error"] = vp.history.last_error
        rec["steps"].append(out)
    rec["undone"] = []
    while len(vp.history.undo_stack) > 0:
        # an edit session ends before its mesh is undone against
        vp.history.undo()
        rec["undone"].append(snap(scene))
    rec["redone"] = []
    while vp.history.redo_stack:
        vp.history.redo()
        rec["redone"].append(snap(scene))
    return rec


# ---- Tier B: the pointer, through a real camera ---------------------------------------------
# Hover, press, drag, release and double-click with the cursor in screen
# pixels. The viewport is a stand-in with a real ``core.camera.Camera`` and the
# viewport's own ray maths (``Viewport._pixel_to_ray``, ``_world_to_pixel``,
# ``_ray_hit_plane``, ``_project_to_lock_line``, ``_coplanar_eps`` bound to
# it); picking is the plain ray cast below (the viewport's cached index is
# the host's business). The pixels are computed here from world points and
# recorded, so Rust is fed the very same doubles.

import types  # noqa: E402

from core.camera import OrbitCamera as Camera  # noqa: E402
from core.guide import Guide  # noqa: E402
from core.orient import _ray_triangle  # noqa: E402
from views.viewport import Viewport  # noqa: E402


class HoverEdge:
    def __init__(self, a, b) -> None:
        self.a, self.b = a, b


class PointerViewport(StubViewport):
    snap_threshold_px = 9.0
    _hover_edge = None

    def __init__(self, scene, cam, w, h) -> None:
        super().__init__(scene)
        self.camera, self._w, self._h = cam, w, h
        for name in ("_pixel_to_ray", "_ray_hit_plane", "_project_to_lock_line",
                     "_world_to_pixel", "_coplanar_eps"):
            setattr(self, name, types.MethodType(getattr(Viewport, name), self))

    def width(self):
        return self._w

    def height(self):
        return self._h

    def pick_face_any(self, x, y):
        origin, direction = self._pixel_to_ray(x, y)
        if origin is None or direction is None:
            return None, None
        hits = []
        for f in self.scene.mesh.faces:
            ts = [t for t in (_ray_triangle(origin, direction, tri) for tri in f.triangulate())
                  if t is not None]
            if ts:
                hits.append((f, min(ts)))
        if not hits:
            return None, None
        best_t = min(t for _, t in hits)
        eps = self._coplanar_eps(origin, direction, best_t)
        cand = [f for f, t in hits if t <= best_t + eps]
        if len(cand) == 1:
            return cand[0], None
        return min(cand, key=lambda f: f.area()), None


def scn_ground_rect():
    scene = Scene()
    hist = History(scene)
    rect(scene, hist, 0.0, 0.0, 0.0, 8.0, 5.0)
    return scene


def scn_ground_guides():
    scene = Scene()
    hist = History(scene)
    rect(scene, hist, 0.0, 0.0, 0.0, 4.0, 4.0)
    scene.guides.append(Guide(V(4, 4, 2.5)))
    scene.guides.append(Guide(V(0, 6, 3.0), V(1, 0, 0)))
    return scene


def scn_slab():
    scene = Scene()
    hist = History(scene)
    rect(scene, hist, 0.0, 0.0, 0.0, 12.0, 7.32)
    return scene


def scn_rect_box():
    scene, hist = cube(size=2.0, height=2.0, x0=6.0, y0=0.0)
    rect(scene, hist, 0.0, 0.0, 0.0, 4.0, 4.0)
    return scene


def scn_ring_free():
    scene = Scene()
    outer = [V(-2, -2, 0), V(2, -2, 0), V(2, 2, 0), V(-2, 2, 0)]
    hole = [V(-1.8, -1.8, 0), V(-1.8, 1.8, 0), V(1.8, 1.8, 0), V(1.8, -1.8, 0)]
    scene.mesh.add_face(outer, [hole])
    return scene


POINTER_SCENES = {"ground_rect": scn_ground_rect, "ground_guides": scn_ground_guides, "slab": scn_slab,
                  "rect_box": scn_rect_box, "ring_free": scn_ring_free}

CAM_GROUND = dict(target=(4, 2.5, 0), distance=18.0, yaw=-60.0, pitch=6.0, w=1600, h=900)
CAM_CUBE = dict(target=(2, 2, 1.5), distance=16.0, yaw=-50.0, pitch=25.0, w=1600, h=900)
CAM_SLAB = dict(target=(6, 3.66, 0), distance=22.0, yaw=-50.0, pitch=30.0, w=1600, h=900)
CAM_RING = dict(target=(0, 0, 0), distance=14.0, yaw=-50.0, pitch=30.0, w=1600, h=900)
CAM_RECT_BOX = dict(target=(4, 2, 1), distance=20.0, yaw=-55.0, pitch=28.0, w=1280, h=800)
CAM_ORTHO = dict(target=(2, 2, 1.5), distance=16.0, yaw=-50.0, pitch=25.0, w=1600, h=900, perspective=False)

POINTER_PROGRAMS = []


def pprog(scene, cam, *events):
    POINTER_PROGRAMS.append((scene, cam, list(events)))


def at(p, dx=0.0, dy=0.0):
    """A world point, plus a pixel offset."""
    return ("at", p, dx, dy)


def E(kind, where=None, ctrl=False, d=None, edge=None):
    return {"e": kind, "where": where, "ctrl": ctrl, "d": d, "edge": edge}


def above(p, dz):
    return (p[0], p[1], p[2] + dz)


C = (4, 2.5, 0)
for rise in (None,):
    pass
# a press-drag-release stroke on a free sheet (Rafael's ground-level camera)
pprog("ground_rect", CAM_GROUND, E("hover", C), E("click", C), *[E("hover", above(C, z)) for z in (0.5, 1.5, 2.8, 4.0)],
      E("release", above(C, 4.0)), E("undo"), E("redo"))
pprog("ground_rect", CAM_GROUND, E("hover", C), E("click", C), E("hover", above(C, 2.0)), E("click", above(C, 2.0)),
      E("hover", (1, 1, 0)), E("click", (1, 1, 0)), E("hover", (1, 1, 1.0)), E("release", (1, 1, 1.0)))
pprog("ground_rect", CAM_GROUND, E("hover", C), E("click", C, ctrl=True), E("hover", above(C, 1.5), ctrl=True),
      E("hover", above(C, 1.5)), E("click", above(C, 1.5), ctrl=True))
pprog("ground_rect", CAM_GROUND, E("hover", C), E("click", C), E("hover", above(C, 1.0)), E("value", d=2.5),
      E("value", d=3.0), E("value", d=-1.0))
pprog("ground_rect", CAM_GROUND, E("hover", C), E("click", C), E("hover", at(C, 2, 1)), E("release", at(C, 2, 1)),
      E("hover", above(C, 1.0)), E("cancel"), E("hover", C), E("click", C), E("hover", above(C, -1.0)), E("click", above(C, -1.0)))
pprog("ground_rect", CAM_GROUND, E("hover", C), E("click", C), E("hover", above(C, 2.0)), E("click", above(C, 2.0)),
      E("hover", (2, 1, 2)), E("double", (6, 3, 0)), E("double", (2, 1, 2)))
pprog("ground_rect", CAM_GROUND, E("hover", C), E("click", C), E("hover", above(C, 2.0)), E("click", above(C, 2.0)),
      E("value", d=1.0), E("value", d=-0.5), E("hover", C), E("value", d=1.0))
# measuring from the click, not the centroid
pprog("slab", CAM_SLAB, E("hover", (0.4, 0.4, 0)), E("click", (0.4, 0.4, 0)), E("hover", (0.4, 0.4, 0.3)),
      E("hover", (0.4, 0.4, 1.3)), E("click", (0.4, 0.4, 1.3)))
pprog("ring_free", CAM_RING, E("hover", (1.9, 0, 0)), E("click", (1.9, 0, 0)), E("hover", (1.9, 0, 1.0)),
      E("release", (1.9, 0, 1.0)))
# distance inference: a corner, an edge, a face, the origin, guides
RB = (6, 0, 2)
pprog("rect_box", CAM_RECT_BOX, E("hover", (2, 2, 0)), E("click", (2, 2, 0)), E("hover", at(RB, 3, 2)),
      E("hover", above((2, 2, 0), 1.0)), E("hover", at((8, 2, 2), 1, 1)), E("hover", at((7, 1, 2), 0, 0)),
      E("click", at(RB, 3, 2)))
pprog("rect_box", CAM_RECT_BOX, E("hover", (2, 2, 0)), E("click", (2, 2, 0)),
      E("hover", at((7, 0, 2), 0, 14), edge=((6, 0, 2), (8, 0, 2))),
      E("hover", at((7, 2, 2), 0, 12), edge=((6, 2, 2), (8, 2, 2))),
      E("hover", (7, 1, 2)), E("hover", (7, 0.5, 2)),
      E("release", (7, 0.5, 2)))
pprog("rect_box", CAM_RECT_BOX, E("hover", (2, 2, 0)), E("click", (2, 2, 0)), E("hover", (0, 0, 0)),
      E("hover", at((0, 0, 0), 5, 0)), E("hover", (6.5, 1, 1)), E("hover", (7.5, 0, 1)), E("click", (7.5, 0, 1)))
pprog("ground_guides", CAM_CUBE, E("hover", (2, 2, 0)), E("click", (2, 2, 0)), E("hover", at((4, 4, 2.5), 2, 1)),
      E("hover", at((2, 6, 3.0), 1, 2)), E("hover", above((2, 2, 0), 1.0)), E("click", at((4, 4, 2.5), 2, 1)))
# a prism cap: the cap itself travels in the model
TOP = (2, 2, 3)
pprog("cube", CAM_CUBE, E("hover", TOP), E("click", TOP), E("hover", above(TOP, 1.3)), E("hover", above(TOP, 2.0)),
      E("hover", above(TOP, -1.0)), E("hover", above(TOP, 0.7)), E("click", above(TOP, 0.7)), E("undo"), E("redo"))
pprog("cube", CAM_CUBE, E("hover", TOP), E("click", TOP), E("hover", above(TOP, 1.1)), E("hover", above(TOP, 1.9)),
      E("hover", above(TOP, 0.37)), E("value", d=1.25), E("value", d=2.0))
pprog("cube", CAM_CUBE, E("hover", TOP), E("click", TOP), E("hover", above(TOP, 1.1)), E("hover", above(TOP, 1.9), ctrl=True),
      E("hover", above(TOP, 1.6), ctrl=True), E("hover", above(TOP, 1.6)), E("click", above(TOP, 1.6)))
pprog("cube", CAM_CUBE, E("hover", TOP), E("click", TOP), E("hover", above(TOP, -2.0)), E("hover", above(TOP, -3.5)),
      E("hover", above(TOP, -2.5)), E("click", above(TOP, -2.5)))
pprog("cube", CAM_CUBE, E("hover", TOP), E("click", TOP), E("hover", above(TOP, 2.0)), E("release", above(TOP, 2.0)),
      E("hover", (4, 2, 1.5)), E("click", (4, 2, 1.5)), E("hover", (5.2, 2, 1.5)), E("release", (5.2, 2, 1.5)))
pprog("cube", CAM_CUBE, E("hover", TOP), E("click", TOP), E("hover", above(TOP, 1.0)), E("click", above(TOP, 1.0)),
      E("value", d=2.0), E("value", d=-1.0), E("hover", (4, 2, 1.5)), E("double", (4, 2, 1.5)))
pprog("cube", CAM_ORTHO, E("hover", TOP), E("click", TOP), E("hover", above(TOP, 1.6)), E("click", above(TOP, 1.6)))
pprog("cube", CAM_ORTHO, E("hover", (4, 2, 1.5)), E("click", (4, 2, 1.5)), E("hover", (5.5, 2, 1.5)),
      E("release", (5.5, 2, 1.5)))
pprog("disc", CAM_RING, E("hover", (0, 0, 0)), E("click", (0, 0, 0)), E("hover", (0, 0, 1.0)), E("hover", (0, 0, 2.2)),
      E("release", (0, 0, 2.2)))
pprog("disc", CAM_RING, E("hover", (0.5, 0.5, 0)), E("click", (0.5, 0.5, 0), ctrl=True), E("hover", (0.5, 0.5, 1.0)),
      E("click", (0.5, 0.5, 1.0)))
pprog("hexagon", CAM_RING, E("hover", (0, 0, 0)), E("click", (0, 0, 0)), E("hover", (0, 0, 1.5)), E("click", (0, 0, 1.5)))
pprog("painted_rect", CAM_GROUND, E("hover", C), E("click", C), E("hover", above(C, 1.2)), E("hover", above(C, 2.0)),
      E("click", above(C, 2.0)))
pprog("ring_on_box", CAM_RING, E("hover", (0, 0, 3)), E("hover", (0, 2.5, 3)), E("click", (0, 2.5, 3)))
# the inward limit, and a click that pushes nothing
pprog("cube", CAM_CUBE, E("hover", TOP), E("click", TOP), E("hover", above(TOP, -6.0)), E("hover", above(TOP, -4.0)),
      E("click", above(TOP, -4.0)))
pprog("cube", CAM_CUBE, E("hover", TOP), E("click", TOP), E("hover", TOP), E("click", TOP), E("click", TOP),
      E("release", TOP), E("cancel"))
pprog("cube", CAM_CUBE, E("hover", TOP), E("click", TOP), E("hover", above(TOP, 2.0)), E("value", d=1.0),
      E("value", d=2.5), E("undo"), E("value", d=2.5))


# staged previews: an attached face dragged in and out (recess, bump, through-hole)
IN = (1.5, 1.5, 3)
pprog("cube_top_rect", CAM_CUBE, E("hover", IN), E("click", IN), E("hover", above(IN, -0.6)), E("hover", above(IN, -1.7)),
      E("hover", above(IN, 2.0)), E("hover", above(IN, -2.0)), E("click", above(IN, -2.0)), E("undo"), E("redo"))
pprog("cube_top_rect", CAM_CUBE, E("hover", IN), E("click", IN), E("hover", above(IN, -1.0)), E("hover", above(IN, -2.2)),
      E("hover", above(IN, -2.9)), E("hover", above(IN, -3.0)), E("hover", above(IN, -3.4)), E("release", above(IN, -3.4)))
pprog("cube_top_rect", CAM_CUBE, E("hover", IN), E("click", IN), E("hover", above(IN, 1.2)), E("hover", above(IN, 2.5)),
      E("value", d=1.0), E("hover", IN), E("click", IN), E("hover", above(IN, -0.5)), E("value", d=-2.0))
pprog("cube_top_rect", CAM_CUBE, E("hover", IN), E("click", IN), E("hover", above(IN, -1.0)), E("hover", above(IN, -2.0), ctrl=True),
      E("hover", above(IN, -1.5), ctrl=True), E("hover", above(IN, -1.5)), E("hover", above(IN, -0.4)), E("click", above(IN, -0.4)))
pprog("cube_top_rect", CAM_CUBE, E("hover", IN), E("click", IN), E("hover", above(IN, -1.0)), E("hover", above(IN, 1.0)),
      E("hover", above(IN, -1.0)), E("hover", above(IN, 1.2)), E("hover", above(IN, -1.2)), E("click", above(IN, -1.2)))
pprog("cube_top_rect", CAM_CUBE, E("hover", IN), E("click", IN), E("hover", above(IN, -6.0)), E("hover", above(IN, -4.0)),
      E("click", above(IN, -4.0)))
pprog("cube_two_floors", CAM_CUBE, E("hover", (3, 2, 3)), E("click", (3, 2, 3)), E("hover", (3, 2, 4.0)), E("hover", (3, 2, 5.0)),
      E("hover", (3, 2, 6.2)), E("hover", (3, 2, 2.0)), E("hover", (3, 2, 1.0)), E("click", (3, 2, 1.0)))
pprog("cube_two_floors", CAM_CUBE, E("hover", (1, 2, 5)), E("click", (1, 2, 5)), E("hover", (1, 2, 4.0)), E("hover", (1, 2, 2.9)),
      E("hover", (1, 2, 2.0)), E("hover", (1, 2, 3.0)), E("release", (1, 2, 3.0)))
pprog("step_square", CAM_CUBE, E("hover", (1.5, 1.5, 2)), E("click", (1.5, 1.5, 2)), E("hover", (1.5, 1.5, 2.5)),
      E("hover", (1.5, 1.5, 3.0)), E("hover", (1.5, 1.5, 3.6)), E("hover", (1.5, 1.5, 1.5)), E("click", (1.5, 1.5, 3.0)))
pprog("ring_on_box", CAM_RING, E("hover", (0, 2.5, 3)), E("click", (0, 2.5, 3)), E("hover", (0, 2.5, 2.2)), E("hover", (0, 2.5, 4.5)),
      E("hover", (0, 2.5, 1.0)), E("click", (0, 2.5, 1.0)))
pprog("gable", dict(target=(2, 2, 2), distance=16.0, yaw=-40.0, pitch=25.0, w=1600, h=900),
      E("hover", (3.0, 2, 3.5)), E("click", (3.0, 2, 3.5)), E("hover", (3.0, 2, 4.0)), E("hover", (3.0, 2, 4.5)),
      E("hover", (3.0, 2, 3.0)), E("hover", (3.0, 2, 2.5)), E("click", (3.0, 2, 2.5)))
pprog("cube_side_rect", dict(target=(3, 2, 1.5), distance=16.0, yaw=-20.0, pitch=20.0, w=1600, h=900),
      E("hover", (4, 1.75, 1.5)), E("click", (4, 1.75, 1.5)), E("hover", (4.8, 1.75, 1.5)), E("hover", (6.0, 1.75, 1.5)),
      E("hover", (3.0, 1.75, 1.5)), E("hover", (1.0, 1.75, 1.5)), E("release", (1.0, 1.75, 1.5)))
pprog("painted_box", CAM_RING, E("hover", (0, 0, 3)), E("click", (0, 0, 3)), E("hover", (0, 0, 2.0)), E("hover", (0, 0, 3.8)),
      E("hover", (0, 0, 2.4)), E("click", (0, 0, 2.4)))


def run_pointer(path: Path, cam_spec: dict, events: list) -> dict:
    ORD.clear()
    KEEP.clear()
    scene = reopen(path)
    register_all(scene.mesh)
    for g in scene.groups:
        register_all(g.mesh)
    cm._CURVE_COUNTER = 5000
    tp.PushPullTool.last_distance = None
    cam = Camera()
    cam.target = V(*cam_spec["target"])
    cam.distance = cam_spec["distance"]
    cam.yaw = math.radians(cam_spec["yaw"])
    cam.pitch = math.radians(cam_spec["pitch"])
    cam.aspect = cam_spec["w"] / cam_spec["h"]
    cam.perspective = cam_spec.get("perspective", True)
    vp = PointerViewport(scene, cam, cam_spec["w"], cam_spec["h"])
    tool = tp.PushPullTool()
    rec = {"cam": cam_spec, "pre": snap(scene), "events": []}

    def pixel(where):
        dx = dy = 0.0
        if where[0] == "at":
            _, where, dx, dy = where
        px, py = vp._world_to_pixel(V(*where))
        return px + dx, py + dy

    def face_ix(f):
        return scene.mesh.faces.index(f) if f is not None and f in scene.mesh.faces else None

    for ev in events:
        vp.last_status = None
        kind = ev["e"]
        out = {"e": kind, "ctrl": ev["ctrl"], "d": ev["d"]}
        if ev["where"] is not None:
            x, y = pixel(ev["where"])
            out["x"], out["y"] = x, y
        vp._hover_edge = None
        if ev["edge"] is not None:
            a, b = ev["edge"]
            vp._hover_edge = HoverEdge(V(*a), V(*b))
            out["edge"] = [pbits(V(*a)), pbits(V(*b))]
        c = ToolContext(viewport=vp, world=V(0, 0, 0), screen=QPointF(out.get("x", 0.0), out.get("y", 0.0)),
                        modifiers=Qt.ControlModifier if ev["ctrl"] else Qt.NoModifier, snap=None)
        if kind == "hover":
            tool.on_hover(c)
        elif kind == "click":
            tool.on_click(c)
        elif kind == "double":
            tool.on_double_click(c)
        elif kind == "release":
            tool.on_release(vp)
        elif kind == "value":
            out["ret"] = tool.on_value(vp, ev["d"])
        elif kind == "cancel":
            tool.on_cancel(vp)
        elif kind == "undo":
            vp.history.undo()
        elif kind == "redo":
            vp.history.redo()
        else:
            raise ValueError(kind)
        out["ext"] = tool.extrusion
        out["ik"] = tool._inference_kind
        out["ip"] = None if tool._inference_point is None else pbits(tool._inference_point)
        out["hover"] = face_ix(vp.last_hover)
        out["supp"] = None if vp.suppressed is None else sorted(i for i in map(face_ix, vp.suppressed) if i is not None)
        out["light"] = [[[pbits(v) for v in f.vertices], [[pbits(v) for v in h] for h in f.holes],
                         sha(_dumps(canonical(f.attrs)), 16) if f.attrs else ""]
                        for f in tool.preview_faces()]
        out["band"] = [[pbits(a), pbits(b)] for a, b in tool.rubber_band_lines()]
        lab = tool.value_label()
        out["label"] = None if lab is None else [lab[0], pbits(lab[1])]
        out["drag"] = tool.dragging
        out["status"] = vp.last_status
        out["last_distance"] = tp.PushPullTool.last_distance
        out["depth"] = len(vp.history.undo_stack)
        out["state"] = snap(scene)
        rec["events"].append(out)
    return rec


def main() -> None:
    only_pointer = "pointer" in sys.argv[1:]
    if not only_pointer:
        shutil.rmtree(OUT, ignore_errors=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / ".gitattributes").write_text("* -text\n", encoding="utf-8", newline="\n")
    paths = {}
    for name, build in {**SCENES, **POINTER_SCENES}.items():
        s = build()
        path = OUT / f"{name}.igz.json"
        save_scene_json(s, path)
        paths[name] = path
    if not only_pointer:
        cases = []
        bad = 0
        for scene_name, steps in PROGRAMS:
            try:
                rec = run_program(paths[scene_name], steps)
            except LookupError:
                bad += 1
                continue
            rec["scene"] = scene_name
            cases.append(rec)
        n_steps = sum(len(c["steps"]) for c in cases)
        (OUT / "cases.json").write_text(json.dumps(cases, sort_keys=True, separators=(",", ":")),
                                        encoding="utf-8", newline="\n")
        print(len(cases), "programs,", n_steps, "steps", bad, "broken")
    pcases = []
    for scene_name, cam, events in POINTER_PROGRAMS:
        rec = run_pointer(paths[scene_name], cam, events)
        rec["scene"] = scene_name
        pcases.append(rec)
    (OUT / "pointer.json").write_text(json.dumps(pcases, sort_keys=True, separators=(",", ":")),
                                      encoding="utf-8", newline="\n")
    print(len(pcases), "pointer programs,", sum(len(c["events"]) for c in pcases), "events")


if __name__ == "__main__":
    main()
