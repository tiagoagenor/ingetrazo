# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
"""Fixtures that hold ``ingetrazo-ops``' Follow Me to the Python app
(docs/rust-rewrite-plan.md, M3.6): the REAL ``tools.followme.FollowMeTool``
driven the way the tests drive it (a stub viewport with a real ``History``),
over profiles and paths of every kind: straight and bent open paths, a path
the profile stands in the middle of, closed loops (a moulding mitred round a
slab, a welded torus), lathes about an axis (a washer, a sphere with the path
circle already drawn, a profile across the axis, a tube, a torus section),
profiles with one and two holes, near-collinear segments, a 180 degree
reversal and other paths the sweep declines, a curved sweep whose seams
soften, a boss grown out of a face, a blind tunnel dug through a wall's hole,
a profile inside an open group, and random profiles on random planes.

Three ways in, as in the tool: path edges selected first (``edges``), a face
whose outline is the path (``face``), and the dragged path (``drag``: click
the profile, touch edges, Alt over a face, click or release).

Each scene is saved as a bare-JSON .igz (``<scene>.igz.json``) and reopened,
so Python and Rust start from the very same bits. ``cases.json`` records per
case what the tool did: the path it read from the selection (before the
profile orients it), what ``_commit`` was handed (profile, oriented path,
closed), whether ``sweep_profile`` accepted, the history depth after the
commit and the active mesh's whole ordered state before, after the commit,
after the undo, after the redo and after a second undo (vertices, edges with
their flags and faces, faces with their loops, holes, interior flag, attrs).

Python's own order is not repeatable in one place, pinned here: the selected
edges and faces are ``scene.selection``, a ``set`` of objects; it iterates in
creation order. Everything else Follow Me reads is a list.

Committed; rerun only to widen (main venv, this tree on PYTHONPATH):

    PYTHONPATH=. QT_QPA_PLATFORM=offscreen PYTHONUTF8=1 \\
        python scripts/golden/make_followme_fixtures.py

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
from PySide6.QtWidgets import QApplication  # noqa: E402

from core.group import Group  # noqa: E402
from core.history import History  # noqa: E402
from core.mesh import Mesh  # noqa: E402
from core.scene import Scene  # noqa: E402
from core.vecmath import QVector3D as V  # noqa: E402
from formats import igz  # noqa: E402
from tests.golden.signature import _dumps, canonical, sha  # noqa: E402
import tools.followme as fm  # noqa: E402
from tools.base import ToolContext  # noqa: E402

OUT = ROOT / "rust" / "crates" / "ingetrazo-ops" / "tests" / "data" / "followme"

_APP = QApplication.instance() or QApplication([])

# ---- pinning Python's order (as make_stitch_fixtures.py) ---------------------------
#
# ``core.mesh.set`` iterates its mesh objects in CREATION order: the sweep's
# stitch splits an edge where a profile corner lies on it and collapses the
# vertex again, and ``collapse_vertex`` orders the merged edge's faces from
# ``set(e1.faces)`` (address order in Python). Likewise ``_coplanar_component``
# and ``_trace_loops``, which the stitch/orient passes may reach. The objects
# are kept alive (TAB) so no ``id`` is ever recycled.

import core.history as ch  # noqa: E402
import core.mesh as cm  # noqa: E402

ORD: dict = {}                       # id(object) -> creation ordinal
TAB = {"v": [], "e": [], "f": []}    # ordinal -> object


def _reg(kind: str, o) -> None:
    if id(o) not in ORD:
        ORD[id(o)] = len(TAB[kind])
        TAB[kind].append(o)


class DetSet(set):
    """A set that iterates mesh objects in creation order."""

    def __iter__(self):
        return iter(sorted(set.__iter__(self), key=lambda o: ORD.get(id(o), 1 << 60)))


cm.set = DetSet


def _wrap(name: str, kind: str) -> None:
    orig = getattr(Mesh, name)

    def wrapper(self, *a, **k):
        o = orig(self, *a, **k)
        _reg(kind, o)
        return o
    setattr(Mesh, name, wrapper)


_wrap("vertex", "v")
_wrap("_link_edge", "e")
_wrap("add_face", "f")

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


class Pinned(set):
    """The selection, iterated in the order given."""

    def __init__(self, ordered) -> None:
        super().__init__(ordered)
        self._order = list(ordered)

    def __iter__(self):
        return iter(self._order)


# ---- scenes -------------------------------------------------------------------------

N = 24


def xz_square(m, x0=0.0, z0=0.0, size=1.0, y=0.0):
    return m.add_face([V(x0, y, z0), V(x0 + size, y, z0),
                       V(x0 + size, y, z0 + size), V(x0, y, z0 + size)])


def chain(m, pts, closed=False):
    n = len(pts)
    for i in range(n if closed else n - 1):
        m.add_edge(pts[i], pts[(i + 1) % n])


def circle_pts(cx, cy, cz, r, n, plane="xy", phase=0.0):
    out = []
    for k in range(n):
        a = 2 * math.pi * k / n + phase
        c, s = r * math.cos(a), r * math.sin(a)
        if plane == "xy":
            out.append(V(cx + c, cy + s, cz))
        elif plane == "xz":
            out.append(V(cx + c, cy, cz + s))
        else:
            out.append(V(cx, cy + c, cz + s))
    return out


def box_faces(m, x, y, z, sx, sy, sz):
    x1, y1, z1 = x + sx, y + sy, z + sz
    for pts in ([V(x, y, z), V(x, y1, z), V(x1, y1, z), V(x1, y, z)],
                [V(x, y, z1), V(x1, y, z1), V(x1, y1, z1), V(x, y1, z1)],
                [V(x, y, z), V(x1, y, z), V(x1, y, z1), V(x, y, z1)],
                [V(x, y1, z), V(x, y1, z1), V(x1, y1, z1), V(x1, y1, z)],
                [V(x, y, z), V(x, y, z1), V(x, y1, z1), V(x, y1, z)],
                [V(x1, y, z), V(x1, y1, z), V(x1, y1, z1), V(x1, y, z1)]):
        m.add_face(pts)


def scenes() -> dict:
    out = {}

    # --- straight / bent / crossing
    m = Mesh()
    xz_square(m)
    chain(m, [V(0, 0, 0), V(0, 3, 0)])
    out["straight"] = (m, [])

    m = Mesh()
    xz_square(m)
    chain(m, [V(0.5, -2, 0.5), V(0.5, 3, 0.5)])
    out["crossing"] = (m, [])

    m = Mesh()
    m.add_face([V(0, 0, 0), V(2, 0, 0), V(2, 0, 0.5), V(0.5, 0, 0.5),
                V(0.5, 0, 2), V(0, 0, 2)])
    chain(m, [V(0, 0, 0), V(0, 4, 0), V(3, 4, 0), V(3, 4, 3), V(6, 6, 3)])
    out["bend"] = (m, [])

    m = Mesh()
    xz_square(m)
    chain(m, [V(0, 0, 0), V(0, 4, 0), V(4, 4, 0)])
    m.add_edge(V(0, 4, 0), V(0, 4, 3))               # a branch at (0,4,0)
    out["branch"] = (m, [])

    # --- closed loops
    m = Mesh()
    m.add_face([V(0, 0, 0), V(6, 0, 0), V(6, 6, 0), V(0, 6, 0)])
    m.add_face([V(0, 0, 0.5), V(-1, 0, 0.5), V(-1, 0, 1.5), V(0, 0, 1.5)])
    out["frame"] = (m, [])

    m = Mesh()
    m.add_face([V(0, 0, 0), V(0, 0, 1), V(1, 0, 1), V(1, 0, 0)])   # mitre profile at the corner
    chain(m, [V(0, 0, 0), V(6, 0, 0), V(6, 6, 0), V(0, 6, 0)], closed=True)
    out["frame_edges"] = (m, [])

    m = Mesh()
    m.add_face([V(0, 0, 0), V(2, 0, 0), V(2, 0, 2), V(0, 0, 2)],
               [[V(0.5, 0, 0.5), V(0.5, 0, 1.5), V(1.5, 0, 1.5), V(1.5, 0, 0.5)]])
    chain(m, [V(0, 0, 0), V(8, 0, 0), V(8, 8, 0), V(0, 8, 0)], closed=True)
    out["frame_hole"] = (m, [])

    m = Mesh()
    m.add_face([V(4, 0, 0), V(5, 0, 0), V(5, 0, 1), V(4, 0, 1)])
    chain(m, [V(4.5 * math.cos(2 * math.pi * i / 12), 4.5 * math.sin(2 * math.pi * i / 12), 0.5)
              for i in range(12)], closed=True)
    out["torus12"] = (m, [])

    # --- lathes
    m = Mesh()
    m.add_face([V(1.5, 0, 0), V(2.5, 0, 0), V(2.5, 0, 1), V(1.5, 0, 1)])
    chain(m, circle_pts(0, 0, 0, 1.0, N), closed=True)
    out["lathe_rect"] = (m, [])

    m = Mesh()
    ph = math.radians(7.5)
    m.add_face([V(math.cos(ph) * 1.5 + 0.0, math.sin(ph) * 1.5, 0), V(math.cos(ph) * 2.5, math.sin(ph) * 2.5, 0),
                V(math.cos(ph) * 2.5, math.sin(ph) * 2.5, 1), V(math.cos(ph) * 1.5, math.sin(ph) * 1.5, 1)])
    chain(m, circle_pts(0, 0, 0, 1.0, N), closed=True)
    out["lathe_rect75"] = (m, [])

    for tag, phi in (("0", 0.0), ("75", 7.5), ("33", 33.0)):
        m = Mesh()
        chain(m, circle_pts(0, 0, 0, 1.0, N), closed=True)      # the equator, already drawn
        u = (math.cos(math.radians(phi)), math.sin(math.radians(phi)))
        m.add_face([V(u[0] * math.cos(2 * math.pi * k / N), u[1] * math.cos(2 * math.pi * k / N),
                      math.sin(2 * math.pi * k / N)) for k in range(N)])
        out[f"sphere{tag}"] = (m, [])

    m = Mesh()
    u = (math.cos(math.radians(33)), math.sin(math.radians(33)))
    m.add_face([V(u[0] * 0.5 * math.cos(2 * math.pi * k / N), u[1] * 0.5 * math.cos(2 * math.pi * k / N),
                  0.5 * math.sin(2 * math.pi * k / N)) for k in range(N)])
    m.add_face(circle_pts(0, 0, 0, 1.0, N))                      # the path as a face
    out["sphere_small_face"] = (m, [])

    m = Mesh()
    m.add_face([V(2 + 0.3 * math.cos(2 * math.pi * k / N), 0, 0.3 * math.sin(2 * math.pi * k / N))
                for k in range(N)])
    chain(m, circle_pts(0, 0, 0, 2.0, N), closed=True)
    out["torus_rev"] = (m, [])

    m = Mesh()
    m.add_face([V(1, 0, 0), V(3, 0, 0), V(3, 0, 2), V(1, 0, 2)],
               [[V(1.5, 0, 0.5), V(1.5, 0, 1.5), V(2.5, 0, 1.5), V(2.5, 0, 0.5)]])
    chain(m, circle_pts(0, 0, 0, 1.0, 16), closed=True)
    out["lathe_tube"] = (m, [])

    m = Mesh()
    m.add_face([V(-1, 0, 0), V(2, 0, 0), V(2, 0, 2), V(-1, 0, 2)],
               [[V(0.5, 0, 0.5), V(0.5, 0, 1.5), V(1.5, 0, 1.5), V(1.5, 0, 0.5)]])
    chain(m, circle_pts(0, 0, 0, 1.0, 16), closed=True)
    out["lathe_across"] = (m, [])

    m = Mesh()
    m.add_face([V(-1, 0, 0), V(2, 0, 0), V(2, 0, 2), V(-1, 0, 2)],
               [[V(-0.5, 0, 0.5), V(-0.5, 0, 1.5), V(0.5, 0, 1.5), V(0.5, 0, 0.5)]])
    chain(m, circle_pts(0, 0, 0, 1.0, 16), closed=True)
    out["lathe_hole_across"] = (m, [])

    m = Mesh()
    m.add_face([V(0, 0, 0), V(1.2, 0, 0), V(1.2, 0, 1.5), V(0, 0, 1.5)])   # touches the axis
    chain(m, circle_pts(0, 0, 0, 1.0, 20), closed=True)
    out["lathe_axis"] = (m, [])

    # --- holes, open path
    m = Mesh()
    m.add_face([V(0, 0, 0), V(2, 0, 0), V(2, 0, 2), V(0, 0, 2)],
               [[V(0.5, 0, 0.5), V(1.5, 0, 0.5), V(1.5, 0, 1.5), V(0.5, 0, 1.5)]])
    chain(m, [V(0, 0, 0), V(0, 5, 0)])
    out["tube"] = (m, [])

    m = Mesh()
    m.add_face([V(0, 0, 0), V(3, 0, 0), V(3, 0, 2), V(0, 0, 2)],
               [[V(0.2, 0, 0.2), V(1.2, 0, 0.2), V(1.2, 0, 1.8), V(0.2, 0, 1.8)],
                [V(1.8, 0, 0.2), V(2.8, 0, 0.2), V(2.8, 0, 1.8), V(1.8, 0, 1.8)]])
    chain(m, [V(1.5, 0, 1), V(1.5, 3, 1), V(4.5, 5, 2)])
    out["two_holes"] = (m, [])

    # --- near collinear / declines
    m = Mesh()
    xz_square(m)
    chain(m, [V(0, 0, 0), V(0, 2, 0), V(0.001, 4, 0), V(0.001, 6, 0.0005), V(0.002, 8, 0.0005)])
    out["near_collinear"] = (m, [])

    m = Mesh()
    xz_square(m)
    m.add_edge(V(0, 0, 0), V(0, 3, 0))
    m.add_edge(V(0, 3, 0), V(0, 1, 0))                          # folds back on itself
    out["reversal"] = (m, [])

    m = Mesh()
    xz_square(m)
    chain(m, [V(0, 0, 0), V(0, 3, 0), V(0, 3.0002, 0.0003), V(0, 5, 0)])
    out["tiny_span"] = (m, [])

    m = Mesh()
    xz_square(m)
    m.add_edge(V(2, 0, 0), V(2, 0, 2))                          # in the profile's plane
    chain(m, [V(0, 0, 0), V(0, 3, 0)])
    out["inplane"] = (m, [])

    # --- curved sweep, softening
    m = Mesh()
    m.add_face(circle_pts(5, 0, 0, 0.3, 12, "xz"))
    chain(m, [V(5 * math.cos(t * math.pi / 2 / 12), 5 * math.sin(t * math.pi / 2 / 12), 0) for t in range(13)])
    out["arc"] = (m, [])

    m = Mesh()
    m.add_face([V(5, 0, 0), V(5.4, 0, 0), V(5.4, 0, 0.4), V(5, 0, 0.4)])
    chain(m, [V(5 * math.cos(t * math.pi / 2 / 8), 5 * math.sin(t * math.pi / 2 / 8), 0) for t in range(9)])
    chain(m, [V(5 * math.cos(math.pi / 2), 5 * math.sin(math.pi / 2), 0), V(0, 7, 0)])
    out["arc_line"] = (m, [])

    # --- over existing geometry
    m = Mesh()
    box_faces(m, 0, 0, 0, 4, 4, 2)
    # replace the top face by a ring + a disc: draw the profile on it
    top = [f for f in m.faces if all(abs(v.z() - 2) < 1e-9 for v in f.vertices)][0]
    m.remove_face(top)
    disc = circle_pts(2, 2, 2, 0.8, 16)
    m.add_face([V(0, 0, 2), V(4, 0, 2), V(4, 4, 2), V(0, 4, 2)], [list(reversed(disc))])
    m.add_face(disc)
    m.add_edge(V(2, 2, 2), V(2, 2, 3.5))
    out["boss"] = (m, [])

    m = Mesh()
    box_faces(m, 0, 0, 0, 4, 4, 4)
    front = [f for f in m.faces if all(abs(v.y()) < 1e-9 for v in f.vertices)][0]
    m.remove_face(front)
    hole = [V(1, 0, 1), V(3, 0, 1), V(3, 0, 3), V(1, 0, 3)]
    m.add_face([V(0, 0, 0), V(4, 0, 0), V(4, 0, 4), V(0, 0, 4)], [list(reversed(hole))])
    m.add_face(hole)
    chain(m, [V(2, 0, 2), V(2, 2, 2)])
    out["tunnel"] = (m, [])

    m = Mesh()
    box_faces(m, 0, 0, 0, 4, 4, 4)
    front = [f for f in m.faces if all(abs(v.y()) < 1e-9 for v in f.vertices)][0]
    m.remove_face(front)
    m.add_face([V(0, 0, 0), V(4, 0, 0), V(4, 0, 4), V(0, 0, 4)], [list(reversed(hole))])
    m.add_face(hole)
    chain(m, [V(2, 0, 2), V(2, 4, 2)])                         # right through to the back wall
    out["tunnel_through"] = (m, [])

    # --- inside a group
    g = Mesh()
    xz_square(g, 1, 1, 2)
    chain(g, [V(2, 0, 2), V(2, 3, 2), V(5, 3, 2)])
    loose = Mesh()
    xz_square(loose, 10, 10, 1)
    out["in_group"] = (loose, [("G", g)])
    return out


def build_scene(loose: Mesh, groups: list) -> Scene:
    s = Scene()
    for f in loose.faces:
        s.mesh.add_face([QV(v.position) for v in f.loop],
                        [[QV(v.position) for v in h] for h in f.hole_loops] or None)
    for e in loose.edges:
        if not e.faces:
            s.mesh.add_edge(QV(e.v0.position), QV(e.v1.position))
    for name, gm in groups:
        s.groups.append(Group(gm, name=name))
    return s


def QV(p):
    return V(p.x(), p.y(), p.z())


# ---- random cases ---------------------------------------------------------------------

def random_scene(seed: int):
    rnd = random.Random(31000 + seed)
    nrm = [rnd.uniform(-1, 1) for _ in range(3)]
    ln = math.sqrt(sum(c * c for c in nrm))
    nrm = [c / ln for c in nrm]
    helper = (1, 0, 0) if abs(nrm[0]) < 0.9 else (0, 1, 0)
    u = [nrm[1] * helper[2] - nrm[2] * helper[1], nrm[2] * helper[0] - nrm[0] * helper[2],
         nrm[0] * helper[1] - nrm[1] * helper[0]]
    lu = math.sqrt(sum(c * c for c in u))
    u = [c / lu for c in u]
    w = [nrm[1] * u[2] - nrm[2] * u[1], nrm[2] * u[0] - nrm[0] * u[2], nrm[0] * u[1] - nrm[1] * u[0]]
    org = [rnd.uniform(-3, 3) for _ in range(3)]
    k = rnd.randint(3, 9)
    pts = []
    for i in range(k):
        a = 2 * math.pi * i / k
        r = rnd.uniform(0.4, 1.0)
        pts.append(V(*(org[j] + u[j] * r * math.cos(a) + w[j] * r * math.sin(a) for j in range(3))))
    m = Mesh()
    m.add_face(pts)
    # the path: from the profile's centroid along the normal, bending a bit
    p = [org[j] for j in range(3)]
    d = nrm[:]
    path = [V(*p)]
    for _ in range(rnd.randint(2, 6)):
        step = rnd.uniform(0.8, 2.5)
        bend = [rnd.uniform(-0.6, 0.6) for _ in range(3)]
        d = [d[j] + bend[j] for j in range(3)]
        ld = math.sqrt(sum(c * c for c in d))
        d = [c / ld for c in d]
        p = [p[j] + d[j] * step for j in range(3)]
        path.append(V(*p))
    chain(m, path)
    return m


# ---- running the tool -------------------------------------------------------------------

class Vp:
    def __init__(self, scene):
        self.scene = scene
        self.history = History(scene)
        self.messages = []
        self._pick = None
        self._edge = None

    def update(self):
        pass

    def set_hover(self, *_):
        pass

    def flash_status(self, text, msec=2500):
        self.messages.append(text)

    def pick_face(self, x, y):
        return self._pick

    def pick_edge(self, x, y):
        return self._edge


def ctx(vp, x=0.0, y=0.0, world=None, mods=Qt.NoModifier):
    return ToolContext(viewport=vp, world=world or V(0, 0, 0), screen=QPointF(x, y),
                       modifiers=mods, snap=None)


def loose_edges(m):
    return [i for i, e in enumerate(m.edges) if not e.faces]


def edge_ix(m, p, q):
    for i, e in enumerate(m.edges):
        a, b = e.v0.position, e.v1.position
        if ((a - p).length() < 1e-6 and (b - q).length() < 1e-6) or \
           ((a - q).length() < 1e-6 and (b - p).length() < 1e-6):
            return i
    raise KeyError((p, q))


def run_case(path: Path, name: str, flow: str, spec: dict, edit):
    scene = reopen(path)
    if edit is not None:
        scene.begin_group_edit(scene.groups[edit])
    m = scene.mesh
    vp = Vp(scene)
    tool = fm.FollowMeTool()
    commits, sweeps = [], []
    orig_commit = tool._commit

    def rec_commit(viewport, face, pth, closed):
        commits.append({"face": m.faces.index(face), "path": [pbits(p) for p in pth],
                        "closed": bool(closed)})
        return orig_commit(viewport, face, pth, closed)
    tool._commit = rec_commit
    orig_sweep = fm.sweep_profile

    def rec_sweep(mesh, face, pth, closed):
        r = orig_sweep(mesh, face, pth, closed)
        sweeps.append(bool(r))
        return r
    fm.sweep_profile = rec_sweep
    rec = {"scene": name, "flow": flow, "edit": edit, "spec": spec, "pre": mesh_state(m)}
    try:
        if flow in ("edges", "face"):
            if flow == "edges":
                scene.selection = Pinned([m.edges[i] for i in spec["edges"]])
            else:
                scene.selection = Pinned([m.faces[spec["path_face"]]])
            tool.on_activate(vp)
            rec["path0"] = ([pbits(p) for p in tool._path], bool(tool._closed)) if tool._path is not None else None
            vp._pick = m.faces[spec["profile"]]
            tool.on_hover(ctx(vp))
            tool.on_click(ctx(vp))
        else:
            tool.on_activate(vp)
            rec["path0"] = None
            vp._pick = m.faces[spec["profile"]]
            tool.on_hover(ctx(vp))
            tool.on_click(ctx(vp, 10, 10))
            for st in spec["steps"]:
                vp._edge = m.edges[st["edge"]] if st.get("edge") is not None else None
                vp._pick = m.faces[st["face"]] if st.get("face") is not None else None
                mods = Qt.AltModifier if st.get("alt") else Qt.NoModifier
                w = V(*st["world"]) if st.get("world") else V(0, 0, 0)
                tool.on_hover(ctx(vp, st.get("x", 20), 10, world=w, mods=mods))
            end = spec.get("finish", "click")
            if end == "click":
                tool.on_click(ctx(vp, 99, 99))
            elif end == "release":
                tool.on_release(vp)
        rec["commits"] = commits
        rec["sweeps"] = sweeps
        rec["messages"] = list(vp.messages)
        rec["undo_len"] = len(vp.history.undo_stack)
        rec["redo_len"] = len(vp.history.redo_stack)
        rec["post"] = mesh_state(scene.mesh)
        if vp.history.undo_stack:
            assert vp.history.undo()
            rec["undone"] = mesh_state(scene.mesh)
            assert rec["undone"] == rec["pre"], (name, flow)
            assert vp.history.redo()
            rec["redone"] = mesh_state(scene.mesh)
            assert rec["redone"] == rec["post"], (name, flow)
            assert vp.history.undo()
            rec["undone2"] = mesh_state(scene.mesh)
            assert rec["undone2"] == rec["pre"]
    finally:
        fm.sweep_profile = orig_sweep
    return rec


def case_list(opened: dict) -> list:
    cases = []

    def add(scene, flow, spec, edit=None):
        cases.append((scene, flow, spec, edit))

    def drag(scene, profile, steps, finish="click", edit=None):
        add(scene, "drag", {"profile": profile, "steps": steps, "finish": finish}, edit)

    def E(name):
        return opened[name].mesh

    # straight
    m = E("straight")
    add("straight", "edges", {"edges": loose_edges(E("straight")), "profile": 0})
    drag("straight", 0, [{"edge": loose_edges(E("straight"))[0], "world": (0, 1, 0)}])
    drag("straight", 0, [{"edge": loose_edges(E("straight"))[0], "world": (0, 1, 0), "x": 5}], finish="release")
    drag("straight", 0, [{"edge": loose_edges(E("straight"))[0], "world": (0, 1, 0), "x": 80}], finish="release")
    drag("straight", 0, [], finish="click")                        # nothing touched
    drag("straight", 0, [{"edge": loose_edges(E("straight"))[0], "world": (0, 1, 0)}], finish="none")
    # crossing: the profile sits mid-edge
    add("crossing", "edges", {"edges": loose_edges(E("crossing")), "profile": 0})
    drag("crossing", 0, [{"edge": loose_edges(E("crossing"))[0], "world": (0.5, 2.5, 0.5)}])
    drag("crossing", 0, [{"edge": loose_edges(E("crossing"))[0], "world": (0.5, -1.5, 0.5)}])
    # bent
    m = E("bend")
    ed = loose_edges(m)
    add("bend", "edges", {"edges": ed, "profile": 0})
    add("bend", "edges", {"edges": list(reversed(ed)), "profile": 0})
    add("bend", "edges", {"edges": ed[:2], "profile": 0})
    add("bend", "edges", {"edges": [ed[1], ed[3], ed[0]], "profile": 0})      # a gap: no path
    drag("bend", 0, [{"edge": ed[0], "world": (0, 2, 0)}, {"edge": ed[1], "world": (1, 4, 0)},
                     {"edge": ed[2], "world": (3, 4, 1)}, {"edge": ed[3], "world": (4, 5, 3)}])
    drag("bend", 0, [{"edge": ed[0], "world": (0, 2, 0)}, {"edge": ed[3], "world": (4, 5, 3)}])   # bridged
    drag("bend", 0, [{"edge": ed[0], "world": (0, 2, 0)}, {"edge": ed[1], "world": (1, 4, 0)},
                     {"edge": ed[2], "world": (3, 4, 1)}, {"edge": ed[1], "world": (1, 4, 0)}])   # shrinks
    drag("bend", 0, [{"edge": ed[3], "world": (4, 5, 3)}, {"edge": ed[2], "world": (3, 4, 1)}])   # from the far end
    # branch: invalid selection and a drag over the fork
    m = E("branch")
    ed = loose_edges(m)
    add("branch", "edges", {"edges": ed, "profile": 0})
    add("branch", "edges", {"edges": ed[:2], "profile": 0})
    drag("branch", 0, [{"edge": ed[0], "world": (0, 2, 0)}, {"edge": ed[1], "world": (2, 4, 0)}])
    drag("branch", 0, [{"edge": ed[0], "world": (0, 2, 0)}, {"edge": ed[2], "world": (0, 4, 2)}])
    # frames
    add("frame", "face", {"path_face": 0, "profile": 1})
    drag("frame", 1, [{"face": 0, "alt": True}])
    drag("frame", 1, [{"face": 1, "alt": True}])                  # Alt over the profile: ignored
    m = E("frame_edges")
    add("frame_edges", "edges", {"edges": loose_edges(m), "profile": 0})
    add("frame_edges", "edges", {"edges": list(reversed(loose_edges(m))), "profile": 0})
    ed = loose_edges(m)
    drag("frame_edges", 0, [{"edge": ed[0], "world": (3, 0, 0)}, {"edge": ed[1], "world": (6, 3, 0)},
                            {"edge": ed[2], "world": (3, 6, 0)}, {"edge": ed[3], "world": (0, 3, 0)}])
    m = E("frame_hole")
    add("frame_hole", "edges", {"edges": loose_edges(m), "profile": 0})
    m = E("torus12")
    add("torus12", "edges", {"edges": loose_edges(m), "profile": 0})
    # lathes
    add("lathe_rect", "edges", {"edges": loose_edges(E("lathe_rect")), "profile": 0})
    add("lathe_rect75", "edges", {"edges": loose_edges(E("lathe_rect75")), "profile": 0})
    add("lathe_rect75", "edges", {"edges": list(reversed(loose_edges(E("lathe_rect75")))), "profile": 0})
    for tag in ("0", "75", "33"):
        add(f"sphere{tag}", "edges", {"edges": loose_edges(E(f"sphere{tag}")), "profile": 0})
    add("sphere_small_face", "face", {"path_face": 1, "profile": 0})
    add("torus_rev", "edges", {"edges": loose_edges(E("torus_rev")), "profile": 0})
    add("lathe_tube", "edges", {"edges": loose_edges(E("lathe_tube")), "profile": 0})
    add("lathe_across", "edges", {"edges": loose_edges(E("lathe_across")), "profile": 0})
    add("lathe_hole_across", "edges", {"edges": loose_edges(E("lathe_hole_across")), "profile": 0})
    add("lathe_axis", "edges", {"edges": loose_edges(E("lathe_axis")), "profile": 0})
    # holes
    add("tube", "edges", {"edges": loose_edges(E("tube")), "profile": 0})
    drag("tube", 0, [{"edge": loose_edges(E("tube"))[0], "world": (0, 2, 0)}])
    ed = loose_edges(E("two_holes"))
    add("two_holes", "edges", {"edges": ed, "profile": 0})
    drag("two_holes", 0, [{"edge": ed[0], "world": (1.5, 1, 1)}, {"edge": ed[1], "world": (3, 4, 1.5)}])
    # near collinear and declines
    ed = loose_edges(E("near_collinear"))
    add("near_collinear", "edges", {"edges": ed, "profile": 0})
    drag("near_collinear", 0, [{"edge": ed[0], "world": (0, 1, 0)}, {"edge": ed[3], "world": (0, 7, 0)}])
    ed = loose_edges(E("reversal"))
    add("reversal", "edges", {"edges": ed, "profile": 0})
    drag("reversal", 0, [{"edge": ed[0], "world": (0, 1, 0)}, {"edge": ed[1], "world": (0, 2, 0)}])
    ed = loose_edges(E("tiny_span"))
    add("tiny_span", "edges", {"edges": ed, "profile": 0})
    ed = loose_edges(E("inplane"))
    drag("inplane", 0, [{"edge": ed[0], "world": (2, 0, 1)}])
    drag("inplane", 0, [{"edge": ed[0], "world": (2, 0, 1)}, {"edge": ed[1], "world": (0, 1, 0)}])
    # curved sweeps
    ed = loose_edges(E("arc"))
    add("arc", "edges", {"edges": ed, "profile": 0})
    add("arc", "edges", {"edges": list(reversed(ed)), "profile": 0})
    drag("arc", 0, [{"edge": ed[0], "world": (5, 0.2, 0)}, {"edge": ed[5], "world": (4, 3, 0)},
                    {"edge": ed[11], "world": (0.3, 5, 0)}])
    ed = loose_edges(E("arc_line"))
    add("arc_line", "edges", {"edges": ed, "profile": 0})
    drag("arc_line", 0, [{"edge": ed[0], "world": (5, 0.2, 0)}, {"edge": ed[-1], "world": (0, 6, 0)}])
    # over existing geometry
    for sc in ("boss", "tunnel", "tunnel_through"):
        m = E(sc)
        prof = len(m.faces) - 1
        ed = loose_edges(m)
        add(sc, "edges", {"edges": ed, "profile": prof})
        drag(sc, prof, [{"edge": ed[0], "world": (2, 1, 2)}])
    # inside a group
    m = opened["in_group"].groups[0].mesh
    ed = loose_edges(m)
    add("in_group", "edges", {"edges": ed, "profile": 0}, edit=0)
    drag("in_group", 0, [{"edge": ed[0], "world": (2, 1, 2)}, {"edge": ed[1], "world": (3, 3, 2)}], edit=0)
    # random profiles on random planes
    for seed in range(1, 15):
        sc = f"rand{seed}"
        m = opened[sc].mesh
        ed = loose_edges(m)
        add(sc, "edges", {"edges": ed, "profile": 0})
        if seed % 3 == 0:
            add(sc, "edges", {"edges": list(reversed(ed)), "profile": 0})
    return cases


def main() -> None:
    shutil.rmtree(OUT, ignore_errors=True)
    OUT.mkdir(parents=True)
    opened: dict = {}
    allsc = scenes()
    for seed in range(1, 15):
        allsc[f"rand{seed}"] = (random_scene(seed), [])
    for name, (loose, groups) in allsc.items():
        s = build_scene(loose, groups)
        path = OUT / f"{name}.igz.json"
        save_scene_json(s, path)
        opened[name] = reopen(path)
    out_cases = []
    for name, flow, spec, edit in case_list(opened):
        out_cases.append(run_case(OUT / f"{name}.igz.json", name, flow, spec, edit))
    ok = sum(1 for c in out_cases if c.get("sweeps") and c["sweeps"][-1])
    declined = sum(1 for c in out_cases if c.get("sweeps") and not c["sweeps"][-1])
    nocommit = sum(1 for c in out_cases if not c.get("commits"))
    soft = sum(1 for c in out_cases if "post" in c and any(e[2] for e in c["post"]["e"]))
    (OUT / "cases.json").write_text(
        json.dumps(out_cases, sort_keys=True, separators=(",", ":")),
        encoding="utf-8", newline="\n")
    print(len(out_cases), "cases;", ok, "swept,", declined, "declined,", nocommit, "without a commit,",
          soft, "with soft edges")
    for c in out_cases:
        s = c.get("sweeps")
        print(f"  {c['scene']:20s} {c['flow']:5s} {'ok' if s and s[-1] else ('DECLINED' if s else 'nocommit'):9s}"
              f" f {len(c['pre']['f'])}->{len(c['post']['f'])}")


if __name__ == "__main__":
    main()
