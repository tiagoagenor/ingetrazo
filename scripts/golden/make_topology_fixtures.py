# SPDX-License-Identifier: GPL-3.0-or-later
"""Fixtures that hold ``ingetrazo-model``'s ``topology`` to
``core/topology.py`` (docs/rust-rewrite-plan.md, M3.2): random scenes
(rectangles drawn across each other, nested holes, T-junctions, edge graphs,
warped and rotated faces, boxes) and, on each, a sequence of calls to the
REAL Python functions with their arguments and results recorded bit for bit.
Mutating functions (heal, fold, sweep, orient, prune) also record the whole
mesh state after the call. Objects are named by creation order, which is
their arena index in Rust. Committed; rerun only to widen:

    python scripts/golden/make_topology_fixtures.py

What is forced to be deterministic (Python iterates sets of objects in
address order, which no run repeats):

* ``Vertex.edges`` iterates by edge creation order (``_OSet``) — only
  ``defects_near`` and ``find_edge`` on duplicate edges would notice;
* ``fold_nonplanar_faces`` builds ``pieces`` as a set of faces: the
  generator runs a copy of the module whose ``pieces`` keeps insertion order;
* ``core.orient.orient_outward`` (``heal_overlapping_faces`` on a 3D model)
  is replaced by a no-op: it is ``core/orient.py``'s port, not this one's.

Deleted with the Python app at M10.
"""
from __future__ import annotations

import json
import math
import random
import struct
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from core import mesh as cm  # noqa: E402
from core.vecmath import QVector3D as V  # noqa: E402

OUT = ROOT / "rust" / "crates" / "ingetrazo-model" / "tests" / "data" / "topology"


def h32(x: float) -> str:
    return struct.pack(">f", x).hex()


def h64(x: float) -> str:
    return struct.pack(">d", float(x)).hex()


def pbits(p) -> list:
    return [h32(p.x()), h32(p.y()), h32(p.z())]


# ---- Determinism patches -------------------------------------------------------

CHANGED: dict = {}      # mutator -> [calls, calls that changed the mesh]
ORD = {"v": {}, "e": {}, "f": {}}
KEEP: list = []
REG = [True]            # off while the generator builds throwaway objects


def _patch_ctor(cls, tag):
    orig = cls.__init__

    def init(self, *a, **k):
        orig(self, *a, **k)
        if REG[0]:
            ORD[tag][id(self)] = len(ORD[tag])
            KEEP.append(self)
    cls.__init__ = init


class _OSet(set):
    """A set that iterates in creation order of its edges."""

    def __iter__(self):
        return iter(sorted(set.__iter__(self), key=lambda e: ORD["e"][id(e)]))


_orig_vertex_init = cm.Vertex.__init__


def _vertex_init(self, position):
    _orig_vertex_init(self, position)
    self.edges = _OSet()
    if REG[0]:
        ORD["v"][id(self)] = len(ORD["v"])
        KEEP.append(self)


cm.Vertex.__init__ = _vertex_init
_patch_ctor(cm.Edge, "e")
_patch_ctor(cm.Face, "f")


def tmp_face(pts):
    """A throwaway mesh Face over fresh vertices (not named)."""
    REG[0] = False
    try:
        return cm.Face([cm.Vertex(p) for p in pts])
    finally:
        REG[0] = True


def tmp_verts(pts):
    REG[0] = False
    try:
        return [cm.Vertex(p) for p in pts]
    finally:
        REG[0] = True


class _OrderedPieces(dict):
    """An insertion-ordered set for ``fold_nonplanar_faces``'s ``pieces``."""

    def __init__(self, it=()):
        super().__init__((x, None) for x in it)

    def add(self, x):
        self.setdefault(x, None)

    def discard(self, x):
        self.pop(x, None)


def _load_topology():
    src = (ROOT / "core" / "topology.py").read_text(encoding="utf-8")
    old = """        pieces = {
            mesh.add_face([QVector3D(a), QVector3D(b), QVector3D(c)])
            for a, b, c in tris
        }"""
    new = """        pieces = _OrderedPieces(
            mesh.add_face([QVector3D(a), QVector3D(b), QVector3D(c)])
            for a, b, c in tris
        )"""
    assert old in src
    mod = types.ModuleType("topology_det")
    mod.__dict__["_OrderedPieces"] = _OrderedPieces
    mod.__dict__["__name__"] = "topology_det"
    exec(compile(src.replace(old, new), "core/topology.py", "exec"), mod.__dict__)
    return mod


T = _load_topology()

import core.orient  # noqa: E402

core.orient.orient_outward = lambda mesh, *a, **k: []


# ---- Argument / result encoding ----------------------------------------------------

class A:
    """An argument: the Python value and its JSON form."""

    def __init__(self, py, js):
        self.py, self.js = py, js


def P(p):
    return A(p, pbits(p))


def PL(ps):
    return A(list(ps), [pbits(p) for p in ps])


def PLL(lls):
    return A([list(x) for x in lls], [[pbits(p) for p in x] for x in lls])


def D(x):
    return A(float(x), h64(x))


def I(x):
    return A(x, x)


def B(x):
    return A(x, x)


def S(x):
    return A(x, x)


def NONE():
    return A(None, None)


def FR(f):
    return A(f, ORD["f"][id(f)])


def FRL(fs):
    return A(list(fs), [ORD["f"][id(f)] for f in fs])


def ER(e):
    return A(e, ORD["e"][id(e)])


def ERL(es):
    return A(list(es), [ORD["e"][id(e)] for e in es])


def VRL(vs):
    return A(list(vs), [ORD["v"][id(v)] for v in vs])


def P2(p):
    return A((float(p[0]), float(p[1])), [h64(p[0]), h64(p[1])])


def P2L(ps):
    return A([(float(a), float(b)) for a, b in ps], [[h64(a), h64(b)] for a, b in ps])


def enc(x):
    if isinstance(x, V):
        return pbits(x)
    if isinstance(x, bool):
        return x
    if isinstance(x, int):
        return x
    if isinstance(x, float):
        return h64(x)
    if hasattr(x, "dtype"):
        return h64(float(x))
    if x is None:
        return None
    if isinstance(x, str):
        return x
    if isinstance(x, cm.Face):
        return ORD["f"][id(x)]
    if isinstance(x, cm.Edge):
        return ORD["e"][id(x)]
    if isinstance(x, dict):
        return [[enc(k), enc(v)] for k, v in x.items()]
    if isinstance(x, (list, tuple)):
        return [enc(i) for i in x]
    raise TypeError(type(x))


def state(mesh) -> dict:
    o = ORD
    return {
        "v": [[o["v"][id(v)], *pbits(v.position)] for v in mesh.vertices],
        "e": [[o["e"][id(e)], o["v"][id(e.v0)], o["v"][id(e.v1)], [o["f"][id(f)] for f in e.faces],
               sorted(o["e"][id(x)] for x in e.v0.edges), sorted(o["e"][id(x)] for x in e.v1.edges),
               e.soft, e.layer, e.curve]
              for e in mesh.edges],
        "f": [[o["f"][id(f)], [o["v"][id(v)] for v in f.loop],
               [[o["v"][id(v)] for v in h] for h in f.hole_loops], f.attrs]
              for f in mesh.faces],
        "reg": sorted([*k, o["v"][id(v)]] for k, v in mesh._registry.items()),
    }


# ---- Scenes -----------------------------------------------------------------------------

class Scene:
    """A mesh built by recorded ops, then a recorded list of calls."""

    def __init__(self, seed: int) -> None:
        for d in ORD.values():
            d.clear()
        KEEP.clear()
        self.r = random.Random(seed)
        self.mesh = cm.Mesh()
        self.build: list = []
        self.calls: list = []
        self.footprints: list = []

    # -- building ops
    def add_face(self, outer, holes=()):
        self.build.append(["add_face", [pbits(p) for p in outer], [[pbits(p) for p in h] for h in holes]])
        return self.mesh.add_face(outer, [list(h) for h in holes] or None)

    def add_edge(self, a, b):
        try:
            self.mesh.add_edge(a, b)
        except ValueError:
            self.build.append(["add_edge", pbits(a), pbits(b), "degenerate"])
            return
        self.build.append(["add_edge", pbits(a), pbits(b)])

    def set_attrs(self, f, attrs):
        f.attrs = dict(attrs)
        self.build.append(["attrs", ORD["f"][id(f)], attrs])

    def remove_face(self, f):
        self.build.append(["remove_face", ORD["f"][id(f)]])
        self.mesh.remove_face(f)

    # -- calling
    def call(self, name, args, fn=None, mut=False, tag=None):
        fn = fn or getattr(T, name)
        before = json.dumps(state(self.mesh)) if mut else None
        try:
            ret = fn(*[a.py for a in args])
        except Exception as exc:  # noqa: BLE001
            if mut:
                raise
            return ("raised", exc)
        rec = {"c": [name, *[a.js for a in args]], "r": enc(ret)}
        if mut:
            CHANGED[name] = CHANGED.get(name, [0, 0])
            CHANGED[name][0] += 1
            CHANGED[name][1] += json.dumps(state(self.mesh)) != before
        if tag == "fp":
            self.footprints.append(ret)
        if mut:
            rec["s"] = state(self.mesh)
        self.calls.append(rec)
        return ret


def lerp(p, q, t):
    return V(p.x() + (q.x() - p.x()) * t, p.y() + (q.y() - p.y()) * t, p.z() + (q.z() - p.z()) * t)


def rect_pts(plane, x0, y0, x1, y1, c):
    """Corners of a rectangle in an axis plane (counter-clockwise in (a, b))."""
    if plane == "z":
        mk = lambda a, b: V(a, b, c)
    elif plane == "x":
        mk = lambda a, b: V(c, a, b)
    else:
        mk = lambda a, b: V(a, c, b)
    return [mk(x0, y0), mk(x1, y0), mk(x1, y1), mk(x0, y1)]


def coord(r, jitter=0.08):
    c = r.randint(0, 12) * 0.25
    roll = r.random()
    if roll < jitter:
        c += r.choice([1e-5, -1e-5, 4e-5, -4.9e-5, 5e-5, 2e-4])
    return c


def build_rects(s: Scene):
    r = s.r
    plane = r.choice(["z", "z", "z", "x", "y"])
    c = r.choice([0.0, 0.0, 0.0, 1.0])
    n = r.randint(2, 7)
    for _ in range(n):
        x0, y0 = coord(r), coord(r)
        w, h = r.randint(1, 8) * 0.25, r.randint(1, 8) * 0.25
        pts = rect_pts(plane, x0, y0, x0 + w, y0 + h, c + (1e-5 if r.random() < 0.05 else 0.0))
        if r.random() < 0.3:
            pts.reverse()
        holes = []
        if r.random() < 0.3 and w >= 0.75 and h >= 0.75:
            holes.append(rect_pts(plane, x0 + 0.25, y0 + 0.25, x0 + w * 0.5, y0 + h * 0.5, c))
            if r.random() < 0.5:   # a hole nested in the hole: add_face drops it
                holes.append(rect_pts(plane, x0 + 0.3, y0 + 0.3, x0 + w * 0.4, y0 + h * 0.4, c))
        s.add_face(pts, holes)
    if r.random() < 0.4:
        for _ in range(r.randint(1, 4)):
            s.add_edge(*[V(coord(r), coord(r), c) if plane == "z" else V(coord(r), c, coord(r)) for _ in range(2)])


def build_nested(s: Scene):
    """A big face, rectangles drawn inside it, faces filling holes."""
    r = s.r
    plane = r.choice(["z", "z", "x"])
    big = rect_pts(plane, 0, 0, 3, 3, 0.0)
    holes = []
    fills = []
    for i in range(r.randint(1, 4)):
        x0, y0 = r.randint(0, 4) * 0.5 + 0.25, r.randint(0, 4) * 0.5 + 0.25
        w = r.choice([0.25, 0.5, 0.75])
        hole = rect_pts(plane, x0, y0, x0 + w, y0 + w, 0.0)
        if i == 0 or not any(abs(h[0].x() - hole[0].x()) < 1.0 and abs(h[0].y() - hole[0].y()) < 1.0 for h in holes):
            holes.append(hole)
            fills.append(hole)
    s.add_face(big, holes)
    for f in fills:
        if r.random() < 0.8:
            s.add_face(f)
    if r.random() < 0.5:  # a face drawn over the mother without a hole (spurious overlap)
        s.add_face(rect_pts(plane, 0.5, 0.5, 2.5, 2.5, 0.0))
    if r.random() < 0.4:
        s.add_face(rect_pts(plane, 0, 0, 1.5, 1.5, 0.0))
    for f in list(s.mesh.faces):
        if r.random() < 0.3:
            s.set_attrs(f, {"color": [r.randint(0, 255), r.randint(0, 255), r.randint(0, 255)]})


def build_tjunction(s: Scene):
    r = s.r
    plane = r.choice(["z", "x", "y"])
    s.add_face(rect_pts(plane, 0, 0, 2, 1, 0.0))
    s.add_face(rect_pts(plane, 0, 1, 1, 2, 0.0))
    s.add_face(rect_pts(plane, 1, 1, 2, 2, 0.0))
    if r.random() < 0.6:
        s.add_face(rect_pts(plane, 2, 0, 3, 1.5, 0.0))
        s.add_face(rect_pts(plane, 2, 1.5, 3, 2.5, 0.0))
    for _ in range(r.randint(0, 3)):
        a, b = coord(r), coord(r)
        s.add_edge(*[rect_pts(plane, a, b, a + 1, b + 1, 0.0)[i] for i in (0, 1)])
    for f in list(s.mesh.faces):
        if r.random() < 0.3:
            s.set_attrs(f, {"mat": r.choice(["brick", "glass"])})


def build_edges(s: Scene):
    r = s.r
    flat = r.random() < 0.6
    for _ in range(r.randint(4, 14)):
        a = V(r.randint(0, 6) * 0.5, r.randint(0, 6) * 0.5, 0.0 if flat else r.randint(0, 2) * 0.5)
        b = V(r.randint(0, 6) * 0.5, r.randint(0, 6) * 0.5, 0.0 if flat else r.randint(0, 2) * 0.5)
        s.add_edge(a, b)
    for _ in range(r.randint(0, 3)):
        x, y = r.randint(0, 4) * 0.5, r.randint(0, 4) * 0.5
        s.add_face(rect_pts("z", x, y, x + 1, y + 0.5, 0.0))


def build_warped(s: Scene):
    r = s.r
    for _ in range(r.randint(1, 4)):
        x, y = r.randint(0, 4) * 0.5, r.randint(0, 4) * 0.5
        n = r.choice([4, 4, 5, 6])
        pts = []
        for k in range(n):
            ang = 2 * math.pi * k / n
            rad = 0.5 + 0.3 * (k % 2)
            pts.append(V(x + rad * math.cos(ang), y + rad * math.sin(ang), 0.0))
        k = r.randrange(n)
        pts[k] = V(pts[k].x(), pts[k].y(), r.choice([0.002, 0.05, 0.3, 0.8]))
        if r.random() < 0.3:
            pts[(k + 2) % n] = V(pts[(k + 2) % n].x(), pts[(k + 2) % n].y(), r.choice([0.02, -0.2]))
        s.add_face(pts)
    if r.random() < 0.5:
        s.add_face(rect_pts("z", 0, 0, 1, 1, 0.0))


def build_box(s: Scene):
    r = s.r
    x0, y0, z0 = coord(r) * 0.5, coord(r) * 0.5, 0.0
    w, d, h = r.choice([1.0, 2.0]), r.choice([1.0, 1.5]), r.choice([0.5, 1.0])
    x1, y1, z1 = x0 + w, y0 + d, z0 + h
    P = V
    faces = [
        [P(x0, y0, z0), P(x0, y1, z0), P(x1, y1, z0), P(x1, y0, z0)],
        [P(x0, y0, z1), P(x1, y0, z1), P(x1, y1, z1), P(x0, y1, z1)],
        [P(x0, y0, z0), P(x1, y0, z0), P(x1, y0, z1), P(x0, y0, z1)],
        [P(x1, y0, z0), P(x1, y1, z0), P(x1, y1, z1), P(x1, y0, z1)],
        [P(x1, y1, z0), P(x0, y1, z0), P(x0, y1, z1), P(x1, y1, z1)],
        [P(x0, y1, z0), P(x0, y0, z0), P(x0, y0, z1), P(x0, y1, z1)],
    ]
    for f in faces:
        if r.random() < 0.15:
            f = list(reversed(f))
        s.add_face(f)
    if r.random() < 0.6:   # a stacked block on the top
        s.add_face(rect_pts("z", x0 + 0.25, y0 + 0.25, x0 + 0.75, y0 + 0.75, z1))
    if r.random() < 0.4:   # a face on a wall
        s.add_face(rect_pts("y", x0 + 0.2, z0 + 0.1, x0 + 0.6, z0 + 0.4, y0))
    if r.random() < 0.3:
        s.add_face(rect_pts("z", x0 - 0.5, y0 + 0.25, x0 + 0.5, y0 + 0.75, z1))


def build_rotated(s: Scene):
    """Planar polygons in random planes at float coordinates (non-axis)."""
    r = s.r
    for _ in range(r.randint(1, 3)):
        o = [r.uniform(-5, 5) for _ in range(3)]
        a = [r.gauss(0, 1) for _ in range(3)]
        la = math.sqrt(sum(x * x for x in a))
        u = [x / la for x in a]
        b = [r.gauss(0, 1) for _ in range(3)]
        dp = sum(i * j for i, j in zip(u, b))
        b = [j - dp * i for i, j in zip(u, b)]
        lb = math.sqrt(sum(x * x for x in b))
        w = [x / lb for x in b]
        k = r.choice([3, 4, 4, 5, 6])
        pts = []
        for i in range(k):
            ang = 2 * math.pi * i / k + r.uniform(-0.2, 0.2)
            rad = r.uniform(0.5, 3.0)
            x, y = rad * math.cos(ang), rad * math.sin(ang)
            pts.append(V(*[o[j] + x * u[j] + y * w[j] for j in range(3)]))
        s.add_face(pts)


def build_mother(s: Scene):
    """A big face with the faces of its subdivision drawn over it (a redundant
    mother), orphan edges lying over other edges."""
    r = s.r
    plane = r.choice(["z", "z", "x", "y"])
    s.add_face(rect_pts(plane, 0, 0, 3, 3, 0.0))
    if r.random() < 0.6:   # strictly inside: what the heal's mother test reads
        quads = [(0.25, 0.25, 1.5, 1.5), (1.5, 0.25, 2.75, 1.5), (1.5, 1.5, 2.75, 2.75), (0.25, 1.5, 1.5, 2.75)]
    else:
        quads = [(0, 0, 1.5, 1.5), (1.5, 0, 3, 1.5), (1.5, 1.5, 3, 3), (0, 1.5, 1.5, 3)]
    r.shuffle(quads)
    for q in quads[: r.choice([1, 2, 3, 4, 4])]:
        pts = rect_pts(plane, *q, 0.0)
        if r.random() < 0.2:
            pts.reverse()
        s.add_face(pts)
    if r.random() < 0.5:
        s.add_edge(*rect_pts(plane, 0, 0, 2, 0, 0.0)[:2])
        s.add_edge(*rect_pts(plane, 0.5, 0, 1, 0, 0.0)[:2])
    if r.random() < 0.4:
        s.add_edge(*rect_pts(plane, 0, 3, 3, 3, 0.0)[:2])
        s.add_edge(*rect_pts(plane, 1, 3, 2.5, 3, 0.0)[:2])
    for f in list(s.mesh.faces):
        if r.random() < 0.4:
            s.set_attrs(f, {"color": [r.randint(0, 255), 0, 0]})


def build_partial(s: Scene):
    """A wall with a door piece drawn over it (partial overlap), pieces
    reversed, a window inside."""
    r = s.r
    plane = r.choice(["z", "x"])
    s.add_face(rect_pts(plane, 0, 0, 4, 3, 0.0))
    for q in r.sample([(3, 0, 4, 2), (3, 0, 5, 2), (1, 1, 2, 2), (0, 2, 1.5, 3), (2.5, 0.5, 3.5, 1.5), (1, 0, 2, 1)], r.randint(1, 3)):
        pts = rect_pts(plane, *q, 0.0)
        if r.random() < 0.3:
            pts.reverse()
        s.add_face(pts)
    if r.random() < 0.4:
        s.add_face(rect_pts(plane, 0, 0, 4, 3, 1.0))


SCENES = [(build_rects, 4), (build_nested, 4), (build_tjunction, 3), (build_edges, 3), (build_warped, 4),
          (build_box, 3), (build_rotated, 2), (build_mother, 4), (build_partial, 4)]


# ---- Call makers ---------------------------------------------------------------------------

def rpoint(s: Scene):
    r = s.r
    m = s.mesh
    roll = r.random()
    if roll < 0.35 and m.vertices:
        return V(r.choice(m.vertices).position)
    if roll < 0.55 and m.edges:
        e = r.choice(m.edges)
        return lerp(e.a, e.b, r.choice([0.25, 0.5, 0.75, 0.3333]))
    if roll < 0.7:
        return V(coord(s.r), coord(s.r), r.choice([0.0, 0.0, 0.5]))
    if roll < 0.85 and m.vertices:
        p = r.choice(m.vertices).position
        return V(p.x() + r.choice([1e-5, -3e-5, 6e-5, 2e-4]), p.y(), p.z())
    return V(r.uniform(-3, 5), r.uniform(-3, 5), r.uniform(-1, 2))


def rface(s: Scene):
    return s.r.choice(s.mesh.faces) if s.mesh.faces else None


def redge(s: Scene):
    return s.r.choice(s.mesh.edges) if s.mesh.edges else None


def boundary_point(s: Scene, f):
    r = s.r
    vs = f.vertices
    i = r.randrange(len(vs))
    if r.random() < 0.5:
        return V(vs[i])
    return lerp(vs[i], vs[(i + 1) % len(vs)], r.choice([0.25, 0.5, 0.75]))


def sub_loop(s: Scene, f):
    """A loop sharing a corner or a side with face ``f``, or lying inside it
    (rectangular faces): built on its first corner and its sides."""
    r = s.r
    vs = f.vertices
    if len(vs) < 4:
        return None
    p0, p1, p3 = vs[0], vs[1], vs[3]
    t, u = r.choice([0.25, 0.5, 0.75, 1.0]), r.choice([0.25, 0.5, 0.75, 1.0])
    kind = r.choice(["corner", "side", "inner", "band", "over"])
    def at(a, b):
        return V(p0.x() + (p1.x() - p0.x()) * a + (p3.x() - p0.x()) * b,
                 p0.y() + (p1.y() - p0.y()) * a + (p3.y() - p0.y()) * b,
                 p0.z() + (p1.z() - p0.z()) * a + (p3.z() - p0.z()) * b)
    if kind == "corner":
        pts = [at(0, 0), at(t, 0), at(t, u), at(0, u)]
    elif kind == "side":
        pts = [at(0, 0), at(t, 0), at(t, u), at(0, u)] if r.random() < 0.5 else [at(1 - t, 0), at(1, 0), at(1, u), at(1 - t, u)]
    elif kind == "inner":
        pts = [at(0.1, 0.1), at(0.1 + 0.2 * t, 0.1), at(0.1 + 0.2 * t, 0.1 + 0.2 * u), at(0.1, 0.1 + 0.2 * u)]
    elif kind == "band":
        pts = [at(0, 0.25 * t), at(1, 0.25 * t), at(1, 0.25 * t + 0.25), at(0, 0.25 * t + 0.25)]
    else:
        pts = [at(-0.25, 0.25), at(0.5, 0.25), at(0.5, 0.75), at(-0.25, 0.75)]
    if r.random() < 0.3:
        pts.reverse()
    return pts


def poly_in_plane(r, k=None, planar=True):
    k = k or r.choice([3, 4, 5, 6, 7])
    o = [r.uniform(-4, 4) for _ in range(3)]
    if r.random() < 0.5:
        u, w = [1, 0, 0], [0, 1, 0]
    else:
        a = [r.gauss(0, 1) for _ in range(3)]
        la = math.sqrt(sum(x * x for x in a))
        u = [x / la for x in a]
        b = [r.gauss(0, 1) for _ in range(3)]
        dp = sum(i * j for i, j in zip(u, b))
        b = [j - dp * i for i, j in zip(u, b)]
        lb = math.sqrt(sum(x * x for x in b))
        w = [x / lb for x in b]
    pts = []
    for i in range(k):
        ang = 2 * math.pi * i / k + r.uniform(-0.3, 0.3)
        rad = r.uniform(0.4, 3.0)
        x, y = rad * math.cos(ang), rad * math.sin(ang)
        pts.append([o[j] + x * u[j] + y * w[j] for j in range(3)])
    if not planar:
        j = r.randrange(k)
        n = [u[1] * w[2] - u[2] * w[1], u[2] * w[0] - u[0] * w[2], u[0] * w[1] - u[1] * w[0]]
        d = r.choice([1e-4, 5e-4, 2e-3, 0.05])
        pts[j] = [pts[j][i] + n[i] * d for i in range(3)]
    return [V(*p) for p in pts]


MAKERS: list = []


def maker(weight=1):
    def deco(fn):
        MAKERS.append((fn, weight))
        return fn
    return deco


@maker(2)
def m_same_position(s):
    a = rpoint(s)
    b = V(a.x() + s.r.choice([0, 1e-5, 5e-5, 6e-5, 1e-3]), a.y(), a.z() + s.r.choice([0, 0, 4e-5]))
    s.call("same_position", [P(a), P(b)])


@maker(2)
def m_is_planar(s):
    r = s.r
    pts = poly_in_plane(r, planar=r.random() < 0.6)
    args = [PL(pts)]
    if r.random() < 0.4:
        args.append(D(r.choice([1e-4, 1e-3, 1e-2])))
    s.call("is_planar", args)


@maker(4)
def m_segment_intersection(s):
    r = s.r
    roll = r.random()
    if roll < 0.35:
        c = V(r.uniform(-3, 3), r.uniform(-3, 3), r.uniform(-3, 3))
        d1 = V(r.gauss(0, 1), r.gauss(0, 1), r.gauss(0, 1))
        d2 = V(r.gauss(0, 1), r.gauss(0, 1), r.gauss(0, 1))
        t1, t2 = r.uniform(0.2, 2), r.uniform(0.2, 2)
        s1, s2 = r.uniform(0.2, 2), r.uniform(0.2, 2)
        skew = r.choice([0.0, 0.0, 0.0, 5e-5, 3e-4])
        p1, p2 = c - d1 * t1, c + d1 * s1
        p3, p4 = c - d2 * t2 + V(0, 0, skew), c + d2 * s2 + V(0, 0, skew)
    elif roll < 0.55:
        a, b, c2 = rpoint(s), rpoint(s), rpoint(s)
        p1, p2, p3, p4 = a, b, c2, rpoint(s)
    elif roll < 0.7:   # shared endpoint / touching
        a, b, c2 = rpoint(s), rpoint(s), rpoint(s)
        p1, p2, p3, p4 = a, b, lerp(a, b, r.choice([0.0, 0.5, 1.0])), c2
    elif roll < 0.85:   # guide lines
        p1, p2 = V(-10000, 0.5, 0), V(10000, 0.5, 0)
        x = r.uniform(-2, 2)
        p3, p4 = V(x, -10000, 0), V(x, 10000, 0)
    else:   # parallel / collinear
        a = rpoint(s)
        d = V(r.choice([1, 0.5, 2]), r.choice([0, 1]), 0)
        p1, p2 = a, a + d
        p3, p4 = a + d * r.choice([0.5, 1, 3]), a + d * r.choice([2, 4])
    args = [P(p1), P(p2), P(p3), P(p4)]
    if r.random() < 0.3:
        args.append(D(r.choice([1e-4, 1e-3, 1e-6])))
    s.call("segment_intersection", args)


@maker(2)
def m_order_along(s):
    r = s.r
    a, b = rpoint(s), rpoint(s)
    pts = [lerp(a, b, r.choice([0.2, 0.5, 0.8, 0.5, 0.0, 1.0, 0.3, 0.7])) for _ in range(r.randint(0, 6))]
    pts = [V(p.x() + r.choice([0, 0, 2e-5]), p.y(), p.z()) for p in pts]
    s.call("_order_along", [P(a), P(b), PL(pts)])


@maker(3)
def m_2d(s):
    r = s.r
    k = r.choice([3, 4, 5, 6])
    poly = []
    for i in range(k):
        ang = 2 * math.pi * i / k
        rad = r.uniform(0.5, 2)
        poly.append((rad * math.cos(ang), rad * math.sin(ang)))
    if r.random() < 0.4:
        poly = [(float(r.randint(0, 4)), float(r.randint(0, 4))) for _ in range(k)]
    if r.random() < 0.5:
        p = (float(r.randint(0, 4)), float(r.randint(0, 4)))
    else:
        p = (r.uniform(-2, 4), r.uniform(-2, 4))
    s.call("_strictly_inside_2d", [P2(p), P2L(poly)])
    s.call("_point_inside_2d", [P2(p), P2L(poly)])
    a, b = poly[0], poly[1]
    q = (a[0] + (b[0] - a[0]) * r.choice([0.0, 0.5, 1.0, 1.5]), a[1] + (b[1] - a[1]) * r.choice([0.5, 0.5, 1.0]))
    s.call("_on_segment_2d", [P2(q), P2(a), P2(b)])


@maker(3)
def m_edge_queries(s):
    es = s.mesh.edges
    if not es:
        return
    r = s.r
    e = redge(s)
    a, b = (e.a, e.b) if r.random() < 0.6 else (rpoint(s), rpoint(s))
    if r.random() < 0.5:
        a, b = b, a
    s.call("find_duplicate_edge", [ERL(es), P(a), P(b)])
    s.call("find_smallest_cycle_through", [ERL(es), P(a), P(b)] + ([I(r.choice([3, 4, 5]))] if r.random() < 0.3 else []))
    s.call("find_cycles_through", [ERL(es), P(a), P(b)] + ([I(r.choice([1, 2, 3]))] if r.random() < 0.5 else []))
    s.call("plan_edge_split", [ERL(es), P(a), P(b)])


def _graph_names(lst):
    return [ORD["e"][id(e)] for e in lst]


@maker(2)
def m_cycle_graph(s):
    """A CycleGraph kept up to date edge by edge, queried through it."""
    r = s.r
    es = list(s.mesh.edges)
    if len(es) < 2:
        return
    first = es[: r.randint(1, len(es))]
    g = T.CycleGraph(first)
    ops = [["init", _graph_names(first)]]
    cur = list(first)
    rest = [e for e in es if e not in cur]
    for _ in range(r.randint(1, 6)):
        if rest and r.random() < 0.5:
            e = rest.pop(r.randrange(len(rest)))
            g.add(e)
            cur.append(e)
            ops.append(["add", ORD["e"][id(e)]])
        elif cur:
            e = cur.pop(r.randrange(len(cur)))
            g.remove(e)
            ops.append(["remove", ORD["e"][id(e)]])
    qa = redge(s)
    a, b = (qa.a, qa.b) if r.random() < 0.7 else (rpoint(s), rpoint(s))
    mr = r.choice([1, 2])
    ret = T.find_cycles_through(cur, a, b, mr, graph=g)
    plain = T.find_cycles_through(cur, a, b, mr)
    assert [[pbits(p) for p in c] for c in ret] == [[pbits(p) for p in c] for c in plain]
    s.calls.append({"c": ["cycle_graph", ops, pbits(a), pbits(b), mr], "r": enc(ret)})


@maker(2)
def m_face_exists(s):
    f = rface(s)
    if f is None:
        return
    cyc = list(f.vertices)
    if s.r.random() < 0.4:
        s.r.shuffle(cyc)
    if s.r.random() < 0.3:
        cyc = cyc[:-1]
    s.call("face_exists", [FRL(s.mesh.faces), PL(cyc)])


@maker(4)
def m_containment(s):
    f = rface(s)
    if f is None:
        return
    r = s.r
    loop = sub_loop(s, f) or [rpoint(s) for _ in range(4)]
    s.call("loop_inside_face", [FR(f), PL(loop)] + ([B(True)] if r.random() < 0.5 else []))
    ex = [FR(r.choice(s.mesh.faces))] if r.random() < 0.4 else []
    s.call("find_containing_face", [FRL(s.mesh.faces), PL(loop)] + ex)
    if r.random() < 0.5:
        s.call("_g_inside_a_hole", [FR(f), PL(loop)])
    p = V(r.choice(f.vertices).x() * 0.5 + f.centroid().x() * 0.5, f.centroid().y(), f.centroid().z())
    s.call("_point_in_face_solid", [FR(f), P(p if r.random() < 0.5 else f.centroid())])
    holed = [g for g in s.mesh.faces if g.hole_loops]
    if holed:
        g = r.choice(holed)
        hole = [v.position for v in r.choice(g.hole_loops)]
        c = V(sum(p.x() for p in hole) / len(hole), sum(p.y() for p in hole) / len(hole), sum(p.z() for p in hole) / len(hole))
        shrunk = [lerp(p, c, 0.3) for p in hole]
        s.call("_g_inside_a_hole", [FR(g), PL(shrunk)])
        s.call("_g_inside_a_hole", [FR(g), PL(hole)])
        s.call("_point_in_face_solid", [FR(g), P(c)])
        s.call("loop_inside_face", [FR(g), PL(shrunk), B(True)])
        s.call("loop_inside_face", [FR(g), PL(shrunk)])


@maker(2)
def m_orphans(s):
    es, fs = s.mesh.edges, s.mesh.faces
    r = s.r
    if not es:
        return
    vs = r.sample(s.mesh.vertices, min(len(s.mesh.vertices), r.randint(1, 4)))
    s.call("orphaned_edges_at", [ERL(es), FRL(fs), PL([v.position for v in vs])])


@maker(3)
def m_push(s):
    f = rface(s)
    if f is None:
        return
    r = s.r
    vs = f.vertices
    i = r.randrange(len(vs))
    a, b = vs[i], vs[(i + 1) % len(vs)]
    if r.random() < 0.4:
        a, b = lerp(a, b, 0.25), lerp(a, b, 0.75)
    s.call("classify_push_edge", [FR(f), P(a), P(b), FRL(s.mesh.faces)])
    s.call("face_is_bordered", [FR(f), FRL(s.mesh.faces)])
    g = rface(s)
    s.call("_segment_on_face_boundary", [P(a), P(b), FR(g)])
    s.call("_faces_coplanar", [P(f.normal()), P(g.normal())])
    s.call("_unit_coplanar", [P(f.normal()), P(g.normal())])
    s.call("_point_on_seg_incl", [P(rpoint(s)), P(a), P(b)])
    s.call("_point_on_segment_3d", [P(lerp(a, b, r.choice([0.0, 0.5, 1.0, 0.00001, 0.5]))), P(a), P(b)])


@maker(3)
def m_refine(s):
    f = rface(s)
    if f is None:
        return
    r = s.r
    pts = [v.position for v in s.mesh.vertices] + [lerp(f.vertices[0], f.vertices[1], 0.5)]
    pts = r.sample(pts, min(len(pts), 8))
    s.call("refine_loop_with_points", [PL(f.vertices), PL(pts)] + ([D(1e-3)] if r.random() < 0.2 else []))


@maker(5)
def m_chords(s):
    f = rface(s)
    if f is None:
        return
    r = s.r
    a, b = boundary_point(s, f), boundary_point(s, f)
    s.call("split_face_by_chord", [FR(f), P(a), P(b)])
    s.call("find_chord_split", [FRL(s.mesh.faces), P(a), P(b)])
    e = redge(s)
    if e is not None:
        mid = lerp(e.a, e.b, 0.5)
        skip = [] if r.random() < 0.5 else [a, b]
        s.call("split_edge_in_faces", [FRL(s.mesh.faces), P(e.a), P(e.b), P(mid), PL(skip)])
    s.call("_locate_on_loop", [PL(f.vertices), P(a)])


@maker(5)
def m_subtract(s):
    f = rface(s)
    if f is None:
        return
    loop = sub_loop(s, f)
    if loop is None:
        return
    s.call("find_subdividing_chain", [FR(f), PL(loop)])
    s.call("subtract_loop_from_face", [FR(f), PL(loop)])
    s.call("carve_loop_by_chords", [FR(f), PL(loop)])
    chain = T.find_subdividing_chain(f, loop)
    if chain is not None:
        s.call("split_face_by_chain", [FR(f), PL(chain)])


@maker(3)
def m_offset(s):
    r = s.r
    roll = r.random()
    f = rface(s)
    if roll < 0.5 and f is not None:
        loop, normal = f.vertices, f.normal()
    else:
        loop = poly_in_plane(r, planar=True)
        normal = tmp_face(loop).normal()
    d = r.choice([0.05, 0.1, -0.1, 0.3, 0.8, -0.25, 0.0, 0.001, 1.2, 0.4])
    s.call("offset_loop", [PL(loop), P(normal), D(d)])
    if r.random() < 0.5:
        s.call("max_offset_distance", [PL(loop), P(normal)] + ([D(r.choice([1.0, -1.0]))] if r.random() < 0.5 else []))


@maker(3)
def m_maximal_holes(s):
    r = s.r
    holes = []
    plane = r.choice(["z", "x"])
    for _ in range(r.randint(1, 5)):
        x0, y0 = r.randint(0, 6) * 0.25, r.randint(0, 6) * 0.25
        w = r.choice([0.25, 0.5, 1.0, 1.5])
        holes.append(rect_pts(plane, x0, y0, x0 + w, y0 + w * r.choice([1, 0.5]), 0.0))
    if r.random() < 0.3:
        holes.append(list(reversed(holes[0])))
    res = T._maximal_holes(holes)
    keep = [i for i, h in enumerate(holes) if any(h is k for k in res)]
    s.calls.append({"c": ["maximal_holes", [[pbits(p) for p in h] for h in holes]], "r": keep})
    # also loop_inside_loop and interior_point on those
    if len(holes) >= 2:
        n = V(0, 0, 1) if plane == "z" else V(1, 0, 0)
        s.call("_loop_inside_loop", [PL(holes[0]), PL(holes[1]), P(n)])
        s.call("_interior_point", [PL(holes[1]), P(n)])


@maker(3)
def m_polys(s):
    r = s.r
    pts = poly_in_plane(r, planar=True)
    if r.random() < 0.4:
        # bow tie
        pts = [V(0, 0, 0), V(1, 1, 0), V(1, 0, 0), V(0, 1, 0)] if r.random() < 0.5 else [V(0, 0, 0), V(2, 0, 0), V(2, 1, 0), V(1, 1, 0), V(1, -1, 0)]
    nrm = tmp_face(pts).normal()
    verts = tmp_verts(pts)
    # _loop_crosses_itself takes mesh Vertex objects: record by positions
    ret = T._loop_crosses_itself(verts, nrm)
    s.calls.append({"c": ["_loop_crosses_itself", [pbits(p) for p in pts], pbits(nrm)], "r": ret})
    s.call("_interior_point", [PL(pts), P(nrm)])
    s.call("_loop_inside_loop", [PL(pts[:3] if len(pts) > 3 else pts), PL(pts), P(nrm)])


@maker(2)
def m_collinear(s):
    r = s.r
    es = s.mesh.edges
    if len(es) >= 2 and r.random() < 0.6:
        e1, e2 = r.choice(es), r.choice(es)
        s.call("_collinear_overlapping", [ER(e1), ER(e2)])
    else:
        a = rpoint(s)
        d = V(1, 0, 0)
        e1 = (a, a + d * r.choice([1, 2]))
        e2 = (a + d * r.choice([0.5, 1.5, 2.5]) + V(0, r.choice([0, 1e-5, 5e-4, 2e-3]), 0), a + d * 3)
        T_e = types.SimpleNamespace
        s.calls.append({"c": ["_collinear_overlapping_pts", pbits(e1[0]), pbits(e1[1]), pbits(e2[0]), pbits(e2[1])],
                        "r": T._collinear_overlapping(T_e(a=e1[0], b=e1[1]), T_e(a=e2[0], b=e2[1]))})


@maker(2)
def m_defects(s):
    m = s.mesh
    if not m.vertices:
        return
    r = s.r
    vs = r.sample(m.vertices, min(len(m.vertices), r.randint(1, 6)))
    before = T.defects_near(vs)
    s.call("defects_near", [VRL(vs)])
    s.calls.append({"c": ["new_defects", list(before.values()), list(T.defects_near(m.vertices).values())],
                    "r": T.new_defects(before, T.defects_near(m.vertices))})
    if len(m.faces) >= 2:
        f, g = r.choice(m.faces), r.choice(m.faces)
        s.call("_coplanar_overlap", [FR(f), FR(g)])


@maker(2)
def m_flat(s):
    s.call("_mesh_is_flat", [A(s.mesh, None)])
    fp = s.call("winding_footprint", [A(s.mesh, None)], tag="fp")
    if isinstance(fp, tuple) and fp and fp[0] == "raised":
        return
    f = rface(s)
    if f is not None and fp is not None:
        s.call("_footprint_normal", [A(fp, ["fp", len(s.footprints) - 1]), FR(f)])


# ---- Mutators ----

def _fp_arg(s):
    if s.footprints and s.footprints[-1] is not None and s.r.random() < 0.6:
        return A(s.footprints[-1], ["fp", len(s.footprints) - 1])
    return A(None, None)


@maker(3)
def m_orient(s):
    if not s.mesh.faces:
        return
    fp = _fp_arg(s)
    s.call("orient_coplanar_faces", [A(s.mesh, None), fp], mut=True)


@maker(3)
def m_sweep(s):
    r = s.r
    roll = r.random()
    if roll < 0.4:
        s.call("sweep_tjunctions", [A(s.mesh, None)], mut=True)
    elif roll < 0.8 and s.mesh.vertices:
        keys = {T._key(r.choice(s.mesh.vertices).position) for _ in range(r.randint(1, 3))}
        s.call("sweep_tjunctions", [A(s.mesh, None), A(keys, [[h64(c) for c in k] for k in sorted(keys)])], mut=True)
    else:
        s.call("resolve_tjunctions", [A(s.mesh, None)], mut=True)


@maker(4)
def m_fold(s):
    s.call("fold_nonplanar_faces", [A(s.mesh, None)] + ([D(r) for r in [s.r.choice([1e-3, 1e-4, 0.05])]] if s.r.random() < 0.3 else []), mut=True)


@maker(2)
def m_prune(s):
    s.call("prune_collinear_orphan_edges", [A(s.mesh, None)], mut=True)


@maker(8)
def m_heal(s):
    r = s.r
    if r.random() < 0.4:
        s.call("winding_footprint", [A(s.mesh, None)], tag="fp")
    fp = _fp_arg(s)
    args = [A(s.mesh, None)]
    cov = r.choice([0.5, 0.5, 0.3, 0.9])
    partial = r.choice([None, None, True, False])
    args += [D(cov), A(partial, partial), fp]
    s.call("heal_overlapping_faces", args, mut=True)


def run_scene(seed: int, steps: int):
    r0 = random.Random(seed)
    builders, weights = zip(*SCENES)
    kind = r0.choices(builders, weights=weights)[0]
    s = Scene(seed)
    kind(s)
    makers, mw = zip(*MAKERS)
    for _ in range(steps):
        r0.choices(makers, weights=mw)[0](s)
    return s


def dump(s: Scene) -> dict:
    return {"build": s.build, "calls": s.calls}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    n_scenes = int(sys.argv[1]) if len(sys.argv) > 1 else 420
    ncalls = 0
    kinds: dict = {}
    with open(OUT / "topology_cases.jsonl", "w", encoding="ascii", newline="\n") as fh:
        seed = 20261021
        done = 0
        while done < n_scenes:
            seed += 1
            r0 = random.Random(seed)
            try:
                s = run_scene(seed, r0.randint(5, 14))
            except Exception as exc:  # noqa: BLE001 — a mutator raised: drop the scene
                print("scene", seed, "dropped:", type(exc).__name__, exc)
                continue
            fh.write(json.dumps(dump(s), separators=(",", ":")) + "\n")
            done += 1
            for c in s.calls:
                kinds[c["c"][0]] = kinds.get(c["c"][0], 0) + 1
            ncalls += len(s.calls)
    (OUT / ".gitattributes").write_text("# Bits Python produced; compared exactly.\n* -text\n")
    print("scenes", n_scenes, "calls", ncalls, "bytes", (OUT / "topology_cases.jsonl").stat().st_size)
    print("mutators that changed the mesh:", CHANGED)
    for k in sorted(kinds):
        print(f"  {k}: {kinds[k]}")


if __name__ == "__main__":
    main()
