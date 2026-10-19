# SPDX-License-Identifier: GPL-3.0-or-later
"""Fixtures that hold ``ingetrazo-geom``'s fillet and vertex-bevel planning
to ``core/fillet.py`` and ``core/vertex_bevel.py``, bit for bit
(docs/rust-rewrite-plan.md, M1.5 and M1.7).

Small meshes (boxes, turned and squashed boxes, prisms, an L, a pyramid, a
tetrahedron, folded sheets at awkward angles, a box with a hole through
it, loose edges), each dumped as plain topology; then the REAL
``plan_fillet`` / ``plan_vertex_bevel`` run on many choices of edges or
corners, radii, measures, segment counts, profiles, corner styles and
drawn shapes. Every float is written as exact bits (float32 ``>f``,
float64 ``>d``); a point is its three float32 in one 24-hex string; a
refusal is its message. Also ``math.hypot`` and ``bevel_profiles.sample``,
which the Rust copy of those needs. Committed, so cargo test needs no
Python; rerun only to widen them:

    PYTHONPATH=. python scripts/golden/make_fillet_fixtures.py

Python iterates a vertex's edges (a set) and a vertex's faces (a set of
objects) in hash order; the planners' results depend on that order in
places, so this script fixes both: a vertex lists its edges, and
``Vertex.faces()`` returns its faces, in ascending index. The Rust side
takes the same orders as input.

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

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6 import __version__ as pyside_version
from PySide6.QtCore import qVersion
from PySide6.QtGui import QMatrix4x4, QVector3D

from core import bevel_profiles
from core.fillet import plan_fillet
from core.mesh import Mesh, Vertex
from core.vertex_bevel import plan_vertex_bevel

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "rust" / "crates" / "ingetrazo-geom" / "tests" / "data"
BUDGET = 2_050_000          # bytes of fixtures, about


def h32(x: float) -> str:
    return struct.pack(">f", x).hex()


def h64(x: float) -> str:
    return struct.pack(">d", x).hex()


def pt(p: QVector3D) -> str:
    return h32(p.x()) + h32(p.y()) + h32(p.z())


def loop_bits(lp) -> list:
    return [pt(p) for p in lp]


P = QVector3D


# ---- the vertex/face orders -------------------------------------------------

_FIDX: dict = {}


def _faces_sorted(self):
    out = {}
    for e in self.edges:
        for f in e.faces:
            out[id(f)] = f
    return sorted(out.values(), key=lambda f: _FIDX[id(f)])


Vertex.faces = _faces_sorted


# ---- meshes -----------------------------------------------------------------

def box_loops(L=2.0, W=1.0, H=1.0):
    return [
        [P(0, 0, 0), P(0, W, 0), P(L, W, 0), P(L, 0, 0)],
        [P(0, 0, H), P(L, 0, H), P(L, W, H), P(0, W, H)],
        [P(0, 0, 0), P(L, 0, 0), P(L, 0, H), P(0, 0, H)],
        [P(L, 0, 0), P(L, W, 0), P(L, W, H), P(L, 0, H)],
        [P(L, W, 0), P(0, W, 0), P(0, W, H), P(L, W, H)],
        [P(0, W, 0), P(0, 0, 0), P(0, 0, H), P(0, W, H)],
    ]


def prism_loops(foot, h):
    n = len(foot)
    loops = [list(reversed([P(x, y, 0) for x, y in foot])), [P(x, y, h) for x, y in foot]]
    for i in range(n):
        (ax, ay), (bx, by) = foot[i], foot[(i + 1) % n]
        loops.append([P(ax, ay, 0), P(bx, by, 0), P(bx, by, h), P(ax, ay, h)])
    return loops


def tunnel(L, W, H, hl, hw):
    """A slab with a rectangular hole through it (faces with holes)."""
    x0, x1 = (L - hl) / 2, (L + hl) / 2
    y0, y1 = (W - hw) / 2, (W + hw) / 2
    outer = box_loops(L, W, H)
    top_hole = [P(x0, y0, H), P(x0, y1, H), P(x1, y1, H), P(x1, y0, H)]       # clockwise from above
    bot_hole = [P(x0, y0, 0), P(x1, y0, 0), P(x1, y1, 0), P(x0, y1, 0)]
    faces = [(outer[0], [bot_hole]), (outer[1], [top_hole])] + [(lp, None) for lp in outer[2:]]
    inner = [
        [P(x0, y0, 0), P(x0, y0, H), P(x1, y0, H), P(x1, y0, 0)],     # faces +Y (into the hole)
        [P(x1, y0, 0), P(x1, y0, H), P(x1, y1, H), P(x1, y1, 0)],
        [P(x1, y1, 0), P(x1, y1, H), P(x0, y1, H), P(x0, y1, 0)],
        [P(x0, y1, 0), P(x0, y1, H), P(x0, y0, H), P(x0, y0, 0)],
    ]
    return faces + [(lp, None) for lp in inner]


def pyramid():
    b = [P(0, 0, 0), P(2, 0, 0), P(2, 2, 0), P(0, 2, 0)]
    a = P(1, 1, 1.2)
    return [list(reversed(b))] + [[b[i], b[(i + 1) % 4], a] for i in range(4)]


def tetra():
    a, b, c, d = P(0, 0, 0), P(2, 0, 0), P(0.5, 1.6, 0), P(0.7, 0.5, 1.8)
    return [[a, c, b], [a, b, d], [b, c, d], [c, a, d]]


def fold(theta_deg, n=1):
    """``n`` quads in a row; the first two meet at ``theta_deg`` (open ends)."""
    t = math.radians(theta_deg)
    q0 = [P(0, 0, 0), P(2, 0, 0), P(2, 1, 0), P(0, 1, 0)]
    c, s = math.cos(t), math.sin(t)
    q1 = [P(0, 1, 0), P(2, 1, 0), P(2, 1 + c, s), P(0, 1 + c, s)]
    faces = [q0, q1]
    if n > 2:
        faces.append([P(0, 1 + c, s), P(2, 1 + c, s), P(2, 1 + 2 * c, 2 * s), P(0, 1 + 2 * c, 2 * s)])
    return faces


def slanted():
    return [[P(0, 0, 0), P(0, 1, 0), P(2, 1, 0), P(2.5, 0, 0)],
            [P(0, 0, 1), P(2.5, 0, 1), P(2, 1, 1), P(0, 1, 1)],
            [P(0, 0, 0), P(2.5, 0, 0), P(2.5, 0, 1), P(0, 0, 1)],
            [P(2.5, 0, 0), P(2, 1, 0), P(2, 1, 1), P(2.5, 0, 1)],
            [P(2, 1, 0), P(0, 1, 0), P(0, 1, 1), P(2, 1, 1)],
            [P(0, 1, 0), P(0, 0, 0), P(0, 0, 1), P(0, 1, 1)]]


def build(faces, xf=None, extra=None, paint=None, rng=None):
    """A Mesh from ``faces`` ([loop] or (loop, holes)), optionally mapped
    through ``xf``; ``extra`` adds loose edges."""
    m = Mesh()
    mp = (lambda p: xf.map(p)) if xf is not None else (lambda p: P(p))
    for f in faces:
        loop, holes = f if isinstance(f, tuple) else (f, None)
        m.add_face([mp(p) for p in loop], [[mp(p) for p in h] for h in holes] if holes else None)
    for a, b in extra or []:
        m.add_edge(mp(a), mp(b))
    palette = [(0.5, 0.2, 0.2), (0.1, 0.9, 0.1), "wood", None]
    for i, f in enumerate(m.faces):
        if paint == "same":
            f.attrs["color"] = palette[0]
        elif paint == "mixed":
            c = palette[rng.randrange(4)]
            if c is not None:
                f.attrs["color"] = c
            if rng.random() < 0.3:
                f.attrs["opacity"] = 0.5
    for i, e in enumerate(m.edges):
        e.layer = f"E{i}"
    for i, f in enumerate(m.faces):
        _FIDX[id(f)] = i
    eidx = {id(e): i for i, e in enumerate(m.edges)}
    for v in m.vertices:
        v.edges = sorted(v.edges, key=lambda e: eidx[id(e)])
    return m


def dump(m: Mesh, mid: int) -> dict:
    vidx = {id(v): i for i, v in enumerate(m.vertices)}
    fidx = {id(f): i for i, f in enumerate(m.faces)}
    eidx = {id(e): i for i, e in enumerate(m.edges)}
    return {
        "t": "mesh", "id": mid,
        "v": [pt(v.position) for v in m.vertices],
        "ve": [[eidx[id(e)] for e in v.edges] for v in m.vertices],
        "e": [[vidx[id(e.v0)], vidx[id(e.v1)], [fidx[id(f)] for f in e.faces]] for e in m.edges],
        "f": [{"l": [vidx[id(v)] for v in f.loop],
               "h": [[vidx[id(v)] for v in h] for h in f.hole_loops],
               "p": [[k, repr(f.attrs[k])] for k in ("color", "texture", "mat", "opacity")
                     if k in f.attrs]} for f in m.faces],
    }


def meshes(rng: random.Random):
    out = []

    def add(name, faces, **kw):
        out.append((name, build(faces, rng=rng, **kw)))

    def xform():
        xf = QMatrix4x4()
        xf.translate(rng.uniform(-9, 9), rng.uniform(-9, 9), rng.uniform(-3, 3))
        xf.scale(rng.choice([1.0, 1.0, 0.5, 1.7]), rng.choice([1.0, 0.8, 1.0]), rng.choice([1.0, 0.5]))
        xf.rotate(rng.uniform(0, 360), QVector3D(rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(-1, 1)))
        return xf

    add("box", box_loops(), paint="same")
    add("box_mixed", box_loops(2.0, 1.0, 1.0), paint="mixed")
    add("cube3", box_loops(3.0, 3.0, 3.0))
    add("slab", box_loops(4.0, 3.0, 0.3), paint="same")
    add("post", box_loops(0.4, 0.4, 3.0))
    for _ in range(4):
        L, W, H = (rng.choice([0.5, 1.0, 1.5, 2.0, 2.4]) for _ in range(3))
        add(f"box{L}x{W}x{H}", box_loops(L, W, H), paint="mixed")
    for k in range(6):
        add(f"turned{k}", box_loops(*(rng.choice([1.0, 2.0, 1.5]) for _ in range(3))), xf=xform(), paint="mixed")
    add("issue74", box_loops(2.0, 2.0, 2.0), xf=_issue74())
    turn = QMatrix4x4()
    turn.rotate(33.0, QVector3D(1, 1, 0))
    add("cube2_turned", box_loops(2.0, 2.0, 2.0), xf=turn)
    for n in (3, 5, 6, 8, 12):
        foot = [(math.cos(2 * math.pi * i / n) * 1.5, math.sin(2 * math.pi * i / n) * 1.5) for i in range(n)]
        add(f"prism{n}", prism_loops(foot, 1.2), paint="same" if n % 2 else "mixed")
    add("prism_irregular", prism_loops([(0, 0), (2, 0), (2.4, 1.1), (1, 1.9), (-0.3, 1.0)], 1.0))
    add("prism_thin_wedge", prism_loops([(0, 0), (3, 0), (3, 0.15)], 1.0))
    add("L", prism_loops([(0, 0), (2, 0), (2, 1), (1, 1), (1, 2), (0, 2)], 1.0), paint="same")
    add("U", prism_loops([(0, 0), (3, 0), (3, 2), (2, 2), (2, 1), (1, 1), (1, 2), (0, 2)], 1.5))
    add("slanted", slanted())
    add("pyramid", pyramid())
    add("tetra", tetra())
    add("tunnel", tunnel(4.0, 3.0, 1.0, 1.2, 1.0), paint="same")
    add("tunnel_turned", tunnel(4.0, 3.0, 1.0, 1.2, 1.0), xf=xform())
    for th in (3, 20, 60, 90, 120, 160, 178, 90):
        add(f"fold{th}", fold(th, 3 if th in (90, 120) else 2))
    add("quad", [[P(0, 0, 0), P(2, 0, 0), P(2, 1, 0), P(0, 1, 0)]])
    add("triangle", [[P(0, 0, 0), P(2, 0, 0), P(0.4, 1.3, 0)]])
    add("box_loose", box_loops(), extra=[(P(5, 5, 5), P(6, 5, 5)), (P(0, 0, 0), P(-1, 0, 0))])
    post = box_loops() + [
        [P(0, 0, 1), P(0, 0, 2), P(0, -1, 2), P(0, -1, 1)],
        [P(0, 0, 1), P(1, 0, 1), P(1, 0, 2), P(0, 0, 2)]]
    add("box_post", post)
    return out


def _issue74():
    xf = QMatrix4x4()
    xf.translate(8.5, -3.0, 0.4)
    xf.scale(1.0, 1.0, 0.5)
    xf.rotate(33.0, QVector3D(0.3, 1.0, 0.45))
    return xf


# ---- cases ------------------------------------------------------------------

def shape_choices(rng):
    r = rng.random()
    if r < 0.5:
        return None
    if r < 0.84:
        return rng.choice(["steps", "supports", "cornice", "crown"])
    if r < 0.92:
        return [(1.0, 0.0, False), (1.0, 0.3, False), (0.6, 0.6, True), (0.3, 0.9, True), (0.0, 1.0, False)]
    if r < 0.97:
        return [(1.0, 0.0, False), (0.0, 1.0, False)]
    return [(1.0, 0.0, False), (0.5, 0.0, False), (0.5, 0.5, False), (0.0, 0.5, False), (0.0, 1.0, False)]


def shape_bits(shape):
    if shape is None or isinstance(shape, str):
        return shape
    return [[h64(x), h64(y), bool(s)] for x, y, s in shape]


def pick_edges(m: Mesh, rng):
    r = rng.random()
    if rng.random() < 0.95:
        # an edge between exactly two faces, mostly; the refusals get their share below
        pool = [e for e in m.edges if len(e.faces) == 2] or list(m.edges)
    else:
        pool = list(m.edges)
    n = len(pool)
    if r < 0.30:
        return [pool[rng.randrange(n)]]
    if r < 0.50:
        return list(pool)
    if r < 0.62:
        f = m.faces[rng.randrange(len(m.faces))]
        lp = f.loop
        es = [m.find_edge(lp[i], lp[(i + 1) % len(lp)]) for i in range(len(lp))]
        rng.shuffle(es)
        return es
    if r < 0.85:
        return rng.sample(pool, min(n, rng.randint(2, 6)))
    v = m.vertices[rng.randrange(len(m.vertices))]
    es = list(v.edges)
    return es if rng.random() < 0.6 else es[:2] if len(es) > 1 else es


def scale_of(m: Mesh) -> float:
    es = [e for e in m.edges if e.faces]
    return min((e.length() for e in es), default=1.0)


def out_of_plan(plan, m: Mesh) -> dict:
    eidx = {id(e): i for i, e in enumerate(m.edges)}
    fidx = {id(f): i for i, f in enumerate(m.faces)}
    strip = plan.face_attrs.get("strip")
    return {
        "removed_faces": [fidx[id(f)] for f in plan.removed_faces],
        "removed_edges": [eidx[id(e)] for e in plan.removed_edges],
        "rebuilt": [[fidx[id(f)], loop_bits(lp), [loop_bits(h) for h in hs]]
                    for f, (lp, hs, _a, _i) in zip(plan.removed_faces, plan.rebuilt)],
        "strips": [loop_bits(x) for x in plan.strips],
        "patches": [loop_bits(x) for x in plan.patches],
        "soft": [[pt(p), pt(q)] for p, q in plan.soft],
        "curves": [loop_bits(x) for x in plan.curves],
        "strip_attrs": None if strip is None else [[k, repr(v)] for k, v in strip.items()],
        "flags": [[pt(p), pt(q), int(fl[2][1:])] for p, q, fl in plan.flags],
    }


def run(fn, *args):
    try:
        return fn(*args)
    except Exception as exc:        # a Python crash is not a fixture
        return exc


def fillet_case(m: Mesh, mid: int, rng):
    edges = pick_edges(m, rng)
    s = scale_of(m)
    radius = s * rng.choice([0.02, 0.05, 0.08, 0.1, 0.15, 0.2, 0.2, 0.3, 0.45, 0.6, 3.0])
    radius = float(struct.unpack("<f", struct.pack("<f", radius))[0]) if rng.random() < 0.5 else radius
    heavy = len(edges) > 4
    segments = rng.choice([1, 2, 3, 4]) if heavy else rng.choice([1, 2, 3, 4, 6, 8, 12])
    profile = rng.choice([0.5] * 10 + [0.25, 0.7, 0.9, 0.1, 0.33, 0.5000000001, 0.6, 0.15] * 2 + [1.0, 0.0, -0.2])
    measure = rng.choice(["radius", "radius", "offset", "width", "depth", "nonsense"])
    corners = rng.choice(["round", "round", "cut", "nonsense"])
    shape = shape_choices(rng)
    if shape is not None and isinstance(shape, str) and rng.random() < 0.02:
        shape = "unknown"
    res = run(plan_fillet, m, edges, radius, segments, profile, measure, corners, shape)
    if isinstance(res, Exception):
        return None
    eidx = {id(e): i for i, e in enumerate(m.edges)}
    row = {"t": "fillet", "mesh": mid, "edges": [eidx[id(e)] for e in edges],
           "radius": h64(radius), "segments": segments, "profile": h64(profile),
           "measure": measure, "corners": corners, "shape": shape_bits(shape)}
    row["out"] = {"err": res} if isinstance(res, str) else {"ok": out_of_plan(res, m)}
    return row


def bevel_case(m: Mesh, mid: int, rng):
    n = len(m.vertices)
    r = rng.random()
    if r < 0.4:
        verts = [m.vertices[rng.randrange(n)]]
    elif r < 0.5:
        verts = list(m.vertices)
    else:
        verts = rng.sample(m.vertices, min(n, rng.randint(2, 4)))
    s = scale_of(m)
    size = s * rng.choice([0.02, 0.05, 0.1, 0.1, 0.2, 0.3, 0.3, 0.45, 0.6, 1.0, 3.0])
    heavy = len(verts) > 3
    segments = rng.choice([1, 2, 3]) if heavy else rng.choice([1, 2, 3, 4, 6, 8])
    profile = rng.choice([0.5] * 8 + [0.25, 0.7, 0.9, 0.1, 0.33, 0.6, 1.0])
    res = run(plan_vertex_bevel, m, verts, size, segments, profile)
    if isinstance(res, Exception):
        return None
    vidx = {id(v): i for i, v in enumerate(m.vertices)}
    row = {"t": "bevel", "mesh": mid, "verts": [vidx[id(v)] for v in verts],
           "size": h64(size), "segments": segments, "profile": h64(profile)}
    row["out"] = {"err": res} if isinstance(res, str) else {"ok": out_of_plan(res, m)}
    return row


def sweep_rows(ms):
    """The three edges at a corner of a box, and sometimes every edge, with
    every shape and corner style: the corner patches (spheres,
    superellipsoids, drawn products, steps, flat cuts) are the part random
    picks rarely reach."""
    by_name = dict((n, (i, m)) for i, (n, m) in enumerate(ms))
    shapes = [None, "steps", "supports", "cornice", "crown",
              [(1.0, 0.0, False), (1.0, 0.3, False), (0.6, 0.6, True), (0.3, 0.9, True), (0.0, 1.0, False)]]
    rows = []

    def one(mid, m, edges, rad, segments, profile, measure, corners, shape):
        res = run(plan_fillet, m, edges, rad, segments, profile, measure, corners, shape)
        if isinstance(res, Exception):
            return
        eidx = {id(e): i for i, e in enumerate(m.edges)}
        rows.append({"t": "fillet", "mesh": mid, "edges": [eidx[id(e)] for e in edges],
                     "radius": h64(rad), "segments": segments, "profile": h64(profile),
                     "measure": measure, "corners": corners, "shape": shape_bits(shape),
                     "out": {"err": res} if isinstance(res, str) else {"ok": out_of_plan(res, m)}})

    for name, rad, verts, only in (("box", 0.12, (0, 6), None), ("cube3", 0.2, (0,), (None, "steps", "supports"))):
        mid, m = by_name[name]
        for vi in verts:
            at = list(m.vertices[vi].edges)
            for shape in shapes:
                if only is not None and shape not in only:
                    continue
                fixed = isinstance(shape, str) and shape in ("cornice", "crown")    # ignore the segment count
                for corners in ("round", "cut"):
                    for segments in ((3,) if fixed else (2, 3, 4) if shape is None else (2, 3)):
                        if shape is None:
                            for profile, measure in ((0.5, "offset"), (0.5, "radius"), (0.3, "width"), (0.8, "depth")):
                                one(mid, m, at, rad, segments, profile, measure, corners, shape)
                        else:
                            one(mid, m, at, rad, segments, 0.5, "offset", corners, shape)
        if name == "box":
            for shape in (None, "steps", "supports"):
                for corners in ("round", "cut"):
                    one(mid, m, list(m.edges), rad, 2, 0.5, "offset", corners, shape)
    # corners of a turned cube, where an asymmetric outline falls back to the blended grid
    mid, m = by_name["cube2_turned"]
    for vi in range(len(m.vertices)):
        one(mid, m, list(m.vertices[vi].edges), 0.2, 2, 0.5, "offset", "round", "cornice")
    # a whole 3 m cube with the shapes whose corners fall back to the blended grid
    mid, m = by_name["cube3"]
    for shape in ("supports", "crown", "steps"):
        one(mid, m, list(m.edges), 0.2, 3, 0.5, "offset", "round", shape)
    return rows


def side_rows(rng):
    rows = []
    for _ in range(400):
        mag = 10.0 ** rng.uniform(-8, 3)
        x, y = rng.uniform(-1, 1) * mag, rng.uniform(-1, 1) * mag * rng.choice([1, 1, 1e-3, 1e3])
        rows.append({"t": "hypot", "x": h64(x), "y": h64(y), "r": h64(math.hypot(x, y))})
    shapes = ["steps", "supports", "cornice", "crown"]
    custom = [[(1.0, 0.0, False), (1.0, 0.3, False), (0.6, 0.6, True), (0.3, 0.9, True), (0.0, 1.0, False)],
              [(1.0, 0.0, True), (0.7, 0.2, True), (0.4, 0.6, True), (0.0, 1.0, True)],
              [(1.0, 0.0, False), (0.0, 1.0, False)]]
    for sh in shapes + custom:
        for segs in (1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 16):
            pts = bevel_profiles.sample(sh if isinstance(sh, str) else tuple(sh), segs)
            rows.append({"t": "sample", "shape": shape_bits(sh), "segments": segs,
                         "pts": [[h64(x), h64(y)] for x, y in pts]})
    return rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = random.Random(20261019)
    meta = {"qt": qVersion(), "pyside": pyside_version, "platform": sys.platform,
            "python": sys.version.split()[0]}
    lines = [json.dumps({"meta": meta})]
    size = len(lines[0])
    counts = {"mesh": 0, "fillet": 0, "bevel": 0, "ok": 0, "err": 0}
    ms = meshes(rng)
    rows = []
    for mid, (name, m) in enumerate(ms):
        rows.append(json.dumps(dump(m, mid), separators=(",", ":")))
        counts["mesh"] += 1
    size += sum(len(r) + 1 for r in rows)
    lines += rows
    for r in side_rows(rng):
        s = json.dumps(r, separators=(",", ":"))
        lines.append(s)
        size += len(s) + 1
    for r in sweep_rows(ms):
        s = json.dumps(r, separators=(",", ":"))
        lines.append(s)
        size += len(s) + 1
        counts["fillet"] += 1
        counts["ok" if "ok" in r["out"] else "err"] += 1
    attempts = 0
    while size < BUDGET and attempts < 20000:
        attempts += 1
        mid = rng.randrange(len(ms))
        m = ms[mid][1]
        row = (fillet_case if rng.random() < 0.55 else bevel_case)(m, mid, rng)
        if row is None:
            continue
        s = json.dumps(row, separators=(",", ":"))
        if len(s) > 120_000:
            continue
        lines.append(s)
        size += len(s) + 1
        counts[row["t"]] += 1
        counts["ok" if "ok" in row["out"] else "err"] += 1
    with open(OUT / "fillet.jsonl", "w", encoding="ascii", newline="\n") as fh:
        fh.write("\n".join(lines) + "\n")
    gi = OUT / ".gitattributes"
    if not gi.exists():
        gi.write_text("* -text\n", encoding="ascii")
    print("written to", (OUT / "fillet.jsonl").relative_to(ROOT), size, "bytes", counts, meta)


if __name__ == "__main__":
    main()
