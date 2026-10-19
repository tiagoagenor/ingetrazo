# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
"""Fixtures for ``ingetrazo-geom``'s ``sweep`` and ``simplify`` modules
(docs/rust-rewrite-plan.md, M1.6 and M1.9).

They run the REAL Python — ``core.sweep`` (rings, previews, the faces
``sweep_profile`` hands ``Mesh.add_face``, dragged paths, ``orient_closed_path``)
and ``formats.fuse.simplify_mesh`` on meshes built with
``core.mesh.Mesh.add_face`` — over many inputs, and write inputs and outputs
as exact bits (float32 as ``>f`` hex, float64 as ``>d`` hex). Committed, so
``cargo test`` needs no Python; rerun only to widen them:

    PYTHONPATH=. QT_QPA_PLATFORM=offscreen PYTHONUTF8=1 \\
        python scripts/golden/make_sweep_fixtures.py

``sweep_profile`` itself needs the live mesh for the stitch, so it is run with
``run_stitch`` and ``orient_outward`` replaced by recorders and ``add_face``
spied on: what is compared is exactly what the geometry hands the mesh step.

Deleted with the Python app at M10.
"""
from __future__ import annotations

import json
import math
import random
import struct
import sys
from pathlib import Path

from PySide6 import __version__ as pyside_version
from PySide6.QtCore import qVersion

import core.sweep as sweep
import core.topology as topology
from core.mesh import Mesh
from core.vecmath import QVector3D
from formats import fuse

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "rust" / "crates" / "ingetrazo-geom" / "tests" / "data"

QV = QVector3D


def f32(x: float) -> float:
    return struct.unpack("<f", struct.pack("<f", x))[0]


def h32(x: float) -> str:
    return struct.pack(">f", x).hex()


def h64(x: float) -> str:
    return struct.pack(">d", x).hex()


def pb(p) -> list:
    return [h32(p.x()), h32(p.y()), h32(p.z())]


def pbs(pts) -> list:
    return [pb(p) for p in pts]


class Digest:
    """FNV-1a 64 over a stream of u32 words, mirrored in the Rust test."""

    def __init__(self) -> None:
        self.h = 0xCBF29CE484222325

    def word(self, w: int) -> None:
        for shift in (0, 8, 16, 24):
            self.h ^= (w >> shift) & 0xFF
            self.h = (self.h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF

    def pt(self, p) -> None:
        for c in (p.x(), p.y(), p.z()):
            self.word(struct.unpack("<I", struct.pack("<f", c))[0])

    def poly(self, outer, holes) -> None:
        self.word(len(outer))
        for p in outer:
            self.pt(p)
        self.word(len(holes))
        for h in holes:
            self.word(len(h))
            for p in h:
                self.pt(p)

    def hex(self) -> str:
        return "%016x" % self.h


def digest_polys(polys) -> str:
    d = Digest()
    d.word(len(polys))
    for outer, holes in polys:
        d.poly(outer, holes)
    return d.hex()


class Gen:
    def __init__(self, seed: int) -> None:
        self.r = random.Random(seed)

    def unit(self):
        while True:
            v = [self.r.gauss(0, 1) for _ in range(3)]
            n = math.sqrt(sum(x * x for x in v))
            if n > 1e-3:
                return [x / n for x in v]

    def axis_or_unit(self):
        if self.r.random() < 0.4:
            return list(self.r.choice([(0, 0, 1), (0, 0, -1), (0, 1, 0), (1, 0, 0), (0, -1, 0), (-1, 0, 0)]))
        return self.unit()

    def frame(self, n):
        """Two unit vectors perpendicular to n and to each other."""
        a = (1.0, 0.0, 0.0) if abs(n[0]) < 0.9 else (0.0, 1.0, 0.0)
        u = [a[1] * n[2] - a[2] * n[1], a[2] * n[0] - a[0] * n[2], a[0] * n[1] - a[1] * n[0]]
        ln = math.sqrt(sum(x * x for x in u))
        u = [x / ln for x in u]
        v = [n[1] * u[2] - n[2] * u[1], n[2] * u[0] - n[0] * u[2], n[0] * u[1] - n[1] * u[0]]
        return u, v

    def num(self, lo=-6.0, hi=6.0):
        r = self.r.random()
        if r < 0.2:
            return float(self.r.randint(int(lo), int(hi)))
        if r < 0.4:
            return round(self.r.uniform(lo, hi), self.r.randint(1, 3))
        return self.r.uniform(lo, hi)


def at(o, u, v, x, y, w=None, z=0.0):
    return QV(*(o[k] + u[k] * x + v[k] * y for k in range(3)))


def shapes(g: Gen, kind: str):
    """2-D outlines (outer, holes) in profile coordinates."""
    s = g.r.choice([0.3, 0.5, 1.0, 2.0, 3.7])
    if kind == "rect":
        w, h = s * g.r.choice([1, 1.5, 2]), s
        return [(0, 0), (w, 0), (w, h), (0, h)], []
    if kind == "centered":
        return [(-s, -s / 2), (s, -s / 2), (s, s / 2), (-s, s / 2)], []
    if kind == "tri":
        return [(0, 0), (s * 2, 0), (s * 0.7, s * 1.6)], []
    if kind == "L":
        return [(0, 0), (2 * s, 0), (2 * s, s), (s, s), (s, 2 * s), (0, 2 * s)], []
    if kind == "holed":
        return ([(0, 0), (4 * s, 0), (4 * s, 3 * s), (0, 3 * s)],
                [[(s, s), (s, 2 * s), (2 * s, 2 * s), (2 * s, s)]])
    if kind == "twoholes":
        return ([(0, 0), (6 * s, 0), (6 * s, 3 * s), (0, 3 * s)],
                [[(s, s), (s, 2 * s), (2 * s, 2 * s), (2 * s, s)],
                 [(3 * s, s), (3 * s, 2 * s), (4 * s, 2 * s), (4 * s, s)]])
    if kind == "circle":
        m = g.r.choice([6, 8, 12, 24])
        return [(s * math.cos(2 * math.pi * i / m), s * math.sin(2 * math.pi * i / m)) for i in range(m)], []
    if kind == "ring":
        m = g.r.choice([8, 12])
        return ([(2 * s * math.cos(2 * math.pi * i / m), 2 * s * math.sin(2 * math.pi * i / m)) for i in range(m)],
                [[(s * math.cos(2 * math.pi * i / m), s * math.sin(2 * math.pi * i / m)) for i in reversed(range(m))]])
    if kind == "random":
        m = g.r.randint(3, 7)
        ang = sorted(g.r.uniform(0, 2 * math.pi) for _ in range(m))
        return [(g.r.uniform(0.4, 1.6) * s * math.cos(a), g.r.uniform(0.4, 1.6) * s * math.sin(a)) for a in ang], []
    if kind == "sliver":
        return [(0, 0), (s, 0), (s, 1e-4 * s), (0, 1e-4 * s)], []
    raise KeyError(kind)


PROFILE_KINDS = ["rect", "centered", "tri", "L", "holed", "twoholes", "circle", "ring", "random", "sliver"]


def profile_in_plane(g: Gen, kind, o, n):
    u, v = g.frame(n)
    # a random in-plane twist
    t = g.r.uniform(0, 2 * math.pi)
    u, v = ([math.cos(t) * u[k] + math.sin(t) * v[k] for k in range(3)],
            [-math.sin(t) * u[k] + math.cos(t) * v[k] for k in range(3)])
    outer, holes = shapes(g, kind)
    return ([at(o, u, v, x, y) for x, y in outer], [[at(o, u, v, x, y) for x, y in h] for h in holes])


def make_face(outer, holes):
    mesh = Mesh()
    face = mesh.add_face(outer, holes or None)
    return mesh, face


def face_dump(face) -> dict:
    return {"o": pbs(face.vertices), "h": [pbs(h) for h in face.holes]}


def centroid_pt(face):
    return face.centroid()


def polyline(g: Gen, start, d0, k, turn_max=2.2):
    """k+1 points from start, first direction d0, random bends."""
    pts = [QV(*start)]
    d = list(d0)
    cur = list(start)
    for i in range(k):
        L = g.r.choice([0.5, 1.0, 2.0, 3.0, g.r.uniform(0.3, 5)])
        cur = [cur[j] + d[j] * L for j in range(3)]
        pts.append(QV(*cur))
        # bend: rotate d about a random perpendicular axis
        r = g.r.random()
        if r < 0.15:
            ang = 0.0
        elif r < 0.3:
            ang = g.r.choice([1e-3, 1e-4, 1e-5, 1e-6, 1e-7]) * g.r.choice([-1, 1])
        elif r < 0.55:
            ang = math.pi / 2 * g.r.choice([-1, 1])
        else:
            ang = g.r.uniform(-turn_max, turn_max)
        ax = g.unit() if g.r.random() < 0.6 else g.axis_or_unit()
        # Rodrigues, axis made perpendicular to d
        dot = sum(ax[j] * d[j] for j in range(3))
        ax = [ax[j] - dot * d[j] for j in range(3)]
        ln = math.sqrt(sum(x * x for x in ax))
        if ln < 1e-6:
            continue
        ax = [x / ln for x in ax]
        c, s = math.cos(ang), math.sin(ang)
        cr = [ax[1] * d[2] - ax[2] * d[1], ax[2] * d[0] - ax[0] * d[2], ax[0] * d[1] - ax[1] * d[0]]
        d = [d[j] * c + cr[j] * s for j in range(3)]
    return pts


def closed_polygon(g: Gen, kind, c, n, size):
    u, v = g.frame(n)
    if kind == "square":
        pts = [(-size, -size), (size, -size), (size, size), (-size, size)]
    elif kind == "rect":
        pts = [(-2 * size, -size), (2 * size, -size), (2 * size, size), (-2 * size, size)]
    elif kind == "tri":
        pts = [(-size, -size), (size, -size), (0, size)]
    elif kind == "L":
        pts = [(0, 0), (2 * size, 0), (2 * size, size), (size, size), (size, 2 * size), (0, 2 * size)]
    else:
        m = g.r.randint(5, 9)
        pts = [(size * math.cos(2 * math.pi * i / m), size * math.sin(2 * math.pi * i / m)) for i in range(m)]
    return [at(c, u, v, x, y) for x, y in pts]


# -- sweep_rings / previews / plan --------------------------------------------

def run_plan(outer, holes, path, closed):
    """The real sweep_profile with the stitch and orientation recorded."""
    mesh = Mesh()
    face = mesh.add_face(outer, holes or None)
    calls = []
    orig = mesh.add_face

    def spy(lp, hl=None):
        calls.append(([QV(p) for p in lp], [[QV(p) for p in h] for h in (hl or [])]))
        return orig(lp, hl)

    mesh.add_face = spy
    seeds = []
    real_stitch, real_orient = sweep.run_stitch, sweep.orient_outward
    sweep.run_stitch = lambda m, seed, *a, **k: seeds.append(set(seed))
    sweep.orient_outward = lambda m: None
    try:
        ok = sweep.sweep_profile(mesh, face, path, closed)
    finally:
        sweep.run_stitch, sweep.orient_outward = real_stitch, real_orient
    if not ok:
        return None
    return calls, (seeds[0] if seeds else set())


def sweep_case(face, path, closed) -> dict:
    case = {"k": "sweep", "face": face_dump(face), "path": pbs(path), "closed": closed}
    got = sweep.sweep_rings(face, path, closed)
    if got is None:
        case["rings"] = None
    else:
        rings, dirs, spans = got
        rd = Digest()
        rd.word(len(rings))
        npts = 0
        for ring in rings:
            rd.word(len(ring))
            for lp in ring:
                rd.word(len(lp))
                npts += len(lp)
                for p in lp:
                    rd.pt(p)
        case["rings"] = {"digest": rd.hex(), "n": len(rings), "pts": npts, "d": pbs(dirs), "s": spans}
        if npts <= 16:
            case["rings"]["r"] = [[pbs(lp) for lp in ring] for ring in rings]
    prev = sweep.sweep_preview_faces(face, path, closed)
    case["preview"] = digest_polys([(f.vertices, f.holes or []) for f in prev])
    case["preview_n"] = len(prev)
    outer = list(face.vertices)
    holes = [list(h) for h in face.holes]
    res = run_plan(outer, holes, path, closed)
    if res is None:
        case["plan"] = None
    else:
        calls, seed = res
        walls = len(calls) if closed else len(calls) - 2
        sg = Digest()
        for lp in [outer] + holes:
            n = len(lp)
            for i in range(n):
                sg.pt(lp[i])
                sg.pt(lp[(i + 1) % n])
        seeds = sorted((tuple(0.0 if c == 0 else c for c in k)) for k in seed)
        sd = Digest()
        for k in seeds:
            for c in k:
                b = struct.pack(">d", c)
                sd.word(int.from_bytes(b[0:4], "big"))
                sd.word(int.from_bytes(b[4:8], "big"))
        case["plan"] = {"digest": digest_polys(calls), "n": len(calls), "walls": walls,
                        "segs": sg.hex(), "seed_n": len(seeds), "seed": sd.hex()}
    return case


def sweep_cases(g: Gen, n: int):
    out = []
    # open paths
    for i in range(n):
        kind = g.r.choice(PROFILE_KINDS)
        o = [g.num(-8, 8) for _ in range(3)]
        nrm = g.axis_or_unit()
        outer, holes = profile_in_plane(g, kind, o, nrm)
        _m, face = make_face(outer, holes)
        c = face.centroid()
        start = [c.x(), c.y(), c.z()]
        if g.r.random() < 0.25:
            start = [start[j] + g.r.uniform(-0.5, 0.5) for j in range(3)]
        d0 = nrm if g.r.random() < 0.75 else g.unit()
        if g.r.random() < 0.5:
            d0 = [-x for x in d0]
        path = polyline(g, start, d0, g.r.choice([1, 1, 2, 3, 4, 6]))
        r = g.r.random()
        if r < 0.05 and len(path) >= 3:
            path[g.r.randrange(1, len(path))] = QV(path[0])  # a return trip / duplicate
        elif r < 0.08:
            path.append(QV(path[-1]))                         # zero-length span
        elif r < 0.12 and len(path) >= 3:
            # a 180 degree reversal
            p = path[1]
            path = [path[0], p, QV(path[0].x(), path[0].y(), path[0].z())]
        out.append(sweep_case(face, path, False))
    # closed polygon paths
    for i in range(n // 2):
        kind = g.r.choice(PROFILE_KINDS[:-1])
        c = [g.num(-5, 5) for _ in range(3)]
        nrm = g.axis_or_unit()
        size = g.r.choice([1.0, 2.0, 3.0, 0.7])
        pts = closed_polygon(g, g.r.choice(["square", "rect", "tri", "L", "poly"]), c, nrm, size)
        # profile perpendicular to the segment leaving vertex 0 (sometimes elsewhere)
        k = 0 if g.r.random() < 0.7 else g.r.randrange(len(pts))
        seg = pts[(k + 1) % len(pts)] - pts[k]
        dseg = [seg.x(), seg.y(), seg.z()]
        ln = math.sqrt(sum(x * x for x in dseg))
        dseg = [x / ln for x in dseg]
        o = [pts[k].x(), pts[k].y(), pts[k].z()]
        outer, holes = profile_in_plane(g, kind, o, dseg)
        _m, face = make_face(outer, holes)
        if g.r.random() < 0.6:
            pts = sweep.orient_closed_path(pts, face)
        out.append(sweep_case(face, pts, True))
    # circles: lathe (revolution) cases and near misses
    for i in range(n // 2):
        a = g.axis_or_unit()
        e1, e2 = g.frame(a)
        c = [g.num(-4, 4) for _ in range(3)]
        R = g.r.choice([1.0, 2.0, 5.0, g.r.uniform(0.5, 4)])
        m = g.r.choice([6, 8, 12, 16, 24, 32, 48])
        ph = g.r.choice([0.0, g.r.uniform(0, 6.28)])
        pts = [QV(*(c[j] + R * (math.cos(2 * math.pi * t / m + ph) * e1[j] + math.sin(2 * math.pi * t / m + ph) * e2[j])
                    for j in range(3))) for t in range(m)]
        if g.r.random() < 0.3:
            pts.reverse()
        # the profile's plane contains the axis and the first path point's radial (or another)
        rad_i = 0 if g.r.random() < 0.5 else g.r.randrange(m)
        rv = [pts[rad_i].x() - c[0], pts[rad_i].y() - c[1], pts[rad_i].z() - c[2]]
        dp = sum(rv[j] * a[j] for j in range(3))
        rv = [rv[j] - dp * a[j] for j in range(3)]
        ln = math.sqrt(sum(x * x for x in rv))
        rv = [x / ln for x in rv]
        mode = g.r.choice(["circle", "circle", "cross", "rect", "holed", "axis", "off", "tilted"])
        sx = g.r.uniform(0.1, 0.4) * R
        if mode == "circle":
            cx, cr = R * 0.0 + g.r.uniform(1.0, 1.6) * sx * 2, sx
            m2 = 12
            loop = [(cx + cr * math.cos(2 * math.pi * t / m2), cr * math.sin(2 * math.pi * t / m2)) for t in range(m2)]
            hl = []
        elif mode == "cross":
            loop = [(-sx, -sx), (sx, -sx), (sx, sx), (-sx, sx)]
            hl = []
        elif mode == "rect":
            loop = [(sx, 0), (2 * sx, 0), (2 * sx, 2 * sx), (sx, 2 * sx)]
            hl = []
        elif mode == "holed":
            loop = [(sx, 0), (5 * sx, 0), (5 * sx, 4 * sx), (sx, 4 * sx)]
            hl = [[(2 * sx, sx), (2 * sx, 3 * sx), (4 * sx, 3 * sx), (4 * sx, sx)]]
        elif mode == "axis":
            loop = [(0, 0), (2 * sx, 0), (2 * sx, 3 * sx), (0, 3 * sx)]
            hl = []
        else:  # off / tilted: not a lathe plane
            loop = [(sx, 0), (2 * sx, 0), (2 * sx, 2 * sx), (sx, 2 * sx)]
            hl = []
        # profile coordinates: x along rv, y along the axis
        sh = g.r.uniform(-2, 2) * sx if mode != "axis" else 0.0
        tw = 0.0
        if mode == "off":
            tw = g.r.choice([2e-3, 0.05, 0.3]) * R
        side = [a[1] * rv[2] - a[2] * rv[1], a[2] * rv[0] - a[0] * rv[2], a[0] * rv[1] - a[1] * rv[0]]
        tilt = g.r.choice([1e-4, 0.01, 0.2]) if mode == "tilted" else 0.0

        def pt(x, y):
            return QV(*(c[j] + rv[j] * x + a[j] * (y + sh) + side[j] * (tw + tilt * y) for j in range(3)))

        outer = [pt(x, y) for x, y in loop]
        holes = [[pt(x, y) for x, y in h] for h in hl]
        _m, face = make_face(outer, holes)
        out.append(sweep_case(face, pts, True))
    return out


def hypot_cases(g: Gen, n: int):
    out = []
    for i in range(n):
        r = g.r.random()
        if r < 0.3:
            a, b = (g.r.uniform(-10, 10), g.r.uniform(-10, 10))
        elif r < 0.5:
            a, b = (f32(g.r.uniform(-1, 1)), f32(g.r.uniform(-1, 1)))
        elif r < 0.65:
            a, b = (g.r.uniform(-1, 1) * 10 ** g.r.randint(-300, 300), g.r.uniform(-1, 1) * 10 ** g.r.randint(-300, 300))
        elif r < 0.8:
            a, b = (g.r.choice([0.0, -0.0, 1.0, 3.0, 5e-324, 1e-310]), g.r.uniform(-4, 4))
        else:
            a, b = (float(g.r.randint(-30, 30)), float(g.r.randint(-30, 30)))
        out.append({"k": "hypot", "a": h64(a), "b": h64(b), "r": h64(math.hypot(a, b))})
    return out


# -- paths, orientation, dragged paths ----------------------------------------

def order_cases(g: Gen, n: int):
    out = []
    for i in range(n):
        mesh = Mesh()
        shape = g.r.choice(["open", "open", "closed", "branch", "two", "single", "eight", "loop3"])
        m = g.r.randint(2, 9)
        pts = [QV(*(g.num(-5, 5) for _ in range(3))) for _ in range(m + 1)]
        segs = []
        if shape == "open":
            segs = [(k, k + 1) for k in range(m)]
        elif shape == "closed":
            segs = [(k, (k + 1) % (m + 1)) for k in range(m + 1)]
        elif shape == "loop3":
            segs = [(0, 1), (1, 2), (2, 0)]
        elif shape == "branch":
            segs = [(k, k + 1) for k in range(m)] + [(1, m)]
            if m < 3:
                segs = [(0, 1), (1, 2), (1, 3)]
                pts = pts + [QV(9, 9, 9), QV(8, 8, 8)]
        elif shape == "two":
            segs = [(k, k + 1) for k in range(m) if k != m // 2]
        elif shape == "single":
            segs = [(0, 1)]
        else:
            segs = [(0, 1), (1, 2), (2, 0), (2, 3), (3, 4), (4, 2)]
            pts = pts + [QV(7, 1, 2)] * 3
        used = sorted({v for s in segs for v in s})
        for p in pts:
            pass
        edges = []
        for a, b in segs:
            if a >= len(pts) or b >= len(pts):
                continue
            if g.r.random() < 0.5:
                a, b = b, a
            try:
                edges.append(mesh.add_edge(pts[a], pts[b]))
            except ValueError:
                pass
        g.r.shuffle(edges)
        if not edges:
            continue
        ids = {id(v): k for k, v in enumerate(mesh.vertices)}
        res = sweep.order_path_edges(edges)
        case = {"k": "order", "pos": pbs([v.position for v in mesh.vertices]),
                "edges": [[ids[id(e.v0)], ids[id(e.v1)]] for e in edges]}
        case["res"] = None if res is None else {"p": pbs(res[0]), "c": res[1]}
        out.append(case)
    return out


def orient_cases(g: Gen, n: int):
    out = []
    for i in range(n):
        kind = g.r.choice(PROFILE_KINDS[:-1])
        c = [g.num(-4, 4) for _ in range(3)]
        nrm = g.axis_or_unit()
        pts = closed_polygon(g, g.r.choice(["square", "rect", "tri", "L", "poly"]), c, nrm, g.r.choice([1.0, 2.0, 3.0]))
        if g.r.random() < 0.3:
            pts.reverse()
        k = g.r.randrange(len(pts))
        o = [pts[k].x() + g.r.choice([0, 0, 0.3]), pts[k].y(), pts[k].z()]
        outer, holes = profile_in_plane(g, kind, o, g.axis_or_unit())
        _m, face = make_face(outer, holes)
        if g.r.random() < 0.1:
            pts = pts[:2]
        out.append({"k": "orient", "face": face_dump(face), "pts": pbs(pts), "res": pbs(sweep.orient_closed_path(pts, face))})
    return out


def manual_cases(g: Gen, n: int):
    out = []
    # path start: an edge across the profile's plane, or not
    for i in range(n):
        kind = g.r.choice(PROFILE_KINDS[:-1])
        o = [g.num(-3, 3) for _ in range(3)]
        nrm = g.axis_or_unit()
        outer, holes = profile_in_plane(g, kind, o, nrm)
        _m, face = make_face(outer, holes)
        c = face.centroid()
        r = g.r.random()
        a = QV(*(c.x() + g.num(-3, 3), c.y() + g.num(-3, 3), c.z() + g.num(-3, 3)))
        b = QV(*(c.x() + g.num(-3, 3), c.y() + g.num(-3, 3), c.z() + g.num(-3, 3)))
        if r < 0.2:
            b = QV(a.x() + 1, a.y(), a.z())
        mesh = Mesh()
        try:
            e = mesh.add_edge(a, b)
        except ValueError:
            continue
        toward = None if g.r.random() < 0.4 else QV(*(g.num(-4, 4) for _ in range(3)))
        ids = {id(v): k for k, v in enumerate(mesh.vertices)}
        p, chain = sweep.manual_path_start(face, e, toward)
        out.append({"k": "mstart", "face": face_dump(face), "a": [ids[id(e.v0)], pb(e.v0.position)],
                    "b": [ids[id(e.v1)], pb(e.v1.position)],
                    "toward": None if toward is None else pb(toward),
                    "res": [None if p is None else pb(p), [ids[id(v)] for v in chain]]})
    # path extension over graphs
    for i in range(n // 2):
        mesh = Mesh()
        kind = g.r.choice(["line", "ring", "grid", "ladder", "tree"])
        if kind == "line":
            P = [QV(k, 0, 0) for k in range(g.r.randint(4, 10))]
            segs = [(k, k + 1) for k in range(len(P) - 1)]
        elif kind == "ring":
            m = g.r.randint(4, 10)
            P = [QV(math.cos(2 * math.pi * k / m), math.sin(2 * math.pi * k / m), 0) for k in range(m)]
            segs = [(k, (k + 1) % m) for k in range(m)]
        elif kind == "grid":
            w, hh = g.r.randint(2, 4), g.r.randint(2, 4)
            P = [QV(x, y, 0) for y in range(hh) for x in range(w)]
            segs = [(y * w + x, y * w + x + 1) for y in range(hh) for x in range(w - 1)]
            segs += [(y * w + x, (y + 1) * w + x) for y in range(hh - 1) for x in range(w)]
        elif kind == "ladder":
            m = g.r.randint(3, 6)
            P = [QV(k, 0, 0) for k in range(m)] + [QV(k, 1, 0) for k in range(m)]
            segs = [(k, k + 1) for k in range(m - 1)] + [(m + k, m + k + 1) for k in range(m - 1)] + [(0, m), (m - 1, 2 * m - 1)]
        else:
            m = g.r.randint(5, 10)
            P = [QV(k, 0, 0) for k in range(m)]
            segs = [(k, k + 1) for k in range(m - 1)] + [(2, m)]
            P.append(QV(2, 1, 0))
        for a, b in segs:
            mesh.add_edge(P[a], P[b])
        # deterministic incident order: the mesh's edge order
        for v in mesh.vertices:
            v.edges = [e for e in mesh.edges if e.v0 is v or e.v1 is v]
        vid = {id(v): k for k, v in enumerate(mesh.vertices)}
        eid = {id(e): k for k, e in enumerate(mesh.edges)}
        skip_set = set()
        if g.r.random() < 0.3:
            skip_set = {g.r.randrange(len(mesh.edges)) for _ in range(2)}
        skip = (lambda e: eid[id(e)] in skip_set) if skip_set else None
        start = g.r.randrange(len(mesh.vertices))
        chain = [mesh.vertices[start]]
        limit = g.r.choice([400, 400, 3, 6])
        steps = []
        for _ in range(g.r.randint(3, 14)):
            ei = g.r.randrange(len(mesh.edges))
            changed = sweep.manual_path_extend(chain, mesh.edges[ei], skip, limit)
            steps.append([ei, changed, [vid[id(v)] for v in chain]])
        out.append({"k": "mext", "pos": pbs([v.position for v in mesh.vertices]),
                    "edges": [[vid[id(e.v0)], vid[id(e.v1)]] for e in mesh.edges],
                    "skip": sorted(skip_set), "start": start, "limit": limit, "steps": steps})
    return out


def facet_cases(g: Gen, n: int):
    out = []
    for i in range(n):
        mesh = Mesh()
        ang = g.r.choice([0.0, 1e-3, 0.01, 0.05, 0.1, 0.3, 0.5, 0.55, 0.5536, 0.9, 1.5, g.r.uniform(0, 3)])
        a0 = g.unit()
        # two triangles with normals angle apart
        u, v = g.frame(a0)
        a1 = [math.cos(ang) * a0[k] + math.sin(ang) * u[k] for k in range(3)]
        tris = []
        for nrm in (a0, a1):
            uu, vv = g.frame(nrm)
            o = [g.num(-3, 3) for _ in range(3)]
            tris.append(mesh.add_face([at(o, uu, vv, 0, 0), at(o, uu, vv, 1, 0), at(o, uu, vv, 0, 1.3)]))
        n0, n1 = tris[0].normal(), tris[1].normal()
        d = QVector3D.dotProduct(n0.normalized(), n1.normalized())
        out.append({"k": "facet", "n0": pb(n0), "n1": pb(n1), "soft": bool(sweep._CURVE_FACET_COS < d < 0.99995)})
    return out


# -- simplify ----------------------------------------------------------------

class HoleWatch:
    """Notes whether Mesh.add_face's _maximal_holes dropped a hole."""

    def __init__(self) -> None:
        self.dropped = False
        self.real = topology._maximal_holes

    def __enter__(self):
        def watched(holes):
            res = self.real(holes)
            if len(res) != len(holes):
                self.dropped = True
            return res
        topology._maximal_holes = watched
        return self

    def __exit__(self, *exc):
        topology._maximal_holes = self.real


class AttrTable:
    """The per-attrs facts of formats.fuse._fuse_native, as ids."""

    def __init__(self) -> None:
        self.attrs: list = []
        self.sig_ids: dict = {}
        self.group_ids: dict = {}
        self.smooth_ids: dict = {}
        self.none_sig = fuse._attrs_sig(None)

    def index(self, attrs):
        if not attrs:
            return None
        for i, a in enumerate(self.attrs):
            if a == attrs:
                return i
        self.attrs.append(dict(attrs))
        return len(self.attrs) - 1

    def info(self, i):
        a = self.attrs[i]
        s = fuse._attrs_sig(a)
        if s == self.none_sig:
            sid = -2
        else:
            sid = self.sig_ids.setdefault(s, len(self.sig_ids))
        t = s[1]
        if t is not None and t[4]:
            grp = self.group_ids.setdefault((s[0], s[2], t[0]), len(self.group_ids))
            uvw = list(t[4])
        else:
            grp, uvw = -1, []
        sm = fuse._smooth_sig(a)
        smid = -2 if sm is None else self.smooth_ids.setdefault(sm, len(self.smooth_ids))
        return {"sig": sid, "grp": grp, "rank": 2 if t is not None else 1,
                "uvw": [h64(x) for x in uvw], "smooth": smid}


LAYERS = [None, None, "walls", "doors"]


def dump_mesh(mesh, table: AttrTable) -> dict:
    vid = {id(v): k for k, v in enumerate(mesh.vertices)}
    fid = {id(f): k for k, f in enumerate(mesh.faces)}
    layers = {n: i for i, n in enumerate(["walls", "doors"])}

    def attr_of(f):
        return table.index(f.attrs)

    return {
        "v": pbs([v.position for v in mesh.vertices]),
        "e": [[vid[id(e.v0)], vid[id(e.v1)], sorted(fid[id(f)] for f in e.faces),
               [e.soft, e.curve, None if e.layer is None else layers[e.layer], e.hidden]] for e in mesh.edges],
        "f": [[[vid[id(v)] for v in f.loop], [[vid[id(v)] for v in h] for h in f.hole_loops], f.interior,
               attr_of(f)] for f in mesh.faces],
    }


def edge_flag_list(mesh):
    layers = {n: i for i, n in enumerate(["walls", "doors"])}
    return [[pb(e.v0.position), pb(e.v1.position),
             [e.soft, e.curve, None if e.layer is None else layers[e.layer], e.hidden]] for e in mesh.edges]


ATTR_POOL = [
    None,
    {"color": (200, 30, 30)},
    {"color": (200, 30, 30)},
    {"color": (30, 200, 30)},
    {"color": (30, 30, 200), "layer": "a"},
    {"texture": {"path": "brick.png", "sw": 1.0, "sh": 1.0, "rot": 0, "uvw": [0.5, 0.25, 0.0, 0.1, 0.9, 0.0]}},
    {"texture": {"path": "brick.png", "sw": 1.0, "sh": 1.0, "rot": 0, "uvw": [0.505, 0.25, 0.0, 0.1, 0.9, 0.0]}},
    {"texture": {"path": "brick.png", "sw": 1.0, "sh": 1.0, "rot": 0, "uvw": [0.9, 0.25, 0.0, 0.1, 0.9, 0.0]}},
    {"color": (200, 30, 30), "texture": {"path": "wood.png", "sw": 2.0, "sh": 2.0, "rot": 90, "uvw": None}},
    {"tag": "x"},
]


def build_grid(g: Gen, plane, w, h, tilt, noise, split, colors):
    """A triangulated grid; ``tilt`` degrees between neighbouring rows of
    cells (a gently rounded patch), ``noise`` jitters heights."""
    mesh = Mesh()
    u, v = plane

    def P(x, y):
        # fold the grid about the x axis: row y rises by an arc of `tilt` degrees per row
        a = math.radians(tilt) * y
        return QV(*(u[k] * x + v[k] * (y * math.cos(a)) + [0, 0, 1][k] * (y * math.sin(a))
                    + g.r.uniform(-noise, noise) * (1 if noise else 0) for k in range(3)))

    grid = {(x, y): P(x, y) for x in range(w + 1) for y in range(h + 1)}
    for x in range(w):
        for y in range(h):
            a, b, c, d = grid[x, y], grid[x + 1, y], grid[x + 1, y + 1], grid[x, y + 1]
            at = ATTR_POOL[colors(x, y)] if colors else None
            if split == "ab" or (split == "mix" and g.r.random() < 0.5):
                tris = [(a, b, c), (a, c, d)]
            else:
                tris = [(a, b, d), (b, c, d)]
            for t in tris:
                f = mesh.add_face(t)
                if at:
                    f.attrs.update(at)
    return mesh


def random_flags(g: Gen, mesh):
    for e in mesh.edges:
        r = g.r.random()
        if r < 0.15:
            e.soft = True
        elif r < 0.25:
            e.hidden = True
        elif r < 0.35:
            e.curve = g.r.randrange(4)
        if g.r.random() < 0.1:
            e.layer = g.r.choice(["walls", "doors"])


def simplify_case(mesh, angle, principal=False):
    table = AttrTable()
    # the source mesh's vertices once; faces and edges index them
    svid = {id(v): k for k, v in enumerate(mesh.vertices)}
    sbits = {}
    for k, v in enumerate(mesh.vertices):
        sbits.setdefault(tuple(pb(v.position)), k)
    src_faces = []
    for f in mesh.faces:
        src_faces.append([[svid[id(v)] for v in f.loop], [[svid[id(v)] for v in h] for h in f.hole_loops],
                          f.interior, table.index(f.attrs)])
    infos = [table.info(i) for i in range(len(table.attrs))]
    layers = {n: i for i, n in enumerate(["walls", "doors"])}
    case = {"k": "simplify", "angle": h64(angle), "principal": principal,
            "pos": pbs([v.position for v in mesh.vertices]), "faces": src_faces, "attrs": infos,
            "edges": [[svid[id(e.v0)], svid[id(e.v1)],
                       [e.soft, e.curve, None if e.layer is None else layers[e.layer], e.hidden]]
                      for e in mesh.edges]}
    # the real thing
    try:
        with HoleWatch() as watch:
            removed = fuse.simplify_mesh(mesh, angle, principal)
    except ValueError:
        case["res"] = "error"
        return case
    if watch.dropped:
        return None
    case["removed"] = removed
    res = dump_mesh(mesh, table)
    # every result vertex sits at a source position: store its index there
    res["v"] = [sbits[tuple(b)] for b in res["v"]]
    case["res"] = res
    # attrs seen only after simplification must already be in the table
    assert len(table.attrs) == len(infos), "simplify invented attrs"
    return case


def simplify_cases(g: Gen, n: int):
    out = []
    planes = [([1, 0, 0], [0, 1, 0]), ([1, 0, 0], [0, 0, 1]), ([0, 1, 0], [0, 0, 1])]
    angles = [0.0, 0.0, 0.0, 0.02, 0.5, 1.0, 1.0, 2.5, 5.0]
    # planar and gently curved grids
    for i in range(n):
        plane = g.r.choice(planes)
        if g.r.random() < 0.3:
            plane = (g.unit(), g.unit())
        tilt = g.r.choice([0.0, 0.0, 0.01, 0.3, 0.8, 1.5, 3.0, 8.0, 25.0, 30.0, 31.0, 31.8, 32.0, 33.0, 35.0, 45.0])
        noise = g.r.choice([0.0, 0.0, 0.0, 1e-5, 3e-5, 2e-4])
        colors = None
        if g.r.random() < 0.5:
            mode = g.r.choice(["half", "stripe", "rand"])
            if mode == "half":
                colors = lambda x, y: 1 if x < 2 else 3
            elif mode == "stripe":
                colors = lambda x, y: 1 + (y % 3)
            else:
                rr = random.Random(g.r.random())
                colors = lambda x, y, rr=rr: rr.choice([0, 1, 1, 2, 3, 5, 6, 7, 8, 9])
        mesh = build_grid(g, plane, g.r.randint(1, 6), g.r.randint(1, 5), tilt, noise,
                          g.r.choice(["ab", "ad", "mix"]), colors)
        if g.r.random() < 0.4:
            random_flags(g, mesh)
        c = simplify_case(mesh, g.r.choice(angles), g.r.random() < 0.15)
        if c is not None:
            out.append(c)
    # cylinder strips (the test's), with extra collinear vertices on the seams
    for i in range(n // 3):
        mesh = Mesh()
        segs = g.r.choice([12, 30, 60, 90])
        radius = g.r.choice([1.0, 10.0])
        span = g.r.choice([math.pi / 2, math.pi, 2 * math.pi])
        hgt = g.r.choice([1.0, 2.0])
        sub = g.r.random() < 0.4
        for k in range(segs):
            a0, a1 = span * k / segs, span * (k + 1) / segs
            a = QV(radius * math.cos(a0), radius * math.sin(a0), 0)
            b = QV(radius * math.cos(a1), radius * math.sin(a1), 0)
            cpt = QV(radius * math.cos(a1), radius * math.sin(a1), hgt)
            d = QV(radius * math.cos(a0), radius * math.sin(a0), hgt)
            if sub:
                m1 = QV(a.x(), a.y(), hgt / 2)
                m2 = QV(b.x(), b.y(), hgt / 2)
                mesh.add_face((a, b, m2, m1))
                mesh.add_face((m1, m2, cpt, d))
            else:
                mesh.add_face((a, b, cpt))
                mesh.add_face((a, cpt, d))
        if g.r.random() < 0.4:
            random_flags(g, mesh)
        c = simplify_case(mesh, g.r.choice([0.0, 1.0, 2.0, 5.0]))
        if c is not None:
            out.append(c)
    # solids: cubes whose faces are subdivided, with colours, holes, interior
    for i in range(n // 3):
        mesh = Mesh()
        s = g.r.choice([1.0, 2.0])
        corners = {(x, y, z): QV(x * s, y * s, z * s) for x in (0, 1) for y in (0, 1) for z in (0, 1)}
        quads = [((0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0, 0)), ((0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)),
                 ((0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1)), ((1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1)),
                 ((1, 1, 0), (0, 1, 0), (0, 1, 1), (1, 1, 1)), ((0, 1, 0), (0, 0, 0), (0, 0, 1), (0, 1, 1))]
        for qi, q in enumerate(quads):
            p = [corners[c] for c in q]
            at = ATTR_POOL[g.r.choice([0, 1, 3, 4])]
            mode = g.r.choice(["quad", "tris", "fan", "holed", "interior"])
            if mode == "quad":
                f = mesh.add_face(p)
                if at:
                    f.attrs.update(at)
            elif mode == "tris":
                for t in ((p[0], p[1], p[2]), (p[0], p[2], p[3])):
                    f = mesh.add_face(t)
                    if at:
                        f.attrs.update(at)
            elif mode == "fan":
                cen = QV(sum(x.x() for x in p) / 4, sum(x.y() for x in p) / 4, sum(x.z() for x in p) / 4)
                for k in range(4):
                    f = mesh.add_face((p[k], p[(k + 1) % 4], cen))
                    if at:
                        f.attrs.update(at)
            elif mode == "holed":
                cen = [sum(x.x() for x in p) / 4, sum(x.y() for x in p) / 4, sum(x.z() for x in p) / 4]
                inner = [QV(*(cen[j] + (p[k] - QV(*cen)).x() * 0.3 * (j == 0) + (p[k] - QV(*cen)).y() * 0.3 * (j == 1)
                              + (p[k] - QV(*cen)).z() * 0.3 * (j == 2) for j in range(3))) for k in range(4)]
                f = mesh.add_face(p, [inner])
                if at:
                    f.attrs.update(at)
            else:
                f = mesh.add_face(p)
                f.interior = True
        if g.r.random() < 0.5:
            random_flags(g, mesh)
        c = simplify_case(mesh, g.r.choice([0.0, 1.0]))
        if c is not None:
            out.append(c)
    # soups: duplicates (two-sided), slivers, coincident triangles, loose edges
    for i in range(n // 3):
        mesh = Mesh()
        grid = build_grid(g, ([1, 0, 0], [0, 1, 0]), g.r.randint(1, 3), g.r.randint(1, 3), 0.0, 0.0, "mix", None)
        for f in grid.faces:
            pts = f.vertices
            mesh.add_face(pts)
            if g.r.random() < 0.3:
                mesh.add_face(list(reversed(pts)))      # the back copy
            if g.r.random() < 0.15:
                mesh.add_face((pts[0], pts[1], QV(pts[0].x() + 5e-5, pts[0].y(), pts[0].z())))   # a sliver
        for _ in range(g.r.randint(0, 3)):
            try:
                mesh.add_edge(QV(*(g.num(-2, 2) for _ in range(3))), QV(*(g.num(-2, 2) for _ in range(3))))
            except ValueError:
                pass
        random_flags(g, mesh)
        c = simplify_case(mesh, g.r.choice([0.0, 1.0, 5.0]), g.r.random() < 0.2)
        if c is not None:
            out.append(c)
    # hinges: panels folded by an angle around the soften threshold (cos 0.85 = 31.788 deg)
    for i in range(n // 3):
        mesh = Mesh()
        th = math.degrees(math.acos(0.85)) + g.r.choice([-3, -1, -0.3, -0.05, -0.01, 0.0, 0.01, 0.05, 0.3, 1, 3])
        a = math.radians(th)
        x = y = 0.0
        d = [1.0, 0.0, 0.0]
        pts = [QV(0, 0, 0)]
        ang = 0.0
        for k in range(g.r.randint(2, 4)):
            ang += a * g.r.choice([1, 1, -1])
            pts.append(QV(pts[-1].x() + math.cos(ang), pts[-1].y(), pts[-1].z() + math.sin(ang)))
        for k in range(len(pts) - 1):
            p0, p1 = pts[k], pts[k + 1]
            f = mesh.add_face((p0, p1, QV(p1.x(), 1.0, p1.z()), QV(p0.x(), 1.0, p0.z())))
            at = ATTR_POOL[g.r.choice([0, 1, 1, 3])]
            if at:
                f.attrs.update(at)
        c = simplify_case(mesh, g.r.choice([0.0, 1.0]))
        if c is not None:
            out.append(c)
    # empty mesh and bad angles
    for ang in (float("nan"), float("inf"), -0.5, 5.0001, 7.0):
        c = simplify_case(build_grid(g, ([1, 0, 0], [0, 1, 0]), 2, 2, 0.0, 0.0, "ab", None), ang)
        out.append(c)
    out.append(simplify_case(Mesh(), 1.0))
    return out


def cos_cases():
    out = []
    angles = [0.0, 1e-9, 0.01, 0.02, 0.025783, 0.0258, 0.03, 0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 4.9999, 5.0,
              5.0001, -0.0, -1.0, float("nan"), float("inf")]
    angles += [i * 0.05 for i in range(0, 101)]
    for a in angles:
        if not math.isfinite(a) or not 0.0 <= a <= 5.0:
            out.append({"k": "cos", "a": h64(a), "r": None})
        else:
            out.append({"k": "cos", "a": h64(a), "r": h64(math.cos(max(math.radians(a), 0.00045)))})
    return out


def write(name: str, meta: dict, rows) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / name, "w", encoding="ascii", newline="\n") as fh:
        fh.write(json.dumps({"meta": meta}) + "\n")
        for r in rows:
            fh.write(json.dumps(r, separators=(",", ":")) + "\n")
    print(name, len(rows), "rows", (OUT / name).stat().st_size, "bytes")


def main() -> None:
    meta = {"qt": qVersion(), "pyside": pyside_version, "python": sys.version.split()[0], "platform": sys.platform}
    g = Gen(20261019)
    write("sweep_rings.jsonl", meta, sweep_cases(g, int(sys.argv[1]) if len(sys.argv) > 1 else 300))
    rows = []
    rows += hypot_cases(g, 400)
    rows += order_cases(g, 300)
    rows += orient_cases(g, 200)
    rows += manual_cases(g, 300)
    rows += facet_cases(g, 300)
    write("sweep_misc.jsonl", meta, rows)
    rows = cos_cases() + simplify_cases(g, int(sys.argv[2]) if len(sys.argv) > 2 else 80)
    write("simplify.jsonl", meta, rows)


if __name__ == "__main__":
    main()
