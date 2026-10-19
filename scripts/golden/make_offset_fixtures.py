# SPDX-License-Identifier: GPL-3.0-or-later
"""Fixtures that hold ``ingetrazo-geom``'s ``offset`` and ``bevel_profiles``
to the Python ``core.offset`` / ``core.bevel_profiles``, bit for bit
(docs/rust-rewrite-plan.md, M1.4).

Random planar polygons (convex, star, concave, with collinear points, tiny
edges, duplicated points, either winding), open chains, positive, negative
and zero distances, and the real functions' answers as exact bits: float32
as ``>f`` hex, float64 as ``>d`` hex. ``offset_regions`` also records what
the arrangement and the triangulation returned inside it, so the Rust port
(which takes both as closures until M1.3 / the arrangement land) replays
them and must reproduce the same regions. Committed, so cargo test needs no
Python; rerun only to widen them:

    python scripts/golden/make_offset_fixtures.py

Deleted with the Python app at M10.
"""
from __future__ import annotations

import json
import math
import random
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from PySide6 import __version__ as pyside_version  # noqa: E402
from PySide6.QtCore import qVersion  # noqa: E402

import core.arrangement as arrangement  # noqa: E402
import core.triangulate as triangulate_mod  # noqa: E402
from core import bevel_profiles as bp  # noqa: E402
from core import offset as off  # noqa: E402
from core.topology import _offset_line_intersection  # noqa: E402
from core.vecmath import QVector3D  # noqa: E402

OUT = ROOT / "rust" / "crates" / "ingetrazo-geom" / "tests" / "data"


def f32(x: float) -> float:
    return struct.unpack("<f", struct.pack("<f", x))[0]


def h32(x: float) -> str:
    return struct.pack(">f", x).hex()


def h64(x: float) -> str:
    return struct.pack(">d", x).hex()


def vb(v) -> list:
    return [h32(v.x()), h32(v.y()), h32(v.z())]


def lb(loop) -> list:
    return [vb(p) for p in loop]


def mk(bits_or_xyz) -> QVector3D:
    return QVector3D(*bits_or_xyz)


# ---- recorders around the two collaborators -------------------------------

LOG: dict = {"arr": None, "tri": []}
_DEPTH = [0]
_real_arr = arrangement.planar_arrangement
_real_tri = triangulate_mod.triangulate


def _faces_bits(faces):
    return [[lb(outer), [lb(h) for h in holes]] for outer, holes in faces]


def _rec_arr(segs, origin, normal):
    rec = {"segs": [[vb(a), vb(b)] for a, b in segs], "origin": vb(origin), "normal": vb(normal)}
    _DEPTH[0] += 1
    try:
        faces = _real_arr(segs, origin, normal)
    except Exception:
        rec["faces"] = None
        LOG["arr"] = rec
        raise
    finally:
        _DEPTH[0] -= 1
    rec["faces"] = _faces_bits(faces)
    LOG["arr"] = rec
    return faces


def _rec_tri(loop, holes, normal):
    if _DEPTH[0]:
        return _real_tri(loop, holes, normal)
    rec = {"loop": lb(loop), "holes": len(holes), "normal": vb(normal)}
    try:
        tris = _real_tri(loop, holes, normal)
    except Exception:
        rec["tris"] = None
        LOG["tri"].append(rec)
        raise
    rec["tris"] = [[vb(a), vb(b), vb(c)] for a, b, c in tris]
    LOG["tri"].append(rec)
    return tris


arrangement.planar_arrangement = _rec_arr
triangulate_mod.triangulate = _rec_tri


# ---- generators -----------------------------------------------------------

class Gen:
    def __init__(self, seed: int) -> None:
        self.r = random.Random(seed)

    def plane(self):
        """A (possibly non-unit) normal and an origin."""
        r = self.r.random()
        if r < 0.35:
            n = self.r.choice([(0, 0, 1), (0, 0, -1), (0, 1, 0), (1, 0, 0), (0, -1, 0)])
            n = QVector3D(*n)
        else:
            n = QVector3D(*[f32(self.r.uniform(-1, 1)) for _ in range(3)])
            if n.lengthSquared() < 1e-3:
                n = QVector3D(0, 0, 1)
        if self.r.random() < 0.3:
            n = n * f32(self.r.choice([0.5, 2.0, 3.0, 0.01, 7.5]))
        origin = QVector3D(*[f32(round(self.r.uniform(-20, 20), self.r.randint(0, 3)))
                             for _ in range(3)])
        return n, origin

    def poly2d(self):
        """(points, scale): a polygon in plane coordinates."""
        r = self.r
        scale = r.choice([0.05, 0.2, 1.0, 2.5, 10.0, 40.0]) * r.uniform(0.5, 1.5)
        kind = r.choice(["convex", "star", "random", "rect", "L", "U", "saw", "dumbbell"])
        n = r.randint(3, 12)
        if kind == "convex":
            angs = sorted(r.uniform(0, 2 * math.pi) for _ in range(n))
            pts = [(scale * math.cos(a), scale * math.sin(a)) for a in angs]
        elif kind == "star":
            k = r.randint(3, 8)
            pts = []
            for i in range(2 * k):
                a = math.pi * i / k
                rad = scale * (1.0 if i % 2 == 0 else r.uniform(0.2, 0.7))
                pts.append((rad * math.cos(a), rad * math.sin(a)))
        elif kind == "random":
            angs = sorted(r.uniform(0, 2 * math.pi) for _ in range(n))
            pts = [(scale * r.uniform(0.3, 1.0) * math.cos(a),
                    scale * r.uniform(0.3, 1.0) * math.sin(a)) for a in angs]
        elif kind == "rect":
            w, h = scale * r.uniform(0.2, 2), scale * r.uniform(0.2, 2)
            pts = [(0, 0), (w, 0), (w, h), (0, h)]
        elif kind == "L":
            w, h = scale * r.uniform(1, 2), scale * r.uniform(1, 2)
            a, b = w * r.uniform(0.2, 0.6), h * r.uniform(0.2, 0.6)
            pts = [(0, 0), (w, 0), (w, b), (a, b), (a, h), (0, h)]
        elif kind == "U":
            w, h = scale * r.uniform(1, 2), scale * r.uniform(1, 2)
            a, b = w * r.uniform(0.25, 0.4), h * r.uniform(0.3, 0.7)
            pts = [(0, 0), (w, 0), (w, h), (w - a, h), (w - a, b), (a, b), (a, h), (0, h)]
        elif kind == "saw":
            k = r.randint(2, 5)
            w = scale * 2
            pts = [(0, 0), (w, 0)]
            for i in range(k):
                x = w - w * (i + 0.5) / k
                pts += [(x + w / (4 * k), scale), (x - w / (4 * k), scale * 0.3)]
            pts += [(0, scale)]
        else:  # dumbbell
            w = scale
            pts = [(0, 0), (w, 0), (w, w * 0.4), (w * 2, w * 0.4), (w * 2, 0), (w * 3, 0),
                   (w * 3, w), (w * 2, w), (w * 2, w * 0.6), (w, w * 0.6), (w, w), (0, w)]
        # Perturbations the app meets for real.
        if r.random() < 0.25:                      # collinear points
            out = []
            for i, p in enumerate(pts):
                q = pts[(i + 1) % len(pts)]
                out.append(p)
                if r.random() < 0.5:
                    t = r.uniform(0.2, 0.8)
                    out.append((p[0] + (q[0] - p[0]) * t, p[1] + (q[1] - p[1]) * t))
            pts = out
        if r.random() < 0.12:                      # tiny edge
            i = r.randrange(len(pts))
            p = pts[i]
            pts.insert(i + 1, (p[0] + r.choice([1e-12, 1e-8, 1e-5]), p[1]))
        if r.random() < 0.08:                      # duplicated point
            i = r.randrange(len(pts))
            pts.insert(i, pts[i])
        if r.random() < 0.3:
            pts = pts[::-1]
        if r.random() < 0.2:
            k = r.randrange(len(pts))
            pts = pts[k:] + pts[:k]
        return pts, scale

    def lift(self, pts2d, normal, origin):
        u, w = arrangement.plane_basis(normal)
        out = []
        for x, y in pts2d:
            out.append(origin + u * f32(x) + w * f32(y))
        return out

    def dist(self, scale):
        r = self.r.random()
        if r < 0.05:
            return 0.0
        if r < 0.08:
            return self.r.choice([1e-10, -1e-10, 1e-9, 5e-10])
        sign = self.r.choice([1.0, -1.0])
        return sign * scale * self.r.choice([0.01, 0.03, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.7, 1.5])

    def case(self):
        normal, origin = self.plane()
        pts2d, scale = self.poly2d()
        loop = self.lift(pts2d, normal, origin)
        return loop, normal, self.dist(scale), scale


# ---- fixture builders -----------------------------------------------------

def primitives(count: int, seed: int) -> list:
    g = Gen(seed)
    rows = []
    for _ in range(count):
        kind = g.r.choice(["line", "line", "basis", "pip", "dist", "inside", "interior"])
        if kind == "line":
            def vec(lo=-10.0, hi=10.0):
                return QVector3D(*[f32(g.r.uniform(lo, hi)) for _ in range(3)])
            n, _o = g.plane()
            p0, p1 = vec(), vec()
            u = vec(-1, 1).normalized()
            v = (u if g.r.random() < 0.1 else vec(-1, 1)).normalized()
            if g.r.random() < 0.1:
                v = (u * -1.0).normalized()
            res = _offset_line_intersection(p0, u, p1, v, n.normalized())
            rows.append({"k": "line", "p0": vb(p0), "u": vb(u), "p1": vb(p1), "v": vb(v),
                         "n": vb(n.normalized()), "out": None if res is None else vb(res)})
        elif kind == "basis":
            n, _o = g.plane()
            u, w = arrangement.plane_basis(n)
            rows.append({"k": "basis", "n": vb(n), "u": vb(u), "w": vb(w)})
        elif kind == "pip":
            m = g.r.randint(3, 9)
            poly = [(g.r.uniform(-5, 5), g.r.uniform(-5, 5)) for _ in range(m)]
            if g.r.random() < 0.3:     # horizontal edges, points on rows
                poly = [(round(x), round(y)) for x, y in poly]
            p = (g.r.uniform(-5, 5), g.r.uniform(-5, 5))
            if g.r.random() < 0.3:
                p = (float(round(p[0])), float(round(p[1])))
            rows.append({"k": "pip", "poly": [[h64(x), h64(y)] for x, y in poly],
                         "p": [h64(p[0]), h64(p[1])],
                         "out": arrangement._point_in_polygon(p, poly)})
        else:
            loop, normal, d, scale = g.case()
            n = normal.normalized()
            if kind == "dist":
                p = QVector3D(*[f32(g.r.uniform(-scale, scale)) for _ in range(3)]) + loop[0]
                rows.append({"k": "dist", "p": vb(p), "loop": lb(loop),
                             "out": h64(off._dist_to_boundary(p, loop))})
            elif kind == "inside":
                p = QVector3D(*[f32(g.r.uniform(-scale, scale)) for _ in range(3)]) + loop[0]
                rows.append({"k": "inside", "p": vb(p), "loop": lb(loop), "n": vb(n),
                             "out": bool(off._inside(p, loop, n))})
            else:
                LOG["tri"].clear()
                p = off.interior_point(loop, n)
                rows.append({"k": "interior", "loop": lb(loop), "n": vb(n),
                             "tri": list(LOG["tri"]),
                             "out": None if p is None else vb(p)})
    return rows


def segments_cases(count: int, seed: int) -> list:
    g = Gen(seed)
    rows = []
    for _ in range(count):
        loop, normal, d, _s = g.case()
        n = normal.normalized()
        if g.r.random() < 0.1:
            d = g.r.choice([2.0, -2.0]) * abs(d)
        segs = off._offset_segments(loop, n, d)
        rows.append({"loop": lb(loop), "n": vb(n), "d": h64(d),
                     "out": [[vb(a), vb(b)] for a, b in segs]})
    return rows


def chain_cases(count: int, seed: int) -> list:
    g = Gen(seed)
    rows = []
    for _ in range(count):
        normal, origin = g.plane()
        scale = g.r.choice([0.1, 1.0, 5.0])
        m = g.r.randint(1, 7)
        pts2d = [(g.r.uniform(-scale, scale), g.r.uniform(-scale, scale)) for _ in range(m)]
        if g.r.random() < 0.3 and m >= 2:                 # repeated / tiny
            i = g.r.randrange(m)
            pts2d.insert(i, pts2d[i])
        if g.r.random() < 0.2 and m >= 3:                 # collinear run
            a, b = pts2d[0], pts2d[1]
            pts2d.insert(1, ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2))
        if g.r.random() < 0.2:                            # sharp spike
            a = pts2d[0]
            pts2d += [(a[0] + scale, a[1]), (a[0] + scale * 0.1, a[1] + scale * 0.01)]
        pts = g.lift(pts2d, normal, origin)
        d = g.dist(scale)
        res = off.offset_chain(pts, normal, d)
        rows.append({"pts": lb(pts), "n": vb(normal), "d": h64(d),
                     "out": None if res is None else lb(res)})
    return rows


def regions_cases(count: int, seed: int) -> list:
    g = Gen(seed)
    rows = []
    for _ in range(count):
        loop, normal, d, _s = g.case()
        if g.r.random() < 0.04:
            loop = loop[: g.r.randint(0, 2)]
        if g.r.random() < 0.03:
            normal = QVector3D(0, 0, 0)
        LOG["arr"] = None
        LOG["tri"].clear()
        res = off.offset_regions(loop, normal, d)
        rows.append({"loop": lb(loop), "n": vb(normal), "d": h64(d),
                     "arr": LOG["arr"], "tri": list(LOG["tri"]),
                     "out": _faces_bits(res)})
    return rows


def steps_rows() -> list:
    return [{"k": "steps_count", "s": s, "out": bp.steps_count(s)}
            for s in list(range(-3, 41)) + [100, 101]]


def shape_json(shape):
    if isinstance(shape, str):
        return {"name": shape}
    return {"controls": [[h64(x), h64(y), bool(s)] for x, y, s in shape]}


def pts_bits(pts):
    return [[h64(x), h64(y)] for x, y in pts]


def random_controls(g: Gen, valid: bool) -> list:
    r = g.r
    m = r.randint(2, 12)
    xs = sorted((r.random() for _ in range(m - 2)), reverse=True)
    ys = sorted(r.random() for _ in range(m - 2))
    pts = [(1.0, 0.0, False)]
    for x, y in zip(xs, ys):
        if r.random() < 0.3:
            x = round(x, 2)
            y = round(y, 2)
        pts.append((x, y, r.random() < 0.55))
    pts.append((0.0, 1.0, False))
    if not valid:
        k = r.choice(["short", "start", "end", "range"])
        if k == "short":
            pts = pts[:1]
        elif k == "start":
            pts[0] = (0.9, 0.1, False)
        elif k == "end":
            pts[-1] = (0.1, 0.9, True)
        else:
            pts[len(pts) // 2] = (1.5, -0.3, True)
    elif r.random() < 0.2:                       # on the border within 1e-9
        pts[0] = (1.0 + 5e-10, -5e-10, False)
    return pts


def bevel_rows(seed: int) -> list:
    g = Gen(seed)
    rows = steps_rows()
    segs = list(range(-2, 25)) + [31, 40, 101]
    for name in list(bp.PRESETS) + ["nope", "", "superellipse"]:
        for s in segs:
            try:
                out = pts_bits(bp.sample(name, s))
            except ValueError:
                out = None
            rows.append({"k": "sample", "shape": shape_json(name), "s": s, "out": out})
            rows.append({"k": "controls", "shape": shape_json(name), "s": s,
                         "out": shape_json(bp.controls(name, s))["controls"]})
    for _ in range(120):
        ctl = random_controls(g, g.r.random() < 0.7)
        s = g.r.randint(-1, 40)
        rows.append({"k": "valid", "c": shape_json(ctl)["controls"], "out": bp.valid(ctl)})
        rows.append({"k": "controls", "shape": shape_json(ctl), "s": s,
                     "out": shape_json(bp.controls(ctl, s))["controls"]})
        rows.append({"k": "sample", "shape": shape_json(ctl), "s": s,
                     "out": pts_bits(bp.sample(ctl, s))})
    for name in bp.PRESETS:
        pts = bp.sample(name, 9)
        rows.append({"k": "mirrored", "pts": pts_bits(pts), "out": pts_bits(bp.mirrored(pts))})
    rows.append({"k": "mirrored", "pts": [], "out": []})
    # math.hypot itself: CPython's own algorithm, not the C library's.
    for _ in range(1500):
        e1, e2 = g.r.randint(-30, 30), g.r.randint(-30, 30)
        x = g.r.uniform(-1, 1) * 2.0 ** e1
        y = g.r.uniform(-1, 1) * 2.0 ** e2
        if g.r.random() < 0.05:
            x = 0.0
        rows.append({"k": "hypot", "x": h64(x), "y": h64(y), "out": h64(math.hypot(x, y))})
    for x, y in [(3.0, 4.0), (1e-310, 2e-310), (1e300, 1e300), (5e-324, 0.0), (0.0, 0.0),
                 (1.0, 1e-17), (-3.0, 4.0), (1e-200, 1e-200), (0.1, 0.2)]:
        rows.append({"k": "hypot", "x": h64(x), "y": h64(y), "out": h64(math.hypot(x, y))})
    return rows


def write(name: str, meta: dict, rows: list) -> None:
    path = OUT / name
    with open(path, "w", encoding="ascii", newline="\n") as fh:
        fh.write(json.dumps({"meta": meta}) + "\n")
        for r in rows:
            fh.write(json.dumps(r, separators=(",", ":")) + "\n")
    print(f"{name}: {len(rows)} rows, {path.stat().st_size / 1e6:.2f} MB")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / ".gitattributes").write_text("# Bits Python produced; compared exactly.\n* -text\n",
                                        encoding="ascii", newline="\n")
    meta = {"qt": qVersion(), "pyside": pyside_version, "platform": sys.platform,
            "python": sys.version.split()[0]}
    write("offset_primitives.jsonl", meta, primitives(700, 20261019))
    write("offset_segments.jsonl", meta, segments_cases(250, 7))
    write("offset_chain.jsonl", meta, chain_cases(300, 11))
    write("offset_regions.jsonl", meta, regions_cases(130, 23))
    write("bevel_profiles.jsonl", meta, bevel_rows(31))


if __name__ == "__main__":
    main()
