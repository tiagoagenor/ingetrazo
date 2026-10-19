# SPDX-License-Identifier: GPL-3.0-or-later
"""Fixtures holding ``ingetrazo-geom::triangulate`` to
``core.triangulate.triangulate`` (M1.3): random loops — convex, concave
(stars, combs), with holes, collinear runs, duplicates, tilted planes,
too short — with and without a given normal, and the triangles Python
returns, as float32 bits. Committed; rerun only to widen:

    python scripts/golden/make_triangulate_fixtures.py

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

from core.triangulate import triangulate  # noqa: E402
from core.vecmath import QMatrix4x4, QVector3D as V  # noqa: E402

OUT = ROOT / "rust" / "crates" / "ingetrazo-geom" / "tests" / "data"


def h32(x: float) -> str:
    return struct.pack(">f", x).hex()


def pb(p) -> list:
    return [h32(p.x()), h32(p.y()), h32(p.z())]


def ring(r, n, rad, cx=0.0, cy=0.0, star=False, comb=False):
    pts = []
    for i in range(n):
        a = 2 * math.pi * i / n
        k = rad * (0.45 if star and i % 2 else 1.0)
        if comb:
            k = rad * (0.3 + 0.7 * r.random())
        pts.append((cx + k * math.cos(a), cy + k * math.sin(a)))
    return pts


def case(r) -> dict:
    kind = r.choice(["convex", "star", "comb", "rect_holes", "collinear", "dup", "short"])
    if kind == "convex":
        pts = ring(r, r.randint(3, 12), r.uniform(0.5, 20))
    elif kind == "star":
        pts = ring(r, 2 * r.randint(3, 8), r.uniform(0.5, 20), star=True)
    elif kind == "comb":
        pts = ring(r, r.randint(5, 20), r.uniform(0.5, 20), comb=True)
    elif kind == "collinear":
        pts = [(0, 0), (1, 0), (2, 0), (3, 0), (3, 2), (1.5, 2), (0, 2)]
    elif kind == "dup":
        pts = [(0, 0), (2, 0), (2, 0), (2, 2), (0, 2)]
    elif kind == "short":
        pts = [(0, 0), (1, 0)]
    else:
        w, h_ = r.uniform(4, 30), r.uniform(4, 30)
        pts = [(0, 0), (w, 0), (w, h_), (0, h_)]
    holes2 = []
    if kind == "rect_holes" or (kind in ("convex", "star") and r.random() < 0.3):
        span = max(abs(c) for p in pts for c in p) or 1.0
        for _ in range(r.randint(1, 3)):
            cx = r.uniform(0.2, 0.8) * (pts[2][0] if kind == "rect_holes" else 0.3 * span)
            cy = r.uniform(0.2, 0.8) * (pts[2][1] if kind == "rect_holes" else 0.3 * span)
            s = 0.08 * span
            hole = [(cx - s, cy - s), (cx - s, cy + s), (cx + s, cy + s), (cx + s, cy - s)]
            holes2.append(hole)
    m = QMatrix4x4()
    if r.random() < 0.6:
        m.rotate(r.uniform(-180, 180), r.uniform(-1, 1), r.uniform(-1, 1), r.uniform(-1, 1))
    m.translate(r.uniform(-50, 50), r.uniform(-50, 50), r.uniform(-5, 5))
    if r.random() < 0.3:
        pts = pts[::-1]
    outer = [m.map(V(x, y, 0)) for x, y in pts]
    holes = [[m.map(V(x, y, 0)) for x, y in hl] for hl in holes2]
    normal = None
    if r.random() < 0.4:
        normal = m.mapVector(V(0, 0, r.choice([1.0, -1.0, 2.5])))
    tris = triangulate(outer, holes, normal)
    return {"kind": kind, "outer": [pb(p) for p in outer], "holes": [[pb(p) for p in hl] for hl in holes],
            "normal": pb(normal) if normal is not None else None,
            "tris": [[pb(a), pb(b), pb(c)] for a, b, c in tris]}


def main() -> None:
    r = random.Random(1310)
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "triangulate.jsonl", "w", encoding="ascii", newline="\n") as fh:
        for _ in range(1500):
            fh.write(json.dumps(case(r), separators=(",", ":")) + "\n")
    print("written", (OUT / "triangulate.jsonl").stat().st_size)


if __name__ == "__main__":
    main()
