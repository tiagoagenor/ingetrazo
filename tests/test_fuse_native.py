# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
"""The Rust fuse_regions (formats.fuse.fuse_coplanar_loops) gives the
Python's polygons: the same outlines point for point, the same holes, the
same attrs objects and originals, in the same order — on triangle soups,
a 0.5 lattice, two-sided duplicates (textured / coloured / bare copies),
UV maps at the closeness tolerance, a faceted cylinder (drift guard),
non-manifold and touching contours, holes, n-gons and principal planes."""
from __future__ import annotations

import math
import random

import pytest
from PySide6.QtGui import QMatrix4x4, QVector3D as V

import core._native as _native
from formats import fuse

nat = _native.available()
pytestmark = pytest.mark.skipif(
    nat is None or not hasattr(nat, "fuse_regions")
    or _native.qt_hypot() is None, reason="native module or hypot unknown")


def _picture(out):
    return [([p.toTuple() for p in outer], [[p.toTuple() for p in h]
                                            for h in holes],
             id(attrs), [id(o) for o in originals])
            for outer, holes, attrs, originals in out]


def _both(monkeypatch, loops, **kw):
    res = []
    for mod in (None, nat):
        monkeypatch.setattr(_native, "native", mod)
        res.append(_picture(fuse.fuse_coplanar_loops(loops, **kw)))
    return res


RED = {"color": [0.8, 0.1, 0.1]}
BLUE = {"color": [0.1, 0.1, 0.8]}


def _tex(uvw, path="a.png"):
    return {"texture": {"path": path, "sw": 1.0, "sh": 1.0, "uvw": uvw}}


def _grid(rng, n, z=0.0, m=None):
    """An n x n grid of quads split into triangles, random paints."""
    paints = [None, RED, BLUE, _tex([1, 0, 0, 0, 0, 1, 0, 0])]
    loops = []
    for i in range(n):
        for j in range(n):
            a, b = V(i * .5, j * .5, z), V(i * .5 + .5, j * .5, z)
            c, d = V(i * .5 + .5, j * .5 + .5, z), V(i * .5, j * .5 + .5, z)
            if m is not None:
                a, b, c, d = (m.map(p) for p in (a, b, c, d))
            paint = rng.choice(paints) if rng.random() < 0.3 else paints[0]
            loops.append(([a, b, c], paint))
            loops.append(([a, c, d], paint))
    rng.shuffle(loops)
    return loops


def _cylinder(sides, z0=0.0, z1=2.0):
    loops = []
    for k in range(sides):
        t0, t1 = 2 * math.pi * k / sides, 2 * math.pi * (k + 1) / sides
        p0, p1 = (math.cos(t0), math.sin(t0)), (math.cos(t1), math.sin(t1))
        loops.append(([V(*p0, z0), V(*p1, z0), V(*p1, z1), V(*p0, z1)], None))
    return loops


@pytest.mark.parametrize("seed", range(12))
def test_grids_and_soups(monkeypatch, seed):
    rng = random.Random(seed)
    m = None
    if seed % 3:
        m = QMatrix4x4()
        m.rotate(rng.uniform(0, 90), rng.random(), rng.random(), 0.3)
        m.translate(rng.uniform(-50, 50), 0, rng.uniform(-5, 5))
    loops = _grid(rng, rng.randrange(1, 9), m=m)
    loops += [([V(*(rng.uniform(-3, 3) for _ in range(3))) for _ in range(3)],
               rng.choice([None, RED])) for _ in range(rng.randrange(0, 20))]
    py, rs = _both(monkeypatch, loops)
    assert rs == py


def test_two_sided_duplicates_keep_the_best_copy(monkeypatch):
    rng = random.Random(3)
    base = _grid(rng, 4)
    loops = []
    for pts, paint in base:
        loops.append((pts, None))
        loops.append((list(reversed(pts)), rng.choice([RED, _tex(
            [1, 0, 0, 0, 0, 1, 0, 0]), None])))
    py, rs = _both(monkeypatch, loops)
    assert rs == py


@pytest.mark.parametrize("delta", [0.0, 0.009, 0.0199, 0.0201, 0.03, 0.5])
def test_uv_maps_near_the_tolerance(monkeypatch, delta):
    loops = []
    for i in range(6):
        a, b, c, d = V(i, 0, 0), V(i + 1, 0, 0), V(i + 1, 1, 0), V(i, 1, 0)
        uvw = [1.0, 0.0, 0.0, delta * (i % 2), 0.0, 1.0, 0.0, 0.0]
        tex = _tex(uvw)
        loops += [([a, b, c], tex), ([a, c, d], tex)]
    loops.append(([V(0, 2, 0), V(1, 2, 0), V(1, 3, 0)], _tex([1] * 5)))
    py, rs = _both(monkeypatch, loops)
    assert rs == py


@pytest.mark.parametrize("sides", [6, 24, 96, 400])
@pytest.mark.parametrize("cos_tol", [0.99999, math.cos(math.radians(5))])
def test_faceted_cylinder_drift(monkeypatch, sides, cos_tol):
    py, rs = _both(monkeypatch, _cylinder(sides), cos_tol=cos_tol)
    assert rs == py


def test_holes_touching_and_non_manifold(monkeypatch):
    loops = []
    # A ring of 8 quads around a square hole.
    for i, j in ((0, 0), (1, 0), (2, 0), (0, 1), (2, 1), (0, 2), (1, 2),
                 (2, 2)):
        loops.append(([V(i, j, 0), V(i + 1, j, 0), V(i + 1, j + 1, 0),
                       V(i, j + 1, 0)], RED))
    # Two squares touching at a corner (degree-4 vertex: kept as is).
    loops += [([V(5, 5, 0), V(6, 5, 0), V(6, 6, 0), V(5, 6, 0)], BLUE),
              ([V(6, 6, 0), V(7, 6, 0), V(7, 7, 0), V(6, 7, 0)], BLUE),
              ([V(6, 5, 0), V(7, 5, 0), V(6, 6, 0)], BLUE)]
    # A fin: three faces on one edge.
    loops += [([V(10, 0, 0), V(11, 0, 0), V(11, 1, 0)], None),
              ([V(10, 0, 0), V(11, 1, 0), V(10, 1, 0)], None),
              ([V(10, 0, 0), V(11, 1, 0), V(10.5, 0.5, 1)], None)]
    # Degenerate and tiny.
    loops += [([V(0, 0, 9), V(1, 0, 9)], None),
              ([V(0, 0, 9), V(1, 0, 9), V(2, 0, 9)], None)]
    for principal in (False, True):
        py, rs = _both(monkeypatch, loops, principal_planes=principal)
        assert rs == py


@pytest.mark.parametrize("seed", range(4))
def test_principal_planes_on_tilted_grids(monkeypatch, seed):
    rng = random.Random(40 + seed)
    m = QMatrix4x4()
    m.rotate(rng.choice([0.0, 1e-5, 30.0]), 1.0, 0.0, 0.0)
    loops = _grid(rng, 5, m=m)
    py, rs = _both(monkeypatch, loops, principal_planes=True)
    assert rs == py


def test_key_rounding_tie(monkeypatch):
    """1/32 is a float32 and 1/32 × 1e4 = 312.5: round(x, 4) goes to the
    even 0.0312, so the vertex below does NOT weld with 0.03128 (0.0313)
    and the two triangles share no edge."""
    a = [V(1 / 32, 0, 0), V(0, 1, 0), V(-1, 0, 0)]
    b = [V(0.03128, 0, 0), V(1, 1, 0), V(0, 1, 0)]
    py, rs = _both(monkeypatch, [(a, None), (b, None),
                                 ([V(1 / 32, 0, 0), V(1, 0, 0), V(1, 1, 0)],
                                  None)])
    assert rs == py
