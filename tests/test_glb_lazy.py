# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
"""A big glTF builds as a lazy mesh with the engine on: its plan is the
mesh building it makes, and that mesh is the engine-off build — every
vertex, edge (soft and hidden flags from the smoothing and foliage passes
worked out on arrays), face and face attrs — on faceted curves, colour
boundaries, a cutout (alpha) texture and an opaque one."""
from __future__ import annotations

import math
import random

import numpy as np
import pytest

import core._native as _native
from core import mesh as M
from formats import glb
from tests.test_lazy_mesh_plan import _compare

nat = _native.available()
pytestmark = pytest.mark.skipif(
    nat is None or _native.qt_hypot() is None,
    reason="native module or hypot unknown")


def _signature(m):
    vid = {id(v): i for i, v in enumerate(m.vertices)}
    fid = {id(f): i for i, f in enumerate(m.faces)}
    return ([v.position.toTuple() for v in m.vertices],
            [(vid[id(e.v0)], vid[id(e.v1)], bool(e.soft),
              bool(getattr(e, "hidden", False)), [fid[id(f)] for f in e.faces])
             for e in m.edges],
            [([vid[id(v)] for v in f.loop], f.attrs) for f in m.faces])


def _cylinder(sides, r, h, x0):
    tris = []
    for k in range(sides):
        t0, t1 = 2 * math.pi * k / sides, 2 * math.pi * (k + 1) / sides
        a = (x0 + r * math.cos(t0), r * math.sin(t0))
        b = (x0 + r * math.cos(t1), r * math.sin(t1))
        tris += [[(*a, 0), (*b, 0), (*b, h)], [(*a, 0), (*b, h), (*a, h)]]
    return np.asarray(tris, dtype=np.float32)


def _batches(tmp_path, seed):
    from PySide6.QtGui import QImage
    rng = random.Random(seed)
    leaf = tmp_path / "leaf.png"
    img = QImage(4, 4, QImage.Format_ARGB32)
    img.fill(0x80336633)
    img.save(str(leaf))
    wall = tmp_path / "wall.png"
    img = QImage(4, 4, QImage.Format_RGB32)
    img.fill(0x8B5A2B)
    img.save(str(wall))
    out = []
    for k in range(rng.randrange(2, 5)):
        tris = _cylinder(rng.choice([12, 48, 200]), rng.uniform(0.5, 3),
                         rng.uniform(1, 4), k * 8.0)
        rgba = rng.choice([None, (1.0, 1.0, 1.0, 1.0), (0.8, 0.2, 0.1, 1.0)])
        out.append((tris, None, (None, rgba, rng.choice([None, 0.5]))))
    # Textured: a cutout card soup and an opaque plane.
    for path in (leaf, wall):
        n = rng.randrange(300, 900)
        tris = []
        for i in range(n):
            x, y = (i % 30) * 0.1, (i // 30) * 0.1
            tris.append([(x, y, 5), (x + 0.1, y, 5), (x, y + 0.1, 5.01)])
        tris = np.asarray(tris, dtype=np.float32)
        uvs = np.asarray([[(0, 0), (1, 0), (0, 1)]] * n, dtype=np.float32)
        out.append((tris, uvs, (str(path), None, None)))
    return out


@pytest.mark.parametrize("seed", range(5))
def test_lazy_glb_is_the_built_one(monkeypatch, tmp_path, seed):
    monkeypatch.setattr(glb, "_LAZY_MIN_TRIS", 1)
    monkeypatch.setattr(_native, "native", nat)
    batches = _batches(tmp_path, seed)
    monkeypatch.setattr(_native, "wanted", lambda: False)
    eager = glb.build_mesh(batches)
    monkeypatch.setattr(_native, "wanted", lambda: True)
    lazy = glb.build_mesh(batches)
    assert type(lazy) is M.DeferredMesh
    from core import lazy_mesh
    _compare(lazy_mesh.plan_of(lazy), lazy)            # builds it
    a, b = _signature(eager), _signature(lazy)
    assert b[0] == a[0]
    assert b[1] == a[1]
    assert b[2] == a[2]
    assert any(e[2] for e in a[1]) and any(e[3] for e in a[1]), \
        "soft and hidden edges both exercised"


def test_small_glb_stays_eager(monkeypatch, tmp_path):
    monkeypatch.setattr(_native, "wanted", lambda: True)
    tris = _cylinder(8, 1, 1, 0)
    assert type(glb.build_mesh([(tris, None, (None, None, None))])) is M.Mesh
