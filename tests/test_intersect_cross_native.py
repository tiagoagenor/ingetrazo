# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
"""The Rust face crossing (core.intersect PolySet: world polygons, cross,
intervals, overlap, dedupe) gives Intersect Faces' very segments, bit for
bit and in order, in all three modes: tilted plates with holes, lattice
boxes whose corners sit on the cut lines, placed groups and components,
near-parallel faces, Newell lengths at Qt's fuzz of 0 and of 1, and
1e-3…1e4 scales. Compared in one process: the selection is a set."""
from __future__ import annotations

import random

import pytest
from PySide6.QtGui import QMatrix4x4, QVector3D as V

import core._native as _native
from core import intersect
from core.group import Group
from core.intersect import (WITH_CONTEXT, WITH_MODEL, WITH_SELECTION,
                            segments_for)
from core.scene import Scene

nat = _native.available()
pytestmark = pytest.mark.skipif(
    nat is None or not hasattr(nat, "PolySet") or _native.qt_hypot() is None,
    reason="native module or hypot unknown")


def _both(monkeypatch, scene, mode):
    out = []
    for mod in (None, nat):
        monkeypatch.setattr(_native, "native", mod)
        out.append([(p.toTuple(), q.toTuple())
                    for p, q in segments_for(scene, mode)])
    return out


def _add(mesh, pts, holes=None):
    try:
        mesh.add_face(pts, holes)
    except (ValueError, TypeError):
        pass


def _plate(rng, scale, lattice):
    m = QMatrix4x4()
    if lattice:
        m.translate(*(rng.randrange(-4, 4) * 0.5 * scale for _ in range(3)))
        axis = rng.choice([(1, 0, 0), (0, 1, 0), (0, 0, 1)])
        m.rotate(rng.choice([0, 90, 180, 270]), *axis)
    else:
        m.translate(*(rng.uniform(-3, 3) * scale for _ in range(3)))
        m.rotate(rng.uniform(0, 360), rng.random(), rng.random(),
                 rng.random() + 0.1)
    w = (rng.randrange(1, 6) * 0.5 if lattice else rng.uniform(0.5, 3)) * scale
    d = (rng.randrange(1, 6) * 0.5 if lattice else rng.uniform(0.5, 3)) * scale
    outer = [m.map(V(x, y, 0)) for x, y in ((0, 0), (w, 0), (w, d), (0, d))]
    hole = None
    if rng.random() < 0.3:
        hole = [[m.map(V(x, y, 0)) for x, y in
                 ((w * .25, d * .25), (w * .25, d * .75), (w * .75, d * .75),
                  (w * .75, d * .25))]]
    return outer, hole


def _scene(seed):
    rng = random.Random(seed)
    scale = rng.choice([1e-3, 1.0, 1.0, 37.5, 1e4])
    lattice = rng.random() < 0.5
    scene = Scene()
    for _ in range(rng.randrange(8, 30)):
        _add(scene.mesh, *_plate(rng, scale, lattice))
    for k in range(rng.randrange(1, 4)):
        g = Group(name=f"g{k}")
        for _ in range(rng.randrange(3, 10)):
            _add(g.mesh, *_plate(rng, scale, lattice))
        if rng.random() < 0.7:
            g.xform = QMatrix4x4()
            g.xform.translate(*(rng.uniform(-1, 1) * scale for _ in range(3)))
            g.xform.rotate(rng.uniform(0, 90), 0.0, 0.0, 1.0)
        scene.groups.append(g)
    return scene, rng


@pytest.mark.parametrize("seed", range(16))
@pytest.mark.parametrize("mode", [WITH_SELECTION, WITH_MODEL, WITH_CONTEXT])
def test_segments_are_pythons(monkeypatch, seed, mode):
    scene, rng = _scene(seed)
    things = list(scene.mesh.faces) + list(scene.groups)
    rng.shuffle(things)
    scene.selection.update(things[:max(1, len(things) // 2)]
                           if mode != WITH_SELECTION else things)
    py, rs = _both(monkeypatch, scene, mode)
    assert rs == py


def test_near_parallel_and_fuzz_lengths(monkeypatch):
    """Faces a hair off parallel (den < 1e-12 in float32), and faces whose
    Newell length sits within 1e-5 of 0 and of 1 (normalized() returns a
    null vector, or the vector as is)."""
    scene = Scene()
    m = scene.mesh
    _add(m, [V(0, 0, 0), V(4, 0, 0), V(4, 4, 0), V(0, 4, 0)])
    _add(m, [V(0, 0, 1e-7), V(4, 0, 2e-7), V(4, 4, 1e-7), V(0, 4, 0)])
    # Newell length 2·area: a 0.5 x 1 face is 1.0; nudged to 0.999998.
    _add(m, [V(1, -1, -1), V(1, -1 + 0.4999995, -1), V(1, -0.5000005, 0),
             V(1, -1, 0)])
    _add(m, [V(2, -1, -0.5), V(2.0005, -1, -0.5), V(2.0005, 1, 0.5)])
    _add(m, [V(-1, 2, -1), V(5, 2, -1), V(5, 2, 1), V(-1, 2, 1)])
    _add(m, [V(2, 0, -1), V(2, 4, -1), V(2, 4, 1), V(2, 0, 1)])   # on a corner line
    scene.selection.update(scene.mesh.faces)
    py, rs = _both(monkeypatch, scene, WITH_SELECTION)
    assert rs == py and rs


def test_empty_sides(monkeypatch):
    scene = Scene()
    _add(scene.mesh, [V(0, 0, 0), V(1, 0, 0), V(1, 1, 0)])
    scene.selection.update(scene.mesh.faces)
    for mode in (WITH_SELECTION, WITH_MODEL, WITH_CONTEXT):
        py, rs = _both(monkeypatch, scene, mode)
        assert rs == py == []


def test_bad_pairs_are_refused():
    import numpy as np
    s = nat.PolySet(np.zeros((3, 3), np.float32), np.array([0, 3]),
                    np.array([0, 1]), _native.qt_hypot())
    with pytest.raises(ValueError):
        s.cross_pairs(None, np.array([0]), np.array([5]))
    with pytest.raises(ValueError):
        nat.PolySet(np.zeros((3, 3), np.float32), np.array([0, 2]),
                    np.array([0, 1]), _native.qt_hypot())
    assert intersect._poly_set([]) is not None or nat is None
