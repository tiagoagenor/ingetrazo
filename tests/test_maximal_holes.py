# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
"""core.topology._maximal_holes with its box prefilter keeps exactly the
holes the all-pairs version kept: nested, equal-area, side-by-side and
touching holes, concave ones, tilted planes, a hole lifted a float32 hair
off the plane, and a façade of many windows."""
from __future__ import annotations

import random

import pytest
from PySide6.QtGui import QMatrix4x4, QVector3D as V

from core import topology as T


def _oracle(holes):
    """The all-pairs version, as it was."""
    def area(loop):
        a = 0.0
        n = len(loop)
        for i in range(n):
            a += loop[i].x() * loop[(i + 1) % n].y()
            a -= loop[(i + 1) % n].x() * loop[i].y()
        return abs(a) / 2.0
    areas = [area(h) for h in holes]
    keep = []
    for i, h in enumerate(holes):
        nrm = V.crossProduct(h[1] - h[0], h[2] - h[0])
        if nrm.length() < 1e-9:
            keep.append(h)
            continue
        nrm = nrm.normalized()
        if not any(j != i and areas[j] > areas[i] + 1e-6
                   and T._loop_inside_loop(h, holes[j], nrm)
                   for j in range(len(holes))):
            keep.append(h)
    return keep


def _rect(x, y, w, h, z=0.0):
    return [V(x, y, z), V(x + w, y, z), V(x + w, y + h, z), V(x, y + h, z)]


def _scene(rng, tilt):
    holes = []
    for _ in range(rng.randrange(2, 40)):
        x, y = rng.randrange(0, 20) * 0.5, rng.randrange(0, 20) * 0.5
        w, h = rng.randrange(1, 8) * 0.5, rng.randrange(1, 8) * 0.5
        holes.append(_rect(x, y, w, h))
        if rng.random() < 0.3:                       # nested inside it
            holes.append(_rect(x + w * .25, y + h * .25, w * .5, h * .5))
        if rng.random() < 0.15:                      # an L (concave)
            holes.append([V(x, y, 0), V(x + w, y, 0), V(x + w, y + h * .3, 0),
                          V(x + w * .3, y + h * .3, 0), V(x + w * .3, y + h, 0),
                          V(x, y + h, 0)])
    if tilt:
        m = QMatrix4x4()
        m.rotate(rng.uniform(5, 80), rng.random(), rng.random(), 0.2)
        holes = [[m.map(p) for p in h] for h in holes]
    return holes


@pytest.mark.parametrize("seed", range(25))
@pytest.mark.parametrize("tilt", [False, True])
def test_same_holes_as_all_pairs(seed, tilt):
    holes = _scene(random.Random(seed), tilt)
    assert [id(h) for h in T._maximal_holes(holes)] == \
        [id(h) for h in _oracle(holes)]


def test_a_hole_a_hair_off_the_plane_is_still_nested():
    outer = _rect(0, 0, 4, 4)
    inner = _rect(1, 1, 1, 1, z=1e-7)                # float32 wobble
    holes = [outer, inner]
    assert T._maximal_holes(holes) == _oracle(holes) == [outer]


def test_many_windows():
    holes = [_rect(i * 1.0, j * 2.0, 0.6, 1.2) for i in range(40)
             for j in range(10)]
    holes += [_rect(i * 1.0 + 0.1, 0.2, 0.2, 0.2) for i in range(40)]
    assert [id(h) for h in T._maximal_holes(holes)] == \
        [id(h) for h in _oracle(holes)]
