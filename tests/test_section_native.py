# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
"""The Rust section_loops (core.hlr) is the Python one: the same closed
rings, point for point and in order — on lattice boxes cut through their
corners (points on the merge cells' borders), nested rings, open chains
dropped, chords that touch the plane at a point, shuffled chords, jittered
endpoints within and beyond the tolerance, and 1e-3…1e4 scales."""
from __future__ import annotations

import random

import numpy as np
import pytest

import core._native as _native
from core import hlr

nat = _native.available()
pytestmark = pytest.mark.skipif(
    nat is None or not hasattr(nat, "section_loops"),
    reason="native module not built")


def _both(monkeypatch, chords, tol=None):
    out = []
    for mod in (None, nat):
        monkeypatch.setattr(_native, "native", mod)
        out.append([r.tolist() for r in hlr.section_loops(chords, tol)])
    return out


def _ring(pts):
    return [(pts[i], pts[(i + 1) % len(pts)]) for i in range(len(pts))]


def _scene(rng, scale):
    chords = []
    for _ in range(rng.randrange(1, 8)):
        x0, y0 = rng.randrange(-6, 6) * 0.5, rng.randrange(-6, 6) * 0.5
        w, h = rng.randrange(1, 5) * 0.5, rng.randrange(1, 5) * 0.5
        sq = [(x0, y0, 0.0), (x0 + w, y0, 0.0), (x0 + w, y0 + h, 0.0),
              (x0, y0 + h, 0.0)]
        if rng.random() < 0.5:              # split edges: more chords
            sq = [p for i, p in enumerate(sq) for p in
                  (p, tuple((p[k] + sq[(i + 1) % 4][k]) / 2 for k in range(3)))]
        chords += _ring(sq)
        if rng.random() < 0.3:              # a hole inside it
            chords += _ring([(x0 + w / 4, y0 + h / 4, 0.0),
                             (x0 + 3 * w / 4, y0 + h / 4, 0.0),
                             (x0 + w / 2, y0 + 3 * h / 4, 0.0)])
    if rng.random() < 0.5:                  # an open chain (dropped)
        chords += [((9.0, 9.0, 0.0), (10.0, 9.0, 0.0)),
                   ((10.0, 9.0, 0.0), (10.0, 10.0, 0.0))]
    if rng.random() < 0.5:                  # touching at a point
        chords.append(((3.25, 3.25, 0.0), (3.25, 3.25, 0.0)))
    rng.shuffle(chords)
    chords = [(b, a) if rng.random() < 0.5 else (a, b) for a, b in chords]
    arr = np.asarray(chords, dtype=np.float64) * scale
    if rng.random() < 0.5:                  # jitter near the tolerance
        arr += rng.choice([1e-12, 1e-9, 1e-7]) * scale * np.asarray(
            [[[rng.uniform(-1, 1) for _ in range(3)] for _ in range(2)]
             for _ in range(len(chords))])
    return arr


@pytest.mark.parametrize("seed", range(30))
def test_rings_are_pythons(monkeypatch, seed):
    rng = random.Random(seed)
    arr = _scene(rng, rng.choice([1e-3, 1.0, 37.5, 1e4]))
    py, rs = _both(monkeypatch, arr)
    assert rs == py


@pytest.mark.parametrize("tol", [1e-9, 1e-4, 0.26])
def test_explicit_tolerance(monkeypatch, tol):
    arr = _scene(random.Random(99), 1.0)
    py, rs = _both(monkeypatch, arr, tol)
    assert rs == py


def test_a_box_cut_on_cell_borders(monkeypatch):
    """Corners exactly on multiples of the merge cell."""
    tol = 0.5
    sq = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0)]
    arr = np.asarray(_ring(sq), dtype=np.float64)
    py, rs = _both(monkeypatch, arr, tol)
    assert rs == py and len(rs) == 1


def test_empty_and_open(monkeypatch):
    assert _both(monkeypatch, []) == [[], []]
    open_chain = np.asarray([((0, 0, 0), (1, 0, 0)), ((1, 0, 0), (2, 0, 0))],
                            dtype=np.float64)
    assert _both(monkeypatch, open_chain) == [[], []]


def test_ties_the_merge_and_walk_decide(monkeypatch):
    """Inputs where a detail decides: endpoints exactly ``tol`` apart (<=
    merges them), a point within ``tol`` of two nodes in different cells
    (the dx→dy→dz order picks one), and a chord touching the plane at a
    ring's corner (a == b chords are skipped)."""
    tol = 0.25
    sq = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0)]
    exact = [(a, b) for a, b in _ring(sq)]
    exact[1] = ((1.0 + tol, 0.0, 0.0), exact[1][1])        # tol apart
    py, rs = _both(monkeypatch, np.asarray(exact, dtype=np.float64), tol)
    assert rs == py and len(rs) == 1

    tol = 0.5
    two = [((0.0, 0.0, 0.0), (3.0, 0.0, 0.0)),            # node B at z 0
           ((0.0, 0.0, 0.8), (3.0, 0.0, 0.8)),            # node C at z 0.8
           ((3.0, 0.0, 0.0), (3.0, 3.0, 0.0)),
           ((3.0, 3.0, 0.0), (0.0, 0.0, 0.4)),            # P: within tol of both
           ((3.0, 0.0, 0.8), (3.0, 3.0, 0.8)),
           ((3.0, 3.0, 0.8), (0.0, 0.0, 0.8))]
    py, rs = _both(monkeypatch, np.asarray(two, dtype=np.float64), tol)
    assert rs == py and rs

    corner = _ring(sq) + [((1.0, 1.0, 0.0), (1.0, 1.0, 0.0))]
    corner = [corner[-1]] + corner[:-1]                   # the point chord first
    py, rs = _both(monkeypatch, np.asarray(corner, dtype=np.float64), 1e-7)
    assert rs == py and len(rs) == 1
