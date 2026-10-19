# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
"""The Rust arrangement split (core.arrangement._build_graph's all-pairs
pass) is the Python one, point for point and in order: on random
segments, a 0.5 lattice (collinear overlaps, T-junctions, shared ends),
1e-6…1e5 scales and degenerate ones (zero length, hits a hair outside the
ends, near-parallel) — and so are the graph and the traced faces."""
from __future__ import annotations

import random

import numpy as np
import pytest

import core._native as _native
from core import arrangement as A

nat = _native.available()
pytestmark = pytest.mark.skipif(
    nat is None or not hasattr(nat, "arrangement_splits"),
    reason="native module not built")


def _both(monkeypatch, fn, *args):
    out = []
    for mod in (None, nat):
        monkeypatch.setattr(_native, "native", mod)
        out.append(fn(*args))
    return out


def _random(rng, n, scale):
    return [((rng.uniform(-1, 1) * scale, rng.uniform(-1, 1) * scale),
             (rng.uniform(-1, 1) * scale, rng.uniform(-1, 1) * scale))
            for _ in range(n)]


def _lattice(rng, n):
    def p():
        return (rng.randrange(-4, 5) * 0.5, rng.randrange(-4, 5) * 0.5)
    out = []
    for _ in range(n):
        a = p()
        if rng.random() < 0.5:                      # axis-aligned: overlaps
            b = (a[0], a[1] + rng.randrange(-4, 5) * 0.5) if rng.random() \
                < 0.5 else (a[0] + rng.randrange(-4, 5) * 0.5, a[1])
        else:
            b = p()
        out.append((a, b))
    return out


def _degenerate(rng):
    e = A._EPS
    return [((0.0, 0.0), (0.0, 0.0)),              # zero length
            ((0.0, 0.0), (1.0, 0.0)),
            ((1.0 + e * 0.5, -1.0), (1.0 + e * 0.5, 1.0)),   # just past b
            ((-e * 2, -1.0), (-e * 2, 1.0)),       # just before a
            ((0.5, 0.0), (0.5, 1e-10)),            # near-degenerate T
            ((0.0, 1e-5), (1.0, 1e-5)),            # parallel within _ON_SEG
            ((0.25, 0.0), (0.75, 0.0)),            # collinear inside
            ((0.0, 0.0), (1.0, 1e-9)),             # rxs ~ 1e-9
            ((0.3, -0.5), (0.3, 0.5)), ((0.3, -0.5), (0.3, 0.5)),   # doubled
            # Tiny parallels: "collinear" (|qp x r| < 1e-9) yet 1.5e-4 and
            # 0.5e-4 apart — _ON_SEG decides which ends count.
            ((5.0, 5.0), (5.0 + 5e-6, 5.0)),
            ((5.0, 5.0 + 1.5e-4), (5.0 + 5e-6, 5.0 + 1.5e-4)),
            ((5.0, 5.0 + 0.5e-4), (5.0 + 5e-6, 5.0 + 0.5e-4))]


CASES = ([("random", s) for s in range(6)] + [("lattice", s) for s in range(6)]
         + [("scale", s) for s in range(6)] + [("degenerate", 0)])


def _segments(kind, seed):
    rng = random.Random(seed)
    if kind == "random":
        return _random(rng, rng.randrange(2, 60), 1.0)
    if kind == "lattice":
        return _lattice(rng, rng.randrange(2, 60))
    if kind == "scale":
        return _random(rng, 40, [1e-6, 1e-3, 1.0, 1e3, 1e5][seed % 5])
    return _degenerate(rng)


@pytest.mark.parametrize("kind,seed", CASES)
def test_split_points_are_pythons(monkeypatch, kind, seed):
    segs = _segments(kind, seed)
    py, rs = _both(monkeypatch, A._split_points, segs)
    assert rs == py


@pytest.mark.parametrize("kind,seed", CASES)
def test_graph_and_faces_are_pythons(monkeypatch, kind, seed):
    segs = _segments(kind, seed)
    (pc, pa), (rc, ra) = _both(monkeypatch, A._build_graph, segs)
    assert rc == pc and ra == pa


def test_empty_and_single(monkeypatch):
    for segs in ([], [((0.0, 0.0), (1.0, 0.0))]):
        py, rs = _both(monkeypatch, A._split_points, segs)
        assert rs == py


def test_rejects_a_bad_shape():
    with pytest.raises(ValueError):
        nat.arrangement_splits(np.zeros((2, 3, 2)))
