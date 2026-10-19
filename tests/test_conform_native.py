# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
"""The Rust T-junction conforming (core.solids._conform_t_junctions, the
#227 retry) gives the Python's vertices, triangles and face ids, exactly:
rows of doors cut into a wall (T-junctions along shared edges), random
lattice triangles with vertices on, near and just off their edges, ends
within the 1e-9 parameter guard, degenerate edges, and 1e-3…1e5 scales."""
from __future__ import annotations

import random

import numpy as np
import pytest

import core._native as _native
from core import solids

nat = _native.available()
pytestmark = pytest.mark.skipif(
    nat is None or not hasattr(nat, "conform_t_junctions"),
    reason="native module not built")


def _both(monkeypatch, v, t, f):
    out = []
    for mod in (None, nat):
        monkeypatch.setattr(_native, "native", mod)
        vv, tt, ff = solids._conform_t_junctions(v, t, f)
        out.append((vv.tolist(), tt.tolist(), ff.tolist(),
                    vv.dtype.str, tt.dtype.str, ff.dtype.str))
    return out


def _doors(n):
    """A wall strip split by n doors: the wall's long bottom edge carries
    every door's corners as T-junctions."""
    v = [(0.0, 0.0, 0.0), (n * 2.0, 0.0, 0.0), (n * 2.0, 0.0, 3.0),
         (0.0, 0.0, 3.0)]
    t = [(0, 1, 2), (0, 2, 3)]
    for k in range(n):
        x0, x1 = k * 2.0 + 0.5, k * 2.0 + 1.5
        base = len(v)
        v += [(x0, 0.0, 0.0), (x1, 0.0, 0.0), (x1, 0.0, 2.0), (x0, 0.0, 2.0)]
        t += [(base, base + 1, base + 2), (base, base + 2, base + 3)]
    return np.array(v), np.array(t), np.arange(len(t)) % 3


def _random(rng, scale):
    grid = [(rng.randrange(0, 5) * 0.5, rng.randrange(0, 5) * 0.5,
             rng.randrange(0, 2) * 0.5) for _ in range(40)]
    v = list(grid)
    t = []
    for _ in range(30):
        a, b, c = rng.sample(range(len(grid)), 3)
        t.append((a, b, c))
        if rng.random() < 0.5:                   # a vertex on/near edge a-b
            s = rng.choice([0.5, 0.25, 1e-10, 1 - 1e-10, 0.75])
            off = rng.choice([0.0, 0.0, 5e-10, 2e-9, 1e-6])
            p, q = np.array(grid[a]), np.array(grid[b])
            v.append(tuple(p + s * (q - p) + off))
    v.append(v[0])                               # a duplicate vertex
    t.append((0, 0, 1))                          # a degenerate edge
    return np.array(v) * scale, np.array(t), \
        np.array([rng.randrange(5) for _ in t])


@pytest.mark.parametrize("n", [1, 4, 25])
def test_rows_of_doors(monkeypatch, n):
    py, rs = _both(monkeypatch, *_doors(n))
    assert rs == py
    assert len(py[1]) > 2 + 2 * n, "the wall was split"


@pytest.mark.parametrize("seed", range(20))
def test_random_lattice(monkeypatch, seed):
    rng = random.Random(seed)
    v, t, f = _random(rng, rng.choice([1e-3, 1.0, 1.0, 250.0, 1e5]))
    py, rs = _both(monkeypatch, v, t, f)
    assert rs == py


def test_refuses_bad_input():
    with pytest.raises(ValueError):
        nat.conform_t_junctions(np.zeros((2, 3)), np.array([[0, 1, 5]]),
                                np.array([0]), 1e-9)


def test_ties_that_decide(monkeypatch):
    """A vertex exactly ``tol`` off an edge (<= takes it), and a split
    triangle whose corners make the centroid's sum order show (0.1 + 0.2 +
    0.3 is not 0.3 + 0.2 + 0.1 in floats)."""
    tol = 1e-9 * 2.0                       # rel_tol × the largest |coord|
    v = np.array([(0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (0.0, 2.0, 0.0),
                  (1.0, tol, 0.0)])
    py, rs = _both(monkeypatch, v, np.array([(0, 1, 2)]), np.array([0]))
    assert rs == py and len(py[1]) == 4
    v = np.array([(0.1, 0.7, 0.3), (0.2, 0.1, 0.9), (0.3, 0.4, 0.1)])
    mid = (v[0] + v[1]) / 2
    v = np.vstack([v, mid])
    py, rs = _both(monkeypatch, v, np.array([(0, 1, 2)]), np.array([0]))
    assert rs == py and len(py[0]) == 5


@pytest.mark.parametrize("seed", range(10))
def test_outer_shell_roots_and_mask(monkeypatch, seed):
    """_outer_shell_mask's union-find in Rust: the same root per triangle
    and the same kept shells — boxes, voids (inverted boxes inside),
    vertex indices shuffled so the merges run in a scrambled order."""
    rng = random.Random(seed)
    quads = ((0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5),
             (2, 3, 7, 6), (3, 0, 4, 7))
    v, t = [], []
    for k in range(rng.randrange(1, 6)):
        o = [rng.uniform(-9, 9) for _ in range(3)]
        s = rng.uniform(0.5, 3)
        void = k and rng.random() < 0.4
        base = len(v)
        v += [(o[0] + x * s, o[1] + y * s, o[2] + z * s)
              for x, y, z in ((0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
                              (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1))]
        for q in quads:
            for tri in ((q[0], q[1], q[2]), (q[0], q[2], q[3])):
                tri = tuple(base + i for i in tri)
                t.append(tri[::-1] if void else tri)
    perm = list(range(len(v)))
    rng.shuffle(perm)
    v = np.array([v[perm.index(i)] for i in range(len(v))])
    t = np.array([[perm[i] for i in tri] for tri in t])
    rng.shuffle(t)
    out = []
    for mod in (None, nat):
        monkeypatch.setattr(_native, "native", mod)
        out.append(solids._outer_shell_mask(v, t).tolist())
    assert out[0] == out[1]
    assert nat.shell_roots(t.astype(np.int64), len(v)).tolist() == \
        _python_roots(v, t)


def _python_roots(v, t):
    parent = list(range(len(v)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for a, b, c in t.tolist():
        ra, rb, rc = find(a), find(b), find(c)
        parent[rb] = ra
        parent[find(rc)] = ra
    return [find(a) for a in t[:, 0].tolist()]
