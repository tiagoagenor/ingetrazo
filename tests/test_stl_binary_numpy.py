# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
"""A binary STL's facets read in NumPy (formats.stl._binary_triangles)
build the mesh the record-by-record loop built: random and lattice
triangles, degenerate ones and ones at the 1e-12 area threshold, more
than one 4 096-triangle batch, scales, and the same error on a coordinate
out of range."""
from __future__ import annotations

import math
import random
import struct

import numpy as np
import pytest

from core.mesh import Mesh
from formats import stl


def _oracle(raw, count, scale, target):
    batch = []

    def flush():
        if batch:
            n = len(batch)
            target.add_faces_bulk(
                np.asarray(batch, dtype=np.float64).reshape(-1, 3),
                np.full(n, 3, dtype=np.int64), np.ones(n, dtype=np.int64))
            batch.clear()
    for index in range(count):
        record = raw[index * 50:(index + 1) * 50]
        if len(record) != 50:
            raise ValueError("STL binary facet data is truncated")
        data = struct.unpack("<12fH", record)
        tri = tuple(stl._scaled_point(data[i:i + 3], scale) for i in (3, 6, 9))
        if stl._non_degenerate(tri):
            batch.append(tri)
            if len(batch) >= stl._BATCH_SIZE:
                flush()
    flush()


def _raw(tris):
    out = bytearray()
    for t in tris:
        out += struct.pack("<12fH", 0.0, 0.0, 0.0, *[c for p in t for c in p], 0)
    return bytes(out)


def _signature(m):
    vid = {id(v): i for i, v in enumerate(m.vertices)}
    return ([v.position.toTuple() for v in m.vertices],
            [[vid[id(v)] for v in f.loop] for f in m.faces])


def _tris(rng, n):
    out = []
    for _ in range(n):
        kind = rng.random()
        if kind < 0.5:
            out.append([[rng.randrange(-6, 6) * 0.5 for _ in range(3)]
                        for _ in range(3)])
        elif kind < 0.8:
            out.append([[rng.uniform(-50, 50) for _ in range(3)]
                        for _ in range(3)])
        elif kind < 0.9:                              # degenerate
            p = [rng.uniform(-5, 5) for _ in range(3)]
            out.append([p, p, [p[0] + 1, p[1], p[2]]])
        else:                                         # at the area threshold
            s = math.sqrt(1e-12) * rng.choice([0.999, 1.0, 1.001, 1.0005])
            out.append([[0, 0, 0], [s, 0, 0], [0, s, 0]])
    return out


@pytest.mark.parametrize("seed,n", [(0, 50), (1, 900), (2, 5000), (3, 9000)])
@pytest.mark.parametrize("scale", [1.0, 0.001, 25.4])
def test_same_mesh_as_the_record_loop(seed, n, scale):
    raw = _raw(_tris(random.Random(seed), n))
    a, b = Mesh(), Mesh()
    _oracle(raw, n, scale, a)
    stl._binary_triangles(raw, n, scale, b, None)
    assert _signature(b) == _signature(a)
    assert len(b.faces) > 0


def test_out_of_range_raises_the_same():
    tris = _tris(random.Random(5), 20)
    tris[7][1][2] = 3.0e38
    raw = _raw(tris)
    with pytest.raises(ValueError) as want:
        _oracle(raw, 20, 2.0, Mesh())
    with pytest.raises(ValueError) as got:
        stl._binary_triangles(raw, 20, 2.0, Mesh(), None)
    assert str(got.value) == str(want.value)


def test_threshold_decided_by_hypot():
    """Lengths within the recheck band go through math.hypot, as before."""
    s = math.sqrt(2e-12)
    raw = _raw([[[0, 0, 0], [s, 0, 0], [0, s, 0]]])
    a, b = Mesh(), Mesh()
    _oracle(raw, 1, 1.0, a)
    stl._binary_triangles(raw, 1, 1.0, b, None)
    assert _signature(b) == _signature(a)


def test_near_threshold_asks_math_hypot(monkeypatch):
    """Inside the recheck band the decision is math.hypot's: np.sqrt of the
    summed squares and math.hypot can part by an ulp there (no natural
    float32 input was found that shows it, so a stand-in hypot proves the
    band is read)."""
    s = math.sqrt(1e-12) * 1.0002                    # length 1.0004e-12
    raw = _raw([[[0, 0, 0], [s, 0, 0], [0, s, 0]]])
    calls = []

    def fake(*xs):
        calls.append(xs)
        return 0.0                                   # "degenerate"
    monkeypatch.setattr(stl.math, "hypot", fake)
    m = Mesh()
    stl._binary_triangles(raw, 1, 1.0, m, None)
    assert calls and not m.faces
