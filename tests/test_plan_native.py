# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
"""The Rust plan topology (core.lazy_mesh.plan after the weld) is the NumPy
replay, field for field: edges in order and orientation, soft/hidden,
curve and layer (last wins), ring corners and each edge's faces — on raw
batches with duplicate, reversed and welded-shut edges, holes, and faces
sharing edges with the loose ones. And the planned mesh is still the
built one with the engine on (test_lazy_mesh_plan's comparison)."""
from __future__ import annotations

import random

import numpy as np
import pytest

import core._native as _native
from core import lazy_mesh
from formats import igz
from core.scene import Scene
from tests.test_deferred_mesh import _meshes, _random_scene
from tests.test_lazy_mesh_plan import _compare

nat = _native.available()
pytestmark = pytest.mark.skipif(
    nat is None or not hasattr(nat, "plan_topology"),
    reason="native module not built")


def _batch(rng):
    """A format-3 block: points on a coarse lattice nudged around the weld
    cell, edges and faces over them (rings of 1 outer + 0–2 holes)."""
    npts = rng.randrange(3, 120)
    pos = np.array([[rng.randrange(-3, 3) * 0.5 + rng.choice(
        [0, 0, 0, 4.9e-5, 5.1e-5, 1e-6]) for _ in range(3)]
        for _ in range(npts)])
    ne = rng.randrange(0, 80)
    edges = np.array([[rng.randrange(npts), rng.randrange(npts)]
                      for _ in range(ne)], dtype=np.int64).reshape(-1, 2)
    if ne and rng.random() < 0.5:                 # duplicates, reversed
        dup = edges[rng.sample(range(ne), min(ne, 5))][:, ::-1]
        edges = np.concatenate([edges, dup])
    ne = len(edges)
    flags = curves = layers = None
    if rng.random() < 0.8:
        flags = np.array([rng.randrange(4) for _ in range(ne)], np.int64)
        if rng.random() < 0.7:
            curves = np.array([rng.choice([-1, -1, 0, 3, 7])
                               for _ in range(ne)], np.int64)
        if rng.random() < 0.7:
            layers = np.array([rng.choice([-1, 0, 1])
                               for _ in range(ne)], np.int64)
    rings, sizes, counts = [], [], []
    for _ in range(rng.randrange(0, 25)):
        nr = 1 + (rng.random() < 0.3) * rng.randrange(1, 3)
        counts.append(nr)
        for _ in range(nr):
            k = rng.randrange(3, 7)
            sizes.append(k)
            rings += [rng.randrange(npts) for _ in range(k)]
    return (pos, edges, flags, curves, layers, ["Guias", "Muros"],
            np.array(rings, np.int64), np.array(sizes, np.int64),
            np.array(counts, np.int64))


def _plans(monkeypatch, args, sequential=False):
    out = []
    for mod in (None, nat):
        monkeypatch.setattr(_native, "native", mod)
        out.append(lazy_mesh.plan(*args, sequential=sequential))
    return out


def _same(a, b):
    assert a.pos.tolist() == b.pos.tolist()
    assert a.edges.tolist() == b.edges.tolist()
    assert a.edges.shape == b.edges.shape
    assert a.soft.tolist() == b.soft.tolist()
    assert a.hidden.tolist() == b.hidden.tolist()
    assert a.curve == b.curve
    assert a.layer == b.layer
    assert a.corners.tolist() == b.corners.tolist()
    assert a.ring_sizes.tolist() == b.ring_sizes.tolist()
    assert a.ring_counts.tolist() == b.ring_counts.tolist()
    assert a.ef_start.tolist() == b.ef_start.tolist()
    assert a.ef_face.tolist() == b.ef_face.tolist()


@pytest.mark.parametrize("seed", range(40))
def test_rust_plan_is_numpy_plan(monkeypatch, seed):
    rng = random.Random(seed)
    for _ in range(10):
        args = _batch(rng)
        py, rs = _plans(monkeypatch, args, sequential=rng.random() < 0.3)
        _same(py, rs)


def test_empty_block(monkeypatch):
    empty = np.empty(0, np.int64)
    args = (np.zeros((0, 3)), np.zeros((0, 2), np.int64), None, None, None,
            [], empty, empty, empty)
    _same(*_plans(monkeypatch, args))


def test_inconsistent_input_is_refused():
    with pytest.raises(ValueError):
        nat.plan_topology(np.arange(3), np.array([[0, 5]]), None, None, None,
                          np.empty(0, np.int64), np.empty(0, np.int64),
                          np.empty(0, np.int64))


@pytest.mark.parametrize("seed", range(6))
def test_planned_mesh_is_built_mesh_with_the_engine(monkeypatch, tmp_path,
                                                    seed):
    monkeypatch.setattr(igz, "_LARGE_DOCUMENT", 0)
    monkeypatch.setattr(igz, "_BULK_MIN", 1)
    monkeypatch.setattr(_native, "wanted", lambda: True)
    monkeypatch.setattr(_native, "native", nat)
    path = tmp_path / "doc.igz"
    igz.save_scene(_random_scene(random.Random(seed)), path)
    scene = Scene()
    igz.load_into(scene, path)
    for m in _meshes(scene):
        _compare(lazy_mesh.plan_of(m), m)
