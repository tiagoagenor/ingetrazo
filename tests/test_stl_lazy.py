# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
"""A big binary STL imports as a lazy mesh with the engine on: its plan is
the mesh building it makes, index for index; built, it is the mesh the
engine-off import makes; and putting it in the scene builds nothing."""
from __future__ import annotations

import random
import struct

import pytest

import core._native as _native
from core import mesh as M
from core.scene import Scene
from formats import stl
from tests.test_lazy_mesh_plan import _compare


def _signature(m):
    vid = {id(v): i for i, v in enumerate(m.vertices)}
    return ([v.position.toTuple() for v in m.vertices],
            [(vid[id(e.v0)], vid[id(e.v1)]) for e in m.edges],
            [[vid[id(v)] for v in f.loop] for f in m.faces])


def _write(path, seed, n):
    rng = random.Random(seed)
    out = bytearray(b"\0" * 80) + struct.pack("<I", n)
    for i in range(n):
        x, y = (i % 60) * 0.5, (i // 60) * 0.5
        z = rng.choice([0.0, 0.0, 0.5, rng.uniform(0, 1)])
        pts = [(x, y, z), (x + 0.5, y, z), (x, y + 0.5, z + 1e-5)]
        if rng.random() < 0.5:
            pts = [pts[0], pts[2], pts[1]]
        out += struct.pack("<12fH", 0, 0, 0, *[c for p in pts for c in p], 0)
    path.write_bytes(bytes(out))


@pytest.mark.parametrize("seed,n", [(0, 401), (1, 2500), (2, 9000)])
def test_big_stl_is_lazy_and_the_same(monkeypatch, tmp_path, seed, n):
    path = tmp_path / "m.stl"
    _write(path, seed, n)
    monkeypatch.setattr(_native, "native", _native.available())
    monkeypatch.setattr(_native, "wanted", lambda: True)
    lazy = stl.parse_stl(path)
    assert type(lazy) is M.DeferredMesh
    builds = M.DEFERRED_BUILDS
    scene = Scene()
    stl.add_stl_mesh(scene, path, lazy)
    assert M.DEFERRED_BUILDS == builds, "inserting builds nothing"
    from core import lazy_mesh
    plan = lazy_mesh.plan_of(lazy)
    _compare(plan, lazy)                     # builds it, and compares
    monkeypatch.setattr(_native, "wanted", lambda: False)
    eager = stl.parse_stl(path)
    assert type(eager) is M.Mesh
    assert _signature(lazy) == _signature(eager)


def test_small_stl_stays_eager(monkeypatch, tmp_path):
    path = tmp_path / "s.stl"
    _write(path, 3, 50)
    monkeypatch.setattr(_native, "wanted", lambda: True)
    assert type(stl.parse_stl(path)) is M.Mesh
