# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
"""Which group a pick landed on, read without building a lazy mesh: the
pick index's owner_at is entities[i][1], for plain pairs and lazy ones, and
asking it leaves every lazy mesh deferred (a Select hover used to build
the group under the cursor)."""
from __future__ import annotations

import random

import numpy as np
import pytest

import core._native as _native
from core import mesh as M
from core.scene import Scene
from formats import igz
from tests.test_deferred_mesh import _meshes, _random_scene
from views.array_bake import DeferredFaces, LazyPairs, PickEntities


@pytest.fixture
def lazy_meshes(monkeypatch, tmp_path):
    monkeypatch.setattr(igz, "_LARGE_DOCUMENT", 0)
    monkeypatch.setattr(igz, "_BULK_MIN", 1)
    monkeypatch.setattr(_native, "wanted", lambda: True)
    monkeypatch.setattr(_native, "native", _native.available())
    path = tmp_path / "doc.igz"
    igz.save_scene(_random_scene(random.Random(3)), path)
    scene = Scene()
    igz.load_into(scene, path)
    meshes = [m for m in _meshes(scene) if type(m) is M.DeferredMesh]
    if len(meshes) < 2:
        pytest.skip("this build defers no meshes")
    return meshes


def test_owner_at_is_the_pair_owner_and_builds_nothing(lazy_meshes):
    owners = [object() for _ in lazy_meshes]
    plain = [("face-a", "plain-1"), ("face-b", "plain-2")]
    ents = PickEntities(plain)
    expect = ["plain-1", "plain-2"]
    for m, owner in zip(lazy_meshes, owners):
        n = M.face_count(m)
        ents.extend(LazyPairs(DeferredFaces(m, np.arange(n)), owner))
        expect += [owner] * n
    ents.extend([("face-c", "plain-3")])
    expect.append("plain-3")
    assert len(ents) == len(expect)
    builds = M.DEFERRED_BUILDS
    assert [ents.owner_at(i) for i in range(len(ents))] == expect
    assert ents.owner_at(-1) == "plain-3"
    assert all(type(m) is M.DeferredMesh for m in lazy_meshes)
    assert M.DEFERRED_BUILDS == builds
    # And it is what the pairs themselves say, once built.
    assert [ents[i][1] for i in range(len(ents))] == expect
    assert M.DEFERRED_BUILDS == builds + len(lazy_meshes)
    with pytest.raises(IndexError):
        ents.owner_at(len(ents))
