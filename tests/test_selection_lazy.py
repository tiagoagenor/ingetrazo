# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
"""Selecting a lazy group measures it from its plan: the vertex positions,
placement points, oriented box and counts are the built mesh's, and none
of them builds it."""
from __future__ import annotations

import random

import pytest

import core._native as _native
from core import mesh as M
from core.group import oriented_bounds, placement_points
from core.scene import Scene
from formats import igz
from tests.test_deferred_mesh import _random_scene


@pytest.mark.parametrize("seed", range(4))
def test_lazy_measures_are_the_built_ones(monkeypatch, tmp_path, seed):
    monkeypatch.setattr(igz, "_LARGE_DOCUMENT", 0)
    monkeypatch.setattr(igz, "_BULK_MIN", 1)
    monkeypatch.setattr(_native, "wanted", lambda: True)
    monkeypatch.setattr(_native, "native", _native.available())
    path = tmp_path / "doc.igz"
    igz.save_scene(_random_scene(random.Random(seed)), path)
    scene = Scene()
    igz.load_into(scene, path)
    groups = [g for g in scene.groups if type(g.mesh) is M.DeferredMesh]
    if not groups:
        pytest.skip("this build defers no meshes")
    builds = M.DEFERRED_BUILDS
    lazy = [(M.vertex_positions(g.mesh).tolist(), M.vertex_count(g.mesh),
             M.face_count(g.mesh), placement_points(g).tolist(),
             oriented_bounds(g.mesh)[1:]) for g in groups]
    assert M.DEFERRED_BUILDS == builds, "measuring builds nothing"
    assert all(type(g.mesh) is M.DeferredMesh for g in groups)
    for g, got in zip(groups, lazy):
        g.mesh.faces                                    # built now
        assert type(g.mesh) is M.Mesh
        want = (M.vertex_positions(g.mesh).tolist(), len(g.mesh.vertices),
                len(g.mesh.faces), placement_points(g).tolist(),
                oriented_bounds(g.mesh)[1:])
        assert got == want
