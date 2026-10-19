# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
"""A lazy group's solid volume (Entity Info on a click) from its plan and
the Rust signed_volume — nothing built — is solid_volume of the built
group, ==, on closed boxes, prisms with holes, random lattice solids at
1e-3…1e4 scale and open or stray-edged meshes; and the Python path, forced,
gives the same."""
from __future__ import annotations

import random

import pytest
from PySide6.QtGui import QVector3D as V

import core._native as _native
from core import mesh as M
from core import solids
from core.group import Group
from core.scene import Scene
from formats import igz

nat = _native.available()
pytestmark = pytest.mark.skipif(
    nat is None or not hasattr(nat, "signed_volume")
    or _native.qt_hypot() is None, reason="native module or hypot unknown")


def _box(m, o, s):
    x, y, z = o
    a, b, c = s
    p = [V(x, y, z), V(x + a, y, z), V(x + a, y + b, z), V(x, y + b, z),
         V(x, y, z + c), V(x + a, y, z + c), V(x + a, y + b, z + c),
         V(x, y + b, z + c)]
    for q in ((0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5),
              (2, 3, 7, 6), (3, 0, 4, 7)):
        m.add_face([p[i] for i in q])


def _holed_prism(m, s, h):
    """A square tube: top and bottom faces with a square hole."""
    out = [V(0, 0, 0), V(s, 0, 0), V(s, s, 0), V(0, s, 0)]
    inn = [V(s / 4, s / 4, 0), V(3 * s / 4, s / 4, 0),
           V(3 * s / 4, 3 * s / 4, 0), V(s / 4, 3 * s / 4, 0)]

    def up(ps):
        return [V(p.x(), p.y(), h) for p in ps]
    m.add_face(list(reversed(out)), [list(reversed(inn))])
    m.add_face(up(out), [up(inn)])
    for ring, sign in ((out, 1), (inn, -1)):
        for i in range(4):
            a, b = ring[i], ring[(i + 1) % 4]
            q = [a, b, up([b])[0], up([a])[0]]
            m.add_face(q if sign > 0 else list(reversed(q)))


def _scene(seed, scales):
    rng = random.Random(seed)
    scene = Scene()
    for k in range(10):
        g = Group(name=f"g{k}")
        kind = k % 5
        scale = rng.choice(scales)
        if kind == 0:
            _box(g.mesh, (rng.uniform(-5, 5), 0.5, 0), (scale, 2 * scale,
                                                        0.5 * scale))
        elif kind == 1:
            _holed_prism(g.mesh, scale, 0.7 * scale)
        elif kind == 2:                       # two boxes, one shell each
            _box(g.mesh, (0, 0, 0), (1, 1, 1))
            _box(g.mesh, (3, 0.5, 0), (0.5, 0.5, 2))
        elif kind == 3:                       # open: a box without its top
            _box(g.mesh, (0, 0, 0), (scale, scale, scale))
            g.mesh.remove_face(g.mesh.faces[1])
        else:                                 # a stray edge on a closed box
            _box(g.mesh, (0, 0, 0), (1, 1, 1))
            g.mesh.add_edge(V(5, 5, 5), V(6, 5, 5))
        scene.groups.append(g)
    return scene


def _load(monkeypatch, tmp_path, seed, scales=(0.37, 1.0, 13.25, 1e4)):
    monkeypatch.setattr(igz, "_LARGE_DOCUMENT", 0)
    monkeypatch.setattr(igz, "_BULK_MIN", 1)
    monkeypatch.setattr(_native, "wanted", lambda: True)
    monkeypatch.setattr(_native, "native", nat)
    path = tmp_path / f"s{seed}.igz"
    igz.save_scene(_scene(seed, scales), path)
    opened = Scene()
    igz.load_into(opened, path)
    return opened


@pytest.mark.parametrize("seed", range(6))
def test_lazy_volume_is_the_built_one(monkeypatch, tmp_path, seed):
    scene = _load(monkeypatch, tmp_path, seed)
    groups = [g for g in scene.groups if type(g.mesh) is M.DeferredMesh]
    assert len(groups) == 10
    # A face at Qt's fuzz of length 0 or 1 sends its group to the Python
    # (a 1 x 0.5 face's Newell length is 0.99999988): only those build.
    refused = sum(solids._lazy_solid_report(g) is None for g in groups)
    assert refused <= 2
    builds = M.DEFERRED_BUILDS
    lazy = [solids.solid_volume(g) for g in groups]
    assert M.DEFERRED_BUILDS == builds + refused
    built = []
    for g in groups:
        g.mesh.faces
        built.append(solids.solid_volume(g))
    assert lazy == built
    kinds = [v is not None for v in lazy]
    assert kinds == [True, True, True, False, False] * 2


def test_tiny_faces_take_the_python_path(monkeypatch, tmp_path):
    """A face whose Newell length is within Qt's fuzz of 0 triangulates
    differently in the viewport's kernel: the volume builds the group then,
    and is the built one."""
    scene = _load(monkeypatch, tmp_path, 9, scales=(1e-3,))
    lazy = [solids.solid_volume(g) for g in scene.groups]
    tiny = [g for g in scene.groups if type(g.mesh) is M.Mesh]
    assert tiny, "1 mm faces fall back"
    for g in scene.groups:
        g.mesh.faces                            # built now
    built = [solids.solid_volume(g) for g in scene.groups]
    assert lazy == built


def test_python_path_when_forced(monkeypatch, tmp_path):
    scene = _load(monkeypatch, tmp_path, 7)
    monkeypatch.setattr(_native, "native", None)
    builds = M.DEFERRED_BUILDS
    vols = [solids.solid_volume(g) for g in scene.groups]
    assert M.DEFERRED_BUILDS == builds + 10       # no engine: built
    assert [v is not None for v in vols] == [True, True, True, False,
                                             False] * 2


def test_a_placed_group_falls_back(monkeypatch, tmp_path):
    from core.vecmath import QMatrix4x4
    scene = _load(monkeypatch, tmp_path, 8)
    g = scene.groups[0]
    g.xform = QMatrix4x4()
    g.xform.translate(1, 2, 3)
    assert solids._lazy_solid_report(g) is None
