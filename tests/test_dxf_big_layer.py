# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
"""A DXF layer of more 3DFACEs than the importers' fuse limit is fused in
one pass (formats.fuse) instead of run_stitch's quadratic coplanar merge:
the same surface — the same faces, with the same area per plane — as the
stitch made, on two planes with an opening and a tilted face."""
from __future__ import annotations

import pytest

ezdxf = pytest.importorskip("ezdxf")

from core.scene import Scene  # noqa: E402
from formats import dae, dxf_in  # noqa: E402


def _write(path, n):
    doc = ezdxf.new()
    msp = doc.modelspace()
    for i in range(n):
        for j in range(n):
            if (i, j) == (n // 2, n // 2):
                continue                           # an opening
            msp.add_3dface([(i, j, 0), (i + 1, j, 0), (i + 1, j + 1, 0),
                            (i, j + 1, 0)])
            msp.add_3dface([(i, 0, j + 10), (i + 1, 0, j + 10),
                            (i + 1, 0, j + 11), (i, 0, j + 11)])
    msp.add_3dface([(0, 0, 40), (3, 0, 40), (3, 2, 41), (0, 2, 41)])
    doc.saveas(path)


def _summary(scene):
    meshes = [scene.mesh] + [g.mesh for g in scene.groups]
    faces = [f for m in meshes for f in m.faces]
    by_plane: dict = {}
    for f in faces:
        n = f.normal()
        key = (round(abs(n.x()), 3), round(abs(n.y()), 3), round(abs(n.z()), 3))
        by_plane[key] = by_plane.get(key, 0.0) + f.area()
    return len(faces), {k: round(v, 6) for k, v in sorted(by_plane.items())}


def _load(path):
    fn = getattr(dxf_in, "load_dxf", None) or getattr(dxf_in, "import_dxf")
    scene = Scene()
    fn(scene, path)
    return scene


def test_big_layer_fuses_to_the_stitched_surface(monkeypatch, tmp_path):
    path = tmp_path / "big.dxf"
    _write(path, 15)                               # 2 × 224 + 1 faces
    fused = _summary(_load(path))
    monkeypatch.setattr(dae, "_MAX_FUSE_LOOPS", 10 ** 9)
    stitched = _summary(_load(path))
    assert fused == stitched
