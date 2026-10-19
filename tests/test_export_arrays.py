# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
"""Exports from arrays (formats.meshexport.batch_triangles, the Rust
tri_normals) give what the face-by-face path gives: the very triangles of
export_triangles, in order — concave and holed faces, slivers the area
filter drops, tiny faces and faces whose Newell length sits at Qt's fuzz
of 0 or 1 (those take Face.triangulate) — and STL files byte for byte, in
every unit, on random scenes with placed groups and on the city model."""
from __future__ import annotations

import random
from pathlib import Path

import pytest
from PySide6.QtGui import QMatrix4x4, QVector3D as V

import core._native as _native
from core.group import Group
from core.scene import Scene
from formats import meshexport, stl

nat = _native.available()
pytestmark = pytest.mark.skipif(
    nat is None or not hasattr(nat, "tri_normals")
    or _native.qt_hypot() is None, reason="native module or hypot unknown")

ROOT = Path(__file__).resolve().parent.parent


def _add(mesh, pts, holes=None):
    try:
        return mesh.add_face(pts, holes)
    except (ValueError, TypeError):
        return None


def _scene(seed):
    rng = random.Random(seed)
    scene = Scene()
    m = scene.mesh
    for _ in range(rng.randrange(5, 25)):
        x, y, z = (rng.randrange(-8, 8) * 0.5 for _ in range(3))
        w, h = rng.randrange(1, 6) * 0.5, rng.randrange(1, 6) * 0.5
        kind = rng.random()
        if kind < 0.3:                                   # holed
            _add(m, [V(x, y, z), V(x + w, y, z), V(x + w, y + h, z),
                     V(x, y + h, z)],
                 [[V(x + w * .3, y + h * .3, z), V(x + w * .3, y + h * .7, z),
                   V(x + w * .7, y + h * .7, z), V(x + w * .7, y + h * .3, z)]])
        elif kind < 0.5:                                 # concave L
            _add(m, [V(x, y, z), V(x + w, y, z), V(x + w, y + h * .3, z),
                     V(x + w * .3, y + h * .3, z), V(x + w * .3, y + h, z),
                     V(x, y + h, z)])
        elif kind < 0.65:                                # collinear corners
            _add(m, [V(x, y, z), V(x + w / 2, y, z), V(x + w, y, z),
                     V(x + w, y + h, z), V(x, y + h, z)])
        elif kind < 0.8:                                 # tiny (fuzz of 0)
            s = rng.choice([1e-3, 2e-3, 3e-4])
            _add(m, [V(x, y, z), V(x + s, y, z), V(x + s, y + s, z)])
        elif kind < 0.9:                                 # tiny holed / concave
            s = rng.choice([1e-3, 2e-3])
            _add(m, [V(x, y, z), V(x + s, y, z), V(x + s, y + s * .3, z),
                     V(x + s * .3, y + s * .3, z), V(x + s * .3, y + s, z),
                     V(x, y + s, z)])
            _add(m, [V(x + 1, y, z), V(x + 1 + s, y, z),
                     V(x + 1 + s, y + s, z), V(x + 1, y + s, z)],
                 [[V(x + 1 + s * .25, y + s * .25, z),
                   V(x + 1 + s * .25, y + s * .75, z),
                   V(x + 1 + s * .75, y + s * .75, z),
                   V(x + 1 + s * .75, y + s * .25, z)]])
        else:                                            # Newell ~ 1 (a 1 x 0.5)
            _add(m, [V(x, y, z), V(x + 1, y, z), V(x + 1, y + 0.5, z),
                     V(x, y + 0.5, z)])
    for k in range(rng.randrange(1, 3)):
        g = Group(name=f"g{k}")
        for _ in range(rng.randrange(2, 8)):
            x, y = rng.uniform(-3, 3), rng.uniform(-3, 3)
            _add(g.mesh, [V(x, y, 0), V(x + 1, y, 0), V(x + 1, y + 1, 0.3),
                          V(x, y + 1, 0.3)])
        g.xform = QMatrix4x4()
        g.xform.translate(rng.uniform(-9, 9), 0, rng.uniform(0, 4))
        g.xform.rotate(rng.uniform(0, 360), 0, 0, 1)
        scene.groups.append(g)
    return scene


def _python_triangles(faces):
    return [tuple(p.toTuple() for p in t)
            for f in faces for t in meshexport.export_triangles(f)]


@pytest.mark.parametrize("seed", range(15))
def test_batch_triangles_are_export_triangles(monkeypatch, seed):
    monkeypatch.setattr(_native, "native", nat)
    faces = list(stl._faces(_scene(seed)))
    pos, tris = meshexport.batch_triangles(faces)
    got = [tuple(tuple(float(c) for c in pos[k]) for k in t)
           for t in tris.tolist()]
    assert got == _python_triangles(faces)


def _stl_bytes(monkeypatch, scene, tmp_path, use, unit):
    monkeypatch.setattr(_native, "native", nat if use else None)
    out = tmp_path / f"{use}-{unit}.stl"
    stl.save_stl(scene, out, unit)
    return out.read_bytes()


@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("unit", ["m", "mm"])
def test_stl_bytes(monkeypatch, tmp_path, seed, unit):
    scene = _scene(seed)
    a = _stl_bytes(monkeypatch, scene, tmp_path, False, unit)
    b = _stl_bytes(monkeypatch, scene, tmp_path, True, unit)
    assert a == b and len(a) > 84


def test_city_stl_bytes(monkeypatch, tmp_path):
    path = ROOT / "benchmarks" / "models" / "city-M.igz"
    if not path.exists():
        pytest.skip("scripts/bench_models.py M not run")
    from formats import igz
    scene = Scene()
    igz.load_into(scene, path)
    a = _stl_bytes(monkeypatch, scene, tmp_path, False, "m")
    b = _stl_bytes(monkeypatch, scene, tmp_path, True, "m")
    assert a == b


def _obj_bytes(monkeypatch, scene, tmp_path, use):
    from formats import obj
    monkeypatch.setattr(_native, "native", nat if use else None)
    d = tmp_path / ("arrays" if use else "loop")
    d.mkdir(exist_ok=True)
    obj.save_obj(scene, d / "m.obj")
    return (d / "m.obj").read_bytes(), (d / "m.mtl").read_bytes()


def _paint(scene, rng, tmp_path):
    from PySide6.QtGui import QImage
    png = tmp_path / "t.png"
    img = QImage(4, 4, QImage.Format_RGB32)
    img.fill(0x8B5A2B)
    img.save(str(png))
    for f in scene.mesh.faces:
        r = rng.random()
        if r < 0.25:
            f.attrs["texture"] = {"path": str(png), "sw": 0.5, "sh": 0.7,
                                  "rot": 30}
        elif r < 0.4:
            f.attrs["texture"] = {"path": str(png),
                                  "uvw": [1.0, 0.0, 0.0, 0.25, 0.0, 1.0,
                                          0.0, 0.5]}
        elif r < 0.6:
            f.attrs["color"] = [0.2, 0.4, 0.6, 1.0]


@pytest.mark.parametrize("seed", range(8))
def test_obj_bytes(monkeypatch, tmp_path, seed):
    scene = _scene(seed)
    _paint(scene, random.Random(seed), tmp_path)
    a = _obj_bytes(monkeypatch, scene, tmp_path, False)
    b = _obj_bytes(monkeypatch, scene, tmp_path, True)
    assert a == b and len(a[0]) > 40


def test_city_obj_bytes(monkeypatch, tmp_path):
    path = ROOT / "benchmarks" / "models" / "city-M.igz"
    if not path.exists():
        pytest.skip("scripts/bench_models.py M not run")
    from formats import igz
    scene = Scene()
    igz.load_into(scene, path)
    assert _obj_bytes(monkeypatch, scene, tmp_path, False) == \
        _obj_bytes(monkeypatch, scene, tmp_path, True)


def test_obj_vertex_key_tie(monkeypatch, tmp_path):
    """2**-7 is a float32 and 2**-7 × 1e6 = 7812.5: round(x, 6) goes to the
    even 0.007812, so it stays apart from 0.007813 (half-up would weld)."""
    scene = Scene()
    _add(scene.mesh, [V(2 ** -7, 0, 0), V(1, 0, 0), V(1, 1, 0)])
    g = Group(name="apart")              # its own mesh: no weld with the loose one
    _add(g.mesh, [V(0.007813, 0, 0), V(1, 1, 0), V(0, 1, 0)])
    scene.groups.append(g)
    a = _obj_bytes(monkeypatch, scene, tmp_path, False)
    b = _obj_bytes(monkeypatch, scene, tmp_path, True)
    assert a == b


def _file_bytes(monkeypatch, scene, tmp_path, use, ext):
    from formats import dae, gltf
    monkeypatch.setattr(_native, "native", nat if use else None)
    d = tmp_path / f"{ext}-{use}"
    d.mkdir(exist_ok=True)
    out = d / f"m.{ext}"
    (gltf.save_glb if ext == "glb" else dae.save_dae)(scene, out)
    data = out.read_bytes()
    if ext == "dae":                     # the clock, not the geometry
        import re
        data = re.sub(rb"<(created|modified)>[^<]*</\1>", b"", data)
    return data


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("ext", ["glb", "dae"])
def test_glb_and_dae_bytes(monkeypatch, tmp_path, seed, ext):
    scene = _scene(seed)
    _paint(scene, random.Random(seed), tmp_path)
    a = _file_bytes(monkeypatch, scene, tmp_path, False, ext)
    b = _file_bytes(monkeypatch, scene, tmp_path, True, ext)
    assert a == b and len(a) > 100
