# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
"""core.vecmath: the one door to Qt's vector math (rust-migration-plan.md,
Phase 0) — nothing in core/tools/formats/georef goes around it, and its
array helpers are exact both ways."""
import ast
from pathlib import Path

import numpy as np
import pytest
from PySide6 import QtGui

from core import vecmath
from core.vecmath import QVector3D, array_to_points, points_to_array

ROOT = Path(__file__).resolve().parent.parent
PACKAGES = ("core", "tools", "formats", "georef")


def test_names_are_qts_own_classes():
    for name in ("QVector2D", "QVector3D", "QVector4D", "QMatrix4x4",
                 "QQuaternion"):
        assert getattr(vecmath, name) is getattr(QtGui, name)


def test_no_module_imports_qt_math_around_vecmath():
    offenders = []
    for pkg in PACKAGES:
        for path in sorted((ROOT / pkg).rglob("*.py")):
            if path == ROOT / "core" / "vecmath.py":
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if (isinstance(node, ast.ImportFrom)
                        and node.module == "PySide6.QtGui"):
                    bad = {a.name for a in node.names} & set(vecmath.__all__)
                    if bad:
                        rel = path.relative_to(ROOT).as_posix()
                        offenders.append(f"{rel}:{node.lineno} {sorted(bad)}")
    assert not offenders, "import these from core.vecmath:\n" + "\n".join(offenders)


def _samples():
    rng = np.random.default_rng(20261018)
    return np.concatenate([
        rng.normal(size=(500, 3)),
        rng.normal(size=(500, 3)) * rng.choice([1e-6, 1e-3, 1, 1e3, 1e5],
                                               size=(500, 1)),
        np.round(rng.normal(size=(200, 3)) * 4) / 2,      # 0.5 lattice
    ]).astype(np.float32)


def test_points_to_array_is_exact():
    rows = _samples()
    pts = [QVector3D(*r) for r in rows.tolist()]
    arr = points_to_array(pts)
    assert arr.dtype == np.float32 and arr.shape == rows.shape
    assert arr.tobytes() == rows.tobytes()


def test_array_to_points_is_exact():
    rows = _samples()
    pts = array_to_points(rows)
    assert [p.toTuple() for p in pts] == [tuple(r) for r in rows.tolist()]
    assert points_to_array(pts).tobytes() == rows.tobytes()


def test_float64_rounds_like_qvector3d():
    rows = np.array([[0.1, 0.2, 0.3], [1e5 + 1e-3, -2.5, 7.0]])
    assert ([p.toTuple() for p in array_to_points(rows)]
            == [QVector3D(*r).toTuple() for r in rows.tolist()])


@pytest.mark.parametrize("empty", [[], np.zeros((0, 3))])
def test_empty(empty):
    assert points_to_array(empty).shape == (0, 3)
    assert array_to_points(empty) == []
