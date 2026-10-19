# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
"""The one door through which ``core``, ``tools``, ``formats`` and ``georef``
reach Qt's vector math (docs/rust-migration-plan.md, Phase 0).

The names are Qt's own classes, not copies: ``isinstance`` checks and the
plugin API keep working, and every result stays bit for bit what it was.
What this module adds is the boundary to the Rust engine — the kernels take
float32 arrays, never Qt objects — so a module whose hot loop moves to Rust
packs its points here instead of by hand. ``tests/test_vecmath.py`` keeps
the four packages from importing these names from ``PySide6.QtGui``."""
from __future__ import annotations

import numpy as np
from PySide6.QtGui import QMatrix4x4, QQuaternion, QVector2D, QVector3D, QVector4D

__all__ = ["QMatrix4x4", "QQuaternion", "QVector2D", "QVector3D", "QVector4D",
           "points_to_array", "array_to_points"]


def points_to_array(points) -> np.ndarray:
    """``(n, 3)`` float32 array of the points' x, y, z — exact: QVector3D
    holds float32 already. An empty input gives a ``(0, 3)`` array."""
    arr = np.array([(p.x(), p.y(), p.z()) for p in points], dtype=np.float32)
    return arr.reshape(-1, 3)


def array_to_points(arr) -> list[QVector3D]:
    """The rows of an ``(n, 3)`` array as QVector3D, the inverse of
    :func:`points_to_array` (a float64 array is rounded to float32, as
    QVector3D itself would)."""
    return [QVector3D(x, y, z) for x, y, z in np.asarray(arr).reshape(-1, 3).tolist()]
