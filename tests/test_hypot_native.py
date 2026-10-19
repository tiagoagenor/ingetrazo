# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
"""The Rust engine knows the QVector3D::length() of this platform's C++
library: libstdc++ (Linux), MSVC STL (Windows) or libc++ (macOS) — each a
named formula that core._native proves against Qt itself at start-up. On
Linux and Windows one of them must be this Qt's; a formula that matches
reproduces length() AND normalized() bit for bit on hard inputs."""
from __future__ import annotations

import sys

import numpy as np
import pytest
from PySide6.QtGui import QVector3D

import core._native as _native

nat = _native.available()
pytestmark = pytest.mark.skipif(
    nat is None or not hasattr(nat, "qt_vector_probe"),
    reason="native module not built")


def _rows():
    rng = np.random.default_rng(20261018)
    return np.concatenate([
        rng.normal(size=(4000, 3)),
        rng.normal(size=(3000, 3)) * rng.choice(
            [1e-30, 1e-20, 1e-6, 1e-3, 1, 1e3, 1e5, 1e20, 1e30],
            size=(3000, 1)),
        np.round(rng.normal(size=(1500, 3)) * 4) / 2,
        rng.normal(size=(500, 3)) * [1, 0, 0] + [0, 1e-6, 0],
        [[0, 0, 0], [3, 4, 0], [0, 0, -2], [1e-38, 0, 0], [3e38, 3e38, 0],
         [np.inf, 1, 0], [1, -np.inf, 2]],
    ]).astype(np.float32)


def _matches(name, rows):
    ln, nz, _u, _v = nat.qt_vector_probe(rows, name)
    qt_len = np.array([QVector3D(*r).length() for r in rows.tolist()],
                      np.float32)
    qt_nrm = np.array([QVector3D(*r).normalized().toTuple()
                       for r in rows.tolist()], np.float32)
    # Bit for bit; an infinite component normalizes to NaN on both sides.
    return (np.array_equal(ln, qt_len, equal_nan=True)
            and np.array_equal(nz, qt_nrm, equal_nan=True))


def test_known_formulas():
    for name in _native._HYPOTS:
        nat.qt_vector_probe(np.zeros((1, 3), np.float32), name)
    with pytest.raises(ValueError):
        nat.qt_vector_probe(np.zeros((1, 3), np.float32), "nope")


@pytest.mark.skipif(sys.platform == "darwin",
                    reason="libc++'s formula is proven on macOS hosts only")
def test_this_platforms_formula_is_found_and_exact():
    found = _native._probe_hypot(nat)
    assert found is not None, "no known formula is this Qt's length()"
    assert found == {"win32": "msvc"}.get(sys.platform, found)
    with np.errstate(over="ignore", invalid="ignore"):
        assert _matches(found, _rows())


def test_the_formulas_differ():
    """Each name is its own formula: on random rows no two agree on
    everything (so the probe cannot pick a wrong one by accident)."""
    rows = _rows()[:4000]
    out = {n: nat.qt_vector_probe(rows, n)[0].tolist()
           for n in _native._HYPOTS}
    names = list(out)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            assert out[a] != out[b], (a, b)
