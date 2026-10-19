# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
"""formats.fuse._collapse_all collapses the very vertices, in the very
order, that rescanning the vertex list from the start after each collapse
did: grids of quads with their edges subdivided (shared subdivisions,
chains of collinear points, near-collinear ones that must stay), and the
whole Simplify Mesh on them."""
from __future__ import annotations

import random

import pytest
from PySide6.QtGui import QVector3D as V

from core.mesh import Mesh
from formats import fuse


def _oracle(mesh):
    order = []
    while True:
        for vertex in list(mesh.vertices):
            if mesh.collapsible_vertex(vertex):
                order.append(vertex.position.toTuple())
                mesh.collapse_vertex(vertex)
                break
        else:
            return order


def _signature(mesh):
    vid = {id(v): i for i, v in enumerate(mesh.vertices)}
    return ([v.position.toTuple() for v in mesh.vertices],
            # Undirected: collapse_vertex orients its new edge by the order of
            # v.edges, a set of objects hashed by address — two identical
            # meshes differ there with the old loop too.
            [tuple(sorted((vid[id(e.v0)], vid[id(e.v1)]))) for e in mesh.edges],
            [[vid[id(v)] for v in f.loop] for f in mesh.faces])


def _mesh(seed):
    rng = random.Random(seed)
    m = Mesh()
    n = rng.randrange(2, 6)
    for i in range(n):
        for j in range(n):
            ring = []
            corners = [(i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)]
            for k in range(4):
                (x0, y0), (x1, y1) = corners[k], corners[(k + 1) % 4]
                ring.append(V(x0, y0, 0))
                for s in range(rng.randrange(0, 4)):          # collinear
                    t = (s + 1) / 5
                    jitter = rng.choice([0.0, 0.0, 0.0, 0.02])  # a bend
                    ring.append(V(x0 + (x1 - x0) * t + jitter * (y1 - y0),
                                  y0 + (y1 - y0) * t, 0))
            try:
                m.add_face(ring)
            except ValueError:
                pass
    return m


@pytest.mark.parametrize("seed", range(20))
def test_same_collapses_in_the_same_order(seed, monkeypatch):
    a, b = _mesh(seed), _mesh(seed)
    want = _oracle(a)
    got = []
    real = Mesh.collapse_vertex

    def spy(self, v):
        got.append(v.position.toTuple())
        return real(self, v)
    monkeypatch.setattr(Mesh, "collapse_vertex", spy)
    fuse._collapse_all(b)
    assert got == want
    assert _signature(b) == _signature(a)


@pytest.mark.parametrize("seed", range(6))
def test_simplify_mesh_is_unchanged(seed, monkeypatch):
    a, b = _mesh(100 + seed), _mesh(100 + seed)
    monkeypatch.setattr(fuse, "_collapse_all", _oracle)
    ra = fuse.simplify_mesh(a)
    monkeypatch.undo()
    rb = fuse.simplify_mesh(b)
    assert ra == rb and _signature(a) == _signature(b)
