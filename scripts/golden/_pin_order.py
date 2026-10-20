# SPDX-License-Identifier: GPL-3.0-or-later
"""Pins the order Python does not repeat, so the REAL ``run_stitch`` and
``orient_outward`` can run inside the import fixtures
(make_mesh_formats_fixtures.py, make_dae_fixtures.py).

It is the pinning of make_stitch_fixtures.py (the Rust stitch takes the same
order): ``core.mesh.set`` iterates mesh objects in creation order,
``core.history._coplanar_component`` hands the merge its faces in creation
order and ``core.mesh._trace_loops`` starts a loop at the first outline edge
not traced yet. Objects are numbered when they are built (their constructors),
which covers the bulk paths (``add_faces_bulk``, ``add_edges_welded``) the
imports use as well as ``add_face``.

``install()`` once, before importing anything that builds meshes.

Deleted with the Python app at M10.
"""
from __future__ import annotations

import itertools

ORD: dict = {}      # id(object) -> creation ordinal
KEEP: list = []     # keeps the numbered objects alive (ids stay unique)


class DetSet(set):
    """A set that iterates mesh objects in creation order."""

    def __iter__(self):
        return iter(sorted(set.__iter__(self),
                           key=lambda o: ORD.get(id(o), 1 << 60)))


def _det_trace_loops(outline):
    adj: dict = {}
    edges: dict = {}
    for u, w in outline:
        if u is w:
            return None
        adj.setdefault(u, []).append(w)
        adj.setdefault(w, []).append(u)
        edges.setdefault(frozenset((u, w)), (u, w))
    for nbrs in adj.values():
        if len(nbrs) != 2:
            return None
    loops = []
    while edges:
        a, b = next(iter(edges.values()))
        edges.pop(frozenset((a, b)), None)
        loop = [a, b]
        prev, cur = a, b
        while True:
            nbrs = adj[cur]
            nxt = nbrs[0] if nbrs[0] is not prev else nbrs[1]
            edges.pop(frozenset((cur, nxt)), None)
            if nxt is loop[0]:
                break
            loop.append(nxt)
            prev, cur = cur, nxt
        loops.append(loop)
    return loops


def install() -> None:
    import core.history as ch
    import core.mesh as cm

    cm.set = DetSet

    def number(cls):
        orig = cls.__init__

        def init(self, *a, **k):
            orig(self, *a, **k)
            ORD[id(self)] = len(KEEP)
            KEEP.append(self)
        cls.__init__ = init

    for cls in (cm.Vertex, cm.Edge, cm.Face):
        number(cls)

    orig_component = ch._coplanar_component
    ch._coplanar_component = lambda *a, **k: DetSet(orig_component(*a, **k))
    cm._trace_loops = _det_trace_loops


def fresh_group_names() -> None:
    """``Group N`` names count from 1 for the next import."""
    import core.group
    core.group._counter = itertools.count(1)
