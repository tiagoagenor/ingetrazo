# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
"""The edge planner's live cycle graph (core.topology.CycleGraph, #541)
plans exactly what rebuilding the adjacency per segment planned: the same
commands, in the same order, with the same points — on crossing grids,
diagonals over faces, doubled and welded-shut strokes, in 2D and 3D."""
from __future__ import annotations

import random

import pytest
from PySide6.QtGui import QVector3D as V

import core.topology as T
from core.edits import build_add_edges, plan_edge_commands
from core.history import History
from core.scene import Scene


def _sig(obj, depth=0):
    """A comparable picture of a command: points by value, scene objects
    by identity (both plans run on the same scene)."""
    if isinstance(obj, V):
        return ("V", obj.toTuple())
    if isinstance(obj, (list, tuple)):
        return tuple(_sig(x, depth + 1) for x in obj)
    if isinstance(obj, dict):
        return tuple(sorted((k, _sig(v, depth + 1)) for k, v in obj.items()))
    if isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    if depth == 0 or type(obj).__module__.startswith("core.geometry"):
        fields = getattr(obj, "__dict__", None)
        if fields is None:
            fields = {s: getattr(obj, s, None)
                      for s in getattr(type(obj), "__slots__", ())}
        return (type(obj).__name__,
                tuple((k, _sig(v, depth + 1)) for k, v in sorted(fields.items())
                      if not k.startswith("_")))
    return ("id", id(obj))


def _strokes(rng, n, flat):
    def p():
        z = 0.0 if flat else rng.randrange(0, 3) * 0.5
        return V(rng.randrange(0, 7) * 0.5, rng.randrange(0, 7) * 0.5, z)
    out = []
    for _ in range(n):
        a = p()
        b = p() if rng.random() < 0.6 else V(a.x(), a.y() + 1.5, a.z())
        if (a - b).length() > 1e-6:
            out.append((a, b))
    return out


def _plan_both(monkeypatch, scene, segs):
    live = plan_edge_commands(scene, segs)
    monkeypatch.setattr(T, "CycleGraph", lambda *a, **k: None)
    rebuilt = plan_edge_commands(scene, segs)
    monkeypatch.undo()
    return [_sig(c) for c in live], [_sig(c) for c in rebuilt]


@pytest.mark.parametrize("seed", range(12))
@pytest.mark.parametrize("flat", [True, False])
def test_live_graph_plans_what_the_rebuilt_one_did(monkeypatch, seed, flat):
    rng = random.Random(seed)
    scene = Scene()
    History(scene).execute(build_add_edges(scene, _strokes(rng, 25, flat),
                                           detect_faces=True))
    segs = _strokes(rng, 30, flat)
    segs += segs[:3]                                  # doubled strokes
    live, rebuilt = _plan_both(monkeypatch, scene, segs)
    assert live == rebuilt
    assert live, "the plan does something"


def test_graph_keeps_edge_order_through_removals():
    from core.geometry import Edge
    e = [Edge(V(0, 0, 0), V(1, 0, 0)), Edge(V(0, 0, 0), V(0, 1, 0)),
         Edge(V(0, 0, 0), V(0, 0, 1)), Edge(V(1, 0, 0), V(0, 0, 0))]
    g = T.CycleGraph(e)
    k0 = T._key(V(0, 0, 0))
    g.remove(e[1])
    g.add(Edge(V(0, 0, 0), V(2, 0, 0)))
    want = T._cycle_adjacency([e[0], e[2], e[3],
                               Edge(V(0, 0, 0), V(2, 0, 0))], None, None)
    assert [k for k, _ in g.adj[k0]] == [k for k, _ in want[k0]]
    g.remove(e[1])                                    # twice: a no-op
    assert len(g.adj[k0]) == 4


def test_without_view_is_the_rebuilt_adjacency():
    from core.geometry import Edge
    rng = random.Random(4)
    pts = [V(rng.randrange(4) * 0.5, rng.randrange(4) * 0.5, 0)
           for _ in range(40)]
    edges = [Edge(pts[i], pts[i + 1]) for i in range(39)]
    g = T.CycleGraph(edges)
    for edge in edges[:15]:
        ka, kb = T._key(edge.a), T._key(edge.b)
        want = T._cycle_adjacency(edges, ka, kb)
        view = T._AdjWithout(g.adj, ka, kb)
        nodes = set(want) | set(g.adj)
        for u in nodes:
            assert (u in view) == (u in want)
            assert [k for k, _ in view.get(u, ())] == \
                [k for k, _ in want.get(u, ())]
