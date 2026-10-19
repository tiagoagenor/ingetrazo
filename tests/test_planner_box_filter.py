# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
"""The edge planner's box filter (core.edits._Nearby) plans what the full
lists plan: the same commands, in the same order, with the same points —
including its fast path over the planner's own edge list (rows in list
order, followed through add_edge/drop_edge, #541) — and that fast path is
the one a plan takes."""
from __future__ import annotations

import random

import pytest

import core.edits as E
from core.edits import build_add_edges, plan_edge_commands
from core.history import History
from core.scene import Scene
from tests.test_cycle_graph import _sig, _strokes


def _full_lists(monkeypatch):
    """A _Nearby that filters nothing: every search gets the whole list."""
    monkeypatch.setattr(E._Nearby, "edges", lambda self, items, a, b: items)
    monkeypatch.setattr(E._Nearby, "faces", lambda self, items, a, b: items)
    monkeypatch.setattr(E._Nearby, "faces_around",
                        lambda self, items, pts: items)


@pytest.mark.parametrize("seed", range(10))
@pytest.mark.parametrize("flat", [True, False])
def test_box_filter_plans_what_the_full_lists_plan(monkeypatch, seed, flat):
    rng = random.Random(100 + seed)
    scene = Scene()
    History(scene).execute(build_add_edges(scene, _strokes(rng, 30, flat),
                                           detect_faces=True))
    segs = _strokes(rng, 40, flat)
    segs += segs[:4]
    filtered = [_sig(c) for c in plan_edge_commands(scene, segs)]
    _full_lists(monkeypatch)
    full = [_sig(c) for c in plan_edge_commands(scene, segs)]
    assert filtered == full and filtered


def test_the_fast_path_is_taken(monkeypatch):
    rng = random.Random(7)
    scene = Scene()
    History(scene).execute(build_add_edges(scene, _strokes(rng, 30, True),
                                           detect_faces=True))
    calls = {"fast": 0, "walk": 0}
    fast, walk = E._Nearby._pick_live_edges, E._Nearby._pick

    def count_fast(self, pts):
        calls["fast"] += 1
        return fast(self, pts)

    def count_walk(self, items, pts):
        if items is self._sim:
            calls["walk"] += 1
        return walk(self, items, pts)
    monkeypatch.setattr(E._Nearby, "_pick_live_edges", count_fast)
    monkeypatch.setattr(E._Nearby, "_pick", count_walk)
    plan_edge_commands(scene, _strokes(rng, 40, True))
    assert calls["fast"] > 40 and calls["walk"] == 0


def test_an_unhooked_append_falls_back_to_the_walk():
    from core.geometry import Edge
    from core.vecmath import QVector3D as V
    sim = [Edge(V(0, 0, 0), V(1, 0, 0))]
    near = E._Nearby(sim, [])
    sim.append(Edge(V(0, 0.001, 0), V(1, 0.001, 0)))   # no add_edge
    got = near.edges(sim, V(0.5, -1, 0), V(0.5, 1, 0))
    assert got == sim
