# SPDX-License-Identifier: GPL-3.0-or-later
"""The golden recorder (tests/golden): commands recorded from a History
replay on a fresh scene to the same L2 signature, step by step — and a
changed argument does not (docs/rust-rewrite-plan.md, M0.4)."""
from __future__ import annotations

import json

import pytest
from PySide6.QtGui import QVector3D as V

from core.history import (AddEdgeCommand, AddFaceCommand, CompoundCommand,
                          FlipFacesCommand, History, MakeGroupCommand,
                          MoveGroupCommand)
from core.scene import Scene
from tests.golden import recorder
from tests.golden.signature import scene_signature, signature_hash


@pytest.fixture
def recording():
    recorder.take_histories()
    recorder.start()
    yield
    recorder.stop()
    recorder.take_histories()


def _square(x0, y0, s=1.0):
    return [V(x0, y0, 0), V(x0 + s, y0, 0), V(x0 + s, y0 + s, 0), V(x0, y0 + s, 0)]


def _record():
    scene = Scene()
    scene.mesh.add_edge(V(9, 9, 0), V(9, 10, 0))   # setup before the first command
    history = History(scene)
    history.execute(AddFaceCommand(_square(0, 0)))
    history.execute(CompoundCommand([AddFaceCommand(_square(2, 0)),
                                     AddEdgeCommand(V(5, 0, 0), V(6, 1, 0))]))
    face = next(f for f in scene.mesh.faces
                if {(v.position.x(), v.position.y()) for v in f.loop} ==
                {(0, 0), (1, 0), (1, 1), (0, 1)})
    history.execute(FlipFacesCommand([face]))
    history.execute(MakeGroupCommand(
        (f for f in list(scene.mesh.faces) if f is not face),   # a generator
        [], name=None))
    history.execute(MoveGroupCommand(scene.groups[0], V(0, 0, 3)))
    history.undo()
    history.redo()
    return scene, history


def test_recorded_script_replays_to_the_same_signatures(recording):
    scene, history = _record()
    script = recorder.script_of(history)
    assert script is not None and not script["errors"], script and script["errors"]
    ops = [s["op"] for s in script["steps"]]
    assert ops == ["add_face", "compound", "flip_faces", "make_group",
                   "move_group", "undo", "redo"]
    assert all("error" not in s for s in script["steps"]), script["steps"]
    assert script["steps"][-1]["sig"] == signature_hash(scene_signature(scene))
    # The script is plain JSON.
    json.dumps({k: v for k, v in script.items() if k != "start"})
    assert recorder.replay(script) == {"ok": True, "at": None, "why": ""}


def test_a_changed_argument_diverges(recording):
    _scene, history = _record()
    script = recorder.script_of(history)
    move = next(s for s in script["steps"] if s["op"] == "move_group")
    move["args"]["delta"] = {"$v": [0, 0, 2]}
    res = recorder.replay(script)
    assert res["ok"] is False and res["at"] == script["steps"].index(move)


def test_a_change_outside_the_history_is_reported(recording):
    scene = Scene()
    history = History(scene)
    history.execute(AddFaceCommand(_square(0, 0)))
    scene.mesh.add_edge(V(7, 7, 0), V(8, 8, 0))       # not through the history
    history.execute(AddFaceCommand(_square(3, 0)))
    errors = recorder.script_of(history)["errors"]
    assert errors == ["step 1: the scene changed outside the history"]


def test_op_names():
    assert recorder.op_name(AddEdgeCommand) == "add_edge"
    assert recorder.op_name(MakeGroupCommand) == "make_group"
    reg = recorder.registry()
    assert reg["flip_faces"] is FlipFacesCommand
